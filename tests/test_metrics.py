"""Metrics, checked against reference implementations on synthetic data.

The metrics are implemented in numpy rather than delegating to scipy/sklearn so
that these comparisons are genuine independent checks. Rule R6: a verifier
passing means only that the thing it tests is right, and a number guaranteed by
construction proves nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from cleft.eval import metrics

from fixtures import builders

TOL = 1e-10


@pytest.fixture
def pair():
    return builders.graded_pair(n=120, seed=7)


# --------------------------------------------------------------------------
# agreement with reference implementations
# --------------------------------------------------------------------------


def test_pcc_matches_scipy(pair):
    truth, pred = pair
    assert metrics.pcc(truth, pred) == pytest.approx(
        stats.pearsonr(truth, pred).statistic, abs=TOL
    )


def test_pcc_matches_scipy_on_many_seeds():
    for seed in range(20):
        truth, pred = builders.graded_pair(n=60, seed=seed)
        assert metrics.pcc(truth, pred) == pytest.approx(
            stats.pearsonr(truth, pred).statistic, abs=TOL
        )


def test_spearman_matches_scipy(pair):
    truth, pred = pair
    assert metrics.spearman(truth, pred) == pytest.approx(
        stats.spearmanr(truth, pred).statistic, abs=TOL
    )


def test_spearman_handles_ties_like_scipy():
    """Grades are coarse, so ties are the normal case, not an edge case."""
    rng = np.random.default_rng(3)
    truth = rng.integers(1, 6, size=200).astype(float)
    pred = rng.integers(1, 6, size=200).astype(float)
    assert metrics.spearman(truth, pred) == pytest.approx(
        stats.spearmanr(truth, pred).statistic, abs=TOL
    )


def test_qwk_3cat_matches_sklearn(pair):
    truth, pred = pair
    expected = cohen_kappa_score(
        metrics.to_3class(truth), metrics.to_3class(pred), weights="quadratic"
    )
    assert metrics.qwk_3cat(truth, pred) == pytest.approx(expected, abs=TOL)


def test_qwk_3cat_matches_sklearn_on_many_seeds():
    for seed in range(20):
        truth, pred = builders.graded_pair(n=80, seed=100 + seed)
        expected = cohen_kappa_score(
            metrics.to_3class(truth), metrics.to_3class(pred), weights="quadratic"
        )
        assert metrics.qwk_3cat(truth, pred) == pytest.approx(expected, abs=TOL)


def test_mae_and_rmse_match_closed_form(pair):
    truth, pred = pair
    assert metrics.mae(truth, pred) == pytest.approx(np.abs(truth - pred).mean(), abs=TOL)
    assert metrics.rmse(truth, pred) == pytest.approx(
        np.sqrt(((truth - pred) ** 2).mean()), abs=TOL
    )


# --------------------------------------------------------------------------
# the 3-class collapse
# --------------------------------------------------------------------------


def test_3class_thresholds_are_fixed_a_priori():
    """2.5 and 3.5, set in advance, never tuned. Part 4.2."""
    assert metrics.CLASS_THRESHOLDS == (2.5, 3.5)


def test_3class_collapse_boundaries():
    values = np.array([1.0, 2.0, 2.49, 2.5, 3.0, 3.49, 3.5, 4.0, 5.0])
    # {1,2} -> 0, {3} -> 1, {4,5} -> 2; a value exactly on a threshold goes up.
    assert metrics.to_3class(values).tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_3class_collapse_rejects_out_of_range_input():
    with pytest.raises(ValueError, match="range"):
        metrics.to_3class(np.array([0.5, 3.0]))


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        metrics.pcc(np.arange(5.0), np.arange(6.0))


def test_nan_raises_rather_than_propagating():
    """A silent nan metric once looked like a model that had not learned."""
    truth = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="nan"):
        metrics.pcc(truth, np.array([1.0, np.nan, 3.0]))


def test_constant_prediction_gives_nan_pcc_explicitly():
    """PCC is undefined against a constant. Say so, do not return 0."""
    truth = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(metrics.pcc(truth, np.full(4, 2.0)))


# --------------------------------------------------------------------------
# BCa bootstrap
# --------------------------------------------------------------------------


def test_bca_is_reproducible_under_a_fixed_seed(pair):
    truth, pred = pair
    a = metrics.bca_ci(metrics.pcc, truth, pred, n_boot=500, seed=1337)
    b = metrics.bca_ci(metrics.pcc, truth, pred, n_boot=500, seed=1337)
    assert a == b


def test_bca_changes_with_the_seed(pair):
    truth, pred = pair
    a = metrics.bca_ci(metrics.pcc, truth, pred, n_boot=500, seed=1337)
    b = metrics.bca_ci(metrics.pcc, truth, pred, n_boot=500, seed=2024)
    assert a != b


def test_bca_brackets_the_point_estimate(pair):
    truth, pred = pair
    point = metrics.pcc(truth, pred)
    lo, hi = metrics.bca_ci(metrics.pcc, truth, pred, n_boot=2000, seed=1337)
    assert lo < point < hi


def test_bca_resamples_pairs_together(pair):
    """Resampling truth and prediction independently would destroy the pairing."""
    truth, pred = pair
    lo, hi = metrics.bca_ci(metrics.pcc, truth, pred, n_boot=2000, seed=1337)
    assert lo > 0.5, "a strong correlation must not bootstrap down to nothing"


def test_paired_delta_of_a_variable_with_itself_contains_zero(pair):
    truth, pred = pair
    delta, lo, hi = metrics.paired_delta_bca(
        metrics.pcc, truth, pred, pred, n_boot=500, seed=1337
    )
    assert delta == pytest.approx(0.0, abs=TOL)
    assert lo <= 0.0 <= hi


def test_paired_delta_detects_a_real_difference(pair):
    """A prediction with extra noise must be measurably worse."""
    truth, good = pair
    rng = np.random.default_rng(11)
    bad = np.clip(good + rng.normal(0.0, 1.5, size=good.size), 1.0, 5.0)
    delta, lo, hi = metrics.paired_delta_bca(
        metrics.pcc, truth, good, bad, n_boot=2000, seed=1337
    )
    assert delta > 0
    assert lo > 0, "CI must exclude 0 for a difference this large"


def test_paired_delta_of_two_equally_good_predictions_does_not_claim_a_win():
    """Two arms perturbed by the same amount in different directions tie."""
    truth, base = builders.graded_pair(n=200, seed=5)
    rng = np.random.default_rng(23)
    a = np.clip(base + rng.normal(0.0, 0.3, size=base.size), 1.0, 5.0)
    b = np.clip(base + rng.normal(0.0, 0.3, size=base.size), 1.0, 5.0)
    delta, lo, hi = metrics.paired_delta_bca(
        metrics.pcc, truth, a, b, n_boot=2000, seed=1337
    )
    assert lo <= 0.0 <= hi, f"claimed a win of {delta} with CI ({lo}, {hi})"


def test_bca_default_n_boot_is_ten_thousand():
    """Part 4.3 fixes the resample count; it is not a knob to tune per table."""
    import inspect

    assert inspect.signature(metrics.bca_ci).parameters["n_boot"].default == 10000


def test_bca_rejects_a_sample_too_small_to_jackknife():
    with pytest.raises(ValueError, match="too small|n="):
        metrics.bca_ci(metrics.pcc, np.array([1.0, 2.0]), np.array([1.0, 2.0]), n_boot=50)
