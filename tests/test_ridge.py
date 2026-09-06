"""The ridge regressor for the mirror-difference arm (Phase 4 §3.1).

Torch-free and therefore fully exercised here, which is the point: the whole arm
-- features and regressor -- is verified on the laptop before any cluster time is
spent on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.train.harness import TrainConfig, run_cv
from cleft.train.ridge import DEFAULT_ALPHA, RidgeBackbone, RidgeError


def linear_problem(n=60, d=6, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, d))
    weights = rng.normal(size=d)
    labels = 3.0 + features @ weights + rng.normal(0.0, noise, size=n)
    return features, labels


# --------------------------------------------------------------------------
# the harness protocol
# --------------------------------------------------------------------------


def test_an_unfitted_ridge_predicts_the_training_mean():
    """Gate 3, for this arm. An untrained model predicts the mean."""
    model = RidgeBackbone()
    labels = np.array([1.0, 2.0, 3.0, 4.0])
    model.reset(labels)
    assert model.predict(np.zeros((3, 5))) == pytest.approx(2.5)


def test_it_recovers_a_linear_signal():
    features, labels = linear_problem()
    model = RidgeBackbone()
    model.reset(labels)
    model.train_epoch(features, labels)

    predictions = model.predict(features)
    assert np.corrcoef(predictions, labels)[0, 1] > 0.95


def test_fitting_is_idempotent():
    """Closed form: a second epoch changes nothing, so early stopping selects
    epoch 1. That is honest for this solver, not a defect in the policy."""
    features, labels = linear_problem()
    model = RidgeBackbone()
    model.reset(labels)

    model.train_epoch(features, labels)
    first = model.predict(features)
    model.train_epoch(features, labels)
    assert np.allclose(first, model.predict(features))


def test_a_negative_alpha_is_refused():
    with pytest.raises(RidgeError, match="non-negative"):
        RidgeBackbone(alpha=-1.0)


def test_more_regularisation_shrinks_the_weights():
    features, labels = linear_problem()
    weak, strong = RidgeBackbone(alpha=0.01), RidgeBackbone(alpha=1000.0)
    for model in (weak, strong):
        model.reset(labels)
        model.train_epoch(features, labels)
    assert np.abs(strong._weights).sum() < np.abs(weak._weights).sum()


def test_the_intercept_is_not_penalised():
    """Shrinking the intercept would pull predictions toward zero rather than
    toward the mean, which on a 1-5 scale is a large artificial bias."""
    features, labels = linear_problem()
    model = RidgeBackbone(alpha=1e9)
    model.reset(labels)
    model.train_epoch(features, labels)
    assert model.predict(features) == pytest.approx(labels.mean(), abs=0.05)


def test_a_constant_feature_does_not_blow_up():
    features, labels = linear_problem()
    features[:, 0] = 7.0
    model = RidgeBackbone()
    model.reset(labels)
    model.train_epoch(features, labels)
    assert np.all(np.isfinite(model.predict(features)))


def test_it_reports_its_parameter_count():
    features, labels = linear_problem(d=22)
    model = RidgeBackbone()
    model.reset(labels)
    model.train_epoch(features, labels)
    assert model.parameter_report["trainable_parameters"] == 23  # 22 weights + bias


# --------------------------------------------------------------------------
# what this arm's seed variance actually measures
# --------------------------------------------------------------------------


def cv_at(seed: int) -> np.ndarray:
    features, labels = linear_problem(n=60)
    ids = list(range(1, 61))
    assignments = {pid: (pid - 1) % 5 for pid in ids}
    result = run_cv(
        features=features,
        labels=labels,
        patient_ids=ids,
        assignments=assignments,
        make_backbone=lambda: RidgeBackbone(seed=seed),
        config=TrainConfig(seed=seed, max_epochs=3, patience=2),
    )
    return result.oof_predictions


def test_the_solver_itself_is_deterministic():
    """Same data, same alpha, same answer -- twice, byte for byte."""
    features, labels = linear_problem()
    runs = []
    for _ in range(2):
        model = RidgeBackbone()
        model.reset(labels)
        model.train_epoch(features, labels)
        runs.append(model.predict(features))
    assert np.array_equal(runs[0], runs[1])


def test_the_seed_still_moves_the_result_and_here_is_why():
    """**Not zero variance**, and the reason matters for how the band is read.

    A closed-form solve has no initialisation and no early stopping, so the
    natural expectation is that the seed changes nothing. But
    ``harness.inner_val_split`` seeds off ``config.seed``, so the seed decides
    which patients are held OUT of the training fold. The fit moves with the
    seed even though the solver does not.

    So this arm's SD is real, and it measures **split sensitivity** -- a
    different quantity from the initialisation sensitivity the Phase 3 probe's
    band measures. PLAN §4.12.1: every arm reports its own.
    """
    assert not np.allclose(cv_at(1337), cv_at(7))


def test_the_same_seed_reproduces_exactly():
    assert np.array_equal(cv_at(1337), cv_at(1337))


def test_the_default_alpha_is_fixed_and_recorded():
    """Not selected against anything. If the arm needs tuning to beat the probe,
    it did not beat the probe."""
    assert DEFAULT_ALPHA == 1.0
    assert RidgeBackbone().alpha == DEFAULT_ALPHA
