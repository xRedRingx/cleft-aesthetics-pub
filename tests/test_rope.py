"""The Bayesian correlated t-test arithmetic (Phase 23's ROPE half).

The parts that can be checked without any run directory -- per-fold PCC,
the paired differences, the posterior, the three probabilities and the
verdict -- are module-level functions and are tested here directly.

**Every parameter is either the source's or ruled**; none is chosen in
this module. See ``phase23.EXIT_CRITERIA["the_settings_as_ruled"]``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cleft import phase23, rope


# --------------------------------------------------------------------------
# per-fold PCC and the paired differences
# --------------------------------------------------------------------------


def _rows(truth, pred, folds):
    return [
        {"patient_id": i + 1, "truth": t, "prediction": p, "fold": f}
        for i, (t, p, f) in enumerate(zip(truth, pred, folds))
    ]


def test_per_fold_pcc_matches_a_direct_computation():
    rng = np.random.default_rng(3)
    truth = rng.uniform(1, 5, 40)
    pred = truth + rng.normal(0, 0.7, 40)
    folds = [i % 5 for i in range(40)]

    got = rope.per_fold_pcc(_rows(truth, pred, folds))
    assert sorted(got) == [0, 1, 2, 3, 4]
    for fold in range(5):
        mask = np.array(folds) == fold
        expected = np.corrcoef(truth[mask], pred[mask])[0, 1]
        assert got[fold] == pytest.approx(expected)


def test_a_degenerate_fold_is_reported_not_silently_dropped():
    """A fold whose predictions are constant has an undefined PCC. It
    must surface, because phase23 named degenerate folds as a live risk
    in advance."""
    truth = [1.0, 2.0, 3.0, 4.0]
    pred = [2.5, 2.5, 3.0, 4.0]          # fold 0 constant
    folds = [0, 0, 1, 1]
    got = rope.per_fold_pcc(_rows(truth, pred, folds))
    assert math.isnan(got[0]), "a constant fold must be NaN, not 0.0"
    assert not math.isnan(got[1])


def test_the_paired_differences_are_fold_by_fold_within_seed():
    a = {1337: {0: 0.30, 1: 0.20}, 2024: {0: 0.10, 1: 0.40}}
    b = {1337: {0: 0.25, 1: 0.25}, 2024: {0: 0.15, 1: 0.35}}
    x, n, dropped = rope.paired_fold_differences(a, b)
    assert n == 4
    assert dropped == 0
    assert sorted(np.round(x, 4).tolist()) == [-0.05, -0.05, 0.05, 0.05]


def test_only_shared_seeds_and_folds_are_paired():
    """A ten-seed arm against a five-seed one reduces to the shared
    five -- the qualification phase23 already registered."""
    a = {s: {f: 0.2 for f in range(5)} for s in (1337, 2024, 7, 99, 12345,
                                                 42, 271828)}
    b = {s: {f: 0.1 for f in range(5)} for s in (1337, 2024, 7, 99, 12345)}
    x, n, dropped = rope.paired_fold_differences(a, b)
    assert n == 25, "five shared seeds x five folds"
    assert np.allclose(x, 0.1)


def test_a_nan_fold_is_dropped_from_the_pairing_and_counted():
    a = {1337: {0: 0.3, 1: float("nan")}}
    b = {1337: {0: 0.2, 1: 0.2}}
    x, n, dropped = rope.paired_fold_differences(a, b)
    assert n == 1 and dropped == 1
    assert x.tolist() == pytest.approx([0.1])


# --------------------------------------------------------------------------
# the posterior -- the source's formula, checked against its own algebra
# --------------------------------------------------------------------------


def test_the_posterior_scale_is_the_papers_formula():
    """St(mu; n-1, xbar, (1/n + rho/(1-rho)) sigmahat^2)."""
    x = np.array([0.01, 0.02, -0.01, 0.03, 0.00])
    df, loc, scale = rope.posterior(x, rho=0.2)
    n = len(x)
    assert df == n - 1
    assert loc == pytest.approx(float(np.mean(x)))
    variance = float(np.var(x, ddof=1))
    expected = math.sqrt((1 / n + 0.2 / (1 - 0.2)) * variance)
    assert scale == pytest.approx(expected)
    # rho/(1-rho) at the ruled rho is 0.25.
    assert 0.2 / (1 - 0.2) == 0.25


def test_the_three_probabilities_sum_to_one_and_bracket_the_rope():
    x = np.array([0.005, 0.010, -0.002, 0.008, 0.001] * 5)
    probs = rope.probabilities(x, rho=0.2, half_width=0.0183)
    assert set(probs) == {"left", "rope", "right"}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert all(0.0 <= v <= 1.0 for v in probs.values())
    # These observations sit inside the rope, so P(rope) dominates.
    assert probs["rope"] > probs["left"] and probs["rope"] > probs["right"]


def test_a_large_positive_effect_puts_the_mass_on_the_right():
    x = np.full(25, 0.12) + np.linspace(-0.001, 0.001, 25)
    probs = rope.probabilities(x, rho=0.2, half_width=0.0183)
    assert probs["right"] > 0.95
    assert probs["rope"] < 0.01


def test_a_wide_posterior_decides_nothing_even_when_centred_in_the_rope():
    """The distinction the paper draws explicitly: a wide posterior is
    UNCERTAINTY, not similarity."""
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 0.20, 25)          # centred, but very wide
    probs = rope.probabilities(x, rho=0.2, half_width=0.0183)
    assert probs["rope"] < 0.95
    assert rope.verdict(probs, threshold=0.95) == "no decision"


def test_the_probabilities_are_an_integral_of_the_same_posterior():
    x = np.array([0.004, -0.001, 0.009, 0.002, 0.000] * 5)
    df, loc, scale = rope.posterior(x, rho=0.2)
    from scipy.stats import t as student

    w = 0.0183
    probs = rope.probabilities(x, rho=0.2, half_width=w)
    assert probs["left"] == pytest.approx(student.cdf(-w, df, loc, scale))
    assert probs["right"] == pytest.approx(student.sf(w, df, loc, scale))
    assert probs["rope"] == pytest.approx(
        student.cdf(w, df, loc, scale) - student.cdf(-w, df, loc, scale)
    )


# --------------------------------------------------------------------------
# the verdict -- criterion 6 as a tested literal
# --------------------------------------------------------------------------


def test_the_three_verdicts_at_the_ruled_threshold():
    assert rope.verdict(
        {"left": 0.01, "rope": 0.97, "right": 0.02}, threshold=0.95
    ) == "practically equivalent"
    assert rope.verdict(
        {"left": 0.96, "rope": 0.03, "right": 0.01}, threshold=0.95
    ) == "practically different"
    assert rope.verdict(
        {"left": 0.02, "rope": 0.01, "right": 0.97}, threshold=0.95
    ) == "practically different"
    assert rope.verdict(
        {"left": 0.30, "rope": 0.50, "right": 0.20}, threshold=0.95
    ) == "no decision"


def test_no_decision_is_never_equivalence_at_the_boundary():
    """Criterion 6, as a literal: 0.95 exactly does not decide, and
    nothing below it may be called equivalent."""
    assert rope.verdict(
        {"left": 0.0, "rope": 0.95, "right": 0.05}, threshold=0.95
    ) == "no decision"
    assert rope.verdict(
        {"left": 0.0, "rope": 0.9499, "right": 0.0501}, threshold=0.95
    ) == "no decision"
    # The strict inequality is the paper's: P(.) > 0.95.
    assert "P(.) > 0.95" in phase23.THE_DECISION_THRESHOLD_RULED["the_ruling"]
    assert rope.VERDICTS == (
        "practically equivalent", "practically different", "no decision",
    )
    assert "equivalent" not in "no decision"


# --------------------------------------------------------------------------
# the description that gates nothing
# --------------------------------------------------------------------------


def test_the_description_reports_shape_and_gates_nothing():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held
    ``describe(x, dropped_folds=2)`` -- an integer count, which is all
    the signature could carry. The count reached metrics.json and the
    CAUSE never did, so three contrasts ran at n=24 with no reason
    beside them. The signature now takes the records."""
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.05])
    described = rope.describe(x, dropped=[_drop(1, 3), _drop(2, 4)],
                              unpaired=[])
    assert described["n"] == 5
    assert described["mean"] == pytest.approx(float(np.mean(x)))
    assert described["sd"] == pytest.approx(float(np.std(x, ddof=1)))
    assert described["min"] == 0.0 - 0.02
    assert described["max"] == 0.05
    assert described["degenerate_folds_dropped"] == 2
    # It carries no verdict, threshold or decision of any kind.
    for key in described:
        assert "verdict" not in key and "threshold" not in key


def test_the_width_is_recomputed_not_read_from_a_constant():
    """Criterion 2's protection: the task recomputes and refuses on
    mismatch, the shape Phase 20's Arm C used."""
    from cleft.train.phase3 import combined_claimable_delta

    # [UPDATED 2026-09-02] Asserted EXACTLY, not to four places. The
    # rounded 0.0183 was a display quoted as the quantity and the guard
    # refused a launch over it (phase23.THE_WIDTH_WAS_A_ROUNDED_DISPLAY).
    assert rope.rope_half_width(sd=0.0148, n_seeds=5) == (
        combined_claimable_delta(0.0148, 5, 0.0148, 5)["arm_means_95"]
    )
    assert rope.rope_half_width(sd=0.0148, n_seeds=5) == 0.018346
    assert round(0.018346, 4) == 0.0183, "the rounded form, for the record"
    # A different sd gives a different width -- so the function is not
    # returning a constant.
    assert rope.rope_half_width(sd=0.03, n_seeds=5) > 0.03


def test_the_module_chooses_nothing():
    """Every parameter is passed in; none is defaulted to a value this
    project would be choosing."""
    import inspect

    for name in ("posterior", "probabilities", "verdict"):
        signature = inspect.signature(getattr(rope, name))
        for parameter in signature.parameters.values():
            if parameter.name in ("x", "probs"):
                continue
            assert parameter.default is inspect.Parameter.empty, (
                f"rope.{name}({parameter.name}=...) has a default; the "
                "caller must supply every ruled parameter"
            )


# --------------------------------------------------------------------------
# [2026-09-02] The drop was counted, never recorded
# --------------------------------------------------------------------------


def _drop(seed, fold, arm="arm", reason="constant prediction within the fold"):
    return {
        "seed": seed, "fold": fold, "arms": [arm],
        "causes": [{
            "arm": arm, "reason": reason, "n_patients": 47,
            "truth_sd": 1.5, "prediction_sd": 0.0,
        }],
    }


def _detail_rows(truth, pred, folds):
    return [
        {"patient_id": i, "fold": f, "truth": t, "prediction": p}
        for i, (t, p, f) in enumerate(zip(truth, pred, folds))
    ]


def test_per_fold_detail_names_the_cause_of_every_degeneracy():
    """Three causes, each named as itself rather than as 'undefined'."""
    constant_prediction = rope.per_fold_detail(
        _detail_rows([1.0, 2.0, 3.0], [4.0, 4.0, 4.0], [0, 0, 0])
    )[0]
    assert constant_prediction["undefined_because"] == (
        "constant prediction within the fold"
    )
    assert constant_prediction["prediction_sd"] == 0.0
    assert constant_prediction["truth_sd"] > 0.0
    assert math.isnan(constant_prediction["pcc"])

    constant_truth = rope.per_fold_detail(
        _detail_rows([4.0, 4.0, 4.0], [1.0, 2.0, 3.0], [0, 0, 0])
    )[0]
    assert constant_truth["undefined_because"] == (
        "constant truth within the fold"
    )

    both = rope.per_fold_detail(
        _detail_rows([4.0, 4.0], [1.0, 1.0], [0, 0])
    )[0]
    assert both["undefined_because"] == (
        "constant truth AND constant prediction within the fold"
    )

    thin = rope.per_fold_detail(_detail_rows([4.0], [1.0], [0]))[0]
    assert thin["undefined_because"] == "fewer than two patients in the fold"
    assert thin["n_patients"] == 1

    # A defined fold says so with None, not with a string.
    fine = rope.per_fold_detail(
        _detail_rows([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [0, 0, 0])
    )[0]
    assert fine["undefined_because"] is None
    assert not math.isnan(fine["pcc"])


def test_the_two_forms_cannot_disagree_about_which_folds_are_degenerate():
    """``per_fold_pcc`` reads ``per_fold_detail``; one rule, not two."""
    rows = _detail_rows(
        [1.0, 2.0, 3.0, 5.0, 5.0, 5.0], [1.0, 3.0, 2.0, 1.0, 2.0, 3.0],
        [0, 0, 0, 1, 1, 1],
    )
    detail = rope.per_fold_detail(rows)
    plain = rope.per_fold_pcc(rows)
    assert set(plain) == set(detail)
    for fold in plain:
        if math.isnan(plain[fold]):
            assert math.isnan(detail[fold]["pcc"])
            assert detail[fold]["undefined_because"] is not None
        else:
            assert plain[fold] == detail[fold]["pcc"]
            assert detail[fold]["undefined_because"] is None


def test_a_dropped_fold_is_reported_with_arm_seed_fold_and_cause():
    """The four things the record lacked, all four present."""
    good = _detail_rows([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [0, 0, 0])
    dead = _detail_rows([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], [0, 0, 0])
    detail_a = {7: rope.per_fold_detail(good)}
    detail_b = {7: rope.per_fold_detail(dead)}
    arm_a = {7: {f: d["pcc"] for f, d in detail_a[7].items()}}
    arm_b = {7: {f: d["pcc"] for f, d in detail_b[7].items()}}

    report = rope.paired_fold_report(
        "arm_C", arm_a, detail_a, "probe", arm_b, detail_b
    )
    assert report["n"] == 0 and len(report["dropped"]) == 1
    drop = report["dropped"][0]
    assert drop["seed"] == 7 and drop["fold"] == 0
    assert drop["arms"] == ["probe"]          # which arm, and only that one
    cause = drop["causes"][0]
    assert cause["arm"] == "probe"
    assert cause["reason"] == "constant prediction within the fold"
    assert cause["prediction_sd"] == 0.0 and cause["n_patients"] == 3
    # Nothing in the record is null.
    assert all(v is not None for v in cause.values())


def test_an_unpaired_fold_is_not_reported_as_a_degeneracy():
    """Both shorten n; only one is a degeneracy, and the old record
    could not tell them apart because the intersection erased one."""
    detail_a = {1: rope.per_fold_detail(
        _detail_rows([1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
                     [1.0, 2.0, 4.0, 3.0, 1.0, 2.0], [0, 0, 0, 1, 1, 1])
    )}
    detail_b = {1: rope.per_fold_detail(
        _detail_rows([1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [0, 0, 0])
    )}
    arm_a = {1: {f: d["pcc"] for f, d in detail_a[1].items()}}
    arm_b = {1: {f: d["pcc"] for f, d in detail_b[1].items()}}

    report = rope.paired_fold_report("a", arm_a, detail_a, "b", arm_b, detail_b)
    assert report["n"] == 1
    assert report["dropped"] == []
    assert report["unpaired"] == [
        {"seed": 1, "fold": 1, "present_in": "a", "absent_from": "b"}
    ]
    # And the counting form agrees on n while seeing no degeneracy.
    x, n, dropped = rope.paired_fold_differences(arm_a, arm_b)
    assert n == 1 and dropped == 0
    assert list(x) == list(report["x"])


def test_a_degenerate_cell_without_a_cause_raises_rather_than_writing_null():
    arm = {1: {0: float("nan")}}
    with pytest.raises(ValueError, match="no per-fold detail"):
        rope.paired_fold_report("a", arm, {}, "b", {1: {0: 0.5}}, {1: {0: {}}})


def test_describe_carries_the_records_and_both_counts():
    x = np.array([0.01, -0.02, 0.03])
    described = rope.describe(
        x,
        dropped=[_drop(1, 2, arm="arm_C")],
        unpaired=[{"seed": 3, "fold": 4, "present_in": "a",
                   "absent_from": "b"}],
    )
    assert described["n_dropped"] == 1
    assert described["n_unpaired"] == 1
    # The shipped name is kept and equals the new one by construction,
    # so the two runs' metrics.json compare directly.
    assert described["degenerate_folds_dropped"] == described["n_dropped"]
    assert described["dropped_folds"][0]["causes"][0]["arm"] == "arm_C"
    assert described["unpaired_folds"][0]["absent_from"] == "b"
    # No key in the description is null or an unexplained blank.
    for key, value in described.items():
        assert value is not None, key


def test_describe_requires_both_accounts_with_no_default():
    """The caller that wrote nulls had nothing to pass. A default would
    let the next one do it silently again."""
    import inspect

    parameters = inspect.signature(rope.describe).parameters
    for name in ("dropped", "unpaired"):
        assert parameters[name].default is inspect.Parameter.empty, name
    with pytest.raises(TypeError):
        rope.describe(np.array([0.1, 0.2]))
