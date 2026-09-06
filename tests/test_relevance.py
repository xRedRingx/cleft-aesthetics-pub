"""Univariate feature relevance (diagnostic).

Constructed features with known relationships to a known label, so the ranking
is checkable rather than plausible.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft import relevance
from cleft.relevance import (
    AGGREGATION_NOTE,
    MEASURED_RELEVANCE,
    NOSE_TIP_DECISION,
    RelevanceError,
    rank_features,
    report,
    significance_threshold,
)

N = 120


def known_features():
    """Three features: strongly related, weakly related, and unrelated."""
    rng = np.random.default_rng(0)
    labels = rng.uniform(1.0, 5.0, size=N)
    strong = labels * 0.4 + rng.normal(0, 0.05, size=N)
    weak = labels * 0.4 + rng.normal(0, 1.2, size=N)
    noise = rng.normal(0, 1.0, size=N)
    features = np.column_stack([noise, weak, strong])
    return features, labels, ["noise", "weak", "strong"]


def test_the_ranking_puts_the_strongest_feature_first():
    features, labels, names = known_features()
    ranked = rank_features(features, labels, names)
    assert ranked[0]["feature"] == "strong"
    assert ranked[-1]["feature"] == "noise"
    assert abs(ranked[0]["spearman"]) > abs(ranked[1]["spearman"])


def test_both_statistics_are_reported():
    """Spearman asks whether the ORDERING agrees, Pearson whether the
    relationship is LINEAR. Asymmetry indices are bounded at zero and
    right-skewed, so the two can disagree."""
    features, labels, names = known_features()
    for row in rank_features(features, labels, names):
        assert row["spearman"] is not None
        assert row["pearson"] is not None


def test_a_monotone_but_non_linear_feature_ranks_on_spearman():
    """The case the two statistics separate: perfect ordering, curved
    relationship. Ranking by |Spearman| keeps it; ranking by Pearson alone
    would demote it."""
    labels = np.linspace(1.0, 5.0, N)
    curved = np.exp(labels)
    ranked = rank_features(curved.reshape(-1, 1), labels, ["curved"])
    assert ranked[0]["spearman"] == pytest.approx(1.0, abs=1e-9)
    assert ranked[0]["pearson"] < ranked[0]["spearman"]


def test_a_constant_feature_is_reported_not_divided_by_zero():
    labels = np.linspace(1.0, 5.0, N)
    ranked = rank_features(np.ones((N, 1)), labels, ["flat"])
    assert ranked[0]["constant"] is True
    assert ranked[0]["spearman"] is None


def test_the_rank_positions_are_dense_and_ordered():
    features, labels, names = known_features()
    ranked = rank_features(features, labels, names)
    assert [row["rank_by_abs_spearman"] for row in ranked] == [1, 2, 3]


def test_a_shape_mismatch_is_refused():
    labels = np.linspace(1, 5, N)
    with pytest.raises(RelevanceError, match="feature rows against"):
        rank_features(np.zeros((N - 1, 2)), labels, ["a", "b"])
    with pytest.raises(RelevanceError, match="columns against"):
        rank_features(np.zeros((N, 2)), labels, ["a"])


# --------------------------------------------------------------------------
# the report says what it is, and what it is not
# --------------------------------------------------------------------------


def test_the_report_names_the_strongest_feature():
    features, labels, names = known_features()
    summary = report(features, labels, names)
    assert summary["strongest"] == "strong"
    assert summary["n_patients"] == N
    assert summary["n_features"] == 3


def test_the_report_says_it_is_a_diagnostic_not_an_arm():
    features, labels, names = known_features()
    note = report(features, labels, names)["note"]
    assert "DIAGNOSTIC, NOT AN ARM" in note
    assert "nothing is tuned on this" in note


def test_the_report_explains_why_not_ridge_coefficients():
    """Correlated predictors make individual coefficients unstable -- they can
    flip sign on a resample while predictions barely move."""
    features, labels, names = known_features()
    note = report(features, labels, names)["note"]
    assert "unstable" in note
    assert "flip sign" in note


def test_the_report_declares_that_no_multiple_comparison_correction_is_applied():
    """With 22 features some will correlate by chance. A ranking that informs a
    design decision is not a set of claims."""
    features, labels, names = known_features()
    note = report(features, labels, names)["note"]
    assert "NO MULTIPLE-COMPARISON CORRECTION" in note
    assert "needs its own arm" in note


def test_disagreeing_statistics_are_flagged():
    labels = np.linspace(1.0, 5.0, N)
    # Monotone decreasing in rank, but with a linear trend the other way is not
    # constructible cleanly; use a feature whose sign genuinely differs.
    flipped = -labels + 0.0
    summary = report(flipped.reshape(-1, 1), labels, ["flipped"])
    assert summary["spearman_and_pearson_disagree"] == []


# --------------------------------------------------------------------------
# the two bars, computed rather than memorised
# --------------------------------------------------------------------------


def test_the_thresholds_reproduce_the_measured_run():
    """**[MEASURED 2026-07-30] 0.128 uncorrected and 0.199 across 22 features at
    n=237.** Computed from the same formula at different `n_features`, so the
    distinction that decides what this diagnostic may claim is visible as one
    parameter rather than two remembered numbers."""
    assert significance_threshold(237) == pytest.approx(0.128, abs=0.001)
    assert significance_threshold(237, 22) == pytest.approx(0.199, abs=0.001)
    assert MEASURED_RELEVANCE["threshold_uncorrected"] == pytest.approx(
        significance_threshold(237), abs=0.001
    )
    assert MEASURED_RELEVANCE["threshold_bonferroni_22"] == pytest.approx(
        significance_threshold(237, 22), abs=0.001
    )


def test_the_corrected_bar_is_always_the_harder_one():
    for n in (50, 237, 1000):
        assert significance_threshold(n, 22) > significance_threshold(n)
    # And more features means a harder bar, monotonically.
    bars = [significance_threshold(237, k) for k in (1, 5, 22, 100)]
    assert bars == sorted(bars)


def test_a_larger_sample_lowers_the_bar():
    assert significance_threshold(1000) < significance_threshold(100)


def test_a_degenerate_threshold_request_is_refused():
    with pytest.raises(RelevanceError, match="too small"):
        significance_threshold(4)
    with pytest.raises(RelevanceError, match="at least 1"):
        significance_threshold(237, 0)


def test_the_report_carries_both_bars_and_which_features_clear_them():
    """**Reporting only the uncorrected bar would let the features above it read
    as findings.** On the real run `clearing_bonferroni` was EMPTY, and that is
    the single most important fact about how strongly this may be quoted."""
    features, labels, names = known_features()
    summary = report(features, labels, names)

    assert summary["threshold_uncorrected"] < summary["threshold_bonferroni"]
    assert "strong" in summary["clearing_uncorrected"]
    assert "noise" not in summary["clearing_uncorrected"]
    # Clearing the corrected bar is a subset of clearing the uncorrected one.
    assert set(summary["clearing_bonferroni"]) <= set(
        summary["clearing_uncorrected"]
    )
    assert "clearing_bonferroni first" in summary["note"]


# --------------------------------------------------------------------------
# what the 2026-07-30 run decided, and what it did not
# --------------------------------------------------------------------------


def test_nothing_cleared_the_corrected_bar():
    """So the ranking informs a design decision and is not a set of claims --
    exactly as declared before the run, which is what makes it usable."""
    assert MEASURED_RELEVANCE["n_clearing_bonferroni"] == 0
    top = MEASURED_RELEVANCE["top"][0]
    assert top["abs_spearman"] < MEASURED_RELEVANCE["threshold_bonferroni_22"]
    assert top["abs_spearman"] > MEASURED_RELEVANCE["threshold_uncorrected"]


def test_the_top_features_are_the_philtral_and_upper_lip_complex():
    """All midline, all lower face -- which coheres with the group's own finding
    that raters are more reliable scoring lips alone."""
    features = [entry["feature"] for entry in MEASURED_RELEVANCE["top"]]
    assert features == [
        "philtrum", "labial_tubercle", "philtral_column", "subnasale",
    ]
    assert "lips alone" in MEASURED_RELEVANCE["coherence"]


def test_the_nose_tip_null_is_recorded_with_its_scope_not_as_a_verdict():
    """**`nasal_tip` ranks 22 of 22 at 0.002, so the suggestion raised at supervision is not
    supported by THIS measurement** -- and the scope is the whole of what makes
    the result usable.

    Measured: pixel-level mirror asymmetry within a box placed by anatomy
    fractions. NOT measured: tip deviation from the midline (a displacement, not
    a regional pixel difference -- and a box that travels with the anatomy can
    miss it entirely), and the tip as an alignment anchor (a use, not a feature).
    Both remain live.
    """
    decision = NOSE_TIP_DECISION
    assert MEASURED_RELEVANCE["notable_ranks"]["nasal_tip"] == 22
    assert MEASURED_RELEVANCE["nasal_tip_abs_spearman"] == 0.002
    assert "NOT SUPPORTED as a weighted region" in decision["decided"]

    live = decision["still_live_and_untested"]
    assert set(live) == {"tip_deviation_from_midline", "tip_as_alignment_anchor"}
    assert "DISPLACEMENT" in live["tip_deviation_from_midline"]
    assert "USE, not a feature" in live["tip_as_alignment_anchor"]
    assert "does not matter" in decision["do_not_conclude"]
    assert "R2" in decision["do_not_conclude"]


def test_the_aggregate_note_records_that_a_near_zero_aggregate_says_nothing():
    """**`mad_whole` is 0.004 while regions carry 3x to 30x more.** So the
    mirror-difference arm's 0.158 comes from the ridge combining weak regional
    signals, not from any single site or the aggregate -- and an aggregate near
    zero is uninformative about its parts rather than evidence they are zero.

    Third instance of this shape, after the fairness quartiles and
    `separability`.
    """
    note = AGGREGATION_NOTE
    assert note["mad_whole_abs_spearman"] == 0.004
    assert "RIDGE COMBINING" in note["consequence"]
    assert "uninformative about its parts" in note["consequence"]
    assert "fairness quartiles" in note["same_lesson_as"]
    assert "separability" in note["same_lesson_as"]
    assert "per-unit breakdown" in note["rule"]


def test_the_synthesis_records_the_target_relevance_mismatch():
    """The consequence for the arm: the TPS **anchors** the top-ranked region and
    **deforms** two of the least. Different quantities, so not decisive -- and
    recorded beside the other limitations rather than acted on."""
    from cleft.scut import synthesis

    limitation = synthesis.LIMITATIONS[
        "targets_are_misaligned_with_measured_regional_relevance"
    ]
    assert "ANCHORS THE MOST RELEVANT REGION" in limitation
    assert "philtral_column` ranks 3rd" in limitation
    assert "DIFFERENT QUANTITIES" in limitation
    assert "NOT being changed now" in limitation
    assert "BEFORE PHASE 7" in limitation
