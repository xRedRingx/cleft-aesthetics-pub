"""Reliability, tested on matrices small enough to check by hand.

Brief §3: the synthetic cohort's random grades will not reproduce 0.1605, so
these functions are tested against hand-computable matrices and against the
*relationships* between the measured values, not against the cohort fixture.

The relationship tests are the point of §1.5. Asserting sqrt(reliability) ==
ceiling in code is what stops the two quantities collapsing into one again.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from cleft.data import reliability as R

TOL = 1e-9

#: The documented values are quoted to 4 decimal places, so a value derived from
#: two of them can differ in the fifth. Tighter than this asserts the rounding,
#: not the relationship.
DOC = 1e-4


def panel(n: int = 200, k: int = 5, seed: int = 0, noise: float = 1.2) -> np.ndarray:
    """Grades from raters who partly agree, which is what real panels look like.

    Independent uniform grades have a mean inter-rater r of about zero, and
    Spearman-Brown on a near-zero or negative r produces a meaningless (possibly
    negative) reliability. Testing the summary on noise would therefore exercise
    an input the function will never see.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(1.0, 5.0, size=n)
    columns = [
        np.clip(np.rint(truth + rng.normal(0.0, noise, size=n)), 1, 5).astype(np.int64)
        for _ in range(k)
    ]
    return np.column_stack(columns)


# --------------------------------------------------------------------------
# the relationship that was got wrong twice
# --------------------------------------------------------------------------


def test_the_ceiling_is_the_square_root_of_reliability():
    """§1.5, encoded so the distinction cannot collapse again."""
    assert R.pcc_ceiling(R.RELIABILITY_237) == pytest.approx(R.PCC_CEILING_237, abs=5e-5)
    assert R.pcc_ceiling(R.RELIABILITY_251) == pytest.approx(R.PCC_CEILING_251, abs=5e-5)


def test_reliability_follows_from_the_mean_inter_rater_correlation():
    """Spearman-Brown at k=5 turns the measured r into the measured reliability."""
    assert R.spearman_brown(R.MEAN_R_237, 5) == pytest.approx(R.RELIABILITY_237, abs=DOC)
    assert R.spearman_brown(R.MEAN_R_251, 5) == pytest.approx(R.RELIABILITY_251, abs=DOC)


def test_reliability_and_ceiling_are_different_numbers():
    """The whole point. If these ever compare equal, something has been merged."""
    assert R.RELIABILITY_237 != R.PCC_CEILING_237
    assert R.PCC_CEILING_237 > R.RELIABILITY_237, (
        "a square root of a number below 1 is larger than the number; the ceiling "
        "is always above the reliability"
    )


def test_the_237_cohort_is_the_one_that_matters():
    """Both cohorts are recorded, so a 251-row number is recognisable not alarming."""
    assert R.RELIABILITY_237 > R.RELIABILITY_251
    assert R.PCC_CEILING_237 > R.PCC_CEILING_251


def test_qwk_is_not_a_ceiling():
    """0.4173 is an agreement index and has been mistaken for a bound before."""
    assert R.QWK_237 < R.PCC_CEILING_237
    assert R.QWK_237 != R.RELIABILITY_237


@pytest.mark.parametrize("population", [237, 251])
def test_agreement_indices_are_named_by_population(population):
    """Criterion 8 compares against the 237 column; the brief quotes 251."""
    assert R.FLEISS_BY_POPULATION[population] > 0
    assert R.QWK_BY_POPULATION[population] > 0


def test_the_two_populations_give_different_agreement_values():
    """If these ever compare equal, one has been copied over the other."""
    assert R.FLEISS_237 != R.FLEISS_251
    assert R.QWK_237 != R.QWK_251
    assert R.FLEISS_237 > R.FLEISS_251
    assert R.QWK_237 > R.QWK_251
    assert R.MANIFEST_POPULATION == 237


def test_fleiss_is_low_and_that_is_the_research_problem():
    """0.166 is 'slight' agreement. It is the problem being studied, not a defect."""
    assert 0.10 < R.FLEISS_237 < 0.25


# --------------------------------------------------------------------------
# Spearman-Brown, by hand
# --------------------------------------------------------------------------


def test_spearman_brown_worked_by_hand():
    # k=5, r=0.5 -> 2.5 / (1 + 2.0) = 0.8333...
    assert R.spearman_brown(0.5, 5) == pytest.approx(2.5 / 3.0, abs=TOL)
    # A single-rater mean is just the rater.
    assert R.spearman_brown(0.4, 2) == pytest.approx(0.8 / 1.4, abs=TOL)


def test_spearman_brown_is_monotonic_in_k():
    values = [R.spearman_brown(0.4696, k) for k in (2, 3, 5, 10)]
    assert values == sorted(values), "more raters cannot reduce the mean's reliability"


def test_spearman_brown_at_k_one_is_the_rater_itself():
    """k=1 is meaningful: one rater's reliability is its mean correlation with
    a parallel rater. It is what gives the single-rater ceiling sqrt(0.4696).
    """
    assert R.spearman_brown(0.4696, 1) == pytest.approx(0.4696, abs=TOL)


def test_spearman_brown_rejects_k_below_one():
    with pytest.raises(R.ReliabilityError, match="k must be"):
        R.spearman_brown(0.5, 0)


def test_ceiling_rejects_negative_reliability():
    with pytest.raises(R.ReliabilityError, match="negative"):
        R.pcc_ceiling(-0.1)


# --------------------------------------------------------------------------
# Fleiss kappa, by hand
# --------------------------------------------------------------------------


def test_fleiss_kappa_perfect_agreement():
    """Two subjects, three raters, unanimous each time."""
    matrix = np.array([[1, 1, 1], [5, 5, 5]])
    assert R.fleiss_kappa(matrix) == pytest.approx(1.0, abs=TOL)


def test_fleiss_kappa_worked_example():
    """Three subjects, four raters, two categories used.

    S1 unanimous 1        -> P = (16-4)/12 = 1
    S2 split 2/2          -> P = (4+4-4)/12 = 1/3
    S3 unanimous 2        -> P = 1
    P_bar = 7/9;  p1 = p2 = 0.5;  P_e = 0.5
    kappa = (7/9 - 1/2) / (1/2) = 5/9
    """
    matrix = np.array([[1, 1, 1, 1], [1, 1, 2, 2], [2, 2, 2, 2]])
    assert R.fleiss_kappa(matrix) == pytest.approx(5.0 / 9.0, abs=TOL)


def test_fleiss_kappa_is_near_zero_for_chance_agreement():
    rng = np.random.default_rng(0)
    matrix = rng.integers(1, 6, size=(400, 5))
    assert abs(R.fleiss_kappa(matrix)) < 0.05


def test_fleiss_kappa_undefined_when_everyone_uses_one_category():
    matrix = np.array([[3, 3, 3], [3, 3, 3]])
    with pytest.raises(R.ReliabilityError, match="undefined"):
        R.fleiss_kappa(matrix)


def test_fleiss_kappa_is_invariant_to_categories_nobody_used():
    """Unweighted Fleiss kappa genuinely does not care about empty categories.

    An unused category contributes zero to both the per-subject agreement and to
    the expected-agreement sum, so it cancels. This is asserted rather than
    assumed because the opposite is easy to believe: for a *weighted* index the
    category set changes the distance scale, and conflating the two is the sort
    of quantity confusion R2 exists to catch.
    """
    matrix = np.array([[1, 1, 2, 2], [2, 2, 1, 1]])
    assert R.fleiss_kappa(matrix) == pytest.approx(
        R.fleiss_kappa(matrix, categories=(1, 2)), abs=TOL
    )


def test_the_category_set_must_still_span_every_grade_present():
    """Declaring the full 1-5 scale is what makes the computation total.

    Narrowing it to what happens to appear is how a grade used by one rater and
    not the other ends up unrepresented.
    """
    assert R.CATEGORIES == (1, 2, 3, 4, 5)
    matrix = np.array([[1, 3], [2, 1]])
    with pytest.raises(KeyError):
        R.pairwise_qwk(matrix, categories=(1, 2))


# --------------------------------------------------------------------------
# QWK, against sklearn
# --------------------------------------------------------------------------


def test_pairwise_qwk_matches_sklearn():
    rng = np.random.default_rng(4)
    matrix = rng.integers(1, 6, size=(120, 5))
    for (i, j), got in zip(
        [(a, b) for a in range(5) for b in range(a + 1, 5)],
        R.pairwise_qwk(matrix),
    ):
        expected = cohen_kappa_score(
            matrix[:, i], matrix[:, j], labels=list(R.CATEGORIES), weights="quadratic"
        )
        assert got == pytest.approx(expected, abs=1e-9)


def test_pairwise_qwk_pair_count():
    rng = np.random.default_rng(5)
    matrix = rng.integers(1, 6, size=(50, 5))
    assert len(R.pairwise_qwk(matrix)) == 10, "5 raters give 10 unordered pairs"


def test_qwk_of_identical_raters_is_one():
    rng = np.random.default_rng(6)
    column = rng.integers(1, 6, size=60)
    matrix = np.column_stack([column, column])
    assert R.mean_pairwise_qwk(matrix) == pytest.approx(1.0, abs=TOL)


# --------------------------------------------------------------------------
# correlations and alpha
# --------------------------------------------------------------------------


def test_mean_inter_rater_r_of_identical_raters_is_one():
    rng = np.random.default_rng(7)
    column = rng.integers(1, 6, size=80)
    matrix = np.column_stack([column, column, column])
    assert R.mean_inter_rater_r(matrix) == pytest.approx(1.0, abs=1e-12)


def test_pairwise_r_pair_count():
    rng = np.random.default_rng(8)
    matrix = rng.integers(1, 6, size=(40, 5))
    assert len(R.pairwise_r(matrix)) == 10


def test_cronbach_alpha_of_identical_items_is_one():
    rng = np.random.default_rng(9)
    column = rng.integers(1, 6, size=100).astype(float)
    matrix = np.column_stack([column] * 4)
    assert R.cronbach_alpha(matrix) == pytest.approx(1.0, abs=1e-12)


def test_cronbach_alpha_is_low_for_independent_items():
    rng = np.random.default_rng(10)
    matrix = rng.integers(1, 6, size=(500, 5))
    assert R.cronbach_alpha(matrix) < 0.15


def test_alpha_and_reliability_are_distinct_quantities():
    """Raw alpha is not the standardised form, and the brief lists both."""
    rng = np.random.default_rng(11)
    matrix = rng.integers(1, 6, size=(200, 5))
    alpha = R.cronbach_alpha(matrix)
    standardised = R.spearman_brown(R.mean_inter_rater_r(matrix), 5)
    assert not math.isclose(alpha, standardised, abs_tol=1e-12)


# --------------------------------------------------------------------------
# per-rater
# --------------------------------------------------------------------------


def test_item_total_returns_one_entry_per_rater():
    rng = np.random.default_rng(12)
    matrix = rng.integers(1, 6, size=(100, 5))
    stats = R.item_total(matrix)
    assert [s.index for s in stats] == [0, 1, 2, 3, 4]


def test_item_total_is_corrected_excludes_the_rater_itself():
    """An uncorrected item-total correlates a rater with a total containing it.

    That inflates every value. Here rater 0 is pure noise while raters 1-4 agree,
    so a corrected coefficient must be near zero.
    """
    rng = np.random.default_rng(13)
    signal = rng.integers(1, 6, size=300)
    noise = rng.integers(1, 6, size=300)
    matrix = np.column_stack([noise, signal, signal, signal, signal])
    stats = R.item_total(matrix)

    # Rater 0 is noise, and the mean of the other four is pure signal.
    assert abs(stats[0].r) < 0.15

    # Rater 1 agrees with raters 2-4 exactly, but its "others" mean still
    # contains the noise column, so the corrected value is high and NOT 1.0.
    # An uncorrected item-total would put rater 1 much closer to 1 by including
    # itself in the total, which is the inflation being guarded against.
    assert 0.85 < stats[1].r < 1.0


def test_item_total_reports_both_pearson_and_spearman():
    rng = np.random.default_rng(14)
    matrix = rng.integers(1, 6, size=(150, 5))
    for stat in R.item_total(matrix):
        assert not math.isnan(stat.r)
        assert not math.isnan(stat.rho)


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------


def test_summarise_reports_every_documented_quantity():
    summary = R.summarise(panel(n=200, seed=15))

    assert summary.n_subjects == 200 and summary.n_raters == 5
    assert summary.pcc_ceiling == pytest.approx(
        math.sqrt(summary.reliability), abs=TOL
    )
    assert len(summary.per_rater) == 5

    payload = summary.as_dict()
    for key in (
        "mean_inter_rater_r",
        "reliability_spearman_brown",
        "pcc_ceiling",
        "cronbach_alpha",
        "fleiss_kappa",
        "mean_pairwise_qwk",
    ):
        assert key in payload


def test_summary_is_shareable_aggregates_only():
    """No patient-level value may appear in the SHAREABLE payload."""
    payload = R.summarise(panel(n=60, seed=16)).as_dict()
    assert "grades" not in payload and "matrix" not in payload
    assert all(
        not isinstance(value, (list, np.ndarray)) or key == "per_rater"
        for key, value in payload.items()
    )


# --------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.array([1, 2, 3]), "2-D"),
        (np.array([[1, 2, 3, 4, 5]]), "subjects"),
        (np.array([[1], [2]]), "raters"),
        (np.array([[1, 2], [3, 9]]), "outside"),
    ],
)
def test_bad_matrices_are_rejected(matrix, message):
    with pytest.raises(R.ReliabilityError, match=message):
        R.summarise(matrix)
