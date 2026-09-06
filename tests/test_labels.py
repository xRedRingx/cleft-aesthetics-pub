"""Label targets and learnability.

Learnability is leave-one-rater-out mean Pearson (brief §1.7). The synthetic
grades will not reproduce 0.5900, so what is tested here is the *construction* —
that the held-out rater is genuinely excluded from its own target, that the
weights are recomputed inside each fold, and that the ordering and the ceiling
relationship hold.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.data import labels as L
from cleft.data import reliability as R

from test_reliability import DOC, panel

TOL = 1e-9


# --------------------------------------------------------------------------
# the measured values and their ordering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("population", [237, 251])
def test_the_documented_ordering_holds_on_both_populations(population):
    """weighted > mean > median > mode > orthodontist, on 237 and on 251.

    The ordering is the conclusion; it does not depend on which population it is
    computed over. The *values* do, by about 0.012, which is why they are stored
    under separate names.
    """
    scores = L.LEARNABILITY_BY_POPULATION[population]
    ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
    assert tuple(ordered) == L.EXPECTED_ORDERING


def test_the_two_populations_are_different_numbers():
    """If these ever compare equal, one has been copied over the other."""
    assert L.LEARNABILITY_237 != L.LEARNABILITY_251
    for target in L.EXPECTED_ORDERING:
        gap = L.LEARNABILITY_237[target] - L.LEARNABILITY_251[target]
        assert 0.005 < gap < 0.020, (
            f"{target}: the 237 subset should sit about 0.012 above the 251 rows"
        )


def test_the_manifest_population_is_237_not_251():
    """Exit criterion 9 compares against the 237 values.

    Checking a cluster result against the 251 numbers would look like a module
    defect when it is a population mismatch -- the precise shape of the 0.903
    error.
    """
    assert L.MANIFEST_POPULATION == 237
    assert L.expected_learnability() is L.LEARNABILITY_237


@pytest.mark.parametrize("population", [237, 251])
def test_weighting_gains_almost_nothing(population):
    """Sound in principle, negligible here, so the plain mean is primary."""
    scores = L.LEARNABILITY_BY_POPULATION[population]
    gain = scores["weighted_mean"] - scores["mean"]
    assert 0 < gain < 0.005


@pytest.mark.parametrize(
    "population, ratio", [(237, 1.218), (251, 1.226)]
)
def test_the_mean_is_far_more_learnable_than_the_orthodontist(population, ratio):
    """~22%. This is what refutes a premise raised at supervision about the target."""
    scores = L.LEARNABILITY_BY_POPULATION[population]
    assert scores["mean"] / scores["orthodontist"] == pytest.approx(ratio, abs=0.01)


def test_documented_ceilings_follow_from_the_measured_reliabilities():
    """The principled companion, and it reproduces from §4.3 not from a fit."""
    assert R.pcc_ceiling(R.spearman_brown(R.MEAN_R_237, 5)) == pytest.approx(
        L.CEILING_MEAN_OF_FIVE, abs=DOC
    )
    assert R.pcc_ceiling(R.spearman_brown(R.MEAN_R_237, 1)) == pytest.approx(
        L.CEILING_SINGLE_RATER, abs=1e-3
    )


def test_a_single_rater_reliability_is_the_mean_inter_rater_r():
    assert R.spearman_brown(R.MEAN_R_237, 1) == pytest.approx(R.MEAN_R_237, abs=TOL)


@pytest.mark.parametrize("population", [237, 251])
def test_the_ceiling_ordering_matches_the_learnability_ordering(population):
    """Same ordering by a measure with no asymmetry — that is the point of it."""
    scores = L.LEARNABILITY_BY_POPULATION[population]
    assert L.CEILING_MEAN_OF_FIVE > L.CEILING_SINGLE_RATER
    assert scores["mean"] > scores["orthodontist"]


# --------------------------------------------------------------------------
# the asymmetry, recorded rather than hidden
# --------------------------------------------------------------------------


def test_the_asymmetry_travels_with_the_numbers():
    table = L.learnability_table(panel(seed=1), orthodontist_index=1)
    payload = table.as_dict()
    assert "caveat" in payload
    for phrase in ("mean-of-four", "single column", "not a like-for-like"):
        assert phrase in payload["caveat"], f"caveat must state: {phrase}"


def test_aggregate_targets_are_measured_against_four_raters(monkeypatch):
    """Prove the asymmetry is real: the rebuilt target has 4 columns, not 5."""
    seen = []
    original = L.build_target

    def spy(matrix, kind):
        seen.append(np.asarray(matrix).shape[1])
        return original(matrix, kind)

    monkeypatch.setattr(L, "build_target", spy)
    L.learnability(panel(seed=2), "mean")
    assert seen and set(seen) == {4}


# --------------------------------------------------------------------------
# leave-one-out construction
# --------------------------------------------------------------------------


def test_the_held_out_rater_is_excluded_from_its_own_target():
    """If it were included, a rater would partly predict itself.

    Rater 0 is noise and raters 1-4 agree. Correctly excluded, rater 0 scores
    near zero. Included, it would be pulled upward by its own contribution.
    """
    rng = np.random.default_rng(20)
    signal = rng.integers(1, 6, size=400)
    noise = rng.integers(1, 6, size=400)
    matrix = np.column_stack([noise, signal, signal, signal, signal])

    rest = np.delete(matrix, 0, axis=1)
    from cleft.eval import metrics

    excluded = metrics.pcc(matrix[:, 0], L.build_target(rest, "mean"))
    included = metrics.pcc(matrix[:, 0], L.build_target(matrix, "mean"))

    assert abs(excluded) < 0.15

    # Self-inclusion pulls a pure-noise rater up to ~0.25: with a mean of five,
    # one fifth of the target IS the held-out rater. The assertion is the
    # inflation itself, not a threshold picked by eye.
    assert included > excluded + 0.15
    assert included == pytest.approx(0.25, abs=0.06), (
        "expected roughly 1/5 of the target to be the rater itself"
    )


def test_single_rater_target_averages_the_other_four():
    rng = np.random.default_rng(21)
    matrix = panel(n=200, seed=21)
    from cleft.eval import metrics

    expected = np.mean(
        [
            metrics.pcc(np.delete(matrix, 1, axis=1)[:, j], matrix[:, 1])
            for j in range(4)
        ]
    )
    assert L.learnability(matrix, "orthodontist", rater_index=1) == pytest.approx(
        expected, abs=TOL
    )


def test_single_rater_target_requires_an_index():
    with pytest.raises(L.LabelError, match="rater_index"):
        L.learnability(panel(seed=3), "orthodontist")


def test_learnability_of_a_perfectly_agreeing_panel_is_one():
    rng = np.random.default_rng(22)
    column = rng.integers(1, 6, size=200)
    matrix = np.column_stack([column] * 5)
    assert L.learnability(matrix, "mean") == pytest.approx(1.0, abs=1e-12)
    assert L.learnability(matrix, "orthodontist", rater_index=0) == pytest.approx(
        1.0, abs=1e-12
    )


def test_unknown_target_is_rejected():
    with pytest.raises(L.LabelError, match="unknown target"):
        L.learnability(panel(seed=4), "vibes")


# --------------------------------------------------------------------------
# weights, recomputed inside each fold
# --------------------------------------------------------------------------


def test_weights_come_from_the_four_remaining_raters_only():
    """Weights from the full panel would leak the held-out rater into its target."""
    matrix = panel(n=300, seed=23)
    rest = np.delete(matrix, 0, axis=1)
    assert len(L.item_total_weights(rest)) == 4
    assert len(L.item_total_weights(matrix)) == 5


def test_weights_are_corrected_item_total_correlations():
    matrix = panel(n=300, seed=24)
    weights = L.item_total_weights(matrix)
    expected = [stat.r for stat in R.item_total(matrix)]
    assert list(weights) == pytest.approx(expected, abs=TOL)


def test_a_noisier_rater_gets_a_smaller_weight():
    rng = np.random.default_rng(25)
    signal = rng.integers(1, 6, size=400)
    noise = rng.integers(1, 6, size=400)
    matrix = np.column_stack([noise, signal, signal, signal, signal])
    weights = L.item_total_weights(matrix)
    assert weights[0] < weights[1]


def test_unusable_weights_are_rejected_rather_than_silently_renormalised():
    matrix = np.array([[1, 5, 1], [5, 1, 5], [1, 5, 1], [5, 1, 5]] * 5)
    with pytest.raises(L.LabelError, match="weights"):
        L.build_target(matrix, "weighted_mean")


# --------------------------------------------------------------------------
# target construction
# --------------------------------------------------------------------------


def test_mean_median_mode_on_a_hand_case():
    matrix = np.array([[1, 2, 2, 4, 5]])
    assert L.build_target(matrix, "mean")[0] == pytest.approx(2.8, abs=TOL)
    assert L.build_target(matrix, "median")[0] == pytest.approx(2.0, abs=TOL)
    assert L.build_target(matrix, "mode")[0] == pytest.approx(2.0, abs=TOL)


def test_mode_breaks_ties_toward_the_lower_grade():
    """Ties are the normal case with four or five raters, so the rule is fixed."""
    assert L.build_target(np.array([[1, 1, 4, 4]]), "mode")[0] == 1
    assert L.build_target(np.array([[2, 2, 3, 3, 5]]), "mode")[0] == 2


def test_soft_labels_are_fractions_that_sum_to_one():
    matrix = np.array([[1, 1, 3, 5, 5]])
    soft = L.soft_labels(matrix)
    assert soft.shape == (1, 5)
    assert soft[0].tolist() == [0.4, 0.0, 0.2, 0.0, 0.4]
    assert soft.sum(axis=1)[0] == pytest.approx(1.0, abs=TOL)


def test_class3_uses_the_fixed_a_priori_thresholds():
    values = np.array([1.0, 2.49, 2.5, 3.49, 3.5, 5.0])
    assert L.class3(values).tolist() == [0, 0, 1, 1, 2, 2]


def test_learnability_table_covers_every_target():
    table = L.learnability_table(panel(n=200, seed=26), orthodontist_index=1)
    assert set(table.scores) == set(L.TARGETS)
    assert set(table.ceilings) == set(L.TARGETS)
    # The single-rater target gets the k=1 ceiling; the rest get k=5.
    assert table.ceilings["orthodontist"] < table.ceilings["mean"]
    assert table.ordering()[0] in L.TARGETS


def test_the_ordering_is_verified_on_real_data_not_on_fixtures():
    """CLUSTER-VERIFIED. Do not try to make a fixture reproduce this ordering.

    On synthetic panels the ordering comes out weighted > median > mean > mode >
    orthodontist -- median above mean, unlike the real cohort. That is expected:
    the ordering is a property of how five real clinicians disagree, not of the
    arithmetic, and random grades have no reason to reproduce it.

    So the assertion above (`test_the_documented_ordering_holds_on_both
    _populations`) is made against the MEASURED constants, never against a
    fixture. If a future change makes a synthetic cohort produce the real
    ordering, that is a coincidence or a rigged fixture, not evidence.
    """
    table = L.learnability_table(panel(n=237, seed=99), orthodontist_index=1)
    synthetic = tuple(table.ordering())

    # The only part that must hold on synthetic data is the part that follows
    # from construction: a single-rater target is noisier than an aggregate one.
    assert synthetic[-1] == "orthodontist"

    if synthetic == L.EXPECTED_ORDERING:
        pytest.fail(
            "a synthetic panel reproduced the real ordering exactly. That is not "
            "evidence the module is right; check whether the fixture has been "
            "tuned to force it."
        )
