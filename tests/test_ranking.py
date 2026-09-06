"""The pairwise-logistic ranking objective (Phase 22's six arms).

Following ``train/ldl.py``'s pattern exactly: the parts that can be
checked without torch -- pair construction, the tie rule, the loss
arithmetic, the two head maps and the gate-3 bias -- are module-level
functions, tested here directly. The torch fit loop is thin and is
verified in ``scripts/verify_backbone_builds.py``.

**Fold honesty is structural, and that is what the pair tests check**:
``train_epoch`` receives only its fold's training rows, so pairs formed
inside it cannot span the train/test boundary.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cleft import phase22
from cleft.train import ranking


# --------------------------------------------------------------------------
# pair construction, and the ruled tie rule
# --------------------------------------------------------------------------


def test_all_pairs_are_formed_and_ties_are_dropped():
    """The ruling: drop tied pairs from training."""
    labels = np.array([2.0, 2.0, 2.2, 2.4, 2.4, 2.4])
    high, low = ranking.ordered_pairs(labels)

    # 15 pairs, 4 tied (one in the 2.0s, three in the 2.4s), 11 ordered.
    assert len(high) == len(low) == 11
    assert 6 * 5 // 2 - 4 == 11

    # Every returned pair is strictly ordered, high first.
    assert np.all(labels[high] > labels[low])
    # And no tie survived.
    assert not np.any(labels[high] == labels[low])


def test_every_ordered_pair_appears_exactly_once():
    rng = np.random.default_rng(11)
    labels = rng.integers(10, 16, size=25) / 5.0
    high, low = ranking.ordered_pairs(labels)

    seen = set(zip(high.tolist(), low.tolist()))
    assert len(seen) == len(high), "no pair repeated"
    # Each unordered index pair appears at most once, in one direction.
    unordered = {frozenset(p) for p in seen}
    assert len(unordered) == len(seen)
    # And the count matches a direct enumeration.
    expected = sum(
        1 for i in range(len(labels)) for j in range(i + 1, len(labels))
        if labels[i] != labels[j]
    )
    assert len(high) == expected


def test_pair_construction_is_deterministic_and_needs_no_seed():
    """All pairs each epoch: nothing is sampled, so no sampler seed
    exists to declare (phase22.PAIR_CONSTRUCTION_RULED)."""
    labels = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    first = ranking.ordered_pairs(labels)
    for _ in range(5):
        again = ranking.ordered_pairs(labels)
        assert np.array_equal(first[0], again[0])
        assert np.array_equal(first[1], again[1])
    assert "no sampler seed" in " ".join(
        phase22.PAIR_CONSTRUCTION_RULED["the_ruling"].split()
    ).lower()


def test_pairs_are_confined_to_the_rows_given_which_is_fold_honesty():
    """Structural, not policed: the function cannot reach a row it was
    not handed, so a pair cannot span the train/test boundary."""
    labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    train_rows = np.array([0, 1, 2, 3])  # the fold's training rows

    high, low = ranking.ordered_pairs(labels[train_rows])
    # Indices are into the SUBSET, and every one is in range.
    assert high.max() < len(train_rows)
    assert low.max() < len(train_rows)
    # Mapping back, no test-fold patient appears.
    touched = set(train_rows[np.concatenate([high, low])].tolist())
    assert touched <= set(train_rows.tolist())
    assert not (touched & {4, 5, 6, 7}), "no test row reachable"


def test_a_fully_tied_fold_yields_no_pairs_and_does_not_crash():
    """The degenerate case: every label equal. Zero pairs, zero loss,
    no division by zero."""
    labels = np.full(6, 2.4)
    high, low = ranking.ordered_pairs(labels)
    assert len(high) == 0
    assert ranking.pairwise_logistic_loss(np.zeros(6), high, low) == 0.0


def test_the_tie_counts_reproduce_the_banked_measurement():
    """The rule the arms implement is the rule that was measured."""
    record = phase22.TIE_RULE_RULED
    assert "DROP TIED PAIRS FROM TRAINING" in record["the_ruling"]
    observed = phase22.TIE_FRACTION_OBSERVED
    assert observed["n_ties"] == 2345
    assert observed["n_pairs"] == 27966
    # The surviving pair count named in the ruling.
    assert "25,621" in record["the_ruling"]
    assert 27966 - 2345 == 25621


# --------------------------------------------------------------------------
# the loss
# --------------------------------------------------------------------------


def test_the_loss_is_softplus_of_the_negated_gap():
    scores = np.array([2.0, 1.0, 0.0])
    high = np.array([0, 0, 1])
    low = np.array([1, 2, 2])
    gaps = scores[high] - scores[low]  # 1.0, 2.0, 1.0
    expected = float(np.mean([math.log1p(math.exp(-g)) for g in gaps]))
    assert ranking.pairwise_logistic_loss(scores, high, low) == pytest.approx(
        expected
    )


def test_the_loss_falls_as_the_ordering_improves():
    high, low = np.array([0]), np.array([1])
    losses = [
        ranking.pairwise_logistic_loss(np.array([g, 0.0]), high, low)
        for g in (-2.0, -1.0, 0.0, 1.0, 2.0, 5.0)
    ]
    assert losses == sorted(losses, reverse=True), "monotone in the gap"
    # At a zero gap the loss is log 2 -- a coin flip.
    assert losses[2] == pytest.approx(math.log(2.0))


def test_the_loss_has_no_finite_optimum_which_is_the_ruled_property():
    """Hard targets aim at ORDER ALONE: the loss keeps falling as the
    gap widens, so nothing pins the score scale (ATTRACTOR_FINDING)."""
    high, low = np.array([0]), np.array([1])
    previous = None
    for gap in (1.0, 10.0, 100.0):
        value = ranking.pairwise_logistic_loss(np.array([gap, 0.0]), high, low)
        if previous is not None:
            assert value < previous
        previous = value
    assert previous < 1e-40, "still falling, no attractor"
    assert "NO FINITE OPTIMUM" in " ".join(
        phase22.ATTRACTOR_FINDING["the_hard_target_ruling_avoids_it"].split()
    )


def test_the_loss_is_invariant_to_a_shift_of_all_scores():
    """It reads gaps, not levels -- the property that makes it a
    ranking objective rather than a regression one."""
    scores = np.array([0.3, 1.7, -0.5, 2.2])
    high = np.array([1, 3, 0])
    low = np.array([0, 2, 2])
    base = ranking.pairwise_logistic_loss(scores, high, low)
    for shift in (-10.0, 0.5, 100.0):
        assert ranking.pairwise_logistic_loss(
            scores + shift, high, low
        ) == pytest.approx(base)


# --------------------------------------------------------------------------
# the two heads
# --------------------------------------------------------------------------


def test_the_bounded_head_maps_onto_the_label_scale():
    raw = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    scores = ranking.bounded_scores(raw)
    assert np.all(scores > 1.0) and np.all(scores < 5.0)
    assert scores[2] == pytest.approx(3.0), "zero maps to the midpoint"
    assert np.all(np.diff(scores) > 0), "monotone, so ordering is preserved"


def test_the_bounded_head_saturates_at_the_endpoints_and_that_is_documented():
    """Found by this test, not assumed: beyond ~|37| the logistic
    underflows and the score is EXACTLY an endpoint. Harmless -- both
    metrics accept it -- but the docstring says so rather than implying
    an open interval."""
    extreme = ranking.bounded_scores(np.array([-50.0, 50.0]))
    assert extreme[0] == 1.0 and extreme[1] == 5.0
    assert np.all(extreme >= 1.0) and np.all(extreme <= 5.0)
    assert "It SATURATES" in ranking.bounded_scores.__doc__
    assert "CLOSED [1, 5]" in ranking.bounded_scores.__doc__


def test_the_bounded_bias_makes_an_untrained_head_predict_the_mean():
    """Gate 3 by construction, as EmbeddingHeadBackbone does it --
    frozen apparatus untouched."""
    for mean in (1.5, 2.7544, 3.0, 4.4):
        bias = ranking.bias_for_mean(mean, bounded=True)
        assert ranking.bounded_scores(np.array([bias]))[0] == pytest.approx(
            mean
        )
    # Unbounded: the bias IS the mean.
    assert ranking.bias_for_mean(2.7544, bounded=False) == 2.7544


def test_the_bounded_bias_refuses_a_mean_off_the_label_scale():
    for bad in (1.0, 5.0, 0.5, 7.0):
        with pytest.raises(ranking.RankingError, match="label scale"):
            ranking.bias_for_mean(bad, bounded=True)


def test_bounding_preserves_order_so_both_heads_optimise_the_same_thing():
    """The heads differ in SCALE, not in what the loss can express --
    which is why the bounded-vs-unbounded contrast is a clean one."""
    raw = np.array([-2.0, -0.5, 0.0, 0.7, 3.0])
    assert np.array_equal(np.argsort(raw), np.argsort(ranking.bounded_scores(raw)))


# --------------------------------------------------------------------------
# what the module is, and is not
# --------------------------------------------------------------------------


def test_the_module_imports_without_torch_and_keeps_torch_in_the_methods():
    """The suite's standing constraint: torch stays out of the test
    extra, so the testable core must be importable without it."""
    import inspect

    source = inspect.getsource(ranking)
    # No module-level torch import.
    for line in source.splitlines():
        if line.startswith("import torch") or line.startswith("from torch"):
            raise AssertionError(f"module-level torch import: {line!r}")
    # The backbone's methods import it locally, as LDL's do.
    assert "import torch" in source


def test_the_backbone_satisfies_the_frozen_protocol():
    """train_epoch(features, labels) -> float, the Protocol in FROZEN
    harness.py that eight backbones already implement."""
    import inspect

    backbone = ranking.RankingHeadBackbone(bounded=True)
    for method in ("reset", "train_epoch", "predict"):
        assert callable(getattr(backbone, method)), method
    signature = inspect.signature(ranking.RankingHeadBackbone.train_epoch)
    assert list(signature.parameters)[1:] == ["features", "labels"]

    harness = inspect.getsource(
        __import__("cleft.train.harness", fromlist=["harness"])
    )
    assert "def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:" in harness


def test_the_backbone_mirrors_the_embedding_head_so_only_the_loss_varies():
    """LDL's own justification, applied here: same optimiser, learning
    rate, weight decay and seed, so the arm varies the OBJECTIVE and
    nothing else."""
    from cleft.train.torch_backbone import EmbeddingHeadBackbone

    mirror = EmbeddingHeadBackbone()
    arm = ranking.RankingHeadBackbone(bounded=False)
    for field in ("learning_rate", "weight_decay", "max_steps", "seed"):
        assert getattr(arm, field) == getattr(mirror, field), field


def test_the_backbone_reports_what_it_trains():
    """The report is module-level so it is checkable without torch --
    the same reason LDL puts its testable parts outside the class."""
    for bounded in (True, False):
        report = ranking.parameter_report(bounded)
        assert report["loss"] == "pairwise_logistic"
        assert report["targets"] == "hard"
        assert report["tied_pairs"] == "dropped"
        assert report["head"] == ("bounded_1_to_5" if bounded else "unbounded")
        assert report["monitor"] == (
            "inner_val_mse" if bounded else "inner_val_pcc"
        )


def test_no_second_implementation_of_the_shipped_extract_and_project_path():
    """The rejected option (b): reusing Phase 17's Siamese branches
    would have been a second implementation of shipped code."""
    import inspect

    source = inspect.getsource(ranking)
    for token in ("projection", "left_right", "distance_to_grade",
                  "FrozenExtractor"):
        assert token not in source, token


# --------------------------------------------------------------------------
# [2026-09-01] min_separation: R-clear's restriction, which did not exist
# --------------------------------------------------------------------------


def test_min_separation_changes_the_pair_count_and_matches_the_labels():
    """**Pre-fix this failed**: ``ordered_pairs`` took only labels, so
    R-all and R-clear built identical pair sets and the four cohort runs
    at 4472be00 measured one arm twice.

    A count assertion alone would pass if the threshold were applied to
    the wrong quantity, so the expectation is computed INDEPENDENTLY from
    the fixture's own labels rather than from the function under test.
    """
    labels = np.array([2.0, 2.0, 2.2, 2.4, 2.6, 3.0, 3.4])

    for threshold in (0.0, 0.2, 0.4, 0.6, 1.0):
        high, low = ranking.ordered_pairs(labels, threshold)

        # Independent expectation: a plain double loop over the labels.
        expected = [
            (i, j)
            for i in range(len(labels))
            for j in range(i + 1, len(labels))
            if labels[i] != labels[j]
            and abs(labels[i] - labels[j]) >= threshold
        ]
        assert len(high) == len(expected), threshold
        # And every kept pair really clears the threshold, in the right
        # orientation -- not merely the right COUNT.
        assert np.all(labels[high] > labels[low]), threshold
        assert np.all(
            np.abs(labels[high] - labels[low]) >= threshold
        ), threshold

    # The counts must actually DIFFER across thresholds, or the test
    # would pass on a function that ignores its argument.
    counts = [
        len(ranking.ordered_pairs(labels, t)[0]) for t in (0.0, 0.4, 1.0)
    ]
    assert counts[0] > counts[1] > counts[2], counts


def test_r_all_and_r_clear_are_not_the_same_arm():
    """The defect in its own terms: two backbones differing ONLY in
    pair_source must build different pair sets."""
    labels = np.array([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    se_diff = 0.398942

    r_all = ranking.pair_census(labels, 0.0)
    r_clear = ranking.pair_census(labels, se_diff)

    assert r_all["n_pairs_trained"] != r_clear["n_pairs_trained"]
    assert r_clear["n_pairs_trained"] < r_all["n_pairs_trained"]
    # The census separates the two rulings: ties are dropped from every
    # arm, the threshold only from R-clear.
    assert r_all["n_pairs_below_threshold_dropped"] == 0
    assert r_clear["n_pairs_below_threshold_dropped"] > 0
    assert r_all["n_pairs_tied_dropped"] == r_clear["n_pairs_tied_dropped"]


def test_the_two_arms_reach_different_trained_weights():
    """**The stronger form.** Different pair counts could still leave
    the fit identical if the extra pairs carried no gradient. They do
    not: the trained weights differ."""
    pytest.importorskip("torch")

    rng = np.random.default_rng(31)
    features = rng.normal(size=(24, 6)).astype(np.float32)
    labels = rng.integers(10, 16, size=24) / 5.0

    trained = {}
    for name, threshold in (("r_all", 0.0), ("r_clear", 0.398942)):
        arm = ranking.RankingHeadBackbone(
            bounded=True, min_separation=threshold, max_steps=20, seed=1337
        )
        arm.reset(labels)
        arm.train_epoch(features, labels)
        trained[name] = arm.predict(features)
        assert arm.census()["min_separation"] == threshold

    # Same data, same seed, same everything but the pair set.
    assert not np.allclose(trained["r_all"], trained["r_clear"]), (
        "two arms differing only in min_separation reached identical "
        "predictions -- the restriction is not reaching the fit"
    )


def test_the_parameter_report_carries_the_restriction():
    for threshold in (0.0, 0.398942):
        report = ranking.parameter_report(True, threshold)
        assert report["min_separation"] == threshold
