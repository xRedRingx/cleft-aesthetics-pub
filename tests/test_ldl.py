"""Label Distribution Learning: the target construction, gate 3, and the leak.

The head itself is torch and is verified in scripts/verify_backbone_builds.py.
Everything that decides whether the arm measures what it claims -- the truth
vector, the gate-3 bias, the packing, and the isolation of the target columns
from the input -- is torch-free and tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.train import ldl


def rows_from(distributions):
    """Manifest-shaped rows carrying a distribution and its expectation."""
    out = []
    for values in distributions:
        row = {name: f"{v}" for name, v in zip(ldl.SOFT_COLUMNS, values)}
        row["mean"] = f"{float(np.dot(values, ldl.GRADES)):.10f}"
        out.append(row)
    return out


FIVE_RATERS = [
    [0.0, 0.4, 0.2, 0.4, 0.0],   # {2,2,3,4,4} -- mean 3.0
    [0.0, 0.0, 1.0, 0.0, 0.0],   # {3,3,3,3,3} -- mean 3.0, different case
    [0.2, 0.2, 0.2, 0.2, 0.2],   # mean 3.0, maximally uncertain
    [0.0, 0.0, 0.0, 0.6, 0.4],   # mean 4.4
]


# --------------------------------------------------------------------------
# the target, and the truth vector it must agree with
# --------------------------------------------------------------------------


def test_the_distribution_is_read_normalised_and_validated():
    values = ldl.target_distributions(rows_from(FIVE_RATERS))
    assert values.shape == (4, 5)
    assert np.allclose(values.sum(axis=1), 1.0)

    # The two cases the arm exists to distinguish have the SAME mean and
    # different distributions -- which is the point of the stage.
    assert abs(ldl.expectation(values)[0] - ldl.expectation(values)[1]) < 1e-12
    assert not np.allclose(values[0], values[1])

    # A set that sums to 0.999 is renormalised rather than shifting every KL
    # by a per-patient constant.
    skewed = ldl.target_distributions(rows_from([[0.0, 0.4, 0.4, 0.199, 0.0]]))
    assert np.allclose(skewed.sum(axis=1), 1.0)

    with pytest.raises(ldl.LDLError, match="no target at all"):
        ldl.target_distributions(rows_from([[0.0] * 5]))
    with pytest.raises(ldl.LDLError, match="negative"):
        ldl.target_distributions(rows_from([[-0.1, 0.5, 0.6, 0.0, 0.0]]))
    with pytest.raises(ldl.LDLError, match="soft_"):
        ldl.target_distributions([{"mean": "3.0"}])


def test_the_truth_vector_is_the_mean_arms_and_that_is_checked():
    """**The whole comparability argument.** LDL is scored on `mean` so it is
    comparable with the mean arm; that holds only if `mean` really is the
    expectation of the soft columns. If it were a different weighting the arm
    would train toward one quantity and be scored against another, and the
    resulting PCC would answer a question nobody asked."""
    distributions = ldl.target_distributions(rows_from(FIVE_RATERS))
    labels = ldl.expectation(distributions)

    report = ldl.assert_mean_is_the_soft_expectation(labels, distributions)
    assert report["checked"] is True
    assert report["worst_absolute_disagreement"] < 1e-9

    # A `mean` column that is something else -- here the median -- is refused
    # rather than silently accepted.
    with pytest.raises(ldl.LDLError, match="not the expectation"):
        ldl.assert_mean_is_the_soft_expectation(labels + 0.25, distributions)

    with pytest.raises(ldl.LDLError, match="labels against"):
        ldl.assert_mean_is_the_soft_expectation(labels[:2], distributions)


# --------------------------------------------------------------------------
# gate 3: an untrained head predicts the training-fold mean
# --------------------------------------------------------------------------


def test_the_gate3_bias_predicts_the_training_mean_exactly():
    """A zero-weight softmax head with zero bias emits the uniform
    distribution, whose expectation is 3.0 -- NOT the training mean. That
    naive initialisation fails gate 3, which is why the bias is the log of the
    training marginal: softmax(log q) == q, so epoch 0 predicts q, whose
    expectation is the mean of the training patients' own expectations."""
    distributions = ldl.target_distributions(rows_from(FIVE_RATERS))
    train_mean = float(ldl.expectation(distributions).mean())

    bias = ldl.log_marginal_bias(distributions)
    # softmax(bias), computed here rather than trusted.
    shifted = np.exp(bias - bias.max())
    predicted = shifted / shifted.sum()
    assert abs(float(ldl.expectation(predicted[None, :])[0]) - train_mean) < 1e-9

    # And the naive alternative really does fail, so the construction is
    # load-bearing rather than decorative.
    uniform = np.full(5, 0.2)
    assert abs(float(ldl.expectation(uniform[None, :])[0]) - 3.0) < 1e-12
    assert abs(3.0 - train_mean) > 0.1, (
        "if the training mean were 3.0 this fixture could not tell the correct "
        "initialisation from the naive one"
    )


def test_a_grade_with_no_votes_does_not_produce_an_infinite_bias():
    """The tails ARE sparse -- it is why CORAL/CORN were rejected -- so a
    training fold with no votes at grade 5 is expected, not exceptional."""
    distributions = ldl.target_distributions(
        rows_from([[0.4, 0.6, 0.0, 0.0, 0.0], [0.6, 0.4, 0.0, 0.0, 0.0]])
    )
    bias = ldl.log_marginal_bias(distributions)
    assert np.all(np.isfinite(bias))
    # The floor is small enough not to move the marginal that matters.
    shifted = np.exp(bias - bias.max())
    predicted = shifted / shifted.sum()
    assert abs(float(ldl.expectation(predicted[None, :])[0]) - 1.5) < 1e-5


# --------------------------------------------------------------------------
# the leak: targets ride in the feature row and must never be read as input
# --------------------------------------------------------------------------


def test_packing_round_trips_and_survives_harness_slicing():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(4, 7))
    distributions = ldl.target_distributions(rows_from(FIVE_RATERS))

    packed = ldl.pack_targets(embeddings, distributions)
    assert packed.shape == (4, 7 + 5)

    rows = np.array([2, 0])
    back_embeddings, back_targets = ldl.unpack_targets(packed[rows], 7)
    assert np.allclose(back_embeddings, embeddings[rows])
    assert np.allclose(back_targets, distributions[rows])

    with pytest.raises(ldl.LDLError, match="packed rows are"):
        ldl.unpack_targets(packed, 3)
    with pytest.raises(ldl.LDLError, match="distributions must be"):
        ldl.pack_targets(embeddings, distributions[:2])


class _HonestHead:
    """Reads only the embedding half."""

    def __init__(self, dim):
        self.dim = dim

    def predict(self, packed):
        embeddings, _ = ldl.unpack_targets(packed, self.dim)
        return embeddings.sum(axis=1)


class _LeakingHead:
    """Reads the LABEL out of the target columns -- the catastrophic defect.

    Deliberately `t @ GRADES` rather than `packed.sum(axis=1)`. A head summing
    the target columns adds a constant, because every valid distribution sums
    to 1, and leaks nothing; the leak that matters recovers the expectation,
    which IS the label the arm is scored on.
    """

    def __init__(self, dim):
        self.dim = dim

    def predict(self, packed):
        array = np.asarray(packed, dtype=float)
        return array[:, : self.dim].sum(axis=1) + array[:, -5:] @ ldl.GRADES


class _ConstantWeightHead:
    """Reads `sum(targets)`, which is 1.0 for every valid distribution."""

    def __init__(self, dim):
        self.dim = dim

    def predict(self, packed):
        array = np.asarray(packed, dtype=float)
        return array[:, : self.dim].sum(axis=1) + array[:, -5:].sum(axis=1)


def test_the_target_isolation_check_catches_a_leaking_head():
    """**The sharpest defect available in this arm.** The label rides inside
    the feature array, so a head spanning the full packed width predicts the
    label from the label and scores near-perfectly -- with the right shapes,
    a valid fit and a plausible metrics.json.

    Tested by perturbing the target columns and requiring the prediction not
    to move, which checks the PROPERTY. Inspecting a weight matrix's width
    would not: a matrix of the correct width could still be applied to the
    wrong slice.
    """
    rng = np.random.default_rng(1)
    embeddings = rng.normal(size=(4, 7))
    packed = ldl.pack_targets(
        embeddings, ldl.target_distributions(rows_from(FIVE_RATERS))
    )

    report = ldl.assert_head_ignores_targets(_HonestHead(7), packed)
    assert report["checked"] is True
    assert report["prediction_shift_when_targets_perturbed"] == 0
    # The perturbation moved the expectation -- which is the quantity a leak
    # recovers, so a check that left it unchanged would detect nothing.
    assert report["expectation_shift_applied"] > 1.0

    with pytest.raises(ldl.LDLError, match="reaching the model"):
        ldl.assert_head_ignores_targets(_LeakingHead(7), packed)


def test_a_permutation_perturbation_would_have_missed_the_leak():
    """**Why the check perturbs the expectation and not the arrangement.**

    Every valid distribution sums to 1, so permuting the target columns leaves
    any constant-weight reader seeing exactly what it saw. An earlier version
    of this check reversed the distribution -- which is a permutation, and
    which for palindromic rows is not even that. It appeared to work only
    because it assigned a slice from an overlapping view of its own memory.

    So: a constant-weight reader is not a leak (it adds 1.0 to every
    prediction and carries no label information), and the reader that IS a
    leak recovers the expectation. The check has to move that.
    """
    rng = np.random.default_rng(2)
    embeddings = rng.normal(size=(4, 7))
    distributions = ldl.target_distributions(rows_from(FIVE_RATERS))
    packed = ldl.pack_targets(embeddings, distributions)

    # Reading sum(targets) is harmless: it is 1.0 for every row, so it cannot
    # carry the label. The check correctly does NOT flag it.
    report = ldl.assert_head_ignores_targets(_ConstantWeightHead(7), packed)
    assert report["checked"] is True

    # And a permutation really would have been blind to the real leak.
    permuted = distributions[:, ::-1]
    assert np.allclose(permuted.sum(axis=1), distributions.sum(axis=1)), (
        "a permutation preserves the sum, which is what a constant-weight "
        "reader sees -- so it cannot distinguish the two heads above"
    )
    # Three of the four fixture rows are palindromic, so a reversal is close
    # to a no-op even for a weighted reader.
    palindromic = sum(
        1 for row in distributions if np.allclose(row, row[::-1])
    )
    assert palindromic >= 3


# --------------------------------------------------------------------------
# what the label field means
# --------------------------------------------------------------------------


def test_an_ldl_arm_stays_inside_the_parameter_check():
    """**The 2026-07-28 defect, in the place it would recur a third time.**

    The LDL feature kind is a SUFFIX on whatever it sits on, so
    ``precomputed_backbone_embeddings+label_distribution`` matches neither
    FROZEN_EMBEDDINGS nor ARTIFACT_EMBEDDINGS exactly. A membership test
    against those sets would have let every LDL arm fall out of the check
    silently -- exactly how the patch arms escaped it the first time.
    """
    from cleft.train import phase3

    report = {
        "features": f"frozen_backbone_embeddings{phase3.LDL_FEATURE_SUFFIX}",
        "feature_dim": 768,
    }
    # A 768-dim embedding gives a 768x5 + 5 = 3,845-parameter head.
    summary = phase3.parameter_summary(
        "head", report,
        {"total_parameters": 3845, "trainable_parameters": 3845},
        backbone="vit_b16",
    )
    assert summary["trainable_parameters"] == 3845

    # The scalar head's width is NOT accepted for an LDL arm: 769 would mean
    # the head is a single output and the distribution is not being modelled.
    with pytest.raises(phase3.Phase3Error, match="5-output softmax"):
        phase3.parameter_summary(
            "head", report,
            {"total_parameters": 769, "trainable_parameters": 769},
            backbone="vit_b16",
        )

    # And a missing width is refused rather than skipped.
    with pytest.raises(phase3.Phase3Error, match="nothing to compare"):
        phase3.parameter_summary(
            "head",
            {"features": f"frozen_backbone_embeddings{phase3.LDL_FEATURE_SUFFIX}"},
            {"total_parameters": 3845, "trainable_parameters": 3845},
            backbone="vit_b16",
        )


def test_ldl_is_not_a_manifest_column():
    """Every other label names a column; this one names five plus a
    construction, so the loader has to treat it specially rather than passing
    it to a column lookup that would raise."""
    assert ldl.LDL_LABEL == "ldl"
    assert ldl.LDL_LABEL not in ldl.SOFT_COLUMNS
    assert len(ldl.SOFT_COLUMNS) == ldl.N_GRADES == len(ldl.GRADES)
    assert list(ldl.GRADES) == [1.0, 2.0, 3.0, 4.0, 5.0]
