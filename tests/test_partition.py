"""Sensitivity to partitioning (Phase 4).

The measurement runs on the cluster. What is testable here is that the thing
being measured is the thing being named: several genuinely different partitions
of the same patients, the frozen generator untouched, the training-set confound
reported, and the un-measured quantity recorded as un-measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.train import partition as P

N = 60


def cohort(n: int = N):
    ids = list(range(1, n + 1))
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=n).tolist()
    # Every class needs at least n_folds members for the frozen generator.
    for index in range(30):
        labels[index] = index % 3
    return ids, labels


# --------------------------------------------------------------------------
# the frozen generator is used unchanged
# --------------------------------------------------------------------------


def test_the_seed_does_not_vary_the_partition():
    """**Why fold count is varied instead of a seed.**

    ``folds.generate`` runs with ``shuffle=False`` and its docstring says the
    seed is recorded but does not affect the partition. Regenerating at five
    seeds would return one fold set five times and a spread of exactly zero --
    which would read as "fold assignment does not matter" while actually meaning
    nothing was varied.
    """
    from cleft.data import folds

    ids, labels = cohort()
    first = folds.generate(ids, labels, n_folds=5, seed=1337).assignments
    second = folds.generate(ids, labels, n_folds=5, seed=7).assignments
    assert first == second


def test_different_fold_counts_give_different_partitions():
    ids, labels = cohort()
    five = P.assignments_for(ids, labels, 5)
    ten = P.assignments_for(ids, labels, 10)

    assert len(set(five.values())) == 5
    assert len(set(ten.values())) == 10
    assert five != ten


def test_every_patient_is_assigned_exactly_once():
    ids, labels = cohort()
    for n_folds in P.DEFAULT_FOLD_COUNTS:
        assignments = P.assignments_for(ids, labels, n_folds)
        assert sorted(assignments) == sorted(ids)


# --------------------------------------------------------------------------
# leave-one-patient-out
# --------------------------------------------------------------------------


def test_leave_one_out_gives_one_patient_per_fold():
    ids, _ = cohort()
    assignments = P.leave_one_out_assignments(ids)
    assert len(set(assignments.values())) == len(ids)
    assert P.training_sizes(assignments)["max_train_size"] == len(ids) - 1


def test_leave_one_out_does_not_go_through_the_stratifier():
    """There is nothing to stratify -- every test fold is one patient, so there
    is no assignment choice to make. The frozen generator would refuse it anyway,
    because no class has 237 members."""
    from cleft.data import folds

    ids, labels = cohort()
    with pytest.raises(folds.FoldError):
        folds.generate(ids, labels, n_folds=len(ids))

    # And the direct construction succeeds.
    assert len(P.leave_one_out_assignments(ids)) == len(ids)


def test_duplicate_patient_ids_are_refused():
    with pytest.raises(P.PartitionError, match="unique"):
        P.leave_one_out_assignments([1, 2, 2, 3])


# --------------------------------------------------------------------------
# the confound, quantified rather than hidden
# --------------------------------------------------------------------------


def test_more_folds_means_more_training_data():
    """**The confound, and it is not separable here.** Fold count changes the
    partition AND how much each model sees, so the spread mixes the two."""
    ids, labels = cohort()
    five = P.training_sizes(P.assignments_for(ids, labels, 5))
    ten = P.training_sizes(P.assignments_for(ids, labels, 10))
    lopo = P.training_sizes(P.leave_one_out_assignments(ids))

    assert five["mean_train_size"] < ten["mean_train_size"] < lopo["mean_train_size"]
    assert lopo["mean_train_size"] == len(ids) - 1


def test_the_report_carries_the_training_sizes_per_partitioning():
    """So a reader can see which of the two is moving."""
    results = {
        "5fold": {"pcc": 0.25, "training_sizes": {"mean_train_size": 189.6}},
        "lopo": {"pcc": 0.27, "training_sizes": {"mean_train_size": 236.0}},
    }
    report = P.sensitivity(results)
    assert report["training_sizes"]["5fold"]["mean_train_size"] == 189.6
    assert report["training_sizes"]["lopo"]["mean_train_size"] == 236.0


# --------------------------------------------------------------------------
# the report says what it measures, and what it does not
# --------------------------------------------------------------------------


def test_the_report_names_the_quantity_it_measures():
    report = P.sensitivity({"5fold": {"pcc": 0.25}, "10fold": {"pcc": 0.24}})
    assert report["measures"] == "sensitivity_to_partitioning"
    assert "not fold-assignment variance" in report["note"].lower()


def test_the_report_names_the_quantity_it_does_not_measure():
    """Paired BCa captures patient sampling; the seed band captures training
    procedure; fold assignment is caught by neither, and the design fixes folds
    so it cannot see it. A gap in the claim criterion, for limitations."""
    missing = P.sensitivity({"5fold": {"pcc": 0.25}})["does_not_measure"]
    assert missing["quantity"] == "fold-assignment variance"
    assert missing["captured_by_paired_bca"] == "patient sampling"
    assert missing["captured_by_seed_band"] == "training procedure"
    assert missing["captured_by_neither"] == "fold assignment"
    assert "shuffle=False" in missing["why_the_design_cannot_see_it"]
    assert missing["belongs_in"].startswith("limitations")


def test_the_report_calls_itself_a_bound_not_a_measurement():
    missing = P.sensitivity({"5fold": {"pcc": 0.25}})["does_not_measure"]
    assert "BOUND on it, not a measurement" in missing["what_this_gives_instead"]
    assert "confounded with training-set size" in missing["what_this_gives_instead"]


def test_the_headline_is_lopo_against_the_quoted_number():
    """Leave-one-out has no assignment choice at all, so its distance from the
    5-fold number bounds how much that number depends on partitioning."""
    report = P.sensitivity(
        {"5fold": {"pcc": 0.2529}, "10fold": {"pcc": 0.26}, "lopo": {"pcc": 0.2680}}
    )
    assert report["lopo_minus_5fold"] == pytest.approx(0.0151, abs=1e-6)
    assert report["range"] == pytest.approx(0.0151, abs=1e-6)


def test_the_headline_is_absent_when_either_end_is():
    report = P.sensitivity({"6fold": {"pcc": 0.25}, "10fold": {"pcc": 0.26}})
    assert report["lopo_minus_5fold"] is None


def test_the_note_says_to_compare_against_the_seed_band():
    """A spread inside the band is not a finding."""
    note = P.sensitivity({"5fold": {"pcc": 0.25}})["note"]
    assert "COMPARE THE RANGE AGAINST THE SEED BAND" in note


def test_an_empty_comparison_is_refused():
    with pytest.raises(P.PartitionError, match="no partitionings"):
        P.sensitivity({})


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def test_the_partitionings_are_named_and_complete():
    ids, labels = cohort()
    schemes = P.partitionings(ids, labels)
    assert set(schemes) == {"5fold", "6fold", "10fold", "lopo"}
    for assignments in schemes.values():
        assert sorted(assignments) == sorted(ids)


def test_lopo_can_be_left_out():
    ids, labels = cohort()
    assert P.LOPO not in P.partitionings(ids, labels, include_lopo=False)


def test_the_override_refuses_a_partition_missing_a_patient(tmp_path):
    """A partitioning that does not cover the manifest would silently drop
    patients from the OOF vector and still produce a number."""
    from test_phase3 import artifacts

    from cleft.train import phase3

    manifest_dir, staged_dir = artifacts(tmp_path)
    with pytest.raises(phase3.Phase3Error, match="absent from it"):
        phase3.run(
            manifest_dir=manifest_dir,
            staged_dir=staged_dir,
            backbone="stub",
            assignments_override={1: 0},
            log=lambda *_: None,
        )


def test_an_alternative_partitioning_runs_end_to_end(tmp_path):
    """The override reaches the harness and the OOF vector still covers everyone
    -- gate 6 reconstructs it independently, so this is not self-confirming."""
    from test_phase3 import artifacts

    from cleft.train import phase3

    manifest_dir, staged_dir = artifacts(tmp_path)
    _, labels, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, "g1", "mean"
    )
    lopo = P.leave_one_out_assignments(patient_ids)

    result = phase3.run(
        manifest_dir=manifest_dir,
        staged_dir=staged_dir,
        backbone="stub",
        assignments_override=lopo,
        log=lambda *_: None,
    )
    assert result.summary["oof"]["n"] == len(patient_ids)
    assert len(result.cv.folds) == len(patient_ids)


# --------------------------------------------------------------------------
# cost: the frozen-embedding design has to actually hold at 237 folds
# --------------------------------------------------------------------------


def count_extractions(monkeypatch):
    """Count calls to ``prepare_features`` -- where the backbone meets images."""
    from cleft.train import phase3

    calls = {"n": 0}
    real = phase3.prepare_features

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(phase3, "prepare_features", counting)
    return calls


@pytest.mark.parametrize("n_folds", [2, 5, 10, N])
def test_extraction_count_is_independent_of_fold_count(tmp_path, monkeypatch, n_folds):
    """**The frozen-embedding design's whole promise.**

    A frozen backbone's output is a fixed function of the image, so it must be
    computed once whatever the partitioning. If extraction were inside the fold
    loop, leave-one-patient-out would do 237x the work of a 5-fold run and the
    arm would be unrunnable -- which is exactly the shape of a hang.
    """
    from test_phase3 import artifacts

    from cleft.train import phase3

    manifest_dir, staged_dir = artifacts(tmp_path)
    _, _, patient_ids, _ = phase3.load_inputs(manifest_dir, staged_dir, "g1", "mean")

    if n_folds == N:
        assignments = P.leave_one_out_assignments(patient_ids)
    else:
        assignments = {pid: i % n_folds for i, pid in enumerate(patient_ids)}

    calls = count_extractions(monkeypatch)
    phase3.run(
        manifest_dir=manifest_dir,
        staged_dir=staged_dir,
        backbone="stub",
        assignments_override=assignments,
        log=lambda *_: None,
    )
    assert calls["n"] == 1, (
        f"{n_folds} folds triggered {calls['n']} extractions; it must be 1 "
        "whatever the fold count"
    )


def test_features_are_reused_across_partitionings(tmp_path, monkeypatch):
    """Once per RUN still meant once per PARTITIONING. For a frozen backbone
    that is pure waste, so the arm extracts once for all of them."""
    from test_phase3 import artifacts

    from cleft.train import phase3

    manifest_dir, staged_dir = artifacts(tmp_path)
    images, _, patient_ids, _ = phase3.load_inputs(
        manifest_dir, staged_dir, "g1", "mean"
    )
    once = phase3.prepare_features(images, "stub", "head", {}, "g1")

    calls = count_extractions(monkeypatch)
    for n_folds in (5, 6, 10):
        phase3.run(
            manifest_dir=manifest_dir,
            staged_dir=staged_dir,
            backbone="stub",
            assignments_override={
                pid: i % n_folds for i, pid in enumerate(patient_ids)
            },
            features_override=once,
            log=lambda *_: None,
        )
    assert calls["n"] == 0, "the override must skip extraction entirely"


def test_a_mismatched_feature_override_is_refused(tmp_path):
    """Silently training on the wrong rows would still produce a number."""
    from test_phase3 import artifacts

    from cleft.train import phase3

    manifest_dir, staged_dir = artifacts(tmp_path)
    _, _, patient_ids, _ = phase3.load_inputs(manifest_dir, staged_dir, "g1", "mean")

    with pytest.raises(phase3.Phase3Error, match="features_override has"):
        phase3.run(
            manifest_dir=manifest_dir,
            staged_dir=staged_dir,
            backbone="stub",
            assignments_override={pid: i % 5 for i, pid in enumerate(patient_ids)},
            features_override=(np.zeros((3, 4), dtype=np.float32), {}),
            log=lambda *_: None,
        )


def test_training_cost_scales_with_fold_count_not_extraction(tmp_path):
    """What DOES grow with fold count, so the real cost driver is on record:
    every fold trains its own head, and leave-one-out has 237 of them."""
    from test_phase3 import artifacts

    from cleft.train import phase3

    manifest_dir, staged_dir = artifacts(tmp_path)
    _, _, patient_ids, _ = phase3.load_inputs(manifest_dir, staged_dir, "g1", "mean")

    five = phase3.run(
        manifest_dir=manifest_dir, staged_dir=staged_dir, backbone="stub",
        assignments_override={pid: i % 5 for i, pid in enumerate(patient_ids)},
        log=lambda *_: None,
    )
    lopo = phase3.run(
        manifest_dir=manifest_dir, staged_dir=staged_dir, backbone="stub",
        assignments_override=P.leave_one_out_assignments(patient_ids),
        log=lambda *_: None,
    )
    assert len(lopo.cv.folds) == len(patient_ids)
    assert len(lopo.cv.folds) > len(five.cv.folds)


# --------------------------------------------------------------------------
# stratification is on class3, not on the training label
# --------------------------------------------------------------------------


def test_the_raw_grade_is_refused_as_a_stratification_label():
    """The generator stratifies on the 3-class collapse. Passing the raw 1-5
    grade aborts the run at the first fold count -- with ~237 distinct values
    there is nothing to stratify on."""
    ids, _ = cohort()
    grades = [1.5 + 0.01 * index for index in range(len(ids))]
    with pytest.raises(P.PartitionError, match="3-class collapse"):
        P.assignments_for(ids, grades, 5)


def test_class3_is_accepted():
    ids, labels = cohort()
    assert len(P.assignments_for(ids, labels, 5)) == len(ids)


def test_the_shipped_config_is_valid(repo_root):
    from cleft.config import load_config

    task = load_config(repo_root / "configs" / "p4_partition.yaml")["task"]
    assert task["kind"] == "partition_sensitivity"
    assert list(task["fold_counts"]) == [5, 6, 10]
    assert task["include_lopo"] is True
    # The same arm the bar was measured with, or the spread is not about
    # partitioning.
    bar = load_config(repo_root / "configs" / "p3_train_cv.yaml")["task"]
    assert task["backbone"] == bar["backbone"]
    assert task["geometry"] == bar["geometry"]
    assert task["label"] == bar["label"]
