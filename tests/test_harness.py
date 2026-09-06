"""The minimal training path, and gates 3, 4 and 6.

No torch. The backbone arrives through a factory, so everything here runs on the
laptop and the real ViT-B/16 is only ever met on the cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.train import gates as G
from cleft.train import harness as H
from cleft.train import stub as S

N_PATIENTS = 60
N_FOLDS = 5


def cohort(n=N_PATIENTS, seed=0, signal=0.8):
    """Patients whose features carry a real but noisy signal about the label."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(1.0, 5.0, size=n)
    features = np.column_stack(
        [
            truth * signal + rng.normal(0.0, 0.5, size=n),
            rng.normal(0.0, 1.0, size=n),
            rng.normal(0.0, 1.0, size=n),
        ]
    )
    ids = list(range(1, n + 1))
    assignments = {pid: index % N_FOLDS for index, pid in enumerate(ids)}
    return features, truth, ids, assignments


@pytest.fixture(scope="module")
def run():
    features, labels, ids, assignments = cohort()
    return H.run_cv(
        features=features,
        labels=labels,
        patient_ids=ids,
        assignments=assignments,
        make_backbone=S.StubBackbone,
        config=H.TrainConfig(max_epochs=12, patience=3),
    )


# --------------------------------------------------------------------------
# the path runs end to end
# --------------------------------------------------------------------------


def test_every_fold_runs(run):
    assert len(run.folds) == N_FOLDS


def test_every_patient_gets_exactly_one_prediction(run):
    assert len(run.oof_ids) == N_PATIENTS
    assert len(set(run.oof_ids)) == N_PATIENTS


def test_the_pooled_metrics_are_reported(run):
    reported = run.metrics()
    for key in ("pcc", "spearman", "qwk_3cat", "mae", "rmse", "selected_epochs"):
        assert key in reported
    assert reported["n"] == N_PATIENTS


def test_the_stub_actually_learns(run):
    """If it did not, early stopping and the curve would be untested."""
    assert run.metrics()["pcc"] > 0.5


def test_oof_truth_matches_the_labels(run):
    features, labels, ids, _ = cohort()
    label_of = dict(zip(ids, labels))
    assert run.oof_truth == pytest.approx([label_of[p] for p in run.oof_ids])


def test_the_run_is_deterministic():
    features, labels, ids, assignments = cohort()
    kwargs = dict(
        features=features, labels=labels, patient_ids=ids,
        assignments=assignments, make_backbone=S.StubBackbone,
        config=H.TrainConfig(max_epochs=8, patience=3),
    )
    first, second = H.run_cv(**kwargs), H.run_cv(**kwargs)
    assert first.oof_ids == second.oof_ids
    assert np.array_equal(first.oof_predictions, second.oof_predictions)


# --------------------------------------------------------------------------
# gate 3 -- epoch-0 calibration
# --------------------------------------------------------------------------


def test_a_clean_backbone_passes_epoch_zero(run):
    for fold in run.folds:
        assert fold.epoch0_pred_mean == pytest.approx(fold.train_label_mean, abs=0.3)


def test_epoch0_mse_matches_a_constant_predictor(run):
    """The clean stub predicts one value, so the identity should hold exactly."""
    for fold in run.folds:
        assert fold.epoch0_inner_val_mse >= fold.inner_val_var - 1e-9


def test_the_gate_has_teeth_against_the_real_defect():
    """Not "some wrong number" -- the SR-GNN signature specifically.

    The void ladder's broken arms sat at epoch-0 inner_val_mse 1.12-8.63 against
    0.40-0.88 clean. The stub is offset so it lands in that band, so this proves
    the assertion catches THAT defect rather than merely being strict.
    """
    features, labels, ids, assignments = cohort()
    # Placed inside the measured broken band for THIS cohort's variance, so the
    # stub reproduces the void-ladder signature rather than an arbitrary error.
    variance = float(np.var(labels))
    offset = S.offset_for_target_mse(variance, target_mse=variance + 1.9)

    with pytest.raises(H.HarnessError) as excinfo:
        H.run_cv(
            features=features,
            labels=labels,
            patient_ids=ids,
            assignments=assignments,
            make_backbone=lambda: S.BrokenHeadBackbone(offset=offset),
            config=H.TrainConfig(max_epochs=4, patience=2),
        )

    message = str(excinfo.value)
    assert "epoch-0" in message
    assert "BatchNorm" in message, "the message must name the defect it models"
    assert "label variance" in message


@pytest.mark.parametrize("target", [1.12, 3.0, 8.63])
def test_the_whole_measured_broken_band_is_caught(target):
    """1.12-8.63 was the observed range; the gate must reject all of it.

    Labels are drawn with a variance near the real cohort's rather than uniform
    on 1-5: averaging five raters compresses the spread, so a uniform draw has a
    variance ABOVE the bottom of the broken band and 1.12 would not be
    expressible as an offset at all.
    """
    rng = np.random.default_rng(3)
    labels = np.clip(rng.normal(2.6, 0.8, size=200), 1.0, 5.0)
    variance = float(np.var(labels))
    assert variance < 1.0, "the real label variance is well under the broken band"
    offset = S.offset_for_target_mse(variance, target_mse=target)

    with pytest.raises(H.HarnessError, match="predicted mean"):
        H.assert_epoch0_calibrated(
            pred_mean=float(np.mean(labels)) + offset,
            inner_val_mse=variance + offset**2,
            inner_val_var=variance,
            inner_val_mean=float(np.mean(labels)),
            train_label_mean=float(np.mean(labels)),
            train_label_var=variance,
        )


@pytest.mark.parametrize("mse", [0.40, 0.60, 0.88])
def test_the_measured_clean_band_is_accepted(mse):
    """0.40-0.88 against an inner-val variance in that range must pass."""
    H.assert_epoch0_calibrated(
        pred_mean=2.5,
        inner_val_mse=mse,
        inner_val_var=mse,
        inner_val_mean=2.5,
        train_label_mean=2.5,
        train_label_var=0.64,
    )


def test_a_mis_scaled_head_is_caught_by_the_mean_check():
    with pytest.raises(H.HarnessError, match="predicted mean"):
        H.assert_epoch0_calibrated(
            pred_mean=4.9,
            inner_val_mse=0.6,
            inner_val_var=0.64,
            inner_val_mean=2.5,
            train_label_mean=2.5,
            train_label_var=0.64,
        )


def test_the_dispersion_check_is_exact_for_a_constant_predictor():
    """The correction that made this gate usable at any sample size.

    mse = var(inner) + (mean(inner) - c)^2 holds exactly for a constant c, so
    comparing against that expectation gives ratio 1 however small the inner-val
    split is. Two earlier drafts compared against a raw variance -- the training
    fold's, then the inner-val's -- and each failed a clean stub (2.09, then
    3.21) purely because mean(inner) drifts from mean(train) by sampling.
    """
    # A large, entirely legitimate gap between the inner-val and predicted means.
    H.assert_epoch0_calibrated(
        pred_mean=2.5,
        inner_val_mse=0.234 + 0.72**2,
        inner_val_var=0.234,
        inner_val_mean=2.5 + 0.72,
        train_label_mean=2.5,
        train_label_var=0.64,
    )
    # Same numbers, but the model's outputs are not constant.
    with pytest.raises(H.HarnessError, match="outputs vary"):
        H.assert_epoch0_calibrated(
            pred_mean=2.5,
            inner_val_mse=3.0,
            inner_val_var=0.234,
            inner_val_mean=2.5 + 0.72,
            train_label_mean=2.5,
            train_label_var=0.64,
        )


def test_zero_variance_labels_are_rejected():
    with pytest.raises(H.HarnessError, match="zero variance"):
        H.assert_epoch0_calibrated(
            pred_mean=3.0, inner_val_mse=0.0, inner_val_var=0.5, inner_val_mean=3.0,
            train_label_mean=3.0, train_label_var=0.0,
        )
    with pytest.raises(H.HarnessError, match="inner-val labels have zero variance"):
        H.assert_epoch0_calibrated(
            pred_mean=3.0, inner_val_mse=0.0, inner_val_var=0.0, inner_val_mean=3.0,
            train_label_mean=3.0, train_label_var=0.5,
        )


def test_the_offset_helper_produces_the_requested_mse():
    """So the teeth-test lands in the measured band by construction, not by luck."""
    variance = 0.64
    for target in (1.2, 3.0, 8.6):
        offset = S.offset_for_target_mse(variance, target)
        assert variance + offset**2 == pytest.approx(target, abs=1e-9)


def test_the_measured_bands_are_recorded():
    assert S.CLEAN_EPOCH0_MSE == (0.40, 0.88)
    assert S.BROKEN_EPOCH0_MSE == (1.12, 8.63)


# --------------------------------------------------------------------------
# gate 4 -- epoch policy
# --------------------------------------------------------------------------


def test_a_selected_epoch_is_recorded_per_fold(run):
    for fold in run.folds:
        assert fold.selected_epoch >= 1
        assert fold.selected_epoch <= len(fold.curve)


def test_the_curve_records_both_loss_and_metric(run):
    """The void ladder's collapse was visible in PCC while the loss looked fine."""
    for entry in run.folds[0].curve:
        assert {"epoch", "train_loss", "inner_val_mse", "inner_val_pcc"} <= set(entry)


def test_early_stopping_actually_stops():
    features, labels, ids, assignments = cohort(signal=0.0)  # nothing to learn
    result = H.run_cv(
        features=features, labels=labels, patient_ids=ids, assignments=assignments,
        make_backbone=S.StubBackbone,
        config=H.TrainConfig(max_epochs=40, patience=2),
    )
    assert all(len(f.curve) < 40 for f in result.folds), "patience never triggered"


def test_patience_is_identical_across_folds(run):
    """PLAN §4.6: identical across every arm, not a per-arm knob."""
    config = H.TrainConfig()
    assert isinstance(config.patience, int)
    assert "patience" not in {f.fold for f in run.folds}


def test_an_unknown_monitor_is_rejected():
    with pytest.raises(H.HarnessError, match="monitor"):
        H.TrainConfig(monitor="test_pcc")


def test_the_monitor_can_be_switched_to_pcc():
    features, labels, ids, assignments = cohort()
    result = H.run_cv(
        features=features, labels=labels, patient_ids=ids, assignments=assignments,
        make_backbone=S.StubBackbone,
        config=H.TrainConfig(max_epochs=8, patience=3, monitor="inner_val_pcc"),
    )
    assert len(result.folds) == N_FOLDS


# --------------------------------------------------------------------------
# the split -- inner-val comes from the training folds only
# --------------------------------------------------------------------------


def test_inner_val_never_touches_the_test_fold(run):
    for fold in run.folds:
        assert not set(fold.inner_val_ids) & set(fold.test_ids)
        assert not set(fold.train_ids) & set(fold.test_ids)
        assert not set(fold.train_ids) & set(fold.inner_val_ids)


def test_the_split_is_deterministic():
    ids = list(range(1, 41))
    assert H.inner_val_split(ids, 0.2, 1337, 0) == H.inner_val_split(ids, 0.2, 1337, 0)


def test_different_folds_hold_out_different_patients():
    ids = list(range(1, 41))
    first = H.inner_val_split(ids, 0.2, 1337, 0)[1]
    second = H.inner_val_split(ids, 0.2, 1337, 1)[1]
    assert first != second, "every fold holding out the same positions is a smell"


def test_a_frac_that_leaves_no_training_data_is_rejected():
    with pytest.raises(H.HarnessError, match="inner_val_frac"):
        H.TrainConfig(inner_val_frac=0.9)


def test_a_patient_with_no_fold_is_rejected():
    features, labels, ids, assignments = cohort()
    del assignments[ids[0]]
    with pytest.raises(H.HarnessError, match="no fold"):
        H.run_cv(
            features=features, labels=labels, patient_ids=ids,
            assignments=assignments, make_backbone=S.StubBackbone,
        )


def test_a_length_mismatch_is_rejected():
    features, labels, ids, assignments = cohort()
    with pytest.raises(H.HarnessError, match="length mismatch"):
        H.run_cv(
            features=features, labels=labels[:-1], patient_ids=ids,
            assignments=assignments, make_backbone=S.StubBackbone,
        )


# --------------------------------------------------------------------------
# gate 6 -- OOF integrity by reconstruction
# --------------------------------------------------------------------------


def test_oof_reconstruction_passes_on_a_clean_run(run):
    _, _, ids, assignments = cohort()
    report = G.reconstruct_oof(
        run.fold_record(),
        expected_patients=set(ids),
        expected_assignments=assignments,
        oof_ids=run.oof_ids,
    )
    assert report["n_patients"] == N_PATIENTS
    assert report["every_patient_predicted_once"]
    assert sum(report["fold_sizes"]) == N_PATIENTS


def test_reconstruction_catches_a_patient_training_on_itself(run):
    record = run.fold_record()
    record["folds"][0]["train_ids"].append(record["folds"][0]["test_ids"][0])
    _, _, ids, assignments = cohort()

    with pytest.raises(G.GateError, match="own training data"):
        G.reconstruct_oof(
            record, expected_patients=set(ids),
            expected_assignments=assignments, oof_ids=run.oof_ids,
        )


def test_reconstruction_catches_inner_val_drawn_from_the_test_fold(run):
    record = run.fold_record()
    record["folds"][1]["inner_val_ids"].append(record["folds"][1]["test_ids"][0])
    _, _, ids, assignments = cohort()

    with pytest.raises(G.GateError, match="own training data|overlaps the test"):
        G.reconstruct_oof(
            record, expected_patients=set(ids),
            expected_assignments=assignments, oof_ids=run.oof_ids,
        )


def test_reconstruction_catches_a_fold_assignment_that_disagrees(run):
    _, _, ids, assignments = cohort()
    tampered = dict(assignments)
    tampered[run.folds[0]["test_ids"][0] if isinstance(run.folds[0], dict) else run.folds[0].test_ids[0]] = 9

    with pytest.raises(G.GateError, match="cleft_v1 assigns"):
        G.reconstruct_oof(
            run.fold_record(), expected_patients=set(ids),
            expected_assignments=tampered, oof_ids=run.oof_ids,
        )


def test_reconstruction_catches_a_missing_patient(run):
    _, _, ids, assignments = cohort()
    with pytest.raises(G.GateError, match="never appear"):
        G.reconstruct_oof(
            run.fold_record(),
            expected_patients=set(ids) | {9999},
            expected_assignments={**assignments, 9999: 0},
            oof_ids=run.oof_ids,
        )


def test_reconstruction_catches_a_duplicate_in_the_pooled_vector(run):
    _, _, ids, assignments = cohort()
    with pytest.raises(G.GateError, match="duplicate|covers"):
        G.reconstruct_oof(
            run.fold_record(), expected_patients=set(ids),
            expected_assignments=assignments,
            oof_ids=run.oof_ids + [run.oof_ids[0]],
        )


def test_an_empty_fold_record_is_rejected():
    with pytest.raises(G.GateError, match="no folds"):
        G.reconstruct_oof(
            {"folds": []}, expected_patients=set(),
            expected_assignments={}, oof_ids=[],
        )


# --------------------------------------------------------------------------
# gate 5 -- metric verification
# --------------------------------------------------------------------------


def test_metric_verification_passes():
    report = G.verify_metrics()
    assert report["max_abs_difference"] < 1e-9
    assert set(report["metrics_verified"]) == {
        "pcc", "spearman", "qwk_3cat", "mae", "rmse"
    }


def test_metric_verification_checks_the_paired_machinery():
    report = G.verify_metrics()
    lo, hi = report["paired_self_ci"]
    assert lo <= 0.0 <= hi
    assert report["paired_self_delta"] == pytest.approx(0.0, abs=1e-12)


def test_metric_verification_runs_inside_the_run_not_only_in_tests():
    """So a cluster result carries evidence its own metric code was checked."""
    import inspect

    from cleft.train import gates

    assert inspect.isfunction(gates.verify_metrics)
    assert "scipy" in inspect.getsource(gates.verify_metrics)
