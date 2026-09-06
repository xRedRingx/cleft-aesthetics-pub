"""Phase 3 wiring: loading, the task, and the factory guarantees.

Everything here runs with the stub backbone. The torch path is exercised only on
the cluster, but its module must still import without torch installed -- which is
itself asserted below.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from cleft.train import determinism as D
from cleft.train import phase3 as P3
from cleft.run import main

from fixtures import builders

N = 40
FOLDS = 5


def artifacts(root, n=N, misalign=False):
    """A manifest and a staged tensor with matching row order."""
    manifest_dir = root / "cleft_v1"
    staged_dir = root / "staged_v1"
    manifest_dir.mkdir(parents=True)
    staged_dir.mkdir(parents=True)

    rng = np.random.default_rng(0)
    truth = np.clip(rng.normal(2.6, 0.8, size=n), 1.0, 5.0)

    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for index in range(n):
        pid = index + 1
        value = truth[index]
        lines.append(
            f"{pid},{1000 + pid},,{value:.4f},{value:.4f},{value:.4f},{value:.4f},"
            f"{value:.4f},0.2,0.2,0.2,0.2,0.2,{index % 3},{index % FOLDS}"
        )
    (manifest_dir / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Features carry a real signal so the stub has something to learn.
    images = np.zeros((n, 8, 8, 3), dtype=np.uint8)
    for index in range(n):
        images[index] = int(np.clip(truth[index] * 40, 0, 255))
    np.save(staged_dir / "staged_patient_g1.npy", images)
    np.save(staged_dir / "staged_patient_g2.npy", images[::-1].copy())

    order = list(range(1, n + 1))
    if misalign:
        order[0], order[1] = order[1], order[0]
    geometry = ["# CLUSTER-ONLY: patient-keyed geometry", "patient_id,aspect_ratio"]
    geometry += [f"{pid},0.74" for pid in order]
    (staged_dir / "geometry.csv").write_text("\n".join(geometry) + "\n", encoding="utf-8")

    return manifest_dir, staged_dir


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("p3")
    manifest_dir, staged_dir = artifacts(root)
    result = P3.run(
        manifest_dir=manifest_dir,
        staged_dir=staged_dir,
        backbone="stub",
        train_config=None,
        log=lambda *_: None,
    )
    return result, manifest_dir, staged_dir


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def test_inputs_load_and_align(built):
    _, manifest_dir, staged_dir = built
    features, labels, ids, assignments = P3.load_inputs(
        manifest_dir, staged_dir, "g1", "mean"
    )
    assert len(features) == len(labels) == len(ids) == N
    assert set(assignments.values()) == set(range(FOLDS))


def test_a_misaligned_staged_tensor_is_rejected(tmp_path):
    """The failure that would train every patient against another's label."""
    manifest_dir, staged_dir = artifacts(tmp_path / "bad", misalign=True)
    with pytest.raises(P3.Phase3Error, match="row order does not match"):
        P3.load_inputs(manifest_dir, staged_dir, "g1", "mean")


def test_a_missing_geometry_is_reported_with_what_exists(built):
    _, manifest_dir, staged_dir = built
    with pytest.raises(P3.Phase3Error, match="staged_patient_g1.npy"):
        P3.load_inputs(manifest_dir, staged_dir, "g9", "mean")


def test_an_unknown_label_column_is_rejected(built):
    _, manifest_dir, staged_dir = built
    with pytest.raises(P3.Phase3Error, match="not a manifest column"):
        P3.load_inputs(manifest_dir, staged_dir, "g1", "beauty")


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def test_the_run_produces_pooled_oof(built):
    result, _, _ = built
    assert result.summary["oof"]["n"] == N
    assert len(result.cv.oof_ids) == N


def test_gates_5_and_6_run_inside_the_run(built):
    """Not only in the suite: a result must carry evidence of its own gates."""
    result, _, _ = built
    assert result.gate_reports["gate5_metric_verification"]["max_abs_difference"] < 1e-9
    assert result.gate_reports["gate6_oof_reconstruction"]["n_patients"] == N


def test_epoch0_is_reported_per_fold(built):
    result, _, _ = built
    epoch0 = result.summary["epoch0"]
    assert len(epoch0["dispersion_ratio"]) == FOLDS

    # The stub emits one value, so the identity is exact and the head sits on
    # the training mean. Both are what gate 3 asserts.
    for ratio in epoch0["dispersion_ratio"]:
        assert ratio == pytest.approx(1.0, abs=0.01)
    for offset in epoch0["head_offset_sd"]:
        assert offset == pytest.approx(0.0, abs=0.01)


def test_the_geometry_note_travels_with_the_result(built):
    """So the freeze cannot be read as having settled a Phase 7 arm."""
    result, _, _ = built
    assert result.summary["geometry"] == "g1"
    assert "Phase 7 arm" in result.summary["geometry_note"]


def test_the_determinism_record_says_what_was_configured(built):
    result, _, _ = built
    record = result.summary["determinism"]
    assert record["seed"] == 1337
    assert record["cublas_workspace_config"] == D.CUBLAS_WORKSPACE_CONFIG
    # torch is absent on the laptop, and that is recorded rather than assumed.
    assert record["torch"] is False


def test_an_unknown_backbone_is_rejected():
    with pytest.raises(P3.Phase3Error, match="unknown backbone"):
        P3.make_factory("resnet50", "head", {}, 1337)


def test_only_the_torch_backbones_gate_on_torch():
    """A positive list, so a torch-free arm added later does not have to
    remember to exclude itself. "Anything that is not the stub" already made
    Phase 4's mirror arm unrunnable on the laptop it was written to be tested
    on."""
    # swin_b joined at Phase 7. It never builds a torch BACKBONE -- its
    # features arrive as a pooled artifact -- but its HEAD is torch, so the
    # determinism gate applies exactly as for vit_b16. The list stays positive
    # and explicit: a torch-free arm added later still has to opt in.
    # srgnn joined at Road B Branch 3 on the same footing as swin_b: its
    # concat arm builds no torch backbone (crop vectors arrive as a
    # region_vectors artifact) but its HEAD is torch.
    # The 7D arms (vit_b32/vit_b8/mvitv2_b and "concat") joined 2026-08-15
    # on the same footing again: artifact-fed, torch head.
    assert P3.TORCH_BACKBONES == (
        "vit_b16", "swin_b", "srgnn", "vit_b32", "vit_b8", "mvitv2_b",
        "concat",
    )
    # The list stays a strict subset of the registry, so a torch-free arm
    # added later still has to opt in rather than being swept in.
    assert set(P3.TORCH_BACKBONES) < set(P3.BACKBONES)
    assert set(P3.TORCH_BACKBONES) < set(P3.BACKBONES)
    for torch_free in ("stub", "ridge"):
        assert torch_free not in P3.TORCH_BACKBONES
    assert set(P3.TORCH_BACKBONES) <= set(P3.BACKBONES)
    for torch_free in ("stub", "ridge"):
        assert torch_free in P3.BACKBONES
        assert torch_free not in P3.TORCH_BACKBONES


# --------------------------------------------------------------------------
# the sanity check -- and why beats_constant is no longer the primary one
# --------------------------------------------------------------------------


def cv_of(predictions, truth):
    from cleft.train.harness import CVResult

    ids = list(range(1, len(truth) + 1))
    return CVResult(
        folds=[],
        oof_ids=ids,
        oof_predictions=np.asarray(predictions, dtype=float),
        oof_truth=np.asarray(truth, dtype=float),
    )


def test_the_run_reports_both_sanity_quantities(built):
    result, _, _ = built
    sanity = result.summary["sanity"]
    assert set(sanity) >= {
        "primary", "pcc", "pcc_positive",
        "constant_predictor_rmse", "arm_rmse", "beats_constant",
        "shrinkage", "n_distinct_3class_bins", "interpretation",
    }
    assert isinstance(sanity["beats_constant"], bool)
    assert isinstance(sanity["pcc_positive"], bool)


def test_pcc_is_the_primary_sanity_check_not_rmse():
    """[MEASURED 2026-07-28] beats_constant was false in 3 of 10 gate-2 seeds
    while pooled OOF PCC was 0.23-0.27 in all ten. In a PCC-primary project the
    RMSE comparison must not be the thing that decides an arm learned."""
    assert P3.sanity_report(cv_of([1.0, 2.0], [1.0, 2.0]))["primary"] == "pcc_positive"


def test_a_shrunk_predictor_correlates_while_losing_to_a_constant():
    """The exact divergence the sweep produced, reproduced deterministically.

    A head that predicts the mean plus a small multiple of the centred truth has
    perfect rank agreement and near-perfect PCC, and still scores a worse RMSE
    than the constant it is shrinking toward. Gating on beats_constant would
    reject it; the project's primary metric would not.
    """
    truth = np.linspace(1.0, 5.0, 40)
    centre = truth.mean()
    # Shrink hard toward the mean, then offset -- correlation survives both.
    shrunk = centre + 0.05 * (truth - centre) + 0.9

    report = P3.sanity_report(cv_of(shrunk, truth))

    assert report["pcc"] > 0.99, "a shrunk predictor still correlates"
    assert report["pcc_positive"] is True
    assert report["beats_constant"] is False, "and still loses on RMSE"
    assert report["shrinkage"] < 0.1, "shrinkage is what explains the divergence"


def test_the_sanity_report_carries_the_interpretation_with_the_numbers():
    """A reader meeting beats_constant: false must not have to rediscover why."""
    text = P3.sanity_report(cv_of([1.0, 2.0], [1.0, 2.0]))["interpretation"]
    assert "calibration" in text.lower()
    assert "0.23-0.27" in text, "the measured divergence, not a vague caveat"


def test_the_sanity_check_still_flags_an_arm_that_learned_nothing():
    """The original job survives: the first Phase 3 run scored RMSE 0.658
    against a label SD of 0.628, and its PCC was negative too."""
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    inverted = P3.sanity_report(cv_of([5.0, 4.0, 3.0, 2.0, 1.0], truth))
    assert inverted["beats_constant"] is False
    assert inverted["pcc_positive"] is False

    good = P3.sanity_report(cv_of(truth + 0.1, truth))
    assert good["beats_constant"] is True
    assert good["pcc_positive"] is True


# --------------------------------------------------------------------------
# what the arm actually trains
# --------------------------------------------------------------------------

#: The numbers the defective report carried, kept verbatim as the teeth-test.
DEFECT_BACKBONE_PARAMETERS = 85_798_656
VIT_B16_HEAD_PARAMETERS = 769  # 768 embedding weights + 1 bias


def test_the_run_reports_what_it_actually_trains(built):
    result, _, _ = built
    parameters = result.summary["parameters"]
    assert set(parameters) >= {
        "policy", "total_parameters", "trainable_parameters",
        "trainable_fraction", "components",
    }
    assert parameters["policy"] == result.summary["train_config"]["trainable"]


def test_a_frozen_backbone_arm_reports_the_head_not_the_backbone():
    """[MEASURED defect, 2026-07-28] metrics.json said trainable_parameters
    85,798,656 with trainable_fraction 1.0 while train_config said
    trainable: head. The head is 769 parameters."""
    report = P3.parameter_summary(
        "head",
        {
            "features": P3.WHOLE_IMAGE_EMBEDDINGS,
            "backbone_parameters": DEFECT_BACKBONE_PARAMETERS,
            "backbone_trainable_parameters": 0,
        },
        {
            "total_parameters": VIT_B16_HEAD_PARAMETERS,
            "trainable_parameters": VIT_B16_HEAD_PARAMETERS,
        },
        backbone="vit_b16",
    )

    assert report["trainable_parameters"] == VIT_B16_HEAD_PARAMETERS
    assert report["total_parameters"] == (
        DEFECT_BACKBONE_PARAMETERS + VIT_B16_HEAD_PARAMETERS
    )
    assert report["trainable_fraction"] < 1e-4
    assert report["components"]["frozen_feature_extractor"]["trainable"] == 0


def test_the_defective_report_is_refused():
    """The exact shape of the defect: a head arm on a real backbone whose whole
    parameter count is reported as its trainable set, with nothing frozen."""
    with pytest.raises(P3.Phase3Error, match="2026-07-28 defect"):
        P3.parameter_summary(
            "head",
            {"features": P3.WHOLE_IMAGE_EMBEDDINGS},
            {
                "total_parameters": DEFECT_BACKBONE_PARAMETERS,
                "trainable_parameters": DEFECT_BACKBONE_PARAMETERS,
            },
            backbone="vit_b16",
        )


def test_a_backbone_that_trains_under_a_head_policy_is_refused():
    with pytest.raises(P3.Phase3Error, match="trains nothing in the backbone"):
        P3.parameter_summary(
            "head",
            {
                "backbone_parameters": DEFECT_BACKBONE_PARAMETERS,
                "backbone_trainable_parameters": DEFECT_BACKBONE_PARAMETERS,
            },
            {"total_parameters": 769, "trainable_parameters": 769},
            backbone="vit_b16",
        )


def test_a_patch_arm_is_not_exempt_from_the_parameter_check():
    """The check keys off a SET of feature kinds. Matching one literal would
    have let every Phase 4 patch arm skip the check that exists because this
    reporting was wrong once already."""
    assert P3.PATCH_EMBEDDINGS in P3.FROZEN_EMBEDDINGS
    assert P3.WHOLE_IMAGE_EMBEDDINGS in P3.FROZEN_EMBEDDINGS

    with pytest.raises(P3.Phase3Error, match="2026-07-28 defect"):
        P3.parameter_summary(
            "head",
            {"features": P3.PATCH_EMBEDDINGS},
            {
                "total_parameters": DEFECT_BACKBONE_PARAMETERS,
                "trainable_parameters": DEFECT_BACKBONE_PARAMETERS,
            },
            backbone="vit_b16",
        )


def test_the_stub_is_allowed_to_be_its_own_whole_model():
    """The stub has no backbone to freeze, so trainable_fraction 1.0 is honest
    for it. The check must not fire on the laptop path it does not describe."""
    report = P3.parameter_summary(
        "head", {}, {"total_parameters": 65, "trainable_parameters": 65}
    )
    assert report["trainable_fraction"] == 1.0
    assert report["backbone"] == "stub"


def test_an_arm_with_no_backbone_at_all_is_allowed_to_be_its_own_whole_model():
    """The mirror-difference arm has no extractor to account for -- its features
    come from geometry, not from a network. The check keys on the FEATURES, so a
    named backbone that consumes no frozen embeddings is not the defect."""
    report = P3.parameter_summary(
        "head",
        {"features": "mirror_difference_index"},
        {"total_parameters": 23, "trainable_parameters": 23},
        backbone="ridge",
    )
    assert report["trainable_fraction"] == 1.0
    assert report["features"] == "mirror_difference_index"


def test_a_full_policy_that_does_not_train_everything_is_refused():
    with pytest.raises(P3.Phase3Error, match="policy is 'full'"):
        P3.parameter_summary(
            "full",
            {"backbone_parameters": 1000, "backbone_trainable_parameters": 0},
            {"total_parameters": 769, "trainable_parameters": 769},
            backbone="vit_b16",
        )


def test_a_full_arm_reports_every_parameter_trainable():
    report = P3.parameter_summary(
        "full",
        {},
        {
            "total_parameters": DEFECT_BACKBONE_PARAMETERS,
            "trainable_parameters": DEFECT_BACKBONE_PARAMETERS,
        },
    )
    assert report["trainable_fraction"] == 1.0


def test_the_frozen_extraction_report_does_not_claim_a_trainable_set():
    """The key that misled is gone from the feature report by name.

    ``features.trainable_parameters`` was read as the arm's trainable count. No
    key there may look like one again -- the arm-level numbers live under
    ``parameters`` and nowhere else.
    """
    import inspect

    from cleft.train import torch_backbone

    source = inspect.getsource(torch_backbone.extract_embeddings)
    assert "**count_parameters(model)" not in source
    assert '"backbone_parameters"' in source


# --------------------------------------------------------------------------
# the training record -- the provenance gap
# --------------------------------------------------------------------------


def test_train_config_records_everything_that_determines_the_number(built):
    """The first keeper run could not say what learning rate produced its result."""
    result, _, _ = built
    recorded = result.summary["train_config"]
    for key in (
        "learning_rate", "optimizer", "weight_decay", "trainable",
        "batch_size", "loss", "label_scale", "seed",
        "max_epochs", "patience", "inner_val_frac", "monitor",
    ):
        assert key in recorded, f"train_config does not record {key}"


def test_the_trainable_policy_is_recorded_not_implied(built):
    result, _, _ = built
    assert result.summary["train_config"]["trainable"] in ("head", "full")


def test_the_feature_representation_is_recorded(built):
    """Raw images or frozen embeddings changes what the number means."""
    result, _, _ = built
    assert "features" in result.summary["features"]


# --------------------------------------------------------------------------
# gate 2 -- seed variance
# --------------------------------------------------------------------------


def test_seed_variance_reports_a_distribution_not_a_number():
    report = P3.seed_variance([0.30, 0.34, 0.28, 0.36, 0.31])
    for key in ("mean", "sd", "min", "max", "range", "interval_95", "n_seeds"):
        assert key in report
    assert report["n_seeds"] == 5
    assert report["range"] == pytest.approx(0.08, abs=1e-9)


def test_seed_variance_states_the_claimable_floor():
    """The output is a RULE: no delta smaller than the band is claimable."""
    report = P3.seed_variance([0.30, 0.34, 0.28, 0.36, 0.31])
    assert report["claimable_delta_floor"] == report["range"]
    assert "claimable" in report["note"]


def test_seed_variance_needs_more_than_one_seed():
    with pytest.raises(P3.Phase3Error, match="at least two seeds"):
        P3.seed_variance([0.3])


# --------------------------------------------------------------------------
# gate 2's result -- the band every Phase 7 claim is measured against
# --------------------------------------------------------------------------


def test_the_measured_seed_band_is_recorded():
    """[MEASURED 2026-07-28] Ten seeds, frozen linear probe, one job.

    Asserted here because the tests are the record: a number that matters lives
    in one and is reproducible from the run directory, not only in prose.
    """
    band = P3.MEASURED_SEED_BAND
    assert band["n_seeds"] == 10
    assert band["mean"] == 0.2529
    assert band["sd"] == 0.0137
    assert (band["min"], band["min_seed"]) == (0.2347, 7)
    assert (band["max"], band["max_seed"]) == (0.2719, 1337)

    # min and max are each recorded to 4 dp, so their difference can sit one
    # unit in the last place away from the recorded range -- 0.0372 against
    # 0.0373 here. That is rounding, not a disagreement, and asserting equality
    # would be asserting an artifact. R2: check it is the same quantity before
    # calling it an error.
    assert round(abs((band["max"] - band["min"]) - band["range"]), 4) <= 1e-4


def test_gate_1s_byte_identical_result_is_recorded():
    """[MEASURED 2026-07-28] Three runs of the shipped config on one GPU agreed
    to the last digit. Phase 7 owes a determinism re-run when the patch path
    changes the harness, and this is what it compares against."""
    reference = P3.GATE1_REFERENCE
    assert reference["n_runs"] == 3
    assert reference["seed"] == 1337
    assert reference["pcc"] == 0.2719366015264466
    assert reference["mae"] == 0.5309369017806235
    assert reference["rmse"] == 0.6515587916820617

    # The same seed's entry in the sweep, to 4 dp. Same quantity, so they agree.
    assert round(reference["pcc"], 4) == P3.MEASURED_SEED_BAND["max"]


def test_the_band_names_the_quantity_it_measures():
    """[CORRECTED 2026-07-28] "seed SD" alone does not identify a quantity.

    The gate-2 band is initialisation AND inner-val split variance combined --
    the embeddings are extracted once so the features contribute nothing, but
    ``harness.inner_val_split`` seeds off ``config.seed``, which every arm
    inherits. The Phase 4 ridge arm reports a seed SD too, and that one is split
    variance alone. Comparing them as though they were the same number is the
    R2 error, so the constant names itself.
    """
    assert (
        P3.MEASURED_SEED_BAND["measures"]
        == "head_initialisation_and_inner_val_split"
    )


def test_the_seed_reaches_the_inner_val_split():
    """The mechanism behind the correction above, asserted rather than described.

    If this ever stopped being true, the band would quietly become a different
    quantity and nothing else in the suite would notice.
    """
    from cleft.train.harness import inner_val_split

    train_ids = list(range(1, 41))
    one, _ = inner_val_split(train_ids, 0.2, seed=1337, fold=0)
    other, _ = inner_val_split(train_ids, 0.2, seed=7, fold=0)
    assert one != other, "the seed must change which patients are held out"


def test_the_combined_threshold_names_both_quantities():
    """[MEASURED 2026-07-28] The mirror index against the probe: 0.019 on arm
    means, 0.046 on single runs. The phrase "combined seed uncertainty" admits
    both readings and they differ by ~2.5x, so the function returns both and says
    which is the criterion."""
    combined = P3.combined_claimable_delta(0.0137, 10, 0.019, 5)
    assert combined["primary"] == "arm_means_95"
    assert round(combined["arm_means_95"], 3) == 0.019
    assert round(combined["single_run_95"], 3) == 0.046
    assert combined["single_run_95"] > combined["arm_means_95"]


def test_the_mirror_delta_clears_both_readings():
    """-0.095 against the probe, so the conclusion does not depend on which
    reading is taken. It will not always be so."""
    combined = P3.combined_claimable_delta(0.0137, 10, 0.019, 5)
    assert 0.095 > combined["single_run_95"] > combined["arm_means_95"]


def test_a_zero_seed_count_is_refused():
    with pytest.raises(P3.Phase3Error, match="must be positive"):
        P3.combined_claimable_delta(0.01, 0, 0.01, 5)


# --------------------------------------------------------------------------
# determinism, re-checked on the Phase 4 patch path
# --------------------------------------------------------------------------


def test_the_fingerprint_is_stable_for_identical_predictions():
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    predictions = truth + 0.1
    assert P3.determinism_fingerprint(
        cv_of(predictions, truth)
    ) == P3.determinism_fingerprint(cv_of(predictions.copy(), truth.copy()))


def test_the_fingerprint_moves_on_the_smallest_change():
    """Pooling is where a nondeterministic reduction hides, and a reordered sum
    over 27 float32 embeddings moves only the last digits. The digest must catch
    that, not round it away."""
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    base = truth + 0.1
    nudged = base.copy()
    nudged[2] += 1e-6

    assert (
        P3.predictions_digest(cv_of(base, truth))
        != P3.predictions_digest(cv_of(nudged, truth))
    )


def test_the_fingerprint_depends_on_patient_order():
    """Two runs that predicted the same values for different patients are not
    the same run."""
    from cleft.train.harness import CVResult

    truth = np.array([1.0, 2.0, 3.0])
    forward = CVResult(
        folds=[], oof_ids=[1, 2, 3],
        oof_predictions=np.array([1.1, 2.1, 3.1]), oof_truth=truth,
    )
    swapped = CVResult(
        folds=[], oof_ids=[2, 1, 3],
        oof_predictions=np.array([1.1, 2.1, 3.1]), oof_truth=truth,
    )
    assert P3.predictions_digest(forward) != P3.predictions_digest(swapped)


def test_the_run_carries_a_fingerprint(built):
    result, _, _ = built
    fingerprint = result.summary["fingerprint"]
    assert set(fingerprint) == {
        "predictions_sha256", "pcc", "mae", "rmse", "selected_epochs", "n",
    }
    assert len(fingerprint["predictions_sha256"]) == 64


def test_gate1_reproduction_is_exact_not_tolerant():
    """Gate 1's claim is byte-identity. A tolerance would quietly weaken it."""
    reference = P3.GATE1_REFERENCE
    exact = {
        "pcc": reference["pcc"],
        "mae": reference["mae"],
        "rmse": reference["rmse"],
    }
    assert P3.compare_to_gate1(exact)["matches"] is True

    nudged = {**exact, "pcc": reference["pcc"] + 1e-12}
    comparison = P3.compare_to_gate1(nudged)
    assert comparison["matches"] is False
    assert "pcc" in comparison["differences"]


def test_the_claimable_delta_at_one_five_and_ten_seeds():
    """1.96 * sd * sqrt(2/n). The one-seed figure is essentially the whole
    observed range, which is why a single seed resolves nothing."""
    assert round(P3.claimable_delta(1), 3) == 0.038
    assert round(P3.claimable_delta(5), 3) == 0.017
    assert round(P3.claimable_delta(10), 3) == 0.012


def test_one_seed_resolves_less_than_the_observed_spread():
    band = P3.MEASURED_SEED_BAND
    assert P3.claimable_delta(1) > band["range"] * 0.9


def test_more_seeds_resolve_smaller_deltas():
    deltas = [P3.claimable_delta(n) for n in (1, 5, 10, 20)]
    assert deltas == sorted(deltas, reverse=True)


def test_the_band_is_derived_from_the_sweep_not_from_the_constant():
    """A re-measured arm must report its own band. Reusing this one where the
    training procedure differs is the QWK-as-ceiling mistake in a new place."""
    tight = P3.seed_variance([0.50, 0.50, 0.501, 0.499, 0.5005])
    assert tight["claimable_delta_at_n_seeds"]["10"] < P3.claimable_delta(10)
    assert "needs its own" in tight["note"]


def test_seed_variance_reports_the_band_at_every_reported_count():
    reported = P3.seed_variance([0.21, 0.25, 0.27])["claimable_delta_at_n_seeds"]
    assert sorted(reported) == sorted(
        str(n) for n in P3.REPORTED_SEED_COUNTS
    )


def test_a_seed_sweep_writes_one_output_set_per_seed(
    tmp_path, clean_repo, monkeypatch
):
    """One job, one env.json, per-seed files -- and no seed may clobber another."""
    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    manifest_dir, staged_dir = artifacts(tmp_path / "art")

    from cleft.provenance import hash_dir

    config = builders.write_config(
        tmp_path / "sweep.yaml",
        tier="dev", phase="p3",
        inputs=[
            {"name": "m", "path": str(manifest_dir),
             "rollup_sha256": hash_dir(manifest_dir)["rollup"]},
            {"name": "s", "path": str(staged_dir),
             "rollup_sha256": hash_dir(staged_dir)["rollup"]},
        ],
        task={
            "kind": "train_cv", "manifest_artifact": "m", "staged_artifact": "s",
            "geometry": "g1", "label": "mean", "backbone": "stub",
            "seeds": [1, 2, 3],
            "max_epochs": 4, "patience": 2, "inner_val_frac": 0.2,
            "monitor": "inner_val_mse",
        },
    )
    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])

    outputs = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    names = {entry["name"] for entry in outputs["outputs"]}
    for seed in (1, 2, 3):
        assert f"seed_{seed}__predictions.csv" in names
        assert f"seed_{seed}__metrics.json" in names
    assert "seed_variance.json" in names

    variance = json.loads((run_dir / "seed_variance.json").read_text(encoding="utf-8"))
    assert variance["n_seeds"] == 3

    # The gate-1 verdict must SURVIVE TO DISK, at every seed count.
    assert "gate1_reproduction.json" in names
    verdict = json.loads(
        (run_dir / "gate1_reproduction.json").read_text(encoding="utf-8")
    )
    assert verdict["checked"] is False, "the stub arm is not the reference arm"
    assert verdict["why"]


def test_the_gate1_verdict_reaches_a_file_at_every_seed_count(
    tmp_path, clean_repo, monkeypatch
):
    """**[DEFECT 2026-08-01] It did not.** `summary["gate1_reproduction"]` was
    assigned after `write_outputs` had already written metrics.json: at one
    seed the mutation landed on an object whose file was already on disk, and
    at several seeds the dict was never written by anything. So the verdict
    existed only in stdout and log.txt.

    That is the project's own failure shape in the worst place -- the
    determinism re-verification is closed by reading this verdict, and the
    verdict was not being kept. One seed is the case that looks most like it
    works, because the key really is present in the in-memory summary, so it
    is the one asserted here.
    """
    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    manifest_dir, staged_dir = artifacts(tmp_path / "art_single")

    from cleft.provenance import hash_dir

    config = builders.write_config(
        tmp_path / "single.yaml",
        tier="dev", phase="p3",
        inputs=[
            {"name": "m", "path": str(manifest_dir),
             "rollup_sha256": hash_dir(manifest_dir)["rollup"]},
            {"name": "s", "path": str(staged_dir),
             "rollup_sha256": hash_dir(staged_dir)["rollup"]},
        ],
        task={
            "kind": "train_cv", "manifest_artifact": "m", "staged_artifact": "s",
            "geometry": "g1", "label": "mean", "backbone": "stub",
            "max_epochs": 4, "patience": 2, "inner_val_frac": 0.2,
            "monitor": "inner_val_mse",
        },
    )
    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])

    path = run_dir / "gate1_reproduction.json"
    assert path.is_file(), (
        "the gate-1 verdict must be written to a file, not only logged -- "
        "metrics.json is written before the verdict is computed"
    )
    verdict = json.loads(path.read_text(encoding="utf-8"))
    assert verdict["checked"] is False

    # And it is declared as an output, so it is hashed and recorded like every
    # other artifact rather than appearing on disk unaccounted for.
    outputs = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    assert "gate1_reproduction.json" in {e["name"] for e in outputs["outputs"]}


def test_a_duplicate_seed_is_rejected(tmp_path, clean_repo, monkeypatch):
    """Two runs of the same seed would overwrite each other's outputs."""
    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    manifest_dir, staged_dir = artifacts(tmp_path / "art2")

    from cleft.provenance import hash_dir

    config = builders.write_config(
        tmp_path / "dupe.yaml",
        tier="dev", phase="p3",
        inputs=[
            {"name": "m", "path": str(manifest_dir),
             "rollup_sha256": hash_dir(manifest_dir)["rollup"]},
            {"name": "s", "path": str(staged_dir),
             "rollup_sha256": hash_dir(staged_dir)["rollup"]},
        ],
        task={
            "kind": "train_cv", "manifest_artifact": "m", "staged_artifact": "s",
            "geometry": "g1", "label": "mean", "backbone": "stub",
            "seeds": [7, 7], "max_epochs": 3, "patience": 2,
            "inner_val_frac": 0.2, "monitor": "inner_val_mse",
        },
    )
    with pytest.raises(ValueError, match="duplicate seed"):
        main(["--config", str(config), "--out", str(tmp_path / "runs")])


# --------------------------------------------------------------------------
# freezing
# --------------------------------------------------------------------------


def test_freezing_leaves_only_the_head_trainable():
    from cleft.models import factory

    class FakeParam:
        def __init__(self, n):
            self._n = n
            self.requires_grad = True

        def numel(self):
            return self._n

        def requires_grad_(self, value):
            self.requires_grad = value

    head_param = FakeParam(769)
    body_param = FakeParam(86_000_000)

    class FakeHead:
        def parameters(self):
            return [head_param]

    class FakeModel:
        def get_classifier(self):
            return FakeHead()

        def parameters(self):
            return [body_param, head_param]

    report = factory.freeze_backbone(FakeModel())
    assert report["trainable_parameters"] == 769
    assert report["frozen_parameters"] == 86_000_000
    assert body_param.requires_grad is False
    assert head_param.requires_grad is True


def test_freezing_everything_is_refused():
    from cleft.models import factory

    class Empty:
        def get_classifier(self):
            return self

        def parameters(self):
            return []

    with pytest.raises(factory.FrozenBackboneError, match="no trainable"):
        factory.freeze_backbone(Empty())


# --------------------------------------------------------------------------
# determinism configuration
# --------------------------------------------------------------------------


def test_configure_seeds_without_torch():
    record = D.configure(99)
    assert record["seed"] == 99
    assert record["python_random"] and record["numpy_legacy_global"]


def test_configure_demands_torch_when_it_matters():
    with pytest.raises(D.DeterminismError, match="torch is required"):
        D.configure(1, require_torch=True)


def test_the_cublas_variable_is_the_documented_one():
    """It must be set before the CUDA context exists, so it is also in the image."""
    assert D.CUBLAS_WORKSPACE_CONFIG == ":4096:8"


# --------------------------------------------------------------------------
# the factory guarantee
# --------------------------------------------------------------------------


def test_the_torch_backbone_module_imports_without_torch():
    """The whole point of the injected factory."""
    from cleft.train import torch_backbone

    assert torch_backbone.TorchBackbone is not None


def test_the_factory_never_hardcodes_normalization():
    """The Stage-1 defect: constants that are silently wrong for some backbones.

    Even the allowlisted file must READ them from the model, not write them.
    """
    import inspect

    from cleft.models import factory

    source = inspect.getsource(factory)
    for literal in ("0.485", "0.456", "0.406", "0.229", "0.224", "0.225"):
        assert literal not in source, (
            f"{literal} is hardcoded in the factory; normalization must come from "
            "the model's own pretrained_cfg"
        )


def test_normalization_reads_the_models_own_config():
    from cleft.models import factory

    class FakeModel:
        pretrained_cfg = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}

    assert factory.normalization_for(FakeModel()) == ((0.5,) * 3, (0.5,) * 3)


def test_a_model_without_a_config_is_refused_not_defaulted():
    """Matched case-insensitively: the behaviour under test is the REFUSAL, not
    the message's typography. This asserted the exact phrase and broke when the
    warning was capitalised for emphasis in 2026-07-31's normalization work --
    a test failing for a reason unrelated to its claim.
    """
    from cleft.models import factory

    with pytest.raises(factory.FactoryError, match="(?i)do not substitute a default"):
        factory.normalization_for(object())


# --------------------------------------------------------------------------
# as a task
# --------------------------------------------------------------------------


def test_the_task_writes_outputs_at_the_right_tiers(tmp_path, clean_repo, monkeypatch):
    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    manifest_dir, staged_dir = artifacts(tmp_path / "art")

    from cleft.provenance import hash_dir

    config = builders.write_config(
        tmp_path / "p3.yaml",
        tier="dev",
        phase="p3",
        inputs=[
            {"name": "m", "path": str(manifest_dir),
             "rollup_sha256": hash_dir(manifest_dir)["rollup"]},
            {"name": "s", "path": str(staged_dir),
             "rollup_sha256": hash_dir(staged_dir)["rollup"]},
        ],
        task={
            "kind": "train_cv",
            "manifest_artifact": "m",
            "staged_artifact": "s",
            "geometry": "g1",
            "label": "mean",
            "backbone": "stub",
            "max_epochs": 6,
            "patience": 2,
            "inner_val_frac": 0.2,
            "monitor": "inner_val_mse",
        },
    )

    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])
    outputs = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    tiers = {entry["name"]: entry["tier"] for entry in outputs["outputs"]}

    assert tiers["metrics.json"] == "SHAREABLE"
    assert tiers["curves.csv"] == "SHAREABLE"
    assert tiers["predictions.csv"] == "CLUSTER-ONLY"
    assert tiers["fold_record.json"] == "CLUSTER-ONLY"


def test_predictions_carry_one_row_per_patient(tmp_path, clean_repo, monkeypatch):
    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    manifest_dir, staged_dir = artifacts(tmp_path / "art")

    from cleft.provenance import hash_dir

    config = builders.write_config(
        tmp_path / "p3.yaml",
        tier="dev", phase="p3",
        inputs=[
            {"name": "m", "path": str(manifest_dir),
             "rollup_sha256": hash_dir(manifest_dir)["rollup"]},
            {"name": "s", "path": str(staged_dir),
             "rollup_sha256": hash_dir(staged_dir)["rollup"]},
        ],
        task={
            "kind": "train_cv", "manifest_artifact": "m", "staged_artifact": "s",
            "geometry": "g1", "label": "mean", "backbone": "stub",
            "max_epochs": 4, "patience": 2, "inner_val_frac": 0.2,
            "monitor": "inner_val_mse",
        },
    )
    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])

    lines = (run_dir / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# CLUSTER-ONLY")
    rows = list(csv.DictReader(lines[1:]))
    assert len(rows) == N
    assert len({r["patient_id"] for r in rows}) == N


def test_the_phase_4_arm_configs_are_valid(repo_root):
    """Every shipped arm loads, and the arms that should differ in one field do."""
    from cleft.config import load_config

    arms = {
        name: load_config(repo_root / "configs" / f"{name}.yaml")["task"]
        for name in (
            "p4_patch_grid",
            "p4_patch_anatomy",
            "p4_patch_random",
            "p4_geometry_random_g1",
        )
    }

    # The three scheme arms differ in patch_scheme and nothing else.
    schemes = {n: a for n, a in arms.items() if n.startswith("p4_patch_")}
    assert {a["patch_scheme"] for a in schemes.values()} == {
        "grid", "anatomy", "random",
    }
    for arm in schemes.values():
        assert arm["geometry"] == "g2", "the scheme comparison is only clean at G2"
    for field in ("label", "backbone", "trainable", "pooling", "seeds", "max_epochs"):
        assert len({str(a[field]) for a in schemes.values()}) == 1, (
            f"the scheme arms disagree on {field}, so they differ in more than "
            "the one factor they are supposed to"
        )

    # The geometry probe differs from the G2 random arm in geometry alone.
    probe, g2_random = arms["p4_geometry_random_g1"], arms["p4_patch_random"]
    assert probe["geometry"] == "g1" and g2_random["geometry"] == "g2"
    assert probe["patch_scheme"] == g2_random["patch_scheme"] == "random"
    differing = {k for k in probe if str(probe[k]) != str(g2_random.get(k))}
    assert differing == {"geometry"}, (
        f"the geometry probe differs in {differing}, not geometry alone"
    )


def test_every_phase_4_arm_runs_five_seeds(repo_root):
    """§1: a single-seed baseline cannot be compared with anything."""
    from cleft.config import load_config

    for name in (
        "p4_patch_grid", "p4_patch_anatomy", "p4_patch_random",
        "p4_geometry_random_g1",
    ):
        task = load_config(repo_root / "configs" / f"{name}.yaml")["task"]
        assert len(task["seeds"]) >= 5, name


def test_the_shipped_config_is_valid(repo_root):
    from cleft.config import load_config

    cfg = load_config(repo_root / "configs" / "p3_train_cv.yaml")
    task = cfg["task"]
    assert cfg["tier"] == "keeper", "the freeze cites this run"
    assert task["backbone"] == "vit_b16"
    assert task["geometry"] == "g1"
    assert task["label"] == "mean", "the PRIMARY target per §4.2"
    assert task["monitor"] == "inner_val_mse"


def test_the_concat_source_verifies_each_constituent_and_sums_the_width(tmp_path):
    """**The concat arm cannot smuggle an unverified set past the checks**:
    every constituent goes through load_features -- same row order, same
    pairing against its OWN backbone -- and the head-width check holds over
    the summed dimension."""
    import numpy as np

    from cleft import embeddings
    from cleft.train import pooled

    ids = [1, 2, 3, 4]
    rng = np.random.default_rng(0)
    sources = []
    for backbone, dim in (("vit_b8", 3), ("vit_b16", 5), ("vit_b32", 2)):
        directory = tmp_path / f"{backbone}__imagenet__g1"
        embeddings.save(
            directory, rng.normal(size=(4, dim)).astype(np.float32),
            backbone=backbone, backbone_kind="transformer", init="imagenet",
            geometry="g1", variant=None, checkpoint_sha256=None,
            patient_ids=ids, manifest_ids=ids,
        )
        sources.append(pooled.PooledSource(
            directory=directory, backbone=backbone, init="imagenet",
            geometry="g1",
        ))

    features, report = pooled.load_concat_features(sources, ids)
    assert features.shape == (4, 10)
    assert report["feature_dim"] == 10
    assert [c["backbone"] for c in report["constituents"]] == [
        "vit_b8", "vit_b16", "vit_b32"
    ]
    assert "+0.0221" in report["measured_prior"]
    # The summed width is what the head check compares against.
    pooled.assert_head_matches_embedding_width(10, 11)

    # One constituent declared under the wrong backbone refuses -- the
    # pairing check runs PER SET, not on the concatenation.
    wrong = [sources[0], pooled.PooledSource(
        directory=sources[1].directory, backbone="vit_b32",
        init="imagenet", geometry="g1",
    )]
    with pytest.raises(pooled.PooledFeatureError, match="backbone"):
        pooled.load_concat_features(wrong, ids)
    # And one set concatenated is not a concat arm.
    with pytest.raises(pooled.PooledFeatureError, match="at least two"):
        pooled.load_concat_features(sources[:1], ids)
