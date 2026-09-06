"""The embedding plan and the extraction task (Phase 6 §4).

The plan derives the required sets from the arm list; the shipped config must
enumerate exactly those; and the task must write artifacts whose row order,
kind, and checkpoint provenance are asserted rather than assumed. The stub
backbones run the whole path without torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cleft import embedding_plan as plan
from cleft import embeddings as emb
from cleft.train.extract import (
    ExtractError,
    backbone_kind_for,
    checkpoint_input_name,
    extract_features,
)

from fixtures import builders


# --------------------------------------------------------------------------
# the plan: derived, not cross-producted
# --------------------------------------------------------------------------


def test_the_plan_derives_seventeen_sets_from_thirteen_checkpoints():
    """[Instruction 2026-07-31; Stage C widened 2026-08-01] The arm list
    needs 17 sets against a naive product of ~48: feature maps are
    per-checkpoint, schemes apply at cleft-train time, and only Stage E's
    scheme arms multiply -- by scheme-matched checkpoints, per the
    consistency rule."""
    sets = plan.required_sets()
    assert len(sets) == 17
    checkpoints = plan.required_checkpoints()
    assert len(checkpoints) == 13
    summary = plan.summary()
    assert summary["required_sets"] < summary["naive_cross_product"]


def test_stage_c_is_a_mechanism_test_on_both_transformers():
    """[DECIDED 2026-08-01] Choosing ViT alone because pretraining showed the
    largest gap would select on the outcome. The tiling mechanism is a PRIOR
    and it PREDICTS -- stated before the cleft runs -- that ViT separates at
    G1 and Swin does not (Swin's pretraining indifference: 0.8718 vs 0.8719).
    Both transformers therefore run both geometries on the masked init."""
    assert plan.GEOMETRY_STAGE_BACKBONES == ("vit_b16", "swin_b")
    stage_c = {
        (entry["backbone"], entry["geometry"])
        for entry in plan.required_sets()
        if entry["init"] == "scut_masked" and entry["backbone_kind"] == "transformer"
    }
    assert stage_c == {
        ("vit_b16", "g1"), ("vit_b16", "g2"),
        ("swin_b", "g1"), ("swin_b", "g2"),
    }

    # The prediction is recorded in the plan, dated, ahead of the runs --
    # stated afterwards it would be indistinguishable from a post-hoc story.
    source = Path(plan.__file__).read_text(encoding="utf-8")
    assert "ViT separates, Swin does not" in source
    assert "0.8718" in source
    # And the confirmed placements carry the supervision-thirds note.
    assert "Stage E" in source and "thirds grid" in source


def test_every_set_respects_the_crossing_rule():
    """A masked init is geometry-bound; the derivation must never emit a
    crossed pair. Asserted through the artifact format's own check."""
    for entry in plan.required_sets():
        emb.check_init_geometry(entry["init"], entry["geometry"], entry["variant"])


def test_schemes_appear_exactly_where_they_can_act():
    """pretrain_scheme on graph sets with a checkpoint; None for imagenet
    (no pretraining to have a scheme) and for transformers (no scheme axis).
    Stage E's three generated schemes are present for SR-GNN at G2."""
    for entry in plan.required_sets():
        has_scheme = entry["pretrain_scheme"] is not None
        is_graph = entry["backbone_kind"] == "graph"
        has_checkpoint = entry["variant"] is not None
        assert has_scheme == (is_graph and has_checkpoint), entry

    stage_e = {
        entry["pretrain_scheme"]
        for entry in plan.required_sets()
        if entry["backbone"] == "srgnn"
        and entry["init"] == "scut_masked"
        and entry["geometry"] == "g2"
    }
    assert stage_e == {"native", "grid", "anatomy", "random"}


def test_every_required_checkpoint_is_producible_by_a_shipped_config(repo_root):
    """The plan consumes only runs the config lattice can produce -- the
    reachability rule, applied to checkpoints."""
    for checkpoint in plan.required_checkpoints():
        assert (repo_root / "configs" / checkpoint["config"]).is_file(), (
            f"{checkpoint['config']} does not exist; the plan requires a "
            "checkpoint nothing can produce"
        )


def test_the_shipped_extraction_config_equals_the_derivation(
    repo_root, monkeypatch
):
    """The config is the provenance record, so it enumerates the sets -- and
    this test is what keeps it equal to the derivation. An arm-list change
    lands as a failing test here, never as a silently missing set."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    cfg = load_config(repo_root / "configs" / "p6_extract_embeddings.yaml")

    declared = [
        {
            "backbone": entry["backbone"],
            "init": entry["init"],
            "geometry": entry["geometry"],
            "pretrain_scheme": entry["pretrain_scheme"],
        }
        for entry in cfg["task"]["sets"]
    ]
    derived = [
        {
            "backbone": entry["backbone"],
            "init": entry["init"],
            "geometry": entry["geometry"],
            "pretrain_scheme": entry["pretrain_scheme"],
        }
        for entry in plan.required_sets()
    ]
    assert declared == derived, "the shipped set list drifted from the plan"

    # Every non-imagenet set's checkpoint input is declared, by its
    # conventional name; nothing else is.
    input_names = {entry["name"] for entry in cfg["inputs"]}
    needed = {
        checkpoint_input_name(entry)
        for entry in plan.required_sets()
        if entry["variant"] is not None
    }
    assert needed <= input_names, sorted(needed - input_names)
    assert input_names - needed == {"manifest_v1", "staged_v1"}


# --------------------------------------------------------------------------
# extraction mechanics, torch-free
# --------------------------------------------------------------------------


def test_stub_extractors_produce_the_right_kinds():
    images = np.arange(2 * 8 * 8 * 3, dtype=np.uint8).reshape(2, 8, 8, 3)
    pooled, info = extract_features("stub", "imagenet", images, checkpoint_path=None)
    assert pooled.shape == (2, 3) and info["backbone_kind"] == "transformer"

    maps, info = extract_features(
        "stub_graph", "imagenet", images, checkpoint_path=None
    )
    assert maps.shape == (2, 3, 2, 2) and info["backbone_kind"] == "graph"

    # Content-dependent: different images, different features (row-order teeth).
    assert not np.allclose(pooled[0], pooled[1])
    with pytest.raises(ExtractError, match="unknown backbone"):
        backbone_kind_for("bogus")


def test_imagenet_refuses_a_supplied_checkpoint(tmp_path):
    """Silently ignoring one would misdescribe the set's provenance."""
    from cleft.train import extract as extract_module

    with pytest.raises(ExtractError, match="takes no checkpoint"):
        extract_module._load_model("vit_b16", "imagenet", tmp_path / "x.npz")


# --------------------------------------------------------------------------
# the task, end to end through main()
# --------------------------------------------------------------------------


def make_artifacts(root: Path, n: int = 10):
    """Manifest + staged tensors with matching row order (phase3's shape)."""
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
        value = truth[index]
        lines.append(
            f"{index + 1},{1000 + index},,{value:.4f},{value:.4f},{value:.4f},"
            f"{value:.4f},{value:.4f},0.2,0.2,0.2,0.2,0.2,{index % 3},{index % 5}"
        )
    (manifest_dir / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    images = np.zeros((n, 8, 8, 3), dtype=np.uint8)
    for index in range(n):
        images[index] = int(np.clip(truth[index] * 40, 0, 255))
    np.save(staged_dir / "staged_patient_g1.npy", images)
    np.save(staged_dir / "staged_patient_g2.npy", images.copy())

    geometry = ["# CLUSTER-ONLY: patient-keyed geometry", "patient_id,aspect_ratio"]
    geometry += [f"{pid},0.74" for pid in range(1, n + 1)]
    (staged_dir / "geometry.csv").write_text("\n".join(geometry) + "\n", encoding="utf-8")
    return manifest_dir, staged_dir


def declare(name: str, path: Path) -> dict:
    from cleft.provenance import hash_path

    return {
        "name": name,
        "path": str(path),
        "rollup_sha256": hash_path(path)["rollup"],
    }


def test_the_extraction_task_writes_asserted_artifacts(tmp_path, clean_repo):
    from cleft.run import main
    from cleft.train import checkpoint as ckpt_module

    manifest_dir, staged_dir = make_artifacts(tmp_path)

    # A checkpoint FILE input for the masked stub set: the stub extractor
    # never loads it, but the task must still hash-verify and record it.
    checkpoint_file = tmp_path / "pretrained.npz"
    ckpt_module.save(
        checkpoint_file,
        ckpt_module.Checkpoint(
            step=1, epoch=1, arrays={"model__w": np.zeros(3)},
            numpy_rng=ckpt_module.capture_numpy_rng(np.random.default_rng(0)),
        ),
    )

    config = builders.write_config(
        tmp_path / "extract.yaml",
        phase="p6",
        inputs=[
            declare("manifest_v1", manifest_dir),
            declare("staged_v1", staged_dir),
            declare("ckpt_stub_graph_native_masked_g1", checkpoint_file),
        ],
        task={
            "kind": "extract_embeddings",
            "manifest_artifact": "manifest_v1",
            "staged_artifact": "staged_v1",
            "out_version": "embeddings_test_v1",
            "batch_size": 4,
            "sets": [
                {"backbone": "stub", "init": "imagenet", "geometry": "g1"},
                {
                    "backbone": "stub_graph", "init": "scut_masked",
                    "geometry": "g1", "pretrain_scheme": "native",
                },
            ],
        },
    )

    def run_once(out):
        return main(["--config", str(config), "--out", str(tmp_path / out)])

    import contextlib
    import io

    import cleft.provenance.context as context_module

    with contextlib.redirect_stdout(io.StringIO()):
        original = context_module.default_repo_root
        context_module.default_repo_root = lambda: clean_repo
        try:
            run_dir = run_once("runs")
        finally:
            context_module.default_repo_root = original

    out_root = clean_repo / "data" / "embeddings" / "embeddings_test_v1"
    pooled_dir = out_root / "stub__imagenet__g1"
    map_dir = out_root / "stub_graph__scut_masked__g1__native"
    assert pooled_dir.is_dir() and map_dir.is_dir()

    manifest_ids = list(range(1, 11))
    values, meta = emb.load(pooled_dir, manifest_ids=manifest_ids)
    assert meta["kind"] == "pooled" and values.shape[0] == 10
    assert meta["pretrain_scheme"] is None
    assert meta["row_order_checked_against_manifest"] is True

    values, meta = emb.load(map_dir, manifest_ids=manifest_ids)
    assert meta["kind"] == "feature_map" and values.shape == (10, 3, 2, 2)
    assert meta["pretrain_scheme"] == "native"
    assert meta["variant"] == "masked_g1"
    from cleft.provenance import hash_path

    assert meta["checkpoint_sha256"] == hash_path(checkpoint_file)["rollup"], (
        "the recorded checkpoint hash must be the declared input's"
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_sets"] == 2
    assert metrics["artifact"]["rollup_sha256_for_configs"]

    # Immutability: a second run into the same version refuses.
    with pytest.raises(ValueError, match="immutable"):
        with contextlib.redirect_stdout(io.StringIO()):
            context_module.default_repo_root = lambda: clean_repo
            try:
                run_once("runs2")
            finally:
                context_module.default_repo_root = original


def test_a_scut_set_without_its_checkpoint_input_is_refused(tmp_path, clean_repo):
    from cleft.run import main

    manifest_dir, staged_dir = make_artifacts(tmp_path)
    config = builders.write_config(
        tmp_path / "extract.yaml",
        phase="p6",
        inputs=[
            declare("manifest_v1", manifest_dir),
            declare("staged_v1", staged_dir),
        ],
        task={
            "kind": "extract_embeddings",
            "manifest_artifact": "manifest_v1",
            "staged_artifact": "staged_v1",
            "out_version": "embeddings_test_v1",
            "batch_size": 4,
            "sets": [
                {
                    "backbone": "stub_graph", "init": "scut_original",
                    "geometry": "g2", "pretrain_scheme": "native",
                },
            ],
        },
    )

    import contextlib
    import io

    import cleft.provenance.context as context_module

    original = context_module.default_repo_root
    context_module.default_repo_root = lambda: clean_repo
    try:
        with pytest.raises(ValueError, match="ckpt_stub_graph_native_original"):
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--config", str(config), "--out", str(tmp_path / "runs")])
    finally:
        context_module.default_repo_root = original

def test_per_fold_reextraction_writes_five_leak_free_sets(tmp_path, clean_repo):
    """**The build side of the leak.** Five sets, each adapted on its own
    fold's TRAINING patients -- and each must record who it adapted on, or the
    consuming arm has nothing to check against.

    The sets must also genuinely DIFFER. If BN re-estimation produced five
    identical arrays the whole per-fold apparatus would be ceremony, and every
    leak assertion downstream would pass on data that never varied -- failure
    mode 5 in PLAN R7's tally, built in by construction.
    """
    import contextlib
    import io

    import cleft.provenance.context as context_module
    from cleft.run import main

    from cleft.data.manifest import load_manifest

    manifest_dir, staged_dir = make_artifacts(tmp_path)
    config = builders.write_config(
        tmp_path / "perfold.yaml",
        phase="p6",
        inputs=[declare("manifest_v1", manifest_dir), declare("staged_v1", staged_dir)],
        task={
            "kind": "extract_embeddings",
            "manifest_artifact": "manifest_v1",
            "staged_artifact": "staged_v1",
            "out_version": "embeddings_perfold_v1",
            "batch_size": 4,
            "per_fold_bn_reestimation": True,
            "sets": [{"backbone": "stub_graph", "init": "imagenet", "geometry": "g2"}],
        },
    )

    with contextlib.redirect_stdout(io.StringIO()):
        original = context_module.default_repo_root
        context_module.default_repo_root = lambda: clean_repo
        try:
            main(["--config", str(config), "--out", str(tmp_path / "runs_pf")])
        finally:
            context_module.default_repo_root = original

    out_root = clean_repo / "data" / "embeddings" / "embeddings_perfold_v1"
    rows = load_manifest(manifest_dir / "manifest.csv")
    ids = [int(r["patient_id"]) for r in rows]
    fold_of = {int(r["patient_id"]): int(r["fold"]) for r in rows}

    arrays = {}
    for fold in sorted(set(fold_of.values())):
        directory = out_root / f"stub_graph__imagenet__g2__adabn_fold{fold}"
        assert directory.is_dir(), f"fold {fold}'s set was not written"
        values, meta = emb.load(directory, manifest_ids=ids)
        arrays[fold] = values

        assert meta["fold"] == fold
        adapted = set(meta["adapted_on_patient_ids"])
        test_ids = {pid for pid in ids if fold_of[pid] == fold}
        # The whole point, stated two ways: every training patient is in, and
        # no test patient is.
        assert adapted == {pid for pid in ids if fold_of[pid] != fold}
        assert not (adapted & test_ids)
        # And the artifact's own recorded ids satisfy the consumer's check.
        emb.assert_no_test_fold_leak(meta, sorted(test_ids))

    for other in range(1, 5):
        assert not np.allclose(arrays[0], arrays[other]), (
            f"fold 0 and fold {other} produced identical features; the "
            "per-fold artifact would be five copies of one set"
        )


def test_the_randomised_control_is_built_marked_and_refused_elsewhere(
    tmp_path, clean_repo
):
    """**Phase 8 arm B's component-role control, end to end through the
    TASK.** The builder was tested first and the task second once already
    this phase; this drives the config -> extract -> save -> consume chain
    the cluster will run.

    Three things it must get right, each silent otherwise: the control's
    features differ from the real set's, its metadata says it is a control,
    and every consumer that did not ask for one refuses it.
    """
    import contextlib
    import io

    import cleft.provenance.context as context_module
    from cleft.run import main

    manifest_dir, staged_dir = make_artifacts(tmp_path)
    config = builders.write_config(
        tmp_path / "extract.yaml",
        phase="p8",
        inputs=[
            declare("manifest_v1", manifest_dir),
            declare("staged_v1", staged_dir),
        ],
        task={
            "kind": "extract_embeddings",
            "manifest_artifact": "manifest_v1",
            "staged_artifact": "staged_v1",
            "out_version": "embeddings_control_v1",
            "batch_size": 4,
            "sets": [
                {"backbone": "stub_graph", "init": "imagenet", "geometry": "g1"},
                {
                    "backbone": "stub_graph", "init": "imagenet",
                    "geometry": "g1", "randomise": True,
                },
            ],
        },
    )
    with contextlib.redirect_stdout(io.StringIO()):
        original = context_module.default_repo_root
        context_module.default_repo_root = lambda: clean_repo
        try:
            main(["--config", str(config), "--out", str(tmp_path / "runs")])
        finally:
            context_module.default_repo_root = original

    out_root = clean_repo / "data" / "embeddings" / "embeddings_control_v1"
    real_dir = out_root / "stub_graph__imagenet__g1"
    control_dir = out_root / "stub_graph__imagenet__g1__randomised"
    # The control is a SEPARATE directory whose name says what it is -- two
    # sets differing only in their contents is the failure PLAN Part 1 opens
    # with.
    assert real_dir.is_dir() and control_dir.is_dir()

    manifest_ids = list(range(1, 11))
    real_values, real_meta = emb.load(real_dir, manifest_ids=manifest_ids)
    control_values, control_meta = emb.load(control_dir, manifest_ids=manifest_ids)

    assert real_meta["randomisation"] is None
    assert control_meta["randomisation"]["randomised"] is True
    # **The features genuinely differ.** A control identical to the real set
    # would report the explanation unchanged under randomisation -- the worst
    # verdict, from a test that never ran.
    assert real_values.shape == control_values.shape
    assert not np.array_equal(real_values, control_values)

    # And the pairing check refuses each in the other's place.
    common = dict(
        kind="feature_map", backbone="stub_graph", init="imagenet",
        geometry="g1", checkpoint_sha256=None, region_scheme="native",
    )
    with pytest.raises(emb.EmbeddingError, match="did not ask for one"):
        emb.check_pairing(control_meta, **common)
    with pytest.raises(emb.EmbeddingError, match="secretly the real set"):
        emb.check_pairing(real_meta, **common, expect_randomised=True)
    emb.check_pairing(real_meta, **common)
    emb.check_pairing(control_meta, **common, expect_randomised=True)


def test_the_control_refuses_the_combinations_that_would_mean_nothing():
    """Per-fold BN re-estimation on a noise backbone measures the noise's
    statistics; the pooling axis is a ViT's and arm B is a graph arm."""
    from cleft.train import extract

    images = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    with pytest.raises(extract.ExtractError, match="not combined"):
        extract.extract_features(
            "stub_graph", "imagenet", images, checkpoint_path=None,
            adapt_rows=np.array([0, 1]), randomise=True,
        )
    with pytest.raises(extract.ExtractError, match="pooling axis is a ViT"):
        extract.extract_features(
            "stub", "imagenet", images, checkpoint_path=None,
            block=11, token="cls", randomise=True,
        )
