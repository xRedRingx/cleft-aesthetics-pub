"""The graph cleft arm: packing, pairing, the frozen-harness ride, the band
config. The stub_graph backbone runs the whole regime without torch; the
torch half (warm start, classifier re-init, forward_from_features composition)
is verified by scripts/verify_backbone_builds.py where torch exists.
"""

from __future__ import annotations

def _referenced_inputs(task: dict) -> list[str]:
    """[2026-09-01] The input names a task block references.

    ``schema.validate`` now refuses a task that names an input the
    config does not declare, so a scaffold config must declare the
    placeholders its task uses. Derived from the task rather than
    hand-listed, so these fixtures cannot drift from what they exercise.
    """
    from cleft.config.schema import _input_references

    return sorted({value for _, value in _input_references(task)})


import json
from pathlib import Path

import numpy as np
import pytest

from cleft import embeddings as emb
from cleft.train import graph_cleft as gc
from cleft.train.harness import TrainConfig

from fixtures import builders
from test_extract import make_artifacts


# --------------------------------------------------------------------------
# packing
# --------------------------------------------------------------------------


def test_pack_and_unpack_round_trip_exactly():
    rng = np.random.default_rng(1)
    maps = rng.normal(size=(5, 3, 2, 2)).astype(np.float32)
    boxes = rng.uniform(size=(5, 4, 4)).astype(np.float32)

    packed = gc.pack(maps, boxes)
    assert packed.shape == (5, 3 * 2 * 2 + 4 * 4)
    maps_back, boxes_back = gc.unpack(packed, (3, 2, 2), 4)
    assert np.array_equal(maps, maps_back)
    assert np.array_equal(boxes, boxes_back)

    packed = gc.pack(maps, None)
    maps_back, boxes_back = gc.unpack(packed, (3, 2, 2), 0)
    assert np.array_equal(maps, maps_back) and boxes_back is None

    with pytest.raises(gc.GraphCleftError, match="must be \\(N, C, H, W\\)"):
        gc.pack(maps.reshape(5, -1), None)
    with pytest.raises(gc.GraphCleftError, match="layout"):
        gc.unpack(packed, (3, 2, 2), 7)


# --------------------------------------------------------------------------
# the pairing checks: the artifact must BE the declared arm
# --------------------------------------------------------------------------


def metadata(**overrides) -> dict:
    base = {
        "kind": "feature_map",
        "backbone": "stub_graph",
        "init": "scut_masked",
        "geometry": "g2",
        "pretrain_scheme": "native",
        "checkpoint_sha256": "a" * 64,
    }
    base.update(overrides)
    return base


def check(**overrides):
    arguments = dict(
        backbone="stub_graph", init="scut_masked", geometry="g2",
        region_scheme="native", checkpoint_sha256="a" * 64,
    )
    meta_overrides = overrides.pop("meta", {})
    arguments.update(overrides)
    return gc.check_artifact_pairing(metadata(**meta_overrides), **arguments)


def test_the_pairing_checks_refuse_every_mismatch_by_name():
    check()  # the aligned case passes

    with pytest.raises(gc.GraphCleftError, match="feature_map artifacts"):
        check(meta={"kind": "pooled"})
    with pytest.raises(gc.GraphCleftError, match="artifact backbone"):
        check(meta={"backbone": "srgnn"})
    with pytest.raises(gc.GraphCleftError, match="artifact init"):
        check(meta={"init": "scut_original"})
    with pytest.raises(gc.GraphCleftError, match="artifact geometry"):
        check(meta={"geometry": "g1"})
    with pytest.raises(gc.GraphCleftError, match="scheme consistency"):
        check(region_scheme="grid")
    with pytest.raises(gc.GraphCleftError, match="none was declared"):
        check(checkpoint_sha256=None)
    with pytest.raises(gc.GraphCleftError, match="PAIRING"):
        check(checkpoint_sha256="b" * 64)


def test_imagenet_has_no_scheme_to_be_consistent_with():
    """Any cleft scheme is legitimate on an imagenet init -- recorded, not
    implied -- and an imagenet artifact CARRYING a scheme is malformed."""
    for scheme in ("native", "grid", "anatomy", "random"):
        gc.check_artifact_pairing(
            metadata(init="imagenet", pretrain_scheme=None, checkpoint_sha256=None),
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme=scheme, checkpoint_sha256=None,
        )
    # [UPDATED 2026-09-02] The message generalised when check_pairing was
    # keyed on the VARIANT rather than on the name "imagenet": the branch
    # now covers every init with no pretraining of ours, foundation inits
    # included. Same refusal, wider and truer sentence.
    with pytest.raises(
        gc.GraphCleftError, match="no pretraining of ours to have"
    ):
        gc.check_artifact_pairing(
            metadata(init="imagenet", checkpoint_sha256=None),
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", checkpoint_sha256=None,
        )

    # And the arm may not DECLARE one either. An imagenet graph arm never
    # loads a checkpoint (`run` seed-initialises its layers), so a declared
    # input would be hash-verified and written into inputs.json while feeding
    # nothing -- a provenance record claiming data fed a run it did not, which
    # is the rule pretrain.MASKED_INPUT_NAME exists for. This refusal arrived
    # with the shared checker; the graph path did not have it before.
    with pytest.raises(gc.GraphCleftError, match="no pretraining checkpoint"):
        gc.check_artifact_pairing(
            metadata(init="imagenet", pretrain_scheme=None, checkpoint_sha256=None),
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", checkpoint_sha256="a" * 64,
        )


def test_the_stub_arm_is_native_only():
    with pytest.raises(gc.GraphCleftError, match="native-only"):
        gc.boxes_for_arm(
            backbone="stub_graph", region_scheme="grid", geometry="g2",
            geometry_rows=None, patient_ids=[1], staged_images=None,
        )
    assert gc.boxes_for_arm(
        backbone="stub_graph", region_scheme="native", geometry="g2",
        geometry_rows=None, patient_ids=[1], staged_images=None,
    ) is None


def test_generated_schemes_need_recorded_content_boxes():
    with pytest.raises(gc.GraphCleftError, match="content_x"):
        gc.content_boxes_from_geometry(
            [{"patient_id": "1", "aspect_ratio": "0.74"}], [1]
        )
    boxes = gc.content_boxes_from_geometry(
        [
            {"patient_id": "1", "content_x": "10", "content_y": "0",
             "content_w": "200", "content_h": "224"},
        ],
        [1],
    )
    assert boxes.tolist() == [[10, 0, 200, 224]]


# --------------------------------------------------------------------------
# the regime, end to end on the stub
# --------------------------------------------------------------------------


def maps_from_staged(staged_dir: Path, geometry: str) -> np.ndarray:
    """Tiny feature maps carrying the fixture's label signal."""
    images = np.load(staged_dir / f"staged_patient_{geometry}.npy")
    maps = images.astype(np.float32).mean(axis=(1, 2))  # (n, 3)
    return np.tile(maps[:, :, None, None], (1, 1, 2, 2))  # (n, 3, 2, 2)


def make_embedding_set(
    directory: Path, staged_dir: Path, *, init="scut_masked", geometry="g2",
    scheme="native", checkpoint_sha256="a" * 64,
) -> Path:
    emb.save(
        directory, maps_from_staged(staged_dir, geometry),
        backbone="stub_graph", backbone_kind="graph",
        init=init, geometry=geometry,
        variant=emb.expected_variant(init, geometry),
        checkpoint_sha256=checkpoint_sha256,
        patient_ids=list(range(1, 11)),
        pretrain_scheme=None if init == "imagenet" else scheme,
    )
    return directory


def test_the_third_regime_runs_through_the_frozen_harness(tmp_path):
    """Frozen maps in, the harness's own CV around them: gates 5 and 6 run
    inside, the summary declares the regime, and the fingerprint makes two
    same-seed runs comparable by string -- which is also asserted."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, init="imagenet", checkpoint_sha256=None
    )

    def once():
        return gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dir=embeddings_dir,
            checkpoint_path=None, checkpoint_sha256=None,
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", seed=7,
            train_config=TrainConfig(
                max_epochs=6, patience=3, inner_val_frac=0.25, seed=7
            ),
            log=lambda *_: None,
        )

    result = once()
    summary = result.summary
    assert summary["regime"] == "graph_third"
    assert summary["features"]["features"] == "frozen_backbone_feature_maps"
    assert summary["oof"]["pcc"] > 0.9, "the label is in the maps; the stub must learn"
    gate6 = summary["gates"]["gate6_oof_reconstruction"]
    assert gate6["every_patient_predicted_once"] is True
    assert gate6["no_patient_in_own_training_fold"] is True
    assert summary["n_patients"] == 10

    again = once()
    assert (
        summary["fingerprint"]["predictions_sha256"]
        == again.summary["fingerprint"]["predictions_sha256"]
    ), "two same-seed runs must be identical"


def test_a_mispaired_checkpoint_refuses_before_any_training(tmp_path):
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, checkpoint_sha256="a" * 64
    )
    with pytest.raises(gc.GraphCleftError, match="PAIRING"):
        gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dir=embeddings_dir,
            checkpoint_path=tmp_path / "never_read.npz",
            checkpoint_sha256="b" * 64,
            backbone="stub_graph", init="scut_masked", geometry="g2",
            region_scheme="native", seed=7,
            train_config=TrainConfig(
                max_epochs=4, patience=2, inner_val_frac=0.25, seed=7
            ),
            log=lambda *_: None,
        )


def test_the_handler_sweeps_seeds_and_reports_the_band(tmp_path, clean_repo):
    """The seed-band mechanism itself: N seeds in one job, per-seed prefixed
    outputs, and gate-2-style variance derived from THIS sweep."""
    from cleft.run import main

    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, checkpoint_sha256=None, init="imagenet"
    )

    config = builders.write_config(
        tmp_path / "band.yaml",
        phase="p6",
        inputs=[
            {
                "name": name,
                "path": str(path),
                "rollup_sha256": __import__("cleft.provenance", fromlist=["hash_path"]).hash_path(path)["rollup"],
            }
            for name, path in (
                ("manifest_v1", manifest_dir),
                ("staged_v1", staged_dir),
                ("set_v1", embeddings_dir),
            )
        ],
        task={
            "kind": "train_graph_cv",
            "manifest_artifact": "manifest_v1",
            "staged_artifact": "staged_v1",
            "embeddings_artifact": "set_v1",
            "backbone": "stub_graph",
            "init": "imagenet",
            "geometry": "g2",
            "region_scheme": "native",
            "seeds": [7, 99],
            "max_epochs": 5,
            "patience": 2,
            "inner_val_frac": 0.25,
            "monitor": "inner_val_mse",
        },
    )

    import contextlib
    import io

    import cleft.provenance.context as context_module

    original = context_module.default_repo_root
    context_module.default_repo_root = lambda: clean_repo
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])
    finally:
        context_module.default_repo_root = original

    band = json.loads((run_dir / "seed_variance.json").read_text(encoding="utf-8"))
    assert band["n_seeds"] == 2
    for seed in (7, 99):
        assert (run_dir / f"seed_{seed}__metrics.json").exists()


# --------------------------------------------------------------------------
# per-fold BN re-extraction: the leak is the thing to get right
# --------------------------------------------------------------------------


def make_per_fold_sets(root: Path, staged_dir: Path, manifest_dir: Path, *, leak=None):
    """Five sets, each adapted on its own fold's TRAINING patients.

    ``leak=f`` builds fold ``f``'s set adapted on ALL patients instead -- the
    defect the check exists for, so the check can be shown to fire.
    """
    from cleft.data.manifest import load_manifest

    rows = load_manifest(manifest_dir / "manifest.csv")
    ids = [int(r["patient_id"]) for r in rows]
    fold_of = {int(r["patient_id"]): int(r["fold"]) for r in rows}
    folds = sorted(set(fold_of.values()))

    maps = maps_from_staged(staged_dir, "g2")
    directories = {}
    for fold in folds:
        adapt = ids if leak == fold else [p for p in ids if fold_of[p] != fold]
        rows_used = [ids.index(p) for p in adapt]
        # Same dependence real BN re-estimation has: a shift set by whoever
        # was adapted on. Without it the five sets would be identical and
        # every assertion below would pass on data that never varied.
        shifted = maps - maps[rows_used].mean(axis=0, keepdims=True)
        directory = root / f"set_fold{fold}"
        emb.save(
            directory, shifted,
            backbone="stub_graph", backbone_kind="graph",
            init="imagenet", geometry="g2",
            variant=None, checkpoint_sha256=None,
            patient_ids=ids, manifest_ids=ids,
            pretrain_scheme=None,
            fold=fold, adapted_on_patient_ids=adapt,
            bn_reestimation={"n_batchnorm_modules": 0, "n_rows": len(adapt)},
        )
        directories[fold] = directory
    return directories


def test_the_per_fold_path_runs_and_the_five_sets_genuinely_differ(tmp_path):
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    directories = make_per_fold_sets(tmp_path, staged_dir, manifest_dir)

    # The premise: if the sets were identical, everything below would pass on
    # one array copied five times.
    first = np.load(directories[0] / "values.npy")
    second = np.load(directories[1] / "values.npy")
    assert not np.allclose(first, second), (
        "the per-fold sets must actually differ, or this whole path is "
        "exercising five copies of one artifact"
    )

    result = gc.run(
        manifest_dir=manifest_dir, staged_dir=staged_dir,
        embeddings_dirs=directories,
        checkpoint_path=None, checkpoint_sha256=None,
        backbone="stub_graph", init="imagenet", geometry="g2",
        region_scheme="native", seed=7,
        train_config=TrainConfig(
            max_epochs=6, patience=3, inner_val_frac=0.25, seed=7
        ),
        log=lambda *_: None,
    )
    features = result.summary["features"]
    assert features["per_fold_bn_reextraction"] is True
    assert features["n_fold_sets"] == 5

    # The leak verdict is recorded per fold, in the SHAREABLE summary -- a
    # check whose result is not written down is one nobody can confirm ran.
    checks = features["test_fold_leak_check"]
    assert set(checks) == {"0", "1", "2", "3", "4"}
    for report in checks.values():
        assert report["checked"] is True and report["overlap"] == 0
        assert report["n_adapted_on"] > 0

    gate6 = result.summary["gates"]["gate6_oof_reconstruction"]
    assert gate6["every_patient_predicted_once"] is True
    assert gate6["no_patient_in_own_training_fold"] is True


def test_the_leak_check_fires_when_a_fold_adapted_on_its_own_test_patients(tmp_path):
    """**The teeth.** A set adapted on all 237 has every shape right, a valid
    hash, 237 rows in manifest order, and would score BETTER. Nothing
    downstream can see it, which is why it is asserted rather than arranged."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    directories = make_per_fold_sets(tmp_path, staged_dir, manifest_dir, leak=2)

    with pytest.raises(emb.EmbeddingError, match="TEST-FOLD LEAK"):
        gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dirs=directories,
            checkpoint_path=None, checkpoint_sha256=None,
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", seed=7,
            train_config=TrainConfig(
                max_epochs=3, patience=2, inner_val_frac=0.25, seed=7
            ),
            log=lambda *_: None,
        )


def test_the_leak_check_reports_honestly_on_a_set_that_adapted_on_nothing():
    """An ordinary set cannot leak, and the report says that rather than
    claiming a check it did not perform."""
    report = emb.assert_no_test_fold_leak({"fold": None}, [1, 2, 3])
    assert report["checked"] is False and "cannot leak" in report["why"]


def test_a_set_built_for_another_fold_is_refused(tmp_path):
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    directories = make_per_fold_sets(tmp_path, staged_dir, manifest_dir)
    swapped = dict(directories)
    swapped[0], swapped[1] = directories[1], directories[0]

    with pytest.raises(gc.GraphCleftError, match="records fold"):
        gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dirs=swapped,
            checkpoint_path=None, checkpoint_sha256=None,
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", seed=7,
            train_config=TrainConfig(
                max_epochs=3, patience=2, inner_val_frac=0.25, seed=7
            ),
            log=lambda *_: None,
        )


def test_a_missing_fold_set_is_refused(tmp_path):
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    directories = make_per_fold_sets(tmp_path, staged_dir, manifest_dir)
    del directories[3]

    with pytest.raises(gc.GraphCleftError, match="no embedding set was"):
        gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dirs=directories,
            checkpoint_path=None, checkpoint_sha256=None,
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", seed=7,
            log=lambda *_: None,
        )


def test_exactly_one_feature_source_must_be_declared(tmp_path):
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    directories = make_per_fold_sets(tmp_path, staged_dir, manifest_dir)
    single = make_embedding_set(
        tmp_path / "single", staged_dir, init="imagenet", checkpoint_sha256=None
    )
    for kwargs in (
        {},
        {"embeddings_dir": single, "embeddings_dirs": directories},
    ):
        with pytest.raises(gc.GraphCleftError, match="exactly one of"):
            gc.run(
                manifest_dir=manifest_dir, staged_dir=staged_dir,
                checkpoint_path=None, checkpoint_sha256=None,
                backbone="stub_graph", init="imagenet", geometry="g2",
                region_scheme="native", seed=7, log=lambda *_: None,
                **kwargs,
            )


def test_the_row_level_check_catches_a_fold_used_for_the_wrong_rows(tmp_path):
    """The second half of the leak defence. The artifact check catches a set
    built wrong; this catches a correct set applied to the wrong fold, which
    is the only thing that can see the factory's fold counter drifting from
    the harness's iteration."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    directories = make_per_fold_sets(tmp_path, staged_dir, manifest_dir)
    maps_by_fold = {
        fold: np.load(directory / "values.npy")
        for fold, directory in directories.items()
    }

    backbone = gc.IndexedStubBackbone(
        maps_by_fold=maps_by_fold, fold=0, n_regions=0,
        # Fold 0's test rows, per make_artifacts' `index % 5` assignment.
        test_rows=frozenset({0, 5}),
    )
    backbone.reset(np.array([2.0, 3.0]))
    # Row 5 is fold 0's own test patient: training on it is the leak.
    features = gc.pack_indexed(10, None)[[5, 6, 7]]
    with pytest.raises(gc.GraphCleftError, match="belong to this fold's TEST set"):
        backbone.train_epoch(features, np.array([2.0, 3.0, 4.0]))


def test_pack_indexed_round_trips_and_survives_harness_slicing():
    """The index column is the identity, so any slice the harness takes
    carries exactly the rows it took -- which is what makes the indirection
    checkable rather than a second place for a misalignment to live."""
    boxes = np.arange(10 * 3 * 4, dtype=np.float32).reshape(10, 3, 4)
    packed = gc.pack_indexed(10, boxes)
    assert packed.shape == (10, 1 + 12)

    rows = np.array([7, 2, 9])
    indices, boxes_back = gc.unpack_indexed(packed[rows], 3)
    assert indices.tolist() == rows.tolist()
    assert np.array_equal(boxes_back, boxes[rows])

    indices, none_boxes = gc.unpack_indexed(gc.pack_indexed(4, None), 0)
    assert indices.tolist() == [0, 1, 2, 3] and none_boxes is None
    with pytest.raises(gc.GraphCleftError, match="indexed rows are"):
        gc.unpack_indexed(packed, 7)


def test_the_reextraction_configs_are_the_2x2_they_claim(repo_root, monkeypatch):
    """Four cells, two measured, two shipped. They must differ in exactly the
    two normalisation halves and nothing else, or a difference between them is
    not attributable to normalisation."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    extract = load_config(repo_root / "configs" / "p6_extract_srgnn_adabn.yaml")
    assert extract["task"]["per_fold_bn_reestimation"] is True
    assert extract["task"]["out_version"] != "embeddings_v1", (
        "embeddings_v1 is immutable and holds the maps the 0.1340 and 0.0468 "
        "were measured on; they must stay readable"
    )

    cell_c = load_config(
        repo_root / "configs" / "p6_srgnn_probe_adabn_artifact.yaml"
    )["task"]
    cell_d = load_config(
        repo_root / "configs" / "p6_srgnn_probe_adabn_artifact_full.yaml"
    )["task"]

    # C and D differ ONLY in the post-backbone half.
    differing = {
        key for key in set(cell_c) | set(cell_d)
        if cell_c.get(key) != cell_d.get(key)
    }
    assert differing == {"trainable"}
    assert cell_c["trainable"] == "classifier"
    assert cell_d["trainable"] == "classifier_adabn"

    # Both consume the per-fold artifact, all five folds, and neither declares
    # a single-set artifact alongside it.
    for cell in (cell_c, cell_d):
        assert set(cell["embeddings_artifacts_by_fold"]) == {0, 1, 2, 3, 4}
        assert not cell.get("embeddings_artifact")
        assert cell["seeds"] == [1337]

    # And they match the measured cells A and B everywhere else, so the 2x2
    # varies only what it claims to.
    cell_a = load_config(
        repo_root / "configs" / "p6_srgnn_frozen_graph.yaml"
    )["task"]
    for field in (
        "backbone", "init", "geometry", "region_scheme", "label",
        "max_epochs", "patience", "inner_val_frac", "monitor",
        "learning_rate", "weight_decay", "batch_size", "seeds",
    ):
        assert cell_c[field] == cell_a[field], f"{field} drifted from cell A"


#: [MEASURED 2026-08-01] The per-fold artifact was produced and its five
#: rollups were filled into both arm configs from ``declare_inputs.py``. The
#: date is here rather than in prose so the assertion below can quote it.
PER_FOLD_ARTIFACT_PRODUCED = "2026-08-01"


def test_the_reextraction_hashes_are_real(repo_root, monkeypatch):
    """**INVERTED 2026-08-01 -- the fourth instance of this cycle.**

    This previously asserted the five per-fold hashes were all-zeros, which
    was correct exactly as long as ``data/embeddings/embeddings_adabn_v1`` did
    not exist, and stale the moment the re-extraction produced it. Inverted
    rather than deleted: an unfilled placeholder must still fail, or a
    regression to the pre-extraction state would look like a passing suite and
    the arms would refuse at guard 3 on the cluster instead.

    **The pattern is now built for rather than reacted to.** Two invariants in
    ``tests/test_smoke_run.py`` never need flipping at all -- the same
    artifact path must declare the same hash in every config, and a
    placeholder must be documented. This test stays because those express a
    RELATION and this expresses a FACT about one artifact: that it exists.

    The fold count is derived from the config, not written as ``5``, so a
    sixth fold would be covered without another edit.
    """
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    for name in (
        "p6_srgnn_probe_adabn_artifact.yaml",
        "p6_srgnn_probe_adabn_artifact_full.yaml",
    ):
        cfg = load_config(repo_root / "configs" / name)
        by_name = {entry["name"]: entry for entry in cfg["inputs"]}
        per_fold = cfg["task"]["embeddings_artifacts_by_fold"]
        assert per_fold, f"{name} declares no per-fold sets"

        for fold, input_name in sorted(per_fold.items()):
            rollup = by_name[input_name]["rollup_sha256"]
            assert rollup != "0" * 64, (
                f"{name}: fold {fold}'s embedding hash regressed to the "
                f"all-zeros placeholder. The artifact was produced "
                f"{PER_FOLD_ARTIFACT_PRODUCED}; guard 3 would refuse this run "
                "on the cluster, and a green suite here would not have said so."
            )
            assert len(rollup) == 64 and set(rollup) <= set("0123456789abcdef"), (
                f"{name}: fold {fold}'s rollup is not a sha256 digest"
            )

        # Every other declared input was already real and must stay so.
        for other in ("manifest_v1", "staged_v1", "ckpt_srgnn_native_masked_g2"):
            assert by_name[other]["rollup_sha256"] != "0" * 64


# --------------------------------------------------------------------------
# the shipped seed-band config: the gate, written down
# --------------------------------------------------------------------------


def test_the_seed_band_config_is_the_gate_it_claims_to_be(repo_root, monkeypatch):
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    cfg = load_config(repo_root / "configs" / "p6_srgnn_seed_band.yaml")
    task = cfg["task"]

    assert cfg["tier"] == "keeper", "the band is citable; every claim divides by it"
    assert task["backbone"] == "srgnn"
    assert task["init"] == "scut_masked" and task["geometry"] == "g2"
    assert task["region_scheme"] == "native", "the canonical ladder arm"

    seeds = task["seeds"]
    assert len(seeds) == 10 and len(set(seeds)) == 10, (
        "ten distinct seeds -- brief §5, the gate-2 procedure on the third regime"
    )
    # The cleft convention, not pretraining's fixed budget: early stopping
    # stays where collapse is the documented behaviour.
    assert task["patience"] == 5 and task["monitor"] == "inner_val_mse"

    by_name = {entry["name"]: entry for entry in cfg["inputs"]}
    assert task["checkpoint"] in by_name
    assert by_name[task["checkpoint"]]["rollup_sha256"] != "0" * 64, (
        "the checkpoint hash is real (verified 2026-08-01)"
    )
    # This assertion previously held the hash to all-zeros -- correct exactly
    # as long as the extraction artifact did not exist, stale the moment it
    # did (the extraction ran 2026-08-01; srgnn__scut_masked__g2__native is
    # 95,136,804 bytes, the (237, 2048, 7, 7) float32 decision exactly).
    # Inverted rather than deleted: an unfilled placeholder must still fail.
    # A placeholder assertion is correct exactly once -- same cycle as the
    # xception constant's unverified marker.
    assert by_name[task["embeddings_artifact"]]["rollup_sha256"] != "0" * 64, (
        "the embeddings hash is real (the extraction artifact was produced "
        "and verified 2026-08-01); an all-zeros value here would mean the "
        "config regressed to its pre-extraction placeholder"
    )


# --------------------------------------------------------------------------
# the measured band: the gate's result, written down
# --------------------------------------------------------------------------


def test_the_measured_band_is_arithmetically_self_consistent():
    """The pasted figures are re-derived, not retyped.

    The band arrives by paste (no cluster access), so the one thing checkable
    on the laptop is whether its parts agree with each other: the range must
    be max-min, and the three claimable deltas must all be 1.96*sd*sqrt(2/n)
    from ONE sd. A transcription slip in any single field breaks that.
    """
    import math

    band = gc.MEASURED_GRAPH_SEED_BAND

    assert band["max"] > band["mean"] > band["min"]
    assert round(band["max"] - band["min"], 6) == band["range"]

    # Every delta implies an sd; all three must agree with the recorded one to
    # within its own rounding, since the run derived them from the unrounded
    # value. Recomputing FROM the rounded sd would tolerate a wrong sd -- this
    # direction does not.
    implied = [
        delta / (1.96 * math.sqrt(2.0 / int(n)))
        for n, delta in band["claimable_delta_at_n_seeds"].items()
    ]
    for value in implied:
        assert abs(value - band["sd"]) < 5e-7, (
            f"claimable delta implies sd {value}, recorded sd is {band['sd']}"
        )
    assert max(implied) - min(implied) < 1e-6, "the three deltas disagree on sd"

    # The formula the project already owns, applied to this arm's own sd --
    # never the frozen probe's constant (PLAN §4.12.1).
    from cleft.train import phase3

    for n, delta in band["claimable_delta_at_n_seeds"].items():
        assert abs(phase3.claimable_delta(int(n), sd=band["sd"]) - delta) < 2e-6


def test_the_band_says_which_quantity_it_is():
    """PLAN §4.12.1: two arms can both report 'seed SD' and mean different
    things. This one warm-starts the graph layers identically across seeds, so
    the one source a reader would ASSUME a graph band covers is absent -- and
    a band that does not say so is the R2 error waiting to happen."""
    band = gc.MEASURED_GRAPH_SEED_BAND

    assert "graph-layer initialisation" in band["does_not_measure"]
    assert "warm-started" in band["does_not_measure"]
    for source in ("classifier_init", "batch_order", "inner_val_split"):
        assert source in band["measures"], f"{source} is a source this seed drives"

    # It is this arm's band, and it is not the frozen probe's.
    from cleft.train import phase3

    assert band["sd"] != phase3.MEASURED_SEED_BAND["sd"]
    assert band["against_frozen_probe_sd"] == phase3.MEASURED_SEED_BAND["sd"]
    assert band["arm"] != phase3.MEASURED_SEED_BAND["arm"]


def test_the_stub_refuses_the_classifier_diagnostic(tmp_path):
    """The stub is a linear head under BOTH policies, so it cannot distinguish
    them. Letting it run would produce a metrics.json reporting
    'trainable: classifier' on an arm that implements no such thing -- a check
    reporting success on its own failure mode, which is the tally's whole
    subject. Same reasoning as the native-only rule."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, init="imagenet", checkpoint_sha256=None
    )
    with pytest.raises(gc.GraphCleftError, match="linear head under every policy"):
        gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dir=embeddings_dir,
            checkpoint_path=None, checkpoint_sha256=None,
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", trainable="classifier", seed=7,
            train_config=TrainConfig(
                max_epochs=3, patience=2, inner_val_frac=0.25, seed=7
            ),
            log=lambda *_: None,
        )


def test_an_unknown_trainable_policy_is_refused_by_name(tmp_path):
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, init="imagenet", checkpoint_sha256=None
    )
    with pytest.raises(gc.GraphCleftError, match="unknown trainable policy"):
        gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dir=embeddings_dir,
            checkpoint_path=None, checkpoint_sha256=None,
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", trainable="everything", seed=7,
            log=lambda *_: None,
        )


def test_the_default_policy_is_the_regime_not_the_diagnostic(tmp_path):
    """The brief's trap is a trap for LADDER arms. The default must be the
    regime, so freezing the graph layers is only ever reached deliberately."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, init="imagenet", checkpoint_sha256=None
    )
    result = gc.run(
        manifest_dir=manifest_dir, staged_dir=staged_dir,
        embeddings_dir=embeddings_dir,
        checkpoint_path=None, checkpoint_sha256=None,
        backbone="stub_graph", init="imagenet", geometry="g2",
        region_scheme="native", seed=7,
        train_config=TrainConfig(
            max_epochs=4, patience=2, inner_val_frac=0.25, seed=7
        ),
        log=lambda *_: None,
    )
    assert result.summary["trainable"] == "graph_layers"
    assert result.summary["regime"] == "graph_third"
    assert result.summary["train_config"]["eval_mode_while_training"] is False
    # A regime run carries the band; a diagnostic run carries the diagnostic.
    assert result.summary["seed_band"] is gc.MEASURED_GRAPH_SEED_BAND
    assert "diagnostic" not in result.summary


def test_the_diagnostic_config_is_the_hypothesis_test_it_claims_to_be(
    repo_root, monkeypatch
):
    """It must differ from the band arm in the policy and the rate ONLY -- the
    data, the folds, the label and the stopping rule all identical, or the
    comparison it exists to make is against something else."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    band = load_config(repo_root / "configs" / "p6_srgnn_seed_band.yaml")["task"]
    diag = load_config(repo_root / "configs" / "p6_srgnn_frozen_graph.yaml")["task"]

    assert diag["trainable"] == "classifier"
    assert diag["seeds"] == [1337] and band["seeds"][0] == 1337, (
        "the single seed is the band's first, so the arms compare seed to seed "
        "as well as to the band mean"
    )
    # 1e-3 is the probe rate the 0.2529 comparator used, not the 12M-stack
    # rate. The two arms differ here ON PURPOSE and the record says so.
    assert diag["learning_rate"] == 0.001 and band["learning_rate"] == 0.0001

    for field in (
        "kind", "backbone", "init", "geometry", "region_scheme", "label",
        "max_epochs", "patience", "inner_val_frac", "monitor", "weight_decay",
        "batch_size",
    ):
        assert diag[field] == band[field], (
            f"{field} differs between the diagnostic and the arm it diagnoses; "
            "the comparison would not be about the training policy"
        )

    diag_cfg = load_config(repo_root / "configs" / "p6_srgnn_frozen_graph.yaml")
    band_cfg = load_config(repo_root / "configs" / "p6_srgnn_seed_band.yaml")
    assert (
        {(e["name"], e["path"], e["rollup_sha256"]) for e in diag_cfg["inputs"]}
        == {(e["name"], e["path"], e["rollup_sha256"]) for e in band_cfg["inputs"]}
    ), "same artifacts, same verified hashes: the arms differ in policy, not data"


def test_the_diagnostic_record_states_its_readings_before_it_runs(repo_root):
    """Stating the outcome readings afterwards would be indistinguishable from
    explaining away whichever number arrived -- the ARM_PREDICTION pattern."""
    record = gc.FROZEN_GRAPH_DIAGNOSTIC

    assert record["is_a_ladder_arm"] is False
    assert record["trainable"] == "classifier"
    assert record["eval_mode_while_training"] is True, (
        "dropout and BN must be off, or the 'frozen' representation moves"
    )
    # The asymmetry is the substance: one direction is conclusive and the
    # other is not, and the record must not read as though both were.
    assert "CONCLUSIVE" in record["outcome_high"]
    assert "NOT established" in record["outcome_low"]
    assert "AdaBN" in record["outcome_low"], (
        "the low-score alternative explanation must be named, not implied"
    )
    assert record["seeds"] == 1 and "nothing finer" in record["what_one_seed_licenses"]

    # The eval-mode claim is MEASURED, and its negative was run first: in
    # train mode eight tensors move rather than two, the extra six being BN
    # buffers that requires_grad never protected.
    verified = record["eval_mode_verified"]
    assert verified["moves_under_policy"] == ["classifier.bias", "classifier.weight"]
    assert verified["moves_without_eval_mode"] > len(verified["moves_under_policy"]), (
        "if train mode moved no more than eval mode, the policy would be inert "
        "and the check that asserts it would pass on its own failure mode"
    )
    assert "buffers" in verified["why_it_matters"]

    # The config it names must exist, or the record points at nothing.
    assert (repo_root / record["config"]).is_file()

    # SR-GNN's classifier is Linear(DENSE_DIM, 1): the recorded trainable count
    # is derived from the ported spec, not typed in beside it.
    from cleft.models import srgnn

    assert record["trainable_parameters"] == srgnn.parameter_count(
        num_outputs=1
    )["classifier"]


def test_both_probe_policies_are_grouped_not_matched_by_literal():
    """**A membership test, not ``== "classifier"``.** The AdaBN probe freezes
    exactly the same parameters, runs in the same eval mode and is equally
    not-a-ladder-arm, so every site deciding "is this the diagnostic" must
    treat the two alike. A literal comparison would have trained the AdaBN
    arm's graph layers while its config said otherwise -- the 2026-07-28
    shape, where a check keyed to one string silently excluded a new arm."""
    assert set(gc.PROBE_POLICIES) == {"classifier", "classifier_adabn"}
    assert set(gc.PROBE_POLICIES) < set(gc.TRAINABLE_POLICIES)
    assert "graph_layers" not in gc.PROBE_POLICIES, "the regime is not a probe"

    # No site may decide a probe question by comparing against the bare
    # literal. This is a statement about how the decision is WRITTEN, not
    # about what it returns, so it is checked against the parsed module.
    #
    # **Via the AST, not a text search.** A substring scan for '== "classifier"'
    # fires on the comments that document this very rule -- which is the same
    # defect in miniature: matching text rather than the property. The AST
    # cannot be fooled by a comment or a docstring.
    import ast

    tree = ast.parse(Path(gc.__file__).read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(c, ast.Constant) and c.value == "classifier"
            for c in node.comparators
        )
    ]
    assert not offenders, (
        f"lines {offenders} decide a probe question by comparing to the "
        '"classifier" literal; use PROBE_POLICIES so a third policy cannot '
        "silently miss the branch"
    )

    # And the check has teeth: the same walk DOES find the one legitimate
    # literal comparison, against "classifier_adabn", which is a question
    # about that policy specifically rather than about probes in general.
    adabn_sites = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Compare)
        and any(
            isinstance(c, ast.Constant) and c.value == "classifier_adabn"
            for c in node.comparators
        )
    ]
    assert adabn_sites, (
        "the AST walk found no string comparisons at all, so its silence "
        "about 'classifier' is not evidence"
    )


def test_the_stub_refuses_the_adabn_diagnostic_too(tmp_path):
    """It has no BatchNorm to re-estimate, so the arm would report a policy it
    does not implement -- and this is the refusal the PROBE_POLICIES grouping
    exists to make automatic."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, init="imagenet", checkpoint_sha256=None
    )
    with pytest.raises(gc.GraphCleftError, match="linear head under every policy"):
        gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dir=embeddings_dir,
            checkpoint_path=None, checkpoint_sha256=None,
            backbone="stub_graph", init="imagenet", geometry="g2",
            region_scheme="native", trainable="classifier_adabn", seed=7,
            log=lambda *_: None,
        )


def test_the_adabn_config_differs_in_exactly_one_field(repo_root, monkeypatch):
    """The pair is a one-factor comparison or it is nothing: if anything else
    moved, a difference between them would not be attributable to the BN
    re-estimation."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    plain = load_config(repo_root / "configs" / "p6_srgnn_frozen_graph.yaml")
    adabn = load_config(repo_root / "configs" / "p6_srgnn_frozen_graph_adabn.yaml")

    differing = {
        key for key in set(plain["task"]) | set(adabn["task"])
        if plain["task"].get(key) != adabn["task"].get(key)
    }
    assert differing == {"trainable"}, (
        f"the pair differs in {sorted(differing)}; it must differ in the "
        "training policy alone"
    )
    assert adabn["task"]["trainable"] == "classifier_adabn"
    assert adabn["task"]["seeds"] == [1337]
    assert adabn["tier"] == "keeper"

    assert (
        {(e["name"], e["path"], e["rollup_sha256"]) for e in adabn["inputs"]}
        == {(e["name"], e["path"], e["rollup_sha256"]) for e in plain["inputs"]}
    ), "same artifacts, same verified hashes"


def test_the_determinism_config_is_the_regime_at_one_seed(repo_root, monkeypatch):
    """Determinism is two SEPARATE jobs at the SAME seed. The band's ten seeds
    in one job is the opposite shape -- it varies the thing being held
    constant and shares one process -- so this config exists rather than
    reusing it.

    It must run the REGIME: gate 1 tests the harness, and what the harness
    gained since gate 1 was measured is the graph-training path. A frozen
    probe would exercise the classifier and leave that path unverified."""
    from cleft.config import load_config

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    det = load_config(repo_root / "configs" / "p6_determinism_graph.yaml")
    band = load_config(repo_root / "configs" / "p6_srgnn_seed_band.yaml")

    assert det["task"]["seeds"] == [1337], "one seed, two jobs"
    assert det["task"]["trainable"] == "graph_layers", (
        "the regime, not a probe -- the graph-training path is what needs "
        "re-verifying"
    )
    assert det["tier"] == "keeper"

    # Identical to the band arm apart from the seed list, so what is being
    # measured twice is the arm the ladder will actually use.
    differing = {
        key for key in set(det["task"]) | set(band["task"])
        if det["task"].get(key) != band["task"].get(key)
    }
    assert differing == {"seeds"}, (
        f"the determinism arm differs from the band arm in {sorted(differing)}; "
        "it must differ only in how many seeds run"
    )

    # The false-pass modes are written down, because a matching digest over
    # two aborted runs matches trivially.
    text = (repo_root / "configs" / "p6_determinism_graph.yaml").read_text(
        encoding="utf-8"
    )
    assert "FALSE PASS" in text
    for mode in ("aborted early", "resumed the first", "two seeds", "2.3e-05"):
        assert mode in text, f"the false-pass list must name {mode!r}"

    # The graph cleft path has no `deterministic` knob -- it routes through the
    # frozen determinism.configure unconditionally, so these runs are flag-ON
    # while the thirty pretraining runs were deliberately flag-off. Importing
    # that setting across is a live misreading, so the config names it.
    assert "flag-ON" in text
    from cleft.train import graph_cleft

    source = Path(graph_cleft.__file__).read_text(encoding="utf-8")
    assert "determinism.configure(" in source
    assert '"deterministic"' not in source, (
        "if the graph path ever gains a deterministic knob, this config's "
        "flag-ON claim stops being true"
    )


def test_gate1_reproduction_needs_no_config_edit(repo_root, monkeypatch):
    """The OTHER half of the item. GATE1_REFERENCE can only be reproduced by
    the arm that produced it, and `run.task_train_cv` gates the comparison on
    exactly that. p3_train_cv.yaml still satisfies every condition -- including
    carrying seed 1337 -- so reproducing the reference is a re-run, not an
    edit to a keeper config."""
    from cleft.config import load_config
    from cleft.train import phase3

    monkeypatch.setenv("CLEFT_SCUT_ROOT", "/nonexistent/sentinel")
    task = load_config(repo_root / "configs" / "p3_train_cv.yaml")["task"]

    assert task["backbone"] == "vit_b16"
    assert task["geometry"] == "g1"
    assert task["trainable"] == "head"
    assert task["label"] == "mean"
    assert task.get("patch_scheme", "whole") == "whole"
    assert phase3.GATE1_REFERENCE["seed"] in task["seeds"], (
        "seed 1337 must be in the list or task_train_cv skips the comparison"
    )
    # And it must NOT have acquired a pooled artifact: the reference was
    # measured on live extraction, and an artifact-fed arm reaches its vectors
    # through a different code path.
    assert not task.get("embeddings_artifact")
    assert task.get("init", "imagenet") == "imagenet"


def test_the_adabn_record_states_its_readings_before_it_runs(repo_root):
    record = gc.ADABN_DIAGNOSTIC

    assert record["is_a_ladder_arm"] is False
    assert record["trainable"] == "classifier_adabn"
    assert (repo_root / record["config"]).is_file()
    # **The asymmetry SURVIVES, and an earlier draft of this test asserted it
    # did not.** The high branch is conclusive; the low branch is not, because
    # this arm reaches 2 of SR-GNN's 42 BatchNorms and the other 40 were fixed
    # at extraction time. The record must NOT claim a low result establishes
    # the architecture reading -- that overclaim was caught by adversarial
    # review before the arm ran, and this assertion is what keeps it out.
    assert "CONCLUSIVE" in record["outcome_high"]
    assert "statistics mismatch" in record["outcome_high"]
    assert "NARROWED, not removed" in record["outcome_low"]
    assert "NOT established" in record["outcome_low"]
    # The affirmative claim specifically, not the substring "ESTABLISH" --
    # which "NOT established" also contains, so a bare negative scan could
    # never distinguish the two readings it is meant to separate.
    lowered = record["outcome_low"].lower()
    for affirmative in ("is established", "reading is established", "establishes"):
        assert affirmative not in lowered, (
            "a low result narrows the normalisation alternative; it does not "
            "establish the architecture reading, because the extraction-time "
            "mismatch is baked into the artifact"
        )

    # And the scope limit is recorded as data, not prose, with what would
    # actually close the question.
    scope = record["does_not_test"]
    assert scope["modules_adapted"] == 2 and scope["modules_total"] == 42
    assert "EXTRACTION" in scope["unreachable"] or "extraction" in scope["unreachable"]
    assert "extract.py" in scope["evidence"]
    assert "LayerNorm" in scope["comparator_asymmetry"], (
        "the confound is one-directional: ViT has no running statistics, so "
        "nothing equivalent happened on the 0.2529 side"
    )
    assert "RE-EXTRACTION" in scope["what_would_test_it"].upper()
    assert "not another cleft arm" in scope["what_would_test_it"]
    assert record["seeds"] == 1
    assert "quotable as a value" in record["one_seed_caveat"]
    assert "head-init defect" in record["precedent"], (
        "the precedent is this project's own, not a literature citation alone"
    )


def test_the_bn_mechanism_facts_are_measured_and_name_the_noop(repo_root):
    """Each of these is a way the arm runs to completion, reports the policy,
    and changes nothing. They are measured rather than recalled, and the
    reset one is the sharpest: a pretrained counter in the thousands makes
    the cumulative update negligible."""
    facts = gc.MECHANISM_VERIFIED

    assert facts["momentum_none_is_cumulative"] is True
    assert facts["after_reset_equals_mean_of_batch_stats"] is True
    assert facts["bn_only_train_mode_leaves_dropout_off"] is True

    noop = facts["without_reset_is_effectively_a_noop"]
    # It got 0.14% of the way. Derived rather than retyped, so the recorded
    # fraction cannot drift from the measurement it summarises.
    travelled = abs(noop["after_four_batches"] - noop["start_running_mean"])
    distance = abs(noop["target_mean"] - noop["start_running_mean"])
    assert abs(travelled / distance - noop["fraction_of_the_way"]) < 5e-4
    assert noop["fraction_of_the_way"] < 0.01, (
        "if the un-reset update were substantial the reset would be tidiness, "
        "not a correctness requirement"
    )
    assert "MANDATORY" in noop["verdict"]

    # The batch-of-one guard exists because torch raises, not defensively.
    assert "Expected more than 1 value per channel" in (
        facts["batch_of_one_raises_in_train_mode"]
    )
    # And the approximation is stated rather than glossed.
    assert "not exactly the fold's" in facts["equal_batch_weighting_caveat"]

    # **The batching numbers are this arm's REAL folds, re-derived from the
    # project's own splitter.** An earlier draft said "~121 rows, 7 batches
    # and one of 9" -- it applied the 0.2 inner-val fraction twice, to a count
    # that was already the inner-train. Checked here against the splitter so
    # the record cannot drift from what the arm does.
    from cleft.train.harness import inner_val_split

    measured = facts["measured_batching"]
    ids = list(range(1, 238))
    assignments = {pid: i % 5 for i, pid in enumerate(ids)}
    sizes, last, counts_per_fold = [], [], []
    for fold in range(5):
        outer = [pid for pid in ids if assignments[pid] != fold]
        train_ids, _ = inner_val_split(outer, 0.2, 1337, fold)
        n = len(train_ids)
        batch = measured["batch_size"]
        widths = [min(s + batch, n) - s for s in range(0, n, batch)]
        sizes.append(n)
        last.append(widths[-1])
        counts_per_fold.append(len(widths))

    assert sorted(sizes) == sorted(measured["inner_train_sizes"])
    assert sorted(last) == sorted(measured["last_batch_sizes"])
    assert set(counts_per_fold) == {measured["n_batches_per_fold"]}
    # The over-weighting the caveat describes, derived rather than retyped.
    assert abs(1 / measured["n_batches_per_fold"] - measured["short_batch_weight"]) < 1e-9
    for width, n in zip(last, sizes):
        assert round(width / n, 4) in measured["short_batch_true_share"]


def test_adabn_is_srgnn_only_by_architecture_and_the_counts_are_real():
    """**AG-Net has no BatchNorm after the frozen boundary at all** -- all 53
    of its modules are inside the ResNet-50 that never runs when the feature
    map arrives precomputed. Two consequences: the policy must refuse there
    rather than adapt nothing, and the normalisation-mismatch alternative this
    whole arm exists to test cannot arise for AG-Net in the first place.

    The counts are cross-checked against the BUILT models in
    ``scripts/verify_backbone_builds.py``, where torch exists -- not here
    behind an importorskip. torch stays out of the test extra deliberately,
    and a suite test that skips is a test that passes without asserting
    anything, which is the pattern this project spends the most effort
    avoiding."""
    counts = gc.MECHANISM_VERIFIED["post_backbone_batchnorms"]

    assert counts["srgnn"]["after_frozen_boundary"] == 2
    assert counts["srgnn"]["named"] == ["bn1", "bn2"]
    assert counts["agnet"]["after_frozen_boundary"] == 0, (
        "if AG-Net grew a post-backbone BatchNorm, the SR-GNN-only claim and "
        "the no-caveat-needed conclusion both change"
    )
    assert counts["agnet"]["total"] > counts["agnet"]["after_frozen_boundary"], (
        "AG-Net does have BatchNorms -- they are all in the frozen backbone, "
        "which is the point"
    )
    assert "SR-GNN-only by architecture" in counts["consequence"]

    # **What does NOT follow, asserted explicitly.** An earlier draft
    # concluded that AG-Net therefore needs no statistics caveat. That reads
    # "no BatchNorm after the boundary" as "no BatchNorm involved": its 53
    # modules ran at EXTRACTION with SCUT statistics, so its whole chain is
    # baked into the artifact and none of it is reachable at cleft time.
    # AG-Net is the worse case, not the exempt one.
    assert "not_a_consequence" in counts
    assert "worse case, not the" in counts["not_a_consequence"]
    assert "no statistics caveat" not in counts["consequence"], (
        "the exemption claim must not return to the consequence field"
    )

    # The verify script is where these are re-derived from built models, so
    # the record and its cross-check cannot drift apart unnoticed.
    verify = (
        Path(gc.__file__).parents[3] / "scripts" / "verify_backbone_builds.py"
    ).read_text(encoding="utf-8")
    assert "post_backbone_batchnorms" in verify, (
        "the recorded counts must be cross-checked against the built models "
        "somewhere torch exists, or they are numbers nobody re-derived"
    )


def test_the_configuration_reading_is_recorded_as_dead_with_its_evidence():
    """The status field said OPEN and named what would flip it. That ran, so
    this asserts the rewrite happened rather than the placeholder going
    stale -- the cycle every artifact here has been through."""
    record = gc.THIRD_REGIME_MEAN_IS_OPEN

    # Phrased case-insensitively: the status text was rewritten when the 2x2
    # closed the question, and keying on its exact wording is what made three
    # earlier assertions stale. The claim is what must survive, not the prose.
    assert "configuration reading dead" in record["status"].lower()
    closed = record["configuration_reading_closed_by"]
    assert closed["pcc"] == 0.1340
    assert closed["trainable_parameters"] == 2049, (
        "the argument IS the parameter count: 2,049 cannot overfit 152 samples"
    )
    # **The overclaim guard, re-pointed.** While the question was open this
    # asserted "not yet established". The 2x2 closed it, so the guard now
    # checks the other direction: the status may say the reading STANDS, but
    # NORMALISATION_2X2 must still carry the residual, or "every reachable
    # correction is worse" quietly becomes "no correction could help" -- the
    # exact overclaim this sequence already made once.
    assert "architecture reading stands" in record["status"].lower()
    assert "unpursued rather than eliminated" in (
        gc.NORMALISATION_2X2["what_stands"]["residual"]
    )
    assert record["surviving_alternative"], (
        "what was considered and tested must stay recorded even once refuted"
    )
    assert "adabn" in record["flipped_by"].lower()

    # The trained arm is reframed, and the direction is the finding.
    reframe = closed["reframes_the_trained_arm"]
    assert "opposite to overfitting" in reframe
    assert "NOT claimable" in reframe, (
        "+0.017 is inside the 0.031 five-seed threshold and must say so"
    )

    # One seed against SD 0.025: the conclusion survives the interval, the
    # value does not. Both stated, and the interval checked against the band.
    caveat = record["one_seed_caveat"]
    assert caveat["value_is_quotable"] is False
    assert caveat["conclusion_survives_it"] is True
    assert caveat["band_sd_used"] == gc.MEASURED_GRAPH_SEED_BAND["sd"]
    low, high = caveat["plausible_interval"]
    assert low < closed["pcc"] < high
    assert high < gc.MEASURED_GRAPH_SEED_BAND["mean"] + 0.05 < 0.2529, (
        "the interval's top must fall well short of the ViT probe, which is "
        "what lets the conclusion survive a one-seed measurement"
    )


def test_the_frozen_probe_result_is_paired_with_its_prediction():
    """The reading applied is the one written down before the number arrived.
    Recorded beside the prediction so the pairing is visible rather than
    asserted in prose somewhere else."""
    result = gc.FROZEN_GRAPH_DIAGNOSTIC["result"]

    assert result["branch_taken"] == "outcome_low"
    assert result["pcc"] == 0.1340
    assert result["trainable_parameters"] == gc.FROZEN_GRAPH_DIAGNOSTIC[
        "trainable_parameters"
    ], "the freeze took on the cluster as it did locally"
    assert "configuration reading is dead" in result["licensed"].lower()
    assert "NOT thereby established" in result["licensed"]
    assert "One seed" in result["not_licensed"]


def test_the_2x2_refutes_both_accounts_by_their_own_predictions():
    """**Both live accounts died on predictions registered before the cells
    ran**, which is the entire return on registering them. An account adjusted
    after each result would have survived all four and explained nothing."""
    record = gc.NORMALISATION_2X2
    cells = record["cells"]

    # The ordering IS the finding: more cleft adaptation, monotonically worse.
    ranked = sorted(cells, key=lambda k: -cells[k]["pcc"])
    assert ranked == ["A", "C", "B", "D"]
    assert "monotonically worse" in record["ordering"]

    # Each cell is the combination it claims to be -- the 2x2 varies exactly
    # two binary factors, and a mislabelled cell would invert the reading.
    assert cells["A"]["backbone_bn"] == "scut" and cells["A"]["head_bn"] == "scut"
    assert cells["D"]["backbone_bn"] == "cleft" and cells["D"]["head_bn"] == "cleft"
    assert {(c["backbone_bn"], c["head_bn"]) for c in cells.values()} == {
        ("scut", "scut"), ("scut", "cleft"), ("cleft", "scut"), ("cleft", "cleft")
    }

    normalisation = record["refutes"]["normalisation_hypothesis"]
    assert "0.25" in normalisation["predicted"]
    assert normalisation["measured"] == cells["D"]["pcc"]
    assert "REFUTED" in normalisation["verdict"]

    # The partial-adaptation account predicted the CONSISTENT cell recovers.
    # Derived rather than restated: D below B is what kills it.
    partial = record["refutes"]["partial_adaptation_account"]
    assert cells["D"]["pcc"] < cells["B"]["pcc"]
    assert "REFUTED" in partial["verdict"]
    assert "never the mechanism" in partial["verdict"]


def test_the_unpursued_explanation_carries_its_arithmetic_and_stays_unpursued():
    """A named mechanism with the numbers attached, and an explicit decision
    NOT to chase it -- running variants until one improves would be searching
    for a configuration rather than testing a hypothesis."""
    unpursued = gc.NORMALISATION_2X2["unpursued_explanation"]
    maths = unpursued["arithmetic"]

    # Ten batches per layer for 2048 channels, against SCUT's ~104.
    assert -(-maths["cleft_images"] // maths["cleft_batch_size"]) == (
        maths["cleft_batches_per_layer"]
    )
    assert maths["cleft_batches_per_layer"] < maths["scut_batches"] / 10, (
        "the claim is that SCUT's statistics are far better estimated; if the "
        "batch counts were comparable the explanation would not stand up"
    )
    assert maths["layers_composing"] == 40
    assert unpursued["status"].startswith("UNPURSUED")
    assert "search" in unpursued["status"]
    assert unpursued["would_be_tested_by"]


def test_what_stands_is_stated_without_overclaiming():
    """The architecture reading now rests on A being best of four with every
    REACHABLE correction measured and worse. That is not the same as "no
    correction could help", and the record must not say it is -- this sequence
    already produced one overclaim of exactly that shape."""
    stands = gc.NORMALISATION_2X2["what_stands"]

    assert stands["best_srgnn_configuration"] == "A"
    assert stands["frozen_probe"] == gc.NORMALISATION_2X2["cells"]["A"]["pcc"]
    assert stands["trained_graph_layers"] == gc.MEASURED_GRAPH_SEED_BAND["mean"]
    assert stands["vit_frozen_probe"] == 0.2529

    # The fractions are derived, so they cannot drift from the numbers above.
    for key, value in (
        ("frozen", stands["frozen_probe"]),
        ("trained_graph_layers", stands["trained_graph_layers"]),
    ):
        assert abs(
            stands["fraction_of_vit"][key] - value / stands["vit_frozen_probe"]
        ) < 5e-4

    # The residual is named and scoped to ESTIMATION rather than domain, and
    # is recorded as unpursued rather than eliminated.
    assert "reachable" in stands["architecture_reading"]
    assert "QUALITY, not domain" in stands["residual"]
    assert "unpursued rather than eliminated" in stands["residual"]


def test_the_open_question_is_closed_and_says_what_closed_it():
    """Four rewrites, each the flip its predecessor asked for. This asserts the
    last one landed -- a regression to any earlier state would mean a recorded
    result was lost."""
    record = gc.THIRD_REGIME_MEAN_IS_OPEN

    assert record["status"].startswith("CLOSED")
    assert "ARCHITECTURE READING STANDS" in record["status"]
    assert record["closed_by"] == "NORMALISATION_2X2"
    # The forward-looking field became a backward-looking one; a "flips_when"
    # still present would mean something is still pending that is not.
    assert "flips_when" not in record
    assert "p6_extract_srgnn_adabn" in record["flipped_by"]
    assert "2 of 42" in record["why_a_re_extraction_was_required"]

    # Both readings must stay recorded even now that one of them stands, or
    # the write-up loses what was considered and rejected.
    assert set(record["readings"]) == {"architecture", "configuration"}


def test_the_mean_question_names_the_measurement_that_closes_it():
    """Two readings fit the same evidence, so the record must hold both and
    name the measurement that separates them.

    **This has now tracked the question through its whole life**, and the
    assertion moved each time the state did: it first required the status to
    start with "OPEN", then that it did NOT, and now that the naming field is
    the backward-looking ``flipped_by`` rather than a pending ``flips_when``.
    Each version was correct exactly once, which is why the durable invariants
    live in tests/test_smoke_run.py and this one asserts the SHAPE that does
    not change -- both readings recorded, the separating measurement named,
    and the delta consistent with the numbers it is derived from.
    """
    record = gc.THIRD_REGIME_MEAN_IS_OPEN

    assert set(record["readings"]) == {"architecture", "configuration"}
    assert not record["status"].startswith("OPEN"), (
        "the status regressed to OPEN; the question was closed 2026-08-01 by "
        "the completed 2x2 and that result must not be lost from this record"
    )
    assert "flipped_by" in record and "flips_when" not in record, (
        "the record must name what DID close it, not what would"
    )
    assert "trainable='classifier'" in record["separated_by"]

    # The delta is the recorded band's mean against the recorded probe mean --
    # derived here so a later edit to either cannot leave a stale difference.
    from cleft.train import phase3

    assert record["graph_regime_mean"] == gc.MEASURED_GRAPH_SEED_BAND["mean"]
    assert record["frozen_vit_probe_mean"] == phase3.MEASURED_SEED_BAND["mean"]
    assert (
        round(record["graph_regime_mean"] - record["frozen_vit_probe_mean"], 5)
        == round(record["delta"], 5)
    )

# --------------------------------------------------------------------------
# the AG-Net determinism opt-out
# --------------------------------------------------------------------------


def test_an_agnet_arm_must_declare_determinism_explicitly(tmp_path):
    """**Neither default is defensible, so there is none.**

    Default-on and the arm dies mid-run on an op nobody expected (the
    deterministic roi_align substitution needs a C++ compiler the pinned image
    lacks). Default-off and its results silently carry a ~2.3e-05 tolerance
    nobody declared, which every comparison against it has to know about.
    """
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, init="imagenet", checkpoint_sha256=None
    )
    with pytest.raises(gc.GraphCleftError, match="must declare `deterministic`"):
        gc.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            embeddings_dir=embeddings_dir,
            checkpoint_path=None, checkpoint_sha256=None,
            backbone="agnet", init="imagenet", geometry="g2",
            region_scheme="native", seed=7, log=lambda *_: None,
        )


def test_the_opt_out_is_refused_at_load_time_too(repo_root, monkeypatch):
    """Runtime refusal costs a queued job; load-time refusal costs nothing."""
    from cleft.config.schema import ConfigError, validate

    def config(**overrides):
        task = {
            "kind": "train_graph_cv", "manifest_artifact": "m",
            "staged_artifact": "s", "embeddings_artifact": "e",
            "backbone": "agnet", "init": "imagenet", "geometry": "g2",
            "region_scheme": "native", "max_epochs": 40, "patience": 5,
            "inner_val_frac": 0.2, "monitor": "inner_val_mse",
        }
        task.update(overrides)
        return {
            "schema_version": 1, "phase": "p7", "tier": "dev", "seed": 1337,
            "inputs": [
                {"name": name, "path": f"/tmp/{name}",
                 "rollup_sha256": "ab" * 32}
                for name in _referenced_inputs(task)
            ], "task": task,
        }

    with pytest.raises(ConfigError, match="task.deterministic is not set"):
        validate(config())
    # Declared either way, it validates -- the requirement is that it is SAID.
    assert validate(config(deterministic=False))["task"]["deterministic"] is False
    assert validate(config(deterministic=True))["task"]["deterministic"] is True
    # And no other graph backbone is forced to answer a question that is not
    # one for it: SR-GNN is bitwise under the flag.
    assert validate(config(backbone="srgnn"))["task"]["deterministic"] is None


def test_the_relaxation_goes_through_configure_and_records_its_bound(tmp_path):
    """**The setup is not reimplemented.** `determinism.configure` is called
    unconditionally -- it owns the python/numpy/torch seeding and the
    CUBLAS_WORKSPACE_CONFIG export, and a second copy of that is the CuBLAS
    defect (every backbone raising in the image, caught by the maintainer rather
    than a test). Only the ONE setting that cannot be satisfied is relaxed,
    and the record is corrected so it does not claim a flag that is off.
    """
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, init="imagenet", checkpoint_sha256=None
    )
    result = gc.run(
        manifest_dir=manifest_dir, staged_dir=staged_dir,
        embeddings_dir=embeddings_dir,
        checkpoint_path=None, checkpoint_sha256=None,
        backbone="stub_graph", init="imagenet", geometry="g2",
        region_scheme="native", deterministic=False, seed=7,
        train_config=TrainConfig(
            max_epochs=4, patience=2, inner_val_frac=0.25, seed=7
        ),
        log=lambda *_: None,
    )
    policy = result.summary["determinism_policy"]
    assert policy["declared"] is False
    assert policy["use_deterministic_algorithms"] is False
    assert "tolerance" in policy["comparison"]

    relaxed = policy["relaxed"]
    from cleft.models.agnet import TRAINING_DETERMINISM

    # The bound is READ from the measurement, not retyped beside it.
    assert relaxed["measured_tolerance"] == TRAINING_DETERMINISM[
        "flag_off_post_restore_step_max_diff"
    ]
    assert "NOT bitwise" in relaxed["how_to_compare"]
    assert "seed band" in relaxed["how_to_compare"], (
        "an AG-Net delta needs its own band with kernel noise as the floor"
    )
    # The seeding still happened -- only the flag moved.
    assert "determinism.configure still ran" in relaxed["seeding_unaffected"]
    assert result.summary["determinism"]["seed"] == 7
    assert result.summary["determinism"]["cublas_workspace_config"]


def test_an_undeclared_arm_keeps_the_flag_and_says_so(tmp_path):
    """Every backbone but AG-Net is bitwise under the flag, so omitting the
    field must leave it ON -- the opt-out is an exception, not a new default."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    embeddings_dir = make_embedding_set(
        tmp_path / "set", staged_dir, init="imagenet", checkpoint_sha256=None
    )
    result = gc.run(
        manifest_dir=manifest_dir, staged_dir=staged_dir,
        embeddings_dir=embeddings_dir,
        checkpoint_path=None, checkpoint_sha256=None,
        backbone="stub_graph", init="imagenet", geometry="g2",
        region_scheme="native", seed=7,
        train_config=TrainConfig(
            max_epochs=4, patience=2, inner_val_frac=0.25, seed=7
        ),
        log=lambda *_: None,
    )
    policy = result.summary["determinism_policy"]
    assert policy["declared"] is None
    assert "relaxed" not in policy
    # In the test venv torch is absent, so the flag was never SET rather than
    # turned off -- and the record must say that rather than reporting
    # AG-Net's measured tolerance for a run that has nothing to do with it.
    assert policy["comparison"] in (
        "bitwise", "not applicable: torch absent, so the flag was never set"
    )
    assert "tolerance" not in policy["comparison"]


# --------------------------------------------------------------------------
# Stage E0: the scheme axis closes, and what the attempt that got it wrong shows
# --------------------------------------------------------------------------


def test_the_scheme_axis_is_recorded_as_the_fourth_null():
    """Four independent attempts, and the last is the one measured under the
    condition the scheme design was built for: cleft data, trained graph
    layers, message passing live, and no checkpoint confound."""
    record = gc.SCHEME_AXIS_AT_CLEFT

    assert record["null_number"] == 4
    assert record["claimable"] is False
    assert set(record["means"]) == set(gc.CLEFT_SCHEMES)

    # The range is derived from the means, not retyped beside them.
    values = list(record["means"].values())
    assert abs((max(values) - min(values)) - record["range"]) < 5e-4

    # The largest gap must be the one the verdict rests on, and it must fall
    # SHORT of its own threshold -- otherwise this is not a null.
    gap = record["largest_gap"]
    assert gap["delta"] < gap["threshold"]
    assert abs(gap["delta"] - record["range"]) < 5e-4, (
        "the largest gap IS the range; if they disagree one is mis-stated"
    )

    # Why imagenet is the controlled condition, stated rather than implied.
    assert "no pretraining checkpoint" in record["why_imagenet"]
    assert "placement ALONE" in record["why_imagenet"]


def test_the_transfer_finding_is_recorded_with_what_it_overturns():
    """**The licence that was wrongly granted, now measured.** Four
    checkpoints statistically equal on SCUT spread by 0.095 at cleft time, of
    which placement is 0.016. A held-out score on the pretraining task does
    not predict which representation transfers."""
    record = gc.PRETRAINING_DOES_NOT_PREDICT_TRANSFER

    # The premise: they really were equal on SCUT, by that project's own band.
    from cleft.train.pretrain import SCHEME_AXIS_AT_PRETRAINING

    assert record["scut_scheme_range"] == SCHEME_AXIS_AT_PRETRAINING["range_max"]
    assert record["scut_scheme_range"] < record["scut_seed_band"] + 0.002

    # And the conclusion: the cleft spread is far larger, and placement is
    # only a small part of it.
    # ~5.9x: the spread with scheme-matched checkpoints against the spread
    # with one shared representation. Asserted as a ratio rather than a
    # difference so it states the size of the effect, not just its sign.
    ratio = (
        record["cleft_range_with_those_checkpoints"]
        / record["cleft_range_with_one_representation"]
    )
    assert ratio > 5.0, f"transfer quality dominates placement by {ratio:.1f}x"
    assert record["attributable_to_placement"] == gc.SCHEME_AXIS_AT_CLEFT["range"]
    assert "does not predict" in record["verdict"]

    # It generalises: every arm that picks a checkpoint on a SCUT number
    # inherits the gap, and the record must say so rather than scoping it to
    # Stage E.
    assert "Q1 and Q2" in record["consequence"]


def test_stage_e_no_longer_claims_a_licence_it_does_not_have():
    """The licence text said the pretraining null made the four checkpoints
    equivalent. That inference is measured false, and the lattice must say so
    where the comparison is declared -- not only in a separate record."""
    from cleft import ladder

    licence = ladder.LICENCES["region_scheme:checkpoint"]
    assert "NOT LICENSED" in licence
    assert "MEASURED FALSE" in licence
    assert "Stage E0 is the controlled comparison" in licence
    assert "NOT attributable to placement" in licence
