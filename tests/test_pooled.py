"""The pooled-artifact feature source: the Phase 7 entry piece.

``prepare_features`` extracts live at ImageNet init only, so the ladder's
pretrained transformer arms had no route to their own representation. This
module is that route, and every check the graph path enforces has to hold here
too -- checkpoint hash equality, init/geometry agreement, row order against the
manifest -- because each of them fails silently.
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


import numpy as np
import pytest

from cleft import embeddings as emb
from cleft.train import phase3, pooled
from cleft.train.harness import TrainConfig

from test_extract import make_artifacts


def write_set(
    directory,
    *,
    n: int = 10,
    dim: int = 6,
    backbone: str = "vit_b16",
    init: str = "scut_masked",
    geometry: str = "g2",
    checkpoint_sha256: str | None = "a" * 64,
    patient_ids: list | None = None,
    pretrain_scheme: str | None = None,
    values: np.ndarray | None = None,
    backbone_kind: str = "transformer",
):
    """One pooled set on disk, with knobs for each failure under test."""
    if values is None:
        rng = np.random.default_rng(0)
        values = rng.normal(size=(n, dim)).astype(np.float32)
    emb.save(
        directory, values,
        backbone=backbone, backbone_kind=backbone_kind,
        init=init, geometry=geometry,
        variant=emb.expected_variant(init, geometry),
        checkpoint_sha256=checkpoint_sha256,
        patient_ids=patient_ids or list(range(1, n + 1)),
        pretrain_scheme=pretrain_scheme,
    )
    return directory


def source(directory, **overrides) -> pooled.PooledSource:
    arguments = dict(
        directory=directory, backbone="vit_b16", init="scut_masked",
        geometry="g2", checkpoint_sha256="a" * 64,
    )
    arguments.update(overrides)
    return pooled.PooledSource(**arguments)


# --------------------------------------------------------------------------
# the pairing checks, mirrored from the graph path
# --------------------------------------------------------------------------


def test_the_aligned_case_loads_and_reports_its_provenance(tmp_path):
    directory = write_set(tmp_path / "set")
    values, report = pooled.load_features(source(directory), list(range(1, 11)))

    assert values.shape == (10, 6)
    assert report["features"] == pooled.PRECOMPUTED_EMBEDDINGS
    assert report["feature_dim"] == 6
    assert report["checkpoint_sha256"] == "a" * 64
    assert report["row_order_asserted_against_manifest"] is True
    # No extractor in this run -- reported as an explicit zero, because an
    # absent key and a measured zero read identically downstream.
    assert report["backbone_parameters"] == 0
    assert "EXTRACTION run" in report["extractor_note"]


def test_a_mispaired_checkpoint_is_refused(tmp_path):
    """The sharpest one: features from one representation trained under
    another's name would fit a ridge and produce a plausible number."""
    directory = write_set(tmp_path / "set", checkpoint_sha256="a" * 64)
    with pytest.raises(pooled.PooledFeatureError, match="PAIRING"):
        pooled.load_features(
            source(directory, checkpoint_sha256="b" * 64), list(range(1, 11))
        )


def test_a_pretrained_init_without_a_declared_checkpoint_is_refused(tmp_path):
    directory = write_set(tmp_path / "set")
    with pytest.raises(pooled.PooledFeatureError, match="none was declared"):
        pooled.load_features(
            source(directory, checkpoint_sha256=None), list(range(1, 11))
        )


def test_an_imagenet_arm_may_not_declare_a_checkpoint(tmp_path):
    """ImageNet has no pretraining checkpoint, so a declared one describes a
    run this is not -- and it would still be hash-verified and recorded in
    inputs.json, a provenance record claiming data fed a run it did not."""
    directory = write_set(tmp_path / "set", init="imagenet", checkpoint_sha256=None)
    with pytest.raises(pooled.PooledFeatureError, match="no pretraining checkpoint"):
        pooled.load_features(
            source(directory, init="imagenet", checkpoint_sha256="a" * 64),
            list(range(1, 11)),
        )


def test_backbone_init_and_geometry_must_agree(tmp_path):
    for field, wrong in (
        ("backbone", "swin_b"), ("init", "scut_original"), ("geometry", "g1"),
    ):
        directory = write_set(tmp_path / f"set_{field}")
        with pytest.raises(pooled.PooledFeatureError, match=f"artifact {field}"):
            pooled.load_features(
                source(directory, **{field: wrong}), list(range(1, 11))
            )


def test_a_feature_map_artifact_is_refused(tmp_path):
    """The mirror of the graph path's refusal of pooled sets. Flattened into a
    linear head a map is 100,352 columns over 237 patients -- it would run."""
    maps = np.zeros((10, 4, 2, 2), dtype=np.float32)
    directory = write_set(
        tmp_path / "set", backbone="srgnn", backbone_kind="graph", values=maps,
    )
    with pytest.raises(pooled.PooledFeatureError, match="consumes pooled artifacts"):
        pooled.load_features(source(directory, backbone="srgnn"), list(range(1, 11)))


def test_a_transformer_arm_refuses_a_set_carrying_a_scheme(tmp_path):
    """A transformer has no region-scheme axis, so a set recording one came
    from a graph checkpoint and has reached the wrong arm."""
    directory = write_set(tmp_path / "set", pretrain_scheme="grid")
    with pytest.raises(pooled.PooledFeatureError, match="no scheme axis"):
        pooled.load_features(source(directory), list(range(1, 11)))


def test_row_order_is_asserted_against_the_manifest_read_the_arm_used(tmp_path):
    """The brief's own warning. A permutation trains every patient against
    another patient's label and still produces a plausible PCC."""
    shuffled = [1, 2, 3, 4, 5, 6, 7, 8, 10, 9]
    directory = write_set(tmp_path / "set", patient_ids=shuffled)
    with pytest.raises(pooled.PooledFeatureError, match="different order"):
        pooled.load_features(source(directory), list(range(1, 11)))

    missing = write_set(tmp_path / "other", patient_ids=list(range(2, 12)))
    with pytest.raises(pooled.PooledFeatureError, match="row set does not match"):
        pooled.load_features(source(missing), list(range(1, 11)))


def test_pointing_at_the_embeddings_root_says_so(tmp_path):
    root = tmp_path / "embeddings_v1"
    write_set(root / "vit_b16__scut_masked__g2")
    with pytest.raises(pooled.PooledFeatureError, match="ONE embedding set"):
        pooled.load_features(source(root / "nonexistent"), list(range(1, 11)))


# --------------------------------------------------------------------------
# the parameter check: a new feature kind must not silently skip it
# --------------------------------------------------------------------------


def test_the_artifact_kind_stays_inside_the_parameter_check(tmp_path):
    """**The 2026-07-28 defect, in the place it would recur.** A feature kind
    that falls through ``assert_parameters_match_policy`` is exempted from the
    check that exists because this reporting was wrong once already -- which
    is exactly what happened to the patch arms. So the kind is enumerated and
    answered with its own condition rather than missing the branch."""
    assert pooled.PRECOMPUTED_EMBEDDINGS in phase3.ARTIFACT_EMBEDDINGS
    assert pooled.PRECOMPUTED_EMBEDDINGS not in phase3.FROZEN_EMBEDDINGS

    # A head over 768-dim embeddings is 769 parameters. The backbone's count
    # reported as the arm's trainable set is the defect, and it fails here.
    with pytest.raises(phase3.Phase3Error, match="2026-07-28 defect"):
        phase3.parameter_summary(
            "head",
            {"features": pooled.PRECOMPUTED_EMBEDDINGS, "feature_dim": 768},
            {"total_parameters": 85_798_656, "trainable_parameters": 85_798_656},
            backbone="vit_b16",
        )

    report = phase3.parameter_summary(
        "head",
        {"features": pooled.PRECOMPUTED_EMBEDDINGS, "feature_dim": 768},
        {"total_parameters": 769, "trainable_parameters": 769},
        backbone="vit_b16",
    )
    assert report["trainable_parameters"] == 769


def test_the_head_width_check_refuses_a_missing_width(tmp_path):
    """A check that passes because it had no input is not a check (PLAN R7,
    instance 5). Without feature_dim there is nothing to compare against."""
    with pytest.raises(pooled.PooledFeatureError, match="nothing to compare"):
        pooled.assert_head_matches_embedding_width(None, 769)


# --------------------------------------------------------------------------
# end to end through the harness
# --------------------------------------------------------------------------


def test_an_artifact_fed_arm_runs_through_the_frozen_harness(tmp_path):
    """The whole point: a pretrained-init transformer arm that could not run
    at all before. The stub head stands in for torch."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    images = np.load(staged_dir / "staged_patient_g2.npy")
    # Features carrying the label signal, so a working arm must learn.
    signal = images.astype(np.float32).mean(axis=(1, 2, 3))[:, None]
    values = np.hstack([signal, signal * 0.5]).astype(np.float32)
    directory = write_set(tmp_path / "set", values=values, dim=2)

    result = phase3.run(
        manifest_dir=manifest_dir, staged_dir=staged_dir,
        geometry="g2", label="mean", backbone="stub", trainable="head",
        pooled_source=source(directory), seed=7,
        train_config=TrainConfig(
            max_epochs=8, patience=3, inner_val_frac=0.25, seed=7
        ),
        log=lambda *_: None,
    )
    summary = result.summary
    assert summary["features"]["features"] == pooled.PRECOMPUTED_EMBEDDINGS
    assert summary["init"] == "scut_masked"
    assert summary["feature_provenance"] == "precomputed embedding artifact"
    assert summary["oof"]["pcc"] > 0.9, "the label is in the vectors; it must learn"

    # An arm WITHOUT an artifact is at imagenet, and says so rather than
    # leaving it to be inferred from a missing field.
    live = phase3.run(
        manifest_dir=manifest_dir, staged_dir=staged_dir,
        geometry="g2", label="mean", backbone="stub", trainable="head", seed=7,
        train_config=TrainConfig(
            max_epochs=4, patience=2, inner_val_frac=0.25, seed=7
        ),
        log=lambda *_: None,
    )
    assert live.summary["init"] == "imagenet"
    assert live.summary["feature_provenance"] == "extracted live in this run"


def test_a_wrong_length_artifact_is_caught_by_the_row_order_check(tmp_path):
    """There is no separate row-COUNT check on this path, and there should not
    be: ``assert_row_order`` pins the set and the order, which subsumes the
    count. This asserts the stronger check actually covers the weaker case,
    rather than a redundant one sitting beside it looking like evidence."""
    manifest_dir, staged_dir = make_artifacts(tmp_path)
    short = write_set(tmp_path / "short", n=9, patient_ids=list(range(1, 10)))

    with pytest.raises(phase3.Phase3Error, match="row set does not match"):
        phase3.run(
            manifest_dir=manifest_dir, staged_dir=staged_dir,
            geometry="g2", label="mean", backbone="stub", trainable="head",
            pooled_source=source(short), seed=7,
            train_config=TrainConfig(
                max_epochs=2, patience=1, inner_val_frac=0.25, seed=7
            ),
            log=lambda *_: None,
        )


def test_every_transformer_set_the_plan_derives_is_consumable_here(tmp_path):
    """The loop the ladder depends on: extraction produced sets from
    ``embedding_plan``, and this is what consumes them. If the plan derives a
    transformer set whose kind this path refuses, the arm cannot run -- which
    is the state the whole module exists to leave.

    Eight of the twelve Stage D arms are at pretrained inits, so before this
    path existed most of the ladder had no route to its own representation."""
    from cleft import embedding_plan
    from cleft.models.factory import BACKBONES

    transformer_sets = [
        entry for entry in embedding_plan.required_sets()
        if BACKBONES[entry["backbone"]]["kind"] == "transformer"
    ]
    assert transformer_sets, "the plan must derive transformer sets at all"

    pretrained = [e for e in transformer_sets if e["init"] != "imagenet"]
    assert pretrained, (
        "if every transformer set were imagenet, live extraction would have "
        "sufficed and this module would not be needed"
    )

    for index, entry in enumerate(transformer_sets):
        # A transformer set carries no scheme -- asserted, because a set that
        # did would be refused by the consumer at run time.
        assert entry["pretrain_scheme"] is None

        directory = write_set(
            tmp_path / f"set_{index}",
            backbone=entry["backbone"], init=entry["init"],
            geometry=entry["geometry"],
            checkpoint_sha256=None if entry["init"] == "imagenet" else "a" * 64,
        )
        values, report = pooled.load_features(
            source(
                directory, backbone=entry["backbone"], init=entry["init"],
                geometry=entry["geometry"],
                checkpoint_sha256=(
                    None if entry["init"] == "imagenet" else "a" * 64
                ),
            ),
            list(range(1, 11)),
        )
        assert report["artifact_kind"] == "pooled"
        assert len(values) == 10


# --------------------------------------------------------------------------
# the loader refuses configs that would run the wrong arm
# --------------------------------------------------------------------------


def test_a_pretrained_init_without_an_artifact_is_refused_at_load(tmp_path):
    """Live extraction is ImageNet-only, so this config would train on
    ImageNet vectors while metrics.json claimed scut_masked. Every field
    would look right."""
    from cleft.config.schema import ConfigError, validate

    def config(**task_overrides):
        task = {
            "kind": "train_cv", "manifest_artifact": "m", "staged_artifact": "s",
            "geometry": "g2", "label": "mean", "backbone": "vit_b16",
            "max_epochs": 10, "patience": 3, "inner_val_frac": 0.2,
            "monitor": "inner_val_mse",
        }
        task.update(task_overrides)
        return {
            "schema_version": 1, "phase": "p7", "tier": "dev", "seed": 1337,
            "inputs": [
                {"name": name, "path": f"/tmp/{name}",
                 "rollup_sha256": "ab" * 32}
                for name in _referenced_inputs(task)
            ], "task": task,
        }

    with pytest.raises(ConfigError, match="Live extraction is ImageNet-only"):
        validate(config(init="scut_masked"))

    with pytest.raises(ConfigError, match="no task.checkpoint is declared"):
        validate(config(init="scut_masked", embeddings_artifact="e"))

    with pytest.raises(ConfigError, match="ImageNet has no pretraining checkpoint"):
        validate(config(init="imagenet", checkpoint="c"))

    with pytest.raises(ConfigError, match="WHOLE-IMAGE vectors"):
        validate(
            config(
                init="scut_masked", embeddings_artifact="e", checkpoint="c",
                patch_scheme="grid",
            )
        )

    # The aligned case validates.
    validated = validate(
        config(init="scut_masked", embeddings_artifact="e", checkpoint="c")
    )
    assert validated["task"]["init"] == "scut_masked"
    # And an arm that declares nothing is at imagenet by default.
    assert validate(config())["task"]["init"] == "imagenet"
