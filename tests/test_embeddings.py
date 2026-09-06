"""The embedding artifact: one format, one loader, and the crossed-pair refusal.

No torch. The artifact is numpy plus JSON, which is the point -- Phase 7 reads it
without constructing a model.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from cleft import embeddings as emb

PATIENTS = [f"p{i}" for i in range(6)]


def pooled(n=6, d=8):
    return np.arange(n * d, dtype=float).reshape(n, d)


def per_region(n=6, r=27, d=8):
    return np.arange(n * r * d, dtype=float).reshape(n, r, d)


def regions(r=27):
    return [{"id": i, "band": f"b{i % 3}"} for i in range(r)]


# --------------------------------------------------------------------------
# the arithmetic: 24 sets from 12 runs
# --------------------------------------------------------------------------


def test_the_twenty_four_sets_come_from_twelve_runs():
    """4 x 3 inits x 2 geometries = 24 sets, and **12** runs -- not 8.

    Runs are counted per VARIANT, not per init. The first version of
    ``expected_set_count`` multiplied by ``len(INITS) - 1`` and returned 8,
    because it treated 'scut_masked' as one checkpoint when it is
    geometry-bound and needs one per geometry.
    """
    counts = emb.expected_set_count(n_backbones=4)
    assert counts["embedding_sets"] == 24
    assert counts["pretraining_runs"] == 12
    assert counts["variants"] == ["masked_g1", "masked_g2", "original"]
    assert set(emb.INITS) == {"imagenet", "scut_original", "scut_masked"}


def test_the_run_count_follows_the_rule_rather_than_agreeing_with_it():
    """Derived from ``expected_variant``, so if the geometry-binding rule ever
    changed the count would follow. A hardcoded 3 would agree with the rule
    today and silently stop describing it."""
    counts = emb.expected_set_count(n_backbones=1)
    derived = {
        emb.expected_variant(init, geometry)
        for init in emb.INITS
        for geometry in emb.GEOMETRIES
    } - {None}
    assert counts["pretraining_runs"] == len(derived)
    assert set(counts["variants"]) == derived


# --------------------------------------------------------------------------
# the init/geometry rule
# --------------------------------------------------------------------------


def test_a_masked_init_is_bound_to_its_geometry():
    assert emb.expected_variant("scut_masked", "g1") == "masked_g1"
    assert emb.expected_variant("scut_masked", "g2") == "masked_g2"
    assert emb.expected_variant("scut_original", "g1") == "original"
    assert emb.expected_variant("imagenet", "g2") is None


def test_a_crossed_masked_pair_is_refused():
    """**The constraint that makes 12 runs worth having.** A masked-G1 checkpoint
    feeding G2 embeddings would run, produce 237 vectors and fit a ridge -- while
    measuring the geometry effect WITHOUT the domain matching masking provides,
    which is a different comparison than the arm claims."""
    with pytest.raises(emb.EmbeddingError, match="GEOMETRY-BOUND"):
        emb.check_init_geometry("scut_masked", "g2", "masked_g1")
    with pytest.raises(emb.EmbeddingError, match="GEOMETRY-BOUND"):
        emb.check_init_geometry("scut_masked", "g1", "masked_g2")


def test_the_matched_pairs_pass():
    for geometry in ("g1", "g2"):
        assert emb.check_init_geometry(
            "scut_masked", geometry, f"masked_{geometry}"
        )["variant"] == f"masked_{geometry}"


def test_an_imagenet_init_carrying_a_checkpoint_is_refused():
    """ImageNet needs no pretraining run; a variant here means the set came from
    somewhere its label does not say."""
    with pytest.raises(emb.EmbeddingError, match="must come from"):
        emb.check_init_geometry("imagenet", "g1", "original")


def test_unknown_inits_and_geometries_are_refused_by_name():
    with pytest.raises(emb.EmbeddingError, match="unknown init"):
        emb.expected_variant("scut_synthetic", "g1")
    with pytest.raises(emb.EmbeddingError, match="unknown geometry"):
        emb.expected_variant("imagenet", "g3")


# --------------------------------------------------------------------------
# row order: the failure that still produces a plausible number
# --------------------------------------------------------------------------


def test_a_reordered_row_set_is_refused_and_names_the_first_difference():
    """Same patients, different order: every patient trained against another
    patient's label, and nothing downstream would notice."""
    shuffled = PATIENTS[:2][::-1] + PATIENTS[2:]
    with pytest.raises(emb.EmbeddingError, match="different order") as excinfo:
        emb.assert_row_order(shuffled, PATIENTS)
    assert "row 0" in str(excinfo.value)


def test_a_missing_or_extra_row_is_reported_differently_from_a_reorder():
    """Different causes, different messages -- a reorder is a bug in the writer,
    a missing patient is a bug in the selection."""
    with pytest.raises(emb.EmbeddingError, match="row set does not match"):
        emb.assert_row_order(PATIENTS[:-1], PATIENTS)
    with pytest.raises(emb.EmbeddingError, match="unexpected"):
        emb.assert_row_order(PATIENTS + ["p99"], PATIENTS)


def test_the_matching_order_passes_silently():
    assert emb.assert_row_order(PATIENTS, PATIENTS) is None


# --------------------------------------------------------------------------
# one format, two kinds
# --------------------------------------------------------------------------


def feature_map(n=6, c=8, size=7):
    rng = np.random.default_rng(3)
    return rng.normal(size=(n, c, size, size)).astype(np.float32)


def test_a_transformer_writes_pooled_and_a_graph_model_writes_a_feature_map(tmp_path):
    """[DECIDED 2026-07-31] Graph extraction stores the frozen backbone's
    feature MAP: SR-GNN's self-attention consumes the flattened per-region
    descriptor (the 6.42M mass a 2048-wide region vector would delete) and
    AG-Net's trainable SAGAN precedes regions entirely, so per_region could
    not feed either verified stack."""
    meta = emb.save(
        tmp_path / "vit", pooled(), backbone="vit_b16",
        backbone_kind="transformer", init="imagenet", geometry="g1",
        variant=None, checkpoint_sha256=None, patient_ids=PATIENTS,
        manifest_ids=PATIENTS,
    )
    assert meta["kind"] == "pooled"
    assert meta["n_regions"] is None

    meta = emb.save(
        tmp_path / "srgnn", feature_map(), backbone="srgnn",
        backbone_kind="graph", init="scut_masked", geometry="g1",
        variant="masked_g1", checkpoint_sha256="a" * 64,
        patient_ids=PATIENTS, manifest_ids=PATIENTS,
    )
    assert meta["kind"] == "feature_map"
    assert meta["n_regions"] is None
    assert meta["feature_dim"] == 8
    assert meta["map_size"] == [7, 7]


def test_one_loader_reads_both_kinds(tmp_path):
    """**Two loaders would be two places for the row-order bug to live**, and
    Phase 7 consumes both kinds."""
    emb.save(
        tmp_path / "a", pooled(), backbone="swin_b", backbone_kind="transformer",
        init="scut_original", geometry="g2", variant="original",
        checkpoint_sha256="b" * 64, patient_ids=PATIENTS,
    )
    emb.save(
        tmp_path / "b", feature_map(), backbone="agnet", backbone_kind="graph",
        init="scut_masked", geometry="g2", variant="masked_g2",
        checkpoint_sha256="c" * 64, patient_ids=PATIENTS,
    )

    values, meta = emb.load(tmp_path / "a", manifest_ids=PATIENTS)
    assert meta["kind"] == "pooled" and values.shape == (6, 8)
    values, meta = emb.load(tmp_path / "b", manifest_ids=PATIENTS)
    assert meta["kind"] == "feature_map" and values.shape == (6, 8, 7, 7)


def test_a_feature_map_refuses_baked_in_regions(tmp_path):
    """The map PRECEDES regions; the scheme is applied at cleft-train time
    through each patient's own geometry. Baking a region set in would
    re-couple extraction to the scheme axis -- exactly what the feature map
    decouples."""
    with pytest.raises(emb.EmbeddingError, match="PRECEDES regions"):
        emb.save(
            tmp_path / "x", feature_map(), backbone="srgnn",
            backbone_kind="graph", init="imagenet", geometry="g1", variant=None,
            checkpoint_sha256=None, patient_ids=PATIENTS, regions=regions(),
        )


def test_the_superseded_per_region_kind_is_still_validated_at_load(tmp_path):
    """per_region is superseded for graph EXTRACTION, but the loader may meet
    one; its invariants stay live rather than becoming dead prose. A crafted
    artifact whose region geometry does not describe its array must refuse."""
    import json as json_module

    directory = tmp_path / "old"
    directory.mkdir()
    np.save(directory / "values.npy", per_region(r=27))
    (directory / "metadata.json").write_text(
        json_module.dumps({
            "kind": "per_region",
            "init": "imagenet", "geometry": "g1", "variant": None,
            "patient_ids": list(PATIENTS),
            "regions": [{"id": i} for i in range(37)],
        }),
        encoding="utf-8",
    )
    with pytest.raises(emb.EmbeddingError, match="does not describe the array"):
        emb.load(directory)


def test_the_wrong_rank_for_the_declared_kind_is_refused(tmp_path):
    with pytest.raises(emb.EmbeddingError, match="must be \\(n_patients, feature_dim\\)"):
        emb.save(
            tmp_path / "x", per_region(), backbone="vit_b16",
            backbone_kind="transformer", init="imagenet", geometry="g1",
            variant=None, checkpoint_sha256=None, patient_ids=PATIENTS,
        )


def test_the_kind_follows_the_backbone_kind_not_the_array_rank():
    """Declared, not inferred. Inferring from rank would silently accept a
    pooled array from a graph backbone -- the discarded-features bug this
    mapping exists to prevent. Graph maps to feature_map [DECIDED
    2026-07-31]: the frozen boundary for both graph models is the backbone's
    final map, and per_region could not feed either verified stack."""
    assert emb.KIND_FOR_BACKBONE_KIND == {
        "transformer": "pooled", "graph": "feature_map"
    }
    assert "per_region" in emb.EMBEDDING_KINDS, (
        "superseded, not deleted: the loader may meet one"
    )
    from cleft.models import factory

    for name, spec in factory.BACKBONES.items():
        assert spec["kind"] in emb.KIND_FOR_BACKBONE_KIND, name


# --------------------------------------------------------------------------
# provenance and immutability
# --------------------------------------------------------------------------


def test_a_checkpoint_derived_set_must_record_its_checkpoint(tmp_path):
    with pytest.raises(emb.EmbeddingError, match="no checkpoint hash"):
        emb.save(
            tmp_path / "x", pooled(), backbone="vit_b16",
            backbone_kind="transformer", init="scut_original", geometry="g1",
            variant="original", checkpoint_sha256=None, patient_ids=PATIENTS,
        )


def test_an_existing_artifact_is_never_overwritten(tmp_path):
    kwargs = dict(
        backbone="vit_b16", backbone_kind="transformer", init="imagenet",
        geometry="g1", variant=None, checkpoint_sha256=None,
        patient_ids=PATIENTS,
    )
    emb.save(tmp_path / "v1", pooled(), **kwargs)
    with pytest.raises(emb.EmbeddingError, match="immutable"):
        emb.save(tmp_path / "v1", pooled(), **kwargs)


def test_duplicate_patient_ids_are_refused(tmp_path):
    with pytest.raises(emb.EmbeddingError, match="duplicates"):
        emb.save(
            tmp_path / "x", pooled(), backbone="vit_b16",
            backbone_kind="transformer", init="imagenet", geometry="g1",
            variant=None, checkpoint_sha256=None,
            patient_ids=["p0"] * 6,
        )


def test_the_loader_rechecks_rather_than_trusting_what_was_written(tmp_path):
    """An artifact can be correct when written and wrong when read -- edited,
    truncated, or produced by an older writer. The loader is the last point at
    which anything checks before a number comes out the other end."""
    emb.save(
        tmp_path / "v1", pooled(), backbone="vit_b16",
        backbone_kind="transformer", init="scut_masked", geometry="g1",
        variant="masked_g1", checkpoint_sha256="d" * 64, patient_ids=PATIENTS,
    )
    path = tmp_path / "v1" / "metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["geometry"] = "g2"  # the crossed pair, introduced after the write
    path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(emb.EmbeddingError, match="GEOMETRY-BOUND"):
        emb.load(tmp_path / "v1")

def test_every_init_round_trips_through_save_and_check_pairing():
    """**The test that would have caught it, and it is EXECUTED.**

    [FOUND 2026-09-02, AT FEATURE LOAD ON THE CLUSTER] ``save`` requires a
    checkpoint hash ``if variant is not None``; ``check_pairing`` required
    one ``if init != "imagenet"``. The foundation inits are neither, so the
    extraction wrote ``checkpoint_sha256: None`` legitimately and the
    reader demanded a declaration for a checkpoint that does not exist.

    **A source check could not have caught this** -- both branches read
    perfectly well on their own. What catches it is writing an artifact for
    every member of the vocabulary and then reading it back: a new init
    whose checkpoint requirement is unsatisfiable fails HERE, in the suite,
    rather than at feature load after two extraction runs.
    """
    import tempfile

    import numpy as np

    from cleft import embeddings

    patient_ids = [1, 2, 3]
    for init in embeddings.ALL_INITS:
        for geometry in embeddings.GEOMETRIES:
            variant = embeddings.expected_variant(init, geometry)
            checkpoint = "a" * 64 if variant is not None else None
            with tempfile.TemporaryDirectory() as tmp:
                directory = pathlib.Path(tmp) / "set"
                embeddings.save(
                    directory, np.zeros((3, 768), dtype=np.float32),
                    backbone="vit_b16", backbone_kind="transformer",
                    init=init, geometry=geometry, variant=variant,
                    checkpoint_sha256=checkpoint,
                    patient_ids=patient_ids, manifest_ids=patient_ids,
                )
                _, metadata = embeddings.load(
                    directory, manifest_ids=patient_ids
                )
                # The declaration the schema would permit for this init.
                embeddings.check_pairing(
                    metadata, kind="pooled", backbone="vit_b16", init=init,
                    geometry=geometry, checkpoint_sha256=checkpoint,
                    region_scheme=None,
                )


def test_the_pairing_guard_refuses_both_crossed_directions():
    """It asked one direction; the reverse would have loaded silently."""
    import tempfile

    import numpy as np

    from cleft import embeddings

    patient_ids = [1, 2, 3]

    def _artifact(tmp, init, variant, checkpoint):
        directory = pathlib.Path(tmp) / "set"
        embeddings.save(
            directory, np.zeros((3, 768), dtype=np.float32),
            backbone="vit_b16", backbone_kind="transformer",
            init=init, geometry="g1", variant=variant,
            checkpoint_sha256=checkpoint,
            patient_ids=patient_ids, manifest_ids=patient_ids,
        )
        return embeddings.load(directory, manifest_ids=patient_ids)[1]

    # (a) a foundation arm that declares a checkpoint it cannot have.
    with tempfile.TemporaryDirectory() as tmp:
        metadata = _artifact(tmp, "dino_in1k", None, None)
        with pytest.raises(
            embeddings.EmbeddingError, match="consumes no pretraining"
        ):
            embeddings.check_pairing(
                metadata, kind="pooled", backbone="vit_b16",
                init="dino_in1k", geometry="g1",
                checkpoint_sha256="b" * 64, region_scheme=None,
            )

    # (b) THE REVERSE, which nothing checked: the ARTIFACT carries a
    # checkpoint hash while the arm declares none.
    with tempfile.TemporaryDirectory() as tmp:
        metadata = _artifact(tmp, "scut_original", "original", "c" * 64)
        metadata["init"] = "dino_in1k"          # the set is not this init
        with pytest.raises(
            embeddings.EmbeddingError,
            match="came from a different init than the arm declares",
        ):
            embeddings.check_pairing(
                metadata, kind="pooled", backbone="vit_b16",
                init="dino_in1k", geometry="g1",
                checkpoint_sha256=None, region_scheme=None,
            )

    # (c) a ladder init still MUST declare, and the message names its variant.
    with tempfile.TemporaryDirectory() as tmp:
        metadata = _artifact(tmp, "scut_masked", "masked_g1", "d" * 64)
        with pytest.raises(embeddings.EmbeddingError, match="masked_g1"):
            embeddings.check_pairing(
                metadata, kind="pooled", backbone="vit_b16",
                init="scut_masked", geometry="g1",
                checkpoint_sha256=None, region_scheme=None,
            )


def test_the_reader_and_the_writer_key_on_the_same_fact():
    """The defect in one sentence: two rules for one question."""
    import inspect

    from cleft import embeddings

    writer = inspect.getsource(embeddings.save)
    reader = inspect.getsource(embeddings.check_pairing)
    assert "if variant is not None and not checkpoint_sha256:" in writer
    assert "if own_variant is not None:" in reader
    # The stale test is gone from the reader's CODE -- the dated comments
    # quote it on purpose, which is why this looks at code lines only.
    code = "\n".join(
        line for line in reader.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert 'init != "imagenet"' not in code
    assert 'if init == "imagenet":' not in code
