"""The embedding artifact: one format for both backbone kinds, one loader.

Phase 6 §4 produces **24 sets** -- 4 backbones x 3 inits x 2 geometries -- over
the 237 cleft patients, and Phase 7 turns every ladder arm into a ridge fit over
them. This module defines what one of those sets *is*.

----------------------------------------------------------------------------
ONE FORMAT, TWO LIVE KINDS (AND ONE SUPERSEDED)
----------------------------------------------------------------------------
The transformers and the graph models do not carry the same thing:

* ``pooled`` -- ``(n_patients, feature_dim)``. ViT-B/16 and Swin-B freeze
  entirely at cleft time, so one vector per image is everything downstream needs.
* ``feature_map`` -- ``(n_patients, channels, height, width)``: the frozen
  backbone's final feature map, which is the graph backbones' actual frozen
  boundary. SR-GNN and AG-Net **train everything after the backbone on cleft
  data**, and the map is the exact input that trainable stack consumes.

> **[DECIDED 2026-07-31] ``per_region`` is SUPERSEDED for graph extraction,
> and the reason is the ported forwards, not taste.** The original design --
> ``(n_patients, n_regions, 2048)`` -- was written before the verified
> implementations landed, and it cannot feed either of them:
>
> * SR-GNN's ``SeqSelfAttention`` consumes the **flattened** per-region
>   descriptor -- 2048x7x7 = 100,352 dims -- which is where 6.42M of the
>   transplant-verified model lives. A 2048-wide region vector deletes that
>   layer's input, which is **the exact mass the reconstruction famously
>   missed** (srgnn.py's 8.07M lesson, nearly repeated in an artifact schema).
> * AG-Net's trainable SAGAN self-attention operates on the feature map
>   **before regions exist at all**; per-region anything is downstream of a
>   layer that trains.
>
> The feature map is strictly better on every axis: it is the true frozen
> boundary for both models, it is smaller (237 x 2048 x 7 x 7 = ~95 MB per
> set against ~2.6 GB of flattened descriptors), and it is **scheme-free** --
> the region scheme is applied at cleft-train time through each patient's own
> recorded geometry, so schemes multiply ARMS, not extractions. The
> ``per_region`` kind stays in the closed set and its validation stays live
> (the loader may meet one), but no backbone kind maps to it.

**They share one container and one loader**, distinguished by a ``kind`` field.
Two loaders would be two places for the row-order bug to live, and row order is
the failure that matters most here: a silent misalignment trains every patient
against another patient's label and still produces a plausible number.

----------------------------------------------------------------------------
THE INIT/GEOMETRY RULE, ASSERTED RATHER THAN REMEMBERED
----------------------------------------------------------------------------
Three inits: ``imagenet``, ``scut_original``, ``scut_masked``. The last one is
**geometry-matched and never crossed** -- a masked-G1 checkpoint feeds G1 cleft
embeddings only, masked-G2 feeds G2 only.

**That constraint is the reason there are 12 pretraining runs rather than 8.**
The geometry arm compares a matched-G1 pipeline against a matched-G2 one; a
crossed pair would measure the geometry effect *without* the domain matching that
masking exists to provide, which is a different comparison than the one the arm
claims to make. So it is checked in code at write time, not left to whoever
assembles the run list.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: What an artifact holds. Closed set: a new kind has to be declared, not
#: inferred from an array's rank. ``per_region`` is superseded for graph
#: extraction (see the docstring) but stays validated: deleting a kind the
#: loader might meet would turn an old artifact into an unreadable one.
EMBEDDING_KINDS = ("pooled", "per_region", "feature_map", "region_vectors")

#: **[ADDED 2026-08-14] ``region_vectors`` -- (n_patients, n_regions,
#: feature_dim), and NOT a revival of the superseded ``per_region``.**
#:
#: The two have the same shape and opposite provenance, which is exactly the
#: confusion worth naming before it happens (R2: one name, two quantities).
#:
#: * ``per_region`` was superseded because it sliced regions OUT OF the
#:   feature map of a **whole-frame** pass. That deleted SR-GNN's
#:   ``SeqSelfAttention`` input -- the flattened 2048x7x7 descriptor where
#:   6.42M of the transplant-verified model lives -- and it baked the region
#:   scheme into extraction, re-coupling it to an axis the feature map
#:   decouples.
#: * ``region_vectors`` is Road B Branch 3: each region is a **pixel crop
#:   passed through the backbone on its own**, so the vector is a genuine
#:   whole-image embedding of that region rather than a slice of a map. The
#:   backbone never sees the whole face, which is the branch's premise, not
#:   an artefact of the format. Nothing downstream expects a 7x7 map here
#:   because nothing here pools one.
#:
#: **The regions list is mandatory and ordered**, for the reason the v1
#: staging manifests had to say which side their upsampling flags measured:
#: an artifact should say what its axes MEAN. A consumer must be able to ask
#: which column is ``philtral_column`` rather than trusting positional order.
REGION_VECTORS_IS_NOT_PER_REGION = {
    "added": "2026-08-14",
    "same_shape_opposite_provenance": (
        "per_region sliced a whole-frame feature map; region_vectors "
        "embeds each pixel crop independently"
    ),
    "why_per_region_was_superseded": (
        "it deleted SR-GNN's SeqSelfAttention input (the flattened "
        "2048x7x7 descriptor) and baked the scheme into extraction"
    ),
    "why_that_does_not_transfer": (
        "Branch 3's backbone never sees the whole face by design, and no "
        "consumer here pools a 7x7 map -- the graph arm takes the crop "
        "vectors as nodes directly"
    ),
    "regions_are_mandatory": (
        "an artifact should say what its axes mean; a consumer asks which "
        "column is philtral_column rather than trusting position"
    ),
}

#: Which kind each backbone kind produces. The mapping is here rather than in
#: ``factory`` so the artifact format owns its own invariant. Graph maps to
#: ``feature_map`` [DECIDED 2026-07-31]: the frozen boundary for both graph
#: models is the backbone's final map, and everything after it trains on
#: cleft data.
KIND_FOR_BACKBONE_KIND = {"transformer": "pooled", "graph": "feature_map"}

#: The three inits of the Phase 7 ladder. ``imagenet`` needs no checkpoint, which
#: is why 4 backbones x 3 inits x 2 geometries = 24 sets come from only 12
#: pretraining runs.
#:
#: **[2026-09-02] THIS TUPLE IS THE LADDER'S LATTICE AND STAYS CLOSED AT
#: THREE.** ``expected_set_count`` multiplies by ``len(INITS)`` and
#: ``embedding_plan`` iterates it to derive the required sets, so **adding a
#: member here silently derives new lattice cells that nobody designed** --
#: exactly what registering Phase 7D's backbones did to the ladder
#: (``factory.LADDER_BACKBONES``: 'registering Phase 7D's three backbones
#: for construction silently derived twenty-one new LADDER arms'). The
#: precedent's fix is the shape used below: **membership of a lattice and
#: membership of a vocabulary are different facts, and the iteration names
#: which one it means.**
INITS = ("imagenet", "scut_original", "scut_masked")

#: **[ADDED 2026-09-02] Inits that are NOT ladder inits.**
#:
#: A foundation checkpoint is a legitimate weight source and **not a variant
#: of this project's own pretraining**, which is what ``INITS`` is about:
#: ``VARIANT_FOR_INIT`` maps an init to one of OUR SCUT runs, and
#: ``check_init_geometry`` exists to stop a masked-G1 checkpoint feeding G2.
#: DINOv2 and DINO answer that question with "none of them" -- the same
#: answer ``imagenet`` gives, and a true one rather than a degenerate one.
#:
#: **They are named apart from ``imagenet`` because they are not ImageNet
#: weights** (DINOv2 is LVD-142M; DINO is ImageNet-1k WITHOUT labels), and
#: the set directory carries the init in its name. Three provenances under
#: one name would be the R2 shape (``phase25.THE_WEIGHTS_ARE_FACTORY_
#: PRETRAINED_LIKE_THE_PROBES``).
FOUNDATION_INITS = ("dinov2_lvd142m", "dino_in1k")

#: **The full vocabulary an ARTIFACT may carry**, as distinct from the
#: lattice above. Every init-valued config field is checked against THIS;
#: every count and every derived set iterates ``INITS``.
ALL_INITS = INITS + FOUNDATION_INITS

#: Which pretraining variant an init consumes. ``scut_masked`` is geometry-bound.
#: The foundation inits consume none, exactly as ``imagenet`` consumes none.
VARIANT_FOR_INIT = {
    "imagenet": None,
    "scut_original": "original",
    "scut_masked": "masked_{geometry}",
    "dinov2_lvd142m": None,
    "dino_in1k": None,
}

GEOMETRIES = ("g1", "g2")


class EmbeddingError(RuntimeError):
    """The embedding artifact is not valid."""


def expected_variant(init: str, geometry: str) -> str | None:
    """Which pretraining checkpoint this (init, geometry) pair must come from."""
    if init not in ALL_INITS:
        raise EmbeddingError(
            f"unknown init {init!r}; expected one of {ALL_INITS}. The first "
            f"{len(INITS)} are the LADDER's inits ({INITS}) and the rest are "
            f"foundation checkpoints ({FOUNDATION_INITS}), which consume no "
            "pretraining checkpoint of ours"
        )
    if geometry not in GEOMETRIES:
        raise EmbeddingError(
            f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}"
        )
    template = VARIANT_FOR_INIT[init]
    return None if template is None else template.format(geometry=geometry)


def check_init_geometry(init: str, geometry: str, variant: str | None) -> dict:
    """**The masked init and its geometry must agree.** Refuses if they do not.

    A crossed pair -- a masked-G1 checkpoint feeding G2 embeddings -- would still
    run, still produce 237 vectors, and still fit a ridge. It would simply be
    measuring something nobody designed: the geometry arm's comparison depends on
    each side being domain-matched, and crossing removes exactly that.
    """
    wanted = expected_variant(init, geometry)
    if variant != wanted:
        raise EmbeddingError(
            f"init {init!r} at geometry {geometry!r} must come from the "
            f"{wanted!r} checkpoint, not {variant!r}.\n"
            "  A masked init is GEOMETRY-BOUND: masked-G1 feeds G1 embeddings "
            "only, masked-G2 feeds G2 only, never crossed. That constraint is "
            "why there are 12 pretraining runs rather than 8 -- the geometry arm "
            "compares a matched-G1 pipeline against a matched-G2 one, and a "
            "crossed pair measures the geometry effect WITHOUT the domain "
            "matching that masking exists to provide."
        )
    return {"init": init, "geometry": geometry, "variant": variant}


def _validate(
    values: np.ndarray,
    kind: str,
    patient_ids: list,
    regions: list | None,
) -> dict:
    if kind not in EMBEDDING_KINDS:
        raise EmbeddingError(f"unknown kind {kind!r}; expected one of {EMBEDDING_KINDS}")

    array = np.asarray(values)
    if kind == "pooled":
        if array.ndim != 2:
            raise EmbeddingError(
                f"a pooled artifact must be (n_patients, feature_dim); got "
                f"shape {array.shape}"
            )
        if regions:
            raise EmbeddingError(
                "a pooled artifact carries no region geometry, but regions were "
                "supplied. If these are per-region features, declare the kind."
            )
        n_regions, feature_dim = None, int(array.shape[1])
    elif kind == "feature_map":
        if array.ndim != 4:
            raise EmbeddingError(
                f"a feature_map artifact must be (n_patients, channels, "
                f"height, width); got shape {array.shape}"
            )
        if regions:
            raise EmbeddingError(
                "a feature_map artifact carries no region geometry: the map "
                "PRECEDES regions, and the scheme is applied at cleft-train "
                "time through each patient's own recorded geometry. Baking a "
                "region set in here would re-couple extraction to the scheme "
                "axis, which is exactly what the feature map decouples."
            )
        n_regions, feature_dim = None, int(array.shape[1])
    else:
        if array.ndim != 3:
            raise EmbeddingError(
                f"a {kind} artifact must be (n_patients, n_regions, "
                f"feature_dim); got shape {array.shape}"
            )
        if not regions:
            raise EmbeddingError(
                f"a {kind} artifact must carry the region geometry that "
                "produced it. Without it the array is a block of numbers whose "
                "columns cannot be attributed to anatomy, and the graph layers "
                "that consume it cannot be checked against the regions they "
                "were trained on."
            )
        if len(regions) != array.shape[1]:
            raise EmbeddingError(
                f"{len(regions)} region definitions against {array.shape[1]} "
                "region columns; the geometry does not describe the array"
            )
        n_regions, feature_dim = int(array.shape[1]), int(array.shape[2])

    if len(patient_ids) != array.shape[0]:
        raise EmbeddingError(
            f"{len(patient_ids)} patient ids against {array.shape[0]} rows"
        )
    if len(set(patient_ids)) != len(patient_ids):
        raise EmbeddingError("patient ids contain duplicates; row order is ambiguous")

    shape = {"n_patients": int(array.shape[0]), "n_regions": n_regions,
             "feature_dim": feature_dim}
    if kind == "feature_map":
        shape["map_size"] = [int(array.shape[2]), int(array.shape[3])]
    return shape


def assert_row_order(patient_ids, manifest_ids) -> None:
    """**Row order must match the manifest, exactly and in order.**

    The brief's own warning, and it is the sharpest failure available here: a
    silent misalignment trains every patient against another patient's label and
    **still produces a plausible number**. Nothing downstream would flag it --
    the shapes are right, the ridge fits, the PCC is merely wrong.

    Same-set-different-order is checked separately from missing-or-extra, because
    the two have different causes and the message should say which.
    """
    ours, theirs = list(patient_ids), list(manifest_ids)
    if ours == theirs:
        return
    if set(ours) == set(theirs):
        first = next(
            i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b
        )
        raise EmbeddingError(
            f"row order does not match the manifest: same {len(ours)} patients, "
            f"different order, first difference at row {first} "
            f"({ours[first]!r} against {theirs[first]!r}). Every patient would "
            "be trained against another patient's label, and the result would "
            "look entirely plausible."
        )
    missing = sorted(set(theirs) - set(ours))
    extra = sorted(set(ours) - set(theirs))
    raise EmbeddingError(
        f"row set does not match the manifest: {len(missing)} missing "
        f"{missing[:5]}, {len(extra)} unexpected {extra[:5]}"
    )


def check_pairing(
    metadata: dict,
    *,
    kind: str,
    backbone: str,
    init: str,
    geometry: str,
    checkpoint_sha256: str | None,
    region_scheme: str | None = None,
    expect_randomised: bool = False,
) -> None:
    """**The artifact must BE the arm's declared representation, exactly.**

    One checker for both consuming paths. The graph arm wants a
    ``feature_map`` and the transformer ladder wants a ``pooled`` set, and
    everything else they must verify is identical -- so writing it twice would
    put the same invariant in two places for one of them to drift out of.
    That is the ``is_absolute_path`` / CuBLAS shape, twice recorded, and the
    scheme rule below is exactly the sort of clause that would have been
    updated in one copy only.

    What it refuses, and why each one is silent otherwise:

    * **the wrong kind** -- a pooled artifact has already discarded what the
      graph layers train on; a feature map fed to a linear head would be
      flattened into 100,352 columns for 237 patients, which is memorisation
      rather than a probe. Both run. Neither raises on its own.
    * **backbone / init / geometry disagreement** -- the config and the
      artifact would be describing different arms while the run reports the
      config's.
    * **scheme inconsistency** -- a graph checkpoint fine-tuned under one
      region scheme has scheme-shaped weights; ``region_scheme=None`` declares
      an arm with no scheme axis at all (the transformers), and an artifact
      carrying a scheme there is a graph set reaching a transformer arm.
    * **checkpoint mismatch** -- the sharpest one. Features from one
      representation with weights from another train against inputs they never
      see again, and nothing downstream would notice.

    * **a randomised control reaching a real arm, or a real set standing in
      for the control** -- ``expect_randomised`` is checked in BOTH
      directions, and both are silent. A training arm fed noise-backbone
      features would fit, score badly, and be read as a result about the
      architecture. A control fed the REAL features would report the
      explanation unchanged under randomisation -- similarity 1.0, the worst
      verdict -- from a test that was never run. That second direction is
      the one ``phase8.COMPONENT_ROLE_IS_A_NO_OP`` already produced once.

    Row order is NOT checked here: ``load`` asserts it against the manifest,
    which is the read the consuming arm actually uses.
    """
    randomisation = metadata.get("randomisation")
    is_randomised = bool((randomisation or {}).get("randomised"))
    if is_randomised != bool(expect_randomised):
        if is_randomised:
            raise EmbeddingError(
                "this is a RANDOMISED-BACKBONE control set and the consumer "
                "did not ask for one. Its features come from a backbone "
                f"whose {randomisation.get('n_parameters', 0):,} parameters "
                "are noise; an arm training on them would fit, score badly, "
                "and be read as evidence about the architecture "
                "(phase8.COMPONENT_ROLE_IS_A_NO_OP)"
            )
        raise EmbeddingError(
            "the consumer asked for a RANDOMISED-BACKBONE control and this "
            "set is the real representation. A control that is secretly the "
            "real set reports the explanation unchanged under randomisation "
            "-- similarity 1.0, the worst verdict -- from a test that never "
            "ran"
        )

    if metadata.get("kind") != kind:
        if kind == "feature_map":
            detail = (
                "A pooled artifact has already discarded what the graph layers "
                "train on."
            )
        else:
            detail = (
                "A feature_map artifact precedes pooling: flattened into a "
                "linear head it would be 100,352 columns over 237 patients."
            )
        raise EmbeddingError(
            f"this arm consumes {kind} artifacts; this one is "
            f"{metadata.get('kind')!r}. {detail}"
        )

    for key, wanted in (
        ("backbone", backbone), ("init", init), ("geometry", geometry),
    ):
        if metadata.get(key) != wanted:
            raise EmbeddingError(
                f"artifact {key} is {metadata.get(key)!r} but the arm declares "
                f"{wanted!r}; the config and the artifact disagree about what "
                "this arm is"
            )

    recorded_scheme = metadata.get("pretrain_scheme")
    # **[2026-09-02] Keyed on the VARIANT, not on the name.** An init with no
    # pretraining of ours has no scheme to be consistent with -- which is
    # true of ``imagenet`` and equally of the foundation inits. It read
    # ``init == "imagenet"`` and the foundation case fell through to the
    # branch below and passed BY LUCK; luck is not a check.
    own_variant = expected_variant(init, geometry)
    if own_variant is None:
        if recorded_scheme is not None:
            raise EmbeddingError(
                f"the artifact carries pretrain_scheme {recorded_scheme!r} "
                f"but init {init!r} has no pretraining of ours to have a "
                "scheme"
            )
        # Any cleft scheme is legitimate on an imagenet init -- there is no
        # pretraining structure to be consistent WITH. Recorded, not implied.
    elif region_scheme is None:
        # An arm with no scheme axis. A transformer set records None; anything
        # else means a graph checkpoint's map has reached a transformer arm.
        if recorded_scheme is not None:
            raise EmbeddingError(
                f"the artifact carries pretrain_scheme {recorded_scheme!r} but "
                f"the {backbone!r} arm has no scheme axis. A scheme is a graph "
                "backbone's node structure; a set carrying one did not come "
                "from this arm's pretraining."
            )
    elif recorded_scheme != region_scheme:
        raise EmbeddingError(
            f"scheme consistency: the artifact's map came from the "
            f"{recorded_scheme!r} checkpoint but the arm trains under "
            f"{region_scheme!r}. The graph layers must see the same node "
            "structure in pretraining and cleft fine-tuning."
        )

    # **[2026-09-02] THE READER NOW KEYS ON THE WRITER'S OWN RULE.**
    #
    # ``save`` requires a checkpoint hash ``if variant is not None`` -- i.e.
    # when the set came from one of OUR pretraining runs. This read
    # ``init != "imagenet"``, and the two disagreed the moment an init
    # existed that is neither imagenet nor ours: **the artifact recorded
    # ``checkpoint_sha256: None`` legitimately, and the guard demanded a
    # declaration for a checkpoint that does not exist.**
    #
    # A foundation checkpoint's weights are timm's, downloaded at extraction
    # from an external source, and **this codebase has no hash of them** --
    # the same limitation ``imagenet`` has always had and which nothing
    # stated (``phase25.WHAT_THE_TWO_INITS_CAN_HONESTLY_DECLARE``). The
    # guard is not weakened for them: it asks for what exists.
    if own_variant is not None:
        if checkpoint_sha256 is None:
            raise EmbeddingError(
                f"init {init!r} comes from the {own_variant!r} checkpoint and "
                "the artifact must be traced to it, and none was declared"
            )
        if metadata.get("checkpoint_sha256") != checkpoint_sha256:
            raise EmbeddingError(
                "PAIRING: the artifact's features came from checkpoint "
                f"{metadata.get('checkpoint_sha256')!r} but the declared "
                f"checkpoint hashes {checkpoint_sha256!r}. Training against "
                "features from a DIFFERENT representation than the one "
                "declared would run to completion and nothing downstream "
                "would raise."
            )
    else:
        if checkpoint_sha256 is not None:
            raise EmbeddingError(
                f"the {init!r} arm declared checkpoint {checkpoint_sha256!r}, "
                "but this init consumes no pretraining checkpoint of ours, "
                "so the declaration describes a run this one is not"
            )
        # **The reverse direction, which nothing checked.** If the ARTIFACT
        # carries a checkpoint hash while the arm declares none, the set did
        # not come from this init -- and it would have loaded silently.
        if metadata.get("checkpoint_sha256") is not None:
            raise EmbeddingError(
                f"the artifact was traced to checkpoint "
                f"{metadata['checkpoint_sha256']!r} but init {init!r} "
                "consumes no pretraining checkpoint of ours. The set came "
                "from a different init than the arm declares."
            )


def assert_no_test_fold_leak(metadata: dict, test_patient_ids) -> dict:
    """**The fold's test patients must not have contributed to its features.**

    Per-fold BN re-extraction re-estimates the backbone's running statistics
    on cleft data, which makes every feature vector depend on WHICH patients
    were adapted on. Adapt on all 237 and the test fold's own statistics are
    inside the representation used to predict it -- an arm that leaks that way
    produces a better number, with 237 rows, the right shapes, a valid hash
    and every gate passing. Nothing downstream can see it.

    So it is checked here, at consumption, against the ids the artifact
    recorded. Not arranged at build time and trusted: the build and the
    consumption are different runs on different machines, and the one that
    reports the number is the one that has to be sure.

    An ordinary set (no adaptation) passes trivially and says so -- it was
    adapted on nothing, so it can leak nothing.
    """
    adapted = metadata.get("adapted_on_patient_ids")
    if not adapted:
        return {
            "checked": False,
            "why": "this set was adapted on nothing, so it cannot leak",
            "fold": metadata.get("fold"),
        }

    test_ids = set(int(pid) for pid in test_patient_ids)
    adapted_ids = set(int(pid) for pid in adapted)
    overlap = sorted(test_ids & adapted_ids)
    if overlap:
        raise EmbeddingError(
            f"TEST-FOLD LEAK: {len(overlap)} of this fold's {len(test_ids)} "
            f"test patients contributed to the BatchNorm statistics baked "
            f"into its features, e.g. {overlap[:5]}. The arm would predict "
            "those patients with a representation calibrated partly on them, "
            "score better for it, and produce a completely plausible "
            "metrics.json. Re-extract adapting on the TRAINING fold only "
            "(extract.PER_FOLD_REEXTRACTION)."
        )
    return {
        "checked": True,
        "fold": metadata.get("fold"),
        "n_test_patients": len(test_ids),
        "n_adapted_on": len(adapted_ids),
        "overlap": 0,
    }


#: The shapes a consumer may ask a set for. One artifact serves three, so
#: the consumer DECLARES which -- see ``as_consumed``.
CONSUMPTIONS = ("pooled", "concat", "nodes")


def as_consumed(values: np.ndarray, metadata: dict, consume: str) -> np.ndarray:
    """One set, in the shape its consumer declared it needs.

    **[DECIDED 2026-08-14] The reshape lives HERE -- in the module that owns
    the artifact format -- and the CONSUMER declares it.**

    A ``region_vectors`` set is ``(N, R, D)`` and three consumers want three
    things from it: the concat arms want ``(N, R*D)``, the graph arm wants
    ``R`` nodes of ``D``, and the whole-frame control is already ``(N, D)``.
    Four shapes, and the wrong pairing is silent -- a graph arm handed a
    flattened vector still trains and still reports a number.

    **Why not infer it from the array.** "3-D means flatten" would give the
    graph arm a flattened set the moment someone reused this loader, and the
    failure would be a plausible number rather than an error. The
    declaration is what makes a mismatch loud.

    **Why not at extraction.** Storing flattened would discard the structure
    the graph arm needs while saving the concat arms nothing: ``(N, R*D)``
    is a VIEW of ``(N, R, D)``, not a different measurement. One artifact,
    three readings.

    **The flatten order is part of the contract**: C order, so the result is
    ``R`` contiguous blocks of ``D``, in ``metadata["regions"]`` order. That
    list is what makes a flattened vector interpretable at all -- block
    ``i`` is region ``regions[i]`` -- and it is why ``region_vectors``
    refuses to be written without names.
    """
    if consume not in CONSUMPTIONS:
        raise EmbeddingError(
            f"unknown consumption {consume!r}; expected one of {CONSUMPTIONS}"
        )
    array = np.asarray(values)
    kind = metadata.get("kind")

    if consume == "pooled":
        if kind != "pooled" or array.ndim != 2:
            raise EmbeddingError(
                f"consume='pooled' needs a pooled (n, dim) set; this is "
                f"{kind!r} with shape {array.shape}"
            )
        return array

    if kind != "region_vectors" or array.ndim != 3:
        raise EmbeddingError(
            f"consume={consume!r} needs a region_vectors (n, regions, dim) "
            f"set; this is {kind!r} with shape {array.shape}"
        )
    regions = metadata.get("regions") or []
    if len(regions) != array.shape[1]:
        raise EmbeddingError(
            f"{len(regions)} region names against {array.shape[1]} region "
            "columns; the set cannot be indexed by name"
        )
    if consume == "nodes":
        return array
    return array.reshape(array.shape[0], array.shape[1] * array.shape[2])


def save(
    directory,
    values: np.ndarray,
    *,
    backbone: str,
    backbone_kind: str,
    init: str,
    geometry: str,
    variant: str | None,
    checkpoint_sha256: str | None,
    patient_ids: list,
    manifest_ids: list | None = None,
    regions: list | None = None,
    pretrain_scheme: str | None = None,
    fold: int | None = None,
    adapted_on_patient_ids: list | None = None,
    bn_reestimation: dict | None = None,
    block: int | None = None,
    token: str | None = None,
    kind: str | None = None,
    randomisation: dict | None = None,
) -> dict:
    """Write one embedding set. Every invariant is checked before anything lands.

    Order matters: the init/geometry rule and the row order are checked *first*,
    so a crossed pair or a misaligned artifact never reaches disk to be consumed
    by something that trusts it.

    ``kind`` overrides the backbone-derived kind, and the ONLY override
    allowed is ``region_vectors`` (``REGION_VECTORS_IS_NOT_PER_REGION``).
    Road B Branch 3 crops regions BEFORE the backbone, so what a
    transformer produces there is 22 vectors per patient rather than one --
    the shape follows the input, not the backbone kind. Every other caller
    keeps the derived kind, so the invariant that a graph backbone writes
    feature maps is untouched.
    """
    expected_kind = KIND_FOR_BACKBONE_KIND.get(backbone_kind)
    if expected_kind is None:
        raise EmbeddingError(
            f"unknown backbone kind {backbone_kind!r}; expected one of "
            f"{sorted(KIND_FOR_BACKBONE_KIND)}"
        )
    if kind is not None:
        if kind != "region_vectors":
            raise EmbeddingError(
                f"kind={kind!r} is not an allowed override; only "
                "'region_vectors' may differ from the backbone-derived "
                f"kind ({expected_kind!r}), because only Branch 3 crops "
                "regions before the backbone"
            )
        expected_kind = kind

    check_init_geometry(init, geometry, variant)
    if manifest_ids is not None:
        assert_row_order(patient_ids, manifest_ids)
    shape = _validate(values, expected_kind, patient_ids, regions)

    if variant is not None and not checkpoint_sha256:
        raise EmbeddingError(
            f"init {init!r} comes from the {variant!r} checkpoint, but no "
            "checkpoint hash was given. An embedding set whose checkpoint is "
            "unrecorded cannot be traced back to the run that produced it."
        )

    directory = Path(directory)
    if directory.exists():
        raise EmbeddingError(
            f"{directory} already exists. Data artifacts are immutable "
            "(PLAN §2.6): create a new version, never overwrite."
        )
    directory.mkdir(parents=True)

    np.save(directory / "values.npy", np.asarray(values))
    metadata = {
        "kind": expected_kind,
        "backbone": backbone,
        "backbone_kind": backbone_kind,
        "init": init,
        "geometry": geometry,
        "variant": variant,
        #: [DECIDED 2026-07-31] Which pretraining scheme's checkpoint this map
        #: came from. A graph backbone fine-tuned under a scheme has
        #: scheme-shaped WEIGHTS, and the cleft arm consuming this set must
        #: train its graph layers under the SAME scheme -- the consistency
        #: rule. Recorded here so the cleft loader can refuse a mismatch
        #: rather than trust the run list. None for imagenet (no pretraining)
        #: and for transformers (no scheme axis).
        "pretrain_scheme": pretrain_scheme,
        #: **Per-fold BN re-extraction (``extract.PER_FOLD_REEXTRACTION``).**
        #: ``fold`` is the fold this set is FOR -- the one whose test patients
        #: must NOT have contributed -- and ``adapted_on_patient_ids`` is the
        #: list that must be checked against them. Both None for an ordinary
        #: set, which is adapted on nothing and valid for every fold.
        #:
        #: The ids are stored rather than a count or a hash, because the
        #: consuming arm has to compute an intersection with the test fold.
        #: A count would let a leak through while looking accounted for.
        "fold": fold,
        "adapted_on_patient_ids": (
            list(adapted_on_patient_ids) if adapted_on_patient_ids else None
        ),
        "bn_reestimation": bn_reestimation,
        #: **Arm B's component-role CONTROL, marked in the artifact itself.**
        #: ``extract.RANDOMISED_BACKBONE``: the features come from a backbone
        #: whose parameters are noise, and the set exists only so arm B's
        #: explanation can be tested against a destroyed representation.
        #: ``check_pairing`` refuses it for any consumer that did not ask for
        #: one, and refuses a real set for a consumer that did -- both
        #: directions, because both are silent otherwise. None for every
        #: ordinary set, which is what the ladder consumes.
        "randomisation": randomisation,
        #: **Phase 7B's pooling axis, recorded so a trial's provenance names
        #: which representation it used.** Two sets from one checkpoint at
        #: different depths are different representations; without these a
        #: reader meeting a set has the directory name and nothing that
        #: confirms it. ``final_norm_at_every_depth`` is recorded because the
        #: axis would otherwise vary depth AND normalisation together, and
        #: which of those a difference came from is not recoverable later.
        "block": None if block is None else int(block),
        "token": token,
        "final_norm_at_every_depth": None if block is None else True,
        "checkpoint_sha256": checkpoint_sha256,
        "patient_ids": list(patient_ids),
        "regions": list(regions) if regions else None,
        **shape,
        "row_order_checked_against_manifest": manifest_ids is not None,
        "note": (
            "ONE FORMAT: 'pooled' is (n_patients, feature_dim) for the frozen "
            "transformers; 'feature_map' is (n_patients, channels, height, "
            "width) -- the frozen backbone boundary for the graph models, "
            "whose entire post-backbone stack trains on cleft data and "
            "consumes the map (SR-GNN's self-attention takes the FLATTENED "
            "per-region descriptor; AG-Net's SAGAN precedes regions). The "
            "region scheme is applied at cleft-train time, never baked in. "
            "A masked init is GEOMETRY-BOUND and never crossed."
        ),
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load(directory, manifest_ids: list | None = None) -> tuple[np.ndarray, dict]:
    """Read one embedding set, re-checking what was checked at write time.

    **One loader for both kinds.** Two would be two places for the row-order bug
    to live, and Phase 7 consumes both.

    The invariants are re-asserted rather than trusted: an artifact can be
    correct when written and wrong when read -- edited, truncated, or produced by
    an older writer -- and this is the last point at which anything checks before
    a number comes out the other end.
    """
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        raise EmbeddingError(f"{directory} has no metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    values = np.load(directory / "values.npy")
    _validate(
        values, metadata["kind"], metadata["patient_ids"], metadata.get("regions")
    )
    check_init_geometry(
        metadata["init"], metadata["geometry"], metadata["variant"]
    )
    if manifest_ids is not None:
        assert_row_order(metadata["patient_ids"], manifest_ids)
    return values, metadata


def expected_set_count(n_backbones: int = 4) -> dict:
    """The 24 and the 12, derived so the arithmetic is visible rather than stated.

    **Runs are counted per VARIANT, not per init**, and the two are not the same
    number -- which is the whole reason 12 rather than 8. There are three inits,
    but ``scut_masked`` is geometry-bound and therefore needs a *separate
    checkpoint per geometry*, so the distinct variants are ``original``,
    ``masked_g1`` and ``masked_g2``. ``imagenet`` needs none.

    Deriving it from ``expected_variant`` rather than writing ``3`` means the
    count follows the rule if the rule ever changes, instead of agreeing with it
    by coincidence. The first version of this function multiplied by
    ``len(INITS) - 1`` and returned 8.
    """
    variants = {
        expected_variant(init, geometry)
        for init in INITS
        for geometry in GEOMETRIES
    }
    variants.discard(None)
    return {
        "backbones": n_backbones,
        "inits": len(INITS),
        "geometries": len(GEOMETRIES),
        "variants": sorted(variants),
        "embedding_sets": n_backbones * len(INITS) * len(GEOMETRIES),
        "pretraining_runs": n_backbones * len(variants),
        "note": (
            "24 sets from 12 runs. Runs are per VARIANT, not per init: "
            "'imagenet' needs no checkpoint, 'scut_original' needs one, and "
            "'scut_masked' needs one PER GEOMETRY because it is geometry-bound "
            "-- masked-G1 for G1, masked-G2 for G2, never crossed."
        ),
    }
