"""Pooled embedding artifacts as a cleft feature source: the Phase 7 entry.

**The hole this closes.** ``phase3.prepare_features`` extracts embeddings
LIVE, by running ``torch_backbone.extract_embeddings`` over the staged images
-- and that path builds ViT-B/16 at ImageNet init and nothing else. So every
transformer arm the ladder needs at ``scut_original`` or ``scut_masked`` had
no way to reach its own representation: the twelve pooled sets extraction
produced (run p6-extract) were unconsumable by the task meant to consume them.
Stage D is 12 arms and 8 of them are pretrained inits, so this was most of the
ladder.

The graph path solved the same problem for feature maps in ``graph_cleft``.
This is its mirror for pooled sets, and deliberately not a second
implementation of it: the pairing rules are ``embeddings.check_pairing``,
called by both.

----------------------------------------------------------------------------
WHAT IT VERIFIES, AND WHY EACH FAILURE IS SILENT WITHOUT IT
----------------------------------------------------------------------------
* **row order against the manifest** -- ``embeddings.load`` asserts it, using
  the patient list the ARM read, not a second read of its own. The brief's own
  warning and the sharpest failure available: a misalignment trains every
  patient against another patient's label and still produces a plausible PCC.
* **checkpoint hash equality** -- the artifact's recorded
  ``checkpoint_sha256`` against the declared checkpoint input's VERIFIED hash
  (guard 3 recomputed it before the run began). An arm claiming
  ``scut_masked`` while consuming ``scut_original`` vectors fits a ridge just
  as well.
* **init and geometry agreement** -- including the geometry-bound masked rule,
  which ``embeddings.load`` re-asserts from the artifact's own side.
* **no scheme** -- a transformer has no region-scheme axis, so a set carrying
  a ``pretrain_scheme`` is a graph artifact that has reached the wrong arm.
* **the kind** -- ``pooled``, not ``feature_map``. Flattening a map into a
  linear head would give 100,352 columns over 237 patients, which is
  memorisation wearing a probe's name.

----------------------------------------------------------------------------
THE PARAMETER REPORT SAYS THE EXTRACTOR IS ELSEWHERE
----------------------------------------------------------------------------
An artifact-fed arm has **no frozen extractor in this run** -- the backbone
ran in the extraction run and is not instantiated here. That is why the
feature kind is its own string rather than ``frozen_backbone_embeddings``:
reusing that one would make ``assert_parameters_match_policy`` demand an
in-run backbone parameter count that does not exist, and inventing one would
be reporting a model this run never built.

**It does not follow that the check should skip.** A new feature kind quietly
falling out of the parameter check is precisely the 2026-07-28 defect (PLAN
R7, instance 4: a check keyed to one exact feature string, silently exempting
every patch arm). So this kind gets its OWN necessary condition, and a
stronger one than the original: a linear head over a D-dim embedding has
exactly ``D + 1`` parameters, so the reported trainable count is checked
against the artifact's own ``feature_dim``. If that report ever carried a
backbone's 86M again, this fires.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import embeddings

#: **What a ladder config declares to reach this path.** Written down because
#: the twelve Stage D arms plus Stage C's four are otherwise an excavation
#: through three modules, and because two of these fields are refused at LOAD
#: time when they disagree (``config.schema._check_train_cv_init``).
#:
#: ::
#:
#:     task:
#:       kind: train_cv
#:       backbone: vit_b16          # must equal the artifact's
#:       init: scut_masked          # imagenet | scut_original | scut_masked
#:       geometry: g2               # masked inits are geometry-BOUND
#:       embeddings_artifact: <input name of ONE pooled set directory>
#:       checkpoint: <input name of the pretraining checkpoint>
#:       trainable: head
#:
#: ``init: imagenet`` takes neither ``embeddings_artifact`` nor ``checkpoint``
#: and runs live extraction, which is the Phase 3 path unchanged.
#:
#: **The hashes come from ``scripts/declare_inputs.py`` or a run's own
#: recorded metrics, never from a chat log**, and no shipped ladder config
#: exists yet for exactly that reason: the embedding sets and the pretraining
#: run directories live on the cluster, so their rollups and their directory
#: NAMES cannot be produced here. Inventing a run-directory path would be
#: worse than an obviously-empty hash, because it reads as real provenance.
LADDER_CONFIG_SHAPE = {
    "task_fields": (
        "backbone", "init", "geometry", "embeddings_artifact", "checkpoint",
        "trainable",
    ),
    "imagenet_declares": ("backbone", "geometry", "trainable"),
    "refused_at_load": (
        "a pretrained init without an embeddings_artifact (would train on "
        "ImageNet vectors while claiming the pretraining); a pretrained init "
        "without a checkpoint (nothing to pair the artifact against); an "
        "imagenet init WITH a checkpoint; an artifact beside a patch_scheme"
    ),
    "hashes_from": "scripts/declare_inputs.py or a run's own recorded metrics",
}

#: What an artifact-fed arm reports under ``features``. Distinct from
#: ``phase3.WHOLE_IMAGE_EMBEDDINGS`` because the provenance genuinely differs:
#: there is no extractor in this run to account for, and the checkpoint the
#: vectors came from is recorded instead.
PRECOMPUTED_EMBEDDINGS = "precomputed_backbone_embeddings"


class PooledFeatureError(RuntimeError):
    """The pooled artifact is not the arm's declared representation."""


@dataclass(frozen=True)
class PooledSource:
    """One arm's declaration of which pooled set it consumes.

    Every field is checked against the artifact rather than trusted, which is
    the whole reason this is a declaration and not just a path.
    """

    directory: Path
    backbone: str
    init: str
    geometry: str
    #: The declared checkpoint input's VERIFIED hash -- guard 3's own
    #: recomputation, never the config's declaration. None for imagenet, which
    #: has no pretraining checkpoint and where declaring one is refused.
    checkpoint_sha256: str | None = None
    #: **How this arm reads the set** (``embeddings.as_consumed``).
    #: "pooled" is every Road A arm: the artifact is already
    #: (n_patients, dim). "concat" is Road B Branch 3, where the artifact
    #: is a ``region_vectors`` set of (n, regions, dim) and the arm trains
    #: on the flattened (n, regions*dim). Declared rather than inferred:
    #: the same set also feeds a graph arm as nodes, and a silently
    #: flattened graph arm would train and report a number.
    consume: str = "pooled"


def load_features(
    source: PooledSource, patient_ids: list[int]
) -> tuple[np.ndarray, dict]:
    """The (n_patients, feature_dim) matrix this arm trains on, verified.

    ``patient_ids`` must be the order the arm's own manifest read produced, so
    the row-order assertion compares the artifact against the labels that will
    actually be paired with it.
    """
    directory = Path(source.directory)
    if not directory.is_dir():
        raise PooledFeatureError(
            f"{directory} is not a directory. A pooled arm consumes ONE "
            "embedding set directory (the one holding values.npy and "
            "metadata.json), not the embeddings_v1 root."
        )

    expected_kind = "pooled" if source.consume == "pooled" else "region_vectors"
    try:
        values, metadata = embeddings.load(directory, manifest_ids=patient_ids)
        embeddings.check_pairing(
            metadata,
            kind=expected_kind,
            backbone=source.backbone,
            init=source.init,
            geometry=source.geometry,
            checkpoint_sha256=source.checkpoint_sha256,
            # A transformer has no scheme axis; a set carrying one is a graph
            # artifact that has reached the wrong arm.
            region_scheme=None,
        )
        values = embeddings.as_consumed(values, metadata, source.consume)
    except embeddings.EmbeddingError as exc:
        raise PooledFeatureError(str(exc)) from exc

    report = {
        "features": PRECOMPUTED_EMBEDDINGS,
        "shape": [int(v) for v in values.shape],
        # **The width the HEAD sees**, which is what the D+1 parameter
        # check compares against. For a concat arm that is regions*dim,
        # not the artifact's own per-region dim -- reporting the latter
        # would make the check fire on a correct arm.
        "feature_dim": int(values.shape[1]),
        "consume": source.consume,
        "artifact_feature_dim": int(metadata["feature_dim"]),
        "embedding_set": directory.name,
        "artifact_kind": metadata["kind"],
        "backbone": metadata["backbone"],
        "init": metadata["init"],
        "geometry": metadata["geometry"],
        "variant": metadata["variant"],
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        # The row order was asserted against THIS run's manifest read, not
        # against the artifact's own recorded copy of it.
        "row_order_asserted_against_manifest": True,
        # There is no extractor in this run to account for. Reported as zero
        # explicitly rather than omitted: an absent key and a measured zero
        # read identically to a consumer, and only one of them is a claim.
        "backbone_parameters": 0,
        "backbone_trainable_parameters": 0,
        "extractor_note": (
            "the backbone ran in the EXTRACTION run, not this one -- the "
            "vectors are precomputed and the checkpoint they came from is "
            "recorded above. This arm instantiates no backbone, so it reports "
            "no in-run backbone parameters rather than inventing a count."
        ),
    }
    return values, report


def load_concat_features(
    sources: list[PooledSource], patient_ids: list[int]
) -> tuple[np.ndarray, dict]:
    """Arm 4: several pooled sets, verified INDIVIDUALLY, concatenated.

    Every constituent goes through ``load_features`` -- the same row-order
    assertion against this run's manifest read, the same pairing check
    against its OWN declared backbone -- so the concat arm cannot be a way
    to smuggle an unverified set past the checks. The concatenation itself
    is ``np.concatenate`` on axis 1, which is exactly what Phase 7B's
    ``combine(kind="concat")`` reduces to; the head-width check then holds
    over the SUMMED dimension automatically (D+1 over the concat width).

    **The 7B prior rides in the report** rather than being left for the
    reader to remember: embedding concatenation measured +0.0221
    inner-validation and -0.0542 out-of-fold on this cohort
    (``phase7b.SEARCH_AXIS_VERDICTS``), and an arm carrying a measured prior
    against it must say so where its result will be read.
    """
    if len(sources) < 2:
        raise PooledFeatureError(
            "a concat arm needs at least two sets; one set concatenated is "
            "that set under a different arm's name"
        )
    blocks, reports = [], []
    for source in sources:
        values, report = load_features(source, patient_ids)
        blocks.append(values)
        reports.append(report)
    heights = {block.shape[0] for block in blocks}
    if len(heights) != 1:
        raise PooledFeatureError(
            f"the constituent sets disagree on patient count: {sorted(heights)}. "
            "Row order was asserted per set, so this means the manifest reads "
            "diverged -- refusing to align by truncation."
        )
    features = np.concatenate(blocks, axis=1)
    report = {
        "features": PRECOMPUTED_EMBEDDINGS,
        "shape": [int(v) for v in features.shape],
        "feature_dim": int(features.shape[1]),
        "consume": "concat_across_artifacts",
        "constituents": [
            {
                "embedding_set": entry["embedding_set"],
                "backbone": entry["backbone"],
                "feature_dim": entry["feature_dim"],
            }
            for entry in reports
        ],
        "concatenation_order": [entry["embedding_set"] for entry in reports],
        "row_order_asserted_against_manifest": True,
        "backbone_parameters": 0,
        "backbone_trainable_parameters": 0,
        "extractor_note": (
            "the backbones ran in their EXTRACTION runs, not this one; each "
            "constituent's provenance is under 'constituents'"
        ),
        "measured_prior": (
            "phase7b.SEARCH_AXIS_VERDICTS: embedding concatenation measured "
            "+0.0221 inner-validation and -0.0542 out-of-fold on this "
            "cohort. A prior against this arm, stated where its result is "
            "read"
        ),
    }
    return features, report


def assert_head_matches_embedding_width(
    feature_dim: int | None, trainable_parameters: int
) -> None:
    """A linear head over D-dim embeddings has exactly ``D + 1`` parameters.

    The necessary condition that keeps this feature kind inside the parameter
    check rather than skipping it. It is strictly stronger than the
    ``backbone_parameters > 0`` test it replaces: that one only established
    that SOME extractor was reported, while this pins the trained module's
    size to the representation it sits on. The 2026-07-28 defect --
    ``trainable_parameters: 85798656`` beside a config saying
    ``trainable: head`` -- fails this immediately.
    """
    if feature_dim is None:
        # The width is what the check is made of. Without it there is nothing
        # to compare against, and returning quietly would reproduce the very
        # failure this branch exists to prevent -- a check that passes because
        # it had no input (PLAN R7, instance 5).
        raise PooledFeatureError(
            "an artifact-fed arm reported no feature_dim, so the head-width "
            "check has nothing to compare against. Refused rather than "
            "skipped: a check that passes on missing input is not a check."
        )
    expected = int(feature_dim) + 1
    if trainable_parameters != expected:
        raise PooledFeatureError(
            f"policy is 'head' over {feature_dim}-dim precomputed embeddings, "
            f"so the head is {expected} parameters ({feature_dim} weights and "
            f"a bias) -- but {trainable_parameters} are reported trainable. A "
            "count far above this is the 2026-07-28 defect: a backbone's "
            "parameters reported as the arm's trainable set."
        )
