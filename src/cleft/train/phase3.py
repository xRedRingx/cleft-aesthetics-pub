"""Phase 3 orchestration: load, train, run the gates, write.

Wiring only. Same discipline as the Phase 1 and Phase 2 builds -- validate
everything, write the artifact last.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.manifest import load_manifest
from ..eval import metrics
from .ldl import SOFT_COLUMNS
from ..provenance import atomic_write_text
from . import determinism, gates
from .harness import CVResult, TrainConfig, run_cv
from .stub import StubBackbone

#: Which staged tensor the whole-image baseline reads.
#:
#: G1 is unmodified staging and is the NEUTRAL default for gate purposes. It
#: carries no implication about the G1-vs-G2 comparison, which is a Phase 7 arm.
#: Recorded in the config so the freeze does not silently bake in an arm.
DEFAULT_GEOMETRY = "g1"

#: **This tuple and the schema's ``train_cv.backbone`` choices are two
#: lists of the same thing** and they disagreed silently on 2026-08-14:
#: ``srgnn`` was added to the schema, so the config validated, and the run
#: died at ``make_factory`` after submission. A static test now requires
#: every schema choice to be CONSTRUCTIBLE here, which catches the
#: disagreement in either direction.
BACKBONES = (
    "vit_b16", "swin_b", "srgnn", "stub", "ridge",
    # Phase 7D's patch axis plus its hierarchical arm -- ARTIFACT-ONLY, like
    # swin_b: live extraction builds ViT-B/16 and nothing else.
    "vit_b32", "vit_b8", "mvitv2_b",
    # **Phase 25's two self-supervised checkpoints -- ARTIFACT-ONLY**, for
    # the same reason as swin_b and the 7D family: live extraction builds
    # ViT-B/16 and nothing else, so an arm naming either of these without a
    # declared embeddings artifact is refused by _check_train_cv_init.
    "vit_b14_dinov2", "vit_b16_dino",
    # Arm 4: a linear head over several pooled sets CONCATENATED
    # (pooled.load_concat_features). Artifact-only by construction -- there
    # is nothing to extract live, and each constituent set is
    # pairing-checked against its OWN backbone.
    "concat",
)

#: Which backbones actually reach torch. Determinism is gated on THIS rather than
#: on "anything that is not the stub", because that phrasing silently required
#: torch for every arm added later -- and Phase 4's mirror-difference arm is
#: torch-free by design, so it would have been unrunnable on the laptop it was
#: written to be testable on. A positive list means a new torch-free arm does not
#: have to remember to exclude itself.
#: swin_b joins at Phase 7. It never builds a torch BACKBONE here -- its
#: features arrive as a pooled artifact -- but its HEAD is torch, so the
#: determinism gate applies exactly as for vit_b16.
#: srgnn joins at Road B Branch 3 for the same reason swin_b did: its
#: concat arm builds no torch backbone here -- the crop vectors arrive as
#: a region_vectors artifact -- but its HEAD is torch, so the determinism
#: gate applies exactly as for the other two.
TORCH_BACKBONES = (
    "vit_b16", "swin_b", "srgnn",
    # The 7D arms fit EmbeddingHeadBackbone, which is torch -- including
    # "concat", whose head is the same module over a wider vector.
    "vit_b32", "vit_b8", "mvitv2_b", "concat",
)

#: Feature representations, as recorded in ``metrics.json`` under ``features``.
#: The parameter check keys off these rather than off the backbone name, because
#: what it needs to know is whether a **frozen extractor exists** -- which is a
#: fact about the features, not about which model consumed them.
#:
#: A set rather than one string: every arm that runs images through the frozen
#: backbone belongs here, whole-image and patch-fed alike. Adding a feature kind
#: that has an extractor and forgetting to add it here would silently exempt that
#: arm from the check.
WHOLE_IMAGE_EMBEDDINGS = "frozen_backbone_embeddings"
PATCH_EMBEDDINGS = "frozen_backbone_patch_embeddings"
FROZEN_EMBEDDINGS = frozenset({WHOLE_IMAGE_EMBEDDINGS, PATCH_EMBEDDINGS})

#: Arms whose embeddings were extracted in a DIFFERENT run and arrive as an
#: artifact -- the Phase 7 transformer ladder at pretrained inits. They have no
#: in-run extractor to account for, so ``FROZEN_EMBEDDINGS``'s condition
#: ("a frozen extractor must be reported") is not the right one for them.
#:
#: **They are kept inside the parameter check rather than exempted**, with
#: their own necessary condition: a linear head over D-dim embeddings is
#: exactly D+1 parameters, checked against the artifact's own ``feature_dim``.
#: A new feature kind silently falling out of this check is the 2026-07-28
#: defect exactly (a check keyed to one exact feature string, which then
#: skipped every patch arm), so the kind is enumerated here and answered
#: rather than allowed to miss the branch.
ARTIFACT_EMBEDDINGS = frozenset({"precomputed_backbone_embeddings"})

#: Suffix an LDL arm appends to whatever feature kind it sits on -- the
#: five-rater distribution rides in the row alongside the embedding.
#:
#: **It is a SUFFIX so the check cannot be escaped by composition.** An LDL
#: arm over precomputed embeddings reports
#: ``precomputed_backbone_embeddings+label_distribution``, which matches
#: neither ``FROZEN_EMBEDDINGS`` nor ``ARTIFACT_EMBEDDINGS`` exactly -- and a
#: membership test against those sets would have let every LDL arm skip the
#: parameter check entirely. That is the 2026-07-28 defect reappearing through
#: a new feature kind, which is precisely how it appeared the first time.
LDL_FEATURE_SUFFIX = "+label_distribution"


class Phase3Error(RuntimeError):
    """The Phase 3 run cannot proceed."""


@dataclass
class Phase3Result:
    cv: CVResult
    summary: dict
    gate_reports: dict


def load_geometry_rows(staged_dir: Path) -> list[dict]:
    """``geometry.csv`` as rows, for the per-image patch mapping.

    Separate from ``load_inputs`` rather than folded into it: the whole-image arm
    does not need it, and widening a function every arm calls in order to serve
    one of them is how a shared path accumulates arguments nobody reads.
    """
    from ..cluster_csv import read_cluster_csv

    path = staged_dir / "geometry.csv"
    if not path.is_file():
        raise Phase3Error(
            f"{path} does not exist. A patch arm needs the per-patient content "
            "box, without which every patient would receive identical pixel "
            "boxes and the same anatomy would sit at a different height in each."
        )
    # The first line is the CLUSTER-ONLY marker, as in manifest.csv. This used
    # to be `DictReader(lines[1:])` -- one of three independent skips, and the
    # weak version: it drops exactly one line whether or not it is a marker.
    return read_cluster_csv(path, expect=("patient_id",))


def load_inputs(
    manifest_dir: Path, staged_dir: Path, geometry: str, label: str,
    require_images: bool = True,
):
    """Manifest labels and folds, aligned to the staged tensor's row order.

    **``require_images=False`` is the artifact path** [ADDED 2026-08-14].
    An arm consuming a precomputed embedding set never reads pixels -- the
    backbone ran in the extraction run -- so loading the staged tensor here
    was pure waste on every ladder arm, and an outright BLOCKER for Road B
    Branch 3, whose non-square staging can never produce a stacked tensor
    at all (per-patient shapes differ, so ``np.stack`` raises; recorded in
    ``roadb_staging.SHAPE_IS_FIXED_AT_STAGING``).

    Everything else this function returns comes from the MANIFEST, not the
    tensor, and the ``geometry.csv`` row-order assertion -- the check that
    actually matters -- does not need the pixels either. So the skip drops
    a read, not a guarantee.

    Verified rather than assumed: a static test walks ``run`` and asserts
    ``images`` is unread on the artifact path, so a future use cannot
    silently receive ``None``.
    """
    rows = load_manifest(manifest_dir / "manifest.csv")
    if label not in rows[0]:
        raise Phase3Error(
            f"label {label!r} is not a manifest column; available: {sorted(rows[0])}"
        )

    features = None
    if require_images:
        tensor_path = staged_dir / f"staged_patient_{geometry}.npy"
        if not tensor_path.is_file():
            raise Phase3Error(
                f"{tensor_path} does not exist. Available: "
                f"{sorted(p.name for p in staged_dir.glob('staged_patient_*.npy'))}"
            )
        features = np.load(tensor_path)

    # The staged tensor's row order is geometry.csv's, which is the manifest's.
    # Asserted rather than assumed: a silent misalignment would train every
    # patient against another patient's label and still produce a number.
    geometry_csv = staged_dir / "geometry.csv"
    if geometry_csv.is_file():
        from ..cluster_csv import read_cluster_csv

        staged_ids = [
            int(r["patient_id"])
            for r in read_cluster_csv(geometry_csv, expect=("patient_id",))
        ]
        manifest_ids = [int(r["patient_id"]) for r in rows]
        if staged_ids != manifest_ids:
            raise Phase3Error(
                "the staged tensor's row order does not match the manifest: "
                f"{len(staged_ids)} staged rows vs {len(manifest_ids)} manifest "
                "rows, first difference at index "
                f"{next((i for i, (a, b) in enumerate(zip(staged_ids, manifest_ids)) if a != b), 'n/a')}"
            )

    # Only when there are images to count. The geometry.csv assertion above
    # is what guarantees the ORDER, and it ran either way.
    if features is not None and len(features) != len(rows):
        raise Phase3Error(
            f"{len(features)} staged images against {len(rows)} manifest rows"
        )

    patient_ids = [int(r["patient_id"]) for r in rows]
    labels = np.array([float(r[label]) for r in rows], dtype=float)
    assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}
    return features, labels, patient_ids, assignments


def prepare_features(
    images: np.ndarray,
    backbone: str,
    trainable: str,
    config: dict,
    geometry: str = DEFAULT_GEOMETRY,
    patch_scheme: str = "whole",
    geometry_rows: list[dict] | None = None,
    feature_source: str = "mirror",
) -> tuple[np.ndarray, dict]:
    """Raw images for full fine-tuning; frozen embeddings for a head-only arm.

    With the backbone frozen its output is a fixed function of the image, so it
    is extracted ONCE here rather than recomputed every epoch, every fold and
    every seed. That is far cheaper, and it removes the features as a source of
    seed-to-seed difference: they are byte-identical across seeds.

    **It does not make the seeds differ in initialisation alone.** An earlier
    revision of this docstring said it did. ``harness.inner_val_split`` seeds off
    ``config.seed``, so the seed also decides which patients are held out of the
    training fold -- which every arm inherits, whatever its model. The gate-2 band
    is therefore initialisation *and* split variance combined. See PLAN §4.12.
    """
    if backbone == "ridge":
        # Phase 4 §3.1 and §3.2. The features ARE the model in both: they are
        # computed from the geometry and the regressor only maps them to a grade.
        # G2 is required and refused mechanically in each module.
        if feature_source == "symnose":
            from ..geometry import symnose

            features, names = symnose.feature_matrix(images, geometry=geometry)
            return features, {
                "features": "symnose_index",
                "feature_names": names,
                "shape": list(features.shape),
                **symnose.describe(),
            }

        from ..geometry import mirror

        features, names = mirror.feature_matrix(images, geometry=geometry)
        return features, {
            "features": "mirror_difference_index",
            "feature_names": names,
            "shape": list(features.shape),
            **mirror.describe(),
        }

    #: **Phase 7C: raw pixels plus a per-patient augmentation-strength mask.**
    #:
    #: The augmenting arm keeps ``trainable: head`` -- the backbone is still
    #: frozen and the head is still 769 parameters, so the policy check is
    #: unchanged. What differs is that features are recomputed every epoch
    #: from augmented pixels, so the harness must be handed images.
    #:
    #: The mask travels as a FOURTH CHANNEL rather than as a separate array,
    #: because ``run_fold`` slices ``features[train_rows]`` and anything held
    #: beside the rows could not be sliced with them -- recovering the indices
    #: inside the backbone would mean recomputing ``inner_val_split`` outside
    #: the frozen harness, which is the one rule that must not have two
    #: implementations.
    augmentation = config.get("augmentation")
    if augmentation:
        from . import augment as augment_module

        masks = np.ones(images.shape[:3], dtype=np.float32)
        if augmentation.get("region_aware"):
            from .. import phase7c

            if geometry_rows is None:
                raise Phase3Error(
                    "a region-aware augmentation arm needs geometry.csv for "
                    "the per-patient region boxes; call load_geometry_rows "
                    "and pass geometry_rows"
                )
            masks = augment_module.protection_masks_for(
                images, geometry_rows, phase7c.PROTECTED_REGIONS,
                geometry=geometry,
                inside_strength=phase7c.REGION_AWARE["inside_strength"],
                sigma_fraction=phase7c.BLEND["sigma_fraction_of_crop_width"],
            ).astype(np.float32)

        stacked = np.concatenate(
            [np.asarray(images, dtype=np.float32), masks[..., None]], axis=3
        )
        return stacked, {
            "features": "augmented_pixels_and_strength_mask",
            "shape": list(stacked.shape),
            "augmentation": dict(augmentation),
            "region_aware": bool(augmentation.get("region_aware")),
            "note": (
                "channels 0-2 are pixels, channel 3 is the per-patient "
                "augmentation strength. The mask rides with the rows so the "
                "harness's own slicing carries it"
            ),
        }

    if backbone == "stub" or trainable == "full":
        return images, {"features": "raw_images", "shape": list(images.shape)}

    if patch_scheme != "whole":
        # Phase 4 §3.3/§3.4. Same frozen backbone as the whole-image probe, so
        # the comparison is about WHERE the model looks, not what it is.
        from ..geometry import patch_features
        from .torch_backbone import extract_patch_embeddings

        if geometry_rows is None:
            raise Phase3Error(
                f"the {patch_scheme!r} arm needs geometry.csv for the per-image "
                "mapping; call load_geometry_rows and pass geometry_rows"
            )

        patches = patch_features.patches_for(patch_scheme, geometry)
        pooling = config.get("pooling", patch_features.DEFAULT_POOLING)
        embeddings, report = extract_patch_embeddings(
            images,
            geometry_rows,
            patches,
            pooling=pooling,
            batch_size=config.get("batch_size", 32),
        )
        report.update(patch_features.describe(patch_scheme, geometry, patches, pooling))
        return embeddings, report

    from .torch_backbone import extract_embeddings

    embeddings, report = extract_embeddings(
        images,
        batch_size=config.get("batch_size", 32),
    )
    report["features"] = WHOLE_IMAGE_EMBEDDINGS
    return embeddings, report


class _AugmentingFactory:
    """Hands out one augmenting backbone per fold, counting as it goes.

    **Why a counter.** The ``Backbone`` factory protocol is zero-argument, so
    the fold cannot be passed in -- and the augmentation RNG wants a
    per-fold coordinate so the five folds do not draw identical view
    sequences. ``harness.run_cv`` calls the factory exactly once per fold, in
    the order it walks them, so the call index is the fold's ORDINAL POSITION.

    That is an assumption about frozen code, so it is **checked rather than
    trusted**: ``assert_called_once_per_fold`` compares the count against the
    number of folds the run actually had, and the ordinal is named as an
    ordinal rather than as a fold id, because for a non-contiguous fold set
    the two would differ and only one of them is what this holds.

    The frozen backbone is built once and shared: it is identical for every
    fold, and rebuilding ViT-B/16 five times per seed would dominate the arm.
    """

    def __init__(self, backbone, config, seed, augmentation):
        from .. import phase7c
        from .augment import Policy

        self.calls = 0
        self.ordinals: list[int] = []
        self._backbone = backbone
        self._seed = seed
        self._config = config
        self._policy = Policy(
            photometric=bool(augmentation.get("photometric")),
            geometric=bool(augmentation.get("geometric")),
            rotation=bool(augmentation.get("rotation")),
            region_aware=bool(augmentation.get("region_aware")),
        )
        self._photometric = phase7c.PHOTOMETRIC
        self._geometric = phase7c.GEOMETRIC
        self._extractor = None

    def __call__(self):
        from .torch_backbone import AugmentingHeadBackbone, FrozenExtractor

        if self._extractor is None:
            self._extractor = FrozenExtractor(
                self._backbone,
                batch_size=self._config.get("batch_size", 32),
            )
        ordinal = self.calls
        self.calls += 1
        self.ordinals.append(ordinal)
        return AugmentingHeadBackbone(
            policy=self._policy,
            photometric_settings=self._photometric,
            geometric_settings=self._geometric,
            seed=self._seed,
            fold_ordinal=ordinal,
            extractor=self._extractor,
            learning_rate=self._config.get("learning_rate", 1e-3),
            weight_decay=self._config.get("weight_decay", 0.01),
        )

    def assert_called_once_per_fold(self, n_folds: int) -> None:
        if self.ordinals != list(range(n_folds)):
            raise Phase3Error(
                f"the augmenting factory was called {self.ordinals} for "
                f"{n_folds} folds. It must be exactly once per fold, in "
                "order, or the per-fold augmentation streams do not "
                "correspond to the folds they augmented"
            )


def _augmenting_factory(backbone, config, seed, augmentation):
    return _AugmentingFactory(backbone, config, seed, augmentation)


def make_factory(backbone: str, trainable: str, config: dict, seed: int):
    """A zero-argument callable returning a fresh backbone."""
    if backbone == "stub":
        return StubBackbone
    if backbone == "ridge":
        from .ridge import RidgeBackbone

        return lambda: RidgeBackbone(
            alpha=config.get("alpha", 1.0), seed=seed
        )
    #: **[ADDED 2026-09-01] Phase 22's ranking arms.** The objective is
    #: carried in ``backbone_config``, NOT in a new backbone name and NOT
    #: in a train_cv field -- so ``train_cv``'s closed vocabulary and the
    #: shipped ``siamese_contrastive`` are both untouched
    #: (``phase22.AXIS_RULED``). The head itself is the shipped scalar
    #: one; only the loss that consumes it is new.
    if config.get("objective") == "pairwise_logistic":
        if trainable != "head":
            raise Phase3Error(
                f"the ranking objective with trainable={trainable!r} is not a "
                "Phase 22 arm: the registration is frozen backbone + linear "
                "probe throughout (phase22.HEAD_RULED)."
            )
        if "bounded" not in config:
            raise Phase3Error(
                "the ranking objective needs an explicit `bounded`: the "
                "bounded and unbounded heads are SEPARATE REGISTERED ARMS "
                "(phase22.HEAD_BOUNDING_RULED), so defaulting one would "
                "silently pick which arm this is."
            )
        # [ADDED 2026-09-01] Same refusal for the pair restriction. R-all
        # and R-clear differ in THIS AND NOTHING ELSE, so a default here
        # would silently make them the same arm -- which is exactly what
        # happened while it was absent (phase22.PAIR_SOURCE_WAS_NOT_CONSUMED).
        if "min_separation" not in config:
            raise Phase3Error(
                "the ranking objective needs an explicit `min_separation`: "
                "R-all (0.0) and R-clear (SE_diff) are SEPARATE REGISTERED "
                "ARMS differing in this alone, so defaulting it would "
                "silently pick which arm this is."
            )
        from .ranking import RankingHeadBackbone

        return lambda: RankingHeadBackbone(
            bounded=bool(config["bounded"]),
            min_separation=float(config["min_separation"]),
            learning_rate=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 0.01),
            seed=seed,
        )
    if backbone == "srgnn" and trainable != "head":
        # A trained SR-GNN is train_graph_cv's arm, with graph layers and
        # a feature map. Here it can only be a linear head over
        # precomputed crop vectors, so anything else is a config that
        # would build the wrong model under the right name.
        raise Phase3Error(
            f"backbone 'srgnn' with trainable={trainable!r} is not this "
            "task's arm: train_cv gives it a linear head over a "
            "precomputed region_vectors set. A trained SR-GNN is "
            "train_graph_cv."
        )
    if backbone in ("vit_b32", "vit_b8", "mvitv2_b", "concat") and (
        trainable != "head"
    ):
        # These arms exist ONLY as linear probes over precomputed pooled
        # sets (7D's registration is frozen-backbone-plus-probe throughout),
        # and "concat" has no single model to fine-tune at all.
        raise Phase3Error(
            f"backbone {backbone!r} with trainable={trainable!r} is not a 7D "
            "arm: the registration is frozen backbone + linear probe "
            "throughout (ladder.PHASE_7D_PATCH_AXIS_REGISTERED)."
        )
    if backbone in (
        "vit_b16", "swin_b", "srgnn", "vit_b32", "vit_b8", "mvitv2_b",
        "concat",
        # Phase 25: both are frozen-backbone linear probes over a declared
        # pooled artifact, which is the same regime as swin_b and the 7D
        # family -- the head is what trains, whatever produced the vectors.
        "vit_b14_dinov2", "vit_b16_dino",
    ):
        augmentation = config.get("augmentation")
        if augmentation and trainable == "head":
            return _augmenting_factory(backbone, config, seed, augmentation)
        if trainable == "head":
            from .torch_backbone import EmbeddingHeadBackbone

            return lambda: EmbeddingHeadBackbone(
                learning_rate=config.get("learning_rate", 1e-3),
                weight_decay=config.get("weight_decay", 0.01),
                seed=seed,
            )
        from .torch_backbone import TorchBackbone

        return lambda: TorchBackbone(
            learning_rate=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.01),
            batch_size=config.get("batch_size", 16),
            trainable=trainable,
            seed=seed,
        )
    raise Phase3Error(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")


#: What the interpretation below is drawn from, so the text is not a bare claim.
#: [MEASURED 2026-07-28, gate 2] Ten seeds of the frozen linear probe: pooled OOF
#: PCC 0.2347-0.2719 in every one, and ``beats_constant`` false in three of them.
SANITY_DIVERGENCE = "pcc 0.23-0.27 in 10/10 seeds, beats_constant false in 3/10"


def sanity_report(cv: CVResult) -> dict:
    """Did this arm learn anything? Two quantities, and PCC is the primary one.

    **[MEASURED 2026-07-28] why there are two.** ``beats_constant`` was written
    as the gate-2 blocker after the first Phase 3 run scored RMSE 0.658 against a
    label SD of 0.628 -- worse than predicting the mean. It did its job there. But
    across the ten-seed sweep it was **false in 3 of 10 seeds while pooled OOF PCC
    was 0.23-0.27 in all ten**, so as a pass/fail it would have condemned three
    seeds of an arm that plainly correlates.

    That is not noise, it is the expected behaviour of a shrunk predictor. A
    regression head fit under MSE on 152 samples compresses its outputs toward
    the label mean, and a compressed predictor keeps its *correlation* while its
    *RMSE* approaches a constant predictor's. The two quantities answer different
    questions -- R2 again -- and only one of them is the project's primary metric.

    So ``beats_constant`` stays, because calibration is worth recording, and
    ``pcc_positive`` becomes primary. ``shrinkage`` is reported beside them
    because it is the quantity that explains a divergence between the two, and
    reading either number without it invites the same confusion again.
    """
    truth = np.asarray(cv.oof_truth, dtype=float)
    predictions = np.asarray(cv.oof_predictions, dtype=float)

    baseline_rmse = float(np.std(truth))
    arm_rmse = float(np.sqrt(np.mean((truth - predictions) ** 2)))
    correlation = metrics.pcc(truth, predictions)
    truth_sd = float(np.std(truth))
    prediction_sd = float(np.std(predictions))

    return {
        # The one that decides whether an arm learned, in a PCC-primary project.
        "primary": "pcc_positive",
        "pcc": round(float(correlation), 6),
        "pcc_positive": bool(correlation > 0),
        # Calibration, recorded but NOT a gate. See the docstring.
        "beats_constant": bool(arm_rmse < baseline_rmse),
        "constant_predictor_rmse": round(baseline_rmse, 6),
        "arm_rmse": round(arm_rmse, 6),
        # sd(predictions) / sd(truth). Well below 1 means the head is shrinking
        # toward the mean, which is what separates the two checks above.
        "shrinkage": round(prediction_sd / truth_sd, 4) if truth_sd > 0 else 0.0,
        "n_distinct_3class_bins": int(
            len(np.unique(np.clip(np.rint(predictions), 1, 5)))
        ),
        "interpretation": (
            "pcc_positive is primary. beats_constant compares RMSE against a "
            "constant predictor and is a CALIBRATION check, not a learning "
            "check: in gate 2 it was false in 3 of 10 seeds while pooled OOF PCC "
            "was 0.23-0.27 in all ten, because a head fit under MSE on 152 "
            "samples shrinks toward the label mean (see shrinkage) and a shrunk "
            "predictor keeps its correlation while its RMSE approaches a "
            "constant's. Do not gate an arm on beats_constant."
        ),
    }


# --------------------------------------------------------------------------
# what the arm actually trains
# --------------------------------------------------------------------------


def parameter_summary(
    trainable: str,
    feature_report: dict,
    head_report: dict,
    *,
    backbone: str = "stub",
) -> dict:
    """The arm's real trainable set, assembled from the modules that reported it.

    **[MEASURED defect, 2026-07-28] why this exists.** ``metrics.json`` recorded
    ``trainable_parameters: 85798656`` with ``trainable_fraction: 1.0`` on a run
    whose ``train_config`` said ``trainable: head`` and whose head is **769**
    parameters. The numbers came from the frozen extraction pass, where every
    parameter has ``requires_grad`` set simply because nothing had asked it not
    to -- the model is never trained, so freezing it would have changed nothing
    about the run and nothing about the result. Only the report was wrong.

    A wrong parameter count is not cosmetic here. It is the number a reader uses
    to judge whether an arm was over-parameterised for 237 images, which is the
    single fact that explains the first Phase 3 run.
    """
    frozen_total = int(feature_report.get("backbone_parameters", 0))
    frozen_trainable = int(feature_report.get("backbone_trainable_parameters", 0))
    trained_total = int(head_report.get("total_parameters", 0))
    trained_trainable = int(head_report.get("trainable_parameters", 0))

    total = frozen_total + trained_total
    count = frozen_trainable + trained_trainable
    report = {
        "policy": trainable,
        "backbone": backbone,
        "features": feature_report.get("features", "unknown"),
        # Carried through for the artifact-fed head-width check, which needs
        # the representation's width to have anything to assert.
        "feature_dim": feature_report.get("feature_dim"),
        "total_parameters": total,
        "trainable_parameters": count,
        "trainable_fraction": round(count / total, 9) if total else 0.0,
        "components": {
            # Runs under no_grad and is discarded; absent for a "full" arm, which
            # has no frozen part.
            "frozen_feature_extractor": {
                "parameters": frozen_total,
                "trainable": frozen_trainable,
            },
            # What the optimiser was actually handed.
            "trained_module": {
                "parameters": trained_total,
                "trainable": trained_trainable,
            },
        },
    }
    assert_parameters_match_policy(report)
    return report


def assert_parameters_match_policy(report: dict) -> None:
    """The reported count must be consistent with the declared policy.

    Runtime, not only in the suite (R3). The defect this catches was a report
    that contradicted the config sitting beside it in the same file, and neither
    one was checked against the other.
    """
    policy = report["policy"]
    frozen = report["components"]["frozen_feature_extractor"]

    # Whether a frozen extractor must be reported is a fact about the FEATURES,
    # not about the backbone name. The stub is its own whole model; so is a ridge
    # over mirror-difference features, which has no backbone at all. Both report
    # every parameter trainable, and for both that is honest rather than the
    # defect. Only an arm consuming frozen embeddings has an extractor to account
    # for -- and that arm is exactly where the 2026-07-28 defect lived.
    #
    # A SET, not one string: the Phase 4 patch arms report
    # "frozen_backbone_patch_embeddings" and have exactly the same extractor to
    # account for. Matching one literal would have let every patch arm skip the
    # check that exists because this reporting was wrong once already.
    features = report["features"]
    is_ldl = features.endswith(LDL_FEATURE_SUFFIX)
    base_features = features[: -len(LDL_FEATURE_SUFFIX)] if is_ldl else features

    consumes_frozen_embeddings = base_features in FROZEN_EMBEDDINGS
    consumes_artifact_embeddings = base_features in ARTIFACT_EMBEDDINGS

    if policy == "head" and is_ldl:
        # An LDL head is (D x 5) + 5, not D + 1. Checked against the recorded
        # embedding width, which is the same shape of assertion the artifact
        # arms get: pin the trained module's size to the representation it
        # sits on, so a backbone's parameter count reported here would fail.
        width = report.get("feature_dim")
        expected = None if width is None else int(width) * 5 + 5
        if expected is None:
            raise Phase3Error(
                "an LDL arm reported no feature_dim, so the head-width check "
                "has nothing to compare against. Refused rather than skipped."
            )
        if report["trainable_parameters"] != expected:
            raise Phase3Error(
                f"policy is 'head' over {width}-dim embeddings with a "
                f"5-output softmax, so the head is {expected} parameters "
                f"({width}x5 weights and 5 biases) -- but "
                f"{report['trainable_parameters']} are reported trainable."
            )
        return

    if policy == "head":
        if frozen["trainable"] != 0:
            raise Phase3Error(
                f"policy is 'head' but the frozen feature extractor reports "
                f"{frozen['trainable']} trainable parameters; a head-only arm "
                "trains nothing in the backbone"
            )
        if consumes_artifact_embeddings:
            # No extractor in THIS run -- the backbone ran in the extraction
            # run. So the condition below ("some extractor was reported")
            # cannot apply, and the arm gets a stronger one instead: the head
            # must be exactly as wide as the representation it sits on. See
            # train.pooled, which owns the message.
            from .pooled import (
                PooledFeatureError,
                assert_head_matches_embedding_width,
            )

            try:
                assert_head_matches_embedding_width(
                    report.get("feature_dim"), report["trainable_parameters"]
                )
            except PooledFeatureError as exc:
                raise Phase3Error(str(exc)) from exc
            return
        if not consumes_frozen_embeddings:
            return

        if frozen["parameters"] <= 0:
            raise Phase3Error(
                f"policy is 'head' on backbone {report['backbone']!r} over "
                f"{WHOLE_IMAGE_EMBEDDINGS} but no frozen feature extractor is "
                f"reported, so the {report['trainable_parameters']} parameters "
                "counted trainable are the whole model. This is the 2026-07-28 "
                "defect: the frozen extraction pass's parameter count reported as "
                "the arm's trainable set, giving trainable_fraction 1.0 beside a "
                "config saying trainable: head."
            )
        if report["trainable_parameters"] >= report["total_parameters"]:
            raise Phase3Error(
                f"policy is 'head' but {report['trainable_parameters']} of "
                f"{report['total_parameters']} parameters are reported trainable "
                f"(fraction {report['trainable_fraction']})"
            )
    elif policy == "full":
        if report["total_parameters"] and report["trainable_fraction"] < 1.0:
            raise Phase3Error(
                f"policy is 'full' but only {report['trainable_parameters']} of "
                f"{report['total_parameters']} parameters are reported trainable"
            )


def run(
    *,
    manifest_dir: Path,
    staged_dir: Path,
    geometry: str = DEFAULT_GEOMETRY,
    label: str = "mean",
    backbone: str = "vit_b16",
    trainable: str = "head",
    seed: int = 1337,
    patch_scheme: str = "whole",
    feature_source: str = "mirror",
    #: Phase 7: consume a precomputed pooled embedding set instead of running
    #: the backbone over the images. ``prepare_features``'s live extraction
    #: builds ViT-B/16 at ImageNet init and nothing else, so every ladder arm
    #: at a pretrained init needs this. See ``train.pooled``.
    pooled_source=None,
    assignments_override: dict[int, int] | None = None,
    features_override: tuple | None = None,
    #: **[ADDED 2026-08-31] Phase 20's permutation control.** An index
    #: permutation, APPLIED to the labels immediately after they load and
    #: nowhere else -- so a permuted arm is the probe's own fit path with
    #: one vector reordered, and the control cannot drift from the thing
    #: it controls (``phase20.ARM_P_REGISTERED``).
    #:
    #: This function never DRAWS one. The draw lives in
    #: ``phase20.permute_within_strata``, is seeded there, and reaches
    #: here already made. ``None`` is the pre-existing path, byte for
    #: byte.
    label_permutation=None,
    train_config: TrainConfig | None = None,
    backbone_config: dict | None = None,
    log=print,
) -> Phase3Result:
    determinism_record = determinism.configure(
        seed, require_torch=(backbone in TORCH_BACKBONES)
    )
    log(f"determinism: {determinism_record}")

    # **`label: ldl` is not a manifest column.** Every other label names one;
    # this one names five (soft_1..soft_5) plus a construction, and is SCORED
    # on `mean` so its truth vector is the mean arm's -- which is the whole
    # basis on which Stage G may compare them. See train.ldl.
    from .ldl import LDL_LABEL

    is_ldl = label == LDL_LABEL
    # **The staged tensor is only needed by the paths that READ pixels.**
    # `features_override` counts its rows and live extraction runs the
    # backbone over it; the artifact path does neither, and for Road B
    # Branch 3 the tensor cannot exist at all (non-square staging stores
    # per patient). Asserted by a static test over this function.
    needs_images = pooled_source is None or features_override is not None
    images, labels, patient_ids, assignments = load_inputs(
        manifest_dir, staged_dir, geometry, "mean" if is_ldl else label,
        require_images=needs_images,
    )
    if label_permutation is not None:
        # **Phase 20's control, applied HERE and only here.** Everything
        # downstream -- features, folds, the head, the pooling, the
        # metrics -- is the probe's, untouched. What differs is which
        # patient each label belongs to.
        #
        # The images, patient_ids and fold assignments are deliberately
        # NOT reordered: permuting them too would just relabel the
        # cohort and reproduce the probe exactly. Breaking the
        # patient-label pairing is the whole measurement.
        order = np.asarray(label_permutation, dtype=int)
        if order.shape != labels.shape:
            raise Phase3Error(
                f"label_permutation has {order.shape} entries against "
                f"{labels.shape} labels"
            )
        if sorted(order.tolist()) != list(range(len(labels))):
            raise Phase3Error(
                "label_permutation is not a permutation of "
                f"0..{len(labels) - 1}: it must reorder the labels, never "
                "resample them"
            )
        labels = labels[order]

    if assignments_override is not None:
        # Phase 4's partition-sensitivity arm. cleft_v1's folds are unchanged on
        # disk and remain the ones every ladder arm uses; this substitutes an
        # alternative partitioning of the SAME patients for one measurement.
        missing = [pid for pid in patient_ids if pid not in assignments_override]
        if missing:
            raise Phase3Error(
                f"the assignments override covers {len(assignments_override)} "
                f"patients but {len(missing)} in the manifest are absent from it, "
                f"e.g. {missing[:5]}"
            )
        assignments = dict(assignments_override)
    log(f"{len(patient_ids)} patients, geometry {geometry}, label {label!r}")

    backbone_config = backbone_config or {}
    if features_override is not None:
        # Phase 4's partition arm runs the SAME arm over several partitionings of
        # the same patients. The features do not depend on the partitioning at
        # all, so extracting them once and passing them in turns N extractions
        # into one. Extraction was never inside the fold loop -- it is called
        # once per run, below -- but once per RUN still meant once per
        # partitioning, and for a frozen backbone that is pure waste.
        features, feature_report = features_override
        if len(features) != len(images):
            raise Phase3Error(
                f"features_override has {len(features)} rows against "
                f"{len(images)} staged images"
            )
    elif pooled_source is not None:
        # The artifact path. Row order is asserted inside, against the
        # patient_ids THIS run read from the manifest -- so the check compares
        # the vectors against the labels they will actually be paired with,
        # rather than against a second read of the artifact's own copy.
        # A LIST of sources is arm 4's concat: each constituent verified
        # individually, then concatenated (pooled.load_concat_features).
        from .pooled import PooledFeatureError, load_concat_features, load_features

        try:
            if isinstance(pooled_source, (list, tuple)):
                features, feature_report = load_concat_features(
                    list(pooled_source), patient_ids
                )
            else:
                features, feature_report = load_features(
                    pooled_source, patient_ids
                )
        except PooledFeatureError as exc:
            raise Phase3Error(str(exc)) from exc
        # No row-COUNT check here, deliberately. `features_override` needs one
        # because it arrives unverified, but this path has already asserted
        # the artifact's patient ids equal the manifest's *in order*, and
        # `load_inputs` asserted the staged tensor against the same manifest.
        # A count comparison would therefore be a check that cannot fail --
        # and one sitting beside a stronger check it is derived from reads as
        # independent evidence when it is none.
        if "constituents" in feature_report:
            log(
                f"pooled artifacts (CONCAT): "
                + " + ".join(
                    f"{c['embedding_set']}({c['feature_dim']})"
                    for c in feature_report["constituents"]
                )
                + f" -> {feature_report['shape']}"
            )
        else:
            log(
                f"pooled artifact: {feature_report['embedding_set']} "
                f"({feature_report['shape']}), checkpoint "
                f"{str(feature_report['checkpoint_sha256'])[:8]}"
            )
    else:
        # **geometry.csv is needed by two different arms now.** A patch arm
        # needs it for the per-patch mapping; a REGION-AWARE augmentation arm
        # needs it for the same mapping applied to the protection boxes. It
        # was loaded only for the first, so arms 4 and 5 would have raised on
        # a missing argument -- which is the honest failure, but only after
        # the whitelist above stopped them reaching this line at all.
        needs_geometry = patch_scheme != "whole" or bool(
            (backbone_config.get("augmentation") or {}).get("region_aware")
        )
        features, feature_report = prepare_features(
            images,
            backbone,
            trainable,
            backbone_config,
            geometry,
            patch_scheme,
            load_geometry_rows(staged_dir) if needs_geometry else None,
            feature_source,
        )
    log(f"features: {feature_report.get('features')}")

    ldl_report = None
    if is_ldl:
        # The distributions ride in the feature row: the frozen harness has no
        # side channel, and slices rows agnostically. That puts the LABEL
        # inside the input array, so the head's isolation from it is asserted
        # rather than arranged (train.ldl).
        from .ldl import (
            assert_mean_is_the_soft_expectation,
            pack_targets,
            target_distributions,
        )

        rows = load_manifest(Path(manifest_dir) / "manifest.csv")
        distributions = target_distributions(rows)
        # The manifest's own consistency: `mean` must BE the expectation of
        # the soft columns, or this arm trains toward one quantity and is
        # scored against another.
        consistency = assert_mean_is_the_soft_expectation(labels, distributions)
        embedding_dim = int(np.asarray(features).shape[1])
        features = pack_targets(features, distributions)
        feature_report = {
            **feature_report,
            "features": f"{feature_report.get('features')}+label_distribution",
            "embedding_dim": embedding_dim,
            # The parameter check reads this to derive the expected head width.
            "feature_dim": embedding_dim,
            "label_distribution_columns": list(SOFT_COLUMNS),
            "packed_shape": list(features.shape),
        }
        ldl_report = {
            "head": "softmax_5",
            "loss": "kl_divergence",
            "trained_on": "soft_1..soft_5 (the five-rater distribution)",
            "scored_on": (
                "the expectation over grades 1-5, against the SAME `mean` "
                "truth vector as the mean arm -- which is what makes the "
                "Stage G comparison legal"
            ),
            "manifest_consistency": consistency,
        }
        log(f"ldl: packed {features.shape}, embedding_dim {embedding_dim}")

    config = train_config or TrainConfig(seed=seed)

    # Keep the instances the harness built, so the parameter count comes off the
    # module that was actually optimised rather than being derived a second time
    # from the embedding width. A second derivation is a second thing that can be
    # wrong, and being wrong here is what this reporting fixes.
    trained: list = []

    if is_ldl:
        from .ldl import LDLHeadBackbone

        def factory():
            return LDLHeadBackbone(
                embedding_dim=feature_report["embedding_dim"],
                learning_rate=backbone_config.get("learning_rate", 1e-3),
                weight_decay=backbone_config.get("weight_decay", 0.01),
                seed=seed,
            )
    else:
        factory = make_factory(backbone, trainable, backbone_config, seed)

    def tracked_factory():
        instance = factory()
        trained.append(instance)
        return instance

    cv = run_cv(
        features=features,
        labels=labels,
        patient_ids=patient_ids,
        assignments=assignments,
        make_backbone=tracked_factory,
        config=config,
        log=log,
    )

    parameters = parameter_summary(
        trainable,
        feature_report,
        getattr(trained[-1], "parameter_report", {}) if trained else {},
        backbone=backbone,
    )
    log(
        f"parameters: {parameters['trainable_parameters']} trainable of "
        f"{parameters['total_parameters']} (policy {trainable!r})"
    )

    # Gates 5 and 6 run HERE, on this run's own output, not only in the suite.
    gate_reports = {
        "gate5_metric_verification": gates.verify_metrics(seed=seed),
        "gate6_oof_reconstruction": gates.reconstruct_oof(
            cv.fold_record(),
            expected_patients=set(patient_ids),
            expected_assignments=assignments,
            oof_ids=cv.oof_ids,
        ),
    }
    log("gates 5 and 6 passed")

    summary = {
        "n_patients": len(patient_ids),
        "geometry": geometry,
        "geometry_note": (
            "G1 is unmodified staging, the neutral default for gate purposes. "
            "It carries no implication about the G1-vs-G2 comparison, which is a "
            "Phase 7 arm."
        ),
        "label": label,
        **({"ldl": ldl_report} if ldl_report else {}),
        "backbone": backbone,
        "patch_scheme": patch_scheme,
        # Which pretraining the representation came from. Live extraction is
        # ImageNet-only, so an arm without a pooled artifact IS at imagenet --
        # stated rather than left to be inferred from the absence of a field.
        "init": (
            feature_report.get("init", "imagenet")
            if pooled_source is not None
            else "imagenet"
        ),
        "feature_provenance": (
            "precomputed embedding artifact"
            if pooled_source is not None
            else "extracted live in this run"
        ),
        "seed": seed,
        # One field two runs can be compared on, instead of diffing two CSVs by
        # hand. See determinism_fingerprint.
        "fingerprint": determinism_fingerprint(cv),
        "determinism": determinism_record,
        # EVERYTHING that determines the number. The first Phase 3 run recorded
        # only the epoch policy, so its metrics.json could not say what learning
        # rate, optimiser or trainable-parameter policy produced its result --
        # a provenance gap in a keeper run, independent of the result being bad.
        "train_config": {
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "inner_val_frac": config.inner_val_frac,
            "monitor": config.monitor,
            "seed": config.seed,
            "optimizer": "AdamW" if backbone != "stub" else "sgd_stub",
            "learning_rate": backbone_config.get(
                "learning_rate", 1e-3 if trainable == "head" else 1e-4
            ),
            "weight_decay": backbone_config.get("weight_decay", 0.01),
            "batch_size": backbone_config.get("batch_size", 16),
            "trainable": trainable,
            "loss": "mse",
            "label_scale": "raw_1_to_5",
        },
        "features": feature_report,
        # What the arm trains, counted off the module that trained. NOT read out
        # of `features` -- that reported the frozen extraction pass and said 1.0.
        "parameters": parameters,
        "sanity": sanity_report(cv),
        "oof": cv.metrics(),
        "epoch0": {
            "predicted_mean": [round(f.epoch0_pred_mean, 4) for f in cv.folds],
            "train_label_mean": [round(f.train_label_mean, 4) for f in cv.folds],
            # How far the head sits from where it should, in label SDs. This is
            # the number that catches the defect.
            "head_offset_sd": [
                round(
                    abs(f.epoch0_pred_mean - f.train_label_mean)
                    / max(np.sqrt(f.train_label_var), 1e-12),
                    3,
                )
                for f in cv.folds
            ],
            "inner_val_mse": [round(f.epoch0_inner_val_mse, 4) for f in cv.folds],
            "inner_val_var": [round(f.inner_val_var, 4) for f in cv.folds],
            # MSE over what a constant prediction at the head's value would
            # score. Exactly 1.0 for a head emitting one value, at any sample
            # size -- NOT mse/variance, which drifts with the inner-val split.
            "dispersion_ratio": [
                round(
                    f.epoch0_inner_val_mse
                    / max(
                        f.inner_val_var
                        + (f.inner_val_mean - f.epoch0_pred_mean) ** 2,
                        1e-12,
                    ),
                    3,
                )
                for f in cv.folds
            ],
        },
        "gates": gate_reports,
    }
    return Phase3Result(cv=cv, summary=summary, gate_reports=gate_reports)


#: [MEASURED 2026-07-28, gate 1] Three runs of the shipped config, unchanged, on
#: one GPU -- p3-frozen-1, p3-determinism-c, p3-determinism-d -- produced
#: byte-identical predictions.csv and the same selected epochs.
#: ``use_deterministic_algorithms(True)`` raised on nothing, so no op in this
#: harness lacks a deterministic kernel.
#:
#: **Gate 1 tests the HARNESS.** It was void before this, because the harness had
#: changed since it last passed, and Phase 7 owes another re-run when the patch
#: path lands. These are the values that re-run compares against.
GATE1_REFERENCE = {
    "n_runs": 3,
    "runs": ("p3-frozen-1", "p3-determinism-c", "p3-determinism-d"),
    "seed": 1337,
    "pcc": 0.2719366015264466,
    "mae": 0.5309369017806235,
    "rmse": 0.6515587916820617,
    "measured": "2026-07-28",
}

#: [MEASURED 2026-07-28, gate 2] The band the whole project is measured against.
#: Ten seeds, one job, frozen ViT-B/16 embeddings + linear head, G1, mean label,
#: 5-fold CV over the Phase 1 folds.
#:
#: **It is initialisation AND split variance combined**, and this sweep cannot
#: separate them. The embeddings are extracted once and reused, so the features
#: contribute nothing -- but the seed also drives ``harness.inner_val_split``, so
#: it decides which patients train. ``measures`` below names the quantity,
#: because "seed SD" alone does not: the Phase 4 ridge arm reports a seed SD too
#: and that one is split variance only.
#:
#: **It is the band for THIS arm.** Any arm with a substantially different
#: training procedure -- unfrozen backbones, the patch path, SR-GNN -- has more
#: sources of seed sensitivity and must have its own band measured. Reusing this
#: one there would be the QWK-as-ceiling mistake in a new place (R2).
MEASURED_SEED_BAND = {
    "n_seeds": 10,
    "mean": 0.2529,
    "sd": 0.0137,
    "min": 0.2347,
    "min_seed": 7,
    "max": 0.2719,
    "max_seed": 1337,
    "range": 0.0373,
    "arm": "frozen vit_b16 embeddings + linear head, g1, label mean, 5-fold CV",
    "measures": "head_initialisation_and_inner_val_split",
    "measured": "2026-07-28",
}

#: **[MEASURED 2026-08-01] Gate 1 RE-VERIFIED on the extended harness. The
#: last outstanding Phase 6 item, closed.**
#:
#: Gate 1 tests the HARNESS, and the harness had gained three backbones, a
#: graph-training path, a pooled-artifact feature source, a per-fold artifact
#: path and two probe policies since the reference was measured on
#: 2026-07-28. It was void until this ran.
#:
#: **Both halves passed.**
#:
#: * ``configs/p3_train_cv.yaml``, UNEDITED, reproduced ``GATE1_REFERENCE``
#:   exactly -- PCC 0.2719366015264466, the full-precision value, not a
#:   rounded match. No numeric drift on that path across everything Phases 4-6
#:   added. The config needed no edit because seed 1337 is already in its
#:   list, which is all ``run.task_train_cv``'s reference_arm predicate
#:   requires.
#: * ``configs/p6_determinism_graph.yaml`` run twice as separate jobs at the
#:   same seed: byte-identical on ``fingerprint.predictions_sha256``. That is
#:   the half GATE1_REFERENCE cannot cover, since only the arm that produced
#:   the reference can reproduce it -- the graph-training path needed its own
#:   two runs.
#:
#: The verdicts live in the run directories (``gate1_reproduction.json``,
#: added 2026-08-01 because the field had been computed after metrics.json was
#: written and survived only in log.txt -- a gate closed by reading a number
#: that was not being kept).
#:
#: **Still owed, and NOT covered by this**: the patch path. PLAN's Phase 4
#: section asks for it and phase3.determinism_fingerprint requests it by name.
#: PLAN's phrasing for it is unexecutable as written -- it says "compared
#: against GATE1_REFERENCE", but reference_arm is False for any
#: ``patch_scheme != "whole"``, so a patch arm records
#: ``gate1_reproduction.checked: false`` and never compares. The honest test
#: is two same-seed runs on ``predictions_sha256``, as for the graph arm.
GATE1_REVERIFIED = {
    "measured": "2026-08-01",
    "reference_reproduced": True,
    "reference_pcc": 0.2719366015264466,
    "reference_config": "configs/p3_train_cv.yaml (unedited)",
    "graph_half": {
        "config": "configs/p6_determinism_graph.yaml",
        "runs": 2,
        "predictions_sha256": (
            "2c7b998ece397e3eec6f1afde50a315609146b698b6233971c06156b12a7489a"
        ),
        "byte_identical": True,
    },
    "harness_changes_since_the_reference": (
        "three backbones, the graph-training path, the pooled-artifact feature "
        "source, the per-fold artifact path, two probe policies"
    ),
    "still_owed": (
        "the patch path -- PLAN asks for it and its phrasing is unexecutable "
        "(reference_arm is False for any patch arm, so no comparison happens); "
        "the honest test is two same-seed runs on predictions_sha256"
    ),
}

#: Seed counts the claimable delta is reported at. One is what a careless
#: comparison actually uses.
REPORTED_SEED_COUNTS = (1, 5, 10)


def claimable_delta(n_seeds: int, sd: float = MEASURED_SEED_BAND["sd"]) -> float:
    """The smallest PCC difference two arms at ``n_seeds`` seeds can be claimed.

    Two independent arm means, each over ``n_seeds`` seeds, differ by a standard
    error of ``sd * sqrt(2/n)``; at 95% that is ``1.96`` of them::

        delta = 1.96 * sd * sqrt(2 / n_seeds)

    At the measured SD of 0.0137 that gives **0.038 at one seed, 0.017 at five,
    0.012 at ten** -- and the one-seed figure is essentially the whole observed
    range, which is the point. A single seed resolves nothing this project is
    trying to measure.

    A formula rather than three written-down numbers, so a re-measured SD gives a
    re-derived band instead of a stale one somebody has to remember to update.
    """
    if n_seeds < 1:
        raise Phase3Error(f"n_seeds must be positive, got {n_seeds}")
    return 1.96 * float(sd) * float(np.sqrt(2.0 / n_seeds))


def predictions_digest(cv: CVResult) -> str:
    """SHA-256 over the pooled OOF predictions, patient order included.

    ``predictions.csv`` is CLUSTER-ONLY, so "the two files are byte-identical"
    is a claim only someone on the cluster can check, by eye, on two large files.
    This digest is computed from the same numbers at the same precision the CSV
    is written with, travels in the SHAREABLE ``metrics.json``, and reduces the
    check to comparing two strings.
    """
    import hashlib

    digest = hashlib.sha256()
    for pid, truth, prediction in zip(cv.oof_ids, cv.oof_truth, cv.oof_predictions):
        digest.update(f"{pid},{truth:.6f},{prediction:.6f}\n".encode("utf-8"))
    return digest.hexdigest()


def determinism_fingerprint(cv: CVResult) -> dict:
    """Everything two runs must agree on to be called identical.

    **Gate 1 tests the harness, and Phase 4 changes the path into it.** Patch
    extraction and pooling are new code between the images and the harness, and
    pooling is exactly where a nondeterministic reduction hides -- a reordered
    sum over 27 float32 patch embeddings would move the last digits and nothing
    downstream would notice. ``harness.py`` asked for this re-run itself; the
    arms simply arrived at Phase 4 rather than Phase 7.

    The digest is the operative field. The scalars are here so a mismatch says
    *how far apart* the two runs are, which distinguishes a genuine
    nondeterminism from a changed arm.
    """
    pooled = cv.metrics()
    return {
        "predictions_sha256": predictions_digest(cv),
        "pcc": pooled["pcc"],
        "mae": pooled["mae"],
        "rmse": pooled["rmse"],
        "selected_epochs": list(pooled["selected_epochs"]),
        "n": pooled["n"],
    }


def compare_to_gate1(fingerprint: dict, reference: dict = GATE1_REFERENCE) -> dict:
    """Does this run reproduce the recorded gate-1 result exactly?

    Only meaningful for the arm that produced ``GATE1_REFERENCE`` -- the
    whole-image probe at G1, seed 1337. For any other arm the numbers are
    *expected* to differ, and the comparison would be a category error, so the
    caller decides when to ask. See ``run.task_train_cv``.
    """
    fields = ("pcc", "mae", "rmse")
    differences = {
        field: {"run": fingerprint[field], "reference": reference[field]}
        for field in fields
        if fingerprint[field] != reference[field]
    }
    return {
        "reference": {field: reference[field] for field in fields},
        "matches": not differences,
        "differences": differences,
        "note": (
            "Exact equality, not a tolerance: gate 1's claim is that two runs of "
            "the same config on the same GPU are byte-identical, and a tolerance "
            "would convert that claim into a weaker one without saying so."
        ),
    }


def combined_claimable_delta(
    sd_a: float, n_a: int, sd_b: float, n_b: int
) -> dict:
    """The threshold a delta between two arms must clear. Two figures, named.

    PLAN §4.3 requires "the delta exceeds the combined seed uncertainty of the
    two arms". That phrase admits two readings, and they differ by a factor of
    about 2.5 -- so it is computed here rather than derived by hand each time,
    and both are returned with the quantity spelled out.

    ``arm_means_95`` -- **the primary figure.** Each arm is reported as a mean
    over its seeds, so the uncertainty on their difference is the standard error
    of that difference::

        1.96 * sqrt(sd_a^2 / n_a + sd_b^2 / n_b)

    ``single_run_95`` -- the conservative companion, ``1.96 * sqrt(sd_a^2 +
    sd_b^2)``. This is the spread of a difference between two **single runs**,
    one from each arm. It ignores the averaging, so it is always the larger of
    the two, and it answers "could one seed of A and one seed of B have differed
    by this much" rather than "do these arms differ".

    Use ``arm_means_95`` for a claim about arms. Quote ``single_run_95`` when the
    point is that a difference survives even without averaging -- which is a
    stronger statement, and worth making when it is true.

    [MEASURED 2026-07-28] For the mirror index against the ViT probe the two are
    0.019 and 0.046, and the -0.095 delta clears both, by 5.1x and 2.1x. Same
    conclusion either way there. It will not always be.
    """
    for name, value in (("n_a", n_a), ("n_b", n_b)):
        if value < 1:
            raise Phase3Error(f"{name} must be positive, got {value}")

    arm_means = 1.96 * float(np.sqrt(sd_a**2 / n_a + sd_b**2 / n_b))
    single_run = 1.96 * float(np.sqrt(sd_a**2 + sd_b**2))
    return {
        "arm_means_95": round(arm_means, 6),
        "single_run_95": round(single_run, 6),
        "primary": "arm_means_95",
        "inputs": {"sd_a": sd_a, "n_a": n_a, "sd_b": sd_b, "n_b": n_b},
        "note": (
            "arm_means_95 = 1.96*sqrt(sd_a^2/n_a + sd_b^2/n_b) is the uncertainty "
            "on the difference of the two arm MEANS, and is the claim threshold. "
            "single_run_95 = 1.96*sqrt(sd_a^2 + sd_b^2) is the spread of a "
            "difference between two single runs; it ignores the averaging and is "
            "always larger. They are different quantities -- say which one a "
            "number is."
        ),
    }


def seed_variance(pooled_pccs: list[float]) -> dict:
    """Gate 2's output: the band below which no delta is claimable.

    Reported as a distribution, never as a single number. The spread IS the
    result -- it is the denominator for every claim in the project, and each arm
    needs enough seeds that its mean is estimated tighter than the effect being
    claimed.

    [MEASURED] context: the void ladder saw the same nominal arm differ by 0.068
    PCC across SHAs, comparable to the entire effect it was meant to measure.
    """
    values = np.asarray(pooled_pccs, dtype=float)
    if values.size < 2:
        raise Phase3Error("seed variance needs at least two seeds")
    return {
        "n_seeds": int(values.size),
        "mean": round(float(values.mean()), 6),
        "sd": round(float(values.std(ddof=1)), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "range": round(float(values.max() - values.min()), 6),
        "interval_95": [
            round(float(np.percentile(values, 2.5)), 6),
            round(float(np.percentile(values, 97.5)), 6),
        ],
        "claimable_delta_floor": round(float(values.max() - values.min()), 6),
        # Derived from THIS sweep's SD, not from the recorded constant, so a
        # re-measured arm reports its own band rather than inheriting one.
        "claimable_delta_at_n_seeds": {
            str(n): round(
                claimable_delta(n, sd=float(values.std(ddof=1))), 6
            )
            for n in REPORTED_SEED_COUNTS
        },
        "note": (
            "No delta smaller than claimable_delta_floor is claimable from a "
            "single seed. Report the spread, never the mean alone. "
            "claimable_delta_at_n_seeds is 1.96 * sd * sqrt(2/n) -- the 95% "
            "resolution of a difference between two arm means at n seeds each. "
            "This band is for the arm that produced it; an arm with a "
            "substantially different training procedure needs its own."
        ),
    }


def write_outputs(result: Phase3Result, ctx, prefix: str = "") -> None:
    """Run-directory outputs, at the right tiers."""
    ctx.path(f"{prefix}metrics.json", tier="SHAREABLE").write_text(
        json.dumps(result.summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    header = ["fold", "epoch", "train_loss", "inner_val_mse", "inner_val_pcc"]
    lines = [",".join(header)]
    for fold in result.cv.folds:
        for entry in fold.curve:
            lines.append(
                f"{fold.fold},{entry['epoch']},{entry['train_loss']:.6f},"
                f"{entry['inner_val_mse']:.6f},{entry['inner_val_pcc']:.6f}"
            )
    ctx.path(f"{prefix}curves.csv", tier="SHAREABLE").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    rows = ["# CLUSTER-ONLY: patient-keyed", "patient_id,truth,prediction,fold"]
    fold_of = {
        pid: f.fold for f in result.cv.folds for pid in f.test_ids
    }
    for pid, truth, prediction in zip(
        result.cv.oof_ids, result.cv.oof_truth, result.cv.oof_predictions
    ):
        rows.append(f"{pid},{truth:.6f},{prediction:.6f},{fold_of[pid]}")
    ctx.path(f"{prefix}predictions.csv", tier="CLUSTER-ONLY").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    # Gate 6 reconstructs from this, so it must survive the run.
    atomic_write_text(
        ctx.path(f"{prefix}fold_record.json", tier="CLUSTER-ONLY"),
        json.dumps(result.cv.fold_record(), indent=2, sort_keys=True) + "\n",
    )
