"""SCUT pretraining: full fine-tuning on the official split, checkpointed to NFS.

Phase 6's build (PLAN Part 5, brief §3). Twelve runs -- 4 backbones x
{original, masked-G1, masked-G2} -- each producing one pretrained checkpoint
whose SCUT test correlation is recorded before anything downstream consumes it.

----------------------------------------------------------------------------
WHAT THE LOOP IS, AND WHAT IT IS NOT
----------------------------------------------------------------------------
This is NOT the frozen harness. ``harness.py`` runs 5-fold CV over 237 cleft
patients with a frozen backbone; this trains every parameter on 3,300 SCUT
faces against the dataset's own 1-5 label, once, for a FIXED epoch budget with
best-checkpoint selection on a held-out inner validation set -- no early
stopping (``FIXED_BUDGET_POLICY``). The harness stays untouched; what is reused
from it is IMPORTED, never modified: ``inner_val_split`` (so the split
semantics are the project's one implementation) and
``assert_epoch0_calibrated`` (gate 3, which exists precisely for the BatchNorm
backbones this phase introduces).

**Full fine-tuning is asserted, not assumed.** [DECIDED, brief §2.3]
pretraining trains everything; ``run_pretraining`` refuses a model whose
parameter report shows anything frozen. The opposite policy -- freezing -- is
asserted by the cleft path. A run that silently trained 769 of 86M parameters
here would produce a plausible checkpoint that pretrained nothing.

----------------------------------------------------------------------------
THE SPLIT IS READ, THE LABEL ARRIVES WITH IT, AND THE COUNTS ARE DECLARED
----------------------------------------------------------------------------
``scut.labels.read_labelled_split`` -- the official 3300/2200 split, CM152
excluded (test side, so training is untouched at 3,300). The config declares
``expect_train``/``expect_test`` and the run refuses a disagreement, the same
shape as the smoke task's ``n_samples``: the official counts cannot be
hardcoded here because the laptop tests run on synthetic roots, and a check
that special-cases the fixture is a check that cannot fail (PLAN R7).

Inner validation is carved FROM THE TRAIN SIDE by the frozen
``inner_val_split``; the test side is touched exactly once, at the end, by the
restored best weights. Best-checkpoint selection reads inner-val, never test --
selecting a checkpoint on the number the band check reads would make that
number unreadable.

----------------------------------------------------------------------------
CHECKPOINTING: THE DURABLE UNIT IS THE EPOCH [DECIDED]
----------------------------------------------------------------------------
The brief says "checkpoint every N steps"; the unit here is the EPOCH, and the
deviation is deliberate, recorded, and bounded. A mid-epoch resume needs the
batch permutation and the loop's position inside it serialised alongside the
RNG; an epoch boundary needs neither, because the permutation is drawn from
the checkpointed RNG stream at epoch start. At batch 32 over 2,970 faces an
epoch is ~93 optimizer steps -- minutes of GPU time -- so a pause loses
bounded work, and the byte-identity property the entry gate verified
(``docs/VERIFY_checkpoint.md``) holds exactly. ``checkpoint_every`` counts
epochs.

The checkpoint carries: model arrays, optimizer slots, the BEST-so-far model
arrays (without them a resume after the best epoch could not reproduce the
final restore), the numpy RNG driving the permutations, torch RNG state when
torch is present, early-stopping counters, and the durable ``curves.csv`` row
count. On resume the journal is truncated to that count -- rows written after
the checkpoint are re-produced, and keeping them would duplicate epochs.

``checkpoint.npz`` is resume state, not an output: it is deleted when the run
completes and ``pretrained.npz`` -- the best weights, in the same verified
format -- is what downstream consumes.

**Byte-identity is a property of the kernels as well as of this loop**
[MEASURED 2026-07-31, in the image]. The loop restores state exactly; whether
the continued run is BITWISE-identical then depends on the ops. Flag-off on
the image's CUDA: ViT-B/16, Swin-B and SR-GNN are bitwise (repeat-gradient
and post-restore step both 0.0); AG-Net is 2.26e-05 on the post-restore step
(roi_align's atomicAdd backward; 2.31e-05 in the corroborating local run --
see ``models.agnet.TRAINING_DETERMINISM``). **The twelve configs run
flag-off** [DECIDED]: the flag buys the three bitwise backbones nothing, and
cannot fix AG-Net in the image, where the deterministic roi_align
substitution dies on the missing C++ compiler. AG-Net's numbers therefore
carry a ~2.3e-05 bound on same-seed reproducibility and resume comparisons,
its determinism check is a tolerance rather than an equality, and any AG-Net
delta needs its own seed band measured under that condition -- the band's
floor is kernel noise, not zero. When ``deterministic: true`` is ever wanted,
it is applied through the frozen ``determinism.configure`` -- which owns the
CUBLAS_WORKSPACE_CONFIG export the flag requires -- never by setting the
flag directly.

----------------------------------------------------------------------------
THE BAND CHECK GATES CONSUMPTION [LITERATURE, brief §3.1]
----------------------------------------------------------------------------
Published SCUT-FBP5500 performance: ResNeXt-50 ~0.90, label-distribution
methods ~0.95, recent hybrids ~0.926. An ORIGINAL-source run below
``FAILURE_FLOOR`` (0.85) means pretraining failed and its checkpoint must not
be consumed; ``band_check`` records that verdict in metrics.json where the
embedding extraction can read it.

MASKED variants are never compared against the published band -- the mask
removes information the beauty rating depends on. Each is compared against its
own backbone's original run, a comparison that needs two runs and is therefore
recorded as ``decided_at_review``, not asserted here. Writing a verdict this
run cannot compute would be a check that reports success on its own failure
mode.

----------------------------------------------------------------------------
TORCH STAYS OUT OF THE TEST EXTRA
----------------------------------------------------------------------------
Module-level code is numpy only. ``TorchPretrainModel`` imports torch inside
its methods and is constructed only when a config names a real backbone;
``StubPretrainModel`` satisfies the same protocol -- including optimizer
momentum, so a resume that loses optimizer state diverges in tests exactly as
it would on the cluster.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from ..eval import metrics
from . import checkpoint as ckpt
from .harness import assert_epoch0_calibrated, inner_val_split

#: [LITERATURE, brief §3.1] Where published SCUT-FBP5500 correlation lands.
#: ResNeXt-50 ~0.90; label-distribution methods ~0.95; recent hybrids ~0.926.
PUBLISHED_BAND = (0.88, 0.93)

#: Below this, pretraining FAILED and the checkpoint must not be consumed.
#: Applies to original-source runs only; masked variants are compared against
#: their own backbone's original run, never this floor.
FAILURE_FLOOR = 0.85

#: The three pretraining sources. One field, not two: original-vs-masked and
#: G1-vs-G2 folded into a single axis so every config differs from its lattice
#: neighbours in exactly one field (the Phase 4 pattern).
#: **[2026-08-24] ``masked_original`` is Phase 15's addition**: the
#: WHOLE image through the frozen ``stage``, stored in the artifact
#: rather than read live. SCUT's ``original`` reads its root directly
#: (flat, .jpg, 350x350 square); MEBeauty's images are nested,
#: mixed-extension and non-square, so its uncropped variant is staged
#: ONCE into the artifact and read by the same stem-keyed loader as
#: g1/g2 -- reusing the row-order guarantee instead of writing a second
#: path that could lose it (phase15.ARTIFACT_FORMAT_IS_SCUTS).
SOURCES = ("original", "masked_g1", "masked_g2", "masked_original")

#: [DECIDED 2026-07-31] The region-scheme axis, for the GRAPH backbones only.
#: "native" is the architecture's own generator (SR-GNN's 26 grid-combinatorial
#: boxes, AG-Net's SIFT+GMM regions); grid/anatomy/random are Phase 2's
#: generators, mapped through each face's recorded content box exactly as the
#: Phase 4 patch probes mapped them. The SAME scheme must be used in
#: pretraining and cleft fine-tuning, so the graph layers see consistent node
#: structure throughout. Transformers have no region structure and take
#: "none"; the loop refuses a scheme that cannot act on its backbone -- a knob
#: that changes nothing is how unconfigured arms happen.
#:
#: **[MEASURED, PLAN §4.4] Scheme comparisons are read at G2.** At G1 the
#: schemes carry different background exposure -- grid has white in 10 of 27
#: patches (up to 49.9%), anatomy's lateral_orbit regions are 76.4% outside
#: the mask, random has none -- so a G1 scheme difference is partly a
#: background difference. The G1 runs exist for the geometry axis; the scheme
#: axis is compared at G2.
REGION_SCHEMES = ("none", "native", "grid", "anatomy", "random")

#: The schemes that need generated boxes (native uses the model's own).
GENERATED_SCHEMES = ("grid", "anatomy", "random")

#: **[MEASURED 2026-07-31, the thirty fixed-budget runs -- figures pasted by
#: the maintainer] The scheme axis is NULL at pretraining.** Across all six
#: backbone-source lattices the scheme ranges span 0.0016 to 0.0154 test PCC,
#: inside the 0.0137 seed band, and no scheme wins twice in the same backbone
#: across sources. Third independent line pointing the same way, after Phase
#: 4's pooled patch probes and the superseded (epoch-confounded) SR-GNN
#: scheme run.
#:
#: **The caveat is the scope, and it must travel with the number: this
#: measures the scheme's effect on PRETRAINING, not on cleft fine-tuning with
#: trained graph layers.** Pretraining fine-tunes the whole model on 3,300
#: faces, which can absorb a region layout; the cleft arms train ONLY the
#: graph layers on 237, where the layout is most of what the model can vary.
#: Reading this as "scheme does not matter" would be the aggregation error in
#: a new place -- a correct number attached to a question it cannot answer
#: (PLAN R2). The Stage E comparison at cleft time remains live, at G2.
SCHEME_AXIS_AT_PRETRAINING = {
    "measured": "2026-07-31",
    "environment": "the pinned image, thirty fixed-budget runs (pasted)",
    "range_min": 0.0016,
    "range_max": 0.0154,
    "seed_band": 0.0137,
    "verdict": "null at pretraining: ranges inside the seed band, no scheme "
               "winning twice in the same backbone across sources",
    "independent_confirmations": (
        "Phase 4 pooled patch probes; the superseded epoch-confounded run; "
        "this lattice"
    ),
    "caveat": (
        "scheme effect on PRETRAINING only -- cleft fine-tuning trains the "
        "graph layers on 237 images where the layout is most of what varies; "
        "Stage E at G2 remains the live comparison"
    ),
    #: **[MEASURED 2026-08-01] The caveat is now discharged, and the axis is
    #: closed.** Stage E0 ran the cleft comparison with the checkpoint held
    #: constant (ImageNet init, one scheme-free representation, ten seeds):
    #: range 0.0161 against a 0.0286 threshold, nothing claimable. Fourth
    #: independent null. ``graph_cleft.SCHEME_AXIS_AT_CLEFT``.
    #:
    #: **And a second finding came out of the attempt that got it wrong.**
    #: Stage E consumed the four scheme-matched checkpoints and spread by
    #: 0.095 -- so these four, statistically equal HERE on 2,199 held-out
    #: faces, are not equal in what they transfer to 237 clinical images.
    #: This record's null is about SCUT, and licensing a cleft comparison on
    #: it was an inference, not a measurement
    #: (``graph_cleft.PRETRAINING_DOES_NOT_PREDICT_TRANSFER``).
    "closed_at_cleft": {
        "measured": "2026-08-01",
        "arm": "Stage E0 -- srgnn imagenet g2, four schemes, 10 seeds",
        "range": 0.0161,
        "threshold": 0.0286,
        "claimable": False,
        "null_number": 4,
        "but_the_checkpoints_were_not_equivalent": (
            "the same four checkpoints spread 0.095 at cleft time; this "
            "null is about SCUT and does not license a cleft comparison"
        ),
    },
}

#: **[MEASURED 2026-07-31, same runs] Geometry separates, for one backbone,
#: with a candidate mechanism.** ViT-B/16 masked-G1 0.7893 against masked-G2
#: 0.8306 -- the lowest number in the table and a 0.041 gap -- while Swin-B is
#: indifferent and the graph backbones marginally prefer G1 (~0.007).
#:
#: The mechanism is [REASONED], not measured: ViT's fixed 16x16 patch tiling
#: meets G1's high-contrast baked white corners, and a hard white edge inside
#: a patch is exactly what a fixed tiling cannot adapt around -- Swin's
#: shifted windows and the graph models' region pooling both can. Plausible,
#: unverified, and it matters for the cleft ladder: if it holds, the ViT-G1
#: cleft arms carry a backbone-specific geometry penalty that is an artefact
#: of tiling, not of the fill. Do not promote this to a claim without its own
#: arm.
GEOMETRY_EFFECT_AT_PRETRAINING = {
    "measured": "2026-07-31",
    "vit_b16_masked_g1": 0.7893,
    "vit_b16_masked_g2": 0.8306,
    "gap": 0.041,
    "swin_b": "indifferent",
    "graph_backbones": "marginally prefer G1 (~0.007)",
    "mechanism": (
        "[REASONED] ViT's fixed 16x16 tiling meeting G1's high-contrast "
        "white corners; unverified"
    ),
}

#: The input name a masked-source config must declare, and an original-source
#: config must NOT: an input a run never reads would still be hash-verified and
#: recorded in inputs.json, a provenance record claiming data fed a run it did
#: not feed.
MASKED_INPUT_NAME = "masked_scut"

#: **[2026-08-10] The pretrained init, as a DECLARED input -- by conventional
#: name, like the masked artifact above.** Until this existed the ImageNet
#: init was the one input guard 3 never verified: ``pretrained=True``
#: re-downloads at run time (HOME is ephemeral), and when two ViT-512 runs at
#: one SHA produced different checkpoints, NO field could say whether the
#: inits differed -- the comparison was unarbitrable because the quantity was
#: never captured (``roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE``). Declared, the
#: init is a versioned NFS artifact (``data/inits/init_<backbone>_v1``, built
#: by the ``snapshot_pretrained_init`` task) whose rollup guard 3 verifies
#: and inputs.json records.
INIT_INPUT_NAME = "pretrained_init"

#: The snapshot artifact's weights file.
INIT_WEIGHTS_NAME = "init.npz"

#: The deliverable's filename: the best weights in the verified checkpoint
#: format, so the embedding extraction loads it with ``checkpoint.load`` and
#: inherits the digest verification.
PRETRAINED_NAME = "pretrained.npz"

#: **[DECIDED 2026-07-31] Fixed epoch budget with best-checkpoint selection.
#: NO early stopping in pretraining.** Carried into every metrics.json so a
#: reader of any run meets the policy and its provenance without excavating
#: this module.
#:
#: **What forced it [MEASURED, the four SR-GNN scheme originals].** The scheme
#: ranking tracked epochs run, not scheme: anatomy and random selected epoch 5
#: and stopped at 13 because inner_val_mse never beat its epoch-5 value --
#: for anatomy by 2.3e-05 (0.155406 vs 0.155429), a near-tie on a flat noisy
#: curve. Patience terminated at whichever point the noise happened to land,
#: native and grid caught a marginal improvement inside their window, and
#: none of the eight completed runs had converged: inner-val PCC was still
#: climbing at every stopping point. The replay over all eight curves showed
#: every run changing selection under a PCC monitor, six running out of
#: recorded curve -- so switching the monitored quantity would only move
#: where the noise bites. **Early stopping on a flat noisy curve terminates
#: at a random point; the failure mode is the mechanism, not the metric.**
#:
#: **Why this does not contradict Phase 3 gate 4.** Gate 4 rejected fixed
#: budgets because the void ladder's 40-epoch budget measured post-collapse
#: endpoints -- on 152 cleft samples, inner-val PCC peaked at epochs 2-11 and
#: decayed to ~0 in 20 of 20 fold-runs. On SCUT's 3,300 there is no collapse:
#: PCC climbs monotonically in all eight curves. The reasoning was sound and
#: does not transfer to this data size. **Early stopping stays for the cleft
#: ladder, where collapse is the documented behaviour.**
#:
#: N=30 covers every observed run except SR-GNN original's 34, and PCC had
#: largely plateaued by then in all four scheme curves. Selection is by
#: inner_val_pcc, the primary metric -- legitimate because inner-val is
#: carved from the TRAIN side and the test set is disjoint, touched once.
#: Comparability across the thirty arms is by construction; the cost is
#: knowable in advance.
FIXED_BUDGET_POLICY = {
    "decided": "2026-07-31",
    "policy": "fixed_epoch_budget_best_checkpoint",
    "early_stopping": False,
    "why": (
        "patience terminated on noise in a flat MSE region (anatomy epochs 5 "
        "vs 7: 2.3e-5 apart), so run length tracked luck rather than scheme, "
        "and none of the eight completed runs had converged"
    ),
    "gate4_distinction": (
        "the void ladder's fixed budget measured post-collapse endpoints on "
        "152 cleft samples; on SCUT's 3,300 there is no collapse -- PCC "
        "climbs monotonically -- so gate 4's rejection of fixed budgets does "
        "not transfer. Early stopping stays for the cleft ladder."
    ),
    "supersedes": (
        "the eight pre-policy runs (four backbone originals; four SR-GNN "
        "scheme originals) -- superseded, not deleted: their curves are the "
        "evidence for this decision"
    ),
}

#: Array-name prefixes inside checkpoints. ``best__`` shadows ``model__`` so a
#: resume can reproduce the final restore exactly.
MODEL_PREFIX = "model__"
OPTIMIZER_PREFIX = "opt__"
BEST_PREFIX = "best__"


class PretrainError(RuntimeError):
    """The pretraining run cannot proceed, or its inputs are not what the
    config declared."""


@dataclass(frozen=True)
class PretrainConfig:
    """Every training knob, validated. All of it lands in metrics.json."""

    #: The FIXED budget: every run trains exactly this many epochs, and the
    #: kept checkpoint is the best by ``monitor``. There is deliberately no
    #: patience field -- see ``FIXED_BUDGET_POLICY``.
    epochs: int
    inner_val_frac: float
    seed: int
    #: [DECIDED 2026-07-31, from image measurements] Whether the torch model
    #: runs under ``torch.use_deterministic_algorithms(True)``, applied via
    #: the frozen ``determinism.configure``. **The twelve shipped configs say
    #: FALSE**, and the reasoning is measured, not assumed: flag-off in the
    #: image, ViT-B/16, Swin-B and SR-GNN are bitwise already (repeat-grad
    #: and post-restore step both 0.0), so the flag buys them nothing; and
    #: it CANNOT fix AG-Net there -- torchvision's deterministic roi_align
    #: substitution needs torch.compile, which needs inductor, which needs a
    #: C++ compiler the pinned image does not have (InvalidCxxCompiler).
    #: Machinery for no gain. AG-Net's bound is recorded instead:
    #: ``models.agnet.TRAINING_DETERMINISM``, ~2.3e-05 on the post-restore
    #: step, and any AG-Net comparison carries it.
    deterministic: bool
    #: What best-checkpoint selection reads. inner_val_pcc [DECIDED
    #: 2026-07-31]: the primary metric, on a split carved from the TRAIN side
    #: with the test set disjoint and touched once.
    monitor: str = "inner_val_pcc"
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    #: Epochs between checkpoints. See the module docstring for why the unit
    #: is the epoch.
    checkpoint_every: int = 1
    #: Gate 3 tolerances, matching the frozen harness defaults.
    epoch0_mean_tol: float = 0.5
    epoch0_mse_tol: float = 0.6

    def __post_init__(self) -> None:
        if self.monitor not in ("inner_val_mse", "inner_val_pcc"):
            raise PretrainError(f"unknown monitor {self.monitor!r}")
        if not 0.0 < self.inner_val_frac < 0.5:
            raise PretrainError(f"inner_val_frac out of range: {self.inner_val_frac}")
        if self.epochs < 1:
            raise PretrainError("epochs must be positive")
        if self.batch_size < 1:
            raise PretrainError("batch_size must be positive")
        if self.checkpoint_every < 1:
            raise PretrainError("checkpoint_every must be positive")


class PretrainModel(Protocol):
    """What the loop needs from a model. Torch and stub both satisfy it.

    Distinct from the harness ``Backbone`` protocol on purpose: pretraining
    needs per-batch steps (the loop owns the shuffle, so batch order lives in
    the checkpointed RNG) and state capture (so a resume restores optimizer
    slots, not only weights).
    """

    parameter_report: dict

    def reset(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
        """Build/initialise. The head must start predicting the train mean."""

    def train_batch(
        self, features: np.ndarray, labels: np.ndarray,
        boxes: np.ndarray | None = None,
    ) -> float:
        """One optimizer step. ``boxes`` is (B, R, 4) in the backbone's own
        convention for generated region schemes; None means the model's
        native regions (or no regions at all)."""

    def predict(
        self, features: np.ndarray, boxes: np.ndarray | None = None
    ) -> np.ndarray:
        """Predictions on the raw 1-5 scale."""

    def state_arrays(self, *, include_optimizer: bool = True) -> dict[str, np.ndarray]:
        """Everything a resume restores, as named numpy arrays."""

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        """Restore captured state. Optimizer slots restored when present."""


# --------------------------------------------------------------------------
# the stub: numpy, momentum SGD, deterministic
# --------------------------------------------------------------------------


@dataclass
class StubPretrainModel:
    """A linear head on flattened pixels, trained by momentum SGD.

    Momentum is not decoration: it is optimizer state a naive resume loses,
    which is exactly what the resume test must be able to catch. Zero-init
    weights and a train-mean bias make epoch 0 calibrated, so the stub passes
    gate 3 the same way the torch models do.

    Standardisation statistics come from the FULL training set at ``reset``,
    not from the first batch, so the model is independent of batch order and
    the loop's shuffle can be exercised freely.
    """

    learning_rate: float = 0.15
    momentum: float = 0.9
    parameter_report: dict = field(default_factory=dict)

    _weights: np.ndarray | None = field(default=None, repr=False)
    _bias: float = 0.0
    _velocity_w: np.ndarray | None = field(default=None, repr=False)
    _velocity_b: float = 0.0
    _centre: np.ndarray | None = field(default=None, repr=False)
    _scale: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def _flatten(features: np.ndarray) -> np.ndarray:
        array = np.asarray(features, dtype=float)
        return array.reshape(array.shape[0], -1) if array.ndim > 1 else array[:, None]

    def reset(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
        design = self._flatten(train_features)
        self._centre = design.mean(axis=0)
        spread = design.std(axis=0)
        # The sqrt(D) factor keeps the gradient scale independent of the
        # feature count, so one learning rate is stable whether the fixture
        # hands over 192 flattened pixels or 150,528.
        self._scale = np.where(spread > 1e-12, spread, 1.0) * np.sqrt(design.shape[1])
        self._weights = np.zeros(design.shape[1], dtype=float)
        self._velocity_w = np.zeros_like(self._weights)
        self._velocity_b = 0.0
        self._bias = float(np.mean(train_labels))
        n = int(self._weights.size) + 1
        self.parameter_report = {
            "total_parameters": n,
            "trainable_parameters": n,
            "trainable_fraction": 1.0,
        }

    def _design(self, features: np.ndarray) -> np.ndarray:
        return (self._flatten(features) - self._centre) / self._scale

    def train_batch(
        self, features: np.ndarray, labels: np.ndarray,
        boxes: np.ndarray | None = None,
    ) -> float:
        # The stub has no regions; boxes are accepted so the loop's plumbing
        # can be exercised torch-free, and ignored.
        scaled = self._design(features)
        residual = scaled @ self._weights + self._bias - labels
        grad_w = 2.0 * scaled.T @ residual / len(labels)
        grad_b = 2.0 * float(residual.mean())
        self._velocity_w = self.momentum * self._velocity_w + grad_w
        self._velocity_b = self.momentum * self._velocity_b + grad_b
        self._weights = self._weights - self.learning_rate * self._velocity_w
        self._bias = self._bias - self.learning_rate * self._velocity_b
        return float(np.mean(residual**2))

    def predict(
        self, features: np.ndarray, boxes: np.ndarray | None = None
    ) -> np.ndarray:
        if self._weights is None:
            raise PretrainError("predict before reset")
        return self._design(features) @ self._weights + self._bias

    def state_arrays(self, *, include_optimizer: bool = True) -> dict[str, np.ndarray]:
        if self._weights is None:
            raise PretrainError("state_arrays before reset")
        arrays = {
            MODEL_PREFIX + "weights": self._weights.copy(),
            MODEL_PREFIX + "bias": np.array([self._bias], dtype=float),
            MODEL_PREFIX + "centre": self._centre.copy(),
            MODEL_PREFIX + "scale": self._scale.copy(),
        }
        if include_optimizer:
            arrays[OPTIMIZER_PREFIX + "velocity_w"] = self._velocity_w.copy()
            arrays[OPTIMIZER_PREFIX + "velocity_b"] = np.array(
                [self._velocity_b], dtype=float
            )
        return arrays

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        self._weights = np.asarray(arrays[MODEL_PREFIX + "weights"], dtype=float).copy()
        self._bias = float(arrays[MODEL_PREFIX + "bias"][0])
        self._centre = np.asarray(arrays[MODEL_PREFIX + "centre"], dtype=float).copy()
        self._scale = np.asarray(arrays[MODEL_PREFIX + "scale"], dtype=float).copy()
        if OPTIMIZER_PREFIX + "velocity_w" in arrays:
            self._velocity_w = np.asarray(
                arrays[OPTIMIZER_PREFIX + "velocity_w"], dtype=float
            ).copy()
            self._velocity_b = float(arrays[OPTIMIZER_PREFIX + "velocity_b"][0])


# --------------------------------------------------------------------------
# the real thing: any factory backbone, AdamW, full fine-tuning
# --------------------------------------------------------------------------


@dataclass
class TorchPretrainModel:
    """Factory backbone + regression head, MSE, everything trainable.

    Normalization is read off the model's own preprocessing
    (``factory.normalization_for``), which REFUSES a model that reports none --
    that refusal is the Stage-1 defect guard and it applies here unchanged. A
    graph backbone landing without a stamp fails at reset with the message that
    says exactly what to provide.
    """

    name: str
    #: False only for mechanics verification (scripts/verify_pretrain_torch.py):
    #: the twelve runs always start from pretrained weights.
    pretrained: bool = True
    #: A declared init artifact (data/inits/init_<backbone>_v1). When set, the
    #: model is built WITHOUT downloading and the snapshot's weights load with
    #: strict accounting -- the init the run started from is then a
    #: guard-3-verified input rather than whatever a download returned.
    init_dir: Any = None
    #: See PretrainConfig.deterministic. Applied at reset, one-way: True sets
    #: the global flag; False leaves whatever the process already has, so a
    #: verification script's own setting is never silently clobbered.
    deterministic: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    seed: int = 1337
    device: str = "cuda"
    parameter_report: dict = field(default_factory=dict)

    _model: Any = field(default=None, repr=False)
    _optimizer: Any = field(default=None, repr=False)
    _mean: Any = field(default=None, repr=False)
    _std: Any = field(default=None, repr=False)

    def _classifier(self) -> Any:
        """The regression head, however the model names it.

        timm models expose ``get_classifier()``; the ported graph models name
        the head ``classifier``. Anything else is refused by name so the
        contract is visible at the failure site rather than guessed at.
        """
        model = self._model
        if hasattr(model, "get_classifier"):
            return model.get_classifier()
        if hasattr(model, "classifier"):
            return model.classifier
        raise PretrainError(
            f"{type(model).__name__} exposes neither get_classifier() nor "
            ".classifier; the pretraining head cannot be initialised. The "
            "ported graph models must name their head 'classifier'."
        )

    def reset(self, train_features: np.ndarray, train_labels: np.ndarray) -> None:
        import torch

        from ..models.factory import (
            count_parameters,
            create_backbone,
            normalization_for,
        )

        if self.deterministic:
            # THROUGH the frozen determinism module, never reimplemented here.
            # [MEASURED 2026-07-31, in the image] The first version set the
            # flag directly and every backbone raised "not deterministic
            # because it uses CuBLAS": use_deterministic_algorithms needs
            # CUBLAS_WORKSPACE_CONFIG exported before the first CUDA context,
            # which determinism.configure owns as part of the Phase 0
            # contract. A second copy of that setup is the two-predicates
            # defect class, and it was -- caught by the maintainer, not by a
            # test.
            from . import determinism

            determinism.configure(self.seed, require_torch=True)

        torch.manual_seed(self.seed)
        # **[CORRECTED 2026-08-09, gate 2] The training path never told the
        # factory its input size** -- a 512/768 pretraining run would have
        # died at the ViT/Swin 224 asserts, after submission. The features
        # carry their own size (N, H, W, 3), so the builder consumes it the
        # same way extract_embeddings does (factory.timm_kwargs_for); 224
        # keeps the historical path byte-identical.
        input_size = tuple(
            int(v) for v in np.asarray(train_features).shape[1:3]
        )
        # **[DECIDED 2026-08-10; sufficiency OVERTURNED 2026-08-11] ViT
        # trains UNFUSED -- which removes ONE measured nondeterminism
        # source and is NOT sufficient**: the unfused pair still diverged
        # 0.035 (roadb.UNFUSED_WAS_NOT_SUFFICIENT -- the deterministic-flag
        # probe convicted the op CLASS, not fused attention alone). The
        # toggle stays because its runs are among the recorded draws and
        # reverting would churn ViT's procedure a third time; ViT cells are
        # recorded as draws (~0.055 spread at 224), Road A's treatment.
        # Mechanics unchanged: build-time flag, toggled around the build
        # ONLY (extraction stays fused and byte-compatible with Road A's
        # embeddings), ViT only; the task dict is untouched, so the
        # lattice's byte-equality with Road A survives.
        unfused = self.name == "vit_b16"
        if unfused:
            from timm import layers as timm_layers

            fused_before = timm_layers.use_fused_attn()
            timm_layers.set_fused_attn(False)
        try:
            if self.init_dir is not None:
                # The DECLARED init: build without downloading, load the
                # snapshot, and account for every key -- a silent partial
                # load would be the undeclared input back in a quieter form.
                self._model = create_backbone(
                    self.name, pretrained=False, num_outputs=1,
                    input_size=input_size,
                )
                load_declared_init(self._model, self.init_dir)
            else:
                self._model = create_backbone(
                    self.name, pretrained=self.pretrained, num_outputs=1,
                    input_size=input_size,
                )
        finally:
            if unfused:
                timm_layers.set_fused_attn(fused_before)
        if unfused:
            attn = self._model.blocks[0].attn
            if getattr(attn, "fused_attn", False):
                raise PretrainError(
                    "the fused_attn toggle did not take: blocks[0].attn is "
                    "still fused. The nondeterministic path would train "
                    "again -- refusing (VIT_PRETRAINING_IS_NONDETERMINISTIC)."
                )

        mean, std = normalization_for(self._model)
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self._mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self._std = torch.tensor(std, device=device).view(1, -1, 1, 1)

        # The head starts at the training mean, so an untrained model predicts
        # the mean -- which gate 3 then CHECKS rather than assumes.
        head = self._classifier()
        with torch.no_grad():
            if hasattr(head, "weight"):
                torch.nn.init.zeros_(head.weight)
            if getattr(head, "bias", None) is not None:
                head.bias.fill_(float(np.mean(train_labels)))

        # FULL fine-tuning [DECIDED, brief §2.3]: nothing is frozen here.
        # 5,499 images is enough, and it is the point of the exercise.
        for parameter in self._model.parameters():
            parameter.requires_grad_(True)
        self.parameter_report = count_parameters(self._model)

        self._model.to(device)
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _prepare(self, features: np.ndarray) -> Any:
        import torch

        device = next(self._model.parameters()).device
        batch = torch.as_tensor(np.asarray(features, dtype=np.float32) / 255.0)
        batch = batch.permute(0, 3, 1, 2).to(device)
        return (batch - self._mean) / self._std

    def _forward(self, batch, boxes: np.ndarray | None):
        """The model call, with generated boxes when a scheme supplies them."""
        if boxes is None:
            return self._model(batch)
        import torch

        return self._model(
            batch,
            torch.as_tensor(
                np.asarray(boxes, dtype=np.float32), device=batch.device
            ),
        )

    def train_batch(
        self, features: np.ndarray, labels: np.ndarray,
        boxes: np.ndarray | None = None,
    ) -> float:
        import torch

        self._model.train()
        device = next(self._model.parameters()).device
        target = torch.as_tensor(np.asarray(labels, dtype=np.float32), device=device)
        self._optimizer.zero_grad(set_to_none=True)
        prediction = self._forward(self._prepare(features), boxes).squeeze(-1)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        self._optimizer.step()
        return float(loss.item())

    def predict(
        self, features: np.ndarray, boxes: np.ndarray | None = None
    ) -> np.ndarray:
        import torch

        self._model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(features), self.batch_size):
                stop = start + self.batch_size
                batch = self._prepare(features[start:stop])
                sliced = None if boxes is None else boxes[start:stop]
                out.append(
                    self._forward(batch, sliced).squeeze(-1).float().cpu().numpy()
                )
        return np.concatenate(out) if out else np.empty(0)

    def state_arrays(self, *, include_optimizer: bool = True) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for key, value in self._model.state_dict().items():
            arrays[MODEL_PREFIX + key] = value.detach().cpu().numpy().copy()
        if include_optimizer:
            state = self._optimizer.state_dict()["state"]
            for index, slots in state.items():
                for slot, value in slots.items():
                    arrays[f"{OPTIMIZER_PREFIX}{index}__{slot}"] = (
                        value.detach().cpu().numpy().copy()
                    )
        return arrays

    def load_state_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        import torch

        model_state = {
            key[len(MODEL_PREFIX):]: torch.as_tensor(np.asarray(value))
            for key, value in arrays.items()
            if key.startswith(MODEL_PREFIX)
        }
        expected = set(self._model.state_dict())
        if set(model_state) != expected:
            missing = sorted(expected - set(model_state))[:3]
            surplus = sorted(set(model_state) - expected)[:3]
            raise PretrainError(
                f"checkpoint model arrays do not match this model: missing "
                f"{missing}, surplus {surplus}. Resuming a different "
                "architecture is not a resume."
            )
        self._model.load_state_dict(model_state)

        slots: dict[int, dict[str, Any]] = {}
        for key, value in arrays.items():
            if not key.startswith(OPTIMIZER_PREFIX):
                continue
            index_text, slot = key[len(OPTIMIZER_PREFIX):].split("__", 1)
            slots.setdefault(int(index_text), {})[slot] = torch.as_tensor(
                np.asarray(value)
            )
        if slots:
            state = self._optimizer.state_dict()
            state["state"] = slots
            self._optimizer.load_state_dict(state)


def load_declared_init(model, init_dir) -> dict:
    """Load a snapshot artifact's weights with STRICT accounting.

    The snapshot holds the upstream pretrained state at ``num_classes=0`` --
    backbone weights only, no head. Loading it into the training model
    (``num_outputs=1``) therefore expects missing keys that are EXACTLY the
    head's, and nothing unexpected; anything else means the snapshot and the
    model disagree about what the backbone is, and a silent partial load
    would be the undeclared input back in a quieter form.
    """
    import torch

    path = Path(init_dir) / INIT_WEIGHTS_NAME
    if not path.is_file():
        raise PretrainError(
            f"the declared init artifact has no {INIT_WEIGHTS_NAME}: {path}. "
            "Build it with the snapshot_pretrained_init task."
        )
    arrays = np.load(path)
    state = {key: torch.as_tensor(np.asarray(arrays[key])) for key in arrays.files}

    model_state = model.state_dict()
    mismatched = sorted(
        key for key in state
        if key in model_state and tuple(model_state[key].shape) != tuple(state[key].shape)
    )
    if mismatched:
        raise PretrainError(
            f"{len(mismatched)} snapshot keys have different shapes in the "
            f"built model, e.g. {mismatched[:4]}. The snapshot was taken for "
            "a different build of this backbone."
        )
    result = model.load_state_dict(state, strict=False)
    unexpected = sorted(result.unexpected_keys)
    stray = sorted(k for k in result.missing_keys if not k.startswith("head"))
    if unexpected or stray:
        raise PretrainError(
            "the declared init does not account for the model: "
            f"unexpected {unexpected[:4]}, non-head missing {stray[:4]}. "
            "Refusing a partial load."
        )
    return {
        "loaded": len(state),
        "head_initialised_fresh": sorted(result.missing_keys),
    }


def make_model(
    backbone: str, config: PretrainConfig, init_dir=None
) -> PretrainModel:
    """The model for a config's ``backbone``. "stub" is the laptop path."""
    if backbone == "stub":
        return StubPretrainModel()
    return TorchPretrainModel(
        name=backbone,
        deterministic=config.deterministic,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        batch_size=config.batch_size,
        seed=config.seed,
        init_dir=init_dir,
    )


# --------------------------------------------------------------------------
# data: the split, and the two feature sources
# --------------------------------------------------------------------------


def read_split_or_refuse(
    root: Path, *, expect_train: int, expect_test: int
) -> tuple[dict[str, float], dict[str, float]]:
    """Both sides of the official split, counts checked against the config.

    The declared expectation is the runtime invariant (R3): the shipped configs
    declare 3300/2199 -- asserted by test -- so a SCUT copy with a different
    split cannot quietly train. Overlap is refused outright: training on test
    images would inflate the exact number the band check reads.
    """
    from ..scut import labels as scut_labels

    train = scut_labels.read_labelled_split(root, "train")
    test = scut_labels.read_labelled_split(root, "test")

    overlap = set(train) & set(test)
    if overlap:
        raise PretrainError(
            f"{len(overlap)} stem(s) appear on BOTH sides of the split, e.g. "
            f"{sorted(overlap)[:5]}. Training on test images would inflate the "
            "correlation the band check reads."
        )
    for side, got, expected in (("train", len(train), expect_train),
                                ("test", len(test), expect_test)):
        if got != expected:
            raise PretrainError(
                f"config declares expect_{side}={expected} but the split has "
                f"{got} after exclusion. For the official split the shipped "
                "configs declare 3300/2199 (CM152 is on the test side); a "
                "disagreement means this is not the data the config claims."
            )
    return train, test


def load_original_features(
    root: Path, stems: list[str], log=lambda *_: None
) -> tuple[np.ndarray, np.ndarray]:
    """Raw SCUT staged through the FROZEN path; returns (images, content boxes).

    SCUT ships 350x350, so in practice ``stage`` is a pure resize and the white
    pad never appears -- but going through the frozen composition keeps the
    original variant's resampling identical to the masked variants', so
    original-vs-masked is about the mask rather than about interpolation. The
    content boxes come off each ``Staged`` directly -- read, not assumed
    square -- and feed the generated region schemes.
    """
    from ..geometry import render
    from ..geometry.staging import stage
    from ..scut.dataset import IMAGE_DIR

    images = []
    boxes = []
    for count, stem in enumerate(stems, 1):
        path = Path(root) / IMAGE_DIR / f"{stem}.jpg"
        if not path.is_file():
            raise PretrainError(
                f"{path} is missing: the split names {stem!r} but the image "
                "directory does not carry it. The labels and the images are "
                "out of step, which is not a gap to skip over."
            )
        staged = stage(render.load_image(path))
        images.append(staged.image)
        boxes.append(staged.content_box)
        if count % 1000 == 0:
            log(f"  staged {count}/{len(stems)} originals")
    return np.stack(images), np.array(boxes, dtype=np.int64)


def load_masked_features(artifact_dir: Path, geometry: str, stems: list[str]) -> np.ndarray:
    """Rows of the masked artifact for ``stems``, aligned by faces.json.

    Alignment is BY STEM through the artifact's own index, and every requested
    stem must be present: a positional read would train every face against
    another face's label and still produce a plausible number -- the embedding
    module's row-order lesson, applied where it applies here.
    """
    artifact_dir = Path(artifact_dir)
    faces_path = artifact_dir / "faces.json"
    if not faces_path.is_file():
        raise PretrainError(f"{faces_path} does not exist; not a masked SCUT artifact")
    faces = json.loads(faces_path.read_text(encoding="utf-8"))

    index_of: dict[str, int] = {}
    for position, entry in enumerate(faces):
        stem = entry["stem"]
        if stem in index_of:
            raise PretrainError(
                f"faces.json lists {stem!r} twice (rows {index_of[stem]} and "
                f"{position}); the artifact index cannot be trusted"
            )
        index_of[stem] = position

    missing = [stem for stem in stems if stem not in index_of]
    if missing:
        raise PretrainError(
            f"{len(missing)} split stem(s) are not in the masked artifact, "
            f"e.g. {missing[:5]}. The artifact and the split disagree about "
            "what the dataset contains."
        )

    array_path = artifact_dir / f"masked_{geometry}.npy"
    if not array_path.is_file():
        raise PretrainError(f"{array_path} does not exist")
    array = np.load(array_path, mmap_mode="r")
    if array.shape[0] != len(faces):
        raise PretrainError(
            f"masked_{geometry}.npy has {array.shape[0]} rows but faces.json "
            f"has {len(faces)}: the index does not describe the arrays beside it"
        )
    rows = np.array([index_of[stem] for stem in stems], dtype=int)
    return np.array(array[rows])


def load_features(
    *,
    source: str,
    scut_root: Path,
    masked_dir: Path | None,
    stems: list[str],
    with_boxes: bool = False,
    log=lambda *_: None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """(features, content boxes or None) for one side of the split.

    Row i is stems[i]. Content boxes are fetched only when a generated region
    scheme needs them (``with_boxes``): for originals they come off each
    ``Staged`` for free, for masked sources from the artifact's own per-face
    records.
    """
    if source not in SOURCES:
        raise PretrainError(f"unknown source {source!r}; expected one of {SOURCES}")
    if source == "original":
        features, boxes = load_original_features(scut_root, stems, log)
        return features, (boxes if with_boxes else None)
    if masked_dir is None:
        raise PretrainError(
            f"source {source!r} needs the masked artifact and none was given"
        )
    geometry = source.removeprefix("masked_")
    features = load_masked_features(masked_dir, geometry, stems)
    boxes = (
        load_masked_content_boxes(masked_dir, geometry, stems)
        if with_boxes
        else None
    )
    return features, boxes


# --------------------------------------------------------------------------
# region-scheme boxes: Phase 2's generators, each face's own content box
# --------------------------------------------------------------------------


def backbone_kind(backbone: str) -> str:
    """"stub", or the factory registry's kind. Importable without torch."""
    if backbone == "stub":
        return "stub"
    from ..models.factory import backbone_spec

    return backbone_spec(backbone)["kind"]


def check_scheme_for_backbone(backbone: str, region_scheme: str) -> None:
    """Refuse a scheme that cannot act on this backbone.

    A transformer given "grid" would train identically and record a scheme it
    never used -- the knob-that-cannot-act defect. A graph backbone given
    "none" would silently run its native regions under a config that does not
    say so.
    """
    if region_scheme not in REGION_SCHEMES:
        raise PretrainError(
            f"unknown region_scheme {region_scheme!r}; expected one of "
            f"{REGION_SCHEMES}"
        )
    kind = backbone_kind(backbone)
    if kind == "graph" and region_scheme == "none":
        raise PretrainError(
            f"backbone {backbone!r} is a graph model and needs a region "
            "scheme: 'native' for the architecture's own generator, or one of "
            f"{GENERATED_SCHEMES}."
        )
    if kind != "graph" and region_scheme != "none":
        raise PretrainError(
            f"backbone {backbone!r} has no region structure, so region_scheme "
            f"{region_scheme!r} could not change its run. Declare 'none'."
        )


def scheme_frame_boxes(
    region_scheme: str, geometry: str, content_boxes: np.ndarray
) -> np.ndarray:
    """(N, R, 4) frame-pixel (x, y, w, h) boxes, one row set per face.

    Phase 2's machinery unchanged: ``patches_for`` generates the scheme's
    normalised patches once, and ``map_patches`` places them through each
    face's own content box -- the same per-image mapping the patch probes use,
    with its inside-content assertion intact. Two faces with different aspect
    ratios therefore get different pixel boxes for the same scheme, which is
    the property ``assert_varies_with_aspect_ratio`` guards in Phase 2.
    """
    from ..geometry import patch_features
    from ..geometry.mapping import map_patches
    from ..geometry.staging import OUTPUT_SIZE, Staged

    if region_scheme not in GENERATED_SCHEMES:
        raise PretrainError(
            f"scheme_frame_boxes is for {GENERATED_SCHEMES}; {region_scheme!r} "
            "does not generate boxes"
        )
    patches = patch_features.patches_for(region_scheme, geometry)

    canvas = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)
    all_boxes = np.empty((len(content_boxes), len(patches), 4), dtype=np.float32)
    for index, box in enumerate(content_boxes):
        shim = Staged(
            image=canvas,
            source_size=(int(box[2]), int(box[3])),
            content_box=tuple(int(v) for v in box),
            scale=1.0,
        )
        mapped = map_patches(shim, patches)
        all_boxes[index] = np.array([m.pixels for m in mapped], dtype=np.float32)
    return all_boxes


def boxes_for_model(backbone: str, frame_boxes: np.ndarray) -> np.ndarray:
    """Convert frame-pixel (x, y, w, h) into the backbone's own convention.

    SR-GNN pools on its 42x42 map in map-pixel (x, y, w, h); AG-Net's
    roi_align takes normalised (x1, y1, x2, y2). The conversion lives here so
    the models' verified forwards keep their Stage 1 conventions untouched.
    """
    from ..geometry.staging import OUTPUT_SIZE

    boxes = np.asarray(frame_boxes, dtype=np.float32)
    if backbone == "srgnn":
        from ..models.srgnn import ROI_RESOLUTION

        return boxes * (ROI_RESOLUTION / OUTPUT_SIZE)
    if backbone == "agnet":
        out = boxes / OUTPUT_SIZE
        out[..., 2] = out[..., 0] + out[..., 2]
        out[..., 3] = out[..., 1] + out[..., 3]
        return out
    raise PretrainError(
        f"no box convention registered for backbone {backbone!r}; add it here "
        "when a new graph backbone lands rather than guessing a format"
    )


def load_masked_content_boxes(
    artifact_dir: Path, geometry: str, stems: list[str]
) -> np.ndarray:
    """(N, 4) recorded content boxes for ``stems``, aligned by faces.json.

    The masked artifact records ``content_box`` per face per geometry -- the
    frozen ``stage()``'s own output at build time -- precisely so this needs no
    re-derivation. Same stem-alignment discipline as the feature loader.
    """
    artifact_dir = Path(artifact_dir)
    faces = json.loads((artifact_dir / "faces.json").read_text(encoding="utf-8"))
    by_stem: dict[str, list] = {}
    for entry in faces:
        record = entry.get(geometry)
        if record is None or "content_box" not in record:
            raise PretrainError(
                f"faces.json entry for {entry.get('stem')!r} carries no "
                f"{geometry} content_box; this artifact predates the recorded "
                "geometry and cannot feed a generated region scheme"
            )
        by_stem[entry["stem"]] = record["content_box"]
    missing = [stem for stem in stems if stem not in by_stem]
    if missing:
        raise PretrainError(
            f"{len(missing)} stem(s) have no recorded content box, e.g. "
            f"{missing[:5]}"
        )
    return np.array([by_stem[stem] for stem in stems], dtype=np.int64)


# --------------------------------------------------------------------------
# the band check
# --------------------------------------------------------------------------


def band_check(
    source: str, backbone: str, test_pcc: float, region_scheme: str = "none"
) -> dict:
    """The consumption verdict, per source. Recorded in metrics.json.

    Original: pass/fail against ``FAILURE_FLOOR``, with the published band for
    context. Masked: no verdict HERE -- the reference is the same backbone's
    SAME-SCHEME original run, which this run cannot see, so the verdict is
    deferred to review by name rather than fabricated.
    """
    if source not in SOURCES:
        raise PretrainError(f"unknown source {source!r}")
    arm = backbone if region_scheme == "none" else f"{backbone} {region_scheme}"
    if source == "original":
        consumable = bool(np.isfinite(test_pcc) and test_pcc >= FAILURE_FLOOR)
        return {
            "applies": True,
            "arm": arm,
            "test_pcc": test_pcc,
            "failure_floor": FAILURE_FLOOR,
            "published_band": list(PUBLISHED_BAND),
            "published_band_provenance": (
                "[LITERATURE, brief §3.1] ResNeXt-50 ~0.90, label-distribution "
                "methods ~0.95, recent hybrids ~0.926 on SCUT-FBP5500"
            ),
            "checkpoint_consumable": consumable,
            "note": (
                "consumable: test PCC at or above the floor. A run well below "
                "it means pretraining failed, and nothing downstream may "
                "consume its checkpoint."
                if consumable
                else "PRETRAINING FAILED: test PCC is below the floor. Do not "
                "extract embeddings from this checkpoint; diagnose the run "
                "instead."
            ),
        }
    return {
        "applies": False,
        "test_pcc": test_pcc,
        "checkpoint_consumable": None,
        "decided_at_review": True,
        "compare_against": f"{arm} original",
        "note": (
            "masked variants are compared against their OWN backbone's "
            "same-scheme original run, never the published band: the mask "
            "removes information the beauty rating depends on, so a lower "
            "score is expected and is not evidence of failure. This run "
            "cannot see the original run's number, so the verdict is taken "
            "at review, not fabricated here."
        ),
    }


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


@dataclass
class PretrainResult:
    paused: bool
    summary: dict
    #: (stem, truth, prediction) per test face, in row order. Public SCUT data.
    test_table: list[tuple[str, float, float]] = field(default_factory=list)


def _score(inner_mse: float, inner_pcc: float, monitor: str) -> float:
    score = inner_mse if monitor == "inner_val_mse" else -inner_pcc
    return float("inf") if np.isnan(score) else float(score)


def _truncate_journal(path: Path, keep_rows: int) -> None:
    """Cut ``curves.csv`` back to the checkpointed row count (plus header)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) - 1 < keep_rows:
        raise PretrainError(
            f"{path.name} has {len(lines) - 1} rows but the checkpoint claims "
            f"{keep_rows} durable epochs. The journal is short of its "
            "checkpoint, so epochs the run believes are durable are missing; "
            "delete the run directory and start over rather than resuming "
            "into a gap."
        )
    # Bytes, not text mode: on Windows write_text translates \n to \r\n, which
    # would make a truncated-and-regrown journal differ from an uninterrupted
    # one at exactly the bytes the resume test compares.
    path.write_bytes(
        "".join(line + "\n" for line in lines[: 1 + keep_rows]).encode("utf-8")
    )


def run_pretraining(
    *,
    scut_root: Path,
    source: str,
    backbone: str,
    region_scheme: str,
    masked_dir: Path | None,
    expect_train: int,
    expect_test: int,
    run_dir: Path,
    curves_path: Path,
    pretrained_path: Path,
    config: PretrainConfig,
    model_factory: Callable[[], PretrainModel] | None = None,
    init_dir: Path | None = None,
    log=lambda *_: None,
    stop_after_epochs: int | None = None,
    on_epoch: Callable[[int, PretrainModel], None] | None = None,
    labels_by_stem: tuple | None = None,
) -> PretrainResult:
    """One pretraining run, resumable at epoch granularity.

    ``stop_after_epochs`` is a TEST SEAM simulating the pod being killed after
    that many epochs of this invocation: the resume tests are what give the
    checkpoint wiring teeth on a machine with no Run:AI to pause it.

    ``on_epoch`` is an OBSERVATION hook, the 8c ``on_step`` pattern one level
    up: called as ``on_epoch(0, model)`` once on a fresh start (the untrained
    state) and ``on_epoch(epoch, model)`` after each epoch's evaluation. The
    caller must leave the model and every RNG stream exactly as found -- the
    loop does not fence it, the same contract the head hook carries.
    """
    run_dir = Path(run_dir)
    check_scheme_for_backbone(backbone, region_scheme)
    generated = region_scheme in GENERATED_SCHEMES

    # **[2026-08-24] ``labels_by_stem`` threads a NON-SCUT split through
    # the same recipe** -- the ``box``/``size`` pattern again, and the
    # third time this phase. Default None keeps every existing call
    # byte-identical: SCUT still reads its own split from its own root.
    # Phase 15 passes MEBeauty's SHIPPED universal split, restricted to
    # the staged survivors, because the recipe must be the Phase 6 one
    # and only the DATA may differ (phase15.STOP_3_AMENDED).
    if labels_by_stem is None:
        train_labels_by_stem, test_labels_by_stem = read_split_or_refuse(
            Path(scut_root), expect_train=expect_train,
            expect_test=expect_test,
        )
    else:
        train_labels_by_stem, test_labels_by_stem = labels_by_stem
        if len(train_labels_by_stem) != expect_train or (
            len(test_labels_by_stem) != expect_test
        ):
            raise PretrainError(
                f"supplied split is {len(train_labels_by_stem)}/"
                f"{len(test_labels_by_stem)}, declared "
                f"{expect_train}/{expect_test} -- the counts are asserted "
                "whoever supplies them"
            )

    # Row order is sorted stem, recorded in the summary. Sorted rather than
    # file order so the order is a property of the SET, not of one file's
    # incidental layout.
    train_stems = sorted(train_labels_by_stem)
    test_stems = sorted(test_labels_by_stem)
    train_labels_all = np.array([train_labels_by_stem[s] for s in train_stems])
    test_labels = np.array([test_labels_by_stem[s] for s in test_stems])

    log(f"loading features: source {source!r}, scheme {region_scheme!r}, "
        f"{len(train_stems)} train / {len(test_stems)} test")
    train_features_all, train_content = load_features(
        source=source, scut_root=Path(scut_root), masked_dir=masked_dir,
        stems=train_stems, with_boxes=generated, log=log,
    )
    test_features, test_content = load_features(
        source=source, scut_root=Path(scut_root), masked_dir=masked_dir,
        stems=test_stems, with_boxes=generated, log=log,
    )

    # Generated schemes: Phase 2's patches placed through each face's own
    # content box, converted once into the backbone's convention. [DECIDED]
    # For the ORIGINAL source the full-square (G2) parameterisation is used --
    # an unmasked face has no trapezium, and G2's mask IS the full frame.
    train_boxes_all = test_boxes = None
    n_regions = None
    if generated:
        patch_geometry = (
            "g2" if source == "original" else source.removeprefix("masked_")
        )
        train_frame = scheme_frame_boxes(region_scheme, patch_geometry, train_content)
        test_frame = scheme_frame_boxes(region_scheme, patch_geometry, test_content)
        train_boxes_all = boxes_for_model(backbone, train_frame)
        test_boxes = boxes_for_model(backbone, test_frame)
        n_regions = int(train_boxes_all.shape[1])
        log(f"region scheme {region_scheme!r}: {n_regions} boxes per face "
            f"({patch_geometry} parameterisation)")

    # Inner validation from the TRAIN side, by the frozen implementation.
    # fold=0: there is one "fold" here. Deterministic given the seed, so a
    # resumed run recomputes the same split without consulting the RNG stream.
    train_index, inner_index = inner_val_split(
        list(range(len(train_stems))), config.inner_val_frac, config.seed, 0
    )
    fit_rows = np.array(train_index, dtype=int)
    inner_rows = np.array(inner_index, dtype=int)
    fit_features = train_features_all[fit_rows]
    fit_labels = train_labels_all[fit_rows]
    inner_features = train_features_all[inner_rows]
    inner_labels = train_labels_all[inner_rows]
    fit_boxes = None if train_boxes_all is None else train_boxes_all[fit_rows]
    inner_boxes = None if train_boxes_all is None else train_boxes_all[inner_rows]

    model = (model_factory or (lambda: make_model(backbone, config, init_dir)))()
    model.reset(fit_features, fit_labels)

    # FULL fine-tuning is the policy [DECIDED]; a model reporting anything
    # frozen did not run the arm this config declares.
    report = model.parameter_report
    if report.get("trainable_parameters") != report.get("total_parameters"):
        raise PretrainError(
            f"pretraining is FULL fine-tuning, but the model reports "
            f"{report.get('trainable_parameters')} trainable of "
            f"{report.get('total_parameters')} total. A partially frozen "
            "pretraining run would produce a checkpoint that pretrained "
            "nothing while looking like one that did."
        )

    fingerprint = {
        "backbone": backbone,
        "source": source,
        "region_scheme": region_scheme,
        "seed": config.seed,
        "monitor": config.monitor,
        "n_fit": len(train_index),
        "n_inner_val": len(inner_index),
        "n_test": len(test_stems),
    }

    rng = np.random.default_rng(config.seed)
    epoch = 0
    best_epoch = 0
    best_score: float | None = None
    best_arrays: dict[str, np.ndarray] | None = None
    resumed = False
    epoch0_record: dict | None = None

    found = ckpt.latest(run_dir)
    if found is not None:
        state = ckpt.load(found)
        recorded = state.extra.get("fingerprint", {})
        if recorded != fingerprint:
            raise PretrainError(
                f"checkpoint fingerprint {recorded} does not match this run "
                f"{fingerprint}. Resuming a different arm's checkpoint is not "
                "a resume, it is a different run wearing this one's directory."
            )
        model.load_state_arrays(
            {k: v for k, v in state.arrays.items() if not k.startswith(BEST_PREFIX)}
        )
        best_arrays = {
            k[len(BEST_PREFIX):]: v
            for k, v in state.arrays.items()
            if k.startswith(BEST_PREFIX)
        } or None
        ckpt.restore_numpy_rng(rng, state.numpy_rng)
        ckpt.restore_torch_rng(state.torch_rng, state.torch_cuda_rng)
        epoch = state.step
        best_epoch = state.extra["best_epoch"]
        best_score = state.extra["best_score"]
        _truncate_journal(curves_path, state.output_rows["curves"])
        resumed = True
        epoch0_record = state.extra.get("epoch0")
        log(f"RESUMING at epoch {epoch}, best {best_epoch} "
            f"({config.monitor} {best_score})")
    else:
        # Gate 3, on a FRESH start only: after a resume the model is trained
        # and the untrained-head property no longer holds, by design.
        epoch0_predictions = model.predict(inner_features, inner_boxes)
        epoch0_mean = float(np.mean(epoch0_predictions))
        epoch0_mse = float(np.mean((epoch0_predictions - inner_labels) ** 2))
        assert_epoch0_calibrated(
            pred_mean=epoch0_mean,
            inner_val_mse=epoch0_mse,
            inner_val_var=float(np.var(inner_labels)),
            inner_val_mean=float(np.mean(inner_labels)),
            train_label_mean=float(np.mean(fit_labels)),
            train_label_var=float(np.var(fit_labels)),
            mean_tol=config.epoch0_mean_tol,
            mse_tol=config.epoch0_mse_tol,
            where="pretrain epoch 0",
        )
        epoch0_record = {"pred_mean": epoch0_mean, "inner_val_mse": epoch0_mse}
        # Bytes for the same reason as _truncate_journal: the journal is LF on
        # every platform, matching the appends below (newline="").
        curves_path.write_bytes(b"epoch,train_loss,inner_val_mse,inner_val_pcc\n")

    # Epoch 0 for the observation hook, FRESH starts only: after a resume
    # that state is trained away, and whatever the hook derived from it is
    # already on disk from the invocation that saw it.
    if on_epoch is not None and not resumed:
        on_epoch(0, model)

    epochs_this_invocation = 0
    while epoch < config.epochs:
        if stop_after_epochs is not None and epochs_this_invocation >= stop_after_epochs:
            log(f"paused (test seam) after epoch {epoch}")
            return PretrainResult(paused=True, summary={"epochs_completed": epoch})

        epoch += 1
        epochs_this_invocation += 1

        # The loop owns the shuffle, so batch order lives in the checkpointed
        # RNG stream rather than inside the model.
        order = rng.permutation(len(fit_labels))
        total, seen = 0.0, 0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            loss = model.train_batch(
                fit_features[index], fit_labels[index],
                None if fit_boxes is None else fit_boxes[index],
            )
            total += loss * len(index)
            seen += len(index)
        train_loss = total / max(seen, 1)

        inner_predictions = model.predict(inner_features, inner_boxes)
        inner_mse = float(np.mean((inner_predictions - inner_labels) ** 2))
        inner_pcc = metrics.pcc(inner_labels, inner_predictions)

        with curves_path.open("a", encoding="utf-8", newline="") as handle:
            handle.write(f"{epoch},{train_loss!r},{inner_mse!r},{inner_pcc!r}\n")

        # Best-checkpoint selection only -- NOTHING here stops the run. The
        # budget is fixed [DECIDED 2026-07-31, FIXED_BUDGET_POLICY]: patience
        # terminated on noise in a flat region and made run length track
        # luck rather than scheme.
        score = _score(inner_mse, inner_pcc, config.monitor)
        if best_score is None or score < best_score - 1e-12:
            best_score, best_epoch = score, epoch
            best_arrays = model.state_arrays(include_optimizer=False)

        log(f"epoch {epoch}/{config.epochs}: train_loss {train_loss:.4f}, "
            f"inner_val_mse {inner_mse:.4f}, inner_val_pcc {inner_pcc:.4f}"
            + (" *" if best_epoch == epoch else ""))

        if on_epoch is not None:
            on_epoch(epoch, model)

        # Checkpoint on schedule -- but never for a state the run is done
        # with: at the budget the deliverable is written instead.
        if (
            epoch < config.epochs
            and epoch % config.checkpoint_every == 0
        ):
            torch_rng, cuda_rng = ckpt.capture_torch_rng()
            arrays = model.state_arrays(include_optimizer=True)
            for key, value in (best_arrays or {}).items():
                arrays[BEST_PREFIX + key] = value
            ckpt.save(
                run_dir / ckpt.CHECKPOINT_NAME,
                ckpt.Checkpoint(
                    step=epoch,
                    epoch=epoch,
                    arrays=arrays,
                    numpy_rng=ckpt.capture_numpy_rng(rng),
                    output_rows={"curves": epoch},
                    extra={
                        "fingerprint": fingerprint,
                        "best_epoch": best_epoch,
                        "best_score": best_score,
                        "epoch0": epoch0_record,
                    },
                    torch_rng=torch_rng,
                    torch_cuda_rng=cuda_rng,
                ),
            )

    # ---- the deliverable: best weights, evaluated once on the test side ----
    if best_arrays is None:
        raise PretrainError("no epoch ever completed; nothing to evaluate")
    model.load_state_arrays(best_arrays)
    test_predictions = model.predict(test_features, test_boxes)

    summary = {
        "task": "pretrain",
        "backbone": backbone,
        "source": source,
        "region_scheme": region_scheme,
        "n_regions": n_regions,
        "seed": config.seed,
        "n_train": len(train_stems),
        "n_fit": len(train_index),
        "n_inner_val": len(inner_index),
        "n_test": len(test_stems),
        "row_order": "sorted stem",
        "label_source": "the official split files (filename rating per line)",
        "epochs_budget": config.epochs,
        "epochs_run": epoch,
        "selected_epoch": best_epoch,
        "selection_policy": FIXED_BUDGET_POLICY,
        "resumed": resumed,
        "epoch0": epoch0_record,
        "test": {
            "pcc": metrics.pcc(test_labels, test_predictions),
            "spearman": metrics.spearman(test_labels, test_predictions),
            "mae": metrics.mae(test_labels, test_predictions),
            "rmse": metrics.rmse(test_labels, test_predictions),
        },
        "train_config": {
            "epochs": config.epochs,
            "inner_val_frac": config.inner_val_frac,
            "monitor": config.monitor,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "checkpoint_every_epochs": config.checkpoint_every,
            "deterministic": config.deterministic,
        },
        "parameters": dict(model.parameter_report),
        "trainable_policy": "full fine-tuning, asserted at reset",
    }
    summary["band_check"] = band_check(
        source, backbone, summary["test"]["pcc"], region_scheme
    )

    # The best weights in the verified checkpoint format: same loader, same
    # digest verification, for whatever extracts embeddings from this.
    ckpt.save(
        pretrained_path,
        ckpt.Checkpoint(
            step=best_epoch,
            epoch=best_epoch,
            arrays={MODEL_PREFIX + k if not k.startswith(MODEL_PREFIX) else k: v
                    for k, v in best_arrays.items()},
            numpy_rng=ckpt.capture_numpy_rng(rng),
            output_rows={},
            extra={
                "fingerprint": fingerprint,
                "selected_epoch": best_epoch,
                "test_pcc": summary["test"]["pcc"],
                "band_check": summary["band_check"],
            },
        ),
    )
    # Resume state is not an output. Leaving it would offer a completed run
    # something to resume into.
    (run_dir / ckpt.CHECKPOINT_NAME).unlink(missing_ok=True)

    table = [
        (stem, float(truth), float(prediction))
        for stem, truth, prediction in zip(test_stems, test_labels, test_predictions)
    ]
    return PretrainResult(paused=False, summary=summary, test_table=table)
