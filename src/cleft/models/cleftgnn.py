"""CleftGNN, built to the group's own artifacts under Phase 10's registration.

Component provenance (``phase10.PHASE_10_REGISTERED``, conflicts recorded
there per component):

* **regions**: the NOTEBOOK'S enumeration -- all 3x3 spans INCLUDING the
  nine 1x1 cells, whole image appended, de-duplicated: 36 regions
  (``phase10.NOTEBOOK_ROI_SET_MEASURED``; the maintainer's re-decision
  2026-08-16: "what the group ran" survives the measurement).
* **backbone**: ResNet-50, ImageNet init, Table-1 best, **FROZEN**.
  [FIDELITY CORRECTION 2026-08-17, phase10.BACKBONE_IS_FROZEN] The earlier
  "full fine-tune" was an INFERENCE from the manuscript's "ResNet-50
  backbone", not their recipe. The notebook -- the only executable
  artifact the group ran -- freezes its backbone, and lr 0.01 belongs to
  that regime. Frozen means ``requires_grad=False`` AND ``eval()`` during
  training, so BatchNorm running statistics do not drift: the void
  ladder's own defect was missing BN running statistics, and leaving that
  implicit here is how it would return.
* **RoI pooling / GNN mechanics**: the notebook -- roi_align at pool 7,
  GAP, Linear to 512; MLP 512->1024->1024; APPNP K=1 alpha=0.3 over the
  complete graph (with self-loops the symmetric-normalised adjacency of a
  complete graph is J/N, so one propagation step is the node MEAN --
  implemented directly, the srgnn port's verified reading); sigmoid at
  K==1 per the notebook.
* **gated attention pooling**: the notebook -- sigmoid(W1 f) * (W2 f),
  summed over regions.
* **eq (9)'s sum**: the branches are summed plainly as the manuscript
  states, with the GNN branch NORMALISED first
  (``phase10.GNN_BRANCH_NORM``) -- the measured correction for a 5.8x
  magnitude imbalance that drowned the SABM branch.
* **SABM**: the MANUSCRIPT alone (absent from the notebook): eq (7)
  m_i = VerticalCenter(r_i)/224 (the accepted normalisation, committed
  blind); eq (8) v_a = sum_i(F_att,i * m_i), literal, no extra
  nonlinearity; eq (9) v_r = f_t + v_a, ADDITIVE. The attention's elided
  argument resolved by the notebook: tanh(QK^T/sqrt(d)) before Softmax.
* **classifier**: Linear(1024 -> 5), grades Excellent(1)..Very Poor(5).

``CleftGNNBackbone`` implements the frozen harness's protocol with the
registered recipe: cross-entropy on the consensus grade, plain SGD
lr 0.01 momentum 0 (literal; the no-momentum caveat travels), batch 16
[CORRECTED 2026-08-17: the notebook DOES state ``BATCH_SIZE = 16``, cell
9 -- ``phase10.NOTEBOOK_RECIPE_IS_ADAM``; the value was right by
coincidence and the claim about the artifact was wrong],
LayerNorm on the fused feature before the classifier and a Laplace-biased
head with the framework's own small-random weights
(phase10.FUSED_NORM_AND_STANDARD_INIT, phase10.THIN_CLASS_SMOOTHING): the
bias keeps epoch 0 near the train-fold label mean with every grade
reachable, and the norm keeps the first step from blowing up on a fused
feature that measures ~21 before it. ``predict`` returns the
softmax-EXPECTED grade (the registered primary reading); Top-1 rides
through ``predict_probabilities`` for the beside-reading.

**TWO RECIPES, ONE MECHANISM** (``recipe=``, ``phase10.NOTEBOOK_RECIPE_
CELL_REGISTERED``). Everything above describes ``recipe="manuscript"``,
which is unchanged. ``recipe="notebook"`` is the group's OTHER artifact
-- the only executable one they have shared -- and it differs in exactly
four places, every one of them verified in the notebook's own source:

* **backbone**: a frozen ViT-B/16 (``vit_base_patch16_224``,
  ``num_classes=0``) read through the notebook's OWN extraction path --
  ``patch_embed``, concatenated class token, ``+ pos_embed``, ``blocks``,
  class token dropped, reshaped to (B, 768, 14, 14). The notebook does
  NOT apply the final ``model.norm``; that omission is replicated rather
  than repaired, because this cell exists to run their code, not a
  corrected version of it (``phase10.NOTEBOOK_BACKBONE_AND_ACM_CONFIRMED``).
* **the attention head**: theirs, not the manuscript's SABM. A learned
  ``acm_w_beta`` softmax OVER REGIONS replaces eq (7)'s vertical mask,
  and a final ``sigmoid`` bounds the result into (0, 1). There is no
  ``m_vertical`` on this path at all.
* **fusion**: ``f_t + f_t * v``, i.e. f_t * (1 + v), the notebook's
  literal expression -- MULTIPLICATIVE where the manuscript's eq (9) is
  additive (``phase10.NOTEBOOK_FUSION_IS_MULTIPLICATIVE``).
* **optimiser**: ``Adam`` at lr 0.001 with torch's default betas and eps,
  which is what the notebook runs; the manuscript says SGD at 0.01.

**And it carries NEITHER LayerNorm.** ``gnn_norm`` and ``fused_norm``
were both measured corrections for the additive path's scale problems;
the notebook has no normalisation anywhere, and adding ours would make a
spanning result unreadable -- we could not tell their recipe from our
repairs. So the multiplicative fused feature is LARGER than the additive
one that broke SGD (|f_t| * (1 + v), v in (0,1)), and whether Adam
absorbs it is exactly the hypothesis this cell measures.

One implementation of everything the two share -- rois, roi_align, the
region proposer, the GNN, the gated pooling, the classifier, the harness
protocol, the stage instrument. A second copy of that arithmetic is how a
table starts describing something the model is not doing
(``phase10.PROBE_RECONSTRUCTED_THE_PIPELINE``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

IMAGE_SIZE = 224
GRID = 3
N_REGIONS = 36
NUM_CLASSES = 5
FEATURE_DIM = 512
GNN_OUT_DIM = 1024
ALPHA = 0.3
POOL_SIZE = 7

#: The two documented recipes, each faithful to one of the group's own
#: artifacts (``phase10.NOTEBOOK_RECIPE_CELL_REGISTERED``). There is no
#: third, and there will not be one without a third documented artifact.
RECIPES = ("manuscript", "notebook")

#: **[RULED 2026-08-30 -- the Phase 10 ANNEX's recipe,
#: deliberately NOT a member of ``RECIPES``.]** The sentence above
#: stands: there is still no third documented artifact. This is a RULED
#: COMPOSITION of the two that exist, for the annex's compute gate and
#: nothing else (``phase10_annex.RULING_B_FULL_TRAINABLE``,
#: ``NORMALISATION_RULED``, ``RECIPE_FIXED_TO_NOTEBOOK``): the
#: manuscript's ARCHITECTURE (ResNet-50, SABM eq 7-9 additive) with the
#: notebook's EXECUTED training values (Adam 0.001, batch 16, 5 epochs,
#: CE, last epoch is the model), the backbone FULL TRAINABLE by ruling
#: (b), the notebook's CIFAR-10 statistics by the normalisation ruling
#: -- and NONE of this project's measured repairs (no gnn_norm, no
#: fused_norm, no Laplace-biased init, no early stopping): the group's
#: numbers were produced without them, and carrying ours would make any
#: gap unreadable -- their recipe or our repairs, indistinguishable.
ANNEX_RECIPE = "annex_fullfit"

#: The notebook's normalisation constants, verbatim
#: (``phase10.NOTEBOOK_BUDGET_AND_NORMALISATION``): CIFAR-10 statistics
#: under an ImageNet-pretrained backbone -- **a defect in the notebook,
#: not a recipe choice**, and on the annex recipe it is REPLICATED
#: KNOWINGLY (``phase10_annex.NORMALISATION_RULED``; the write-up
#: sentence is "replicated knowingly," never "a recipe").
CIFAR10_NORMALISATION = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

#: The notebook's backbone, verbatim, and the patch grid its extraction
#: path hard-codes: 224 / 16 = 14.
NOTEBOOK_BACKBONE = "vit_base_patch16_224"
PATCH_RESOLUTION = 14


class CleftGNNError(RuntimeError):
    """The build or a contract around it is not usable."""


def cleftgnn_rois(image_size: int = IMAGE_SIZE, grid: int = GRID) -> np.ndarray:
    """The notebook's enumeration, verbatim semantics: ALL spans of the
    grid including the 1x1 cells, whole image appended, de-duplicated --
    (36, 4) as (x, y, w, h) in image pixels."""
    cell = image_size / grid
    rois = []
    for r in range(grid):
        for c in range(grid):
            for span_h in range(1, grid - r + 1):
                for span_w in range(1, grid - c + 1):
                    rois.append((c * cell, r * cell, span_w * cell, span_h * cell))
    rois.append((0.0, 0.0, float(image_size), float(image_size)))
    unique = sorted(set(rois))
    return np.array(unique, dtype=np.float32)


def vertical_mask(rois: np.ndarray, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Eq (7) under the accepted normalisation: m_i = y_center / image
    height -- larger toward the bottom, where the cleft anatomy sits."""
    rois = np.asarray(rois, dtype=np.float64)
    return ((rois[:, 1] + rois[:, 3] / 2.0) / image_size).astype(np.float64)


def expected_grade(probabilities: np.ndarray) -> np.ndarray:
    """The registered primary reading: sum_k p_k * k over grades 1..5."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    grades = np.arange(1, NUM_CLASSES + 1, dtype=np.float64)
    return probabilities @ grades


#: The stages a forward records, in pipeline order. One list, so the
#: probe and the arm cannot disagree about what a stage table contains.
#: Stages absent on a given recipe are simply not recorded, and
#: ``stage_ratios`` skips them -- so one order covers both paths without
#: either pretending to have run the other's stages.
STAGE_ORDER = (
    "input", "backbone_map", "region_vectors", "region_proposer",
    "gnn_mlp", "appnp", "sigmoid_f_hat", "gated_pooling_f_t",
    "sabm_v_a", "acm_v", "gnn_norm_f_t", "fused", "after_layernorm",
    "logits",
)


#: Which stages each recipe actually records. An instrument needs this to
#: tell "this recipe has no such stage" from "the model stopped recording
#: one" -- without it, a recipe-aware probe would have to rebuild the
#: answer, which is the ``phase10.PROBE_RECONSTRUCTED_THE_PIPELINE``
#: defect wearing a different hat.
RECIPE_STAGES = {
    "manuscript": tuple(
        name for name in STAGE_ORDER if name != "acm_v"
    ),
    "notebook": tuple(
        name for name in STAGE_ORDER
        if name not in ("sabm_v_a", "gnn_norm_f_t", "after_layernorm")
    ),
    # [2026-08-30] The annex: the manuscript's stages MINUS the two
    # repair stages it does not carry (and minus the notebook-only
    # acm_v, as on the manuscript path).
    "annex_fullfit": tuple(
        name for name in STAGE_ORDER
        if name not in ("acm_v", "gnn_norm_f_t", "after_layernorm")
    ),
}

#: What the classifier actually receives on each path. The gate-3
#: mechanism lives in the magnitude of THIS tensor
#: (``phase10.GATE_3_FAILED_ON_THE_NOTEBOOK_CELL``), and naming it here
#: keeps every instrument reading the same one.
CLASSIFIER_INPUT_STAGE = {
    "manuscript": "after_layernorm",
    "notebook": "fused",
    "annex_fullfit": "fused",
}


def stage_ratios(stages: dict) -> dict:
    """``{stage: (magnitude, across-image sd, ratio)}`` from a forward's
    recorded tensors.

    **One implementation, used by both the probe and the arm** -- the
    ``phase10.PROBE_RECONSTRUCTED_THE_PIPELINE`` lesson applied before it
    can be repeated: a second copy of this arithmetic is how a table
    starts describing something the model is not doing.
    """
    out: dict = {}
    for name in STAGE_ORDER:
        if name not in stages:
            continue
        flat = stages[name].reshape(stages[name].shape[0], -1)
        magnitude = float(flat.abs().mean())
        across = float(flat.std(dim=0).mean())
        out[name] = (
            magnitude, across, across / magnitude if magnitude else float("nan")
        )
    return out


def _as_grades(labels) -> np.ndarray:
    """Labels to integer grades 1..5, ASSERTED rather than rounded.

    **[2026-08-17] There is no rounding step in this model.** The training
    label is the score sheet's Median of five integer grades, which is itself
    an integer (``phase10.CONSENSUS_LABEL_IS_THE_MEDIAN``), so a rounding
    call here could only ever hide a label that had arrived as something
    else -- and ``np.round`` is half-to-EVEN, so it would not even have been
    the half-up rule anyone would assume. Refusing is the honest form.
    """
    values = np.asarray(labels, dtype=float)
    grades = values.astype(int)
    if not np.array_equal(values, grades.astype(float)):
        offenders = values[values != grades.astype(float)][:3]
        raise CleftGNNError(
            f"training labels must be integer grades; got non-integral "
            f"{offenders.tolist()}. This model trains cross-entropy on the "
            "sheet's Median column, never on a rounded continuous target."
        )
    if grades.size and (grades.min() < 1 or grades.max() > NUM_CLASSES):
        raise CleftGNNError(
            f"grades {grades.min()}..{grades.max()} outside 1..{NUM_CLASSES}"
        )
    return grades


def build(*, pretrained: bool = True, recipe: str = "manuscript"):
    """The torch model. Asserts its own region count at construction.

    ``recipe`` selects which of the group's two artifacts this instance is
    faithful to (``RECIPES``). ``"manuscript"`` is the original build and
    is unchanged; ``"notebook"`` differs in the four places the module
    docstring enumerates, and in nothing else.
    """
    import timm
    import torch
    import torch.nn as nn

    # [2026-08-30] ANNEX_RECIPE joins the admissible set WITHOUT joining
    # RECIPES: it is a ruled composition, not a third documented
    # artifact, and the sentence below stays true of documented ones.
    if recipe not in RECIPES + (ANNEX_RECIPE,):
        raise CleftGNNError(
            f"recipe {recipe!r} is not one of {RECIPES + (ANNEX_RECIPE,)}; "
            "a third documented recipe needs a third documented artifact "
            "by the group, and the annex recipe is the maintainer's ruled "
            "composition (phase10_annex), not a free slot"
        )

    class NotebookViT(nn.Module):
        """The notebook's ViT-B/16 extraction path, replicated as written.

        Frozen twice over, as the notebook freezes it: ``requires_grad =
        False`` on every parameter AND ``torch.no_grad()`` around the
        forward. The final ``model.norm`` is NOT applied -- the notebook
        does not apply it, and this cell runs their code rather than a
        corrected version of it.
        """

        def __init__(self) -> None:
            super().__init__()
            self.model = timm.create_model(
                NOTEBOOK_BACKBONE, pretrained=pretrained, num_classes=0
            )
            self.output_channels = self.model.embed_dim
            self.patch_resolution = PATCH_RESOLUTION
            for parameter in self.model.parameters():
                parameter.requires_grad = False

        def forward(self, x):
            with torch.no_grad():
                embeddings = self.model.patch_embed(x)
                embeddings = torch.cat(
                    (
                        self.model.cls_token.expand(x.shape[0], -1, -1),
                        embeddings,
                    ),
                    dim=1,
                )
                embeddings = embeddings + self.model.pos_embed
                encoded = self.model.blocks(embeddings)
                patch_tokens = encoded[:, 1:]
                batch, _, channels = patch_tokens.shape
                size = self.patch_resolution
                return patch_tokens.transpose(1, 2).reshape(
                    batch, channels, size, size
                )

    class CleftGNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recipe = recipe
            if recipe == "notebook":
                self.backbone = NotebookViT()
                channels = self.backbone.output_channels
            else:
                self.backbone = timm.create_model(
                    "resnet50", pretrained=pretrained, features_only=True,
                    out_indices=(4,),
                )
                channels = self.backbone.feature_info.channels()[-1]
            rois = cleftgnn_rois()
            if len(rois) != N_REGIONS:
                raise CleftGNNError(
                    f"{len(rois)} regions from the notebook enumeration; "
                    f"the registration fixes {N_REGIONS}"
                )
            boxes = rois.copy()
            boxes[:, 2] = rois[:, 0] + rois[:, 2]
            boxes[:, 3] = rois[:, 1] + rois[:, 3]
            self.register_buffer(
                "roi_boxes", torch.as_tensor(boxes, dtype=torch.float32)
            )
            if recipe != "notebook":
                # SABM eq (7) exists only on the manuscript-architecture
                # paths (manuscript, and [2026-08-30] the annex, which
                # runs the same architecture). The notebook has no
                # vertical mask at all, so it gets no buffer -- a
                # notebook forward that reached for one would fail
                # loudly rather than weight regions nobody asked to
                # weight.
                self.register_buffer(
                    "m_vertical",
                    torch.as_tensor(vertical_mask(rois), dtype=torch.float32),
                )
            self.region_proposer = nn.Linear(channels, FEATURE_DIM)
            self.gnn_mlp = nn.Sequential(
                nn.Linear(FEATURE_DIM, GNN_OUT_DIM),
                nn.ReLU(),
                nn.Linear(GNN_OUT_DIM, GNN_OUT_DIM),
            )
            self.w1_pool = nn.Linear(GNN_OUT_DIM, GNN_OUT_DIM)
            self.w2_pool = nn.Linear(GNN_OUT_DIM, GNN_OUT_DIM)
            self.acm_q = nn.Linear(FEATURE_DIM, FEATURE_DIM)
            self.acm_k = nn.Linear(FEATURE_DIM, FEATURE_DIM)
            self.acm_v = nn.Linear(FEATURE_DIM, GNN_OUT_DIM)
            if recipe == "notebook":
                # **THE NOTEBOOK'S ATTENTION HEAD, and the whole of its
                # difference from SABM**: a learned scalar per region,
                # softmaxed OVER REGIONS, replaces eq (7)'s vertical mask
                # -- so the pooling weights are learned rather than fixed
                # by geometry -- and a final sigmoid bounds v into (0, 1)
                # because it is about to MULTIPLY f_t rather than be added
                # to it. Neither piece has a manuscript counterpart.
                self.acm_w_beta = nn.Linear(GNN_OUT_DIM, 1)
                # No gnn_norm, no fused_norm: the notebook normalises
                # nowhere, and ours were measured repairs for the additive
                # path. Carrying them here would make a spanning result
                # unreadable -- their recipe or our repairs, indistinguishable
                # (phase10.NOTEBOOK_RECIPE_CELL_REGISTERED).
                self.classifier = nn.Linear(GNN_OUT_DIM, NUM_CLASSES)
                return
            if recipe == ANNEX_RECIPE:
                # [2026-08-30] The annex carries NEITHER repair, on the
                # notebook cell's own reasoning extended by the ruling:
                # the group's numbers were produced with no
                # normalisation anywhere, and the annex exists to run
                # what produced their numbers
                # (phase10_annex.RECIPE_FIXED_TO_NOTEBOOK
                # ["phase_10_deviations_do_not_ride"]). Whether the
                # additive fusion trains stably under Adam with a
                # trainable backbone is part of what the gate MEASURES,
                # not something to repair in advance.
                self.classifier = nn.Linear(GNN_OUT_DIM, NUM_CLASSES)
                return
            # **[ROUTE 3, 2026-08-17, phase10.GNN_BRANCH_NORM]** The GNN
            # branch normalised immediately before eq. (9)'s sum, with NO
            # learnable affine: the measured failure is that f_t carries
            # |mean| 4.27 against v_a's 0.73 (~5.8x) while itself varying
            # by only 0.0068, so eq. (9)'s plain sum drowns the one branch
            # that preserves image-dependence. affine=False because a
            # learnable scale could restore exactly the imbalance being
            # corrected, and no source asks for one here. The notebook's
            # sigmoid line and its torch.sum are both untouched.
            self.gnn_norm = nn.LayerNorm(GNN_OUT_DIM, elementwise_affine=False)
            # **[FIX (a) 2026-08-17, phase10.FUSED_NORM_AND_STANDARD_INIT]**
            # LayerNorm on the FUSED feature, immediately before the
            # classifier. It leaves the notebook's ``torch.sum`` intact and
            # normalises only what happens AFTER the sum -- which no source
            # specifies -- so it is the lowest-fidelity-cost place to put
            # the scale right. Measured cause: |f_t + v_a| ~ 21 on real
            # features drove the first lr-0.01 step to blow the logits up.
            self.fused_norm = nn.LayerNorm(GNN_OUT_DIM)
            self.classifier = nn.Linear(GNN_OUT_DIM, NUM_CLASSES)

        def timm_backbone(self):
            """The timm model itself, whichever recipe is in force.

            The normalisation statistics are resolved from THIS model's
            own data config, exactly as everywhere else in the project --
            which is also the reason the notebook cell does not adopt the
            notebook's CIFAR-10 statistics: those are a defect in the
            notebook, not a recipe choice
            (``phase10.NOTEBOOK_BUDGET_AND_NORMALISATION``). Returned
            rather than stored, so the module tree does not count the
            backbone's parameters twice.
            """
            return (
                self.backbone.model if self.recipe == "notebook"
                else self.backbone
            )

        def forward(self, x, stages=None):
            """The forward pass. ``stages`` is an OBSERVATION hook: pass a
            dict and every named intermediate is stored in it, unchanged.

            **[2026-08-17, phase10.PROBE_RECONSTRUCTED_THE_PIPELINE]** It
            exists because the scale probe used to REBUILD this pipeline
            by hand to report per-stage statistics, and the rebuild
            silently fell behind the model -- it never applied
            ``gnn_norm``, so the pod's post-fix table was identical to the
            pre-fix one to four decimals. One computation, one
            implementation: the instrument now reads what the model
            actually ran. Passing nothing changes nothing.
            """
            import torch
            import torch.nn.functional as F
            from torchvision.ops import roi_align

            def record(name, value):
                if stages is not None:
                    stages[name] = value
                return value

            record("input", x)

            feature_map = record(
                "backbone_map",
                # The notebook's adapter already returns one map; the
                # features_only ResNet returns a list of them.
                self.backbone(x) if self.recipe == "notebook"
                else self.backbone(x)[-1],
            )
            batch = x.shape[0]
            index = torch.arange(
                batch, device=x.device, dtype=self.roi_boxes.dtype
            ).repeat_interleave(N_REGIONS).unsqueeze(1)
            boxes = torch.cat(
                [index, self.roi_boxes.repeat(batch, 1)], dim=1
            )
            pooled = roi_align(
                feature_map, boxes, output_size=(POOL_SIZE, POOL_SIZE),
                spatial_scale=feature_map.shape[-1] / IMAGE_SIZE,
                sampling_ratio=-1,
            )
            vectors = record(
                "region_vectors",
                pooled.mean(dim=(2, 3)).reshape(batch, N_REGIONS, -1),
            )
            regions = record("region_proposer", self.region_proposer(vectors))

            # APPNP K=1 over the complete graph: one step is the node mean.
            hidden = record("gnn_mlp", self.gnn_mlp(regions))
            propagated = record("appnp", (
                ALPHA * hidden
                + (1.0 - ALPHA) * hidden.mean(dim=1, keepdim=True)
            ))
            f_hat = record("sigmoid_f_hat", torch.sigmoid(propagated))

            gate = torch.sigmoid(self.w1_pool(f_hat))
            f_t = record("gated_pooling_f_t", (gate * self.w2_pool(f_hat)).sum(dim=1))

            # The attention, shared to the point where the two artifacts
            # part company: Q, K, tanh(QK^T/sqrt(d)), Softmax, V.
            queries = self.acm_q(regions)
            keys = self.acm_k(regions)
            attention = F.softmax(
                torch.tanh(
                    queries @ keys.transpose(1, 2) / (FEATURE_DIM ** 0.5)
                ),
                dim=2,
            )
            f_att = attention @ self.acm_v(regions)

            if self.recipe == "notebook":
                # Their pooling: a learned scalar per region, softmaxed
                # over the region axis, then sigmoid. The notebook loops
                # over the batch; this is the same arithmetic batched,
                # with its dim=0 (over regions) becoming dim=1 here.
                w_r = F.softmax(self.acm_w_beta(f_att), dim=1)
                v = record(
                    "acm_v", torch.sigmoid((f_att * w_r).sum(dim=1))
                )
                # ``f_t + f_t * v`` is the notebook's literal expression,
                # kept literal rather than folded to f_t * (1 + v): the
                # two agree in real arithmetic and can differ in the last
                # bit, and the artifact is what this cell replicates.
                fused = record("fused", f_t + f_t * v)
                return record("logits", self.classifier(fused))

            # SABM eq (8): the fixed vertical weighting, the manuscript's.
            v_a = record(
                "sabm_v_a", (f_att * self.m_vertical.view(1, -1, 1)).sum(dim=1)
            )

            if self.recipe == ANNEX_RECIPE:
                # [2026-08-30] Eq (9)'s sum, PLAIN -- no gnn_norm, no
                # fused_norm; the annex runs the architecture as the
                # manuscript states it, unrepaired.
                fused = record("fused", f_t + v_a)
                return record("logits", self.classifier(fused))

            fused = record("fused", record("gnn_norm_f_t", self.gnn_norm(f_t)) + v_a)
            normed = record("after_layernorm", self.fused_norm(fused))
            return record("logits", self.classifier(normed))

    return CleftGNN()


@dataclass
class CleftGNNBackbone:
    """The harness protocol, with the registered recipe."""

    learning_rate: float = 0.01
    momentum: float = 0.0
    batch_size: int = 16
    seed: int = 1337
    pretrained: bool = True
    device: str = "cuda"
    #: Which of the group's artifacts this cell is faithful to
    #: (``RECIPES``), and the optimiser that artifact runs. The defaults
    #: are the manuscript's, so the existing arm is untouched by the
    #: notebook cell's arrival.
    recipe: str = "manuscript"
    optimizer: str = "sgd"
    #: **[2026-08-30] "backbone_frozen" everywhere but the annex.** The
    #: frozen policy is the registered one for both documented recipes
    #: (``phase10.BACKBONE_IS_FROZEN``); "full" is admitted ONLY under
    #: ``ANNEX_RECIPE`` (``phase10_annex.RULING_B_FULL_TRAINABLE`` -- a
    #: scoped departure, not a general unlock), and the annex recipe
    #: conversely REQUIRES "full": a frozen annex arm would reintroduce
    #: 'we froze it' as the alternative explanation the ruling exists to
    #: close. Both directions refused at reset, loudly.
    trainable: str = "backbone_frozen"
    parameter_report: dict = field(default_factory=dict)
    #: **[2026-08-17]** Record the stage table once per epoch, so
    #: criterion (i) is answered on the ARM'S OWN fold rather than
    #: assumed from the probe's single initialisation batch
    #: (``phase10.CRITERION_I_UNREPORTED``). It costs NO extra forward:
    #: the capture rides the inner-val prediction the harness already
    #: makes. ``train_epoch`` runs exactly once per epoch immediately
    #: before that prediction, so counting it here indexes the ratios by
    #: the true epoch number -- and the fold's SELECTED epoch is then
    #: retrievable by lookup, which matters because the frozen harness
    #: does not restore best weights.
    record_stages: bool = False
    stage_ratios_by_epoch: dict = field(default_factory=dict)
    _epoch: int = 0

    _model: Any = field(default=None, repr=False)
    _optimizer: Any = field(default=None, repr=False)
    _mean: Any = field(default=None, repr=False)
    _std: Any = field(default=None, repr=False)

    def reset(self, train_labels: np.ndarray) -> None:
        import timm
        import torch

        # [2026-08-30] The scoped-departure contract, refused BEFORE any
        # model is built: "full" exists only under the annex recipe, and
        # the annex recipe exists only full-trainable.
        if self.trainable not in ("backbone_frozen", "full"):
            raise CleftGNNError(
                f"trainable {self.trainable!r} is neither 'backbone_frozen' "
                "nor the annex's 'full'"
            )
        if self.trainable == "full" and self.recipe != ANNEX_RECIPE:
            raise CleftGNNError(
                f"trainable 'full' with recipe {self.recipe!r}: full "
                "fine-tuning is operationalised in the Phase 10 annex ONLY "
                "(phase10_annex.RULING_B_FULL_TRAINABLE -- a scoped "
                "departure, not a general unlock); every other recipe "
                "keeps the registered frozen backbone "
                "(phase10.BACKBONE_IS_FROZEN)"
            )
        if self.recipe == ANNEX_RECIPE and self.trainable != "full":
            raise CleftGNNError(
                f"recipe {ANNEX_RECIPE!r} with trainable "
                f"{self.trainable!r}: the annex is full-trainable BY "
                "RULING -- a frozen annex arm would leave 'we froze it' "
                "as the permanent alternative explanation the ruling "
                "exists to close (phase10_annex.RULING_B_FULL_TRAINABLE)"
            )

        torch.manual_seed(self.seed)
        self._model = build(pretrained=self.pretrained, recipe=self.recipe)

        # **THIS PROJECT'S NORMALISATION ON BOTH DOCUMENTED CELLS.** The
        # statistics come from the backbone's own timm data config, as
        # they do everywhere else here. The notebook normalises with
        # CIFAR-10 statistics under an ImageNet-pretrained ViT, which is
        # a DEFECT in the notebook rather than a recipe choice, and
        # replicating it would mis-scale the frozen features this cell
        # depends on (phase10.NOTEBOOK_BUDGET_AND_NORMALISATION).
        # [2026-08-30] THE ANNEX IS THE RULED EXCEPTION: it reproduces
        # the notebook's CIFAR-10 constants, recorded as a DELIBERATELY
        # REPLICATED DEFECT -- replicated knowingly, never "a recipe"
        # (phase10_annex.NORMALISATION_RULED: every factor other than
        # the split must be what actually produced their numbers,
        # defects included).
        device = torch.device(
            self.device if torch.cuda.is_available() else "cpu"
        )
        if self.recipe == ANNEX_RECIPE:
            mean, std = CIFAR10_NORMALISATION
        else:
            config = timm.data.resolve_model_data_config(
                self._model.timm_backbone()
            )
            mean, std = config["mean"], config["std"]
        self._mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self._std = torch.tensor(std, device=device).view(1, -1, 1, 1)

        # Calibrated CE init: zero weights, bias = log train-fold class
        # frequencies -- the epoch-0 EXPECTED VALUE equals the train-fold
        # label mean, which is gate 3's policy translated to CE.
        grades = _as_grades(train_labels)
        counts = np.array(
            [np.sum(grades == g) for g in range(1, NUM_CLASSES + 1)],
            dtype=np.float64,
        )
        # **LAPLACE, not epsilon** (phase10.THIN_CLASS_SMOOTHING): the old
        # 1e-8 put an absent class at p ~ 1e-10, i.e. bias -23.65, which
        # made that grade unreachable and biased the expected-value
        # reading. (count+1)/(N+K) gives a reachable ~-5.3 instead.
        frequencies = (counts + 1.0) / (counts.sum() + NUM_CLASSES)
        # **[FIX (b) 2026-08-17]** The classifier's WEIGHT keeps the
        # framework's own small-random init (nn.Linear's kaiming-uniform,
        # bound 1/sqrt(1024) ~ 0.031) -- the zero-weight init was MINE, a
        # translation of gate 3 into cross-entropy, and it is mine to give
        # up. It also produced an artifact worth losing: with a zero
        # weight every image received identical logits, so the epoch-0
        # prediction vector was CONSTANT BY CONSTRUCTION and any PCC
        # against it was nan for reasons unrelated to training
        # (phase10.SD_ZERO_HAS_TWO_CAUSES). The Laplace BIAS stays: it is
        # what keeps epoch 0 near the label mean and every grade
        # reachable.
        # [2026-08-30] The Laplace-biased init was one of THIS project's
        # registered deviations, and the annex carries none of them
        # (phase10_annex.RECIPE_FIXED_TO_NOTEBOOK): the classifier keeps
        # the framework's own init, as their code's did. ``_as_grades``
        # above still ran -- the integer-grade contract holds on every
        # recipe.
        if self.recipe != ANNEX_RECIPE:
            with torch.no_grad():
                self._model.classifier.bias.copy_(
                    torch.as_tensor(np.log(frequencies), dtype=torch.float32)
                )

        backbone_total = sum(p.numel() for p in self._model.backbone.parameters())
        if self.trainable == "full":
            # [2026-08-30, annex only -- the guards at the top of reset
            # already bound this to ANNEX_RECIPE] EVERYTHING trains,
            # asserted rather than assumed, in both directions: a
            # silently-frozen tensor here would rebuild the 'we froze
            # it' explanation the ruling closes.
            frozen = sum(
                p.numel() for p in self._model.parameters()
                if not p.requires_grad
            )
            if frozen:
                raise CleftGNNError(
                    f"{frozen} parameters are frozen under trainable "
                    "'full'; the annex trains everything by ruling"
                )
            backbone_report = 0
        else:
            # **THE BACKBONE IS FROZEN** (phase10.BACKBONE_IS_FROZEN).
            # Asserted rather than assumed: a backbone that silently
            # trained would be the arm nobody registered, and it is what
            # diverged.
            for parameter in self._model.backbone.parameters():
                parameter.requires_grad_(False)
            backbone_trainable = sum(
                p.numel() for p in self._model.backbone.parameters()
                if p.requires_grad
            )
            if backbone_trainable:
                raise CleftGNNError(
                    f"{backbone_trainable} backbone parameters are still "
                    "trainable; the registered policy freezes the backbone"
                )
            backbone_report = backbone_total
        total = sum(p.numel() for p in self._model.parameters())
        trainable = sum(
            p.numel() for p in self._model.parameters() if p.requires_grad
        )
        if trainable != total - backbone_report:
            raise CleftGNNError(
                f"{trainable} trainable of {total} does not equal the "
                f"expected count {total - backbone_report}; something "
                "was frozen or unfrozen that the policy does not name"
            )
        if not trainable:
            raise CleftGNNError("nothing is trainable; the arm would learn nothing")
        self.parameter_report = {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_backbone_parameters": backbone_report,
            "recipe": self.recipe,
            "optimizer": self.optimizer,
            "trainable": self.trainable,
        }

        self._epoch = 0
        self.stage_ratios_by_epoch = {}

        self._model.to(device)
        parameters = [p for p in self._model.parameters() if p.requires_grad]
        if self.optimizer == "adam":
            if self.momentum:
                raise CleftGNNError(
                    f"momentum {self.momentum} was declared alongside Adam; "
                    "Adam has no momentum knob and the notebook passes only "
                    "lr, so the config is claiming something the run cannot "
                    "honour"
                )
            # The notebook's line, whole: optim.Adam(trainable_params,
            # lr=LR) -- betas and eps left at torch's defaults because the
            # notebook leaves them there, and weight decay absent because
            # the notebook passes none.
            self._optimizer = torch.optim.Adam(
                parameters, lr=self.learning_rate
            )
        elif self.optimizer == "sgd":
            self._optimizer = torch.optim.SGD(
                parameters, lr=self.learning_rate, momentum=self.momentum,
            )
        else:
            raise CleftGNNError(
                f"optimizer {self.optimizer!r} is neither the manuscript's "
                "'sgd' nor the notebook's 'adam'"
            )

    def _batches(self, features, labels, shuffle: bool):
        import torch

        device = next(self._model.parameters()).device
        order = np.arange(len(features))
        if shuffle:
            order = np.random.default_rng(self.seed).permutation(order)
        for start in range(0, len(order), self.batch_size):
            rows = order[start:start + self.batch_size]
            batch = torch.as_tensor(
                np.asarray(features[rows], dtype=np.float32) / 255.0
            ).permute(0, 3, 1, 2).to(device)
            batch = (batch - self._mean) / self._std
            target = (
                None if labels is None
                else torch.as_tensor(
                    _as_grades(labels[rows]) - 1,
                    dtype=torch.long, device=device,
                )
            )
            yield batch, target

    def train_epoch(self, features: np.ndarray, labels: np.ndarray) -> float:
        import torch

        self._epoch += 1
        self._model.train()
        # Frozen means eval too: BatchNorm running statistics must not
        # drift, and batch statistics would make the "frozen"
        # representation depend on the batch it arrived in. On the
        # notebook's ViT this is a no-op in value -- it has no BatchNorm
        # and timm builds it with every drop rate at 0, so train and eval
        # compute the same thing -- and it is kept anyway so "frozen" has
        # one meaning across both recipes.
        # [2026-08-30] Under the annex's 'full' the guard is KEYED OFF,
        # not deleted: a full fine-tune trains BatchNorm normally --
        # running statistics update -- which is what their training
        # would have done (phase10_annex.RULING_B_FULL_TRAINABLE).
        if self.trainable != "full":
            self._model.backbone.eval()
        total, seen = 0.0, 0
        for batch, target in self._batches(features, labels, shuffle=True):
            self._optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                self._model(batch), target
            )
            loss.backward()
            self._optimizer.step()
            total += float(loss.item()) * len(target)
            seen += len(target)
        return total / max(seen, 1)

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        import torch

        self._model.eval()
        out = []
        with torch.no_grad():
            for batch, _ in self._batches(features, None, shuffle=False):
                # The stage table rides the first batch of the first
                # prediction after each epoch's training -- no extra forward.
                if (
                    self.record_stages
                    and self._epoch not in self.stage_ratios_by_epoch
                ):
                    stages: dict = {}
                    logits = self._model(batch, stages=stages)
                    self.stage_ratios_by_epoch[self._epoch] = stage_ratios(stages)
                    out.append(torch.softmax(logits, dim=1).cpu().numpy())
                    continue
                out.append(
                    torch.softmax(self._model(batch), dim=1).cpu().numpy()
                )
        return (
            np.concatenate(out) if out else np.empty((0, NUM_CLASSES))
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """The registered PRIMARY reading: the softmax-expected grade."""
        return expected_grade(self.predict_probabilities(features))
