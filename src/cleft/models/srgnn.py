"""SR-GNN: region nodes, APPNP propagation, attentional refinement.

**[LITERATURE]** Bera, Wharton, Liu, Bessis & Behera, "SR-GNN: Spatial Relation-
Aware Graph Neural Network for Fine-Grained Image Categorization", *IEEE TIP*
31:6017-6031 (2022).

----------------------------------------------------------------------------
PROVENANCE: PORTED FROM A WEIGHT-TRANSPLANT-VERIFIED IMPLEMENTATION
----------------------------------------------------------------------------
The architecture below is **not reconstructed from the paper**. It is ported from
the Stage 1 implementation at ``cleft-aesthetic-assessment/models/srgnn.py``,
which was verified against the authors' released TensorFlow weights:

* **257 of 257 weight tensors map** onto the port's parameters;
* outputs match the authors' model to **< 1e-6 at every internal stage** --
  backbone, ROI pooling, both branches, classifier;
* corroborated across three checkpoints.

**That is a stronger guarantee than a paper.** A described architecture can be
implemented several ways and still read as faithful; one that loads the original
weights and reproduces the original activations to 1e-6 cannot be wrong about its
shapes, its layer order, or its widths. The parameter count below is therefore a
**cross-check on the port, not the verification of it** -- which matters, because
it does not match the paper exactly (see ``check_against_paper``).

An earlier revision of this module reconstructed the architecture from the
citation and the abstract and came out **8.07M short**. The missing mass was
almost entirely ``SeqSelfAttention`` over the *flattened* region features:
2048x7x7 = 100,352 dimensions projected to 32 units, twice, is **6.42M
parameters in one layer** that a reconstruction had no way to guess. Recorded
because the reconstruction looked reasonable and was wrong by a quarter.

----------------------------------------------------------------------------
WHAT WAS AND WAS NOT PORTED
----------------------------------------------------------------------------
**Ported:** the architecture -- ``SRGNN``, ``SeqSelfAttention``,
``SeqWeightedAttention``, the ROI pooling and the fully-connected GCN-normalised
edge weights.

**Not ported:** every piece of training scaffolding -- datasets, loops,
schedulers, logging, checkpointing. This project has its own, frozen, and the
fresh-start rule exists because *that* code accumulated defects with unclear
provenance. A model definition validated by weight transplant is the opposite
case and is exactly what the rule's "lessons, not code" carve-out is for.

**Reimplemented, and now VERIFIED:** the Stage 1 port calls
``torch_geometric.nn.APPNP``, which is not in this project's pinned image. At
``K=1``, ``alpha=0.3``, ``normalize=False`` and ``add_self_loops=False`` the
propagation reduces to one line -- ``h = (1-alpha) * A_hat @ h + alpha * h``, on
weights this module computes itself -- so it is written directly rather than
adding a dependency.

**[MEASURED 2026-07-31] Parity confirmed to machine precision.**
``scripts/verify_appnp_parity.py`` runs both implementations on identical inputs:
**worst agreement 0.92 ULP** over region counts 5/27/37, widths 16/1024, and both
float32 and float64 -- sub-single-ULP everywhere. Checked at ``K`` = 1, 2, 5 and
10 as well, so the equivalence is not an accident of the port's ``K=1``.

This was the one part of the file whose provenance was not the transplant, and
it is no longer [REASONED]. The check was run against ``torch_geometric`` 2.8.0
outside the project environment, so nothing was added to the test extra.

> **[CORRECTED 2026-07-31]** An earlier revision of this paragraph said
> ``torch_geometric`` "remains absent from the pinned image". That is wrong
> about the image: ``docker/requirements.txt`` has pinned
> ``torch-geometric==2.6.1`` since Phase 0, and the numpy<2 guard names its
> ABI explicitly. What is true is that **this module does not import it** --
> the direct propagation stands on the 0.92-ULP parity, not on the
> dependency's absence.

----------------------------------------------------------------------------
THIS MODULE IMPORTS WITHOUT TORCH
----------------------------------------------------------------------------
Torch lives inside ``build``. Module level holds the specification as plain data
plus ``parameter_count``, which derives the total arithmetically. Two independent
derivations of one number: this arithmetic on the laptop, and the built model's
own count on the cluster, asserted against it at construction.
"""

from __future__ import annotations

# --- the specification, as ported ------------------------------------------
# Every value here is read off the transplant-verified implementation, not
# inferred. The defaults are its defaults.

#: **The timm model name, spelled EXPLICITLY.**
#:
#: **[MEASURED 2026-07-31]** ``xception`` is deprecated in the pinned image and
#: timm silently remaps it to ``legacy_xception``. It resolves correctly today,
#: and that is exactly the problem: a version bump could remap it to a different
#: Xception variant, the parameter count would move, and the port would quietly
#: stop corresponding to the transplant-verified architecture. A model that is
#: *plausible* rather than *right* is the failure this project keeps finding.
#:
#: So the alias is not relied on, and ``build`` asserts the parameter count at
#: construction: a remap fails loudly instead of producing a plausible model.
BACKBONE_TIMM_NAME = "legacy_xception"

#: Xception's final conv stage. [LITERATURE] Chollet (CVPR 2017).
BACKBONE_CHANNELS = 2048

#: ROI pooling output side, and the resolution the feature map is upsampled to
#: before cropping. Paper Sec III-B; HOG cell size 14x14 on a 42x42 map.
POOL_SIZE = 7
ROI_RESOLUTION = 42

#: 26 grid-combinatorial boxes plus the whole image appended as region 27.
#: Matches the paper's R=27 -- and, separately, the supervision material's "27 patches" description
#: that PLAN §4.5 settled the patch scheme against.
N_ROIS = 26
N_REGIONS = N_ROIS + 1

#: Flattened per-region descriptor: C x p x p. **This is the number that a
#: reconstruction cannot guess**, and it is where 6.42M of the model lives.
REGION_FEATURE_DIM = BACKBONE_CHANNELS * POOL_SIZE * POOL_SIZE  # 100,352


#: **[AUTHORISED 2026-09-06] PERMISSION TO PUBLISH THIS FILE, and the
#: reason permission was needed rather than inherited.**
#:
#: Recorded here rather than in a phase module because a reader who
#: opens this file is the reader who needs it.
PUBLICATION_AUTHORISED = {
    "authorised": "2026-09-06",

    "what_it_covers": (
        "**this file, ``src/cleft/models/srgnn.py``, and its publication "
        "in this repository's public snapshot.** It covers the ported "
        "architecture and nothing else: no upstream weights are "
        "redistributed here, and no dataset is"
    ),
    "who_it_is_held_from": (
        "**two sources, both with standing to give it.** The supervisor "
        "of this work, who is an author on the SR-GNN paper, and the "
        "owner of the upstream repository, who is its corresponding "
        "author. Named by ROLE here, as everything in this record is"
    ),

    "why_permission_was_NEEDED_and_not_inherited": (
        "**the upstream repository carries no LICENSE file.** "
        "``https://github.com/ArdhenduBehera/SR-GNN`` releases the "
        "implementation and a TrainedModels directory and states no "
        "licence, in the repository or in its README. Neither the "
        "paper's arXiv page nor the corresponding author's own page "
        "links to a licensed release. **No public grant exists, so "
        "nothing was inherited**, and a port of it is a derivative work "
        "of code that was never licensed for redistribution"
    ),
    "so_the_basis_is_the_permission_and_the_record_says_so": (
        "**this file is published on the authorisation above, not on an "
        "upstream licence.** The distinction matters because the two "
        "look identical from outside and behave differently: a licence "
        "would travel to whoever receives this repository, and a "
        "permission does not"
    ),
    "and_it_grants_nothing_onward": (
        "**a recipient of this repository receives no right to the "
        "upstream work.** The Apache-2.0 grant over the rest of the "
        "tree does not extend to this file, ``NOTICE`` says so at the "
        "top level, and anyone wanting to reuse the SR-GNN "
        "implementation should approach the upstream authors rather "
        "than relying on this copy"
    ),

    "what_this_does_NOT_change": (
        "**no figure and no finding.** The port's verification stands "
        "exactly as recorded above: 257 of 257 weight tensors mapped, "
        "outputs matching to better than 1e-6 at every internal stage, "
        "corroborated across three checkpoints. The authorisation is "
        "about publication, not about correctness"
    ),
    "and_the_sibling_files_are_NOT_covered_because_they_do_not_need_it": (
        "**``agnet.py`` is not a port**, it is written from the "
        "published paper, so it is this project's own expression and "
        "carries no upstream constraint. ``cleftgnn.py`` is built to a "
        "collaborating group's artifacts under a separate authorisation "
        "held from the supervisor. **Three files, three different "
        "provenances, and treating them alike would be the error this "
        "module's header already warns about**"
    ),
}

#: SeqSelfAttention width. Keras default in the released code.
SELF_ATTENTION_UNITS = 32

#: Branch B widths. Two APPNPConv layers then a Dense back up to dense_dim.
GNN_CHANNELS = 1024
DENSE_DIM = 2048  # must equal BACKBONE_CHANNELS for the Eq. 5 skip connection

#: APPNP propagation. K=1 in the released code; alpha is its teleport.
#: **Parameter-free**, which is why neither appears in the count.
APPNP_K = 1
APPNP_ALPHA = 0.3

#: **[MEASURED 2026-07-31]** Worst disagreement between the direct propagation
#: and ``torch_geometric.nn.APPNP`` over region counts 5/27/37, widths 16/1024,
#: float32 and float64, and K in {1, 2, 5, 10}. Sub-single-ULP: the two are the
#: same computation. ``scripts/verify_appnp_parity.py``.
APPNP_PARITY_WORST_ULP = 0.92

#: **[MEASURED 2026-07-31] The whole model, constructed.** Both counts hit their
#: specifications exactly on the first construction -- 32,896,562 at 200 classes,
#: which is what ``parameter_count`` derives. Recorded because until this ran the
#: specification was arithmetic that nothing had ever tested against an
#: ``nn.Module``.
#:
#: Built with timm 1.0.27 / torch 2.13.0 / torchvision 0.27.1 outside the pinned
#: image, so it is a check on the SPEC rather than on the image -- and the
#: backbone term matched the image-measured constant on a different timm, which
#: is the useful part: the number is not an artefact of one environment.
BUILT_PARAMETERS_CUB = 32_896_562

#: What the first construction actually ran in, as data rather than prose.
#: Assertions about provenance should read a field, not grep a docstring: a
#: reflowed line silently breaks a substring match, which makes the check fail
#: for a reason that has nothing to do with the claim.
BUILD_VERIFICATION = {
    "measured": "2026-07-31",
    "built_parameters_cub": BUILT_PARAMETERS_CUB,
    "matched_spec_first_construction": True,
    "environment": "timm 1.0.27, torch 2.13.0, torchvision 0.27.1",
    "in_pinned_image": False,
    "what_this_proves": (
        "the SPECIFICATION, not the image: built on a different timm than the "
        "one the backbone constant was measured in, and the backbone term still "
        "matched -- so the count is a property of the architecture rather than "
        "of one environment"
    ),
    "appnp_parity_worst_ulp": APPNP_PARITY_WORST_ULP,
    "appnp_is_verified": True,
    "appnp_checked_against": "torch_geometric 2.8.0, outside the project venv",
}

#: [LITERATURE] What the paper reports. The one number here that is not ported,
#: which is why it can adjudicate the port rather than agreeing with it.
PAPER_PARAMETERS = 30_900_000

#: **[MEASURED 2026-07-31, in the pinned image]** ``legacy_xception`` feature
#: body: **20,806,952** parameters. Confirmed in
#: ``redring/cleft-aesthetics@sha256:2135e27b5d82b28cb5e2059c606aadf5736df80e50cb4e52fc669cc8ba33d2ab``
#: -- an exact match to the arithmetic derivation (22.86M total less a 2048->1000
#: classifier), so the two independent routes agree.
#:
#: **The digest IS the provenance.** This number is only true of that image: a
#: different timm would resolve the backbone differently, which is why
#: ``BACKBONE_TIMM_NAME`` is explicit and why ``build`` re-asserts at
#: construction rather than trusting this constant.
XCEPTION_FEATURE_PARAMETERS = 20_806_952

#: The image the parameter constants were measured in. Recorded beside them
#: because a measured constant without the environment that produced it is a
#: number nobody can re-derive.
#: On ONE line deliberately. Split across two, the first line ends with the
#: digest prefix and no hex, and ``test_no_truncated_image_digest_anywhere``
#: flags it -- correctly. A digest that reads as truncated in source is one a
#: reader cannot verify by looking, whatever the runtime value turns out to be.
#: (Naming the offending token in this comment trips the same check, which is
#: the detector being consistent rather than over-eager.)
MEASURED_IN_IMAGE = "redring/cleft-aesthetics@sha256:2135e27b5d82b28cb5e2059c606aadf5736df80e50cb4e52fc669cc8ba33d2ab"  # noqa: E501


class SRGNNSpecError(RuntimeError):
    """The built model disagrees with the specification."""


def parameter_count(
    *,
    backbone_parameters: int = XCEPTION_FEATURE_PARAMETERS,
    num_outputs: int = 1,
) -> dict:
    """Total parameters, derived from the ported specification.

    Per component, because a total that is 2M out says nothing about where.
    """
    # SeqSelfAttention: Wt and Wx project the FLATTENED region descriptor to
    # `units`, both without bias; plus the additive bias and the scorer.
    self_attention = (
        2 * REGION_FEATURE_DIM * SELF_ATTENTION_UNITS  # Wt, Wx (bias=False)
        + SELF_ATTENTION_UNITS                          # bh
        + SELF_ATTENTION_UNITS + 1                      # Wa (with bias)
    )

    # Branch B: APPNPConv x2 (the Dense parts; propagation is parameter-free),
    # then GlobalSumPool + Dense back to dense_dim.
    gnn_mlp1 = BACKBONE_CHANNELS * GNN_CHANNELS + GNN_CHANNELS
    gnn_mlp2 = GNN_CHANNELS * GNN_CHANNELS + GNN_CHANNELS
    gnn_out = GNN_CHANNELS * DENSE_DIM + DENSE_DIM

    weighted_attention = BACKBONE_CHANNELS + 1          # Linear(C, 1)
    batch_norms = 2 * (2 * BACKBONE_CHANNELS)           # bn1, bn2: weight+bias
    classifier = DENSE_DIM * num_outputs + num_outputs

    head = (
        self_attention + gnn_mlp1 + gnn_mlp2 + gnn_out
        + weighted_attention + batch_norms + classifier
    )
    return {
        "backbone": backbone_parameters,
        "self_attention": self_attention,
        "gnn_mlp1": gnn_mlp1,
        "gnn_mlp2": gnn_mlp2,
        "gnn_out": gnn_out,
        "weighted_attention": weighted_attention,
        "batch_norms": batch_norms,
        "classifier": classifier,
        "head_total": head,
        "total": backbone_parameters + head,
    }


def check_against_paper(num_outputs: int = 200, tolerance: float = 0.10) -> dict:
    """Does the ported spec land near the paper's ~30.9M?

    Defaults to ``num_outputs=200`` because the paper's figure is for CUB, and
    comparing a 1-output regression head against a 200-way classifier would be
    comparing two different models -- a quantity confusion in the check itself.

    **This cross-checks the port; it does not verify it.** The verification is
    the weight transplant: 257/257 tensors, <1e-6 at every stage. If this
    disagreed sharply the port would be in question, but a few percent against a
    figure quoted as "~30.9M" is within what counting conventions cover.
    """
    counts = parameter_count(num_outputs=num_outputs)
    total = counts["total"]
    ratio = total / PAPER_PARAMETERS
    return {
        "spec_total": total,
        "paper_total": PAPER_PARAMETERS,
        "ratio": round(ratio, 4),
        "difference": total - PAPER_PARAMETERS,
        "num_outputs": num_outputs,
        "agrees_within_tolerance": abs(ratio - 1.0) <= tolerance,
        "tolerance": tolerance,
        "components": counts,
        "verified_by": (
            "weight transplant against the authors' released weights: 257/257 "
            "tensors mapped, outputs match to <1e-6 at every internal stage, "
            "corroborated across three checkpoints. THAT is the verification; "
            "this count is a cross-check on the port."
        ),
        "note": (
            "The largest term is self_attention at 6.42M -- SeqSelfAttention "
            "over the FLATTENED 100,352-dim region descriptor. A reconstruction "
            "of this architecture from the paper's description missed it "
            "entirely and came out 8.07M short, which is why the port exists."
        ),
    }


#: **[MEASURED 2026-07-31, in the pinned image -- figures pasted by the
#: maintainer] SR-GNN trains BITWISE-deterministic flag-off on the image's
#: CUDA**: repeat-gradient 0.0, post-restore step 0.0. First measured on an
#: RTX 3060 / torch 2.13 with the same result, so the figure held across two
#: devices and two torch versions.
#:
#: Stated precisely because the OP CLASS it pools with is not deterministic
#: in general: an isolated bilinear ``interpolate`` backward under dense
#: collisions measured 5.25e-06 in the image (4.77e-06 locally). The model's
#: instantiation of the op lands deterministic; the class does not guarantee
#: it, which is why ``scripts/verify_train_determinism.py`` exists to
#: re-measure after any device or torch change rather than carrying the
#: figure across.
#:
#: The twelve pretraining configs run ``deterministic: false`` [DECIDED]:
#: this backbone is bitwise without the flag, so the flag buys nothing --
#: and enabling it requires the CUBLAS workspace export owned by the frozen
#: ``determinism.configure``, never a directly-set flag (the direct set
#: raised CuBLAS errors on every backbone in the image).
TRAINING_DETERMINISM = {
    "measured": "2026-07-31",
    "environment": (
        "the pinned image (figures pasted by the maintainer); first measured "
        "on RTX 3060 / torch 2.13.0+cu126 with the same result"
    ),
    "device": "cuda",
    "flag_off_repeat_grad_max_diff": 0.0,
    "flag_off_post_restore_step_max_diff": 0.0,
    "bitwise_flag_off": True,
    "op_class_caveat": (
        "bilinear interpolate backward is atomicAdd-based on CUDA and measured "
        "5.25e-06 nondeterministic in an isolated dense-collision probe (image; "
        "4.77e-06 locally); this model's use of it measured bitwise. A device "
        "or torch change can move that -- re-measure with "
        "scripts/verify_train_determinism.py rather than carrying this figure "
        "across."
    ),
    "decision": "the twelve pretraining configs run deterministic: false",
    "authoritative_for_runs": "the image run of scripts/verify_train_determinism.py",
}

#: The grid-combinatorial ROI generation's own parameters. [LITERATURE] paper
#: Sec III-B defaults: a 3x3 grid, minimum extent two cells. With these and
#: ``ROI_RESOLUTION`` the enumeration yields exactly ``N_ROIS`` boxes; the
#: whole image is appended as region 27 inside ``forward``.
GRID_SIZE = 3
GRID_MIN_SIZE = 2


def grid_rois(
    resolution: int = ROI_RESOLUTION,
    grid_size: int = GRID_SIZE,
    min_size: int = GRID_MIN_SIZE,
):
    """The 26 grid-combinatorial region boxes, (x, y, w, h) in map pixels.

    Ported verbatim from the Stage 1 implementation (``srgnn_regions.get_rois``)
    -- every valid box over the grid excluding the full image. numpy only, so
    the enumeration is laptop-testable while the model itself is not.
    """
    import numpy as np

    coords = []
    step = resolution / grid_size
    for c1 in range(grid_size + 1):
        for c2 in range(grid_size + 1):
            for r1 in range(grid_size + 1):
                for r2 in range(grid_size + 1):
                    x0, x1 = int(c1 * step), int(c2 * step)
                    y0, y1 = int(r1 * step), int(r2 * step)
                    if x1 > x0 and y1 > y0 and (
                        (x1 - x0) >= step * min_size or (y1 - y0) >= step * min_size
                    ):
                        if not (x0 == y0 == 0 and x1 == y1 == resolution):
                            coords.append([x0, y0, x1 - x0, y1 - y0])
    return np.array(coords, dtype=np.float32)


def build(*, pretrained: bool = True, num_outputs: int = 1, **overrides):
    """The torch model: the transplant-verified architecture, constructed.

    **Where each part comes from -- moved, not rewritten:**

    * the construction (attribute names, shapes, widths) is verbatim from the
      first-construction script that measured ``BUILT_PARAMETERS_CUB`` hitting
      the spec exactly, so the construction evidence carries over;
    * the forward pass is the Stage 1 implementation
      (``cleft-aesthetic-assessment/models/srgnn.py``), which is the
      weight-transplant-verified lineage, with ONE substitution: the
      ``torch_geometric.nn.APPNP`` call is replaced by the direct propagation
      this module specifies, verified against it to 0.92 ULP
      (``scripts/verify_appnp_parity.py``) -- and batched over the batch
      dimension, which the linear operator permits;
    * the attention layers carry the released-code semantics only
      (``sigmoid`` -> softmax self-attention, softmax weighted attention).
      Stage 1's ablation knobs are deliberately not carried: a knob no config
      can reach is how unconfigured arms happen.

    ``forward(x)`` uses the 26 grid boxes plus the whole image -- the paper's
    own regions, which is what SCUT pretraining wants. ``forward(x, boxes)``
    accepts explicit boxes ((R, 4) shared, or (B, R, 4) per image) for the
    cleft path later.

    Asserts its own parameter count against ``parameter_count()`` per
    component, so a port that drifts from the recorded specification fails at
    construction rather than after a pretraining run, and stamps its
    normalization from the timm backbone's own ``pretrained_cfg`` -- read,
    never assumed.
    """
    if overrides:
        raise SRGNNSpecError(
            f"unknown build override(s): {sorted(overrides)}. No overrides are "
            "defined; silently absorbing one would run an arm nobody configured."
        )
    try:
        import numpy as np
        import timm
        import torch
        import torch.nn.functional as F
        from torch import nn
    except ImportError as exc:
        raise SRGNNSpecError(
            f"building SR-GNN needs torch and timm, which are not importable "
            f"here ({exc}). They are in the pinned image; the laptop suite "
            "exercises the loop through the stub backbone."
        ) from None

    from .factory import stamp_normalization

    class SeqSelfAttention(nn.Module):
        """Ported: additive self-attention over the flattened region
        descriptors, released-code semantics (sigmoid THEN softmax)."""

        def __init__(self, feature_dim, units=SELF_ATTENTION_UNITS):
            super().__init__()
            self.Wt = nn.Linear(feature_dim, units, bias=False)
            self.Wx = nn.Linear(feature_dim, units, bias=False)
            self.bh = nn.Parameter(torch.zeros(units))
            self.Wa = nn.Linear(units, 1, bias=True)

        def forward(self, x):
            q = self.Wt(x).unsqueeze(2)
            k = self.Wx(x).unsqueeze(1)
            h = torch.tanh(q + k + self.bh)
            logits = self.Wa(h).squeeze(-1)
            a = torch.softmax(torch.sigmoid(logits), dim=-1)
            return torch.bmm(a, x)

    class SeqWeightedAttention(nn.Module):
        def __init__(self, feature_dim):
            super().__init__()
            self.W = nn.Linear(feature_dim, 1, bias=True)

        def forward(self, x):
            a = torch.softmax(self.W(x).squeeze(-1), dim=1)
            return (x * a.unsqueeze(-1)).sum(dim=1), a

    def roi_pool(full_img, boxes_xywh, pool_size, resolution):
        """Bilinear-resize each ROI to (pool_size, pool_size) -> (B, R, C, p, p).

        Stage 1's ``_roi_pool``, verbatim: one box set shared across the batch.
        """
        outs = []
        for (x, y, w, h) in boxes_xywh.tolist():
            x, y = int(x), int(y)
            w, h = max(int(w), 1), max(int(h), 1)
            x2, y2 = min(x + w, resolution), min(y + h, resolution)
            crop = full_img[:, :, y:y2, x:x2]
            outs.append(
                F.interpolate(
                    crop, size=(pool_size, pool_size),
                    mode="bilinear", align_corners=False,
                )
            )
        return torch.stack(outs, dim=1)

    def roi_pool_per_image(full_img, boxes_bxywh, pool_size, resolution):
        """Stage 1's ``_roi_pool_per_image``: each image pooled with its OWN
        boxes -- the cleft path, where staging places the trapezium differently
        in every image."""
        rows = []
        for i in range(full_img.shape[0]):
            cols = []
            for (x, y, w, h) in boxes_bxywh[i].tolist():
                x, y = int(x), int(y)
                w, h = max(int(w), 1), max(int(h), 1)
                x2, y2 = min(x + w, resolution), min(y + h, resolution)
                crop = full_img[i : i + 1, :, y:y2, x:x2]
                cols.append(
                    F.interpolate(
                        crop, size=(pool_size, pool_size),
                        mode="bilinear", align_corners=False,
                    )[0]
                )
            rows.append(torch.stack(cols, dim=0))
        return torch.stack(rows, dim=0)

    class SRGNN(nn.Module):
        def __init__(self, num_outputs=1, pretrained=False):
            super().__init__()
            self.backbone = timm.create_model(
                BACKBONE_TIMM_NAME, pretrained=pretrained, features_only=True
            )
            self.self_attn = SeqSelfAttention(
                REGION_FEATURE_DIM, units=SELF_ATTENTION_UNITS
            )
            self.gnn_mlp1 = nn.Linear(BACKBONE_CHANNELS, GNN_CHANNELS)
            self.gnn_mlp2 = nn.Linear(GNN_CHANNELS, GNN_CHANNELS)
            self.gnn_out = nn.Linear(GNN_CHANNELS, DENSE_DIM)
            self.weighted_attn = SeqWeightedAttention(BACKBONE_CHANNELS)
            self.bn1 = nn.BatchNorm1d(BACKBONE_CHANNELS)
            self.bn2 = nn.BatchNorm1d(DENSE_DIM)
            self.classifier = nn.Linear(DENSE_DIM, num_outputs)
            self.dropout = nn.Dropout(0.2)
            #: The paper's own regions, for the boxes=None path. A buffer so it
            #: moves with the model and is recorded in its state.
            self.register_buffer(
                "default_boxes", torch.as_tensor(grid_rois()), persistent=True
            )

        @staticmethod
        def _propagate(h, alpha=APPNP_ALPHA, k=APPNP_K):
            """APPNP, written directly: x_{t+1} = (1-a) A_hat x_t + a x_0.

            The fully-connected GCN-normalised A_hat -- 2/(n+1) diagonal,
            1/(n+1) off-diagonal -- applied as a dense matmul batched over the
            batch dimension. Verified against ``torch_geometric.nn.APPNP`` to
            0.92 ULP worst case (``APPNP_PARITY_WORST_ULP``).
            """
            n = h.shape[1]
            a_hat = torch.full((n, n), 1.0 / (n + 1), device=h.device, dtype=h.dtype)
            a_hat.fill_diagonal_(2.0 / (n + 1))
            initial = h
            for _ in range(k):
                h = (1.0 - alpha) * torch.matmul(a_hat, h) + alpha * initial
            return h

        def forward(self, x, boxes=None):
            return self.forward_with_maps(x, boxes)[0]

        def forward_with_maps(self, x, boxes=None):
            feat = self.backbone(x)[-1]                       # (B, C, h, w)
            return self._from_features(feat, boxes)

        def forward_from_features(self, feat, boxes=None):
            """The cleft path: the FROZEN backbone's map arrives precomputed
            (the feature_map embedding artifact) and everything from here on
            trains -- the third regime. Bitwise-identical to forward() given
            the same map, asserted by scripts/verify_backbone_builds.py."""
            return self._from_features(feat, boxes)[0]

        def forward_from_features_with_weights(self, feat, boxes=None):
            """The cleft path, returning the REGION WEIGHTS as well.

            [ADDED 2026-08-14] Phase 8 arm B's explanation is
            ``region_w`` -- ``SeqWeightedAttention``'s softmax over
            regions -- and ``forward_from_features`` discards it, so the
            artifact-fed path had no way to see what the model weighted.

            **An accessor, not a second computation.** It returns the
            tuple ``_from_features`` already builds, exactly as
            ``forward_with_maps`` does for the raw-image path. A hook
            would re-read the same forward; reimplementing the pooling
            would put a parallel path under the same name
            (``phase8.SRGNN_NODE_WEIGHTS_R10_READ``).

            The weight vector is ``(B, N_REGIONS)`` = 27, of which the
            LAST entry is the whole feature map appended before attention
            -- ``N_REGIONS = N_ROIS + 1`` -- not an anatomy region.
            """
            return self._from_features(feat, boxes)

        def _from_features(self, feat, boxes=None):
            b, c = feat.shape[:2]
            full_img = F.interpolate(
                feat, size=(ROI_RESOLUTION, ROI_RESOLUTION),
                mode="bilinear", align_corners=False,
            )

            if boxes is None:
                roi = roi_pool(full_img, self.default_boxes, POOL_SIZE, ROI_RESOLUTION)
            elif isinstance(boxes, (list, tuple)):
                roi = roi_pool(full_img, boxes[0], POOL_SIZE, ROI_RESOLUTION)
            elif boxes.dim() == 3:
                roi = roi_pool_per_image(full_img, boxes, POOL_SIZE, ROI_RESOLUTION)
            else:
                roi = roi_pool(full_img, boxes, POOL_SIZE, ROI_RESOLUTION)

            # The whole image appended as the last region, from the raw
            # backbone output.
            if feat.shape[-2:] == (POOL_SIZE, POOL_SIZE):
                whole = feat
            else:
                whole = F.interpolate(
                    feat, size=(POOL_SIZE, POOL_SIZE),
                    mode="bilinear", align_corners=False,
                )
            roi = torch.cat([roi, whole.unsqueeze(1)], dim=1)  # (B, N, C, p, p)
            n = roi.size(1)

            jcvs = roi.reshape(b, n, -1)
            jcvs = self.dropout(jcvs)

            # Branch A (Sec III-D): self-attention, then per-region GMP.
            a = self.self_attn(jcvs)
            a = a.reshape(b, n, c, POOL_SIZE, POOL_SIZE)
            x_gmp = a.amax(dim=(3, 4))

            # Branch B (Sec III-C): GAP each region, APPNPConv x2. Each conv is
            # Dense then one propagation -- the released code's Spektral output
            # Dense is linear, so there is NO sigmoid anywhere in this branch.
            x1 = jcvs.reshape(b, n, c, POOL_SIZE, POOL_SIZE).mean(dim=(3, 4))
            h = self._propagate(self.gnn_mlp1(x1))
            h = self._propagate(self.gnn_mlp2(h))
            x3 = self.gnn_out(h.sum(dim=1))

            # Branch A weighted attention -> BN -> sigmoid (Eq. 5).
            x2, region_w = self.weighted_attn(x_gmp)
            x2 = torch.sigmoid(self.bn1(x2))

            # Skip connection: ft_bar = ft + x2 * ft.
            x4 = self.dropout(x3)
            out = self.bn2(x4 + x2 * x4)
            return self.classifier(out), region_w

    model = SRGNN(num_outputs=num_outputs, pretrained=pretrained)

    if len(np.asarray(model.default_boxes.cpu())) != N_ROIS:
        raise SRGNNSpecError(
            f"grid enumeration produced {len(model.default_boxes)} boxes, not "
            f"{N_ROIS}; the region set is not the paper's"
        )

    # The construction assertion, per component, so a drift says WHERE.
    spec = parameter_count(num_outputs=num_outputs)
    components = {
        "backbone": model.backbone,
        "self_attention": model.self_attn,
        "gnn_mlp1": model.gnn_mlp1,
        "gnn_mlp2": model.gnn_mlp2,
        "gnn_out": model.gnn_out,
        "weighted_attention": model.weighted_attn,
        "classifier": model.classifier,
    }
    drifted = []
    for name, module in components.items():
        got = sum(p.numel() for p in module.parameters())
        if got != spec[name]:
            drifted.append(f"{name}: built {got:,} vs spec {spec[name]:,}")
    batch_norms = sum(
        p.numel() for m in (model.bn1, model.bn2) for p in m.parameters()
    )
    if batch_norms != spec["batch_norms"]:
        drifted.append(f"batch_norms: built {batch_norms:,} vs spec {spec['batch_norms']:,}")
    total = sum(p.numel() for p in model.parameters())
    if drifted or total != spec["total"]:
        raise SRGNNSpecError(
            f"the constructed model drifts from the ported specification "
            f"(total {total:,} vs {spec['total']:,}): {'; '.join(drifted) or 'total only'}. "
            "Do NOT adjust the spec to match -- the correspondence with the "
            "weight-transplant-verified architecture depends on these counts."
        )
    assert_backbone_is_the_measured_one(model.backbone)

    # Normalization: READ off the timm backbone's own pretrained_cfg and
    # stamped onto the wrapper, because normalization_for does not descend
    # into children. Refused, never defaulted, if the cfg is missing.
    cfg = getattr(model.backbone, "pretrained_cfg", None) or getattr(
        model.backbone, "default_cfg", None
    )
    if not cfg or cfg.get("mean") is None or cfg.get("std") is None:
        raise SRGNNSpecError(
            f"the {BACKBONE_TIMM_NAME!r} feature wrapper reports no mean/std "
            "in its pretrained_cfg; refusing to guess. DO NOT SUBSTITUTE A "
            "DEFAULT -- that is the Stage-1 defect."
        )
    stamp_normalization(
        model, cfg["mean"], cfg["std"],
        source=f"timm pretrained_cfg ({BACKBONE_TIMM_NAME}, features_only wrapper)",
    )
    return model


def assert_backbone_is_the_measured_one(model) -> dict:
    """Check a built backbone against the measured constant. Cluster-side.

    **This is what makes the deprecated-alias risk survivable.** ``xception``
    resolves to ``legacy_xception`` today; if a version bump ever pointed it at a
    different variant, the count would move and the port would silently stop
    matching the transplant-verified architecture. Called at build time, that
    becomes a loud failure instead of a plausible model.
    """
    actual = sum(p.numel() for p in model.parameters())
    if actual != XCEPTION_FEATURE_PARAMETERS:
        raise SRGNNSpecError(
            f"the backbone has {actual:,} parameters; "
            f"{XCEPTION_FEATURE_PARAMETERS:,} was measured in "
            f"{MEASURED_IN_IMAGE}. Either the image changed or "
            f"{BACKBONE_TIMM_NAME!r} now resolves to a different Xception "
            "variant. Do NOT update the constant to match -- the port's "
            "correspondence with the weight-transplant-verified architecture "
            "depends on this being the same backbone."
        )
    return {"backbone_parameters": actual, "measured_in": MEASURED_IN_IMAGE}
