"""AG-Net: SIFT+GMM semantic regions, self-attention, inter-attention.

**[LITERATURE]** Bera, Wharton, Liu, Bessis & Behera, "Attend and Guide (AG-Net):
A Keypoints-Driven Attention-Based Deep Network for Image Recognition",
*IEEE TIP* 30:3691-3704 (2021).

============================================================================
PROVENANCE IS **NOT** SR-GNN's. READ THIS BEFORE COMPARING THEM.
============================================================================
``srgnn.py`` is a port of an implementation verified by **weight transplant**
against the authors' released weights -- 257/257 tensors, <1e-6 at every internal
stage. **No equivalent exists for AG-Net**, and treating the two as equally
verified would be the provenance error this project keeps finding.

What AG-Net has instead:

* a **paper-faithful implementation** written from Bera 2021 Sections III-B/III-C
  and Fig. 1, ported here;
* **no released authors' weights**, so no transplant is possible;
* one third-party reproduction (``AG-Net-main``), which a Stage 1 forensic
  analysis **discredited**: its headline 98.3% on Caltech-256 is void -- built on
  broken within-category indices over a 7-class, ~92%-contaminated subset, so it
  "is not a Caltech-256 result at all". Its box geometry is buggy and its trained
  ``delta`` is ~0.007, meaning the self-attention it claims to demonstrate was
  effectively off.

**So the reproduction is a reference for what the code did, not evidence that it
was right**, and the Stage 1 implementation deliberately diverges from it in
documented places (below). AG-Net's arm therefore carries a weaker provenance
claim than SR-GNN's, and the write-up must say so rather than describing "two
ported architectures" as though they were equally underwritten.

----------------------------------------------------------------------------
DELIBERATE DIVERGENCES FROM THE THIRD-PARTY REPRODUCTION
----------------------------------------------------------------------------
Both replace per-slot parameter lists with **shared weights**, which is the
standard reading of the paper and is why this model is far smaller than that
reproduction's 57.2M checkpoint:

* **SE-Residual**: one shared SE block applied to every region. The reproduction
  used a ``ModuleList`` indexed by region position; the paper (Sec III-B) cites
  SENet and describes a block applied to each region's features, with no
  position-specific weights. Shared also means it receives R+1 times the gradient
  signal, which matters at the paper's learning rate.
* **Inter-attention**: one shared ``W_u / W'_u / W_m / W_alpha`` across regions
  and batch items. The reproduction indexed weights by **batch position**, so the
  parameters an image was scored with depended on where it landed in the batch --
  and its own text claims "shared 1x1 convolutions for parameter efficiency",
  which its code contradicts.

**Do not treat the 57.2M figure as a target.** It counts an architecture this
port intentionally does not implement.

----------------------------------------------------------------------------
WHAT THE REGION COUNT IS, AND WHY THERE IS NO GMM DECISION TO MAKE
----------------------------------------------------------------------------
The open question was whether SIFT+GMM regions -- image-dependent and variable in
number -- would need padding, variable-length storage, or a fixed component count
recorded as a departure. **None of them: the authors' own design already fixes
it.** ``kappa`` is the GMM component count, and the regions are the kappa
primaries plus every unordered pair, so::

    R + 1 = 1 + kappa * (kappa + 1) / 2

is **constant by construction** for a fixed kappa. At the reproduction's
``kappa=8`` that is 37 regions. The artifact is rectangular without any departure
being recorded, and the comparison against SR-GNN's 27 is fair because both are
fixed counts rather than one fixed and one padded.

----------------------------------------------------------------------------
THIS MODULE IMPORTS WITHOUT TORCH
----------------------------------------------------------------------------
Torch lives inside ``build``. Module level holds the specification plus
``parameter_count``, derived arithmetically, so the built model can be asserted
against it on the cluster.
"""

from __future__ import annotations

#: ResNet-50's final conv stage. [LITERATURE] Bera 2021 uses ResNet-50; SR-GNN
#: uses Xception. Both tap 2048 channels, which is why the two arms' region
#: descriptors are the same width despite different backbones.
BACKBONE_CHANNELS = 2048

#: **[MEASURED 2026-07-31, in the pinned image]** torchvision ResNet-50 minus
#: ``avgpool`` and ``fc`` (the port keeps ``children()[:8]``): **23,508,032**,
#: from a full model of **25,557,032** less a 2048->1000 classifier. Both
#: confirmed in the image below, matching the arithmetic exactly.
RESNET50_BODY_PARAMETERS = 23_508_032

#: [MEASURED] The full model, recorded so the subtraction is checkable rather
#: than asserted.
RESNET50_FULL_PARAMETERS = 25_557_032

#: The image the parameter constants were measured in. **The digest IS the
#: provenance**: these numbers are only true of this environment, and a measured
#: constant without the environment that produced it cannot be re-derived.
#:
#: Unlike SR-GNN's Xception, ResNet-50 comes from torchvision under an explicit
#: weights enum rather than a deprecated alias, so there is no silent-remap risk
#: here -- but the build-time assertion is kept symmetric anyway, because "this
#: one is safe" is how the other one would have been missed.
#: On ONE line: split across two, the first ends with the digest prefix and no
#: hex, and the digest-hygiene test flags it -- correctly. See srgnn.py.
MEASURED_IN_IMAGE = "redring/cleft-aesthetics@sha256:2135e27b5d82b28cb5e2059c606aadf5736df80e50cb4e52fc669cc8ba33d2ab"  # noqa: E501

#: GMM component count. The regions are the kappa primaries plus every unordered
#: pair, so the total is fixed once kappa is: R+1 = 1 + kappa*(kappa+1)/2.
KAPPA = 8

#: SAGAN self-attention reduces channels by 8 for the query/key projections.
SELF_ATTENTION_REDUCTION = 8

#: SE block bottleneck. [LITERATURE] SENet's default.
SE_REDUCTION = 16

#: Inter-attention channel reduction in the spatial path.
INTER_ATTENTION_REDUCTION = 8

#: ROI-align output side for each region.
ROI_SIZE = 7

#: **[MEASURED 2026-07-31] AG-Net is the one backbone that cannot report its own
#: preprocessing, and that is a hole in the Stage-1 guard rather than a detail.**
#:
#: The backbone is torchvision's ResNet-50, and torchvision models carry **no**
#: ``pretrained_cfg`` and no ``default_cfg`` -- not on the model, not on the
#: ``children()[:8]`` body this port uses. ``factory.normalization_for`` had
#: nothing to read for it, so the options were to refuse (breaking AG-Net) or to
#: substitute a default, which IS the Stage-1 defect.
#:
#: Resolved by reading torchvision's own provenance: the weights enum carries
#: ``transforms()`` with the mean and std those weights were trained under, and
#: ``build`` stamps them via ``factory.stamp_normalization`` with the enum named
#: as the source. The numbers still come from the weights; nothing is assumed.
#:
#: **The four backbones split two-two**, which is why no default would have been
#: even majority-right: ViT-B/16 and SR-GNN's Xception use 0.5/0.5, Swin-B and
#: this ResNet-50 use ImageNet statistics.
#:
#: ``IMAGENET1K_V1`` and ``V2`` carry the SAME mean/std and differ only in resize
#: (256 vs 232) -- which does not reach this pipeline, since staged crops arrive
#: at 224 already and are not resized again. Recorded so the weights choice is
#: not mistaken for a normalization choice: they are different quantities.
NORMALIZATION_SOURCE = "torchvision ResNet50_Weights.<enum>.transforms()"
RESNET_WEIGHTS_DEFAULT = "IMAGENET1K_V2"

#: **[MEASURED 2026-07-31] The whole model, constructed.** Hit its specification
#: exactly on the first construction -- 30,742,924 at 200 classes. Recorded
#: because until this ran the specification was arithmetic that nothing had ever
#: tested against an ``nn.Module``.
#:
#: Built with torchvision 0.27.1 / torch 2.13.0 outside the pinned image, so it
#: checks the SPEC rather than the image; the ResNet-50 body term matched the
#: image-measured constant on a different torchvision, which is what makes the
#: number a property of the architecture rather than of one environment.
BUILT_PARAMETERS_CUB = 30_742_924

#: What the first construction ran in, as data rather than prose. See
#: ``srgnn.BUILD_VERIFICATION`` for why this is a field and not a docstring.
BUILD_VERIFICATION = {
    "measured": "2026-07-31",
    "built_parameters_cub": BUILT_PARAMETERS_CUB,
    "matched_spec_first_construction": True,
    "environment": "torch 2.13.0, torchvision 0.27.1",
    "in_pinned_image": False,
    "what_this_proves": (
        "the SPECIFICATION, not the image: the ResNet-50 body term matched the "
        "image-measured constant on a different torchvision"
    ),
}

#: **There is no published parameter count for AG-Net to check against**, unlike
#: SR-GNN's ~30.9M. The third-party reproduction's checkpoint carries 57,223,560
#: parameters over 542 tensors, but that counts per-slot ``ModuleList`` weights
#: this port deliberately shares, so it is **not a target** -- it is recorded only
#: so the difference is explained rather than discovered.
REPRODUCTION_PARAMETERS = 57_223_560


def region_count(kappa: int = KAPPA) -> int:
    """Regions including the whole image -- the ceiling, not a constant.

    **[QUALIFIED 2026-08-09, gate 3]** "Constant for a fixed kappa", as this
    read before, is exact only on content with enough SIFT keypoints: below
    ``MIN_KEYPOINTS_FOR_FULL_KAPPA`` the clustering falls back to
    ``FALLBACK_KAPPA`` (4 regions), and keypoint-starved content collapses
    to the whole image alone (2 regions -- measured on flat and near-flat
    images). The batch-padding machinery in ``pool_regions_spatial`` exists
    BECAUSE realised counts vary per image. On full-keypoint content the
    law is exact: 36 ``generate_srs`` boxes plus the appended whole image
    = 37 (``roadb.GRAPH_GEOMETRY_AT_RESOLUTION``). Note the two quantities:
    ``generate_srs`` returns the boxes WITHOUT the whole image, which is
    appended at pooling -- comparing an srs count against this function's
    value under one name cost a review round.
    """
    if kappa < 1:
        raise ValueError(f"kappa must be positive, got {kappa}")
    return 1 + kappa * (kappa + 1) // 2


def parameter_count(
    *,
    backbone_parameters: int = RESNET50_BODY_PARAMETERS,
    channels: int = BACKBONE_CHANNELS,
    num_outputs: int = 1,
) -> dict:
    """Total parameters, derived from the ported specification.

    Note what does **not** appear: the region count. Every attention and SE
    weight is shared across regions, so the parameter total is independent of
    ``kappa`` -- which is the whole point of the divergence from the
    reproduction, and is worth being able to see in the arithmetic.
    """
    reduced = max(channels // SELF_ATTENTION_REDUCTION, 1)

    # SAGAN self-attention: 1x1 convs for query, key, value, plus the delta gate.
    self_attention = (
        (channels * reduced + reduced)      # query_conv
        + (channels * reduced + reduced)    # key_conv
        + (channels * channels + channels)  # value_conv
        + 1                                 # delta
    )

    # SE-Residual: one shared block, squeeze to channels // SE_REDUCTION.
    se_hidden = max(channels // SE_REDUCTION, 1)
    se_residual = (
        (channels * se_hidden + se_hidden) + (se_hidden * channels + channels)
    )

    # Inter-attention, spatial path: 1x1 convs, shared across regions and batch.
    inter_reduced = max(channels // INTER_ATTENTION_REDUCTION, 1)
    inter_attention = (
        (channels * inter_reduced)              # Wu (bias=False)
        + (channels * inter_reduced + inter_reduced)  # Wu_prime
        + (inter_reduced + 1)                   # Wm
        + (channels + 1)                        # Wa
    )

    fusion = channels + 1                       # shared W_omega, C -> 1
    classifier = channels * num_outputs + num_outputs

    head = self_attention + se_residual + inter_attention + fusion + classifier
    return {
        "backbone": backbone_parameters,
        "self_attention": self_attention,
        "se_residual": se_residual,
        "inter_attention": inter_attention,
        "fusion": fusion,
        "classifier": classifier,
        "head_total": head,
        "total": backbone_parameters + head,
    }


def describe(kappa: int = KAPPA, num_outputs: int = 1) -> dict:
    """The specification and what can and cannot be checked about it."""
    counts = parameter_count(num_outputs=num_outputs)
    return {
        "regions": region_count(kappa),
        "kappa": kappa,
        "region_count_is_constant": True,
        "parameters": counts,
        "verified_by": (
            "NOT a weight transplant -- no authors' weights are released for "
            "AG-Net. This is a paper-faithful implementation from Bera 2021 "
            "Sec III-B/III-C. The one third-party reproduction was forensically "
            "discredited (its 98.3% is void, its box geometry buggy, its trained "
            "self-attention effectively off), so it is a reference for what code "
            "did, not evidence that it was right."
        ),
        "provenance_is_weaker_than_srgnn": True,
        "reproduction_parameters": REPRODUCTION_PARAMETERS,
        "why_not_a_target": (
            "the reproduction indexes SE and inter-attention weights per region "
            "and per BATCH POSITION; this port shares them, per the paper. Its "
            "57.2M counts an architecture deliberately not implemented here."
        ),
    }


#: **[MEASURED 2026-07-31, in the pinned image -- figures pasted by the
#: maintainer] AG-Net trains deterministic to ~2.3e-05, not exactly, and the
#: flag cannot close it there.** Post-restore step max |diff| **2.26e-05** in
#: the image (2.31e-05 in the local corroborating run) where every other
#: backbone's restore check is byte-identical. Mechanism: ``roi_align``'s
#: CUDA backward uses atomicAdd. The contrast confirms it -- ViT-B/16,
#: Swin-B and SR-GNN measured bitwise flag-off in the same image run, and
#: the isolated-op probes put both pooling ops in the atomics class
#: (interpolate backward 5.25e-06, roi_align backward 1.49e-08): only
#: AG-Net's instantiation lands on colliding atomics. Same-instance
#: repeat-gradient is 0.0 even for AG-Net; the divergence appears across
#: model instances, where allocation moves the atomic scheduling.
#:
#: **Why the determinism flag is NOT the fix here [MEASURED]:**
#:
#: * torchvision's flag-on route substitutes a deterministic pure-PyTorch
#:   roi_align -- but the route is gated on torch.compile, which needs
#:   inductor, which needs a C++ compiler, and the pinned image has none:
#:   ``InvalidCxxCompiler``, measured. The substitution is unreachable where
#:   the runs happen.
#: * the op never RAISES under the flag either way, so gate 1's
#:   "``use_deterministic_algorithms(True)`` raised on nothing" canary is
#:   structurally blind to it. Nondeterminism here is detected by comparing
#:   two runs, never by the flag erroring.
#:
#: **So the twelve pretraining configs run ``deterministic: false``**
#: [DECIDED]: the flag buys the three bitwise backbones nothing and cannot
#: help this one. The bound is the recorded property: same-seed AG-Net runs
#: differ by up to ~2.3e-05, its resume comparison is a tolerance rather
#: than an equality, and any AG-Net delta needs its own seed band measured
#: UNDER that condition -- the band's floor is kernel noise, not zero.
TRAINING_DETERMINISM = {
    "measured": "2026-07-31",
    "environment": "the pinned image (figures pasted by the maintainer)",
    "device": "cuda",
    "flag_off_post_restore_step_max_diff": 2.26e-05,
    "local_corroboration_max_diff": 2.31e-05,
    "flag_off_repeat_grad_max_diff": 0.0,
    "bitwise_flag_off": False,
    "mechanism": "torchvision::roi_align CUDA backward uses atomicAdd",
    "under_deterministic_flag": (
        "cannot fix it in the image: the deterministic substitution never "
        "raises (gate-1's raised-on-nothing canary cannot see this op) and "
        "is gated on torch.compile -> inductor -> a C++ compiler the image "
        "does not have (InvalidCxxCompiler, measured)"
    ),
    "flag_off_consequence": (
        "same-seed reproducibility and resume identity bounded at ~2.3e-05; "
        "an AG-Net delta needs its own seed band measured under that condition"
    ),
    "decision": "the twelve pretraining configs run deterministic: false",
    "authoritative_for_runs": "the image run of scripts/verify_train_determinism.py",
}

#: [DECIDED 2026-07-31] The GMM's k-means initialisation is SEEDED (Stage 1
#: left it unseeded). Stage 1 generated regions once in its data pipeline, so
#: nondeterminism never re-entered a run; here regions are generated inside
#: ``forward``, and an unseeded GMM would give a resumed run different boxes
#: than the run it resumes -- breaking the byte-identity the checkpoint
#: contract promises. One fixed seed, not the config seed: the regions are a
#: property of the IMAGE, and must not vary across arms.
GMM_RANDOM_STATE = 0

#: Section III-A thresholds, from the paper via the Stage 1 port: top 50% of
#: keypoints by response; fewer than 9 keypoints drops kappa to 2; fewer than
#: 2 usable keypoints falls back to the whole image as the single region.
MIN_KEYPOINTS_FOR_FULL_KAPPA = 9
FALLBACK_KAPPA = 2


def generate_srs(image_bgr, kappa: int = KAPPA, reg_covar: float = 1e-6):
    """Semantic regions: SIFT keypoints clustered by GMM, primaries plus pairs.

    Ported verbatim from Stage 1 (``agnet_regions.generate_srs``) except for
    ``GMM_RANDOM_STATE`` -- see that constant. Returns (R, 4) float32 boxes as
    normalised (x1, y1, x2, y2); falls back to the whole image when detection
    or clustering fails, which downstream padding absorbs.

    cv2 and sklearn are imported here, not at module level: both are in the
    pinned image, and the module must stay importable without them.
    """
    from collections import defaultdict

    import cv2
    import numpy as np
    from sklearn.mixture import GaussianMixture

    h, w = image_bgr.shape[:2]

    sift = cv2.SIFT_create()
    kps, _ = sift.detectAndCompute(image_bgr, None)
    kps = sorted(kps, key=lambda k: k.response, reverse=True)

    kappa_eff = kappa
    if len(kps) < MIN_KEYPOINTS_FOR_FULL_KAPPA:
        kappa_eff = FALLBACK_KAPPA
    else:
        kps = kps[: max(len(kps) // 2, 1)]

    if len(kps) > 1:
        pts = cv2.KeyPoint_convert(kps)
        unique_pts = np.unique(pts, axis=0)
        n_comp = min(kappa_eff, len(unique_pts))

        if n_comp > 1:
            try:
                gmm = GaussianMixture(
                    n_components=n_comp, covariance_type="full",
                    init_params="kmeans", reg_covar=reg_covar,
                    max_iter=100, tol=1e-3, n_init=1,
                    random_state=GMM_RANDOM_STATE,
                )
                gmm.fit(unique_pts)
                gmm_labels = gmm.predict(pts)

                clusters = defaultdict(list)
                for pt, lbl in zip(pts, gmm_labels):
                    clusters[lbl].append(pt)

                prim = {}
                for lbl, points in clusters.items():
                    xs, ys = zip(*points)
                    prim[lbl] = dict(min_x=min(xs), max_x=max(xs),
                                     min_y=min(ys), max_y=max(ys))

                boxes = []
                for i, pi in prim.items():
                    for j, pj in prim.items():
                        if i <= j:
                            boxes.append([
                                min(pi["min_x"], pj["min_x"]),
                                min(pi["min_y"], pj["min_y"]),
                                max(pi["max_x"], pj["max_x"]),
                                max(pi["max_y"], pj["max_y"]),
                            ])

                if boxes:
                    arr = np.array(boxes, dtype=np.float32)
                    arr[:, [0, 2]] /= w
                    arr[:, [1, 3]] /= h
                    return np.clip(arr, 0.0, 1.0)
            except (ValueError, FloatingPointError):
                pass

    return np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)


def build(*, pretrained: bool = True, num_outputs: int = 1, **overrides):
    """The torch model: the paper-faithful assembly, constructed.

    **Where each part comes from -- moved, not rewritten:**

    * the construction is verbatim from the first-construction script that
      measured ``BUILT_PARAMETERS_CUB`` hitting the spec exactly (the shared-
      weights layout, NOT the discredited reproduction's per-slot lists);
    * the forwards are the Stage 1 spatial path
      (``cleft-aesthetic-assessment/models/agnet.py`` and
      ``agnet_interattention.py``): SAGAN self-attention, shared SE-Residual
      on spatial region maps, spatial inter-attention returning the refined
      map Eq. 4 expects, and the GMP/GAP fusion head;
    * region proposals are Stage 1's ``generate_srs`` (above), with the GMM
      seeded -- the one recorded deviation.

    **``forward(x)`` generates its own regions** [DECIDED]: Stage 1's data
    pipeline supplied boxes, but the pretraining loop hands the model images
    and nothing else. The input is denormalised back to uint8 through the
    model's own stamp (so SIFT sees the image, not standardised floats), run
    as BGR to match cv2's grayscale weighting, and the boxes are cached per
    image content -- SIFT+GMM runs once per distinct image, not once per
    epoch. ``forward(x, boxes)`` still accepts explicit per-image boxes.

    Asserts its parameter count against ``parameter_count()`` per component
    and the ResNet-50 body against the measured constant, then stamps
    normalization from ``ResNet50_Weights.IMAGENET1K_V2.transforms()`` -- the
    weights' own preprocessing, read off the enum, never assumed
    (``NORMALIZATION_SOURCE``).
    """
    if overrides:
        raise ValueError(
            f"unknown build override(s): {sorted(overrides)}. No overrides are "
            "defined; silently absorbing one would run an arm nobody configured."
        )
    try:
        import hashlib

        import numpy as np
        import torch
        import torchvision
        from torch import nn
        from torchvision.models import ResNet50_Weights
        from torchvision.ops import roi_align
    except ImportError as exc:
        raise ValueError(
            f"building AG-Net needs torch and torchvision, which are not "
            f"importable here ({exc}). They are in the pinned image; the "
            "laptop suite exercises the loop through the stub backbone."
        ) from None

    from .factory import NORMALIZATION_ATTRIBUTE, stamp_normalization

    class SelfAttention(nn.Module):
        """SAGAN self-attention (Eq. 1); delta starts at zero -> identity."""

        def __init__(self, c):
            super().__init__()
            r = max(c // SELF_ATTENTION_REDUCTION, 1)
            self.query_conv = nn.Conv2d(c, r, 1)
            self.key_conv = nn.Conv2d(c, r, 1)
            self.value_conv = nn.Conv2d(c, c, 1)
            self.delta = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            b, c, h, w = x.shape
            n = h * w
            q = self.query_conv(x).view(b, -1, n).permute(0, 2, 1)
            k = self.key_conv(x).view(b, -1, n)
            attention = torch.softmax(torch.bmm(q, k), dim=-1)
            v = self.value_conv(x).view(b, -1, n)
            out = torch.bmm(v, attention.permute(0, 2, 1)).view(b, c, h, w)
            return self.delta * out + x

    def se_block(c, reduction):
        r = max(c // reduction, 1)
        return nn.Sequential(
            nn.Linear(c, r), nn.ReLU(inplace=True), nn.Linear(r, c), nn.Sigmoid()
        )

    class SEResidualSpatial(nn.Module):
        """ONE shared SE block over every region's spatial map (Sec III-B).
        Shared is the paper's reading and why the count is kappa-independent."""

        def __init__(self, c):
            super().__init__()
            self.SE = se_block(c, SE_REDUCTION)

        def forward(self, x):
            b, r1, c, h, w = x.shape
            squeezed = x.mean(dim=(3, 4))
            scale = self.SE(squeezed.reshape(b * r1, c)).view(b, r1, c, 1, 1)
            return x + x * scale

    class InterAttentionSpatial(nn.Module):
        """Eq. 2-3 on spatial region maps via shared 1x1 convs; returns the
        SPATIAL refined map f_hat that Eq. 4's head pools."""

        def __init__(self, c):
            super().__init__()
            rc = max(c // INTER_ATTENTION_REDUCTION, 1)
            self.Wu = nn.Conv2d(c, rc, 1, bias=False)
            self.Wu_prime = nn.Conv2d(c, rc, 1, bias=True)
            self.Wm = nn.Conv2d(rc, 1, 1)
            self.Wa = nn.Conv2d(c, 1, 1)

        def forward(self, fr):
            b, r1, c, h, w = fr.shape
            flat = fr.reshape(b * r1, c, h, w)
            up = self.Wu(flat).reshape(b, r1, -1, h, w)
            vp = self.Wu_prime(flat).reshape(b, r1, -1, h, w)
            rc = up.shape[2]

            u = torch.tanh(up.unsqueeze(2) + vp.unsqueeze(1))
            m = torch.sigmoid(
                self.Wm(u.reshape(b * r1 * r1, rc, h, w))
            ).reshape(b, r1, r1, h, w)

            m_perm = m.permute(0, 3, 4, 1, 2).reshape(b * h * w, r1, r1)
            fr_perm = fr.permute(0, 3, 4, 1, 2).reshape(b * h * w, r1, c)
            alpha = torch.bmm(m_perm, fr_perm)
            alpha = alpha.reshape(b, h, w, r1, c).permute(0, 3, 4, 1, 2)

            score = self.Wa(alpha.reshape(b * r1, c, h, w)).reshape(b, r1, h, w)
            w_spatial = torch.softmax(score, dim=1)
            importance = w_spatial.mean(dim=(2, 3))

            f_hat = (w_spatial.unsqueeze(2) * alpha).sum(1)
            return f_hat, importance

    class GapGmpFusion(nn.Module):
        """Eq. 4: softmax blend of GMP and GAP through one shared W_omega."""

        def __init__(self, c):
            super().__init__()
            self.w = nn.Linear(c, 1)

        def forward(self, gap, gmp):
            omega = torch.softmax(
                torch.cat([self.w(gmp), self.w(gap)], dim=-1), dim=-1
            )
            return omega[:, 0:1] * gmp + omega[:, 1:2] * gap

    def pool_regions_spatial(feat, boxes, output_size):
        """ROI-align each region keeping its (C, s, s) map; variable counts
        padded to the batch maximum by repeating the whole-image region."""
        b, c, h, w = feat.shape
        region_maps, counts = [], []
        for i in range(b):
            bx = boxes[i].clone()
            bx[:, [0, 2]] *= w
            bx[:, [1, 3]] *= h
            whole = torch.tensor([[0, 0, w, h]], dtype=bx.dtype, device=bx.device)
            bx = torch.cat([bx, whole], 0)
            idx = torch.zeros((bx.size(0), 1), dtype=bx.dtype, device=bx.device)
            pooled = roi_align(feat[i : i + 1], torch.cat([idx, bx], 1), output_size)
            region_maps.append(pooled)
            counts.append(pooled.size(0))

        r_max = max(counts)
        padded = []
        for maps in region_maps:
            if maps.size(0) < r_max:
                pad = maps[-1:].repeat(r_max - maps.size(0), 1, 1, 1)
                maps = torch.cat([maps, pad], 0)
            padded.append(maps)
        return torch.stack(padded, 0)

    class AGNet(nn.Module):
        def __init__(self, num_outputs=1, pretrained=False):
            super().__init__()
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            resnet = torchvision.models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(resnet.children())[:8])
            c = BACKBONE_CHANNELS
            self.self_attn = SelfAttention(c)
            self.se = SEResidualSpatial(c)
            self.inter_attn = InterAttentionSpatial(c)
            self.fusion = GapGmpFusion(c)
            self.classifier = nn.Linear(c, num_outputs)
            #: Per-image-content region cache: SIFT+GMM is a pure function of
            #: the pixels (seeded GMM), so each distinct image is proposed
            #: once, not once per epoch. Plain attribute, deliberately not
            #: state: a restored model regenerates identical boxes.
            self._region_cache: dict[str, np.ndarray] = {}

        def _generate_regions(self, x):
            """Boxes for a normalised batch, through the model's own stamp."""
            stamp = getattr(self, NORMALIZATION_ATTRIBUTE, None)
            if stamp is None:
                raise RuntimeError(
                    "AG-Net has no normalization stamp; regions cannot be "
                    "generated from standardised floats. Build through "
                    "agnet.build(), which stamps it."
                )
            mean = torch.tensor(stamp["mean"], device=x.device).view(1, -1, 1, 1)
            std = torch.tensor(stamp["std"], device=x.device).view(1, -1, 1, 1)
            images = (
                ((x * std + mean).clamp(0.0, 1.0) * 255.0)
                .round().to(torch.uint8)
                .permute(0, 2, 3, 1).cpu().numpy()
            )
            boxes = []
            for image in images:
                bgr = image[:, :, ::-1]
                key = hashlib.sha1(np.ascontiguousarray(bgr)).hexdigest()
                if key not in self._region_cache:
                    self._region_cache[key] = generate_srs(bgr)
                boxes.append(
                    torch.as_tensor(self._region_cache[key], device=x.device)
                )
            return boxes

        def forward(self, x, boxes=None):
            return self.forward_with_maps(x, boxes)[0]

        def forward_with_maps(self, x, boxes=None):
            feat = self.backbone(x)
            if boxes is None:
                boxes = self._generate_regions(x)
            return self._from_features(feat, boxes)

        def forward_from_features(self, feat, boxes):
            """The cleft path: the FROZEN backbone's map arrives precomputed
            and everything from here on trains -- including the SAGAN
            self-attention, which is why the boundary sits BEFORE it. Boxes
            are required: SIFT+GMM generation needs the image, which this
            path deliberately does not have, so the caller supplies the
            scheme's boxes (native = SR-GNN-style generated sets do not apply
            here; AG-Net's cleft native path also receives explicit boxes,
            precomputed from the images once). Bitwise-identical to forward()
            given the same map and boxes, asserted by
            scripts/verify_backbone_builds.py."""
            return self._from_features(feat, boxes)[0]

        def forward_from_features_with_weights(self, feat, boxes):
            """The cleft path, returning ``importance`` as well.

            [ADDED 2026-08-14] Found by grep while adding SR-GNN's, and
            it is the SAME defect: ``_from_features`` computes
            ``importance`` -- ``inter_attn``'s per-region weights, which
            are arm C's explanation -- and ``forward_from_features``
            returns ``[0]``, so the artifact-fed path is blind to what
            the raw-image path has always had.

            Two of two graph models, from one cause: the cleft path was
            written to return logits for training and the explanation
            rode along unused (``phase8.THE_ARTIFACT_PATH_DISCARDS_THE_
            EXPLANATION``). Fixed here rather than at arm C, where it
            would have cost a diagnosis instead of a grep.

            An accessor: the tuple ``_from_features`` already builds.
            """
            return self._from_features(feat, boxes)

        def _from_features(self, feat, boxes):
            feat = self.self_attn(feat)
            rmaps = pool_regions_spatial(feat, boxes, ROI_SIZE)
            rmaps = self.se(rmaps)
            f_hat, importance = self.inter_attn(rmaps)
            gmp = f_hat.amax(dim=(2, 3))
            gap = f_hat.mean(dim=(2, 3))
            descriptor = self.fusion(gap, gmp)
            return self.classifier(descriptor), importance

    model = AGNet(num_outputs=num_outputs, pretrained=pretrained)

    # The construction assertion, per component, so a drift says WHERE.
    spec = parameter_count(num_outputs=num_outputs)
    components = {
        "backbone": model.backbone,
        "self_attention": model.self_attn,
        "se_residual": model.se,
        "inter_attention": model.inter_attn,
        "fusion": model.fusion,
        "classifier": model.classifier,
    }
    drifted = []
    for name, module in components.items():
        got = sum(p.numel() for p in module.parameters())
        if got != spec[name]:
            drifted.append(f"{name}: built {got:,} vs spec {spec[name]:,}")
    total = sum(p.numel() for p in model.parameters())
    if drifted or total != spec["total"]:
        raise ValueError(
            f"the constructed model drifts from the ported specification "
            f"(total {total:,} vs {spec['total']:,}): {'; '.join(drifted) or 'total only'}. "
            "Do NOT adjust the spec to match; and do NOT reach for the "
            "reproduction's 57.2M -- it counts per-slot weights this port "
            "deliberately shares."
        )
    body = sum(p.numel() for p in model.backbone.parameters())
    if body != RESNET50_BODY_PARAMETERS:
        raise ValueError(
            f"the ResNet-50 body has {body:,} parameters; "
            f"{RESNET50_BODY_PARAMETERS:,} was measured in {MEASURED_IN_IMAGE}. "
            "torchvision has changed under this port; do not update the "
            "constant to match."
        )

    # The stamp: torchvision models report no pretrained_cfg, so the values
    # come from the weights enum's own transforms() -- read, never assumed.
    # The enum is named regardless of `pretrained`, because it is where the
    # values came from either way.
    transforms = ResNet50_Weights.IMAGENET1K_V2.transforms()
    stamp_normalization(
        model, transforms.mean, transforms.std,
        source="torchvision ResNet50_Weights.IMAGENET1K_V2.transforms()",
    )
    return model
