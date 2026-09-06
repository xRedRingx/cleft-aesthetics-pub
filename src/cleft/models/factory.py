"""Model construction, and the one place allowed to name normalization values.

``tests/test_normalization.py`` allowlists this file and fails if an ImageNet
mean/std triple appears anywhere else in ``src/``. It guards a real Stage-1 bug:
hardcoded ImageNet constants applied to backbones that were not trained with
them, which is silently wrong for those and silently right for the rest.

**Even here the constants are not written down.** ``normalization_for`` reads the
values off the model's own ``pretrained_cfg``, so a backbone with different
preprocessing gets its own. A hardcoded triple in the allowlisted file would
satisfy the test while reintroducing the defect.

timm and torch are imported lazily, so this module can be imported -- and the
rest of the package tested -- without either installed.
"""

from __future__ import annotations

import math

from typing import Any

#: [DECIDED] Phase 3's single backbone. LayerNorm, so no BatchNorm complications
#: while the harness is being built; SR-GNN and AG-Net arrive in Phase 6 and are
#: where the BN work bites.
DEFAULT_BACKBONE = "vit_base_patch16_224"

#: **[DECIDED 2026-07-31] Phase 6's four backbones, and what each one IS.**
#:
#: The ``kind`` is load-bearing rather than descriptive, because it decides three
#: separate things and they must not drift apart:
#:
#: * what the cleft fine-tuning trains -- ``transformer`` freezes the backbone and
#:   fits a linear head; ``graph`` freezes the backbone and trains the graph
#:   layers, because for SR-GNN and AG-Net **the message passing IS the
#:   contribution** and freezing it would reduce the arm to a linear probe on
#:   Xception features;
#: * what an embedding artifact holds -- a pooled ``(N, D)`` vector for the
#:   transformers, a per-region ``(N, R, D)`` array for the graph models, since a
#:   single pooled vector cannot feed a graph layer that has not been trained yet;
#: * which seed band applies. Phase 3's SD 0.0137 was measured on a frozen linear
#:   probe. A model with trainable graph parameters is a **third regime** and its
#:   band is a Phase 6 gate, not an inheritance.
BACKBONES = {
    "vit_b16": {
        "timm_name": "vit_base_patch16_224",
        "kind": "transformer",
        "norm": "LayerNorm",
        "embedding_dim": 768,
    },
    # **[MEASURED 2026-07-31]** embedding_dim confirmed against the BUILT model,
    # not claimed: timm reports num_features 1024 and a forward pass at
    # num_classes=0 returns 1024. 86,743,224 parameters.
    #
    # Its normalization is ImageNet statistics where ViT-B/16's is 0.5/0.5 --
    # measured, and exactly why `normalization_for` reads each model's own
    # `pretrained_cfg` instead of a shared constant. Two backbones in the same
    # registry disagreeing is the Stage-1 defect's precondition.
    "swin_b": {
        "timm_name": "swin_base_patch4_window7_224",
        "kind": "transformer",
        "norm": "LayerNorm",
        "embedding_dim": 1024,
    },
    # **Phase 7D's patch axis [MEASURED 2026-08-15, against the PINNED timm
    # 1.0.7 installed standalone].** All three ViT-B variants are 768-dim
    # CLS-token models on the 0.5/0.5 normalization, in21k->in1k pretrained
    # (patch32's default tag is augreg where 16/8 are augreg2 -- a recipe
    # difference, recorded in ladder.PHASE_7D_BUILT). Grids 7x7 / 14x14 /
    # 28x28: NATIVE to each model, which is the axis's whole point.
    "vit_b32": {
        "timm_name": "vit_base_patch32_224",
        "kind": "transformer",
        "norm": "LayerNorm",
        "embedding_dim": 768,
    },
    "vit_b8": {
        "timm_name": "vit_base_patch8_224",
        "kind": "transformer",
        "norm": "LayerNorm",
        "embedding_dim": 768,
    },
    # **[MEASURED 2026-08-15, pinned timm 1.0.7] MViTv2-Base, NOT the
    # paper's MViTv1 -- v1 does not exist in the pin at all** (the only
    # mvit* entries are mvitv2_*; samvit_* is SAM, unrelated). The
    # substitution and its two architectural deltas are recorded in
    # ladder.PHASE_7D_BUILT. 50,703,744 parameters, num_features 768,
    # NO cls token: global_pool='avg' over the final stage's 49 tokens,
    # ImageNet normalization statistics (not 0.5/0.5) -- both read off the
    # built model's own pretrained_cfg, which is why normalization_for
    # exists.
    "mvitv2_b": {
        "timm_name": "mvitv2_base",
        "kind": "transformer",
        "norm": "LayerNorm",
        "embedding_dim": 768,
    },
    # **[MEASURED 2026-09-02, Phase 25 -- and the measurement's home is
    # named because it is NOT the suite's.** timm 1.0.27 / torch 2.13 under
    # Python 3.13; the suite's 3.11 venv has neither installed, so the
    # tests that check these facts SKIP there. The cluster is pinned at
    # timm 1.0.7 and **whether 1.0.7 carries either tag is a BLOCKING READ
    # that cannot be answered from this machine** --
    # phase25.THE_CHECKPOINTS_RESOLVED.]
    #
    # **vit_b14_dinov2** -- DINOv2 base, self-supervised on LVD-142M.
    # num_features 768 (86,579,712 parameters), num_classes 0 already,
    # ImageNet normalization statistics from its own pretrained_cfg, and
    # **native input 518x518, which no other entry in this registry is**.
    # That is why timm_kwargs_for gained NATIVE_INPUT_SIZE: its (224, 224)
    # shortcut returns {} and DINOv2 built with no kwargs raises
    # `AssertionError: Input height (224) doesn't match model (518)` at the
    # first forward pass.
    "vit_b14_dinov2": {
        "timm_name": "vit_base_patch14_dinov2.lvd142m",
        "kind": "transformer",
        "norm": "LayerNorm",
        "embedding_dim": 768,
    },
    # **vit_b16_dino** -- the SAME-ENCODER control: original DINO, patch 16,
    # native 224, num_classes 0. **The same timm architecture as the probe's
    # `vit_base_patch16_224`, a different pretrained tag** -- which is what
    # makes it the only contrast in this project that varies the training
    # OBJECTIVE with the encoder held exactly fixed. It is DINO, not DINOv2
    # (phase25.THE_TWO_ARMS_RULED: the control isolates the objective, not
    # DINOv2 specifically).
    "vit_b16_dino": {
        "timm_name": "vit_base_patch16_224.dino",
        "kind": "transformer",
        "norm": "LayerNorm",
        "embedding_dim": 768,
    },
    "srgnn": {
        "timm_name": None,  # built here, not by timm
        "kind": "graph",
        "norm": "BatchNorm",
        "embedding_dim": 2048,  # Xception penultimate, per region
    },
    "agnet": {
        "timm_name": None,
        "kind": "graph",
        "norm": "BatchNorm",
        "embedding_dim": 2048,
    },
}

#: **Backbones whose NATIVE input is not 224.** Empty until Phase 25:
#: every earlier entry is a 224 model, which is why ``timm_kwargs_for``
#: could treat 224 as "no kwargs needed" for eight years of this project's
#: history. A 518-native model needs ``dynamic_img_size`` to accept 224,
#: and its position grid is then interpolated DOWN (37x37 -> 16x16) rather
#: than up, which is a first for this record and is a declared caveat on
#: the arm (``phase25.THE_TWO_ARMS_RULED``).
NATIVE_INPUT_SIZE = {
    "vit_base_patch14_dinov2.lvd142m": 518,
}

#: The two kinds, as a closed set, so a new backbone has to declare which it is
#: rather than defaulting into the transformer path by omission.
BACKBONE_KINDS = ("transformer", "graph")

#: **The Phase 7 ladder's backbones, as a CLOSED set -- distinct from the
#: constructable registry above.** [FOUND 2026-08-15] The ladder and the
#: embedding plan derived their lattices by iterating ``BACKBONES``
#: directly, so registering Phase 7D's three backbones for construction
#: silently derived twenty-one new LADDER arms -- Stage D cells, Q1/Q2
#: comparisons and extraction sets that no one designed. Registration for
#: construction and membership of the ladder are different facts, and the
#: iteration now names which one it means. 7D's arms live in their own
#: configs (``generate_phase7d_configs.py``) and compare against the
#: ladder's control without joining its lattice.
LADDER_BACKBONES = ("vit_b16", "swin_b", "srgnn", "agnet")


class FactoryError(RuntimeError):
    """A backbone could not be created."""


def backbone_spec(name: str) -> dict:
    """The registry entry for ``name``, refusing an unknown one by name.

    Takes the short name (``swin_b``), not the timm name: the short name is what
    a config declares and what an embedding artifact records, so it is the one
    that has to be stable.
    """
    if name not in BACKBONES:
        raise FactoryError(
            f"unknown backbone {name!r}; expected one of {sorted(BACKBONES)}"
        )
    spec = dict(BACKBONES[name])
    if spec["kind"] not in BACKBONE_KINDS:
        raise FactoryError(
            f"backbone {name!r} declares kind {spec['kind']!r}, not one of "
            f"{BACKBONE_KINDS}. The kind decides what fine-tuning trains, what "
            "an embedding artifact holds, and which seed band applies -- it "
            "cannot be left to a default."
        )
    return spec


def timm_kwargs_for(timm_name: str, input_size=None) -> dict:
    """Extra timm kwargs for a non-default input size. Torch-free on purpose.

    **[2026-08-09] Written because the 768 seed band would have died at the
    patch-embed assertion** -- ``create_backbone`` built
    ``vit_base_patch16_224`` with no dynamic sizing, so any non-224 input was
    an R10 fault waiting for a cluster run to find it. The routing is a pure
    function so the laptop suite can pin it without torch.

    * ``None`` or ``(224, 224)``: no extra kwargs. **The default path is
      byte-identical to the one that produced every existing number** -- Road
      A's embeddings must not move.
    * ViT at another size: ``dynamic_img_size=True``, measured 2026-08-09
      (``roadb.BACKBONE_SHAPE_CONSTRAINTS``); both dimensions must be
      multiples of the 16-pixel patch, which is what the staging rounds to.
      The position grid is interpolated from 14x14 -- at 768 that is a 48x48
      grid, and the interpolation is part of the procedure being measured,
      not a defect.
    * Swin at another size: refused HERE, loudly -- it asserts 224x224
      exactly, and timm's own failure surfaces as a bare assert deep in
      patch embedding. A dynamic-size swin variant would be a different
      model, so a different arm (``roadb.NON_SQUARE_IS_VIT_ONLY``).
    * Anything else (the CNNs): no extra kwargs -- they accept any size,
      which is its own recorded danger for REGION schemes but is fine for
      whole-image embeddings.
    """
    if input_size is None:
        return {}
    height, width = (int(v) for v in input_size)
    # **[2026-09-02] BEFORE the 224 shortcut, not after.** The shortcut
    # below says "224 needs no kwargs", which is true only because every
    # backbone registered before Phase 25 is 224-NATIVE. DINOv2 is
    # 518-native: built with no kwargs it asserts on a 224 input at the
    # first forward pass -- the exact failure this function exists to
    # prevent, one model class later.
    native = NATIVE_INPUT_SIZE.get(timm_name)
    if native is not None and (height, width) == (native, native):
        return {}          # at its OWN native size, exactly as 224 is for the rest
    if native is not None:
        patch = int(timm_name.removeprefix("vit_base_patch").split("_")[0])
        if height % patch or width % patch:
            raise FactoryError(
                f"{timm_name} needs both dimensions divisible by {patch}, "
                f"got {height}x{width}. Its native size is {native}; "
                "staging rounds to multiples of 16, so only multiples of "
                f"{16 * patch // math.gcd(16, patch)} are reachable."
            )
        return {"dynamic_img_size": True}
    if (height, width) == (224, 224):
        return {}
    if timm_name.startswith("vit_base_patch"):
        # One rule for the whole patch family (7D adds patch32 and patch8):
        # both dimensions must divide by the model's OWN patch size, parsed
        # from the name it was registered under rather than duplicated here.
        patch = int(timm_name.removeprefix("vit_base_patch").split("_")[0])
        if height % patch or width % patch:
            raise FactoryError(
                f"{timm_name} needs both dimensions divisible by {patch}, "
                f"got {height}x{width}. Staging rounds to multiples of 16 "
                "(roadb_staging.SHAPE_IS_FIXED_AT_STAGING), so an input "
                "failing this did not come from a staged artifact."
            )
        return {"dynamic_img_size": True}
    if timm_name == "swin_base_patch4_window7_224":
        if height != width:
            raise FactoryError(
                f"swin_b at a NON-SQUARE size ({height}x{width}) is refused: "
                "non-square is ViT-only (roadb.NON_SQUARE_IS_VIT_ONLY), and "
                "the square-build measurement (roadb.SWIN_AT_RESOLUTION) "
                "says nothing about non-square window padding."
            )
        # [CONFIRMED ON THE PINNED IMAGE 2026-08-09] Building at a declared
        # img_size carries the 224 checkpoint's parameters BYTE-EQUAL --
        # timm 1.0.7 gate run, roadb.SWIN_AT_RESOLUTION. The route is
        # img_size, not dynamic_img_size: swin's positional machinery is
        # window-relative and nothing learned is interpolated.
        return {"img_size": height}
    return {}


def create_backbone(
    name: str = DEFAULT_BACKBONE, *, pretrained: bool = True, num_outputs: int = 1,
    input_size=None,
) -> Any:
    """A backbone with a regression head. Accepts a registry name or a timm name.

    ``num_outputs=1`` and MSE on the raw 1-5 scale (PLAN §4.6). Not CORAL/CORN --
    the tails are too sparse for an ordinal decomposition.

    Registry names route by ``kind``: ``transformer`` goes to timm, ``graph`` to
    the module that implements it. A raw timm name still works, so Phase 3's
    ``vit_base_patch16_224`` call sites are unchanged.

    ``input_size`` is the staged input's ``(height, width)`` when it is not
    the 224 default -- ``timm_kwargs_for`` routes it, and omitting it leaves
    the historical path untouched.
    """
    if name in BACKBONES:
        spec = backbone_spec(name)
        if spec["kind"] == "graph":
            return _create_graph_backbone(name, pretrained, num_outputs)
        name = spec["timm_name"]

    extra = timm_kwargs_for(name, input_size)

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - cluster-only path
        raise FactoryError(
            f"timm is required to build {name!r} and is not installed. It is in "
            "the pinned image; a laptop run should use the stub backbone."
        ) from exc

    try:
        return timm.create_model(
            name, pretrained=pretrained, num_classes=num_outputs, **extra
        )
    except Exception as exc:  # pragma: no cover - cluster-only path
        raise FactoryError(f"timm could not create {name!r}: {exc}") from exc


def _create_graph_backbone(name: str, pretrained: bool, num_outputs: int) -> Any:
    """SR-GNN or AG-Net. Imported here so the package imports without torch."""
    if name == "srgnn":  # pragma: no cover - cluster-only path
        from .srgnn import build

        return build(pretrained=pretrained, num_outputs=num_outputs)
    if name == "agnet":  # pragma: no cover - cluster-only path
        from .agnet import build

        return build(pretrained=pretrained, num_outputs=num_outputs)
    raise FactoryError(f"no graph implementation registered for {name!r}")


#: Attribute a non-timm backbone stamps with its own preprocessing.
#:
#: **[MEASURED 2026-07-31] The hole this closes.** timm models carry
#: ``pretrained_cfg``; **torchvision models carry nothing** -- no
#: ``pretrained_cfg``, no ``default_cfg``, and none on the ``children()[:8]``
#: body AG-Net actually builds. So for one of the four backbones
#: ``normalization_for`` had no source to read, and the two ways out of that are
#: to refuse (breaking AG-Net) or to substitute a default (**the Stage-1 defect
#: itself**).
#:
#: The third way is to make torchvision's own provenance readable: its weights
#: enum carries ``transforms()`` with the mean and std those weights were trained
#: under. ``stamp_normalization`` copies that onto the module, so
#: ``normalization_for`` is still READING the model's own preprocessing rather
#: than being told what to assume.
NORMALIZATION_ATTRIBUTE = "cleft_normalization"


def stamp_normalization(model: Any, mean, std, source: str) -> Any:
    """Attach preprocessing a model cannot report itself. Returns the model.

    **Only for weights whose own metadata supplies the values** -- torchvision's
    ``ResNet50_Weights.X.transforms()``, for instance. It is not a place to put a
    triple somebody believes is right: the value must come from the weights, or
    the caller is guessing and has reintroduced the defect with an extra step.
    ``source`` records where it came from, and travels into the run record.
    """
    if not source:
        raise FactoryError(
            "stamped normalization must name its source. An unattributed triple "
            "is a hardcoded constant with a function call in front of it."
        )
    setattr(
        model,
        NORMALIZATION_ATTRIBUTE,
        {
            "mean": tuple(float(v) for v in mean),
            "std": tuple(float(v) for v in std),
            "source": source,
        },
    )
    return model


def normalization_for(model: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) from the model's OWN preprocessing. Read, never chosen.

    Two sources, both belonging to the model rather than to this function:

    * timm's ``pretrained_cfg`` / ``default_cfg``;
    * a ``cleft_normalization`` stamp, for models that report nothing --
      torchvision's, whose values come from their weights enum's
      ``transforms()``.

    **It refuses if neither exists, and that refusal is the point.** A backbone
    trained with 0.5/0.5 given ImageNet statistics is silently wrong; given to
    one that *was* trained on them it is silently right, which is exactly why the
    defect survives review. **[MEASURED] The four Phase 6 backbones split two-two
    on this** -- ViT-B/16 and Xception at 0.5/0.5, Swin-B and ResNet-50 on
    ImageNet -- so there is no single convention to fall back on even if falling
    back were acceptable.
    """
    stamped = getattr(model, NORMALIZATION_ATTRIBUTE, None)
    if stamped:
        return tuple(stamped["mean"]), tuple(stamped["std"])

    config = getattr(model, "pretrained_cfg", None) or getattr(
        model, "default_cfg", None
    )
    if not config:
        raise FactoryError(
            f"{type(model).__name__} reports no preprocessing: no "
            f"pretrained_cfg, no default_cfg, and no {NORMALIZATION_ATTRIBUTE} "
            "stamp.\n"
            "  DO NOT SUBSTITUTE A DEFAULT. That is the Stage-1 defect this "
            "function exists to prevent, and the four Phase 6 backbones split "
            "two-two between 0.5/0.5 and ImageNet statistics, so no default is "
            "even majority-right.\n"
            "  torchvision models report nothing: build them through a wrapper "
            "that calls stamp_normalization() with the values from their weights "
            "enum's transforms(), so the numbers still come from the weights."
        )

    mean, std = config.get("mean"), config.get("std")
    if mean is None or std is None:
        raise FactoryError(
            f"pretrained_cfg exists but is missing mean/std: {sorted(config)}. "
            "A partially populated config is not a licence to fill in the rest."
        )
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def normalization_source(model: Any) -> str:
    """Where a model's preprocessing came from. Recorded with every run.

    A run whose normalization source is unrecorded cannot be checked afterwards,
    and this is the value most likely to be silently wrong.
    """
    stamped = getattr(model, NORMALIZATION_ATTRIBUTE, None)
    if stamped:
        return stamped["source"]
    if getattr(model, "pretrained_cfg", None):
        return "timm pretrained_cfg"
    if getattr(model, "default_cfg", None):
        return "timm default_cfg"
    raise FactoryError(
        f"{type(model).__name__} has no normalization source; "
        "normalization_for would refuse it too"
    )


#: Which parameters an arm trains. [LITERATURE] For small target datasets,
#: partial or layer-wise fine-tuning is favoured over full fine-tuning, and 237
#: images against ViT-B/16's ~86M parameters is the textbook over-parameterised
#: case: full fine-tuning at any ordinary learning rate destroys the pretrained
#: representation before it can be used.
#:
#: [MEASURED] It did. The first Phase 3 run trained every parameter and produced
#: PCC -0.013, QWK exactly 0.0 (every prediction in one 3-class bin), and RMSE
#: 0.658 against a label SD of 0.628 -- worse than a constant predictor -- with
#: inner-val best at epoch 1 in three folds of five. Training was damaging the
#: model, not improving it.
TRAINABLE_POLICIES = ("head", "full")


class FrozenBackboneError(FactoryError):
    """The backbone could not be frozen."""


def freeze_backbone(model: Any) -> dict:
    """Freeze everything except the classifier head. Returns what was frozen."""
    head = model.get_classifier()
    head_parameters = {id(p) for p in head.parameters()}

    frozen = trainable = 0
    for parameter in model.parameters():
        if id(parameter) in head_parameters:
            parameter.requires_grad_(True)
            trainable += parameter.numel()
        else:
            parameter.requires_grad_(False)
            frozen += parameter.numel()

    if trainable == 0:
        raise FrozenBackboneError(
            "freezing left no trainable parameters; the classifier head was not "
            "found, so nothing would learn"
        )
    return {"trainable_parameters": trainable, "frozen_parameters": frozen}


def count_parameters(model: Any) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": round(trainable / total, 6) if total else 0.0,
    }


def input_size_for(model: Any) -> tuple[int, int, int]:
    """(C, H, W) the model expects, from its own config."""
    config = getattr(model, "pretrained_cfg", None) or getattr(
        model, "default_cfg", None
    )
    if not config or "input_size" not in config:
        raise FactoryError("the model exposes no input_size in its pretrained_cfg")
    return tuple(int(v) for v in config["input_size"])
