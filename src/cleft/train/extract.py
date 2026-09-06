"""Embedding extraction: each backbone's frozen boundary, over the 237.

What one extraction produces is decided by the backbone kind
(``embeddings.KIND_FOR_BACKBONE_KIND``):

* transformers -- the pooled pre-logits vector, ``(n, feature_dim)``. The
  whole model froze at cleft time, so this is everything downstream needs.
* graph models -- the backbone's final feature map, ``(n, channels, 7, 7)``.
  Everything after the map trains on cleft data (SR-GNN's self-attention
  consumes the FLATTENED per-region descriptor; AG-Net's SAGAN precedes
  regions), so the map is the frozen boundary and the only thing worth
  precomputing. Region schemes are applied at cleft-train time.

The checkpoint arrives as a ``pretrained.npz`` in the verified checkpoint
format; loading asserts the model's state-dict keys match exactly, same as a
resume would, so a checkpoint from a different architecture refuses rather
than partially loads. ``init: imagenet`` loads no checkpoint -- the factory's
pretrained weights ARE that init.

Torch never appears at module level. The stub backbones ("stub" pooled,
"stub_graph" feature-map) exercise the task end to end on the laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..models.factory import BACKBONES
from . import checkpoint as ckpt
from .pretrain import MODEL_PREFIX, PretrainError

#: Test-only backbones, mapped onto the two artifact kinds so the whole task
#: -- extraction, saving, row-order assertion -- runs without torch.
STUB_KINDS = {"stub": "transformer", "stub_graph": "graph"}


#: **[DECIDED 2026-08-01] Per-fold BN re-extraction: why it exists, and the
#: one thing it must get right.**
#:
#: **Why.** The frozen SR-GNN probe scored 0.1340 against a ViT linear probe's
#: 0.2529, and the AdaBN cleft arm -- re-estimating the 2 post-backbone
#: BatchNorms -- came back at **0.0468**, worse than doing nothing, outside
#: both pre-registered readings. The mechanism is partial adaptation: two
#: layers calibrated for cleft receiving activations from 40 layers still
#: calibrated for SCUT. **That 0.087 swing is the measurement that makes this
#: necessary**: it proves the 40 extraction-time BatchNorms are load-bearing,
#: so SR-GNN's representation cannot be judged while they are calibrated for
#: the wrong domain -- and ViT carries no equivalent handicap, LayerNorm
#: having no running statistics.
#:
#: **THE LEAK, AND WHY THE ARTIFACT IS FIVE SETS.** Re-estimating BatchNorm on
#: cleft data makes the features depend on which patients were adapted on. Do
#: it once over all 237 and every feature vector carries statistics computed
#: partly from the test fold -- an invisible leak that inflates every arm and
#: nothing downstream would catch. So adaptation is **per fold**, on that
#: fold's TRAINING patients only, and the artifact is five sets rather than
#: one.
#:
#: **Adapted on the OUTER training fold, not the inner.** The inner-val split
#: seeds off ``config.seed`` (``harness.inner_val_split``), so inner-train
#: rows would make the artifact seed-dependent -- five sets per seed, fifty
#: for a ten-seed band, and a different representation for every arm. The
#: outer fold is seed-independent, which is what lets one artifact serve every
#: seed and every arm. The cost is stated rather than hidden: **inner-val
#: patients contribute to the statistics**, so the early-stopping signal is
#: mildly optimistic. It is not a leak into the reported metric, because the
#: test fold is excluded and the test fold is what the OOF number is built
#: from.
#:
#: **The exclusion is asserted, not arranged.** Each set records the patient
#: ids it adapted on, and the consuming arm re-checks that the fold's test
#: patients are absent from that list (``graph_cleft``). A leak here would
#: produce a plausible, better number with every shape correct, so it gets a
#: runtime check rather than a careful construction.
PER_FOLD_REEXTRACTION = {
    "decided": "2026-08-01",
    "forced_by": {
        "plain_probe_pcc": 0.1340,
        "adabn_cleft_pcc": 0.0468,
        "swing": -0.087,
        "reading": (
            "touching 2 of 42 BatchNorms moved the result 0.087, so the 40 "
            "baked into the artifact are load-bearing rather than idle history"
        ),
    },
    "n_sets": 5,
    "adapted_on": "the OUTER training fold (all patients not in the test fold)",
    "why_not_inner": (
        "inner_val_split seeds off config.seed, so inner-train rows would make "
        "the artifact seed-dependent -- 50 sets for a ten-seed band, and a "
        "different representation per arm"
    ),
    "stated_cost": (
        "inner-val patients contribute to the statistics, so the "
        "early-stopping signal is mildly optimistic. NOT a leak into the "
        "reported metric: the test fold is excluded, and the OOF number is "
        "built from test folds only"
    ),
    "leak_check": (
        "each set records adapted_on_patient_ids; the consuming arm asserts "
        "the fold's test patients are absent. A leak would produce a "
        "plausible, better number with every shape correct"
    ),
    "procedure": (
        "reset_running_stats(); BN modules alone to train mode with "
        "momentum=None; forward the fold's training rows under no_grad; "
        "return to eval -- the same procedure verified for the cleft arm"
    ),
    #: **[MEASURED 2026-08-01] The split, and what it means for each backbone.**
    #: SR-GNN: 40 BatchNorms in the backbone (fixed here), 2 after it
    #: (reachable at cleft time). AG-Net: 53 in the backbone, **0** after it.
    #:
    #: So for AG-Net this re-extraction is not one of two options, it is the
    #: ONLY route -- ``trainable: classifier_adabn`` refuses there, having
    #: nothing to adapt. Any AG-Net probe result read before this runs carries
    #: the whole normalisation mismatch with it, uncorrectable at cleft time.
    "reachable_split": {
        "srgnn": {"fixed_here": 40, "reachable_at_cleft_time": 2},
        "agnet": {"fixed_here": 53, "reachable_at_cleft_time": 0},
        "consequence": (
            "for AG-Net this is the only route: the cleft-time adabn policy "
            "refuses there, so an AG-Net probe read before this runs carries "
            "the entire mismatch and cannot be corrected afterwards"
        ),
    },
}


class ExtractError(RuntimeError):
    """The extraction cannot proceed."""


def backbone_kind_for(backbone: str) -> str:
    if backbone in STUB_KINDS:
        return STUB_KINDS[backbone]
    if backbone in BACKBONES:
        return BACKBONES[backbone]["kind"]
    raise ExtractError(f"unknown backbone {backbone!r}")


def _stub_features(
    backbone: str,
    images: np.ndarray,
    adapt_rows: np.ndarray | None = None,
    *,
    block: int | None = None,
    token: str | None = None,
) -> np.ndarray:
    """Deterministic, content-dependent, tiny. Enough for row-order teeth.

    **``adapt_rows`` must genuinely change the output.** The stub has no
    BatchNorm, so it cannot re-estimate anything -- but if the per-fold sets
    came out identical, every laptop test of the per-fold path would be
    exercising five copies of one array, and the leak check would pass on data
    that never varied. That is failure mode 5 in PLAN R7's tally, built in on
    purpose. So the stub subtracts the ADAPTED ROWS' mean, which is exactly
    the dependence BatchNorm re-estimation has: a shift determined by which
    patients were adapted on.

    **``block``/``token`` must change it for the identical reason.** Twelve
    pooling sets that came out byte-identical would make every laptop test of
    that axis pass on one array wearing twelve names, and the search would
    rank twelve copies of one representation. So the stub scales by depth and
    offsets by the pooling rule -- the dependence a real block extraction has,
    in miniature.
    """
    array = np.asarray(images, dtype=np.float32)
    if adapt_rows is not None:
        rows = np.asarray(adapt_rows, dtype=int)
        array = array - array[rows].mean(axis=0, keepdims=True)
    if block is not None:
        array = array * (1.0 + int(block) / 100.0)
        if token == "mean_patch":
            array = array + 1.0
    if backbone == "stub":
        return array.reshape(array.shape[0], -1, array.shape[-1]).mean(axis=1)
    half_h = max(array.shape[1] // 2, 1)
    half_w = max(array.shape[2] // 2, 1)
    quads = np.stack(
        [
            array[:, :half_h, :half_w].mean(axis=(1, 2)),
            array[:, :half_h, half_w:].mean(axis=(1, 2)),
            array[:, half_h:, :half_w].mean(axis=(1, 2)),
            array[:, half_h:, half_w:].mean(axis=(1, 2)),
        ],
        axis=2,
    )  # (n, channels, 4)
    return quads.reshape(array.shape[0], array.shape[-1], 2, 2)


#: resnet50's parameter-carrying children, in the order AG-Net's Sequential
#: body wraps them -- conv1 is index 0, bn1 index 1 (relu and maxpool carry
#: nothing), layer1..layer4 are 4..7. Measured against the built model:
#: 318 mapped keys = resnet50's 320 minus the fc pair.
_AGNET_BODY_INDEX = {
    "conv1": "0", "bn1": "1",
    "layer1": "4", "layer2": "5", "layer3": "6", "layer4": "7",
}


from ..embeddings import FOUNDATION_INITS as _FOUNDATION_INITS

#: **Inits whose weights ARE the factory's pretrained ones**, loading no
#: checkpoint. ``imagenet`` has always meant this; **Phase 25 adds two that
#: mean it about DIFFERENT pretraining** -- DINOv2 on LVD-142M and original
#: DINO on ImageNet-1k without labels. They are named apart from
#: ``imagenet`` because the set directory carries the init in its name, and
#: three provenances under one name is the R2 shape this project refuses.
#:
#: **They are deliberately NOT declared snapshots.** The probe they are
#: compared against (``init: imagenet``) takes its weights the same way, so
#: declaring these and not the probe would make the comparison asymmetric
#: in provenance -- on a one-factor contrast, that is the factor moving
#: twice (``phase25.THE_WEIGHTS_ARE_FACTORY_PRETRAINED_LIKE_THE_PROBES``).
#: **DERIVED, not repeated** [2026-09-02]: the foundation half comes from
#: ``embeddings.FOUNDATION_INITS``, which is the vocabulary's one home. A
#: second literal list is a list that drifts the day a fourth init lands.
FACTORY_PRETRAINED_INITS = ("imagenet",) + _FOUNDATION_INITS


def _load_graph_imagenet_init(model, backbone: str, init_dir: Path) -> None:
    """Load a graph backbone's declared imagenet snapshot into its body.

    STRICT both ways, like ``load_declared_init``: every body key must be
    fed, and every snapshot key must land somewhere or be an expected head
    key -- a silent partial load would be the undeclared input back in a
    quieter form.
    """
    import torch

    from .pretrain import INIT_WEIGHTS_NAME

    path = Path(init_dir) / INIT_WEIGHTS_NAME
    if not path.is_file():
        raise ExtractError(
            f"the declared init artifact has no {INIT_WEIGHTS_NAME}: {path}"
        )
    arrays = np.load(path)
    snapshot = {k: torch.as_tensor(np.asarray(arrays[k])) for k in arrays.files}

    if backbone == "srgnn":
        # [MEASURED] the features_only body's keys are 'body.' + snapshot
        # key, exactly, 274 <-> 274.
        mapped = {f"body.{key}": value for key, value in snapshot.items()}
        leftovers: list = []
        target = model.backbone
    elif backbone == "agnet":
        mapped, leftovers = {}, []
        for key, value in snapshot.items():
            head, _, rest = key.partition(".")
            index = _AGNET_BODY_INDEX.get(head)
            if index is None:
                leftovers.append(key)
                continue
            mapped[f"{index}.{rest}"] = value
        expected_leftovers = {"fc.weight", "fc.bias"}
        if set(leftovers) != expected_leftovers:
            raise ExtractError(
                f"the agnet snapshot's unmapped keys are {sorted(leftovers)[:4]}, "
                f"expected exactly {sorted(expected_leftovers)} -- the "
                "snapshot is not the resnet50 this mapping was measured on"
            )
        target = model.backbone
    else:
        raise ExtractError(
            f"no graph init mapping for {backbone!r}; the measured mappings "
            "cover srgnn and agnet"
        )

    expected = set(target.state_dict())
    if set(mapped) != expected:
        missing = sorted(expected - set(mapped))[:4]
        surplus = sorted(set(mapped) - expected)[:4]
        raise ExtractError(
            f"the {backbone} snapshot does not cover its body: missing "
            f"{missing}, surplus {surplus}. Refusing a partial load."
        )
    target.load_state_dict(mapped)


def _load_model(
    backbone: str,
    init: str,
    checkpoint_path: Path | None,
    input_size=None,
    init_dir: Path | None = None,
):
    """The model whose frozen boundary is being read. Torch only.

    **[CORRECTED 2026-08-11, two holes in one signature.]** This built
    without ``input_size`` -- a 512/768 extraction would have died at the
    ViT 224 assert AFTER submission, the same gate-2 class the pretrain
    builder already had -- and its imagenet branch was the undeclared
    download. ``init_dir`` is the declared snapshot
    (``pretrain.INIT_INPUT_NAME``'s artifact): with it, imagenet sets build
    without downloading and load guard-3-verified bytes with strict
    accounting; without it, the download runs and the report says so. The
    graph backbones' loading route is the standing follow-up, so a graph
    declaration is REFUSED here -- the ``task_pretrain`` rule.
    """
    from ..models.factory import BACKBONES as _BACKBONES
    from ..models.factory import create_backbone

    if init_dir is not None and init not in FACTORY_PRETRAINED_INITS:
        raise ExtractError(
            f"init {init!r} takes its weights wholly from the checkpoint; "
            "a declared pretrained_init would be verified and recorded "
            "without being read."
        )

    if init in FACTORY_PRETRAINED_INITS:
        if checkpoint_path is not None:
            raise ExtractError(
                f"init {init!r} takes no checkpoint -- its weights ARE the "
                "factory's pretrained ones; a checkpoint was supplied, and "
                "silently ignoring it would misdescribe the set's provenance"
            )
        if init_dir is not None:
            model = create_backbone(
                backbone, pretrained=False, num_outputs=1,
                input_size=input_size,
            )
            if _BACKBONES.get(backbone, {}).get("kind") == "graph":
                # [MEASURED 2026-08-11] The graph bodies map onto their
                # snapshots deterministically -- srgnn's features_only body
                # is 'body.'+key exactly (274<->274), agnet's Sequential
                # body is the resnet50 child order (318 = 320 minus fc) --
                # so the EXTRACTION route consumes declared inits for the
                # graphs too, and the vintage rule holds for all 12
                # imagenet sets. Training-side consumption stays the
                # recorded follow-up: task_pretrain still refuses.
                _load_graph_imagenet_init(model, backbone, init_dir)
            else:
                from .pretrain import load_declared_init

                load_declared_init(model, init_dir)
            return model
        return create_backbone(
            backbone, pretrained=True, num_outputs=1, input_size=input_size
        )

    if checkpoint_path is None:
        raise ExtractError(f"init {init!r} needs a pretrained checkpoint")
    import torch

    model = create_backbone(
        backbone, pretrained=False, num_outputs=1, input_size=input_size
    )
    state = ckpt.load(checkpoint_path)
    loaded = {
        key[len(MODEL_PREFIX):]: torch.as_tensor(np.asarray(value))
        for key, value in state.arrays.items()
        if key.startswith(MODEL_PREFIX)
    }
    expected = set(model.state_dict())
    if set(loaded) != expected:
        missing = sorted(expected - set(loaded))[:3]
        surplus = sorted(set(loaded) - expected)[:3]
        raise ExtractError(
            f"checkpoint {checkpoint_path} does not match the {backbone!r} "
            f"architecture: missing {missing}, surplus {surplus}. Extracting "
            "from a partially loaded model would produce plausible vectors "
            "from weights nobody trained."
        )
    model.load_state_dict(loaded)
    return model


#: Arm B's component-role control, and why it has to happen HERE.
#:
#: ``phase8.COMPONENT_ROLE_IS_A_NO_OP``: arm A randomises its frozen backbone
#: and RE-EXTRACTS its map on the same pass, because it runs the backbone
#: live. Arm B's map arrives precomputed, so ``forward_from_features`` enters
#: below the backbone and randomising those parameters at cleft time changes
#: nothing -- measured, bitwise. The only place the same intervention exists
#: for arm B is extraction, which is where the representation is actually
#: computed. That makes it a new artifact with its own hash, exactly as the
#: per-fold BN question needed one (``ADABN_DIAGNOSTIC.does_not_test``).
#:
#: **The endpoint only, not a curve.** The adopted criterion reads the FINAL
#: stage -- everything randomised -- so one artifact answers it. A cumulative
#: curve would be one extraction per stage, and arm A's own curve is
#: (measured) not cumulative past its first patient anyway
#: (``phase8.ARM_A_RANDOMISATION_RESTARTS_DIRTY``).
#:
#: **Parameters, not buffers.** Arm A's randomisation touches
#: ``module.parameters()``; BatchNorm running statistics are buffers and stay
#: as they are. Touching them would be a second difference between the arms,
#: and the comparison exists to have exactly one.
RANDOMISED_BACKBONE = {
    "why_here": (
        "arm A randomises a backbone it runs live; arm B's map is "
        "precomputed, so the same intervention only exists at extraction"
    ),
    "scope": "every parameter of model.backbone, deepest child first",
    "endpoint_only": (
        "the criterion reads the final stage, so one artifact answers it"
    ),
    "buffers_untouched": (
        "parameters only, as arm A's does -- BN running statistics are "
        "buffers, and touching them would be a second difference between "
        "the arms"
    ),
    "seed": 20260814,
}


def randomise_backbone(model, seed: int = RANDOMISED_BACKBONE["seed"]) -> dict:
    """Replace every backbone parameter with noise matched to its spread.

    The extraction-time half of arm B's component-role control
    (``RANDOMISED_BACKBONE``). Deepest child first, so the label matches arm
    A's top-down convention even though only the endpoint is kept.

    ``node_weights.noise_std`` is the shared rule -- one definition for this
    and for the cleft-time parameter walk, so the two randomisations differ
    in what they touch and in nothing else.

    Returns what it did, for the artifact's metadata: a randomisation nobody
    can confirm ran is indistinguishable from a set that was never
    randomised, and those two produce opposite conclusions.
    """
    import torch

    from ..node_weights import noise_std

    body = getattr(model, "backbone", None)
    if body is None:
        raise ExtractError(
            f"{type(model).__name__} has no .backbone to randomise. The "
            "control randomises the frozen representation, and a model "
            "whose backbone is not addressable would be left untouched "
            "while the artifact claimed otherwise"
        )
    groups = list(reversed(list(body.named_children())))
    if not groups:
        raise ExtractError(
            "the backbone has no child modules, so nothing would be "
            "randomised and the control would be a copy of the real set"
        )

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    touched, borrowed, names = 0, 0, []
    for name, module in groups:
        parameters = list(module.parameters())
        if not parameters:
            continue
        with torch.no_grad():
            flat = torch.cat([p.detach().reshape(-1) for p in parameters])
            group_std = float(flat.std(unbiased=False))
            for parameter in parameters:
                if float(parameter.detach().std(unbiased=False)) == 0.0:
                    borrowed += 1
                noise = torch.empty(
                    parameter.shape, dtype=parameter.dtype, device="cpu"
                )
                noise.normal_(
                    0.0, noise_std(parameter, group_std), generator=generator
                )
                parameter.copy_(noise.to(parameter.device))
                touched += parameter.numel()
        names.append(name)
    if not touched:
        raise ExtractError(
            "the backbone has child modules but no parameters, so the "
            "control would be a copy of the real set"
        )
    return {
        "randomised": True,
        "seed": int(seed),
        "n_parameters": int(touched),
        "n_groups": len(names),
        "groups": names,
        "n_borrowed_group_std": borrowed,
        "buffers_untouched": RANDOMISED_BACKBONE["buffers_untouched"],
        "purpose": (
            "phase8 arm B's component-role control -- the same intervention "
            "arm A applies live, moved to where arm B's representation is "
            "computed (phase8.COMPONENT_ROLE_IS_A_NO_OP)"
        ),
    }


def backbone_batch_norms(model) -> list:
    """The BatchNorm modules the EXTRACTION runs -- the backbone's own.

    The exact complement of ``graph_cleft.GraphHeadBackbone._batch_norms``,
    which takes the post-backbone ones. Between them the two cover the whole
    chain, and the split is the artifact boundary: these 40 (SR-GNN) or 53
    (AG-Net) are fixed here, permanently, into the saved map.
    """
    import torch

    body = getattr(model, "backbone", model)
    return [
        module
        for module in body.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    ]


def adapt_backbone_batchnorm(
    model, batches, *, batch_size: int
) -> dict:
    """Re-estimate the backbone's BN running statistics on ``batches``.

    ``batches`` yields already-normalised, already-device-placed input
    tensors -- the caller owns preprocessing, so this function cannot
    accidentally use a different normalisation than the extraction pass that
    follows it.

    The procedure is the one verified for the cleft arm
    (``graph_cleft.MECHANISM_VERIFIED``), and each step is load-bearing:

    * **reset first.** ``momentum=None`` is a CUMULATIVE average weighted
      ``1/num_batches_tracked``, and the checkpoint arrives with that counter
      in the thousands -- measured, four batches then move a statistic 0.14%
      of the way. Without the reset this is a no-op with a flag on it.
    * **BN modules alone into train mode.** ``model.train()`` would switch on
      every dropout and stochastic-depth layer in the backbone, so the
      statistics would describe activations the extraction pass never sees.
    * **no gradients.** Nothing trains; only buffers move.

    Returns what it did, for the artifact's metadata.
    """
    import torch

    modules = backbone_batch_norms(model)
    if not modules:
        raise ExtractError(
            "the backbone has no BatchNorm modules, so there are no running "
            "statistics to re-estimate. Per-fold re-extraction would produce "
            "five identical sets at five times the cost."
        )

    previous = [module.momentum for module in modules]
    for module in modules:
        module.reset_running_stats()
        module.momentum = None

    model.eval()
    for module in modules:
        module.train()

    n_batches = 0
    with torch.no_grad():
        for batch in batches:
            if batch.shape[0] < 2:
                raise ExtractError(
                    f"a BN re-estimation batch has {batch.shape[0]} row(s); "
                    "batch statistics are undefined on one sample and torch "
                    "raises. Adjust batch_size so no batch is a singleton."
                )
            model.backbone(batch) if hasattr(model, "backbone") else model(batch)
            n_batches += 1

    model.eval()
    for module, momentum in zip(modules, previous):
        module.momentum = momentum

    return {
        "n_batchnorm_modules": len(modules),
        "n_batches": n_batches,
        "tracked": [int(m.num_batches_tracked) for m in modules[:3]],
        "procedure": "reset_running_stats, BN-only train, momentum=None, no_grad",
    }


def extract_features(
    backbone: str,
    init: str,
    images: np.ndarray,
    *,
    checkpoint_path: Path | None,
    batch_size: int = 32,
    device: str = "cuda",
    adapt_rows: np.ndarray | None = None,
    block: int | None = None,
    token: str | None = None,
    init_dir: Path | None = None,
    #: Arm B's component-role control (``RANDOMISED_BACKBONE``). The set this
    #: produces is a CONTROL and must never reach a training arm --
    #: ``embeddings.check_pairing`` refuses it unless the consumer asks for
    #: one by name.
    randomise: bool = False,
) -> tuple[np.ndarray, dict]:
    """One set's array, plus what produced it. Deterministic: eval mode,
    no_grad, fixed order, no augmentation.

    ``adapt_rows`` switches on per-fold BN re-extraction: the backbone's own
    BatchNorm statistics are re-estimated on exactly those rows before the
    extraction pass, and the rows must be the fold's TRAINING patients. See
    ``PER_FOLD_REEXTRACTION`` -- doing it once over all 237 leaks the test
    fold into every feature vector.

    ``block``/``token`` switch on Phase 7B's pooling axis: the representation
    after an intermediate transformer block rather than the final pooled one.
    See ``torch_backbone.extract_block_embeddings``.
    """
    kind = backbone_kind_for(backbone)

    if (block is None) != (token is None):
        raise ExtractError(
            "block and token are given together or not at all; a depth "
            "without a pooling rule does not name a representation"
        )
    if block is not None:
        if init != "imagenet":
            raise ExtractError(
                f"the pooling axis is defined at the imagenet init and got "
                f"{init!r}. Intermediate-block extraction builds the backbone "
                "from its pretrained weights and loads no checkpoint, so a "
                "pretrained init here would silently read the WRONG MODEL"
            )
        if adapt_rows is not None:
            raise ExtractError(
                "per-fold BN re-estimation and the pooling axis are not "
                "combined: the axis is a ViT, whose normalisation is "
                "LayerNorm, so there are no running statistics to re-estimate"
            )

    if randomise:
        if adapt_rows is not None:
            raise ExtractError(
                "the randomised control and per-fold BN re-estimation are "
                "not combined: re-estimating statistics on cleft data for a "
                "backbone whose weights are noise measures the noise's "
                "statistics, and the set would vary by fold for a reason "
                "unrelated to either question"
            )
        if block is not None:
            raise ExtractError(
                "the randomised control is arm B's, and arm B is a graph "
                "arm reading a feature map; the pooling axis is a ViT's"
            )

    if backbone in STUB_KINDS:
        values = _stub_features(
            backbone, images, adapt_rows, block=block, token=token
        ).astype(np.float32)
        if randomise:
            # **The stub has no parameters to randomise**, so it must change
            # the features some other way or every laptop test of the
            # control would pass on data identical to the real set -- the
            # same trap the per-fold stub already closes by shifting.
            rng = np.random.default_rng(RANDOMISED_BACKBONE["seed"])
            values = rng.standard_normal(values.shape).astype(np.float32)
            report_randomisation = {
                "randomised": True,
                "seed": RANDOMISED_BACKBONE["seed"],
                "n_parameters": 0,
                "stub": (
                    "no parameters to randomise; the stub returns noise so "
                    "the control genuinely differs from the real set"
                ),
            }
        else:
            report_randomisation = None
        report = {
            "backbone": backbone,
            "backbone_kind": kind,
            "normalization_source": "stub (no normalization)",
        }
        if block is not None:
            report.update({"block": int(block), "token": token})
        if adapt_rows is not None:
            # The stub has no BatchNorm, so it cannot re-estimate anything --
            # but it MUST make the features depend on the adapted rows, or the
            # whole per-fold artifact would be five identical sets and every
            # laptop test of the leak check would pass on data that never
            # varied. `_stub_features` shifts by the adapted rows' mean.
            report["bn_reestimation"] = {
                "n_batchnorm_modules": 0,
                "n_rows": int(len(adapt_rows)),
                "stub": (
                    "no BatchNorm; the stub instead shifts by the adapted "
                    "rows' mean so per-fold sets genuinely differ"
                ),
            }
        if report_randomisation is not None:
            report["randomisation"] = report_randomisation
        return values, report

    if block is not None:
        from .torch_backbone import extract_block_embeddings

        values, report = extract_block_embeddings(
            images, block=block, token=token, name=backbone,
            batch_size=batch_size, device=device,
        )
        report["backbone_kind"] = kind
        report["normalization_source"] = "the model's own pretrained_cfg"
        return values.astype(np.float32), report

    import torch

    from ..models.factory import normalization_for, normalization_source

    input_size = tuple(int(v) for v in np.asarray(images).shape[1:3])
    model = _load_model(
        backbone, init, checkpoint_path,
        input_size=input_size, init_dir=init_dir,
    )
    mean, std = normalization_for(model)
    # **Before ``.to(target)`` and before any forward pass.** The noise is
    # drawn on the CPU from one seeded generator, so the control is
    # reproducible whether or not a GPU is present -- and randomising after
    # the extraction pass would produce a real set with a control's label.
    randomisation = randomise_backbone(model) if randomise else None
    target = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval().to(target)
    mean_t = torch.tensor(mean, device=target).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=target).view(1, -1, 1, 1)

    def prepared(rows) -> "torch.Tensor":
        batch = torch.as_tensor(
            np.asarray(images[rows], dtype=np.float32) / 255.0
        )
        return ((batch.permute(0, 3, 1, 2).to(target)) - mean_t) / std_t

    adaptation = None
    if adapt_rows is not None:
        # **Before the extraction pass, and on the fold's TRAINING rows only.**
        # The same preprocessing closure feeds both, so the statistics cannot
        # be estimated under a different normalisation than they are used with.
        rows = np.asarray(adapt_rows, dtype=int)
        if rows.size < 2:
            raise ExtractError(
                f"BN re-estimation needs at least 2 rows; got {rows.size}"
            )
        adaptation = adapt_backbone_batchnorm(
            model,
            (
                prepared(rows[start : start + batch_size])
                for start in range(0, rows.size, batch_size)
            ),
            batch_size=batch_size,
        )
        adaptation["n_rows"] = int(rows.size)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.as_tensor(
                np.asarray(images[start : start + batch_size], dtype=np.float32)
                / 255.0
            )
            batch = batch.permute(0, 3, 1, 2).to(target)
            batch = (batch - mean_t) / std_t

            if kind == "transformer":
                features = model.forward_features(batch)
                pooled = model.forward_head(features, pre_logits=True)
                out.append(pooled.float().cpu().numpy())
            else:
                # The graph frozen boundary: the backbone's final map. For
                # SR-GNN the timm features wrapper returns a list.
                raw = model.backbone(batch)
                feature_map = raw[-1] if isinstance(raw, (list, tuple)) else raw
                out.append(feature_map.float().cpu().numpy())

    values = np.concatenate(out).astype(np.float32)
    report = {
        "backbone": backbone,
        "backbone_kind": kind,
        "normalization_mean": list(mean),
        "normalization_std": list(std),
        "normalization_source": normalization_source(model),
        "input_size": list(input_size),
    }
    # What the imagenet weights ARE, in every set's record -- declared and
    # verified, or a runtime download nothing captured. The absence names
    # itself, the pretrain task's convention.
    if init == "imagenet":
        report["pretrained_init"] = (
            {"declared": True, "artifact": str(init_dir)}
            if init_dir is not None else
            {
                "declared": False,
                "note": (
                    "init downloaded at run time -- an undeclared input; "
                    "roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE"
                ),
            }
        )
    if adaptation is not None:
        report["bn_reestimation"] = adaptation
    if randomisation is not None:
        report["randomisation"] = randomisation
    return values, report


def checkpoint_input_name(entry: dict) -> str:
    """The conventional input name for one set's checkpoint, mirroring the
    pretraining config naming so the correspondence is legible in the run
    record: ``ckpt_<backbone>[_<scheme>]_<variant>``."""
    if entry["variant"] is None:
        raise PretrainError(
            "this set has no variant, so it has no checkpoint input -- "
            "true of imagenet and of the foundation inits alike"
        )
    scheme = entry.get("pretrain_scheme")
    token = "" if not scheme else f"{scheme}_"
    return f"ckpt_{entry['backbone']}_{token}{entry['variant']}"
