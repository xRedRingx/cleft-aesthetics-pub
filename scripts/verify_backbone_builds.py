"""The graph builds, driven through the REAL path: construct, forward, resume.

**History.** The first version of this script carried the SR-GNN and AG-Net
constructions inline and measured them against the specs -- both hit exactly on
first construction (32,896,562 / 30,742,924 at 200 classes, timm 1.0.27). On
2026-07-31 those constructions moved into ``srgnn.build()`` and
``agnet.build()`` with the Stage 1 forwards; this script now drives the real
builds, so it re-derives the construction evidence through the shipped code
path instead of a copy that could drift from it.

What it checks, per graph backbone:

1. ``build()`` constructs; its own per-component parameter assertions fire
   internally, and the totals are printed against the spec.
2. ``normalization_for`` reads the stamped preprocessing (SR-GNN from the timm
   wrapper's pretrained_cfg, AG-Net from the torchvision weights enum).
3. **The first forward pass either model has ever run**: finite logits of the
   right shape, deterministic in eval mode, for the default-region path and an
   explicit-boxes call.
4. The pretraining integration: ``TorchPretrainModel`` reset (full fine-tuning
   asserted, epoch-0 head at the train mean), one optimizer step, and the
   checkpoint round trip -- restore is bitwise, and one further step from the
   restored torch RNG stays bitwise. SR-GNN's dropout consumes the RNG stream,
   so it exercises the same restore path Swin-B's stochastic depth does.

**Devices, and one documented relaxation.** SR-GNN runs on CPU, where every op
is deterministic and every comparison is bitwise. AG-Net prefers CUDA: its
``roi_align`` CUDA backward uses atomicAdd and is nondeterministic run-to-run,
so the post-restore STEP is compared with a tolerance that atomic noise passes
(~1e-7) and a lost optimizer slot does not (~1e-3) -- the check keeps its
teeth. Restore itself (forward only) stays bitwise everywhere.

**The environment can be the blocker, and the script says so distinctly.**
``torchvision::roi_align`` is PROBED before AG-Net's forward checks:
[MEASURED 2026-07-31] this laptop's torchvision wheel (py313/cu126) is
miswired -- the kernel registered under the CUDA dispatch key is the CPU
implementation (it demands CPU tensors), and no CPU-key kernel exists, so the
op fails in BOTH directions. That is a wheel defect, not a port defect, and
nothing model-side is bent around it. When the probe fails, AG-Net's forward
checks are reported as ENVIRONMENT-BLOCKED and the script exits 2 (against 1
for a real failure): construction and normalization are still verified here,
and the forward half runs in the pinned image, whose torchvision is a
coherent build.

``pretrained=False`` throughout: mechanics, not weights -- no downloads. The
twelve runs use pretrained weights; nothing here changes that path.

Run OUTSIDE the project venv (system python with torch), or in the image::

    python scripts/verify_backbone_builds.py

Exit code 0 only if every check passes.
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cleft.models import agnet, srgnn  # noqa: E402
from cleft.models.factory import normalization_for, normalization_source  # noqa: E402
from cleft.train import checkpoint as ckpt  # noqa: E402
from cleft.train.pretrain import TorchPretrainModel  # noqa: E402

failures: list[str] = []
blocked: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def roi_align_works(device: str) -> bool:
    """Probe the op AG-Net's forward stands on. See the module docstring for
    the miswired-wheel failure this exists to distinguish from a code defect."""
    from torchvision.ops import roi_align

    try:
        feat = torch.zeros(1, 1, 4, 4, device=device)
        rois = torch.tensor([[0.0, 0.0, 0.0, 4.0, 4.0]], device=device)
        roi_align(feat, rois, 2)
        return True
    except (RuntimeError, NotImplementedError) as exc:
        print(f"  [--] torchvision::roi_align unusable on {device}: {exc}")
        return False


def blob_images(n: int, seed: int) -> np.ndarray:
    """Structured uint8 images: a few bright blobs on a dark field, so SIFT
    finds SOME keypoints without noise-level thousands (which would make the
    CPU inter-attention pass needlessly heavy)."""
    rng = np.random.default_rng(seed)
    images = np.full((n, 224, 224, 3), 30, dtype=np.uint8)
    for image in images:
        for _ in range(6):
            cy, cx = rng.integers(30, 194, size=2)
            r = int(rng.integers(8, 20))
            y, x = np.ogrid[:224, :224]
            image[(y - cy) ** 2 + (x - cx) ** 2 <= r * r] = rng.integers(150, 255)
    return images


def smoke_forward(name: str, module, device: str) -> None:
    print(f"{name}: build + forward [{device}]")
    spec = module.parameter_count(num_outputs=200)
    model = module.build(pretrained=False, num_outputs=200)
    total = sum(p.numel() for p in model.parameters())
    check(
        "construction hits the spec (assertions fired inside build)",
        total == spec["total"],
        f"{total:,}",
    )
    mean, std = normalization_for(model)
    check(
        "normalization read, not assumed",
        len(mean) == 3 and len(std) == 3,
        f"{normalization_source(model)}: mean {mean}",
    )

    model.eval().to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        first = model(x)
        second = model(x)
    check(
        "first forward: finite logits, right shape",
        first.shape == (2, 200) and bool(torch.isfinite(first).all()),
        f"shape {tuple(first.shape)}",
    )
    check(
        "eval forward is deterministic",
        bool(torch.equal(first, second)),
    )

    # The explicit-boxes path (the cleft route) alongside the default regions.
    if name == "srgnn":
        boxes = torch.as_tensor(srgnn.grid_rois(), device=device)
        with torch.no_grad():
            explicit = model(x, boxes)
        check(
            "explicit shared boxes reproduce the default-region path",
            bool(torch.equal(first, explicit)),
        )
    else:
        boxes = [
            torch.tensor([[0.1, 0.1, 0.6, 0.6], [0.3, 0.3, 0.9, 0.9]], device=device),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device),
        ]
        with torch.no_grad():
            explicit = model(x, boxes)
        check(
            "explicit per-image boxes run (variable counts padded)",
            explicit.shape == (2, 200) and bool(torch.isfinite(explicit).all()),
        )

    # The generated region schemes [DECIDED 2026-07-31]: Phase 2's grid
    # patches through the per-face mapping, converted to this backbone's
    # convention -- the exact arrays the pretraining loop hands over.
    from cleft.train.pretrain import boxes_for_model, scheme_frame_boxes

    content = np.tile(np.array([[0, 0, 224, 224]], dtype=np.int64), (2, 1))
    frame = scheme_frame_boxes("grid", "g2", content)
    scheme_boxes = torch.as_tensor(
        boxes_for_model(name, frame), device=device
    )
    with torch.no_grad():
        scheme_out = model(x, scheme_boxes)
    check(
        "generated scheme boxes (grid, 27 regions) run end to end",
        scheme_out.shape == (2, 200)
        and bool(torch.isfinite(scheme_out).all())
        and frame.shape == (2, 27, 4),
    )


def pretrain_integration(name: str, device: str, exact: bool) -> None:
    print(f"{name}: pretraining integration [{device}]")
    features = blob_images(4, seed=20260731)
    labels = np.random.default_rng(7).uniform(1.5, 4.5, size=4)

    def fresh() -> TorchPretrainModel:
        model = TorchPretrainModel(
            name=name, pretrained=False, batch_size=2, seed=7, device=device
        )
        model.reset(features, labels)
        return model

    model = fresh()
    report = model.parameter_report
    check(
        "full fine-tuning: trainable == total",
        report["trainable_parameters"] == report["total_parameters"],
        f"{report['trainable_parameters']:,}",
    )
    epoch0 = model.predict(features)
    check(
        "epoch-0 head predicts the train mean",
        bool(np.allclose(epoch0, float(np.mean(labels)), atol=1e-4)),
        f"pred mean {float(np.mean(epoch0)):.6f} vs {float(np.mean(labels)):.6f}",
    )
    loss = model.train_batch(features[:2], labels[:2])
    check("an optimizer step runs", bool(np.isfinite(loss)), f"loss {loss:.4f}")

    arrays = model.state_arrays(include_optimizer=True)
    torch_rng, cuda_rng = ckpt.capture_torch_rng()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "state.npz"
        ckpt.save(
            path,
            ckpt.Checkpoint(
                step=1, epoch=1, arrays=arrays,
                numpy_rng=ckpt.capture_numpy_rng(np.random.default_rng(7)),
                torch_rng=torch_rng, torch_cuda_rng=cuda_rng,
            ),
        )
        loaded = ckpt.load(path)

    resumed = fresh()
    resumed.load_state_arrays(loaded.arrays)
    check(
        "restored model predicts bitwise-identically",
        bool(np.array_equal(model.predict(features), resumed.predict(features))),
    )

    ckpt.restore_torch_rng(loaded.torch_rng, loaded.torch_cuda_rng)
    loss_original = model.train_batch(features[2:], labels[2:])
    after_original = model.predict(features)
    ckpt.restore_torch_rng(loaded.torch_rng, loaded.torch_cuda_rng)
    loss_resumed = resumed.train_batch(features[2:], labels[2:])
    after_resumed = resumed.predict(features)
    if exact:
        check(
            "one post-restore step stays bitwise identical",
            bool(np.array_equal(after_original, after_resumed))
            and loss_original == loss_resumed,
            f"losses {loss_original:.6f} vs {loss_resumed:.6f}",
        )
    else:
        # roi_align's CUDA backward uses atomicAdd: run-to-run noise ~1e-7
        # after one AdamW step, while a LOST optimizer slot moves predictions
        # by ~1e-3. The tolerance sits between the two, so the check keeps
        # its teeth without asserting a determinism the kernel does not offer.
        drift = float(np.max(np.abs(after_original - after_resumed)))
        check(
            "one post-restore step agrees within atomic-noise tolerance",
            bool(np.allclose(after_original, after_resumed, rtol=1e-4, atol=1e-4))
            and bool(np.isclose(loss_original, loss_resumed, rtol=1e-4)),
            f"max |diff| {drift:.3g}, losses {loss_original:.6f} vs {loss_resumed:.6f}",
        )


def graph_cleft_integration(name: str, module, device: str) -> None:
    """The third regime's torch half: the forward split is bitwise, the warm
    start loads, the classifier re-initialises, and only graph layers train."""
    import torch

    from cleft.train.graph_cleft import GraphHeadBackbone, pack
    from cleft.train.pretrain import MODEL_PREFIX

    print(f"{name}: graph cleft integration [{device}]")
    model = module.build(pretrained=False, num_outputs=1)
    model.eval().to(device)

    x = torch.randn(2, 3, 224, 224, device=device)
    if name == "srgnn":
        feature_map = model.backbone(x)[-1]
        boxes = None
        with torch.no_grad():
            full = model(x)
            composed = model.forward_from_features(feature_map)
    else:
        feature_map = model.backbone(x)
        boxes = [
            torch.tensor([[0.1, 0.1, 0.8, 0.8]], device=device),
            torch.tensor([[0.0, 0.0, 1.0, 1.0]], device=device),
        ]
        with torch.no_grad():
            full = model(x, boxes)
            composed = model.forward_from_features(feature_map, boxes)
    check(
        "forward == backbone + forward_from_features, bitwise",
        bool(torch.equal(full, composed)),
    )

    # A "pretraining checkpoint" from the model itself: distinctive values so
    # the warm start is observable.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.001)
    checkpoint_arrays = {
        MODEL_PREFIX + key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }
    reference = checkpoint_arrays[MODEL_PREFIX + "gnn_mlp1.weight"] if (
        name == "srgnn"
    ) else checkpoint_arrays[MODEL_PREFIX + "fusion.w.weight"]

    maps = feature_map.detach().cpu().numpy()
    map_shape = maps.shape[1:]
    if name == "srgnn":
        packed = pack(maps, None)
        n_regions = 0
    else:
        fixed = np.tile(
            np.array([[0.1, 0.1, 0.8, 0.8], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
            (2, 1, 1),
        )
        packed = pack(maps, fixed)
        n_regions = 2

    labels = np.array([2.0, 4.0])
    head = GraphHeadBackbone(
        name=name, map_shape=tuple(map_shape), n_regions=n_regions,
        checkpoint_arrays=checkpoint_arrays, batch_size=2, seed=7, device=device,
    )
    head.reset(labels)

    spec_head = module.parameter_count(num_outputs=1)["head_total"]
    check(
        "trainable == the spec's graph-layer total",
        head.parameter_report["trainable_parameters"] == spec_head,
        f"{head.parameter_report['trainable_parameters']:,}",
    )
    warm = (
        head._model.gnn_mlp1.weight if name == "srgnn" else head._model.fusion.w.weight
    )
    check(
        "graph layers warm-started from the checkpoint",
        bool(np.array_equal(warm.detach().cpu().numpy(), reference)),
    )
    check(
        "classifier re-initialised at the train mean",
        float(head._model.classifier.weight.abs().max()) == 0.0
        and float(head._model.classifier.bias.item()) == 3.0,
    )
    epoch0 = head.predict(packed)
    check(
        "epoch-0 predicts the train mean",
        bool(np.allclose(epoch0, 3.0, atol=1e-4)),
        f"pred mean {float(np.mean(epoch0)):.6f}",
    )
    loss = head.train_epoch(packed, labels)
    check("a graph-layer epoch runs", bool(np.isfinite(loss)), f"loss {loss:.4f}")
    after = head.predict(packed)
    check(
        "training moved the predictions",
        not np.allclose(after, epoch0),
    )


def frozen_graph_diagnostic(name: str, module, device: str) -> None:
    """The classifier-only policy: is the "frozen" stack ACTUALLY frozen?

    ``requires_grad_(False)`` is not the whole of it, and that is the point of
    this check. SR-GNN carries two BatchNorms whose running statistics are
    **buffers, not parameters** -- no gradient flows through them and nothing
    about requires_grad stops train mode from updating them on every batch. It
    also applies ``Dropout(0.2)`` twice. So a policy that froze only the
    parameters would still emit a representation that moved under the head
    fitting on it, and the arm would not be the linear probe its comparison
    against 0.2529 depends on it being.

    Parameters AND buffers are therefore compared bitwise across a training
    epoch. Only the classifier may move.
    """
    import torch

    from cleft.train.graph_cleft import GraphHeadBackbone, pack

    print(f"{name}: frozen-graph diagnostic [{device}]")
    model = module.build(pretrained=False, num_outputs=1)
    model.eval().to(device)

    x = torch.randn(4, 3, 224, 224, device=device)
    if name == "srgnn":
        with torch.no_grad():
            feature_map = model.backbone(x)[-1]
        boxes, n_regions = None, 0
    else:
        with torch.no_grad():
            feature_map = model.backbone(x)
        boxes = np.tile(
            np.array([[0.1, 0.1, 0.8, 0.8]], dtype=np.float32), (4, 1, 1)
        )
        n_regions = 1

    maps = feature_map.detach().cpu().numpy()
    packed = pack(maps, boxes)
    labels = np.array([2.0, 4.0, 3.0, 5.0])

    head = GraphHeadBackbone(
        name=name, map_shape=tuple(maps.shape[1:]), n_regions=n_regions,
        checkpoint_arrays=None, trainable="classifier",
        learning_rate=1e-3, batch_size=2, seed=7, device=device,
    )
    head.reset(labels)

    spec = module.parameter_count(num_outputs=1)
    report = head.parameter_report
    check(
        "trainable == the classifier alone",
        report["trainable_parameters"] == spec["classifier"],
        f"{report['trainable_parameters']:,} (spec {spec['classifier']:,})",
    )
    check(
        "every other parameter is frozen",
        report["frozen_backbone_parameters"] == spec["total"] - spec["classifier"],
        f"{report['frozen_backbone_parameters']:,}",
    )
    check(
        "the run records this as a diagnostic, not a ladder arm",
        "diagnostic_note" in report
        and report["policy"].endswith("trained_classifier"),
        report["policy"],
    )

    epoch0 = head.predict(packed)
    check(
        "epoch-0 predicts the train mean (gate 3)",
        bool(np.allclose(epoch0, float(np.mean(labels)), atol=1e-4)),
        f"pred mean {float(np.mean(epoch0)):.6f}",
    )

    # The snapshot, parameters AND buffers -- the BN running statistics are
    # buffers, which is exactly the state requires_grad does not protect.
    def snapshot() -> dict:
        return {
            key: value.detach().cpu().clone()
            for key, value in list(head._model.named_parameters())
            + list(head._model.named_buffers())
        }

    before = snapshot()
    loss = head.train_epoch(packed, labels)
    after = snapshot()
    check("a classifier-only epoch runs", bool(np.isfinite(loss)), f"loss {loss:.4f}")

    moved = sorted(k for k in before if not torch.equal(before[k], after[k]))
    check(
        "ONLY the classifier moved -- graph layers and BN buffers bitwise equal",
        moved == ["classifier.bias", "classifier.weight"],
        f"moved: {moved}" if moved else "nothing moved",
    )
    bn_buffers = [
        k for k in before if "running_" in k or k.endswith("num_batches_tracked")
    ]
    check(
        "BN buffers exist and are among what stayed put (the check has teeth)",
        bool(bn_buffers) and all(k not in moved for k in bn_buffers),
        f"{len(bn_buffers)} BN buffers held",
    )

    after_predictions = head.predict(packed)
    check(
        "the head learned (predictions moved)",
        not np.allclose(after_predictions, epoch0),
    )

    # The representation is a FIXED function: two passes over the same rows
    # must agree bitwise, which dropout in train mode would break.
    check(
        "the frozen representation is deterministic across passes",
        bool(np.array_equal(head.predict(packed), after_predictions)),
    )


def adabn_diagnostic(name: str, module, device: str) -> None:
    """AdaBN: did the statistics actually MOVE, and did nothing else?

    Three ways this is a no-op that reports success, each checked here against
    the shipped code path rather than in isolation (``MECHANISM_VERIFIED``
    holds the isolated measurements):

    * the running statistics must genuinely change -- without
      ``reset_running_stats()`` the pretrained ``num_batches_tracked`` makes
      the cumulative update negligible and the arm keeps SCUT's statistics;
    * the BN *parameters* (weight/bias) and every other weight must NOT move
      -- only the buffers may, plus the classifier once training starts;
    * the adapted representation must be a fixed function afterwards, or the
      arm is not a probe.
    """
    import torch

    from cleft.train.graph_cleft import GraphHeadBackbone, pack

    print(f"{name}: adabn diagnostic [{device}]")
    model = module.build(pretrained=False, num_outputs=1)
    model.eval().to(device)

    # Cross-check the RECORDED BatchNorm census against this built model. The
    # record drives a real conclusion -- that adabn is SR-GNN-only and that no
    # statistics-mismatch caveat attaches to AG-Net results -- so it must be
    # re-derived rather than remembered. The suite asserts the record's
    # internal consistency; this asserts it against an nn.Module.
    from cleft.train.graph_cleft import MECHANISM_VERIFIED

    census = MECHANISM_VERIFIED["post_backbone_batchnorms"][name]
    found = [
        key
        for key, mod in model.named_modules()
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)
    ]
    after_boundary = [key for key in found if not key.startswith("backbone.")]
    check(
        "the recorded BatchNorm census matches the built model",
        len(found) == census["total"]
        and len(after_boundary) == census["after_frozen_boundary"],
        f"{len(found)} total, {len(after_boundary)} after the frozen boundary "
        f"(recorded {census['total']}/{census['after_frozen_boundary']})",
    )

    x = torch.randn(6, 3, 224, 224, device=device)
    if name == "srgnn":
        with torch.no_grad():
            feature_map = model.backbone(x)[-1]
        boxes, n_regions = None, 0
    else:
        with torch.no_grad():
            feature_map = model.backbone(x)
        boxes = np.tile(
            np.array([[0.1, 0.1, 0.8, 0.8]], dtype=np.float32), (6, 1, 1)
        )
        n_regions = 1

    maps = feature_map.detach().cpu().numpy()
    packed = pack(maps, boxes)
    labels = np.array([2.0, 4.0, 3.0, 5.0, 2.5, 3.5])

    # A checkpoint carrying DISTINCTIVE running statistics and a large
    # num_batches_tracked -- the state a real pretraining checkpoint arrives
    # in, and the one that makes a missing reset invisible.
    # WARM-STARTED, like the real arm: the checkpoint's weights are what make
    # the adapted statistics seed-independent, and an imagenet init would not
    # show that property because its graph layers are seed-initialised.
    from cleft.train.pretrain import MODEL_PREFIX

    checkpoint_arrays = {
        MODEL_PREFIX + key: value.detach().cpu().numpy().copy()
        for key, value in model.state_dict().items()
    }

    def fresh(seed: int) -> GraphHeadBackbone:
        instance = GraphHeadBackbone(
            name=name, map_shape=tuple(maps.shape[1:]), n_regions=n_regions,
            checkpoint_arrays=checkpoint_arrays, trainable="classifier_adabn",
            learning_rate=1e-3, batch_size=2, seed=seed, device=device,
        )
        instance.reset(labels)
        return instance

    head = fresh(7)
    bns = head._batch_norms()

    # **[MEASURED 2026-08-01] AG-Net has NO post-backbone BatchNorm at all** --
    # all 53 of its BN modules sit inside the frozen ResNet-50, which never
    # runs on this path. So the policy has nothing to re-estimate there and
    # must REFUSE rather than run an expensive no-op. That is the whole check
    # for this backbone, and it is a finding rather than a limitation: the
    # normalisation-statistics alternative that motivates this arm cannot
    # arise for AG-Net, because it has no normalisation after the frozen
    # boundary.
    if not bns:
        from cleft.train.graph_cleft import GraphCleftError

        try:
            head.adapt_batchnorm(packed)
            check(f"{name}: adabn refuses a backbone with no BN to adapt", False)
        except GraphCleftError as exc:
            check(
                f"{name} has no post-backbone BatchNorm, and the policy REFUSES",
                "no BatchNorm modules" in str(exc),
                "all its BN lives in the frozen backbone, which never runs here",
            )
        return

    check(
        "adapts the post-backbone BatchNorms only",
        len(bns) == sum(
            1
            for n, m in head._model.named_modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
            and not n.startswith("backbone.")
        )
        and bool(bns),
        f"{len(bns)} adapted of "
        f"{sum(1 for _ in head._model.modules() if isinstance(_, torch.nn.modules.batchnorm._BatchNorm))} "
        "total (the backbone's never run: features are precomputed)",
    )
    with torch.no_grad():
        for bn in bns:
            bn.running_mean.fill_(-99.0)
            bn.running_var.fill_(7.0)
            bn.num_batches_tracked.fill_(3000)

    before = {
        key: value.detach().cpu().clone()
        for key, value in list(head._model.named_parameters())
        + list(head._model.named_buffers())
    }

    # **The representation, not the predictions.** The classifier is
    # zero-initialised, so predict() returns the bias whatever arrives -- a
    # before/after comparison through it is constant BY CONSTRUCTION and
    # would report "no change" even if adaptation were working perfectly.
    # (It did: this check failed that way on first run.) A forward pre-hook
    # on the classifier reads what actually reaches it.
    captured: list = []

    def capture(_module, args):
        captured.append(args[0].detach().cpu().clone())

    handle = head._model.classifier.register_forward_pre_hook(capture)
    head.predict(packed)
    pre_adapt = torch.cat(captured)
    captured.clear()

    record = head.adapt_batchnorm(packed)
    check(
        "adaptation ran over the rows it was given",
        record["n_rows"] == 6 and sum(record["batch_sizes"]) == 6,
        f"{record['n_batches']} batches {record['batch_sizes']}",
    )
    check(
        "num_batches_tracked was RESET (not continued from 3000)",
        all(int(bn.num_batches_tracked) == record["n_batches"] for bn in bns),
        f"{[int(b.num_batches_tracked) for b in bns[:2]]} vs 3000 before",
    )
    check(
        "the running statistics actually moved off the checkpoint's",
        all(
            abs(float(bn.running_mean.abs().max()) - 99.0) > 1.0 for bn in bns
        ),
        f"max |running_mean| now {max(float(b.running_mean.abs().max()) for b in bns):.4f}",
    )

    after = {
        key: value.detach().cpu().clone()
        for key, value in list(head._model.named_parameters())
        + list(head._model.named_buffers())
    }
    moved = sorted(k for k in before if not torch.equal(before[k], after[k]))
    only_buffers = all(
        "running_" in k or k.endswith("num_batches_tracked") for k in moved
    )
    check(
        "ONLY BN buffers moved -- no weight, not even BN's own affine params",
        bool(moved) and only_buffers,
        f"{len(moved)} buffers moved, 0 parameters",
    )

    # Dropout must have stayed off throughout, or the statistics describe
    # activations inference never sees.
    check(
        "every module is back in eval mode after adaptation",
        not any(m.training for m in head._model.modules()),
    )
    # The adaptation pass runs forwards through the classifier too, so the
    # hook has been accumulating during it. Clear immediately before each
    # measured call rather than after.
    captured.clear()
    head.predict(packed)
    post_adapt = torch.cat(captured)
    check(
        "adaptation CHANGED the representation (it was not inert)",
        not torch.allclose(pre_adapt, post_adapt),
        f"max |delta| into the classifier "
        f"{float((post_adapt - pre_adapt).abs().max()):.6g}",
    )
    captured.clear()
    head.predict(packed)
    again = torch.cat(captured)
    handle.remove()
    check(
        "the adapted representation is a fixed function (two passes equal)",
        bool(torch.equal(again, post_adapt)),
    )

    # And the arm still trains only the classifier from here.
    settled = {
        key: value.detach().cpu().clone()
        for key, value in list(head._model.named_parameters())
        + list(head._model.named_buffers())
    }
    loss = head.train_epoch(packed, labels)
    final = {
        key: value.detach().cpu().clone()
        for key, value in list(head._model.named_parameters())
        + list(head._model.named_buffers())
    }
    moved_in_epoch = sorted(k for k in settled if not torch.equal(settled[k], final[k]))
    check("an adabn epoch runs", bool(np.isfinite(loss)), f"loss {loss:.4f}")
    check(
        "after adaptation ONLY the classifier moves -- BN buffers now frozen",
        moved_in_epoch == ["classifier.bias", "classifier.weight"],
        f"moved: {moved_in_epoch}",
    )

    # Idempotent: the second epoch must not re-adapt (it would recompute the
    # same values, but a changing _adapted record would mean it had).
    stats_before = [bn.running_mean.detach().cpu().clone() for bn in bns]
    head.train_epoch(packed, labels)
    check(
        "adaptation happens ONCE per fold, not every epoch",
        all(
            torch.equal(a, bn.running_mean.detach().cpu())
            for a, bn in zip(stats_before, bns)
        ),
    )

    # Seed independence, and it is CONDITIONAL on the warm start. The pass is
    # unshuffled and gradient-free, so the statistics depend only on the fold
    # and on the weights producing the activations -- and those are identical
    # across seeds only because the checkpoint supplies them. Both instances
    # here are warm-started from the same arrays, which is the real arm's
    # configuration.
    seven, nine = fresh(7), fresh(999)
    record_a = seven.adapt_batchnorm(packed)
    nine.adapt_batchnorm(packed)
    check(
        "warm-started: the adapted statistics do not depend on the seed",
        all(
            torch.equal(a.running_mean.cpu(), b.running_mean.cpu())
            and torch.equal(a.running_var.cpu(), b.running_var.cpu())
            for a, b in zip(seven._batch_norms(), nine._batch_norms())
        )
        and record_a["seed_independent"] is True,
    )

    # And the record must NOT claim it where it does not hold: an imagenet
    # init seed-initialises the graph layers, so the activations -- and these
    # statistics -- do vary with the seed. Asserted rather than assumed,
    # because claiming the convenient case is how a caveat gets lost.
    cold = GraphHeadBackbone(
        name=name, map_shape=tuple(maps.shape[1:]), n_regions=n_regions,
        checkpoint_arrays=None, trainable="classifier_adabn",
        learning_rate=1e-3, batch_size=2, seed=7, device=device,
    )
    cold.reset(labels)
    cold_record = cold.adapt_batchnorm(packed)
    other_cold = GraphHeadBackbone(
        name=name, map_shape=tuple(maps.shape[1:]), n_regions=n_regions,
        checkpoint_arrays=None, trainable="classifier_adabn",
        learning_rate=1e-3, batch_size=2, seed=999, device=device,
    )
    other_cold.reset(labels)
    other_cold.adapt_batchnorm(packed)
    differ = any(
        not torch.equal(a.running_mean.cpu(), b.running_mean.cpu())
        for a, b in zip(cold._batch_norms(), other_cold._batch_norms())
    )
    check(
        "imagenet init: the record does NOT claim seed independence, and the "
        "statistics really do differ",
        cold_record["seed_independent"] is False and differ,
    )


def per_fold_reextraction(name: str, module, device: str) -> None:
    """The EXTRACTION-side adaptation: the 40 layers the cleft arm cannot reach.

    This is the half that the AdaBN cleft result made necessary -- touching 2
    of 42 BatchNorms moved the result 0.087, so the 40 fixed at extraction are
    load-bearing. Here they are re-estimated per fold, and the checks are the
    same shape as the cleft arm's: the statistics must genuinely move off the
    checkpoint's, only buffers may change, and different adapted rows must
    give different features -- otherwise five per-fold sets would be five
    copies and every leak assertion downstream would be vacuous.
    """
    import torch

    from cleft.train.extract import adapt_backbone_batchnorm, backbone_batch_norms

    print(f"{name}: per-fold BN re-extraction [{device}]")
    model = module.build(pretrained=False, num_outputs=1)
    model.eval().to(device)

    bns = backbone_batch_norms(model)
    check(
        "the backbone's own BatchNorms are the ones re-estimated here",
        len(bns) > 0,
        f"{len(bns)} in the backbone (the cleft arm reaches the other "
        f"{sum(1 for _ in model.modules() if isinstance(_, torch.nn.modules.batchnorm._BatchNorm)) - len(bns)})",
    )
    with torch.no_grad():
        for bn in bns:
            bn.running_mean.fill_(-99.0)
            bn.num_batches_tracked.fill_(3000)

    before = {
        key: value.detach().cpu().clone()
        for key, value in list(model.named_parameters())
        + list(model.named_buffers())
    }

    images = torch.randn(8, 3, 224, 224, device=device)
    record = adapt_backbone_batchnorm(
        model, (images[i : i + 4] for i in range(0, 8, 4)), batch_size=4
    )
    check(
        "adaptation ran and reset the counter",
        record["n_batches"] == 2
        and all(int(bn.num_batches_tracked) == 2 for bn in bns),
        f"{record['n_batchnorm_modules']} modules, {record['n_batches']} batches",
    )
    check(
        "the statistics moved off the checkpoint's",
        all(abs(float(bn.running_mean.abs().max()) - 99.0) > 1.0 for bn in bns),
    )

    after = {
        key: value.detach().cpu().clone()
        for key, value in list(model.named_parameters())
        + list(model.named_buffers())
    }
    moved = [k for k in before if not torch.equal(before[k], after[k])]
    check(
        "ONLY buffers moved -- no weight, not even BN's affine parameters",
        bool(moved)
        and all(
            "running_" in k or k.endswith("num_batches_tracked") for k in moved
        ),
        f"{len(moved)} buffers, 0 parameters",
    )
    check("every module back in eval mode", not any(m.training for m in model.modules()))

    # **Different adapted rows must give different features.** If they did
    # not, five per-fold sets would be five copies of one and every leak
    # assertion downstream would pass on data that never varied.
    def features_after(rows) -> torch.Tensor:
        # **Seed the build.** `pretrained=False` draws fresh random weights on
        # every call, so two "identical" models would differ and the
        # determinism check below would fail for a reason that has nothing to
        # do with adaptation. The real extraction always loads a checkpoint,
        # which fixes the weights; this reproduces that condition.
        torch.manual_seed(20260801)
        fresh = module.build(pretrained=False, num_outputs=1)
        fresh.eval().to(device)
        adapt_backbone_batchnorm(
            fresh, (images[rows][i : i + 4] for i in range(0, len(rows), 4)),
            batch_size=4,
        )
        with torch.no_grad():
            raw = fresh.backbone(images)
        return (raw[-1] if isinstance(raw, (list, tuple)) else raw).detach().cpu()

    fold_a = features_after([0, 1, 2, 3])
    fold_b = features_after([4, 5, 6, 7])
    check(
        "different adapted rows give DIFFERENT features (per-fold is not a copy)",
        not torch.allclose(fold_a, fold_b),
        f"max |delta| {float((fold_a - fold_b).abs().max()):.6g}",
    )
    check(
        "the same adapted rows reproduce exactly (deterministic)",
        bool(torch.equal(features_after([0, 1, 2, 3]), fold_a)),
    )


def ldl_head(device: str) -> None:
    """The label-distribution head: gate 3, the KL fit, and the leak.

    The torch-free half (targets, the gate-3 bias, packing, the isolation
    check) is in tests/test_ldl.py. This runs the fit loop, which is what
    needs torch, and re-runs the isolation check against the REAL head rather
    than a stand-in -- that is the one that matters, since the stand-ins in
    the suite are written to pass or fail by construction.
    """
    from cleft.train import ldl
    from cleft.train.ldl import LDLHeadBackbone

    print(f"ldl: head + KL loss [{device}]")
    rng = np.random.default_rng(20260801)
    n, dim = 40, 12

    # A learnable signal: the distribution's centre tracks a feature.
    embeddings = rng.normal(size=(n, dim))
    centre = np.clip(3.0 + embeddings[:, 0], 1.0, 5.0)
    targets = np.zeros((n, 5))
    for row, value in enumerate(centre):
        low = int(np.floor(value)) - 1
        frac = value - np.floor(value)
        targets[row, min(low, 4)] += 1.0 - frac
        targets[row, min(low + 1, 4)] += frac
    targets /= targets.sum(axis=1, keepdims=True)
    labels = ldl.expectation(targets)
    packed = ldl.pack_targets(embeddings, targets)

    head = LDLHeadBackbone(embedding_dim=dim, learning_rate=1e-2, seed=7)
    head.reset(labels)

    report = head.parameter_report
    check(
        "the head is 5 outputs over the embedding, and counts them",
        report["trainable_parameters"] == dim * 5 + 5,
        f"{report['trainable_parameters']} = {dim}x5 + 5",
    )

    # **Gate 3.** The bias is the log of the training marginal, so an
    # untrained head predicts the training mean rather than the uniform
    # distribution's 3.0. Verified against this fold's own marginal.
    bias = ldl.log_marginal_bias(targets)
    shifted = np.exp(bias - bias.max())
    marginal = shifted / shifted.sum()
    check(
        "the gate-3 bias predicts the training mean, not 3.0",
        abs(float(ldl.expectation(marginal[None, :])[0]) - float(labels.mean()))
        < 1e-6,
        f"marginal E={float(ldl.expectation(marginal[None, :])[0]):.6f} vs "
        f"train mean {float(labels.mean()):.6f}",
    )

    before = head.predict(packed)
    loss_first = head.train_epoch(packed, labels)
    loss_second = head.train_epoch(packed, labels)
    check(
        "the KL loss is finite and decreasing",
        bool(np.isfinite(loss_first)) and loss_second < loss_first,
        f"{loss_first:.6f} -> {loss_second:.6f}",
    )

    after = head.predict(packed)
    check(
        "predictions stay on the 1-5 scale",
        bool(np.all(after >= 1.0) and np.all(after <= 5.0)),
        f"range [{after.min():.3f}, {after.max():.3f}]",
    )
    check("training moved the predictions", not np.allclose(before, after))

    from cleft.eval import metrics

    check(
        "the head learns the signal it was given",
        metrics.pcc(labels, after) > 0.8,
        f"pcc {metrics.pcc(labels, after):.4f}",
    )

    # **THE LEAK, against the real head.** The label rides in the feature row;
    # a head reading those columns would predict the label from the label.
    try:
        ldl.assert_head_ignores_targets(head, packed)
        check("the trained head ignores the target columns", True)
    except ldl.LDLError as exc:
        check("the trained head ignores the target columns", False, str(exc)[:90])

    # And no gradient path exists from the targets either -- a head could
    # ignore them at predict time while still fitting through them.
    import torch

    check(
        "the weight matrix spans the embedding only, not the packed row",
        tuple(head._weights.shape) == (dim, 5),
        f"{tuple(head._weights.shape)}",
    )
    check(
        "no NaN reached the parameters",
        bool(torch.isfinite(head._weights).all() and torch.isfinite(head._bias).all()),
    )


def main() -> int:
    # Deterministic conv algorithm selection wherever cudnn is involved; the
    # remaining CUDA nondeterminism (roi_align backward) is handled by the
    # tolerance above.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # SR-GNN verifies on CPU, where every comparison is bitwise. AG-Net needs
    # a device whose torchvision carries roi_align -- this laptop's wheel has
    # no CPU kernel for it, so CUDA when available; without either, AG-Net's
    # forward is NOT verifiable in this environment and the script must say
    # so rather than print a pass that checked nothing.
    agnet_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"timm/torch: torch {torch.__version__}; "
        f"specs SR-GNN {srgnn.parameter_count(num_outputs=200)['total']:,} / "
        f"AG-Net {agnet.parameter_count(num_outputs=200)['total']:,} at 200 classes"
    )
    smoke_forward("srgnn", srgnn, "cpu")
    pretrain_integration("srgnn", "cpu", exact=True)
    graph_cleft_integration("srgnn", srgnn, "cpu")
    frozen_graph_diagnostic("srgnn", srgnn, "cpu")
    adabn_diagnostic("srgnn", srgnn, "cpu")
    per_fold_reextraction("srgnn", srgnn, "cpu")
    ldl_head("cpu")

    if roi_align_works(agnet_device):
        smoke_forward("agnet", agnet, agnet_device)
        pretrain_integration("agnet", agnet_device, exact=agnet_device == "cpu")
        graph_cleft_integration("agnet", agnet, agnet_device)
        frozen_graph_diagnostic("agnet", agnet, agnet_device)
        adabn_diagnostic("agnet", agnet, agnet_device)
    else:
        # Construction and the stamp do not need the op; verify what can be.
        print(f"agnet: build only [{agnet_device}] -- forward ENVIRONMENT-BLOCKED")
        spec = agnet.parameter_count(num_outputs=200)
        model = agnet.build(pretrained=False, num_outputs=200)
        total = sum(p.numel() for p in model.parameters())
        check("construction hits the spec", total == spec["total"], f"{total:,}")
        mean, _ = normalization_for(model)
        check(
            "normalization read, not assumed",
            len(mean) == 3,
            normalization_source(model),
        )
        blocked.append(
            "agnet forward/integration: torchvision::roi_align is unusable in "
            "this environment (miswired wheel); run this script in the pinned "
            "image to verify the forward half"
        )

    print()
    if failures:
        print(f"NOT VERIFIED: {len(failures)} check(s) failed: {failures}")
        return 1
    if blocked:
        print("VERIFIED WHERE POSSIBLE; ENVIRONMENT-BLOCKED elsewhere:")
        for item in blocked:
            print(f"  - {item}")
        return 2
    print("ALL CHECKS PASSED: both graph builds construct, forward, and resume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
