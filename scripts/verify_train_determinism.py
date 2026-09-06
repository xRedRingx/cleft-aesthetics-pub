"""Per-backbone training-step determinism, measured on the training device.

**Why this exists [2026-07-31].** AG-Net's post-restore step in the pinned
image measured max |diff| 2.31e-05 where every other restore check in this
project has been byte-identical. The suspected mechanism is atomicAdd in
``roi_align``'s CUDA backward -- but SR-GNN's ROI pooling runs through
bilinear ``F.interpolate``, whose CUDA backward is ALSO atomics-based, so
"SR-GNN should stay bitwise on GPU" is a hypothesis this script tests rather
than assumes. A backbone that is deterministic to ~1e-5 rather than exactly
has a weaker resume guarantee and a same-seed reproducibility the seed band
must be measured under; that is a property to record, not a defect to hide.

**What the determinism flag actually does here, read from torchvision
source.** ``torchvision.ops.roi_align`` (since well before the image's 0.23)
checks ``torch.are_deterministic_algorithms_enabled()`` and, on CUDA, routes
to ``_roi_align`` -- a pure-PyTorch, deterministic implementation. So the
flag neither raises on roi_align nor leaves it nondeterministic: it
SUBSTITUTES. PyTorch's own ``upsample_bilinear2d`` backward has no
deterministic CUDA implementation and RAISES under the flag. Consequences to
verify, not trust: flag-on may make AG-Net fully deterministic (at the python
roi_align's speed), while flag-on CRASHES SR-GNN training. Gate 1's "raised
on nothing" canary therefore covers SR-GNN's op (it would raise) but says
nothing about AG-Net's (it silently switches paths).

Probes, per backbone on its training device (CUDA), plus op-level isolates:

  A. op probes -- interpolate-backward and roi_align-backward under the flag
     (raise / substitute / run) and, flag off, a repeat-backward max |diff|.
  B. repeat-gradient -- identical weights, batch and RNG; two full
     forward+backward passes; bitwise-compare every parameter's gradient.
     Isolates kernel nondeterminism from the restore machinery entirely.
  C. post-restore step -- the resume contract's own comparison
     (verify_backbone_builds pattern): bitwise, or the measured bound.
  D. the same one step under ``use_deterministic_algorithms(True)`` -- runs
     deterministically, or raises naming the op.

On this laptop the flag-OFF AG-Net paths are environment-blocked (the local
torchvision is a +cpu wheel against a cu126 torch -- mismatched install, its
custom ops miswired); the flag-ON path uses the python roi_align and runs
anywhere. The image runs everything.

**[MEASURED 2026-07-31, in the image -- two findings this script's first
version produced, one of them its own defect.]**

* Every flag-ON cell raised "not deterministic because it uses CuBLAS": the
  first version set ``use_deterministic_algorithms(True)`` directly, without
  the ``CUBLAS_WORKSPACE_CONFIG`` export the flag requires -- which the
  frozen ``determinism.configure`` owns as part of the Phase 0 contract. A
  second copy of that setup is the two-``is_absolute_path``-predicates class,
  and this script and pretrain.py were both instances. Flag-ON now goes
  THROUGH ``determinism.configure``.
* The roi_align flag-ON substitution raised ``InvalidCxxCompiler``: it is
  gated on torch.compile -> inductor -> a C++ compiler, and the pinned image
  has none. **The deterministic substitution is unreachable in the image**,
  so flag-ON cannot fix AG-Net there; its bound is recorded instead
  (``models.agnet.TRAINING_DETERMINISM``), and the twelve pretraining configs
  run ``deterministic: false``.

Flag-OFF image measurements: vit_b16 / swin_b / srgnn bitwise (repeat-grad
and post-restore 0.0); agnet post-restore 2.26e-05 (2.31e-05 in the local
corroboration); isolated ops: interpolate backward 5.25e-06, roi_align
backward 1.49e-08 -- both atomics-class, only AG-Net's instantiation lands on
colliding atomics.

Run OUTSIDE the project venv, and in the image::

    python scripts/verify_train_determinism.py

Exit 0: all probes ran, records printed. Exit 2: some probes were
environment-blocked (records printed for the rest). Exit 1: a probe failed in
a way that is not an environment limit.
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
import torchvision  # noqa: E402

from cleft.train import checkpoint as ckpt  # noqa: E402
from cleft.train import determinism  # noqa: E402
from cleft.train.pretrain import TorchPretrainModel  # noqa: E402

BACKBONES = ("vit_b16", "swin_b", "srgnn", "agnet")


def flag_on() -> None:
    """The ONE deterministic setup path -- the frozen module's, which exports
    CUBLAS_WORKSPACE_CONFIG before enabling the flag. Never set the flag
    directly: that is how the first version of this script raised CuBLAS
    errors on all four backbones in the image."""
    determinism.configure(7, require_torch=True)

failures: list[str] = []
blocked: list[str] = []
record: dict[str, dict] = {}


def note(name: str, outcome: str, detail: str = "") -> None:
    print(f"  [{outcome}] {name}" + (f" -- {detail}" if detail else ""))
    if outcome == "FAIL":
        failures.append(name)
    if outcome == "BLOCKED":
        blocked.append(f"{name}: {detail}")


def blob_images(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    images = np.full((n, 224, 224, 3), 30, dtype=np.uint8)
    for image in images:
        for _ in range(6):
            cy, cx = rng.integers(30, 194, size=2)
            r = int(rng.integers(8, 20))
            y, x = np.ogrid[:224, :224]
            image[(y - cy) ** 2 + (x - cx) ** 2 <= r * r] = rng.integers(150, 255)
    return images


# --------------------------------------------------------------------------
# A. op-level probes
# --------------------------------------------------------------------------


def probe_interpolate(device: str) -> None:
    print("op probe: bilinear interpolate backward")
    x = torch.randn(2, 8, 7, 7, device=device, requires_grad=True)

    def one_backward() -> torch.Tensor:
        if x.grad is not None:
            x.grad = None
        y = torch.nn.functional.interpolate(
            x, size=(42, 42), mode="bilinear", align_corners=False
        )
        # A gather-ish consumer so upstream grads collide in the backward.
        (y * torch.randn_like(y)).sum().backward()
        return x.grad.detach().clone()

    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device == "cuda" else None
    first = one_backward()
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    second = one_backward()
    diff = float((first - second).abs().max())
    note(
        f"flag OFF, repeat backward on {device}",
        "ok",
        f"max |grad diff| {diff:.3g} ({'bitwise' if diff == 0 else 'NOT bitwise'})",
    )
    record.setdefault("ops", {})[f"interpolate_backward_{device}_repeat_diff"] = diff

    try:
        flag_on()
        one_backward()
        note(f"flag ON on {device}", "ok", "ran (deterministic implementation exists)")
        record["ops"][f"interpolate_backward_{device}_flag"] = "runs"
    except RuntimeError as exc:
        note(f"flag ON on {device}", "ok", f"RAISES: {str(exc)[:90]}")
        record["ops"][f"interpolate_backward_{device}_flag"] = "raises"
    finally:
        torch.use_deterministic_algorithms(False)


def probe_roi_align(device: str) -> None:
    print("op probe: roi_align backward")
    from torchvision.ops import roi_align

    feat = torch.randn(1, 8, 42, 42, device=device, requires_grad=True)
    rois = torch.tensor(
        [[0.0, 0.0, 0.0, 42.0, 42.0], [0.0, 10.0, 10.0, 40.0, 40.0]], device=device
    )

    def one_backward() -> torch.Tensor:
        if feat.grad is not None:
            feat.grad = None
        pooled = roi_align(feat, rois, 7)
        (pooled * torch.randn_like(pooled)).sum().backward()
        return feat.grad.detach().clone()

    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device == "cuda" else None
    try:
        first = one_backward()
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        second = one_backward()
        diff = float((first - second).abs().max())
        note(
            f"flag OFF, repeat backward on {device}",
            "ok",
            f"max |grad diff| {diff:.3g} ({'bitwise' if diff == 0 else 'NOT bitwise'})",
        )
        record.setdefault("ops", {})[f"roi_align_backward_{device}_repeat_diff"] = diff
    except (RuntimeError, NotImplementedError) as exc:
        note(
            f"flag OFF on {device}", "BLOCKED",
            f"custom op unusable in this environment: {str(exc)[:80]}",
        )

    try:
        flag_on()
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        first = one_backward()
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        second = one_backward()
        diff = float((first - second).abs().max())
        note(
            f"flag ON on {device} (python _roi_align route)",
            "ok" if diff == 0 else "FAIL",
            f"max |grad diff| {diff:.3g}",
        )
        record["ops"][f"roi_align_backward_{device}_flag"] = (
            "substitutes_deterministic" if diff == 0 else f"substitutes_but_diff_{diff:.3g}"
        )
    except (RuntimeError, NotImplementedError) as exc:
        # A GENUINE determinism refusal names the missing implementation; any
        # other error is the environment (e.g. the miswired local wheel, or
        # the python route gated off where torch.compile is unsupported).
        message = str(exc).splitlines()[0]
        genuine = "deterministic implementation" in message
        note(
            f"flag ON on {device}",
            "ok" if genuine else "BLOCKED",
            f"{'RAISES: ' if genuine else ''}{message[:90]}",
        )
        record.setdefault("ops", {})[f"roi_align_backward_{device}_flag"] = (
            "raises" if genuine else "environment-blocked"
        )
    finally:
        torch.use_deterministic_algorithms(False)


# --------------------------------------------------------------------------
# B/C/D. model-level probes
# --------------------------------------------------------------------------


def fresh_model(name: str, device: str, features, labels) -> TorchPretrainModel:
    model = TorchPretrainModel(
        name=name, pretrained=False, batch_size=2, seed=7, device=device
    )
    model.reset(features, labels)
    return model


def repeat_gradient(name: str, device: str, features, labels) -> None:
    """B: two identical forward+backward passes; compare every grad bitwise."""
    model = fresh_model(name, device, features, labels)
    inner = model._model
    x = model._prepare(features[:2])
    target = torch.as_tensor(
        np.asarray(labels[:2], dtype=np.float32), device=x.device
    )

    def grads() -> list[torch.Tensor]:
        inner.zero_grad(set_to_none=True)
        prediction = inner(x).squeeze(-1)
        torch.nn.functional.mse_loss(prediction, target).backward()
        return [
            p.grad.detach().clone()
            for p in inner.parameters()
            if p.grad is not None
        ]

    inner.train()
    state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if device == "cuda" else None
    first = grads()
    torch.random.set_rng_state(state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    second = grads()

    worst = 0.0
    for a, b in zip(first, second):
        worst = max(worst, float((a - b).abs().max()))
    note(
        f"{name}: repeat-gradient on {device}",
        "ok",
        f"max |grad diff| {worst:.3g} ({'bitwise' if worst == 0 else 'NOT bitwise'})",
    )
    record.setdefault(name, {})[f"repeat_grad_{device}"] = worst


def post_restore_step(name: str, device: str, features, labels) -> None:
    """C: the resume contract's comparison, on the training device."""
    model = fresh_model(name, device, features, labels)
    model.train_batch(features[:2], labels[:2])

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

    resumed = fresh_model(name, device, features, labels)
    resumed.load_state_arrays(loaded.arrays)
    restore_diff = float(
        np.max(np.abs(model.predict(features) - resumed.predict(features)))
    )

    ckpt.restore_torch_rng(loaded.torch_rng, loaded.torch_cuda_rng)
    model.train_batch(features[2:], labels[2:])
    after_original = model.predict(features)
    ckpt.restore_torch_rng(loaded.torch_rng, loaded.torch_cuda_rng)
    resumed.train_batch(features[2:], labels[2:])
    after_resumed = resumed.predict(features)
    step_diff = float(np.max(np.abs(after_original - after_resumed)))

    note(
        f"{name}: restore on {device}",
        "ok" if restore_diff == 0 else "FAIL",
        f"max |diff| {restore_diff:.3g}",
    )
    note(
        f"{name}: post-restore step on {device}",
        "ok",
        f"max |diff| {step_diff:.3g} ({'bitwise' if step_diff == 0 else 'NOT bitwise'})",
    )
    record.setdefault(name, {})[f"post_restore_step_{device}"] = step_diff


def step_under_flag(name: str, device: str, features, labels) -> None:
    """D: can this backbone train under use_deterministic_algorithms(True) --
    and when it can, is the step BITWISE-repeatable there? That second half is
    the number that makes flag-on a real uniform mode for the twelve runs."""
    try:
        flag_on()
        model = fresh_model(name, device, features, labels)
        inner = model._model
        x = model._prepare(features[:2])
        target = torch.as_tensor(
            np.asarray(labels[:2], dtype=np.float32), device=x.device
        )

        def grads() -> list[torch.Tensor]:
            inner.zero_grad(set_to_none=True)
            prediction = inner(x).squeeze(-1)
            torch.nn.functional.mse_loss(prediction, target).backward()
            return [
                p.grad.detach().clone()
                for p in inner.parameters()
                if p.grad is not None
            ]

        inner.train()
        state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if device == "cuda" else None
        first = grads()
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        second = grads()
        worst = 0.0
        for a, b in zip(first, second):
            worst = max(worst, float((a - b).abs().max()))
        note(
            f"{name}: repeat-gradient UNDER the flag on {device}",
            "ok" if worst == 0 else "FAIL",
            f"max |grad diff| {worst:.3g} "
            f"({'bitwise' if worst == 0 else 'NOT bitwise under the flag'})",
        )
        record.setdefault(name, {})[f"flag_step_{device}"] = (
            "runs_bitwise" if worst == 0 else f"runs_but_diff_{worst:.3g}"
        )
    except (RuntimeError, NotImplementedError) as exc:
        message = str(exc).splitlines()[0]
        genuine = "deterministic implementation" in message
        note(
            f"{name}: one step under the flag on {device}",
            "ok" if genuine else "BLOCKED",
            f"{'RAISES: ' if genuine else ''}{message[:110]}",
        )
        record.setdefault(name, {})[f"flag_step_{device}"] = (
            f"raises: {message[:90]}" if genuine else "environment-blocked"
        )
    finally:
        torch.use_deterministic_algorithms(False)


def main() -> int:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
        f"device {device}"
    )
    if device != "cuda":
        print("NO CUDA: the questions under test are CUDA-kernel questions.")
        blocked.append("no CUDA device; run where the models actually train")

    probe_interpolate(device)
    probe_roi_align(device)

    features = blob_images(4, seed=20260731)
    labels = np.random.default_rng(7).uniform(1.5, 4.5, size=4)

    for name in BACKBONES:
        print(f"{name}:")
        try:
            repeat_gradient(name, device, features, labels)
            post_restore_step(name, device, features, labels)
        except (RuntimeError, NotImplementedError) as exc:
            note(
                f"{name}: flag-OFF probes on {device}", "BLOCKED",
                f"{str(exc).splitlines()[0][:90]}",
            )
        step_under_flag(name, device, features, labels)

    print("\nRECORD (paste-ready):")
    for key in ("ops", *BACKBONES):
        if key in record:
            print(f"  {key}:")
            for probe, value in record[key].items():
                print(f"    {probe}: {value}")

    print()
    if failures:
        print(f"NOT VERIFIED: {len(failures)} probe(s) failed: {failures}")
        return 1
    if blocked:
        print("PROBED WHERE POSSIBLE; ENVIRONMENT-BLOCKED elsewhere:")
        for item in blocked:
            print(f"  - {item}")
        return 2
    print("ALL PROBES RAN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
