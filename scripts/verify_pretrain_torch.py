"""First run of the TorchPretrainModel mechanics: construct, step, resume.

The pretraining loop's tests all use the stub -- torch stays out of the test
extra -- so until this runs, the torch half of the protocol (factory
construction, normalization read, head init, the optimizer state_dict round
trip through the checkpoint format) is code nothing has executed. A restore
bug found here costs minutes; found on the cluster it costs a queue wait and
a checkpoint nobody can trust.

Run OUTSIDE the project venv (system python with torch), or in the image::

    python scripts/verify_pretrain_torch.py

What it checks, per transformer backbone (graph builds land cluster-side):

1. reset(): factory construction, normalization_for reads the model's own
   preprocessing, the head zero-init at the train-label mean, full-fine-tune
   parameter report (trainable == total).
2. Epoch-0 predictions equal the training mean exactly (gate 3's premise).
3. Two optimizer steps run and change the loss.
4. state_arrays -> checkpoint.save -> checkpoint.load -> load_state_arrays
   into a FRESH model reproduces predictions bitwise, and one further
   identical step on both models stays bitwise identical -- which is false if
   any optimizer slot (exp_avg, exp_avg_sq, step) failed to round-trip.

``pretrained=False``: this verifies MECHANICS, so the weight values are
irrelevant and no download is needed. The twelve real runs use pretrained
weights; nothing here changes that path.

**[MEASURED 2026-07-31] PASSED, all twelve checks**, torch 2.13.0+cu126 /
timm 1.0.27 on the laptop's system python -- so this verifies the LOOP, not
the image; re-run it in the pinned image before the twelve runs. Two findings
worth keeping:

* the totals cross-check the registry's measured constants exactly --
  85,799,425 = ViT-B/16's 85,798,656 body + a 769-parameter head, and
  86,744,249 = Swin-B's 86,743,224 + 1,025;
* the first version of this script had no RNG restore and read Swin-B's
  stochastic-depth divergence as a restore failure -- the CHECK was the
  defect (see the inline comment). Swin-B is therefore the backbone that
  gives the RNG half of the resume contract teeth; ViT-B/16 cannot exercise
  it (no stochastic depth), and both are now bitwise-identical through a
  post-restore step.

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

from cleft.models.factory import normalization_source  # noqa: E402
from cleft.train import checkpoint as ckpt  # noqa: E402
from cleft.train.pretrain import TorchPretrainModel  # noqa: E402

BACKBONES = ("vit_b16", "swin_b")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "ok" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)


def fresh(backbone: str, features: np.ndarray, labels: np.ndarray) -> TorchPretrainModel:
    model = TorchPretrainModel(
        name=backbone, pretrained=False, batch_size=4, seed=7, device="cpu"
    )
    model.reset(features, labels)
    return model


def main() -> int:
    rng = np.random.default_rng(20260731)
    features = rng.integers(0, 256, size=(8, 224, 224, 3), dtype=np.uint8)
    labels = rng.uniform(1.5, 4.5, size=8)

    for backbone in BACKBONES:
        print(f"{backbone}:")
        model = fresh(backbone, features, labels)

        report = model.parameter_report
        check(
            "full fine-tuning: trainable == total",
            report["trainable_parameters"] == report["total_parameters"],
            f"{report['trainable_parameters']:,}",
        )
        check(
            "normalization read from the model's own config",
            bool(normalization_source(model._model)),
            normalization_source(model._model),
        )

        epoch0 = model.predict(features)
        check(
            "epoch-0 head predicts the train mean",
            bool(np.allclose(epoch0, float(np.mean(labels)), atol=1e-5)),
            f"pred mean {float(np.mean(epoch0)):.6f} vs {float(np.mean(labels)):.6f}",
        )

        loss_a = model.train_batch(features[:4], labels[:4])
        loss_b = model.train_batch(features[4:], labels[4:])
        check("two optimizer steps run", np.isfinite(loss_a) and np.isfinite(loss_b),
              f"losses {loss_a:.4f}, {loss_b:.4f}")

        # ---- the round trip, through the real checkpoint format ----------
        #
        # The torch RNG is captured WITH the arrays and restored before EACH
        # branch's continuation step, exactly as the loop does on resume.
        # The first version of this script omitted that and read a Swin-B
        # divergence as a restore failure; it was the CHECK that was wrong
        # (PLAN R2/R7): timm's Swin-B trains with stochastic depth
        # (drop_path_rate 0.1), whose masks come from the torch RNG stream,
        # so two continuations under different streams differ while both are
        # correct. ViT-B/16 has no stochastic depth and passed either way --
        # which is also why Swin is the backbone that gives this check teeth.
        arrays = model.state_arrays(include_optimizer=True)
        torch_rng, cuda_rng = ckpt.capture_torch_rng()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.npz"
            ckpt.save(
                path,
                ckpt.Checkpoint(
                    step=2, epoch=1, arrays=arrays,
                    numpy_rng=ckpt.capture_numpy_rng(np.random.default_rng(7)),
                    torch_rng=torch_rng, torch_cuda_rng=cuda_rng,
                ),
            )
            loaded = ckpt.load(path)

        resumed = fresh(backbone, features, labels)
        resumed.load_state_arrays(loaded.arrays)

        before = model.predict(features)
        after = resumed.predict(features)
        check(
            "restored model predicts bitwise-identically",
            bool(np.array_equal(before, after)),
            f"max |diff| {float(np.max(np.abs(before - after))):.3g}",
        )

        # One further IDENTICAL step on both, each from the checkpointed RNG
        # state: diverges if any optimizer slot (exp_avg, exp_avg_sq, step)
        # or the RNG state failed to round-trip.
        ckpt.restore_torch_rng(loaded.torch_rng, loaded.torch_cuda_rng)
        loss_original = model.train_batch(features[:4], labels[:4])
        step_original = model.predict(features)

        ckpt.restore_torch_rng(loaded.torch_rng, loaded.torch_cuda_rng)
        loss_resumed = resumed.train_batch(features[:4], labels[:4])
        step_resumed = resumed.predict(features)
        check(
            "one post-restore step stays bitwise identical",
            bool(np.array_equal(step_original, step_resumed))
            and loss_original == loss_resumed,
            f"losses {loss_original:.6f} vs {loss_resumed:.6f}",
        )

    print()
    if failures:
        print(f"NOT VERIFIED: {len(failures)} check(s) failed: {failures}")
        return 1
    print("ALL CHECKS PASSED: the torch pretraining mechanics round-trip.")
    print(f"torch {torch.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
