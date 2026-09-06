"""Everything gate 1 needs to make a run reproducible.

Phase 3 §3 gate 1. Determinism does not happen by hope; each of these is a
deliberate setting, and the module records what it actually managed to set so a
run's ``env.json`` says whether it was configured or merely intended.

**Do not turn `use_deterministic_algorithms` off to make a run pass.** If an op
the model needs has no deterministic kernel it will raise, and that is the gate
working -- it fails loudly rather than producing a quietly irreproducible number.
Report the op.

Importable without torch: the seeding of ``random`` and numpy happens either way,
and the torch settings are recorded as unavailable rather than silently skipped.
"""

from __future__ import annotations

import os
import random

import numpy as np

#: cuBLAS needs this set BEFORE the CUDA context is created, so it is exported
#: here and also set in the Dockerfile -- setting it after the first CUDA call
#: has no effect and would leave the run non-deterministic while looking fine.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class DeterminismError(RuntimeError):
    """Determinism could not be configured."""


def configure(seed: int, *, require_torch: bool = False) -> dict:
    """Seed everything and switch on deterministic kernels. Returns what was set."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_CONFIG)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    record = {
        "seed": seed,
        "python_random": True,
        "numpy_legacy_global": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch": False,
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark_disabled": False,
    }

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        if require_torch:
            raise DeterminismError(
                f"torch is required for a determinism-gated run but is not "
                f"importable: {exc}"
            ) from None
        return record

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    record.update(
        torch=True,
        torch_version=torch.__version__,
        torch_deterministic_algorithms=True,
        cudnn_deterministic=True,
        cudnn_benchmark_disabled=True,
    )
    return record


def worker_init_fn(worker_id: int):  # pragma: no cover - needs torch
    """Seed each DataLoader worker.

    Without this, workers share the parent seed and their augmentation streams
    are correlated but not reproducible across runs with different worker counts.
    """
    import torch

    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int):  # pragma: no cover - needs torch
    """A seeded generator for DataLoader shuffling."""
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
