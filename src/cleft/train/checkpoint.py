"""Checkpointing for the Phase 6 pretraining path. A resumed run must be
byte-identical to an uninterrupted one.

**Why this exists — [MEASURED 2026-07-28], PLAN §2.7.** Run identity survives a
pause; training does not. Run:AI resumes by recreating the pod, the run
directory reattaches via ``JOB_UUID``, and the gate-2 sweep silently redid all
ten seeds in the second pod with no symptom but wall time. A Phase 6 job paused
at hour three would restart at epoch zero. The resume mechanism reattaches the
directory, not the work; this module is the work half. It is a Phase 6 entry
gate, built in Phase 5.

**The contract, for the loop Phase 6 builds.** ``harness.py`` is frozen and is
not touched; the pretraining loop is new code and consumes this module:

1. On start, ``latest(run_dir)`` — if a checkpoint exists, ``load`` it,
   ``restore_numpy_rng`` (and torch RNG on the cluster), restore every array,
   and **truncate append-style outputs to the recorded row counts**
   (``output_rows``). Rows written after the last checkpoint were durable but
   will be re-produced; keeping them would duplicate steps, and byte-identity
   is the test that catches either mistake.
2. Every N steps, ``save(path, Checkpoint(...))`` with **everything the next
   step reads**: weights, optimizer slots, step and epoch counters, the RNG
   state, and the durable row counts. Anything omitted here is a divergence
   the resume test will find.
3. The final outputs of a resumed run must equal the uninterrupted run's
   **byte for byte** — ``tests/test_checkpoint.py`` asserts exactly that on a
   stub loop, RNG stream included, and also asserts the counterfactual: a
   resume that reseeds instead of restoring diverges. A byte-identity test
   that cannot fail is not a test.

**Atomicity is inherited, not reimplemented.** ``provenance.atomic_write_bytes``
writes to a temp file, fsyncs, then ``os.replace`` — a pause mid-save leaves
the previous checkpoint intact and loadable, which is precisely the Run:AI
failure mode it was written for.

**Scope, stated narrowly.** numpy RNG support is PCG64 / PCG64DXSM — what
``np.random.default_rng`` produces — and anything else is refused loudly
rather than half-serialised. Torch RNG state is captured when torch is
importable and restored only into an environment that matches (CUDA states
restore onto the same device count); tests use the numpy path only, because
torch stays out of the test extra.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..provenance.context import atomic_write_bytes

#: Format version. Bump on any change to the payload layout; ``load`` refuses
#: versions it does not know rather than guessing.
CHECKPOINT_VERSION = 1

#: The checkpoint's name inside a run directory. One file, atomically replaced;
#: history is not kept because a resume only ever wants the newest state.
CHECKPOINT_NAME = "checkpoint.npz"

#: Bit generators whose state round-trips through JSON as plain integers.
#: MT19937 keeps its key as an ndarray and is refused until someone actually
#: needs it -- narrow and tested beats broad and assumed.
SUPPORTED_BIT_GENERATORS = ("PCG64", "PCG64DXSM")

_HEADER_KEY = "header__json"
_ARRAY_PREFIX = "array__"
_TORCH_KEY = "torch__rng"
_CUDA_PREFIX = "cuda__"


class CheckpointError(RuntimeError):
    """The checkpoint could not be saved, loaded, or trusted."""


@dataclass(frozen=True)
class Checkpoint:
    """Everything the next training step reads. If the loop consults state that
    is not in here, a resumed run cannot be byte-identical -- add it."""

    #: Completed steps. The resumed loop starts at ``step + 1``.
    step: int
    epoch: int
    #: Weights and optimizer slots, keyed by the loop's own names.
    arrays: dict[str, np.ndarray]
    #: ``Generator.bit_generator.state`` -- captured with ``capture_numpy_rng``.
    numpy_rng: dict
    #: Durable row counts for append-style outputs (e.g. ``{"curves": 40}``).
    #: On resume the loop truncates each file to its recorded count, because
    #: rows written after this checkpoint will be produced again.
    output_rows: dict[str, int] = field(default_factory=dict)
    #: JSON-serialisable scalars the loop wants back (loss EMA, best metric...).
    extra: dict = field(default_factory=dict)
    #: ``torch.get_rng_state()`` bytes, when torch was present at capture.
    torch_rng: bytes | None = None
    #: ``torch.cuda.get_rng_state_all()`` bytes, one entry per device.
    torch_cuda_rng: tuple[bytes, ...] = ()


# --------------------------------------------------------------------------
# RNG state, captured and restored exactly
# --------------------------------------------------------------------------


def capture_numpy_rng(generator: np.random.Generator) -> dict:
    """The generator's full bit-generator state, JSON-serialisable."""
    state = generator.bit_generator.state
    name = state.get("bit_generator")
    if name not in SUPPORTED_BIT_GENERATORS:
        raise CheckpointError(
            f"bit generator {name!r} is not supported; expected one of "
            f"{SUPPORTED_BIT_GENERATORS}. Its state does not round-trip "
            "through JSON as integers, and a half-serialised RNG resumes "
            "into a different stream -- silently."
        )
    return json.loads(json.dumps(state))


def restore_numpy_rng(generator: np.random.Generator, state: dict) -> None:
    """Set the generator to a captured state, in place."""
    name = state.get("bit_generator")
    have = type(generator.bit_generator).__name__
    if name != have:
        raise CheckpointError(
            f"checkpoint carries {name!r} RNG state but the loop's generator "
            f"is {have!r}; restoring across bit generators is not a resume, "
            "it is a different run"
        )
    generator.bit_generator.state = state


def capture_torch_rng() -> tuple[bytes | None, tuple[bytes, ...]]:
    """Torch CPU and per-device CUDA RNG state, or (None, ()) without torch."""
    try:
        import torch
    except Exception:  # noqa: BLE001 - absence of torch is the normal case here
        return None, ()
    cpu = bytes(torch.get_rng_state().numpy().tobytes())
    cuda = tuple(
        bytes(s.numpy().tobytes()) for s in torch.cuda.get_rng_state_all()
    )
    return cpu, cuda


def restore_torch_rng(cpu: bytes | None, cuda: tuple[bytes, ...]) -> None:
    """Restore captured torch state. Refuses environment mismatches loudly."""
    if cpu is None:
        if cuda:
            raise CheckpointError("CUDA RNG state present without CPU state")
        return
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise CheckpointError(
            "the checkpoint carries torch RNG state but torch is not "
            f"importable: {exc}. Resuming this run in an environment without "
            "torch cannot reproduce it."
        ) from None
    torch.set_rng_state(torch.frombuffer(bytearray(cpu), dtype=torch.uint8).clone())
    devices = torch.cuda.device_count()
    if len(cuda) != devices:
        raise CheckpointError(
            f"checkpoint carries CUDA RNG state for {len(cuda)} device(s) but "
            f"{devices} are visible. A resumed run must see the environment it "
            "paused in; this one does not."
        )
    for index, blob in enumerate(cuda):
        torch.cuda.set_rng_state(
            torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone(), index
        )


# --------------------------------------------------------------------------
# save / load
# --------------------------------------------------------------------------


def _array_digest(array: np.ndarray) -> str:
    payload = (
        str(array.dtype).encode()
        + b"|"
        + str(array.shape).encode()
        + b"|"
        + np.ascontiguousarray(array).tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def save(path: str | Path, checkpoint: Checkpoint) -> None:
    """Write atomically: a pause mid-save leaves the previous file loadable."""
    entries: dict[str, np.ndarray] = {}
    digests: dict[str, str] = {}
    for name, array in checkpoint.arrays.items():
        if not isinstance(array, np.ndarray):
            raise CheckpointError(
                f"array {name!r} is {type(array).__name__}, not ndarray. "
                "Convert deliberately at the call site; silent coercion here "
                "would hide a dtype change that alters results."
            )
        entries[_ARRAY_PREFIX + name] = array
        digests[name] = _array_digest(array)

    if checkpoint.torch_rng is not None:
        entries[_TORCH_KEY] = np.frombuffer(checkpoint.torch_rng, dtype=np.uint8)
    for index, blob in enumerate(checkpoint.torch_cuda_rng):
        entries[f"{_CUDA_PREFIX}{index}"] = np.frombuffer(blob, dtype=np.uint8)

    header = {
        "version": CHECKPOINT_VERSION,
        "step": int(checkpoint.step),
        "epoch": int(checkpoint.epoch),
        "numpy_rng": checkpoint.numpy_rng,
        "output_rows": {k: int(v) for k, v in checkpoint.output_rows.items()},
        "extra": checkpoint.extra,
        "array_sha256": digests,
        "has_torch_rng": checkpoint.torch_rng is not None,
        "n_cuda_rng": len(checkpoint.torch_cuda_rng),
    }
    entries[_HEADER_KEY] = np.frombuffer(
        json.dumps(header, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **entries)
    atomic_write_bytes(path, buffer.getvalue())


def load(path: str | Path) -> Checkpoint:
    """Read and verify. Corruption and unknown versions are refused, named."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"no checkpoint at {path}")
    try:
        with np.load(path) as bundle:
            payload = {name: bundle[name] for name in bundle.files}
    except CheckpointError:
        raise
    except Exception as exc:  # noqa: BLE001 - any unreadable file gets one message
        raise CheckpointError(f"checkpoint at {path} is unreadable: {exc}") from exc

    if _HEADER_KEY not in payload:
        raise CheckpointError(f"checkpoint at {path} has no header")
    header = json.loads(bytes(payload[_HEADER_KEY].tobytes()).decode("utf-8"))

    version = header.get("version")
    if version != CHECKPOINT_VERSION:
        raise CheckpointError(
            f"checkpoint version {version!r} is not {CHECKPOINT_VERSION}; "
            "refusing to guess at a layout this code does not know"
        )

    arrays: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key.startswith(_ARRAY_PREFIX):
            arrays[key[len(_ARRAY_PREFIX):]] = value

    declared = header.get("array_sha256", {})
    if set(declared) != set(arrays):
        raise CheckpointError(
            f"checkpoint at {path} declares arrays {sorted(declared)} but "
            f"carries {sorted(arrays)}"
        )
    for name, expected in declared.items():
        actual = _array_digest(arrays[name])
        if actual != expected:
            raise CheckpointError(
                f"array {name!r} in {path} fails its digest -- the checkpoint "
                "is corrupt, and resuming from corrupt state would produce a "
                "run that looks fine and reproduces nothing"
            )

    torch_rng = (
        bytes(payload[_TORCH_KEY].tobytes()) if header.get("has_torch_rng") else None
    )
    cuda = tuple(
        bytes(payload[f"{_CUDA_PREFIX}{index}"].tobytes())
        for index in range(int(header.get("n_cuda_rng", 0)))
    )

    return Checkpoint(
        step=int(header["step"]),
        epoch=int(header["epoch"]),
        arrays=arrays,
        numpy_rng=header["numpy_rng"],
        output_rows={k: int(v) for k, v in header.get("output_rows", {}).items()},
        extra=header.get("extra", {}),
        torch_rng=torch_rng,
        torch_cuda_rng=cuda,
    )


def latest(run_dir: str | Path) -> Path | None:
    """The run directory's checkpoint, if one exists. One file by design."""
    candidate = Path(run_dir) / CHECKPOINT_NAME
    return candidate if candidate.exists() else None
