"""Checkpointing (Phase 5, a Phase 6 entry gate): a resumed run must be
byte-identical to an uninterrupted one, RNG state included.

The loop here is a stub of the Phase 6 pretraining loop -- numpy only, torch
stays out of the test extra -- but it exercises every element of the contract:
weights, an optimizer slot with momentum (state a naive resume loses), an RNG
stream consumed every step (divergence the moment restoration is wrong), an
append-style output that must be truncated to the checkpointed row count, and
a final artifact whose bytes are compared.

The counterfactual is tested too: a resume that RESEEDS instead of restoring
diverges. A byte-identity test that cannot fail is not a test (PLAN R7's
running tally).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cleft.train import checkpoint


# --------------------------------------------------------------------------
# the stub pretraining loop
# --------------------------------------------------------------------------


def stub_pretraining_run(
    run_dir: Path,
    *,
    total_steps: int = 11,
    seed: int = 7,
    checkpoint_every: int = 4,
    stop_after: int | None = None,
    resume_by_reseeding: bool = False,
) -> str:
    """A miniature pretraining loop honouring the checkpoint contract.

    Every step consumes the RNG (a synthetic "batch"), updates momentum and
    weights, and appends a row to ``curves.csv``. Every ``checkpoint_every``
    steps it saves. ``stop_after`` simulates the pod being killed mid-run --
    after a checkpoint, possibly with extra rows already written.

    ``resume_by_reseeding`` is the deliberate mistake: restore everything
    EXCEPT the RNG stream. It exists so the byte-identity assertion is shown
    to have teeth.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    curves = run_dir / "curves.csv"
    target = run_dir / checkpoint.CHECKPOINT_NAME

    found = checkpoint.latest(run_dir)
    if found is not None:
        state = checkpoint.load(found)
        rng = np.random.default_rng(seed if resume_by_reseeding else None)
        if not resume_by_reseeding:
            checkpoint.restore_numpy_rng(rng, state.numpy_rng)
        weights = state.arrays["weights"].copy()
        momentum = state.arrays["momentum"].copy()
        step = state.step
        # Truncate the append-style output to the durable count: rows written
        # after the checkpoint will be produced again by the resumed steps.
        lines = curves.read_text().split("\n")
        keep = 1 + state.output_rows["curves"]  # header + durable rows
        curves.write_bytes(("\n".join(lines[:keep]) + "\n").encode())
    else:
        rng = np.random.default_rng(seed)
        weights = np.zeros(6)
        momentum = np.zeros(6)
        step = 0
        curves.write_bytes(b"step,loss\n")

    while step < total_steps:
        if stop_after is not None and step >= stop_after:
            return "paused"
        step += 1
        batch = rng.standard_normal(6)
        gradient = weights - batch
        momentum = 0.9 * momentum + gradient
        weights = weights - 0.1 * momentum
        loss = float((gradient**2).mean())
        with curves.open("a", newline="") as stream:
            stream.write(f"{step},{loss!r}\n")
        if step % checkpoint_every == 0:
            checkpoint.save(
                target,
                checkpoint.Checkpoint(
                    step=step,
                    epoch=step // checkpoint_every,
                    arrays={"weights": weights, "momentum": momentum},
                    numpy_rng=checkpoint.capture_numpy_rng(rng),
                    output_rows={"curves": step},
                ),
            )

    np.save(run_dir / "weights.npy", weights)
    (run_dir / "metrics.json").write_bytes(
        json.dumps({"final_loss": repr(loss), "steps": step}).encode()
    )
    return "finished"


def output_bytes(run_dir: Path) -> dict[str, bytes]:
    return {
        name: (run_dir / name).read_bytes()
        for name in ("curves.csv", "weights.npy", "metrics.json")
    }


# --------------------------------------------------------------------------
# byte-identity, and its teeth
# --------------------------------------------------------------------------


def test_a_resumed_run_is_byte_identical_to_an_uninterrupted_one(tmp_path):
    """THE gate. Killed mid-run after a checkpoint, with two rows written
    beyond it -- the resume must truncate them, restore weights, momentum and
    the RNG stream, and land on identical bytes for every output."""
    clean = tmp_path / "clean"
    assert stub_pretraining_run(clean) == "finished"

    paused = tmp_path / "paused"
    assert stub_pretraining_run(paused, stop_after=6) == "paused"
    # The kill left rows 5 and 6 in curves.csv but the checkpoint at step 4:
    # exactly the shape of a pod pause between checkpoint and next save.
    assert stub_pretraining_run(paused) == "finished"

    assert output_bytes(paused) == output_bytes(clean)


def test_a_resume_that_reseeds_instead_of_restoring_diverges(tmp_path):
    """The counterfactual that proves the test above can fail. Identical
    everything except the RNG stream -> different bytes."""
    clean = tmp_path / "clean"
    stub_pretraining_run(clean)

    reseeded = tmp_path / "reseeded"
    stub_pretraining_run(reseeded, stop_after=6)
    stub_pretraining_run(reseeded, resume_by_reseeding=True)

    assert output_bytes(reseeded)["curves.csv"] != output_bytes(clean)["curves.csv"]


def test_a_pause_landing_exactly_on_a_checkpoint_also_resumes_identically(tmp_path):
    """The truncation edge case where there is nothing to truncate."""
    clean = tmp_path / "clean"
    stub_pretraining_run(clean)

    paused = tmp_path / "paused"
    stub_pretraining_run(paused, stop_after=8)
    stub_pretraining_run(paused)

    assert output_bytes(paused) == output_bytes(clean)


def test_two_pauses_in_one_run_still_land_identically(tmp_path):
    """Run:AI can pause more than once; each resume must compose."""
    clean = tmp_path / "clean"
    stub_pretraining_run(clean)

    paused = tmp_path / "paused"
    stub_pretraining_run(paused, stop_after=5)
    stub_pretraining_run(paused, stop_after=9)
    stub_pretraining_run(paused)

    assert output_bytes(paused) == output_bytes(clean)


# --------------------------------------------------------------------------
# RNG state round-trips
# --------------------------------------------------------------------------


def test_the_rng_stream_continues_exactly_where_it_stopped():
    source = np.random.default_rng(1337)
    source.standard_normal(100)
    state = checkpoint.capture_numpy_rng(source)
    expected = source.standard_normal(50)

    restored = np.random.default_rng()
    checkpoint.restore_numpy_rng(restored, state)
    assert np.array_equal(restored.standard_normal(50), expected)


def test_the_rng_state_survives_json_and_the_npz_round_trip(tmp_path):
    source = np.random.default_rng(99)
    source.standard_normal(17)
    target = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(
        target,
        checkpoint.Checkpoint(
            step=1,
            epoch=0,
            arrays={},
            numpy_rng=checkpoint.capture_numpy_rng(source),
        ),
    )
    expected = source.standard_normal(5)

    restored = np.random.default_rng()
    checkpoint.restore_numpy_rng(restored, checkpoint.load(target).numpy_rng)
    assert np.array_equal(restored.standard_normal(5), expected)


def test_an_unsupported_bit_generator_is_refused_not_half_serialised():
    legacy = np.random.Generator(np.random.MT19937(1))
    with pytest.raises(checkpoint.CheckpointError, match="MT19937"):
        checkpoint.capture_numpy_rng(legacy)


def test_restoring_across_bit_generators_is_refused():
    source = np.random.default_rng(1)
    state = checkpoint.capture_numpy_rng(source)
    other = np.random.Generator(np.random.MT19937(1))
    with pytest.raises(checkpoint.CheckpointError, match="different run"):
        checkpoint.restore_numpy_rng(other, state)


# --------------------------------------------------------------------------
# the file: versioning, integrity, atomicity
# --------------------------------------------------------------------------


def a_checkpoint(step: int = 3) -> checkpoint.Checkpoint:
    rng = np.random.default_rng(5)
    rng.standard_normal(4)
    return checkpoint.Checkpoint(
        step=step,
        epoch=1,
        arrays={"weights": np.arange(6, dtype=float), "momentum": np.ones(6)},
        numpy_rng=checkpoint.capture_numpy_rng(rng),
        output_rows={"curves": step},
        extra={"loss_ema": 0.25},
    )


def test_a_checkpoint_round_trips_completely(tmp_path):
    target = tmp_path / checkpoint.CHECKPOINT_NAME
    original = a_checkpoint()
    checkpoint.save(target, original)
    loaded = checkpoint.load(target)

    assert loaded.step == original.step
    assert loaded.epoch == original.epoch
    assert set(loaded.arrays) == set(original.arrays)
    for name in original.arrays:
        assert np.array_equal(loaded.arrays[name], original.arrays[name])
        assert loaded.arrays[name].dtype == original.arrays[name].dtype
    assert loaded.numpy_rng == original.numpy_rng
    assert loaded.output_rows == original.output_rows
    assert loaded.extra == original.extra
    assert loaded.torch_rng is None
    assert loaded.torch_cuda_rng == ()


def test_an_unknown_version_is_refused(tmp_path):
    target = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(target, a_checkpoint())
    _rewrite_header(target, {"version": 99})
    with pytest.raises(checkpoint.CheckpointError, match="version"):
        checkpoint.load(target)


def test_a_tampered_array_fails_its_digest(tmp_path):
    """The braces to the zip CRC's belt: a checkpoint whose arrays do not
    match their recorded digests is corrupt, and resuming from corrupt state
    would produce a run that looks fine and reproduces nothing."""
    target = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(target, a_checkpoint())
    _tamper_array(target, "weights")
    with pytest.raises(checkpoint.CheckpointError, match="digest"):
        checkpoint.load(target)


def test_flipped_bytes_anywhere_are_refused(tmp_path):
    target = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(target, a_checkpoint())
    blob = bytearray(target.read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    target.write_bytes(bytes(blob))
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.load(target)


def test_saving_replaces_atomically_so_the_previous_survives_a_failed_save(
    tmp_path, monkeypatch
):
    """A pause mid-save must leave the previous checkpoint loadable -- the
    exact Run:AI failure atomic_write_bytes exists for. Simulated by making
    the replacement step raise."""
    target = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(target, a_checkpoint(step=3))

    import cleft.provenance.context as context

    def exploding_replace(source, destination):
        raise OSError("simulated pause mid-save")

    monkeypatch.setattr(context.os, "replace", exploding_replace)
    with pytest.raises(OSError, match="simulated pause"):
        checkpoint.save(target, a_checkpoint(step=4))
    monkeypatch.undo()

    assert checkpoint.load(target).step == 3


def test_latest_finds_the_checkpoint_and_only_in_its_run_dir(tmp_path):
    assert checkpoint.latest(tmp_path) is None
    target = tmp_path / checkpoint.CHECKPOINT_NAME
    checkpoint.save(target, a_checkpoint())
    assert checkpoint.latest(tmp_path) == target


def test_missing_and_headerless_files_are_refused(tmp_path):
    with pytest.raises(checkpoint.CheckpointError, match="no checkpoint"):
        checkpoint.load(tmp_path / "absent.npz")

    bare = tmp_path / "bare.npz"
    np.savez(bare, stray=np.zeros(2))
    with pytest.raises(checkpoint.CheckpointError, match="header"):
        checkpoint.load(bare)


def test_non_ndarray_state_is_refused_at_save(tmp_path):
    bad = checkpoint.Checkpoint(
        step=1,
        epoch=0,
        arrays={"weights": [1.0, 2.0]},  # type: ignore[dict-item]
        numpy_rng=checkpoint.capture_numpy_rng(np.random.default_rng(1)),
    )
    with pytest.raises(checkpoint.CheckpointError, match="ndarray"):
        checkpoint.save(tmp_path / "x.npz", bad)


# --------------------------------------------------------------------------
# torch state: the no-torch contract (torch stays out of the test extra)
# --------------------------------------------------------------------------


def test_torch_rng_capture_is_explicitly_absent_without_torch():
    try:
        import torch  # noqa: F401

        pytest.skip("torch is importable here; the no-torch contract is moot")
    except ImportError:
        pass
    assert checkpoint.capture_torch_rng() == (None, ())
    checkpoint.restore_torch_rng(None, ())  # a no-op, not an error


def test_cuda_state_without_cpu_state_is_refused():
    with pytest.raises(checkpoint.CheckpointError, match="without CPU"):
        checkpoint.restore_torch_rng(None, (b"\x00",))


def test_torch_state_in_a_torchless_environment_is_refused():
    try:
        import torch  # noqa: F401

        pytest.skip("torch is importable here; the refusal cannot fire")
    except ImportError:
        pass
    with pytest.raises(checkpoint.CheckpointError, match="not\\s+importable"):
        checkpoint.restore_torch_rng(b"\x00\x01", ())


# --------------------------------------------------------------------------
# white-box helpers for the tamper tests
# --------------------------------------------------------------------------


def _read_all(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as bundle:
        return {name: bundle[name] for name in bundle.files}


def _write_all(path: Path, payload: dict[str, np.ndarray]) -> None:
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    path.write_bytes(buffer.getvalue())


def _rewrite_header(path: Path, updates: dict) -> None:
    payload = _read_all(path)
    header = json.loads(bytes(payload["header__json"].tobytes()).decode())
    header.update(updates)
    payload["header__json"] = np.frombuffer(
        json.dumps(header, sort_keys=True).encode(), dtype=np.uint8
    )
    _write_all(path, payload)


def _tamper_array(path: Path, name: str) -> None:
    payload = _read_all(path)
    tampered = payload[f"array__{name}"].copy()
    tampered.flat[0] += 1
    payload[f"array__{name}"] = tampered
    _write_all(path, payload)
