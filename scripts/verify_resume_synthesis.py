"""Byte-identical resume on the REAL synthesis pipeline, not the stub loop.

`tests/test_checkpoint.py` proves the mechanism on a miniature loop. This proves
it on the code that will actually be paused: images, TPS warps, staging, memmaps
and the JSONL journals.

  1. build N faces uninterrupted           -> artifact A
  2. build N faces, KILLED mid-run         -> partial working directory
  3. resume it to completion               -> artifact B
  4. every file in A and B compared BYTE FOR BYTE

Run as a subprocess so the kill is a real process death -- an exception inside
one process would unwind cleanly and prove less.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"X:\cleft-aesthetics")
SCRATCH = Path(
    os.environ.get(
        "CLEFT_SCRATCH",
        str(pathlib.Path(tempfile.gettempdir()) / "cleft-scratch"),
    )
)
PY = REPO / ".venv" / "Scripts" / "python.exe"
#: **The same shape as the cluster gate** (configs/p5_synthesis_gate_{a,b}.yaml):
#: 200 faces, a checkpoint every 50, so there are four and the kill lands between
#: two of them. A kill landing exactly ON a checkpoint would leave nothing to
#: truncate and would not exercise the journal rollback at all -- which is the
#: part of the resume that can actually be wrong.
#:
#: Kept identical to the gate on purpose: this script and the cluster procedure
#: should differ only in HOW the interruption arrives (a process kill here, a
#: Run:AI pause there), not in what is being interrupted.
N_FACES = 200
KILL_AFTER = 60          # kill once a checkpoint past this appears...
KILL_AFTER_2 = 150       # ...and again during the resume, from a rewritten journal
CHECKPOINT_EVERY = 50
DRIFT_SECONDS = 6.0      # ...but only after a few more faces have been journalled,
                         # so there is something for the resume to roll back

CONFIG = """schema_version: 1
phase: p5
tier: dev
seed: 1337

inputs:
  - name: scut_root
    path: ${{CLEFT_SCUT_ROOT}}
    rollup_sha256: 1041ceed0ef458e97c8be845eda30cf7957067fa730a4975a0f83bd8ed4947ec

task:
  kind: asymmetry_synthesis
  scut_root: scut_root
  geometries: ["g1", "g2"]
  n_faces: {n_faces}
  n_sheet: 3
  magnitudes: [0.0, 0.015, 0.025, 0.035]
  ar_sampling: observed
  checkpoint_every: {every}
  write_artifact: true
  out_version: {version}
"""


def hash_tree(root: Path) -> dict[str, str]:
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            out[path.relative_to(root).as_posix()] = digest
    return out


def checkpoint_step(working: Path) -> int:
    """How far the build says it has got, read from its own checkpoint."""
    path = working / "checkpoint.npz"
    if not path.exists():
        return 0
    try:
        sys.path.insert(0, str(REPO / "src"))
        from cleft.train import checkpoint as ckpt

        return ckpt.load(path).step
    except Exception:
        return 0  # mid-write; the atomic replace means the next poll sees it


def run(config: Path, out_root: Path, kill_at: tuple[Path, int] | None = None) -> str:
    """Run the task. ``kill_at`` is (working_dir, faces) -- poll the build's own
    checkpoint and kill the PROCESS once it passes that many faces.

    Killed rather than raised: a real Run:AI pause deletes the pod, so an
    exception inside the process would unwind cleanly and prove less.
    """
    env = dict(os.environ)
    env["CLEFT_SCUT_ROOT"] = r"C:\data\scut-fbp5500\SCUT-FBP5500_v2"
    env["PYTHONPATH"] = str(REPO / "src")
    process = subprocess.Popen(
        [str(PY), "-m", "cleft.run", "--config", str(config), "--out", str(out_root)],
        cwd=str(REPO), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )

    if kill_at is not None:
        working, target = kill_at
        import time

        while process.poll() is None:
            time.sleep(0.05)
            if checkpoint_step(working) >= target:
                # **Let a few more faces land BEFORE killing.** Killing the
                # instant the checkpoint appears leaves the journals with exactly
                # the checkpointed number of rows -- so the resume would have
                # nothing to truncate, and the rollback, which is the part of the
                # resume that can actually be wrong, would never be exercised.
                # The test would pass without testing it.
                time.sleep(DRIFT_SECONDS)
                process.kill()
                process.wait()
                step = checkpoint_step(working)
                rows = len(
                    (working / "faces.jsonl").read_text(encoding="utf-8").splitlines()
                )
                if rows <= step:
                    raise SystemExit(
                        f"journal has {rows} rows against a checkpoint at {step}: "
                        "nothing beyond the checkpoint, so the resume would not "
                        "exercise the truncation. Raise DRIFT_SECONDS."
                    )
                return (
                    f"KILLED at checkpoint {step} with {rows} journal rows "
                    f"({rows - step} beyond the checkpoint, to be rolled back)"
                )
        # The loop only exits when the process died on its own -- which is either
        # a build too short to catch or a crash, and those need different fixes.
        output, _ = process.communicate()
        if process.returncode != 0:
            print(output[-3000:])
            raise SystemExit(f"the build to be interrupted CRASHED ({process.returncode})")
        raise SystemExit(
            "the build finished before reaching the kill point; lower KILL_AFTER "
            "or raise n_faces, or this proves nothing"
        )

    output, _ = process.communicate()
    if process.returncode != 0:
        print(output[-2500:])
        raise SystemExit(f"run failed with {process.returncode}")
    return "completed"


def main() -> None:
    work = SCRATCH / "resume_check"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    for version in ("resume_clean", "resume_paused"):
        for path in (REPO / "data" / "scut" / version,
                     REPO / "data" / "scut" / f"{version}.inprogress"):
            if path.exists():
                shutil.rmtree(path)

    # ---- 1. uninterrupted -------------------------------------------------
    clean_cfg = work / "clean.yaml"
    clean_cfg.write_text(
        CONFIG.format(n_faces=N_FACES, every=CHECKPOINT_EVERY, version="resume_clean"),
        encoding="utf-8",
    )
    print("1. uninterrupted build ...", run(clean_cfg, work / "out_clean"))

    # ---- 2. killed mid-run ------------------------------------------------
    paused_cfg = work / "paused.yaml"
    paused_cfg.write_text(
        CONFIG.format(n_faces=N_FACES, every=CHECKPOINT_EVERY, version="resume_paused"),
        encoding="utf-8",
    )
    partial = REPO / "data" / "scut" / "resume_paused.inprogress"
    print("2. interrupted build  ...", run(paused_cfg, work / "out_paused",
                                           kill_at=(partial, KILL_AFTER)))

    # **A SECOND interruption, during the resume.** One kill exercises one
    # rollback from a pristine journal. A second kill rolls back a journal that
    # a previous resume already rewrote -- which is where a counter recorded
    # beside the resume point can drift out of step with it, and the cluster
    # gate's 211-row faces.json is exactly what that drift looks like.
    print("2b. interrupted again ...", run(paused_cfg, work / "out_paused_2",
                                           kill_at=(partial, KILL_AFTER_2)))

    assert partial.is_dir(), "no in-progress directory survived the kill"
    assert not (REPO / "data" / "scut" / "resume_paused").exists(), (
        "the FINAL artifact path exists after a kill -- a half-built artifact is "
        "indistinguishable from a complete one"
    )
    print(f"   partial survived: {sorted(p.name for p in partial.iterdir())}")

    # ---- 3. resume --------------------------------------------------------
    print("3. resumed build      ...", run(paused_cfg, work / "out_paused2"))

    # ---- 4. compare -------------------------------------------------------
    a = hash_tree(REPO / "data" / "scut" / "resume_clean")
    b = hash_tree(REPO / "data" / "scut" / "resume_paused")

    # The index, checked explicitly rather than only via the byte comparison:
    # the cluster gate failed on faces.json alone while every array matched, so
    # "the arrays are identical" is not the same statement as "the artifact is
    # correct".
    import json as _json

    for name in ("resume_clean", "resume_paused"):
        faces = _json.loads(
            (REPO / "data" / "scut" / name / "faces.json").read_text(encoding="utf-8")
        )
        stems = [entry["stem"] for entry in faces]
        status = "OK" if len(faces) == N_FACES and len(set(stems)) == N_FACES else "BAD"
        print(f"   index {name}: {len(faces)} rows, {len(set(stems))} unique  [{status}]")

    print(f"\n4. comparing {len(a)} files")
    if set(a) != set(b):
        print(f"   FILE SETS DIFFER: only-clean={sorted(set(a)-set(b))} "
              f"only-paused={sorted(set(b)-set(a))}")
    identical = [k for k in sorted(set(a) & set(b)) if a[k] == b[k]]
    differing = [k for k in sorted(set(a) & set(b)) if a[k] != b[k]]
    for name in identical:
        print(f"   IDENTICAL  {name}")
    for name in differing:
        print(f"   DIFFERS    {name}")

    total_bytes = sum(
        p.stat().st_size for p in (REPO / "data" / "scut" / "resume_clean").rglob("*")
        if p.is_file()
    )
    print(f"\n   {len(identical)}/{len(a)} files byte-identical "
          f"({total_bytes / 1024**2:.1f} MiB compared)")
    print("RESULT:", "PASS" if not differing and set(a) == set(b) else "FAIL")


if __name__ == "__main__":
    main()
