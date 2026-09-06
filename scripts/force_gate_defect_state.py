"""Force the exact state that produced the cluster gate's 211-row faces.json,
and show the fix recovers from it.

I could not reproduce the failure locally even with two kills, so instead of
guessing the mechanism this constructs the STATE it must have been in: a
checkpoint whose recorded `output_rows["faces"]` is larger than its `step`, with a
journal to match. That is the only shape that yields 211 for a 200-face build --
final = keep + (200 - step), so keep = step + 11.

The old code truncated to `output_rows` and would keep all 211. The new code
derives the truncation from `step` alone, so it must land on 200 regardless of
what the recorded counter says.
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(r"X:\cleft-aesthetics")
sys.path.insert(0, str(REPO / "src"))
PY = REPO / ".venv" / "Scripts" / "python.exe"
SCRATCH = Path(
    os.environ.get(
        "CLEFT_SCRATCH",
        str(pathlib.Path(tempfile.gettempdir()) / "cleft-scratch"),
    )
)
N_FACES = 120
DRIFT = 11               # the cluster saw exactly 11 duplicate stems
VERSION = "forced_defect"

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
  geometries: ["g1"]
  n_faces: {n}
  n_sheet: 2
  magnitudes: [0.0, 0.025]
  ar_sampling: observed
  checkpoint_every: 40
  write_artifact: true
  out_version: {version}
"""


def launch(config: Path, out_root: Path):
    env = dict(os.environ)
    env["CLEFT_SCUT_ROOT"] = r"C:\data\scut-fbp5500\SCUT-FBP5500_v2"
    env["PYTHONPATH"] = str(REPO / "src")
    return subprocess.run(
        [str(PY), "-m", "cleft.run", "--config", str(config), "--out", str(out_root)],
        cwd=str(REPO), env=env, capture_output=True, text=True,
    )


def main() -> None:
    from cleft.train import checkpoint as ckpt

    work = SCRATCH / "forced"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for suffix in ("", ".inprogress"):
        path = REPO / "data" / "scut" / (VERSION + suffix)
        if path.exists():
            shutil.rmtree(path)

    cfg = work / "forced.yaml"
    cfg.write_text(CONFIG.format(n=N_FACES, version=VERSION), encoding="utf-8")

    # ---- 1. reference ----------------------------------------------------
    result = launch(cfg, work / "out_a")
    assert result.returncode == 0, result.stdout[-2000:]
    reference = json.loads(
        (REPO / "data" / "scut" / VERSION / "faces.json").read_text(encoding="utf-8")
    )
    print(f"reference build: {len(reference)} faces")
    shutil.rmtree(REPO / "data" / "scut" / VERSION)

    # ---- 2. rebuild a partial, then CORRUPT the recorded counter ---------
    working = REPO / "data" / "scut" / (VERSION + ".inprogress")
    result = launch(cfg, work / "out_b")
    assert result.returncode == 0, result.stdout[-2000:]
    # Roll the completed artifact back into a partial at face `step`.
    complete = REPO / "data" / "scut" / VERSION
    complete.rename(working)

    step = 80
    faces = [
        json.loads(line)
        for line in (working / "faces.jsonl").read_text(encoding="utf-8").splitlines()
    ] if (working / "faces.jsonl").exists() else [
        {"stem": e["stem"], **e} for e in reference
    ]
    # The journal as it would look at the pause: step + DRIFT rows.
    (working / "faces.jsonl").write_text(
        "".join(json.dumps(e, sort_keys=True) + "\n" for e in faces[: step + DRIFT]),
        encoding="utf-8",
    )
    survival_rows = (step + DRIFT) * 1  # one deformed magnitude in this config
    (working / "survival.jsonl").write_text(
        "".join(
            json.dumps({"stem": faces[i]["stem"], "min_survival_ratio": 1.1},
                       sort_keys=True) + "\n"
            for i in range(survival_rows)
        ),
        encoding="utf-8",
    )
    (working / "faces.json").unlink(missing_ok=True)

    # **The corruption**: step says 80, the recorded counter says 91.
    ckpt.save(
        working / ckpt.CHECKPOINT_NAME,
        ckpt.Checkpoint(
            step=step,
            epoch=0,
            arrays={},
            numpy_rng=ckpt.capture_numpy_rng(__import__("numpy").random.default_rng(1337)),
            output_rows={"faces": step + DRIFT, "survival": survival_rows},
        ),
    )
    print(f"forced state: step={step}, output_rows.faces={step + DRIFT}, "
          f"journal={step + DRIFT} rows")

    # ---- 3. resume from the corrupted state ------------------------------
    result = launch(cfg, work / "out_c")
    if result.returncode != 0:
        print(result.stdout[-3000:])
        raise SystemExit("resume from the forced state FAILED to complete")

    published = json.loads(
        (REPO / "data" / "scut" / VERSION / "faces.json").read_text(encoding="utf-8")
    )
    stems = [e["stem"] for e in published]
    print(f"\nafter resume: {len(published)} rows, {len(set(stems))} unique")
    print(f"old behaviour would have given: {step + DRIFT + (N_FACES - step)} rows")
    for line in result.stdout.splitlines():
        if "RESUMING" in line or "index verified" in line:
            print("  " + line.strip()[:150])

    ok = len(published) == N_FACES and len(set(stems)) == N_FACES
    print("\nRESULT:", "PASS -- derived count ignored the corrupted counter" if ok
          else "FAIL")
    shutil.rmtree(REPO / "data" / "scut" / VERSION, ignore_errors=True)


if __name__ == "__main__":
    main()
