"""Generate the synthetic smoke input artifact.

Run from the repo root:

    python scripts/make_smoke_input.py

Writes ``data/smoke/v1/`` and prints the rollup hash to paste into
``configs/smoke.yaml``. The output is deterministic, so re-running it on any
machine reproduces the same hash; if it does not, something about the artifact
contract has changed and the run guard will say so.

This is the pattern every real data artifact follows (Part 2.6): create a new
version directory, never modify an existing one. ``MANIFEST.json`` records the
payload rollup and how the artifact was generated. The config separately
declares the rollup of the whole directory *including* ``MANIFEST.json``, so
relabelling a manifest without rebuilding the payload also breaks the hash --
that is exactly the failure that put ``masked_v2`` labels on v3 weights.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cleft.provenance import hash_dir  # noqa: E402

ARTIFACT = REPO / "data" / "smoke" / "v1"
N = 64
SEED = 1337


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> int:
    if ARTIFACT.exists():
        print(f"{ARTIFACT} already exists. Data artifacts are immutable: create v2.")
        print(f"current rollup: {hash_dir(ARTIFACT)['rollup']}")
        return 1

    ARTIFACT.mkdir(parents=True)

    rng = np.random.default_rng(SEED)
    truth = rng.uniform(1.0, 5.0, size=N)
    pred = np.clip(truth + rng.normal(0.0, 0.8, size=N), 1.0, 5.0)

    with (ARTIFACT / "values.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "truth", "pred"])
        for i, (t, p) in enumerate(zip(truth, pred), start=1):
            writer.writerow([f"S{i:04d}", f"{t:.6f}", f"{p:.6f}"])

    payload = hash_dir(ARTIFACT)
    manifest = {
        "artifact": "smoke/v1",
        "purpose": "synthetic input for configs/smoke.yaml; no real data",
        "generated_by": "scripts/make_smoke_input.py",
        "generator_git_sha": git_sha(),
        "n_samples": N,
        "seed": SEED,
        "numpy_version": np.__version__,
        "payload_rollup": payload["rollup"],
        "payload_files": payload["files"],
        "payload_total_bytes": payload["total_bytes"],
    }
    (ARTIFACT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    full = hash_dir(ARTIFACT)
    print(f"payload rollup     : {payload['rollup']}")
    print(f"directory rollup   : {full['rollup']}   <- put this in configs/smoke.yaml")
    print(f"files              : {full['file_count']}, {full['total_bytes']} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
