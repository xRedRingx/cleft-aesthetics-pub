"""Measure a partially-written embedding artifact: whole, or partial?

    PYTHONPATH=src python scripts/verify_recon_artifact.py \
        data/embeddings/embeddings_recon_v1 data/manifests/cleft_v1

**[2026-08-24, phase13.WRITER_DIED_AFTER_SAVE]** ``p13-extract-recon``
attempt 0 wrote ``embeddings_recon_v1`` and then died at ``KeyError:
'rollup'`` composing its own metrics -- after the save, before the
bookkeeping. The run record cannot say whether the artifact is whole,
and the immutability guard makes the state sticky: every retry refuses
until someone MEASURES the directory.

This script measures it. It READS ONLY -- it deletes nothing, writes
nothing, and decides nothing. The verdict it prints is evidence for the maintainer's decision (keep, or delete and re-extract), not the decision.

WHOLE requires all of: both files present; values (n, 768) float; every
value finite; metadata's patient_ids exactly the manifest's ids IN
ORDER; and the recorded shape agreeing with the array on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

EXPECTED_DIM = 768
EXPECTED_PATIENTS = 237


def check(artifact: Path, manifest: Path) -> int:
    from cleft.data.manifest import load_manifest

    failures: list[str] = []
    notes: list[str] = []

    values_path = artifact / "values.npy"
    metadata_path = artifact / "metadata.json"
    for path in (values_path, metadata_path):
        if not path.is_file():
            failures.append(f"MISSING FILE: {path.name}")
    if failures:
        print("\n".join(failures))
        print("\nVERDICT: PARTIAL -- the save did not finish.")
        return 1

    values = np.load(values_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = load_manifest(manifest / "manifest.csv")
    manifest_ids = [int(row["patient_id"]) for row in rows]

    print(f"artifact : {artifact}")
    print(f"values   : shape {values.shape}, dtype {values.dtype}")
    print(f"files    : {sorted(p.name for p in artifact.iterdir())}")

    if values.ndim != 2 or values.shape[1] != EXPECTED_DIM:
        failures.append(
            f"SHAPE: {values.shape}, expected (n, {EXPECTED_DIM})"
        )
    if values.shape[0] != EXPECTED_PATIENTS:
        failures.append(
            f"ROWS: {values.shape[0]}, expected {EXPECTED_PATIENTS} -- a "
            "short array is the signature of a truncated write"
        )

    finite = np.isfinite(values)
    n_bad = int(finite.size - finite.sum())
    zero_rows = int((~values.any(axis=1)).sum()) if values.ndim == 2 else -1
    print(f"finite   : {finite.sum()}/{finite.size} ({n_bad} non-finite)")
    print(f"all-zero rows: {zero_rows}")
    if n_bad:
        failures.append(f"NON-FINITE: {n_bad} values are nan or inf")
    if zero_rows:
        failures.append(
            f"ALL-ZERO ROWS: {zero_rows} -- rows never written by the "
            "extractor read as zeros, which is what a partial write looks "
            "like from inside a correctly-shaped array"
        )

    recorded = metadata.get("patient_ids")
    if recorded is None:
        failures.append("METADATA: no patient_ids recorded")
    else:
        recorded = [int(p) for p in recorded]
        if recorded != manifest_ids:
            first = next(
                (
                    i for i, (a, b) in enumerate(zip(recorded, manifest_ids))
                    if a != b
                ),
                min(len(recorded), len(manifest_ids)),
            )
            failures.append(
                f"ROW ORDER: metadata patient_ids differ from cleft_v1 at "
                f"index {first} ({len(recorded)} recorded vs "
                f"{len(manifest_ids)} in the manifest)"
            )
        else:
            notes.append(
                f"row order matches cleft_v1 exactly ({len(recorded)} ids)"
            )
        if len(recorded) != values.shape[0]:
            failures.append(
                f"DISAGREEMENT: {len(recorded)} patient_ids but "
                f"{values.shape[0]} rows -- metadata and array disagree"
            )

    for key in ("kind", "backbone", "init", "geometry"):
        notes.append(f"{key} = {metadata.get(key)!r}")
    if metadata.get("init") != "imagenet" or metadata.get("geometry") != "g1":
        failures.append(
            f"IDENTITY: init/geometry are {metadata.get('init')}/"
            f"{metadata.get('geometry')}, expected imagenet/g1"
        )

    print("\n".join(f"note     : {n}" for n in notes))
    if failures:
        print("\n" + "\n".join(f"FAIL     : {f}" for f in failures))
        print(
            "\nVERDICT: PARTIAL (or wrong). Delete the directory and let "
            "the fixed task re-extract into the same version -- nothing "
            "downstream consumed it."
        )
        return 1
    print(
        "\nVERDICT: WHOLE. Every check passes: shape, finiteness, no "
        "all-zero rows, and row order identical to cleft_v1. The crash "
        "was after the save, in the run's own bookkeeping only -- keep "
        "the artifact and declare it (phase13.WRITER_DIED_AFTER_SAVE)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    return check(args.artifact, args.manifest)


if __name__ == "__main__":
    raise SystemExit(main())
