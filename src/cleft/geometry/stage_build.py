"""Phase 2 cluster build: stage all 237, generate patches, write the artifact.

Exit criteria 7-10:

  7. staging runs over all 237 without error; the aspect-ratio distribution
     matches the measured 0.553-1.099
  8. patch generation succeeds at both geometries; report patch counts and, for
     G1, the white-fraction distribution per band
  9. contact sheet reviewed (done separately)
 10. artifact versioned and hashed per PLAN §2.6

The artifact is the Phase 3 input: staged tensors, per-image geometry, and the
patch definitions that produced them. Staging every image again in Phase 3 would
be both slow and a chance for the two runs to disagree.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..data.manifest import load_manifest
from ..provenance import atomic_write_text, hash_dir
from . import contact, patches as P, render
from .mapping import map_patches
from .staging import AR_MAX, AR_MIN, OUTPUT_SIZE, stage
from .trapezium import DEFAULT as G1_TRAPEZIUM
from .trapezium import unwarp

GEOMETRIES = ("g1", "g2")
GENERATORS = ("grid", "anatomy", "random")


class StageBuildError(RuntimeError):
    """The Phase 2 build cannot proceed."""


@dataclass
class StageBuildResult:
    artifact_dir: Path
    summary: dict


def build(
    *,
    manifest_dir: Path,
    patient_folders: Path,
    artifact_dir: Path,
    config: P.PatchConfig | None = None,
    generators: tuple[str, ...] = GENERATORS,
    geometries: tuple[str, ...] = GEOMETRIES,
    log=print,
) -> StageBuildResult:
    """Validate everything, then write. Same discipline as the Phase 1 build."""
    if artifact_dir.exists():
        raise StageBuildError(
            f"{artifact_dir} already exists. Data artifacts are immutable: bump "
            "the version rather than modifying one results may already cite."
        )
    config = config or P.PatchConfig()

    rows = load_manifest(manifest_dir / "manifest.csv")
    log(f"staging {len(rows)} patients from {patient_folders}")

    # ---- stage every image ------------------------------------------------
    staged_by_geometry: dict[str, list[np.ndarray]] = {g: [] for g in geometries}
    geometry_rows: list[dict] = []

    for row in rows:
        patient_id = int(row["patient_id"])
        path = contact.find_image(patient_folders, patient_id, int(row["frontal_id"]))
        image = render.load_image(path)
        base = stage(image)

        for geometry in geometries:
            array = base.image if geometry == "g1" else unwarp(base.image)
            staged_by_geometry[geometry].append(array)

        geometry_rows.append(
            {
                "patient_id": patient_id,
                "frontal_id": int(row["frontal_id"]),
                "source_w": base.source_size[0],
                "source_h": base.source_size[1],
                "aspect_ratio": round(base.aspect_ratio, 6),
                "content_x": base.content_box[0],
                "content_y": base.content_box[1],
                "content_w": base.content_box[2],
                "content_h": base.content_box[3],
                "pad_fraction": round(base.pad_fraction, 6),
            }
        )

    ratios = [r["aspect_ratio"] for r in geometry_rows]
    log(f"  aspect ratio {min(ratios):.3f} .. {max(ratios):.3f}")
    _check_aspect_ratios(ratios)

    # ---- patches, per generator per geometry ------------------------------
    patch_sets: dict[str, list[P.Patch]] = {}
    patch_report: dict[str, dict] = {}

    for generator in generators:
        for geometry in geometries:
            mask = G1_TRAPEZIUM if geometry == "g1" else contact.G2_TRAPEZIUM
            produced = P.get(generator).generate(mask, config)
            key = f"{generator}_{geometry}"
            patch_sets[key] = produced
            patch_report[key] = P.summarise(produced)
            log(
                f"  {key}: {len(produced)} patches, "
                f"worst coverage {patch_report[key]['coverage_min']:.3f}"
            )

    # ---- the mapping actually works for every patient ---------------------
    _check_mapping(rows, geometry_rows, patch_sets, geometries)

    summary = {
        "n_patients": len(rows),
        "output_size": OUTPUT_SIZE,
        "geometries": list(geometries),
        "generators": list(generators),
        "aspect_ratio": {
            "min": round(min(ratios), 4),
            "max": round(max(ratios), 4),
            "median": round(float(np.median(ratios)), 4),
            "expected_min": AR_MIN,
            "expected_max": AR_MAX,
        },
        "pad_fraction": {
            "min": round(min(r["pad_fraction"] for r in geometry_rows), 4),
            "max": round(max(r["pad_fraction"] for r in geometry_rows), 4),
            "mean": round(float(np.mean([r["pad_fraction"] for r in geometry_rows])), 4),
        },
        "patches": patch_report,
        "anatomy_parameters": {
            "v_offset": config.anatomy_v_offset,
            "scale": config.anatomy_scale,
            "scale_x": config.anatomy_scale_x,
            "box_scale_x": config.anatomy_box_scale_x,
        },
    }

    _write(artifact_dir, staged_by_geometry, geometry_rows, patch_sets, summary)
    log(f"artifact written: {artifact_dir}")
    return StageBuildResult(artifact_dir=artifact_dir, summary=summary)


def _check_aspect_ratios(ratios: list[float]) -> None:
    """Exit criterion 7: the distribution must match what was measured."""
    tolerance = 0.02
    if min(ratios) < AR_MIN - tolerance or max(ratios) > AR_MAX + tolerance:
        raise StageBuildError(
            f"aspect ratios span {min(ratios):.4f}..{max(ratios):.4f}, outside the "
            f"measured {AR_MIN}..{AR_MAX}. Either the crops are not the ones this "
            "was measured on, or a crop is malformed."
        )


def _check_mapping(rows, geometry_rows, patch_sets, geometries) -> None:
    """Two patients of different aspect ratio must get different pixel boxes.

    The safeguard from brief §5, asserted over the real cohort rather than over
    two synthetic images.
    """
    from .staging import Staged

    by_ratio = sorted(geometry_rows, key=lambda r: r["aspect_ratio"])
    narrow, wide = by_ratio[0], by_ratio[-1]
    if abs(narrow["aspect_ratio"] - wide["aspect_ratio"]) < 1e-6:
        raise StageBuildError("every crop has the same aspect ratio; that cannot be")

    def fake(row):
        return Staged(
            image=np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8),
            source_size=(row["source_w"], row["source_h"]),
            content_box=(
                row["content_x"], row["content_y"], row["content_w"], row["content_h"]
            ),
            scale=1.0,
        )

    key = f"grid_{geometries[0]}"
    a = [m.pixels for m in map_patches(fake(narrow), patch_sets[key])]
    b = [m.pixels for m in map_patches(fake(wide), patch_sets[key])]
    if a == b:
        raise StageBuildError(
            f"patients {narrow['patient_id']} (AR {narrow['aspect_ratio']:.3f}) and "
            f"{wide['patient_id']} (AR {wide['aspect_ratio']:.3f}) received "
            "IDENTICAL pixel boxes. Patches are not being mapped per image."
        )


def _write(artifact_dir, staged_by_geometry, geometry_rows, patch_sets, summary) -> None:
    artifact_dir.mkdir(parents=True)

    # Staged tensors are patient images: CLUSTER-ONLY, and named so the tier
    # guard would refuse to let them be marked otherwise.
    for geometry, arrays in staged_by_geometry.items():
        np.save(artifact_dir / f"staged_patient_{geometry}.npy", np.stack(arrays))

    header = list(geometry_rows[0])
    lines = ["# CLUSTER-ONLY: patient-keyed geometry", ",".join(header)]
    lines += [",".join(str(row[k]) for k in header) for row in geometry_rows]
    atomic_write_text(artifact_dir / "geometry.csv", "\n".join(lines) + "\n")

    # Patch definitions are normalised boxes: no patient data at all.
    atomic_write_text(
        artifact_dir / "patches.json",
        json.dumps(
            {
                key: [
                    {
                        "id": p.id, "band": p.band, "x": p.x, "y": p.y,
                        "w": p.w, "h": p.h, "coverage": p.coverage,
                        "mirror_id": p.mirror_id, "clamped": p.clamped,
                    }
                    for p in produced
                ]
                for key, produced in patch_sets.items()
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    payload = hash_dir(artifact_dir)
    atomic_write_text(
        artifact_dir / "MANIFEST.json",
        json.dumps(
            {
                "artifact": artifact_dir.name,
                "phase": "p2",
                "summary": summary,
                "payload_rollup": payload["rollup"],
                "payload_files": payload["files"],
                "payload_total_bytes": payload["total_bytes"],
                "staged_tier": "CLUSTER-ONLY",
                "note": (
                    "staged_patient_*.npy are (N, 224, 224, 3) uint8, row order "
                    "matching geometry.csv. patches.json boxes are normalised to "
                    "the trapezium bbox; map them per image with "
                    "cleft.geometry.staging.content_box_to_pixels."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
