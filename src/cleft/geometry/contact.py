"""Build the Phase 2 contact sheet from real crops.

Brief §7. Renders a handful of patients at both geometries so the band
boundaries can be checked against actual anatomy before anything trains on them.

The PNG is CLUSTER-ONLY and stays there. What comes back is the reviewer's text
plus the SHAREABLE aggregate report this also produces.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np

from ..data.manifest import load_manifest
from . import render
from .mapping import map_patches
from .patches import PatchConfig, get
from .staging import Staged, stage
from .trapezium import DEFAULT as G1_TRAPEZIUM
from .trapezium import Trapezium, unwarp

#: Under G2 the mask IS the full square, so nothing is excluded.
G2_TRAPEZIUM = Trapezium(top_half_width=0.5, bot_half_width=0.5)

GEOMETRIES = ("g1", "g2")


class ContactError(RuntimeError):
    """The contact sheet could not be built."""


@dataclass
class SheetResult:
    geometry: str
    sheet: np.ndarray
    labels: list[str]
    reports: list[dict]
    cell: tuple[int, int]


def select_patients(rows: list[dict], n: int) -> list[dict]:
    """Pick ``n`` patients spread evenly across the three reporting classes.

    Deterministic: no sampling, no seed. A contact sheet that showed a different
    set each run would make "the bands look right" unrepeatable, and the whole
    point is a judgement someone can come back to.

    Stratified because grade 4-5 patients are the ones whose anatomy is most
    distorted, and they are exactly where a band boundary is most likely to land
    somewhere silly. A sheet of nine grade-1 faces would look reassuring and
    prove nothing.
    """
    if n < 1:
        raise ContactError(f"n_patients must be positive, got {n}")

    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["class3"], []).append(row)
    for members in by_class.values():
        members.sort(key=lambda r: int(r["patient_id"]))

    chosen: list[dict] = []
    classes = sorted(by_class)
    index = 0
    while len(chosen) < min(n, len(rows)):
        bucket = by_class[classes[index % len(classes)]]
        position = index // len(classes)
        if position < len(bucket):
            chosen.append(bucket[position])
        elif all(index // len(classes) >= len(by_class[c]) for c in classes):
            break
        index += 1
    return chosen[:n]


def find_image(folders_root: Path, patient_id: int, image_id: int) -> Path:
    """Locate one image, tolerating whatever the filename convention turns out to be."""
    folder = folders_root / str(patient_id)
    if not folder.is_dir():
        raise ContactError(f"no folder for patient {patient_id} at {folder}")

    matches = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file()
        and not path.name.startswith("._")
        and str(image_id) in path.stem
    ]
    if not matches:
        raise ContactError(
            f"no file for image {image_id} in {folder}. Present: "
            f"{[p.name for p in sorted(folder.iterdir())][:10]}"
        )
    return matches[0]


def _stage_for(geometry: str, image: np.ndarray) -> tuple[Staged, Trapezium | None]:
    staged = stage(image)
    if geometry == "g1":
        return staged, G1_TRAPEZIUM
    if geometry == "g2":
        # Unwarp the STAGED square, so every patient receives identical
        # treatment rather than distortion proportional to their own crop.
        unwarped = Staged(
            image=unwarp(staged.image),
            source_size=staged.source_size,
            content_box=staged.content_box,
            scale=staged.scale,
        )
        # Boundary drawing is pointless under G2: the mask is the whole square.
        return unwarped, None
    raise ContactError(f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}")


def build_sheet(
    geometry: str,
    selected: list[dict],
    folders_root: Path,
    *,
    generator: str = "grid",
    config: PatchConfig | None = None,
    columns: int = 4,
) -> SheetResult:
    config = config or PatchConfig()
    mask = G1_TRAPEZIUM if geometry == "g1" else G2_TRAPEZIUM
    patches = get(generator).generate(mask, config)

    panels, labels, reports = [], [], []
    for row in selected:
        patient_id = int(row["patient_id"])
        path = find_image(folders_root, patient_id, int(row["frontal_id"]))
        staged, boundary = _stage_for(geometry, render.load_image(path))
        mapped = map_patches(staged, patches)

        panels.append(render.Panel(str(patient_id), render.render_overlay(staged, mapped, boundary)))
        # The label carries the grade, because "does this band land right" has a
        # different answer for a grade 1 face and a grade 5 one.
        labels.append(f"{patient_id}  c{row['class3']}  ar={staged.aspect_ratio:.2f}")
        report = render.overlay_report(mapped, staged)
        report["patient_id"] = patient_id
        reports.append(report)

    if not panels:
        raise ContactError("no patients selected; nothing to render")

    cell = (max(p.image.shape[1] for p in panels), max(p.image.shape[0] for p in panels))
    return SheetResult(
        geometry=geometry,
        sheet=render.tile(panels, columns=columns),
        labels=labels,
        reports=reports,
        cell=cell,
    )


@dataclass(frozen=True)
class Variant:
    v_offset: float
    scale: float
    scale_x: float
    box_scale_x: float = 1.0

    @property
    def label(self) -> str:
        base = f"dy={self.v_offset:+.2f} s={self.scale:.2f} sx={self.scale_x:.2f}"
        return base if self.box_scale_x == 1.0 else f"{base} bx={self.box_scale_x:.2f}"


def build_sweep(
    selected: list[dict],
    folders_root: Path,
    variants: list[Variant],
    *,
    geometry: str = "g2",
    base: PatchConfig | None = None,
) -> SheetResult:
    """One sheet, the same faces under every candidate placement.

    Six configs would answer the same question in six round trips, and comparing
    six separate sheets by memory is exactly how "it looks a bit better" replaces
    a decision. Here every candidate is on one image, in a grid: variants down,
    faces across, each panel labelled with the parameters that produced it.

    ``geometry`` defaults to G2 because that is where the region-scheme ablation
    has to run anyway -- see the note in aggregate().
    """
    if not variants:
        raise ContactError("no variants to sweep")
    base = base or PatchConfig()

    panels, labels, reports = [], [], []
    staged_cache: dict[int, tuple[Staged, Trapezium | None]] = {}

    for variant in variants:
        config = replace(
            base,
            anatomy_v_offset=variant.v_offset,
            anatomy_scale=variant.scale,
            anatomy_scale_x=variant.scale_x,
            anatomy_box_scale_x=variant.box_scale_x,
        )
        mask = G1_TRAPEZIUM if geometry == "g1" else G2_TRAPEZIUM
        patches = get("anatomy").generate(mask, config)

        for row in selected:
            patient_id = int(row["patient_id"])
            if patient_id not in staged_cache:
                path = find_image(folders_root, patient_id, int(row["frontal_id"]))
                staged_cache[patient_id] = _stage_for(geometry, render.load_image(path))
            staged, boundary = staged_cache[patient_id]

            mapped = map_patches(staged, patches)
            panels.append(
                render.Panel(
                    variant.label,
                    render.render_overlay(staged, mapped, boundary, labels=True),
                )
            )
            labels.append(f"{variant.label}  #{patient_id}")
            report = render.overlay_report(mapped, staged)
            report["patient_id"] = patient_id
            report["variant"] = variant.label
            reports.append(report)

    columns = len(selected)
    cell = (max(p.image.shape[1] for p in panels), max(p.image.shape[0] for p in panels))
    return SheetResult(
        geometry=geometry,
        sheet=render.tile(panels, columns=columns),
        labels=labels,
        reports=reports,
        cell=cell,
    )


def variant_grid(
    v_offsets: list[float],
    scales: list[float],
    scales_x: list[float],
    box_scales_x: list[float] | None = None,
) -> list[Variant]:
    """Cartesian product, in a stable order so the sheet layout is predictable."""
    return [
        Variant(v_offset=dy, scale=s, scale_x=sx, box_scale_x=bx)
        for dy, s, sx, bx in product(
            v_offsets, scales, scales_x, box_scales_x or [1.0]
        )
    ]


def sweep_geometry_report(
    variants: list[Variant], geometry: str = "g2", base: PatchConfig | None = None
) -> dict[str, dict]:
    """Per-variant geometry facts, computed without touching an image.

    A patch set is the same for every face, so clamping and overlap are
    properties of the variant rather than of the panel. Reporting them per
    variant is what lets a candidate be excluded on a number rather than on how
    it looked at panel scale.
    """
    from .patches import (
        assert_no_clamping,
        clamped_regions,
        overlap_report,
        region_coverage,
    )

    base = base or PatchConfig()
    mask = G1_TRAPEZIUM if geometry == "g1" else G2_TRAPEZIUM
    out: dict[str, dict] = {}

    for variant in variants:
        config = replace(
            base,
            anatomy_v_offset=variant.v_offset,
            anatomy_scale=variant.scale,
            anatomy_scale_x=variant.scale_x,
            anatomy_box_scale_x=variant.box_scale_x,
        )
        patches = get("anatomy").generate(mask, config)
        report = overlap_report(patches)
        report["clamped_regions"] = clamped_regions(patches)
        report["n_clamped"] = sum(1 for p in patches if p.clamped)

        # Coverage is reported at BOTH geometries, not just the one being swept.
        # The sweep runs at G2, where the mask is the whole square and every
        # coverage is trivially 1.0 -- so reporting only the sweep's own geometry
        # would hide precisely the failure this exists to surface. The G1 column
        # is what shows that a variant chosen here would feed background if it
        # were ever run there.
        g2_patches = get("anatomy").generate(G2_TRAPEZIUM, config)
        g1_patches = get("anatomy").generate(G1_TRAPEZIUM, config)
        report["region_coverage_g2"] = region_coverage(g2_patches, G2_TRAPEZIUM)
        report["region_coverage_g1"] = region_coverage(g1_patches, G1_TRAPEZIUM)

        below = {
            name: value
            for name, value in report["region_coverage_g1"].items()
            if value < 0.5
        }
        report["regions_below_half_coverage_g1"] = sorted(below)
        worst_name = min(
            report["region_coverage_g1"], key=report["region_coverage_g1"].get
        )
        report["worst_region_g1"] = {
            "region": worst_name,
            "coverage": report["region_coverage_g1"][worst_name],
        }

        try:
            assert_no_clamping(patches)
            report["candidate"] = True
        except Exception:
            # Not a candidate: a clamped set only looks larger, because the
            # outermost regions were pulled back inside the frame to fit.
            report["candidate"] = False
        out[variant.label] = report
    return out


def sweep_clamp_report(result: SheetResult) -> dict[str, list[str]]:
    """Which variants clamped a region, and which regions.

    Reported per variant rather than in aggregate: a variant that only looks
    slightly wider because its commissures were pulled back inside the frame is
    not a candidate, and at panel scale nothing about it looks wrong.
    """
    out: dict[str, list[str]] = {}
    for report in result.reports:
        if report.get("clamped_regions"):
            out.setdefault(report["variant"], [])
            for name in report["clamped_regions"]:
                if name not in out[report["variant"]]:
                    out[report["variant"]].append(name)
    return {k: sorted(v) for k, v in out.items()}


def aggregate(results: list[SheetResult]) -> dict:
    """SHAREABLE summary. Patient ids are dropped here; the sheet keeps them.

    NOTE, recorded from the first anatomy sheet: anatomy regions touch **zero**
    white under G1 (worst coverage 1.0), where the grid had 10 of 27 patches
    carrying up to 50% white. The regions are small and placed inward, so they
    never reach the mask edge.

    That means anatomy-vs-grid and G1-vs-G2 are **confounded**: measured at G1,
    a grid-vs-anatomy difference is partly a background-exposure difference. The
    region-scheme ablation must be run at **G2**, where both are zero.
    """
    out: dict[str, dict] = {}
    for result in results:
        coverages = [r["min_coverage"] for r in result.reports]
        ratios = [r["aspect_ratio"] for r in result.reports]
        out[result.geometry] = {
            "n_panels": len(result.reports),
            "n_patches": result.reports[0]["n_patches"],
            "aspect_ratio_min": min(ratios),
            "aspect_ratio_max": max(ratios),
            "worst_patch_coverage": round(min(coverages), 4),
            "mean_patches_with_white": round(
                float(np.mean([r["n_patches_with_white"] for r in result.reports])), 2
            ),
        }
    return out
