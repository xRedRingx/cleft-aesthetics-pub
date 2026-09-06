"""Piecewise-affine warp over the shipped landmark triangulation.

Phase 17 arm B's transformation family (``phase17.PHASE_17_RULINGS``):
Rosero warps by piecewise affine over landmark triangulation; this
module supplies that family **in-repo** -- Delaunay from ``scipy.spatial``
(already in the pinned image), barycentric inverse mapping,
nearest-neighbour sampling to match ``staging.resize_nearest``. **No
skimage, no detector, nothing beyond the pinned image.**

What is REUSED rather than re-derived: the displacement rule is
``synthesis.control_points`` -- the same four targets, the same
direction/weight constants, the same anchor ring, the same
magnitude-in-crop-widths convention -- so B-vs-C (piecewise vs TPS on
shared training) isolates the TRANSFORMATION FAMILY and nothing else.
The parked module itself is not extended; this is a sibling, and the
suite it mirrors keeps running on both.

Sampling and framing follow ``synthesis.warp_content`` deliberately:
full source image in, only the crop box rewritten, inverse mapping so
there are no holes, explicit identity at magnitude zero.
"""

from __future__ import annotations

import numpy as np

from . import synthesis


class PiecewiseError(RuntimeError):
    pass


#: **The two declared adaptations** (phase17.PHASE_17_RULINGS['arm_b']):
#: named at build time, in the module that implements them, so they are
#: DECLARED where the code lives and not only in the phase record.
DECLARED_ADAPTATIONS = {
    "stated": (
        "DECLARED, not discovered -- both adaptations were named in El "
        "the maintainer's ruling before this module existed, and the claim "
        "wording is bound: never 'we replicated Rosero'"
    ),
    "adaptation_1": (
        "shipped landmarks in place of detection: the warp reads "
        "SCUT's shipped 86-point .pts files (Phase 5's own reader); no "
        "detector runs anywhere, per the recorded constraint's basis "
        "(a rejection of detection, not of landmark data)"
    ),
    "adaptation_2": (
        "fixed midline in place of a landmark-localised split: the "
        "left/right views split at the staged square's vertical "
        "centre (phase17.left_right_views), not at a per-face landmark "
        "midline -- the same fixed-axis discipline the "
        "mirror-difference instrument uses, and the same trade the TPS "
        "module recorded when it anchored the midline"
    ),
}


def _triangle_affines(target: np.ndarray, source: np.ndarray, simplices):
    """Per-triangle affine maps TARGET -> SOURCE, for inverse sampling."""
    transforms = []
    for triangle in simplices:
        t = target[triangle]
        s = source[triangle]
        matrix = np.column_stack([t, np.ones(3)])
        try:
            affine = np.linalg.solve(matrix, s)
        except np.linalg.LinAlgError as error:
            raise PiecewiseError(
                f"degenerate triangle {triangle.tolist()}: {error}"
            ) from error
        transforms.append(affine)
    return transforms


def warp_points(image: np.ndarray, source: np.ndarray,
                target: np.ndarray) -> np.ndarray:
    """Warp so each SOURCE control point lands on its TARGET.

    Inverse mapping over the Delaunay triangulation of the TARGET
    points: each output pixel inside the hull finds its triangle and
    reads its source position through that triangle's affine; pixels
    outside the hull are copied unchanged (identity), which is what
    makes the deformation local by construction. Vertices map exactly
    -- barycentric interpolation is exact at the corners -- so an
    all-anchor call is the identity map, asserted by the suite.
    """
    from scipy.spatial import Delaunay

    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise PiecewiseError(f"expected a 2-D or 3-D image, got {array.shape}")
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2:
        raise PiecewiseError(
            f"control shapes disagree: {source.shape} vs {target.shape}"
        )
    if np.allclose(source, target):
        return array.copy()

    triangulation = Delaunay(target)
    transforms = _triangle_affines(target, source, triangulation.simplices)

    height, width = array.shape[:2]
    rows, columns = np.mgrid[0:height, 0:width]
    grid = np.column_stack([columns.ravel() + 0.5, rows.ravel() + 0.5])
    simplex = triangulation.find_simplex(grid)

    out = array.copy()
    flat_out = out.reshape(height * width, *array.shape[2:])
    flat_in = array.reshape(height * width, *array.shape[2:])
    for index, affine in enumerate(transforms):
        inside = np.flatnonzero(simplex == index)
        if not len(inside):
            continue
        mapped = np.column_stack(
            [grid[inside], np.ones(len(inside))]
        ) @ affine
        cols = np.clip((mapped[:, 0] - 0.5).round().astype(int), 0, width - 1)
        rws = np.clip((mapped[:, 1] - 0.5).round().astype(int), 0, height - 1)
        flat_out[inside] = flat_in[rws * width + cols]
    return out


def forward_points(source: np.ndarray, target: np.ndarray,
                   points: np.ndarray) -> np.ndarray:
    """The FORWARD piecewise map at arbitrary points: triangulate the
    SOURCE control set, carry each query point through its triangle's
    source->target affine; points outside the hull are unmoved (the
    same locality-by-construction as the warp itself)."""
    from scipy.spatial import Delaunay

    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    queries = np.asarray(points, dtype=float)
    triangulation = Delaunay(source)
    transforms = _triangle_affines(source, target, triangulation.simplices)
    simplex = triangulation.find_simplex(queries)
    moved = queries.copy()
    for index, affine in enumerate(transforms):
        inside = np.flatnonzero(simplex == index)
        if len(inside):
            moved[inside] = np.column_stack(
                [queries[inside], np.ones(len(inside))]
            ) @ affine
    return moved


def deformation_summary(source, target, image, warped, points, box,
                        side: str, magnitude: float) -> dict:
    """The record-contract fields the task's post-artifact summary
    consumes, measured on THIS family's warp.

    **[2026-08-30, run p17-synth-pwa: the KeyError: 'identity' crash.]**
    The summary reads ``record["identity"]``,
    ``record["deformation"]["max_unintended_motion_frac_width"]`` and
    ``record["deformation"]["pixel_change"]["y_min"/"outside_box"]``
    from every non-identity record -- the contract
    ``synthesis.warp_content`` establishes. The first piecewise version
    recorded neither key, so the run completed all 5,499 faces, wrote
    and renamed the artifact, and died in the summary. The fields here
    mirror the TPS report's DEFINITIONS -- unintended motion is the
    contralateral, midline and box-corner movement through the FORWARD
    map, in fractions of crop width -- measured through this family's
    own map rather than copied from the spline's.
    """
    from . import landmarks as L

    array = np.asarray(points, dtype=float)
    width = box[2]
    indices = synthesis.target_indices(side)
    moved = forward_points(source, target, array)

    requested, realised = {}, {}
    for name, index in indices.items():
        dx, dy = synthesis.DIRECTIONS[name]
        want = np.array([(1.0 if side == "left" else -1.0) * dx, dy])
        want = want * synthesis.WEIGHTS[name] * magnitude * width
        got = moved[index] - array[index]
        requested[name] = float(np.hypot(*want))
        realised[name] = float(np.hypot(*got))

    other = 1 if side == "left" else 0
    contralateral = [
        L.ALAR_BASE[other], L.NOSTRIL_SILL[other],
        L.CUPIDS_BOW_PEAK[other], L.MOUTH_CORNERS[other],
    ]
    contra_motion = [
        float(np.hypot(*(moved[i] - array[i]))) / width for i in contralateral
    ]
    midline_motion = [
        float(np.hypot(*(moved[i] - array[i]))) / width
        for i in L.MIDLINE_LANDMARKS
    ]
    x, y, w, h = box
    corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
    corner_motion = [
        float(np.hypot(*(m - c))) / width
        for m, c in zip(forward_points(source, target, corners), corners)
    ]
    return {
        "requested_px": requested,
        "realised_px": realised,
        "contralateral_motion_frac_width": contra_motion,
        "midline_motion_frac_width": midline_motion,
        "corner_motion_frac_width": corner_motion,
        "max_unintended_motion_frac_width": max(
            contra_motion + midline_motion + corner_motion
        ),
        "pixel_change": synthesis.pixel_change_extent(image, warped, box),
    }


def warp_content(image: np.ndarray, points: np.ndarray, box, side: str,
                 magnitude: float):
    """``synthesis.warp_content``'s signature, piecewise-affine family.

    Same control points, same identity-at-zero contract, same
    (warped, record) return -- so ``task_asymmetry_synthesis`` switches
    families on one config key and everything downstream is shared.
    """
    source, target, detail = synthesis.control_points(
        points, box, side, magnitude
    )
    if magnitude == 0:
        return np.asarray(image).copy(), {
            "side": side,
            "magnitude": 0.0,
            "identity": True,
            "warp_family": "piecewise_affine",
            "targets": detail,
        }
    warped = warp_points(image, source, target)
    return warped, {
        "side": side,
        "magnitude": round(float(magnitude), 6),
        # The record CONTRACT the task's summary consumes -- the same
        # keys synthesis.warp_content writes (deformation_summary's
        # docstring carries the crash this fixed).
        "identity": False,
        "warp_family": "piecewise_affine",
        "targets": detail,
        "deformation": deformation_summary(
            source, target, image, warped, points, box, side, magnitude
        ),
    }


def displaced_control_points_for_tests() -> tuple[np.ndarray, np.ndarray]:
    """A small deterministic control set for the suite: a ring of
    anchors (undisplaced) around two displaced interior points."""
    anchors = np.array([
        [10.0, 10.0], [110.0, 10.0], [210.0, 10.0],
        [10.0, 110.0], [210.0, 110.0],
        [10.0, 210.0], [110.0, 210.0], [210.0, 210.0],
    ])
    moved_source = np.array([[90.0, 100.0], [130.0, 120.0]])
    moved_target = moved_source + np.array([[6.0, -4.0], [-5.0, 3.0]])
    source = np.vstack([moved_source, anchors])
    target = np.vstack([moved_target, anchors])
    return source, target
