"""MEBeauty's dlib-68 landmarks: the mapping, and the checks that own it.

**The mapping is [LITERATURE], not measured** (the ruling
2026-08-24). The indices below are the standard dlib-68 convention, and
this project's own record says what that is worth: ``scut.landmarks``
opens with "guessing landmark indices produces geometry that looks
plausible and is wrong", and one of ITS sub-group entries -- read off
renders rather than measured -- was wrong until the numbers caught it.

So the indices are a HYPOTHESIS the data must confirm. The three checks
in ``verify_mapping`` are SCUT's own, re-run on this dataset:

1. the midline landmarks sit on the midline (mirror-pair |x| ~ 0),
2. the two eye groups' centres are symmetric about it,
3. the proposed mouth corners are the WIDEST symmetric pair at corner
   height.

**The direction of accommodation is the failure mode.** If a check
fails, the MAPPING reopens -- never the check. A threshold loosened
until the indices pass is a mapping that was never verified, wearing a
verification's name. The thresholds are declared in the config before
the run for exactly that reason.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: The dlib-68 groups. **[LITERATURE 2026-08-24, the ruling --
#: the standard convention, NOT measured here.]** Every one is checked
#: against the data by ``verify_mapping`` before any pixel is staged.
N_LANDMARKS = 68

#: **THE IMAGE-LEFT / SUBJECT-RIGHT TRAP, NAMED.** dlib's "left eye" is
#: the SUBJECT's left, which appears on the IMAGE's RIGHT in a frontal
#: photograph. ``scut.placement``'s ``EYE_LEFT`` means IMAGE-left. So
#: dlib 36-41 (subject-right eye) is the IMAGE-LEFT group and pairs with
#: SCUT's ``EYE_LEFT``. Getting this backwards mirrors every derived
#: quantity while leaving every shape plausible -- which is why it is
#: written here and asserted in ``verify_mapping`` rather than trusted.
EYE_IMAGE_LEFT = (36, 37, 38, 39, 40, 41)
EYE_IMAGE_RIGHT = (42, 43, 44, 45, 46, 47)

#: The midline pair, mirroring SCUT's (SUBNASALE, CUPIDS_BOW_DIP).
SUBNASALE = 33
CUPIDS_BOW_DIP = 51
MIDLINE_LANDMARKS = (SUBNASALE, CUPIDS_BOW_DIP)

#: The vertical span: brows to the outer mouth contour, the same two
#: structures SCUT's BROW_INDICES/MOUTH_INDICES define it from.
BROW_INDICES = tuple(range(17, 27))
MOUTH_OUTER_INDICES = tuple(range(48, 60))

#: The proposed corners, image-left first (SCUT's MOUTH_CORNERS order).
MOUTH_CORNERS = (48, 54)

#: Mirror pairs used to estimate the face's own midline without assuming
#: any single landmark is on it -- the unbiased estimator SCUT's
#: ``placement.midline_x`` correction established.
MIRROR_PAIRS = (
    (36, 45), (39, 42), (37, 44), (38, 43), (41, 46), (40, 47),
    (31, 35), (32, 34), (48, 54), (49, 53), (50, 52), (59, 55), (58, 56),
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),
)


class MappingError(RuntimeError):
    """The dlib-68 mapping did not survive its checks."""


def mirror_pair_midline(points: np.ndarray) -> float:
    """The face's midline x, from mirror pairs only.

    Never a single landmark and never the image centre: the estimator
    is unbiased with respect to any one index being mislabelled, which
    is the property the checks need in order to test the midline
    landmarks THEMSELVES.
    """
    array = np.asarray(points, dtype=float)
    centres = [
        (array[a][0] + array[b][0]) / 2.0 for a, b in MIRROR_PAIRS
    ]
    return float(np.median(centres))


def face_width(points: np.ndarray) -> float:
    array = np.asarray(points, dtype=float)
    return float(array[:, 0].max() - array[:, 0].min())


def verify_mapping(
    points_by_key: dict,
    *,
    midline_tolerance: float,
    eye_symmetry_tolerance: float,
    corner_margin: float,
) -> dict:
    """Run SCUT's three checks over every face. Report, never adjust.

    Returns the measured report. The CALLER decides pass/fail against
    the DECLARED thresholds; this function reports what it measured and
    whether each check cleared the threshold it was given. Nothing here
    tunes anything.
    """
    midline_offsets = {index: [] for index in MIDLINE_LANDMARKS}
    eye_asymmetries = []
    corner_is_widest = []
    corner_gaps = []

    for points in points_by_key.values():
        array = np.asarray(points, dtype=float)
        if array.shape[0] != N_LANDMARKS:
            raise MappingError(
                f"{array.shape[0]} landmarks, expected {N_LANDMARKS} -- "
                "this is not a dlib-68 set"
            )
        width = face_width(array)
        if width <= 0:
            continue
        midline = mirror_pair_midline(array)

        # Check 1: the midline landmarks sit on the midline.
        for index in MIDLINE_LANDMARKS:
            midline_offsets[index].append(
                abs(array[index][0] - midline) / width
            )

        # Check 2: the eye groups' centres are symmetric about it. The
        # IMAGE-LEFT group must sit on the image-left side -- the trap,
        # asserted rather than assumed.
        left = array[list(EYE_IMAGE_LEFT)].mean(axis=0)
        right = array[list(EYE_IMAGE_RIGHT)].mean(axis=0)
        eye_asymmetries.append({
            "imbalance": abs(
                (midline - left[0]) - (right[0] - midline)
            ) / width,
            "left_is_image_left": bool(left[0] < right[0]),
        })

        # Check 3: the proposed corners are the widest symmetric pair at
        # corner height -- SCUT's own criterion, which caught its wrong
        # index. Every other symmetric mouth pair must be narrower.
        a, b = MOUTH_CORNERS
        proposed = abs(array[b][0] - array[a][0])
        others = [
            abs(array[q][0] - array[p][0])
            for p, q in ((49, 53), (50, 52), (59, 55), (58, 56), (61, 63))
        ]
        widest_other = max(others) if others else 0.0
        corner_is_widest.append(proposed >= widest_other * (1 + corner_margin))
        corner_gaps.append((proposed - widest_other) / width)

    n = len(eye_asymmetries)
    if not n:
        raise MappingError("no usable faces: nothing was verified")

    def _summarise(values):
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "max": float(array.max()),
        }

    imbalance = [entry["imbalance"] for entry in eye_asymmetries]
    report = {
        "n_faces": n,
        "check_1_midline": {
            str(index): _summarise(values)
            for index, values in midline_offsets.items()
        },
        "check_1_passes": all(
            float(np.median(values)) <= midline_tolerance
            for values in midline_offsets.values()
        ),
        "check_2_eye_symmetry": _summarise(imbalance),
        "check_2_image_left_fraction": float(
            np.mean([e["left_is_image_left"] for e in eye_asymmetries])
        ),
        "check_2_passes": bool(
            float(np.median(imbalance)) <= eye_symmetry_tolerance
            and np.mean([e["left_is_image_left"] for e in eye_asymmetries])
            > 0.99
        ),
        "check_3_corner_widest_fraction": float(np.mean(corner_is_widest)),
        "check_3_gap": _summarise(corner_gaps),
        "check_3_passes": bool(np.mean(corner_is_widest) > 0.99),
        "thresholds_declared": {
            "midline_tolerance": midline_tolerance,
            "eye_symmetry_tolerance": eye_symmetry_tolerance,
            "corner_margin": corner_margin,
        },
        "the_rule": (
            "if a check fails the MAPPING reopens -- never the check. A "
            "threshold loosened until the indices pass is a mapping "
            "that was never verified"
        ),
    }
    report["all_pass"] = bool(
        report["check_1_passes"]
        and report["check_2_passes"]
        and report["check_3_passes"]
    )
    return report


# --------------------------------------------------------------------------
# the landmark-quality screen (phase15.STOP_2A_II_BUILT)
# --------------------------------------------------------------------------

#: The five checks, in the order they are applied. ``degenerate`` is a
#: HARD REFUSAL and short-circuits: a row with no usable geometry cannot
#: meaningfully fail the other four, and reporting four derived NaNs
#: beside it would read as four separate problems.
QUALITY_CHECKS = ("degenerate", "bounds", "span", "ordering", "interocular")


def bounds_overshoot(points: np.ndarray, image_size: tuple) -> dict:
    """How far outside its image each point falls -- the RAW geometry,
    independent of any tolerance (phase15.BOUNDS_VIOLATION_READINGS).

    Reported so the violation can be seen as a DISTRIBUTION rather than
    a verdict: a set whose jaw grazes the border by two pixels and a set
    computed in a foreign coordinate frame both "fail bounds", and only
    the magnitudes tell them apart.
    """
    array = np.asarray(points, dtype=float)
    width, height = float(image_size[0]), float(image_size[1])
    left = np.maximum(0.0, -array[:, 0])
    right = np.maximum(0.0, array[:, 0] - width)
    top = np.maximum(0.0, -array[:, 1])
    bottom = np.maximum(0.0, array[:, 1] - height)
    pixels = np.maximum(np.maximum(left, right), np.maximum(top, bottom))
    fractions = np.maximum(
        np.maximum(left / width, right / width),
        np.maximum(top / height, bottom / height),
    ) if width > 0 and height > 0 else np.zeros_like(pixels)
    outside = pixels > 0
    n_outside = int(outside.sum())
    return {
        "n_points_outside": n_outside,
        "n_points": int(array.shape[0]),
        "overshoot_px_max": float(pixels.max()) if n_outside else 0.0,
        "overshoot_px_median": (
            float(np.median(pixels[outside])) if n_outside else 0.0
        ),
        "overshoot_fraction_max": (
            float(fractions.max()) if n_outside else 0.0
        ),
        "overshoot_fraction_median": (
            float(np.median(fractions[outside])) if n_outside else 0.0
        ),
    }


def quality_screen(
    points: np.ndarray,
    image_size: tuple,
    *,
    span_band: tuple,
    interocular_band: tuple,
    bounds_margin: float,
    bounds_max_points_outside: int = 0,
) -> dict:
    """Is this face's landmark set USABLE GEOMETRY, or a detector failure?

    **[2026-08-24, phase15.STOP_2A_II_BUILT]** The pose screen is blind
    to this: a misplaced-but-SYMMETRIC cloud passes a centroid-offset
    test happily, so faces whose detector failed or half-failed reached
    the accepted set and were caught only by eye on the mapping sheets.
    These checks test what the eye tested.

    Every threshold ARRIVES from the caller (declared in the config
    before the run); nothing here is computed from the data. The rule is
    2a's rule unchanged: a face that fails is REJECTED -- the check is
    never loosened to keep it.

    ``image_size`` is ``(width, height)``, read from the image HEADER --
    the verdict never decodes a pixel.
    """
    array = np.asarray(points, dtype=float)
    width, height = float(image_size[0]), float(image_size[1])
    report: dict = {"checks": {}}

    # 1. Degenerate: no finite geometry at all. Hard refusal.
    finite = bool(np.isfinite(array).all())
    span_x = float(array[:, 0].max() - array[:, 0].min()) if finite else 0.0
    span_y = float(array[:, 1].max() - array[:, 1].min()) if finite else 0.0
    degenerate = not finite or span_x <= 0 or span_y <= 0
    report["checks"]["degenerate"] = {
        "passes": not degenerate,
        "all_finite": finite,
        "span_x": span_x,
        "span_y": span_y,
    }
    report["span_x"], report["span_y"] = span_x, span_y
    if degenerate:
        report["passes"] = False
        report["failed"] = ["degenerate"]
        report["reason"] = (
            "no usable geometry: the row is empty, non-finite or "
            "collapsed to a point"
        )
        return report

    # 2. Bounds: the constellation must lie ON the image it claims.
    #    A set computed in a DIFFERENT coordinate frame -- a crop, a
    #    resize, another file of the same name -- lands outside, and
    #    that is the signature this check exists to catch.
    margin_x, margin_y = bounds_margin * width, bounds_margin * height
    outside = (
        (array[:, 0] < -margin_x) | (array[:, 0] > width + margin_x)
        | (array[:, 1] < -margin_y) | (array[:, 1] > height + margin_y)
    )
    # **[2026-08-24] Two declared tolerances, both strict by default**
    # (bounds_margin 0.0, bounds_max_points_outside 0), so the shipped
    # behaviour is unchanged until a value is set from a MEASURED
    # distribution (phase15.BOUNDS_VIOLATION_READINGS). The raw
    # violation geometry is reported either way.
    report["checks"]["bounds"] = {
        "passes": bool(
            int(outside.sum()) <= int(bounds_max_points_outside)
        ),
        "n_outside": int(outside.sum()),
        "max_points_outside_allowed": int(bounds_max_points_outside),
        "violation": bounds_overshoot(array, image_size),
        "worst_x_excursion": float(
            max(
                (-array[:, 0]).max() / width if width else 0.0,
                (array[:, 0] - width).max() / width if width else 0.0,
            )
        ),
        "worst_y_excursion": float(
            max(
                (-array[:, 1]).max() / height if height else 0.0,
                (array[:, 1] - height).max() / height if height else 0.0,
            )
        ),
    }

    # 3. Span: a face fills a plausible fraction of its own image. Too
    #    small is a collapsed detection or a foreign coordinate frame;
    #    too large is a set that is not a face.
    fraction_x = span_x / width if width else float("inf")
    fraction_y = span_y / height if height else float("inf")
    low, high = span_band
    report["checks"]["span"] = {
        "passes": bool(
            low <= fraction_x <= high and low <= fraction_y <= high
        ),
        "fraction_x": fraction_x,
        "fraction_y": fraction_y,
        "band": [low, high],
    }

    # 4. Ordering: brows above eyes above nose-base above mouth. A
    #    scattered or shuffled constellation breaks it while keeping any
    #    symmetric statistic intact.
    brow_y = float(np.median(array[list(BROW_INDICES), 1]))
    eye_y = float(np.median(
        array[list(EYE_IMAGE_LEFT) + list(EYE_IMAGE_RIGHT), 1]
    ))
    nose_y = float(array[SUBNASALE, 1])
    mouth_y = float(np.median(array[list(MOUTH_OUTER_INDICES), 1]))
    report["checks"]["ordering"] = {
        "passes": bool(brow_y < eye_y < nose_y < mouth_y),
        "brow_y": brow_y,
        "eye_y": eye_y,
        "nose_y": nose_y,
        "mouth_y": mouth_y,
    }

    # 5. Inter-ocular distance, as a fraction of the face's own span:
    #    scale-free, so it tests the CONSTELLATION's shape rather than
    #    the image's size.
    left = array[list(EYE_IMAGE_LEFT)].mean(axis=0)
    right = array[list(EYE_IMAGE_RIGHT)].mean(axis=0)
    interocular = float(np.hypot(*(right - left)) / span_x)
    io_low, io_high = interocular_band
    report["checks"]["interocular"] = {
        "passes": bool(io_low <= interocular <= io_high),
        "interocular_over_span": interocular,
        "band": [io_low, io_high],
    }

    failed = [
        name for name in QUALITY_CHECKS
        if name in report["checks"] and not report["checks"][name]["passes"]
    ]
    report["failed"] = failed
    report["passes"] = not failed
    return report


# --------------------------------------------------------------------------
# placement: MEBeauty's anchors, SCUT's formula (phase15.STOP_2B_BUILT)
# --------------------------------------------------------------------------


def vertical_span(points: np.ndarray, lower_margin: float | None = None):
    """(top, bottom): brow to just below the lower vermillion, dlib-68.

    The same two structures SCUT's ``vertical_span`` uses -- brows and
    the outer mouth contour -- read from the dlib groups.
    ``LOWER_MARGIN`` is INHERITED from ``scut.placement``, never
    restated: "just below the lower vermillion" is one decision and it
    was made once (0.06, reviewed on the placement contact sheet).
    """
    from .scut import placement

    margin = placement.LOWER_MARGIN if lower_margin is None else lower_margin
    array = np.asarray(points, dtype=float)
    top = float(array[list(BROW_INDICES), 1].min())
    lip_bottom = float(array[list(MOUTH_OUTER_INDICES), 1].max())
    span = lip_bottom - top
    if span <= 0:
        raise MappingError(
            f"brow at y={top} is not above the lower lip at y={lip_bottom}: "
            "the landmark groups are inverted for this face"
        )
    return top, lip_bottom + margin * span


def midline_x(points: np.ndarray) -> float:
    """The face's own midline, dlib-68.

    SCUT's estimator, transposed: the eye-centre midpoint averaged with
    the two midline landmarks (33, 51), so the eyes fix the upper face
    and the midline points fix it lower down, and no single mislabelled
    index can move it far. The mirror-pair estimator is used for the
    VERIFICATION checks; this one mirrors what SCUT stages with.
    """
    array = np.asarray(points, dtype=float)
    left = array[list(EYE_IMAGE_LEFT)].mean(axis=0)
    right = array[list(EYE_IMAGE_RIGHT)].mean(axis=0)
    eye_midpoint = (left[0] + right[0]) / 2.0
    return float(np.mean([
        eye_midpoint,
        array[SUBNASALE, 0],
        array[CUPIDS_BOW_DIP, 0],
    ]))


def crop_box(points: np.ndarray, aspect_ratio: float, lower_margin=None):
    """The content box, from MEBeauty's anchors and SCUT's formula."""
    from .scut import placement

    top, bottom = vertical_span(points, lower_margin)
    return placement.box_from_anchors(
        top, bottom, midline_x(points), aspect_ratio
    )


# --------------------------------------------------------------------------
# the comparative verdict (phase15.STOP_3_AMENDED)
# --------------------------------------------------------------------------

#: Phase 6's banked SCUT cells, by (init, geometry) -- the comparators,
#: read from the ladder rather than restated here.
def banked_cell(init: str, geometry: str) -> float:
    """The banked cleft-side PCC for a SCUT/ImageNet cell."""
    from . import ladder

    imagenet, scut_original, scut_masked_g1 = ladder.STAGE_D1_AT_G1[
        "cells"
    ]["vit_b16"]
    cells = {
        ("imagenet", "g1"): imagenet,
        ("scut_original", "g1"): scut_original,
        ("scut_masked", "g1"): scut_masked_g1,
        ("scut_masked", "g2"): ladder.STAGE_G_LABEL_FORMULATION[
            "triples"
        ]["scut_masked__g2"]["arms"]["mean"]["pcc"],
    }
    if (init, geometry) not in cells:
        raise MappingError(
            f"no banked cell for {init!r} at {geometry!r}; the "
            f"comparators are {sorted(cells)}"
        )
    return cells[(init, geometry)]


def verdict_delta(
    mine: dict, comparator_init: str, comparator_geometry: str,
) -> dict:
    """One MEBeauty arm against its SAME-NAMED, SAME-GEOMETRY cell.

    **[BINDING 2026-08-24, the amendment 1 -- ASSERTED HERE, NOT
    ONLY IN PROSE]** Masked SCUT sits at 0.0830 at G1 and 0.2001 at G2:
    a 0.117 spread LARGER THAN ANY PLAUSIBLE DATASET EFFECT. A pooled
    or crossed verdict would report a GEOMETRY difference wearing a
    DATASET's name, so a cross-geometry pair is REFUSED rather than
    computed.

    ``mine`` is ``{"geometry": ..., "pcc": ..., "sd": ...}``. The
    threshold is the two arms' own five-seed SDs (PLAN 4.12.1,
    inherited never), and the caller declares it in the config before
    any number exists.
    """
    if mine["geometry"] != comparator_geometry:
        raise MappingError(
            f"REFUSED: {mine['geometry']!r} against a "
            f"{comparator_geometry!r} comparator. The masked cells "
            "differ by 0.117 ACROSS GEOMETRY -- larger than any "
            "plausible dataset effect -- so a crossed verdict would "
            "report a geometry difference wearing a dataset's name "
            "(phase15.STOP_3_AMENDED, amendment 1)"
        )
    reference = banked_cell(comparator_init, comparator_geometry)
    imagenet = banked_cell("imagenet", "g1")
    return {
        "geometry": mine["geometry"],
        "comparator": f"{comparator_init}@{comparator_geometry}",
        "mine_pcc": float(mine["pcc"]),
        "comparator_pcc": reference,
        "delta_vs_comparator": round(float(mine["pcc"]) - reference, 6),
        # Amendment 3: the ImageNet anchor rides in EVERY reading.
        "imagenet_pcc": imagenet,
        "delta_vs_imagenet": round(float(mine["pcc"]) - imagenet, 6),
        "the_imagenet_anchor": (
            "an arm that beats its SCUT cell while still losing to "
            "ImageNet is NOT a win: the ladder's finding is that SCUT "
            "pretraining is worse than no beauty pretraining at all"
        ),
    }


# --------------------------------------------------------------------------
# MEBeauty's OWN set namespace (phase15.STOP_4_PROPOSED)
# --------------------------------------------------------------------------

#: **[RULED 2026-08-24, the maintainer] MEBeauty's inits live HERE, not in
#: ``embeddings.INITS``.** Phase 6's lattice is closed and its 24/12
#: arithmetic describes THAT lattice; adding a fourth and fifth init to
#: it would silently change what a completed phase's plan means, and an
#: undeclared artifact swap manufacturing factor effects is exactly
#: ``ladder.MASKED_G1_ARTEFACT``'s shape. The two lattices are never
#: pooled: a table carrying cells from both must say which is which.
MEBEAUTY_INITS = ("mebeauty_masked", "mebeauty_original")

#: Which checkpoint each init consumes. ``mebeauty_masked`` is
#: GEOMETRY-BOUND for the same reason SCUT's is -- the masked variant
#: is staged per geometry, so a G1 checkpoint feeding G2 embeddings
#: would still run, still produce 237 vectors, and still fit a ridge.
VARIANT_FOR_MEBEAUTY_INIT = {
    "mebeauty_masked": "masked_{geometry}",
    "mebeauty_original": "masked_original",
}


def expected_mebeauty_variant(init: str, geometry: str) -> str:
    """Which pretraining source this (init, geometry) pair must come
    from -- MEBeauty's own table, never the ladder's."""
    if init not in MEBEAUTY_INITS:
        raise MappingError(
            f"unknown MEBeauty init {init!r}; expected one of "
            f"{MEBEAUTY_INITS}. The ladder's inits live in "
            "embeddings.INITS and are NOT interchangeable with these"
        )
    template = VARIANT_FOR_MEBEAUTY_INIT[init]
    if "{geometry}" not in template:
        return template
    from .embeddings import GEOMETRIES

    if geometry not in GEOMETRIES:
        raise MappingError(
            f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}"
        )
    return template.format(geometry=geometry)


def mebeauty_set_count() -> dict:
    """MEBeauty's OWN count arithmetic, derived rather than stated.

    Three sets from three checkpoints over one backbone -- and the
    number is computed from the tables above so it follows the rule if
    the rule changes, the same discipline
    ``embeddings.expected_set_count`` applies to the ladder's 24/12.
    **These counts are never added to the ladder's.**
    """
    from .embeddings import GEOMETRIES

    variants = set()
    for init in MEBEAUTY_INITS:
        template = VARIANT_FOR_MEBEAUTY_INIT[init]
        if "{geometry}" in template:
            variants.update(
                template.format(geometry=g) for g in GEOMETRIES
            )
        else:
            variants.add(template)
    return {
        "backbones": 1,
        "inits": len(MEBEAUTY_INITS),
        "variants": sorted(variants),
        "embedding_sets": len(variants),
        "pretraining_runs": len(variants),
        "never_pooled_with": (
            "embeddings.expected_set_count's 24/12 -- that arithmetic "
            "describes Phase 6's lattice and this one describes "
            "MEBeauty's; a table carrying both must say which cell "
            "came from which (phase15.STOP_4_PROPOSED)"
        ),
    }


def assert_run_is_the_declared_arm(metrics: dict, declared_source: str) -> str:
    """The pretraining run's OWN record must name the arm the config
    declares -- or the extraction refuses.

    **[2026-08-24, phase15.STOP_3_PRETRAINING_BANKED]** The
    arm-to-run mapping could not be confirmed on the laptop (no
    cluster access), so the confirmation moves into the code that
    CONSUMES the checkpoint. A swapped mapping fails here rather than
    propagating into a verdict wearing the wrong arm's name -- the
    checkpoint-identity lesson at the one place it can still be
    caught.
    """
    recorded = metrics.get("source")
    if recorded != declared_source:
        raise MappingError(
            f"the declared arm is {declared_source!r} but the run's own "
            f"metrics.json records source {recorded!r}. The mapping "
            "between arms and run directories is ASSERTED, never "
            "trusted (phase15.STOP_4_PROPOSED)"
        )
    return recorded


# --------------------------------------------------------------------------
# MEBeauty's own set I/O -- the ladder's writer is not widened
# --------------------------------------------------------------------------


def save_set(directory, values, *, init: str, geometry: str,
             variant: str, checkpoint_sha256, patient_ids, manifest_ids,
             run_name: str) -> dict:
    """Write one MEBeauty cleft-side embedding set.

    **Its OWN writer, because ``embeddings.save`` validates against the
    LADDER's ``INITS`` and widening that constant is what the maintainer's
    ruling forbids** (phase15.STOP_4_PROPOSED). What is NOT duplicated
    is the property that matters: row order is checked with the
    ladder's own ``assert_row_order``, whose docstring says why -- a
    silent misalignment trains every patient against another patient's
    label and still produces a plausible number.
    """
    import json

    import numpy as np

    from .embeddings import assert_row_order
    from .provenance import atomic_write_text
    from .provenance.hashing import hash_dir

    directory = Path(directory)
    if directory.exists():
        raise MappingError(
            f"{directory} already exists. Data artifacts are immutable: "
            "create a new version, never overwrite"
        )
    if init not in MEBEAUTY_INITS:
        raise MappingError(
            f"unknown MEBeauty init {init!r}; this writer serves "
            f"{MEBEAUTY_INITS} and NOT the ladder's embeddings.INITS"
        )
    if variant != expected_mebeauty_variant(init, geometry):
        raise MappingError(
            f"init {init!r} at {geometry!r} must come from variant "
            f"{expected_mebeauty_variant(init, geometry)!r}, not "
            f"{variant!r} -- the geometry-bound checkpoint rule"
        )
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise MappingError(f"values {values.shape}: expected (n, dim)")
    assert_row_order(list(patient_ids), list(manifest_ids))
    if len(patient_ids) != values.shape[0]:
        raise MappingError(
            f"{len(patient_ids)} patient ids for {values.shape[0]} rows"
        )

    directory.mkdir(parents=True)
    np.save(directory / "values.npy", values)
    atomic_write_text(
        directory / "metadata.json",
        json.dumps({
            "namespace": "mebeauty",
            "kind": "pooled",
            "backbone": "vit_b16",
            "init": init,
            "geometry": geometry,
            "variant": variant,
            "checkpoint_sha256": checkpoint_sha256,
            "patient_ids": [int(p) for p in patient_ids],
            "n_patients": int(values.shape[0]),
            "feature_dim": int(values.shape[1]),
            "row_order_checked_against_manifest": True,
            "generating_run": run_name,
            "never_pooled_with": (
                "the ladder's embedding sets: this set's init is in "
                "mebeauty.MEBEAUTY_INITS, not embeddings.INITS, and a "
                "table carrying both must say which lattice each cell "
                "came from (phase15.STOP_4_PROPOSED)"
            ),
        }, indent=2, sort_keys=True) + "\n",
    )
    payload = hash_dir(directory)
    atomic_write_text(
        directory / "MANIFEST.json",
        json.dumps({
            "artifact": directory.name,
            "payload_rollup": payload["rollup"],
            "payload_files": payload["files"],
            "generating_run": run_name,
        }, indent=2, sort_keys=True) + "\n",
    )
    return payload


def load_set(directory, manifest_ids=None):
    """Read what ``save_set`` wrote, re-checking it.

    The invariants are re-asserted rather than trusted: an artifact can
    be correct when written and wrong when read, and this is the last
    point at which anything checks before a number comes out.
    """
    import json

    import numpy as np

    from .embeddings import assert_row_order

    directory = Path(directory)
    metadata = json.loads(
        (directory / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("namespace") != "mebeauty":
        raise MappingError(
            f"{directory} is not a MEBeauty set (namespace "
            f"{metadata.get('namespace')!r}) -- the ladder's sets are "
            "read by embeddings.load, and the two are never crossed"
        )
    values = np.load(directory / "values.npy")
    if values.shape[0] != metadata["n_patients"]:
        raise MappingError(
            f"{values.shape[0]} rows against metadata's "
            f"{metadata['n_patients']}: the artifact was edited"
        )
    if metadata["variant"] != expected_mebeauty_variant(
        metadata["init"], metadata["geometry"]
    ):
        raise MappingError(
            f"stored variant {metadata['variant']!r} is not the one "
            f"{metadata['init']!r} at {metadata['geometry']!r} must "
            "come from -- a crossed checkpoint"
        )
    if manifest_ids is not None:
        assert_row_order(metadata["patient_ids"], list(manifest_ids))
    return values, metadata
