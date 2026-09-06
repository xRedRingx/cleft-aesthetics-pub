"""Road B Phase 2: staging at ten settings, and the gates each one needs.

**The per-setting gate is the PIXEL residual, and it is new work.**
``trapezium.asymmetry_is_preserved`` takes no size argument, so it runs ONCE
as the analytic check; running it eight times would be eight identical passes
(``roadb.ASYMMETRY_GATE_IS_RESOLUTION_BLIND``). What varies with resolution is
the resampling in ``trapezium.unwarp``, and that has never been gated at any
size.

**And the gate is on the ORDERING, not on a per-cell threshold.** The residual
is expected to shrink with resolution, so a setting whose residual RISES is a
defect rather than noise -- and a tolerance each cell passes individually
would hide a path that gets worse where it should get better. Measured shape:
3.29e-03 at 224, 1.40e-03 at 512, 9.14e-04 at 768.
"""

from __future__ import annotations

import numpy as np

from . import roadb


class StagingError(RuntimeError):
    """A Road B staging gate refused."""


#: Where the probe mark is placed: half-way down, at 10% of the local
#: half-width left of the midline. Off-midline because a mark ON the midline
#: is symmetric and would pass any map, including a broken one -- the same
#: reason ``asymmetry_at`` uses offsets rather than a single point.
PROBE_HEIGHT = 0.5
PROBE_OFFSET_FRACTION = 0.10

#: ViT-B/16's patch size. A staged non-square image must be a multiple of it
#: in BOTH dimensions or the backbone refuses the input outright.
PATCH = 16


def pixel_asymmetry_residual(size: int, *, height: float = PROBE_HEIGHT,
                             offset_fraction: float = PROBE_OFFSET_FRACTION
                             ) -> float:
    """How far the unwarped mark lands from where the analytic map says.

    This is the quantity that varies with resolution. It is resampling
    quantisation rather than a defect -- but it is **six orders of magnitude
    looser than the analytic 1e-9**, which is why "verified to machine
    epsilon" describes the map and not the staged pixels.
    """
    from .geometry import trapezium

    image = np.zeros((size, size, 3), dtype=np.uint8)
    row = int(round(height * size))
    row = min(max(row, 1), size - 2)
    half = trapezium.DEFAULT.half_width_at(height)
    column = int(round((trapezium.MIDLINE - offset_fraction * half) * size))
    image[row - 1:row + 2, column - 1:column + 2] = 255

    out = trapezium.unwarp(image)
    columns = np.where(out[row].max(axis=-1) > 0)[0]
    if not columns.size:
        raise StagingError(
            f"the probe mark vanished under unwarp at size {size}; the "
            "residual cannot be measured, which is a failure not a zero"
        )
    centre = (columns.mean() + 0.5) / out.shape[1]
    return float(abs(centre - trapezium.unwarp_x(column / size, height)))


def assert_residual_monotone(residuals: dict) -> dict:
    """**The gate: the residual must SHRINK as resolution rises.**

    A per-cell tolerance is not enough. Each cell could sit under its own
    threshold while the path gets worse where it should get better, and that
    is a defect a threshold cannot see -- the same shape as a claim that
    clears its bar while the trend runs the wrong way.

    Asserted on the ORDERING rather than on the three measured numbers, so the
    gate does not encode this machine's arithmetic.
    """
    sizes = sorted(residuals)
    if len(sizes) < 2:
        raise StagingError(
            f"monotonicity needs at least two resolutions, got {sizes}"
        )
    rising = [
        (small, large) for small, large in zip(sizes, sizes[1:])
        if residuals[large] >= residuals[small]
    ]
    if rising:
        detail = "; ".join(
            f"{small}->{large}: {residuals[small]:.3e} -> {residuals[large]:.3e}"
            for small, large in rising
        )
        raise StagingError(
            f"the pixel residual rises with resolution at {detail}. It is "
            "resampling quantisation and must shrink; a rise is a defect in "
            "the staging path, not noise"
        )
    return {
        "residuals": {int(k): float(v) for k, v in residuals.items()},
        "monotone": True,
        "gate": "the residual shrinks as resolution rises",
        "analytic_tolerance": 1e-9,
        "note": (
            "six orders looser than the analytic map's tolerance -- 'machine "
            "epsilon' describes the map, not these pixels"
        ),
    }


def residual_families(observed: dict) -> dict:
    """Group measured residuals into ``(aspect, geometry)`` families.

    ``observed`` is ``{setting name: residual}``. Road A's 224 is folded into
    both SQUARE families from ``roadb.ROAD_A_224_PIXEL_RESIDUAL`` -- it is a
    square staging, so it is the third point there and belongs to neither
    non-square family.

    **The ordering is within a family because the residual tracks column
    count** (``roadb.RESIDUAL_FAMILIES``): non-square 512 has ~379 columns and
    a larger residual than square 512, which is fewer columns rather than a
    defect.
    """
    settings = {s["name"]: s for s in roadb.staging_settings()}
    unknown = sorted(set(observed) - set(settings))
    if unknown:
        raise StagingError(f"not Road B staging settings: {unknown}")

    families: dict[tuple, dict] = {}
    for name, residual in observed.items():
        setting = settings[name]
        key = (setting["aspect"], setting["geometry"])
        families.setdefault(key, {})[setting["resolution"]] = float(residual)

    road_a = roadb.ROAD_A_224_PIXEL_RESIDUAL
    for key in families:
        if key[0] == "square":
            families[key][road_a["size"]] = float(road_a["residual"])
    return {"/".join(key): value for key, value in sorted(families.items())}


def assert_residual_monotone_by_family(observed: dict) -> dict:
    """The gate, per family. Every family's residual must shrink with size.

    Reported per family rather than pooled: a pooled ordering would mix
    column counts and call a non-square setting a defect for having fewer.
    """
    families = residual_families(observed)
    report, failures = {}, []
    for name, residuals in families.items():
        try:
            report[name] = assert_residual_monotone(residuals)
            report[name]["n_points"] = len(residuals)
        except StagingError as error:
            failures.append(f"{name}: {error}")
            report[name] = {
                "residuals": residuals, "monotone": False,
                "n_points": len(residuals), "error": str(error),
            }
    if failures:
        raise StagingError(
            f"{len(failures)} of {len(families)} families are not monotone in "
            "resolution: " + " | ".join(failures)
        )
    return {
        "families": report,
        "n_families": len(report),
        "three_point_families": sorted(
            name for name, entry in report.items() if entry["n_points"] >= 3
        ),
        "road_a_224_from": "roadb.ROAD_A_224_PIXEL_RESIDUAL",
    }


def content_fraction(shape, geometry: str, content_box=None) -> float:
    """Fraction of the FRAME that is content, for THIS setting.

    Reported per setting so the geometry axis is visible in the artifact
    rather than inferred -- and so a later phase reads the saliency
    expectation instead of assuming one. 0.1989 non-content at G1 non-square
    and 0.0000 at G2 are different baselines.

    ``shape`` is an int for a square frame or ``(height, width)`` for a
    non-square one. The box defaults to the whole frame, which is the
    non-square semantics: there, the content box IS the image.

    **[CORRECTED 2026-08-09] This used to average over a square canvas the
    non-square artifact does not contain.** A (224, 160) staged image scored
    the trapezium fraction diluted by 64 columns of imaginary pad -- ~0.57
    against the 0.8011 the axis records -- and a later phase reading it as a
    saliency expectation would have inherited the wrong-region kind of wrong
    that phase8's framing finding came from. The mean now runs over the frame
    the backbone sees, whatever its shape.
    """
    import numbers

    from .train.augment_sheet import content_mask

    if isinstance(shape, numbers.Integral):
        height = width = int(shape)
    else:
        height, width = (int(v) for v in shape[:2])
    box = content_box if content_box is not None else (0, 0, width, height)
    mask = content_mask(max(height, width), box, geometry)
    return float(mask[:height, :width].mean())


def upsampling_flags(longer_sides, target: int) -> dict:
    """Which patients are interpolated at this target, per patient.

    **The 768 arm may mix two populations** -- interpolated and downsampled
    -- so it is reported BOTH ways when any patient interpolates: all
    patients, and restricted to those that genuinely downsample. Without the
    per-patient flag the trend is partly about interpolation and nothing
    says so.

    **[CORRECTED 2026-08-09, sheet review] The argument is the LONGER side.**
    Both staging paths compute ``scale = target / max(width, height)``, so
    content is interpolated iff the longer side is under the target. The
    first version took the shorter side and over-counted -- the review's 41
    flagged patients looked sharp because most of them downsample.
    ``roadb.UPSAMPLING_FLAG_MEASURED_THE_WRONG_SIDE``.
    """
    sides = np.asarray(longer_sides, dtype=float)
    upsampled = sides < target
    return {
        "target": int(target),
        "n": int(sides.size),
        "n_upsampled": int(upsampled.sum()),
        "fraction_upsampled": float(upsampled.mean()) if sides.size else 0.0,
        "per_patient": [bool(v) for v in upsampled],
        "report_both_ways": bool(upsampled.any()),
        # Which quantity these flags are, named in the artifact -- a v1
        # manifest (shorter-side flags) and a later one must not be read as
        # the same quantity.
        "measures": (
            "longer side of the source against the target -- staging scales "
            "by the longer side, so that is the side interpolation follows"
        ),
        # R2's population trap, named where the number lives: the brief's
        # ceilings were measured over all 473 sources, frontal AND basal, by
        # the shorter side; a staging run counts the manifest's frontal rows.
        "population": (
            "the staged manifest rows -- the brief's ceilings are over all "
            f"{roadb.N_SOURCES} sources, frontal and basal, by the shorter "
            "side"
        ),
    }


def assert_upsampling_counts(flags: dict, expected: dict | None = None) -> None:
    """The brief's counts, asserted as the CEILING they are.

    **[CORRECTED 2026-08-09] The brief's numbers are a different population
    than a staging run's.** 7-at-512 and 81-at-768 were measured over all 473
    source images -- frontal AND basal -- while the task stages the manifest's
    237 frontal rows. Asserting equality would abort every real 512 and 768
    run on a number no laptop can know: which of the upsamplers are frontal is
    cluster state (R10's widened clause), and the fixture that made the old
    equality pass was 473 synthetic sides -- the code agreeing with the
    fixture rather than with the data.

    What IS sound either way: the frontal subset can never upsample MORE than
    all 473 did, so above the ceiling means the sources changed or the staging
    did -- refused. At 224 the ceiling is zero, so equality still holds there
    exactly. The run records the measured frontal count in the artifact, the
    first place it CAN be measured; a re-run is then held to that record by
    guard 3 rather than by this ceiling.

    **[2026-08-09, second correction] The ceiling survives the flag's change
    of side.** The 473 counts were shorter-side counts and the flag now takes
    the longer side; a longer-side-flagged source is always shorter-side-
    flagged too, so the old counts bound the new quantity a fortiori.
    ``roadb.UPSAMPLING_FLAG_MEASURED_THE_WRONG_SIDE``.

    **[2026-08-09, third layer] Exact for the measured population.** The
    longer-side counts over the 237 frontal sources are now measured -- 0 at
    224, 1 at 512, 21 at 768 (``roadb.MEASURED_INTERPOLATION_COUNTS``) -- so
    a staging of that population is held to them exactly, in both
    directions: a v1-style shorter-side recount (41 at 768) refuses instead
    of passing under the ceiling. Any other population size falls back to
    the ceilings, because holding one cohort to another cohort's measurement
    is the population trap this function was corrected for.
    """
    target = flags["target"]
    if expected is None:
        measured = roadb.MEASURED_INTERPOLATION_COUNTS
        if flags["n"] == roadb.N_FRONTAL and target in measured["by_target"]:
            exact = measured["by_target"][target]
            if flags["n_upsampled"] != exact:
                raise StagingError(
                    f"{flags['n_upsampled']} patients interpolate at "
                    f"{target}, and the measured frontal count is {exact} "
                    f"of {roadb.N_FRONTAL} (longer side, 2026-08-09). "
                    "Either the sources changed or the staging did"
                )
    ceilings = expected or roadb.UPSAMPLING_COUNTS
    if target not in ceilings:
        return
    if flags["n_upsampled"] > ceilings[target]:
        raise StagingError(
            f"{flags['n_upsampled']} patients upsample at {target}, and the "
            f"brief measured {ceilings[target]} of {roadb.N_SOURCES} across "
            "ALL sources, frontal and basal. A frontal subset exceeding the "
            "whole is impossible unless the sources changed or the staging "
            "did"
        )


def manifest_upsampling_quantity(manifest: dict) -> dict:
    """Which quantity a staged artifact's upsampling flags are.

    **v1 manifests predate the flag's correction and CANNOT be amended**:
    artifacts are immutable, and their rollups are recorded in the gate
    verdict and the sheets config, so an edit would stale every one of those
    records. The correction therefore lives reader-side -- a manifest
    without a ``measures`` field is a shorter-side record, deprecated, and
    the measured longer-side counts ride along from
    ``roadb.MEASURED_INTERPOLATION_COUNTS`` so no reader has to take
    ``n_upsampled`` for the current definition.

    The deprecated flags remain USABLE for one purpose: they are a superset
    of the interpolated set (shorter-side-flagged includes every
    longer-side-flagged source), which is why the reviewed detail strip
    missed nothing.
    """
    upsampling = manifest["upsampling"]
    if "measures" in upsampling:
        return {"measures": upsampling["measures"], "deprecated": False}
    return {
        "measures": (
            "shorter side (v1 -- predates the 2026-08-09 correction; an "
            "overcount, valid as a superset of the interpolated set)"
        ),
        "deprecated": True,
        "corrected_counts": dict(
            roadb.MEASURED_INTERPOLATION_COUNTS["by_target"]
        ),
    }


def parity(observed_boxes: dict, recorded_boxes: dict, size: int) -> dict:
    """Do the new content boxes agree with the recorded cleft geometry?

    **Measured, not claimed from a shared code path.** A shared implementation
    is an argument that two things SHOULD agree; this is whether they do.
    Boxes are compared in NORMALISED units so a 512 box and a 224 box are
    comparable at all.
    """
    shared = sorted(set(observed_boxes) & set(recorded_boxes))
    if not shared:
        raise StagingError(
            "no patient appears in both the observed and recorded geometry; "
            "parity cannot be measured"
        )
    deviations = []
    for patient in shared:
        observed = np.asarray(observed_boxes[patient], dtype=float) / size
        expected = np.asarray(recorded_boxes[patient], dtype=float)
        deviations.append(float(np.abs(observed - expected).max()))
    array = np.asarray(deviations, dtype=float)
    return {
        "n_compared": len(shared),
        "n_observed_only": len(set(observed_boxes) - set(recorded_boxes)),
        "n_recorded_only": len(set(recorded_boxes) - set(observed_boxes)),
        "max_deviation": float(array.max()),
        "median_deviation": float(np.median(array)),
        "measured_not_claimed": (
            "a shared code path is an argument that two things should agree; "
            "this is whether they do"
        ),
    }


#: **[DECIDED 2026-08-08] "Frozen code, different argument" reads safer than
#: it is.**
#:
#: ``staging.stage`` is parameterised on ``size`` and has only ever run at
#: **224**. The six square cells therefore need no new code and no unfreezing
#: -- but that is a statement about the call site, not about the regime.
#:
#: The trapezium IS resolution-invariant (measured,
#: ``roadb.TRAPEZIUM_IS_RESOLUTION_INVARIANT``). What is untested is
#: ``resize_nearest`` at 512 and 768, and **that is exactly where the pixel
#: residual lives**: the residual measures resampling, and resampling is the
#: part of the path that has never run at these sizes.
#:
#: So the square cells are cheap to build without being free of risk, and the
#: residual gate is what catches it. Recorded because "reuses frozen code"
#: invites the opposite reading.
FROZEN_CODE_NEW_REGIME = {
    "decided": "2026-08-08",
    "cheap": "staging.stage already takes size; the six square cells add no code",
    "not_free_of_risk": (
        "stage has only ever run at 224. resize_nearest at 512 and 768 is an "
        "untested regime, and the pixel residual measures exactly that"
    ),
    "what_catches_it": "the per-family residual gate",
    "why_recorded": "'reuses frozen code' invites the opposite reading",
}


def stage_non_square(image, size: int):
    """Stage without padding: resize so the LONGER side is ``size``.

    **The content pixels are identical to ``staging.stage``'s at the same
    size** -- that function pads to square and then resizes, so the content
    lands in the same pixel box either way. Non-square removes the pad and
    changes nothing else, which is what makes the aspect and resolution axes
    orthogonal (``roadb.TWO_TWENTY_FOUR_NON_SQUARE``).

    New code because ``staging.py`` is frozen and always pads. It reuses the
    frozen ``resize_nearest`` and ``Staged`` so the resampling is the same
    operation the square path uses -- a second resize implementation would
    make the aspect axis a resampling comparison as well.
    """
    from .geometry.staging import Staged, resize_nearest, _check

    array = _check(image)
    height, width = array.shape[:2]
    if height >= width:
        out_h, out_w = int(size), max(1, int(round(width * size / height)))
    else:
        out_w, out_h = int(size), max(1, int(round(height * size / width)))
    # **The shape is fixed HERE, and this is the only place it can be.**
    # SHAPE_IS_FIXED_AT_STAGING: ViT asserts both dimensions divisible by the
    # patch size, so an arbitrary aspect cannot reach the backbone at all.
    out_w = max(PATCH, int(round(out_w / PATCH)) * PATCH)
    out_h = max(PATCH, int(round(out_h / PATCH)) * PATCH)
    resized = resize_nearest(array, out_w, out_h)
    return Staged(
        image=resized,
        source_size=(width, height),
        # No pad: the content box IS the image.
        content_box=(0, 0, out_w, out_h),
        scale=out_w / width,
    )


#: **[MEASURED 2026-08-09] Where the shape is fixed, and by what.**
#:
#: Non-square staging preserves each patient's aspect ratio, and those span
#: 0.553 to 1.099 -- so the staged arrays have different shapes and cannot be
#: stacked. Storage is therefore **per patient**, which is the honest artifact:
#: it preserves the ratio exactly and carries what the setting means.
#:
#: **But the shape has to be fixed somewhere, and the options are not
#: symmetric.** Measured against a real ViT-B/16:
#:
#:     (224, 224)   197 tokens, grid 14x14
#:     (224, 124)   FAILS -- width not divisible by the patch size
#:     (224, 204)   FAILS -- same
#:
#: **``dynamic_img_size=True`` does not rescue it.** The assertion is on
#: divisibility by 16, not on squareness, so an arbitrary aspect cannot reach
#: the backbone at all. And mixed shapes cannot be batched.
#:
#: **So the shape is fixed AT STAGING, by rounding each dimension to a
#: multiple of 16.** The alternatives all fix it at extraction, and both are
#: worse:
#:
#: * **pad to square there** -- reintroduces the pad, which is the whole point
#:   of the setting, just relocated and unrecorded;
#: * **resize to square there** -- anisotropic distortion of exactly the
#:   asymmetry the label is about.
#:
#: **The cost, stated: rounding is itself an anisotropic distortion**, bounded
#: by 8 pixels on a dimension. That is <=6.5% of the narrowest patient's width
#: at 224, ~2.8% at 512 and ~1.9% at 768 -- so **the distortion shrinks with
#: resolution, in the same direction as the pixel residual.** The 224
#: non-square cell is the weakest of the three and 768 the cleanest, which is
#: worth knowing when the pad pair is read.
#:
#: **Downstream cost of per-patient storage:** every consumer expecting
#: ``(N, H, W, 3)`` changes -- extraction, batching, and Phase 8's fixed 14x14
#: token grid, which becomes per patient.
SHAPE_IS_FIXED_AT_STAGING = {
    "measured": "2026-08-09",
    "storage": "per patient -- the arrays have different shapes and cannot stack",
    "aspect_range": (0.553, 1.099),
    "vit_requires": "both dimensions divisible by 16, even with dynamic_img_size",
    "measured_failures": {"(224,124)": "AssertionError", "(224,204)": "AssertionError"},
    "fixed_where": "at staging, by rounding each dimension to a multiple of 16",
    "why_not_at_extraction": (
        "padding there reintroduces the pad unrecorded; resizing there is "
        "anisotropic distortion of the asymmetry the label is about"
    ),
    "cost": (
        "rounding is itself anisotropic, bounded by 8px on a dimension -- "
        "<=6.5% at 224, ~2.8% at 512, ~1.9% at 768"
    ),
    "shrinks_with_resolution": (
        "same direction as the pixel residual: the 224 non-square cell is the "
        "weakest and 768 the cleanest"
    ),
    "downstream": (
        "every consumer expecting (N, H, W, 3) changes -- extraction, "
        "batching, and phase8's fixed 14x14 token grid becomes per patient"
    ),
}
