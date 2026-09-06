"""Masked SCUT variants at both geometries, and the parity measurement.

Phase 5 §2. Produces the pretraining input Phase 6 consumes.

**Every geometry call goes through the frozen cleft path.** ``trapezium.mask``,
``staging.stage`` and ``trapezium.unwarp`` are used exactly as the cleft pipeline
uses them -- this module contains no reimplementation of any of them, because a
second implementation is where the two domains would silently diverge. The three
staging modules are in ``FROZEN_FILES`` precisely so that rule is mechanical
rather than remembered: SCUT-specific entry points that *call* them, never edits.

**The pipeline, per face -- the frozen composition, exactly:**

1. ``placement.crop_box`` -- the content box from the landmarks: brow to just
   below the lower vermillion, width from a sampled aspect ratio, centred on the
   face's own midline.
2. ``apply_arrival_mask`` -- white outside the trapezium normalised to the
   CROP CONTENT, baked in before staging. This is the state a cleft crop
   *arrives* in: the trapezium is measured "normalised within the crop content"
   (``trapezium.py``), so its corners are content-relative, not square-relative.
3. ``staging.stage`` -- pad-square-then-resize to 224 with white pad -- then,
   for **G2**, ``trapezium.unwarp`` on the staged square. That order is the
   frozen cleft path verbatim (``stage_build.py`` stages, then unwarps
   ``base.image``), and matching it is the whole point: the pad IS stretched
   into G2 by that path, for cleft exactly as here. **G1** is the staged square
   as-is; its white corners arrived baked into the content, as cleft's did.

**Parity is measured, not claimed.** A shared code path is an argument for
parity; the distributions are the evidence. ``parity_report`` compares the
produced set against the recorded cleft figures in
``staging.CLEFT_STAGED_GEOMETRY`` -- aspect ratio, pad fraction, white fraction --
and reports the gap per statistic rather than a verdict.
"""

from __future__ import annotations

import numpy as np

from ..geometry import trapezium as trapezium_module
from ..geometry.staging import (
    CLEFT_STAGED_GEOMETRY,
    OUTPUT_SIZE,
    PAD_VALUE,
    stage,
)
from . import ar_distribution, placement

GEOMETRIES = ("g1", "g2")

#: Sampling the aspect ratio from the measured cleft distribution rather than
#: fixing it at the median. A single-AR masked set would give every pretrained
#: model a constant framing that the clinical data varies around by +/-0.18, and
#: the fixed version is a special case of this one (``spread=0``), so building
#: the sampled version costs almost nothing and makes the question answerable.
#: ``empirical``     piecewise-linear inverse CDF over the cohort's measured
#:                    percentiles. **Superseded by ``observed``**: measured over
#:                    5,499 faces it realises 2.75% landscape against the
#:                    cohort's 0.42%, because interpolating across
#:                    ``[p95, p100]`` invents a uniform tail the data does not
#:                    have (``ar_distribution.AR_TAIL_DEFECT``). Kept so earlier
#:                    runs can be re-derived.
#: ``uniform_range``  uniform over [min, max]. **Kept only so the earlier runs
#:                    can be re-derived**, and named honestly: it was called
#:                    "sampled", which read as though it sampled the cohort. It
#:                    gives mean 0.826 against the true 0.7475, nearly one SD high.
#: ``fixed_median``   every face at the cohort median. The no-variation control.
#: ``observed``       bootstrap over the 237 raw cleft ratios -- **the correct
#:                    one**, once ``ar_distribution.AR_OBSERVED`` is populated.
#:                    Reproduces the cohort exactly, sparse tail included, and
#:                    removes the interpolation step rather than adding one.
AR_SAMPLING = ("observed", "empirical", "uniform_range", "fixed_median")

#: **``observed`` since 2026-07-30**, when the 237 raw cleft ratios were recorded
#: in ``ar_distribution.AR_OBSERVED``. The bootstrap reproduces the cohort exactly
#: -- 0.42% landscape against the interpolating sampler's 2.75% -- and has no
#: interpolation step to go wrong.
#:
#: The move was guarded rather than remembered:
#: ``test_the_default_sampler_tracks_whether_the_ratios_are_available`` was
#: **observed failing** on the populated constant with the default still
#: ``empirical``, then passing once it moved. A guard that had stayed silent there
#: would have left the fix present and unused, which looks exactly like the fix
#: being applied.
DEFAULT_AR_SAMPLING = "observed"


#: **[MEASURED 2026-07-30] RESOLVED -- the invariant was a quantity confusion,
#: and the real defect was the mask's coordinate frame, not the operation order.**
#:
#: **The "zero white" invariant rested on a misread quantity (R2).** p2-stage-1's
#: ``white_fraction_max: 0.0`` for ``grid_g2``/``anatomy_g2``/``random_g2`` is
#: ``patches.summarise``: ``1 - min(coverage)``, where coverage is the GEOMETRIC
#: overlap of a patch box with the generating mask -- and at G2 that mask is
#: ``contact.G2_TRAPEZIUM``, the full square. Every G2 patch coverage is 1.0 **by
#: construction**; no pixel is consulted anywhere in that number. It cannot fail,
#: so it cannot establish a pixel invariant -- kin to the checks-that-cannot-fail
#: tally in PLAN R7, and the fifth time the probe rather than the code was the
#: defect.
#:
#: **The pixel truth, measured by replaying the frozen path** (bake the
#: content-relative trapezium into a synthetic crop, then ``stage`` ->
#: ``unwarp``, exactly ``stage_build.py``'s composition): cleft G2 is NEVER
#: white-free. White ~= pad_fraction: 0.0026 at AR 1.0, 0.259 at the median
#: 0.7404, 0.446 at AR 0.5531. The pad is stretched into G2 by the frozen path
#: itself -- for cleft exactly as for SCUT.
#:
#: **So the retracted "matched behaviour" conclusion is un-retracted for the
#: pad**: a cleft crop at the same ratio does pad identically, and its pad does
#: reach G2. The retraction mistook a patch-geometry constant for a pixel
#: measurement. What the conclusion still missed: parity failed anyway, in the
#: opposite direction --
#:
#: **The actual defect: the trapezium was applied in the wrong frame.** Cleft
#: crops arrive with the trapezium baked in normalised to the CROP CONTENT
#: ("normalised within the crop content", ``trapezium.py``). ``build_one``
#: instead applied ``trapezium.mask`` to the padded SQUARE after staging -- a
#: wider trapezium at every row for portrait crops, and a no-op for G2 (unwarp
#: samples only inside-trapezium pixels). Masked SCUT therefore had too LITTLE
#: white: at the median AR, G2 0.092 against cleft's 0.259, G1 0.283 against
#: cleft's 0.407. The operation order was already matched; the arrival state was
#: not. Fixed by ``apply_arrival_mask`` before ``stage``.
#:
#: **The left/right gap is matched behaviour, and its "likely cause" was also
#: wrong.** The frozen replay shows the same gap: 17 of 23 cohort ARs nonzero,
#: max 0.0089 (G1) / 0.0114 (G2) -- including at even 29/29 pads and at AR 1.0
#: with no pad at all, so the odd-pad-split explanation fails. The source is
#: ``resize_nearest``'s floor-based index mapping, which is left-biased by half a
#: source pixel, plus edge rasterisation. Post-fix the masked path is the frozen
#: composition bit-for-bit, so its gap IS the cleft path's gap at the same
#: content geometry. Caveat, stated not assumed: real cleft crops' baked masks
#: are hand-made and vary per patient; their actual per-quadrant figures live
#: only in the cluster arrays. The replay is the canonical-geometry expectation,
#: not a per-patient measurement.
G2_WHITE_DEFECT = {
    "status": "RESOLVED",
    "measured": "2026-07-29",
    "rediagnosed": "2026-07-30",
    "invariant_was": "cleft G2 white_fraction_max is 0.0 (p2-stage-1)",
    "invariant_source": (
        "patches.summarise: white_fraction_max = 1 - min(patch coverage), "
        "coverage measured against contact.G2_TRAPEZIUM -- the full square -- "
        "so 0.0 BY CONSTRUCTION. A patch-geometry quantity with no pixels in "
        "it, misread as a pixel fact. R2: right number, wrong quantity."
    ),
    "pixel_truth": {
        "method": (
            "frozen-path replay: bake content-relative trapezium into a "
            "synthetic crop, stage, unwarp -- stage_build.py's composition "
            "verbatim, measured at the cohort AR percentiles"
        ),
        "g2_white_by_ar": {"1.0": 0.0026, "0.7404": 0.2594, "0.5531": 0.4462},
        "g2_white_rule_of_thumb": "approximately the pad fraction",
        "g1_white_at_median": 0.407,
    },
    "actual_defect": (
        "mask applied square-relative AFTER staging instead of content-relative "
        "BEFORE it. Cleft crops arrive with the trapezium baked into the crop "
        "content; the square-relative mask is wider at every row for portrait "
        "crops and a no-op for G2. Masked SCUT had too LITTLE white, not "
        "wrongly-present white. Operation order was already matched."
    ),
    "fix": (
        "apply_arrival_mask (content-relative, pre-stage), then stage, then "
        "unwarp for G2 -- the frozen composition exactly, asserted "
        "pixel-identical by test"
    ),
    "unretracted": (
        "'a cleft crop at the same ratio pads identically' was correct and is "
        "reinstated: the refutation misread a geometric constant as a pixel "
        "measurement -- the probe was the defect, fifth instance"
    ),
    "second_defect_resolution": (
        "left/right white asymmetry is MATCHED behaviour: the frozen replay "
        "shows gaps on 17/23 cohort ARs (max 0.0089 G1 / 0.0114 G2), including "
        "at even 29/29 pads and at AR 1.0 with no pad, so the odd-pad-split "
        "hypothesis was wrong too. Cause: resize_nearest's floor-based, "
        "left-biased index mapping plus edge rasterisation. Post-fix the gap "
        "is the cleft path's own, bit-for-bit at the same content geometry. "
        "Real cleft arrays' per-patient figures remain cluster-only; the "
        "replay is the canonical-geometry expectation."
    ),
    "blocked_nothing_after": "2026-07-30",
}


#: **[MEASURED 2026-07-30, 5,499-face build] ``white_fraction`` counts pure-white
#: PIXELS, not pad-and-mask, and the two are not the same quantity.**
#:
#: The test is ``all(pixel == 255)``, so a face photographed against a white
#: background contributes its own white -- and if that background sits
#: asymmetrically in the crop, it inflates ``white_left_right_gap`` too. The name
#: says pad and mask; the computation says any white pixel. R2 again, in this
#: module's own instrument.
#:
#: **Measured, by computing each face's gap against the geometry-only prediction
#: for the same crop dimensions:**
#:
#: | quantity | value |
#: |---|---|
#: | geometry-only gap, max | **0.00893** (G1), 0.01339 (G2) |
#: | frozen cleft replay, max | 0.0089 (G1), 0.0114 (G2) |
#: | build's reported max | 0.046 (G1), 0.049 (G2) |
#: | faces with any white content in the crop | 1.33% |
#: | faces whose gap it moves by >0.001 | 0.47% |
#: | corr(content white, excess gap) | **0.977** |
#:
#: So the geometry-only gap agrees with the frozen replay to three decimals, and
#: **the entire 4x excess in the tail is image content**. The extreme case in a
#: 1,500-face sample is AF1333: gap 0.01674 of which 0.01662 is content.
#:
#: **What this does and does not affect.** Mean and median gap are geometry
#: (computed 0.00461/0.00769 against the build's 0.00464/0.00769), so the
#: distribution and the parity conclusion stand -- content white is a rare tail,
#: not a systematic bias. It does mean the white statistics are **not purely a
#: geometry comparison** against cleft, whose crops arrive on a white-padded
#: background of their own. Reported rather than corrected: separating the two
#: needs a content-white measurement per face, which is
#: ``white_content_fraction`` below.
WHITE_IS_ANY_WHITE_PIXEL = {
    "measured": "2026-07-30, full build plus a 1500-face geometry comparison",
    "geometry_only_gap_max": {"g1": 0.00893, "g2": 0.01339},
    "frozen_replay_gap_max": {"g1": 0.0089, "g2": 0.0114},
    "build_reported_gap_max": {"g1": 0.046, "g2": 0.049},
    "faces_with_white_content": 0.0133,
    "faces_where_it_moves_the_gap": 0.0047,
    "correlation_content_white_to_excess_gap": 0.977,
    "affects": (
        "the TAIL only. Mean and median gap are geometry (0.00461/0.00769 "
        "computed against 0.00464/0.00769 reported), so the distribution shape "
        "and the parity conclusion stand."
    ),
    "quantity_confusion": (
        "the metric is named for pad and mask and computes ALL pure-white "
        "pixels, so a white-background photograph counts as mask"
    ),
}

#: **[MEASURED 2026-07-30] The G1 gap distribution is BIMODAL, which is why its
#: mean sits below its median.**
#:
#: Over all 5,499 faces: **2,421 (44%) below 0.0004 and 2,378 (43%) above
#: 0.0086**, with 13% spread between. Mean 0.00461, median 0.00769.
#:
#: A mean below a median normally reads as left skew, which would be odd for a
#: quantity bounded below at zero with a long right tail. It is not skew: the
#: median falls inside the upper mode while the mean is pulled between the two.
#: **Reading the pair as a skew statistic would describe a shape this
#: distribution does not have** -- a summary statistic answering the wrong
#: question about a distribution it does not fit.
#:
#: The two modes are the pad's parity: at a given staged content width the pad
#: either splits evenly between left and right or cannot, and ``resize_nearest``'s
#: left-biased mapping turns the one-pixel residue into a fixed offset. Nothing
#: continuous connects them, so the shape is two spikes rather than a spread.
GAP_DISTRIBUTION_IS_BIMODAL = {
    "measured": "2026-07-30, exact over all 5499 faces",
    "n_below_0_0004": 2421,
    "n_above_0_0086": 2378,
    "fraction_between": 0.13,
    "mean": 0.00461,
    "median": 0.00769,
    "why_mean_below_median": (
        "BIMODALITY, not left skew: the median falls inside the upper mode while "
        "the mean is pulled between the two. Reading mean-minus-median as a skew "
        "statistic would describe a shape this distribution does not have."
    ),
    "mechanism": (
        "the pad either splits evenly between left and right or cannot, and "
        "resize_nearest's left-biased mapping turns the one-pixel residue into a "
        "fixed offset. Nothing continuous connects the two cases."
    ),
}


class MaskedError(RuntimeError):
    """A masked variant could not be built."""


def sample_aspect_ratios(n: int, seed: int, mode: str = DEFAULT_AR_SAMPLING):
    """One aspect ratio per face, from the measured cleft distribution.

    **Default ``observed``: a bootstrap with replacement over the 237 raw cleft
    ratios** (``ar_distribution.AR_OBSERVED``). It reproduces the cohort's
    distribution exactly -- including a sparse tail, which is where every
    parameterised alternative fails -- and has no interpolation step to go wrong.

    **Two supersessions, in order, and both were measured rather than argued.**

    ``uniform_range`` drew uniformly over [min, max] on the grounds that three
    order statistics support no more. It gives mean **0.826** against the
    cohort's **0.7475**, nearly one SD high, because the range is dominated by
    two thin tails -- 236 of 237 crops are portrait.

    ``empirical`` fixed the bulk with a piecewise-linear inverse CDF over 21
    percentiles, and the full 5,499-face build showed what it did to the tail:
    **2.75% landscape against the cohort's 0.42%**, because interpolating across
    ``[p95, p100] = [0.9173, 1.0986]`` invents a uniform tail over a bin holding
    one observation at its top. See ``ar_distribution.AR_TAIL_DEFECT``, where the
    predicted and observed fractions agree to four decimals.

    Both are kept so the runs made with them can be re-derived. A superseded
    sampler is not a deleted one.

    Seeded and returned per face so the draw is reproducible and the realised
    distribution can be compared against the cleft one.
    """
    if mode not in AR_SAMPLING:
        raise MaskedError(f"unknown ar sampling {mode!r}; expected one of {AR_SAMPLING}")

    ratios = CLEFT_STAGED_GEOMETRY["aspect_ratio"]
    if mode == "fixed_median":
        return np.full(n, ratios["median"], dtype=float)
    if mode == "uniform_range":
        return np.random.default_rng(seed).uniform(ratios["min"], ratios["max"], size=n)
    if mode == "observed":
        return ar_distribution.sample_observed(n, seed)
    return ar_distribution.sample_empirical(n, seed)


def crop_content(image: np.ndarray, box) -> np.ndarray:
    """Cut the content box out of the source image, clipped to its bounds."""
    x, y, w, h = box
    height, width = image.shape[:2]
    x0 = max(int(round(x)), 0)
    y0 = max(int(round(y)), 0)
    x1 = min(int(round(x + w)), width)
    y1 = min(int(round(y + h)), height)
    if x1 - x0 < 2 or y1 - y0 < 2:
        raise MaskedError(
            f"content box {box} leaves nothing inside a {width}x{height} image"
        )
    return image[y0:y1, x0:x1]


def apply_arrival_mask(content: np.ndarray) -> np.ndarray:
    """White outside the trapezium normalised to the CROP CONTENT -- the state a
    cleft crop arrives in, reproduced on a SCUT crop before staging.

    The cleft crops **arrived** with white corners baked in at source, and the
    trapezium is measured *within the crop content* (``trapezium.py``): 0.301
    half-width at the brow, full crop width at the lip. So the mask must be
    applied to the crop BEFORE ``stage``, in content coordinates -- applying
    ``trapezium.mask`` to the padded square instead was this module's defect:
    a wider trapezium at every row for portrait crops, and a no-op for G2.

    The arithmetic is the frozen ``Trapezium``'s own (``half_width_array``), on
    the same pixel-centre rule ``trapezium.mask`` uses; on a square input this
    reproduces ``trapezium.mask`` exactly, and a test pins that so the two
    cannot drift.
    """
    height, width = content.shape[:2]
    ys = (np.arange(height) + 0.5) / height
    xs = (np.arange(width) + 0.5) / width
    half = trapezium_module.DEFAULT.half_width_array(ys)
    inside = np.abs(xs[None, :] - trapezium_module.MIDLINE) <= half[:, None]
    out = content.copy()
    out[~inside] = PAD_VALUE
    return out


def build_one(
    image: np.ndarray,
    points: np.ndarray,
    aspect_ratio: float,
    geometry: str,
    lower_margin: float = placement.LOWER_MARGIN,
    size: int = OUTPUT_SIZE,
    box: tuple | None = None,
) -> tuple[np.ndarray, dict]:
    """One face to one masked variant, plus what was measured doing it.

    **[2026-08-09] ``size`` threads Road B's resolution axis through the same
    frozen composition** -- ``stage`` already took the argument, exactly the
    reuse pattern Phase 2 ran. The default keeps every existing call
    byte-identical to the composition the bit-for-bit test pins. **Every
    SCUT source is 350x350**, so any size above that interpolates every face
    uniformly (1.46x at 512, 2.19x at 768) -- what higher-resolution
    pretraining adapts is positional geometry, not finer detail
    (``roadb.PHASE_6_STRUCTURE``), and the sheet must make that visible.
    """
    if geometry not in GEOMETRIES:
        raise MaskedError(f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}")

    # **[2026-08-24] ``box`` threads Phase 15's landmark convention
    # through the same frozen composition** -- the pattern ``size``
    # already set for Road B's resolution axis. Default None keeps every
    # existing call byte-identical to what the bit-for-bit test pins;
    # MEBeauty passes a box from its own anchors (mebeauty.crop_box),
    # and everything downstream of this line is convention-agnostic.
    if box is None:
        box = placement.crop_box(points, aspect_ratio, lower_margin)
    inside_frame = placement.box_is_inside(box, image.shape)
    content = crop_content(image, box)

    # The REQUESTED ratio and the one actually produced. They agree while the
    # box fits the source image and diverge silently the moment it is clipped --
    # and parity measured against the requested value would then be measuring an
    # intention rather than an output. Both are recorded so the difference is
    # visible instead of assumed away.
    realised = content.shape[1] / content.shape[0]

    # White that was ALREADY IN THE PHOTOGRAPH, measured on the raw crop before
    # any mask or pad exists. Without this, `white_fraction` conflates the
    # trapezium and the pad with a white studio background, and the conflation is
    # invisible -- see WHITE_IS_ANY_WHITE_PIXEL, where content white explains the
    # entire tail of the left/right gap at correlation 0.977.
    content_white = float(
        np.all(content == PAD_VALUE, axis=2).mean()
        if content.ndim == 3
        else (content == PAD_VALUE).mean()
    )

    # The frozen composition, verbatim: the crop arrives with the trapezium
    # baked into its content, is staged (pad-square-then-resize), and G2
    # unwarps the STAGED SQUARE -- stage_build.py's order. The pad reaches G2
    # here exactly as it does for cleft; see G2_WHITE_DEFECT.
    staged = stage(apply_arrival_mask(content), size=size)
    masked = staged.image
    if geometry == "g2":
        masked = trapezium_module.unwarp(masked)

    white = np.all(masked == PAD_VALUE, axis=2) if masked.ndim == 3 else masked == PAD_VALUE
    height, width = white.shape
    top, left = height // 2, width // 2
    quadrants = {
        "top_left": round(float(white[:top, :left].mean()), 6),
        "top_right": round(float(white[:top, left:].mean()), 6),
        "bottom_left": round(float(white[top:, :left].mean()), 6),
        "bottom_right": round(float(white[top:, left:].mean()), 6),
    }
    return masked, {
        "geometry": geometry,
        # Split by quadrant so ASYMMETRY is visible. A symmetric trapezium over
        # centred padding cannot produce a left/right difference, so any gap here
        # is a defect rather than a property -- and an aggregate white fraction
        # hides it completely. See G2_WHITE_DEFECT.
        "white_by_quadrant": quadrants,
        "white_left_right_gap": round(
            abs(
                (quadrants["top_left"] + quadrants["bottom_left"])
                - (quadrants["top_right"] + quadrants["bottom_right"])
            )
            / 2.0,
            6,
        ),
        "aspect_ratio": round(float(realised), 6),
        "aspect_ratio_requested": round(float(aspect_ratio), 6),
        "box_inside_frame": bool(inside_frame),
        "pad_fraction": round(float(staged.pad_fraction), 6),
        "white_fraction": round(float(white.mean()), 6),
        # White already present in the photograph, so `white_fraction` can be
        # read as geometry plus content rather than as geometry alone. A cleft
        # crop has no equivalent -- its background is the pad -- so a face with a
        # white studio backdrop is NOT matched behaviour, it is a SCUT property
        # the comparison has to account for. See WHITE_IS_ANY_WHITE_PIXEL.
        "white_content_fraction": round(content_white, 6),
        "content_box": [int(v) for v in staged.content_box],
        "intercanthal_angle_deg": round(placement.intercanthal_angle(points), 3),
    }


# --------------------------------------------------------------------------
# the sheet: placement and result answer different questions
# --------------------------------------------------------------------------


def placement_panel(
    image: np.ndarray,
    points: np.ndarray,
    aspect_ratio: float,
    size: int = OUTPUT_SIZE,
) -> np.ndarray:
    """``annotate_placement`` at the sheet's own panel size.

    **[CORRECTED 2026-08-09, sheet review]** The annotated source rendered at
    its native 350px inside cells sized by the masked panels -- mostly empty
    grey at 768, and the size mismatch is what broke the caption alignment.
    Resized with the frozen ``resize_nearest``, so upscaling keeps it
    BLOCKY-sharp: beside it, the masked panels' smoothing is the
    interpolation the write-up guard names, and the comparison sharpens
    rather than blurs.
    """
    from ..geometry.staging import resize_nearest

    annotated = annotate_placement(image, points, aspect_ratio)
    if annotated.shape[:2] == (size, size):
        return annotated
    return resize_nearest(annotated, size, size)


def annotate_placement(
    image: np.ndarray, points: np.ndarray, aspect_ratio: float
) -> np.ndarray:
    """The content box and trapezium drawn on the source face.

    Rendered beside the masked output because the two answer different
    questions -- *is the trapezium in the right place on the face* and *does the
    masked result look like a cleft crop* -- and one sheet means one review pass.
    """
    canvas = np.asarray(image).copy()
    if canvas.ndim == 2:
        canvas = np.stack([canvas] * 3, axis=2)
    height, width = canvas.shape[:2]

    x, y, w, h = placement.crop_box(points, aspect_ratio)
    yellow = np.array([255, 255, 0], dtype=np.uint8)
    magenta = np.array([255, 0, 255], dtype=np.uint8)

    def put(row: int, column: int, colour) -> None:
        if 0 <= row < height and 0 <= column < width:
            canvas[row, column] = colour

    for step in range(int(round(w))):
        put(int(round(y)), int(round(x)) + step, yellow)
        put(int(round(y + h)) - 1, int(round(x)) + step, yellow)
    for step in range(int(round(h))):
        put(int(round(y)) + step, int(round(x)), yellow)
        put(int(round(y)) + step, int(round(x + w)) - 1, yellow)

    # The trapezium edges, as fractions of the content width.
    centre = x + w / 2.0
    for step in range(int(round(h))):
        fraction = step / max(h - 1, 1)
        half = (
            trapezium_module.TOP_HALF_WIDTH
            + fraction
            * (trapezium_module.BOT_HALF_WIDTH - trapezium_module.TOP_HALF_WIDTH)
        ) * w
        put(int(round(y)) + step, int(round(centre - half)), magenta)
        put(int(round(y)) + step, int(round(centre + half)), magenta)

    # The landmark midline, cyan: the thing that must not be the image centre.
    mid = int(round(placement.midline_x(points)))
    for step in range(int(round(h))):
        put(int(round(y)) + step, mid, np.array([0, 255, 255], dtype=np.uint8))
    return canvas


def cleft_reference_panel(size: int = OUTPUT_SIZE) -> np.ndarray:
    """The expected cleft trapezium at the median cleft AR, from the record.

    Drawn from ``CLEFT_STAGED_GEOMETRY`` rather than from memory, so *does this
    look like a cleft crop* is a **visual comparison on the same sheet** instead
    of a recollection. No cleft image is used or needed -- the shape comes from
    the recorded numbers.
    """
    panel = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    ratios = CLEFT_STAGED_GEOMETRY["aspect_ratio"]
    trap = CLEFT_STAGED_GEOMETRY["trapezium"]

    # A median-AR content box, letterboxed into the square exactly as staging
    # would place it.
    content_w = int(round(size * ratios["median"])) if ratios["median"] <= 1 else size
    content_h = size if ratios["median"] <= 1 else int(round(size / ratios["median"]))
    x0 = (size - content_w) // 2
    y0 = (size - content_h) // 2
    panel[y0 : y0 + content_h, x0 : x0 + content_w] = np.array(
        [225, 225, 225], dtype=np.uint8
    )

    grey = np.array([120, 120, 120], dtype=np.uint8)
    centre = x0 + content_w / 2.0
    for step in range(content_h):
        fraction = step / max(content_h - 1, 1)
        half = (
            trap["top_hw"] + fraction * (trap["bot_hw"] - trap["top_hw"])
        ) * content_w
        row = y0 + step
        for column in (int(round(centre - half)), int(round(centre + half))):
            if 0 <= column < size:
                panel[row, column] = grey
    return panel


# --------------------------------------------------------------------------
# parity, measured
# --------------------------------------------------------------------------


def _spread(values) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "min": round(float(array.min()), 6),
        "median": round(float(np.median(array)), 6),
        "max": round(float(array.max()), 6),
        "mean": round(float(array.mean()), 6),
        "sd": round(float(array.std(ddof=1)), 6) if array.size > 1 else 0.0,
    }


def parity_report(records: list[dict]) -> dict:
    """Masked SCUT against the recorded cleft geometry. Gaps, not a verdict.

    **A shared code path is an argument for parity; this is the evidence.** The
    cleft figures come from ``staging.CLEFT_STAGED_GEOMETRY``, which is sourced
    from the p2-stage-1 artifact and lives in the codebase precisely so this
    comparison does not read them out of a brief.

    No pass/fail. What counts as "close enough" on a domain gap is a judgement
    about pretraining transfer, not a threshold anyone here can justify, so the
    numbers are reported and the reader decides.
    """
    if not records:
        raise MaskedError("no records to compare")

    cleft = CLEFT_STAGED_GEOMETRY
    produced = {
        # The REALISED ratio -- what the set actually has, not what was asked for.
        "aspect_ratio": _spread([r["aspect_ratio"] for r in records]),
        "aspect_ratio_requested": _spread(
            [r.get("aspect_ratio_requested", r["aspect_ratio"]) for r in records]
        ),
        "pad_fraction": _spread([r["pad_fraction"] for r in records]),
        "white_fraction": _spread([r["white_fraction"] for r in records]),
    }

    gaps = {}
    for key, reference in (
        ("aspect_ratio", cleft["aspect_ratio"]),
        ("pad_fraction", cleft["pad_fraction"]),
    ):
        mine = produced[key]
        # **[2026-08-24] An undefined statistic says WHY it is undefined**
        # (phase15.PARITY_NONE_DIAGNOSED). ``CLEFT_STAGED_GEOMETRY``
        # carries DIFFERENT statistics for different quantities --
        # aspect_ratio has min/median/max, pad_fraction has
        # min/mean/max -- so exactly one gap per key is computable, and
        # a None here has always meant "the reference does not record
        # that statistic", never "the group was empty". Returning a
        # bare None left the caller to guess between those, and a
        # caller that formatted it crashed. The reading-count guard's
        # principle, one level down: do not emit a number that does not
        # exist, and name the reason it does not.
        undefined = {}
        for statistic in ("median", "mean"):
            if statistic not in reference:
                undefined[f"{statistic}_gap"] = (
                    f"the cleft reference for {key!r} records "
                    f"{sorted(reference)} and no {statistic!r}; "
                    f"{mine['n']} produced values were summarised, so "
                    "the group is NOT empty -- the statistic is simply "
                    "not defined for this comparison"
                )
        gaps[key] = {
            "cleft": reference,
            "scut": mine,
            "median_gap": (
                round(mine["median"] - reference["median"], 6)
                if "median" in reference
                else None
            ),
            "mean_gap": (
                round(mine["mean"] - reference["mean"], 6)
                if "mean" in reference
                else None
            ),
            "undefined": undefined,
        }

    gaps_lr = [r.get("white_left_right_gap", 0.0) for r in records]
    content_white = [r.get("white_content_fraction", 0.0) for r in records]
    return {
        "n_faces": len(records),
        "produced": produced,
        "gaps": gaps,
        # The AR distribution against the cohort's own shape, not just its range.
        "ar_vs_cohort": ar_distribution.describe(
            [r["aspect_ratio"] for r in records]
        ),
        # Left/right asymmetry, which an aggregate white fraction hides. A
        # symmetric trapezium over centred padding cannot produce a gap here.
        "white_left_right_gap": _spread(gaps_lr),
        "n_left_right_asymmetric": sum(1 for g in gaps_lr if g > 1e-9),
        # White that was in the photograph, separated from pad-and-mask white so
        # the two are not read as one. A cleft crop has no equivalent, so this is
        # a SCUT property the comparison accounts for rather than matched
        # behaviour. See WHITE_IS_ANY_WHITE_PIXEL.
        "white_content_fraction": _spread(content_white),
        "n_faces_with_white_content": sum(1 for v in content_white if v > 0),
        "white_is_any_white_pixel": dict(WHITE_IS_ANY_WHITE_PIXEL),
        "gap_distribution_is_bimodal": dict(GAP_DISTRIBUTION_IS_BIMODAL),
        "g2_white_defect": dict(G2_WHITE_DEFECT),
        "cleft_reference": dict(cleft),
        "n_box_outside_frame": sum(1 for r in records if not r["box_inside_frame"]),
        # Derived from the RECORDED cleft constants at the cohort's mean pad,
        # not from memory. G1 white = pad + the baked trapezium's complement of
        # the content: 1 - area_of_content * (1 - pad_mean). G2 white ~= the pad
        # fraction itself -- the frozen path stretches the pad into G2 (measured;
        # see G2_WHITE_DEFECT). Per-face values vary with AR around these, plus
        # rasterisation in the fourth decimal. The old expectation here, 0.198,
        # was the square-relative trapezium's complement -- the same wrong frame
        # the build itself had, and right only at AR 1.
        "expected_white_fraction": {
            "g1_at_cleft_mean_pad": round(
                1.0
                - cleft["trapezium"]["area_of_content"]
                * (1.0 - cleft["pad_fraction"]["mean"]),
                6,
            ),
            "g2_at_cleft_mean_pad": cleft["pad_fraction"]["mean"],
        },
        "note": (
            "PARITY IS MEASURED, NOT CLAIMED FROM A SHARED CODE PATH. The cleft "
            "figures come from staging.CLEFT_STAGED_GEOMETRY (p2-stage-1), which "
            "is in the codebase so this comparison does not read them out of a "
            "brief. "
            "NO PASS/FAIL: what counts as close enough on a domain gap is a "
            "judgement about pretraining transfer, not a threshold justifiable "
            "here, so the gaps are reported and the reader decides. "
            "white_fraction expectations are derived from the recorded cleft "
            "constants: G1 near 1 - 0.802*(1 - pad_mean), G2 near pad_mean -- "
            "NOT zero: the frozen cleft path stretches the pad into G2, and the "
            "earlier zero-white invariant was a patch-geometry constant misread "
            "as a pixel fact (see g2_white_defect). "
            "n_box_outside_frame counts faces whose content box ran off the "
            "source image and was clipped -- clipping puts padding INSIDE the "
            "content box, which no cleft crop has."
        ),
    }
