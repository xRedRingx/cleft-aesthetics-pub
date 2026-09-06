"""Placing the cleft trapezium on a SCUT face, from its landmarks.

**This is where the domain match is won or lost.** The masked SCUT set exists so
Phase 6 can pretrain on the same *kind* of image the cleft arms fine-tune on. If
the trapezium lands on different anatomy here than it does on a cleft crop, the
pretraining teaches the wrong thing and nothing downstream would say so.

**The anatomical anchors, stated before they are coded** (PLAN §4.4):

* the cleft crop spans **eyebrow to just below the lower vermillion border**;
* so the top edge comes from the **brow group** (22-42) and the bottom from the
  **lower-lip outer arc** (80-84), plus a small margin for "just below";
* ``H`` is that span, ``W = aspect_ratio x H`` with the ratio sampled from the
  measured cleft distribution, and the box is centred on the **landmark
  midline**;
* the trapezium half-widths then follow as fractions of ``W`` -- 0.301 at the
  top, 0.500 at the bottom -- exactly as for cleft.

Two things are asserted rather than assumed, because both are silent when wrong.

**The midline comes from the landmarks, never from the image centre.** SCUT faces
are not guaranteed centred in frame. Centring the crop on the image would put a
horizontal offset into the pretraining data that the cleft data does not have,
and the model would learn it as if it were anatomy.

**Levelling is a declared choice, not a default.** See ``SCUT_HEAD_TILT``.
"""

from __future__ import annotations

import numpy as np

from ..geometry.trapezium import BOT_HALF_WIDTH, TOP_HALF_WIDTH
from .landmarks import LANDMARK_GROUPS, MIDLINE_LANDMARKS, MIRROR_PAIRS

#: **[MEASURED 2026-07-28]** Which eye is which, derived from the data rather
#: than assumed: the eye group split by x against the face centre over 917 faces,
#: with a stable answer on every one and no ambiguous index.
EYE_LEFT = (43, 44, 45, 46, 47, 48, 49, 58)
EYE_RIGHT = (50, 51, 52, 53, 54, 55, 56, 57, 59)

#: **[MEASURED 2026-07-28] SCUT head tilt, over 917 sampled faces.** Intercanthal
#: angle against horizontal, degrees.
#:
#: The distribution is centred (median -0.15) but **wide**: SD 3.1 degrees, 37%
#: of faces beyond 2 degrees, 10% beyond 5, extremes near +/-17.
#:
#: **[OBSERVED 2026-07-29, Phase 2 cleft contact sheet] The cleft crops are NOT
#: levelled. So levelling is OFF, and the SCUT tilt spread is a MATCHED PROPERTY
#: rather than a defect.**
#:
#: Levelling SCUT would have *created* the domain gap it was meant to close: one
#: domain levelled and the other not is a difference that has nothing to do with
#: the mask, and the model would have had it available as a cue.
#:
#: **A finding from the same sheet, and it is not about alignment.** On the cleft
#: crops the apparent tilt is **worst at the nose tips, less at the eyes, least at
#: the lips**. Head roll cannot do that -- roll rotates every structure equally.
#: A nose deviating more than the eyes and the lips is what **nasal deviation**
#: looks like: septum and tip displaced off the facial midline, a hallmark of
#: unilateral cleft and one of the four things Asher-McDade scores.
#:
#: So it is probably **clinical signal, not misalignment** -- which strengthens
#: the case for leaving both domains unlevelled twice over. Correcting it would
#: suppress something that exists in only one of the two domains, and it is
#: something the label is partly *about*.
SCUT_HEAD_TILT_LEVELLING = "off"
SCUT_HEAD_TILT = {
    "n_sampled": 917,
    "median_deg": -0.149,
    "mean_deg": -0.339,
    "sd_deg": 3.099,
    "p05_deg": -5.710,
    "p95_deg": 4.075,
    "min_deg": -17.294,
    "max_deg": 16.880,
    "fraction_beyond_2deg": 0.374,
    "fraction_beyond_5deg": 0.099,
    "fraction_beyond_10deg": 0.014,
    "cleft_levelling_state": "NOT_LEVELLED",
    "cleft_levelling_evidence": "observed on the Phase 2 cleft contact sheet, 2026-07-29",
    "levelling": "off -- matched to the cleft domain",
    "apparent_cleft_tilt_is": (
        "probably nasal deviation, not head roll: worst at the nose tips, less at "
        "the eyes, least at the lips. Roll rotates every structure equally."
    ),
}

#: How far below the lower vermillion the crop reaches, as a fraction of the
#: brow-to-lip span. "Just below the lower vermillion border" (PLAN §4.4) needs a
#: number; this is it.
#:
#: **[REVIEWED 2026-07-29] Confirmed on the placement contact sheet** -- lower
#: vermillion visible, no chin, framing reading like the cleft crops. No sweep
#: was run and none is needed; it is a config field so one is a config change if
#: a later sheet disagrees.
LOWER_MARGIN = 0.06


class PlacementError(RuntimeError):
    """The trapezium could not be placed on this face."""


def eye_centres(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(image-left, image-right) eye centres."""
    array = np.asarray(points, dtype=float)
    return array[list(EYE_LEFT)].mean(axis=0), array[list(EYE_RIGHT)].mean(axis=0)


def intercanthal_angle(points: np.ndarray) -> float:
    """Angle of the eye-to-eye line against horizontal, in degrees."""
    left, right = eye_centres(points)
    return float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))


def midline_x(points: np.ndarray) -> float:
    """The face's own vertical midline, in pixels.

    **From the landmarks, never the image centre.** Three independent structures
    are averaged -- the eye-centre midpoint, the subnasale (66) and the
    cupid's-bow dip (76) -- so the eyes fix the horizontal centre of the upper
    face, the two midline landmarks fix it lower down, and a single mislabelled
    point cannot move the midline far.

    > **[MEASURED 2026-07-30] This replaced a definition that was displaced off
    > the midline on every face, and the correction is the whole reason the
    > detail above is spelled out.** The previous version averaged the eye
    > midpoint with ``mean(x)`` of nose indices 60-65, described as "the bridge
    > ... close to the midline on a frontal face". **The nose contour is a trace,
    > not a span**: 60-65 descend the image-LEFT flank of the nose and never
    > approach the midline (measured 0.117-0.238 of crop width from it), so that
    > term sat 0.167 crop widths to the image-left and dragged the average to
    > **-0.0915 crop widths -- about 10.6px on a 115px crop, negative on every
    > one of 300 faces** (range -0.209 to -0.035). The single midline point the
    > nose group does contain is 66, which the trace visits between the nostril
    > sills; the old slice stopped one index short of it.
    >
    > **Chosen on held-out structures, because the obvious score was circular.**
    > The first comparison ranked candidates by mean mirror-pair asymmetry
    > including the pairs that defined them -- and the pair centre is that
    > score's own minimiser, so it won by construction (PLAN R7's tally). The
    > decisive measurement scores candidates built only from the eye centres and
    > the midline singletons 66 and 76, none of which is a member of any scored
    > pair, over all 12 verified pairs on 500 faces:
    >
    > | midline | mean pair asymmetry (crop widths) |
    > |---|---|
    > | **(eye + 66 + 76) / 3** | **0.0260** |
    > | (66 + 76) / 2 | 0.0263 |
    > | 76 alone | 0.0274 |
    > | 66 alone | 0.0289 |
    > | eye midpoint alone | 0.0498 |
    > | previous (eye + mean 60-65) | **0.1714** |
    >
    > The chosen form is **6.6x better than the previous one and beats it on
    > 98.8% of faces**, while keeping the three-structure redundancy the old
    > docstring claimed and did not have. It is statistically tied with
    > ``(66 + 76) / 2``; the eye term is kept for that redundancy.
    >
    > **The fixture could not have caught this.** ``a_face`` in
    > ``test_scut_placement.py`` alternates sides within each group, so its nose
    > group averages to the midline by construction and any left-flank bias
    > cancels. A fixture matching the measured trace structure is now used for
    > the midline tests.
    """
    array = np.asarray(points, dtype=float)
    left, right = eye_centres(array)
    eye_mid = (left[0] + right[0]) / 2.0
    midline_points = array[list(MIDLINE_LANDMARKS), 0]
    return float((eye_mid + midline_points.sum()) / (1 + len(midline_points)))


def mirror_pair_asymmetry(points: np.ndarray, midline: float | None = None) -> dict:
    """How far off-centre a midline estimate is, measured on the mirror pairs.

    **The instrument that caught the displaced midline, kept so the next one
    cannot hide.** A facial midline must be equidistant from the two members of
    each mirror pair; the mean absolute discrepancy is therefore a direct check
    on any candidate, and it is reported in units of CROP WIDTH because that is
    what decides where the trapezium sits.

    ``signed_offset`` is the estimate minus the pairs' own centre: a *consistent
    sign across faces* is the signature of a structural bias rather than noise,
    and it is what distinguished the previous definition (negative on 300 of 300
    faces) from ordinary scatter.

    **Not itself the estimator.** The pair centre minimises this score by
    construction, so this measures a candidate and must never be used to pick
    one by minimising it -- see the held-out comparison in ``midline_x``.
    """
    array = np.asarray(points, dtype=float)
    if midline is None:
        midline = midline_x(array)
    top, bottom = vertical_span(array)
    # The width the mask is placed in, at the cohort median AR. Reporting in
    # pixels alone would make a 10px offset unreadable without knowing the crop.
    width = 0.7404 * (bottom - top)

    centres, gaps = [], []
    for a, b in MIRROR_PAIRS:
        centres.append((array[a, 0] + array[b, 0]) / 2.0)
        gaps.append(abs(abs(array[b, 0] - midline) - abs(midline - array[a, 0])))
    pair_centre = float(np.mean(centres))

    return {
        "midline_x": round(float(midline), 3),
        "pair_centre_x": round(pair_centre, 3),
        "n_pairs": len(MIRROR_PAIRS),
        "crop_width_px": round(float(width), 3),
        "mean_asymmetry_px": round(float(np.mean(gaps)), 3),
        "mean_asymmetry_frac_width": round(float(np.mean(gaps) / width), 6),
        "max_asymmetry_frac_width": round(float(np.max(gaps) / width), 6),
        "signed_offset_frac_width": round(float((midline - pair_centre) / width), 6),
    }


def vertical_span(points: np.ndarray, lower_margin: float = LOWER_MARGIN) -> tuple[float, float]:
    """(top, bottom) in pixels: brow to just below the lower vermillion."""
    array = np.asarray(points, dtype=float)
    top = float(array[list(LANDMARK_GROUPS["brows"]), 1].min())
    lip_bottom = float(array[list(LANDMARK_GROUPS["mouth"]), 1].max())
    span = lip_bottom - top
    if span <= 0:
        raise PlacementError(
            f"brow at y={top} is not above the lower lip at y={lip_bottom}; the "
            "landmark groups are inverted for this face"
        )
    return top, lip_bottom + lower_margin * span


def box_from_anchors(
    top: float, bottom: float, centre: float, aspect_ratio: float
) -> tuple[float, float, float, float]:
    """(x, y, w, h) from the three anchors, ONCE.

    **[EXTRACTED 2026-08-24]** ``crop_box`` reads its anchors from the
    86-point SCUT convention; Phase 15's MEBeauty set is dlib-68 and
    reads them from different indices. The ANCHORS necessarily differ;
    the FORMULA must not, so it lives here and both callers use it
    (``mebeauty.crop_box``). A second copy of "H from the anatomy, W =
    ratio x H, centred on the midline" is how two datasets end up
    framed differently while both configs say the same thing.
    """
    if aspect_ratio <= 0:
        raise PlacementError(f"aspect_ratio must be positive, got {aspect_ratio}")
    height = bottom - top
    if height <= 0:
        raise PlacementError(
            f"top {top} is not above bottom {bottom}: no content box exists"
        )
    width = aspect_ratio * height
    return (centre - width / 2.0, top, width, height)


def crop_box(
    points: np.ndarray,
    aspect_ratio: float,
    lower_margin: float = LOWER_MARGIN,
) -> tuple[float, float, float, float]:
    """(x, y, w, h) in pixels for the content box, centred on the midline.

    ``H`` comes from the anatomy; ``W = aspect_ratio x H`` with the ratio sampled
    from the measured cleft distribution, so masked SCUT varies in framing the
    way the clinical set does instead of presenting one constant shape.
    """
    if aspect_ratio <= 0:
        raise PlacementError(f"aspect_ratio must be positive, got {aspect_ratio}")

    top, bottom = vertical_span(points, lower_margin)
    return box_from_anchors(top, bottom, midline_x(points), aspect_ratio)


def box_is_inside(box: tuple[float, float, float, float], shape) -> bool:
    """Does the content box fit in the image? Reported, not silently padded.

    Cleft crops arrive pre-cropped and are fully inside their own frame. A SCUT
    box running off the edge would have to be padded, which puts padding *inside*
    the content box -- something no cleft crop has -- so the rate is measured and
    the sheet judges it.
    """
    x, y, w, h = box
    height, width = shape[:2]
    return x >= 0 and y >= 0 and x + w <= width and y + h <= height


def trapezium_half_widths() -> dict:
    """The mask fractions, taken from the frozen cleft definition, not restated."""
    return {"top": TOP_HALF_WIDTH, "bottom": BOT_HALF_WIDTH}


def describe(points: np.ndarray, aspect_ratio: float) -> dict:
    """Everything about one placement. SHAREABLE -- SCUT is public."""
    box = crop_box(points, aspect_ratio)
    top, bottom = vertical_span(points)
    return {
        "box": [round(float(v), 3) for v in box],
        "aspect_ratio": round(float(aspect_ratio), 4),
        "midline_x": round(midline_x(points), 3),
        "intercanthal_angle_deg": round(intercanthal_angle(points), 3),
        "brow_y": round(top, 3),
        "lower_crop_y": round(bottom, 3),
        "half_widths": trapezium_half_widths(),
        # Reported per face so a displaced midline shows up in the run rather
        # than in a later audit. See mirror_pair_asymmetry.
        "midline_check": mirror_pair_asymmetry(points),
    }
