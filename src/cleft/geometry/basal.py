"""Basal (submental) staging support -- Phase 12, stop 2.

**No anatomical parameter lives here, and that is the finding**
(``phase12.STOP_2_STAGING_DECISION``): the whole-image G1 path is
``staging.stage()`` -- pad-square-white-centre-resize -- whose only input
fact about the image is its aspect ratio. The frontal trapezium's
measured half-widths are consumed by G2's unwarp and by the patch
generators, and Phase 12's arms use neither. This module holds the two
MEASUREMENTS the frontal-only characterisation left open, with their
registered thresholds, and nothing else.

The frozen modules (``staging``, ``trapezium``, ``mapping``) are imported,
never modified.
"""

from __future__ import annotations

import numpy as np

#: Corner square side, in SOURCE pixels, for the whiteness measurement.
#: The white-pad rationale is about the SOURCE crop's corners ("the
#: corners already arrive white, baked in at source" -- a frontal
#: measurement), so it is measured there, not on the staged output.
CORNER_SIDE = 16

#: A pixel counts as white at or above this, per channel. Not 255 exactly:
#: JPEG ringing puts baked-white corners at 250-255, and demanding 255
#: would flag compression noise rather than background colour.
WHITE_LEVEL = 250

#: Per-image flag: below this corner-white fraction an image is FLAGGED on
#: the review sheets. Registered before any basal pixel was read.
FLAG_BELOW = 0.90

#: The reopening threshold, registered before any pixel: if the COHORT
#: MEDIAN corner-white fraction is below this, the white-pad rationale
#: measured on frontals does not transfer to basals, and the staging
#: decision REOPENS before any embedding is extracted.
REOPEN_IF_MEDIAN_BELOW = 0.90

#: The frontal cohort's measured aspect-ratio range, carried for the
#: side-by-side report. Informational: ``stage()`` is total over any
#: ratio, so a basal ratio outside this range is reported and shown to
#: the eye, never refused.
FRONTAL_AR_RANGE = (0.553, 1.099)

#: Review-sheet layout: every one of the 236 staged crops is rendered, in
#: ascending corner-white order so the most suspicious images land on
#: sheet 1, in front of the eye first.
SHEET_COLUMNS = 6
PANELS_PER_SHEET = 24


class BasalStagingError(ValueError):
    """A basal-staging contract is not usable."""


def pad_fraction_for_ar(aspect_ratio: float) -> float:
    """Pad fraction as a pure function of the aspect ratio.

    Exact for pad-square-then-resize up to integer rounding (verified
    against the frozen ``staging.stage()`` on synthetic images, agreement
    to ~0.002): ``1 - ar`` for tall images, ``1 - 1/ar`` for wide ones.
    Zero at ar = 1, growing toward both extremes -- so on any AR interval
    the maximum pad sits at an ENDPOINT, which is what lets a cohort's
    pad be bounded from its AR range alone
    (``phase12.PAD_DECISION_REOPENED``)."""
    if aspect_ratio <= 0:
        raise BasalStagingError(f"aspect ratio {aspect_ratio} is not positive")
    return (
        1.0 - aspect_ratio if aspect_ratio <= 1.0 else 1.0 - 1.0 / aspect_ratio
    )


def corner_white_fraction(image: np.ndarray, side: int = CORNER_SIDE,
                          level: int = WHITE_LEVEL) -> float:
    """Fraction of the four source-corner squares that is white.

    A pixel is white when EVERY channel is at or above ``level``. Measured
    on the source image because that is where the frontal rationale was
    measured; the staged output's corners are white by construction (the
    pad) and would measure 1.0 on any image at all.
    """
    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    h, w = array.shape[:2]
    s = min(side, h, w)
    corners = [
        array[:s, :s], array[:s, w - s:], array[h - s:, :s], array[h - s:, w - s:],
    ]
    white = [np.all(c >= level, axis=2) for c in corners]
    return float(np.mean([np.mean(m) for m in white]))


def whiteness_verdict(fractions: list[float]) -> dict:
    """Apply the registered thresholds to the measured cohort.

    Returns the verdict the run stamps into its metrics -- computed by
    rule, not read off a plot.
    """
    values = np.asarray(fractions, dtype=float)
    if values.size == 0:
        raise BasalStagingError("no corner-white fractions to judge")
    median = float(np.median(values))
    flagged = int(np.sum(values < FLAG_BELOW))
    reopen = bool(median < REOPEN_IF_MEDIAN_BELOW)
    return {
        "median": median,
        "min": float(values.min()),
        "max": float(values.max()),
        "flagged_below_0_90": flagged,
        "reopen_threshold": REOPEN_IF_MEDIAN_BELOW,
        "white_pad_rationale_transfers": not reopen,
        "verdict": (
            "REOPEN -- the cohort median corner-white fraction is below "
            f"{REOPEN_IF_MEDIAN_BELOW}: the white-pad rationale measured "
            "on frontals does NOT transfer, and the staging decision "
            "reopens before any embedding is extracted"
            if reopen else
            "TRANSFERS -- the basal corners are white-dominant like the "
            "frontals', so pad value 255 keeps one fill in the image "
            "rather than two. Flagged individuals still go in front of "
            "the eye, first"
        ),
    }
