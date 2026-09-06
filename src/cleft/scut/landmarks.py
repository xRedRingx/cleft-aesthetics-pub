"""SCUT-FBP5500's 86-point landmark files.

``.pts`` format, verified against the shipped files: little-endian ``int32``
count (86), then 86 x (x, y) ``float32``, in image pixel coordinates.

**The index semantics are not documented.** The dataset README describes the
files as "86 coordinates for each face" and says nothing about which index is an
eyebrow and which a lip corner. Phase 5 needs exactly that, to place the
trapezium against the same anatomical span the cleft crop covers.

**So the layout is derived from the data and checked visually, never assumed.**
Guessing landmark indices produces geometry that looks plausible and is wrong --
the trapezium would land on the wrong anatomy for every one of 5,499 faces, every
downstream masked variant would inherit it, and nothing would raise. ``annotate``
renders the indices onto a real face so the mapping can be read off a picture
rather than guessed from a convention.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

#: Every shipped file carries this count -- except the one that carries none.
N_LANDMARKS = 86

#: ``CM152.pts`` is **4 bytes, count 0**: empty as shipped by the authors, not a
#: detection failure on our side. Verified on the laptop and the cluster copies.
EMPTY_LANDMARK_FILES = ("CM152",)


#: **[MEASURED 2026-07-28] The 86-point layout, read off annotated renders and
#: then verified numerically.** The dataset documents none of this.
#:
#: Derivation: every point was drawn on four faces with its index labelled and
#: coloured by position, the groups read off the picture, and the reading then
#: checked on six faces by asserting the vertical ordering that anatomy requires
#: -- brows above eyes above nose above mouth. It holds on all six with clear
#: margins (e.g. AF1: brows 131, eyes 155, nose 185, mouth 231).
#:
#: ``FACE_OUTLINE`` runs clockwise from the top of the head: 0 at the crown, down
#: the image-left side to 11 at the chin, back up the image-right side to 21.
#: Within each half, index order is *spatial*, which is what lets a mirror pair
#: be found by reflection rather than by a hand-written table.
LANDMARK_GROUPS = {
    "face_outline": range(0, 22),
    "brows": range(22, 43),
    "eyes": range(43, 60),
    "nose": range(60, 73),
    "mouth": range(73, 86),
}

#: The vertical span the cleft crop covers: **eyebrow to just below the lower
#: vermillion border** (PLAN §4.4). These are the two groups that define it.
BROW_INDICES = LANDMARK_GROUPS["brows"]
MOUTH_INDICES = LANDMARK_GROUPS["mouth"]

#: **[MEASURED 2026-07-30] The nose and mouth contours are TRACES, not spans.**
#:
#: The coarse groups above were read off renders and then verified numerically.
#: The sub-group detail below was originally read off renders and **not**
#: verified, and one entry was wrong -- so it is now derived the same way the
#: groups were, over 300-500 faces, against the mirror-pair centre (the only
#: unbiased midline estimator; see ``placement.midline_x``).
#:
#: **The trace structure, measured.** Both contours run down one side, across the
#: bottom, and back up the other -- they do not sweep left-to-right:
#:
#: * nose: 60-64 descend the **image-left** flank (x from -0.12 to -0.24 of crop
#:   width), 65 the left nostril sill, **66 the midline** (x = +0.0003), 67 the
#:   right sill, 68-72 ascend the image-right flank.
#: * mouth: **73 the image-right commissure** (x = +0.311), 74-78 the upper
#:   vermillion arc leading leftward, **79 the image-left commissure**
#:   (x = -0.307), 80-85 the lower vermillion arc returning rightward.
#:
#: **The correction.** ``MOUTH_CORNERS`` read ``(79, 85)``. Index 85 is a
#: lower-lip point: |x| = 0.236 at y = 0.868, against the commissures' |x| = 0.31
#: at y = 0.81. The image-right commissure is **73**, which pairs with 79 to
#: within 0.004 of crop width -- the widest symmetric pair at corner height.
#: ``LOWER_LIP_OUTER`` also stopped at 84, excluding 85, which the same
#: measurement shows pairs with 80.
MOUTH_CORNERS = (79, 73)          # image-left, image-right [MEASURED]
UPPER_LIP_OUTER = range(74, 79)   # BETWEEN the commissures: the vermillion arc
LOWER_LIP_OUTER = range(80, 86)   # 85 included -- it pairs with 80 [MEASURED]

#: **[MEASURED 2026-07-30] Midline landmarks.** Mean |x| from the mirror-pair
#: centre over 300 faces, as a fraction of crop width: 66 is 0.0003, 76 is
#: 0.0045. Both are on the midline; **no index in 60-65 is** (0.117-0.238).
SUBNASALE = 66                    # columella base, lowest midline nose point
CUPIDS_BOW_DIP = 76               # labiale superius, the dip between the peaks
MIDLINE_LANDMARKS = (SUBNASALE, CUPIDS_BOW_DIP)

#: **[MEASURED 2026-07-30] The synthesis targets** (PLAN §4.11 Q-b), per side,
#: as (image-left, image-right).
#:
#: ``ALAR_BASE`` -- the widest nose pair, |x| = 0.242/0.239 at y = 0.60/0.59.
#: ``NOSTRIL_SILL`` -- |x| = 0.135 both sides at y = 0.65, the pair flanking 66.
#: ``CUPIDS_BOW_PEAK`` -- |x| = 0.090/0.086 at y = 0.769/0.772. Verified as
#: **peaks** and not merely arc points: both sit HIGHER than the dip at 76 on
#: **281 of 300** faces, which is the cupid's-bow signature.
ALAR_BASE = (64, 68)
NOSTRIL_SILL = (65, 67)
CUPIDS_BOW_PEAK = (77, 75)        # image-left, image-right

#: **The 86-point set has no philtral-column landmark**, and this is stated
#: rather than substituted silently. The philtral columns are the ridges running
#: from each cupid's-bow peak up to the nostril sill on the same side; the set
#: samples their two ENDPOINTS and nothing between. So a synthesis that displaces
#: "the philtral column" (PLAN §4.11 Q-b) displaces that segment's endpoints --
#: the peak and the sill -- and the column between them follows from the spline's
#: interpolation, not from a landmark. **[DECIDED 2026-07-30]**
PHILTRAL_COLUMN_ENDPOINTS = {"left": (77, 65), "right": (75, 67)}

#: **[MEASURED 2026-07-30] Verified mirror pairs**, nose and mouth. Each pair's
#: two members sit at the same height (|dy| <= 0.010 of the brow-to-lip span) and
#: equal distances from the mirror-pair centre (|d|x|| <= 0.017 of crop width).
#: Used to derive the midline and to check that a synthesis displaced one side.
MIRROR_PAIRS = (
    (60, 72), (61, 71), (62, 70), (63, 69), (64, 68), (65, 67),
    (73, 79), (74, 78), (75, 77), (80, 85), (81, 84), (82, 83),
)


class LandmarkError(RuntimeError):
    """A landmark file could not be read."""


def load_pts(path: str | Path) -> np.ndarray:
    """(86, 2) float array of (x, y) in pixels. Raises on the empty file."""
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) < 4:
        raise LandmarkError(
            f"{path.name} is {len(payload)} bytes and carries no count. "
            f"{path.stem} is shipped empty by the dataset authors if it is in "
            f"{EMPTY_LANDMARK_FILES}; that is a property of the data, not a "
            "failure here."
        )

    count = struct.unpack("<i", payload[:4])[0]
    if count <= 0:
        raise LandmarkError(
            f"{path.name} declares {count} landmarks -- shipped empty. See "
            "EMPTY_LANDMARK_FILES."
        )
    if count != N_LANDMARKS:
        raise LandmarkError(
            f"{path.name} declares {count} landmarks, expected {N_LANDMARKS}"
        )

    expected = 4 + count * 8
    if len(payload) < expected:
        raise LandmarkError(
            f"{path.name} is {len(payload)} bytes; {expected} needed for "
            f"{count} points"
        )
    points = np.frombuffer(payload[4:expected], dtype="<f4").reshape(count, 2)
    return np.asarray(points, dtype=np.float64)


def is_empty(path: str | Path) -> bool:
    """Is this one of the files shipped without landmarks?"""
    path = Path(path)
    if path.stem in EMPTY_LANDMARK_FILES:
        return True
    return path.stat().st_size < 8


def bounds(points: np.ndarray) -> dict:
    """Extent of a landmark set, for sanity checks and reporting."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise LandmarkError(f"expected an (N, 2) array, got {array.shape}")
    return {
        "x_min": float(array[:, 0].min()),
        "x_max": float(array[:, 0].max()),
        "y_min": float(array[:, 1].min()),
        "y_max": float(array[:, 1].max()),
        "n": int(array.shape[0]),
    }


def annotate(
    image: np.ndarray, points: np.ndarray, every: int = 1, radius: int = 2
) -> np.ndarray:
    """Draw the landmarks on a copy of the image, coloured by index.

    **This is how the index semantics get established.** Colour runs blue-to-red
    with the index, so the ordering of the 86 points is readable directly off the
    picture: which run traces the jaw, which the brows, which the lips. Guessing
    that from a published convention and being wrong would put the trapezium on
    the wrong anatomy for every face in the set.
    """
    canvas = np.asarray(image).copy()
    if canvas.ndim == 2:
        canvas = np.stack([canvas] * 3, axis=2)
    height, width = canvas.shape[:2]

    array = np.asarray(points, dtype=float)
    for index, (x, y) in enumerate(array[::every]):
        position = index * every
        fraction = position / max(len(array) - 1, 1)
        colour = np.array(
            [int(255 * fraction), 40, int(255 * (1.0 - fraction))], dtype=np.uint8
        )
        row, column = int(round(y)), int(round(x))
        top, bottom = max(row - radius, 0), min(row + radius + 1, height)
        left, right = max(column - radius, 0), min(column + radius + 1, width)
        if top < bottom and left < right:
            canvas[top:bottom, left:right] = colour
    return canvas
