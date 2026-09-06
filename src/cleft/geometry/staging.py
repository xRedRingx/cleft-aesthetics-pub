"""Stage a cleft crop to a 224x224 tensor.

PLAN §4.4, brief §2. **Pad-square-then-resize, white pad.**

Two alternatives are rejected and must not be reintroduced:

``stretch``
    Aspect ratio varies 0.553-1.099 across the cohort, so stretching distorts
    each patient by a *different* amount. On a task whose signal is left-right
    asymmetry, a per-patient geometric distortion is not a nuisance -- it is a
    confound correlated with nothing clinical.

``centre-crop``
    Deletes the nasal bridge at the top and the lower vermillion at the bottom,
    which is most of what the raters were looking at.

The pad is **white** because the corners already arrive white, baked in at
source. There are no un-masked originals: alpha is dropped at ``convert('RGB')``,
so any "background removal" reduces to choosing a fill, and two PNGs -- one
background-removed in an editor -- were bit-identical after loading, 0 of 36,278
pixels differing. Matching the existing fill keeps one fill value in the image
rather than two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Output side length. Square, because every backbone downstream expects it.
OUTPUT_SIZE = 224

#: The pad value, matching the white already baked into the corners.
PAD_VALUE = 255

#: Measured aspect-ratio range across the 237 crops (width / height).
AR_MIN, AR_MAX = 0.553, 1.099
AR_MEDIAN = 0.740

#: **[MEASURED] The cleft cohort's staged geometry, from the `p2-stage-1` run.**
#:
#: Recorded here so the Phase 5 geometry-parity check **sources them from the
#: codebase rather than from a brief**. A figure that lives only in a document
#: cannot be re-derived, and parity measured against a number nobody can trace is
#: not a measurement.
#:
#: These describe the 237 clinical crops after pad-square-then-resize. Phase 5's
#: masked SCUT set must land in the same distributions, and PLAN §4.4 and
#: `trapezium.py` carry the trapezium half-widths that go with them.
CLEFT_STAGED_GEOMETRY = {
    "source": "p2-stage-1 artifact (data/staged/staged_v1)",
    "n_patients": 237,
    "aspect_ratio": {"min": 0.5531, "median": 0.7404, "max": 1.0986},
    #: How much of the padded square is white padding rather than image.
    "pad_fraction": {"min": 0.0179, "mean": 0.2534, "max": 0.4464},
    "trapezium": {"top_hw": 0.301, "bot_hw": 0.500, "area_of_content": 0.802},
}

#: Shorthand for the figure the parity check compares against most often.
PAD_FRACTION_MEAN = 0.2534


class StagingError(ValueError):
    """The image cannot be staged."""


@dataclass(frozen=True)
class Staged:
    """A staged image and everything needed to map coordinates back.

    ``content_box`` is where the original pixels landed inside the padded square,
    in output pixel coordinates: (x0, y0, w, h). It is the whole reason per-image
    mapping works -- a patch expressed relative to the crop content can be placed
    correctly whatever the source aspect ratio was.
    """

    image: np.ndarray
    source_size: tuple[int, int]
    content_box: tuple[int, int, int, int]
    scale: float

    @property
    def aspect_ratio(self) -> float:
        width, height = self.source_size
        return width / height

    @property
    def pad_fraction(self) -> float:
        """How much of the output is padding rather than image."""
        _, _, w, h = self.content_box
        return 1.0 - (w * h) / (self.image.shape[0] * self.image.shape[1])


def _check(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise StagingError(f"expected a 2-D or 3-D image, got shape {array.shape}")
    if array.ndim == 3 and array.shape[2] not in (1, 3):
        raise StagingError(
            f"expected 1 or 3 channels, got {array.shape[2]}. Alpha is dropped at "
            "load; a 4-channel image here means the loader changed."
        )
    if min(array.shape[:2]) == 0:
        raise StagingError(f"image has a zero dimension: {array.shape}")
    return array


def resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour resize, dependency-free and exactly reproducible.

    Deliberately not an interpolating resize: staging must be deterministic and
    identical on the laptop and the cluster, and nearest-neighbour has no library
    version to disagree about. Phase 3 may swap this for a higher-quality filter,
    but only behind a config value and with the determinism gate re-run.
    """
    source_h, source_w = image.shape[:2]
    rows = np.floor(np.arange(height) * source_h / height).astype(int)
    cols = np.floor(np.arange(width) * source_w / width).astype(int)
    rows = np.clip(rows, 0, source_h - 1)
    cols = np.clip(cols, 0, source_w - 1)
    return image[rows[:, None], cols[None, :]]


def stage(image, size: int = OUTPUT_SIZE, pad_value: int = PAD_VALUE) -> Staged:
    """Pad to square with white, then resize to ``size`` x ``size``.

    Pad first, then resize: padding after resizing would mean the content had
    already been squashed to a square and the aspect ratio destroyed.
    """
    array = _check(image)
    source_h, source_w = array.shape[:2]

    # Scale the longer side to `size`, so the content fits exactly.
    scale = size / max(source_w, source_h)
    content_w = max(1, min(size, int(round(source_w * scale))))
    content_h = max(1, min(size, int(round(source_h * scale))))
    resized = resize_nearest(array, content_w, content_h)

    shape = (size, size, array.shape[2]) if array.ndim == 3 else (size, size)
    canvas = np.full(shape, pad_value, dtype=array.dtype)

    # Centre the content; an off-centre pad would put the same anatomy at
    # different heights for different aspect ratios.
    x0 = (size - content_w) // 2
    y0 = (size - content_h) // 2
    canvas[y0 : y0 + content_h, x0 : x0 + content_w] = resized

    return Staged(
        image=canvas,
        source_size=(source_w, source_h),
        content_box=(x0, y0, content_w, content_h),
        scale=scale,
    )


def content_to_output(staged: Staged, x: float, y: float) -> tuple[float, float]:
    """Map a point normalised to the CROP CONTENT into output pixel coordinates.

    This is the per-image mapping the brief calls load-bearing. A patch box is
    defined relative to the content, never to the padded square, so two patients
    with different aspect ratios get different pixel boxes for the same anatomy.
    Fixed boxes in padded-square coordinates would put the lips band on one
    patient's chin and another's nose, with no error raised.
    """
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise StagingError(f"({x}, {y}) is outside the normalised unit square")
    x0, y0, width, height = staged.content_box
    return x0 + x * width, y0 + y * height


def content_box_to_pixels(
    staged: Staged, box: tuple[float, float, float, float]
) -> tuple[int, int, int, int]:
    """Map a normalised (x, y, w, h) box into integer output pixel coordinates."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        raise StagingError(f"box has a non-positive dimension: {box}")
    left, top = content_to_output(staged, x, y)
    right, bottom = content_to_output(staged, min(1.0, x + w), min(1.0, y + h))
    return (
        int(round(left)),
        int(round(top)),
        max(1, int(round(right - left))),
        max(1, int(round(bottom - top))),
    )
