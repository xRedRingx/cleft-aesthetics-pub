"""Draw the geometry onto real crops, so it can be looked at.

Brief §7. This is the Phase 2 equivalent of the folder-scan diagnostic: cheap,
cluster-only, reviewed by eye and reported back in text. It answers the four
things no fixture can:

* do the band boundaries land sensibly -- eyes in the top band, alar bases off
  the eyes/nose seam, commissures inside the bottom band?
* does the centre patch of the bottom band cover the philtrum and cupid's bow?
* how much white do the outer G1 patches actually contain?
* does G2 look anatomically plausible, or grotesque?

If the boundaries land badly the fix is to MOVE them, from anatomy, on this
sheet, before anything trains. Never to tune them against a metric.

**Output is CLUSTER-ONLY and always will be.** It renders patient faces. The
tier guard in ``provenance.context`` refuses to mark any filename containing
"contact_sheet", "overlay" or "montage" as SHAREABLE, so this cannot be
mislabelled even by accident.

Drawing is pure numpy and therefore testable on the laptop; Pillow is used only
to load the real JPEGs and to save the finished sheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mapping import MappedPatch
from .staging import Staged
from .trapezium import Trapezium

#: One colour per band, so bands are separable at a glance. Mirror partners share
#: a colour -- they are the same measurement on two sides.
BAND_COLOURS: dict[str, tuple[int, int, int]] = {
    "top": (220, 50, 50),
    "middle": (50, 160, 220),
    "bottom": (60, 190, 90),
    "random": (200, 120, 30),
}

#: Fallback palette for the anatomy generator, whose "bands" are structure names.
PALETTE: tuple[tuple[int, int, int], ...] = (
    (220, 50, 50), (50, 160, 220), (60, 190, 90), (200, 120, 30),
    (170, 90, 200), (230, 190, 40), (40, 200, 190), (230, 110, 160),
    (120, 120, 220),
)

MASK_COLOUR = (255, 0, 255)
GRID_LABEL_HEIGHT = 18


class RenderError(ValueError):
    """The overlay could not be drawn."""


def _as_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    # **[ADDED 2026-08-14] Name the upstream mistake instead of its
    # symptom.** A 0-d array is what ``np.asarray`` returns for a Path or
    # None, so "got ()" is never a malformed image -- it is a caller that
    # passed the wrong thing. That reading cost a cluster run: the crop
    # sheet called ``save_sheet(path, sheet)`` with the arguments swapped
    # and the error pointed at the renderer, so the diagnosis went looking
    # for empty panel lists upstream. Both sheet defects so far have been
    # the render layer reporting something decided by its caller.
    if array.ndim == 0:
        raise RenderError(
            f"expected an image array, got a 0-d {type(image).__name__} "
            f"({image!r:.60}). That is not a malformed image -- it is the "
            "wrong argument: save_sheet takes (SHEET, PATH), tile takes a "
            "panel list. Check the call site's argument order."
        )
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RenderError(f"expected a greyscale or RGB image, got {array.shape}")
    return array.astype(np.uint8).copy()


def colour_for(band: str, index: int) -> tuple[int, int, int]:
    return BAND_COLOURS.get(band, PALETTE[index % len(PALETTE)])


def draw_rect(
    canvas: np.ndarray,
    box: tuple[int, int, int, int],
    colour: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    """Outline only -- filling would hide the anatomy being inspected."""
    x, y, w, h = box
    height, width = canvas.shape[:2]
    t = max(1, thickness)

    for edge_y in (y, y + h - t):
        top = max(0, min(height, edge_y))
        canvas[top : min(height, top + t), max(0, x) : min(width, x + w)] = colour
    for edge_x in (x, x + w - t):
        left = max(0, min(width, edge_x))
        canvas[max(0, y) : min(height, y + h), left : min(width, left + t)] = colour


def draw_mask_boundary(
    canvas: np.ndarray,
    staged: Staged,
    trapezium: Trapezium,
    colour: tuple[int, int, int] = MASK_COLOUR,
    thickness: int = 1,
) -> None:
    """Mark the trapezium edge, so white corners are visible against the patches."""
    x0, y0, cw, ch = staged.content_box
    height, width = canvas.shape[:2]

    for row in range(max(0, y0), min(height, y0 + ch)):
        y = (row - y0 + 0.5) / ch
        for edge in (trapezium.left_edge_at(y), trapezium.right_edge_at(y)):
            column = int(round(x0 + edge * cw))
            lo = max(0, column - thickness // 2)
            canvas[row, lo : min(width, lo + thickness)] = colour


#: Short forms for the anatomy structures. Without these the sheet shows coloured
#: boxes and no names, so it can only be judged on gross placement -- which makes
#: the next iteration impressionistic instead of "philtral_L is 0.04 too high".
ABBREVIATIONS: dict[str, str] = {
    "glabella": "glab",
    "nasal_root": "nroot",
    "nasal_dorsum_upper": "dors_u",
    "nasal_dorsum_lower": "dors_l",
    "nasal_tip": "tip",
    "columella": "colum",
    "subnasale": "subn",
    "philtrum": "philt",
    "labial_tubercle": "tuberc",
    "medial_canthus": "canth",
    "lateral_orbit": "orbit",
    "nasal_sidewall": "sidew",
    "alar_rim": "rim",
    "alar_base": "ala",
    "nostril_sill": "sill",
    "philtral_column": "philtral",
    "vermillion_border": "verm",
    "commissure": "comm",
}


def abbreviate(band: str, centre_x: float, clamped: bool = False) -> str:
    """``alar_base`` on the left becomes ``ala_L``; a clamped box gets a ``!``."""
    short = ABBREVIATIONS.get(band, band[:8])
    if abs(centre_x - 0.5) > 1e-9:
        short = f"{short}_{'L' if centre_x < 0.5 else 'R'}"
    return f"{short}!" if clamped else short


def render_overlay(
    staged: Staged,
    mapped: list[MappedPatch],
    trapezium: Trapezium | None = None,
    *,
    thickness: int = 1,
    labels: bool = True,
) -> np.ndarray:
    """One panel: the staged image, the mask boundary, and the patch boxes.

    ``labels`` names each box. On for anything being judged by eye: a sheet of
    unnamed boxes can only answer "does this look roughly right", and the point
    of the exercise is to check a specific region against a specific structure.
    """
    canvas = _as_rgb(staged.image)
    if trapezium is not None:
        draw_mask_boundary(canvas, staged, trapezium, thickness=thickness)
    for index, item in enumerate(mapped):
        draw_rect(canvas, item.pixels, colour_for(item.band, index), thickness)
    if labels:
        canvas = _draw_patch_labels(canvas, mapped)
    return canvas


def _draw_patch_labels(canvas: np.ndarray, mapped: list[MappedPatch]) -> np.ndarray:
    """Name each box, in its own colour, just inside its top-left corner."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    for index, item in enumerate(mapped):
        x, y, _, _ = item.pixels
        text = abbreviate(item.band, item.patch.centre_x, item.patch.clamped)
        # A black backing line, because the crops are pale and thin coloured
        # text on skin is unreadable at contact-sheet scale.
        draw.text((x + 2, y + 1), text, fill=(0, 0, 0))
        draw.text((x + 1, y), text, fill=colour_for(item.band, index))
    return np.asarray(image)


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Panel:
    label: str
    image: np.ndarray


def cell_for(panels: list[Panel]) -> tuple[int, int]:
    """The ``(width, height)`` of the grid cell ``tile`` will lay these in.

    **[CORRECTED 2026-08-09] One source of truth, because there were two.**
    ``tile`` computed the cell internally as the max over panel images, while
    three call sites passed ``save_sheet`` a cell derived from ``panels[0]``
    -- correct only while panels[0] happens to be the largest. The masked
    SCUT sheet at 768 broke that: the placement panel is source-scale (350px)
    and the masked panels 768, so every caption drifted by the difference and
    labels landed on neighbouring faces -- the reader-misleads-reviewer
    failure the missing-crop refusal exists for, one layer up. Callers use
    THIS, and ``tile`` uses it too, so the two cannot disagree again.
    """
    if not panels:
        raise RenderError("no panels to measure")
    images = [_as_rgb(p.image) for p in panels]
    return (
        max(im.shape[1] for im in images),
        max(im.shape[0] for im in images),
    )


def tile(panels: list[Panel], columns: int = 4, gap: int = 6) -> np.ndarray:
    """Lay panels out in a grid on a mid-grey background.

    Mid-grey rather than white: the crops are white-padded and the mask corners
    are white, so a white background would make it impossible to see where an
    image ends and the sheet begins -- which is one of the things being checked.
    """
    if not panels:
        raise RenderError("no panels to tile")
    if columns < 1:
        raise RenderError(f"columns must be positive, got {columns}")

    images = [_as_rgb(p.image) for p in panels]
    cell_w, cell_h = cell_for(panels)
    rows = (len(images) + columns - 1) // columns

    sheet = np.full(
        (
            rows * (cell_h + GRID_LABEL_HEIGHT + gap) + gap,
            columns * (cell_w + gap) + gap,
            3,
        ),
        128,
        dtype=np.uint8,
    )

    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y = gap + row * (cell_h + GRID_LABEL_HEIGHT + gap) + GRID_LABEL_HEIGHT
        x = gap + column * (cell_w + gap)
        sheet[y : y + image.shape[0], x : x + image.shape[1]] = image
    return sheet


def fit_label(draw, label: str, cell_width: int) -> str:
    """Trim a label to its cell, MARKING the trim with an ellipsis.

    **[2026-08-24] ``save_sheet`` never truncated at all**, and that was
    the defect. A label wider than its cell overflowed into the next
    column, and because labels are drawn in index order the NEXT
    column's label overdrew its tail -- cutting the name mid-word with
    nothing to say it had been cut (the last column simply ran off the
    sheet and was clipped). A reader then looks up a filename that does
    not exist, which is exactly how two Phase 15 diagnoses came back
    "no such file" (``phase15.SHEET_LABEL_DEFECT``).

    Trimming is not the fix by itself -- a trimmed name is still not a
    name. Callers that need lookup write a sidecar of FULL labels; this
    makes the loss visible so nobody mistakes a trimmed label for a
    filename.
    """
    if not label:
        return label

    def width(text: str) -> float:
        try:
            return float(draw.textlength(text))
        except AttributeError:  # very old Pillow
            return len(text) * 6.0

    limit = max(cell_width - 4, 8)
    if width(label) <= limit:
        return label
    ellipsis = "\u2026"
    trimmed = label
    while trimmed and width(trimmed + ellipsis) > limit:
        trimmed = trimmed[:-1]
    return trimmed + ellipsis


def save_sheet(sheet: np.ndarray, path: str | Path, labels: list[str] | None = None,
               columns: int = 4, gap: int = 6, cell: tuple[int, int] | None = None) -> Path:
    """Write the sheet as a PNG, with panel labels if Pillow can draw them.

    CLUSTER-ONLY. The caller must have claimed the path through
    ``ctx.path(..., tier="CLUSTER-ONLY")``, which the tier guard enforces by
    filename.
    """
    from PIL import Image, ImageDraw

    image = Image.fromarray(_as_rgb(sheet))
    if labels and cell:
        cell_w, cell_h = cell
        draw = ImageDraw.Draw(image)
        for index, label in enumerate(labels):
            row, column = divmod(index, columns)
            x = gap + column * (cell_w + gap)
            y = gap + row * (cell_h + GRID_LABEL_HEIGHT + gap) + 3
            draw.text(
                (x, y), fit_label(draw, label, cell_w), fill=(255, 255, 255)
            )

    path = Path(path)
    image.save(path, format="PNG")
    return path


def load_image(path: str | Path) -> np.ndarray:
    """Load a crop as RGB. Alpha is dropped here, deliberately and visibly.

    That drop is why pixel-level background removal is impossible: after it, a
    "background removed" PNG and its original are bit-identical.
    """
    from PIL import Image

    with Image.open(path) as handle:
        return np.asarray(handle.convert("RGB"))


# --------------------------------------------------------------------------
# what the sheet is for
# --------------------------------------------------------------------------


def overlay_report(mapped: list[MappedPatch], staged: Staged) -> dict:
    """Aggregate numbers to accompany the visual check. SHAREABLE.

    The sheet itself cannot leave the cluster, so these are what actually travel
    back in text alongside "the bands look right".
    """
    coverages = [m.patch.coverage for m in mapped]
    clamped = sorted({m.band for m in mapped if m.patch.clamped})
    return {
        "n_patches": len(mapped),
        "aspect_ratio": round(staged.aspect_ratio, 4),
        "pad_fraction": round(staged.pad_fraction, 4),
        "min_coverage": round(min(coverages), 4),
        "mean_coverage": round(float(np.mean(coverages)), 4),
        "n_patches_with_white": sum(1 for c in coverages if c < 1.0 - 1e-9),
        "clamped_regions": clamped,
        "n_clamped": sum(1 for m in mapped if m.patch.clamped),
    }
