"""Map normalised patches into each image's own pixel coordinates.

Brief §5, and the part of Phase 2 the brief calls load-bearing.

Cleft aspect ratios span 0.553-1.099 — a factor of two. After pad-to-square, the
same anatomy therefore sits at a different height in the frame for different
patients. A patch box fixed in **padded-square** coordinates would put the "lips"
band on one patient's chin and another's nose, and nothing would raise: every
patch would still be a valid rectangle, every model would still train, and every
number would still be produced.

So patches are normalised to the **trapezium bounding box** and mapped through
each image's own ``content_box``. The trapezium bbox and the content box coincide
by construction — the trapezium spans the full content width at y=1 and the full
content height — which is why one mapping serves both.

``assert_varies_with_aspect_ratio`` is the safeguard the brief asks for. It is
cheap, it runs at build time, and it is the only thing standing between a silent
misalignment and a plausible-looking result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .patches import Patch
from .staging import Staged, content_box_to_pixels, resize_nearest


class MappingError(ValueError):
    """A patch could not be placed in this image."""


@dataclass(frozen=True)
class MappedPatch:
    """One patch, placed in one image's output pixel coordinates."""

    patch: Patch
    pixels: tuple[int, int, int, int]

    @property
    def id(self) -> int:
        return self.patch.id

    @property
    def band(self) -> str:
        return self.patch.band

    @property
    def mirror_id(self) -> int | None:
        return self.patch.mirror_id

    @property
    def area(self) -> int:
        return self.pixels[2] * self.pixels[3]


def map_patches(staged: Staged, patches: list[Patch]) -> list[MappedPatch]:
    """Place every patch into this image's pixel grid."""
    if not patches:
        raise MappingError("no patches to map")
    mapped = [MappedPatch(p, content_box_to_pixels(staged, p.box)) for p in patches]
    assert_inside_content(staged, mapped)
    return mapped


def assert_inside_content(staged: Staged, mapped: list[MappedPatch]) -> None:
    """No patch may extend into the padding or off the image.

    A patch overlapping the pad would feed the model white bars whose width
    depends on the patient's aspect ratio — a per-patient artefact correlated
    with nothing clinical.
    """
    x0, y0, cw, ch = staged.content_box
    size = staged.image.shape[0]

    for item in mapped:
        px, py, pw, ph = item.pixels
        if pw < 1 or ph < 1:
            raise MappingError(
                f"patch {item.id} collapsed to {pw}x{ph} pixels in this image"
            )
        if px < 0 or py < 0 or px + pw > size or py + ph > size:
            raise MappingError(
                f"patch {item.id} at {item.pixels} falls outside the {size}x{size} image"
            )
        # One pixel of slack for rounding at the content edges.
        if px + 1 < x0 or py + 1 < y0 or px + pw > x0 + cw + 1 or py + ph > y0 + ch + 1:
            raise MappingError(
                f"patch {item.id} at {item.pixels} extends into the padding "
                f"(content box {staged.content_box}). Patches are normalised to "
                "the trapezium bbox and must never reach the pad."
            )


def assert_mirror_pairs_are_mirrored(
    staged: Staged, mapped: list[MappedPatch], tolerance: int = 1
) -> None:
    """Mirror pairs must still be mirrored after the mapping.

    The mapping is a uniform scale and translation, so this should hold exactly
    up to rounding. Checking it in PIXEL space rather than trusting the normalised
    pairing is what catches an asymmetric content box or an off-by-one in the
    rounding — either of which would put a left-right bias into the instrument
    used to measure left-right bias.
    """
    x0, _, cw, _ = staged.content_box
    by_id = {item.id: item for item in mapped}

    for item in mapped:
        if item.mirror_id is None or item.mirror_id not in by_id:
            continue
        partner = by_id[item.mirror_id]
        px, _, pw, _ = item.pixels
        expected_x = 2 * x0 + cw - px - pw
        if abs(partner.pixels[0] - expected_x) > tolerance:
            raise MappingError(
                f"patch {item.id} and its mirror {partner.id} are not mirrored in "
                f"pixels: {item.pixels} vs {partner.pixels}, expected x="
                f"{expected_x}. The asymmetry instrument is skewed."
            )
        if abs(partner.pixels[2] - pw) > tolerance:
            raise MappingError(
                f"mirror patches {item.id} and {partner.id} differ in width: "
                f"{pw} vs {partner.pixels[2]}"
            )


def assert_varies_with_aspect_ratio(
    a: Staged, b: Staged, patches: list[Patch]
) -> None:
    """THE safeguard (brief §5).

    Two images of different aspect ratio must receive different pixel boxes. If
    they do not, patches are effectively fixed in padded-square coordinates and
    the anatomy they cover drifts silently between patients.
    """
    if abs(a.aspect_ratio - b.aspect_ratio) < 1e-6:
        raise MappingError(
            "these two images have the same aspect ratio, so this check proves "
            f"nothing (both {a.aspect_ratio:.4f}). Pick images that differ."
        )

    # Boxes computed directly rather than through map_patches, so this check
    # tests ONLY its own property. Routing through map_patches would let the
    # containment guard raise first and leave this one never exercised in
    # isolation -- which is exactly what happened the first time it was written.
    boxes_a = [content_box_to_pixels(a, p.box) for p in patches]
    boxes_b = [content_box_to_pixels(b, p.box) for p in patches]

    if boxes_a == boxes_b:
        raise MappingError(
            f"two images with aspect ratios {a.aspect_ratio:.4f} and "
            f"{b.aspect_ratio:.4f} received IDENTICAL pixel boxes. Patches are "
            "being placed in padded-square coordinates rather than mapped through "
            "each image's content box, so the same box covers different anatomy "
            "in different patients."
        )


def extract(
    staged: Staged, mapped: list[MappedPatch], output_size: int
) -> np.ndarray:
    """Crop each patch and resize to one common size.

    "Resolution by coverage": every patch gets the same output size, so a bottom
    patch covering 1/7 of the width carries more detail than a top patch covering
    1/3. Output size is deliberately not varied per band.
    """
    if output_size < 1:
        raise MappingError(f"output_size must be positive, got {output_size}")

    channels = staged.image.shape[2:] if staged.image.ndim == 3 else ()
    out = np.empty((len(mapped), output_size, output_size, *channels), dtype=staged.image.dtype)

    for index, item in enumerate(mapped):
        x, y, w, h = item.pixels
        crop = staged.image[y : y + h, x : x + w]
        out[index] = resize_nearest(crop, output_size, output_size)
    return out


def mapping_summary(staged: Staged, mapped: list[MappedPatch]) -> dict:
    """Aggregate description. SHAREABLE — geometry only, no pixel values."""
    areas = [m.area for m in mapped]
    by_band: dict[str, list[int]] = {}
    for item in mapped:
        by_band.setdefault(item.band, []).append(item.area)

    return {
        "n_patches": len(mapped),
        "aspect_ratio": round(staged.aspect_ratio, 4),
        "content_box": list(staged.content_box),
        "pad_fraction": round(staged.pad_fraction, 4),
        "patch_area_px_min": min(areas),
        "patch_area_px_max": max(areas),
        "patch_area_px_mean": round(float(np.mean(areas)), 1),
        "per_band_mean_area_px": {
            band: round(float(np.mean(values)), 1)
            for band, values in sorted(by_band.items())
        },
    }
