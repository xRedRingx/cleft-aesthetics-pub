"""Per-image mapping — the load-bearing part of Phase 2.

Exit criterion 4: two synthetic images of different aspect ratio must receive
different pixel boxes. Everything else here supports that one property, because
its failure mode is silent: fixed boxes in padded-square coordinates produce
valid rectangles, a model that trains, and numbers that look fine.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry import mapping as M
from cleft.geometry import patches as P
from cleft.geometry import staging as S
from cleft.geometry.trapezium import Trapezium

G2 = Trapezium(top_half_width=0.5, bot_half_width=0.5)

#: The measured extremes of the cohort, and the median.
TALL = (int(round(300 * 0.553)), 300)
MEDIAN = (int(round(300 * 0.740)), 300)
WIDE = (int(round(300 * 1.099)), 300)


def crop(size, value: int = 120) -> np.ndarray:
    width, height = size
    return np.full((height, width, 3), value, dtype=np.uint8)


@pytest.fixture
def grid_patches():
    return P.get("grid").generate(G2)


@pytest.fixture
def tall():
    return S.stage(crop(TALL))


@pytest.fixture
def wide():
    return S.stage(crop(WIDE))


# --------------------------------------------------------------------------
# THE safeguard
# --------------------------------------------------------------------------


def test_different_aspect_ratios_give_different_pixel_boxes(tall, wide, grid_patches):
    """Exit criterion 4."""
    M.assert_varies_with_aspect_ratio(tall, wide, grid_patches)

    boxes_tall = [m.pixels for m in M.map_patches(tall, grid_patches)]
    boxes_wide = [m.pixels for m in M.map_patches(wide, grid_patches)]
    assert boxes_tall != boxes_wide


def test_the_safeguard_fires_when_boxes_do_not_vary(tall, wide, grid_patches, monkeypatch):
    """Prove it has teeth: simulate the bug it exists to catch.

    Padded-square coordinates ignore the content box, so every image gets the
    same boxes. The check must refuse that.
    """
    def fixed(staged, box):
        x, y, w, h = box
        size = staged.image.shape[0]
        return (
            int(round(x * size)),
            int(round(y * size)),
            max(1, int(round(w * size))),
            max(1, int(round(h * size))),
        )

    monkeypatch.setattr(M, "content_box_to_pixels", fixed)
    with pytest.raises(M.MappingError, match="IDENTICAL pixel boxes"):
        M.assert_varies_with_aspect_ratio(tall, wide, grid_patches)


def test_the_containment_guard_also_catches_the_padded_square_bug(tall, grid_patches, monkeypatch):
    """Defence in depth, and in practice this one fires first.

    The aspect-ratio check needs two images; containment catches the same bug on
    one, because a box in padded-square coordinates reaches into the pad of any
    image that is not square. Both are kept: containment is the earlier signal,
    the aspect-ratio check is the one that names the actual cause.
    """
    def fixed(staged, box):
        x, y, w, h = box
        size = staged.image.shape[0]
        return (
            int(round(x * size)),
            int(round(y * size)),
            max(1, int(round(w * size))),
            max(1, int(round(h * size))),
        )

    monkeypatch.setattr(M, "content_box_to_pixels", fixed)
    with pytest.raises(M.MappingError, match="padding"):
        M.map_patches(tall, grid_patches)


def test_the_safeguard_refuses_a_meaningless_comparison(grid_patches):
    """Two images of the same AR prove nothing, so comparing them is an error."""
    a = S.stage(crop((100, 200)))
    b = S.stage(crop((150, 300)))
    with pytest.raises(M.MappingError, match="same aspect ratio"):
        M.assert_varies_with_aspect_ratio(a, b, grid_patches)


def test_the_same_anatomy_tracks_the_content_box(grid_patches):
    """The positive statement: a band sits at the same FRACTION of the content."""
    for size in (TALL, MEDIAN, WIDE):
        staged = S.stage(crop(size))
        mapped = M.map_patches(staged, grid_patches)
        _, y0, _, ch = staged.content_box

        bottom = [m for m in mapped if m.band == "bottom"]
        top_of_band = min(m.pixels[1] for m in bottom)
        assert (top_of_band - y0) / ch == pytest.approx(2 / 3, abs=0.02)


# --------------------------------------------------------------------------
# containment
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [TALL, MEDIAN, WIDE, (200, 200)])
def test_no_patch_reaches_the_padding(size, grid_patches):
    staged = S.stage(crop(size))
    M.map_patches(staged, grid_patches)  # asserts internally


def test_a_patch_extending_into_the_pad_is_rejected(tall, grid_patches):
    mapped = M.map_patches(tall, grid_patches)
    bad = M.MappedPatch(mapped[0].patch, (0, 0, 224, 10))
    with pytest.raises(M.MappingError, match="padding"):
        M.assert_inside_content(tall, mapped + [bad])


def test_a_collapsed_patch_is_rejected(tall, grid_patches):
    mapped = M.map_patches(tall, grid_patches)
    bad = M.MappedPatch(mapped[0].patch, (10, 10, 0, 5))
    with pytest.raises(M.MappingError, match="collapsed"):
        M.assert_inside_content(tall, mapped + [bad])


def test_patches_cover_the_content_box_together(grid_patches):
    staged = S.stage(crop(MEDIAN))
    mapped = M.map_patches(staged, grid_patches)
    x0, y0, cw, ch = staged.content_box
    assert min(m.pixels[0] for m in mapped) == pytest.approx(x0, abs=1)
    assert max(m.pixels[0] + m.pixels[2] for m in mapped) == pytest.approx(x0 + cw, abs=1)
    assert min(m.pixels[1] for m in mapped) == pytest.approx(y0, abs=1)


# --------------------------------------------------------------------------
# mirror symmetry survives the mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [TALL, MEDIAN, WIDE])
def test_mirror_pairs_stay_mirrored_in_pixels(size, grid_patches):
    """Checked in pixel space, not inherited from the normalised pairing.

    An asymmetric content box or an off-by-one in rounding would put a left-right
    bias into the instrument used to measure left-right bias.
    """
    staged = S.stage(crop(size))
    M.assert_mirror_pairs_are_mirrored(staged, M.map_patches(staged, grid_patches))


def test_a_skewed_pair_is_caught(tall, grid_patches):
    mapped = M.map_patches(tall, grid_patches)
    victim = next(m for m in mapped if m.mirror_id is not None and m.mirror_id != m.id)
    shifted = M.MappedPatch(victim.patch, (victim.pixels[0] + 6, *victim.pixels[1:]))
    others = [m for m in mapped if m.id != victim.id]

    with pytest.raises(M.MappingError, match="not mirrored"):
        M.assert_mirror_pairs_are_mirrored(tall, others + [shifted])


def test_the_centre_patch_is_centred_in_pixels(grid_patches):
    staged = S.stage(crop(MEDIAN))
    x0, _, cw, _ = staged.content_box
    for item in M.map_patches(staged, grid_patches):
        if abs(item.patch.centre_x - 0.5) <= 1e-9:
            centre = item.pixels[0] + item.pixels[2] / 2
            assert centre == pytest.approx(x0 + cw / 2, abs=1)


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def test_every_patch_extracts_to_the_same_size(grid_patches):
    """Resolution by coverage: one output size, whatever the patch covers."""
    staged = S.stage(crop(MEDIAN))
    stack = M.extract(staged, M.map_patches(staged, grid_patches), output_size=64)
    assert stack.shape == (27, 64, 64, 3)


def test_a_narrow_patch_retains_more_detail_per_pixel(grid_patches):
    """A bottom patch covers 1/7 of the width, a top patch 1/3 -- at one size."""
    staged = S.stage(crop(MEDIAN))
    mapped = M.map_patches(staged, grid_patches)
    top_area = np.mean([m.area for m in mapped if m.band == "top"])
    bottom_area = np.mean([m.area for m in mapped if m.band == "bottom"])
    assert bottom_area < top_area


def test_extraction_is_deterministic(grid_patches):
    staged = S.stage(crop(MEDIAN))
    mapped = M.map_patches(staged, grid_patches)
    assert np.array_equal(
        M.extract(staged, mapped, 32), M.extract(staged, mapped, 32)
    )


def test_extraction_preserves_content(grid_patches):
    """A uniform crop must extract as that uniform value, not as padding."""
    staged = S.stage(crop(MEDIAN, value=77))
    stack = M.extract(staged, M.map_patches(staged, grid_patches), 16)
    assert (stack == 77).all(), "extraction picked up padding"


def test_greyscale_extracts(grid_patches):
    staged = S.stage(np.full((300, 222), 90, dtype=np.uint8))
    stack = M.extract(staged, M.map_patches(staged, grid_patches), 16)
    assert stack.shape == (27, 16, 16)


def test_a_bad_output_size_is_rejected(grid_patches):
    staged = S.stage(crop(MEDIAN))
    with pytest.raises(M.MappingError, match="output_size"):
        M.extract(staged, M.map_patches(staged, grid_patches), 0)


def test_mapping_an_empty_patch_list_is_rejected():
    with pytest.raises(M.MappingError, match="no patches"):
        M.map_patches(S.stage(crop(MEDIAN)), [])


# --------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------


def test_the_summary_is_geometry_only(grid_patches):
    staged = S.stage(crop(TALL))
    summary = M.mapping_summary(staged, M.map_patches(staged, grid_patches))
    assert summary["n_patches"] == 27
    assert summary["aspect_ratio"] == pytest.approx(0.553, abs=0.01)
    assert set(summary["per_band_mean_area_px"]) == {"top", "middle", "bottom"}
    assert "pixels" not in summary and "image" not in summary


def test_the_summary_records_the_pad_fraction(grid_patches):
    """A useful cluster diagnostic: how much of the frame is padding."""
    tall = S.stage(crop(TALL))
    wide = S.stage(crop(WIDE))
    a = M.mapping_summary(tall, M.map_patches(tall, grid_patches))
    b = M.mapping_summary(wide, M.map_patches(wide, grid_patches))
    assert a["pad_fraction"] > b["pad_fraction"]
