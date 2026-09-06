"""Staging: pad-square-then-resize, white pad, 224.

The property that matters most is the last section: two images of different
aspect ratio must receive DIFFERENT pixel boxes for the same normalised anatomy.
Everything else is in service of that.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry import staging as S


def crop(width: int, height: int, value: int = 120) -> np.ndarray:
    """A synthetic crop with a distinguishable interior."""
    return np.full((height, width, 3), value, dtype=np.uint8)


# --------------------------------------------------------------------------
# shape and padding
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "width, height",
    [(100, 181), (150, 203), (203, 203), (220, 200), (100, 100)],
)
def test_output_is_always_square_224(width, height):
    staged = S.stage(crop(width, height))
    assert staged.image.shape == (224, 224, 3)


def test_the_pad_is_white():
    """Matching the fill already baked into the corners, not a second colour."""
    staged = S.stage(crop(100, 200))
    x0, y0, w, h = staged.content_box
    assert x0 > 0, "a tall crop must be padded left and right"
    assert (staged.image[:, :x0] == S.PAD_VALUE).all()
    assert (staged.image[:, x0 + w :] == S.PAD_VALUE).all()


def test_content_is_centred():
    """Off-centre padding would put the same anatomy at different heights."""
    staged = S.stage(crop(100, 200))
    x0, _, w, _ = staged.content_box
    assert abs(x0 - (224 - w - x0)) <= 1


def test_a_square_crop_needs_no_padding():
    staged = S.stage(crop(200, 200))
    assert staged.content_box == (0, 0, 224, 224)
    assert staged.pad_fraction == pytest.approx(0.0, abs=1e-9)


def test_aspect_ratio_is_preserved_not_stretched():
    """The rejected alternative, asserted so it cannot creep back."""
    for width, height in [(100, 181), (150, 203), (220, 200)]:
        staged = S.stage(crop(width, height))
        _, _, w, h = staged.content_box
        assert w / h == pytest.approx(width / height, abs=0.02), (
            "content aspect ratio changed: this is a stretch, which is rejected"
        )


def test_nothing_is_cropped_away():
    """The other rejected alternative: centre-crop deletes the bridge and lip."""
    staged = S.stage(crop(100, 200))
    _, _, w, h = staged.content_box
    assert max(w, h) == 224, "the longer side must fit exactly, not overflow"
    assert w > 0 and h > 0


def test_the_measured_aspect_ratio_range_is_recorded():
    assert S.AR_MIN == 0.553 and S.AR_MAX == 1.099
    assert S.AR_MIN < S.AR_MEDIAN < S.AR_MAX


@pytest.mark.parametrize("ratio", [0.553, 0.740, 1.099])
def test_the_real_aspect_ratio_extremes_stage_cleanly(ratio):
    height = 300
    staged = S.stage(crop(int(round(height * ratio)), height))
    assert staged.image.shape == (224, 224, 3)
    assert staged.aspect_ratio == pytest.approx(ratio, abs=0.01)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_staging_is_deterministic():
    """Exit criterion 6. Same input, byte-identical output, every time."""
    source = crop(137, 199)
    first = S.stage(source)
    second = S.stage(source)
    assert np.array_equal(first.image, second.image)
    assert first.content_box == second.content_box


def test_staging_does_not_mutate_its_input():
    source = crop(120, 180)
    before = source.copy()
    S.stage(source)
    assert np.array_equal(source, before)


def test_greyscale_is_accepted():
    staged = S.stage(np.full((180, 120), 90, dtype=np.uint8))
    assert staged.image.shape == (224, 224)


def test_a_four_channel_image_is_rejected():
    """Alpha is dropped at load; four channels means the loader changed."""
    with pytest.raises(S.StagingError, match="channel"):
        S.stage(np.zeros((100, 100, 4), dtype=np.uint8))


def test_a_zero_dimension_image_is_rejected():
    with pytest.raises(S.StagingError, match="zero dimension"):
        S.stage(np.zeros((0, 100, 3), dtype=np.uint8))


# --------------------------------------------------------------------------
# per-image mapping -- the safeguard
# --------------------------------------------------------------------------


def test_two_different_aspect_ratios_give_different_pixel_boxes():
    """Exit criterion 4, and the single most important assertion in Phase 2.

    Fixed boxes in padded-square coordinates would place the same normalised
    "lips" band on one patient's chin and another's nose, silently.
    """
    tall = S.stage(crop(100, 181))
    wide = S.stage(crop(220, 200))

    lips = (0.0, 2 / 3, 1.0, 1 / 3)
    assert S.content_box_to_pixels(tall, lips) != S.content_box_to_pixels(wide, lips)


def test_the_same_aspect_ratio_gives_the_same_box():
    """The converse: the mapping depends on geometry, not on identity."""
    a = S.stage(crop(100, 200))
    b = S.stage(crop(150, 300))
    box = (0.25, 0.25, 0.5, 0.5)
    assert S.content_box_to_pixels(a, box) == S.content_box_to_pixels(b, box)


def test_a_mapped_box_lands_inside_the_content_not_the_padding():
    for width, height in [(100, 181), (220, 200), (150, 203)]:
        staged = S.stage(crop(width, height))
        x0, y0, w, h = staged.content_box
        bx, by, bw, bh = S.content_box_to_pixels(staged, (0.0, 0.0, 1.0, 1.0))
        assert (bx, by) == (x0, y0)
        assert bw == pytest.approx(w, abs=1) and bh == pytest.approx(h, abs=1)


def test_the_full_content_box_maps_to_the_whole_content():
    staged = S.stage(crop(100, 200))
    assert S.content_to_output(staged, 0.0, 0.0) == (
        staged.content_box[0],
        staged.content_box[1],
    )


def test_coordinates_outside_the_unit_square_are_rejected():
    staged = S.stage(crop(100, 200))
    for x, y in [(-0.1, 0.5), (0.5, 1.2)]:
        with pytest.raises(S.StagingError, match="unit square"):
            S.content_to_output(staged, x, y)


def test_a_degenerate_box_is_rejected():
    staged = S.stage(crop(100, 200))
    with pytest.raises(S.StagingError, match="non-positive"):
        S.content_box_to_pixels(staged, (0.1, 0.1, 0.0, 0.2))


def test_pad_fraction_grows_with_aspect_ratio_extremity():
    """A useful diagnostic number: how much of the 224 square is padding."""
    square = S.stage(crop(200, 200))
    tall = S.stage(crop(110, 200))
    assert square.pad_fraction < tall.pad_fraction
    assert 0.0 <= tall.pad_fraction < 1.0
