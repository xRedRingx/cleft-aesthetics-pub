"""The trapezium mask, and the gate that decides whether G2 exists.

Brief §3.3: "this is not verified and G2 must not be used until it is." The
asymmetry section below IS that verification. If it fails, G2 is dead and G1 is
the only geometry — so these tests are not a formality, they are the decision.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry import trapezium as T

TOL = 1e-9


# --------------------------------------------------------------------------
# the measured geometry
# --------------------------------------------------------------------------


def test_the_measured_constants():
    assert T.TOP_HALF_WIDTH == 0.301
    assert T.BOT_HALF_WIDTH == 0.500
    assert T.MIDLINE == 0.5


def test_the_trapezium_widens_from_brow_to_lip():
    tz = T.DEFAULT
    assert tz.half_width_at(0.0) < tz.half_width_at(0.5) < tz.half_width_at(1.0)
    assert tz.half_width_at(0.0) == pytest.approx(0.301, abs=TOL)
    assert tz.half_width_at(1.0) == pytest.approx(0.500, abs=TOL)


def test_the_area_matches_the_measured_80_percent():
    assert T.DEFAULT.area_fraction() == pytest.approx(
        T.EXPECTED_AREA_FRACTION, abs=0.002
    )


def test_the_trapezium_is_symmetric_about_the_midline():
    """Asymmetry is the signal; a lopsided mask would manufacture some."""
    tz = T.DEFAULT
    for y in (0.0, 0.3, 0.7, 1.0):
        half = tz.half_width_at(y)
        assert tz.contains(0.5 - half, y)
        assert tz.contains(0.5 + half, y)
        assert not tz.contains(0.5 - half - 1e-6, y)
        assert not tz.contains(0.5 + half + 1e-6, y)


def test_scalar_and_array_edges_agree():
    """The vectorised forms duplicate the arithmetic; pin them to the scalars.

    They exist for speed -- coverage used to loop over rows in Python -- so the
    duplication has to be held in place by a test rather than by care.
    """
    heights = np.linspace(0.0, 1.0, 21)
    for trapezium in (T.DEFAULT, T.Trapezium(skew=0.1), T.Trapezium(midline=0.45)):
        assert trapezium.left_edge_array(heights) == pytest.approx(
            [trapezium.left_edge_at(float(y)) for y in heights], abs=TOL
        )
        assert trapezium.right_edge_array(heights) == pytest.approx(
            [trapezium.right_edge_at(float(y)) for y in heights], abs=TOL
        )
        assert trapezium.half_width_array(heights) == pytest.approx(
            [trapezium.half_width_at(float(y)) for y in heights], abs=TOL
        )


def test_the_array_form_clips_out_of_range_heights():
    """The scalar form raises; the array form is used on patch extents that may
    sit a hair outside [0, 1] after rounding, so it clamps instead."""
    assert T.DEFAULT.half_width_array(np.array([-0.1, 1.1])) == pytest.approx(
        [T.TOP_HALF_WIDTH, T.BOT_HALF_WIDTH], abs=TOL
    )


def test_a_height_outside_the_unit_interval_is_rejected():
    with pytest.raises(T.TrapeziumError, match="outside"):
        T.DEFAULT.half_width_at(1.5)


# --------------------------------------------------------------------------
# the G1 mask
# --------------------------------------------------------------------------


def test_the_mask_is_narrow_at_the_top_and_full_at_the_bottom():
    m = T.mask(224)
    assert m.shape == (224, 224)
    assert m[0].sum() < m[-1].sum()
    assert m[-1].sum() == pytest.approx(224, abs=2), "the lip row spans full width"


def test_the_mask_is_left_right_symmetric():
    m = T.mask(224)
    assert np.array_equal(m, m[:, ::-1])


def test_the_mask_covers_about_80_percent():
    assert T.mask(512).mean() == pytest.approx(T.EXPECTED_AREA_FRACTION, abs=0.01)


def test_the_corners_are_outside_the_mask():
    """The white corners G1 excludes from computation rather than from pixels."""
    m = T.mask(224)
    assert not m[0, 0] and not m[0, -1]


# --------------------------------------------------------------------------
# THE GATE — G2 must not be used until this passes
# --------------------------------------------------------------------------


def test_the_gate_passes():
    """Exit criterion 5. If this ever fails, G2 is dead — do not weaken it."""
    passed, checks = T.asymmetry_is_preserved()
    assert passed, [
        (c.height, c.offset_in, c.recovered, c.error) for c in checks if c.error > 1e-9
    ]
    assert len(checks) >= 15, "the gate must test several heights and offsets"


def test_asymmetry_is_recovered_exactly_at_every_height():
    """The scale factor varies with row, so one height would prove nothing."""
    for y in (0.0, 0.25, 0.5, 0.75, 1.0):
        for offset in (0.02, 0.05, 0.10):
            if offset > T.DEFAULT.half_width_at(y):
                continue
            check = T.asymmetry_at(y, offset)
            assert check.recovered == pytest.approx(offset, abs=TOL), (
                f"asymmetry lost at height {y}, offset {offset}"
            )


def test_the_scale_factor_really_does_vary_with_height():
    """If it did not, the whole concern would be hypothetical."""
    top = T.asymmetry_at(0.0, 0.05)
    bottom = T.asymmetry_at(1.0, 0.05)
    assert top.scale_factor > bottom.scale_factor
    assert bottom.scale_factor == pytest.approx(1.0, abs=TOL), (
        "the lip row already spans full width, so it is not stretched"
    )
    assert top.scale_factor == pytest.approx(1 / 0.602, abs=1e-6)


def test_the_raw_output_offset_is_magnified_but_the_recovery_is_not():
    """Distances DO change; what survives is asymmetry after the known rescale."""
    check = T.asymmetry_at(0.0, 0.05)
    assert check.offset_out > check.offset_in, "the top row is stretched"
    assert check.recovered == pytest.approx(check.offset_in, abs=TOL)


def test_a_symmetric_pair_stays_symmetric():
    """The strongest statement of the property: mirrored in, mirrored out."""
    for y in (0.0, 0.4, 1.0):
        for offset in (0.03, 0.08):
            if offset > T.DEFAULT.half_width_at(y):
                continue
            left = T.unwarp_x(0.5 - offset, y)
            right = T.unwarp_x(0.5 + offset, y)
            assert (left + right) / 2 == pytest.approx(0.5, abs=TOL)
            assert 0.5 - left == pytest.approx(right - 0.5, abs=TOL)


def test_the_gate_rejects_an_offset_outside_the_mask():
    with pytest.raises(T.TrapeziumError, match="outside the trapezium"):
        T.asymmetry_at(0.0, 0.4)


@pytest.mark.parametrize("skew", [0.05, -0.05, 0.2])
def test_the_gate_fails_on_a_skewed_stretch(skew):
    """Prove the gate has teeth. A skewed mask magnifies +d and -d differently.

    Without this, a gate that returned True unconditionally would be
    indistinguishable from one that verified something.
    """
    passed, checks = T.asymmetry_is_preserved(T.Trapezium(skew=skew))
    assert not passed, f"the gate passed a stretch skewed by {skew}"
    assert any(c.centre_error > 1e-9 for c in checks), (
        "a mirrored pair must stop straddling the midline under a skewed stretch"
    )


def test_a_displaced_midline_is_NOT_what_this_gate_catches():
    """Recorded because it was my first attempt at the teeth test, and it was wrong.

    A trapezium whose midline sits at 0.42 is still symmetric ABOUT THAT MIDLINE,
    so it preserves asymmetry perfectly and the gate rightly passes it. Perturbing
    the midline does not perturb the property under test; it breaks midline
    IDENTIFICATION, which is a different question and not this gate's job.
    """
    passed, _ = T.asymmetry_is_preserved(T.Trapezium(midline=0.42))
    assert passed, (
        "a displaced-but-symmetric trapezium should still preserve asymmetry"
    )


def test_the_gate_measures_a_mirrored_pair_not_a_single_point():
    """A single displaced point cannot tell preservation from a uniform shift."""
    check = T.asymmetry_at(0.3, 0.05)
    assert check.centre_out == pytest.approx(0.5, abs=TOL)
    assert check.centre_error == pytest.approx(0.0, abs=TOL)


# --------------------------------------------------------------------------
# the unwarp itself
# --------------------------------------------------------------------------


def test_unwarp_removes_the_white_corners():
    """Exit criterion 2: every patch inside the mask under G2."""
    size = 128
    image = np.where(T.mask(size), 100, 255).astype(np.uint8)
    out = T.unwarp(image)
    assert (out == 255).sum() == 0, "unwarping must leave no white corner behind"


def test_unwarp_preserves_left_right_symmetry_of_the_image():
    size = 128
    image = np.where(T.mask(size), 100, 255).astype(np.uint8)
    out = T.unwarp(image)
    assert np.array_equal(out, out[:, ::-1])


def test_unwarp_moves_a_mark_the_predicted_distance():
    """An actual pixel, not just the coordinate function."""
    size = 200
    image = np.full((size, size), 0, dtype=np.uint8)
    y = 0.0
    row = 0
    column = int(round((0.5 + 0.05) * size))
    image[row, column] = 255

    out = T.unwarp(image)
    found = np.flatnonzero(out[row] == 255)
    assert found.size >= 1

    expected = T.unwarp_x(0.5 + 0.05, (row + 0.5) / size) * size
    assert abs(float(found.mean()) - expected) <= 2.0


def test_unwarp_is_deterministic():
    image = np.where(T.mask(64), 100, 255).astype(np.uint8)
    assert np.array_equal(T.unwarp(image), T.unwarp(image))


def test_unwarp_rejects_a_bad_shape():
    with pytest.raises(T.TrapeziumError, match="2-D or 3-D"):
        T.unwarp(np.zeros((4, 4, 4, 4)))
