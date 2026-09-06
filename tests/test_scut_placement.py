"""Placing the cleft trapezium on a SCUT face (Phase 5).

Synthetic landmark sets shaped like real ones: brows above eyes above nose above
mouth, in the index groups derived in `scut.landmarks`. The real faces are
checked on the contact sheet, which is the only thing that can answer whether
the trapezium lands on the right anatomy.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.scut import landmarks, placement


def a_face(midline: float = 175.0, tilt_deg: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """86 points with the real group layout and a known midline and tilt."""
    points = np.zeros((landmarks.N_LANDMARKS, 2), dtype=float)
    rng = np.random.default_rng(0)

    def place(indices, y_centre, spread_x, spread_y=4.0):
        for offset, index in enumerate(indices):
            side = -1 if offset % 2 == 0 else 1
            points[index] = (
                midline + side * spread_x * (0.4 + 0.6 * rng.random()),
                y_centre + spread_y * (rng.random() - 0.5),
            )

    place(landmarks.LANDMARK_GROUPS["face_outline"], 170 * scale, 80)
    place(landmarks.LANDMARK_GROUPS["brows"], 120 * scale, 45)
    place(landmarks.LANDMARK_GROUPS["nose"], 190 * scale, 15)
    place(landmarks.LANDMARK_GROUPS["mouth"], 235 * scale, 30)

    # The eyes must respect the MEASURED left/right split, not alternate: the
    # intercanthal angle is the line between the two eye centres, so a fixture
    # that scatters them measures nothing.
    for index in placement.EYE_LEFT:
        points[index] = (midline - 40 + 8 * rng.random(), 150 * scale + 3 * rng.random())
    for index in placement.EYE_RIGHT:
        points[index] = (midline + 32 + 8 * rng.random(), 150 * scale + 3 * rng.random())

    if tilt_deg:
        radians = np.radians(tilt_deg)
        rotation = np.array(
            [[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]]
        )
        centre = np.array([midline, 175 * scale])
        points = (points - centre) @ rotation.T + centre
    return points


#: **[MEASURED 2026-07-30] The real nose and mouth trace geometry**, as
#: (x from the facial midline, y from the brow) in units of crop width and
#: brow-to-lip span, averaged over 300 SCUT faces.
#:
#: This table exists because ``a_face`` **could not detect a displaced midline**.
#: It alternates sides within each group, so its nose group averages to the
#: midline by construction and a left-flank bias cancels exactly -- which is why
#: the previous ``midline_x``, off by 0.09 crop widths on every real face, passed
#: its test. A fixture symmetric by construction cannot test a symmetry
#: assumption (PLAN R7's running tally).
MEASURED_TRACE = {
    60: (-0.1204, 0.2348), 61: (-0.1366, 0.3335), 62: (-0.1629, 0.4251),
    63: (-0.2055, 0.5125), 64: (-0.2423, 0.6023), 65: (-0.1352, 0.6495),
    66: (+0.0003, 0.6761), 67: (+0.1349, 0.6476), 68: (+0.2388, 0.5933),
    69: (+0.2009, 0.5092), 70: (+0.1558, 0.4249), 71: (+0.1237, 0.3325),
    72: (+0.1040, 0.2325), 73: (+0.3109, 0.8051), 74: (+0.1913, 0.7842),
    75: (+0.0899, 0.7688), 76: (+0.0045, 0.7886), 77: (-0.0863, 0.7720),
    78: (-0.1896, 0.7920), 79: (-0.3067, 0.8146), 80: (-0.2278, 0.8777),
    81: (-0.1443, 0.9191), 82: (-0.0444, 0.9420), 83: (+0.0605, 0.9410),
    84: (+0.1558, 0.9133), 85: (+0.2357, 0.8682),
}


def a_realistic_face(
    midline: float = 175.0, brow_y: float = 120.0, span: float = 130.0
) -> np.ndarray:
    """86 points whose nose and mouth follow the MEASURED trace structure.

    The groups outside 60-85 keep ``a_face``'s scatter -- only the nose and mouth
    layouts were measured, and inventing the rest would be fixture fiction.
    ``span`` is brow-to-lower-lip; crop width is the cohort median AR times it.
    """
    points = a_face(midline=midline)
    width = 0.7404 * span
    for index, (x, y) in MEASURED_TRACE.items():
        points[index] = (midline + x * width, brow_y + y * span)

    # The brows must sit at brow_y and the mouth must be the lowest group, or
    # vertical_span would not reproduce the units the table is expressed in.
    brows = list(landmarks.LANDMARK_GROUPS["brows"])
    points[brows, 1] = brow_y + np.linspace(0.0, 0.04 * span, len(brows))
    for index in placement.EYE_LEFT:
        points[index] = (midline - 0.30 * width, brow_y + 0.23 * span)
    for index in placement.EYE_RIGHT:
        points[index] = (midline + 0.30 * width, brow_y + 0.23 * span)
    return points


# --------------------------------------------------------------------------
# the midline comes from the landmarks, never the image centre
# --------------------------------------------------------------------------


def test_the_midline_follows_the_face_not_the_frame():
    """**[MEASURED] Why this matters:** across 611 SCUT faces the landmark
    midline sits a mean 8.9px from the image centre (SD 10.7, max 57) against a
    crop only ~95px wide. Centring on the frame would put a horizontal offset
    into the pretraining data that the cleft data does not have, and the model
    would learn it as anatomy.
    """
    centred = placement.midline_x(a_face(midline=175.0))
    shifted = placement.midline_x(a_face(midline=205.0))
    assert centred == pytest.approx(175.0, abs=6.0)
    assert shifted - centred == pytest.approx(30.0, abs=6.0)


def test_the_crop_box_is_centred_on_that_midline():
    points = a_face(midline=205.0)
    x, _, w, _ = placement.crop_box(points, 0.74)
    assert x + w / 2 == pytest.approx(placement.midline_x(points), abs=1e-9)


# --------------------------------------------------------------------------
# the midline lands on the FACIAL midline, on realistic trace geometry
# --------------------------------------------------------------------------


def test_the_midline_lands_on_the_facial_midline_not_the_nose_flank():
    """**THE regression test for the displaced midline [MEASURED 2026-07-30].**

    The previous definition averaged the eye midpoint with ``mean(x)`` of nose
    indices 60-65, on the stated grounds that they run down the bridge "close to
    the midline on a frontal face". They run down the image-LEFT FLANK: measured
    0.117-0.238 of crop width from the midline, never near it. The estimate came
    out **0.0915 crop widths to the image-left on every one of 300 real faces**
    -- about 10.6px on a 115px crop.

    Asserted on ``a_realistic_face``, which reproduces the measured trace, at a
    tolerance the old definition **cannot** pass: it lands at -0.09, the fix at
    under 0.01.
    """
    points = a_realistic_face(midline=175.0)
    report = placement.mirror_pair_asymmetry(points)
    width = report["crop_width_px"]

    assert abs(report["signed_offset_frac_width"]) < 0.01, (
        "the midline is displaced from the mirror pairs' own centre; the old "
        "definition sat at -0.0915"
    )
    assert placement.midline_x(points) == pytest.approx(175.0, abs=0.01 * width)


def old_midline_x(points: np.ndarray) -> float:
    """The definition in force until 2026-07-30: eye midpoint averaged with
    ``mean(x)`` of nose indices 60-65. Kept here, and only here, so the
    regression test above can be shown to have teeth."""
    nose = list(landmarks.LANDMARK_GROUPS["nose"])
    left, right = placement.eye_centres(points)
    eye_mid = (left[0] + right[0]) / 2.0
    return float((eye_mid + points[nose[: len(nose) // 2], 0].mean()) / 2.0)


def test_the_old_midline_definition_fails_on_realistic_trace_geometry():
    """**The counterfactual, so the regression test is shown to have teeth.** A
    regression test the regression would also have passed proves nothing."""
    points = a_realistic_face(midline=175.0)
    report = placement.mirror_pair_asymmetry(points, midline=old_midline_x(points))
    assert report["signed_offset_frac_width"] < -0.05, (
        "the old definition should sit ~0.084 crop widths toward image-left on "
        "measured trace geometry, matching the -0.0915 seen on 300 real faces"
    )


def test_the_symmetric_fixture_ENDORSES_the_old_definition():
    """**[MEASURED 2026-07-30] The fixture did not merely fail to catch the bug
    -- it preferred it, and would have rejected the fix.**

    On ``a_face`` the nose group alternates sides, so ``mean(x)`` of 60-65 lands
    on the midline by construction while indices 66 and 76 are ordinary scatter
    points carrying no midline meaning. So the broken definition scores +0.010
    there and the correct one +0.077: **the fixture inverts the ranking**.

    This is the sixth entry in PLAN R7's tally, and a different shape from the
    others -- not a check that passes on its own failure mode, but a fixture whose
    construction makes the wrong answer look right. Recorded as a test rather
    than a comment because the temptation to reach for the convenient fixture in
    the next midline change is exactly what this guards.
    """
    fixture = a_realistic_face(midline=175.0)
    symmetric = a_face(midline=175.0)

    on_realistic_old = placement.mirror_pair_asymmetry(
        fixture, midline=old_midline_x(fixture)
    )["signed_offset_frac_width"]
    on_realistic_new = placement.mirror_pair_asymmetry(fixture)[
        "signed_offset_frac_width"
    ]
    on_symmetric_old = placement.mirror_pair_asymmetry(
        symmetric, midline=old_midline_x(symmetric)
    )["signed_offset_frac_width"]
    on_symmetric_new = placement.mirror_pair_asymmetry(symmetric)[
        "signed_offset_frac_width"
    ]

    # On measured geometry the fix wins by a wide margin.
    assert abs(on_realistic_new) < abs(on_realistic_old) / 10.0
    # On the symmetric fixture the ranking is INVERTED -- the trap, asserted.
    assert abs(on_symmetric_old) < abs(on_symmetric_new), (
        "a_face is expected to favour the old definition; if this ever stops "
        "being true the fixture changed and this warning can be revisited"
    )


def test_the_midline_diagnostic_reports_the_signed_offset_not_just_its_size():
    """A consistent SIGN across faces is what distinguishes structural bias from
    scatter -- it is what identified the old definition as a defect."""
    report = placement.mirror_pair_asymmetry(a_realistic_face())
    assert set(report) >= {
        "midline_x", "pair_centre_x", "n_pairs", "crop_width_px",
        "mean_asymmetry_px", "mean_asymmetry_frac_width",
        "max_asymmetry_frac_width", "signed_offset_frac_width",
    }
    assert report["n_pairs"] == len(landmarks.MIRROR_PAIRS)


def test_a_deliberately_shifted_midline_is_reported_as_shifted():
    """The diagnostic must be able to fail: shift the estimate and the reported
    offset must follow it, in the right direction and magnitude."""
    points = a_realistic_face(midline=175.0)
    width = placement.mirror_pair_asymmetry(points)["crop_width_px"]
    shifted = placement.mirror_pair_asymmetry(points, midline=175.0 + 0.1 * width)
    assert shifted["signed_offset_frac_width"] == pytest.approx(0.1, abs=0.02)
    assert shifted["mean_asymmetry_frac_width"] > 0.1


# --------------------------------------------------------------------------
# the anatomical anchors
# --------------------------------------------------------------------------


def test_the_span_runs_brow_to_below_the_lower_lip():
    points = a_face()
    top, bottom = placement.vertical_span(points)
    brow_top = points[list(landmarks.LANDMARK_GROUPS["brows"]), 1].min()
    lip_bottom = points[list(landmarks.LANDMARK_GROUPS["mouth"]), 1].max()

    assert top == pytest.approx(brow_top)
    assert bottom > lip_bottom, "'just below' the lower vermillion, not level with it"


def test_the_lower_margin_scales_with_the_face():
    """A fixed pixel margin would be a different margin on every face."""
    small = placement.vertical_span(a_face(scale=1.0))
    large = placement.vertical_span(a_face(scale=2.0))
    small_overshoot = small[1] - a_face(scale=1.0)[73:86, 1].max()
    large_overshoot = large[1] - a_face(scale=2.0)[73:86, 1].max()
    assert large_overshoot > small_overshoot


def test_an_inverted_face_is_refused():
    points = a_face()
    points[list(landmarks.LANDMARK_GROUPS["brows"]), 1] += 200
    with pytest.raises(placement.PlacementError, match="not above the lower lip"):
        placement.vertical_span(points)


def test_the_width_follows_the_sampled_aspect_ratio():
    """H comes from anatomy; W = AR x H, so masked SCUT varies in framing the
    way the clinical set does rather than presenting one constant shape."""
    points = a_face()
    _, _, narrow, height = placement.crop_box(points, 0.553)
    _, _, wide, _ = placement.crop_box(points, 1.099)
    assert narrow / height == pytest.approx(0.553)
    assert wide / height == pytest.approx(1.099)
    assert wide > narrow


def test_a_non_positive_aspect_ratio_is_refused():
    with pytest.raises(placement.PlacementError, match="must be positive"):
        placement.crop_box(a_face(), 0.0)


def test_the_half_widths_come_from_the_frozen_cleft_definition():
    """Restating them here would be a second place for the mask to live."""
    from cleft.geometry import trapezium

    assert placement.trapezium_half_widths() == {
        "top": trapezium.TOP_HALF_WIDTH,
        "bottom": trapezium.BOT_HALF_WIDTH,
    }


# --------------------------------------------------------------------------
# tilt, measured and declared rather than silently corrected
# --------------------------------------------------------------------------


def test_the_intercanthal_angle_recovers_a_known_tilt():
    assert placement.intercanthal_angle(a_face(tilt_deg=0.0)) == pytest.approx(0.0, abs=2.0)
    assert placement.intercanthal_angle(a_face(tilt_deg=8.0)) == pytest.approx(8.0, abs=3.0)


def test_the_measured_tilt_distribution_is_recorded():
    """[MEASURED 2026-07-28] 917 faces: SD 3.1 degrees, 37% beyond 2, 10% beyond
    5. Not negligible, so it is a declared choice rather than an oversight."""
    tilt = placement.SCUT_HEAD_TILT
    assert tilt["n_sampled"] == 917
    assert tilt["sd_deg"] == pytest.approx(3.099)
    assert tilt["fraction_beyond_2deg"] == pytest.approx(0.374)
    assert tilt["fraction_beyond_5deg"] == pytest.approx(0.099)


def test_levelling_is_off_because_the_cleft_crops_are_not_levelled():
    """[OBSERVED 2026-07-29, Phase 2 cleft contact sheet] Neither domain is
    levelled, so levelling SCUT would CREATE the gap it was meant to close. The
    SCUT tilt spread is a matched property, not a defect."""
    tilt = placement.SCUT_HEAD_TILT
    assert tilt["cleft_levelling_state"] == "NOT_LEVELLED"
    assert "contact sheet" in tilt["cleft_levelling_evidence"]
    assert placement.SCUT_HEAD_TILT_LEVELLING == "off"
    assert "matched to the cleft domain" in tilt["levelling"]


def test_the_apparent_cleft_tilt_is_recorded_as_probable_nasal_deviation():
    """**Worst at the nose tips, less at the eyes, least at the lips.** Head roll
    rotates every structure equally, so that pattern is not roll -- it is septum
    and tip displaced off the facial midline, a hallmark of unilateral cleft and
    one of the four things Asher-McDade scores.

    Which strengthens leaving both domains unlevelled twice over: correcting it
    would suppress something present in only one domain, and something the label
    is partly about.
    """
    finding = placement.SCUT_HEAD_TILT["apparent_cleft_tilt_is"]
    assert "nasal deviation" in finding
    assert "not head roll" in finding
    assert "rotates every structure equally" in finding


def test_the_lower_margin_is_recorded_as_reviewed():
    """Confirmed on the placement sheet -- lower vermillion visible, no chin."""
    assert placement.LOWER_MARGIN == 0.06


def test_the_eye_groups_are_disjoint_and_cover_the_eye_indices():
    """[MEASURED] Derived by splitting the eye group on x over 917 faces, stable
    on every one, no ambiguous index."""
    left, right = set(placement.EYE_LEFT), set(placement.EYE_RIGHT)
    assert not left & right
    assert left | right == set(landmarks.LANDMARK_GROUPS["eyes"])


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


def test_a_box_inside_the_frame_is_reported_as_such():
    points = a_face()
    assert placement.box_is_inside(placement.crop_box(points, 0.74), (350, 350))


def test_a_box_running_off_the_edge_is_reported_not_padded():
    """Padding inside the content box is something no cleft crop has, so the
    rate is measured and the sheet judges it."""
    points = a_face(midline=20.0)
    assert not placement.box_is_inside(placement.crop_box(points, 1.099), (350, 350))


def test_describe_carries_the_placement_and_its_inputs():
    described = placement.describe(a_face(), 0.74)
    assert set(described) >= {
        "box", "aspect_ratio", "midline_x", "intercanthal_angle_deg",
        "brow_y", "lower_crop_y", "half_widths",
    }
