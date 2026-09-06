"""The segmentation feasibility gate (Phase 4 §2).

The gate itself runs on the cluster against real crops. What is testable here is
that the instrument works: that the discriminant separates something known to be
separable, that the plausibility criteria reject what they should, and that a
"no" comes out as a "no" rather than as an error.

Synthetic fixtures throughout -- a "face" is a skin-coloured rectangle with a
redder patch where lips would be. That is enough to prove the machinery
distinguishes a case it should from one it should not, which is all a laptop can
prove about a question that is fundamentally about real JPEG artefacts.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cleft.geometry import segmentation as S
from cleft.run import main

from fixtures import builders

SIZE = 224
SKIN = (222, 184, 156)
LIP = (190, 70, 80)


#: Staging pads with 255, so the frame a real crop arrives in is part-white.
#: Fixtures carry that padding because run 1's central failure was caused by it.
PAD = (255, 255, 255)


def synthetic_face(
    lip_colour=LIP,
    lip_y=0.78,
    lip_half_height=0.05,
    lip_half_width=0.18,
    pad: float = 0.12,
    nose: bool = False,
) -> np.ndarray:
    """Skin on white padding, with a redder band low and central where lips are.

    ``nose=True`` adds a second red region in the *middle* band -- the measured
    failure mode: the discriminant is right that both are red, and only position
    tells them apart.
    """
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[:, :] = PAD
    edge = int(pad * SIZE)
    image[edge : SIZE - edge, edge : SIZE - edge] = SKIN

    if nose:
        image[int(0.44 * SIZE) : int(0.56 * SIZE), int(0.42 * SIZE) : int(0.58 * SIZE)] = (
            lip_colour
        )

    y0 = int((lip_y - lip_half_height) * SIZE)
    y1 = int((lip_y + lip_half_height) * SIZE)
    x0 = int((0.5 - lip_half_width) * SIZE)
    x1 = int((0.5 + lip_half_width) * SIZE)
    image[y0:y1, x0:x1] = lip_colour
    return image


def two_lipped_face(
    fissure_darkness: int = 40,
    upper=(190, 70, 80),
    lower=(185, 68, 78),
    pad: float = 0.12,
) -> np.ndarray:
    """Upper lip, oral fissure, lower lip -- the run-3 failure mode.

    The two lips are deliberately near-identical in colour, because that is the
    actual mechanism: Chan-Vese segments regions of similar intensity and the two
    lips ARE similar, so nothing but the thin dark fissure separates them.
    ``fissure_darkness`` sets how pronounced that line is -- which is exactly
    what decided whether run 2 crossed on a given patient.
    """
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[:, :] = PAD
    edge = int(pad * SIZE)
    image[edge : SIZE - edge, edge : SIZE - edge] = SKIN

    x0, x1 = int(0.32 * SIZE), int(0.68 * SIZE)
    image[int(0.72 * SIZE) : int(0.78 * SIZE), x0:x1] = upper
    image[int(0.78 * SIZE) : int(0.80 * SIZE), x0:x1] = (
        fissure_darkness,
        fissure_darkness // 2,
        fissure_darkness // 2,
    )
    image[int(0.80 * SIZE) : int(0.86 * SIZE), x0:x1] = lower
    return image


def dark_skinned_face(pad: float = 0.12) -> np.ndarray:
    """The run-3 fairness case: lip-versus-skin contrast compressed.

    Not a claim about any real patient -- a fixture whose contrast is compressed
    in the same channels the measured failure was, so the screen and the
    normalisation hypothesis can be exercised on the laptop.
    """
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[:, :] = PAD
    edge = int(pad * SIZE)
    image[edge : SIZE - edge, edge : SIZE - edge] = (105, 62, 48)
    image[int(0.73 * SIZE) : int(0.83 * SIZE), int(0.32 * SIZE) : int(0.68 * SIZE)] = (
        118,
        58,
        50,
    )
    return image


def featureless_face(pad: float = 0.12) -> np.ndarray:
    """Skin on padding, nothing else. The methods must say so."""
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[:, :] = PAD
    edge = int(pad * SIZE)
    image[edge : SIZE - edge, edge : SIZE - edge] = SKIN
    return image


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------


def test_hsv_matches_known_values():
    """Checked against the standard conversion, not against itself."""
    pixels = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255], [0, 0, 0]]],
        dtype=np.uint8,
    )
    hsv = S.to_hsv(pixels)[0]
    assert hsv[0] == pytest.approx([0.0, 1.0, 1.0], abs=1e-4)       # red
    assert hsv[1] == pytest.approx([1 / 3, 1.0, 1.0], abs=1e-4)     # green
    assert hsv[2] == pytest.approx([2 / 3, 1.0, 1.0], abs=1e-4)     # blue
    assert hsv[3] == pytest.approx([0.0, 0.0, 1.0], abs=1e-4)       # white
    assert hsv[4] == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)       # black


def test_a_greyscale_image_is_refused():
    with pytest.raises(S.SegmentationError, match="expected an RGB image"):
        S.to_hsv(np.zeros((8, 8), dtype=np.uint8))


def test_the_discriminant_scores_lips_above_skin():
    discriminant = S.lip_discriminant(synthetic_face())
    lip_row, skin_row = int(0.78 * SIZE), int(0.2 * SIZE)
    assert discriminant[lip_row, SIZE // 2] > discriminant[skin_row, SIZE // 2]


def test_the_discriminant_is_bounded():
    values = S.lip_discriminant(synthetic_face())
    assert values.min() >= 0.0 and values.max() <= 1.0


# --------------------------------------------------------------------------
# otsu and separability
# --------------------------------------------------------------------------


def test_otsu_separates_two_clean_modes():
    values = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    threshold, eta = S.otsu(values)
    assert 0.1 < threshold < 0.9
    assert eta > 0.95, "two clean modes must score near 1"


def test_otsu_reports_no_separability_for_one_blob():
    _, eta = S.otsu(np.random.default_rng(0).normal(0.5, 0.01, size=1000))
    assert eta < 0.9


def test_a_constant_image_has_zero_separability():
    region = np.ones((SIZE, SIZE), dtype=bool)
    assert S.separability_of(featureless_face(pad=0.0), region) == 0.0


def test_a_face_with_lips_separates_better_than_one_without():
    """The core claim of the diagnostic, on a case where the answer is known."""
    lips, plain = synthetic_face(), featureless_face()
    assert S.separability_of(lips, S.face_mask(lips)) > S.separability_of(
        plain, S.face_mask(plain)
    )


def test_otsu_refuses_an_empty_array():
    with pytest.raises(S.SegmentationError, match="empty"):
        S.otsu(np.array([]))


# --------------------------------------------------------------------------
# THE RUN-1 CORRECTION: separability was measuring the wrong contrast
# --------------------------------------------------------------------------


def test_whole_frame_separability_measures_the_padding_not_the_lips():
    """[MEASURED 2026-07-28] Run 1 scored 0.888 over the frame and it meant
    nothing about lips.

    A face with NO lips at all still separates strongly over the whole frame,
    because skin against white padding is the dominant contrast. That is the
    entire defect, reproduced.
    """
    plain = featureless_face()
    assert S.separability_of(plain) > 0.8, "skin against pad looks strongly bimodal"
    assert S.separability_of(plain, S.face_mask(plain)) < S.separability_of(plain), (
        "inside the face there is nothing to separate, and it must say so"
    )


def test_in_region_separability_distinguishes_what_whole_frame_cannot():
    """The corrected number does the job the old one appeared to."""
    lips, plain = synthetic_face(), featureless_face()

    whole_frame_gap = abs(S.separability_of(lips) - S.separability_of(plain))
    in_region_gap = abs(
        S.separability_of(lips, S.face_mask(lips))
        - S.separability_of(plain, S.face_mask(plain))
    )
    assert in_region_gap > whole_frame_gap


# --------------------------------------------------------------------------
# the face mask and the spatial prior
# --------------------------------------------------------------------------


def test_the_face_mask_excludes_the_white_padding():
    mask = S.face_mask(featureless_face(pad=0.12))
    assert not mask[2, 2], "the corner is padding"
    assert mask[SIZE // 2, SIZE // 2], "the centre is face"
    assert 0.4 < mask.mean() < 0.9


def test_the_face_mask_survives_an_unsaturated_highlight():
    """A specular highlight is not background, and a hole in the region would
    let a method find an edge that is not anatomy."""
    image = featureless_face()
    image[100:110, 100:110] = (252, 252, 252)
    assert S.face_mask(image)[105, 105]


def test_an_all_padding_image_yields_an_empty_face():
    image = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    assert not S.face_mask(image).any()


def test_the_bottom_prior_matches_the_phase_2_band():
    """Bands come from PatchConfig, so the two instruments cannot drift apart."""
    from cleft.geometry.patches import PatchConfig

    bottom = [b for b in PatchConfig().bands if b.name == "bottom"][0]
    mask = S.prior_mask((SIZE, SIZE), "bottom")
    assert not mask[int(bottom.y0 * SIZE) - 5].any()
    assert mask[int(bottom.y0 * SIZE) + 5].all()


def test_no_prior_selects_everything():
    assert S.prior_mask((SIZE, SIZE), "none").all()


def test_an_unknown_prior_is_rejected():
    with pytest.raises(S.SegmentationError, match="unknown spatial_prior"):
        S.prior_mask((SIZE, SIZE), "eyebrows")


def test_the_analysis_region_is_face_and_prior_together():
    image = synthetic_face()
    region = S.analysis_region(image, "bottom")
    assert not region[2, 2], "padding is excluded even inside the band"
    assert not region[int(0.2 * SIZE), SIZE // 2], "the top band is excluded"
    assert region[int(0.78 * SIZE), SIZE // 2], "the lip row survives both"


# --------------------------------------------------------------------------
# the three methods
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", S.METHODS)
def test_every_method_returns_a_boolean_mask(method):
    image = synthetic_face()
    mask = S.segment(image, method, S.analysis_region(image, "bottom"))
    assert mask.dtype == bool
    assert mask.shape == (SIZE, SIZE)


@pytest.mark.parametrize("method", S.METHODS)
def test_every_method_finds_the_planted_lips(method):
    """Not a claim about real data -- a claim that the instrument is not broken."""
    image = synthetic_face()
    mask = S.segment(image, method, S.analysis_region(image, "bottom"))
    description = S.describe_mask(mask)
    assert description["centroid_y"] > 0.5, f"{method} found something too high"
    assert description["area"] > 0.0


@pytest.mark.parametrize("method", S.METHODS)
def test_no_method_escapes_its_region(method):
    """The prior is a hard constraint, not a hint."""
    image = synthetic_face(nose=True)
    region = S.analysis_region(image, "bottom")
    mask = S.segment(image, method, region)
    assert not (mask & ~region).any()


@pytest.mark.parametrize("method", S.METHODS)
def test_no_method_crashes_on_a_featureless_image(method):
    """A "no" must come back as a verdict, not as a traceback."""
    image = featureless_face()
    mask = S.segment(image, method, S.analysis_region(image, "bottom"))
    assert mask.shape == (SIZE, SIZE)


@pytest.mark.parametrize("method", S.METHODS)
def test_an_empty_region_yields_an_empty_mask(method):
    empty = np.zeros((SIZE, SIZE), dtype=bool)
    assert not S.segment(synthetic_face(), method, empty).any()


def test_an_unknown_method_is_rejected():
    with pytest.raises(S.SegmentationError, match="unknown method"):
        S.segment(synthetic_face(), "watershed")


def test_chan_vese_terminates_on_a_uniform_image():
    """The loop must not spin when there is nothing to separate."""
    image = featureless_face()
    assert S.chan_vese(
        image, S.analysis_region(image, "bottom"), iterations=1000
    ).shape == (SIZE, SIZE)


# --------------------------------------------------------------------------
# THE RUN-1 CORRECTION: the prior separates lips from nose
# --------------------------------------------------------------------------


def test_without_a_prior_the_nose_comes_out_with_the_lips():
    """[MEASURED 2026-07-28] Every patient the discriminant worked on gave lips
    AND nose. The discriminant is not wrong -- both really are red."""
    image = synthetic_face(nose=True)
    mask = S.segment(image, "otsu_threshold", S.analysis_region(image, "none"))
    assert mask[int(0.50 * SIZE), SIZE // 2], "the nose is selected"
    assert mask[int(0.78 * SIZE), SIZE // 2], "and so are the lips"


def test_the_bottom_prior_drops_the_nose_and_keeps_the_lips():
    """Position separates what no threshold could."""
    image = synthetic_face(nose=True)
    mask = S.segment(image, "otsu_threshold", S.analysis_region(image, "bottom"))
    assert not mask[int(0.50 * SIZE), SIZE // 2], "the nose is gone"
    assert mask[int(0.78 * SIZE), SIZE // 2], "the lips remain"


def test_the_adaptive_cut_survives_a_skin_tone_the_fixed_cut_does_not():
    """[MEASURED 2026-07-28] 3 of 9 collapsed to the whole face because the
    overall tone sat near the fixed 0.35 and everything passed it."""
    ruddy = synthetic_face()
    skin = np.array(S.SKIN if hasattr(S, "SKIN") else SKIN, dtype=np.uint8)
    # A skin tone whose discriminant sits above the fixed cut everywhere.
    ruddy[ruddy[:, :, 0] == skin[0]] = (215, 120, 110)

    region = S.analysis_region(ruddy, "bottom")
    fixed = S.describe_mask(S.colour_separation(ruddy, region), region)
    adaptive = S.describe_mask(S.otsu_threshold(ruddy, region), region)

    assert fixed["saturates_region"], "the fixed cut takes everything"
    assert not adaptive["saturates_region"], "the per-image cut does not"


# --------------------------------------------------------------------------
# plausibility -- the criteria must reject, or they are decoration
# --------------------------------------------------------------------------


def test_a_plausible_region_passes():
    plausible, reasons = S.is_plausible(
        {"area": 0.08, "centroid_x": 0.5, "centroid_y": 0.78, "largest_component": 0.95}
    )
    assert plausible and reasons == []


def test_a_region_too_high_on_the_face_is_rejected():
    """The failure mode that looks most convincing on a thumbnail: the method
    found the eyes or the nose and produced a tidy blob doing it."""
    plausible, reasons = S.is_plausible(
        {"area": 0.08, "centroid_x": 0.5, "centroid_y": 0.2, "largest_component": 0.95}
    )
    assert not plausible
    assert any("too high on the face" in r for r in reasons)


def test_a_fragmented_region_is_rejected():
    plausible, reasons = S.is_plausible(
        {"area": 0.08, "centroid_x": 0.5, "centroid_y": 0.78, "largest_component": 0.1}
    )
    assert not plausible
    assert any("fragmented" in r for r in reasons)


def test_a_region_covering_the_whole_face_is_rejected():
    plausible, _ = S.is_plausible(
        {"area": 0.9, "centroid_x": 0.5, "centroid_y": 0.78, "largest_component": 1.0}
    )
    assert not plausible


def test_a_speckle_is_rejected():
    plausible, _ = S.is_plausible(
        {"area": 0.001, "centroid_x": 0.5, "centroid_y": 0.78, "largest_component": 1.0}
    )
    assert not plausible


# --------------------------------------------------------------------------
# RUN 2: SymNose is IN, and the instrument is the contour
# --------------------------------------------------------------------------


def test_the_instrument_is_the_contour_not_the_threshold():
    """[DECIDED 2026-07-28] SymNose mirrors a traced boundary and superimposes
    it, so boundary noise enters the asymmetry measurement AS asymmetry. Otsu
    thresholds per pixel and its edge is pixelated; Chan-Vese carries a curvature
    term and its edge is smooth. They find the same region -- which is why Otsu
    stays as a comparator -- but a jagged edge would manufacture the signal being
    measured."""
    assert S.UNCONSTRAINED == "chan_vese_relaxed"
    assert S.COMPARATOR == "otsu_threshold"
    # Run 3: the instrument is that contour with the upper-lip constraint on it.
    assert S.INSTRUMENT == "upper_lip"
    assert S.INSTRUMENT in S.METHODS and S.COMPARATOR in S.METHODS


def test_the_unrelaxed_contour_is_still_run():
    """Rendered beside the relaxed mask so the relaxation's magnitude is visible
    on the sheet rather than being a number nobody can see the size of."""
    assert "chan_vese" in S.METHODS


def test_colour_separation_is_kept_as_a_record_not_a_candidate():
    """Unchanged from run 1 apart from being confined to the band, and out."""
    assert "colour_separation" in S.METHODS
    assert S.INSTRUMENT != "colour_separation"


def test_the_relaxed_contour_contains_the_unrelaxed_one():
    """Relaxation moves the boundary OUTWARD. A relaxation that lost pixels
    would be doing something other than what it says."""
    image = synthetic_face()
    region = S.analysis_region(image, "bottom")
    tight = S.chan_vese(image, region, relaxation=0.0)
    loose = S.chan_vese(image, region, relaxation=0.3)
    assert loose.sum() >= tight.sum()


def test_zero_relaxation_is_the_unrelaxed_contour():
    image = synthetic_face()
    region = S.analysis_region(image, "bottom")
    assert np.array_equal(
        S.chan_vese(image, region, relaxation=0.0),
        S.segment(image, "chan_vese", region),
    )


def test_the_relaxation_is_a_fraction_of_each_images_own_class_gap():
    """One rule, one parameter, applied identically -- NOT a per-patient tune.

    A fixed pixel dilation would be a different relaxation on every patient,
    because contrast and crop scale vary. Expressed against the class gap it
    lands in the same place relative to each image's own separation, which is
    what makes it fixed rather than tuned.
    """
    faint = synthetic_face(lip_colour=(205, 150, 140))
    stark = synthetic_face(lip_colour=(190, 40, 50))

    for image in (faint, stark):
        region = S.analysis_region(image, "bottom")
        tight = S.chan_vese(image, region, relaxation=0.0).sum()
        loose = S.chan_vese(image, region, relaxation=S.CONTOUR_RELAXATION).sum()
        assert loose >= tight


def test_the_relaxation_default_is_small():
    """SymNose measures the UPPER lip. Over-relaxing crosses the oral fissure
    into the lower lip and silently changes what is measured."""
    assert 0.0 < S.CONTOUR_RELAXATION <= 0.25


# --------------------------------------------------------------------------
# RUN 3 FINDING 1: the contour crosses the oral fissure
# --------------------------------------------------------------------------


def test_the_fissure_is_found_between_the_two_lips():
    image = two_lipped_face()
    region = S.analysis_region(image, "bottom")
    contour = S.chan_vese(image, region, relaxation=S.CONTOUR_RELAXATION)
    row = S.fissure_row(image, contour)

    assert row is not None
    assert int(0.77 * SIZE) <= row <= int(0.82 * SIZE), (
        "the fissure must land on the dark line, not on a mask edge"
    )


def test_the_fissure_search_ignores_the_mask_edges():
    """Without the interior margin the darkest row is often the mask's own
    bottom edge, where the contour is thinning into shadow."""
    image = two_lipped_face()
    region = S.analysis_region(image, "bottom")
    contour = S.chan_vese(image, region, relaxation=S.CONTOUR_RELAXATION)
    rows = np.nonzero(contour.any(axis=1))[0]
    row = S.fissure_row(image, contour)

    height = rows.max() - rows.min()
    assert row > rows.min() + 0.2 * height
    assert row < rows.max() - 0.2 * height


def test_a_single_lip_has_no_fissure():
    """The detector must not fire on a correct upper-lip-only mask.

    [DEFECT, found by test 2026-07-28] The first version fired on everything and
    cut every mask in half -- which would have looked like SUCCESS on the screen,
    because a halved mask satisfies fraction_above_split by construction. Two
    causes, both real: the row profile was smoothed with ``mode="same"``, which
    zero-pads, so the end rows came out at two thirds of their true value and the
    darkest row was always an end of the profile; and there was no requirement
    that the dip be deep enough to be a dark LINE rather than the darkest row of
    something uniform.
    """
    image = synthetic_face()
    region = S.analysis_region(image, "bottom")
    contour = S.chan_vese(image, region, relaxation=S.CONTOUR_RELAXATION)
    assert contour.any(), "the fixture must produce a mask for this to mean anything"
    assert S.fissure_row(image, contour) is None


def test_the_profile_smoothing_does_not_manufacture_a_dip():
    """The zero-padding defect, isolated: a flat profile must stay flat."""
    flat = np.full(9, 0.75)
    padded = np.convolve(np.pad(flat, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    assert padded == pytest.approx(flat)

    zero_padded = np.convolve(flat, np.ones(3) / 3.0, mode="same")
    assert zero_padded[0] < flat[0] * 0.7, "which is what the defect looked like"


def test_a_dip_too_shallow_to_be_a_dark_line_is_not_a_fissure():
    """Honest, and it has a consequence: nothing is constrained on that patient,
    so the mask may still span both lips while passing the criterion trivially.
    n_fissure_found is reported against n_patients so those are visible."""
    image = two_lipped_face(fissure_darkness=190)
    region = S.analysis_region(image, "bottom")
    contour = S.chan_vese(image, region, relaxation=S.CONTOUR_RELAXATION)
    assert S.fissure_row(image, contour) is None


def test_the_verdict_reports_how_many_splits_were_found():
    """The geometric split always fires, so every patient has one -- which is
    the trade against the fissure rule, not an improvement in every direction."""
    summary = probe([two_lipped_face(), two_lipped_face(fissure_darkness=190)])
    assert summary["fissure"]["n_split_found"] == 2
    assert summary["fissure"]["n_patients"] == 2
    assert summary["fissure"]["split_rule"] == S.DEFAULT_SPLIT_RULE


def test_the_scan_reports_the_measurement_whether_or_not_it_fires():
    """[RUN 4] found/not-found cannot say whether a threshold is too strict.

    2 of 9 fired at 0.08 and six masks still spanned both lips. The depth is the
    number that turns "the threshold is too strict" from an inference into a
    distribution -- and that distribution decides whether lowering it recovers
    the six or admits noise.
    """
    image = two_lipped_face(fissure_darkness=190)
    region = S.analysis_region(image, "bottom")
    contour = S.chan_vese(image, region, relaxation=S.CONTOUR_RELAXATION)

    scan = S.fissure_scan(image, contour)
    assert S.fissure_row(image, contour) is None, "below the threshold"
    assert scan["row"] is not None, "but the candidate and its depth are reported"
    assert scan["depth"] > 0.0
    assert 0.0 <= scan["row_relative"] <= 1.0


def test_sharpness_separates_a_faint_line_from_a_flat_profile():
    """Depth alone cannot: at 0.03 a shallow dark line and the darkest row of
    something uniform look identical. Sharpness is what tells them apart, and it
    is what makes lowering the depth threshold safe."""
    faint = two_lipped_face(fissure_darkness=150)
    region = S.analysis_region(faint, "bottom")
    line = S.fissure_scan(faint, S.chan_vese(faint, region, relaxation=S.CONTOUR_RELAXATION))

    flat_image = synthetic_face()
    flat_region = S.analysis_region(flat_image, "bottom")
    flat = S.fissure_scan(
        flat_image, S.chan_vese(flat_image, flat_region, relaxation=S.CONTOUR_RELAXATION)
    )

    assert line["sharpness"] > S.FISSURE_MIN_SHARPNESS
    assert flat["sharpness"] < S.FISSURE_MIN_SHARPNESS


@pytest.mark.parametrize("threshold", S.FISSURE_THRESHOLD_SWEEP)
def test_no_swept_threshold_fires_on_a_single_lip(threshold):
    """**The regression stays in the loop across the whole sweep.**

    A threshold low enough to fire on 9 of 9 while landing on arbitrary dark rows
    is worse than one that fires on 2: a false fissure halves a correct mask, and
    the halved mask then satisfies fraction_above_split by construction. That
    is the same false positive the smoothing defect produced, so every candidate
    threshold has to be checked against a mask that has no fissure.
    """
    image = synthetic_face()
    region = S.analysis_region(image, "bottom")
    contour = S.chan_vese(image, region, relaxation=S.CONTOUR_RELAXATION)
    assert S.fissure_row(image, contour, min_depth=threshold) is None


def test_the_sweep_reports_what_each_threshold_would_have_found():
    scans = [
        {"row": 10, "depth": 0.02, "sharpness": 2.0},
        {"row": 10, "depth": 0.04, "sharpness": 2.0},
        {"row": 10, "depth": 0.09, "sharpness": 2.0},
        # Deep enough for every threshold, but indistinguishable from noise.
        {"row": 10, "depth": 0.30, "sharpness": 0.4},
    ]
    table = S.sweep_fissure_thresholds(scans)

    assert table["0.02"]["n_found"] == 4
    assert table["0.08"]["n_found"] == 2
    # Detection rate is not the outcome: one of those four is noise at every
    # threshold, and counting it as a detection is how a sweep goes wrong.
    assert table["0.02"]["n_sharp"] == 3
    assert table["0.02"]["n_noise"] == 1


def test_the_sweep_covers_the_thresholds_asked_for():
    assert S.FISSURE_THRESHOLD_SWEEP == (0.02, 0.03, 0.05, 0.08)
    assert set(S.sweep_fissure_thresholds([])) == {
        str(t) for t in S.FISSURE_THRESHOLD_SWEEP
    }


def test_the_verdict_reports_the_depth_distribution_and_the_sweep():
    summary = probe(
        [two_lipped_face(), two_lipped_face(fissure_darkness=150), synthetic_face()]
    )
    fissure = summary["fissure"]

    assert len(fissure["depths_sorted"]) == 3
    assert fissure["depths_sorted"] == sorted(fissure["depths_sorted"])
    assert fissure["threshold_in_use"] == S.FISSURE_MIN_DEPTH
    assert set(fissure["sweep"]) == {str(t) for t in S.FISSURE_THRESHOLD_SWEEP}
    # The sweep is kept as the EVIDENCE for abandoning the fissure route, not as
    # a knob still to be turned.
    assert "THE FISSURE ROUTE IS ABANDONED" in fissure["note"]
    assert "0.047 and 0.114" in fissure["note"]


def test_the_verdict_names_the_floor():
    """If the six undetected patients sit at the noise floor, no threshold
    recovers them and that is a limitation of the method on this data."""
    summary = probe([synthetic_face() for _ in range(3)])
    floor = summary["fissure"]["floor"]

    assert floor["n_below_noise"] == 3
    assert floor["sharpness_criterion"] == S.FISSURE_MIN_SHARPNESS
    assert "NO threshold recovers it" in floor["note"]
    assert "not a defect to fix" in floor["note"]
    assert "fairness" in floor["note"]


def test_a_mask_too_small_to_have_a_fissure_returns_none():
    """The question was not asked rather than answered yes."""
    tiny = np.zeros((SIZE, SIZE), dtype=bool)
    tiny[100:103, 100:110] = True
    assert S.fissure_row(two_lipped_face(), tiny) is None
    assert S.fissure_row(two_lipped_face(), np.zeros((SIZE, SIZE), bool)) is None


def test_above_fissure_keeps_only_the_upper_lip():
    image = two_lipped_face()
    region = S.analysis_region(image, "bottom")
    contour = S.chan_vese(image, region, relaxation=S.CONTOUR_RELAXATION)
    row = S.fissure_row(image, contour)
    upper = S.above_fissure(contour, row)

    assert upper.sum() < contour.sum(), "something was removed"
    assert not upper[row:].any(), "nothing survives below the fissure"
    assert S.fraction_above_split(upper, row) == 1.0


def test_no_fissure_means_nothing_is_constrained():
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[100:140, 90:150] = True
    assert np.array_equal(S.above_fissure(mask, None), mask)
    assert S.fraction_above_split(mask, None) == 1.0


# --- the regression: the criterion must FAIL what it was written for ---------


def crossing_fraction(image, relaxation) -> float:
    region = S.analysis_region(image, "bottom")
    contour = S.chan_vese(image, region, relaxation=relaxation)
    return S.fraction_above_split(contour, S.fissure_row(image, contour))


def test_the_relaxed_contour_crosses_the_fissure():
    """[MEASURED 2026-07-28] Run 2 relaxed crossed on 9 of 9. The new criterion
    must reject that, or it is a formality rather than a check."""
    assert crossing_fraction(two_lipped_face(), S.CONTOUR_RELAXATION) < (
        S.PLAUSIBLE["fraction_above_split_min"]
    )


def test_a_faint_fissure_is_crossed_even_unrelaxed():
    """[MEASURED 2026-07-28] Run 2 crossed on 3 of 9 at relaxation 0 -- so the
    relaxation made a pre-existing failure universal rather than creating it,
    and REDUCING IT RETURNS TO 3 OF 9, NOT TO 0.

    Faint fissure: the two lips are near-identical and the dark line barely
    separates them, which is the patient-to-patient variation that decided it.
    """
    assert crossing_fraction(two_lipped_face(fissure_darkness=150), 0.0) < (
        S.PLAUSIBLE["fraction_above_split_min"]
    )


def test_the_constraint_is_the_fix_not_a_smaller_relaxation():
    """The whole argument of finding 1, as an assertion.

    Reducing the relaxation does not fix a faint-fissure image; applying the
    anatomical constraint does. Intensity does not encode an anatomical
    boundary, so an intensity-based method needs to be told where one is.
    """
    faint = two_lipped_face(fissure_darkness=150)
    limit = S.PLAUSIBLE["fraction_above_split_min"]

    assert crossing_fraction(faint, 0.0) < limit, "smaller relaxation still crosses"

    region = S.analysis_region(faint, "bottom")
    constrained = S.segment(faint, "upper_lip", region, relaxation=0.0)
    contour = S.chan_vese(faint, region, relaxation=0.0)
    assert S.fraction_above_split(
        constrained, S.fissure_row(faint, contour)
    ) >= limit


def test_the_instrument_satisfies_the_criterion_by_construction():
    """Stated rather than treated as evidence -- exactly like centroid_y under
    the bottom-band prior. The informative number is the unconstrained one."""
    image = two_lipped_face()
    results = S.run_methods(image, "bottom")
    by_method = {r.method: r for r in results}

    assert by_method[S.INSTRUMENT].description["fraction_above_split"] == 1.0
    assert (
        by_method[S.UNCONSTRAINED].description["fraction_above_split"]
        < S.PLAUSIBLE["fraction_above_split_min"]
    )


def test_the_fissure_is_located_once_from_the_unconstrained_contour():
    """Per-method location would let a method that found only the upper lip
    declare its own top edge the fissure and pass trivially."""
    image = two_lipped_face()
    rows = {
        result.description["fissure_row"] for result in S.run_methods(image, "bottom")
    }
    assert len(rows) == 1


def test_a_crossing_mask_is_rejected_with_the_reason():
    image = two_lipped_face()
    results = S.run_methods(image, "bottom")
    unconstrained = {r.method: r for r in results}[S.UNCONSTRAINED]

    assert not unconstrained.plausible
    assert any("UPPER lip" in reason for reason in unconstrained.reasons)


def test_the_verdict_counts_the_crossings():
    summary = probe([two_lipped_face() for _ in range(4)])
    assert summary["fissure"]["n_crossing"] == 4
    assert summary["fissure"]["measured_on"] == S.UNCONSTRAINED
    assert "3 OF 9, NOT 0" in summary["fissure"]["note"]


# --------------------------------------------------------------------------
# RUN 3 FINDINGS 2 AND 3: the fairness limitation
# --------------------------------------------------------------------------


def test_the_normalisations_are_offered_and_default_off():
    """Default off so the fissure constraint and the fairness hypothesis are
    separate runs -- R6, one change at a time."""
    assert S.DEFAULT_NORMALISATION == "none"
    assert set(S.DISCRIMINANT_NORMALISATIONS) == {"none", "rank", "zscore"}


def test_rank_normalisation_spreads_a_compressed_discriminant():
    """The mechanism the hypothesis rests on: on darker skin the lip-versus-skin
    contrast compresses, and a percentile map restores full range whatever the
    original compression was."""
    image = dark_skinned_face()
    region = S.analysis_region(image, "bottom")

    raw = S.discriminant_for(image, region, "none")[region]
    ranked = S.discriminant_for(image, region, "rank")[region]

    assert raw.max() - raw.min() < 1.0
    assert ranked.max() == pytest.approx(1.0)
    assert ranked.min() == pytest.approx(0.0)


def test_normalisation_uses_only_pixels_inside_the_region():
    """Otherwise the white padding would set the scale, which is the run-1
    defect in a new place."""
    image = dark_skinned_face()
    region = S.analysis_region(image, "bottom")
    values = S.discriminant_for(image, region, "rank")
    assert values[~region].max() == 0.0


def test_an_unknown_normalisation_is_rejected():
    region = np.ones((SIZE, SIZE), dtype=bool)
    with pytest.raises(S.SegmentationError, match="unknown normalisation"):
        S.normalise_within(np.zeros((SIZE, SIZE)), region, "histogram")


def test_normalisation_is_a_rule_for_everyone_not_a_patient_fix():
    """A parameter adjusted until one image works would fit that image and hide
    the fairness problem. The same rule must apply to every patient, so it must
    change the light-skinned fixture's handling too -- or it is doing nothing
    except where someone looked."""
    region_light = S.analysis_region(synthetic_face(), "bottom")
    light_raw = S.discriminant_for(synthetic_face(), region_light, "none")[region_light]
    light_ranked = S.discriminant_for(synthetic_face(), region_light, "rank")[region_light]
    assert not np.allclose(light_raw, light_ranked)


def test_colour_separation_is_not_normalised():
    """It is the run-1 record of what a FIXED cut does. Normalising would turn
    that into a quantile cut, and it would no longer be that record."""
    image = dark_skinned_face()
    region = S.analysis_region(image, "bottom")
    assert np.array_equal(
        S.segment(image, "colour_separation", region, normalisation="none"),
        S.segment(image, "colour_separation", region, normalisation="rank"),
    )


def test_the_fairness_screen_is_declared_and_reports_its_proxy():
    summary = probe([synthetic_face(), dark_skinned_face()])
    fairness = summary["fairness"]

    assert fairness["declared"] == "2026-07-28, before run 3"
    assert "median_value_in_region" in fairness["proxy"]
    assert fairness["darkest_median_value"] <= fairness["lightest_median_value"]
    assert isinstance(fairness["passes"], bool)


def test_the_fairness_note_names_the_cause_and_the_caveats():
    """"Hard image" and "the method assumes a skin tone this patient does not
    have" are different findings recorded differently, and n=1 is not a sample."""
    note = probe([synthetic_face(), dark_skinned_face()])["fairness"]["note"]
    assert "implicitly calibrated on lighter skin" in note
    assert "NOT REPRESENTATIVE" in note
    assert "British cleft cohort" in note
    assert "HIDE the fairness problem" in note


def test_the_attribution_note_warns_that_agreement_is_not_cause():
    """genuine_difficulty reached the right verdict for the wrong reason on run
    3: it reads as "hard image" and the cause was the method's assumptions."""
    note = attribution_for(synthetic_face())["attribution_note"]
    assert "FAIRNESS limitation" in note
    assert "DO NOT EXHAUST THE CAUSES" in note


# --------------------------------------------------------------------------
# attributing a failure: limitation or defect
# --------------------------------------------------------------------------


def test_iou_is_zero_for_disjoint_masks_and_one_for_identical():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[:5] = True
    b[5:] = True
    assert S.iou(a, b) == 0.0
    assert S.iou(a, a) == 1.0
    assert S.iou(b.copy() & False, b & False) == 0.0


def attribution_for(image, spatial_prior="bottom") -> dict:
    region = S.analysis_region(image, spatial_prior)
    return S.attribute(S.run_methods(image, spatial_prior), image, region)


def test_a_working_image_attributes_as_ok():
    """A two-lipped face is the realistic input: the contour spans both lips and
    the split divides them. A single-lip fixture is the limitation case below."""
    assert attribution_for(two_lipped_face())["attribution"] == "ok"


def test_agreeing_methods_that_both_fail_are_genuine_difficulty():
    """Two independent methods finding the same non-lip thing points at the
    image -- poor repair with no clear vermillion border, an open mouth putting
    teeth where the lip should be, light that flattens the colour difference.
    That is a LIMITATION to record, not a defect to fix."""
    results = [
        S.MethodResult(
            method=method,
            mask=np.zeros((SIZE, SIZE), dtype=bool),
            description={"area": 0.0},
            plausible=False,
            reasons=["nothing found"],
        )
        for method in S.METHODS
    ]
    agreeing = np.zeros((SIZE, SIZE), dtype=bool)
    agreeing[150:170, 100:130] = True
    for result in results:
        result.mask = agreeing.copy()

    image = featureless_face()
    region = S.analysis_region(image, "bottom")
    assert S.attribute(results, image, region)["attribution"] == "genuine_difficulty"


def test_disagreeing_methods_are_method_breakdown():
    """If the two disagree about what is there, at least one is failing. That is
    a DEFECT to fix, not a limitation to record."""
    results = []
    for index, method in enumerate(S.METHODS):
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        # Disjoint masks: no overlap at all between instrument and comparator.
        mask[160 + index * 12 : 168 + index * 12, 40 + index * 40 : 70 + index * 40] = True
        results.append(
            S.MethodResult(
                method=method, mask=mask, description={"area": 0.001},
                plausible=False, reasons=["nothing found"],
            )
        )

    image = featureless_face()
    region = S.analysis_region(image, "bottom")
    assert S.attribute(results, image, region)["attribution"] == "method_breakdown"


def test_the_attribution_explains_how_to_treat_each_outcome():
    described = attribution_for(synthetic_face())["attribution_note"]
    assert "limitation" in described.lower()
    assert "defect" in described.lower()
    assert "LOOK AT THE SHEET" in described


def test_the_attribution_reports_the_relaxation_growth_per_patient():
    """The magnitude check: a mask that grew by half has probably crossed into
    the lower lip."""
    attribution = attribution_for(synthetic_face())
    assert attribution["relaxation_growth"] >= 1.0
    assert "relaxation_growth_high" in attribution


# --------------------------------------------------------------------------
# the verdict's new blocks
# --------------------------------------------------------------------------


def test_the_verdict_counts_failures_by_attribution():
    good = {m: True for m in S.METHODS}
    bad = {m: False for m in S.METHODS}
    summary = S.verdict(
        [report(0.8, good)] * 6
        + [report(0.8, bad, attribution="genuine_difficulty")]
        + [report(0.8, bad, attribution="method_breakdown")]
    )
    assert summary["failures"]["n_failures"] == 2
    assert summary["failures"]["by_attribution"] == {
        "genuine_difficulty": 1,
        "method_breakdown": 1,
    }
    assert "CLUSTER-ONLY" in summary["failures"]["note"]


def test_the_verdict_reports_the_relaxation_it_applied():
    summary = S.verdict([report(0.8, {m: True for m in S.METHODS}, growth=1.3)] * 4)
    relaxation = summary["relaxation"]
    assert relaxation["relaxation"] == S.CONTOUR_RELAXATION
    assert relaxation["mean_growth"] == 1.3
    assert relaxation["n_growth_high"] == 0
    assert "UPPER lip" in relaxation["note"]


def test_an_over_grown_mask_is_flagged():
    summary = S.verdict([report(0.8, {m: True for m in S.METHODS}, growth=2.4)] * 3)
    assert summary["relaxation"]["n_growth_high"] == 3


def test_the_verdict_names_the_instrument():
    summary = S.verdict([report(0.8, {m: True for m in S.METHODS})])
    assert summary["instrument"] == S.INSTRUMENT
    assert summary["comparator"] == S.COMPARATOR


def test_the_screen_reports_the_role_it_plays():
    """It flags candidates for review. It has never decided anything on its own,
    and run 1 is the proof: the screen said proceed, the sheet said undecided."""
    summary = S.verdict([report(0.8, {m: True for m in S.METHODS})])
    assert summary["role"] == "screen"
    assert summary["decided_by"] == "human review of the contact sheet"
    assert "NOT A GATE" in summary["note"]
    assert summary["criteria_provenance"]["amendment_is_post_hoc"] is True


def test_every_failing_criterion_is_reported_not_just_the_first():
    """"It failed on area" and "it failed on everything" are different findings,
    and the second is much stronger evidence that nothing was found."""
    _, reasons = S.is_plausible(
        {"area": 0.9, "centroid_x": 0.05, "centroid_y": 0.1, "largest_component": 0.1}
    )
    assert len(reasons) == 4


def test_the_criteria_numbers_are_exactly_as_pre_declared():
    """Pinned. The amendment changed which denominator area_max is read against;
    it changed no number, and a diff here should be a deliberate act."""
    assert S.PLAUSIBLE == {
        "area_min": 0.02,
        "area_max": 0.35,
        "centroid_y_min": 0.50,
        "centroid_x_min": 0.30,
        "centroid_x_max": 0.70,
        "largest_component_min": 0.60,
        # ADDED before run 3, and expected to fail on arrival -- see
        # CRITERIA_PROVENANCE. Adding a criterion is a different act from
        # loosening one, and this test is where that act has to be visible.
        "fraction_above_split_min": 0.80,
    }


def test_the_miscalibrated_floor_is_recorded_and_left_alone():
    """[MEASURED run 5] upper_lip mean area 0.0230 against an area_min of 0.02.

    The floor was calibrated for a both-lips target, so a correct upper-lip mask
    sits on it and about half fall through. It stays wrong: area_max was
    re-expressed after run 1, fraction_above_fissure added before run 3 and
    renamed at run 5, and a fourth amendment would only fit the criterion set
    more tightly to the data. What it means is that the SCREEN NEVER DECIDED
    ANYTHING -- the human review was the operative procedure every time.
    """
    assert S.PLAUSIBLE["area_min"] == 0.02, "unchanged, deliberately"
    assert "known_miscalibrated" in S.CRITERIA_PROVENANCE
    assert "0.0230" in S.CRITERIA_PROVENANCE["known_miscalibrated"]
    assert "on purpose" in S.CRITERIA_PROVENANCE["known_miscalibrated"]


def test_the_amendment_records_itself_as_post_hoc():
    """Nobody wrote "relative to the search region" before run 1. The record has
    to say that rather than implying it was always explicit."""
    provenance = S.CRITERIA_PROVENANCE
    assert provenance["amendment_is_post_hoc"] is True
    assert provenance["numbers_unchanged"] is True
    assert "before run 1" in provenance["pre_declared"]
    assert "after run 1" in provenance["amended"]
    assert "modest fraction of what was searched" in provenance["invariant"]


# --------------------------------------------------------------------------
# the hole the spatial prior opens -- detected, reported, NOT patched
# --------------------------------------------------------------------------


def test_band_filling_is_rejected_by_the_region_relative_ceiling():
    """The failure the amendment exists for.

    Under the literal frame-relative reading this passed everything: the
    face-masked bottom band is 0.168 of the frame, comfortably inside
    [0.02, 0.35], with centroid_y 0.77, centroid_x 0.50 and one component. Read
    against the search region it is 1.0 and fails outright.
    """
    face = S.face_mask(featureless_face())
    region = S.prior_mask((SIZE, SIZE), "bottom") & face
    description = S.describe_mask(region, region)

    # The literal reading, kept so the reason for the amendment stays visible.
    assert (
        S.PLAUSIBLE["area_min"] <= description["area"] <= S.PLAUSIBLE["area_max"]
    ), "frame-relative area cannot catch band-filling -- this is why it changed"

    plausible, reasons = S.is_plausible(description)
    assert not plausible
    assert any("SEARCH REGION" in reason for reason in reasons)


def test_a_speckle_is_rejected_by_the_frame_relative_floor():
    """The OTHER end, and why one denominator is not enough.

    Reading area_min against the region too would admit this: a blob of 0.4% of
    the frame scores 0.023 against a floor of 0.02. It sits low, on the midline,
    in one piece -- so centroid_y, centroid_x and largest_component all pass it,
    and nothing else would have caught it.
    """
    face = S.face_mask(featureless_face())
    region = S.prior_mask((SIZE, SIZE), "bottom") & face

    speckle = np.zeros((SIZE, SIZE), dtype=bool)
    side = int(np.sqrt(0.004 * SIZE * SIZE))
    row = int(0.80 * SIZE)
    speckle[row : row + side, SIZE // 2 - side // 2 : SIZE // 2 + side // 2] = True
    description = S.describe_mask(speckle & region, region)

    # Region-relative would have passed it, and nothing else rejects it.
    assert description["area_within_region"] > S.PLAUSIBLE["area_min"]
    assert description["centroid_y"] > S.PLAUSIBLE["centroid_y_min"]
    assert description["largest_component"] >= S.PLAUSIBLE["largest_component_min"]

    plausible, reasons = S.is_plausible(description)
    assert not plausible
    assert any("speckle" in reason for reason in reasons)


def test_a_genuine_lip_region_passes_both_ends():
    image = synthetic_face()
    region = S.analysis_region(image, "bottom")
    description = S.describe_mask(S.otsu_threshold(image, region), region)

    plausible, reasons = S.is_plausible(description)
    assert plausible and reasons == []
    assert description["saturates_region"] is False


def test_with_no_search_region_the_two_denominators_coincide():
    """So the amendment is provably conservative: with no region restriction it
    reduces exactly to the pre-declared form, and run 1 is unaffected."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[100:140, 90:150] = True
    description = S.describe_mask(mask)
    assert description["area"] == description["area_within_region"]


def test_saturation_stays_a_separate_flag():
    """area_max now catches the same case, but the two can disagree at the
    margin and a mask at 0.88 of its region is worth seeing regardless."""
    face = S.face_mask(featureless_face())
    region = S.prior_mask((SIZE, SIZE), "bottom") & face
    description = S.describe_mask(region, region)
    assert description["saturates_region"] is True
    assert "saturates_region" not in " ".join(S.is_plausible(description)[1])


# --------------------------------------------------------------------------
# describing a mask
# --------------------------------------------------------------------------


def test_an_empty_mask_describes_itself_without_dividing_by_zero():
    description = S.describe_mask(np.zeros((16, 16), dtype=bool))
    assert description["area"] == 0.0
    assert description["largest_component"] == 0.0
    assert description["n_pixels"] == 0


def test_the_centroid_finds_a_known_block():
    mask = np.zeros((100, 100), dtype=bool)
    mask[70:90, 40:60] = True
    description = S.describe_mask(mask)
    assert description["centroid_y"] == pytest.approx(0.795, abs=0.01)
    assert description["centroid_x"] == pytest.approx(0.495, abs=0.01)


def test_two_blobs_halve_the_largest_component_share():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:20, 10:20] = True
    mask[70:80, 70:80] = True
    assert S.largest_component_share(mask) == pytest.approx(0.5)


def test_the_overlay_marks_only_the_mask():
    image = featureless_face()
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[100:120, 100:120] = True
    tinted = S.overlay(image, mask)
    assert not np.array_equal(tinted[110, 110], image[110, 110])
    assert np.array_equal(tinted[10, 10], image[10, 10])


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------


def report(
    separability: float,
    plausible: dict,
    saturating: dict | None = None,
    attribution: str = "ok",
    growth: float = 1.2,
) -> dict:
    saturating = saturating or {m: False for m in S.METHODS}
    return {
        "patient_id": 1,
        "separability": separability,
        "attribution": {
            "attribution": attribution,
            "relaxation_growth": growth,
            "relaxation_growth_high": growth > S.RELAXATION_GROWTH_LIMIT,
            "instrument_comparator_iou": 0.9,
        },
        "methods": {
            method: {
                "area": 0.08,
                "area_within_region": 0.95 if saturating[method] else 0.3,
                "saturates_region": saturating[method],
                "centroid_x": 0.5,
                "centroid_y": 0.78,
                "largest_component": 0.9,
                "plausible": plausible[method],
                "reasons": [],
            }
            for method in S.METHODS
        },
    }


def test_a_method_plausible_on_a_majority_is_usable():
    good = {m: True for m in S.METHODS}
    bad = {m: False for m in S.METHODS}
    summary = S.verdict([report(0.8, good)] * 3 + [report(0.8, bad)])
    assert summary["methods"]["otsu_threshold"]["usable"] is True
    assert summary["symnose_screen"] == "proceed"


def test_one_good_patient_out_of_nine_is_not_a_technique():
    good = {m: True for m in S.METHODS}
    bad = {m: False for m in S.METHODS}
    summary = S.verdict([report(0.8, good)] + [report(0.8, bad)] * 8)
    assert summary["usable_methods"] == []
    assert summary["symnose_screen"] == "out"


def test_weak_separability_is_flagged_independently_of_the_methods():
    """The stronger signal: if the discriminant is not bimodal, no threshold on
    it could have worked, whatever the methods scored."""
    good = {m: True for m in S.METHODS}
    summary = S.verdict([report(0.05, good)] * 3)
    assert summary["separability_is_weak"] is True
    assert summary["symnose_screen"] == "proceed", (
        "the two signals are reported separately -- a weak discriminant does not "
        "silently overrule what was actually measured"
    )


def test_the_verdict_records_the_criteria_it_applied():
    """So a reader can see they were not loosened until something passed."""
    summary = S.verdict([report(0.8, {m: True for m in S.METHODS})])
    assert summary["plausibility_criteria"] == S.PLAUSIBLE


def test_a_method_that_passes_by_saturating_is_flagged_in_the_verdict():
    """The one field to read before the recommendation. A method here passed
    while segmenting nothing."""
    good = {m: True for m in S.METHODS}
    saturating = {m: m == "chan_vese" for m in S.METHODS}
    summary = S.verdict([report(0.8, good, saturating)] * 5)

    assert summary["usable_but_saturating"] == ["chan_vese"]
    assert summary["methods"]["chan_vese"]["n_plausible"] == 5
    assert summary["methods"]["chan_vese"]["n_plausible_not_saturating"] == 0
    assert summary["methods"]["otsu_threshold"]["n_plausible_not_saturating"] == 5


def test_a_clean_pass_is_not_flagged():
    summary = S.verdict([report(0.8, {m: True for m in S.METHODS})] * 5)
    assert summary["usable_but_saturating"] == []


def test_the_verdict_says_which_contrast_separability_measures():
    """The run-1 correction, carried in the record rather than remembered."""
    summary = S.verdict([report(0.8, {m: True for m in S.METHODS})])
    assert summary["separability_measures"] == "lip_vs_skin_within_the_analysis_region"
    assert "face-versus-background" in summary["note"]


def test_the_verdict_explains_what_out_means():
    summary = S.verdict([report(0.8, {m: False for m in S.METHODS})])
    assert "limitations" in summary["note"]
    assert "mirror-difference" in summary["note"]


def test_the_verdict_carries_no_patient_data():
    """SHAREABLE. Counts and aggregates only -- the per-patient list stays on the
    cluster with the sheet."""
    summary = S.verdict([report(0.8, {m: True for m in S.METHODS})] * 3)
    assert "patient_id" not in summary
    rendered = repr(summary)
    assert "patient_id" not in rendered


def test_an_empty_report_list_is_refused():
    with pytest.raises(S.SegmentationError, match="no per-patient reports"):
        S.verdict([])


# --------------------------------------------------------------------------
# end to end on synthetic faces
# --------------------------------------------------------------------------


def probe(images, spatial_prior=S.DEFAULT_SPATIAL_PRIOR) -> dict:
    """The whole diagnostic over a set of images, as the task runs it."""
    reports = []
    for index, image in enumerate(images):
        region = S.analysis_region(image, spatial_prior)
        results = S.run_methods(image, spatial_prior)
        per_method = {
            result.method: {
                **result.description,
                "plausible": result.plausible,
                "reasons": result.reasons,
            }
            for result in results
        }
        reports.append(
            {
                "patient_id": index,
                "separability": S.separability_of(image, region),
                "methods": per_method,
                "attribution": S.attribute(results, image, region),
            }
        )
    return S.verdict(reports)


def test_the_geometric_split_halves_a_mask_that_was_already_upper_lip_only():
    """**The trade, and it must not be a surprise on the sheet.**

    The fissure rule was correct wherever it fired and inert on 7 of 9. The
    geometric split always fires -- it has no not-found case -- so it halves a
    contour that captured only the upper lip, which happens when a patient's
    lower lip is outside the crop. That was patient 17 in run 4.

    And no shape statistic can distinguish the two cases: removed_fraction is
    ~0.5 either way, because the two masks look the same. Only the sheet can say
    which patient is which.
    """
    upper_only = attribution_for(synthetic_face())
    both_lips = attribution_for(two_lipped_face())

    assert upper_only["split_removed_fraction"] > 0.25, "a correct mask is halved"
    assert both_lips["split_removed_fraction"] > 0.25, "and so is a both-lips one"
    assert abs(
        upper_only["split_removed_fraction"] - both_lips["split_removed_fraction"]
    ) < 0.35, "which is why the number cannot tell them apart"


def test_the_verdict_states_the_split_limitation():
    summary = probe([synthetic_face(), two_lipped_face()])
    limitation = summary["fissure"]["limitation"]
    assert "ALWAYS FIRES" in limitation
    assert "cannot tell them apart" in limitation
    assert "ONLY THE SHEET" in limitation
    assert len(summary["fissure"]["removed_fraction_sorted"]) == 2


def test_the_diagnostic_says_yes_on_faces_that_have_lips():
    summary = probe(
        [two_lipped_face(upper=(190, 70, 80 + offset)) for offset in range(5)]
    )
    assert summary["usable_methods"], (
        "on synthetic faces with an unambiguous lip region every method should "
        "succeed; if this fails the instrument is broken, not the data"
    )
    assert summary["symnose_screen"] == "proceed"
    assert summary["usable_but_saturating"] == [], (
        "and it must succeed by segmenting, not by filling the band"
    )


def test_the_diagnostic_says_no_on_faces_that_have_none():
    """The answer the protocol predicts for real data. It must come out cleanly."""
    summary = probe([featureless_face() for _ in range(5)])
    assert summary["symnose_screen"] == "out" or summary[
        "usable_but_saturating"
    ], (
        "with nothing to find, either nothing passes or whatever passes is "
        "flagged as having filled its region"
    )
    assert summary["separability_is_weak"] is True


def test_the_prior_changes_the_answer_on_a_face_with_a_nose():
    """End to end, the second fix: without the prior the mask spans nose and
    lips and sits too high; with it, the lips alone."""
    images = [synthetic_face(nose=True) for _ in range(5)]
    without = probe(images, spatial_prior="none")
    with_prior = probe(images, spatial_prior="bottom")

    assert (
        with_prior["methods"]["otsu_threshold"]["mean_centroid_y"]
        > without["methods"]["otsu_threshold"]["mean_centroid_y"]
    )


# --------------------------------------------------------------------------
# as a task
# --------------------------------------------------------------------------


def test_the_task_writes_a_cluster_only_sheet_and_a_shareable_verdict(
    tmp_path, clean_repo, monkeypatch
):
    """The sheet renders patient faces, so the tier guard must keep it on the
    cluster while the verdict travels."""
    from test_contact import write_cohort

    from cleft.provenance import hash_dir

    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    folders = tmp_path / "photos"
    rows = write_cohort(folders, n_patients=6)

    artifact = tmp_path / "cleft_v1"
    artifact.mkdir()
    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for row in rows:
        lines.append(
            f"{row['patient_id']},{row['frontal_id']},,2.0,2.0,2.0,2.0,2.0,"
            f"0.2,0.2,0.2,0.2,0.2,{row['class3']},0"
        )
    (artifact / "manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = builders.write_config(
        tmp_path / "probe.yaml",
        tier="dev",
        phase="p4",
        inputs=[
            {
                "name": "m",
                "path": str(artifact),
                "rollup_sha256": hash_dir(artifact)["rollup"],
            },
            {
                "name": "p",
                "path": str(folders),
                "rollup_sha256": hash_dir(folders)["rollup"],
            },
        ],
        task={
            "kind": "segmentation_probe",
            "manifest_artifact": "m",
            "patient_folders": "p",
            "n_patients": 3,
        },
    )

    run_dir = main(["--config", str(config), "--out", str(tmp_path / "runs")])

    outputs = json.loads((run_dir / "outputs.json").read_text(encoding="utf-8"))
    tiers = {entry["name"]: entry["tier"] for entry in outputs["outputs"]}
    assert tiers["contact_sheet_segmentation.png"] == "CLUSTER-ONLY"
    assert tiers["metrics.json"] == "SHAREABLE"

    # Where a failing patient is NAMED. "One patient failed" is not actionable;
    # "patient 143 failed and the two methods agreed" is. The filename contains
    # "patient", so the tier guard would refuse to let it be SHAREABLE.
    assert tiers["segmentation_per_patient.json"] == "CLUSTER-ONLY"
    per_patient = json.loads(
        (run_dir / "segmentation_per_patient.json").read_text(encoding="utf-8")
    )
    assert len(per_patient) == 3
    assert all("patient_id" in entry for entry in per_patient)
    assert all("attribution" in entry for entry in per_patient)

    # And the SHAREABLE side carries counts only.
    assert "patient_id" not in (run_dir / "metrics.json").read_text(encoding="utf-8")
    assert (run_dir / "contact_sheet_segmentation.png").stat().st_size > 0

    summary = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert summary["n_patients"] == 3
    assert summary["symnose_screen"] in ("proceed", "out")
    assert sorted(summary["methods"]) == sorted(S.METHODS)


def test_the_shipped_config_is_valid(repo_root):
    from cleft.config import load_config

    loaded = load_config(repo_root / "configs" / "p4_segmentation_probe.yaml")
    assert loaded["task"]["kind"] == "segmentation_probe"
    assert loaded["task"]["geometry"] == S.DEFAULT_GEOMETRY
    assert loaded["task"]["spatial_prior"] == S.DEFAULT_SPATIAL_PRIOR
    assert loaded["task"]["face_mask"] is True


def test_run_1_is_still_reproducible(tmp_path):
    """`spatial_prior: none` plus `face_mask: false` is the whole frame, which is
    what run 1 searched. Kept so the run that produced the whole-face masks can
    be re-derived rather than only described."""
    image = synthetic_face()
    region = S.analysis_region(image, "none", use_face_mask=False)
    assert region.all()

    description = S.describe_mask(S.otsu_threshold(image, region), region)
    assert description["area"] == description["area_within_region"]
