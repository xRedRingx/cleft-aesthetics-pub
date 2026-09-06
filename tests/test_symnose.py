"""SymNose reimplementation (Phase 4 §3.2).

Mirror the segmented upper lip about the midline, superimpose, quantify the
mismatch. What is testable on the laptop is that the instrument measures
asymmetry and not something else: a symmetric mask must score zero, a known
asymmetry must score more than a smaller one, and the measures must not move
when nothing about the shape does.

Synthetic masks throughout. No real data, at any point.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry import symnose as SN

SIZE = 224
MID = (SIZE - 1) / 2.0


def lip_mask(
    half_width: int = 40, height: int = 14, offset: int = 0, top: int = 150
) -> np.ndarray:
    """A rectangular "upper lip", optionally shifted off the midline."""
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    centre = int(round(MID)) + offset
    mask[top : top + height, centre - half_width : centre + half_width] = True
    return mask


def lopsided_mask(left_extra: int = 12) -> np.ndarray:
    """Symmetric, then more lip added on one side only."""
    mask = lip_mask()
    centre = int(round(MID))
    mask[150:164, centre - 40 - left_extra : centre - 40] = True
    return mask


# --------------------------------------------------------------------------
# mirroring
# --------------------------------------------------------------------------


def test_mirroring_a_symmetric_mask_returns_it_unchanged():
    mask = lip_mask()
    assert np.array_equal(SN.mirror_about(mask, MID), mask)


def test_mirroring_twice_is_the_identity():
    mask = lopsided_mask()
    once = SN.mirror_about(mask, MID)
    assert np.array_equal(SN.mirror_about(once, MID), mask)


def test_mirroring_moves_an_off_centre_mask_to_the_other_side():
    mask = lip_mask(offset=20)
    mirrored = SN.mirror_about(mask, MID)
    assert not np.array_equal(mirrored, mask)
    columns = np.nonzero(mask.any(axis=0))[0]
    mirror_columns = np.nonzero(mirrored.any(axis=0))[0]
    assert columns.mean() > MID > mirror_columns.mean()


def test_the_image_midline_and_the_mask_centroid_are_different_axes():
    """A crop that is not centred on the face turns miscentring into apparent
    asymmetry about the image midline. Neither axis is right on its own."""
    mask = lip_mask(offset=25)
    assert SN.axis_for(mask, "image_midline") == pytest.approx(MID)
    assert SN.axis_for(mask, "mask_centroid") > MID


def test_an_unknown_axis_is_rejected():
    with pytest.raises(SN.SymNoseError, match="unknown axis"):
        SN.axis_for(lip_mask(), "nose_tip")


def test_the_boundary_is_the_outline_not_the_area():
    """SymNose superimposes traced boundaries, which is why the primary index is
    a boundary distance."""
    mask = lip_mask()
    outline = SN.boundary_of(mask)
    assert outline.sum() < mask.sum()
    assert not SN.boundary_of(np.zeros((SIZE, SIZE), dtype=bool)).any()


# --------------------------------------------------------------------------
# the index measures asymmetry
# --------------------------------------------------------------------------


def test_a_symmetric_lip_scores_zero():
    """The floor. Anything above zero here is an artefact of the instrument."""
    measures = SN.symnose_measures(lip_mask())
    assert measures["index"] == 0.0
    assert measures["asymmetry_1_minus_dice"] == 0.0
    assert measures["overlap_dice"] == 1.0
    assert measures["area_difference_fraction"] == 0.0


def test_an_asymmetric_lip_scores_above_zero():
    assert SN.symnose_measures(lopsided_mask())["index"] > 0.0


def test_more_asymmetry_scores_higher():
    """Monotonicity. Without it the index orders patients by nothing."""
    small = SN.symnose_measures(lopsided_mask(left_extra=6))["index"]
    large = SN.symnose_measures(lopsided_mask(left_extra=24))["index"]
    assert large > small > 0.0


def test_the_side_the_asymmetry_falls_on_does_not_change_the_index():
    """Laterality is not recorded anywhere in this cohort (PLAN §4.1), so an
    index that scored left- and right-sided asymmetry differently would be
    measuring something the labels cannot contain."""
    left = lopsided_mask()
    right = SN.mirror_about(left, MID)
    assert SN.symnose_measures(left)["index"] == pytest.approx(
        SN.symnose_measures(right)["index"], abs=1e-9
    )


def test_the_index_is_scale_free():
    """Normalised by the lip's own scale, so it does not simply track how big the
    crop was -- two lips of different size with the same shape score the same."""
    small = SN.symnose_measures(lopsided_mask(left_extra=6))["index"]
    scaled = np.zeros((SIZE, SIZE), dtype=bool)
    source = lopsided_mask(left_extra=6)
    rows, columns = np.nonzero(source)
    # Double the mask about its own centre, shape preserved.
    scaled[
        np.clip((rows - 157) * 2 + 157, 0, SIZE - 1),
        np.clip((columns - int(MID)) * 2 + int(MID), 0, SIZE - 1),
    ] = True
    from scipy.ndimage import binary_closing

    scaled = binary_closing(scaled, structure=np.ones((3, 3)))
    assert SN.symnose_measures(scaled)["index"] == pytest.approx(small, rel=0.6)


def test_an_empty_mask_does_not_divide_by_zero():
    measures = SN.symnose_measures(np.zeros((SIZE, SIZE), dtype=bool))
    assert measures["index"] == 0.0
    assert measures["area_px"] == 0


# --------------------------------------------------------------------------
# the interpretable half
# --------------------------------------------------------------------------


def test_the_half_measures_find_a_known_area_difference():
    """"The left vermillion is 12% larger than the right" is a sentence a
    clinician can check on the photograph. A boundary distance is not."""
    measures = SN.symnose_measures(lopsided_mask(left_extra=12))
    assert measures["left_area"] > measures["right_area"]
    assert measures["area_difference_fraction"] > 0.0


def test_the_centroid_offset_reports_the_crop_centring_confound():
    """Small means the axis choice does not matter; large is a confound to
    declare rather than a parameter to pick."""
    assert SN.symnose_measures(lip_mask())["centroid_offset_px"] < 1.0
    assert SN.symnose_measures(lip_mask(offset=25))["centroid_offset_px"] > 20.0


def test_mirroring_about_the_centroid_removes_pure_translation():
    """**Exactly why the centroid cannot be the primary axis.**

    A purely translated lip scores zero about its own centroid. In a unilateral
    cleft lateral displacement is genuine asymmetry and often the largest
    component of it, so an index that discards it is measuring the wrong thing.
    """
    shifted = lip_mask(offset=25)
    measures = SN.symnose_measures(shifted)
    assert measures["index_image_midline"] > measures["index_mask_centroid"]
    assert measures["index_mask_centroid"] == pytest.approx(0.0, abs=1e-9)


def test_the_image_midline_is_the_primary_index():
    measures = SN.symnose_measures(lopsided_mask())
    assert measures["primary_axis"] == "image_midline"
    assert measures["index"] == measures["index_image_midline"]
    assert SN.describe()["primary_axis"] == "image_midline"


def test_both_indices_are_always_reported():
    """Whichever is primary, the other travels with it and so does the offset."""
    for axis in SN.AXES:
        measures = SN.symnose_measures(lopsided_mask(), axis=axis)
        assert measures["index"] == measures[f"index_{axis}"]
        assert "index_image_midline" in measures
        assert "index_mask_centroid" in measures
        assert "centroid_offset_px" in measures


def test_the_lateral_component_is_the_gap_between_the_two_indices():
    """Large offsets make this the lateral-displacement part of the asymmetry --
    a finding in its own right, separating "the lip is shifted" from "the lip is
    misshapen"."""
    shifted = SN.symnose_measures(lip_mask(offset=25))
    centred = SN.symnose_measures(lip_mask())

    assert shifted["lateral_component"] == pytest.approx(
        shifted["index_image_midline"] - shifted["index_mask_centroid"], abs=1e-9
    )
    assert shifted["lateral_component"] > 0.0
    assert centred["lateral_component"] == pytest.approx(0.0, abs=1e-9)


def test_a_centred_symmetric_mask_makes_the_two_indices_agree():
    """If the offsets are a few pixels, miscentring is a theoretical worry."""
    measures = SN.symnose_measures(lip_mask())
    assert measures["centroid_offset_px"] < 1.0
    assert measures["index_image_midline"] == pytest.approx(
        measures["index_mask_centroid"], abs=1e-9
    )


def test_an_asymmetric_lip_shifts_its_own_centroid():
    """**The offset is not purely a miscentring measure, and that limits it.**

    A lip with more vermillion on one side moves its own centroid, so a large
    centroid_offset_px can mean a miscentred crop OR genuine asymmetry -- and the
    offset alone cannot separate them. Here the crop is perfectly centred and the
    offset is 6px, entirely from the asymmetry.
    """
    measures = SN.symnose_measures(lopsided_mask(left_extra=12))
    assert measures["centroid_offset_px"] > 3.0
    assert "CANNOT SEPARATE" in SN.describe()["axis_note"]


# --------------------------------------------------------------------------
# the feature matrix and the record
# --------------------------------------------------------------------------


def test_g1_is_refused_with_the_reason():
    with pytest.raises(SN.SymNoseError, match="wrong line"):
        SN.feature_matrix(np.zeros((1, SIZE, SIZE, 3), dtype=np.uint8), geometry="g1")


def test_the_feature_names_match_what_is_computed():
    measures = SN.symnose_measures(lopsided_mask())
    for name in SN.FEATURE_NAMES:
        assert name in measures, f"{name} is declared but not computed"


def test_describe_names_the_comparable_statistic():
    """[LITERATURE] The protocol's finding is about RANKED correlation, so
    Spearman is the comparable statistic and PCC is not. Quoting PCC against it
    would be the same quantity confusion R2 exists for."""
    described = SN.describe()
    assert "Spearman" in described["spearman_note"]
    assert "RANKED" in described["spearman_note"]
    assert "different quantities" in described["spearman_note"]


def test_describe_inherits_the_segmentation_limitations():
    """This arm is not a fresh start: it carries every constraint of the
    segmentation that feeds it, and the write-up must say so."""
    note = SN.describe()["note"]
    assert "unrecoverable" in note
    assert "halves a mask" in note
    assert "lighter skin" in note
    assert "not caveats about the segmentation alone" in note


def test_the_expected_correlation_is_stated_before_the_run():
    """**Before the number lands, not after.**

    SymNose measures upper-lip asymmetry; the label is an Asher-McDade composite
    over nasal form, nasal symmetry, nasolabial profile and vermillion border.
    Only part of the target is what this index measures, so a modest correlation
    is the expected outcome. Saying so afterwards would be indistinguishable
    from explaining away a disappointing number.
    """
    note = SN.describe()["construct_note"]
    assert "Asher-McDade" in note
    assert "MODEST CORRELATION IS THE EXPECTED OUTCOME" in note
    assert "before the run rather than after it" in note


def test_describe_explains_why_the_midline_is_primary():
    note = SN.describe()["axis_note"]
    assert "PRESERVES LATERAL DISPLACEMENT" in note
    assert "cannot be primary" in note


def test_describe_names_the_segmentation_instrument():
    from cleft.geometry import segmentation

    assert SN.describe()["segmentation_instrument"] == segmentation.INSTRUMENT


# --------------------------------------------------------------------------
# the audit: does the segmentation hold at 237, or only at the gate's 9?
# --------------------------------------------------------------------------


def face_with_lip(
    lip_colour=(190, 70, 80), pad: float = 0.12, skin=(222, 184, 156)
) -> np.ndarray:
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    image[:, :] = (255, 255, 255)
    edge = int(pad * SIZE)
    image[edge : SIZE - edge, edge : SIZE - edge] = skin
    image[int(0.72 * SIZE) : int(0.86 * SIZE), int(0.32 * SIZE) : int(0.68 * SIZE)] = (
        lip_colour
    )
    return image


def test_the_audit_reports_a_pass_rate_and_a_breakdown():
    images = np.stack([face_with_lip() for _ in range(4)])
    aggregate, per_patient = SN.audit(images)

    assert aggregate["n_patients"] == 4
    assert 0.0 <= aggregate["pass_rate"] <= 1.0
    assert len(per_patient) == 4
    from cleft.geometry import segmentation

    assert set(aggregate["failures_by_criterion"]) == set(segmentation.CRITERION_NAMES)


def test_the_breakdown_separates_a_cause_from_a_symptom():
    """A criterion that only ever co-occurs with others is not the reason; one
    that fails alone is."""
    aggregate, _ = SN.audit(np.stack([face_with_lip() for _ in range(4)]))
    for name, count in aggregate["failures_by_sole_criterion"].items():
        assert count <= aggregate["failures_by_criterion"][name]


def test_the_audit_reports_a_distribution_per_feature():
    aggregate, _ = SN.audit(np.stack([face_with_lip() for _ in range(4)]))
    for name in SN.FEATURE_NAMES:
        distribution = aggregate["feature_distributions"][name]
        assert distribution["n"] == 4
        assert distribution["min"] <= distribution["median"] <= distribution["max"]


def test_identical_inputs_make_every_feature_near_constant():
    """**The other diagnosis.** If the index barely moves across patients, the
    segmentation is producing similar masks regardless of input, and the arm
    measured nothing because nothing varied."""
    aggregate, _ = SN.audit(np.stack([face_with_lip() for _ in range(4)]))
    assert "index" in aggregate["near_constant_features"] or (
        aggregate["feature_distributions"]["index"]["sd"] == 0.0
    )


def test_varying_inputs_are_not_near_constant():
    images = np.stack(
        [face_with_lip(lip_colour=(190, 70, 80 + 12 * offset)) for offset in range(5)]
    )
    for offset in range(5):
        images[offset, 150 : 150 + 4 * offset, 60:80] = (190, 70, 80)

    aggregate, _ = SN.audit(images)
    assert aggregate["feature_distributions"]["index"]["n_distinct"] > 1


def test_the_audit_splits_the_pass_rate_by_brightness():
    """**The reason the audit runs.** The fairness finding rested on one patient
    in nine; this makes it a cohort-scale measurement."""
    images = np.stack(
        [face_with_lip(pad=0.10 + 0.005 * offset) for offset in range(8)]
    )
    fairness = SN.audit(images)[0]["fairness_by_brightness"]

    assert fairness["n_patients"] == 8
    assert set(fairness["quartiles"]) == set(SN.BRIGHTNESS_QUARTILES)
    assert sum(q["n"] for q in fairness["quartiles"].values()) == 8
    for quartile in fairness["quartiles"].values():
        if quartile["n"]:
            assert 0.0 <= quartile["pass_rate"] <= 1.0
            assert quartile["brightness_min"] <= quartile["brightness_max"]


def test_darker_patients_land_in_the_darkest_quartile():
    """The split has to actually order by the proxy, or the gradient means
    nothing. Graded tones rather than two groups: tied values collapse the
    quartiles, which is a fixture artifact and not what 237 real crops look
    like."""
    images = np.stack(
        [face_with_lip(skin=(100 + 18 * step, 60 + 16 * step, 46 + 14 * step))
         for step in range(8)]
    )
    quartiles = SN.audit(images)[0]["fairness_by_brightness"]["quartiles"]
    populated = [
        quartiles[name] for name in SN.BRIGHTNESS_QUARTILES if quartiles[name]["n"]
    ]
    assert len(populated) >= 2

    ceilings = [q["brightness_max"] for q in populated]
    assert ceilings == sorted(ceilings), "quartiles must run darkest to lightest"


def test_the_breakdown_says_how_it_fails_not_only_that_it_does():
    """Over-selection was patient 5's signature, so area_max in the darkest
    quartile is the specific pattern to look for."""
    from cleft.geometry import segmentation

    fairness = SN.audit(np.stack([face_with_lip() for _ in range(4)]))[0][
        "fairness_by_brightness"
    ]
    # Below four patients the split returns {"insufficient": True} with no
    # quartiles at all, and the loop below would pass over an empty dict without
    # checking anything. Assert there is something to iterate.
    assert fairness["quartiles"], "the split needs at least four patients"
    for quartile in fairness["quartiles"].values():
        if quartile["n"]:
            assert set(quartile["failures_by_criterion"]) == set(
                segmentation.CRITERION_NAMES
            )


def test_the_fairness_note_keeps_the_proxy_caveat_and_the_null_case():
    """A gradient is evidence of a brightness-linked failure pattern, not of a
    skin-tone one -- and a flat result is a finding that narrows the claim."""
    note = SN.audit(np.stack([face_with_lip() for _ in range(4)]))[0][
        "fairness_by_brightness"
    ]["note"]
    assert "PROXY, NOT SKIN TONE" in note
    assert "BRIGHTNESS-LINKED FAILURE PATTERN" in note
    assert "A FLAT RESULT IS ALSO A FINDING" in note
    assert "ethnicity data this cohort does not record" in note


def test_too_few_patients_to_split_says_so():
    fairness = SN.pass_rate_by_brightness([{"passes": True}, {"passes": False}])
    assert fairness["insufficient"] is True


def test_the_audit_states_what_a_low_pass_rate_would_mean():
    """A low rate makes the SymNose zero uninterpretable as a statement about
    anatomy; a high rate makes it a real finding."""
    aggregate, _ = SN.audit(np.stack([face_with_lip()]))
    note = aggregate["note"]
    assert "DIAGNOSTIC ONLY -- nothing is tuned" in note
    assert "says nothing about the construct" in note
    assert "near_constant_features is the other diagnosis" in note


def test_the_audit_records_the_gate_sample_fraction():
    """The gate validated on 9 patients. Against 237 that is 3.8%."""
    aggregate, _ = SN.audit(np.stack([face_with_lip() for _ in range(9)]))
    assert aggregate["gate_sample_fraction"] == 1.0


def test_the_null_is_recorded_as_uninformative():
    """**A null result for a non-faithful reimplementation, and nothing more.**

    Not evidence about upper-lip asymmetry, not about SymNose, and not for the
    scope decision that retired it. The index carries head tilt, crop centring
    and lateral displacement, any of which plausibly dominates the asymmetry it
    was meant to measure.
    """
    result = SN.MEASURED_RESULT
    assert result["pcc_mean"] == -0.066
    assert result["spearman_mean"] == -0.003
    assert result["faithful"] is False
    assert "NULL RESULT FOR A NON-FAITHFUL REIMPLEMENTATION" in result["interpretation"]
    assert "not evidence for the scope decision" in result["interpretation"].lower()


def test_this_is_not_claimed_to_be_symnose():
    """The differences are not refinements to add later: no intercanthal
    levelling, no registration, no nose, one axis instead of four."""
    described = SN.describe()
    assert described["is_symnose"] is False
    assert described["arm"] == "mirrored_upper_lip_index"

    faithfulness = SN.FAITHFULNESS
    assert faithfulness["faithful"] is False
    assert "intercanthal" in faithfulness["symnose_does"]
    assert "four" in faithfulness["symnose_does"]
    assert "FIXED IMAGE MIDLINE" in faithfulness["this_module_does"]
    assert "No rotation correction" in faithfulness["this_module_does"]
    assert "DOMINATES" in faithfulness["confounds_carried"]


def test_symnose_is_out_on_a_scope_boundary_not_on_the_null():
    """Every SymNose face measure depends on user-placed roundels at named
    anatomical points. Automating it needs landmark detection, which is outside
    the project's remit by supervisory decision. That argument would stand
    identically had this index scored 0.4."""
    why = SN.WHY_OUT
    assert why["reason"] == "scope boundary"
    assert "roundels" in why["argument"]
    assert "LANDMARK DETECTION" in why["argument"]
    assert "not deep learning" in why["argument"]
    assert "must not be offered in support" in why["not_evidenced_by"]
    assert why["faithful_reimplementation"] == "not attempted, by decision"


def test_the_null_and_the_scope_decision_are_kept_independent():
    """Presenting one in support of the other would be an argument neither of
    them makes."""
    assert "must not be offered in support" in SN.WHY_OUT["not_evidenced_by"]
    assert "mirror_difference_pcc" not in SN.MEASURED_RESULT, (
        "the comparison against the mirror index framed the null as a finding; "
        "it is not one, so the comparison does not belong beside it"
    )


def test_the_shipped_audit_config_is_valid(repo_root):
    from cleft.config import load_config

    loaded = load_config(repo_root / "configs" / "p4_symnose_audit.yaml")
    task = loaded["task"]
    assert task["kind"] == "symnose_audit"
    # It must audit exactly what the arm ran, or it explains a different run.
    arm = load_config(repo_root / "configs" / "p4_symnose_cv.yaml")["task"]
    assert task["geometry"] == arm["geometry"]


def test_the_shipped_config_is_valid(repo_root):
    from cleft.config import load_config

    loaded = load_config(repo_root / "configs" / "p4_symnose_cv.yaml")
    task = loaded["task"]
    assert task["kind"] == "train_cv"
    assert task["feature_source"] == "symnose"
    assert task["backbone"] == "ridge"
    assert task["geometry"] == SN.REQUIRED_GEOMETRY
    assert len(task["seeds"]) == 5
