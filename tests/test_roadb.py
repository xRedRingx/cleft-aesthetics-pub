"""Road B's Phase 2 structure, checked rather than read.

The axis that matters here is geometry, and it matters because the brief's
first draft collapsed it: non-square staging removes the pad, not all non-face
content. Everything below either pins that measurement or pins a requirement
Road A learned the hard way.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft import roadb


# --------------------------------------------------------------------------
# the measurement that made geometry an axis
# --------------------------------------------------------------------------


def test_non_square_removes_the_pad_but_not_the_corners():
    """**[MEASURED] The mask is a trapezium inside a rectangle.**

    Removing the pad leaves the corners, so non-square HALVES the non-content
    at G1 and only G2 reaches zero. Recomputed here from the same function the
    figure came from, so the record cannot drift from the code.
    """
    from cleft.train.augment_sheet import content_mask

    record = roadb.NON_CONTENT_FRACTION
    # Non-square: the content box IS the image, so there is no pad.
    for geometry, key in (("g1", "g1_non_square"), ("g2", "g2_non_square")):
        mask = content_mask(224, (0, 0, 224, 224), geometry)
        assert 1.0 - mask.mean() == pytest.approx(record[key], abs=5e-4), geometry

    assert record["g2_non_square"] == 0.0
    assert record["g1_non_square"] > 0.0, (
        "if G1 non-square reached zero the geometry axis would be pointless"
    )
    # Square staging at the median AR carries pad AND corners.
    width = int(round(224 * 0.7404))
    square = content_mask(224, ((224 - width) // 2, 0, width, 224), "g1")
    assert 1.0 - square.mean() == pytest.approx(
        record["g1_square_median_ar"], abs=5e-4
    )


def test_non_square_at_g1_halves_the_non_content_rather_than_removing_it():
    """The claim the brief now makes, as arithmetic."""
    record = roadb.NON_CONTENT_FRACTION
    ratio = record["g1_non_square"] / record["g1_square_median_ar"]
    assert 0.4 < ratio < 0.6, f"not a halving: {ratio:.3f}"
    assert "trapezium inside a rectangle" in record["why_it_is_an_axis"]


def test_the_two_non_square_expectations_are_different_baselines():
    """A normalised saliency map read at either needs its OWN expectation --
    which is what Phase 8's refuted framing finding was about."""
    record = roadb.NON_CONTENT_FRACTION
    assert record["g1_non_square"] != record["g2_non_square"]
    assert "different expectations" in record["consequence_for_saliency"]


# --------------------------------------------------------------------------
# the eight settings
# --------------------------------------------------------------------------


def test_there_are_ten_settings_not_four_and_not_eight():
    """**Four collapsed geometry; eight left the non-square families on two
    points, which is a delta and 29-of-30 says a delta cannot be claimed.**"""
    settings = roadb.staging_settings()
    assert len(settings) == 10
    assert len({s["name"] for s in settings}) == 10
    assert {s["resolution"] for s in settings} == {224, 512, 768}
    assert {s["aspect"] for s in settings} == {"square", "nonsquare"}
    assert {s["geometry"] for s in settings} == {"g1", "g2"}
    # 3 resolutions x 2 aspects x 2 geometries = 12, minus the two 224-square
    # cells Road A already has.
    assert len(settings) == 3 * 2 * 2 - 2


def test_the_eighth_cell_is_the_one_that_answers_the_question():
    """768 non-square G2 is the only setting with zero non-content. Every
    other cell leaves some."""
    settings = {s["name"]: s for s in roadb.staging_settings()}
    zero = sorted(
        name for name, s in settings.items()
        if s["expected_non_content"] == 0.0
    )
    assert zero == [
        "roadB_p2_224_nonsquare_g2", "roadB_p2_512_nonsquare_g2",
        "roadB_p2_768_nonsquare_g2",
    ]
    assert "roadB_p2_768_nonsquare_g2" in settings


def test_every_non_square_setting_carries_its_own_expectation():
    """Reported per setting so the axis is visible rather than inferred, and
    so a later phase reads the expectation instead of assuming one."""
    for setting in roadb.staging_settings():
        if setting["aspect"] == "nonsquare":
            expected = roadb.NON_CONTENT_FRACTION[
                f"{setting['geometry']}_non_square"
            ]
            assert setting["expected_non_content"] == expected, setting["name"]
        else:
            # Square staging's expectation is per patient, not per setting:
            # it depends on that patient's aspect ratio.
            assert setting["expected_non_content"] is None, setting["name"]


def test_the_resolution_axis_has_three_points_for_a_reason():
    """**A 0.03 improvement will not be claimable** -- 29 of 30 Road A
    comparisons were unresolvable. A monotone trend across three points is
    reportable where a single pair is not."""
    from cleft import ladder

    assert roadb.RESOLUTIONS == (224, 512, 768)
    assert roadb.CONTROL_RESOLUTION == 224
    assert len(roadb.RESOLUTIONS) >= 3
    assert ladder.COHORT_CANNOT_RESOLVE["at_scale"]["withdrawn"] == 29


# --------------------------------------------------------------------------
# the mixed population at 768
# --------------------------------------------------------------------------


def test_768_is_flagged_as_mixing_two_populations():
    """17.1% interpolated, 82.9% downsampled. Reported BOTH ways or the trend
    is partly about interpolation."""
    settings = {s["name"]: s for s in roadb.staging_settings()}
    assert all(
        s["mixes_populations"] for s in settings.values() if s["resolution"] == 768
    )
    assert roadb.UPSAMPLING_COUNTS[768] == 81
    assert roadb.UPSAMPLING_COUNTS[512] == 7
    assert roadb.UPSAMPLING_COUNTS[224] == 0
    fraction = roadb.UPSAMPLING_COUNTS[768] / roadb.N_SOURCES
    assert fraction == pytest.approx(0.171, abs=5e-4)


def test_512_also_mixes_though_far_less():
    """7 of 473 is 1.5% -- small, but not zero, and the flag is per patient
    so the distinction is available rather than assumed away."""
    assert roadb.UPSAMPLING_COUNTS[512] > 0
    fraction = roadb.UPSAMPLING_COUNTS[512] / roadb.N_SOURCES
    assert fraction == pytest.approx(0.015, abs=5e-4)


# --------------------------------------------------------------------------
# the requirements Road A learned the hard way
# --------------------------------------------------------------------------


def test_every_phase_2_requirement_is_carried():
    required = roadb.PHASE_2_REQUIREMENTS
    for key in ("asymmetry_gate", "flag_upsampled", "contact_sheet",
                "parity", "content_fraction"):
        assert key in required and required[key], key
    assert "machine epsilon" in required["asymmetry_gate"]
    assert "7 at 512, 81 at 768" in required["flag_upsampled"]
    assert "not claimed" in required["parity"]


def test_the_zero_pad_consumers_are_named_and_their_state_recorded():
    """**With no pad, ``pad_fraction`` is zero and the content box IS the
    image.** Both readers must behave at zero rather than divide by it."""
    consumers = roadb.ZERO_PAD_CONSUMERS
    assert "gradcam_sheet.out_of_content" in consumers
    assert consumers["gradcam_sheet.out_of_content"].startswith("HANDLED")
    assert "never exercised" in consumers["phase7c protection mask"]


def test_the_handled_consumer_really_is_handled():
    """Asserted against the code, not the record -- the record claims
    ``out_of_content`` returns None at a zero expectation, so check it does."""
    from cleft import gradcam_sheet

    report = gradcam_sheet.out_of_content(
        np.ones((14, 14)), np.ones((224, 224), dtype=bool)
    )
    assert report["ratio"] is None
    assert report["above"] is None
    assert "does not exist" in report["undefined_because"]


# --------------------------------------------------------------------------
# the gate that cannot be run per setting
# --------------------------------------------------------------------------


def test_the_analytic_asymmetry_gate_takes_no_size_argument():
    """**[MEASURED] It is blind to resolution**, so running it per setting
    would yield eight identical passes and test nothing about the settings."""
    import inspect

    from cleft.geometry import trapezium

    parameters = set(inspect.signature(trapezium.asymmetry_is_preserved).parameters)
    assert not parameters & {"size", "resolution", "width", "height"}, parameters
    assert parameters == {"trapezium", "heights", "offsets", "tolerance"}
    assert "no size argument" in roadb.ASYMMETRY_GATE_IS_RESOLUTION_BLIND[
        "analytic_gate"
    ]


def test_the_pixel_residual_is_what_varies_and_it_shrinks_with_resolution():
    """The resolution-dependent behaviour lives in ``unwarp``, not in the
    analytic map -- and it has never been gated at any size."""
    from cleft.geometry import trapezium

    residuals = roadb.ASYMMETRY_GATE_IS_RESOLUTION_BLIND["pixel_residual"]
    measured = {}
    for size in (224, 512, 768):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        row = size // 2
        half = trapezium.DEFAULT.half_width_at(0.5)
        column = int(round((0.5 - 0.10 * half) * size))
        image[row - 1:row + 2, column - 1:column + 2] = 255
        out = trapezium.unwarp(image)
        columns = np.where(out[row].max(axis=-1) > 0)[0]
        centre = (columns.mean() + 0.5) / out.shape[1]
        measured[size] = abs(centre - trapezium.unwarp_x(column / size, 0.5))

    for size, expected in residuals.items():
        assert measured[size] == pytest.approx(expected, abs=5e-5), size
    # Monotone in resolution, which is what "resampling quantisation" predicts.
    assert measured[224] > measured[512] > measured[768]
    # And six orders looser than the analytic tolerance.
    assert measured[224] > 1e-9 * 1e5


def test_machine_epsilon_describes_the_map_not_the_pixels():
    """Road A's claim is true of the analytic map. A reader will take it for
    the staged pixels, which are six orders looser."""
    record = roadb.ASYMMETRY_GATE_IS_RESOLUTION_BLIND
    assert "of the analytic map" in record["machine_epsilon_means_the_map"]
    assert "PIXEL residual" in record["phase_2_gate"]
    assert "analytic gate runs ONCE" in record["phase_2_gate"]


# --------------------------------------------------------------------------
# the trapezium, asserted rather than assumed
# --------------------------------------------------------------------------


def test_the_trapezium_half_widths_are_resolution_invariant():
    """**A mask specified in fractions SHOULD be resolution-invariant, which
    is a hope until it is a test.** Recomputed from the mask at each size."""
    from cleft.geometry.trapezium import BOT_HALF_WIDTH, TOP_HALF_WIDTH
    from cleft.train.augment_sheet import content_mask

    record = roadb.TRAPEZIUM_IS_RESOLUTION_INVARIANT
    assert record["specified"] == {"top": TOP_HALF_WIDTH, "bot": BOT_HALF_WIDTH}

    for size in (224, 512, 768):
        mask = content_mask(size, (0, 0, size, size), "g1")
        top = mask[0].sum() / 2 / size
        bot = mask[-1].sum() / 2 / size
        tolerance = 1.5 / size
        assert abs(top - TOP_HALF_WIDTH) <= tolerance, (size, top)
        assert abs(bot - BOT_HALF_WIDTH) <= tolerance, (size, bot)
        assert top == pytest.approx(record["realised"][size]["top"], abs=5e-5)
        assert bot == pytest.approx(record["realised"][size]["bot"], abs=5e-5)


# --------------------------------------------------------------------------
# the sheet's density, decided before it is built
# --------------------------------------------------------------------------


def test_the_sheet_density_decision_names_what_each_sheet_answers():
    """**A sheet too dense to read is a sheet nobody reviews**, and every
    phase that skipped review produced a defect the tests missed."""
    record = roadb.SHEET_DENSITY
    # [CORRECTED 2026-08-09] "eight" predated the two 224 non-square cells.
    assert record["axis_sheet"]["columns"] == "all ten settings"
    assert record["detail_strip"]["scale"] == "NATIVE"
    # And the strip's counts are the MEASURED frontal ones, not the 473-source
    # ceilings the first draft carried.
    assert "1 at 512" in record["detail_strip"]["patients"]
    assert "41 at 768" in record["detail_strip"]["patients"]
    # The reason there are two: downscaling destroys the sharpness difference.
    assert "destroys the sharpness" in record["why_two"]
    assert "ten files" in record["rejected"]


# --------------------------------------------------------------------------
# the per-setting gate: the pixel residual, and its ORDERING
# --------------------------------------------------------------------------


def test_the_pixel_residual_matches_the_recorded_shape():
    """Recomputed from the code, so the record cannot drift from it."""
    from cleft import roadb_staging

    recorded = roadb.ASYMMETRY_GATE_IS_RESOLUTION_BLIND["pixel_residual"]
    for size, expected in recorded.items():
        measured = roadb_staging.pixel_asymmetry_residual(size)
        assert measured == pytest.approx(expected, abs=5e-5), size


def test_the_gate_is_on_the_ordering_not_a_per_cell_tolerance():
    """**A threshold each cell passes individually can still hide a path that
    gets worse where it should get better.** So the gate is monotonicity."""
    from cleft import roadb_staging

    good = {224: 3.29e-03, 512: 1.40e-03, 768: 9.14e-04}
    report = roadb_staging.assert_residual_monotone(good)
    assert report["monotone"] is True
    assert report["analytic_tolerance"] == 1e-9

    # Every cell here is small, and the path still gets worse. A tolerance
    # would pass it.
    rising = {224: 1.0e-04, 512: 2.0e-04, 768: 3.0e-04}
    assert all(v < 1e-3 for v in rising.values()), "the fixture must pass a tolerance"
    with pytest.raises(roadb_staging.StagingError, match="rises with resolution"):
        roadb_staging.assert_residual_monotone(rising)


def test_a_single_resolution_cannot_be_gated_on_ordering():
    from cleft import roadb_staging

    with pytest.raises(roadb_staging.StagingError, match="at least two"):
        roadb_staging.assert_residual_monotone({512: 1e-3})


def test_the_real_residuals_pass_their_own_gate():
    from cleft import roadb_staging

    measured = {
        size: roadb_staging.pixel_asymmetry_residual(size)
        for size in (224, 512, 768)
    }
    assert roadb_staging.assert_residual_monotone(measured)["monotone"]


# --------------------------------------------------------------------------
# flags, parity, content fraction
# --------------------------------------------------------------------------


def test_upsampling_flags_are_per_patient_and_the_counts_are_asserted():
    from cleft import roadb_staging

    sides = np.array([300] * 7 + [900] * 466, dtype=float)
    flags = roadb_staging.upsampling_flags(sides, 512)
    assert flags["n_upsampled"] == 7
    assert len(flags["per_patient"]) == 473
    assert flags["report_both_ways"] is True
    roadb_staging.assert_upsampling_counts(flags)

    flags["n_upsampled"] = 9
    with pytest.raises(roadb_staging.StagingError, match="brief measured 7"):
        roadb_staging.assert_upsampling_counts(flags)


def test_224_upsamples_nothing_so_both_ways_is_not_claimed():
    from cleft import roadb_staging

    flags = roadb_staging.upsampling_flags(np.full(473, 900.0), 224)
    assert flags["n_upsampled"] == 0
    assert flags["report_both_ways"] is False


def test_parity_is_measured_between_boxes_not_asserted_from_a_shared_path():
    from cleft import roadb_staging

    observed = {1: (10, 0, 200, 224), 2: (20, 0, 180, 224)}
    recorded = {
        1: (10 / 224, 0.0, 200 / 224, 1.0),
        2: (20 / 224, 0.0, 180 / 224, 1.0),
    }
    report = roadb_staging.parity(observed, recorded, 224)
    assert report["n_compared"] == 2
    assert report["max_deviation"] == pytest.approx(0.0, abs=1e-9)
    assert "whether they do" in report["measured_not_claimed"]

    with pytest.raises(roadb_staging.StagingError, match="no patient appears"):
        roadb_staging.parity({9: (0, 0, 1, 1)}, recorded, 224)


def test_content_fraction_is_per_setting_and_shows_the_geometry_axis():
    from cleft import roadb_staging

    g1 = roadb_staging.content_fraction(224, "g1")
    g2 = roadb_staging.content_fraction(224, "g2")
    assert g2 == pytest.approx(1.0)
    assert g1 == pytest.approx(1.0 - roadb.NON_CONTENT_FRACTION["g1_non_square"],
                               abs=5e-4)
    assert g1 < g2, "the geometry axis must be visible in this number"


# --------------------------------------------------------------------------
# the two sheets
# --------------------------------------------------------------------------


def _crops(size=64):
    settings = [s["name"] for s in roadb.staging_settings()]
    rng = np.random.default_rng(3)
    return {
        name: {p: rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
               for p in (1, 2, 3)}
        for name in settings
    }


def test_the_axis_sheet_puts_every_setting_on_one_page():
    """**How the settings differ IS the review**, so the axis has to be
    visible at a glance rather than across ten files."""
    from cleft import roadb_sheet

    result = roadb_sheet.axis_sheet(_crops(), [1, 2, 3])
    assert len(result["columns"]) == len(roadb.staging_settings()) == 10
    assert result["sheet"].ndim == 3
    assert "sharpness" in result["does_not_answer"], (
        "the axis sheet must say what it cannot answer"
    )


def test_the_axis_sheet_downscales_and_says_what_that_costs():
    from cleft import roadb_sheet

    small = roadb_sheet.downscale(np.zeros((512, 512, 3), np.uint8))
    assert small.shape == (roadb_sheet.DISPLAY_SIZE, roadb_sheet.DISPLAY_SIZE, 3)
    assert "512 and 768 render identically here" in (
        roadb_sheet.axis_sheet(_crops(), [1])["does_not_answer"]
    )


def test_a_missing_crop_is_refused_rather_than_leaving_a_gap():
    """A column that silently skips a patient makes the rows stop meaning the
    same thing."""
    from cleft import roadb_sheet

    crops = _crops()
    del crops["roadB_p2_512_square_g1"][2]
    with pytest.raises(ValueError, match="crops missing"):
        roadb_sheet.axis_sheet(crops, [1, 2, 3])


def test_the_detail_strip_is_native_and_reports_what_it_truncated():
    """**A silently truncated sheet reads as complete.**"""
    from cleft import roadb_sheet

    crops = _crops()
    flagged = {"roadB_p2_768_square_g1": [1, 2, 3]}
    strip = roadb_sheet.detail_strip(crops, flagged, max_patients=2)
    assert strip["native"] is True
    assert strip["shown"]["roadB_p2_768_square_g1"] == {"n_flagged": 3, "n_shown": 2}
    assert strip["truncated"]["roadB_p2_768_square_g1"] == 1


def test_the_detail_strip_is_empty_when_nothing_upsamples():
    """At 224 nothing does, and a row there would be an empty claim."""
    from cleft import roadb_sheet

    strip = roadb_sheet.detail_strip(_crops(), {"roadB_p2_512_square_g1": []})
    assert strip["sheet"] is None
    assert "nothing to inspect" in strip["why_empty"]


def test_the_review_questions_cover_both_sheets():
    from cleft import roadb_sheet

    joined = " ".join(roadb_sheet.REVIEW_QUESTIONS)
    assert "AXIS SHEET" in joined and "DETAIL STRIP" in joined
    assert "pad present at square" in joined
    assert "corners present at G1" in joined


# --------------------------------------------------------------------------
# eight artifacts, and why not one
# --------------------------------------------------------------------------


def test_each_setting_gets_its_own_artifact():
    """**Guard 3 verifies the ROLLUP**, so one artifact would couple all
    eight: re-running any cell breaks guard 3 for every config declaring it,
    including configs whose own setting never changed."""
    settings = roadb.staging_settings()
    artifacts = [s["artifact"] for s in settings]
    assert len(set(artifacts)) == len(settings) == 10
    for setting in settings:
        assert setting["artifact"].startswith("roadb_")
        assert setting["artifact"].endswith("_v1")
        # The artifact name must carry all three axes, or two cells collide.
        for axis in (str(setting["resolution"]), setting["aspect"],
                     setting["geometry"]):
            assert axis in setting["artifact"], (setting["artifact"], axis)


def test_the_reason_for_eight_artifacts_is_recorded_with_its_contrast():
    """A future session will see masked SCUT bundled and Road B split, and
    should know it was a decision rather than drift."""
    record = roadb.ARTIFACT_PER_SETTING
    assert "guard 3" in record["reason"]
    assert "never changed" in record["reason"]
    assert "consumed selectively" in record["differs_from_masked_scut_because"]
    assert "always built AND consumed together" in (
        record["differs_from_masked_scut_because"]
    )
    # The gate result travels with the setting, not in a shared summary.
    assert "shared summary" in record["per_artifact_manifest"]


def test_the_declare_script_would_find_road_b_configs():
    """**[MEASURED 2026-08-08] It would not have.** ``roadb_*`` does not match
    ``p7*``, so the tool would have reported "nothing to paste" over eight
    unresolved artifacts -- the identical gap it was widened for once already.
    """
    import fnmatch
    import importlib.util
    import sys as _sys
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "_declare",
        _Path(roadb.__file__).parents[2] / "scripts" / "declare_ladder_inputs.py",
    )
    module = importlib.util.module_from_spec(spec)
    _sys.modules["_declare"] = module
    spec.loader.exec_module(module)

    globs = module.DISCOVERY_GLOBS
    assert isinstance(globs, tuple) and len(globs) > 1, (
        "a single prefix is the wrong shape; this has now been widened twice"
    )
    for setting in roadb.staging_settings():
        name = f"{setting['name'].lower()}.yaml"
        assert any(fnmatch.fnmatch(name, pattern) for pattern in globs), name
    # And the Phase 7 family is still covered.
    assert any(fnmatch.fnmatch("p7c_paired_selected30.yaml", p) for p in globs)


# --------------------------------------------------------------------------
# the gate: separate, per family, and readable for staleness
# --------------------------------------------------------------------------


def test_road_a_224_is_recorded_with_provenance_and_reproduces():
    """**One source, and the gate reads it from here.** Recomputing 224 in
    every Road B run would stage Road A's artifact eight times for a number
    that already exists -- so it is recorded, and a test recomputes it."""
    from cleft import roadb_staging

    record = roadb.ROAD_A_224_PIXEL_RESIDUAL
    assert record["provenance"].startswith("[MEASURED]")
    assert "not an assumption" in record["provenance"]
    assert roadb_staging.pixel_asymmetry_residual(record["size"]) == (
        pytest.approx(record["residual"], abs=5e-5)
    )


def test_the_residual_tracks_column_count_not_nominal_size():
    """**The measured reason the ordering is within a family.** Non-square 512
    has ~379 columns, so its residual is larger than square 512's -- fewer
    columns, not a defect."""
    from cleft import roadb_staging

    for columns, expected in roadb.RESIDUAL_FAMILIES["by_columns"].items():
        assert roadb_staging.pixel_asymmetry_residual(columns) == (
            pytest.approx(expected, abs=5e-5)
        ), columns
    by_columns = roadb.RESIDUAL_FAMILIES["by_columns"]
    assert by_columns[379] > by_columns[512], (
        "if non-square no longer had a larger residual the within-family rule "
        "would need re-reading"
    )


def test_families_group_by_aspect_and_geometry_with_road_a_in_the_square_ones():
    from cleft import roadb_staging

    observed = {s["name"]: 1e-3 for s in roadb.staging_settings()}
    families = roadb_staging.residual_families(observed)
    assert sorted(families) == [
        "nonsquare/g1", "nonsquare/g2", "square/g1", "square/g2",
    ]
    # [UPDATED] Every family now has a 224 point, but from different sources:
    # the square families get Road A's recorded residual, the non-square ones
    # get it from their OWN staging settings.
    for name, residuals in families.items():
        assert 224 in residuals, name
    assert families["square/g1"][224] == roadb.ROAD_A_224_PIXEL_RESIDUAL["residual"]
    assert families["nonsquare/g1"][224] != (
        roadb.ROAD_A_224_PIXEL_RESIDUAL["residual"]
    ), "the non-square 224 must come from its own run, not Road A's square one"


def test_every_family_has_three_points_after_the_224_non_square_cells():
    """Road A has no non-square 224, so the three-point trend argument applies
    to two families and not four. A limitation of the design, recorded."""
    from cleft import roadb_staging

    observed = {
        s["name"]: {224: 3e-3, 512: 2e-3, 768: 1e-3}[s["resolution"]]
        for s in roadb.staging_settings()
    }
    report = roadb_staging.assert_residual_monotone_by_family(observed)
    # [UPDATED] 224 non-square was added, so ALL FOUR families now have three
    # points and §5's trend argument covers the G2 non-square cell.
    assert report["three_point_families"] == [
        "nonsquare/g1", "nonsquare/g2", "square/g1", "square/g2",
    ]
    assert report["n_families"] == 4


def test_a_family_that_rises_fails_and_the_others_are_still_reported():
    from cleft import roadb_staging

    observed = {}
    for setting in roadb.staging_settings():
        good = {224: 3e-3, 512: 2e-3, 768: 1e-3}[setting["resolution"]]
        # One family gets worse where it should get better.
        if (setting["aspect"], setting["geometry"]) == ("nonsquare", "g2"):
            good = {224: 1e-3, 512: 2e-3, 768: 3e-3}[setting["resolution"]]
        observed[setting["name"]] = good
    with pytest.raises(roadb_staging.StagingError, match="nonsquare/g2"):
        roadb_staging.assert_residual_monotone_by_family(observed)


def test_a_pooled_ordering_would_have_called_non_square_a_defect():
    """The reason the gate is per family, as arithmetic: pooled, the real
    column counts put non-square 512 above square 512."""
    from cleft import roadb_staging

    pooled = {
        512: roadb_staging.pixel_asymmetry_residual(379),   # non-square 512
        568: roadb_staging.pixel_asymmetry_residual(512),   # square 512
    }
    assert pooled[512] > pooled[568], "the confusion this rule prevents"


def test_a_setting_that_is_not_road_bs_is_refused():
    from cleft import roadb_staging

    with pytest.raises(roadb_staging.StagingError, match="not Road B"):
        roadb_staging.residual_families({"p7_d1_vit_b16_imagenet_g1": 1e-3})


def test_224_non_square_is_two_cells_not_one():
    """**Both non-square families need a third point**, and (nonsquare, g1)
    and (nonsquare, g2) are distinct settings."""
    record = roadb.TWO_TWENTY_FOUR_NON_SQUARE
    assert len(record["added"]) == 2
    assert record["totals"]["settings"] == 10
    names = {s["name"] for s in roadb.staging_settings()}
    assert set(record["added"]) <= names


def test_224_square_is_not_restaged_because_road_a_has_it():
    """Road A's staged_v1 is the control. Re-staging it would produce a second
    artifact of the same pixels with a different hash."""
    names = {s["name"] for s in roadb.staging_settings()}
    assert "roadB_p2_224_square_g1" not in names
    assert "roadB_p2_224_square_g2" not in names
    assert "staged_v1" in roadb.TWO_TWENTY_FOUR_NON_SQUARE["224_square_is_not_restaged"]


def test_the_aspect_axis_does_not_change_content_resolution():
    """**stage pads THEN resizes**, so at a given target the content occupies
    the same pixel box either way. That is what makes 224 non-square the
    cleanest test of the pad claim rather than merely a third point."""
    record = roadb.TWO_TWENTY_FOUR_NON_SQUARE
    assert "does not change content resolution" in record["why_meaningful"]
    assert "confounds pad removal with resolution" in record["cleanest_pad_test"]
    import inspect

    from cleft.geometry import staging

    assert "Pad to square" in inspect.getdoc(staging.stage)


# --------------------------------------------------------------------------
# the wiring
# --------------------------------------------------------------------------


def test_all_three_tasks_are_registered():
    from cleft.run import TASKS

    assert TASKS["roadb_stage"].__name__ == "stage"
    assert TASKS["roadb_residual_gate"].__name__ == "residual_gate"
    assert TASKS["roadb_sheets"].__name__ == "sheets"


def test_there_is_one_config_per_setting_plus_the_gate_and_sheets(repo_root):
    # Scoped to roadb_p2_* -- Phase 3 ships its own roadb_p3_* configs.
    shipped = {p.stem for p in (repo_root / "configs").glob("roadb_p2_*.yaml")}
    expected = {s["name"].lower() for s in roadb.staging_settings()}
    expected |= {"roadb_p2_residual_gate", "roadb_p2_sheets"}
    assert shipped == expected
    assert len(shipped) == 12


def test_the_sheets_config_declares_the_same_artifacts_as_the_gate(repo_root):
    """The sheets and the verdict must describe the SAME bytes, or the review
    blesses artifacts the gate never judged. Same names, same hashes."""
    import yaml

    gate = yaml.safe_load(
        (repo_root / "configs" / "roadb_p2_residual_gate.yaml")
        .read_text(encoding="utf-8")
    )
    sheets = yaml.safe_load(
        (repo_root / "configs" / "roadb_p2_sheets.yaml")
        .read_text(encoding="utf-8")
    )
    assert sheets["task"]["kind"] == "roadb_sheets"
    assert {e["name"] for e in sheets["inputs"]} == {
        s["artifact"] for s in roadb.staging_settings()
    }
    gate_hashes = {e["name"]: e["rollup_sha256"] for e in gate["inputs"]}
    for entry in sheets["inputs"]:
        assert entry["rollup_sha256"] == gate_hashes[entry["name"]], entry["name"]


def test_each_staging_config_names_its_own_artifact(repo_root):
    """**out_version must equal the setting's artifact** or a partial re-run
    moves the wrong hash."""
    import yaml

    for setting in roadb.staging_settings():
        payload = yaml.safe_load(
            (repo_root / "configs" / f"{setting['name'].lower()}.yaml")
            .read_text(encoding="utf-8")
        )
        assert payload["task"]["setting"] == setting["name"]
        assert payload["task"]["out_version"] == setting["artifact"]


def test_the_gate_declares_every_artifact_so_guard_3_hashes_them(repo_root):
    """It reads artifacts it does not own; declaring them is what makes the
    rollups available to record."""
    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "roadb_p2_residual_gate.yaml")
        .read_text(encoding="utf-8")
    )
    declared = {entry["name"] for entry in payload["inputs"]}
    assert declared == {s["artifact"] for s in roadb.staging_settings()}
    assert len(declared) == 10


def test_the_gate_records_the_rollups_it_read():
    """**A re-run after the verdict leaves the verdict silently wrong**, so
    the summary carries what it described -- the same reason run directories
    carry input hashes.

    **[CORRECTED 2026-08-09] This test asserted ``rollup_sha256`` was in the
    source -- it pinned the defect.** That is the CONFIG's field name; at
    runtime ``_verify_inputs`` supplies ``rollup``, and the gate died on the
    cluster reading the config key from the runtime entries. A check agreeing
    with the code it guards is the tally's oldest shape. It now asserts the
    read is of the key the context produces, and the runtime test above
    drives the gate on real-shaped entries.
    """
    import inspect

    from cleft import roadb_tasks

    source = inspect.getsource(roadb_tasks.residual_gate)
    assert '"artifacts_read": rollups' in source
    assert 'rollups[name] = entry["rollup"]' in source
    # The docstring may NAME the old key while recording the correction;
    # what must never return is the assignment reading it.
    assert 'rollups[name] = entry["rollup_sha256"]' not in source, (
        "the config field name is back; it does not exist on ctx.inputs"
    )
    assert "makes this verdict stale" in source


def test_the_clean_pad_pair_is_recorded_on_the_224_cells_only():
    """It is a Phase 4 or 7 arm, not a staging output -- the artifact records
    that it exists so a later phase does not rediscover it."""
    from cleft import roadb_tasks

    for setting in roadb.staging_settings():
        pair = roadb_tasks.clean_pad_pair(setting)
        if setting["resolution"] == roadb.CONTROL_RESOLUTION:
            assert pair["against"].startswith("staged_v1")
            assert "constant content resolution" in pair["isolates"]
            assert "Phase 4 or 7" in pair["not_a_staging_output"]
        else:
            assert pair is None, setting["name"]


def test_the_frozen_code_new_regime_note_is_recorded():
    """"Reuses frozen code" invites the wrong reading: stage has only ever run
    at 224, and resize_nearest at 512 is where the residual lives."""
    from cleft import roadb_staging

    record = roadb_staging.FROZEN_CODE_NEW_REGIME
    assert "only ever run at 224" in record["not_free_of_risk"]
    assert "residual gate" in record["what_catches_it"]


def test_non_square_keeps_the_same_content_pixels_as_square():
    """**Measured**: stage pads THEN resizes, so the content box is identical
    at a given target. That is what makes the aspect axis orthogonal to
    resolution."""
    from cleft import roadb_staging
    from cleft.geometry import staging

    image = np.zeros((300, 200, 3), dtype=np.uint8)
    image[50:250, 40:160] = 200
    square = staging.stage(image, size=224)
    non_square = roadb_staging.stage_non_square(image, 224)
    # **[UPDATED 2026-08-09] Equal WITHIN the 16-rounding**, not exactly.
    # SHAPE_IS_FIXED_AT_STAGING rounds each dimension to a multiple of the
    # patch size because ViT refuses anything else, and that rounding is
    # itself an anisotropic distortion of up to 8px. Asserting exact equality
    # would now be asserting the rounding does not happen.
    for observed, expected in zip(non_square.content_box[2:],
                                  square.content_box[2:]):
        assert abs(observed - expected) <= roadb_staging.PATCH // 2
    assert non_square.image.shape[0] % roadb_staging.PATCH == 0
    assert non_square.image.shape[1] % roadb_staging.PATCH == 0
    assert square.pad_fraction > 0.0
    assert non_square.pad_fraction == 0.0


def test_the_source_path_is_resolved_by_road_as_own_finder(tmp_path):
    """**[CORRECTED 2026-08-09] It built ``<root>/<frontal_id>.jpg`` and the
    layout is ``<root>/<patient_id>/<frontal_id>.jpg``.**

    Fixed by delegating rather than repairing: a fourth path builder would
    have had to rediscover the AppleDouble and Thumbs.db exclusions, folder
    143's 524-not-523 exception and folder 238's single image, all of which
    ``contact.find_image`` already carries.
    """
    import inspect

    from cleft import roadb_tasks

    source = inspect.getsource(roadb_tasks.source_image)
    assert "find_image" in source
    assert "with_suffix" not in source, "a private path builder is back"

    # And it really walks the patient folder, including past a Thumbs.db.
    root = tmp_path / "photos"
    (root / "1").mkdir(parents=True)
    from PIL import Image

    Image.new("RGB", (8, 8)).save(root / "1" / "241.jpg")
    (root / "1" / "Thumbs.db").write_bytes(b"junk")
    (root / "1" / "._241.jpg").write_bytes(b"sidecar")
    image = roadb_tasks.source_image(root, {"patient_id": 1, "frontal_id": 241})
    assert image.shape[:2] == (8, 8)


def test_the_finder_ignores_sidecars_and_thumbs():
    """Excluded by basename, not extension -- ``._241.jpg`` is a .jpg."""
    import inspect

    from cleft.geometry import contact

    source = inspect.getsource(contact.find_image)
    assert 'startswith("._")' in source


def test_non_square_is_vit_only_and_the_reason_is_measured():
    """**Swin asserts equality with 224x224, so no rounding reaches it**, and
    the CNNs accept everything while silently changing the ROI grid."""
    constraints = roadb.BACKBONE_SHAPE_CONSTRAINTS
    assert "exactly" in constraints["swin_b"]
    assert "7x5" in constraints["cnns_are_the_dangerous_case"]

    decision = roadb.NON_SQUARE_IS_VIT_ONLY
    assert decision["non_square_cells"] == "ViT-B/16 only"
    assert decision["square_cells"].startswith("four-backbone")
    assert "different arm" in decision["rejected"]["swin_dynamic_variant"]
    assert "dangerous" in decision["rejected"]["let_the_cnns_run"]
    # Both costs named, and the pad pair's caveat is explicit.
    assert len(decision["costs"]) == 2
    assert "NOT one factor apart in general" in " ".join(decision["costs"])


def test_the_supervisors_intent_is_recorded_as_open_not_decided():
    """No backbone consumes variable shapes, so per-patient proportions are
    unreachable. The choice between approximations goes back with the
    measurement rather than being made here."""
    record = roadb.ORIGINAL_PROPORTIONS_ARE_NOT_REACHABLE
    assert record["status"].startswith("goes back to supervision")
    assert "cannot be batched" in record["why_unreachable"]
    assert set(record["approximations"]) == {"round_to_16", "one_common_shape"}


# --------------------------------------------------------------------------
# the stage task, end to end -- the lines only a run used to reach
# --------------------------------------------------------------------------
#
# **[2026-08-09] None of these existed, and that is why four defects sat in a
# task the suite called green.** stage(ctx) had no runtime test, so np.stack
# on unstackable non-square shapes, a G2 setting whose pixels were G1's, a
# content fraction measured on a square frame that does not ship, and a
# residual recorded at the wrong column count were all reachable only by a
# cluster run -- the exact class R10 exists for, except every one of them is
# testable on a laptop with a three-patient fixture.


class _StageCtx:
    """The slice of RunContext the tasks consume: config, inputs, repo_root,
    path, log, atomic. The real one carries guards this fixture does not
    exercise -- guard 3 is tested where it lives."""

    def __init__(self, root, task, inputs):
        self.repo_root = root / "repo"
        self.run_dir = root / "run"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = {"task": task}
        self.inputs = inputs
        self.logged = []

    def log(self, message):
        self.logged.append(message)

    def path(self, name, tier):
        return self.run_dir / name

    def atomic(self, name, tier):
        import contextlib

        @contextlib.contextmanager
        def _claim():
            yield self.run_dir / name

        return _claim()


def _sources_fixture(root, sources):
    """A schema-valid manifest plus the real folder layout.

    ``sources`` is ``{patient_id: (width, height)}``. The layout is
    ``<root>/<patient_id>/<frontal_id>.jpg`` because ``source_image`` resolves
    through ``contact.find_image`` -- the resolver R10's fifth instance was
    about.
    """
    from PIL import Image

    manifest_dir = root / "cleft_v1"
    folders = root / "photos"
    manifest_dir.mkdir(parents=True)
    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    rng = np.random.default_rng(7)
    for index, (patient_id, (width, height)) in enumerate(sorted(sources.items())):
        frontal = 1000 + patient_id
        lines.append(
            f"{patient_id},{frontal},,2.5,2.5,2.5,2.5,2.5,"
            f"0.2,0.2,0.2,0.2,0.2,{index % 3},{index % 5}"
        )
        folder = folders / str(patient_id)
        folder.mkdir(parents=True)
        # Structured content, not a flat fill: unwarp of a constant image is
        # the identity, and a fixture that cannot show the G2 difference
        # would pass a task that never unwarps.
        array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        Image.fromarray(array).save(folder / f"{frontal}.jpg")
    (manifest_dir / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return manifest_dir, folders


def _run_stage(tmp_path, setting_name, sources):
    import json

    from cleft import roadb_tasks

    setting = {s["name"]: s for s in roadb.staging_settings()}[setting_name]
    root = tmp_path / setting_name
    manifest_dir, folders = _sources_fixture(root, sources)
    ctx = _StageCtx(
        root,
        task={
            "kind": "roadb_stage",
            "setting": setting_name,
            "manifest_artifact": "manifest_v1",
            "patient_folders": "patient_folders",
            "out_version": setting["artifact"],
        },
        inputs=[
            {"name": "manifest_v1", "path": str(manifest_dir)},
            {"name": "patient_folders", "path": str(folders)},
        ],
    )
    roadb_tasks.stage(ctx)
    artifact = ctx.repo_root / "data" / "staged" / setting["artifact"]
    manifest = json.loads((artifact / "MANIFEST.json").read_text(encoding="utf-8"))
    return artifact, manifest


#: Three aspects straddling 1.0, so the non-square path exercises both the
#: taller-than-wide and wider-than-full branches and the widths differ.
_TRIO = {1: (300, 405), 2: (260, 240), 3: (270, 300)}


def test_the_stage_task_unwarps_g2_rather_than_relabelling_g1(tmp_path):
    """**A G2 artifact whose pixels are G1's makes five settings aliases of
    the other five.** Road A bakes G2 in at staging -- stage_build: stage,
    then unwarp the padded square -- and Road B's task must be the same
    composition at a new size."""
    from cleft import roadb_tasks
    from cleft.geometry import staging
    from cleft.geometry.trapezium import unwarp

    pair = {1: (300, 405), 2: (260, 240)}
    g1_dir, _ = _run_stage(tmp_path, "roadB_p2_512_square_g1", pair)
    g2_dir, g2_manifest = _run_stage(tmp_path, "roadB_p2_512_square_g2", pair)

    g1 = np.load(g1_dir / "staged_patient_g1.npy")
    g2 = np.load(g2_dir / "staged_patient_g2.npy")
    # Square storage is Road A's layout unchanged: stacked, row order =
    # manifest order, so frozen readers consume it at a new size.
    assert g1.shape == (2, 512, 512, 3)
    assert g2.shape == (2, 512, 512, 3)
    assert not np.array_equal(g1, g2), "g2 pixels are g1's relabelled"

    # Bit-for-bit the frozen composition, same claim masked SCUT makes.
    source = roadb_tasks.source_image(
        tmp_path / "roadB_p2_512_square_g2" / "photos",
        {"patient_id": 1, "frontal_id": 1001},
    )
    expected = unwarp(staging.stage(source, size=512).image)
    assert np.array_equal(g2[0], expected)
    assert "unwarp" in (g2_manifest["g2_pixels"] or "")


def test_the_stage_task_stores_non_square_per_patient(tmp_path):
    """**The decided storage, implemented:** non-square shapes differ per
    patient and cannot stack (SHAPE_IS_FIXED_AT_STAGING), so a stacked file
    is impossible and per-patient files are the artifact."""
    artifact, manifest = _run_stage(tmp_path, "roadB_p2_224_nonsquare_g1", _TRIO)

    assert not (artifact / "staged_patient_g1.npy").exists()
    per_patient = sorted(artifact.glob("staged_patient_g1_p*.npy"))
    assert [p.name for p in per_patient] == [
        "staged_patient_g1_p001.npy",
        "staged_patient_g1_p002.npy",
        "staged_patient_g1_p003.npy",
    ]
    shapes = [np.load(p).shape for p in per_patient]
    assert len(set(shapes)) > 1, "equal shapes would not exercise the reason"
    for shape in shapes:
        assert shape[0] % 16 == 0 and shape[1] % 16 == 0, shape
    assert "per patient" in manifest["storage"]
    assert "cannot stack" in manifest["storage"]
    # The artifact says which side its flags measure, so a v1 manifest
    # (shorter-side flags) and a v2 one cannot be read as the same quantity.
    assert "longer side" in manifest["upsampling"]["measures"]


def test_the_non_square_residual_is_at_its_own_column_count(tmp_path):
    """**The residual tracks column count** (RESIDUAL_FAMILIES), so a
    non-square artifact recording the square-frame residual would hand the
    family gate the wrong quantity -- and its 224 point would collide with
    Road A's square 3.29e-03, which the family test says must differ."""
    from cleft import roadb_staging

    _, manifest = _run_stage(tmp_path, "roadB_p2_224_nonsquare_g1", _TRIO)

    # Widths for the trio at 224, after the 16-rounding: 160, 224, 208.
    assert manifest["residual_columns"] == 208
    assert manifest["pixel_residual"] == pytest.approx(
        roadb_staging.pixel_asymmetry_residual(208), abs=1e-12
    )
    assert manifest["pixel_residual"] != pytest.approx(
        roadb.ROAD_A_224_PIXEL_RESIDUAL["residual"], abs=1e-5
    ), "the non-square 224 residual must come from its own columns"


def test_the_content_fraction_describes_the_frame_that_ships(tmp_path):
    """**[CORRECTED 2026-08-09] It was computed on a square canvas the
    non-square artifact does not contain**, diluting the trapezium fraction
    by pad columns the setting exists to remove -- the wrong-region kind of
    wrong that phase8's framing finding came from."""
    _, g1 = _run_stage(tmp_path, "roadB_p2_224_nonsquare_g1", _TRIO)
    _, g2 = _run_stage(tmp_path, "roadB_p2_224_nonsquare_g2", _TRIO)

    expected_g1 = 1.0 - roadb.NON_CONTENT_FRACTION["g1_non_square"]
    assert g1["content_fraction"] == pytest.approx(expected_g1, abs=0.02)
    assert g2["content_fraction"] == pytest.approx(1.0, abs=1e-9)
    low, high = g1["content_fraction_range"]
    assert low <= g1["content_fraction"] <= high


def _real_input_entry(name, path, rollup):
    """A ctx.inputs entry with the shape ``_verify_inputs`` ACTUALLY produces.

    The keys are parsed from the producer's source, so this fixture cannot
    invent a shape and then pass against it -- which is precisely how the
    gate's ``rollup_sha256`` read would have survived a hand-built fixture.
    """
    import ast
    import inspect

    from cleft.provenance import context

    produced = None
    for node in ast.walk(ast.parse(inspect.getsource(context))):
        if isinstance(node, ast.FunctionDef) and node.name == "_verify_inputs":
            for inner in ast.walk(node):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr == "append"
                        and inner.args and isinstance(inner.args[0], ast.Dict)):
                    produced = {k.value for k in inner.args[0].keys}
    assert produced, "_verify_inputs no longer appends a dict literal"

    entry = {
        "name": name,
        "path": str(path),
        "declared_rollup": rollup,
        "rollup": rollup,
        "file_count": 3,
        "total_bytes": 999,
    }
    assert set(entry) == produced, (
        f"this fixture drifted from _verify_inputs: {sorted(produced)}"
    )
    return entry


def test_the_residual_gate_runs_on_real_shaped_input_entries(tmp_path):
    """**[MEASURED 2026-08-09] The gate died on the cluster with
    KeyError: 'rollup_sha256'** -- it read the CONFIG's field name from the
    runtime entries, where ``_verify_inputs`` puts the verified hash under
    ``rollup``. Guard 3 had just verified all ten artifacts; the run failed
    reading back the hashes it had checked. This drives the gate end to end
    on entries with the producer's shape, which no hand-shaped fixture can
    do honestly."""
    import json

    from cleft import roadb_tasks

    residual_by_resolution = {
        "square": {512: 1.401e-03, 768: 9.136e-04},
        "nonsquare": {224: 3.066e-03, 512: 1.892e-03, 768: 1.240e-03},
    }
    inputs, rollups = [], {}
    for index, setting in enumerate(roadb.staging_settings()):
        artifact = tmp_path / setting["artifact"]
        artifact.mkdir()
        residual = residual_by_resolution[setting["aspect"]][setting["resolution"]]
        (artifact / "MANIFEST.json").write_text(
            json.dumps({"pixel_residual": residual}), encoding="utf-8"
        )
        rollup = f"{index:064x}"
        rollups[setting["artifact"]] = rollup
        inputs.append(_real_input_entry(setting["artifact"], artifact, rollup))

    ctx = _StageCtx(tmp_path, task={"kind": "roadb_residual_gate"}, inputs=inputs)
    roadb_tasks.residual_gate(ctx)

    summary = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    # The rollups it read are recorded, so a later reader can tell whether
    # the verdict still describes what is on disk.
    assert summary["artifacts_read"] == rollups
    assert summary["n_families"] == 4
    assert sorted(summary["three_point_families"]) == [
        "nonsquare/g1", "nonsquare/g2", "square/g1", "square/g2",
    ]
    for family in summary["families"].values():
        assert family["monotone"] is True
    # A verdict reader meets four pairs of identical g1/g2 residuals; the
    # reason travels with the verdict rather than being rediscovered.
    assert "no staged pixel" in summary["residual_geometry_note"]["why"]


# --------------------------------------------------------------------------
# the sheets task -- the last piece before the review, and the review gates
# Phase 3
# --------------------------------------------------------------------------


def _fabricated_artifacts(root, upsampled_by_resolution):
    """The ten artifacts in the REAL layout, tiny.

    ``upsampled_by_resolution`` is ``{resolution: [patient ids flagged]}``.
    Square settings store one stacked array; non-square store per patient
    with varying widths, so the sheet task walks both storage layouts.
    """
    import json

    from cleft import roadb_tasks

    patients = [1, 2, 3, 4, 5]
    rng = np.random.default_rng(11)
    inputs = []
    for index, setting in enumerate(roadb.staging_settings()):
        artifact = root / setting["artifact"]
        artifact.mkdir(parents=True)
        geometry = setting["geometry"]
        rows, images = [], {}
        for position, patient in enumerate(patients):
            if setting["aspect"] == "square":
                width = 14 + 2 * position
                box = ((24 - width) // 2, 0, width, 24)
                images[patient] = rng.integers(0, 256, (24, 24, 3), dtype=np.uint8)
                pad = 1.0 - (width * 24) / (24 * 24)
            else:
                width = 14 + 2 * position
                box = (0, 0, width, 24)
                images[patient] = rng.integers(0, 256, (24, width, 3), dtype=np.uint8)
                pad = 0.0
            rows.append({
                "patient_id": patient, "frontal_id": 1000 + patient,
                "content_x": box[0], "content_y": box[1],
                "content_w": box[2], "content_h": box[3],
                "pad_fraction": round(pad, 6),
            })
        if setting["aspect"] == "square":
            np.save(
                artifact / f"staged_patient_{geometry}.npy",
                np.stack([images[p] for p in patients]),
            )
        else:
            for patient in patients:
                np.save(
                    artifact / f"staged_patient_{geometry}_p{patient:03d}.npy",
                    images[patient],
                )
        roadb_tasks.write_geometry_csv(artifact / "geometry.csv", rows)
        flags = [
            patient in upsampled_by_resolution.get(setting["resolution"], [])
            for patient in patients
        ]
        (artifact / "MANIFEST.json").write_text(
            json.dumps({
                "pixel_residual": 1e-3,
                "upsampling": {
                    "target": setting["resolution"], "n": len(patients),
                    "n_upsampled": sum(flags), "per_patient": flags,
                },
            }),
            encoding="utf-8",
        )
        inputs.append(
            _real_input_entry(setting["artifact"], artifact, f"{index:064x}")
        )
    return inputs


def test_the_sheets_task_renders_both_sheets_from_the_artifacts(tmp_path):
    """**Built and tested was not wired: nothing called roadb_sheet.** This
    drives the task end to end over both storage layouts -- the review it
    feeds gates Phase 3."""
    import json

    from cleft import roadb_tasks

    inputs = _fabricated_artifacts(tmp_path, {512: [3], 768: [2, 5]})
    ctx = _StageCtx(
        tmp_path,
        task={"kind": "roadb_sheets", "max_patients": 41, "columns": 6},
        inputs=inputs,
    )
    roadb_tasks.sheets(ctx)

    assert (ctx.run_dir / "contact_sheet_axis.png").stat().st_size > 0
    assert (ctx.run_dir / "contact_sheet_detail.png").stat().st_size > 0

    metrics = json.loads((ctx.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["axis"]["columns"]) == 10
    assert metrics["axis"]["n_patients"] == 4
    # The flagged counts come from the artifacts' own manifests.
    counts = metrics["flagged_counts"]
    assert all(counts[s["name"]] == 2 for s in roadb.staging_settings()
               if s["resolution"] == 768)
    assert all(counts[s["name"]] == 1 for s in roadb.staging_settings()
               if s["resolution"] == 512)
    assert all(counts[s["name"]] == 0 for s in roadb.staging_settings()
               if s["resolution"] == 224)
    assert metrics["detail"]["truncated"] == {}, "41 covers every flagged patient"
    # It reads artifacts it does not own, so the rollups are recorded --
    # the same staleness discipline as the gate.
    assert metrics["artifacts_read"] == {
        entry["name"]: entry["rollup"] for entry in inputs
    }
    assert list(metrics["review_questions"])
    # These fixture manifests are v1-shaped (no "measures" key), so the
    # metrics must say the strip was drawn from the deprecated shorter-side
    # flags and carry the corrected counts alongside.
    assert all(
        "shorter side" in value for value in metrics["flags_measure"].values()
    )
    # JSON keys are strings after the round trip.
    assert metrics["corrected_interpolation_counts"]["by_target"] == {
        "224": 0, "512": 1, "768": 21,
    }

    # Patient ids are patient-keyed, so they live in the CLUSTER-ONLY file
    # and not in the SHAREABLE metrics.
    selection = json.loads(
        (ctx.run_dir / "sheets_selection.json").read_text(encoding="utf-8")
    )
    assert len(selection["axis_patients"]) == 4
    assert selection["flagged"]["roadB_p2_768_square_g1"] == [2, 5]
    assert "axis_patients" not in metrics


def test_the_axis_patients_span_the_measured_aspect_range(tmp_path):
    """Framing defects live at the aspect extremes -- the widest and
    narrowest crops are where the pad and the corners differ most -- so the
    selection is measured from geometry.csv, extremes included, rather than
    hand-picked."""
    import json

    from cleft import roadb_tasks

    inputs = _fabricated_artifacts(tmp_path, {})
    ctx = _StageCtx(tmp_path, task={"kind": "roadb_sheets"}, inputs=inputs)
    roadb_tasks.sheets(ctx)

    selection = json.loads(
        (ctx.run_dir / "sheets_selection.json").read_text(encoding="utf-8")
    )
    # Fixture aspects rise with patient id (widths 14..22 at height 24), so
    # the extremes are patients 1 and 5 and the middle two are spread.
    chosen = selection["axis_patients"]
    assert chosen[0] == 1 and chosen[-1] == 5
    assert len(chosen) == len(set(chosen)) == 4


def test_a_missing_setting_fails_the_sheets_task_loudly(tmp_path):
    """A sheet silently missing a column would review nine settings as ten."""
    import pytest as _pytest

    from cleft import roadb_tasks

    inputs = _fabricated_artifacts(tmp_path, {})
    short = [e for e in inputs if "768_nonsquare_g2" not in e["name"]]
    ctx = _StageCtx(tmp_path, task={"kind": "roadb_sheets"}, inputs=short)
    with _pytest.raises(ValueError, match="not declared"):
        roadb_tasks.sheets(ctx)


def test_the_backbone_input_routing_is_pinned_without_torch():
    """**[MEASURED 2026-08-09, system torch] ``create_backbone`` built ViT at
    a fixed 224** -- the 768 seed band would have died on the cluster at
    "Input height (768) doesn't match model (224)": R10's sixth instance,
    prevented by checking first. The routing is a pure function so the suite
    pins it without torch."""
    from cleft.models.factory import FactoryError, timm_kwargs_for

    assert timm_kwargs_for("vit_base_patch16_224", None) == {}
    assert timm_kwargs_for("vit_base_patch16_224", (224, 224)) == {}, (
        "the 224 path must stay byte-identical to the one that produced "
        "every Road A number"
    )
    for size in ((512, 512), (768, 768), (224, 160), (208, 224)):
        assert timm_kwargs_for("vit_base_patch16_224", size) == {
            "dynamic_img_size": True
        }, size
    with pytest.raises(FactoryError, match="divisible by 16"):
        timm_kwargs_for("vit_base_patch16_224", (224, 150))
    # Swin at SQUARE non-224 routes to img_size -- unlocked by the pinned
    # gate run (SWIN_AT_RESOLUTION.pinned_confirmation). Non-square swin
    # stays refused: ViT-only.
    assert timm_kwargs_for("swin_base_patch4_window7_224", (768, 768)) == {
        "img_size": 768
    }
    assert timm_kwargs_for("swin_base_patch4_window7_224", (512, 512)) == {
        "img_size": 512
    }
    with pytest.raises(FactoryError, match="ViT-only"):
        timm_kwargs_for("swin_base_patch4_window7_224", (224, 160))
    # The CNNs accept any size for whole-image embeddings.
    assert timm_kwargs_for("xception", (768, 768)) == {}


def test_the_band_design_is_recorded_with_its_preregistered_rule():
    """**The decision rule is registered before either band exists** -- a
    window chosen after seeing the result is a forking path (the 7C
    lesson)."""
    record = roadb.SEED_BAND_DESIGN
    assert record["hypothesis"] == (
        "three bands, one per resolution, measured at square G1"
    )
    # The stakes reframed by §4.12.1: bands plan, they never denominate.
    assert "OWN seed SD" in record["not_the_denominator"]
    assert "cannot corrupt a claim" in record["not_the_denominator"]
    # Ten seeds see ~2x; forty see ~1.4x. Stated before anyone reads an
    # agreement off two ten-seed numbers.
    assert "~2x" in record["sensitivity"]
    assert "~1.4x" in record["sensitivity"]
    assert "F" in record["decision_rule_preregistered"]
    assert "LARGER" in record["decision_rule_preregistered"]
    assert "before either number exists" in record["decision_rule_preregistered"]
    # Q1 stays open and is answered by measurement, not argument.
    assert record["square_g1_representativeness"].startswith("OPEN")


def test_the_band_configs_pair_with_road_as_procedure(repo_root):
    """**The 768 and 512 sweeps run Road A's own ten seeds** so per-seed
    values pair across the whole resolution series; **the 224 replicate runs
    a disjoint ten on Road A's own artifact** -- Road A's seeds there would
    reproduce 0.0137 bit-for-bit (determinism) and measure nothing. Every
    other field matches p3_train_cv, so the procedures differ only in the
    staged artifact and the seed draw."""
    import yaml

    def load(stem):
        return yaml.safe_load(
            (repo_root / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )

    p3 = load("p3_train_cv")
    band_768 = load("roadb_p3_seed_band_768_square_g1")
    band_512 = load("roadb_p3_seed_band_512_square_g1")
    band_224 = load("roadb_p3_seed_band_224_square_g1")

    road_a_seeds = p3["task"]["seeds"]
    assert band_768["task"]["seeds"] == road_a_seeds
    assert band_512["task"]["seeds"] == road_a_seeds, (
        "512 must pair per-seed with both extremes"
    )
    assert len(band_224["task"]["seeds"]) == 10
    assert not set(band_224["task"]["seeds"]) & set(road_a_seeds), (
        "the replicate must be an independent draw"
    )

    for band in (band_768, band_512, band_224):
        assert band["phase"] == "roadb_p3"
        assert band["tier"] == "keeper"
        assert band["task"]["kind"] == "train_cv"
        for field in ("trainable", "geometry", "label", "backbone",
                      "max_epochs", "patience", "inner_val_frac", "monitor",
                      "learning_rate", "weight_decay", "batch_size"):
            assert band["task"][field] == p3["task"][field], field

    # Each resolution arm reads its artifact at the hash the residual gate
    # verified.
    declared = {e["name"]: e for e in band_768["inputs"]}
    assert band_768["task"]["staged_artifact"] == "roadb_768_square_g1_v1"
    assert declared["roadb_768_square_g1_v1"]["rollup_sha256"].startswith(
        "b4cbd5150cb08f5ad"
    )
    declared = {e["name"]: e for e in band_512["inputs"]}
    assert band_512["task"]["staged_artifact"] == "roadb_512_square_g1_v1"
    assert declared["roadb_512_square_g1_v1"]["rollup_sha256"].startswith(
        "fa0ba0269491f33c0"
    )
    # The replicate reads Road A's staged_v1 at Road A's hash.
    p3_declared = {e["name"]: e for e in p3["inputs"]}
    replicate = {e["name"]: e for e in band_224["inputs"]}
    assert band_224["task"]["staged_artifact"] == "staged_v1"
    assert replicate["staged_v1"]["rollup_sha256"] == (
        p3_declared["staged_v1"]["rollup_sha256"]
    )
    assert replicate["manifest_v1"]["rollup_sha256"] == (
        p3_declared["manifest_v1"]["rollup_sha256"]
    )


def test_the_band_rule_resolves_to_one_band_at_all_three_points():
    """**[MEASURED 2026-08-09, three points] Every pairwise F(9,9) interval
    contains 1 -- one planning band.** And the middle point, meant to
    confirm, has the WIDEST spread (0.021425, ratio 1.72 against 224, near
    the edge of what ten seeds can see) -- so the registered conservatism
    takes the largest of the three: a planning band below an observed arm SD
    would under-plan the arms that look like 512."""
    from cleft.train import phase3

    record = roadb.SEED_BAND_RESOLVED
    assert record["applied_rule"] == "SEED_BAND_DESIGN.decision_rule_preregistered"
    sds = {
        224: record["sweeps"]["roadb_224_replicate"]["sd"],
        512: record["sweeps"]["roadb_512"]["sd"],
        768: record["sweeps"]["roadb_768"]["sd"],
    }
    assert sds == {224: 0.012436, 512: 0.021425, 768: 0.016739}
    assert sds[512] / sds[224] == pytest.approx(1.723, abs=2e-3)
    assert record["outcome"] == "one planning band"
    # The registered value was "the larger of the two"; 512 exceeded both
    # extremes, and the rule's own conservatism rationale takes the largest.
    assert record["planning_band"] == 0.021425
    assert record["two_point_planning_band"] == 0.016739
    assert "under-plan" in record["three_point_application"]
    assert "1.10" in record["noise_floor"]

    # The claimable-delta table re-derived from the resolved band, by the
    # formula rather than by hand (PLAN 4.12: a re-measured SD gives a
    # re-derived band).
    for n_seeds, expected in record["claimable_delta"].items():
        assert phase3.claimable_delta(
            int(n_seeds), sd=record["planning_band"]
        ) == pytest.approx(expected, abs=5e-4), n_seeds


def test_the_resolution_drop_verification_passed_and_is_recorded():
    """**The gate this record put in front of the finding was passed, not
    skipped**: input_size [768, 768], dynamic_input true, embedding_dim 768,
    run finalized cleanly. The name keeps PENDING because that is what the
    record WAS -- the status says what it is, the G2_WHITE_DEFECT
    convention."""
    record = roadb.RESOLUTION_DROP_PENDING_VERIFICATION
    assert record["status"].startswith("VERIFIED")
    assert record["means"] == {224: 0.2319, 768: 0.0898}
    assert record["delta"] == pytest.approx(-0.1421, abs=1e-3)
    # Both candidate explanations were recorded before either was chosen,
    # and the verification decided: plumbing is excluded, real stands.
    assert "plumbing" in record["explanations"]
    assert "real" in record["explanations"]
    verification = record["verification"]
    assert verification["input_size"] == [768, 768]
    assert verification["dynamic_input"] is True
    assert verification["embedding_dim"] == 768
    # The checklist named the wrong FILE -- a seeded sweep's summary is
    # seed_variance.json, not metrics.json -- and that correction is part
    # of the record so the next checklist reads the right one.
    assert "seed_variance.json" in verification["record_read"]
    assert "wrong file" in verification["checklist_correction"]
    # What the finding became once verified is scoped in its own constant.
    assert "BRANCH_1" in record["resolved_into"]


def test_branch_1s_premise_is_refuted_for_frozen_transfer_precisely_scoped():
    """**A cliff, not a decline** -- 512 and 768 are indistinguishable and
    both sit ~0.146 below 224. What is refuted is that a frozen ImageNet ViT
    can USE the recovered pixels; the 10.8x discard stays measured and true,
    and fine-tuning at resolution is explicitly NOT tested."""
    record = roadb.BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER
    assert record["means"] == {224: 0.2319, 512: 0.0857, 768: 0.0898}
    assert "cliff" in record["shape"] and "not a decline" in record["shape"]
    # 512 and 768 are indistinguishable by the project's own named quantity.
    assert record["plateau"]["delta"] == pytest.approx(0.0041, abs=1e-3)
    assert record["plateau"]["arm_means_95"] == pytest.approx(0.017, abs=1e-3)
    # The scope, stated in both directions.
    assert "frozen" in record["refuted"]
    assert "10.8" in record["still_true"]
    assert "fine-tun" in record["not_tested"]
    assert "position embeddings" in record["not_tested"]
    assert "Phase 6" in record["not_tested"]
    # And the stake: this distinction decides the road's continuation.
    assert "continues" in record["decides"] and "stops" in record["decides"]
    # The cliff reading: leaving 224 is the damage; magnitude barely matters.
    assert "qualitatively the same violation" in record["reading"]


def test_the_continuation_decision_is_recorded_with_its_reasoning():
    """**[DECIDED 2026-08-09] Road B continues, and Phase 6 runs ALL THREE
    resolutions.** Dropping 768 would choose on a difference the frozen data
    does not support -- the paired-audit error -- and a pair gives up the
    trend that makes the axis claimable."""
    record = roadb.ROAD_B_CONTINUES
    assert "not 512 alone" in record["decision"]
    assert "paired-audit error" in record["why_not_drop_768"]
    assert "no more out-of-distribution" in record["why_not_drop_768"]
    assert "claimable" in record["why_three_points"]
    # The scope sentence that justifies continuing at all.
    assert "frozen ImageNet features worse" in record["why_continue"]
    assert "not a consideration" in record["cost"]


def test_swin_is_not_excluded_from_the_resolution_axis():
    """**[MEASURED 2026-08-09] Building swin AT img_size 512/768 carries the
    224 checkpoint's parameters unchanged -- zero shape mismatches** -- so
    the square axis stays four-backbone, the opposite of the non-square
    outcome. The recorded 224 constraint was about FEEDING, not BUILDING."""
    record = roadb.SWIN_AT_RESOLUTION
    assert "IDENTICAL" in record["result"]
    assert "zero mismatches" in record["result"]
    assert "window-relative" in record["why"]
    assert "cleaner resolution transfer than ViT" in record["why"]
    assert "feeding" in record["feeding_vs_building"]
    # All three caveats named: padding, pinned timm, non-square not reopened.
    assert "pads windows" in record["caveats"]["window_padding"]
    assert "1.0.7" in record["caveats"]["pinned_timm"]
    assert "not reopened" in record["caveats"]["non_square_not_reopened"]
    assert record["consequence"] == "the square resolution axis stays four-backbone"

    # The pinned-image gate run CONFIRMED it, strongly -- and the
    # buffer-not-gated decision was vindicated rather than merely unfired.
    confirmation = record["pinned_confirmation"]
    assert "1.0.7" in confirmation["confirmed"]
    assert "byte-equal" in confirmation["result"]
    assert "attn_mask" in confirmation["buffers"]
    assert "shift is skipped" in confirmation["buffers"], (
        "the deepest-stage name difference must carry its explanation"
    )
    assert confirmation["gating_decision"].startswith("VINDICATED")


def test_the_swin_prediction_is_registered_before_the_runs():
    """**The mechanism contrast is measured on both architectures, so the
    prediction is committed now** -- outcomes and readings before numbers,
    the pattern that saved 7C. The two regimes are named separately so the
    readings cannot blur (R2)."""
    record = roadb.SWIN_RESOLUTION_PREDICTION
    assert "before any Phase 6 run" in record["registered"]
    assert "interpolates nothing" in record["measured_contrast"]
    assert "Swin survives" in record["prediction"]
    # Each regime commits BOTH directions -- what supports and what refutes.
    for regime in ("frozen_regime", "trained_regime"):
        assert record[regime]["supports"]
        assert record[regime]["refutes"]
    assert "afflicts both" in record["frozen_regime"]["refutes"]
    assert "never positional" in record["trained_regime"]["refutes"]


def test_the_masked_scut_resolution_configs_are_the_frozen_composition_resized(repo_root):
    """**Build order step 2**: one artifact per size, G1+G2 bundled within
    each (the recorded masked-SCUT decision), the interpolation named in the
    config rather than passing unremarked."""
    import yaml

    def load(stem):
        return yaml.safe_load(
            (repo_root / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )

    road_a = load("p5_masked_scut")
    for stem, size in (
        ("roadb_p5_masked_scut_512", 512),
        ("roadb_p5_masked_scut_768", 768),
    ):
        payload = load(stem)
        assert payload["phase"] == "roadb_p5"
        # KEEPER, deliberately departing from Road A's dev (born dev, never
        # changed): the artifact feeds twelve pretraining cells, and
        # keeper's clean-tree guarantee is the one thing MANIFEST
        # provenance cannot supply.
        assert payload["tier"] == "keeper"
        task = payload["task"]
        assert task["kind"] == "masked_scut"
        assert task["size"] == size
        assert task["out_version"] == f"masked_{size}_v1"
        assert task["geometries"] == ["g1", "g2"], "bundling preserved"
        # Same sampling, same cohort, same persistence mode as Road A's --
        # the only scientific difference is the size.
        for field in ("ar_sampling", "n_faces", "write_artifact",
                      "checkpoint_every"):
            assert task[field] == road_a["task"][field], field
        # Same SCUT root at the same verified hash.
        assert payload["inputs"][0]["rollup_sha256"] == (
            road_a["inputs"][0]["rollup_sha256"]
        )
        # The interpolation is in the config's own text, not a surprise.
        text = (repo_root / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        assert "350x350" in text
        assert "VISIBLE" in text


def test_the_phase6_structure_names_every_cell_and_its_gate():
    """**No cell is 'ready' at a new size** -- each names the gate that must
    pass before it runs, so no cluster run discovers what a laptop check
    could have said (R10, five-for-five and counting)."""
    cells = roadb.phase6_resolution_cells()
    assert len(cells) == 12  # four backbones x three resolutions
    by_key = {(c["backbone"], c["resolution"]): c["status"] for c in cells}
    for backbone in roadb.PHASE_6_STRUCTURE["backbones"]:
        assert by_key[(backbone, 224)] == "road A's own regime"
        for resolution in (512, 768):
            assert by_key[(backbone, resolution)].startswith("pending:"), (
                backbone, resolution,
            )
    # ViT's training-path gate is distinct from its confirmed extraction.
    assert "TRAINING path" in by_key[("vit_b16", 512)]
    assert "pinned" in by_key[("swin_b", 768)]
    assert "Xception" in by_key[("srgnn", 512)]
    # The four pre-run gates are enumerated in the structure itself.
    assert len(roadb.PHASE_6_STRUCTURE["gates_before_any_run"]) == 4


def test_scut_at_the_new_resolutions_is_uniformly_interpolated():
    """**Every SCUT image is 350x350** (the repo's own record), so 512 and
    768 upsample ALL 5500 -- not a mixed population, a uniformly
    interpolated one, and the consequence is stated: higher-resolution
    pretraining adapts positional geometry, not finer detail."""
    scut = roadb.PHASE_6_STRUCTURE["scut_staging"]
    assert "350x350" in scut["source_size"]
    assert "all 5500" in scut["upsampling"][512]
    assert "1.46x" in scut["upsampling"][512]
    assert "2.19x" in scut["upsampling"][768]
    assert "positional geometry" in scut["consequence"]
    assert "2758" in scut["consequence"], (
        "the asymmetry against the cleft sources must be visible"
    )
    # The write-up guard, in supervision-facing terms, recorded BEFORE the runs.
    assert "fine detail" in scut["write_up_guard"]
    assert "adapted positional geometry" in scut["write_up_guard"]
    assert "supervision sees this BEFORE the runs" in scut["write_up_guard"]
    # The ratios really are 512/350 and 768/350.
    assert 512 / 350 == pytest.approx(1.463, abs=1e-3)
    assert 768 / 350 == pytest.approx(2.194, abs=1e-3)


def test_the_build_order_puts_the_falsifier_first(repo_root):
    """**The swin gate is the one thing that could refute the four-backbone
    conclusion, and it is short -- so it runs first**, then SCUT staging and
    sheets, then the pretraining path. And the gate is fully wired: task,
    schema, config."""
    import yaml

    order = roadb.PHASE_6_STRUCTURE["build_order"]
    assert order.startswith("1. the swin build gate")
    assert order.index("swin") < order.index("SCUT") < order.index("pretraining")

    from cleft.run import TASKS

    assert TASKS["roadb_swin_build_gate"].__name__ == "swin_build_gate"

    payload = yaml.safe_load(
        (repo_root / "configs" / "roadb_p6_swin_build_gate.yaml")
        .read_text(encoding="utf-8")
    )
    assert payload["phase"] == "roadb_p6"
    assert payload["tier"] == "keeper"
    assert payload["task"]["kind"] == "roadb_swin_build_gate"
    assert payload["task"]["sizes"] == [512, 768]
    assert payload["task"]["pretrained"] is True
    assert "inputs" not in payload, "the gate reads no data"

    # The gate's criterion is parameters, not buffers -- gating on buffers
    # would refuse over a quantity the claim is not about (R2).
    import inspect

    from cleft import roadb_tasks

    source = inspect.getsource(roadb_tasks.swin_build_gate)
    assert "named_parameters" in source
    assert "reported, not gated" in source
    assert "REFUSED" in source
    assert "timm_version" in source


def test_gate_2_threads_input_size_and_states_its_scope(repo_root):
    """**A pass must never be read as more than construction plus optimiser
    registration** -- training stability at interpolated positions is Phase
    6's first run, and both the task and the config say so."""
    import inspect

    import yaml

    from cleft import roadb_tasks
    from cleft.train import pretrain

    # The fix is in the pretrain path itself: reset derives input_size from
    # the features' own shape and hands it to the factory.
    source = inspect.getsource(pretrain.TorchPretrainModel.reset)
    assert "input_size" in source
    assert "train_features" in source.split("input_size")[1].split("=")[1], (
        "the size must come from the features' own shape"
    )

    record = roadb.PRETRAIN_BUILDER_THREADS_INPUT_SIZE
    assert "224 asserts" in record["fault"]
    assert record["measured"]["vit_b16"] == 85_799_425
    assert record["measured"]["swin_b"] == 86_744_249
    assert "nothing learned" in record["parameters_size_invariant"]
    assert "not a gate" in record["scope"]["not_covered"]

    from cleft.run import TASKS

    assert TASKS["roadb_pretrain_build_gate"].__name__ == "pretrain_build_gate"
    payload = yaml.safe_load(
        (repo_root / "configs" / "roadb_p6_pretrain_build_gate.yaml")
        .read_text(encoding="utf-8")
    )
    assert payload["task"]["kind"] == "roadb_pretrain_build_gate"
    assert payload["task"]["backbones"] == ["vit_b16", "swin_b", "srgnn", "agnet"]
    assert payload["task"]["sizes"] == [224, 512, 768]
    assert payload["tier"] == "keeper"
    assert "inputs" not in payload
    text = (repo_root / "configs" / "roadb_p6_pretrain_build_gate.yaml").read_text(
        encoding="utf-8"
    )
    assert "NOT COVERED" in text and "first pretraining run" in text

    gate_source = inspect.getsource(roadb_tasks.pretrain_build_gate)
    assert "NOT COVERED" in gate_source
    assert "parameters_size_invariant" in gate_source
    assert "REFUSED" in gate_source


def test_the_twelve_pretraining_cells_mirror_road_a_field_for_field(repo_root):
    """**"Same procedure, different input" is byte equality, not
    proofreading**: each cell's task dict IS its Road A counterpart's --
    since gate 2 the builder derives input size from the data, so the task
    carries no size field -- and only the declared masked artifact differs."""
    import yaml

    counterparts = {
        "vit_b16": "p6_pretrain_vit_b16_masked_g1",
        "swin_b": "p6_pretrain_swin_b_masked_g1",
        "srgnn": "p6_pretrain_srgnn_native_masked_g1",
        "agnet": "p6_pretrain_agnet_native_masked_g1",
    }
    artifacts = {224: "masked_v1", 512: "masked_512_v1", 768: "masked_768_v1"}
    # Scoped past roadb_p6_pretrain_build_gate.yaml, which the wider glob
    # catches.
    shipped = sorted(
        (repo_root / "configs").glob("roadb_p6_pretrain_*_masked_g1_*.yaml")
    )
    assert len(shipped) == 12
    placeholder = "0" * 64

    def load(stem):
        return yaml.safe_load(
            (repo_root / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )

    for backbone, counterpart_stem in counterparts.items():
        road_a = load(counterpart_stem)
        road_a_inputs = {e["name"]: e for e in road_a["inputs"]}
        for size in (224, 512, 768):
            stem = f"roadb_{counterpart_stem}_{size}"
            payload = load(stem)
            # THE assertion: the task block is Road A's, field for field.
            assert payload["task"] == road_a["task"], stem
            assert payload["phase"] == "roadb_p6"
            assert payload["tier"] == road_a["tier"]
            assert payload["seed"] == road_a["seed"]

            declared = {e["name"]: e for e in payload["inputs"]}
            assert declared["masked_scut"]["path"].endswith(
                f"data/scut/{artifacts[size]}"
            ), stem
            assert declared["scut_root"]["rollup_sha256"] == (
                road_a_inputs["scut_root"]["rollup_sha256"]
            )
            if size == 224:
                # The SAME artifact as Road A's run, at Road A's hash.
                assert declared["masked_scut"]["rollup_sha256"] == (
                    road_a_inputs["masked_scut"]["rollup_sha256"]
                )
            else:
                digest = declared["masked_scut"]["rollup_sha256"]
                assert len(digest) == 64, stem  # placeholder or pasted

            text = (repo_root / "configs" / f"{stem}.yaml").read_text(
                encoding="utf-8"
            )
            assert "OWN seed regime" in text, stem
            if backbone == "vit_b16" and size == 512:
                assert "LAUNCH THIS CELL FIRST" in text
                assert "SWIN_RESOLUTION_PREDICTION" in text
            else:
                assert "runs FIRST" in text, stem
            if backbone in ("vit_b16", "swin_b") and size == 768:
                assert "COSTING NOTE" in text, stem
                assert "heaviest" in text, stem
            if size == 224:
                assert "RE-RUN OF ROAD A" in text, stem


def test_sibling_cells_at_a_resolution_declare_one_identical_artifact(repo_root):
    """**A resolution axis where one cell reads a different artifact than
    its siblings would produce a backbone difference that is really an
    input difference, and nothing downstream would say so.** The four cells
    at each size must declare the identical artifact at the identical hash
    -- and the three sizes' artifacts must be pairwise DISTINCT, or two
    resolutions silently read one input."""
    import yaml

    by_size = {224: set(), 512: set(), 768: set()}
    for path in (repo_root / "configs").glob(
        "roadb_p6_pretrain_*_masked_g1_*.yaml"
    ):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        size = int(path.stem.rsplit("_", 1)[1])
        entry = {e["name"]: e for e in payload["inputs"]}["masked_scut"]
        by_size[size].add((entry["path"], entry["rollup_sha256"]))

    for size, declared in by_size.items():
        assert len(declared) == 1, (
            f"the four {size} cells disagree on their artifact: {declared}"
        )
    placeholder = "0" * 64
    hashes = {size: next(iter(declared))[1] for size, declared in by_size.items()}
    assert len(set(hashes.values())) == 3, (
        f"two resolutions declare one artifact hash: {hashes}"
    )
    # Resolved, not placeholders -- the artifacts exist and their hashes are
    # pasted; a regression to all-zeros here would silently re-block launch.
    assert hashes[512].startswith("cab0007c")
    assert hashes[768].startswith("72080a4c")
    assert hashes[224] != placeholder


def test_the_vit_reproducibility_stop_is_recorded_with_its_decision_tree():
    """**The 224 re-run cells caught it, exactly as designed** -- and the
    record separates what code already answers from what only the run
    records can, with the sha comparison as the decider."""
    record = roadb.VIT_NOT_REPRODUCIBLE
    assert record["deltas_224"]["vit_b16"] == 0.023
    assert record["deltas_224"]["swin_b"] == 0.0001
    assert record["vit_512_pair"]["checkpoints"] == ("b151203a", "4f45c33e")
    # The two code-level answers are pinned facts, not prose.
    from cleft.models.factory import timm_kwargs_for

    assert timm_kwargs_for("vit_base_patch16_224", (224, 224)) == {}, (
        "if this stops holding, answered_from_code.dynamic_at_224 is stale"
    )
    assert "dict-equality" in record["answered_from_code"]["deterministic"]
    # Both mechanisms named, the decider stated, the provenance hole owned.
    assert "guard 3 never hashes" in record["mechanisms"]["undeclared_init"]
    assert "atomics" in record["mechanisms"]["dynamic_resample"]
    assert "declared, hashed" in record["decision"]
    assert "cannot arbitrate" in record["local_probe"]


def test_the_init_is_declarable_and_the_hole_is_recorded(repo_root):
    """**The comparison could not be made, and that was the answer**: the
    only sha in the records was the OUTPUT's. The init is now a versioned,
    hashed artifact; runs that skip it name their own absence; and Road A's
    thirty share the hole -- recorded with exactly the claim the write-up
    may and may not make."""
    import yaml

    record = roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE
    assert "guard 3 never hashed" in record["hole"]
    assert "conventional-name route" in record["fix"]
    assert "REFUSES a graph declaration" in record["scope"]
    assert "not demonstrable for any of them" in record["road_a"]
    assert "must not claim" in record["road_a"]
    assert "agree ->" in record["experiment"] and "differ ->" in record["experiment"]
    assert "UNARBITRABLE" in roadb.VIT_NOT_REPRODUCIBLE["comparison_outcome"]

    # The constants the wiring keys on, and the loader's contract.
    from cleft.train import pretrain

    assert pretrain.INIT_INPUT_NAME == "pretrained_init"
    assert pretrain.INIT_WEIGHTS_NAME == "init.npz"
    import inspect

    source = inspect.getsource(pretrain.load_declared_init)
    assert "Refusing a partial load" in source
    assert 'startswith("head")' in source

    # Four snapshot configs, one per backbone, no inputs (the LAST
    # undeclared download, once).
    from cleft.run import TASKS

    assert TASKS["snapshot_pretrained_init"].__name__ == "snapshot_pretrained_init"
    for backbone in ("vit_b16", "swin_b", "srgnn", "agnet"):
        payload = yaml.safe_load(
            (repo_root / "configs" / f"roadb_init_snapshot_{backbone}.yaml")
            .read_text(encoding="utf-8")
        )
        assert payload["task"]["kind"] == "snapshot_pretrained_init"
        assert payload["task"]["backbone"] == backbone
        assert payload["tier"] == "keeper"
        assert "inputs" not in payload

    # Transformer cells declare the init; graph cells must NOT -- the task
    # refuses a declaration it cannot consume, so a config carrying one
    # would be unrunnable.
    hashes = {"vit_b16": set(), "swin_b": set()}
    for path in (repo_root / "configs").glob(
        "roadb_p6_pretrain_*_masked_g1_*.yaml"
    ):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared = {e["name"] for e in payload["inputs"]}
        backbone = payload["task"]["backbone"]
        if backbone in ("vit_b16", "swin_b"):
            assert "pretrained_init" in declared, path.stem
            entry = {e["name"]: e for e in payload["inputs"]}["pretrained_init"]
            assert entry["path"].endswith(f"data/inits/init_{backbone}_v1")
            hashes[backbone].add(entry["rollup_sha256"])
        else:
            assert "pretrained_init" not in declared, path.stem
    # Each transformer's three cells start from ONE declared init at the
    # pasted rollup -- the sibling invariant, at the init layer.
    for backbone, prefix in (("vit_b16", "c974c782"), ("swin_b", "934fb980")):
        assert len(hashes[backbone]) == 1, backbone
        assert next(iter(hashes[backbone])).startswith(prefix), backbone
        assert record["snapshots"][backbone]["rollup_prefix"] == prefix
    # And the artifact carries the upstream's own constraint statement.
    assert "dynamic_img_size overrides" in (
        record["vit_snapshot"]["fixed_input_size"]
    )
    # The asymmetry that bears on the 224 question: only ViT is hub-only.
    assert "hub-only" in record["snapshots"]["vit_b16"]["url"]
    assert "asset" in record["snapshots"]["swin_b"]["url"]
    assert "only ViT" in record["snapshots"]["asymmetry"]


def test_the_512_pair_dissolved_as_cross_device_and_224_remains():
    """**[MEASURED, env.json] Two runs on different hardware are not a
    determinism test at all** -- the branch the decision tree lacked. Road A
    is single-device by the sweep, the 768 OOMs are not diagnostic, and the
    224 divergence stands alone with the hub-only init as its leading
    explanation."""
    # The thread now lives in the init record -- the dissolution, the sweep,
    # the remaining question and the plan sit beside the snapshots that
    # decide them.
    record = roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE
    resolved = record["resolved_512"]
    assert resolved["devices"] == {
        "0.7412": "NVIDIA A40", "0.7680": "RTX PRO 6000 Blackwell",
    }
    assert "not a determinism test" in resolved["third_branch"]
    assert "pin the GPU model" in resolved["third_branch"]
    assert "trains nothing" in record["road_a_single_device"]
    assert "not hardware-mixed" in record["road_a_single_device"]
    remaining = record["remaining_224"]
    assert remaining["delta"] == 0.023
    assert "HUB-ONLY" in remaining["leading_explanation"] or "hub" in (
        remaining["leading_explanation"]
    )
    assert "not the discriminator" in remaining["flag_does_not_predict"]
    assert "img_size at build" in remaining["flag_does_not_predict"]
    assert "not diagnostic" in record["768_cells"]
    # The next runs carry their readings BEFORE launch, including the
    # subtlety: the snapshot holds TODAY'S bytes, so under drift the
    # declared 224 run reproduces 0.8120, and 0.7893 would mean NO drift.
    plan = roadb.PRETRAINED_INIT_IS_NOW_DECLARABLE["next_runs"]
    assert "drift-consistent" in plan["1_vit_224_declared"]["reproduces_0.8120"]
    assert "no drift" in plan["1_vit_224_declared"]["reproduces_0.7893"]
    assert "same GPU MODEL" in plan["2_vit_512_pair"]
    assert "whole device" in plan["3_relaunch_768"]


def test_a_graph_backbone_refuses_a_declared_init_it_cannot_consume(tmp_path):
    """**An input the run never reads would still be hash-verified and
    recorded** -- a provenance record claiming weights fed a run they did
    not feed. Refused before any torch import."""
    from cleft.run import task_pretrain

    ctx = _StageCtx(
        tmp_path,
        task={
            "kind": "pretrain", "backbone": "srgnn", "source": "masked_g1",
            "region_scheme": "native", "scut_root": "scut_root",
            "epochs": 30, "inner_val_frac": 0.1, "monitor": "inner_val_pcc",
            "deterministic": False, "batch_size": 32,
            "learning_rate": 1e-4, "weight_decay": 0.01,
            "expect_train": 3300, "expect_test": 2199, "checkpoint_every": 1,
        },
        inputs=[
            {"name": "scut_root", "path": str(tmp_path / "scut")},
            {"name": "masked_scut", "path": str(tmp_path / "masked")},
            {"name": "pretrained_init", "path": str(tmp_path / "init")},
        ],
    )
    with pytest.raises(ValueError, match="do not consume"):
        task_pretrain(ctx)


def test_the_nondeterminism_finding_and_road_as_cost_are_recorded(repo_root):
    """**The pre-registered third reading fired**: three runs, three
    checkpoints, pinned weights, one hardware. Road A's cost is recorded
    FIRST -- its ViT figures are draws, and the geometry comparison's 0.041
    sits against a ~0.030 spread. The mechanism is measured to the laptop's
    limit, and the pinned probe ships with both surfaces committed."""
    import yaml

    finding = roadb.VIT_PRETRAINING_IS_NONDETERMINISTIC
    assert finding["spread"] == 0.030
    assert finding["runs"]["road_a"]["selected_epoch"] == 23
    assert finding["runs"]["road_b_declared"]["ckpt"] == "1f0015c8"
    assert "not the init" in finding["constraints_discharged"]
    assert "not numerical drift" in finding["trajectory_level"]
    assert "fused_attn=True" in finding["mechanism"]["inspection"]
    assert "fused_attn=False" in finding["mechanism"]["inspection"]
    assert "MATH forced 0.0" in finding["mechanism"]["controlled_contrast"]
    assert "MEASURED YES" in finding["mechanism"]["deterministic_false_answer"]
    assert "bitwise-equal" in finding["probe"]["runs"]
    assert "nothing " in finding["fix_options_flagged"].lower() or (
        "nondeterministic past" in finding["fix_options_flagged"]
    )

    cost = roadb.ROAD_A_VIT_ARMS_ARE_DRAWS
    assert cost["draws"] == {
        "vit_masked_g1": 0.7893, "vit_original": 0.8914,
        "vit_masked_g2": 0.8306,
    }
    assert "0.041 against a ~0.030 spread" in cost["inherits"]
    assert "LIMITATION, not a fix" in cost["discipline"]
    assert "swin" in cost["unaffected"]

    # The probe config: the vit-224 cell with EXACTLY one task field changed.
    def load(stem):
        return yaml.safe_load(
            (repo_root / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )

    probe = load("roadb_p6_det_probe_vit_224")
    cell = load("roadb_p6_pretrain_vit_b16_masked_g1_224")
    probe_task = dict(probe["task"])
    cell_task = dict(cell["task"])
    assert probe_task.pop("deterministic") is True
    assert cell_task.pop("deterministic") is False
    assert probe_task == cell_task, (
        "the probe must differ from the cell in the deterministic field alone"
    )
    # Same declared inputs, same hashes -- including the pinned init.
    assert {e["name"]: e["rollup_sha256"] for e in probe["inputs"]} == {
        e["name"]: e["rollup_sha256"] for e in cell["inputs"]
    }


def test_the_probe_conviction_and_the_unfused_decision_are_recorded():
    """**The second surface fired: bitwise pair, fused path convicted by
    elimination.** The decision (fused off for ViT builds), the numerics
    verdict (same computation, different fp order -- so the unfused
    endpoint is its own value), and the fourth-value observation are all
    recorded with their epistemics stated."""
    finding = roadb.VIT_PRETRAINING_IS_NONDETERMINISTIC
    probe = finding["probe_outcome"]
    assert probe["pcc"] == 0.835346
    assert probe["ckpt_prefix"] == "58fe1c5647741b4b"
    assert "not the default" in probe["fallback"]
    assert "chance-compatible" in finding["fourth_value_observation"]
    assert "own value" in finding["fourth_value_observation"]

    numerics = finding["numerics"]
    assert numerics["verdict"] == (
        "same computation, different fp evaluation order"
    )
    assert numerics["forward_max_diffs"]["fused_default_vs_unfused"] == 9.5e-07
    assert "reproducibility, not the number" in numerics["consequence"]

    decision = finding["decision"]
    assert decision["chosen"] == "fused_attn off for ViT pretraining builds"
    assert "byte-equal" in decision["why"]
    assert "extraction stays fused" in decision["implementation"]
    assert "SAME configs" in decision["reruns"]
    assert "undisturbed" in decision["reruns"]

    # And the implementation is really in the build path: build-only
    # toggle, keyed to ViT, with the loud refusal if it did not take.
    import inspect

    from cleft.train import pretrain

    source = inspect.getsource(pretrain.TorchPretrainModel.reset)
    assert "set_fused_attn(False)" in source
    assert 'self.name == "vit_b16"' in source
    assert "did not take" in source
    assert "set_fused_attn(fused_before)" in source, "the flag must restore"


def test_the_phase6_results_and_the_confirmed_prediction_are_recorded():
    """**Flat where readable, a shape not a claim for SR-GNN's mid-peak,
    and the trained-regime prediction confirmed with its registration
    date.**"""
    results = roadb.PHASE_6_RESOLUTION_RESULTS
    assert results["table"]["srgnn"] == {224: 0.8412, 512: 0.8474, 768: 0.8427}
    assert results["table"]["agnet"][768] == 0.8074
    assert len(results["table"]["vit_b16"][224]) == 4
    assert "FLAT where readable" in results["trend"]
    assert "a shape" in results["srgnn_shape_not_claim"]
    assert "crossing near 512" in results["srgnn_shape_not_claim"]

    confirmed = roadb.SWIN_RESOLUTION_PREDICTION["confirmed"]
    assert confirmed["registered"] == "2026-08-09, before any Phase 6 run"
    assert "0.146 cliff" in confirmed["measured"]
    assert "never run" in confirmed["frozen_regime_status"]


def test_the_unfused_overturn_names_its_own_inference_errors():
    """**The probe convicted the class, not the member** -- the flag forces
    fallbacks for every nondeterministic op -- **and one forward-backward
    does not bound thirty epochs.** Both errors recorded where the next
    reader of the conviction will meet them, the decision superseded in
    place, and the toggle's comment no longer claims sufficiency."""
    record = roadb.UNFUSED_WAS_NOT_SUFFICIENT
    assert record["unfused_pair"]["pccs"] == (0.8001, 0.7648)
    assert record["unfused_pair"]["spread"] == 0.035
    assert "all-four-224 spread 0.055" in record["unfused_pair"]["note"]
    assert "convicts the class" in record["inference_errors"]["class_not_member"]
    assert "thirty" in record["inference_errors"]["one_step_no_bound"]
    assert "stop chasing" in record["decision"]
    assert "is not taken" in record["decision"]
    assert "the same size" in record["cost"]
    assert "three clouds" in record["cost"]
    assert "churn the procedure a third time" in record["toggle_stays"]
    # The old decision is superseded IN PLACE, and the build comment no
    # longer claims sufficiency.
    assert "OVERTURNED" in (
        roadb.VIT_PRETRAINING_IS_NONDETERMINISTIC["decision"]["superseded"]
    )
    import inspect

    from cleft.train import pretrain

    source = inspect.getsource(pretrain.TorchPretrainModel.reset)
    assert "NOT sufficient" in source
    assert "UNFUSED_WAS_NOT_SUFFICIENT" in source


def test_the_phase7_pairing_is_matched_pipelines_with_reasons():
    """**Three cells per backbone, not nine** -- the Stage C trap named,
    one variable per step, readability -- with the interesting mismatch
    cell preserved as ONE pre-registered arm rather than a lattice."""
    record = roadb.PHASE_7_RESOLUTION_PAIRING
    assert record["answer"] == (
        "matched pipelines -- three cells per backbone, not nine"
    )
    assert "design, not mechanics" in record["mechanically_possible_to_differ"]
    assert "Stage C trap" in record["reasons"]["stage_c_lesson"]
    assert "3x3 grid" in record["reasons"]["one_variable_per_step"]
    assert "three clouds" in record["reasons"]["readability"]
    assert "ONE pre-registered arm" in record["the_interesting_cell"]
    assert "arms-one-at-a-time" in record["the_interesting_cell"]
    assert "Attribution, not cost" in record["the_interesting_cell"]


def test_the_extraction_plan_is_derived_from_the_arm_list():
    """**24 consumed, 23 extracted, 1 reused** -- imagenet sets shared per
    (backbone, resolution), swin-224-masked reused by bitwise identity, and
    the two forced rules registered before any extraction: the ViT
    draw-pick rule and init-vintage consistency."""
    plan = roadb.PHASE_7_EXTRACTION_PLAN
    assert (plan["sets_consumed"], plan["extractions"], plan["reuses"]) == (
        24, 23, 1
    )
    assert "never by score" in plan["vit_draw_pick_rule"]
    assert "declared snapshots, 224 included" in plan["init_vintage_rule"]
    assert "GATE" in plan["init_vintage_rule"]

    sets = roadb.phase7_extraction_sets()
    assert len(sets) == 24
    reused = [s for s in sets if s["reuses"]]
    assert len(reused) == 1
    assert (reused[0]["backbone"], reused[0]["resolution"]) == ("swin_b", 224)
    assert "bitwise-identical" in reused[0]["reuses"]
    for entry in sets:
        assert entry["geometry"] == "g1", "matched pipelines are G1"
        if entry["init"] == "imagenet":
            assert entry["checkpoint"] is None
            assert entry["init_source"].startswith(f"init_{entry['backbone']}_v1")
        else:
            assert "cell's" in entry["checkpoint"]
        expected_store = (
            "feature_maps" if entry["backbone"] in ("srgnn", "agnet")
            else "pooled_vectors"
        )
        assert entry["stores"] == expected_store, entry["backbone"]
    # ViT masked sets carry the pick rule inline, where the config writer
    # will read it.
    vit_masked = [
        s for s in sets
        if s["backbone"] == "vit_b16" and s["init"] == "scut_masked"
    ]
    assert len(vit_masked) == 3
    assert all("draw-pick rule" in s["checkpoint"] for s in vit_masked)


def test_the_extract_init_gate_is_built_with_both_holes_closed(tmp_path):
    """**Building the gate found the second gate-2 hole in the same
    signature** -- ``_load_model`` without ``input_size`` would have killed
    all 16 new-resolution extractions after submission. Both threads are in
    the source, the refusals sit at the failure site, and the unread-
    declaration guard fires at task level before any file is read."""
    import inspect

    from cleft.train import extract

    source = inspect.getsource(extract._load_model)
    assert "input_size=input_size" in source or "input_size=input_size," in (
        inspect.getsource(extract.extract_features)
    )
    assert "load_declared_init" in source
    assert "recorded follow-up" in source, "the graph refusal must say why"
    assert "without being read" in source, "non-imagenet declarations refused"
    features_source = inspect.getsource(extract.extract_features)
    assert "init_dir=init_dir" in features_source
    assert "pretrained_init" in features_source, (
        "every imagenet set's report names declared-or-downloaded"
    )

    # The task-level unread-declaration guard, before any file access.
    from cleft.run import task_extract_embeddings

    ctx = _StageCtx(
        tmp_path,
        task={
            "kind": "extract_embeddings", "out_version": "x_v1",
            "manifest_artifact": "manifest_v1", "staged_artifact": "staged",
            "batch_size": 32,
            # A masked-only config: the declared init would be verified and
            # recorded without ever being read.
            "sets": [
                {"backbone": "vit_b16", "init": "scut_masked",
                 "geometry": "g1"},
            ],
        },
        inputs=[
            {"name": "manifest_v1", "path": str(tmp_path / "m")},
            {"name": "staged", "path": str(tmp_path / "s")},
            {"name": "pretrained_init", "path": str(tmp_path / "init")},
        ],
    )
    with pytest.raises(ValueError, match="no imagenet set"):
        task_extract_embeddings(ctx)
    # And the graph imagenet route is REAL now -- the measured mappings --
    # so the extract-side graph refusal is gone while the TRAINING side
    # keeps it (task_pretrain).
    from cleft.train import extract as extract_module

    import inspect as _inspect

    assert "_load_graph_imagenet_init" in _inspect.getsource(
        extract_module._load_model
    )
    assert "recorded follow-up: task_pretrain still refuses" in (
        _inspect.getsource(extract_module._load_model)
    )

    # And the record: the second hole named, the determinism measurement
    # with its pre-registered pinned confirmation.
    plan = roadb.PHASE_7_EXTRACTION_PLAN
    assert "second gate-2 hole" in plan["gate_built"]
    assert "16 of the 23" in plan["gate_built"]
    assert "no backward" in plan["extraction_determinism"]["local"]
    assert "runs twice" in plan["extraction_determinism"]["pinned_confirmation"]
    assert "batch stops" in plan["extraction_determinism"]["pinned_confirmation"]
    # BOTH branches are one recorded decision -- pass clears the batch,
    # the check copy is never declared and is deleted after comparison.
    on_pass = plan["extraction_determinism"]["on_pass"]
    assert "PROCEED with no further per-set double runs" in on_pass
    assert "_detcheck_v1" in on_pass
    assert "no config ever declares" in on_pass
    assert "STOPS" in plan["extraction_determinism"]["on_fail"]


#: Builder calls that are size-correct WITHOUT threading input_size, each
#: with the reason -- an entry here is a recorded decision, not an escape.
SIZE_CORRECT_WITHOUT_THREADING = {
    ("train/torch_backbone.py", "FrozenExtractor.__init__"): (
        "phase 8 / grad-cam machinery, built before any data arrives; Road "
        "A ran it at 224 only. A Road B Phase 8 at the new resolutions MUST "
        "thread a constructor input_size first -- this entry is that named "
        "gate, not a waiver"
    ),
    ("train/torch_backbone.py", "TorchBackbone.reset"): (
        "the full-fine-tune path -- REJECTED policy (PCC -0.013, PLAN "
        "4.7), 224-only Road A machinery no config runs; threading would "
        "modernise a path the project measured and declined"
    ),
    ("train/graph_cleft.py", "GraphHeadBackbone.reset"): (
        "graph builds ignore input_size by design: create_backbone routes "
        "kind=graph before timm_kwargs_for, and the geometry gate measured "
        "the graphs size-agnostic (fixed 42x42 map; image-space regions)"
    ),
}


def test_every_backbone_builder_call_threads_input_size_or_is_recorded():
    """**Two occurrences of an identical hole in two paths is a pattern,
    not two incidents.** Both known sites (the pretrain builder, then
    extract._load_model) were found by reading a function for another
    reason; a third would fail on the cluster after submission like the
    others. This is R10's surface with a new question -- a builder called
    without the argument that makes it size-correct -- swept statically:
    every ``create_backbone`` call either threads ``input_size`` or is
    recorded in the allowlist with the reason fixed-size is correct there.
    """
    import ast
    from pathlib import Path as _Path

    import cleft

    root = _Path(cleft.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))

        stack: list[str] = []

        def visit(node):
            pushed = False
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                stack.append(node.name)
                pushed = True
            if isinstance(node, ast.Call):
                func = node.func
                name = (
                    func.id if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute)
                    else None
                )
                if name == "create_backbone":
                    threaded = any(
                        kw.arg == "input_size" for kw in node.keywords
                    )
                    context = ".".join(
                        part for part in stack
                        if not part.startswith("_") or part == "__init__"
                    ) or "<module>"
                    if not threaded and (rel, context) not in (
                        SIZE_CORRECT_WITHOUT_THREADING
                    ):
                        offenders.append(f"{rel}:{node.lineno} in {context}")
            for child in ast.iter_child_nodes(node):
                visit(child)
            if pushed:
                stack.pop()

        visit(tree)

    assert not offenders, (
        "builder calls without input_size and without a recorded reason: "
        f"{offenders}. Thread the staged shape through, or record why "
        "fixed-size is correct there in SIZE_CORRECT_WITHOUT_THREADING."
    )
    # The allowlist itself is load-bearing prose: every entry carries a
    # reason, and the Phase 8 entry is a named GATE.
    for key, reason in SIZE_CORRECT_WITHOUT_THREADING.items():
        assert len(reason) > 40, key
    assert "named gate" in SIZE_CORRECT_WITHOUT_THREADING[
        ("train/torch_backbone.py", "FrozenExtractor.__init__")
    ]


def test_the_extraction_configs_enumerate_the_derived_sets(repo_root):
    """**23 configs + the detcheck twin, one set each, per-set artifacts**
    (the grouping decision) -- imagenet cells at the declared snapshot
    hashes, masked cells at documented PENDING placeholders under the
    draw-pick rule, and the swin-224-masked reuse deliberately absent."""
    import yaml

    shipped = sorted((repo_root / "configs").glob("roadb_p7_extract_*.yaml"))
    assert len(shipped) == 24  # 23 extractions + the detcheck twin

    derived = [s for s in roadb.phase7_extraction_sets() if not s["reuses"]]
    assert len(derived) == 23
    stems = {p.stem for p in shipped}
    for entry in derived:
        short = "imagenet" if entry["init"] == "imagenet" else "masked"
        assert (
            f"roadb_p7_extract_{entry['backbone']}_{short}_"
            f"{entry['resolution']}"
        ) in stems
    assert "roadb_p7_extract_swin_b_masked_224" not in stems, (
        "the reuse must not have a config -- Road A's set serves it"
    )
    assert "roadb_p7_extract_vit_b16_imagenet_512_detcheck" in stems

    placeholder = "0" * 64
    for path in shipped:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        task = payload["task"]
        assert payload["phase"] == "roadb_p7"
        assert len(task["sets"]) == 1, "one set, its own artifact"
        entry = task["sets"][0]
        declared = {e["name"]: e for e in payload["inputs"]}
        if entry["init"] == "imagenet":
            init_input = declared["pretrained_init"]
            assert init_input["path"].endswith(
                f"init_{entry['backbone']}_v1"
            )
            assert init_input["rollup_sha256"] != placeholder, path.stem
            assert entry.get("pretrain_scheme") is None
        else:
            ckpt = [n for n in declared if n.startswith("ckpt_")]
            assert len(ckpt) == 1, path.stem
            text = path.read_text(encoding="utf-8")
            if declared[ckpt[0]]["rollup_sha256"] == placeholder:
                assert "DRAW-PICK RULE" in text, path.stem
            else:
                # A filled hash with a pending path would declare bytes at
                # a directory that does not exist -- the two resolve
                # together or the header says which half is missing.
                assert "/PENDING_" not in declared[ckpt[0]]["path"], path.stem
                assert declared[ckpt[0]]["path"].endswith("/pretrained.npz")
            if entry["backbone"] in ("srgnn", "agnet"):
                assert entry["pretrain_scheme"] == "native"
        # The artifact version carries the resolution; the set name does
        # not -- the recorded naming decision.
        short = "imagenet" if entry["init"] == "imagenet" else "masked"
        resolution = path.stem.replace("_detcheck", "").rsplit("_", 1)[1]
        assert task["out_version"].startswith(
            f"roadb_emb_{entry['backbone']}_{short}_g1_{resolution}"
        ), path.stem
        assert task["staged_artifact"] == {
            "224": "staged_v1",
            "512": "roadb_512_square_g1_v1",
            "768": "roadb_768_square_g1_v1",
        }[resolution], path.stem
        if "detcheck" in path.stem:
            assert task["out_version"].endswith("_detcheck_v1")
            assert "RUNS SECOND" in path.read_text(encoding="utf-8")
    # The queue leader carries its designation.
    leader = (repo_root / "configs" /
              "roadb_p7_extract_vit_b16_imagenet_512.yaml")
    assert "LEADS THE QUEUE" in leader.read_text(encoding="utf-8")

    # The two ViT masked cells are FULLY resolved -- hash and path -- at
    # the recorded picks; their headers point at the pick rationale.
    picks = roadb.VIT_DRAW_PICKS
    for resolution, pick in ((224, picks["vit_224"]), (512, picks["vit_512"])):
        payload = yaml.safe_load(
            (repo_root / "configs" /
             f"roadb_p7_extract_vit_b16_masked_{resolution}.yaml")
            .read_text(encoding="utf-8")
        )
        entry = {e["name"]: e for e in payload["inputs"]}[
            "ckpt_vit_b16_masked_g1"
        ]
        assert entry["rollup_sha256"].startswith(pick["ckpt_prefix"])
        assert pick["picked"] in entry["path"]
        assert picks["sha"] in entry["path"]
        assert "/PENDING_" not in entry["path"]
    # And both rationales say what a later reader needs.
    assert "COINCIDENTALLY" in picks["vit_224"]["note"]
    assert "never reads scores" in picks["vit_224"]["note"]
    assert "not a rule violation" in picks["vit_512"]["note"]
    assert "no checkpoint" in picks["vit_512"]["note"]

    # The confinement statement and the pinned confirmation are recorded.
    confirmed = roadb.PHASE_7_EXTRACTION_PLAN["extraction_determinism"][
        "confirmed"
    ]
    assert confirmed["values_md5"] == "916b430c57232cf35178962f667c4117"
    assert "one layer of draws, not two" in confirmed["confinement"]
    assert "237 patients" in roadb.PHASE_7_EXTRACTION_PLAN[
        "imagenet_sets_built"
    ]

    # And the grouping + graph-route records exist with their reasons.
    plan = roadb.PHASE_7_EXTRACTION_PLAN
    assert "ARTIFACT_PER_SETTING's argument" in plan["grouping"]
    assert "artifact VERSION, not the set name" in plan["grouping"]
    assert "strict both ways" in plan["graph_init_route"]


def test_the_phase7_arm_structure_settles_all_three_questions():
    """**24 arms, 8 families, the shape is the claim; the per-cell band
    rule dissolves on inspection; ViT is descriptive by decision; the null
    reading is registered before any arm runs.**"""
    record = roadb.PHASE_7_ARM_STRUCTURE
    assert record["arms"] == 24
    assert record["families"]["count"] == 8
    assert "peaked" in record["families"]["shape_is_the_claim"]
    assert "not headlined" in record["families"]["shape_is_the_claim"]
    assert record["init_contrasts"]["count"] == 12
    assert "DESCRIPTIVE" in record["init_contrasts"]["role"]
    assert "provenance-not-physics" in record["road_a_anchor"]

    # The band narrowing: the per-cell rule governed Phase 6 reads, and no
    # Phase 7 arm makes one; the minimum new measurement is ONE graph sweep.
    narrowing = record["band_narrowing"]
    assert "does NOT apply as written" in narrowing["per_cell_rule"]
    assert "ACROSS resolutions" in narrowing["per_cell_rule"]
    assert "ONE graph-head confirmation sweep" in (
        narrowing["minimum_measurement"]
    )
    assert "srgnn masked-512" in narrowing["minimum_measurement"]
    assert narrowing["transformer_arms"].startswith("zero new bands")
    # [AMENDED 2026-08-12] per_arm was one number (5); PHASE_7_SEED_COUNTS
    # settled it by regime, and the amendment keeps the old value visible.
    assert record["seeds"]["per_arm"] == {"transformer": 5, "graph": 10}
    assert record["seeds"]["was_per_arm"] == 5
    assert record["seeds"]["band"] == 0.021425

    # ViT: identical runs, different reporting, decided in advance -- and
    # the second variance layer named precisely.
    vit = record["vit_reporting"]
    assert vit["mode"].startswith("DESCRIPTIVE")
    assert "rides on top" in vit["second_layer"]
    assert "VIT_DRAW_PICKS" in vit["inline_caveat"]

    # The null reading, registered with its meaning.
    assert "real finding" in record["null_reading_registered"]
    assert "10.8x" in record["null_reading_registered"]
    assert "before any arm runs" in record["null_reading_registered"]

    # The arms as data: 24, 8 families of 3, ViT families carry no trend
    # claim, the reuse arm reads Road A's set.
    arms = roadb.phase7_arms()
    assert len(arms) == 24
    families = {}
    for arm in arms:
        families.setdefault(arm["family"], []).append(arm)
    assert len(families) == 8
    assert all(len(members) == 3 for members in families.values())
    for arm in arms:
        if arm["backbone"] == "vit_b16":
            assert arm["trend_claim"] is False
            assert "draw caveat" in arm["reporting"]
        else:
            assert arm["trend_claim"] is True
    reuse_arms = [a for a in arms if "bitwise reuse" in a["set_artifact"]]
    assert len(reuse_arms) == 1
    assert (reuse_arms[0]["backbone"], reuse_arms[0]["resolution"]) == (
        "swin_b", 224
    )


def test_the_seed_counts_follow_the_regime_not_the_claim_type():
    """**Transformer 5, graph 10 -- Road A's ladder rule kept, with Road
    B's numbers behind it -- and the trend/descriptive answer is
    structural: the contrasts own no arms.**"""
    from cleft import ladder

    record = roadb.PHASE_7_SEED_COUNTS
    # The rule IS Road A's rule -- if SEEDS_BY_KIND ever moves, this
    # decision must be re-read rather than silently diverge.
    assert record["per_regime"] == ladder.SEEDS_BY_KIND
    assert record["fits"]["total"] == 12 * 5 + 12 * 10 == 180
    assert record["fits"]["uniform_five"] == 120
    assert record["fits"]["uniform_ten"] == 240
    assert "0.027 -> 0.019" in record["transformer_five"]
    assert "resolve at neither count" in record["transformer_five"]
    assert "0.025053" in record["graph_ten"]
    assert "BE the srgnn masked-512 arm" in record["graph_ten"]
    assert "own no arms" in record["same_for_trend_and_descriptive"]
    assert "pairs of the same 24" in record["same_for_trend_and_descriptive"]
    assert "reuses what already ran" in record["extension_escape"]

    # The arms carry the decision: 12 five-seed transformer, 12 ten-seed
    # graph, and the counts come from the record rather than beside it.
    arms = roadb.phase7_arms()
    by_regime = {"transformer": set(), "graph": set()}
    for arm in arms:
        by_regime[arm["regime"]].add(arm["seeds"])
        assert arm["seeds"] == record["per_regime"][arm["regime"]]
    assert by_regime == {"transformer": {5}, "graph": {10}}
    assert sum(a["seeds"] for a in arms) == record["fits"]["total"]


def test_the_graph_band_sweep_is_preregistered_with_both_readings(repo_root):
    """**The sweep config exists, both readings are in its header before
    it runs, its comparator is Road A's recorded band, and its task block
    IS the srgnn masked-512 arm's** -- the doubling is dict equality, not
    intent."""
    import yaml

    from cleft.train import graph_cleft

    record = roadb.PHASE_7_GRAPH_BAND_SWEEP
    against = record["compared_against"]
    assert against["sd"] == graph_cleft.MEASURED_GRAPH_SEED_BAND["sd"]
    assert against["n_seeds"] == graph_cleft.MEASURED_GRAPH_SEED_BAND[
        "n_seeds"
    ]
    assert against["run"] == graph_cleft.MEASURED_GRAPH_SEED_BAND["run"]
    assert "LARGER of the two SDs" in record["rule"]["inside"]
    assert "proceed" in record["rule"]["inside"]
    assert "per-resolution graph bands" in record["rule"]["outside"]
    assert "re-reads" in record["rule"]["outside"]
    assert "~2x" in record["sensitivity"]
    assert "byte-identical across seeds" in record["seed_does_not_vary"]
    assert "bit-for-bit" in record["doubles_as"]

    sweep_path = repo_root / "configs" / f"{record['config']}.yaml"
    text = sweep_path.read_text(encoding="utf-8")
    # Both readings, registered in the header -- the Phase 3 discipline.
    assert "CI CONTAINS 1" in text
    assert "CI EXCLUDES 1" in text
    assert "REGISTERED BEFORE THE RUN" in text
    assert "p6-srgnn-seedband-1" in text
    assert "0.025053" in text
    # Post-declare-pass: every input hash real, said in the header.
    assert "READY TO LAUNCH" in text

    sweep = yaml.safe_load(text)
    arm = yaml.safe_load(
        (repo_root / "configs" / "roadb_p7_arm_srgnn_masked_512.yaml")
        .read_text(encoding="utf-8")
    )
    assert sweep["task"] == arm["task"], (
        "the sweep no longer doubles as the arm"
    )
    assert sweep["inputs"] == arm["inputs"]

    from cleft import ladder

    assert sweep["task"]["seeds"] == list(ladder.SEED_POOL), (
        "the F-comparison needs the graph band's own ten"
    )
    assert sweep["task"]["kind"] == "train_graph_cv"

    # The paired checkpoint is the extraction config's, verbatim.
    ex = yaml.safe_load(
        (repo_root / "configs" / "roadb_p7_extract_srgnn_masked_512.yaml")
        .read_text(encoding="utf-8")
    )
    ex_ckpt = [e for e in ex["inputs"] if e["name"].startswith("ckpt_")]
    declared = {e["name"]: e for e in sweep["inputs"]}
    assert declared["ckpt_srgnn_native_masked_g1"] == ex_ckpt[0]


def test_the_graph_band_reading_fired_and_its_cost_is_recorded(monkeypatch):
    """**The pre-registered rule applied: CI contains 1 -> one band, the
    larger (0.041452), graph arms proceed** -- with the claimable deltas
    re-derived through the project's own formula, not quoted."""
    from cleft.train import phase3
    from cleft.train.graph_cleft import MEASURED_GRAPH_SEED_BAND

    record = roadb.PHASE_7_GRAPH_BAND_RESOLVED
    assert record["applied_rule"] == "PHASE_7_GRAPH_BAND_SWEEP.rule"
    assert record["road_a_sd"] == MEASURED_GRAPH_SEED_BAND["sd"]
    assert record["n_seeds"] == MEASURED_GRAPH_SEED_BAND["n_seeds"]

    # The statistic, re-derived: ratio and the F(9,9) interval the rule
    # registered (observed ratio x [1/2.01, 2.01]).
    ratio = record["road_b_sd"] / record["road_a_sd"]
    assert abs(ratio - record["ratio"]) < 5e-4
    low, high = record["f_interval_95"]
    assert abs(low - ratio / 2.01) < 5e-3
    assert abs(high - ratio * 2.01) < 5e-3
    assert low < 1 < high
    assert record["contains_one"] is True
    # The registered INSIDE branch: one band, the larger of the two.
    assert record["planning_band"] == max(
        record["road_b_sd"], record["road_a_sd"]
    ) == 0.041452
    assert record["graph_arms"].startswith("PROCEED")

    # Claimable deltas come from train/phase3, not from arithmetic here.
    for n, quoted in record["claimable_delta"].items():
        assert abs(
            phase3.claimable_delta(n, record["planning_band"]) - quoted
        ) < 5e-4, n

    # The cost, stated before the arms run.
    assert "0.02-0.08" in record["cost"]
    assert "before they run" in record["cost"]
    assert "SHAPE claim" in record["does_not_change"]
    # Escalation past ten is a DESIGN change, not a config edit, and the
    # code is what says so -- exercised rather than asserted in prose.
    from cleft import ladder

    assert len(ladder.SEED_POOL) == 10
    assert max(ladder.SEEDS_BY_KIND.values()) <= len(ladder.SEED_POOL)
    monkeypatch.setitem(
        ladder.SEEDS_BY_KIND, "graph", len(ladder.SEED_POOL) + 1
    )
    with pytest.raises(ladder.LadderError):
        ladder.seeds_for("graph")
    assert "new registered seeds" in record["escalation"]


def test_the_224_anchor_points_carry_what_each_one_actually_tests():
    """**Four 224 transformer points, three different kinds** -- two
    genuine vintage anchors that passed, one tautological identity, one
    that consumes a different checkpoint and tests nothing."""
    from cleft import ladder

    record = roadb.PHASE_7_ANCHOR_INDEPENDENCE

    # Anchor 1: ViT, against Road A's FULL-PRECISION value, not a
    # rounding of it.
    vit = record["vit_imagenet_224"]
    assert vit["road_a"] == ladder.CLEAN_GEOMETRY_MEASUREMENTS[
        "imagenet"
    ]["g1"]["mean"] == 0.25206
    assert vit["road_a_own_sd"] == ladder.CLEAN_GEOMETRY_MEASUREMENTS[
        "imagenet"
    ]["g1"]["sd"]
    assert abs(vit["road_b"] - vit["road_a"]) < 1e-4
    # The rounding caveat is real: Road A's own records disagree in the
    # fourth decimal, which is why "four decimals" was the wrong frame.
    d1 = ladder.STAGE_D1_AT_G1
    assert d1["cells"]["vit_b16"][d1["inits"].index("imagenet")] == 0.2520
    assert "rounding" in vit["rounding_caveat"]
    assert "0.25206" in vit["rounding_caveat"]

    # Anchor 2: Swin, exact at the recorded precision -- two backbones,
    # so one agreeing point is not carrying the vintage rule alone.
    swin = record["swin_imagenet_224"]
    assert swin["road_a"] == d1["cells"]["swin_b"][
        d1["inits"].index("imagenet")
    ] == 0.1076
    assert swin["road_b"] == swin["road_a"]
    assert swin["road_a_own_sd"] == d1["sd"]["swin_b"]["imagenet"]

    # Not an anchor: Road B's masked 224 consumes its OWN checkpoint.
    not_anchor = record["vit_masked_224_is_NOT_an_anchor"]
    assert not_anchor["road_a"] == d1["cells"]["vit_b16"][
        d1["inits"].index("scut_masked")
    ] == 0.0830
    assert "never expected to reproduce" in not_anchor["why_not"]
    assert "0.055" in not_anchor["why_not"]

    # Tautological: identical config, so identity is guaranteed.
    reuse = record["swin_masked_224"]
    assert reuse["value"] == d1["cells"]["swin_b"][
        d1["inits"].index("scut_masked")
    ] == 0.1327
    assert reuse["sd"] == d1["sd"]["swin_b"]["scut_masked"] == 0.0267
    assert "not evidence" in reuse["why_identical"]
    assert "independent evidence FOR Road B" in record["reporting_rule"]
    assert "both transformer anchors" in record["provenance_verdict"]


def test_the_reuse_arm_is_literally_road_as_run(repo_root):
    """The swin-224 duplicate explained at the bytes: the Road B arm's
    task and inputs are dict-equal to Road A's Stage C config, so two
    arms reporting one number is one computation, not a coincidence."""
    import yaml

    def load(stem):
        return yaml.safe_load(
            (repo_root / "configs" / f"{stem}.yaml").read_text(
                encoding="utf-8"
            )
        )

    road_b = load("roadb_p7_arm_swin_b_masked_224")
    road_a = load("p7_c_swin_b_scut_masked_g1")
    assert road_b["task"] == road_a["task"]
    assert road_b["inputs"] == road_a["inputs"]
    # Only the phase differs -- which is what makes it a Road A run under
    # a Road B name rather than a Road B measurement.
    assert road_b["phase"] != road_a["phase"]
    assert "dict-equal" in roadb.PHASE_7_ANCHOR_INDEPENDENCE[
        "swin_masked_224"
    ]["config_identity"]


def test_the_first_transformer_arms_are_recorded_without_being_read():
    """**Ten values on the record, no shape claimed** -- the hold is
    explicit and its reason is recorded rather than left as silence."""
    record = roadb.PHASE_7_FIRST_TRANSFORMER_ARMS
    assert record["arms_in"] == 10
    assert record["arms_outstanding"] == 13
    assert record["arms_in"] + record["arms_outstanding"] == 23  # 24 - reuse

    values = record["values"]
    families = {a["family"] for a in roadb.phase7_arms()}
    assert set(values) <= families, "a family name that is not an arm family"
    # The two masked families are pending their 768 point; the two
    # imagenet families are complete.
    for family, points in values.items():
        assert set(points) == {224, 512, 768}
        if family in record["complete_families"]:
            assert all(v is not None for v in points.values()), family
        else:
            assert points[768] is None, family
    assert len(record["complete_families"]) == 2

    # The hold, and its reason, on the record.
    assert "held at instruction" in record["not_a_trend_reading"]
    assert "outstanding" in record["not_a_trend_reading"]
    assert "change what the design can say" in record[
        "report_eligible_but_held"
    ]
    # The three observations stay descriptions: attention, not dismissal,
    # and the negative recorded flatly with its draw caveat.
    obs = record["observations"]
    assert "frozen cliff" in obs["vit_falls_above_224"]
    assert "rather than dismissal" in obs["swin_imagenet_rises_at_512"]
    assert "one family of eight" in obs["swin_imagenet_rises_at_512"]
    assert "draw caveat" in obs["vit_masked_512_is_negative"]
    assert values["vit_b16__scut_masked"][512] < 0


def test_every_recorded_contrast_re_derives_from_the_arms_own_sds():
    """**The corrections that matter are arithmetic, so they are checked
    as arithmetic** -- every recorded delta, threshold and sigma comes
    back out of ``phase3.combined_claimable_delta`` from the two arms'
    own SDs, never from the planning band."""
    from cleft.train.phase3 import combined_claimable_delta

    record = roadb.PHASE_7_TWENTY_TWO_ARMS
    values = record["values"]
    assert record["arms"] == sum(len(v) for v in values.values()) == 22
    assert len(values) == 8
    complete = [f for f, points in values.items() if len(points) == 3]
    assert len(complete) == record["complete_families"] == 6

    n_of = {
        family: 10 if family.split("__")[0] in ("srgnn", "agnet") else 5
        for family in values
    }

    def contrast(family, lo, hi):
        (mean_a, sd_a), (mean_b, sd_b) = values[family][lo], values[family][hi]
        n = n_of[family]
        t = combined_claimable_delta(sd_b, n, sd_a, n)
        delta = mean_b - mean_a
        return delta, t["arm_means_95"], delta / (t["arm_means_95"] / 1.96)

    # Endpoint contrasts: recorded triples must re-derive exactly.
    for family, (delta, threshold, sigma) in record[
        "endpoint_contrasts_224_to_768"
    ].items():
        d, t, s = contrast(family, 224, 768)
        assert abs(d - delta) < 5e-5, family
        assert abs(t - threshold) < 5e-4, family
        assert abs(s - sigma) < 5e-2, family

    # Correction one: three of four graph families clear on their OWN
    # SDs, where the 0.036 planning band would find one.
    graph = [f for f in complete if f.split("__")[0] in ("srgnn", "agnet")]
    assert len(graph) == 4
    clearing = [
        f for f in graph
        if abs(record["endpoint_contrasts_224_to_768"][f][0])
        > record["endpoint_contrasts_224_to_768"][f][1]
    ]
    assert len(clearing) == 3, clearing
    assert "agnet__imagenet" not in clearing  # the shape through noise
    band = roadb.PHASE_7_GRAPH_BAND_RESOLVED["claimable_delta"][10]
    against_band = [
        f for f in graph
        if abs(record["endpoint_contrasts_224_to_768"][f][0]) > band
    ]
    assert len(against_band) == 3
    # ...and the record says WHY the band is not the denominator.
    assert "4.12.1" in record["band_is_not_the_denominator"]
    assert "OWN SDs" in record["band_is_not_the_denominator"]

    # Correction two: Swin is a family, not a cell -- both non-224
    # points clear, and 512 also clears the conservative companion.
    for resolution, expected_sigma in ((512, 7.16), (768, 2.77)):
        d, t, s = contrast("swin_b__imagenet", 224, resolution)
        assert d > t > 0, resolution
        assert abs(s - expected_sigma) < 5e-2, resolution
    swin = roadb.PHASE_7_SWIN_512["observed"]
    single_run = combined_claimable_delta(
        values["swin_b__imagenet"][512][1], 5,
        values["swin_b__imagenet"][224][1], 5,
    )["single_run_95"]
    assert abs(single_run - swin["single_run_95"]) < 5e-4
    assert swin["delta_vs_own_224"] > single_run, (
        "the record claims it survives without averaging"
    )

    # Three of fourteen cells sit above their own 224; the two that
    # clear are both swin imagenet's.
    above, above_clearing = [], []
    for family, points in values.items():
        for resolution in (512, 768):
            if resolution not in points:
                continue
            d, t, _s = contrast(family, 224, resolution)
            if d > 0:
                above.append((family, resolution))
                if d > t:
                    above_clearing.append((family, resolution))
    assert len(above) == 3
    assert above_clearing == [
        ("swin_b__imagenet", 512), ("swin_b__imagenet", 768),
    ]


def test_the_paired_results_are_recorded_with_what_they_do_not_show():
    """**Three survive, seventeen withdraw** -- and the record separates
    the clean pair from the draw-confounded one, and this set from Road
    A's lone caveated survivor."""
    from cleft import ladder

    record = roadb.PHASE_7_PAIRED_RESULTS
    assert record["survived"] == 3
    assert record["withdrawn"] == 17
    assert record["survived"] + record["withdrawn"] == len(
        roadb.paired_claim_pairs("roadb_resolution")
    ) == 20
    assert len(record["survivors"]) == 3
    assert all(
        s["contrast"].startswith("vit_b16") for s in record["survivors"]
    ), "every survivor is ViT -- the record says so and must stay true"
    assert all(s["delta"] < 0 for s in record["survivors"]), "all falling"
    assert all(s["excluding"] == "5/5" for s in record["survivors"])

    # The confound is scoped to the ONE masked survivor: imagenet arms
    # read a single declared snapshot at every resolution.
    confounded = [s for s in record["survivors"] if s["confounded_by_draw"]]
    assert len(confounded) == 1
    assert "masked" in confounded[0]["contrast"]
    assert "DECLARED snapshot" in record[
        "draw_spread_does_not_enter_the_clean_pair"
    ]
    assert "ViT-specific" in record["draw_spread_DOES_touch_the_masked_survivor"]

    # Not the first ever: Road A's audit had one, and its caveats are
    # quoted from Road A's own record rather than paraphrased.
    audit = ladder.LADDER_PAIRED_AUDIT
    assert audit["survived"] == 1
    assert "0.0065" in audit["survivor"]["quote_it_with"]
    assert "LADDER_PAIRED_AUDIT" in record["not_the_first_ever"]
    assert "COHERENT" in record["not_the_first_ever"]
    # And the margin range: Road A's two highest failed, at 4.69 and 4.34.
    highest = audit["margin_does_not_predict_condition_1"][
        "two_highest_margins_both_fail"
    ]
    assert {entry["margin"] for entry in highest} == {4.69, 4.34}
    assert min(s["margin"] for s in record["survivors"]) > max(
        entry["margin"] for entry in highest
    ), "the record claims these sit above anything Road A reached"


def test_the_withdrawn_gain_and_the_confirmed_prediction_stay_separate():
    """**Two statements about Swin, only one of which moved.** The gain
    is withdrawn; the pre-registered non-collapse stands, and the record
    refuses to let either one stand in for the other."""
    outcome = roadb.PHASE_7_SWIN_512["outcome"]

    gain = outcome["the_gain"]
    assert gain["verdict"] == "WITHDRAWN"
    assert gain["condition_2"] == "3.66x"
    assert "1 of 5" in gain["condition_1"]
    assert "NOT claimable at n=237" in gain["reading"]
    # It failed in the band where Road A's 3.62x failed and 3.83x lived.
    from cleft import ladder

    assert ladder.LADDER_PAIRED_AUDIT["survivor"]["margin"] == 3.83
    assert "3.62x" in gain["context"] and "3.83x" in gain["context"]

    prediction = outcome["the_prediction"]
    assert prediction["verdict"].startswith("STANDS")
    assert "collapse" in prediction["what_it_said"]
    assert "asymmetry" in prediction["what_the_criterion_returned"]
    # The honesty clause: Swin's side is a null, not a positive result.
    assert "UNRESOLVED NULL" in prediction["stated_honestly"]
    assert "not proof of no change" in prediction["stated_honestly"]
    assert "launder a selected maximum" in outcome["why_they_must_not_merge"]

    # The G2 proposal is withdrawn as advice, with its reason.
    assert "cannot resolve a gain of that size" in roadb.PHASE_7_SWIN_512[
        "g2_replication_superseded"
    ]


def test_the_seed_vs_cohort_pattern_is_one_record_not_three():
    """**Fifth instance, and the first predicted in advance.**"""
    record = roadb.SEED_THRESHOLDS_DO_NOT_PREDICT_THE_COHORT
    assert record["instance"] == 5
    # The three Road B withdrawals that looked resolvable on seed SDs.
    looked = record["road_b_withdrawals_that_looked_resolvable"]
    assert len(looked) == 3
    assert all("withdrawn" in text for text in looked.values())
    assert "called IN ADVANCE" in record["what_is_new"]
    assert "PHASE_7_SWIN_512" in record["what_is_new"]
    assert "patient-level disagreement" in record["mechanism"]
    assert "ONE pattern, not as N separate" in record["reporting_rule"]

    # The prior instances name real records, and those records agree.
    from cleft import ladder, phase7c

    assert phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT[
        "intervals_excluding_zero"
    ] == "0 of 45"
    assert ladder.LADDER_PAIRED_AUDIT["withdrawn"] == 25


def test_what_road_b_established_states_both_halves():
    """**One claim, one null, and an explicit list of what it does not
    say** -- written so the write-up cannot drift past the evidence."""
    record = roadb.ROAD_B_ESTABLISHED
    assert record["arms"] == {"designed": 24, "ran": 22, "contrasts_tested": 20}
    claimable = record["claimable"]
    assert claimable["contrasts"] == 3
    assert "HURT" in claimable["finding"]
    assert all(delta < 0 for delta in claimable["deltas"].values())
    assert "anticipated FLATNESS" in claimable["stronger_than_registered"]
    assert "17 of 20" in record["not_claimable"]
    assert "NO case where the pixels help" in record["not_claimable"]
    assert "n=237" in record["premise_answered"]
    # The four things it must not be read as saying.
    for phrase in (
        "NOT that the pixels are uninformative",
        "NOT that resolution never helps",
        "NOT that the TRAINED regime",
    ):
        assert phrase in record["does_not_say"]
    assert "29 of 30" in record["the_finding_underneath"]
    # The registered null said flat; the measurement is stronger, and the
    # registration it supersedes is still readable.
    assert "flat at cleft time" in roadb.PHASE_7_ARM_STRUCTURE[
        "null_reading_registered"
    ]


def test_the_closing_decision_answers_both_questions_on_evidence():
    """**Reportable now; nothing further worth running** -- each on the
    evidence rather than on tidiness, including a recommendation
    reversed by the result."""
    record = roadb.PHASE_7_CLOSING
    assert record["reportable_now"] is True
    why = record["why_the_pending_arms_cannot_change_it"]
    assert "DESCRIPTIVE by pre-run decision" in why
    assert "3.08x" in why
    assert "do not wait" in record["how_to_report_them"]

    further = record["nothing_further_is_worth_running"]
    assert "optional completeness" in further["minus_3_cells"]
    assert further["g2_replication"].startswith("WITHDRAWN")
    assert "second unresolved null" in further["g2_replication"]
    assert "DIFFERENT question" in further["non_square_artifacts"]
    # The one thing worth doing costs nothing and could still say
    # something new.
    assert "no GPU, no new fits" in record["the_one_thing_worth_doing"]
    assert "two code SHAs" in record["the_one_thing_worth_doing"]
    assert set(roadb.PAIRED_CLAIM_DUPLICATE_RUNS["duplicates"]) == {
        "srgnn_masked_512", "swin_masked_224",
    }
    # The two arms the closing decision reports as unrun are the two the
    # arm record still lists as pending -- both masked 768 transformers.
    pending = roadb.PHASE_7_TWENTY_TWO_ARMS["pending"]
    assert len(pending) == 2
    assert all("768" in cell for cell in pending)
    assert {cell.split("__")[0] for cell in pending} == {"swin_b", "vit_b16"}


def test_the_duplicate_runs_bound_the_projects_nondeterminism():
    """**Fifteen files byte-identical across two scheduling accidents** --
    which confines nondeterminism to pretraining and rules out the
    reading that the graph band measures flakiness."""
    record = roadb.DUPLICATE_RUNS_WERE_A_DETERMINISM_CHECK
    assert record["files"] == 15
    assert record["result"] == "byte-identical on both pairs"
    pairs = record["pairs"]
    # Ten graph seeds plus five transformer seeds is the fifteen.
    assert pairs["srgnn_masked_512"]["seeds"] + pairs["swin_masked_224"][
        "seeds"
    ] == record["files"]
    assert set(pairs) == set(
        roadb.PAIRED_CLAIM_DUPLICATE_RUNS["duplicates"]
    ), "the check covers exactly the duplicates the routing record named"

    establishes = record["establishes"]
    # 1. The confinement, and it needs BOTH prior measurements to hold.
    confinement = establishes["nondeterminism_is_confined_to_pretraining"]
    assert "extraction measured bitwise earlier" in confinement
    assert "head fitting bitwise" in confinement
    assert "reproducible even where the checkpoints" in confinement
    assert "one layer of draws, not two" in roadb.PHASE_7_EXTRACTION_PLAN[
        "extraction_determinism"
    ]["confirmed"]["confinement"]
    # 2. The band means seed variance -- the alternative reading named
    #    and eliminated, not merely asserted.
    band = establishes["the_graph_band_is_seed_variance"]
    assert "flakiness" in band
    assert "DIFFERENT seeds do" in band
    assert str(roadb.PHASE_7_GRAPH_BAND_RESOLVED["planning_band"]) in band


def test_the_closing_record_states_all_four_parts(repo_root):
    """**Claimable, withdrawn, never run, established** -- Phase 8's
    closing shape, with the flags that must travel with each number."""
    from cleft import phase8

    record = roadb.ROAD_B_PHASE_7_CLOSING
    # It is named apart from the DECISION record on purpose (R2).
    assert "DECISION" in record["distinct_from"]
    assert "STATEMENT" in record["distinct_from"]
    assert roadb.PHASE_7_CLOSING["reportable_now"] is True

    # Same shape as Phase 8's closing -- a dated close, a status line,
    # and a guard against the write-up overreaching.
    assert {"closed", "status"} <= set(record)
    assert {"closed", "status"} <= set(phase8.PHASE_8_CLOSING)
    assert "do_not_write_up_as" in record
    assert "do_not_write_up_as_complete" in phase8.PHASE_8_CLOSING

    # CLAIMABLE: three, all ViT, and the confound flag is IN the entry
    # rather than in a footnote elsewhere.
    assert len(record["claimable"]) == 3
    assert all("vit_b16" in entry for entry in record["claimable"])
    confounded = [e for e in record["claimable"] if "CONFOUNDED" in e]
    assert len(confounded) == 1 and "masked" in confounded[0]
    assert "travels with the number" in confounded[0]
    assert record["the_claim"].endswith("more pixels measurably HURT")

    # WITHDRAWN: seventeen, with the gain and the prediction kept apart.
    withdrawn = record["withdrawn"]
    assert withdrawn["count"] == 17
    assert withdrawn["count"] + len(record["claimable"]) == 20
    assert "3.66x" in withdrawn["swin_gain"]
    assert "not claimable at n=237" in withdrawn["swin_gain"]
    assert "unresolved null" in withdrawn["swin_prediction_stands_separately"]
    assert "pre-registered prediction held" in withdrawn[
        "swin_prediction_stands_separately"
    ]
    assert "ONE instance of a five-instance pattern" in withdrawn[
        "graph_declines"
    ]

    # NEVER RUN: named, with why, and marked decided rather than pending.
    never = record["never_run"]
    assert set(never["cells"]) == {"swin_b masked-768", "vit_b16 masked-768"}
    assert "could never have carried a claim" in never["why"]
    assert "DESCRIPTIVE by pre-run decision" in never["why"]
    assert "3.08x" in never["why"]
    assert never["decided_not_deferred"] == "2026-08-12"
    assert "NOT BEING RUN" in roadb.PHASE_7_CLOSING[
        "nothing_further_is_worth_running"
    ]["minus_3_cells"]

    # ESTABLISHES: the premise, the answer, and the sentence.
    establishes = record["establishes"]
    assert "99% of available pixels" in establishes["premise"]
    assert "demonstrably hurts" in establishes["answer"]
    assert "anticipated FLATNESS" in establishes["null_fired_stronger"]
    assert "nothing smaller" in establishes["the_sentence"]
    assert "next dataset has to beat" in establishes["the_sentence"]
    assert "NO intervals anywhere" in establishes["comparator_context"]

    # The guard names the four misreadings, including the confound one.
    guard = record["do_not_write_up_as"]
    for phrase in (
        "NOT 'the pixels are uninformative'",
        "NOT 'resolution never helps'",
        "TRAINED regime",
        "without its confound flag",
    ):
        assert phrase in guard


def test_the_crop_geometry_survives_the_aspect_the_frozen_check_rejects():
    """**The measured landmine, pinned**: the frozen containment check
    bounds both axes by ``shape[0]`` and rejects a legitimate box on a
    landscape frame. Branch 3's own check bounds per axis and does not."""
    import numpy as np

    from cleft import roadb_regioncrop as rc
    from cleft.geometry import mapping
    from cleft.roadb_staging import stage_non_square

    patches = rc.protected_patches()
    assert len(patches) == 22

    landscape = np.zeros((2173, 2424, 3), dtype=np.uint8)
    staged = stage_non_square(landscape, 768)
    height, width = staged.image.shape[:2]
    assert width > height, "the case the frozen check gets wrong"

    # The frozen path REJECTS this image -- pinned, so if mapping.py is
    # ever unfrozen and fixed, this test says so rather than passing on.
    with pytest.raises(mapping.MappingError, match="falls outside"):
        mapping.map_patches(staged, patches)

    # Branch 3's own path accepts it, and every box really is inside.
    boxes = rc.region_boxes(staged, patches)
    assert len(boxes) == 22
    for x, y, w, h in boxes:
        assert 0 <= x and x + w <= width
        assert 0 <= y and y + h <= height

    # Portrait and the extreme tall aspect work too -- the cohort spans
    # 0.553 to 1.099, so all three regimes are real.
    for shape in ((2424, 2173), (2170, 1200)):
        other = stage_non_square(np.zeros((*shape, 3), np.uint8), 768)
        crops = rc.crops_for(other, patches)
        assert crops.shape == (22, rc.CROP_OUTPUT_SIZE,
                               rc.CROP_OUTPUT_SIZE, 3)

    record = roadb.FROZEN_CONTAINMENT_IS_SQUARE_ONLY
    assert "bounds BOTH axes" in record["what"]
    assert "aspect-correct" in record["the_arithmetic_is_fine"]
    assert "236 of 237" in record["failure_shape"]
    assert "die on the last" in record["failure_shape"]
    # The cohort fact the failure shape rests on, from its own record.
    from cleft.scut import ar_distribution

    assert ar_distribution.CLEFT_AR["n_landscape"] == 1
    assert ar_distribution.CLEFT_AR["n"] == 237
    assert ar_distribution.CLEFT_AR["max"] > 1.0, (
        "landscape means aspect above 1 -- if the cohort record ever "
        "says otherwise, the failure shape above is stale"
    )


def test_train_cv_routes_through_as_consumed_and_reports_the_head_width(
    tmp_path,
):
    """**The wiring, end to end for the concat arms**: a region_vectors
    set reaches a linear head as (n, regions*dim), and the parameter
    check sees the width the HEAD sees."""
    import numpy as np

    from cleft import embeddings
    from cleft.train.pooled import PooledSource, load_features

    names = [f"random_{i:02d}" for i in range(22)]
    values = np.zeros((3, 22, 8), dtype=np.float32)
    directory = tmp_path / "set"
    embeddings.save(
        directory, values, backbone="vit_b16", backbone_kind="transformer",
        init="imagenet", geometry="g2", variant=None,
        checkpoint_sha256=None, patient_ids=[1, 2, 3], regions=names,
        kind="region_vectors",
    )

    source = PooledSource(
        directory=directory, backbone="vit_b16", init="imagenet",
        geometry="g2", consume="concat",
    )
    features, report = load_features(source, [1, 2, 3])
    assert features.shape == (3, 22 * 8)
    # THE detail that would otherwise fire the D+1 check on a correct arm.
    assert report["feature_dim"] == 176
    assert report["artifact_feature_dim"] == 8
    assert report["consume"] == "concat"

    # Default stays pooled, so every Road A arm is untouched.
    assert PooledSource(
        directory=directory, backbone="vit_b16", init="imagenet",
        geometry="g2",
    ).consume == "pooled"
    # ...and a pooled arm pointed at a region_vectors set is refused
    # rather than silently flattened.
    from cleft.train.pooled import PooledFeatureError

    with pytest.raises(PooledFeatureError):
        load_features(
            PooledSource(
                directory=directory, backbone="vit_b16", init="imagenet",
                geometry="g2", consume="pooled",
            ),
            [1, 2, 3],
        )


def test_an_artifact_fed_arm_never_reads_the_staged_tensor():
    """**The blocker, and the static check that keeps the fix safe.**

    ``phase3.run`` loaded the stacked staged tensor unconditionally, which
    Road B Branch 3's non-square staging can never produce. The tensor is
    unread on the artifact path -- proven here by walking the function, so
    a future use cannot silently receive ``None``.
    """
    import ast
    from pathlib import Path as _Path

    from cleft.train import phase3

    source = _Path(phase3.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    run_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    uses = [
        node.lineno for node in ast.walk(run_fn)
        if isinstance(node, ast.Name) and node.id == "images"
        and isinstance(node.ctx, ast.Load)
    ]
    lines = source.splitlines()
    for lineno in uses:
        window = "\n".join(lines[max(0, lineno - 12):lineno])
        assert (
            "features_override is not None" in window
            or "prepare_features(" in lines[lineno - 2:lineno + 1][0]
            or "prepare_features" in window
        ), (
            f"images is read at line {lineno} on a path that may be the "
            "artifact one; require_images=False would pass None there"
        )

    # The skip is conditional on the arm being artifact-fed AND not using
    # the override, which is the branch that counts image rows.
    run_src = ast.get_source_segment(source, run_fn) or ""
    assert (
        "needs_images = pooled_source is None or features_override is not None"
        in run_src
    )
    assert "require_images=needs_images" in run_src

    # load_inputs still asserts row ORDER without the pixels -- the skip
    # drops a read, not a guarantee.
    load_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_inputs"
    )
    load_src = ast.get_source_segment(source, load_fn) or ""
    assert "geometry_csv" in load_src
    assert "row order does not match the manifest" in load_src
    assert "features is not None and len(features)" in load_src

    record = roadb.ARTIFACT_ARMS_DO_NOT_READ_PIXELS
    assert "exactly three places" in record["measured"]
    assert "None is on the artifact path" in record["measured"]
    assert "checkable" in record["declaration_kept"]
    assert "no number can move" in record["fix_is_in_phase3"]
    assert "still runs" in record["drops_a_read_not_a_guarantee"]


def test_the_branch3_arms_still_declare_their_staging(repo_root):
    """**The declaration stays, and it is not for the pixels.** It carries
    provenance -- which staging the crops were cut from, guard-3 verified
    -- and geometry.csv, whose row-order assertion still runs."""
    import yaml

    staging = roadb.REGION_CROP_STRUCTURE["staging"]["artifact"]
    for path in sorted((repo_root / "configs").glob("roadb_p7c_arm_*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        declared = {e["name"]: e for e in payload["inputs"]}
        assert payload["task"]["staged_artifact"] == staging, path.stem
        assert staging in declared, path.stem
        assert declared[staging]["rollup_sha256"] != "0" * 64, path.stem

    # Both sides of crop_vs_whole declare the SAME staging, which is the
    # matched-pipeline claim the corrected control exists to make -- and
    # it is only checkable because the declaration was kept.
    def staged_of(key):
        payload = yaml.safe_load(
            (repo_root / "configs" / f"roadb_p7c_arm_{key}.yaml")
            .read_text(encoding="utf-8")
        )
        return {e["name"]: e for e in payload["inputs"]}[staging]

    assert staged_of("control_whole_vit") == staged_of("anatomy_concat_vit")


def test_the_graph_node_path_refuses_at_load_with_what_remains():
    """**Not wired, and it says so before a queue wait.** The gap is
    structural, so it is scoped rather than guessed."""
    import ast

    from cleft.config.schema import TASK_SPECS

    record = roadb.GRAPH_NODE_PATH_IS_NOT_WIRED
    assert record["not_wired"] == "train_graph_cv with consume: nodes"
    assert "pack requires" in record["why_not_a_reshape"]
    assert "second construction path" in record["why_not_a_reshape"]
    assert "may run" in record["unaffected"]
    assert "R10" in record["why_scoped_not_guessed"]

    # The refusal exists in the task, not merely in the record.
    from pathlib import Path as _Path

    source = (
        _Path(roadb.__file__).with_name("run.py").read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    task_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "task_train_graph_cv"
    )
    body = ast.get_source_segment(source, task_fn) or ""
    assert "consume\") == \"nodes\"" in body or "'nodes'" in body
    assert "GRAPH_NODE_PATH_IS_NOT_WIRED" in body

    # The schema doc no longer says feature_map only -- the mismatch that
    # would have surfaced on the cluster.
    doc = TASK_SPECS["train_graph_cv"]["embeddings_artifact"].doc
    assert "region_vectors" in doc
    assert "feature_map" in doc


def test_the_consumption_layer_reshapes_and_refuses(tmp_path):
    """**One artifact, three readings, declared not inferred** -- and the
    flatten order is part of the contract because a flattened vector is
    only interpretable through its region names."""
    import numpy as np

    from cleft import embeddings

    names = [f"random_{i:02d}" for i in range(22)]
    values = np.arange(2 * 22 * 4, dtype=np.float32).reshape(2, 22, 4)
    region_meta = {"kind": "region_vectors", "regions": names}
    pooled_meta = {"kind": "pooled", "regions": None}

    # nodes: as stored. concat: C order, so block i IS region i.
    assert embeddings.as_consumed(values, region_meta, "nodes") is values
    flat = embeddings.as_consumed(values, region_meta, "concat")
    assert flat.shape == (2, 88)
    for index in range(22):
        assert np.array_equal(flat[:, index * 4:(index + 1) * 4],
                              values[:, index, :]), (
            "block i must be region i -- the order the record documents"
        )

    # The refusals are the point: a mismatch is loud, not silent.
    with pytest.raises(embeddings.EmbeddingError, match="needs a pooled"):
        embeddings.as_consumed(values, region_meta, "pooled")
    with pytest.raises(embeddings.EmbeddingError, match="region_vectors"):
        embeddings.as_consumed(
            np.zeros((2, 4), np.float32), pooled_meta, "concat"
        )
    with pytest.raises(embeddings.EmbeddingError, match="unknown consumption"):
        embeddings.as_consumed(values, region_meta, "flatten")
    # A set whose names do not describe its columns cannot be indexed.
    with pytest.raises(embeddings.EmbeddingError, match="cannot be indexed"):
        embeddings.as_consumed(
            values, {"kind": "region_vectors", "regions": names[:3]}, "concat"
        )
    # pooled passes through untouched.
    pooled = np.zeros((2, 4), np.float32)
    assert embeddings.as_consumed(pooled, pooled_meta, "pooled") is pooled

    record = roadb.REGION_SET_CONSUMPTION
    assert "as_consumed" in record["where"]
    assert "the consumer" in record["who_decides"]
    assert "report a number" in record["why_declared_not_inferred"]
    assert "VIEW" in record["why_not_at_extraction"]
    assert "C order" in record["flatten_order"]
    assert set(record["readings"]) == set(embeddings.CONSUMPTIONS)


def test_the_five_branch3_arms_declare_how_they_read_their_sets(repo_root):
    """Five arms, four crop sets, and every one names its reading."""
    import yaml

    from cleft import ladder

    arms = {a["key"]: a for a in roadb.regioncrop_arms()}
    assert len(arms) == 5
    expected = {
        "control_whole_vit": ("train_cv", "pooled", 5),
        "anatomy_concat_vit": ("train_cv", "concat", 5),
        "random_concat_vit": ("train_cv", "concat", 5),
        "anatomy_concat_srgnn": ("train_cv", "concat", 10),
        "anatomy_graph_srgnn": ("train_graph_cv", "nodes", 10),
    }
    for key, (kind, consume, seeds) in expected.items():
        payload = yaml.safe_load(
            (repo_root / "configs" / f"roadb_p7c_arm_{key}.yaml")
            .read_text(encoding="utf-8")
        )
        task = payload["task"]
        assert payload["phase"] == "roadb_p7c"
        assert task["kind"] == kind, key
        assert task["consume"] == consume, key
        assert task["seeds"] == list(ladder.SEED_POOL[:seeds]), key
        assert task["backbone"] == arms[key]["backbone"], key
        assert task["geometry"] == "g2", key
        assert task["staged_artifact"] == arms[key]["staging"], key
        # Every arm declares an embeddings artifact -- an srgnn train_cv
        # arm without one is refused at load time, by the same rule that
        # covers swin_b.
        declared = {e["name"]: e for e in payload["inputs"]}
        assert "embeddings" in declared, key

    # The two SR-GNN arms read the SAME set: the combination contrast
    # varies combination alone, which is the whole reason arm 5 exists.
    def emb(key):
        payload = yaml.safe_load(
            (repo_root / "configs" / f"roadb_p7c_arm_{key}.yaml")
            .read_text(encoding="utf-8")
        )
        return {e["name"]: e for e in payload["inputs"]}["embeddings"]["path"]

    assert emb("anatomy_graph_srgnn") == emb("anatomy_concat_srgnn")
    # ...and the anatomy/random pair reads DIFFERENT sets, since placement
    # is the factor.
    assert emb("anatomy_concat_vit") != emb("random_concat_vit")

    # The re-review is recorded with why it was worth doing.
    review = roadb.CROP_SHEETS_RE_REVIEWED
    assert "asserted something false" in review["why"]
    assert "layout-independent" in review["verdicts_unchanged"]
    assert "misleading-captions" in review["what_changed"]


def test_the_random_names_are_positional_free_and_the_boxes_still_match():
    """**The collision was a LABEL defect, not a construction one.** The
    boxes were matched throughout; only the side suffix -- an anatomical
    fact -- was being computed from a randomised position."""
    from cleft import roadb_regioncrop as rc
    from cleft.geometry.render import abbreviate

    anatomy = rc.protected_patches()
    random_boxes = rc.random_patches(anatomy)

    # The construction was never wrong: this is the evidence, re-derived.
    assert [p.band for p in anatomy] == [p.band for p in random_boxes]
    assert [(p.w, p.h) for p in anatomy] == [(p.w, p.h) for p in random_boxes]
    assert rc.area_matches(anatomy, random_boxes)

    # The old naming really does collide -- pinned, so the defect cannot
    # silently return through a different call site.
    old = [abbreviate(p.band, p.x + p.w / 2, p.clamped) for p in random_boxes]
    assert len(set(old)) < len(old), (
        "if abbreviate stops colliding on random placement, this record "
        "needs re-reading rather than the test relaxing"
    )
    # Anatomy is exactly where abbreviate IS correct: 8 midline boxes at
    # x=0.5 take no suffix, 7 pairs take opposite sides.
    anatomy_names = rc.region_names(anatomy, "anatomy")
    assert len(set(anatomy_names)) == 22
    assert sum(1 for n in anatomy_names if not n.endswith(("_L", "_R"))) == 8

    # The fix: neutral, unique, position-independent.
    names = rc.region_names(random_boxes, "random")
    assert names == [f"random_{i:02d}" for i in range(22)]
    assert not any(n.endswith(("_L", "_R")) for n in names)

    # The size matching is recorded rather than implied by a label.
    provenance = rc.size_provenance(anatomy, random_boxes, "random")
    assert set(provenance) == set(names)
    for name, source, template in zip(names, provenance.values(), anatomy):
        assert source["inherits_size_from"] == template.band
        assert source["w"] == round(template.w, 6)
    assert rc.size_provenance(anatomy, anatomy, "anatomy") == {}

    # Duplicates refuse with the reason, whatever produces them.
    with pytest.raises(rc.RegionCropError, match="not unique"):
        rc.region_names(anatomy[:2] + anatomy[:2], "anatomy")

    record = roadb.RANDOM_ARM_NAMES_WERE_THE_DEFECT
    assert "does NOT vary" in record["not_the_cause"]
    assert "laterality is an" in record["the_cause"]
    assert "not anatomical" in record["why_not_anatomy_names"]
    assert "something false" in record["second_consequence"]
    assert "strictly one-factor" in record["the_contrast_is_unaffected"]


def test_the_random_control_matches_anatomy_in_all_three_respects():
    """Count, size distribution and total area identical **by
    construction** -- the anatomy boxes themselves, re-placed. Only
    placement differs, which is what makes the arm decide whether a
    finding is about anatomy or about cropping."""
    from cleft import roadb_regioncrop as rc

    anatomy = rc.protected_patches()
    random_boxes = rc.random_patches(anatomy)
    assert rc.area_matches(anatomy, random_boxes)
    assert len(random_boxes) == len(anatomy) == 22
    # Every box moved -- otherwise "placement varies" would be false for
    # the ones that happened to land where they started.
    moved = sum(
        1 for a, b in zip(anatomy, random_boxes) if (a.x, a.y) != (b.x, b.y)
    )
    assert moved == 22
    # Seeded, so the control is a reproducible artifact.
    assert [
        (p.x, p.y) for p in rc.random_patches(anatomy)
    ] == [(p.x, p.y) for p in random_boxes]
    assert [
        (p.x, p.y) for p in rc.random_patches(anatomy, seed=99)
    ] != [(p.x, p.y) for p in random_boxes]


def test_the_crop_sheet_shows_native_resolution_beside_the_face():
    """The gate's design: crops are pasted at their OWN pixel size, so the
    reviewer judges the information rather than the resampler."""
    import numpy as np

    from cleft import roadb_regioncrop as rc
    from cleft.roadb_staging import stage_non_square

    staged = stage_non_square(
        np.random.default_rng(0).integers(
            0, 255, (2424, 2173, 3)
        ).astype(np.uint8),
        768,
    )
    patches = rc.protected_patches()
    sheet = rc.crop_sheet(staged, patches)
    face_h, face_w = staged.image.shape[:2]
    # The face is on the left, at its own size, and the crops sit beside
    # it -- so the panel is wider than the face and at least as tall.
    assert sheet.shape[1] > face_w
    assert sheet.shape[0] >= face_h
    assert sheet.dtype == np.uint8

    sizes = rc.native_crop_sizes(staged, patches)
    assert sizes["n_regions"] == 22
    # The magnification premise, measured: the brief predicted ~96 px and
    # every crop UPsamples to the 224 input rather than being shrunk.
    assert sizes["brief_predicted_width_px"] == 96
    assert sizes["width_px"]["max"] <= rc.CROP_OUTPUT_SIZE
    assert sizes["upsample_to_backbone"]["min"] >= 1.0


def test_the_graph_arm_pools_and_says_what_that_gives_up():
    """**Option B, with its cost on the record**: pooled nodes keep the
    combination contrast one-factor, and the arm is explicitly NOT
    SR-GNN as published."""
    import numpy as np

    from cleft import embeddings, roadb_regioncrop as rc

    # The mechanism: graph backbones return maps by decision, so the
    # pooling is a response to a contract, not a workaround.
    assert embeddings.KIND_FOR_BACKBONE_KIND["graph"] == "feature_map"
    maps = np.arange(2 * 4 * 2 * 3, dtype=np.float32).reshape(2, 4, 2, 3)
    pooled = rc.pool_feature_maps(maps)
    assert pooled.shape == (2, 4)
    assert np.allclose(pooled, maps.mean(axis=(2, 3)))
    with pytest.raises(rc.RegionCropError, match="feature maps"):
        rc.pool_feature_maps(np.zeros((5, 2048)))

    record = roadb.GRAPH_ARM_POOLS_TO_VECTORS
    assert record["option"].startswith("B")
    assert "2.2M" in record["why_not_store_the_maps"]
    assert "two factors" in record["why_not_store_the_maps"]
    # The disclaimer that stops a later CleftGNN reading.
    assert "NOT replicate SR-GNN" in record["not_srgnn_as_published"]
    assert "SeqSelfAttention" in record["not_srgnn_as_published"]
    # And why the superseded record's size argument is not authority here.
    assert "SUPERSET" in record["per_region_size_argument_does_not_transfer"]
    assert "independent" in record["per_region_size_argument_does_not_transfer"]

    # Both SR-GNN arms still sit on identical nodes, which is the point.
    arms = {a["key"]: a for a in roadb.regioncrop_arms()}
    graph, concat = arms["anatomy_graph_srgnn"], arms["anatomy_concat_srgnn"]
    assert graph["backbone"] == concat["backbone"] == "srgnn"
    assert graph["input"] == concat["input"] == "anatomy_crops"


def test_the_control_runs_through_the_crop_path(repo_root):
    """**Both failures dissolved by one change.** The whole frame is one
    region, cut and resized like every crop, so crop-vs-whole differs in
    content alone -- and the stacked-tensor path is not involved."""
    import numpy as np
    import yaml

    from cleft import roadb_regioncrop as rc
    from cleft.roadb_staging import stage_non_square

    patch = rc.whole_frame_patch()
    assert (patch.x, patch.y, patch.w, patch.h) == (0.0, 0.0, 1.0, 1.0)
    staged = stage_non_square(
        np.zeros((2424, 2173, 3), dtype=np.uint8), 768
    )
    box = rc.region_boxes(staged, [patch])[0]
    height, width = staged.image.shape[:2]
    assert box == (0, 0, width, height), "the whole content box, exactly"
    # It goes through the same resize, so the control inherits the same
    # square stretch the crops do -- comparator, not confound.
    assert rc.crop_region(staged, box).shape == (
        rc.CROP_OUTPUT_SIZE, rc.CROP_OUTPUT_SIZE, 3
    )

    config = yaml.safe_load(
        (repo_root / "configs" / "roadb_p7c_crops_vit_b16_whole.yaml")
        .read_text(encoding="utf-8")
    )
    assert config["task"]["kind"] == "roadb_region_crop_extract"
    assert config["task"]["layout"] == "whole"
    # The superseded config must be gone, not merely unused.
    assert not (
        repo_root / "configs"
        / "roadb_p7c_extract_vit_b16_imagenet_768_nsq_g2.yaml"
    ).exists()
    # And the arm reads the set this produces.
    arm = yaml.safe_load(
        (repo_root / "configs" / "roadb_p7c_arm_control_whole_vit.yaml")
        .read_text(encoding="utf-8")
    )
    emb = {e["name"]: e for e in arm["inputs"]}["embeddings"]
    assert config["task"]["out_version"] in emb["path"]
    assert emb["path"].endswith("vit_b16__imagenet__whole")


def test_the_as_fed_sheet_shows_the_square_stretch(repo_root):
    """**The second instance of the general point**: a sheet showing the
    artifact rather than the input answers a different question."""
    import numpy as np

    from cleft import roadb_regioncrop as rc
    from cleft.roadb_staging import stage_non_square

    staged = stage_non_square(
        np.random.default_rng(3).integers(
            0, 255, (2424, 2173, 3)
        ).astype(np.uint8),
        768,
    )
    patches = rc.protected_patches()
    native = rc.crop_sheet(staged, patches)
    as_fed = rc.crop_sheet(staged, patches, as_fed=True)
    # The as-fed panels are all 224 squares, so its grid is taller and
    # wider than the native one's ragged cells -- they are different
    # pictures of the same crops.
    assert as_fed.shape != native.shape

    record = roadb.CROPS_ARE_SQUARE_STRETCHED
    ratios = record["source_aspect_ratios"]
    boxes = rc.region_boxes(staged, patches)
    measured = [box[2] / box[3] for box in boxes]
    assert round(min(measured), 2) == ratios["min"]
    assert round(max(measured), 2) == ratios["max"]
    assert ratios["min"] < 1 < ratios["max"], (
        "some regions are squeezed and some stretched"
    )
    assert "confounds no" in record["confounds_nothing"] or (
        "identically" in record["confounds_nothing"]
    )
    assert "wrong to a clinician" in record["but"]

    # The general rule. Three instances now -- the as-fed review added a
    # CONFIRMING one; test_both_crop_gates_passed covers what that means.
    rule = roadb.SHEETS_MUST_SHOW_WHAT_THE_CONSUMER_RECEIVES
    assert set(rule["instances"]) == {
        "masked_scut", "region_crops", "region_crops_as_fed",
    }
    assert "interpolation" in rule["instances"]["masked_scut"]
    assert "SQUARED" in rule["instances"]["region_crops"]
    assert "second sheet that does" in rule["rule"]

    # Both sheets ship, and each names which question it answers.
    for stem, phrase in (
        ("roadb_p7c_region_crop_sheet", "AS CUT"),
        ("roadb_p7c_region_crop_sheet_as_fed", "AS THE BACKBONE RECEIVES"),
    ):
        text = (repo_root / "configs" / f"{stem}.yaml").read_text("utf-8")
        assert phrase in text, stem


def test_both_crop_gates_passed_and_asked_different_questions():
    """**Two sheets, two questions, both passed** -- and the rule they
    come from now has a confirming instance as well as a catching one."""
    native = roadb.CROP_SHEET_REVIEW_PASSED
    as_fed = roadb.CROP_SHEET_AS_FED_REVIEW_PASSED

    # Gate 1 is explicitly one of two, and names what it could not cover.
    assert "gate 1 of 2" in native["verdict"]
    assert native["partner_gate"] == "CROP_SHEET_AS_FED_REVIEW_PASSED"
    assert "square resize" in native["did_not_cover"]
    assert "AS CUT" in native["did_not_cover"]

    # Gate 2 passed, on the question gate 1 could not ask.
    assert as_fed["verdict"].startswith("PASSED")
    assert "unblocked" in as_fed["verdict"]
    assert "survive the transform" in as_fed["question"]
    assert "recognisable after the resize" in as_fed["finding"]
    assert "hides transformation" in as_fed["neither_sheet_substitutes"]
    assert "hides how much information" in as_fed["neither_sheet_substitutes"]

    # The decision it licenses, and that it rests on a review.
    assert "DECLINED" in as_fed["consequence"]
    assert "rather than an argument" in as_fed["consequence"]
    # It covered the measured range, not a convenient middle.
    ratios = roadb.CROPS_ARE_SQUARE_STRETCHED["source_aspect_ratios"]
    assert str(ratios["min"]) in as_fed["range_covered"]
    assert str(ratios["max"]) in as_fed["range_covered"]
    assert roadb.CROPS_ARE_SQUARE_STRETCHED["reviewed"].startswith("accepted")

    # The rule has all three modes -- and the third is why it is credible.
    rule = roadb.SHEETS_MUST_SHOW_WHAT_THE_CONSUMER_RECEIVES
    assert rule["modes_seen"] == ("prevented", "caught", "confirmed")
    assert len(rule["instances"]) == 3
    assert rule["instances"]["masked_scut"].startswith("PREVENTED")
    assert rule["instances"]["region_crops"].startswith("CAUGHT")
    assert rule["instances"]["region_crops_as_fed"].startswith("CONFIRMED")
    assert "not running" in rule["why_the_confirming_case_matters"]
    assert "manufactured" in rule["why_the_confirming_case_matters"]


def test_the_crop_sheet_review_is_recorded_as_the_gate_it_is():
    """**PASSED, with the observation explained and the claim narrowed.**
    The review is the one check no test substitutes for, so its verdict,
    what it covered, and what it implies are all on the record."""
    record = roadb.CROP_SHEET_REVIEW_PASSED
    assert record["verdict"].startswith("PASSED")
    # [UPDATED 2026-08-14] Criterion 3 needed BOTH sheets, so this gate's
    # verdict no longer claims to meet it alone -- the partner does that.
    assert "gate 1 of 2" in record["verdict"]
    assert "unblocked" in roadb.CROP_SHEET_AS_FED_REVIEW_PASSED["verdict"]
    assert "SEPARATE patches" in record["checked"], (
        "both philtral columns present is the mirrored-pair expansion "
        "doing what the 22-of-27 count claims"
    )

    # Checked at BOTH ends of the size range, not the middle -- and the
    # extremes bracket the widths the code itself produces.
    extremes = record["checked_at_both_extremes"]
    small = extremes["patient_205"]["median_crop_width_px"]
    large = extremes["patient_89"]["median_crop_width_px"]
    assert small < large
    assert "smallest crops" in extremes["why"]

    # **The two populations must not be conflated.** The 99/149/198
    # figures are ONE patient's 22 regions; these are medians ACROSS
    # patients, and 97 < 99 is consistent rather than contradictory. This
    # assertion exists because the first version of it compared them
    # directly and failed -- correctly.
    magnification = roadb.CROP_MAGNIFICATION_MEASURED
    assert "NOT across patients" in magnification["measured_px_population"]
    across = magnification["across_patients"]
    assert across["median_px"]["patient_205"] == small
    assert across["median_px"]["patient_89"] == large
    assert small < magnification["measured_px"]["min"], (
        "the record must keep saying WHY a cross-patient median can sit "
        "below a within-patient minimum"
    )
    # What survives either way: every crop still upsamples to 224.
    from cleft import roadb_regioncrop as rc

    assert large < rc.CROP_OUTPUT_SIZE and small < rc.CROP_OUTPUT_SIZE
    assert "a fortiori" in across["reading"]

    # The observation is recorded WITH its explanation, so a later reader
    # does not rediscover it as a defect.
    assert "upper lip" in record["observation"]
    assert "NOT a placement error" in record["explanation"]
    assert "content box" in record["explanation"]

    # And the limitation that survives the explanation, which is the part
    # that bears on what a positive result may say.
    assert "ANATOMICAL PROPORTION" in record["what_the_mapping_does_not_remove"]
    assert "per-patient landmarks" in record[
        "what_the_mapping_does_not_remove"
    ]
    consequence = record["consequence_for_the_claim"]
    assert "crops at anatomical positions help" in consequence
    assert "NOT 'landmark-precise anatomy helps'" in consequence
    assert "before any arm runs" in consequence

    # The gate it releases: the anatomy-vs-random contrast is the one the
    # narrowed claim attaches to.
    assert any(
        c["key"] == "anatomy_vs_random" and c["strictly_one_factor"]
        for c in roadb.REGION_CROP_CONTRASTS
    )


def test_the_crop_sheet_task_runs_end_to_end_and_writes_pngs(tmp_path):
    """**The test that was missing.** ``crop_sheet`` was tested directly
    and the TASK was not, so ``save_sheet(path, sheet)`` -- the arguments
    reversed -- shipped and died on the cluster. Exercising the task is
    what covers the call site."""
    import numpy as np

    from cleft import roadb, roadb_tasks
    from cleft.roadb_staging import stage_non_square

    setting = [
        s for s in roadb.staging_settings()
        if s["artifact"] == "roadb_768_nonsquare_g2_v1"
    ][0]
    root = tmp_path / setting["artifact"]
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    patients = [11, 22, 33]
    lines = ["patient_id,frontal_id,aspect_ratio"]
    for patient_id in patients:
        staged = stage_non_square(
            rng.integers(0, 255, (2424, 2173, 3)).astype(np.uint8), 768
        )
        np.save(
            root / f"staged_patient_g2_p{patient_id:03d}.npy", staged.image
        )
        lines.append(f"{patient_id},{patient_id}0,0.897")
    (root / "geometry.csv").write_text("\n".join(lines) + "\n", "utf-8")

    ctx = _StageCtx(
        tmp_path,
        {
            "kind": "roadb_region_crop_sheet",
            "staged_artifact": setting["artifact"],
            "n_patients": 2,
            "columns": 6,
        },
        [
            {"name": setting["artifact"], "path": str(root),
             "rollup": "a" * 64},
        ],
    )
    roadb_tasks.region_crop_sheet(ctx)

    written = sorted(p.name for p in ctx.run_dir.glob("crops_*.png"))
    # Both layouts, both sampled patients -- and REAL PNGs, which is what
    # the reversed call never produced.
    assert len(written) == 4, written
    # Filenames carry the MODE as well as the layout, so a native and an
    # as-fed sheet for the same patient never collide in a run directory.
    assert any(name.startswith("crops_native_anatomy_") for name in written)
    assert any(name.startswith("crops_native_random_") for name in written)
    for name in written:
        assert (ctx.run_dir / name).stat().st_size > 1000, name

    import json

    summary = json.loads((ctx.run_dir / "metrics.json").read_text("utf-8"))
    assert summary["n_regions"] == 22
    assert summary["sheets_written"] == 4
    assert summary["every_crop_upsamples_to_224"] is True
    assert "reviewed" in summary["gate"]
    # Patient ids stay CLUSTER-ONLY, never in the SHAREABLE metrics.
    assert "11" not in json.dumps(summary.get("median_crop_width_px", {}))
    sizes = json.loads((ctx.run_dir / "crop_sizes.json").read_text("utf-8"))
    assert set(sizes) <= {str(p) for p in patients}


def test_an_empty_sheet_refuses_with_the_reason_not_a_shape_error(tmp_path):
    """**A zero-panel sheet is a review gate that cannot fail visibly**,
    so emptiness refuses upstream -- and the renderer now names a swapped
    argument instead of reporting its symptom."""
    import numpy as np

    from cleft import roadb_regioncrop as rc
    from cleft.geometry.render import RenderError, save_sheet
    from cleft.roadb_staging import stage_non_square

    staged = stage_non_square(
        np.zeros((2424, 2173, 3), dtype=np.uint8), 768
    )
    with pytest.raises(rc.RegionCropError, match="empty layout"):
        rc.crop_sheet(staged, [])

    # The class both sheet defects belong to: the render layer reporting
    # something its caller decided. A 0-d array is never a malformed
    # image -- it is the wrong argument, and the message says so.
    with pytest.raises(RenderError, match="argument order"):
        save_sheet(tmp_path / "x.png", np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(RenderError, match="0-d"):
        save_sheet(None, tmp_path / "y.png")
    # A real array still works, so the guard is narrow.
    assert save_sheet(
        np.zeros((4, 4, 3), np.uint8), tmp_path / "z.png"
    ).exists()


def test_region_vectors_is_a_new_kind_with_named_axes(tmp_path):
    """**(N, 22, D) with its region names**, and refused without them --
    an artifact says what its axes mean. Not a revival of ``per_region``,
    which was superseded for a reason that does not transfer."""
    import numpy as np

    from cleft import embeddings

    names = [f"region_{i}" for i in range(22)]
    values = np.zeros((3, 22, 768), dtype=np.float32)
    common = dict(
        backbone="vit_b16", backbone_kind="transformer", init="imagenet",
        geometry="g2", variant=None, checkpoint_sha256=None,
        patient_ids=[1, 2, 3],
    )
    embeddings.save(
        tmp_path / "ok", values, regions=names, kind="region_vectors",
        **common
    )
    loaded, metadata = embeddings.load(tmp_path / "ok", [1, 2, 3])
    assert loaded.shape == (3, 22, 768)
    assert metadata["kind"] == "region_vectors"
    # THE point: which column is which, carried with the array.
    assert metadata["regions"] == names

    # Refused without the names -- positional trust is what this prevents.
    with pytest.raises(embeddings.EmbeddingError, match="region geometry"):
        embeddings.save(
            tmp_path / "unnamed", values, regions=None,
            kind="region_vectors", **common
        )
    # Refused when the names do not describe the array.
    with pytest.raises(embeddings.EmbeddingError, match="does not describe"):
        embeddings.save(
            tmp_path / "short", values, regions=names[:5],
            kind="region_vectors", **common
        )
    # The override is narrow: nothing else may diverge from the derived
    # kind, so a graph backbone still cannot write pooled vectors.
    with pytest.raises(embeddings.EmbeddingError, match="not an allowed"):
        embeddings.save(
            tmp_path / "wrong", values, regions=names, kind="per_region",
            **common
        )

    record = embeddings.REGION_VECTORS_IS_NOT_PER_REGION
    assert "sliced a whole-frame feature map" in record[
        "same_shape_opposite_provenance"
    ]
    assert "embeds each pixel crop" in record[
        "same_shape_opposite_provenance"
    ]
    assert "SeqSelfAttention" in record["why_per_region_was_superseded"]
    assert "never sees the whole face" in record["why_that_does_not_transfer"]
    assert "philtral_column" in record["regions_are_mandatory"]


def test_the_measured_magnification_supersedes_the_briefs_estimate():
    """**The premise got bigger, not true** -- and the record says which."""
    import numpy as np

    from cleft import roadb_regioncrop as rc
    from cleft.roadb_staging import stage_non_square

    record = roadb.CROP_MAGNIFICATION_MEASURED
    assert record["brief_predicted_px"] == 96
    assert record["measured_px"]["median"] == 149
    assert record["measured_px"]["min"] > record["brief_predicted_px"]
    assert "uniform 1/6" in record["why_the_estimate_was_low"]
    assert "bigger, not true" in record["what_it_does_not_settle"]

    # Re-measured here, so the recorded numbers cannot drift from the code
    # that produced them.
    staged = stage_non_square(
        np.zeros((2424, 2173, 3), dtype=np.uint8), 768
    )
    sizes = rc.native_crop_sizes(staged, rc.protected_patches())
    assert sizes["width_px"] == record["measured_px"]
    assert sizes["upsample_to_backbone"]["min"] >= 1.0, (
        "the claim is that NO region is shrunk on its way in"
    )


def test_the_region_crop_arms_make_every_contrast_one_factor():
    """**Five arms, four contrasts, one factor each** -- the two
    matched-pipeline defects in the brief, corrected in the enumeration
    rather than in prose."""
    from cleft import ladder, phase7c

    record = roadb.REGION_CROP_STRUCTURE
    # The justification that was refuted must be named as refuted.
    assert "1.065x chance" in record["justification"]["refuted"]
    assert "Must not be repeated" in record["justification"]["refuted"]
    assert "magnification" in record["justification"]["holds"]

    # The regions are Phase 7C's list, and the expansion arithmetic is
    # checked against it rather than trusted.
    assert record["regions"]["names"] == len(phase7c.PROTECTED_REGIONS) == 15
    assert record["regions"]["boxes"] == 22
    assert "8 midline + 7 bilateral" in record["regions"]["expansion"]
    for excluded in ("glabella", "medial_canthus", "lateral_orbit"):
        assert excluded not in phase7c.PROTECTED_REGIONS

    # The staging is one of the ten, already built -- not a new one.
    artifacts = {s["artifact"] for s in roadb.staging_settings()}
    assert record["staging"]["artifact"] in artifacts
    setting = [
        s for s in roadb.staging_settings()
        if s["artifact"] == record["staging"]["artifact"]
    ][0]
    assert (setting["aspect"], setting["geometry"]) == ("nonsquare", "g2")
    assert setting["expected_non_content"] == 0.0, (
        "the brief's reason for G2 non-square is content fraction 1.0000"
    )

    arms = {arm["key"]: arm for arm in roadb.regioncrop_arms()}
    assert len(arms) == 5, "four in the brief plus the corrected control"
    assert all(arm["init"] == "imagenet" for arm in arms.values())
    assert all(
        arm["staging"] == record["staging"]["artifact"]
        for arm in arms.values()
    ), "every arm at the crops' own staging -- defect 1"
    # Seed counts follow the regime, as everywhere else on this road.
    for arm in arms.values():
        regime = "graph" if arm["backbone"] == "srgnn" else "transformer"
        assert arm["seeds"] == roadb.PHASE_7_SEED_COUNTS["per_regime"][regime]

    # THE property, and the honest exception the test itself found: a
    # whole frame has no combination rule, so crop-vs-whole changes
    # input and head dimension together and cannot be decomposed. Two
    # contrasts are strictly one-factor; the other two declare the
    # indivisibility rather than claiming to isolate cropping.
    factors = ("backbone", "input", "combination", "init", "staging")
    strict, indivisible = [], []
    for contrast in roadb.REGION_CROP_CONTRASTS:
        a, b = arms[contrast["a"]], arms[contrast["b"]]
        differing = [f for f in factors if a[f] != b[f]]
        assert a["seeds"] == b["seeds"], contrast["key"]
        assert a["backbone"] == b["backbone"], contrast["key"]
        if contrast["strictly_one_factor"]:
            assert differing == [contrast["varies"]], (
                f"{contrast['key']} varies {differing}, not just "
                f"{contrast['varies']}"
            )
            strict.append(contrast["key"])
        else:
            # Exactly the whole-frame side, and exactly because its
            # combination is undefined rather than different.
            assert None in (a["combination"], b["combination"])
            assert set(differing) == {"input", "combination"}
            assert "indivisible" in contrast
            indivisible.append(contrast["key"])
    assert strict == ["anatomy_vs_random", "graph_vs_concat"]
    assert indivisible == ["crop_vs_whole", "random_vs_whole"]

    # And the defects are recorded, one accepted and one flagged.
    defects = record["defects_found_in_the_brief"]
    assert defects["control_mismatch"].startswith("ACCEPTED")
    assert "three factors" in defects["control_mismatch"] or (
        "cropping, aspect AND geometry" in defects["control_mismatch"]
    )
    assert defects["combination_mismatch"].startswith("FLAGGED")
    assert "does not cover" in defects["combination_mismatch"]
    # The reason the simpler fix was rejected is a real measurement.
    from cleft.train.graph_cleft import MEASURED_GRAPH_SEED_BAND

    assert "graph-layer initialisation" in MEASURED_GRAPH_SEED_BAND[
        "does_not_measure"
    ]
    # The init cost is Road A's measured pair, not an estimate.
    d1 = ladder.STAGE_D1_AT_G1["sd"]["srgnn"]
    assert str(d1["imagenet"]) in record["init_is_imagenet_throughout"]["cost"]
    assert str(d1["scut_masked"]) in record["init_is_imagenet_throughout"][
        "cost"
    ]
    # The two corrections that are about how to BUILD it.
    assert "extraction ONCE" in record["precompute_once"]
    assert "not a transplant" in record["the_real_build_cost"]
    assert "42x42" in record["the_real_build_cost"]


def test_the_margin_structure_keeps_the_exception_that_refutes_it():
    """**The tempting sharpening is wrong, and the record says so.** A
    3.83x margin passed condition 1, so "nothing below ~6x has ever
    passed" is refuted by the project's own audit."""
    from cleft import ladder

    record = roadb.CONDITION_1_MARGIN_STRUCTURE

    # The exception is real and comes from Road A's own record.
    survivor = ladder.LADDER_PAIRED_AUDIT["survivor"]
    assert survivor["margin"] == 3.83
    assert survivor["n_excluding_zero"] == survivor["n_seeds"]
    assert survivor["margin"] in record["survivors"]
    assert min(record["survivors"]) == 3.83

    # Floor, ceiling, and the mixed middle that is the actual finding.
    assert max(record["withdrawals"]) == 4.69
    assert all(m <= 2.55 or m >= 3.62 for m in record["withdrawals"])
    # [2026-08-15] 7D filled the empty 4.7x-6.91x interval with two passes,
    # so the clean-pass region is asserted at its NEW floor -- and the
    # update is dated in the record, not silently absorbed.
    clean = [m for m in record["survivors"] if m >= 5.32]
    assert len(clean) == 5
    assert all(m > max(record["withdrawals"]) for m in clean)
    assert 6.42 in record["survivors"] and 5.32 in record["survivors"]
    assert 1.77 in record["withdrawals"] and 1.89 in record["withdrawals"]
    assert "five for five" in record["ceiling"]
    assert "was 6.91x" in record["ceiling"], "the old figure stays visible"
    assert "SUPERSEDED 2026-08-15" in record["what_is_true_instead"]
    assert "5.32x" in record["what_is_true_instead"]
    # 3.83 passed while 4.34 and 4.69 -- both larger -- failed.
    assert 4.34 in record["withdrawals"] and 4.69 in record["withdrawals"]
    assert "does not order it" in record["the_middle_is_the_finding"]

    # The correction is kept, not smoothed away.
    assert "refuted by" in record["the_tempting_sharpening_is_wrong"]
    assert "USABLE claim" in record["what_is_true_instead"]
    assert "0.0065" in record["what_is_true_instead"]
    # ...and it agrees with Road A's own standing conclusion.
    assert "does not govern" in ladder.LADDER_PAIRED_AUDIT[
        "margin_does_not_predict_condition_1"
    ]["consequence"]


def test_branch_3_closes_with_nothing_claimable_and_two_findings():
    """**Closed: no claim, three withdrawals, one thing never built, and
    an answer to the question that was asked.**"""
    record = roadb.ROAD_B_BRANCH_3_CLOSING
    assert record["claimable"] == []
    assert set(record["withdrawn"]) == {
        "anatomy_vs_random", "random_vs_whole", "crop_vs_whole",
    }
    # The margins match what the arms record, so the two cannot drift.
    outcome = roadb.REGION_CROP_ARMS_OBSERVED["outcome"]["results"]
    for key, entry in record["withdrawn"].items():
        assert outcome[key]["margin"] == entry["margin"]
        assert outcome[key]["claimable"] is False

    # The descriptive residue, with the null that makes it coherent.
    assert "MATCH a whole frame" in record["supported_descriptively"]
    assert "~60% of the image" in record["supported_descriptively"]
    assert "periphery is free" in record["the_shape"]
    assert "cannot prove" in record["the_shape"]

    # What was never built, and that its contrast is unanswered.
    never = record["never_built"]
    assert "graph_vs_concat" in never["what"]
    assert "not a reshape" in never["why"]
    assert "UNANSWERED" in never["consequence"]
    assert "orphaned half" in never["consequence"]

    # The the request raised at supervision, answered rather than deflected.
    answer = record["answers_the_supervision_request"]
    assert "manual patch selection" in answer
    assert "like a whole frame and better than random" in answer
    assert "not a failure to do the work" in answer

    # The two findings that came from the branch rather than its arms.
    findings = record["findings_from_the_branch_not_its_arms"]
    assert "0.0002" in findings["srgnn_ceiling"]
    assert "selection" in findings["refused_fifth_contrast"]

    # And the misreadings the write-up must avoid -- including the one a
    # null invites: "cropping does not work".
    guard = record["do_not_write_up_as"]
    assert "NOT 'cropping does not work'" in guard
    assert "MATCH the whole frame" in guard
    assert "cannot resolve it either way" in guard or (
        "cannot resolve either way" in guard
    )


def test_the_backbone_pair_is_one_factor_and_deliberately_untested():
    """**A legitimate one-factor contrast that is NOT added**, because
    adding it after seeing it separate would be the selection the Swin
    peak was refused for."""
    arms = {a["key"]: a for a in roadb.regioncrop_arms()}
    vit, srgnn = arms["anatomy_concat_vit"], arms["anatomy_concat_srgnn"]

    # It really does vary backbone alone -- the note that said otherwise
    # was written before the matched partner existed.
    differing = [
        field for field in ("input", "combination", "init", "staging")
        if vit[field] != srgnn[field]
    ]
    assert differing == [], differing
    assert vit["backbone"] != srgnn["backbone"]

    # And it is absent from the registered set, in both directions.
    keys = {c["key"] for c in roadb.REGION_CROP_CONTRASTS}
    assert "backbone" not in {c["varies"] for c in roadb.REGION_CROP_CONTRASTS}
    for contrast in roadb.REGION_CROP_CONTRASTS:
        a, b = arms[contrast["a"]], arms[contrast["b"]]
        assert a["backbone"] == b["backbone"], contrast["key"]
    assert len(keys) == 4

    # Reported descriptively instead, with what makes it clean.
    record = roadb.REGION_CROP_ARMS_OBSERVED["srgnn_on_identical_pixels"]
    assert record["delta_vs_vit"] < 0

    # The sharp version, re-derived from Road A's own record: SR-GNN's
    # best prior arm is a WHOLE-FRAME one, and 22 magnified anatomy crops
    # land 0.0002 away from it.
    from cleft import ladder

    d1 = ladder.STAGE_D1_AT_G1
    best_prior = max(d1["cells"]["srgnn"])
    assert record["srgnn_best_prior"]["pcc"] == best_prior == 0.1719
    observed = roadb.REGION_CROP_ARMS_OBSERVED["values"][
        "anatomy_concat_srgnn"
    ]["pcc"]
    assert abs(
        abs(best_prior - observed) - record["difference_from_best_prior"]
    ) < 1e-9
    assert record["difference_from_best_prior"] < 0.001
    assert "whole frame" in record["srgnn_best_prior"]["arm"].lower()
    # ...and it is a CEILING observation, not a contrast: nothing claimed.
    assert "CEILING observation" in record["not_a_controlled_comparison"]
    assert "Nothing is claimed" in record["not_a_controlled_comparison"]
    assert "backbone caps the result" in record["reading"]
    assert "identical by bytes" in record["why_cleaner_than_any_other_backbone_pair"]
    # The precision that stops "identical crop set" being misread as one
    # artifact: the artifact IS the encoding.
    assert "IS the encoding" in record["artifacts_differ_necessarily"]


def test_the_region_crop_pairs_are_registered_before_any_arm_runs():
    """The scope resolves through the SAME paired machinery, and the
    contrasts are fixed by the design rather than by an outcome."""
    from cleft import ladder
    from cleft.config.schema import TASK_SPECS

    assert "roadb_regioncrop" in TASK_SPECS["paired_claims"]["scope"].choices

    # [UPDATED 2026-08-14] Four contrasts are REGISTERED; the scope emits
    # only those whose both arms have run, so graph_vs_concat is absent
    # while the node path is unwired and returns without an edit here.
    assert len(roadb.REGION_CROP_CONTRASTS) == 4
    pairs = roadb.paired_claim_pairs("roadb_regioncrop")
    ran = set(roadb.REGION_CROP_ARMS_OBSERVED["values"])
    computable = {
        f"regioncrop__{c['key']}" for c in roadb.REGION_CROP_CONTRASTS
        if {c["a"], c["b"]} <= ran
    }
    assert {p["key"] for p in pairs} == computable
    assert "regioncrop__graph_vs_concat" not in computable
    assert "regioncrop__anatomy_vs_random" in computable, (
        "the strictly one-factor test of anatomy must be computable now"
    )
    # No recorded margin: nothing has run, and a zero would read as
    # measured.
    assert all("recorded" not in pair for pair in pairs)
    # The derivations are ladder's, as for roadb_resolution.
    vectors = ladder.paired_claim_vectors("roadb_regioncrop", pairs=pairs)
    stems = ladder.paired_claim_stems("roadb_regioncrop", pairs=pairs)
    # The three ViT arms that ran, five seeds each.
    assert len(stems) == 3 and len(vectors) == 3 * 5 == 15
    groups = ladder.paired_claim_seed_groups("roadb_regioncrop", pairs=pairs)
    assert set(groups) == {tuple(ladder.SEED_POOL[:5])}, (
        "only the transformer band is live; the graph band returns with "
        "the graph arms"
    )
    # No ViT-vs-SR-GNN pair: that would vary backbone and combination.
    arms = {arm["key"]: arm for arm in roadb.regioncrop_arms()}
    for pair in pairs:
        a = arms[pair["a"].replace("roadb_p7c_arm_", "")]
        b = arms[pair["b"].replace("roadb_p7c_arm_", "")]
        assert a["backbone"] == b["backbone"], pair["key"]

    with pytest.raises(ValueError, match="unknown Road B paired-claim scope"):
        roadb.paired_claim_pairs("roadb_nonsense")


def test_the_region_crop_control_configs_are_shipped(repo_root):
    """**Defect 1, corrected in shipped bytes**: a whole-frame ViT arm at
    the CROPS' staging, so crop-vs-whole varies input alone."""
    import yaml

    from cleft import ladder

    def load(stem):
        return yaml.safe_load(
            (repo_root / "configs" / f"{stem}.yaml").read_text(
                encoding="utf-8"
            )
        )

    staging = roadb.REGION_CROP_STRUCTURE["staging"]["artifact"]
    # [UPDATED 2026-08-14] The control moved to the CROP path: the
    # extraction task cannot read non-square staging (no stacked tensor,
    # and non-square frames cannot batch). Its own test covers why.
    extract = load("roadb_p7c_crops_vit_b16_whole")
    assert extract["phase"] == "roadb_p7c"
    assert extract["task"]["kind"] == "roadb_region_crop_extract"
    assert extract["task"]["staged_artifact"] == staging
    assert extract["task"]["layout"] == "whole"
    assert extract["task"]["backbone"] == "vit_b16"
    declared = {e["name"]: e for e in extract["inputs"]}
    # Imagenet extracts from the DECLARED snapshot (the vintage rule).
    assert declared["pretrained_init"]["path"].endswith("init_vit_b16_v1")
    assert declared["pretrained_init"]["rollup_sha256"] != "0" * 64
    # The staged declaration is CARRIED from the residual gate, so the
    # control cannot read different bytes from the gate's verdict.
    gate = {
        e["name"]: e
        for e in load("roadb_p2_residual_gate")["inputs"]
    }
    assert declared[staging] == gate[staging]

    arm = load("roadb_p7c_arm_control_whole_vit")
    assert arm["task"]["staged_artifact"] == staging
    assert arm["task"]["geometry"] == "g2"
    assert arm["task"]["seeds"] == list(ladder.SEED_POOL[:5])
    assert arm["task"]["backbone"] == "vit_b16"
    assert arm["task"]["init"] == "imagenet"
    assert arm["task"]["trainable"] == "head"
    # Field-for-field Road A's counterpart outside the identity fields.
    counterpart = load("p7_d1_vit_b16_imagenet_g1")
    # `consume` is a Branch 3 field, not a procedure field: it names HOW
    # the artifact is read, which Road A arms never had a choice about.
    identity = {"staged_artifact", "geometry", "seeds", "consume"}
    assert set(arm["task"]) - {"consume"} == set(counterpart["task"])
    assert arm["task"]["consume"] == "pooled"
    for key, value in counterpart["task"].items():
        if key not in identity:
            assert arm["task"][key] == value, key

    # Both headers say why the brief's control was replaced.
    for stem in ("roadb_p7c_crops_vit_b16_whole",
                 "roadb_p7c_arm_control_whole_vit"):
        text = (repo_root / "configs" / f"{stem}.yaml").read_text(
            encoding="utf-8"
        )
        assert "76.4%" in text or "square G1" in text
    arm_text = (repo_root / "configs" /
                "roadb_p7c_arm_control_whole_vit.yaml").read_text(
                    encoding="utf-8")
    # The bar is stated before the branch runs.
    assert "ViT-collapse size and nothing smaller" in arm_text


def test_the_paired_scope_reuses_road_as_implementation(repo_root):
    """**R10 applied to a whole path**: Road B adds a SCOPE and a pair
    enumeration, not a second paired-comparison. The task, the BCa, the
    loader, the truth cross-check and the short-band refusal are one
    implementation walked by both roads."""
    import ast

    from cleft import ladder

    pairs = roadb.paired_claim_pairs("roadb_resolution")

    # The three derivations are ladder's, taking Road B's pairs -- if
    # roadb ever grows its own copies, this fails.
    assert not hasattr(roadb, "paired_claim_vectors")
    assert not hasattr(roadb, "paired_claim_seed_groups")
    assert not hasattr(roadb, "paired_claim_stems")
    vectors = ladder.paired_claim_vectors("roadb_resolution", pairs=pairs)
    stems = ladder.paired_claim_stems("roadb_resolution", pairs=pairs)
    groups = ladder.paired_claim_seed_groups("roadb_resolution", pairs=pairs)
    # ...and passing pairs does not disturb Road A's own scopes.
    assert ladder.paired_claim_stems("headline") == sorted({
        p["a"] for p in ladder.paired_claim_pairs("headline")
    } | {p["b"] for p in ladder.paired_claim_pairs("headline")})

    assert len(vectors) == 170
    assert len(stems) == 22
    assert set(groups) == {
        tuple(ladder.SEED_POOL[:5]), tuple(ladder.SEED_POOL[:10]),
    }

    # Exactly ONE call site computes a paired comparison, and the task
    # dispatches its SOURCE rather than branching its arithmetic.
    run_src = (repo_root / "src" / "cleft" / "run.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(run_src)
    task_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "task_paired_claims"
    )
    calls = [
        node for node in ast.walk(task_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "paired_comparison"
    ]
    assert len(calls) == 1, "a second paired_comparison call site appeared"
    # And the BCa itself is DEFINED once in the whole tree -- Road A
    # already calls it from two tasks, which is reuse; a second
    # definition would be the duplication.
    definitions = [
        path.name
        for path in (repo_root / "src" / "cleft").rglob("*.py")
        if "def paired_comparison(" in path.read_text(encoding="utf-8")
    ]
    assert definitions == ["phase7b.py"], definitions
    # The scope is a schema choice, not a new task kind.
    from cleft.config.schema import TASK_SPECS

    assert "roadb_resolution" in TASK_SPECS["paired_claims"]["scope"].choices
    assert "roadb_paired_claims" not in TASK_SPECS


def test_the_road_b_pairs_are_uniform_by_construction():
    """**No selection at any level**: every pairwise resolution contrast
    among arms that ran, derived from the recorded values -- so the
    interesting cell is embedded in a set fixed by the design."""
    from cleft import ladder
    from cleft.train.phase3 import combined_claimable_delta

    pairs = roadb.paired_claim_pairs("roadb_resolution")
    values = roadb.PHASE_7_TWENTY_TWO_ARMS["values"]

    # The rule: all C(n,2) resolution pairs per family, for arms with
    # recorded values. 6 complete families x 3 + 2 incomplete x 1.
    expected = sum(
        len(points) * (len(points) - 1) // 2 for points in values.values()
    )
    assert len(pairs) == expected == 20
    endpoints = [p for p in pairs if p["recorded"]["is_endpoint"]]
    assert len(endpoints) == 6, "one endpoint contrast per complete family"

    by_family = {}
    for arm in roadb.phase7_arms():
        by_family.setdefault(arm["family"], {})[arm["resolution"]] = arm

    for pair in pairs:
        family = pair["key"].split("__")[1] + "__" + pair["key"].split("__")[2]
        lo, hi = (int(x) for x in pair["key"].rsplit("__", 1)[1].split("_to_"))
        assert pair["question"] == pair["varies"] == "resolution"
        # Seeds follow the regime, and both sides share them.
        regime = by_family[family][hi]["regime"]
        assert pair["seeds"] == list(
            ladder.SEED_POOL[: roadb.PHASE_7_SEED_COUNTS["per_regime"][regime]]
        )
        # The recorded condition-2 numbers re-derive from the arms' SDs.
        (mean_lo, sd_lo), (mean_hi, sd_hi) = values[family][lo], values[family][hi]
        threshold = combined_claimable_delta(
            sd_hi, len(pair["seeds"]), sd_lo, len(pair["seeds"])
        )["arm_means_95"]
        assert abs(pair["recorded"]["delta"] - (mean_hi - mean_lo)) < 1e-6
        assert abs(pair["recorded"]["threshold"] - threshold) < 5e-4
        assert abs(
            pair["recorded"]["margin"] - abs(mean_hi - mean_lo) / threshold
        ) < 5e-3

    # Swin's interior is in the set, and so is every other family's --
    # which is what stops the interesting cell from being selected.
    keys = {p["key"] for p in pairs}
    assert "resolution__swin_b__imagenet__224_to_512" in keys
    interior_families = {
        p["key"].split("__")[1] for p in pairs
        if not p["recorded"]["is_endpoint"]
    }
    assert len(interior_families) == 4  # all four backbones

    # [CORRECTED 2026-08-12] Every stem is its OWN arm's run -- both
    # cells that looked like they needed routing have their own runs, and
    # using them keeps each family at one code SHA.
    stems = {p["a"] for p in pairs} | {p["b"] for p in pairs}
    assert "roadb_p7_graph_seed_band" not in stems
    assert "roadb_p7_arm_srgnn_masked_512" in stems
    assert all(stem.startswith("roadb_p7_arm_") for stem in stems)
    duplicates = roadb.PAIRED_CLAIM_DUPLICATE_RUNS
    assert duplicates["rule"].startswith("every stem is its own")
    assert "ONE code SHA" in duplicates["why"]
    assert set(duplicates["duplicates"]) == {
        "srgnn_masked_512", "swin_masked_224",
    }
    assert "determinism check" in duplicates["free_check_available"]
    assert not hasattr(roadb, "PAIRED_CLAIM_RUN_STEMS"), (
        "the routing table is gone, not merely emptied"
    )
    coverage = roadb.PAIRED_CLAIM_COVERAGE
    assert coverage["scope"] == "roadb_resolution"
    assert "re-introduce the selection" in coverage["why_uniform"]
    assert "no claims for condition" in coverage["excludes_init_contrasts"]
    assert "enter automatically" in coverage["excludes_pending"]


def test_the_paired_config_declares_exactly_the_derived_vectors(repo_root):
    """The config's declared inputs must parse back, through the loader's
    OWN convention, to exactly the (stem, seed) set the scope needs --
    the integration risk a shape test would miss."""
    import yaml

    from cleft import ladder, phase7c

    path = repo_root / "configs" / "roadb_p7_paired_resolution.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["task"] == {
        "kind": "paired_claims", "scope": "roadb_resolution", "n_boot": 10000,
    }
    assert payload["phase"] == "roadb_p7"
    assert payload["tier"] == "keeper"

    pairs = roadb.paired_claim_pairs("roadb_resolution")
    required = set(ladder.paired_claim_vectors("roadb_resolution", pairs=pairs))

    declared = {entry["name"]: entry for entry in payload["inputs"]}
    assert len(declared) == len(payload["inputs"]) == len(required)
    # Round-trip through the loader's parser, not a re-implementation.
    paths_by_stem = phase7c.oof_paths_from_inputs(declared)
    parsed = {
        (stem, seed) for stem, seeds in paths_by_stem.items() for seed in seeds
    }
    assert parsed == required

    text = path.read_text(encoding="utf-8")
    placeholder = "0" * 64
    if any(e["rollup_sha256"] == placeholder for e in payload["inputs"]):
        assert "PLACEHOLDER" in text.upper()

    # **FULLY RESOLVED** -- run listing then declare pass, both landed.
    # Every path names its own arm's run directory, every hash is real,
    # and every hash is DISTINCT: two seeds cannot write byte-identical
    # prediction files, so a repeat would mean one vector declared twice.
    from cleft.run_names import check_run_dir

    digests = set()
    for entry in payload["inputs"]:
        assert entry["path"].endswith("__predictions.csv")
        assert "/PENDING_" not in entry["path"]
        assert entry["rollup_sha256"] != placeholder
        digests.add(entry["rollup_sha256"])
        directory = entry["path"].rsplit("/", 2)[1]
        # The guard, applied to the bytes this run will actually read.
        check_run_dir(directory)
        stem = directory.split("__")[0]
        seed = entry["path"].rsplit("/", 1)[1].split("__")[0][len("seed_"):]
        assert entry["name"] == f"oof_{stem}_seed_{seed}"
    assert len(digests) == len(payload["inputs"]), (
        "two vectors share a hash -- distinct seeds cannot agree bytewise"
    )

    # The header states what the run is and why it is not more seeds.
    assert "FITS NOTHING" in text
    assert "PHASE_7_SWIN_512" in text
    assert "0 of 45 intervals excluding zero" in text
    assert "RESOLVED" in text
    assert "PAIRED_CLAIM_DUPLICATE_RUNS" in text


def test_the_mistyped_launch_is_recorded_with_the_guard_it_argues_for(
    repo_root,
):
    """**A run directory whose two identity halves disagreed** -- deleted,
    and the gap it exposed named with the guard that would close it."""
    record = roadb.MISTYPED_LAUNCH_DELETED
    assert record["directory"].endswith("roadb-p7-arm-swin-b-masked-768")
    assert "roadb_p7_arm_swin_b_masked_224" in record["directory"]
    # The identity halves disagree -- which is the whole finding.
    stem, _sha, job = record["directory"].split("__")
    assert job != stem.replace("_", "-")
    assert "ran the 224 config" in record["what_it_was"]
    assert "swin_b__scut_masked__g1" in record["identified_by"]
    assert "placeholders" in record["corroborated_by"]
    assert record["action"].startswith("deleted")
    assert "carries the identity TWICE" in record["why_dangerous"]
    assert "whole claim is its shape" in record["why_dangerous"]
    assert "not by the apparatus" in record["nothing_checks_agreement"]
    assert "harness.py is" in record["proposed_guard"]

    # [BUILT 2026-08-12] The guard exists, and the record says why the
    # literal rule could not ship -- measured, not asserted.
    built = record["built"]
    assert built["where"].startswith("cleft/run_names.py")
    assert "may DROP" in built["rule"] and "CONTRADICT" in built["rule"]
    assert "22 of 83" in built["why_not_the_literal_rule"]
    assert "switched off" in built["why_not_the_literal_rule"]
    assert record["status"].startswith("BUILT")

    from cleft.run_names import RunNameError, check_run_dir

    with pytest.raises(RunNameError):
        check_run_dir(record["directory"])

    # The deleted directory could not have been a real 768 result: that
    # config still declares placeholders, so it cannot have run.
    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "roadb_p7_arm_swin_b_masked_768.yaml")
        .read_text(encoding="utf-8")
    )
    assert any(
        entry["rollup_sha256"] == "0" * 64 for entry in payload["inputs"]
    ), "the corroboration in the record no longer holds"


def test_the_shape_statistic_gap_is_named_and_closed():
    """**Range is selected, the endpoint contrast is not** -- so the
    claim narrows to direction over fixed endpoints and the interior is
    described."""
    record = roadb.PHASE_7_SHAPE_STATISTIC
    assert "never a shape STATISTIC" in record["gap"]
    assert "winner's" in record["range_is_selected"]
    assert "anti-conservative" in record["range_is_selected"]
    assert "224->768" in record["claim_statistic"]
    assert "fixed by the design" in record["claim_statistic"]
    assert "own SDs" in record["claim_statistic"]
    assert "described, not claimed" in record["interior_is_descriptive"]
    assert "narrow what is claimed" in record["honest_summary"]

    # It is recorded as a FINDING -- a defect in the project's own
    # pre-registration -- and it sits beside the arms it amends.
    assert "pre-registration defect" in record["is_a_finding"]
    assert "beside the arms" in record["is_a_finding"]
    assert "before any shape was reported" in record["caught_when"]
    assert "PAIRED_BCA_WITHDREW_EVERY_VERDICT" in record[
        "the_failure_it_avoids"
    ]
    assert "registered a TOPIC" in record["generalisable_lesson"]
    # The registration it amends points at it, and its own text is
    # untouched so the defect stays legible.
    families = roadb.PHASE_7_ARM_STRUCTURE["families"]
    assert families["statistic_was_never_specified"] == (
        "see PHASE_7_SHAPE_STATISTIC"
    )
    assert "monotone up / monotone down / flat / peaked" in families[
        "shape_is_the_claim"
    ]

    # The registered claim it amends still says shape is the claim -- the
    # amendment narrows it rather than contradicting it.
    assert "shape_is_the_claim" in roadb.PHASE_7_ARM_STRUCTURE["families"]
    # Every family's range exceeds the graph band, which is exactly why
    # range cannot be the statistic.
    band = roadb.PHASE_7_GRAPH_BAND_RESOLVED["claimable_delta"][10]
    for family, points in roadb.PHASE_7_TWENTY_TWO_ARMS["values"].items():
        means = [m for m, _sd in points.values()]
        assert max(means) - min(means) > band, family


def test_swin_512_separates_what_was_predicted_from_what_was_not():
    """**The prediction covered the absence of a collapse, not the gain**
    -- and more seeds is refused for a reason that is not selection."""
    record = roadb.PHASE_7_SWIN_512
    prediction = roadb.SWIN_RESOLUTION_PREDICTION

    # What the registered branch actually says, quoted from the record it
    # cites rather than paraphrased.
    assert "near its own 224 level" in prediction["frozen_regime"]["supports"]
    assert "near its own 224 level" in record["what_was_predicted"]
    assert "ABSENCE of a collapse" in record["what_was_predicted"]
    assert "post-hoc" in record["what_was_not_predicted"]
    assert "largest number" in record["what_was_not_predicted"]

    # More seeds: refused, and the ORDER of the reasons is the point.
    seeds = record["more_seeds"]
    assert seeds["verdict"] == "REFUSED"
    assert "already answered" in seeds["first_reason"]
    assert "not the binding uncertainty" in seeds["first_reason"]
    assert "winner's-curse" in seeds["second_reason"]

    # The right test is the project's own, it can fail, and it needs no
    # cluster time -- each of which is checked against the real source.
    right = record["right_test"]
    assert right["what"] == "the per-seed paired BCa, phase7b.paired_comparison"
    assert hasattr(__import__(
        "cleft.phase7b", fromlist=["paired_comparison"]
    ), "paired_comparison")
    from cleft import phase7c

    withdrew = phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT
    assert withdrew["intervals_excluding_zero"] == "0 of 45"
    assert "withdrew all nine" in right["why"]
    assert "PAIRED_BCA_WITHDREW_EVERY_VERDICT" in right["why"]
    assert "can fail" in right["why"]
    assert "seed_<n>__predictions.csv" in right["costs_no_cluster_time"]
    assert "ALL endpoint contrasts" in right["scope"]

    # The frozen-regime record's "never run" is corrected in place.
    corrected = prediction["confirmed"]["frozen_regime_answered_after_all"]
    assert corrected["reading"] == "frozen_regime.supports"
    assert prediction["confirmed"]["frozen_regime_status"].startswith(
        "never run"
    ), "the superseded text stays visible beside the correction"

    # The G2 option is registered before any number exists, and its
    # artifacts really are among the staged ten and unconsumed.
    assert "Registered, not proposed" in record["if_a_new_cell_is_ever_wanted"]
    artifacts = {s["artifact"] for s in roadb.staging_settings()}
    assert {"roadb_512_square_g2_v1", "roadb_768_square_g2_v1"} <= artifacts
    assert "square_g2" in roadb.PHASE_7_STAGED_CONSUMPTION["unconsumed"]


def test_the_premise_reading_is_written_before_the_last_arms_land():
    """**The registered null fired in a stronger form** -- declining, not
    flat -- with the one predicted exception named and the cohort caveat
    carried."""
    record = roadb.PHASE_7_PREMISE_READING
    assert "two arms pending, deliberately" in record["read"]
    assert "10.8x" in record["premise"]
    assert record["registered_null"].startswith("flat at cleft time")
    assert "flat at cleft time" in roadb.PHASE_7_ARM_STRUCTURE[
        "null_reading_registered"
    ]
    assert "DECLINING" in record["observed"]
    assert "swin_b__imagenet" in record["the_exception"]
    assert "named in advance" in record["the_exception"]
    # Both halves: what it supports and what it does not.
    assert "frozen backbones at n=237" in record["supports"]
    assert "0.0857/0.0898" in record["supports"], (
        "the independent-route reproduction is the corroboration"
    )
    assert "NOT 'the pixels are uninformative'" in record["does_not_support"]
    assert "NOT 'resolution never helps'" in record["does_not_support"]
    assert "withdrew nine verdicts" in record["cohort_caveat"]
    # Why it is safe to write now: neither pending arm can move it.
    assert "descriptive by decision" in record["why_now"]
    assert "-0.103 at 512" in record["why_now"]
    pending = roadb.PHASE_7_TWENTY_TWO_ARMS["pending"]
    assert all("768" in cell for cell in pending)
    assert len(pending) == 2


def test_the_arm_configs_derive_from_the_enumeration(repo_root):
    """**24 arm configs, one per phase7_arms() entry: counterpart task
    blocks field-for-field outside identity fields, checkpoints verbatim
    from the extraction configs, the reuse arm verbatim from Road A's,
    and exactly ONE config fully resolved today** -- the partial-launch
    ledger as shipped bytes."""
    import yaml

    # Graph counterparts are PER BACKBONE: agnet arms carry
    # `deterministic: false` (the measured roi_align bound, declared
    # rather than defaulted) and srgnn arms must NOT -- schema-enforced
    # both ways, which is what caught the shared-counterpart version.
    counterparts = {
        ("transformer", "imagenet"): "p7_d1_vit_b16_imagenet_g1",
        ("transformer", "scut_masked"): "p7_c_swin_b_scut_masked_g1",
        ("srgnn", "imagenet"): "p7_d1_srgnn_imagenet_g1_native",
        ("srgnn", "scut_masked"): "p7_d1_srgnn_scut_masked_g1_native",
        ("agnet", "imagenet"): "p7_d1_agnet_imagenet_g1_native",
        ("agnet", "scut_masked"): "p7_d1_agnet_scut_masked_g1_native",
    }
    identity_fields = {"backbone", "init", "staged_artifact", "seeds",
                       "checkpoint"}
    staged = {224: "staged_v1", 512: "roadb_512_square_g1_v1",
              768: "roadb_768_square_g1_v1"}
    placeholder = "0" * 64

    def load(stem):
        return yaml.safe_load(
            (repo_root / "configs" / f"{stem}.yaml").read_text(
                encoding="utf-8"
            )
        )

    shipped = sorted((repo_root / "configs").glob("roadb_p7_arm_*.yaml"))
    assert len(shipped) == 24

    fully_resolved, blocked = [], []
    for arm in roadb.phase7_arms():
        short = "imagenet" if arm["init"] == "imagenet" else "masked"
        stem = f"roadb_p7_arm_{arm['backbone']}_{short}_{arm['resolution']}"
        payload = load(stem)
        task = payload["task"]
        declared = {e["name"]: e for e in payload["inputs"]}
        text = (repo_root / "configs" / f"{stem}.yaml").read_text(
            encoding="utf-8"
        )

        # Field-for-field the counterpart's procedure outside identity.
        kind = (
            arm["backbone"] if arm["regime"] == "graph" else "transformer"
        )
        counterpart = load(counterparts[(kind, arm["init"])])
        assert set(task) == set(counterpart["task"]), stem
        for key, value in counterpart["task"].items():
            if key not in identity_fields:
                assert task[key] == value, f"{stem}: {key}"
        assert task["backbone"] == arm["backbone"], stem
        assert task["init"] == arm["init"], stem
        assert task["staged_artifact"] == staged[arm["resolution"]], stem
        assert len(task["seeds"]) == arm["seeds"], stem
        assert task["staged_artifact"] in declared, stem

        # The header carries the arm's identity from the enumeration.
        assert arm["family"] in text, stem
        assert "PHASE_7_SEED_COUNTS" in text, stem

        if arm["init"] == "imagenet":
            assert not any(n.startswith("ckpt_") for n in declared), stem
        elif "bitwise reuse" in arm["set_artifact"]:
            road_a = {
                e["name"]: e
                for e in load("p7_c_swin_b_scut_masked_g1")["inputs"]
            }
            assert declared["embeddings"] == road_a["embeddings"], stem
            assert declared["ckpt_swin_b_masked_g1"] == (
                road_a["ckpt_swin_b_masked_g1"]
            ), stem
            assert "BITWISE REUSE" in text, stem
        else:
            ex = load(
                f"roadb_p7_extract_{arm['backbone']}_masked_"
                f"{arm['resolution']}"
            )
            ex_ckpt = [
                e for e in ex["inputs"] if e["name"].startswith("ckpt_")
            ]
            assert len(ex_ckpt) == 1, stem
            assert declared[ex_ckpt[0]["name"]] == ex_ckpt[0], (
                f"{stem}: checkpoint diverged from the extraction config"
            )

        # Launch gates in the headers, per the records.
        band_cell = (arm["backbone"], arm["resolution"], arm["init"]) == (
            "srgnn", 512, "scut_masked"
        )
        if band_cell:
            assert "SWEEP DOUBLES AS THIS ARM'S RUN" in text, stem
        elif arm["regime"] == "graph":
            assert "waits on the graph band sweep's READING" in text, stem
        else:
            assert "Not gated by the sweep" in text, stem
        if arm["backbone"] == "vit_b16":
            assert "DESCRIPTIVE" in text, stem
            assert "VIT_DRAW_PICKS" in text, stem

        hashes = [e["rollup_sha256"] for e in payload["inputs"]]
        if all(h != placeholder for h in hashes):
            fully_resolved.append(stem)
        if "BLOCKED ON THE -3 CELL" in text:
            blocked.append(stem)
            assert declared["embeddings"]["rollup_sha256"] == placeholder

    # The partial-launch ledger, as shipped bytes. At decision time one
    # config was resolved; the declare pass landed the same day
    # (declaration_pass_landed) and filled the 22 -- so every arm except
    # the two blocked on the -3 cells is now fully resolved.
    expected_resolved = sorted(
        f"roadb_p7_arm_{a['backbone']}_"
        f"{'imagenet' if a['init'] == 'imagenet' else 'masked'}_"
        f"{a['resolution']}"
        for a in roadb.phase7_arms()
        if not (
            a["init"] == "scut_masked"
            and a["backbone"] in ("swin_b", "vit_b16")
            and a["resolution"] == 768
        )
    )
    assert sorted(fully_resolved) == expected_resolved
    assert len(fully_resolved) == 22
    assert sorted(blocked) == [
        "roadb_p7_arm_swin_b_masked_768", "roadb_p7_arm_vit_b16_masked_768",
    ]
    ledger = roadb.PHASE_7_PARTIAL_LAUNCH["configs"]
    assert ledger["total"] == 25 == (
        ledger["resolved_today"]
        + ledger["awaiting_one_declaration_pass"]
        + ledger["blocked_on_minus3_cells"]
    )
    record = roadb.PHASE_7_PARTIAL_LAUNCH
    assert "RUN level" in record["split"]
    assert "REPORT level" in record["split"]
    assert "a line, not a shape" in record["report_gate"]
    assert "the twelfth graph arm" in record["graph_arms_wait_on"]
    assert record["transformer_arms_wait_on"].startswith("nothing")
    assert "a paste, not a build" in record["the_22s_blocker"]
    # The pass landed the same day, recorded beside the kept ledger --
    # and its four unresolved are the two -3 cells doubled, not new
    # problems.
    landed = record["declaration_pass_landed"]
    assert "21 hashed" in landed
    assert "the two -3 cells doubled" in landed
    assert "READY TO LAUNCH" in landed


def test_the_staged_consumption_is_recorded_not_implicit():
    """**Three staging inputs, TWO of the ten** -- 224 square is staged_v1,
    deliberately not among the ten -- and each unconsumed artifact carries
    why it exists and what could still consume it."""
    record = roadb.PHASE_7_STAGED_CONSUMPTION
    assert "not one of the ten" in record["consumed"]
    assert "roadb_512_square_g1_v1" in record["consumed"]
    assert "roadb_768_square_g1_v1" in record["consumed"]
    assert record["of_the_ten"].startswith("two consumed; eight unconsumed")
    assert "masked-G2 pretraining cells" in record["unconsumed"]["square_g2"]
    assert "contingent, not an oversight" in record["unconsumed"]["square_g2"]
    assert "BY DESIGN" in record["unconsumed"]["non_square"]
    # The nuance the nondeterminism finding adds: the pad pair survives in
    # the frozen regime.
    assert "READABLE as FROZEN imagenet arms" in record["unconsumed"]["non_square"]
    assert "registered" in record["not_waste"]


def test_the_twelve_cells_record_names_order_cost_and_band_rule():
    record = roadb.PHASE_6_TWELVE_CELLS
    assert "vit_b16_masked_g1_512" in record["launch_first"]
    assert "turns on it" in record["launch_first"]
    assert "2,304 tokens" in record["heaviest"]
    assert "costed in the configs" in record["heaviest"]
    assert "bitwise" in record["224_cells"]
    assert "determinism check" in record["224_cells"]
    assert "before that cell's band exists" in record["seed_bands"]


def test_the_rebuild_determinism_finding_is_recorded():
    """Byte-identical payload rollups across two runs at different SHAs --
    an unplanned gate-1-shaped result on the masked staging path."""
    record = roadb.MASKED_SCUT_REBUILD_WAS_A_DETERMINISM_CHECK
    assert record["payload_rollups_identical"][512].startswith("52273b8d")
    assert record["payload_rollups_identical"][768].startswith("c514727b")
    assert "different SHAs" in record["measured"]
    assert "only MANIFEST.json" in record["what_differed"]


def test_gate_3_is_measured_and_neither_graph_backbone_is_excluded(repo_root):
    """**The question resolves differently per backbone, which is the
    finding**: SR-GNN's regions are input-size-INVARIANT (fixed 42x42 map),
    AG-Net's are resolution-COVARIANT by specification (image-space). One
    rule for both would have refused a design as a defect (R2)."""
    import yaml

    record = roadb.GRAPH_GEOMETRY_AT_RESOLUTION
    assert record["srgnn"]["answer"].startswith("YES, and input-size-INVARIANT")
    assert record["srgnn"]["maps"] == {224: (7, 7), 512: (16, 16), 768: (24, 24)}
    assert record["srgnn"]["interpolation_into_42"][224] == 6.0
    assert record["srgnn"]["interpolation_into_42"][768] == 1.75
    assert "COMPOSE" in record["srgnn"]["composition"]
    assert record["agnet"]["answer"].startswith("YES, and resolution-COVARIANT")
    assert "not a defect" in record["agnet"]["answer"]
    assert "roi_align" in record["local_wheel_defect"]
    assert "DISCHARGED" in record["local_wheel_defect"], (
        "the pinned run executed the real op; the substitute caveat must "
        "not read as still carried"
    )
    assert "all four" in record["consequence"]

    # The 36-vs-37 resolution: two quantities under one name, in the gate's
    # own report -- and the count law is conditional, so 'constant by
    # construction' is qualified where it was stated.
    gap = record["agnet"]["thirty_six_vs_thirty_seven"]
    assert "one name" in gap["was"]
    assert "28" in gap["not"], "a cluster shortfall would give 28, not 36"
    assert "2 regions" in gap["law_is_conditional"]
    from cleft.models import agnet as agnet_module

    assert agnet_module.region_count() == 37
    assert agnet_module.KAPPA * (agnet_module.KAPPA + 1) // 2 == 36
    assert "ceiling, not a constant" in (agnet_module.region_count.__doc__)

    # The interpolation factors really are 42 over the measured maps.
    for size, (h, w) in record["srgnn"]["maps"].items():
        assert record["srgnn"]["interpolation_into_42"][size] == (
            pytest.approx(42 / w, abs=5e-3)
        ), size

    # And the gate is wired: task, schema, config -- per-backbone contracts
    # in the source, placement drift reported rather than gated.
    from cleft.run import TASKS

    assert TASKS["roadb_graph_geometry_gate"].__name__ == "graph_geometry_gate"
    payload = yaml.safe_load(
        (repo_root / "configs" / "roadb_p6_graph_geometry_gate.yaml")
        .read_text(encoding="utf-8")
    )
    assert payload["task"]["kind"] == "roadb_graph_geometry_gate"
    assert payload["task"]["sizes"] == [224, 512, 768]
    assert payload["task"]["backbones"] == ["srgnn", "agnet"]
    assert payload["tier"] == "keeper"
    assert "inputs" not in payload

    import inspect

    from cleft import roadb_tasks

    source = inspect.getsource(roadb_tasks.graph_geometry_gate)
    assert "reported, not gated" in source
    assert "boxes_input_size_invariant" in source
    assert "placement_varies_with_resolution" in source
    assert "REFUSED" in source


def test_the_detail_strip_keeps_one_setting_per_row():
    """**A row that mixes settings makes the rows stop meaning one thing**
    (SHEET_DENSITY: one row per setting). A setting wraps across rows at the
    column width instead of truncating into another setting's row."""
    from cleft import roadb_sheet

    crops = _crops()
    flagged = {
        "roadB_p2_512_square_g1": [1],
        "roadB_p2_768_square_g1": [1, 2, 3],
    }
    strip = roadb_sheet.detail_strip(crops, flagged, max_patients=6, columns=2)
    assert strip["columns"] == 2
    assert strip["labels"] == [
        "1 512 sq g1 NATIVE 64px", "",
        "1 768 sq g1 NATIVE 64px", "2 768 sq g1 NATIVE 64px",
        "3 768 sq g1 NATIVE 64px", "",
    ]
    assert strip["shown"]["roadB_p2_768_square_g1"]["n_shown"] == 3


def test_identical_g1_g2_residuals_are_by_construction_not_aliasing(tmp_path):
    """**[MEASURED 2026-08-09, first gate run] Within an aspect, g1 and g2
    record IDENTICAL residuals to full precision** -- the exact numeric
    signature the G2-aliasing defect had, so the reason is pinned here for
    the reader who meets four identical pairs.

    ``pixel_asymmetry_residual`` is a synthetic probe: it builds a blank
    frame at the setting's column count and runs it through ``unwarp``. It
    reads no staged pixel and takes no geometry argument, so the residual is
    a property of column count alone. **Identical residuals are expected;
    identical ARRAYS would be the defect** -- and the arrays differ.
    """
    import inspect

    from cleft import roadb_staging

    parameters = set(
        inspect.signature(roadb_staging.pixel_asymmetry_residual).parameters
    )
    assert "geometry" not in parameters, (
        "the probe grew a geometry argument; the identity reasoning is stale"
    )
    assert "size" in parameters

    record = roadb.RESIDUAL_IS_GEOMETRY_BLIND
    assert "no staged pixel" in record["why"]
    assert "geometry argument" in record["why"]
    assert "ARRAYS" in record["identical_pairs_expected"] or "arrays" in (
        record["what_answers_aliasing_instead"]
    )

    g1_dir, g1 = _run_stage(tmp_path, "roadB_p2_224_nonsquare_g1", _TRIO)
    g2_dir, g2 = _run_stage(tmp_path, "roadB_p2_224_nonsquare_g2", _TRIO)
    # Exact equality, not approx: the probe is deterministic and identical
    # inputs give identical floats. That is the point being recorded.
    assert g1["pixel_residual"] == g2["pixel_residual"]
    assert g1["residual_columns"] == g2["residual_columns"]
    one = np.load(g1_dir / "staged_patient_g1_p001.npy")
    other = np.load(g2_dir / "staged_patient_g2_p001.npy")
    assert not np.array_equal(one, other), (
        "identical arrays would be the aliasing defect, not the identity"
    )


def test_the_sheet_review_is_recorded_with_its_refuted_hypothesis():
    """**[MEASURED 2026-08-09] The review passed, and it raised a hypothesis
    that then failed -- both are recorded**, because the refuted version is
    the one a reader is most likely to reach for on seeing the sheet: the
    same face drawn smaller inside a padded square, beside a full-frame
    non-square panel, READS as compressed."""
    record = roadb.SHEET_REVIEW_PASSED
    assert record["hypothesis"]["refuted"] is True
    assert "compress" in record["hypothesis"]["claim"]
    # It would have been a Road A defect touching every arm.
    assert "0.2520" in record["hypothesis"]["stakes"]
    # The measurement that killed it, with its numbers.
    assert record["refutation"]["max_deviation"] == 0.00383
    assert "0.4%" in record["refutation"]["result"]
    assert "pixel rounding" in record["refutation"]["result"]
    assert "Road A is clean" in record["refutation"]["result"]
    # And the reframing: budget, not distortion.
    assert "196" in record["reframing"]["argument"]
    assert "constant white" in record["reframing"]["argument"]
    assert "does not distort" in record["reframing"]["not"]


def test_square_staging_preserves_aspect_ratio_to_pixel_rounding():
    """The review hypothesis at the mechanism level: if square staging
    compressed faces, the content box's ratio would drift from the source's.
    It does not -- the deviation is the half-pixel rounding of each content
    dimension, which is the 0.00383 the cluster measured."""
    from cleft.geometry import staging

    for source_w, source_h in ((700, 900), (300, 405), (900, 820), (260, 240)):
        for size in (224, 512, 768):
            staged = staging.stage(
                np.zeros((source_h, source_w, 3), dtype=np.uint8), size=size
            )
            _, _, w, h = staged.content_box
            tolerance = 2.0 / min(w, h)
            assert abs(w / h - source_w / source_h) <= tolerance, (
                (source_w, source_h, size, w, h)
            )


def test_interpolation_is_governed_by_the_longer_side_not_the_shorter():
    """**[CORRECTED 2026-08-09, sheet review] The flag measured the wrong
    side.** ``staging.stage`` computes ``scale = size / max(w, h)`` -- it
    scales by the LONGER side -- so content is interpolated iff the longer
    side is under the target. The flag compared the SHORTER side, which
    over-counts: the review's 41 flagged patients looked sharp because a
    source under 768 on its shorter side and over it on its longer
    DOWNSAMPLES. The review question predicted this failure mode verbatim.
    """
    from cleft import roadb_staging
    from cleft.geometry import staging

    # Shorter side under the target, longer side over it: both content
    # dimensions land at or below the source's -- no interpolation.
    over = staging.stage(np.zeros((300, 200, 3), dtype=np.uint8), size=224)
    _, _, w, h = over.content_box
    assert w <= 200 and h <= 224

    # Longer side under the target: both dimensions exceed the source's --
    # genuine interpolation.
    under = staging.stage(np.zeros((200, 180, 3), dtype=np.uint8), size=224)
    _, _, w, h = under.content_box
    assert w > 180 and h > 200

    # The flag now takes the longer side, and says so.
    flags = roadb_staging.upsampling_flags([300.0], 224)
    assert flags["n_upsampled"] == 0, (
        "the old shorter-side rule would have flagged this patient"
    )
    assert "longer side" in flags["measures"]

    record = roadb.UPSAMPLING_FLAG_MEASURED_THE_WRONG_SIDE
    assert "scale by the LONGER side" in record["why_wrong"]
    assert "ceiling" in record["ceilings_still_hold"]
    assert "pixels" in record["v1_manifests"], (
        "the record must say the staged pixels are unaffected"
    )


def test_the_measured_interpolation_counts_are_recorded_with_their_reading():
    """**[MEASURED 2026-08-09] Longer side below target over the 237 frontal
    sources: 0 at 224, 1 at 512, 21 at 768.** The shorter-side flag
    overcounted 768 by twenty -- the twenty sharp panels the review saw --
    so the true 768 rate is 8.9%, not 17.3%."""
    record = roadb.MEASURED_INTERPOLATION_COUNTS
    assert record["by_target"] == {224: 0, 512: 1, 768: 21}
    assert record["n_frontal"] == roadb.N_FRONTAL == 237
    assert record["longer_side_stats"] == {"median": 2758, "p05": 732, "min": 340}
    assert "8.9%" in record["rate_768"] and "17.3%" in record["rate_768"]
    assert "twenty" in record["overcount"]
    # The reading the resolution trend is read against: 512 effectively
    # clean, 768 not -- and both-ways is a CAVEAT on one cell, not a mixed
    # population.
    assert "effectively clean" in record["reading"][512]
    assert "NOT clean" in record["reading"][768]
    assert "CAVEAT" in record["reporting"]
    assert "mixed population" in record["reporting"]
    # The reviewed strip was drawn from the superset, so it missed nothing.
    assert "superset" in record["v1_strip_missed_nothing"]


def test_the_upsampling_gate_is_exact_for_the_measured_population():
    """Two layers now. **Exact against the measured frontal longer-side
    counts when the population is the 237** -- the check the design wanted,
    finally with population-and-quantity-matched numbers, so a v1-style
    shorter-side recount (41 at 768) can no longer pass silently. **The 473
    shorter-side ceilings for any other population**, valid a fortiori."""
    from cleft import roadb_staging

    # Exact layer: the measured counts pass...
    ok = roadb_staging.upsampling_flags(
        np.array([700.0] * 21 + [2758.0] * 216), 768
    )
    assert ok["n"] == 237 and ok["n_upsampled"] == 21
    roadb_staging.assert_upsampling_counts(ok)
    assert "473" in ok["population"]

    # ...and any drift refuses, in BOTH directions -- including the old
    # shorter-side count, which is the misreading this layer now blocks.
    for count in (41, 20):
        drifted = roadb_staging.upsampling_flags(
            np.array([700.0] * count + [2758.0] * (237 - count)), 768
        )
        with pytest.raises(roadb_staging.StagingError,
                           match="measured frontal count is 21"):
            roadb_staging.assert_upsampling_counts(drifted)

    at_512 = roadb_staging.upsampling_flags(
        np.array([300.0] * 3 + [900.0] * 234), 512
    )
    with pytest.raises(roadb_staging.StagingError,
                       match="measured frontal count is 1"):
        roadb_staging.assert_upsampling_counts(at_512)

    at_224 = roadb_staging.upsampling_flags(
        np.array([200.0] + [900.0] * 236), 224
    )
    with pytest.raises(roadb_staging.StagingError,
                       match="measured frontal count is 0"):
        roadb_staging.assert_upsampling_counts(at_224)

    # Ceiling layer: a different population size falls back to the 473
    # shorter-side ceilings -- a fixture or a future cohort is not held to
    # another cohort's measurement.
    other = roadb_staging.upsampling_flags(
        np.array([300.0] * 7 + [900.0] * 466), 512
    )
    roadb_staging.assert_upsampling_counts(other)
    above = roadb_staging.upsampling_flags(
        np.array([300.0] * 8 + [900.0] * 465), 512
    )
    with pytest.raises(roadb_staging.StagingError, match="brief measured 7"):
        roadb_staging.assert_upsampling_counts(above)


def test_v1_manifests_are_read_as_the_deprecated_quantity_they_are():
    """**The side-naming alone is NOT enough for v1**: those manifests
    predate the ``measures`` key and cannot be amended -- artifacts are
    immutable and their rollups are recorded in the gate verdict and the
    sheets config, so an edit would stale every one of those records. The
    corrected figures therefore ride along READER-side."""
    from cleft import roadb_staging

    v1 = {"upsampling": {"target": 768, "n_upsampled": 41, "per_patient": []}}
    quantity = roadb_staging.manifest_upsampling_quantity(v1)
    assert quantity["deprecated"] is True
    assert "shorter side" in quantity["measures"]
    assert quantity["corrected_counts"] == {224: 0, 512: 1, 768: 21}

    current = {"upsampling": {"target": 768, "measures": "longer side ..."}}
    quantity = roadb_staging.manifest_upsampling_quantity(current)
    assert quantity["deprecated"] is False
    assert "corrected_counts" not in quantity
