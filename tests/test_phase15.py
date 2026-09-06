"""Phase 15's registration: the survey, the frontal screen, mediapipe."""

from __future__ import annotations

from pathlib import Path

from cleft import phase15


def test_the_survey_names_its_lead_with_verified_source_details():
    record = phase15.CANDIDATE_SURVEY_REGISTERED
    assert "the maintainer's acquisition gate" in record["registered"]
    lead = record["lead_candidate"]
    assert lead["name"] == "MEBeauty"
    assert "Lebedeva" in lead["source"]
    assert "fbplab/MEBeauty-database" in lead["source"]
    assert "2,550" in lead["size"]
    assert "SIX ethnicities" in lead["demographics"]
    assert "31% of males under 20" in lead["demographics"]
    assert "~300 raters" in lead["labels"]
    # The landmarks are shipped -- the detector verdict stays dormant.
    assert "SHIPPED" in lead["landmarks"]
    assert "MEDIAPIPE_VERDICT_ENCODED" in lead["landmarks"]
    assert "CITATION\nREQUIRED" in lead["license"] or "CITATION" in (
        lead["license"]
    )
    # Acquisition is a deliberate act, and code downloads nothing.
    assert "THE ACQUISITION ACT" in record["acquisition"]
    assert "Nothing\ndownloads by code" in record["acquisition"] or (
        "Nothing" in record["acquisition"]
    )


def test_the_mechanism_asymmetry_is_encoded_as_binding():
    mechanisms = phase15.CANDIDATE_SURVEY_REGISTERED["mechanisms"]
    assert "ALIVE AND STRONG" in mechanisms["i_demographics"]
    assert "un-measured axis" in mechanisms["i_demographics"]

    # Mechanism ii is DEAD AND INVERTED, and the asymmetry it creates
    # binds every future reading -- win strong, loss ambiguous.
    scale = mechanisms["ii_scale"]
    assert "DEAD AND INVERTED" in scale
    assert "HALF" in scale and "5,500" in scale
    assert "IN SCUT'S FAVOUR" in scale
    assert "WIN is STRONG" in scale
    assert "LOSS is AMBIGUOUS" in scale
    assert "not attributable" in scale
    assert "No reading\nmay be registered" in scale or "No reading" in scale

    assert "ALIVE REGARDLESS" in mechanisms["iii_ranking_deliverable"]

    alternates = phase15.CANDIDATE_SURVEY_REGISTERED[
        "alternates_named_not_pursued"
    ]
    assert "DOMINATED" in alternates
    assert "WRONG LABEL\nTYPE" in alternates or "WRONG LABEL" in alternates
    assert "UNVERIFIED ALTERNATES" in alternates
    assert "presenting one as fresh" in alternates


def test_the_frontal_screen_is_registered_first_with_its_discipline():
    record = phase15.FRONTAL_FRACTION_FIRST
    assert "before any staging or pretraining" in record["registered"]
    assert "IN-THE-WILD" in record["why_first"]
    assert "WORSENS the scale asymmetry" in record["why_first"]
    assert "BY\nHOW MUCH" in record["why_first"] or "BY" in record["why_first"]
    # Shipped landmarks, threshold declared before the screen runs.
    assert "SHIPPED landmarks" in record["the_screen"]
    assert "DECLARED IN THE CONFIG" in record["the_screen"]
    assert "not a threshold" in record["the_screen"]
    assert "FIRST banked number" in record["first_banked_number"]
    assert "not the headline 2,550" in record["first_banked_number"]


def test_the_mediapipe_verdict_is_in_the_repo_at_last():
    record = phase15.MEDIAPIPE_VERDICT_ENCODED
    assert "July session handoff" in record["encoded"]
    assert "GL libraries" in record["verdict"]
    assert "no-root" in record["verdict"]
    assert "wrong-version" in record["verdict"]
    assert "AVOID DETECTORS" in record["verdict"]
    assert "Dormant for MEBeauty" in record["verdict"]

    assert set(phase15.summary()) == {
        "survey", "frontal_first", "mediapipe",
        "landing", "clone_contents", "exit_criteria", "stop_1",
        "stop_1_first_run", "stop_1_banked", "stop_2_plan",
        "stop_2_rulings", "stop_2a", "stop_2a_banked",
        "mapping_sheets", "stop_2a_ii", "stop_2a_ii_banked",
        "the_119_refuted", "overlay_defect", "label_defect",
        "diagnosis_verdicts", "withdrawals", "anatomically_blind",
        "second_eye_pass", "bounds_readings", "bounds_distribution",
        "bounds_tolerance", "tolerance_consequence", "stop_2b",
        "parity_none", "price_sentence", "stop_2b_banked",
        "padding_property", "staged_sheets", "framing_varies",
        "stop_3_proposed", "stop_3_amended", "original_arm_pads",
        "key_mismatch", "v1_survives", "mechanism_not_rate",
        "synthetic_sweep", "pipeline_not_sources",
        "pretraining_banked", "stop_4_proposed", "mapping_verified",
        "probe_layer_ruled", "trainable_guard_fired", "stop_4",
        "sequence_renumbered_3", "phase_16_scheduled", "anchor_loop",
        "sequence_renumbered_4",
        "stop_4a_banked", "two_digests", "stop_4b_banked",
        "reading_1", "reading_2", "reading_3", "phase_6_reframed",
        "exit_walk", "phase_16_verdict", "criterion_1_amended",
        "closing",
    }


# --------------------------------------------------------------------------
# 2026-08-24, the landing corrected; criteria agreed; stop 1 built
# --------------------------------------------------------------------------


def test_the_landing_correction_and_clone_contents_are_recorded():
    landing = phase15.LANDING_CORRECTED
    assert "SOURCE dataset" in landing["the_correction"]
    assert "/home/user/codex/mebeauty/MEBeauty-database" in (
        landing["the_correction"]
    )
    assert "REJECTED" in landing["the_correction"]
    assert "source-vs-derived boundary" in landing["why"]
    assert "gitignore hazard" in landing["why"]
    assert "${CLEFT_MEBEAUTY_ROOT}" in landing["portability"]
    assert "${CLEFT_SCUT_ROOT}" in landing["portability"]

    contents = phase15.CLONE_CONTENTS
    assert "651M, 9,808 files" in contents["contents"]
    assert "DIRECTORY" in contents["contents"]
    assert "BANKED" in contents["survey_item_scores"]
    assert "per-rater or distributional" in contents["survey_item_scores"]
    assert "does NOT consume by default" in (
        contents["survey_item_cropped_images"]
    )
    assert "DATED DECISION" in contents["survey_item_cropped_images"]
    # The declare pass's numbers differ from the ls-level listing, and
    # the record says why -- so neither reads as an error later.
    reconciliation = contents["declare_reconciliation"]
    assert "9,808" in reconciliation and "9,806" in reconciliation
    assert "242,062,060" in reconciliation
    assert ".git" in reconciliation and "204 MiB" in reconciliation
    assert "6719959b" in reconciliation
    assert "AGREEMENT, not drift" in reconciliation


def test_the_seven_exit_criteria_are_agreed_as_drafted():
    record = phase15.PHASE_15_EXIT_CRITERIA
    assert "all seven as drafted" in record["agreed"]
    criteria = record["criteria"]
    assert len(criteria) == record["expected_count"] == 7
    assert [c.split(".")[0] for c in criteria] == [
        str(i) for i in range(1, 8)
    ]
    joined = " ".join(criteria)
    assert "CLEFT-SIDE TRANSFER" in joined
    assert "0.7893" in joined
    assert "SCREENED count" in joined
    assert "levelling-OFF" in joined and "CLEFT_AR" in joined
    assert "VARIANT_FOR_INIT" in joined
    assert "BANKED Phase 6 figures" in joined and "0.2520" in joined
    assert "DECLARED BEFORE ANY NUMBER" in joined
    assert "mechanism-ii asymmetry sentence" in joined
    assert "READING_COUNT_GUARD" in joined


def test_the_survey_task_keeps_its_registered_boundaries():
    import inspect

    from cleft import run as run_module

    assert "survey_mebeauty" in run_module.TASKS
    source = inspect.getsource(run_module.task_survey_mebeauty)

    # Public data: no gate, no patient artifact, SHAREABLE sheets.
    assert "cleft_reconstruction_acknowledged" not in source
    assert "staged_patient" not in source
    assert 'tier="SHAREABLE"' in source
    assert 'tier="CLUSTER-ONLY"' not in source
    # The shipped-but-not-consumed boundary, by CALL SHAPE (the
    # word-matching lesson): the only directory whose files the task
    # opens is original_images -- via the originals index -- and the
    # summary names the not-consumed set explicitly.
    assert '(root / "original_images").rglob' in source
    assert "render.load_image(originals[" in source
    # [REFINED 2026-08-24, STOP_1_FIRST_RUN] The family measurement
    # LISTS cropped_images basenames (metadata for the key-resolution
    # report); no crop PIXEL is ever opened -- load_image has exactly
    # one call site, over originals.
    assert source.count("render.load_image(") == 1
    assert '(root / "cropped_images").rglob' in source
    assert 'root / "FaceNet_512_features"' not in source
    assert 'root / "geometric_features.csv"' not in source
    assert '"not_consumed"' in source
    # Counts are reported, never gated (a survey measures).
    assert "report, never gate" in source
    # The threshold comes from the config, before the screen.
    assert 'task["frontal_max_offset"]' in source

    # [2026-08-24, after the first run's refusal] The parser now
    # targets the MEASURED format (packed cell, 136 numerics, dlib-68)
    # and refuses surprises with the evidence in the message.
    parser = inspect.getsource(run_module._mebeauty_landmarks)
    assert "measured" in parser
    assert "136" in parser
    assert "first 60 chars" in parser.lower()


def test_the_pose_statistic_behaves_as_registered():
    """Measured: frontal-centred landmarks pass, skewed sets fail, and
    the statistic needs no landmark semantics."""
    import numpy as np

    # A symmetric x-set: centroid at the box midpoint -> offset 0.
    xs = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    offset = abs(xs.mean() - (xs.max() + xs.min()) / 2) / (
        xs.max() - xs.min()
    )
    assert offset == 0.0

    # A profile-like skew: mass piled on one side -> large offset.
    skewed = np.array([10.0, 12.0, 13.0, 14.0, 50.0])
    skewed_offset = abs(
        skewed.mean() - (skewed.max() + skewed.min()) / 2
    ) / (skewed.max() - skewed.min())
    assert skewed_offset > 0.2


def test_the_survey_config_is_env_portable_with_declared_threshold():
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "configs" / "p15_survey.yaml").read_text(
        encoding="utf-8"
    )
    payload = yaml.safe_load(text)
    task = payload["task"]
    assert task["kind"] == "survey_mebeauty"
    assert task["frontal_max_offset"] == 0.08
    assert task["published_images"] == 2550
    entry = payload["inputs"][0]
    assert entry["name"] == "mebeauty_root"
    assert entry["path"] == "${CLEFT_MEBEAUTY_ROOT}"
    # The two-pass state is consistent with the header.
    if set(entry["rollup_sha256"]) == {"0"}:
        assert "PLACEHOLDER" in text
    assert "CLEFT_MEBEAUTY_ROOT=/home/user/codex/mebeauty" in text
    assert "NO GATE" in text

    spec = importlib.util.spec_from_file_location(
        "generate_phase15_configs",
        repo / "scripts" / "generate_phase15_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, the first run: refusal honoured, format measured, parser fixed
# --------------------------------------------------------------------------


def test_the_first_run_is_banked_with_the_measured_format():
    record = phase15.STOP_1_FIRST_RUN
    assert "2,539" in record["inventory_banked"]
    assert "ELEVEN SHORT" in record["inventory_banked"]
    assert "TWO FAMILIES" in record["inventory_banked"]
    assert "UNREAD" in record["inventory_banked"]
    assert "did its job" in record["the_refusal"]
    assert "136" in record["the_measured_format"]
    assert "68 (x, y) pairs" in record["the_measured_format"]
    assert "THIRD SCORE SURFACE" in record["the_measured_format"]
    # README silent -> the families are measured, not guessed.
    assert "does NOT distinguish" in (
        record["families_unresolved_readme_silent"]
    )
    assert "MEASURED, not guessed" in (
        record["families_unresolved_readme_silent"]
    )
    assert "REPORT, NEVER\nGATE" in record["third_score_surface"] or (
        "REPORT, NEVER" in record["third_score_surface"]
    )
    assert "opens NO\ncrop pixel" in record["boundary_refined"] or (
        "NO" in record["boundary_refined"]
    )
    assert record["rerun"].startswith("p15-survey-2")


def test_the_parser_handles_the_measured_format_and_refuses_surprises(
    tmp_path,
):
    import numpy as np

    from cleft.run import _mebeauty_landmarks

    pairs = ",".join(str(10 + i) for i in range(136))
    good = (
        ",image,score,landmarks\n"
        '0,/home/ubuntu/x/images/a.jpg,3.79,"' + pairs + ',"\n'
    )
    path = tmp_path / "landmarks.csv"
    path.write_text(good, encoding="utf-8")
    xs, scores = _mebeauty_landmarks(path)
    assert len(xs["a.jpg"]) == 68
    assert scores["a.jpg"] == 3.79

    # A cell with the wrong count refuses with the cell's prefix.
    path.write_text(
        ',image,score,landmarks\n0,/x/b.jpg,4.2,"1,2,3,4"\n',
        encoding="utf-8",
    )
    import pytest

    with pytest.raises(ValueError, match="First 60 chars"):
        _mebeauty_landmarks(path)

    # A different header refuses with the measured one in the message.
    path.write_text("image,x1,y1\na.jpg,1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="measured"):
        _mebeauty_landmarks(path)


# --------------------------------------------------------------------------
# 2026-08-24, stop 1 banked (the funnel, the label source); stop 2 proposed
# --------------------------------------------------------------------------


def test_stop_one_is_banked_as_a_funnel_with_every_loss_named():
    record = phase15.STOP_1_BANKED
    assert "single clean attempt" in record["banked"]
    funnel = record["the_funnel"]
    # Each step and each loss, in order.
    for step in ("2,550", "2,539", "2,445", "1,519"):
        assert step in funnel, step
    assert "eleven never shipped" in funnel
    assert "~94" in funnel
    assert "62.1%" in funnel
    assert "0.08" in funnel
    assert "first banked number is 1,519" in funnel

    # The scale asymmetry now carries the REAL figure.
    scale = record["scale_asymmetry_restated"]
    assert "0.28x" in scale and "1,519 vs 5,500" in scale
    assert "0.46x headline" in scale
    assert "AT\n0.28x THE DATA" in scale or "0.28x THE DATA" in scale
    assert "loss is ambiguous" in scale

    # The label source was MEASURED, not read off a silent README.
    label = record["label_source_decided"]
    assert "KEY-RESOLUTION MEASUREMENT" in label
    assert "silent README" in label
    assert "2,198 rows" in label
    assert "cropped-keyed subset" in label
    assert "universal family is the phase's label source" in label

    assert "0.07" in record["third_surface_banked"]
    assert "report-never-gate" in record["third_surface_banked"]


def test_the_eye_review_records_the_threshold_trade_without_moving_it():
    record = phase15.STOP_1_BANKED
    review = record["eye_review_PASS"]
    assert "PASS at 0.08" in review
    assert "threshold UNCHANGED" in review
    assert "mechanism i in\npictures" in review or "mechanism i" in review
    assert "three-quarter/profile" in review

    # The recoverable band is recorded as a PRICE, and the price was
    # accepted rather than renegotiated after the fact.
    trade = record["threshold_trade_recorded"]
    assert "0.08-0.12" in trade
    assert "~5% usable count" in trade
    assert "pose purity" in trade
    assert "ACCEPTED BY REVIEW" in trade
    assert "no post-hoc adjustment" in trade
    assert "is not a threshold" in trade
    assert "SCUT-precedented" in record["occlusion_note"]


def test_the_stop_two_plan_measures_the_mapping_before_trusting_it():
    record = phase15.STOP_2_PLAN_PROPOSED
    assert "NOTHING built" in record["proposed"]
    assert "the ruling" in record["proposed"]
    # Two sub-stops: verification and sheets BEFORE the batch.
    assert "stop 2a" in record["structure"] and "stop 2b" in record["structure"]
    assert "no pixels" in record["structure"]
    assert "BEFORE any embedding" in record["structure"]

    mapping = record["mapping_proposed_not_trusted"]
    assert "PROPOSED" in mapping and "VERIFIED" in mapping
    # The dlib-68 anchors, and the SCUT lesson cited by its own words.
    assert "36-41" in mapping and "42-47" in mapping
    assert "33" in mapping and "51" in mapping
    assert "17-26" in mapping and "48-59" in mapping
    assert "looks plausible and is wrong" in mapping
    assert "MEASURED, then sheets, then batch" in mapping
    # The image-left/subject-right trap is named, not assumed away.
    assert "image-left" in mapping

    split = record["split_proposed"]
    assert "SHIPPED" in split
    assert "never our own re-cut" in split
    assert "MEASURED at stop 2a" in split

    inherit = record["inheritances"]
    assert "levelling OFF" in inherit
    assert "CLEFT_AR" in inherit
    assert "parity checks re-run" in inherit
    assert "G1 AND G2" in inherit
    assert "VARIANT_FOR_INIT" in inherit

    # The derived artifact goes under data/ -- the mirror image of the
    # source-clone correction.
    assert "data/mebeauty_staged" in record["configs_planned"]
    assert "where\nthe source clone does not" in (
        record["configs_planned"]
    ) or "the source clone does not" in record["configs_planned"]
    assert "two" in record["eye_gates"]


# --------------------------------------------------------------------------
# 2026-08-24, both rulings given; stop 2a built (mapping verification)
# --------------------------------------------------------------------------


def _synthetic_dlib68(rng, midline=100.0, width=80.0):
    """A SYMMETRIC dlib-68 face: mirror pairs mirrored about the
    midline, 33 and 51 on it, eyes where the convention puts them."""
    import numpy as np

    points = np.zeros((68, 2), dtype=float)
    for i in range(17):
        t = (i - 8) / 8.0
        points[i] = (midline + t * width, 150 + 40 * (1 - t * t))
    for i in range(5):
        points[17 + i] = (midline - width * (0.75 - 0.12 * i), 40)
        points[26 - i] = (midline + width * (0.75 - 0.12 * i), 40)
    for i in range(4):
        points[27 + i] = (midline, 50 + 10 * i)
    points[31] = (midline - 18, 95)
    points[32] = (midline - 9, 98)
    points[33] = (midline, 100)
    points[34] = (midline + 9, 98)
    points[35] = (midline + 18, 95)
    for i in range(6):
        angle = i * np.pi / 3
        points[36 + i] = (
            midline - 40 + 10 * np.cos(angle), 60 + 5 * np.sin(angle)
        )
        points[42 + i] = (
            midline + 40 - 10 * np.cos(angle), 60 + 5 * np.sin(angle)
        )
    points[48] = (midline - 30, 125)
    points[54] = (midline + 30, 125)
    for i, dx in enumerate((-18, -8, 0, 8, 18)):
        points[49 + i] = (midline + dx, 118)
    for i, dx in enumerate((18, 8, 0, -8, -18)):
        points[55 + i] = (midline + dx, 132)
    points[51] = (midline, 118)
    for i in range(8):
        points[60 + i] = (midline + (i - 3.5) * 5, 125)
    return points + rng.normal(0, 0.6, points.shape)


def _verify(faces):
    from cleft import mebeauty

    return mebeauty.verify_mapping(
        faces, midline_tolerance=0.02, eye_symmetry_tolerance=0.02,
        corner_margin=0.05,
    )


def test_the_mapping_checks_pass_a_correct_mapping():
    """Measured, not assumed: a symmetric dlib-68 face clears all three
    checks at the declared thresholds."""
    import numpy as np

    rng = np.random.default_rng(1337)
    report = _verify({f"f{i}": _synthetic_dlib68(rng) for i in range(120)})
    assert report["all_pass"] is True
    assert report["check_1_passes"] and report["check_2_passes"]
    assert report["check_3_passes"]
    assert report["check_2_image_left_fraction"] == 1.0
    assert report["check_3_corner_widest_fraction"] == 1.0
    # The rule travels in the report itself.
    assert "MAPPING reopens" in report["the_rule"]
    assert report["thresholds_declared"]["midline_tolerance"] == 0.02


def test_each_check_fails_the_wrong_mapping_it_exists_for():
    """The property that matters: a check that cannot fail verifies
    nothing. Each of the three wrong mappings is caught by its own
    check."""
    import numpy as np

    rng = np.random.default_rng(1337)
    good = {f"f{i}": _synthetic_dlib68(rng) for i in range(120)}

    # 1. The IMAGE-LEFT / subject-right trap: eye groups swapped.
    swapped = {}
    for key, points in good.items():
        p = points.copy()
        p[36:42], p[42:48] = points[42:48].copy(), points[36:42].copy()
        swapped[key] = p
    swap = _verify(swapped)
    assert swap["all_pass"] is False
    assert swap["check_2_passes"] is False
    assert swap["check_2_image_left_fraction"] == 0.0

    # 2. A midline landmark that is not on the midline.
    offmid = {}
    for key, points in good.items():
        p = points.copy()
        p[33] = (p[33][0] + 20, p[33][1])
        offmid[key] = p
    off = _verify(offmid)
    assert off["all_pass"] is False
    assert off["check_1_passes"] is False
    assert off["check_1_midline"]["33"]["median"] > 0.02

    # 3. Corners that are not the widest symmetric pair -- SCUT's own
    # criterion, which caught its wrong index.
    narrow = {}
    for key, points in good.items():
        p = points.copy()
        p[48] = (p[48][0] + 25, p[48][1])
        p[54] = (p[54][0] - 25, p[54][1])
        narrow[key] = p
    bad_corners = _verify(narrow)
    assert bad_corners["all_pass"] is False
    assert bad_corners["check_3_passes"] is False


def test_the_mapping_module_tags_itself_literature_and_names_the_trap():
    import inspect

    from cleft import mebeauty

    source = inspect.getsource(mebeauty)
    assert "[LITERATURE" in source
    assert "not measured" in source.lower()
    # The SCUT lesson is quoted (wrapped across lines in the
    # docstring, so match the distinctive fragment).
    assert "plausible and is wrong" in source
    # The trap is named where the constant is defined.
    assert "IMAGE-LEFT / SUBJECT-RIGHT TRAP" in source
    assert mebeauty.EYE_IMAGE_LEFT == (36, 37, 38, 39, 40, 41)
    assert mebeauty.EYE_IMAGE_RIGHT == (42, 43, 44, 45, 46, 47)
    assert mebeauty.MIDLINE_LANDMARKS == (33, 51)
    assert mebeauty.MOUTH_CORNERS == (48, 54)
    assert mebeauty.N_LANDMARKS == 68
    # The direction of accommodation, stated in the module.
    assert "never the check" in source
    # verify_mapping tunes nothing: no threshold is computed from data.
    verify = inspect.getsource(mebeauty.verify_mapping)
    assert "percentile(" in verify  # reports distributions...
    assert "tolerance =" not in verify  # ...but assigns no threshold


def test_the_rulings_and_the_2a_build_are_recorded():
    rulings = phase15.STOP_2_RULINGS
    assert "[LITERATURE]" in rulings["mapping_is_literature_not_measured"]
    assert "NOT measured" in rulings["mapping_is_literature_not_measured"]
    assert "NEVER ADJUSTED" in rulings["the_direction_of_accommodation"]
    assert "IS the failure mode" in rulings["the_direction_of_accommodation"]
    assert "36-41 is the IMAGE-LEFT group" in rulings["the_image_left_trap"]
    assert "MEASURED, then SHEETS, then BATCH" in rulings["order_is_binding"]
    # The split's added reason, in the maintainer's own terms.
    split = rulings["split_approved_with_its_reason"]
    assert "HOME-MADE PARTITION" in split
    assert "CONFOUND THE PHASE CAN AVOID" in split
    assert "LOWER_MARGIN 0.06" in rulings["inherited_unchanged"]

    built = phase15.STOP_2A_BUILT
    assert "NOT launched" in built["built"]
    assert "NO PIXEL is read for the checks" in built["task"]
    assert "tunes\nnothing" in built["checks_own_the_mapping"] or (
        "tunes" in built["checks_own_the_mapping"]
    )
    assert "a failed verification is a finding" in (
        built["checks_own_the_mapping"]
    )
    assert "cannot corrupt a\nmeasurement" in (
        built["no_pixel_for_the_verdict"]
    ) or "cannot corrupt" in built["no_pixel_for_the_verdict"]


def test_the_2a_task_and_config_hold_the_order_and_the_thresholds():
    import importlib.util
    import inspect

    import yaml

    from cleft import run as run_module

    assert "verify_mebeauty_mapping" in run_module.TASKS
    source = inspect.getsource(run_module.task_verify_mebeauty_mapping)
    # The screen is REPRODUCED, not re-decided: a mismatch refuses.
    assert 'task["expect_usable"]' in source
    assert "measuring different sets" in source
    # Thresholds arrive from the config; nothing is computed here.
    for key in (
        "midline_tolerance", "eye_symmetry_tolerance", "corner_margin",
    ):
        assert f'task["{key}"]' in source, key
    # The verdict is landmarks-only: pixels appear only in the sheet
    # section, after the report is built.
    assert source.index("verify_mapping(") < source.index(
        "render.load_image("
    )
    assert "the MAPPING reopens" in source

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "configs" / "p15_verify_mapping.yaml").read_text(
        encoding="utf-8"
    )
    payload = yaml.safe_load(text)
    task = payload["task"]
    assert task["kind"] == "verify_mebeauty_mapping"
    assert task["expect_usable"] == 1519
    assert task["frontal_max_offset"] == 0.08
    assert task["midline_tolerance"] == 0.02
    assert task["eye_symmetry_tolerance"] == 0.02
    assert task["corner_margin"] == 0.05
    assert "[LITERATURE]" in text
    assert "NEVER" in text and "adjusted" in text
    assert "MEASURED, then SHEETS, then\n# BATCH" in text or (
        "MEASURED, then SHEETS" in text
    )
    # Same root, no new declaration.
    entry = payload["inputs"][0]
    assert entry["path"] == "${CLEFT_MEBEAUTY_ROOT}"
    survey = yaml.safe_load(
        (repo / "configs" / "p15_survey.yaml").read_text(encoding="utf-8")
    )
    assert entry["rollup_sha256"] == survey["inputs"][0]["rollup_sha256"]

    spec = importlib.util.spec_from_file_location(
        "generate_phase15_configs_2a",
        repo / "scripts" / "generate_phase15_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, 2a banked and reviewed; the landmark-quality screen built
# --------------------------------------------------------------------------


def _wellformed_dlib68(rng, midline=250.0, width=200.0, top=120.0):
    """A well-formed face inside a 500x600 image."""
    import numpy as np

    p = np.zeros((68, 2), dtype=float)
    for i in range(17):
        t = (i - 8) / 8.0
        p[i] = (midline + t * width, top + 260 + 90 * (1 - t * t))
    for i in range(5):
        p[17 + i] = (midline - width * (0.75 - 0.12 * i), top + 20)
        p[26 - i] = (midline + width * (0.75 - 0.12 * i), top + 20)
    for i in range(4):
        p[27 + i] = (midline, top + 60 + 25 * i)
    p[31] = (midline - 45, top + 190)
    p[32] = (midline - 22, top + 196)
    p[33] = (midline, top + 200)
    p[34] = (midline + 22, top + 196)
    p[35] = (midline + 45, top + 190)
    for i in range(6):
        a = i * np.pi / 3
        p[36 + i] = (midline - 95 + 25 * np.cos(a), top + 95 + 12 * np.sin(a))
        p[42 + i] = (midline + 95 - 25 * np.cos(a), top + 95 + 12 * np.sin(a))
    p[48] = (midline - 75, top + 265)
    p[54] = (midline + 75, top + 265)
    for i, dx in enumerate((-45, -20, 0, 20, 45)):
        p[49 + i] = (midline + dx, top + 248)
    for i, dx in enumerate((45, 20, 0, -20, -45)):
        p[55 + i] = (midline + dx, top + 282)
    p[51] = (midline, top + 248)
    for i in range(8):
        p[60 + i] = (midline + (i - 3.5) * 12, top + 265)
    return p + rng.normal(0, 1.0, p.shape)


_SIZE = (500, 600)
_BANDS = {
    "span_band": (0.15, 1.10),
    "interocular_band": (0.25, 0.70),
    "bounds_margin": 0.0,
}


def test_the_quality_screen_passes_a_well_formed_face():
    import numpy as np

    from cleft import mebeauty

    rng = np.random.default_rng(7)
    report = mebeauty.quality_screen(
        _wellformed_dlib68(rng), _SIZE, **_BANDS
    )
    assert report["passes"] is True
    assert report["failed"] == []
    assert set(report["checks"]) == set(mebeauty.QUALITY_CHECKS)
    assert 0.15 <= report["checks"]["span"]["fraction_x"] <= 1.10
    assert 0.25 <= report["checks"]["interocular"][
        "interocular_over_span"
    ] <= 0.70


def test_each_quality_check_fails_the_failure_it_exists_for():
    """A screen that cannot fail verifies nothing. Five bad cases, each
    caught by its own check -- measured, not assumed."""
    import numpy as np

    from cleft import mebeauty

    rng = np.random.default_rng(7)
    good = _wellformed_dlib68(rng)

    # 1. An empty/all-zero row: the 119.jpg family. Hard refusal, and it
    #    SHORT-CIRCUITS -- four derived NaNs would read as four problems.
    zeros = mebeauty.quality_screen(np.zeros((68, 2)), _SIZE, **_BANDS)
    assert zeros["passes"] is False
    assert zeros["failed"] == ["degenerate"]
    assert "no usable geometry" in zeros["reason"]

    # 2. A foreign coordinate frame -- computed on another image.
    foreign = mebeauty.quality_screen(good * 3.0, _SIZE, **_BANDS)
    assert foreign["passes"] is False
    assert "bounds" in foreign["failed"]
    assert foreign["checks"]["bounds"]["n_outside"] > 60

    # 3. A collapsed detection: a tiny symmetric cloud. THE POINT of
    #    this stop -- it sails through the pose screen.
    tiny = (good - good.mean(axis=0)) * 0.05 + np.array([250.0, 300.0])
    xs = tiny[:, 0]
    pose_offset = abs(
        xs.mean() - (xs.max() + xs.min()) / 2
    ) / (xs.max() - xs.min())
    assert pose_offset <= 0.08, "the pose screen would accept this"
    collapsed = mebeauty.quality_screen(tiny, _SIZE, **_BANDS)
    assert collapsed["passes"] is False
    assert "span" in collapsed["failed"]

    # 4. A drifted constellation: brows below the mouth.
    scattered = good.copy()
    scattered[list(mebeauty.BROW_INDICES), 1] += 260
    drifted = mebeauty.quality_screen(scattered, _SIZE, **_BANDS)
    assert drifted["passes"] is False
    assert "ordering" in drifted["failed"]

    # 5. Fused eyes: everything else holds.
    fused = good.copy()
    fused[list(mebeauty.EYE_IMAGE_LEFT)] += np.array([90.0, 0.0])
    fused[list(mebeauty.EYE_IMAGE_RIGHT)] -= np.array([90.0, 0.0])
    eyes = mebeauty.quality_screen(fused, _SIZE, **_BANDS)
    assert eyes["passes"] is False
    assert "interocular" in eyes["failed"]


def test_stop_2a_is_banked_with_the_count_gap_stated():
    record = phase15.STOP_2A_BANKED
    assert "1,519" in record["usable_reproduced"]
    assert "0.01302" in record["check_1_midline"]
    assert "0.01294" in record["check_1_midline"]
    assert "1.0000" in record["check_2_eye_symmetry"]
    assert "TRAP IS CLOSED" in record["check_2_eye_symmetry"]
    assert "1.0000" in record["check_3_corners"]
    assert "ALL CHECKS PASS" in record["verdict"]
    assert "no threshold was touched" in record["verdict"]
    # The two counts are stated TOGETHER, with the shortfall named.
    gap = record["the_count_gap_stated"]
    assert "1,519" in gap and "1,326" in gap
    assert "932" in gap and "394" in gap
    assert "DO NOT COVER" in gap
    assert "193" in gap
    assert "0.24x" in record["proposed_ruling_for_2b"]
    assert "PROPOSED" in record["proposed_ruling_for_2b"]


def test_the_sheet_review_records_the_defect_and_the_withdrawal():
    record = phase15.MAPPING_SHEETS_REVIEWED
    assert "DO NOT REOPEN IT" in record["verdict"]
    assert "confirming check 2's 1.0000 by eye" in record["verdict"]
    # The named faces, so a later turn can find them.
    defect = record["the_defect_the_checks_cannot_see"]
    for name in ("girl-3956612", "man-1868320", "119.jpg"):
        assert name in defect, name
    assert "NO landmarks and NO midline" in defect
    # Prevalence is an impression and says so in its own key.
    assert "EYE_IMPRESSION" in "".join(record)
    assert "NOT A COUNT" in record["prevalence_EYE_IMPRESSION"]
    assert "never as a rate" in record["prevalence_EYE_IMPRESSION"]
    # The withdrawal, by name, with the discipline stated.
    withdrawal = record["withdrawal"]
    assert "WITHDRAWN" in withdrawal
    assert "man-2785071" in withdrawal
    assert "this was misread the pose" in withdrawal
    assert "SAME RETRACTION DISCIPLINE AS" in withdrawal
    assert "not\nquietly dropped" in withdrawal or "quietly dropped" in (
        withdrawal
    )
    assert "BLIND to it" in record["cause"]


def test_the_2a_ii_record_and_config_carry_the_screen_as_registered():
    import importlib.util

    import yaml

    record = phase15.STOP_2A_II_BUILT
    assert "NOT launched" in record["built"]
    for check in (
        "DEGENERATE", "BOUNDS", "SPAN", "ORDERING", "INTEROCULAR",
    ):
        assert check in record["the_five_checks"], check
    assert "SCALE-FREE" in record["the_five_checks"]
    # The measured table, including the case that motivates the stop.
    measured = record["measured_against_bad_cases"]
    assert "cannot fail verifies nothing" in measured
    assert "0.0399" in measured and "0.0006" in measured
    assert "sail through the 0.08 screen" in measured
    assert "0.0252" in measured
    # The named diagnosis is structural, and 119's cause is a HYPOTHESIS.
    named = record["the_named_diagnosis"]
    assert "EXPLAIN ITSELF BY NAME" in named
    assert "MAY NOT BE UNIQUE TO IT" in named
    assert "DECLARED LIST" in named
    hypothesis = record["the_119_hypothesis_not_a_finding"]
    assert "HYPOTHESIS" in hypothesis
    assert "NOT\nMEASURED" in hypothesis or "NOT MEASURED" in hypothesis
    assert "discarded rather than banked" in hypothesis
    assert "PRICE OF" in record["the_falling_count_is_the_price"]
    # Assert the PROPERTY the field claims, not a phrase: the verdict
    # reads headers, and decoding is confined to the sheets.
    no_pixel = record["verdict_decodes_no_pixel"]
    assert "DIMENSIONS" in no_pixel
    assert "decoded solely for the sheets" in no_pixel
    assert "cannot corrupt a measurement" in no_pixel

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "configs" / "p15_screen_landmarks.yaml").read_text(
        encoding="utf-8"
    )
    payload = yaml.safe_load(text)
    task = payload["task"]
    assert task["kind"] == "screen_mebeauty_landmarks"
    assert task["expect_usable"] == 1519
    assert (task["span_min"], task["span_max"]) == (0.15, 1.10)
    assert (task["interocular_min"], task["interocular_max"]) == (0.25, 0.70)
    # [UPDATED 2026-08-24] bounds_margin was 0.0 when this test was
    # written and is 0.10 since the distribution was measured and the
    # tolerance ruled (phase15.BOUNDS_TOLERANCE_RULED). The values live
    # in test_the_shipped_configs_carry_the_ruled_tolerance; what this
    # test owns is the screen's SHAPE, which is unchanged.
    assert task["bounds_margin"] == 0.10
    assert "119.jpg" in task["diagnose_images"]
    assert "BLIND to detector" in text
    assert "never loosened" in text
    assert "NEVER DECODES A PIXEL" in text
    # Same root, no new declaration.
    survey = yaml.safe_load(
        (repo / "configs" / "p15_survey.yaml").read_text(encoding="utf-8")
    )
    assert payload["inputs"][0]["rollup_sha256"] == (
        survey["inputs"][0]["rollup_sha256"]
    )

    spec = importlib.util.spec_from_file_location(
        "generate_phase15_configs_2aii",
        repo / "scripts" / "generate_phase15_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


def test_the_screen_task_reproduces_refuses_and_diagnoses_by_name():
    import inspect

    from cleft import run as run_module

    assert "screen_mebeauty_landmarks" in run_module.TASKS
    source = inspect.getsource(run_module.task_screen_mebeauty_landmarks)
    # Stop 1's set is reproduced, and a mismatch refuses.
    assert 'task["expect_usable"]' in source
    assert "measuring" in source and "different sets" in source
    # Thresholds arrive from the config; nothing is computed here.
    for key in (
        "span_min", "span_max", "interocular_min", "interocular_max",
        "bounds_margin",
    ):
        assert f'task["{key}"]' in source, key
    # The named diagnosis is driven by the declared list.
    assert 'task.get("diagnose_images")' in source
    assert "DIAGNOSIS" in source
    # The verdict never decodes: sizes come from headers, and the only
    # decode is in the sheet section, after the screen.
    assert "_image_size(" in source
    assert source.index("mebeauty.quality_screen(") < source.index(
        "render.load_image("
    )
    header_reader = inspect.getsource(run_module._image_size)
    assert "handle.size" in header_reader
    assert "no pixel is decoded" in header_reader.lower()
    # The falling count is named as the price, in the run's own log.
    assert "PRICE OF" in source


# --------------------------------------------------------------------------
# 2026-08-24, the screen banked; 119 refuted; two renderer defects fixed
# --------------------------------------------------------------------------


def test_the_screen_is_banked_with_its_monolithic_failure_mode():
    record = phase15.STOP_2A_II_BANKED
    assert "1,464 survive of 1,519" in record["the_screen"]
    assert "96.4%" in record["the_screen"]
    mono = record["monolithic_failure_mode"]
    assert "ALL 55 REJECTIONS ARE ON BOUNDS" in mono
    assert "ZERO on degenerate" in mono
    assert "MONOLITHIC" in mono
    assert "UNVERIFIED" in mono          # the suspect stays a suspect
    assert "0.27x" in record["scale_asymmetry_restated"]
    assert "PRICE OF QUALITY" in record["scale_asymmetry_restated"]
    assert "OUR OWN sheet renderer" in (
        record["diagnoses_two_of_three_unexpected"]
    )


def test_the_119_hypothesis_is_refuted_by_name():
    record = phase15.THE_119_HYPOTHESIS_REFUTED
    assert "OUTSIDE" in record["what_was_registered"]
    measured = record["what_was_measured"]
    assert "0.0129" in measured
    assert "x[1167, 3101]" in measured and "y[2892, 4743]" in measured
    assert "4000 x\n6000" in measured or "4000 x" in measured
    assert "CORRECT AND IN BOUNDS" in measured
    assert "REFUTED" in record["the_hypothesis_is_refuted"]
    assert "by name" in record["the_hypothesis_is_refuted"]
    # The self-critical clause: the wrong guess blamed the data.
    assert "blames the data" in record["the_hypothesis_is_refuted"]
    assert "MANUFACTURED A FALSE\nALARM" in (
        record["where_the_defect_actually_was"]
    ) or "FALSE" in record["where_the_defect_actually_was"]


def test_the_overlay_defect_is_recorded_with_its_measured_mechanism():
    record = phase15.SHEET_OVERLAY_DEFECT
    assert "SOURCE resolution" in record["defect"]
    mechanism = record["the_mechanism_measured"]
    assert "17.9 x 26.8" in mechanism
    assert "FOUR survived" in mechanism
    assert "16.8%" in mechanism
    assert "2125 and 2142" in mechanism
    assert "281 pixels" in mechanism
    assert "SCALE-DEPENDENT" in record["why_it_read_as_a_data_defect"]
    fix = record["the_fix"]
    assert "resize FIRST" in fix
    assert "1,539 green and 630 red" in fix
    assert "Both sheet call sites" in fix
    assert "DIAGNOSIS SHEET" in record["the_evidence"]


def test_the_overlay_fix_is_scale_invariant():
    """Measured: the same overlay at 4000x6000 and at 500x600, where
    the old code left four green pixels at the larger size."""
    import numpy as np

    from cleft.run import _overlay_panel

    rng = np.random.default_rng(0)
    base_points = np.stack([
        rng.uniform(1167, 3101, 68), rng.uniform(2892, 4743, 68),
    ], axis=1)
    counts = []
    for width, height in ((4000, 6000), (500, 600)):
        points = base_points.copy()
        points[:, 0] *= width / 4000
        points[:, 1] *= height / 6000
        panel = _overlay_panel(
            np.full((height, width, 3), 200, np.uint8), points, 224,
            midline=float(np.median(points[:, 0])),
        )
        assert panel.shape == (224, 224, 3)
        green = int(((panel[:, :, 1] == 255) & (panel[:, :, 0] == 0)).sum())
        red = int(((panel[:, :, 0] == 255) & (panel[:, :, 1] == 0)).sum())
        counts.append((green, red))
        assert green > 500, (width, height, green)
        assert red > 100, (width, height, red)
    assert counts[0] == counts[1], "the overlay must not depend on scale"


def test_the_label_defect_is_recorded_and_truncation_is_now_visible():
    record = phase15.SHEET_LABEL_DEFECT
    assert "row=False image=False" in record["defect"]
    mechanism = record["the_mechanism"]
    assert "never truncated at all" in mechanism
    assert "OVERFLOWED" in mechanism and "OVERDREW" in mechanism
    assert "clipped" in mechanism
    # Both diagnoses are INCONCLUSIVE, not negative findings.
    assert "INCONCLUSIVE" in record["the_consequence"]
    assert "girl-3956612" in record["the_consequence"]
    assert "never existed" in record["the_consequence"]
    fix = record["the_fix"]
    assert "fit_label" in fix and "ellipsis" in fix
    assert "SIDECAR" in fix and "SUBSTRING" in fix
    assert "Not claimable until then" in record["re_diagnosis_pending"]
    assert "INSTRUMENT" in record["the_general_lesson"]

    # And the fix behaves: a long label is trimmed AND marked.
    from PIL import Image, ImageDraw

    from cleft.geometry import render

    draw = ImageDraw.Draw(Image.new("RGB", (400, 40)))
    short = "119.jpg"
    assert render.fit_label(draw, short, 224) == short
    long_label = "bui-thanh-tam-D6meliiyaWs-unsplash-and-then-some-more.jpg"
    trimmed = render.fit_label(draw, long_label, 224)
    assert trimmed != long_label
    assert trimmed.endswith("\u2026")
    assert long_label.startswith(trimmed[:-1])


def test_the_diagnosis_path_matches_truncated_names_and_shows_evidence():
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_screen_mebeauty_landmarks)
    # Substring matching, with the matches reported.
    assert "SUBSTRING" in source
    assert '"matches": matches' in source
    assert '"n_matches"' in source
    assert "NO MATCH" in source
    # The evidence sheet exists and uses the fixed overlay.
    assert "mebeauty_diagnosis_sheet.png" in source
    assert "_overlay_panel(" in source
    # Full-label sidecars beside every sheet family: the quality
    # sheets and the diagnosis sheet here, the mapping sheets in 2a.
    assert source.count("_write_label_sidecar(") == 2
    mapping = inspect.getsource(run_module.task_verify_mebeauty_mapping)
    assert mapping.count("_write_label_sidecar(") == 1
    sidecar = inspect.getsource(run_module._write_label_sidecar)
    assert "panel_index" in sidecar
    assert "looked up" in sidecar

    # The overlay helper resizes before drawing -- the fix, in code.
    overlay = inspect.getsource(run_module._overlay_panel)
    assert overlay.index("_resize_nearest(") < overlay.index("for x, y in")
    assert "DISPLAY space" in overlay


# --------------------------------------------------------------------------
# 2026-08-24, the three diagnoses settled; two withdrawals; one limitation
# --------------------------------------------------------------------------


def test_the_three_diagnoses_are_banked_with_their_verdicts():
    record = phase15.DIAGNOSIS_VERDICTS
    assert "fixed diagnosis sheet" in record["banked"]

    # 119: the fix verified by its own evidence, chain closed.
    renders = record["119_jpg_renders"]
    assert "RENDERS ITS OVERLAY" in renders
    assert "VERIFIED\nBY ITS OWN EVIDENCE" in renders or (
        "VERIFIED" in renders
    )
    assert "MANUFACTURED the" in renders and "DISSOLVED it" in renders
    assert "DATA WAS\nCORRECT THROUGHOUT" in renders or (
        "CORRECT THROUGHOUT" in renders
    )

    # girl-3956612: withdrawn, with the mechanism that made it wrong.
    withdrawn = record["girl_3956612_withdrawn"]
    assert "WITHDRAWN" in withdrawn
    assert "PRE-FIX sheet" in withdrawn
    assert "WRONG PANEL" in withdrawn
    assert "not quietly dropped" in withdrawn

    # man-1868320: real, and the screen passes it on every check.
    real = record["man_1868320_real"]
    assert "REAL" in real
    assert "0.0307" in real
    for check in ("bounds", "span", "ordering", "interocular"):
        assert check in real, check

    # The honest scoreboard.
    board = record["the_scoreboard"]
    assert "TWO OF THE THREE ORIGINAL" in board
    assert "OUR OWN INSTRUMENT" in board


def test_both_withdrawals_are_recorded_together_with_cost_and_point():
    # [UPDATED 2026-08-24] Written when there were two; a THIRD
    # withdrawal arrived the same day (the framing over-call). The
    # durable property is that they are recorded TOGETHER with the cost
    # and the point, not that there are exactly two.
    record = phase15.EYE_IMPRESSION_WITHDRAWALS
    assert "together, deliberately" in record["recorded"]
    assert "THIRD" in record["recorded"]
    assert "FRAMING_VARIES_LIMITATION" in record["third"]
    assert "not a confound between them" in record["third"]
    assert "man-2785071" in record["first"]
    assert "misread the pose" in record["first"]
    assert "girl-3956612_1920" in record["second"]
    assert "wrong panel" in record["second"]
    # The cost is stated as a cost, not softened.
    cost = record["the_cost"]
    assert "THREE of a second reading impressions" in cost
    assert "withdrawn by name" in cost
    assert "as findable as the observations" in cost
    assert "recorded errors" in cost
    # The point: withdrawals are what make impressions usable at all.
    point = record["the_point"]
    assert "cited forever" in point
    assert "TWO false" in point and "alarms" in point
    assert "data problem" in point
    # And the discipline did not suppress the surviving observation.
    survived = record["what_survived_it"]
    assert "man-1868320" in survived
    assert "ANATOMICALLY_BLIND_LIMITATION" in survived
    assert "did not suppress" in survived


def test_the_anatomical_blindness_is_a_registered_limitation():
    record = phase15.ANATOMICALLY_BLIND_LIMITATION
    limitation = record["the_limitation"]
    assert "VALID ON EVERY MEASURABLE AXIS" in limitation
    assert "0.0307" in limitation
    assert "GEOMETRICALLY COMPLETE" in limitation
    assert "ANATOMICALLY BLIND" in limitation

    # No new unvalidated instrument, citing the Phase 13 precedent.
    not_building = record["not_building_a_detector"]
    assert "NOT BUILDING AN ANATOMICAL-CORRECTNESS DETECTOR" in not_building
    assert "NEW UNVALIDATED INSTRUMENT" in not_building
    assert "scar-trace precedent" in not_building
    assert "phase13" in not_building
    # And the precedent it cites really says that.
    from cleft import phase13

    scar = phase13.STOP_3_REGISTERED["e_scar_trace_not_testable"]["ruling"]
    assert "NEW UNVALIDATED" in scar

    # The rate is unquantified, and the record says why the three
    # faces were not a sample.
    rate = record["the_rate_is_unquantified"]
    assert "ONE CONFIRMED INSTANCE" in rate
    assert "THREE NAMED FACES" in rate
    assert "not a sample" in rate
    assert "RATE IS UNQUANTIFIED" in rate
    assert "no rate may be quoted" in rate

    assert "STAGED-SHEET EYE GATE" in record["the_catch_net"]
    assert "2b" in record["the_catch_net"]
    # The requantify trigger is written BEFORE the numbers exist.
    trigger = record["the_trigger_to_requantify"]
    assert "REQUANTIFIED BEFORE PRETRAINING" in trigger
    assert "written now" in trigger
    assert "after seeing" in trigger


# --------------------------------------------------------------------------
# 2026-08-24, the second eye pass; the bounds violation measured, not tuned
# --------------------------------------------------------------------------


def test_the_second_eye_pass_is_banked_with_the_disputed_rejects():
    record = phase15.SECOND_EYE_PASS
    assert "FIXED sheets" in record["reviewed"]
    assert "PASS FOR STAGING" in record["survivors_PASS"]
    assert "119.jpg" in record["survivors_PASS"]
    # The two new instances are impressions, and the record refuses to
    # build an instrument for them.
    mild = record["two_mild_instances_EYE_IMPRESSION"]
    assert "jon-ly-ADBOC3UP4eQ" in mild and "girl-3447599" in mild
    assert "EYE-IMPRESSION not measurement" in mild
    assert "UNQUANTIFIED" in mild
    assert "NO\nNEW INSTRUMENT" in mild or "NO NEW INSTRUMENT" in mild.replace(
        "NO\nNEW", "NO NEW"
    )
    # The eye disputes the rejects, with the faces named.
    disputed = record["rejects_disputed"]
    assert "DO NOT DESERVE REJECTION" in disputed
    for name in ("pexels-cottonbro", "models-2158971", "baby-5925923"):
        assert name in disputed, name
    # The mechanism is SUSPECTED, and says so.
    mechanism = record["the_suspected_mechanism"]
    assert "TIGHT" in mechanism and "CROPS" in mechanism
    assert "OUR CHECK" in mechanism
    assert "Suspected, not established" in mechanism
    assert "man-2785071" in record["man_2785071_returns"]


def test_the_bounds_readings_are_registered_before_the_number():
    record = phase15.BOUNDS_VIOLATION_READINGS
    assert "before any violation number exists" in record["registered"]
    assert "never as\na pass/fail" in record["what_is_measured"] or (
        "pass/fail" in record["what_is_measured"]
    )
    # Both branches exist, and the split branch keeps the populations
    # apart rather than averaging them.
    assert "TOO STRICT" in record["reading_if_marginal"]
    assert "BEFORE any re-screen" in record["reading_if_marginal"]
    assert "STAY REJECTED" in record["reading_if_split"]
    assert "REPORTED SEPARATELY" in record["reading_if_split"]
    assert "0.27x is not final" in record["either_way"]

    # The provenance is stated in the record, not hidden in a diff.
    provenance = record["the_provenance_stated_plainly"]
    assert "AFTER AN EYE PASS" in provenance
    assert "rather than hidden in a config diff" in provenance
    assert "argued, not assumed" in provenance

    # And the exception is argued against the phase's own precedent.
    why = record["why_this_differs_from_the_0_08_precedent"]
    assert "0.08 pose threshold was NOT moved" in why
    assert "recorded as the price" in why
    assert "measuring something OTHER" in why
    assert "not the same act" in why

    # No value is proposed, and the record says why.
    value = record["the_value_is_not_proposed_here"]
    assert "NO TOLERANCE VALUE IS PROPOSED" in value
    assert "strict by default" in value
    assert "tuned to the eye's verdict" in value


def test_the_violation_is_measured_as_a_distribution_not_a_verdict():
    """Measured: the two candidate populations are separable by the
    numbers the screen now reports -- which is what makes reading them
    a reading rather than a preference."""
    import numpy as np

    from cleft import mebeauty

    rng = np.random.default_rng(7)
    good = _wellformed_dlib68(rng)

    # A clean face reports no violation at all.
    assert mebeauty.bounds_overshoot(good, _SIZE)["n_points_outside"] == 0

    # Population A: a jaw grazing the border -- few points, tiny margin.
    marginal = good.copy()
    marginal[0, 0] = -3.0
    marginal[16, 0] = float(_SIZE[0]) + 3.0
    a = mebeauty.bounds_overshoot(marginal, _SIZE)
    assert a["n_points_outside"] == 2
    assert a["overshoot_px_max"] == 3.0
    assert a["overshoot_fraction_max"] < 0.01

    # Population B: a foreign coordinate frame -- most points, huge.
    foreign = good * 3.0
    b = mebeauty.bounds_overshoot(foreign, _SIZE)
    assert b["n_points_outside"] > 60
    assert b["overshoot_fraction_max"] > 1.0

    # The declared tolerance separates them -- and BOTH tolerances are
    # strict by default, so the shipped behaviour is unchanged.
    strict = mebeauty.quality_screen(
        marginal, _SIZE, bounds_margin=0.0, bounds_max_points_outside=0,
        **{k: v for k, v in _BANDS.items() if k != "bounds_margin"},
    )
    assert strict["passes"] is False
    tolerant = mebeauty.quality_screen(
        marginal, _SIZE, bounds_margin=0.0, bounds_max_points_outside=2,
        **{k: v for k, v in _BANDS.items() if k != "bounds_margin"},
    )
    assert tolerant["passes"] is True
    still_rejected = mebeauty.quality_screen(
        foreign, _SIZE, bounds_margin=0.0, bounds_max_points_outside=2,
        **{k: v for k, v in _BANDS.items() if k != "bounds_margin"},
    )
    assert still_rejected["passes"] is False


def test_the_screen_reports_the_distribution_and_chooses_nothing():
    import importlib.util
    import inspect

    import yaml

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_screen_mebeauty_landmarks)
    # A distribution: histogram plus quantiles, over the bounds rejects.
    assert "points_outside_histogram" in source
    assert "overshoot_fraction_max" in source
    assert "_quantiles(" in source
    assert "AS A DISTRIBUTION" in source
    # The run chooses nothing, and says so in its own log.
    assert "NO TOLERANCE IS CHOSEN BY THIS RUN" in source
    assert "BOUNDS_VIOLATION_READINGS" in source
    # The tolerance is read from the config, never computed.
    assert 'task.get("bounds_max_points_outside")' in source
    assert "bounds_max_points_outside=max_points_outside" in source

    # [UPDATED 2026-08-24] This asserted BOTH tolerances strict, which
    # was the state until the distribution was measured and the maintainer
    # ruled (phase15.BOUNDS_TOLERANCE_RULED). The durable property is
    # not the VALUE -- it is that the task reads the tolerance from the
    # config and computes none, which the source assertions above pin.
    # The shipped values are asserted in
    # test_the_shipped_configs_carry_the_ruled_tolerance.
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "configs" / "p15_screen_landmarks.yaml").read_text(
        encoding="utf-8"
    )
    task = yaml.safe_load(text)["task"]
    assert isinstance(task["bounds_margin"], float)
    assert isinstance(task["bounds_max_points_outside"], int)
    assert "VISIBLE CONFIG DIFF" in text

    spec = importlib.util.spec_from_file_location(
        "generate_phase15_configs_bounds",
        repo / "scripts" / "generate_phase15_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, the distribution banked, the tolerance ruled, 2b built
# --------------------------------------------------------------------------


def test_the_distribution_fires_branch_one_and_reports_branch_two_empty():
    record = phase15.BOUNDS_DISTRIBUTION_BANKED
    histogram = record["points_outside_histogram"]
    for bucket in ("1:7", "2:9", "3:22", "4:6", "5:3", "6-10:8"):
        assert bucket in histogram, bucket
    # The empty tail is part of the finding.
    assert "11-20:0" in histogram and "21-68:0" in histogram
    assert "MAX 9 of 68" in histogram
    assert "7.8%" in record["overshoot"] and "1.6%" in record["overshoot"]
    assert "TOO STRICT" in record["branch_1_fires"]
    assert "TIGHT-CROP" in record["branch_1_fires"]
    # Branch 2 not firing is stated, not omitted.
    empty = record["branch_2_has_no_instance"]
    assert "NO SECOND POPULATION" in empty
    assert "169%" in empty
    assert "part of reading it" in empty
    # And the earlier suspicion is refuted by name.
    wrong = record["the_suspicion_was_wrong"]
    assert "REFUTES" in wrong
    assert "quietly dropped" in wrong


def test_the_tolerance_is_ruled_with_its_provenance_in_the_open():
    record = phase15.BOUNDS_TOLERANCE_RULED
    values = record["the_values"]
    assert "= 10" in values and "0.10" in values
    assert "JUST ABOVE" in values
    assert "VISIBLE CONFIG" in values

    provenance = record["the_provenance_in_the_open"]
    assert "AFTER AN EYE PASS" in provenance
    assert "not hidden in a diff" in provenance
    assert "argued where it can be found" in provenance

    why = record["why_generous_rather_than_tight"]
    assert "SINGLE CONTINUOUS MODE" in why
    assert "NO MEASURED BASIS" in why
    assert "comfort of looking strict" in why
    assert "6-10 bucket" in why

    assert "0.08 pose threshold" in record["what_stays_untouched"]
    assert "stands as the price" in record["what_stays_untouched"]


def test_the_consequence_is_stated_rather_than_glossed():
    record = phase15.TOLERANCE_CONSEQUENCE
    consequence = record["the_consequence"]
    assert "REJECTS NOTHING ON" in consequence
    assert "1,519" in consequence and "0.28x" in consequence
    assert "PURELY PROTECTIVE" in consequence
    # Protective is defended, not merely asserted.
    still = record["still_protective"]
    assert "foreign-frame case" in still
    assert "collapsed clouds" in still
    assert "not thereby a guard that catches nothing" in still
    # What the exercise produced, including the source finding.
    produced = record["what_the_exercise_produced"]
    assert "SOUND\nWITHIN THE USABLE SET" in produced or (
        "SOUND" in produced
    )
    assert "TWO RENDERER" in produced and "ONE REAL" in produced
    # The tally, and what it predicts.
    tally = record["the_tally"]
    assert "THREE INSTRUMENT DEFECTS AGAINST ONE DATA DEFECT" in tally
    assert "what we built" in tally


def test_the_shipped_configs_carry_the_ruled_tolerance():
    import yaml

    repo = Path(__file__).resolve().parents[1]
    for name in ("p15_screen_landmarks.yaml", "p15_stage_mebeauty.yaml"):
        payload = yaml.safe_load(
            (repo / "configs" / name).read_text(encoding="utf-8")
        )
        task = payload["task"]
        assert task["bounds_margin"] == 0.10, name
        assert task["bounds_max_points_outside"] == 10, name
    screen = (repo / "configs" / "p15_screen_landmarks.yaml").read_text(
        encoding="utf-8"
    )
    # The header carries the measured basis and the provenance.
    assert "median 3" in screen and "max 9 of 68" in screen
    assert "7.8%" in screen
    assert "NO INSTANCE" in screen
    assert "AFTER an eye pass" in screen
    # The header wraps mid-phrase, so match what is actually written.
    assert "rejects NOTHING on MEBeauty" in screen
    assert "PROTECTIVE" in screen


def test_the_box_formula_is_shared_and_the_anchors_differ():
    """One formula, two anchor extractors -- so the two datasets cannot
    be framed differently while both configs say the same thing."""
    import inspect

    import numpy as np

    from cleft import mebeauty
    from cleft.scut import placement

    # SCUT's crop_box now delegates to the shared formula.
    scut_source = inspect.getsource(placement.crop_box)
    assert "box_from_anchors(" in scut_source
    mebeauty_source = inspect.getsource(mebeauty.crop_box)
    assert "box_from_anchors(" in mebeauty_source

    # The formula itself: H from the anatomy, W = ratio x H, centred.
    box = placement.box_from_anchors(10.0, 110.0, 50.0, 0.75)
    assert box == (50.0 - 37.5, 10.0, 75.0, 100.0)
    # And it refuses an inverted span rather than producing a box.
    import pytest

    with pytest.raises(placement.PlacementError, match="not above"):
        placement.box_from_anchors(110.0, 10.0, 50.0, 0.75)
    with pytest.raises(placement.PlacementError, match="positive"):
        placement.box_from_anchors(10.0, 110.0, 50.0, 0.0)

    # MEBeauty's anchors, on a synthetic dlib-68 face: the box is
    # centred on the midline and spans brow to below the lip.
    rng = np.random.default_rng(7)
    points = _wellformed_dlib68(rng)
    x, y, w, h = mebeauty.crop_box(points, 0.75)
    brow_top = points[list(mebeauty.BROW_INDICES), 1].min()
    lip_bottom = points[list(mebeauty.MOUTH_OUTER_INDICES), 1].max()
    assert abs(y - brow_top) < 1e-9
    assert y + h > lip_bottom          # reaches BELOW the lower lip
    assert abs(w / h - 0.75) < 1e-9    # the requested ratio
    centre = mebeauty.midline_x(points)
    assert abs((x + w / 2) - centre) < 1e-9

    # LOWER_MARGIN is inherited, never restated.
    assert "placement.LOWER_MARGIN" in inspect.getsource(
        mebeauty.vertical_span
    )
    assert not hasattr(mebeauty, "LOWER_MARGIN")


def test_build_one_keeps_scuts_calls_byte_identical():
    """The `box` parameter defaults to None, so the frozen composition
    is unchanged for every existing caller."""
    import inspect

    from cleft.scut import masked

    signature = inspect.signature(masked.build_one)
    assert signature.parameters["box"].default is None
    source = inspect.getsource(masked.build_one)
    assert "if box is None:" in source
    assert "placement.crop_box(points, aspect_ratio, lower_margin)" in source


def test_the_2b_task_reproduces_asserts_and_gates():
    import inspect

    from cleft import run as run_module

    assert "stage_mebeauty" in run_module.TASKS
    source = inspect.getsource(run_module.task_stage_mebeauty)
    # Reproduced, not re-decided: both screens declared, count asserted.
    assert 'task["expect_survivors"]' in source
    assert "stages a different set" in source
    assert "mebeauty.quality_screen(" in source
    # The split halves are asserted, and uncovered faces are kept.
    assert 'task[f"expect_{half}"]' in source
    assert '"uncovered"' in source
    # The composition is SCUT's, with MEBeauty's box.
    assert "masked_module.build_one(" in source
    assert "mebeauty.crop_box(" in source
    assert "box=box" in source
    # Parity re-run, immutability, .inprogress, MANIFEST.
    assert "parity_report(" in source
    assert "already exists" in source
    assert ".inprogress" in source
    assert "MANIFEST.json" in source
    # Nothing is embedded; the sheets gate first.
    assert "extract_features" not in source
    assert "BEFORE any embedding" in source

    import yaml

    repo = Path(__file__).resolve().parents[1]
    text = (repo / "configs" / "p15_stage_mebeauty.yaml").read_text(
        encoding="utf-8"
    )
    task = yaml.safe_load(text)["task"]
    assert task["kind"] == "stage_mebeauty"
    assert task["expect_survivors"] == 1519
    assert (task["expect_train"], task["expect_test"]) == (932, 394)
    # [UPDATED 2026-08-24] geometries -> variants, and the third
    # variant (original) was added by amendment 2.
    assert task["variants"] == ["g1", "g2", "original"]
    assert task["ar_sampling"] == "observed"
    assert "COMPOSITION IS SCUT'S, VERBATIM" in text
    assert "levelling OFF" in text
    assert "UNCOVERED" in text
    assert "NOTHING IS EMBEDDED HERE" in text


def test_stop_2b_records_what_it_took_as_agreed():
    record = phase15.STOP_2B_BUILT
    assert "NOT launched" in record["built"]
    assert "VERBATIM" in record["composition_is_scuts"]
    assert "box_from_anchors" in record["composition_is_scuts"]
    assert "byte-identical" in record["composition_is_scuts"]
    assert "levelling OFF" in record["inherited_not_restated"]
    assert "CLEFT_AR" in record["inherited_not_restated"]
    assert "never as a verdict" in record["inherited_not_restated"]
    assert "expect_survivors 1519" in record["reproduced_not_redecided"]
    # The reading of "as agreed" is stated so a misreading is visible.
    agreed = record["stage_all_split_recorded"]
    assert "taken as agreed" in agreed
    assert "if that reading" in agreed
    assert "config change and not a" in agreed
    assert "BEFORE" in record["the_eye_gate"]
    assert "mebeauty_masked_v1" in record["artifact"]
    assert "DERIVED artifact" in record["artifact"]


# --------------------------------------------------------------------------
# 2026-08-24, the parity None diagnosed: an asymmetric reference schema
# --------------------------------------------------------------------------


def _staging_records(n=8):
    """Real build_one records, so the reproduction is the real shape."""
    import numpy as np

    from cleft import mebeauty
    from cleft.scut import masked

    rng = np.random.default_rng(1337)
    points = _wellformed_dlib68(rng)
    image = np.full((600, 500, 3), 180, dtype=np.uint8)
    image[120:520, 130:370] = 140
    records = []
    for ratio in masked.sample_aspect_ratios(n, 1337):
        _, record = masked.build_one(
            image=image, points=points, aspect_ratio=float(ratio),
            geometry="g1", box=mebeauty.crop_box(points, float(ratio)),
        )
        records.append(record)
    return records


def test_the_cleft_reference_carries_different_statistics_per_quantity():
    """The measured cause: this is why exactly one gap per key is
    computable, and why a None here never meant an empty group."""
    from cleft.geometry.staging import CLEFT_STAGED_GEOMETRY

    aspect = CLEFT_STAGED_GEOMETRY["aspect_ratio"]
    pad = CLEFT_STAGED_GEOMETRY["pad_fraction"]
    assert "median" in aspect and "mean" not in aspect
    assert "mean" in pad and "median" not in pad


def test_parity_report_names_why_a_statistic_is_undefined():
    from cleft.scut import masked

    parity = masked.parity_report(_staging_records())
    gaps = parity["gaps"]

    # Exactly one gap per key is defined, by the reference's shape.
    assert gaps["aspect_ratio"]["median_gap"] is not None
    assert gaps["aspect_ratio"]["mean_gap"] is None
    assert gaps["pad_fraction"]["mean_gap"] is not None
    assert gaps["pad_fraction"]["median_gap"] is None

    # And each undefined statistic says WHY, names the reference's own
    # keys, and reports the member count -- so "undefined" can never be
    # read as "the group was empty".
    aspect_reason = gaps["aspect_ratio"]["undefined"]["mean_gap"]
    assert "'aspect_ratio'" in aspect_reason
    assert "no 'mean'" in aspect_reason
    assert "NOT empty" in aspect_reason
    pad_reason = gaps["pad_fraction"]["undefined"]["median_gap"]
    assert "no 'median'" in pad_reason
    assert str(gaps["pad_fraction"]["scut"]["n"]) in pad_reason
    # The defined ones carry no spurious explanation.
    assert "median_gap" not in gaps["aspect_ratio"]["undefined"]
    assert "mean_gap" not in gaps["pad_fraction"]["undefined"]


def test_the_task_reports_defined_gaps_and_survives_an_undefined_one():
    """The crash was a formatting assumption, so the fix is checked at
    the formatting: every defined gap prints, every undefined one is
    named, and no single missing statistic suppresses the line."""
    import inspect

    from cleft import run as run_module
    from cleft.scut import masked

    parity = masked.parity_report(_staging_records())
    printed = []
    for key, value in parity["gaps"].items():
        defined = [
            f"{statistic} gap {value[statistic]:+.4f}"
            for statistic in ("median_gap", "mean_gap")
            if value.get(statistic) is not None
        ]
        printed.append(f"parity {key}: " + (", ".join(defined) or "NONE"))
    assert len(printed) == 2
    assert all("NONE" not in line for line in printed)

    source = inspect.getsource(run_module.task_stage_mebeauty)
    # The old shape -- formatting median_gap unconditionally -- is gone.
    assert "median gap {value['median_gap']:+.4f}" not in source
    assert "is not None" in source
    assert "UNDEFINED" in source
    # Completion is announced before reporting begins, and a reporting
    # failure says the work survived it.
    assert "STAGING COMPLETE" in source
    assert "WORK COMPLETED, REPORT FAILED" in source
    assert source.index("STAGING COMPLETE") < source.index(
        "parity_report("
    )
    # The .inprogress removal is loud and explains itself.
    assert "removing an incomplete" in source
    assert "cannot be" in source and "resumed" in source


def test_the_price_sentence_only_fires_on_an_actual_loss():
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_screen_mebeauty_landmarks)
    assert "lost = len(usable) - len(survivors)" in source
    assert "if lost > 0 else" in source
    assert "NOTHING FELL at this tolerance" in source
    # The registered sentence is inside the conditional branch.
    price_at = source.index("PRICE OF QUALITY")
    conditional_at = source.index("if lost > 0 else")
    assert price_at < conditional_at


def test_the_parity_diagnosis_is_recorded_with_its_refutation():
    record = phase15.PARITY_NONE_DIAGNOSED
    assert "+3 retries" in record["defect"]
    assert "STAGING succeeded every time" in record["defect"]
    cause = record["the_measured_cause"]
    assert "REPRODUCED LOCALLY" in cause
    assert "NO mean" in cause and "NO median" in cause
    assert "+0.0265" in cause and "-0.0458" in cause
    assert "TASK'S LOG LINE" in cause
    assert "BEFORE ctx.log" in record["why_nothing_printed"]

    # The registered hypothesis is refuted, and no finding is invented.
    refuted = record["the_hypothesis_refuted"]
    assert "NOT AN EMPTY GROUP AND NOT AN EMPTY AR BIN" in refuted
    assert "No coverage finding is warranted" in refuted
    assert "none is registered" in refuted
    # But the real question keeps its home.
    coverage = record["where_the_coverage_question_does_live"]
    assert "n_box_outside_frame" in coverage
    assert "INTEGER PIXEL ROUNDING" in coverage
    assert "not clipping" in coverage

    assert "DID ITS JOB" in record["inprogress_worked_as_designed"]
    assert "WITHHELD" in record["inprogress_worked_as_designed"]
    disposition = record["inprogress_disposition"]
    assert "NOT REUSABLE" in disposition
    assert "REMOVES IT AUTOMATICALLY" in disposition
    assert "~20s" in disposition
    for part in ("(1)", "(2)", "(3)"):
        assert part in record["the_three_fixes"], part

    price = phase15.PRICE_SENTENCE_FIRED_ON_NO_LOSS
    assert "NOTHING FELL" in price["defect"]
    assert "READING_COUNT_GUARD" in price["why_it_matters_despite_being_cosmetic"]
    assert "CONDITIONAL" in price["the_fix"]


# --------------------------------------------------------------------------
# 2026-08-24, the artifact exists; the eye passes; stop 3 proposed
# --------------------------------------------------------------------------


def test_stop_2b_is_banked_with_its_parity_near_zero():
    record = phase15.STOP_2B_BANKED
    assert "single clean attempt" in record["banked"]
    run = record["the_run"]
    assert "932 / 394 / " in run and "193" in run
    assert "STAGING COMPLETE printed BEFORE reporting" in run
    assert "195d80ab" in run
    parity = record["parity"]
    assert "-0.0007" in parity and "+0.0031" in parity
    assert "NEAR ZERO" in parity
    # The undefined map is confirmed working, not merely shipped.
    assert "non-empty member count" in parity
    assert "PARITY_NONE_DIAGNOSED" in parity
    assert "first\ntry" in record["the_fixes_held"] or "first" in (
        record["the_fixes_held"]
    )


def test_the_padding_is_registered_as_an_artifact_property():
    record = phase15.PADDING_IS_AN_ARTIFACT_PROPERTY
    measured = record["measured"]
    assert "0 of 1,519" in measured
    assert "BOTH\ngeometries" in measured or "BOTH" in measured
    assert "TIGHT IN-THE-WILD PORTRAITURE" in record["the_cause"]
    assert "AT CLEFT PROPORTIONS" in record["the_cause"]
    # It agrees with the parity rather than contradicting it.
    consistent = record["consistent_with_the_parity"]
    assert "+0.0031" in consistent
    assert "cleft crops are padded too" in consistent
    assert "agree rather than\nconflict" in consistent or "agree" in (
        consistent
    )
    # And it travels.
    travels = record["it_travels"]
    assert "ARTIFACT PROPERTY, NOT AN OBSERVATION" in travels
    assert "belongs in the sentence" in travels


def test_the_staged_sheets_pass_and_the_trigger_did_not_fire():
    record = phase15.STAGED_SHEETS_REVIEWED
    assert record["reviewed"].endswith("PASS")
    verdict = record["verdict"]
    assert "SAME\nINSTRUMENT" in verdict or "SAME" in verdict
    assert "G1 and G2 behaving as expected" in verdict
    assert "mechanism i" in verdict
    # The registered requantify trigger is answered explicitly.
    caught = record["the_catch_net_caught_nothing_new"]
    assert "did NOT\nfire" in caught or "did NOT" in caught
    assert "ANATOMICALLY_BLIND_LIMITATION" in caught
    assert "pretraining is not blocked" in caught


def test_the_framing_limitation_carries_the_governing_reading_not_the_proposal():
    record = phase15.FRAMING_VARIES_LIMITATION
    assert "operative one" in record["registered"]
    assert "THIRD AXIS OF UNATTRIBUTABILITY" in record["what_was_proposed"]

    # The correction, with the mechanism that makes it a shared property.
    correction = record["the_correction_is_the_reading"]
    assert "SCUT VARIES THE SAME WAY" in correction
    assert "CLINICAL PROTOCOL" in correction
    assert "NOT a MEBeauty-vs-SCUT confound" in correction
    assert "SHARED\nPROPERTY" in correction or "SHARED" in correction

    # The withdrawal, by name, with what it would have cost.
    withdrawn = record["the_over_call_withdrawn"]
    assert "WITHDRAWN BY NAME" in withdrawn
    assert "OVER-CALL" in withdrawn
    assert "does not exist there" in withdrawn
    assert "Third eye-impression withdrawal" in withdrawn
    assert "man-2785071" in withdrawn and "girl-3956612" in withdrawn

    assert "STATED ONCE" in record["nothing_can_be_done"]
    # A declined measurement is a decision and is recorded as one.
    declined = record["a_measurement_declined"]
    assert "DECISION-IRRELEVANT" in declined
    assert "no value it could return" in declined
    assert "looks like an oversight" in declined


def test_stop_three_is_proposed_with_geometry_specific_comparators():
    record = phase15.STOP_3_PROPOSED
    assert "agreement or amendment" in record["proposed"]
    arms = record["the_arms"]
    assert "1,326" in arms and "932" in arms and "394" in arms
    assert "BYTE-IDENTICAL" in arms
    assert "NOT pretrained on" in arms
    assert "VARIANT_FOR_INIT" in record["checkpoint_discipline"]
    assert "phantom-findings" in record["checkpoint_discipline"]

    # The comparators are the record's own, and geometry-specific --
    # with the reason that makes pooling a category error.
    comparators = record["the_comparators_are_banked_and_geometry_specific"]
    assert "0.2520" in comparators and "0.1952" in comparators
    assert "0.0830" in comparators and "0.2001" in comparators
    assert "WITHIN geometry" in comparators
    assert "NEVER\npooled" in comparators or "NEVER" in comparators
    # And those figures are the ladder's own.
    from cleft import ladder

    assert ladder.STAGE_D1_AT_G1["cells"]["vit_b16"] == (0.2520, 0.1952, 0.0830)
    g2 = ladder.STAGE_G_LABEL_FORMULATION["triples"]["scut_masked__g2"]
    assert g2["arms"]["mean"]["pcc"] == 0.2001

    statistic = record["the_verdict_statistic_proposed"]
    assert "4.12.1" in statistic
    assert "DECLARED IN THE CONFIG before any number" in statistic
    assert "COHORT_CANNOT_RESOLVE" in statistic

    readings = record["readings_to_register_before_any_number"]
    assert "0.28x" in readings
    assert "NOT\nattributable" in readings or "attributable" in readings
    assert "still a measured answer" in readings

    bound = record["the_asymmetry_sentence_is_bound"]
    assert "EVERY reading carries the mechanism-ii sentence" in bound
    assert "PADDING_IS_AN_ARTIFACT_PROPERTY" in bound
    assert "FRAMING_VARIES_LIMITATION" in bound
    assert "no config, no task, no arm" in record["nothing_built"]


# --------------------------------------------------------------------------
# 2026-08-24, stop 3 built: four amendments, three arms, one seam each
# --------------------------------------------------------------------------


def test_the_within_geometry_rule_is_asserted_in_code_not_only_prose():
    """Amendment 1: a crossed verdict is REFUSED, because the masked
    cells differ across geometry by more than any dataset effect."""
    import pytest

    from cleft import mebeauty

    # The comparators are the ladder's own.
    assert mebeauty.banked_cell("scut_masked", "g1") == 0.0830
    assert mebeauty.banked_cell("scut_masked", "g2") == 0.2001
    assert mebeauty.banked_cell("imagenet", "g1") == 0.2520
    assert mebeauty.banked_cell("scut_original", "g1") == 0.1952
    spread = abs(
        mebeauty.banked_cell("scut_masked", "g2")
        - mebeauty.banked_cell("scut_masked", "g1")
    )
    assert round(spread, 4) == 0.1171

    # Same geometry computes.
    verdict = mebeauty.verdict_delta(
        {"geometry": "g1", "pcc": 0.15}, "scut_masked", "g1"
    )
    assert verdict["delta_vs_comparator"] == round(0.15 - 0.0830, 6)
    # Amendment 3: the ImageNet anchor rides in every verdict.
    assert verdict["imagenet_pcc"] == 0.2520
    assert verdict["delta_vs_imagenet"] == round(0.15 - 0.2520, 6)
    assert "NOT a win" in verdict["the_imagenet_anchor"]
    # And that example IS the predicted shape: beats SCUT, loses to
    # ImageNet.
    assert verdict["delta_vs_comparator"] > 0
    assert verdict["delta_vs_imagenet"] < 0

    # Crossed geometry is refused, with the reason.
    with pytest.raises(mebeauty.MappingError, match="REFUSED"):
        mebeauty.verdict_delta(
            {"geometry": "g1", "pcc": 0.15}, "scut_masked", "g2"
        )
    with pytest.raises(mebeauty.MappingError, match="wearing a dataset"):
        mebeauty.verdict_delta(
            {"geometry": "g2", "pcc": 0.15}, "scut_masked", "g1"
        )
    # An unknown cell is refused rather than guessed.
    with pytest.raises(mebeauty.MappingError, match="no banked cell"):
        mebeauty.banked_cell("scut_original", "g2")


def test_the_caveats_are_data_and_differ_for_the_original_arm():
    """Amendment 4: four bound to a masked number, three to the
    original -- returned as DATA so a reading cannot quote a number
    without them."""
    masked = phase15.caveats_for("masked_g1")
    original = phase15.caveats_for("masked_original")

    shared = {
        "mechanism_ii_asymmetry", "framing_variance",
        "anatomical_blindness",
    }
    assert shared <= set(masked) and shared <= set(original)
    # [UPDATED 2026-08-24] Both padding clauses were CORRECTED: the
    # masked one rested on a null read (0/1,519 was never measured) and
    # the original one on a synthetic that measured a mechanism, not a
    # rate. What this test owns is that the two arms carry DIFFERENT
    # padding clauses; the values live in
    # test_the_caveats_now_quote_measured_distributions.
    masked_padding = [k for k in masked if k.startswith("padding")]
    original_padding = [k for k in original if k.startswith("padding")]
    assert len(masked_padding) == len(original_padding) == 1
    assert masked_padding != original_padding
    assert "0.28x" in masked["mechanism_ii_asymmetry"]
    assert "UNATTRIBUTABLE" in masked["mechanism_ii_asymmetry"]


def test_the_original_arm_padding_refinement_is_measured():
    """The amendment said the original arm has no padding property; the
    frozen stage says otherwise for non-square sources, and the record
    carries the measurement rather than either assumption."""
    import numpy as np

    from cleft.geometry.staging import stage

    # SCUT's shape: square, so stage is a pure resize and pads nothing.
    square = stage(np.full((350, 350, 3), 200, np.uint8), size=224)
    assert square.pad_fraction == 0.0
    # MEBeauty's shape: non-square, so stage pads.
    portrait = stage(np.full((600, 500, 3), 200, np.uint8), size=224)
    assert portrait.pad_fraction > 0.15

    record = phase15.ORIGINAL_ARM_PADS_TOO
    assert "NOT\nQUITE" in record["what_is_measured"] or "NOT" in (
        record["what_is_measured"]
    )
    assert "0.1652" in record["what_is_measured"]
    assert "350x350 SQUARE" in record["what_is_measured"]
    assert "SWAPS one padding story" in record["the_correction"]
    assert "still SMALLER" in record["the_correction"]
    assert "measured number" in record["how_it_is_handled"]
    assert "the amendment's reading stands" in record["how_it_is_handled"]


def test_stop_three_records_all_four_amendments():
    record = phase15.STOP_3_AMENDED
    one = record["amendment_1_within_geometry"]
    assert "ASSERTED IN CODE" in one
    assert "0.117" in one and "wearing a" in one
    two = record["amendment_2_original_arm"]
    assert "0.1952" in two and "BEST transfer cell" in two
    assert "no landmarks, no " in two
    three = record["amendment_3_imagenet_anchor"]
    assert "0.2520" in three
    assert "WORSE THAN NO BEAUTY PRETRAINING" in three
    assert "must read as such" in three
    four = record["amendment_4_caveats_bound"]
    assert "FOUR caveats" in four and "THREE to the" in four
    build = record["the_build"]
    assert "masked_original" in build
    assert "932 fit / 394 held" in build
    assert "byte-identical" in build
    assert "4.12.1" in record["the_verdict_statistic"]


def test_the_pretraining_seams_preserve_every_scut_call():
    """Three seams this phase, all defaulting to the existing
    behaviour: box, labels_by_stem, and the artifact format."""
    import inspect

    from cleft.train import pretrain

    signature = inspect.signature(pretrain.run_pretraining)
    assert signature.parameters["labels_by_stem"].default is None
    source = inspect.getsource(pretrain.run_pretraining)
    assert "if labels_by_stem is None:" in source
    assert "read_split_or_refuse(" in source
    # The supplied split is asserted against the declared counts.
    assert "whoever supplies them" in source
    # The new source is documented where it is declared.
    assert "masked_original" in pretrain.SOURCES
    assert "original" in pretrain.SOURCES  # unchanged
    module = inspect.getsource(pretrain)
    assert "MEBeauty's images are nested" in module


def test_the_mebeauty_pretraining_task_reuses_the_recipe():
    import inspect

    from cleft import run as run_module

    assert "pretrain_mebeauty" in run_module.TASKS
    source = inspect.getsource(run_module.task_pretrain_mebeauty)
    # The frozen loop, not a copy of it.
    assert "pretrain.run_pretraining(" in source
    assert "pretrain.PretrainConfig(" in source
    assert "for epoch in" not in source     # no training loop here
    assert "labels_by_stem=(train_labels, test_labels)" in source
    # The shipped split, restricted -- never a home-made cut.
    assert "never a home-made cut" in source
    # The source-side PCC is provenance, not the verdict.
    assert "PROVENANCE, not" in source
    assert "caveats_for(" in source


def test_the_stop_three_configs_derive_the_recipe_and_two_pass():
    import importlib.util

    import yaml

    repo = Path(__file__).resolve().parents[1]
    p6 = yaml.safe_load(
        (repo / "configs" / "p6_pretrain_vit_b16_original.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]

    for variant, source_name in (
        ("g1", "masked_g1"), ("g2", "masked_g2"),
        ("original", "masked_original"),
    ):
        name = f"p15_pretrain_mebeauty_{variant}.yaml"
        text = (repo / "configs" / name).read_text(encoding="utf-8")
        task = yaml.safe_load(text)["task"]
        assert task["kind"] == "pretrain_mebeauty"
        assert task["source"] == source_name
        assert (task["expect_train"], task["expect_test"]) == (932, 394)
        # EVERY recipe knob equals Phase 6's -- the fact, not the claim.
        for key in (
            "epochs", "inner_val_frac", "monitor", "deterministic",
            "learning_rate", "weight_decay", "batch_size",
            "checkpoint_every", "backbone", "region_scheme",
        ):
            assert task[key] == p6[key], (name, key)
        assert "ONLY THE" in text and "DATA DIFFERS" in text
        assert "ImageNet anchor" in text
        assert "PROVENANCE" in text
        # The staged artifact is two-pass.
        entry = next(
            e for e in yaml.safe_load(text)["inputs"]
            if e["name"] == "mebeauty_staged"
        )
        if set(entry["rollup_sha256"]) == {"0"}:
            assert "PLACEHOLDER" in text
        assert entry["path"].endswith("mebeauty_masked_v2")

    # The staging config is v2 with three variants.
    stage_text = (repo / "configs" / "p15_stage_mebeauty.yaml").read_text(
        encoding="utf-8"
    )
    stage_task = yaml.safe_load(stage_text)["task"]
    assert stage_task["out_version"] == "mebeauty_masked_v2"
    assert stage_task["variants"] == ["g1", "g2", "original"]
    assert "faces.json" in stage_text
    assert "third cell SCUT occupies" in stage_text

    spec = importlib.util.spec_from_file_location(
        "generate_phase15_configs_stop3",
        repo / "scripts" / "generate_phase15_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


def test_the_staging_task_writes_the_scut_artifact_format():
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_stage_mebeauty)
    # The frozen loader's names, so it reads the artifact unchanged.
    assert 'f"masked_{variant}.npy"' in source
    assert '"faces.json"' in source
    # The original variant goes through the FROZEN stage, not build_one.
    assert "frozen_stage(image, size=size)" in source
    assert "original_pads" in source
    # And its pad fraction is measured and reported, not assumed away.
    assert "pad fraction mean" in source
    assert "carry ANY pad" in source


# --------------------------------------------------------------------------
# 2026-08-24, the key mismatch: diagnosed, fixed, and covered
# --------------------------------------------------------------------------


def test_the_key_mismatch_is_recorded_with_its_coverage_answer():
    record = phase15.KEY_MISMATCH_DIAGNOSED
    assert "+4 retries" in record["defect"]
    assert "BEFORE any staging" in record["defect"]
    # The cause is owned, not passivised.
    assert "COPY-PASTE" in record["the_cause_is_mine"]
    assert "LEFT THE OLD ONE IN" in record["the_cause_is_mine"]
    # Why the schema could not have caught it.
    assert "DECLARES, not what the" in record["why_the_schema_could_not_catch_it"]
    # The coverage question is answered directly: no, and here is the fix.
    coverage = record["the_coverage_answer"]
    assert "NO TEST EXERCISED" in coverage
    assert "subscripts" in coverage
    assert ".get" in coverage
    # The new test was verified to fail on the real defect.
    verified = record["the_test_was_verified_to_fail"]
    assert "cannot fail verifies nothing" in verified
    assert "AST-based" in verified
    assert "word-matching trap" in verified
    pattern = record["the_second_run_time_only_defect_this_week"]
    assert "Two run-time-only defects in a week" in pattern
    assert "compiles and passes every existing test" in pattern
    assert "cluster is not a test environment" in pattern
    assert "before any side effect" in record["the_fix"]


def test_the_two_corrections_are_recorded_as_sent():
    record = phase15.V1_SURVIVES_CORRECTION
    assert "correcting an earlier message" in record["corrected"]
    assert "SURVIVES" in record["a_v1_survives"]
    assert "v2.inprogress" in record["a_v1_survives"]
    assert "WITHDRAWN" in record["a_v1_survives"]
    assert "HYGIENE, not" in record["b_ordering_is_hygiene"]
    assert "NO ARTIFACT WAS LOST" in record["b_ordering_is_hygiene"]
    assert "retraction discipline" in record["why_it_is_recorded"]


def test_the_staging_task_reads_its_config_before_it_destroys_anything():
    """The fix, in code: every subscript happens above the rmtree."""
    import inspect

    from cleft import run as run_module
    from tests.test_smoke_run import _task_keys_read  # noqa: F401

    source = inspect.getsource(run_module.task_stage_mebeauty)
    removal = source.index("shutil.rmtree")
    for key in (
        "variants", "out_version", "sheet_faces", "expect_survivors",
        "bounds_margin", "bounds_max_points_outside",
    ):
        assert f'task["{key}"]' in source, key
        assert source.index(f'task["{key}"]') < removal, key
    # The renamed key is gone from the CODE (it survives only in the
    # comment that explains the rename -- checked by parsing, not by
    # searching for the word).
    import ast

    tree = ast.parse(source.strip())
    subscripted = {
        node.slice.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "task"
        and isinstance(node.slice, ast.Constant)
    }
    assert "geometries" not in subscripted
    assert "variants" in subscripted


# --------------------------------------------------------------------------
# 2026-08-24, both padding records corrected: a null read and a synthetic
# --------------------------------------------------------------------------


def test_the_original_arm_padding_is_settled_by_the_distribution():
    record = phase15.ORIGINAL_ARM_PADS_TOO
    # The original text is preserved...
    assert "NOT\nQUITE" in record["what_is_measured"] or "0.1652" in (
        record["what_is_measured"]
    )
    # ...and the correction sits beside it, with the distribution.
    measured = record["measured_and_withdrawn"]
    assert "mean 0.0019" in measured
    assert "median 0.0000" in measured
    assert "10 of 1,519" in measured
    assert "NEAR-SQUARE" in measured
    assert "swap-story is WITHDRAWN" in measured
    assert "amendment 2 claimed" in measured
    assert "NEGLIGIBLE" in measured
    # And who was right is stated, not blurred.
    assert "the maintainer's, not the record's" in record["whose_reading_won"]


def test_mechanism_is_not_rate_is_recorded_with_its_family():
    record = phase15.MECHANISM_IS_NOT_RATE
    assert "0.1652" in record["what_happened"]
    assert "10 of 1,519" in record["what_happened"]
    assert "MECHANISM IS NOT RATE" in record["the_lesson"].upper()
    assert "can this happen" in record["the_lesson"]
    assert "how often" in record["the_lesson"]
    # The family: three synthetics this phase, none of them a rate.
    family = record["the_family"]
    assert "collapsed-cloud" in family
    assert "foreign-frame" in family
    assert "NO instance" in family
    assert "none of them measured a rate" in family
    assert "PROVISIONAL" in record["the_discipline"]


def test_the_padding_property_is_corrected_to_a_measured_quantity():
    record = phase15.PADDING_IS_AN_ARTIFACT_PROPERTY
    # The original claim is preserved verbatim...
    assert "0 of 1,519" in record["measured"]
    # ...and the correction names the null read precisely.
    null_read = record["corrected_the_number_was_a_null_read"]
    assert "NOT A MEASUREMENT" in null_read
    assert "inside_frame" in null_read and "box_inside_frame" in null_read
    assert "None" in null_read
    assert "1,516 of 1,519" in null_read
    assert "one of them was not reading it" in null_read

    # The two-questions point, which reconciles the eye with the count.
    two = record["corrected_two_different_questions"]
    assert "DIFFERENT QUESTIONS" in two
    assert "CLEFT ASPECT RATIO" in two
    assert "WHETHER OR NOT" in two.replace("\n", " ")

    # Restated in a quantity the artifact actually carries.
    restated = record["corrected_the_property_restated"]
    assert "PAD FRACTION" in restated
    assert "+0.0031" in restated and "0.2534" in restated
    assert "0.2565" in restated
    assert "WORKING, not a defect" in restated
    # And it is not a differentiator, with the honest limit on that claim.
    not_diff = record["corrected_it_is_not_a_differentiator"]
    assert "NOT a MEBeauty-vs-SCUT caveat" in not_diff
    assert "SAME build_one path" in not_diff
    assert "not re-measured" in not_diff
    assert "framing-variance correction had" in not_diff

    # The cleft reference really carries that mean.
    from cleft.geometry.staging import CLEFT_STAGED_GEOMETRY

    assert CLEFT_STAGED_GEOMETRY["pad_fraction"]["mean"] == 0.2534


def test_the_caveats_now_quote_measured_distributions():
    masked = phase15.caveats_for("masked_g1")
    original = phase15.caveats_for("masked_original")
    # The old wrong clauses are gone by name.
    assert "padding_property" not in masked
    assert "padding_measured_not_assumed" not in original
    # The new ones quote measured numbers.
    assert "0.2565" in masked["padding_matched_by_design"]
    assert "NOT a differentiator" in masked["padding_matched_by_design"]
    assert "0.0019" in original["padding_negligible_measured"]
    assert "10 of 1,519" in original["padding_negligible_measured"]


def test_the_staging_task_reads_build_ones_real_record_keys():
    """The regression for the null read: every key the staging task
    pulls out of a build_one record must be a key build_one writes."""
    import ast
    import inspect

    from cleft import run as run_module
    from cleft.scut import masked

    build_source = inspect.getsource(masked.build_one)
    written = {
        node.value
        for node in ast.walk(ast.parse(build_source.strip()))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    task_source = inspect.getsource(run_module.task_stage_mebeauty)
    read = {
        node.args[0].value
        for node in ast.walk(ast.parse(task_source.strip()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "record"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert read, "the task reads no record keys; this checks nothing"
    unknown = sorted(key for key in read if key not in written)
    assert not unknown, (
        f"the staging task reads record key(s) build_one never writes: "
        f"{unknown} -- this is the 0/1,519 null read "
        "(phase15.PADDING_IS_AN_ARTIFACT_PROPERTY, corrected)"
    )
    # And the specific key that was wrong is now right.
    assert "box_inside_frame" in read
    assert "inside_frame" not in read


# --------------------------------------------------------------------------
# 2026-08-24, the padding caveat accepted; the synthetic sweep; the pattern
# --------------------------------------------------------------------------


def test_the_padding_caveat_is_accepted_with_its_amendment_inside_it():
    record = phase15.PADDING_IS_AN_ARTIFACT_PROPERTY
    accepted = record["accepted_as_binding"]
    assert "BINDING" in accepted
    assert "0.2565" in accepted and "0.2534" in accepted
    assert "0.0019" in accepted and "10 of 1,519" in accepted
    assert "MEASURED" in accepted

    # The amendment lives INSIDE the caveat, not beside it.
    travels = record["travels_as_shared_construction"]
    assert "SHARED\nCONSTRUCTION" in travels or "SHARED" in travels
    assert "not as a MEBeauty-specific" in travels
    assert "must not be quoted as one" in travels

    # The unmeasured half is named, with the condition attached.
    unmeasured = record["the_unmeasured_half_named"]
    assert "NOT RE-MEASURED" in unmeasured
    assert "CONSTRUCTION IDENTITY" in unmeasured
    assert "MUST BE MEASURED FIRST" in unmeasured
    assert "rather\nTHAN INFERRED" in unmeasured or "INFERRED" in unmeasured


def test_the_pipeline_not_sources_pattern_is_recorded_on_its_third_instance():
    record = phase15.ARTIFACT_DIFFERENCES_ARE_PIPELINE_DIFFERENCES
    assert "third instance" in record["recorded"]
    three = record["the_three"]
    assert "FRAMING VARIANCE" in three
    assert "PADDING" in three
    assert "ASPECT\nRATIO" in three or "ASPECT" in three
    assert "CLEFT_AR parity is INHERITED" in three

    pattern = record["the_pattern"]
    assert "DIFFERENCES IN OUR PIPELINE" in pattern
    assert "shared BY DESIGN" in pattern
    assert "disguised as MEBeauty's peculiarities" in pattern

    check = record["the_check_it_implies"]
    assert "same code" in check
    # And what a real source difference looks like.
    survives = record["what_survives_as_a_real_difference"]
    assert "DEMOGRAPHICS" in survives and "SIZE" in survives
    assert "0.28x" in survives


def test_mechanism_is_not_rate_is_flagged_for_the_writeup():
    record = phase15.MECHANISM_IS_NOT_RATE
    flagged = record["for_the_writeup_methods_chapter"]
    assert "METHODS CHAPTER" in flagged
    assert "not only the" in flagged
    assert "0 real instances" in flagged and "10 of\n1,519" in flagged or (
        "10 of" in flagged
    )
    assert "nobody\ncan calibrate" in flagged or "calibrate" in flagged
    assert "SYNTHETIC_CAVEAT_SWEEP" in record["applied_retroactively"]


def test_the_synthetic_sweep_marks_every_guard():
    record = phase15.SYNTHETIC_CAVEAT_SWEEP
    assert "backwards" in record["swept"]

    # Every quality check is marked, and four of the five never fired.
    mechanism_only = [
        key for key, value in record.items()
        if isinstance(value, str) and value.startswith("MECHANISM-ONLY")
    ]
    assert set(mechanism_only) == {
        "quality_screen_degenerate", "quality_screen_span",
        "quality_screen_ordering", "quality_screen_interocular",
        "mapping_verification",
    }
    for key in mechanism_only:
        assert "0" in record[key], key      # the real-instance count

    # Bounds is the split case, and the record says so.
    bounds = record["quality_screen_bounds"]
    assert "BOTH" in bounds
    assert "55 of 1,519" in bounds
    assert "0 real\ninstances" in bounds or "0 real" in bounds
    assert "mechanism-only" in bounds and "rate-measured" in bounds

    # The two rate-measured paddings.
    assert "RATE-MEASURED" in record["original_arm_padding"]
    assert "0.1652" in record["original_arm_padding"]
    assert "10 of 1,519" in record["original_arm_padding"]
    assert "RATE-MEASURED" in record["masked_arm_padding"]
    assert "null read, not a synthetic" in record["masked_arm_padding"]

    # The non-synthetic caveats are marked separately, and both admit
    # they carry no rate.
    other = record["not_synthetic_motivated_at_all"]
    assert "ANATOMICAL BLINDNESS" in other
    assert "FRAMING" in other
    assert "UNQUANTIFIED" in other
    assert "DECLINED as decision-irrelevant" in other
    assert "neither carries" in other
    assert "COUNT RATIO" in record["mechanism_ii_is_neither"]

    # The arithmetic, and the correction to it.
    shows = record["what_the_sweep_shows"]
    assert "EIGHT GUARDS" in shows
    assert "SEVEN NEVER FIRED" in shows
    assert "rejects NOTHING at the ruled tolerance" in shows
    assert "EVIDENCE ABOUT THE DATASET" in shows
    correction = record["the_count_is_stated_because_it_was_first_stated_wrong"]
    assert "SIX in its first draft" in correction
    assert "5 quality + 3 mapping = 8" in correction

    # And the count matches the guards the code actually has.
    from cleft import mebeauty

    assert len(mebeauty.QUALITY_CHECKS) == 5
    report = mebeauty.verify_mapping(
        {f"f{i}": _synthetic_dlib68(__import__("numpy").random.default_rng(1))
         for i in range(4)},
        midline_tolerance=0.02, eye_symmetry_tolerance=0.02,
        corner_margin=0.05,
    )
    mapping_checks = [k for k in report if k.endswith("_passes")]
    assert len(mapping_checks) == 3


# --------------------------------------------------------------------------
# 2026-08-24, the three arms banked; the namespace ruled and built
# --------------------------------------------------------------------------


def test_the_pretraining_arms_are_banked_with_the_mapping_unconfirmed():
    record = phase15.STOP_3_PRETRAINING_BANKED
    figures = record["figures_as_reported"]
    for value in ("0.6905", "0.6685", "0.7186", "epoch 26", "epoch 6", "epoch 7"):
        assert value in figures, value

    # this machine cannot confirm the mapping and says so, with the fix.
    mapping = record["the_mapping_is_not_confirmed_here"]
    assert "CANNOT BE CONFIRMED HERE" in mapping
    assert "no cluster access" in mapping
    assert "AS REPORTED, not" in mapping
    assert "EXTRACTION TASK ASSERTS IT" in mapping

    # Observation 1, with the checkable/uncheckable comparators split.
    harder = record["observation_1_harder_on_its_own_task"]
    assert "0.7893" in harder and "0.8306" in harder
    assert "-0.0988" in harder and "-0.1621" in harder
    assert "OPERATOR-SUPPLIED and NOT in the repo" in harder
    assert "PROVENANCE rather than verdict" in harder
    # The two banked comparators really are the ladder's.
    from cleft import ladder

    assert ladder.MASKED_G1_ARTEFACT["scut_test_pcc"] == 0.7893
    assert ladder.MASKED_G1_ARTEFACT["against"]["masked_g2"] == 0.8306

    # Observation 2 is registered BEFORE transfer numbers exist.
    inversion = record["observation_2_the_geometry_ordering_inverts"]
    assert "CANNOT BE RETROFITTED" in inversion
    assert "INVERTS" in inversion
    assert "no reading is attached" in inversion.lower()
    assert "checkable rather than persuasive" in inversion

    # Observation 3 reads the dynamics as the asymmetry made visible.
    memorised = record["observation_3_the_arms_memorised_their_fit_set"]
    assert "0.002-0.03" in memorised
    assert "MEMORISED" in memorised
    assert "0.28x asymmetry is visible" in memorised
    assert "working exactly as designed" in memorised
    assert "run lengths" in memorised
    assert "cleft-side probe decides" in record["the_verdict_is_elsewhere"]


def test_mebeauty_has_its_own_namespace_and_the_ladder_is_untouched():
    from cleft import embeddings, mebeauty

    # MEBeauty's own inits and variant table.
    assert mebeauty.MEBEAUTY_INITS == (
        "mebeauty_masked", "mebeauty_original",
    )
    assert mebeauty.expected_mebeauty_variant("mebeauty_masked", "g1") == (
        "masked_g1"
    )
    assert mebeauty.expected_mebeauty_variant("mebeauty_masked", "g2") == (
        "masked_g2"
    )
    assert mebeauty.expected_mebeauty_variant(
        "mebeauty_original", "g1"
    ) == "masked_original"

    # Its own arithmetic, DERIVED from its own tables.
    counts = mebeauty.mebeauty_set_count()
    assert counts["embedding_sets"] == 3
    assert counts["inits"] == 2
    assert counts["variants"] == ["masked_g1", "masked_g2", "masked_original"]
    assert "never_pooled_with" in counts

    # **THE LADDER IS UNTOUCHED** -- the ruling's whole point.
    assert embeddings.INITS == ("imagenet", "scut_original", "scut_masked")
    assert set(mebeauty.MEBEAUTY_INITS) & set(embeddings.INITS) == set()
    ladder_counts = embeddings.expected_set_count(n_backbones=4)
    assert ladder_counts["embedding_sets"] == 24
    assert ladder_counts["pretraining_runs"] == 12
    assert ladder_counts["inits"] == 3

    # A ladder init is not accepted by MEBeauty's table, and vice versa.
    import pytest

    with pytest.raises(mebeauty.MappingError, match="unknown MEBeauty init"):
        mebeauty.expected_mebeauty_variant("scut_masked", "g1")


def test_the_arm_to_run_mapping_is_asserted_not_trusted():
    import pytest

    from cleft import mebeauty

    assert mebeauty.assert_run_is_the_declared_arm(
        {"source": "masked_g1"}, "masked_g1"
    ) == "masked_g1"
    # A swap refuses, and a missing field refuses too.
    with pytest.raises(mebeauty.MappingError, match="ASSERTED, never"):
        mebeauty.assert_run_is_the_declared_arm(
            {"source": "masked_g2"}, "masked_g1"
        )
    with pytest.raises(mebeauty.MappingError):
        mebeauty.assert_run_is_the_declared_arm({}, "masked_g1")
    # The pretraining task writes the field this reads.
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_pretrain_mebeauty)
    assert '"source": source' in source


def test_stop_four_records_the_ruling_and_puts_the_open_item():
    record = phase15.STOP_4_PROPOSED
    ruling = record["the_ruling"]
    assert "OWN registered set namespace" in ruling
    assert "FROZEN AND UNTOUCHED" in ruling
    assert "NEVER POOLED" in ruling
    assert "PHANTOM-FINDINGS SHAPE" in ruling
    assert "MASKED_G1_ARTEFACT" in ruling
    # What it forbids, concretely enough to check against.
    forbids = record["what_the_ruling_forbids_concretely"]
    assert "expected_set_count" in forbids
    assert "embedding_plan" in forbids
    assert "which lattice each came from" in forbids

    assert "GEOMETRY-BOUND" in record["the_namespace"]
    assert "3 sets" in record["the_namespace"]
    asserted = record["the_mapping_gets_asserted_not_trusted"]
    assert "REFUSES" in asserted
    assert "no cluster access" in asserted
    assert "rather than propagating into a verdict" in asserted
    assert "BYTE-IDENTICAL" in record["the_probes"]
    assert "DECLARED IN" in record["the_verdict"]
    assert "3 extraction jobs" in record["compute_shape"]

    # The open item is PUT, with a recommendation and a reason.
    item = record["one_item_for_a_ruling"]
    assert "PROBE'S CONFIG LAYER" in item
    assert "train_cv" in item
    assert "probe_mebeauty" in item
    assert "recommendation" in item
    assert "put rather than assumed" in item


# --------------------------------------------------------------------------
# 2026-08-24, stop 4 built: extraction, probes, and the verified mapping
# --------------------------------------------------------------------------


def test_the_mapping_is_verified_and_the_withdrawn_number_is_named():
    record = phase15.MAPPING_VERIFIED_AND_A_WITHDRAWAL
    assert "OWN source field" in record["verified"]
    mapping = record["the_mapping"]
    for value in ("masked_g1", "masked_g2", "masked_original",
                  "0.6905", "0.6685", "0.7186"):
        assert value in mapping, value
    assert "VERIFIED\nBASELINE" in mapping or "VERIFIED" in mapping

    # The name trap, closed.
    naming = record["the_original_arms_source_string"]
    assert "masked_original" in naming
    assert "UNCROPPED" in naming
    assert "no trapezium" in naming

    # The withdrawal, with what it costs and what it does not.
    withdrawal = record["the_0_8914_withdrawal"]
    assert "0.8914 IS WITHDRAWN" in withdrawal
    assert "ABSENT from ladder.py" in withdrawal
    assert "UNAVAILABLE" in withdrawal
    assert "DIFFERENT KEYS" in withdrawal
    assert "no cluster access" in withdrawal
    cost = record["what_the_withdrawal_costs"]
    assert "0.1952" in cost and "0.0280" in cost
    assert "needs nothing from the withdrawn figure" in cost

    # And 0.8914 really is absent from the ladder.
    repo = Path(__file__).resolve().parents[1]
    ladder_text = (repo / "src" / "cleft" / "ladder.py").read_text(
        encoding="utf-8"
    )
    assert "0.8914" not in ladder_text


def test_the_probe_layer_ruling_is_recorded_with_its_reason():
    record = phase15.PROBE_LAYER_RULED
    reason = record["the_reason_given"]
    assert "SHARED VOCABULARY" in reason
    assert "manufacture a factor effect" in reason
    assert "stays CLOSED" in reason
    assert "difference-set test" in record["what_was_built"]
    assert "features_override" in record["how_the_features_reach_it"]
    assert "no new seam" in record["how_the_features_reach_it"]

    # train_cv's vocabulary really is CLOSED -- the ladder's three, and
    # no MEBeauty init has leaked into it.
    from cleft.config.schema import TASK_SPECS

    # [UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] Phase 25's ruling
    # added two FACTORY-PRETRAINED inits -- DINOv2 on LVD-142M and DINO on
    # unlabelled ImageNet-1k. They are named apart from "imagenet" because
    # neither is ImageNet-supervised, and the set directory carries the
    # init in its name (phase25.THE_WEIGHTS_ARE_FACTORY_PRETRAINED_LIKE_
    # THE_PROBES). The three SCUT/ImageNet values are unchanged.
    assert TASK_SPECS["train_cv"]["init"].choices == (
        "imagenet", "scut_original", "scut_masked",
        "dinov2_lvd142m", "dino_in1k",
    )
    # The new kinds exist, and they are separate keys.
    assert "probe_mebeauty" in TASK_SPECS
    assert "extract_mebeauty_embeddings" in TASK_SPECS


def test_the_extraction_asserts_the_mapping_and_the_variant():
    import inspect

    from cleft import run as run_module

    assert "extract_mebeauty_embeddings" in run_module.TASKS
    source = inspect.getsource(
        run_module.task_extract_mebeauty_embeddings
    )
    # The mapping assertion comes BEFORE the checkpoint is loaded.
    assert "assert_run_is_the_declared_arm(" in source
    assert source.index("assert_run_is_the_declared_arm(") < source.index(
        "extract_features("
    )
    # The geometry-bound variant is checked against the declared source.
    assert "expected_mebeauty_variant(" in source
    assert "geometry-bound checkpoint rule" in source
    # MEBeauty's own writer, and the ladder's untouched.
    assert "mebeauty.save_set(" in source
    assert "embeddings.save(" not in source
    # The source-side figure is carried as provenance, not verdict.
    assert "PROVENANCE, not the" in source


def test_the_probe_uses_the_shared_entry_point_and_binds_its_caveats():
    import inspect

    from cleft import run as run_module

    assert "probe_mebeauty" in run_module.TASKS
    source = inspect.getsource(run_module.task_probe_mebeauty)
    # The SAME phase3 entry point, with the existing override seam.
    assert "phase3.run(" in source
    assert "features_override=(values" in source
    assert "phase3.write_outputs(" in source
    # The verdict machinery: within-geometry, threshold, anchor, caveats.
    assert "mebeauty.verdict_delta(" in source
    assert "phase3.combined_claimable_delta(" in source
    assert 'task["comparator_sd"]' in source
    assert "the_imagenet_anchor" in source
    assert "phase15.caveats_for(" in source
    assert "caveats_bound" in source
    # No ledger row.
    assert "No ledger row" in source


def test_the_six_stop_four_configs_declare_before_any_number():
    import yaml

    repo = Path(__file__).resolve().parents[1]
    p7 = yaml.safe_load(
        (repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    expected = {
        "g1": (
            "mebeauty_masked", "g1", "masked_g1", "scut_masked",
            0.0830, 0.0247,
        ),
        "g2": (
            "mebeauty_masked", "g2", "masked_g2", "scut_masked",
            0.2001, 0.0142,
        ),
        "original": (
            "mebeauty_original", "g1", "masked_original",
            "scut_original", 0.1952, 0.0280,
        ),
    }
    for arm, (
        init, geometry, src, comp, comp_pcc, comp_sd,
    ) in expected.items():
        extract = yaml.safe_load(
            (repo / "configs" / f"p15_extract_mebeauty_{arm}.yaml").read_text(
                encoding="utf-8"
            )
        )["task"]
        assert extract["kind"] == "extract_mebeauty_embeddings"
        assert extract["init"] == init
        assert extract["geometry"] == geometry
        assert extract["expect_source"] == src
        assert extract["expect_patients"] == 237

        probe_text = (
            repo / "configs" / f"p15_probe_mebeauty_{arm}.yaml"
        ).read_text(encoding="utf-8")
        probe = yaml.safe_load(probe_text)["task"]
        assert probe["kind"] == "probe_mebeauty"
        assert probe["comparator_init"] == comp
        assert probe["comparator_sd"] == comp_sd
        assert probe["comparator_n"] == 5
        # THE DIFFERENCE-SET TEST: every recipe knob equals A's.
        for key in (
            "label", "backbone", "trainable", "max_epochs", "patience",
            "inner_val_frac", "monitor", "seeds", "learning_rate",
            "weight_decay", "batch_size",
        ):
            assert probe[key] == p7[key], (arm, key)
        assert "ITS OWN KIND, NOT train_cv" in probe_text
        assert "BEFORE ANY NUMBER" in probe_text
        assert "ImageNet anchor" in probe_text
        assert "FOUR CAVEATS ARE BOUND" in probe_text
        # The header quotes the comparator MEAN too, so the prose and
        # the machine-read sd cannot drift apart silently.
        assert f"= {comp_pcc}" in probe_text

    # **THE COMPARATORS ARE THE LADDER'S OWN CELLS**, not numbers
    # retyped into the generator: mean AND sd, read from the two banked
    # tables. A drift in either direction fails here.
    from cleft import ladder

    g1_cells = dict(zip(
        ladder.STAGE_D1_AT_G1["inits"],
        ladder.STAGE_D1_AT_G1["cells"]["vit_b16"],
    ))
    g2_cells = dict(zip(
        ladder.STAGE_D_AT_G2["inits"],
        ladder.STAGE_D_AT_G2["cells"]["vit_b16"],
    ))
    g1_sd = ladder.STAGE_D1_AT_G1["sd"]["vit_b16"]
    g2_sd = ladder.STAGE_D_AT_G2["sd"]["vit_b16"]

    assert (g1_cells["scut_masked"], g1_sd["scut_masked"]) == (0.0830, 0.0247)
    assert (g2_cells["scut_masked"], g2_sd["scut_masked"]) == (0.2001, 0.0142)
    assert (g1_cells["scut_original"], g1_sd["scut_original"]) == (
        0.1952, 0.0280,
    )
    # And the anchor every reading carries.
    assert (g1_cells["imagenet"], g1_sd["imagenet"]) == (0.2520, 0.0148)


def test_mebeauty_sets_are_written_and_read_by_their_own_namespace():
    import tempfile

    import numpy as np
    import pytest

    from cleft import mebeauty

    ids = [1, 2, 3]
    directory = Path(tempfile.mkdtemp()) / "set"
    mebeauty.save_set(
        directory, np.zeros((3, 768)), init="mebeauty_masked",
        geometry="g1", variant="masked_g1", checkpoint_sha256="ab" * 32,
        patient_ids=ids, manifest_ids=ids, run_name="r",
    )
    values, metadata = mebeauty.load_set(directory, ids)
    assert values.shape == (3, 768)
    assert metadata["namespace"] == "mebeauty"
    assert metadata["init"] == "mebeauty_masked"
    assert "never_pooled_with" in metadata

    # A crossed checkpoint is refused at WRITE...
    with pytest.raises(mebeauty.MappingError, match="must come from"):
        mebeauty.save_set(
            Path(tempfile.mkdtemp()) / "x", np.zeros((3, 768)),
            init="mebeauty_masked", geometry="g2", variant="masked_g1",
            checkpoint_sha256="a", patient_ids=ids, manifest_ids=ids,
            run_name="r",
        )
    # ...and a ladder init is refused by MEBeauty's writer.
    with pytest.raises(mebeauty.MappingError, match="unknown MEBeauty init"):
        mebeauty.save_set(
            Path(tempfile.mkdtemp()) / "y", np.zeros((3, 768)),
            init="scut_masked", geometry="g1", variant="masked_g1",
            checkpoint_sha256="a", patient_ids=ids, manifest_ids=ids,
            run_name="r",
        )
    # Row order is checked with the LADDER's own checker.
    from cleft.embeddings import EmbeddingError

    with pytest.raises(EmbeddingError):
        mebeauty.load_set(directory, [3, 2, 1])


def test_stop_four_is_recorded_as_built():
    record = phase15.STOP_4_BUILT
    assert "NOT launched" in record["built"]
    assert "three arms" in record["the_extraction"]
    assert "CLUSTER-ONLY" in record["the_extraction"]
    assert "eleven knobs" in record["the_probes"]
    assert "0.0297/0.0664" in record["the_probes"]
    asserted = record["what_is_asserted_in_code"]
    for item in ("arm-to-run mapping", "geometry-bound variant",
                 "row order", "within-geometry"):
        assert item in asserted, item
    assert "cannot be read out of the" in record["what_rides_in_every_reading"]
    assert "24/12" in record["the_lattices_stay_apart"]


def test_the_trainable_guard_firing_is_recorded_with_both_halves():
    record = phase15.TRAINABLE_GUARD_FIRED
    assert "choices=None" in record["what_fired"]
    assert "caught before launch" in record["what_fired"]
    assert "('head', 'full')" in record["the_fix"]
    # The distinction that keeps the two checks from being confused.
    both = record["why_the_difference_set_test_did_not_catch_it"]
    assert "AGREEMENT, not admissibility" in both
    assert "not" in both and "substitutes for each other" in both
    # And the narrowing, with the order that keeps the rule intact.
    assert "MEASURED, not argued" in record["then_the_guard_itself_was_narrowed"]
    assert "came SECOND" in record["the_rule_this_did_not_break"]

    # The field really is constrained now.
    from cleft.config.schema import TASK_SPECS

    assert TASK_SPECS["probe_mebeauty"]["trainable"].choices == (
        "head", "full",
    )


# --------------------------------------------------------------------------
# 2026-08-24, the pretraining run directories declared
# --------------------------------------------------------------------------


def _phase15_generator(tag: str):
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        f"generate_phase15_configs_{tag}",
        repo / "scripts" / "generate_phase15_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pretrain_run_directories_are_the_pasted_ones():
    """The three declared run directories, and the contract they obey."""
    import yaml

    repo = Path(__file__).resolve().parents[1]
    expected = {
        arm: (
            f"p15_pretrain_mebeauty_{arm}__dad0e892"
            f"__p15-pretrain-mebeauty-{arm}"
        )
        for arm in ("g1", "g2", "original")
    }
    sha8s = set()
    for arm, directory in expected.items():
        payload = yaml.safe_load(
            (repo / "configs" / f"p15_extract_mebeauty_{arm}.yaml").read_text(
                encoding="utf-8"
            )
        )
        entry = next(
            e for e in payload["inputs"] if e["name"] == "pretrain_run"
        )
        assert entry["path"].endswith("/runs/keeper/p15/" + directory)
        assert "PENDING" not in entry["path"]
        # **DECLARED 2026-08-24** from declare_inputs.py on the cluster
        # (535 files each, ~327 MB). Never invented on the laptop --
        # this assertion changed from "still a placeholder" to "a real
        # digest" only when the maintainer pasted the measured values.
        rollup = entry["rollup_sha256"]
        assert len(rollup) == 64
        assert set(rollup) <= set("0123456789abcdef")
        assert set(rollup) != {"0"}

        # The run-directory contract: <stem>__<sha8>__<job-id>.
        stem, sha8, job = directory.split("__")
        assert stem == f"p15_pretrain_mebeauty_{arm}"
        assert len(sha8) == 8 and all(c in "0123456789abcdef" for c in sha8)
        assert job == f"p15-pretrain-mebeauty-{arm}"
        sha8s.add(sha8)

    # **All three agree on the sha8, and that is correct rather than
    # suspicious**: the field is the GIT COMMIT's short sha, not the
    # config's, so runs launched from one commit share it. Three
    # DIFFERENT configs sharing a per-config hash would be the defect.
    assert sha8s == {"dad0e892"}
    context = (repo / "src" / "cleft" / "provenance" / "context.py").read_text(
        encoding="utf-8"
    )
    assert 'f"{self.config_path.stem}__{self.git.sha8}__{suffix}"' in context


def test_a_corrected_run_directory_resets_its_hash(tmp_path, monkeypatch):
    """The carry is PATH-PINNED, measured rather than asserted in prose.

    The earlier carry returned the whole shipped entry whenever its path
    was not a PENDING one, which made the CONFIG authoritative after the
    first fill: correcting a run directory in the generator would have
    been ignored, and the stale path would have kept a hash that still
    verified. This is that hole, checked shut.
    """
    import yaml

    module = _phase15_generator("carry")
    monkeypatch.setattr(module, "REPO", tmp_path)
    (tmp_path / "configs").mkdir()

    real = "a" * 64
    declared = "/cluster/runs/keeper/p15/run_a"
    (tmp_path / "configs" / "c.yaml").write_text(
        yaml.safe_dump({"inputs": [
            {"name": "pretrain_run", "path": declared,
             "rollup_sha256": real},
        ]}),
        encoding="utf-8",
    )

    # Same path -> the verified hash is carried forward.
    assert module._carried_named("c.yaml", "pretrain_run", declared) == {
        "name": "pretrain_run", "path": declared, "rollup_sha256": real,
    }
    # CORRECTED path -> the hash resets, so guard 3 refuses until the new
    # directory is declared. The old hash does not ride along.
    corrected = module._carried_named(
        "c.yaml", "pretrain_run", "/cluster/runs/keeper/p15/run_b"
    )
    assert corrected["path"] == "/cluster/runs/keeper/p15/run_b"
    assert set(corrected["rollup_sha256"]) == {"0"}
    # A missing config is the placeholder, not a crash.
    assert set(
        module._carried_named("absent.yaml", "pretrain_run", declared)[
            "rollup_sha256"
        ]
    ) == {"0"}


def test_stop_four_is_fully_declared():
    """Stop 4's declaration state, counted rather than remembered.

    **Third update, 2026-08-24.** This list held six placeholders, then
    three, and now none; each time it FAILED first and was changed
    deliberately. That is the point of pinning a state instead of
    trusting a memory of it -- and it is why a placeholder cannot
    quietly outlive the run that was supposed to fill it.
    """
    import yaml

    repo = Path(__file__).resolve().parents[1]
    runs, sets = {}, {}
    for arm in ("g1", "g2", "original"):
        for stage in ("extract", "probe"):
            path = repo / "configs" / f"p15_{stage}_mebeauty_{arm}.yaml"
            text = path.read_text(encoding="utf-8")
            payload = yaml.safe_load(text)
            for entry in payload["inputs"]:
                assert "PENDING" not in entry["path"], (stage, arm)
                rollup = entry["rollup_sha256"]
                assert len(rollup) == 64, (stage, arm, entry["name"])
                assert set(rollup) != {"0"}, (stage, arm, entry["name"])
                if entry["name"] == "pretrain_run":
                    runs[arm] = rollup
                elif entry["name"] == "mebeauty_embeddings":
                    sets[arm] = rollup
            # Nothing outstanding, and the header says so.
            assert "**RESOLVED.**" in text, (stage, arm)
            assert "PLACEHOLDER" not in text, (stage, arm)

    # **BOTH TRIPLES ARE DISTINCT.** Three arms declared against one
    # artifact would verify perfectly and answer the wrong question --
    # the failure a hash check cannot catch alone, because each hash is
    # correct for the artifact it names.
    assert set(runs) == set(sets) == {"g1", "g2", "original"}
    assert len(set(runs.values())) == 3
    assert len(set(sets.values())) == 3

    assert runs == {
        "g1": "5c606603950fabda04e3af8048f899f0f8fcca0bf2d8510c733293074ff59f22",
        "g2": "8332f096546d11df0cb3484c45da8fafa4b0a37eed75b9a8673f3110333b1d54",
        "original":
            "1e423efa061ec7440691d90525cd3a8f2766ed1cf9064d58ce90723b095f1823",
    }
    assert sets == {
        "g1": "ac3b607b039b90877811e48317c7d1ea5664cd3c73670286cd16543f0cefdc06",
        "g2": "7e20eaf8e15339b21fd17f40cc80131dbdb4207a2c3da1c76ad9df1fb10a5f87",
        "original":
            "f0a71930075adbd599eefe56894f803eb7981920dea560723f1c0e71a3308a20",
    }

    # **NONE OF THESE IS A LOG PREFIX.** The extraction logs printed
    # 71082846 / 86e30fb8 / fa7bf516, which are PAYLOAD digests over a
    # different scope (phase15.TWO_DIGESTS_ONE_WORD). A fill taken from
    # the log would look right and be wrong; this asserts it did not
    # happen.
    for short in ("71082846", "86e30fb8", "fa7bf516"):
        assert not any(v.startswith(short) for v in sets.values()), short

    assert _phase15_generator("pending").main(["--check"]) == 0


def test_the_extractions_are_banked_with_the_verdict_withheld():
    record = phase15.STOP_4A_BANKED
    assert "green on all three" in record["the_assertion_held"]
    assert "(237, 768)" in record["the_sets"]
    for digest in ("ac3b607b", "7e20eaf8", "f0a71930"):
        assert digest in record["the_sets"], digest
    assert "38 bytes larger" in record["the_sets"]
    assert "The verdict is not here" in record["provenance_not_verdict"]
    for pcc in ("0.6905", "0.6685", "0.7186"):
        assert pcc in record["provenance_not_verdict"], pcc


def test_the_two_digests_are_measured_and_the_scope_is_named():
    record = phase15.TWO_DIGESTS_ONE_WORD
    covers = record["what_each_covers"]
    assert "values.npy" in covers and "metadata.json" in covers
    assert "AS IT STANDS" in covers and "declare_inputs.py" in covers
    why = record["why_they_cannot_coincide"]
    assert "MEASURED rather than argued" in why
    assert "9dcb54dc" in why and "2bc9c01e" in why
    assert "looks\nexactly like a right one" in record["the_harm_it_invites"] \
        or "exactly like a right one" in record["the_harm_it_invites"]
    scope = record["the_scope_measured_not_assumed"]
    assert "exactly two writers" in scope
    assert "embeddings.save" in scope and "NO manifest" in scope
    assert "the CALLEE" in record["an_earlier_static_survey_was_wrong"]
    assert "WRITER_DIED_AFTER_SAVE" in record["and_the_fix_had_its_own_defect"]


def test_the_two_digests_really_differ_and_the_manifest_holds_the_payload():
    """The record's central claim, re-measured here rather than quoted.

    A set is built, both digests are taken, and MANIFEST.json is opened.
    If a future writer ever hashed AFTER the manifest -- making the two
    coincide -- this fails and the record above needs rewriting.
    """
    import json
    import tempfile

    import numpy as np

    from cleft import mebeauty
    from cleft.provenance.hashing import hash_dir

    ids = list(range(1, 8))
    directory = Path(tempfile.mkdtemp()) / "mebeauty_g1_cleft_v1"
    payload = mebeauty.save_set(
        directory, np.zeros((7, 768), dtype="float32"),
        init="mebeauty_masked", geometry="g1", variant="masked_g1",
        checkpoint_sha256="ab" * 32, patient_ids=ids, manifest_ids=ids,
        run_name="r",
    )
    declare = hash_dir(directory)

    assert sorted(p.name for p in directory.iterdir()) == [
        "MANIFEST.json", "metadata.json", "values.npy",
    ]
    # Payload covers TWO files; the declarable rollup covers three.
    assert set(payload["files"]) == {"values.npy", "metadata.json"}
    assert set(declare["files"]) == {
        "values.npy", "metadata.json", "MANIFEST.json",
    }
    # They differ, and neither is a prefix of the other -- which is what
    # makes a fill-from-the-log a silent error rather than an obvious one.
    assert payload["rollup"] != declare["rollup"]
    assert not declare["rollup"].startswith(payload["rollup"][:8])
    # And the manifest carries the PAYLOAD digest, which is why.
    manifest = json.loads(
        (directory / "MANIFEST.json").read_text(encoding="utf-8")
    )
    assert manifest["payload_rollup"] == payload["rollup"]


def test_both_trapped_log_lines_name_which_digest_to_declare():
    import inspect

    from cleft import run as run_module

    for task in (
        run_module.task_extract_mebeauty_embeddings,
        run_module.task_extract_scut_embeddings,
    ):
        source = inspect.getsource(task)
        assert "declare = hash_dir(out_dir)" in source, task.__name__
        assert "rollup_sha256_for_configs" in source, task.__name__
        assert "(DECLARE THIS)" in source, task.__name__
        assert "payload_rollup" in source, task.__name__
        assert "in MANIFEST.json" in source, task.__name__
        # The bare word that caused this must not stand alone any more.
        assert 'f"{payload[\'rollup\'][:8]} -- CLUSTER-ONLY' not in source


def test_the_untrapped_writers_were_measured_not_assumed():
    """The ladder's writer emits no manifest, so its callers are safe.

    This is the discriminator the record names. If ``embeddings.save``
    ever grew a MANIFEST.json, every task built on it would start
    logging an undeclarable digest -- and this fails first.
    """
    import tempfile

    import numpy as np

    from cleft import embeddings

    directory = Path(tempfile.mkdtemp()) / "vit_b16__imagenet__g1"
    written = embeddings.save(
        directory, np.zeros((4, 768), dtype="float32"),
        backbone="vit_b16", init="imagenet", geometry="g1",
        backbone_kind="transformer", variant=None, checkpoint_sha256=None,
        patient_ids=[1, 2, 3, 4], manifest_ids=[1, 2, 3, 4],
    )
    assert sorted(p.name for p in directory.iterdir()) == [
        "metadata.json", "values.npy",
    ]
    assert "MANIFEST.json" not in [p.name for p in directory.iterdir()]
    # It returns METADATA, not a hash payload -- so a caller cannot
    # accidentally log a payload digest from it.
    assert "rollup" not in written


def test_every_run_py_function_imports_the_names_it_uses(
    name="hash_dir",
):
    """The defect the two-digest fix introduced, guarded repo-wide.

    ``run.py`` imports its helpers PER FUNCTION. A new call site that
    forgets the import raises NameError only when that task runs -- and
    for a writer, only AFTER the artifact is on disk
    (phase13.WRITER_DIED_AFTER_SAVE). Static, so it costs nothing.
    """
    import ast

    repo = Path(__file__).resolve().parents[1]
    source = (repo / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    module_level = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                module_level.add(alias.asname or alias.name)

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        uses = any(
            isinstance(sub, ast.Name) and sub.id == name
            for sub in ast.walk(node)
        )
        if not uses or name in module_level:
            continue
        imported = any(
            isinstance(sub, ast.ImportFrom)
            and any(alias.name == name for alias in sub.names)
            for sub in ast.walk(node)
        )
        if not imported:
            offenders.append(node.name)

    assert offenders == [], (
        f"{offenders} call {name}() without importing it; run.py imports "
        "per function, so this is a NameError at run time -- and for a "
        "writer, one that fires after the artifact is already written"
    )


# --------------------------------------------------------------------------
# 2026-08-24, the probes: the phase's verdict
# --------------------------------------------------------------------------


def test_every_banked_cell_reproduces_its_own_threshold():
    """The phase's arithmetic, recomputed from the plan's own formula.

    Each cell declares a delta against a banked SCUT comparator and the
    threshold it cleared. Both are rederived here from the two arms'
    five-seed SDs via ``phase3.combined_claimable_delta`` (PLAN 4.12.1),
    so a retyped figure fails rather than propagating into a verdict.
    """
    from cleft import ladder
    from cleft.train import phase3

    g1_cells = dict(zip(
        ladder.STAGE_D1_AT_G1["inits"],
        ladder.STAGE_D1_AT_G1["cells"]["vit_b16"],
    ))
    g2_cells = dict(zip(
        ladder.STAGE_D_AT_G2["inits"],
        ladder.STAGE_D_AT_G2["cells"]["vit_b16"],
    ))
    g1_sd = ladder.STAGE_D1_AT_G1["sd"]["vit_b16"]
    g2_sd = ladder.STAGE_D_AT_G2["sd"]["vit_b16"]

    comparators = {
        "mebeauty_masked@g1": (g1_cells["scut_masked"], g1_sd["scut_masked"]),
        "mebeauty_masked@g2": (g2_cells["scut_masked"], g2_sd["scut_masked"]),
        "mebeauty_original": (
            g1_cells["scut_original"], g1_sd["scut_original"],
        ),
    }
    anchor, anchor_sd = g1_cells["imagenet"], g1_sd["imagenet"]
    assert (anchor, anchor_sd) == (0.2520, 0.0148)

    for name, cell in phase15.STOP_4B_BANKED["cells"].items():
        comp, comp_sd = comparators[name]
        delta = round(cell["pcc"] - comp, 4)
        assert delta == cell["vs_scut"], (name, delta)

        threshold = phase3.combined_claimable_delta(
            cell["sd"], cell["n"], comp_sd, cell["n"]
        )
        assert round(threshold["arm_means_95"], 4) == cell["threshold"], name
        assert (abs(delta) > cell["threshold"]) == cell["clears"], name
        assert cell["clears"] is True, name
        assert (abs(delta) > threshold["single_run_95"]) == cell[
            "survives_single_run_95"
        ], name

        # The ImageNet anchor rides in every cell.
        assert round(cell["pcc"] - anchor, 4) == cell["vs_imagenet"], name
        against_anchor = phase3.combined_claimable_delta(
            cell["sd"], cell["n"], anchor_sd, cell["n"]
        )
        assert round(
            against_anchor["arm_means_95"], 4
        ) == cell["imagenet_threshold"], name

        # And each cell's own 95% interval, from its own spread.
        half = 1.96 * cell["sd"] / (cell["n"] ** 0.5)
        assert [
            round(cell["pcc"] - half, 4), round(cell["pcc"] + half, 4)
        ] == cell["own_95"], name

    assert "recomputed on the laptop" in (
        phase15.STOP_4B_BANKED["the_thresholds_reproduce"]
    )


def test_no_cell_beats_imagenet_and_exactly_one_reaches_parity():
    """The registered 'not a win' guard, checked against the numbers."""
    cells = phase15.STOP_4B_BANKED["cells"]

    beats = [
        name for name, c in cells.items()
        if c["vs_imagenet"] > c["imagenet_threshold"]
    ]
    assert beats == [], "the 'not a win' sentence would be wrong"

    parity = [
        name for name, c in cells.items()
        if abs(c["vs_imagenet"]) <= c["imagenet_threshold"]
    ]
    assert parity == ["mebeauty_original"]
    assert cells["mebeauty_original"]["vs_imagenet"] == 0.0017


def test_reading_one_records_the_correction_that_halves_its_claim():
    """g1 is claimably below zero; g2 is not, and the record says so."""
    cells = phase15.STOP_4B_BANKED["cells"]
    g1 = cells["mebeauty_masked@g1"]["own_95"]
    g2 = cells["mebeauty_masked@g2"]["own_95"]
    assert g1[1] < 0, "g1 must exclude zero"
    assert g2[0] < 0 < g2[1], "g2 must span zero"

    record = phase15.READING_1_MASKED_ARMS_AT_ZERO
    assert "at or below zero" in record["the_sentence"].lower() or (
        "no usable grade signal" in record["the_sentence"]
    )
    correction = record["one_correction_to_that_sentence"]
    assert "Only g1 is CLAIMABLY below zero" in correction
    assert "SPANS zero" in correction
    assert "[-0.0914, -0.0472]" in correction
    assert "[-0.0665,\n+0.0067]" in correction or "+0.0067" in correction
    assert "ANTI-CORRELATED" in record[
        "why_a_loss_at_negative_pcc_is_a_stronger_statement"
    ]
    # The caveat that binds, and what resolves it.
    assert "UNATTRIBUTABLE" in record["the_caveat_that_binds_here"]
    assert "READING_3" in record["the_caveat_that_binds_here"]


def test_reading_two_separates_parity_from_victory():
    record = phase15.READING_2_ORIGINAL_REACHES_PARITY
    assert "PARITY, not victory" in record["the_sentence"]
    assert "FIRST arm" in record["the_sentence"]
    assert "+0.0017" in record["the_sentence"]
    assert "0.0234" in record["the_sentence"]
    # The asymmetry runs the other way for a win.
    strong = record["the_asymmetry_runs_in_its_favour"]
    assert "WIN\nis STRONG" in strong or "is STRONG" in strong
    assert "0.28x" in strong
    # The guard fires and is bounded.
    fires = record["the_not_a_win_sentence_fires_correctly"]
    assert "no cell in this phase beats ImageNet" in fires
    assert "not a result" in fires
    # And the honest ceiling.
    ceiling = record["what_this_cell_is"]
    assert "does\nnot beat ImageNet" in ceiling or "not beat ImageNet" in ceiling
    assert "single_run_95" in ceiling


def test_reading_three_escapes_the_size_confound_and_says_where_it_stops():
    record = phase15.READING_3_MASKING_NOT_BEAUTY
    assert "+0.3230" in record["the_sentence"]
    assert "+0.2836" in record["the_sentence"]

    escape = record["why_this_contrast_escapes_the_mechanism_ii_caveat"]
    assert "WITHIN MEBeauty" in escape
    assert "held constant" in escape
    assert "only the crop differs" in escape

    # It does NOT claim more than a within-source contrast can carry.
    residual = record["what_the_contrast_is_still_confounded_by"]
    assert "THE CROP AS A WHOLE" in residual
    assert "crop-geometry sweep" in residual

    boundary = record["the_boundary_of_the_claim"]
    assert "SOURCE-DEPENDENT" in boundary
    assert "-0.0568" in boundary


def test_the_reframing_changes_no_banked_number():
    """A reframing, checked against the ladder it reframes."""
    from cleft import ladder

    record = phase15.PHASE_6_REFRAMED
    assert "REFRAMING, not a re-measurement" in record["registered_as"]
    assert "no number moves" in record["registered_as"]

    # The banked cell it reframes is untouched.
    g1_cells = dict(zip(
        ladder.STAGE_D1_AT_G1["inits"],
        ladder.STAGE_D1_AT_G1["cells"]["vit_b16"],
    ))
    assert g1_cells["scut_masked"] == 0.0830
    assert g1_cells["scut_original"] == 0.1952

    assert "not as a correction of the record" in record[
        "why_this_is_not_a_new_claim_about_the_banked_cell"
    ]
    test = record["what_would_test_it"]
    assert "DISAGREE at the original geometry" in test
    assert "0.1952" in test and "0.2537" in test
    assert "crop-geometry sweep" in test
    assert "not weakened" in record["the_caveats_bound_to_it"]


def test_the_exit_walk_covers_all_seven_and_flags_what_it_cannot_settle():
    walk = phase15.PHASE_15_EXIT_WALK
    numbered = [k for k in walk if k[0].isdigit()]
    assert len(numbered) == 7
    assert sorted(k[0] for k in numbered) == list("1234567")
    assert phase15.PHASE_15_EXIT_CRITERIA["expected_count"] == 7

    for key in numbered:
        assert "MET" in walk[key], key

    # Criterion 1's ambiguity is FLAGGED, not decided by the record.
    first = walk["1_best_is_cleft_side_transfer"]
    assert "ambiguity flagged rather than resolved by" in first
    assert "trainable: head" in first
    assert "has not\nbeen run" in first or "has not" in first
    assert "a maintainer decision" in first

    # And the source-side figures stayed out of the verdict.
    assert "provenance" in first


def test_the_phase_16_verdict_is_per_cell_and_names_its_own_limits():
    verdict = phase15.PHASE_16_VERDICT_PROPOSED
    assert "the ruling is" in verdict["proposed"]
    assert "0.7893 lesson" in verdict["the_definition_it_answers"]
    assert "no single 'best dataset'" in verdict["stated_per_cell_not_pooled"]

    cells = verdict["the_cells"]
    assert "MEBeauty is better" in cells
    assert "SCUT is better" in cells
    for figure in ("0.2537", "0.1952", "+0.0585", "0.0830", "-0.0693",
                   "0.2001", "-0.0299"):
        assert figure in cells, figure

    attached = verdict["the_masking_finding_attached"]
    assert "trapezium is where it" in attached
    assert "single_run_95" in attached

    forward = verdict["what_phase_16_should_carry_forward"]
    assert "mebeauty_original" in forward
    assert "does not BEAT ImageNet" in forward

    limits = verdict["the_caveats_that_travel_with_it"]
    assert "every cell here is ViT-B/16" in limits
    assert "backbone-dependent" in limits
    assert "crop-geometry sweep" in verdict["what_would_overturn_it"]


def test_no_ledger_row_was_written_for_this_phase():
    """Criterion 7: descriptive ranking, no ledger row without the bar."""
    repo = Path(__file__).resolve().parents[1]
    for name in ("p15_probe_mebeauty_g1", "p15_probe_mebeauty_g2",
                 "p15_probe_mebeauty_original"):
        text = (repo / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
        assert "DESCRIPTIVE ranking" in text
        assert "no ledger row" in text
        assert "COHORT_CANNOT_RESOLVE" in text
    assert "no ledger row" in phase15.PHASE_15_EXIT_WALK[
        "7_readings_pre_committed_no_ledger_row"
    ]


# --------------------------------------------------------------------------
# 2026-08-24, Phase 15 closed
# --------------------------------------------------------------------------


def test_the_fourth_withdrawal_is_recorded_and_kept_distinct():
    record = phase15.EYE_IMPRESSION_WITHDRAWALS
    fourth = record["fourth_and_of_a_different_kind"]
    assert "A numeric over-call" in fourth
    assert "TRUE of masked@g1" in fourth
    assert "NOT TRUE of masked@g2" in fourth
    assert "[-0.0914," in fourth and "+0.0067]" in fourth
    assert "at or below zero" in fourth

    kinds = record["the_tally_by_kind"]
    assert "THREE withdrawn eye impressions" in kinds
    assert "ONE withdrawn" in kinds
    assert "fail differently" in kinds
    # The eye-impression tally sentence stays arithmetically true.
    assert "THREE of a second reading impressions" in record["the_cost"]

    # And the intervals it rests on are the banked ones.
    cells = phase15.STOP_4B_BANKED["cells"]
    assert cells["mebeauty_masked@g1"]["own_95"] == [-0.0914, -0.0472]
    assert cells["mebeauty_masked@g2"]["own_95"] == [-0.0665, 0.0067]


def test_criterion_one_is_amended_with_its_provenance_labelled():
    record = phase15.CRITERION_1_AMENDED
    assert "amend, do not defer" in record["ruled"]
    assert "LOOSELY AT SCHEDULING" in record["what_was_withdrawn"]
    assert "trainable: head" in record["what_was_withdrawn"]

    # the maintainer's reason is recorded as the maintainer's, and the part
    # it was not possible to find is said to be unfound rather than
    # quietly adopted.
    assert "memorises" in record["the_reason"]
    provenance = record["the_provenance_of_that_reason"]
    assert "the maintainer's recollection, and labelled as such" in provenance
    assert "no result of that kind was found" in provenance
    assert "trainable: full" in provenance

    # The independently measurable half, re-measured here.
    repo = Path(__file__).resolve().parents[1]
    values = {}
    for config in sorted((repo / "configs").glob("*.yaml")):
        for line in config.read_text(encoding="utf-8").splitlines():
            if line.startswith("  trainable: "):
                key = line.split(": ", 1)[1].strip()
                values[key] = values.get(key, 0) + 1
    # [2026-08-30 THE PIN FIRED EXACTLY AS DESIGNED, and the record WAS
    # revisited rather than the test relaxed.] A fine-tuning config now
    # exists -- ONE, the Phase 10 annex's compute gate -- because El
    # the maintainer ruled full fine-tuning operationalised IN THAT ANNEX ONLY
    # (phase10_annex.RULING_B_FULL_TRAINABLE, a scoped departure, not a
    # general unlock). CRITERION_1_AMENDED carries the dated pointer
    # (its own scoped_exception_2026_08_30), asserted below.
    #
    # What the amendment rested on is UNCHANGED where it mattered: the
    # census of every OTHER value is identical to the recorded one, so
    # "the fine-tuning sense was never operationalised" remains true of
    # every arm the amendment was about. The annex is the named
    # exception, not a counter-example.
    # [2026-08-31] Stated as an INVARIANT rather than a count, because
    # the count goes stale predictably -- it moved the moment the annex
    # shipped its second config, which is the fourth-placeholder-flip
    # lesson. The property that must hold: every OTHER vocabulary is
    # exactly what the amendment recorded, and every config that sets
    # 'full' is an ANNEX config. A fine-tuning config anywhere else
    # fires this.
    # [2026-08-31, THE PIN FIRED AGAIN AND AGAIN AS DESIGNED.] `head`
    # moved 81 -> 83 when Phase 20 shipped its two permutation arms at
    # the probe's own `trainable: head` -- which is precisely what makes
    # them controls for the probe rather than for some other arm.
    #
    # Note what did NOT move: the amendment was about the FINE-TUNING
    # sense of `trainable`, and every 'full' config is still an annex
    # config. A growing `head` count is the project shipping arms; it is
    # not evidence about the amendment either way. The count is updated
    # dated rather than loosened, and the delta is enumerated below.
    # moved 83 -> 85 when Phase 25 shipped its two self-supervised arms,
    # both at the probe's own `trainable: head` -- which is exactly what
    # makes them one-factor contrasts against it (p25_arm_d1.yaml,
    # p25_arm_d2.yaml; phase25.THE_TWO_ARMS_RULED). Updated dated rather
    # than loosened, delta enumerated, and nothing about the amendment's
    # fine-tuning sense of `trainable` moved.
    # [UPDATED 2026-09-06] 85 -> 101 when Phase 26 shipped its sixteen
    # calibration cells, every one at the probe's own `trainable: head`.
    # The grid varies weight decay and patience and holds everything
    # else, so the head vocabulary moves by exactly 16 and nothing about
    # the amendment's fine-tuning sense of `trainable` moves at all.
    assert {k: v for k, v in values.items() if k != "full"} == {
        "head": 101, "graph_layers": 34,
        "classifier": 2, "classifier_adabn": 2,
    }, values
    p20_head = sorted(
        config.name for config in sorted((repo / "configs").glob("p20_*.yaml"))
        if any(
            line == "  trainable: head"
            for line in config.read_text(encoding="utf-8").splitlines()
        )
    )
    assert p20_head == [
        "p20_permutation_plain.yaml", "p20_permutation_stratified.yaml",
    ], "the 81 -> 83 delta is Phase 20's two permutation arms and nothing else"
    assert phase15.CRITERION_1_AMENDED["scoped_exception_2026_08_30"] == (
        "phase10_annex.RULING_B_FULL_TRAINABLE"
    )
    full_configs = [
        config.name for config in sorted((repo / "configs").glob("*.yaml"))
        if any(
            line.strip() == "trainable: full"
            for line in config.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert full_configs, "the scoped exception exists; something must set it"
    assert values.get("full") == len(full_configs)
    for name in full_configs:
        assert name.startswith("p10x_"), (
            f"{name} sets trainable: full outside the Phase 10 annex -- the "
            "withdrawal in CRITERION_1_AMENDED covers everywhere else, and "
            "this record must be revisited"
        )

    assert "FROZEN-BACKBONE LINEAR-PROBE TRANSFER" in record[
        "what_criterion_1_now_reads_as"
    ]
    # The amendment names its own suspicious shape and why it is allowed.
    shape = record["the_shape_of_this_amendment"]
    assert "narrowed AFTER the numbers existed" in shape
    assert "REMOVES an unmeasured sense" in shape
    assert "would not be admissible" in shape

    # The original wording is preserved, unedited.
    assert "both " + "registered senses" in (
        phase15.PHASE_15_EXIT_CRITERIA["criteria"][0]
    )


def test_the_closing_cites_only_what_is_recorded():
    """Every figure in the closing traced to the record it came from.

    The closing was written under a no-new-claims rule. This checks the
    rule mechanically for the numbers: each one must also appear in the
    record the closing draws it from.
    """
    closing = phase15.PHASE_15_CLOSING
    assert "all seven exit criteria met" in closing["closed"]

    # The funnel, against stop 1's banked funnel.
    funnel = closing["the_funnel"]
    banked = phase15.STOP_1_BANKED["the_funnel"]
    for count in ("2,550", "2,539", "2,445", "1,519", "62.1%"):
        assert count in funnel and count in banked, count
    assert "0.28x" in funnel
    assert "932 / 394 / 193" in funnel
    assert "932 / 394 / 193" in phase15.STOP_2B_BANKED["the_run"]

    # The tolerance, against the ruling.
    tolerance = closing["the_tolerance_ruling_and_its_provenance"]
    ruling = phase15.BOUNDS_TOLERANCE_RULED
    assert "10 points outside, 0.10" in tolerance
    assert "9 points and 7.8%" in tolerance.replace("\n", " ")
    assert "chosen from a measured" in tolerance
    assert "MEASURED DISTRIBUTION AFTER AN EYE PASS" in ruling[
        "the_provenance_in_the_open"
    ]

    # The guard tally, against the sweep.
    guards = closing["eight_guards_seven_unfired"]
    sweep = phase15.SYNTHETIC_CAVEAT_SWEEP["what_the_sweep_shows"]
    assert "EIGHT" in guards and "SEVEN NEVER FIRED" in guards
    assert "EIGHT GUARDS" in sweep and "SEVEN NEVER FIRED" in sweep
    assert "55 times" in guards and "55 times" in sweep
    assert "first stated as SIX" in guards

    # The withdrawals, all four, by name.
    withdrawals = closing["four_withdrawals_by_name"]
    for name in ("man-2785071", "girl-3956612_1920", "THIRD-AXIS FRAMING"):
        assert name in withdrawals, name
    assert "the maintainer's" in withdrawals

    # The two method lessons.
    assert "MECHANISM, not a PREVALENCE" in closing["mechanism_is_not_rate"]
    assert "0.1652" in closing["mechanism_is_not_rate"]
    assert "0.0019" in closing["mechanism_is_not_rate"]
    assert "SHARED" in closing["artifact_differences_are_pipeline_differences"]

    # The two-digest trap.
    digests = closing["the_two_digest_trap"]
    assert "cannot coincide" in digests
    assert "exactly two writers" in digests


def test_the_closing_carries_the_probe_numbers_it_was_given():
    closing = phase15.PHASE_15_CLOSING
    cells = phase15.STOP_4B_BANKED["cells"]

    verdicts = closing["the_three_probe_verdicts"]
    # Every banked cell's pcc, its interval, its delta and its threshold
    # must appear in the closing that quotes them -- a retyped figure in
    # the prose fails against the record it claims to be citing.
    for name, cell in cells.items():
        assert f"{cell['pcc']:+.4f}" in verdicts, name
        low, high = cell["own_95"]
        assert f"[{low:+.4f}, {high:+.4f}]" in verdicts, name
        assert f"{cell['vs_scut']:+.4f}" in verdicts, name
        assert f"{cell['threshold']:.4f}" in verdicts, name
    assert "ANTI-CORRELATED" in verdicts
    assert "SPANS" in verdicts and "ZERO" in verdicts
    assert "PARITY" in verdicts
    assert "Two findings, not one" in verdicts

    single = closing["the_single_run_comparison"]
    assert "0.0692" in single and "0.0863" in single
    assert "0.0585 against 0.0700" in single
    assert "MOST ROBUST" in single

    contrast = closing["the_within_source_contrast"]
    assert "+0.3230" in contrast and "+0.2836" in contrast
    assert "ESCAPES the mechanism-ii" in contrast
    assert "THE CROP AS A WHOLE" in contrast
    assert "-0.0568" in contrast

    reframing = closing["the_reframing"]
    assert "0.0830" in reframing
    assert "not a new claim about it" in reframing
    assert "crop-geometry sweep" in reframing


def test_the_closing_states_the_verdict_and_what_survives():
    closing = phase15.PHASE_15_CLOSING

    verdict = closing["the_per_cell_verdict_for_phase_16"]
    assert "no single best dataset" in verdict
    assert "pooling would" in verdict
    for figure in ("0.2537", "0.1952", "0.0830", "-0.0693",
                   "0.2001", "-0.0299"):
        assert figure in verdict, figure
    assert "mebeauty_original" in verdict
    assert "does NOT beat ImageNet" in verdict
    assert "ONE-BACKBONE BOUND" in verdict

    meaning = closing["what_transfer_means_here"]
    assert "FROZEN-BACKBONE LINEAR-PROBE TRANSFER and nothing else" in meaning
    assert "No arm was fine-tuned" in meaning

    forward = closing["carried_forward_by_name"]
    for name in ("mebeauty_original", "phase15.caveats_for",
                 "MECHANISM_IS_NOT_RATE",
                 "ARTIFACT_DIFFERENCES_ARE_PIPELINE_DIFFERENCES",
                 "PHASE_6_REFRAMED"):
        assert name in forward, name
    # Everything named as carried forward must actually exist.
    assert callable(phase15.caveats_for)
    for attribute in ("MECHANISM_IS_NOT_RATE",
                      "ARTIFACT_DIFFERENCES_ARE_PIPELINE_DIFFERENCES",
                      "PHASE_6_REFRAMED"):
        assert hasattr(phase15, attribute), attribute

    assert "three instrument defects against one data defect" in closing[
        "the_tally"
    ]


def test_the_phase_is_closed_and_the_walk_agrees():
    walk = phase15.PHASE_15_EXIT_WALK
    assert "MET AS AMENDED" in walk["1_best_is_cleft_side_transfer"]
    assert "CRITERION_1_AMENDED" in walk["1_best_is_cleft_side_transfer"]
    # The flag that prompted the amendment is preserved, not deleted.
    assert "ambiguity flagged" in walk["1_best_is_cleft_side_transfer"]

    assert "the_phase_is_closed" in walk
    assert "all seven are met" in walk["the_phase_is_closed"]
    assert "nothing is left open" in walk["the_phase_is_closed"]

    numbered = [k for k in walk if k[0].isdigit()]
    assert len(numbered) == 7
    for key in numbered:
        assert "MET" in walk[key], key
