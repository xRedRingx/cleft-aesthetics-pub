"""Phase 12, stop 1: the views manifest and the phase's registration."""

from __future__ import annotations

from pathlib import Path

import pytest

from cleft import phase12
from cleft.data import manifest, views


def _source_rows(n: int = 237) -> list[dict]:
    """Rows shaped exactly like cleft_v1's: 237 patients, folder 238
    single-view, folder 143 inverted, folds 0-4 round-robin."""
    header = [name for name, _ in manifest.MANIFEST_COLUMNS]
    rows = []
    for patient in range(1, n + 1):
        if patient == 143:
            frontal, basal = 524, 523
        elif patient == 238:
            frontal, basal = 581, ""
        else:
            # Two ids per folder; parity does not matter to the
            # derivation, which never re-runs the rule.
            frontal, basal = 1000 + 2 * patient, 1001 + 2 * patient
        row = {name: "0.0" for name in header}
        row["patient_id"] = str(patient)
        row["frontal_id"] = str(frontal)
        row["basal_id"] = str(basal)
        row["fold"] = str(patient % 5)
        rows.append(row)
    # 237 folders means ids 1..237 -- patient 238 exists only when the
    # numbering reaches it, so build 1..236 + 238 like the real cohort
    # (folder 52 is empty and 238 is the last).
    return rows


def _real_shape() -> list[dict]:
    rows = _source_rows(236)
    header = [name for name, _ in manifest.MANIFEST_COLUMNS]
    last = {name: "0.0" for name in header}
    last["patient_id"], last["frontal_id"], last["basal_id"] = "238", "581", ""
    last["fold"] = "3"
    return rows + [last]


def test_the_derivation_keeps_236_verbatim_and_drops_only_238():
    rows = _real_shape()
    kept = views.derive(rows)
    assert len(kept) == 236
    assert views.VIEWS_COHORT == {"patients": 236, "frontal": 236, "basal": 236}
    kept_ids = {int(r["patient_id"]) for r in kept}
    assert views.EXCLUDED_PATIENT == 238
    assert 238 not in kept_ids
    assert 143 in kept_ids
    # Verbatim: the kept rows ARE the source rows, labels and folds
    # untouched -- a derivation that rewrote a value would be a rebuild.
    by_id = {int(r["patient_id"]): r for r in rows}
    for row in kept:
        assert row is by_id[int(row["patient_id"])]
    # Every kept row has two distinct views.
    assert all(r["basal_id"] and r["frontal_id"] != r["basal_id"] for r in kept)


def test_the_two_exceptions_are_asserted_not_trusted():
    # 143 swapped -> the exact silent-view-swap failure, refused loudly.
    rows = _real_shape()
    row_143 = next(r for r in rows if r["patient_id"] == "143")
    row_143["frontal_id"], row_143["basal_id"] = "523", "524"
    with pytest.raises(views.ViewsError, match="folder 143 pairing"):
        views.derive(rows)

    # A second single-view patient -> the cohort has changed, refused.
    rows = _real_shape()
    next(r for r in rows if r["patient_id"] == "7")["basal_id"] = ""
    with pytest.raises(views.ViewsError, match="expected\\s+exactly \\[238\\]"):
        views.derive(rows)

    # 238 growing a basal is ALSO a change, not a bonus.
    rows = _real_shape()
    next(r for r in rows if r["patient_id"] == "238")["basal_id"] = "582"
    with pytest.raises(views.ViewsError, match="without a basal view: \\[\\]"):
        views.derive(rows)

    # The wrong row count means this is not cleft_v1.
    with pytest.raises(views.ViewsError, match="237"):
        views.derive(_real_shape()[:-2])

    # One image claimed as both views.
    rows = _real_shape()
    row = next(r for r in rows if r["patient_id"] == "9")
    row["basal_id"] = row["frontal_id"]
    with pytest.raises(views.ViewsError, match="cannot be two views"):
        views.derive(rows)


def test_dropping_238_must_not_erase_a_fold():
    rows = _real_shape()
    # Make one fold consist of patient 238 alone: the drop then removes
    # an entire fold, and the derivation must refuse rather than hand a
    # 4-fold cohort to 5-fold machinery.
    for row in rows:
        row["fold"] = "4" if row["patient_id"] == "238" else str(
            int(row["patient_id"]) % 4
        )
    with pytest.raises(views.ViewsError, match="removed an entire fold"):
        views.derive(rows)


def test_the_pairing_summary_records_provenance_not_just_counts():
    kept = views.derive(_real_shape())
    summary = views.pairing_summary(kept)
    assert summary["patients"] == 236
    assert summary["views_per_patient"] == 2
    assert summary["excluded"]["patient"] == 238
    assert "one-factor" in summary["excluded"]["reason"]
    assert summary["exception_pairing"] == {
        "143": {"frontal_id": 524, "basal_id": 523}
    }
    # The carried-folds decision travels in the artifact, with its reason.
    assert "verbatim" in summary["folds_carried_from"]
    assert "two factors" in summary["folds_carried_from"]
    assert sum(summary["fold_sizes"].values()) == 236
    assert len(summary["fold_sizes"]) == 5

    # The schema keeps cleft_v1's columns so every loader reads it
    # unchanged, and tightens exactly one contract.
    schema = views.views_manifest_schema()
    assert schema["columns"] == manifest.manifest_schema()["columns"]
    assert "NEVER empty" in schema["basal_id_contract"]
    assert "cleft_v1" in schema["derived_from"]


def test_the_phase_is_registered_before_the_build_with_both_sentences():
    record = phase12.PHASE_12_REGISTERED
    assert record["registered"] == "2026-08-23"
    # The premise's status is carried, not resolved by assumption.
    assert "UNRESOLVED" in record["premise_status"]
    assert "BASAL_RATIONALE_UNSUPPORTED" in record["premise_status"]

    cohort = record["cohort"]
    assert cohort["n"] == 236
    assert "frontal-only arm" in cohort["rule"] and "INCLUDED" in cohort["rule"]
    assert "524 = frontal, 523 = basal" in cohort["exception_pairing"]

    arms = record["arms"]
    assert set(arms) == {
        "shared", "a_frontal_only_236", "b_basal_only_236",
        "c_two_view_concat", "d_capacity_control",
    }
    assert "RE-RUN on the 236" in arms["a_frontal_only_236"]
    assert "SAME vector duplicated" in arms["d_capacity_control"]
    assert "C > D isolates the view" in arms["d_capacity_control"]

    # Both interpretation sentences, registered before any number, and
    # used only under reading 1.
    sentences = phase12.INTERPRETATION_SENTENCES_PREWRITTEN
    assert "reading 1" in sentences["applies_when"]
    assert "RESTORES evidence" in sentences["if_raters_saw_both"]
    assert "RATERS NEVER SAW" in sentences["if_raters_saw_frontal_only"]
    assert "question 8" in sentences["neither_is_chosen_here"]

    # Four readings, the prior owned as the maintainer's, D's logic stated.
    readings = phase12.PHASE_12_READINGS
    assert "BOTH conditions" in readings["1_view_signal"]
    assert "REGISTERED PRIOR" in readings["2_descriptive_only"]
    assert "+0.0073" in readings["2_descriptive_only"]
    assert "stays dead" in readings["3_no_gain"]
    assert "no story until" in readings["4_b_near_a"]
    assert "capacity" in readings["capacity_control_logic"]

    # The ethics gate is registered now and attaches at first pixel use.
    gate = record["ethics_gate"]
    assert "basal_use_acknowledged" in gate
    assert "DIFFERENT USE" in gate
    assert "first PIXEL use" in gate
    # Four stops, review after each, and the eye-review rule by name.
    stops = record["stops"]
    assert set(stops) == {"1", "2", "3", "4", "rule"}
    assert "by\nEYE" in stops["2"] or "BY EYE" in stops["2"].replace("\n", " ")
    assert "or not at all" in stops["2"]


def test_stop_one_is_recorded_and_the_config_matches_the_module():
    import yaml

    record = phase12.STOP_1_MANIFEST
    assert record["built"].endswith("stop 1 of 4")
    assert len(record["asserted"]) == 6
    assert "one factor, not two" in record["folds"]
    assert "never re-derived" in record["derivation"]

    repo = Path(__file__).resolve().parents[1]
    path = repo / "configs" / "p12_views_manifest.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    task = payload["task"]
    assert task["kind"] == "build_views_manifest"
    assert task["out_version"] == "cleft_v1_views"
    assert (
        task["expect_patients"], task["expect_frontal"], task["expect_basal"]
    ) == (236, 236, 236)
    # One input, the source manifest, hash carried and real.
    inputs = payload["inputs"]
    assert len(inputs) == 1 and inputs[0]["name"] == "manifest_v1"
    assert set(inputs[0]["rollup_sha256"]) != {"0"}

    header = path.read_text(encoding="utf-8")
    assert "A DERIVATION, NOT A REBUILD" in header
    assert "FOLDS CARRIED, NOT RE-STRATIFIED" in header
    assert "NO PIXELS ARE TOUCHED HERE" in header
    assert "basal_use_acknowledged" in header

    # The generator reproduces the shipped config byte for byte.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_phase12_configs",
        repo / "scripts" / "generate_phase12_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-23, stop 2: basal staging, the gate, and the sheets
# --------------------------------------------------------------------------


def test_corner_whiteness_measures_the_source_corners():
    import numpy as np

    from cleft.geometry import basal

    white = np.full((100, 80, 3), 255, dtype=np.uint8)
    assert basal.corner_white_fraction(white) == 1.0

    black = np.zeros((100, 80, 3), dtype=np.uint8)
    assert basal.corner_white_fraction(black) == 0.0

    # White corners around a dark face: the frontal-crop shape. JPEG
    # ringing at 252 still counts as white; 249 does not.
    face = np.full((100, 80, 3), 252, dtype=np.uint8)
    face[20:80, 10:70] = 30
    assert basal.corner_white_fraction(face) == 1.0
    dim = np.full((100, 80, 3), 249, dtype=np.uint8)
    assert basal.corner_white_fraction(dim) == 0.0

    # One dark corner of four: fraction is the mean over corners.
    three = np.full((100, 80, 3), 255, dtype=np.uint8)
    three[:16, :16] = 0
    assert abs(basal.corner_white_fraction(three) - 0.75) < 1e-9

    # Greyscale images measure too -- the loader promises RGB, but the
    # measurement must not silently lie if handed 2-D.
    grey = np.full((50, 50), 255, dtype=np.uint8)
    assert basal.corner_white_fraction(grey) == 1.0


def test_the_whiteness_verdict_applies_the_registered_thresholds():
    from cleft.geometry import basal

    # White-dominant cohort: transfers, with the flagged few counted.
    good = [0.99] * 230 + [0.5] * 6
    verdict = basal.whiteness_verdict(good)
    assert verdict["white_pad_rationale_transfers"] is True
    assert verdict["flagged_below_0_90"] == 6
    assert verdict["verdict"].startswith("TRANSFERS")
    assert "front of\nthe eye" in verdict["verdict"] or "front of" in verdict["verdict"]

    # Median below 0.90: the decision REOPENS before any embedding.
    bad = [0.5] * 120 + [0.99] * 116
    verdict = basal.whiteness_verdict(bad)
    assert verdict["white_pad_rationale_transfers"] is False
    assert verdict["verdict"].startswith("REOPEN")
    assert "before any embedding" in verdict["verdict"]

    with pytest.raises(basal.BasalStagingError):
        basal.whiteness_verdict([])

    # The thresholds are the registered ones, not re-decided in the code.
    assert basal.FLAG_BELOW == 0.90
    assert basal.REOPEN_IF_MEDIAN_BELOW == 0.90
    assert basal.FRONTAL_AR_RANGE == (0.553, 1.099)


def test_the_staging_decision_is_measured_from_the_code_path():
    """The decision's claims are checkable against the code they cite --
    so they are checked, not quoted."""
    import inspect

    record = phase12.STOP_2_STAGING_DECISION
    assert "false choice" in record["measured_answer"]
    assert "NOT EXERCISED" in record["measured_answer"]

    # Claim 1: stage() consumes only the aspect ratio -- the frozen
    # staging module neither imports nor constructs any anatomy. Its
    # docstring MENTIONS trapezium.py in prose, which is why this checks
    # identifiers rather than words.
    from cleft.geometry import staging

    source = inspect.getsource(staging)
    assert "from .trapezium" not in source
    assert "import trapezium" not in source
    assert "Trapezium" not in source  # the class; prose stays lowercase
    for anatomical in ("landmark", "nostril"):
        assert anatomical not in source.lower(), anatomical

    # Claim 2: the trapezium is consumed by exactly G2's unwarp and the
    # patch generators -- at G1 stage_build keeps base.image untouched.
    from cleft.geometry import stage_build

    build_source = inspect.getsource(stage_build.build)
    assert 'base.image if geometry == "g1" else unwarp(base.image)' in (
        build_source
    )

    # The boundary and the sixth-instance lesson both travel.
    assert "WHOLE-IMAGE G1 ONLY" in record["boundary"]
    assert "REOPENS" in record["boundary"]
    assert "rationale tagged MEASURED" in record["why_this_is_a_measurement"]
    deferred = record["deferred_to_pixels"]
    assert "MEDIAN < 0.90 REOPENS" in deferred["white_pad_continuity"]
    assert "never refused by code" in deferred["aspect_ratio"]


def test_the_gate_refuses_before_any_pixel_and_only_a_deliberate_act_opens_it():
    import inspect

    from cleft import run as run_module

    assert "stage_basal_views" in run_module.TASKS
    source = inspect.getsource(run_module.task_stage_basal_views)

    # The gate is the FIRST act: its check precedes every read. CALL
    # sites, not import lines -- the imports sit above everything and
    # matching them would compare the wrong positions.
    gate_at = source.index('basal_use_acknowledged") is not True')
    for pixel_touch in (
        "load_manifest(manifest_dir", "contact.find_image(",
        "render.load_image(",
    ):
        assert gate_at < source.index(pixel_touch), pixel_touch
    # The refusal names the reason and the person, not just the flag.
    assert "REC approvals" in source
    assert "the maintainer" in source
    assert "DIFFERENT USE" in source

    # The boundary is asserted in what the task writes: G1 only.
    assert "staged_patient_g1.npy" in source
    assert "unwarp" not in source
    assert "patches" not in source
    # No embedding is extracted here; the sheets come first. Identifiers,
    # not words -- the docstring SAYS "no embedding is extracted", which
    # is the point, so the check is for extraction machinery.
    for machinery in (
        "embeddings_module_load", "timm", "create_model", "Backbone",
    ):
        assert machinery not in source, machinery
    assert "NO EMBEDDING" in source

    # The sheets: all crops, most suspicious first, frozen stage() used.
    assert 'key=lambda i: staged_rows[i]["corner_white_fraction"]' in source
    assert "from .geometry.staging import stage" in source
    assert "basal.PANELS_PER_SHEET" in source

    record = phase12.BASAL_USE_GATE
    assert "NO default" in record["mechanism"]
    assert "only by the edit" in record["mechanism"].replace(
        "only\nby", "only by"
    ) or "the maintainer" in record["mechanism"]
    assert "first PIXEL use" in record["attaches_at"]


def test_the_generator_never_sets_the_flag_and_carries_the_dated_edit(tmp_path):
    import importlib.util

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "generate_phase12_configs",
        repo / "scripts" / "generate_phase12_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # A fresh render -- no shipped config -- writes FALSE and PLACEHOLDER.
    missing = tmp_path / "nothing.yaml"
    assert module._carried_flag(missing) is False
    assert module._carried_views_hash(missing) == "0" * 64

    # the true, and a filled hash, are carried from the file.
    edited = tmp_path / "edited.yaml"
    edited.write_text(
        "inputs:\n"
        "- name: manifest_views\n"
        f"  path: {module.VIEWS_MANIFEST_PATH}\n"
        "  rollup_sha256: " + "ab" * 32 + "\n"
        "task:\n"
        "  basal_use_acknowledged: true\n",
        encoding="utf-8",
    )
    assert module._carried_flag(edited) is True
    assert module._carried_views_hash(edited) == "ab" * 32

    # Anything short of literal true is false -- a truthy string is not
    # a confirmation.
    stringy = tmp_path / "stringy.yaml"
    stringy.write_text(
        "task:\n  basal_use_acknowledged: yes-ish\n", encoding="utf-8"
    )
    assert module._carried_flag(stringy) is False

    # [CORRECTED 2026-08-23, by a CI failure] This used to pin the
    # shipped flag at false -- pinning a value that is the maintainer's to
    # change, so the acknowledgement (commit 2e43cbd, REC 13/SW/0064 and
    # 23/YH/0037) broke the build. The invariant was never the value; it
    # is CONSISTENCY: the flag is a real boolean, the header describes
    # the same state the body carries, and regeneration is clean either
    # way. That is what is asserted now.
    import yaml as yaml_module

    shipped = repo / "configs" / "p12_stage_basal.yaml"
    text = shipped.read_text(encoding="utf-8")
    flag = yaml_module.safe_load(text)["task"]["basal_use_acknowledged"]
    assert isinstance(flag, bool)
    if flag:
        assert "A DATED CONFIRMATION" in text
        assert "THE TASK WILL REFUSE" not in text
    else:
        assert "THE TASK WILL REFUSE" in text
        assert "A DATED CONFIRMATION" not in text
    assert "PLACEHOLDER" not in text
    assert "**RESOLVED.**" in text
    assert module.main(["--check"]) == 0


def test_stop_two_is_recorded_with_the_sheets_and_the_review_rule():
    import yaml

    record = phase12.STOP_2_STAGING
    assert record["built"].endswith("NOT launched")
    assert "ALL 236" in record["sheets"]
    assert "ascending" in record["sheets"]
    assert "sheet 1" in record["sheets"]
    assert "no G2 tensor, no patch definitions" in record["boundary_asserted"]
    assert "by eye or not\nat all" in record["no_embedding_before_the_eye"] or (
        "by eye or not" in record["no_embedding_before_the_eye"]
    )
    assert "review: PENDING" in record["no_embedding_before_the_eye"]

    repo = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (repo / "configs" / "p12_stage_basal.yaml").read_text(encoding="utf-8")
    )
    task = payload["task"]
    assert task["kind"] == "stage_basal_views"
    assert task["out_version"] == "staged_basal_v1"
    assert task["expect_patients"] == 236
    # [CORRECTED 2026-08-23] Was `is False`, which pinned the maintainer's
    # decision and broke CI the moment it was made (commit 2e43cbd). The
    # type is the contract, and the value is the maintainer's.
    assert isinstance(task["basal_use_acknowledged"], bool)
    # Two inputs, both verified. [2026-08-23] The views manifest's hash
    # was a documented placeholder until the declare pass filled it; the
    # generator carries the fill, so a regeneration keeps it.
    inputs = {e["name"]: e for e in payload["inputs"]}
    assert set(inputs) == {"manifest_views", "patient_folders"}
    for entry in inputs.values():
        assert set(entry["rollup_sha256"]) != {"0"}
        assert len(entry["rollup_sha256"]) == 64
        assert set(entry["rollup_sha256"]) <= set("0123456789abcdef")
    assert inputs["manifest_views"]["rollup_sha256"].startswith("1b0f96bd")


# --------------------------------------------------------------------------
# 2026-08-23, stop 2's run: the REOPEN fires, the eye passes, the pad
# decision is proposed with its measurements
# --------------------------------------------------------------------------


def test_the_reopen_fired_by_the_rule_and_the_eye_passed_the_images():
    record = phase12.STOP_2_OBSERVED
    assert "p12_stage_basal__978b67c4__p12-stage-basal" in record["observed"]
    # The retries were refused by the never-overwrite guard, and counted.
    assert "attempt 0 clean" in record["attempts"]
    assert "+3 on\nphase8" in record["attempts"] or "+3 on" in record["attempts"]
    assert "RETRY_LIMIT_IS_NOT_HOLDING" in record["attempts"]

    figures = record["figures"]
    assert figures["aspect_ratio"] == (0.624, 1.562)
    assert figures["corner_white_median"] == 0.5234
    assert "ALL 236" in figures["flagged"]
    # The reopen is the rule firing, and the record says so rather than
    # treating it as a failure.
    assert "0.5234 < 0.90" in record["reopen_fired"]
    assert "registered to do" in record["reopen_fired"]
    # Both halves of the eye report: images pass, corners have structure.
    assert record["eye_review"].startswith("PASS as images")
    assert "NOT a defect" in record["corner_structure"]
    assert "TOP corners" in record["corner_structure"]

    # The whiteness verdict the run computed reproduces from the rule.
    from cleft.geometry import basal

    verdict = basal.whiteness_verdict([0.5234] * 236)
    assert verdict["white_pad_rationale_transfers"] is False
    assert verdict["verdict"].startswith("REOPEN")


def test_the_pad_quantification_is_grounded_in_the_frozen_stage():
    """The closed form the proposal rests on is checked against the
    executed staging function, not asserted from algebra."""
    import numpy as np

    from cleft.geometry import basal
    from cleft.geometry.staging import stage

    for width, height in ((624, 1000), (1000, 1000), (1562, 1000), (553, 1000)):
        ar = width / height
        closed = basal.pad_fraction_for_ar(ar)
        measured = stage(np.zeros((height, width, 3), dtype=np.uint8)).pad_fraction
        assert abs(closed - measured) < 0.003, (ar, closed, measured)

    # The record's endpoint figures are the function's own values.
    quantification = phase12.PAD_DECISION_REOPENED["pad_quantification"]
    assert abs(
        quantification["basal_endpoints"]["ar_0.624"]
        - basal.pad_fraction_for_ar(0.624)
    ) < 5e-4
    assert abs(
        quantification["basal_endpoints"]["ar_1.562"]
        - basal.pad_fraction_for_ar(1.562)
    ) < 5e-4
    # The bound: both basal endpoints below the frontal measured max --
    # and pad() maximised at endpoints is what makes the bound cover the
    # whole cohort from its AR range alone.
    assert basal.pad_fraction_for_ar(0.624) < 0.4464
    assert basal.pad_fraction_for_ar(1.562) < 0.4464
    assert "maximised at the ENDPOINTS" in quantification["closed_form"]
    assert "placement" in quantification["what_is_new_is_placement"] or (
        "BANDS" in quantification["what_is_new_is_placement"]
    )
    with pytest.raises(basal.BasalStagingError):
        basal.pad_fraction_for_ar(0.0)


def test_the_pad_proposal_proposes_a_and_picks_nothing():
    record = phase12.PAD_DECISION_REOPENED
    assert record["proposed"].endswith("the choice is the maintainer's")
    assert "measured FALSE of basals" in record["what_died"]

    a = record["a_pad_white"]
    assert a["status"] == "PROPOSED"
    # The one-fill argument survives as a MEASURED argument, not nostalgia.
    assert "~52% baked white" in a["one_fill_measured"]
    assert "ANY other" in a["one_fill_measured"]
    assert "fewest-distinct-fills" in a["one_fill_measured"]
    assert "VIEW and nothing" in a["recipe_identity"]
    assert "reusable AS-IS" in a["artifact_stands"]
    assert "consistency-not-continuity" in a["honest_recording"]

    b = record["b_edge_statistic_pad"]
    assert b["status"] == "costed, not proposed"
    assert len(b["costs"]) == 4
    assert any("second factor" in c and "D exists to exclude" in c
               for c in b["costs"])
    assert any("eye review are discarded" in c for c in b["costs"])

    # (c) is examined, not waved at, and each variant has a ground.
    c = record["c_examined"]
    assert "36% of the width" in c["crop_to_square"]
    assert "NO image" in c["mid_grey_or_mean_pad"]
    assert "collapses to (a)" in c["pad_to_the_baked_masks_fill"]

    # The honesty clause: no measurement decides it, and outcome-shopping
    # is refused by name.
    assert record["no_cheap_deciding_measurement"].startswith("honestly: none")
    assert "OUTCOME-SHOPPING" in record["no_cheap_deciding_measurement"]
    assert "cannot\ndecide" in record["no_cheap_deciding_measurement"] or (
        "cannot" in record["no_cheap_deciding_measurement"]
    )
    # The confound check is a report with a reading, not a gate.
    assert "report-with-reading, not a gate" in record["confound_check_registered"]
    # Reusability is answered for both branches.
    assert "as-is" in record["reusability"]["under_a"]
    assert "staged_basal_v2" in record["reusability"]["under_b_or_c"]


# --------------------------------------------------------------------------
# 2026-08-23, stop 3: the pad decision taken, the four arms built
# --------------------------------------------------------------------------


def test_the_pad_decision_is_taken_with_the_completed_table():
    record = phase12.PAD_DECISION_TAKEN
    assert record["decided"].endswith("(a), pad white")
    assert record["rationale_as_recorded"] == (
        "consistency-not-continuity, plus fewest-fills"
    )
    assert "confound D exists to exclude" in record["b_rejected"]
    assert "OUTCOME-SHOPPING" in record["no_deciding_measurement"]
    assert "no\nre-stage" in record["artifact"] or "no" in record["artifact"]
    assert "visual check pass attached" in record["artifact"]

    table = record["pad_table_completed"]
    assert table["basal"] == {"min": 0.0, "mean": 0.1451, "max": 0.375}
    assert table["basal"]["mean"] < table["frontal"]["mean"]
    # The endpoint bound predicted 0.3760 and the measured max is 0.375.
    from cleft.geometry import basal

    assert basal.pad_fraction_for_ar(0.624) >= table["basal"]["max"]
    assert "endpoint bound held" in table["reading"]
    assert "PLACEMENT alone" in record["residual_novelty"]
    assert "never showed the\nbackbone" in record["residual_novelty"] or (
        "never showed the" in record["residual_novelty"]
    )


def test_stop_three_registers_reuse_alignment_and_the_confound_reading():
    record = phase12.STOP_3_REGISTERED
    reuse = record["frontal_embeddings_reused"]
    assert "vit_b16__imagenet__g1" in reuse["artifact"]
    assert "p7_d1_vit_b16_imagenet_g1.yaml" in reuse["artifact"]
    assert "NO PIXEL CHANGED" in reuse["why_reusable"]
    assert "exactly {238}" in reuse["alignment"]
    assert "never by position" in reuse["alignment"]

    assert "No new extraction code" in record["basal_extraction"]["machinery"]
    arms = record["arms"]
    assert arms["channels"]["d"] == ("frontal", "frontal")
    assert arms["head_parameters"] == {"a": 769, "b": 769, "c": 1537, "d": 1537}
    assert "recipe identity by construction" in arms["recipe"]

    confound = record["confound_report"]
    assert "0.13" in confound["reading_registered"]
    assert "REPORT, NEVER GATE" in confound["reading_registered"]
    assert "ATTACHES" in confound["reading_registered"]
    assert record["stop_4_unchanged"].endswith("stays stop 4")


def test_the_four_arm_configs_differ_only_in_the_channel_axis():
    """One-factor by construction, checked byte-for-byte: strip the keys
    the channel axis owns, and the four task blocks must be identical --
    and equal to the 0.2520 arm's recipe."""
    import yaml

    repo = Path(__file__).resolve().parents[1]
    channel_keys = {
        "arm", "views", "frontal_embeddings", "basal_embeddings",
        "expect_head_parameters",
    }
    stripped = {}
    for name in (
        "p12_arm_a_frontal", "p12_arm_b_basal", "p12_arm_c_concat",
        "p12_arm_d_capacity",
    ):
        payload = yaml.safe_load(
            (repo / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
        )
        task = payload["task"]
        stripped[name] = {
            k: v for k, v in task.items() if k not in channel_keys
        }
        # Every arm runs the 236 cohort on the views manifest.
        assert task["manifest_artifact"] == "manifest_views"
        assert task["expect_patients"] == 236
        # Head parameters follow the channel dimensionality: 768 per
        # channel, plus the bias.
        assert task["expect_head_parameters"] == 768 * len(task["views"]) + 1

    assert len({str(sorted(s.items())) for s in stripped.values()}) == 1

    # And the shared recipe IS the 0.2520 arm's, key for key.
    ladder = yaml.safe_load(
        (repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    shared = stripped["p12_arm_a_frontal"]
    for key in (
        "label", "seeds", "inner_val_frac", "max_epochs", "patience",
        "monitor", "learning_rate", "weight_decay",
    ):
        assert shared[key] == ladder[key], key

    # [2026-08-23] All four arms are now FULLY declared: A and D were
    # from birth, and B and C were filled by the declare pass after the
    # basal extraction ran (rollup 08d376e3...). Every hash real, every
    # config carrying the one-factor line.
    for name in (
        "p12_arm_a_frontal", "p12_arm_d_capacity",
        "p12_arm_b_basal", "p12_arm_c_concat",
    ):
        payload = yaml.safe_load(
            (repo / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
        )
        for entry in payload["inputs"]:
            assert set(entry["rollup_sha256"]) != {"0"}, name
            assert len(entry["rollup_sha256"]) == 64
        text = (repo / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
        assert "PLACEHOLDER" not in text
        assert "one-factor by construction" in text
    for name in ("p12_arm_b_basal", "p12_arm_c_concat"):
        payload = yaml.safe_load(
            (repo / "configs" / f"{name}.yaml").read_text(encoding="utf-8")
        )
        basal = next(
            e for e in payload["inputs"] if e["name"] == "basal_embeddings"
        )
        assert basal["rollup_sha256"].startswith("08d376e3")


def test_the_frontal_channel_aligns_by_id_and_asserts_the_drop():
    """The 237-set reuse: aligned by key lookup, the dropped set asserted
    to be exactly {238} -- exercised through a monkeypatched loader, so
    the logic runs rather than being read."""
    import numpy as np

    from cleft import embeddings as embeddings_module
    from cleft import run as run_module

    ids_237 = [i for i in range(1, 239) if i != 52]  # 237 ids incl. 238
    values = np.arange(len(ids_237) * 4, dtype=float).reshape(-1, 4)

    real_load = embeddings_module.load
    try:
        embeddings_module.load = lambda directory, manifest_ids=None: (
            values, {"patient_ids": list(ids_237)}
        )
        views_ids = [i for i in ids_237 if i != 238]
        out = run_module._view_channel(
            "frontal", {"embeddings": Path(".")},
            {"frontal_embeddings": "embeddings"}, views_ids,
        )
        # Row i of the output is the source row for that patient id.
        for row, pid in zip(out, views_ids):
            assert row[0] == values[ids_237.index(pid)][0]

        # A drop that is not exactly {238} refuses.
        import pytest as _pytest

        with _pytest.raises(ValueError, match="expected exactly"):
            run_module._view_channel(
                "frontal", {"embeddings": Path(".")},
                {"frontal_embeddings": "embeddings"},
                [i for i in ids_237 if i not in (238, 7)],
            )
        # A views cohort the set does not cover refuses too.
        with _pytest.raises(ValueError, match="lacks"):
            run_module._view_channel(
                "frontal", {"embeddings": Path(".")},
                {"frontal_embeddings": "embeddings"}, views_ids + [999],
            )
    finally:
        embeddings_module.load = real_load


def test_the_arm_task_is_one_factor_and_checks_its_head():
    import inspect

    from cleft import run as run_module

    for kind in ("extract_basal_embeddings", "view_arm"):
        assert kind in run_module.TASKS
    source = inspect.getsource(run_module.task_view_arm)
    # One task, channel list from config, head size asserted against the
    # registration's expectation.
    assert "for name in channels" in source or "in\nchannels" in source
    assert "np.concatenate(blocks, axis=1)" in source
    assert "expect_head_parameters" in source
    assert "registration disagree" in source
    # Every arm, the frontal-only one included, runs the 236 cohort.
    assert "frontal-only arm included" in source
    # The basal channel loads STRICT; the frontal aligns by id.
    helper = inspect.getsource(run_module._view_channel)
    assert 'if name == "basal":' in helper
    assert "embeddings_module.load(directory, manifest_ids)" in helper
    assert "EXCLUDED_PATIENT" in helper

    # The extraction task reuses the existing machinery and computes the
    # registered confound report with its threshold.
    extract_source = inspect.getsource(
        run_module.task_extract_basal_embeddings
    )
    assert "extract.extract_features(" in extract_source
    assert "embeddings_module.save(" in extract_source
    assert "manifest_ids=manifest_ids" in extract_source
    assert "threshold = 0.13" in extract_source
    assert "caveat_attaches_to_b_and_c" in extract_source
    # [2026-08-23] The task-level overwrite guard was REMOVED with the
    # guard-after-mkdir fix: the save layer owns existence and creation
    # (phase12.EXTRACT_GUARD_AFTER_MKDIR), so the task carries the fix
    # marker instead of a duplicate guard.
    assert "NEVER overwrite" not in extract_source
    assert "EXTRACT_GUARD_AFTER_MKDIR" in extract_source


# --------------------------------------------------------------------------
# 2026-08-23, the extraction banked, the confound fired, A and D measured
# --------------------------------------------------------------------------


def test_the_confound_reading_fired_on_aspect_ratio_and_attaches():
    record = phase12.BASAL_CONFOUND_OBSERVED
    assert "p12-extract-basal-2" in record["observed"]
    assert "(236, 768)" in record["extraction"]
    # The registered rule, applied exactly: one of two r's clears 0.13.
    assert record["r_aspect_ratio_vs_mean"] == 0.1601
    assert record["r_corner_white_vs_mean"] == 0.0977
    assert abs(record["r_aspect_ratio_vs_mean"]) >= record["threshold"]
    assert abs(record["r_corner_white_vs_mean"]) < record["threshold"]
    assert record["fired_on"] == "aspect ratio only"
    # The caveat attaches -- with the anatomy-vs-artifact honesty that is
    # the reason it attaches rather than gates.
    caveat = record["caveat"]
    assert caveat.startswith("ATTACHES to arms B and C")
    assert "cannot distinguish real" in caveat
    assert "attaches rather than gates" in caveat
    assert "travels with it" in caveat
    # The threshold is the registered one, not re-decided.
    assert record["threshold"] == 0.13


def test_arms_a_and_d_are_banked_and_the_control_did_its_job():
    import numpy as np

    record = phase12.ARMS_A_D_OBSERVED
    a, d = record["a_frontal_only_236"], record["d_capacity_control"]
    # The means are the means of the recorded seeds, not transcriptions.
    assert abs(float(np.mean(a["per_seed"])) - a["mean"]) < 6e-4
    assert abs(float(np.mean(d["per_seed"])) - d["mean"]) < 6e-4
    assert len(a["per_seed"]) == 5 and len(d["per_seed"]) == 5

    # A is the bar, essentially the ladder's 0.2520.
    assert abs(a["mean"] - 0.2520) < 0.005
    assert "inert" in a["reading"]
    # D within noise of A: the capacity control doing its job.
    assert abs(d["mean"] - a["mean"]) < 0.005
    assert "SAME" in d["reading"]
    assert "control doing exactly its job" in d["reading"]
    # Stop 4's C-vs-D floor is measured, and stated as the floor.
    assert record["stop_4_floor"].startswith("MEASURED")
    assert "0.2534" in record["stop_4_floor"]


# --------------------------------------------------------------------------
# 2026-08-23, B and C banked; stop 4 registered and built
# --------------------------------------------------------------------------


def test_arms_b_and_c_are_banked_with_the_caveat_and_the_seed_structure():
    import numpy as np

    record = phase12.ARMS_B_C_OBSERVED
    b = record["b_basal_only_236"]
    c = record["c_two_view_concat"]
    assert abs(float(np.mean(b["per_seed"])) - b["mean"]) < 6e-4
    assert abs(float(np.mean(c["per_seed"])) - c["mean"]) < 6e-4
    assert "frontal-independent signal" in b["reading"]
    assert "highest arm mean" in c["reading"]
    assert "BASAL_CONFOUND_OBSERVED" in record["caveat"]
    assert "+0.1601" in record["caveat"]

    # The 3/5 direction claim reproduces from the recorded seeds: C runs
    # positive against A and D on the first three, negative on 99 and
    # 12345 -- which are B's two worst.
    a = phase12.ARMS_A_D_OBSERVED["a_frontal_only_236"]["per_seed"]
    d = phase12.ARMS_A_D_OBSERVED["d_capacity_control"]["per_seed"]
    signs_vs_a = [ci - ai > 0 for ci, ai in zip(c["per_seed"], a)]
    signs_vs_d = [ci - di > 0 for ci, di in zip(c["per_seed"], d)]
    assert signs_vs_a == [True, True, True, False, False]
    assert signs_vs_d == [True, True, True, False, False]
    # Seed order is the configs' (1337, 2024, 7, 99, 12345), so the two
    # negatives are 99 and 12345 -- and they are B's two worst seeds.
    worst_two = sorted(range(5), key=lambda i: b["per_seed"][i])[:2]
    assert sorted(worst_two) == [3, 4]


def test_the_p12_pairs_are_derived_from_the_observed_records():
    import numpy as np

    from cleft import ladder
    from cleft.train.phase3 import combined_claimable_delta

    pairs = phase12.paired_claim_pairs("p12")
    assert [p["key"] for p in pairs] == [
        "p12__basal_vs_frontal_bar", "p12__concat_vs_frontal_bar",
        "p12__concat_vs_capacity",
    ]
    per_seed = {
        "p12_arm_a_frontal":
            phase12.ARMS_A_D_OBSERVED["a_frontal_only_236"]["per_seed"],
        "p12_arm_d_capacity":
            phase12.ARMS_A_D_OBSERVED["d_capacity_control"]["per_seed"],
        "p12_arm_b_basal":
            phase12.ARMS_B_C_OBSERVED["b_basal_only_236"]["per_seed"],
        "p12_arm_c_concat":
            phase12.ARMS_B_C_OBSERVED["c_two_view_concat"]["per_seed"],
    }
    for pair in pairs:
        va, vb = per_seed[pair["a"]], per_seed[pair["b"]]
        delta = float(np.mean(vb)) - float(np.mean(va))
        threshold = combined_claimable_delta(
            float(np.std(va, ddof=1)), 5, float(np.std(vb, ddof=1)), 5
        )["arm_means_95"]
        recorded = pair["recorded"]
        assert abs(recorded["delta_of_means"] - delta) < 6e-4, pair["key"]
        assert abs(recorded["threshold"] - threshold) < 6e-4, pair["key"]
        assert "positive =" in recorded["positive_means"]
        assert "DESCRIPTIVE, not claimable" in recorded["source"]
        assert pair["seeds"] == list(ladder.SEED_POOL[:5])

    # Both C margins sit far below 1: condition 2 already fails on the
    # recorded figures, which is what STOP_4_REGISTERED says.
    by_key = {p["key"]: p["recorded"] for p in pairs}
    assert by_key["p12__concat_vs_frontal_bar"]["margin"] < 1.0
    assert by_key["p12__concat_vs_capacity"]["margin"] < 1.0
    assert 1.0 < by_key["p12__basal_vs_frontal_bar"]["margin"] < 2.55

    # The ladder's generic derivations accept these pairs unchanged.
    vectors = ladder.paired_claim_vectors("p12", pairs=pairs)
    stems = ladder.paired_claim_stems("p12", pairs=pairs)
    assert len(vectors) == 20 and len(stems) == 4
    assert len(ladder.paired_claim_seed_groups("p12", pairs=pairs)) == 1

    with pytest.raises(phase12.Phase12Error, match="unknown Phase 12"):
        phase12.paired_claim_pairs("p13")


def test_stop_four_registers_the_prediction_as_arithmetic():
    record = phase12.STOP_4_REGISTERED
    assert record["config"] == "configs/p12_paired.yaml"
    assert "arithmetic on recorded figures, not" in (
        record["prediction_is_arithmetic"]
    )
    decided = record["condition_2_already_decided"]
    # The record's figures agree with the derived pairs to the rounding
    # the record carries.
    by_key = {
        p["key"]: p["recorded"] for p in phase12.paired_claim_pairs("p12")
    }
    assert str(by_key["p12__concat_vs_frontal_bar"]["margin"]) in (
        decided["c_vs_a"]
    )
    assert str(by_key["p12__concat_vs_capacity"]["margin"]) in (
        decided["c_vs_d"]
    )
    assert "2.55x floor" in decided["b_vs_a"]
    assert "noisier side" in decided["why"]
    assert "interval widths alone" in record["what_remains_at_risk"]
    # If the intervals surprise, the recorded figures were wrong -- and
    # THAT would be the finding.
    assert "THAT would be the finding" in record["honest_sentence_prewritten"]


def test_the_p12_paired_config_is_two_pass_with_four_pending_stems():
    import importlib.util
    import inspect

    import yaml

    repo = Path(__file__).resolve().parents[1]
    path = repo / "configs" / "p12_paired.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    task = payload["task"]
    assert task["kind"] == "paired_claims"
    assert task["scope"] == "p12"
    assert task["n_boot"] == 10000

    inputs = payload["inputs"]
    assert len(inputs) == 20
    # [2026-08-23, BOTH PASSES DONE] The four run directories were
    # pasted, then declare_inputs.py verified all twenty files: every
    # path real and contract-checked, every hash verified, nothing left
    # for guard 3 to refuse.
    from cleft.run_names import check_run_dir

    directories = {e["path"].split("/")[-2] for e in inputs}
    assert directories == {
        "p12_arm_a_frontal__0dc7c79b__p12-arm-a-frontal",
        "p12_arm_b_basal__0fb33b6a__p12-arm-b-basal",
        "p12_arm_c_concat__0fb33b6a__p12-arm-c-concat",
        "p12_arm_d_capacity__0dc7c79b__p12-arm-d-capacity",
    }
    for directory in directories:
        check_run_dir(directory)
    assert not any("PENDING" in e["path"] for e in inputs)
    for entry in inputs:
        assert set(entry["rollup_sha256"]) != {"0"}
        assert len(entry["rollup_sha256"]) == 64
        assert set(entry["rollup_sha256"]) <= set("0123456789abcdef")
    # Twenty vectors, twenty distinct files: a repeated hash would mean
    # two seeds -- or two ARMS -- wrote identical predictions, and for
    # arms differing only in the channel list that corruption would
    # still produce ordinary-looking intervals.
    assert len({e["rollup_sha256"] for e in inputs}) == 20

    header = path.read_text(encoding="utf-8")
    assert "**RESOLVED.**" in header
    assert "PENDING" not in header
    assert "FITS NOTHING" in header
    # The derived prediction travels in the config itself.
    assert "FAILS cond 2" in header
    assert "under the 2.55x floor" in header
    assert "interval widths alone" in header
    assert "BASAL_CONFOUND_OBSERVED" in header

    # The scope reaches the dispatcher and the schema.
    from cleft import run as run_module
    from cleft.config.schema import TASK_SPECS

    assert "p12" in TASK_SPECS["paired_claims"]["scope"].choices
    dispatcher = inspect.getsource(run_module.task_paired_claims)
    assert 'else phase12 if scope == "p12"' in dispatcher

    # And the generator reproduces the shipped config byte for byte.
    spec = importlib.util.spec_from_file_location(
        "generate_phase12_configs",
        repo / "scripts" / "generate_phase12_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-23, the paired verdicts land and Phase 12 closes
# --------------------------------------------------------------------------


def test_the_paired_verdicts_confirm_the_prediction_and_read_two_fires():
    record = phase12.PAIRED_OBSERVED
    verdicts = record["verdicts"]
    # Every margin matches the derived prediction exactly.
    derived = {
        p["key"]: p["recorded"] for p in phase12.paired_claim_pairs("p12")
    }
    assert verdicts["b_vs_a"]["margin"] == derived[
        "p12__basal_vs_frontal_bar"
    ]["margin"]
    assert verdicts["c_vs_a"]["margin"] == derived[
        "p12__concat_vs_frontal_bar"
    ]["margin"]
    assert verdicts["c_vs_d"]["margin"] == derived[
        "p12__concat_vs_capacity"
    ]["margin"]
    # 0/5 everywhere; condition 1 false on all three; only B-vs-A passes
    # condition 2, so only it is WITHDRAWN rather than unresolved.
    for cell in verdicts.values():
        assert cell["excludes_zero"] == "0 of 5"
        assert cell["condition_1"] is False
    assert verdicts["b_vs_a"]["condition_2"] is True
    assert verdicts["b_vs_a"]["verdict"] == "WITHDRAWN"
    assert verdicts["c_vs_a"]["condition_2"] is False
    assert verdicts["c_vs_d"]["condition_2"] is False

    # The honest split: the arithmetic was derived, the intervals were
    # what the run alone could answer -- and they came back stronger.
    confirmed = record["prediction_confirmed"]
    assert "arithmetic on recorded" in confirmed
    assert "0/5 everywhere" in confirmed
    assert "wider than the" in confirmed
    # Reading 2 verbatim; the sentences stay unused by their own clause.
    assert "not claimable" in record["reading_2_fired_verbatim"]
    assert "UNUSED" in record["reading_2_fired_verbatim"]
    # The sharpest finding: the sixth arrival, direction never in doubt.
    sharpest = record["sharpest_finding"]
    assert "even B-vs-A is WITHDRAWN" in sharpest
    assert "DIRECTION NOBODY DOUBTS" in sharpest
    assert "SIXTH arrival" in sharpest
    assert "+0.1601" in record["caveat"]


def test_the_three_p12_rows_are_ledgered_and_earlier_pins_hold():
    from cleft import results_ledger

    results_ledger.validate()
    assert len(results_ledger.ENTRIES) >= 31
    # Appending leaves every earlier prefix checksum exactly as it was.
    assert results_ledger.cumulative_checksum(28) == (
        "5fa5962f0858b76e847de73ae2e7d24912472b3717ed9b022d097ad8e2c4892f"
    )
    assert results_ledger.cumulative_checksum(31) == (
        "c96084c2841c10061fad2f927a1324ca1fa7c442e71b78a2aea6e0787af2995d"
    )

    b_row = results_ledger.ENTRIES[28]
    c_a_row = results_ledger.ENTRIES[29]
    c_d_row = results_ledger.ENTRIES[30]
    assert b_row["id"] == "p12-basal-vs-frontal-bar-withdrawn"
    assert b_row["status"] == "WITHDRAWN"
    assert c_a_row["id"] == "p12-concat-vs-frontal-bar-unresolved"
    assert c_d_row["id"] == "p12-concat-vs-capacity-unresolved"
    for row in (c_a_row, c_d_row):
        assert row["status"] == "UNRESOLVED-WITHDRAWN"
    for row in (b_row, c_a_row, c_d_row):
        assert row["phase"] == "p12"
        assert "0 of 5" in row["condition_1"]
        # The confound caveat travels on every row, by name and figure.
        assert any("+0.1601" in c for c in row["caveats"]), row["id"]

    # The sharpest sentence lives in the claim, not a footnote.
    assert "SIXTH" in b_row["claim"]
    assert "DIRECTION NOBODY DOUBTS" in b_row["claim"]
    assert any("cannot resolve even a gap" in c for c in b_row["caveats"])
    # C-vs-A carries reading 2 and keeps the sentences unused.
    assert "fired\nverbatim" in c_a_row["claim"] or "verbatim" in c_a_row["claim"]
    assert any("UNUSED" in c for c in c_a_row["caveats"])
    assert any("stays dead" in c for c in c_a_row["caveats"])
    # C-vs-D names what the control controlled for.
    assert any("control did its job" in c for c in c_d_row["caveats"])
    assert any("travel together" in c for c in c_d_row["caveats"])


def test_phase_12_closes_on_the_four_stops_and_carries_five_by_name():
    record = phase12.PHASE_12_CLOSING
    assert record["closed"] == "2026-08-23"
    assert "descriptively yes, claimably no" in record["answer"]
    assert "UNSUPPORTED" in record["answer"]

    stops = record["stops_walked"]
    assert len(stops) == 5
    assert all(v.startswith("MET") for v in stops.values())
    assert "143 and 238 exactly" in stops["1_manifest"]
    assert "REOPEN fired" in stops["2_staging"]
    assert "REC 13/SW/0064 and 23/YH/0037" in stops["2_staging"]
    assert "guard-after-mkdir" in stops["3_arms"]
    assert "+0.1601" in stops["3_arms"]
    assert "reading 2" in stops["4_paired"] or "reading 2" in (
        stops["4_paired"].lower()
    )

    assert "none claimable" in record["no_new_claims"]
    carried = record["carried_forward_open"]
    assert len(carried) == 5
    for name in (
        "supervision question 8", "r(AR, mean) = +0.1601",
        "whole-image-G1-only", "RETRY_LIMIT_IS_NOT_HOLDING",
        "PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE",
    ):
        assert any(name in item for item in carried), name
    # Question 8's carry states precisely what still depends on it.
    q8 = next(item for item in carried if "question 8" in item)
    assert "WORDING" in q8 and "banked unused" in q8
    assert "Phase 13" in record["next"]

    # Every carried record resolves, so a carried name cannot rot.
    from cleft import phase8, phase9

    assert phase12.STOP_2_STAGING_DECISION["boundary"]
    assert phase12.BASAL_CONFOUND_OBSERVED["r_aspect_ratio_vs_mean"] == 0.1601
    assert phase8.RETRY_LIMIT_IS_NOT_HOLDING["status"].startswith("OPEN")
    assert phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE


# --------------------------------------------------------------------------
# 2026-08-23, the second sequence amendment
# --------------------------------------------------------------------------


def test_the_second_sequence_amendment_is_dated_and_leaves_the_first_visible():
    from cleft import phase8, phase11

    record = phase12.PHASE_SEQUENCE_RENUMBERED_2
    assert record["decided"].endswith("the second sequence amendment")
    assert "PHASE_SEQUENCE_RENUMBERED" in record["first_amendment"]

    # 13 is unchanged across the two amendments; the write-up moves to 17
    # and appears exactly once in the new sequence.
    first = phase11.PHASE_SEQUENCE_RENUMBERED
    assert first["becomes"]["13"] == "decoder/reconstruction"
    assert record["becomes"]["13"].startswith("decoder/reconstruction")
    assert record["was"]["14"] == "write-up"
    writeups = [k for k, v in record["becomes"].items() if v == "write-up"]
    assert writeups == ["17"]
    # The decoder's registered warning travels with its slot.
    assert "not scoring like a clinician" in record["becomes"]["13"]
    # Phase 15's exit-criteria note is registered NOW, with its consumer.
    assert "comparative verdict against\nSCUT" in record["becomes"]["15"] or (
        "comparative verdict against" in record["becomes"]["15"]
    )
    assert "measured\nanswer, not a vibe" in record["becomes"]["15"] or (
        "not a vibe" in record["becomes"]["15"]
    )
    assert "Rosero" in record["becomes"]["16"]

    # Three status changes, each dated to a phase.
    changes = record["status_changes"]
    assert changes["tstr"].endswith("(Phase 16)")
    assert changes["ldl"].endswith("(Phase 14)")
    assert changes["roadb_branch_2"].endswith("(Phase 15)")

    # The LDL scheduling reckons with the ladder's measured negative --
    # the figures are the ladder's own, checked against its record.
    from cleft import ladder

    reckon = record["ldl_prior_measurement_to_reckon_with"]
    assert "0.2336" in reckon and "0.2520" in reckon
    assert "LDL does\nnot help" in reckon or "does" in reckon
    assert "EXPLICITLY" in reckon
    assert "Scheduling is not reopening" in reckon
    ladder_source = __import__("inspect").getsource(ladder)
    assert "LDL does not help" in ladder_source
    assert "0.2336" in ladder_source

    # The first amendment carries the dated pointer forward; the parked
    # records stay as written with their own dated pointers.
    assert "PHASE_SEQUENCE_RENUMBERED_2" in first["second_amendment"]
    assert phase8.PHASE_8_COMPLETE is not None or True  # module import guard
    phase8_source = __import__("inspect").getsource(phase8)
    assert "stays parked, unchanged" in phase8_source
    assert "road_b_branch_2_unparked" in phase8_source
    assert record["nothing_built"].startswith("Phase 13")


# --------------------------------------------------------------------------
# 2026-08-23, the three primaries banked
# --------------------------------------------------------------------------


def test_the_primaries_are_banked_with_the_label_correction_first():
    record = phase12.PRIMARY_SOURCES_BANKED
    assert len(record["sources"]) == 3
    assert "NotebookLM -> [LITERATURE-primary]" in record["tag_upgrade"]

    # (1) The label correction: composite -> derived single overall
    # score, with the PLAN's wording preserved at its cited lines.
    label = record["label_correction"]
    assert "PLAN.md:903, 913, 1512" in label["was"]
    assert "SINGLE OVERALL SCORE" in label["becomes"]
    assert "FOUR COMPONENTS" in label["confirmed"]
    assert "TRIED AND REJECTED" in label["confirmed"]
    assert "SEVENTH quantity-under-one-name" in label["error_class"]
    assert "preserved at its cited" in label["error_class"]

    # (2)-(3) The departure and the anchors carry the primary tag and
    # the figures they rest on.
    assert "SIX ORTHODONTISTS" in record["panel_departure"]["theirs"]
    assert "0.4696" in record["panel_departure"]["reading"]
    assert "not a defect" in record["panel_departure"]["reading"]
    anchors = record["reliability_anchors"]
    assert "0.43-0.60" in anchors["figures"]
    assert "NO new coefficients" in anchors["figures"]
    assert "founding papers" in anchors["reading"]

    # (4) Both halves of the ceiling's provenance, and the trap.
    ceiling = record["ceiling_provenance"]
    assert "lineage's OWN machinery" in ceiling["the_lineages_half"]
    assert "Fleiss 1986" in ceiling["the_lineages_half"]
    assert "APPLIED ON TOP" in ceiling["our_half"]
    assert "0.90/0.902" in ceiling["trap_registered"]
    assert "0.9032" in ceiling["trap_registered"]
    assert "NEVER placed in proximity" in ceiling["trap_registered"]
    # The trap's two figures really are different quantities in the code.
    from cleft.data import reliability

    assert reliability.PCC_CEILING_237 == 0.9032
    assert reliability.RELIABILITY_237 == 0.8158

    # (5) The distribution match cites both sides' numbers.
    match = record["distribution_match"]
    assert "2.8-3.4" in match["theirs"]
    assert "5/89/110/30/3" in match["ours"]
    assert "lineage-typical" in match["reading"]

    # (6) Basal absent at source -- and question 8 explicitly survives.
    basal = record["basal_absent_at_source"]
    assert "ALL THREE papers" in basal["measured_at_source"]
    assert "card overlays" in basal["measured_at_source"]
    assert "FULL PRIMARY SUPPORT" in basal["consequence"]
    assert "ENTIRELY OUTSIDE THE INSTRUMENT" in basal["consequence"]
    assert "stays\nopen" in basal["consequence"] or "stays" in basal["consequence"]
    assert "not our\npanel's practice" in basal["consequence"] or (
        "panel's practice" in basal["consequence"]
    )

    # (7) The ancestral citation is banked BESIDE the arrivals, not as one.
    ancestor = record["resolution_findings_ancestor"]
    assert "statistically significant differences" in ancestor["their_chain"]
    assert "1992 vocabulary" in ancestor["reading"]
    assert "not as a\nseventh arrival" in ancestor["reading"] or (
        "not as a" in ancestor["reading"]
    )


def test_the_in_place_upgrades_landed_without_touching_the_originals():
    from cleft import ladder
    from cleft.data import reliability as reliability_module
    import inspect

    # The ladder's [LITERATURE] half is upgraded, dated, in place -- and
    # both the original field and the UNRESOLVED item stand untouched.
    record = ladder.BASAL_RATIONALE_UNSUPPORTED
    upgraded = record["literature_upgraded_to_primary"]
    assert upgraded.startswith("2026-08-23")
    assert "held\nat source" in upgraded or "held" in upgraded
    assert "PRIMARY_SOURCES_BANKED" in upgraded
    assert "not what our panel was shown" in upgraded
    assert "NotebookLM" in record["literature"]  # the original, preserved
    assert "UNSUPPORTED" in record["unresolved"]

    # reliability.py carries the provenance split beside its constant.
    source = inspect.getsource(reliability_module)
    assert "LINEAGE'S own" in source
    assert "applied on top" in source
    assert "coincidentally adjacent" in source

    # (8) The LDL reckoning is accepted, dated, and the pinned figures
    # stand in the field it accepts.
    sequence = phase12.PHASE_SEQUENCE_RENUMBERED_2
    assert sequence["ldl_reckoning_accepted"].startswith("2026-08-23")
    assert "pinned\nfigures stand" in sequence["ldl_reckoning_accepted"] or (
        "figures stand" in sequence["ldl_reckoning_accepted"]
    )
    assert "0.2336" in sequence["ldl_prior_measurement_to_reckon_with"]


# --------------------------------------------------------------------------
# 2026-08-23, the guard-after-mkdir defect and its regression test
# --------------------------------------------------------------------------


def test_the_guard_after_mkdir_defect_is_recorded_with_its_fix_side():
    record = phase12.EXTRACT_GUARD_AFTER_MKDIR
    assert record["defect"].endswith("fixed")
    assert "attempts 0-6" in record["run"]
    assert "created by attempt 0 itself" in record["evidence"]
    assert "empty directory its own process just made" in record["mechanism"]
    # The fix is on the task side; the guard keeps its teeth, with the
    # husk argument by name.
    assert record["fix_side"].startswith("the TASK")
    assert "guard is untouched" in record["fix_side"]
    assert "husk" in record["fix_side"]
    assert record["same_pattern_sweep"].startswith("CLEAN")
    assert "+7" in record["retry_item"]
    assert "embeddings_basal_v1" in record["cleanup"]
    assert "run\ndirectories are ordinary voids" in record["cleanup"] or (
        "ordinary voids" in record["cleanup"]
    )

    # The fixed task neither pre-checks nor creates the save path.
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_extract_basal_embeddings)
    assert "out_dir.mkdir" not in source
    assert "out_dir.exists()" not in source
    assert "EXTRACT_GUARD_AFTER_MKDIR" in source
    # embeddings.save still owns both, atomically.
    save_source = inspect.getsource(__import__(
        "cleft.embeddings", fromlist=["save"]
    ).save)
    assert "directory.exists()" in save_source
    assert "directory.mkdir(parents=True)" in save_source


def _views_artifacts(root, n=6):
    """A views-shaped manifest + basal staged artifact, small and synthetic."""
    import numpy as np

    manifest_dir = root / "cleft_v1_views"
    staged_dir = root / "staged_basal_v1"
    manifest_dir.mkdir(parents=True)
    staged_dir.mkdir(parents=True)

    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    rng = __import__("numpy").random.default_rng(0)
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    geometry = [
        "# CLUSTER-ONLY: patient-keyed geometry",
        "patient_id,basal_id,aspect_ratio,pad_fraction,corner_white_fraction",
    ]
    for index in range(n):
        value = float(np.clip(rng.normal(2.7, 0.7), 1.0, 5.0))
        lines.append(
            f"{index + 1},{1000 + index},{2000 + index},{value:.4f},"
            f"{value:.4f},{value:.4f},{value:.4f},{value:.4f},"
            f"0.2,0.2,0.2,0.2,0.2,{index % 3},{index % 5}"
        )
        geometry.append(
            f"{index + 1},{2000 + index},{0.9 + 0.1 * (index % 3):.3f},"
            f"0.15,{0.4 + 0.05 * index:.3f}"
        )
    (manifest_dir / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (staged_dir / "geometry.csv").write_text(
        "\n".join(geometry) + "\n", encoding="utf-8"
    )
    np.save(
        staged_dir / "staged_patient_g1.npy",
        np.zeros((n, 8, 8, 3), dtype=np.uint8),
    )
    return manifest_dir, staged_dir


def test_the_extraction_runs_once_and_refuses_twice_from_the_save_layer(
    tmp_path, clean_repo
):
    """The registered regression: first run succeeds against a clean
    tree; the second REFUSES, and the refusal fires from the SAVE layer
    on the artifact -- never from task scaffolding. The backbone is
    stubbed at the extraction call because the test's subject is the
    directory lifecycle, not ViT-B/16."""
    import contextlib
    import io as io_module

    import numpy as np

    from fixtures import builders
    import cleft.provenance.context as context_module
    from cleft import embeddings as embeddings_module
    from cleft.run import main
    from cleft.train import extract as extract_module
    from test_extract import declare

    manifest_dir, staged_dir = _views_artifacts(tmp_path, n=6)
    config = builders.write_config(
        tmp_path / "extract_basal.yaml",
        phase="p12",
        inputs=[
            declare("manifest_views", manifest_dir),
            declare("staged_basal", staged_dir),
        ],
        task={
            "kind": "extract_basal_embeddings",
            "manifest_artifact": "manifest_views",
            "staged_artifact": "staged_basal",
            "backbone": "vit_b16",
            "init": "imagenet",
            "geometry": "g1",
            "out_version": "embeddings_basal_v1",
            "expect_patients": 6,
        },
    )

    real_extract = extract_module.extract_features
    real_root = context_module.default_repo_root
    extract_module.extract_features = lambda backbone, init, images, **kw: (
        np.arange(len(images) * 4, dtype=np.float32).reshape(len(images), 4),
        {"backbone": backbone, "init": init, "stubbed_for": "lifecycle test"},
    )
    context_module.default_repo_root = lambda: clean_repo
    try:
        with contextlib.redirect_stdout(io_module.StringIO()):
            main(["--config", str(config), "--out", str(tmp_path / "runs1")])
            artifact = (
                clean_repo / "data" / "embeddings" / "embeddings_basal_v1"
                / "vit_b16__imagenet__g1"
            )
            # The artifact exists and is not a husk.
            assert (artifact / "values.npy").is_file()
            assert (artifact / "metadata.json").is_file()

            # Second run: refused BY THE SAVE LAYER, on the artifact.
            with pytest.raises(
                embeddings_module.EmbeddingError, match="already exists"
            ):
                main([
                    "--config", str(config), "--out", str(tmp_path / "runs2"),
                ])
    finally:
        extract_module.extract_features = real_extract
        context_module.default_repo_root = real_root


def test_the_task_derives_and_refuses_to_overwrite():
    import inspect

    from cleft import run as run_module

    assert "build_views_manifest" in run_module.TASKS
    source = inspect.getsource(run_module.task_build_views_manifest)
    # The derivation and its invariants live in data/views.py -- the task
    # wires, it does not re-implement.
    assert "views.derive(rows)" in source
    assert "views.pairing_summary" in source
    assert "views.views_manifest_schema()" in source
    # Config and code must agree on the cohort; neither wins silently.
    assert "views.VIEWS_COHORT" in source
    assert "silently preferred" in source
    # Artifact discipline: never overwrite a version.
    assert "already exists" in source
    assert "NEVER overwrite" in source
    # Same tier marker and columns as cleft_v1.
    assert "CLUSTER-ONLY: patient-keyed" in source
    assert "MANIFEST_COLUMNS" in source


# --------------------------------------------------------------------------
# [2026-08-31] The two-view claim, contradicted by the primary manuscripts
# --------------------------------------------------------------------------


def _flat_p12(text: str) -> str:
    return " ".join(text.split())


#: Every quote attributed to a source document, keyed to where it must
#: appear in the record. Asserted VERBATIM -- the instruction was "do not
#: paraphrase", and a test that matched loosely would not enforce it.
DOCUMENTED_QUOTES = {
    ("bruce_ready_draft", "data_source_ethics_and_image_preprocessing"): (
        "The primary image dataset comprised 181 standardised frontal "
        "two-dimensional facial photographs of five-year-old children who "
        "had undergone primary surgical repair for cUCL."
    ),
    ("bruce_ready_draft", "the_crop"): (
        "The resulting cropped Frontal-Eye-View (FEV) included the medial "
        "canthi, nose and upper lip region."
    ),
    ("bruce_ready_draft", "human_assessor_panel"): (
        "Each cropped image was independently scored by a multidisciplinary "
        "panel of five cleft professionals... Assessors scored postoperative "
        "facial appearance using a five-point VAS."
    ),
    ("latest_version_draft", "figure_2_legend"): (
        "Expert cleft professionals evaluate 2D cropped, Frontal eye View "
        "(FEV) photographs (depicting the post-operative state) and each "
        "assign a ground-truth Visual Analogue Score (VAS)."
    ),
    ("bmj_open", "facial_appearance"): (
        "A two-dimensional assessment of the child's face was made using "
        "frontal photographs... The images were anonymised and cropped to "
        "allow unbiased assessment of only the nose and lip area."
    ),
    ("bakaki_thesis", "section_4_3_1"): (
        "the facial images partially only reveal the nose and mouth/lips."
    ),
}


def test_every_documented_quote_is_present_verbatim():
    """The instruction was 'verify each quote's presence; do not
    paraphrase'. This is that verification."""
    record = phase12.RATING_PROCEDURE_DOCUMENTED
    for (section, key), quote in DOCUMENTED_QUOTES.items():
        assert section in record, section
        assert key in record[section], f"{section}.{key}"
        assert _flat_p12(record[section][key]) == _flat_p12(quote), (
            f"{section}.{key} does not match the source quote verbatim"
        )

    # Each source document is NAMED beside its quotes, not just cited.
    assert record["bruce_ready_draft"]["document"] == (
        "Bruce-ready version - minus Jonathan's edits.docx"
    )
    assert record["latest_version_draft"]["document"] == (
        "The latest version - mid re-write post Physicists comments "
        "+ BR tracked changes.docx"
    )
    assert record["bmj_open"]["document"] == "bmjopen-15-8.pdf"
    assert record["bakaki_thesis"]["document"] == (
        "Paul_Bakaki_Final_Thesis (1).pdf"
    )

    # The tag, exactly as instructed.
    assert record["tag"] == (
        "[LITERATURE-primary, via NotebookLM against the supervisor's own "
        "source documents, the maintainer 2026-08-31]"
    )
    finding = record["the_finding"]
    assert "ONE CROPPED FRONTAL-EYE-VIEW" in finding
    for view in ("BASAL", "SUBMENTAL", "PROFILE", "LATERAL"):
        assert view in finding, view


def test_the_docx_quotes_were_verified_at_source_and_the_pdfs_were_not():
    """The record must not present the maintainer's NotebookLM reading and this
    session's own extraction as one thing."""
    verified = phase12.RATING_PROCEDURE_DOCUMENTED[
        "verified_at_source_2026_08_31"
    ]
    assert "independently of the NotebookLM route" in verified["by"]
    assert "ALL SEVEN" in verified["docx_quotes_found"]

    # The negative that no quote can carry, with its one honest exception.
    negative = _flat_p12(verified["the_negative_no_quote_can_carry"])
    assert "appear ZERO times" in negative
    assert "'basal' appears exactly ONCE" in negative
    assert "author surname 'Basalamah, A.', not a view" in negative

    # And the unverified half is named as unverified.
    not_verified = _flat_p12(verified["not_verified_here"])
    assert "bmjopen-15-8.pdf" in not_verified
    assert "Paul_Bakaki_Final_Thesis (1).pdf" in not_verified
    assert "rest on an off-machine NotebookLM verification alone" in not_verified


def test_the_rationale_is_upgraded_to_contradicted_with_the_original_intact():
    from cleft import ladder

    record = ladder.BASAL_RATIONALE_UNSUPPORTED

    # The 2026-08-23 reasoning is preserved, word for word.
    assert record["downgrade"].startswith("MEASURED rationale -> UNSUPPORTED")
    assert "NOT refuted" in record["downgrade"]
    assert "PLAN 4.6, tagged [MEASURED]" in record["original_claim_preserved"]
    assert "251 rows" in record["measured"]
    assert "IDs 241-742" in record["measured"]
    assert "Asher-McDade masks frontal and LATERAL" in record["literature"]

    upgrade = _flat_p12(record["upgraded_to_contradicted_2026_08_31"])
    assert "UNSUPPORTED -> CONTRADICTED" in upgrade
    assert "ONE CROPPED FRONTAL-EYE-VIEW" in upgrade
    assert "phase12.RATING_PROCEDURE_DOCUMENTED" in upgrade

    # WHY it could not have been settled before -- the distinction the
    # 2026-08-23 record was careful about.
    why = _flat_p12(record["what_changed_and_why_it_could_not_have_before"])
    assert "measured the ARTIFACT" in why
    assert "describe the PROCEDURE" in why
    assert "NOTHING DISTINGUISHED" in why

    # Both bodies of evidence stand together; neither is superseded.
    together = _flat_p12(record["both_bodies_of_evidence_stand_together"])
    assert "NOT superseded" in together
    for piece in ("251 rows keyed to frontal IDs",
                  "folder 143's absent basal 523",
                  "2 coincidental adjacent-ID pairs",
                  "frontal and LATERAL"):
        assert piece in together, piece

    # The old "unresolved" item is marked resolved, not deleted.
    assert record["unresolved"].startswith("[RESOLVED 2026-08-31")
    assert "nothing in the sheet could distinguish" in record["unresolved"]

    # And the linked inference is VOID, not merely unrelied-on.
    linked = _flat_p12(record["linked_item_corrected"])
    assert "VOID, not merely unrelied-on" in linked
    assert "docs/PLAN.md line 865) was never touched" in linked


def test_the_plan_carries_both_dated_corrections_with_originals_visible():
    """The governing document is where the claim entered, so it is where
    the correction has to land."""
    plan = (Path(__file__).resolve().parents[1] / "docs" / "PLAN.md").read_text(
        encoding="utf-8"
    )

    # Both originals still visible, unedited.
    assert (
        "**[MEASURED]** Two views carry the label — raters saw frontal "
        "and basal together." in plan
    )
    assert (
        "**[REASONED]** that the missing basal view explains the r≈0.3 "
        "plateau." in plan
    )

    # Both corrections present, dated, in the document's own convention.
    assert plan.count("[CORRECTED 2026-08-31]") == 2
    assert "The line above is wrong, and its tag was wrong before that." in plan
    assert "The plateau inference above is VOID, not merely unsupported." in plan

    # Each correction points at its governing record.
    for pointer in ("ladder.BASAL_RATIONALE_UNSUPPORTED",
                    "phase12.RATING_PROCEDURE_DOCUMENTED",
                    "phase12.SENTENCE_SELECTED",
                    "phase12.TWO_VIEW_CLAIM_PROVENANCE"):
        assert pointer in plan, pointer

    # The correction follows its line, not the other way round.
    assert plan.index("Two views carry the label") < plan.index(
        "The line above is wrong"
    )
    assert plan.index("explains the r≈0.3 plateau") < plan.index(
        "The plateau inference above is VOID"
    )
    # And it says plainly that no number moved.
    assert "**Nothing measured changes.**" in plan


def test_the_prewritten_sentence_is_selected_without_amending_either():
    record = phase12.SENTENCE_SELECTED
    assert record["which"] == "if_raters_saw_frontal_only"

    # The selected text IS the registered sentence, by comparison not by
    # retype -- a selection that quietly reworded would be an amendment.
    registered = phase12.INTERPRETATION_SENTENCES_PREWRITTEN
    assert _flat_p12(record["the_selected_sentence"]) == _flat_p12(
        registered["if_raters_saw_frontal_only"]
    )
    # BOTH sentences survive, and the unselected one is untouched.
    assert "adding the basal view RESTORES evidence" in registered[
        "if_raters_saw_both"
    ]
    assert registered["neither_is_chosen_here"].startswith(
        "the supervision answer to ask-list question 8 selects the sentence"
    )
    assert registered["selected_2026_08_31"] == "SENTENCE_SELECTED"

    # Nothing measured changes -- the registration's own promise, kept.
    nothing = _flat_p12(record["nothing_measured_changes"])
    assert "the arms and every number are identical under both" in nothing
    assert "No arm was re-run, no figure moves, no ledger row is touched" in nothing

    # Arm C's gain reads as unresolved, against figures live at source.
    declared = phase12.PAIRED_OBSERVED["arms_as_declared"]
    assert declared["a"]["mean"] == 0.2505
    assert declared["c"]["mean"] == 0.2640
    assert phase12.ARMS_B_C_OBSERVED["c_two_view_concat"]["mean"] == 0.2640
    gain = round(declared["c"]["mean"] - declared["a"]["mean"], 4)
    assert gain == 0.0135

    means = _flat_p12(record["what_it_means_for_arm_c"])
    assert "0.2640 vs 0.2505, +0.0135" in means
    assert "ALREADY inside the resolution floor" in means
    assert "MARGINALLY OR NOT AT ALL, UNRESOLVED" in means

    # The gain really is below anything this cohort has resolved.
    from cleft import ladder

    assert gain < ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"]
    assert gain < 0.04, "and below the band measured UNRESOLVABLE"

    # The reading that fired is unchanged.
    assert "2_descriptive_only" in record["the_fired_reading_is_unchanged"]


def test_supervisor_question_8_is_answered_but_not_closed():
    record = phase12.SUPERVISOR_QUESTION_8_DOCUMENTARILY_ANSWERED
    assert "NOT CLOSED" in record["status"]
    assert "PENDING IN-PERSON CONFIRMATION" in record["status"]

    # The question is carried as asked, from its own home.
    from cleft import ladder

    assert "were they shown the frontal photograph only" in _flat_p12(
        ladder.BASAL_RATIONALE_UNSUPPORTED["for_supervisor"]
    )
    assert "were they shown the frontal photograph only" in _flat_p12(
        record["the_question_as_asked"]
    )
    assert "DOCUMENTARILY ANSWERED" in ladder.BASAL_RATIONALE_UNSUPPORTED[
        "for_supervisor_2026_08_31"
    ]

    # The reason it stays open is substantive, not deferential.
    why = _flat_p12(record["why_it_is_not_closed_outright"])
    assert "describe the procedure their authors WROTE DOWN" in why
    assert "A DOCUMENT IS EVIDENCE ABOUT A PROCEDURE, NOT THE PROCEDURE" in why
    # It becomes a confirmation, with the quotes attached.
    assert "CONFIRMATION rather than a request to recall unaided" in _flat_p12(
        record["what_to_put_to_supervision"]
    )
    # And either answer moves nothing measured.
    assert "nothing measured" in record["what_turns_on_the_answer"]
    assert "STILL identical" in _flat_p12(record["what_turns_on_the_answer"])


def test_the_two_view_error_provenance_is_recorded_like_the_others():
    record = phase12.TWO_VIEW_CLAIM_PROVENANCE
    entered = _flat_p12(record["how_it_entered"])
    assert "[MEASURED]" in entered
    assert "GOVERNING DOCUMENT" in entered
    assert "NO SOURCE" in entered
    assert "NEVER MEASURED" in entered

    survived = _flat_p12(record["how_long_it_survived"])
    assert "EIGHT DAYS" in survived
    assert "the PLAN line was never corrected" in survived
    assert "this session" in _flat_p12(record["it_was_repeated"])
    assert "the challenge was it" in _flat_p12(record["how_it_was_caught"])

    # Sixth instance, and its own lesson turned on itself.
    lesson = _flat_p12(record["its_own_lesson_applied_to_itself"])
    assert "SIXTH instance" in lesson
    assert "precisely the claim nobody audits" in lesson
    from cleft import ladder

    assert "precisely the claim nobody audits" in _flat_p12(
        ladder.BASAL_RATIONALE_UNSUPPORTED["lesson"]
    )
    assert ladder.BASAL_RATIONALE_UNSUPPORTED["error_class"]["instance"] == (
        "the SIXTH of its kind"
    )
    # The second lesson, which is the one this cycle actually added.
    second = _flat_p12(record["the_second_lesson"])
    assert "does not reach the governing document is not finished" in second
    assert "code > this document > memory" in second


def test_no_number_moved_and_no_ledger_row_was_added():
    """The whole correction is interpretive. This is the guard."""
    from cleft import ladder, results_ledger

    arms = phase12.ARMS_B_C_OBSERVED
    assert arms["b_basal_only_236"]["mean"] == 0.1880
    assert arms["c_two_view_concat"]["mean"] == 0.2640
    assert arms["b_basal_only_236"]["per_seed"] == (
        0.1926, 0.2163, 0.2307, 0.1341, 0.1662
    )
    assert arms["c_two_view_concat"]["per_seed"] == (
        0.2736, 0.2818, 0.3028, 0.2285, 0.2333
    )
    assert phase12.PAIRED_OBSERVED["arms_as_declared"] == {
        "a": {"mean": 0.2505, "sd": 0.0077},
        "b": {"mean": 0.1880, "sd": 0.0388},
        "c": {"mean": 0.2640, "sd": 0.0321},
        "d": {"mean": 0.2534, "sd": 0.0105},
    }
    # [2026-09-06] Was `len(ENTRIES) == 38`, a global count standing in
    # for "THIS CORRECTION added no ledger row". The first rewrite read
    # it as "Phase 12 has no rows", which is FALSE: the phase has three
    # (p12-basal-vs-frontal-bar-withdrawn and the two unresolved
    # concat contrasts), all dated 2026-08-23 and all predating this
    # correction. The relation the test means is that the phase's rows
    # are exactly those three and the correction added none.
    p12_rows = [e for e in results_ledger.ENTRIES if e["phase"] == "p12"]
    assert [e["id"] for e in p12_rows] == [
        "p12-basal-vs-frontal-bar-withdrawn",
        "p12-concat-vs-frontal-bar-unresolved",
        "p12-concat-vs-capacity-unresolved",
    ]
    assert {e["date"] for e in p12_rows} == {"2026-08-23"}
    assert results_ledger.validate() is None
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"] == 0.1386
