"""Record-versus-artifact agreement as a standing invariant.

The suite owns everything except the artifact read: the gather from the
records, its coverage, and the comparison's teeth. The read itself is a
declared job (``record_audit.STANDING_CHECK_DESIGN``) because the
prediction CSVs are CLUSTER-ONLY.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cleft import phase18, record_audit, roadb
from cleft.config.schema import ConfigError, validate
from cleft.train import graph_cleft

REPO = Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# Stage E's two banked values
# --------------------------------------------------------------------------


def test_stage_e_per_arm_banks_the_two_missing_cells():
    record = graph_cleft.STAGE_E_PER_ARM
    assert record["arms"] == {
        "p7_e_srgnn_scut_masked_g2_grid": {
            "pcc": 0.0889, "sd": 0.0224, "n_seeds": 10,
        },
        "p7_e_srgnn_scut_masked_g2_anatomy": {
            "pcc": 0.0554, "sd": 0.0251, "n_seeds": 10,
        },
    }
    # Sourced from run 5's recomputation, NOT a re-run, and tagged.
    provenance = _flat(record["provenance"])
    assert provenance.startswith("[REPORTED]")
    assert "p18_metric_space__9411267e__p18-metric-space-5" in provenance
    assert "NO arm was re-run" in provenance
    assert "CLUSTER-ONLY" in provenance
    # Descriptive: no ledger row from THIS banking.
    # [2026-08-31] Was `len(ENTRIES) == 37`, which asserted a global
    # count to mean "this change banked nothing" -- an absolute-state
    # instrument that goes stale the moment anything else appends, and
    # it did (ENTRIES[37], the condition-split correction). Rewritten as
    # the RELATION it was always trying to express: no ledger entry
    # references these arms. True at 37, at 38, and at any future size.
    from cleft import results_ledger

    for arm in record["arms"]:
        for entry in results_ledger.ENTRIES:
            assert arm not in str(entry), (arm, entry["id"])


def test_the_stage_e_values_reconcile_both_banked_ranges():
    """The check that makes the two values safe to bank: they reproduce
    two independently-banked ranges to four decimals, and anatomy is the
    low endpoint the docstring has quoted since 2026-08-01."""
    banked = graph_cleft.PRETRAINING_DOES_NOT_PREDICT_TRANSFER
    arms = graph_cleft.STAGE_E_PER_ARM["arms"]

    stage_e = {
        "native": 0.1507,   # Stage D's masked-G2 srgnn cell
        "grid": arms["p7_e_srgnn_scut_masked_g2_grid"]["pcc"],
        "anatomy": arms["p7_e_srgnn_scut_masked_g2_anatomy"]["pcc"],
        "random": graph_cleft.STAGE_E_COMPLETE["mean"],
    }
    spread = max(stage_e.values()) - min(stage_e.values())
    assert spread == pytest.approx(
        banked["cleft_range_with_those_checkpoints"], abs=5e-5
    )
    # Anatomy IS the low endpoint.
    assert min(stage_e, key=stage_e.get) == "anatomy"
    assert stage_e["anatomy"] == 0.0554

    # And the E0 range reproduces from the scheme means, unchanged.
    e0 = dict(graph_cleft.SCHEME_AXIS_AT_CLEFT["means"])
    assert max(e0.values()) - min(e0.values()) == pytest.approx(
        banked["cleft_range_with_one_representation"], abs=5e-5
    )
    # The pointer from the range record to the per-arm one.
    assert banked["endpoints_now_traceable_2026_08_31"] == "STAGE_E_PER_ARM"


# --------------------------------------------------------------------------
# the gather
# --------------------------------------------------------------------------


def test_the_gather_covers_all_sixty_eight_locked_arms():
    """[2026-08-31] Was 63 of 68 while the p16/p17 values lived only in
    prose; phase16.ARM_MEANS and phase17.ARM_MEANS closed that gap the
    same day, so coverage is now complete."""
    banked = record_audit.banked_arm_pcc()
    locked = {entry["name"] for entry in phase18.locked_arm_entries()}
    assert set(banked) == locked
    assert len(banked) == 68

    resolved = {n for n, c in banked.items() if c["pcc"] is not None}
    assert len(resolved) == 68, "coverage is 68 of 68"
    # The gap set is EXACT and empty -- an arm losing its banked value
    # fails here rather than quietly dropping out of the comparison.
    assert record_audit.unresolved(banked) == []
    assert record_audit.BANKED_VALUE_GAPS == ()
    assert record_audit.BANKED_VALUE_GAP_REASONS == {}


def test_the_closed_gap_keeps_its_history():
    """An empty list with no memory reads as though coverage was always
    complete. It was not, for one day."""
    record = record_audit.BANKED_VALUE_GAPS_CLOSED
    assert record["opened"].startswith("2026-08-31")
    assert record["closed"].startswith("2026-08-31")
    assert record["coverage_now"] == "68 of 68"
    was = _flat(record["was"])
    for arm in ("p16_anchor_loop", "p16_identity_baseline", "p17_arm_a",
                "p17_arm_b", "p17_arm_c"):
        assert arm in was
    assert "only in prose" in was
    assert "NOT regex-parsed" in _flat(record["closed_by"])
    assert "prose unedited" in _flat(record["closed_by"])
    assert "fails loudly" in _flat(record["why_the_empty_tuple_stays"])


def test_the_five_banked_constants_cross_check_against_the_ledger():
    """The transcription is verified, not trusted: each PCC reproduces
    the ledger's paired delta against the 0.2520 probe, and each sd
    reproduces the threshold that ledger row records."""
    from cleft import phase16, phase17
    from cleft.train import phase3

    probe, n = 0.2520, 5

    a = phase17.ARM_MEANS["arms"]["p17_arm_a"]
    c = phase17.ARM_MEANS["arms"]["p17_arm_c"]
    # A and C reproduce the ledger's mean paired delta EXACTLY.
    assert a["pcc"] - probe == pytest.approx(-0.0186, abs=5e-5)
    assert c["pcc"] - probe == pytest.approx(-0.2705, abs=5e-5)
    # B does not, and the record says why (paired vs pooled).
    b = phase17.ARM_MEANS["arms"]["p17_arm_b"]
    assert b["pcc"] - probe == pytest.approx(-0.2564, abs=5e-5)
    assert "paired-vs-pooled" in _flat(phase17.ARM_MEANS["checked_against"])

    # The sds reproduce the ledger's thresholds.
    for arm, expected in ((b, 0.0307), (c, 0.0530)):
        got = phase3.combined_claimable_delta(arm["sd"], n, 0.0148, n)
        assert got["arm_means_95"] == pytest.approx(expected, abs=5e-5)
    # A's is 0.0135 against a banked 0.0136 -- 4-dp rounding, recorded.
    got_a = phase3.combined_claimable_delta(a["sd"], n, 0.0148, n)
    assert got_a["arm_means_95"] == pytest.approx(0.0135, abs=5e-5)
    assert "0.0135 for A against a banked 0.0136" in _flat(
        phase17.ARM_MEANS["checked_against"]
    )

    # Phase 16's loop sd reproduces its own threshold.
    loop = phase16.ARM_MEANS["arms"]["p16_anchor_loop"]
    assert phase3.combined_claimable_delta(loop["sd"], n, 0.0148, n)[
        "arm_means_95"
    ] == pytest.approx(0.0329, abs=5e-5)
    # The identity baseline is deterministic: no sd to compare.
    identity = phase16.ARM_MEANS["arms"]["p16_identity_baseline"]
    assert identity["sd"] is None
    assert "floating point" in _flat(identity["deterministic"])
    # And the direction that IS the finding.
    assert loop["pcc"] < identity["pcc"]


def test_the_prose_the_constants_came_from_is_unedited():
    """The closings still say what they said; the constants sit beside
    them with a dated pointer, and did not replace them."""
    from cleft import phase16, phase17

    closing17 = phase17.PHASE_17_CLOSING["criterion_2_three_arms_zero_real"]
    assert "A 0.2334 (sd 0.0044)" in closing17
    assert "B -0.0044 (sd 0.0317)" in closing17
    assert "C -0.0185 (sd 0.0586)" in closing17
    assert phase17.PHASE_17_CLOSING["arm_means_structured_2026_08_31"] == (
        "ARM_MEANS"
    )

    closing16 = phase16.PHASE_16_CLOSING["criterion_6_identity_baseline"]
    assert "identity 0.2151" in closing16 and "0.2040" in closing16
    assert phase16.PHASE_16_CLOSING["arm_means_structured_2026_08_31"] == (
        "ARM_MEANS"
    )
    assert "untouched" in _flat(phase16.ARM_MEANS["the_prose_is_unedited"])

    # The constants agree with the prose they were transcribed from.
    assert phase17.ARM_MEANS["arms"]["p17_arm_a"]["pcc"] == 0.2334
    assert phase16.ARM_MEANS["arms"]["p16_anchor_loop"]["pcc"] == 0.2040


def test_every_resolved_value_carries_the_record_it_came_from():
    banked = record_audit.banked_arm_pcc()
    for name, cell in banked.items():
        if cell["pcc"] is None:
            continue
        source = cell["source"]
        assert source, name
        # The source names a real module-level record, not a document.
        module = source.split(".")[0]
        assert module in {
            "ladder", "roadb", "graph_cleft", "phase11", "phase12", "phase15",
            "phase16", "phase17",
        }, (name, source)
        assert ".md" not in source and "docs/" not in source


def test_spot_checks_resolve_to_the_right_cells():
    """Read the same values a second way and require agreement -- a
    mis-parse that landed on a neighbouring cell would pass a coverage
    count but fail here."""
    from cleft import ladder

    banked = record_audit.banked_arm_pcc()
    inits = list(ladder.STAGE_D1_AT_G1["inits"])

    assert banked["p7_d1_vit_b16_imagenet_g1"]["pcc"] == (
        ladder.STAGE_D1_AT_G1["cells"]["vit_b16"][inits.index("imagenet")]
    )
    assert banked["p7_c_vit_b16_scut_masked_g1"]["pcc"] == (
        ladder.STAGE_D1_AT_G1["cells"]["vit_b16"][inits.index("scut_masked")]
    )
    assert banked["p7_c0_vit_b16_scut_original_g1"]["pcc"] == (
        ladder.STAGE_D1_AT_G1["cells"]["vit_b16"][inits.index("scut_original")]
    )
    assert banked["p7_d_swin_b_scut_original_g2"]["pcc"] == (
        ladder.STAGE_D_AT_G2["cells"]["swin_b"][inits.index("scut_original")]
    )
    # The anchor, the one number most often quoted.
    assert banked["p7_d1_vit_b16_imagenet_g1"]["pcc"] == 0.2520
    # Road B's masked spelling bridges to the record's scut_masked.
    # (The resolution keys are ints in the record, not strings.)
    assert banked["roadb_vit_b16_masked_224"]["pcc"] == (
        roadb.PHASE_7_TWENTY_TWO_ARMS["values"]["vit_b16__scut_masked"][224][0]
    )
    # And the freshly banked Stage E cell resolves to its new record.
    assert banked["p7_e_srgnn_scut_masked_g2_anatomy"]["pcc"] == 0.0554
    assert "STAGE_E_PER_ARM" in (
        banked["p7_e_srgnn_scut_masked_g2_anatomy"]["source"]
    )


def test_an_unparseable_arm_name_raises_rather_than_guessing():
    assert record_audit._parse_ladder_name("p7_d1_vit_b16_imagenet_g1") == (
        "vit_b16", "imagenet", "g1"
    )
    # No geometry, unknown backbone, unknown init -> None, so the caller
    # falls through to the explicit table instead of guessing a cell.
    for bad in ("p7_d1_vit_b16_imagenet", "p7_d1_resnet_imagenet_g1",
                "p7_d1_vit_b16_madeup_g1", "p12_arm_a_frontal"):
        assert record_audit._parse_ladder_name(bad) is None
    with pytest.raises(record_audit.RecordAuditError, match="not in"):
        record_audit._ladder_cell("vit_b16", "nonsense", "g1")
    with pytest.raises(record_audit.RecordAuditError, match="not present"):
        record_audit._ladder_cell("resnet50", "imagenet", "g1")


# --------------------------------------------------------------------------
# the comparison -- teeth
# --------------------------------------------------------------------------


def test_a_matching_artifact_agrees():
    banked = record_audit.banked_arm_pcc()
    measured = {
        n: c["pcc"] for n, c in banked.items() if c["pcc"] is not None
    }
    verdict = record_audit.compare_to_artifact(banked, measured)
    assert verdict["agree"] is True
    assert verdict["n_compared"] == 68
    assert verdict["mismatches"] == []
    assert verdict["largest_delta"]["delta"] == 0.0


def test_a_planted_mismatch_fires():
    """The teeth. On this machine the comparison never runs against a
    real artifact, so without this a green suite would be a green light
    with nothing behind it."""
    banked = record_audit.banked_arm_pcc()
    measured = {
        n: c["pcc"] for n, c in banked.items() if c["pcc"] is not None
    }
    measured["p7_d1_vit_b16_imagenet_g1"] += 0.02
    verdict = record_audit.compare_to_artifact(banked, measured)
    assert verdict["agree"] is False
    assert [m["arm"] for m in verdict["mismatches"]] == [
        "p7_d1_vit_b16_imagenet_g1"
    ]
    fired = verdict["mismatches"][0]
    assert fired["delta"] == pytest.approx(0.02)
    assert fired["banked"] == 0.2520
    assert "ladder.STAGE_D1_AT_G1" in fired["source"]

    # Just under the threshold does NOT fire -- the bound is real.
    measured["p7_d1_vit_b16_imagenet_g1"] = 0.2520 + 0.004
    assert record_audit.compare_to_artifact(banked, measured)["agree"] is True


def test_one_sided_arms_are_reported_not_skipped():
    banked = record_audit.banked_arm_pcc()
    measured = {
        n: c["pcc"] for n, c in banked.items() if c["pcc"] is not None
    }
    dropped = measured.pop("p12_arm_c_concat")
    measured["an_arm_nobody_banked"] = dropped
    verdict = record_audit.compare_to_artifact(banked, measured)
    assert "p12_arm_c_concat" in verdict["banked_without_artifact"]
    assert "an_arm_nobody_banked" in verdict["artifact_without_banked"]
    # Present on one side only, so it is not counted as agreement.
    assert "p12_arm_c_concat" not in verdict["deltas"]


def test_a_nonpositive_threshold_is_refused():
    with pytest.raises(record_audit.RecordAuditError, match="positive"):
        record_audit.compare_to_artifact({}, {}, threshold=0.0)


# --------------------------------------------------------------------------
# the records
# --------------------------------------------------------------------------


def test_the_one_off_result_is_recorded_as_a_snapshot():
    record = record_audit.RECORD_ARTIFACT_CHECK
    assert record["arms_compared"] == 66
    assert record["mismatches_at_0_005"] == 0
    assert record["largest_delta"] == {"arm": "p17_arm_b", "delta": 0.0007}
    assert record["verdict"].startswith("PASS")
    assert "rounding on a 4-dp quote" in _flat(record["largest_delta_reading"])
    # Why 66 and not 68 -- the two Stage E cells came FROM this table.
    why = _flat(record["why_66_not_68"])
    assert "STAGE_E_PER_ARM banked them the same day FROM THIS TABLE" in why
    assert "a re-run today would compare 68" in why
    # And it says plainly what it is not.
    assert "SNAPSHOT" in record["what_it_is_not"]
    assert 0.0007 < record_audit.AGREEMENT_THRESHOLD


def test_the_first_execution_is_recorded_with_its_precision_story():
    record = record_audit.FIRST_EXECUTION
    assert record["agree"] is True
    assert record["arms_compared"] == 63          # at the time it ran
    assert record["mismatches"] == 0
    assert record["threshold"] == record_audit.AGREEMENT_THRESHOLD
    assert record["largest_delta"] == {
        "arm": "p7_d_swin_b_scut_original_g2", "delta": 5.052e-05,
    }
    assert record["banked_without_artifact"] == ()
    assert record["artifact_without_banked"] == ()
    # The expected provenance warning is named, not hidden.
    assert "keeper_outside_pinned_image" in record["compute"]
    assert "CPU-only" in record["compute"]
    # The gap it reported is closed now, and the record says so.
    assert "COVERED NOW" in record["uncovered_reported_by_name"]
    assert "a re-run compares 68" in record["uncovered_reported_by_name"]

    # THE POINT: the machinery is stricter than the document.
    precision = _flat(record["precision_distinction"])
    assert "0.0007" in precision and "5.052e-05" in precision
    assert "PARSED TO FOUR DECIMALS OUT OF A MARKDOWN DOCUMENT" in precision
    assert "FULL-PRECISION banked values from the records" in precision
    assert "stricter than the document it replaces" in precision
    # And the arithmetic of that claim holds.
    assert 5.052e-05 < 0.0007 < record_audit.AGREEMENT_THRESHOLD
    assert record_audit.RECORD_ARTIFACT_CHECK[
        "superseded_as_the_live_check_by"
    ] == "FIRST_EXECUTION"


def test_the_job_id_hazard_is_reported_and_measured_not_assumed():
    record = record_audit.JOB_ID_STALENESS_HAZARD
    assert "RECOMMENDATION" in record["reported"]
    assert "nothing is edited" in record["reported"]
    assert "no_result_is_affected" in record
    assert "LEGIBILITY" in record["no_result_is_affected"]
    assert "MISTYPED_LAUNCH_DELETED" in record["no_result_is_affected"]

    # The claim that the existing guard misses it is MEASURED here, on
    # the real directory name, not asserted in prose.
    from cleft import run_names

    name = "record_artifact_check__e2d8280f__p7d1-classification-metrics"
    stem, sha, job = run_names.parse_run_dir(name)
    assert stem == "record_artifact_check"
    assert sha == "e2d8280f"
    assert job == "p7d1-classification-metrics"
    assert run_names.job_id_contradictions(name) == []
    run_names.check_run_dir(name)   # passes: no axis token on either side
    assert "ZERO contradictions" in record[
        "the_existing_guard_passes_it_measured"
    ]

    # WARN not refuse, with the reason the existing guard was softened.
    assert record["recommendation"].startswith("WARN, do not refuse")
    assert "22 of 83" in record["recommendation"]
    assert "switched-off guard is worse than none" in record["recommendation"]

    # The proposed signal, and a worked example of why it would not
    # fire on a legitimate abbreviation.
    proposed = _flat(record["what_the_warning_would_check"])
    assert "shares ANY token with the config stem" in proposed
    legit = "p6_pretrain_swin_b_masked_g1__aaaaaaaa__p6-pt-swin-g1-v2"
    legit_stem, _, legit_job = run_names.parse_run_dir(legit)
    shared = run_names._tokens(legit_stem) & run_names._tokens(legit_job)
    assert shared, "the abbreviation shares tokens, so it would not warn"
    assert not (
        run_names._tokens(stem) & run_names._tokens(job)
    ), "this run shares none, which is the signal"

    # The real fix is upstream, and building it would be an escalation.
    assert "unset CLEFT_JOB_ID" in _flat(record["the_real_fix_is_upstream"])
    assert "FROZEN apparatus" in record["not_built"]


def test_the_design_says_job_not_suite_and_why():
    record = record_audit.STANDING_CHECK_DESIGN
    assert record["verdict"] == "A DECLARED JOB, not a suite test"
    why = _flat(record["why_not_the_suite"])
    assert "CLUSTER-ONLY" in why
    assert "A suite test cannot read them" in why

    # The pin was considered, and is unavailable for TWO stated reasons.
    pin = _flat(record["the_pin_option"])
    assert "CONSIDERED AND UNAVAILABLE" in pin
    assert "RECORD-VERSUS-RECORD" in pin
    assert "only the SUMMARY reached this machine" in pin
    assert "record-versus-SNAPSHOT" in pin

    owns = _flat(record["what_the_suite_owns"])
    assert "TEETH TEST" in owns
    assert "quietly covered three arms cannot pass" in owns
    # One implementation, and the closed phase left alone.
    one = _flat(record["one_implementation"])
    assert "read_cluster_csv" in one and "eval.metrics.pcc" in one
    assert "does not touch the Phase 18 task" in one


# --------------------------------------------------------------------------
# the job's surface
# --------------------------------------------------------------------------


def test_the_job_is_wired_and_reuses_the_shared_readers():
    import inspect

    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS, task_record_artifact_check

    assert "record_artifact_check" in TASK_SPECS
    assert TASKS["record_artifact_check"] is task_record_artifact_check

    source = inspect.getsource(task_record_artifact_check)
    # Reuse, not reimplementation.
    assert "read_cluster_csv" in source
    assert "frozen_metrics.pcc" in source
    assert "record_audit.banked_arm_pcc()" in source
    # It defines no metric and no second reader of its own.
    assert "def pcc" not in source
    assert "csv.reader" not in source
    # A mismatch is REPORTED, not raised: the comparison's result lands
    # in the written summary rather than in an exception. (Checked by
    # what the task writes, not by grepping for the word "raise" -- the
    # record explaining the policy contains it.)
    after = source.split("verdict = record_audit.compare_to_artifact")[1]
    assert 'raise ' not in after
    assert '"mismatches": verdict["mismatches"]' in after
    assert '"agree": verdict["agree"]' in after


def test_the_shipped_config_carries_no_banked_value():
    text = (REPO / "configs" / "record_artifact_check.yaml").read_text(
        encoding="utf-8"
    )
    config = yaml.safe_load(text)
    validate(config)
    task = config["task"]
    assert task["kind"] == "record_artifact_check"
    assert task["threshold"] == record_audit.AGREEMENT_THRESHOLD
    assert len(task["arms"]) == 68

    # THE POINT: no arm carries a value. The task reads banked figures
    # from the records at run time, so this is a record-versus-artifact
    # check and not a document-versus-artifact one.
    for arm in task["arms"]:
        assert set(arm) == {"name", "input", "seeds", "csv", "n_patients"}
        assert "pcc" not in arm and "banked" not in arm

    # Inputs carried from the Phase 18 config, byte for byte.
    p18 = yaml.safe_load(
        (REPO / "configs" / "p18_metric_space.yaml").read_text(
            encoding="utf-8"
        )
    )
    p18_by_name = {e["name"]: e for e in p18["inputs"]}
    assert len(config["inputs"]) == 68
    for entry in config["inputs"]:
        assert entry == p18_by_name[entry["name"]]

    # The header states coverage either way. [2026-08-31] With the gap
    # closed the loop over BANKED_VALUE_GAPS would be VACUOUS, so the
    # assertion is written to have teeth in both states: named arms when
    # there are gaps, an explicit all-covered line when there are none.
    if record_audit.BANKED_VALUE_GAPS:
        assert "NOT COVERED" in text
        for gap in record_audit.BANKED_VALUE_GAPS:
            assert gap in text
    else:
        assert "ALL 68 ARMS COVERED" in text
        assert "BANKED_VALUE_GAPS_CLOSED" in text
        assert "NOT COVERED" not in text


def test_the_threshold_cannot_be_widened_in_a_config():
    """A configurable tolerance is a tolerance someone widens until the
    check passes."""
    config = yaml.safe_load(
        (REPO / "configs" / "record_artifact_check.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["task"]["threshold"] = 0.05
    with pytest.raises(ConfigError):
        validate(config)


def test_the_grep_that_did_not_run_is_recorded_with_its_measurement():
    """A search that silently does not run is not a negative result.

    Recorded 2026-09-05: ``git grep`` with a pattern beginning ``/`` is
    rewritten by MSYS path conversion and returns a plausible wrong
    answer rather than an error.
    """
    record = record_audit.THE_GREP_THAT_DID_NOT_RUN
    assert record["found"] == "2026-09-05"

    what = " ".join(record["what_happens"].split())
    assert "BEGINS WITH" in what
    assert "It does not error" in what

    measured = " ".join(record["the_measurement"].split())
    assert "1 hit" in measured and "3,006" in measured

    invalid = " ".join(record["what_it_invalidates"].split())
    assert "partly blind" in invalid
    assert "do not stand" in invalid

    rule = " ".join(record["the_rule"].split())
    assert "ripgrep" in rule
    assert "cross-check a negative" in rule

    # It is filed as a member of a family the record already knows.
    shape = " ".join(record["the_general_shape"].split())
    assert "Already up to date" in shape
    assert "test_environment.py" in shape
    assert "REPOSITORY_POLICY" in shape
    # ...and the sibling it cites is live at its source.
    assert "confirmed only that no NEW commits existed upstream" in (
        " ".join(record_audit.REPOSITORY_POLICY["what_it_cost"].split())
    )


def test_the_pronoun_sweep_is_recorded_as_a_continuation():
    """Recorded 2026-09-06, with the reason the name pass missed it."""
    record = record_audit.THE_PRONOUNS_THE_NAME_SWEEP_MISSED
    assert record["continued"] == "2026-09-06"

    why = " ".join(record["why_it_was_missed"].split())
    assert "searched for names and never looked at pronouns" in why
    assert "cannot find what the sweep never looked for" in why

    swept = " ".join(record["what_was_swept"].split())
    assert "39 of 42" in swept
    assert "what turns on the answer" in swept

    # [UPDATED 2026-09-06] The three exceptions were removed by writing
    # the fact out in full, so the record reports none rather than three.
    left = " ".join(record["WHAT_WAS_LEFT_AND_WHY"].split())
    assert "none, and that took a second pass" in left
    assert "requested and not supplied" in left
    assert "no exemption list at all" in left

    rule = " ".join(record["the_general_rule_it_sets"].split())
    assert "a quoted artifact is not a reason to keep one" in rule
    assert "was never a necessary exception" in rule

    # And the record it describes really does state the fact without
    # reproducing the artifact.
    from cleft import phase21

    header = phase21.__doc__
    assert "the ruling" in header and "came back" in header
    proposed = " ".join(str(phase21.ARM_SET_RULE_PROPOSED).split())
    assert "AWAITING A RULING" in proposed


def test_no_tracked_prose_carries_a_masculine_pronoun():
    """**[TIGHTENED 2026-09-06] No exemptions, and none are needed.**

    The first sweep kept three quotations of an empty placeholder on
    the ground that reproducing it was the evidence. The record states
    the fact directly now, so there is nothing to except and this guard
    asserts a clean tree rather than a counted set of exceptions.

    Case INSENSITIVE, and it matches inside identifiers. The first
    pass was neither, which is how a capitalised form and three key
    names survived it.
    """
    import re
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
        check=True,
    ).stdout.split()
    pattern = re.compile(r"(?<![a-z])(his|him|himself)(?![a-z])", re.I)

    offenders = []
    for name in tracked:
        if name.rsplit(".", 1)[-1] not in ("py", "md", "yaml", "yml", "txt"):
            continue
        # This module defines the pattern, so it necessarily contains
        # it. That is the only file skipped, and it is skipped whole
        # rather than line by line so no prose can hide behind it.
        if name == "tests/test_record_audit.py":
            continue
        path = REPO / name
        if not path.is_file():
            continue
        try:
            body = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # pragma: no cover
            continue
        for number, line in enumerate(body.split("\n"), start=1):
            if pattern.search(line):
                offenders.append(f"{name}:{number} {line.strip()[:70]}")

    assert not offenders, (
        "masculine pronouns in tracked prose:\n  " + "\n  ".join(offenders)
        + "\n\nSweep to the role. There is no exemption list, and the one "
        "that existed was removed by writing the fact out in full instead."
    )
