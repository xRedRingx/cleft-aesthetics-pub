"""Phase 24: the reckoning, the concessions and the substituted measurement.

**Every figure this phase quotes is asserted against the record it comes
from**, not against itself -- the concessions are only as good as their
sources, and a concession resting on a misquoted number would be worse
than no concession.
"""

from __future__ import annotations

import pathlib

import pytest

from cleft import phase24

REPO = pathlib.Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# the reckoning
# --------------------------------------------------------------------------


def test_the_reckoning_leads_with_what_is_conceded():
    record = phase24.PHASE_24_RECKONING
    assert record["tag"].startswith("[REGISTERED]")

    conceded = _flat(record["what_is_conceded"])
    assert "three of four designs, and the fourth at its premise" in conceded
    for token in ("CONCEDED-COVERED", "CONCEDED-MEASURED",
                  "CONCEDED at its premise"):
        assert token in conceded, token

    precedent = _flat(record["the_precedent_that_governs"])
    assert "phase14.PHASE_14_CONCEDED_COVERED" in precedent
    assert "the phase doing its job, not the phase failing" in precedent

    # The phase does not inflate itself.
    shape = _flat(record["the_honest_shape_of_this_phase"])
    assert "measurement, not modelling" in shape
    assert "is a small phase" in shape
    assert "RESTATES and DESCRIBES; it does not CLAIM" in _flat(
        record["no_ledger_row"]
    )


def test_phase_14s_precedent_is_quoted_as_phase_14_wrote_it():
    from cleft import phase14

    source = phase14.PHASE_14_CONCEDED_COVERED
    assert "The concession is the phase doing its job, not the phase failing" \
        in _flat(phase14.__doc__)
    # The reopen condition is quoted, not paraphrased.
    quoted = _flat(
        phase24.THE_LABEL_DISTRIBUTION_HEAD_CONCEDED_COVERED[
            "the_reopen_condition_STANDS_UNCHANGED"
        ]
    )
    original = _flat(source["what_would_reopen"])
    for fragment in ("a mechanism differing from the closed arm's on a "
                     "named axis", "Scheduling was not reopening; closing is "
                     "not forbidding"):
        assert fragment in original, fragment
        assert fragment in quoted, fragment


# --------------------------------------------------------------------------
# design 1 -- conceded on identity
# --------------------------------------------------------------------------


def test_design_1s_figures_match_the_ladders_own_stage_g():
    from cleft import ladder

    record = phase24.THE_LABEL_DISTRIBUTION_HEAD_CONCEDED_COVERED
    measured = _flat(record["what_it_measured"])

    triples = ladder.STAGE_G_LABEL_FORMULATION["triples"]
    g1 = triples["imagenet__g1"]["arms"]
    g2 = triples["scut_masked__g2"]["arms"]
    assert g1["ldl"]["pcc"] == 0.2336 and g1["mean"]["pcc"] == 0.2520
    assert g2["ldl"]["pcc"] == 0.1583 and g2["mean"]["pcc"] == 0.2001
    for token in ("0.2336", "0.2520", "0.1583", "0.2001", "-0.0184",
                  "-0.0418"):
        assert token in measured, token
    # Claimably worse at G2 -- the ladder's own word.
    assert triples["scut_masked__g2"]["rederived"]["ldl"]["claimable"] is True
    assert "CLAIMABLY WORSE" in measured
    assert "No label alternative beats the mean anywhere" in measured


def test_the_no_fishing_clause_is_quoted_and_its_purpose_named():
    from cleft import phase14

    record = phase24.THE_LABEL_DISTRIBUTION_HEAD_CONCEDED_COVERED
    clause = _flat(record["the_clause_that_forbids_renaming_it"])
    original = _flat(
        phase14.PHASE_14_CONCEDED_COVERED["variants_named_not_pursued"]
    )
    for fragment in (
        "EXPLICITLY UNREGISTERED -- no fishing",
        "LDL-style loss as PRETRAINING for a scalar head",
        "distribution supervision at a DIFFERENT REPRESENTATION",
        "presenting either as a fresh idea that escapes this record",
    ):
        assert fragment in original, fragment
        assert fragment in clause, fragment

    # And the record says WHY it is quoted here.
    why = _flat(record["rebuilding_it_under_a_new_number_is_what_that_clause_prevents"])
    assert "the closed arm with a phase number in front of it" in why


# --------------------------------------------------------------------------
# design 2 -- conceded on a run
# --------------------------------------------------------------------------


def test_design_2s_five_cells_match_the_rater_screen_exactly():
    from cleft import phase10, results_ledger

    record = phase24.THE_PER_RATER_HEADS_CONCEDED_MEASURED
    measured = _flat(record["the_measurement"])

    observed = phase10.RATER_SCREEN_OBSERVED["per_rater"]
    expected = {
        "cleft patient": -0.0098, "orthodontist": 0.1606,
        "speech and language therapist": 0.2150,
        "plastic surgeon": 0.2703, "psychologist": 0.1362,
    }
    for rater, pcc in expected.items():
        assert observed[rater]["pcc"] == pcc, rater
    for token in ("-0.0098", "0.1606", "0.2150", "0.2703", "0.1362",
                  "0.2520", "+0.0183", "1.24"):
        assert token in measured, token
    assert "Four below 0.2520; one nominally above and INSIDE the band" \
        in measured

    # The ledger row is real, and is row 22 with the status claimed.
    row = results_ledger.ENTRIES[22]
    assert row["id"] == "p10-rater-screen-mixed"
    assert row["status"] == "DESCRIPTIVE"
    assert "row 22" in _flat(record["it_is_ledgered"])


def test_design_2_is_the_registered_ladder_whose_trigger_did_not_fire():
    from cleft import phase10

    ladder_record = phase10.RATER_LADDER_CONDITIONAL
    assert ladder_record["runs"] == 400
    assert "not built" in ladder_record["registered"]
    assert "the maintainer's word regardless" in ladder_record["trigger"]

    record = phase24.THE_PER_RATER_HEADS_CONCEDED_MEASURED
    claim = _flat(record["a_phase_24_of_per_rater_heads_IS_the_registered_ladder"])
    assert "400 runs, REGISTERED AND NOT BUILT" in claim
    assert "the screen did not refute it" in claim
    assert "UNMET" in claim
    # The other trigger is not quietly dropped.
    assert "not being exercised here" in _flat(
        record["the_maintainers_word_remains_a_live_trigger"]
    )
    assert phase10.RATER_SCREEN_OBSERVED["ladder_not_triggered"].startswith(
        "no cell beyond the band"
    )


# --------------------------------------------------------------------------
# design 3 -- conceded at its premise
# --------------------------------------------------------------------------


def test_design_3s_taus_match_phase_21_and_are_not_lower_bounds():
    from cleft import phase21

    cross = phase21.ARM_B_OBSERVED["cross_reference"]
    assert cross["residual_sign"]["tau_b"] == 0.0727
    assert cross["worst_quartile"]["tau_b"] == -0.0811
    assert cross["residual_sign"]["permutation_p"] == 0.1226
    assert cross["worst_quartile"]["permutation_p"] == 0.0835

    record = phase24.AGREEMENT_WEIGHTED_TRAINING_CONCEDED_AT_ITS_PREMISE
    measured = _flat(record["the_measurement_that_refutes_it"])
    for token in ("+0.0727", "-0.0811", "0.1226", "0.0835",
                  "b_hardness_does_not_correlate_with_disagreement"):
        assert token in measured, token
    assert "OPPOSITE IN SIGN" in measured

    # The correction ordering is stated, not assumed.
    bounds = _flat(record["these_taus_POSTDATE_the_correction_and_are_not_lower_bounds"])
    assert "LOWER BOUND" in bounds
    assert "six Phase 18 figures" in bounds
    assert "These two are not" in bounds
    # phase21 really does carry the defect record.
    assert "kendall_tau_b" in _flat(str(phase21.summary()))


# --------------------------------------------------------------------------
# design 4 -- conceded with a reason, and not silently
# --------------------------------------------------------------------------


def test_design_4_is_conceded_with_its_reason_and_not_folded_in():
    record = phase24.DISAGREEMENT_AS_AUXILIARY_TARGET_STATUS

    # The honest part comes first: it is NOT covered by an arm.
    uncovered = _flat(record["it_is_NOT_covered_by_any_arm"])
    assert "No arm in this project has predicted rater disagreement" in uncovered
    assert "This design is genuinely unbuilt" in uncovered

    # It is the cheapest, which is why the reason has to be real.
    reachable = _flat(record["and_it_is_manifest_reachable"])
    assert "4.44e-16" in reachable
    assert "No score sheet needed" in reachable

    # The non-refuted motivation is NAMED and then answered.
    survives = _flat(
        record["the_one_motivation_that_would_NOT_be_refuted_and_why_it_fails_HERE"]
    )
    assert "multi-task learning as REGULARISATION" in survives
    assert "does not route through the refuted premise" in survives
    assert "THE BACKBONE IS FROZEN" in survives
    assert "nothing to regularise" in survives

    assert "the concession is on the record with its reason" in _flat(
        record["it_is_not_conceded_silently"]
    )
    assert "an arm that trains its representation" in _flat(
        record["what_would_reopen_it"]
    )


def test_the_frozen_backbone_claim_is_true_of_the_recipe():
    """The structural argument fails if any of our own arms fine-tunes."""
    from cleft import phase10

    screen = (REPO / "configs" / "p10_rater_screen.yaml").read_text(
        encoding="utf-8"
    )
    assert "only the head refits" in screen
    # train/ldl.py and train/ranking.py both head over frozen embeddings.
    # The screen's task heads over PRE-EXTRACTED embeddings: the
    # backbone is not in the graph at all.
    run_body = (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    task = run_body.split("def task_rater_screen(")[1].split("\ndef ")[0]
    assert "EmbeddingHeadBackbone" in task
    assert "embeddings_module_load(embeddings_dir, patient_ids)" in task
    # The one end-to-end fine-tune in the record is the CleftGNN
    # replication, and its launches are VOID.
    assert "fine-tunes ResNet-50 end to end" in _flat(
        str(phase10.summary())
    ) or "fine-tune" in (
        REPO / "src" / "cleft" / "phase10.py"
    ).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# the substitute
# --------------------------------------------------------------------------


def test_the_substituted_deliverable_is_registered_and_not_computed():
    record = phase24.THE_SUBSTITUTED_DELIVERABLE
    assert record["tag"].startswith("[REGISTERED]")
    assert "not computed" in record["tag"]

    quantity = _flat(record["the_quantity"])
    assert "k = 1..20" in quantity
    assert "0.85, 0.90 and 0.95" in quantity
    assert "DIFFERENT QUANTITIES" in quantity

    population = _flat(record["the_population_is_237_and_only_237"])
    assert "MEAN_R_237 = 0.4696, never MEAN_R_251 = 0.4560" in population
    assert "third time that trap has been walked into" in population

    # It really is a projection beyond what exists.
    assert "Only k = 5 is a measurement" in _flat(
        phase24.THE_ASSUMPTIONS_DECLARED[
            "assumption_3_the_curve_extrapolates_beyond_what_exists"
        ]
    )


def test_the_gap_the_substitute_fills_is_real():
    """Registered as: the function takes any k and nothing uses one."""
    from cleft.data import labels, reliability

    import inspect

    # The function accepts arbitrary k.
    assert "k: int = N_RATERS" in inspect.getsource(reliability.spearman_brown)
    # Every caller in src/ passes 5 (or the matrix width) or 1.
    source = "\n".join(
        (REPO / "src" / "cleft" / "data" / name).read_text(encoding="utf-8")
        for name in ("reliability.py", "labels.py")
    )
    assert "spearman_brown(mean_r, array.shape[1])" in source
    assert 'R.spearman_brown(R.mean_inter_rater_r(matrix), k)' in source
    assert labels.attenuation_ceiling is not None
    # And no G-theory anywhere.
    for path in (REPO / "src").rglob("*.py"):
        if path.name == "phase24.py":
            continue          # it is where the ABSENCE is recorded
        body = path.read_text(encoding="utf-8").lower()
        assert "generalizability theory" not in body, path
        assert "d-study" not in body, path


def test_the_lineage_check_recomputes_and_is_labelled_a_check():
    from cleft.data import reliability as R

    record = phase24.THE_LINEAGE_CHECK
    assert record["tag"].startswith("[LITERATURE] + [MEASURED]")
    assert "not a reading" in record["tag"]

    # Ours does NOT reproduce their 0.9 ...
    ours = R.spearman_brown(R.MEAN_R_237, 6)
    assert ours == pytest.approx(0.8416, abs=5e-5)
    assert "0.8416" in _flat(record["does_our_mean_r_reproduce_it"])
    assert "NO" in _flat(record["does_our_mean_r_reproduce_it"])

    # ... and theirs does, exactly, from their own banked single-judge.
    theirs = R.spearman_brown(0.60, 6)
    assert theirs == pytest.approx(0.90, abs=1e-12)
    check = _flat(record["and_the_formula_CHECKS_OUT_on_their_own_number"])
    assert "spearman_brown(0.60, 6) = 0.9000" in check
    assert "r = 0.60 is precisely the single-rater agreement" in check
    assert "entirely in the INPUT, not in the formula" in check

    # The quote and the source are the record's own.
    from cleft import phase12

    banked = phase12.PRIMARY_SOURCES_BANKED
    assert "six examiners would produce a pooled panel reliability of 0.9" in \
        _flat(banked["ceiling_provenance"]["the_lineages_half"])
    assert "0.60 (total)" in _flat(banked["reliability_anchors"]["figures"])
    assert "Asher-McDade 1992 Part 4 (the operational panel)" in banked["sources"]
    assert "Asher-McDade 1992 Part 4" in _flat(record["the_literature_projection"])


def test_the_pre_emption_of_reading_b_is_declared_not_discovered():
    record = phase24.THE_LINEAGE_CHECK
    pre_empt = _flat(record["IT_PARTIALLY_PRE_EMPTS_READING_B_AND_THAT_IS_SAID_HERE"])
    assert "below 0.90" in pre_empt
    assert "It does not settle it" in pre_empt
    assert "not blind" in pre_empt
    # And the readings themselves say so too.
    assert "NOT blind" in _flat(
        phase24.READINGS_COMMITTED["the_readings_are_NOT_blind"]
    )


# --------------------------------------------------------------------------
# limitations and the prohibition
# --------------------------------------------------------------------------


def test_the_assumptions_are_declared_with_the_bias_direction():
    record = phase24.THE_ASSUMPTIONS_DECLARED

    one = _flat(record["assumption_1_exchangeable_raters_of_equal_quality"])
    assert "0.654" in one and "0.628" in one
    assert "-0.0098 to 0.2703" in one
    assert "A sixth rater in the projection is an average rater who does not exist" \
        in one

    two = _flat(record["assumption_2_errors_uncorrelated_across_raters"])
    assert "UPPER bound" in two
    assert "bias direction is known and unstated is worse" in two

    assert "these_are_limitations_not_disqualifications" in record


def test_the_item_total_span_is_flagged_as_absent_from_the_record():
    """0.554-0.677 is in the message and NOT in the repository."""
    flag = _flat(phase24.THE_ASSUMPTIONS_DECLARED[
        "the_item_total_SPAN_is_not_in_the_record"
    ])
    assert "THAT RANGE IS NOT IN THIS REPOSITORY" in flag
    assert "SLT 0.654 and orthodontist 0.628" in flag
    assert "gets a dated correction" in flag

    # Verified, not asserted: neither figure appears in any record module.
    for path in (REPO / "src" / "cleft").glob("*.py"):
        body = path.read_text(encoding="utf-8")
        if path.name == "phase24.py":
            continue
        assert "0.554" not in body, path
        assert "0.677" not in body, path
    # And the two that ARE banked live where the record says.
    reliability = (
        REPO / "src" / "cleft" / "data" / "reliability.py"
    ).read_text(encoding="utf-8")
    assert "0.654" in reliability and "0.628" in reliability


def test_a_ceiling_is_not_a_score_is_a_tested_literal():
    literal = phase24.A_CEILING_IS_NOT_A_SCORE
    assert literal.startswith("A HIGHER CEILING IS NOT A HIGHER SCORE.")
    assert "FORBIDDEN" in literal
    assert "0.2520 against a ceiling of 0.9032" in literal
    assert "the gap to the ceiling, not the ceiling, is the binding" in literal

    # The banked figures the literal leans on are real.
    from cleft.data import reliability as R

    assert R.PCC_CEILING_237 == 0.9032
    assert R.pcc_ceiling(R.RELIABILITY_237) == pytest.approx(0.9032, abs=5e-5)

    # It has siblings, and they are the record's own.
    siblings = phase24.PROHIBITION_SIBLINGS
    assert len(siblings) == 3
    joined = " ".join(_flat(s) for s in siblings)
    assert "different quantities, coincidentally adjacent" in joined
    assert "population mismatch" in joined


# --------------------------------------------------------------------------
# readings, criteria, settings, and the negative space
# --------------------------------------------------------------------------


def test_both_readings_are_useful_and_neither_moves_a_figure():
    record = phase24.READINGS_COMMITTED
    assert "COSTED recommendation" in _flat(record["a_a_modest_k_reaches_0_90"])
    assert "reliability is EXPENSIVE" in _flat(
        record["b_0_90_needs_many_raters"]
    )
    assert "neither outcome is the hoped-for one" in _flat(
        record["the_two_are_not_ranked_in_advance"]
    )
    unchanged = _flat(record["what_NEITHER_outcome_changes"])
    assert "no banked figure moves" in unchanged
    assert "0.2520" in unchanged and "0.9032" in unchanged
    assert "No ledger row" in _flat(record["descriptive_no_ledger_row"])

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 24 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p24" or "p24-" in e["id"]
    ]
    results_ledger.validate()          # raises if the chain broke


def test_the_exit_criteria_are_a_draft_and_the_settings_are_open():
    draft = phase24.EXIT_CRITERIA_DRAFT
    assert draft["status"] == "DRAFT, NOT LOCKED"
    numbered = [k for k in draft if k[0].isdigit()]
    assert len(numbered) == 9
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 10))
    assert "fourth instance of a trap already walked into three times" in _flat(
        draft["2_the_population_is_237_and_the_trap_is_named"]
    )
    assert "however big it is" in _flat(
        draft["7_the_inverse_is_reported_even_when_it_is_large"]
    )

    settings = phase24.SETTINGS_TO_DECLARE
    assert len(settings) == 5
    for key, text in settings.items():
        assert "**Open**" in text or "**open" in text or "open," in text, key
    assert "1..20" in _flat(settings["1_the_k_range"])
    assert "0.85, 0.90, 0.95" in _flat(
        settings["2_the_reliability_targets_for_the_inverse"]
    )


def test_no_config_no_task_and_no_run_directory():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held the
    phase at registration -- one function and no script. **The
    measurement is now built**, as ruled, and what the pin holds is what
    the ruling actually forbade: this phase touches NO run machinery.
    Setting 4: 'the laptop. NO RUN DIRECTORY.'"""
    from cleft.config import schema

    assert not list((REPO / "configs").glob("p24*.yaml"))
    assert not any(k.startswith("p24") for k in schema.TASK_SPECS)

    run_body = (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    assert "phase24" not in run_body
    assert "task_p24" not in run_body

    # The script exists and reads nothing: no artifact, no declare.
    script = (REPO / "scripts" / "phase24_reliability_curve.py").read_text(
        encoding="utf-8"
    )
    assert "Runs on the laptop and reads NOTHING" in script
    # Not by keyword-hunting the prose -- by the CODE reading nothing.
    import ast
    import inspect

    tree = ast.parse(script)
    calls = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "read_text" not in calls and "open" not in calls
    assert "read_cluster_csv" not in script
    # And the measurement itself takes no inputs at all.
    assert list(inspect.signature(phase24.report).parameters) == []
    assert list(inspect.signature(phase24.lineage_check).parameters) == []

    # The module computes, and computes with numpy nowhere.
    module = (REPO / "src" / "cleft" / "phase24.py").read_text(encoding="utf-8")
    assert "import numpy" not in module


def test_the_summary_carries_every_record():
    assert sorted(phase24.summary()) == [
        "assumptions",
        "ceiling_is_not_a_score",
        "curve_observed",
        "design_1_label_distribution",
        "design_2_per_rater_heads",
        "design_3_agreement_weighted",
        "design_4_disagreement_auxiliary",
        "exit_criteria",
        "exit_criteria_draft",
        "lineage_check",
        "prohibition_siblings",
        "readings",
        "reckoning",
        "settings_ruled",
        "settings_to_declare",
        "substituted_deliverable",
    ]

# --------------------------------------------------------------------------
# [2026-09-02] The lock and the measurement
# --------------------------------------------------------------------------


def test_the_criteria_are_locked_with_the_settings_and_the_clause():
    criteria = phase24.EXIT_CRITERIA
    assert criteria["status"] == "LOCKED"
    numbered = [k for k in criteria if k[0].isdigit()]
    assert len(numbered) == 9
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 10))

    # Criterion 7 is the one that had to survive an unflattering number.
    seven = _flat(criteria["7_the_inverse_is_reported_however_large_it_is"])
    assert "whatever it is" in seven
    assert "choosing the range after seeing the numbers" in seven

    added = _flat(criteria["nothing_added_after"])
    assert "NOTHING IS ADDED ONCE NUMBERS EXIST" in added
    assert "does not extend the table" in added

    # The five settings are ruled and carried inside the lock.
    settings = criteria["settings"]
    assert settings is phase24.THE_SETTINGS_RULED
    assert len([k for k in settings if k[0].isdigit()]) == 5
    assert "k = 1..20" in _flat(settings["1_k_range"])
    assert "0.85 / 0.90 / 0.95" in _flat(
        settings["2_targets_and_how_k_is_reported"]
    )
    assert "the integer the actionable figure" in _flat(
        settings["2_targets_and_how_k_is_reported"]
    ).replace("named ", "")
    assert "237 only" in _flat(settings["3_population"])
    assert "NO RUN DIRECTORY" in _flat(settings["4_where_it_runs"])
    assert "NOT recomputed" in _flat(
        settings["5_mean_r_is_the_banked_constant"]
    )

    # The draft is preserved beside the lock, not overwritten.
    assert phase24.EXIT_CRITERIA_DRAFT["status"] == "DRAFT, NOT LOCKED"


def test_the_curve_reuses_the_shipped_function_and_adds_no_second_one():
    """A second implementation of a banked formula would be one quantity
    computed two ways."""
    import inspect

    from cleft.data import reliability as R

    body = inspect.getsource(phase24.curve)
    assert "R.spearman_brown(mean_r, int(k))" in body
    assert "R.pcc_ceiling(value)" in body
    # The formula itself appears NOWHERE in phase24's code.
    module = (REPO / "src" / "cleft" / "phase24.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in module.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "k * mean_r / (1" not in code
    assert "(k - 1) * mean_r" not in code

    rows = phase24.curve(R.MEAN_R_237)
    assert [row["k"] for row in rows] == list(range(1, 21))
    for row in rows:
        assert row["pcc_ceiling"] == pytest.approx(
            R.pcc_ceiling(row["reliability"]), abs=1e-12
        )


def test_the_curve_reproduces_the_two_banked_constants_at_k_five():
    """The free check: k = 5 is the one point that is a measurement."""
    from cleft.data import reliability as R

    at_five = next(r for r in phase24.curve(R.MEAN_R_237) if r["k"] == 5)
    assert at_five["reliability"] == pytest.approx(0.8157, abs=5e-5)
    assert at_five["reliability"] == pytest.approx(R.RELIABILITY_237, abs=1e-4)
    assert round(at_five["pcc_ceiling"], 4) == R.PCC_CEILING_237 == 0.9032

    recorded = _flat(phase24.THE_CURVE_OBSERVED[
        "the_curve_REPRODUCES_the_two_banked_constants_at_k_5"
    ])
    assert "a free check, and it passed" in recorded
    assert "0.8157" in recorded and "0.8158" in recorded
    assert "the banked figure's own rounding" in recorded


def test_the_inverse_is_verified_through_the_forward_function():
    from cleft.data import reliability as R

    expected = {0.85: (6.4003, 7), 0.90: (10.1652, 11), 0.95: (21.4600, 22)}
    for target, (exact, integer) in expected.items():
        got = phase24.raters_required(R.MEAN_R_237, target)
        assert got["k_exact"] == pytest.approx(exact, abs=5e-5), target
        assert got["k_integer"] == integer, target
        # FIRST meets or exceeds: k does, k-1 does not.
        assert R.spearman_brown(R.MEAN_R_237, integer) >= target
        assert R.spearman_brown(R.MEAN_R_237, integer - 1) < target
        assert got["reliability_at_k_integer"] >= target

    # A target below the single-rater bar needs one rater, not zero.
    assert phase24.raters_required(R.MEAN_R_237, 0.4)["k_integer"] == 1
    for bad in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match="reliability target"):
            phase24.raters_required(R.MEAN_R_237, bad)


def test_the_0_95_answer_is_outside_the_table_and_the_table_is_not_extended():
    from cleft.data import reliability as R

    data = phase24.report()
    at_95 = next(r for r in data["inverse"] if r["target"] == 0.95)
    assert at_95["k_integer"] == 22
    assert at_95["within_the_ruled_table"] is False
    # The ruled range is untouched by the unflattering answer.
    assert max(row["k"] for row in data["curve"]) == 20
    assert phase24.K_RANGE == tuple(range(1, 21))

    recorded = _flat(phase24.THE_CURVE_OBSERVED[
        "the_0_95_target_is_OUTSIDE_the_ruled_table_and_is_reported_anyway"
    ])
    assert "22 raters is beyond the ruled k = 1..20" in recorded
    assert "The table is NOT extended to 22" in recorded
    # And it is printed rather than quietly dropped.
    assert "BEYOND the ruled k=1..20 table, reported anyway" in phase24.render()
    assert R.spearman_brown(R.MEAN_R_237, 22) >= 0.95


def test_the_lineage_check_is_in_the_output_not_only_the_record():
    from cleft.data import reliability as R

    check = phase24.lineage_check()
    assert check["their_projection_reproduced"] == pytest.approx(0.90, abs=1e-12)
    assert check["reproduces_exactly"] is True
    assert check["ours_at_the_same_k"] == pytest.approx(0.8416, abs=5e-5)
    assert check["r_implied_by_their_projection"] == pytest.approx(0.60, abs=1e-12)
    assert check["r_implied_by_their_projection"] == pytest.approx(
        phase24.LINEAGE_SINGLE_JUDGE_R, abs=1e-12
    )

    text = phase24.render()
    assert "spearman_brown(0.6, 6) = 0.9000" in text
    assert "reproduces their 0.9 EXACTLY" in text
    assert "spearman_brown(0.4696, 6) = 0.8416" in text
    assert "the whole discrepancy is in the INPUT" in text
    assert "calibrated examiners against our uncalibrated mixed panel" in text
    # Their 0.60 is the record's, not invented here.
    from cleft import phase12

    assert "0.60 (total)" in _flat(
        phase12.PRIMARY_SOURCES_BANKED["reliability_anchors"]["figures"]
    )
    assert R.spearman_brown(phase24.LINEAGE_SINGLE_JUDGE_R, 6) == pytest.approx(
        0.9, abs=1e-12
    )


def test_the_bindings_are_carried_into_the_output():
    text = phase24.render()

    # The prohibition, verbatim.
    assert phase24.A_CEILING_IS_NOT_A_SCORE in text
    assert "0.2520 against a ceiling of 0.9032" in text

    # Both assumptions, with the bias direction.
    assert "EXCHANGEABLE RATERS OF EQUAL QUALITY" in text
    assert "ERRORS UNCORRELATED ACROSS RATERS" in text
    assert "THE CURVE IS AN UPPER BOUND" in text
    assert "buy LESS than the formula says" in text
    assert "Only k = 5 is a MEASUREMENT" in text

    # The population trap, named rather than assumed known.
    assert "NOT the 251-row population" in text
    assert "0.903 ceiling" in text and "0.807" in text

    # Reliability and ceiling kept apart.
    assert "DIFFERENT QUANTITIES" in text
    assert "never reported as the same column" in text

    # And the readings' provenance.
    assert "THE READINGS WERE NOT BLIND" in text
    assert "0.8416 at k = 6 BEFORE the readings fired" in text


def test_which_reading_fired_and_the_nuance_it_did_not_anticipate():
    record = phase24.THE_CURVE_OBSERVED
    fired = _flat(record["which_reading_fired"])
    assert "(b)" in fired
    assert "0.90 needs 11 raters" in fired
    assert "more than DOUBLE" in fired
    assert "reliability is expensive" in fired

    nuance = _flat(record["and_the_honest_nuance_the_readings_did_not_anticipate"])
    assert "TARGET-DEPENDENT" in nuance
    assert "reading (a) fires at 0.85" in nuance
    assert "Both readings are true, at different targets" in nuance
    assert "saying only (b) would overstate the cost" in nuance

    assert "restated here where the numbers are" in _flat(
        record["the_readings_were_NOT_blind"]
    )
    assert record["tag"].startswith("[MEASURED]")
    assert "no ledger row" in record["tag"]


def test_the_new_same_digit_collision_is_real_and_recorded():
    """Reliability at k=10 (237) and PCC_CEILING_251 are both 0.8985."""
    from cleft.data import reliability as R

    at_ten = R.spearman_brown(R.MEAN_R_237, 10)
    assert round(at_ten, 4) == 0.8985
    assert R.PCC_CEILING_251 == 0.8985
    # Same digits, genuinely different quantities.
    assert at_ten != R.PCC_CEILING_251

    recorded = _flat(phase24.THE_CURVE_OBSERVED[
        "A_NEW_SAME_DIGIT_COLLISION_FOUND_IN_THIS_PHASES_OWN_OUTPUT"
    ])
    assert "0.8985" in recorded
    assert "different quantities on BOTH axes" in recorded
    assert "The fourth instance of the same-digit family" in recorded


def test_nothing_banked_moved_and_the_ledger_is_untouched():
    from cleft import results_ledger
    from cleft.data import reliability as R

    assert R.MEAN_R_237 == 0.4696
    assert R.RELIABILITY_237 == 0.8158
    assert R.PCC_CEILING_237 == 0.9032
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 24 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p24" or "p24-" in e["id"]
    ]
    results_ledger.validate()
    assert "none" in _flat(phase24.THE_CURVE_OBSERVED["no_banked_figure_moved"])
