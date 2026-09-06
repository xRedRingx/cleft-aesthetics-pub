"""Phase 23 -- the statistical instruments, the ROPE half only.

Registration only. Nothing is built: no task, no schema kind, no config,
no run. A negative-space test holds that state, and the exit criteria
are a DRAFT rather than a lock.

Every figure the restate cites is checked against the record that holds
it, and the parameters taken from the source are separated from the one
this project supplies.
"""

from __future__ import annotations

from pathlib import Path

import pathlib

import pytest

from cleft import ladder, literature, phase21, phase23
from cleft.config import schema

REPO = Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# the reckoning
# --------------------------------------------------------------------------


def test_the_reckoning_states_the_limit_in_the_banks_own_words():
    record = phase23.PHASE_23_RECKONING
    limit = _flat(record["what_it_does_not_do"])
    assert "changes NO FIGURE and resolves NOTHING" in limit
    # Quoted, not paraphrased -- and the quote is live at its source.
    quote = "it does not RESOLVE anything the cohort cannot resolve"
    assert quote in limit
    assert quote in literature.SOURCES_BANKED["benavoli_rope"][
        "what_it_does_not_do"
    ]
    assert "restate the limit in better language, not lift it" in limit

    does = _flat(record["what_it_does"])
    assert "does not distinguish 'these two arms are the same' from" in does
    assert "a conclusion NHST cannot reach" in does


def test_every_figure_the_reckoning_cites_is_live():
    from collections import Counter

    from cleft import results_ledger

    holds = _flat(phase23.PHASE_23_RECKONING["what_the_record_holds"])
    assert "0.04 to 0.10" in holds and "30 tested, 1 survived" in holds
    assert "0.04 to 0.10" in _flat(
        ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"]["unresolvable_band"]
    )
    assert "0.1386" in holds
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"]["delta"] == 0.1386

    # Nineteen rows: 8 + 11, counted at source.
    assert "Nineteen ledger rows" in holds
    statuses = Counter(str(e.get("status")) for e in results_ledger.ENTRIES)
    assert statuses["UNRESOLVED-WITHDRAWN"] == 8
    assert statuses["WITHDRAWN"] == 11
    assert "8 UNRESOLVED-WITHDRAWN and 11 WITHDRAWN" in holds
    assert hasattr(phase21, "THE_SEVENTEEN_WAS_NEVER_COUNTED")


def test_coverage_is_a_clean_negative_and_says_why_n_excluding_zero_is_not_it():
    coverage = _flat(phase23.PHASE_23_RECKONING["coverage_nothing_to_concede"])
    assert "nothing in the record computes a P(A>B)" in coverage
    assert "`n_excluding_zero` is NOT P(A>B)" in coverage
    assert "count of SEEDS on a FIXED SPLIT" in coverage
    assert "paired indicator over INDEPENDENT SPLITS" in coverage
    # The honest-audit note, and its cited precedent is live.
    assert "CAN come back positive" in _flat(
        phase23.PHASE_23_RECKONING["the_honest_form_of_the_audit"]
    )
    assert "ENSEMBLE_COMBINES" in phase21.PHASE_21_RECKONING["a_is_not_covered"]


# --------------------------------------------------------------------------
# the half that is not built
# --------------------------------------------------------------------------


def test_the_split_randomised_half_is_refused_for_a_measured_reason():
    import numpy as np

    from cleft.data import folds

    record = phase23.SPLIT_RANDOMISED_HALF_NOT_BUILT
    reason = _flat(record["the_measured_reason"])
    assert "UNREACHABLE BY PARAMETER" in reason
    assert "60 of 60 patients in their fold" in reason
    assert "DIFFERENT SPLITTER, not a seed here" in reason

    # Re-run the measurement rather than trusting the sentence.
    rng = np.random.default_rng(0)
    ids = list(range(1, 61))
    labels = rng.integers(0, 3, 60).tolist()
    a = folds.generate(patient_ids=ids, labels=labels, seed=1337)
    b = folds.generate(patient_ids=ids, labels=labels, seed=2024)
    assert a.assignments == b.assignments
    # And the frozen module says it itself.
    source = (REPO / "src" / "cleft" / "data" / "folds.py").read_text(
        encoding="utf-8"
    )
    assert "does not affect the partition" in source
    assert "shuffle=False" in source

    cost = _flat(record["and_the_cost_from_the_banks_own_N"])
    assert "58 runs" in cost and "1,972 runs" in cost
    assert 29 * 2 == 58 and 68 * 29 == 1972
    assert "no citation by itself justifies" in cost

    # It is unregistered rather than merely postponed.
    assert "EXPLICITLY UNREGISTERED -- no fishing" in _flat(
        record["registered_so_it_cannot_return_as_a_fresh_idea"]
    )
    # And the limitation it does NOT fix is stated.
    stays = _flat(record["what_is_therefore_true_and_stays_true"])
    assert "does not fix that" in stays
    assert "FixHOptEst" in stays


# --------------------------------------------------------------------------
# the method: the source's parameters, and the one that is ours
# --------------------------------------------------------------------------


def test_every_method_parameter_is_attributed_to_the_source():
    record = phase23.THE_METHOD_REGISTERED
    prior = _flat(record["the_prior_from_the_paper"])
    assert "mu_0 = 0, k_0 -> infinity, a = -1/2, b = 0" in prior
    assert "Taken from the paper, not chosen here" in prior
    assert "St(mu; n-1, xbar, (1/n + rho/(1-rho)) * sigmahat^2)" in prior

    rho = _flat(record["rho_from_the_fold_ratio"])
    assert "rho = 0.2" in rho
    # The paper's practice, and ours is the same construction smaller.
    assert "rho = 1/10 with n = 100, from 10 runs of 10-fold" in rho
    assert "rho = 1/5 with n = 25, from 5 runs of 5-fold" in rho
    assert 1 / 5 == 0.2 and 10 * 10 == 100 and 5 * 5 == 25

    # [UPDATED 2026-09-02] These two asserted the ORIGINAL wording,
    # which the dated correction PRESERVES -- so they passed while
    # saying the opposite of what the record now claims. Both now
    # assert the CORRECTION, with the original checked as preserved.
    rule = _flat(record["the_decision_rule_and_its_derivation"])
    assert "THIS IS OURS, NOT THE PAPER'S" in rule
    assert "introduces the matrix as an EXAMPLE" in rule
    assert "Consider for instance the following loss matrix" in rule
    assert "THE_DECISION_THRESHOLD_RULED" in rule
    # the original, preserved
    assert "TAKEN FROM THE PAPER rather than chosen" in rule

    ours = _flat(record["what_this_project_supplies"])
    assert "TWO, not one" in ours
    assert "ROPE width (0.018346) AND the decision threshold (0.95)" in ours
    # [UPDATED 2026-09-02] Full precision here too: this is a LIVE
    # claim about what the project supplies, not a preserved original.
    assert "Four remain the source's" in ours
    # the original, preserved
    assert "the ROPE width, and nothing else" in ours


def test_the_observation_vector_is_twenty_five_per_contrast():
    vector = _flat(phase23.THE_METHOD_REGISTERED["the_observation_vector"])
    assert "25 per contrast" in vector
    assert "5 folds x 5 seeds" in vector
    # The CSVs really carry a fold column, so per-fold PCC is computable.
    from cleft.cluster_csv import PREDICTIONS_COLUMNS

    assert "fold" in PREDICTIONS_COLUMNS
    assert list(PREDICTIONS_COLUMNS) == [
        "patient_id", "truth", "prediction", "fold"
    ]


# --------------------------------------------------------------------------
# the ROPE
# --------------------------------------------------------------------------


def test_the_rope_width_reproduces_from_its_named_inputs():
    from cleft.train.phase3 import combined_claimable_delta

    record = phase23.THE_ROPE_RULED
    # [UPDATED 2026-09-02] Full precision, with the rounded original
    # preserved and marked as the form the ruling was issued in.
    assert "+/- 0.018346" in _flat(record["the_width"])
    assert "CORRECTED 2026-09-02" in _flat(record["the_width"])
    assert "ORIGINAL: '+/- 0.0183'" in _flat(record["the_width"])
    threshold = combined_claimable_delta(0.0148, 5, 0.0148, 5)
    assert threshold["arm_means_95"] == 0.018346
    assert "recomputed at run time rather than pinned as a literal" in _flat(
        record["the_width"]
    ).lower()

    ground = _flat(record["the_ground"])
    assert "smaller than what the criterion can DISTINGUISH at five seeds" in ground
    assert "Closer than we can measure" in ground


def test_the_four_rejected_widths_are_unregistered_with_their_values():
    from cleft.train.phase3 import MEASURED_SEED_BAND, combined_claimable_delta

    rejected = _flat(
        phase23.THE_ROPE_RULED["the_four_rejected_candidates_unregistered"]
    )
    assert "EXPLICITLY UNREGISTERED -- no fishing" in rejected
    for value in ("0.0137", "0.0410", "0.04-0.10", "0.1386"):
        assert value in rejected, value
    # Each value is real at its source.
    assert MEASURED_SEED_BAND["sd"] == 0.0137
    assert round(
        combined_claimable_delta(0.0148, 5, 0.0148, 5)["single_run_95"], 4
    ) == 0.0410
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"]["delta"] == 0.1386
    # And each carries a reason, not just a value.
    assert "a spread, not a distinguishability threshold" in rejected
    assert "two SINGLE runs" in rejected
    assert "a RANGE rather than a width" in rejected
    assert "far too wide" in rejected


def test_a_per_contrast_width_is_refused_with_its_arithmetic():
    why = _flat(phase23.THE_ROPE_RULED["why_not_a_per_contrast_width"])
    assert "EIGHTEEN DIFFERENT ROPEs" in why
    assert "fixed domain-level width" in why
    assert "incomparable with each other" in why
    assert len(phase23.THE_ROWS_SCOPED["the_count"].split()) > 0
    assert "17" in phase23.THE_ROWS_SCOPED["the_count"]


def test_the_width_discipline_is_recorded_as_entirely_this_projects():
    record = phase23.THE_ROPE_RULED
    ours = _flat(record["the_paper_gives_no_guidance_and_that_is_ours_to_carry"])
    assert "addresses neither how to choose the width, nor post-hoc" in ours
    assert "FIXED BEFORE ANY POSTERIOR IS COMPUTED" in ours
    assert "one width for all eighteen rows" in ours
    # Verified against the bank: it really holds nothing on this.
    entry = literature.SOURCES_BANKED["benavoli_rope"]
    assert not any(
        "width" in str(v).lower() for v in entry.values()
    ), "the bank now says something about the width; re-read the entry"


# --------------------------------------------------------------------------
# extensions, silences, and the excluded model
# --------------------------------------------------------------------------


def test_the_four_extensions_and_silences_are_named():
    record = phase23.DECLARED_EXTENSIONS_AND_SILENCES
    numbered = [k for k in record if k[0].isdigit()]
    assert len(numbered) == 4
    assert sorted(int(k.split("_")[0]) for k in numbered) == [1, 2, 3, 4]

    pcc = _flat(record["1_pcc_is_a_non_additive_metric"])
    assert "an EXTENSION, not precedent" in pcc
    assert "places no restriction on how each fold's number was produced" in pcc
    assert "Nadeau & Bengio" in pcc
    assert "does **not** derive its variance on observation-wise losses" in pcc

    small_n = _flat(record["3_small_n_is_never_discussed"])
    assert "n = 100" in small_n and "ours is 25" in small_n
    assert "no assurance at this size and none is claimed" in small_n

    rho = _flat(record["4_extreme_rho_is_not_warned_about"])
    assert "rho/(1-rho) = 0.25" in rho
    assert 0.2 / (1 - 0.2) == 0.25


def test_the_hierarchical_model_is_excluded_with_its_assumption_quoted():
    record = phase23.THE_HIERARCHICAL_MODEL_IS_EXCLUDED
    assumption = _flat(record["the_assumption_quoted"])
    assert "unit of replication is the DATASET" in assumption
    assert "independent exchangeable units" in assumption

    why = _flat(record["why_it_does_not_apply"])
    assert "eighteen contrasts on ONE cohort are not eighteen datasets" in why
    assert "237 patients, one manifest, one fold structure" in why

    caution = _flat(record["the_paper_does_not_prohibit_the_misuse"])
    assert "reason for caution rather than permission" in caution
    assert "has not licensed it" in caution


# --------------------------------------------------------------------------
# scope and pairing
# --------------------------------------------------------------------------


def test_seventeen_rows_with_three_recovered_and_two_excluded():
    record = phase23.THE_ROWS_SCOPED
    recovered = record["three_recovered_by_reading_or_measurement_never_by_matching"]
    assert sorted(recovered) == [
        "p7b_claimably_worse", "p7c_nine_verdicts", "roadb_region_crop",
    ]
    assert "19 - 2 = **17 rows in scope**, 45 contrasts" in record["the_count"]

    # Each recovery names HOW it was found, and none by pattern.
    p7c = _flat(recovered["p7c_nine_verdicts"])
    assert "26be5779" in p7c and "READ from" in p7c
    assert "not chosen by plausibility" in p7c
    roadb = _flat(recovered["roadb_region_crop"])
    assert "52875413" in roadb and "READ from" in roadb
    assert "anatomy_concat_srgnn, runs TEN seeds" in roadb
    p7b = _flat(recovered["p7b_claimably_worse"])
    assert "p7b_search__eb4477d9__p7b-search-3" in p7b
    assert "IDENTIFIED BY MEASUREMENT" in p7b
    assert "-0.05428245764369638" in p7b
    assert "Not matched by sha or suffix" in p7b


def test_p8_is_excluded_in_principle_not_for_missing_runs():
    excluded = _flat(phase23.THE_ROWS_SCOPED["p8_excluded_in_principle"])
    assert "NOT restatable by a ROPE, and not because its runs are missing" in excluded
    assert "no per-seed PCC vector exists" in excluded
    assert "no condition_1 or condition_2" in excluded
    assert "SPATIAL MASS over n = 15" in excluded
    assert "REFUTED BY RE-MEASUREMENT" in excluded
    assert "runs surviving is beside the point" in excluded

    # The ledger row really carries no conditions.
    from cleft import results_ledger

    row = next(
        e for e in results_ledger.ENTRIES
        if e.get("id") == "p8-framing-not-anatomy"
    )
    assert not row.get("condition_1") and not row.get("condition_2")



PAIRING_EXPRESSION = (
    'assignments = {int(r["patient_id"]): int(r["fold"]) for r in rows}'
)


def _pairing_line() -> int:
    """Where the fold-assignment expression actually sits, 1-indexed.

    **[2026-09-02] Derived rather than hardcoded.** The record cited line
    209; Phase 25 registered two backbones above it and the expression
    moved. The mechanism did not change, and a line number that only a
    human re-reads is a citation that rots silently.
    """
    lines = (REPO / "src" / "cleft" / "train" / "phase3.py").read_text(
        encoding="utf-8"
    ).splitlines()
    for index, line in enumerate(lines):
        if PAIRING_EXPRESSION in line:
            return index + 1
    raise AssertionError("the fold-assignment expression is gone from phase3")

def test_pairing_is_established_by_construction_with_three_qualifications():
    record = phase23.PAIRING_ESTABLISHED_BY_CONSTRUCTION
    mechanism = _flat(record["the_mechanism"])
    assert "READ FROM THE MANIFEST'S `fold` COLUMN" in mechanism
    # **The cited number is DERIVED, not trusted** -- see the dated
    # correction in the record: an edit above this expression moves it.
    assert f"line {_pairing_line()}" in mechanism
    assert "BYTE-IDENTICAL folds by construction" in mechanism
    assert "no arm generates anything" in mechanism

    # The cited line really is that expression.
    lines = (REPO / "src" / "cleft" / "train" / "phase3.py").read_text(
        encoding="utf-8"
    ).splitlines()
    assert PAIRING_EXPRESSION in lines[_pairing_line() - 1]

    qualifications = [k for k in record if k.startswith("qualification_")]
    assert len(qualifications) == 3
    assert "SHARED FIVE" in _flat(record["qualification_1_ten_seed_arms"])
    assert "DIFFERENT MEMBER SET" in _flat(record["qualification_2_the_236_cohort"])
    assert "CANNOT be paired" in _flat(
        record["qualification_3_partition_sensitivity"]
    )
    # views.py really carries the folds rather than re-stratifying.
    views = (REPO / "src" / "cleft" / "data" / "views.py").read_text(
        encoding="utf-8"
    )
    assert "Why folds are carried, not re-stratified" in views


# --------------------------------------------------------------------------
# readings and normality
# --------------------------------------------------------------------------


def test_the_readings_are_committed_before_any_number():
    record = phase23.READINGS_COMMITTED
    assert "before any posterior is computed" in record["committed"]

    equivalent = _flat(record["p_rope_above_0_95"])
    assert "PRACTICALLY EQUIVALENT" in equivalent
    assert "NHST CANNOT REACH" in equivalent

    different = _flat(record["p_left_or_p_right_above_0_95"])
    assert "WITHDRAWN under PLAN 4.3" in different
    assert "on the same data" in different
    assert "The banked verdict does not change" in different
    assert "neither overturning the other" in different

    undecided = _flat(record["none_above_0_95"])
    assert "NO DECISION" in undecided
    assert "all three probabilities are reported regardless" in undecided
    assert "Table 9" in undecided


def test_the_expected_outturn_and_the_distinction_it_must_not_blur():
    record = phase23.READINGS_COMMITTED
    expectation = _flat(record["the_registered_expectation"])
    assert "most rows are expected to return NO DECISION" in expectation
    assert "22% of NHST's failed rejections cleared the ROPE at n = 100" in expectation
    assert "a lower rate should be expected" in expectation
    assert "not a disappointment reframed" in expectation

    distinction = _flat(record["no_decision_is_NOT_equivalence"])
    assert "WIDE posterior means UNCERTAINTY" in distinction
    assert "NARROW posterior sitting inside the ROPE means SIMILARITY" in distinction
    assert "the paper distinguishes them explicitly" in distinction
    assert "in Bayesian clothing" in distinction


def test_normality_is_described_and_gates_nothing():
    record = phase23.NORMALITY_DESCRIBED_NOT_TESTED
    assert "no threshold, no test, no branch" in _flat(record["it_gates_nothing"])
    why = _flat(record["why_it_may_not_gate_anything"])
    assert "choosing a method after seeing data" in why
    assert "Describing is honest; branching is not" in why
    # The degenerate-fold caution is named in advance.
    caution = _flat(record["a_caution_worth_recording_in_advance"])
    assert "65.3% degenerate cells" in caution
    assert "not a surprise that invites a mid-analysis decision" in caution


# --------------------------------------------------------------------------
# the draft, the settings, and negative space
# --------------------------------------------------------------------------


def test_the_exit_criteria_are_a_draft_with_eleven_numbered():
    record = phase23.EXIT_CRITERIA_DRAFT
    assert "NOT LOCKED" in record["drafted"]
    assert record["tag"].startswith("[DRAFT]")
    numbered = [k for k in record if k[0].isdigit()]
    assert len(numbered) == 11
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 12))
    # The load-bearing ones.
    assert "every result is withdrawn" in _flat(
        record["2_the_rope_is_fixed_before_any_posterior"]
    )
    assert "no ledger row is edited" in _flat(
        record["7_the_banked_verdicts_are_unchanged"]
    )
    assert "never written as equivalence" in record[
        "6_no_decision_is_never_written_as_equivalence"
    ] or "checked against the write-up text" in _flat(
        record["6_no_decision_is_never_written_as_equivalence"]
    )


def test_the_settings_separate_ours_from_the_sources():
    record = phase23.SETTINGS_TO_DECLARE
    numbered = [k for k in record if k[0].isdigit()]
    assert len(numbered) == 6
    assert "0.0183" in record["1_rope_half_width"]
    assert "0.2" in record["2_rho"]
    assert "0.95" in record["3_decision_threshold"]
    assert "the paper's" in record["3_decision_threshold"]
    assert "NEVER tuned across runs" in _flat(record["the_standing_clause"])
    # A derivation is not a config field.
    assert "invite editing a derivation" in _flat(record["what_is_NOT_a_setting"])


def test_exactly_the_ruled_phase_23_machinery_is_built():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held the
    phase at registration only. The lock ruled the build, so the pin
    now holds what it was standing for: EXACTLY the ruled machinery --
    one task, one config, and the split-randomised half still absent."""
    from cleft import results_ledger
    from cleft.run import TASKS

    assert "p23_rope" in TASKS
    assert "p23_rope" in schema.TASK_SPECS
    assert sorted(
        p.name for p in (REPO / "configs").glob("p23*.yaml")
    ) == ["p23_rope.yaml"]

    # The half that is NOT built stays not built.
    for kind in list(TASKS) + list(schema.TASK_SPECS):
        for token in ("split_random", "p_a_beats_b", "bouthillier"):
            assert token not in kind.lower(), (kind, token)

    # phase23 still holds records and one accessor; the arithmetic
    # lives in cleft.rope and the plumbing in run.py.
    callables = [
        n for n in dir(phase23)
        if callable(getattr(phase23, n)) and not n.startswith("_")
    ]
    assert callables == ["summary"], callables
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 23 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p23" or "p23-" in e["id"]
    ]
    assert results_ledger.validate() is None


def test_the_summary_is_the_whole_registration():
    assert sorted(phase23.summary()) == [
        "attributions_corrected",
        "closing",
        "completion_to_forty_five",
        "decision_threshold",
        "degenerate_fold",
        "drop_counted_not_recorded",
        "exit_criteria",
        "exit_criteria_draft",
        "expectation_held",
        "extensions_and_silences",
        "fifteen_not_enumerable",
        "forty_five_outturn",
        "hierarchical_excluded",
        "iem_row_excluded",
        "method",
        "normality",
        "normality_observed",
        "observed",
        "pairing",
        "readings",
        "reckoning",
        "rerun_citable",
        "rope",
        "rounded_quotation_sweep",
        "rows_scoped",
        "settings",
        "split_randomised_not_built",
        "substantive_finding",
        "sweep_presence_not_content",
        "two_decisions",
        "two_further_decisions",
        "width_precision_corrected",
    ]
    for key, value in phase23.summary().items():
        assert isinstance(value, dict), key
        assert "tag" in value, key


# --------------------------------------------------------------------------
# [2026-09-02] The banked passages, the corrected attributions, the rulings
# --------------------------------------------------------------------------


def test_nine_passages_are_banked_with_their_section_numbers():
    """Every Phase 23 parameter must cite a quote, not a relayed
    summary. The request called them eight and listed nine."""
    entry = literature.SOURCES_BANKED["benavoli_rope"]
    passages = entry["passages"]
    numbered = [k for k in passages if k[0].isdigit()]
    assert len(numbered) == 9
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 10))

    blob = " ".join(str(v) for v in passages.values())
    for quote in (
        "mu_0 = 0, k_0 -> infinity, a = -1/2, b = 0",
        "St(mu; n-1, xbar, (1/n + rho/(1-rho)) sigmahat^2)",
        "the vector of differences of accuracy",
        "rho-hat = 0 regardless the observations",
        "the folds are designed not to overlap",
        "In classification 1% seems to be a reasonable choice",
        "the integral of the posterior over the rope interval",
        "n = 100 (10 runs of 10-fold cross-validation)",
        "Since 0.05 * 20 = 1",
    ):
        assert quote in blob, quote

    # Provenance says plainly that the paper is not reachable here.
    assert "NOT reachable from this machine" in passages["provenance"]
    assert "1606.04316" in entry["source"]


def test_the_five_unverified_attributions_are_recorded_with_their_mechanism():
    record = phase23.THE_FIVE_SOURCE_ATTRIBUTIONS_WERE_UNVERIFIED
    whose = _flat(record["whose_error"])
    assert "the record's" in whose
    assert "CONVERSATIONAL RELAY" in whose
    assert "no citable entry behind them" in whose

    shape = _flat(record["the_shape_it_repeats"])
    assert "the 0.068 and the 17" in shape
    assert "Third instance of citation-without-entry" in shape
    # It cost nothing measured, and that is said.
    assert "nothing measured" in _flat(record["what_it_cost"])
    assert "verify its own method section" in _flat(record["what_it_cost"])


def test_rho_stays_source_attributed_and_carries_its_approximation():
    record = phase23.THE_METHOD_REGISTERED
    why = _flat(record["rho_is_not_a_choice_it_is_unidentifiable"])
    assert "cannot be estimated" in why or "does not allow to estimate" in why
    assert "rho-hat = 0 regardless the observations" in why
    assert "no version of this analysis in which rho is fitted" in why

    caveat = _flat(record["and_its_approximation_travels_with_it"])
    assert "the folds are designed not to overlap" in caveat
    assert "applied outside the case it was derived for" in caveat
    assert "travels with every rho = 0.2 in this phase" in caveat


def test_the_decision_threshold_is_ruled_as_ours_with_its_ground():
    record = phase23.THE_DECISION_THRESHOLD_RULED
    assert "20:1 loss ratio" in _flat(record["the_ruling"])
    assert "P(.) > 0.95" in _flat(record["the_ruling"])
    assert 0.05 * 20 == 1

    ground = _flat(record["the_ground"])
    assert "twenty times worse than no conclusion" in ground
    assert "one result in thirty" in ground
    # The flattering direction is named.
    flatters = _flat(record["and_the_direction_that_flatters_is_named"])
    assert "declare equivalence more readily" in flatters
    assert "against the phase's own interest" in flatters
    # The paper supplies form, not authority.
    assert "supplies the FORM and the arithmetic" in _flat(
        record["the_papers_matrix_is_its_basis_not_its_authority"]
    )


def test_the_iem_row_is_excluded_on_units_with_both_alternatives_refused():
    record = phase23.THE_IEM_ROW_EXCLUDED
    why = _flat(record["why_the_rope_cannot_be_applied"])
    assert "+/- 0.0183 is a PCC width" in why
    assert "opposite direction of good" in why
    assert "two quantities under one width" in why
    assert "R2 error" in why

    one = _flat(record["alternative_1_a_second_iem_derived_width_NOT_TAKEN"])
    assert "break the fixed-domain-width ruling" in one
    assert "same relaxation under a different name" in one

    two = _flat(record["alternative_2_deriving_one_now_NOT_TAKEN"])
    assert "no banked IEM seed-variance figure exists" in two
    assert "at the moment that row was inconvenient to leave out" in two
    assert "derived to admit a specific case is not a threshold" in two

    door = _flat(record["the_door_left_open"])
    assert "INDEPENDENTLY DERIVED IEM width" in door
    assert "footnote 3" in door
    assert "a different value could be more suitable" in door
    # The footnote it leans on is banked.
    assert "a different value could be more suitable" in " ".join(
        str(v) for v in literature.SOURCES_BANKED["benavoli_rope"][
            "passages"].values()
    )

    cost = _flat(record["what_it_costs"])
    assert "one contrast and ZERO declares" in cost
    assert "SAME two run directories" in cost


def test_the_scope_is_seventeen_rows_forty_five_contrasts_seventy_seven_dirs():
    record = phase23.THE_ROWS_SCOPED
    figures = _flat(record["the_three_figures_recomputed"])
    assert "17 rows, 45 contrasts, 77 distinct run directories" in figures
    assert "rows aggregate contrasts" in figures
    assert "roadb-resolution-withdrawn alone is 17" in figures
    assert "wrong by a factor of two and a half" in figures
    assert 45 / 18 > 2.4

    unchanged = _flat(record["the_run_directory_count_is_unchanged_by_the_exclusion"])
    assert "77 before and 77 after" in unchanged
    assert "shares p11_paired.yaml" in unchanged
    assert "above Phase 18's 68" in unchanged

    # The 91 was wrong and the record says why.
    wrong = _flat(record["an_earlier_count_of_91_was_wrong"])
    assert "Embedding artifacts share that shape" in wrong
    assert "over-planned the declare cycle by fourteen" in wrong
    assert 91 - 77 == 14


# --------------------------------------------------------------------------
# [2026-09-02] The lock, the task, the config, and the two sweeps
# --------------------------------------------------------------------------


def test_the_exit_criteria_are_locked_with_the_draft_preserved():
    locked = phase23.EXIT_CRITERIA
    assert locked["tag"] == "[LOCKED]"
    numbered = [k for k in locked if k[0].isdigit()]
    assert len(numbered) == 11
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 12))
    assert "17 rows, 45 contrasts, 77 run directories" in _flat(locked["scope"])
    # Two ours, four source.
    settings = locked["the_settings_as_ruled"]
    assert len([k for k in settings if k.startswith("ours_")]) == 2
    assert len([k for k in settings if k.startswith("source_")]) == 4
    # The draft is preserved unchanged beside it.
    assert phase23.EXIT_CRITERIA_DRAFT["tag"].startswith("[DRAFT]")
    assert len([k for k in phase23.EXIT_CRITERIA_DRAFT if k[0].isdigit()]) == 11


def test_criterion_2s_withdrawal_clause_is_enforced_not_merely_stated():
    """The strongest protection on the width: the task RECOMPUTES and
    refuses a config whose declared value disagrees."""
    import inspect

    from cleft import rope
    from cleft import run as run_module

    assert "EVERY RESULT IS WITHDRAWN" in _flat(
        phase23.EXIT_CRITERIA["2_the_rope_is_fixed_before_any_posterior"]
    )
    body = inspect.getsource(run_module.task_p23_rope)
    assert "rope.rope_half_width(" in body
    assert 'task["expect_rope_half_width"]' in body
    assert "raise ValueError(" in body
    assert "criterion 2" in body
    # And the recomputation really is a computation, not a constant.
    assert rope.rope_half_width(sd=0.0148, n_seeds=5) != rope.rope_half_width(
        sd=0.02, n_seeds=5
    )


def test_the_task_never_writes_no_decision_as_equivalence():
    """Criterion 6, as a tested literal."""
    from cleft import rope

    assert rope.VERDICTS[2] == "no decision"
    # Exhaustive over the probability simplex: no input produces
    # "practically equivalent" unless P(rope) strictly exceeds 0.95.
    for rope_p in (0.0, 0.5, 0.9, 0.949, 0.95):
        probs = {"left": (1 - rope_p) / 2, "rope": rope_p,
                 "right": (1 - rope_p) / 2}
        assert rope.verdict(probs, threshold=0.95) != "practically equivalent"
    assert rope.verdict(
        {"left": 0.02, "rope": 0.951, "right": 0.029}, threshold=0.95
    ) == "practically equivalent"


def test_the_config_has_no_placeholder_and_p7b_is_declared_here_first():
    """[UPDATED TWICE 2026-09-02, THIS PIN FIRED BOTH TIMES] It held
    zero placeholders at thirty contrasts, then exactly one when the
    completion added p7b's winner. **the declare filled it**, so
    the pin holds the end state: no placeholder remains, and p7b is the
    one entry this config declares FIRST rather than carries."""
    import yaml

    path = REPO / "configs" / "p23_rope.yaml"
    assert path.is_file()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    placeholders = [
        e["name"] for e in config["inputs"]
        if set(e["rollup_sha256"]) == {"0"}
    ]
    assert placeholders == [], placeholders

    # p7b's winner is declared HERE and in no other config, so it is
    # exempt from the byte-identity check below by construction rather
    # than by being unfilled.
    declared_here_first = {"p7b_search"}
    p7b = next(e for e in config["inputs"] if e["name"] == "p7b_search")
    assert p7b["rollup_sha256"] == (
        "10ea3efa083e7c025fa6ae1288b5b73e28c7e9cbe87f37ff7bfa67c2e92ba7da"
    )
    assert len(p7b["rollup_sha256"]) == 64
    assert set(p7b["rollup_sha256"]) <= set("0123456789abcdef")

    # Every non-manifest input is declared identically by some shipped
    # config -- carried verbatim, not rebuilt.
    shipped = {}
    for stem in (
        "p7_paired_ladder", "p7b_search", "p7c_paired_selected30",
        "p7d_paired", "roadb_p7_paired_resolution",
        "roadb_p7c_paired_regioncrop", "p10_paired", "p11_paired",
        "p12_paired", "p16_anchor_loop", "p17_family_analysis",
        # [2026-09-02] p18 declares the p16 anchor-loop RUN DIRECTORY,
        # which p16_anchor_loop.yaml itself does not -- it declares only
        # the probe. Omitting it made this check look like a mismatch.
        "p18_metric_space",
    ):
        for entry in yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )["inputs"]:
            shipped[entry["path"]] = entry["rollup_sha256"]
    for entry in config["inputs"]:
        if entry["name"] in declared_here_first:
            continue          # declared here first, so nowhere to match
        assert entry["path"] in shipped, entry["name"]
        assert entry["rollup_sha256"] == shipped[entry["path"]], entry["name"]


def test_the_config_no_longer_omits_anything():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held the
    config's own statement that it was incomplete at thirty. The
    completion removes the omission, so the pin now holds that the
    config carries the full forty-five and claims no exemption."""
    import yaml

    text = (REPO / "configs" / "p23_rope.yaml").read_text(encoding="utf-8")
    assert "FIFTEEN CONTRASTS ARE NOT IN THIS CONFIG" not in text
    config = yaml.safe_load(text)
    assert len(config["task"]["contrasts"]) == 45
    assert config["task"]["expect_contrasts"] == 45
    assert 30 + 15 == 45
    # All four previously-missing rows are present.
    rows = {c["row"] for c in config["task"]["contrasts"]}
    for row in ("p7c-nine-verdicts", "p17-a-vs-probe",
                "p16-anchor-loop-unresolved", "p7b-claimably-worse"):
        assert row in rows, row


def test_the_config_matches_its_generator():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/generate_phase23_config.py", "--check"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# ---- SWEEP 1: config -> code ---------------------------------------------


def test_every_declared_value_reaches_a_consumption_site():
    """The sweep the settings-consumption defect earned."""
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p23_rope)
    consumed = {
        "contrasts": 'task["contrasts"]',
        "seeds": 'task["seeds"]',
        "probe_seed_sd": 'task["probe_seed_sd"]',
        "probe_n_seeds": 'task["probe_n_seeds"]',
        "expect_rope_half_width": 'task["expect_rope_half_width"]',
        "rho": 'task["rho"]',
        "decision_threshold": 'task["decision_threshold"]',
        "full_n": 'task["full_n"]',
        "expect_contrasts": 'task["expect_contrasts"]',
    }
    for field, expression in consumed.items():
        assert expression in body, (field, expression)
    # Every spec field is accounted for -- nothing declared and unread.
    accounted = set(consumed) | {"kind", "expect_patients"}
    # And the field the sweep removed is gone from the spec entirely.
    assert "manifest_artifact" not in schema.TASK_SPECS["p23_rope"]
    assert set(schema.TASK_SPECS["p23_rope"]) <= accounted


# ---- SWEEP 2: code -> record ---------------------------------------------


def test_every_computed_value_reaches_the_record():
    """The mirror sweep the null-condition defect earned: a value the
    task computes must land in metrics.json, not only in a log line.

    **[UPDATED 2026-09-02. This sweep, and its sibling below, BOTH
    passed while the defect was live.]** They assert that
    ``observations`` is written. It was -- carrying nulls. A
    source-text sweep can see a key and never its content; the
    executed check is
    ``test_the_assembled_record_carries_the_drop_not_just_the_shortfall``
    (phase23.THE_SWEEP_CHECKED_PRESENCE_NOT_CONTENT).
    """
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p23_rope)
    written = body.split("tmp.write_text(")[1]
    for computed in (
        "rope_half_width", "rho", "decision_threshold", "n_contrasts",
        "verdict_counts", "contrasts", "attribution",
    ):
        assert f'"{computed}"' in written, computed
    # Per contrast: n, df, all three probabilities, the verdict and the
    # description all reach the result dict.
    per_contrast = inspect.getsource(run_module.p23_contrast_record)
    per_contrast = per_contrast.split("return {")[1]
    for field in ('"n"', '"degrees_of_freedom"', '"probabilities"',
                  '"verdict"', '"observations"'):
        assert field in per_contrast, field
    # n != 25 is visible in the record, not buried in a log.
    assert '"n_is_the_full_form"' in per_contrast
    # And so is the account of WHY it is not 25.
    assert '"cells_accounted"' in per_contrast


def test_nothing_banked_is_recomputed_by_the_task():
    """Criterion 7: the ledger is untouched."""
    import inspect

    from cleft import results_ledger
    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p23_rope)
    assert "results_ledger" not in body
    assert "paired_comparison" not in body
    # combined_claimable_delta is reached only through rope, never
    # called here; the mention in the docstring does not count.
    code = chr(10).join(
        line for line in body.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "combined_claimable_delta(" not in code
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 23 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p23" or "p23-" in e["id"]
    ]
    assert results_ledger.validate() is None


def test_the_guard_refuses_a_width_differing_in_the_seventh_decimal():
    """**No tolerance, on either side.** The guard's purpose is refusing
    a width nobody ruled; slack in it defeats it, which is why the
    config carries the formula's own value rather than a display of it.
    """
    import inspect

    from cleft import rope
    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p23_rope)
    # An exact comparison, not a tolerance.
    assert "if half_width != declared_width:" in body
    # Scan the CODE, not the comment explaining what it replaced.
    code = chr(10).join(
        line for line in body.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "5e-7" not in code and "abs(half_width" not in code
    assert "EXACT, no tolerance on either side" in body

    exact = rope.rope_half_width(sd=0.0148, n_seeds=5)
    # A seventh-decimal difference is a different number, and the guard
    # compares numbers, not roundings.
    assert exact != exact + 1e-7
    assert exact != 0.0183, "the rounded display is NOT the quantity"
    assert round(exact, 4) == 0.0183


def test_the_declared_width_is_identical_to_the_recomputed_one():
    """Not 'agrees to four places' -- the same number."""
    import yaml

    from cleft.train.phase3 import combined_claimable_delta

    config = yaml.safe_load(
        (REPO / "configs" / "p23_rope.yaml").read_text(encoding="utf-8")
    )
    declared = config["task"]["expect_rope_half_width"]
    assert declared == combined_claimable_delta(
        0.0148, 5, 0.0148, 5
    )["arm_means_95"]
    assert declared == 0.018346


def test_the_width_correction_records_its_provenance_and_the_rejected_option():
    record = phase23.THE_WIDTH_WAS_A_ROUNDED_DISPLAY
    what = _flat(record["what_happened"])
    assert "0.0183460 against a declared 0.0183000" in what
    assert "before any posterior existed" in what or "nothing had to be withdrawn" in what

    whose = _flat(record["whose_error"])
    assert "the record's" in whose
    assert "FOUR-DECIMAL DISPLAY" in whose
    assert "as if it were the value" in whose

    family = _flat(record["the_family_it_belongs_to"])
    assert "the 0.068" in family and "the 17" in family
    assert "caught: by MACHINERY, not by reading" in family

    # The ruling is unchanged; only its expression.
    assert "a QUANTITY, not a numeral" in _flat(
        record["the_ruling_is_unchanged_only_its_expression"]
    )
    # And why seven digits, for the later reader.
    rejected = _flat(record["the_alternative_rejected"])
    assert "slack in it defeats it" in rejected
    assert "wonder why the config carries seven digits" in rejected


def test_the_rounded_quotation_sweep_reports_its_one_finding():
    """The same shape elsewhere -- found, and deliberately not fixed."""
    import math

    import yaml

    from cleft.data import reliability

    record = phase23.ROUNDED_QUOTATION_SWEEP
    assert "targeted check and not an exhaustive one" in _flat(
        record["the_method_and_its_limit"]
    )

    # The finding is real: declared 0.398942, exact 0.3989418.
    exact = 0.657279 * math.sqrt(2.0) * math.sqrt(
        1.0 - float(reliability.RELIABILITY_237)
    )
    declared = yaml.safe_load(
        (REPO / "configs" / "p22_r_clear_bounded.yaml").read_text(
            encoding="utf-8")
    )["task"]["se_diff_threshold"]
    assert declared == 0.398942
    assert declared != exact
    assert abs(declared - exact) < 5e-6, "absorbed by p22's tolerance"

    # And it is NOT changed, for a stated reason.
    why = _flat(record["and_it_is_NOT_being_changed"])
    assert "arms have already RUN against this config" in why
    assert "STANDING ITEM for the maintainer, not acted on" in why
    assert "multiples of 0.2" in why

    # The two guards differ because the two quantities do.
    differ = _flat(record["why_p23s_guard_is_exact_and_p22s_is_not"])
    assert "derived from two CONSTANTS" in differ
    assert "float noise between runs" in differ


# --------------------------------------------------------------------------
# [2026-09-02] The close-out
# --------------------------------------------------------------------------


def test_the_verdict_counts_sum_to_the_contrasts_that_ran():
    import yaml

    record = phase23.ROPE_OBSERVED
    verdicts = record["verdicts"]
    assert verdicts == {
        "practically equivalent": 0,
        "practically different": 2,
        "no decision": 28,
    }
    assert sum(verdicts.values()) == 30
    # [UPDATED 2026-09-02] The BANKED outturn is the thirty-contrast
    # run's. The config has since been completed to forty-five and is
    # awaiting a rerun that REPLACES this result rather than adding to
    # it, so the two counts differ on purpose.
    config = yaml.safe_load(
        (REPO / "configs" / "p23_rope.yaml").read_text(encoding="utf-8")
    )
    assert config["task"]["expect_contrasts"] == 45
    assert sum(verdicts.values()) == 30, "the run that produced these"
    assert config["task"]["full_n"] == 25
    assert "all thirty at n = 25, df 24" in _flat(
        record["every_contrast_at_the_full_form"]
    )
    provenance = _flat(record["provenance"])
    assert "CLUSTER-ONLY" in provenance
    assert "What WAS verified on this machine" in provenance


def test_the_highest_p_rope_is_pinned_and_nothing_is_equivalent():
    record = phase23.THE_SUBSTANTIVE_FINDING
    answer = _flat(record["the_answer"])
    assert "NOT ONE of the thirty contrasts is demonstrably the same" in answer
    assert "0.3271" in answer
    assert "srgnn imagenet 512->768" in answer
    assert 0.3271 < 0.95
    assert phase23.ROPE_OBSERVED["verdicts"]["practically equivalent"] == 0

    establishes = _flat(record["what_it_establishes"])
    assert "region of IGNORANCE, not a region of SAMENESS" in establishes
    assert "could not previously make" in establishes

    does_not = _flat(record["what_it_does_NOT_establish"])
    assert "no contrast is shown to be NON-equivalent either" in does_not
    assert "is not 'shown to differ'" in does_not


def test_the_two_decisions_are_pinned_with_their_probabilities():
    import yaml

    record = phase23.THE_TWO_DECISIONS
    first = record["1_vit_b16_imagenet_224_to_768"]
    second = record["2_p10_replication_vs_probe"]
    assert first["p_right"] == 0.9993
    assert second["p_left"] == 0.9817
    assert first["verdict"] == second["verdict"] == "practically different"
    assert first["p_right"] > 0.95 and second["p_left"] > 0.95

    config = yaml.safe_load(
        (REPO / "configs" / "p23_rope.yaml").read_text(encoding="utf-8")
    )
    keys = {c["key"] for c in config["task"]["contrasts"]}
    assert first["key"] in keys
    assert second["key"] in keys

    means = _flat(first["what_it_means"])
    assert "REGISTERED INSTRUMENTS-DISAGREE CASE" in means
    assert "banked verdict is UNCHANGED" in means
    assert "WITHDRAWN under PLAN 4.3" in _flat(
        phase23.READINGS_COMMITTED["p_left_or_p_right_above_0_95"]
    )


def test_the_near_misses_are_observations_not_verdicts():
    near = _flat(
        phase23.THE_TWO_DECISIONS["the_two_near_misses_as_observations"]
    )
    assert "0.9328" in near and "0.9132" in near
    assert "decided at 0.90, not at the ruled 0.95" in near
    assert "the threshold is OURS" in near
    assert "Neither is a verdict" in near
    for value in (0.9328, 0.9132):
        assert 0.90 < value < 0.95


def test_the_registered_expectation_fired():
    record = phase23.THE_EXPECTATION_HELD
    assert "22%" in _flat(record["what_was_registered"])
    landed = _flat(record["what_landed"])
    assert "28 of 30 no decision" in landed
    assert "ZERO cleared the ROPE" in landed
    assert "lower rate than Benavoli" in landed
    assert "22% of NHST" in _flat(
        phase23.READINGS_COMMITTED["the_registered_expectation"]
    )
    assert "not a disappointment reframed" in _flat(
        record["why_this_matters_procedurally"]
    )


def test_the_fifteen_missing_are_recorded_by_name():
    record = phase23.THE_FIFTEEN_NOT_ENUMERABLE
    by_row = record["the_fifteen_by_row"]
    assert sorted(by_row) == [
        "p16-anchor-loop-unresolved", "p17", "p7b-claimably-worse",
        "p7c-nine-verdicts",
    ]
    assert "9." in by_row["p7c-nine-verdicts"]
    assert "4." in by_row["p17"]
    assert 9 + 4 + 1 + 1 == 15
    assert 30 + 15 == 45

    why = _flat(record["it_is_a_limitation_of_the_record_not_the_method"])
    assert "applies perfectly well to all fifteen" in why
    assert "record's enumerability is" in why
    assert "Deferred, not abandoned" in _flat(record["they_remain_recoverable"])


def test_normality_was_described_and_gated_nothing():
    record = phase23.NORMALITY_AS_OBSERVED
    assert "no degenerate fold was encountered" in _flat(
        record["what_the_run_reports"]
    ).lower()
    gated = _flat(record["no_branch_depended_on_them"])
    assert "criterion 9 held" in gated.lower()
    assert "choosing a method after seeing data" in gated


def test_the_closing_walks_eleven_criteria_and_marks_the_partial_ones():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held the
    closing at [CLOSED, PARTIAL], true while thirty of forty-five
    contrasts had run. The rerun completed the scope, so the pin now
    holds the end state -- and still holds that the earlier state is
    recorded rather than overwritten."""
    from cleft import results_ledger

    record = phase23.PHASE_23_CLOSING
    walk = record["criterion_walk"]
    assert len(walk) == 11
    assert sorted(int(k.split("_")[0]) for k in walk) == list(range(1, 12))

    finding = _flat(record["the_finding"])
    assert "GENUINELY UNCERTAIN, NOT QUIETLY EQUIVALENT" in finding
    assert "0.3271" in finding

    two = _flat(walk["2_the_rope_is_fixed_before_any_posterior"])
    assert "THE GUARD FIRED IN REAL USE" in two
    assert "BEFORE any posterior existed" in two
    assert "the criterion WORKING, not as an incident" in two

    six = _flat(walk["6_no_decision_is_never_written_as_equivalence"])
    assert "no opportunity to fail" in six
    assert "worth marking as such" in six

    assert "vacuous in this subset" in _flat(
        walk["4_the_pairing_qualifications_travel"]
    )
    assert "PARTIAL" in _flat(walk["8_seventeen_rows_and_no_more"])
    assert record["tag"].startswith("[CLOSED, COMPLETE]")
    assert "was [CLOSED, PARTIAL]" in record["tag"]

    assert "ledger stands at 38" in _flat(
        walk["7_the_banked_verdicts_are_unchanged"]
    )
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 23 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p23" or "p23-" in e["id"]
    ]
    assert results_ledger.validate() is None
    assert "does not CLAIM" in _flat(record["no_ledger_row"])

    half = _flat(record["the_half_that_stays_unbuilt"])
    assert "NOTED-NOT-COMMITTED and unbuilt" in half
    assert "1,972 runs" in half
    assert "limitation therefore stands" in half


# --------------------------------------------------------------------------
# [2026-09-02] The completion to forty-five
# --------------------------------------------------------------------------


def _gen_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_p23", REPO / "scripts" / "generate_phase23_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p7cs_nine_are_built_by_construction_not_by_a_literal():
    """paired_matrix's rule: six against identity plus three declared.
    The count must fall out of the construction, not be asserted."""
    from cleft import phase7c

    module = _gen_module()
    pairs = module.p7c_nine()
    assert len(pairs) == 9

    identity = module._p7c_full(phase7c.IDENTITY_ARM)
    against = [p for p in pairs if p["key"].startswith("vs_identity__")]
    declared = [p for p in pairs if p["key"].startswith("declared__")]
    assert len(against) == 6 and len(declared) == 3
    assert 6 + 3 == 9
    # Every identity comparison really uses the identity arm.
    assert all(p["a"] == identity for p in against)
    # And the three are comparisons()'s own, by question name.
    questions = {c["question"] for c in phase7c.comparisons()}
    assert {p["key"].removeprefix("declared__") for p in declared} == questions
    # Six against identity means every non-identity arm exactly once.
    assert len({p["b"] for p in against}) == 6
    assert identity not in {p["b"] for p in against}


def test_the_p7c_naming_reconciliation_would_catch_a_miss():
    """A prefix mismatch MISSES SILENTLY rather than failing, so the
    generator refuses when the identity arm is not among the arms."""
    from cleft import phase7c

    module = _gen_module()
    # The short key and the full stem are different strings, and the
    # reconciliation maps one to the other.
    assert phase7c.IDENTITY_ARM == "0_identity"
    assert module._p7c_full("0_identity") == "p7c_0_identity"
    assert module._p7c_full("p7c_2_geometric") == "p7c_2_geometric"

    # Every arm the construction names is a real declared run stem.
    stems = set(module._p7c_arms())
    pairs = module.p7c_nine()
    for pair in pairs:
        assert pair["a"] in stems, pair
        assert pair["b"] in stems, pair

    # And the refusal exists: an identity arm outside the list raises.
    source = (
        REPO / "scripts" / "generate_phase23_config.py"
    ).read_text(encoding="utf-8")
    assert "is not among" in source
    assert "silently miss" in source


def test_p17s_four_map_to_their_rows_and_the_fifth_is_correctly_excluded():
    import yaml

    from cleft import results_ledger

    config = yaml.safe_load(
        (REPO / "configs" / "p23_rope.yaml").read_text(encoding="utf-8")
    )
    p17 = [c for c in config["task"]["contrasts"] if c["row"].startswith("p17")]
    assert len(p17) == 4
    assert {c["row"] for c in p17} == {
        "p17-a-vs-probe", "p17-c-vs-probe", "p17-a-vs-c", "p17-b-vs-c",
    }
    # The fifth family member is CLAIMABLE and outside the scope.
    row_33 = results_ledger.ENTRIES[33]
    assert row_33["id"] == "p17-b-vs-probe"
    assert row_33["status"] == "CLAIMABLE"
    assert not any(c["row"] == "p17-b-vs-probe" for c in config["task"]["contrasts"])
    # Its exclusion is recorded as confirming the mapping.
    assert "confirms the mapping rather than leaving it inferred" in _flat(
        phase23.THE_COMPLETION_TO_FORTY_FIVE["the_fifteen_and_their_sources"][
            "p17_four"]
    )


def test_the_config_is_forty_five_with_every_hash_filled():
    """[UPDATED 2026-09-02, THIS PIN FIRED AS DESIGNED] It held the
    completion at one placeholder. the declare filled it, so the
    pin now holds the end state: forty-five contrasts, none pending."""
    import pathlib

    import yaml

    config = yaml.safe_load(
        (REPO / "configs" / "p23_rope.yaml").read_text(encoding="utf-8")
    )
    task = config["task"]
    assert len(task["contrasts"]) == task["expect_contrasts"] == 45
    assert len({c["row"] for c in task["contrasts"]}) == 17

    placeholders = [
        e["name"] for e in config["inputs"]
        if set(e["rollup_sha256"]) == {"0"}
    ]
    assert placeholders == []
    # p7b's winner is declared HERE first -- no shipped source to match,
    # so it is exempt by construction rather than by being unfilled.
    declared_here_first = {"p7b_search"}

    # Every OTHER entry is byte-identical to a shipped config's own.
    import glob

    shipped = {}
    for path in glob.glob(str(REPO / "configs" / "*.yaml")):
        if path.endswith("p23_rope.yaml"):
            continue
        other = yaml.safe_load(
            pathlib.Path(path).read_text(encoding="utf-8")
        )
        if not isinstance(other, dict):
            continue
        for entry in other.get("inputs") or []:
            shipped.setdefault(entry["path"], entry["rollup_sha256"])
    for entry in config["inputs"]:
        if entry["name"] in declared_here_first:
            continue
        assert entry["path"] in shipped, entry["name"]
        assert entry["rollup_sha256"] == shipped[entry["path"]], entry["name"]


def test_the_task_refuses_a_silent_partial_run():
    """The 30-contrast config called itself incomplete in a comment;
    the completed one refuses in code."""
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p23_rope)
    assert 'len(task["contrasts"]) != int(task["expect_contrasts"])' in body
    # The message wraps across two source lines.
    assert "A partial " in body and "run is refused" in body
    # The refusal is BEFORE any posterior, not after. **[UPDATED
    # 2026-09-02, THIS PIN FIRED AS DESIGNED] It read
    # ``rope.probabilities(`` out of the task's own source. The
    # posterior now lives in ``p23_contrast_record``, extracted so the
    # sweep can EXECUTE the assembly, so the pin follows it to the
    # first call of the assembler.**
    refusal = body.index("A partial ")
    first_posterior = body.index("p23_contrast_record(")
    assert refusal < first_posterior
    assert "rope.probabilities(" in inspect.getsource(
        run_module.p23_contrast_record
    )


def test_the_task_reads_both_declaration_forms():
    """p17, p16 and p7b arms are declared as DIRECTORIES; p7c and the
    rest as per-seed files. Both must be reachable."""
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p23_rope)
    assert 'declared.get(f"{arm}__seed_{seed}")' in body
    assert "declared.get(arm)" in body
    assert 'f"seed_{seed}__predictions.csv"' in body
    # The filename is the writer's, not guesswork.
    writer = (REPO / "src" / "cleft" / "train" / "phase3.py").read_text(
        encoding="utf-8"
    )
    assert 'f"{prefix}predictions.csv"' in writer


def test_the_thirteen_enumerable_correction_is_recorded():
    record = phase23.THE_FIFTEEN_NOT_ENUMERABLE
    correction = _flat(record["CORRECTED_2026_09_02_thirteen_WERE_enumerable"])
    assert "FALSE for thirteen of the fifteen" in correction
    assert "preserved as written" in correction
    assert "not a filter over twenty-one" in correction
    assert "Only p7b's winner was genuinely unrecorded" in correction
    # The original claim is still there, unedited.
    assert "not enumerable from any readable record" in _flat(
        record["it_is_a_limitation_of_the_record_not_the_method"]
    ) or "applies perfectly well to all fifteen" in _flat(
        record["it_is_a_limitation_of_the_record_not_the_method"]
    )
    provenance = _flat(record["the_provenance_of_the_wrong_claim"])
    assert "docstring ONE LINE ABOVE" in provenance
    assert "the record's" in provenance


def test_the_criterion_11_grounds_and_their_condition():
    record = phase23.THE_COMPLETION_TO_FORTY_FIVE
    grounds = _flat(record["the_criterion_11_grounds"])
    assert "FULLY DETERMINED, NOT CHOSEN" in grounds
    assert "complement of the thirty within the locked forty-five" in grounds
    assert "Completion, not addition" in grounds

    condition = _flat(record["the_condition_stated_explicitly"])
    assert "criterion 11 WOULD bite" in condition
    assert "thirty results already visible" in condition
    assert "cannot borrow this precedent without meeting it" in condition

    # Criterion 8's count corrected, and the criterion really says 17.
    assert "It says SEVENTEEN" in _flat(record["criterion_8s_row_count_corrected"])
    assert "8_seventeen_rows_and_no_more" in phase23.EXIT_CRITERIA

    # The delta is +12 and the record says why it is not +11.
    delta = _flat(record["the_config_delta_measured"])
    assert "+12 -- not +11" in delta
    assert "p16_anchor_loop__f342fed9__p16-anchor-loop" in delta
    assert "reported '0 new'" in delta

    # The rerun replaces rather than supplements.
    replaces = _flat(record["the_rerun_replaces_rather_than_supplements"])
    assert "REPLACE the thirty" in replaces
    assert "which is itself a check" in replaces


# ---- SWEEP 1: config -> code, for the new surface ------------------------


def test_the_new_declaration_form_reaches_code():
    """No new config FIELD was added -- the completion changed the
    input NAMING, not the schema. The sweep confirms that."""
    import inspect

    from cleft import run as run_module

    # The spec is unchanged: no new field to trace.
    assert set(schema.TASK_SPECS["p23_rope"]) == {
        "kind", "contrasts", "seeds", "probe_seed_sd", "probe_n_seeds",
        "expect_rope_half_width", "rho", "decision_threshold", "full_n",
        "expect_contrasts", "expect_patients",
    }
    body = inspect.getsource(run_module.task_p23_rope)
    for field in schema.TASK_SPECS["p23_rope"]:
        if field in ("kind", "expect_patients"):
            continue
        assert f'task["{field}"]' in body, field


# ---- SWEEP 2: code -> record, for what the additions compute -------------


def test_everything_the_additions_compute_reaches_the_record():
    """**[UPDATED 2026-09-02, THIS SWEEP MISSED A DEFECT.]** It read
    the task's SOURCE for field names and asserted ``observations``
    among them. It was there, **carrying nulls** -- three contrasts ran
    at n=24 with no reason beside them. Presence of a key is a weaker
    claim than presence of its content. The source check is kept, now
    following the extracted assembler, and the CONTENT check is the
    executed test below (phase23.THE_SWEEP_CHECKED_PRESENCE_NOT_CONTENT).
    """
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p23_rope)
    written = body.split("tmp.write_text(")[1]
    per_contrast = inspect.getsource(run_module.p23_contrast_record)
    per_contrast = per_contrast.split("return {")[1]
    for field in ('"row"', '"a"', '"b"', '"n"', '"degrees_of_freedom"',
                  '"probabilities"', '"verdict"', '"observations"'):
        assert field in per_contrast, field
    assert '"n_contrasts"' in written and '"verdict_counts"' in written


# --------------------------------------------------------------------------
# [2026-09-02] The drop that was counted and never recorded
# --------------------------------------------------------------------------


def _arm(rows_by_seed):
    """An arm in the shape ``p23_contrast_record`` consumes."""
    from cleft import rope

    detail = {s: rope.per_fold_detail(r) for s, r in rows_by_seed.items()}
    return {
        "pcc": {
            s: {f: d["pcc"] for f, d in folds.items()}
            for s, folds in detail.items()
        },
        "detail": detail,
        "missing_seeds": [],
    }


def _csv(fold_values):
    """``{fold: [(truth, prediction), ...]}`` -> prediction CSV rows."""
    out, patient = [], 0
    for fold, pairs in fold_values.items():
        for truth, prediction in pairs:
            out.append({
                "patient_id": f"p{patient}", "fold": fold,
                "truth": truth, "prediction": prediction,
            })
            patient += 1
    return out


LIVE = [(1.0, 1.1), (2.0, 1.9), (3.0, 3.4), (4.0, 3.6)]
SHIFTED = [(1.0, 1.4), (2.0, 2.2), (3.0, 2.7), (4.0, 4.1)]
DEAD = [(1.0, 2.5), (2.0, 2.5), (3.0, 2.5), (4.0, 2.5)]


# ---- SWEEP 2, EXECUTED: the record must CARRY, not merely contain --------


def test_the_assembled_record_carries_the_drop_not_just_the_shortfall():
    """The defect, reproduced and then absent: an n short by one, with
    the arm, the seed, the fold and the cause beside it."""
    from cleft.run import p23_contrast_record

    probe = _arm({
        1: _csv({0: LIVE, 1: LIVE}),
        2: _csv({0: SHIFTED, 1: LIVE}),
    })
    arm_c = _arm({
        1: _csv({0: SHIFTED, 1: DEAD}),      # seed 1 fold 1 degenerate
        2: _csv({0: LIVE, 1: SHIFTED}),
    })
    record = p23_contrast_record(
        {"key": "p17-a-vs-c", "row": "p17-a-vs-c", "a": "arm_C",
         "b": "probe"},
        arm_c, probe,
        rho=0.2, half_width=0.018346, threshold=0.95, full_n=4,
    )
    assert record["n"] == 3 and record["degrees_of_freedom"] == 2
    assert record["n_is_the_full_form"] is False

    observed = record["observations"]
    assert observed["n_dropped"] == 1
    assert observed["degenerate_folds_dropped"] == 1
    drop = observed["dropped_folds"][0]
    assert drop["seed"] == 1 and drop["fold"] == 1
    assert drop["arms"] == ["arm_C"]
    assert drop["causes"][0]["reason"] == (
        "constant prediction within the fold"
    )
    assert drop["causes"][0]["prediction_sd"] == 0.0
    assert drop["causes"][0]["truth_sd"] > 0.0
    assert observed["n_unpaired"] == 0

    # **The check the old sweep could not make**: nothing the record
    # carries is null, at any depth.
    def no_nulls(value, path="record"):
        if isinstance(value, dict):
            for key, item in value.items():
                assert item is not None, f"{path}.{key}"
                no_nulls(item, f"{path}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                assert item is not None, f"{path}[{i}]"
                no_nulls(item, f"{path}[{i}]")

    no_nulls(record)
    # And the shortfall is fully accounted for.
    assert record["cells_accounted"] == 4
    assert (
        record["n"] + observed["n_dropped"] + observed["n_unpaired"] == 4
    )


def test_an_unpaired_cell_is_recorded_as_itself_not_as_a_degeneracy():
    """n=24 had two possible causes and the record distinguished
    neither. It distinguishes them now."""
    from cleft.run import p23_contrast_record

    probe = _arm({1: _csv({0: LIVE, 1: LIVE}), 2: _csv({0: SHIFTED})})
    thin = _arm({1: _csv({0: SHIFTED}), 2: _csv({0: LIVE})})
    #                     ^ seed 1 fold 1 absent from this arm entirely
    record = p23_contrast_record(
        {"key": "k", "row": "r", "a": "thin", "b": "probe"},
        thin, probe,
        rho=0.2, half_width=0.018346, threshold=0.95, full_n=3,
    )
    observed = record["observations"]
    assert record["n"] == 2
    assert observed["n_dropped"] == 0          # NOT a degeneracy
    assert observed["n_unpaired"] == 1
    gap = observed["unpaired_folds"][0]
    assert gap == {"seed": 1, "fold": 1, "present_in": "probe",
                   "absent_from": "thin"}


def test_a_reduced_n_the_record_cannot_account_for_is_refused():
    """A short n with no words for it stops the run rather than being
    written -- which is what the shipped record did instead."""
    import pytest

    from cleft.run import p23_contrast_record

    a = _arm({1: _csv({0: LIVE, 1: SHIFTED})})
    b = _arm({1: _csv({0: SHIFTED, 1: LIVE})})
    a["missing_seeds"] = [2]
    b["missing_seeds"] = [2]
    with pytest.raises(ValueError, match="Refusing to write a reduced n"):
        p23_contrast_record(
            {"key": "k", "row": "r", "a": "a", "b": "b"}, a, b,
            rho=0.2, half_width=0.018346, threshold=0.95, full_n=4,
        )


def test_the_task_passes_the_records_through_and_logs_the_cause():
    import inspect

    from cleft import run as run_module

    task = inspect.getsource(run_module.task_p23_rope)
    assert "rope.per_fold_detail(rows)" in task
    assert '"missing_seeds": missing' in task
    assert "DROPPED seed" in task and "UNPAIRED seed" in task
    # The skipped seed is recorded rather than merely continued past.
    assert "missing.append(int(seed))" in task


def test_the_defect_and_the_sweep_lesson_are_recorded():
    record = phase23.THE_DROP_WAS_COUNTED_NOT_RECORDED
    mechanism = _flat(record["the_mechanism_never_populated"])
    assert "NEVER POPULATED" in mechanism
    assert "never existed" in mechanism
    assert "Not overwritten and not" in mechanism
    assert "discards it in the same" in _flat(record["where_the_drop_HAPPENS"])
    second = _flat(record["the_SECOND_mechanism_that_was_invisible"])
    assert "INTERSECTION" in second and "counted nowhere" in second
    assert "code -> record, the third instance" in _flat(
        record["the_error_class"]
    )
    assert record["tag"].startswith("[FIXED, NOT RERUN]")

    sweep = phase23.THE_SWEEP_CHECKED_PRESENCE_NOT_CONTENT
    assert "It does. It arrived carrying nulls" in _flat(
        sweep["what_the_sweep_asserted"]
    )
    assert "reaching the record is not carrying the value" in _flat(
        sweep["why_that_is_not_enough"]
    )
    assert "must execute the assembly, not read" in _flat(
        sweep["the_rule_this_leaves"]
    )


def test_the_forty_five_outturn_and_the_two_further_decisions():
    outturn = phase23.THE_FORTY_FIVE_OUTTURN
    assert outturn["run"] == "p23_rope__8bbb9a1b__p23-rope-45"
    counts = _flat(outturn["the_counts"])
    assert "0 practically equivalent, 4 practically different, 41 no" in counts
    assert "0.3452" in _flat(outturn["the_highest_p_rope_anywhere"])
    assert "42 contrasts at n=25 and THREE at n=24" in _flat(outturn["the_ns"])
    identical = _flat(outturn["the_thirty_recomputed_IDENTICALLY"])
    assert "both original decisions to four decimals" in identical
    assert "a free check, and it passed" in identical

    decisions = phase23.THE_TWO_FURTHER_DECISIONS
    assert decisions["3_p17_c_vs_probe"]["p_left"] == 0.9990
    assert decisions["3_p17_c_vs_probe"]["n"] == 24
    a_vs_c = decisions["4_p17_a_vs_c"]
    assert a_vs_c["p_right"] == 0.9910
    assert "the attribution is not claimed" in _flat(
        a_vs_c["what_phase_17_recorded"]
    )
    disagree = _flat(a_vs_c["it_is_an_INSTRUMENTS_DISAGREE_CASE"])
    assert "Phase 17's verdict STANDS UNCHANGED" in disagree
    assert "The disagreement IS the finding" in disagree.replace("**", "")
    assert "no row is added" in disagree
    assert "does not make the scheme attribution a CLAIM" in _flat(
        a_vs_c["what_it_does_NOT_license"]
    )


def test_the_closing_is_complete_on_the_contrast_count():
    closing = phase23.PHASE_23_CLOSING
    criterion_8 = _flat(closing["criterion_walk"]["8_seventeen_rows_and_no_more"])
    assert "COMPLETE: 45 of 45 contrasts across all SEVENTEEN rows" in criterion_8
    # The two standing reductions are unchanged, and said so here.
    assert "NOTED, NOT COMMITTED" in criterion_8
    assert "IEM row stays EXCLUDED" in criterion_8
    # The first run's paragraph is preserved, not edited.
    assert "PARTIAL AT THE FIRST RUN" in criterion_8

    # Criterion 4 went from vacuous to live, and failed its first test.
    criterion_4 = _flat(closing["criterion_walk"]["4_the_pairing_qualifications_travel"])
    assert "NO LONGER VACUOUS, AND NOT MET ON ITS FIRST LIVE TEST" in criterion_4
    assert "PENDING THE RERUN" in criterion_4
    assert "vacuous in this subset" in criterion_4   # original preserved

    assert "COMPLETED 2026-09-02 by run p23_rope__8bbb9a1b" in _flat(
        closing["closed"]
    )
    # No ledger row, at forty-five as at thirty.
    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 23 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p23" or "p23-" in e["id"]
    ]
    results_ledger.validate()          # raises if the chain broke


# --------------------------------------------------------------------------
# [2026-09-02] The rerun, and the fold it made readable
# --------------------------------------------------------------------------


def test_the_rerun_is_the_citable_run_and_supersedes_for_observations_only():
    record = phase23.THE_RERUN_IS_THE_CITABLE_RUN
    assert record["run"] == "p23_rope__402a8477__p23-rope-45-2"
    assert record["supersedes"] == "p23_rope__8bbb9a1b__p23-rope-45"

    reproduced = _flat(record["what_it_reproduced"])
    assert "verdicts IDENTICAL" in reproduced
    assert "0 practically equivalent, 4 practically different, 41 no" in reproduced
    assert "four decimals" in reproduced

    superseded = _flat(record["SUPERSEDED_FOR_OBSERVATIONS_ONLY_NOT_VOID"])
    assert "NUMBERS WERE CORRECT AND ARE UNCHANGED" in superseded
    assert "only its drop records were empty" in superseded
    assert "nothing is withdrawn" in superseded

    # The distinction from VOID is kept, and it cites the precedent.
    void = _flat(record["the_distinction_from_VOID_and_why_it_matters"])
    assert "a VOID run's numbers are WRONG" in void
    assert "ladder.SIBLING_RUNS_AUDIT" in void
    assert "right numbers, INCOMPLETE RECORD" in void
    from cleft import ladder

    assert "VOID-versus-good, where wrong numbers give " in _flat(
        ladder.SIBLING_RUNS_AUDIT["the_partial_is_a_new_shape"]
    )


def test_the_degenerate_fold_is_banked_with_its_cause():
    record = phase23.THE_DEGENERATE_FOLD
    cell = _flat(record["the_cell"])
    for token in ("p17-c-vs-probe", "seed 1337", "fold 1", "p17_arm_c",
                  "constant prediction within the fold", "47",
                  "exactly 0.0", "0.6159"):
        assert token in cell, token

    accounting = _flat(record["the_accounting_closes"])
    assert "24 + 1 dropped + 0 unpaired = 25" in accounting
    assert "REFUSED" in accounting
    assert 24 + 1 + 0 == 25

    # What it says about the ARM, not the arithmetic.
    arm = _flat(record["what_it_says_about_the_ARM_not_the_arithmetic"])
    assert "rounds a distance to a grade" in arm
    assert "-0.0185" in arm
    assert "collapsed all 47 patients onto ONE grade" in arm
    assert "A property of the arm, evidenced" in arm
    # The -0.0185 is Phase 17's own number, quoted rather than new.
    from cleft import phase17

    assert "-0.0185" in _flat(
        phase17.UNPREDICTED_PATTERN["the_observed_pattern"]
    )

    before = _flat(record["the_mechanism_was_DISTINGUISHED_BEFORE_THE_RERUN"])
    assert "degenerate_folds_dropped" in before
    assert "confirmed rather than discovered" in before
    assert "the cause, not the class" in before


def test_criterion_4_carries_three_states_with_the_originals_preserved():
    """Vacuous, then failed, then met -- and a criterion that was never
    exercised is not a criterion that passed."""
    walk = phase23.PHASE_23_CLOSING["criterion_walk"]
    criterion = _flat(walk["4_the_pairing_qualifications_travel"])

    # State 1, as written at thirty.
    assert "vacuous in this subset" in criterion
    assert "untested rather than unmet" in criterion
    # State 2, as written at forty-five.
    assert "NO LONGER VACUOUS, AND NOT MET ON ITS FIRST LIVE TEST" in criterion
    # State 3, on the rerun.
    assert "UPDATED AGAIN 2026-09-02 on the rerun" in criterion
    assert "MET." in criterion
    assert "24 + 1 + 0 = 25" in criterion
    assert "THREE STATES, and all three stand as written" in criterion
    assert "never exercised is not a criterion that passed" in criterion
    # The three are dated in place; none replaced another.
    assert criterion.index("vacuous in this subset") < criterion.index(
        "NO LONGER VACUOUS"
    ) < criterion.index("UPDATED AGAIN")


def test_the_probed_key_correction_points_at_the_real_field_name():
    """A reader searching for ``dropped`` must find the pointer."""
    record = phase23.THE_DROP_WAS_COUNTED_NOT_RECORDED
    line = _flat(
        record["CORRECTED_2026_09_02_the_field_is_dropped_folds_not_dropped"]
    )
    assert "observations.dropped_folds" in line
    assert "n_dropped" in line and "unpaired_folds" in line
    assert "is the ARGUMENT NAME of ``rope.describe``, not a key" in line
    assert "read the list as ABSENT when it was present" in line

    # The code says it too, where the mistake is made.
    import inspect

    from cleft import rope

    doc = inspect.getdoc(rope.describe)
    assert "emitted keys are ``dropped_folds`` and ``unpaired_folds``" in doc
    assert "neither is a key in the\nrecord" in doc
    # And the names really are what the record says they are.
    emitted = rope.describe(
        __import__("numpy").array([0.1, 0.2]), dropped=[], unpaired=[]
    )
    assert "dropped_folds" in emitted and "dropped" not in emitted
    assert "unpaired_folds" in emitted and "unpaired" not in emitted


def test_the_phase_is_closed_COMPLETE_with_three_reductions_standing():
    closing = phase23.PHASE_23_CLOSING
    tag = _flat(closing["tag"])
    assert tag.startswith("[CLOSED, COMPLETE]")
    assert "was [CLOSED, PARTIAL]" in tag           # the earlier state kept
    assert "45 of 45 across seventeen rows" in tag

    landed = _flat(closing["COMPLETED_2026_09_02_the_rerun_landed"])
    assert "p23_rope__402a8477__p23-rope-45-2" in landed
    assert "24 + 1 + 0 = 25" in landed

    open_still = _flat(closing["THE_PHASE_IS_COMPLETE_AND_WHAT_STAYS_OPEN"])
    assert "complete is not comprehensive" in open_still
    assert "NOTED, NOT COMMITTED and UNBUILT" in open_still
    assert "IEM row stays EXCLUDED on units" in open_still
    assert "five-seeds-on-a-FIXED-SPLIT limitation stands" in open_still
    assert "says nothing about a split it never varied" in open_still
    # The split really is fixed in the frozen apparatus, and this
    # phase says so in its own words rather than by assertion here.
    assert "shuffle=False" in " ".join(
        str(v) for v in phase23.SPLIT_RANDOMISED_HALF_NOT_BUILT.values()
    )

    # RESTATES, does not CLAIM: no row, at forty-five as at thirty.
    from cleft import results_ledger

    assert "does not CLAIM" in _flat(closing["no_ledger_row"])
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 23 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p23" or "p23-" in e["id"]
    ]
    results_ledger.validate()          # raises if the chain broke
