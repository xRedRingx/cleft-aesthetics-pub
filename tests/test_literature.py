"""The literature bank, and the tag discipline it is written under."""

from __future__ import annotations

import math
import re
from itertools import permutations
from pathlib import Path

import pytest

from cleft import literature, phase21

REPO = Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# tag discipline
# --------------------------------------------------------------------------


def test_the_tags_are_exactly_three():
    assert literature.TAG_DISCIPLINE["the_three"] == (
        "[LITERATURE]", "[MEASURED]", "[REASONED]"
    )
    why = _flat(literature.TAG_DISCIPLINE["why_it_is_not_cosmetic"])
    assert "precisely the claim nobody audits" in why
    assert "unauditable by the very sweep" in why
    # The lesson it cites is live at its own source.
    from cleft import ladder

    assert "precisely the claim nobody audits" in _flat(
        ladder.BASAL_RATIONALE_UNSUPPORTED["lesson"]
    )
    # A DATE is explicitly not a provenance clause -- the house
    # convention survives the ruling.
    assert "STAYS" in literature.TAG_DISCIPLINE[
        "a_date_is_not_a_provenance_clause"
    ]


def test_every_record_in_the_bank_carries_a_plain_tag():
    """Nothing in the new module may reintroduce a compound tag."""
    banked = [
        literature.GEIRHOS_ERROR_CONSISTENCY,
        literature.BOUTHILLIER_SAMPLE_SIZE,
        literature.NADEAU_BENGIO_DOES_NOT_APPLY,
    ]
    for record in banked:
        assert record["tag"] == "[LITERATURE]", record["source"]
    for key, entry in literature.SOURCES_BANKED.items():
        if key == "provenance":
            continue
        assert entry["tag"] == "[LITERATURE]", key
    assert literature.COMPARATORS_BANKED["guan"]["tag"] == "[LITERATURE]"
    assert literature.COMPARATORS_BANKED["heinrich"]["tag"] == "[LITERATURE]"
    # The one [REASONED] in the bank is the Guan inference, tagged as one.
    assert literature.COMPARATORS_BANKED["guan"][
        "the_n_274_basis_is_an_inference"
    ]["tag"] == "[REASONED]"


def test_the_normalised_tags_kept_their_originals_visible():
    """Every normalisation is dated in place with the original wording
    preserved -- a correction that erases what it corrected is not one."""
    # Counted with whitespace normalised: the markers wrap across `#:`
    # comment lines and adjacent string literals, so a raw search
    # undercounts. phase10_annex has seven -- one loss line plus a
    # docstring and a value marker for each of the three per-rater
    # blocks.
    normalised = {
        "src/cleft/phase10_annex.py": 7,
        "src/cleft/phase17.py": 1,
        "src/cleft/record_audit.py": 1,
        "src/cleft/train/graph_cleft.py": 1,
        "src/cleft/phase12.py": 1,
    }
    for path, expected in normalised.items():
        raw = (REPO / path).read_text(encoding="utf-8")
        flat = re.sub(r'"\s*\n\s*"', "", raw)
        flat = re.sub(r"\s*\n\s*#:\s*", " ", flat)
        flat = re.sub(r"\s+", " ", flat)
        count = len(re.findall(r"TAG NORMALISED 2026-09-01", flat, re.I))
        assert count == expected, f"{path}: {count} markers, expected {expected}"
        # The original wording survives beside EVERY marker, not just one.
        originals = len(re.findall(r"normalised 2026-09-01 from", flat, re.I))
        assert originals == expected, (
            f"{path}: {originals} originals preserved against {expected} "
            "markers -- a correction that erases what it corrected is not one"
        )


def test_the_load_bearing_variants_are_reported_not_silently_changed():
    """Variants a test asserts by exact string were REPORTED and left in
    place, per the instruction. This pins which ones remain, so a later
    silent change fires here."""
    remaining = {}
    for path in sorted((REPO / "src" / "cleft").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for variant in ("[LITERATURE-primary", "[REPORTED, not measured]",
                        "[REPORTED, not verified here]",
                        "[REPORTED, verified against",
                        "[MANUSCRIPT,", "[REASONED -> MEASURED"):
            if variant in text:
                remaining.setdefault(variant, []).append(path.name)
    assert sorted(remaining) == [
        "[LITERATURE-primary", "[MANUSCRIPT,", "[REASONED -> MEASURED",
        "[REPORTED, not measured]", "[REPORTED, not verified here]",
        "[REPORTED, verified against",
    ], sorted(remaining)
    # Each is genuinely asserted by a test -- that is why it stayed.
    tests = {
        p.name: p.read_text(encoding="utf-8")
        for p in (REPO / "tests").glob("test_*.py")
    }
    for variant in remaining:
        assert any(variant in text for text in tests.values()), variant


# --------------------------------------------------------------------------
# Geirhos
# --------------------------------------------------------------------------


def test_geirhos_sanctions_arm_bs_binarisations():
    record = literature.GEIRHOS_ERROR_CONSISTENCY
    assert record["source"].startswith("2006.16736.pdf")
    assert "is not in this repository" in record["provenance"]

    assert "only analyse whether the decisions were correct/incorrect" in (
        record["the_metric_is_defined_for_binary_outcomes"]
    )
    assert "must be thresholded and binarized" in (
        record["continuous_outputs_must_be_binarised"]
    )
    sanctioned = _flat(record["arm_bs_binarisations_are_sanctioned"])
    assert "the sanctioned adaptation, not an invented one" in sanctioned
    # Phase 21 carries the dated pointer.
    assert "literature.GEIRHOS_ERROR_CONSISTENCY" in phase21.ARM_B_REGISTERED[
        "binarisation_is_sanctioned_2026_09_01"
    ]


def test_the_published_bands_are_context_and_the_lock_is_untouched():
    record = literature.GEIRHOS_ERROR_CONSISTENCY
    assert "0.32-0.48" in record["human_to_human_band"]
    assert "0.62-0.79" in record["cnn_to_cnn_band"]
    assert "0.793" in record["cnn_to_cnn_band"]
    assert "0.066-0.068" in record["cnn_to_human_ood"]

    # The human band's endpoints follow from the four quoted points.
    points = [0.33, 0.32, 0.48, 0.37]
    assert (min(points), max(points)) == (0.32, 0.48)

    # Where the locked threshold sits -- re-derived, not asserted.
    high = float(phase21.CONSISTENCY_THRESHOLDS_RULED["high_value"])
    assert high == 0.50
    assert round(high - 0.48, 2) == 0.02
    assert round(0.62 - high, 2) == 0.12
    sits = _flat(record["where_arm_bs_locked_threshold_sits"])
    assert "0.02 ABOVE the human ceiling" in sits
    assert "0.12 BELOW the CNN floor" in sits

    # CONTEXT, not amendment -- and the lock really is untouched.
    context = _flat(record["context_not_amendment"])
    assert "is locked and stays locked" in context
    assert "A SENTENCE BESIDE THE RESULT" in context
    assert phase21.CONSISTENCY_THRESHOLDS_RULED["low_value"] == 0.05
    assert phase21.CONSISTENCY_THRESHOLDS_RULED["high_value"] == 0.50
    assert "published_band_for_context_2026_09_01" in (
        phase21.CONSISTENCY_THRESHOLDS_RULED
    )


def test_both_instability_warnings_are_banked_with_arm_bs_arithmetic():
    record = literature.GEIRHOS_ERROR_CONSISTENCY

    # The denominator warning, and Arm B's own numbers -- recomputed.
    for p, expected in ((0.50, 0.5000), (0.25, 0.6250)):
        c_exp = p * p + (1 - p) * (1 - p)
        assert c_exp == pytest.approx(expected, abs=1e-9)
        assert 1 - c_exp > 0.3, "far from the degenerate end"
    denominator = _flat(record["instability_denominator"])
    assert "c_exp = 0.500" in denominator and "0.625" in denominator
    assert "NO RISK" in denominator

    # The small-trial warning, and the cohort's own count.
    trials = _flat(record["instability_small_trial_counts"])
    assert "flags it at 160" in trials
    assert "Arm B has 237" in trials
    assert 237 > 160


# --------------------------------------------------------------------------
# Bouthillier, and the arm noted but not committed
# --------------------------------------------------------------------------


def test_bouthillier_qualifies_the_five_seed_protocol_without_voiding_it():
    record = literature.BOUTHILLIER_SAMPLE_SIZE
    assert record["source"].startswith("2103.03098.pdf")
    assert "not to be confused with dataset size n" in record["what_a_sample_is"]
    assert "severe underestimations of standard error" in (
        record["five_seeds_on_a_fixed_split_is_their_biased_estimator"]
    )
    assert "UNDER HALF" in record["data_sampling_dominates"]
    assert "gamma = 0.75, N = 29" in record["their_p_a_beats_b"]

    independently = _flat(record["this_project_arrived_there_independently"])
    assert "0.068" in independently
    assert "LARGER THAN THE SEED BAND" in independently
    assert "VARIES THE SMALLER SOURCE" in independently

    # A limitation, not a defect -- and it says what stays valid.
    limitation = _flat(record["a_limitation_not_a_defect"])
    assert "VALID FOR WHAT IT MEASURES" in limitation
    assert "Nothing banked becomes wrong" in limitation


def test_the_split_randomised_arm_is_noted_and_not_committed():
    record = literature.SPLIT_RANDOMISED_ARM_NOTED
    assert record["status"].startswith("NOTED, NOT COMMITTED")
    why = _flat(record["why_not_committed"])
    assert "scope arriving through a citation" in why
    assert "would need its own registration" in why
    assert "re-run under re-splitting" in _flat(
        record["what_would_have_to_be_ruled_first"]
    )

    # Noted means noted: no task, no kind, no config.
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    for name in list(TASK_SPECS) + list(TASKS):
        assert "split_random" not in name.lower(), name
        assert "p_a_beats" not in name.lower(), name
    assert list((REPO / "configs").glob("*split_random*.yaml")) == []


# --------------------------------------------------------------------------
# Nadeau & Bengio -- a correction to a the record assertion
# --------------------------------------------------------------------------


def test_nadeau_bengio_does_not_apply_and_the_error_is_named():
    record = literature.NADEAU_BENGIO_DOES_NOT_APPLY
    does_not = _flat(record["it_does_not_apply_to_correlation"])
    assert "AVERAGE OF PER-EXAMPLE LOSSES" in does_not
    assert "NON-LINEAR AND NON-ADDITIVE" in does_not
    assert record["smallest_validated_n"] == 200
    assert record["what_it_would_apply_to"] == "MSE or MAE"

    error = _flat(record["the_error_and_where_it_came_from"])
    assert "proposed in discussion" in error
    assert "2026-08-31" in error
    # Filed beside the other four named error-provenance records, all of which exist.
    from cleft import ladder, phase12, phase20

    assert hasattr(ladder, "THE_ERROR_PROVENANCE")
    assert hasattr(phase20, "CROSS_TARGET_ERROR_PROVENANCE")
    assert hasattr(phase20, "S_DESCRIPTION_ERROR_PROVENANCE")
    assert hasattr(phase12, "TWO_VIEW_CLAIM_PROVENANCE")


def test_the_project_had_already_retracted_nadeau_bengio_twice():
    """The sharper half: the record was right before the source was
    consulted, so this was a failure to consult it."""
    record = literature.NADEAU_BENGIO_DOES_NOT_APPLY
    assert "nothing measured" in _flat(record["what_it_did_not_touch"])
    assert "TWICE OVER" in _flat(record["the_sharper_point"])
    assert "failure to consult it" in _flat(record["the_sharper_point"])

    # Both retractions are real, at source.
    plan = (REPO / "docs" / "PLAN.md").read_text(encoding="utf-8")
    assert "corrects a test this design" in " ".join(plan.split())
    stale = (REPO / "docs" / "FROZEN_KNOWN_STALE.md").read_text(
        encoding="utf-8"
    )
    assert "Nadeau" in stale
    assert "not applied anywhere" in " ".join(stale.split())


# --------------------------------------------------------------------------
# the remaining sources
# --------------------------------------------------------------------------


def test_each_banked_source_names_its_file_and_what_it_qualifies():
    sources = literature.SOURCES_BANKED
    assert sorted(k for k in sources if k != "provenance") == [
        "benavoli_rope", "rankiqa", "swayamdipta_cartography",
        "watson_wright", "zimmerman_williams",
    ]
    for key, entry in sources.items():
        if key == "provenance":
            continue
        assert entry["source"], key
        assert entry["how_it_bears"], key

    # Each contradiction or qualification is stated, not implied.
    assert "region of practical equivalence discovered empirically" in _flat(
        sources["benavoli_rope"]["how_it_bears"]
    )
    assert "does not RESOLVE anything" in _flat(
        sources["benavoli_rope"]["what_it_does_not_do"]
    )
    assert "known BY CONSTRUCTION" in _flat(sources["rankiqa"]["how_it_bears"])
    assert "not a prediction for n=237" in _flat(sources["rankiqa"]["flagged"])
    assert "Not checked here" in _flat(
        sources["zimmerman_williams"]["flagged"]
    )
    assert "small-n behaviour" in _flat(sources["watson_wright"]["flagged"])

    # Swayamdipta: regression is NOT addressed, and Arm B is not it.
    cartography = sources["swayamdipta_cartography"]
    assert cartography["whether_regression_is_addressed"].startswith("**NO.**")
    assert "across TRAINING EPOCHS" in _flat(cartography["how_it_bears"])
    assert "across ARMS at convergence" in _flat(cartography["how_it_bears"])
    assert "different-quantities-" in _flat(cartography["how_it_bears"])


# --------------------------------------------------------------------------
# the comparators
# --------------------------------------------------------------------------


def test_guans_n_inference_is_tagged_reasoned_and_its_arithmetic_checked():
    guan = literature.COMPARATORS_BANKED["guan"]
    inference = guan["the_n_274_basis_is_an_inference"]
    assert inference["tag"] == "[REASONED]"
    assert "not stated by the authors" in _flat(inference["the_inference"])

    # The arithmetic: exact minimum two-tailed p is 2/n!.
    for n, claimed in ((5, 0.0167), (6, 0.00278), (7, 0.000397)):
        assert 2 / math.factorial(n) == pytest.approx(claimed, abs=5e-5)
    assert "2/n!" in inference["the_arithmetic"]

    # Brute-force the permutation null rather than trusting the formula.
    for n in (5, 6):
        base = list(range(n))
        extreme = sum(
            1 for perm in permutations(base)
            if abs(1 - 6 * sum((a - b) ** 2 for a, b in zip(base, perm))
                   / (n * (n * n - 1))) >= 1.0 - 1e-12
        )
        assert extreme / math.factorial(n) == pytest.approx(
            2 / math.factorial(n), abs=1e-12
        )

    # And the record separates what was checked from what was inferred.
    verified = _flat(inference["verified_here"])
    assert "the arithmetic is checked, the inference is not" in verified
    assert "recorded AS an inference" in verified


def test_guans_label_collapsing_supports_three_not_five():
    guan = literature.COMPARATORS_BANKED["guan"]
    collapsing = _flat(guan["label_collapsing_supports_three_not_five"])
    assert "27.3% to 48.2%" in collapsing
    assert "0.31 to 0.53" in collapsing
    assert "THREE_NOT_FIVE" in collapsing
    # The ratios quoted are the ratios.
    assert round(48.2 / 27.3, 2) == 1.77
    assert round(0.53 / 0.31, 2) == 1.71
    assert "1.77x and 1.71x" in collapsing

    # The record it supports exists.
    from cleft import classification

    assert hasattr(classification, "THREE_NOT_FIVE")

    # And the comparator's own agreement figures are banked beside it.
    assert "Fleiss kappa 0.31" in guan["rater_agreement"]
    assert "27.3% identical grades" in guan["rater_agreement"]
    assert "0.23 / 0.37 / 0.33" in guan["rater_agreement"]
    assert "no confidence intervals" in _flat(
        guan["what_it_does_not_report"]
    )


def test_heinrich_is_banked_without_being_used_as_a_licence():
    heinrich = literature.COMPARATORS_BANKED["heinrich"]
    assert "single examiner per time point" in heinrich["design"]
    assert "could not be formally assessed" in heinrich["reliability"]
    assert "478 landmarks validated ONLY QUALITATIVELY" in _flat(
        heinrich["what_it_lacks"]
    )
    bears = _flat(heinrich["how_it_bears"])
    assert "not a licence" in bears
    assert "being more careful than a weak comparator is not evidence" in bears


def test_the_pearson_spearman_hazard_is_a_tested_note():
    hazard = _flat(literature.COMPARATORS_BANKED[
        "the_pearson_vs_spearman_hazard"
    ])
    assert "Guan reports SPEARMAN" in hazard
    assert "0.2520 is PEARSON" in hazard
    assert "not comparable as stated" in hazard
    assert "must name which coefficient each figure is" in hazard

    # The two figures are live at their sources, and really are
    # different coefficients.
    from cleft import ladder

    assert ladder.TRADE_OFF_PAIR["result"]["vit_paired_mean"] == 0.2520
    assert "0.892" in literature.COMPARATORS_BANKED["guan"][
        "reported_correlation"
    ]
    assert "Spearman" in literature.COMPARATORS_BANKED["guan"][
        "reported_correlation"
    ]


def test_the_bank_states_that_nothing_was_verified_at_source():
    """Unlike phase12's manuscripts, none of these PDFs is reachable.
    The record must not blur the two."""
    assert "None of the cited PDFs is reachable" in " ".join(
        literature.__doc__.split()
    )
    for record in (literature.GEIRHOS_ERROR_CONSISTENCY,
                   literature.BOUTHILLIER_SAMPLE_SIZE,
                   literature.NADEAU_BENGIO_DOES_NOT_APPLY):
        assert "is not in this repository" in record["provenance"]
    assert "NONE reachable" in literature.SOURCES_BANKED["provenance"]

    # And the contrast with phase12 is real: there, two of four WERE.
    from cleft import phase12

    verified = phase12.RATING_PROCEDURE_DOCUMENTED[
        "verified_at_source_2026_08_31"
    ]
    assert "ALL SEVEN" in verified["docx_quotes_found"]


def test_the_summary_carries_every_record():
    assert sorted(literature.summary()) == [
        "bouthillier", "comparators", "geirhos", "nadeau_bengio",
        "sources", "split_randomised_arm", "tag_discipline",
    ]


# --------------------------------------------------------------------------
# [2026-09-02] Two corrections found at the Phase 23 verification
# --------------------------------------------------------------------------


def _flat23(text: str) -> str:
    return " ".join(text.split())


def test_the_seventeen_is_corrected_to_nineteen_and_counted_at_source():
    """It entered the amendment as a phase's motivation without ever
    being counted."""
    from collections import Counter

    from cleft import phase21, results_ledger

    statuses = Counter(str(e.get("status")) for e in results_ledger.ENTRIES)
    assert statuses["UNRESOLVED-WITHDRAWN"] == 8
    assert statuses["WITHDRAWN"] == 11
    assert 8 + 11 == 19
    # 17 is no grouping of the ledger's statuses.
    assert 17 not in set(statuses.values())
    assert 17 != 8 + 11

    motivation = phase21.PHASE_SEQUENCE_EXTENDED_7[
        "phase_23_the_statistical_instruments"]["motivated_by"]
    assert "**19 unresolved-or-withdrawn ledger ROWS**" in _flat23(motivation)
    assert "CORRECTED 2026-09-02" in motivation
    # The original is preserved in the correction record.
    record = phase21.THE_SEVENTEEN_WAS_NEVER_COUNTED
    assert "17 unresolved ledger rows" in record["the_original"]
    assert "the record's" in _flat23(record["whose_error"])
    assert "No turn ever counted it" in _flat23(record["whose_error"])
    # The direction that made it hard to notice is named.
    assert "does not change the decision is the hardest kind to notice" in (
        _flat23(record["what_it_cost"])
    )


def test_the_row_versus_contrast_distinction_is_recorded():
    from cleft import phase22, results_ledger

    record = __import__(
        "cleft.phase21", fromlist=["x"]
    ).THE_SEVENTEEN_WAS_NEVER_COUNTED
    unit = _flat23(record["the_unit_the_count_needs"])
    assert "ROWS AGGREGATE CONTRASTS" in unit
    assert "nine verdicts in one row" in unit
    assert "19 ROWS" in unit and "29 WITHDRAWN CONTRASTS" in unit
    assert "neither may be written as the other" in unit
    # Row 5 really is nine verdicts.
    assert "nine augmentation verdicts" in results_ledger.ENTRIES[5]["claim"]
    # And it cites the sibling reconciliation, which exists.
    assert hasattr(phase22, "CONDITION_PHENOMENON_COUNT_RECONCILED")


def test_the_sweep_left_road_bs_own_seventeen_alone():
    """A real 17 about a different quantity, correct as it stands."""
    from cleft import roadb

    results = roadb.PHASE_7_PAIRED_RESULTS
    assert results["withdrawn"] == 17
    assert results["survived"] == 3
    assert "20 pairs" in results["run"]
    assert 3 + 17 == 20

    record = __import__(
        "cleft.phase21", fromlist=["x"]
    ).THE_SEVENTEEN_WAS_NEVER_COUNTED
    sweep = _flat23(record["the_sweep"])
    assert "ONE occurrence in the ledger-row sense" in sweep
    assert "CORRECT AS IT STANDS -- not touched" in sweep
    # The origin is offered as a hypothesis, not asserted.
    origin = _flat23(record["a_plausible_origin_offered_as_a_HYPOTHESIS"])
    assert "NOT as an established provenance" in origin
    assert "the number's origin is unknown" in origin


def test_the_partition_sensitivity_claim_is_withdrawn_with_both_reasons():
    from pathlib import Path

    from cleft import literature

    record = literature.PARTITION_SENSITIVITY_CLAIM_WITHDRAWN
    assert "0.068" in record["the_original"]
    assert "ARRIVED AT THE SAME CONCLUSION INDEPENDENTLY" in record["the_original"]

    # Defect 1: no banked source.
    one = _flat23(record["defect_1_the_figure_has_no_source"])
    assert "appears NOWHERE in the record except the entry asserting it" in one
    assert "no phase4 module" in one
    repo = Path(__file__).resolve().parents[1]
    assert not (repo / "src" / "cleft" / "phase4.py").exists()
    # The arm itself is real, which the record says.
    assert (repo / "configs" / "p4_partition.yaml").is_file()

    # Defect 2: not the same quantity, in the task's own words.
    two = _flat23(record["defect_2_the_quantities_are_not_comparable"])
    assert "NOT fold-assignment variance" in two
    assert "fold COUNTS" in two
    run_source = _flat23(
        (repo / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    )
    assert "NOT fold-assignment variance" in run_source
    assert "the frozen generator is deterministic" in run_source

    # The withdrawal does not overshoot.
    remains = _flat23(record["what_remains_true_so_the_withdrawal_does_not_overshoot"])
    assert "Bouthillier's finding stands as banked" in remains
    assert "limitation stands" in remains
    assert "Only the claim to have arrived there independently is withdrawn" in remains
    assert "has NOT independently measured split-induced variance" in _flat23(
        record["and_what_is_now_a_stated_gap"]
    )
    # And the entry it corrects carries the withdrawal in place.
    entry = literature.BOUTHILLIER_SAMPLE_SIZE
    assert "WITHDRAWN 2026-09-02" in entry["this_project_arrived_there_independently"]
    assert "ORIGINAL, PRESERVED" in entry["this_project_arrived_there_independently"]
    assert "has NOT independently measured" in _flat23(entry["what_replaces_it"])


def test_the_same_digit_collision_hazard_is_named_at_three_instances():
    from pathlib import Path

    from cleft import literature

    record = literature.SAME_DIGIT_COLLISIONS
    assert sorted(record["the_three"]) == ["0_068", "0_1386", "0_628"]

    # The 0.068 collision is real: ladder carries it as SHA drift.
    repo = Path(__file__).resolve().parents[1]
    ladder_source = (repo / "src" / "cleft" / "ladder.py").read_text(
        encoding="utf-8"
    )
    assert "0.068 PCC drift" in ladder_source
    assert "SHA DRIFT" in record["the_three"]["0_068"]

    why = _flat23(record["why_it_is_a_hazard_and_not_a_coincidence"])
    assert "reads as corroboration instead of as an error" in why
    assert "quote figures from their structured source" in _flat23(
        record["the_defence_that_works"]
    )


def test_split_randomisation_is_recorded_as_unreachable_without_scoping():
    """Measured on this machine, and confirmed in frozen apparatus."""
    from pathlib import Path

    import numpy as np

    from cleft import literature
    from cleft.data import folds

    record = literature.SPLIT_RANDOMISATION_IS_NOT_REACHABLE
    assert "60 of 60 keep their fold" in _flat23(record["the_measurement"])

    # Re-run the measurement rather than trusting the sentence.
    rng = np.random.default_rng(0)
    ids = list(range(1, 61))
    labels = rng.integers(0, 3, 60).tolist()
    a = folds.generate(patient_ids=ids, labels=labels, seed=1337)
    b = folds.generate(patient_ids=ids, labels=labels, seed=2024)
    assert a.assignments == b.assignments, "the seed now moves the partition"

    # The frozen module says so itself.
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "src" / "cleft" / "data" / "folds.py").read_text(
        encoding="utf-8"
    )
    assert "does not affect the partition" in source
    assert "shuffle=False" in source
    assert "different splitter, not a seed here" in _flat23(
        record["why_the_frozen_module_already_says_so"]
    )

    # The cost, and the refusal to scope.
    cost = _flat23(record["the_cost_from_the_banks_own_N"])
    assert "58 runs" in cost and "1,972 runs" in cost
    assert 68 * 29 == 1972
    scope = _flat23(record["what_this_record_does_and_does_not_do"])
    assert "does NOT scope Phase 23" in scope
    assert "ADDRESS it rather than INHERIT it" in scope
