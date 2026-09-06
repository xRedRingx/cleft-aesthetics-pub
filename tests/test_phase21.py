"""Phase 21 -- the ensemble probe and error-consistency diagnosis.

Registration only. Nothing is built: no task, no schema kind, no config,
no run. A negative-space test holds that state until it is ruled the
arm-set rule and Arm B's consistency threshold.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from cleft import ladder, phase7b, phase12, phase18, phase20, phase21
from cleft.data import manifest as manifest_module

REPO = Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# the reckoning, verified against the records it cites
# --------------------------------------------------------------------------


def test_the_reckoning_quotes_each_record_at_source():
    record = phase21.PHASE_21_RECKONING

    # The resolution floor, live at its own home.
    floor = ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"]
    assert floor["smallest_resolvable"]["delta"] == 0.1386
    assert "0.1386" in record["resolution_floor"]
    assert "0.04 to 0.10" in record["resolution_floor"]
    assert "0.04 to 0.10" in _flat(floor["unresolvable_band"])

    # The probe, from the constant the reckoning names.
    result = ladder.TRADE_OFF_PAIR["result"]
    assert result["vit_paired_mean"] == 0.2520
    assert result["vit_sd"] == 0.0148
    assert "0.2520, sd 0.0148" in record["the_probe"]
    assert "TRADE_OFF_PAIR" in record["the_probe"]

    # Phase 20's stability figures, all four.
    sds = phase20.SEED_STABILITY_EVIDENCE["sds_by_arm"]
    for value in (0.1109, 0.0802, 0.0222, 0.0148):
        assert str(value) in record["seed_stability"], value
    assert [sds[k]["sd"] for k in sds] == [0.1109, 0.0802, 0.0222, 0.0148]


def test_the_arm_count_is_counted_not_quoted():
    """68 is asserted by counting ARM_LIST_LOCKED, not by trusting a
    sentence -- the group totals are pinned so a change fires here."""
    included = phase18.ARM_LIST_LOCKED["included"]
    per_group = [len(spec["runs"]) for spec in included.values()]
    assert per_group == [12, 18, 1, 4, 3, 2, 3, 10, 12, 3]
    assert sum(per_group) == 68
    assert len(included) == 10
    assert "68 arms, COUNTED AT SOURCE" in phase21.PHASE_21_RECKONING[
        "the_arm_list"
    ]
    assert "12+18+1+4+3+2+3+10+12+3" in phase21.PHASE_21_RECKONING["the_arm_list"]

    # Every included arm has per-seed CSVs BY CONSTRUCTION: the exclusion
    # that guarantees it exists and is named.
    assert "no_per_seed_oof_csvs" in phase18.ARM_LIST_LOCKED["excluded"]


def test_arm_a_is_not_covered_and_phase_7bs_reason_cannot_apply():
    """Phase 7B ruled out AVERAGING for a dimensional reason. Predictions
    are scalars, so the reason cannot reach this arm."""
    record = phase21.PHASE_21_RECKONING

    # Phase 7B's ensembling really is feature concatenation.
    assert phase7b.ENSEMBLE_COMBINES == (
        "concat", "concat_standardised", "concat_l2norm"
    )
    assert "concat" in _flat(record["a_is_not_covered"])
    assert "FEATURE concatenation" in _flat(record["a_is_not_covered"])

    # And its stated reason is quoted, at source. The `#:` comment
    # markers survive plain whitespace normalisation, so they are
    # stripped before matching -- otherwise a quote spanning two comment
    # lines would never match and the check would be vacuous.
    import inspect

    source = " ".join(
        token for token in inspect.getsource(phase7b).split() if token != "#:"
    )
    for phrase in ("Averaging is unavailable",
                   "ViT-B/16 768 dims and Swin-B 1024",
                   "a projection is a learned layer rather than an ensemble"):
        assert phrase in source, f"not at source: {phrase}"
        assert phrase in _flat(record["a_is_not_covered"]), f"not quoted: {phrase}"

    why = _flat(record["why_that_reason_cannot_apply_to_arm_a"])
    assert "PREDICTIONS ARE SCALARS" in why
    assert "does not exist at the PREDICTION layer" in why

    # Coverage that DOES exist is conceded rather than ignored.
    conceded = _flat(record["coverage_conceded_where_it_exists"])
    assert "+0.0073" in conceded and "withdrawn" in conceded


def test_arm_b_is_not_covered_and_the_gap_is_named_as_structural():
    record = phase21.PHASE_21_RECKONING
    not_covered = _flat(record["b_is_not_covered"])
    assert "no error-consistency work of any kind exists" in not_covered
    assert "_band_error_by_grade" in not_covered
    assert "BY GRADE" in not_covered

    # The near-miss is real and really does group by grade, not patient.
    import inspect

    from cleft import run as run_module

    assert "_band_error_by_grade" in dir(run_module)
    body = inspect.getsource(run_module._band_error_by_grade)
    assert "by_grade" in body

    structural = _flat(record["why_the_gap_is_structural"])
    assert "AGGREGATE PCC" in structural
    assert "one number per seed" in structural
    assert "INVISIBLE TO EVERY INSTRUMENT" in structural

    # [2026-09-01, THIS PIN FIRED AS DESIGNED] It held the negative space
    # "nothing in src implements error consistency" until Arm B was
    # built. Arm B is built, so the pin is updated with the date rather
    # than deleted -- what it now holds is that the implementation lives
    # in ONE place and that the reckoning's claim was true when made.
    implementers = sorted(
        path.name for path in (REPO / "src" / "cleft").rglob("*.py")
        if "error_consistency" in path.read_text(encoding="utf-8").lower()
    )
    # phase21 registers it, run implements it, schema names the kind,
    # literature banks the source [2026-09-01] -- and nothing else
    # mentions it.
    assert implementers == [
        "literature.py", "phase21.py", "run.py", "schema.py",
    ], implementers
    # Geirhos is CITED in the registration and in the task that
    # implements it -- a citation beside the code is the point -- and
    # nowhere else.
    citers = sorted(
        path.name for path in (REPO / "src" / "cleft").rglob("*.py")
        if "geirhos" in path.read_text(encoding="utf-8").lower()
    )
    # [2026-09-01] literature.py joined when the source was banked --
    # the registration cites it, the task implements it, and the bank
    # records what it says. Three homes, one implementation.
    # [UPDATED 2026-09-01] phase22.py joined as a fourth CITER when the
    # restate registered its feature-difference diagnostic -- the OOD
    # band is quoted there as CONTEXT beside the derived low threshold,
    # never as the threshold. Still ONE implementation: phase22 does not
    # appear in `implementers` above, because it computes nothing.
    assert citers == [
        "literature.py", "phase21.py", "phase22.py", "run.py",
    ], citers
    import inspect

    assert "Geirhos, Meding & Wichmann (NeurIPS 2020)" in inspect.getdoc(
        run_module.task_p21_error_consistency
    )


# --------------------------------------------------------------------------
# the heterogeneity nobody had counted
# --------------------------------------------------------------------------


def test_the_cohort_and_seed_heterogeneity_is_measured_not_assumed():
    included = phase18.ARM_LIST_LOCKED["included"]

    # Four arms on 236, sixty-four on 237.
    by_cohort: dict[int, int] = {}
    for spec in included.values():
        by_cohort[spec["n_patients"]] = by_cohort.get(
            spec["n_patients"], 0
        ) + len(spec["runs"])
    assert by_cohort == {237: 64, 236: 4}
    assert included["p12_view_ablation"]["n_patients"] == 236

    record = phase21.COHORT_AND_SEED_HETEROGENEITY
    cohort = _flat(record["four_arms_are_on_236_not_237"])
    assert "p12_view_ablation carries n_patients: 236" in cohort
    assert "MISALIGNED PATIENT SETS" in cohort
    # Phase 12's own refusal of the analogous move is real.
    assert "different cohort (236 vs 237)" in _flat(
        phase12.PAIRED_CLAIM_COVERAGE["excluded"][
            "any_pair_against_the_237_ladder"
        ]
    )

    # Thirty arms on ten seeds, thirty-eight on five.
    by_seeds: dict[int, int] = {}
    for spec in included.values():
        by_seeds[len(spec["seeds"])] = by_seeds.get(
            len(spec["seeds"]), 0
        ) + len(spec["runs"])
    assert by_seeds == {5: 38, 10: 30}

    # The ten-seed set really is a superset starting with the same five.
    five = tuple(included["p7_transformer"]["seeds"])
    ten = tuple(included["p7_graph"]["seeds"])
    assert five == (1337, 2024, 7, 99, 12345)
    assert ten[:5] == five
    assert set(five) < set(ten)

    seeds = _flat(record["thirty_arms_have_ten_seeds"])
    assert "p7_graph (18) and roadb_resolution_graph (12)" in seeds
    assert "strict SUPERSET" in seeds
    assert "must be DECLARED, not assumed" in seeds
    assert "R2 shape" in _flat(record["neither_is_a_blocker"])


# --------------------------------------------------------------------------
# the selection problem: the ruling is OPEN and the prohibition is not
# --------------------------------------------------------------------------


def test_the_arm_set_rule_is_open_and_not_invented():
    """the ruling arrived as an empty placeholder. The record must
    say so rather than fill it in."""
    record = phase21.ARM_SET_RULE_PROPOSED
    status = _flat(record["status"])
    assert "AWAITING A RULING" in status
    assert "arrived EMPTY" in status
    assert "inventing one and attributing it would be worse" in status

    # The half that IS ruled is recorded as binding.
    constraint = _flat(record["the_constraint_is_ruled_and_binds"])
    assert "NEVER CONSULTS OOF PCC" in constraint
    assert "BEFORE ANY AVERAGING" in constraint

    # Three candidates, none of which consults a result, and a stated
    # preference rather than a menu.
    candidates = [k for k in record if k.startswith("candidate_")]
    assert len(candidates) == 3
    assert "registration's preference" in _flat(
        record["candidate_2_all_237_cohort_arms_64"]
    )
    # Candidate 3's own weakness is named, not hidden.
    assert "SCHEDULING ARTIFACT" in _flat(
        record["candidate_3_one_per_architecture_family"]
    )

    # The draft criteria say the rule is unsettled.
    unsettled = _flat(phase21.EXIT_CRITERIA_DRAFT["what_is_not_yet_settled"])
    assert "empty placeholder" in unsettled


def test_the_selection_prohibition_is_a_tested_literal():
    from cleft import phase10_annex

    prohibition = phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION
    assert isinstance(prohibition, str)
    # Sibling to the three that already exist.
    for other in (ladder.DETECTION_FLOOR_PROHIBITION,
                  phase10_annex.ANNEX_PROHIBITION,
                  phase20.RESIDUAL_PROHIBITION):
        assert isinstance(other, str)

    assert "NEVER SELECTED FOR THE ENSEMBLE BY THEIR OOF PCC" in prohibition
    assert "selection on the evaluation set" in prohibition
    assert "SWEEP-REPORTING-ITS-BEST FAILURE IN A NEW COSTUME" in prohibition
    # It refuses the sensitivity-check escape hatch explicitly.
    assert "not quoted as a sensitivity check" in prohibition
    assert "not a weaker version of this arm" in prohibition
    # The precedent it names is real.
    assert "near-constant predictor first in a 68-arm ladder" in " ".join(
        (REPO / "src" / "cleft" / "phase18.py").read_text(
            encoding="utf-8"
        ).split()
    )


# --------------------------------------------------------------------------
# the two arms
# --------------------------------------------------------------------------


def test_arm_a_is_a_simple_mean_with_the_reason_it_cannot_be_learned():
    record = phase21.ARM_A_REGISTERED
    design = _flat(record["design"])
    assert "SIMPLE MEAN" in design and "WITHIN EACH SEED" in design
    assert "OUT-OF-FOLD" in design and "five seeds" in design
    assert "BOTH of PLAN 4.3's conditions" in design

    why = _flat(record["why_a_simple_mean_and_not_a_learned_one"])
    assert "the prohibition again with extra steps" in why
    assert "NO FREE PARAMETERS" in why

    # It runs on cached CSVs; no new hash.
    inputs = _flat(record["inputs"])
    assert "seed_<n>__predictions.csv" in inputs
    assert "No GPU, no new artifact, no new hash" in inputs
    # And that CSV contract is real.
    assert "seed_<n>__predictions.csv" in " ".join(
        (REPO / "src" / "cleft" / "phase18.py").read_text(
            encoding="utf-8"
        ).split()
    )

    # The prior is registered MODEST, before the number.
    prior = _flat(record["the_prior"])
    assert "+0.0073" in prior and "WITHDRAWN" in prior
    assert "0.1386" in prior
    assert "the registered expectation is reading 2" in prior


def test_arm_b_registers_two_binarisations_that_are_never_averaged():
    record = phase21.ARM_B_REGISTERED
    assert "Geirhos, Meding & Wichmann (NeurIPS 2020)" in record["the_method"]
    assert "(c_obs - c_exp) / (1 - c_exp)" in record["the_method"]
    assert "p1*p2 + (1-p1)*(1-p2)" in record["the_method"]

    binarisations = record["two_binarisations_both_reported"]
    assert sorted(binarisations) == [
        "not_averaged_together", "residual_sign",
        "worst_quartile_absolute_residual",
    ]
    assert "0.25 BY CONSTRUCTION" in binarisations[
        "worst_quartile_absolute_residual"
    ]
    assert "R2 shape" in _flat(binarisations["not_averaged_together"])

    deliverables = _flat(record["deliverables"])
    for piece in ("pairwise consistency matrix", "HARDNESS VECTOR",
                  "INTER-RATER DISAGREEMENT"):
        assert piece in deliverables, piece
    assert record["status"] == "DESCRIPTIVE, no ledger row"


# --------------------------------------------------------------------------
# the inter-rater disagreement, and the tie constraint
# --------------------------------------------------------------------------


def test_the_rater_sd_is_derivable_not_a_column_and_the_record_says_so():
    record = phase21.INTER_RATER_DISAGREEMENT_MEASURED

    # It really is not a manifest column.
    columns = [name for name, _ in manifest_module.MANIFEST_COLUMNS]
    assert "sd" not in columns
    assert not any("sd" in name for name in columns)
    for k in range(1, 6):
        assert f"soft_{k}" in columns
    assert "false as a column" in _flat(record["it_is_not_a_manifest_column"])

    # The reconstruction is exact -- re-measured here, not trusted.
    rng = np.random.default_rng(20260831)
    worst = 0.0
    for _ in range(4000):
        grades = rng.integers(1, 6, size=5)
        soft = np.array([(grades == g).sum() / 5 for g in range(1, 6)])
        counts = np.rint(soft * 5).astype(int)
        rebuilt = np.repeat(np.arange(1, 6), counts)
        worst = max(worst, abs(float(np.std(grades, ddof=1))
                               - float(np.std(rebuilt, ddof=1))))
    assert worst < 1e-12
    assert "4.44e-16" in record["verified_not_argued"]

    # The soft-label module's own premise, at source. It lives in the
    # comment above TOLERANCE, not in the module docstring.
    from cleft.data import softlabels

    softlabels_source = " ".join(
        token
        for token in (REPO / "src" / "cleft" / "data" / "softlabels.py")
        .read_text(encoding="utf-8").split()
        if token != "#:"
    )
    assert (
        "Soft labels are fractions of five raters, so every value is a "
        "multiple of 0.2" in softlabels_source
    )
    assert "multiple of 0.2" in _flat(record["how_it_is_recovered"])
    assert softlabels.SOFT_COLUMNS == (
        "soft_1", "soft_2", "soft_3", "soft_4", "soft_5"
    )


def test_only_26_distinct_disagreement_values_exist_and_that_is_registered():
    """A tie constraint discovered before the run, not in it."""
    from itertools import combinations_with_replacement

    values = {
        round(float(np.std(combo, ddof=1)), 9)
        for combo in combinations_with_replacement(range(1, 6), 5)
    }
    assert len(values) == 26
    assert min(values) == 0.0
    assert round(max(values), 4) == 2.1909

    record = phase21.INTER_RATER_DISAGREEMENT_MEASURED
    stated = _flat(record["only_26_distinct_values"])
    assert "only 26 DISTINCT SD VALUES" in stated
    assert "2.1909" in stated
    assert "HEAVILY TIED" in stated
    assert "rules out a naive Pearson correlation" in stated
    assert "BEFORE the run" in stated
    # The handling is declared in advance.
    assert "Spearman with a tie correction" in _flat(
        record["the_cross_reference_must_handle_ties"]
    )

    # The second route exists, and the reason for cross-checking is the
    # project's own defect history.
    second = _flat(record["a_second_route_for_cross_check"])
    assert "run.phase10_rater_grades" in second
    assert "already produced two defects" in second
    from cleft import run as run_module

    assert hasattr(run_module, "phase10_rater_grades")


# --------------------------------------------------------------------------
# the threshold, and the honesty about its two halves
# --------------------------------------------------------------------------


def test_the_low_threshold_is_derived_from_a_measured_null():
    record = phase21.CONSISTENCY_THRESHOLD_PROPOSED
    assert record["status"].startswith("PROPOSED -- NOT LOCKED")

    def kappa(e1, e2):
        c_obs = float(np.mean(e1 == e2))
        p1, p2 = float(np.mean(e1)), float(np.mean(e2))
        c_exp = p1 * p2 + (1 - p1) * (1 - p2)
        return 0.0 if c_exp >= 1.0 else (c_obs - c_exp) / (1 - c_exp)

    rng = np.random.default_rng(20260831)
    for p, sd_stated in ((0.50, 0.0651), (0.25, 0.0649)):
        draws = np.array([
            kappa(rng.random(237) < p, rng.random(237) < p)
            for _ in range(4000)
        ])
        assert abs(float(draws.mean())) < 0.01, "null is centred on zero"
        assert float(draws.std(ddof=1)) == pytest.approx(sd_stated, abs=0.004)
        assert str(sd_stated) in record["the_measured_null"]

    # The mean over pairs is tighter, and the LOW threshold clears it.
    draws = np.array([
        np.mean([kappa(rng.random(237) < 0.25, rng.random(237) < 0.25)
                 for _ in range(50)])
        for _ in range(400)
    ])
    p99 = float(np.percentile(draws, 99))
    assert p99 < 0.05, "the proposed LOW threshold sits above the null p99"
    assert 0.05 > 2 * p99, "and by more than a factor of two, as claimed"
    assert "0.022" in record["the_mean_over_pairs_is_far_tighter"]
    assert "<= 0.05" in record["proposed_low"]
    assert "CONSERVATIVE" in record["proposed_low"]


def test_the_high_threshold_is_flagged_as_the_weaker_of_the_two():
    """The asymmetry is the point: one threshold is measured and one is
    not, and the record must not present them as equals."""
    record = phase21.CONSISTENCY_THRESHOLD_PROPOSED
    assert ">= 0.50" in record["proposed_high"]
    assert "MIDPOINT OF ITS OWN SCALE" in record["proposed_high"]

    honest = _flat(record["the_two_are_not_equally_grounded"])
    assert "LOW is DERIVED" in honest
    assert "HIGH is PRINCIPLED BUT NOT MEASURED" in honest
    assert "NOT the Landis-Koch convention" in honest
    assert "stated rather than smoothed over" in honest
    # And the band between is wide on purpose.
    assert "(0.05, 0.50)" in record["between_is_deliberately_wide"]
    assert "partial is the honest cell" in record["between_is_deliberately_wide"]


# --------------------------------------------------------------------------
# readings, binding, sequence
# --------------------------------------------------------------------------


def test_every_reading_is_committed_including_the_combinations():
    readings = phase21.READINGS_COMMITTED
    arm_a = [k for k in readings if k.startswith("a_")]
    arm_b = [k for k in readings if k.startswith("b_")]
    combos = [k for k in readings if k.startswith("combination_")]
    assert len(arm_a) == 3 and len(arm_b) == 4 and len(combos) == 3

    # A's registered expectation is the modest one, and says so.
    assert "THE REGISTERED EXPECTATION" in readings["a_above_but_unresolved"]
    assert "neither is dressed as the other" in _flat(
        readings["a_above_but_unresolved"]
    )
    # A-at-or-below feeds Arm B, which is why the combinations exist.
    assert "predicts HIGH consistency" in _flat(
        readings["a_at_or_below_the_probe"]
    )

    # The coherent account and the contradiction are both named, and the
    # contradiction is NOT given a story.
    assert "COHERENT SHARED-CEILING ACCOUNT" in readings[
        "combination_a_fails_and_b_high"
    ]
    contradiction = _flat(readings["combination_a_fails_and_b_low"])
    assert "A CONTRADICTION REQUIRING EXPLANATION, NOT A STORY" in contradiction
    assert "so it cannot be narrated away after the fact" in contradiction
    # What would have to be checked is named in advance, all three.
    for probe in ("(i)", "(ii)", "(iii)"):
        assert probe in contradiction, probe

    # The unpredicted-pattern escape is the live precedent, not a theory.
    after = _flat(readings["no_reading_is_invented_after"])
    assert "phase17.UNPREDICTED_PATTERN" in after
    assert "Phase 20's Arm S required" in after
    assert phase20.S_PATTERN_UNPREDICTED["no_cell_fired"]


def test_neither_arm_may_touch_phase_20s_bounded_territory():
    binding = phase21.PHASE_21_BINDING
    assert "phase20.RESIDUAL_PROHIBITION is unaffected" in _flat(
        binding["the_binding"]
    )
    tempting = _flat(binding["the_distinction_that_will_be_tempting"])
    assert "what the MODEL CANNOT REACH" in tempting
    assert "not a claim about what the 0.2520 CONTAINS" in tempting
    assert "CEILING'S LOCUS" in tempting

    # The figures it cites are live at source.
    assert "60.2% retained" in binding["arm_b_cannot_unbound_phase_20"]
    assert phase20.ARMS_OBSERVED["diagnostic"]["retained_fraction_mean"] == 0.602
    assert "0.2520 - 0.0414 = 0.2106 IS NOT CLEFT-SPECIFIC" in (
        phase20.RESIDUAL_PROHIBITION
    )


def test_the_sixth_amendment_extends_and_tests_the_fifths_rule():
    from cleft import phase11, phase15

    sixth = phase21.PHASE_SEQUENCE_EXTENDED_6
    assert sixth["becomes"]["19"].startswith("write-up (UNCHANGED")
    assert "NOTHING" in sixth["was"]["21"]
    assert sixth["status_changes"]["write_up"] == (
        "Phase 19 -> Phase 19 (unchanged)"
    )

    # The full backward chain, all five.
    for key, target in (
        ("first_amendment", "phase11.PHASE_SEQUENCE_RENUMBERED"),
        ("second_amendment", "phase12.PHASE_SEQUENCE_RENUMBERED_2"),
        ("third_amendment", "phase15.PHASE_SEQUENCE_RENUMBERED_3"),
        ("fourth_amendment", "phase15.PHASE_SEQUENCE_RENUMBERED_4"),
        ("fifth_amendment", "phase20.PHASE_SEQUENCE_EXTENDED_5"),
    ):
        assert sixth[key].startswith(target), key

    # It is the TEST of the fifth's rule, and says so.
    test_of = _flat(sixth["it_is_the_test_of_the_fifth_amendments_rule"])
    assert "written UNDER the write-up-runs-last rule" in test_of
    assert "forces none under this one" in test_of
    assert "THE WRITE-UP RUNS LAST REGARDLESS OF ITS NUMBER" in _flat(
        phase20.PHASE_SEQUENCE_EXTENDED_5[
            "the_write_up_is_last_by_rule_not_by_number"
        ]
    )

    # All five earlier amendments carry the dated pointer.
    for amendment in (
        phase11.PHASE_SEQUENCE_RENUMBERED,
        phase12.PHASE_SEQUENCE_RENUMBERED_2,
        phase15.PHASE_SEQUENCE_RENUMBERED_3,
        phase15.PHASE_SEQUENCE_RENUMBERED_4,
        phase20.PHASE_SEQUENCE_EXTENDED_5,
    ):
        assert amendment["sixth_amendment"] == (
            "phase21.PHASE_SEQUENCE_EXTENDED_6"
        )


def test_the_exit_criteria_are_a_draft_and_name_what_is_open():
    draft = phase21.EXIT_CRITERIA_DRAFT
    assert draft["status"].startswith("DRAFT")
    assert len(draft["criteria"]) == 11
    joined = " ".join(draft["criteria"])
    assert "never consults OOF PCC" in joined
    assert "SELECTION_ON_EVALUATION_DATA_PROHIBITION" in joined
    assert "THE COHORT QUESTION ANSWERED EXPLICITLY" in joined
    assert "never averaged together" in joined
    assert "TIE-AWARE method declared in advance" in joined
    assert "BOTH routes" in joined
    assert "suite green" in joined
    # Arm A's ledger row is NOT assumed.
    assert "Arm A's row is to be ruled and is NOT assumed" in joined


# --------------------------------------------------------------------------
# the negative space -- nothing is built
# --------------------------------------------------------------------------


def test_nothing_is_built_for_phase_21():
    """**[UPDATED 2026-09-01 -- THIS PIN FIRED, AS DESIGNED.]**

    It held the registration-only state "until it is ruled". He
    ruled, both arms are built, and the pin is updated with the date
    rather than deleted. What it holds now is the one thing that must
    stay true across the build: **the arms read cached predictions and
    nothing else, and no ledger row appeared.**
    """
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    for kind in ("p21_ensemble_probe", "p21_error_consistency"):
        assert kind in TASK_SPECS and kind in TASKS, kind
    # [2026-09-01] Two reckoning-input configs joined -- neither is
    # Phase 21 machinery either; they feed Phase 22's restate. The
    # pin fired; updated dated.
    assert sorted(p.name for p in (REPO / "configs").glob("p21_*.yaml")) == [
        "p21_cohort_pair_separation.yaml", "p21_cross_arm_shrinkage.yaml",
        "p21_ensemble_probe.yaml", "p21_error_consistency.yaml",
    ]
    assert (REPO / "scripts" / "generate_phase21_configs.py").is_file()

    # No embeddings, no staged pixels: neither config declares one.
    import yaml

    for stem in ("p21_ensemble_probe", "p21_error_consistency"):
        config = yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )
        names = {e["name"] for e in config["inputs"]}
        assert not [n for n in names if "embedding" in n], stem
        assert not [n for n in names if "staged" in n], stem
        assert "cleft_reconstruction_acknowledged" not in config["task"]

    # Still no ledger row -- Arm B is descriptive, Arm A's is unruled.
    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 21 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p21" or "p21-" in e["id"]
    ]
    assert results_ledger.validate() is None
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p21" or "ensemble" in e["id"]
    ]


def test_the_summary_carries_every_record():
    # [UPDATED 2026-09-01] Seven added by the three rulings, the lock,
    # the statistic proposal, the rejected candidate, and the tau-b
    # defect found while building.
    # [UPDATED 2026-09-01, close-out] Nine added by the two arms'
    # outturns, the post-hoc analyses, the shrinkage mechanism and its
    # prohibition, the reading error, the combination cell, the forward
    # lead, and the closing. This pin fires on every fill by design.
    # [UPDATED 2026-09-01, later] Two more: the seventh amendment (a
    # dated addendum to a closed phase, scheduling 22-25) and the
    # write-up's number recorded as OPEN rather than resolved.
    # [UPDATED 2026-09-01, three items] Four more: the 19 ruling, the
    # 0.477 mis-attribution, the cohort-pair separation design, and the
    # pairing-axis assumption.
    # [UPDATED 2026-09-01] Two more: the cohort-pair separation build
    # and the cross-arm shrinkage registration, both reckoning inputs.
    # [2026-09-01, later] One more: the dispersion statistic ruled,
    # keeping the rejected range version with its simulation.
    # [UPDATED 2026-09-01, both measurements ran] Four more: the two
    # outturns, the arm A shrinkage control, and what it forces on
    # Phase 22's reckoning.
    # [UPDATED 2026-09-02] The seventeen correction, filed beside
    # the other named error-provenance records.
    assert sorted(phase21.summary()) == [
        "arm_a",
        "arm_a_explained_by_arm_b",
        "arm_a_observed",
        "arm_a_shrinkage_control",
        "arm_b",
        "arm_b_observed",
        "arm_set_candidate_rejected",
        "arm_set_rule",
        "arm_set_ruled",
        "binding",
        "closing",
        "cohort_pair_separation",
        "cohort_pair_separation_built",
        "cohort_pair_separation_observed",
        "consistency_threshold",
        "consistency_thresholds_ruled",
        "cross_arm_shrinkage",
        "cross_arm_shrinkage_observed",
        "disagreement_statistic",
        "exit_criteria",
        "exit_criteria_draft",
        "heterogeneity",
        "inter_rater_disagreement",
        "pairing_axis_assumption",
        "phase_22_consequence",
        "post_hoc",
        "ranking_losses_noted",
        "readings",
        "reckoning",
        "scale_invariance_prohibition",
        "seed_basis_ruled",
        "selection_prohibition",
        "sequence_extended_6",
        "sequence_extended_7",
        "shrinkage_dispersion_ruled",
        "shrinkage_figure_misattributed",
        "shrinkage_mechanism",
        "shrinkage_reading_error",
        "tau_b_defect",
        "the_seventeen_was_never_counted",
        "write_up_number_open",
        "write_up_number_ruled",
    ]


# --------------------------------------------------------------------------
# [2026-09-01] The three rulings, the lock, and the build
# --------------------------------------------------------------------------


def test_the_arm_set_is_ruled_and_the_filter_reproduces_it():
    ruled = phase21.ARM_SET_RULED
    assert ruled["the_set"] == "all 64 arms on the 237 cohort"

    # Reproduced from the lock, not trusted.
    entries = phase18.locked_arm_entries()
    keep = [e for e in entries if e["n_patients"] == 237]
    drop = [e for e in entries if e["n_patients"] != 237]
    assert len(entries) == 68 and len(keep) == 64 and len(drop) == 4
    assert sorted(e["name"] for e in drop) == [
        "p12_arm_a_frontal", "p12_arm_b_basal",
        "p12_arm_c_concat", "p12_arm_d_capacity",
    ]
    assert {e["group"] for e in drop} == {"p12_view_ablation"}
    assert len({e["run_dir"] for e in keep}) == 63

    # Exactly one exclusion, on a data property.
    one = _flat(ruled["exactly_one_exclusion"])
    assert "236 PATIENTS, NOT 237" in one
    assert "MISALIGNED PATIENT SETS" in one
    data_property = _flat(ruled["it_is_a_data_property_not_a_performance_one"])
    assert "without opening a single prediction file" in data_property
    assert "does not consult it, and could not have" in data_property
    assert "before the arms were built" in _flat(
        ruled["declared_before_any_averaging"]
    )
    assert phase21.ARM_SET_RULE_PROPOSED["ruled_2026_09_01"] == "ARM_SET_RULED"


def test_the_rejected_candidate_carries_its_reason_and_its_provenance():
    """A proxy for the forbidden quantity is the forbidden quantity."""
    rejected = phase21.ARM_SET_CANDIDATE_REJECTED
    assert rejected["the_rule"] == (
        "restricting to 'arms sharing the probe's recipe'"
    )
    reason = _flat(rejected["the_reason"])
    assert "expected to perform well" in reason
    assert "selection problem in a different form" in reason
    assert "pass a LITERAL reading" in reason
    assert "A PROXY FOR THE FORBIDDEN QUANTITY IS THE FORBIDDEN QUANTITY" in reason

    # The provenance gap is stated, not smoothed: this rule is NOT one
    # of the three written candidates.
    provenance = _flat(rejected["provenance_note_2026_09_01"])
    assert "NOT among the three candidates written" in provenance
    assert "not the same rule" in provenance
    written = [
        k for k in phase21.ARM_SET_RULE_PROPOSED if k.startswith("candidate_")
    ]
    assert len(written) == 3
    for key in written:
        assert "sharing the probe's recipe" not in phase21.ARM_SET_RULE_PROPOSED[key]
    # Candidate 3 really does select by a non-performance key.
    assert "alphabetically first run stem" in phase21.ARM_SET_RULE_PROPOSED[
        "candidate_3_one_per_architecture_family"
    ]
    assert "the REASON is what has force" in _flat(
        rejected["why_it_is_recorded_anyway"]
    )


def test_the_seed_basis_is_ruled_with_the_alternative_recorded():
    ruled = phase21.SEED_BASIS_RULED
    assert "1337, 2024, 7, 99, 12345" in ruled["the_basis"]
    assert "STRICT SUPERSET" in ruled["why_it_works"]
    assert "ONE THING throughout" in ruled["why_it_works"]

    # The alternative, and WHY it was not taken -- an unequal weighting
    # nobody chose.
    alt = _flat(ruled["the_alternative_not_taken"])
    assert "TWICE AS HEAVILY" in alt
    assert "NOBODY CHOSE" in alt
    assert "ladder.SEEDS_BY_KIND" in alt

    # The weighting really would have been 2:1, re-derived here.
    entries = [e for e in phase18.locked_arm_entries() if e["n_patients"] == 237]
    ten = [e for e in entries if len(e["seeds"]) == 10]
    five = [e for e in entries if len(e["seeds"]) == 5]
    assert len(ten) == 30 and len(five) == 34
    assert len(ten[0]["seeds"]) == 2 * len(five[0]["seeds"])
    assert tuple(ten[0]["seeds"])[:5] == tuple(five[0]["seeds"])


def test_the_thresholds_are_ruled_with_their_asymmetry_preserved():
    ruled = phase21.CONSISTENCY_THRESHOLDS_RULED
    assert ruled["low"] == "mean pairwise kappa <= 0.05"
    assert ruled["high"] == "mean pairwise kappa >= 0.50"

    assert "+0.022" in ruled["low_is_derived"]
    assert "harder to declare" in ruled["low_is_derived"]
    high = _flat(ruled["high_is_convention_not_measurement"])
    assert "MIDPOINT OF THAT SCALE" in high
    assert "does not borrow Landis-Koch's authority" in high
    assert "LESS-GROUNDED HALF" in high

    # The reason no measured HIGH can exist -- the part that matters.
    why = _flat(ruled["why_no_measured_high_threshold_exists"])
    assert "cannot be built without assuming the answer" in why
    assert "ENCODE the assumption rather than test it" in why
    assert "INDEPENDENCE IS A SPECIFIC, SIMULABLE HYPOTHESIS" in why
    assert "a property of the question" in why
    assert "not equally grounded" in _flat(
        ruled["the_asymmetry_is_part_of_the_ruling"]
    )
    assert phase21.CONSISTENCY_THRESHOLD_PROPOSED["ruled_2026_09_01"] == (
        "CONSISTENCY_THRESHOLDS_RULED"
    )


# --------------------------------------------------------------------------
# the tau-b defect, found while building
# --------------------------------------------------------------------------


def test_the_tau_b_defect_is_recorded_and_the_function_is_fixed():
    from cleft import phase11

    record = phase21.TAU_B_DEFECT_CORRECTED
    assert "while building this phase" in record["found"]
    assert "removed from both factors" in _flat(record["the_defect"])
    assert "0.6667 for [1, 1, 2]" in _flat(record["the_decisive_symptom"])

    # The symptom is GONE, checked directly.
    for vector in ([1, 1, 2], [1, 1, 2, 2, 3], [0, 0, 0, 1, 1, 2, 3, 3]):
        assert phase11.kendall_tau_b(vector, vector) == pytest.approx(1.0)

    # One-directional, and the record says so.
    assert "LOWER BOUND on its true value" in _flat(
        record["the_error_is_one_directional"]
    )

    # The six affected Phase 18 figures are named, and the verdict flagged.
    downstream = _flat(record["downstream_and_to_be_ruled"])
    for figure in ("0.5180", "0.3784", "0.7024", "0.6295", "0.3392663"):
        assert figure in downstream, figure
    assert "PRE-REGISTERED VERDICT" in downstream
    # Those figures are live at their source.
    phase18_text = " ".join(
        (REPO / "src" / "cleft" / "phase18.py").read_text(
            encoding="utf-8"
        ).split()
    )
    for figure in ("0.5180", "0.3784", "0.7024", "0.6295", "0.3392663"):
        assert figure in phase18_text, figure

    # "Probably" is refused as an answer, and Phase 18 is not rewritten.
    without = _flat(record["what_can_be_said_without_the_artifacts"])
    assert "Probably is not measured" in without
    assert "UPWARD, OR NOT AT ALL" in without
    assert "nothing there is rewritten by this record" in without

    # Fixed rather than worked around, with the reason.
    assert "two-implementations defect" in _flat(
        record["why_it_was_fixed_rather_than_worked_around"]
    )


def test_there_is_exactly_one_tau_b_in_the_project():
    """The fix must not have spawned a second copy.

    Name-matching is too broad -- ``phase16.anchor_tau`` is a distance
    SCALE, not a rank statistic, and shares only the letters. The
    invariant is that exactly one function COMPUTES tau-b, identified by
    its concordant/discordant machinery, and everything else delegates.
    """
    import re

    # The word "concordant" appears in prose across several phases ("d1
    # reading concordant", "concordant primaries"), so presence is too
    # broad a signal. The counter INITIALISATION is the implementation's
    # own fingerprint.
    implementations = []
    for path in sorted((REPO / "src" / "cleft").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"concordant\s*=\s*discordant\s*=", text):
            implementations.append(path.name)
    assert implementations == ["phase11.py"], implementations

    # Every other tau-b caller delegates rather than reimplements.
    from cleft import phase16, phase18

    p18 = (REPO / "src" / "cleft" / "phase18.py").read_text(encoding="utf-8")
    assert "phase11.kendall_tau_b(accuracy, qwk)" in p18
    assert "no second implementation" in p18
    # anchor_tau is a different quantity entirely -- a distance scale.
    assert "mean squared anchor-anchor" in phase16.anchor_tau.__doc__
    assert "concordant" not in phase16.anchor_tau.__doc__

    # And D3's figure now flows through the CORRECTED function.
    assert callable(phase18.tau_accuracy_vs_qwk)


# --------------------------------------------------------------------------
# the statistic proposal
# --------------------------------------------------------------------------


def test_the_disagreement_statistic_is_proposed_with_a_validated_null():
    from cleft import phase11

    record = phase21.DISAGREEMENT_STATISTIC_PROPOSED
    assert record["status"].startswith("PROPOSED 2026-09-01")
    assert "phase11.kendall_tau_b" in record["the_statistic"]
    assert "defined for ties by construction" in _flat(record["why_tau_b"])
    assert "PRECISION without introducing BIAS" in _flat(record["why_tau_b"])
    assert "26 distinct values" in _flat(record["why_not_pearson"])
    assert "One implementation, not two" in _flat(record["why_not_spearman"])

    null = _flat(record["the_null"])
    assert "PRESERVES BOTH MARGINAL TIE STRUCTURES EXACTLY" in null
    assert "not at an assumed one" in null
    assert "0.040 against a nominal 0.050" in _flat(
        record["validated_before_proposing"]
    )
    assert "picked to make a story" in _flat(
        record["declared_now_not_at_analysis_time"]
    )

    # The permutation null really does preserve both tie structures.
    rng = np.random.default_rng(20260901)
    disagreement = rng.choice(np.linspace(0.0, 2.19, 26), 237)
    hardness = rng.binomial(64, 0.5, 237).astype(float)
    before = sorted(np.bincount(
        np.unique(hardness, return_inverse=True)[1]
    ).tolist())
    shuffled = hardness.copy()
    rng.shuffle(shuffled)
    after = sorted(np.bincount(
        np.unique(shuffled, return_inverse=True)[1]
    ).tolist())
    assert before == after, "permutation preserves the tie structure exactly"

    # And the null it generates is centred on zero.
    draws = []
    work = hardness.copy()
    for _ in range(300):
        rng.shuffle(work)
        draws.append(phase11.kendall_tau_b(disagreement, work))
    assert abs(float(np.mean(draws))) < 0.02


# --------------------------------------------------------------------------
# the lock
# --------------------------------------------------------------------------


def test_the_exit_criteria_are_locked_with_the_three_rulings_closed():
    locked = phase21.EXIT_CRITERIA
    assert "2026-09-01" in locked["locked"]
    closed = locked["the_three_rulings_closed"]
    assert sorted(closed) == ["arm_set", "seed_basis", "thresholds"]
    for value in closed.values():
        assert value.startswith("RULED"), value

    assert len(locked["criteria"]) == 12
    joined = " ".join(locked["criteria"])
    assert "the probe (0.2520)" in joined
    assert "withdrawn Phase 7B concat result (+0.0073)" in joined
    assert "as though combination had never been tried" in joined
    assert "THE ALL-68 VARIANT IS NOT RUN" in joined
    assert "RECORDED WITH THE REASON" in joined
    assert "never averaged together" in joined
    assert "OBSERVED tie structure" in joined
    assert "BOTH ROUTES" in joined.upper()
    assert "suite green" in joined

    # The draft is preserved and points forward.
    assert phase21.EXIT_CRITERIA_DRAFT["status"].startswith("DRAFT")
    assert phase21.EXIT_CRITERIA_DRAFT["superseded_2026_09_01"] == (
        "EXIT_CRITERIA"
    )
    assert len(phase21.EXIT_CRITERIA_DRAFT["criteria"]) == 11


def test_the_lock_forbids_adding_to_itself_and_excludes_the_tau_question():
    locked = phase21.EXIT_CRITERIA
    clause = _flat(locked["locked"])
    assert "nothing is added after this record" in clause
    assert "never a retro-fitted entry" in clause
    # The precedent is live: Phase 20 honoured it within a day.
    assert "HONOURED IT within a day" in _flat(locked["nothing_added_after"])
    assert hasattr(phase20, "LOCK_LIMITATION_STRATIFICATION_UNVERIFIED")

    # The tau question is deliberately OUTSIDE the lock.
    outside = _flat(locked["what_this_lock_does_not_cover"])
    assert "a ruling owed" in outside
    assert "not a Phase 21 criterion" in outside
    assert "exactly the retro-fit the clause above forbids" in outside
    joined = " ".join(locked["criteria"])
    assert "tau" not in joined.lower() or "TAU_B_DEFECT" not in joined


# --------------------------------------------------------------------------
# the build: two tasks, cached predictions only
# --------------------------------------------------------------------------


def test_both_arms_are_built_and_registered():
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    for kind in ("p21_ensemble_probe", "p21_error_consistency"):
        assert kind in TASK_SPECS, kind
        assert kind in TASKS, kind
    assert (REPO / "configs" / "p21_ensemble_probe.yaml").is_file()
    assert (REPO / "configs" / "p21_error_consistency.yaml").is_file()
    assert (REPO / "scripts" / "generate_phase21_configs.py").is_file()


def test_neither_arm_reads_pixels_trains_or_globs():
    """Cached OOF predictions only: no artifact, no hash, no GPU."""
    import inspect

    from cleft import run as run_module

    reader = inspect.getsource(run_module._p21_load_arm_predictions)
    for task in (run_module.task_p21_ensemble_probe,
                 run_module.task_p21_error_consistency):
        body = inspect.getsource(task) + reader
        for forbidden in ("import torch", "prepare_features", "np.load(",
                          ".glob(", "rglob", "extract_embeddings",
                          "RidgeBackbone", "TrainConfig"):
            assert forbidden not in body, f"{forbidden} in {task.__name__}"
        # Every arm reaches the task through the config's literal list,
        # via the ONE shared reader.
        assert "_p21_load_arm_predictions(" in inspect.getsource(task)
    # The reader is shared, not duplicated, and it is the only CSV path.
    assert 'task["arms"]' in reader
    assert "PREDICTIONS_COLUMNS" in reader
    assert inspect.getsource(run_module).count(
        "def _p21_load_arm_predictions"
    ) == 1


def test_arm_a_averages_predictions_and_contrasts_both_anchors():
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p21_ensemble_probe)
    # A simple mean over the stacked arms.
    assert "stacked.mean(axis=0)" in body

    # No WEIGHTING CODE -- checked against the executable lines only, so
    # the docstring's "no weights" cannot satisfy or break the check.
    code = "\n".join(
        line for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    code = code[code.index("task = ctx.config"):]
    for forbidden in ("weight", "coef_", "lstsq", "argsort", "np.average"):
        assert forbidden not in code.lower(), f"{forbidden} in the arm's code"

    # The paired criterion, through the shipped implementation -- not a
    # bootstrap written here.
    assert "metrics.paired_delta_bca(" in body
    assert "phase3.combined_claimable_delta(" in body
    assert "condition_1" in body and "condition_2" in body

    # Both anchors reported, and the all-68 absence recorded with its
    # reason rather than left silent.
    assert 'task["concat_precedent_delta"]' in body
    assert "against_the_concat_precedent" in body
    assert "would read as though combination had never" in body
    assert "all_68_variant" in body and "NOT RUN" in body


def test_arm_b_reports_both_binarisations_separately():
    import inspect

    from cleft import run as run_module

    # [UPDATED 2026-09-01, THIS PIN FIRED ON A REFACTOR] The two
    # binarisations moved OUT of the task body into module-level
    # `error_indicators`, so Phase 22's diagnostic could call the same
    # implementation instead of writing a second. The pin now checks
    # both homes, which is what it was standing for: the names are
    # explicit and neither is a default.
    body = inspect.getsource(run_module.task_p21_error_consistency)
    indicators = inspect.getsource(run_module.error_indicators)
    assert "residual_sign" in indicators
    assert "worst_quartile" in indicators
    # Still no silent fallback: an unknown name raises.
    assert "is not a registered binarisation" in indicators
    # And the task still reads them from its config, separately.
    assert "kendall_tau_b" in body
    assert "hardness" in body
    assert "by_binarisation" in body or "binarisation" in body

    # The kappa arithmetic has ONE implementation, and the task uses it.
    assert "error_consistency_matrix(" in body
    matrix = inspect.getsource(run_module.error_consistency_matrix)
    assert "c_exp" in matrix and "c_obs" in matrix
    # It is not duplicated back into the task.
    assert "c_exp" not in body


def test_the_configs_carry_the_ruled_set_and_no_new_hash():
    import yaml

    def load(stem):
        return yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )

    # Every input must already be declared, identically, by a shipped
    # config: Phase 18 for the run directories and the manifest, Phase
    # 13's ceiling for the score sheet. No new hash enters.
    known = {}
    for donor in ("p18_metric_space", "p13_confound_ceiling"):
        for entry in load(donor)["inputs"]:
            known.setdefault(
                entry["name"], (entry.get("path"), entry.get("rollup_sha256"))
            )

    for stem in ("p21_ensemble_probe", "p21_error_consistency"):
        config = load(stem)
        # Every input is one Phase 18 already declares, identically.
        for entry in config["inputs"]:
            assert entry["name"] in known, f"{stem}: {entry['name']}"
            assert (entry.get("path"), entry.get("rollup_sha256")) == known[
                entry["name"]
            ], f"{stem}: {entry['name']} differs from p18's declaration"
            rollup = entry.get("rollup_sha256") or ""
            assert len(rollup) == 64 and set(rollup) != {"0"}, entry["name"]

        task = config["task"]
        # The ruled arm set, exactly: 64 arms, none on 236.
        assert len(task["arms"]) == 64, stem
        assert not [a for a in task["arms"] if a["n_patients"] != 237]
        assert not [a for a in task["arms"] if a["name"].startswith("p12_")]
        # The shared five seeds for every arm, without exception.
        for arm in task["arms"]:
            assert arm["seeds"] == [1337, 2024, 7, 99, 12345], arm["name"]
        # 63 distinct run-dir inputs plus the manifest.
        names = {a["input"] for a in task["arms"]}
        assert len(names) == 63


def test_the_ensemble_config_declares_both_anchors():
    import yaml

    task = yaml.safe_load(
        (REPO / "configs" / "p21_ensemble_probe.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert task["probe_pcc"] == 0.2520
    assert task["concat_precedent_delta"] == 0.0073
    assert task["probe_arm"] == "p7_d1_vit_b16_imagenet_g1"
    # The probe is IN the ruled set -- the contrast is within-phase.
    assert task["probe_arm"] in {a["name"] for a in task["arms"]}


def test_the_consistency_config_carries_the_ruled_thresholds():
    import yaml

    task = yaml.safe_load(
        (REPO / "configs" / "p21_error_consistency.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert task["low_kappa"] == 0.05
    assert task["high_kappa"] == 0.50
    assert task["n_permutations"] >= 10000
    assert task["binarisations"] == ["residual_sign", "worst_quartile"]


# --------------------------------------------------------------------------
# [2026-09-01] The close-out
# --------------------------------------------------------------------------


def _resolve21(path: str):
    """Resolve a "RECORD['key']..." pointer against phase21."""
    head, *keys = re.findall(r"^(\w+)|\['([^']+)'\]", path)
    node = getattr(phase21, head[0])
    for _, key in keys:
        node = node[key]
    return node


def test_every_fired_cell_resolves_by_identity():
    """Fired cells are named by POINTER, never retyped."""
    assert _resolve21(phase21.ARM_A_OBSERVED["cell_fired"]) is (
        phase21.READINGS_COMMITTED["a_above_but_unresolved"]
    )
    assert _resolve21(
        phase21.ARM_B_OBSERVED["cross_reference"]["cell_fired"]
    ) is phase21.READINGS_COMMITTED[
        "b_hardness_does_not_correlate_with_disagreement"
    ]
    # Arm B's own cell is named for BOTH binarisations.
    assert phase21.ARM_B_OBSERVED["cell_fired"].startswith(
        "READINGS_COMMITTED['b_high_consistency']"
    )
    assert "BOTH" in phase21.ARM_B_OBSERVED["cell_fired"]
    assert "b_high_consistency" in phase21.READINGS_COMMITTED
    # And the combination cell.
    assert "combination_a_fails_and_b_high" in phase21.ARM_A_EXPLAINED_BY_ARM_B[
        "the_cell_fired"
    ]
    assert "combination_a_fails_and_b_high" in phase21.READINGS_COMMITTED


def test_arm_a_arithmetic_reconciles_from_its_own_per_seed_values():
    record = phase21.ARM_A_OBSERVED
    per_seed = np.array(record["pcc_by_seed"])
    assert len(per_seed) == 5
    assert float(per_seed.mean()) == pytest.approx(record["pcc_mean"], abs=5e-5)
    assert float(per_seed.std(ddof=1)) == pytest.approx(record["pcc_sd"], abs=5e-5)

    deltas = np.array(record["paired_deltas_by_seed"])
    assert float(deltas.mean()) == pytest.approx(record["mean_gain"], abs=5e-5)
    assert int((deltas < 0).sum()) == 1
    assert "0 of 5" in record["n_excluding_zero"]

    # Highest in the project, above both anchors.
    assert record["pcc_mean"] > 0.2552 > 0.2520
    assert ladder.TRADE_OFF_PAIR["result"]["vit_paired_mean"] == 0.2520

    # Unresolved despite being 3.2x the concat precedent.
    assert round(record["mean_gain"] / 0.0073, 1) == 3.2
    assert "STILL NOT CLAIMABLE" in record["against_the_concat_precedent"]


def test_the_ensemble_is_less_stable_and_that_is_an_observation():
    record = phase21.ARM_A_OBSERVED
    observation = _flat(record["observation_the_ensemble_is_less_stable"])
    assert "an OBSERVATION" in observation
    assert "1.61x WIDER" in observation
    assert "opposite of what ensembling normally buys" in observation
    assert "No cell anticipated it" in observation

    # Re-derived: the ensemble spread really is 1.61x the probe's.
    per_seed = np.array(record["pcc_by_seed"])
    ratio = float(per_seed.std(ddof=1)) / ladder.TRADE_OFF_PAIR["result"]["vit_sd"]
    assert round(ratio, 2) == 1.61

    # And the duplicate caveat is stated, not buried.
    caveat = _flat(record["the_duplicate_caveat"])
    assert "DOUBLE-WEIGHTS those four" in caveat
    assert "legitimate member under ARM_SET_RULED" in caveat
    assert "NOT an equal-weight average of 64 distinct predictors" in caveat
    from cleft import phase18

    assert len(phase18.DUPLICATION_PROVEN_AT_DEPTH[
        "four_pairs_identical_on_every_seed"
    ]) == 4


def test_arm_b_bands_are_recorded_as_measured_not_as_summarised():
    record = phase21.ARM_B_OBSERVED
    assert record["residual_sign"]["n_pairs"] == 64 * 63 // 2 == 2016
    rs = record["residual_sign"]["mean_kappa"]
    wq = record["worst_quartile"]["mean_kappa"]

    # Both clear the ruled HIGH threshold.
    high = phase21.CONSISTENCY_THRESHOLDS_RULED["high_value"]
    assert rs >= high and wq >= high

    # The band comparison, re-derived -- and it is NOT "at or above the
    # top" of the CNN range for either binarisation.
    assert 0.62 <= rs <= 0.79, "residual sign is INSIDE the CNN band"
    assert wq < 0.62, "worst quartile is BELOW the CNN floor"
    assert rs > 0.48 and wq > 0.48, "both above the human ceiling"
    assert round(0.79 - rs, 4) == 0.0888
    assert round(0.62 - wq, 4) == 0.0471

    measured = _flat(record["geirhos_bands_measured_against"])
    assert "INSIDE the CNN-to-CNN band" in measured
    assert "BELOW the CNN floor" in measured
    assert "neither sits 'at or above the top'" in measured
    assert "the kind of sentence that drifts" in measured

    # Context, not a threshold -- and the lock was not moved.
    assert phase21.CONSISTENCY_THRESHOLDS_RULED["high_value"] == 0.50
    assert "reported BESIDE the result" in record["context_not_a_threshold"]


def test_the_cross_reference_rules_out_label_noise():
    cross = phase21.ARM_B_OBSERVED["cross_reference"]
    rs = cross["residual_sign"]
    wq = cross["worst_quartile"]
    assert rs["tau_b"] == 0.0727 and rs["permutation_p"] == 0.1226
    assert wq["tau_b"] == -0.0811 and wq["permutation_p"] == 0.0835
    # Insignificant, and opposite in sign.
    assert rs["permutation_p"] > 0.05 and wq["permutation_p"] > 0.05
    assert (rs["tau_b"] > 0) != (wq["tau_b"] > 0)
    assert "OPPOSITE IN SIGN" in cross["reading"]
    assert "the comfortable answer" in _flat(cross["what_it_rules_out"])

    # The preconditions that justified the statistic, measured.
    pre = _flat(phase21.ARM_B_OBSERVED["the_statistics_preconditions_measured"])
    assert "2.22e-16" in pre
    assert "18 distinct values" in pre and "largest tie group of 59" in pre
    assert "not Pearson" in pre


def test_the_post_hoc_analyses_are_tagged_and_fire_no_cell():
    record = phase21.POST_HOC_ANALYSES
    assert record["tag"] == "[POST-HOC]"
    assert "run AFTER the numbers existed" in _flat(
        record["not_registered_in_advance"]
    )
    assert "NO CELL FIRES ON THEM" in _flat(record["not_registered_in_advance"])

    # The bimodality partitions the cohort exactly.
    assert 117 + 36 + 84 == 237
    for piece in ("117 of 237", "36", "84", "62.8 of 64"):
        assert piece in record["1_hardness_is_bimodal"], piece

    spreads = _flat(record["2_the_hard_and_easy_sets_differ_in_SPREAD_not_LEVEL"])
    assert "2.7500" in spreads and "2.8017" in spreads
    assert "1.2086" in spreads and "0.3077" in spreads
    assert "17.6x" in spreads and "9.7x" in spreads
    assert round(1.2086 / 0.3077, 2) == 3.93
    assert "3.93x" in spreads

    assert "+0.7521" in record["3_worst_quartile_hardness_vs_distance_from_the_centre"]
    assert "-0.8470" in record["4_residual_sign_hardness_vs_signed_distance"]

    # The magnitude ordering is stated, not glossed.
    largest = _flat(record["which_is_actually_the_largest"])
    assert "-0.8470 is the largest BY MAGNITUDE" in largest
    assert "largest POSITIVE" in largest
    assert abs(-0.8470) > abs(0.7521) > 0.7024

    # No patient ids anywhere in the module.
    assert "CLUSTER-ONLY" in record["no_patient_ids"]


def test_no_patient_ids_entered_the_record():
    """Patient-keyed values are cluster-only."""
    text = (REPO / "src" / "cleft" / "phase21.py").read_text(encoding="utf-8")
    for forbidden in ("patient_id:", "patient_ids =", '"patient_id"'):
        assert forbidden not in text, forbidden
    # No bare list of three-digit ids.
    assert not re.search(r"patients?\s*[:=]\s*\(\s*\d{1,3}\s*,\s*\d{1,3}", text)


def test_the_shrinkage_mechanism_connects_two_prior_records():
    record = phase21.SHRINKAGE_MECHANISM
    finding = _flat(record["the_finding"])
    assert "ONE MECHANISM" in finding
    assert "SHRINK TOWARD THE LABEL MEAN" in finding
    assert "not because they share a representational limitation" in finding
    assert "-0.8470" in finding and "+0.7521" in finding
    assert "SHRINKAGE ARTIFACT" in finding

    twice = _flat(record["measured_twice_before_never_connected"])
    assert "0.477 on SCRAMBLED labels" in twice
    assert "since 2026-07-28" in twice

    # Phase 3's interpretation is quoted verbatim, and is live at source.
    import inspect

    from cleft.train import phase3 as phase3_module

    # Adjacent string literals are joined before matching: the quote
    # spans a concatenation seam, and a plain whitespace normalisation
    # would leave the quote characters in the middle of it.
    raw = (REPO / "src" / "cleft" / "train" / "phase3.py").read_text(
        encoding="utf-8"
    )
    joined = re.sub(r'"\s*\n\s*"', "", raw)
    source = " ".join(joined.split())
    assert (
        "samples shrinks toward the label mean (see shrinkage) and a shrunk "
        "predictor keeps its correlation while its RMSE approaches a "
        "constant's" in source
    )
    assert "shrinks toward the label mean" in twice
    # The function is sanity_report, and it really does compute the
    # quantity the record names.
    assert hasattr(phase3_module, "sanity_report")
    assert "phase3.sanity_report" in twice
    body = inspect.getsource(phase3_module.sanity_report)
    assert '"shrinkage": round(prediction_sd / truth_sd, 4)' in body

    assert "printed beside every run since Phase 3" in _flat(
        record["what_that_means_about_the_project"]
    )


def test_the_scale_invariance_prohibition_is_a_tested_literal():
    from cleft import ladder as ladder_mod
    from cleft import phase10_annex, phase20
    from cleft.eval import metrics

    prohibition = phase21.SCALE_INVARIANCE_PROHIBITION
    assert isinstance(prohibition, str)
    for sibling in (ladder_mod.DETECTION_FLOOR_PROHIBITION,
                    phase10_annex.ANNEX_PROHIBITION,
                    phase20.RESIDUAL_PROHIBITION,
                    phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION):
        assert isinstance(sibling, str)

    assert "CANNOT EXPLAIN THE PCC CEILING" in prohibition
    assert "fix the shrinkage and the correlation rises" in prohibition
    assert "PEARSON IS SCALE-INVARIANT" in prohibition
    assert "EXACTLY UNCHANGED" in prohibition
    assert "narrowing and not an answer" in prohibition
    assert "No phase is gated on any of this." in prohibition

    # The claim, re-measured here rather than trusted.
    rng = np.random.default_rng(20260901)
    truth = rng.uniform(1, 5, 237)
    shrunk = 2.7544 + 0.4 * (truth - 2.7544) + rng.normal(0, 0.5, 237)
    base = metrics.pcc(truth, shrunk)
    centre = float(shrunk.mean())
    for factor in (1.5, 2.0, 2.5):
        rescaled = centre + factor * (shrunk - centre)
        assert metrics.pcc(truth, rescaled) == pytest.approx(base, abs=1e-15)
        rmse_base = float(np.sqrt(np.mean((truth - shrunk) ** 2)))
        rmse_new = float(np.sqrt(np.mean((truth - rescaled) ** 2)))
        assert rmse_new != pytest.approx(rmse_base, abs=1e-6), "RMSE moves"


def test_the_reading_error_is_named_beside_the_other_six():
    from cleft import ladder as ladder_mod
    from cleft import literature, phase12, phase18, phase20

    record = phase21.SHRINKAGE_READING_ERROR_PROVENANCE
    error = _flat(record["the_error"])
    assert "unidentified image property" in error
    assert "BEFORE running the extremity cross-reference" in error
    assert "ESTIMATOR property" in _flat(record["why_it_was_wrong"])
    assert "one command" in _flat(record["how_close_the_right_answer_was"])
    assert "the order was wrong" in _flat(record["how_close_the_right_answer_was"])
    assert "seventh instance" in _flat(record["the_pattern"])

    for module, name in (
        (ladder_mod, "THE_ERROR_PROVENANCE"),
        (phase20, "CROSS_TARGET_ERROR_PROVENANCE"),
        (phase20, "S_DESCRIPTION_ERROR_PROVENANCE"),
        (phase12, "TWO_VIEW_CLAIM_PROVENANCE"),
        (literature, "NADEAU_BENGIO_DOES_NOT_APPLY"),
        (phase18, "TAU_RECOMPUTATION_PROVENANCE"),
    ):
        assert hasattr(module, name), name
        assert name in record["filed_beside"], name


def test_arm_a_is_explained_by_arm_b_with_the_speculation_tagged():
    record = phase21.ARM_A_EXPLAINED_BY_ARM_B
    mechanism = _flat(record["the_mechanism"])
    assert "the mean of shrunk predictors is a shrunk predictor" in mechanism
    assert "+0.0234 is what remains" in mechanism
    assert "err INDEPENDENTLY" in _flat(
        record["why_unresolved_rather_than_merely_small"]
    )
    # The seed-spread explanation is offered as a candidate, not a finding.
    spread = _flat(record["and_it_explains_the_seed_spread_too"])
    assert spread.startswith("[REASONED, not measured]")
    assert "NOT measured and is offered as a candidate, not a finding" in spread


def test_the_forward_lead_is_noted_not_committed():
    record = phase21.RANKING_LOSSES_NOTED
    assert record["status"].startswith("NOTED, NOT COMMITTED, NOT REGISTERED")
    assert "no mean to shrink toward" in _flat(record["the_lead"])
    assert "NEW PHASE with its own registration" in _flat(
        record["why_not_committed"]
    )
    assert "SCALE_INVARIANCE_PROHIBITION applies" in _flat(
        record["it_is_not_a_fix_for_the_ceiling"]
    )
    assert "rankiqa" in record["the_nearest_literature"]

    # Noted means noted: no task, no kind, no config.
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    for name in list(TASK_SPECS) + list(TASKS):
        assert "ranking" not in name.lower(), name
        assert "pairwise" not in name.lower(), name


def test_the_closing_walks_all_twelve_criteria():
    closing = phase21.PHASE_21_CLOSING
    criteria = sorted(k for k in closing if k.startswith("criterion_"))
    assert len(criteria) == 12, criteria
    assert len(phase21.EXIT_CRITERIA["criteria"]) == 12
    for key in criteria:
        assert closing[key].startswith("MET"), key
    assert "feb53133" in closing["closed"]
    assert "single-attempt" in closing["closed"]

    headline = _flat(closing["the_headline"])
    assert "SHRINKAGE, NOT REPRESENTATION" in headline
    assert "NAMING IT DOES NOT LIFT IT" in headline

    open_q = _flat(closing["the_open_question_restated_honestly"])
    assert "REMAINS UNEXPLAINED" in open_q
    assert "ONE CANDIDATE MECHANISM ELIMINATED" in open_q
    assert "narrowed the question; it did not answer it" in open_q

    eliminated = _flat(closing["what_was_eliminated"])
    assert "label noise" in eliminated
    assert "shared representational limitation" in eliminated

    # No ledger row, and the ledger really did not move.
    from cleft import results_ledger

    assert "ledger stands at 38" in closing["criterion_10_no_ledger_row"]
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 21 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p21" or "p21-" in e["id"]
    ]
    assert results_ledger.validate() is None
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p21" or "ensemble" in e["id"]
    ]


# --------------------------------------------------------------------------
# [2026-09-01] The seventh amendment, and 19 left open
# --------------------------------------------------------------------------


def test_the_seventh_amendment_carries_the_full_backward_chain():
    from cleft import phase11, phase12, phase15, phase20

    seventh = phase21.PHASE_SEQUENCE_EXTENDED_7
    for key, target in (
        ("first_amendment", "phase11.PHASE_SEQUENCE_RENUMBERED"),
        ("second_amendment", "phase12.PHASE_SEQUENCE_RENUMBERED_2"),
        ("third_amendment", "phase15.PHASE_SEQUENCE_RENUMBERED_3"),
        ("fourth_amendment", "phase15.PHASE_SEQUENCE_RENUMBERED_4"),
        ("fifth_amendment", "phase20.PHASE_SEQUENCE_EXTENDED_5"),
        ("sixth_amendment", "phase21.PHASE_SEQUENCE_EXTENDED_6"),
    ):
        assert seventh[key].startswith(target), key

    # All six earlier amendments carry the dated pointer back.
    for amendment in (
        phase11.PHASE_SEQUENCE_RENUMBERED,
        phase12.PHASE_SEQUENCE_RENUMBERED_2,
        phase15.PHASE_SEQUENCE_RENUMBERED_3,
        phase15.PHASE_SEQUENCE_RENUMBERED_4,
        phase20.PHASE_SEQUENCE_EXTENDED_5,
        phase21.PHASE_SEQUENCE_EXTENDED_6,
    ):
        assert amendment["seventh_amendment"] == (
            "phase21.PHASE_SEQUENCE_EXTENDED_7"
        )

    # It schedules exactly four, in order, and renumbers nothing.
    assert sorted(seventh["becomes"]) == ["22", "23", "24", "25"]
    assert "NOTHING" in seventh["was"]["22"]
    assert "nothing moved -- 19, 20 and 21 are untouched" in _flat(
        seventh["nothing_silently_renumbered"]
    )


def test_the_four_phases_are_scheduled_not_registered():
    seventh = phase21.PHASE_SEQUENCE_EXTENDED_7
    keys = [k for k in seventh if k.startswith("phase_2")]
    assert len(keys) == 4, keys
    for key in keys:
        block = seventh[key]
        assert block["status"] == "SCHEDULED, NOT REGISTERED", key
        assert "what_is_not_decided_here" in block, key
        assert "reading" in block["what_is_not_decided_here"].lower(), key

    # One step earlier than either existing precedent, both of which exist.
    from cleft import phase15

    step = _flat(seventh["scheduled_not_registered"])
    assert "SCHEDULED-NOT-SCOPED" in step
    assert "REGISTERED-NOT-BUILT" in step
    assert "a place in the sequence and a motivation, and nothing else" in step
    assert "scope is proposed at the phase's RESTATE" in _flat(
        phase15.PHASE_16_SCHEDULED["scope_is_not_set_here"]
    )
    assert "build nothing tonight" in phase15.ANCHOR_LOOP_REGISTERED[
        "registered"
    ]


def test_each_scheduled_phase_names_its_motivation_from_the_record():
    seventh = phase21.PHASE_SEQUENCE_EXTENDED_7

    # 22 -- the shrinkage measurements, live at their sources.
    p22 = seventh["phase_22_ranking_and_pairwise_losses"]
    motive = _flat(p22["motivated_by"])
    assert "0.477 on SCRAMBLED labels" in motive
    assert "-0.847" in motive
    assert "no mean to shrink toward" in motive
    assert "0.477 on SCRAMBLED labels" in _flat(
        phase21.SHRINKAGE_MECHANISM["measured_twice_before_never_connected"]
    )
    assert "-0.8470" in phase21.POST_HOC_ANALYSES[
        "4_residual_sign_hardness_vs_signed_distance"
    ]
    # Both pair sources, and the caveat that travels with them.
    assert "BOTH" in p22["pair_source_ruled"]
    assert "synthetic TPS" in p22["pair_source_ruled"]
    caveat = _flat(p22["the_caveat_that_travels"])
    assert "CERTAIN and ours is the UNCERTAIN THING" in caveat
    from cleft import literature

    assert "known BY CONSTRUCTION" in _flat(
        literature.SOURCES_BANKED["rankiqa"]["how_it_bears"]
    )

    # 23 -- the biased-estimator limitation, live at its source.
    p23 = seventh["phase_23_the_statistical_instruments"]
    assert "BIASED ESTIMATOR" in _flat(p23["motivated_by"])
    assert "17 unresolved ledger rows" in _flat(p23["motivated_by"])
    assert "neither changes a figure" in _flat(p23["neither_raises_pcc"])
    assert "severe underestimations of standard error" in literature.\
        BOUTHILLIER_SAMPLE_SIZE[
            "five_seeds_on_a_fixed_split_is_their_biased_estimator"
        ]

    # 24 -- sharpened by Phase 21's refutation, with the figures.
    p24 = seventh["phase_24_all_five_raters"]
    sharp = _flat(p24["sharpened_by_phase_21"])
    assert "does NOT track disagreement" in sharp
    assert "+0.0727" in sharp and "-0.0811" in sharp
    assert "extremes are not the contested cases" in sharp
    cross = phase21.ARM_B_OBSERVED["cross_reference"]
    assert cross["residual_sign"]["tau_b"] == 0.0727
    assert cross["worst_quartile"]["tau_b"] == -0.0811

    # 25 -- last, on the project's own weak prior.
    p25 = seventh["phase_25_foundation_model_features"]
    last = _flat(p25["why_last_of_the_compute_phases"])
    assert "0.2537 against ImageNet 0.2520" in last
    assert "weakest prior" in last


def test_the_three_binding_clauses():
    seventh = phase21.PHASE_SEQUENCE_EXTENDED_7
    one = _flat(seventh["binding_1_scope_at_the_restate"])
    assert "NEVER HERE" in one
    assert "Motivation is not scope" in one
    two = _flat(seventh["binding_2_cancellation_is_recorded"])
    assert "CANCELLED WITH A REASON, never silently dropped" in two
    assert "indistinguishable from one that never had them" in two
    three = _flat(seventh["binding_3_reorder_on_evidence"])
    assert "REORDERED OR CANCELLED on evidence" in three
    assert "Phase 22's result may change what 24 should measure" in three
    assert "why no readings are written now" in three

    # And no readings were written: no scheduled block carries one.
    for key in (k for k in seventh if k.startswith("phase_2")):
        block = seventh[key]
        assert not [k for k in block if "reading" in k], key
        assert not [k for k in block if "criteri" in k], key


# --------------------------------------------------------------------------
# 19: reported, not decided
# --------------------------------------------------------------------------


def test_the_chain_records_19_as_the_write_up_three_times():
    from cleft import phase15, phase20

    r4 = phase15.PHASE_SEQUENCE_RENUMBERED_4
    assert r4["becomes"]["19"] == "write-up"
    assert r4["status_changes"]["write_up"] == "Phase 18 -> Phase 19"

    e5 = phase20.PHASE_SEQUENCE_EXTENDED_5
    assert e5["becomes"]["19"] == "write-up (UNCHANGED in place and content)"
    assert e5["status_changes"]["write_up"] == "Phase 19 -> Phase 19 (unchanged)"

    e6 = phase21.PHASE_SEQUENCE_EXTENDED_6
    assert e6["becomes"]["19"] == "write-up (UNCHANGED in place and content)"

    # The rule's own words, verbatim.
    rule = _flat(e5["the_write_up_is_last_by_rule_not_by_number"])
    assert "THE WRITE-UP RUNS LAST REGARDLESS OF ITS NUMBER" in rule
    assert "a reader must not infer execution order from 19 vs 20" in rule
    assert "RENUMBER WAS CONSIDERED AND REJECTED" in rule
    assert "It has already moved twice (18 -> 19 across two amendments)" in rule


def test_19_is_reported_open_and_not_decided():
    record = phase21.WRITE_UP_NUMBER_OPEN
    # [2026-09-01, THIS PIN FIRED AS DESIGNED] It held the question OPEN
    # until the ruling was. He ruled (a), the status carries the marker,
    # and the original wording is preserved behind it -- so the pin now
    # holds that the record was reported-not-decided AND that the ruling
    # is recorded, rather than that it is still open.
    assert record["status"].startswith("[RULED 2026-09-01 -- candidate (a)")
    assert "REPORTED, NOT DECIDED" in record["status"]
    assert phase21.WRITE_UP_NUMBER_RULED["the_ruling"]
    assert "19 REMAINS" in record["candidate_a"]
    assert "19 is RETIRED" in record["candidate_b"]
    assert "19 is void and never used" in record["candidate_b"]

    supports = _flat(record["what_the_chains_wording_supports"])
    assert supports.startswith("**(a).")
    assert "THREE explicit assignments" in supports
    assert "PRESUPPOSES the write-up has one" in supports
    assert "that same move, deferred" in supports
    assert "PRESUPPOSES 19 is live" in supports

    # It does not decide, and says why.
    assert "(b) may be ruled; it is the project sequence" in _flat(
        record["this_is_not_a_recommendation_against_b"]
    )
    assert "REVERSAL of the fourth and fifth amendments" in _flat(
        record["this_is_not_a_recommendation_against_b"]
    )
    assert "not voided on a paraphrase" in _flat(
        record["why_it_is_reported_and_not_applied"]
    )

    # And the amendment leaves 19 alone.
    seventh = phase21.PHASE_SEQUENCE_EXTENDED_7
    assert "19" not in seventh["becomes"]
    assert seventh["status_changes"]["write_up"].startswith("UNRESOLVED")


def test_nothing_was_built_for_22_to_25():
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    # [UPDATED 2026-09-01] Phase 22 IS built -- arms, tie measurement
    # and the two post-run steps. The pin now holds what remains true:
    # NOTHING exists for 23 to 25, and Phase 22's own task set is
    # exactly the four that were ruled.
    # [UPDATED 2026-09-02] Phase 23's ROPE half is BUILT, as ruled.
    # The pin holds what remains: nothing for 24 or 25, and Phase 23's
    # machinery is exactly the one ruled task.
    # [UPDATED A SEVENTH TIME 2026-09-02] Phase 25's three contrasts need
    # a reader, as Phase 22's did, so ONE p25 task now exists. Phase 24
    # still has none and needs none -- its deliverable is a laptop
    # computation with no run directory. The token sweep therefore holds
    # p24 at zero and p25 at exactly the contrast reader.
    for name in list(TASK_SPECS) + list(TASKS):
        for token in ("p24", "dinov2"):
            assert token not in name.lower(), (name, token)
    assert sorted(
        n for n in set(list(TASK_SPECS) + list(TASKS)) if "p25" in n
    ) == ["p25_contrasts"]
    assert sorted(
        n for n in set(list(TASK_SPECS) + list(TASKS))
        if "p23" in n or "rope" in n
    ) == ["p23_rope"]
    assert sorted(
        n for n in set(list(TASK_SPECS) + list(TASKS)) if "p22" in n
    ) == ["p22_contrasts", "p22_diagnostic"]
    # [UPDATED 2026-09-01, THIS PIN FIRED AS DESIGNED] Phase 22's
    # RESTATE was written -- registration only, no machinery. The
    # module now exists; what must not exist is a config or a task.
    # 23 to 25 remain untouched in every respect.
    # [UPDATED AGAIN 2026-09-01] Phase 22's TIE MEASUREMENT was built --
    # one task, one config -- because it GATES the target-map ruling.
    # It is a precondition, not an arm. 23 to 25 remain untouched.
    # [UPDATED A THIRD TIME 2026-09-02] Phase 23's ROPE half is now
    # BUILT, as ruled -- one task, one config. What the pin holds is
    # that it is EXACTLY that one config. 24 and 25 remain untouched.
    assert (REPO / "src" / "cleft" / "phase22.py").exists()
    assert (REPO / "src" / "cleft" / "phase23.py").exists()
    assert sorted(
        p.name for p in (REPO / "configs").glob("p23*.yaml")
    ) == ["p23_rope.yaml"]
    # [UPDATED A FOURTH TIME 2026-09-02, THIS PIN FIRED AS DESIGNED]
    # Phase 24's RESTATE was written -- registration only, and it
    # CONCEDES three of the amendment's four designs rather than
    # building them. The module now exists; what must not exist is a
    # config, a task or an arm. 25 remains untouched in every respect.
    assert (REPO / "src" / "cleft" / "phase24.py").exists()
    assert list((REPO / "configs").glob("p24*.yaml")) == []
    # [UPDATED A FIFTH TIME 2026-09-02, THIS PIN FIRED AS DESIGNED]
    # Phase 25's RESTATE was written -- registration only, one arm
    # registered and not built, and the ruling is RUN IT. What
    # must not exist is a config, a task, or a DINOv2 backbone entry.
    # The chain 22-25 now has a module for every entry and a config for
    # none of 24 or 25, which is what the pin holds from here.
    # [UPDATED A SIXTH TIME 2026-09-02, THIS PIN FIRED AS DESIGNED]
    # Phase 25 is BUILT: the ruling was two arms, and the phase shipped
    # two extraction configs and two arm configs. The chain 22-25 is now
    # fully built, so what this pin holds from here is that **no p25 TASK
    # exists** -- the arms reuse extract_embeddings and train_cv, which
    # is the whole claim that they are one-factor contrasts against the
    # probe rather than a new machine.
    assert (REPO / "src" / "cleft" / "phase25.py").exists()
    assert sorted(
        path.name for path in (REPO / "configs").glob("p25*.yaml")
    ) == [
        # [UPDATED 2026-09-02] The contrasts config joined when the two
        # arm run-directory names arrived.
        "p25_arm_d1.yaml", "p25_arm_d2.yaml", "p25_contrasts.yaml",
        "p25_extract_d1_dino.yaml", "p25_extract_d2_dinov2.yaml",
    ]
    from cleft.config.schema import TASK_SPECS as _SPECS
    from cleft.models.factory import BACKBONES

    # [UPDATED 2026-09-02] The ARMS still need no task -- they reuse
    # extract_embeddings and train_cv, which is the one-factor claim.
    # The three CONTRASTS need a reader, as Phase 22's did, and it is
    # exactly one.
    assert [k for k in _SPECS if k.startswith("p25")] == ["p25_contrasts"]
    assert len(BACKBONES) == 9
    # [UPDATED AGAIN 2026-09-01] The six ranking arms were built once
    # the tie rule closed. What stays true here is that NONE of them is
    # a Phase 21 measurement: the p21 reckoning inputs are unchanged.
    # [UPDATED 2026-09-01] Post-run configs joined; the tie measurement
    # is still there and none of these is a Phase 21 measurement.
    p22_configs = {p.name for p in (REPO / "configs").glob("p22_*.yaml")}
    assert "p22_tie_fraction.yaml" in p22_configs
    assert {"p22_diagnostic.yaml", "p22_contrasts.yaml",
            "p22_shrinkage.yaml"} <= p22_configs

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 21 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p21" or "p21-" in e["id"]
    ]
    assert results_ledger.validate() is None


# --------------------------------------------------------------------------
# [2026-09-01] Three items before Phase 22 is scoped
# --------------------------------------------------------------------------


def test_19_is_ruled_a_with_the_four_supporting_points():
    from cleft import phase15, phase20

    record = phase21.WRITE_UP_NUMBER_RULED
    assert "candidate (a)" in record["ruled"]
    assert "19 REMAINS" in record["the_ruling"]

    # Each supporting point checked against the chain it cites.
    one = _flat(record["supporting_point_1_three_assignments"])
    assert "PHASE_SEQUENCE_RENUMBERED_4" in one
    assert phase15.PHASE_SEQUENCE_RENUMBERED_4["becomes"]["19"] == "write-up"
    assert phase20.PHASE_SEQUENCE_EXTENDED_5["becomes"]["19"] == (
        "write-up (UNCHANGED in place and content)"
    )
    assert phase21.PHASE_SEQUENCE_EXTENDED_6["becomes"]["19"] == (
        "write-up (UNCHANGED in place and content)"
    )

    two = _flat(record["supporting_point_2_regardless_of_its_number"])
    assert "presupposes the write-up HAS one" in two
    rule = _flat(
        phase20.PHASE_SEQUENCE_EXTENDED_5[
            "the_write_up_is_last_by_rule_not_by_number"
        ]
    )
    assert "THE WRITE-UP RUNS LAST REGARDLESS OF ITS NUMBER" in rule

    three = _flat(record["supporting_point_3_relocation_already_rejected"])
    assert "would move on every future addition, unboundedly" in three
    assert "would move on every future addition, unboundedly" in rule

    four = _flat(record["supporting_point_4_the_stated_cost"])
    assert "no such cost if 19 is void" in four
    assert "a reader must not infer execution order from 19 vs 20" in rule

    # (b) named as a reversal, and not taken.
    b = _flat(record["b_was_not_taken"])
    assert "WOULD HAVE BEEN A REVERSAL" in b
    assert "not a clarification" in b
    assert "It was not taken" in b

    # The open record stands, marked ruled.
    assert phase21.WRITE_UP_NUMBER_OPEN["status"].startswith(
        "[RULED 2026-09-01 -- candidate (a)"
    )
    assert "REPORTED, NOT DECIDED" in phase21.WRITE_UP_NUMBER_OPEN["status"]
    assert "RULED (a) 2026-09-01" in phase21.PHASE_SEQUENCE_EXTENDED_7[
        "status_changes"
    ]["write_up"]


def test_the_0_477_attribution_is_corrected_with_the_original_visible():
    record = phase21.SHRINKAGE_FIGURE_MISATTRIBUTED
    assert "**Phase 20** measured head shrinkage at 0.477" in record[
        "what_the_record_said"
    ]
    truth = _flat(record["what_is_true"])
    assert "phase20.py contains no occurrence of 0.477" in truth

    # Re-verified here: phase20 really is silent on it.
    phase20_text = (REPO / "src" / "cleft" / "phase20.py").read_text(
        encoding="utf-8"
    )
    for token in ("0.477", "shrinkage", "sanity_report", "prediction_sd"):
        assert token not in phase20_text, token

    assert "p20_permutation_plain__dc4605bf" in record["the_correct_attribution"]
    assert "seed 1337" in record["the_correct_attribution"]
    assert record["verification_status"].startswith(
        "**[SUPPLIED, NOT VERIFIED HERE]**"
    )
    assert "CLUSTER-ONLY and unreachable" in record["verification_status"]
    assert "the record's" in record["whose_error"]
    assert "attributed to a PHASE when it came from a RUN" in _flat(
        record["the_shape_it_repeats"]
    )

    # The originals are marked in place, not deleted.
    assert phase21.SHRINKAGE_MECHANISM[
        "measured_twice_before_never_connected"
    ].startswith("[MIS-ATTRIBUTED -- corrected 2026-09-01")
    assert "0.477 on SCRAMBLED labels" in phase21.SHRINKAGE_MECHANISM[
        "measured_twice_before_never_connected"
    ]
    assert "run output, not a phase20 record figure" in (
        phase21.PHASE_SEQUENCE_EXTENDED_7[
            "phase_22_ranking_and_pairwise_losses"
        ]["motivated_by"]
    )

    # Filed beside all seven.
    from cleft import ladder, literature, phase12, phase18, phase20

    for module, name in (
        (ladder, "THE_ERROR_PROVENANCE"),
        (phase20, "CROSS_TARGET_ERROR_PROVENANCE"),
        (phase20, "S_DESCRIPTION_ERROR_PROVENANCE"),
        (phase12, "TWO_VIEW_CLAIM_PROVENANCE"),
        (literature, "NADEAU_BENGIO_DOES_NOT_APPLY"),
        (phase18, "TAU_RECOMPUTATION_PROVENANCE"),
    ):
        assert hasattr(module, name), name
        assert name in record["filed_beside"], name
    assert "SHRINKAGE_READING_ERROR_PROVENANCE" in record["filed_beside"]


def test_the_general_shrinkage_claim_is_separated_from_the_one_figure():
    record = phase21.SHRINKAGE_FIGURE_MISATTRIBUTED
    where = _flat(record["where_else_it_is_load_bearing"])
    assert "nowhere outside this module" in where
    assert "no verdict turns on it" in where

    # Re-derived: 0.477 appears only in phase21 and its test file.
    hits = sorted(
        path.name for path in (REPO / "src").rglob("*.py")
        if "0.477" in path.read_text(encoding="utf-8")
    )
    assert hits == ["phase21.py"], hits

    general = _flat(record["does_any_record_treat_shrinkage_as_general"])
    assert "yes, and that is the more serious half" in general
    assert "ONE SEED OF ONE ARM ON SCRAMBLED LABELS" in general
    assert "an illustration presented as a measurement" in general
    # The genuinely cross-arm support is named.
    assert "-0.8470" in general

    # And the cheap route to a real cross-arm figure.
    cheap = _flat(record["a_proper_cross_arm_figure_is_cheap_and_reachable"])
    assert "recomputable from the cached per-seed CSVs" in cheap
    assert "NO new artifact and NO new hash" in cheap
    from cleft.cluster_csv import PREDICTIONS_COLUMNS

    assert PREDICTIONS_COLUMNS == ("patient_id", "truth", "prediction", "fold")
    assert "NOT BUILT, NOT REGISTERED" in record["what_it_would_take"]


def test_the_pair_separation_derivation_reproduces_from_banked_figures():
    import math

    from cleft.data import reliability

    record = phase21.COHORT_PAIR_SEPARATION_DESIGNED
    assert record["designed"].endswith("DESIGNED, NOT RUN")

    # The banked inputs, live.
    assert reliability.MEAN_R_237 == 0.4696
    assert reliability.RELIABILITY_237 == 0.8158
    assert reliability.spearman_brown(0.4696, 5) == pytest.approx(0.8157, abs=5e-5)

    # The derivation, recomputed.
    err = 1 - reliability.RELIABILITY_237
    assert err == pytest.approx(0.1842, abs=5e-5)
    k_single = math.sqrt(err)
    k_diff = math.sqrt(2) * k_single
    assert k_single == pytest.approx(0.4292, abs=5e-5)
    assert k_diff == pytest.approx(0.6070, abs=5e-5)
    derivation = _flat(record["the_derivation"])
    assert "0.4292" in derivation and "0.6070" in derivation
    assert "sqrt(0.1842)" in derivation

    # C(237,2).
    assert 237 * 236 // 2 == 27966
    assert "27,966" in record["the_quantity"]

    # It is a derivation, and sd_obs is computed not quoted.
    assert "No record states either SE" in record[
        "it_is_a_derivation_not_a_record"
    ]
    sd = _flat(record["sd_obs_is_computed_not_quoted"])
    assert "AT RUN TIME" in sd
    assert "**not** the docstring figure 0.628" in sd
    assert "ITEM-TOTAL CORRELATION" in sd, "the adjacent 0.628 trap is named"

    # The sqrt(2) is stated, with both scales reported.
    assert "1.41x larger" in _flat(record["the_sqrt_2_matters"])
    assert "reported against BOTH" in _flat(record["the_sqrt_2_matters"])
    assert "1x, 2x and 3x" in record["the_profile_not_a_threshold"]
    assert "six numbers" in record["the_profile_not_a_threshold"]


def test_the_pair_separation_readings_are_committed_before_it_runs():
    record = phase21.COHORT_PAIR_SEPARATION_DESIGNED

    clear = _flat(record["reading_most_pairs_clear_the_noise"])
    assert "VIABLE" in clear
    assert "training on order that exists" in clear

    fail = _flat(record["reading_most_pairs_do_not_clear_it"])
    assert "COIN FLIPS" in fail
    assert "SCOPING FACT" in fail
    assert "not discovered in its result" in fail
    assert "indistinguishable, after the fact" in fail
    assert "not a post-hoc filter chosen to make the arm work" in fail

    assert "phase17.UNPREDICTED_PATTERN" in record["no_reading_is_invented_after"]

    # It bears on the cohort arm only.
    only = _flat(record["it_bears_on_the_cohort_arm_only"])
    assert "EXACT BY CONSTRUCTION" in only
    assert "says nothing about the synthetic half" in only

    # The prior is tagged as an expectation, not a result.
    prior = _flat(record["the_prior_registered_before_the_number"])
    assert prior.startswith("[REASONED, an expectation and not a result]")
    assert "0.24 / 0.46 / 0.64" in prior and "0.33 / 0.61 / 0.80" in prior
    assert "the discreteness is doing something" in prior

    assert "CLUSTER-SIDE" in record["not_run_here"]
    assert "no machinery built" in record["not_run_here"]


def test_the_pairing_axis_assumption_is_named_not_scoped():
    record = phase21.PAIRING_AXIS_ASSUMPTION
    assert record["registered"].endswith("NAMED, NOT SCOPED")
    assumption = _flat(record["the_assumption"])
    assert "THE SAME FACE AGAINST ITSELF" in assumption
    assert "DIFFERENT PATIENTS" in assumption
    assert "BRIDGES them" in assumption

    provides = _flat(record["what_the_artifact_actually_provides"])
    assert "0.0, 0.015, 0.025, 0.035" in provides
    assert "run._synth_index" in provides
    assert "says nothing about face X against face Y" in provides
    # The reader it names is real.
    from cleft import run as run_module

    assert hasattr(run_module, "_synth_index")

    # Phase 17's pairs really are intra-face.
    import inspect

    body = inspect.getsource(run_module.task_siamese_contrastive)
    assert "symmetric iff the face is undeformed" in body
    assert "1.0 if magnitude == 0 else 0.0" in body
    assert "left versus right view of a SINGLE face" in _flat(
        record["phase_17s_pairs_are_intra_face_too"]
    )

    # Same family as magnitude-to-grade, whose wording is live at source.
    from cleft.scut import synthesis

    assert "ORDERED series, not a graded one" in synthesis.LIMITATIONS[
        "magnitude_to_grade_is_an_assumption"
    ]
    family = _flat(record["it_is_the_same_family_as_magnitude_to_grade"])
    assert "WITHIN faces transfers to an ordering BETWEEN patients" in family
    assert "not yet written down anywhere" in family
    assert "must ADDRESS it rather than inherit it silently" in _flat(
        record["named_not_scoped"]
    )


def test_no_phase_22_machinery_was_built():
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    # [2026-09-01, THIS PIN FIRED AS DESIGNED] It forbade "separation"
    # in any task name while the cohort-pair measurement was only a
    # design. That measurement is now built as a RECKONING INPUT, and
    # the pin is updated to hold what it was standing for: no PHASE 22
    # machinery, and no ranking or pairwise-loss task.
    # [UPDATED A THIRD TIME 2026-09-01] It forbade "rank" in any task
    # name while Phase 22 was unbuilt. The six arms are now built as
    # ruled, so the pin holds what it was always standing for: the
    # Phase 22 machinery that exists is EXACTLY the ruled set, and
    # nothing arrived unannounced.
    # [UPDATED 2026-09-01] The two POST-RUN steps joined once the four
    # cohort arms produced run directories. The pin still holds what it
    # stands for: the Phase 22 machinery is EXACTLY the ruled set.
    phase_22_tasks = sorted(
        name for name in TASKS
        if "rank" in name or "tie" in name or "p22" in name
    )
    assert phase_22_tasks == [
        "p22_contrasts", "p22_diagnostic", "tie_fraction", "train_rank_cv",
    ], phase_22_tasks
    assert {"cohort_pair_separation", "cross_arm_shrinkage"} <= set(TASKS)
    # [UPDATED 2026-09-01, FIRED A SECOND TIME AS DESIGNED] The restate
    # was written. The module exists and holds REGISTRATION ONLY -- the
    # pin now checks that, which is what "no machinery" always meant.
    assert (REPO / "src" / "cleft" / "phase22.py").exists()
    from cleft import phase22

    # [UPDATED 2026-09-01] The module gained the family enumeration when
    # the arms were built. The fit path still lives in run.py and
    # train/ranking.py, as every other arm's does.
    assert [
        n for n in dir(phase22)
        if callable(getattr(phase22, n)) and not n.startswith("_")
    ] == ["contrast_family", "summary"]
    assert len(phase22.contrast_family()) == 15
    # [UPDATED AGAIN 2026-09-01] The tie measurement is the ONE p22
    # config, and it is a PRECONDITION of the target-map ruling rather
    # than an arm. No arm config exists.
    # [UPDATED AGAIN 2026-09-01] The six ranking arms were built once
    # the tie rule closed. What stays true here is that NONE of them is
    # a Phase 21 measurement: the p21 reckoning inputs are unchanged.
    # [UPDATED 2026-09-01] Post-run configs joined; the tie measurement
    # is still there and none of these is a Phase 21 measurement.
    p22_configs = {p.name for p in (REPO / "configs").glob("p22_*.yaml")}
    assert "p22_tie_fraction.yaml" in p22_configs
    assert {"p22_diagnostic.yaml", "p22_contrasts.yaml",
            "p22_shrinkage.yaml"} <= p22_configs
    # The separation config exists and is a RECKONING INPUT under the
    # p21 prefix -- what must not exist is a p22 one.
    assert sorted(
        p.name for p in (REPO / "configs").glob("*separation*.yaml")
    ) == ["p21_cohort_pair_separation.yaml"]

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 21 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p21" or "p21-" in e["id"]
    ]
    assert results_ledger.validate() is None


# --------------------------------------------------------------------------
# [2026-09-01] Two reckoning measurements, built from cached data
# --------------------------------------------------------------------------


def test_both_reckoning_measurements_are_built():
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    for kind in ("cohort_pair_separation", "cross_arm_shrinkage"):
        assert kind in TASK_SPECS, kind
        assert kind in TASKS, kind
    assert (REPO / "configs" / "p21_cohort_pair_separation.yaml").is_file()
    assert (REPO / "configs" / "p21_cross_arm_shrinkage.yaml").is_file()
    assert (REPO / "scripts" / "generate_reckoning_configs.py").is_file()

    # They are RECKONING INPUTS, and the record says so.
    assert "reckoning input" in phase21.COHORT_PAIR_SEPARATION_BUILT[
        "status"
    ].lower()
    assert "reckoning input" in phase21.CROSS_ARM_SHRINKAGE_REGISTERED[
        "status"
    ].lower()
    for record in (phase21.COHORT_PAIR_SEPARATION_BUILT,
                   phase21.CROSS_ARM_SHRINKAGE_REGISTERED):
        assert "no ledger row" in record["status"].lower()


def test_neither_reads_pixels_trains_or_globs():
    import inspect

    from cleft import run as run_module

    for task in (run_module.task_cohort_pair_separation,
                 run_module.task_cross_arm_shrinkage):
        body = inspect.getsource(task)
        for forbidden in ("import torch", "prepare_features", "np.load(",
                          ".glob(", "rglob", "RidgeBackbone", "TrainConfig",
                          "extract_embeddings"):
            assert forbidden not in body, f"{forbidden} in {task.__name__}"


def test_pair_separation_derives_its_scales_and_never_quotes_sd():
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_cohort_pair_separation)
    # sd_obs is COMPUTED from the manifest, never quoted.
    assert "np.std(means)" in body
    # 0.628 appears in the task's own docstring, saying it is NOT
    # used. What must hold is that no EXECUTABLE line uses it.
    code = body[body.index('task = ctx.config'):]
    assert "0.628" not in code, "the docstring figure must not be used"
    assert "0.628" in body, "and the task says so, in its docstring"
    assert "0.4292" not in body and "0.6070" not in body, (
        "the multipliers are DERIVED from RELIABILITY_237, not pasted"
    )
    # The derivation comes from the banked reliability constant.
    assert "RELIABILITY_237" in body
    assert "math.sqrt" in body or "** 0.5" in body

    # And it is printed beside the result.
    assert "derivation" in body


def test_the_pair_separation_arithmetic_is_what_the_design_registered():
    """Re-derived here from the banked figures, independently of the task."""
    import math

    from cleft.data import reliability

    err = 1 - reliability.RELIABILITY_237
    k_single = math.sqrt(err)
    k_diff = math.sqrt(2) * k_single
    assert k_single == pytest.approx(0.4292, abs=5e-5)
    assert k_diff == pytest.approx(0.6070, abs=5e-5)
    assert 237 * 236 // 2 == 27966

    design = phase21.COHORT_PAIR_SEPARATION_DESIGNED
    assert "0.4292" in design["the_derivation"]
    assert "0.6070" in design["the_derivation"]

    # The built record points back at the design rather than restating it.
    built = phase21.COHORT_PAIR_SEPARATION_BUILT
    assert "COHORT_PAIR_SEPARATION_DESIGNED" in built["the_design"]


def test_the_pair_separation_config_declares_the_cell_in_advance():
    import yaml

    task = yaml.safe_load(
        (REPO / "configs" / "p21_cohort_pair_separation.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert task["kind"] == "cohort_pair_separation"
    assert task["label"] == "mean"
    assert task["expect_patients"] == 237
    # The scale and multiple the reading turns on are DECLARED, so the
    # cell cannot be chosen after the six fractions are seen.
    assert task["primary_scale"] == "se_diff"
    assert task["primary_multiple"] == 1
    assert task["multiples"] == [1, 2, 3]
    # Only the manifest: no embeddings, no staged pixels, no run dirs.
    inputs = yaml.safe_load(
        (REPO / "configs" / "p21_cohort_pair_separation.yaml").read_text(
            encoding="utf-8"
        )
    )["inputs"]
    assert [e["name"] for e in inputs] == ["manifest_v1"]


def test_the_0_628_collision_is_guarded():
    """reliability.item_total's docstring carries 0.628 as an ITEM-TOTAL
    CORRELATION. The label sd is a different quantity with the same
    digits, and nothing may read one for the other."""
    from cleft.data import reliability

    doc = " ".join(reliability.item_total.__doc__.split())
    assert "0.628" in doc
    assert "orthodontist" in doc, "it is an item-total correlation"

    # The design record names the collision.
    sd = _flat(
        phase21.COHORT_PAIR_SEPARATION_DESIGNED["sd_obs_is_computed_not_quoted"]
    )
    assert "ITEM-TOTAL CORRELATION" in sd
    assert "coincidentally the same digits" in sd

    # And the built record carries the guard forward.
    assert "0.628" in _flat(
        phase21.COHORT_PAIR_SEPARATION_BUILT["the_collision_guarded"]
    )


def test_cross_arm_shrinkage_reuses_the_phase_21_reader():
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_cross_arm_shrinkage)
    assert "_p21_load_arm_predictions(" in body
    # No second CSV reader.
    assert "read_cluster_csv" not in body
    assert inspect.getsource(run_module).count(
        "def _p21_load_arm_predictions"
    ) == 1
    # The reader gained truth, and its existing callers still work.
    signature = inspect.signature(run_module._p21_load_arm_predictions)
    assert "with_truth" in signature.parameters
    assert signature.parameters["with_truth"].default is False


def test_the_denominator_is_verified_constant_not_assumed():
    """The truth vector is identical across arms on the 237 cohort, so
    sd(truth) is a constant and the ratio is driven entirely by
    prediction spread. The task VERIFIES that rather than assuming it."""
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_cross_arm_shrinkage)
    assert "truth_is_constant" in body
    assert "np.allclose" in body or "array_equal" in body

    record = phase21.CROSS_ARM_SHRINKAGE_REGISTERED
    stated = _flat(record["the_denominator_is_a_constant"])
    assert "on the 237 cohort every arm scores the same truth vector" in stated
    assert "driven ENTIRELY by prediction spread" in stated
    assert "VERIFIED, not assumed" in stated
    assert "refuses to continue otherwise" in stated


def test_the_shrinkage_arm_set_and_its_deviation_are_stated():
    """The instruction said 'the locked 68'; the built measurement covers
    the 64-arm ruled set. The deviation is stated, with its reason."""
    import yaml

    record = phase21.CROSS_ARM_SHRINKAGE_REGISTERED
    deviation = _flat(record["the_arm_set_and_why_it_is_64_not_68"])
    assert "64" in deviation and "68" in deviation
    assert "236" in deviation
    assert "cohort guard" in deviation
    assert "ARM_SET_RULED" in deviation

    task = yaml.safe_load(
        (REPO / "configs" / "p21_cross_arm_shrinkage.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert len(task["arms"]) == 64
    assert not [a for a in task["arms"] if a["n_patients"] != 237]
    assert not [a for a in task["arms"] if a["name"].startswith("p12_")]
    assert task["probe_arm"] == "p7_d1_vit_b16_imagenet_g1"
    assert task["probe_arm"] in {a["name"] for a in task["arms"]}


def test_the_shrinkage_readings_are_committed_with_declared_thresholds():
    import yaml

    record = phase21.CROSS_ARM_SHRINKAGE_REGISTERED
    readings = record["readings"]
    assert sorted(readings) == [
        "every_arm_shrinks_substantially",
        "shrinkage_varies_widely",
        "some_arm_does_not_shrink",
    ]
    assert "MEASURED rather than illustrated" in _flat(
        readings["every_arm_shrinks_substantially"]
    )
    assert "0.477 becomes one instance" in _flat(
        readings["every_arm_shrinks_substantially"]
    )
    assert "NEEDS QUALIFYING" in _flat(readings["shrinkage_varies_widely"])
    assert "WEAKER THAN STATED" in _flat(readings["shrinkage_varies_widely"])
    assert "EXCEPTION TO NAME" in _flat(readings["some_arm_does_not_shrink"])

    # The thresholds are DECLARED in the config, not chosen after.
    task = yaml.safe_load(
        (REPO / "configs" / "p21_cross_arm_shrinkage.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert 0.0 < task["shrinks_substantially_at"] < 1.0
    assert 0.0 < task["varies_widely_at"] < 1.0
    assert task["shrinks_substantially_at"] < task["fails_to_shrink_at"] <= 1.0
    # Precedence is declared, so two cells cannot both fire.
    assert "precedence" in record
    assert "some_arm_does_not_shrink" in record["precedence"]

    # And the thresholds are mine, flagged as such.
    assert "the choice made when it was written" in _flat(record["the_thresholds_are_not_ruled"])


def test_both_configs_carry_only_existing_inputs():
    import yaml

    def load(stem):
        return yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )

    known = {}
    for donor in ("p18_metric_space", "p21_ensemble_probe"):
        for entry in load(donor)["inputs"]:
            known.setdefault(
                entry["name"], (entry.get("path"), entry.get("rollup_sha256"))
            )
    for stem in ("p21_cohort_pair_separation", "p21_cross_arm_shrinkage"):
        for entry in load(stem)["inputs"]:
            assert entry["name"] in known, f"{stem}: {entry['name']}"
            assert (entry.get("path"), entry.get("rollup_sha256")) == known[
                entry["name"]
            ], f"{stem}: {entry['name']} differs from its donor"
            rollup = entry.get("rollup_sha256") or ""
            assert len(rollup) == 64 and set(rollup) != {"0"}, entry["name"]


def test_neither_measurement_is_phase_22_machinery():
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    # [UPDATED 2026-09-01] Phase 22 has its own tasks now; what this
    # pin holds is that NEITHER Phase 21 measurement is one of them.
    for name in ("cohort_pair_separation", "cross_arm_shrinkage"):
        assert "p22" not in name, name
        assert name in TASKS
    # [UPDATED AGAIN 2026-09-01] These two measurements are still Phase
    # 21 reckoning inputs. The only p22 config is the tie measurement,
    # which is a different thing from either of them.
    # [UPDATED AGAIN 2026-09-01] The six ranking arms were built once
    # the tie rule closed. What stays true here is that NONE of them is
    # a Phase 21 measurement: the p21 reckoning inputs are unchanged.
    # [UPDATED 2026-09-01] Post-run configs joined; the tie measurement
    # is still there and none of these is a Phase 21 measurement.
    p22_configs = {p.name for p in (REPO / "configs").glob("p22_*.yaml")}
    assert "p22_tie_fraction.yaml" in p22_configs
    assert {"p22_diagnostic.yaml", "p22_contrasts.yaml",
            "p22_shrinkage.yaml"} <= p22_configs
    assert "tie" not in "cohort_pair_separation cross_arm_shrinkage"

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 21 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p21" or "p21-" in e["id"]
    ]
    assert results_ledger.validate() is None


# --------------------------------------------------------------------------
# [2026-09-01] The dispersion statistic ruled: SD at 0.10, range rejected
# --------------------------------------------------------------------------


def test_the_cell_tests_the_sample_sd_not_the_range():
    import inspect
    import yaml

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_cross_arm_shrinkage)
    assert "dispersion = float(np.std(means, ddof=1))" in body
    assert "elif dispersion >= widely_at:" in body
    # The range is still computed and reported, and decides nothing.
    assert "spread_range = float(means.max() - means.min())" in body
    assert "elif spread_range >=" not in body
    assert '"range": spread_range,' in body

    task = yaml.safe_load(
        (REPO / "configs" / "p21_cross_arm_shrinkage.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert task["varies_widely_at"] == 0.10
    # The other two stand as built.
    assert task["fails_to_shrink_at"] == 0.90
    assert task["shrinks_substantially_at"] == 0.75


def test_the_range_version_is_kept_with_the_measurement_that_killed_it():
    record = phase21.SHRINKAGE_DISPERSION_STATISTIC_RULED
    assert "sample SD at 0.10" in record["ruled"]
    assert "range at 0.30 rejected" in record["ruled"]
    assert "kept here with its numbers, not overwritten" in _flat(
        record["the_rejected_version"]
    )
    assert "more informative than a threshold that was merely chosen well" in (
        _flat(record["why_the_rejected_version_is_kept"])
    )

    # The four grounds, each with its figure.
    sim = _flat(record["the_simulation"])
    for figure in ("0.2282", "0.3750", "0.1439", "0.0033", "0.9465",
                   "0.1414", "0.0029"):
        assert figure in sim, figure
    assert "0.2282" in record["ground_1_the_estimand_grows_with_n"]
    assert "94.65%" in record["ground_2_it_fires_on_homogeneous_arms"]
    assert "0.33%" in record["ground_2_it_fires_on_homogeneous_arms"]
    assert "TWO of 64 arms" in record["ground_3_two_arms_versus_all_of_them"]
    assert "0.144 -> 0.003" in record["ground_4_the_error_rate_falls_with_n"]


def test_the_simulated_grounds_reproduce():
    """The four grounds are measurements; they are re-measured here."""
    rng = np.random.default_rng(20260901)
    sigma = 0.08
    ranges, sds = {}, {}
    for n in (8, 64):
        x = rng.normal(0.55, sigma, size=(6000, n))
        ranges[n] = x.max(axis=1) - x.min(axis=1)
        sds[n] = x.std(axis=1, ddof=1)

    # 1. the range's estimand grows with n; the sd's does not.
    assert ranges[64].mean() > ranges[8].mean() + 0.10
    assert abs(sds[64].mean() - sds[8].mean()) < 0.01
    assert sds[64].mean() == pytest.approx(sigma, abs=0.005)

    # 2. at n=64 the range fires overwhelmingly, the sd almost never.
    assert float(np.mean(ranges[64] >= 0.30)) > 0.90
    assert float(np.mean(sds[64] >= 0.10)) < 0.02

    # 4. the sd's false-fire rate FALLS with n.
    assert float(np.mean(sds[64] >= 0.10)) < float(np.mean(sds[8] >= 0.10))

    # And the exact chi-square agrees, which the record cites.
    stats = pytest.importorskip("scipy.stats")
    exact64 = float(stats.chi2.sf(63 * (0.10 / sigma) ** 2, 63))
    assert exact64 == pytest.approx(0.0029, abs=5e-4)
    assert float(np.mean(sds[64] >= 0.10)) == pytest.approx(exact64, abs=0.01)


def test_the_strictness_is_recorded_with_the_ground():
    record = phase21.SHRINKAGE_DISPERSION_STATISTIC_RULED
    strict = _flat(record["it_is_not_a_re_expression"])
    assert "4.70" in strict and "d2 constant" in strict
    assert "SD 0.10 corresponds to RANGE 0.47" in strict
    assert "SUBSTANTIALLY STRICTER" in strict
    assert "dead zone correspondingly enlarge" in strict

    ground = _flat(record["the_ground_for_accepting_the_strictness"])
    assert "WEAKENS Phase 21's account" in ground
    assert "HARDER TO OVERTURN, NOT EASIER" in ground
    assert "guards against a comfortable reading" in ground

    # The d2 ratio, re-derived.
    rng = np.random.default_rng(20260901)
    x = rng.normal(0.55, 0.08, size=(6000, 64))
    ratio = float(
        (x.max(axis=1) - x.min(axis=1)).mean() / x.std(axis=1, ddof=1).mean()
    )
    assert ratio == pytest.approx(4.70, abs=0.05)
    assert 0.10 * ratio == pytest.approx(0.47, abs=0.01)


def test_the_discriminator_profile_is_on_the_record():
    record = phase21.SHRINKAGE_DISPERSION_STATISTIC_RULED
    profile = _flat(record["the_discriminators_measured_profile"])
    for figure in ("0.1005", "0.1133", "0.1257", "0.0996", "0.0029"):
        assert figure in profile, figure
    assert "means it" in profile

    # Re-derived from the exact distribution.
    stats = pytest.importorskip("scipy.stats")
    for true_sd, expected in ((0.1005, 0.50), (0.1133, 0.90), (0.1257, 0.99)):
        p = float(stats.chi2.sf(63 * (0.10 / true_sd) ** 2, 63))
        assert p == pytest.approx(expected, abs=0.02), true_sd
    assert float(stats.chi2.sf(63 * (0.10 / 0.09) ** 2, 63)) == pytest.approx(
        0.0996, abs=0.005
    )


def test_the_precedence_and_provenance_now_read_as_ruled():
    record = phase21.CROSS_ARM_SHRINKAGE_REGISTERED
    precedence = _flat(record["precedence"])
    assert "SAMPLE SD of the per-arm means, ddof=1" in precedence
    assert "a RANGE statistic was rejected on evidence" in precedence
    assert "single non-shrinking arm is the more consequential fact" in precedence

    # The original provenance note is preserved, marked superseded.
    assert record["the_thresholds_are_not_ruled"].startswith(
        "[SUPERSEDED 2026-09-01"
    )
    assert "the choice made when it was written" in record["the_thresholds_are_not_ruled"]
    ruled = _flat(record["the_thresholds_are_ruled"])
    assert "RULED 2026-09-01 " in ruled
    assert "STAND AS BUILT" in ruled
    assert "The flagging worked" in ruled
    assert "changed on a measurement rather than argued about after a result" in (
        ruled
    )


def test_the_dead_zone_is_named_and_not_papered_over():
    record = phase21.SHRINKAGE_DISPERSION_STATISTIC_RULED
    dead = _flat(record["the_dead_zone_stays_named"])
    assert "0.75 < max < 0.90" in dead
    assert "PLAUSIBLE outcome, not an exotic one" in dead
    assert "widens this zone relative to the range version" in dead
    assert "NO CELL MAY BE STRETCHED TO COVER IT" in dead
    assert "phase17.UNPREDICTED_PATTERN" in dead

    # The task really does refuse to stretch: cell None -> observation.
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_cross_arm_shrinkage)
    assert "cell = None" in body
    assert "NO COMMITTED CELL FIRES" in body
    assert "no cell is stretched" in body


def test_the_loader_refuses_a_range_calibrated_value_as_an_sd():
    from cleft.config.schema import ConfigError, _validate_task

    import yaml

    raw = yaml.safe_load(
        (REPO / "configs" / "p21_cross_arm_shrinkage.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert _validate_task(dict(raw))["varies_widely_at"] == 0.10
    # Out of range is refused, and the message says it is an SD.
    with pytest.raises(ConfigError, match="It is an SD, not a range"):
        _validate_task({**raw, "varies_widely_at": 1.5})
    # The ordering guard on the other two still holds.
    with pytest.raises(ConfigError, match="substantially < fails"):
        _validate_task({**raw, "shrinks_substantially_at": 0.95})


# --------------------------------------------------------------------------
# [2026-09-01] Both reckoning measurements ran; the arm A control
# --------------------------------------------------------------------------


def test_the_separation_measurement_fires_the_viable_cell():
    record = phase21.COHORT_PAIR_SEPARATION_OBSERVED
    assert record["n_pairs"] == 237 * 236 // 2 == 27966
    assert record["sd_obs"] == 0.657279

    # The two scales reproduce from sd_obs and the banked reliability.
    import math

    from cleft.data import reliability

    k_single = math.sqrt(1 - reliability.RELIABILITY_237)
    assert record["se_single"] == pytest.approx(
        record["sd_obs"] * k_single, abs=5e-6
    )
    assert record["se_diff"] == pytest.approx(
        record["sd_obs"] * math.sqrt(2) * k_single, abs=5e-6
    )
    # SE_diff is the larger scale, so its fractions are >= SE_single's.
    f = record["fraction_below"]
    for m in (1, 2, 3):
        assert f[f"se_diff_x{m}"] >= f[f"se_single_x{m}"], m
    # Monotone in the multiple.
    for scale in ("se_single", "se_diff"):
        vals = [f[f"{scale}_x{m}"] for m in (1, 2, 3)]
        assert vals == sorted(vals), scale

    # The declared cell fired, by identity.
    assert record["primary_scale_declared"] == "se_diff_x1"
    assert f["se_diff_x1"] == 0.2485
    assert f["se_diff_x1"] < 0.5, "most pairs CLEAR the noise"
    assert _resolve21(record["cell_fired"]) is (
        phase21.COHORT_PAIR_SEPARATION_DESIGNED[
            "reading_most_pairs_clear_the_noise"
        ]
    )
    assert "VIABLE" in record["the_verdict"]
    assert "THREE-QUARTERS" in record["the_verdict"]


def test_the_priors_miss_is_recorded_in_the_favourable_direction():
    record = phase21.COHORT_PAIR_SEPARATION_OBSERVED
    prior = _flat(record["the_priors_performance"])
    assert "expectation was 0.33" in prior
    assert "0.2485" in prior
    assert "MORE USABLE PAIRS THAN PREDICTED" in prior
    assert "wrong in the FAVOURABLE direction" in prior
    assert "easy to leave unremarked" in prior
    # Discreteness was named in advance, and the outturn supports it.
    assert "discreteness" in _flat(
        phase21.COHORT_PAIR_SEPARATION_DESIGNED[
            "the_prior_registered_before_the_number"
        ]
    )
    assert record["n_distinct_separations"] == 17
    assert "17 distinct values" in prior
    # The registered prior really did say 0.33 for se_diff.
    assert "0.33 / 0.61 / 0.80" in phase21.COHORT_PAIR_SEPARATION_DESIGNED[
        "the_prior_registered_before_the_number"
    ]


def test_the_shrinkage_measurement_replaces_the_illustration():
    record = phase21.CROSS_ARM_SHRINKAGE_OBSERVED
    across = record["across_arms"]
    assert across["min"] == 0.0656 and "identity" in across["min_arm"]
    assert across["median"] == 0.3289
    assert across["max"] == 2.1730 and across["max_arm"] == "p17_arm_a"
    assert record["probe"] == 0.4947
    assert across["sd"] == 0.2704

    # The denominator is the separation run's own sd_obs -- same cohort,
    # same label, so the two measurements agree on it.
    assert "0.657279" in record["denominator_verified"]
    assert phase21.COHORT_PAIR_SEPARATION_OBSERVED["sd_obs"] == 0.657279

    # The fired cell, by identity, and the precedence that chose it.
    assert _resolve21(record["cell_fired"]) is (
        phase21.CROSS_ARM_SHRINKAGE_REGISTERED["readings"][
            "some_arm_does_not_shrink"
        ]
    )
    assert across["max"] >= 0.90, "the fail line"
    assert across["sd"] >= 0.10, "the widely line would also have cleared"
    assert "more consequential fact" in _flat(record["precedence_held"])

    # What it replaces.
    replaces = _flat(record["what_it_replaces"])
    assert "ONE SEED OF ONE ARM" in replaces
    assert "64 ARMS MEASURED FROM CACHED CSVs" in replaces
    assert "63 of them shrinking" in replaces
    assert "SHRINKAGE_FIGURE_MISATTRIBUTED" in replaces
    # 63 shrink, one does not.
    assert 64 - 1 == 63


def test_arm_a_is_the_control_the_registered_reading_named():
    record = phase21.ARM_A_THE_SHRINKAGE_CONTROL
    expands = _flat(record["arm_a_expands"])
    assert "2.1730" in expands
    assert "NO LABEL MEAN TO SHRINK TOWARD" in expands
    assert "cannot regress to its centre" in expands

    three = _flat(record["one_property_explains_three_banked_figures"])
    for figure in ("1.878", "2.7544", "1.4348", "0.6587", "1.61",
                   "0.51-0.62"):
        assert figure in three, figure
    assert "SAME FACT" in three

    control = _flat(record["the_control"])
    assert "+0.0956" in control and "-0.0743" in control
    assert "+0.7012" in control and "+0.5729" in control
    assert "CHANCE LEVEL" in control
    assert "~0.07" in control
    # Geirhos's OOD band is live in the literature bank.
    from cleft import literature

    assert "0.066-0.068" in literature.GEIRHOS_ERROR_CONSISTENCY[
        "cnn_to_human_ood"
    ]
    # Arm A's kappa really is nearer the OOD band than the all-pairs mean.
    assert abs(0.0956 - 0.07) < abs(0.0956 - 0.7012)

    # The registered reading's own words, quoted from the record.
    fires = _flat(record["the_registered_readings_own_words_fire"])
    assert "the control the account never had" in fires
    assert "the control the account never had" in _flat(
        phase21.CROSS_ARM_SHRINKAGE_REGISTERED["readings"][
            "some_arm_does_not_shrink"
        ]
    )
    assert "The second branch is what happened" in fires

    # Both ends of the argument, with the figures live at their sources.
    closed = _flat(record["the_argument_closed_at_both_ends"])
    assert "-0.8470" in closed and "+0.7521" in closed
    assert "CHANCE" in closed
    assert "-0.8470" in phase21.POST_HOC_ANALYSES[
        "4_residual_sign_hardness_vs_signed_distance"
    ]
    assert "+0.7521" in phase21.POST_HOC_ANALYSES[
        "3_worst_quartile_hardness_vs_distance_from_the_centre"
    ]
    assert phase21.ARM_B_OBSERVED["residual_sign"]["mean_kappa"] == 0.7012
    assert phase21.ARM_B_OBSERVED["worst_quartile"]["mean_kappa"] == 0.5729

    # And the kappa figures are tagged post-hoc, firing no cell.
    post_hoc = _flat(record["the_kappa_figures_are_post_hoc"])
    assert "[POST-HOC]" in post_hoc
    assert "firing NO committed cell" in post_hoc
    assert phase21.POST_HOC_ANALYSES["tag"] == "[POST-HOC]"


def test_phase_22s_consequence_is_recorded_before_its_restate():
    from cleft import phase17

    record = phase21.PHASE_22_CONSEQUENCE_RECORDED
    assert "BEFORE Phase 22's restate" in record["recorded"]

    proof = _flat(record["arm_a_is_an_existence_proof"])
    assert "0.2334" in proof and "0.2520" in proof
    assert "UNRESOLVED" in proof
    # Both figures live at their sources.
    assert phase17.ARM_MEANS["arms"]["p17_arm_a"]["pcc"] == 0.2334
    from cleft import ladder

    assert ladder.TRADE_OFF_PAIR["result"]["vit_paired_mean"] == 0.2520

    assert "DEMONSTRATED BY A MEASURED CASE" in _flat(
        record["not_shrinking_buys_nothing_on_pcc"]
    )
    narrows = _flat(record["the_hypothesis_narrows"])
    assert "DIFFERENT FEATURES" in narrows
    assert "AVOID THE MEAN" in narrows
    assert "SEPARABLE" in narrows

    must = _flat(record["it_must_appear_in_the_reckoning"])
    assert "may not be discovered afterwards" in must
    assert "positive hypothesis" in must

    harder = _flat(record["and_the_test_is_harder_than_the_hypothesis"])
    assert "0.04 to 0.10" in harder
    assert "weaker AND harder to test" in harder
    assert "0.04 to 0.10" in _flat(
        ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"]["unresolvable_band"]
    )

    # The prohibition it defers to is untouched.
    assert "CANNOT EXPLAIN THE PCC CEILING" in (
        phase21.SCALE_INVARIANCE_PROHIBITION
    )


def test_the_repository_policy_and_its_violation_are_recorded():
    """[UPDATED 2026-09-03] The record was rewritten as single-maintainer
    repository policy -- one committer, cluster pull-only, runs launched
    by the maintainer -- so the pin follows it. **Every fact it held is
    still asserted**: the commit sha, the unpushed cost, what stopped
    what, the check, and the second-error diagnosis."""
    from cleft import record_audit

    record = record_audit.REPOSITORY_POLICY
    policy = _flat(record["the_policy"])
    assert "one committer" in policy
    assert "PULL-ONLY" in policy
    assert "launched by the maintainer" in policy
    assert "not sufficient authorisation" in policy

    violation = _flat(record["the_violation_2026_09_01"])
    assert "9c91064" in violation
    assert "git pull --ff-only" in violation
    assert "all three outside the policy" in violation
    assert "not lifted by a sentence in a message" in violation

    cost = _flat(record["what_it_cost"])
    assert "the commit never reached the remote" in cost
    assert "78f559f" in cost and "48a0c69" in cost
    assert "neither config existed" in cost
    assert "confirmed only that no NEW commits existed upstream" in cost

    stopped = _flat(record["what_stopped_what"])
    assert "guard 3" in stopped
    assert "The commit was stopped by nothing" in stopped
    assert "no guard in the repository can prevent one" in stopped

    check = _flat(record["the_check_added"])
    assert "git rev-list --count @{u}..HEAD" in check
    assert "git status -sb" in check
    assert "a REPORT, not a GUARD" in check
    assert "reported at the end of every working session" in check

    worse = _flat(record["the_second_error_is_the_worse_one"])
    assert "a rule broken" in worse and "a claim not checked" in worse
    # The lesson it cites is live at its source.
    from cleft import ladder

    assert "precisely the claim nobody audits" in _flat(
        ladder.BASAL_RATIONALE_UNSUPPORTED["lesson"]
    )

    stops = _flat(record["what_happens_when_a_message_says_commit"])
    assert "the policy is cited and the work stops there" in stops
    assert "a standing policy re-opened on each request is not standing" in stops

    # No party is named anywhere in the record.
    blob = " ".join(str(value) for value in record.values())
    for name in ("Claude", "El Mahdi", "Liu"):
        assert name not in blob, name


#: The canonical upstream, owner included. The repository NAME alone is
#: not a discriminator: every fork shares it, and so does the public
#: snapshot at ``cleft-aesthetics-pub``.
CANONICAL_SLUG = "xRedRingx/cleft-aesthetics"


def _origin_url(repo) -> str:
    """``origin``'s URL, or an empty string if there is no origin."""
    import subprocess

    result = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        cwd=repo, capture_output=True, text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _is_canonical_origin(url: str) -> bool:
    """Is this URL the canonical upstream, and not something that merely
    starts with it?

    **[2026-09-06] The bug this replaces was a bare ``in``.** The slug is
    a prefix of ``xRedRingx/cleft-aesthetics-pub``, so a substring test
    called the public snapshot canonical. What follows the slug must be a
    BOUNDARY: end of string, ``.git``, or a trailing slash. Anything else
    is a different repository whose name happens to start the same way.
    """
    if not url or CANONICAL_SLUG not in url:
        return False
    tail = url.split(CANONICAL_SLUG, 1)[1]
    return tail in ("", ".git", "/", "/.git")


def test_the_boundary_discriminator_matches_the_private_repo_and_nothing_else():
    """[ADDED 2026-09-06] Both directions, on all three URL forms.

    A discriminator asserted rather than exercised is how the suffix bug
    survived. These are the URLs that actually exist: the private
    repository over HTTPS and SSH, and the public snapshot, which must
    NOT match."""
    CANONICAL = [
        "https://github.com/xRedRingx/cleft-aesthetics.git",
        "https://github.com/xRedRingx/cleft-aesthetics",
        "git@github.com:xRedRingx/cleft-aesthetics.git",
    ]
    NOT_CANONICAL = [
        # The public snapshot. This is the one that used to match.
        "https://github.com/xRedRingx/cleft-aesthetics-pub.git",
        "https://github.com/xRedRingx/cleft-aesthetics-pub",
        "git@github.com:xRedRingx/cleft-aesthetics-pub.git",
        # A fork by anyone else, which shares the name and not the owner.
        "https://github.com/someone-else/cleft-aesthetics.git",
        # A name that merely contains the slug.
        "https://github.com/xRedRingx/cleft-aesthetics-archive.git",
        # No origin at all.
        "",
    ]
    for url in CANONICAL:
        assert _is_canonical_origin(url), f"should match: {url}"
    for url in NOT_CANONICAL:
        assert not _is_canonical_origin(url), f"should NOT match: {url}"

    # And the specific regression, stated as itself: the old test was
    # `CANONICAL_SLUG in url`, which is True for the public snapshot.
    public = "https://github.com/xRedRingx/cleft-aesthetics-pub.git"
    assert CANONICAL_SLUG in public, "the substring bug is real"
    assert not _is_canonical_origin(public), "and the anchor fixes it"


def test_the_boundary_check_is_runnable_and_currently_clean():
    """The check is only useful if it runs. It does, and it reports
    zero unpushed commits, which is what the policy holding looks like
    from the repository's side."""
    import subprocess

    # [AMENDED 2026-09-05] **Skip outside this working copy.** The guard
    # asserts a POLICY about who commits to this repository, and that
    # policy says nothing about anyone else's clone. Before this, a
    # reader who cloned the public repo and made one commit of their own
    # got a red suite whose message told them they had violated a rule
    # they had never agreed to -- the worst possible first impression,
    # and a false one. The guard is unchanged HERE, which is the only
    # place it means anything.
    # The discriminator is the CANONICAL UPSTREAM, owner included --
    # not the repository name, which every fork shares.
    #
    # **[CORRECTED 2026-09-06] The match was a bare substring test and it
    # matched a SUFFIX.** The public snapshot is
    # `xRedRingx/cleft-aesthetics-pub`, and the canonical slug is a
    # PREFIX of it, so `CANONICAL in origin` was true there and the guard
    # activated on exactly the reader the 2026-09-05 amendment was
    # written to protect. The amendment was right and its implementation
    # was not. `_is_canonical_origin` anchors on the slug's boundary, so
    # a suffix no longer matches, and `test_the_boundary_discriminator`
    # exercises it in BOTH directions rather than asserting it.
    if not _is_canonical_origin(_origin_url(REPO)):
        pytest.skip(
            f"origin is not {CANONICAL_SLUG} -- the one-committer policy "
            "is about this repository, not about a reader's fork or clone"
        )
    result = subprocess.run(
        ["git", "rev-list", "--count", "@{u}..HEAD"],
        cwd=REPO, capture_output=True, text=True,
    )
    if result.returncode != 0:  # pragma: no cover - no upstream configured
        pytest.skip("no upstream branch configured in this checkout")
    ahead = int(result.stdout.strip())
    assert ahead == 0, (
        f"{ahead} commit(s) ahead of the remote. This repository has one "
        "committer and the working session does not commit "
        "(record_audit.REPOSITORY_POLICY); if this fires, a commit was "
        "made that should not have been"
    )
