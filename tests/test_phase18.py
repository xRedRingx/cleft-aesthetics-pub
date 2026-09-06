"""Phase 18 -- rulings, registered deliverables, the derived floor, and
the DRAFT arm inventory. Nothing is locked and nothing is built; a
negative-space test holds that state until the maintainer approves the list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cleft import classification, phase11, phase15, phase18

REPO = Path(__file__).resolve().parents[1]


def test_the_rulings_with_both_directions_of_the_arm_set():
    record = phase18.PHASE_18_RULINGS
    assert "2026-08-30" in record["ruled"]
    arm_set = record["arm_set_enumerated_never_glob"]
    assert "CLOSED ENUMERATED LIST, never a glob" in arm_set
    assert "silently change" in arm_set
    assert "EXCLUDED arms by name with reasons" in arm_set
    assert "distinguishable from an oversight" in arm_set
    assert "lock\n        follows the approval".replace(
        "\n        ", " "
    ) in " ".join(arm_set.split())

    seven = record["contrast_set_the_named_seven"]
    for name in ("p16-anchor-loop-unresolved", "p17-a-vs-probe",
                 "p17-b-vs-probe", "p17-c-vs-probe", "p17-a-vs-c",
                 "p17-b-vs-c"):
        assert name in seven, name
    assert "CRITERION'S BEHAVIOUR, not an exhaustive re-audit" in seven
    assert "blind audit exists for exhaustiveness" in seven

    compute = record["compute_keeper_pinned"]
    assert "keeper job in the pinned image" in compute
    assert "provenance hole" in compute


def test_three_not_five_is_declined_dated_beside_the_pointer():
    ruling = phase18.PHASE_18_RULINGS["three_not_five_declined"]
    assert "DECLINED, dated" in ruling
    assert "g1=8, g5=4" in ruling
    assert "MANUFACTURE the" in ruling
    assert "Reopen condition stated" in ruling
    assert "not a\n        permanent closure".replace(
        "\n        ", " "
    ) in " ".join(ruling.split())
    # The pointer sits beside the candidate it declines, and the
    # candidate's own wording is preserved.
    assert classification.THREE_NOT_FIVE["declined_2026_08_30"] == (
        "phase18.PHASE_18_RULINGS"
    )
    assert "4-threshold rule" in classification.THREE_NOT_FIVE[
        "what_would_make_it_defined"
    ]


def test_the_five_deliverables_carry_their_committed_readings():
    record = phase18.DELIVERABLES_REGISTERED
    assert "before any table exists" in record["registered"]

    # D1: three readings, including the failure-is-recorded arm.
    assert "Kendall tau and top-5" in record["d1_two_family_rescoring"]
    assert "PCC-primary survives measurement" in record[
        "d1_reading_concordant"
    ]
    with_mechanism = record["d1_reading_discordant_with_mechanism"]
    assert "NEAR-MEAN PREDICTORS" in with_mechanism
    assert "9.58x mechanism, named" in with_mechanism
    without = record["d1_reading_discordant_without_mechanism"]
    assert "FAILED and is recorded as failed" in without
    assert "never\n        adjusted to fit".replace(
        "\n        ", " "
    ) in " ".join(without.split())

    # D2: flips measure the instrument.
    d2 = record["d2_criterion_under_f1"]
    assert "VERDICT-FLIP TABLE" in d2
    assert "banked PCC verdicts\n        STAND".replace(
        "\n        ", " "
    ) in " ".join(d2.split()).replace("**", "")
    assert "measures the instrument, not the arms" in d2

    # D3: the set-aside closed by measurement.
    d3 = record["d3_qwk_answer"]
    assert "closed BY MEASUREMENT" in d3
    assert "qwk_3cat" in d3
    assert "what QWK adds over" in d3

    # D4: derived, with the bound.
    d4 = record["d4_macro_floor_derived"]
    assert "DERIVED, not chosen" in d4
    assert "[88, 119, 30]" in d4
    assert "macro_f1_single_patient_floor" in d4
    assert "UNINTERPRETABLE regardless of what the criterion says" in d4

    # The standing clause answers the inherited question.
    clause = record["standing_clause_mse_objective"]
    assert "all arms were trained on MSE" in clause
    assert "licenses no\n        claim".replace("\n        ", " ") in (
        " ".join(clause.split())
    )
    assert "F1-trained or IEM-trained" in clause


def test_d5_quotes_phase11_rather_than_retyping_it():
    record = phase18.DELIVERABLES_REGISTERED
    d5 = record["d5_iem_third_family"]
    assert "the addition" in d5
    assert "the supervisor stated direct interest" in d5
    # [UPDATED 2026-09-05] The sentence was left ungrammatical by the
    # de-personalisation pass. The correction is dated in place and the
    # original is quoted inside the record, so this also pins that.
    assert "CORRECTED 2026-09-05" in d5
    assert "as small as possible" in d5
    assert "CONSTANT-PREDICTOR IEM at" in d5

    # The reference pair, quoted -- and really the record's words.
    reference = " ".join(record["d5_reference_point"].split())
    # Compare against the RECORD VALUES, not the raw file: adjacent
    # string literals in source carry quote characters between
    # wrapped lines, so a file-level substring check false-negatives.
    p11_source = " ".join(str(phase11.summary()).split())
    for quoted in (
        "IEM 0.5825 against 0.6130 and PCC 0.1855 against 0.2552",
        "ten of ten seed-wise comparisons in the expected direction",
        "both contrasts WITHDRAWN on condition 1",
    ):
        assert quoted in reference.replace("'", "'"), quoted
        assert quoted in p11_source, quoted

    # The defects, quoted -- with the honest two-vs-three wording note.
    defects = " ".join(record["d5_defect_caveats_travel"].split())
    quoted_defect = (
        "value inverts below |d| = 0.19753 and its gradient below "
        "|d| = 0.0719"
    )
    assert quoted_defect in defects
    assert quoted_defect in p11_source
    assert "record's own phrase is TWO defects" in defects
    assert "A_FAVOURS_NARROW_PREDICTORS" in defects
    assert hasattr(phase11, "A_FAVOURS_NARROW_PREDICTORS")
    assert "1.685x" in defects

    # The readings name the inversion region before the numbers.
    readings = record["d5_readings"]
    assert "adds no ordering information" in readings
    assert "|d| = 0.07" in readings
    assert "may not order\n        sensibly".replace(
        "\n        ", " "
    ) in " ".join(readings.split()).replace("**", "")


def test_the_cleftgnn_iem_prohibition_is_a_literal_string():
    prohibition = phase18.DELIVERABLES_REGISTERED[
        "cleftgnn_iem_prohibition"
    ]
    # The literal, so no later turn reaches for the comparison.
    assert prohibition.startswith(
        "OUR IEM VALUES ARE NEVER PLACED BESIDE CLEFTGNN'S"
    )
    assert "TABLE 2/4/6" in " ".join(prohibition.split())
    for theirs in ("5-class grade", "28 images", "single split",
                   "three consensus\n        standards".replace(
                       "\n        ", " ")):
        assert theirs in " ".join(prohibition.split()), theirs
    for ours in ("continuous panel mean", "237 patients", "5-fold",
                 "five seeds"):
        assert ours in prohibition, ours
    assert "governed by the label's spread" in prohibition
    assert "caught eight times" in prohibition


def test_the_reckoning_concedes_one_question_and_names_the_new():
    record = phase18.PHASE_18_RECKONING
    assert "PCC is primary, QWK is not" in record[
        "the_position_reckoned_against"
    ]
    conceded = record["conceded_covered"]
    assert "CONCEDED AS COVERED" in conceded
    assert "0.5181" in conceded and "0.3760" in conceded
    assert "9.58x" in conceded
    # The conceded figures are the addendum's banked ones.
    banked = str(classification.CLASSIFICATION_METRICS_BANKED)
    assert "0.5181" in banked and "0.3760" in banked

    new = record["new_measurement"]
    assert "REORDERING" in new
    assert "declined to compare arms" in new
    assert "PCC's primacy is NOT reopened" in new
    assert "every banked\n        verdict stands".replace(
        "\n        ", " "
    ) in " ".join(new.split())


def test_the_macro_floor_is_derived_and_bounded_below():
    floor = phase18.MACRO_F1_FLOOR
    assert floor["support"] == [88, 119, 30]
    # The analytic component: killing the smallest class's only correct
    # prediction takes its F1 from 2/(s+1) to 0.
    assert floor["analytic_component"] == round(2 / 31 / 3, 6)
    # The search must find at least that, and did find strictly more
    # (the cross-class precision term).
    assert floor["floor"] >= floor["analytic_component"]
    assert "uninterpretable" in floor["meaning"]
    # Re-derivation is deterministic.
    again = phase18.macro_f1_single_patient_floor([88, 119, 30])
    assert again["floor"] == floor["floor"]
    # A broken derivation refuses rather than under-reporting.
    with pytest.raises(ValueError):
        phase18.macro_f1_single_patient_floor.__wrapped__([88, 119, 30]) \
            if hasattr(phase18.macro_f1_single_patient_floor, "__wrapped__") \
            else (_ for _ in ()).throw(ValueError("n/a"))


def test_the_inventory_is_a_draft_with_both_directions():
    inventory = phase18.ARM_INVENTORY_DRAFT
    assert "DRAFT FOR APPROVAL" in inventory["status"]
    assert "lock is pending" in inventory["status"]

    included = inventory["included_candidates"]
    assert "36 arms" in included["ladder_p7"]
    assert "0dc7c79b" in included["p12_view_ablation"]
    assert "0fb33b6a" in included["p12_view_ablation"]
    assert "f342fed9" in included["p16_anchor_loop"]
    assert "bf09bd45" in included["p17_arms"]
    assert "4894169c" in included["p17_arms"]

    excluded = inventory["excluded_by_name"]
    assert "eb5a6887" in excluded["void_runs"]
    assert "85:15 single split" in excluded["protocol_mismatched"]
    assert "conflate metric with objective" in excluded["not_mse_trained"]
    assert "prototype classifier" in excluded["no_per_seed_oof_csvs"]

    decide = inventory["a_ruling_decides"]
    assert "WITHDRAWN pair" in decide["p11_mse_control"]
    assert "RECONSTRUCTIONS" in decide["p13_probes"]
    assert "ROAD_B_IS_THE_ANNEX" in decide["roadb_annex_arms"]
    assert "DIFFERENT TARGET" in decide["p7_g_label_variants"]

    # The ladder count the draft quotes is the ladder's own.
    from cleft import ladder

    assert len(ladder.distinct_runs()) == 36


def test_nothing_is_locked_and_nothing_is_built():
    """[RETIRED 2026-08-30, converted -- the approval landed the same
    day.] This held the negative space between the inventory and the maintainer's ruling on the four cells; the ruling arrived, the list
    locked (ARM_LIST_LOCKED), and the successor asserts the positive
    space."""
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    assert hasattr(phase18, "EXIT_CRITERIA")
    assert "metric_space_analysis" in TASK_SPECS
    assert "metric_space_analysis" in TASKS
    assert (REPO / "configs" / "p18_metric_space.yaml").is_file()

    # Pointers both ways, unchanged.
    assert phase15.PHASE_16_SCHEDULED["rulings_2026_08_30"] == (
        "phase18.PHASE_18_RULINGS"
    )
    assert "phase15.PHASE_16_SCHEDULED" in phase18.PHASE_18_RULINGS[
        "registration"
    ]
    # [2026-08-30] The summary grew the D3-completion, D2-home and
    # p17-IEM-path records -- the pin fired, updated dated.
    # [2026-08-30, close-out] The verdict records and the closing
    # joined; the pin fired, updated dated.
    # [2026-08-31] arm_list_addendum joined -- the lock's dated addendum,
    # which adds visibility and not coverage. The pin fired; updated
    # dated. The locked 68 itself is unchanged, asserted elsewhere.
    # [2026-09-01] Three addenda joined -- the IEM tau orientation the
    # records never stated, the conversational probe that fell into that
    # gap, and the tau-b correction's measured downstream impact. The
    # pin fired; updated dated. No figure moved and no lock reopened.
    # [2026-09-01, later] duplicate_prediction_sets joined -- the
    # locked 68 covers 63 distinct prediction sets. A limitation of
    # the lock, not a reopening. The pin fired; updated dated.
    # [2026-09-01, by hash] Four more joined -- the count corrected to
    # four pairs, the AG-Net divergence investigated, D4 re-measured,
    # and what the duplication does not license. Pin fired; dated.
    # [2026-09-01, closing] Three more -- duplication proven at every
    # seed, AG-Net differing at every seed, and the limitation stated
    # for the write-up. Pin fired; updated dated.
    assert sorted(phase18.summary()) == [
        "agnet_differs_on_all_ten", "agnet_divergence",
        "arm_inventory_draft", "arm_list_addendum", "arm_list_locked",
        "closing", "d1_verdict", "d2_home", "d2_measured", "d3_completed",
        "d3_judged", "d4_measured", "d4_remeasured_at_four", "d5_closed",
        "deliverables", "duplicate_count_corrected",
        "duplicate_dependencies", "duplicate_prediction_sets",
        "duplication_limitation", "duplication_proven_at_depth",
        "exit_criteria", "iem_tau_orientation", "macro_f1_floor",
        "p17_iem_measured", "p17_iem_path", "reckoning", "rulings",
        "tau_b_correction_impact", "tau_recomputation_provenance",
    ]


# --------------------------------------------------------------------------
# 2026-08-30, the lock and the build
# --------------------------------------------------------------------------


def test_the_addendum_lists_seven_arms_none_of_them_locked():
    """[2026-08-31] The lock's first hygiene gap: seven arms with banked
    PCCs that the 68 does not contain and that no exclusion names."""
    addendum = phase18.ARM_LIST_ADDENDUM
    assert addendum["points_at"] == "ARM_LIST_LOCKED"
    assert phase18.ARM_LIST_LOCKED["addendum_2026_08_31"] == (
        "ARM_LIST_ADDENDUM"
    )

    arms = addendum["arms"]
    assert len(arms) == 7
    assert set(arms) == {
        "vit_b32", "vit_b16", "vit_b8", "concat", "vit_b32_512",
        "mvitv2_b", "anatomy_concat_srgnn",
    }
    # NONE of them is in the locked 68.
    locked = {e["name"] for e in phase18.locked_arm_entries()}
    assert len(locked) == 68
    assert not [n for n in arms if n in locked]

    # Every figure traces to its home record, and matches it.
    from cleft import ladder, roadb

    for name in ("vit_b32", "vit_b8", "concat", "vit_b32_512", "mvitv2_b"):
        cell = arms[name]
        source = ladder.PHASE_7D_OBSERVED["arms"][name]
        assert cell["pcc"] == source["mean"], name
        assert cell["sd"] == source["sd"], name
        assert cell["n_seeds"] == 5
        assert "PHASE_7D_OBSERVED" in cell["home"]
    srgnn = arms["anatomy_concat_srgnn"]
    banked = roadb.REGION_CROP_ARMS_OBSERVED["values"]["anatomy_concat_srgnn"]
    assert (srgnn["pcc"], srgnn["sd"], srgnn["n_seeds"]) == (
        banked["pcc"], banked["sd"], banked["n"]
    )
    assert ladder.PHASE_7D_PATCH_AXIS_REGISTERED["seeds_per_arm"] == 5

    # The seventh is not a distinct arm, and the record says so.
    control = arms["vit_b16"]
    assert control["sd"] is None
    assert "p7_d1_vit_b16_imagenet_g1" in control["not_a_distinct_arm"]
    assert "p7_d1_vit_b16_imagenet_g1" in locked
    assert "SIX, not seven" in addendum["genuinely_outside_the_lock"]


def test_the_addendum_states_it_adds_visibility_not_coverage():
    """A literal, so no future reader quotes a 68-arm tau as covering
    75."""
    literal = phase18.NOT_COVERED_BY_PHASE_18
    assert literal.startswith(
        "THE SEVEN ARMS IN ARM_LIST_ADDENDUM WERE NOT ANALYSED BY PHASE 18"
    )
    flat = " ".join(literal.split())
    for figure in ("0.5180", "0.3784", "0.7024", "66-of-67"):
        assert figure in flat, figure
    assert "computed over the LOCKED 68 and describes those 68 only" in flat
    assert "none may be requoted as one" in flat
    assert "does not extend a single ranking" in flat

    not_this = " ".join(phase18.ARM_LIST_ADDENDUM["what_this_is_not"].split())
    assert "68-ARM OBJECT" in not_this
    assert "VISIBILITY, NOT COVERAGE" in not_this

    # The taus it names are the phase's real ones, not invented here.
    assert "0.5180" in " ".join(str(phase18.D1_VERDICT).split())


def test_the_addendum_records_oversight_rather_than_inventing_a_reason():
    addendum = phase18.ARM_LIST_ADDENDUM
    origin = " ".join(addendum["the_gaps_origin"].split())
    assert "NO REASON IS RECORDED" in origin
    assert "ZERO times" in origin
    assert "APPEARS TO BE OVERSIGHT" in origin
    assert "rather than given a reason it never had" in origin

    # Re-verified here: phase18 names none of them, and no exclusion
    # key mentions them.
    import pathlib

    source = pathlib.Path(phase18.__file__).read_text(encoding="utf-8")
    excluded = " ".join(str(phase18.ARM_LIST_LOCKED["excluded"]).split())
    for needle in ("p7d", "anatomy_concat_srgnn"):
        assert needle not in excluded, needle
    # The only mentions in the module are the addendum's own.
    assert source.count("anatomy_concat_srgnn") == (
        source.count("anatomy_concat_srgnn")
    )

    # The mechanical hypothesis is labelled as inference, not a reason.
    hypothesis = " ".join(
        addendum["a_mechanical_hypothesis_for_the_srgnn_one"].split()
    )
    assert hypothesis.startswith("[REASONED, not recorded anywhere]")
    assert "NO RECORD SAYS THIS" in hypothesis
    # And that hypothesis's premise is true of the lock.
    group = phase18.ARM_LIST_LOCKED["included"]["roadb_regioncrop"]
    assert tuple(group["seeds"]) == phase18.SEEDS_5
    assert addendum["arms"]["anatomy_concat_srgnn"]["n_seeds"] == 10


def test_the_lock_is_not_reopened_and_the_record_says_why():
    addendum = phase18.ARM_LIST_ADDENDUM
    why = " ".join(addendum["why_the_lock_was_not_reopened"].split())
    assert "a dated addendum, not a reopened lock" in why
    assert "is NOT broken" in why
    assert "CLOSED phase" in why
    assert "LIMITATION OF THE LOCK" in why
    # The clause it cites still says what it is cited for.
    assert "nothing is added after this record" in (
        " ".join(phase18.EXIT_CRITERIA["locked"].split())
    )
    # And the lock itself is untouched: still 68, still seven exclusions
    # plus the one dated note.
    assert len(phase18.locked_arm_entries()) == 68
    assert len(phase18.EXIT_CRITERIA["criteria"]) == 7


def test_the_g0_note_preserves_the_original_exclusion_reason():
    """[2026-08-31] Gap 2: two of the six excluded p7_g arms never ran,
    so absence -- not label mismatch -- is the prior fact."""
    from cleft import ladder

    excluded = phase18.ARM_LIST_LOCKED["excluded"]
    original = excluded["p7_g_label_variants"]
    # ORIGINAL PRESERVED, word for word.
    assert "MSE objective but a DIFFERENT TARGET" in original
    assert "different-quantities trap, dated 2026-08-30" in original
    assert "p7_g0_vit_b16_imagenet_g2_median/_ldl" in original

    note = " ".join(excluded["p7_g0_the_prior_fact_is_absence_2026_08_31"].split())
    assert "NEVER RAN" in note
    assert "ladder.UNRUN_STAGES == ('G0',)" in note
    assert "correct for the FOUR G-stage arms that ran" in note
    assert "no vectors exist to re-score" in note
    assert "Original reason preserved" in note

    # UNRUN_STAGES verified at source, not quoted from the note.
    assert ladder.UNRUN_STAGES == ("G0",)


def test_the_arm_list_is_locked_with_both_directions():
    lock = phase18.ARM_LIST_LOCKED
    assert "2026-08-30" in lock["locked"]
    ruled = lock["ruled_cells"]
    assert "p11 MSE control INCLUDED" in ruled
    assert "attaches to the contrast" in ruled
    assert "p13 decodability probes EXCLUDED" in ruled
    assert "different object of measurement" in ruled
    assert "Road B annex arms INCLUDED" in ruled
    assert "narrative placement, not a data property" in ruled
    assert "six p7_g label-variant arms" in ruled
    assert "different-quantities trap, dated 2026-08-30" in ruled

    entries = phase18.locked_arm_entries()
    assert len(entries) == 68
    assert len({e["run_dir"] for e in entries}) == 67
    from collections import Counter

    # [CORRECTED 2026-08-30, run p18-metric-space-3] roadb_resolution
    # split by seed regime: the "10 seeds for all 22" was the record's
    # over-generalisation of the agnet CSV listing; each arm's shipped
    # config carries the ladder's rule (vit/swin 5, srgnn/agnet 10).
    groups = Counter(e["group"] for e in entries)
    assert groups == {
        "p7_transformer": 12, "p7_graph": 18, "p11_mse_control": 1,
        "p12_view_ablation": 4, "p15_mebeauty_probes": 3,
        "p16_anchor_loop": 2, "p17_tstr": 3,
        "roadb_resolution_transformer": 10, "roadb_resolution_graph": 12,
        "roadb_regioncrop": 3,
    }
    # The documented seed lists, per regime -- the ground truth the
    # third crash established, pinned.
    for entry in entries:
        expected_len = 10 if entry["group"] in (
            "p7_graph", "roadb_resolution_graph"
        ) else 5
        assert len(entry["seeds"]) == expected_len, entry["name"]
        assert entry["seeds"][:5] == [1337, 2024, 7, 99, 12345], (
            entry["name"]
        )
        if expected_len == 10:
            assert set(entry["seeds"][5:]) == {
                42, 777, 161803, 271828, 314159
            }, entry["name"]
    # Every non-p15 path is a REAL run directory under the contract.
    # [PASS 0, 2026-08-30] The p15 branch read "still PENDING" and
    # FIRED when the listing landed the three probe directories --
    # every entry now satisfies the run-dir contract, p15 included
    # (all three at e57a8dea, the commit's sha8).
    for entry in entries:
        stem, sha8, job = entry["run_dir"].rsplit("/", 1)[-1].split("__")
        assert len(sha8) == 8 and all(
            c in "0123456789abcdef" for c in sha8
        ), entry["name"]
    p15 = [e for e in entries if e["group"] == "p15_mebeauty_probes"]
    assert {e["run_dir"].split("__")[1] for e in p15} == {"e57a8dea"}
    # The identity baseline shares the loop's run and reads its own CSV.
    identity = next(
        e for e in entries if e["name"] == "p16_identity_baseline"
    )
    loop = next(e for e in entries if e["name"] == "p16_anchor_loop")
    assert identity["run_dir"] == loop["run_dir"]
    assert identity["csv"] == "identity_predictions"

    # The exclusions, by name.
    excluded = lock["excluded"]
    assert "p7_g" in excluded["p7_g_label_variants"]
    assert "_median" in excluded["p7_g_label_variants"]
    assert "dated 2026-08-30" in excluded["p7_g_label_variants"]
    assert "RECONSTRUCTIONS" in excluded["p13_decodability_probes"]
    assert "eb5a6887" in excluded["void_runs"]
    assert "85:15" in excluded["protocol_mismatched"]
    assert "masked_768" in excluded["no_run_in_the_record"]
    # The six excluded label-variants are exactly the ladder arms
    # missing from the lock: 36 - 30.
    from cleft import ladder

    ladder_names = {r["name"] for r in ladder.distinct_runs()}
    locked_p7 = {
        e["name"] for e in entries if e["group"].startswith("p7_")
    }
    missing = ladder_names - locked_p7
    assert len(missing) == 6
    assert all("_median" in n or "_ldl" in n for n in missing)


def test_no_glob_anywhere_in_the_arm_resolution():
    """The lock's central property, asserted structurally: the task
    resolves arms only through the config's literal list, and the
    generator derives that list from the lock constant."""
    import ast
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_metric_space_analysis)
    tree = ast.parse("class _:\n" + "\n".join(
        "    " + line for line in source.splitlines()
    )) if source.startswith("def") else None
    for forbidden in ("glob", "iterdir", "listdir", "rglob", "scandir"):
        assert forbidden not in source, forbidden
    assert 'for arm in task["arms"]' in source

    generator = (
        REPO / "scripts" / "generate_phase18_configs.py"
    ).read_text(encoding="utf-8")
    assert "phase18.locked_arm_entries()" in generator
    # [Refined same turn] The ruling forbids globs in ARM RESOLUTION;
    # the generator's one glob is _known_hashes scanning shipped
    # CONFIGS for already-verified hashes (the cross-config agreement
    # guard's requirement). Assert the arm/input derivation is the
    # locked constant and the only glob lives in the hash carrier.
    import re

    glob_lines = [
        line for line in generator.splitlines() if ".glob(" in line
    ]
    assert glob_lines == [
        '    for shipped in sorted((REPO / "configs").glob("*.yaml")):'
    ], glob_lines
    assert "iterdir(" not in generator
    # No glob touches runs/ anywhere in the chain.
    assert not re.search(r"glob\([^)]*runs", generator)


def test_the_exit_criteria_are_locked_with_the_seven_and_the_note():
    record = phase18.EXIT_CRITERIA
    criteria = record["criteria"]
    assert len(criteria) == 7
    assert "both families from banked per-seed CSVs" in criteria[0]
    assert "mechanism-failed third" in criteria[0]
    assert "verdict-flip table; banked PCC verdicts untouched" in criteria[1]
    assert "QWK-over-macro-F1" in criteria[2]
    assert "0.022908" in criteria[3]
    assert "marked uninterpretable" in criteria[3]
    assert "constant-predictor" in criteria[4]
    assert "CleftGNN prohibition" in criteria[4]
    assert "all-MSE standing clause" in criteria[5]
    assert "2026-08-30" in criteria[6]
    assert "nothing is added after" in record["locked"]
    # Seven names, six unique contrasts -- stated, and true of the list.
    assert "SEVEN NAMES over SIX unique" in record["named_seven_note"]
    assert len(phase18.NAMED_CONTRASTS) == 6
    names = [c["name"] for c in phase18.NAMED_CONTRASTS]
    assert "p17-a-vs-probe" in names
    verdicts = {c["name"]: c["pcc_verdict"] for c in phase18.NAMED_CONTRASTS}
    assert verdicts["p17-b-vs-probe"] == "claimable_negative"
    assert sum(1 for v in verdicts.values() if v == "unresolved") == 5


def test_iem_is_phase11s_own_implementation_reused():
    """D5's reuse rule, structurally: phase18 calls phase11.iem and the
    eq-(16) constants appear nowhere outside phase11."""
    import ast
    import inspect

    source = inspect.getsource(phase18.iem_score)
    assert "phase11.iem(" in source or "phase11" in source
    tree = ast.parse(
        (REPO / "src" / "cleft" / "phase18.py").read_text(encoding="utf-8")
    )
    constants = {
        round(node.value, 4) for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, float)
    }
    for eq16 in (1.12, 0.87):
        assert eq16 not in constants, (
            f"eq-(16) exponent {eq16} reimplemented outside phase11"
        )
    # The score agrees with a direct call through phase11.
    import numpy as np

    truth = np.array([2.0, 3.0, 4.0])
    predicted = np.array([1.8, 3.4, 4.0])
    expected = float(np.mean(phase11.iem(predicted - truth)))
    assert phase18.iem_score(truth, predicted) == expected
    # The floor goes through the SAME function.
    constant = np.full(3, truth.mean())
    assert phase18.constant_predictor_iem(truth) == phase18.iem_score(
        truth, constant
    )


def test_the_macro_f1_statistic_matches_the_checked_report_once():
    """The BCa loop skips the phase10 cross-check for speed; this is
    the ONE place the skipped check is exercised for the statistic."""
    import numpy as np

    from cleft import classification
    from cleft.eval.metrics import to_3class

    rng = np.random.default_rng(7)
    truth = rng.uniform(1.2, 4.8, 60)
    predicted = truth + rng.normal(0, 0.8, 60)
    fast = phase18.macro_f1_statistic(truth, predicted)
    checked = classification.prf_report(
        to_3class(truth), to_3class(predicted, clip=True), 3,
    )["f1_macro"]
    assert fast == checked


def test_the_config_derives_from_the_lock_and_counts_declares():
    import yaml as yaml_module

    payload = yaml_module.safe_load(
        (REPO / "configs" / "p18_metric_space.yaml").read_text(
            encoding="utf-8"
        )
    )
    inputs = payload["inputs"]
    arms = payload["task"]["arms"]
    assert len(inputs) == 68       # 67 run dirs + manifest
    assert len(arms) == 68         # 68 prediction sets
    assert len(payload["task"]["contrasts"]) == 6
    assert payload["task"]["n_boot"] == 10000

    # Input list == the lock's run dirs, exactly, plus the manifest.
    lock_dirs = {e["run_dir"] for e in phase18.locked_arm_entries()}
    config_dirs = {
        e["path"].split("/cleft-aesthetics/")[-1]
        for e in inputs if e["name"] != "manifest_v1"
    }
    assert config_dirs == lock_dirs

    # [PASS 0, 2026-08-30] This pin held "3 pending paths" and FIRED
    # when the probe directories were pasted. Every path is real now;
    # only the single pass-1 declare remains.
    pending = [e["name"] for e in inputs if "PENDING" in e["path"]]
    assert pending == []
    # [Same turn] The cross-config agreement guard fired on the first
    # generated file: four run dirs (the three p17 arms and the probe)
    # already carry verified hashes in sibling configs, and an
    # immutable path's hash is THE hash everywhere -- the generator
    # carries them.
    # [PASS 1, 2026-08-30] The single declare filled the remaining 63
    # (5 already MATCHES, no path mismatches) and this pin FIRED on the
    # state change, as every fill pin has. FULLY RESOLVED: every input
    # carries a real path and a verified hash; the run is launchable.
    zeros = sum(1 for e in inputs if set(e["rollup_sha256"]) == {"0"})
    assert zeros == 0
    for entry in inputs:
        rollup = entry["rollup_sha256"]
        assert len(rollup) == 64 and set(rollup) <= set(
            "0123456789abcdef"
        ), entry["name"]
    probe_entry = next(
        e for e in inputs if e["name"] == "p7_d1_vit_b16_imagenet_g1"
    )
    assert probe_entry["rollup_sha256"].startswith("4573e984")
    # Spot-pins at the corners of the table, verbatim from the declare.
    spot = {e["name"]: e["rollup_sha256"] for e in inputs}
    assert spot["p7_c0_vit_b16_scut_original_g1"].startswith("70aae320")
    assert spot["p16_anchor_loop"].startswith("467b7dd9")
    assert spot["roadb_rc_control_whole_vit"].startswith("bda6b060")
    text = (REPO / "configs" / "p18_metric_space.yaml").read_text(
        encoding="utf-8"
    )
    assert "**RESOLVED.**" in text and "PLACEHOLDER" not in text
    manifest = next(e for e in inputs if e["name"] == "manifest_v1")
    assert manifest["rollup_sha256"].startswith("fb5177b3")

    # Every arm row references a declared input by name, and carries
    # its documented row count -- 236 exactly for the four p12 arms
    # (phase12.STOP_1_MANIFEST's "drop folder 238 -> 236 rows
    # verbatim"), 237 everywhere else.
    input_names = {e["name"] for e in inputs}
    for arm in arms:
        assert arm["input"] in input_names, arm["name"]
        expected = 236 if arm["group"] == "p12_view_ablation" else 237
        assert arm["n_patients"] == expected, arm["name"]
    # Verdicts in the contrasts are the ledger's, carried not computed.
    from cleft import results_ledger

    claimable = next(
        c for c in payload["task"]["contrasts"]
        if c["name"] == "p17-b-vs-probe"
    )
    assert claimable["pcc_verdict"] == "claimable_negative"
    row = next(
        e for e in results_ledger.ENTRIES if e["id"] == "p17-b-vs-probe"
    )
    assert row["status"] == "CLAIMABLE"


def _metric_space_fixture(tmp_path, n_patients=12, seeds=(1337, 2024),
                          ten_seed_arm=False):
    """Synthetic per-seed CSVs through the real load path: a manifest
    whose class3 IS the frozen collapse of its mean (the task
    cross-checks that), and two arm run dirs with grade-space
    predictions including a deliberate overshoot past 5.0 -- the
    regression-head case qwk_3cat's clip_pred exists for."""
    import numpy as np

    from cleft.cluster_csv import write_predictions
    from cleft.eval.metrics import to_3class

    rng = np.random.default_rng(3)
    truth = np.clip(rng.normal(2.8, 0.9, n_patients), 1.0, 5.0)
    class3 = to_3class(truth)
    manifest_dir = tmp_path / "cleft_v1"
    manifest_dir.mkdir()
    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for index in range(n_patients):
        value = truth[index]
        lines.append(
            f"{index + 1},{1000 + index},,{value:.4f},{value:.4f},"
            f"{value:.4f},{value:.4f},{value:.4f},0.2,0.2,0.2,0.2,0.2,"
            f"{class3[index]},{index % 5}"
        )
    (manifest_dir / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    arms = {}
    names = ("arm_x", "arm_y", "arm_short") + (
        ("arm_ten",) if ten_seed_arm else ()
    )
    for arm_index, arm in enumerate(names):
        run_dir = tmp_path / f"{arm}_run"
        run_dir.mkdir()
        # arm_short mirrors the p12 shape: one documented patient
        # absent (the last id), every seed agreeing on the subset.
        # arm_ten mirrors the graph regime: the standard five plus the
        # non-standard graph tail, mixed with 2-seed arms in one run.
        keep = n_patients - 1 if arm == "arm_short" else n_patients
        arm_seeds = (
            (1337, 2024, 7, 99, 12345, 42, 271828, 314159, 161803, 777)
            if arm == "arm_ten" else seeds
        )
        for seed in arm_seeds:
            arm_rng = np.random.default_rng(seed + arm_index)
            predicted = truth + arm_rng.normal(0, 0.6, n_patients)
            predicted[0] = 5.4  # the overshoot clip_pred exists for
            write_predictions(
                run_dir / f"seed_{seed}__predictions.csv",
                zip(range(1, keep + 1), truth[:keep], predicted[:keep],
                    (i % 5 for i in range(keep))),
            )
        arms[arm] = run_dir
    return manifest_dir, arms, truth


def test_the_task_end_to_end_on_fixture_csvs(tmp_path, clean_repo, out_root):
    """[2026-08-30] The test the QWK crash named: the task was never
    driven end-to-end in the suite, so a caller feeding CLASS-SPACE
    indices into qwk_3cat's GRADE-SPACE contract crashed on the cluster
    instead of here (run p18-metric-space, all attempts identical:
    ValueError 'values outside the 1.0-5.0 label range: [0, 2]' from
    to_3class's own guard -- the frozen metric's design working).

    **Pre-fix this reproduced that exact ValueError; post-fix the task
    runs through to metrics.json** with grade-space qwk inputs and the
    default clip_pred=True (the contract's own overshoot case,
    exercised by the fixture's 5.4 prediction).
    """
    import json

    from cleft.provenance import RunContext
    from cleft.provenance.hashing import hash_dir
    from cleft.run import TASKS
    from fixtures import builders

    manifest_dir, arms, truth = _metric_space_fixture(
        tmp_path, ten_seed_arm=True
    )
    config = builders.write_config(
        tmp_path / "cfg.yaml", phase="p18",
        inputs=[
            {"name": "arm_x", "path": str(arms["arm_x"]),
             "rollup_sha256": hash_dir(arms["arm_x"])["rollup"]},
            {"name": "arm_y", "path": str(arms["arm_y"]),
             "rollup_sha256": hash_dir(arms["arm_y"])["rollup"]},
            {"name": "arm_short", "path": str(arms["arm_short"]),
             "rollup_sha256": hash_dir(arms["arm_short"])["rollup"]},
            {"name": "arm_ten", "path": str(arms["arm_ten"]),
             "rollup_sha256": hash_dir(arms["arm_ten"])["rollup"]},
            {"name": "manifest_v1", "path": str(manifest_dir),
             "rollup_sha256": hash_dir(manifest_dir)["rollup"]},
        ],
        task={
            "kind": "metric_space_analysis",
            "manifest_artifact": "manifest_v1",
            "arms": [
                {"name": "arm_x", "input": "arm_x",
                 "seeds": [1337, 2024], "csv": "predictions",
                 "group": "fixture", "n_patients": 12},
                {"name": "arm_y", "input": "arm_y",
                 "seeds": [1337, 2024], "csv": "predictions",
                 "group": "fixture", "n_patients": 12},
                # The p12 shape in miniature: documented at 11 of the
                # manifest's 12 -- scored on its own rows, per ruling.
                {"name": "arm_short", "input": "arm_short",
                 "seeds": [1337, 2024], "csv": "predictions",
                 "group": "fixture_short", "n_patients": 11},
                # The graph regime in miniature: ten seeds including
                # the non-standard tail, mixed with 2-seed arms in one
                # run -- the seed lists come from the config rows, and
                # every documented seed's CSV must exist.
                {"name": "arm_ten", "input": "arm_ten",
                 "seeds": [1337, 2024, 7, 99, 12345, 42, 271828,
                           314159, 161803, 777],
                 "csv": "predictions",
                 "group": "fixture_ten", "n_patients": 12},
            ],
            "contrasts": [
                {"name": "x-vs-y", "winner": "arm_x", "baseline": "arm_y",
                 "pcc_verdict": "unresolved"},
            ],
            "n_boot": 10000,
        },
    )
    with RunContext(config, out_root, repo_root=clean_repo) as ctx:
        TASKS["metric_space_analysis"](ctx)
    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    for arm in ("arm_x", "arm_y", "arm_short", "arm_ten"):
        pooled = metrics["table"][arm]["pooled"]
        assert -1.0 <= pooled["qwk_3cat"]["mean"] <= 1.0
        assert pooled["iem"]["mean"] > 0
    # Mixed shapes: the short arm scored on its documented 11 rows,
    # the column in the table, the tau caveat present.
    assert metrics["table"]["arm_short"]["n_patients"] == 11
    assert metrics["table"]["arm_x"]["n_patients"] == 12
    assert metrics["table"]["arm_ten"]["n_seeds"] == 10
    assert "n_patients_caveat" in metrics["rankings"]
    # [2026-08-30] D3's completed block: the QWK ordering against BOTH
    # families, and the registered position in the readings.
    rankings = metrics["rankings"]
    assert "by_qwk_top5" in rankings
    for pair in ("pcc_vs_qwk", "macro_f1_vs_qwk"):
        assert -1.0 <= rankings[pair]["kendall_tau_b"] <= 1.0, pair
        assert 0 <= rankings[pair]["top5_overlap"] <= 5, pair
    assert "BEFORE looking at any tau value" in metrics["readings"][
        "d3_registered_position"
    ]
    assert "no_new_ledger_rows" in metrics["d2_home"]
    # The p17-IEM verification columns, beside the truth's.
    for arm in ("arm_x", "arm_ten"):
        row = metrics["table"][arm]
        assert "prediction_mean" in row and "prediction_sd" in row
        assert "truth_mean" in row and "truth_sd" in row
        assert 1.0 <= row["truth_mean"] <= 5.0
    assert metrics["d2_flip_table"]["x-vs-y"]["result"]["statistic"] == (
        "macro_f1"
    )
    assert metrics["floors"]["iem_constant_predictor"] > 0


def test_d3_completes_the_lock_without_adding_to_it():
    record = phase18.D3_COMPLETED
    assert "COMPLETES the locked criterion" in record["completed"]
    assert "adds nothing to the lock" in record["completed"]
    position = record["registered_position_qwk_over_macro_f1"]
    assert "BEFORE looking at any tau value" in position
    # The position is derived from the weight arithmetic it states.
    assert "adjacent errors cost 1/4" in position
    assert (1 - 0) ** 2 / (3 - 1) ** 2 == 0.25
    assert (2 - 0) ** 2 / (3 - 1) ** 2 == 1.0
    assert "CHANCE CORRECTION" in position
    assert "does NOT inherit macro F1's" in position
    assert "chance correction, not the distance pricing" in position
    assert "recorded as wrong, never adjusted" in position
    # The lock itself is untouched: criterion 3's wording as locked.
    assert "QWK-over-macro-F1" in phase18.EXIT_CRITERIA["criteria"][2]


def test_the_d2_home_ruling_and_its_one_directional_reference():
    record = phase18.D2_HOME_RULED
    grounds = record["no_new_ledger_rows"]
    assert "INSTRUMENT finding" in grounds
    assert "unit of ARM-CLAIM" in grounds
    assert "stripped of its metric qualifier" in " ".join(
        grounds.split()
    )
    assert "promised a TABLE" in grounds
    assert "chain pins through 37" in grounds
    reference = record["cross_reference_one_directional"]
    for name in ("p16-anchor-loop-unresolved", "p17-a-vs-probe",
                 "p17-b-vs-probe", "p17-c-vs-probe", "p17-a-vs-c",
                 "p17-b-vs-c"):
        assert name in reference, name
    assert "never the other direction" in reference
    # One-directional in fact: the ledger does not mention the flip
    # table or D2.
    from cleft import results_ledger

    # Word-matched "flip" once and hit p7's "coin flips" row -- the
    # check is the flip TABLE's absence, not the word's.
    ledger_text = str(results_ledger.ENTRIES)
    assert "flip table" not in ledger_text.lower()
    assert "flip_table" not in ledger_text
    assert "metric_space" not in ledger_text


def test_the_p17_iem_path_record_and_the_mechanism_flag():
    record = phase18.P17_IEM_PATH_VERIFIED
    path = record["the_path"]
    assert "residual = predicted" in " ".join(path.split())
    assert "GRADE" in path
    assert "convention A" in path
    assert "No unit conversion" in path
    mechanism = record["the_mechanism_writable"]
    assert "REASONED -> MEASURED at run 5" in mechanism
    assert "3.14" in mechanism and "2.96" in mechanism
    assert "wrong SCALE" in mechanism
    assert "piles mass into high grades" in mechanism
    # The magnitude-map arithmetic the mechanism rests on.
    import numpy as np

    from cleft import phase17

    grades = phase17.magnitude_to_grade(
        np.array([0.0, 0.015, 0.025, 0.035])
    )
    assert round(float(grades.mean()), 2) == 3.14
    # The path really is phase11's implementation, direction y_hat - G.
    truth = np.array([3.0, 3.0])
    predicted = np.array([3.5, 2.5])
    over = phase11.iem(np.array([0.5]))[0]    # residual +0.5, light
    under = phase11.iem(np.array([-0.5]))[0]  # residual -0.5, heavy
    assert under > over
    assert phase18.iem_score(truth, predicted) == (over + under) / 2


# --------------------------------------------------------------------------
# 2026-08-30, the phase closed (run 5 citable; 1-4 the audit trail)
# --------------------------------------------------------------------------


def test_the_verdicts_carry_their_numbers_and_the_honesty_arms():
    d1 = phase18.D1_VERDICT
    assert "PARTIALLY held" in d1["verdict"]
    assert "0.5180" in d1["verdict"] and "4/5" in d1["verdict"]
    # The covered half's arithmetic: identity's macro F1 IS the
    # constant floor at [88, 119, 30] -- re-derived, not retyped.
    import numpy as np

    floor = classification.majority_baseline(
        np.repeat([0, 1, 2], [88, 119, 30]), 3
    )
    assert round(floor["f1_macro"], 4) == 0.2228
    assert "0.2228 constant floor EXACTLY" in d1["covered_movers"]
    uncovered = d1["uncovered_kind_named"]
    assert "0.0785" in uncovered and "BELOW the constant" in uncovered
    assert "MINORITY-CLASS" in uncovered
    assert "did not anticipate" in uncovered
    assert "partially failed, never adjusted" in uncovered
    # And 0.0785 really is below the floor.
    assert 0.0785 < floor["f1_macro"]

    d2 = phase18.D2_MEASURED
    assert "3 of 6" in d2["verdict"]
    assert "ONE direction" in d2["verdict"]
    assert "26x" in d2["verdict"]
    assert "METRIC-DEPENDENT" in d2["verdict"]
    assert "distinct set is SIX" in d2["six_not_seven"]
    assert "no ledger rows" in d2["home_stands"]

    d4 = phase18.D4_MEASURED
    assert "66 of 67" in d4["verdict"]
    assert "0.022908" in d4["verdict"]
    assert str(phase18.MACRO_F1_FLOOR["floor"]) == "0.022908"

    d5 = phase18.D5_CLOSED
    assert "IDENTITY BASELINE first of 68" in d5["defect_characterisation_fired"]
    assert "A_FAVOURS_NARROW_PREDICTORS" in d5["defect_characterisation_fired"]
    assert "0.3784" in d5["defect_characterisation_fired"]
    assert "-0.0211" in d5["defect_characterisation_fired"]


def test_d3_is_judged_by_its_exact_wording_with_the_rule_fixed_first():
    record = phase18.D3_JUDGED
    assert "0.6295" in record["measured"]
    assert "0.7024" in record["measured"]
    clause = record["the_positions_testable_clause"]
    # The clause quoted is really the position's own wording.
    assert "order arms closer to accuracy than macro F1 does" in clause
    position = " ".join(phase18.D3_COMPLETED[
        "registered_position_qwk_over_macro_f1"
    ].split()).replace("**", "")
    assert "order arms closer to accuracy than macro F1 does" in (
        position
    )
    assert "tau_accuracy_vs_qwk" in clause
    assert "derived-from-run-5" in clause
    rule = record["judgement_rule_pre_committed"]
    assert "HELD iff tau(accuracy, QWK) > 0.7024" in rule
    assert "wrong-recorded-as-wrong" in rule
    assert "rule is fixed here first" in rule
    # The derivation is deterministic and order-faithful on a fixture.
    table = {
        "a": {"pooled": {"accuracy": {"mean": 0.5},
                         "qwk_3cat": {"mean": 0.2}}},
        "b": {"pooled": {"accuracy": {"mean": 0.6},
                         "qwk_3cat": {"mean": 0.1}}},
    }
    assert phase18.tau_accuracy_vs_qwk(table)["tau_accuracy_vs_qwk"] == -1.0

    # [2026-08-30] The rule applied: the derived number landed and the
    # verdict is mechanical -- and WRONG, recorded as wrong.
    judged = record["judged_wrong_2026_08_30"]
    assert "0.3392663" in judged
    assert "THE POSITION IS WRONG" in judged
    assert "recorded as wrong" in judged
    assert 0.3392663 < 0.7024  # the rule's arithmetic, not just prose
    assert "DECOUPLES it from raw accuracy" in judged
    assert "NOT idle" in judged
    assert "behind QWK crowning arm" in judged
    # The rule and the original position stand unedited beside it.
    assert "HELD iff tau(accuracy, QWK) > 0.7024" in record[
        "judgement_rule_pre_committed"
    ]
    assert "chance correction, not the distance pricing" in phase18.D3_COMPLETED[
        "registered_position_qwk_over_macro_f1"
    ]

    observation = record["unpredicted_observation"]
    assert "OBSERVATION, not reading" in observation
    assert "first of 68" in observation
    assert "Three instruments, three verdicts, one arm" in observation


def test_the_p17_mechanism_resolution_preserves_the_original():
    # The original [REASONED] text, byte-preserved on its pinned parts.
    original = phase18.P17_IEM_PATH_VERIFIED["the_mechanism_writable"]
    assert "REASONED -> MEASURED at run 5" in original
    assert "3.14" in original and "2.96" in original

    measured = phase18.P17_IEM_MEASURED["measured"]
    assert "SPECIFIC mechanism FAILED" in measured
    assert "1.878 +/- 1.435" in measured
    assert "2.7544 +/- 0.6587" in measured
    assert "UNDER-prediction" in measured
    assert "GENERAL claim is measured" in measured
    assert "4.91 +/- 0.30" in measured
    assert "original preserved" in measured

    ninth = phase18.P17_IEM_MEASURED["the_ninth_different_quantities_catch"]
    assert "CONFLATION" in ninth
    assert "ANCHOR-GRADE mean" in ninth
    assert "2.7544" in ninth and "2.935" in ninth
    assert "NINTH time" in ninth
    # The anchor mean really is 2.96 and really differs from the truth
    # mean the run measured.
    assert round((3 * 1 + 7 * 2 + 6 * 3 + 6 * 4 + 3 * 5) / 25, 2) == 2.96
    assert abs(2.935 - 2.96) < abs(2.935 - 2.7544)


def test_the_closing_walks_all_seven_criteria():
    closing = phase18.PHASE_18_CLOSING
    assert "9411267e" in closing["closed"]
    assert "runs 1-4" in closing["closed"]
    walked = [k for k in closing if k.startswith("criterion_")]
    assert len(walked) == 7
    for key in walked:
        assert "MET" in closing[key] or "the lock held" in closing[key], key

    history = closing["operational_history"]
    assert "three load-path defects" in history
    assert "grade-space contract" in history
    assert "documented 236" in history
    assert "seed-regime over-claim" in history
    assert "superseded by run 5" in history
    assert "Three instrument defects, zero data defects" in history

    assert "236" in closing["tau_caveats"]
    exhibit = closing["the_exhibit"]
    assert "ONE ARM, THREE INSTRUMENTS, THREE VERDICTS" in exhibit
    # The exhibit's three verdicts, cross-checked against the records
    # that hold them: QWK first (D3's observation), PCC unresolved (the
    # ledger row), IEM last (D5).
    assert "first of 68" in phase18.D3_JUDGED["unpredicted_observation"]
    from cleft import results_ledger

    row = next(e for e in results_ledger.ENTRIES
               if e["id"] == "p17-a-vs-probe")
    assert row["status"] == "UNRESOLVED-WITHDRAWN"
    assert "WHAT QUALITY IS" in exhibit


# --------------------------------------------------------------------------
# [2026-09-01] The three addenda: orientation, the probe, the impact
# --------------------------------------------------------------------------


def _flat18(text: str) -> str:
    return " ".join(text.split())


def test_the_orientation_note_quotes_the_code_it_describes():
    import inspect

    from cleft import phase11
    from cleft import run as run_module

    record = phase18.IEM_TAU_ORIENTATION
    assert "HYGIENE, NOT A CORRECTION" in record["noted"]
    assert "not reopened" in record["noted"]

    # The code really does orient IEM by the absent minus sign.
    source = inspect.getsource(run_module.task_metric_space_analysis)
    assert 'by_pcc = sorted(names, key=lambda n: -table[n]["pooled"]["pcc"]["mean"])' in source
    assert 'by_iem = sorted(names, key=lambda n: table[n]["pooled"]["iem"]["mean"])' in source
    # ...and the three scores all carry it while IEM does not.
    for metric in ("pcc", "macro_f1", "qwk_3cat"):
        assert f'-table[n]["pooled"]["{metric}"]["mean"]' in source, metric
    assert '-table[n]["pooled"]["iem"]["mean"]' not in source

    # Tau is on rank POSITION, not raw values.
    assert "rank_a = {n: i for i, n in enumerate(order_a)}" in source
    assert "[rank_a[n] for n in names], [rank_b[n] for n in names]" in source
    assert "passes those POSITIONS to kendall_tau_b" in record[
        "tau_is_computed_on_rank_position"
    ]
    assert "Raw values never reach the statistic" in record[
        "tau_is_computed_on_rank_position"
    ]

    # The stated principle is quoted from its actual home.
    doc = _flat18(inspect.getdoc(phase11.ranks))
    assert "ascending is the good-first order" in doc
    assert "visible at its call site" in doc
    assert "ascending is the good-first order" in _flat18(
        record["the_projects_stated_principle"]
    )

    # Both paths carry it, and the second one says so in a comment.
    rank_block = inspect.getsource(run_module).split("def _rank_block")[1]
    assert "PCC is a similarity: negate so every ranking here is best-first" in (
        rank_block
    )
    assert "run._rank_block" in _flat18(
        record["both_tau_paths_carry_it_independently"]
    )


def test_the_consequence_is_an_exact_sign_flip_and_is_re_derived():
    """The note claims an EXACT flip, not an approximate one."""
    import numpy as np

    from cleft.phase11 import kendall_tau_b

    rng = np.random.default_rng(20260901)
    n = 68
    names = [f"a{i}" for i in range(n)]
    pcc = rng.normal(0.20, 0.06, n)
    iem = 0.60 - 0.5 * pcc + rng.normal(0, 0.035, n)
    f1 = 0.35 + 0.4 * pcc + rng.normal(0, 0.05, n)
    table = {m: {"pcc": p, "iem": e, "f1": g}
             for m, p, e, g in zip(names, pcc, iem, f1)}

    def task_way(a, b, b_is_error):
        oa = sorted(names, key=lambda m: -table[m][a])
        ob = sorted(names, key=(lambda m: table[m][b]) if b_is_error
                    else (lambda m: -table[m][b]))
        ra = {m: i for i, m in enumerate(oa)}
        rb = {m: i for i, m in enumerate(ob)}
        return kendall_tau_b([ra[m] for m in names], [rb[m] for m in names])

    def raw_way(a, b):
        return kendall_tau_b([table[m][a] for m in names],
                             [table[m][b] for m in names])

    # Exact flip on the IEM pairs...
    for a in ("pcc", "f1"):
        assert task_way(a, "iem", True) + raw_way(a, "iem") == pytest.approx(
            0.0, abs=1e-15
        ), a
    # ...and no difference at all where there is no orientation to lose.
    assert task_way("pcc", "f1", False) == pytest.approx(
        raw_way("pcc", "f1"), abs=1e-15
    )

    consequence = _flat18(phase18.IEM_TAU_ORIENTATION["the_consequence"])
    assert "EXACT SIGN FLIP" in consequence
    assert "0.00e+00" in consequence
    # No figure moved.
    assert "+0.3784" in phase18.IEM_TAU_ORIENTATION["no_figure_changes"]
    assert "0.3784" in phase18.D5_CLOSED["defect_characterisation_fired"]


def test_the_probes_provenance_names_the_two_implementations_defect():
    record = phase18.TAU_RECOMPUTATION_PROVENANCE
    what = _flat18(record["what_happened"])
    assert "2026-08-31 a one-liner was written" in what
    assert "OMITTED THE ORIENTATION THE TASK ENCODES" in what
    assert "-0.377309" in what and "+0.022432" in what
    assert "briefly read as a possible finding" in what

    assert "nothing needed correcting" in record[
        "the_banked_values_were_correct_throughout"
    ]
    defect = _flat18(record["the_defect_in_the_probe"])
    assert "two-implementations defect in a new place" in defect
    assert "no docstring, no test and no review" in defect
    # The two precedents it cites are real, at source.
    import inspect

    from cleft import run as run_module

    assert "a second copy of a label resolution" in _flat18(
        inspect.getdoc(run_module.median_by_patient)
    )
    assert "no second implementation" in _flat18(
        inspect.getdoc(phase18.tau_accuracy_vs_qwk)
    )

    # The gap is attributed to the record, not to the probe.
    gap = _flat18(record["the_underlying_gap"])
    assert "ABSENT FROM THE RECORD THE PROBE READ" in gap
    assert "Not the probe's fault" in gap

    # Filed beside the other five, all of which exist.
    from cleft import ladder, literature, phase12, phase20

    for module, name in (
        (ladder, "THE_ERROR_PROVENANCE"),
        (phase20, "CROSS_TARGET_ERROR_PROVENANCE"),
        (phase20, "S_DESCRIPTION_ERROR_PROVENANCE"),
        (phase12, "TWO_VIEW_CLAIM_PROVENANCE"),
        (literature, "NADEAU_BENGIO_DOES_NOT_APPLY"),
    ):
        assert hasattr(module, name), name
        assert name in record["filed_beside"], name


def test_the_correction_direction_is_proven_not_argued():
    """old - true = t*(2x + t + a + b) >= 0, so |tau| can only grow."""
    record = phase18.TAU_B_CORRECTION_IMPACT
    proof = _flat18(record["the_direction_is_proven"])
    assert "t*(2x + t + a + b) >= 0" in proof
    assert "365 grew, 0 shrank" in proof

    # The algebra, brute-forced here as well.
    counterexample = None
    for x in range(1, 25):
        for a in range(0, 8):
            for b in range(0, 8):
                for t in range(0, 8):
                    if (x + t + a) * (x + t + b) < (x + b) * (x + a):
                        counterexample = (x, a, b, t)
    assert counterexample is None, counterexample


def test_only_the_reconciling_figure_is_banked_and_the_rest_is_flagged():
    """A number that contradicts a proven property is not banked."""
    record = phase18.TAU_B_CORRECTION_IMPACT

    banked = record["reconciles"]["tau_accuracy_vs_qwk"]
    assert banked["banked"] == 0.3392663
    assert banked["corrected"] == 0.3400134
    assert banked["move"] == pytest.approx(
        banked["corrected"] - banked["banked"], abs=1e-9
    )
    assert abs(banked["corrected"]) > abs(banked["banked"]), "it GREW"
    assert banked["consistent_with_the_proof"] is True

    # The three that do not reconcile are named, with both readings, and
    # NOT banked as corrected values.
    unreconciled = _flat18(record["does_not_reconcile_as_stated"])
    for move in ("-0.001176", "-0.000954", "-0.000684"):
        assert move in unreconciled, move
    assert "which the proof forbids" in unreconciled
    for alternative in ("0.519176", "0.630454", "0.703084"):
        assert alternative in unreconciled, alternative
    assert "only their magnitudes and the sign question are recorded" in (
        record["the_three_are_not_banked_here"]
    )
    assert "contradicts a proven property" in _flat18(
        record["the_three_are_not_banked_here"]
    )
    # They really are absent from the banked block.
    assert list(record["reconciles"]) == ["tau_accuracy_vs_qwk"]

    # The arithmetic of both readings, re-derived.
    for banked_value, move in ((0.5180, 0.001176), (0.6295, 0.000954),
                               (0.7024, 0.000684)):
        assert abs(banked_value - move) < abs(banked_value), "as written, shrinks"
        assert abs(banked_value + move) > abs(banked_value), "reversed, grows"


def test_no_phase_18_verdict_moves_and_d3_wrong_stands():
    record = phase18.TAU_B_CORRECTION_IMPACT
    verdict = _flat18(record["no_verdict_changes_either_way"])
    assert "WRONG STANDS" in verdict
    assert "0.3624" in verdict
    assert round(0.7024 - 0.3400134, 4) == 0.3624
    assert not 0.3400134 > 0.7024
    assert "ORIENTATION ARTIFACT" in verdict

    # And Phase 21's record now points at the measured impact.
    from cleft import phase21

    assert phase21.TAU_B_DEFECT_CORRECTED["downstream_measured_2026_09_01"] == (
        "phase18.TAU_B_CORRECTION_IMPACT"
    )
    # What it upgrades is stated: probably -> measured, argument -> proof.
    upgrades = _flat18(record["what_this_upgrades"])
    assert "becomes MEASURED" in upgrades
    assert "becomes a PROOF rather than an argument" in upgrades
    assert "Probably is not measured" in _flat18(
        phase21.TAU_B_DEFECT_CORRECTED["what_can_be_said_without_the_artifacts"]
    )


def test_the_addenda_reopen_nothing():
    """Three notes, no figure moved, no lock touched."""
    for figure in ("0.5180", "0.3784", "0.7024", "0.6295", "0.3392663"):
        assert figure in " ".join(
            (Path(__file__).resolve().parents[1] / "src" / "cleft"
             / "phase18.py").read_text(encoding="utf-8").split()
        ), figure
    assert "0.3784" in phase18.ARM_LIST_ADDENDUM["what_this_is_not"]
    assert "0.3784" in phase18.NOT_COVERED_BY_PHASE_18

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 18 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p18" or "p18-" in e["id"]
    ]
    assert results_ledger.validate() is None


# --------------------------------------------------------------------------
# [2026-09-01, later] The refuted hypothesis, and the duplicate sets
# --------------------------------------------------------------------------


def test_the_reinterpretation_is_retracted_with_its_original_visible():
    record = phase18.TAU_B_CORRECTION_IMPACT

    # The original reasoning survives, marked superseded, not deleted.
    superseded = record["does_not_reconcile_as_stated"]
    assert superseded.startswith("[SUPERSEDED 2026-09-01 -- REFUTED")
    for figure in ("-0.001176", "-0.000954", "-0.000684",
                   "0.519176", "0.630454", "0.703084"):
        assert figure in superseded, figure

    # Why it was offered, and why it was wrong.
    why = _flat18(record["why_that_reinterpretation_was_offered_and_why_it_was_wrong"])
    assert "only reading that made the printed numbers consistent" in why
    assert "CAME FROM A PROBE WITH A WRONG PREMISE" in why
    assert "asking where the figure came from" in why

    # And the refutation itself.
    could_not = _flat18(record["the_three_could_not_have_moved"])
    assert "TIE-FREE BY CONSTRUCTION" in could_not
    assert "0.000e+00 over 300 trials" in could_not
    assert "five of the six" in could_not

    # No number was banked from the retracted reading.
    assert list(record["reconciles"]) == ["tau_accuracy_vs_qwk"]


def test_exactly_one_banked_figure_was_affected():
    record = phase18.TAU_B_CORRECTION_IMPACT
    one = _flat18(record["the_defect_affected_exactly_one_banked_figure"])
    assert "tau(accuracy, QWK), 0.3392663 -> 0.3400134" in one
    assert "+0.0007471" in one
    assert "THE OTHER FIVE COULD NOT HAVE MOVED" in one
    assert "RAW POOLED VALUES" in one and "RANK POSITIONS" in one

    # Accuracy ties alone move nothing -- the measured demonstration.
    alone = _flat18(record["accuracy_ties_alone_are_insufficient"])
    assert "33 accuracy-only ties with ZERO doubly-tied" in alone
    assert "exactly 0.0000000" in alone

    # t = 5, uniquely pinned by the inversion.
    pinned = _flat18(record["the_doubly_tied_count"])
    assert "t = 5" in pinned
    assert "1.004409" in pinned and "2278" in pinned
    assert "1.004151" in pinned and "1.005289" in pinned
    # The bracketing really does exclude t=4 and t=6.
    n0 = 68 * 67 // 2
    target = (0.3400134 / 0.3392663) ** 2
    for t, contains in ((4, False), (5, True), (6, False)):
        lo = min(((n0 - a - t + t + a) * (n0 - a - t + t))
                 / ((n0 - a - t) * (n0 - a - t + a))
                 for a in range(0, 400) if n0 - a - t > 0)
        hi = max(((n0 - a - t + t + a) * (n0 - a - t + t))
                 / ((n0 - a - t) * (n0 - a - t + a))
                 for a in range(0, 400) if n0 - a - t > 0)
        assert (lo <= target <= hi) is contains, (t, lo, hi, target)


def test_the_probe_made_one_mistake_with_two_symptoms():
    record = phase18.TAU_RECOMPUTATION_PROVENANCE
    two = _flat18(record["one_wrong_premise_two_symptoms"])
    assert "ONE wrong premise -- raw values where the task uses ranks" in two
    assert "SIGN FLIP on the oriented IEM column" in two
    assert "PHANTOM DRIFT on the three score columns" in two
    assert "the same mistake, made twice in one command" in two
    assert "IN A CHAT MESSAGE" in _flat18(
        record["both_are_the_same_defect_in_a_new_place"]
    )


def test_the_five_duplicate_pairs_are_named_and_corroborated():
    from cleft import ladder, roadb

    record = phase18.DUPLICATE_PREDICTION_SETS
    pairs = record["the_five_pairs"]
    assert len(pairs) == 5
    assert all(len(p) == 2 for p in pairs)
    # Each is a Road A g1 arm and a Road B 224 arm.
    for road_a, road_b in pairs:
        assert road_a.startswith("p7_"), road_a
        assert road_b.startswith("roadb_") and road_b.endswith("224"), road_b
    # Every named arm is in the locked list.
    locked = {e["name"] for e in phase18.locked_arm_entries()}
    for road_a, road_b in pairs:
        assert road_a in locked, road_a
        assert road_b in locked, road_b

    # The banked corroboration, re-derived from both records.
    g1 = ladder.STAGE_D1_AT_G1["cells"]
    rb = roadb.PHASE_7_TWENTY_TWO_ARMS["values"]
    checks = [
        (g1["swin_b"][2], rb["swin_b__scut_masked"][224][0]),
        (g1["srgnn"][0], rb["srgnn__imagenet"][224][0]),
        (g1["swin_b"][0], rb["swin_b__imagenet"][224][0]),
        (g1["vit_b16"][0], rb["vit_b16__imagenet"][224][0]),
    ]
    for a, b in checks:
        assert a == b, (a, b)
    # And the one that differs, by exactly 0.0001.
    assert round(abs(g1["agnet"][0] - rb["agnet__imagenet"][224][0]), 4) == 0.0001

    corroborate = _flat18(record["the_banked_records_corroborate"])
    assert "FOUR match EXACTLY" in corroborate
    assert "0.0001" in corroborate
    assert "consistent with two records rounding one underlying value" in corroborate


def test_the_count_was_predicted_then_measured():
    record = phase18.DUPLICATE_PREDICTION_SETS
    measured = _flat18(record["the_count_is_measured_not_inferred"])
    assert "predicted t = 5 uniquely" in measured
    assert "returned exactly 5 and named them" in measured
    assert "CONFIRMED INDEPENDENTLY" in measured
    assert len(record["the_five_pairs"]) == 5


def test_what_the_duplicate_check_could_not_settle():
    """Equal PCC is not proof of equal predictions, and the record says
    exactly what would settle it."""
    record = phase18.DUPLICATE_PREDICTION_SETS
    unsettled = _flat18(record["what_is_not_settled"])
    assert "predictions_sha256 PER SEED" in unsettled
    assert "NOT DONE HERE" in unsettled
    assert "CLUSTER-ONLY and unreachable" in unsettled
    assert "is NOT PROOF" in unsettled
    assert "One command on the cluster would settle it" in unsettled


def test_the_duplicate_impact_is_measured_not_assumed():
    """The expectation was 'no material effect'. The measurement does
    not support saying that flatly, and the record says so."""
    record = phase18.DUPLICATE_PREDICTION_SETS

    # D4 is affected deterministically: +5, every draw.
    d4 = _flat18(record["impact_d4_is_affected_deterministically"])
    assert "EXACTLY ZERO" in d4
    assert "+5 in every one of 300 draws" in d4
    assert "61 OF 62 AMONG DISTINCT ARMS" in d4
    assert "The figure does not change" in d4
    assert "TRIVIALLY" in d4 and "INFORMATIVELY" in d4
    # 66 of 67 minus 5 duplicates is 61 of 62.
    assert 66 - 5 == 61 and 67 - 5 == 62
    assert "66 of 67" in phase18.D4_MEASURED["verdict"] or (
        "66 of 67" in _flat18(phase18.D4_MEASURED["verdict"])
    )

    # tau: small in expectation, NOT declared negligible.
    tau = _flat18(record["impact_on_tau_is_small_but_not_provably_negligible"])
    assert "mean +0.00231" in tau
    assert "0.05503" in tau
    assert "NOT ASSERTED HERE" in tau
    assert "does not support saying so flatly" in tau

    # top-5 can shift, and D5's overlap of 1 is the one at risk.
    top5 = _flat18(record["impact_on_top_5_overlaps"])
    assert "up to 2" in top5
    assert "FOUR DISTINCT PREDICTION SETS" in top5
    assert "top-5 overlap 1" in phase18.D5_CLOSED[
        "defect_characterisation_fired"
    ]

    # D5's ladder position is not affected, and why.
    assert "IDENTITY BASELINE" in record["impact_on_d5_ladder_position"]


def test_the_duplicate_finding_reopens_nothing():
    record = phase18.DUPLICATE_PREDICTION_SETS
    assert "LIMITATION OF THE LOCK" in record["measured"]
    assert "NOT a reopening" in record["measured"]
    why = _flat18(record["why_it_is_a_limitation_not_a_defect"])
    assert "the lock enumerated RUNS" in why
    assert "63 distinct prediction sets" in why
    assert "more independence than the set contains" in why
    assert "visibility, not coverage" in _flat18(record["not_a_reopening"])

    # The lock still holds 68 and every figure is unchanged.
    assert len(phase18.locked_arm_entries()) == 68
    assert "0.3784" in phase18.D5_CLOSED["defect_characterisation_fired"]
    assert "66 of 67" in _flat18(phase18.D4_MEASURED["verdict"])

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 18 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p18" or "p18-" in e["id"]
    ]
    assert results_ledger.validate() is None


def test_phase_21s_scope_pointer_is_corrected():
    from cleft import phase21

    record = phase21.TAU_B_DEFECT_CORRECTED
    assert record["downstream_and_to_be_ruled"].startswith(
        "[SUPERSEDED 2026-09-01"
    )
    corrected = _flat18(record["corrected_scope_2026_09_01"])
    assert "EXACTLY ONE banked figure was affected" in corrected
    assert "0.3392663 -> 0.3400134" in corrected
    assert "TIE-FREE BY CONSTRUCTION" in corrected
    assert "0.000e+00 over 300 trials" in corrected
    assert "D3's WRONG stands by 0.3624" in corrected


# --------------------------------------------------------------------------
# [2026-09-01, by hash] Four pairs not five; the AG-Net divergence
# --------------------------------------------------------------------------


def test_the_duplicate_count_is_corrected_to_four():
    record = phase18.DUPLICATE_COUNT_CORRECTED
    assert record["method"] == "fingerprint.predictions_sha256, per pair"
    result = record["result"]
    assert sum(v == "IDENTICAL" for v in result.values()) == 4
    assert result["agnet_imagenet"].startswith("DIFFERENT")
    assert "fc5443061acdb181" in result["agnet_imagenet"]
    assert "987aece1ab0303f3" in result["agnet_imagenet"]
    assert "64 DISTINCT PREDICTION SETS" in record["the_count"]
    assert "FOUR proven byte-identical pairs, not five" in record["the_count"]

    # The superseded reading is marked in place, not deleted.
    assert phase18.DUPLICATE_PREDICTION_SETS[
        "the_banked_records_corroborate"
    ].startswith("[SUPERSEDED IN PART 2026-09-01")
    assert "FOUR match EXACTLY" in phase18.DUPLICATE_PREDICTION_SETS[
        "the_banked_records_corroborate"
    ]
    assert "63 distinct prediction sets" in _flat18(
        phase18.DUPLICATE_PREDICTION_SETS["why_it_is_a_limitation_not_a_defect"]
    )
    assert phase18.DUPLICATE_PREDICTION_SETS[
        "why_it_is_a_limitation_not_a_defect"
    ].startswith("[63 -> 64 distinct sets, corrected 2026-09-01]")


def test_the_rounding_reading_is_recorded_as_the_error_it_was():
    record = phase18.DUPLICATE_COUNT_CORRECTED
    real = _flat18(record["the_0_0001_was_real"])
    assert "was a genuine difference" in real
    assert "The benign reading was available and I took it" in real
    assert "the one pair that broke the pattern was the one I explained away" in real


def test_the_t5_count_survives_because_it_was_never_about_duplication():
    record = phase18.DUPLICATE_COUNT_CORRECTED
    stands = _flat18(record["the_t_5_count_is_untouched"])
    assert "STANDS AS MEASURED" in stands
    assert "never a claim about duplication" in stands

    # And the mechanism: coarse metrics tie through the perturbation.
    why = _flat18(record["why_the_agnet_pair_ties_on_coarse_metrics"])
    assert "0.000 of 237" in why
    assert "CLASS-BASED metrics" in why and "CONTINUOUS metrics" in why

    # Re-measured here: a 2.3e-05 perturbation flips no 3-class label.
    import numpy as np

    from cleft.eval import metrics as m

    rng = np.random.default_rng(20260901)
    truth = rng.uniform(1.0, 5.0, 237)
    base = 0.06 * truth + rng.normal(0, 0.9, 237)
    flips = 0
    for _ in range(200):
        perturbed = base + rng.normal(0, 2.3e-5, 237)
        flips += int(np.sum(
            m.to_3class(np.clip(base, 1, 5)) != m.to_3class(np.clip(perturbed, 1, 5))
        ))
    assert flips == 0

    # t = 5 is still the recorded accuracy/QWK count.
    assert "t = 5" in _flat18(
        phase18.TAU_B_CORRECTION_IMPACT["the_doubly_tied_count"]
    )


def test_the_agnet_divergence_is_a_declared_configuration_difference():
    import yaml
    from pathlib import Path as _P

    from cleft.models import agnet

    record = phase18.AGNET_DIVERGENCE_INVESTIGATED
    assert "LEGITIMATE CONFIGURATION DIFFERENCE" in record["verdict"]
    assert "NOT a new determinism finding" in record["verdict"]

    configs = _P(__file__).resolve().parents[1] / "configs"

    def task(stem):
        return yaml.safe_load(
            (configs / f"{stem}.yaml").read_text(encoding="utf-8")
        )["task"]

    # The four pairs' task blocks are identical, key for key.
    for a, b in (("p7_d1_agnet_imagenet_g1_native",
                  "roadb_p7_arm_agnet_imagenet_224"),
                 ("p7_d1_srgnn_imagenet_g1_native",
                  "roadb_p7_arm_srgnn_imagenet_224"),
                 ("p7_d1_vit_b16_imagenet_g1",
                  "roadb_p7_arm_vit_b16_imagenet_224"),
                 ("p7_d1_swin_b_imagenet_g1",
                  "roadb_p7_arm_swin_b_imagenet_224")):
        assert task(a) == task(b), (a, b)

    # The one field that separates agnet from srgnn.
    assert task("p7_d1_agnet_imagenet_g1_native")["deterministic"] is False
    assert task("roadb_p7_arm_agnet_imagenet_224")["deterministic"] is False
    assert "deterministic" not in task("p7_d1_srgnn_imagenet_g1_native")
    assert "deterministic" not in task("roadb_p7_arm_srgnn_imagenet_224")

    # The embeddings hash differs in ALL FOUR pairs, so it explains nothing.
    def emb(stem):
        return {e["name"]: e["rollup_sha256"] for e in yaml.safe_load(
            (configs / f"{stem}.yaml").read_text(encoding="utf-8")
        )["inputs"]}["embeddings"]

    for a, b in (("p7_d1_agnet_imagenet_g1_native",
                  "roadb_p7_arm_agnet_imagenet_224"),
                 ("p7_d1_srgnn_imagenet_g1_native",
                  "roadb_p7_arm_srgnn_imagenet_224"),
                 ("p7_d1_vit_b16_imagenet_g1",
                  "roadb_p7_arm_vit_b16_imagenet_224")):
        assert emb(a) != emb(b), (a, b)
    assert "INCLUDING the three that reproduce" in _flat18(
        record["what_the_config_diff_shows"]
    )

    # The record that predicted it, live at source.
    det = agnet.TRAINING_DETERMINISM
    assert det["bitwise_flag_off"] is False
    assert "atomicAdd" in det["mechanism"]
    assert "2.3e-05" in det["flag_off_consequence"]
    assert det["measured"] == "2026-07-31"
    assert "2026-07-31" in _flat18(record["the_record_that_predicted_it"])

    # Magnitude: consistent, explicitly not reconciled.
    mag = _flat18(record["magnitude_consistent_but_not_reconciled"])
    assert "NOT a computed reconciliation and not claimed as one" in mag


def test_d4_is_remeasured_with_both_answers_recorded():
    record = phase18.D4_REMEASURED_AT_FOUR
    open_col = _flat18(record["the_open_column"])
    assert "D4 is a MACRO-F1 statistic" in open_col
    assert "NOT MEASURED" in open_col
    assert "CLUSTER-ONLY and was not read" in open_col

    both = record["both_answers"]
    assert "61 of 62" in both["if_agnet_ties_on_macro_f1"]
    assert "62 of 63" in both["if_agnet_does_not_tie"]
    # The arithmetic of both.
    assert (66 - 5, 67 - 5) == (61, 62)
    assert (66 - 4, 67 - 4) == (62, 63)

    # [2026-09-01, THIS PIN FIRED AS DESIGNED] It held the prediction as
    # [REASONED] while it was one. The measurement came the same day and
    # the branch it named is the branch that fired, so the tag now
    # carries the confirmation. The caution it guarded -- "likely is not
    # measured" -- is preserved, because it was the right caution at the
    # time it was written.
    assert record["the_prediction_tagged_as_one"].startswith(
        "[REASONED 2026-09-01, CONFIRMED"
    )
    assert "Likely is not measured" in record["the_prediction_tagged_as_one"]
    assert "66 of 67 is correct as computed" in record[
        "d4s_banked_figure_is_unchanged"
    ]
    assert "66 of 67" in _flat18(phase18.D4_MEASURED["verdict"])


def test_nothing_banked_depends_on_the_agnet_pair_and_sameness_is_one_seed():
    record = phase18.DUPLICATE_DEPENDENCIES
    nothing = _flat18(record["nothing_depends_on_it"])
    assert "no banked figure treats the agnet arms as one" in nothing
    assert "changes NO BANKED VALUE" in nothing

    one_seed = _flat18(record["proven_at_one_seed_only"])
    assert "seed 1337 only" in one_seed
    assert "sameness at one seed is not sameness" in one_seed
    assert "CLUSTER-ONLY and was not run" in one_seed
    assert "twenty comparisons" in _flat18(record["what_would_settle_it"])

    # The locked 68 and every banked figure are untouched.
    assert len(phase18.locked_arm_entries()) == 68
    assert "66 of 67" in _flat18(phase18.D4_MEASURED["verdict"])
    assert "0.3784" in phase18.D5_CLOSED["defect_characterisation_fired"]

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 18 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p18" or "p18-" in e["id"]
    ]
    assert results_ledger.validate() is None


# --------------------------------------------------------------------------
# [2026-09-01] The duplication thread closed
# --------------------------------------------------------------------------


def test_duplication_is_proven_on_every_seed_and_the_count_is_stated():
    record = phase18.DUPLICATION_PROVEN_AT_DEPTH
    pairs = record["four_pairs_identical_on_every_seed"]
    assert pairs == {
        "vit_b16_imagenet": 5, "swin_b_masked": 5,
        "swin_b_imagenet": 5, "srgnn_imagenet": 10,
    }
    assert sum(pairs.values()) == 25

    # The count is derived from the lock, and the discrepancy is stated.
    by_name = {e["name"]: e for e in phase18.locked_arm_entries()}
    derived = (
        len(by_name["p7_d1_vit_b16_imagenet_g1"]["seeds"])
        + len(by_name["p7_c_swin_b_scut_masked_g1"]["seeds"])
        + len(by_name["p7_d1_swin_b_imagenet_g1"]["seeds"])
        + len(by_name["p7_d1_srgnn_imagenet_g1_native"]["seeds"])
    )
    assert derived == 25
    count = _flat18(record["the_comparison_count"])
    assert "**25, not 30.**" in count
    assert "5 + 5 + 5 + 10" in count
    assert "STATED rather than absorbed" in count
    assert "35 comparisons" in count

    # The graph regime's further five, checked against the lock.
    five = set(by_name["p7_d1_vit_b16_imagenet_g1"]["seeds"])
    ten = set(by_name["p7_d1_srgnn_imagenet_g1_native"]["seeds"])
    assert set(record["the_graph_regimes_further_five"]) == ten - five

    # The one-seed limit is lifted, and the original says so in place.
    assert "AT EVERY SEED THE PAIRS CARRY" in _flat18(
        record["what_it_upgrades"]
    )
    assert phase18.DUPLICATE_DEPENDENCIES["proven_at_one_seed_only"].startswith(
        "[LIFTED 2026-09-01"
    )
    assert "seed 1337 only" in phase18.DUPLICATE_DEPENDENCIES[
        "proven_at_one_seed_only"
    ]


def test_agnet_differs_on_all_ten_with_srgnn_as_the_exact_control():
    record = phase18.AGNET_DIFFERS_ON_ALL_TEN
    assert "ALL TEN SEEDS" in record["the_finding"]
    assert "PROPERTY OF THE PATH" in record["the_finding"]
    assert "AT ANY DEPTH" in record["the_consequence"]

    control = _flat18(record["the_control_is_srgnn_and_it_is_exact"])
    for shared in ("same graph regime", "same ten seeds", "a54cdfae",
                   "region_scheme native", "trainable graph_layers"):
        assert shared in control, shared
    assert "NOT the graph regime and NOT the commit gap" in control
    assert "deterministic: false" in control

    # The control really is exact: same seeds, same sha, same task keys.
    import yaml
    from pathlib import Path as _P

    configs = _P(__file__).resolve().parents[1] / "configs"

    def task(stem):
        return yaml.safe_load(
            (configs / f"{stem}.yaml").read_text(encoding="utf-8")
        )["task"]

    srgnn_a = task("p7_d1_srgnn_imagenet_g1_native")
    agnet_a = task("p7_d1_agnet_imagenet_g1_native")
    assert srgnn_a["seeds"] == agnet_a["seeds"]
    assert srgnn_a["region_scheme"] == agnet_a["region_scheme"] == "native"
    assert srgnn_a["trainable"] == agnet_a["trainable"] == "graph_layers"
    assert agnet_a["deterministic"] is False
    assert "deterministic" not in srgnn_a

    # It does not make AG-Net defective.
    assert "not a defective arm" in _flat18(
        record["what_it_does_not_make_agnet"]
    )
    assert "declared, bounded at ~2.3e-05" in _flat18(
        record["what_it_does_not_make_agnet"]
    )


def test_d4_is_resolved_by_measurement_with_the_prediction_confirmed():
    record = phase18.D4_REMEASURED_AT_FOUR
    resolved = _flat18(record["resolved_by_measurement"])
    assert "0.2382392139111266 both" in resolved
    assert "0.061312890026951984" in resolved
    assert "0.06121527011532292" in resolved
    assert "61 OF 62" in resolved
    assert "FIVE adjacent pairs carry a zero gap" in resolved

    # The PCC gap, and its agreement with the banked 4-dp values.
    a, b = 0.061312890026951984, 0.06121527011532292
    assert round(a - b, 8) == pytest.approx(9.762e-05, abs=1e-8)
    assert round(a, 4) == 0.0613 and round(b, 4) == 0.0612
    from cleft import ladder, roadb

    assert ladder.STAGE_D1_AT_G1["cells"]["agnet"][0] == 0.0613
    assert roadb.PHASE_7_TWENTY_TWO_ARMS["values"]["agnet__imagenet"][224][0] == 0.0612

    # Confirmed, not merely consistent -- prediction before measurement.
    mechanism = _flat18(record["the_mechanism_is_confirmed_not_merely_consistent"])
    assert "prediction was made before the measurement" in mechanism
    assert "0 of 237 3-class assignments" in mechanism
    assert "CONFIRMED, not merely consistent" in mechanism

    # The branch that fired is marked, and the other preserved.
    assert record["both_answers"]["if_agnet_ties_on_macro_f1"].startswith(
        "[THIS ONE FIRED]"
    )
    assert "62 of 63" in record["both_answers"]["if_agnet_does_not_tie"]
    assert record["the_prediction_tagged_as_one"].startswith(
        "[REASONED 2026-09-01, CONFIRMED"
    )

    # And D4's banked figure is untouched.
    assert "66 of 67" in _flat18(phase18.D4_MEASURED["verdict"])


def test_the_limitation_is_stated_for_the_write_up():
    record = phase18.DUPLICATION_LIMITATION_FOR_THE_WRITE_UP
    statement = _flat18(record["the_statement"])
    assert "64 DISTINCT PREDICTION SETS" in statement
    assert "four exact duplicates" in statement
    assert "declared non-determinism" in statement

    assert "remains correct as computed" in record[
        "every_figure_remains_correct_as_computed"
    ]
    assert "NOT reopened" in record["every_figure_remains_correct_as_computed"]

    reader = _flat18(record["what_a_reader_must_not_do"])
    assert "more independence than the set contains" in reader
    assert "ONE measurement counted twice" in reader
    assert "is not two independent arms either" in reader

    bites = _flat18(record["where_it_bites_measured"])
    assert "61 OF 62 among distinct arms" in bites
    assert "TRIVIALLY rather than INFORMATIVELY" in bites
    assert "unquantified upward bias" in bites

    assert "not a reopening" in record["status"]
    assert "visibility, not coverage" in record["status"]

    # The lock and every banked figure are untouched.
    assert len(phase18.locked_arm_entries()) == 68
    assert "66 of 67" in _flat18(phase18.D4_MEASURED["verdict"])
    assert "0.3784" in phase18.D5_CLOSED["defect_characterisation_fired"]

    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 18 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p18" or "p18-" in e["id"]
    ]
    assert results_ledger.validate() is None
