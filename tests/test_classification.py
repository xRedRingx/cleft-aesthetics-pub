"""The post-Phase-15-closing classification addendum, and Phase 16's
place in the sequence.

Two things are checked here that prose cannot check for itself: that the
macro convention has exactly one definition in the repository, and that
the discretisation is the frozen rule rather than a fourth copy of it.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from cleft import classification, phase10, phase12, phase14, phase15
from cleft.eval import metrics as frozen_metrics

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# the metrics themselves
# --------------------------------------------------------------------------


def test_the_report_is_hand_checkable():
    """One small case, worked by hand, so the implementation is pinned
    to arithmetic rather than to itself."""
    truth = np.array([0, 0, 0, 1, 1, 1, 1, 2])
    predicted = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    report = classification.prf_report(truth, predicted, 3)

    assert report["support"] == [3, 4, 1]
    assert report["predicted_counts"] == [2, 4, 2]
    assert report["confusion"] == [[2, 1, 0], [0, 3, 1], [0, 0, 1]]
    assert report["n"] == 8
    assert report["accuracy"] == pytest.approx(6 / 8)

    # class 0: P 2/2, R 2/3, F1 0.8 | class 1: P 3/4, R 3/4, F1 0.75
    # class 2: P 1/2, R 1/1, F1 2/3
    assert report["per_class"]["0"]["f1"] == pytest.approx(0.8)
    assert report["per_class"]["1"]["f1"] == pytest.approx(0.75)
    assert report["per_class"]["2"]["f1"] == pytest.approx(2 / 3)
    assert report["f1_macro"] == pytest.approx((0.8 + 0.75 + 2 / 3) / 3)
    assert report["f1_weighted"] == pytest.approx(
        (0.8 * 3 + 0.75 * 4 + (2 / 3) * 1) / 8
    )


def test_there_is_exactly_one_macro_convention():
    """The cross-check that keeps two implementations from becoming two
    conventions -- the defect shape this repository has met repeatedly
    (cluster_csv's docstring records three instances of another)."""
    rng = np.random.default_rng(11)
    for _ in range(25):
        truth = rng.integers(0, 3, size=40)
        predicted = rng.integers(0, 3, size=40)
        mine = classification.prf_report(truth, predicted, 3)
        theirs = phase10.top1_macro_prf(truth + 1, predicted + 1, n_classes=3)
        for key in ("precision_macro", "recall_macro", "f1_macro", "accuracy"):
            assert mine[key] == pytest.approx(theirs[key], abs=1e-12), key

    # And the check is live: a divergence raises rather than passing.
    assert "phase10.top1_macro_prf" in mine["macro_convention"]


def test_the_absent_class_convention_is_phase_tens():
    """A class with no truths is EXCLUDED from the macro average; a
    class with no predictions has precision 0. Both inherited, both
    stated, because both change the number."""
    truth = np.array([0, 0, 1, 1])
    predicted = np.array([0, 0, 1, 2])
    report = classification.prf_report(truth, predicted, 3)
    assert report["classes_present"] == [0, 1]
    assert report["classes_absent_excluded_from_macro"] == [2]
    # Class 2 was predicted once and is never true: precision 0, and it
    # does not drag the macro average because it has no support.
    assert report["per_class"]["2"]["precision"] == 0.0
    assert report["per_class"]["2"]["support"] == 0
    assert report["f1_macro"] == pytest.approx(
        (report["per_class"]["0"]["f1"] + report["per_class"]["1"]["f1"]) / 2
    )


def test_the_majority_floor_disagrees_with_itself_across_metrics():
    """The most useful single number in the addendum: a constant
    predictor scores its accuracy floor and a POOR macro F1."""
    truth = np.array([0] * 94 + [1] * 119 + [2] * 24)
    floor = classification.majority_baseline(truth, 3)

    assert floor["majority_class"] == 1
    assert floor["accuracy"] == pytest.approx(119 / 237)
    # phase9 banked this floor; the addendum's config asserts it too.
    assert round(floor["accuracy"], 3) == 0.502
    assert floor["chance"] == pytest.approx(1 / 3)
    # Always-class-1: precision 119/237, recall 1, F1 ~0.668, and zero
    # on both other classes -> macro F1 is a third of that.
    assert floor["f1_macro"] == pytest.approx(
        (2 * (119 / 237) / (1 + 119 / 237)) / 3
    )
    assert floor["f1_macro"] < floor["accuracy"] / 2


def test_the_discretisation_is_the_frozen_rule_not_a_copy():
    """No fourth copy of 2.5/3.5. The module delegates, and its source
    says so -- checked by AST, not by reading the comment."""
    values = np.array([1.0, 2.49, 2.5, 3.0, 3.5, 3.51, 5.0])
    assert list(classification.discretise(values)) == list(
        frozen_metrics.to_3class(values)
    )

    source = (REPO / "src" / "cleft" / "classification.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "discretise"
    )
    calls = [
        ast.unparse(node.func) for node in ast.walk(function)
        if isinstance(node, ast.Call)
    ]
    assert "frozen_metrics.to_3class" in calls
    # The thresholds appear NOWHERE in this module as literals.
    for literal in ("2.5", "3.5"):
        assert f"< {literal}" not in source and f"> {literal}" not in source


def test_refusals():
    with pytest.raises(classification.ClassificationError, match="truths"):
        classification.confusion([0, 1], [0, 1, 2], 3)
    with pytest.raises(classification.ClassificationError, match="outside"):
        classification.confusion([0, 3], [0, 1], 3)
    with pytest.raises(classification.ClassificationError, match="no rows"):
        classification.confusion([], [], 3)


# --------------------------------------------------------------------------
# the registration
# --------------------------------------------------------------------------


def test_the_readings_are_registered_before_any_number():
    record = classification.CLASSIFICATION_METRICS_SECONDARY
    assert "before any number exists" in record["registered"]
    assert "SECONDARY and DESCRIPTIVE" in record["what_this_is"]
    assert "NEVER A CLAIM. NEVER A LEDGER ROW." in record["what_this_is_not"]
    assert "No training" in record["no_retraining"]

    # The three caveats the maintainer required, each saying its own thing.
    assert "CLASS-SUPPORT VECTOR TRAVELS" in record["caveat_a_support"]
    assert "5/89/110/30/3" in record["caveat_a_support"]
    macro = record["caveat_b_macro_averaging"]
    assert "3-PATIENT CLASS EQUALLY WITH A" in macro
    assert "110-PATIENT" in macro
    assert "DIFFERENCE is the informative quantity" in macro
    reporting = record["caveat_c_discretisation_is_reporting"]
    assert "REPORTING STEP, NOT THE TRAINING" in reporting
    assert "would be a different model" in reporting

    assert "0.502" in record["the_floor"]
    assert "refuses" in record["the_floor"]
    # PCC's primacy is restated, not re-argued from the new numbers.
    assert "PRE-REGISTERED" in record["why_pcc_stays_primary"]
    assert "would still be secondary" in record["why_pcc_stays_primary"]


def test_three_classes_not_five_is_answered_plainly():
    record = classification.THREE_NOT_FIVE
    assert "IT COLLAPSES TO THREE" in record["the_answer"]
    assert "ONLY THREE ARE DEFINED" in record["the_answer"]
    assert "FOUR thresholds" in record["why_five_is_not_reported"]
    assert "NOWHERE IN THE RECORD" in record["why_five_is_not_reported"]
    assert "UNDEFINED rather than invented" in record["why_five_is_not_reported"]
    # What would make it defined is named, so the gap is actionable.
    assert "4-threshold rule" in record["what_would_make_it_defined"]

    # And the claim is true of the frozen function: two thresholds, three
    # outputs, no fifth class anywhere.
    values = np.linspace(1.0, 5.0, 401)
    assert sorted(set(frozen_metrics.to_3class(values).tolist())) == [0, 1, 2]


def test_the_addendum_config_declares_everything_before_the_run():
    text = (REPO / "configs" / "p7_d1_classification_metrics.yaml").read_text(
        encoding="utf-8"
    )
    config = yaml.safe_load(text)
    task = config["task"]

    assert task["kind"] == "classification_metrics"
    assert task["arm"] == "p7_d1_vit_b16_imagenet_g1"
    assert task["n_classes"] == 3
    assert task["expect_majority"] == 0.502

    # The seeds are the ARM's own, read from its shipped config.
    arm = yaml.safe_load(
        (REPO / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert task["seeds"] == arm["task"]["seeds"]
    assert len(task["seeds"]) == 5

    # The header states what it is and is not, before any number.
    for phrase in (
        "NEVER A CLAIM. NEVER A LEDGER",
        "POD ARITHMETIC",
        "THREE CLASSES, NOT FIVE",
        "THE FLOOR IS DECLARED BEFORE THE RUN",
        "THREE CAVEATS RIDE IN THE ARTIFACT",
    ):
        assert phrase in text, phrase
    # **UPDATED 2026-08-24, when the run was declared.** This assertion
    # read "PLACEHOLDER is still in the header" and FAILED when the hash
    # landed -- which is what pinning the declaration state is for. The
    # config is now fully resolved and says so.
    assert "**RESOLVED.**" in text
    assert "PLACEHOLDER" not in text
    assert all(
        set(entry["rollup_sha256"]) != {"0"} for entry in config["inputs"]
    )

    spec = importlib.util.spec_from_file_location(
        "generate_addendum_configs",
        REPO / "scripts" / "generate_addendum_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["--check"]) == 0


def test_the_task_reads_the_stored_class3_and_asserts_the_floor():
    import inspect

    from cleft import run as run_module

    assert "classification_metrics" in run_module.TASKS
    source = inspect.getsource(run_module.task_classification_metrics)

    # Truth's classes come from the MANIFEST COLUMN, not re-derived.
    assert 'int(r["class3"])' in source
    # The floor is asserted before any per-seed number is computed.
    assert source.index("expect_majority") < source.index("for seed in seeds")
    assert "Refusing: a metric quoted against the wrong floor" in source
    # The three-way consistency check between stored and derived classes.
    assert "stored != derived" in source
    # Pooling is per-seed metrics, not averaged predictions -- and the
    # source says why.
    assert "never trained or evaluated" in source
    # No ledger row, and the caveats are written into the artifact.
    assert "NO LEDGER ROW" in source
    assert "the_caveats" in source


def test_the_arm_run_directory_is_the_pasted_one_and_ties_to_the_arm():
    """The declared input, and the contract that ties it to the arm.

    **The stem check is the one that matters here.** This addendum
    claims to describe a specific arm's predictions; the run-directory
    contract makes its stem the ARM CONFIG's stem, so a directory from
    any other arm fails this test rather than being silently tabulated
    under the wrong arm's name.
    """
    config = yaml.safe_load(
        (REPO / "configs" / "p7_d1_classification_metrics.yaml").read_text(
            encoding="utf-8"
        )
    )
    entry = next(e for e in config["inputs"] if e["name"] == "arm_run")
    assert "PENDING" not in entry["path"]
    directory = entry["path"].rsplit("/", 1)[-1]
    assert entry["path"].endswith("/runs/keeper/p7/" + directory)

    stem, sha8, job = directory.split("__")
    assert stem == config["task"]["arm"] == "p7_d1_vit_b16_imagenet_g1"
    assert (REPO / "configs" / f"{stem}.yaml").is_file()
    assert len(sha8) == 8 and all(c in "0123456789abcdef" for c in sha8)
    assert job == "p7-d1-vit-imagenet"

    # **DECLARED 2026-08-24** from declare_inputs.py on the cluster
    # (287 files, ~3.2 MB). Never invented on the laptop: this assertion
    # changed from "still a placeholder" to "a real digest" only when
    # the maintainer pasted the measured value.
    rollup = entry["rollup_sha256"]
    assert len(rollup) == 64
    assert set(rollup) <= set("0123456789abcdef")
    assert set(rollup) != {"0"}
    assert rollup == (
        "4573e984a20c7771e4ebbda1b4f17e52eb63ea33c178be1a76a1af0ffd2873a2"
    )
    # **AND IT IS NOT THE ARM'S OWN CONFIG HASH.** The addendum declares
    # the arm's RUN DIRECTORY -- 287 files of outputs -- not the config
    # that produced it. Two different objects, and confusing them would
    # declare something that never contained a prediction.
    assert rollup != entry["path"]
    manifest_entry = next(
        e for e in config["inputs"] if e["name"] == "manifest_v1"
    )
    assert rollup != manifest_entry["rollup_sha256"]

    # The contract this rests on, cited rather than remembered.
    context = (REPO / "src" / "cleft" / "provenance" / "context.py").read_text(
        encoding="utf-8"
    )
    assert 'f"{self.config_path.stem}__{self.git.sha8}__{suffix}"' in context

    spec = importlib.util.spec_from_file_location(
        "generate_addendum_configs_armrun",
        REPO / "scripts" / "generate_addendum_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ARM_RUN_DIR == directory
    assert module.main(["--check"]) == 0


# --------------------------------------------------------------------------
# 2026-08-24, the addendum ran
# --------------------------------------------------------------------------


def test_the_banked_figures_reproduce_from_their_own_support_vector():
    """Every derived number re-derived here, so the record cannot drift
    from the arithmetic it claims."""
    record = classification.CLASSIFICATION_METRICS_BANKED
    assert "b677f53c" in record["banked"]
    assert "single" in record["banked"] and "clean attempt" in record["banked"]

    support = [88, 119, 30]
    assert sum(support) == 237
    floor = classification.majority_baseline(
        np.repeat([0, 1, 2], support), 3
    )
    # The floor check the task performed, re-performed.
    assert round(floor["accuracy"], 4) == 0.5021
    assert round(floor["f1_macro"], 4) == 0.2228
    assert "0.5021" in record["the_floor_check_passed"]
    assert "0.5020" in record["the_floor_check_passed"]
    assert str(support) in record["the_support"]

    # The two gains and their ratio.
    accuracy, f1_macro = 0.5181, 0.3760
    gain_accuracy = accuracy - floor["accuracy"]
    gain_f1 = f1_macro - floor["f1_macro"]
    assert round(gain_accuracy, 4) == 0.0160
    assert round(gain_f1, 4) == 0.1532
    assert round(gain_f1 / gain_accuracy, 2) == 9.58
    for figure in ("+0.0160", "+0.1532", "9.58"):
        assert figure in record["the_headline"], figure
    assert "order of magnitude apart" in record["the_headline"]

    # The intervals, on the mean of five seeds.
    for value, sd, low, high in (
        (accuracy, 0.0159, 0.5042, 0.5320),
        (f1_macro, 0.0166, 0.3614, 0.3906),
    ):
        half = 1.96 * sd / np.sqrt(5)
        assert round(value - half, 4) == low
        assert round(value + half, 4) == high
        assert f"[{low:.4f}, {high:.4f}]" in record["the_intervals_sharpen_it"]
    # The refinement that matters: one lower bound clears its floor by a
    # hair, the other by a mile.
    assert round(0.5042 - floor["accuracy"], 4) == 0.0021
    assert round(0.3614 - floor["f1_macro"], 4) == 0.1386
    assert "BARELY separable" in record["the_intervals_sharpen_it"]
    assert "distinguishable from guessing" in record["the_intervals_sharpen_it"]

    # Per-seed ranges bracket the pooled means.
    pooled = record["pooled"]
    assert "0.4937-0.5359" in pooled and "0.3509-0.3956" in pooled
    assert 0.4937 <= accuracy <= 0.5359
    assert 0.3509 <= f1_macro <= 0.3956


def test_the_support_is_not_the_median_distribution():
    """A confusion the record heads off by name: the mean-collapsed
    supports are not the median-grade counts, and one number coincides."""
    record = classification.CLASSIFICATION_METRICS_BANKED
    median_counts = [5, 89, 110, 30, 3]
    assert sum(median_counts) == 237
    # The median grades collapsed at 2.5/3.5 would give this instead:
    collapsed = [
        median_counts[0] + median_counts[1],
        median_counts[2],
        median_counts[3] + median_counts[4],
    ]
    assert collapsed == [94, 110, 33]
    assert collapsed != [88, 119, 30]
    assert "94/110/33" in record["the_support"]
    assert "NOT the median-grade distribution" in record["the_support"]
    assert "coincidence of" in record["the_support"]


def test_the_addendum_stays_secondary_after_producing_a_result():
    record = classification.CLASSIFICATION_METRICS_BANKED
    assert "SECONDARY AND DESCRIPTIVE" in record["what_these_are"]
    assert "no ledger" in record["what_these_are"].lower()
    assert "still its PCC" in record["what_these_are"]

    # It refuses the reading its own headline invites.
    refused = record["what_this_does_not_say"]
    assert "does NOT say macro F1 is the better metric" in refused
    assert "property of the" in refused and "not evidence about the model" in refused

    # The floor's own split is attributed to the distribution, not the arm.
    assert "before any arm is involved" in record["the_floors_own_disagreement"]

    # All three caveats travel.
    attached = record["caveats_attached"]
    for phrase in ("[88, 119, 30]", "30-patient class equally with the",
                   "REPORTING step", "would be a different arm"):
        assert phrase in attached, phrase


def test_the_five_class_incomparability_names_the_numbers_it_guards():
    record = classification.CLASSIFICATION_METRICS_BANKED
    assert "FOUR-threshold" in record["the_five_class_family_is_still_undefined"]

    trap = record["not_comparable_with_cleftgnns_tables"]
    assert "0.3760 and 0.3352 look adjacent" in trap
    assert "3 classes vs 5" in trap
    assert "Neither" in trap and "quoted beside the other" in trap

    # The figures it guards against are really in phase10, and really
    # 5-class: top1_macro_prf defaults to five and the faithful arm's
    # call site takes the default.
    import inspect

    from cleft import run as run_module

    assert "0.3352" in phase10.FAITHFUL_ARM_CLOSING["sentence"]
    assert "0.077-0.154" in phase10.FAITHFUL_ARM_CLOSING["sentence"]
    signature = inspect.signature(phase10.top1_macro_prf)
    assert signature.parameters["n_classes"].default == 5
    # [2026-08-30 the pin fired as designed] There are now TWO call
    # sites taking the 5-class default, and the second is legitimate:
    # the Phase 10 annex's compute gate, where 5-class IS constructed
    # the way theirs is -- an integer 1-5 target under cross-entropy,
    # argmax, no threshold invented (the SCOPED exception dated beside
    # THREE_NOT_FIVE). The incomparability rule this test guards is
    # UNCHANGED and covers the annex too: those figures characterise
    # OUR replication under their protocol and are never placed beside
    # their tables (phase10_annex.ANNEX_PROHIBITION).
    source = inspect.getsource(run_module)
    assert source.count("phase10.top1_macro_prf(truth, top1)") == 2, (
        "the faithful arm's or the annex gate's call site moved"
    )
    assert "def task_cleftgnn_faithful" in source
    assert "def task_p10x_gate_fullfit" in source
    from cleft import phase10_annex

    assert phase10_annex.ANNEX_PROHIBITION.startswith(
        "THESE FIGURES CHARACTERISE OUR REPLICATION UNDER THEIR PROTOCOL"
    )
    assert classification.THREE_NOT_FIVE["scoped_exception_2026_08_30"] == (
        "phase10_annex.FIVE_CLASS_SCOPED_EXCEPTION"
    )

    # And the rule it extends exists, under the name cited.
    rule = record["the_rule_now_has_a_second_instance"]
    assert "CLEFTGNN_COMPARATOR_TABLES" in rule
    assert "quantity_distinct_from_0_2520" in (
        rule + str(phase10.CLEFTGNN_COMPARATOR_TABLES["the_598_cell"])
    )
    assert "quantity_distinct_from_0_2520" in phase10.CLEFTGNN_COMPARATOR_TABLES[
        "the_598_cell"
    ]
    assert "sharing a NAME" in rule


# --------------------------------------------------------------------------
# Phase 16 on the sequence
# --------------------------------------------------------------------------


def test_the_third_renumbering_keeps_the_old_mapping_visible():
    record = phase15.PHASE_SEQUENCE_RENUMBERED_3
    assert record["was"] == {
        "16": "TSTR on the best from 15", "17": "write-up",
    }
    assert "METRIC-SPACE ABLATION" in record["becomes"]["16"]
    assert "TSTR" in record["becomes"]["17"]
    assert record["becomes"]["18"] == "write-up"
    assert record["status_changes"]["tstr"] == "Phase 16 -> Phase 17"

    # The chain of amendments is walkable in both directions.
    assert "phase11.PHASE_SEQUENCE_RENUMBERED" in record["first_amendment"]
    assert "phase12.PHASE_SEQUENCE_RENUMBERED_2" in record["second_amendment"]
    assert phase12.PHASE_SEQUENCE_RENUMBERED_2["third_amendment"] == (
        "phase15.PHASE_SEQUENCE_RENUMBERED_3"
    )
    # The older record is NOT edited to match -- it still says 16: TSTR.
    assert "TSTR" in phase12.PHASE_SEQUENCE_RENUMBERED_2["becomes"]["16"]

    # [2026-08-24] The FOURTH amendment extends the chain: anchor loop
    # to 16, metric-space ablation to 18, write-up to 19, TSTR staying
    # at 17. Walked in both directions like the links before it.
    fourth = phase15.PHASE_SEQUENCE_RENUMBERED_4
    assert fourth["was"] == {
        "16": "the METRIC-SPACE ABLATION (working title)",
        "17": "TSTR on the best from 15",
        "18": "write-up",
    }
    assert "ANCHOR LOOP" in fourth["becomes"]["16"]
    assert "ANCHOR_LOOP_REGISTERED" in fourth["becomes"]["16"]
    assert "TSTR" in fourth["becomes"]["17"]
    assert "METRIC-SPACE ABLATION" in fourth["becomes"]["18"]
    assert fourth["becomes"]["19"] == "write-up"
    assert fourth["status_changes"]["tstr"] == "Phase 17 -> Phase 17 (unchanged)"
    # Forward from every earlier link...
    assert record["fourth_amendment"] == "PHASE_SEQUENCE_RENUMBERED_4"
    assert phase12.PHASE_SEQUENCE_RENUMBERED_2["fourth_amendment"] == (
        "phase15.PHASE_SEQUENCE_RENUMBERED_4"
    )
    # ...and backward through all three predecessors.
    assert "PHASE_SEQUENCE_RENUMBERED_3" in fourth["third_amendment"]
    assert "phase12.PHASE_SEQUENCE_RENUMBERED_2" in fourth["second_amendment"]
    assert "phase11.PHASE_SEQUENCE_RENUMBERED" in fourth["first_amendment"]
    # The superseded records keep their old numbers, unedited.
    assert "METRIC-SPACE ABLATION" in record["becomes"]["16"]
    assert phase15.PHASE_16_SCHEDULED["renumbered_to_18"] == (
        "PHASE_SEQUENCE_RENUMBERED_4"
    )
    assert phase15.ANCHOR_LOOP_REGISTERED["promoted"] == (
        "PHASE_SEQUENCE_RENUMBERED_4"
    )
    assert "noted item under PHASE_16_SCHEDULED" in (
        phase15.ANCHOR_LOOP_REGISTERED["where_it_runs"]
    )

    # The stated dependency: 16 does not wait on 18, because its
    # evaluation is pass zero's ALREADY-REGISTERED one -- and that
    # registration really carries the metrics the record names.
    from cleft import phase9

    inversion = fourth["the_apparent_inversion"]
    assert "LOOKS inverted. It is not." in inversion
    assert "PROTOTYPE_CLASSIFIER_REGISTERED" in inversion
    assert "chooses no metrics fresh" in inversion
    registered = str(phase9.PROTOTYPE_CLASSIFIER_REGISTERED)
    for named in ("PCC", "Spearman", "0.333", "0.502", "2.5/3.5"):
        assert named in registered, named
    gain = fourth["the_compensating_gain"]
    assert "ALL results" in gain and "16 and 17 included" in gain
    assert "snapshot that goes stale" in gain

    # Every module still naming an old phase number carries a pointer
    # to the LATEST amendment.
    from cleft import classification as classification_module

    for module in (phase12, phase14, phase15, classification_module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        if any(f"Phase 1{n}" in source for n in (6, 7, 8)):
            assert "PHASE_SEQUENCE_RENUMBERED_4" in source, module.__name__


def test_phase_16_is_scheduled_with_its_question_and_not_its_scope():
    record = phase15.PHASE_16_SCHEDULED
    assert record["working_title"] == "metric-space ablation"
    question = record["the_registered_question"]
    assert "classification metrics measure on this cohort" in question
    assert "where do the two disagree" in question

    # Scope withheld, with the reason drawn from Phase 15's own lesson.
    scope = record["scope_is_not_set_here"]
    assert "proposed at the phase's" in scope and "RESTATE" in scope
    assert "CRITERION_1_AMENDED" in scope

    # The four noted items, marked as noted rather than committed.
    assert "none of them committed" in record["noted_content_not_a_scope"]
    assert "changes the RANKING" in record[
        "noted_1_the_full_ladder_under_both_families"
    ]
    assert "not idle" in record["noted_2_the_criterions_two_conditions"]
    assert "Phase 11 already found" in record[
        "noted_2_the_criterions_two_conditions"
    ]
    assert "SET ASIDE" in record["noted_3_the_qwk_question"]
    assert "5/89/110/30/3" in record["noted_4_what_macro_averaging_can_support"]

    # The reckoning discipline, named to the phase that set it.
    reckoning = record["must_open_with_a_reckoning"]
    assert "phase14.PHASE_14_CONCEDED_COVERED" in reckoning
    assert "PCC is primary, QWK is not" in reckoning
    assert "CONCEDING" in reckoning
    assert "Scheduling is not reopening" in reckoning

    # Everything it names must exist.
    assert hasattr(phase14, "PHASE_14_CONCEDED_COVERED")
    assert hasattr(frozen_metrics, "qwk_3cat")
    assert hasattr(classification, "CLASSIFICATION_METRICS_SECONDARY")
    assert "classification.CLASSIFICATION_METRICS_SECONDARY" in record[
        "what_it_already_has_in_hand"
    ]
    assert "declines to" in record["what_it_already_has_in_hand"]


def test_the_anchor_loop_is_registered_and_nothing_is_built():
    """Registration only: the design, its readings, its prohibition --
    and NO task, NO schema kind, NO config exists for it."""
    record = phase15.ANCHOR_LOOP_REGISTERED
    assert "build nothing tonight" in record["registered"]

    # The premise is CITED, and the citations resolve.
    premise = record["pass_zero_is_measured"]
    assert "PROTOTYPE_CLASSIFIER_OBSERVED" in premise
    assert "p9-anchor-classifier-convergence" in premise
    assert "4/25" in premise and "3/25" in premise and "4.75/25" in premise
    assert "lives entirely in what pass zero lacks" in premise
    from cleft import phase9, results_ledger

    observed = phase9.PROTOTYPE_CLASSIFIER_OBSERVED
    assert observed["anchor_self_consistency"]["euclidean"] == "4/25"
    assert observed["anchor_self_consistency"]["cosine"] == "3/25"
    assert "3/7/6/6/3" in observed["anchor_self_consistency"][
        "chance_expectation"
    ]
    # The banked wording is carried precisely, not rounded to "at chance".
    assert "marginally ABOVE chance" in premise
    assert "marginally ABOVE" in observed["cells"]["stated_precisely"]
    assert any(
        entry["id"] == "p9-anchor-classifier-convergence"
        for entry in results_ledger.ENTRIES
    )

    # (1) Anchors: per-anchor assignment, fixed for all time, and the
    # instability they are immune to is the MEASURED persistence.
    anchors = record["anchors"]
    assert "each anchor its" in anchors and "own prototype" in anchors
    assert "PER-ANCHOR" in anchors
    assert "FIXED for all time" in anchors
    assert "0.394/0.427/0.414" in anchors
    figures = phase9.PROTOTYPES_OBSERVED["figures"]
    measured = {cell["persistence"] for cell in figures.values()}
    assert {0.394, 0.427, 0.414} <= measured

    # (2) The loop: folds, seeds, and the forbidden variant named.
    loop = record["the_loop"]
    assert "TRAINING FOLDS ONLY" in loop
    assert "out-of-fold on cleft_v1" in loop
    assert "five seeds" in loop
    assert "alternatives NOTED, not" in loop
    forbidden = record["the_forbidden_version_named"]
    assert "MEMORISATION and is forbidden" in forbidden
    assert "criterion" in forbidden and "UNCHANGED" in forbidden

    # (3) The deliverable is what the probe cannot produce, and the
    # registration says so BEFORE any null could motivate retrofitting.
    deliverable = record["the_explainability_deliverable"]
    assert "per-patient mismatch records" in deliverable
    assert "WITH DISTANCES" in deliverable
    assert "linear probe cannot produce" in deliverable
    assert "not retrofitted" in deliverable

    # (4) The prohibition cites the two measurements behind it.
    prohibition = record["prohibition_no_decoded_prototypes"]
    assert "SCUT_NORMAL_PRIOR" in prohibition
    assert "+0.4856" in prohibition and "+0.0082" in prohibition
    assert "explanation that LIES" in prohibition
    assert "REAL images" in prohibition
    from cleft import phase13

    assert hasattr(phase13, "SCUT_NORMAL_PRIOR")
    phase13_source = Path(phase13.__file__).read_text(encoding="utf-8")
    assert "0.4856" in phase13_source and "0.0082" in phase13_source

    # (5) Both readings committed, success defined as parity.
    assert "FIFTH convergent" in record["reading_if_null"]
    assert "COHORT_CANNOT_RESOLVE" in record["reading_if_null"]
    assert "surprise that matters more" in record["reading_if_above"]
    assert "parity-with-explanations, not victory" in record[
        "success_is_defined_before_the_run"
    ]

    # (6) Caveats: the inherited one matches pass zero's banked caveat.
    caveats = record["caveats_bound"]
    assert "not verified-unanimous" in caveats
    assert "not verified-unanimous" in observed["caveat"]
    assert "question 5's neighbour" in caveats
    assert "one backbone" in caveats
    assert "ImageNet anchor rides" in caveats

    # It sits under PHASE_16_SCHEDULED as registered-not-built.
    noted = phase15.PHASE_16_SCHEDULED["noted_5_the_anchor_loop"]
    assert "REGISTERED-NOT-BUILT" in noted
    assert "ANCHOR_LOOP_REGISTERED" in noted
    assert "only the BUILD belongs to" in noted

    # [RETIRED 2026-08-29] This test's tail asserted NOTHING was built.
    # That guard held through registration, rulings and compute
    # verification, and ended when the exit criteria locked
    # (phase16.NEGATIVE_SPACE_RETIRED -- the dated conversion lives in
    # test_phase16.py). What remains here is the registration-era fact
    # this test still owns: the build happened THROUGH the registration,
    # not around it.
    from cleft import phase16
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    assert "2026-08-29" in phase16.NEGATIVE_SPACE_RETIRED["retired"]
    assert "anchor_loop" in TASK_SPECS and "anchor_loop" in TASKS
    assert "phase15.ANCHOR_LOOP_REGISTERED" in phase16.ANCHOR_LOOP_RULINGS[
        "registration"
    ]
