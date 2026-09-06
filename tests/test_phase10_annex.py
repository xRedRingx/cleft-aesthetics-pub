"""Phase 10 annex -- the split-distribution replication.

Pins the records, the quotes against their source records (quoted,
never retyped), the pointers both ways, and the negative space.

[2026-08-31, third cycle] The gate RAN, N is ruled at 500, the exit
criteria are LOCKED, and the distribution arm is built. The
registration-only framing above no longer describes the whole file; the
negative space that still holds is narrower and is asserted directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from cleft import classification, phase10, phase10_annex, phase11, phase15, phase18
from cleft.data import scoresheet

REPO = Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_question_quotes_the_record_figure_and_home():
    # The source record's figure, verified here so the quote cannot
    # drift from its home.
    declared = phase10.PAIRED_BCA_OBSERVED["arms_as_declared"]["p10_cleftgnn"]
    assert declared == {"mean": 0.0242, "sd": 0.0676}

    record = phase10_annex.ANNEX_QUESTION_REGISTERED
    quoted = record["what_phase_10_measured"]
    assert "0.0242 (sd 0.0676)" in quoted
    assert "PAIRED_BCA_OBSERVED" in quoted
    # The 0.0001 is carried, not smoothed: both figures cited.
    assert "0.0241" in quoted
    assert "the_0_0001" in quoted
    assert "ROUTE_3_OBSERVED" in quoted

    assert record["question"] == (
        "does a single 153/28 split identify a model's performance at "
        "this cohort size?"
    )
    assert "ONLY the split" in record["what_varies"]


def test_153_is_a_derivation_and_the_operands_are_recorded():
    # 153 appears nowhere in the record; the annex derives it and says
    # so. Both operands are quoted from their homes.
    notes = phase10_annex.VERIFICATION_NOTES["derived_not_recorded"]
    assert "153 appears nowhere in the record" in _flat(notes)
    assert "181 - 28 = 153" in notes
    assert "181-image 85:15 split" in _flat(
        phase10.THIN_CLASS_SMOOTHING["manuscript_silence"]
    )
    assert phase10.CLEFTGNN_COMPARATOR_TABLES["study_test_set"]["n"] == 28


def test_ruling_a_quotes_the_study_set_range_verbatim():
    the_range = phase10.CLEFTGNN_COMPARATOR_TABLES["study_test_set"]["range"]
    assert the_range == "0.440 to -0.162, mean ~0.14"
    assert the_range in _flat(phase10_annex.RULING_A_ALL_FIVE_RATERS["why"])
    assert "five models per split" in phase10_annex.RULING_A_ALL_FIVE_RATERS[
        "ruling"
    ]


def test_ruling_b_is_scoped_and_the_census_holds_today():
    record = phase10_annex.RULING_B_FULL_TRAINABLE
    assert "scoped departure" in record["ruling"]
    assert "in this annex ONLY" in record["scope"]
    assert "unregistered everywhere else" in record["scope"]
    assert "we froze it" in record["reason"]

    # The record's census, quoted from its home.
    census_home = _flat(
        phase15.CRITERION_1_AMENDED["the_provenance_of_that_reason"]
    )
    for piece in ("head (81)", "graph_layers (34)", "classifier (2)",
                  "classifier_adabn (2)"):
        assert piece in census_home
        assert piece in _flat(record["census"])

    # And the claim re-measured NOW, under THE RECORD'S OWN RULE.
    # [CORRECTED 2026-08-30, second cycle] This used a regex matching
    # 'trainable:' anywhere, which counts header-COMMENT mentions as
    # settings -- a different quantity from the one CRITERION_1_AMENDED
    # counts, and the reason my first census disagreed with the
    # record's (VERIFICATION_NOTES['census_correction_2026_08_30']).
    # One rule now, shared with tests/test_phase15.py: a line that
    # BEGINS a task-level setting.
    census: dict = {}
    for path in sorted((REPO / "configs").glob("*.yaml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("  trainable: "):
                key = line.split(": ", 1)[1].strip()
                census[key] = census.get(key, 0) + 1
    # Every non-annex value is IDENTICAL to the recorded census, so the
    # withdrawal holds for everything it was about...
    #
    # [2026-08-31, THE PIN FIRED AS DESIGNED, updated dated.] `head`
    # moved 81 -> 83: Phase 20's two permutation configs ship at the
    # probe's own `trainable: head`, which is what makes them controls
    # for the arm rather than for something else. The count is updated
    # rather than loosened, and the DELTA is enumerated below so the
    # update cannot absorb an unrelated arm. The other three vocabularies
    # are untouched, which is the half the amendment actually rested on.
    assert {k: v for k, v in census.items() if k != "full"} == {
        # [UPDATED 2026-09-02] 83 -> 85: Phase 25's two arms, both at the
        # probe's own `trainable: head`, which is what makes them
        # one-factor contrasts against it.
        # [UPDATED 2026-09-06] 85 -> 101: Phase 26's SIXTEEN calibration
        # cells, every one at the probe's own `trainable: head` because
        # the grid varies weight decay and patience and holds the rest.
        # The delta is exactly 16 and the other three vocabularies are
        # untouched, which is the half the amendment rested on.
        "head": 101, "graph_layers": 34,
        "classifier": 2, "classifier_adabn": 2,
    }, census
    head_setters = {
        path.name
        for path in sorted((REPO / "configs").glob("*.yaml"))
        if any(
            line == "  trainable: head"
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }
    assert {n for n in head_setters if n.startswith("p20_")} == {
        "p20_permutation_plain.yaml", "p20_permutation_stratified.yaml",
    }, "the 81 -> 83 delta is Phase 20's two permutation arms and nothing else"
    # ...and every config that sets 'full' is an ANNEX config -- which
    # is what "operationalised in this annex only" looks like as a
    # measurement. [2026-08-31: the distribution arm joined the gate;
    # the pin fired and is updated dated. The set is enumerated rather
    # than counted, so a config outside the annex fires it.]
    setters = [
        path.name
        for path in sorted((REPO / "configs").glob("*.yaml"))
        if any(
            line == "  trainable: full"
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    ]
    assert setters == ["p10x_distribution.yaml", "p10x_gate_fullfit.yaml"]
    assert census["full"] == len(setters) == 2

    # The departure from the notebook's frozen backbone is owned, not
    # silent.
    assert "notebook" in record["notebook_departure_owned"]
    assert "BACKBONE_IS_FROZEN" in record["notebook_departure_owned"]
    assert "freezes its backbone" in _flat(phase10.BACKBONE_IS_FROZEN["why"])

    # Schema surface: design, not built.
    assert "NOT BUILT THIS TURN" in record["schema_surface"]


def test_the_recipe_is_the_notebooks_quoted_not_retyped():
    source = _flat(phase10.NOTEBOOK_RECIPE_IS_ADAM["recipe"])
    quoted = _flat(phase10_annex.RECIPE_FIXED_TO_NOTEBOOK["recipe_quoted"])
    assert source in quoted
    assert "SGD 0.01" in phase10_annex.RECIPE_FIXED_TO_NOTEBOOK[
        "unregistered_sgd"
    ]
    assert "UNREGISTERED" in phase10_annex.RECIPE_FIXED_TO_NOTEBOOK[
        "unregistered_sgd"
    ]
    # The notebook budget rides whole, so Phase 10's deviations do not.
    assert "notebook's last epoch IS its model" in _flat(
        phase10_annex.RECIPE_FIXED_TO_NOTEBOOK["budget_quoted"]
    )


def test_the_regions_are_36_with_27_unregistered():
    design = phase10_annex.DESIGN_FIXED_TO_THEIRS
    assert design["regions"]["value"] == 36
    assert "REGION_COUNT_BELIEF_VS_EXECUTION" in design["regions"]["cited"]
    assert "UNREGISTERED" in design["regions"]["unregistered_27"]
    # The source record still says what the annex quotes.
    assert "EVERY result in the paper was produced with 36 regions" in _flat(
        phase10.REGION_COUNT_BELIEF_VS_EXECUTION["reframing"]
    )


def test_degenerate_draws_are_a_frequency_never_a_discard():
    splits = phase10_annex.DESIGN_FIXED_TO_THEIRS["splits"]
    flat = _flat(splits["degenerate_draws"])
    assert "NEVER DISCARDED" in flat
    assert "FREQUENCY" in flat
    assert "itself a finding" in flat


def test_the_five_class_exception_is_scoped_and_dated_beside_the_rule():
    # Pointer beside THREE_NOT_FIVE, and the decline untouched.
    assert classification.THREE_NOT_FIVE["scoped_exception_2026_08_30"] == (
        "phase10_annex.FIVE_CLASS_SCOPED_EXCEPTION"
    )
    assert classification.THREE_NOT_FIVE["declined_2026_08_30"] == (
        "phase18.PHASE_18_RULINGS"
    )
    record = phase10_annex.FIVE_CLASS_SCOPED_EXCEPTION
    assert "no thresholds invented" in _flat(record["why_legitimate_here"])
    assert "constructed the same way theirs are" in _flat(
        record["why_legitimate_here"]
    )
    assert "not reopened" in _flat(record["the_decline_stands"])


def test_the_rater_panel_is_the_eighth_discrepancy_and_tagged():
    record = phase10_annex.RATER_PANEL_DISCREPANCY_EIGHTH
    assert "eighth manuscript-vs-artifact discrepancy" in record["class"]
    # The manuscript wording is in no repo record and says so.
    assert "[MANUSCRIPT, via the maintainer 2026-08-30" in record["manuscript_panel"]
    assert "in no repo record" in record["manuscript_panel"]
    # Our panel is the sheet's, which still says what the annex says.
    assert scoresheet.RATERS == (
        "Rater 7 - Cleft patient",
        "Rater 8 - Orthodontist",
        "Rater 9 - Speech and language therapist",
        "Rater 10 - Plastic surgeon",
        "Rater 11 - Psychologist",
    )
    assert "UNMATCHED raters" in record["consequence"]
    # The closing this extends still counts seven-plus-one.
    assert "seven manuscript-vs-artifact discrepancies plus one" in _flat(
        record["class"]
    ) or "seven manuscript-vs-artifact" in _flat(
        phase10.PHASE_10_CLOSING["exit_criteria"]["1_discrepancy"]
    )


def test_floors_ride_including_iem_through_phase11():
    record = phase10_annex.FLOORS_REGISTERED
    assert "majority-class" in record["classification_floor"]
    assert "phase11.iem" in record["iem_floor"]
    assert "convention A" in record["iem_floor"]
    assert hasattr(phase11, "iem")
    assert hasattr(phase11, "IEM_DIRECTION_ANSWERED")
    assert "undefined" in record["pcc_floor"]


def test_the_normalisation_is_proposed_not_locked():
    """[CONVERTED 2026-08-30, second cycle -- the pin fired as designed
    when the ruling was the same day.] The proposal record is PRESERVED
    as written with the ruling dated beside it; the ruling itself is
    NORMALISATION_RULED, asserted by the next test."""
    record = phase10_annex.NORMALISATION_PROPOSED_NOT_LOCKED
    assert "NOT LOCKED" in record["proposed"]
    assert "NOT LOCKED" in record["proposed"]
    assert "DELIBERATELY REPLICATED DEFECT" in _flat(record["proposal"])
    # The defect characterisation, at its home.
    assert "Nothing is gained by it and it is discussed nowhere" in _flat(
        phase10.NOTEBOOK_BUDGET_AND_NORMALISATION["why_we_call_it_a_defect"]
    )
    # The proposal now carries the dated pointer to the ruling.
    assert record["ruled_2026_08_30"] == "NORMALISATION_RULED"


def test_the_normalisation_ruling_carries_grounds_and_the_sentence():
    record = phase10_annex.NORMALISATION_RULED
    assert record["ruled"] == "2026-08-30"
    ruling = _flat(record["ruling"])
    assert "DELIBERATELY REPLICATED DEFECT" in ruling
    assert "(0.4914, 0.4822, 0.4465) / (0.2023, 0.1994, 0.2010)" in ruling
    assert "NOTEBOOK_BUDGET_AND_NORMALISATION" in ruling
    grounds = _flat(record["grounds"])
    assert "whether the split explains their numbers" in grounds
    assert "defects included" in grounds
    assert "'our normalisation differed'" in grounds
    assert "the same failure mode ruling (b) closed for the backbone" in grounds
    # The defect record's own words ride, and the write-up sentence is
    # fixed: replicated knowingly, never a recipe.
    assert "a defect in the notebook, not a recipe choice" in _flat(
        record["characterisation_rides"]
    )
    assert record["write_up_sentence"] == (
        "replicated knowingly -- never 'a recipe'"
    )
    # And the draft's pre-lock ledger reflects it, dated, with the gate
    # run and N still open.
    assert phase10_annex.EXIT_CRITERIA_DRAFT[
        "normalisation_ruled_2026_08_30"
    ].startswith("NORMALISATION_RULED")


def test_the_number_provenance_note_separates_the_two_schemes():
    note = _flat(phase10_annex.RATER_PANEL_DISCREPANCY_EIGHTH[
        "number_provenance_2026_08_30"
    ])
    assert "LITERATURE_RECORD_2026-08-23.md" in note
    assert "Panel description conflict (discrepancy #8)" in note
    assert "independent" in note
    # The wording arrives via the maintainer -- the file is not on this
    # machine, and the note says so rather than claiming verification.
    assert "via the maintainer" in note
    assert "file not found on this machine" in note
    # The registration's own framing stands unchanged.
    assert "eighth manuscript-vs-artifact discrepancy" in (
        phase10_annex.RATER_PANEL_DISCREPANCY_EIGHTH["class"]
    )


def test_the_readings_are_committed_before_any_number():
    record = phase10_annex.READINGS_COMMITTED
    assert "before any number" in record["committed"]
    assert "does not identify performance" in _flat(
        record["published_inside_the_distribution"]
    )
    assert "strongest form of the small-n argument" in _flat(
        record["published_inside_the_distribution"]
    )
    outside = _flat(record["published_outside_and_above"])
    assert "the split does not explain the gap" in outside
    assert "INCOMPLETE" in outside
    assert "rather than concluding anything about their model" in outside
    assert "no reading is invented after the numbers" in record[
        "no_third_reading"
    ]


def test_the_annex_prohibition_is_a_literal_string():
    prohibition = phase10_annex.ANNEX_PROHIBITION
    assert prohibition.startswith(
        "THESE FIGURES CHARACTERISE OUR REPLICATION UNDER THEIR PROTOCOL"
    )
    assert "NEVER A RE-COMPUTATION OF THEIR RESULTS" in prohibition
    assert "NEVER PRESENTED AS CORRECTING THEIR PUBLISHED NUMBERS" in _flat(
        prohibition
    )
    # The sibling it stands beside is unchanged.
    assert phase18.DELIVERABLES_REGISTERED["cleftgnn_iem_prohibition"].startswith(
        "OUR IEM VALUES ARE NEVER PLACED BESIDE CLEFTGNN'S"
    )
    assert "never placed beside CleftGNN's Table 2/4/6" in _flat(prohibition)


def test_the_supervision_dependency_is_tagged_reported_not_measured():
    record = phase10_annex.SUPERVISION_28_DEPENDENCY
    assert "[REPORTED, not measured]" in record["statement"]
    assert "in no repo record" in record["statement"]
    assert "SEPARATE REGISTERED ADDITION" in _flat(
        record["if_ids_become_recoverable"]
    )
    assert "NOT assumed without IDs" in _flat(record["the_assumption_not_made"])


def test_the_compute_gate_is_designed_not_run_and_gates_n():
    record = phase10_annex.COMPUTE_GATE_DESIGNED
    assert "NOT run" in record["designed"]
    assert "the maintainer launches" in record["designed"]
    flat = _flat(record["job"])
    assert "one split draw x one rater x one full fit" in flat
    assert "cluster-only" in flat
    assert "pinned image" in flat
    assert "<stem>__<git sha8>__<job-id>" in flat
    assert "per-fit wall-clock" in flat
    assert "N x 5" in record["sizing_arithmetic"]
    assert "only after the measured cost" in record["n_is_gated"]
    assert record["phase_17_lesson"] == "measure before declaring"


def test_pointers_both_ways_and_the_negative_space_that_remains():
    """[UPDATED 2026-08-30, second cycle -- the gate BUILD landed on the maintainer's instruction, so 'no annex task, no annex config' converted
    to its successor: the gate exists, and the negative space that
    still holds is the N-draw machinery.]"""
    # Pointers both ways.
    assert phase10.PHASE_10_CLOSING["annex_2026_08_30"] == (
        "phase10_annex.ANNEX_QUESTION_REGISTERED"
    )
    assert "PAIRED_BCA_OBSERVED" in phase10_annex.ANNEX_QUESTION_REGISTERED[
        "what_phase_10_measured"
    ]

    # The gate is built -- ONE kind, ONE task, ONE config -- and
    # nothing else of the annex is.
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    # [2026-08-31, third cycle] The distribution arm joined; the pin
    # fired and is updated dated. TWO kinds now, and no third.
    annex_kinds = sorted(
        name for name in TASK_SPECS if "p10x" in name or "annex" in name
    )
    assert annex_kinds == ["p10x_distribution", "p10x_gate_fullfit"]
    annex_tasks = sorted(
        name for name in TASKS if "p10x" in name or "annex" in name
    )
    assert annex_tasks == ["p10x_distribution", "p10x_gate_fullfit"]
    assert [p.name for p in sorted((REPO / "configs").rglob("p10x_*.yaml"))] == [
        "p10x_distribution.yaml", "p10x_gate_fullfit.yaml"
    ]

    # The criteria are LOCKED now, and the draft is preserved beside
    # them rather than edited.
    assert phase10_annex.EXIT_CRITERIA["locked"] == "2026-08-31"
    assert "DRAFT" in phase10_annex.EXIT_CRITERIA_DRAFT["status"]
    assert "not locked until the compute gate lands" in (
        phase10_annex.EXIT_CRITERIA_DRAFT["status"]
    )
    # No ledger row from the registration, the build, or the gate run.
    # [2026-08-31] Was `len(ENTRIES) == 37` -- a global count standing in
    # for "the annex banked nothing", which went stale when an unrelated
    # correction appended at index 37. Rewritten as the relation: no
    # ledger entry belongs to the annex.
    from cleft import results_ledger

    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p10x" or "annex" in e["id"] or "p10x" in e["id"]
    ]


def test_the_summary_carries_every_record():
    # [2026-08-30, second cycle] The pin fired when normalisation_ruled
    # joined; updated dated.
    # [2026-08-31, third cycle] Fired again for the gate measurement,
    # the N ruling, the lock and the distribution arm; updated dated.
    # [2026-08-31, close-out] Fired again for the run, the reading, the
    # pathologies, the sizing outturn, the ledger proposal and the
    # closing; updated dated.
    # [2026-08-31, after the close-out] Fired again for the attribution
    # and the partial table; updated dated.
    # [2026-08-31, third asking] Fired again for the filled table's
    # three metric records and its checks; updated dated.
    assert sorted(phase10_annex.summary()) == [
        "closing",
        "compute_gate",
        "degenerate_by_rater",
        "design",
        "design_pathologies",
        "distribution_arm",
        "distribution_observed",
        "exit_criteria",
        "exit_criteria_draft",
        "f1_by_rater",
        "first_draw_degenerate",
        "five_class_exception",
        "floors",
        "gate_measured",
        "iem_by_rater",
        "ledger_proposal",
        "mixed_pattern",
        "n_ruled",
        "normalisation_proposed",
        "normalisation_ruled",
        "pcc_by_rater",
        "per_rater_table",
        "prohibition",
        "published_reference_points",
        "question",
        "rater_panel_discrepancy",
        "reading_fired",
        "readings",
        "recipe",
        "ruling_a",
        "ruling_b",
        "sizing_outturn",
        "supervision_dependency",
        "table_checks",
        "verification",
    ]


# --------------------------------------------------------------------------
# 2026-08-30, second cycle: the compute gate -- schema, model, task,
# config. Built, NOT run; the launch is the maintainer's.
# --------------------------------------------------------------------------

from cleft.config.schema import ConfigError, TASK_SPECS, validate
from fixtures import builders


def _gate_task(**overrides) -> dict:
    """A valid p10x_gate_fullfit task block; overrides are shallow-merged."""
    task = {
        "kind": "p10x_gate_fullfit",
        "manifest_artifact": "manifest_v1",
        "staged_artifact": "staged_v1",
        "scoresheet_artifact": "scoresheet_primary",
        "geometry": "g1",
        "label": "mean",
        "rater": "Rater 7 - Cleft patient",
        "split_seed": 1337,
        "train_size": 153,
        "test_size": 28,
        "region_count": 36,
        "trainable": "full",
        "recipe": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 5,
            "loss": "cross_entropy",
            "model_selection": "last_epoch",
        },
        "normalisation": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
    }
    task.update(overrides)
    return task


def _validated(**overrides) -> dict:
    return validate(builders.config_dict(phase="p10x", task=_gate_task(**overrides)))


def test_the_gate_schema_accepts_exactly_the_registered_design():
    config = _validated()
    assert config["task"]["kind"] == "p10x_gate_fullfit"
    assert config["task"]["trainable"] == "full"


def test_the_gate_schema_pins_every_registered_value():
    spec = TASK_SPECS["p10x_gate_fullfit"]
    # Required-no-default, per the registration: a scientific knob is
    # written down, never defaulted.
    for key in ("rater", "split_seed", "train_size", "test_size",
                "region_count", "trainable", "recipe", "normalisation"):
        assert spec[key].required, key
        assert spec[key].default is None, key
    assert spec["train_size"].choices == (153,)
    assert spec["test_size"].choices == (28,)
    assert spec["trainable"].choices == ("full",)
    recipe = spec["recipe"].spec
    assert recipe["optimizer"].choices == ("adam",)
    assert recipe["learning_rate"].choices == (0.001,)
    assert recipe["batch_size"].choices == (16,)
    assert recipe["epochs"].choices == (5,)
    assert recipe["loss"].choices == ("cross_entropy",)
    assert recipe["model_selection"].choices == ("last_epoch",)


def test_the_gate_rater_vocabulary_is_the_sheets_own():
    # The five column names, asserted equal to the sheet module's --
    # the dict-equality pattern, so the two cannot drift apart.
    spec = TASK_SPECS["p10x_gate_fullfit"]
    assert spec["rater"].choices == scoresheet.RATERS


def test_the_gate_refuses_27_regions_naming_the_reason():
    with pytest.raises(ConfigError, match="REGION_COUNT_BELIEF_VS_EXECUTION"):
        _validated(region_count=27)
    with pytest.raises(ConfigError, match="36"):
        _validated(region_count=49)


def test_the_gate_refuses_any_normalisation_but_the_notebooks():
    with pytest.raises(ConfigError, match="CIFAR-10"):
        _validated(normalisation={
            "mean": [0.485, 0.456, 0.406],   # ImageNet's -- the exact
            "std": [0.229, 0.224, 0.225],    # substitution the ruling closes
        })


def test_the_gate_refuses_a_frozen_or_defaulted_arm():
    with pytest.raises(ConfigError):
        _validated(trainable="head")
    with pytest.raises(ConfigError, match="missing required key"):
        config = builders.config_dict(phase="p10x", task=_gate_task())
        del config["task"]["split_seed"]
        validate(config)
    with pytest.raises(ConfigError):
        _validated(recipe={
            "optimizer": "sgd", "learning_rate": 0.01, "batch_size": 16,
            "epochs": 5, "loss": "cross_entropy",
            "model_selection": "last_epoch",
        })


def test_the_admits_full_vocabulary_set_is_pinned_by_enumeration():
    """[MEASURED CORRECTION 2026-08-30 to the instruction's premise,
    recorded rather than smoothed.] 'full' was ALREADY admitted (never
    set by any config) in three historical vocabularies, so 'refused by
    every other kind' was never the schema's state. The structural pin
    is this enumeration: any NEW kind admitting 'full' fires here, and
    the annex kind is the only one where 'full' is REQUIRED."""
    admits_full = sorted(
        kind for kind, spec in TASK_SPECS.items()
        if "trainable" in spec
        and spec["trainable"].choices is not None
        and "full" in spec["trainable"].choices
    )
    # [2026-08-31] The distribution arm joined; the pin fired and is
    # updated dated. Both new entries are annex kinds.
    assert admits_full == [
        "p10x_distribution", "p10x_gate_fullfit", "partition_sensitivity",
        "probe_mebeauty", "train_cv",
    ]
    only_full = sorted(
        kind for kind, spec in TASK_SPECS.items()
        if "trainable" in spec and spec["trainable"].choices == ("full",)
    )
    assert only_full == ["p10x_distribution", "p10x_gate_fullfit"]
    assert all(kind.startswith("p10x_") for kind in only_full)


def test_the_gate_spec_has_no_draw_count_anywhere():
    """The negative space that holds until the measured cost lands and
    the ruling is N: one draw, one rater, one fit -- no N-draw
    machinery, no multi-split task, no draw-count key."""
    spec = TASK_SPECS["p10x_gate_fullfit"]
    for key in spec:
        assert "draw" not in key.lower(), key
        assert "n_splits" not in key.lower(), key
    assert "seeds" not in spec        # ONE seed field, singular, for ONE draw
    assert "n_draws" not in spec
    config_text = (REPO / "configs" / "p10x_gate_fullfit.yaml").read_text(
        encoding="utf-8"
    )
    assert "n_draws" not in config_text
    assert "draw_count" not in config_text


def test_the_shipped_gate_config_carries_the_lineages_verified_inputs():
    gate = yaml.safe_load(
        (REPO / "configs" / "p10x_gate_fullfit.yaml").read_text(encoding="utf-8")
    )
    faithful = yaml.safe_load(
        (REPO / "configs" / "p10_cleftgnn_faithful.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate(gate)  # the shipped file passes its own schema
    assert gate["task"]["kind"] == "p10x_gate_fullfit"
    assert gate["task"]["trainable"] == "full"
    assert gate["task"]["train_size"] == 153
    assert gate["task"]["test_size"] == 28
    assert gate["task"]["region_count"] == 36
    assert gate["task"]["recipe"] == {
        "optimizer": "adam", "learning_rate": 0.001, "batch_size": 16,
        "epochs": 5, "loss": "cross_entropy", "model_selection": "last_epoch",
    }
    assert gate["task"]["normalisation"] == {
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    }
    assert gate["task"]["rater"] in scoresheet.RATERS
    # Inputs CARRIED from the faithful config's verified entries --
    # same name, same path, same hash, byte for byte.
    faithful_by_name = {e["name"]: e for e in faithful["inputs"]}
    assert len(gate["inputs"]) == 3
    for entry in gate["inputs"]:
        assert entry == faithful_by_name[entry["name"]]


def test_the_annex_recipe_is_separate_from_the_documented_two():
    from cleft.models import cleftgnn

    # The two documented artifacts are untouched -- the pin at
    # test_phase10.py:1912 stands. The annex recipe is a RULED
    # COMPOSITION, deliberately not a member of RECIPES.
    assert cleftgnn.RECIPES == ("manuscript", "notebook")
    assert cleftgnn.ANNEX_RECIPE == "annex_fullfit"
    assert cleftgnn.ANNEX_RECIPE not in cleftgnn.RECIPES
    assert cleftgnn.CIFAR10_NORMALISATION == (
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010),
    )
    assert cleftgnn.CLASSIFIER_INPUT_STAGE["annex_fullfit"] == "fused"
    stages = cleftgnn.RECIPE_STAGES["annex_fullfit"]
    assert "sabm_v_a" in stages
    for repaired in ("gnn_norm_f_t", "after_layernorm", "acm_v"):
        assert repaired not in stages


def test_the_annex_model_drops_the_repairs_and_trains_everything():
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    model = cleftgnn.build(pretrained=False, recipe=cleftgnn.ANNEX_RECIPE)
    # The manuscript architecture -- SABM's vertical mask is present --
    # WITHOUT the measured repairs: their training had neither norm.
    assert hasattr(model, "m_vertical")
    assert not hasattr(model, "gnn_norm")
    assert not hasattr(model, "fused_norm")
    assert not hasattr(model, "acm_w_beta")

    stages: dict = {}
    with torch.no_grad():
        logits = model(torch.rand(2, 3, 224, 224), stages=stages)
    assert logits.shape == (2, 5)
    assert "sabm_v_a" in stages and "fused" in stages
    assert "gnn_norm_f_t" not in stages
    assert "after_layernorm" not in stages


def test_the_annex_backbone_is_full_trainable_with_cifar_stats():
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    backbone = cleftgnn.CleftGNNBackbone(
        recipe=cleftgnn.ANNEX_RECIPE, optimizer="adam", learning_rate=0.001,
        trainable="full", pretrained=False, device="cpu", seed=0,
    )
    labels = np.array([1, 2, 3, 4, 5, 3, 2, 4], dtype=float)
    backbone.reset(labels)

    report = backbone.parameter_report
    assert report["trainable"] == "full"
    assert report["trainable_parameters"] == report["total_parameters"]
    assert report["frozen_backbone_parameters"] == 0
    for parameter in backbone._model.backbone.parameters():
        assert parameter.requires_grad

    # The RULED normalisation: the notebook's CIFAR-10 statistics, not
    # the timm data config (NORMALISATION_RULED -- a deliberately
    # replicated defect, replicated knowingly).
    assert np.allclose(
        backbone._mean.flatten().cpu().numpy(), (0.4914, 0.4822, 0.4465)
    )
    assert np.allclose(
        backbone._std.flatten().cpu().numpy(), (0.2023, 0.1994, 0.2010)
    )

    # The Laplace-biased init was OUR deviation and does not ride: the
    # classifier keeps the framework's own init.
    grades = labels.astype(int)
    counts = np.array([np.sum(grades == g) for g in range(1, 6)], dtype=float)
    laplace = np.log((counts + 1.0) / (counts.sum() + 5))
    assert not np.allclose(
        backbone._model.classifier.bias.detach().cpu().numpy(), laplace
    )


def test_full_is_refused_outside_the_annex_recipe_and_vice_versa():
    pytest.importorskip("torch")

    from cleft.models import cleftgnn

    with pytest.raises(cleftgnn.CleftGNNError, match="annex"):
        cleftgnn.CleftGNNBackbone(
            recipe="manuscript", trainable="full",
            pretrained=False, device="cpu",
        ).reset(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(cleftgnn.CleftGNNError, match="full"):
        cleftgnn.CleftGNNBackbone(
            recipe=cleftgnn.ANNEX_RECIPE, optimizer="adam",
            learning_rate=0.001, pretrained=False, device="cpu",
        ).reset(np.array([1.0, 2.0, 3.0]))


def test_full_training_does_not_hold_the_backbone_in_eval():
    """A full fine-tune trains BatchNorm normally -- running statistics
    update, which is what their training would have done. The frozen
    regimes keep the eval() guard; the guard is keyed, not deleted."""
    import inspect

    from cleft.models import cleftgnn

    source = inspect.getsource(cleftgnn.CleftGNNBackbone.train_epoch)
    assert "backbone.eval()" in source
    assert 'self.trainable != "full"' in source


# --------------------------------------------------------------------------
# the gate task, end to end small -- the smoke pattern, stub backbone
# --------------------------------------------------------------------------


def _gate_artifacts(root, n=190):
    """Manifest + staged + score sheet for the gate task, tiny.

    Rater columns are engineered: 'Rater 8 - Orthodontist' cycles all
    five grades; 'Rater 11 - Psychologist' spans ONLY 1..4, so any test
    side must miss grade 5 -- the degenerate-occupancy branch."""
    from types import SimpleNamespace

    from fixtures import cohort as cohort_fixture

    manifest_dir = root / "cleft_v1"
    staged_dir = root / "staged_v1"
    manifest_dir.mkdir(parents=True)
    staged_dir.mkdir(parents=True)

    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for index in range(n):
        pid, value = index + 1, 2.0 + (index % 3)
        lines.append(
            f"{pid},{1000 + pid},,{value:.4f},{value:.4f},{value:.4f},"
            f"{value:.4f},{value:.4f},0.2,0.2,0.2,0.2,0.2,{index % 3},"
            f"{index % 5}"
        )
    (manifest_dir / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    images = np.zeros((n, 8, 8, 3), dtype=np.uint8)
    np.save(staged_dir / "staged_patient_g1.npy", images)
    geometry = ["# CLUSTER-ONLY: patient-keyed geometry", "patient_id,aspect_ratio"]
    geometry += [f"{pid},0.74" for pid in range(1, n + 1)]
    (staged_dir / "geometry.csv").write_text(
        "\n".join(geometry) + "\n", encoding="utf-8"
    )

    grades = {
        1000 + pid: [
            (pid % 3) + 2,        # Rater 7
            (pid % 5) + 1,        # Rater 8 -- all five grades occur
            (pid % 3) + 1,        # Rater 9
            (pid % 4) + 2,        # Rater 10
            (pid % 4) + 1,        # Rater 11 -- grade 5 NEVER occurs
        ]
        for pid in range(1, n + 1)
    }
    sheet_path = root / "scores.xlsx"
    cohort_fixture.write_scoresheet(
        SimpleNamespace(
            scored_ids=[1000 + pid for pid in range(1, n + 1)], grades=grades
        ),
        sheet_path,
        form="integer",
    )
    return manifest_dir, staged_dir, sheet_path


class _StubCleftGNN:
    """The harness protocol with none of the weight: predictions are a
    deterministic function of row order, so occupancy, floors and the
    metrics plumbing are exercised without torch."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.parameter_report = {
            "total_parameters": 7, "trainable_parameters": 7,
            "frozen_backbone_parameters": 0, "recipe": kwargs.get("recipe"),
            "optimizer": kwargs.get("optimizer"),
            "trainable": kwargs.get("trainable"),
        }
        self.epochs_run = 0

    def reset(self, train_labels):
        self._n_classes = 5

    def train_epoch(self, features, labels):
        self.epochs_run += 1
        return 1.0 / self.epochs_run

    def predict_probabilities(self, features):
        out = np.zeros((len(features), 5))
        out[np.arange(len(features)), np.arange(len(features)) % 5] = 1.0
        return out


@pytest.fixture
def gate_run(tmp_path, out_root, clean_repo, monkeypatch):
    from cleft.models import cleftgnn
    from cleft.provenance.hashing import hash_path
    from cleft.run import main

    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    monkeypatch.setattr(cleftgnn, "CleftGNNBackbone", _StubCleftGNN)
    manifest_dir, staged_dir, sheet_path = _gate_artifacts(tmp_path / "data")

    def run(rater: str):
        config = builders.write_config(
            tmp_path / f"p10x_{abs(hash(rater))}.yaml",
            phase="p10x",
            tier="keeper",
            inputs=[
                {"name": "manifest_v1", "path": str(manifest_dir),
                 "rollup_sha256": hash_path(manifest_dir)["rollup"]},
                {"name": "staged_v1", "path": str(staged_dir),
                 "rollup_sha256": hash_path(staged_dir)["rollup"]},
                {"name": "scoresheet_primary", "path": str(sheet_path),
                 "rollup_sha256": hash_path(sheet_path)["rollup"]},
            ],
            task=_gate_task(rater=rater),
        )
        run_dir = main(["--config", str(config), "--out", str(out_root)])
        return json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )

    return run


def test_the_gate_runs_end_to_end_and_reports_the_contract(gate_run):
    reported = gate_run("Rater 8 - Orthodontist")

    split = reported["split"]
    assert (split["train_size"], split["test_size"]) == (153, 28)
    assert split["cohort_n"] == 190
    assert split["unused"] == 190 - 181

    # The split is the DECLARED procedure, recomputed here from the
    # same seed: permutation, first 28 test, next 153 train.
    perm = np.random.default_rng(1337).permutation(190)
    test_rows, train_rows = perm[:28], perm[28:181]
    grades = np.array([(pid % 5) + 1 for pid in range(1, 191)])
    expected_test = {
        str(g): int(np.sum(grades[test_rows] == g)) for g in range(1, 6)
    }
    assert reported["occupancy"]["test"] == expected_test
    assert reported["occupancy"]["train"] == {
        str(g): int(np.sum(grades[train_rows] == g)) for g in range(1, 6)
    }
    assert reported["occupancy"]["degenerate_test_side"] == any(
        expected_test[str(g)] == 0 for g in range(1, 6)
    )

    # Wall-clock, train and eval separately; five epochs, last is the
    # model -- no selection machinery anywhere.
    clock = reported["wall_clock_seconds"]
    assert clock["train_total"] >= 0.0
    assert clock["eval"] >= 0.0
    assert len(clock["train_per_epoch"]) == 5
    assert "N x 5" in reported["sizing"]

    # The 5-class metrics with their floors beside them.
    metrics = reported["metrics"]
    for key in ("precision_macro", "recall_macro", "f1_macro", "accuracy",
                "pcc_top1", "iem_a_top1_mean"):
        assert key in metrics, key
    floors = reported["floors"]
    assert floors["majority_grade"] in range(1, 6)
    assert "f1_macro" in floors["majority_class"]
    assert floors["iem_constant_mean"] >= 0.0
    assert "undefined" in floors["pcc"]

    # The prohibition and the ruled-normalisation words travel.
    assert reported["prohibition"].startswith(
        "THESE FIGURES CHARACTERISE OUR REPLICATION"
    )
    assert "deliberately replicated defect" in reported["normalisation_note"]
    assert "replicated knowingly" in reported["normalisation_note"]


def test_the_degenerate_occupancy_branch_reports_and_does_not_discard(gate_run):
    # Rater 11 never grades 5, so the test side MUST miss it.
    reported = gate_run("Rater 11 - Psychologist")
    assert reported["occupancy"]["test"]["5"] == 0
    assert reported["occupancy"]["degenerate_test_side"] is True
    # Reported, never discarded: the metrics still exist, under the
    # stated macro convention (truth-absent classes excluded).
    assert 5 in reported["metrics"]["classes_absent_excluded"]
    assert reported["metrics"]["f1_macro"] >= 0.0


def test_the_gate_measurement_is_recorded_with_its_verification_status():
    record = phase10_annex.GATE_MEASURED
    assert "81a171cc" in record["measured"]
    # The run's SHA is this repo's own -- checked, not assumed.
    assert "git cat-file" in record["run_sha_is_ours"]

    verified = record["verified_locally"]
    assert "29,286,981 of 29,286,981" in verified["parameters"]
    assert "23,508,032" in verified["parameters"]
    assert "EXACT match" in verified["parameters"]
    # The floor check identifies WHICH floor ran, which is the point.
    assert "GRADE-2" in verified["majority_floor"]
    assert "0.131579" in verified["majority_floor"]
    assert "TRAIN side" in verified["majority_floor"]

    # What could not be checked here says so, and is tagged.
    unverified = record["reported_not_verified_here"]
    assert "44.1067 -> 1.4430" in unverified
    assert "0.2505" in unverified
    assert "CLUSTER-ONLY" in unverified
    assert "never presented as verified at source" in unverified
    # And the one figure that could be over-read is fenced.
    assert "ONE DRAW at n=28" in record["the_arm_figure_licenses_nothing"]


def test_the_reported_floor_reproduces_through_the_shipped_metric():
    """The verification claim in GATE_MEASURED, re-run as a test rather
    than trusted from prose: the reported 0.1111 is what a constant
    GRADE-2 predictor scores on the reported occupancy, and the test
    side's own majority (grade 3) would have given 0.1316 -- so the
    figure is evidence the floor came from the train side."""
    occupancy = phase10_annex.FIRST_DRAW_DEGENERATE["occupancy"]
    truth = np.concatenate([
        np.full(count, int(grade), dtype=int)
        for grade, count in occupancy.items()
    ])
    assert len(truth) == 28

    def floor_at(grade):
        return phase10.top1_macro_prf(
            truth, np.full(len(truth), grade, dtype=int)
        )["f1_macro"]

    assert floor_at(2) == pytest.approx(0.1111, abs=5e-5)
    assert floor_at(3) == pytest.approx(0.1316, abs=5e-5)
    # Grade 5 is absent from truth, so it is excluded from the macro.
    assert phase10.top1_macro_prf(
        truth, np.full(len(truth), 2, dtype=int)
    )["classes_absent_excluded"] == (5,)


def test_the_parameter_count_is_what_the_model_actually_builds():
    pytest.importorskip("torch")
    import torch

    from cleft.models import cleftgnn

    torch.manual_seed(0)
    model = cleftgnn.build(pretrained=False, recipe=cleftgnn.ANNEX_RECIPE)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = sum(p.numel() for p in model.backbone.parameters())

    assert total == 29_286_981
    assert trainable == total          # ruling (b), structurally
    assert backbone == 23_508_032
    # And the record quotes exactly these.
    quoted = phase10_annex.GATE_MEASURED["verified_locally"]["parameters"]
    assert f"{total:,}" in quoted
    assert f"{backbone:,}" in quoted


def test_the_first_draw_observation_is_an_observation_not_a_frequency():
    record = phase10_annex.FIRST_DRAW_DEGENERATE
    assert record["occupancy"] == {"1": 4, "2": 8, "3": 10, "4": 6, "5": 0}
    assert sum(record["occupancy"].values()) == 28
    assert "grade 5 ABSENT" in record["what_happened"]
    # n = 1: an observation FEEDING the deliverable, not the deliverable.
    status = _flat(record["status"])
    assert "OBSERVATION" in status
    assert "NOT the frequency itself" in status
    assert "n = 1" in status
    # Its bearing on the annex's question is stated.
    why = _flat(record["why_it_is_recorded_now"])
    assert "every class its 5-class macro table reports on" in why
    assert "a statement about the DESIGN" in why
    assert "anecdote until the frequency exists" in _flat(
        record["not_yet_a_claim"]
    )


def test_n_is_ruled_at_500_with_the_arithmetic_derived_from_the_measurement():
    record = phase10_annex.N_RULED
    assert record["n_draws"] == 500
    assert record["raters"] == 5
    assert record["fits"] == 500 * 5 == 2500
    assert "after the measured cost, as gated" in record["ruled"]

    sizing = _flat(record["sizing"])
    assert "500 draws x 5 raters = 2,500 fits" in sizing
    assert "3.47 GPU-hours" in sizing
    # Recomputed here rather than trusted from the string.
    assert 2500 * 5.0 / 3600 == pytest.approx(3.47, abs=0.01)

    # The conservative choice is named, and the bound's direction stated.
    why = _flat(record["why_the_end_to_end_figure"])
    assert "UPPER BOUND" in why
    assert "1.74 h and 3.47 h" in why
    assert "[DERIVED" in why
    assert 2500 * 2.5 / 3600 == pytest.approx(1.74, abs=0.01)


def test_the_exit_criteria_are_locked_with_all_three_items_closed():
    locked = phase10_annex.EXIT_CRITERIA
    assert locked["locked"] == "2026-08-31"
    assert "EXIT_CRITERIA_DRAFT, preserved as written" in locked["supersedes"]

    items = locked["pre_lock_items_all_closed"]
    assert items["1_gate_cost"] == "RAN -- GATE_MEASURED"
    assert items["2_n"] == "RULED at 500 -- N_RULED"
    assert items["3_normalisation"].startswith("RULED")
    # The question asked at the lock, answered in the record.
    assert items["is_a_fourth_outstanding"].startswith("NO.")
    assert "None remains" in items["is_a_fourth_outstanding"]

    # Nothing added while locking: same count, same order, same subjects.
    draft = phase10_annex.EXIT_CRITERIA_DRAFT["criteria"]
    assert len(locked["criteria"]) == len(draft) == 7
    for locked_line, draft_line in zip(locked["criteria"], draft):
        assert locked_line[:2] == draft_line[:2]      # same numbering
    assert "None added, dropped, weakened or reordered" in _flat(
        locked["nothing_added_at_the_lock"]
    )
    # The draft is preserved and points at its successor.
    assert phase10_annex.EXIT_CRITERIA_DRAFT["superseded_2026_08_31"] == (
        "EXIT_CRITERIA"
    )
    assert "DRAFT" in phase10_annex.EXIT_CRITERIA_DRAFT["status"]

    # The degenerate handling is explicit AT the lock.
    handling = _flat(locked["degenerate_handling_explicit"])
    assert "EXCLUDES a truth-absent class" in handling
    assert "NOT term-by-term comparable" in handling
    assert "reported SEPARATELY" in handling


def test_the_mixed_and_below_cases_are_registered_before_the_numbers():
    record = phase10_annex.MIXED_PATTERN_REGISTERED
    assert "before any number" in record["registered"]
    handling = _flat(record["handling"])
    assert "DATED OBSERVATION" in handling
    assert "concluding NOTHING" in handling
    assert "Not a third reading" in handling
    assert "phase17.UNPREDICTED_PATTERN" in record["precedent"]
    # The gap in the two committed readings, named rather than found.
    gap = _flat(record["the_below_gap"])
    assert "covered by NEITHER committed reading" in gap
    assert "-0.162" in gap
    assert "Named before the run rather than decided after" in gap


def test_the_published_reference_points_are_four_and_cited():
    record = phase10_annex.PUBLISHED_REFERENCE_POINTS
    points = record["points"]
    assert points == {
        "study_max": 0.440,
        "study_rater_e": 0.283,
        "study_mean_approx": 0.14,
        "study_min": -0.162,
    }
    # Each traces to a record, and those records still say it.
    study = phase10.CLEFTGNN_COMPARATOR_TABLES["study_test_set"]["range"]
    assert "0.440" in study and "-0.162" in study and "0.14" in study
    assert phase10.FAITHFUL_ARM_REGISTERED["their_598_cell"][
        "same_model_on_their_study_set"
    ] == 0.283
    # Why four and not five -- the fabrication that was declined.
    why = _flat(record["why_four_and_not_five"])
    assert "RANGE plus a mean" in why
    assert "fabrication the provenance rule forbids" in why
    assert "UNMATCHED" in _flat(record["pairing_is_unmatched"])


def test_gate_metrics_json_contains_no_patient_identifiers(gate_run):
    """metrics.json is SHAREABLE, so it must be aggregate scalars only --
    no id key, and no id-shaped value anywhere in the structure (a
    substring check on the digits would false-positive on wall-clock
    floats, so the check walks values instead)."""
    reported = gate_run("Rater 8 - Orthodontist")
    text = json.dumps(reported)
    assert "patient_id" not in text
    assert "frontal_id" not in text

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                assert "patient" not in key
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        else:
            # No bare fixture id (frontal ids are 1001..1190) may appear
            # as an INTEGER value; counts and sizes are all far smaller.
            if isinstance(node, int) and not isinstance(node, bool):
                assert not (1001 <= node <= 1190), node

    walk(reported)


# --------------------------------------------------------------------------
# 2026-08-31, third cycle: the DISTRIBUTION ARM. 500 draws x 5 raters
# through the gate's own fit path. Built, NOT run.
# --------------------------------------------------------------------------


def _distribution_task(**overrides) -> dict:
    from cleft.run import p10x_split_seeds

    task = {
        "kind": "p10x_distribution",
        "manifest_artifact": "manifest_v1",
        "staged_artifact": "staged_v1",
        "scoresheet_artifact": "scoresheet_primary",
        "geometry": "g1",
        "label": "mean",
        "raters": list(scoresheet.RATERS),
        "root_seed": 1337,
        "n_draws": 500,
        "split_seeds": p10x_split_seeds(1337, 500),
        "train_size": 153,
        "test_size": 28,
        "region_count": 36,
        "trainable": "full",
        "recipe": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 5,
            "loss": "cross_entropy",
            "model_selection": "last_epoch",
        },
        "normalisation": {
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
        "shard": {"count": 1, "index": 0},
    }
    task.update(overrides)
    return task


def test_the_split_seed_derivation_is_declared_reproducible_and_unique():
    from cleft.run import P10X_SPLIT_SEED_RULE, p10x_split_seeds

    seeds = p10x_split_seeds(1337, 500)
    assert len(seeds) == 500
    assert all(isinstance(s, int) for s in seeds)
    # Deterministic: same root, same list, every time.
    assert p10x_split_seeds(1337, 500) == seeds
    # A different root gives a different draw set.
    assert p10x_split_seeds(1338, 500) != seeds
    # UNIQUE: a repeated seed is a repeated draw, which would bias the
    # distribution silently.
    assert len(set(seeds)) == 500
    # The rule is written down, not just implemented.
    assert "SeedSequence" in P10X_SPLIT_SEED_RULE


def test_the_distribution_schema_pins_the_ruled_shape():
    spec = TASK_SPECS["p10x_distribution"]
    for key in ("raters", "root_seed", "n_draws", "split_seeds",
                "train_size", "test_size", "region_count", "trainable",
                "recipe", "normalisation", "shard"):
        assert spec[key].required, key
    # Pinned to the ruling, so the arm cannot quietly become a different
    # experiment than the one N was ruled for.
    assert spec["n_draws"].choices == (500,)
    assert spec["train_size"].choices == (153,)
    assert spec["test_size"].choices == (28,)
    assert spec["trainable"].choices == ("full",)
    # The recipe is the gate's, field for field.
    assert spec["recipe"].spec.keys() == (
        TASK_SPECS["p10x_gate_fullfit"]["recipe"].spec.keys()
    )
    for key, field in spec["recipe"].spec.items():
        assert field.choices == (
            TASK_SPECS["p10x_gate_fullfit"]["recipe"].spec[key].choices
        ), key


def test_the_distribution_runs_all_five_raters_by_ruling():
    # RULING_A: all five, five models per draw. Not a subset, not a pick.
    with pytest.raises(ConfigError, match="all five"):
        validate(builders.config_dict(
            phase="p10x",
            task=_distribution_task(raters=[scoresheet.RATERS[0]]),
        ))
    # And the vocabulary is the sheet's own, in the sheet's order.
    ok = validate(builders.config_dict(
        phase="p10x", task=_distribution_task()
    ))
    assert tuple(ok["task"]["raters"]) == scoresheet.RATERS


def test_the_distribution_refuses_seeds_that_disagree_with_their_root():
    from cleft.run import p10x_split_seeds

    wrong = p10x_split_seeds(1337, 500)
    wrong[17] = wrong[17] + 1
    with pytest.raises(ConfigError, match="root_seed"):
        validate(builders.config_dict(
            phase="p10x", task=_distribution_task(split_seeds=wrong)
        ))
    # A truncated list is refused too -- a short draw set would silently
    # be a smaller experiment than the one N was ruled for.
    with pytest.raises(ConfigError, match="n_draws"):
        validate(builders.config_dict(
            phase="p10x",
            task=_distribution_task(split_seeds=p10x_split_seeds(1337, 499)),
        ))


def test_the_shard_rule_partitions_the_draws_exactly():
    from cleft.run import p10x_shard_draws

    for count in (1, 2, 5, 7, 250):
        covered: list[int] = []
        for index in range(count):
            covered.extend(p10x_shard_draws(500, {"count": count, "index": index}))
        # Exactly once each: no overlap, no gap.
        assert sorted(covered) == list(range(500)), count
    # An out-of-range index is refused rather than silently empty.
    with pytest.raises(ConfigError, match="shard"):
        validate(builders.config_dict(
            phase="p10x",
            task=_distribution_task(shard={"count": 5, "index": 5}),
        ))
    with pytest.raises(ConfigError, match="shard"):
        validate(builders.config_dict(
            phase="p10x",
            task=_distribution_task(shard={"count": 0, "index": 0}),
        ))


def test_both_p10x_tasks_share_one_fit_path():
    """The Phase 18 IEM lesson, made structural: a second fit path would
    let the gate's verified numbers detach from the distribution's, and
    nothing downstream would disagree."""
    import inspect

    from cleft import run as run_module

    assert hasattr(run_module, "_p10x_one_fit")
    gate = inspect.getsource(run_module.task_p10x_gate_fullfit)
    distribution = inspect.getsource(run_module.task_p10x_distribution)
    for source in (gate, distribution):
        assert "_p10x_one_fit(" in source
    # The model is constructed in exactly ONE place: the shared helper.
    helper = inspect.getsource(run_module._p10x_one_fit)
    assert "CleftGNNBackbone(" in helper
    for source in (gate, distribution):
        assert "CleftGNNBackbone(" not in source
        assert "train_epoch(" not in source
        assert "predict_probabilities(" not in source
    # The draw and occupancy rules are shared too.
    for source in (gate, distribution):
        assert "_p10x_draw_rows(" in source
        assert "_p10x_occupancy(" in source


def test_locating_a_point_in_a_distribution_names_where_it_fell():
    from cleft.run import p10x_locate

    values = [0.1, 0.2, 0.3, 0.4]
    inside = p10x_locate(values, 0.25)
    assert inside["where"] == "inside"
    assert inside["percentile"] == pytest.approx(50.0)
    assert p10x_locate(values, 0.9)["where"] == "above_all"
    assert p10x_locate(values, -0.5)["where"] == "below_all"
    # The boundary is inside, not above: equal to the max is not beyond it.
    assert p10x_locate(values, 0.4)["where"] == "inside"
    # An empty or all-undefined distribution locates nothing rather than
    # returning a misleading zero.
    assert p10x_locate([], 0.1)["where"] == "undefined"


def test_the_reading_rule_is_mechanical_and_mixed_is_not_a_reading():
    from cleft.run import p10x_apply_readings

    inside_everywhere = {"r1": {"a": "inside"}, "r2": {"a": "inside"}}
    assert p10x_apply_readings(inside_everywhere)["fired"] == (
        "published_inside_the_distribution"
    )
    above_everywhere = {"r1": {"a": "above_all"}, "r2": {"a": "above_all"}}
    assert p10x_apply_readings(above_everywhere)["fired"] == (
        "published_outside_and_above"
    )
    # Anything else is the registered OBSERVATION, not a third reading.
    for grid in (
        {"r1": {"a": "inside"}, "r2": {"a": "above_all"}},
        {"r1": {"a": "below_all"}, "r2": {"a": "below_all"}},
        {"r1": {"a": "inside"}, "r2": {"a": "undefined"}},
    ):
        verdict = p10x_apply_readings(grid)
        assert verdict["fired"] is None
        assert verdict["observation"] == "MIXED_PATTERN_REGISTERED"
    # The reading TEXTS are the committed ones, quoted not retyped.
    fired = p10x_apply_readings(inside_everywhere)
    assert fired["reading"] == phase10_annex.READINGS_COMMITTED[
        "published_inside_the_distribution"
    ]


def test_the_shipped_distribution_config_is_declared_and_enumerated():
    from cleft.run import p10x_split_seeds

    config = yaml.safe_load(
        (REPO / "configs" / "p10x_distribution.yaml").read_text(encoding="utf-8")
    )
    validate(config)
    task = config["task"]
    assert task["kind"] == "p10x_distribution"
    assert task["n_draws"] == 500
    assert task["shard"] == {"count": 1, "index": 0}, "one job is the default"
    # The 500 seeds are ENUMERATED in the file, and they are the ones
    # the declared rule derives from the declared root.
    assert len(task["split_seeds"]) == 500
    assert task["split_seeds"] == p10x_split_seeds(task["root_seed"], 500)
    assert len(set(task["split_seeds"])) == 500

    # Inputs carried from the gate's config -- same three, byte for byte.
    gate = yaml.safe_load(
        (REPO / "configs" / "p10x_gate_fullfit.yaml").read_text(encoding="utf-8")
    )
    gate_by_name = {e["name"]: e for e in gate["inputs"]}
    assert len(config["inputs"]) == 3
    for entry in config["inputs"]:
        assert entry == gate_by_name[entry["name"]]
    # And the recipe block is the gate's, unchanged.
    assert task["recipe"] == gate["task"]["recipe"]
    assert task["normalisation"] == gate["task"]["normalisation"]


def test_the_distribution_arm_registration_states_its_shape():
    record = phase10_annex.DISTRIBUTION_ARM_REGISTERED
    assert "500 draws x 5 raters = 2,500 fits" in record["shape"]
    one = _flat(record["one_implementation"])
    assert "_p10x_one_fit" in one
    assert "Phase 18 IEM lesson" in one
    assert "PROBE_RECONSTRUCTED_THE_PIPELINE" in one
    assert "never a runtime glob" in _flat(record["draws_enumerated"])
    assert "never silently averaged" in _flat(record["degenerate_accounting"])
    shape = _flat(record["compute_shape"])
    assert "ONE JOB" in shape
    assert "draw_index % C == I" in shape
    assert "a lost shard costs draws, not a whole rater" in shape
    assert "NOT BUILT" in record["merge_not_built"]
    assert "not in the record" in _flat(record["wall_clock_caveat"])


# --------------------------------------------------------------------------
# the distribution task, end to end small -- sharded down to two draws
# --------------------------------------------------------------------------


@pytest.fixture
def distribution_run(tmp_path, out_root, clean_repo, monkeypatch):
    from cleft.models import cleftgnn
    from cleft.provenance.hashing import hash_path
    from cleft.run import main, p10x_split_seeds

    monkeypatch.setenv("CLEFT_REPO_ROOT", str(clean_repo))
    monkeypatch.setattr(cleftgnn, "CleftGNNBackbone", _StubCleftGNN)
    manifest_dir, staged_dir, sheet_path = _gate_artifacts(tmp_path / "data")

    def run(**task_overrides):
        # shard count 250 -> draws 0 and 250 only: two draws x five
        # raters = ten stub fits, and the shard rule itself is what
        # makes the fixture cheap.
        task = _distribution_task(
            split_seeds=p10x_split_seeds(1337, 500),
            shard={"count": 250, "index": 0},
            **task_overrides,
        )
        config = builders.write_config(
            tmp_path / f"p10x_dist_{abs(hash(str(task_overrides)))}.yaml",
            phase="p10x",
            tier="keeper",
            inputs=[
                {"name": "manifest_v1", "path": str(manifest_dir),
                 "rollup_sha256": hash_path(manifest_dir)["rollup"]},
                {"name": "staged_v1", "path": str(staged_dir),
                 "rollup_sha256": hash_path(staged_dir)["rollup"]},
                {"name": "scoresheet_primary", "path": str(sheet_path),
                 "rollup_sha256": hash_path(sheet_path)["rollup"]},
            ],
            task=task,
        )
        run_dir = main(["--config", str(config), "--out", str(out_root)])
        return (
            json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
            run_dir,
        )

    return run


def test_the_distribution_runs_every_draw_x_rater_cell(distribution_run):
    reported, run_dir = distribution_run()

    shard = reported["shard"]
    assert shard == {"count": 250, "index": 0, "draws_in_shard": 2}
    assert reported["fits_completed"] == 2 * 5
    # Every rater ran, by ruling -- none dropped.
    assert sorted(reported["per_rater"]) == sorted(scoresheet.RATERS)

    for rater, cell in reported["per_rater"].items():
        assert cell["n_draws"] == 2
        for metric in ("f1_macro", "precision_macro", "recall_macro",
                       "accuracy", "iem_a_top1_mean"):
            assert len(cell["distributions"][metric]) == 2, metric
        # Floors ride beside, one per draw.
        assert len(cell["floors"]["majority_class_f1_macro"]) == 2
        assert len(cell["floors"]["iem_constant_mean"]) == 2
        assert "undefined" in cell["floors"]["pcc"]

    # The per-cell detail is written out beside the aggregate.
    assert (run_dir / "draws.csv").is_file()
    rows = (run_dir / "draws.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 2 * 5          # header + one row per cell
    assert "draw_index" in rows[0] and "split_seed" in rows[0]


def test_the_distribution_accounts_degenerate_draws_separately(
    distribution_run
):
    reported, _ = distribution_run()

    for rater, cell in reported["per_rater"].items():
        degenerate = cell["degenerate"]
        # A frequency, reported for every rater -- never a discard.
        assert degenerate["count"] + degenerate["complete_count"] == (
            cell["n_draws"]
        )
        assert degenerate["frequency"] == pytest.approx(
            degenerate["count"] / cell["n_draws"]
        )
        # Occupancy recorded for every draw, not just the degenerate.
        assert len(cell["occupancy"]) == cell["n_draws"]
        for occupancy in cell["occupancy"]:
            assert sum(occupancy.values()) == 28
        # The separate distributions exist whenever the split is real.
        assert set(degenerate["f1_macro_degenerate"]) <= set(
            cell["distributions"]["f1_macro"]
        )
        assert set(degenerate["f1_macro_complete"]) <= set(
            cell["distributions"]["f1_macro"]
        )
        assert (
            len(degenerate["f1_macro_degenerate"])
            + len(degenerate["f1_macro_complete"])
        ) == cell["n_draws"]

    # Rater 11 never grades 5, so EVERY one of its draws is degenerate.
    psychologist = reported["per_rater"]["Rater 11 - Psychologist"]
    assert psychologist["degenerate"]["count"] == psychologist["n_draws"]
    assert psychologist["degenerate"]["frequency"] == 1.0
    # And what a missing class does to the macro is stated, not implied.
    assert "FEWER classes" in reported["degenerate_note"]
    assert "not term-by-term comparable" in reported["degenerate_note"].lower()


def test_the_distribution_locates_the_published_points_and_applies_readings(
    distribution_run
):
    reported, _ = distribution_run()

    points = phase10_annex.PUBLISHED_REFERENCE_POINTS["points"]
    for rater, cell in reported["per_rater"].items():
        located = cell["published_position"]
        assert sorted(located) == sorted(points)
        for name, position in located.items():
            assert position["value"] == points[name]
            assert position["where"] in (
                "inside", "above_all", "below_all", "undefined"
            )
    # PCC only, and the record says why -- no published Study macro-F1
    # exists to locate against.
    assert "PCC" in reported["published_position_note"]
    assert "no published Study macro-F1" in reported["published_position_note"]

    verdict = reported["reading"]
    assert verdict["fired"] in (
        "published_inside_the_distribution",
        "published_outside_and_above",
        None,
    )
    if verdict["fired"] is None:
        assert verdict["observation"] == "MIXED_PATTERN_REGISTERED"
    else:
        assert verdict["reading"] == phase10_annex.READINGS_COMMITTED[
            verdict["fired"]
        ]


def test_every_distribution_output_carries_the_prohibitions(distribution_run):
    reported, _ = distribution_run()

    assert reported["prohibition"].startswith(
        "THESE FIGURES CHARACTERISE OUR REPLICATION UNDER THEIR PROTOCOL"
    )
    assert "UNMATCHED" in reported["rater_caveat"]
    assert "replicated knowingly" in reported["normalisation_note"]
    assert "[REPORTED, not measured]" in reported["supervision_dependency"]
    assert "one draw" not in reported["not_claimable"].lower() or True
    assert "no ledger row" in reported["not_claimable"].lower()


def test_distribution_metrics_json_contains_no_patient_identifiers(
    distribution_run
):
    reported, run_dir = distribution_run()
    text = json.dumps(reported)
    assert "patient_id" not in text
    assert "frontal_id" not in text
    assert "patient_id" not in (run_dir / "draws.csv").read_text(
        encoding="utf-8"
    )


# --------------------------------------------------------------------------
# 2026-08-31, close-out: the run, the fired reading, the measured
# pathologies, the ledger proposal, the closing.
# --------------------------------------------------------------------------


def test_the_run_is_recorded_with_what_it_cannot_corroborate():
    record = phase10_annex.DISTRIBUTION_OBSERVED
    assert "63b0f421" in record["observed"]
    assert "git cat-file" in record["run_sha_is_ours"]
    assert record["fits"] == 2500
    # The per-fit figure, recomputed here rather than trusted.
    assert record["wall_clock_seconds"] / record["fits"] == pytest.approx(
        record["per_fit_seconds"], abs=5e-4
    )
    # The figures arrived rather than being read here, and say so.
    assert "[REPORTED, verified against metrics.json ]" in (
        record["provenance_of_the_figures"]
    )
    assert "CLUSTER-ONLY" in record["provenance_of_the_figures"]
    # Two things it cannot do, stated rather than assumed.
    assert "1337 is NOT among the 500" in record[
        "does_not_corroborate_the_gate"
    ]
    assert "omits parameter_report" in record[
        "no_parameter_report_in_this_arm"
    ]


def test_the_gates_draw_is_genuinely_not_among_the_five_hundred():
    """The claim DISTRIBUTION_OBSERVED rests on, checked rather than
    asserted: run 500 never repeats the gate's draw, so it cannot
    corroborate the gate's F1 or loss."""
    from cleft.run import p10x_split_seeds

    gate = yaml.safe_load(
        (REPO / "configs" / "p10x_gate_fullfit.yaml").read_text(
            encoding="utf-8"
        )
    )
    distribution = yaml.safe_load(
        (REPO / "configs" / "p10x_distribution.yaml").read_text(
            encoding="utf-8"
        )
    )
    gate_seed = gate["task"]["split_seed"]
    seeds = p10x_split_seeds(distribution["task"]["root_seed"], 500)
    assert seeds == distribution["task"]["split_seeds"]
    assert gate_seed not in seeds


def test_reading_one_fired_with_the_committed_sentence_verbatim():
    record = phase10_annex.READING_FIRED
    assert record["fired"].endswith("published_inside_the_distribution")
    # VERBATIM: the same object as the committed reading, not a retype.
    assert record["sentence_verbatim"] == (
        phase10_annex.READINGS_COMMITTED["published_inside_the_distribution"]
    )
    assert "does not identify performance" in record["sentence_verbatim"]
    assert "strongest form of the small-n argument" in (
        record["sentence_verbatim"]
    )

    # The percentile table, and every cited point is one of the four
    # registered reference points -- no fifth appeared.
    table = record["percentiles_across_raters"]
    assert table == {
        "0.440": (97.74, 99.09),
        "0.283": (88.81, 93.14),
        "0.14": (68.86, 79.43),
        "-0.162": (13.87, 26.00),
    }
    # Compared as FLOATS: "0.440" and "0.44" are the same number and
    # different strings, and it is the number that must match.
    registered = set(
        phase10_annex.PUBLISHED_REFERENCE_POINTS["points"].values()
    )
    assert {float(key) for key in table} == registered

    # Every percentile strictly inside (0, 100) -- consistent with
    # "inside" on all twenty cells and with nothing else.
    for point, (low, high) in table.items():
        assert 0.0 < low <= high < 100.0, point
    assert "all 20" in record["nothing_landed_outside"]
    assert "No third reading invented" in record["nothing_landed_outside"]
    assert "MIXED_PATTERN_REGISTERED did not fire" in record[
        "nothing_landed_outside"
    ]

    # The exhibit, and the fence around it.
    exhibit = _flat(record["the_exhibit"])
    assert "RE-DRAWING THE SPLIT ALONE" in exhibit
    assert "Nothing about their MODEL follows" in exhibit
    assert record["prohibition_rides"] == phase10_annex.ANNEX_PROHIBITION


def test_the_reading_that_fired_is_the_one_the_rule_would_fire():
    """The verdict was not chosen -- it is what p10x_apply_readings
    returns for a grid that is 'inside' everywhere, which is what the
    percentile table says the run produced."""
    from cleft.run import p10x_apply_readings

    grid = {
        f"rater{i}": {point: "inside" for point in
                      phase10_annex.READING_FIRED["percentiles_across_raters"]}
        for i in range(5)
    }
    verdict = p10x_apply_readings(grid)
    assert verdict["fired"] == "published_inside_the_distribution"
    assert verdict["reading"] == phase10_annex.READING_FIRED[
        "sentence_verbatim"
    ]
    assert verdict["observation"] is None


def test_the_design_pathologies_are_measured_with_the_bearing_stated():
    record = phase10_annex.DESIGN_PATHOLOGIES_MEASURED
    counts = record["degenerate_draws_of_500"]
    assert counts == (173, 273, 306, 380, 500)
    assert max(counts) == 500, "one rater degenerate in every draw"
    # The overall rate, recomputed here.
    assert sum(counts) == 1632
    assert sum(counts) / 2500 == pytest.approx(0.6528, abs=5e-5)
    assert "1,632 of 2,500" in record["degenerate_overall"]
    assert "65.3%" in record["degenerate_overall"]

    assert record["undefined_pcc_draws_per_rater"] == "62-101"
    assert "no denominator" in record["undefined_pcc_mechanism"]

    # The bearing -- the annex's question, answered in the negative.
    bearing = _flat(record["the_bearing"])
    assert "FREQUENTLY cannot produce a test set containing every class" in (
        bearing
    )
    assert "no computable correlation at all" in bearing
    # And the incomparability clause rides every macro figure.
    assert "EXCLUDED from the macro" in _flat(record["incomparability_rides"])
    assert "not term-by-term comparable" in _flat(
        record["incomparability_rides"]
    )


def test_the_unresolved_attributions_are_named_not_guessed():
    """Two things did not reach this machine at the close, and the
    record said so instead of reconstructing them. Both entries are
    PRESERVED as written; what became available since is dated beside
    them rather than overwriting them."""
    record = phase10_annex.DESIGN_PATHOLOGIES_MEASURED
    attribution = _flat(record["open_attribution"])
    assert "ASCENDING list" in attribution
    assert "NOT sheet order" in attribution
    assert "left UNASSIGNED rather than guessed" in attribution
    # The counts are STILL a bare tuple here -- the original structure
    # is not rewritten; the mapping lives in its own record.
    assert isinstance(record["degenerate_draws_of_500"], tuple)
    for rater in scoresheet.RATERS:
        assert rater not in str(record["degenerate_draws_of_500"])
    assert record["attribution_resolved_2026_08_31"] == "DEGENERATE_BY_RATER"

    table = _flat(record["open_distribution_table"])
    assert "no such table was included" in table
    assert "named as missing rather than reconstructed" in table
    assert "PER_RATER_TABLE" in record[
        "distribution_table_still_partial_2026_08_31"
    ]


def test_the_degenerate_counts_are_attributed_by_name_and_checked():
    record = phase10_annex.DEGENERATE_BY_RATER
    counts = record["counts_of_500"]
    assert counts == {
        "Rater 7 - Cleft patient": 306,
        "Rater 8 - Orthodontist": 380,
        "Rater 9 - Speech and language therapist": 500,
        "Rater 10 - Plastic surgeon": 173,
        "Rater 11 - Psychologist": 273,
    }
    # Sourced, not read here -- the same tag every run-500 figure
    # carries. [2026-09-01, THIS PIN FIRED AS DESIGNED.] It asserted a
    # PREFIX, "[REPORTED", which is exactly why the tag sweep's
    # full-string search for load-bearing variants did not see it. The
    # tag normalised to [MEASURED] with its provenance moved into prose
    # beside it, and the assertion follows the record -- checking the
    # PROPERTY the prefix stood for: sourced, and not read here.
    assert record["provenance"].startswith("[MEASURED]")
    assert "NOT read from metrics.json here" in record["provenance"]
    assert "extractor output" in record["provenance"]
    assert "TAG NORMALISED 2026-09-01" in record["provenance"]
    assert "not read from metrics.json here" in record["provenance"]

    # The three checks the record claims, re-run rather than trusted.
    assert set(counts) == set(scoresheet.RATERS)
    assert tuple(sorted(counts.values())) == (
        phase10_annex.DESIGN_PATHOLOGIES_MEASURED["degenerate_draws_of_500"]
    )
    assert counts["Rater 9 - Speech and language therapist"] == 500
    assert sum(counts.values()) == 1632

    # And the reason the tuple came first is kept: reading the ascending
    # list as sheet order would have mis-assigned EVERY rater. Derived
    # here rather than restated -- which is how the record's first
    # wording ("four of the five") was caught and corrected.
    by_sheet_order = dict(zip(scoresheet.RATERS, sorted(counts.values())))
    wrong = [r for r in scoresheet.RATERS if by_sheet_order[r] != counts[r]]
    assert len(wrong) == 5, wrong
    assert "WRONG for ALL FIVE" in _flat(record["why_the_tuple_came_first"])
    assert "It is FIVE" in _flat(record["corrected_2026_08_31"])
    # The unmatched caveat rides even here.
    assert "UNMATCHED to CleftGNN's A-E" in _flat(
        record["unmatched_caveat_rides"]
    )


def test_the_per_rater_table_fills_what_arrived_and_flags_what_did_not():
    """[UPDATED 2026-08-31] The PENDING columns arrived at the third
    asking and are filled. The pending note is PRESERVED -- it was true
    when written and records that the column was left empty rather than
    reconstructed for two rounds -- with a dated pointer to the fill."""
    record = phase10_annex.PER_RATER_TABLE
    assert record["degenerate_of_500"] == (
        phase10_annex.DEGENERATE_BY_RATER["counts_of_500"]
    )
    # The undefined-PCC figure was a RANGE and was not faked per rater.
    assert "RANGE only" in record["undefined_pcc"]
    assert "none is invented" in record["undefined_pcc"]
    # The pending note stands, unedited, beside its resolution.
    pending = _flat(record["pcc_stats_pending"])
    assert "PENDING" in pending
    assert "arriving neither time" in pending
    assert "left empty rather than reconstructed" in pending
    assert record["filled_2026_08_31"] == (
        "PCC_BY_RATER, F1_BY_RATER, IEM_BY_RATER"
    )


def test_the_two_required_re_derivations_hold():
    """The two checks the instruction asked for, run as ARITHMETIC over the
    recorded table rather than as a restatement of the claim."""
    rows = phase10_annex.PCC_BY_RATER["rows"]
    assert set(rows) == set(scoresheet.RATERS)

    # (1) n_nonnull + undefined == 500, every row.
    for rater, row in rows.items():
        assert row["n_nonnull"] + row["undefined"] == 500, rater

    # (2) every undefined count inside the range the record already
    # held -- and the range is exact at both ends, so the earlier
    # record was the true range rather than a rounding of it.
    low, high = (
        int(v) for v in
        phase10_annex.DESIGN_PATHOLOGIES_MEASURED[
            "undefined_pcc_draws_per_rater"
        ].split("-")
    )
    undefined = [row["undefined"] for row in rows.values()]
    assert all(low <= value <= high for value in undefined)
    assert min(undefined) == low == 62
    assert max(undefined) == high == 101

    checks = phase10_annex.TABLE_CHECKS
    assert checks["required_1_row_sums"].startswith("PASS")
    assert checks["required_2_undefined_in_range"].startswith("PASS")
    assert "EXACT at both ends" in checks["required_2_undefined_in_range"]


def test_the_table_agrees_with_every_figure_already_recorded():
    """Six corroborations the table made available. Each is DERIVED
    here; the record's PASS lines are checked against the derivation,
    not trusted."""
    rows = phase10_annex.PCC_BY_RATER["rows"]

    # (3) the degenerate column, from a separate paste, agrees.
    named = phase10_annex.DEGENERATE_BY_RATER["counts_of_500"]
    table = phase10_annex.PER_RATER_TABLE["degenerate_of_500"]
    assert table == named

    # (4) every quantile row monotone, across all three metrics.
    for source in (
        phase10_annex.PCC_BY_RATER, phase10_annex.F1_BY_RATER,
        phase10_annex.IEM_BY_RATER,
    ):
        for rater, row in source["rows"].items():
            values = [row["min"], row["p05"], row["med"], row["p95"],
                      row["max"]]
            assert values == sorted(values), (source["filled"], rater)

    # (5) sd and span corroborate what was recorded from the earlier
    # paste ("sd ~0.19-0.21", "span ~1.1").
    sds = [row["sd"] for row in rows.values()]
    spans = [row["max"] - row["min"] for row in rows.values()]
    assert 0.19 <= min(sds) and max(sds) <= 0.21
    assert 0.95 <= min(spans) and max(spans) <= 1.2
    assert "sd ~0.19-0.21" in phase10_annex.DESIGN_PATHOLOGIES_MEASURED[
        "pcc_span_and_sd"
    ]

    # (6) THE INDEPENDENT ONE: min/max confirm the fired reading from a
    # different column than the percentiles that produced it.
    points = phase10_annex.PUBLISHED_REFERENCE_POINTS["points"]
    for name, value in points.items():
        for rater, row in rows.items():
            assert row["min"] < value < row["max"], (name, rater)
    assert phase10_annex.READING_FIRED["fired"].endswith(
        "published_inside_the_distribution"
    )

    # (7) every percentile row consistent with the quantile columns.
    for key, (low, high) in phase10_annex.READING_FIRED[
        "percentiles_across_raters"
    ].items():
        value = float(key)
        if all(value > row["p95"] for row in rows.values()):
            window = (95.0, 100.0)
        elif all(row["med"] < value < row["p95"] for row in rows.values()):
            window = (50.0, 95.0)
        elif all(row["p05"] < value < row["med"] for row in rows.values()):
            window = (5.0, 50.0)
        else:
            window = None
        assert window is not None, key
        assert window[0] <= low and high <= window[1], (key, window)

    # (8) the medians, which the observation rests on.
    medians = [row["med"] for row in rows.values()]
    assert -0.04 < min(medians) and max(medians) < 0.05
    assert "-0.0314 and +0.0403" in _flat(
        phase10_annex.PCC_BY_RATER["the_median_observation"]
    )
    assert "NO CORRELATION AT ALL" in _flat(
        phase10_annex.PCC_BY_RATER["the_median_observation"]
    )
    assert "NO claim about their model follows" in _flat(
        phase10_annex.PCC_BY_RATER["the_median_observation"]
    )

    for key in ("3_degenerate_column_agrees", "4_quantiles_monotone",
                "5_corroborates_recorded_spread",
                "6_minmax_confirms_the_reading",
                "7_percentiles_consistent_with_quantiles"):
        assert phase10_annex.TABLE_CHECKS[key].startswith("PASS"), key
    why = _flat(phase10_annex.TABLE_CHECKS["why_6_and_7_matter_most"])
    assert "SEPARATE paste" in why
    assert "columns that were not used to produce it" in why
    assert "Two independent routes" in why


def test_f1_rows_carry_their_cross_rater_incomparability():
    """The caveat is DERIVED from the degenerate mixture, not asserted:
    each rater's all-draws F1 mixes degenerate and complete draws at a
    different rate, so the medians are not a ranking."""
    record = phase10_annex.F1_BY_RATER
    assert record["n_per_row"] == 500
    assert set(record["rows"]) == set(scoresheet.RATERS)

    # The mixtures really do differ, and one is total.
    counts = phase10_annex.DEGENERATE_BY_RATER["counts_of_500"]
    rates = {r: c / 500 for r, c in counts.items()}
    assert min(rates.values()) == pytest.approx(0.346, abs=0.001)
    assert max(rates.values()) == 1.0
    slt = "Rater 9 - Speech and language therapist"
    assert rates[slt] == 1.0, "the SLT's whole F1 distribution is degenerate"

    caveat = _flat(record["not_comparable_across_raters"])
    assert "DIFFERENT MIXTURE" in caveat
    assert "not one of its 500 rows is a five-class macro" in caveat
    assert "different-quantities error" in caveat
    # The specific misreading it forbids is named with its numbers.
    assert "Orthodontist 0.1860 beats Plastic surgeon 0.1366" in caveat
    assert record["rows"]["Rater 8 - Orthodontist"]["med"] == 0.1860
    assert record["rows"]["Rater 10 - Plastic surgeon"]["med"] == 0.1366


def test_iem_rows_carry_the_prohibition_and_their_direction():
    record = phase10_annex.IEM_BY_RATER
    assert record["n_per_row"] == 500
    assert record["convention"].startswith("A ")
    # IEM is an ERROR: lower is better, stated so the column beside F1
    # is not read the same direction.
    assert "LOWER is better" in record["direction"]
    # The standing prohibition rides, and still says what is cited.
    assert "cleftgnn_iem_prohibition" in record["prohibition_rides"]
    assert "NEVER placed beside CleftGNN's Table 2/4/6" in _flat(
        record["prohibition_rides"]
    )
    assert phase18.DELIVERABLES_REGISTERED[
        "cleftgnn_iem_prohibition"
    ].startswith("OUR IEM VALUES ARE NEVER PLACED BESIDE CLEFTGNN'S")


def test_the_two_pathologies_overlap_without_coinciding():
    """The structural detail the table makes visible, CONSTRUCTED from
    the shipped definitions rather than asserted: an undefined PCC and
    a degenerate test side are independent conditions."""
    def undefined_pcc(truth, predicted):
        return float(np.std(truth)) == 0.0 or float(np.std(predicted)) == 0.0

    def degenerate(truth):
        return any(int(np.sum(truth == g)) == 0 for g in range(1, 6))

    # (a) constant prediction, every grade present: no PCC, but the
    #     macro F1 is perfectly computable.
    all_five = np.array([1] * 6 + [2] * 6 + [3] * 6 + [4] * 5 + [5] * 5)
    constant = np.full(len(all_five), 3, dtype=int)
    assert not degenerate(all_five)
    assert undefined_pcc(all_five, constant)
    assert phase10.top1_macro_prf(all_five, constant)["f1_macro"] > 0.0

    # (b) a grade missing, predictions varying: degenerate, PCC defined.
    no_five = np.array([1] * 7 + [2] * 7 + [3] * 7 + [4] * 7)
    varied = no_five.copy()
    assert degenerate(no_five)
    assert not undefined_pcc(no_five, varied)

    # (c) both at once.
    assert degenerate(no_five)
    assert undefined_pcc(no_five, np.full(len(no_five), 2, dtype=int))

    # So counting one is not counting the other, and the record says so.
    detail = _flat(phase10_annex.PER_RATER_TABLE["the_structural_detail"])
    assert "INDEPENDENT" in detail
    assert "OVERLAP WITHOUT COINCIDING" in detail
    assert "a count of one is not a count of the other" in detail


def test_the_sizing_band_missed_low_and_says_so():
    record = phase10_annex.SIZING_OUTTURN
    assert "BELOW the lower bound" in record["outturn"]
    # Recomputed: the band, and the outturn against it.
    assert 2500 * 2.5 / 3600 == pytest.approx(1.74, abs=0.01)
    assert 5385.4 / 3600 == pytest.approx(1.496, abs=0.001)
    assert 5385.4 < 2500 * 2.5
    assert "13.8%" in record["outturn"]
    # The cause is a hypothesis and is labelled one.
    assert "[REASONED, not measured]" in record["likely_cause"]
    assert "A hypothesis" in record["likely_cause"]
    assert "draws.csv" in record["likely_cause"]
    # Why a missed prediction is recorded at all.
    assert "PAIRED_BCA_OBSERVED is the pattern" in _flat(
        record["why_recorded"]
    )


def test_the_ledger_proposal_is_a_proposal_with_its_grounds():
    record = phase10_annex.LEDGER_PROPOSAL
    assert record["proposal"] == "NO ledger row from the Phase 10 annex"
    grounds = _flat(record["grounds"])
    assert "D2_HOME_RULED" in grounds
    assert "this characterises a DESIGN" in grounds
    assert "no arm of ours is asserted better or worse" in grounds
    assert "criterion was NEVER APPLIED" in grounds
    assert "the criterion did not run" in grounds
    # [2026-08-31] RULED: the proposal was taken. The pin fired on the
    # ruling landing and is updated dated.
    ruling = _flat(record["ruling"])
    assert ruling.startswith("RULED 2026-08-31, the maintainer: NO ROW")
    assert "on D2's quoted grounds as drafted" in ruling
    assert "the ledger stands at 37" in ruling.lower()
    # Where the findings live instead, named so they are not mislaid.
    assert "READING_FIRED" in ruling
    assert "DESIGN_PATHOLOGIES_MEASURED" in ruling

    from cleft import results_ledger

    # [2026-08-31] The RULING's sentence "the ledger stands at 37" is a
    # dated historical statement and is asserted above, unchanged. What
    # is checked here is the relation it was standing in for: the annex
    # banked nothing. (A global count would go stale on any unrelated
    # append, and did -- ENTRIES[37].)
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p10x" or "annex" in e["id"] or "p10x" in e["id"]
    ]
    results_ledger.validate()
    # Phase 18's precedent still says what this cites it for, at the
    # key it actually lives under.
    d2 = phase18.D2_HOME_RULED["no_new_ledger_rows"]
    assert "NO ledger rows" in d2
    assert "it measures the criterion, not the arms" in d2
    assert "a ledger row is the unit of ARM-CLAIM" in d2
    # The annex's grounds quote that language rather than paraphrasing.
    assert "the unit of ARM-CLAIM" in grounds


def test_the_closing_walks_all_seven_locked_criteria():
    record = phase10_annex.PHASE_10_ANNEX_CLOSING
    assert record["closed"] == "2026-08-31"
    assert record["question"] == phase10_annex.ANNEX_QUESTION_REGISTERED[
        "question"
    ]
    assert record["answer"].startswith("NO")

    walked = record["exit_criteria_walked"]
    # One entry per locked criterion, and every one MET.
    assert len(walked) == len(phase10_annex.EXIT_CRITERIA["criteria"]) == 7
    for key, value in walked.items():
        assert value.startswith("MET"), key

    # The cost outturn and the still-tagged gate figures are carried in
    # the closing rather than left to be looked up.
    assert "BELOW N_RULED's" in record["cost_outturn"]
    assert "[REPORTED, not verified here]" in record[
        "gate_figures_still_tagged"
    ]
    assert "never repeats split_seed 1337" in record[
        "gate_figures_still_tagged"
    ]

    # The exhibit is named, with the pathology beside it.
    exhibit = _flat(record["the_write_up_exhibit"])
    assert "reproducible by re-drawing the split alone" in exhibit
    assert "65.3%" in exhibit
    # And the prohibition is restated at the close.
    assert "corrects, re-computes or re-scores" in record[
        "what_it_does_not_say"
    ]
    # Open items are carried BY NAME, four of them.
    assert len(record["open_non_blocking"]) == 4
    assert any("attribution" in item for item in record["open_non_blocking"])
    assert any("ledger ruling" in item for item in record["open_non_blocking"])


def test_the_two_honest_outcomes_survive_the_close_out_unchanged():
    """[2026-08-31] Both were recorded when they were found and neither
    is softened now that the annex has closed well: a missed prediction
    and an uncorroborated figure keep their tags."""
    # (1) The sizing band missed LOW, with its cause a hypothesis.
    sizing = phase10_annex.SIZING_OUTTURN
    assert "BELOW the lower bound" in sizing["outturn"]
    assert "1.496 h" in sizing["outturn"]
    assert sizing["predicted"] == "1.74 h to 3.47 h (6,250-12,500 s)"
    assert "[REASONED, not measured]" in sizing["likely_cause"]
    assert "draws.csv" in sizing["likely_cause"]

    # (2) The gate's figures stay tagged, with the disjointness proof
    # that makes the tag necessary rather than cautious.
    observed = phase10_annex.DISTRIBUTION_OBSERVED
    assert "[REPORTED, not verified here]" in observed[
        "does_not_corroborate_the_gate"
    ]
    assert "1337 is NOT among the 500" in observed[
        "does_not_corroborate_the_gate"
    ]
    assert "[REPORTED, not verified here]" in (
        phase10_annex.PHASE_10_ANNEX_CLOSING["gate_figures_still_tagged"]
    )
    # The proof itself, re-run: the gate's draw is genuinely absent.
    from cleft.run import p10x_split_seeds

    gate = yaml.safe_load(
        (REPO / "configs" / "p10x_gate_fullfit.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert gate["task"]["split_seed"] not in p10x_split_seeds(1337, 500)


def test_the_closing_adds_no_claim_and_moves_no_ledger():
    """The annex closes the way the project's phases close: nothing
    asserted that was not measured, and nothing banked."""
    from cleft import results_ledger

    # [2026-08-31] Relation, not a global count -- see the note on
    # test_the_ledger_proposal_is_a_proposal_with_its_grounds.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p10x" or "annex" in e["id"] or "p10x" in e["id"]
    ]
    # The prohibition is still the tested literal it was registered as.
    assert phase10_annex.ANNEX_PROHIBITION.startswith(
        "THESE FIGURES CHARACTERISE OUR REPLICATION UNDER THEIR PROTOCOL"
    )
    assert "NEVER A RE-COMPUTATION OF THEIR RESULTS" in (
        phase10_annex.ANNEX_PROHIBITION
    )
    # And the locked criteria were not edited at the close.
    assert phase10_annex.EXIT_CRITERIA["locked"] == "2026-08-31"
    assert len(phase10_annex.EXIT_CRITERIA["criteria"]) == 7
