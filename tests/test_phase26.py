"""Phase 26: the calibration ablation, restated 2026-09-05.

Nothing is locked and nothing is built. These pin the restate: the
reckoning stated before the case, the four rulings, the grid as a design
rather than a sized thing, and exit criteria that are a DRAFT.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cleft import phase26

REPO = Path(__file__).resolve().parents[1]


def test_the_reckoning_states_the_counter_evidence_first():
    """The gap is real, and the record's own figures say why."""
    record = phase26.THE_RECKONING
    assert record["reckoned"].startswith("2026-09-05")

    gap = " ".join(record["the_gap_is_real_and_it_is_the_union"].split())
    assert "no record in this repository reports all six" in gap
    assert "maximum co-occurrence of THREE" in gap
    assert "GATE1_REFERENCE" in gap

    # ...and that one row is a different quantity from the probe.
    r2 = " ".join(record["and_that_one_row_is_a_different_quantity"].split())
    assert "ONE SEED" in r2
    assert "0.2719366" in r2 and "0.2520" in r2
    assert "R2 error" in r2

    banked = " ".join(record["what_IS_banked_for_the_probe"].split())
    assert "three of the six" in banked
    assert "0.4947" in banked and "0.5181" in banked
    assert "IEM does not exist for the probe at all" in banked

    never = " ".join(record["no_training_setting_has_ever_been_swept"].split())
    assert "this would be the first" in never
    assert "24 trials" in never


def test_the_reckonings_figures_are_live_at_their_homes():
    """Each figure the reckoning quotes, re-read at source."""
    from cleft import ladder, phase21, phase22
    from cleft.train import phase3

    assert ladder.BEST_ARM["pcc"] == 0.2520
    assert ladder.BEST_ARM["sd"] == 0.0148
    assert phase21.CROSS_ARM_SHRINKAGE_OBSERVED["probe"] == 0.4947
    assert phase22.FIVE_ARMS_DO_NOT_SHRINK["the_probe"]["shrinkage"] == 0.4947

    # The one-seed row, and the PCC that makes it a different quantity.
    reference = phase3.GATE1_REFERENCE
    flat = str(reference)
    assert "0.2719" in flat
    assert "0.2520" not in flat


def test_the_prohibition_is_carried_by_reference_not_copied():
    """A second copy is a second thing to drift."""
    from cleft import phase21

    assert phase26.THE_PROHIBITION is phase21.SCALE_INVARIANCE_PROHIBITION

    bound = phase26.WHAT_THIS_PHASE_MAY_NOT_CLAIM
    assert bound["carries"].startswith("phase21.SCALE_INVARIANCE_PROHIBITION")

    first = " ".join(bound["1_no_reading_toward_the_ceiling"].split())
    assert "may not be read as bearing on the PCC ceiling" in first
    assert "0.00e+00" in first

    second = " ".join(bound["2_no_attribution_of_PCC_movement"].split())
    assert "may not be attributed to the calibration change" in second

    defects = " ".join(bound["4_the_IEM_defects_travel"].split())
    assert "0.19753" in defects and "0.0719" in defects and "1.685x" in defects
    assert "moves arms toward the region where the metric misorders" in defects


def test_budget_was_rejected_on_a_measurement_not_dropped():
    """Ruling one. The ground is the record's, not the message's."""
    from cleft import phase7c

    record = phase26.THE_SECOND_FACTOR_RULED
    assert record["ruled"].startswith("2026-09-05")

    why = " ".join(record["why_budget_is_inert_HERE"].split())
    assert "ROUND_2_IS_ROUND_1_TRUNCATED" in why
    assert "epoch n's fit is independent of the budget" in why
    assert "identical_fits`` 23" in why and "total_fits`` 35" in why
    assert "CHECKPOINT SELECTION WINDOW" in why

    # ...and those two counts are live at their home.
    truncated = phase7c.ROUND_2_IS_ROUND_1_TRUNCATED
    assert truncated["identical_fits"] == 23
    assert truncated["total_fits"] == 35
    assert "independent of the budget" in truncated["why"]

    tried = " ".join(record["and_it_was_already_tried_on_this_exact_arm"].split())
    assert "Arm 0 held at 0.2520 sd 0.0148" in tried
    assert "the gate survived the budget change" in tried
    assert "already exists in the record and moved nothing" in tried

    rejected = " ".join(record["REJECTED_NOT_DROPPED"].split())
    assert "recorded so nobody revisits it" in rejected
    assert "has to overturn" in rejected


def test_patience_is_the_factor_and_the_far_end_is_shipped():
    from cleft import phase7c

    record = phase26.THE_SECOND_FACTOR_RULED

    what = " ".join(record["what_patience_IS"].split())
    assert "ONE_EPOCH_IS_NOT_AUGMENTATION" in what
    assert "patience terminates at epoch 1" in what

    far = " ".join(record["the_far_end_is_already_shipped"].split())
    assert "max_epochs 30, patience 30" in far
    assert "needs no new code" in far

    # The shipped policy really is what the restate says it is.
    policy = phase7c.EPOCH_POLICY
    assert policy["max_epochs"] == 30
    assert policy["patience"] == 30
    assert policy["early_stopping_can_fire"] is False

    held = " ".join(record["budget_is_HELD_and_here_is_the_value"].split())
    assert "max_epochs is held at 30" in held
    assert "does not bind" in held
    assert "would confound the axis" in held


def test_the_axis_is_respecified_and_the_alternative_is_named():
    """Ruling two. A dropped factor is a rejection or a concession, and
    this is neither."""
    record = phase26.THE_AXIS_IS_RESPECIFIED
    assert "remains two factor" in " ".join(record["the_ruling"].split())

    alternative = " ".join(record["the_alternative_that_was_NOT_taken"].split())
    assert "single-factor weight-decay sweep" in alternative
    assert "was not taken" in alternative

    concession = " ".join(record["what_a_concession_would_have_looked_like"].split())
    assert "cancelled with a reason and never silently dropped" in concession
    assert "respecification" in concession


def test_the_floor_is_recomputed_and_recovered_and_both_are_required():
    """Ruling three. Two computations, and disagreement is a finding."""
    from cleft import phase18

    record = phase26.THE_FLOOR_RULED
    problem = " ".join(record["the_problem"].split())
    assert "exists as a FUNCTION and not as a figure" in problem
    assert "No numeric value for it exists anywhere in this repository" in problem
    assert "floor beside every figure" in problem

    # The function exists and the requirement is live.
    assert callable(phase18.constant_predictor_iem)
    assert "floor beside every figure" in (
        phase18.DELIVERABLES_REGISTERED["d5_iem_third_family"]
    )

    both = " ".join(record["why_BOTH"].split())
    assert "two independent computations agreeing is stronger" in both
    assert "a disagreement is itself a finding" in both
    assert "is not a failure of this phase" in both

    limit = " ".join(record["what_it_does_not_do"].split())
    assert "CASE_IEM_IDENTITY_OPEN" in limit


def test_timing_is_reported_because_the_cost_claim_is_unmeasured():
    """Ruling four. The instrumentation exists and has never been banked."""
    from cleft import phase20

    record = phase26.TIMING_IS_REPORTED
    why = " ".join(record["why"].split())
    assert "no banked runtime exists" in why
    assert "Expectation: seconds" in why
    assert "the expectation is not the measurement" in why

    # Both halves of that quotation are live at their home.
    gate = phase20.COMPUTE_GATE_DESIGNED
    assert "Expectation: seconds" in gate["why_it_is_probably_trivial"]
    assert "measures rather than assumes" in (
        gate["the_expectation_is_not_the_measurement"]
    )

    exists = " ".join(record["the_instrumentation_already_exists_and_already_runs"].split())
    assert "wall_clock_seconds_by_seed" in exists
    assert "never banked" in exists

    banked = " ".join(record["what_gets_banked"].split())
    assert "The unit is the fold-fit" in banked


def test_the_grid_is_a_design_and_is_not_sized():
    record = phase26.THE_GRID_AS_A_DESIGN

    shape = " ".join(record["shape"].split())
    assert "two factors, fully crossed" in shape
    assert "weight decay" in shape and "patience" in shape

    unsized = " ".join(record["the_count_is_NOT_chosen_here"].split())
    assert "levels are chosen when the phase locks" in unsized
    assert "25 times W times P fold-fits" in unsized
    # No cell count is asserted anywhere in the design.
    assert "cells" not in shape or "W times P" in unsized

    forced = " ".join(record["one_config_per_cell_is_forced"].split())
    assert "no list-valued fit knob" in forced
    assert "give a list to sweep in one job" in forced

    lattice = " ".join(record["the_lattice_discipline_still_applies"].split())
    assert "CONTRASTS must each stay one-factor" in lattice

    trap = " ".join(record["the_trap_that_would_make_it_vacuous"].split())
    assert "measures nothing on that axis" in trap
    assert "selected_epoch to be reported per cell" in trap


def test_the_settings_that_must_be_declared_are_named_with_reasons():
    record = phase26.SETTINGS_NEEDING_DECLARATION
    assert "30, held" in record["max_epochs"]

    monitor = " ".join(record["monitor"].split())
    assert "inner_val_mse" in monitor
    assert "patience CAN fire and is the factor" in monitor

    # [UPDATED 2026-09-06] The reason was corrected. It used to cite the
    # preceding phase as a live instance of the hazard, which overstated
    # it: the divergence there is real and inert. The reason is now this
    # phase, where patience does change the fit.
    batch = " ".join(record["batch_size"].split())
    assert "Phase 26 varies patience, which does change the fit" in batch
    assert "declaring one costs nothing" in batch
    assert "CORRECTED 2026-09-06" in batch
    assert "it is INERT" in batch

    # The default really is not the probe's, which is why declaring it
    # costs nothing and omitting it would not be free.
    from cleft.config.schema import TASK_SPECS

    assert TASK_SPECS["train_cv"]["batch_size"].default == 16


def test_the_union_names_its_one_gap_and_touches_no_frozen_module():
    record = phase26.WHAT_THE_UNION_REQUIRES

    free = " ".join(record["four_come_free"].split())
    assert "PCC, RMSE, MAE and shrinkage" in free
    assert "CVResult.metrics" in free

    gap = " ".join(record["IEM_IS_THE_ONE_GAP"].split())
    assert "does not emit IEM at all" in gap
    assert "phase18.iem_score" in gap

    frozen = " ".join(record["and_no_frozen_module_needs_touching"].split())
    assert "eval/metrics.py" in frozen
    assert "which is not frozen" in frozen

    # The freeze claim is checked against the freeze guard itself.
    guard = (REPO / "tests" / "test_frozen_apparatus.py").read_text(
        encoding="utf-8"
    )
    for name in ("eval/metrics.py", "train/harness.py", "data/folds.py"):
        assert name in guard, name


def test_the_exit_criteria_are_a_draft_and_carry_no_readings():
    """The lock is a separate act. Readings are written there, never
    here, so no reading can be written after a number exists."""
    record = phase26.EXIT_CRITERIA_DRAFT
    assert record["status"].startswith("DRAFT")
    assert "NOT LOCKED" in record["drafted"]

    for key in ("1_all_six_for_every_cell", "2_the_floor_beside_every_IEM_figure",
                "3_the_axis_is_shown_to_have_moved", "4_the_prohibition_travels",
                "5_the_IEM_defects_travel", "6_per_fit_wall_clock_banked",
                "7_one_factor_contrasts_only",
                "8_a_ledger_row_only_if_claimable"):
        assert key in record, key

    inert = " ".join(record["3_the_axis_is_shown_to_have_moved"].split())
    assert "that is the phase's result on that axis" in inert

    null = " ".join(record["8_a_ledger_row_only_if_claimable"].split())
    assert "predicts no claimable row" in null

    absent = " ".join(record["what_is_NOT_here"].split())
    assert "readings" in absent
    assert "written at the lock, not at the restate" in absent

    # No key in the draft is a reading.
    assert not any("reading" in key for key in record if key != "what_is_NOT_here")


def test_the_build_ships_configs_and_launches_nothing():
    """[UPDATED 2026-09-06 at the lock] This asserted no config and no
    task existed, which was the state at the restate. The lock builds
    both. What must still be absent is a RESULT: no run directory, no
    ledger row, and every declaration pending."""
    import yaml

    from cleft import results_ledger
    from cleft.run import TASKS

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38` on
    # the line above the relation below. The relation is the claim,
    # the count was an absolute-state instrument standing beside it,
    # and it went stale when Phase 27 appended entry 39. Removed
    # rather than bumped, because bumping it only defers the next
    # unrelated failure.
    assert not any("p26" in entry["id"] for entry in results_ledger.ENTRIES)

    # Built: seventeen configs and one task.
    assert len(list((REPO / "configs").glob("p26_cell_*.yaml"))) == 16
    assert (REPO / "configs" / "p26_calibration_table.yaml").is_file()
    assert "p26_calibration_table" in TASKS

    # Not run: every cell's directory is a placeholder, so the table
    # refuses rather than reading something nobody declared.
    payload = yaml.safe_load(
        (REPO / "configs" / "p26_calibration_table.yaml").read_text(
            encoding="utf-8"
        )
    )
    placeholder = "0" * 64
    cells = [e for e in payload["inputs"] if e["name"].startswith("p26_cell_")]
    assert len(cells) == 16
    # [UPDATED 2026-09-06, at the declare] Paths and hashes are both
    # real. What still makes this a build rather than a result is that
    # the TABLE has not run: no metrics.json, no ledger row, no figure.
    assert all("/runs/keeper/p26/" in e["path"] for e in cells)
    assert not [e for e in cells if e["rollup_sha256"] == placeholder]

    # And no run directory exists locally, because launches are not ours.
    assert not list((REPO / "runs").glob("**/p26_*")) if (
        REPO / "runs"
    ).is_dir() else True

    # The amendment entry is unchanged: the lock is this module's, not
    # the amendment's, and scope was never written there.
    from cleft import phase25

    entry = phase25.PHASE_SEQUENCE_EXTENDED_8["phase_26_the_calibration_ablation"]
    assert entry["status"] == "SCHEDULED, NOT REGISTERED"


def test_the_summary_is_the_whole_restate_the_lock_and_the_close():
    """[UPDATED 2026-09-06, twice] Two keys joined when criterion 3
    closed and the inert distinction was recorded. Three more joined
    when criterion 6 closed and the runtime was banked."""
    keys = sorted(phase26.summary())
    assert keys == [
        "anchor",
        "axis_respecified",
        "cells_observed",
        "closing",
        "cost_of_patience",
        "criteria_not_met",
        "exit_criteria",
        "exit_criteria_draft",
        "floor",
        "grid",
        "grid_locked",
        "iem_result",
        "may_not_claim",
        "mechanism",
        "patience_inert_in_effect",
        "prohibition",
        "readings",
        "reckoning",
        "runtime",
        "scheduling_claim",
        "second_factor",
        "selected_epochs",
        "settings",
        "timing",
        "union",
    ]


# --------------------------------------------------------------------------
# the lock and the build, 2026-09-06
# --------------------------------------------------------------------------


def test_the_grid_is_sixteen_cells_with_a_reason_for_every_level():
    grid = phase26.THE_GRID_LOCKED
    assert grid["locked"] == "2026-09-06"

    decays = grid["weight_decay_levels"]
    patiences = grid["patience_levels"]
    assert len(decays) == 4 and len(patiences) == 4
    assert len(decays) * len(patiences) == 16

    # The ruled span: below the current value, and well above it.
    assert 0.01 in decays
    assert min(decays) < 0.01 and max(decays) > 0.01
    assert 0.0 in decays, "absence of the mechanism is its own level"

    # The ruled span: from the current 5 to the shipped far end of 30.
    assert min(patiences) == 5 and max(patiences) == 30

    # EVERY level carries its reason, and the reasons are not stubs.
    for level in decays:
        reason = grid["why_each_weight_decay_level"][str(level)]
        assert len(reason) > 120, level
    for level in patiences:
        reason = grid["why_each_patience_level"][str(level)]
        assert len(reason) > 120, level

    # The spacing choice is argued on both axes, and differently.
    assert "log" in grid["why_each_weight_decay_level"]["and_why_log_spacing"]
    assert "not a scale" in grid["why_each_patience_level"][
        "and_why_not_log_spacing_here"
    ]


def test_the_far_end_is_the_shipped_policy_and_the_budget_does_not_bind():
    from cleft import phase7c

    grid = phase26.THE_GRID_LOCKED
    assert max(grid["patience_levels"]) == phase7c.EPOCH_POLICY["patience"]
    assert "30 on every cell" in grid["held"]

    # And the anchor's one difference from the probe is stated.
    note = " ".join(
        grid["and_the_anchor_differs_from_the_probe_in_ONE_held_field"].split()
    )
    assert "30 here against the probe's 40" in note
    assert "ROUND_2_IS_ROUND_1_TRUNCATED" in note


def test_the_readings_were_written_at_the_lock_and_cover_both_directions():
    readings = phase26.READINGS
    assert readings["written"].startswith("2026-09-06")
    assert "before any cell has run" in readings["written"]

    # Each factor has a moves reading AND a does-not-move reading.
    assert "weight_decay_moves_calibration_and_not_PCC" in readings
    assert "weight_decay_moves_nothing" in readings
    assert "patience_moves_the_selected_epoch" in readings
    assert "patience_does_NOT_move_the_selected_epoch" in readings

    null = " ".join(readings["weight_decay_moves_nothing"].split())
    assert "the null, and it is a real answer" in null

    inert = " ".join(readings["patience_does_NOT_move_the_selected_epoch"].split())
    assert "the phase's result on that axis rather than a defect" in inert
    assert "ROUND_2_IS_ROUND_1_TRUNCATED" in inert

    # The prohibition governs every reading.
    bound = " ".join(readings["and_what_NO_outcome_licenses"].split())
    assert "says anything about the PCC ceiling" in bound
    assert "WHAT_THIS_PHASE_MAY_NOT_CLAIM" in bound


def test_the_criteria_are_locked_and_the_draft_is_preserved():
    locked = phase26.EXIT_CRITERIA
    assert locked["locked"] == "2026-09-06"
    assert locked["supersedes"].startswith("EXIT_CRITERIA_DRAFT")

    # The draft is still there, unchanged, and still says it is a draft.
    assert phase26.EXIT_CRITERIA_DRAFT["status"].startswith("DRAFT")

    # Two criteria the draft could not carry, because it had no levels.
    anchor = " ".join(locked["9_the_anchor_cell_reproduces_the_probe"].split())
    assert "0.2520" in anchor and "0.0148" in anchor
    assert "the run is wrong and no cell is read" in anchor

    timing = " ".join(locked["10_the_readings_were_written_at_the_lock"].split())
    assert "predates every number" in timing


def test_the_sixteen_configs_are_the_locked_grid(repo_root):
    import yaml

    grid = phase26.THE_GRID_LOCKED
    want = {
        (float(w), int(p))
        for w in grid["weight_decay_levels"] for p in grid["patience_levels"]
    }
    cells = sorted(repo_root.glob("configs/p26_cell_*.yaml"))
    assert len(cells) == 16

    seen = set()
    probe = yaml.safe_load(
        (repo_root / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    for path in cells:
        task = yaml.safe_load(path.read_text(encoding="utf-8"))["task"]
        assert task["kind"] == "train_cv"
        assert task["max_epochs"] == 30, path.name
        seen.add((float(task["weight_decay"]), int(task["patience"])))
        # Every field that is neither varied nor structural is the
        # probe's, checked over the UNION rather than over a list.
        varied = {"weight_decay", "patience", "max_epochs"}
        structural = {
            "kind", "manifest_artifact", "staged_artifact",
            "embeddings_artifact",
        }
        for field in (set(probe) | set(task)) - varied - structural:
            assert task.get(field) == probe.get(field), (path.name, field)
    assert seen == want


def test_the_table_config_declares_every_cell_and_pends_them_all(repo_root):
    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "p26_calibration_table.yaml").read_text(
            encoding="utf-8"
        )
    )
    task = payload["task"]
    assert task["kind"] == "p26_calibration_table"
    assert task["expect_cells"] == 16
    assert len(task["arms"]) == 16

    # [UPDATED 2026-09-06, at the declare] Paths AND hashes are filled
    # now, so guard 3 is live on all sixteen cells. No placeholder
    # remains, and the config says the declaration is complete.
    placeholder = "0" * 64
    cells = [
        e for e in payload["inputs"] if e["name"].startswith("p26_cell_")
    ]
    assert len(cells) == 16
    assert not [e for e in cells if e["rollup_sha256"] == placeholder]
    text = (repo_root / "configs" / "p26_calibration_table.yaml").read_text(
        encoding="utf-8"
    )
    assert "FULLY DECLARED" in text

    # The anchor is declared rather than discovered.
    assert task["anchor_cell"] in {a["name"] for a in task["arms"]}
    assert task["anchor_pcc"] == 0.2520
    assert task["anchor_sd"] == 0.0148


def test_the_table_task_is_registered_and_refuses_a_placeholder():
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS

    assert "p26_calibration_table" in TASKS
    spec = TASK_SPECS["p26_calibration_table"]
    for field in ("arms", "expect_cells", "anchor_cell", "anchor_pcc"):
        assert field in spec, field
    # The Phase 18 floor is OPTIONAL, so its absence is reportable.
    assert spec["phase18_floor_iem"].required is False

    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p26_calibration_table)
    assert "still" in body and "placeholders" in body
    # All six readouts, and the floor beside them.
    for metric in ("pcc", "shrinkage", "rmse", "mae", "acc3", "iem"):
        assert metric in body, metric
    assert "constant_predictor_iem" in body
    assert "iem_floor" in body
    assert "wall_clock_seconds_by_seed" in body
    assert "seconds_per_fold_fit" in body
    # The cross-check reports its own absence rather than skipping.
    assert "UNAVAILABLE rather than passed" in body


def test_the_generator_check_is_clean(repo_root):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/generate_phase26_configs.py", "--check"],
        cwd=repo_root, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "PATH": ""},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 drifted" in result.stdout


def test_the_run_directories_are_the_supplied_names(repo_root):
    """**Supplied, never constructed.** Job ids abbreviate unpredictably,
    so a directory derived from a config stem is a directory nobody
    declared. These sixteen were given, and the generator refuses a cell
    it has no supplied name for."""
    import importlib.util

    import yaml

    spec = importlib.util.spec_from_file_location(
        "gen_p26", repo_root / "scripts" / "generate_phase26_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.RUN_SHA == "1df28d6f"
    assert len(module.RUN_DIRS) == 16

    payload = yaml.safe_load(
        (repo_root / "configs" / "p26_calibration_table.yaml").read_text(
            encoding="utf-8"
        )
    )
    declared = {
        entry["name"]: entry["path"]
        for entry in payload["inputs"] if entry["name"].startswith("p26_cell_")
    }
    assert len(declared) == 16
    for stem, directory in module.RUN_DIRS.items():
        assert declared[stem].endswith("/runs/keeper/p26/" + directory), stem
        # Every one carries the same sha and a job id matching its cell.
        assert module.RUN_SHA in directory
        assert directory.startswith(stem + "__")
        assert directory.endswith("__" + stem.replace("_", "-"))


def test_the_phase18_floor_is_declared_and_sourced_as_a_read(repo_root):
    """Criterion 2's cross-check needs a second computation, not a
    second copy of the first one."""
    import importlib.util

    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "p26_calibration_table.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["task"]["phase18_floor_iem"] == 0.5371587111538849

    text = (repo_root / "configs" / "p26_calibration_table.yaml").read_text(
        encoding="utf-8"
    )
    assert "READ from that run" in text
    assert "never recomputed here" in text

    spec = importlib.util.spec_from_file_location(
        "gen_p26b", repo_root / "scripts" / "generate_phase26_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PHASE18_FLOOR_IEM == 0.5371587111538849

    # The value is a READ, so nothing in this repository may derive it.
    # If it could be computed here the cross-check would be vacuous.
    source = (
        repo_root / "scripts" / "generate_phase26_configs.py"
    ).read_text(encoding="utf-8")
    assert "constant_predictor_iem" not in source


def test_the_table_still_refuses_until_the_hashes_are_declared():
    """Paths filled is not declared. The placeholder is what stops a run
    reading a directory whose contents nobody has verified."""
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p26_calibration_table)
    assert "declared_rollup" in body
    assert "placeholders" in body
    assert "refuses rather than reading a directory nobody declared" in body


def test_the_sixteen_rollups_are_declared_and_all_distinct(repo_root):
    """**A repeated hash is a paste error, not an agreement.**

    Sixteen cells differing only in weight decay and patience produce run
    directories of nearly identical size and shape, so two identical
    rollups would read as two cells that happen to match when what they
    mean is that one was declared twice. The generator refuses on a
    repeat and this asserts the shipped config.
    """
    import importlib.util

    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "p26_calibration_table.yaml").read_text(
            encoding="utf-8"
        )
    )
    cells = {
        entry["name"]: entry["rollup_sha256"]
        for entry in payload["inputs"] if entry["name"].startswith("p26_cell_")
    }
    assert len(cells) == 16

    digests = list(cells.values())
    assert len(set(digests)) == 16, (
        "a rollup appears twice: "
        f"{sorted(d for d in digests if digests.count(d) > 1)}"
    )
    for stem, digest in cells.items():
        assert len(digest) == 64, stem
        assert all(c in "0123456789abcdef" for c in digest), stem
        assert set(digest) != {"0"}, stem

    # The config carries what the generator holds, so a hand edit to
    # either one is drift rather than a quiet divergence.
    spec = importlib.util.spec_from_file_location(
        "gen_p26c", repo_root / "scripts" / "generate_phase26_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RUN_ROLLUPS == cells


def test_the_generator_refuses_a_repeated_rollup(repo_root):
    """The distinctness rule must reject what it was built for. A rule
    that passes everything has not been shown to test anything."""
    import importlib.util

    import pytest

    spec = importlib.util.spec_from_file_location(
        "gen_p26d", repo_root / "scripts" / "generate_phase26_configs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original = dict(module.RUN_ROLLUPS)
    try:
        # Two cells declared with the same digest, which is the paste
        # error the check exists for.
        module.RUN_ROLLUPS["p26_cell_w3_p3"] = original["p26_cell_w0_p0"]
        with pytest.raises(module.DriftError) as raised:
            module._table_config()
        assert "appear more than once" in str(raised.value)
        assert "paste error" in str(raised.value)

        # And a malformed digest is refused too.
        module.RUN_ROLLUPS = dict(original)
        module.RUN_ROLLUPS["p26_cell_w0_p0"] = "not-a-sha256"
        with pytest.raises(module.DriftError) as raised:
            module._table_config()
        assert "is not a sha256 digest" in str(raised.value)
    finally:
        module.RUN_ROLLUPS = original


def test_nothing_but_the_cell_rollups_moved_at_the_declare(repo_root):
    """The declare fills sixteen hashes and touches nothing else."""
    import yaml

    payload = yaml.safe_load(
        (repo_root / "configs" / "p26_calibration_table.yaml").read_text(
            encoding="utf-8"
        )
    )
    others = [
        entry for entry in payload["inputs"]
        if not entry["name"].startswith("p26_cell_")
    ]
    assert len(others) == 1
    manifest = others[0]
    assert manifest["name"] == "manifest_v1"

    # The manifest declaration is the probe's own, unchanged.
    probe = yaml.safe_load(
        (repo_root / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )
    probe_manifest = next(
        e for e in probe["inputs"] if e["name"] == "manifest_v1"
    )
    assert manifest == probe_manifest

    # The declared floor is untouched by the hash fill.
    assert payload["task"]["phase18_floor_iem"] == 0.5371587111538849
    assert payload["task"]["anchor_cell"] == "p26_cell_w2_p0"


# --------------------------------------------------------------------------
# the close, 2026-09-06
# --------------------------------------------------------------------------

METRICS_FILE = Path("X:/p26_table_metrics.json")


def _run_metrics():
    """The run's own file, if it is reachable. It is a cluster artifact
    that arrived by paste, so the tests that need it skip rather than
    fail on a machine that does not have it."""
    import json

    import pytest

    if not METRICS_FILE.is_file():
        pytest.skip("the run's metrics file is not on this machine")
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


def test_the_closing_states_the_finding_and_what_it_answers():
    closing = phase26.PHASE_26_CLOSING
    assert closing["closed"].startswith("2026-09-06")
    assert "58c7e847" in closing["closed"]

    finding = " ".join(closing["the_finding"].split())
    assert "NEITHER SETTING MOVES ANY READOUT" in finding
    assert "0.0008438819" in finding
    assert "one patient in one seed" in finding

    answers = " ".join(closing["what_it_answers"].split())
    assert "IEM cannot be lowered by tuning these two settings" in answers
    assert "change the fit itself rather than its settings" in answers
    assert "measured that rather than supposing it" in answers


def test_both_registered_readings_fired_in_the_words_they_were_written_in():
    fired = phase26.PHASE_26_CLOSING["readings_that_fired"]
    readings = phase26.READINGS

    null = " ".join(fired["weight_decay_moves_nothing"].split())
    assert "FIRED" in null
    assert "the null, and it is a real answer" in null
    assert "the null, and it is a real answer" in " ".join(
        readings["weight_decay_moves_nothing"].split()
    )

    inert = " ".join(fired["patience_does_NOT_move_the_selected_epoch"].split())
    assert "written in advance as the phase's result on that axis" in inert
    assert "rather than as a defect to work around" in inert
    written = " ".join(readings["patience_does_NOT_move_the_selected_epoch"].split())
    assert "that is the phase's result on that axis rather than a defect" in written

    # And the readings predate the run, which criterion 10 is about.
    assert readings["written"].startswith("2026-09-06")
    assert "before any cell has run" in readings["written"]


def test_the_criterion_walk_is_honest_about_the_two_that_failed():
    walk = phase26.PHASE_26_CLOSING["criterion_walk"]
    assert len(walk) == 10
    assert sorted(int(k.split("_")[0]) for k in walk) == list(range(1, 11))

    # [UPDATED 2026-09-06, twice] Criteria 3 and 6 both closed after
    # the close, each by reading a file the runs already held. Both
    # entries still carry the words they read at the close, in
    # brackets, so the phrase NOT MET survives inside entries that are
    # now met. The verdict is the opening word.
    unmet = [
        k for k, v in walk.items() if v.lstrip("*").startswith("NOT MET")
    ]
    assert unmet == []

    six = " ".join(walk["6_per_fit_wall_clock_banked"].split())
    assert six.startswith("**MET, from 2026-09-06.**")
    assert "It read NOT MET at the close" in six
    assert "a file nothing writes" in six

    three = " ".join(walk["3_the_axis_is_shown_to_have_moved"].split())
    assert three.startswith("**MET, from 2026-09-06.**")
    # The state at the close is preserved rather than overwritten, and
    # the reason it was unmet is still legible.
    assert "It read NOT MET at the close" in three
    assert "emitted no selected epoch" in three

    # And the closing does not claim a clean walk anywhere.
    blob = " ".join(str(v) for v in phase26.PHASE_26_CLOSING.values())
    assert "all ten" not in blob.lower()


def test_the_unmet_criteria_name_the_defect_and_own_it():
    record = phase26.WHAT_CRITERION_3_DID_NOT_GET

    six = " ".join(record["criterion_6_wall_clock"].split())
    assert "Nothing in this repository writes a file by that name" in six
    assert "recorded null rather than raising" in six

    # The restate's own claim is corrected rather than left standing.
    corrected = " ".join(
        record["and_the_restate_was_wrong_about_the_instrumentation"].split()
    )
    assert "not merely unbanked, it is not readable from any file" in corrected
    assert "TIMING_IS_REPORTED is corrected by this entry" in corrected

    shape = " ".join(record["the_shape_it_belongs_to"].split())
    assert "Phase 25 defect class" in shape
    assert "THE_GREP_THAT_DID_NOT_RUN" in shape

    cost = " ".join(record["what_it_costs_and_what_it_does_not"].split())
    assert "no finding depends on either" in cost
    assert "remains an expectation" in cost


def test_the_claim_that_summary_json_is_never_written_is_still_true(repo_root):
    """The defect record says nothing writes that file. If that ever
    stops being true the record is stale, so it is checked rather than
    asserted once."""
    import subprocess

    hits = subprocess.run(
        ["git", "grep", "-n", "summary.json"], cwd=repo_root,
        capture_output=True, text=True,
    ).stdout.strip().split("\n")
    hits = [h for h in hits if h]
    # The reader itself, and prose that mentions it. No writer.
    writers = [h for h in hits if "write_text" in h or "open(" in h]
    assert not writers, f"something writes summary.json now: {writers}"


def test_every_figure_in_the_close_is_in_the_runs_own_file():
    """**The point of the close.** Each figure the record states is
    re-derived from the metrics file, so a transcription error fails
    here rather than standing."""
    metrics = _run_metrics()
    cells = metrics["cells"]

    # The anchor, to full precision.
    assert phase26.THE_ANCHOR_HELD["observed"] == metrics["anchor"]["measured"]
    assert metrics["anchor"]["within_two_seed_sd"] is True

    # The floor and its cross-check.
    assert phase26.THE_IEM_RESULT["floor"] == metrics["iem_floor"]
    check = metrics["floor_cross_check"]
    assert check["agree"] is True
    assert check["absolute_difference"] == 0.0
    assert check["recomputed_here"] == check["phase18_recovered"]
    assert phase26.THE_IEM_RESULT["best_cell"] == min(
        c["mean"]["iem"] for c in cells.values()
    )
    assert phase26.THE_IEM_RESULT["worst_cell"] == max(
        c["mean"]["iem"] for c in cells.values()
    )

    # The four rows, each the mean of its whole weight decay group.
    banked = phase26.THE_SIXTEEN_CELLS_OBSERVED["means_by_weight_decay"]
    for decay, row in banked.items():
        group = [
            c for c in cells.values() if c["weight_decay"] == float(decay)
        ]
        assert len(group) == 4, decay
        for metric, value in row.items():
            assert round(group[0]["mean"][metric], 6) == value, (decay, metric)

    # The spreads.
    spreads = phase26.THE_SIXTEEN_CELLS_OBSERVED["spread_across_the_whole_grid"]
    for metric, value in spreads.items():
        seen = [c["mean"][metric] for c in cells.values()]
        assert round(max(seen) - min(seen), 10) == value, metric


def test_the_bit_identity_and_the_one_patient_claim_reproduce():
    """Two claims that carry the whole finding, re-derived."""
    metrics = _run_metrics()
    cells = metrics["cells"]

    # Bit identity within every weight decay group.
    for decay in metrics["grid"]["weight_decay_levels"]:
        group = [c for c in cells.values() if c["weight_decay"] == decay]
        assert len(group) == 4
        for other in group[1:]:
            assert other["per_seed"] == group[0]["per_seed"], decay
            assert other["mean"] == group[0]["mean"], decay

    # The largest movement is three-class accuracy, and it is exactly
    # one patient in one seed.
    spreads = {
        m: max(c["mean"][m] for c in cells.values())
           - min(c["mean"][m] for c in cells.values())
        for m in ("pcc", "shrinkage", "rmse", "mae", "acc3", "iem")
    }
    assert max(spreads, key=spreads.get) == "acc3"
    assert abs(spreads["acc3"] - 1.0 / (237 * 5)) < 1e-12

    # And the record says so, including that it is not shrinkage.
    note = " ".join(
        phase26.THE_SIXTEEN_CELLS_OBSERVED["and_it_is_NOT_shrinkage"].split()
    )
    assert "second largest movement, not the largest" in note


def test_the_ledger_did_not_move_and_the_phase_claims_nothing():
    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38` on
    # the line above the relation below. The relation is the claim,
    # the count was an absolute-state instrument standing beside it,
    # and it went stale when Phase 27 appended entry 39. Removed
    # rather than bumped, because bumping it only defers the next
    # unrelated failure.
    assert not any("p26" in entry["id"] for entry in results_ledger.ENTRIES)

    closing = " ".join(phase26.PHASE_26_CLOSING["no_ledger_row"].split())
    assert "nothing here is claimable and none was expected" in closing

    # The reckoning predicted it, before the grid existed.
    reckoning = " ".join(
        phase26.EXIT_CRITERIA["8_a_ledger_row_only_if_claimable"].split()
    )
    assert "predicts no claimable row" in reckoning


# --------------------------------------------------------------------------
# criterion 3 closed, criterion 6 still open, 2026-09-06
# --------------------------------------------------------------------------

EPOCHS_FILE = Path("X:/p26_selected_epochs.json")


def test_the_selected_epochs_are_read_and_uniform():
    """400 fold observations, every one at epoch 1, re-derived from the
    file rather than trusted from the record."""
    import json

    import pytest

    if not EPOCHS_FILE.is_file():
        pytest.skip("the selected-epoch file is not on this machine")
    payload = json.loads(EPOCHS_FILE.read_text(encoding="utf-8"))

    assert len(payload) == 16
    values, total = set(), 0
    for cell, body in payload.items():
        epochs = body["selected_epochs"]
        assert len(epochs) == 5, cell
        for seed, folds in epochs.items():
            assert len(folds) == 5, (cell, seed)
            values.update(folds)
            total += len(folds)
    assert total == 400
    assert values == {1}, values

    record = phase26.THE_SELECTED_EPOCHS_OBSERVED
    assert "400 fold observations" in " ".join(record["shape"].split())
    assert "every one of the 400 is epoch 1" in " ".join(
        record["the_result"].split()
    )
    assert "Uniform" in record["the_result"]


def test_criterion_three_is_met_and_says_what_it_used_to_read():
    walk = phase26.PHASE_26_CLOSING["criterion_walk"]
    three = " ".join(walk["3_the_axis_is_shown_to_have_moved"].split())
    assert three.startswith("**MET, from 2026-09-06.**")
    # The state at the close is preserved rather than overwritten.
    assert "It read NOT MET at the close" in three
    assert "400 fold observations" in three

    record = phase26.WHAT_CRITERION_3_DID_NOT_GET
    closed = " ".join(record["criterion_3_CLOSED_2026_09_06"].split())
    assert "recovered from each cell run" in closed
    assert "rather than by re-running anything" in closed
    # And what is still owed is not quietly dropped.
    assert "The task still does not emit it" in closed

    # The mechanism's REPORTED tag is superseded, not deleted.
    mechanism = phase26.THE_MECHANISM_MEASURED
    assert "the_selected_epoch_itself_is_REPORTED_not_read" in mechanism
    superseded = " ".join(
        mechanism["SUPERSEDED_2026_09_06_the_epoch_is_now_READ"].split()
    )
    assert "no longer holds" in superseded
    assert "the whole grid from the runs" in superseded


def test_patience_is_recorded_as_inert_in_effect_not_unreached():
    record = phase26.PATIENCE_IS_INERT_IN_EFFECT_NOT_UNREACHED

    budget = " ".join(record["how_the_budget_is_inert"].split())
    assert "it never reaches the fit" in budget
    assert "ROUND_2_IS_ROUND_1_TRUNCATED" in budget

    patience = " ".join(record["how_patience_is_inert"].split())
    assert "reaches the fit and the fit obeys it" in patience
    assert "the extra epochs are computed, evaluated, and then not chosen" in patience

    line = " ".join(record["the_distinction_in_one_line"].split())
    assert "One is free and the other is paid for" in line

    # The ratio is REPORTED and the record says so rather than banking it.
    reported = " ".join(record["what_it_is_paid_for_is_REPORTED_not_read"].split())
    assert "[REPORTED]" in reported
    assert "The seconds are not banked and criterion 6 is still open" in reported

    why = " ".join(
        record["why_the_correction_is_worth_making_before_the_seconds_arrive"].split()
    )
    assert "invites a reader to conclude that raising patience is free" in why
    assert "should not leave behind a reason to pay for it" in why


def test_the_closing_carries_the_correction_where_it_flattened_the_case():
    closing = phase26.PHASE_26_CLOSING
    correction = " ".join(closing["the_finding_CORRECTED_2026_09_06"].split())
    assert "flattens a distinction that matters" in correction
    assert "Patience moves no OUTPUT" in correction
    # The original sentence is preserved beside it.
    assert "Patience moves nothing at all, exactly" in closing["the_finding"]

    fired = closing["readings_that_fired"]
    note = " ".join(fired["patience_does_NOT_move_the_selected_epoch"].split())
    assert "the conclusion holds and the equivalence does not" in note


def test_criterion_six_closed_where_the_record_said_it_would():
    """[REPLACED 2026-09-06] This test was
    test_criterion_six_is_still_open_and_says_what_it_needs. The
    criterion closed, so the pin now checks that it closed in the
    place and the form the open record specified, which is a stronger
    check than either half alone."""
    record = phase26.WHAT_CRITERION_3_DID_NOT_GET

    # What the open entry said it needed is preserved verbatim.
    still = " ".join(record["criterion_6_STILL_OPEN_2026_09_06"].split())
    assert "the wall clock did not arrive" in still
    assert "in each cell run's LOG" in still
    assert "needs no re-run" in still

    form = " ".join(record["and_the_form_it_must_take_when_it_arrives"].split())
    assert "seconds per seed BY PATIENCE LEVEL, not one number" in form
    assert "The unit stays the fold fit" in form

    # And the close met both, in that place and in that form.
    closed = " ".join(record["criterion_6_CLOSED_2026_09_06"].split())
    assert "wall_clock_seconds_by_seed" in closed
    assert "it needed no re-run" in closed
    assert "seconds per seed by patience level, never one number" in closed

    runtime = phase26.THE_RUNTIME_MEASURED
    assert set(runtime["seconds_per_seed_by_patience"]) == {"5", "10", "20", "30"}
    assert set(runtime["seconds_per_fold_fit_by_patience"]) == {
        "5", "10", "20", "30",
    }
    assert "seconds per seed" in runtime["unit"]

    # The defect that caused the criterion to fail is not claimed fixed.
    defect = " ".join(record["and_the_defect_that_caused_it_is_NOT_fixed"].split())
    assert "still looks for ``summary.json``" in defect
    assert "both fixes remain owed" in defect


def test_the_phase_banks_a_runtime_of_its_own():
    """[REPLACED 2026-09-06] This test was
    test_the_phase_still_banks_no_runtime_of_its_own and it asserted
    that no runtime appeared anywhere in the phase. The wall clock
    arrived, so the pin is inverted: the runtime must be there, the
    entry that denied it must still be readable, and the denial must
    be corrected rather than deleted."""
    record = phase26.WHAT_CRITERION_3_DID_NOT_GET

    # The state at the close is preserved.
    cost = " ".join(record["what_it_costs_and_what_it_does_not"].split())
    assert "no banked runtime for a frozen head refit" in cost
    assert "remains an expectation" in cost

    # And corrected beside itself rather than overwritten.
    fixed = " ".join(record["and_what_it_COSTS_is_corrected_2026_09_06"].split())
    assert "Neither holds" in fixed
    assert "the first of its shape in this record" in fixed
    # The half that still holds is named as still holding.
    assert "no finding depended on the timing then and none depends on it now" in fixed

    # The ruling that asked for the timing records that it got it.
    discharged = " ".join(phase26.TIMING_IS_REPORTED["DISCHARGED_2026_09_06"].split())
    assert "the timing is no longer reported" in discharged
    assert "the record does not rewrite what it ruled" in discharged

    # And a runtime in seconds is now stated somewhere in the phase.
    import re

    blob = " ".join(str(v) for v in phase26.summary().values())
    assert re.search(r"\d+(\.\d+)?\s*seconds? per", blob)


# --------------------------------------------------------------------------
# the runtime, criterion 6, 2026-09-06
# --------------------------------------------------------------------------

CLOCK_FILE = Path("X:/p26_wall_clock.json")

# Every banked figure below is re-derived from the file. If the reader
# cannot reach the file it skips rather than trusting the record.
PATIENCE_LEVELS = (5, 10, 20, 30)
DECAY_LEVELS = (0.0, 0.001, 0.01, 0.1)
# With the best checkpoint at epoch 1 and the budget held at 30, a loop
# with patience p stops at min(1 + p, 30).
EPOCHS_RUN = {5: 6, 10: 11, 20: 21, 30: 30}


def _clock():
    import json

    import pytest

    if not CLOCK_FILE.is_file():
        pytest.skip("the wall-clock file is not on this machine")
    payload = json.loads(CLOCK_FILE.read_text(encoding="utf-8"))
    assert len(payload) == 16
    for cell, seeds in payload.items():
        assert len(seeds) == 5, cell
    return payload


def _cell_means(payload):
    import statistics

    return {
        (iw, ip): statistics.mean(payload[f"w{iw}_p{ip}"].values())
        for iw in range(4)
        for ip in range(4)
    }


def test_the_runtime_is_banked_by_patience_level_and_re_derives():
    """Criterion 6. The figures the record states are recomputed from the
    file, in both units, and the shape the ruling fixed is checked."""
    import statistics

    payload = _clock()
    record = phase26.THE_RUNTIME_MEASURED

    for ip, p in enumerate(PATIENCE_LEVELS):
        pooled = [
            v for iw in range(4) for v in payload[f"w{iw}_p{ip}"].values()
        ]
        assert len(pooled) == 20, p
        assert record["seconds_per_seed_by_patience"][str(p)] == _approx(
            statistics.mean(pooled)
        )
        assert record["seed_sd_by_patience"][str(p)] == _approx(
            statistics.stdev(pooled)
        )
        assert record["seconds_per_fold_fit_by_patience"][str(p)] == _approx(
            statistics.mean(pooled) / 5, places=4
        )

    # The unit is stated and it is not one number.
    assert "seconds per seed" in record["unit"]
    assert "divided by five" in record["unit"]
    assert len(record["seconds_per_seed_by_patience"]) == 4

    # The grid total, and the range a single cell spans.
    allvals = [v for cell in payload.values() for v in cell.values()]
    assert len(allvals) == 80
    whole = " ".join(record["and_the_grid_as_a_whole"].split())
    assert f"{sum(allvals):.1f} seconds" in whole
    assert f"{sum(allvals) / 60:.2f} minutes" in whole
    totals = [sum(cell.values()) for cell in payload.values()]
    assert f"{min(totals):.1f} to {max(totals):.1f} seconds" in whole


def _approx(value, places=3):
    import pytest

    return pytest.approx(value, abs=0.5 * 10 ** (-places))


def test_the_fourfold_figure_is_measured_and_holds_in_every_group():
    """It was REPORTED. It survives the reading, so it is promoted rather
    than restated."""
    payload = _clock()
    means = _cell_means(payload)

    ratios = [means[(iw, 3)] / means[(iw, 0)] for iw in range(4)]
    record = phase26.THE_COST_OF_PATIENCE_MEASURED
    promoted = " ".join(record["the_fourfold_figure_SURVIVES_and_is_promoted"].split())
    assert "[MEASURED, was REPORTED]" in promoted
    for iw, decay in enumerate(DECAY_LEVELS):
        assert f"{ratios[iw]:.3f} at weight decay {decay}" in promoted or (
            f"{ratios[iw]:.3f} at {decay}" in promoted
        ), (decay, ratios[iw])
    import statistics

    assert f"{statistics.mean(ratios):.3f} on average" in promoted
    # And it really is roughly fourfold rather than rounded into shape.
    assert 3.5 < min(ratios) and max(ratios) < 5.0


def test_the_cost_is_linear_in_epochs_rather_than_a_ramp_or_a_step():
    """The question was whether patience is paid smoothly or in one step.
    The answer is neither: it is linear in the epochs the loop runs, and
    patience sets those non-linearly."""
    import statistics

    payload = _clock()
    means = _cell_means(payload)

    # Fit on the two levels every weight decay group agrees on.
    y1 = statistics.mean(means[(iw, 0)] for iw in range(4))
    y2 = statistics.mean(means[(iw, 3)] for iw in range(4))
    b = (y2 - y1) / (EPOCHS_RUN[30] - EPOCHS_RUN[5])
    a = y1 - b * EPOCHS_RUN[5]

    record = phase26.THE_COST_OF_PATIENCE_MEASURED
    fit = " ".join(record["but_it_is_NEITHER_a_ramp_NOR_a_step_in_patience"].split())
    assert f"seconds = {a:.4f} + {b:.4f} * epochs" in fit
    assert "6, 11, 21 and 30 epochs" in fit

    def _within(tol):
        return sum(
            1
            for iw in range(4)
            for ip, p in enumerate(PATIENCE_LEVELS)
            if abs(means[(iw, ip)] / (a + b * EPOCHS_RUN[p]) - 1) <= tol
        )

    assert (_within(0.07), _within(0.10), _within(0.15)) == (11, 12, 13)
    assert "11 of the 16 cells to within 7 percent, 12 within 10 and 13 within 15" in fit
    assert "it misses them by half" in fit

    overhead = " ".join(record["the_fixed_overhead_is_small_and_real"].split())
    assert f"{a:.4f} seconds before any epoch runs" in overhead

    answer = " ".join(record["so_the_answer_to_smooth_or_stepped"].split())
    assert "smooth in epochs, uneven in patience" in answer
    assert "Nothing special happens at any patience level" in answer
    assert "buying epochs at a fixed price per epoch" in answer


def test_weight_decay_does_not_affect_the_cost():
    """Asked directly, and answered from the spread across the whole
    hundredfold decay range against the seed noise."""
    import statistics

    payload = _clock()
    means = _cell_means(payload)

    spreads = {}
    for ip, p in enumerate(PATIENCE_LEVELS):
        row = [means[(iw, ip)] for iw in range(4)]
        spreads[p] = max(row) / min(row)
    rel_sd = statistics.median(
        statistics.stdev(payload[f"w{iw}_p{ip}"].values())
        / statistics.mean(payload[f"w{iw}_p{ip}"].values())
        for iw in range(4)
        for ip in range(4)
    )

    record = " ".join(
        phase26.THE_COST_OF_PATIENCE_MEASURED[
            "weight_decay_does_NOT_affect_the_cost"
        ].split()
    )
    assert f"{spreads[5]:.3f} at patience 5" in record
    assert f"{spreads[10]:.3f} at 10 and {spreads[30]:.3f} at 30" in record
    assert f"median within-cell relative seed sd of {rel_sd:.3f}" in record
    assert f"patience 20 at {spreads[20]:.3f}" in record

    # Three of four levels inside the noise, and the fourth named as the
    # anomaly rather than as an effect of decay.
    for p in (5, 10, 30):
        assert spreads[p] - 1 < 3 * rel_sd, p
    assert "that is the anomaly below rather than an effect of decay" in record


def test_the_three_anomalous_cells_are_named_rather_than_averaged_away():
    """Three patience-20 cells cost about half what the linear model
    predicts. The record states it as an anomaly and does not explain it
    away, and it does not let it touch a finding it cannot touch."""
    import statistics

    payload = _clock()
    means = _cell_means(payload)
    y1 = statistics.mean(means[(iw, 0)] for iw in range(4))
    y2 = statistics.mean(means[(iw, 3)] for iw in range(4))
    b = (y2 - y1) / (EPOCHS_RUN[30] - EPOCHS_RUN[5])
    a = y1 - b * EPOCHS_RUN[5]

    pred21 = a + b * EPOCHS_RUN[20]
    pred11 = a + b * EPOCHS_RUN[10]
    record = phase26.THE_COST_OF_PATIENCE_MEASURED
    anomaly = " ".join(
        record["THE_ANOMALY_THREE_CELLS_COST_HALF_WHAT_THE_MODEL_PREDICTS"].split()
    )

    # The three non-zero decay cells at patience 20, in order.
    off = [means[(iw, 2)] for iw in (1, 2, 3)]
    assert f"cost {off[0]:.3f}, {off[1]:.3f}" in anomaly
    assert f"and {off[2]:.3f} seconds per seed" in anomaly
    assert f"against a predicted {pred21:.3f}" in anomaly
    for obs in off:
        assert f"{obs / pred21:.3f}" in anomaly
        assert 0.5 < obs / pred21 < 0.6
    # And they match the eleven-epoch prediction instead.
    assert f"ELEVEN-epoch prediction of {pred11:.3f}" in anomaly
    for obs in off:
        assert abs(obs / pred11 - 1) < 0.05

    # The zero-decay cell at the same patience does fit the 21-epoch line.
    zero = means[(0, 2)]
    assert abs(zero / pred21 - 1) < 0.05
    assert "is the only one that does" in anomaly

    # Not seed noise: every seed in each cell is near the low value.
    not_noise = " ".join(record["and_it_is_not_seed_noise"].split())
    assert "every seed in each of those three cells is near 6 seconds" in not_noise
    for iw in (1, 2, 3):
        for seed_seconds in payload[f"w{iw}_p2"].values():
            assert 4.0 < seed_seconds < 9.0, (iw, seed_seconds)
    for seed_seconds in payload["w0_p2"].values():
        assert seed_seconds > 9.0

    # It is not claimed to be understood.
    unknown = " ".join(record["what_it_would_mean_and_what_is_NOT_established"].split())
    assert "Whether that is what happened is not established here" in unknown
    assert "needs the per-epoch counts" in unknown

    # It touches no finding, and the record says which thing it does touch.
    touches = " ".join(record["what_it_does_and_does_not_touch"].split())
    assert "it touches no finding" in touches
    assert "All 400 folds selected epoch 1" in touches
    assert "What it does touch is the completeness of the axis" in touches
    assert "the 20 level was exercised once rather than four times" in touches


def test_the_scheduling_claim_is_measured_and_was_generous():
    """The eighth amendment scheduled this phase on a cost claim. That
    claim is now a measurement, and the record says by how much it was
    off rather than only that it held."""
    from cleft import phase20

    payload = _clock()
    record = phase26.THE_SCHEDULING_CLAIM_SETTLED

    totals = [sum(cell.values()) for cell in payload.values()]
    measured = " ".join(record["what_is_measured"].split())
    assert "a cell runs in SECONDS, not minutes" in measured
    assert f"cheapest cell take {min(totals):.1f} seconds" in measured
    assert f"of the dearest {max(totals):.1f}" in measured
    assert "generous by a factor of about ten for a cell" in measured

    # The expectation it settles is live at its home and unchanged.
    settles = " ".join(record["the_expectation_it_settles"].split())
    assert "Expectation: seconds" in settles
    gate = phase20.COMPUTE_GATE_DESIGNED
    assert "Expectation: seconds" in gate["why_it_is_probably_trivial"]
    assert "the expectation is not the measurement" in settles

    # And it is the phase's own contribution rather than a borrowed one.
    first = " ".join(record["and_it_is_the_first_of_its_shape"].split())
    assert "no banked runtime for a frozen head refit on cached" in first
    assert "29,286,981 parameters" in first
    assert "a different quantity under a similar name" in first
    assert "outlives the null it was collected beside" in first


def test_the_runtime_is_the_only_positive_thing_the_phase_banks():
    """The phase closed on a null. The runtime is a measurement it made
    beside that null, and the record must not let it grow into a result
    about the settings."""
    prohibited = phase26.WHAT_THIS_PHASE_MAY_NOT_CLAIM
    assert prohibited  # the prohibitions are unchanged by the timing

    runtime = phase26.THE_RUNTIME_MEASURED
    assert "criterion 6 met" in runtime["tag"]
    assert "the first runtime of this shape" in runtime["tag"]

    # It says nothing about which setting is better, only what each costs.
    blob = " ".join(
        str(v)
        for v in (
            list(runtime.values())
            + list(phase26.THE_COST_OF_PATIENCE_MEASURED.values())
            + list(phase26.THE_SCHEDULING_CLAIM_SETTLED.values())
        )
    )
    for forbidden in ("better", "improves", "outperforms", "recommend"):
        assert forbidden not in blob.lower(), forbidden
