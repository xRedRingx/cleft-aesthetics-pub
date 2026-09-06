"""Phase 22's post-run steps: the diagnostic, shrinkage, and contrasts.

**Every value these configs declare is traced to a consumption site**,
not to a log line -- the standard the settings sweep set after
``pair_source`` reached only a log line and made R-all and R-clear the
same arm (``phase22.PAIR_SOURCE_WAS_NOT_CONSUMED``).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from cleft import phase21, phase22
from cleft.config import schema

REPO = Path(__file__).resolve().parents[1]

STEMS = ("p22_diagnostic", "p22_shrinkage", "p22_contrasts")
NEW_ARMS = (
    "r_all_bounded", "r_all_unbounded",
    "r_clear_bounded", "r_clear_unbounded",
)


def _config(stem: str) -> dict:
    return yaml.safe_load(
        (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
    )


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# the banked outturns
# --------------------------------------------------------------------------


def test_the_four_arms_outturns_reproduce_from_their_per_seed_values():
    record = phase22.COHORT_ARMS_OBSERVED
    assert sorted(record["arms"]) == sorted(NEW_ARMS)
    for name, arm in record["arms"].items():
        values = np.array(arm["per_seed"])
        assert len(values) == 5, name
        assert round(float(values.mean()), 4) == arm["mean"], name
        assert round(float(values.std(ddof=1)), 4) == arm["sd"], name

    # The provenance says plainly that the artifacts were not readable.
    provenance = _flat(record["provenance"])
    assert "CLUSTER-ONLY" in provenance
    assert "runs/keeper` does not exist on the laptop" in provenance
    assert "COMPUTED here from those five values" in provenance


def test_the_bounded_arms_sit_inside_the_unresolvable_band():
    from cleft import ladder

    probe = ladder.TRADE_OFF_PAIR["result"]["vit_paired_mean"]
    assert probe == 0.2520
    arms = phase22.COHORT_ARMS_OBSERVED["arms"]
    for name in ("r_all_bounded", "r_clear_bounded"):
        assert abs(arms[name]["mean"] - probe) < 0.04, name
    # The unbounded arms are much worse, and still short of the
    # smallest delta this project has ever resolved.
    floor = ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"]["delta"]
    assert floor == 0.1386
    for name in ("r_all_unbounded", "r_clear_unbounded"):
        delta = abs(arms[name]["mean"] - probe)
        assert 0.10 < delta < floor, (name, delta)

    stated = _flat(phase22.COHORT_ARMS_OBSERVED["what_the_point_estimates_show"])
    assert "No verdict is read here" in stated


def test_the_census_difference_matches_the_separation_profile():
    stated = _flat(
        phase22.COHORT_ARMS_OBSERVED["the_census_confirms_distinct_pair_sets"]
    )
    assert "25,621" in stated and "21,017" in stated and "4,604" in stated
    # 4,604 is step 1 of the banked profile: the non-tied pairs below
    # SE_diff, which is exactly what R-clear excludes and R-all keeps.
    profile = phase22.TIE_FRACTION_OBSERVED[
        "separation_profile_in_rater_steps"]
    assert int(profile["1"]) == 4604
    assert 25621 - 21017 == 4604
    # And 25,621 is the tie-dropped total.
    assert 27966 - 2345 == 25621


def test_only_eight_of_the_fifteen_are_evaluable_and_seven_are_deferred():
    record = phase22.FAMILY_EVALUABLE_SUBSET
    assert record["evaluable_now"]["count"] == 8
    assert 8 + 7 == len(phase22.contrast_family()) == 15
    deferred = _flat(record["waiting_on_r_syn"])
    assert "SEVEN" in deferred
    assert "axis question is entirely unevaluable" in deferred
    # Deferred, not withdrawn -- criterion 8 still owes them a verdict.
    nothing = _flat(record["nothing_is_added_or_dropped"])
    assert "DEFERRED, not withdrawn" in nothing
    assert "all fifteen" in _flat(
        phase22.EXIT_CRITERIA["8_every_contrast_gets_a_verdict"]
    )


# --------------------------------------------------------------------------
# the configs
# --------------------------------------------------------------------------


def test_the_three_configs_exist_and_match_their_generator():
    import subprocess
    import sys

    for stem in STEMS:
        assert (REPO / "configs" / f"{stem}.yaml").is_file(), stem
    result = subprocess.run(
        [sys.executable, "scripts/generate_phase22_postrun_configs.py",
         "--check"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


NEW_ARM_INPUTS = tuple(f"p22_{a}" for a in NEW_ARMS)


def test_the_four_arm_hashes_are_filled_and_everything_else_is_verbatim():
    """[UPDATED 2026-09-01, THIS PIN FIRED AS DESIGNED] It held the four
    arm inputs at all-zero placeholders. the declare filled them,
    so the pin now holds what it was always standing for: the four are
    REAL, and every other input is a shipped Phase 21 config's own."""
    donors = {}
    for donor_stem in ("p21_error_consistency", "p21_cross_arm_shrinkage"):
        for entry in _config(donor_stem)["inputs"]:
            donors[entry["name"]] = entry
    for stem in STEMS:
        config = _config(stem)
        # No placeholder remains anywhere.
        assert not [
            e["name"] for e in config["inputs"]
            if set(e["rollup_sha256"]) == {"0"}
        ], stem
        for entry in config["inputs"]:
            if entry["name"] in NEW_ARM_INPUTS:
                continue
            assert entry == donors[entry["name"]], (stem, entry["name"])


def test_each_arm_carries_the_same_hash_in_all_three_configs():
    """A per-config divergence would mean one config points at a
    different object under the same name."""
    by_stem = {
        stem: {e["name"]: e["rollup_sha256"] for e in _config(stem)["inputs"]}
        for stem in STEMS
    }
    for arm in NEW_ARM_INPUTS:
        digests = {by_stem[stem][arm] for stem in STEMS}
        assert len(digests) == 1, (arm, digests)
        digest = digests.pop()
        assert len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        assert set(digest) != {"0"}, arm
    # Four DISTINCT run directories, four distinct hashes.
    assert len({by_stem[STEMS[0]][arm] for arm in NEW_ARM_INPUTS}) == 4


def test_the_diagnostic_excludes_arm_a_from_the_comparison_set():
    config = _config("p22_diagnostic")
    task = config["task"]
    names = [a["name"] for a in task["arms"]]
    assert "p17_arm_a" not in names, (
        "arm A is the reference the LOW threshold comes from; including "
        "it would compare it with itself"
    )
    assert task["expect_comparison_arms"] == 63
    assert len(names) == 63 + 4
    assert sorted(task["subject_arms"]) == sorted(NEW_ARMS)
    # The 63 really are the shipped 64 minus arm A.
    shipped = {a["name"] for a in _config("p21_error_consistency")["task"]["arms"]}
    assert len(shipped) == 64
    assert set(names) - set(NEW_ARMS) == shipped - {"p17_arm_a"}


def test_the_contrast_config_resolves_every_evaluable_contrast():
    """**The join that would have failed at launch.** The family names
    arms bare; the declared INPUTS are p22-prefixed. The first generated
    version used the prefixed name for both and resolved ZERO contrasts
    against a config declaring eight."""
    config = _config("p22_contrasts")
    available = {a["name"] for a in config["task"]["arms"]}
    family = phase22.contrast_family()
    evaluable = [
        c for c in family if c["a"] in available and c["b"] in available
    ]
    assert len(evaluable) == config["task"]["expect_evaluable"] == 8
    assert {c["kind"] for c in evaluable} == {"primary", "secondary"}
    assert len([c for c in evaluable if c["kind"] == "primary"]) == 4
    # The probe is declared under the name the family uses for it.
    assert phase22.PROBE in available


def test_every_config_validates():
    from cleft.config.schema import validate

    for stem in STEMS:
        validate(_config(stem))


# --------------------------------------------------------------------------
# THE CONSUMPTION SWEEP -- every declared value reaches acting code
# --------------------------------------------------------------------------


def test_the_diagnostic_reads_its_thresholds_from_phase21_not_the_config():
    """The config DECLARES what it expects to find and the task READS
    the real values -- criterion 6, so a changed source figure stops the
    phase rather than being read past."""
    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p22_diagnostic)
    # Read from phase21...
    assert 'phase21.ARM_B_OBSERVED["residual_sign"]["mean_kappa"]' in body
    assert 'phase21.ARM_A_THE_SHRINKAGE_CONTROL["kappa_measured"]' in body
    # ...and the config's values are CHECKED, never used as thresholds.
    assert 'task["expect_high_kappa"]' in body
    assert 'task["expect_low_kappa"]' in body
    assert "criterion 6" in body

    # The declared expectations match the live sources today.
    task = _config("p22_diagnostic")["task"]
    assert task["expect_high_kappa"] == {
        "residual_sign": phase21.ARM_B_OBSERVED["residual_sign"]["mean_kappa"],
        "worst_quartile": phase21.ARM_B_OBSERVED["worst_quartile"]["mean_kappa"],
    }
    assert task["expect_low_kappa"] == (
        phase21.ARM_A_THE_SHRINKAGE_CONTROL["kappa_measured"]
    )


def test_the_low_threshold_mirror_agrees_with_the_prose_it_mirrors():
    """The standing rule forbids regexing values out of prose, so the
    figures are mirrored structurally -- and the mirror is checked
    against the sentence so the two cannot drift apart."""
    record = phase21.ARM_A_THE_SHRINKAGE_CONTROL
    mirror = record["kappa_measured"]
    prose = _flat(record["the_control"])
    assert f"{mirror['residual_sign']:+.4f}" in prose
    assert f"{mirror['worst_quartile']:+.4f}" in prose
    assert mirror == {"residual_sign": 0.0956, "worst_quartile": -0.0743}


def test_the_diagnostic_uses_the_shipped_kappa_implementation():
    """Reuse, not a second implementation."""
    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p22_diagnostic)
    assert "error_indicators(" in body
    assert "error_consistency_matrix(" in body
    # The arithmetic is not duplicated into this task.
    assert "c_exp" not in body and "c_obs" not in body
    # And the Phase 21 task calls the very same functions.
    shipped = inspect.getsource(run_module.task_p21_error_consistency)
    assert "error_indicators(" in shipped
    assert "error_consistency_matrix(" in shipped


def test_the_contrast_task_uses_paired_comparison_unchanged():
    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p22_contrasts)
    assert "from .phase7b import paired_comparison" in body
    assert "paired_comparison(" in body
    # The pair list is DERIVED, not a config field.
    assert "phase22.contrast_family()" in body
    assert "contrasts" not in schema.TASK_SPECS["p22_contrasts"]
    assert "pairs" not in schema.TASK_SPECS["p22_contrasts"]
    # The locked count is asserted inside the task.
    assert "!= 15" in body


def test_every_declared_value_in_the_new_configs_reaches_acting_code():
    """**The sweep, by the standard the settings sweep set.** Each
    declared field is traced to a line that ACTS on it -- not a log line
    and not a metrics key."""
    from cleft import run as run_module

    diagnostic = inspect.getsource(run_module.task_p22_diagnostic)
    contrasts = inspect.getsource(run_module.task_p22_contrasts)

    # (field, task source, the expression that consumes it)
    consumed = [
        ("manifest_artifact", diagnostic, 'declared[task["manifest_artifact"]]'),
        ("arms", diagnostic, "_p21_load_arm_predictions"),
        ("subject_arms", diagnostic, 'list(task["subject_arms"])'),
        ("expect_comparison_arms", diagnostic,
         'task["expect_comparison_arms"]'),
        ("binarisations", diagnostic, 'task["binarisations"]'),
        ("expect_high_kappa", diagnostic, 'task["expect_high_kappa"]'),
        ("expect_low_kappa", diagnostic, 'task["expect_low_kappa"]'),
        ("manifest_artifact", contrasts, 'declared[task["manifest_artifact"]]'),
        ("arms", contrasts, "_p21_load_arm_predictions"),
        ("expect_evaluable", contrasts, 'task["expect_evaluable"]'),
        ("winner_sd", contrasts, 'task["winner_sd"]'),
        ("n_boot", contrasts, 'task["n_boot"]'),
    ]
    for field, source, expression in consumed:
        assert expression in source, (field, expression)

    # Every field on each spec is either consumed above, or `kind`, or
    # an expect_patients guard -- nothing is declared and unread.
    accounted = {
        "kind", "expect_patients", "seeds", "manifest_artifact", "arms",
        "subject_arms", "expect_comparison_arms", "binarisations",
        "expect_high_kappa", "expect_low_kappa", "expect_evaluable",
        "winner_sd", "n_boot",
    }
    for stem in ("p22_diagnostic", "p22_contrasts"):
        assert set(schema.TASK_SPECS[stem]) <= accounted, stem


def test_the_shrinkage_config_reuses_the_shipped_task_unchanged():
    """No new task: Phase 21's cross_arm_shrinkage with four arms added."""
    config = _config("p22_shrinkage")
    assert config["task"]["kind"] == "cross_arm_shrinkage"
    donor = _config("p21_cross_arm_shrinkage")["task"]
    # The reading thresholds are carried verbatim -- this run adds arms,
    # it does not re-rule anything.
    for field in ("fails_to_shrink_at", "shrinks_substantially_at",
                  "varies_widely_at", "probe_arm"):
        assert config["task"][field] == donor[field], field
    # Four arms added, and the declared count matches.
    assert len(config["task"]["arms"]) == len(donor["arms"]) + 4
    assert config["task"]["expect_arms"] == len(config["task"]["arms"])


def test_the_shrinkage_run_tests_the_registered_prediction():
    """It fires no cell and gates nothing -- it was registered to be
    refuted, and these four arms are the first data bearing on it."""
    record = phase22.BOUNDED_HEAD_PREDICTION_REGISTERED
    prediction = _flat(record["the_prediction"])
    assert "BOUNDED ranking arm shrinks toward the label mean" in prediction
    assert "An UNBOUNDED one does not" in prediction
    assert "fires no cell and gates nothing" in _flat(
        record["how_it_can_be_refuted"]
    )
    # The four arms in the shrinkage config are the ones that bear on it.
    names = {a["name"] for a in _config("p22_shrinkage")["task"]["arms"]}
    assert set(NEW_ARMS) <= names


def test_no_r_syn_arm_appears_in_any_post_run_config():
    """The two R-syn arms are unrun; nothing may reference them."""
    for stem in STEMS:
        config = _config(stem)
        text = yaml.safe_dump(config)
        assert "r_syn" not in text, stem


# --------------------------------------------------------------------------
# [2026-09-01] The run-directory names, and what may not be constructed
# --------------------------------------------------------------------------


def test_the_run_directory_names_are_the_supplied_ones():
    """**Pre-fix all twelve entries pointed at directories that do not
    exist**: the paths were built from the `<stem>__<sha8>__<job-id>`
    contract, and every real name ends `-2`."""
    supplied = {
        "r_all_bounded": "p22_r_all_bounded__44ad1f7e__p22-r-all-bounded-2",
        "r_all_unbounded":
            "p22_r_all_unbounded__44ad1f7e__p22-r-all-unbounded-2",
        "r_clear_bounded":
            "p22_r_clear_bounded__44ad1f7e__p22-r-clear-bounded-2",
        "r_clear_unbounded":
            "p22_r_clear_unbounded__44ad1f7e__p22-r-clear-unbounded-2",
    }
    for stem in STEMS:
        by_name = {e["name"]: e for e in _config(stem)["inputs"]}
        for arm, directory in supplied.items():
            entry = by_name[f"p22_{arm}"]
            assert entry["path"].endswith("/" + directory), (stem, arm)
            # The suffix no contract predicts.
            assert entry["path"].endswith("-2"), (stem, arm)


def test_the_generator_never_constructs_a_run_directory_name():
    """The half of the problem that IS checkable locally.

    A cluster-only path cannot be verified here -- guard 3 at launch is
    the only check that sees the truth -- but a CONSTRUCTED path can be
    prevented, and that is what this holds.
    """
    source = (
        REPO / "scripts" / "generate_phase22_postrun_configs.py"
    ).read_text(encoding="utf-8")

    # The names live in a supplied table.
    assert "RUN_DIRECTORIES = {" in source
    assert "ls runs/keeper/p22" in source
    # And the builder looks one up rather than formatting one.
    assert "RUN_DIRECTORIES.get(arm)" in source
    assert "do not construct it" in source
    # No f-string builds a run-directory name from its parts.
    assert "__{RUN_SHA8}__" not in source
    assert "arm.replace('_', '-')" not in source


def test_the_generator_refuses_an_arm_with_no_supplied_name(monkeypatch):
    """It raises rather than falling back to the contract."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen_postrun",
        REPO / "scripts" / "generate_phase22_postrun_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setitem(module.RUN_DIRECTORIES, "r_all_bounded", "")
    with pytest.raises(SystemExit, match="do not construct it"):
        module._new_arm_inputs()


def test_the_derived_values_really_come_from_their_sources():
    """Everything the audit moved from typed to derived."""
    from cleft import ladder

    diagnostic = _config("p22_diagnostic")["task"]
    contrasts = _config("p22_contrasts")["task"]

    # seeds and n_patients: the arm configs' own.
    arm = yaml.safe_load(
        (REPO / "configs" / "p22_r_all_bounded.yaml").read_text(
            encoding="utf-8")
    )["task"]
    assert diagnostic["seeds"] == arm["seeds"]
    assert diagnostic["expect_patients"] == arm["expect_patients"]

    # the probe stem and its sd: ladder's.
    assert contrasts["winner_sd"] == ladder.TRADE_OFF_PAIR["result"]["vit_sd"]
    probe_stem = ladder.TRADE_OFF_PAIR["best_arm"]["stem"]
    inputs = {e["name"] for e in _config("p22_contrasts")["inputs"]}
    assert probe_stem in inputs

    # the thresholds: phase21's.
    assert diagnostic["expect_high_kappa"] == {
        b: phase21.ARM_B_OBSERVED[b]["mean_kappa"]
        for b in ("residual_sign", "worst_quartile")
    }
    assert diagnostic["expect_low_kappa"] == (
        phase21.ARM_A_THE_SHRINKAGE_CONTROL["kappa_measured"]
    )

    # the csv stem: the writer's.
    writer = (REPO / "src" / "cleft" / "train" / "phase3.py").read_text(
        encoding="utf-8"
    )
    assert 'f"{prefix}predictions.csv"' in writer
    for entry in diagnostic["arms"]:
        if entry["name"] in NEW_ARMS:
            assert entry["csv"] == "predictions"


def test_the_reconstruction_audit_names_what_is_still_constructed():
    """An audit that lists only what was fixed is an audit that hides
    the remainder."""
    record = phase22.POSTRUN_RECONSTRUCTION_AUDIT
    still = record["reconstructed_and_still_reconstructed"]
    assert sorted(still) == ["group_label", "run_root"]
    assert "INVENTED" in _flat(still["group_label"])
    assert "remains constructed" in _flat(still["run_root"])

    fixed = record["also_reconstructed_and_now_derived"]
    assert sorted(fixed) == [
        "n_patients", "seeds", "the_csv_stem", "the_kappa_thresholds",
        "the_probe_stem", "winner_sd",
    ]
    # The one that would have mattered most is named as such.
    assert "wrong baseline" in _flat(fixed["the_probe_stem"])

    how = _flat(record["how_such_names_are_obtained_in_future"])
    assert "SUPPLIED FROM AN `ls`" in how
    assert "REFUSES" in how


def test_the_check_feasibility_is_reported_honestly():
    record = phase22.RUN_PATH_CHECK_FEASIBILITY
    assert "a load-time existence check" in _flat(record["what_is_impossible"])
    guard = _flat(record["the_real_guard_is_guard_3"])
    assert "at launch, on the cluster" in guard
    assert "only check that can see the truth" in guard
    checkable = _flat(record["what_IS_checkable_locally_and_is_now_checked"])
    assert "cannot be verified; a constructed path can be PREVENTED" in checkable
    # The residual risk is stated, not glossed.
    assert "smaller class, not an empty one" in _flat(record["the_residual_risk"])


# --------------------------------------------------------------------------
# [2026-09-02] Both PLAN 4.3 conditions reach the record
# --------------------------------------------------------------------------


def test_condition_1_is_returned_by_paired_comparison_not_rebuilt():
    """**[2026-09-02] The fix.** `all_exclude` was computed inside
    `paired_comparison` and discarded, so callers rebuilt it from three
    other fields -- and the Phase 22 task recorded neither condition,
    leaving eight verdicts with `condition_1: None`.

    The returned field is pinned EQUAL to that derivation, so the two
    cannot drift apart.
    """
    from cleft.phase7b import paired_comparison

    rng = np.random.default_rng(11)
    truth = rng.uniform(1.0, 5.0, 80)
    for winner_noise, baseline_noise in ((0.2, 1.2), (0.9, 0.95), (1.1, 0.3)):
        winner = {s: truth + rng.normal(0, winner_noise, 80) for s in (1, 2, 3)}
        baseline = {
            s: truth + rng.normal(0, baseline_noise, 80) for s in (1, 2, 3)
        }
        result = paired_comparison(
            truth=truth, winner_by_seed=winner, baseline_by_seed=baseline,
            winner_sd=0.0148, n_boot=200,
        )
        returned = result["all_seeds_exclude_zero_one_direction"]
        derived = (
            result["n_excluding_zero"] == result["n_seeds"]
            and result["same_direction"] is True
        )
        assert returned is derived, (returned, derived)
        # Neither condition may be null, and `claimable` is their
        # conjunction -- the rule computed where it lives.
        assert returned is not None
        assert result["exceeds_threshold"] is not None
        assert result["claimable"] == (returned and result["exceeds_threshold"])


def test_no_recorded_verdict_may_carry_a_null_condition():
    """**The test that would have caught it.** A verdict without both
    conditions is a verdict a reader must infer from an adjacent field."""
    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p22_contrasts)
    # Both conditions are read from the result and recorded.
    assert 'result["all_seeds_exclude_zero_one_direction"]' in body
    assert 'result["exceeds_threshold"]' in body
    assert '"condition_1": condition_1' in body
    assert '"condition_2": condition_2' in body
    # And a null is REFUSED rather than written -- asserted on the
    # guard itself, not on its message, which wraps across two source
    # lines and so never appears contiguously in either form.
    assert "if condition_1 is None or condition_2 is None:" in body
    assert "raise ValueError(" in body
    assert "may not be recorded without both" in body

    # Neither condition is rebuilt from n_excluding_zero. Checked
    # against the CODE, not the comment that explains why.
    code = chr(10).join(
        line for line in body.splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "n_excluding_zero" not in code


def test_the_shipped_phase_17_task_no_longer_keeps_a_second_copy():
    """One rule, one implementation: the tstr family task now reads the
    same returned field instead of rebuilding the conjunction."""
    from cleft import run as run_module

    body = inspect.getsource(run_module.task_tstr_family_analysis)
    assert 'result["all_seeds_exclude_zero_one_direction"]' in body
    assert 'result["same_direction"] is True' not in body


def test_the_verdict_vocabulary_distinguishes_withdrawn_from_unresolved():
    """`withdrawn` is condition 1 met and condition 2 not -- a real
    effect too small to clear the combined uncertainty. Collapsing it
    into `unresolved` would lose which condition failed, which is the
    thing this fix exists to preserve."""
    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p22_contrasts)
    assert '"withdrawn" if condition_1 and not condition_2' in body
    assert '"claimable" if result["claimable"]' in body


# --------------------------------------------------------------------------
# [2026-09-02] The close-out
# --------------------------------------------------------------------------


def test_the_shrinkage_prediction_is_recorded_as_refuted():
    """It was registered to be refuted, and it was -- in a direction
    nobody registered."""
    record = phase22.SHRINKAGE_OBSERVED_AND_PREDICTION_REFUTED
    measured = record["measured"]
    # The bounded arms are CALIBRATED, not shrunk.
    for name in ("r_all_bounded", "r_clear_bounded"):
        assert 1.0 < measured[name]["ratio"] < 1.1, name
    # The probe shrinks; the ladder median shrinks harder.
    probe = phase21.CROSS_ARM_SHRINKAGE_OBSERVED["probe"]
    assert probe == 0.4947
    for name in ("r_all_bounded", "r_clear_bounded"):
        assert measured[name]["ratio"] > probe

    refuted = _flat(record["the_prediction_is_refuted"])
    assert "It does not" in refuted
    assert "not shrunk, CALIBRATED" in refuted
    assert "no arm in the ladder achieves" in refuted
    # The half that held is recorded too, not just the half that failed.
    assert "do not shrink" in _flat(record["the_half_that_held"])
    assert "registered TO BE REFUTED and was" in _flat(
        record["it_fired_no_cell_and_gated_nothing"]
    )
    # The prediction it refutes said the opposite, in those words.
    assert "shrinks toward the label mean" in _flat(
        phase22.BOUNDED_HEAD_PREDICTION_REGISTERED["the_prediction"]
    )


def test_five_arms_do_not_shrink_and_none_beats_the_probe():
    record = phase22.FIVE_ARMS_DO_NOT_SHRINK
    five = record["the_five"]
    assert len(five) == 5
    probe = record["the_probe"]
    assert probe == {"shrinkage": 0.4947, "pcc": 0.2520}
    # Every one of the five is less shrunk than the probe...
    for name, arm in five.items():
        assert arm["shrinkage"] > probe["shrinkage"], name
    # ...and every one scores below it.
    assert all(a["pcc"] < probe["pcc"] for a in five.values())
    # Arm A's figures are the banked ones.
    assert five["p17_arm_a"]["shrinkage"] == (
        phase21.CROSS_ARM_SHRINKAGE_OBSERVED["across_arms"]["max"]
    )
    from cleft import phase17

    assert five["p17_arm_a"]["pcc"] == phase17.ARM_MEANS["arms"]["p17_arm_a"]["pcc"]
    # And the four cohort arms' PCCs are the banked means.
    banked = phase22.COHORT_ARMS_OBSERVED["arms"]
    for name in NEW_ARMS:
        assert five[name]["pcc"] == banked[name]["mean"], name

    why = _flat(record["why_the_calibrated_pair_is_the_strongest_case"])
    assert "no degeneracy left to blame" in why
    # The claim is bounded: calibration is not worthless.
    assert "not that calibration is worthless" in _flat(
        record["what_it_does_not_establish"]
    )


def test_the_diagnostic_splits_by_head_and_answers_the_hypothesis():
    record = phase22.DIAGNOSTIC_OBSERVED
    measured = record["measured"]
    # Bounded arms share more error structure than unbounded, on BOTH.
    for binarisation in ("residual_sign", "worst_quartile"):
        values = measured[binarisation]
        assert min(values["r_all_bounded"], values["r_clear_bounded"]) > max(
            values["r_all_unbounded"], values["r_clear_unbounded"]
        ), binarisation
    # The unbounded worst-quartile pair sits at arm A's control level.
    arm_a = phase21.ARM_A_THE_SHRINKAGE_CONTROL["kappa_measured"]
    for name in ("r_all_unbounded", "r_clear_unbounded"):
        assert abs(measured["worst_quartile"][name]) < abs(
            arm_a["worst_quartile"]
        ), name

    # Every cell is `between`, and that cell claims nothing.
    assert "all eight fired `kappa_between`" in _flat(
        record["every_cell_is_between"]
    )
    assert "claim neither" in _flat(record["every_cell_is_between"])

    answer = _flat(record["what_it_establishes_against_the_hypothesis"])
    assert "DID FIND DIFFERENT FEATURES, AND THE DIFFERENT FEATURES ARE WORSE" in answer
    assert "Different is not better" in answer
    assert "it is NEGATIVE" in answer
    # And it is the hypothesis as registered.
    assert "DIFFERENT FEATURES" in _flat(
        phase22.PHASE_22_RECKONING["the_hypothesis_that_survives"]
    )


def test_the_contrasts_are_eight_unresolved_with_the_noise_pair_null():
    record = phase22.CONTRASTS_OBSERVED
    deltas = record["deltas"]
    assert len(deltas) == 8
    # Every declared key is a real contrast in the locked family.
    keys = {c["key"] for c in phase22.contrast_family()}
    assert set(deltas) <= keys
    assert "EIGHT UNRESOLVED" in record["verdict"]

    # The cleanest null: a thousandth of a PCC point at both heads.
    for key in ("secondary__noise_pairs__bounded",
                "secondary__noise_pairs__unbounded"):
        assert abs(deltas[key]) <= 0.0010, key
    null = _flat(record["the_noise_pair_result_is_the_phases_cleanest_null"])
    assert "4,604" in null
    assert "SINGLE controlled factor" in null

    # **The four largest are all ~0.12 and all failed.** The head
    # contrasts are the largest SECONDARIES, not the largest overall --
    # the two unbounded primaries beat them in magnitude.
    ranked = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))
    assert [k for k, _ in ranked[:4]] == [
        "primary__r_all_unbounded_vs_probe",
        "primary__r_clear_unbounded_vs_probe",
        "secondary__head__r_all",
        "secondary__head__r_clear",
    ]
    floor = 0.1386
    for key, value in ranked[:4]:
        assert 0.11 < abs(value) < floor, key
    # A clear gap to the rest.
    assert abs(ranked[4][1]) < 0.01
    corrected = _flat(
        record["the_four_largest_deltas_are_all_about_0_12_and_all_failed"]
    )
    assert "CORRECTED 2026-09-02 before banking" in corrected
    assert "They are not" in corrected
    assert "largest SECONDARIES" in corrected
    # And the four are one effect read twice, not four findings.
    assert "same gap" in _flat(record["the_four_are_one_effect_seen_twice"])
    # A loss is not reported as nothing.
    assert "still not a loss it may report as nothing" in _flat(
        record["the_primaries"]
    )


def test_the_closing_walks_every_criterion_and_does_not_overclaim():
    record = phase22.PHASE_22_CLOSING
    walk = record["criterion_walk"]
    assert len(walk) == 14, sorted(walk)
    assert sorted(int(k.split("_")[0]) for k in walk) == list(range(1, 15))

    finding = _flat(record["the_finding"])
    for clause in ("MATCHES THE PROBE TO WITHIN 0.003",
                   "CALIBRATES BETTER THAN ANY ARM IN THE LADDER",
                   "BUYS NOTHING MEASURABLE ON PCC",
                   "FINDS DIFFERENT FEATURES AND THEY ARE WORSE",
                   "NOISE-PAIR QUESTION IS A CLEAN NULL"):
        assert clause in finding, clause

    # Criterion 8 is UNMET and the phase says so rather than closing.
    assert "NOT MET" in _flat(walk["8_every_contrast_gets_a_verdict"])
    assert "SEVEN ARE STILL OWED ONE" in _flat(
        walk["8_every_contrast_gets_a_verdict"]
    )
    assert record["tag"] == "[CLOSED, PARTIAL]"
    partial = _flat(record["the_phase_does_not_fully_close"])
    assert "stays open on the axis question" in partial
    assert "Saying it closed would be the shape this project has corrected" in partial

    # Criterion 2 is partial and says which arms did not run.
    assert "PARTIAL" in _flat(walk["2_six_arms_built"])
    # Criterion 13 held as a BLOCK -- the criterion working, not failing.
    assert "HELD AS A BLOCK" in _flat(
        walk["13_open_settings_ruled_before_the_run"]
    )
    # The void record is carried forward.
    assert "VOID" in _flat(record["what_was_void_and_stays_void"])
    # And the phase's own cost is on the record.
    cost = _flat(record["the_defects_this_phase_cost"])
    assert "six, all in one task" in cost
    assert "a check that now runs in the suite" in cost


# --------------------------------------------------------------------------
# [2026-09-02] The re-run's verdicts, and the two counts
# --------------------------------------------------------------------------


def test_the_eight_verdicts_split_four_and_four_over_the_same_deltas():
    record = phase22.CONTRAST_VERDICTS_OBSERVED
    passing = record["condition_2_pass_condition_1_fail"]
    failing = record["fail_both_conditions"]
    assert len(passing) == len(failing) == 4
    # Together they are exactly the banked deltas -- no contrast added,
    # dropped or re-valued between the two records.
    assert {**passing, **failing} == phase22.CONTRASTS_OBSERVED["deltas"]
    assert "ALL EIGHT UNRESOLVED" in record["verdict"]

    # The split is the four largest against the four smallest, with a
    # wide gap and nothing between.
    assert min(abs(v) for v in passing.values()) == 0.1186
    assert max(abs(v) for v in failing.values()) == 0.0031
    assert min(abs(v) for v in passing.values()) > 38 * max(
        abs(v) for v in failing.values()
    )
    assert "No contrast sits between" in _flat(
        record["the_split_is_the_four_largest_against_the_four_smallest"]
    )

    # Deltas unchanged from the earlier run.
    assert "deltas are unchanged" in _flat(record["provenance"])


def test_the_four_are_the_sharpest_case_for_requiring_both_conditions():
    """Larger margins than any prior instance, failing on fewer seeds."""
    import re

    from cleft import results_ledger

    why = _flat(record_why := phase22.CONTRAST_VERDICTS_OBSERVED[
        "why_this_is_the_clearest_case_for_BOTH_conditions"
    ])
    assert "three to seven times their own thresholds" in why
    assert "1 of 5 and 2 of 5 seeds" in why
    assert "Condition 1 is what refuses them" in why

    # The prior maximum it cites is real, read from the ledger itself.
    margins = []
    for entry in results_ledger.ENTRIES:
        c1, c2 = str(entry.get("condition_1")), str(entry.get("condition_2"))
        if c1.startswith("FALSE") and c2.startswith("TRUE"):
            found = re.search(r"TRUE at ([0-9.]+)x", c2)
            if found:
                margins.append(float(found.group(1)))
    assert len(margins) == 8
    assert max(margins) == 5.1
    assert "5.1x" in why and "p17-c-vs-probe" in why
    # And the prior seed counts really are 0, 2 and 3 of 5.
    seeds = set()
    for entry in results_ledger.ENTRIES:
        c1 = str(entry.get("condition_1"))
        if c1.startswith("FALSE"):
            found = re.search(r"(\d) of 5 seeds", c1)
            if found:
                seeds.add(int(found.group(1)))
    assert seeds == {0, 2, 3}
    assert "0, 2 and 3 of 5" in why


def test_twelve_and_eight_are_reconciled_explicitly():
    """A later reader must not read the two counts as a contradiction."""
    import re

    from cleft import results_ledger

    record = phase22.CONDITION_PHENOMENON_COUNT_RECONCILED
    both = _flat(record["the_two_numbers"])
    assert "TWELVE instances of the phenomenon project-wide" in both
    assert "EIGHT rows in the ledger" in both
    assert "Both are right" in both

    counts = _flat(record["what_entry_38_counts"])
    assert "LEDGER ROWS" in counts
    assert "Entry 38 stands unchanged" in counts

    # Entry 38 really says eight, and the ledger really has eight.
    entry_38 = results_ledger.ENTRIES[37]
    assert "EIGHT rows have condition 1 FALSE and condition 2 TRUE" in (
        entry_38["claim"]
    )
    indices = [
        i for i, e in enumerate(results_ledger.ENTRIES)
        if str(e.get("condition_1")).startswith("FALSE")
        and str(e.get("condition_2")).startswith("TRUE")
    ]
    assert indices == [25, 26, 27, 28, 31, 32, 34, 35]
    assert len(indices) == 8

    # Phase 22 added no rows, which is why the eight is untouched.
    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 22 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p22" or "p22-" in e["id"]
    ]
    assert results_ledger.validate() is None
    assert not [
        e for e in results_ledger.ENTRIES if "p22" in str(e.get("id"))
    ]
    assert "registered NO LEDGER ROWS" in _flat(record["why_phase_22_adds_none"])
    assert "8 ledger rows + 4 Phase 22 contrasts = **12 instances**" in _flat(
        record["the_arithmetic"]
    )
    assert 8 + 4 == 12
    # And the record says how to quote either without conflating them.
    assert "neither may be written as the other" in _flat(
        record["how_to_quote_either"]
    )


def test_the_closing_cites_the_rerun_and_supersedes_only_the_verdicts():
    record = phase22.PHASE_22_CLOSING
    assert "RE-RUN's" in _flat(record["closed"])

    which = _flat(record["which_run_the_verdicts_come_from"])
    assert "SUPERSEDED FOR THE VERDICTS ONLY" in which
    assert "condition_1: None" in which
    assert "DELTAS were correct and are unchanged" in which
    # Superseded is distinguished from void, which the phase uses
    # elsewhere for a run that measured the wrong thing.
    assert "Superseded is not void" in which
    assert "VOID" in _flat(record["what_was_void_and_stays_void"])

    # Criterion 11 now cites the populated conditions.
    assert "both PLAN 4.3 conditions on every row" in _flat(
        record["criterion_walk"]["11_the_expected_null_reported_as_a_null"]
    )
