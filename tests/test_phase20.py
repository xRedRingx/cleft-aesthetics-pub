"""Phase 20 -- the permutation control. Registration only.

Nothing is built: no task, no schema kind, no config, no run. A
negative-space test holds that state until it is ruled the
stratification rule and the compute question.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cleft import ladder, phase13, phase16, phase17, phase20
from cleft.data import labels as label_module

REPO = Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# the reckoning
# --------------------------------------------------------------------------


def test_the_reckoning_quotes_each_record_at_source():
    record = phase20.PHASE_20_RECKONING

    # Phase 13's confound ceiling -- figures and statistics, both live.
    ceiling = _flat(record["phase_13_confound_ceiling"])
    banked = _flat(phase13.P1_CONFOUND_CEILING_BANKED["figures"])
    for figure in ("+0.1183", "+0.0659", "+0.1216", "+0.0897", "+0.1044",
                   "mean +0.1000", "sd 0.0228"):
        assert figure in banked, figure
        assert figure in ceiling, figure
    stats = _flat(
        phase13.CLOSING_ADDENDUM_REGISTERED["p1_confound_ceiling"]["what"]
    )
    for stat in ("aspect ratio", "brightness mean", "contrast (pixel sd)",
                 "content-pixel fraction", "corner-white fraction"):
        assert stat in stats, stat
        assert stat in ceiling, stat

    # The two real-label arms it is positioned against.
    assert phase17.ARM_MEANS["arms"]["p17_arm_a"]["pcc"] == 0.2334
    assert "0.2334" in record["tstr_arm_a"]
    assert "UNRESOLVED" in record["tstr_arm_a"]
    assert phase16.ARM_MEANS["arms"]["p16_identity_baseline"]["pcc"] == 0.2151
    assert "0.2151" in record["identity_readout"]
    assert "labels are REAL" in _flat(record["tstr_arm_a"])

    # Where the verdicts get read.
    where = _flat(record["where_the_verdicts_are_read"])
    assert "cannot resolve PCC differences of 0.04 to 0.10" in where
    assert "0.04 to 0.10" in _flat(ladder.COHORT_CANNOT_RESOLVE["finding"])
    assert "0.1386" in where
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["quotable"][
        "the_one_figure"
    ] == 0.1386
    assert "detection-floor prohibition applies" in where


def test_the_reckoning_states_what_no_other_record_measures():
    record = phase20.PHASE_20_RECKONING
    what = _flat(record["what_this_phase_measures_that_none_do"])
    assert "DEFINITIONALLY NO SIGNAL" in what
    assert "no measured chance level for its own machinery" in what
    assert "constant-predictor floors" in what
    assert "randomised-WEIGHT controls" in what
    assert "neither controls for label-feature association arising by chance" in (
        what
    )
    assert "how much of 0.2520 survives" in what
    # Conceding was available and was declined, with a reason.
    assert "neither question is conceded" in _flat(
        record["concession_considered_and_declined"]
    )


def test_the_target_mismatch_is_traced_in_code_not_inferred():
    """The reckoning's own finding: the confound ceiling and the
    headline are scored against different labels, and no record says so."""
    record = phase20.CONFOUND_CEILING_TARGET_MISMATCH

    # The ceiling's target, verified at its source.
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module._grades_for)
    assert "The median grade per patient" in source
    assert "MEDIAN grade" in record["the_ceiling_target"]
    ceiling_config = (REPO / "configs" / "p13_confound_ceiling.yaml").read_text(
        encoding="utf-8"
    )
    assert "label:" not in ceiling_config, "the config declares no label"

    # The headline's target, verified at its source.
    probe_config = (
        REPO / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml"
    ).read_text(encoding="utf-8")
    assert "label: mean" in probe_config
    assert "PANEL MEAN" in record["the_headline_target"]

    # The comparison that carries the mismatch, and the arithmetic.
    carried = _flat(record["the_comparison_that_carries_it"])
    assert "~40% of the headline" in carried
    assert "0.397" in carried
    assert round(0.1000 / 0.2520, 3) == 0.397
    # The two targets are not interchangeable, from the label record.
    assert label_module.LEARNABILITY_237["mean"] == 0.6022
    assert label_module.LEARNABILITY_237["median"] == 0.5708
    assert "0.6022" in carried and "0.5708" in carried

    # It is NOT a correction to a closed phase, and says why.
    not_a_correction = _flat(record["not_a_correction_to_phase_13"])
    assert "correct for its own quantity" in not_a_correction
    assert "threshold was pre-declared" in not_a_correction
    assert "the phase is closed" in not_a_correction
    assert "unflagged target change" in not_a_correction
    # And Phase 20 does not inherit it.
    consequence = _flat(record["consequence_for_phase_20"])
    assert "S is registered against the PANEL MEAN" in consequence
    assert "SEPARATE REGISTERED ADDITION" in consequence


# --------------------------------------------------------------------------
# the two arms
# --------------------------------------------------------------------------


def test_arm_p_reuses_the_probes_fit_path():
    record = phase20.ARM_P_REGISTERED
    assert record["role"] == "THE INTEGRITY GATE"
    design = _flat(record["design"])
    assert "shuffled across all 237" in design
    assert "769-parameter linear head" in design
    assert "5-fold OOF" in design
    for seed in ("1337", "2024", "7", "99", "12345"):
        assert seed in design

    reuse = _flat(record["fit_path_reused"])
    assert "run.task_train_cv" in reuse
    assert "DETACH THE CONTROL FROM THE THING IT CONTROLS" in reuse
    assert "PROBE_RECONSTRUCTED_THE_PIPELINE" in reuse
    # The probe's recipe really is train_cv on the mean label.
    probe = (REPO / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
        encoding="utf-8"
    )
    assert "kind: train_cv" in probe
    assert "trainable: head" in probe


def test_arm_p_names_its_degenerate_case_with_the_arithmetic():
    degenerate = _flat(
        phase20.ARM_P_REGISTERED["degenerate_case_named_in_advance"]
    )
    assert "E[fixed points] = 1 for EVERY n" in degenerate
    assert "~1 of 237" in degenerate and "0.42%" in degenerate
    assert "never discarded" in degenerate

    # The claim is arithmetic, and it is checked rather than asserted.
    rng = np.random.default_rng(20260831)
    counts = [
        int(np.sum(rng.permutation(237) == np.arange(237)))
        for _ in range(4000)
    ]
    assert float(np.mean(counts)) == pytest.approx(1.0, abs=0.12)
    assert 1 / 237 == pytest.approx(0.0042, abs=1e-4)


def test_arm_s_stratification_is_proposed_with_its_reasoning():
    record = phase20.ARM_S_PROPOSED
    assert "NOT LOCKED" in record["proposed"]
    assert record["k_strata"] == 10

    variable = _flat(record["stratification_variable"])
    assert "FITTED VALUE over the five statistics" in variable
    assert "+0.1601" in variable, "the single-statistic alternative is named"

    why_in_sample = _flat(record["fitted_in_sample_and_why"])
    assert "DESIGN CHOICE, not an estimate" in why_in_sample
    assert "seed-dependent" in why_in_sample

    # Both directions of the k trade-off are given, not just the choice.
    reasoning = _flat(record["k_reasoning"])
    assert "more strata preserve confound structure more tightly" in reasoning
    assert "fewer do the reverse" in reasoning

    # It is a scientific setting, with the movement rule attached.
    setting = _flat(record["it_is_a_scientific_setting"])
    assert "never tuned across runs" in setting
    assert "dated amendment" in setting
    # And it does not inherit the target mismatch.
    assert "PANEL MEAN" in record["target"]


def test_arm_s_degenerate_frequency_is_predicted_before_the_run():
    degenerate = _flat(record_s := phase20.ARM_S_PROPOSED[
        "degenerate_case_with_arithmetic"
    ])
    assert "E[fixed points] = 1 per stratum for ANY stratum size" in degenerate
    assert "at k = 10 that is 10 of 237 (4.2%" in degenerate
    assert "never discarded" in degenerate and "never redrawn" in degenerate

    # The prediction is arithmetic and is checked here.
    rng = np.random.default_rng(20260831)
    k, n = 10, 237
    sizes = [n // k] * k
    for i in range(n - sum(sizes)):
        sizes[i] += 1
    assert min(sizes) == 23 and sum(sizes) == 237
    assert "smallest is 23" in degenerate

    totals = []
    for _ in range(2000):
        totals.append(sum(
            int(np.sum(rng.permutation(s) == np.arange(s))) for s in sizes
        ))
    assert float(np.mean(totals)) == pytest.approx(float(k), abs=0.25)
    assert k / n == pytest.approx(0.042, abs=1e-3)


# --------------------------------------------------------------------------
# ordering, readings, machinery
# --------------------------------------------------------------------------


def test_the_gate_ordering_is_bound_before_either_runs():
    record = phase20.ORDERING_BOUND
    assert "before either arm runs" in record["bound"]
    assert record["rule"].startswith("Arm P is the GATE")
    fails = _flat(record["if_p_fails"])
    assert "S is UNINTERPRETABLE" in fails
    assert "THE PIPELINE RESULT, not the attribution" in fails
    assert "bigger and more urgent" in fails
    assert "S does not run on a failed gate" in _flat(record["the_phase_stops"])


def test_both_readings_are_committed_with_the_failure_branch_named():
    readings = phase20.READINGS_COMMITTED
    assert "before any number exists" in readings["committed"]

    good = _flat(readings["p_near_zero"])
    assert "NO correlation from scrambled labels" in good
    assert "removes a doubt rather than adding a finding" in good

    # The branch that matters most is fully specified in advance.
    bad = _flat(readings["p_meaningfully_above_zero"])
    assert "EVERY BANKED PCC IN THE PROJECT" in bad
    for diagnosis in ("FOLD CONSTRUCTION", "OOF ASSEMBLY", "THE LABEL JOIN",
                      "POOLING"):
        assert diagnosis in bad, diagnosis
    assert "STOPS and REPORTS" in bad

    # S's three cells, with the arithmetic form given in advance.
    ceiling = _flat(readings["s_near_the_confound_ceiling"])
    assert "residual = 0.2520 - S_mean" in ceiling
    assert "0.1386" in ceiling
    assert "NOT demonstrated to be resolvable" in ceiling
    assert "largely confound structure" in _flat(readings["s_near_0_2520"])
    between = _flat(readings["s_between"])
    assert "STATED, not resolved" in between
    assert "no point estimate promoted to a threshold" in between
    assert "UNPREDICTED_PATTERN" in readings["no_reading_is_invented_after"]


def test_arm_p_is_registered_as_not_fitting_the_paired_criterion():
    record = phase20.CONTRAST_MACHINERY
    does_not = _flat(record["arm_p_does_not_fit_the_paired_criterion"])
    assert "NULL CONSTRUCT, not a competing method" in does_not
    assert "category error" in does_not

    instead = _flat(record["what_is_reported_for_p"])
    assert "ONE-SAMPLE BCa interval over patients against ZERO" in instead
    assert "NO LEDGER ROW" in instead

    # S is distinguished, and the position on it is flagged not settled.
    s = _flat(record["arm_s_is_defined_but_the_verdict_is_not_wanted"])
    assert "is DEFINED" in s
    assert "should NOT carry a claimable/unresolved verdict" in s
    assert "FLAGGED FOR A RULING" in s


def test_the_compute_gate_is_designed_not_run():
    record = phase20.COMPUTE_GATE_DESIGNED
    assert "NOT run" in record["designed"]
    assert "CACHED EMBEDDINGS" in record["why_it_is_probably_trivial"]
    assert "the gate measures rather than assumes" in _flat(
        record["the_expectation_is_not_the_measurement"]
    )
    assert "per-seed wall-clock" in _flat(record["the_job"])
    assert "no new hash" in _flat(record["the_job"])


def test_the_exit_criteria_are_a_draft_and_say_what_is_unsettled():
    draft = phase20.EXIT_CRITERIA_DRAFT
    assert draft["status"].startswith("DRAFT")
    assert len(draft["criteria"]) == 8
    joined = " ".join(draft["criteria"])
    assert "one-sample BCa intervals against zero" in joined
    assert "DEGENERATE-STRATUM FREQUENCY" in joined
    assert "the relevant floor is ZERO" in joined
    assert "NO LEDGER ROW" in joined
    assert "target-difference statement" in joined
    unsettled = _flat(draft["what_is_not_yet_settled"])
    assert "stratification rule" in unsettled
    assert "all the maintainer's" in unsettled


# --------------------------------------------------------------------------
# the negative space -- nothing is built
# --------------------------------------------------------------------------


def test_nothing_is_built_for_phase_20():
    """**[UPDATED 2026-08-31 -- THIS PIN FIRED, AS DESIGNED.]**

    It was written to hold the registration-only state "until the maintainer
    rules". He ruled, the arms are built, and the pin is updated with
    the date rather than deleted -- what it now holds is the ONE thing
    that must stay true across the build: **there is exactly one fit
    path, and no second permutation implementation.**
    """
    import inspect

    from cleft import run as run_module
    from cleft.config.schema import TASK_SPECS
    from cleft.run import TASKS
    from cleft.train import phase3 as phase3_module

    # Built now -- the arms exist and are reachable.
    assert "p20_permutation" in TASK_SPECS and "p20_permutation" in TASKS
    assert (REPO / "configs" / "p20_permutation_plain.yaml").is_file()
    assert (REPO / "configs" / "p20_permutation_stratified.yaml").is_file()
    assert (REPO / "configs" / "p20_confound_ceiling_mean.yaml").is_file()
    assert (REPO / "scripts" / "generate_phase20_configs.py").is_file()
    # [2026-08-31, second cycle] The stratification diagnostic joined.
    assert "p20_stratification_diagnostic" in TASK_SPECS
    assert "p20_stratification_diagnostic" in TASKS
    assert (REPO / "configs" / "p20_stratification_diagnostic.yaml").is_file()

    # ONE fit path. The permutation task delegates to task_train_cv; it
    # does not build a loop of its own.
    body = inspect.getsource(run_module.task_p20_permutation)
    assert "task_train_cv(ctx)" in body
    for forbidden in ("RidgeBackbone", "for fold in", "metrics.pcc"):
        assert forbidden not in body, (
            f"{forbidden!r} in task_p20_permutation: the arm must reuse "
            "the probe's fit path, never reimplement it"
        )

    # ONE permutation implementation. [2026-08-31, second cycle: this
    # pin fired when the diagnostic became a second legitimate CALLER.
    # "One call site" was always a proxy for "one implementation" -- it
    # is now stated as the thing it stood for, which has more teeth: run
    # may CALL the drawer anywhere and may DEFINE one nowhere.]
    source = inspect.getsource(run_module)
    assert "def permute_within_strata" not in source
    assert "def quantile_strata" not in source
    for forbidden in ("np.random.default_rng(seed).permutation(len(",
                      "rng.shuffle(labels"):
        assert forbidden not in source, forbidden
    # Every caller reaches it by the qualified name, so a local rebinding
    # cannot slip a different drawer in under the same call.
    assert source.count("permute_within_strata") == source.count(
        "phase20.permute_within_strata"
    )
    # phase3 may NAME the drawer (its docstring points at it); it may
    # not contain one.
    assert "def permute_within_strata" not in inspect.getsource(phase3_module)
    assert "def quantile_strata" not in inspect.getsource(phase3_module)

    # phase3 APPLIES a permutation it is handed; it never draws one.
    fit = inspect.getsource(phase3_module.run)
    assert "label_permutation" in fit
    for forbidden in ("default_rng", "permutation(len(labels))", "shuffle("):
        assert forbidden not in fit, forbidden

    # Still no ledger row -- LEDGER_RULED, for all three arms.
    from cleft import results_ledger

    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p20" or "permut" in e["id"]
    ]


def test_the_summary_carries_every_record():
    # [UPDATED 2026-08-31] Five records added by the rulings, the
    # addition, and the lock. This pin fires on every fill by design.
    # [UPDATED 2026-08-31, again] Six more once the arms ran: what was
    # observed, S's unfired cells, the lock's limitation, and the
    # diagnostic with its readings and its binding.
    # [UPDATED 2026-08-31, close-out] Six more: the diagnostic's result,
    # the residual prohibition, the stability finding, the description
    # error's provenance, the fifth amendment, and the closing.
    assert sorted(phase20.summary()) == [
        "arm_c", "arm_p", "arm_s", "arms_observed", "closing",
        "compute_gate", "contrast_machinery",
        "cross_target_error_provenance", "diagnostic",
        "diagnostic_binding", "diagnostic_observed", "diagnostic_readings",
        "exit_criteria", "exit_criteria_draft", "ledger_ruled",
        "lock_limitation", "ordering", "readings", "reckoning",
        "residual_prohibition", "s_description_error_provenance",
        "s_pattern_unpredicted", "seed_stability", "sequence_extended_5",
        "stratification_ruled", "target_mismatch",
    ]


# --------------------------------------------------------------------------
# the close-out: fired cells BY IDENTITY, unfired cells preserved
# --------------------------------------------------------------------------


def _resolve(path: str):
    """Resolve a 'RECORD['key']['key']' pointer against phase20."""
    import re

    head, *keys = re.findall(r"^(\w+)|\['([^']+)'\]", path)
    node = getattr(phase20, head[0])
    for _, key in keys:
        node = node[key]
    return node


def test_every_fired_cell_resolves_to_its_committed_reading_by_identity():
    """**The fired cells are named by POINTER, never retyped.** A cell
    quoted into the outturn record could drift from the cell that was
    committed, and the drift would read as a registered reading."""
    fired = {
        "arm_p": phase20.READINGS_COMMITTED["p_near_zero"],
        "arm_c": phase20.ARM_C_REGISTERED["readings"]["near_0_1000"],
        "diagnostic": phase20.DIAGNOSTIC_READINGS_COMMITTED["between"],
    }
    for arm, expected in fired.items():
        pointer = phase20.ARMS_OBSERVED[arm]["cell_fired"]
        assert _resolve(pointer) is expected, (
            f"{arm}: cell_fired points at {pointer!r}, which is not the "
            "committed reading object"
        )
    # Arm S fired nothing, and says so rather than being given a cell.
    assert phase20.ARMS_OBSERVED["arm_s"]["cell_fired"] is None
    # The closing's per-arm verdicts name the same cells.
    closing = phase20.PHASE_20_CLOSING
    assert "p_near_zero" in closing["criterion_5_one_committed_reading_per_arm"]
    assert "near_0_1000" in closing["criterion_5_one_committed_reading_per_arm"]


def test_the_unfired_cells_are_preserved_unamended():
    """Arm S's four cells must survive the close-out untouched."""
    for cell in ("s_near_the_confound_ceiling", "s_near_0_2520", "s_between"):
        text = phase20.READINGS_COMMITTED[cell]
        assert cell in phase20.READINGS_COMMITTED
        # None of the observed figures leaked into a committed cell.
        for figure in ("0.0414", "0.0802", "0.1061", "0.0639", "60.2"):
            assert figure not in text, f"{cell} was edited to fit the outturn"
    # The residual arithmetic in the ceiling cell is still the FORM it
    # was registered as, not the filled-in number.
    assert "residual = 0.2520 - S_mean" in _flat(
        phase20.READINGS_COMMITTED["s_near_the_confound_ceiling"]
    )
    assert "0.2106" not in phase20.READINGS_COMMITTED[
        "s_near_the_confound_ceiling"
    ]


def test_the_observed_figures_reconcile_from_their_own_per_seed_values():
    """Re-derived here, not trusted: the supplied means and sds must
    follow from the supplied per-seed values."""
    observed = phase20.ARMS_OBSERVED
    for arm in ("arm_p", "arm_c"):
        per_seed = np.array(list(observed[arm]["pcc_by_seed"].values()))
        assert len(per_seed) == 5
        assert float(per_seed.mean()) == pytest.approx(
            observed[arm]["pcc_mean"], abs=5e-5
        )
        assert float(per_seed.std(ddof=1)) == pytest.approx(
            observed[arm]["pcc_sd"], abs=5e-5
        )

    # The diagnostic reconciles by TWO independent routes.
    diagnostic = observed["diagnostic"]
    fractions = np.array(list(diagnostic["retained_fraction_by_seed"].values()))
    assert float(fractions.mean()) == pytest.approx(
        diagnostic["retained_fraction_mean"], abs=5e-4
    )
    assert float(fractions.std(ddof=1)) == pytest.approx(
        diagnostic["retained_fraction_sd"], abs=5e-4
    )
    implied = fractions * diagnostic["ceiling_pcc"]
    assert float(implied.mean()) == pytest.approx(
        diagnostic["pcc_mean"], abs=5e-5
    )
    assert float(implied.std(ddof=1)) == pytest.approx(
        diagnostic["pcc_sd"], abs=5e-5
    )
    # And the mean fraction really is the mean PCC over the ceiling.
    assert diagnostic["pcc_mean"] / diagnostic["ceiling_pcc"] == pytest.approx(
        diagnostic["retained_fraction_mean"], abs=1e-3
    )

    # Arm C's difference from Phase 13, from the structured home.
    assert observed["arm_c"]["pcc_mean"] - phase13.P1_CEILING_FIGURES[
        "pcc_mean"
    ] == pytest.approx(observed["arm_c"]["difference"], abs=5e-5)


def test_each_run_sha_is_a_commit_carrying_that_runs_config():
    """The run-dir contract <config-stem>__<sha8>__<job-id>, checked
    against git rather than assumed. A run cannot have used a config
    that did not exist at its own sha."""
    import subprocess

    expected = {
        "arm_p": "p20_permutation_plain",
        "arm_s": "p20_permutation_stratified",
        "arm_c": "p20_confound_ceiling_mean",
        "diagnostic": "p20_stratification_diagnostic",
    }
    for arm, stem in expected.items():
        run = phase20.ARMS_OBSERVED[arm]["run"]
        name, sha = run.rsplit("__", 1)
        assert name == stem, run
        assert len(sha) == 8 and set(sha) <= set("0123456789abcdef"), run

        # The sha is a real commit here...
        resolved = subprocess.run(
            ["git", "rev-parse", "--short=8", f"{sha}^{{commit}}"],
            cwd=REPO, capture_output=True, text=True,
        )
        if resolved.returncode != 0:  # pragma: no cover - shallow clone
            pytest.skip(f"{sha} not resolvable in this checkout")
        assert resolved.stdout.strip() == sha

        # ...and that commit carries the config the run names.
        listed = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}:configs/{stem}.yaml"],
            cwd=REPO, capture_output=True, text=True,
        )
        assert listed.returncode == 0, (
            f"{run}: configs/{stem}.yaml does not exist at {sha}, so the "
            "run could not have used it"
        )

    # The three dc4605bf arms and the 77ea2247 diagnostic really are
    # two distinct shas -- the diagnostic came later, as its config did.
    shas = {
        arm: phase20.ARMS_OBSERVED[arm]["run"].rsplit("__", 1)[1]
        for arm in expected
    }
    assert len({shas["arm_p"], shas["arm_s"], shas["arm_c"]}) == 1
    assert shas["diagnostic"] != shas["arm_p"]
    assert "verified at source" in phase20.ARMS_OBSERVED[
        "rederivation_checks"
    ]["the_run_shas_are_this_repos_own_commits"]


def test_the_strata_the_run_reported_are_the_ones_the_code_produces():
    """The strongest check available without the artifacts: the run's
    reported partition must be what quantile_strata builds."""
    reported = phase20.ARMS_OBSERVED["arm_s"]["strata_sizes"]
    assert sum(reported) == 237 and len(reported) == 10

    strata = phase20.quantile_strata(np.arange(237, dtype=float), 10)
    assert tuple(np.bincount(strata).tolist()) == reported, (
        "the run partitioned the cohort differently from the shipped code"
    )
    # Zero singleton strata, so nothing was immovable by construction --
    # which is what the closing reports, distinctly from fixed points.
    census = phase20.degenerate_strata(strata)
    assert census["n_immovable"] == phase20.ARMS_OBSERVED["arm_s"][
        "singleton_strata"
    ] == 0


def test_both_fixed_point_outturns_agree_with_their_pre_registration():
    """The pre-registered expectation is E = k. Checked against the
    shipped drawer's own sampling distribution, not asserted."""
    observed = phase20.ARMS_OBSERVED
    for arm, k in (("arm_p", 1), ("arm_s", 10)):
        counts = list(observed[arm]["fixed_points_by_seed"].values())
        assert len(counts) == 5
        assert observed[arm]["fixed_points_expected"] == k

        strata = phase20.quantile_strata(np.arange(237, dtype=float), k)
        draws = np.array([
            phase20.fixed_points(phase20.permute_within_strata(strata, seed=s))
            for s in range(3000)
        ])
        means5 = draws[:3000].reshape(-1, 5).mean(axis=1)
        z = (np.mean(counts) - means5.mean()) / means5.std(ddof=1)
        assert abs(z) < 2.0, (
            f"{arm}: observed mean {np.mean(counts)} is z={z:+.2f} from the "
            f"drawer's own distribution -- the pre-registration would not hold"
        )
    # Arm S's mean is stated and is what its own counts give.
    assert np.mean(
        list(observed["arm_s"]["fixed_points_by_seed"].values())
    ) == pytest.approx(observed["arm_s"]["fixed_points_mean"], abs=1e-9)


def test_the_diagnostic_records_its_dispersion_as_a_finding():
    record = phase20.DIAGNOSTIC_OBSERVED
    assert record["cell_fired"] == "DIAGNOSTIC_READINGS_COMMITTED['between']"
    assert _resolve(record["cell_fired"]) is (
        phase20.DIAGNOSTIC_READINGS_COMMITTED["between"]
    )

    # The identity check is recorded as PASSED by a person's diff.
    check = _flat(record["permutation_identity_check"])
    assert "PASSED" in check
    assert "the diff, empty" in check
    assert "not inferred from the code being deterministic" in check

    dispersion = _flat(record["the_per_seed_dispersion_is_its_own_finding"])
    assert "larger than the headline" in dispersion
    assert "MORE CONFOUND SIGNAL THAN THE UNPERMUTED CEILING" in dispersion
    meaning = _flat(record["what_the_dispersion_means"])
    assert "NOT A STABLE PROPERTY OF THE DESIGN" in meaning
    assert "WHY the bound is wide" in meaning

    # The 122% seed really is above the ceiling, and the spread is a
    # factor of six -- both re-derived rather than asserted.
    fractions = phase20.ARMS_OBSERVED["diagnostic"]["retained_fraction_by_seed"]
    assert max(fractions.values()) > 1.0
    assert max(fractions.values()) / min(fractions.values()) > 6.0


def test_the_residual_prohibition_is_a_tested_literal():
    """Sibling to ladder.DETECTION_FLOOR_PROHIBITION and
    phase10_annex.ANNEX_PROHIBITION -- a string, pinned."""
    from cleft import phase10_annex

    prohibition = phase20.RESIDUAL_PROHIBITION
    assert isinstance(prohibition, str)
    assert isinstance(ladder.DETECTION_FLOOR_PROHIBITION, str)
    assert isinstance(phase10_annex.ANNEX_PROHIBITION, str)

    assert "0.2520 - 0.0414 = 0.2106 IS NOT CLEFT-SPECIFIC" in prohibition
    assert "NEVER QUOTED AS SUCH" in prohibition
    assert "under NO branch" in prohibition
    assert "BOUNDED, NOT RESOLVED" in prohibition
    assert "SMALLEST_RESOLVABLE_DIFFERENCE governs" in prohibition
    assert "0.1386" in prohibition and "0.04 to 0.10" in prohibition
    assert "No phase is gated on any of this." in prohibition
    # The arithmetic it forbids is arithmetically correct -- the point
    # is that a true subtraction is not a licensed sentence.
    observed = phase20.ARMS_OBSERVED["arm_s"]["pcc_mean"]
    assert round(0.2520 - observed, 4) == 0.2106
    # And every figure it quotes is live at its own source.
    assert phase20.ARMS_OBSERVED["arm_s"]["pcc_sd"] == 0.0802
    assert phase20.ARMS_OBSERVED["diagnostic"]["retained_fraction_mean"] == 0.602
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"] == 0.1386


def test_the_seed_stability_finding_is_monotone_and_tagged():
    record = phase20.SEED_STABILITY_EVIDENCE
    sds = [record["sds_by_arm"][k]["sd"] for k in (
        "arm_p_scrambled", "arm_s_stratified", "arm_c_confounds", "probe_real"
    )]
    assert sds == [0.1109, 0.0802, 0.0222, 0.0148]
    assert all(a > b for a, b in zip(sds, sds[1:])), "strictly monotone"
    assert sds[0] / sds[-1] == pytest.approx(7.49, abs=0.01)
    assert "7.49x" in record["the_ratio"]

    # Each sd is cited to its run, and the probe's to its banked home.
    for key, entry in record["sds_by_arm"].items():
        assert entry["run"], key
    assert ladder.TRADE_OFF_PAIR["result"]["vit_sd"] == 0.0148
    assert record["sds_by_arm"]["probe_real"]["sd"] == (
        ladder.TRADE_OFF_PAIR["result"]["vit_sd"]
    )
    # The three permutation-family sds match ARMS_OBSERVED exactly.
    assert record["sds_by_arm"]["arm_p_scrambled"]["sd"] == (
        phase20.ARMS_OBSERVED["arm_p"]["pcc_sd"]
    )
    assert record["sds_by_arm"]["arm_s_stratified"]["sd"] == (
        phase20.ARMS_OBSERVED["arm_s"]["pcc_sd"]
    )
    assert record["sds_by_arm"]["arm_c_confounds"]["sd"] == (
        phase20.ARMS_OBSERVED["arm_c"]["pcc_sd"]
    )

    # The tags: the numbers MEASURED, the mechanism REASONED.
    assert record["the_measurement"].startswith("[MEASURED]")
    assert record["the_ratio"].startswith("[MEASURED]")
    assert record["the_mechanism"].startswith(
        "[REASONED -- an argument, not a measurement]"
    )
    assert "is INFERENCE" in record["the_mechanism"]

    # It is a SECOND line, and it licenses nothing about size.
    assert "STABILITY rather than MAGNITUDE" in record["why_it_is_a_second_line"]
    assert "does NOT depend on comparing 0.2520 against any ceiling" in _flat(
        record["why_it_is_a_second_line"]
    )
    assert "nothing about SIZE" in record["what_it_does_not_license"]
    assert "RESIDUAL_PROHIBITION is untouched" in _flat(
        record["what_it_does_not_license"]
    )
    # The confound to the four-way comparison is named, not hidden.
    caveat = _flat(record["caveat_on_the_comparison"])
    assert "changes the MODEL as well as the labels" in caveat
    assert "P -> S -> probe" in caveat
    assert 0.1109 > 0.0802 > 0.0148, "the within-design chain is monotone too"


def test_the_description_error_is_recorded_with_its_provenance():
    """Named on the same footing as the other two recorded errors."""
    record = phase20.S_DESCRIPTION_ERROR_PROVENANCE
    assert "CONVERSATION -- the record's, this session" in record["origin"]
    assert "BOTH HALVES FALSE" in record["the_error"]
    assert "0.0647" in record["the_error"] and "0.0868" in record["the_error"]
    assert "39.0%" in record["the_error"] and "57.3%" in record["the_error"]
    assert "NOT DISTINGUISHABLE FROM ZERO is true" in record["what_survives"]
    assert "THE TEST FAILED" in record["how_it_was_caught"]
    assert "nothing measured, and no reading" in record["what_it_touched"]

    # The two siblings it is filed beside really do name the record too.
    assert "the record's" in phase20.CROSS_TARGET_ERROR_PROVENANCE["origin"]
    assert "the record" in _flat(ladder.THE_ERROR_PROVENANCE["origin"])
    # And the closing counts both of this phase's corrections.
    corrections = _flat(
        phase20.PHASE_20_CLOSING["corrections_recorded_in_this_phase"]
    )
    assert "two, both the record's and both named as such" in corrections
    assert "CROSS_TARGET_ERROR_PROVENANCE" in corrections
    assert "S_DESCRIPTION_ERROR_PROVENANCE" in corrections


# --------------------------------------------------------------------------
# the fifth amendment, and the closing
# --------------------------------------------------------------------------


def test_the_amendment_chain_was_silent_on_20_and_the_fifth_closes_it():
    from cleft import phase11, phase12, phase15

    # Verified at source: the fourth amendment stops at 19.
    fourth = phase15.PHASE_SEQUENCE_RENUMBERED_4
    assert fourth["becomes"]["19"] == "write-up"
    assert "20" not in fourth["becomes"]
    for amendment in (
        phase11.PHASE_SEQUENCE_RENUMBERED,
        phase12.PHASE_SEQUENCE_RENUMBERED_2,
        phase15.PHASE_SEQUENCE_RENUMBERED_3,
        fourth,
    ):
        assert "20" not in amendment.get("becomes", {})

    fifth = phase20.PHASE_SEQUENCE_EXTENDED_5
    assert fifth["was"]["19"] == "write-up"
    assert "NOTHING" in fifth["was"]["20"]
    assert fifth["becomes"]["19"].startswith("write-up (UNCHANGED")
    assert "PERMUTATION CONTROL" in fifth["becomes"]["20"]

    # The full backward chain is carried, all four.
    for key, target in (
        ("first_amendment", "phase11.PHASE_SEQUENCE_RENUMBERED"),
        ("second_amendment", "phase12.PHASE_SEQUENCE_RENUMBERED_2"),
        ("third_amendment", "phase15.PHASE_SEQUENCE_RENUMBERED_3"),
        ("fourth_amendment", "phase15.PHASE_SEQUENCE_RENUMBERED_4"),
    ):
        assert fifth[key].startswith(target), key

    # Whose say-so, answered rather than left implicit.
    whose = _flat(fifth["whose_say_so_the_number_was"])
    assert "registration message" in whose and "opened" in whose
    assert "The registration message" in whose
    assert "The number was authorised; the record was not written" in whose


def test_the_fifth_amendment_is_named_for_what_it_does():
    fifth = phase20.PHASE_SEQUENCE_EXTENDED_5
    # EXTENDED, not RENUMBERED -- and the reason is R2's shape.
    assert not hasattr(phase20, "PHASE_SEQUENCE_RENUMBERED_5")
    why = _flat(fifth["why_the_name_differs"])
    assert "moves nothing" in why and "R2's shape" in why
    assert fifth["status_changes"]["write_up"] == "Phase 19 -> Phase 19 (unchanged)"

    # The renumber was considered and rejected, with the reason.
    rule = _flat(fifth["the_write_up_is_last_by_rule_not_by_number"])
    assert "RENUMBER WAS CONSIDERED AND REJECTED" in rule
    assert "unboundedly" in rule
    assert "THE WRITE-UP RUNS LAST REGARDLESS OF ITS NUMBER" in rule
    assert "a reader must not infer execution order from 19 vs 20" in rule

    # And the four earlier amendments carry the dated pointer.
    from cleft import phase11, phase12, phase15

    for amendment in (
        phase11.PHASE_SEQUENCE_RENUMBERED,
        phase12.PHASE_SEQUENCE_RENUMBERED_2,
        phase15.PHASE_SEQUENCE_RENUMBERED_3,
        phase15.PHASE_SEQUENCE_RENUMBERED_4,
    ):
        assert amendment["fifth_amendment"] == (
            "phase20.PHASE_SEQUENCE_EXTENDED_5"
        )


def test_the_closing_walks_all_nine_locked_criteria():
    closing = phase20.PHASE_20_CLOSING
    criteria_keys = sorted(k for k in closing if k.startswith("criterion_"))
    assert len(criteria_keys) == 9, criteria_keys
    assert len(phase20.EXIT_CRITERIA["criteria"]) == 9
    for key in criteria_keys:
        assert closing[key].startswith("MET"), key

    # Each run is named, and every run named is one of the four.
    runs = {
        phase20.ARMS_OBSERVED[arm]["run"]
        for arm in ("arm_p", "arm_s", "arm_c", "diagnostic")
    }
    assert len(runs) == 4
    for run in runs:
        assert run in closing["closed"], run
    assert "finalized single-attempt" in closing["closed"]

    # The limitation is CARRIED, not folded in.
    limitation = _flat(closing["the_limitation_carried_as_one"])
    assert "AS A LIMITATION OF THE LOCK, not folded into it" in limitation
    assert "keeps its nine byte-unchanged" in limitation
    assert "not a tenth criterion" in limitation
    assert "why the limitation mattered" in limitation

    # The ledger did not move.
    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for Phase 20 banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p20" or "p20-" in e["id"]
    ]
    assert results_ledger.validate() is None
    assert not [
        e for e in results_ledger.ENTRIES
        if e["phase"] == "p20" or "permut" in e["id"]
    ]
    assert "stands at 38 entries" in closing["criterion_6_no_ledger_row"]


def test_the_headline_is_the_gate_not_the_partition():
    closing = phase20.PHASE_20_CLOSING
    headline = _flat(closing["the_headline_for_the_write_up"])
    assert "THE GATE, NOT THE PARTITION" in headline
    assert "INVENTS NOTHING" in headline
    assert "STABLE WHERE A NULL IS NOT" in headline
    assert "BOUNDED, NOT RESOLVED" in headline
    assert "PARTIALLY AND VARIABLY" in headline
    # The three figures the headline rests on, each live at its source.
    assert "-0.0454" in headline
    assert phase20.ARMS_OBSERVED["arm_p"]["pcc_mean"] == -0.0454
    assert "7.5x" in headline
    assert "60.2% retained" in headline

    assert "AN ATTRIBUTION" in closing["what_the_phase_does_not_deliver"]
    assert "RESIDUAL_PROHIBITION" in closing["what_the_phase_does_not_deliver"]
    grew = _flat(closing["the_phase_closed_without_growing"])
    assert "nine criteria walked" in grew
    assert "one limitation carried as a limitation" in grew
    assert "no ledger row" in grew
    assert "nothing added to the lock" in grew


# --------------------------------------------------------------------------
# what the arms produced, and the cell that did not fire
# --------------------------------------------------------------------------


def test_the_observed_figures_carry_their_provenance():
    """These reached the record through the maintainer in conversation, not by
    this session reading the artifacts. That has to be visible."""
    observed = phase20.ARMS_OBSERVED
    provenance = _flat(observed["provenance"])
    assert "SUPPLIED IN CONVERSATION" in provenance
    assert "not read from the run artifacts by this session" in provenance
    assert "metrics.json files remain the source of record" in provenance
    assert "record_audit.compare_to_artifact" in provenance
    # No keeper run directory is reachable, which is WHY nothing could
    # be read at source -- stated, not left as an unexplained absence.
    assert "no keeper run directory is reachable here" in provenance
    assert not sorted((REPO / "runs" / "keeper").glob("*")) if (
        REPO / "runs" / "keeper"
    ).is_dir() else True
    # And what could NOT be checked is enumerated beside what could.
    checks = phase20.ARMS_OBSERVED["rederivation_checks"]
    assert "NO KEEPER RUN DIRECTORY IS REACHABLE" in checks[
        "what_could_not_be_checked"
    ]
    assert "Arm S's per-seed PCCs (not supplied)" in checks[
        "what_could_not_be_checked"
    ]

    from cleft import record_audit

    assert hasattr(record_audit, "compare_to_artifact"), (
        "the standing check the provenance note points at must exist"
    )


def test_arm_p_passed_the_gate_and_arm_c_did_not_move_the_anchor():
    observed = phase20.ARMS_OBSERVED
    assert observed["arm_p"]["pcc_mean"] == -0.0454
    assert observed["arm_p"]["cell_fired"] == (
        "READINGS_COMMITTED['p_near_zero']"
    )
    assert "THE GATE PASSES" in observed["arm_p"]["consequence"]
    # The cell that fired says what the consequence says it says.
    assert "licenses nothing new" in _flat(
        phase20.READINGS_COMMITTED["p_near_zero"]
    )

    arm_c = observed["arm_c"]
    # The arithmetic of the fired cell, re-derived rather than asserted.
    assert arm_c["difference"] == pytest.approx(
        arm_c["pcc_mean"] - arm_c["against_phase_13_median_target"], abs=5e-5
    )
    assert abs(arm_c["difference"]) < arm_c["declared_threshold"]
    assert arm_c["cell_fired"] == (
        "ARM_C_REGISTERED['readings']['near_0_1000']"
    )
    # The threshold really is the one the shipped config declares.
    import yaml

    task = yaml.safe_load(
        (REPO / "configs" / "p20_confound_ceiling_mean.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    assert task["materially_different"] == arm_c["declared_threshold"]
    # And Phase 13's figure is the one banked, from its structured home.
    assert (
        phase13.P1_CEILING_FIGURES["pcc_mean"]
        == arm_c["against_phase_13_median_target"]
    )
    # The anchor did not move, so the lock's contingency does not trigger.
    assert "does NOT trigger" in arm_c["consequence"]


def test_arm_s_fired_no_cell_and_the_cells_are_not_amended():
    record = phase20.S_PATTERN_UNPREDICTED
    assert phase20.ARMS_OBSERVED["arm_s"]["cell_fired"] is None

    assert "PRESERVED AND UNFIRED" in _flat(record["no_cell_fired"])
    assert "a reading written after the number is not a reading" in _flat(
        record["what_this_is_and_is_not"]
    )
    assert "an OBSERVATION, not a registered reading" in _flat(
        record["what_this_is_and_is_not"]
    )
    # Phase 17's precedent is real and says the same thing.
    from cleft import phase17

    assert "a reading written after the number is not a reading" in _flat(
        phase17.UNPREDICTED_PATTERN["what_this_is_and_is_not"]
    )

    # The four S cells still exist, unamended -- no cell was stretched.
    for cell in ("s_near_the_confound_ceiling", "s_near_0_2520", "s_between"):
        assert cell in phase20.READINGS_COMMITTED, cell
        assert "0.0414" not in phase20.READINGS_COMMITTED[cell], (
            f"{cell} was edited to accommodate the observed number"
        )


def test_the_observed_pattern_is_arithmetically_what_it_says():
    """**[2026-08-31] This test was written to the instruction's phrasing
    and FAILED. The record moved to the arithmetic, not the reverse.**

    The registration described S as "near the null, not near the confound
    ceiling" with an interval that "spans both". Measured against the
    figures supplied in the same sentence, both claims invert.
    """
    observed = phase20.ARMS_OBSERVED
    s, p, c = (observed["arm_s"], observed["arm_p"], observed["arm_c"])
    low, high = s["ci_95"]

    # What the interval does and does not contain.
    assert low < 0 < high, "the interval contains zero"
    assert low < c["pcc_mean"] < high, "and the ceiling"
    assert not low < p["pcc_mean"] < high, "but NOT the null's point estimate"
    assert round(low - p["pcc_mean"], 4) == 0.0107
    assert round(high - low, 4) == 0.1957

    # S is nearer the CEILING, which is the reverse of the phrasing.
    to_null = abs(s["pcc_mean"] - p["pcc_mean"])
    to_ceiling = abs(s["pcc_mean"] - c["pcc_mean"])
    assert round(to_null, 4) == 0.0868
    assert round(to_ceiling, 4) == 0.0647
    assert to_ceiling < to_null, "S is nearer the ceiling than the null"
    assert round(s["pcc_mean"] / c["pcc_mean"] * 100, 1) == 39.0
    assert round(
        (s["pcc_mean"] - p["pcc_mean"]) / (c["pcc_mean"] - p["pcc_mean"]) * 100,
        1,
    ) == 57.3

    # And the record says exactly that.
    pattern = _flat(phase20.S_PATTERN_UNPREDICTED["the_observed_pattern"])
    assert "CLOSER TO THE CEILING" in pattern
    assert "39.0% of the ceiling" in pattern and "57.3% of the way" in pattern
    assert "does NOT contain P's point estimate" in pattern
    assert "NOT DISTINGUISHABLE FROM EITHER" in pattern

    # The divergence is named, not smoothed over.
    divergence = _flat(
        phase20.S_PATTERN_UNPREDICTED["divergence_from_the_phrasing_2026_08_31"]
    )
    assert "BOTH claims go the other way" in divergence
    assert "Recorded as measured" in divergence
    # And the sense in which the cells' own wording survives is kept, so
    # the correction does not overshoot into undermining them.
    assert "NOT DISTINGUISHABLE FROM ZERO" in divergence
    assert "the cells are not weakened" in divergence
    assert "THE DIAGNOSTIC IS UNAFFECTED" in divergence

    # What the observation could not settle was left open, then RESOLVED
    # at the close-out when Arm P's spread arrived -- and the resolution
    # is stated with its own limit.
    cannot = _flat(phase20.S_PATTERN_UNPREDICTED["what_this_record_cannot_settle"])
    assert "needs Arm P's own spread" in cannot
    assert "RESOLVED the same day at the close-out" in cannot

    resolved = _flat(
        phase20.S_PATTERN_UNPREDICTED["s_and_p_are_not_distinguishable_2026_08_31"]
    )
    p_low, p_high = p["ci_95"]
    overlap_low, overlap_high = max(low, p_low), min(high, p_high)
    assert overlap_low < overlap_high, "the intervals really do overlap"
    assert round(overlap_high - overlap_low, 4) == 0.1064
    assert overlap_high - overlap_low > 0.5 * (high - low)
    assert "OVERLAP on [-0.0347, +0.0717]" in resolved
    assert "NOT distinguishable" in resolved
    # And the weaker claim is not upgraded to the stronger one.
    assert "not a paired test" in resolved
    assert "is not quoted as the stronger one" in resolved


def test_the_lock_is_not_reopened_by_the_gap_it_left():
    """The lock's own clause says a missing criterion is a LIMITATION.
    This is the test that it was handled that way."""
    limitation = phase20.LOCK_LIMITATION_STRATIFICATION_UNVERIFIED
    gap = _flat(limitation["the_gap"])
    assert "ONLY BETWEEN BINS" in gap
    assert "BOTH-DESTROYED rather than CLEFT-DESTROYED" in gap
    assert "OPPOSITE CONCLUSIONS FROM THE SAME NUMBER" in _flat(
        limitation["why_it_matters"]
    )
    not_reopened = _flat(limitation["the_lock_is_not_reopened"])
    assert "keeps its nine, byte-unchanged" in not_reopened
    assert "not a tenth criterion" in not_reopened
    assert "A LOCK IS NOT MADE VALID BY THE CALENDAR" in not_reopened

    # And it really was not reopened: still nine, and no criterion
    # mentions the diagnostic.
    assert len(phase20.EXIT_CRITERIA["criteria"]) == 9
    joined = " ".join(phase20.EXIT_CRITERIA["criteria"]).lower()
    assert "diagnostic" not in joined
    assert "within bin" not in joined


# --------------------------------------------------------------------------
# the diagnostic: registered, with its readings committed before it runs
# --------------------------------------------------------------------------


def test_the_diagnostic_is_registered_with_its_recipe_tied_to_arm_c():
    record = phase20.STRATIFICATION_DIAGNOSTIC
    assert "readings committed BEFORE it runs" in record["registered"]
    assert "CLEFT-DESTROYED" in record["the_question"]
    assert "BOTH-DESTROYED" in record["the_question"]

    measurement = _flat(record["the_measurement"])
    assert "S'S OWN PERMUTED LABELS" in measurement
    assert "SAME RECIPE AS ARM C" in measurement
    assert "Same strata, same permutations, same five seeds" in measurement
    assert "byte-identically the ones Arm S trained on" in measurement

    assert "no new artifact, no new hash" in _flat(
        record["cheap_by_construction"]
    )
    # The floor is 0 by construction and the ceiling is Arm C's OBSERVED
    # figure -- both known before the run, which is what makes a
    # fraction meaningful.
    anchors = _flat(record["floor_and_ceiling_known_in_advance"])
    assert "floor is 0 BY CONSTRUCTION" in anchors
    assert str(phase20.ARMS_OBSERVED["arm_c"]["pcc_mean"]) in anchors
    assert "FRACTION OF THE CEILING RETAINED" in anchors

    # S cannot answer it -- stated, because that is why this exists.
    assert "nothing in S distinguishes the two" in _flat(
        record["why_s_cannot_answer_it"]
    )


def test_the_diagnostics_three_readings_are_committed_before_the_number():
    readings = phase20.DIAGNOSTIC_READINGS_COMMITTED
    assert sorted(k for k in readings if k.startswith(("near", "between"))) == [
        "between", "near_the_ceiling", "near_zero",
    ]

    survived = _flat(readings["near_the_ceiling"])
    assert "confound structure SURVIVED" in survived
    assert "genuine SIGNAL-DESTRUCTION" in survived
    assert "NOT RECOVERABLE BY THE PROBE WHEN CLEFT STRUCTURE IS ABSENT" in survived
    # Even the favourable branch defers to the binding.
    assert "subject to THE BINDING below, which no branch lifts" in survived

    destroyed = _flat(readings["near_zero"])
    assert "UNINFORMATIVE BY CONSTRUCTION" in destroyed
    assert "only BETWEEN bins" in destroyed
    # The corrected design is NAMED but explicitly not built.
    assert "STATED WITHOUT BEING BUILT" in destroyed
    assert "finer strata, or MATCHING rather than binning" in destroyed
    # And the residual is refused by arithmetic, spelled out.
    assert "0.2520 - 0.0414 = 0.2106" in destroyed
    assert "IS NOT TO BE QUOTED AS CLEFT-SPECIFIC UNDER THIS BRANCH" in destroyed
    observed = phase20.ARMS_OBSERVED["arm_s"]["pcc_mean"]
    assert round(0.2520 - observed, 4) == 0.2106

    partial = _flat(readings["between"])
    assert "BOUNDED RATHER THAN RESOLVED" in partial
    assert "no point estimate promoted to an attribution" in partial

    assert "no reading is added once the number exists" in _flat(
        readings["no_reading_is_invented_after"]
    )
    assert "declared before the run" in _flat(
        readings["the_thresholds_are_declared_in_the_config"]
    )


def test_no_branch_licenses_the_cleft_specific_claim():
    binding = phase20.DIAGNOSTIC_BINDING
    stated = _flat(binding["the_binding"])
    assert "under NO BRANCH" in stated
    assert "Arm S's number alone" in stated
    assert "decides whether S is INTERPRETABLE AT ALL" in stated
    assert "does not convert S into an attribution" in stated

    remains = _flat(binding["what_stands_in_the_way_even_then"])
    assert "sd 0.0802" in remains
    assert "spanning BOTH the null and the ceiling" in remains
    assert "0.1386" in remains
    # The figures it quotes are the observed ones, and the floor it
    # cites is the banked one.
    assert phase20.ARMS_OBSERVED["arm_s"]["pcc_sd"] == 0.0802
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"] == 0.1386
    assert "0.04 to 0.10" in _flat(
        ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"]["unresolvable_band"]
    )


# --------------------------------------------------------------------------
# the three rulings, 2026-08-31
# --------------------------------------------------------------------------


def test_the_stratification_is_locked_with_both_reasons_and_the_trade_off():
    ruled = phase20.STRATIFICATION_RULED
    assert ruled["k_strata"] == 10
    assert "panel mean" in ruled["target"]
    assert "IN-SAMPLE on all 237" in ruled["variable"]
    assert "all five" in ruled["variable"]

    # Both reasons survive the lock, not just the choice.
    single = _flat(ruled["reason_not_a_single_statistic"])
    assert "ONE DIMENSION OF FIVE" in single
    assert "+0.1601" in single, "the alternative is named, not waved away"
    oof = _flat(ruled["reason_not_per_seed_oof"])
    assert "SEED-DEPENDENT" in oof
    assert "DESIGN CHOICE, not an estimate" in oof

    # The trade-off, both directions.
    trade = _flat(ruled["the_trade_off_both_directions"])
    assert "more strata preserve confound structure more tightly" in trade
    assert "fewer strata do the reverse" in trade

    # A locked setting, with the movement rule and the reason for it.
    setting = _flat(ruled["a_locked_scientific_setting"])
    assert "NEVER TUNED ACROSS RUNS" in setting
    assert "dated amendment" in setting
    assert "because a result came back inconvenient" in setting

    # The proposal is preserved and points at the lock.
    assert phase20.ARM_S_PROPOSED["ruled_2026_08_31"] == "STRATIFICATION_RULED"
    assert "NOT LOCKED" in phase20.ARM_S_PROPOSED["proposed"], (
        "the proposal is preserved as written; the lock is a new record"
    )


def test_the_fixed_point_arithmetic_is_pinned_not_asserted():
    """E[fixed points] = k for k strata, whatever the stratum sizes."""
    stated = _flat(phase20.STRATIFICATION_RULED["fixed_point_arithmetic"])
    assert "E[fixed points] = 1 per stratum for ANY stratum size" in stated
    assert "10 of 237 (4.2%)" in stated
    assert "~1 of 237 (0.42%)" in stated
    assert "never discarded, never redrawn" in stated

    # Measured against the shipped drawer, not a hand-rolled one.
    for k, n in ((1, 237), (10, 237)):
        strata = phase20.quantile_strata(np.arange(n, dtype=float), k)
        counts = [
            phase20.fixed_points(
                phase20.permute_within_strata(strata, seed=s)
            )
            for s in range(400)
        ]
        assert float(np.mean(counts)) == pytest.approx(float(k), abs=0.35)


def test_the_ledger_ruling_cites_the_precedent_it_rests_on():
    from cleft import phase18

    ruled = phase20.LEDGER_RULED
    assert "no ledger row for any of the three" in ruled["ruled"]
    assert "category error" in _flat(ruled["arm_p"])
    assert "one-sample BCa interval against ZERO" in _flat(ruled["arm_p"])
    # S's row is refused on what it would SAY, not on whether it exists.
    arm_s = _flat(ruled["arm_s"])
    assert "paired BCa is DEFINED" in arm_s
    assert "ATTRIBUTION, not a method comparison" in arm_s
    assert "not what the measurement means" in arm_s

    # The cited precedent is real and says what the ruling says it says.
    grounds = _flat(ruled["grounds"])
    assert "D2_HOME_RULED" in grounds
    precedent = _flat(phase18.D2_HOME_RULED["no_new_ledger_rows"])
    assert "a ledger row is the unit of ARM-CLAIM" in precedent
    assert "measures the criterion, not the arms" in precedent
    assert "a ledger row is the unit of ARM-CLAIM" in grounds


def test_the_compute_question_is_ruled_without_dropping_the_measurement():
    ruled = _flat(phase20.COMPUTE_GATE_DESIGNED["ruled_2026_08_31"])
    assert "FOLDED INTO THE RUNS" in ruled
    assert "no separate gate job" in ruled
    assert "Per-seed wall-clock is reported by every arm" in ruled
    assert "the measurement survives" in ruled


# --------------------------------------------------------------------------
# Arm C, and the error that produced it
# --------------------------------------------------------------------------


def test_arm_c_is_a_new_arm_and_explicitly_not_a_correction():
    record = phase20.ARM_C_REGISTERED
    why = _flat(record["why_it_exists"])
    assert "MEDIAN grade" in why and "PANEL MEAN" in why
    assert "0.6022" in why and "0.5708" in why
    assert "NO VALID ANCHOR" in why

    not_a_fix = _flat(record["not_a_correction_to_phase_13"])
    assert "CORRECT FOR ITS OWN QUANTITY" in not_a_fix
    assert "nothing there is rewritten" in not_a_fix
    assert "it was a COMPARISON made later" in not_a_fix

    # The figures it quotes are live at source.
    assert label_module.LEARNABILITY_237["mean"] == 0.6022
    assert label_module.LEARNABILITY_237["median"] == 0.5708

    # Independent of the gate -- it does not wait on Arm P.
    assert "does not depend on Arm P" in _flat(record["independent_of_the_gate"])


def test_arm_c_readings_are_committed_both_ways_before_its_number():
    readings = phase20.ARM_C_REGISTERED["readings"]
    assert sorted(readings) == ["materially_different", "near_0_1000"]

    near = _flat(readings["near_0_1000"])
    assert "interchangeable after all" in near
    # The benign branch does not become a defence of having made the error.
    assert "harmless IN EFFECT" in near
    assert "rather than treated as vindication of having made it" in near

    far = _flat(readings["materially_different"])
    assert "NEVER A VALID ANCHOR" in far
    assert "must use Arm C's figure" in far

    # And the committed set points at them rather than absorbing them.
    assert phase20.READINGS_COMMITTED["arm_c_readings_2026_08_31"] == (
        "ARM_C_REGISTERED['readings']"
    )


def test_the_cross_target_error_provenance_is_recorded_against_itself():
    record = phase20.CROSS_TARGET_ERROR_PROVENANCE
    origin = _flat(record["origin"])
    assert "CONVERSATION -- the record's" in origin
    assert "2026-08-31 session" in origin
    assert "~40% of the headline arm's" in origin

    caught = _flat(record["caught_when"])
    assert "before any measurement was built on it" in caught
    assert record["what_it_touched"] == "nothing measured; no run rested on it"

    # The quoted sentence is Phase 13's own, verified at source.
    assert "~40% of the headline arm's" in _flat(
        phase13.P1_CONFOUND_CEILING_BANKED["fires_at_the_boundary"]
    )
    # And the trace it names is real code, not a claim about code.
    import inspect

    from cleft import run as run_module

    assert "The median grade per patient" in inspect.getdoc(
        run_module._grades_for
    )


def test_phase_13_carries_a_dated_pointer_and_is_not_rewritten():
    banked = phase13.P1_CONFOUND_CEILING_BANKED
    # Untouched: every original figure and the original verdict.
    assert "mean +0.1000, sd 0.0228" in banked["figures"]
    assert "~40% of the headline arm's" in _flat(banked["fires_at_the_boundary"])
    assert banked["status"].startswith("DESCRIPTIVE")

    pointer = _flat(banked["target_is_the_median_grade_2026_08_31"])
    assert "NOT A CORRECTION" in pointer
    assert "nothing here is rewritten" in pointer
    assert "phase20.ARM_C_REGISTERED" in pointer
    assert "phase20.CROSS_TARGET_ERROR_PROVENANCE" in pointer


# --------------------------------------------------------------------------
# the lock
# --------------------------------------------------------------------------


def test_the_exit_criteria_are_locked_with_the_three_rulings_closed():
    locked = phase20.EXIT_CRITERIA
    assert "2026-08-31" in locked["locked"]
    closed = locked["the_three_rulings_closed"]
    assert sorted(closed) == ["compute", "ledger", "stratification"]
    for value in closed.values():
        assert value.startswith("RULED"), value

    assert len(locked["criteria"]) == 9
    joined = " ".join(locked["criteria"])
    assert "one-sample BCa intervals against ZERO" in joined
    assert "DEGENERATE-STRATUM FREQUENCY" in joined
    assert "the reference point is ZERO" in joined
    assert "NO LEDGER ROW for any of the three arms" in joined
    assert "target-difference statement" in joined
    assert "ARM C" in joined and "NOT a correction to Phase 13" in joined
    assert "suite green" in joined

    # The draft is preserved and points forward.
    assert phase20.EXIT_CRITERIA_DRAFT["status"].startswith("DRAFT")
    assert phase20.EXIT_CRITERIA_DRAFT["superseded_2026_08_31"] == (
        "EXIT_CRITERIA"
    )
    assert len(phase20.EXIT_CRITERIA_DRAFT["criteria"]) == 8, (
        "the draft keeps its own eight; the lock's ninth is Arm C"
    )


def test_the_lock_forbids_adding_to_itself():
    locked = _flat(phase20.EXIT_CRITERIA["locked"])
    assert "nothing is added after this record" in locked
    assert "a criterion discovered missing later is a limitation" in locked
    assert "never a retro-fitted entry" in locked

    # The clause is Phase 18's, and Phase 18 really did honour it.
    from cleft import phase18

    assert "limitation" in _flat(phase20.EXIT_CRITERIA["nothing_added_after"])
    assert hasattr(phase18, "ARM_LIST_ADDENDUM"), (
        "the precedent cited is Phase 18's addendum-not-reopened move"
    )


def test_the_residual_arithmetic_survives_arm_c_moving_the_anchor():
    moves = _flat(phase20.EXIT_CRITERIA["arm_c_moves_the_anchor"])
    assert "computed against ARM C's figure, with BOTH stated" in moves
    assert "0.1000 named beside it with its target difference" in moves
    assert "never quoted against a ceiling on a different target" in moves

    # S's residual stays tied to the resolution floor either way.
    s_reading = _flat(phase20.READINGS_COMMITTED["s_near_the_confound_ceiling"])
    assert "residual = 0.2520 - S_mean" in s_reading
    assert "0.1386" in s_reading
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"] == 0.1386
    assert phase20.READINGS_COMMITTED["the_anchor_may_move_2026_08_31"] == (
        "EXIT_CRITERIA['arm_c_moves_the_anchor']"
    )


# --------------------------------------------------------------------------
# the machinery: strata, permutation, degeneracy
# --------------------------------------------------------------------------


def test_quantile_strata_partition_the_cohort_at_the_locked_k():
    values = np.arange(237, dtype=float)
    strata = phase20.quantile_strata(values, phase20.STRATIFICATION_RULED["k_strata"])

    assert strata.shape == (237,)
    assert sorted(set(strata.tolist())) == list(range(10))
    sizes = np.bincount(strata)
    assert sizes.sum() == 237
    # Quantile bins of 237 into 10: near-equal, smallest 23.
    assert sizes.min() == 23 and sizes.max() == 24

    # Monotone: a larger value never lands in a lower stratum.
    assert np.all(np.diff(strata) >= 0)


def test_quantile_strata_do_not_split_ties_across_bins():
    """Tied fitted values must share a stratum, or two identical
    patients would be treated as differently confounded."""
    values = np.array([1.0] * 100 + [2.0] * 100, dtype=float)
    strata = phase20.quantile_strata(values, 10)
    for value in (1.0, 2.0):
        assert len(set(strata[values == value].tolist())) == 1


def test_the_permutation_stays_inside_its_strata():
    strata = phase20.quantile_strata(np.arange(237, dtype=float), 10)
    perm = phase20.permute_within_strata(strata, seed=1337)

    assert sorted(perm.tolist()) == list(range(237)), "a permutation, not a draw"
    assert np.all(strata[perm] == strata), (
        "every label lands on a patient in its own stratum"
    )


def test_the_plain_arm_is_the_stratified_arm_at_k_equals_one():
    """Arm P is not a second implementation -- it is k = 1."""
    strata = phase20.quantile_strata(np.arange(237, dtype=float), 1)
    assert set(strata.tolist()) == {0}
    perm = phase20.permute_within_strata(strata, seed=1337)
    assert sorted(perm.tolist()) == list(range(237))


def test_the_permutation_is_reproducible_from_its_seed():
    strata = phase20.quantile_strata(np.arange(237, dtype=float), 10)
    a = phase20.permute_within_strata(strata, seed=2024)
    b = phase20.permute_within_strata(strata, seed=2024)
    c = phase20.permute_within_strata(strata, seed=7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_degenerate_strata_are_reported_not_redrawn():
    """A stratum of one can never move. It is COUNTED, never fixed."""
    strata = np.array([0, 0, 0, 1, 2, 2], dtype=int)
    report = phase20.degenerate_strata(strata)
    assert report["singleton_strata"] == [1]
    assert report["n_immovable"] == 1
    assert "NEVER REDRAWN" in report["policy"]
    assert "bias the control toward zero" in report["policy"], (
        "the policy states the DIRECTION of the bias redrawing would add"
    )

    # And such a stratum's patient really is a fixed point, every seed.
    for seed in (1337, 2024, 7, 99, 12345):
        perm = phase20.permute_within_strata(strata, seed=seed)
        assert perm[3] == 3


def test_fixed_points_counts_unchanged_assignments():
    assert phase20.fixed_points(np.array([0, 1, 2])) == 3
    assert phase20.fixed_points(np.array([1, 0, 2])) == 1
    assert phase20.fixed_points(np.array([1, 2, 0])) == 0


# --------------------------------------------------------------------------
# the wiring: one fit path, and an absent permutation changes nothing
# --------------------------------------------------------------------------


def test_an_absent_permutation_leaves_the_fit_path_byte_identical():
    """``label_permutation=None`` must be the pre-existing code path,
    not a re-implementation that happens to agree."""
    import inspect

    from cleft.train import phase3 as phase3_module

    signature = inspect.signature(phase3_module.run)
    assert "label_permutation" in signature.parameters
    assert signature.parameters["label_permutation"].default is None

    # The permutation is applied AFTER the labels load and NOWHERE else.
    source = inspect.getsource(phase3_module.run)
    load = source.index("images, labels, patient_ids, assignments = load_inputs")
    apply = source.index("if label_permutation is not None")
    assert load < apply, "the permutation applies after the label loads"
    assert source.count("labels = labels[order]") == 1, (
        "the labels are reordered in exactly one place"
    )
    # Only the LABELS move. Permuting the patients or their folds too
    # would relabel the cohort and reproduce the probe exactly.
    for untouched in ("patient_ids = patient_ids[", "images = images[",
                      "assignments = assignments["):
        assert untouched not in source, untouched


def _fake_manifest(tmp_path, n: int = 237):
    """A manifest with the shipped column set, enough to exercise the
    plain arm's strata path off-cluster."""
    from cleft.data.manifest import MANIFEST_COLUMNS

    columns = [name for name, _ in MANIFEST_COLUMNS]
    lines = ["# tier: CLUSTER-ONLY", ",".join(columns)]
    for i in range(n):
        row = {name: "0" for name in columns}
        row["patient_id"] = str(i + 1)
        row["frontal_id"] = str(1000 + i)
        row["mean"] = f"{2.0 + (i % 5) * 0.25:.2f}"
        row["fold"] = str(i % 5)
        lines.append(",".join(row[name] for name in columns))
    directory = tmp_path / "manifest_v1"
    directory.mkdir()
    (directory / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return directory


class _Ctx:
    def __init__(self):
        self.lines = []

    def log(self, message):
        self.lines.append(str(message))


def test_arm_p_strata_path_executes_off_cluster():
    """**Arm P is the gate; its code path must not first execute on the
    cluster.** At k = 1 no fit is needed, so the whole path runs here on
    a synthetic manifest -- and what it produces is checked against the
    pre-registered arithmetic rather than merely inspected.
    """
    import tempfile

    from cleft import run as run_module

    with tempfile.TemporaryDirectory() as raw:
        manifest = _fake_manifest(Path(raw))
        ctx = _Ctx()
        strata, report = run_module._p20_strata(
            ctx,
            {"permutation": {"kind": "plain", "k_strata": 1},
             "manifest_artifact": "manifest_v1", "label": "mean"},
            {"manifest_v1": manifest},
        )

    assert len(strata) == 237 and set(strata.tolist()) == {0}
    assert report["k_strata"] == 1
    assert report["expected_fixed_points"] == 1
    assert report["degenerate_strata"]["n_immovable"] == 0
    assert report["degenerate_strata"]["smallest"] == 237
    assert report["fixed_points_by_seed"] == {}, (
        "the prediction is written BEFORE any seed draws"
    )
    # No ridge was fitted: one stratum decides nothing, and a fit in the
    # record that decides nothing is a fit somebody will later cite.
    assert not any("statistics over" in line for line in ctx.lines)
    assert any("expecting 1 unchanged labels" in line for line in ctx.lines)

    # And the arm's actual per-seed permutations behave as registered.
    counts = [
        phase20.fixed_points(phase20.permute_within_strata(strata, seed=s))
        for s in (1337, 2024, 7, 99, 12345)
    ]
    assert all(0 <= c <= 237 for c in counts)
    assert all(
        np.array_equal(
            np.sort(phase20.permute_within_strata(strata, seed=s)),
            np.arange(237),
        )
        for s in (1337, 2024, 7, 99, 12345)
    )


def test_the_permutation_block_is_declared_and_checked_at_load_time():
    from cleft.config.schema import TASK_SPECS, ConfigError, _validate_task

    spec = TASK_SPECS["train_cv"]
    assert "permutation" in spec and not spec["permutation"].required

    base = dict(
        kind="train_cv", manifest_artifact="m", staged_artifact="s",
        geometry="g1", label="mean", backbone="vit_b16",
        max_epochs=1, patience=1, inner_val_frac=0.2, monitor="inner_val_mse",
    )

    # k must match the arm: plain is k = 1, stratified is the locked 10.
    with pytest.raises(ConfigError, match="k_strata"):
        _validate_task({**base, "permutation": {"kind": "plain", "k_strata": 10}})
    with pytest.raises(ConfigError, match="STRATIFICATION_RULED"):
        _validate_task(
            {**base, "permutation": {"kind": "stratified", "k_strata": 5}}
        )
    with pytest.raises(ConfigError, match="not a permutation kind"):
        _validate_task({**base, "permutation": {"kind": "shuffled", "k_strata": 1}})

    # Both registered arms load.
    for kind, k in (("plain", 1), ("stratified", 10)):
        task = _validate_task(
            {**base, "permutation": {"kind": kind, "k_strata": k}}
        )
        assert task["permutation"]["k_strata"] == k


def test_the_permuted_arms_may_not_claim_the_gate_1_reference():
    """A permuted arm matches the reference arm on every field the
    gate-1 check looks at. It must be excluded by the permutation, or
    the control would be compared to the probe's fingerprint."""
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_train_cv)
    reference = source[source.index("reference_arm = ("):]
    assert 'not task.get("permutation")' in reference[: reference.index("\n    )")]


def test_arm_c_switches_the_label_and_leaves_phase_13_untouched():
    from cleft.config.schema import TASK_SPECS, ConfigError, _validate_task

    spec = TASK_SPECS["confound_ceiling"]
    assert "label" in spec and not spec["label"].required
    assert spec["label"].default is None, (
        "absent means Phase 13's median grade -- the default cannot move"
    )
    # The scoresheet is required exactly when the median is being used.
    assert not spec["scoresheet_artifact"].required
    base = dict(
        kind="confound_ceiling", manifest_artifact="m", staged_artifact="s",
        seeds=[1337], inner_val_frac=0.2, alpha=1.0, substantial_pcc=0.10,
        expect_patients=237,
    )
    with pytest.raises(ConfigError, match="scoresheet_artifact is required"):
        _validate_task(base)
    # Arm C's reading threshold is required with the label -- a
    # threshold chosen after the number is not a threshold.
    with pytest.raises(ConfigError, match="materially_different"):
        _validate_task({**base, "label": "mean"})
    # And a score sheet Arm C never opens may not enter its provenance.
    with pytest.raises(ConfigError, match="never opens the score sheet"):
        _validate_task({
            **base, "label": "mean", "materially_different": 0.02,
            "scoresheet_artifact": "sheet",
        })

    arm_c = _validate_task(
        {**base, "label": "mean", "materially_different": 0.02}
    )
    assert arm_c["label"] == "mean" and arm_c["scoresheet_artifact"] is None
    phase_13 = _validate_task({**base, "scoresheet_artifact": "sheet"})
    assert phase_13["label"] is None, (
        "Phase 13's config is reproducible unchanged; the default cannot move"
    )


def test_arm_c_reads_phase_13s_figure_from_a_structured_home():
    """**Values are never regex-parsed out of prose.** Arm C's reading
    turns on the distance from Phase 13's +0.1000, so that number has a
    machine-readable home and the run reads it there."""
    import inspect

    from cleft import run as run_module

    figures = phase13.P1_CEILING_FIGURES
    assert figures["pcc_mean"] == 0.1000
    assert "MEDIAN grade" in figures["target"]
    assert "never regex-parsed" in _flat(figures["provenance"])

    # Transcribed, and the transcription is CHECKED against the prose.
    seeds = figures["pcc_by_seed"]
    assert sorted(seeds) == [7, 99, 1337, 2024, 12345]
    values = np.array([seeds[s] for s in (1337, 2024, 7, 99, 12345)])
    assert float(values.mean()) == pytest.approx(figures["pcc_mean"], abs=5e-5)
    # The sd does NOT reproduce to 4 dp, and the record says so rather
    # than being adjusted to match: the run computed it at full
    # precision, the record rounds the inputs. Measured, then pinned.
    assert float(values.std(ddof=1)) == pytest.approx(
        figures["sd_rederives_to"], abs=5e-7
    )
    assert figures["sd_rederives_to"] != figures["pcc_sd"]
    assert abs(figures["sd_rederives_to"] - figures["pcc_sd"]) < 1e-4
    assert "the RUN's, computed at full precision" in _flat(
        figures["sd_rederivation_note"]
    )
    banked = _flat(phase13.P1_CONFOUND_CEILING_BANKED["figures"])
    for seed, value in seeds.items():
        assert f"+{value:.4f}" in banked, seed
        assert str(seed) in banked, seed

    # And the run reads the constant, not the sentence.
    body = inspect.getsource(run_module.task_confound_ceiling)
    assert 'phase13.P1_CEILING_FIGURES["pcc_mean"]' in body
    assert '["figures"]' not in body
    assert ".split(" not in body


def test_arm_c_reuses_the_ceiling_statistics_and_recipe():
    """Same five statistics, same ridge, same folds -- one
    implementation, called by both."""
    import inspect

    from cleft import decoder, run as run_module

    ceiling = inspect.getsource(run_module.task_confound_ceiling)
    assert "_confound_statistics(" in ceiling
    assert "_confound_ceiling_oof(" in ceiling
    # The extraction is shared, not duplicated.
    source = inspect.getsource(run_module)
    assert source.count("def _confound_statistics") == 1
    assert source.count("def _confound_ceiling_oof") == 1
    assert source.count("decoder_module.global_statistics(") == 1

    # And it is still the five that cannot see a nose.
    assert len(decoder.GLOBAL_STATISTIC_NAMES) == 5


def test_the_diagnostic_scores_arm_s_own_permuted_labels():
    """**The labels the diagnostic scores must be Arm S's, not a fresh
    draw that resembles them.** Same strata, same seeds, same drawer --
    checked by reproducing the permutation, not by assuming it."""
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_p20_stratification_diagnostic)
    # It builds its strata through the SAME function Arm S used.
    assert "_p20_strata(" in body
    assert "phase20.permute_within_strata(" in body
    # And it scores through the SAME recipe Arm C used.
    assert "_confound_ceiling_oof(" in body
    assert "_confound_statistics(" in body
    # No second copy of any of it.
    assert "RidgeBackbone" not in body
    assert "quantile_strata(" not in body, (
        "the strata come from _p20_strata, which is what Arm S ran"
    )

    # The permutation really is reproducible from the strata and seed
    # alone, which is what makes "S's own labels" true rather than hoped.
    strata = phase20.quantile_strata(np.arange(237, dtype=float), 10)
    labels = np.linspace(1.0, 5.0, 237)
    for seed in (1337, 2024, 7, 99, 12345):
        first = labels[phase20.permute_within_strata(strata, seed=seed)]
        second = labels[phase20.permute_within_strata(strata, seed=seed)]
        assert np.array_equal(first, second)
        # And they are genuinely permuted, not returned unchanged.
        assert not np.array_equal(first, labels)


def test_the_diagnostic_config_carries_arm_s_permutation_block_verbatim():
    """If the two blocks ever differ, the diagnostic measures a
    permutation Arm S never ran -- and nothing else would say so."""
    import yaml

    def task_of(stem):
        return yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )["task"]

    arm_s = task_of("p20_permutation_stratified")
    diagnostic = task_of("p20_stratification_diagnostic")

    assert diagnostic["permutation"] == arm_s["permutation"]
    assert diagnostic["label"] == arm_s["label"] == "mean"
    assert diagnostic["seeds"] == arm_s["seeds"]
    assert diagnostic["manifest_artifact"] == arm_s["manifest_artifact"]
    assert diagnostic["staged_artifact"] == arm_s["staged_artifact"]

    # The recipe half is Arm C's, key for key.
    arm_c = task_of("p20_confound_ceiling_mean")
    for shared in ("alpha", "inner_val_frac", "expect_patients"):
        assert diagnostic[shared] == arm_c[shared], shared

    # The anchor is Arm C's OBSERVED figure, carried, not re-typed from
    # prose -- and the two reading thresholds are declared here.
    assert diagnostic["ceiling_pcc"] == (
        phase20.ARMS_OBSERVED["arm_c"]["pcc_mean"]
    )
    assert 0.0 < diagnostic["destroyed_fraction"] < diagnostic[
        "survived_fraction"
    ] < 1.0


def test_the_diagnostic_thresholds_are_refused_when_incoherent():
    from cleft.config.schema import TASK_SPECS, ConfigError, _validate_task

    assert "p20_stratification_diagnostic" in TASK_SPECS
    base = dict(
        kind="p20_stratification_diagnostic", manifest_artifact="m",
        staged_artifact="s", label="mean",
        permutation={"kind": "stratified", "k_strata": 10},
        seeds=[1337], inner_val_frac=0.2, alpha=1.0,
        ceiling_pcc=0.1061, destroyed_fraction=0.30,
        survived_fraction=0.70, expect_patients=237,
    )
    assert _validate_task(base)["ceiling_pcc"] == 0.1061

    # Ordered thresholds, or the three cells overlap and two fire.
    with pytest.raises(ConfigError, match="destroyed_fraction"):
        _validate_task({**base, "destroyed_fraction": 0.8})
    # A plain permutation has nothing to diagnose: it destroys
    # everything by construction, which is the arm's whole point.
    with pytest.raises(ConfigError, match="plain"):
        _validate_task(
            {**base, "permutation": {"kind": "plain", "k_strata": 1}}
        )
    # A zero ceiling would make every fraction undefined.
    with pytest.raises(ConfigError, match="ceiling_pcc"):
        _validate_task({**base, "ceiling_pcc": 0.0})


def test_the_diagnostic_declares_no_new_artifact():
    """'Cheap -- no new artifact, no new hash', checked."""
    import yaml

    diagnostic = yaml.safe_load(
        (REPO / "configs" / "p20_stratification_diagnostic.yaml").read_text(
            encoding="utf-8"
        )
    )
    arm_s = yaml.safe_load(
        (REPO / "configs" / "p20_permutation_stratified.yaml").read_text(
            encoding="utf-8"
        )
    )
    mine = {e["name"]: e for e in diagnostic["inputs"]}
    theirs = {e["name"]: e for e in arm_s["inputs"]}
    # Every input is one Arm S already declared, identically.
    for name, entry in mine.items():
        assert name in theirs, name
        assert entry == theirs[name], name
    # And the embeddings are NOT declared: the diagnostic is a ridge over
    # five scalars and never opens them.
    assert "embeddings" not in mine
    assert set(mine) == {"manifest_v1", "staged_v1"}


def test_the_three_configs_declare_only_existing_artifacts():
    """No new hash enters: every declared input is one an existing
    config already declares, with the same sha."""
    import yaml

    def entries(stem):
        raw = yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )
        return {e["name"]: e for e in raw.get("inputs", [])}

    known: dict[str, set] = {}
    for path in (REPO / "configs").glob("*.yaml"):
        if path.stem.startswith("p20_"):
            continue
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in raw.get("inputs", []) or []:
            known.setdefault(entry["name"], set()).add(
                (entry.get("path"), entry.get("rollup_sha256"))
            )

    for stem in ("p20_permutation_plain", "p20_permutation_stratified",
                 "p20_confound_ceiling_mean",
                 "p20_stratification_diagnostic"):
        mine = entries(stem)
        assert mine, stem
        for name, entry in mine.items():
            assert name in known, f"{stem} declares an unknown input {name!r}"
            assert (entry.get("path"), entry.get("rollup_sha256")) in known[name], (
                f"{stem}: {name} carries a path/hash pair no other config "
                "declares -- a new hash entered through a Phase 20 config"
            )
            # And it is a real verified hash, not a placeholder.
            rollup = entry.get("rollup_sha256") or ""
            assert len(rollup) == 64 and set(rollup) != {"0"}, (
                f"{stem}: {name} carries a placeholder hash"
            )


def test_the_configs_carry_the_locked_settings_not_free_choices():
    import yaml

    def task_of(stem):
        return yaml.safe_load(
            (REPO / "configs" / f"{stem}.yaml").read_text(encoding="utf-8")
        )["task"]

    plain = task_of("p20_permutation_plain")
    strat = task_of("p20_permutation_stratified")
    arm_c = task_of("p20_confound_ceiling_mean")

    assert plain["permutation"] == {"kind": "plain", "k_strata": 1}
    assert strat["permutation"] == {
        "kind": "stratified",
        "k_strata": phase20.STRATIFICATION_RULED["k_strata"],
    }
    # Every arm on the panel mean -- criterion 7 of the lock.
    for task in (plain, strat, arm_c):
        assert task["label"] == "mean", task["kind"]
    # Five seeds, the project's transformer regime, identical across arms.
    seeds = [1337, 2024, 7, 99, 12345]
    for task in (plain, strat, arm_c):
        assert task["seeds"] == seeds, task["kind"]

    # P and S differ from the probe in the permutation block ALONE.
    probe = yaml.safe_load(
        (REPO / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    for task in (plain, strat):
        differing = {
            key for key in set(task) | set(probe)
            if task.get(key) != probe.get(key)
        }
        assert differing <= {"kind", "permutation"}, (
            f"the control differs from the probe in {sorted(differing)}; it "
            "must differ in the permutation alone (and the kind that "
            "makes the permutation required)"
        )

    # Arm C differs from Phase 13's ceiling in the TARGET and nothing
    # that changes the recipe.
    ceiling = yaml.safe_load(
        (REPO / "configs" / "p13_confound_ceiling.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    differing = {
        key for key in set(arm_c) | set(ceiling)
        if arm_c.get(key) != ceiling.get(key)
    }
    assert differing == {"label", "materially_different", "scoresheet_artifact"}
    for shared in ("alpha", "inner_val_frac", "expect_patients",
                   "substantial_pcc", "staged_artifact", "manifest_artifact"):
        assert arm_c[shared] == ceiling[shared], shared
