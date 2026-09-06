"""Phase 11: the asymmetric-loss pre-step, registered before it runs."""

from __future__ import annotations

import numpy as np
import pytest

from cleft import phase10, phase11


def test_the_exit_criteria_are_written_before_any_work():
    """Phase 10 opened with 'exit criteria -- none exist' among its
    silences. Phase 11 does not."""
    record = phase11.PHASE_11_EXIT_CRITERIA
    assert record["registered"].endswith("before any work")
    criteria = record["criteria"]
    assert len(criteria) == record["expected_count"] == 7
    # Numbered, in order, so a criterion cannot be quietly reordered.
    assert [c.split(".")[0] for c in criteria] == [str(i) for i in range(1, 8)]

    joined = " ".join(criteria)
    assert "NEITHER quotable alone" in joined
    assert "CARRIED WITH EVERY NUMBER" in joined
    assert "POWER CHECK" in joined
    assert "DESCRIPTIVELY" in joined and "no ledger row" in joined
    assert "[1, 5]" in joined
    assert "still BLOCKED" in joined
    assert "suite green" in criteria[-1]

    # Neither outcome is allowed to become a result to defend.
    assert any(
        "not a result to defend" in item for item in record["not_criteria"]
    )


def test_the_crossover_finding_is_computed_from_the_recorded_constants():
    """The defect is arithmetic on constants recorded at Phase 10's
    opening -- so it is checked here, not quoted."""
    record = phase11.IEM_CROSSOVER_INVERTS
    assert record["finding"] == "2026-08-17"
    assert np.isclose(record["crossover"], (0.8 / 1.2) ** 4)
    assert np.isclose(phase11.IEM_CROSSOVER, 0.19753086, atol=1e-8)

    # The branches really do cross there, and really do invert below it.
    x = phase11.IEM_CROSSOVER
    assert np.isclose(
        phase11.iem(np.array([-x]))[0], phase11.iem(np.array([x]))[0]
    )
    small = np.array([-0.1])
    assert phase11.iem(small)[0] < phase11.iem(-small)[0]
    big = np.array([-1.0])
    assert phase11.iem(big)[0] > phase11.iem(-big)[0]
    # Continuous at zero: both branches vanish, nothing jumps.
    assert phase11.iem(np.array([0.0]))[0] == 0.0
    assert phase11.iem(np.array([0.0]), "b_swapped")[0] == 0.0

    assert "penalised LESS" in record["below_it"]
    assert "0.0910" in record["below_it"] and "0.1079" in record["below_it"]
    assert "exponent wins" in record["why"]
    assert "a fifth of a grade" in record["so_the_stated_intent"]
    # Stated as an observation about their equation, not a correction.
    assert "PUBLISHED equation" in record["supervisor_material"]
    assert "theirs to say" in record["supervisor_material"]


def test_the_bounds_are_a_domain_and_the_assertion_follows_from_it():
    record = phase11.IEM_BOUNDS_ARE_A_DOMAIN
    # The recorded bound is (0, 4) and IEM leaves it, so it bounds |d|.
    assert phase10.IEM_CARRIED_FOR_PHASE_11["bounds"] == (0, 4)
    at_four = float(phase11.iem(np.array([-4.0]))[0])
    assert at_four > 4.0
    assert np.isclose(at_four, 5.6688, atol=1e-4)
    assert np.isclose(float(phase11.iem(np.array([4.0]))[0]), 2.6723, atol=1e-4)

    assert "a DOMAIN, not a range" in record["what_it_bounds"]
    assert "5.6688" in record["measured"]
    # The consequence is the assertion, and breaches are findings.
    assert "ASSERTED per arm" in record["consequence"]
    assert "rather than silently clipped" in record["consequence"]


def test_the_case_iem_identity_is_open_and_tagged_reasoned():
    record = phase11.CASE_IEM_IDENTITY_OPEN
    assert record["tag"] == "REASONED"
    assert len(record["three_names"]) == 3
    assert any("CASE" in n for n in record["three_names"])
    assert any("equation (16)" in n for n in record["three_names"])
    assert any("deck" in n for n in record["three_names"])
    # It is an assumption, and named as one.
    assert "nothing in the repository establishes it" in record[
        "the_assumption"
    ]
    # And the pre-step proceeds on the only artifact with an equation.
    assert "only artifact with an equation" in record[
        "the_pre_step_proceeds_on"
    ]
    assert "recoverable" in record["the_pre_step_proceeds_on"]
    assert record["for_supervisor"].startswith("is CASE the same metric")


def test_the_direction_inference_predicts_a_and_does_not_unblock():
    record = phase11.IEM_DIRECTION_INFERENCE
    assert "an INFERENCE, not a resolution" in record["recorded"]
    assert "flatters" in record["third_voice"]
    assert "QUALITY axis" in record["the_reconciliation"]
    assert record["predicts"] == "convention A (a_manuscript_literal)"
    # It explicitly refuses to discharge the block.
    assert "leaves every number plausible" in record["does_not_unblock"]
    assert "Both conventions run regardless" in record["does_not_unblock"]
    assert "vacuum" in record["why_recorded_now"]
    # The block itself is untouched by the inference.
    assert phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION["blocking"] == "2026-08-17"


def test_the_pre_step_registration_records_every_decision_with_its_reason():
    record = phase11.PRE_STEP_REGISTERED
    assert record["registered"].endswith("before any number")
    assert "NEITHER quotable alone" in record["both_conventions"]

    truth = record["truth_is_the_median_grade"]
    assert "not the panel mean" in truth["decision"]
    # The reason includes the crossover interaction, not just provenance.
    assert "MANUFACTURE small |d|" in truth["why"]
    assert "crossover" in truth["why"]
    assert "never as G" in truth["the_csv_truth_column_is_not_G"]

    assert "no subset" in record["coverage"]
    assert "OWN band" in record["seed_bands"] and "shared-five" in record[
        "seed_bands"
    ]
    assert "ranked by IEM alone" in record["ranking_rule"]
    assert "SEPARATE column" in record["ranking_rule"]

    # The control is flagged as ours, with what it separates.
    control = record["the_mae_control_is_ours"]
    assert control.startswith("ADDED, and flagged as ours")
    assert "conflates TWO changes" in control
    assert "panel mean -> median grade" in control

    assert record["status"].startswith("DESCRIPTIVE ONLY")
    assert "no interval" in record["why_descriptive"]
    assert "NOT computed" in record["deferred"]
    assert "UNREGISTERED and BLOCKED" in record["blocked_downstream"]


def test_the_readings_and_the_power_check_are_fixed_before_the_numbers():
    """The rule for 'the ranking moves' is pinned now so it cannot be
    chosen after the numbers arrive."""
    record = phase11.PRE_STEP_READINGS
    assert record["registered"].endswith("before the numbers")
    rule = record["ranking_moves_if"]
    assert "top-5 SET" in rule and "tau-b" in rule and "0.90" in rule

    assert "CONDITIONAL on the power check" in record["reading_1_same_ranking"]
    assert "number to show for it" in record["reading_2_different_rankings"]

    power = record["power_check"]
    assert power["threshold"] == 0.01
    assert "SYMMETRIC about zero" in power["why"]
    assert "is not 'the ambiguity does not bite'" in power["why"]
    assert "UNINFORMATIVE" in power["rule"]
    assert "skew" in power["explanatory_columns"]


def test_the_config_shape_decision_is_argued_from_the_deliverable():
    record = phase11.PRE_STEP_CONFIG_SHAPE
    assert record["shape"].startswith("ONE config")
    assert record["rejected"] == "per-scope configs"
    assert "single cross-arm object" in record["why"]
    assert "merged by hand outside any declared artifact" in record["why"]
    assert "three times" in record["why"]
    assert "NO new declaration is needed" in record["cost_is_size_not_risk"]
    # The cost of the choice is stated, not omitted.
    assert "blocks the whole pre-step" in record["what_it_does_cost"]
    assert "Accepted" in record["what_it_does_cost"]


def test_iem_implements_equation_16_and_the_swap_is_a_mirror():
    d = np.array([-4.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 4.0])
    a = phase11.iem(d)
    b = phase11.iem(d, "b_swapped")
    # Convention B is exactly convention A on the mirrored residual --
    # which is what "the opposite direction convention" means.
    assert np.allclose(b, phase11.iem(-d))
    # The manuscript's heavier branch sits on y_hat - G < 0.
    assert np.isclose(a[1], 1.2 * 1.0 ** 1.12)
    assert np.isclose(a[7], 0.8 * 1.0 ** 0.87)
    # No third form exists.
    assert phase11.CONVENTIONS == ("a_manuscript_literal", "b_swapped")
    with pytest.raises(phase11.Phase11Error, match="third documented"):
        phase11.iem(d, "c_something")


def test_the_rank_helpers_behave_and_live_here_because_metrics_is_frozen():
    assert phase11.ranks([3, 1, 2, 1]) == [4, 1, 3, 1]
    assert phase11.ranks([0.5, 0.1, 0.9]) == [2, 1, 3]
    assert phase11.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert phase11.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert np.isclose(
        phase11.kendall_tau_b([1, 2, 3, 4], [2, 1, 3, 4]), 2 / 3
    )
    with pytest.raises(phase11.Phase11Error, match="not the same arms"):
        phase11.kendall_tau_b([1, 2, 3], [1, 2])

    # eval/metrics.py is frozen and carries no rank statistic, which is
    # why these live in the phase file.
    from cleft.eval import metrics

    assert not hasattr(metrics, "kendall_tau_b")


def test_tau_b_is_exactly_one_against_itself_even_with_ties():
    """**[ADDED 2026-09-01] The test that was missing, and its absence is
    why the defect survived.**

    Every case above uses UNTIED vectors, and the previous
    implementation was exact for those. It counted pairs tied in BOTH
    variables into both denominator factors, which only shows up when
    such pairs exist -- and the sharpest symptom is that a vector
    compared against itself stopped being perfect agreement.
    See phase21.TAU_B_DEFECT_CORRECTED.
    """
    for vector in ([1, 1, 2], [1, 1, 2, 2, 3], [0, 0, 0, 1, 1, 2, 3, 3],
                   [5, 5, 5, 5, 1]):
        assert phase11.kendall_tau_b(vector, vector) == pytest.approx(1.0), (
            f"{vector} against itself is perfect agreement"
        )
        reversed_ = [-v for v in vector]
        assert phase11.kendall_tau_b(vector, reversed_) == pytest.approx(-1.0)

    # Ties in ONE variable only: the previous form was already right
    # here, and it must stay right. Checked against the reference rather
    # than a hand-derived constant -- hand-derivation is exactly where
    # the original error crept in, and it crept into this test's first
    # draft too (1/sqrt(2), where the value is 2/sqrt(6)).
    scipy_stats = pytest.importorskip("scipy.stats")
    assert phase11.kendall_tau_b([1, 1, 2], [1, 2, 3]) == pytest.approx(
        scipy_stats.kendalltau([1, 1, 2], [1, 2, 3]).statistic
    )

    # And the general form matches the reference implementation.
    rng = np.random.default_rng(20260901)
    for levels in (3, 5, 10, 40):
        for _ in range(25):
            a = rng.integers(0, levels, 40).tolist()
            b = rng.integers(0, levels, 40).tolist()
            expected = scipy_stats.kendalltau(a, b).statistic
            assert phase11.kendall_tau_b(a, b) == pytest.approx(
                expected, abs=1e-12
            ), (levels, a, b)


def test_the_pre_step_config_covers_the_ladder_and_declares_nothing_new():
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    path = repo / "configs" / "p11_iem_prestep.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    task = payload["task"]
    assert task["kind"] == "iem_prestep"
    assert task["shared_seeds"] == [1337, 2024, 7, 99, 12345]

    inputs = payload["inputs"]
    assert len(inputs) == 492
    names = [e["name"] for e in inputs]
    assert names[:2] == ["manifest_v1", "scoresheet_primary"]
    vectors = [e for e in inputs if e["name"].startswith("oof_")]
    assert len(vectors) == 490

    # Nothing new, nothing pending: every hash is real.
    placeholder = "0" * 64
    for entry in inputs:
        assert entry["rollup_sha256"] != placeholder
        assert len(entry["rollup_sha256"]) == 64
        assert "/PENDING_" not in entry["path"]

    # 68 arms, and every one carries the shared band.
    arms: dict = {}
    for entry in vectors:
        stem, _, seed = entry["name"][len("oof_"):].rpartition("_seed_")
        arms.setdefault(stem, set()).add(int(seed))
    assert len(arms) == 68
    assert all(set(task["shared_seeds"]) <= seeds for seeds in arms.values())
    assert sorted({len(s) for s in arms.values()}) == [5, 10]

    header = path.read_text(encoding="utf-8")
    assert "FITS NOTHING" in header
    assert "DESCRIPTIVE ONLY" in header
    assert "NEITHER RANKING IS QUOTABLE" in header
    assert "0.19753" in header
    assert "MEDIAN GRADE" in header and "never as G" in header
    assert "NOTHING NEW IS DECLARED" in header
    # The registered exit criteria travel in the config itself.
    for criterion in phase11.PHASE_11_EXIT_CRITERIA["criteria"]:
        assert criterion.split(".")[0] + "." in header


def test_the_generator_excludes_unnamed_vectors_only_when_they_add_nothing():
    """p8c_sheet declares five OOF inputs as bare ``oof_seed_<n>``. They
    cannot be parsed into an arm, and the generator refuses to skip any
    whose FILE is not already carried under a proper name -- otherwise a
    whole arm could vanish from the pre-step in silence."""
    import importlib.util
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "generate_phase11_configs",
        repo / "scripts" / "generate_phase11_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    vectors = module.declared_vectors()
    assert len(vectors) == 490
    assert all(e["name"].startswith("oof_") for e in vectors)
    carried = {e["path"] for e in vectors}

    # [2026-08-17] Phase 11's own arms are declared and verified, and are
    # NOT in the pre-step: its coverage is part of a CLOSED result, and an
    # arm created afterwards is not retroactively part of what it
    # measured. The pinned 68/490 caught this the moment they landed.
    # [2026-08-23] Phase 12's four view-ablation arms joined the
    # exclusion on the same ground -- and a stronger one: they run a
    # DIFFERENT COHORT (236), so they could not even pair against the
    # pre-step's truth column. The closed 68/490 coverage caught them
    # the moment their hashes were verified, exactly as it caught
    # Phase 11's own arms.
    assert module.EXCLUDED_FROM_THE_PRESTEP == (
        "p11_iem_arm", "p11_mse_control",
        "p12_arm_a_frontal", "p12_arm_b_basal",
        "p12_arm_c_concat", "p12_arm_d_capacity",
    )
    assert not any(
        stem in e["name"] for e in vectors
        for stem in module.EXCLUDED_FROM_THE_PRESTEP
    )
    generator = (
        repo / "scripts" / "generate_phase11_configs.py"
    ).read_text(encoding="utf-8")
    assert "coverage is CLOSED" in generator
    assert "silently regrow the" in generator
    assert "PRESTEP_CLOSED" in generator

    # The bare names exist, are excluded, and cost nothing.
    sheet = yaml.safe_load(
        (repo / "configs" / "p8c_sheet.yaml").read_text(encoding="utf-8")
    )
    bare = [
        e for e in sheet["inputs"]
        if e["name"].startswith("oof_")
        and not e["name"][len("oof_"):].rpartition("_seed_")[0]
    ]
    assert len(bare) == 5
    assert all(e["name"] not in {v["name"] for v in vectors} for e in bare)
    assert all(e["path"] in carried for e in bare)

    # And the config it writes is the one on disk.
    assert module.main(["--check"]) == 0


def test_the_task_scores_against_the_median_and_claims_nothing():
    import inspect

    from cleft import run as run_module

    assert "iem_prestep" in run_module.TASKS
    source = inspect.getsource(run_module.task_iem_prestep)
    # G is the sheet's median, resolved by the SHARED helper.
    assert "median_by_patient(rows, sheet_path, patient_ids)" in source
    assert "never as G" in source
    # Both conventions, over the model's own list.
    assert "for convention in phase11.CONVENTIONS" in source
    # The registered rule is applied, not re-decided.
    assert 'tau < 0.90' in source
    assert "power_sufficient" in source and "0.01" in source
    assert "UNINFORMATIVE" in source
    assert "READING 1" in source and "READING 2" in source
    # Exit criteria 2, 3 and 5 are computed rather than asserted after.
    assert "below_crossover_share" in source
    assert "residual_skew" in source
    assert "domain_breaches" in source
    assert "reported, not clipped" in source
    # DESCRIPTIVE: it writes no ledger row and makes no claim.
    assert "DESCRIPTIVE ONLY -- no claim, no ledger row" in source
    assert "results_ledger" not in source
    # It fits nothing: no backbone, no harness, no optimiser.
    for training in ("make_backbone", "harness.run_cv", "train_epoch"):
        assert training not in source


def test_the_median_resolution_has_one_implementation():
    """Two copies of a label resolution is how two runs quietly measure
    against different labels."""
    import inspect

    from cleft import run as run_module

    assert callable(run_module.median_by_patient)
    helper = inspect.getsource(run_module.median_by_patient)
    assert "scoresheet.load_median" in helper
    assert "has no row in the" in helper

    # Both callers use it; neither re-derives the mapping.
    for task in (run_module.task_cleftgnn_cv, run_module.task_iem_prestep):
        source = inspect.getsource(task)
        assert "median_by_patient(" in source
        assert "load_median(" not in source


def test_the_schema_admits_the_pre_step_and_nothing_looser():
    from cleft.config.schema import TASK_SPECS

    spec = TASK_SPECS["iem_prestep"]
    assert spec["kind"].choices == ("iem_prestep",)
    assert set(spec) == {
        "kind", "manifest_artifact", "scoresheet_artifact", "shared_seeds",
    }
    # The score sheet is declared because its Median column IS G.
    assert "Median column is G" in spec["scoresheet_artifact"].doc


def test_the_phase_summary_carries_every_record():
    summary = phase11.summary()
    # Registration first, then what the run observed -- both importable
    # as one object, so a reader never has to know which file to open.
    assert set(summary) == {
        "scope", "exit_criteria", "pre_step", "readings", "config_shape",
        "crossover", "bounds", "case_identity", "direction_inference",
        "observed", "mae_decomposition", "hedging_mechanism",
        "hedging_check", "domain_breaches", "exit_walk", "while_blocked",
        "hedging_check_observed", "narrow_predictors", "errors",
        "supervisor_ask", "closed",
        "direction_answered", "degeneracy_pull", "build_scope",
        "sequence_renumbered",
        "build_registered", "loss_verbatim", "degeneracy_rule",
        "build_exit_criteria", "arms_observed", "paired_coverage",
        "paired_observed", "log_defect", "closing",
    }


# --------------------------------------------------------------------------
# 2026-08-17, the paired run withdraws both, and Phase 11 closes
# --------------------------------------------------------------------------


def test_both_contrasts_withdrew_on_condition_one_as_predicted():
    record = phase11.PAIRED_LOSS_OBSERVED
    assert "p11_paired__75e9dd8a__p11-paired" in record["observed"]
    by_metric = record["by_metric"]
    assert set(by_metric) == {"pcc", "iem"}
    for cell in by_metric.values():
        # PLAN 4.3 needs BOTH; one true and one false withdraws.
        assert cell["condition_1"] is False
        assert cell["condition_2"] is True
        assert cell["verdict"] == "WITHDRAWN"
        # Both margins sit above the 2.55x floor but below the 3.6x
        # region where the outcome starts being ordered by margin.
        assert 2.55 < cell["margin"] < 3.6
    assert by_metric["pcc"]["d"] == -0.0697
    assert by_metric["iem"]["d"] == -0.0305
    assert by_metric["iem"]["lower_is_better"] is True
    assert by_metric["iem"]["baseline_sd_recomputed"] == 0.00744

    # The predictions came from the registration and are checked against it.
    recorded = phase11.paired_claim_pairs("p11_loss")[0]["recorded"]
    for metric in ("pcc", "iem"):
        assert recorded[metric]["delta_of_means"] == by_metric[metric]["d"]
        assert recorded[metric]["margin"] == by_metric[metric]["margin"]
    # Not oversold: the arithmetic was derived, the verdict was at risk.
    assert "arithmetic was never what was at risk" in record["prediction_held"]
    assert "condition 1 would decide" in record["prediction_held"]

    # The figures came from the artifact, and the record says so.
    assert "not the log" in record["figures_are_from_the_artifact"]

    # The descriptive finding is unchanged, and the refusal is the
    # criterion working.
    assert "0.0697" in record["descriptive_finding_stands"]
    assert "belongs to the LOSS" in record["descriptive_finding_stands"]
    assert "a fifth independent time" in record["fifth_arrival"]
    assert "criterion WORKING, not failing" in record["fifth_arrival"]


def test_the_partial_log_is_recorded_as_the_fourth_of_its_family():
    record = phase11.THE_LOG_WAS_PARTIAL
    assert record["defect"].endswith("found and fixed")
    assert "half the result" in record["what"]
    # Named as a recurrence, with its three predecessors.
    family = record["the_family"]
    assert "fourth instance" in family
    assert "rebuilt the pipeline" in family and "25 curves" in family
    assert "U-shape" in family
    assert "THE CHANNEL A HUMAN READS IS NOT, AND NOTHING FAILS" in family
    # Why the criterion did not catch it -- the distinction it lacked.
    assert "REPORTED IN EVERY CHANNEL" in record["why_the_criterion_missed_it"]
    # The fix generalises rather than patching the one case.
    assert "A THIRD metric would print itself" in record["fix"]

    # And the fix is in the code: one line per metric, from by_metric.
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_paired_claims)
    assert "for name, comparison in by_metric.items():" in source
    assert "THE_LOG_WAS_PARTIAL" in source
    assert "lower is better" in source


def test_both_withdrawals_are_ledgered_and_the_earlier_pins_hold():
    from cleft import results_ledger

    results_ledger.validate()
    # [2026-08-23] `>=`, not `==`: the ledger is append-only and Phase 12
    # added rows -- the same lesson test_phase10 already took, missed
    # here in that sweep. The PREFIX CHECKSUMS below are the guard.
    assert len(results_ledger.ENTRIES) >= 28
    # Appending leaves every earlier prefix exactly as it was.
    assert results_ledger.cumulative_checksum(26) == (
        "f310ef8f284ada62525ab6625fcb1561de9914602d3ea12f8a4adf30a33920df"
    )
    assert results_ledger.cumulative_checksum(28) == (
        "5fa5962f0858b76e847de73ae2e7d24912472b3717ed9b022d097ad8e2c4892f"
    )

    pcc_row, iem_row = results_ledger.ENTRIES[26], results_ledger.ENTRIES[27]
    assert pcc_row["id"] == "p11-iem-loss-costs-pcc-withdrawn"
    assert iem_row["id"] == "p11-iem-loss-improves-iem-withdrawn"
    for row in (pcc_row, iem_row):
        assert row["status"] == "WITHDRAWN"
        assert row["phase"] == "p11"
        assert row["corrects"] is None
        assert "FALSE" in row["condition_1"]
        assert "2.55x" in row["condition_2"] or "2.68x" in row["condition_2"]
        # The eq (16) defects travel with every number.
        assert any("0.19753" in c or "same two eq (16) defects" in c
                   for c in row["caveats"])

    # The withdrawn descriptive picture is stated, not hidden.
    assert "DISJOINT ranges" in pcc_row["caveats"][0]
    assert "COHORT_CANNOT_RESOLVE" in pcc_row["caveats"][0]
    assert "belongs to the LOSS" in pcc_row["caveats"][1]
    # The IEM row records its own direction and its pairing obligation.
    assert "LOWER IS BETTER" in iem_row["caveats"][0]
    assert "travel TOGETHER by registration" in iem_row["caveats"][1]


def test_phase_11_closes_on_six_met_criteria_and_carries_four_items():
    record = phase11.PHASE_11_CLOSING
    assert record["closed"] == "2026-08-17"
    criteria = record["exit_criteria"]
    assert len(criteria) == 6
    assert all(v.startswith("MET") for v in criteria.values())
    # Criterion 3 records that it was met in the artifact before the log.
    assert "THE_LOG_WAS_PARTIAL" in criteria["3_both_metrics"]

    established = record["established"]
    assert "0.232" in established["pre_step"] and "0.845" in established[
        "pre_step"
    ]
    assert "0.19753" in established["the_metric_has_two_defects"]
    assert "0.0719" in established["the_metric_has_two_defects"]
    assert "SCORED and TRAINED" in established["the_metric_has_two_defects"]
    assert "0.1855 against 0.2552" in established[
        "the_loss_works_and_the_bill_is_correlation"
    ]
    assert "target change costs nothing" in established[
        "the_loss_works_and_the_bill_is_correlation"
    ]
    assert "FIFTH arrival" in established["and_none_of_it_is_claimable"]

    # The negative result is claimed as a result, because it was registered.
    negative = record["the_predicted_degeneracy_did_not_materialise"]
    assert "0 of 25 in both arms" in negative
    assert "finding rather than a shrug" in negative
    assert "WITHDRAWN" in record["nothing_new_is_claimed"]

    # Four things carried forward, each named so none is lost to a summary.
    carried = record["carried_forward_open"]
    assert len(carried) == 4
    for name in (
        "IEM_CROSSOVER_INVERTS", "IEM_BOUNDS_ARE_A_DOMAIN",
        "A_FAVOURS_NARROW_PREDICTORS", "DEGENERACY_PULL_REGISTERED",
        "SUPERVISOR_CONSOLIDATED_ASK", "PHASE_11_BUILD_REGISTERED",
    ):
        assert any(name in item for item in carried), name
    assert any("still\nDEFERRED" in item or "DEFERRED" in item
               for item in carried)
    assert any("FUTURE ARM" in item for item in carried)
    assert any("none of which blocks anything" in item for item in carried)
    assert "Phase 12" in record["next"]

    # Every carried record resolves, so a carried name cannot rot.
    for name in (
        "IEM_CROSSOVER_INVERTS", "IEM_BOUNDS_ARE_A_DOMAIN",
        "A_FAVOURS_NARROW_PREDICTORS", "DEGENERACY_PULL_REGISTERED",
        "SUPERVISOR_CONSOLIDATED_ASK", "PHASE_11_BUILD_REGISTERED",
    ):
        assert getattr(phase11, name)


def test_the_phase_scope_lists_its_silences_and_their_resolutions():
    scope = phase11.summary()["scope"]
    # The silences were listed, then resolved, and both are visible.
    assert len(scope["silences_resolved"]) == 12
    assert scope["silences_resolved"]["config_shape"].startswith("ONE config")
    assert len(scope["still_open"]) == 3
    assert any("one sentence" in item for item in scope["still_open"])
    # The amendment's two commitments are quoted, not paraphrased.
    assert "costs nothing" in scope["commits"]["pre_step"]
    assert "its own arms" in scope["commits"]["loss_build"]


# --------------------------------------------------------------------------
# 2026-08-17, the pre-step runs: reading 2 on both bands
# --------------------------------------------------------------------------


def test_reading_two_fired_by_the_registered_rule_not_by_preference():
    record = phase11.PRE_STEP_OBSERVED
    assert "p11_iem_prestep__b0de1791__p11-iem-prestep" in record["observed"]
    assert record["verdict"].startswith("READING 2")

    tau = record["tau_b_between_conventions"]
    assert record["registered_threshold"] == 0.90
    # The rule, applied: both bands below the registered threshold.
    assert tau["own"] < record["registered_threshold"]
    assert tau["shared5"] < record["registered_threshold"]
    assert record["top5_disjoint"] is True

    # And the power check passed, so this is not the uninformative branch.
    power = record["power"]
    assert power["floor"] == phase11.PRE_STEP_READINGS["power_check"][
        "threshold"
    ]
    assert power["max_relative_gap"] > power["floor"]
    assert power["sufficient"] is True
    assert "NOT the uninformative branch" in power["reading"]

    assert len(record["top5_a"]) == 5 and len(record["top5_b"]) == 5
    assert not set(record["top5_a"]) & set(record["top5_b"])
    assert "p10_cleftgnn" in record["top5_a"]
    assert "p7_d1_vit_b16_imagenet_g1" in record["top5_b"]
    assert "inverts which end of it is good" in record["the_inversion"]
    assert "NUMBER TO SHOW FOR IT" in record["registered_reading_applied"]


def test_the_mae_control_separates_the_confound_it_was_added_for():
    record = phase11.MAE_CONTROL_DECOMPOSITION
    # The target+family change costs most of the agreement; the asymmetry
    # costs comparatively little.
    assert record["tau_mae_vs_pcc"] < record["tau_iem_a_vs_mae"]
    assert record["tau_iem_a_vs_mae"] > 0.8
    assert record["tau_mae_vs_pcc"] < 0.3
    assert "NOT THE ASYMMETRY" in record["decomposition"]
    assert "confound inside the answer" in record["why_the_control_was_needed"]
    # MAE sits BETWEEN the two mirror-image conventions -- a free check
    # that the implementation is doing what it should.
    between = phase11.PRE_STEP_OBSERVED["tau_b_between_conventions"]["own"]
    assert record["tau_iem_a_vs_mae"] > between
    assert "should look like" in record["free_consistency_check"]


def test_the_proposed_crossover_mechanism_is_tested_and_not_adopted():
    """The explanation offered was tested against the measured label
    distribution and the published constants, and it did not survive."""
    record = phase11.HEDGING_MECHANISM_MEASURED
    assert "crossover" in record["proposed_explanation"]
    assert len(record["not_supported"]) == 2
    assert "NOTHING lands in the crossover region" in record[
        "not_supported"
    ][0]
    assert "not a property of eq (16)" in record["not_supported"][1]

    # What replaced it is measurable and is checked here, not quoted.
    optimum = record["optimal_constant"]
    assert optimum["a"] == 3.000 and optimum["mae"] == 3.000
    assert abs(optimum["a"] - optimum["b"]) > 0.8
    assert "FULL GRADE" in record["the_finding"]
    assert "WHICH DIRECTION TO HEDGE IN" in record["the_finding"]
    # The strong claim is withdrawn, with the reason.
    assert "not established here" in record["degenerate_optimum_not_established"]
    assert "shared with MAE" in record["degenerate_optimum_not_established"]
    assert "illustrative" in record["provenance"]

    # The mechanism reproduces from the measured distribution: a constant
    # at 3 overstates severity for 94 and understates for 33, and under A
    # the 94 take the light branch.
    grades = np.concatenate(
        [np.full(n, g, float) for g, n in ((1, 5), (2, 89), (3, 110),
                                           (4, 30), (5, 3))]
    )
    assert int((grades < 3).sum()) == 94
    assert int((grades > 3).sum()) == 33
    constant = np.full_like(grades, 3.0)
    residual = constant - grades
    # Overstating severity is a POSITIVE residual, which convention A
    # sends to the light branch.
    heavy = phase11.iem(residual)[residual < 0]
    light = phase11.iem(residual)[residual > 0]
    assert np.all(light <= heavy.max())
    # And the optimal constants really are a grade apart.
    grid = np.arange(1.0, 5.001, 0.005)
    best = {
        convention: float(grid[int(np.argmin([
            np.mean(phase11.iem(np.full_like(grades, c) - grades, convention))
            for c in grid
        ]))])
        for convention in phase11.CONVENTIONS
    }
    assert abs(best["a_manuscript_literal"] - 3.0) < 0.01
    assert best["a_manuscript_literal"] - best["b_swapped"] > 0.8


def test_the_arm_level_check_is_registered_with_predictions_before_the_read():
    """Predictions registered before per_arm.csv is read, so the checker
    confirms or refutes rather than illustrates."""
    record = phase11.HEDGING_CHECK_REGISTERED
    assert "before per_arm.csv is read" in record["registered"]
    checks = record["checks"]
    assert set(checks) == {
        "1_crossover", "2_direction_of_hedge", "3_narrowest_spanning",
    }
    assert checks["1_crossover"]["predicted"].startswith("REFUTED")
    assert checks["2_direction_of_hedge"]["predicted"].startswith("HELD")
    # No prediction is invented where the offline work says nothing.
    assert checks["3_narrowest_spanning"]["predicted"].startswith("NONE")
    assert "decoration" in checks["3_narrowest_spanning"]["predicted"]
    # Every threshold is a number, not a vibe.
    assert "1.5x" in checks["1_crossover"]["holds_if"]
    assert "0.3 grades" in checks["2_direction_of_hedge"]["holds_if"]
    # And each outcome has a consequence fixed in advance.
    assert len(record["consequences"]) == 3
    assert any("withdrawn to a question" in c for c in record["consequences"])

    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / record["script"]
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    # The checker reads the registered thresholds' record rather than
    # carrying its own copy of the predictions.
    assert "phase11.HEDGING_CHECK_REGISTERED" in text
    # It prints the PREDICTION beside the outcome; a checker that shows
    # only the outcome lets a refutation read as a result.
    assert text.count("predicted") >= 4
    assert "lets a refutation read as a result" in text


def test_the_domain_breach_direction_is_flagged_rather_than_filed():
    """The reported description does not match the reported numbers, and
    a phase-relevant fact is not recorded on a mismatch."""
    record = phase11.DOMAIN_BREACHES_OBSERVED
    assert record["count"] == 12
    assert "reported, not clipped" in record["criterion_5_worked"]
    # 3.85-4.74 are inside [1,5] and cannot have tripped the assertion.
    assert "INSIDE [1,5]" in record["discrepancy"]
    assert "min < 1.0" in record["discrepancy"]
    # Why it is not bookkeeping: a sub-1 prediction is optimistic.
    assert "OPTIMISTIC" in record["why_it_matters"]
    assert "recording it backwards" in record["why_it_matters"]
    # The count stands, the direction does not.
    assert "STAND" in record["status"] and "does not" in record["status"]

    # The assertion in the task really does fire on min OR max.
    import inspect

    from cleft import run as run_module

    source = inspect.getsource(run_module.task_iem_prestep)
    assert "if low < 1.0 or high > 5.0:" in source


def test_the_exit_walk_meets_seven_and_names_two_reads_before_closing():
    record = phase11.PRESTEP_EXIT_WALK
    criteria = record["criteria"]
    assert len(criteria) == 7
    assert all(v.startswith("MET") for v in criteria.values())
    # Criterion 5 is met as a criterion while its finding is incomplete,
    # and the record says both.
    assert "MET as a criterion" in criteria["5_domain"]
    assert "FINDING is incomplete" in criteria["5_domain"]
    # Criterion 2 was not merely satisfied but used.
    assert "TESTED as the proposed mechanism" in criteria["2_crossover_carried"]

    assert record["can_it_close"].startswith("YES, but not this turn")
    assert len(record["outstanding"]) == 2
    assert "min column" in record["outstanding"][0]
    assert "p11_prestep_check.py" in record["outstanding"][1]
    assert "neither needs a run" in record["both_are_reads"]
    assert "paid for three times" in record["why_not_close_first"]


def test_what_to_do_while_blocked_names_what_not_to_do_and_why():
    record = phase11.PHASE_11_WHILE_BLOCKED
    worth = record["worth_doing"]["consolidate_the_ask_for_supervisor"]
    assert "seven questions" in worth
    assert "three phases" in worth
    assert "NUMBER attached" in record["worth_doing"]["why_now"]

    declined = record["not_worth_doing"]
    assert set(declined) == {
        "provisional_loss_under_a", "fgcm_and_ndcg",
        "a_third_symmetric_variant",
    }
    assert "the blocked thing" in declined["provisional_loss_under_a"]
    assert "multiplies the interpretation" in declined["fgcm_and_ndcg"]
    assert "declined" in declined["a_third_symmetric_variant"]

    # The structural answer refuses to fill the wait with invented work.
    structural = record["the_structural_answer"]
    assert "PHASE 12 next" in structural
    assert "manufacture Phase 11 work" in structural
    assert "how a block stops being visible" in structural


# --------------------------------------------------------------------------
# 2026-08-17, the falsifier inverts both predictions; the pre-step closes
# --------------------------------------------------------------------------


def test_the_falsifier_inverted_both_predictions_and_the_consequence_applied():
    record = phase11.HEDGING_CHECK_OBSERVED
    checks = record["checks"]
    # The two that carried predictions both inverted.
    assert checks["1_crossover"]["predicted"] == "REFUTED"
    assert checks["1_crossover"]["outcome"] == "HELD"
    assert checks["2_direction_of_hedge"]["predicted"] == "HELD"
    assert checks["2_direction_of_hedge"]["outcome"] == "REFUTED"
    assert record["both_predictions_inverted"] is True
    # The one that carried none is the sharpest, and is said to be.
    assert checks["3_narrowest_spanning"]["predicted"] == "none registered"
    assert checks["3_narrowest_spanning"]["outcome"] == "HELD"
    assert "argument for having registered it without one" in record[
        "check_3_is_the_sharpest"
    ]
    # The registered consequence was applied rather than reinterpreted.
    assert "without interpretation" in record["consequence_applied"]

    # And the corrected record says so IN PLACE, not by deletion.
    corrected = phase11.HEDGING_MECHANISM_MEASURED["corrected"]
    assert corrected.startswith("2026-08-17")
    assert "BOTH refutations fail" in corrected
    assert "1.685" in corrected and "+0.0369" in corrected
    # The refuted claims are still there to be seen.
    assert phase11.HEDGING_MECHANISM_MEASURED["optimal_constant"]["a"] == 3.0


def test_the_measured_mechanism_reproduces_and_names_what_it_corrects():
    record = phase11.A_FAVOURS_NARROW_PREDICTORS
    assert record["corrects"] == "HEDGING_MECHANISM_MEASURED"
    mechanism = record["mechanism"]
    assert "1.0715" in mechanism and "2.2611" in mechanism
    assert "0.3072" in mechanism and "0.1823" in mechanism
    assert "1.685x" in mechanism
    assert "charges LESS for optimism" in mechanism

    # The three steps of the mechanism hold as arithmetic, checked here.
    x = phase11.IEM_CROSSOVER
    for magnitude in (0.05, 0.1, 0.15):
        assert magnitude < x
        optimistic = float(phase11.iem(np.array([-magnitude]))[0])
        pessimistic = float(phase11.iem(np.array([magnitude]))[0])
        assert optimistic < pessimistic
    assert 1.685 > 1.5  # the registered threshold, cleared

    # The reproduction is reported WITH the reason the first attempt
    # missed it.
    reproduced = record["reproduced_after_the_fact"]
    assert "0.2574" in reproduced and "0.3460" in reproduced
    assert "0.3072" in reproduced
    assert "tanh arm tightly centred at 2.75" in reproduced
    assert "the synthetic missed it" in reproduced
    assert "about the METRIC, not about our arms" in record["supervisor_material"]


def test_all_four_errors_are_recorded_together_with_one_lesson():
    record = phase11.FOUR_ERRORS_ONE_PLACE
    assert set(record) == {
        "recorded", "policy_1_crossover_account",
        "policy_2_violation_direction", "mine_1_offline_refutation",
        "mine_2_direction_of_hedge", "the_lesson", "what_worked",
    }
    # Right-in-substance is distinguished from right-for-the-reason-given.
    assert "RIGHT IN SUBSTANCE, WRONG IN ITS REASONING" in record[
        "policy_1_crossover_account"
    ]
    assert "MODAL grade" in record["policy_1_crossover_account"]
    assert "min < 1.0" in record["policy_2_violation_direction"]
    # Both of mine are named as the same move.
    assert "treated its residual distribution as representative" in record[
        "mine_1_offline_refutation"
    ]
    assert "extrapolated across that gap" in record["mine_2_direction_of_hedge"]
    lesson = record["the_lesson"]
    assert "NOT A MEASUREMENT OF IT" in lesson
    assert "both of mine are the same move" in lesson.lower()
    assert "not the error but the confidence" in lesson
    # And what caught them is recorded too.
    assert "before the numbers" in record["what_worked"]


def test_the_breach_direction_is_now_recorded_correctly():
    record = phase11.DOMAIN_BREACHES_OBSERVED["resolved"]
    assert "12 arm-bands over 6 distinct arms" in record["count"]
    assert "none exceeds 5.0" in record["all_lower_edge"]
    assert record["minima"] == (0.2142, 0.9707)
    assert "p7_d_agnet_imagenet_g2_native" in record["extreme"]
    assert "BETTER THAN EXCELLENT" in record["what_they_are"]
    # The two conventions score them oppositely, which is the point.
    assert "HEAVY branch" in record["which_branch"]
    assert "LIGHT one" in record["which_branch"]
    assert "finding now has its direction" in record["criterion_5_complete"]
    # The original mismatch is preserved, not overwritten.
    assert "INSIDE [1,5]" in phase11.DOMAIN_BREACHES_OBSERVED["discrepancy"]


def test_the_supervisor_ask_loses_question_one_by_answer_not_by_deletion():
    """[2026-08-17] Question 1 is answered and has left the ask. It is
    recorded as answered rather than dropped, and the remaining six keep
    their original numbers so a reference to 'question 5' still holds."""
    record = phase11.SUPERVISOR_CONSOLIDATED_ASK
    questions = record["questions"]
    # [2026-08-23] Question 8 joined -- the rating-protocol question, the
    # HIGHEST-VALUE item despite carrying the last number: numbers are
    # stable identifiers, not priority, so 2-7 keep theirs.
    # [2026-08-29] Question 9 joined -- Phase 17's landmark-boundary
    # question (tested in test_phase17.py). This pin held the previous
    # count and FIRED on the append, which is what a pinned count is
    # for; updated dated, same rule: 2-8 keep their numbers.
    assert len(questions) == 8
    assert [q["n"] for q in questions] == list(range(2, 10))
    eight = questions[-2]
    assert eight["priority"].startswith("HIGHEST-VALUE")
    assert "frontal photograph only" in eight["ask"]
    assert "frontal and submental together" in eight["ask"]
    assert "frontal's ID" in eight["ask"]
    assert "never saw adds" in " ".join(eight["why"].split())
    assert "question 8" in record["highest_value_as_of_2026_08_23"]
    # Nothing on the list blocks anything now.
    assert not any(q["blocking"] for q in questions)
    assert record["nothing_blocks_now"].startswith("2026-08-17")
    assert "none of them holds up a build" in record["nothing_blocks_now"]

    # The departed question is visible, with why it became askable.
    gone = record["answered_and_removed"]
    assert gone["n"] == 1
    assert "convention A" in gone["answered"]
    why = gone["why_it_was_askable"]
    assert "0.5917" in why and "DISJOINT" in why
    assert "PCC 0.024" in why and "1.7x" in why

    # Every remaining question points at a record that exists -- a
    # consolidated ask whose pointers rot is worse than none.
    from cleft import ladder, phase9, phase10

    # [2026-08-29] phase17 joined when question 9 did -- the walker
    # resolves every cited record, so a new citing module registers here.
    from cleft import phase17

    modules = {
        "phase9": phase9, "phase10": phase10, "ladder": ladder, "": phase11,
        "phase17": phase17,
    }
    for question in list(questions) + [gone]:
        for reference in str(question["records"]).split(", "):
            module, _, name = reference.strip().rpartition(".")
            assert getattr(modules[module], name), reference

    assert "scattered is where a question goes unasked" in record[
        "why_consolidated"
    ]


def test_the_direction_is_answered_and_the_prediction_was_registered_first():
    record = phase11.IEM_DIRECTION_ANSWERED
    assert record["answer"] == "convention A"
    assert "clinician might not investigate" in record["supervisor_reasoning"]
    assert "y_hat - G < 0" in record["on_the_scale"]
    assert "heavier" in record["on_the_scale"]
    # The published equation needs no correction.
    assert "no correction" in record["the_equation_as_printed_is_right"]

    # The prediction was dated BEFORE the answer, and both records say so.
    assert "dated before, confirmed after" in record["predicted_before_answered"]
    confirmed = phase11.IEM_DIRECTION_INFERENCE["confirmed"]
    assert confirmed.startswith("2026-08-17")
    assert "registered BEFORE the answer" in confirmed
    assert phase11.IEM_DIRECTION_INFERENCE["predicts"].startswith(
        "convention A"
    )

    # The deck is resolved against, not deleted.
    outlier = record["the_deck_is_resolved_against_not_deleted"]
    assert "mixes the two axes" in outlier
    assert "Kept because it is what the artifact says" in outlier
    # And the original ambiguity record still stands unedited.
    from cleft import phase10

    assert "the_ambiguity" in phase10.IEM_IS_THEIRS_DIRECTION_AMBIGUOUS


def test_the_block_is_lifted_everywhere_that_carried_it():
    from cleft import phase10

    block = phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION
    assert block["lifted"].startswith("2026-08-17")
    assert "convention A" in block["lifted"]
    assert "no longer blocked" in block["lifted"]
    # The block's own text stays -- it was real, and its reasoning is what
    # made the answer worth asking for.
    assert block["blocking"] == "2026-08-17"
    assert "no internal detector" in block["why_blocking_and_not_a_caveat"]

    # Phase 10's closing carried it as one of six; that item is resolved.
    resolved = phase10.PHASE_10_CLOSING["carried_forward_resolved"]
    assert "item 1" in resolved and "ANSWERED" in resolved
    assert "nothing is blocked now" in resolved
    assert "The other five stand" in resolved
    assert len(phase10.PHASE_10_CLOSING["carried_forward_open"]) == 6

    # Phase 11's own two carriers are updated.
    assert "no longer blocked" in phase11.PRESTEP_CLOSED["unblocked"]
    assert "ANSWERED" in phase11.PHASE_11_SCOPE_AND_SILENCES[
        "still_open_resolved"
    ]


def test_the_degeneracy_pull_is_registered_before_any_arm_exists():
    """The same measurement arriving after a loss arm scored well would
    be an excuse."""
    record = phase11.DEGENERACY_PULL_REGISTERED
    assert record["registered"].endswith("before any arm is built")
    assert "would be an excuse" in record["why_now"]
    assert "1.685x" in record["already_measured"]
    assert "PCC 0.024" in record["already_measured"]

    # The gradient-level pull, checked as arithmetic rather than quoted.
    gradient = record["the_gradient_pull"]
    crossover = gradient["gradient_crossover"]
    assert np.isclose(crossover, (0.696 / 1.344) ** 4, atol=1e-4)
    pess = lambda d: 0.696 * d ** -0.13
    opt = lambda d: 1.344 * d ** 0.12
    assert np.isclose(pess(crossover), opt(crossover), rtol=1e-3)
    # Below it the pessimistic branch pushes harder -- toward optimism.
    assert pess(0.01) > opt(0.01)
    assert np.isclose(pess(0.01), 1.2665, atol=1e-3)
    assert np.isclose(opt(0.01), 0.7734, atol=1e-3)
    # Above it the ordering reverses, as the value-level crossover does.
    assert opt(1.0) > pess(1.0)
    assert "same defect seen from the scoring and the training side" in (
        gradient["same_defect_twice"]
    )
    # The value-level and gradient-level crossovers are different numbers.
    assert crossover < phase11.IEM_CROSSOVER

    # The reading is committed now, in the strongest form.
    reading = record["reading_committed"]
    assert "DEGENERATE OPTIMUM REALISED, NOT A RESULT" in reading
    assert "METRIC's behaviour, not the model's" in reading
    assert "may not be quoted as an improvement, ledgered" in reading


def test_the_diagnostic_is_per_epoch_with_a_trigger_and_a_stated_limit():
    record = phase11.DEGENERACY_PULL_REGISTERED["diagnostic"]
    per_epoch = record["per_epoch_not_per_seed"]
    assert "no extra forward" in per_epoch
    assert "did the loss PULL it there?" in per_epoch
    assert "TRAJECTORY" in per_epoch

    trigger = record["joint_trigger"]
    # All three conditions, same fold, and a count rather than an eyeball.
    assert "span DECREASES" in trigger
    assert "below-crossover share INCREASES" in trigger
    assert "PCC DECREASES" in trigger
    assert "all three, same fold" in trigger
    assert "registered before the run rather than read off it" in trigger

    # What it cannot do is stated, not glossed.
    limit = record["what_it_cannot_do_alone"]
    assert "MATCHED CONTROL ARM trained on MSE" in limit
    assert "build silence" in limit
    # And the maintainer's expectation is credited, with what changed.
    assert "right in content" in record["the_expectation"]


def test_the_build_scope_quotes_the_amendment_and_lists_its_silences():
    record = phase11.PHASE_11_BUILD_SCOPE
    commits = record["commits"]
    assert commits["what_changes"] == "the loss function"
    assert "rather than a re-scoring of existing ones" in commits[
        "its_own_arms"
    ]
    assert "the pre-step WAS the re-scoring" in commits["its_own_arms"]
    # What the amendment does NOT say is recorded too.
    assert "names no backbone, cell, comparison" in commits["and_nothing_else"]

    assert len(record["exists_to_build_from"]) == 6
    assert any("frozen harness" in item for item in record["exists_to_build_from"])
    assert any(
        "comparison POPULATION" in item
        for item in record["exists_to_build_from"]
    )

    silences = record["silences"]
    assert len(silences) == 9
    joined = " ".join(silences)
    assert "EXIT CRITERIA for the build -- none exist" in joined
    assert "WHICH TARGET" in joined and "median GRADE" in joined
    assert "MATCHED MSE CONTROL ARM" in joined
    assert "0.696|d|^-0.13" in joined
    assert "marking its own homework" in joined
    assert "CROSSOVER DEFECT" in joined
    # Nothing is filled: each silence names the choice, not the answer.
    assert "neither is chosen here" in joined

    # And what is NOT a silence any more is separated out.
    assert any(
        "answered, convention A" in item for item in record["not_a_silence"]
    )


# --------------------------------------------------------------------------
# 2026-08-17, the loss arm and its matched control: registered and built
# --------------------------------------------------------------------------


def test_the_build_registration_closes_every_silence_it_was_given():
    record = phase11.PHASE_11_BUILD_REGISTERED
    assert record["registered"].endswith("before the arm exists")
    assert "ONE" in record["cell"] and "0.2520" in record["cell"]
    assert "ladder of twins answers nothing extra" in record["cell"]
    assert "MEDIAN GRADE" in record["target"]
    assert "confound the arm" in record["target"]
    assert "PANEL MEAN" in record["evaluation"]
    # The control is part of the arm, and the reason is the reading.
    control = record["control"]
    assert "PART OF THE ARM, not an optional extra" in control
    assert "indistinguishable" in control
    # Both readings, always.
    assert "marking its own homework" in record["reporting"]
    assert "hides whether the loss did what it was asked" in record["reporting"]
    # Claimability is split, with the reason for the split.
    claim = record["claimability"]
    assert "paired BCa" in claim["iem_vs_mse"]
    assert claim["iem_vs_the_ladder"].startswith("DESCRIPTIVE only")
    assert "not a like-for-like ladder entry" in claim["iem_vs_the_ladder"]
    # The corrected variant is a future arm, never a mid-phase repair.
    assert "FUTURE ARM requiring its own registration" in record[
        "crossover_stays_in"
    ]
    assert "MONITOR_ARM_REGISTRABILITY" in record["crossover_stays_in"]


def test_the_loss_is_verbatim_and_its_one_departure_is_sized():
    record = phase11.IEM_LOSS_VERBATIM
    assert "NOT smoothed" in record["form"]
    assert "metric the group has not published" in record["why_not_smoothed"]
    assert len(record["defects_carried"]) == 3
    departure = record["the_one_departure"]
    assert "1e-8" in departure["what"] and "INSIDE THE POWER ONLY" in departure["what"]
    assert "8.7e-8" in departure["measured_effect"]
    assert "7.63" in departure["measured_effect"]
    # Stated as a departure rather than defended as neutral.
    assert departure["stated_not_defended"].startswith("it IS a departure")
    assert "not alternatives" in departure["stated_not_defended"]

    # The measured effect is checked, not quoted.
    floor = phase11.IEM_GRADIENT_FLOOR
    assert floor == 1e-8
    assert np.isclose(0.8 * floor ** 0.87, 8.7e-8, rtol=0.05)
    assert np.isclose(0.696 * floor ** -0.13, 7.63, rtol=0.01)


def test_the_loss_matches_the_metric_and_survives_exact_zero():
    pytest.importorskip("torch")
    import torch

    # It IS equation (16): the loss equals the metric's mean, elementwise.
    residual = np.array([-2.0, -0.5, -0.05, 0.05, 0.5, 2.0])
    prediction = torch.tensor(residual, dtype=torch.float64)
    target = torch.zeros(len(residual), dtype=torch.float64)
    assert np.isclose(
        float(phase11.iem_loss(prediction, target)),
        float(np.mean(phase11.iem(residual))),
    )
    # The swap is available and is the mirror, as in the metric.
    assert np.isclose(
        float(phase11.iem_loss(prediction, target, convention="b_swapped")),
        float(np.mean(phase11.iem(residual, "b_swapped"))),
    )
    with pytest.raises(phase11.Phase11Error):
        phase11.iem_loss(prediction, target, convention="c")

    # Exact zero: finite gradient, near-zero value. Without the floor this
    # is nan and the run dies.
    y = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    loss = phase11.iem_loss(y, torch.tensor([2.0, 3.0, 4.0]))
    loss.backward()
    assert bool(torch.isfinite(y.grad).all())
    assert float(loss.detach()) < 1e-6

    # The heavier branch really is on the flattering side.
    y2 = torch.tensor([2.5, 3.5], requires_grad=True)
    phase11.iem_loss(y2, torch.tensor([3.0, 3.0])).backward()
    assert abs(float(y2.grad[0])) > abs(float(y2.grad[1]))


def test_the_head_swaps_the_loss_and_nothing_else():
    pytest.importorskip("torch")

    from cleft.train.torch_backbone import EmbeddingHeadBackbone

    head = phase11.make_head(
        loss="iem", learning_rate=0.001, weight_decay=0.01, seed=1337
    )
    # It IS the ladder's arm path, subclassed -- not a reimplementation.
    assert isinstance(head, EmbeddingHeadBackbone)
    assert head.learning_rate == 0.001 and head.weight_decay == 0.01
    assert phase11.make_head(
        loss="mse", learning_rate=0.001, weight_decay=0.01, seed=1337
    ).loss == "mse"
    with pytest.raises(phase11.Phase11Error, match="is not one of"):
        phase11.make_head(
            loss="huber", learning_rate=0.001, weight_decay=0.01, seed=1
        )

    # The recorder captures epoch 0 -- gate 3's own prediction -- which is
    # what the joint trigger measures from.
    labels = np.array([2.0, 3.0, 3.0, 4.0, 3.0, 2.0], dtype=float)
    features = np.eye(6, 4, dtype=np.float32)
    head.reset(labels)
    head.predict(features)
    assert 0 in head.by_epoch
    head.train_epoch(features, labels)
    head.predict(features)
    assert sorted(head.by_epoch) == [0, 1]
    # A second predict in the same epoch does not overwrite the first.
    first = head.by_epoch[1].copy()
    head.predict(features * 2)
    assert np.array_equal(head.by_epoch[1], first)


def test_the_degeneracy_signature_needs_all_three_moves():
    inner = np.array([2.0, 3.0, 3.0, 4.0], dtype=float)
    # Epoch 0: a constant head, as gate 3 requires. Selected epoch: wide
    # and well-correlated -- the opposite of degenerate.
    good = {0: np.full(4, 3.0), 5: np.array([2.1, 2.9, 3.1, 3.9])}
    assert phase11.degeneracy_signature(good, inner, 5)["fired"] is False

    # Degenerate: span narrows, share rises, PCC falls.
    start = np.array([1.5, 2.6, 3.4, 4.5])
    end = np.array([2.85, 2.95, 2.9, 2.8])
    signature = phase11.degeneracy_signature({0: start, 7: end}, inner, 7)
    assert signature["moves"]["span_narrowed"] is True
    assert signature["moves"]["share_rose"] is True
    assert signature["fired"] is True
    # Every move is a plain bool, so the record serialises cleanly.
    assert all(isinstance(v, bool) for v in signature["moves"].values())

    # A missing epoch is said, not guessed.
    absent = phase11.degeneracy_signature({0: start}, inner, 9)
    assert absent["fired"] is None and "not recorded" in absent["reason"]


def test_the_degeneracy_rule_fixes_two_thresholds_before_any_run():
    record = phase11.DEGENERACY_APPLICATION_RULE
    assert record["registered"].endswith("thresholds before the run")
    assert "ALL THREE" in record["per_fold_signature"]
    assert "epoch 0" in record["per_fold_signature"]
    conditions = record["conditions"]
    assert "13 of 25" in conditions["1_majority"]
    assert "at least 5 folds" in conditions["2_gap_over_control"]
    # Condition 2 is named as the one that does the work, with why.
    assert "hedges under any loss" in record["condition_2_does_the_work"]
    assert "MATCHED control" in record["condition_2_does_the_work"]
    # Three outcomes, each with a consequence fixed in advance.
    outcomes = record["outcomes"]
    assert set(outcomes) == {"both_hold", "one_only", "neither"}
    assert "not quotable as an improvement" in outcomes["both_hold"]
    assert "about the CELL" in outcomes["one_only"]
    assert "reportable negative" in outcomes["neither"]
    # And the limitation is stated rather than glossed.
    limit = record["stated_limitation"]
    assert "TWO POINTS" in limit
    assert "cannot see a degeneracy present at initialisation" in limit


def test_the_build_exit_criteria_are_written_before_the_task_existed():
    record = phase11.PHASE_11_BUILD_EXIT_CRITERIA
    assert record["registered"].endswith("before work starts")
    criteria = record["criteria"]
    assert len(criteria) == record["expected_count"] == 6
    assert [c.split(".")[0] for c in criteria] == [str(i) for i in range(1, 7)]
    joined = " ".join(criteria)
    assert "joint trigger" in joined
    assert "differing only in the loss" in joined
    assert "PCC and on IEM" in joined
    assert "BY RULE" in joined
    assert "crossover and gradient defects" in joined
    # An IEM improvement is not a result until the rule has been applied.
    assert any("not a result until" in item for item in record["not_criteria"])


def test_the_arm_and_its_control_differ_in_the_loss_and_nothing_else():
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    arm = yaml.safe_load(
        (repo / "configs" / "p11_iem_arm.yaml").read_text(encoding="utf-8")
    )
    control = yaml.safe_load(
        (repo / "configs" / "p11_mse_control.yaml").read_text(encoding="utf-8")
    )
    # Matched by construction: identical inputs, one task key different.
    assert arm["inputs"] == control["inputs"]
    differing = {
        key for key in set(arm["task"]) | set(control["task"])
        if arm["task"].get(key) != control["task"].get(key)
    }
    assert differing == {"loss"}
    assert arm["task"]["loss"] == "iem" and control["task"]["loss"] == "mse"

    # The cell is the 0.2520 arm's, copied rather than retyped.
    anchor = yaml.safe_load(
        (repo / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )
    for key in ("geometry", "label", "seeds", "inner_val_frac", "max_epochs",
                "patience", "monitor", "learning_rate", "weight_decay"):
        assert arm["task"][key] == anchor["task"][key], key
    anchor_inputs = {e["name"]: e for e in anchor["inputs"]}
    for name in ("manifest_v1", "staged_v1", "embeddings"):
        assert any(e == anchor_inputs[name] for e in arm["inputs"]), name

    # Nothing new is declared.
    placeholder = "0" * 64
    assert len(arm["inputs"]) == 4
    for entry in arm["inputs"]:
        assert entry["rollup_sha256"] != placeholder
        assert len(entry["rollup_sha256"]) == 64

    # Each header carries what its own role requires.
    arm_header = (repo / "configs" / "p11_iem_arm.yaml").read_text(
        encoding="utf-8"
    )
    control_header = (repo / "configs" / "p11_mse_control.yaml").read_text(
        encoding="utf-8"
    )
    assert "VERBATIM, NOT SMOOTHED" in arm_header
    assert "0.19753" in arm_header and "0.0719" in arm_header
    assert "13 of 25" in arm_header
    assert "NOT AN OPTIONAL EXTRA" in control_header
    assert "condition that does the work" in control_header
    for header in (arm_header, control_header):
        assert "MEDIAN GRADE" in header and "PANEL MEAN" in header
        assert "NOTHING NEW IS DECLARED" in header


def test_the_generator_renders_both_from_one_renderer_and_does_not_drift():
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "generate_phase11_configs",
        repo / "scripts" / "generate_phase11_configs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.CONFIGS) == {
        "p11_iem_prestep.yaml", "p11_iem_arm.yaml", "p11_mse_control.yaml",
        "p11_paired.yaml",
    }
    # ONE renderer for both, so they cannot drift apart in anything but
    # the loss -- a fact by construction rather than a claim to check.
    text = (repo / "scripts" / "generate_phase11_configs.py").read_text(
        encoding="utf-8"
    )
    assert text.count("def render_arm(") == 1
    assert 'render_arm("iem")' in text and 'render_arm("mse")' in text
    assert module.main(["--check"]) == 0


def test_the_task_trains_on_the_grade_and_reports_both_readings():
    import inspect

    from cleft import run as run_module

    assert "iem_arm" in run_module.TASKS
    source = inspect.getsource(run_module.task_iem_arm)
    # One code path, the loss the only difference.
    assert 'loss = str(task["loss"])' in source
    assert "phase11.LOSSES" in source
    # Target the grade, evaluate on the mean, report both.
    assert "median_by_patient(by_id, sheet_path, patient_ids)" in source
    assert "labels=grades" in source
    assert "pcc_vs_panel_mean" in source and "iem_vs_median_grade" in source
    # The diagnostic rides the harness's own predictions.
    assert "head.by_epoch" in source
    assert "degeneracy_signature" in source
    assert "degeneracy_curve.csv" in source
    # The defects travel with the numbers, in the artifact itself.
    assert "defects_carried_with_every_number" in source
    # Condition 2 needs the other arm, and the run says so rather than
    # pretending one arm can answer it.
    assert "needs the matched control's count" in source
    # Seed CSVs land in the baseline layout so the paired BCa can consume
    # them; no ledger row is written here.
    assert "write_predictions(" in source
    assert "results_ledger" not in source


# --------------------------------------------------------------------------
# 2026-08-17, both arms run: the registered negative fires
# --------------------------------------------------------------------------


def test_the_registered_negative_fired_and_the_loss_is_judged_on_merits():
    record = phase11.LOSS_ARMS_OBSERVED
    assert "27a18e04" in record["observed"]
    degeneracy = record["degeneracy"]
    assert degeneracy["fired"].startswith("0 of 25")
    assert "NOT MET" in degeneracy["condition_1"]
    assert "MOOT" in degeneracy["condition_2"]
    # The third registered outcome, applied verbatim rather than reworded.
    applied = degeneracy["registered_outcome_applied"]
    assert "did not materialise in TRAINING" in applied
    assert "judged on its merits" in applied
    assert "reportable negative" in (
        phase11.DEGENERACY_APPLICATION_RULE["outcomes"]["neither"]
    )

    # The merits are checked against the per-seed figures, not quoted.
    per_seed = record["per_seed"]
    for metric in ("pcc", "iem"):
        arm = np.array(per_seed[f"iem_arm_{metric}"])
        control = np.array(per_seed[f"mse_control_{metric}"])
        assert len(arm) == len(control) == 5
        # Ten of ten in the expected direction, ranges disjoint.
        assert bool(np.all(arm < control))
        assert arm.max() < control.min()
        cell = record["merits"][metric]
        assert np.isclose(cell["arm"], arm.mean(), atol=1e-4)
        assert np.isclose(cell["control"], control.mean(), atol=1e-4)
        assert np.isclose(cell["delta"], arm.mean() - control.mean(), atol=1e-4)

    # The loss improves what it optimises and costs correlation.
    assert record["merits"]["iem"]["delta"] < 0     # lower IEM is better
    assert record["merits"]["pcc"]["delta"] < 0     # lower PCC is worse
    assert "NOT FREE" in record["merits"]["the_sentence"]
    assert "cost is CORRELATION" in record["merits"]["the_sentence"]

    # The incidental is what makes the attribution clean.
    incidental = record["the_incidental"]
    assert "0.2552" in incidental and "0.2520" in incidental
    assert "inside that arm's own seed sd" in incidental
    assert "attributable to the LOSS and not to the target" in incidental
    assert "two changes and one number" in incidental
    # And it holds arithmetically: the gap is inside the arm's seed sd.
    assert abs(np.mean(per_seed["mse_control_pcc"]) - 0.2520) < 0.0148

    assert "verbatim with them in" in record["defects_still_travel"]


def test_the_loss_contrast_pairs_on_two_metrics_and_derives_both():
    from cleft import ladder

    pairs = phase11.paired_claim_pairs("p11_loss")
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["a"] == phase11.PAIRED_CONTROL_STEM == "p11_mse_control"
    assert pair["b"] == phase11.PAIRED_ARM_STEM == "p11_iem_arm"
    assert pair["seeds"] == list(ladder.SEED_POOL[:5])

    recorded = pair["recorded"]
    assert set(recorded) == {"pcc", "iem"}
    # Derived from the observation, not typed.
    assert recorded["pcc"]["delta_of_means"] == -0.0697
    assert recorded["iem"]["delta_of_means"] == -0.0305
    assert recorded["pcc"]["margin"] == 2.57
    assert recorded["iem"]["margin"] == 2.68
    # The direction is recorded per metric, because they run opposite ways.
    assert recorded["iem"]["lower_is_better"] is True
    assert recorded["pcc"]["lower_is_better"] is False
    # Both margins sit just above the line below which nothing has passed.
    from cleft import roadb

    assert "2.55" in roadb.CONDITION_1_MARGIN_STRUCTURE["floor"]
    assert all(recorded[m]["margin"] > 2.55 for m in ("pcc", "iem"))
    assert all(recorded[m]["margin"] < 3.6 for m in ("pcc", "iem"))

    # The ladder derivations take these pairs unchanged.
    assert len(ladder.paired_claim_vectors("p11_loss", pairs=pairs)) == 10
    assert len(ladder.paired_claim_stems("p11_loss", pairs=pairs)) == 2
    with pytest.raises(phase11.Phase11Error, match="unknown Phase 11"):
        phase11.paired_claim_pairs("p11_other")

    # The coverage record refuses the ladder comparison, with the reason.
    coverage = phase11.PAIRED_CLAIM_COVERAGE
    assert "differing in the loss and nothing else" in coverage["covers"]
    assert "marking its own homework" in coverage["on_two_metrics"]
    assert "DESCRIPTIVE only" in coverage["excluded"]["the_ladder"]
    assert "dress a non-comparison in an interval" in (
        coverage["excluded"]["the_ladder"]
    )


def test_the_paired_comparison_took_a_statistic_without_changing_pcc():
    """The frozen paired_delta_bca was already metric-generic; only
    phase7b hardcoded pcc. Generalising it is why no second paired
    implementation exists."""
    import inspect

    from cleft import phase7b
    from cleft.eval import metrics

    signature = inspect.signature(phase7b.paired_comparison)
    assert signature.parameters["statistic"].default is None
    assert signature.parameters["statistic_name"].default == "pcc"
    # The frozen primitive already took a statistic.
    assert "statistic" in inspect.signature(metrics.paired_delta_bca).parameters

    # BCa needs n >= 10.
    truth = [1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 4.0, 3.0, 2.0, 4.0, 3.0]
    a = {7: [1.2, 2.1, 2.8, 4.2, 4.7, 3.1, 2.2, 3.8, 2.9, 2.3, 3.9, 3.2]}
    b = {7: [2.0, 2.0, 3.5, 3.5, 4.0, 3.0, 2.5, 3.5, 3.1, 2.4, 3.4, 2.8]}
    default = phase7b.paired_comparison(
        truth=truth, winner_by_seed=a, baseline_by_seed=b, winner_sd=0.01,
        n_boot=200,
    )
    # Default behaviour is unchanged, keys and all.
    assert default["statistic"] == "pcc"
    assert "winner_pcc" in default["per_seed"][0]
    assert "baseline_pcc" in default["per_seed"][0]
    explicit = phase7b.paired_comparison(
        truth=truth, winner_by_seed=a, baseline_by_seed=b, winner_sd=0.01,
        n_boot=200, statistic=metrics.pcc, statistic_name="pcc",
    )
    assert explicit["per_seed"][0]["delta"] == default["per_seed"][0]["delta"]

    # And it carries an error metric, keyed by its own name.
    iem_stat = lambda g, p: float(
        np.mean(phase11.iem(np.asarray(p) - np.asarray(g)))
    )
    other = phase7b.paired_comparison(
        truth=truth, winner_by_seed=a, baseline_by_seed=b, winner_sd=0.01,
        n_boot=200, statistic=iem_stat, statistic_name="iem",
    )
    assert other["statistic"] == "iem"
    assert "winner_iem" in other["per_seed"][0]
    assert "winner_pcc" not in other["per_seed"][0]
    # The two-condition machinery is sign-agnostic and still works.
    assert isinstance(other["claimable"], bool)


def test_the_paired_config_is_two_pass_and_declares_the_grade_source():
    from pathlib import Path

    import yaml

    repo = Path(__file__).resolve().parents[1]
    path = repo / "configs" / "p11_paired.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    task = payload["task"]
    assert task["kind"] == "paired_claims"
    assert task["scope"] == "p11_loss"
    # The grade source is declared HERE and nowhere else in a paired config.
    assert task["manifest_artifact"] == "manifest_v1"
    assert task["scoresheet_artifact"] == "scoresheet_primary"

    inputs = payload["inputs"]
    assert len(inputs) == 12
    vectors = [e for e in inputs if e["name"].startswith("oof_")]
    assert len(vectors) == 10
    assert {e["name"].split("_seed_")[0] for e in vectors} == {
        "oof_p11_iem_arm", "oof_p11_mse_control",
    }
    placeholder = "0" * 64
    # [2026-08-17, BOTH PASSES DONE] Run directories pasted, then
    # declare_inputs.py verified the files: no PENDING path, no
    # placeholder hash, nothing left for guard 3 to refuse.
    assert all("/PENDING_" not in e["path"] for e in inputs)

    from cleft.run_names import check_run_dir

    for entry in inputs:
        assert entry["rollup_sha256"] != placeholder
        assert len(entry["rollup_sha256"]) == 64
        assert set(entry["rollup_sha256"]) <= set("0123456789abcdef")
    for entry in vectors:
        # Every vector's path is a real run directory obeying the
        # contract, and both arms carry the SAME config sha -- they were
        # generated together and differ only in the loss.
        directory = entry["path"].split("/")[-2]
        check_run_dir(directory)
        assert directory.split("__")[1] == "27a18e04"
    # Twelve inputs, twelve distinct files: a repeated hash would mean
    # two seeds -- or the two ARMS -- wrote identical predictions, which
    # for a matched pair differing in the loss is exactly the corruption
    # that would still produce ordinary-looking intervals.
    assert len({e["rollup_sha256"] for e in inputs}) == 12

    header = path.read_text(encoding="utf-8")
    assert "**RESOLVED.**" in header
    assert "PLACEHOLDER" not in header.upper()
    assert "PENDING" not in header
    assert "FITS NOTHING" in header
    assert "LIKE-FOR-LIKE" in header
    assert "stays DESCRIPTIVE" in header
    assert "TWO METRICS" in header
    assert "-0.0697" in header and "-0.0305" in header
    assert "2.57x" in header and "2.68x" in header
    assert "2.55x line" in header
    assert "0.19753" in header and "0.0719" in header


def test_the_second_metric_is_an_added_key_not_a_changed_shape():
    import inspect

    from cleft import run as run_module
    from cleft.config.schema import TASK_SPECS

    assert "p11_loss" in TASK_SPECS["paired_claims"]["scope"].choices
    # The grade source is optional, so no existing paired config changes.
    assert TASK_SPECS["paired_claims"]["manifest_artifact"].required is False
    assert TASK_SPECS["paired_claims"]["scoresheet_artifact"].required is False

    source = inspect.getsource(run_module.task_paired_claims)
    assert 'else phase11 if scope == "p11_loss"' in source
    # **ONE call site, inside a loop over the enumeration's metrics** --
    # not a second call site inside an `if scope ==`, which is what
    # roadb's guard forbids and which this first got wrong.
    assert source.count("phase7b.paired_comparison") == 1
    assert 'for spec in pair.get("metrics"' in source
    assert 'if scope == "p11_loss":' not in source
    # The metric is data in the pair, resolved by a shared helper.
    assert "paired_statistic(" in source
    assert '"by_metric": by_metric' in source
    # The defects ride into the artifact rather than being remembered.
    assert "defects_carried" in source

    # The default keeps every other scope on the single PCC it always ran.
    assert run_module.DEFAULT_PAIRED_METRIC == {
        "name": "pcc", "truth": "panel_mean",
    }
    resolver = inspect.getsource(run_module.paired_statistic)
    assert "median_by_patient(" in resolver
    assert "their truth column is the panel mean" in resolver
    assert "panel_mean" in resolver and "median_grade" in resolver
    # The p11 pair names both metrics; nobody else names any.
    from cleft import ladder, roadb

    assert "metrics" in phase11.paired_claim_pairs("p11_loss")[0]
    assert "metrics" not in ladder.paired_claim_pairs("headline")[0]
    assert "metrics" not in roadb.paired_claim_pairs("roadb_resolution")[0]


def test_the_pre_step_closes_on_seven_met_criteria_and_claims_nothing():
    record = phase11.PRESTEP_CLOSED
    assert record["closed"] == "2026-08-17"
    criteria = record["criteria"]
    assert len(criteria) == 7
    assert all(v.startswith("MET") for v in criteria.values())
    assert "finding COMPLETE" in criteria["5_domain"]
    assert "0.2142-0.9707" in criteria["5_domain"]

    # The amendment's question, answered with the confound separated.
    answered = record["what_it_answered"]
    assert "0.232" in answered and "0.845" in answered
    assert "not the asymmetry" in answered
    assert "would have arrived as one figure" in answered

    # And the result nobody asked for, which is the bigger one.
    unasked = record["what_it_answered_unasked"]
    assert "NEAR-CONSTANT PREDICTOR FIRST" in unasked
    assert "PCC 0.024" in unasked
    assert "about the METRIC" in unasked

    assert "no ledger row" in record["nothing_is_claimed"]
    assert len(record["carried_forward"]) == 3
    assert any("BLOCKED" in item for item in record["carried_forward"])
    assert "Phase 12" in record["next"]

    # The walk that preceded it is resolved in place, with its reason.
    resolved = phase11.PRESTEP_EXIT_WALK["resolved"]
    assert "both inverted a prediction" in resolved
    assert "Closing first would have filed" in resolved

    # Nothing from the PRE-STEP reached the ledger. [2026-08-17] Phase 11
    # does have rows now -- the two loss-contrast withdrawals -- so the
    # check is that none of them rests on the pre-step: a ranking has no
    # interval and was DESCRIPTIVE by registration.
    from cleft import results_ledger

    assert any(e["phase"] == "p11" for e in results_ledger.ENTRIES)
    for entry in results_ledger.ENTRIES:
        assert "PRE_STEP" not in entry["record"]
        assert "PRESTEP" not in entry["record"]
