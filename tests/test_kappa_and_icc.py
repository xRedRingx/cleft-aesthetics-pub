"""The kappa limitation, the ICCs and the ceiling range.

All three are record corrections and a reporting form: no ledger row, no
phase, and no existing figure withdrawn.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from cleft import record_audit
from cleft.data import reliability as R

REPO = pathlib.Path(__file__).resolve().parents[1]

#: The ten sites that quote THIS project's kappa.
KAPPA_SITES = (
    "src/cleft/ladder.py",
    "src/cleft/phase12.py",
    "src/cleft/phase22.py",
    "src/cleft/phase24.py",
    "src/cleft/phase9.py",
    "src/cleft/data/reliability.py",
    "src/cleft/data/report.py",
    "src/cleft/train/ldl.py",
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_the_kappa_limitation_states_the_unweighted_property():
    record = record_audit.THE_KAPPA_LIMITATION
    limitation = _flat(record["the_limitation"])
    assert "UNWEIGHTED" in limitation
    assert "1-versus-2 disagreement as identically wrong to a 1-versus-5" \
        in limitation
    assert "wrong statistic for these data" in limitation

    beside = _flat(record["the_distance_aware_figures_beside_it"])
    assert "QWK_237 = 0.4276" in beside
    assert "MEAN_R_237 = 0.4696" in beside
    assert "(i - j)^2 / (n - 1)^2" in beside
    # Both really are banked, and QWK really is quadratic.
    assert R.QWK_237 == 0.4276 and R.MEAN_R_237 == 0.4696
    source = (REPO / "src" / "cleft" / "data" / "reliability.py").read_text(
        encoding="utf-8"
    )
    assert "weights = (grid[:, None] - grid[None, :]) ** 2 / (n - 1) ** 2" in source


def test_no_result_changes_because_kappa_gates_nothing():
    """The claim that makes this a hygiene item and not a correction."""
    record = _flat(record_audit.THE_KAPPA_LIMITATION["NO_RESULT_CHANGES_and_this_is_verified"])
    assert "gates nothing and enters no arithmetic" in record
    assert "CORRECTED 2026-09-03" in record
    assert "Kappa still gates nothing" in record

    # Verified, not asserted: run.py never mentions it.
    # [UPDATED 2026-09-03] run.py DOES reference Fleiss now -- task_rater_icc
    # recomputes it as a cross-check. What the pin holds is the claim that
    # matters: the only use is verification, never a gate.
    run_body = (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    fleiss_lines = [
        line for line in run_body.splitlines() if "fleiss" in line.lower()
    ]
    assert fleiss_lines, "the cross-check should reference it"
    for line in fleiss_lines:
        assert "task_rater_icc" in run_body
        assert not line.lstrip().startswith("if "), line
    # And FLEISS_237 feeds nothing but the by-population dict.
    reliability_src = (
        REPO / "src" / "cleft" / "data" / "reliability.py"
    ).read_text(encoding="utf-8")
    uses = [
        line for line in reliability_src.splitlines()
        if "FLEISS_237" in line and not line.lstrip().startswith("#")
    ]
    assert len(uses) == 2, uses          # the definition and the dict


def test_all_ten_sites_carry_a_pointer_and_none_was_rewritten():
    total = 0
    for relative in KAPPA_SITES:
        body = (REPO / relative).read_text(encoding="utf-8")
        count = body.count("record_audit.THE_KAPPA_LIMITATION")
        assert count >= 1, relative
        total += count
    assert total == 10, total

    # **Nothing was rewritten**: the original sentences are still there.
    assert "Fleiss kappa 0.1662, QWK 0.4276" in (
        REPO / "src" / "cleft" / "ladder.py"
    ).read_text(encoding="utf-8")
    assert "the research problem, not a" in (
        REPO / "src" / "cleft" / "data" / "reliability.py"
    ).read_text(encoding="utf-8")
    assert "Fleiss 0.1662 behind it" in (
        REPO / "src" / "cleft" / "phase22.py"
    ).read_text(encoding="utf-8")


def test_the_two_motivating_sites_are_named():
    where = _flat(record_audit.THE_KAPPA_LIMITATION["where_it_nonetheless_bit"])
    assert "phase12" in where and "train/ldl.py" in where
    assert "Only ``ladder.py:1528`` pairs it with QWK" in where
    assert "overstates the disagreement" in where
    assert "two-view" in _flat(
        record_audit.THE_KAPPA_LIMITATION["the_shape_it_shares_with_the_two_view_claim"]
    )


# --------------------------------------------------------------------------
# the ICCs
# --------------------------------------------------------------------------


def test_the_anova_reproduces_the_published_worked_example():
    """Shrout & Fleiss (1979) Table 1: ICC(2,1) 0.290, ICC(3,1) 0.715,
    ICC(2,k) 0.620, ICC(3,k) 0.909."""
    matrix = np.array([
        [9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
        [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7],
    ])
    got = R._icc_two_way_anova(matrix)
    assert got["icc_2_1"] == pytest.approx(0.290, abs=5e-4)
    assert got["icc_3_1"] == pytest.approx(0.715, abs=5e-4)
    assert got["icc_2_k"] == pytest.approx(0.620, abs=5e-4)
    assert got["icc_3_k"] == pytest.approx(0.909, abs=5e-4)
    # The public entry keeps the 1-5 guard, which is why the split exists.
    with pytest.raises(R.ReliabilityError, match="grades outside"):
        R.icc_two_way(matrix)


def test_icc_3k_is_never_below_icc_2k_and_offset_is_what_separates_them():
    """The prediction's mechanism, demonstrated rather than argued."""
    rng = np.random.default_rng(11)
    truth = rng.integers(1, 6, size=60)

    # No offset: five raters agreeing up to noise.
    balanced = np.clip(
        np.column_stack([truth + rng.integers(-1, 2, 60) for _ in range(5)]),
        1, 5,
    )
    a = R.icc_two_way(balanced)
    # **NOT >= in general**: with no offset the two are equal up to
    # sampling noise and either can come out ahead. This is the case that
    # found the always-claim wrong.
    assert a["icc_2_k"] == pytest.approx(a["icc_3_k"], abs=0.05)

    # One systematically harsh rater: r is blind to it, ICC(2,k) is not.
    harsh = balanced.copy()
    harsh[:, 0] = np.clip(harsh[:, 0] + 2, 1, 5)
    b = R.icc_two_way(harsh)
    assert b["icc_3_k"] > b["icc_2_k"]
    assert (b["icc_3_k"] - b["icc_2_k"]) > (a["icc_3_k"] - a["icc_2_k"])
    assert b["between_rater_ms"] > a["between_rater_ms"]


def test_the_icc_prediction_is_registered_before_the_number():
    record = record_audit.ICC_PREDICTION_REGISTERED
    assert record["tag"].startswith("[REGISTERED]")
    blind = _flat(record["the_two_quantities_measure_different_things"])
    assert "BLIND TO RATER BIAS" in blind
    assert "contributes FULLY to r and NOTHING to absolute agreement" in blind
    prediction = _flat(record["the_prediction"])
    assert "ICC(2,k) <= 0.8158 is expected" in prediction
    assert "THE GAP MEASURES SYSTEMATIC RATER OFFSET" in prediction

    offset = _flat(record["nothing_in_the_record_measures_offset_today"])
    assert "item_total" in offset and "offset-invariant" in offset
    assert "banked reliability stands" in _flat(record["reading_a_close_agreement"])
    assert "ceiling drops with it" in _flat(record["reading_a_materially_lower_ICC"])
    # No threshold is invented to grade the result afterwards.
    assert "not pre-specified, deliberately" in _flat(
        record["what_counts_as_materially_lower"]
    )


def test_the_icc_form_distinction_is_reported_and_not_chosen():
    record = record_audit.ICC_FORM_DISTINCTION
    assert record["tag"].startswith("[REPORTED]")
    assert "SAMPLE of possible raters" in _flat(
        record["icc_2k_raters_as_a_RANDOM_effect"]
    )
    assert "population of interest" in _flat(
        record["icc_3k_raters_as_a_FIXED_effect"]
    )
    # The corrected ordering claim, and why it was wrong.
    fixed = _flat(record["icc_3k_raters_as_a_FIXED_effect"])
    assert "CORRECTED 2026-09-03" in fixed
    assert "exactly when MSC >= MSE" in fixed
    assert "REVERSES by sampling noise" in fixed
    leaning = _flat(record["what_the_record_leans_toward_and_why_that_is_NOT_a_choice"])
    assert "not a ruling" in leaning
    assert "the choice is the maintainer" in leaning
    relation = _flat(record["the_relation_to_spearman_brown"])
    assert "NOT identical to it" in relation


def test_the_icc_task_refuses_a_matrix_that_is_not_the_banked_one():
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_rater_icc)
    # The cross-check comes BEFORE any ICC.
    assert body.index("This is not the matrix") < body.index("R.icc_two_way")
    for name in ("FLEISS_237", "QWK_237", "MEAN_R_237"):
        assert name in body, name
    assert "scoresheet_artifact" in body
    # And it says why the manifest cannot answer this.
    assert "soft_1..soft_5" in body


def test_the_icc_config_declares_both_artifacts_from_their_donors():
    import yaml

    config = yaml.safe_load(
        (REPO / "configs" / "p1b_rater_icc.yaml").read_text(encoding="utf-8")
    )
    donor = {
        entry["name"]: entry for entry in yaml.safe_load(
            (REPO / "configs" / "p10_rater_screen.yaml").read_text(
                encoding="utf-8"
            )
        )["inputs"]
    }
    ours = {entry["name"]: entry for entry in config["inputs"]}
    assert ours["manifest_v1"] == donor["manifest_v1"]
    assert ours["scoresheet_primary"] == donor["scoresheet_primary"]
    assert config["task"]["expect_patients"] == 237
    assert config["task"]["expect_raters"] == 5
    assert not any(
        set(entry["rollup_sha256"]) == {"0"} for entry in config["inputs"]
    )


# --------------------------------------------------------------------------
# the ceiling as a range
# --------------------------------------------------------------------------


def test_the_range_is_a_reporting_form_and_withdraws_nothing():
    record = record_audit.THE_CEILING_AS_A_RANGE
    assert record["tag"].startswith("[REPORTING FORM]")
    form = _flat(record["the_form"])
    assert "sqrt(ICC(2,k)), sqrt(0.8158)" in form
    assert "ordered by the values rather than by the expectation" in form

    which = _flat(record["which_derivation_each_endpoint_comes_from"])
    assert "blind to rater offset" in which
    assert "offset penalised" in which
    assert "two numbers, not an interval" in which

    assert "not a confidence interval" in _flat(record["what_it_is_NOT"])
    standing = _flat(record["the_existing_figure_is_not_withdrawn"])
    assert "0.9032 is correct for what it measures" in standing
    assert "does not exist yet" in _flat(record["it_waits_on_the_number"])

    # 0.9032 is still banked and still what sqrt(0.8158) gives.
    assert R.PCC_CEILING_237 == 0.9032
    assert R.pcc_ceiling(R.RELIABILITY_237) == pytest.approx(0.9032, abs=5e-5)
    # And the prohibition still governs both endpoints.
    from cleft import phase24

    assert "A HIGHER CEILING IS NOT A HIGHER SCORE" in \
        phase24.A_CEILING_IS_NOT_A_SCORE
    assert "A_CEILING_IS_NOT_A_SCORE" in _flat(
        record["and_the_prohibition_still_governs"]
    )


def test_none_of_this_adds_a_ledger_row():
    from cleft import results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for the ICC ruling banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not any(
        "icc" in entry["id"] or "kappa" in entry["id"]
        for entry in results_ledger.ENTRIES
    )
    results_ledger.validate()


# --------------------------------------------------------------------------
# [2026-09-03] The ruling, the reporting form, and the offset
# --------------------------------------------------------------------------

ICC_2K = 0.8069
ICC_3K = 0.8218
SPEARMAN_BROWN = 0.8158


def test_the_ruling_went_against_the_more_flattering_number():
    record = record_audit.ICC_FORM_RULED
    assert record["tag"].startswith("[RULED]")
    ground = _flat(record["the_ground"])
    assert "whether the TASK is automatable" in ground
    assert "generalise to no other panel" in ground
    assert "a much larger concession than the 0.008 it buys" in ground

    knowing = _flat(record["it_was_made_KNOWING_the_other_is_higher"])
    assert "0.8218 against ICC(2,k) 0.8069" in knowing
    assert "0.9065 against 0.8983" in knowing
    assert "went against the more flattering one" in knowing
    # ICC(3,k) really is the higher one, and by the stated amount.
    assert ICC_3K > ICC_2K
    assert round(ICC_3K - ICC_2K, 4) == 0.0149
    assert "0.0149 of reliability" in _flat(record["what_it_costs"])
    assert "is not wrong and is not withdrawn" in _flat(
        record["what_it_does_NOT_settle"]
    )


def test_the_reporting_form_names_every_derivation():
    record = record_audit.THE_PRIMARY_REPORTING_FORM
    reliability = _flat(record["reliability"])
    assert "ICC(2,k) = 0.8069 is PRIMARY" in reliability
    assert "Spearman-Brown 0.8158" in reliability
    assert "ICC(3,k) 0.8218" in reliability
    assert "blind to rater offset" in reliability
    assert "offset removed rather than penalised" in reliability

    ceiling = _flat(record["the_ceiling_is_a_RANGE"])
    assert "[0.8983, 0.9065], with 0.9032 inside it" in ceiling
    assert "Lower endpoint = sqrt(ICC(2,k))" in ceiling
    assert "Upper endpoint = sqrt(ICC(3,k))" in ceiling
    assert "lands in the middle" in ceiling

    # **All three endpoints reproduce.**
    assert round(math.sqrt(ICC_2K), 4) == 0.8983
    assert round(math.sqrt(ICC_3K), 4) == 0.9065
    assert round(math.sqrt(SPEARMAN_BROWN), 4) == 0.9032
    assert R.PCC_CEILING_237 == 0.9032
    assert 0.8983 < R.PCC_CEILING_237 < 0.9065

    assert "one of three defensible derivations" in _flat(
        record["0_9032_STANDS_and_is_not_withdrawn"]
    )
    assert "still a ceiling" in _flat(
        record["the_prohibition_governs_all_three_endpoints"]
    )


def test_the_offset_is_the_first_bias_measurement_and_the_prediction_held():
    record = record_audit.THE_RATER_OFFSET_MEASURED
    finding = _flat(record["the_finding"])
    assert "2.35x the residual" in finding
    assert "MS_R 1.1832 against MS_E 0.5028" in finding
    assert "MS_P 2.6465" in finding
    # The ratio the finding rests on reproduces.
    assert round(1.1832 / 0.5028, 2) == 2.35
    assert round(1.1832 / 2.6465, 2) == 0.45

    first = _flat(record["it_is_the_FIRST_measurement_of_bias_in_the_project"])
    assert "ORDERING, not OFFSET" in first
    assert "invisible to every one of them" in first
    assert "we measured systematic rater offset and it is small" in _flat(
        record["the_sentence_the_write_up_can_now_make"]
    )

    # The registered prediction held, in direction and in sign.
    held = _flat(record["and_the_prediction_held"])
    assert "0.8069 is below 0.8158 by 0.0089" in held
    assert round(SPEARMAN_BROWN - ICC_2K, 4) == 0.0089
    assert "registered before the number and is not amended" in held
    assert "ICC(2,k) <= 0.8158 is expected" in _flat(
        record_audit.ICC_PREDICTION_REGISTERED["the_prediction"]
    )


def test_the_mean_squares_are_flagged_as_not_reconciling():
    """A figure that fails an internal check is not banked as though it
    passed -- one cycle after the fabrication."""
    record = record_audit.THE_RATER_OFFSET_MEASURED
    flagged = _flat(record["THE_MEAN_SQUARES_DO_NOT_RECONCILE_WITH_THE_ICCS"])
    assert "[REPORTED, NOT BANKED AS CONSISTENT.]" in flagged
    assert "ICC(3,k) = 0.8100 and ICC(2,k) = 0.8091" in flagged
    assert "do not reconcile with each other" in flagged

    # The arithmetic the flag rests on, recomputed here.
    n = 237
    ms_p, ms_r, ms_e = 2.6465, 1.1832, 0.5028
    icc3k = (ms_p - ms_e) / ms_p
    icc2k = (ms_p - ms_e) / (ms_p + (ms_r - ms_e) / n)
    assert round(icc3k, 4) == 0.8100
    assert round(icc2k, 4) == 0.8091
    assert abs(icc3k - ICC_3K) > 0.01          # genuinely apart
    # ... while the ICC pair's own ceilings DO close.
    assert round(math.sqrt(ICC_2K), 4) == 0.8983

    unaffected = _flat(record["what_that_does_and_does_not_affect"])
    assert "reporting form do not depend on it" in unaffected
    assert "stands on either set" in unaffected
    assert "THE_FABRICATED_FIGURES_WITHDRAWN" in _flat(
        record["why_it_is_flagged_rather_than_resolved"]
    )


def test_the_ms_naming_hazard_is_recorded_against_the_code():
    hazard = _flat(record_audit.THE_RATER_OFFSET_MEASURED[
        "a_NAMING_hazard_worth_recording"
    ])
    assert "ms_rows`` is MS_P and ``ms_cols`` is MS_R" in hazard

    # The code really does name them that way round.
    import numpy as np

    matrix = np.array([
        [1, 1, 2, 1, 2], [2, 3, 2, 3, 2], [4, 4, 5, 4, 4],
        [5, 5, 4, 5, 5], [3, 2, 3, 3, 4], [1, 2, 1, 2, 1],
    ])
    got = R.icc_two_way(matrix)
    assert set(got) >= {"ms_rows", "ms_cols", "ms_error"}
    source = (REPO / "src" / "cleft" / "data" / "reliability.py").read_text(
        encoding="utf-8"
    )
    assert "ss_rows = k * float(np.sum((row_means - grand) ** 2))" in source
    assert "ss_cols = n * float(np.sum((col_means - grand) ** 2))" in source


def test_the_ceiling_was_never_the_binding_constraint():
    record = record_audit.THE_CEILING_WAS_NEVER_THE_BINDING_CONSTRAINT
    arithmetic = _flat(record["the_arithmetic"])
    assert "span 0.008" in arithmetic
    assert "about 0.65" in arithmetic
    assert "eighty times" in arithmetic
    # The spread and the gap, recomputed.
    spread = max(ICC_2K, SPEARMAN_BROWN, ICC_3K) - min(
        ICC_2K, SPEARMAN_BROWN, ICC_3K
    )
    assert round(spread, 4) == 0.0149
    # From the REPORTED four-decimal endpoints, which is what the
    # record quotes; at full precision it is 0.0083.
    reported = [0.8983, 0.9032, 0.9065]
    assert round(max(reported) - min(reported), 4) == 0.0082
    ceilings = [math.sqrt(v) for v in (ICC_2K, SPEARMAN_BROWN, ICC_3K)]
    assert round(max(ceilings) - min(ceilings), 4) == 0.0083
    assert round(min(ceilings) - 0.2520, 2) == 0.65

    assert "MEASURED rather than argued" in _flat(record["what_it_demonstrates"])
    correction = _flat(record["it_is_a_REPORTING_correction_not_a_sensitivity_check"])
    assert "none of them wrong" in correction
    assert "no conclusion depended on which derivation was used" in correction
    assert "does not make the ceiling irrelevant" in _flat(
        record["what_it_does_NOT_license"]
    )


def test_the_ruling_adds_no_ledger_row_and_opens_no_phase():
    """[UPDATED 2026-09-05] The proxy went stale, the invariant did not.

    This asserted that the ICC ruling opened no phase by checking that
    ``phase26.py`` does not exist. Phase 26 was opened five days later
    by ``phase25.PHASE_SEQUENCE_EXTENDED_8``, so the proxy now fails
    for a reason that has nothing to do with the ICC ruling. The
    invariant it meant is asserted directly instead: the ICC ruling
    banks no ledger row and appears in no phase's opening.
    """
    from cleft import phase25, results_ledger

    # [2026-09-06] Was `assert len(results_ledger.ENTRIES) == 38`, a
    # global count standing in for the ICC ruling banked no ledger row. It went stale when Phase 27
    # appended entry 39. Rewritten as the relation, which does not.
    assert not any(
        "icc" in entry["id"] or "kappa" in entry["id"]
        for entry in results_ledger.ENTRIES
    )
    results_ledger.validate()

    # Nothing the ICC ruling produced reached a ledger row.
    blob = " ".join(str(entry) for entry in results_ledger.ENTRIES)
    assert "ICC(2,k)" not in blob
    assert "0.8069" not in blob

    # And the phase that DOES exist was opened by the amendment, not
    # by the ruling: its motivation names calibration, never the ICC.
    entry = phase25.PHASE_SEQUENCE_EXTENDED_8[
        "phase_26_the_calibration_ablation"
    ]
    assert "ICC" not in " ".join(str(v) for v in entry.values())
