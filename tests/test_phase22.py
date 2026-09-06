"""Phase 22 -- ordering-based training, restated after the refutation.

Registration only. Nothing is built: no task, no schema kind, no config,
no run. A negative-space test holds that state, and the exit criteria are
a DRAFT rather than a lock.

Every figure the reckoning cites is checked against the record that holds
it, and every coverage claim is checked against the file it names -- so
that "nothing tests an ordering objective" fails here if an ordering arm
ever appears.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cleft import ladder, literature, phase14, phase17, phase21, phase22
from cleft.config import schema

REPO = Path(__file__).resolve().parents[1]


def _flat(text: str) -> str:
    return " ".join(text.split())


# --------------------------------------------------------------------------
# the reckoning, and the premise it leads with
# --------------------------------------------------------------------------


def test_the_reckoning_states_the_honest_premise_first():
    """A reckoning that mentions its refutation late is the same document
    with the load-bearing sentence moved somewhere it will not be read."""
    record = phase22.PHASE_22_RECKONING

    # The premise key is first in the record, after only the date.
    keys = list(record)
    assert keys[0] == "reckoned"
    assert keys[1] == "the_honest_premise", keys[:3]

    premise = _flat(record["the_honest_premise"])
    assert "PROPOSED ON A MECHANISM THAT HAS SINCE BEEN MEASURED AND RULED OUT" in premise
    assert "restated in a weaker form" in premise
    assert "moved somewhere it will not be read" in premise
    # And the module docstring says the same, so a reader who never opens
    # a record still meets it.
    doc = _flat(phase22.__doc__)
    assert "weaker form than it was proposed in" in doc
    assert "not in a limitations paragraph at the bottom" in doc


def test_every_figure_the_reckoning_cites_is_live_at_its_source():
    record = phase22.PHASE_22_RECKONING
    holds = _flat(record["what_the_record_now_holds"])

    # 63 of 64 shrink, with the shrinkage figures.
    across = phase21.CROSS_ARM_SHRINKAGE_OBSERVED["across_arms"]
    assert "63 of 64" in holds
    assert str(across["median"]) == "0.3289" and "0.3289" in holds
    assert str(across["min"]) == "0.0656" and "0.0656" in holds
    assert phase21.CROSS_ARM_SHRINKAGE_OBSERVED["probe"] == 0.4947
    assert "0.4947" in holds
    assert "2.1730" in holds and across["max"] == 2.1730

    # The two tau_b figures, from the post-hoc analyses.
    assert "-0.8470" in holds and "+0.7521" in holds
    assert "-0.8470" in phase21.POST_HOC_ANALYSES[
        "4_residual_sign_hardness_vs_signed_distance"
    ]
    assert "+0.7521" in phase21.POST_HOC_ANALYSES[
        "3_worst_quartile_hardness_vs_distance_from_the_centre"
    ]

    # Arm A's kappas and the all-pairs means.
    assert "+0.0956" in holds and "-0.0743" in holds
    assert "+0.7012" in holds and "+0.5729" in holds
    assert phase21.ARM_B_OBSERVED["residual_sign"]["mean_kappa"] == 0.7012
    assert phase21.ARM_B_OBSERVED["worst_quartile"]["mean_kappa"] == 0.5729


def test_the_reckoning_states_what_is_forbidden_and_why_twice_over():
    record = phase22.PHASE_22_RECKONING

    forbids = _flat(record["what_it_forbids"])
    assert "SCALE_INVARIANCE_PROHIBITION" in forbids
    assert "CANNOT EXPLAIN THE PCC CEILING" in forbids
    assert "Arithmetic, not evidence" in forbids
    # The prohibition is live and says the same thing.
    assert "CANNOT EXPLAIN THE PCC CEILING" in (
        phase21.SCALE_INVARIANCE_PROHIBITION
    )

    measured = _flat(record["and_now_a_measured_case_as_well"])
    assert "0.2334" in measured and "0.2520" in measured
    assert "NOT SHRINKING BUYS NOTHING ON PCC" in measured
    assert "DEMONSTRATED BY A MODEL THAT DID IT" in measured
    # Both PCC figures live at their sources, and the delta is inside
    # the band the cohort cannot resolve.
    assert phase17.ARM_MEANS["arms"]["p17_arm_a"]["pcc"] == 0.2334
    assert ladder.TRADE_OFF_PAIR["result"]["vit_paired_mean"] == 0.2520
    assert abs(0.2334 - 0.2520) < 0.04, "inside the unresolvable band"

    # Two independent routes to the same negative.
    assert "by different routes" in measured


def test_the_surviving_hypothesis_is_the_narrower_one_and_arm_a_makes_it_real():
    record = phase22.PHASE_22_RECKONING

    survives = _flat(record["the_hypothesis_that_survives"])
    assert "NARROWER" in survives
    assert "DIFFERENT FEATURES" in survives
    assert "closed at both ends" in survives

    real = _flat(record["arm_a_makes_the_distinction_real"])
    assert "AVOIDING THE MEAN" in real and "FINDING DIFFERENT FEATURES" in real
    assert "measurably separable" in real

    harder = _flat(record["and_the_weaker_hypothesis_is_the_harder_one_to_test"])
    assert "REPRESENTATIONAL" in harder
    assert "the phase's actual instrument" in harder
    assert "weaker AND whose test got harder" in harder


def test_the_assertion_is_recorded_with_its_order():
    """The reckoning is written after the refutation. Nothing in the
    module's structure shows that -- only this record does."""
    record = phase22.THE_STRONGER_MECHANISM_ASSERTED

    what = _flat(record["what_was_asserted"])
    assert "SHRINKAGE" in what
    assert "before any" in what.lower()
    assert "cross-arm shrinkage figure existed" in what

    refuted = _flat(record["what_refuted_it"])
    assert "SCALE_INVARIANCE_PROHIBITION" in refuted
    assert "CROSS_ARM_SHRINKAGE_OBSERVED" in refuted
    assert "ARM_A_THE_SHRINKAGE_CONTROL" in refuted
    assert "AFTER the assertion" in refuted

    order = _flat(record["the_order_is_recorded_not_smoothed"])
    assert "AFTER the refutation, not before it" in order
    assert "NOT a prediction that survived contact with data" in order
    assert "only this record does" in order

    shape = _flat(record["the_familiar_shape"])
    assert "EXPLAINS THE OBSERVATIONS" in shape
    assert "caught by building the measurement it implied" in shape
    # The neighbouring error in the same episode is live.
    assert "SHRINKAGE_FIGURE_MISATTRIBUTED" in shape
    assert hasattr(phase21, "SHRINKAGE_FIGURE_MISATTRIBUTED")

    assert "not owed leniency" in _flat(record["what_it_does_not_excuse"])


# --------------------------------------------------------------------------
# coverage: nothing tests an ordering objective, checked at source
# --------------------------------------------------------------------------


def test_the_regression_arms_fit_an_absolute_target_per_factorys_own_docstring():
    """If an ordinal or ranking head is ever added to the factory, this
    fails and the coverage claim stops being true."""
    text = (REPO / "src" / "cleft" / "models" / "factory.py").read_text(
        encoding="utf-8"
    )
    assert "MSE on the raw 1-5 scale" in text
    assert "Not CORAL/CORN" in text
    assert "the tails are too sparse for an ordinal" in text

    claim = _flat(phase22.COVERAGE_NOTHING_TESTS_ORDERING["the_regression_arms"])
    assert "ABSOLUTE target" in claim and "SQUARED-ERROR loss" in claim
    assert "None ranks" in claim


def test_phase_17s_contrastive_loss_is_binary_and_discards_the_ordering():
    """The arm closest to a ranking objective throws the ordering away --
    checked against the schema's own closed choice."""
    pair_rule = schema.TASK_SPECS["siamese_contrastive"]["pair_rule"]
    assert pair_rule.choices == ("symmetric_if_zero_magnitude",), (
        "if pair_rule ever gains an ordered option, the coverage claim "
        "must be re-audited"
    )

    claim = _flat(
        phase22.COVERAGE_NOTHING_TESTS_ORDERING["phase_17s_contrastive_loss"]
    )
    assert "BINARY" in claim
    assert "DISCARDS THE ORDERING IT READS" in claim
    assert "symmetric_if_zero_magnitude" in claim
    # Four magnitudes collapse to a two-valued relation.
    assert "0.0, 0.015, 0.025, 0.035" in claim
    assert "throws the ordering away" in claim
    assert "binary contrastive" in _flat(phase17.__doc__)


def test_ldl_is_a_distribution_objective_not_an_ordering_one():
    text = (REPO / "src" / "cleft" / "train" / "ldl.py").read_text(
        encoding="utf-8"
    )
    assert "five-output softmax head and a KL loss" in text

    claim = _flat(phase22.COVERAGE_NOTHING_TESTS_ORDERING["ldl"])
    assert "DISTRIBUTION objective" in claim
    assert "KL loss" in claim
    assert "SHAPE of the rater distribution, not the ORDER of patients" in claim


def test_the_coverage_audit_reports_nothing_to_concede_and_says_why_that_matters():
    verdict = _flat(phase22.COVERAGE_NOTHING_TESTS_ORDERING["the_verdict"])
    assert "NOTHING TO CONCEDE" in verdict
    # The honest form of the audit is one that CAN come back positive,
    # and Phase 21's did.
    assert "CAN come back positive" in verdict
    assert "phase7b.ENSEMBLE_COMBINES" in verdict
    assert "ENSEMBLE_COMBINES" in phase21.PHASE_21_RECKONING["a_is_not_covered"]


# --------------------------------------------------------------------------
# the three arms
# --------------------------------------------------------------------------


def test_r_syn_is_registered_on_the_one_exact_ordering_and_names_its_axis():
    record = phase22.ARM_R_SYN_REGISTERED
    why = _flat(record["why_this_ordering"])
    assert "EXACT BY CONSTRUCTION" in why
    assert "0.0, 0.015, 0.025, 0.035" in why
    assert "KNOWN rather than ESTIMATED" in why

    axis = _flat(record["its_pairs_are_within_face"])
    assert "THE SAME FACE" in axis
    assert "nothing about face X against face Y" in axis
    assert "PAIRING_AXIS_ASSUMPTION" in axis
    # The pair count is deferred to the synth index, not guessed here.
    assert "NOT estimated here" in _flat(record["n_pairs_available"])


def test_r_all_trains_on_the_noise_ordered_pairs_deliberately():
    record = phase22.ARM_R_ALL_REGISTERED
    assert "27,966" in _flat(record["what_it_is"])

    deliberate = _flat(record["it_includes_the_noise_ordered_pairs_deliberately"])
    assert "24.85%" in deliberate
    assert "0.398942" in deliberate
    assert "the POINT of the arm, not an oversight" in deliberate

    # Both figures are the separation run's own, by identity.
    observed = phase21.COHORT_PAIR_SEPARATION_OBSERVED
    assert observed["n_pairs"] == 237 * 236 // 2 == 27966
    assert observed["fraction_below"]["se_diff_x1"] == 0.2485
    assert observed["se_diff"] == pytest.approx(0.398942, abs=5e-7)


def test_r_clear_declares_its_restriction_in_advance_and_never_post_hoc():
    record = phase22.ARM_R_CLEAR_REGISTERED
    legal = _flat(record["declared_in_advance_never_post_hoc"])
    assert "BEFORE the separation number existed" in legal
    assert "NEVER A POST-HOC FILTER" in legal
    assert "SELECTION_ON_EVALUATION_DATA_PROHIBITION" in legal
    # The prohibition it defers to is live.
    assert hasattr(phase21, "SELECTION_ON_EVALUATION_DATA_PROHIBITION")

    # 75.15% is the complement of the measured 24.85%, by arithmetic.
    assert "75.15%" in _flat(record["what_it_is"])
    assert round(1 - 0.2485, 4) == 0.7515

    # SE_diff is reproduced from its inputs, not pinned as a literal.
    import math

    from cleft.data import reliability

    assert phase21.COHORT_PAIR_SEPARATION_OBSERVED["se_diff"] == pytest.approx(
        0.657279 * math.sqrt(2) * math.sqrt(1 - reliability.RELIABILITY_237),
        abs=5e-6,
    )
    assert "sqrt(2)" in _flat(record["se_diff_provenance"])


def test_the_two_cohort_arms_differ_in_exactly_one_factor():
    ground = phase22.THE_R_ALL_R_CLEAR_GROUND
    one = _flat(ground["one_factor"])
    assert "EXACTLY ONE FACTOR" in one
    assert "HELP or HURT" in one

    ground_text = _flat(ground["the_ground"])
    assert "ARGUMENT into a MEASUREMENT" in ground_text
    assert "the choice that flatters" in ground_text
    assert "either arm alone is an anecdote" in ground_text
    # The cost is stated, including the outcome that would embarrass the
    # clean story.
    assert "the risk that R-all wins" in _flat(ground["what_it_costs"])


# --------------------------------------------------------------------------
# the pairing-axis assumption
# --------------------------------------------------------------------------


def test_the_pairing_axis_assumption_is_declared_as_an_assumption():
    record = phase22.PAIRING_AXIS_ASSUMPTION
    assert "ASSUMPTION UNDER TEST" in record["declared"]
    assert record["tag"] == "[REASONED] -- an assumption, explicitly not a finding"

    axes = _flat(record["the_two_axes"])
    assert "WITHIN-FACE" in axes and "BETWEEN-PATIENT" in axes
    assert "different relations sharing the word 'ordering'" in axes

    ruling = _flat(record["the_ruling"])
    assert "ASSUMING it" in ruling
    assert "WRITTEN DOWN AS AN ASSUMPTION rather than treated as known" in ruling
    # A null on R-syn keeps both readings available.
    assert "ranking does not help" in ruling
    assert "the axis did not transfer" in ruling
    assert "cannot later pretend the second reading was unavailable" in ruling

    family = _flat(record["same_family_as_magnitude_to_grade"])
    assert "CONSTRUCTED quantity to a CLINICAL one" in family
    assert "neither verified" in family

    # It is scoped: the cohort arms carry no axis assumption.
    scope = _flat(record["it_does_not_touch_the_cohort_arms"])
    assert "NO axis assumption" in scope


# --------------------------------------------------------------------------
# the diagnostic, and its derived thresholds
# --------------------------------------------------------------------------


def test_the_diagnostic_thresholds_are_measurements_with_no_free_parameter():
    """Derived, not chosen: both endpoints are figures already in the
    record, and this test reads them from their sources."""
    record = phase22.FEATURE_DIFFERENCE_DIAGNOSTIC
    derived = _flat(record["thresholds_derived_not_chosen"])
    assert "neither has a free parameter" in derived

    # HIGH is the all-pairs mean, at its source.
    assert "0.7012" in derived and "0.5729" in derived
    assert phase21.ARM_B_OBSERVED["residual_sign"]["mean_kappa"] == 0.7012
    assert phase21.ARM_B_OBSERVED["worst_quartile"]["mean_kappa"] == 0.5729

    # LOW is arm A's measured level, at its source.
    assert "+0.0956" in derived and "-0.0743" in derived
    assert "+0.0956" in _flat(phase21.ARM_A_THE_SHRINKAGE_CONTROL["the_control"])
    assert "-0.0743" in _flat(phase21.ARM_A_THE_SHRINKAGE_CONTROL["the_control"])

    # The band stays wide, and that is stated rather than narrowed.
    wide = _flat(record["the_wide_band_is_honest"])
    assert "WIDE band, and it stays wide" in wide
    assert "inventing a midpoint" in wide
    assert round(0.7012 - 0.0956, 4) == 0.6056, "the band's actual width"


def test_the_diagnostic_readings_are_registered_before_numbers():
    readings = phase22.FEATURE_DIFFERENCE_DIAGNOSTIC["readings"]
    assert sorted(readings) == [
        "kappa_at_or_above_the_all_pairs_mean",
        "kappa_at_or_below_arm_as_level",
        "kappa_between",
    ]

    high = _flat(readings["kappa_at_or_above_the_all_pairs_mean"])
    assert "THE ARM MAKES THE SAME ERRORS" in high
    # It fires regardless of PCC -- including on a PCC gain.
    assert "WHATEVER ITS PCC DOES" in high
    assert "including a PCC gain" in high

    low = _flat(readings["kappa_at_or_below_arm_as_level"])
    assert "LOOKING AT SOMETHING DIFFERENT" in low
    assert "EVIDENCE INDEPENDENT OF THE PCC DELTA" in low
    # Geirhos is context beside it, never a threshold.
    assert "0.066-0.068" in low
    assert "BESIDE it as context, never as a threshold" in low
    assert "0.066-0.068" in literature.GEIRHOS_ERROR_CONSISTENCY[
        "cnn_to_human_ood"
    ]

    between = _flat(readings["kappa_between"])
    assert "REPORT THE VALUE, CLAIM NEITHER" in between
    assert "not 'weak evidence of difference'" in between


def test_the_diagnostic_reuses_phase_21s_machinery_and_excludes_arm_a():
    record = phase22.FEATURE_DIFFERENCE_DIAGNOSTIC
    computed = _flat(record["what_is_computed"])
    assert "63 SHRINKING arms" in computed
    assert "both binarizations" in computed
    assert "same chance correction" in computed
    assert "reported SEPARATELY and never averaged" in computed

    excluded = _flat(record["why_the_63_and_not_the_64"])
    assert "EXCLUDED" in excluded
    assert "both sides of the diagnostic" in excluded
    assert 64 - 1 == 63

    prospective = _flat(record["arm_as_control_applied_prospectively"])
    assert "AFTER the fact" in prospective and "BEFORE any ranking arm runs" in prospective
    assert "post-hoc to pre-registered" in prospective


# --------------------------------------------------------------------------
# the contrast family
# --------------------------------------------------------------------------


def test_the_contrast_family_is_redeclared_at_fifteen_for_six_arms():
    """[UPDATED 2026-09-01, THIS PIN FIRED AS DESIGNED] It held the
    three-arm family at six. the maintainer's head-bounding ruling made six
    arms, so the family is recomputed BEFORE any number -- which is not
    what the nothing-added clause forbids."""
    record = phase22.CONTRAST_FAMILY
    assert record["count"] == 15

    primaries = _flat(record["primaries"])
    assert "SIX" in primaries and "0.2520" in primaries
    for arm in ("R-syn-B", "R-syn-U", "R-all-B", "R-all-U",
                "R-clear-B", "R-clear-U"):
        assert arm in primaries, arm

    secondaries = _flat(record["secondaries"])
    assert "NINE" in secondaries
    # Three groups, and the count is the arithmetic of the design.
    assert "THE HEAD QUESTION (3)" in secondaries
    assert "THE NOISE-PAIR QUESTION (2)" in secondaries
    assert "THE AXIS QUESTION (4)" in secondaries
    assert 3 + 2 + 4 == 9
    assert 6 + 9 == record["count"]
    # Only the head contrasts cross the bounding.
    assert "No contrast crosses the bounding except the three" in secondaries

    # The superseded registration is preserved, with its own count.
    superseded = _flat(record["superseded_registration"])
    assert "count SIX" in superseded
    assert "BEFORE any arm ran, not after a result" in superseded

    nothing = _flat(record["nothing_added_afterwards"])
    assert "[POST-HOC]" in nothing
    assert phase21.POST_HOC_ANALYSES["tag"] == "[POST-HOC]"


def test_the_family_is_confirmed_at_fifteen_against_the_derived_count():
    """[2026-09-01] The ruling was reissued once the real count was
    known, so the family does not rest on a number nobody checked."""
    confirmed = _flat(phase22.CONTRAST_FAMILY["confirmed_at_fifteen"])
    assert "CONFIRMED 2026-09-01, AGAINST THE DERIVED COUNT" in confirmed
    assert "REISSUED here against the derived 15" in confirmed
    # The three grounds.
    assert "SINGLE CONTROLLED FACTOR" in confirmed
    assert "every outcome is reported whatever it is" in confirmed
    assert "the available reductions all cost information" in confirmed
    # The named reduction and why it was refused.
    assert "ONE bounding only" in confirmed
    assert "an asymmetry with no justification" in confirmed
    assert "cheaper by leaving a question half-asked" in confirmed
    # It confirms the count the record actually holds.
    assert phase22.CONTRAST_FAMILY["count"] == 15


def test_the_family_count_is_reported_as_larger_than_the_ruling_assumed():
    """'Roughly doubles' was the ruling's premise; 2.5x is the fact."""
    record = phase22.CONTRAST_FAMILY
    flagged = _flat(record["the_count_is_larger_than_the_ruling_assumed"])
    assert "roughly doubles" in flagged
    assert "6 -> 15, a factor of 2.5" in flagged
    assert "open to re-ruling" in flagged
    assert "rather than quietly locked at 15" in flagged
    # The arithmetic the flag rests on.
    assert 15 / 6 == 2.5
    # And the ruling record points at the flag rather than burying it.
    assert "the_count_is_larger_than_the_ruling_" in _flat(
        phase22.HEAD_BOUNDING_RULED["and_the_count_was_understated_at_the_ruling"]
    )


def test_the_familys_multiplicity_is_recorded_rather_than_hidden():
    """[UPDATED 2026-09-01] the ruling was no alpha correction, on
    three recorded grounds -- the third being that the criterion is
    already stricter than any nominal level."""
    mult = _flat(phase22.CONTRAST_FAMILY["multiplicity_recorded_not_corrected"])
    assert "Fifteen contrasts" in mult
    assert "NO ALPHA CORRECTION" in mult
    assert "FIXED BEFORE ANY NUMBER" in mult
    assert "EVERY outcome is reported whatever it is" in mult
    assert "discover the count afterwards" in mult

    # The third ground is a measured fact, live at its source.
    assert "30 tested, 1 survived, 29 withdrawn" in mult
    band = _flat(
        ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"]["unresolvable_band"]
    )
    assert "30 tested, 1 survived, 29 withdrawn" in band
    assert "0.04 to 0.10" in band


# --------------------------------------------------------------------------
# readings both ways, and the unresolved positive
# --------------------------------------------------------------------------


def test_the_expected_outcome_is_registered_as_unresolved():
    record = phase22.READINGS_BOTH_WAYS
    expected = _flat(record["the_expected_outcome_is_unresolved"])
    assert "0.1386" in expected
    assert "cannot resolve 0.04 to 0.10" in expected
    assert "KNOWING its primary metric will most likely return nothing" in expected

    # Both figures live at their source.
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"] == 0.1386
    assert "0.04 to 0.10" in _flat(
        ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"]["unresolvable_band"]
    )


def test_both_directions_are_registered_per_arm_including_the_loss():
    ways = phase22.READINGS_BOTH_WAYS["per_arm_both_ways"]
    assert sorted(ways) == ["claimable_gain", "claimable_loss", "unresolved"]

    loss = _flat(ways["claimable_loss"])
    assert "claimable LOSS, not as 'no effect'" in loss
    assert "reports gains and shrugs at losses" in loss

    assert "the expected case" in _flat(ways["unresolved"])
    assert "NO CLAIM in either direction" in _flat(ways["unresolved"])


def test_an_unresolved_positive_licenses_a_point_estimate_and_nothing_else():
    """The reading that matters most, because it is the expected one."""
    record = phase22.READINGS_BOTH_WAYS
    licensed = _flat(record["what_an_unresolved_positive_licenses"])
    assert "point estimate with its interval and NO CLAIM" in licensed
    for hedge in ("promising trend", "suggestive of a gain",
                  "consistent with the hypothesis"):
        assert hedge in licensed, hedge
    assert "rests ENTIRELY on the diagnostic" in licensed
    assert "declared UP FRONT, not assembled once the primary disappoints" in licensed


def test_a_claimable_gain_would_still_not_license_the_shrinkage_mechanism():
    record = phase22.READINGS_BOTH_WAYS
    would = _flat(record["what_a_claimable_gain_would_license"])
    assert "raises PCC on this cohort" in would

    not_license = _flat(record["and_what_it_would_STILL_NOT_license"])
    assert "THE SHRINKAGE MECHANISM, WHICH REMAINS FORBIDDEN" in not_license
    assert "0.2334" in not_license
    assert "establish THAT ordering helps, never WHY" in not_license
    # Registered so that reaching for it is a visible contradiction.
    assert "visible contradiction of a pre-committed record" in not_license
    assert "CANNOT EXPLAIN THE PCC CEILING" in (
        phase21.SCALE_INVARIANCE_PROHIBITION
    )


# --------------------------------------------------------------------------
# the exit criteria draft -- a draft, not a lock
# --------------------------------------------------------------------------


def test_the_exit_criteria_are_locked_with_the_rulings_inside_them():
    """[2026-09-01] The draft is superseded; the lock carries the six
    arms, the fifteen contrasts and the settings, and preserves the
    draft unchanged beside it."""
    locked = phase22.EXIT_CRITERIA
    assert locked["tag"] == "[LOCKED]"
    assert "EXIT_CRITERIA_DRAFT, preserved unchanged" in _flat(
        locked["supersedes"]
    )
    # Fourteen: the draft's twelve, four amended, two added.
    numbered = [k for k in locked if k[0].isdigit()]
    assert len(numbered) == 14, numbered
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 15))

    # The four amendments name themselves as amendments.
    assert "[AMENDED AT THE LOCK -- three arms became six]" in _flat(
        locked["2_six_arms_built_as_registered"]
    )
    assert "[AMENDED AT THE LOCK -- six became fifteen]" in _flat(
        locked["7_fifteen_contrasts_and_no_more"]
    )
    assert "15" in _flat(locked["7_fifteen_contrasts_and_no_more"])
    assert "on all SIX arms" in _flat(locked["5_the_diagnostic_runs_on_every_arm"])
    assert "all fifteen" in _flat(locked["8_every_contrast_gets_a_verdict"])

    # The two additions, and what they protect.
    added = _flat(locked["13_the_three_open_settings_are_ruled_before_the_first_run"])
    assert "[ADDED AT THE LOCK]" in added
    assert "may not start with any of the three defaulted silently" in added
    monitor = _flat(
        locked["14_the_unbounded_monitor_limitation_is_reported_wherever_those_arms_are"]
    )
    assert "[ADDED AT THE LOCK]" in monitor
    assert "one candidate explanation among others, never the stated cause" in monitor

    # The nothing-added clause now covers settings too.
    assert "no setting introduced once numbers exist" in _flat(
        locked["12_nothing_added_after_the_lock"]
    )


def test_the_draft_is_preserved_unchanged_beside_the_lock():
    """A superseded draft that gets edited is not a preserved original."""
    draft = phase22.EXIT_CRITERIA_DRAFT
    assert "NOT LOCKED" in draft["drafted"]
    assert draft["tag"] == (
        "[DRAFT] -- not locked, not binding until it is fixed"
    )
    numbered = [k for k in draft if k[0].isdigit()]
    assert len(numbered) == 12, "the draft still has its original twelve"
    # Its three-arm and six-contrast wording is intact.
    assert "2_three_arms_built_as_registered" in draft
    assert "7_six_contrasts_and_no_more" in draft


def test_the_exit_criteria_are_a_draft_and_say_so():
    record = phase22.EXIT_CRITERIA_DRAFT
    assert "NOT LOCKED" in record["drafted"]
    assert record["tag"] == (
        "[DRAFT] -- not locked, not binding until it is fixed"
    )
    # The "a draft is not a lock" sentence is in the module's own text.
    # Strip the `#:` comment markers before flattening, or a sentence
    # that wraps across two comment lines never matches.
    source = _flat(
        (REPO / "src" / "cleft" / "phase22.py")
        .read_text(encoding="utf-8")
        .replace("#:", " ")
    )
    assert "A draft is not a lock" in source
    assert "cannot start against them until they are fixed" in source

    numbered = [k for k in record if k[0].isdigit()]
    assert len(numbered) == 12, numbered
    # Numbered 1..12 with no gaps.
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 13))


def test_the_draft_criteria_carry_the_phases_load_bearing_constraints():
    record = phase22.EXIT_CRITERIA_DRAFT

    assert "no arm dropped, no arm added" in _flat(
        record["2_three_arms_built_as_registered"]
    )
    assert "verified at the config level, not asserted" in _flat(
        record["3_r_all_and_r_clear_differ_in_one_factor_only"]
    )
    assert "recomputed" in _flat(
        record["4_the_se_diff_threshold_is_reproduced_not_pinned"]
    )
    # If a source figure moves, the phase stops rather than reading on.
    assert "the phase stops" in _flat(
        record["6_the_diagnostic_thresholds_are_the_registered_ones"]
    )
    assert "exactly six" in _flat(record["7_six_contrasts_and_no_more"])
    assert "quietly dropped" in _flat(record["8_every_contrast_gets_a_verdict"])
    assert "may not be reported as 'ranking does not help'" in _flat(
        record["9_the_pairing_axis_assumption_is_attached_to_r_syn_wherever_it_is_read"]
    )
    assert "remains binding" in _flat(
        record["10_no_pcc_gain_is_attributed_to_shrinkage"]
    )
    assert "closes with 'unresolved' in it if that is what happened" in _flat(
        record["11_the_expected_null_is_reported_as_a_null"]
    )
    assert "[POST-HOC] and fires nothing" in _flat(
        record["12_nothing_added_after_the_lock"]
    )


# --------------------------------------------------------------------------
# the config-surface report -- a report, not an edit
# --------------------------------------------------------------------------


def test_the_config_surface_report_matches_the_schema_it_describes():
    """Every claim about the surface is checked against the surface."""
    record = phase22.CONFIG_SURFACE_REPORT
    specs = schema.TASK_SPECS

    # 1. No loss field on train_cv.
    assert "loss" not in specs["train_cv"]
    assert "objective" not in specs["train_cv"]
    assert "MISSING ENTIRELY" in _flat(record["1_the_loss"])

    # 2. The only pair machinery is the binary one.
    assert "pair_rule" not in specs["train_cv"]
    assert specs["siamese_contrastive"]["pair_rule"].choices == (
        "symmetric_if_zero_magnitude",
    )
    assert "MISSING FOR THE COHORT" in _flat(record["2_pair_construction"])

    # 3. The readout is a distance, not a per-item score.
    assert specs["siamese_contrastive"]["readout_normalization"].choices == (
        "round_1_plus_4_min_d_over_margin",
    )
    head = _flat(record["3_a_per_item_scalar_head"])
    assert "DISTANCE BETWEEN TWO EMBEDDINGS" in head
    assert "different architectures, not a different setting" in head

    # 4. No cohort pair source; the arm name itself is a closed choice.
    assert specs["siamese_contrastive"]["arm"].choices == (
        "p17_arm_b", "p17_arm_c",
    )
    assert "synth_set" in specs["siamese_contrastive"]
    axis = _flat(record["4_the_inter_patient_pairing_axis"])
    assert "NO COHORT PAIR SOURCE EXISTS" in axis
    assert "p17_arm_b" in axis and "p17_arm_c" in axis

    # And pretrain has no ranking monitor.
    assert specs["pretrain"]["monitor"].choices == (
        "inner_val_mse", "inner_val_pcc",
    )


def test_the_config_surface_report_states_what_it_CAN_express_and_is_not_an_edit():
    record = phase22.CONFIG_SURFACE_REPORT
    can = _flat(record["what_the_surface_CAN_express"])
    assert "EXCEPT the four above" in can
    assert "narrow and specific, not a rewrite" in can
    # Every field it names really is on train_cv.
    for field in ("backbone", "seeds", "learning_rate", "batch_size",
                  "max_epochs", "patience", "trainable", "inner_val_frac"):
        assert field in schema.TASK_SPECS["train_cv"], field

    not_edit = _flat(record["not_an_edit"])
    assert "no schema kind added, no field extended, no config written" in not_edit
    assert "the move this project does not make" in not_edit

    summary = _flat(record["the_honest_summary"])
    assert "reads an ordering and discards it" in summary
    assert "visible at the ruling rather than at the lock" in summary


# --------------------------------------------------------------------------
# [2026-09-01] the maintainer's five rulings, and the lock
# --------------------------------------------------------------------------


def test_the_loss_ruling_records_its_ground_and_rejects_the_hybrid_by_name():
    record = phase22.LOSS_RULED
    assert "PAIRWISE LOGISTIC" in _flat(record["the_ruling"])

    ground = _flat(record["the_ground"])
    assert "P = 0.5 IS 'the panel cannot order these two'" in ground
    assert "24.85%" in ground
    assert "NO MARGIN" in ground
    # 24.85% is measured, not asserted.
    assert phase21.COHORT_PAIR_SEPARATION_OBSERVED[
        "fraction_below"]["se_diff_x1"] == 0.2485

    # (d) rejected by name, with all three reasons.
    d = _flat(record["d_rejected_by_name"])
    assert "REJECTED BY NAME" in d
    assert "reintroduces the shrinking term" in d
    assert "unattributable between the halves" in d
    assert "BREAKS THE DIAGNOSTIC" in d
    assert "SHOULD agree with the shrinking arms" in d

    # (a) and (c) unregistered, Phase 14's pattern, each with a reason.
    unreg = _flat(record["a_and_c_unregistered"])
    assert "EXPLICITLY UNREGISTERED -- no fishing" in unreg
    assert "requires a MARGIN" in unreg
    assert "assumes a consistent TOTAL ORDER" in unreg
    assert "presenting either as a fresh idea" in unreg
    # The phrase is Phase 14's, live at its source.
    assert "EXPLICITLY UNREGISTERED -- no fishing" in _flat(
        phase14.PHASE_14_CONCEDED_COVERED["variants_named_not_pursued"]
    )
    # The margin precedent it cites is real.
    assert "margin = 1.0" in _flat(phase17.DECLARED_SETTINGS_17["margin"])


def test_the_loss_ruling_reports_the_bank_holds_no_sample_size():
    from cleft import literature

    lit = _flat(phase22.LOSS_RULED["literature"])
    assert "The bank holds NO sample size for it" in lit
    assert "comparator error this project has made before" in lit
    # True at source: the only number in the entry is our own 237.
    import re

    entry = literature.SOURCES_BANKED["rankiqa"]
    numbers = set(re.findall(r"\d[\d,]*", " ".join(str(v) for v in entry.values())))
    assert numbers == {"237"}, numbers
    assert "comparator error this project has made before" in entry["flagged"]

    # And the caveat is scoped to the cohort arms, because R-syn's
    # order really is certain by construction -- as theirs is.
    assert "R-all and R-clear ONLY" in lit
    assert "EXACT BY CONSTRUCTION" in _flat(
        phase22.ARM_R_SYN_REGISTERED["why_this_ordering"]
    )


def test_pair_construction_makes_fold_honesty_structural():
    record = phase22.PAIR_CONSTRUCTION_RULED
    assert "ALL PAIRS EACH EPOCH" in _flat(record["the_ruling"])
    assert "no sampler seed" in _flat(record["the_ruling"]).lower()

    structural = _flat(record["fold_honesty_is_structural_not_policed"])
    assert "ONLY that fold's training rows" in structural
    assert "BY CONSTRUCTION" in structural
    assert "nothing has to check that it does not" in structural
    # The harness really does pass only the training rows.
    harness = (REPO / "src" / "cleft" / "train" / "harness.py").read_text(
        encoding="utf-8"
    )
    assert "backbone.train_epoch(features[train_rows], train_labels)" in harness

    # Why it mattered: folds group by PATIENT, and say nothing of pairs.
    why = _flat(record["why_it_mattered_here"])
    assert "per-PATIENT" in why
    assert "says nothing about a PAIR of patients" in why
    assert "a shape the project has never had" in why
    folds = (REPO / "src" / "cleft" / "data" / "folds.py").read_text(
        encoding="utf-8"
    )
    assert "StratifiedGroupKFold" in folds and "groups = patient" in folds


def test_the_difficulty_schedule_rejection_is_scoped_not_general():
    record = phase22.PAIR_CONSTRUCTION_RULED
    c = _flat(record["c_rejected_scoped_not_general"])
    assert "REJECTED BY NAME" in c
    assert "R-clear's filter applied gradually" in c
    assert "blur the single controlled factor" in c
    # Scoped: it is not a verdict on curricula.
    assert "not a judgement on curricula" in c.lower()
    assert "untested here and unaddressed by this record" in c

    # And the arms differ, which is why pair source is per-arm.
    differ = _flat(record["the_three_arms_differ_here"])
    assert "no leak path exists" in differ
    assert "per-arm setting while the sampling MODE is phase-wide" in differ
    # Only training is paired -- evaluation is unchanged.
    assert "only TRAINING is paired" in _flat(record["evaluation_is_unchanged"])


def test_the_head_ruling_rests_on_the_ldl_precedent_and_marks_where_it_stops():
    record = phase22.HEAD_RULED
    assert "REUSE THE SHIPPED factory.build(num_outputs=1)" in _flat(
        record["the_ruling"]
    )

    precedent = _flat(record["the_ldl_precedent"])
    assert "NON-MSE objective" in precedent
    assert "without touching frozen apparatus" in precedent
    # LDL really is a KL objective inside its own train_epoch.
    ldl = (REPO / "src" / "cleft" / "train" / "ldl.py").read_text(
        encoding="utf-8"
    )
    assert "def train_epoch" in ldl and "kl_div" in ldl

    # Where the precedent stops -- the honest limit of the analogy.
    stops = _flat(record["where_the_precedent_stops"])
    assert "stays on the label scale" in stops
    assert "precedent for the LOSS, not for the SCALE" in stops

    # (b) rejected as a second implementation of shipped code.
    b = _flat(record["b_rejected_as_a_second_implementation"])
    assert "SECOND IMPLEMENTATION" in b
    assert "task_siamese_contrastive" in b
    assert "independent of whether it would work" in b

    # What the p17 readout actually is, verified at source.
    readout = _flat(record["what_phase_17s_readout_produces_and_why_it_cannot_serve"])
    assert "left-right ASYMMETRY" in readout.replace("LEFT and RIGHT", "")
    assert "DECLARED FORMULA, not a learned head" in readout
    assert "order two DIFFERENT patients" in readout
    assert "round(1 + 4*min(d/margin, 1))" in readout
    run_text = (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    assert "phase17.distance_to_grade(d, margin=margin)" in run_text

    assert "one backbone class with a ranking train_epoch" in _flat(
        record["the_minimum_honest_new_component"]
    )


def test_the_axis_ruling_touches_neither_closed_vocabulary():
    record = phase22.AXIS_RULED
    assert "A NEW TASK KIND" in _flat(record["the_ruling"])

    untouched = _flat(record["what_it_does_not_touch"])
    assert "neither train_cv's closed vocabulary NOR the shipped" in untouched
    assert "no arm that has produced numbers" in untouched.lower()

    # The pattern it follows is real: those kinds all exist separately.
    for kind in ("train_graph_cv", "train_scut_decoder", "siamese_contrastive"):
        assert kind in schema.TASK_SPECS, kind
    assert "SEPARATE KINDS rather than flags" in _flat(
        record["it_is_how_this_repo_already_grows"]
    )

    # What (b) would have cost is counted, not waved at.
    rejected = _flat(record["what_was_rejected"])
    assert "FIVE closed choices" in rejected
    for field in ("arm", "pair_rule", "readout_normalization",
                  "branch_trainability"):
        assert field in rejected, field
        assert getattr(
            schema.TASK_SPECS["siamese_contrastive"][field], "choices", None
        ), field


def test_head_bounding_rules_both_and_states_the_cost():
    record = phase22.HEAD_BOUNDING_RULED
    ruling = _flat(record["the_ruling"])
    assert "BOTH, on all three arms. SIX ARMS" in ruling
    assert "BOUNDED" in ruling and "UNBOUNDED" in ruling

    ground = _flat(record["the_ground"])
    assert "settled by A PREDICTION otherwise" in ground
    assert "MEASURES it instead of ASSUMING it" in ground
    assert "a lean is not a measurement" in ground

    cost = _flat(record["the_cost_stated_not_absorbed"])
    assert "6 to 15" in cost
    assert "30 tested, 1 survived, 29 withdrawn" in cost
    assert "six arms take the contrast family from 6 to 15" in cost
    assert "absorbs its costs silently" in cost


def test_the_bounding_prediction_is_registered_to_be_refuted():
    record = phase22.BOUNDED_HEAD_PREDICTION_REGISTERED
    assert "BEFORE any arm runs" in record["registered"]
    assert "the record's" in _flat(record["whose"])

    prediction = _flat(record["the_prediction"])
    assert "BOUNDED ranking arm shrinks toward the label mean" in prediction
    assert "NEAR THE ALL-PAIRS LEVEL (0.7012 / 0.5729)" in prediction
    assert "An UNBOUNDED one does not" in prediction
    assert "0.3289" in prediction

    # Grounded in arm A, which is measured.
    assert "2.1730" in _flat(record["why_it_is_predicted"])
    assert phase21.CROSS_ARM_SHRINKAGE_OBSERVED["across_arms"]["max"] == 2.1730

    # It is refutable, fires nothing, and licenses no preference.
    refuted = _flat(record["how_it_can_be_refuted"])
    assert "the prediction is WRONG and is recorded as wrong" in refuted
    assert "fires no cell and gates nothing" in refuted
    assert "may not be cited as a reason to prefer either head" in _flat(
        record["what_it_does_NOT_license"]
    )
    assert record["tag"].startswith("[REASONED]")


def test_the_monitor_bind_is_resolved_without_touching_frozen_apparatus():
    record = phase22.MONITOR_BIND_RESOLVED
    assert "FROZEN harness.py" in _flat(record["the_bind"])

    # Both ruled values already exist in the frozen validator.
    harness = (REPO / "src" / "cleft" / "train" / "harness.py").read_text(
        encoding="utf-8"
    )
    assert 'self.monitor not in ("inner_val_mse", "inner_val_pcc")' in harness
    assert "no third monitor value is added" in _flat(
        record["frozen_apparatus_untouched"]
    ).lower()

    assert "inner_val_mse HONESTLY" in _flat(record["bounded_arms"])

    # The limitation is quoted from harness.py, verbatim.
    unbounded = _flat(record["unbounded_arms"])
    assert "DECLARED LIMITATION" in unbounded
    quote = ("MSE is the training objective; selecting on the reported "
             "metric would be selecting on the thing being claimed")
    assert quote in unbounded
    # Verbatim at source. The `#:` markers must come out first, or a
    # sentence wrapping across two comment lines never matches.
    assert quote in _flat(harness.replace("#:", " "))


def test_the_pairing_makes_the_monitor_limitation_checkable_not_isolated():
    """The honest claim is 'one candidate explanation', not 'the cause'."""
    checkable = _flat(
        phase22.MONITOR_BIND_RESOLVED["the_pairing_makes_the_limitation_CHECKABLE"]
    )
    assert "ONE CANDIDATE EXPLANATION AMONG OTHERS" in checkable
    assert "It does not isolate the selection effect" in checkable
    assert "the bounding itself and ordinary arm-to-arm variation" in checkable


def test_the_rescaling_note_permits_presentation_and_forbids_relabelling():
    record = phase22.RESCALING_NOTE
    arithmetic = _flat(record["the_arithmetic"])
    assert "invariant to affine rescaling" in arithmetic
    assert "without changing its PCC" in arithmetic
    # The same invariance that forbids the mechanism permits this.
    assert "SCALE_INVARIANCE_PROHIBITION" in arithmetic

    assert "NEVER A SECOND ARM" in _flat(record["it_is_a_presentation_step"])
    assert "adds nothing to the family's count of 15" in _flat(
        record["it_is_a_presentation_step"]
    )

    never = _flat(record["and_never_quoted_as_a_bounded_arms_result"])
    assert "WHAT THEY WERE TRAINED TO DO" in never
    assert "answer the phase's own question by relabelling" in never
    assert "R2 shape" in never

    # Rescaling does not recover the absolute metrics.
    not_recovered = _flat(record["what_rescaling_does_not_recover"])
    assert "MSE, acc3" in not_recovered
    assert "only those may be reported" in not_recovered


def test_the_settings_are_enumerated_at_ten_with_two_still_open():
    """[UPDATED 2026-09-01, THIS PIN FIRED AS DESIGNED] It held the
    split at 7 closed / 3 open. The target map was ruled -- conditional
    on the tie measurement -- so it moves to its own bucket."""
    record = phase22.DECLARED_SETTINGS_22
    closed = record["closed_by_the_rulings"]
    conditional = record["ruled_after_the_lock_with_a_precondition"]
    still_open = record["still_open_before_the_first_run"]
    assert len(closed) == 7, sorted(closed)
    assert len(conditional) == 1, sorted(conditional)
    assert len(still_open) == 2, sorted(still_open)
    assert len(closed) + len(conditional) + len(still_open) == 10

    # Numbered 1..10 across all three buckets, no gaps.
    numbers = sorted(
        int(k.split("_")[0])
        for k in list(closed) + list(conditional) + list(still_open)
    )
    assert numbers == list(range(1, 11))

    # The conditional entry preserves its original text.
    eight = _flat(conditional["8_target_probability_map"])
    assert "RULED 2026-09-01 -- HARD TARGETS" in eight
    assert "CONDITIONAL ON THE TIE MEASUREMENT" in eight
    assert "ORIGINAL, 2026-09-01: 'SURFACED BY THE LOSS RULING" in eight

    # The count can grow by a known amount, not by an invention.
    eleventh = _flat(record["and_the_tie_rule_may_become_an_eleventh"])
    assert "Ten today, eleven in that branch" in eleventh
    assert "KNOWN amount rather than by an invention" in eleventh

    # The standing clause is phase17's, verbatim in substance.
    clause = _flat(record["the_standing_clause"])
    assert "NEVER TUNED ACROSS RUNS" in clause
    assert "dated amendment with a reason" in clause
    assert "NEVER tuned across runs" in _flat(
        phase17.DECLARED_SETTINGS_17["never_tuned"]
    )


def test_the_rulings_removed_three_settings_and_surfaced_one():
    record = phase22.DECLARED_SETTINGS_22
    removed = _flat(record["what_the_rulings_removed"])
    assert "three settings vanished" in removed
    for gone in ("MARGIN", "PAIRS-PER-EPOCH", "SAMPLING SEED"):
        assert gone in removed, gone
    assert "commit to FEWER constants, not more" in removed

    # The one nobody had named, surfaced by the loss ruling itself.
    # [UPDATED 2026-09-01] It moved buckets when it was ruled; its
    # original wording is preserved inside the ruled entry.
    surfaced = _flat(record["ruled_after_the_lock_with_a_precondition"][
        "8_target_probability_map"
    ])
    assert "SURFACED BY THE LOSS RULING AND NOT CLOSED BY IT" in surfaced
    assert "must be ruled before the first run" in surfaced
    assert "listed here rather than defaulted" in surfaced
    # And the loss ruling itself points at the gap it opened.
    assert "TARGET PROBABILITY MAP" in _flat(
        phase22.LOSS_RULED["what_the_ruling_leaves_open"]
    )
    # The lock refuses to start without it.
    assert "target probability map" in _flat(
        phase22.EXIT_CRITERIA["13_the_three_open_settings_are_ruled_before_the_first_run"]
    )


def test_the_scalar_head_correction_is_dated_with_its_original_preserved():
    """[2026-09-01] The gap that was not a gap."""
    entry = _flat(phase22.CONFIG_SURFACE_REPORT["3_a_per_item_scalar_head"])
    assert "[CORRECTED 2026-09-01 -- THIS ENTRY WAS WRONG." in entry
    assert "A per-item scalar head is NOT missing" in entry
    assert "factory.build(num_outputs=1) ships" in entry
    assert "the LOSS that runs two items through that head" in entry
    # The original is preserved verbatim, and its still-true part marked.
    assert "ORIGINAL, 2026-09-01: 'THE READOUT IS THE WRONG SHAPE." in entry
    assert "LAST TWO SENTENCES remain true of siamese_contrastive" in entry

    # The claim is true at source: num_outputs=1 is the default.
    factory = (REPO / "src" / "cleft" / "models" / "factory.py").read_text(
        encoding="utf-8"
    )
    assert "num_outputs: int = 1" in factory
    assert "num_classes=num_outputs" in factory

    # The provenance names the failure mode, and the other three
    # entries were re-checked rather than assumed.
    prov = _flat(phase22.CONFIG_SURFACE_REPORT["3_correction_provenance"])
    assert "a recorded error, caught by the record" in prov
    assert "reads only the config layer" in prov
    assert "re-checked at the code layer" in prov


# --------------------------------------------------------------------------
# [2026-09-01] The target map, the attractor finding, the tie measurement
# --------------------------------------------------------------------------


def test_the_target_map_ruling_records_its_ground_positively():
    record = phase22.TARGET_MAP_RULED
    assert "P = 1 if m_i > m_j else 0" in _flat(record["the_ruling"])
    assert "Conditional on the tie measurement" in _flat(record["the_ruling"])

    ground = _flat(record["the_ground"])
    assert "finding lives in the SECONDARY" in ground
    assert "R-clear is ALREADY the honest-about-the-panel arm" in ground
    assert "collapses the pair" in ground

    # Stated positively, not as a concession.
    positive = _flat(record["the_positive_reading"])
    assert "stated positively, not as a concession" in positive
    assert "what R-all IS FOR" in positive
    assert "the thing being exposed" in positive
    assert "not a defect being tolerated" in positive
    # And it matches what R-all was registered for.
    assert "the POINT of the arm, not an oversight" in _flat(
        phase22.ARM_R_ALL_REGISTERED[
            "it_includes_the_noise_ordered_pairs_deliberately"
        ]
    )


def test_the_reasons_against_the_ruling_are_recorded_in_full():
    """A close ruling that records only its winning side is a ruling
    nobody can reweigh later."""
    against = _flat(phase22.TARGET_MAP_RULED["the_reasons_against_in_full"])
    assert "the ruling is CLOSE" in against
    # (i) the measured contradiction.
    assert "CONTRADICTS for 24.85% of pairs" in against
    assert "not a theoretical objection but a measured one" in against
    assert phase21.COHORT_PAIR_SEPARATION_OBSERVED[
        "fraction_below"]["se_diff_x1"] == 0.2485
    # (ii) the probit-EB map's standing.
    assert "PARAMETER-FREE BY DERIVATION" in against
    assert "correct posterior under the project's own error model" in against
    assert "Kelley" in against
    # Both were in front of the ruling, and were outweighed not answered.
    assert "made with both of these in front of it" in against
    assert "they were outweighed" in against


def test_the_probit_derivation_is_kept_on_record_and_is_arithmetically_right():
    """2c was not taken; its derivation survives so it cannot reappear
    later as a fresh idea."""
    import math
    from statistics import NormalDist

    from cleft.data import reliability

    record = phase22.TARGET_MAP_RULED
    unreg = _flat(record["options_2a_2b_2c_unregistered_not_rejected"])
    assert "UNREGISTERED, NOT REJECTED" in unreg
    assert "is defensible" in unreg
    assert "CONSIDERED, its derivation is ON RECORD here, and it was NOT TAKEN" in unreg
    assert "may not be presented later as a fresh idea" in unreg

    derivation = _flat(record["the_derivation_kept_on_record"])
    obs = phase21.COHORT_PAIR_SEPARATION_OBSERVED
    # SE_diff = SE_single * sqrt(2), exactly, in the measured figures.
    assert "1.41421" in derivation
    assert obs["se_diff"] / obs["se_single"] == pytest.approx(
        math.sqrt(2), abs=1e-5
    )
    # The shrinkage factor really is the reliability (Kelley).
    rel = reliability.RELIABILITY_237
    assert 1 - (obs["se_single"] / obs["sd_obs"]) ** 2 == pytest.approx(
        rel, abs=1e-4
    )
    # And the three quoted values reproduce.
    N = NormalDist()
    k = math.sqrt(rel)
    assert "0.9032" in derivation
    assert k == pytest.approx(0.9032, abs=5e-5)
    assert N.cdf(k * 1.0) == pytest.approx(0.8168, abs=5e-5)
    assert N.cdf(k * 0.5) == pytest.approx(0.6742, abs=5e-5)
    assert N.cdf(0.0) == 0.5
    for value in ("0.8168", "0.6742"):
        assert value in derivation, value


def test_options_3_and_4_are_rejected_by_name_with_their_reasons():
    record = phase22.TARGET_MAP_RULED

    three = _flat(record["option_3_rejected_by_name"])
    assert "ACTIVE PULL TOWARD EQUAL SCORES" in three
    assert "a compression term R-clear lacks" in three
    assert "DIFFERENT QUESTION than the one registered" in three
    assert "It does not weaken the secondary; it replaces it" in three

    four = _flat(record["option_4_rejected_by_name"])
    assert "NEW UNDECLARABLE CONSTANT" in four
    assert "the logistic loss ruling was taken to avoid" in four
    assert "PARTIALLY DUPLICATES R-clear's filter" in four
    assert "w's limiting case" in four
    # The duplication is named, as required.
    assert "duplication is named rather than left implicit" in four


def test_the_literature_precedent_is_recorded_as_not_reaching():
    from cleft import literature

    lit = _flat(phase22.TARGET_MAP_RULED["literature_does_not_reach_here"])
    assert "HARD BY CONSTRUCTION" in lit
    assert "covers option 1 ONLY" in lit
    assert "where option 1's assumption is TRUE" in lit
    assert "the precedent does not reach" in lit
    # True at source, both claims.
    assert "known BY CONSTRUCTION" in literature.SOURCES_BANKED["rankiqa"][
        "how_it_bears"
    ]
    assert "NO." in literature.SOURCES_BANKED["swayamdipta_cartography"][
        "whether_regression_is_addressed"
    ]


def test_the_attractor_finding_is_arithmetic_and_checks_out():
    import math
    from statistics import NormalDist

    record = phase22.ATTRACTOR_FINDING
    assert "sigma(ds) - P" in _flat(record["the_gradient"])
    assert "zero at ds = logit(P)" in _flat(record["the_gradient"])

    # The gradient really is zero at logit(P) and positive above.
    def sigma(x):
        return 1 / (1 + math.exp(-x))

    for p in (0.6, 0.75, 0.9):
        gap = math.log(p / (1 - p))
        assert sigma(gap) - p == pytest.approx(0.0, abs=1e-12)
        assert sigma(gap + 0.5) - p > 0, "pulls back once exceeded"
        assert sigma(gap - 0.5) - p < 0

    # A hard target never reverses, at any weight.
    for ds in (-2.0, 0.0, 2.0, 10.0):
        assert sigma(ds) - 1.0 < 0

    diff = _flat(record["so_soft_targets_are_an_attractor_not_a_discount"])
    assert "PUSHES the gap wider without limit" in diff.replace("pushes", "PUSHES")
    assert "EXACTLY THIS MUCH" in diff
    assert "not the same one in different clothes" in diff

    # logit(Phi(z)) ~= 1.6z, to the stated tolerances.
    N = NormalDist()

    def logit_phi(z):
        p = N.cdf(z)
        return math.log(p / (1 - p))

    assert abs(logit_phi(0.5) / (1.6 * 0.5) - 1) < 0.01, "within 1% at z=0.5"
    assert abs(logit_phi(1.0) / (1.6 * 1.0) - 1) < 0.05, "4% at z=1"
    affine = _flat(record["and_a_smooth_maps_optimum_is_affine_in_the_panel_mean"])
    assert "within 1% to z = 0.5" in affine
    assert "AFFINE IN THE PANEL MEAN" in affine
    assert "the same target function the MSE arms fit" in affine


def test_the_attractor_finding_is_scoped_against_the_combined_loss_concern():
    record = phase22.ATTRACTOR_FINDING
    # Recorded separately because it postdates the loss ruling.
    why = _flat(record["why_it_is_recorded_separately"])
    assert "NOT VISIBLE when that ruling was made" in why
    assert "rather than folded in silently" in why

    milder = _flat(record["a_milder_relative_of_the_combined_loss_concern"])
    assert "pin the GAP, not the LEVEL" in milder
    assert "PCC invariance still holds" in milder
    # The prohibition really is untouched by this.
    assert "SCALE_INVARIANCE_PROHIBITION is untouched" in milder
    assert "CANNOT EXPLAIN THE PCC CEILING" in (
        phase21.SCALE_INVARIANCE_PROHIBITION
    )

    avoids = _flat(record["the_hard_target_ruling_avoids_it"])
    assert "NO FINITE OPTIMUM" in avoids
    assert "aim at ORDER ALONE" in avoids
    assert "avoids the finding rather than accommodating it" in avoids


def test_the_tie_measurement_is_designed_with_readings_before_the_number():
    record = phase22.TIE_FRACTION_MEASUREMENT_DESIGNED

    why = _flat(record["why_it_must_run_first"])
    assert "definite WRONG order" in why
    assert "P = 0" in why
    assert "not an error any check would raise" in why

    holds = _flat(record["what_the_record_holds_and_does_not"])
    assert "17 distinct separations" in holds
    assert "DOES NOT HOLD" in holds
    obs = phase21.COHORT_PAIR_SEPARATION_OBSERVED
    assert obs["n_distinct_separations"] == 17
    assert "tie" not in str(obs).lower(), "the record really has no tie count"

    # Cached source, aggregates only.
    source = _flat(record["the_source"])
    assert "same cached manifest the separation run used" in source
    assert "no GPU, no new artifact, no new hash" in source

    assert sorted(record["quantities"]) == [
        "1_tie_count_and_fraction",
        "2_smallest_separations_distribution",
    ]

    # The float hazard is flagged before the build, not after.
    hazard = _flat(record["a_hazard_to_settle_at_build"])
    assert "robust to float representation" in hazard
    assert "exact in decimal but NOT in binary" in hazard
    assert "must be declared, not left to whatever == does" in hazard
    assert "rater count per patient should be confirmed" in hazard


def test_the_tie_boundary_says_which_half_is_derived_and_which_chosen():
    """The reference quantity is derived; the fraction on it is not, and
    the record says so rather than manufacturing a derivation."""
    record = phase22.TIE_FRACTION_MEASUREMENT_DESIGNED
    boundary = _flat(record["the_boundary"])
    assert "PART DERIVED, PART CHOSEN, AND SAID PLAINLY" in boundary
    assert "REFERENCE QUANTITY is derived" in boundary
    assert "FRACTION on it is CHOSEN at one half" in boundary
    assert "No derivation yields a specific number here" in boundary
    assert "the half is a judgement, stated as one" in boundary

    # The arithmetic of both halves.
    assert "12.425%" in boundary
    assert 0.2485 / 2 == 0.12425, 'the exact half, not a rounded one'
    # The reference set's size, and the honesty about its precision:
    # the record holds a rounded fraction, so the count is 6949 or 6950.
    assert "~6,950" in boundary
    assert "the record holds the FRACTION ROUNDED TO FOUR PLACES" in boundary
    assert 0.2485 * 27966 == pytest.approx(6949.55, abs=5e-3)
    # Carrying the rounding interval through really does give [6949, 6950].
    lo = int(-(-0.24845 * 27966 // 1))
    hi = int(0.24855 * 27966)
    assert (lo, hi) == (6949, 6950), (lo, hi)
    assert "[6,949, 6,950]" in boundary

    # Two readings, and the large cell reopens rather than withdraws.
    readings = record["readings"]
    assert sorted(readings) == [
        "ties_are_a_large_fraction", "ties_are_a_small_fraction",
    ]
    small = _flat(readings["ties_are_a_small_fraction"])
    assert "< 12.425%" in small and "STANDS AS ISSUED" in small
    large = _flat(readings["ties_are_a_large_fraction"])
    assert ">= 12.425%" in large
    assert "the number in hand" in large
    assert "not withdrawn by this cell; it is reopened" in large


def test_the_tie_rule_candidates_are_named_and_not_chosen():
    candidates = phase22.TIE_FRACTION_MEASUREMENT_DESIGNED[
        "the_tie_rule_candidates_named_not_chosen"
    ]
    assert "CHOICE AMONG NAMED OPTIONS rather than an invention" in _flat(
        candidates["registered"]
    )
    assert "None is chosen here" in _flat(candidates["registered"])
    assert sorted(k for k in candidates if k != "registered") == [
        "a_drop_tied_pairs",
        "b_keep_with_p_0_5",
        "c_break_ties_by_a_declared_deterministic_rule",
    ]

    # (b) carries the attractor warning, since it is option 3's
    # mechanism on a smaller set.
    b = _flat(candidates["b_keep_with_p_0_5"])
    assert "REINTRODUCES THE ATTRACTOR" in b
    assert "same mechanism that got option 3 rejected" in b
    assert "If it is ruled, it is ruled knowing that" in b

    # (c) names its wrong labels as deliberate.
    c = _flat(candidates["c_break_ties_by_a_declared_deterministic_rule"])
    assert "wrong order to half the tied pairs BY CONSTRUCTION" in c
    assert "deliberate wrong label rather than an arbitrary convention" in c


# --------------------------------------------------------------------------
# [2026-09-01] The tie measurement: the declared rule, the task, the config
# --------------------------------------------------------------------------


def test_the_comparison_rule_is_declared_and_reuses_shipped_code():
    record = phase22.TIE_COMPARISON_RULE_DECLARED
    assert "BEFORE the task was written" in record["declared"]

    rule = _flat(record["the_rule"])
    assert "INTEGER RATER SUMS ARE EQUAL" in rule
    assert "no float comparison and no tolerance" in rule.lower()

    reuse = _flat(record["it_reuses_shipped_code"])
    assert "phase8c.rater_multiset ALREADY SHIPS" in reuse
    assert "reused, not reimplemented" in reuse
    # It really does ship, and really does what the record says.
    from cleft import phase8c

    assert callable(phase8c.rater_multiset)
    assert phase8c.rater_multiset(
        {"soft_1": 0.0, "soft_2": 1.0, "soft_3": 0.0,
         "soft_4": 0.0, "soft_5": 0.0}
    ) == [2, 2, 2, 2, 2]


def test_the_rater_count_is_confirmed_from_the_data_by_refusal():
    """Not assumed to be five -- the shipped recovery refuses a manifest
    that is not five raters, which is the confirmation."""
    from cleft import phase8c

    record = phase22.TIE_COMPARISON_RULE_DECLARED
    confirmed = _flat(record["the_rater_count_is_confirmed_not_assumed"])
    assert "REFUSES otherwise" in confirmed
    assert "not render five invented dots" in confirmed
    assert "stops the run" in confirmed

    # A four-rater panel: the fractions are quarters, not fifths.
    with pytest.raises(phase8c.Phase8cError, match="not a fifth"):
        phase8c.rater_multiset(
            {"soft_1": 0.0, "soft_2": 0.75, "soft_3": 0.25,
             "soft_4": 0.0, "soft_5": 0.0}
        )
    # Fractions that are fifths but do not sum to five raters.
    with pytest.raises(phase8c.Phase8cError, match="not 5"):
        phase8c.rater_multiset(
            {"soft_1": 0.0, "soft_2": 0.4, "soft_3": 0.2,
             "soft_4": 0.0, "soft_5": 0.0}
        )


def test_the_alternatives_are_recorded_with_the_hazard_stated_honestly():
    record = phase22.TIE_COMPARISON_RULE_DECLARED

    # The honest framing: the float route would probably also work.
    inert = _flat(record["the_hazard_is_real_in_principle_and_inert_here"])
    assert "would almost certainly give the same answer" in inert
    assert "one rater-step, 1/5 = 0.2" in inert
    assert "not chosen because floats would fail" in inert
    assert "removes a question instead of bounding it" in inert

    alternatives = record["alternatives_considered"]
    assert sorted(alternatives) == [
        "a_declared_tolerance", "exact_equality_on_the_stored_floats",
    ]
    floats = _flat(alternatives["exact_equality_on_the_stored_floats"])
    assert "Correct by coincidence rather than by construction" in floats
    tol = _flat(alternatives["a_declared_tolerance"])
    assert "TOLERANCE = 1e-6" in tol
    assert "introduces a constant where none is needed" in tol
    # The precedent it cites is live and says what it is quoted saying.
    from cleft.data import softlabels

    assert softlabels.TOLERANCE == 1e-6
    assert "It is NOT a knob" in " ".join(
        (REPO / "src" / "cleft" / "data" / "softlabels.py")
        .read_text(encoding="utf-8").split()
    )


def _tie_manifest(tmp_path, sums):
    """A manifest whose rows have the given INTEGER rater sums, built
    from real five-rater multisets so mean == sum/5 exactly."""
    import csv

    # counts for a sum S, five raters: (S - 10) threes and the rest twos,
    # which covers S in [10, 15] -- enough for a fixture.
    manifest_dir = tmp_path / "cleft_v1"
    manifest_dir.mkdir(parents=True)
    header = (
        "patient_id,frontal_id,basal_id,mean,median,mode,weighted_mean,"
        "orthodontist,soft_1,soft_2,soft_3,soft_4,soft_5,class3,fold"
    )
    lines = ["# CLUSTER-ONLY: patient-keyed", header]
    for index, total in enumerate(sums):
        n_three = total - 10
        n_two = 5 - n_three
        assert 0 <= n_three <= 5, total
        mean = total / 5.0
        soft = [0.0, n_two / 5.0, n_three / 5.0, 0.0, 0.0]
        lines.append(
            f"{index + 1},{1000 + index},,{mean!r},{mean!r},{mean!r},"
            f"{mean!r},{mean!r},"
            + ",".join(repr(s) for s in soft)
            + f",1,{index % 5}"
        )
    (manifest_dir / "manifest.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return manifest_dir


def _tie_config(path, manifest_dir, **overrides):
    """Writes the config to `path` and returns it -- RunContext takes a
    config PATH, not a dict."""
    import yaml

    from cleft.provenance.hashing import hash_dir

    task = {
        "kind": "tie_fraction",
        "manifest_artifact": "manifest_v1",
        "label": "mean",
        "expect_patients": 6,
        "expect_pairs": 15,
        "expect_below_se_diff": None,
        "boundary": 0.12425,
        "report_smallest": 4,
    }
    task.update(overrides)
    # The optional cross-check is OMITTED rather than nulled: the field
    # is a float when present, and fixtures have no separation run to
    # reproduce.
    task = {k: v for k, v in task.items() if v is not None}
    config = {
        "schema_version": 1, "phase": "p22", "tier": "dev",
        "seed": 1337,
        "inputs": [{"name": "manifest_v1", "path": str(manifest_dir),
                    "rollup_sha256": hash_dir(manifest_dir)["rollup"]}],
        "task": task,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def test_the_tie_task_counts_ties_by_hand_checkable_arithmetic(
    tmp_path, clean_repo
):
    """Six patients with rater sums 10,10,11,12,12,12 give 15 pairs:
    four tied, five at one rater-step, six at two. Counted by hand."""
    import json

    from cleft.run import TASKS, RunContext

    manifest_dir = _tie_manifest(tmp_path, [10, 10, 11, 12, 12, 12])
    config = _tie_config(tmp_path / "cfg.yaml", manifest_dir)
    out_root = tmp_path / "runs"
    with RunContext(config, out_root, repo_root=clean_repo) as ctx:
        TASKS["tie_fraction"](ctx)
    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )

    assert metrics["n_patients"] == 6
    assert metrics["n_pairs"] == 15 == 6 * 5 // 2
    assert metrics["n_ties"] == 4
    assert metrics["tie_fraction"] == pytest.approx(4 / 15)

    # The bottom of the distribution, in both unit systems.
    smallest = metrics["smallest_separations"]
    assert [s["rater_steps"] for s in smallest] == [0, 1, 2]
    assert [s["count"] for s in smallest] == [4, 5, 6]
    assert [s["separation"] for s in smallest] == pytest.approx([0.0, 0.2, 0.4])
    assert sum(s["count"] for s in smallest) == 15


def test_the_tie_task_writes_no_patient_ids(tmp_path, clean_repo):
    """Aggregates only -- the design says so and the output must obey."""
    from cleft.run import TASKS, RunContext

    manifest_dir = _tie_manifest(tmp_path, [10, 11, 12, 13])
    config = _tie_config(
        tmp_path / "cfg.yaml", manifest_dir, expect_patients=4, expect_pairs=6
    )
    out_root = tmp_path / "runs"
    with RunContext(config, out_root, repo_root=clean_repo) as ctx:
        TASKS["tie_fraction"](ctx)

    text = (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    assert "patient_id" not in text
    for shareable in ctx.run_dir.rglob("*"):
        if shareable.is_file():
            assert "CLUSTER-ONLY" not in shareable.name


def test_the_tie_task_stops_when_the_cross_check_disagrees(
    tmp_path, clean_repo
):
    """'If either disagrees with the separation run, stop and report
    rather than proceeding.'"""
    from cleft.run import TASKS, RunContext

    manifest_dir = _tie_manifest(tmp_path, [10, 10, 11, 12, 12, 12])
    out_root = tmp_path / "runs"

    # Wrong pair count declared.
    config = _tie_config(tmp_path / "a.yaml", manifest_dir, expect_pairs=99)
    with pytest.raises(ValueError, match="99"):
        with RunContext(config, out_root, repo_root=clean_repo) as ctx:
            TASKS["tie_fraction"](ctx)

    # Wrong below-SE_diff fraction declared.
    config = _tie_config(
        tmp_path / "b.yaml", manifest_dir, expect_below_se_diff=0.9999
    )
    with pytest.raises(ValueError, match="0.9999"):
        with RunContext(config, out_root, repo_root=clean_repo) as ctx:
            TASKS["tie_fraction"](ctx)


def test_the_tie_task_refuses_a_manifest_that_is_not_five_raters(
    tmp_path, clean_repo
):
    """The rater count is confirmed from the data: a four-rater panel
    stops the run rather than producing a tie count."""
    import csv

    from cleft import phase8c
    from cleft.run import TASKS, RunContext

    manifest_dir = _tie_manifest(tmp_path, [10, 11, 12, 13])
    path = manifest_dir / "manifest.csv"
    text = path.read_text(encoding="utf-8").splitlines()
    # Rewrite one row's soft columns as quarters, not fifths.
    rows = text[2].split(",")
    rows[8:13] = ["0.0", "0.75", "0.25", "0.0", "0.0"]
    text[2] = ",".join(rows)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")

    config = _tie_config(
        tmp_path / "cfg.yaml", manifest_dir, expect_patients=4, expect_pairs=6
    )
    out_root = tmp_path / "runs"
    with pytest.raises(phase8c.Phase8cError, match="not a fifth"):
        with RunContext(config, out_root, repo_root=clean_repo) as ctx:
            TASKS["tie_fraction"](ctx)


def test_the_reading_fires_mechanically_against_the_registered_boundary(
    tmp_path, clean_repo
):
    """Neither cell may be stretched. 4/15 = 0.2667 is above 0.12425."""
    import json

    from cleft.run import TASKS, RunContext

    manifest_dir = _tie_manifest(tmp_path, [10, 10, 11, 12, 12, 12])
    config = _tie_config(tmp_path / "cfg.yaml", manifest_dir)
    out_root = tmp_path / "runs"
    with RunContext(config, out_root, repo_root=clean_repo) as ctx:
        TASKS["tie_fraction"](ctx)
    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )

    assert metrics["boundary"] == 0.12425
    assert metrics["cell_fired"] == "ties_are_a_large_fraction"
    # It fires on the arithmetic, and the cell is one of the two
    # registered ones -- by identity, not by spelling.
    assert metrics["cell_fired"] in phase22.TIE_FRACTION_MEASUREMENT_DESIGNED[
        "readings"
    ]

    # Below the boundary fires the other cell.
    manifest_dir = _tie_manifest(tmp_path / "b", [10, 11, 12, 13, 14, 15])
    config = _tie_config(tmp_path / "c.yaml", manifest_dir)
    with RunContext(config, tmp_path / "runs_b", repo_root=clean_repo) as ctx:
        TASKS["tie_fraction"](ctx)
    metrics = json.loads(
        (ctx.run_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["n_ties"] == 0
    assert metrics["cell_fired"] == "ties_are_a_small_fraction"


def test_the_task_reuses_rater_multiset_rather_than_recovering_again():
    """A second recovery of the same quantity would be disqualifying."""
    import inspect

    from cleft import run as run_module

    body = inspect.getsource(run_module.task_tie_fraction)
    assert "rater_multiset" in body
    # Scan the CODE, not the docstring -- the docstring explains the
    # recovery and naturally mentions soft_k.
    code = body.split('"""')[2]
    assert "soft_" not in code, "the soft columns belong to rater_multiset"
    assert "5.0 *" not in code and "* 5" not in code


def test_the_tie_config_declares_the_manifest_only_and_nothing_pending():
    import yaml

    path = REPO / "configs" / "p22_tie_fraction.yaml"
    assert path.is_file()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    # One input, the manifest, carried verbatim from the separation run.
    assert len(config["inputs"]) == 1
    entry = config["inputs"][0]
    separation = yaml.safe_load(
        (REPO / "configs" / "p21_cohort_pair_separation.yaml")
        .read_text(encoding="utf-8")
    )
    donor = separation["inputs"][0]
    assert entry["name"] == donor["name"] == "manifest_v1"
    assert entry["path"] == donor["path"]
    assert entry["rollup_sha256"] == donor["rollup_sha256"]
    assert len(entry["rollup_sha256"]) == 64
    assert set(entry["rollup_sha256"]) != {"0"}, "nothing pending"

    task = config["task"]
    assert task["kind"] == "tie_fraction"
    assert task["expect_patients"] == 237
    assert task["expect_pairs"] == 237 * 236 // 2 == 27966
    assert task["expect_below_se_diff"] == 0.2485
    assert task["boundary"] == 0.12425
    # The cross-check figures are the separation run's own.
    assert phase21.COHORT_PAIR_SEPARATION_OBSERVED["n_pairs"] == 27966
    assert phase21.COHORT_PAIR_SEPARATION_OBSERVED[
        "fraction_below"]["se_diff_x1"] == 0.2485


# --------------------------------------------------------------------------
# [2026-09-01] The tie outturn, the tie rule, and the six arms
# --------------------------------------------------------------------------


def test_the_tie_outturn_is_banked_and_fires_the_small_cell():
    record = phase22.TIE_FRACTION_OBSERVED
    assert record["n_ties"] == 2345
    assert record["n_pairs"] == 27966
    assert record["tie_fraction"] == 0.0839
    assert round(2345 / 27966, 4) == 0.0839
    assert record["tie_fraction"] < record["boundary"] == 0.12425

    # The cell fired by identity against the registered readings.
    assert "ties_are_a_small_fraction" in (
        phase22.TIE_FRACTION_MEASUREMENT_DESIGNED["readings"]
    )
    verdict = _flat(record["the_verdict"])
    assert "STANDS AS ISSUED" in verdict
    assert "The cell was not close" in verdict

    held = _flat(record["the_preconditions_all_held"])
    assert "CONFIRMED FROM THE DATA at 5 for all 237" in held
    assert "EXACTLY ZERO" in held
    assert "27,966 pairs and 0.2485 below SE_diff" in held


def test_the_step_profile_reproduces_all_six_banked_fractions():
    """Two independent measurements of the same cohort, agreeing on
    every threshold -- the integer route and the float route land in
    the same place."""
    import math

    from cleft.data import reliability

    profile = {
        int(k): v for k, v in
        phase22.TIE_FRACTION_OBSERVED["separation_profile_in_rater_steps"]
        .items()
    }
    assert profile == {0: 2345, 1: 4604, 2: 4288, 3: 3790, 4: 3313, 5: 2795}

    sd_obs = phase21.COHORT_PAIR_SEPARATION_OBSERVED["sd_obs"]
    k_single = math.sqrt(1 - reliability.RELIABILITY_237)
    scales = {"se_single": sd_obs * k_single,
              "se_diff": sd_obs * math.sqrt(2) * k_single}
    banked = phase21.COHORT_PAIR_SEPARATION_OBSERVED["fraction_below"]
    n_pairs = 27966
    for name, scale in scales.items():
        for m in (1, 2, 3):
            steps_below = m * scale * 5  # separations are k/5
            counted = sum(c for k, c in profile.items() if k < steps_below)
            assert round(counted / n_pairs, 4) == banked[f"{name}_x{m}"], (
                name, m
            )


def test_the_reference_set_interval_is_resolved_at_6949():
    record = phase22.TIE_FRACTION_OBSERVED
    resolved = _flat(record["the_reference_set_interval_is_RESOLVED"])
    assert "[6,949, 6,950] -> EXACTLY 6,949" in resolved
    assert "Interval RESOLVED" in resolved
    # Steps 0 and 1 are the only ones below SE_diff.
    assert 2345 + 4604 == 6949
    assert round(6949 / 27966, 4) == 0.2485
    # The boundary came from the ROUNDED fraction; the exact count
    # gives 0.124240. The record states both rather than claiming an
    # agreement that is not there -- and the declared 0.12425 stands.
    assert round(6949 / 2 / 27966, 6) == 0.12424
    assert "0.124240" in resolved
    assert "a difference of 1.0e-5" in resolved
    assert "STANDS as the registered boundary and is not restated" in resolved
    assert phase22.TIE_FRACTION_OBSERVED["boundary"] == 0.12425
    # Nothing turns on it: 0.0839 is far below both.
    assert 0.0839 < 0.124240 < 0.12425

    why = _flat(record["why_the_prior_missed_low"])
    assert "STEP FUNCTION, not a continuum" in why
    assert "no mass anywhere between 0.2 and 0.4" in why
    assert "wrong about the SHAPE" in why


def test_the_se_diff_knife_edge_is_recorded_and_not_acted_on():
    """Found while banking the profile: R-clear's whole pair set turns
    on a 0.001058 gap."""
    record = phase22.SE_DIFF_SITS_ON_A_KNIFE_EDGE
    fact = _flat(record["the_fact"])
    assert "0.398942" in fact and "0.001058" in fact
    assert round(0.4 - 0.398942, 6) == 0.001058

    turns = _flat(record["what_turns_on_it"])
    assert "R-clear's ENTIRE PAIR SET" in turns
    assert "4,288 pairs would move" in turns
    assert "0.4018 rather than 0.2485" in turns
    # That alternative fraction is real: steps 0,1,2 of the profile.
    assert round((2345 + 4604 + 4288) / 27966, 4) == 0.4018

    little = _flat(record["how_little_it_would_take"])
    assert "0.265% larger" in little

    why = _flat(record["why_it_is_recorded_and_not_acted_on"])
    assert "choosing a pair set after seeing the distribution" in why
    assert "reported not repaired" in record["tag"]
    # It does not disturb the tie ruling.
    assert "below ANY positive threshold" in _flat(
        record["it_does_not_touch_the_tie_ruling"]
    )


def test_the_tie_rule_ruling_records_its_ground_and_both_rejections():
    record = phase22.TIE_RULE_RULED
    ground = _flat(record["the_ground"])
    assert "a tie carries NO ORDER" in ground
    assert "teaches nothing true" in ground
    assert "NEITHER AN ATTRACTOR NOR A KNOWINGLY-WRONG LABEL" in ground

    faithful = _flat(record["it_keeps_r_all_faithful_to_its_purpose"])
    assert "UNCERTAIN orders, not on ABSENT ones" in faithful
    assert "4,604 of the 6,949 contested pairs -- two-thirds" in faithful
    assert round(4604 / 6949, 4) == 0.6625

    b = _flat(record["candidate_b_rejected_by_name"])
    assert "reintroduces the attractor" in b
    assert "option 3's mechanism on a smaller set" in b
    assert "rather than relaxed because the set is smaller" in b

    c = _flat(record["candidate_c_rejected_by_name"])
    assert "WRONG ORDER to roughly HALF of 2,345 pairs" in c
    assert "1,173" in c
    assert "not a case for INVENTED ones" in c


def test_the_tie_rule_costs_r_clear_nothing_which_sharpens_the_contrast():
    """Ties are step 0, so they were already below SE_diff and already
    outside R-clear's set. The rule changes R-all only."""
    cost = _flat(phase22.TIE_RULE_RULED["what_it_costs"])
    assert "8.39%" in cost
    assert "R-clear loses none of its own pairs at all" in cost
    assert "changes R-all ONLY, which sharpens rather than blurs" in cost
    # R-syn has no ties either: four distinct magnitudes per face.
    assert "four distinct magnitudes per face" in cost


def test_the_settings_are_eleven_with_two_blocking_the_run():
    record = phase22.DECLARED_SETTINGS_22
    closed = record["closed_by_the_rulings"]
    conditional = record["ruled_after_the_lock_with_a_precondition"]
    eleventh = record["the_eleventh_setting_ruled_2026_09_01"]
    still_open = record["still_open_before_the_first_run"]
    assert len(closed) + len(conditional) + len(eleventh) == 9
    assert len(still_open) == 2
    numbers = sorted(
        int(k.split("_")[0])
        for k in list(closed) + list(conditional) + list(eleventh)
        + list(still_open)
    )
    assert numbers == list(range(1, 12)), "eleven settings, no gaps"

    assert "DROP TIED PAIRS FROM TRAINING" in _flat(eleventh["11_tie_rule"])

    blockers = _flat(record["the_two_that_remain_open_and_BLOCK_THE_RUN"])
    assert "settings 9 and 10" in blockers
    assert "CANNOT LAUNCH" in blockers
    assert "are NOT defaulted" in blockers
    assert "four cohort arms are unaffected and are launchable" in blockers


def test_the_contrast_family_enumerates_exactly_fifteen():
    family = phase22.contrast_family()
    assert len(family) == phase22.CONTRAST_FAMILY["count"] == 15
    assert len({c["key"] for c in family}) == 15, "no duplicate contrast"

    primaries = [c for c in family if c["kind"] == "primary"]
    secondaries = [c for c in family if c["kind"] == "secondary"]
    assert len(primaries) == 6 and len(secondaries) == 9

    # Every arm appears once as a primary, against the probe.
    assert sorted(c["a"] for c in primaries) == sorted(phase22.ARMS)
    assert {c["b"] for c in primaries} == {phase22.PROBE}
    assert phase22.PROBE in ladder.TRADE_OFF_PAIR["result"]

    # The three secondary groups, at their declared sizes.
    by_question = {}
    for c in secondaries:
        by_question.setdefault(c["question"], []).append(c)
    assert len(by_question["the head question"]) == 3
    assert len(by_question["the noise-pair question"]) == 2
    assert len(by_question["the axis question"]) == 4


def test_only_the_head_contrasts_cross_the_bounding():
    """Every other contrast is a single controlled factor -- which is
    the ground the family was confirmed on."""
    for contrast in phase22.contrast_family():
        if contrast["kind"] == "primary":
            continue
        a_bounded = contrast["a"].endswith("_bounded")
        b_bounded = contrast["b"].endswith("_bounded")
        if contrast["varies"] == "bounding":
            assert a_bounded != b_bounded, contrast["key"]
        else:
            assert a_bounded == b_bounded, (
                f"{contrast['key']} crosses the bounding without measuring it"
            )


# --------------------------------------------------------------------------
# the six arms: configs, task, and the two that cannot launch
# --------------------------------------------------------------------------


def test_the_six_arm_configs_exist_and_match_their_generator():
    import subprocess
    import sys

    names = sorted(p.name for p in (REPO / "configs").glob("p22_r_*.yaml"))
    assert names == [
        "p22_r_all_bounded.yaml", "p22_r_all_unbounded.yaml",
        "p22_r_clear_bounded.yaml", "p22_r_clear_unbounded.yaml",
        "p22_r_syn_bounded.yaml", "p22_r_syn_unbounded.yaml",
    ]
    result = subprocess.run(
        [sys.executable, "scripts/generate_phase22_configs.py", "--check"],
        cwd=REPO, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_each_arm_config_declares_its_two_factors_and_nothing_else_varies():
    import yaml

    tasks = {}
    for path in (REPO / "configs").glob("p22_r_*.yaml"):
        task = yaml.safe_load(path.read_text(encoding="utf-8"))["task"]
        tasks[task["arm"]] = task
    assert sorted(tasks) == sorted(phase22.ARMS)

    for arm, task in tasks.items():
        assert task["kind"] == "train_rank_cv"
        assert task["bounded"] is arm.endswith("_bounded")
        assert task["monitor"] == (
            "inner_val_mse" if task["bounded"] else "inner_val_pcc"
        )
        assert task["expect_patients"] == 237
        assert task["seeds"] == [1337, 2024, 7, 99, 12345]

    # R-all and R-clear at a fixed head differ in the PAIR SOURCE ALONE.
    # **[CORRECTED 2026-09-01] This is a CONFIG-LEVEL comparison and
    # nothing more.** It was reported last cycle as verifying that the
    # two arms differ in pair_source alone; it verifies that the two
    # FILES do. pair_source reached only a log line, so the arms were
    # identical while this passed. The behavioural check now lives in
    # tests/test_ranking.py (min_separation) and in the end-to-end pair
    # census -- exit criterion 3 needs both halves.
    for head in ("bounded", "unbounded"):
        a, b = tasks[f"r_all_{head}"], tasks[f"r_clear_{head}"]
        differing = {
            k for k in set(a) | set(b) if a.get(k) != b.get(k)
        }
        assert differing == {"arm", "pair_source", "se_diff_threshold"}, (
            head, differing
        )
        assert b["se_diff_threshold"] == 0.398942
        assert a.get("se_diff_threshold") is None


def test_the_r_syn_configs_carry_the_unruled_sentinel_and_cannot_launch():
    """Criterion 13, enforced mechanically: the two open settings reach
    the task as -1 and stop it by name."""
    import yaml

    for head in ("bounded", "unbounded"):
        task = yaml.safe_load(
            (REPO / "configs" / f"p22_r_syn_{head}.yaml")
            .read_text(encoding="utf-8")
        )["task"]
        assert task["pretrain_epochs"] == -1
        assert task["finetune_trainable_code"] == -1
    # And the cohort arms carry no sentinel at all.
    for arm in ("r_all_bounded", "r_clear_unbounded"):
        task = yaml.safe_load(
            (REPO / "configs" / f"p22_{arm}.yaml").read_text(encoding="utf-8")
        )["task"]
        assert "pretrain_epochs" not in task


def test_the_task_refuses_the_unruled_sentinel_by_name(tmp_path, clean_repo):
    import yaml

    from cleft.run import TASKS, RunContext

    source = yaml.safe_load(
        (REPO / "configs" / "p22_r_syn_bounded.yaml").read_text(
            encoding="utf-8"
        )
    )
    # Point the inputs at a throwaway dir so the refusal is about the
    # sentinel and not about missing artifacts.
    stub = tmp_path / "stub"
    stub.mkdir()
    from cleft.provenance.hashing import hash_dir

    source["inputs"] = [
        {"name": entry["name"], "path": str(stub),
         "rollup_sha256": hash_dir(stub)["rollup"]}
        for entry in source["inputs"]
    ]
    source["tier"] = "dev"
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="UNRULED sentinel"):
        with RunContext(path, tmp_path / "runs", repo_root=clean_repo) as ctx:
            TASKS["train_rank_cv"](ctx)


def test_the_ranking_objective_travels_in_backbone_config_not_a_new_vocabulary():
    """AXIS_RULED: train_cv's closed vocabulary and the shipped
    siamese_contrastive are both untouched."""
    import inspect

    from cleft.train import phase3

    factory = inspect.getsource(phase3.make_factory)
    assert 'config.get("objective") == "pairwise_logistic"' in factory
    assert "RankingHeadBackbone" in factory
    # It refuses to guess which arm it is.
    assert "SEPARATE REGISTERED ARMS" in factory

    # train_cv gained no field, and siamese_contrastive is unchanged.
    assert "objective" not in schema.TASK_SPECS["train_cv"]
    assert "bounded" not in schema.TASK_SPECS["train_cv"]
    assert schema.TASK_SPECS["siamese_contrastive"]["arm"].choices == (
        "p17_arm_b", "p17_arm_c",
    )
    assert schema.TASK_SPECS["siamese_contrastive"]["pair_rule"].choices == (
        "symmetric_if_zero_magnitude",
    )


# --------------------------------------------------------------------------
# negative space: registration only
# --------------------------------------------------------------------------


def test_exactly_the_six_registered_arms_are_built():
    """[UPDATED TWICE 2026-09-01] It first held the phase at zero
    machinery, then at the tie measurement alone. Both settings are now
    ruled through, so the six arms are built -- and what the pin holds
    is that they are the SIX REGISTERED ONES and nothing else."""
    from cleft.run import TASKS

    # The one thing that exists.
    assert "tie_fraction" in TASKS
    assert "tie_fraction" in schema.TASK_SPECS
    assert "p22_tie_fraction.yaml" in {
        p.name for p in (REPO / "configs").glob("p22_*.yaml")
    }

    # [UPDATED AGAIN 2026-09-01] The six arms ARE built now, as ruled.
    # What the pin holds is that they are the SIX registered ones and
    # nothing else: one task kind, six configs, no seventh arm.
    assert "train_rank_cv" in TASKS
    assert sorted(
        p.name for p in (REPO / "configs").glob("p22_r_*.yaml")
    ) == [f"p22_{arm}.yaml" for arm in sorted(phase22.ARMS)]
    assert len(phase22.ARMS) == 6

    # The module holds records plus the family enumeration -- the task
    # lives in run.py, as every other task does.
    callables = [
        n for n in dir(phase22)
        if callable(getattr(phase22, n)) and not n.startswith("_")
    ]
    assert callables == ["contrast_family", "summary"], callables


def test_the_summary_is_the_whole_restate():
    # [UPDATED 2026-09-01] Ten more: the five rulings, the registered
    # prediction, the monitor resolution, the rescaling note, the
    # settings enumeration and the lock.
    assert sorted(phase22.summary()) == [
        "arm_r_all",
        "arm_r_clear",
        "arm_r_syn",
        "attractor_finding",
        "axis_ruled",
        "bounded_head_prediction",
        "closing",
        "cohort_arms_observed",
        "condition_phenomenon_count",
        "config_surface",
        "contrast_family",
        "contrast_verdicts_observed",
        "contrasts_observed",
        "coverage",
        "declared_settings",
        "diagnostic",
        "diagnostic_observed",
        "exit_criteria",
        "exit_criteria_draft",
        "family_evaluable_subset",
        "five_arms_do_not_shrink",
        "gate_3_claim_corrected",
        "head_bounding_ruled",
        "head_ruled",
        "loss_ruled",
        "monitor_bind",
        "pair_construction_ruled",
        "pair_source_was_not_consumed",
        "pairing_axis_assumption",
        "postrun_reconstruction_audit",
        "r_all_r_clear_ground",
        "readings",
        "reckoning",
        "rescaling_note",
        "run_path_check_feasibility",
        "se_diff_knife_edge",
        "settings_consumption_sweep",
        "shrinkage_observed",
        "target_map_ruled",
        "the_stronger_mechanism_asserted",
        "tie_comparison_rule",
        "tie_fraction_measurement",
        "tie_fraction_observed",
        "tie_rule_ruled",
    ]
    # Every value is a record, and every record carries a tag.
    for key, value in phase22.summary().items():
        assert isinstance(value, dict), key
        assert "tag" in value, key
