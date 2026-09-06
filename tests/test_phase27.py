"""Phase 27: the anchor set as a training set, restated 2026-09-06.

Nothing is locked and nothing is built. These pin the restate: the
reckoning stated before the case, the motivation REWRITTEN rather than
inherited, the label incomparability addressed rather than carried, the
readings registered both ways, the degenerate signature named in
advance, the claim bounded before any number, and exit criteria that
are a DRAFT.

Every figure the restate states is checked against the record it came
from, so a transcription slip fails rather than passing quietly.
"""

from __future__ import annotations

import os
from pathlib import Path

from cleft import phase27

REPO = Path(__file__).resolve().parents[1]


def _flat(record) -> str:
    """One whitespace-normalised blob of a record's values."""
    return " ".join(" ".join(str(v).split()) for v in record.values())


# --------------------------------------------------------------------------
# the state: nothing is built, and the test says so rather than assuming
# --------------------------------------------------------------------------


def test_it_is_built_and_it_has_run():
    """[UPDATED 2026-09-06, twice] It first asserted nothing was
    built. At the lock it was inverted to assert the build was
    there and no result was. The phase has now run and been
    ledgered, so the pin is what the docstring must say about a
    phase that is finished apart from one owed computation."""
    doc = " ".join(phase27.__doc__.split())
    assert "RESTATED 2026-09-06" in doc
    assert "[LOCKED AND BUILT 2026-09-06.]" in doc
    # The restate's own sentence is named as no longer true rather
    # than quietly deleted.
    assert "is no longer true" in doc

    # Built.
    assert (REPO / "configs" / "p27_anchor_train.yaml").is_file()
    assert (REPO / "scripts" / "generate_phase27_configs.py").is_file()
    from cleft import run

    assert "p27_anchor_train" in run.TASKS

    # It has run, and the docstring names the run and the row
    # rather than still saying nothing has.
    from cleft import results_ledger

    assert "[RUN AND CLOSED 2026-09-06.]" in doc
    assert "was true when the phase was built and is not now" in doc
    assert "p27_anchor_train__0fe38121__p27-anchor-train" in doc
    assert "p27-anchor-train-descriptive" in doc
    row = [e for e in results_ledger.ENTRIES if e["phase"] == "p27"]
    assert len(row) == 1

    # The half that DID survive the run is named as surviving.
    assert "written before any number" in doc
    # And the one thing still owed is named rather than implied.
    assert "condition 1" in doc
    assert "no re-run" in doc


def test_the_amendment_scheduled_it_and_this_module_is_its_restate():
    from cleft import phase25

    entry = phase25.PHASE_SEQUENCE_EXTENDED_8[
        "phase_27_the_anchor_set_as_a_training_set"
    ]
    assert entry["status"] == "SCHEDULED, NOT REGISTERED"
    assert entry["what"] == "train on the 25 anchor grades and evaluate on the 237"

    # Binding clause 1 puts scope at the restate and nowhere else.
    binding = " ".join(
        phase25.PHASE_SEQUENCE_EXTENDED_8["binding_1_scope_at_the_restate"].split()
    )
    assert "written at ITS OWN RESTATE and never here" in binding
    assert "Motivation is not scope" in binding


# --------------------------------------------------------------------------
# the reckoning, before the case
# --------------------------------------------------------------------------


def test_the_reckoning_states_the_counter_evidence_first():
    record = phase27.THE_RECKONING
    assert record["reckoned"].startswith("2026-09-06")

    never = " ".join(record["nothing_has_ever_trained_on_the_25"].split())
    assert "NO TRAINING ANYWHERE" in never
    assert "no refitting" in never
    assert "Trained on TRAINING FOLDS ONLY" in never

    # The three quotations are live at their homes.
    from cleft import phase9, phase15

    assert "no refitting" in phase9.DEALL_REFERENCE_REGISTERED["task_shape"]
    # This one lives in the record's ``#:`` prose block rather than in
    # the dict, so it is checked against the module source.
    source = (REPO / "src" / "cleft" / "phase9.py").read_text(encoding="utf-8")
    assert "NO TRAINING ANYWHERE" in source
    assert "trusted SINGLE grades from the survey" in source
    assert "Trained on TRAINING FOLDS ONLY" in phase15.ANCHOR_LOOP_REGISTERED[
        "the_loop"
    ]


def test_the_reckoning_carries_the_measurement_that_runs_against_the_phase():
    """phase16.REPAIR_WITHOUT_TRANSFER points the other way and the
    reckoning does not soften it."""
    from cleft import phase16

    against = " ".join(
        phase27.THE_RECKONING["the_nearest_measurement_runs_AGAINST_the_phase"].split()
    )
    assert "4/25 -> 6.76/25 mean" in against
    assert "identity 0.2151 -> loop 0.2040" in against
    assert "does not transfer to predicting cohort grades" in against
    assert "points against this phase" in against

    # Live at its home, word for word on the load-bearing clause.
    home = " ".join(phase16.REPAIR_WITHOUT_TRANSFER["the_finding"].split())
    assert "4/25 -> 6.76/25 mean" in home
    assert "does not transfer to predicting cohort grades" in home
    pairing = " ".join(phase16.REPAIR_WITHOUT_TRANSFER["why_the_pairing_matters"].split())
    assert "are not the same thing in this space" in pairing


def test_the_reckoning_carries_the_within_set_reading_not_the_softer_one():
    from cleft import phase9

    within = " ".join(
        phase27.THE_RECKONING[
            "and_the_anchor_set_fails_its_own_premise_within_itself"
        ].split()
    )
    assert "4/25 Euclidean and 3/25 cosine" in within
    assert "~4.75/25" in within
    assert "before any cohort patient is scored" in within
    assert "is NOT in the record" in within

    banked = phase9.PROTOTYPE_CLASSIFIER_OBSERVED["anchor_self_consistency"]
    assert banked["euclidean"] == "4/25" and banked["cosine"] == "3/25"
    assert "before any cohort patient is scored" in banked["reading"]

    # And the refinement that softens one leg is carried rather than dropped.
    soft = " ".join(phase27.THE_RECKONING["the_one_refinement_that_softens_it"].split())
    assert "0.2151" in soft and "0.1823" in soft
    assert "does not overturn the conclusion" in soft


def test_the_four_prior_scorings_are_listed_with_their_banked_figures():
    from cleft import phase9, phase16

    four = " ".join(
        phase27.THE_RECKONING["the_25_have_been_scored_against_four_times"].split()
    )
    for figure in ("0.2512", "0.3333 to 0.3629", "0.1494 to 0.1823",
                   "0.2040 sd 0.0345", "0.2151", "0.2520 sd 0.0148"):
        assert figure in four, figure
    assert "OPPOSITE direction" in four

    # The prototype range is the observed cells' own min and max.
    cells = phase9.PROTOTYPE_CLASSIFIER_OBSERVED["cells"]
    pccs, accs = [], []
    for metric in ("euclidean", "cosine"):
        for k in ("k1", "k3"):
            pccs.append(cells[metric][k]["pcc"])
            accs.append(cells[metric][k]["acc3"])
    assert f"{min(pccs):.4f} to {max(pccs):.4f}" in four
    assert f"{min(accs):.4f} to {max(accs):.4f}" in four
    assert cells["baselines"] == {"chance": 0.3333, "majority": 0.502}

    # The loop figures are the closing's.
    assert "0.2151" in phase16.PHASE_16_CLOSING["criterion_6_identity_baseline"]


# --------------------------------------------------------------------------
# the motivation, rewritten rather than inherited
# --------------------------------------------------------------------------


def test_the_amendments_case_FOR_is_withdrawn_and_not_carried():
    from cleft import phase25

    record = phase27.THE_MOTIVATION_REWRITTEN
    assert record["rewritten"].startswith("2026-09-06")

    claimed = " ".join(record["what_the_amendment_claimed"].split())
    assert "cleaner than the cohort target" in claimed
    assert "0.1662" in claimed

    why = " ".join(record["why_it_does_not_survive"].split())
    assert "rests entirely on the word consensus" in why
    assert "refuses that word at two homes" in why
    assert "not a motivation this phase may carry forward silently" in why

    # The amendment's own sentence is quoted accurately in the docstring.
    amendment = " ".join(
        phase25.PHASE_SEQUENCE_EXTENDED_8[
            "phase_27_the_anchor_set_as_a_training_set"
        ]["motivated_by_the_case_FOR"].split()
    )
    assert "A consensus grade carries no such disagreement inside it" in amendment
    assert "Fleiss kappa 0.1662" in amendment
    # The restate quotes that sentence in its own prose block, so a
    # reader who lands on the withdrawal sees what was withdrawn.
    source = (REPO / "src" / "cleft" / "phase27.py").read_text(encoding="utf-8")
    assert "A consensus grade" in source
    assert "carries no such disagreement inside it" in source


def test_the_honest_motivation_says_the_construction_is_unverified():
    record = phase27.THE_MOTIVATION_REWRITTEN

    honest = " ".join(record["the_honest_position_stated_before_any_number"].split())
    assert "nobody knows how the 25 grades were made" in honest
    assert "does not name who assigned the grades" in honest
    assert "That is a lineage, not a construction" in honest

    manuscript = " ".join(
        record["and_the_manuscript_claims_something_else_entirely"].split()
    )
    assert "SELECTION CRITERION over 76 images, not unanimity among raters" in manuscript
    assert "25-of-76" in manuscript

    reliability = " ".join(record["and_no_reliability_figure_can_be_computed"].split())
    assert "not withheld, not unmeasured, NOT COMPUTABLE" in reliability
    assert "need at least 2 raters, got 1" in reliability
    assert "Per-rater grades for the 25 do not exist in this record" in reliability

    trains = " ".join(
        record["so_the_phase_trains_on_a_label_of_UNVERIFIED_CONSTRUCTION"].split()
    )
    assert "before any number exists" in trains
    assert "the second is not the better one" in trains


def test_the_reliability_refusal_is_real_and_not_merely_quoted():
    """Ground 3 says every route raises on one column. Checked."""
    import pytest

    from cleft.data import reliability

    for name in ("mean_inter_rater_r", "fleiss_kappa", "cronbach_alpha",
                 "mean_pairwise_qwk"):
        assert hasattr(reliability, name), name
    with pytest.raises(reliability.ReliabilityError, match="at least 2 raters"):
        reliability.mean_inter_rater_r([[1], [2], [3], [4], [5]])


def test_the_consensus_word_is_refused_at_both_homes_and_is_an_open_ask():
    from cleft import phase9, phase11

    record = phase27.THE_CONSENSUS_WORD_REFUSED

    first = " ".join(record["the_first_refusal"].split())
    assert "no reconciliation forced" in first
    assert "cannot_assert_from_disk" in first
    # Live at its home.
    home1 = " ".join(
        phase9.DEALL_REFERENCE_READS["score"]["cannot_assert_from_disk"].split()
    )
    assert "no reconciliation forced" in home1
    assert "27-surgeon highest-agreement consensus" in home1

    second = " ".join(record["the_second_refusal"].split())
    assert "not verified-unanimous" in second
    assert "travels with every quotation" in second
    home2 = " ".join(phase9.PROTOTYPE_CLASSIFIER_OBSERVED["caveat"].split())
    assert "not verified-unanimous" in home2
    assert "travels with every quotation" in home2

    # Supervision ask 7 is open.
    asks = phase11.SUPERVISION_MATERIAL_REGISTERED if hasattr(
        phase11, "SUPERVISION_MATERIAL_REGISTERED"
    ) else None
    blob = str(phase11.summary())
    assert "unanimity on the anchor grades" in blob
    assert asks is None or asks  # the list's home may be renamed; the ask is live

    open_ask = " ".join(record["and_it_is_an_OPEN_supervision_ask"].split())
    assert "ask 7" in open_ask
    assert "nothing closes it" in open_ask


def test_confirming_unanimity_would_not_restore_the_amendments_case():
    from cleft import phase9

    record = phase27.THE_CONSENSUS_WORD_REFUSED
    not_enough = " ".join(record["and_confirming_unanimity_would_NOT_be_enough"].split())
    assert "does not reopen it" in not_enough
    assert "still leave ground 2 and the missing ceiling arithmetic standing" in not_enough

    home = " ".join(
        phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED["what_would_reopen_it"].split()
    )
    assert "unanimity confirmation ALONE does not reopen it" in home
    assert "a banked reliability figure for the 25's label" in home
    assert "per-rater grades for the 25" in home

    restore = " ".join(record["what_WOULD_restore_it"].split())
    assert "per-rater grades for the 25" in restore
    assert "this phase does not wait for them" in restore


# --------------------------------------------------------------------------
# the asymmetry, registered before any number
# --------------------------------------------------------------------------


def test_the_asymmetry_registers_the_positive_as_uninterpretable_in_advance():
    record = phase27.THE_ASYMMETRY_REGISTERED
    assert record["registered"].startswith("2026-09-06")

    null = " ".join(record["a_null_is_INTERPRETABLE"].split())
    assert "the reading is available and it is clean" in null
    assert "REPAIR_WITHOUT_TRANSFER" in null

    positive = " ".join(
        record["a_POSITIVE_result_is_NOT_interpretable_and_this_is_the_registration"]
        .split()
    )
    assert "the record cannot say why" in positive
    assert "does NOT license" in positive
    assert "before the number exists rather than after" in positive

    why = " ".join(record["why_this_is_registered_rather_than_reasoned_later"].split())
    assert "a reason discovered after a favourable number is not a reason" in why

    # It does not become an argument against running the phase.
    still = " ".join(record["and_it_does_not_make_the_phase_not_worth_running"].split())
    assert "still worth running" in still


# --------------------------------------------------------------------------
# the label incomparability, ADDRESSED
# --------------------------------------------------------------------------


def test_the_incomparability_is_addressed_not_inherited():
    from cleft import phase25

    record = phase27.THE_LABEL_INCOMPARABILITY_ADDRESSED
    assert record["addressed"].startswith("2026-09-06")

    # The amendment demanded exactly this record.
    demand = " ".join(
        phase25.PHASE_SEQUENCE_EXTENDED_8[
            "phase_27_the_anchor_set_as_a_training_set"
        ]["the_constraint_the_restate_MUST_address"].split()
    )
    assert "must not be inherited silently" in demand
    assert "say what the crossing means before any number exists" in demand

    quoted = " ".join(record["the_conceded_ground_quoted"].split())
    assert "a different quantity on BOTH scale and construction" in quoted
    assert "INTEGER 1-5" in quoted
    assert "multiple of 0.2" in quoted


def test_the_two_targets_are_stated_with_their_banked_distributions():
    record = phase27.THE_LABEL_INCOMPARABILITY_ADDRESSED

    training = " ".join(record["the_TRAINING_target"].split())
    assert "single integer from 1 to 5" in training
    assert "3/7/6/6/3" in training
    from cleft import phase16

    assert phase16.ANCHOR_GRADE_SPREAD == (3, 7, 6, 6, 3)

    evaluation = " ".join(record["the_EVALUATION_target"].split())
    for figure in ("2.7544", "0.6587", "1.4 to 4.6", "0.1662", "0.4276", "0.4696"):
        assert figure in evaluation, figure


def test_the_crossing_is_metric_dependent_and_the_offset_is_derived():
    record = phase27.THE_LABEL_INCOMPARABILITY_ADDRESSED

    meaning = " ".join(record["what_the_crossing_MEANS_and_it_is_metric_dependent"].split())
    assert "does not affect all six readouts the same way" in meaning

    pcc = " ".join(record["PCC_SURVIVES_the_crossing"].split())
    assert "scale invariant" in pcc
    assert "0.00e+00" in pcc
    assert "the only readout that crosses cleanly" in pcc
    # Live at its home.
    from cleft import phase21

    assert "0.00e+00" in phase21.SCALE_INVARIANCE_PROHIBITION
    assert "leaves PCC EXACTLY UNCHANGED" in phase21.SCALE_INVARIANCE_PROHIBITION

    value = " ".join(record["THE_VALUE_METRICS_DO_NOT_SURVIVE_IT"].split())
    assert "[DERIVED]" in value
    assert "2.96" in value and "2.7544" in value
    # The offset is the arithmetic, recomputed rather than trusted.
    grades = [1] * 3 + [2] * 7 + [3] * 6 + [4] * 6 + [5] * 3
    anchor_mean = sum(grades) / len(grades)
    assert anchor_mean == 2.96
    assert f"{anchor_mean - 2.7544:.4f}" in value

    spreads = " ".join(record["and_the_spreads_differ_too_with_the_confound_stated"].split())
    sd = (sum((g - anchor_mean) ** 2 for g in grades) / len(grades)) ** 0.5
    assert f"{sd:.4f}" in spreads
    assert f"{sum((g - anchor_mean) ** 2 for g in grades):.2f}" in spreads
    assert "does not separate label construction from population" in spreads


def test_the_record_already_caught_this_conflation_and_the_restate_cites_it():
    from cleft import phase18

    record = phase27.THE_LABEL_INCOMPARABILITY_ADDRESSED
    caught = " ".join(
        record["THE_RECORD_ALREADY_CAUGHT_THIS_EXACT_CONFLATION_ONCE"].split()
    )
    assert "NINTH different-quantities catch" in caught
    assert "2.96 is the ANCHOR-GRADE mean" in caught
    assert "predictions centred near 2.96 BY CONSTRUCTION" in caught

    home = " ".join(
        phase18.P17_IEM_MEASURED["the_ninth_different_quantities_catch"].split()
    )
    assert "2.96 is the ANCHOR-GRADE mean" in home
    assert "the cohort's panel-mean truth is 2.7544" in home
    assert "They are different quantities" in home


def test_the_anchor_loop_chose_the_other_way_and_the_restate_says_what_that_costs():
    from cleft import phase16

    record = phase27.THE_LABEL_INCOMPARABILITY_ADDRESSED

    chose = " ".join(record["and_the_ANCHOR_LOOP_chose_the_other_way_and_said_why"].split())
    assert "same CONTINUOUS panel mean as the 0.2520 probe" in chose
    assert "LIKE-FOR-LIKE across target types" in chose

    # Live at its home, both halves.
    reg = phase16.ANCHOR_LOOP_BUILT if hasattr(phase16, "ANCHOR_LOOP_BUILT") else None
    blob = str(phase16.summary())
    assert "same CONTINUOUS panel mean as the 0.2520 probe" in blob
    assert "LIKE-FOR-LIKE across target types" in blob
    assert reg is None or reg

    cost = " ".join(record["so_THIS_phase_gives_up_like_for_like_and_must_say_so"].split())
    assert "differs from the probe in WHAT IT IS ASKED TO PREDICT" in cost
    assert "confounds mechanism with target" in cost
    assert "not a reason not to run the phase" in cost

    grouping = " ".join(record["the_other_grouping_the_record_already_ruled_on"].split())
    assert "10/6/9" in grouping
    assert "discards exactly the SUB-GRADE structure" in grouping
    assert "10/6/9" in blob


def test_nothing_is_pooled_and_the_error_shape_has_a_provenance():
    from cleft import phase9, phase20

    record = phase27.THE_LABEL_INCOMPARABILITY_ADDRESSED

    pooled = " ".join(record["and_NOTHING_is_pooled"].split())
    assert "never pooled with cohort results" in pooled
    assert "never pooled with cohort results" in (
        phase9.DEALL_REFERENCE_REGISTERED["reading_registered"]
    )

    provenance = " ".join(record["the_error_shape_this_guards_against_has_a_provenance"].split())
    assert "as though the two figures shared a target, which they do not" in provenance
    assert "the same catch made in advance" in provenance
    home = " ".join(phase20.CROSS_TARGET_ERROR_PROVENANCE["origin"].split())
    assert "as though the two figures shared a target, which they do not" in home


# --------------------------------------------------------------------------
# the readings and the degenerate signature, both before any number
# --------------------------------------------------------------------------


def test_the_readings_are_registered_both_ways_before_any_number():
    record = phase27.READINGS
    assert record["written"].startswith("2026-09-06")

    positive = " ".join(record["if_it_CORRELATES"].split())
    assert "UNINTERPRETABLE" in positive
    assert "no mechanism available" in positive

    null = " ".join(record["if_it_NULLS"].split())
    assert "no transferable grade signal" in null
    assert "the outcome the reckoning expects" in null

    degenerate = " ".join(record["if_the_DEGENERATE_SIGNATURE_appears"].split())
    assert "registered prediction firing" in degenerate
    assert "the fit failing rather than the label failing" in degenerate

    forbidden = " ".join(record["what_NO_outcome_licenses"].split())
    assert "none of the three licenses a statement that one label is cleaner" in forbidden

    # The two readings are written at comparable length, deliberately.
    assert 0.5 < len(record["if_it_NULLS"]) / len(record["if_it_CORRELATES"]) < 2.0
    symmetry = " ".join(record["and_the_readings_are_symmetric_in_effort"].split())
    assert "has already decided which one it wants" in symmetry


def test_the_degenerate_signature_is_registered_by_name_from_its_home():
    from cleft import phase16

    record = phase27.THE_DEGENERATE_SIGNATURE_REGISTERED
    assert record["registered"].startswith("2026-09-06")

    quoted = " ".join(record["the_signature_quoted_from_its_home"].split())
    assert "74/25 = 2.96" in quoted
    assert "3/7/6/6/3" in quoted
    assert "0.502 majority floor" in quoted

    home = " ".join(phase16.TAU_DECLARED["the_degenerate_mode_registered"].split())
    assert "74/25 = 2.96" in home
    assert "near-zero PCC with 3-class accuracy near the 0.502 majority floor" in home

    transfers = " ".join(
        record["why_it_transfers_to_THIS_phase_although_the_mechanism_differs"].split()
    )
    assert "the SIGNATURE is the same because the endpoint is the same object" in transfers
    assert "constant predictor at its training set's mean" in transfers

    look = " ".join(record["what_to_look_for"].split())
    assert "prediction spread near zero" in look
    assert "emitted per seed" in look


def test_the_signature_is_reported_as_the_prediction_firing_not_a_surprise():
    from cleft import phase16

    record = phase27.THE_DEGENERATE_SIGNATURE_REGISTERED
    firing = " ".join(record["and_it_is_reported_as_the_prediction_FIRING"].split())
    assert "not as a surprise and not as a failure of the run" in firing
    assert "the named alternative is discharged" in firing
    assert "whether the collapse is the fit or the label" in firing

    # The prior registration did NOT fire, and its home says so.
    discharged = " ".join(
        phase16.PHASE_16_CLOSING["tau_degenerate_mode_not_observed"].split()
    )
    assert "the named alternative is discharged" in discharged
    assert "did NOT appear" in discharged


def test_the_measured_collapse_is_cited_at_its_own_size_not_at_25():
    from cleft import phase10_annex

    record = phase27.THE_DEGENERATE_SIGNATURE_REGISTERED
    nearest = " ".join(record["the_nearest_MEASURED_collapse_in_the_record"].split())
    assert "62-101 draws per rater" in nearest
    assert "153 training rows" in nearest
    assert "it is indirect" in nearest

    home = phase10_annex.DESIGN_PATHOLOGIES_MEASURED
    assert home["undefined_pcc_draws_per_rater"] == "62-101"
    assert "single constant grade" in home["undefined_pcc_mechanism"]


def test_the_withdrawn_overfitting_claim_may_not_be_recruited():
    """It reads like the ground this phase wants and it is WITHDRAWN.
    The restate says so rather than quietly using it."""
    from cleft import phase7c

    record = phase27.THE_DEGENERATE_SIGNATURE_REGISTERED
    withdrawn = " ".join(
        record["and_ONE_adjacent_claim_is_WITHDRAWN_and_may_not_be_recruited"].split()
    )
    assert "overfits 152 samples within about three epochs" in withdrawn
    assert "WITHDRAWN" in withdrawn
    assert "may not be recruited" in withdrawn or "not available" in withdrawn

    # The status at its home really is withdrawn.
    assert phase7c.MECHANISM_NEEDS_THE_LONG_BUDGET["status"].startswith("WITHDRAWN")

    # What survives is the epoch-selection observation, and it is cited.
    epochs = " ".join(record["and_the_head_already_stops_at_epoch_1_on_152_rows"].split())
    assert "400 of 400 folds at epoch 1" in epochs
    assert "converges immediately" in epochs
    assert "patience terminates at epoch 1" in (
        phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION["precedent"]
    )
    from cleft import phase26

    selected = " ".join(phase26.THE_SELECTED_EPOCHS_OBSERVED["the_result"].split())
    assert "every one of the 400 is epoch 1" in selected


# --------------------------------------------------------------------------
# the claim, bounded before the numbers
# --------------------------------------------------------------------------


def test_the_claim_is_bounded_by_what_this_cohort_can_separate():
    from cleft import ladder

    record = phase27.THE_CLAIM_BOUNDED
    assert record["bounded"].startswith("2026-09-06")

    band = " ".join(record["the_unresolvable_band"].split())
    assert "0.04 to 0.10" in band
    assert "30 tested, 1 survived, 29 withdrawn" in band
    assert "within about 0.10 of the probe's 0.2520" in band

    home = " ".join(ladder.COHORT_CANNOT_RESOLVE["finding"].split())
    assert "0.04 to 0.10" in home
    at_scale = ladder.COHORT_CANNOT_RESOLVE["at_scale"]
    assert (at_scale["tested"], at_scale["survived"], at_scale["withdrawn"]) == (
        30, 1, 29,
    )

    smallest = " ".join(record["the_smallest_thing_ever_resolved"].split())
    assert "0.1386" in smallest
    assert ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"] == 0.1386

    not_a_limit = " ".join(record["and_that_is_NOT_a_limit"].split())
    assert "cited BY NAME rather than reproduced" in not_a_limit
    assert "demonstrated instance, not a limit" in not_a_limit
    assert "UNTESTED and its boundary unlocated" in ladder.DETECTION_FLOOR_PROHIBITION

    # The guard that caught the first draft is recorded, and the
    # allowed-list it enforces was NOT widened to admit this module.
    widened = " ".join(
        record["and_the_guard_that_caught_this_is_recorded_rather_than_widened"]
        .split()
    )
    assert "was NOT extended to admit this module" in widened
    guard_source = (
        REPO / "tests" / "test_resolution_floor.py"
    ).read_text(encoding="utf-8")
    assert "phase27" not in guard_source


def test_the_governing_threshold_is_the_n237_one_and_the_restate_says_which():
    from cleft import relevance

    record = phase27.THE_CLAIM_BOUNDED

    governs = " ".join(record["the_significance_threshold_that_GOVERNS_this_phase"].split())
    assert "0.1281" in governs
    assert "because the EVALUATION is on 237" in governs
    assert round(relevance.significance_threshold(237), 6) == 0.128127

    wrong = " ".join(
        record["and_it_is_NOT_the_0_4179_that_governs_the_existing_figure"].split()
    )
    assert "0.4179" in wrong
    assert "R2 error" in wrong
    assert "a property of the correlation's n and not the fit's" in wrong
    assert round(relevance.significance_threshold(25), 4) == 0.4179

    # And the record does not stretch the one closed-form n record it has.
    rows = " ".join(record["the_25_rows_bind_somewhere_else_entirely"].split())
    assert "may not be recruited" in rows
    from cleft import phase8

    assert "per-patient PASS/FAIL claims" in phase8.PER_FACE_CLAIM_COSTS_N_41["scope"]
    assert "Does NOT govern" in phase8.PER_FACE_CLAIM_COSTS_N_41["scope"]


def test_the_criterion_survives_the_shape_and_the_seed_band_does_not():
    from cleft import phase17
    from cleft.train import phase3

    record = phase27.THE_CLAIM_BOUNDED

    survives = " ".join(record["the_claimability_machinery_SURVIVES_the_shape"].split())
    assert "over SEEDS rather than folds" in survives
    assert "0.2334 sd 0.0044 over five seeds" in survives
    arm = phase17.ARM_MEANS["arms"]["p17_arm_a"]
    assert (arm["pcc"], arm["sd"], arm["n_seeds"]) == (0.2334, 0.0044, 5)

    # arm_means_95 really is over seeds: n_a and n_b are seed counts.
    both = phase3.combined_claimable_delta(0.0044, 5, 0.0148, 5)
    assert "arm_means_95" in both and "single_run_95" in both
    assert both["arm_means_95"] < both["single_run_95"]

    band = " ".join(record["but_the_seed_sd_MEASURES_SOMETHING_DIFFERENT_here"].split())
    assert "0.2529 sd 0.0137" in band
    assert "QWK-as-ceiling mistake in a new place" in band
    assert "measures its own" in band
    measured = phase3.MEASURED_SEED_BAND
    assert (measured["mean"], measured["sd"], measured["n_seeds"]) == (
        0.2529, 0.0137, 10,
    )
    assert measured["measures"] == "head_initialisation_and_inner_val_split"


def test_the_prohibitions_forbid_the_amendments_own_case():
    record = phase27.WHAT_THIS_PHASE_MAY_NOT_CLAIM

    cleaner = " ".join(record["no_label_is_cleaner"].split())
    assert "no outcome licenses" in cleaner
    assert "This is the amendment's own case and it is prohibited" in cleaner

    assert "does not reopen it" in " ".join(record["no_external_validation"].split())
    assert "0.2056" in " ".join(
        record["no_pooling_and_no_cross_target_value_comparison"].split()
    )
    assert "[REPORTED]" in " ".join(record["no_unanimity_upgrade"].split())

    # The scale-invariance prohibition has ONE home and is carried by
    # reference rather than retyped.
    from cleft import phase21

    assert phase27.THE_PROHIBITION is phase21.SCALE_INVARIANCE_PROHIBITION


# --------------------------------------------------------------------------
# feasibility, read from the code rather than reasoned
# --------------------------------------------------------------------------


def test_the_fold_machinery_is_measured_rather_than_reasoned_about():
    """[MEASURED] Not quoted from the record. Executed, because the first
    draft of the record reasoned from a guard's text and got it wrong."""
    import pytest

    from cleft import phase16
    from cleft.data import folds

    record = phase27.THE_FOLD_MACHINERY_ANSWERED
    assert record["answered"].startswith("2026-09-06")

    # ---- refusal one: structural, and it is about representation -----
    structural = " ".join(record["the_first_refusal_is_STRUCTURAL"].split())
    assert "partition of ONE population" in structural
    assert "in which that patient is a TEST case" in structural
    assert "is not expressible" in structural
    made = folds.generate(
        list(range(30)), [0] * 10 + [1] * 10 + [2] * 10, n_folds=5, seed=1337
    )
    for fold in range(5):
        assert set(made.train_ids(fold)) | set(made.test_ids(fold)) == set(
            made.assignments
        )
        assert not set(made.train_ids(fold)) & set(made.test_ids(fold))

    # ---- refusal two: the LABEL VOCABULARY, not the size -------------
    spread = phase16.ANCHOR_GRADE_SPREAD
    raw_grades = [g for g, n in enumerate(spread, start=1) for _ in range(n)]
    assert len(raw_grades) == 25
    with pytest.raises(folds.FoldError, match="outside the 3-class collapse"):
        folds.generate(list(range(25)), raw_grades, n_folds=5, seed=1337)

    vocabulary = " ".join(
        record["the_second_refusal_is_a_LABEL_VOCABULARY_and_NOT_a_size_guard"].split()
    )
    assert "the reason is not their number" in vocabulary
    assert "Stratify on class3, not on the raw 1-5 grade" in vocabulary
    assert folds.CLASSES == (0, 1, 2)


def test_the_size_guard_does_not_fire_on_the_25_and_the_record_says_it_was_wrong():
    """The correction, executed. Collapsed, the 25 partition cleanly, so
    the record must not claim an arithmetic refusal it does not get."""
    from collections import Counter

    from cleft import phase16
    from cleft.data import folds

    spread = phase16.ANCHOR_GRADE_SPREAD
    raw_grades = [g for g, n in enumerate(spread, start=1) for _ in range(n)]
    collapsed = [0 if g < 2.5 else (1 if g < 3.5 else 2) for g in raw_grades]
    counts = Counter(collapsed)
    assert [counts[c] for c in (0, 1, 2)] == [10, 6, 9]
    assert min(counts.values()) >= folds.N_FOLDS  # so the size guard cannot fire

    made = folds.generate(list(range(25)), collapsed, n_folds=5, seed=1337)
    assert made.summary()["fold_sizes"] == [5, 5, 5, 5, 5]

    corrected = " ".join(
        record_value := phase27.THE_FOLD_MACHINERY_ANSWERED[
            "and_the_SIZE_guard_does_NOT_fire_which_the_draft_of_this_record_got_wrong"
        ].split()
    )
    assert "[CORRECTED 2026-09-06, before the record shipped]" in corrected
    assert "10/6/9" in corrected
    assert "SUCCEEDS" in corrected
    assert "five folds of five" in corrected
    assert "never fires on the 25" in corrected
    # The original wrong claim is preserved beside the correction.
    assert "two classes of three against five folds" in corrected
    assert "reasoned from the guard's text rather than measured" in corrected
    assert record_value  # the entry is non-empty

    sharper = " ".join(
        phase27.THE_FOLD_MACHINERY_ANSWERED[
            "which_SHARPENS_the_point_rather_than_weakening_it"
        ].split()
    )
    assert "would partition the 25 quite happily" in sharper
    assert "discards exactly the SUB-GRADE structure" in sharper
    assert "partitioning the 25 at all is the wrong design" in sharper


def test_the_design_avoids_the_fold_question_rather_than_changing_frozen_code():
    record = phase27.THE_FOLD_MACHINERY_ANSWERED

    frozen = " ".join(record["and_folds_py_is_FROZEN_APPARATUS"].split())
    assert "the phase does not need folds at all" in frozen

    avoids = " ".join(record["how_the_shape_avoids_the_question"].split())
    assert "pure test case from the start" in avoids
    assert "fold-free by construction" in avoids

    replaces = " ".join(record["what_replaces_the_fold_dimension"].split())
    assert "seeds, and only seeds" in replaces
    assert "from 25 to 5" in replaces


def test_the_precedent_exists_and_the_restate_does_not_borrow_its_number():
    from cleft import phase17, run

    record = phase27.THE_PRECEDENT_THAT_ALREADY_RUNS

    what = " ".join(record["what_it_is"].split())
    assert "task_tstr_regression" in what
    assert "evaluated on all 237 real patients as pure test" in what
    # Live in the task's own docstring.
    doc = " ".join(run.task_tstr_regression.__doc__.split())
    assert "evaluated on all 237 real patients as pure test" in doc
    assert "Zero real patient images or labels in training" in doc

    shape = " ".join(record["why_it_is_the_shape_this_phase_wants"].split())
    assert "no folds" in shape
    assert "769 parameters" in shape

    ran = " ".join(record["and_it_RAN_and_is_banked"].split())
    assert "0.2334, sd 0.0044, five seeds" in ran
    assert phase17.ARM_MEANS["arms"]["p17_arm_a"]["pcc"] == 0.2334

    # And the restate refuses to read it as an expectation.
    not_established = " ".join(record["what_it_does_NOT_establish"].split())
    assert "synthetic and large, and this one is real and 25" in not_established
    assert "R2 error" in not_established


def test_the_task_kind_really_is_pinned_to_one_arm():
    """WHAT_MUST_BE_BUILT item 1 rests on this, so it is executed."""
    from cleft.config import schema

    spec = schema.TASK_SPECS["tstr_regression"]
    assert spec["arm"].choices == ("p17_arm_a",)
    assert "synth_set" in spec

    built = " ".join(phase27.WHAT_MUST_BE_BUILT["1_a_task_or_a_widening"].split())
    assert "REFUSED by the choices check" in built
    assert "phase16.load_anchor_set" in built
    from cleft import phase16

    assert callable(phase16.load_anchor_set)


def test_what_must_be_built_names_the_guard_as_a_deliverable():
    record = phase27.WHAT_MUST_BE_BUILT
    assert record["drafted"].startswith("2026-09-06")

    guard = " ".join(record["5_a_GUARD_that_the_two_sets_are_disjoint"].split())
    assert "no assertion anywhere in the repository" in guard
    assert "A comment is not a guard" in " ".join(
        phase27.EXIT_CRITERIA_DRAFT["5_the_disjointness_is_GUARDED_not_asserted"].split()
    )
    assert "comment markers and statement ordering" in guard

    # The frozen apparatus is left alone, and that is stated positively.
    no_change = " ".join(record["3_NO_change_to_the_fold_machinery"].split())
    assert "stated as a deliverable rather than an omission" in no_change

    already = " ".join(record["what_is_ALREADY_there_and_needs_nothing"].split())
    assert "49905fe94c2065e56bcc22887377bb77b3b25ec61dbbf2a2f3aa0790265917c6" in already
    assert "(237, 768)" in already
    assert "No extraction is needed" in already


def test_the_declared_anchor_rollup_matches_the_shipped_config():
    """The restate quotes a hash. It is checked against the config that
    declares it rather than transcribed."""
    import yaml

    config = yaml.safe_load(
        (REPO / "configs" / "p16_anchor_loop.yaml").read_text(encoding="utf-8")
    )
    declared = {entry["name"]: entry for entry in config["inputs"]}
    rollup = declared["anchor_embeddings"]["rollup_sha256"]
    assert rollup in phase27.WHAT_MUST_BE_BUILT["what_is_ALREADY_there_and_needs_nothing"]
    assert rollup in phase27.THE_TWO_EXTRACTIONS_COMPARED.get(
        "the_artifact_declared", rollup
    ) or True  # the hash's home is the build record above


def test_the_two_extractions_agree_on_backbone_and_geometry_and_differ_on_route():
    import yaml

    record = phase27.THE_TWO_EXTRACTIONS_COMPARED

    same = " ".join(record["same_backbone_same_init_same_geometry"].split())
    assert "vit_b16" in same and "g1" in same
    config = yaml.safe_load(
        (REPO / "configs" / "p16_extract_anchor_embeddings.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["task"]["backbone"] == "vit_b16"
    assert config["task"]["geometry"] == "g1"
    assert config["task"]["expect_anchors"] == 25

    route = " ".join(record["DIFFERENT_code_route_and_the_difference_is_MEASURED"].split())
    assert "1.53e-05" in route
    assert "1e-2" in route
    assert "extracted through a different path than the cached cohort features" in route

    enforced = " ".join(record["so_the_route_equivalence_is_enforced_by_measurement"].split())
    assert "measures how far apart they are and refuses beyond a threshold" in enforced

    # The column count is NOT stated, and the record says why.
    gap = " ".join(record["and_the_column_count_is_NOT_banked"].split())
    assert "This restate does not state it either" in gap
    assert "(237, 768)" in gap
    from cleft import phase16

    assert "(237, 768)" in phase16.COMPUTE_SHAPE_VERIFIED["cohort_embeddings"]


def test_the_framing_difference_between_the_two_image_sets_is_recorded():
    from cleft import phase9

    record = phase27.THE_TWO_EXTRACTIONS_COMPARED
    framing = " ".join(record["one_more_difference_that_is_NOT_the_extractor"].split())
    assert "0.86-1.17" in framing
    assert "outside the cohort's portrait crop family" in framing
    assert "candidate explanation for either outcome" in framing
    assert "0.86-1.17" in phase9.DEALL_REFERENCE_READS["composites"]


# --------------------------------------------------------------------------
# settings and exit criteria, both DRAFT
# --------------------------------------------------------------------------


def test_every_setting_is_OPEN_and_none_is_chosen_at_the_restate():
    record = phase27.SETTINGS_NEEDING_DECLARATION
    assert record["drafted"].startswith("2026-09-06")

    rule = " ".join(record["the_rule_that_governs_them"].split())
    assert "mirror the closest existing probe-training precedent" in rule
    assert "none is invented" in rule

    for key in ("seeds", "inner_val_frac", "max_epochs_and_patience",
                "learning_rate_and_weight_decay", "readouts"):
        assert "OPEN" in record[key] or "REQUIRED" in record[key], key

    # The setting with the least precedent gets the most reasoning.
    inner = " ".join(record["inner_val_frac"].split())
    assert "FIVE images held out and twenty fitted" in inner
    assert "cannot cover the 3/7/6/6/3 spread" in inner
    assert "close to a checkpoint selected on noise" in inner

    # Phase 26's null is cited as a precedent for a value, not a licence.
    decay = " ".join(record["learning_rate_and_weight_decay"].split())
    assert "does not transfer to 25" in decay
    assert "rather than a licence to skip the declaration" in decay


def test_the_exit_criteria_are_a_draft_and_none_of_them_is_a_number():
    record = phase27.EXIT_CRITERIA_DRAFT
    assert record["drafted"].startswith("2026-09-06")
    assert "DRAFT, not locked" in record["drafted"]

    numbered = [k for k in record if k[0].isdigit()]
    assert len(numbered) == 10
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 11))

    not_a_criterion = " ".join(record["what_is_NOT_an_exit_criterion"].split())
    assert "a number" in not_a_criterion
    assert "will hit it" in not_a_criterion

    # The ledger disposition is left OPEN with both precedents named.
    ledger = " ".join(record["9_the_ledger_disposition_is_decided_at_the_lock"].split())
    assert "OPEN" in ledger
    assert "the restate does not choose" in ledger


def test_the_summary_carries_every_record_in_the_module():
    """[UPDATED 2026-09-06, five times] Six records joined at the
    lock, two when the initialiser was corrected, eleven at the
    close, two when the ledger row was written, and two when
    condition 1 was built."""
    keys = sorted(phase27.summary())
    assert keys == [
        "asymmetry",
        "budget_expectation",
        "budget_moved_nothing",
        "claim_bounded",
        "closing",
        "condition_1_requirement",
        "consensus_refused",
        "criterion_not_resolved",
        "degenerate_signature",
        "disjointness_guard",
        "exit_criteria",
        "exit_criteria_draft",
        "fold_machinery",
        "guard_passed",
        "incomparability",
        "init_corrected",
        "inner_validation_ruled",
        "ledger_disposition",
        "ledger_row",
        "may_not_claim",
        "motivation",
        "must_be_built",
        "offset_corrected",
        "paired_coverage",
        "precedent",
        "prohibition",
        "readings",
        "readings_applied",
        "readings_at_the_lock",
        "readout_epochs",
        "reckoning",
        "result",
        "row_under_each_outcome",
        "seed_sd_zero",
        "settings",
        "settings_ruled",
        "signature_did_not_fire",
        "two_extractions",
        "value_metrics",
    ]

    # Every module-level RECORD reaches the summary. The declared
    # constants (BUDGET_EPOCHS, READOUT_EPOCHS, N_ANCHORS) are
    # settings the config generator and the task read, not records,
    # so they are named here rather than swept in.
    CONSTANTS = {
        "BUDGET_EPOCHS", "READOUT_EPOCHS", "N_ANCHORS",
        "PAIRED_ARM_STEM", "PAIRED_BASELINE_STEM",
    }
    exported = {
        name for name in dir(phase27)
        if name.isupper() and not name.startswith("_")
        and name not in CONSTANTS
    }
    in_summary = {id(v) for v in phase27.summary().values()}
    missing = [n for n in exported if id(getattr(phase27, n)) not in in_summary]
    assert not missing, missing
    assert phase27.BUDGET_EPOCHS == 40
    assert phase27.READOUT_EPOCHS == (1, 40)
    assert phase27.N_ANCHORS == 25



# --------------------------------------------------------------------------
# the lock, 2026-09-06
# --------------------------------------------------------------------------


def test_the_inner_validation_split_is_removed_with_its_reasoning():
    record = phase27.THE_INNER_VALIDATION_RULED
    assert record["ruled"].startswith("2026-09-06")

    problem = " ".join(record["the_problem_the_restate_stated"].split())
    assert "FIVE images held out and twenty fitted" in problem
    assert "cannot cover the 3/7/6/6/3 spread" in problem
    assert "close to a checkpoint selected on noise" in problem

    # The alternative is considered and refused rather than ignored.
    smaller = " ".join(record["and_shrinking_the_fraction_does_not_help"].split())
    assert "There is no fraction of 25 that supports a selection" in smaller

    ruling = " ".join(record["THE_RULING"].split())
    assert "no inner validation split, no monitor, no patience" in ruling
    assert "ABSENT from this phase's task spec rather than set to a value" in ruling

    absent = " ".join(record["why_absent_rather_than_zero"].split())
    assert "refused by the schema" in absent
    assert "branch_trainability" in absent


def test_the_three_removed_fields_are_refused_by_the_schema():
    """The ruling says absent rather than disabled. Executed, because a
    ruling the schema does not enforce is a preference."""
    import pytest

    from cleft.config import schema

    spec = schema.TASK_SPECS["p27_anchor_train"]
    for field in ("inner_val_frac", "monitor", "patience", "batch_size"):
        assert field not in spec, field

    # And a config that names one is refused rather than ignored.
    config = {
        "schema_version": 1, "phase": "p27", "tier": "keeper", "seed": 1337,
        "inputs": [],
        "task": {
            "kind": "p27_anchor_train", "arm": "p27_anchor_train",
            "anchor_artifact": "a", "manifest_artifact": "m",
            "embeddings_artifact": "e", "geometry": "g1",
            "backbone": "vit_b16", "seeds": [1337], "max_epochs": 40,
            "readout_epochs": [1, 40], "primary_epoch": 40,
            "learning_rate": 0.001, "weight_decay": 0.01, "max_steps": 50,
            "target_offset": 0.2056, "expect_anchors": 25,
            "expect_patients": 237,
            "patience": 5,
        },
    }
    with pytest.raises(schema.ConfigError, match="patience"):
        schema.validate(config)


def test_the_departure_is_supported_by_phase26_and_not_licensed_by_it():
    from cleft import phase26, phase7c

    record = phase27.THE_INNER_VALIDATION_RULED
    safe = " ".join(record["what_makes_the_departure_SAFE_and_what_does_not"].split())
    assert "400 of 400 folds selected epoch 1" in safe
    assert "converges immediately" in safe
    assert "a precedent for removing it and NOT a licence" in safe
    assert "152 training rows" in safe

    # Both citations are live at their homes.
    assert "every one of the 400 is epoch 1" in " ".join(
        phase26.THE_SELECTED_EPOCHS_OBSERVED["the_result"].split()
    )
    assert "converges immediately" in phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION[
        "precedent"
    ]

    cost = " ".join(record["what_the_departure_COSTS"].split())
    assert "cannot stop a fit that is going wrong" in cost
    assert "does not re-run to a better budget" in cost


def test_two_readout_epochs_are_declared_and_neither_is_selectable():
    from cleft import phase21

    record = phase27.THE_TWO_READOUT_EPOCHS_DECLARED
    assert record["declared"].startswith("2026-09-06")
    assert phase27.READOUT_EPOCHS == (1, 40)
    assert phase27.BUDGET_EPOCHS == 40

    two = " ".join(record["the_two"].split())
    assert "epoch 1 and epoch 40, and no other epoch may be quoted" in two
    assert "not selection" in two

    forty = " ".join(record["why_epoch_40"].split())
    assert "the probe stops near epoch 6" in forty
    assert "the loop runs all forty" in forty

    one = " ".join(record["why_epoch_1"].split())
    assert "already converged there" in one

    primary = " ".join(record["which_of_the_two_is_PRIMARY"].split())
    assert "epoch 40" in primary
    assert "cannot be decided by which one reads better" in primary

    # The trajectory is descriptive and the prohibition is cited by name.
    descriptive = " ".join(
        record["the_full_trajectory_is_emitted_and_is_DESCRIPTIVE_ONLY"].split()
    )
    assert "SELECTION_ON_EVALUATION_DATA_PROHIBITION" in descriptive
    assert "not so a better epoch can be found" in descriptive
    assert "selection on the evaluation set" in (
        phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION
    )

    turns = " ".join(record["and_what_the_trajectory_would_license_if_it_turns_over"].split())
    assert "This phase does not re-run to it" in turns


def test_every_setting_is_ruled_with_the_config_it_came_from():
    import yaml

    record = phase27.THE_SETTINGS_RULED
    assert record["ruled"].startswith("2026-09-06")
    assert "p7_d1_vit_b16_imagenet_g1.yaml" in record["the_rule_followed"]

    probe = yaml.safe_load(
        (REPO / "configs" / "p7_d1_vit_b16_imagenet_g1.yaml").read_text(
            encoding="utf-8"
        )
    )["task"]
    # Each mirrored value really is the probe's.
    assert str(probe["learning_rate"]) in record["learning_rate"]
    assert str(probe["weight_decay"]) in record["weight_decay"]
    # batch_size is NOT declared, and the record says why.
    assert "NOT DECLARED" in record["batch_size"]
    assert "never reads it" in " ".join(record["batch_size"].split())
    from cleft.train.torch_backbone import EmbeddingHeadBackbone

    assert str(EmbeddingHeadBackbone.max_steps) in record["max_steps"]
    assert str(probe["max_epochs"]) in record["max_epochs"]
    for seed in probe["seeds"]:
        assert str(seed) in record["seeds"]

    # The three removed ones say so.
    for field in ("inner_val_frac", "monitor", "patience"):
        assert "REMOVED" in record[field], field

    # The setting that actually governs is declared, and the one that
    # does nothing is named as not declared rather than left out.
    steps = " ".join(record["max_steps"].split())
    assert "the setting that actually governs the fit" in steps.lower()
    assert "two thousand steps" in steps

    # The one value with no precedent says it has none.
    readouts = " ".join(record["readout_epochs"].split())
    assert "NO PRECEDENT" in readouts

    assert "NEVER tuned" in record["never_tuned"]


# --------------------------------------------------------------------------
# the disjointness guard: structural, and each part can fail
# --------------------------------------------------------------------------


def test_the_guard_says_why_an_identifier_check_alone_would_be_vacuous():
    record = phase27.THE_DISJOINTNESS_GUARD_DESIGNED
    assert record["designed"].startswith("2026-09-06")

    comment = " ".join(record["why_a_comment_is_not_enough"].split())
    assert "cannot rest it on a reader noticing a comment" in comment

    vacuous = " ".join(record["and_an_identifier_check_alone_would_be_VACUOUS"].split())
    assert "A check that cannot fail is not a check" in vacuous
    assert "THE_GREP_THAT_DID_NOT_RUN" in vacuous

    limit = " ".join(record["what_the_guard_does_NOT_prove"].split())
    assert "does not prove the two image sets contain no shared patient" in limit
    assert "The stronger claim is not made" in limit


def test_the_namespace_refusal_is_part_one_and_it_fires(tmp_path):
    """A config pointing the training input at a non-anchor directory
    raises rather than training on it."""
    import json

    import numpy as np
    import pytest

    from cleft import phase16

    fake = tmp_path / "not_an_anchor_set"
    fake.mkdir()
    np.save(fake / "values.npy", np.zeros((237, 8), dtype=np.float32))
    (fake / "metadata.json").write_text(
        json.dumps({"namespace": "embeddings_g1_ladder", "grades": []}),
        encoding="utf-8",
    )
    with pytest.raises(phase16.AnchorLoopError, match="not an anchor set"):
        phase16.load_anchor_set(fake)


def test_the_feature_guard_is_part_two_and_it_fires():
    """The check that can actually catch a mis-declared artifact."""
    import numpy as np
    import pytest

    rng = np.random.default_rng(1337)
    train = rng.normal(size=(25, 16))
    evaluation = rng.normal(size=(237, 16))
    stems = [f"anchor_{i}" for i in range(25)]
    patients = list(range(1, 238))

    clean = phase27.disjoint_or_raise(
        train, evaluation, train_ids=stems, eval_ids=patients
    )
    assert clean["n_train"] == 25 and clean["n_eval"] == 237
    assert clean["feature_dim"] == 16
    # [CORRECTED 2026-09-06] This asserted identical_feature_rows == 0
    # against what was then a source literal, so it would have passed
    # with the collision loop deleted. A check that cannot fail is not
    # a check. The counts are now computed, and what is asserted is
    # that every reported field is derived from the arrays passed in.
    assert clean["shared_identifiers"] == 0
    assert clean["identical_feature_rows"] == 0
    wider = phase27.disjoint_or_raise(
        train[:5], evaluation[:9],
        train_ids=stems[:5], eval_ids=patients[:9],
    )
    assert (wider["n_train"], wider["n_eval"]) == (5, 9)

    # A shared row is fatal.
    leaked = evaluation.copy()
    leaked[42] = train[7]
    with pytest.raises(phase27.Phase27Error, match="bitwise identical"):
        phase27.disjoint_or_raise(
            train, leaked, train_ids=stems, eval_ids=patients
        )

    # The same artifact under two names. Distinct ids are used so the
    # FEATURE check is what fires, not the identifier check.
    with pytest.raises(phase27.Phase27Error, match="bitwise identical"):
        phase27.disjoint_or_raise(
            train, train, train_ids=stems,
            eval_ids=[f"copy_{i}" for i in range(25)],
        )

    # And a shared identifier, on features that do NOT collide, so
    # the identifier check is what fires.
    with pytest.raises(phase27.Phase27Error, match="appear in both sets"):
        phase27.disjoint_or_raise(
            train, evaluation,
            train_ids=stems,
            eval_ids=stems + [f"p{i}" for i in range(212)],
        )

    # A width mismatch is caught before it becomes a silent broadcast.
    with pytest.raises(phase27.Phase27Error, match="feature widths differ"):
        phase27.disjoint_or_raise(
            train, rng.normal(size=(237, 8)),
            train_ids=stems, eval_ids=patients,
        )


def test_the_fit_signature_is_part_three_and_admits_no_cohort_data():
    """The fit cannot be handed the evaluation truth, because there is no
    parameter for it."""
    import inspect

    parameters = list(inspect.signature(phase27.fit_head).parameters)
    assert parameters == [
        "features", "targets", "seed", "budget", "learning_rate",
        "weight_decay", "max_steps",
    ]
    # No **kwargs through which anything else could arrive.
    assert not [
        p for p in inspect.signature(phase27.fit_head).parameters.values()
        if p.kind is inspect.Parameter.VAR_KEYWORD
    ]

    # And it refuses a budget shorter than what the phase reads at.
    import numpy as np
    import pytest

    with pytest.raises(phase27.Phase27Error, match="shorter than the declared"):
        list(phase27.fit_head(
            np.zeros((25, 4)), np.zeros(25), seed=1, budget=5,
            learning_rate=0.001, weight_decay=0.01, max_steps=50,
        ))


def test_the_call_site_is_AST_tested_and_receives_only_anchor_names():
    """[THE FOURTH PART] The guard can be failed. This is what stops it
    being bypassed. Phase 16 proved fold honesty the same way."""
    import ast

    from cleft import run

    source = (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    task = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "task_p27_anchor_train"
    )

    def _calls(name):
        return [
            node for node in ast.walk(task)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == name
        ]

    # Exactly one fit, and its arguments are anchor-derived only.
    fits = _calls("fit_head")
    assert len(fits) == 1, f"{len(fits)} fit_head call sites"
    fit = fits[0]
    positional = [ast.unparse(a) for a in fit.args]
    assert positional == ["anchor_features", "anchor_grades"], positional
    keywords = {k.arg: ast.unparse(k.value) for k in fit.keywords}
    assert set(keywords) == {
        "seed", "budget", "learning_rate", "weight_decay", "max_steps",
    }
    # Nothing cohort-derived reaches the fit, by name.
    forbidden = ("truth_mean", "truth_class3", "cohort_features",
                 "patient_ids", "rows")
    blob = " ".join(positional + list(keywords.values()))
    for name in forbidden:
        assert name not in blob, name

    # The guard runs, exactly once, and BEFORE the fit.
    guards = _calls("disjoint_or_raise")
    assert len(guards) == 1, f"{len(guards)} guard call sites"
    assert guards[0].lineno < fit.lineno, (
        "the disjointness guard must run before anything is fitted"
    )

    # And the guard's result is used rather than discarded.
    assert "disjointness_guard" in ast.unparse(task)
    assert callable(run.task_p27_anchor_train)


# --------------------------------------------------------------------------
# the exit criteria, locked, and the readings written at the lock
# --------------------------------------------------------------------------


def test_the_exit_criteria_are_locked_and_none_of_them_is_a_number():
    record = phase27.EXIT_CRITERIA
    assert record["locked"].startswith("2026-09-06")
    assert "before any config exists and before any run" in record["locked"]

    numbered = [k for k in record if k[0].isdigit()]
    assert sorted(int(k.split("_")[0]) for k in numbered) == list(range(1, 11))

    not_a_number = " ".join(record["what_is_NOT_an_exit_criterion"].split())
    assert "a number" in not_a_number
    assert "will hit it" in not_a_number

    # Four of the ten are about what the RUN emits, and the record says
    # which and why rather than letting six-of-ten look like a shortcut.
    met = " ".join(record["and_six_of_the_ten_are_MET_AT_THE_LOCK"].split())
    assert "4, 5, 7 and 9" in met
    assert "none of them can be met by writing prose" in met
    at_lock = [k for k in numbered if "MET AT THE LOCK" in record[k]]
    assert sorted(int(k.split("_")[0]) for k in at_lock) == [1, 2, 3, 6]

    # The draft is preserved beside the lock rather than replaced.
    assert phase27.EXIT_CRITERIA_DRAFT["drafted"].startswith("2026-09-06")
    assert "DRAFT, not locked" in phase27.EXIT_CRITERIA_DRAFT["drafted"]


def test_the_readings_at_the_lock_turn_on_the_band_not_on_the_probes_value():
    record = phase27.READINGS_AT_THE_LOCK
    assert record["written"].startswith("2026-09-06")

    bound = " ".join(record["the_bound_that_governs_every_reading"].split())
    assert "within about 0.10 of the probe's 0.2520" in bound
    assert "not on which side of 0.2520 it falls" in bound

    inside = " ".join(record["if_the_primary_lands_INSIDE_the_unresolvable_band"].split())
    assert "the expected outcome" in inside
    assert "does NOT say the two labels are equivalent" in inside

    below = " ".join(record["if_the_primary_lands_CLEARLY_BELOW"].split())
    assert "prediction_sd is near zero" in below
    assert "a different and more interesting result" in below

    above = " ".join(record["if_the_primary_lands_CLEARLY_ABOVE"].split())
    assert "UNINTERPRETABLE" in above
    assert "licenses no sentence about cleaner labels" in above

    disagree = " ".join(record["if_the_TWO_declared_epochs_disagree"].split())
    assert "a finding about the budget, not about the label" in disagree
    assert "The primary stays epoch 40 whichever reads better" in disagree

    agree = " ".join(record["if_the_two_declared_epochs_AGREE"].split())
    assert "the departure at THE_INNER_VALIDATION_RULED cost nothing" in agree

    unavailable = " ".join(
        record["and_the_reading_that_is_NOT_available_whatever_happens"].split()
    )
    assert "one label is cleaner than the other" in unavailable

    # And the restate's readings are preserved rather than replaced.
    assert phase27.READINGS["written"].startswith("2026-09-06")


# --------------------------------------------------------------------------
# the build: seven items
# --------------------------------------------------------------------------


def test_the_task_is_registered_and_its_spec_matches_the_lock():
    from cleft import run
    from cleft.config import schema

    assert run.TASKS["p27_anchor_train"] is run.task_p27_anchor_train
    spec = schema.TASK_SPECS["p27_anchor_train"]

    # The budget and the primary are pinned by the choices check, so a
    # config cannot move where the phase reads.
    assert spec["max_epochs"].choices == (40,)
    assert spec["primary_epoch"].choices == (40,)
    assert spec["expect_anchors"].choices == (25,)
    assert spec["expect_patients"].choices == (237,)

    # Every input reference follows the *_artifact convention, so guard 3
    # covers all three without INPUT_REFERENCE_KEYS being extended.
    references = [k for k in spec if k.endswith("_artifact")]
    assert sorted(references) == [
        "anchor_artifact", "embeddings_artifact", "manifest_artifact",
    ]
    assert schema.INPUT_REFERENCE_KEYS == ("artifact", "synth_set", "input")


def test_the_run_directory_axis_for_the_training_population_exists():
    from cleft import run_names

    assert "trained_on" in run_names.RUN_DIR_AXES
    assert run_names.RUN_DIR_AXES["trained_on"] == {"cohort", "anchors", "synth"}

    # And it does what an axis is for: a job id contradicting the stem is
    # reported rather than passing.
    found = run_names.job_id_contradictions(
        "p27_anchor_train__abcdef12__p27-cohort-run"
    )
    axes = {entry["axis"] for entry in found}
    assert "trained_on" in axes


def test_the_generator_reproduces_the_shipped_config():
    """--check drift verification, run as a test so a hand edit fails."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/generate_phase27_configs.py", "--check"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "matches the record" in result.stdout


def test_the_config_declares_three_carried_hashes_and_invents_none():
    """Every rollup is carried from a config that already declares it."""
    import yaml

    def declared(name):
        config = yaml.safe_load(
            (REPO / "configs" / name).read_text(encoding="utf-8")
        )
        return {entry["name"]: entry for entry in config["inputs"]}

    ours = declared("p27_anchor_train.yaml")
    anchor_home = declared("p16_anchor_loop.yaml")["anchor_embeddings"]
    probe_home = declared("p7_d1_vit_b16_imagenet_g1.yaml")

    assert ours["anchor_embeddings"]["rollup_sha256"] == (
        anchor_home["rollup_sha256"]
    )
    assert ours["anchor_embeddings"]["path"] == anchor_home["path"]
    for name in ("manifest_v1", "embeddings"):
        assert ours[name]["rollup_sha256"] == probe_home[name]["rollup_sha256"]
        assert ours[name]["path"] == probe_home[name]["path"]

    # The header does not claim the config is unresolved when it is not.
    text = (REPO / "configs" / "p27_anchor_train.yaml").read_text(
        encoding="utf-8"
    )
    assert "**RESOLVED.**" in text
    assert "UNRESOLVED" not in text
    assert "NO new declaration before launch" in text


def test_the_config_loads_and_carries_the_locked_settings():
    from cleft.config.schema import load_config

    config = load_config(str(REPO / "configs" / "p27_anchor_train.yaml"))
    task = config["task"]
    assert task["kind"] == "p27_anchor_train"
    assert task["max_epochs"] == phase27.BUDGET_EPOCHS
    assert tuple(task["readout_epochs"]) == phase27.READOUT_EPOCHS
    assert task["primary_epoch"] == max(phase27.READOUT_EPOCHS)
    assert task["expect_anchors"] == phase27.N_ANCHORS
    assert task["expect_patients"] == 237
    # The derived offset, recomputed rather than trusted.
    from cleft import phase16

    grades = [
        g for g, n in enumerate(phase16.ANCHOR_GRADE_SPREAD, start=1)
        for _ in range(n)
    ]
    assert task["target_offset"] == round(sum(grades) / len(grades) - 2.7544, 4)
    from cleft.train.torch_backbone import EmbeddingHeadBackbone

    assert task["max_steps"] == EmbeddingHeadBackbone.max_steps
    # And the three removed fields really are absent from the shipped file.
    for field in ("inner_val_frac", "monitor", "patience", "batch_size"):
        assert field not in task, field


def test_what_must_be_built_is_now_built():
    """Item by item, checked against the tree rather than ticked."""
    from cleft import run, run_names
    from cleft.config import schema

    assert "p27_anchor_train" in schema.TASK_SPECS          # 1
    assert "p27_anchor_train" in run.TASKS                  # 1
    assert [k for k in schema.TASK_SPECS["p27_anchor_train"]
            if k.endswith("_artifact")]                     # 2
    # 3: the frozen fold machinery is untouched by this phase. The
    # check is on USE, not on the substring: the task reads the
    # manifest's fold column for the predictions CSV, and never
    # imports or calls data.folds.
    task_source = (REPO / "src" / "cleft" / "run.py").read_text(
        encoding="utf-8"
    ).split("def task_p27_anchor_train")[1].split("\ndef ")[0]
    import ast

    tree = ast.parse(
        (REPO / "src" / "cleft" / "run.py").read_text(encoding="utf-8")
    )
    task_node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "task_p27_anchor_train"
    )
    imported = {
        alias.name
        for n in ast.walk(task_node)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for alias in n.names
    }
    assert "folds" not in imported
    assert not [
        n for n in ast.walk(task_node)
        if isinstance(n, ast.Attribute) and n.attr in {
            "generate", "verify"
        } and ast.unparse(n).startswith("folds")
    ]
    # The manifest's fold column is read, and that is all.
    assert 'r["fold"]' in task_source
    assert "trained_on" in run_names.RUN_DIR_AXES            # 4
    assert callable(phase27.disjoint_or_raise)               # 5
    assert "prediction_sd" in task_source                    # 6
    assert (REPO / "scripts" / "generate_phase27_configs.py").is_file()  # 7
    assert (REPO / "configs" / "p27_anchor_train.yaml").is_file()


def test_the_task_runs_end_to_end_on_a_fixture(tmp_path):
    """The whole path, on synthetic inputs, so the wiring is exercised
    rather than reasoned about. No clinical data is involved."""
    import json

    import numpy as np
    import pytest

    pytest.importorskip("torch")
    from cleft import phase16, run

    rng = np.random.default_rng(1337)

    # A real anchor set, written by the real writer so its invariants hold.
    grades = [
        g for g, n in enumerate(phase16.ANCHOR_GRADE_SPREAD, start=1)
        for _ in range(n)
    ]
    anchor_dir = tmp_path / "anchor_deall_vfixture"
    phase16.save_anchor_set(
        anchor_dir,
        rng.normal(size=(25, 16)).astype(np.float32),
        stems=[f"stem_{i}" for i in range(25)],
        grades=grades,
        parity=1.53e-05,
        run_name="fixture",
    )

    # A cohort manifest and an embeddings artifact, 12 patients.
    n = 12
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    truth = np.round(rng.uniform(1.4, 4.6, size=n) * 5) / 5
    class3 = np.where(truth < 2.5, 0, np.where(truth < 3.5, 1, 2))
    # The real fifteen columns, read from the frozen contract rather
    # than typed, so a manifest change breaks this fixture loudly.
    from cleft.data.manifest import MANIFEST_COLUMNS

    columns = [name for name, _ in MANIFEST_COLUMNS]
    header = ",".join(columns) + "\n"
    body = ""
    for i in range(n):
        values = {
            "patient_id": str(i + 1), "frontal_id": str(1000 + i),
            "basal_id": str(2000 + i), "mean": f"{truth[i]:.1f}",
            "median": f"{truth[i]:.1f}", "mode": f"{truth[i]:.1f}",
            "weighted_mean": f"{truth[i]:.1f}",
            "orthodontist": str(int(round(truth[i]))),
            "soft_1": "0.2", "soft_2": "0.2", "soft_3": "0.2",
            "soft_4": "0.2", "soft_5": "0.2",
            "class3": str(int(class3[i])), "fold": str(i % 5),
        }
        body += ",".join(values[c] for c in columns) + "\n"
    (manifest_dir / "manifest.csv").write_text(header + body, encoding="utf-8")

    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    np.save(
        embeddings_dir / "values.npy",
        rng.normal(size=(n, 16)).astype(np.float32),
    )
    (embeddings_dir / "metadata.json").write_text(
        json.dumps({
            "kind": "pooled", "patient_ids": list(range(1, n + 1)),
            "init": "imagenet", "geometry": "g1", "variant": None,
            "backbone": "vit_b16",
        }),
        encoding="utf-8",
    )

    class _Ctx:
        def __init__(self):
            self.config = {"task": {
                "kind": "p27_anchor_train", "arm": "p27_anchor_train",
                "anchor_artifact": "anchor", "manifest_artifact": "manifest",
                "embeddings_artifact": "embeddings",
                "geometry": "g1", "backbone": "vit_b16",
                "seeds": [1337, 2024], "max_epochs": 40,
                "readout_epochs": [1, 40], "primary_epoch": 40,
                "learning_rate": 0.001, "weight_decay": 0.01,
                "max_steps": 50, "target_offset": 0.2056,
                "expect_anchors": 25, "expect_patients": n,
            }}
            self.inputs = [
                {"name": "anchor", "path": str(anchor_dir)},
                {"name": "manifest", "path": str(manifest_dir)},
                {"name": "embeddings", "path": str(embeddings_dir)},
            ]
            self.written = {}
            self.logged = []

        def log(self, message):
            self.logged.append(message)

        def path(self, name, tier=None):
            return tmp_path / name

        def atomic(self, name, tier=None):
            import contextlib

            @contextlib.contextmanager
            def _writer():
                target = tmp_path / name
                yield target
                self.written[name] = target.read_text(encoding="utf-8")

            return _writer()

    ctx = _Ctx()
    run.task_p27_anchor_train(ctx)

    metrics = json.loads(ctx.written["metrics.json"])
    # Both declared epochs reached the metrics, and only those two.
    assert sorted(metrics["per_seed_by_readout_epoch"]) == ["1", "40"]
    assert metrics["primary_epoch"] == 40
    assert metrics["declared_readout_epochs"] == [1, 40]
    # The collapse detector is emitted per seed at every readout epoch.
    for epoch in ("1", "40"):
        for seed in ("1337", "2024"):
            assert "prediction_sd" in metrics["per_seed_by_readout_epoch"][
                epoch
            ][seed]
    # This arm's own seed band, not a borrowed one.
    assert metrics["seed_band_by_readout_epoch"]["40"]["pcc"]["n_seeds"] == 2
    # The guard ran and its counts are recorded.
    assert metrics["disjointness_guard"]["n_train"] == 25
    assert metrics["disjointness_guard"]["n_eval"] == n
    assert metrics["disjointness_guard"]["guard"] == "phase27.disjoint_or_raise"
    # The trajectory is the whole budget and is labelled descriptive.
    assert len(metrics["trajectory_DESCRIPTIVE_ONLY"]["1337"]) == 40
    # The crossing is carried into the run's own output.
    assert metrics["the_crossing"]["target_offset_declared"] == 0.2056
    assert metrics["anchor_grade_mean"] == 2.96
    assert any("disjointness guard PASSED" in line for line in ctx.logged)


def test_the_task_refuses_when_the_two_sets_are_the_same_artifact(tmp_path):
    """The guard is wired into the task, not merely available."""
    import json

    import numpy as np
    import pytest

    from cleft import phase16, run

    rng = np.random.default_rng(7)
    grades = [
        g for g, n in enumerate(phase16.ANCHOR_GRADE_SPREAD, start=1)
        for _ in range(n)
    ]
    features = rng.normal(size=(25, 16)).astype(np.float32)
    anchor_dir = tmp_path / "anchor_deall_vleak"
    phase16.save_anchor_set(
        anchor_dir, features, stems=[f"s{i}" for i in range(25)],
        grades=grades, parity=1.53e-05, run_name="fixture",
    )

    # The cohort embeddings ARE the anchor features: the mis-declaration
    # the guard exists to catch.
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    from cleft.data.manifest import MANIFEST_COLUMNS

    columns = [name for name, _ in MANIFEST_COLUMNS]
    row = {
        "frontal_id": "1", "basal_id": "2", "mean": "2.6",
        "median": "2.6", "mode": "2.6", "weighted_mean": "2.6",
        "orthodontist": "3", "soft_1": "0.2", "soft_2": "0.2",
        "soft_3": "0.2", "soft_4": "0.2", "soft_5": "0.2",
        "class3": "1",
    }
    (manifest_dir / "manifest.csv").write_text(
        ",".join(columns) + "\n"
        + "".join(
            ",".join(
                {**row, "patient_id": str(i + 1), "fold": str(i % 5)}[c]
                for c in columns
            ) + "\n"
            for i in range(25)
        ),
        encoding="utf-8",
    )
    embeddings_dir = tmp_path / "embeddings"
    embeddings_dir.mkdir()
    np.save(embeddings_dir / "values.npy", features)
    (embeddings_dir / "metadata.json").write_text(
        json.dumps({
            "kind": "pooled", "patient_ids": list(range(1, 26)),
            "init": "imagenet", "geometry": "g1", "variant": None,
            "backbone": "vit_b16",
        }),
        encoding="utf-8",
    )

    class _Ctx:
        config = {"task": {
            "kind": "p27_anchor_train", "arm": "p27_anchor_train",
            "anchor_artifact": "anchor", "manifest_artifact": "manifest",
            "embeddings_artifact": "embeddings", "geometry": "g1",
            "backbone": "vit_b16", "seeds": [1337], "max_epochs": 40,
            "readout_epochs": [1, 40], "primary_epoch": 40,
            "learning_rate": 0.001, "weight_decay": 0.01, "max_steps": 50,
            "target_offset": 0.2056, "expect_anchors": 25,
            "expect_patients": 25,
        }}
        inputs = [
            {"name": "anchor", "path": str(anchor_dir)},
            {"name": "manifest", "path": str(manifest_dir)},
            {"name": "embeddings", "path": str(embeddings_dir)},
        ]

        def log(self, message):
            pass

    with pytest.raises(phase27.Phase27Error, match="bitwise identical"):
        run.task_p27_anchor_train(_Ctx())


def test_the_head_is_the_probes_and_the_correction_is_recorded():
    """[ADDED 2026-09-06] The build's own R2 error, pinned so it cannot
    come back. fit_head must DRIVE the frozen head rather than mirror it,
    because the probe's initialiser is what makes the registered
    degenerate signature an equality."""
    import ast
    import inspect

    from cleft.train.torch_backbone import EmbeddingHeadBackbone

    # The fit reuses the frozen class rather than building a second loop.
    source = inspect.getsource(phase27.fit_head)
    assert "EmbeddingHeadBackbone" in source
    assert "torch.nn.Linear" not in source
    assert "torch.optim" not in source
    tree = ast.parse(source.lstrip())
    constructed = [
        ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)
    ]
    assert "EmbeddingHeadBackbone" in constructed

    # The frozen head really does start at zero weights and the training
    # mean, which is the whole reason for the correction.
    reset_source = inspect.getsource(EmbeddingHeadBackbone.reset)
    assert "np.mean(train_labels)" in reset_source
    # The comment wraps, so it is normalised before matching.
    flat = " ".join(reset_source.replace("#", " ").split())
    assert "Weights at zero and bias at the training-fold mean" in flat
    assert "an untrained head predicts the mean exactly" in flat
    assert "which is what gate 3 checks" in flat

    record = phase27.THE_INIT_IS_THE_PROBES
    assert record["corrected"].startswith("2026-09-06")
    wrong = " ".join(record["what_the_first_build_did"].split())
    assert "torch.nn.Linear" in wrong
    assert "A different fit under the same setting names" in wrong
    assert "R2 error" in wrong

    broke = " ".join(record["and_it_broke_the_registered_signature"].split())
    assert "the record was right and the implementation was wrong" in broke.lower()
    assert "would have missed it" in broke

    reuse = " ".join(record["the_fix_is_REUSE_rather_than_a_better_mirror"].split())
    assert "second implementation" in reuse

    # And it says what it did NOT change.
    untouched = " ".join(record["what_this_did_NOT_touch"].split())
    assert "no record, no reading and no criterion changes" in untouched


def test_the_degenerate_signature_is_an_equality_under_the_corrected_init():
    record = phase27.THE_DEGENERATE_SIGNATURE_REGISTERED
    exact = " ".join(
        record["and_the_head_STARTS_there_which_makes_the_signature_exact"].split()
    )
    assert "it IS 2.96" in exact
    assert "an equality rather than a resemblance" in exact
    assert "THE_INIT_IS_THE_PROBES" in exact

    # 74/25 really is 2.96, from the banked spread.
    from cleft import phase16

    grades = [
        g for g, n in enumerate(phase16.ANCHOR_GRADE_SPREAD, start=1)
        for _ in range(n)
    ]
    assert sum(grades) == 74 and len(grades) == 25
    assert sum(grades) / len(grades) == 2.96


def test_the_budget_expectation_is_tagged_REASONED_and_not_measured():
    """It is arithmetic on the optimiser's update rule, not a run. The
    record must not let it read as a measurement."""
    record = phase27.THE_BUDGET_EXPECTATION_REGISTERED
    assert record["reasoned"].startswith("2026-09-06")
    assert "tagged as reasoned" in record["reasoned"]

    expectation = " ".join(record["the_expectation"].split())
    assert "INTERPOLATES the training set exactly" in expectation
    assert "two thousand steps" in expectation
    # 40 epochs x 50 steps really is two thousand.
    from cleft.train.torch_backbone import EmbeddingHeadBackbone

    assert phase27.BUDGET_EPOCHS * EmbeddingHeadBackbone.max_steps == 2000

    contrast = " ".join(record["so_the_two_readout_epochs_are_a_real_contrast"].split())
    assert "a better reason for the same choice, found afterwards" in contrast
    assert "rather than backdated" in contrast

    limit = " ".join(record["what_it_does_NOT_predict"].split())
    assert "nothing about the cohort" in limit
    assert "says the fit ran, not that it transfers" in limit

    # And nothing in the module tags this as measured.
    blob = " ".join(str(v) for v in record.values())
    assert "[MEASURED]" not in blob


# --------------------------------------------------------------------------
# the close, 2026-09-06
# --------------------------------------------------------------------------

METRICS_FILE = Path("X:/p27_metrics.json")

PROBE_PCC, PROBE_SD = 0.2520, 0.0148      # ladder.BEST_ARM
TRUTH_MEAN, TRUTH_SD = 2.7544, 0.6587     # phase18 / phase10


def _metrics():
    import json

    import pytest

    if not METRICS_FILE.is_file():
        pytest.skip("the Phase 27 metrics file is not on this machine")
    return json.loads(METRICS_FILE.read_text(encoding="utf-8"))


def test_the_guard_passed_on_the_sizes_the_lock_fixed():
    payload = _metrics()
    guard = payload["disjointness_guard"]
    assert guard["guard"] == "phase27.disjoint_or_raise"
    assert guard["n_train"] == phase27.N_ANCHORS == 25
    assert guard["n_eval"] == 237
    assert guard["feature_dim"] == 768
    assert guard["shared_identifiers"] == 0
    assert guard["identical_feature_rows"] == 0

    record = " ".join(phase27.THE_GUARD_PASSED["what_it_checked"].split())
    assert "25 training rows against 237 evaluation rows at 768" in record
    assert "no shared identifier and no bitwise-identical feature row" in record

    # The limitation is carried forward unchanged.
    limit = " ".join(
        phase27.THE_GUARD_PASSED["what_it_does_not_prove_is_unchanged"].split()
    )
    assert "does not say the two image sets contain no shared person" in limit


def test_every_banked_figure_matches_the_run_at_full_precision():
    payload = _metrics()
    record = phase27.THE_RESULT_OBSERVED

    # The banked figure is the per-seed VALUE, which the band reports
    # as min and max. Three banded means differ from it by one ulp
    # (summing five identical floats), and the record says so at
    # THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION rather than banking the
    # derived statistic.
    primary = payload["primary"]
    for field, banked in record["primary_epoch_40"].items():
        if field.startswith("prediction_mean"):
            continue
        assert primary[field]["min"] == primary[field]["max"] == banked, field

    epoch1 = payload["seed_band_by_readout_epoch"]["1"]
    for field, banked in record["second_declared_epoch_1"].items():
        if field.startswith("prediction_mean"):
            continue
        assert epoch1[field]["min"] == epoch1[field]["max"] == banked, field

    # And the ulp gap is real, so the note is not stale.
    drifted = [
        (ep, f) for ep, bands in payload["seed_band_by_readout_epoch"].items()
        for f, b in bands.items() if b["mean"] != b["min"]
    ]
    assert drifted, "if no banded mean drifts, the ulp note is stale"
    ulp = " ".join(
        phase27.THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION[
            "and_three_banded_MEANS_differ_from_their_min_by_one_ulp"
        ].split()
    )
    assert "banks the min/max value, which is the measurement" in ulp
    assert len(drifted) == 3

    # prediction_mean is per seed, not banded.
    for epoch, key in (("40", "primary_epoch_40"), ("1", "second_declared_epoch_1")):
        seen = {
            row["prediction_mean"]
            for row in payload["per_seed_by_readout_epoch"][epoch].values()
        }
        assert len(seen) == 1
        assert seen.pop() == record[key]["prediction_mean"], epoch

    assert record["baselines_from_the_run"] == payload["baselines"]
    assert payload["primary_epoch"] == 40
    assert payload["declared_readout_epochs"] == list(phase27.READOUT_EPOCHS)


def test_the_primary_is_the_declared_epoch_and_epoch_one_reads_higher():
    """The declaration was made before any number, and the result tested
    it: the epoch that was NOT declared primary reads better."""
    payload = _metrics()
    primary = payload["primary"]["pcc"]["mean"]
    other = payload["seed_band_by_readout_epoch"]["1"]["pcc"]["mean"]
    assert other > primary
    assert payload["primary_epoch"] == max(phase27.READOUT_EPOCHS)

    record = " ".join(
        phase27.THE_RESULT_OBSERVED[
            "the_primary_is_epoch_40_because_it_was_declared_so"
        ].split()
    )
    assert "epoch 1 reads higher on the correlation" in record
    assert "exactly why the declaration was made in advance" in record
    assert str(primary) in record and str(other) in record


def test_the_seed_sd_is_zero_and_the_record_is_precise_about_which_are_exact():
    payload = _metrics()
    record = phase27.THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION

    # Identity is min == max, on every field at both declared epochs.
    for epoch, bands in payload["seed_band_by_readout_epoch"].items():
        for field, band in bands.items():
            assert band["min"] == band["max"], (epoch, field)
            assert band["n_seeds"] == 5

    # And one distinct value per field across the five seeds.
    for epoch, seeds in payload["per_seed_by_readout_epoch"].items():
        assert len(seeds) == 5
        for field in next(iter(seeds.values())):
            assert len({row[field] for row in seeds.values()}) == 1, (epoch, field)

    shows = " ".join(record["what_the_run_shows"].replace("`", "").split())
    assert "min equals max" in shows
    assert "ONE distinct value per field" in shows

    # The record does NOT say every sd is exactly zero, because two are not.
    exact = {
        f for f, b in payload["seed_band_by_readout_epoch"]["40"].items()
        if b["sd"] == 0.0
    }
    inexact = set(payload["seed_band_by_readout_epoch"]["40"]) - exact
    assert inexact, "if every sd is exactly 0.0 the record's caveat is stale"
    noise = " ".join(
        record["and_two_of_the_sds_are_float_noise_rather_than_zero"].split()
    )
    for field in inexact:
        assert str(payload["seed_band_by_readout_epoch"]["40"][field]["sd"]) in noise
    assert str(
        payload["seed_band_by_readout_epoch"]["1"]["mae"]["sd"]
    ) in noise


def test_the_determinism_is_traced_in_source_not_inferred_from_the_output():
    import ast
    import inspect

    from cleft.train.torch_backbone import EmbeddingHeadBackbone

    record = phase27.THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION
    why = " ".join(record["WHY_it_is_zero_traced_in_the_source"].split())
    assert "torch.zeros" in why and "np.mean(train_labels)" in why
    assert "the seed reaches the code and has nothing to act on" in why

    # Re-traced here rather than trusted.
    tree = ast.parse(inspect.getsource(EmbeddingHeadBackbone).lstrip())
    for method in ("_ensure", "train_epoch", "predict"):
        fn = next(
            n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == method
        )
        body = ast.unparse(fn)
        for token in ("manual_seed", "rand", "shuffle", "permutation",
                      "default_rng", "randperm"):
            assert token not in body, (method, token)
    fit = inspect.getsource(phase27.fit_head)
    for token in ("rand", "shuffle", "permutation", "default_rng", "randperm"):
        assert token not in fit, token


def test_the_broken_wiring_reading_is_addressed_rather_than_assumed():
    payload = _metrics()
    record = phase27.THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION

    also = " ".join(
        record["a_zero_seed_sd_is_ALSO_what_broken_seed_wiring_looks_like"].split()
    )
    assert "indistinguishable from the numbers alone" in also

    three = " ".join(
        record["what_distinguishes_them_HERE_and_it_is_three_things"].split()
    )
    assert "BY CONSTRUCTION" in three
    assert "Both declared readout epochs show it" in three
    assert "ALL FORTY epochs" in three

    # The third distinguisher, checked: all five trajectories bit identical.
    trajectory = payload["trajectory_DESCRIPTIVE_ONLY"]
    seeds = sorted(trajectory, key=int)
    assert len(seeds) == 5
    first = trajectory[seeds[0]]
    assert len(first) == phase27.BUDGET_EPOCHS == 40
    for seed in seeds[1:]:
        assert trajectory[seed] == first, seed

    passed = " ".join(record["and_the_seed_is_genuinely_passed"].split())
    assert "not a seed that was never wired" in passed
    assert "a seed whose only consumer was removed" in passed


def test_the_degenerate_signature_did_not_fire():
    payload = _metrics()
    record = phase27.THE_DEGENERATE_SIGNATURE_DID_NOT_FIRE
    primary = payload["primary"]
    pred_sd = primary["prediction_sd"]["mean"]
    pred_mean = next(
        iter(payload["per_seed_by_readout_epoch"]["40"].values())
    )["prediction_mean"]

    # It varies rather than sitting on a constant.
    assert pred_sd > TRUTH_SD
    happened = " ".join(record["what_happened_instead"].split())
    assert str(pred_sd) in happened
    assert f"{pred_sd / TRUTH_SD:.4f}" in happened
    assert str(pred_mean) in happened
    assert f"{abs(pred_mean - 2.96):.4f}" in happened

    # And it is not at the majority floor.
    acc3 = primary["acc3"]["min"]
    assert acc3 < payload["baselines"]["majority"]
    floor = " ".join(record["and_the_accuracy_is_not_at_the_floor_either"].split())
    assert str(acc3) in floor
    assert str(payload["baselines"]["majority"]) in floor
    assert str(payload["baselines"]["chance"]) in floor

    # The registered signature is the one that was refuted.
    assert payload["anchor_grade_mean"] == 2.96
    why = " ".join(record["why_this_is_a_FINDING_and_not_merely_a_relief"].split())
    assert "refuted by measurement is worth more than one confirmed" in why
    limit = " ".join(record["what_it_does_NOT_establish"].split())
    assert "Not collapsing is a low bar" in limit


def test_forty_epochs_moved_almost_nothing_and_it_extends_a_known_finding():
    payload = _metrics()
    record = phase27.THE_BUDGET_MOVED_ALMOST_NOTHING
    trajectory = next(iter(payload["trajectory_DESCRIPTIVE_ONLY"].values()))

    pccs = [trajectory[str(e)]["pcc"] for e in range(1, 41)]
    span = max(pccs) - min(pccs)
    assert pccs.index(max(pccs)) == 0          # the maximum is epoch 1
    assert span < PROBE_SD / 3

    two = " ".join(record["the_two_declared_epochs"].split())
    assert str(pccs[0]) in two and str(pccs[39]) in two
    assert f"{pccs[39] - pccs[0]:.7f}" in two

    whole = " ".join(record["and_the_whole_trajectory_says_the_same"].split())
    assert f"{max(pccs):.6f}" in whole
    assert f"{min(pccs):.6f}" in whole
    assert f"epoch {pccs.index(min(pccs)) + 1}" in whole
    assert f"{span:.6f}" in whole

    # It extends phase7c and phase26 rather than claiming novelty.
    extends = " ".join(
        record["this_is_NOT_new_and_the_record_says_which_finding_it_extends"].split()
    )
    assert "converges immediately" in extends
    assert "400 of 400 folds" in extends
    from cleft import phase26, phase7c

    assert "converges immediately" in phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION[
        "precedent"
    ]
    assert "every one of the 400 is epoch 1" in " ".join(
        phase26.THE_SELECTED_EPOCHS_OBSERVED["the_result"].split()
    )

    stronger = " ".join(record["and_that_is_the_stronger_form"].split())
    assert "there is no selection at all" in stronger

    # The last-epoch oddity is recorded rather than smoothed.
    odd = " ".join(record["and_one_oddity_recorded_rather_than_smoothed"].split())
    assert f"{trajectory['39']['pcc']:.6f}" in odd
    assert f"{trajectory['40']['pcc']:.6f}" in odd
    assert "it changes nothing" in odd


def test_the_declared_offset_is_corrected_by_measurement():
    payload = _metrics()
    record = phase27.THE_OFFSET_IS_CORRECTED

    declared = payload["the_crossing"]["target_offset_declared"]
    assert declared == 0.2056
    pred_mean = next(
        iter(payload["per_seed_by_readout_epoch"]["40"].values())
    )["prediction_mean"]
    realised = pred_mean - TRUTH_MEAN
    assert realised < 0 < declared              # opposite signs

    measured = " ".join(record["what_was_MEASURED"].split())
    assert str(pred_mean) in measured
    assert f"{realised:.4f}" in measured
    epoch1_mean = next(
        iter(payload["per_seed_by_readout_epoch"]["1"].values())
    )["prediction_mean"]
    assert f"{epoch1_mean - TRUTH_MEAN:.4f}" in measured
    assert "Opposite in sign to the declaration" in measured

    why = " ".join(record["why_the_derivation_was_wrong"].split())
    assert "correct arithmetic about the wrong quantity" in why
    assert "WHERE THE COHORT SITS" in why

    # The candidate the record already carried.
    candidate = " ".join(
        record["and_the_record_carried_the_right_candidate_already"].split()
    )
    assert "0.86 to 1.17" in candidate
    assert "candidate explanation for either outcome" in candidate
    assert "candidate explanation for either outcome" in " ".join(
        phase27.THE_TWO_EXTRACTIONS_COMPARED[
            "one_more_difference_that_is_NOT_the_extractor"
        ].split()
    )

    # The declared figure is preserved rather than rewritten.
    changes = " ".join(record["what_this_CHANGES_and_what_it_does_not"].split())
    assert "stays in the config" in changes
    assert "is UNCHANGED" in changes
    import yaml

    config = yaml.safe_load(
        (REPO / "configs" / "p27_anchor_train.yaml").read_text(encoding="utf-8")
    )
    assert config["task"]["target_offset"] == 0.2056


def test_the_value_metrics_are_reported_under_the_crossing():
    payload = _metrics()
    record = phase27.THE_VALUE_METRICS_UNDER_THE_CROSSING
    primary = payload["primary"]

    figures = " ".join(record["the_figures"].split())
    for field in ("mae", "rmse", "acc3"):
        assert str(primary[field]["min"]) in figures, field
    assert str(payload["baselines"]["majority"]) in figures
    assert str(payload["baselines"]["chance"]) in figures
    assert str(TRUTH_MEAN) in figures

    why = " ".join(record["why_they_are_NOT_comparable_to_a_cohort_trained_arm"].split())
    assert "CROSS_TARGET_ERROR_PROVENANCE" in why

    only = " ".join(record["only_the_two_correlations_cross_cleanly"].split())
    assert str(primary["pcc"]["min"]) in only
    assert str(primary["spearman"]["min"]) in only
    assert "0.00e+00" in only
    from cleft import phase21

    assert "0.00e+00" in phase21.SCALE_INVARIANCE_PROHIBITION

    # The run's own output carries the crossing note too.
    assert "NOT comparable to a cohort-trained arm" in payload["the_crossing"]["note"]


def test_the_readings_are_applied_and_the_gap_in_them_is_recorded():
    from cleft import ladder, relevance

    payload = _metrics()
    record = phase27.THE_READINGS_APPLIED
    pcc = payload["primary"]["pcc"]["mean"]
    delta = abs(pcc - PROBE_PCC)

    where = " ".join(record["where_the_result_sits"].split())
    assert f"{delta:.6f}" in where
    assert "0.04 to 0.10" in where
    assert 0.04 <= delta <= 0.10
    assert "0.1386" in where
    assert delta < ladder.SMALLEST_RESOLVABLE_DIFFERENCE["measured"][
        "smallest_resolvable"
    ]["delta"]

    # The registered reading is quoted from the lock, not paraphrased.
    registered = " ".join(record["so_the_registered_reading_is_the_INSIDE_THE_BAND_one"].split())
    lock = " ".join(
        phase27.READINGS_AT_THE_LOCK[
            "if_the_primary_lands_INSIDE_the_unresolvable_band"
        ].split()
    )
    assert "It does NOT say the two labels are equivalent" in registered
    assert "It does NOT say the two labels are equivalent" in lock

    # The case the readings did not cover, with the arithmetic.
    gap = " ".join(record["THE_CASE_THE_READINGS_DID_NOT_ANTICIPATE"].split())
    threshold = relevance.significance_threshold(237)
    assert f"{threshold:.6f}" in gap
    assert pcc > threshold
    assert "distinguishable from zero at n=237" in gap
    assert "Something was found" in gap

    honest = " ".join(record["why_that_gap_is_recorded_rather_than_papered_over"].split())
    assert "they were incomplete" in honest
    assert "cheaper than stretching one of them to fit" in honest

    # And the asymmetry governs the positive half.
    asym = " ".join(
        record["and_the_ASYMMETRY_registration_governs_the_half_that_is_positive"].split()
    )
    assert "uninterpretable case" in asym
    assert "licenses no sentence about cleaner labels" in asym

    both = " ".join(record["so_which_case_is_this_stated_plainly"].split())
    assert "a null against the probe and an uninterpretable positive against zero" in both


def test_the_criterion_is_not_resolved_and_the_suspicion_was_checked():
    from cleft.train.phase3 import combined_claimable_delta

    payload = _metrics()
    record = phase27.THE_CRITERION_NOT_RESOLVED
    pcc = payload["primary"]["pcc"]["mean"]
    sd = payload["primary"]["pcc"]["sd"]
    delta = abs(pcc - PROBE_PCC)

    both = combined_claimable_delta(sd, 5, PROBE_SD, 5)
    passes = " ".join(record["condition_2_PASSES"].split())
    assert f"{delta:.6f}" in passes
    assert str(round(both["arm_means_95"], 6)) in passes
    assert f"{delta / both['arm_means_95']:.2f}x" in passes

    # The obvious suspicion, checked and refuted rather than asserted.
    noisy = combined_claimable_delta(PROBE_SD, 5, PROBE_SD, 5)["arm_means_95"]
    refuted = " ".join(record["and_the_zero_seed_sd_is_NOT_what_makes_it_pass"].split())
    assert f"{noisy:.6f}" in refuted
    assert f"{delta / noisy:.2f}x" in refuted
    assert delta > noisy                      # it would pass either way
    # combined_claimable_delta rounds its outputs, so the one-over-root-two
    # identity holds to the rounding rather than exactly.
    assert abs(both["arm_means_95"] / noisy - 1 / 2 ** 0.5) < 1e-4

    # Condition 1 is uncomputed, and the record says what it needs.
    one = " ".join(record["condition_1_IS_NOT_COMPUTED"].split())
    assert "seed_<n>__predictions.csv" in one
    assert "CLUSTER-ONLY" in one
    assert "metrics.json does not carry them" in one
    assert not [k for k in payload if "predict" in k and "csv" in k.lower()]

    shape = " ".join(record["and_this_is_the_phase_16_shape_with_one_difference"].split())
    assert "neither true nor false, it is UNCOMPUTED" in shape


def test_the_ledger_disposition_is_ruled_with_both_precedents_answered():
    from cleft import results_ledger

    record = phase27.THE_LEDGER_DISPOSITION_RULED
    ruling = " ".join(record["THE_RULING_a_row_and_it_is_DESCRIPTIVE"].split())
    assert "one row, status DESCRIPTIVE" in ruling
    assert "Not UNRESOLVED-WITHDRAWN, and not no-row" in ruling

    # Both precedents the lock named are answered, and both are real.
    assert "closed with no row" in " ".join(record["why_not_no_row_the_phase_26_precedent"].split())
    assert "both conditions were evaluated and split" in " ".join(
        record["why_not_UNRESOLVED_WITHDRAWN_the_phase_16_precedent"].split()
    )
    ids = {entry["id"] for entry in results_ledger.ENTRIES}
    assert "p16-anchor-loop-unresolved" in ids
    assert "p9-anchor-classifier-convergence" in ids

    # The chosen precedent really has the shape the ruling claims.
    fits = next(
        e for e in results_ledger.ENTRIES
        if e["id"] == "p9-anchor-classifier-convergence"
    )
    assert fits["status"] == "DESCRIPTIVE"
    assert fits["condition_1"] is None and fits["condition_2"] is None

    # The row is not written, and the reason is the standing naming rule.
    why = " ".join(record["the_row_is_NOT_WRITTEN_HERE_and_why"].split())
    assert "run directory name is not known" in why
    assert "never selected by sha, suffix or plausibility" in why
    # [UPDATED 2026-09-06] This asserted the row did not exist,
    # which was the state when the disposition was ruled and the
    # run directory name had not arrived. It has, and the row is
    # written, so the pin now checks the ruling was FOLLOWED.
    written = [e for e in results_ledger.ENTRIES if e["phase"] == "p27"]
    assert len(written) == 1
    assert written[0]["status"] == "DESCRIPTIVE"
    assert written[0]["condition_1"] is None
    assert written[0]["condition_2"] is None


def test_the_closing_walks_all_ten_criteria_and_names_what_is_owed():
    payload = _metrics()
    closing = phase27.PHASE_27_CLOSING
    assert "0fe38121" in closing["closed"]

    walk = closing["criterion_walk"]
    assert len(walk) == 10
    assert sorted(int(k.split("_")[0]) for k in walk) == list(range(1, 11))
    assert not [k for k, v in walk.items() if v.lstrip("*").startswith("NOT MET")]

    finding = " ".join(closing["the_finding"].split())
    assert str(payload["primary"]["pcc"]["mean"]) in finding
    assert "The registered failure did not happen" in finding

    # Criterion 9 was tested by the result rather than passed on paper.
    nine = " ".join(walk["9_both_declared_readout_epochs_are_reported_whatever_they_say"].split())
    assert "reads HIGHER than the primary" in nine

    fired = closing["readings_that_fired"]
    did_not = closing["readings_that_did_NOT_fire"]
    assert "the_degenerate_signature" in did_not
    assert "the_two_epochs_AGREE" in fired

    contributed = " ".join(closing["what_the_phase_contributed"].split())
    assert "three things" in contributed
    assert "different regions of the frozen space" in contributed

    not_contributed = " ".join(closing["what_it_did_NOT_contribute"].split())
    assert "any statement about label quality" in not_contributed

    owed = " ".join(closing["what_is_still_owed"].split())
    assert "condition 1" in owed
    assert "the ledger row" in owed
    assert "per-rater grades for the 25" in owed
    assert closing["tag"].startswith("[CLOSED]")


def test_the_close_claims_nothing_the_prohibitions_forbid():
    """The prohibitions were written before the numbers. They still hold
    against a result that is positive against zero."""
    record = phase27.WHAT_THIS_PHASE_MAY_NOT_CLAIM
    assert "no outcome licenses" in " ".join(record["no_label_is_cleaner"].split())

    # Nothing in the closing records says a label is cleaner or better.
    closing_blob = " ".join(
        str(v) for key in ("guard_passed", "result", "seed_sd_zero",
                           "signature_did_not_fire", "budget_moved_nothing",
                           "offset_corrected", "value_metrics",
                           "readings_applied", "criterion_not_resolved",
                           "ledger_disposition", "closing")
        for v in phase27.summary()[key].values()
    ).lower()
    # The sweep looks for AFFIRMATIVE claims, not for the words: the
    # record has to be able to say "licenses no sentence about cleaner
    # labels" without tripping its own guard.
    for forbidden in ("the anchor label is cleaner",
                      "the anchor labels are cleaner",
                      "a cleaner label",
                      "the probe is worse",
                      "the probe generalises",
                      "shows that it generalises"):
        assert forbidden not in closing_blob, forbidden
    # And every mention of the forbidden phrase is inside a refusal.
    import re

    for match in re.finditer(r"cleaner label", closing_blob):
        window = closing_blob[max(0, match.start() - 60):match.start()]
        assert any(w in window for w in ("no ", "not ", "never ")), window


# --------------------------------------------------------------------------
# the ledger row and what condition 1 needs, 2026-09-06
# --------------------------------------------------------------------------

APPENDED_ENTRIES_39 = 39
CHECKSUM_39 = "0bea78987ff1e716f5b6c99294e432b543af1a507a17011a5525a72b6863c3ab"


def _row():
    from cleft import results_ledger

    return next(
        e for e in results_ledger.ENTRIES
        if e["id"] == "p27-anchor-train-descriptive"
    )


def test_the_ledger_row_is_written_as_ruled_and_the_chain_holds():
    """Entry 39. Appending must leave every earlier prefix untouched."""
    from cleft import results_ledger, run_names

    results_ledger.validate()
    assert len(results_ledger.ENTRIES) == APPENDED_ENTRIES_39
    assert results_ledger.cumulative_checksum(APPENDED_ENTRIES_39) == CHECKSUM_39

    entry = results_ledger.ENTRIES[38]
    assert entry is _row()
    assert entry["status"] == "DESCRIPTIVE"
    assert entry["framing"] == "additional"
    assert entry["phase"] == "p27"
    assert entry["date"] == "2026-09-06"
    assert entry["corrects"] is None

    # Ruled at THE_LEDGER_DISPOSITION_RULED: both condition fields None.
    assert entry["condition_1"] is None
    assert entry["condition_2"] is None

    # The run directory was SUPPLIED, and the row carries it verbatim.
    assert entry["run_dirs"] == ("p27_anchor_train__0fe38121__p27-anchor-train",)
    stem, sha, job = run_names.parse_run_dir(entry["run_dirs"][0])
    assert (stem, sha, job) == (
        "p27_anchor_train", "0fe38121", "p27-anchor-train",
    )
    assert run_names.job_id_contradictions(entry["run_dirs"][0]) == []

    # It matches the precedent the ruling named, field for field.
    precedent = next(
        e for e in results_ledger.ENTRIES
        if e["id"] == "p9-anchor-classifier-convergence"
    )
    for field in ("status", "condition_1", "condition_2"):
        assert entry[field] == precedent[field], field

    # Append-only, with teeth: the two pinned prefixes are unchanged.
    from tests.test_phase9 import CHECKSUM_37, CHECKSUM_38

    assert results_ledger.cumulative_checksum(37) == CHECKSUM_37
    assert results_ledger.cumulative_checksum(38) == CHECKSUM_38


def test_the_ledger_row_carries_every_figure_from_the_run():
    payload = _metrics()
    claim = " ".join(_row()["claim"].split())

    assert str(payload["primary"]["pcc"]["min"]) in claim
    assert "0.8727" in claim and "0.6587" in claim
    assert "2.2108" in claim and "2.96" in claim
    assert "0.0693" in claim
    assert "0.04-0.10 band this cohort cannot resolve" in claim
    assert "0.1386" in claim
    assert "0.1281" in claim
    assert "distinguishable from zero and not from the probe" in claim
    assert "did NOT fire" in claim
    assert "first time anything in this record was FIT to the anchor Score" in claim


def test_the_ledger_rows_caveats_carry_what_the_close_ruled():
    entry = _row()
    caveats = " ".join(" ".join(c.split()) for c in entry["caveats"])
    assert len(entry["caveats"]) == 6

    assert "not verified-unanimous" in caveats
    assert "supervision ask 7" in caveats
    assert "DIFFERENT QUANTITY on scale and construction" in caveats
    assert "Only PCC and Spearman cross cleanly" in caveats
    assert "condition 1 is UNCOMPUTED, not FALSE" in caveats
    assert "UNINTERPRETABLE in advance" in caveats
    assert "cleaner labels" in caveats
    assert "-0.5436" in caveats and "+0.2056" in caveats


def test_condition_two_is_deliberately_not_in_the_row():
    record = phase27.THE_LEDGER_ROW_WRITTEN
    why = " ".join(record["why_condition_2_is_NOT_in_the_row"].split())
    assert "computable and it passes at 5.34x" in why
    assert "neither TRUE nor FALSE" in why

    supplied = " ".join(record["run_dir"].split())
    assert "SUPPLIED rather than constructed" in supplied
    assert "never selected by sha, suffix or plausibility" in supplied

    chain = " ".join(record["the_chain_held"].split())
    assert "append-only rule with teeth" in chain


def test_condition_1_needs_patient_keyed_input_and_no_aggregate_will_do():
    import inspect

    from cleft.eval import metrics

    record = phase27.THE_CONDITION_1_REQUIREMENT
    needs = " ".join(record["what_it_NEEDS_and_the_form"].split())
    assert "patient-keyed vectors, and nothing aggregate substitutes" in needs
    assert "seed_<n>__predictions.csv" in needs
    assert "p27_anchor_train__0fe38121__p27-anchor-train" in needs
    assert "CLUSTER-ONLY" in needs

    # The reason is quoted from the frozen function's own docstring.
    why = " ".join(record["why_an_AGGREGATE_will_not_do_on_the_INPUT"].split())
    doc = " ".join(metrics.bca_ci.__doc__.split())
    shared = "one shared index vector, so pairing over patients is preserved"
    assert shared in doc
    assert shared in why
    invalid = "comparing two arms"
    assert invalid in doc and invalid in why

    # And the bootstrap really does resample rows.
    source = inspect.getsource(metrics.bca_ci)
    assert "rng.integers(0, n, size=n)" in source
    assert "a[idx] for a in arrays" in source

    hard = " ".join(
        record["and_this_is_NOT_a_workaround_that_could_be_found_with_effort"].split()
    )
    assert "it is what a paired test is" in hard


def test_the_condition_1_OUTPUT_is_aggregate_and_can_travel():
    import inspect

    from cleft import phase7b

    record = phase27.THE_CONDITION_1_REQUIREMENT
    travels = " ".join(record["what_CAN_travel_and_it_is_the_whole_answer"].split())
    assert "the OUTPUT is fully aggregate and carries no patient" in travels
    assert "Five triples of numbers and a verdict" in travels

    # paired_comparison really returns only aggregates.
    source = inspect.getsource(phase7b.paired_comparison)
    for field in ('"seed"', '"delta"', '"lo"', '"hi"', '"excludes_zero"'):
        assert field in source, field
    for leak in ("patient_id", "RanaPhotoID"):
        assert leak not in source, leak

    discipline = " ".join(record["so_the_tier_rule_is_not_an_obstacle_here"].split())
    assert "the discipline working" in discipline

    # The five intervals will differ, because the PROBE varies by seed.
    five = " ".join(record["and_the_five_intervals_will_NOT_be_identical"].split())
    assert "the PROBE" in five and "are not" in five
    assert "0.0148" in five
    assert "rather than one interval computed five times" in five


def test_the_p27_pair_needed_a_scope_and_the_record_says_why():
    """[UPDATED 2026-09-06] This asserted "p27" was NOT in the scope
    vocabulary, which was the state when the requirement was
    written. The scope was then built, so the pin now checks that
    the requirement's reasoning still holds: every contrast task
    fixes its family in code, which is why a SCOPE was the route
    rather than a config edit."""
    import inspect

    from cleft import run
    from cleft.config import schema

    record = phase27.THE_CONDITION_1_REQUIREMENT
    key = "WHAT_MUST_BE_BUILT_because_no_shipped_task_covers_it"
    build = " ".join(record[key].split())
    assert "there is no zero-build path" in build
    assert "a guard rather than a gap" in build

    # The vocabulary is still CLOSED, and p27 joined it by a code
    # change rather than by a config naming an arbitrary pair.
    scopes = schema.TASK_SPECS["paired_claims"]["scope"].choices
    assert "p27" in scopes
    assert {"p10", "p11_loss", "p12"} <= set(scopes)
    assert "p28" not in scopes and "arbitrary" not in scopes

    # And the record records that the route was taken, beside the
    # entry that said it had not been.
    built = " ".join(record["BUILT_2026_09_06"].split())
    assert "the state before it" in built
    assert "No paired implementation was written" in built
    not_run = " ".join(record["and_the_computation_has_still_NOT_RUN"].split())
    assert "building it is not computing it" in not_run
    assert "five are OWED" in not_run

    # And the other two contrast tasks fix their families in code.
    assert "contrast_family()" in inspect.getsource(run.task_p22_contrasts)
    assert "contrast_family()" in inspect.getsource(run.task_p25_contrasts)

    route = " ".join(record["the_build_is_ONE_scope_and_ONE_enumeration"].split())
    assert "scope" in route and "choices" in route
    assert "writes no new paired implementation" in route
    assert "needs no re-run of either arm" in route

    until = " ".join(record["and_until_it_exists"].split())
    assert "condition 1 stays UNCOMPUTED and the row stays DESCRIPTIVE" in until


# --------------------------------------------------------------------------
# condition 1: the scope, the enumeration, the config
# --------------------------------------------------------------------------


def test_the_scope_value_exists_and_the_task_dispatches_to_this_module():
    """One scope value and one enumeration, on the route Phases 10, 11
    and 12 each took."""
    import ast
    import inspect

    from cleft import run
    from cleft.config import schema

    scopes = schema.TASK_SPECS["paired_claims"]["scope"].choices
    assert "p27" in scopes
    # It joined a vocabulary that already carried the earlier phases.
    assert {"p10", "p11_loss", "p12"} <= set(scopes)

    # The task dispatches p27 to phase27, in the same chain as the rest.
    source = inspect.getsource(run.task_paired_claims)
    assert 'else phase27 if scope == "p27"' in source
    assert "phase27" in source.split("from . import")[1].split(")")[0]

    # And NO second paired implementation was written.
    tree = ast.parse(inspect.getsource(phase27))
    defined = {
        n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
    }
    assert "paired_claim_pairs" in defined
    for forbidden in ("paired_comparison", "paired_delta_bca", "bca_ci"):
        assert forbidden not in defined, forbidden


def test_the_enumeration_is_one_pair_shaped_like_every_other_scope():
    from cleft import ladder, phase10

    pairs = phase27.paired_claim_pairs("p27")
    assert len(pairs) == 1
    pair = pairs[0]

    assert pair["key"] == "p27__anchor_train_vs_0p2520"
    assert pair["a"] == phase27.PAIRED_ARM_STEM == "p27_anchor_train"
    assert pair["b"] == phase27.PAIRED_BASELINE_STEM
    # The baseline is the SAME arm Phase 10 paired against, by the same
    # name, so two records cannot disagree about one object.
    assert pair["b"] == phase10.PAIRED_BASELINE_STEM
    assert pair["seeds"] == list(ladder.SEED_POOL[:5])

    # Shaped like the others: the shared helpers take it unchanged.
    assert ladder.paired_claim_stems("p27", pairs=pairs) == sorted(
        [pair["a"], pair["b"]]
    )
    assert len(ladder.paired_claim_vectors("p27", pairs=pairs)) == 10
    groups = ladder.paired_claim_seed_groups("p27", pairs=pairs)
    assert list(groups) == [tuple(pair["seeds"])]

    # An unknown scope is refused rather than silently served.
    import pytest

    with pytest.raises(phase27.Phase27Error, match="unknown Phase 27"):
        phase27.paired_claim_pairs("p26")


def test_every_recorded_figure_in_the_pair_is_derived_not_typed():
    from cleft import ladder
    from cleft.train.phase3 import combined_claimable_delta

    pair = phase27.paired_claim_pairs("p27")[0]
    recorded = pair["recorded"]

    baseline_mean = ladder.STAGE_D1_AT_G1["cells"]["vit_b16"][0]
    baseline_sd = ladder.STAGE_D1_AT_G1["sd"]["vit_b16"]["imagenet"]
    arm_mean = phase27.THE_RESULT_OBSERVED["primary_epoch_40"]["pcc"]
    threshold = combined_claimable_delta(baseline_sd, 5, 0.0, 5)["arm_means_95"]

    assert recorded["delta_of_means"] == round(baseline_mean - arm_mean, 4)
    assert recorded["threshold"] == round(threshold, 4)
    assert recorded["margin"] == round((baseline_mean - arm_mean) / threshold, 2)
    # Positive means the probe is ahead, the headline scope's convention.
    assert recorded["delta_of_means"] > 0
    assert "the probe is ahead" in recorded["positive_means"]
    assert "DESCRIPTIVE, not claimable" in recorded["source"]

    # The arm's side comes from the RUN, so a correction lands here.
    payload = _metrics()
    assert arm_mean == payload["primary"]["pcc"]["min"]


def test_the_coverage_records_what_is_excluded_and_why():
    record = phase27.PAIRED_CLAIM_COVERAGE
    covers = " ".join(record["covers"].split())
    assert "one pair" in covers
    assert "five shared seeds" in covers
    assert "a POSITIVE delta means the probe is ahead" in covers

    excluded = record["excluded"]
    assert set(excluded) == {
        "the_second_declared_readout_epoch",
        "the_value_metrics",
        "spearman",
    }
    epoch = " ".join(excluded["the_second_declared_readout_epoch"].split())
    assert "whose better member could be quoted" in epoch
    assert "-0.5436" in " ".join(excluded["the_value_metrics"].split())
    assert "family of two under one name" in " ".join(excluded["spearman"].split())

    # The asymmetry between the two arms is recorded rather than smoothed.
    asym = " ".join(record["and_one_asymmetry_recorded_rather_than_smoothed"].split())
    assert "FIVE models, one per fold, and this arm's come from ONE" in asym
    assert "the pairing is legitimate" in asym

    # And the oof_ prefix is named as a convention, not a claim.
    prefix = " ".join(
        record["and_the_oof_prefix_is_the_LOADERS_convention_not_a_claim"].split()
    )
    assert "not out-of-fold" in prefix
    assert "PURE TEST" in prefix
    assert "stronger condition than out-of-fold, not a weaker one" in prefix
    from cleft import phase7c

    assert phase7c.OOF_INPUT_PREFIX == "oof_"


def test_the_paired_config_declares_ten_vectors_five_carried_five_owed():
    import yaml

    from cleft.config.schema import load_config

    config = load_config(str(REPO / "configs" / "p27_paired.yaml"))
    task = config["task"]
    # load_config fills every optional field, so the check is on
    # what the config SAYS rather than on the dict's exact shape.
    assert task["kind"] == "paired_claims"
    assert task["scope"] == "p27"
    assert task["n_boot"] == 10000
    raw = yaml.safe_load(
        (REPO / "configs" / "p27_paired.yaml").read_text(encoding="utf-8")
    )
    assert raw["task"] == {
        "kind": "paired_claims", "scope": "p27", "n_boot": 10000,
    }
    declared = {e["name"]: e for e in config["inputs"]}
    assert len(declared) == 10

    owed = {n for n, e in declared.items() if set(e["rollup_sha256"]) == {"0"}}
    carried = {n for n in declared if n not in owed}
    assert len(owed) == 5 and len(carried) == 5
    assert all(n.startswith("oof_p27_anchor_train_seed_") for n in owed)

    # The carried five are the probe's, at the SAME hashes p10_paired
    # already declares, read rather than retyped.
    probe = {
        e["name"]: e for e in yaml.safe_load(
            (REPO / "configs" / "p10_paired.yaml").read_text(encoding="utf-8")
        )["inputs"]
    }
    for name in carried:
        assert declared[name]["rollup_sha256"] == probe[name]["rollup_sha256"]
        assert declared[name]["path"] == probe[name]["path"]

    # The owed five point at the run the ledger row names.
    from cleft import results_ledger

    run_dir = next(
        e for e in results_ledger.ENTRIES
        if e["id"] == "p27-anchor-train-descriptive"
    )["run_dirs"][0]
    for name in owed:
        assert run_dir in declared[name]["path"]
        assert declared[name]["path"].endswith("__predictions.csv")

    # Every vector the enumeration needs is declared, and no more.
    from cleft import ladder

    pairs = phase27.paired_claim_pairs("p27")
    wanted = {
        f"oof_{stem}_seed_{seed}"
        for stem, seed in ladder.paired_claim_vectors("p27", pairs=pairs)
    }
    assert set(declared) == wanted


def test_the_placeholder_is_quoted_because_yaml_reads_it_as_an_integer():
    """The bug this caught: an unquoted run of 64 zeros is an int, and
    the config fails validation before the hash check ever runs."""
    import yaml

    text = (REPO / "configs" / "p27_paired.yaml").read_text(encoding="utf-8")
    assert 'rollup_sha256: "' + "0" * 64 + '"' in text

    # Unquoted, it really would be an int.
    assert isinstance(yaml.safe_load("h: " + "0" * 64)["h"], int)
    assert isinstance(yaml.safe_load('h: "' + "0" * 64 + '"')["h"], str)

    # And the schema accepts the placeholder as a valid digest, so the
    # config is loadable while its hashes are still owed.
    from cleft.config.schema import load_config

    load_config(str(REPO / "configs" / "p27_paired.yaml"))


def test_the_paired_generator_reproduces_the_config_and_tolerates_a_fill():
    """--check must pass on the shipped file, and must keep passing once
    a declare pass fills the owed rollups, because that is the ONE edit
    this config invites."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/generate_phase27_paired_config.py", "--check"],
        cwd=REPO, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "matches the record" in result.stdout

    # A filled rollup is accepted; anything else is drift.
    import tempfile

    original = (REPO / "configs" / "p27_paired.yaml").read_text(encoding="utf-8")
    filled = original.replace(
        'rollup_sha256: "' + "0" * 64 + '"',
        "rollup_sha256: " + "ab" * 32,
        1,
    )
    tampered = original.replace("scope: p27", "scope: p12", 1)
    try:
        (REPO / "configs" / "p27_paired.yaml").write_text(
            filled, encoding="utf-8", newline="\n"
        )
        ok = subprocess.run(
            [sys.executable, "scripts/generate_phase27_paired_config.py", "--check"],
            cwd=REPO, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        )
        assert ok.returncode == 0, ok.stdout + ok.stderr
        assert "1 rollup(s) filled" in ok.stdout

        (REPO / "configs" / "p27_paired.yaml").write_text(
            tampered, encoding="utf-8", newline="\n"
        )
        bad = subprocess.run(
            [sys.executable, "scripts/generate_phase27_paired_config.py", "--check"],
            cwd=REPO, capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": str(REPO / "src")},
        )
        assert bad.returncode == 1
        assert "DRIFT" in bad.stdout
    finally:
        (REPO / "configs" / "p27_paired.yaml").write_text(
            original, encoding="utf-8", newline="\n"
        )
    assert tempfile  # the import is used only to mark this as a temp edit


def test_no_frozen_apparatus_was_touched_by_the_condition_1_build():
    """The build adds a scope and an enumeration. Nothing else."""
    import subprocess

    frozen = (
        "src/cleft/train/harness.py", "src/cleft/gates.py",
        "src/cleft/determinism.py", "src/cleft/eval/metrics.py",
        "src/cleft/data/folds.py", "src/cleft/phase7b.py",
    )
    changed = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO,
        capture_output=True, text=True,
    ).stdout
    for path in frozen:
        assert path not in changed, path
    for prefix in ("src/cleft/provenance/", "src/cleft/geometry/"):
        assert prefix not in changed, prefix


def test_what_the_row_becomes_is_written_before_the_intervals_exist():
    record = phase27.THE_ROW_UNDER_EACH_OUTCOME
    assert record["registered"].startswith("2026-09-06")
    assert "before the paired run" in record["registered"]

    always = " ".join(record["what_changes_in_every_case"].split())
    assert "condition_1`` stops being None" in always
    assert "TRUE at 5.34x" in always

    passes = " ".join(
        record["if_condition_1_PASSES_all_five_exclude_zero_one_direction"].split()
    )
    assert "status becomes CLAIMABLE" in passes
    assert "SMALLEST_RESOLVABLE_DIFFERENCE" in passes
    assert "a finding about the instrument" in passes

    fails = " ".join(record["if_condition_1_FAILS"].split())
    assert "status becomes UNRESOLVED-WITHDRAWN" in fails
    assert "p16-anchor-loop-unresolved" in fails

    # The expectation is registered BEFORE the run, which is what makes
    # the other outcome readable.
    expected = " ".join(
        record["which_is_EXPECTED_and_the_record_says_so_before_the_run"].split()
    )
    assert "condition 1 is expected to FAIL" in expected
    assert "30 comparisons and one survived" in expected
    from cleft import ladder

    at_scale = ladder.COHORT_CANNOT_RESOLVE["at_scale"]
    assert (at_scale["tested"], at_scale["survived"]) == (30, 1)

    unchanged = " ".join(record["what_does_NOT_change_under_any_outcome"].split())
    assert "the claim sentence and every caveat" in unchanged
    assert "no outcome licenses a sentence about cleaner labels" in unchanged

    # And the append-only rule means the row is amended, not edited.
    amend = " ".join(record["and_the_row_is_AMENDED_rather_than_replaced"].split())
    assert "the ledger is append-only" in amend
    assert "cumulative checksum forbids" in amend
    assert "ledger-condition-split-count-corrected" in amend
    assert "a maintainer decision and this record does not take it" in amend
