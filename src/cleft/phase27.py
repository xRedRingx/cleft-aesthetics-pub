"""Phase 27: the anchor set as a training set. RESTATED 2026-09-06, not
locked.

Scheduled by ``phase25.PHASE_SEQUENCE_EXTENDED_8`` as SCHEDULED, NOT
REGISTERED. This module is the restate the amendment's first binding
clause requires, which is where scope lives and where the amendment
deliberately did not put it.

**[LOCKED AND BUILT 2026-09-06.]** The restate above is preserved and
the sentence it carried, that nothing is locked and nothing is built,
is no longer true. ``EXIT_CRITERIA`` is locked with its readings at
``READINGS_AT_THE_LOCK``, ``EXIT_CRITERIA_DRAFT`` is kept beside it,
and the seven build items are built: the task is registered, the
config is generated and fully declared, and the disjointness guard is
four parts that can each fail rather than a comment.

**[RUN AND CLOSED 2026-09-06.]** The sentence this paragraph carried,
that nothing has run and the ledger holds no row, was true when the
phase was built and is not now. The run is
``p27_anchor_train__0fe38121__p27-anchor-train`` and the ledger holds
entry 39, ``p27-anchor-train-descriptive``.

**Every reading below was still written before any number**, which is
the part that mattered and the part that survives. What the run then
did to them is at ``THE_READINGS_APPLIED``, including the one case
they did not anticipate.

**One thing is still owed**: PLAN section 4.3's condition 1, a
per-seed paired BCa over the 237 that needs the CLUSTER-ONLY
prediction files and no re-run. Until it exists the contrast against
the probe is UNRESOLVED and the row is DESCRIPTIVE
(``THE_CONDITION_1_REQUIREMENT``).

**The amendment's case FOR this phase is not carried forward.** It rests
on the anchor labels being a consensus and therefore cleaner, and the
record refuses that word at two homes and carries it as an open
supervision ask. The motivation is rewritten at
``THE_MOTIVATION_REWRITTEN`` rather than inherited, and the rewrite is
weaker than the amendment's and is meant to be.

The phase would be the first time in this record that anything is FIT to
the anchor Score. Everything before it used the 25 as fixed references
that no model trained on.
"""

from __future__ import annotations

from . import phase21

# --------------------------------------------------------------------------
# the reckoning
# --------------------------------------------------------------------------

#: **[RECKONED 2026-09-06] What the record already holds, stated before
#: the phase's own case.**
#:
#: The counter-evidence is the phase's own, so it is held to the standard
#: the record holds its cases to.
THE_RECKONING = {
    "reckoned": "2026-09-06, before any scope was written",

    "nothing_has_ever_trained_on_the_25": (
        "**verified across the repository, and the three registrations "
        "that could have crossed the line each refuse it in writing.** "
        "``phase9.DEALL_REFERENCE_REGISTERED`` task shape: 'score with "
        "arm A's frozen heads, no refitting; all 25 heads score every "
        "image (no fold owns an external image)'. "
        "``phase9.PROTOTYPE_CLASSIFIER_REGISTERED``: 'the 25 graded "
        "images are TRUSTED REFERENCES ... NO TRAINING ANYWHERE'. "
        "``phase15.ANCHOR_LOOP_REGISTERED``: 'Trained on TRAINING FOLDS "
        "ONLY, evaluated out-of-fold on cleft_v1'. **So the crossing "
        "this phase schedules is genuinely untested, and that is the "
        "one thing about it the record supports without qualification**"
    ),
    "and_the_artifact_forbids_it_by_construction": (
        "the anchor artifact's own metadata carries "
        "'anchors_belong_to_no_fold': 'the 25 are references, not "
        "cohort patients; no fold column exists here BY CONSTRUCTION "
        "(EXIT_CRITERIA criterion 1)'. **That absence is an enforced "
        "exit criterion of a closed phase, not an oversight**, and "
        "``phase16.PHASE_16_CLOSING`` criterion 1 records it as MET IN "
        "CODE with three separate tests behind it. A phase that trains "
        "on the 25 does not violate that criterion, which was about "
        "fold honesty within Phase 16, but it does have to say why it "
        "is reaching past a guard another phase locked"
    ),

    "the_25_have_been_scored_against_four_times": (
        "**and every one of the four ran the crossing in the OPPOSITE "
        "direction**, taking a model fitted to the cohort target and "
        "scoring it against the anchor Score. (1) The external "
        "reference, arm A's frozen heads, **PCC 0.2512 at n=25** "
        "(``phase9.PHASE_9_CLOSING`` criterion 3). (2) The prototype "
        "classifier, eight cells, **acc3 0.3333 to 0.3629 and PCC "
        "0.1494 to 0.1823** against chance 0.3333 and majority 0.502 "
        "(``phase9.PROTOTYPE_CLASSIFIER_OBSERVED``). (3) The anchor "
        "loop, **0.2040 sd 0.0345** with an untrained identity readout "
        "at **0.2151**, against the probe's 0.2520 sd 0.0148 "
        "(``phase16.PHASE_16_CLOSING``). (4) Phase 10's faithful arm on "
        "the benchmark shape, **four of five raters constant** "
        "(``phase10.FAITHFUL_ARM_CLOSING``)"
    ),

    "the_nearest_measurement_runs_AGAINST_the_phase": (
        "**``phase16.REPAIR_WITHOUT_TRANSFER`` is the closest thing the "
        "record has to a result on this question, and it is a "
        "dissociation.** A learned metric correction 'clusters the "
        "anchors better (4/25 -> 6.76/25 mean) while being mildly "
        "harmful to cohort PCC (identity 0.2151 -> loop 0.2040, 2 of 5 "
        "seeds above identity)', and the banked sentence is 'The loop "
        "learned something real about the anchor space that does not "
        "transfer to predicting cohort grades'. Its reading is that "
        "'the anchors' neighbourhood geometry and the cohort's grade "
        "geometry are not the same thing in this space'. **That is a "
        "measurement, not a prediction, and it points against this "
        "phase more directly than anything the amendment cites**"
    ),
    "and_the_anchor_set_fails_its_own_premise_within_itself": (
        "``phase9.PROTOTYPE_CLASSIFIER_OBSERVED`` measured anchor "
        "self-consistency at **4/25 Euclidean and 3/25 cosine against a "
        "chance expectation of ~4.75/25**, and its banked reading is "
        "WITHIN-SET: 'the method's premise fails among the trusted "
        "references before any cohort patient is scored'. The "
        "concession is explicit that a softer cross-set framing 'is NOT "
        "in the record' and that the within-set reading 'is the "
        "stronger ground'. **Grade structure was measured absent among "
        "the 25 THEMSELVES**, which is the set this phase proposes to "
        "learn a grade function from"
    ),
    "the_one_refinement_that_softens_it": (
        "``phase9.PROTOTYPE_CLASSIFIER_OBSERVED['readout_refined_2026_"
        "08_29']`` records that part of that failure was the READOUT "
        "rather than the space: the same 25 anchors with a "
        "softmax-expectation readout and nothing trained score 0.2151 "
        "where the best k-nearest-neighbour cell is 0.1823. It is "
        "carried here because the reckoning should not rest on a leg "
        "the record itself softened. **It does not overturn the "
        "conclusion**, and 0.2151 is still below the probe's 0.2520"
    ),

    "what_the_reckoning_therefore_leaves": (
        "**one untested crossing, five convergent nulls around it, and "
        "a measured dissociation pointing the wrong way.** The phase is "
        "worth running because nothing has run it, and for no stronger "
        "reason than that. Any restate claiming a stronger reason is "
        "claiming something this record does not hold"
    ),
}


# --------------------------------------------------------------------------
# the motivation, rewritten rather than inherited
# --------------------------------------------------------------------------

#: **[REWRITTEN 2026-09-06] THE AMENDMENT'S CASE FOR THIS PHASE IS
#: WITHDRAWN AND REPLACED. Nothing is inherited.**
#:
#: The amendment's ``motivated_by_the_case_FOR`` reads: "the anchor
#: labels are cleaner than anything this project trains on. The cohort
#: target is a five-rater mean whose panel agreement is Fleiss kappa
#: 0.1662, banked at ladder.py beside QWK 0.4276. A consensus grade
#: carries no such disagreement inside it".
#:
#: **The second sentence is true and the third is not established.** The
#: kappa is banked and correct. The word "consensus" is not.
THE_MOTIVATION_REWRITTEN = {
    "rewritten": "2026-09-06, at the restate, per binding clause 1",

    "what_the_amendment_claimed": (
        "that the anchor labels are cleaner than the cohort target, "
        "because a consensus grade carries no disagreement inside it "
        "while the cohort's panel mean carries Fleiss kappa 0.1662"
    ),
    "why_it_does_not_survive": (
        "**the case rests entirely on the word consensus, and the "
        "record refuses that word at two homes and carries it as an "
        "open supervision ask.** The amendment says so itself in the "
        "very next key. A motivation that is contradicted in the entry "
        "that states it is not a motivation this phase may carry "
        "forward silently. See THE_CONSENSUS_WORD_REFUSED"
    ),

    "the_honest_position_stated_before_any_number": (
        "**nobody knows how the 25 grades were made.** The record names "
        "a survey-design key document and a CSV column and stops. It "
        "does not name who assigned the grades, how many people "
        "assigned them, or what instrument was used. "
        "``phase9.DEALL_REFERENCE_READS`` gives the whole provenance as "
        "'the survey-design key in Image codes TingLi Pid No (the "
        "survey-design lineage): 25 presentations = 22 unique + 3 "
        "repeats, AOFA tutorial-only; FNGA=2 and FPIA=3 graded in the "
        "CSV, completing all 25'. That is a lineage, not a construction"
    ),
    "and_the_manuscript_claims_something_else_entirely": (
        "**a SELECTION CRITERION over 76 images, not unanimity among "
        "raters.** ``phase9`` ground 4a: the manuscript's claim is 'the "
        "25-of-76 highest-agreement images from a separate 27-surgeon "
        "study', and the concession keeps the two apart because 'they "
        "license different arithmetic'. High agreement among 27 raters "
        "on a separate study is not the same object as the Score column "
        "in this repository, and the record refuses to reconcile them"
    ),
    "and_no_reliability_figure_can_be_computed": (
        "**not withheld, not unmeasured, NOT COMPUTABLE.** "
        "``phase9`` ground 3: a single Score column is one column, and "
        "``mean_inter_rater_r``, ``fleiss_kappa``, ``cronbach_alpha`` "
        "and ``mean_pairwise_qwk`` all raise "
        "ReliabilityError('need at least 2 raters, got 1') through "
        "``reliability._check``. 'Zero within-set rater variance means "
        "there is nothing to average'. **Per-rater grades for the 25 do "
        "not exist in this record**, so the anchor label has no kappa, "
        "no QWK, no inter-rater r and no ceiling, and none of them can "
        "be produced from what is here"
    ),

    "so_the_phase_trains_on_a_label_of_UNVERIFIED_CONSTRUCTION": (
        "**and this restate says so before any number exists, which is "
        "the whole point of saying it here.** The cohort target's noise "
        "is measured and ugly. The anchor target's noise is unmeasured "
        "and unmeasurable from what the record holds. **Those are not "
        "the same situation and the second is not the better one.** A "
        "label whose disagreement is banked at Fleiss 0.1662 is worse "
        "than a clean label and better than a label whose disagreement "
        "is unknown, because the first can be reasoned about"
    ),

    "what_the_phase_is_ACTUALLY_for": (
        "**the crossing is untested and a null would close it.** No "
        "model in this record has ever been fit to the anchor Score, "
        "and the axis stays open until one is. The phase's value is "
        "asymmetric and the asymmetry is registered here rather than "
        "discovered later: **a null closes an axis, and a positive "
        "result would be uninterpretable.** See "
        "THE_ASYMMETRY_REGISTERED"
    ),
    "and_it_is_cheap_which_is_a_reason_to_run_it_and_not_a_reason_to_believe_it": (
        "the anchor embeddings already exist as a declared hashed "
        "artifact and the cohort embeddings already exist, so the phase "
        "is head arithmetic over cached features. "
        "``phase26.THE_RUNTIME_MEASURED`` banked the first runtime of "
        "that shape at **3.843 seconds per seed** at patience 5 over "
        "237 rows. **Cheapness is why it can run before something "
        "expensive, and it is not evidence about the answer**"
    ),

    "tag": "[REWRITTEN] -- the amendment's case is withdrawn, not carried",
}


#: **[RECORDED 2026-09-06] THE WORD THE CASE RESTED ON, AND THE TWO
#: PLACES THE RECORD REFUSES IT.**
#:
#: Quoted rather than summarised, because the amendment's case turns on
#: exactly these sentences.
THE_CONSENSUS_WORD_REFUSED = {
    "recorded": "2026-09-06",

    "the_first_refusal": (
        "``phase9.DEALL_REFERENCE_READS['score']['cannot_assert_from_"
        "disk']``: 'whether this key equals the manuscript's 27-surgeon "
        "highest-agreement consensus; both sources stated, no "
        "reconciliation forced'"
    ),
    "the_second_refusal": (
        "``phase9.PROTOTYPE_CLASSIFIER_REGISTERED``, carried into the "
        "observation's own caveat and into two ledger rows: 'anchor "
        "grades are trusted single grades from the survey lineage, not "
        "verified-unanimous; travels with every quotation'"
    ),
    "and_it_is_an_OPEN_supervision_ask": (
        "**ask 7, ``phase11``, 'unanimity on the anchor grades', "
        "blocking False, records phase9.PHASE_9_CLOSING.** It has been "
        "open since 2026-08-16 and nothing closes it"
    ),
    "the_tag_the_amendment_itself_applied": (
        "the amendment says 'The scheduling carries the word as "
        "REPORTED and the restate may not upgrade it without a source'. "
        "**This restate does not upgrade it. It withdraws the case that "
        "depended on it**"
    ),
    "and_confirming_unanimity_would_NOT_be_enough": (
        "``phase9.what_would_reopen_it`` is explicit: 'the supervision "
        "material's unanimity confirmation ALONE does not reopen it -- "
        "it would close ask 7 and still leave ground 2 and the missing "
        "ceiling arithmetic standing'. **So even a positive answer at "
        "the next supervision meeting does not restore the amendment's "
        "case**, because the case needs a reliability figure and "
        "unanimity is not one"
    ),
    "what_WOULD_restore_it": (
        "**per-rater grades for the 25.** They would make the "
        "reliability computable rather than assumed, which is the first "
        "of the two conditions ``phase9.what_would_reopen_it`` names. "
        "They are not in the record and this phase does not wait for "
        "them"
    ),
}


#: **[REGISTERED 2026-09-06, BEFORE ANY NUMBER] THE ASYMMETRY. A null
#: closes an axis and a positive result would be uninterpretable.**
#:
#: Registered here so that a positive result cannot be read as a
#: discovery after the fact.
THE_ASYMMETRY_REGISTERED = {
    "registered": "2026-09-06, before the phase is locked",

    "a_null_is_INTERPRETABLE": (
        "**if a head fit to the anchor Score does not predict cohort "
        "grades, the reading is available and it is clean.** It joins "
        "the convergent series as a measurement that the anchor label "
        "carries no transferable grade signal in this space, and it "
        "closes the last untested use of the 25. It would also be the "
        "second half of ``phase16.REPAIR_WITHOUT_TRANSFER``, which "
        "measured that the anchor geometry does not transfer, from the "
        "supervised side rather than the metric side"
    ),
    "a_POSITIVE_result_is_NOT_interpretable_and_this_is_the_registration": (
        "**if it correlates, the record cannot say why, because it does "
        "not know how the label was made.** A positive result would be "
        "consistent with the anchor label being a better target, with "
        "it being a differently biased target that happens to align, "
        "with 25 rows landing well by luck at n=5 seeds, and with the "
        "composites' different framing acting as a feature. **The "
        "record cannot separate those, and it will not be able to "
        "separate them after the fact either.** A positive result "
        "therefore does NOT license 'train on cleaner labels', and this "
        "record says so before the number exists rather than after"
    ),
    "why_this_is_registered_rather_than_reasoned_later": (
        "**because the asymmetry is the reason to run the phase, and a "
        "reason discovered after a favourable number is not a reason.** "
        "The record has an error provenance for the shape where an "
        "interpretive frame arrives after the result, "
        "``ladder.SMALLEST_RESOLVABLE_DIFFERENCE['why_this_record_"
        "exists']``. Registering the asymmetry first means a positive "
        "result gets recorded as UNINTERPRETABLE rather than as a win"
    ),
    "and_it_does_not_make_the_phase_not_worth_running": (
        "a phase whose null is clean and whose positive is "
        "uninterpretable is still worth running when the null is the "
        "likely outcome and the axis stays open without it. **What it "
        "is not worth is a scope written as though a positive would "
        "settle something**"
    ),
}


# --------------------------------------------------------------------------
# the label incomparability, ADDRESSED
# --------------------------------------------------------------------------

#: **[ADDRESSED 2026-09-06, NOT INHERITED] THE TWO TARGETS ARE A
#: DIFFERENT QUANTITY, AND THIS PHASE PUTS ONE ON EACH SIDE OF THE
#: CROSSING.**
#:
#: The amendment's ``the_constraint_the_restate_MUST_address`` requires
#: this record. It is the constraint discharged, not repeated.
THE_LABEL_INCOMPARABILITY_ADDRESSED = {
    "addressed": "2026-09-06, at the restate, per the amendment's constraint",

    "the_conceded_ground_quoted": (
        "``phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED`` ground 2: "
        "'**a different quantity on BOTH scale and construction.** The "
        "237's target is the panel MEAN of five integer grades, so it "
        "is continuous on rater-steps of **0.2** (data.softlabels: "
        "soft labels are fractions of five raters, so every value is a "
        "multiple of 0.2). The Deall ``Score`` is an **INTEGER 1-5**. A "
        "correlation against one is not a correlation against the "
        "other'"
    ),

    "the_TRAINING_target": (
        "**the anchor Score. A single integer from 1 to 5, one value "
        "per image, no rater matrix.** Grade spread 3/7/6/6/3 over "
        "grades 1 to 5, frozen at ``phase16.ANCHOR_GRADE_SPREAD`` and "
        "enforced by the artifact writer, which refuses any other "
        "spread with 'these are not the 25 Deall anchors'"
    ),
    "the_EVALUATION_target": (
        "**the cohort panel mean. The arithmetic mean of five integer "
        "grades, continuous on a 0.2 grid.** Declared in every probe "
        "config as ``label: mean``. Its distribution is banked: mean "
        "**2.7544**, sd **0.6587**, range **1.4 to 4.6** "
        "(``phase18``, ``phase10``). Its panel agreement is Fleiss "
        "kappa **0.1662**, with the distance-aware figures QWK "
        "**0.4276** and mean inter-rater r **0.4696** beside it after "
        "the 2026-09-02 unweighted-kappa correction "
        "(``record_audit.THE_KAPPA_LIMITATION``)"
    ),

    "what_the_crossing_MEANS_and_it_is_metric_dependent": (
        "**the crossing does not affect all six readouts the same way, "
        "and treating it as one thing is the error to avoid.** A head "
        "fit to the anchor Score produces predictions in ANCHOR units. "
        "Scoring them against the cohort panel mean asks two different "
        "questions depending on the metric, and the restate separates "
        "them rather than leaving it to the reading"
    ),
    "PCC_SURVIVES_the_crossing": (
        "**because it is scale invariant, which this record verified "
        "rather than assumed.** ``phase21.SCALE_INVARIANCE_PROHIBITION``"
        " verified that rescaling a prediction vector about the cohort "
        "mean 'changes RMSE and leaves PCC EXACTLY UNCHANGED -- "
        "verified here to 0.00e+00 across rescalings of 1.5x, 2.0x and "
        "2.5x'. So a correlation between anchor-unit predictions and "
        "cohort-unit truth is a real correlation and is readable. "
        "**PCC is the only readout that crosses cleanly**"
    ),
    "THE_VALUE_METRICS_DO_NOT_SURVIVE_IT": (
        "**IEM, RMSE and MAE measure the value, and the two targets do "
        "not share a value.** The anchor grade mean is **2.96** (74/25 "
        "from the 3/7/6/6/3 spread, banked at "
        "``phase16.TAU_DECLARED``) and the cohort panel mean is "
        "**2.7544**. **[DERIVED]** the two differ by **0.2056**, so a "
        "model predicting perfectly in anchor units carries a 0.2056 "
        "offset in cohort units before it makes any error at all. "
        "Quoting an IEM or an RMSE across that offset would be "
        "measuring the label difference and attributing it to the model"
    ),
    "THE_RECORD_ALREADY_CAUGHT_THIS_EXACT_CONFLATION_ONCE": (
        "**and it counted it as the NINTH different-quantities "
        "catch, which is the strongest reason this record states the "
        "offset before any number.** "
        "``phase18.P17_IEM_MEASURED['the_ninth_different_quantities_"
        "catch']``: 'the [REASONED] text also said the panel mean "
        "sits near 2.96 -- a CONFLATION: 2.96 is the ANCHOR-GRADE "
        "mean (74/25, the 25 references spread); the cohort's "
        "panel-mean truth is 2.7544. They are different quantities'. "
        "It goes on that reading a prediction mean of 2.935 as "
        "well-calibrated to the cohort would have been the "
        "different-quantities error. **A phase that trains on the "
        "anchor grades produces predictions centred near 2.96 BY "
        "CONSTRUCTION**, so it walks straight into the trap the "
        "record has already sprung nine times"
    ),
    "and_the_ANCHOR_LOOP_chose_the_other_way_and_said_why": (
        "**the closest prior design faced this choice and refused "
        "it.** ``phase16`` decision 1: 'the pull loss and the "
        "readout train against the same CONTINUOUS panel mean as the "
        "0.2520 probe -- the maintainer's ruling, overriding the "
        "lean toward the 3-class variant'. Its rationale is the one "
        "that bears here: 'the continuous target makes the primary "
        "contrast LIKE-FOR-LIKE across target types: loop vs probe "
        "differ in mechanism, not in what they are asked to "
        "predict'"
    ),
    "so_THIS_phase_gives_up_like_for_like_and_must_say_so": (
        "**Phase 27 differs from the probe in WHAT IT IS ASKED TO "
        "PREDICT, which is the thing Phase 16 deliberately held "
        "fixed.** So a Phase 27 arm placed beside the probe is not "
        "the like-for-like contrast the anchor loop was, and the "
        "difference between them confounds mechanism with target. "
        "**This is not a reason not to run the phase. It is the "
        "reason its contrast cannot be read the way the anchor "
        "loop's was**, and it is stated here rather than discovered "
        "when the two numbers sit side by side"
    ),
    "the_other_grouping_the_record_already_ruled_on": (
        "``phase16`` also records that 'a 3-class grouping of the 25 "
        "anchors into 10/6/9 discards exactly the SUB-GRADE "
        "structure registration item 1 exists for'. **So the 25 "
        "collapse to 10/6/9 at the fixed 2.5 and 3.5 edges**, and a "
        "three-class readout on this phase inherits that objection "
        "on the training side as well as the offset on the "
        "evaluation side"
    ),
    "and_the_spreads_differ_too_with_the_confound_stated": (
        "**[DERIVED]** the anchor Score's population sd over its own 25 "
        "images is **1.2159** (sum of squared deviations 36.96 over 25, "
        "square-rooted), against the cohort panel mean's banked "
        "**0.6587**, a ratio of about **1.85**. Averaging five raters "
        "compresses toward the centre and a single grade does not, "
        "which is the expected direction. **The confound is stated "
        "rather than smoothed: these are different image sets, so the "
        "figure does not separate label construction from population**, "
        "and it is recorded as a fact about the two label "
        "distributions and nothing more"
    ),

    "so_the_phase_reports_PCC_as_primary_and_the_value_metrics_ONLY_with_the_offset": (
        "**the ruling this record proposes.** PCC and Spearman cross "
        "and are read normally. IEM, RMSE and MAE are computed and "
        "reported and are NOT comparable to any cohort-trained arm's, "
        "and each one carries the 0.2056 offset beside it. Three-class "
        "accuracy sits between the two: the 2.5 and 3.5 bin edges are "
        "fixed in cohort units, so an anchor-unit prediction meets a "
        "cohort-unit boundary and the readout is affected by the offset "
        "even though it is not a value metric"
    ),
    "and_NOTHING_is_pooled": (
        "``phase9.DEALL_REFERENCE_REGISTERED`` registered the rule "
        "before any number: 'generalisation to a DIFFERENT label ... "
        "external reference only, never pooled with cohort results'. "
        "**This phase inherits that rule in the one direction it does "
        "apply**: no figure from this phase is pooled with a "
        "cohort-trained arm's, and no ledger contrast places them in "
        "one family"
    ),
    "the_error_shape_this_guards_against_has_a_provenance": (
        "``phase20.CROSS_TARGET_ERROR_PROVENANCE``, where two figures "
        "were repeated 'as though the two figures shared a target, "
        "which they do not'. It was caught before anything was built on "
        "it. **This record is the same catch made in advance**"
    ),
}


# --------------------------------------------------------------------------
# the readings, registered both ways before any number
# --------------------------------------------------------------------------

#: **[READINGS WRITTEN AT THE RESTATE 2026-09-06] What each outcome would
#: mean, written before any number exists.**
#:
#: A reading written after a number is not a reading. Three outcomes are
#: registered and the third is a named signature rather than a range.
READINGS = {
    "written": "2026-09-06, before any config exists",

    "if_it_CORRELATES": (
        "**a PCC on the 237 that clears the bound at "
        "THE_CLAIM_BOUNDED.** The reading is registered as "
        "UNINTERPRETABLE per THE_ASYMMETRY_REGISTERED, and the reason "
        "is that the label's construction is unverified, so the record "
        "cannot attribute the result to label quality rather than to "
        "bias, luck or framing. **It would be recorded as a positive "
        "measurement with no mechanism available**, it would reopen "
        "nothing that ``phase9.what_would_reopen_it`` closed, and it "
        "would make per-rater grades for the 25 the highest-value "
        "outstanding ask in the project rather than a housekeeping item"
    ),
    "if_it_NULLS": (
        "**a PCC on the 237 inside the unresolvable band, or below it.** "
        "The reading is that the anchor label carries no transferable "
        "grade signal in this space, and it is clean. It becomes the "
        "supervised half of ``phase16.REPAIR_WITHOUT_TRANSFER``, whose "
        "metric half already measured that the anchor geometry does not "
        "transfer, and it closes the last untested use of the 25. "
        "**This is the outcome the reckoning expects**"
    ),
    "if_the_DEGENERATE_SIGNATURE_appears": (
        "**the registered prediction firing, reported as such and not "
        "as a surprise.** See THE_DEGENERATE_SIGNATURE_REGISTERED. It "
        "is neither of the two outcomes above: it is the fit failing "
        "rather than the label failing, and the two must not be "
        "conflated in the closing"
    ),

    "what_NO_outcome_licenses": (
        "**none of the three licenses a statement that one label is "
        "cleaner than the other**, because the anchor label's "
        "reliability is not computable from what this record holds and "
        "no outcome of this phase makes it computable"
    ),
    "and_the_readings_are_symmetric_in_effort": (
        "the null gets the same amount of writing as the positive, "
        "deliberately. A phase whose expected outcome is written in one "
        "line and whose unexpected outcome is written in five has "
        "already decided which one it wants"
    ),
}


#: **[REGISTERED IN ADVANCE 2026-09-06] THE DEGENERATE SIGNATURE, BY
#: NAME. The record already wrote down what a collapse looks like on
#: this exact set.**
#:
#: Registered so the outcome is attributable rather than mysterious,
#: which is the same discipline ``phase16.TAU_DECLARED`` applied to the
#: same 25 images for a different mechanism.
THE_DEGENERATE_SIGNATURE_REGISTERED = {
    "registered": "2026-09-06, before any number exists",

    "the_signature_quoted_from_its_home": (
        "``phase16.TAU_DECLARED['the_degenerate_mode_registered']``: "
        "'as tau grows the softmax flattens toward a constant predictor "
        "at the anchor-grade mean (74/25 = 2.96 from the 3/7/6/6/3 "
        "spread), which would present as near-zero PCC with 3-class "
        "accuracy near the 0.502 majority floor -- the constant lands "
        "in the mid class'"
    ),
    "why_it_transfers_to_THIS_phase_although_the_mechanism_differs": (
        "**the mechanism there was tau and here it is 25 rows against a "
        "769-parameter head, and the SIGNATURE is the same because the "
        "endpoint is the same object**: a constant predictor at the "
        "anchor grade mean. A linear head's untrained state is by "
        "construction the constant predictor at its training set's "
        "mean, so a fit that learns nothing lands exactly there. The "
        "arithmetic 74/25 = 2.96 is a property of the 3/7/6/6/3 spread "
        "and does not depend on how the collapse happened"
    ),
    "and_the_head_STARTS_there_which_makes_the_signature_exact": (
        "**[SHARPENED 2026-09-06] the untrained state is not near 2.96, "
        "it IS 2.96.** ``EmbeddingHeadBackbone.reset`` sets the weights "
        "to zero and the bias to the training mean, so before the first "
        "step the head predicts 2.96 for every one of the 237. **The "
        "signature is therefore an equality rather than a "
        "resemblance**, and a fit that learns nothing is "
        "distinguishable from one that learns a little by the "
        "prediction spread alone. This only holds because the build "
        "reuses the probe's head, which the first build did not "
        "(THE_INIT_IS_THE_PROBES)"
    ),
    "what_to_look_for": (
        "**PCC near zero on the 237, three-class accuracy near the "
        "0.502 majority floor, and prediction spread near zero across "
        "the 237.** The third is the one that distinguishes a collapse "
        "from a genuine null, and it must be emitted per seed rather "
        "than inferred from the other two"
    ),
    "and_it_is_reported_as_the_prediction_FIRING": (
        "**not as a surprise and not as a failure of the run.** The "
        "record registered this signature on 2026-09-05 for a different "
        "phase on the same 25 images, where it did NOT appear "
        "(``phase16.PHASE_16_CLOSING['tau_degenerate_mode_not_"
        "observed']``: 'the named alternative is discharged'). If it "
        "appears here, the correct sentence is that a registered "
        "prediction fired, and the correct next question is whether the "
        "collapse is the fit or the label"
    ),
    "the_nearest_MEASURED_collapse_in_the_record": (
        "**``phase10_annex.DESIGN_PATHOLOGIES_MEASURED``, and it is at a "
        "training set six times larger than 25.** Over 500 draws per "
        "rater at a 153/28 split it measured 'Undefined PCC: 62-101 "
        "draws per rater, where the model predicted a single constant "
        "grade across all 28 test images, so the correlation has no "
        "denominator'. **That is a collapse to a constant, measured, at "
        "153 training rows.** It is the strongest ground the record has "
        "for expecting one at 25, and it is indirect because the "
        "architecture and the target differ"
    ),
    "and_the_head_already_stops_at_epoch_1_on_152_rows": (
        "``phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION`` records the "
        "precedent as 'a 769-parameter head over frozen embeddings "
        "converges immediately, inner-val never improves, patience "
        "terminates at epoch 1', and ``phase26`` measured **400 of 400 "
        "folds at epoch 1** across a sixteen-cell grid on 2026-09-06. "
        "**At 25 rows there is less to fit, not more**"
    ),
    "and_ONE_adjacent_claim_is_WITHDRAWN_and_may_not_be_recruited": (
        "``phase7c.MECHANISM_NEEDS_THE_LONG_BUDGET`` claimed 'the head "
        "overfits 152 samples within about three epochs' and its status "
        "is **'WITHDRAWN -- see MECHANISM_CLAIM_WITHDRAWN'**, refuted "
        "by its own criterion. It reads like the ground this phase "
        "wants and it is not available. **The epoch-selection "
        "observation survives and the overfitting mechanism does not**"
    ),
}


# --------------------------------------------------------------------------
# the claim, bounded before the numbers
# --------------------------------------------------------------------------

#: **[BOUNDED 2026-09-06, BEFORE ANY NUMBER] What this cohort can and
#: cannot separate, and which significance threshold governs.**
THE_CLAIM_BOUNDED = {
    "bounded": "2026-09-06, before the phase is locked",

    "the_unresolvable_band": (
        "**``ladder.COHORT_CANNOT_RESOLVE``: 'this cohort cannot "
        "resolve PCC differences of 0.04 to 0.10 between arms', "
        "confirmed at scale with 30 tested, 1 survived, 29 withdrawn.** "
        "So a Phase 27 arm landing within about 0.10 of the probe's "
        "0.2520 produces a number this cohort cannot separate from it, "
        "whichever side it falls"
    ),
    "the_smallest_thing_ever_resolved": (
        "**0.1386**, at contrast p7d__b32_512_vs_patch16_512, 5 of 5 "
        "seeds excluding zero at margin 5.32x "
        "(``ladder.SMALLEST_RESOLVABLE_DIFFERENCE``)"
    ),
    "and_that_is_NOT_a_limit": (
        "**``ladder.DETECTION_FLOOR_PROHIBITION`` governs, and it is "
        "cited BY NAME rather than reproduced.** Its own subject "
        "phrase is forbidden everywhere outside its home, which "
        "``tests/test_resolution_floor`` enforces over the whole "
        "tree, so this record obeys the prohibition instead of "
        "quoting it. What it holds, in the clauses that carry no "
        "forbidden phrase: 0.1386 is 'THE SMALLEST RESOLVABLE "
        "DIFFERENCE THIS COHORT HAS DEMONSTRATED' and 'it is a "
        "demonstrated instance, not a limit', while 'the band "
        "between 0.10 and 0.1386 is UNTESTED and its boundary "
        "unlocated'"
    ),
    "and_the_guard_that_caught_this_is_recorded_rather_than_widened": (
        "**[2026-09-06] the first draft of the entry above "
        "reproduced the prohibition verbatim, which put its "
        "forbidden phrase into a second module, and the standing "
        "guard fired.** The entry was rewritten to cite by name. "
        "**The guard's allowed-list was NOT extended to admit this "
        "module**, because widening a guard to accommodate the "
        "record that tripped it is how a guard stops guarding"
    ),

    "the_significance_threshold_that_GOVERNS_this_phase": (
        "**0.1281, the n=237 threshold, because the EVALUATION is on "
        "237.** ``relevance.significance_threshold(237)`` returns "
        "0.128127 and the concession banks it as 0.1281"
    ),
    "and_it_is_NOT_the_0_4179_that_governs_the_existing_figure": (
        "**0.4179 is the n=25 threshold and it governs the external "
        "reference, which is a correlation computed ON the 25.** "
        "``phase9.and_the_n_25_interval_nobody_stated``: 'PCC 0.2512 at "
        "n=25 has a Fisher 95% CI of [-0.1598, 0.5880] -- it SPANS ZERO "
        "-- and the |r| a correlation must reach at n=25 is 0.4179 "
        "(``relevance.significance_threshold(25)``, against 0.1281 at "
        "n=237)'. **Applying 0.4179 to a Phase 27 headline would be an "
        "R2 error**: the training set has 25 rows and the correlation "
        "has 237 points, and the threshold is a property of the "
        "correlation's n and not the fit's"
    ),
    "the_25_rows_bind_somewhere_else_entirely": (
        "**not on the significance threshold, and this record does not "
        "pretend the arithmetic is available.** Twenty five training "
        "rows against a 769-parameter head bear on whether the fit "
        "means anything, and the record holds no closed-form result for "
        "that. ``phase8.PER_FACE_CLAIM_COSTS_N_41`` is the only "
        "closed-form sample-size derivation here and its own scope note "
        "excludes this case: it 'governs per-patient PASS/FAIL claims "
        "and AR's out-of-sample guarantee'. **It may not be recruited "
        "as a sample-size argument against 25 training rows**"
    ),

    "the_claimability_machinery_SURVIVES_the_shape": (
        "**and this is the one structural thing that goes right.** "
        "``train.phase3.combined_claimable_delta`` computes "
        "``arm_means_95 = 1.96*sqrt(sd_a^2/n_a + sd_b^2/n_b)`` over "
        "SEEDS rather than folds, so a fold-free design still supplies "
        "what the criterion needs. ``phase17`` arm A was ledgered on "
        "exactly this shape at **PCC 0.2334 sd 0.0044 over five seeds** "
        "(``phase17.ARM_MEANS``)"
    ),
    "but_the_seed_sd_MEASURES_SOMETHING_DIFFERENT_here": (
        "**and it must not be compared to the cohort arms' band "
        "without saying so.** ``train.phase3.MEASURED_SEED_BAND`` "
        "banks the cohort arm's band as mean 0.2529 sd 0.0137 over ten "
        "seeds and names the quantity "
        "'head_initialisation_and_inner_val_split', with the warning "
        "that 'Any arm with a substantially different training "
        "procedure ... must have its own band measured. Reusing this "
        "one there would be the QWK-as-ceiling mistake in a new place "
        "(R2)'. **A 25-row arm's inner-validation split is a handful of "
        "images, so its seed sd is a different quantity under the same "
        "name and this phase measures its own**"
    ),
    "what_the_phase_may_therefore_claim_at_most": (
        "**a descriptive ordering, and a null with a clean reading.** "
        "Nothing else is reachable. If the arm clears the two "
        "conditions of PLAN section 4.3 against the probe, the "
        "resulting sentence is still bounded by "
        "THE_ASYMMETRY_REGISTERED, which forbids reading a positive as "
        "evidence about label quality"
    ),
}


#: **[PROHIBITED 2026-09-06] WHAT THIS PHASE MAY NOT CLAIM, whatever it
#: measures.**
WHAT_THIS_PHASE_MAY_NOT_CLAIM = {
    "no_label_is_cleaner": (
        "**no outcome licenses 'the anchor labels are cleaner'.** The "
        "anchor label's reliability is not computable from this record "
        "and no result of this phase makes it computable. This is the "
        "amendment's own case and it is prohibited, not merely unproven"
    ),
    "no_external_validation": (
        "``phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED`` conceded the "
        "validation framing, and training on the 25 does not reopen it. "
        "**A number here is not evidence that anything generalises**"
    ),
    "no_pooling_and_no_cross_target_value_comparison": (
        "no figure from this phase is pooled with a cohort-trained "
        "arm's, and no IEM, RMSE or MAE from this phase is placed "
        "beside a cohort-trained one without the 0.2056 offset stated "
        "(THE_LABEL_INCOMPARABILITY_ADDRESSED)"
    ),
    "no_unanimity_upgrade": (
        "the word stays [REPORTED]. The amendment says the restate 'may "
        "not upgrade it without a source' and no source arrived"
    ),
    "the_scale_invariance_prohibition_rides": (
        "``phase21.SCALE_INVARIANCE_PROHIBITION`` is carried by "
        "reference and applies unchanged: nobody may read a shrinkage "
        "or calibration result here as a route to a higher correlation"
    ),
    "and_no_claim_about_the_25_as_a_population": (
        "25 images selected as the highest-agreement 25 of 76 in a "
        "separate study are not a sample of anything this project "
        "defines. **No distributional claim about cleft repair "
        "outcomes may rest on them**"
    ),
}

#: Carried by reference so the prohibition has ONE home.
THE_PROHIBITION = phase21.SCALE_INVARIANCE_PROHIBITION


# --------------------------------------------------------------------------
# feasibility: what the machinery does and what has to be built
# --------------------------------------------------------------------------

#: **[ANSWERED 2026-09-06, FROM THE CODE] WHAT THE FOLD MACHINERY DOES
#: WHEN THE TRAINING SET IS NOT THE EVALUATION SET. It refuses, twice,
#: and for two different reasons.**
THE_FOLD_MACHINERY_ANSWERED = {
    "answered": "2026-09-06, read from data/folds.py rather than reasoned",

    "the_first_refusal_is_STRUCTURAL": (
        "**``data.folds.Folds`` is a partition of ONE population and "
        "cannot represent two.** ``assignments`` is documented as "
        "'patient id -> the fold in which that patient is a TEST case', "
        "and ``train_ids(fold)`` returns the complement inside the same "
        "dict. There is no second id list anywhere in the dataclass. "
        "**A training set disjoint from an evaluation set is not "
        "expressible**, and ``verify`` raises if the fold file names a "
        "patient outside the cohort or omits one inside it"
    ),
    "the_second_refusal_is_a_LABEL_VOCABULARY_and_NOT_a_size_guard": (
        "**[MEASURED 2026-09-06, by running it rather than reading "
        "it] ``data.folds.generate`` refuses the anchor grades, and "
        "the reason is not their number.** ``folds.CLASSES`` is "
        "(0, 1, 2), so the raw 1 to 5 grades are refused outright: "
        "'labels outside the 3-class collapse (0, 1, 2): [3, 4, 5]. "
        "Stratify on class3, not on the raw 1-5 grade.' **The fold "
        "machinery has never accepted a five-grade label from any "
        "phase**, so this is not a refusal aimed at the 25"
    ),
    "and_the_SIZE_guard_does_NOT_fire_which_the_draft_of_this_record_got_wrong": (
        "**[CORRECTED 2026-09-06, before the record shipped] an "
        "earlier draft of this entry said the 3/7/6/6/3 spread has "
        "two classes of three against five folds, so generation "
        "raises. It does not.** Collapsed at the fixed 2.5 and 3.5 "
        "edges the 25 group **10/6/9**, the smallest class has six, "
        "and ``generate`` **SUCCEEDS**, returning five folds of five "
        "with class counts 2/2/1, 2/1/2, 2/1/2, 2/1/2, 2/1/2. **The "
        "too-few-members guard never fires on the 25.** The claim was "
        "reasoned from the guard's text rather than measured, which "
        "is the error this record exists to avoid, and it is "
        "corrected in place rather than deleted"
    ),
    "which_SHARPENS_the_point_rather_than_weakening_it": (
        "**the fold machinery would partition the 25 quite happily "
        "if the grades were collapsed, and that collapse is exactly "
        "what ``phase16`` ruled against**: 'a 3-class grouping of "
        "the 25 anchors into 10/6/9 discards exactly the SUB-GRADE "
        "structure registration item 1 exists for'. So the obstacle "
        "is not that folds refuse the 25. **It is that partitioning "
        "the 25 at all is the wrong design, and the record already "
        "said why on scientific grounds rather than mechanical "
        "ones**"
    ),
    "and_folds_py_is_FROZEN_APPARATUS": (
        "so the vocabulary refusal may not be worked around by "
        "changing it, and the size guard is not in play at all. "
        "**This is not a problem for the phase, because the phase does "
        "not need folds at all**, and that is the point of recording "
        "the refusals rather than treating them as obstacles"
    ),

    "how_the_shape_avoids_the_question": (
        "**every one of the 237 is a pure test case from the start, so "
        "there is nothing to partition.** Cross validation exists to "
        "make each patient a test case exactly once while training on "
        "the rest. When the training rows are a different set entirely, "
        "that machinery has no work to do. The design is fold-free by "
        "construction and touches the frozen apparatus not at all"
    ),
    "what_replaces_the_fold_dimension": (
        "**seeds, and only seeds.** Five folds by five seeds becomes "
        "five seeds, and the fit count per arm drops from 25 to 5. The "
        "seed drives the head initialisation and the inner-validation "
        "split of the 25 training rows, which is the same pair of "
        "sources ``train.phase3.MEASURED_SEED_BAND`` names for the "
        "cohort arms, over a training set six times smaller"
    ),
    "and_the_criterion_still_applies": (
        "``arm_means_95`` is computed over seeds, so PLAN section 4.3 "
        "runs unchanged. See THE_CLAIM_BOUNDED"
    ),
}


#: **[REGISTERED 2026-09-06] THE NEAREST PRECEDENT, AND IT IS CLOSE.**
THE_PRECEDENT_THAT_ALREADY_RUNS = {
    "registered": "2026-09-06, read from run.py",

    "what_it_is": (
        "**``run.task_tstr_regression``, Phase 17 arm A.** Its "
        "docstring: 'frozen ViT-B/16 embeddings of the TPS synthesis "
        "set, a linear head on MAGNITUDE-MAPPED labels ..., evaluated "
        "on all 237 real patients as pure test. Zero real patient "
        "images or labels in training -- the TSTR premise, and the "
        "ordering below is structural: everything before the evaluation "
        "marker touches only the synthetic artifact'"
    ),
    "why_it_is_the_shape_this_phase_wants": (
        "**it trains on a disjoint set with no folds, holds out an "
        "inner-validation fraction per seed, keeps the best checkpoint, "
        "and then scores all 237 with every seed's head.** Its head is "
        "``torch.nn.Linear(train_x.shape[1], 1)``, which at 768 "
        "dimensions is the same 769 parameters this phase would use. "
        "**Phase 27 is that task with the synthesis set replaced by the "
        "anchor artifact**"
    ),
    "and_it_RAN_and_is_banked": (
        "``phase17.ARM_MEANS``: p17_arm_a **PCC 0.2334, sd 0.0044, five "
        "seeds**, against the probe's 0.2520, a delta of -0.0186 that "
        "reproduces the ledger's mean paired delta exactly. **So the "
        "shape is not speculative: a train-on-a-disjoint-set arm has "
        "been run on this cohort and placed under the criterion**"
    ),
    "what_it_does_NOT_establish": (
        "**its training set was synthetic and large, and this one is "
        "real and 25.** The precedent establishes that the SHAPE is "
        "buildable and readable, not that a 25-row version of it fits. "
        "Quoting 0.2334 as an expectation for this phase would be the "
        "R2 error"
    ),
}


#: **[DRAFT 2026-09-06] WHAT HAS TO BE BUILT. Nothing is built by this
#: restate.**
WHAT_MUST_BE_BUILT = {
    "drafted": "2026-09-06, at the restate, and nothing is built",

    "1_a_task_or_a_widening": (
        "**``tstr_regression``'s schema pins ``arm`` to "
        "choices=('p17_arm_a',), so a new arm is REFUSED by the choices "
        "check.** Either the choices widen or a new task kind is "
        "registered. Its training source is ``synth_set``, read through "
        "``_synth_index`` off ``faces.json``, and it would have to "
        "accept an anchor-set artifact instead. **The reader already "
        "exists**: ``phase16.load_anchor_set`` returns values and "
        "metadata and re-checks the 25-row and 3/7/6/6/3 invariants"
    ),
    "2_the_schema_entries": (
        "a task spec for the new kind, and its input reference key "
        "added to ``INPUT_REFERENCE_KEYS`` so guard 3 hashes the anchor "
        "artifact rather than letting it in undeclared"
    ),
    "3_NO_change_to_the_fold_machinery": (
        "**stated as a deliverable rather than an omission.** "
        "``data/folds.py`` is frozen apparatus and this design does not "
        "touch it (THE_FOLD_MACHINERY_ANSWERED)"
    ),
    "4_a_run_directory_axis": (
        "``RUN_DIR_AXES`` pins resolution, backbone, geometry, init and "
        "scheme, and has **no axis for the training population**. Two "
        "arms differing only in what they trained on would not be "
        "distinguishable by directory name, which is a provenance "
        "defect waiting to happen"
    ),
    "5_a_GUARD_that_the_two_sets_are_disjoint": (
        "**there is no assertion anywhere in the repository that would "
        "catch a train-on-A evaluate-on-B task fitting and scoring the "
        "same rows.** In the one precedent the separation rests on "
        "comment markers and statement ordering. **For a phase whose "
        "entire claim is that the two sets are disjoint, that is not "
        "enough**, and the guard is a deliverable rather than a nicety"
    ),
    "6_a_per_seed_prediction_spread_readout": (
        "required by THE_DEGENERATE_SIGNATURE_REGISTERED, because the "
        "spread is what distinguishes a collapse from a null and it "
        "cannot be inferred from PCC and accuracy"
    ),
    "7_the_generator_and_its_check": (
        "a ``scripts/generate_phase27_configs.py`` on the established "
        "pattern, with ``--check`` drift verification, so the config is "
        "derived from this module's locked settings rather than typed"
    ),
    "what_is_ALREADY_there_and_needs_nothing": (
        "**the data.** The anchor embeddings exist as a declared hashed "
        "artifact at rollup "
        "49905fe94c2065e56bcc22887377bb77b3b25ec61dbbf2a2f3aa0790265917c6"
        ", the grades ride inside its ``metadata.json``, and the cohort "
        "embeddings exist at shape (237, 768) "
        "(``phase16.COMPUTE_SHAPE_VERIFIED``). **No extraction is "
        "needed and no pixel is touched**"
    ),
}


#: **[RECORDED 2026-09-06] THE EXTRACTION QUESTION, ANSWERED BEFORE IT
#: IS ASKED.**
THE_TWO_EXTRACTIONS_COMPARED = {
    "recorded": "2026-09-06",

    "same_backbone_same_init_same_geometry": (
        "``configs/p16_extract_anchor_embeddings.yaml`` declares "
        "``backbone: vit_b16`` and ``geometry: g1`` and carries the "
        "cohort embeddings artifact as a declared input. The probe "
        "declares the same backbone, the same init and the same "
        "geometry"
    ),
    "DIFFERENT_code_route_and_the_difference_is_MEASURED": (
        "**the cohort features are read from a cached artifact and the "
        "anchors are staged live from pixels.** "
        "``run.task_extract_anchor_embeddings`` runs "
        "``staging.stage(render.load_image(...))`` then "
        "``FrozenExtractor``, because ``phase16.COMPUTE_SHAPE_VERIFIED`` "
        "established the anchors had no cached features. **The task "
        "then re-extracts a cached cohort row through the live path and "
        "compares**, with banked reference **1.53e-05** and refusal at "
        "**1e-2**, raising 'the anchors would be extracted through a "
        "different path than the cached cohort features'"
    ),
    "so_the_route_equivalence_is_enforced_by_measurement": (
        "at a tolerance about 650 times the observed gap. **The record "
        "does not assert the two extractions are identical. It measures "
        "how far apart they are and refuses beyond a threshold**, which "
        "is the stronger form and is why this phase can use both "
        "artifacts in one fit without a new check"
    ),
    "and_the_column_count_is_NOT_banked": (
        "**a gap, stated.** The cohort side is banked as '(237, 768) "
        "float32'. The anchor artifact's row count is fixed at 25 by "
        "the writer and its column count reaches only the run's own "
        "metrics.json, as ``'shape': list(values.shape)``. No record "
        "quotes it. **This restate does not state it either**, and the "
        "lock should read it from the run rather than assume it"
    ),
    "one_more_difference_that_is_NOT_the_extractor": (
        "``phase9.DEALL_REFERENCE_READS`` records the 25 composites as "
        "'25 files, RGB, aspect ratio 0.86-1.17, 246-712 px' and notes "
        "they 'sit outside the cohort's portrait crop family, which is "
        "weak evidence of a fuller composition'. The staged sheet was "
        "reviewed and PASSED against the eyeball criterion in 2026-08-16. "
        "**The framing difference is a domain shift between the "
        "training and evaluation images and it is recorded here because "
        "it is a candidate explanation for either outcome**"
    ),
}


# --------------------------------------------------------------------------
# settings and exit criteria, both DRAFT
# --------------------------------------------------------------------------

#: **[DRAFT 2026-09-06] SETTINGS NEEDING DECLARATION. Nothing here is
#: chosen. Values are declared at the LOCK, not here.**
SETTINGS_NEEDING_DECLARATION = {
    "drafted": "2026-09-06, and every value below is OPEN",

    "the_rule_that_governs_them": (
        "``phase16.SETTINGS_PROVENANCE``: 'mirror the closest existing "
        "probe-training precedent; report which config each value came "
        "from'. The closest precedent here is "
        "``configs/p17_arm_a``'s task block, because the SHAPE is "
        "``tstr_regression``, and the probe's recipe where the two "
        "agree. **Each value is carried with the config it came from or "
        "declared with its reasoning, and none is invented**"
    ),

    "seeds": (
        "OPEN. The standing five are 1337, 2024, 7, 99, 12345 and every "
        "arm in this record uses them. **Whether five seeds is enough "
        "variation when the training set is 25 rows is a question this "
        "phase raises and does not answer**, and the lock should say "
        "which it chose and why"
    ),
    "inner_val_frac": (
        "OPEN, and **the setting that needs the most reasoning and "
        "has the least precedent.** The probe and the TSTR arm both "
        "use 0.2. At 25 "
        "rows that is FIVE images held out and twenty fitted, and the "
        "held-out five cannot cover the 3/7/6/6/3 spread. **A checkpoint "
        "selected on five images is close to a checkpoint selected on "
        "noise**, and the lock must either declare a different fraction "
        "with its reasoning or declare 0.2 knowing this"
    ),
    "max_epochs_and_patience": (
        "OPEN, and ``phase26`` bears on both. It measured that neither "
        "moves any readout on the cohort arm and that patience is paid "
        "for at a fixed price per epoch. **At 25 rows the head converges "
        "faster still, so the lock should expect epoch 1 and say so "
        "before the run**"
    ),
    "learning_rate_and_weight_decay": (
        "OPEN. ``phase26`` closed the weight decay axis on the cohort "
        "arm across a hundredfold range including zero. **That result "
        "is about 152 training rows and does not transfer to 25**, so "
        "it is a precedent for the value rather than a licence to skip "
        "the declaration"
    ),
    "the_label_field": (
        "**there is no ``label:`` choice to make on the training side, "
        "because the anchor artifact carries exactly one label column.** "
        "The EVALUATION side is ``label: mean``, the cohort panel mean, "
        "and the crossing that creates is the subject of "
        "THE_LABEL_INCOMPARABILITY_ADDRESSED"
    ),
    "readouts": (
        "OPEN, and the crossing constrains them. PCC and Spearman cross "
        "cleanly. IEM, RMSE, MAE and three-class accuracy are affected "
        "by the 0.2056 offset. **Per-seed prediction spread is REQUIRED "
        "rather than optional** (THE_DEGENERATE_SIGNATURE_REGISTERED)"
    ),
}


#: **[DRAFT 2026-09-06] EXIT CRITERIA. A DRAFT, NOT A LOCK.**
#:
#: These are proposed at the restate and locked separately, which is the
#: discipline ``phase26`` followed and ``phase15.PHASE_16_SCHEDULED``
#: established: "A scope written before the phase reads the record is a
#: scope written from memory".
EXIT_CRITERIA_DRAFT = {
    "drafted": "2026-09-06, DRAFT, not locked",

    "1_the_motivation_is_rewritten_not_inherited": (
        "the amendment's case FOR is withdrawn in the record, the "
        "honest motivation stands in its place, and the unanimity word "
        "carries its [REPORTED] tag unchanged "
        "(THE_MOTIVATION_REWRITTEN, THE_CONSENSUS_WORD_REFUSED)"
    ),
    "2_the_crossing_is_stated_before_any_number": (
        "the phase records which readouts cross and which do not, with "
        "the 0.2056 offset derived and shown, before the first config "
        "exists (THE_LABEL_INCOMPARABILITY_ADDRESSED)"
    ),
    "3_the_readings_are_registered_both_ways": (
        "the null and the positive get equal weight, and the positive "
        "is registered as UNINTERPRETABLE in advance "
        "(READINGS, THE_ASYMMETRY_REGISTERED)"
    ),
    "4_the_degenerate_signature_is_named_before_the_run": (
        "the constant predictor at 2.96 with near-zero PCC and accuracy "
        "near 0.502 is registered, and the per-seed prediction spread "
        "is emitted so a collapse is distinguishable from a null"
    ),
    "5_the_disjointness_is_GUARDED_not_asserted": (
        "a test proves no anchor row reaches the evaluation and no "
        "cohort row reaches the fit. **A comment is not a guard**"
    ),
    "6_the_claim_is_bounded_before_the_numbers": (
        "the unresolvable band, the 0.1386 demonstrated instance with "
        "its prohibition, and the n=237 threshold of 0.1281 rather than "
        "the n=25 threshold of 0.4179 (THE_CLAIM_BOUNDED)"
    ),
    "7_the_arm_measures_its_own_seed_band": (
        "the cohort arms' band is not reused, per "
        "``train.phase3.MEASURED_SEED_BAND``'s own warning about "
        "reusing a band across a different training procedure"
    ),
    "8_nothing_is_pooled": (
        "no figure from this phase enters a ledger contrast family with "
        "a cohort-trained arm, and no value metric is placed beside a "
        "cohort-trained one without the offset stated"
    ),
    "9_the_ledger_disposition_is_decided_at_the_lock": (
        "**OPEN.** Whether this phase gets a row at all is a lock "
        "question. ``phase26`` closed with no row. The precedent for a "
        "train-on-disjoint-set arm is ``phase17``, which was ledgered. "
        "**The two precedents point opposite ways and the restate does "
        "not choose**"
    ),
    "10_the_suite_is_green": "the standing criterion on every phase",

    "what_is_NOT_an_exit_criterion": (
        "**a number.** No criterion above requires the phase to reach "
        "any value, and none may be added that does. A phase that must "
        "hit a figure to close is a phase that will hit it"
    ),
}


# --------------------------------------------------------------------------
# the guard and the fit, as functions rather than as comments
# --------------------------------------------------------------------------


class Phase27Error(ValueError):
    """A refusal specific to this phase. Always fatal."""


#: The declared budget and readout epochs, so the config generator and
#: the task read one source rather than two.
BUDGET_EPOCHS = 40
READOUT_EPOCHS = (1, 40)
N_ANCHORS = 25


def disjoint_or_raise(
    train_features, eval_features, *, train_ids, eval_ids
) -> dict:
    """**Part two of the disjointness guard** and the only part that
    compares VALUES rather than declarations.

    **[CORRECTED 2026-09-06] this said it was the only part that can
    catch a mis-declared artifact, which contradicts part one.**
    ``phase16.load_anchor_set`` catches exactly that case and does it
    first: cohort embeddings metadata carries no ``namespace`` key, so
    the anchor loader raises before this function is reached. What is
    true of this part alone is that it reads the ARRAYS, where the
    other three read a declaration, a signature and a call site.

    An identifier comparison alone is VACUOUS here and the record says so
    (``THE_DISJOINTNESS_GUARD_DESIGNED``): anchors are keyed by image
    stem and the cohort by integer patient id, so they cannot collide and
    a check that only compared them would pass by construction. What a
    wrong ``anchor_artifact`` actually looks like is a training row that
    IS an evaluation row, so the features are compared.

    Returns the counts it checked, so a caller can put them in
    ``metrics.json`` and a reader can see the guard ran on the sizes it
    claims. **Raises rather than returning a verdict**: a guard whose
    result can be ignored is a comment with a return value.
    """
    import numpy as np

    train = np.asarray(train_features, dtype=np.float64)
    evaluation = np.asarray(eval_features, dtype=np.float64)
    if train.ndim != 2 or evaluation.ndim != 2:
        raise Phase27Error(
            f"expected two (n, d) matrices, got {train.shape} and "
            f"{evaluation.shape}"
        )
    if train.shape[1] != evaluation.shape[1]:
        raise Phase27Error(
            f"feature widths differ: training {train.shape[1]}, evaluation "
            f"{evaluation.shape[1]}. These are not the same space and the "
            "fit would be meaningless before it was dishonest"
        )
    if len(train_ids) != len(train) or len(eval_ids) != len(evaluation):
        raise Phase27Error(
            f"ids do not align with rows: {len(train_ids)} vs {len(train)}, "
            f"{len(eval_ids)} vs {len(evaluation)}"
        )

    # Identifiers, checked even though the vocabularies differ, because
    # the day they DO collide is the day this matters.
    shared_ids = sorted(
        {str(i) for i in train_ids} & {str(i) for i in eval_ids}
    )
    if shared_ids:
        raise Phase27Error(
            f"{len(shared_ids)} identifier(s) appear in both sets: "
            f"{shared_ids[:5]}. The training set and the evaluation set "
            "must share no row"
        )

    # The check that can actually fire.
    collisions = []
    for row, identifier in enumerate(train_ids):
        matches = np.flatnonzero(
            np.all(evaluation == train[row][None, :], axis=1)
        )
        for match in matches.tolist():
            collisions.append((str(identifier), str(eval_ids[match])))
    if collisions:
        raise Phase27Error(
            f"{len(collisions)} training row(s) are bitwise identical to an "
            f"evaluation row: {collisions[:5]}. Either the declared "
            "artifacts are the same set under two names, or a patient is in "
            "both. Both are fatal to this phase's only claim"
        )

    # **[CORRECTED 2026-09-06] these two were literal zeros.** They are
    # always zero at this point, because every nonzero path has already
    # raised, so the literals were not WRONG. They were a different
    # kind of number: reasoned from control flow, unmarked, sitting in
    # a SHAREABLE artifact beside three fields measured off the arrays.
    # Counting them costs nothing and stays true if a later change ever
    # softens a raise.
    return {
        "n_train": int(len(train)),
        "n_eval": int(len(evaluation)),
        "feature_dim": int(train.shape[1]),
        "shared_identifiers": len(shared_ids),
        "identical_feature_rows": len(collisions),
        "guard": "phase27.disjoint_or_raise",
    }


def fit_head(
    features, targets, *, seed, budget, learning_rate, weight_decay, max_steps
):
    """**Part three of the disjointness guard: the signature.**

    There is no parameter through which cohort truth could reach this
    function. It sees the training features, the training targets, and
    settings. That is the whole surface.

    **The head is ``train.torch_backbone.EmbeddingHeadBackbone``, reused
    rather than reimplemented** (``THE_INIT_IS_THE_PROBES``). That class
    is what ``trainable: head`` trains everywhere in this record, and its
    ``reset`` starts the weights at ZERO and the bias at the TRAINING
    MEAN, so an untrained head predicts the training mean exactly. For
    the 25 anchors that mean is 2.96, which is the constant the
    degenerate signature names.

    **No inner validation, no monitor, no patience, no checkpoint
    selection** (``THE_INNER_VALIDATION_RULED``). The loop runs the
    declared budget and yields the head's state at every epoch, so the
    caller can read the two declared epochs
    (``THE_TWO_READOUT_EPOCHS_DECLARED``) and emit the rest as a
    descriptive trajectory.

    Yields ``(epoch, weights, bias)`` with ``epoch`` one-based and both
    values COPIES, so a later epoch cannot mutate an earlier snapshot.
    """
    import numpy as np

    # The refusals come BEFORE the torch import, so a machine without
    # torch still gets the argument checks. A guard that needs the heavy
    # dependency to say no is a guard that is not tested where the tests
    # run.
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise Phase27Error(
            f"expected (n, d) features and (n,) targets, got {x.shape} and "
            f"{y.shape}"
        )
    if budget < max(READOUT_EPOCHS):
        raise Phase27Error(
            f"budget {budget} is shorter than the declared readout epochs "
            f"{READOUT_EPOCHS}: the phase would have nothing to report at "
            "its own primary"
        )

    from .train.torch_backbone import EmbeddingHeadBackbone

    head = EmbeddingHeadBackbone(
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        max_steps=int(max_steps),
        seed=int(seed),
    )
    head.reset(y)
    for epoch in range(1, int(budget) + 1):
        head.train_epoch(x, y)
        yield epoch, head.head_weights(), float(head._bias.item())


def predict(weights, bias, features):
    """Apply a head's state to a feature matrix. Pure, and separate from
    ``fit_head`` so the evaluation cannot reach back into the fit."""
    import numpy as np

    x = np.asarray(features, dtype=np.float64)
    return (x @ np.asarray(weights, dtype=np.float64) + float(bias)).astype(
        float
    )


# --------------------------------------------------------------------------
# THE LOCK, 2026-09-06
# --------------------------------------------------------------------------

#: **[RULED 2026-09-06, AT THE LOCK] THE INNER VALIDATION SPLIT IS
#: REMOVED. The fit runs a FIXED BUDGET with NO early stopping and NO
#: checkpoint selection.**
#:
#: This is a DEPARTURE from the probe's recipe, and it is recorded as one
#: rather than absorbed. Everything else is mirrored.
THE_INNER_VALIDATION_RULED = {
    "ruled": "2026-09-06, at the lock",

    "the_problem_the_restate_stated": (
        "**the probe and the TSTR arm both hold out ``inner_val_frac`` "
        "0.2. At 25 rows that is FIVE images held out and twenty "
        "fitted.** The five cannot cover the 3/7/6/6/3 spread, because "
        "there are five grades and five images and the draw is random, "
        "so a typical held-out set is missing two or three grades "
        "entirely. **A checkpoint selected on five images is close to a "
        "checkpoint selected on noise**"
    ),
    "and_shrinking_the_fraction_does_not_help": (
        "**considered and rejected.** A smaller fraction holds out four "
        "images or three, which is worse on the same argument. A larger "
        "one takes rows away from a fit that has 25. **There is no "
        "fraction of 25 that supports a selection**, and choosing one "
        "anyway would be declaring a number to satisfy a field rather "
        "than to do a job"
    ),

    "THE_RULING": (
        "**no inner validation split, no monitor, no patience, no "
        "checkpoint selection.** The loop runs a declared fixed budget "
        "and the head at the end of it is the head. The three fields "
        "``inner_val_frac``, ``monitor`` and ``patience`` are ABSENT "
        "from this phase's task spec rather than set to a value, so a "
        "config cannot quietly reintroduce them"
    ),
    "why_absent_rather_than_zero": (
        "**a field set to a disabling value still says the mechanism "
        "exists and was turned off, and invites a later phase to turn "
        "it back on without re-deriving why it was off.** A field that "
        "is not in the spec is refused by the schema. The record's own "
        "precedent is ``phase17``'s ``branch_trainability`` vocabulary, "
        "where the unexercised regime is refused BY THE CHOICES CHECK "
        "rather than discouraged in prose"
    ),

    "what_makes_the_departure_SAFE_and_what_does_not": (
        "**``phase26`` measured that the selection never moves: 400 of "
        "400 folds selected epoch 1 across a sixteen-cell grid** "
        "(``phase26.THE_SELECTED_EPOCHS_OBSERVED``), and "
        "``phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION`` recorded the "
        "mechanism before that, 'a 769-parameter head over frozen "
        "embeddings converges immediately, inner-val never improves, "
        "patience terminates at epoch 1'. **So on the cohort arm the "
        "machinery being removed here was doing nothing.** That is a "
        "precedent for removing it and NOT a licence: both measurements "
        "are at 152 training rows against the continuous panel mean, "
        "and this phase is at 25 against an integer Score. **The "
        "departure is justified by the argument about five images, and "
        "supported rather than established by phase26**"
    ),
    "what_the_departure_COSTS": (
        "**stated rather than minimised. The phase cannot stop a fit "
        "that is going wrong**, so if the budget is too long the head "
        "at the end is overfit and the primary readout says so. The "
        "trajectory is emitted for exactly this reason "
        "(THE_TWO_READOUT_EPOCHS_DECLARED), and the phase does not "
        "re-run to a better budget"
    ),
    "and_it_is_declared_before_any_number": (
        "the ruling is written here, at the lock, before a config "
        "exists and before anything runs. **A recipe departure decided "
        "after a disappointing number is a different object entirely**"
    ),
}


#: **[DECLARED 2026-09-06, AT THE LOCK] TWO READOUT EPOCHS, BOTH FIXED
#: IN ADVANCE. Neither is chosen by a result.**
#:
#: With no checkpoint selection the budget IS the fit, so the phase
#: declares where it reads rather than discovering it.
THE_TWO_READOUT_EPOCHS_DECLARED = {
    "declared": "2026-09-06, at the lock, before any config exists",

    "the_two": (
        "**epoch 1 and epoch 40, and no other epoch may be quoted as a "
        "result.** Both are declared here with their reasons. Reporting "
        "two pre-registered points is not selection, because neither is "
        "picked after the numbers and both are reported whatever they "
        "say"
    ),
    "why_epoch_40": (
        "**it is the budget, mirrored.** The probe declares "
        "``max_epochs: 40``, and ``phase16.SETTINGS_PROVENANCE``'s rule "
        "is to mirror the closest probe-training precedent. **What "
        "changes is its meaning**: for the probe 40 is a ceiling that "
        "patience 5 never reaches, so the probe stops near epoch 6. "
        "Here there is no patience, so the loop runs all forty. **That "
        "is a real difference between this fit and the probe's and it "
        "is stated rather than hidden inside a shared number**"
    ),
    "why_epoch_1": (
        "**because the record's position is that this head class has "
        "already converged there.** It is the epoch every one of "
        "``phase26``'s 400 folds selected, and the epoch "
        "``phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION`` names. Declaring it "
        "in advance means the phase can report what the record expects "
        "AND what the budget delivers, without choosing between them "
        "afterwards"
    ),
    "which_of_the_two_is_PRIMARY": (
        "**epoch 40, the budget's own end.** Declared here so it cannot "
        "be decided by which one reads better. Epoch 1 is reported "
        "beside it as the second declared point, not as an alternative "
        "headline"
    ),

    "the_full_trajectory_is_emitted_and_is_DESCRIPTIVE_ONLY": (
        "**every epoch's cohort readouts are written, and reading the "
        "best of them is PROHIBITED.** "
        "``phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION`` governs "
        "and is carried by reference: the cohort predictions are the "
        "evaluation data, and choosing an epoch by how well it scored "
        "on them and then reporting that score is selection on the "
        "evaluation set. **The trajectory exists so the budget choice "
        "is checkable, not so a better epoch can be found**"
    ),
    "and_what_the_trajectory_would_license_if_it_turns_over": (
        "**registered in advance.** If the cohort PCC rises and then "
        "falls, the budget was too long for 25 rows, and the finding is "
        "that a fixed budget mirrored from a 152-row recipe does not "
        "transfer. **A later phase may then declare a shorter budget "
        "WITH that reason. This phase does not re-run to it**, because "
        "a second budget chosen after seeing the first is the sweep "
        "reporting its best"
    ),
}


#: **[RULED 2026-09-06, AT THE LOCK] EVERY SETTING THE DRAFT LEFT OPEN.
#: Each value carries the config it was mirrored from, or its reasoning
#: if it has no precedent.**
THE_SETTINGS_RULED = {
    "ruled": "2026-09-06, at the lock",
    "the_rule_followed": (
        "``phase16.SETTINGS_PROVENANCE``: 'mirror the closest existing "
        "probe-training precedent; report which config each value came "
        "from'. The closest precedent is "
        "``configs/p7_d1_vit_b16_imagenet_g1.yaml``, the probe itself, "
        "because this phase trains the same head class on the same "
        "frozen space. **Every value below is mirrored from it except "
        "the three that are removed and the one that has no precedent**"
    ),

    "seeds": (
        "**1337, 2024, 7, 99, 12345** -- the probe's ``seeds``, and the "
        "standing five in every arm in this record. Mirrored"
    ),
    "learning_rate": "**0.001** -- the probe's ``learning_rate``. Mirrored",
    "weight_decay": (
        "**0.01** -- the probe's ``weight_decay``, which is also the "
        "project default. ``phase26`` measured this factor inert across "
        "a hundredfold range including zero on the cohort arm, **which "
        "is a precedent for keeping the value and not a licence to skip "
        "declaring it**, since that null is at 152 rows against a "
        "different target"
    ),
    "batch_size": (
        "**NOT DECLARED, and the reason is that it does nothing "
        "here.** ``batch_size`` is consumed by ``TorchBackbone``, the "
        "image-path backbone. ``EmbeddingHeadBackbone``, which is "
        "what ``trainable: head`` actually trains, never reads it and "
        "steps full-batch. **The first build declared it anyway, "
        "which would have shipped a setting with no effect** "
        "(THE_INIT_IS_THE_PROBES)"
    ),
    "max_steps": (
        "**50** -- ``EmbeddingHeadBackbone.max_steps``'s default, "
        "which the probe's config does not override, so 50 is what "
        "the probe runs. **This is the setting that actually governs "
        "the fit**: it is the number of full-batch optimiser steps "
        "inside one epoch, so the budget of 40 epochs is two thousand "
        "steps. Mirrored, and declared here because the first build "
        "declared the wrong one"
    ),
    "max_epochs": (
        "**40** -- the probe's ``max_epochs``, mirrored, and its "
        "meaning changed by the ruling above "
        "(THE_TWO_READOUT_EPOCHS_DECLARED)"
    ),
    "readout_epochs": (
        "**(1, 40)** -- NO PRECEDENT. Declared with its reasoning at "
        "THE_TWO_READOUT_EPOCHS_DECLARED, because no phase in this "
        "record has ever run a head without checkpoint selection and "
        "none has had to say where it reads"
    ),

    "inner_val_frac": "**REMOVED** -- THE_INNER_VALIDATION_RULED",
    "monitor": "**REMOVED** -- THE_INNER_VALIDATION_RULED",
    "patience": "**REMOVED** -- THE_INNER_VALIDATION_RULED",

    "the_training_target": (
        "**the anchor Score, integer 1 to 5, read from the anchor "
        "artifact's own ``metadata.json``.** There is no ``label:`` "
        "choice, because the artifact carries exactly one label column"
    ),
    "the_evaluation_target": (
        "**``label: mean``, the cohort panel mean**, read from the "
        "manifest exactly as every cohort arm reads it. The crossing "
        "this creates is THE_LABEL_INCOMPARABILITY_ADDRESSED"
    ),
    "the_readouts": (
        "**pcc and spearman**, which cross cleanly; **rmse, mae, iem "
        "and acc3**, which carry the 0.2056 offset and are reported "
        "with it attached; and **prediction_sd per seed**, which is "
        "REQUIRED rather than optional because it is what separates a "
        "collapse from a null (THE_DEGENERATE_SIGNATURE_REGISTERED)"
    ),
    "never_tuned": (
        "every value above is a scientific setting under the standing "
        "clause: declared before the first run, NEVER tuned across "
        "runs, and movement is a dated amendment with a reason"
    ),
}


#: **[DESIGNED 2026-09-06, AT THE LOCK] THE DISJOINTNESS GUARD. The
#: phase's entire claim is that the training set and the evaluation set
#: share no patient, so the separation is STRUCTURAL and TESTED rather
#: than documentary.**
#:
#: The one precedent, ``run.task_tstr_regression``, rests its separation
#: on a comment marker and statement ordering. That is not enough here.
THE_DISJOINTNESS_GUARD_DESIGNED = {
    "designed": "2026-09-06, at the lock",

    "why_a_comment_is_not_enough": (
        "**``task_tstr_regression`` separates its two sets with the "
        "sentence 'the ordering below is structural: everything before "
        "the evaluation marker touches only the synthetic artifact', "
        "and a comment line reading TRAINING ON SYNTHETIC ONLY.** It "
        "was true when it was written and nothing enforces it. **A "
        "phase whose whole claim is disjointness cannot rest it on a "
        "reader noticing a comment**"
    ),
    "and_an_identifier_check_alone_would_be_VACUOUS": (
        "**stated because it is the obvious guard and it does "
        "nothing.** The anchors are keyed by image STEM, a string, and "
        "the cohort by patient id, an integer. They cannot collide, so "
        "a guard that only compares identifiers passes by construction "
        "and proves nothing. **A check that cannot fail is not a "
        "check**, which is ``record_audit.THE_GREP_THAT_DID_NOT_RUN``'s "
        "lesson in a new place"
    ),

    "the_guard_is_THREE_PARTS_and_each_can_fail": (
        "**(1) NAMESPACE.** The training features are loaded only "
        "through ``phase16.load_anchor_set``, which refuses anything "
        "whose ``metadata.json`` does not say namespace "
        "``anchor_deall``, and re-checks 25 rows and the 3/7/6/6/3 "
        "spread. **A config pointing the training input at the cohort "
        "embeddings raises rather than training on them.** "
        "**(2) FEATURES.** ``disjoint_or_raise`` compares every "
        "training row against every evaluation row and refuses on any "
        "bitwise-identical pair, which is what a mis-declared artifact "
        "would actually look like. **(3) SIGNATURE.** ``fit_head`` "
        "takes features, targets, and settings, and has NO parameter "
        "through which cohort truth could reach it"
    ),
    "and_the_call_site_is_AST_tested": (
        "**the fourth part, and it is the one that stops the guard "
        "being bypassed rather than failed.** A test parses "
        "``run.task_p27_anchor_train`` and asserts that the single "
        "``fit_head`` call site receives only anchor-derived names, and "
        "that ``disjoint_or_raise`` is called before it. **This is "
        "``phase16``'s criterion 1 shape**, which proved fold honesty "
        "the same way: 'the task's single train_w call site receives "
        "only its output (AST-tested)'"
    ),
    "what_the_guard_does_NOT_prove": (
        "**it does not prove the two image sets contain no shared "
        "patient in the world.** The 25 are a separate study's "
        "benchmark and the record has never checked whether any person "
        "appears in both, and could not from what it holds. **What is "
        "guarded is that no ROW is shared between the fit and the "
        "evaluation.** The stronger claim is not made"
    ),
    "and_that_gap_is_a_LIMITATION_not_an_oversight": (
        "recorded here so a reader does not take the guard for more "
        "than it is. Settling it would need identity information about "
        "both sets that this project does not have and should not seek"
    ),
}


#: **[LOCKED 2026-09-06] EXIT CRITERIA. Ten, locked, with the readings
#: written beside them at the lock and before any config exists.**
#:
#: The draft is at ``EXIT_CRITERIA_DRAFT`` and is preserved. This is what
#: it became.
EXIT_CRITERIA = {
    "locked": "2026-09-06, before any config exists and before any run",

    "1_the_motivation_is_rewritten_not_inherited": (
        "**MET AT THE LOCK.** The amendment's case FOR is withdrawn in "
        "the record (THE_MOTIVATION_REWRITTEN), the honest motivation "
        "stands in its place, and unanimity keeps its [REPORTED] tag"
    ),
    "2_the_crossing_is_stated_before_any_number": (
        "**MET AT THE LOCK.** Which readouts cross and which do not, "
        "with the 0.2056 offset derived and shown "
        "(THE_LABEL_INCOMPARABILITY_ADDRESSED)"
    ),
    "3_the_readings_are_registered_both_ways": (
        "**MET AT THE LOCK.** READINGS and THE_ASYMMETRY_REGISTERED, "
        "with the positive registered UNINTERPRETABLE in advance"
    ),
    "4_the_degenerate_signature_is_named_and_its_detector_emitted": (
        "the constant predictor at 2.96 with near-zero PCC and acc3 "
        "near 0.502 is registered, AND per-seed prediction_sd reaches "
        "``metrics.json`` so a collapse is distinguishable from a null "
        "rather than inferred from the other two"
    ),
    "5_the_disjointness_is_GUARDED_structurally": (
        "namespace, features, signature and an AST test on the call "
        "site, each able to fail "
        "(THE_DISJOINTNESS_GUARD_DESIGNED). **A comment is not a guard**"
    ),
    "6_the_claim_is_bounded_before_the_numbers": (
        "**MET AT THE LOCK.** The unresolvable band, 0.1386 as a "
        "demonstrated instance with its prohibition cited by name, and "
        "the n=237 threshold of 0.1281 rather than the n=25 threshold "
        "of 0.4179 (THE_CLAIM_BOUNDED)"
    ),
    "7_the_arm_measures_its_own_seed_band": (
        "the five-seed sd is computed and reported for this arm rather "
        "than borrowed from ``train.phase3.MEASURED_SEED_BAND``, whose "
        "own warning forbids reuse across a different training "
        "procedure"
    ),
    "8_nothing_is_pooled": (
        "no figure from this phase enters a ledger contrast family with "
        "a cohort-trained arm, and no value metric is placed beside a "
        "cohort-trained one without the offset stated"
    ),
    "9_both_declared_readout_epochs_are_reported_whatever_they_say": (
        "epoch 1 and epoch 40 both reach ``metrics.json``, the primary "
        "is epoch 40 as declared, and no other epoch is quoted as a "
        "result (THE_TWO_READOUT_EPOCHS_DECLARED)"
    ),
    "10_the_suite_is_green": "the standing criterion on every phase",

    "what_is_NOT_an_exit_criterion": (
        "**a number.** No criterion requires the phase to reach any "
        "value, and none may be added that does. A phase that must hit "
        "a figure to close is a phase that will hit it"
    ),
    "and_six_of_the_ten_are_MET_AT_THE_LOCK": (
        "**deliberately, and it is not a shortcut.** Criteria 1, 2, 3 "
        "and 6 are about what the record says before the run, so they "
        "are discharged by the lock itself. The four that remain, 4, 5, "
        "7 and 9, are about what the RUN emits, and none of them can be "
        "met by writing prose"
    ),
}


#: **[WRITTEN AT THE LOCK 2026-09-06] THE READINGS, tied to the locked
#: criteria and to the two declared readout epochs.**
#:
#: ``READINGS`` was written at the restate and is preserved. These are
#: sharper because the design now exists, and they are still written
#: before any config.
READINGS_AT_THE_LOCK = {
    "written": "2026-09-06, at the lock, before any config exists",

    "the_bound_that_governs_every_reading": (
        "**a PCC within about 0.10 of the probe's 0.2520 is a number "
        "this cohort cannot separate from it** "
        "(``ladder.COHORT_CANNOT_RESOLVE``), so the readings below turn "
        "on where a result sits relative to that band and not on which "
        "side of 0.2520 it falls"
    ),

    "if_the_primary_lands_INSIDE_the_unresolvable_band": (
        "**the expected outcome, and the reading is the null.** The "
        "anchor label carries no transferable grade signal this cohort "
        "can distinguish from the probe's, the crossing is closed, and "
        "it becomes the supervised half of "
        "``phase16.REPAIR_WITHOUT_TRANSFER``. **It does NOT say the two "
        "labels are equivalent**, because the band is a statement about "
        "what this cohort can resolve and not about what is there"
    ),
    "if_the_primary_lands_CLEARLY_BELOW": (
        "**a null with a mechanism available, and the mechanism is "
        "already banked.** The reading is that 25 rows against a "
        "769-parameter head do not support a fit, and the collapse "
        "detector says whether that is what happened. If prediction_sd "
        "is near zero the phase reports the registered degenerate "
        "signature firing (THE_DEGENERATE_SIGNATURE_REGISTERED). **If "
        "prediction_sd is healthy and the PCC is still low, that is a "
        "different and more interesting result**: the fit worked and "
        "the label does not transfer"
    ),
    "if_the_primary_lands_CLEARLY_ABOVE": (
        "**registered as UNINTERPRETABLE and reported as such.** "
        "THE_ASYMMETRY_REGISTERED governs: the record cannot attribute "
        "it to label quality, because the label's construction is "
        "unverified. It would make per-rater grades for the 25 the "
        "highest-value outstanding ask in the project. **It licenses no "
        "sentence about cleaner labels**, and it does not reopen "
        "``phase9.EXTERNAL_VALIDATION_ON_THE_25_CONCEDED``"
    ),

    "if_the_TWO_declared_epochs_disagree": (
        "**a finding about the budget, not about the label.** The "
        "reading is that a budget mirrored from a 152-row recipe does "
        "not transfer to 25 rows, which is worth recording because this "
        "record has no other measurement of it. **The primary stays "
        "epoch 40 whichever reads better**, because it was declared "
        "primary before the numbers existed"
    ),
    "if_the_two_declared_epochs_AGREE": (
        "**the departure at THE_INNER_VALIDATION_RULED cost nothing**, "
        "and the record gains a small measurement that removing "
        "checkpoint selection on this head class changes no result. "
        "That would extend ``phase26``'s 400-of-400 finding to a "
        "training set six times smaller and a different target"
    ),

    "and_the_reading_that_is_NOT_available_whatever_happens": (
        "**that one label is cleaner than the other.** No outcome "
        "licenses it, because the anchor label's reliability is not "
        "computable from what this record holds and no result of this "
        "phase makes it computable"
    ),
}


#: **[CORRECTED 2026-09-06, BEFORE LAUNCH] THE HEAD IS THE PROBE'S, AND
#: THE FIRST BUILD OF THIS PHASE GOT IT WRONG. The record's own R2 error
#: class, committed by the record.**
THE_INIT_IS_THE_PROBES = {
    "corrected": "2026-09-06, before anything ran",

    "what_the_first_build_did": (
        "**it used ``torch.nn.Linear``'s default initialisation, which "
        "draws weights and bias uniformly from plus or minus one over "
        "the square root of the width**, while THE_SETTINGS_RULED said "
        "the recipe was mirrored from the probe. Same 769 parameters, "
        "same learning rate, same weight decay, same optimiser class. "
        "**A different fit under the same setting names, which is the R2 "
        "error exactly**"
    ),
    "what_the_probe_actually_does": (
        "``train.torch_backbone.EmbeddingHeadBackbone.reset`` sets the "
        "**weights to ZERO and the bias to the TRAINING-FOLD MEAN**, and "
        "its own comment says why: 'so an untrained head predicts the "
        "mean exactly -- which is what gate 3 checks'. **That is why the "
        "probe converges at epoch 1**: it starts at the constant "
        "predictor and only has to add a signal component"
    ),
    "and_it_broke_the_registered_signature": (
        "**the worse half.** THE_DEGENERATE_SIGNATURE_REGISTERED watches "
        "for a constant predictor at the anchor grade mean 2.96. Under "
        "the probe's init that constant is where the head STARTS, so a "
        "fit that learns nothing sits exactly on it and the signature "
        "fires cleanly. Under a random init the untrained head predicts "
        "near ZERO, so a failed fit would have landed on a different "
        "constant and the registered signature would have missed it. "
        "**The record was right and the implementation was wrong**"
    ),
    "how_it_was_found": (
        "**by simulating the declared budget rather than trusting it.** "
        "Forty AdamW steps at learning rate 0.001 move each parameter by "
        "at most about 0.04, and a bias starting near zero has to travel "
        "2.96 to reach the training mean. **The arithmetic said the fit "
        "could not happen**, which is what sent the reading back to the "
        "probe's own initialiser"
    ),
    "the_fix_is_REUSE_rather_than_a_better_mirror": (
        "``fit_head`` now drives ``EmbeddingHeadBackbone`` itself. **A "
        "mirrored loop is a second implementation of something this "
        "repository already has**, and the record's standing position on "
        "that is ``phase7c``'s 'building a parallel loop would put a "
        "second implementation of the' training path in the tree. "
        "Reusing it makes 'mirrors the probe' true rather than nominal"
    ),
    "and_it_changed_two_settings": (
        "**``max_steps`` joins and ``batch_size`` goes.** "
        "``EmbeddingHeadBackbone.train_epoch`` runs ``max_steps`` "
        "full-batch steps per epoch, default 50, and the probe's config "
        "does not override it. **``batch_size`` never reaches this class "
        "at all**: it is consumed by ``TorchBackbone``, the image-path "
        "backbone, so declaring it here would have been a setting that "
        "does nothing. See THE_SETTINGS_RULED"
    ),
    "what_this_did_NOT_touch": (
        "**no record, no reading and no criterion changes.** The "
        "motivation, the crossing, the readings, the bound and the "
        "prohibitions were all written about the phase's design and not "
        "about its initialiser. What changed is the code and two "
        "settings. **The correction is recorded because the error is "
        "instructive, not because it moved a conclusion**"
    ),
}


#: **[REASONED 2026-09-06, BEFORE THE RUN] What the budget is expected to
#: do, registered so the outcome is attributable.**
#:
#: **[REASONED], not [MEASURED]**: this is arithmetic on the optimiser's
#: own update rule, run outside the repository's training path. The run
#: measures it.
THE_BUDGET_EXPECTATION_REGISTERED = {
    "reasoned": "2026-09-06, before the run, and tagged as reasoned",

    "the_expectation": (
        "**769 parameters against 25 points is an underdetermined system, "
        "so a converging fit INTERPOLATES the training set exactly.** "
        "With the probe's init and 50 full-batch steps per epoch, forty "
        "epochs is two thousand steps, and the training loss is expected "
        "to reach numerical zero well before the budget ends"
    ),
    "so_the_two_readout_epochs_are_a_real_contrast": (
        "**epoch 1 is a lightly fitted head and epoch 40 is a fully "
        "interpolating one**, which is a sharper pair than the restate "
        "knew when it declared them. The declaration stands as written "
        "and its reasons are unchanged. **This is a better reason for "
        "the same choice, found afterwards, and it is recorded as such "
        "rather than backdated**"
    ),
    "what_it_does_NOT_predict": (
        "**nothing about the cohort.** Interpolating 25 training points "
        "says the fit ran, not that it transfers. The 237 are pure test "
        "throughout and no readout on them is anticipated here"
    ),
    "and_if_the_training_loss_does_NOT_fall": (
        "**the registered degenerate signature is the first suspect and "
        "the head's own init makes it legible**: predictions sitting at "
        "2.96 with near-zero spread is an untrained head, not a trained "
        "one that failed. THE_DEGENERATE_SIGNATURE_REGISTERED"
    ),
}


# --------------------------------------------------------------------------
# THE RESULT, 2026-09-06
# --------------------------------------------------------------------------

#: **[OBSERVED 2026-09-06] CRITERION 5 MET. The disjointness guard ran
#: and passed on the sizes the lock fixed.**
THE_GUARD_PASSED = {
    "observed": "2026-09-06, run p27_anchor_train at sha 0fe38121",
    "what_it_checked": (
        "**25 training rows against 237 evaluation rows at 768 "
        "dimensions, no shared identifier and no bitwise-identical "
        "feature row.** Read from the run's ``disjointness_guard`` "
        "block, whose ``guard`` field names ``phase27."
        "disjoint_or_raise`` so the record says which code produced it"
    ),
    "and_the_sizes_are_the_locked_ones": (
        "``n_train`` 25 is ``N_ANCHORS`` and the writer's own "
        "invariant, ``n_eval`` 237 is the cohort, and ``feature_dim`` "
        "768 is the frozen space both artifacts live in. **The guard "
        "confirms the two sets were the sizes the design assumed**, "
        "which a run that silently loaded the wrong artifact would not"
    ),
    "what_it_does_not_prove_is_unchanged": (
        "it says no ROW is shared between the fit and the evaluation. "
        "**It does not say the two image sets contain no shared "
        "person**, which the lock recorded as a limitation rather than "
        "an oversight (THE_DISJOINTNESS_GUARD_DESIGNED)"
    ),
}


#: **[OBSERVED 2026-09-06] THE RESULT. Both declared readout epochs,
#: read from the run.**
#:
#: Full precision as the run wrote it, because a rounded figure quoted
#: later cannot be checked against its source.
THE_RESULT_OBSERVED = {
    "observed": "2026-09-06, run p27_anchor_train at sha 0fe38121",
    "seeds": "five, 1337 / 2024 / 7 / 99 / 12345, as locked",

    "primary_epoch_40": {
        "pcc": 0.18270409137534543,
        "spearman": 0.1255909734414814,
        "prediction_mean": 2.2107740466789556,
        "prediction_sd": 0.8727277862188358,
        "rmse": 1.1312049305108245,
        "mae": 0.8980673322199728,
        "acc3": 0.4388185654008439,
    },
    "second_declared_epoch_1": {
        "pcc": 0.1849907964436629,
        "spearman": 0.14007363810648907,
        "prediction_mean": 2.1454977826417534,
        "prediction_sd": 0.8509391972534811,
        "rmse": 1.1489039005979955,
        "mae": 0.9146817367285309,
        "acc3": 0.4388185654008439,
    },
    "baselines_from_the_run": {
        "chance": 0.3333333333333333,
        "majority": 0.5021097046413502,
    },

    "the_primary_is_epoch_40_because_it_was_declared_so": (
        "**and epoch 1 reads higher on the correlation, which is "
        "exactly why the declaration was made in advance.** The lock "
        "fixed the primary before any number existed "
        "(THE_TWO_READOUT_EPOCHS_DECLARED), so 0.18270409137534543 is "
        "the phase's figure and 0.1849907964436629 is reported beside "
        "it rather than instead of it"
    ),
    "and_the_whole_trajectory_is_in_the_run": (
        "forty epochs by five seeds, written as "
        "``trajectory_DESCRIPTIVE_ONLY``. **Reading the best epoch off "
        "it is prohibited** and the prohibition is "
        "``phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION``, carried "
        "at THE_TWO_READOUT_EPOCHS_DECLARED"
    ),
}


#: **[MEASURED 2026-09-06] CRITERION 7. THE SEED SD IS ZERO, AND IT IS
#: EXPECTED RATHER THAN A DEFECT.**
#:
#: A zero seed sd is also what broken seed wiring looks like, so the
#: record says what distinguishes the two here rather than asserting the
#: benign reading.
THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION = {
    "measured": "2026-09-06, from the run and from the source",

    "what_the_run_shows": (
        "**every readout is identical across all five seeds at both "
        "declared epochs**: ``min`` equals ``max`` on pcc, spearman, "
        "rmse, mae, acc3, prediction_sd and prediction_mean, and the "
        "five per-seed blocks carry ONE distinct value per field. The "
        "banded sd is exactly ``0.0`` for pcc, spearman, rmse and "
        "prediction_sd at both epochs"
    ),
    "and_two_of_the_sds_are_float_noise_rather_than_zero": (
        "**stated because 'exactly zero' would be wrong for two of "
        "them.** ``acc3`` carries sd **6.206335383118183e-17** at both "
        "epochs and ``mae`` carries **1.2412670766236366e-16** at "
        "epoch 1, on values whose min equals their max. That is the "
        "sample-sd arithmetic on identical inputs, not variation. **The "
        "identity is in min equals max, and the sd field is a "
        "derived number that inherits float representation**"
    ),

    "and_three_banded_MEANS_differ_from_their_min_by_one_ulp": (
        "**the same phenomenon on a different field, recorded so a "
        "reader comparing two representations of one figure is not "
        "confused.** ``acc3`` at both epochs and ``mae`` at epoch 1 "
        "have a banded ``mean`` that differs from their identical "
        "``min`` and ``max`` in the last digit, because summing five "
        "identical floats and dividing reintroduces a rounding that "
        "the values themselves do not carry. **THE_RESULT_OBSERVED "
        "banks the min/max value, which is the measurement**, and "
        "not the banded mean, which is a statistic derived from five "
        "copies of it"
    ),
    "WHY_it_is_zero_traced_in_the_source": (
        "**the fit has no stochastic element left.** "
        "``EmbeddingHeadBackbone.reset`` calls ``torch.manual_seed`` "
        "and then sets the weights to ``torch.zeros`` and the bias to "
        "``np.mean(train_labels)``. Neither is a draw. ``_ensure``, "
        "``train_epoch`` and ``predict`` contain no sampling, no "
        "shuffling and no permutation, and the loop is full-batch with "
        "no inner-validation split to draw. **So the seed reaches the "
        "code and has nothing to act on**"
    ),
    "and_it_is_a_CONSEQUENCE_of_the_pre-launch_correction": (
        "**before THE_INIT_IS_THE_PROBES the seed WOULD have "
        "mattered.** ``torch.nn.Linear``'s default initialisation is a "
        "random draw, so the first build of this phase would have "
        "produced five different fits and a non-zero seed sd. "
        "Correcting the initialiser to the probe's own removed the "
        "only consumer of randomness in the path. **The zero is "
        "downstream of a correction made before anything ran**"
    ),

    "a_zero_seed_sd_is_ALSO_what_broken_seed_wiring_looks_like": (
        "**and the record says so rather than assuming the benign "
        "reading.** A task that ignored its ``seeds`` list, or looped "
        "five times over one seed, or overwrote each fit with the "
        "last, would produce this same table. The two are "
        "indistinguishable from the numbers alone"
    ),
    "what_distinguishes_them_HERE_and_it_is_three_things": (
        "**(1) the determinism is BY CONSTRUCTION and the mechanism is "
        "traced in the source above**, not inferred from the output. "
        "**(2) Both declared readout epochs show it**, so it is not an "
        "artefact of reading at one point. **(3) All five seed "
        "trajectories are bit identical across ALL FORTY epochs**, not "
        "only at the two readouts, which is a stronger statement than "
        "the criterion asked for and is checkable in the run's own "
        "``trajectory_DESCRIPTIVE_ONLY``. **A wiring fault would have "
        "to reproduce forty epochs of agreement to look like this**"
    ),
    "and_the_seed_is_genuinely_passed": (
        "``fit_head`` takes ``seed`` and hands it to "
        "``EmbeddingHeadBackbone(seed=...)``, and the task iterates the "
        "config's five. **This is not a seed that was never wired. It "
        "is a seed whose only consumer was removed**, which is a "
        "different fact and a better one"
    ),

    "what_it_COSTS": (
        "**the arm has no seed band of its own to speak of, and "
        "criterion 7 is met by measuring that rather than by "
        "borrowing.** ``train.phase3.MEASURED_SEED_BAND`` names the "
        "cohort arms' quantity as head initialisation and inner-val "
        "split. **This arm has neither**, so its band is not a smaller "
        "version of the cohort arms' band, it is the absence of the two "
        "things that band measures. Reusing 0.0137 here would have been "
        "the R2 error that band's own warning names"
    ),
}


#: **[MEASURED 2026-09-06] CRITERION 4. THE REGISTERED DEGENERATE
#: SIGNATURE DID NOT FIRE, and that is the phase's first substantive
#: finding.**
THE_DEGENERATE_SIGNATURE_DID_NOT_FIRE = {
    "measured": "2026-09-06, from the run",

    "what_was_registered": (
        "a constant predictor at the anchor grade mean, **2.96 from "
        "74/25**, presenting as near-zero correlation with three-class "
        "accuracy near the 0.502 majority floor. Registered in advance "
        "at THE_DEGENERATE_SIGNATURE_REGISTERED and sharpened at the "
        "lock to an EQUALITY, because the corrected initialiser starts "
        "the head on exactly that constant"
    ),
    "what_happened_instead": (
        "**the model varies.** Prediction sd at the primary is "
        "**0.8727277862188358**, against a cohort truth sd of 0.6587 "
        "banked at ``phase10``. **[DERIVED]** that is a ratio of "
        "**1.3249**, so the predictions are more spread than the thing "
        "they predict, not less. And the prediction mean is "
        "**2.2107740466789556**, which is **0.7492 away from the 2.96 "
        "the head started on**. It left its initialisation and it did "
        "not return to a constant"
    ),
    "and_the_accuracy_is_not_at_the_floor_either": (
        "three-class accuracy **0.4388185654008439** against the run's "
        "own majority **0.5021097046413502** and chance "
        "**0.3333333333333333**. **Below the majority floor rather "
        "than at it**, which is not the registered signature: a "
        "constant landing in the mid class would sit near 0.502, not "
        "0.439"
    ),
    "why_this_is_a_FINDING_and_not_merely_a_relief": (
        "**the amendment's case AGAINST this phase was that 25 rows "
        "against a 769-parameter head is an underdetermined fit and "
        "the record predicts it will fail.** It did not fail in the "
        "way the record named. The fit ran, left its start, and "
        "produced varying predictions. **A prediction registered in "
        "advance and refuted by measurement is worth more than one "
        "confirmed**, and this is the phase's first substantive result"
    ),
    "what_it_does_NOT_establish": (
        "**that the fit is good.** Not collapsing is a low bar and the "
        "record does not dress it as more. What the correlation is "
        "worth is THE_READINGS_APPLIED, and it is a separate question "
        "from whether the head degenerated"
    ),
}


#: **[MEASURED 2026-09-06] FORTY EPOCHS MOVED ALMOST NOTHING, and the
#: record already knew why.**
THE_BUDGET_MOVED_ALMOST_NOTHING = {
    "measured": "2026-09-06, from the run's trajectory",

    "the_two_declared_epochs": (
        "correlation **0.1849907964436629 at epoch 1** and "
        "**0.18270409137534543 at epoch 40**, so the whole budget moved "
        "it **[DERIVED] -0.0022867** and moved it DOWN. Prediction sd "
        "rose from **0.8509391972534811** to **0.8727277862188358**. "
        "**More spread, marginally less correlation**"
    ),
    "and_the_whole_trajectory_says_the_same": (
        "over all forty epochs the correlation's **maximum is at epoch "
        "1 (0.184991) and its minimum at epoch 21 (0.182176)**, a total "
        "range of **[DERIVED] 0.002814**. It settles by epoch 2 and "
        "then moves in the fourth decimal. **Two thousand optimiser "
        "steps bought a change smaller than a third of the probe's own "
        "seed sd of 0.0148**"
    ),
    "this_is_NOT_new_and_the_record_says_which_finding_it_extends": (
        "**``phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION`` recorded the "
        "mechanism**: 'a 769-parameter head over frozen embeddings "
        "converges immediately, inner-val never improves, patience "
        "terminates at epoch 1'. **``phase26`` measured it at scale**, "
        "400 of 400 folds selecting epoch 1 across a sixteen-cell grid. "
        "**This extends both to a training set six times smaller and to "
        "a different target**, and it does so without a selection "
        "mechanism, which neither of the earlier measurements could"
    ),
    "and_that_is_the_stronger_form": (
        "``phase26`` could only show that the SELECTION never moved, "
        "because a checkpoint rule was choosing. **Here there is no "
        "selection at all, so what is measured is that the FIT does not "
        "move**, which is the claim the earlier records were reaching "
        "for through a mechanism that hid it"
    ),
    "what_it_says_about_the_departure": (
        "**the departure at THE_INNER_VALIDATION_RULED cost nothing.** "
        "The readings written at the lock registered this case: 'if the "
        "two declared epochs AGREE, the departure cost nothing'. They "
        "agree to three decimal places on the correlation. **Removing "
        "checkpoint selection changed no result here**, which is the "
        "small measurement that reading anticipated"
    ),
    "and_one_oddity_recorded_rather_than_smoothed": (
        "**epoch 40 is not the end of a monotone slide.** The "
        "correlation at epoch 39 is 0.182450 and at epoch 40 is "
        "0.182704, and the prediction mean moves from 2.238201 to "
        "2.210774 in that last step. Something moved at the final "
        "epoch. **It is 0.0003 on the correlation and it changes "
        "nothing**, and it is recorded because a reader who plots the "
        "trajectory will see it"
    ),
}


#: **[CORRECTED 2026-09-06, BY MEASUREMENT] THE DECLARED OFFSET POINTS
#: THE WRONG WAY AND IS THE WRONG SIZE. The lock's derivation was
#: correct arithmetic about the wrong quantity.**
THE_OFFSET_IS_CORRECTED = {
    "corrected": "2026-09-06, from the run",

    "what_was_declared": (
        "**[DERIVED] +0.2056**, the anchor grade mean 2.96 minus the "
        "cohort panel mean 2.7544, reasoned at "
        "THE_LABEL_INCOMPARABILITY_ADDRESSED as the offset a model "
        "predicting in anchor units would carry, and shipped in the "
        "config as ``target_offset``"
    ),
    "what_was_MEASURED": (
        "**the prediction mean is 2.2107740466789556 against a cohort "
        "truth mean of 2.7544, so the realised offset is [DERIVED] "
        "-0.5436.** At the second declared epoch it is -0.6089. "
        "**Opposite in sign to the declaration and about 2.6 times its "
        "magnitude**"
    ),
    "why_the_derivation_was_wrong": (
        "**it was correct arithmetic about the wrong quantity.** The "
        "difference between the two label means is a real number and it "
        "is +0.2056. But that is the offset a model would carry if the "
        "cohort features sat where the anchor features sit. **They do "
        "not.** The head interpolates 25 anchor rows and is then "
        "applied to 237 rows elsewhere in the same space, so the "
        "realised offset is a property of WHERE THE COHORT SITS "
        "relative to the anchors, and the label means do not predict it"
    ),
    "and_the_record_carried_the_right_candidate_already": (
        "``THE_TWO_EXTRACTIONS_COMPARED`` recorded that the 25 "
        "composites sit at aspect ratios 0.86 to 1.17, 'outside the "
        "cohort's portrait crop family', and carried that framing "
        "difference as **'a candidate explanation for either "
        "outcome'**. It was the right thing to carry. **The offset is "
        "the first measurement in this record that the two image sets "
        "occupy different regions of the frozen space**"
    ),
    "what_this_CHANGES_and_what_it_does_not": (
        "**it changes the value metrics' correction and nothing "
        "else.** The crossing analysis's central claim, that pcc and "
        "spearman cross the two targets cleanly while the value metrics "
        "do not, is UNCHANGED and is if anything strengthened: the "
        "value metrics carry an offset two and a half times larger than "
        "declared. **The declared 0.2056 stays in the config and in the "
        "run's own ``the_crossing`` block as what was declared**, and "
        "this entry is what it turned out to be"
    ),
    "and_it_is_the_ninth_catch_shape_arriving_from_the_other_side": (
        "``phase18.P17_IEM_MEASURED`` counted the anchor-mean versus "
        "panel-mean conflation as the NINTH different-quantities catch, "
        "and this phase cited it in advance. **The lock avoided that "
        "trap and fell into an adjacent one**: it kept the two label "
        "means apart correctly and then treated their difference as "
        "predicting the model's output. Two means being different does "
        "not tell you where a fitted model lands"
    ),
}


#: **[OBSERVED 2026-09-06] THE VALUE METRICS, REPORTED SEPARATELY AND
#: UNDER THE CROSSING. Not comparable to any cohort-trained arm.**
THE_VALUE_METRICS_UNDER_THE_CROSSING = {
    "observed": "2026-09-06, from the run, at the primary epoch",

    "the_figures": (
        "**prediction mean 2.2107740466789556** against a cohort truth "
        "mean of **2.7544**; **mean absolute error "
        "0.8980673322199728**; **root mean squared error "
        "1.1312049305108245**; **three-class accuracy "
        "0.4388185654008439** against the run's own majority "
        "**0.5021097046413502** and chance **0.3333333333333333**"
    ),
    "why_they_are_NOT_comparable_to_a_cohort_trained_arm": (
        "**because the offset is declared, real and measured at "
        "-0.5436** (THE_OFFSET_IS_CORRECTED). A model fit to an integer "
        "1 to 5 Score and scored against a 0.2-grid panel mean is "
        "penalised by wherever its output sits, and here it sits half a "
        "grade low. **An MAE of 0.898 beside a cohort-trained arm's MAE "
        "would be measuring the label crossing and attributing it to "
        "the model**, which is the error "
        "``phase20.CROSS_TARGET_ERROR_PROVENANCE`` records"
    ),
    "and_three_class_accuracy_is_affected_too": (
        "**stated because it is not a value metric and is still hit.** "
        "The 2.5 and 3.5 bin edges are fixed in COHORT units, so an "
        "anchor-unit prediction meets a cohort-unit boundary. A "
        "prediction mean half a grade low pushes mass across the lower "
        "edge. **0.4388 below a 0.5021 majority floor is what a shifted "
        "predictor scores, and it is not a statement about ranking**"
    ),
    "only_the_two_correlations_cross_cleanly": (
        "**pcc 0.18270409137534543 and spearman 0.1255909734414814.** "
        "``phase21.SCALE_INVARIANCE_PROHIBITION`` verified that "
        "rescaling a prediction vector about the mean leaves pcc "
        "exactly unchanged to 0.00e+00, and a rank correlation is "
        "invariant to any monotone transform. **A shift of -0.5436 "
        "moves neither.** These are the only two figures from this "
        "phase that may be placed beside a cohort-trained arm's"
    ),
    "and_nothing_is_pooled": (
        "criterion 8. No figure here enters a ledger contrast family "
        "with a cohort-trained arm, and the registered rule it inherits "
        "is ``phase9.DEALL_REFERENCE_REGISTERED``'s 'external reference "
        "only, never pooled with cohort results'"
    ),
}


#: **[APPLIED 2026-09-06] THE READINGS WRITTEN AT THE LOCK, APPLIED. And
#: one case they did not anticipate.**
THE_READINGS_APPLIED = {
    "applied": "2026-09-06, against READINGS_AT_THE_LOCK",

    "where_the_result_sits": (
        "**[DERIVED] the primary is 0.069296 below the probe's "
        "0.2520.** ``ladder.COHORT_CANNOT_RESOLVE`` measured that this "
        "cohort cannot resolve differences of 0.04 to 0.10, confirmed "
        "at scale with 30 tested and 1 surviving. **0.0693 is inside "
        "that band**, and it is far below the 0.1386 that is the "
        "smallest difference this cohort has ever demonstrably resolved"
    ),
    "so_the_registered_reading_is_the_INSIDE_THE_BAND_one": (
        "quoted from the lock: 'the expected outcome, and the reading "
        "is the null ... **It does NOT say the two labels are "
        "equivalent**, because the band is a statement about what this "
        "cohort can resolve and not about what is there'"
    ),
    "what_it_LICENSES": (
        "**a descriptive ordering and nothing more.** The "
        "anchor-trained head scored lower than the probe. That is a "
        "fact about the measurements. **It does not license 'the "
        "anchor label is worse', 'the crossing costs 0.07', or any "
        "sentence in which the difference is attributed to the label**"
    ),
    "what_it_does_NOT_license": (
        "**that the two are equivalent, and that the anchor label "
        "carries no signal.** Both are refuted below. The band says the "
        "cohort cannot separate 0.1827 from 0.2520, which is a "
        "statement about the instrument"
    ),

    "THE_CASE_THE_READINGS_DID_NOT_ANTICIPATE": (
        "**the result is inside the unresolvable band relative to the "
        "probe AND above the threshold relative to zero, and the lock "
        "registered three cases that do not cover that.** "
        "``relevance.significance_threshold(237)`` is 0.128127 and the "
        "primary correlation is 0.18270409137534543, so **the "
        "correlation is distinguishable from zero at n=237**. The lock "
        "wrote 'inside the band' as 'the reading is the null', which "
        "reads as though nothing was found. **Something was found. It "
        "just cannot be separated from the probe**"
    ),
    "why_that_gap_is_recorded_rather_than_papered_over": (
        "**the readings were written before any number and they were "
        "incomplete, and saying so is cheaper than stretching one of "
        "them to fit.** Their three cases were positioned against the "
        "PROBE alone and none was positioned against zero. A phase "
        "whose result is significant against one reference and "
        "unresolvable against another needs both stated, and the lock "
        "gave it one"
    ),
    "and_the_ASYMMETRY_registration_governs_the_half_that_is_positive": (
        "**this is the uninterpretable case for that half.** "
        "THE_ASYMMETRY_REGISTERED registered in advance that a positive "
        "result cannot be attributed, because the label's construction "
        "is unverified. **A correlation of 0.1827 that clears the "
        "zero threshold IS a positive result**, and the record cannot "
        "say whether it reflects label quality, a differently biased "
        "target that happens to align, or the domain shift the offset "
        "just measured. **It licenses no sentence about cleaner "
        "labels**, and it makes per-rater grades for the 25 the "
        "highest-value outstanding ask, exactly as registered"
    ),
    "so_which_case_is_this_stated_plainly": (
        "**a null against the probe and an uninterpretable positive "
        "against zero, in one result.** Neither half may be reported "
        "without the other. Reporting only the null understates what "
        "was measured. Reporting only the positive is the sentence the "
        "asymmetry registration exists to forbid"
    ),
}


#: **[RULED 2026-09-06] THE CRITERION IS NOT SATISFIED, AND THE REASON
#: IS NOT THE ONE THAT WAS SUSPECTED.**
THE_CRITERION_NOT_RESOLVED = {
    "ruled": "2026-09-06",

    "condition_2_PASSES": (
        "**[DERIVED] delta 0.069296 against arm_means_95 of 0.012973, a "
        "margin of 5.34x.** Computed by "
        "``train.phase3.combined_claimable_delta(0.0, 5, 0.0148, 5)``, "
        "this arm's seed sd against the probe's"
    ),
    "and_the_zero_seed_sd_is_NOT_what_makes_it_pass": (
        "**checked, because it was the obvious suspicion and it is "
        "wrong.** A deterministic arm shrinks ``arm_means_95`` by "
        "exactly one over the square root of two, from 0.018346 to "
        "0.012973. **Had this arm carried the probe's own sd of 0.0148, "
        "condition 2 would still pass at 3.78x.** The suspicion is "
        "recorded with its refutation because reasoning about a "
        "criterion is not the same as computing it"
    ),
    "condition_1_IS_NOT_COMPUTED": (
        "**and it is the one that decides.** PLAN section 4.3's first "
        "condition is a per-seed paired BCa over the 237 patients "
        "excluding zero. That needs the per-seed predictions, which the "
        "task wrote as ``seed_<n>__predictions.csv`` at CLUSTER-ONLY "
        "tier. **They did not reach this machine and metrics.json does "
        "not carry them**, so the condition is unevaluated"
    ),
    "so_the_contrast_is_UNRESOLVED_and_the_phase_does_not_claim_it": (
        "**a contrast with one condition passing and the other "
        "unevaluated is not a resolved contrast.** The point estimate "
        "additionally sits inside the band this cohort has never "
        "resolved a difference within. **The phase reports a "
        "descriptive ordering and stops**"
    ),
    "and_this_is_the_phase_16_shape_with_one_difference": (
        "``p16-anchor-loop-unresolved`` was condition 2 TRUE and "
        "condition 1 FALSE. **Here condition 1 is neither true nor "
        "false, it is UNCOMPUTED**, and the two states must not be "
        "written the same way. Computing it needs a read of the "
        "predictions CSVs and no re-run"
    ),
}


#: **[RULED 2026-09-06] THE LEDGER DISPOSITION. The lock left it open
#: with both precedents named. It is ruled here.**
THE_LEDGER_DISPOSITION_RULED = {
    "ruled": "2026-09-06, at the close",

    "what_the_lock_left_open": (
        "``EXIT_CRITERIA`` criterion 9 in the draft: 'Whether this "
        "phase gets a row at all is a lock question. phase26 closed "
        "with no row. The precedent for a train-on-disjoint-set arm is "
        "phase17, which was ledgered. The two precedents point opposite "
        "ways and the restate does not choose'"
    ),

    "THE_RULING_a_row_and_it_is_DESCRIPTIVE": (
        "**one row, status DESCRIPTIVE, both condition fields None.** "
        "Not UNRESOLVED-WITHDRAWN, and not no-row"
    ),
    "why_not_no_row_the_phase_26_precedent": (
        "**``phase26`` closed with no row because it measured that two "
        "settings move nothing, which is a fact about settings and not "
        "a result placed beside another arm.** This phase produced a "
        "correlation on the 237 from a training set nothing in this "
        "record had ever been fit to. **That is a measurement other "
        "phases will want to cite**, and the ledger is where citable "
        "measurements live"
    ),
    "why_not_UNRESOLVED_WITHDRAWN_the_phase_16_precedent": (
        "**because that status means both conditions were evaluated and "
        "split**, which is what ``p16-anchor-loop-unresolved`` records. "
        "Condition 1 here is UNCOMPUTED (THE_CRITERION_NOT_RESOLVED), "
        "and a row claiming a resolved-and-split contrast would say "
        "something the run did not establish"
    ),
    "why_DESCRIPTIVE_and_the_precedent_that_fits": (
        "**``p9-anchor-classifier-convergence`` is the exact shape**: a "
        "measurement on the 25 placed beside the probe, "
        "``condition_1`` None and ``condition_2`` None, status "
        "DESCRIPTIVE. It is also the closest in subject, being the "
        "other phase that took the anchors to the cohort. **The status "
        "matches what was done rather than what was hoped for**"
    ),

    "the_row_is_NOT_WRITTEN_HERE_and_why": (
        "**the run directory name is not known to this record.** The "
        "ledger's ``run_dirs`` field takes the full "
        "``<stem>__<sha8>__<job-id>`` name, the sha reached here as "
        "0fe38121 and the job id did not. **The standing rule is that "
        "runs are named from a supplied name and never selected by "
        "sha, suffix or plausibility**, so the row waits on one line "
        "rather than being guessed"
    ),
    "and_what_the_row_will_say_when_it_is_written": (
        "the claim is the descriptive ordering: a linear head fit on "
        "the 25 anchor grades scores pcc 0.18270409137534543 on the "
        "237 against the probe's 0.2520, a difference of 0.0693 inside "
        "the cohort's measured unresolvable band, with the correlation "
        "itself above the n=237 threshold of 0.128127. **Caveats: the "
        "anchor grades are trusted single grades from the survey "
        "lineage and are not verified-unanimous; the two targets are a "
        "different quantity and only the correlations cross; condition "
        "1 is uncomputed**"
    ),
}


# --------------------------------------------------------------------------
# the closing
# --------------------------------------------------------------------------

#: **[CLOSED 2026-09-06] PHASE 27. The anchor label carries signal the
#: cohort can see and cannot separate from the probe's.**
PHASE_27_CLOSING = {
    "closed": "2026-09-06, run p27_anchor_train at sha 0fe38121",

    "the_finding": (
        "**A HEAD FIT ON 25 ANCHOR GRADES PREDICTS COHORT GRADES ABOVE "
        "CHANCE AND INDISTINGUISHABLY FROM THE PROBE.** Correlation "
        "0.18270409137534543 on all 237 as pure test, above the n=237 "
        "threshold of 0.128127 and 0.0693 below the probe's 0.2520, "
        "which is inside the band this cohort has never resolved a "
        "difference within. **The registered failure did not happen**: "
        "the fit left its initialisation, varies more than the truth it "
        "predicts, and did not collapse to the constant the record "
        "named in advance"
    ),
    "and_the_finding_that_is_NOT_available": (
        "**why.** THE_ASYMMETRY_REGISTERED registered before any number "
        "that a positive result here cannot be attributed, because "
        "nobody knows how the 25 grades were made. **That registration "
        "now binds a real number rather than a hypothetical one**"
    ),

    "criterion_walk": {
        "1_the_motivation_is_rewritten_not_inherited": (
            "**MET at the restate.** The amendment's case FOR rests on "
            "the anchor labels being a consensus, the record refuses "
            "that word at two homes, and the case was withdrawn rather "
            "than carried (THE_MOTIVATION_REWRITTEN, "
            "THE_CONSENSUS_WORD_REFUSED)"
        ),
        "2_the_crossing_is_stated_before_any_number": (
            "**MET at the lock, and CORRECTED by the run.** The "
            "crossing was stated with a derived offset before any "
            "config existed. The offset's SIGN and SIZE were wrong and "
            "are corrected in place (THE_OFFSET_IS_CORRECTED). **The "
            "criterion asked that the crossing be stated first, and it "
            "was**"
        ),
        "3_the_readings_are_registered_both_ways": (
            "**MET at the lock, with a gap recorded at the close.** "
            "Three cases were registered before any number and the "
            "result fell between two of them. The gap is recorded "
            "rather than resolved by stretching a reading "
            "(THE_READINGS_APPLIED)"
        ),
        "4_the_degenerate_signature_is_named_and_its_detector_emitted": (
            "**MET.** Named in advance, detector emitted per seed at "
            "both readout epochs, and the signature DID NOT FIRE "
            "(THE_DEGENERATE_SIGNATURE_DID_NOT_FIRE)"
        ),
        "5_the_disjointness_is_GUARDED_structurally": (
            "**MET.** The guard ran before anything was fitted and "
            "passed at 25 by 237 by 768, no shared identifier and no "
            "identical feature row (THE_GUARD_PASSED)"
        ),
        "6_the_claim_is_bounded_before_the_numbers": (
            "**MET at the lock, and the bound bound.** The result "
            "landed inside the unresolvable band the lock named, and "
            "the n=237 threshold rather than the n=25 one is what the "
            "close applies (THE_CLAIM_BOUNDED, THE_READINGS_APPLIED)"
        ),
        "7_the_arm_measures_its_own_seed_band": (
            "**MET, and the band is zero.** Measured rather than "
            "borrowed, with the mechanism traced in source and the "
            "broken-wiring reading addressed rather than assumed "
            "(THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION)"
        ),
        "8_nothing_is_pooled": (
            "**MET.** The value metrics are reported under the crossing "
            "and named as not comparable, and only the two correlations "
            "are placed beside a cohort-trained arm "
            "(THE_VALUE_METRICS_UNDER_THE_CROSSING)"
        ),
        "9_both_declared_readout_epochs_are_reported_whatever_they_say": (
            "**MET, and it was tested by the result.** Epoch 1 reads "
            "HIGHER than the primary on both correlations. The primary "
            "is epoch 40 because it was declared so before any number, "
            "and epoch 1 is reported beside it "
            "(THE_RESULT_OBSERVED)"
        ),
        "10_the_suite_is_green": "**MET.**",
    },

    "readings_that_fired": {
        "the_two_epochs_AGREE": (
            "the lock registered 'if the two declared epochs AGREE, the "
            "departure at THE_INNER_VALIDATION_RULED cost nothing'. "
            "They agree to three decimals. **Removing checkpoint "
            "selection changed no result** "
            "(THE_BUDGET_MOVED_ALMOST_NOTHING)"
        ),
        "the_result_is_INSIDE_the_band": (
            "the lock's expected outcome, and it happened"
        ),
    },
    "readings_that_did_NOT_fire": {
        "the_degenerate_signature": (
            "registered in advance and refuted by measurement"
        ),
        "the_budget_turning_over": (
            "the lock registered what a rising-then-falling trajectory "
            "would license. **It did not rise.** The maximum is at "
            "epoch 1 and the whole range is 0.002815, so there is no "
            "turnover to read"
        ),
    },

    "what_the_phase_contributed": (
        "**three things, and none of them is about which label is "
        "cleaner.** (1) The crossing is closed: something has now been "
        "fit to the anchor Score and evaluated on the cohort, and the "
        "axis the amendment opened is measured rather than open. (2) "
        "The registered failure did not occur, which is a refuted "
        "prediction. (3) **The first measurement in this record that "
        "the anchor images and the cohort images occupy different "
        "regions of the frozen space**, arriving as a -0.5436 offset "
        "nobody predicted (THE_OFFSET_IS_CORRECTED)"
    ),
    "what_it_did_NOT_contribute": (
        "**any statement about label quality.** The anchor label's "
        "reliability is not computable from what this record holds, no "
        "outcome of this phase makes it computable, and "
        "WHAT_THIS_PHASE_MAY_NOT_CLAIM stands unchanged"
    ),
    "what_is_still_owed": (
        "**condition 1**, a per-seed paired BCa over the 237 from the "
        "predictions CSVs, which needs no re-run; **the ledger row**, "
        "which needs the run directory name; and **per-rater grades for "
        "the 25**, which are what would make any of this "
        "interpretable and are supervision ask 7's neighbour rather "
        "than ask 7 itself"
    ),

    "tag": "[CLOSED] -- one measurement, one refuted prediction, one row owed",
}


#: **[RECORDED 2026-09-06] THE ROW IS WRITTEN.**
THE_LEDGER_ROW_WRITTEN = {
    "written": "2026-09-06, ledger entry 39",
    "id": "p27-anchor-train-descriptive",
    "run_dir": (
        "``p27_anchor_train__0fe38121__p27-anchor-train``, **SUPPLIED "
        "rather than constructed.** Only the sha reached this record "
        "with the result, and the standing rule is that a run is named "
        "from a supplied name and never selected by sha, suffix or "
        "plausibility. The row waited on that one line and no longer "
        "does"
    ),
    "shape": (
        "**status DESCRIPTIVE, framing additional, both condition "
        "fields None**, exactly as ruled at "
        "THE_LEDGER_DISPOSITION_RULED and on the "
        "``p9-anchor-classifier-convergence`` precedent"
    ),
    "why_condition_2_is_NOT_in_the_row": (
        "**it is computable and it passes at 5.34x, and it still does "
        "not belong here.** A row carrying one condition and not the "
        "other reads as a contrast that was run under the criterion. "
        "**It was not.** Condition 1 is UNCOMPUTED, which is neither "
        "TRUE nor FALSE, and the two states must not be written the "
        "same way. The figure lives at THE_CRITERION_NOT_RESOLVED where "
        "its status is legible"
    ),
    "the_chain_held": (
        "appending leaves every earlier prefix checksum unchanged, and "
        "the two the suite pins were checked: n=37 and n=38 both "
        "reproduce their pinned values after the append. **That is the "
        "append-only rule with teeth, and it passed**"
    ),
}


#: **[REQUIRED 2026-09-06] WHAT CONDITION 1 NEEDS, and what would not
#: do instead.**
#:
#: The per-patient predictions are CLUSTER-ONLY and cannot travel. This
#: record says what can.
THE_CONDITION_1_REQUIREMENT = {
    "recorded": "2026-09-06",

    "what_the_computation_IS": (
        "**a per-seed paired BCa over the 237 patients, of the "
        "difference in PCC between this arm and the probe**, and the "
        "verdict that every seed's interval excludes zero in ONE "
        "direction. ``phase7b.paired_comparison`` is the shipped "
        "function and no second implementation may be written, which is "
        "the standing rule the record calls R10"
    ),

    "what_it_NEEDS_and_the_form": (
        "**patient-keyed vectors, and nothing aggregate substitutes.** "
        "For each of the five seeds: the 237 cohort truth values, this "
        "arm's 237 predictions, and the probe's 237 predictions, "
        "aligned on patient id. That is "
        "``seed_<n>__predictions.csv`` from "
        "``p27_anchor_train__0fe38121__p27-anchor-train`` and the same "
        "from the probe's run directory, both CLUSTER-ONLY"
    ),
    "why_an_AGGREGATE_will_not_do_on_the_INPUT": (
        "**because the bootstrap resamples PATIENTS, and its own "
        "docstring says why the pairing has to survive.** "
        "``eval.metrics.bca_ci``: 'All samples are resampled with one "
        "shared index vector, so pairing over patients is preserved ... "
        "comparing two arms' separate CIs is invalid because they share "
        "the same patients.' **The interval depends on the joint "
        "distribution of the per-patient paired differences**, which a "
        "mean, an sd, or any other summary does not determine. Two arms "
        "with identical means and sds can produce intervals on opposite "
        "sides of zero"
    ),
    "and_this_is_NOT_a_workaround_that_could_be_found_with_effort": (
        "**it is what a paired test is.** A summary that determined the "
        "interval would have to carry the 237 paired differences, which "
        "is the patient-keyed vector under another name"
    ),

    "what_CAN_travel_and_it_is_the_whole_answer": (
        "**the OUTPUT is fully aggregate and carries no patient.** "
        "``paired_comparison`` returns, per seed, a delta and its lo and "
        "hi bounds and whether the interval excludes zero, plus the "
        "family-level mean delta, the count excluding zero, the "
        "direction fields and the threshold. **Five triples of numbers "
        "and a verdict.** Nothing in it is patient-identifying, so it "
        "ships back as a summary under the same tier rules every other "
        "phase's metrics.json travels under"
    ),
    "so_the_tier_rule_is_not_an_obstacle_here": (
        "**it is the discipline working.** The computation goes to the "
        "data and the summary comes back, which is what CLUSTER-ONLY is "
        "for. Nothing about this phase needs a patient to leave the "
        "cluster"
    ),

    "and_the_five_intervals_will_NOT_be_identical": (
        "**stated because a deterministic arm invites the assumption "
        "that they would be.** This arm's predictions are bit identical "
        "across seeds, but the PROBE's are not: its seed sd is 0.0148. "
        "So the five paired deltas differ on the baseline side, and "
        "condition 1 is a real five-way check rather than one interval "
        "computed five times"
    ),

    "WHAT_MUST_BE_BUILT_because_no_shipped_task_covers_it": (
        "**there is no zero-build path, and the reason is deliberate.** "
        "Every contrast task in this repository fixes its own family in "
        "code so it cannot grow through a config edit: "
        "``task_p22_contrasts`` refuses unless "
        "``phase22.contrast_family()`` returns exactly 15, "
        "``task_p25_contrasts`` takes its pairs from "
        "``phase25.contrast_family()``, and ``task_paired_claims`` "
        "takes a ``scope`` from a CLOSED vocabulary whose enumeration "
        "lives in each phase's own module. **None of them can be "
        "pointed at a pair they were not written for, and that is a "
        "guard rather than a gap**"
    ),
    "the_build_is_ONE_scope_and_ONE_enumeration": (
        "add ``p27`` to ``paired_claims``'s ``scope`` choices, and a "
        "pair enumeration in this module naming this arm against the "
        "probe on the five shared seeds. **That is the route Phases "
        "10, 11 and 12 each took**, and their scope comments record it. "
        "It writes no new paired implementation, touches no frozen "
        "apparatus, and needs no re-run of either arm"
    ),
    "and_until_it_exists": (
        "**condition 1 stays UNCOMPUTED and the row stays "
        "DESCRIPTIVE.** The phase does not claim the contrast, and "
        "``THE_CRITERION_NOT_RESOLVED`` says so in the record rather "
        "than leaving a reader to infer it from a missing field"
    ),
    "BUILT_2026_09_06": (
        "**the route above was taken and the entries above are the "
        "state before it.** ``p27`` joined ``paired_claims``'s scope "
        "vocabulary, ``paired_claim_pairs`` in this module enumerates "
        "the one pair, and ``run.task_paired_claims`` dispatches to "
        "it in the same chain as Phases 10, 11 and 12. **No paired "
        "implementation was written and no frozen apparatus was "
        "touched**, which the suite asserts rather than the record "
        "claiming it. The config is ``configs/p27_paired.yaml``"
    ),
    "and_the_computation_has_still_NOT_RUN": (
        "**building it is not computing it.** Five of the ten vectors "
        "are declared at hashes carried from "
        "``configs/p10_paired.yaml`` and five are OWED: this arm's "
        "prediction files have never been declared anywhere, so the "
        "config is UNRESOLVED until a declare pass on the cluster "
        "fills them. **Condition 1 remains UNCOMPUTED and the row "
        "remains DESCRIPTIVE**, and what it becomes under either "
        "outcome is at THE_ROW_UNDER_EACH_OUTCOME"
    ),
}


# --------------------------------------------------------------------------
# condition 1: the pair enumeration, on the route the earlier phases took
# --------------------------------------------------------------------------

#: The two run-directory stems this scope reads, named once so the
#: generator, the config and the record cannot drift.
#:
#: The baseline stem is quoted from ``phase10.PAIRED_BASELINE_STEM``
#: rather than retyped, because it is the same arm and a second spelling
#: of one name is how two records come to disagree about one object.
PAIRED_ARM_STEM = "p27_anchor_train"
PAIRED_BASELINE_STEM = "p7_d1_vit_b16_imagenet_g1"


#: **[REGISTERED 2026-09-06] WHAT THE SCOPE COVERS AND WHAT IT DOES
#: NOT.** Shaped like every other scope's so the reader who knows one
#: knows this one.
PAIRED_CLAIM_COVERAGE = {
    "covers": (
        "one pair: this arm against the probe, five shared seeds, the "
        "same 237 patients and the same truth column. The convention is "
        "the headline scope's, ``b`` is the 0.2520 arm, so **a POSITIVE "
        "delta means the probe is ahead**"
    ),
    "excluded": {
        "the_second_declared_readout_epoch": (
            "**epoch 1 is reported but not paired.** The primary was "
            "declared epoch 40 before any number "
            "(THE_TWO_READOUT_EPOCHS_DECLARED), and pairing both would "
            "make the scope a family of two whose better member could "
            "be quoted. One pair, on the declared primary"
        ),
        "the_value_metrics": (
            "RMSE, MAE and three-class accuracy carry a measured "
            "offset of -0.5436 and are not comparable across the two "
            "targets at all (THE_LABEL_INCOMPARABILITY_ADDRESSED). "
            "**Only the correlation crosses**, so only the correlation "
            "is paired"
        ),
        "spearman": (
            "it crosses cleanly too and is NOT paired, because "
            "PLAN section 4.3's condition 1 is defined on the "
            "project's primary metric and adding a second would be a "
            "family of two under one name. Phase 11 is the only scope "
            "that reports two, and it says why"
        ),
    },
    "and_one_asymmetry_recorded_rather_than_smoothed": (
        "**the probe's 237 predictions come from FIVE models, one per "
        "fold, and this arm's come from ONE.** Both are honest held-out "
        "predictions for every patient against the same truth, so the "
        "pairing is legitimate. But they are not the same kind of "
        "object, and a reader should know that before reading an "
        "interval"
    ),
    "and_the_oof_prefix_is_the_LOADERS_convention_not_a_claim": (
        "``phase7c.oof_paths_from_inputs`` requires inputs named "
        "``oof_<stem>_seed_<n>``, so this scope's declarations carry "
        "that prefix. **This arm's vectors are not out-of-fold. They "
        "are PURE TEST**, because no cohort patient was ever in the "
        "fit. That is a stronger condition than out-of-fold, not a "
        "weaker one, and the prefix is a filename convention rather "
        "than a statement about the vectors"
    ),
}


def paired_claim_pairs(scope: str = "p27") -> list[dict]:
    """Phase 27's single contrast, shaped like every other scope's so the
    ONE paired implementation serves it.

    ``run.task_paired_claims`` walks these, ``ladder``'s three
    derivations take them as an argument, and
    ``phase7b.paired_comparison`` does the arithmetic. **Nothing here
    re-implements a comparison**, which is the rule the record calls R10
    and the route ``phase10``, ``phase11`` and ``phase12`` each took.

    **Every recorded figure is DERIVED, not typed.** The arm's side comes
    from ``THE_RESULT_OBSERVED``, which was read from the run, and the
    probe's from ``ladder.STAGE_D1_AT_G1``, so a correction to either
    lands here rather than in one record and not the other.

    **``STAGE_D1_AT_G1``, for the reason phase10 gives.** The probe has
    two recorded figures, 0.2520 through the D1 artifact path and 0.25206
    live, and **the vectors this pair declares are the D1 run's**, so the
    D1 figures are the ones that describe them.
    """
    if scope != "p27":
        raise Phase27Error(f"unknown Phase 27 paired-claim scope {scope!r}")

    from . import ladder
    from .train.phase3 import combined_claimable_delta

    baseline_mean = ladder.STAGE_D1_AT_G1["cells"]["vit_b16"][0]
    baseline_sd = ladder.STAGE_D1_AT_G1["sd"]["vit_b16"]["imagenet"]
    seeds = list(ladder.SEED_POOL[:5])
    arm_mean = THE_RESULT_OBSERVED["primary_epoch_40"]["pcc"]
    #: ZERO, and measured (THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION). The
    #: threshold is smaller for it by one over the square root of two,
    #: and THE_CRITERION_NOT_RESOLVED records that condition 2 passes
    #: either way, so the zero is not doing the work.
    arm_sd = 0.0
    threshold = combined_claimable_delta(
        baseline_sd, len(seeds), arm_sd, len(seeds)
    )["arm_means_95"]
    delta = baseline_mean - arm_mean
    return [{
        "key": "p27__anchor_train_vs_0p2520",
        "question": "does a head fit on the 25 anchor grades reach the probe",
        "varies": "training set+target",
        # `b` is the 0.2520 arm, so a POSITIVE delta means the probe is
        # ahead -- the headline scope's convention, kept.
        "a": PAIRED_ARM_STEM,
        "b": PAIRED_BASELINE_STEM,
        "seeds": seeds,
        "recorded": {
            "delta_of_means": round(delta, 4),
            "threshold": round(threshold, 4),
            "margin": round(delta / threshold, 2),
            "positive_means": "positive = the probe is ahead",
            "source": (
                "THE_RESULT_OBSERVED (read from the run) and "
                "ladder.STAGE_D1_AT_G1 (the D1 artifact path, which is "
                "what the declared vectors are); DESCRIPTIVE, not "
                "claimable -- the run computes the real one"
            ),
        },
    }]


#: **[REGISTERED 2026-09-06, BEFORE THE RUN] WHAT THE LEDGER ROW BECOMES
#: UNDER EACH OUTCOME, and whether its status moves.**
#:
#: Written before the intervals exist, so the row's fate is not decided
#: by reading them.
THE_ROW_UNDER_EACH_OUTCOME = {
    "registered": "2026-09-06, before the paired run",

    "what_changes_in_every_case": (
        "**``condition_1`` stops being None.** It becomes the verdict in "
        "the vocabulary the ledger already uses, which "
        "``p16-anchor-loop-unresolved`` writes as 'FALSE -- 0 of 5 seed "
        "BCa intervals exclude zero, directions mixed'. "
        "**``condition_2`` also stops being None**, because with "
        "condition 1 evaluated the row is a contrast and both fields "
        "belong: it becomes TRUE at 5.34x"
    ),

    "if_condition_1_PASSES_all_five_exclude_zero_one_direction": (
        "**status becomes CLAIMABLE and the row is the phase's only "
        "claim.** Both conditions met, so PLAN section 4.3 is satisfied "
        "and the sentence available is that the probe scored claimably "
        "higher than a head fit on the anchor grades. **The point "
        "estimate 0.0693 sits inside the band this cohort has never "
        "resolved**, so this outcome would be the cohort resolving "
        "something it has not resolved before, and "
        "``ladder.SMALLEST_RESOLVABLE_DIFFERENCE`` would need a dated "
        "amendment naming a new smallest. **That is a finding about the "
        "instrument as much as about the arms**"
    ),
    "if_condition_1_FAILS": (
        "**status becomes UNRESOLVED-WITHDRAWN and the row stays "
        "non-claiming.** Condition 2 TRUE and condition 1 FALSE is "
        "exactly ``p16-anchor-loop-unresolved``'s shape, and it would be "
        "the second anchor contrast to land there. The descriptive "
        "ordering survives unchanged and nothing else does"
    ),
    "which_is_EXPECTED_and_the_record_says_so_before_the_run": (
        "**condition 1 is expected to FAIL.** "
        "``ladder.COHORT_CANNOT_RESOLVE`` measured 30 comparisons and "
        "one survived, and a delta of 0.0693 is inside its 0.04 to 0.10 "
        "unresolvable band. **Registering the expectation before the "
        "intervals exist is what makes the other outcome readable if it "
        "happens**"
    ),

    "what_does_NOT_change_under_any_outcome": (
        "**the claim sentence and every caveat.** The correlation, the "
        "prediction spread, the offset and the six caveats are "
        "measurements and stay as written. A condition-1 verdict adds a "
        "field, it does not revise a figure. **And no outcome licenses a "
        "sentence about cleaner labels**, because "
        "THE_ASYMMETRY_REGISTERED binds on the label's construction "
        "rather than on the interval"
    ),
    "and_the_row_is_AMENDED_rather_than_replaced": (
        "**the ledger is append-only.** A condition field moving from "
        "None to a verdict is a REWRITE of entry 39, which the "
        "cumulative checksum forbids. So the change appends: either the "
        "row is written once, after the paired run, or a correcting "
        "entry appends using the ``corrects`` mechanism, whose only "
        "prior user is "
        "``ledger-condition-split-count-corrected``. **Which of the two "
        "is a maintainer decision and this record does not take it**"
    ),
}


def summary() -> dict:
    """Every record in this module, for the restate sweep."""
    return {
        "reckoning": THE_RECKONING,
        "motivation": THE_MOTIVATION_REWRITTEN,
        "consensus_refused": THE_CONSENSUS_WORD_REFUSED,
        "asymmetry": THE_ASYMMETRY_REGISTERED,
        "incomparability": THE_LABEL_INCOMPARABILITY_ADDRESSED,
        "readings": READINGS,
        "degenerate_signature": THE_DEGENERATE_SIGNATURE_REGISTERED,
        "claim_bounded": THE_CLAIM_BOUNDED,
        "may_not_claim": WHAT_THIS_PHASE_MAY_NOT_CLAIM,
        "prohibition": THE_PROHIBITION,
        "fold_machinery": THE_FOLD_MACHINERY_ANSWERED,
        "precedent": THE_PRECEDENT_THAT_ALREADY_RUNS,
        "must_be_built": WHAT_MUST_BE_BUILT,
        "two_extractions": THE_TWO_EXTRACTIONS_COMPARED,
        "settings": SETTINGS_NEEDING_DECLARATION,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        "inner_validation_ruled": THE_INNER_VALIDATION_RULED,
        "readout_epochs": THE_TWO_READOUT_EPOCHS_DECLARED,
        "settings_ruled": THE_SETTINGS_RULED,
        "disjointness_guard": THE_DISJOINTNESS_GUARD_DESIGNED,
        "exit_criteria": EXIT_CRITERIA,
        "readings_at_the_lock": READINGS_AT_THE_LOCK,
        "init_corrected": THE_INIT_IS_THE_PROBES,
        "budget_expectation": THE_BUDGET_EXPECTATION_REGISTERED,
        "guard_passed": THE_GUARD_PASSED,
        "result": THE_RESULT_OBSERVED,
        "seed_sd_zero": THE_SEED_SD_IS_ZERO_BY_CONSTRUCTION,
        "signature_did_not_fire": THE_DEGENERATE_SIGNATURE_DID_NOT_FIRE,
        "budget_moved_nothing": THE_BUDGET_MOVED_ALMOST_NOTHING,
        "offset_corrected": THE_OFFSET_IS_CORRECTED,
        "value_metrics": THE_VALUE_METRICS_UNDER_THE_CROSSING,
        "readings_applied": THE_READINGS_APPLIED,
        "criterion_not_resolved": THE_CRITERION_NOT_RESOLVED,
        "ledger_disposition": THE_LEDGER_DISPOSITION_RULED,
        "ledger_row": THE_LEDGER_ROW_WRITTEN,
        "condition_1_requirement": THE_CONDITION_1_REQUIREMENT,
        "paired_coverage": PAIRED_CLAIM_COVERAGE,
        "row_under_each_outcome": THE_ROW_UNDER_EACH_OUTCOME,
        "closing": PHASE_27_CLOSING,
    }
