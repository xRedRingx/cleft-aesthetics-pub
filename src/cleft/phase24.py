"""Phase 24: all five raters -- THREE DESIGNS CONCEDED AT THE RECKONING,
one conceded at its premise, and a substituted measurement.

**The phase opens by conceding most of itself**, on Phase 14's precedent:
*"The concession is the phase doing its job, not the phase failing."*
Three of the amendment's four designs are already covered by measurements
in the record, and the fourth's motivation is refuted. What replaces them
is the one rater question nothing in the record has answered -- **how many
raters this cohort would need for a stated reliability** -- which is two
lines on a function that already exists.

**REGISTERED, NOT BUILT.** No task, no config, no arm, no run.
"""

from __future__ import annotations

#: **[RECKONING 2026-09-02] PHASE 24 OPENS BY CONCEDING.**
#:
#: The amendment (``phase21.PHASE_SEQUENCE_EXTENDED_7``) scheduled four
#: designs: a label-distribution head, rater random effects, agreement-
#: weighted training, and disagreement as an auxiliary target. It bound
#: itself explicitly: *"which of the four designs is built, WHETHER ANY
#: IS, the target, the criteria and every reading"* were all left open.
#:
#: **Three are covered and the fourth's motivation is refuted.**
PHASE_24_RECKONING = {
    "opened": "2026-09-02 -- registration only; nothing built",
    "the_precedent_that_governs": (
        "**phase14.PHASE_14_CONCEDED_COVERED.** A phase may close at its "
        "reckoning when the record already covers the mechanism, and "
        "**that is a legitimate outcome rather than a failure**: 'The "
        "shortest phase in the project, and correctly so ... The "
        "concession is the phase doing its job, not the phase failing.' "
        "Phase 14 conceded on IDENTITY, checkable line by line, and this "
        "reckoning concedes on the same grounds where they hold"
    ),
    "what_is_conceded": (
        "**three of four designs, and the fourth at its premise.** "
        "Label-distribution head: CONCEDED-COVERED on identity with the "
        "closed 0.2336 arm. Per-rater heads: CONCEDED-MEASURED -- the "
        "screen ran all five on this cohort. Agreement-weighted "
        "training: CONCEDED at its premise, which Phase 21 refuted. "
        "Disagreement as an auxiliary target: **not covered by any arm**, "
        "and conceded because no non-refuted motivation survives the "
        "frozen-backbone recipe"
    ),
    "what_is_NOT_conceded_and_replaces_them": (
        "**the reliability curve and its inverse** "
        "(THE_SUBSTITUTED_DELIVERABLE). It is the question the four "
        "designs were reaching for -- what the panel's agreement costs, "
        "and what more raters would buy -- **and nothing in the record "
        "computes it**"
    ),
    "the_honest_shape_of_this_phase": (
        "**measurement, not modelling.** Every conceded design would "
        "have trained something; the substitute trains nothing, needs no "
        "cluster, no clinical data and no run. **A phase that concedes "
        "three quarters of its scope and delivers a two-line "
        "computation is a small phase**, and the record should not "
        "inflate it into more"
    ),
    "no_ledger_row": (
        "**this phase RESTATES and DESCRIBES; it does not CLAIM.** The "
        "curve is a projection from a banked constant, not a result "
        "about any model. No row is registered"
    ),
    "tag": "[REGISTERED] -- nothing built",
}


# --------------------------------------------------------------------------
# the four designs, each with its concession
# --------------------------------------------------------------------------

#: **[CONCEDED-COVERED 2026-09-02] Design 1: the label-distribution
#: head.** Phase 14 already conceded this exact mechanism, and the
#: concession is repeated here rather than reopened.
THE_LABEL_DISTRIBUTION_HEAD_CONCEDED_COVERED = {
    "verdict": "CONCEDED-COVERED, on identity",
    "the_identity": (
        "**identical on every axis to the closed arm**, as Phase 14 "
        "established by reading the module rather than trusting a "
        "description: 'a FIVE-OUTPUT SOFTMAX HEAD over the frozen 768-d "
        "embeddings, KL LOSS (target||predicted, batchmean), targets the "
        "SOFT_1..SOFT_5 distribution, readout the EXPECTATION over "
        "grades 1-5 for PCC against the mean column ... HEAD, LOSS, "
        "TARGET, READOUT, RECIPE: identical on every axis. This is "
        "identity, checkable line by line -- not a judgement call'"
    ),
    "what_it_measured": (
        "**ladder.STAGE_G_LABEL_FORMULATION, 2026-08-02, five seeds.** "
        "imagenet/G1: ldl **0.2336** (sd 0.0305) against the mean's "
        "0.2520 (sd 0.0148), delta -0.0184 inside its 0.0297 threshold "
        "-- 'label is NULL at this operating point'. scut_masked/G2: "
        "**0.1583** against 0.2001, delta -0.0418 -- **CLAIMABLY "
        "WORSE**. The stage verdict: 'No label alternative beats the "
        "mean anywhere'"
    ),
    "the_clause_that_forbids_renaming_it": (
        "**phase14.PHASE_14_CONCEDED_COVERED['variants_named_not_"
        "pursued'], quoted because it was written for exactly this "
        "moment**: 'EXPLICITLY UNREGISTERED -- no fishing. Two variants "
        "were named in the restatement as what a genuinely different "
        "mechanism would have to look like: LDL-style loss as "
        "PRETRAINING for a scalar head, and distribution supervision at "
        "a DIFFERENT REPRESENTATION. Neither is registered, neither has "
        "a record-based reason to expect a different outcome, and "
        "**naming them here is what prevents a later turn from "
        "presenting either as a fresh idea that escapes this record**'"
    ),
    "rebuilding_it_under_a_new_number_is_what_that_clause_prevents": (
        "**the amendment scheduled a 'label-distribution head' as Phase "
        "24's first design. That is the closed arm with a phase number "
        "in front of it** -- the precise move the clause above exists to "
        "stop. Recorded so the concession is visibly the clause working, "
        "not a second opinion about the same evidence"
    ),
    "the_reopen_condition_STANDS_UNCHANGED": (
        "phase14's own, quoted and not weakened: 'only a NEW measured "
        "reason -- a mechanism differing from the closed arm's on a "
        "named axis, with a record-based prediction of why the "
        "difference matters. Scheduling was not reopening; closing is "
        "not forbidding.' **Scheduling by the seventh amendment is not "
        "a new measured reason either**"
    ),
}


#: **[CONCEDED-MEASURED 2026-09-02] Design 2: rater random effects /
#: per-rater heads.** Not covered by argument -- covered by a run.
THE_PER_RATER_HEADS_CONCEDED_MEASURED = {
    "verdict": "CONCEDED-MEASURED, on this cohort under this criterion",
    "the_measurement": (
        "**phase10.RATER_SCREEN_OBSERVED, 2026-08-17** -- the 0.2520 arm "
        "refit once per rater, five seeds each, 5-fold OOF: cleft "
        "patient **-0.0098** (sd 0.0207), orthodontist **0.1606** "
        "(0.0207), speech and language therapist **0.2150** (0.0149), "
        "plastic surgeon **0.2703** (0.0196), psychologist **0.1362** "
        "(0.0283). **Four below 0.2520; one nominally above and INSIDE "
        "the band** -- +0.0183 is 1.24 of the arm's own seed sd, within "
        "the registered two-sd refutation threshold"
    ),
    "it_is_ledgered": (
        "**results_ledger row 22, ``p10-rater-screen-mixed``, "
        "DESCRIPTIVE.** Its caveat travels: 'each rater is a DIFFERENT "
        "target, so these are not paired against the 0.2520 arm's "
        "vectors -- the reading is level against the arm and its seed sd "
        "0.0148'"
    ),
    "the_prior_was_committed_both_ways_first": (
        "the registration named both outcomes before any number: every "
        "cell below 0.2520 -> rater-specific modelling measured worse; "
        "any cell beyond the seed band -> prior refuted and the full "
        "ladder justified. **The MIXED reading fired**, which is the "
        "registered third outcome, not an absence of one"
    ),
    "a_phase_24_of_per_rater_heads_IS_the_registered_ladder": (
        "**phase10.RATER_LADDER_CONDITIONAL: 5 raters x 4 backbones x 2 "
        "geometries x 2 inits x 5 seeds = 400 runs, REGISTERED AND NOT "
        "BUILT.** Its trigger is 'the screen refuting its prior, or the "
        "maintainer's word regardless', and **the screen did not refute "
        "it** -- no cell beyond the band. So the record's own condition "
        "for building this is UNMET, and a Phase 24 that trained "
        "per-rater heads would be that ladder under a different name"
    ),
    "the_maintainers_word_remains_a_live_trigger": (
        "**and it is not being exercised here.** The ladder's second "
        "trigger is the maintainer's word regardless of the screen; this "
        "concession is about the EVIDENCE condition, which is unmet. "
        "Recorded so that a later decision to build it is visibly a "
        "decision about scope, exactly as its registration intended"
    ),
    "a_second_independent_route_agrees": (
        "Phase 1 measured the same ordering as LEARNABILITY before any "
        "arm ran -- mean 0.6022 against orthodontist 0.4944 "
        "(labels.LEARNABILITY_237). **Different data path, same "
        "conclusion**: the panel mean is a better target than any single "
        "rater"
    ),
}


#: **[CONCEDED AT ITS PREMISE 2026-09-02] Design 3: agreement-weighted
#: training.**
AGREEMENT_WEIGHTED_TRAINING_CONCEDED_AT_ITS_PREMISE = {
    "verdict": "CONCEDED at its premise -- the premise is refuted, not untested",
    "the_premise_it_needs": (
        "**weighting by disagreement assumes disagreement marks the "
        "cases that matter.** Down-weighting the disputed patients (or "
        "up-weighting the agreed ones) is only a different objective if "
        "the two sets differ in something the model is failing on"
    ),
    "the_measurement_that_refutes_it": (
        "**phase21.ARM_B_OBSERVED['cross_reference'].** tau-b(hardness, "
        "rater disagreement) = **+0.0727** (permutation p 0.1226) on "
        "residual sign and **-0.0811** (p 0.0835) on worst quartile -- "
        "'small, insignificant, and OPPOSITE IN SIGN. Hardness does NOT "
        "track rater disagreement.' The committed cell "
        "``b_hardness_does_not_correlate_with_disagreement`` FIRED, and "
        "what it ruled out was named in advance: 'the comfortable "
        "answer ... hard patients are NOT the patients the panel argued "
        "about'"
    ),
    "these_taus_POSTDATE_the_correction_and_are_not_lower_bounds": (
        "**the distinction matters and is stated rather than assumed.** "
        "``phase11.kendall_tau_b`` was wrong for doubly-tied pairs, so "
        "every tau banked before 2026-09-01 is a LOWER BOUND -- six "
        "Phase 18 figures among them. **These two are not**: the defect "
        "was found and corrected WHILE PHASE 21 WAS BEING BUILT "
        "(phase21's own dated record), and Arm B ran after it. So "
        "+0.0727 and -0.0811 are the corrected statistic, and the "
        "refutation cannot be explained away as an understated tau"
    ),
    "the_preconditions_were_measured_too": (
        "the rater sd agreed to **2.22e-16** across two independent "
        "derivation routes, and disagreement carries **18 distinct "
        "values with a largest tie group of 59** -- which is why tau-b "
        "with a permutation null was the registered statistic and not "
        "Pearson. **The refutation rests on a statistic chosen for this "
        "tie structure in advance**"
    ),
    "what_would_reopen_it": (
        "a measured association between disagreement and something an "
        "arm is failing on -- not necessarily hardness, but SOMETHING, "
        "named and measured before the weighting is built. Weighting by "
        "a variable with no measured bearing is choosing an objective "
        "for its story"
    ),
}


#: **[CONCEDED 2026-09-02, WITH ITS REASON] Design 4: disagreement as an
#: auxiliary target.**
#:
#: **This one is NOT covered by an existing arm**, and the record should
#: say so plainly rather than fold it into the other three.
DISAGREEMENT_AS_AUXILIARY_TARGET_STATUS = {
    "verdict": (
        "CONCEDED -- **not because it was measured, but because no "
        "non-refuted motivation survives**"
    ),
    "it_is_NOT_covered_by_any_arm": (
        "**stated first, because it is the honest part.** No arm in this "
        "project has predicted rater disagreement, as a primary or "
        "auxiliary target. Stage G varied the LABEL (mean / median / "
        "distribution); the rater screen varied the RATER. **Neither "
        "predicts the spread.** This design is genuinely unbuilt"
    ),
    "and_it_is_manifest_reachable": (
        "**the cheapest of the four to build**, which is why its "
        "concession needs a reason rather than a shrug. "
        "phase21.INTER_RATER_DISAGREEMENT_MEASURED: soft_k * 5 is an "
        "exact integer count, the rater multiset is recoverable and the "
        "per-patient sd follows EXACTLY -- verified over 20,000 random "
        "panels to **4.44e-16**. No score sheet needed; the manifest "
        "alone suffices"
    ),
    "the_motivation_it_shares_with_design_3_is_REFUTED": (
        "'the disputed cases are the hard ones' is the same premise "
        "Phase 21 refuted, and an auxiliary target motivated by it "
        "inherits the refutation whole "
        "(AGREEMENT_WEIGHTED_TRAINING_CONCEDED_AT_ITS_PREMISE)"
    ),
    "the_one_motivation_that_would_NOT_be_refuted_and_why_it_fails_HERE": (
        "**multi-task learning as REGULARISATION** -- an auxiliary head "
        "helping the primary by forcing a richer REPRESENTATION, "
        "regardless of whether disagreement marks hard cases. That "
        "motivation does not route through the refuted premise and is a "
        "genuinely different argument. **It fails on this project's "
        "recipe for a structural reason: THE BACKBONE IS FROZEN.** Every "
        "arm in the ladder, p7, p17 and p22 refits a head over "
        "pre-extracted embeddings -- the rater screen's own registration "
        "says 'only the head refits'. **An auxiliary head sharing a "
        "FIXED representation has nothing to regularise**; the two heads "
        "share an input they cannot change. The only end-to-end "
        "fine-tuning in the record is the CleftGNN replication, whose "
        "three launches are ledgered VOID"
    ),
    "the_second_reason_beside_the_structural_one": (
        "**the target is very coarse.** Disagreement takes only 26 "
        "distinct values in principle and **18 on our 237, with a "
        "largest tie group of 59**. An auxiliary target that is nearly "
        "constant over a quarter of the cohort carries little to learn "
        "from even where a representation could move"
    ),
    "what_would_reopen_it": (
        "**an arm that trains its representation.** If this project ever "
        "fine-tunes a backbone on the cleft target, the regularisation "
        "motivation becomes available and unrefuted, and this design "
        "should be reconsidered THEN -- with a record-based prediction "
        "of why the auxiliary signal would help. Recorded as a "
        "condition, not a plan"
    ),
    "it_is_not_conceded_silently": (
        "**the concession is on the record with its reason**, per the "
        "instruction that a design not covered by an arm must not be "
        "quietly folded into the ones that are. If the structural "
        "argument above is wrong, this is the design that should be "
        "built"
    ),
}


# --------------------------------------------------------------------------
# what replaces them
# --------------------------------------------------------------------------

#: **[REGISTERED 2026-09-02 -- NOT COMPUTED] THE SUBSTITUTED
#: DELIVERABLE: the reliability curve and its inverse.**
#:
#: ``reliability.spearman_brown(mean_r, k)`` accepts arbitrary ``k`` and
#: has since it was written. **Every caller passes 5 or 1**: ``summarise``
#: uses ``array.shape[1]``, ``labels`` uses 1 for the orthodontist and 5
#: otherwise. The only k in {2, 3, 10} anywhere is a MONOTONICITY test
#: that banks no value. **Generalizability theory is absent from the
#: repository entirely** -- no G-study, no D-study, nothing.
#:
#: So the question the four designs were reaching for -- what the panel's
#: low agreement costs, and what more raters would buy -- **has never
#: been answered, and answering it needs no cluster, no clinical data and
#: no run**.
THE_SUBSTITUTED_DELIVERABLE = {
    "registered": "2026-09-02 -- REGISTERED, NOT COMPUTED",
    "the_quantity": (
        "**three things from one banked constant.** (i) Reliability at "
        "**k = 1..20** from ``MEAN_R_237 = 0.4696`` via "
        "``reliability.spearman_brown``; (ii) the implied PCC ceiling "
        "``sqrt(reliability)`` at each k, via ``reliability.pcc_ceiling`` "
        "-- the two are DIFFERENT QUANTITIES and the module says so "
        "first; (iii) **the inverse** -- the k required for reliability "
        "0.85, 0.90 and 0.95"
    ),
    "the_population_is_237_and_only_237": (
        "**MEAN_R_237 = 0.4696, never MEAN_R_251 = 0.4560.** The 237 are "
        "the patients with photographs and the cohort the manifest "
        "holds. ``data.reliability``'s own words: comparing a value "
        "computed on one population against one computed on another 'is "
        "exactly how the 0.903 ceiling came to be corrected to 0.807' -- "
        "**the third time that trap has been walked into**. Whether 251 "
        "is reported beside it is a SETTING TO DECLARE, not a default"
    ),
    "why_it_is_the_right_substitute": (
        "**it answers the rater question the conceded designs asked, "
        "from the direction the record can actually support.** All four "
        "asked how to USE five raters better; none of them could, and "
        "three were already measured not to. This asks what five raters "
        "BUY and what more would buy -- which is a property of the "
        "panel, not of a model, and is therefore not refuted by anything "
        "about models"
    ),
    "the_cost": (
        "**two lines on an existing function.** No task is needed that "
        "does not already exist, no config, no embeddings, no patient "
        "data. It is the cheapest deliverable in the project's history "
        "and that is an argument for it, not against"
    ),
    "it_is_a_PROJECTION_not_a_measurement_of_raters_we_have": (
        "**k > 5 describes panels this project never observed.** The "
        "curve is Spearman-Brown extrapolation from the agreement of "
        "five actual raters, and every value at k != 5 is a MODEL of a "
        "panel, not a measurement of one. Tagged accordingly wherever it "
        "is reported"
    ),
    "tag": "[REGISTERED] -- not computed; the numbers do not exist yet",
}


#: **[LITERATURE + ARITHMETIC CHECK 2026-09-02] The lineage's own
#: projection, and whether our ``mean_r`` reproduces it.**
#:
#: **This single point WAS computed at registration**, on the maintainer's
#: instruction, because it is a check on the arithmetic against an
#: independent application of the same formula. **It is reported here as
#: a check and NOT as one of the phase's readings**, and the record notes
#: below which reading it bears on, so nobody can later call the readings
#: blind.
THE_LINEAGE_CHECK = {
    "recorded": "2026-09-02",
    "the_literature_projection": (
        "**[LITERATURE]** phase12.PRIMARY_SOURCES_BANKED"
        "['ceiling_provenance']: 'Spearman-Brown is the lineage's OWN "
        "machinery: 1991 names the formula (citing Fleiss 1986) and Part "
        "4 sizes its panel on the projections (**'six examiners would "
        "produce a pooled panel reliability of 0.9'**)'. The source is "
        "**Asher-McDade 1992 Part 4** (the operational panel), with "
        "Asher-McDade 1991 as the instrument"
    ),
    "does_our_mean_r_reproduce_it": (
        "**NO, and the gap is exactly the agreement gap.** "
        "``spearman_brown(0.4696, 6) = 0.8416``, not 0.90. **Our panel "
        "of six would be 0.8416**"
    ),
    "and_the_formula_CHECKS_OUT_on_their_own_number": (
        "**their projection is exactly reproducible from their own "
        "banked single-judge figure.** phase12 banks '1991 single-judge "
        "ICCs 0.43-0.60 (components), **0.60 (total)**', and "
        "``spearman_brown(0.60, 6) = 0.9000`` -- exact. Inverting: "
        "**r = 0.60 is precisely the single-rater agreement that yields "
        "0.90 at k = 6**. So the arithmetic is confirmed against an "
        "independent application of it, and the discrepancy is entirely "
        "in the INPUT, not in the formula"
    ),
    "what_that_means_for_us": (
        "**the lineage's panel agreed better than ours** -- 0.60 against "
        "our 0.4696, which phase12 already records as 'a measured "
        "lineage departure that PLAUSIBLY contributes to our lower "
        "agreement': theirs was five calibrated examiners with a "
        "pre-rating familiarization task, ours 'five mixed disciplines "
        "(incl. a cleft patient and a psychologist), no recorded "
        "calibration'. **The projection differs because the panel does**"
    ),
    "IT_PARTIALLY_PRE_EMPTS_READING_B_AND_THAT_IS_SAID_HERE": (
        "**stated rather than discovered later.** 0.8416 at k = 6 is "
        "below 0.90, so this check already indicates that reading (b) -- "
        "'0.90 needs many raters' -- is the one likely to fire. **It "
        "does not settle it**: the inverse at 0.90, 0.85 and 0.95 is "
        "still uncomputed, and 'many' is not a number. The readings are "
        "committed below in full knowledge of this one point, and are "
        "therefore **not blind**, which is worth saying plainly"
    ),
    "tag": "[LITERATURE] + [MEASURED] -- one arithmetic check, not a reading",
}


# --------------------------------------------------------------------------
# what the measurement is not
# --------------------------------------------------------------------------

#: **[DECLARED LIMITATIONS 2026-09-02] Spearman-Brown's assumptions, and
#: this panel's departures from them. NOT footnotes.**
THE_ASSUMPTIONS_DECLARED = {
    "declared": "2026-09-02, before any number",
    "assumption_1_exchangeable_raters_of_equal_quality": (
        "**the formula treats the k raters as interchangeable draws of "
        "equal quality.** Ours are not: they are five DIFFERENT "
        "PROFESSIONS -- a cleft patient, an orthodontist, a speech and "
        "language therapist, a plastic surgeon and a psychologist "
        "(``scoresheet.RATERS``) -- and the record has measured them "
        "unequal on two instruments. **Item-total: the SLT is highest at "
        "0.654 against the orthodontist's 0.628** "
        "(``reliability.item_total``). **Trained-model: -0.0098 to "
        "0.2703 across the five** (phase10.RATER_SCREEN_OBSERVED), a "
        "span of 0.28. **A sixth rater in the projection is an average "
        "rater who does not exist**"
    ),
    "assumption_2_errors_uncorrelated_across_raters": (
        "**the formula assumes each rater's error is independent of the "
        "others'.** Shared training, shared conventions and a shared "
        "instrument all correlate errors, and correlated error makes "
        "added raters buy LESS than the formula says. **The direction of "
        "this violation is known and it is optimistic**: the curve is an "
        "UPPER bound on what more raters would deliver, not a central "
        "estimate. Stated because a projection whose bias direction is "
        "known and unstated is worse than one with no projection"
    ),
    "assumption_3_the_curve_extrapolates_beyond_what_exists": (
        "k > 5 describes panels never observed on this cohort; k < 5 "
        "describes subsets never separately validated. **Only k = 5 is a "
        "measurement**; every other point is the formula speaking"
    ),
    "the_item_total_SPAN_is_not_in_the_record": (
        "**[FLAGGED, NOT BANKED.] The restate message gives the "
        "item-total correlations as spanning 0.554 to 0.677. THAT RANGE "
        "IS NOT IN THIS REPOSITORY** -- ``grep`` finds neither figure "
        "anywhere in ``src/``. What the record holds is two values, both "
        "from ``reliability.item_total``'s docstring: **SLT 0.654 and "
        "orthodontist 0.628**. The full five-value vector is computed at "
        "run time by ``item_total`` and lands in the p1 build's own "
        "output, so the span is very likely genuine and read from a run "
        "-- **but it cannot be quoted from the record, and this "
        "limitation is written from the two banked figures instead.** "
        "If the maintainer supplies the run's per-rater block, the span is "
        "bankable and this note gets a dated correction"
    ),
    "these_are_limitations_not_disqualifications": (
        "**the projection is still the best available answer to the "
        "question**, and every psychometric panel projection in the "
        "literature -- including the lineage's own 0.9 at six examiners "
        "-- makes exactly these assumptions. Recorded so the write-up "
        "states them beside the number rather than being asked for them"
    ),
}


#: **[PROHIBITION 2026-09-02] A CEILING IS NOT A SCORE.**
#:
#: The literal below is TESTED, sibling to the record's existing
#: prohibitions.
A_CEILING_IS_NOT_A_SCORE = (
    "A HIGHER CEILING IS NOT A HIGHER SCORE. The reliability curve says "
    "what a model COULD IN PRINCIPLE reach against a k-rater mean; it "
    "says NOTHING about what this or any model WOULD reach. Quoting the "
    "curve as a performance projection -- 'with ten raters the arm would "
    "score X' -- is FORBIDDEN. The project's own evidence is that the "
    "gap to the ceiling, not the ceiling, is the binding constraint: the "
    "best arm sits at 0.2520 against a ceiling of 0.9032, so raising the "
    "ceiling to 0.95 would move nothing that has been measured."
)

#: The prohibition's siblings, so the family is visible in one place.
PROHIBITION_SIBLINGS = (
    "reliability vs ceiling -- 'Reliability and a correlation ceiling "
    "are different quantities, and confusing them has already cost this "
    "project once' (data/reliability.py)",
    "the lineage's projected PANEL RELIABILITY 0.90/0.902 is NOT our "
    "correlation ceiling 0.9032 -- 'different quantities, coincidentally "
    "adjacent, never placed in proximity unqualified' "
    "(data/reliability.py:44)",
    "237 vs 251 -- 'comparing a run against the 251 figures looks like a "
    "module defect and is a population mismatch'",
)


# --------------------------------------------------------------------------
# the readings, before the numbers
# --------------------------------------------------------------------------

#: **[COMMITTED 2026-09-02, BEFORE THE CURVE EXISTS] The readings.**
#:
#: Both outcomes are written out, and **both are useful** -- which is
#: what makes this worth registering rather than just running.
READINGS_COMMITTED = {
    "committed": "2026-09-02, before the curve is computed",
    "a_a_modest_k_reaches_0_90": (
        "**the write-up gains a concrete, COSTED recommendation for "
        "follow-up data collection.** 'This cohort would need N raters "
        "to reach reliability 0.90' is an actionable sentence a clinical "
        "study can act on, and the project currently has none"
    ),
    # **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- 1-vs-2 scores as
    # identically wrong to 1-vs-5. Distance-aware beside it: QWK 0.4276,
    # mean inter-rater r 0.4696. record_audit.THE_KAPPA_LIMITATION.
    "b_0_90_needs_many_raters": (
        "**that is itself the finding, and arguably the better one.** "
        "The panel's low agreement means reliability is EXPENSIVE, and "
        "the write-up says so with a number instead of an adjective. "
        "Fleiss 0.1662 has been the project's stated research problem "
        "throughout; this converts it into a cost"
    ),
    "the_two_are_not_ranked_in_advance": (
        "**neither outcome is the hoped-for one.** Both produce a "
        "write-up sentence the project does not currently have, which is "
        "why the measurement is worth making either way"
    ),
    "what_NEITHER_outcome_changes": (
        "**no banked figure moves.** Not 0.2520, not the 0.9032 ceiling, "
        "not a ledger row, not a Phase 18 figure. The curve is a "
        "projection from MEAN_R_237, which is itself unchanged; it "
        "reinterprets nothing and re-derives nothing"
    ),
    "descriptive_no_ledger_row": (
        "**DESCRIPTIVE. No ledger row.** The ledger holds claims about "
        "arms; this is a property of the panel. The ledger stays at 38 "
        "entries"
    ),
    "the_readings_are_NOT_blind": (
        "**and that is recorded rather than glossed** -- "
        "THE_LINEAGE_CHECK computed one point of the curve at "
        "registration (0.8416 at k = 6), which indicates reading (b). "
        "**The readings are therefore NOT blind.** They are committed "
        "anyway because the inverse is what "
        "either reading actually needs, and 'many' still has to become a "
        "number"
    ),
}


# --------------------------------------------------------------------------
# criteria and settings
# --------------------------------------------------------------------------

#: **[DRAFT 2026-09-02 -- NOT LOCKED] Exit criteria.**
#:
#: Written for the maintainer to rule on, amend or replace. **Nothing here is
#: binding until it is locked**, on the precedent of every prior phase's
#: draft-then-lock.
EXIT_CRITERIA_DRAFT = {
    "status": "DRAFT, NOT LOCKED",
    "1_the_reckoning_leads_with_the_concessions": (
        "the phase states what it concedes BEFORE what it delivers, and "
        "names the record that covers each. Three designs conceded, one "
        "at its premise, with reasons that a reader can check against "
        "the cited records"
    ),
    "2_the_population_is_237_and_the_trap_is_named": (
        "every figure computes from MEAN_R_237 = 0.4696. If 251 is "
        "reported at all it is LABELLED, and the 237/251 trap is named "
        "beside it. **A curve computed from the wrong mean_r is the "
        "fourth instance of a trap already walked into three times**"
    ),
    "3_reliability_and_ceiling_are_never_merged": (
        "both are reported, always separately, always with sqrt named as "
        "the relationship. A single column carrying 'reliability' with "
        "ceiling values in it would be the R2 shape"
    ),
    "4_the_assumptions_travel_with_the_number": (
        "exchangeability, uncorrelated errors and the extrapolation "
        "beyond k = 5 are reported WITH the curve, not in a footnote, "
        "and the known optimistic bias direction is stated"
    ),
    "5_the_ceiling_is_not_quoted_as_a_score": (
        "A_CEILING_IS_NOT_A_SCORE is asserted as a literal and no output "
        "of this phase pairs a k with a predicted PCC for any arm"
    ),
    "6_the_lineage_check_is_reported_as_a_check": (
        "the 0.90-at-six comparison appears as an arithmetic check "
        "against an independent application of the formula, never as "
        "OUR projection, and the 0.60-vs-0.4696 input difference is "
        "stated as the reason for the gap"
    ),
    "7_the_inverse_is_reported_even_when_it_is_large": (
        "**the k required for 0.95 is reported however big it is**, "
        "including 'more than any feasible panel'. Truncating the table "
        "where it becomes unflattering would be choosing the range after "
        "seeing the numbers"
    ),
    "8_no_ledger_row_and_no_banked_figure_moves": (
        "the phase closes DESCRIPTIVE; the ledger stays at 38 entries "
        "and validate() is clean"
    ),
    "9_the_conceded_designs_stay_conceded_with_their_reopen_conditions": (
        "each concession carries the condition that would reopen it, so "
        "closing is visibly not forbidding "
        "(phase14's 'closing is not forbidding')"
    ),
}


#: **[TO DECLARE 2026-09-02] The settings this phase needs ruled.**
SETTINGS_TO_DECLARE = {
    "1_the_k_range": (
        "**proposed 1..20.** 1 is meaningful (it returns mean_r itself, "
        "the single-rater bar) and 20 is far past any feasible panel, "
        "which is what makes the flattening visible. **Open**: whether "
        "the upper bound is 20, or is set by where the ceiling passes a "
        "stated value, or is reported to 10 with the inverse carrying "
        "the tail"
    ),
    "2_the_reliability_targets_for_the_inverse": (
        "**proposed 0.85, 0.90, 0.95.** 0.90 is the lineage's own "
        "target, which is what makes it the comparable one; 0.85 and "
        "0.95 bracket it. **Open**: whether to add 0.80, and whether k "
        "is reported as the real-valued solution or as the ceiling "
        "integer (a panel cannot have 7.3 raters -- **proposed: report "
        "both, with the integer named as the actionable one**)"
    ),
    "3_one_population_or_two": (
        "**proposed 237 only, with 251 named as excluded and why.** "
        "Reporting both invites the comparison the record has three "
        "times had to correct. **Open**: the maintainer may prefer both "
        "columns WITH the trap named, on the precedent of "
        "``FLEISS_BY_POPULATION`` and ``LEARNABILITY_BY_POPULATION``, "
        "which do exactly that"
    ),
    "4_where_it_is_computed": (
        "**open, and it matters for the tier.** The curve needs no "
        "patient data -- only ``MEAN_R_237``, a module constant -- so it "
        "could be a pure record computation on the laptop rather than a "
        "cluster run. **Proposed: laptop, no run directory, no "
        "declared inputs**, since there are none to declare"
    ),
    "5_whether_the_curve_is_recomputed_from_the_matrix_or_the_constant": (
        "**open.** ``MEAN_R_237 = 0.4696`` is a banked constant AND "
        "recomputable from the grade matrix via "
        "``mean_inter_rater_r``. Recomputing needs the score sheet "
        "(cluster-only); using the constant needs nothing. **Proposed: "
        "the constant, with the recomputation named as the check that "
        "already exists in tests/test_reliability.py**"
    ),
}



#: **[RULED 2026-09-02] The five settings, all as proposed.**
THE_SETTINGS_RULED = {
    "ruled": "2026-09-02 -- all five as proposed",
    "1_k_range": (
        "**k = 1..20.** k = 1 returns ``mean_r`` itself (the single-rater "
        "bar) and 20 is past any feasible panel, which is what makes the "
        "flattening visible"
    ),
    "2_targets_and_how_k_is_reported": (
        "**0.85 / 0.90 / 0.95**, each reported as the **real-valued k** "
        "and the **integer k that FIRST meets or exceeds** the target, "
        "**with the integer named the actionable figure** -- a panel "
        "cannot have 6.4 raters. The integer is VERIFIED by the forward "
        "function rather than trusted from a ceiling operation: "
        "``spearman_brown(mean_r, k) >= target`` and "
        "``spearman_brown(mean_r, k - 1) < target``"
    ),
    "3_population": (
        "**237 only**, with the **237-versus-251 trap NAMED in the "
        "record rather than assumed known**. MEAN_R_251 = 0.4560 is not "
        "used and its exclusion is stated in the output, not merely "
        "implied by absence"
    ),
    "4_where_it_runs": (
        "**the laptop. NO RUN DIRECTORY.** It reads no patient data and "
        "needs none -- no manifest, no score sheet, no embeddings, no "
        "declared inputs, no hashes. There is nothing to declare, so "
        "there is no artifact to produce"
    ),
    "5_mean_r_is_the_banked_constant": (
        "**``reliability.MEAN_R_237 = 0.4696``, NOT recomputed.** "
        "Recomputing would require the cluster-only score sheet **for no "
        "gain**: the constant is already the verified 237-population "
        "figure, and ``tests/test_reliability.py`` already asserts that "
        "it reproduces RELIABILITY_237 through the same function"
    ),
}


#: **[LOCKED 2026-09-02] EXIT_CRITERIA.**
#:
#: The nine drafted criteria, unchanged from EXIT_CRITERIA_DRAFT except
#: that the settings they refer to are now ruled, plus the
#: nothing-added-after clause.
EXIT_CRITERIA = {
    "locked": "2026-09-02 -- nine criteria, five ruled settings",
    "1_the_reckoning_leads_with_the_concessions": (
        "the phase states what it CONCEDES before what it delivers, and "
        "names the record that covers each. Three designs conceded, one "
        "at its premise, with reasons a reader can check against the "
        "cited records"
    ),
    "2_the_population_is_237_and_the_trap_is_named": (
        "every figure computes from **MEAN_R_237 = 0.4696**. The "
        "237/251 trap is NAMED in the output, not assumed known. **A "
        "curve computed from the wrong mean_r would be the fourth "
        "instance of a trap already walked into three times**"
    ),
    "3_reliability_and_ceiling_are_never_merged": (
        "both reported, always separately, always with sqrt named as the "
        "relationship. A single column carrying 'reliability' with "
        "ceiling values in it would be the R2 shape"
    ),
    "4_the_assumptions_travel_with_the_number": (
        "exchangeability, uncorrelated errors and the extrapolation "
        "beyond k = 5 are reported **WITH the curve in the output**, not "
        "in a footnote and not only in the record, and **the known "
        "optimistic bias direction is stated**"
    ),
    "5_the_ceiling_is_not_quoted_as_a_score": (
        "``A_CEILING_IS_NOT_A_SCORE`` is asserted as a tested literal "
        "**and printed in the output**; no product of this phase pairs a "
        "k with a predicted PCC for any arm"
    ),
    "6_the_lineage_check_is_reported_as_a_check": (
        "the 0.90-at-six comparison appears **in the output** as an "
        "arithmetic check against an independent application of the "
        "formula, never as OUR projection, with the 0.60-versus-0.4696 "
        "input difference stated as the reason for the gap"
    ),
    "7_the_inverse_is_reported_however_large_it_is": (
        "**the k required for 0.95 is reported whatever it is**, "
        "including a k beyond the 1..20 table. **Truncating the table "
        "where it turns unflattering would be choosing the range after "
        "seeing the numbers**"
    ),
    "8_no_ledger_row_and_no_banked_figure_moves": (
        "the phase is DESCRIPTIVE; the ledger stays at 38 entries and "
        "validate() is clean"
    ),
    "9_the_conceded_designs_stay_conceded_with_their_reopen_conditions": (
        "each concession carries the condition that would reopen it, so "
        "closing is visibly not forbidding (phase14: 'closing is not "
        "forbidding')"
    ),
    "settings": THE_SETTINGS_RULED,
    "nothing_added_after": (
        "**NOTHING IS ADDED ONCE NUMBERS EXIST.** Not a k outside "
        "1..20 in the curve, not a fourth reliability target, not a "
        "second population, not a variant of the inverse. The k = 22 "
        "the 0.95 target requires is REPORTED as an inverse result and "
        "**does not extend the table** -- the table's range was ruled "
        "before the number existed and stays where it was ruled"
    ),
    "status": "LOCKED",
}


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

#: The ruled k range and reliability targets. Constants, so the output
#: cannot quietly disagree with EXIT_CRITERIA about its own scope.
K_RANGE: tuple[int, ...] = tuple(range(1, 21))
RELIABILITY_TARGETS: tuple[float, ...] = (0.85, 0.90, 0.95)

#: **[LITERATURE]** Asher-McDade 1991's single-judge total ICC, banked at
#: ``phase12.PRIMARY_SOURCES_BANKED['reliability_anchors']``: "1991
#: single-judge ICCs 0.43-0.60 (components), 0.60 (total)".
LINEAGE_SINGLE_JUDGE_R = 0.60

#: Part 4's own projection, the figure the check reproduces.
LINEAGE_PROJECTED_RELIABILITY = 0.90
LINEAGE_PROJECTED_K = 6


def curve(mean_r: float, k_values=K_RANGE) -> list[dict]:
    """Reliability and the implied ceiling at each k.

    **Reuses ``reliability.spearman_brown`` and ``reliability.pcc_ceiling``
    and implements neither.** A second implementation of a formula the
    record already banks would be a quantity under one name computed two
    ways, which is the shape this project refuses.

    Reliability and ceiling are returned as SEPARATE keys because they
    are different quantities -- ``data/reliability`` opens by saying so.
    """
    from .data import reliability as R

    rows = []
    for k in k_values:
        value = R.spearman_brown(mean_r, int(k))
        rows.append({
            "k": int(k),
            "reliability": value,
            "pcc_ceiling": R.pcc_ceiling(value),
        })
    return rows


def raters_required(mean_r: float, target: float) -> dict:
    """The inverse: how many raters reach ``target`` reliability.

    Solving ``rho = k r / (1 + (k-1) r)`` for k gives
    ``k = rho (1 - r) / (r (1 - rho))``. **The integer is then VERIFIED
    through the forward function** -- ``spearman_brown`` at k meets the
    target and at k-1 does not -- rather than trusted from a ceiling
    operation, so a float edge case cannot produce an integer that does
    not actually reach the target.
    """
    import math

    from .data import reliability as R

    if not 0.0 < target < 1.0:
        raise ValueError(
            f"a reliability target must lie in (0, 1); got {target}"
        )
    if not 0.0 < mean_r < 1.0:
        raise ValueError(
            f"the mean inter-rater r must lie in (0, 1); got {mean_r}"
        )
    exact = target * (1.0 - mean_r) / (mean_r * (1.0 - target))
    integer = max(1, math.ceil(exact))
    while R.spearman_brown(mean_r, integer) < target:
        integer += 1                      # never taken; the guard is the point
    if integer > 1 and R.spearman_brown(mean_r, integer - 1) >= target:
        raise ValueError(
            f"k={integer} is not the FIRST k reaching {target}: k-1 "
            "already does. The inverse and the forward function "
            "disagree, which must not happen."
        )
    return {
        "target": float(target),
        "k_exact": float(exact),
        "k_integer": int(integer),
        "reliability_at_k_integer": R.spearman_brown(mean_r, integer),
        "pcc_ceiling_at_k_integer": R.pcc_ceiling(
            R.spearman_brown(mean_r, integer)
        ),
        "within_the_ruled_table": bool(integer <= max(K_RANGE)),
    }


def lineage_check() -> dict:
    """Asher-McDade Part 4's projection, reproduced and compared.

    **A check on the arithmetic against an independent application of
    the same formula**, not one of this phase's readings.
    """
    from .data import reliability as R

    ours = R.spearman_brown(R.MEAN_R_237, LINEAGE_PROJECTED_K)
    theirs = R.spearman_brown(LINEAGE_SINGLE_JUDGE_R, LINEAGE_PROJECTED_K)
    implied = (
        LINEAGE_PROJECTED_RELIABILITY
        / (LINEAGE_PROJECTED_K
           - (LINEAGE_PROJECTED_K - 1) * LINEAGE_PROJECTED_RELIABILITY)
    )
    return {
        "k": LINEAGE_PROJECTED_K,
        "their_single_judge_r": LINEAGE_SINGLE_JUDGE_R,
        "their_projection_reproduced": theirs,
        "their_published_projection": LINEAGE_PROJECTED_RELIABILITY,
        "reproduces_exactly": bool(
            abs(theirs - LINEAGE_PROJECTED_RELIABILITY) < 1e-12
        ),
        "our_mean_r": R.MEAN_R_237,
        "ours_at_the_same_k": ours,
        "r_implied_by_their_projection": implied,
        "conclusion": (
            "the formula is CONFIRMED against an independent application "
            "of it, and the whole discrepancy is in the INPUT: their "
            "calibrated examiners against our uncalibrated mixed panel"
        ),
    }


def report() -> dict:
    """Everything this phase measures, as one object. **237 only.**"""
    from .data import reliability as R

    return {
        "population": 237,
        "mean_r": R.MEAN_R_237,
        "mean_r_source": "data.reliability.MEAN_R_237 -- banked, not recomputed",
        "excluded_population": {
            "mean_r_251": R.MEAN_R_251,
            "why": (
                "251 is the all-scored-rows population; 237 is the "
                "patients with photographs and the cohort the manifest "
                "holds. Named rather than silently absent"
            ),
        },
        "curve": curve(R.MEAN_R_237),
        "inverse": [
            raters_required(R.MEAN_R_237, target)
            for target in RELIABILITY_TARGETS
        ],
        "lineage_check": lineage_check(),
        "assumptions": THE_ASSUMPTIONS_DECLARED,
        "prohibition": A_CEILING_IS_NOT_A_SCORE,
        "readings_were_not_blind": (
            "the lineage check produced 0.8416 at k = 6 BEFORE the "
            "readings fired, which already leaned toward the "
            "reliability-is-expensive outcome"
        ),
    }


def render(data: dict | None = None) -> str:
    """The report as text. **Carries the bindings, not only the numbers.**"""
    from .data import reliability as R

    data = report() if data is None else data
    out = [
        "PHASE 24 -- THE RELIABILITY CURVE AND ITS INVERSE",
        "",
        f"population 237 (photographed patients); mean_r = {data['mean_r']}",
        "  BANKED CONSTANT data.reliability.MEAN_R_237, not recomputed.",
        f"  NOT the 251-row population (mean_r "
        f"{data['excluded_population']['mean_r_251']}): comparing a value "
        "computed",
        "  on one population against one computed on another is how the "
        "0.903 ceiling",
        "  once came to be 'corrected' to 0.807. Named, not assumed known.",
        "",
        "  k   reliability   pcc_ceiling = sqrt(reliability)",
    ]
    for row in data["curve"]:
        mark = "   <- the panel we have" if row["k"] == 5 else ""
        out.append(
            f"  {row['k']:>2}      {row['reliability']:.4f}        "
            f"{row['pcc_ceiling']:.4f}{mark}"
        )
    out += [
        "",
        "  RELIABILITY and CEILING are DIFFERENT QUANTITIES; the ceiling "
        "is the square",
        "  root of the reliability and is never reported as the same "
        "column.",
        "",
        "THE INVERSE -- how many raters reach a target reliability",
    ]
    for row in data["inverse"]:
        beyond = "" if row["within_the_ruled_table"] else             "  [BEYOND the ruled k=1..20 table, reported anyway]"
        out.append(
            f"  {row['target']:.2f}:  k = {row['k_exact']:.4f} exact  ->  "
            f"**{row['k_integer']} raters** (ACTIONABLE), reaching "
            f"{row['reliability_at_k_integer']:.4f}{beyond}"
        )
    check = data["lineage_check"]
    out += [
        "",
        "THE LINEAGE CHECK [LITERATURE] -- Asher-McDade 1992 Part 4",
        "  Part 4 sizes its panel on this same formula: 'six examiners "
        "would produce",
        "  a pooled panel reliability of 0.9'.",
        f"  spearman_brown({check['their_single_judge_r']}, "
        f"{check['k']}) = {check['their_projection_reproduced']:.4f}  "
        f"-- reproduces their {check['their_published_projection']} "
        "EXACTLY",
        f"  spearman_brown({check['our_mean_r']}, {check['k']}) = "
        f"{check['ours_at_the_same_k']:.4f}  -- ours at the same k",
        f"  r implied by their projection: "
        f"{check['r_implied_by_their_projection']:.4f}, which is their "
        "banked single-judge total",
        f"  CONCLUSION: {check['conclusion']}",
        "",
        "THE ASSUMPTIONS, AND THE DIRECTION OF THEIR BIAS",
        "  (1) EXCHANGEABLE RATERS OF EQUAL QUALITY. Ours are five "
        "different",
        "      professions, measured unequal on two instruments: "
        "item-total 0.654",
        "      (SLT) against 0.628 (orthodontist), and trained-model "
        "-0.0098 to",
        "      0.2703. A sixth rater in this projection is an average "
        "rater who",
        "      does not exist.",
        "  (2) ERRORS UNCORRELATED ACROSS RATERS. Shared training and a "
        "shared",
        "      instrument correlate errors, and correlated error makes "
        "added raters",
        "      buy LESS than the formula says. **THE CURVE IS AN UPPER "
        "BOUND**, not a",
        "      central estimate. The bias direction is known, so it is "
        "stated.",
        "  (3) Only k = 5 is a MEASUREMENT. Every other point is the "
        "formula speaking.",
        "",
        "THE PROHIBITION",
        f"  {data['prohibition']}",
        "",
        "THE READINGS WERE NOT BLIND",
        f"  {data['readings_were_not_blind']}",
    ]
    assert R.MEAN_R_237 == data["mean_r"]
    return "\n".join(out)



#: **[MEASURED 2026-09-02, ON THE LAPTOP -- no run directory] THE CURVE,
#: THE INVERSE, AND WHICH READING FIRED.**
#:
#: Computed by ``report()`` from ``MEAN_R_237 = 0.4696`` alone. No
#: patient data, no manifest, no score sheet, no declared inputs.
THE_CURVE_OBSERVED = {
    "measured": "2026-09-02, laptop, no run directory",
    "the_shape_of_the_curve": (
        "**0.4696 at k=1, 0.8157 at k=5, 0.8985 at k=10, 0.9465 at "
        "k=20.** The ceiling runs 0.6853 -> 0.9032 -> 0.9479 -> 0.9729. "
        "**It flattens hard**: the first extra rater buys +0.1695 "
        "reliability, the sixth buys +0.0259, the twentieth buys +0.0026 "
        "-- a factor of 65 between the first increment and the last"
    ),
    "the_inverse": (
        "**0.85 -> k = 6.4003 exact, 7 raters actionable** (reaching "
        "0.8611); **0.90 -> k = 10.1652 exact, 11 raters** (0.9069); "
        "**0.95 -> k = 21.4600 exact, 22 raters** (0.9512). Each integer "
        "was VERIFIED through the forward function, not trusted from a "
        "ceiling operation"
    ),
    "the_0_95_target_is_OUTSIDE_the_ruled_table_and_is_reported_anyway": (
        "**22 raters is beyond the ruled k = 1..20**, and criterion 7 "
        "exists for exactly this: 'the k required for 0.95 is reported "
        "whatever it is ... truncating the table where it turns "
        "unflattering would be choosing the range after seeing the "
        "numbers'. **The table is NOT extended to 22** -- its range was "
        "ruled before the number existed and stays where it was ruled. "
        "The inverse is a separate result and carries the value"
    ),

    # ---- the readings -------------------------------------------------
    "which_reading_fired": (
        "**(b), at the target the readings were written against.** "
        "READINGS_COMMITTED framed both outcomes around 0.90, and "
        "**0.90 needs 11 raters -- more than DOUBLE the panel this "
        "project has**. The finding is the registered one: the panel's "
        "low agreement means **reliability is expensive, and the "
        "write-up can now say so with a number instead of an adjective**"
    ),
    "and_the_honest_nuance_the_readings_did_not_anticipate": (
        "**the outcome is TARGET-DEPENDENT, and reading (a) fires at "
        "0.85.** Seven raters for 0.85 is a modest increment -- two more "
        "than we have -- and is exactly the 'concrete, costed "
        "recommendation' reading (a) described. **Both readings are "
        "true, at different targets**, and saying only (b) would "
        "overstate the cost while saying only (a) would understate it. "
        "Recorded because the readings were written as if the answer "
        "were single-valued and it is not"
    ),
    "the_readings_were_NOT_blind": (
        "**restated here where the numbers are.** THE_LINEAGE_CHECK "
        "computed 0.8416 at k = 6 at registration, before the readings "
        "fired, and that already leaned toward (b). The readings are "
        "reported as informed, not blind"
    ),

    # ---- two checks that came free ------------------------------------
    "the_curve_REPRODUCES_the_two_banked_constants_at_k_5": (
        "**a free check, and it passed.** At k = 5 the curve gives "
        "reliability **0.8157** against the banked "
        "``RELIABILITY_237 = 0.8158`` (the 0.0001 is the banked figure's "
        "own rounding, already recorded at "
        "phase21.COHORT_PAIR_SEPARATION_DESIGNED: 'recomputes to 0.8157 "
        "from MEAN_R_237') and ceiling **0.9032** against "
        "``PCC_CEILING_237 = 0.9032`` **exactly**. The curve is anchored "
        "to the record at the one k that is a measurement"
    ),
    "A_NEW_SAME_DIGIT_COLLISION_FOUND_IN_THIS_PHASES_OWN_OUTPUT": (
        "**reliability at k = 10 on the 237 panel is 0.8985, and "
        "``PCC_CEILING_251`` is 0.8985.** Identical to four decimals and "
        "**different quantities on BOTH axes this phase's criteria "
        "warn about**: one is a projected ten-rater RELIABILITY on the "
        "237 population, the other is a correlation CEILING on the 251 "
        "population. Criterion 2 (237 vs 251) and criterion 3 "
        "(reliability vs ceiling) intersect on this one number. **The "
        "fourth instance of the same-digit family** (0.628 item-total vs "
        "sd_obs; 0.90/0.902 panel reliability vs 0.9032 ceiling; and "
        "this). Recorded so a reader who meets 0.8985 in two places "
        "knows they are two things"
    ),

    "no_banked_figure_moved": (
        "**none.** Not 0.2520, not 0.9032, not a Phase 18 figure, not a "
        "ledger row. The ledger stands at 38 entries and validate() is "
        "clean. The curve is a projection FROM a banked constant and "
        "changes nothing it was projected from"
    ),
    "tag": "[MEASURED] -- DESCRIPTIVE, no ledger row",
}


def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "reckoning": PHASE_24_RECKONING,
        "design_1_label_distribution": (
            THE_LABEL_DISTRIBUTION_HEAD_CONCEDED_COVERED
        ),
        "design_2_per_rater_heads": THE_PER_RATER_HEADS_CONCEDED_MEASURED,
        "design_3_agreement_weighted": (
            AGREEMENT_WEIGHTED_TRAINING_CONCEDED_AT_ITS_PREMISE
        ),
        "design_4_disagreement_auxiliary": (
            DISAGREEMENT_AS_AUXILIARY_TARGET_STATUS
        ),
        "substituted_deliverable": THE_SUBSTITUTED_DELIVERABLE,
        "lineage_check": THE_LINEAGE_CHECK,
        "assumptions": THE_ASSUMPTIONS_DECLARED,
        "ceiling_is_not_a_score": A_CEILING_IS_NOT_A_SCORE,
        "prohibition_siblings": PROHIBITION_SIBLINGS,
        "readings": READINGS_COMMITTED,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        "settings_to_declare": SETTINGS_TO_DECLARE,
        "settings_ruled": THE_SETTINGS_RULED,
        "exit_criteria": EXIT_CRITERIA,
        "curve_observed": THE_CURVE_OBSERVED,
    }
