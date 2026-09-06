"""Phase 22: ordering-based training, restated after the measurement refuted it.

Opened 2026-09-01. **This module is the RESTATE and nothing else** -- no
task, no schema kind, no config, no run. The three rulings the maintainer
issued before it was written are recorded at their arms.

The phase is opened in a weaker form than it was proposed in, and the
reason is a measurement rather than an argument. That belongs at the top
of the module, not in a limitations paragraph at the bottom.
"""

from __future__ import annotations


#: **[RECKONED 2026-09-01, AFTER THE REFUTATION, EVERY FIGURE VERIFIED AT
#: SOURCE] The opening reckoning.**
#:
#: **THE HONEST PREMISE, FIRST: this phase was proposed on a mechanism
#: that has since been measured and ruled out, and it is being restated
#: in a weaker form because of that.** Not narrowed for tidiness, not
#: refined -- refuted, and the refutation is the reason the phase looks
#: the way it does.
#:
#: **What the record now holds**, checked rather than recalled:
#:
#: * ``phase21.CROSS_ARM_SHRINKAGE_OBSERVED`` -- **63 of 64 arms shrink**
#:   toward the cohort mean. Median **0.3289**, min **0.0656** (p16
#:   identity), probe **0.4947**. Shrinkage is not a quirk of one arm; it
#:   is what almost every model on this cohort does.
#: * ``phase21.POST_HOC_ANALYSES`` -- error **direction** tracks signed
#:   distance from the cohort mean at tau_b **-0.8470**; **worst-quartile**
#:   error tracks absolute distance at tau_b **+0.7521**. The errors are
#:   organised by distance from the centre, in both senses.
#: * ``phase21.ARM_A_THE_SHRINKAGE_CONTROL`` -- arm A **expands**, at
#:   **2.1730**, and agrees with the other 63 at **chance**: mean kappa
#:   **+0.0956** (residual sign) and **-0.0743** (worst quartile) against
#:   all-pairs **+0.7012** and **+0.5729**.
#:
#: **WHAT THIS FORBIDS.**
#:
#: ``phase21.SCALE_INVARIANCE_PROHIBITION`` rules out the mechanism this
#: phase was proposed on. PCC is invariant to affine rescaling of the
#: prediction vector, so shrinkage **cannot explain the PCC ceiling** and
#: removing shrinkage cannot raise it. That is arithmetic, not evidence,
#: and it was already binding.
#:
#: **What is new is that the record now holds a measured case as well.**
#: Arm A avoids shrinkage entirely -- it expands -- and scores PCC
#: **0.2334** against the probe's **0.2520**. A delta of -0.0186, inside
#: the unresolvable band, indistinguishable. **NOT SHRINKING BUYS NOTHING
#: ON PCC**, and that is now demonstrated by a model that did it rather
#: than deduced from an invariance. The prohibition and the measurement
#: agree, by different routes, which is the strongest form this record
#: can hold a negative in.
#:
#: **THE HYPOTHESIS THAT SURVIVES** is the narrower one: **ordering-based
#: training may find DIFFERENT FEATURES.** Not "it removes shrinkage and
#: therefore scores better" -- that path is closed at both ends. The
#: claim is only that an objective which never sees an absolute target
#: may learn a different representation, and that a different
#: representation is worth measuring whether or not it scores better.
#:
#: Arm A is what makes this a real distinction rather than a retreat:
#: it separates **avoiding the mean** from **finding different features**
#: by doing the first and gaining nothing. Whatever a ranking arm would
#: have to do to earn a claim here, arm A shows it is not simply
#: declining to regress to the centre.
#:
#: **AND THE WEAKER HYPOTHESIS IS THE HARDER ONE TO TEST.** The stronger
#: mechanism at least predicted a PCC gain, which this cohort can measure
#: badly. The surviving one predicts a representational difference, for
#: which no scored quantity existed until Phase 21 built one. That is why
#: the diagnostic in ``FEATURE_DIFFERENCE_DIAGNOSTIC`` is not a
#: supporting analysis but the phase's actual instrument.
PHASE_22_RECKONING = {
    "reckoned": "2026-09-01, at the restate, after the refutation",
    "the_honest_premise": (
        "**THIS PHASE WAS PROPOSED ON A MECHANISM THAT HAS SINCE BEEN "
        "MEASURED AND RULED OUT.** It is restated in a weaker form for "
        "that reason and no other. Stated first because a reckoning "
        "that opens with what survives and mentions the refutation "
        "later is the same document with the load-bearing sentence "
        "moved somewhere it will not be read."
    ),
    "what_the_record_now_holds": (
        "63 of 64 arms shrink toward the cohort mean, median 0.3289, "
        "min 0.0656 (p16 identity), probe 0.4947 "
        "(phase21.CROSS_ARM_SHRINKAGE_OBSERVED); error DIRECTION tracks "
        "signed distance from the mean at tau_b -0.8470 and "
        "WORST-QUARTILE error tracks absolute distance at tau_b +0.7521 "
        "(phase21.POST_HOC_ANALYSES); arm A EXPANDS at 2.1730 and "
        "agrees with the other 63 at chance, +0.0956 / -0.0743 against "
        "all-pairs +0.7012 / +0.5729 "
        "(phase21.ARM_A_THE_SHRINKAGE_CONTROL)"
    ),
    "what_it_forbids": (
        "**SCALE_INVARIANCE_PROHIBITION rules out the shrinkage-removal "
        "mechanism.** PCC is invariant to affine rescaling of the "
        "prediction vector, so shrinkage CANNOT EXPLAIN THE PCC CEILING "
        "and removing it cannot raise PCC. Arithmetic, not evidence, "
        "and already binding before this phase reopened."
    ),
    "and_now_a_measured_case_as_well": (
        "**Arm A avoids shrinkage entirely -- it EXPANDS at 2.1730 -- "
        "and scores PCC 0.2334 against the probe's 0.2520.** Delta "
        "-0.0186, inside the 0.04-0.10 unresolvable band, "
        "indistinguishable. **NOT SHRINKING BUYS NOTHING ON PCC**, now "
        "DEMONSTRATED BY A MODEL THAT DID IT rather than deduced from "
        "an invariance. The prohibition and the measurement agree by "
        "different routes, which is the strongest form this record can "
        "hold a negative in."
    ),
    "the_hypothesis_that_survives": (
        "**the NARROWER one: ordering-based training may find DIFFERENT "
        "FEATURES.** Not 'it removes shrinkage and therefore scores "
        "better' -- that path is closed at both ends. Only that an "
        "objective which never sees an absolute target may learn a "
        "different representation, and that a different representation "
        "is worth measuring whether or not it scores better."
    ),
    "arm_a_makes_the_distinction_real": (
        "it separates AVOIDING THE MEAN from FINDING DIFFERENT FEATURES "
        "by doing the first and gaining nothing. Whatever a ranking arm "
        "must do to earn a claim here, arm A shows it is not simply "
        "declining to regress to the centre. Without arm A the "
        "narrowed hypothesis would be a distinction drawn to keep the "
        "phase alive; with it, the two are measurably separable"
    ),
    "and_the_weaker_hypothesis_is_the_harder_one_to_test": (
        "**the stronger mechanism at least predicted a PCC gain, which "
        "this cohort measures badly. The surviving one predicts a "
        "REPRESENTATIONAL difference, for which no scored quantity "
        "existed until Phase 21 built one.** That is why "
        "FEATURE_DIFFERENCE_DIAGNOSTIC is not a supporting analysis "
        "but the phase's actual instrument. A phase whose hypothesis "
        "got weaker AND whose test got harder is worth saying out loud"
    ),
    "tag": "[MEASURED] for every figure; the reckoning itself [REASONED]",
}


#: **[RECORDED 2026-09-01] it was asserted the stronger mechanism before
#: the measurement refuted it.**
#:
#: In conversation during the Phase 21 close-out, this record previously stated that
#: shrinkage was a mechanism the ranking objective would address --
#: stated as an account of why Phase 22 was worth running, before any
#: cross-arm shrinkage number existed. The measurement that refuted it
#: was built afterwards, in the same sequence of turns.
#:
#: **The order matters and is recorded rather than smoothed.** This
#: reckoning is written **after** the refutation, not before it. It is
#: not a prediction that survived contact with data; it is a position
#: revised because data arrived. A reader who cannot tell those apart
#: cannot weigh the phase, and nothing in the module's structure
#: distinguishes them -- only this record does.
#:
#: **The same shape as the errors this project catalogues.** An account
#: that explains the observations, asserted with confidence, never
#: measured. It was not caught by a check; it was caught by building the
#: measurement it implied. ``phase21.SHRINKAGE_FIGURE_MISATTRIBUTED``
#: records the neighbouring error in the same episode -- a figure
#: attributed to a phase that never held it.
THE_STRONGER_MECHANISM_ASSERTED = {
    "recorded": "2026-09-01, in the restate, unprompted by any check",
    "what_was_asserted": (
        "during the Phase 21 close-out this record previously stated that SHRINKAGE was "
        "a mechanism the ranking objective would address -- offered as "
        "the account of why Phase 22 was worth running, **before any "
        "cross-arm shrinkage figure existed**"
    ),
    "what_refuted_it": (
        "SCALE_INVARIANCE_PROHIBITION (arithmetic: PCC is invariant to "
        "affine rescaling, so shrinkage cannot explain the ceiling) and "
        "then CROSS_ARM_SHRINKAGE_OBSERVED with "
        "ARM_A_THE_SHRINKAGE_CONTROL (a measured case: arm A avoids "
        "shrinkage and gains nothing). **Both arrived AFTER the "
        "assertion, in the same sequence of turns.**"
    ),
    "the_order_is_recorded_not_smoothed": (
        "**this reckoning is written AFTER the refutation, not before "
        "it.** It is NOT a prediction that survived contact with data; "
        "it is a position revised because data arrived. A reader who "
        "cannot tell those apart cannot weigh the phase, and nothing "
        "in the module's structure distinguishes them -- only this "
        "record does"
    ),
    "the_familiar_shape": (
        "an account that EXPLAINS THE OBSERVATIONS, asserted with "
        "confidence, never measured. Not caught by a check -- caught by "
        "building the measurement it implied. "
        "phase21.SHRINKAGE_FIGURE_MISATTRIBUTED records the "
        "neighbouring error in the same episode"
    ),
    "what_it_does_not_excuse": (
        "the restate is not owed leniency for being honest about its "
        "own history. The arms below are judged on what they measure, "
        "and a phase opened on a refuted mechanism has to earn its "
        "place on the surviving hypothesis alone"
    ),
    "tag": "[REASONED]",
}


#: **[AUDITED 2026-09-01, AT SOURCE] Nothing in the record tests an
#: ordering objective. There is nothing to concede.**
#:
#: The coverage question this project asks before opening any phase --
#: *does the record already answer it?* -- returns a clean negative here,
#: and the negative was checked at source rather than recalled:
#:
#: * ``models/factory.py`` states the objective in its own docstring:
#:   ``num_outputs=1`` and **MSE on the raw 1-5 scale (PLAN §4.6). Not
#:   CORAL/CORN -- the tails are too sparse for an ordinal
#:   decomposition.** Every regression arm in the project fits an
#:   absolute target with a squared-error loss. None ranks.
#: * **Phase 17's contrastive loss is binary.** The
#:   ``siamese_contrastive`` kind's ``pair_rule`` has exactly one choice,
#:   ``symmetric_if_zero_magnitude`` -- same/different, not
#:   more/less. It **reads the magnitude ordering and discards it**: four
#:   magnitudes per face (0.0, 0.015, 0.025, 0.035) collapse to a
#:   two-valued relation. The arm closest to a ranking objective in the
#:   whole record is the one that throws the ordering away.
#: * **LDL is a distribution objective.** ``train/ldl.py``: a five-output
#:   softmax head and a **KL loss** against ``soft_1..soft_5``. It fits
#:   the shape of the rater distribution, not the order of patients.
#:
#: **So no concession is owed.** This is worth stating explicitly because
#: the honest form of a coverage audit is one that can come back
#: positive -- Phase 21's did, twice
#: (``phase7b.ENSEMBLE_COMBINES`` as a near-miss). Here it does not.
COVERAGE_NOTHING_TESTS_ORDERING = {
    "audited": "2026-09-01, read at source, not recalled",
    "the_regression_arms": (
        "**models/factory.py's own docstring**: num_outputs=1 and MSE "
        "ON THE RAW 1-5 SCALE (PLAN 4.6), and explicitly **'Not "
        "CORAL/CORN -- the tails are too sparse for an ordinal "
        "decomposition'**. Every regression arm fits an ABSOLUTE target "
        "with a SQUARED-ERROR loss. None ranks. The ordinal option was "
        "considered and declined on a stated ground"
    ),
    "phase_17s_contrastive_loss": (
        "**BINARY, and it DISCARDS THE ORDERING IT READS.** The "
        "siamese_contrastive kind's pair_rule has exactly ONE choice, "
        "symmetric_if_zero_magnitude -- same/different, not more/less. "
        "Four magnitudes per face (0.0, 0.015, 0.025, 0.035) collapse "
        "to a two-valued relation. **The arm closest to a ranking "
        "objective in the entire record is the one that throws the "
        "ordering away.**"
    ),
    "ldl": (
        "**a DISTRIBUTION objective**: train/ldl.py fits soft_1..soft_5 "
        "with a five-output softmax head and a KL loss, reporting the "
        "expectation so PCC stays comparable. It fits the SHAPE of the "
        "rater distribution, not the ORDER of patients"
    ),
    "the_verdict": (
        "**NOTHING TO CONCEDE.** No arm in the record optimises an "
        "ordering. Stated explicitly because the honest form of a "
        "coverage audit is one that CAN come back positive -- Phase "
        "21's did, with phase7b.ENSEMBLE_COMBINES as a near-miss. Here "
        "it does not, and the difference is worth marking"
    ),
    "tag": "[MEASURED] -- each claim read out of the named file",
}


#: **[REGISTERED 2026-09-01] Arm R-syn: rank-pretrain
#: on the synthetic ordering.**
#:
#: The synthetic TPS set carries an ordering that is **exact by
#: construction** -- four magnitudes per face (0.0, 0.015, 0.025, 0.035),
#: and a larger warp is unambiguously a larger deformation. No panel, no
#: reliability ceiling, no noise floor. It is the only ordering in this
#: project that is known rather than estimated.
#:
#: **Its pairs are WITHIN-FACE.** The construction says *this face at
#: 0.035 is more deformed than the same face at 0.015*; it says nothing
#: about face X against face Y. That is the whole content of
#: ``PAIRING_AXIS_ASSUMPTION`` and the reason this arm is registered
#: separately from the cohort arms rather than pooled with them.
ARM_R_SYN_REGISTERED = {
    "registered": "2026-09-01, ruled at the restate",
    "what_it_is": (
        "rank-pretrain on the SYNTHETIC TPS ordering, then fine-tune on "
        "the cohort. The pretraining signal is a pairwise ranking loss "
        "over pairs whose true order is known"
    ),
    # **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- 1-vs-2 scores as
    # identically wrong to 1-vs-5. Distance-aware beside it: QWK 0.4276,
    # mean inter-rater r 0.4696. record_audit.THE_KAPPA_LIMITATION.
    "why_this_ordering": (
        "**EXACT BY CONSTRUCTION.** Four magnitudes per face (0.0, "
        "0.015, 0.025, 0.035) and a larger warp IS a larger "
        "deformation. No panel, no reliability ceiling, no noise floor. "
        "**The only ordering in this project that is KNOWN rather than "
        "ESTIMATED** -- every cohort ordering is a panel mean with "
        "Fleiss 0.1662 behind it"
    ),
    "its_pairs_are_within_face": (
        "the construction says *this face at 0.035 is more deformed "
        "than THE SAME FACE at 0.015*. **It says nothing about face X "
        "against face Y.** That is the whole content of "
        "PAIRING_AXIS_ASSUMPTION and the reason this arm is registered "
        "separately from the cohort arms rather than pooled with them"
    ),
    "n_pairs_available": (
        "within-face only: C(4,2) = 6 ordered pairs per face, times the "
        "synthesis set's face count. To be stated exactly from the "
        "synth index at build, NOT estimated here"
    ),
    "tag": "[REGISTERED]",
}


#: **[REGISTERED 2026-09-01] Arm R-all: pairwise
#: ranking on every cohort pair.**
#:
#: All **27,966** pairs of the 237 patients, ordered by panel mean --
#: **including the 24.85% that fall below SE_diff and are therefore
#: ordered by chance.** That inclusion is the point of the arm, not an
#: oversight in it.
ARM_R_ALL_REGISTERED = {
    "registered": "2026-09-01, ruled at the restate",
    "what_it_is": (
        "pairwise ranking on ALL 27,966 cohort pairs (C(237,2)), "
        "ordered by panel mean"
    ),
    "it_includes_the_noise_ordered_pairs_deliberately": (
        "**24.85% of pairs fall below SE_diff = 0.398942 and are "
        "ordered by chance** "
        "(phase21.COHORT_PAIR_SEPARATION_OBSERVED). This arm trains on "
        "them anyway. **That inclusion is the POINT of the arm, not an "
        "oversight in it** -- it is the naive construction anyone would "
        "write, and it is here so that its cost or benefit is measured "
        "instead of assumed"
    ),
    "the_pair_count_is_verified": (
        "237 * 236 / 2 = 27966, and the separation run counted the same "
        "27966 -- the two agree by identity, not by coincidence"
    ),
    "tag": "[REGISTERED]",
}


#: **[REGISTERED 2026-09-01] Arm R-clear: the same,
#: restricted to pairs that clear SE_diff.**
#:
#: Identical to R-all in every respect except the pair set: only pairs
#: whose panel-mean separation exceeds **SE_diff = 0.398942**, which is
#: **75.15%** of them.
#:
#: **THE RESTRICTION IS DECLARED IN ADVANCE, AND THAT IS WHAT MAKES IT
#: LEGAL.** ``phase21.COHORT_PAIR_SEPARATION_DESIGNED`` registered the
#: allowance before the separation number existed; the threshold is the
#: measurement's own SE_diff, not a value chosen once the pair
#: distribution was visible. **It is never a post-hoc filter.** A
#: restriction picked after seeing which pairs the model got wrong would
#: be selection on evaluation data
#: (``phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION``); this one is
#: fixed before any model is trained and derives from the panel's
#: reliability alone.
ARM_R_CLEAR_REGISTERED = {
    "registered": "2026-09-01, ruled at the restate",
    "what_it_is": (
        "pairwise ranking on cohort pairs RESTRICTED to those whose "
        "panel-mean separation exceeds SE_diff = 0.398942 -- 75.15% of "
        "the 27,966. Identical to R-all in every other respect"
    ),
    "declared_in_advance_never_post_hoc": (
        "**phase21.COHORT_PAIR_SEPARATION_DESIGNED registered the "
        "allowance BEFORE the separation number existed**, and the "
        "threshold is the measurement's OWN SE_diff -- not a value "
        "chosen once the pair distribution was visible. **NEVER A "
        "POST-HOC FILTER.** A restriction picked after seeing which "
        "pairs the model got wrong would be selection on evaluation "
        "data (phase21.SELECTION_ON_EVALUATION_DATA_PROHIBITION); this "
        "one is fixed before any model trains and derives from the "
        "panel's reliability alone"
    ),
    "se_diff_provenance": (
        "0.398942 = sd_obs * sqrt(2) * sqrt(1 - RELIABILITY_237), with "
        "sd_obs = 0.657279 measured on the cohort. Reproduced in the "
        "suite from its two inputs rather than pinned as a literal"
    ),
    "tag": "[REGISTERED]",
}


#: **[RECORDED 2026-09-01] Why both R-all and R-clear
#: run.**
#:
#: The two arms differ in **exactly one factor**: whether pairs the panel
#: cannot order are trained on. Same loss, same pairing axis, same
#: backbone, same seeds, same cohort. **Their contrast therefore measures
#: whether noise-ordered pairs help or hurt**, and it is the only
#: quantity in this phase that a single controlled difference isolates.
#:
#: **the ground, as recorded:** running both turns the
#: question from an argument into a measurement. **Running clear-only
#: alone would have been the choice that flatters** -- it is the version
#: with the better story ("we trained on the pairs the panel can
#: actually resolve"), and it would have produced a number with nothing
#: to compare it against. The comparison is the deliverable; either arm
#: alone is an anecdote.
THE_R_ALL_R_CLEAR_GROUND = {
    "recorded": "2026-09-01, the ground, as framed",
    "one_factor": (
        "**R-all and R-clear differ in EXACTLY ONE FACTOR** -- whether "
        "pairs the panel cannot order are trained on. Same loss, same "
        "pairing axis, same backbone, same seeds, same cohort. Their "
        "contrast measures whether noise-ordered pairs HELP or HURT, "
        "and it is the only quantity in this phase isolated by a single "
        "controlled difference"
    ),
    "the_ground": (
        "**running both turns the question from an ARGUMENT into a "
        "MEASUREMENT.** Running clear-only alone would have been **the "
        "choice that flatters** -- the version with the better story "
        "('we trained on the pairs the panel can actually resolve'), "
        "producing a number with nothing to compare it against. The "
        "COMPARISON is the deliverable; either arm alone is an anecdote"
    ),
    "what_it_costs": (
        "one extra arm's compute, and the risk that R-all wins -- which "
        "would say the noise-ordered pairs are doing something the "
        "clean story did not predict. That is the outcome worth paying "
        "for, and the one clear-only could never have produced"
    ),
    "tag": "[REASONED]",
}


#: **[DECLARED UNDER TEST 2026-09-01] The pairing-axis
#: assumption.**
#:
#: Synthetic ordering is **within-face**: the same face at two warp
#: magnitudes. Cohort ordering is **between-patient**: patient X against
#: patient Y. Rank-pretraining on the first and fine-tuning on the second
#: **assumes the ordering transfers across axes** -- that a
#: representation which learns "more warped than the same face" is a
#: representation that helps rank "worse outcome than a different
#: patient".
#:
#: **The ruling: we are assuming it, and it is written down as an
#: assumption rather than treated as known.** The arm is not blocked on
#: proving the transfer; it is registered with the assumption named, so
#: that a null result on R-syn is readable as *either* "ranking does not
#: help" *or* "the axis did not transfer", and the record cannot later
#: pretend the second reading was unavailable.
#:
#: **Same family as magnitude-to-grade** -- the assumption Phase 17
#: carries, that a synthetic warp magnitude stands in for a clinical
#: grade. Both are transfers from a constructed quantity to a clinical
#: one, both are load-bearing, and neither is verified.
PAIRING_AXIS_ASSUMPTION = {
    "declared": "2026-09-01, the ruling, as an ASSUMPTION UNDER TEST",
    "the_two_axes": (
        "**synthetic ordering is WITHIN-FACE** (the same face at two "
        "warp magnitudes); **cohort ordering is BETWEEN-PATIENT** "
        "(patient X against patient Y). They are different relations "
        "sharing the word 'ordering'"
    ),
    "what_is_assumed": (
        "that rank-pretraining on the first and fine-tuning on the "
        "second **TRANSFERS ACROSS AXES** -- that a representation "
        "which learns 'more warped than the same face' helps rank "
        "'worse outcome than a different patient'"
    ),
    "the_ruling": (
        "**we are ASSUMING it, and it is WRITTEN DOWN AS AN ASSUMPTION "
        "rather than treated as known.** The arm is not blocked on "
        "proving the transfer; it is registered with the assumption "
        "named, so a null on R-syn is readable as EITHER 'ranking does "
        "not help' OR 'the axis did not transfer' -- and the record "
        "cannot later pretend the second reading was unavailable"
    ),
    "same_family_as_magnitude_to_grade": (
        "Phase 17's standing assumption that a synthetic warp magnitude "
        "stands in for a clinical grade. **Both are transfers from a "
        "CONSTRUCTED quantity to a CLINICAL one, both load-bearing, "
        "neither verified.** Naming the family matters: it is the same "
        "kind of gap, and this project has one of them already"
    ),
    "it_does_not_touch_the_cohort_arms": (
        "R-all and R-clear train on between-patient pairs directly and "
        "carry NO axis assumption. The assumption is scoped to R-syn "
        "alone, which is why the R-syn vs cohort-arm secondaries exist"
    ),
    "tag": "[REASONED] -- an assumption, explicitly not a finding",
}


#: **[REGISTERED 2026-09-01, READINGS BEFORE NUMBERS]
#: The feature-difference diagnostic.**
#:
#: Every ranking arm's OOF predictions get **Phase 21's error
#: consistency** computed against the 63 shrinking arms, on **both
#: binarizations** (residual sign, worst quartile), with **the same
#: chance correction** -- the machinery is reused unchanged, so the
#: figures land on the same scale as everything Phase 21 measured.
#:
#: **Low kappa is the feature-difference signature.** This is arm A's
#: control applied **prospectively** rather than post hoc: Phase 21
#: discovered after the fact that the one arm doing something different
#: agreed with the rest at chance. Phase 22 registers that reading before
#: any ranking arm runs.
#:
#: **THE THRESHOLDS ARE DERIVED, NOT CHOSEN**, and both endpoints are
#: measurements already in the record:
#:
#: * **HIGH = the all-pairs mean** -- 0.7012 (residual sign), 0.5729
#:   (worst quartile), ``phase21.ARM_B_OBSERVED``. An arm at or above it
#:   is agreeing as much as a typical pair of shrinking arms agree.
#: * **LOW = arm A's measured level** -- +0.0956 and -0.0743,
#:   ``phase21.ARM_A_THE_SHRINKAGE_CONTROL``. An arm at or below it is
#:   agreeing as little as the one arm known to be doing something else.
#:
#: Neither endpoint has a free parameter. **The band between them is wide
#: (0.0956 to 0.7012 on residual sign) and that is honest** -- those are
#: the two levels this project has actually measured, and inventing a
#: midpoint to narrow the band would be the choice the derivation avoids.
FEATURE_DIFFERENCE_DIAGNOSTIC = {
    "registered": "2026-09-01, the ruling, BEFORE any number",
    "what_is_computed": (
        "**every ranking arm's OOF predictions against the 63 SHRINKING "
        "arms**, using Phase 21's error consistency unchanged: both "
        "binarizations (residual sign, worst quartile), the same chance "
        "correction, reported SEPARATELY and never averaged. Reusing "
        "the machinery is what puts the figures on the same scale as "
        "everything Phase 21 measured"
    ),
    "why_the_63_and_not_the_64": (
        "arm A is EXCLUDED from the comparison set because it is the "
        "reference the low threshold comes from. Including it would put "
        "the same arm on both sides of the diagnostic"
    ),
    "arm_as_control_applied_prospectively": (
        "**Phase 21 discovered AFTER the fact that the one arm doing "
        "something different agreed with the rest at chance. Phase 22 "
        "registers that reading BEFORE any ranking arm runs.** The same "
        "observation, moved from post-hoc to pre-registered, which is "
        "the only move that turns it into evidence"
    ),
    "thresholds_derived_not_chosen": (
        "**both endpoints are MEASUREMENTS already in the record, "
        "neither has a free parameter.** HIGH = the all-pairs mean, "
        "0.7012 (residual sign) / 0.5729 (worst quartile), "
        "phase21.ARM_B_OBSERVED. LOW = arm A's measured level, +0.0956 "
        "/ -0.0743, phase21.ARM_A_THE_SHRINKAGE_CONTROL"
    ),
    "the_wide_band_is_honest": (
        "0.0956 to 0.7012 on residual sign is a WIDE band, and it stays "
        "wide. **Those are the two levels this project has actually "
        "measured; inventing a midpoint to narrow it would be exactly "
        "the choice the derivation avoids.** An arm landing in the "
        "middle gets its value reported and no claim -- which is the "
        "correct outcome for a value the record cannot interpret"
    ),
    "readings": {
        "kappa_at_or_above_the_all_pairs_mean": (
            "**>= 0.7012 (residual sign) or >= 0.5729 (worst quartile): "
            "THE ARM MAKES THE SAME ERRORS.** It is the same thing "
            "under a new objective -- a different loss reaching the "
            "same representation. **This reading holds WHATEVER ITS PCC "
            "DOES**, including a PCC gain: an arm that scores higher "
            "while making the same errors has not found different "
            "features, and the phase's hypothesis is unsupported by "
            "that arm regardless of its delta"
        ),
        "kappa_at_or_below_arm_as_level": (
            "**<= +0.0956 (residual sign) or <= -0.0743 (worst "
            "quartile): THE ARM IS LOOKING AT SOMETHING DIFFERENT.** "
            "This is the phase's actual hypothesis, and it is **EVIDENCE "
            "INDEPENDENT OF THE PCC DELTA** -- it can fire on an arm "
            "whose PCC is unresolved, which given the cohort is the "
            "expected case. Reported with Geirhos's CNN-to-human OOD "
            "band (0.066-0.068) BESIDE it as context, never as a "
            "threshold"
        ),
        "kappa_between": (
            "**REPORT THE VALUE, CLAIM NEITHER.** The band is wide and "
            "the record holds no measured level inside it. A value here "
            "is reported as a point estimate with its spread and no "
            "reading attached -- not 'weak evidence of difference', not "
            "'trending toward the mean'. Neither"
        ),
    },
    "reported_separately_never_averaged": (
        "as phase21.ARM_B_REGISTERED requires -- the two binarizations "
        "answer different questions and collapsing them would be the R2 "
        "shape. An arm may fire different readings on the two, and that "
        "is reported as it stands, not resolved"
    ),
    "tag": "[REGISTERED] -- readings committed before numbers",
}


#: **[REGISTERED 2026-09-01, COUNT FIXED AT SIX] The contrast family.**
#:
#: Phase 17's pattern: the family is declared with its count fixed, every
#: outcome is registered as claimable / withdrawn / unresolved in
#: advance, and **nothing is added afterwards.**
#:
#: **Three PRIMARIES** -- each ranking arm against the 0.2520 probe,
#: under the full PLAN §4.3 criterion.
#:
#: **Three SECONDARIES** -- R-all vs R-clear (the noise-pair question,
#: one controlled factor); R-syn vs R-all and R-syn vs R-clear (the axis
#: question, against each cohort arm separately because the two cohort
#: arms are not interchangeable).
#:
#: **The multiplicity is recorded rather than corrected.** Six contrasts
#: against a cohort that resolves almost nothing will produce a spread of
#: point estimates; no alpha correction is applied because **each
#: contrast is registered and reported individually, not searched over
#: for the best one.** The protection here is the fixed count and the
#: pre-written readings, not a correction factor.
CONTRAST_FAMILY = {
    "registered": "2026-09-01, count FIXED, Phase 17's pattern",
    "redeclared": (
        "**2026-09-01, AT THE LOCK, BEFORE ANY NUMBER.** The maintainer's "
        "head-bounding ruling took the phase from three arms to SIX, "
        "so the family is recomputed. **The original registration is "
        "preserved in `superseded_registration` below.** Redeclaring "
        "before any arm runs is not the thing the nothing-added clause "
        "forbids; adding after a number is"
    ),
    "count": 15,
    "confirmed_at_fifteen": (
        "**[CONFIRMED 2026-09-01, AGAINST THE DERIVED "
        "COUNT.] The earlier ruling was issued on the record's 'roughly "
        "doubles'; it is REISSUED here against the derived 15**, so "
        "the family does not rest on a count nobody checked. The three "
        "grounds: **(i)** every contrast is a SINGLE CONTROLLED "
        "FACTOR; **(ii)** every outcome is reported whatever it is; "
        "**(iii)** **the available reductions all cost information** -- "
        "taking the axis question at ONE bounding only would compare "
        "R-syn under one head and not the other, **an asymmetry with "
        "no justification**. A cheaper family was available and would "
        "have been cheaper by leaving a question half-asked"
    ),
    "the_count_is_larger_than_the_ruling_assumed": (
        "**REPORTED, NOT ABSORBED. The ruling was issued on 'six arms "
        "roughly doubles the contrast family'. It does not double it: "
        "6 -> 15, a factor of 2.5.** Doubling the ARMS more than "
        "doubles the contrasts because the head question adds a "
        "within-arm pair (3) and the axis question is taken separately "
        "at each bounding (4 instead of 2). **The ruling was with a "
        "smaller number in view than the one the design "
        "produces, and that is open to re-ruling** -- the count is stated "
        "here rather than quietly locked at 15"
    ),
    "primaries": (
        "**SIX**, each under the full PLAN 4.3 criterion (per-seed "
        "paired BCa excluding zero AND delta exceeding combined seed "
        "uncertainty), each arm against the 0.2520 probe: R-syn-B, "
        "R-syn-U, R-all-B, R-all-U, R-clear-B, R-clear-U "
        "(B = bounded head, U = unbounded)"
    ),
    "secondaries": (
        "**NINE**, in three groups. **THE HEAD QUESTION (3)**: "
        "R-syn-B vs R-syn-U, R-all-B vs R-all-U, R-clear-B vs "
        "R-clear-U -- one controlled factor each, the bounding alone. "
        "**THE NOISE-PAIR QUESTION (2)**: R-all-B vs R-clear-B and "
        "R-all-U vs R-clear-U -- taken at EACH bounding, because a "
        "single cross-bounding comparison would confound the pair set "
        "with the head. **THE AXIS QUESTION (4)**: R-syn against each "
        "cohort arm WITHIN its own bounding -- R-syn-B vs R-all-B, "
        "R-syn-B vs R-clear-B, R-syn-U vs R-all-U, R-syn-U vs "
        "R-clear-U. **No contrast crosses the bounding except the "
        "three that exist to measure it**, which is what keeps every "
        "other contrast to one controlled factor"
    ),
    "superseded_registration": (
        "ORIGINAL, 2026-09-01, three arms, count SIX: three primaries "
        "(R-syn, R-all, R-clear each vs the probe) and three "
        "secondaries (R-all vs R-clear; R-syn vs R-all; R-syn vs "
        "R-clear). **Superseded by the head-bounding ruling BEFORE any "
        "arm ran, not after a result**"
    ),
    "nothing_added_afterwards": (
        "**the count is SIX and it is fixed at registration.** No "
        "contrast may be added after the numbers arrive, including one "
        "that looks obvious in hindsight. If an unregistered comparison "
        "turns out to matter it is reported as [POST-HOC], fires no "
        "cell, and supports no claim -- the treatment "
        "phase21.POST_HOC_ANALYSES received"
    ),
    "multiplicity_recorded_not_corrected": (
        "**[RULED 2026-09-01] Fifteen contrasts against a "
        "cohort that resolves almost nothing will produce a spread of "
        "point estimates. NO ALPHA CORRECTION**, on three recorded "
        "grounds: **(i)** the count is FIXED BEFORE ANY NUMBER and "
        "cannot grow; **(ii)** EVERY outcome is reported whatever it "
        "is, so there is no selection of survivors to correct for; "
        "**(iii)** **the two-condition PLAN 4.3 criterion is already "
        "far stricter than any nominal level** -- "
        "ladder.SMALLEST_RESOLVABLE_DIFFERENCE records **30 tested, 1 "
        "survived, 29 withdrawn** in the 0.04-0.10 band. A criterion "
        "admitting 1 in 30 is not a criterion that needs its alpha "
        "tightened. **The protection is the fixed count, the "
        "pre-written readings and the criterion itself, not a "
        "correction factor** -- stated so a reader can judge that "
        "protection rather than discover the count afterwards"
    ),
    "every_outcome_pre_assigned": (
        "each of the six gets CLAIMABLE / WITHDRAWN / UNRESOLVED "
        "assigned at the lock, before any run, per READINGS_BOTH_WAYS"
    ),
    "tag": "[REGISTERED]",
}


#: **[REGISTERED 2026-09-01, BEFORE ANY NUMBER] Readings both ways per
#: arm -- and the one that matters most.**
#:
#: **The expected outcome is UNRESOLVED, and that is registered rather
#: than discovered.** A gain must clear **0.1386** to be claimable
#: (``ladder.SMALLEST_RESOLVABLE_DIFFERENCE`` -- the smallest delta ever
#: to satisfy both PLAN §4.3 conditions), and this cohort **cannot
#: resolve differences of 0.04 to 0.10**. A ranking arm producing a
#: genuine but modest improvement lands in the band the cohort cannot
#: see. **The phase is being opened knowing its primary metric will most
#: likely return nothing.**
#:
#: **WHAT AN UNRESOLVED POSITIVE LICENSES -- the reading that matters
#: most, registered in advance.** The PCC delta is reported as **a point
#: estimate with its interval and NO CLAIM.** Not "a promising trend",
#: not "suggestive of a gain", not "consistent with the hypothesis". A
#: point estimate and an interval that includes zero, described as such.
#: **The phase's finding then rests entirely on the diagnostic** --
#: which is why ``FEATURE_DIFFERENCE_DIAGNOSTIC`` exists and why it was
#: registered with derived thresholds before any arm ran. A phase whose
#: expected primary outcome is null needs a second instrument that can
#: still return something, and it needs it declared up front, not
#: assembled once the primary disappoints.
READINGS_BOTH_WAYS = {
    "registered": "2026-09-01, before any number, both directions per arm",
    "the_expected_outcome_is_unresolved": (
        "**a gain must clear 0.1386 to be claimable** "
        "(ladder.SMALLEST_RESOLVABLE_DIFFERENCE -- the smallest delta "
        "ever to satisfy both PLAN 4.3 conditions) **and this cohort "
        "cannot resolve 0.04 to 0.10.** A ranking arm producing a "
        "genuine but modest improvement lands in the band the cohort "
        "cannot see. **The phase is opened KNOWING its primary metric "
        "will most likely return nothing**, and that is registered "
        "rather than discovered afterwards"
    ),
    "per_arm_both_ways": {
        "claimable_gain": (
            "delta vs the probe clears 0.1386 AND satisfies both PLAN "
            "4.3 conditions -> **CLAIMABLE**: ordering-based training "
            "raises PCC on this cohort, at the stated magnitude, for "
            "that arm"
        ),
        "claimable_loss": (
            "delta clears the criterion in the NEGATIVE direction -> "
            "**recorded as a claimable LOSS, not as 'no effect'.** An "
            "arm that measurably hurts is a finding, and the "
            "asymmetry that reports gains and shrugs at losses is how "
            "a record drifts optimistic"
        ),
        "unresolved": (
            "**the expected case.** Delta inside the band -> reported "
            "as a point estimate with its interval and NO CLAIM in "
            "either direction"
        ),
    },
    "what_an_unresolved_positive_licenses": (
        "**the reading that matters most.** The PCC delta is reported "
        "as **a point estimate with its interval and NO CLAIM** -- not "
        "'a promising trend', not 'suggestive of a gain', not "
        "'consistent with the hypothesis'. A point estimate and an "
        "interval including zero, described as such. **The phase's "
        "finding then rests ENTIRELY on the diagnostic**, which is why "
        "FEATURE_DIFFERENCE_DIAGNOSTIC exists and why its thresholds "
        "were derived before any arm ran. A phase whose expected "
        "primary outcome is null needs a second instrument declared UP "
        "FRONT, not assembled once the primary disappoints"
    ),
    "what_a_claimable_gain_would_license": (
        "that **ordering-based training raises PCC on this cohort**, at "
        "the measured magnitude, for the arms that cleared it -- "
        "reported with the pairing-axis assumption attached to R-syn "
        "and the noise-pair contrast attached to R-all/R-clear"
    ),
    "and_what_it_would_STILL_NOT_license": (
        "**THE SHRINKAGE MECHANISM, WHICH REMAINS FORBIDDEN.** A PCC "
        "gain would NOT be evidence that the arm gained by avoiding "
        "shrinkage: SCALE_INVARIANCE_PROHIBITION forbids that reading "
        "arithmetically, and arm A demonstrates it empirically at "
        "0.2334. **A gain would establish THAT ordering helps, never "
        "WHY** -- and the why is precisely the sentence a write-up "
        "reaches for. Registered here so that reaching for it is a "
        "visible contradiction of a pre-committed record rather than a "
        "plausible inference nobody checks"
    ),
    "tag": "[REGISTERED] -- readings committed before numbers",
}


#: **[DRAFT 2026-09-01, NOT LOCKED] Exit criteria.**
#:
#: Twelve, in the project's standing form, to be locked with the maintainer's
#: amendments before anything is built. **A draft is not a lock**: these
#: are proposed, and the phase cannot start against them until they are
#: fixed.
EXIT_CRITERIA_DRAFT = {
    "drafted": "2026-09-01, NOT LOCKED -- proposed for the amendment",
    "1_reckoning_stands_at_the_top": (
        "PHASE_22_RECKONING and THE_STRONGER_MECHANISM_ASSERTED "
        "are in the record before any arm is built, with the refuted "
        "mechanism and the order of events both stated"
    ),
    "2_three_arms_built_as_registered": (
        "R-syn, R-all, R-clear, each matching its registration -- no "
        "arm dropped, no arm added, no pair set altered after the lock"
    ),
    "3_r_all_and_r_clear_differ_in_one_factor_only": (
        "verified at the config level, not asserted: identical apart "
        "from the pair set. If any other field differs, the secondary "
        "contrast measures something else and must be withdrawn"
    ),
    "4_the_se_diff_threshold_is_reproduced_not_pinned": (
        "0.398942 recomputed from sd_obs and RELIABILITY_237 at build "
        "time and checked against the separation run's value"
    ),
    "5_the_diagnostic_runs_on_every_arm": (
        "error consistency against the 63 shrinking arms, both "
        "binarizations, same chance correction, reported separately "
        "and never averaged"
    ),
    "6_the_diagnostic_thresholds_are_the_registered_ones": (
        "HIGH = 0.7012 / 0.5729, LOW = +0.0956 / -0.0743, both read "
        "from their phase21 sources at run time. **If either source "
        "figure has changed, the phase stops and the change is "
        "investigated before the diagnostic is read**"
    ),
    "7_six_contrasts_and_no_more": (
        "the family is reported at exactly six, each with its "
        "pre-assigned verdict. Any additional comparison appears tagged "
        "[POST-HOC], fires no cell, supports no claim"
    ),
    "8_every_contrast_gets_a_verdict": (
        "claimable / withdrawn / unresolved for all six, including the "
        "ones that returned nothing. **A contrast with no verdict is a "
        "contrast quietly dropped**"
    ),
    "9_the_pairing_axis_assumption_is_attached_to_r_syn_wherever_it_is_read": (
        "any R-syn result, positive or null, carries "
        "PAIRING_AXIS_ASSUMPTION with it -- a null on R-syn may not be "
        "reported as 'ranking does not help' without the alternative "
        "reading beside it"
    ),
    "10_no_pcc_gain_is_attributed_to_shrinkage": (
        "READINGS_BOTH_WAYS['and_what_it_would_STILL_NOT_license'] is "
        "checked against the write-up text. "
        "SCALE_INVARIANCE_PROHIBITION remains binding through the "
        "phase's close"
    ),
    "11_the_expected_null_is_reported_as_a_null": (
        "an unresolved delta is a point estimate with an interval and "
        "no claim. **The phase closes with 'unresolved' in it if that "
        "is what happened**, and the diagnostic carries whatever "
        "finding there is"
    ),
    "12_nothing_added_after_the_lock": (
        "the standing clause: no arm, no contrast, no reading, no "
        "threshold introduced once numbers exist. Anything discovered "
        "afterwards is [POST-HOC] and fires nothing"
    ),
    "tag": "[DRAFT] -- not locked, not binding until it is fixed",
}


#: **[REPORTED 2026-09-01, NOT AN EDIT] What a ranking arm needs that the
#: config surface cannot express.**
#:
#: Read out of ``config/schema.py`` at source. **This is a report, not a
#: proposal and not a change** -- no schema kind is added, no field
#: extended, nothing edited. the maintainer decides what the surface becomes.
#:
#: **Four things are missing, and the nearest existing surface points the
#: wrong way on every one of them.**
#:
#: The nearest kind is ``siamese_contrastive`` (Phase 17's arms B and C).
#: Its closed choices are: ``arm`` restricted to exactly
#: ``('p17_arm_b', 'p17_arm_c')`` -- a new arm cannot even be named;
#: ``pair_rule`` restricted to ``symmetric_if_zero_magnitude`` -- binary,
#: not ordered; ``readout_normalization`` fixed at
#: ``round_1_plus_4_min_d_over_margin`` -- a **distance between two
#: embeddings** mapped to a grade, not a per-item score;
#: ``branch_trainability`` frozen; ``synth_set`` as the only pair source,
#: with no cohort-pair input at all. **Every closed choice on the closest
#: surface excludes what a ranking arm needs.**
#:
#: ``train_cv`` has **no loss field at all** -- the objective is implicit
#: in ``models/factory.py`` (MSE on the raw scale) and selected only
#: indirectly through ``label``. ``pretrain`` monitors
#: ``inner_val_mse`` / ``inner_val_pcc``, with no ranking metric.
CONFIG_SURFACE_REPORT = {
    "reported": "2026-09-01, read at source, AS A REPORT -- nothing edited",
    "not_an_edit": (
        "**no schema kind added, no field extended, no config written.** "
        "The surface is reported as it stands so the maintainer can decide "
        "what it becomes. Extending a frozen-adjacent surface on "
        "an unprompted addition is the move this project does not make"
    ),
    "1_the_loss": (
        "**MISSING ENTIRELY.** train_cv has NO loss or objective field: "
        "the objective is implicit in models/factory.py -- num_outputs=1 "
        "and MSE on the raw 1-5 scale -- and reachable only indirectly "
        "through `label`. A pairwise ranking loss (margin ranking, "
        "RankNet-style, or a Plackett-Luce listwise) has no field to "
        "live in"
    ),
    "2_pair_construction": (
        "**MISSING FOR THE COHORT.** The only pair machinery in the "
        "schema is siamese_contrastive's `pair_rule`, whose sole choice "
        "is symmetric_if_zero_magnitude -- BINARY same/different over "
        "SYNTHETIC faces. There is no field expressing 'all C(237,2) "
        "cohort pairs ordered by panel mean', and none expressing the "
        "SE_diff restriction that separates R-all from R-clear"
    ),
    "3_a_per_item_scalar_head": (
        "**[CORRECTED 2026-09-01 -- THIS ENTRY WAS WRONG.** A per-item "
        "scalar head is NOT missing. **factory.build(num_outputs=1) "
        "ships**: it returns a timm model with num_classes=1, and "
        "EVERY regression arm in the project already uses it. The "
        "missing piece is **the LOSS that runs two items through that "
        "head and compares their scalars** -- not the head. The "
        "original entry, preserved below, was true OF THE SIAMESE KIND "
        "and generalised wrongly to the project, which overstated what "
        "Phase 22 has to build.** "
        "ORIGINAL, 2026-09-01: 'THE READOUT IS THE WRONG SHAPE. "
        "siamese_contrastive's readout_normalization is fixed at "
        "round_1_plus_4_min_d_over_margin -- it maps a DISTANCE BETWEEN "
        "TWO EMBEDDINGS to a grade. A ranking arm needs a per-item "
        "scalar score, compared pairwise in the loss and read out "
        "singly at inference. Those are different architectures, not a "
        "different setting.' **The original's LAST TWO SENTENCES "
        "remain true of siamese_contrastive specifically; only the "
        "generalisation to the project was wrong**"
    ),
    "3_correction_provenance": (
        "**a recorded error, caught by the record, in the cycle after it was "
        "written.** The config-surface report was compiled by reading "
        "schema.py's TASK_SPECS -- where train_cv indeed has no head "
        "field -- WITHOUT reading factory.py's build signature, where "
        "num_outputs=1 has always been the default. **A surface audit "
        "that reads only the config layer misses what the code layer "
        "already provides**, which is the specific failure mode here "
        "and is worth naming because the same audit style produced the "
        "other three entries. Those three were re-checked at the code "
        "layer when this one was corrected and stand"
    ),
    "4_the_inter_patient_pairing_axis": (
        "**NO COHORT PAIR SOURCE EXISTS.** siamese_contrastive takes "
        "`synth_set` and nothing else; its `arm` field is closed to "
        "('p17_arm_b', 'p17_arm_c'), so a new arm cannot be named "
        "without touching the schema. R-all and R-clear need "
        "between-patient pairs drawn from the manifest, which no task "
        "kind currently reads for pairing"
    ),
    "what_the_surface_CAN_express": (
        "backbone, seeds, staged and manifest artifacts, learning rate, "
        "batch size, epochs, patience, trainability, inner validation "
        "fraction -- **every arm-shaping field a ranking arm needs "
        "EXCEPT the four above.** The gap is narrow and specific, not "
        "a rewrite: one objective, one pair source, one head, one axis"
    ),
    "the_honest_summary": (
        "**the closest existing kind is the one that reads an ordering "
        "and discards it, and every one of its closed choices points "
        "away from ranking.** That is a coherent statement of how much "
        "work Phase 22 is, and it is reported before anything is built "
        "so the cost is visible at the ruling rather than at the lock"
    ),
    "tag": "[MEASURED] -- field names and choices read from schema.py",
}


#: **[RULED 2026-09-01] The loss: pairwise logistic
#: (RankNet-style) with soft targets.**
#:
#: **The ground, recorded with it:** its native parameterisation
#: expresses what this cohort forces. **P = 0.5 IS "the panel cannot
#: order these two"** -- which is **24.85%** of R-all's pairs, a
#: measured fraction, not an allowance. And it needs **no margin**,
#: committing the phase to one fewer undeclarable constant.
LOSS_RULED = {
    "ruled": "2026-09-01, before the lock",
    "the_ruling": (
        "**(b) PAIRWISE LOGISTIC (RankNet-style), SOFT TARGETS.** "
        "log(1 + exp(-(s_i - s_j))) over per-item scores, with a target "
        "probability rather than a hard order"
    ),
    "the_ground": (
        "**its native parameterisation expresses what this cohort "
        "forces: P = 0.5 IS 'the panel cannot order these two', which "
        "is 24.85% of R-all's pairs** -- a measured fraction "
        "(phase21.COHORT_PAIR_SEPARATION_OBSERVED), not an allowance. "
        "**And it needs NO MARGIN, committing the phase to one fewer "
        "undeclarable constant** than the hinge would"
    ),
    "d_rejected_by_name": (
        "**(d) COMBINED MSE + rank is REJECTED BY NAME**, for three "
        "reasons: **(i)** it reintroduces the shrinking term -- MSE on "
        "the raw scale is exactly what the 63 arms optimise and "
        "exactly what produces shrinkage, so the phase would carry "
        "inside it the mechanism it exists to avoid; **(ii)** it makes "
        "any gain **unattributable between the halves** -- a hybrid's "
        "PCC cannot be assigned to the ranking term; **(iii)** it "
        "**BREAKS THE DIAGNOSTIC**, because a half-MSE arm SHOULD "
        "agree with the shrinking arms, so a high kappa would carry no "
        "information. **Rejected on the record's own grounds, not on "
        "taste**"
    ),
    "a_and_c_unregistered": (
        "**EXPLICITLY UNREGISTERED -- no fishing** (Phase 14's "
        "pattern). **(a) pairwise margin/hinge**: viable and cheap, "
        "unregistered because it requires a MARGIN, which "
        "phase17.DECLARED_SETTINGS_17 establishes is a scientific "
        "setting defining the score unit -- a constant the ruling "
        "deliberately avoided committing to. **(c) listwise "
        "(Plackett-Luce / ListNet)**: unregistered because it assumes "
        "a consistent TOTAL ORDER, which the panel does not supply "
        "below SE_diff, and needs a list sampler this phase has no "
        "reason to build. **Naming both here is what stops a later "
        "turn presenting either as a fresh idea that escapes this "
        "record**"
    ),
    "what_the_ruling_leaves_open": (
        "**the TARGET PROBABILITY MAP.** 'Soft targets' requires a "
        "rule mapping panel separation to P_ij, and no ruling supplies "
        "one. See DECLARED_SETTINGS_22 -- **it is a scientific setting "
        "surfaced BY this ruling and not closed by it**"
    ),
    "literature": (
        "RankIQA is the closest precedent and matches this shape -- "
        "ranked pairs, pairwise ranking loss, then fine-tune for the "
        "scalar target. **The bank holds NO sample size for it**: "
        "SOURCES_BANKED['rankiqa'] carries a placeholder, and its "
        "flagged clause says quoting their smallest dataset as "
        "encouragement would be the comparator error this project has "
        "made before. **Its 'pair order certain by construction' "
        "caveat lands on R-all and R-clear ONLY -- R-syn's order IS "
        "certain by construction, exactly as theirs is**"
    ),
    "tag": "[RULED]",
}


#: **[RULED 2026-09-01] Pair construction: all pairs each
#: epoch, formed inside ``train_epoch``.**
#:
#: **Fold honesty becomes STRUCTURAL rather than policed.** The frozen
#: harness hands the backbone ``train_epoch(features[train_rows],
#: train_labels)`` -- **only that fold's training rows**. A pair formed
#: inside that call is between two training patients **by
#: construction**; there is no code path by which it could span the
#: boundary, so nothing has to check that it does not.
PAIR_CONSTRUCTION_RULED = {
    "ruled": "2026-09-01, before the lock",
    "the_ruling": (
        "**(a) ALL PAIRS EACH EPOCH, FORMED INSIDE train_epoch.** "
        "~17.9k pairs per epoch from ~190 training patients, cheap on "
        "frozen embeddings, **deterministic -- no sampler seed**"
    ),
    "fold_honesty_is_structural_not_policed": (
        "**the frozen harness passes train_epoch ONLY that fold's "
        "training rows** (harness.py: "
        "backbone.train_epoch(features[train_rows], train_labels)), so "
        "**a pair formed inside that call is between two TRAINING "
        "patients BY CONSTRUCTION.** There is no code path by which it "
        "spans the train/test boundary, so **nothing has to check that "
        "it does not** -- the guarantee is structural. This is the "
        "property that made (a) preferable to any design precomputing "
        "a pair list outside the loop"
    ),
    "why_it_mattered_here": (
        "**folds.py grouping is per-PATIENT** (StratifiedGroupKFold, "
        "groups = patient). It guarantees a patient never spans the "
        "boundary and **says nothing about a PAIR of patients** -- A in "
        "fold 1 and B in fold 3 are each honest while the pair "
        "straddles. **This is a shape the project has never had**, "
        "since every arm to date scored patients independently. The "
        "ruling removes the risk by construction instead of adding a "
        "check for it"
    ),
    "c_rejected_scoped_not_general": (
        "**(c) DIFFICULTY SCHEDULE is REJECTED BY NAME, and the "
        "rejection is SCOPED TO THIS PHASE.** Ordering pairs by "
        "separation IS R-clear's filter applied gradually, so it would "
        "**blur the single controlled factor between R-all and "
        "R-clear** -- the one cleanly isolated contrast in the phase. "
        "**This is not a judgement on curricula**, which are untested "
        "here and unaddressed by this record; it is a statement that "
        "this particular schedule collides with this particular "
        "family"
    ),
    "b_unregistered": (
        "**(b) per-batch sampling: EXPLICITLY UNREGISTERED.** It is "
        "the option for a fine-tuned backbone and would add two "
        "settings (pairs-per-epoch, sampling seed). Unregistered "
        "because (a) is affordable on frozen embeddings and costs "
        "neither"
    ),
    "the_three_arms_differ_here": (
        "**R-syn's pairs are WITHIN-FACE and SYNTHETIC**, drawn before "
        "any cohort fold is opened -- **no leak path exists and fold "
        "honesty is not a question for it**. R-all and R-clear are "
        "between-patient and real, and carry the whole issue. The "
        "asymmetry is why pair SOURCE is a per-arm setting while the "
        "sampling MODE is phase-wide"
    ),
    "evaluation_is_unchanged": (
        "**only TRAINING is paired.** Evaluation stays per-patient PCC "
        "exactly as every other arm, so the OOF CSVs land in the "
        "format phase21's diagnostic already reads -- no new "
        "evaluation object, and no pair-level bookkeeping the fold "
        "artifact cannot express"
    ),
    "tag": "[RULED]",
}


#: **[RULED 2026-09-01] The head: reuse the shipped scalar
#: head with a ranking ``train_epoch``.**
#:
#: On the ``train/ldl.py`` precedent -- **a non-MSE objective living
#: entirely in its own ``train_epoch`` without touching frozen
#: apparatus.** ``train_epoch(features, labels) -> float`` is a Protocol
#: in frozen ``harness.py``, and eight backbones already implement it.
HEAD_RULED = {
    "ruled": "2026-09-01, before the lock",
    "the_ruling": (
        "**(a) REUSE THE SHIPPED factory.build(num_outputs=1) SCALAR "
        "HEAD, with a ranking train_epoch.** The head is not new; the "
        "loss that runs two items through it is"
    ),
    "the_ldl_precedent": (
        "**train/ldl.py is the shape**: a NON-MSE objective (KL over a "
        "five-output softmax) living **entirely inside its own "
        "train_epoch**, satisfying the frozen "
        "train_epoch(features, labels) -> float Protocol **without "
        "touching frozen apparatus**. Eight backbones implement that "
        "Protocol today. A ranking loss takes the same route"
    ),
    "where_the_precedent_stops": (
        "**LDL reports its distribution's EXPECTATION, so its output "
        "stays on the label scale. A pure ranking arm's scores do "
        "not.** LDL is a precedent for the LOSS, not for the SCALE -- "
        "and the scale is what MONITOR_BIND_RESOLVED had to settle"
    ),
    "b_rejected_as_a_second_implementation": (
        "**(b) Siamese reuse of Phase 17's branches is REJECTED: it "
        "would be a SECOND IMPLEMENTATION of the shipped "
        "run.task_siamese_contrastive extract-and-project path.** "
        "Disqualifying on the standing rule, independent of whether it "
        "would work"
    ),
    "what_phase_17s_readout_produces_and_why_it_cannot_serve": (
        "read from run.task_siamese_contrastive: "
        "**distance = ||left@w.T - right@w.T||** between the LEFT and "
        "RIGHT views of ONE patient's face, then "
        "phase17.distance_to_grade = round(1 + 4*min(d/margin, 1)). It "
        "does emit one number per patient -- but the quantity is **that "
        "patient's own left-right ASYMMETRY**, and the map to a grade "
        "is **a DECLARED FORMULA, not a learned head**. It cannot "
        "serve because the quantity is intra-face symmetry rather than "
        "a learned quality score, and **nothing in it is ever trained "
        "to order two DIFFERENT patients**"
    ),
    "the_minimum_honest_new_component": (
        "**one backbone class with a ranking train_epoch.** That is "
        "the whole of it. The head ships, the Protocol ships, the "
        "harness is untouched, and the fold honesty comes free from "
        "PAIR_CONSTRUCTION_RULED"
    ),
    "tag": "[RULED]",
}


#: **[RULED 2026-09-01] The axis: a new task kind.**
#:
#: One new ``TASK_SPECS`` entry plus its task function. **Touches
#: neither ``train_cv``'s closed vocabulary nor the shipped
#: ``siamese_contrastive``** -- which is how this repo already grows:
#: ``train_graph_cv``, ``train_scut_decoder`` and
#: ``siamese_contrastive`` are all separate kinds rather than flags on
#: ``train_cv``.
AXIS_RULED = {
    "ruled": "2026-09-01, before the lock",
    "the_ruling": (
        "**(a) A NEW TASK KIND.** One new TASK_SPECS entry plus its "
        "task function"
    ),
    "what_it_does_not_touch": (
        "**neither train_cv's closed vocabulary NOR the shipped "
        "siamese_contrastive.** No existing field is reopened, and no "
        "arm that has produced numbers has its configuration surface "
        "put back in play"
    ),
    "it_is_how_this_repo_already_grows": (
        "train_graph_cv, train_scut_decoder and siamese_contrastive "
        "are all SEPARATE KINDS rather than flags on train_cv. The "
        "ruling follows the standing pattern rather than inventing one"
    ),
    "what_was_rejected": (
        "**(b) extending siamese_contrastive** would have opened FIVE "
        "closed choices (arm, pair_rule, readout_normalization, "
        "branch_trainability, plus a cohort pair source) on a kind "
        "with shipped results. **(c) a pairing block on train_cv** "
        "would have opened the vocabulary every shipped regression arm "
        "depends on. Both unregistered"
    ),
    "tag": "[RULED]",
}


#: **[RULED 2026-09-01] Head bounding: BOTH, on all three
#: arms. Six arms.**
#:
#: **The ground, recorded with it: the bounded-versus-unbounded question
#: was going to be settled by the record's PREDICTION otherwise.** Running
#: both measures it instead of assuming it.
#:
#: The prediction is therefore registered below **as an expectation the
#: phase will confirm or refute**, not as a design assumption -- which
#: is the only form in which it can be wrong in public.
HEAD_BOUNDING_RULED = {
    "ruled": "2026-09-01, before the lock",
    "the_ruling": (
        "**(c) BOTH, on all three arms. SIX ARMS.** R-syn, R-all and "
        "R-clear each in two versions: a **BOUNDED** head (scores on "
        "the 1-5 label scale) and an **UNBOUNDED** head (order only)"
    ),
    "the_ground": (
        "**the bounded-versus-unbounded question was going to be "
        "settled by A PREDICTION otherwise** -- that bounding "
        "makes an arm shrink like the other 63 and blunts the "
        "diagnostic. **Running both MEASURES it instead of ASSUMING "
        "it.** The prediction was offered as a lean in a report; a "
        "lean is not a measurement, and a design that quietly encodes "
        "one is a design with an unmeasured assumption inside it"
    ),
    "the_cost_stated_not_absorbed": (
        "**six arms take the contrast family from 6 to 15** "
        "(CONTRAST_FAMILY), and **the record's own margin evidence "
        "predicts most contrasts return unresolved** -- "
        "ladder.SMALLEST_RESOLVABLE_DIFFERENCE: 30 tested, 1 survived, "
        "29 withdrawn in the 0.04-0.10 band. **The ruling was with "
        "that in view.** Recorded because a doubled arm count "
        "on a cohort that resolves almost nothing is a cost, and a "
        "phase that absorbs its costs silently reports a cheaper "
        "experiment than it ran"
    ),
    "and_the_count_was_understated_at_the_ruling": (
        "**The ruling was issued on 'roughly doubles'; the actual "
        "factor is 2.5 (6 -> 15).** See "
        "CONTRAST_FAMILY['the_count_is_larger_than_the_ruling_"
        "assumed']. Flagged rather than locked past"
    ),
    "tag": "[RULED]",
}


#: **[REGISTERED PREDICTION 2026-09-01, BEFORE ANY ARM RUNS] What the record
#: expects bounding to do -- stated so the phase can refute it.**
#:
#: This is the prediction that would otherwise have been encoded as a
#: design choice. It is written down as **an expectation with a
#: committed reading**, so that the six-arm design either confirms it or
#: catches it.
BOUNDED_HEAD_PREDICTION_REGISTERED = {
    "registered": "2026-09-01, BEFORE any arm runs, as an expectation",
    "whose": (
        "**the record's**, offered as a lean in the options report and "
        "promoted to a registered prediction by the ruling "
        "rather than left as an assumption inside the design"
    ),
    "the_prediction": (
        "**a BOUNDED ranking arm shrinks toward the label mean like "
        "the other 63, and its diagnostic kappa sits NEAR THE "
        "ALL-PAIRS LEVEL (0.7012 / 0.5729). An UNBOUNDED one does "
        "not** -- its scores carry no label-scale target to regress "
        "toward, so its shrinkage figure should resemble arm A's "
        "expansion rather than the median 0.3289, and its kappa should "
        "sit lower"
    ),
    "why_it_is_predicted": (
        "a head trained to sit on the 1-5 scale has a mean to regress "
        "toward; an unbounded score has none. **arm A is the measured "
        "precedent** -- it expands at 2.1730 precisely because it has "
        "no label mean to shrink to "
        "(phase21.ARM_A_THE_SHRINKAGE_CONTROL)"
    ),
    "how_it_can_be_refuted": (
        "**the three bounded-vs-unbounded secondaries measure exactly "
        "this.** If bounded and unbounded arms show comparable "
        "shrinkage, or comparable kappa, the prediction is WRONG and "
        "is recorded as wrong with its date. **It fires no cell and "
        "gates nothing** -- it is an expectation on the record, not a "
        "criterion"
    ),
    "what_it_does_NOT_license": (
        "**it may not be cited as a reason to prefer either head, "
        "before or after the run.** Both are registered; the "
        "prediction exists to be tested by them, not to weight them"
    ),
    "tag": "[REASONED] -- a registered prediction, explicitly not a finding",
}


#: **[RESOLVED 2026-09-01 BY THE SAME RULING] The monitor bind.**
#:
#: ``TrainConfig.monitor`` accepts only ``inner_val_mse`` or
#: ``inner_val_pcc``, validated in **frozen** ``harness.py``. The
#: bounding ruling resolves the bind without touching it: **bounded arms
#: monitor MSE honestly; unbounded arms monitor PCC, as a declared
#: limitation.**
MONITOR_BIND_RESOLVED = {
    "resolved": "2026-09-01, by the head-bounding ruling itself",
    "the_bind": (
        "TrainConfig.monitor accepts ONLY inner_val_mse or "
        "inner_val_pcc, validated in **FROZEN harness.py**. For an "
        "unbounded ranking arm, **MSE is meaningless** -- the "
        "predictions are not on the label scale -- and **PCC is the "
        "claimed metric**"
    ),
    "bounded_arms": (
        "**monitor inner_val_mse HONESTLY.** Their scores are on the "
        "1-5 scale, so MSE means what it means for every other arm and "
        "the standing house choice applies unchanged"
    ),
    "unbounded_arms": (
        "**monitor inner_val_pcc, RECORDED AS A DECLARED LIMITATION.** "
        "harness.py's own comment, quoted: *'MSE is the training "
        "objective; selecting on the reported metric would be "
        "selecting on the thing being claimed.'* **That is exactly "
        "what these three arms do**, and it is disclosed at "
        "registration rather than discovered in review"
    ),
    "frozen_apparatus_untouched": (
        "**no third monitor value is added and harness.py is not "
        "touched.** Both ruled values already exist in the frozen "
        "validator"
    ),
    "the_pairing_makes_the_limitation_CHECKABLE": (
        "**this is what the six-arm design buys beyond the head "
        "question.** A limitation that can only be disclosed is worth "
        "less than one that can be bounded: **if the two heads' PCCs "
        "differ, the selection effect is ONE CANDIDATE EXPLANATION "
        "AMONG OTHERS** for the difference -- alongside the bounding "
        "itself and ordinary arm-to-arm variation. **It does not "
        "isolate the selection effect**, and saying so is the point: "
        "the pairing turns an undiscussable limitation into a "
        "quantity with at least one measurement bearing on it"
    ),
    "tag": "[RULED] -- with a declared limitation attached",
}


#: **[RECORDED 2026-09-01] The rescaling note.**
#:
#: PCC is invariant to affine rescaling, so an unbounded arm's
#: predictions **can** be mapped onto the 1-5 scale for reporting
#: **without changing its PCC**. That is a **presentation step**.
#:
#: **It is never a second arm, and a rescaled vector is never quoted as
#: if it were a bounded arm's result.** The two differ in what they were
#: trained to do, not merely in how they are displayed -- and the whole
#: head question is whether that difference matters.
RESCALING_NOTE = {
    "recorded": "2026-09-01, at the lock, before any number",
    "the_arithmetic": (
        "**PCC is invariant to affine rescaling of the prediction "
        "vector**, so an unbounded arm's scores can be mapped onto the "
        "1-5 scale for reporting **without changing its PCC by any "
        "amount**. The same invariance that forbids the shrinkage "
        "mechanism (SCALE_INVARIANCE_PROHIBITION) permits this"
    ),
    "it_is_a_presentation_step": (
        "**NEVER A SECOND ARM.** A rescaled unbounded arm is the same "
        "arm displayed differently; it fires no cell of its own, "
        "appears in no contrast, and adds nothing to the family's "
        "count of 15"
    ),
    "and_never_quoted_as_a_bounded_arms_result": (
        "**a rescaled unbounded vector is NOT a bounded arm's result.** "
        "The two differ in WHAT THEY WERE TRAINED TO DO, not in how "
        "they are displayed -- **and the whole head question is "
        "whether that difference matters.** Quoting one as the other "
        "would answer the phase's own question by relabelling, which "
        "is the R2 shape: two quantities under one name"
    ),
    "what_rescaling_does_not_recover": (
        "**MSE, acc3, and every absolute metric remain unavailable "
        "for an unbounded arm even after rescaling**, because the map "
        "is chosen post hoc from the arm's own outputs. Only the "
        "rank-based figures (PCC, Spearman) are unaffected, and only "
        "those may be reported"
    ),
    "tag": "[REASONED] -- arithmetic plus a reporting prohibition",
}


#: **[ENUMERATED 2026-09-01, AT THE LOCK] The YAML settings.**
#:
#: **Ten named settings. Seven closed by the rulings, three still
#: open.** Every one under the standing clause, verbatim from
#: ``phase17.DECLARED_SETTINGS_17``: declared in the YAML before the
#: first run, **NEVER tuned across runs**, movement is a dated amendment
#: with a reason.
#:
#: **The count moved DOWN, not up.** The options report estimated 10-11
#: contingent settings; the rulings **removed three** (margin,
#: pairs-per-epoch, sampling seed -- all vanished with the logistic loss
#: and all-pairs sampling) and **surfaced one nobody had named** (the
#: target probability map).
DECLARED_SETTINGS_22 = {
    "enumerated": "2026-09-01, at the lock, after the five rulings",
    "the_standing_clause": (
        "verbatim from phase17.DECLARED_SETTINGS_17: **declared in the "
        "YAML before the first run, NEVER TUNED ACROSS RUNS, movement "
        "is a dated amendment with a reason**"
    ),
    "closed_by_the_rulings": {
        "1_loss": "pairwise_logistic -- LOSS_RULED",
        "2_pair_sampling": (
            "all_pairs_per_epoch -- PAIR_CONSTRUCTION_RULED. "
            "Deterministic, so NO sampler seed exists to declare"
        ),
        "3_head_bounding": (
            "bounded | unbounded, per arm, BOTH built -- "
            "HEAD_BOUNDING_RULED"
        ),
        "4_monitor": (
            "inner_val_mse for bounded arms, inner_val_pcc for "
            "unbounded -- MONITOR_BIND_RESOLVED. Derived from setting "
            "3 but written explicitly in each YAML, because a derived "
            "value that is never written is a value nobody can check"
        ),
        "5_prediction_scale_convention": (
            "**OOF CSVs carry RAW SCORES**; any mapping to 1-5 is a "
            "presentation step applied afterwards and never stored as "
            "the arm's predictions -- RESCALING_NOTE"
        ),
        "6_pair_source": (
            "per arm: synth_within_face (R-syn) | cohort_all (R-all) | "
            "cohort_se_diff (R-clear) -- the three registrations"
        ),
        "7_se_diff_threshold": (
            "**0.398942**, R-clear arms only, RECOMPUTED at build from "
            "sd_obs and RELIABILITY_237 rather than pinned as a "
            "literal -- ARM_R_CLEAR_REGISTERED"
        ),
    },
    "ruled_after_the_lock_with_a_precondition": {
        "8_target_probability_map": (
            "**[RULED 2026-09-01 -- HARD TARGETS, P = 1 if m_i > m_j "
            "else 0, CONDITIONAL ON THE TIE MEASUREMENT.** "
            "TARGET_MAP_RULED. The ruling is not final until "
            "TIE_FRACTION_MEASUREMENT_DESIGNED runs: if ties are a "
            "large fraction it is reopened, and the TIE RULE itself "
            "then becomes an eleventh setting.** ORIGINAL, "
            "2026-09-01: 'SURFACED BY THE LOSS RULING AND NOT CLOSED "
            "BY IT. Soft targets requires a rule mapping panel "
            "separation to P_ij, and no ruling supplies one. This is a "
            "scientific setting in the full sense -- it decides what "
            "the 24.85% of below-SE_diff pairs teach the model, which "
            "is the exact quantity the R-all/R-clear contrast exists "
            "to measure. It must be ruled before the first run, and it "
            "is listed here rather than defaulted.' **The original's "
            "reasoning is what the ruling answered, and it stands**"
        ),
    },
    "still_open_before_the_first_run": {
        "9_r_syn_pretrain_epochs": (
            "R-syn only. Not covered by any ruling; the standing "
            "max_epochs/patience apply to fine-tuning, not to a "
            "pretrain stage"
        ),
        "10_r_syn_finetune_trainability": (
            "R-syn only. **Expressible by the EXISTING train_cv "
            "`trainable` field (head | full, default head)** -- so it "
            "needs a declared VALUE, not a new field"
        ),
    },
    "already_fixed_by_house_rule_needing_no_new_declaration": (
        "seeds (5), max_epochs, patience, inner_val_frac, backbone, "
        "learning rate, batch size"
    ),
    "what_the_rulings_removed": (
        "**three settings vanished**: the MARGIN (the logistic loss "
        "needs none -- LOSS_RULED), PAIRS-PER-EPOCH and the SAMPLING "
        "SEED (all-pairs is deterministic -- PAIR_CONSTRUCTION_RULED). "
        "**The rulings made the phase commit to FEWER constants, not "
        "more**"
    ),
    "the_eleventh_setting_ruled_2026_09_01": {
        "11_tie_rule": (
            "**DROP TIED PAIRS FROM TRAINING** -- TIE_RULE_RULED. "
            "Registered as possibly-an-eleventh before the "
            "measurement; it became one, and is CLOSED. The 2,345 "
            "exact ties are excluded from every ranking arm's pair "
            "set. **The measurement fired "
            "`ties_are_a_small_fraction`, so the target-map ruling was "
            "not reopened** -- the tie rule is a declared detail "
            "inside it, which is what that cell said it would be"
        ),
    },
    "the_two_that_remain_open_and_BLOCK_THE_RUN": (
        "**settings 9 and 10 -- R-syn's pretrain epochs and R-syn's "
        "fine-tune trainability -- ARE STILL OPEN.** "
        "**EXIT_CRITERIA criterion 13 gates the run on all settings "
        "being ruled, so the two R-syn arms CANNOT LAUNCH.** They are "
        "reported as blockers and are NOT defaulted: the R-syn configs "
        "carry an explicit UNRULED sentinel that refuses at load "
        "rather than a plausible number nobody ruled. **The four "
        "cohort arms are unaffected and are launchable**"
    ),
    "and_the_tie_rule_may_become_an_eleventh": (
        "**if TIE_FRACTION_MEASUREMENT_DESIGNED fires "
        "`ties_are_a_large_fraction`, the TIE RULE becomes a declared "
        "setting in its own right** -- candidates already named in "
        "that record so the count can grow by a KNOWN amount rather "
        "than by an invention. **Ten today, eleven in that branch**"
    ),
    "tag": "[RULED] for 1-7, [RULED-CONDITIONAL] for 8, [OPEN] for 9-10",
}


#: **[LOCKED 2026-09-01] Exit criteria.**
#:
#: The draft in ``EXIT_CRITERIA_DRAFT`` is superseded by this lock, with
#: the five rulings closed inside it, the six arms, the redeclared
#: family of fifteen, the ten enumerated settings and the nothing-added
#: clause. **The draft is preserved unchanged for comparison.**
EXIT_CRITERIA = {
    "locked": "2026-09-01, after the maintainer's five rulings",
    "supersedes": (
        "EXIT_CRITERIA_DRAFT, preserved unchanged. The draft's twelve "
        "criteria are carried forward; four are amended by the "
        "rulings and two are added"
    ),
    "1_reckoning_stands_at_the_top": (
        "PHASE_22_RECKONING and THE_STRONGER_MECHANISM_ASSERTED "
        "are in the record before any arm is built, with the refuted "
        "mechanism and the order of events both stated"
    ),
    "2_six_arms_built_as_registered": (
        "**[AMENDED AT THE LOCK -- three arms became six]** R-syn, "
        "R-all and R-clear, each bounded AND unbounded, every one "
        "matching its registration. No arm dropped, no arm added, no "
        "pair set altered after this lock"
    ),
    "3_r_all_and_r_clear_differ_in_one_factor_only": (
        "verified at the config level, not asserted: identical apart "
        "from the pair set, **at each bounding separately**. If any "
        "other field differs, the secondary contrast measures "
        "something else and must be withdrawn"
    ),
    "4_the_se_diff_threshold_is_reproduced_not_pinned": (
        "0.398942 recomputed from sd_obs and RELIABILITY_237 at build "
        "time and checked against the separation run's value"
    ),
    "5_the_diagnostic_runs_on_every_arm": (
        "error consistency against the 63 shrinking arms, both "
        "binarizations, same chance correction, reported separately "
        "and never averaged -- **on all SIX arms**"
    ),
    "6_the_diagnostic_thresholds_are_the_registered_ones": (
        "HIGH = 0.7012 / 0.5729, LOW = +0.0956 / -0.0743, both read "
        "from their phase21 sources at run time. **If either source "
        "figure has changed, the phase stops and the change is "
        "investigated before the diagnostic is read**"
    ),
    "7_fifteen_contrasts_and_no_more": (
        "**[AMENDED AT THE LOCK -- six became fifteen]** the family is "
        "reported at exactly **15** (6 primaries + 9 secondaries), "
        "each with its pre-assigned verdict. Any additional comparison "
        "appears tagged [POST-HOC], fires no cell, supports no claim"
    ),
    "8_every_contrast_gets_a_verdict": (
        "claimable / withdrawn / unresolved for all fifteen, including "
        "the ones that returned nothing. **A contrast with no verdict "
        "is a contrast quietly dropped**"
    ),
    "9_the_pairing_axis_assumption_is_attached_to_r_syn_wherever_it_is_read": (
        "any R-syn result, positive or null, at either bounding, "
        "carries PAIRING_AXIS_ASSUMPTION with it -- a null on R-syn "
        "may not be reported as 'ranking does not help' without the "
        "alternative reading beside it"
    ),
    "10_no_pcc_gain_is_attributed_to_shrinkage": (
        "READINGS_BOTH_WAYS['and_what_it_would_STILL_NOT_license'] is "
        "checked against the write-up text. "
        "SCALE_INVARIANCE_PROHIBITION remains binding through the "
        "phase's close"
    ),
    "11_the_expected_null_is_reported_as_a_null": (
        "an unresolved delta is a point estimate with an interval and "
        "no claim. **The phase closes with 'unresolved' in it if that "
        "is what happened**, and the diagnostic carries whatever "
        "finding there is"
    ),
    "12_nothing_added_after_the_lock": (
        "the standing clause: no arm, no contrast, no reading, no "
        "threshold, no setting introduced once numbers exist. Anything "
        "discovered afterwards is [POST-HOC] and fires nothing"
    ),
    "13_the_three_open_settings_are_ruled_before_the_first_run": (
        "**[ADDED AT THE LOCK]** "
        "DECLARED_SETTINGS_22['still_open_before_the_first_run'] -- "
        "the target probability map, R-syn's pretrain epochs and its "
        "fine-tune trainability. **The phase may not start with any of "
        "the three defaulted silently**; the target probability map in "
        "particular decides what the below-SE_diff pairs teach, which "
        "is what the R-all/R-clear contrast exists to measure"
    ),
    "14_the_unbounded_monitor_limitation_is_reported_wherever_those_arms_are": (
        "**[ADDED AT THE LOCK]** MONITOR_BIND_RESOLVED travels with "
        "every unbounded-arm result: these three arms select on their "
        "claimed metric. **And the bounded/unbounded PCC difference is "
        "reported without attributing it to the selection effect** -- "
        "which is one candidate explanation among others, never the "
        "stated cause"
    ),
    "tag": "[LOCKED]",
}


#: **[RULED 2026-09-01] The target probability map: HARD
#: TARGETS, with the tie measurement as its precondition.**
#:
#: **P = 1 if m_i > m_j else 0.** The setting the loss ruling left open
#: and ``EXIT_CRITERIA`` criterion 13 gates on.
#:
#: **The ground, recorded with it:** the phase's primary metric is
#: expected to return unresolved, **so the finding lives in the
#: secondary** -- and a smooth map softens the one contrast isolated by
#: a single controlled factor. **R-clear is already the
#: honest-about-the-panel arm; making R-all honest too collapses the
#: pair.**
#:
#: And the coherent reading stated **positively, not as a concession**:
#: **training R-all on labels known to be wrong for a quarter of pairs
#: is what R-all IS FOR.** The assumption's falseness is the thing being
#: exposed, not a defect being tolerated.
TARGET_MAP_RULED = {
    "ruled": "2026-09-01, with both sides in front of it",
    "the_ruling": (
        "**OPTION 1, HARD TARGETS: P = 1 if m_i > m_j else 0**, "
        "regardless of separation. **Conditional on the tie "
        "measurement** -- see TIE_FRACTION_MEASUREMENT_DESIGNED"
    ),
    "the_ground": (
        "**the primary metric is expected to return UNRESOLVED, so the "
        "finding lives in the SECONDARY** -- and a smooth map softens "
        "the one contrast isolated by a single controlled factor. "
        "**R-clear is ALREADY the honest-about-the-panel arm; making "
        "R-all honest too collapses the pair** into two arms that "
        "differ in little"
    ),
    "the_positive_reading": (
        "**stated positively, not as a concession: training R-all on "
        "labels KNOWN TO BE WRONG for a quarter of pairs is what R-all "
        "IS FOR.** The assumption's falseness is **the thing being "
        "exposed**, not a defect being tolerated. R-all is the naive "
        "construction anyone would write, built so its cost is "
        "measured rather than assumed"
    ),
    "the_reasons_against_in_full": (
        "**recorded in full because they are strong and the ruling is "
        "CLOSE.** **(i)** Hard targets **assume every pair carries a "
        "real order, which the separation measurement CONTRADICTS for "
        "24.85% of pairs** -- this is not a theoretical objection but a "
        "measured one, from this cohort "
        "(phase21.COHORT_PAIR_SEPARATION_OBSERVED). **(ii)** The "
        "probit empirical-Bayes map (2c) is **PARAMETER-FREE BY "
        "DERIVATION**, uses only measured inputs (sd_obs, SE_single, "
        "RELIABILITY_237), and **is the correct posterior under the "
        "project's own error model** -- P(order is true | d) = "
        "Phi(sqrt(rel) * d / SE_diff), where the shrinkage factor is "
        "exactly the reliability by Kelley's formula. **The ruling was "
        "made with both of these in front of it**, and neither was "
        "answered -- they were outweighed"
    ),
    "option_3_rejected_by_name": (
        "**REJECTED. P = 0.5 is an ACTIVE PULL TOWARD EQUAL SCORES**, "
        "not an absence of signal (ATTRACTOR_FINDING). So R-all would "
        "carry **a compression term R-clear lacks**, and the contrast "
        "would measure *'does adding a compression term on "
        "unresolvable pairs help or hurt'* -- **a DIFFERENT QUESTION "
        "than the one registered**, which is whether noise-ordered "
        "pairs help or hurt. It does not weaken the secondary; it "
        "replaces it"
    ),
    "option_4_rejected_by_name": (
        "**REJECTED. w's shape is a NEW UNDECLARABLE CONSTANT** -- "
        "exactly the cost the logistic loss ruling was taken to avoid "
        "(LOSS_RULED). And **a weight falling with separation "
        "PARTIALLY DUPLICATES R-clear's filter**: R-clear is w's "
        "limiting case, w = 0 below SE_diff and 1 above. The "
        "duplication is named rather than left implicit"
    ),
    "options_2a_2b_2c_unregistered_not_rejected": (
        "**UNREGISTERED, NOT REJECTED -- and the distinction is "
        "deliberate.** The probit-EB map (2c) **is defensible**: "
        "parameter-free, correct under the error model, honest at d = "
        "0 where hard targets are undefined. **It was CONSIDERED, its "
        "derivation is ON RECORD here, and it was NOT TAKEN.** It may "
        "not be presented later as a fresh idea that escapes this "
        "record -- Phase 14's pattern. **2b** is 2c with a flat prior, "
        "which overstates confidence because the true grades are "
        "concentrated; **2a** is 2b's shape with a logistic in place of "
        "the probit, unmotivated by the error model"
    ),
    "the_derivation_kept_on_record": (
        "**2c in full, so it survives the phase**: panel-mean error is "
        "Gaussian with SD = SE_single, so d_obs = delta_true + eps "
        "with eps ~ N(0, SE_diff^2) -- and **SE_diff = SE_single * "
        "sqrt(2) holds exactly in the measured figures** (0.398942 / "
        "0.282095 = 1.41421). With the prior sigma_true^2 = sd_obs^2 - "
        "SE_single^2, the posterior shrinkage factor is **exactly "
        "RELIABILITY_237** (Kelley), giving P = "
        "Phi(sqrt(0.8158) * d / SE_diff) = Phi(0.9032 * d / SE_diff). "
        "At d = SE_diff it returns 0.8168; at half, 0.6742; at zero, "
        "0.5"
    ),
    "literature_does_not_reach_here": (
        "**RankIQA's targets are HARD BY CONSTRUCTION** -- the bank's "
        "own words are that their pair order is known by construction. "
        "**So the precedent covers option 1 ONLY, and covers it in the "
        "setting where option 1's assumption is TRUE.** Nothing banked "
        "addresses uncertain pair order; cartography does not apply "
        "(defined on gold-label probability, explicitly not "
        "regression). **The record says the precedent does not reach "
        "rather than borrowing RankIQA's shape and inheriting an "
        "assumption its data earned and ours does not**"
    ),
    "tag": "[RULED] -- conditional on the tie measurement",
}


#: **[MEASURED 2026-09-01, ARITHMETIC] The attractor finding.**
#:
#: **It changes the standing of the loss ruling and was not visible when
#: that ruling was made**, which is why it is recorded rather than
#: folded in silently.
#:
#: Under cross-entropy against target P, the gradient with respect to
#: the score gap is **sigma(ds) - P**, zero at **ds = logit(P)**,
#: negative below and **positive above**. So a soft target does not
#: DISCOUNT a pair -- **it specifies a TARGET GAP and pulls the scores
#: back together once they exceed it.**
ATTRACTOR_FINDING = {
    "found": "2026-09-01, at the target-map options report, by arithmetic",
    "why_it_is_recorded_separately": (
        "**it changes the standing of LOSS_RULED and was NOT VISIBLE "
        "when that ruling was made.** Recorded rather than folded in "
        "silently, so the loss ruling can be read with what was and "
        "was not known at the time"
    ),
    "the_gradient": (
        "under cross-entropy against target P, the gradient w.r.t. the "
        "score gap is **sigma(ds) - P**, **zero at ds = logit(P)**, "
        "negative below and **POSITIVE ABOVE**"
    ),
    "so_soft_targets_are_an_attractor_not_a_discount": (
        "**a hard target with weight w gives w*(sigma(ds) - 1) < 0 "
        "ALWAYS** -- it pushes the gap wider without limit, and "
        "weighting changes only the urgency. **A soft target P "
        "specifies a TARGET GAP logit(P) and pulls the scores back "
        "together once they exceed it.** Downweighting says 'separate "
        "these, less urgently'; a soft target says 'separate these by "
        "EXACTLY THIS MUCH'. **Genuinely different objects, not the "
        "same one in different clothes**"
    ),
    "and_a_smooth_maps_optimum_is_affine_in_the_panel_mean": (
        "**logit(Phi(z)) ~= 1.6z -- within 1% to z = 0.5, WHERE ALL "
        "BELOW-THRESHOLD PAIRS SIT** (4% at z = 1, 17% at z = 2). So a "
        "smooth probit map's optimum is s_i - s_j ~= (1.6 / SE_diff) * "
        "(m_i - m_j), i.e. **s AFFINE IN THE PANEL MEAN -- the same "
        "target function the MSE arms fit**"
    ),
    "a_milder_relative_of_the_combined_loss_concern": (
        "**the same family as the concern that rejected option (d)**, "
        "and milder in a way worth stating precisely: **soft targets "
        "pin the GAP, not the LEVEL**, so the shrinkage arithmetic "
        "differs and **PCC invariance still holds** "
        "(SCALE_INVARIANCE_PROHIBITION is untouched). A smooth map "
        "would make the ranking arm aim at the regression arm's target "
        "function while differing in LOSS GEOMETRY -- not the same "
        "arm, but a much smaller distinction than the phase's "
        "hypothesis needs"
    ),
    "the_hard_target_ruling_avoids_it": (
        "**hard targets have NO FINITE OPTIMUM** -- the loss pushes "
        "ds -> infinity for every ordered pair, and what stops it is "
        "the CONFLICTING PAIRS. They **aim at ORDER ALONE**, which is "
        "the objective the phase registered. TARGET_MAP_RULED avoids "
        "the finding rather than accommodating it"
    ),
    "tag": "[MEASURED] -- arithmetic, checked numerically",
}


#: **[DESIGNED 2026-09-01, READINGS BEFORE THE NUMBER] The exact-tie
#: fraction.**
#:
#: **Hard targets assign every exact tie a definite WRONG order**:
#: ``m_i > m_j`` is false for a tie, so a tie silently receives P = 0
#: rather than the 0.5 the panel supports. The record holds **24.85%
#: below SE_diff across 17 distinct separations** and **not the tie
#: count**.
#:
#: This runs **before anything is built**. It is a precondition of
#: ``TARGET_MAP_RULED``, not a follow-up to it.
TIE_FRACTION_MEASUREMENT_DESIGNED = {
    "designed": "2026-09-01, readings registered BEFORE the number",
    "why_it_must_run_first": (
        "**hard targets assign every exact tie a definite WRONG "
        "order.** m_i > m_j is FALSE for a tie, so a tie silently "
        "receives **P = 0** -- not the 0.5 the panel supports, and not "
        "an error any check would raise. **The ruling cannot be "
        "specified without knowing how many pairs this touches**"
    ),
    "what_the_record_holds_and_does_not": (
        "**HOLDS**: 24.85% of the 27,966 pairs below SE_diff, across "
        "**17 distinct separations** "
        "(phase21.COHORT_PAIR_SEPARATION_OBSERVED). **DOES NOT HOLD**: "
        "how many of those are exactly zero. Ties are inside the "
        "24.85% and were never separated out"
    ),
    "the_source": (
        "**the same cached manifest the separation run used** -- "
        "no GPU, no new artifact, no new hash, no clinical data "
        "leaving the cluster. Aggregates only"
    ),
    "quantities": {
        "1_tie_count_and_fraction": (
            "the **number** and **fraction** of the 27,966 pairs with "
            "|m_i - m_j| exactly zero"
        ),
        "2_smallest_separations_distribution": (
            "the **smallest few distinct separations** with their pair "
            "counts, for context -- the tie count means little without "
            "knowing what sits just above it"
        ),
    },
    "a_hazard_to_settle_at_build": (
        "**'exactly zero' needs a definition robust to float "
        "representation.** Panel means are stored as floats; if they "
        "are k/5 of integer ratings the grid is exact in decimal but "
        "NOT in binary, so `a == b` and `abs(a - b) < eps` can "
        "disagree. **The comparison rule must be declared, not left to "
        "whatever == does** -- and the rater count per patient should "
        "be confirmed rather than assumed to be five"
    ),
    "the_boundary": (
        "**PART DERIVED, PART CHOSEN, AND SAID PLAINLY.** The "
        "REFERENCE QUANTITY is derived: **the R-all/R-clear contrast "
        "is defined on the below-SE_diff pairs, 24.85% of 27,966 = "
        "~6,950 pairs** (0.2485 * 27,966 = 6,949.55 -- **the record "
        "holds the FRACTION ROUNDED TO FOUR PLACES, not the count**, "
        "so the true figure lies in [6,949, 6,950] once the rounding "
        "interval is carried through, and this measurement settles "
        "it), "
        "so that set -- not the whole cohort -- is what "
        "a tie rule competes with. **The FRACTION on it is CHOSEN at "
        "one half**, on the ground that a majority is the point where "
        "the tie rule stops being a detail inside a larger effect and "
        "becomes the dominant treatment. **Boundary: ties at 12.425% of "
        "all pairs** (half of 24.85%). **No derivation yields a "
        "specific number here, and one is not manufactured** -- the "
        "half is a judgement, stated as one"
    ),
    "readings": {
        "ties_are_a_small_fraction": (
            "**tie fraction < 12.425% of all pairs** (ties are a "
            "MINORITY of the below-SE_diff set): **hard targets are "
            "specifiable with a declared tie rule, and "
            "TARGET_MAP_RULED STANDS AS ISSUED.** Most of what "
            "distinguishes R-all from R-clear is genuinely-ordered "
            "but uncertain pairs, and the tie rule is a detail within "
            "a larger effect"
        ),
        "ties_are_a_large_fraction": (
            "**tie fraction >= 12.425%** (ties are a MAJORITY of the "
            "below-SE_diff set): **the tie rule stops being a detail "
            "and becomes a setting that materially shapes what R-all "
            "learns** -- the R-all/R-clear contrast would then be "
            "mostly a contrast about ties. **The ruling is again with "
            "the number in hand.** The hard-target ruling is not "
            "withdrawn by this cell; it is reopened"
        ),
    },
    "the_tie_rule_candidates_named_not_chosen": {
        "registered": (
            "**named NOW so the ruling that follows the measurement is "
            "a CHOICE AMONG NAMED OPTIONS rather than an invention.** "
            "None is chosen here"
        ),
        "a_drop_tied_pairs": (
            "exclude them from training entirely. **Changes R-all's "
            "pair set**, which is the one thing the R-all/R-clear "
            "contrast is defined on -- so the size of that change is "
            "exactly what the measurement reports"
        ),
        "b_keep_with_p_0_5": (
            "**REINTRODUCES THE ATTRACTOR for exactly those pairs** "
            "(ATTRACTOR_FINDING): P = 0.5 is an active pull toward "
            "equal scores, so tied pairs would carry a compression "
            "term. **Recorded as such** -- this is the same mechanism "
            "that got option 3 rejected, applied to a smaller set. If "
            "it is ruled, it is ruled knowing that"
        ),
        "c_break_ties_by_a_declared_deterministic_rule": (
            "a fixed rule (patient id order, or any declared "
            "tiebreak). **Keeps the pure hard-target objective and "
            "assigns a wrong order to half the tied pairs BY "
            "CONSTRUCTION** -- which is defensible under the same "
            "positive reading that justified R-all, and must be "
            "recorded as a deliberate wrong label rather than an "
            "arbitrary convention"
        ),
    },
    "tag": "[REGISTERED] -- readings and boundary committed before the number",
}


#: **[DECLARED 2026-09-01, BEFORE IMPLEMENTATION] The tie comparison
#: rule: EXACT INTEGER EQUALITY ON THE RECOVERED RATER SUMS.**
#:
#: Two patients tie iff their **integer rater sums are equal**. The sum
#: is recovered by ``phase8c.rater_multiset`` -- **already shipped**, and
#: reused rather than reimplemented. **No float comparison and no
#: tolerance appear anywhere in the tie test.**
#:
#: **The rater count is CONFIRMED FROM THE DATA, not assumed.**
#: ``rater_multiset`` asserts that every ``5 * soft_k`` is an integer and
#: that the counts sum to five, and **refuses** otherwise -- in its own
#: words, *"a manifest with a different rater count must refuse here, not
#: render five invented dots."* So a manifest that is not five raters
#: stops the run instead of producing a tie count from invented panels.
TIE_COMPARISON_RULE_DECLARED = {
    "declared": "2026-09-01, BEFORE the task was written",
    "the_rule": (
        "**two patients TIE iff their INTEGER RATER SUMS ARE EQUAL.** "
        "S = sum of the five recovered grades, an integer in [5, 25]. "
        "**S_i == S_j is exact integer equality -- no float comparison "
        "and no tolerance appears anywhere in the tie test**"
    ),
    "it_reuses_shipped_code": (
        "**phase8c.rater_multiset ALREADY SHIPS** and is reused, not "
        "reimplemented -- a second recovery of the same quantity would "
        "be disqualifying on the standing rule. It is the same "
        "function phase8c's panels use"
    ),
    "the_rater_count_is_confirmed_not_assumed": (
        "**rater_multiset ASSERTS that every 5 * soft_k is an integer "
        "and that the counts sum to five, and REFUSES otherwise** -- in "
        "its own words, 'a manifest with a different rater count must "
        "refuse here, not render five invented dots'. A manifest that "
        "is not five raters **stops the run** rather than producing a "
        "tie count from invented panels. **The confirmation is the "
        "existing assertion, not a new check**"
    ),
    "why_the_integer_route": (
        "**it removes the question rather than bounding it, at no "
        "cost.** Since every mean is S/5, S_i == S_j holds exactly "
        "when the two means are equal AS RATIONALS -- which is what "
        "'exactly zero separation' means. Nothing is left to the CSV's "
        "decimal formatting or to float rounding"
    ),
    "the_hazard_is_real_in_principle_and_inert_here": (
        "**stated honestly: the float route would almost certainly "
        "give the same answer.** The smallest possible NON-ZERO "
        "separation is one rater-step, 1/5 = 0.2 -- about 2e14 times "
        "any plausible float error, so no borderline case can exist. "
        "**The integer rule is not chosen because floats would fail; "
        "it is chosen because it removes a question instead of "
        "bounding it**, and costs nothing to do so"
    ),
    "alternatives_considered": {
        "exact_equality_on_the_stored_floats": (
            "**would work TODAY, by an accident of formatting.** Equal "
            "rationals written by one writer parse to identical "
            "floats, so `==` succeeds -- but the rule would then "
            "silently depend on the CSV writer's decimal "
            "representation, a property this record does not control "
            "and does not check. **Correct by coincidence rather than "
            "by construction**"
        ),
        "a_declared_tolerance": (
            "**would also work, and the precedent exists**: "
            "data.softlabels.TOLERANCE = 1e-6, documented there as "
            "covering 'decimal formatting in the CSV, nothing more. It "
            "is NOT a knob.' Given the 0.2 grid, ANY tolerance between "
            "about 1e-9 and 0.1 returns the same tie count. **Not "
            "taken because it introduces a constant where none is "
            "needed** -- the integer route has no free parameter at "
            "all"
        ),
    },
    "tag": "[DECLARED] -- the rule, before the number it produces",
}


#: **[OBSERVED 2026-09-01] The tie measurement's outturn.**
#:
#: Run ``p22_tie_fraction__16b6fa2f``. **2,345 of 27,966 pairs are exact
#: ties (0.0839)** under integer equality -- **below the registered
#: 12.425% boundary**, so ``ties_are_a_small_fraction`` fired and
#: ``TARGET_MAP_RULED`` **stands as issued**.
TIE_FRACTION_OBSERVED = {
    "observed": "2026-09-01, run p22_tie_fraction__16b6fa2f",
    "provenance": "read from the run's own outputs",
    "n_ties": 2345,
    "n_pairs": 27966,
    "tie_fraction": 0.0839,
    "boundary": 0.12425,
    "cell_fired": "readings['ties_are_a_small_fraction']",
    "the_verdict": (
        "**0.0839 is BELOW the registered 0.12425 boundary, so "
        "`ties_are_a_small_fraction` fired and TARGET_MAP_RULED STANDS "
        "AS ISSUED.** Hard targets are specifiable with a declared tie "
        "rule. **The cell was not close**: 0.0839 against 0.12425 is "
        "clear of the line by a third of the boundary's own value, so "
        "no rounding question arises and nothing was stretched"
    ),
    "the_preconditions_all_held": (
        "**rater count CONFIRMED FROM THE DATA at 5 for all 237** "
        "(rater_multiset refusing anything else); **the label agreed "
        "with the integer form to EXACTLY ZERO** -- not merely within "
        "1e-9, so `mean` is the rater sum over five with no float "
        "residue at all; and **both cross-checks reproduced**: 27,966 "
        "pairs and 0.2485 below SE_diff, matching the separation run"
    ),
    "separation_profile_in_rater_steps": {
        "0": 2345, "1": 4604, "2": 4288, "3": 3790, "4": 3313, "5": 2795,
    },
    "the_profile_reproduces_all_six_banked_fractions": (
        "**a mutual consistency check neither run was designed to "
        "provide.** Every one of the separation run's six fractions "
        "falls out of this step profile exactly: se_single 1x/2x/3x = "
        "0.2485 / 0.4018 / 0.6558 and se_diff 1x/2x/3x = 0.2485 / "
        "0.5373 / 0.7557. **Two independent measurements of the same "
        "cohort, agreeing on all six thresholds** -- the integer route "
        "and the float route land in the same place, which is the "
        "strongest available evidence that the tie count is right"
    ),
    "what_it_shows_about_the_contested_set": (
        "**of the 6,949 pairs below SE_diff, 2,345 are EXACT TIES "
        "(33.75%) and 4,604 carry a REAL BUT UNCERTAIN ORDER "
        "(66.25%).** Roughly a third and two-thirds. **So the tie rule "
        "is a detail inside a larger effect, exactly as the reading "
        "anticipated** -- the majority of what distinguishes R-all "
        "from R-clear is genuinely-ordered pairs the panel cannot "
        "resolve, not pairs with no order at all"
    ),
    "the_reference_set_interval_is_RESOLVED": (
        "**[6,949, 6,950] -> EXACTLY 6,949.** The record carried that "
        "interval because it held the below-SE_diff fraction ROUNDED "
        "TO FOUR PLACES and not the count; steps 0 and 1 are the only "
        "ones below SE_diff (1.99471 rater-steps), so 2,345 + 4,604 = "
        "**6,949**. **Interval RESOLVED.** The boundary was derived as "
        "half of the ROUNDED fraction: 0.2485 / 2 = 0.12425. "
        "Recomputed from the exact count it is 6,949 / 2 / 27,966 = "
        "**0.124240** -- a difference of 1.0e-5, which at four places "
        "is 0.1242 against 0.1243. **The declared 0.12425 STANDS as "
        "the registered boundary and is not restated**: it was fixed "
        "before the number, and the measured 0.0839 is clear of either "
        "figure by a third of the boundary's own value, so nothing "
        "turns on the fifth decimal"
    ),
    "why_the_prior_missed_low": (
        "**the quantity is a STEP FUNCTION, not a continuum.** The "
        "registered prior expected 0.33 and 0.2485 landed. Panel means "
        "sit on a 0.2 grid, so pair separations do too: there is no "
        "mass anywhere between 0.2 and 0.4, and a normal approximation "
        "that spreads probability smoothly across that gap counts "
        "pairs that cannot exist. **The prior was not wrong about the "
        "spread; it was wrong about the SHAPE**, and discreteness was "
        "named in advance as the candidate reason"
    ),
    "tag": "[MEASURED]",
}


#: **[FOUND 2026-09-01, WHILE BANKING THE PROFILE] SE_diff falls 0.001058
#: below a grid point, and R-clear's pair set turns on that gap.**
#:
#: **Not a defect and not a reason to change anything -- a fragility
#: that must be on record before the arms are built.**
SE_DIFF_SITS_ON_A_KNIFE_EDGE = {
    "found": "2026-09-01, by the record, while checking the profile arithmetic",
    "the_fact": (
        "**SE_diff = 0.398942, and the label grid's next point is "
        "0.4. The gap is 0.001058.** Separations are multiples of 0.2, "
        "so 'below SE_diff' resolves to 'steps 0 and 1' -- but only "
        "just: **step 2 sits 0.001058 above the threshold and is "
        "excluded by that margin**"
    ),
    "what_turns_on_it": (
        "**R-clear's ENTIRE PAIR SET.** R-clear trains on pairs "
        "clearing SE_diff, which is steps >= 2. Had SE_diff landed "
        "just above 0.4, R-clear would be steps >= 3 and **4,288 pairs "
        "would move from R-clear's training set to the excluded "
        "side** -- and the below-threshold fraction would read 0.4018 "
        "rather than 0.2485"
    ),
    "how_little_it_would_take": (
        "**a label sd 0.265% larger.** SE_diff = sd_obs * 0.606961, so "
        "sd_obs = 0.659022 would put SE_diff at exactly 0.4, against "
        "the measured 0.657279. A different cohort draw, or a "
        "reliability figure a hair lower, flips it"
    ),
    "why_it_is_recorded_and_not_acted_on": (
        "**the threshold is DERIVED and was DECLARED before any of "
        "this was visible** (ARM_R_CLEAR_REGISTERED), and moving it "
        "now -- in either direction, for any reason -- would be "
        "choosing a pair set after seeing the distribution it selects. "
        "**The knife edge is a property of this cohort, not an error "
        "in the derivation.** It is recorded so that R-clear's result "
        "is read knowing its pair set had a coin-flip quality to it, "
        "and so nobody later presents 'separation >= two rater-steps' "
        "as though it were the designed criterion rather than what "
        "0.398942 happened to select"
    ),
    "it_does_not_touch_the_tie_ruling": (
        "ties are step 0 and are below ANY positive threshold, so the "
        "tie fraction 0.0839 is unaffected by where SE_diff falls"
    ),
    "tag": "[MEASURED] -- a fragility, reported not repaired",
}


#: **[RULED 2026-09-01] The tie rule: DROP TIED PAIRS FROM
#: TRAINING.**
#:
#: **The ground:** a tie carries no order, so under a hard-target loss
#: training on it teaches nothing true. And it is **the only candidate
#: that introduces neither an attractor nor a knowingly-wrong label.**
#:
#: It keeps R-all faithful to its purpose: **the arm that trains on
#: uncertain orders, not on absent ones.**
TIE_RULE_RULED = {
    "ruled": "2026-09-01, with the measurement in hand",
    "the_ruling": (
        "**DROP TIED PAIRS FROM TRAINING.** The 2,345 exact ties are "
        "excluded from every ranking arm's pair set; the remaining "
        "25,621 pairs carry an order and are trained on"
    ),
    "the_ground": (
        "**a tie carries NO ORDER, so under a hard-target loss "
        "training on it teaches nothing true.** And it is **the ONLY "
        "candidate that introduces NEITHER AN ATTRACTOR NOR A "
        "KNOWINGLY-WRONG LABEL** -- the other two each introduce one"
    ),
    "it_keeps_r_all_faithful_to_its_purpose": (
        "**R-all is the arm that trains on UNCERTAIN orders, not on "
        "ABSENT ones.** Its registered purpose is to carry the 24.85% "
        "of pairs the panel cannot resolve; a pair with no order at "
        "all is a different object, and including it would change what "
        "the R-all/R-clear contrast measures. **The measurement shows "
        "the cost is small**: 4,604 of the 6,949 contested pairs -- "
        "two-thirds -- survive the rule and still separate R-all from "
        "R-clear"
    ),
    "candidate_b_rejected_by_name": (
        "**P = 0.5 REJECTED: it reintroduces the attractor for exactly "
        "those pairs** (ATTRACTOR_FINDING) -- option 3's mechanism on "
        "a smaller set. A 0.5 target is an active pull toward equal "
        "scores, so 2,345 pairs would carry a compression term. "
        "**Rejected on the same ground that rejected option 3, applied "
        "consistently rather than relaxed because the set is smaller**"
    ),
    "candidate_c_rejected_by_name": (
        "**DETERMINISTIC TIE-BREAKING REJECTED: it assigns a WRONG "
        "ORDER to roughly HALF of 2,345 pairs BY CONSTRUCTION** -- "
        "about 1,173 deliberately wrong labels. **A deliberate wrong "
        "label, not a convention.** It is defensible under the same "
        "positive reading that justified R-all, and it is not taken: "
        "R-all's case is that UNCERTAIN orders are worth training on, "
        "which is not a case for INVENTED ones"
    ),
    "what_it_costs": (
        "**8.39% of all pairs leave every arm's training set**, R-syn "
        "excepted (its synthetic ordering has no ties -- four distinct "
        "magnitudes per face). The cost falls on R-all and R-clear "
        "equally in absolute terms, and R-clear loses none of its own "
        "pairs at all: **ties are step 0, so they were already below "
        "SE_diff and already outside R-clear's set.** The rule "
        "therefore changes R-all ONLY, which sharpens rather than "
        "blurs the contrast"
    ),
    "tag": "[RULED]",
}


#: **[CORRECTED 2026-09-01] The gate-3 claim outran the code.**
#:
#: Last cycle's report and the generated config headers stated: *"Gate 3
#: is satisfied BY CONSTRUCTION: weights at zero and the bias placed so
#: the untrained head predicts the training-fold mean, exactly as
#: EmbeddingHeadBackbone does it."*
#:
#: **The untrained head could not predict at all.** ``predict`` went
#: straight to ``embeddings @ self._weights``, and ``reset`` leaves
#: ``_weights`` as ``None``, so the harness's gate-3 call raised
#: ``TypeError: unsupported operand type(s) for @: 'Tensor' and
#: 'NoneType'`` on all four arms.
GATE_3_CLAIM_CORRECTED = {
    "corrected": "2026-09-01, after the third launch attempt failed",
    "the_original_claim": (
        "**PRESERVED VERBATIM**: 'Gate 3 is satisfied BY CONSTRUCTION: "
        "weights at zero and the bias placed so the untrained head "
        "predicts the training-fold mean, exactly as "
        "EmbeddingHeadBackbone does it.' Written in the cycle that "
        "built the arms, in the report and in all six generated config "
        "headers"
    ),
    "what_was_wrong_with_it": (
        "**TWO things.** (i) **'weights at zero' is false** -- reset "
        "sets `_weights = None`, because the embedding width is not "
        "known until features arrive. EmbeddingHeadBackbone does the "
        "same, so the comparison was apt and the DESCRIPTION of it was "
        "not. (ii) **'the untrained head predicts the training-fold "
        "mean' described a path that did not exist**: predict went "
        "straight to `embeddings @ self._weights` and raised on None. "
        "**The arithmetic held; the delivery did not.**"
    ),
    "where_the_bias_is_actually_set": (
        "**in `reset`**, via `bias_for_mean(mean(train_labels), "
        "bounded=...)` -- and that part was always correct. For the "
        "bounded head it is `logit((mean - 1) / 4)`, so `1 + 4 * "
        "sigmoid(bias)` returns the mean exactly; for the unbounded "
        "head the bias IS the mean. What was missing was any code path "
        "that USED it before the first train_epoch"
    ),
    "was_gate_3_ever_exercised_on_this_path": (
        "**NO.** The end-to-end test ran with `backbone: stub`, which "
        "reaches StubBackbone before make_factory's ranking branch, so "
        "`RankingHeadBackbone.predict` was never called by the harness. "
        "The unit test "
        "`test_the_bounded_bias_makes_an_untrained_head_predict_the_mean` "
        "checked `bias_for_mean` and `bounded_scores` **as functions**, "
        "never through the backbone. **The arithmetic was tested and "
        "the wiring never was**, which is precisely the gap that let "
        "the claim stand"
    ),
    "what_is_true_now": (
        "`reset` places the bias; `_weights` stays None until the first "
        "`train_epoch`; and **`predict` returns that bias for every row "
        "while `_weights` is None** -- mirroring "
        "EmbeddingHeadBackbone.predict, which has always done exactly "
        "this. Gate 3's epoch-0 prediction is therefore the "
        "training-fold mean, and the frozen gate is satisfied rather "
        "than amended"
    ),
    "the_lesson_it_repeats": (
        "**a claim about behaviour, checked only at the level of its "
        "arithmetic.** The same shape as this project's other "
        "provenance errors, and the fourth consecutive defect in this "
        "task from new code being self-consistent and untested against "
        "the interface it must satisfy"
    ),
    "tag": "[MEASURED] -- the failure, and where the claim outran it",
}


#: **[VOID 2026-09-01] The four cohort runs at 4472be00 measured ONE ARM
#: TWICE.**
#:
#: R-all and R-clear produced identical results -- seed 12345 bounded
#: **0.2497 both**, unbounded **0.1514 both**, and fold-stopping epochs
#: matching exactly (6/6/6/6/6 bounded, 11/6/6/6/7 unbounded). Two arms
#: differing by ~4,600 training pairs cannot have identical trajectories.
#:
#: **``pair_source`` reached the log line and the metrics file and
#: nothing else.** It was never placed into ``backbone_config``, so it
#: could not reach ``make_factory`` or ``RankingHeadBackbone`` -- and
#: ``ordered_pairs(labels)`` had no parameter to receive it if it had.
#: **R-clear's SE_diff restriction was never implemented at all.**
PAIR_SOURCE_WAS_NOT_CONSUMED = {
    "found": "2026-09-01, , from identical results across arms",
    "the_evidence": (
        "**seed 12345: bounded 0.2497 for BOTH R-all and R-clear; "
        "unbounded 0.1514 for BOTH.** Fold-stopping epochs identical too "
        "-- 6/6/6/6/6 bounded, 11/6/6/6/7 unbounded. **Two arms "
        "differing by ~4,600 training pairs cannot have identical "
        "training trajectories**, which is what made it visible"
    ),
    "where_the_value_stopped": (
        "**at the log line.** `pair_source` appeared in exactly four "
        "places in src/: the schema Field (validation only), two phase "
        "records (prose), `run.py` where it is LOGGED, and `run.py` "
        "where it is written into metrics.json. **It was never put into "
        "`backbone_config`**, which is the only route into the backbone, "
        "and `ordered_pairs(labels)` took labels alone -- **there was no "
        "parameter to receive it**. R-clear's restriction was not "
        "defaulted or branched past; it did not exist"
    ),
    "se_diff_threshold_too": (
        "**declared, validated, written into both R-clear configs, and "
        "read by nothing.** It appeared only in the schema Field and in "
        "prose records. EXIT_CRITERIA criterion 4 required it to be "
        "RECOMPUTED at build and CHECKED against the separation run's "
        "value; **that was never implemented either**"
    ),
    "the_runs_are_VOID_for_the_contrast": (
        "**the four cohort runs at 4472be00 are VOID for the "
        "R-all/R-clear contrast** -- two of the fifteen (the noise-pair "
        "question at each bounding). They measured one arm twice. **They "
        "are NOT void for the primaries**: R-all-bounded and "
        "R-all-unbounded are honest R-all results, and what was "
        "mislabelled R-clear is a second R-all. Nothing is banked from "
        "them either way; they are recorded as void so no later turn "
        "reads a 0.2497 pair as agreement between two arms"
    ),
    "why_the_tests_passed": (
        "**the test reported as verifying that R-all and R-clear 'differ "
        "in pair_source alone' compared TWO YAML DICTS.** It asserted "
        "the configs differ in exactly {arm, pair_source, "
        "se_diff_threshold} -- true, and silent about whether the "
        "difference reaches any code. **A config-level comparison "
        "reported as a behavioural one.** And the end-to-end test drives "
        "the real backbone through `train_epoch`, where pair "
        "construction happens, so it HAD the opportunity to observe pair "
        "counts and asserted only that the loss was finite"
    ),
    "the_class_this_belongs_to": (
        "**a value existing where it was expected and not being used "
        "where it mattered** -- the third of that shape and the fifth "
        "defect in this task. The signature audit checked that CALLS "
        "match SIGNATURES; nothing checked that DECLARED SETTINGS reach "
        "CONSUMING CODE. See SETTINGS_CONSUMPTION_SWEEP"
    ),
    "tag": "[MEASURED] -- the defect, and what it voids",
}


#: **[SWEPT 2026-09-01] Does each declared setting reach code that acts
#: on it, or only a log line?**
#:
#: Run after the third defect of that shape, over all eleven settings.
#: **Two were dead (6 and 7) and a third was worse than dead**: R-syn's
#: pair source would have fallen through to R-all's behaviour.
SETTINGS_CONSUMPTION_SWEEP = {
    "swept": "2026-09-01, all eleven declared settings, traced in code",
    "1_loss": (
        "pairwise_logistic. NOT a config field -- it is the module. "
        "Consumed at ranking.train_epoch's softplus. **LIVE**"
    ),
    "2_pair_sampling": (
        "all_pairs_per_epoch. NOT a config field. Consumed by "
        "ordered_pairs building every pair. **LIVE**"
    ),
    "3_head_bounding": (
        "`bounded` -> backbone_config -> RankingHeadBackbone.bounded -> "
        "_score. **LIVE, and proven by the runs**: bounded and unbounded "
        "differ"
    ),
    "4_monitor": (
        "`monitor` -> TrainConfig(monitor=) -> the harness's early "
        "stopping. **LIVE**"
    ),
    "5_prediction_scale": (
        "raw scores in the CSVs. Not read by any branch -- it is a "
        "PROPERTY of the unbounded head rather than a switch. "
        "**SATISFIED BY CONSTRUCTION**, and asserted in the end-to-end "
        "test rather than assumed"
    ),
    "6_pair_source": (
        "**WAS DEAD -- log line only. FIXED 2026-09-01**: -> "
        "min_separation -> backbone_config -> "
        "RankingHeadBackbone.min_separation -> ordered_pairs"
    ),
    "7_se_diff_threshold": (
        "**WAS DEAD -- declared and read by nothing. FIXED 2026-09-01**: "
        "recomputed from sd_obs and RELIABILITY_237, CHECKED against the "
        "declared value (criterion 4), and used as min_separation"
    ),
    "8_target_map": (
        "hard targets. NOT a config field -- the loss IS the hard-target "
        "form, softplus(-gap) being cross-entropy at P=1. **LIVE**"
    ),
    "9_r_syn_pretrain_epochs": (
        "UNRULED sentinel -1; the task refuses it by name. **No training "
        "code reads it, because R-syn is not implemented.** Correct for "
        "now, and it must become live before R-syn runs"
    ),
    "10_r_syn_finetune_trainability": "as 9. **REFUSAL ONLY**",
    "11_tie_rule": (
        "dropped. NOT a config field -- `difference != 0.0` in "
        "ordered_pairs. **LIVE**"
    ),
    "the_third_finding": (
        "**R-syn's `pair_source: synth_within_face` was WORSE THAN "
        "DEAD.** With pair_source finally consumed it would have fallen "
        "through to `min_separation = 0.0` -- **cohort pairs, reported "
        "as R-syn**, an arm trained on the wrong data under the right "
        "name. It now raises by name. The arm was already blocked by its "
        "sentinel, but that block would have lifted the moment settings "
        "9 and 10 were ruled, and this one does not depend on it"
    ),
    "what_the_sweep_ends": (
        "**the audits so far each checked one layer**: that calls match "
        "signatures, that references resolve, that the Protocol is "
        "satisfied. **None checked that a declared value reaches code "
        "that acts on it.** This one did, over every setting rather than "
        "the one that broke"
    ),
    "tag": "[MEASURED] -- traced in code, not recalled",
}


#: **[OBSERVED 2026-09-01] The four cohort arms, at sha 44ad1f7e.**
#:
#: **The census confirms the arms are finally distinct**: R-all 25,621
#: pairs trained, R-clear 21,017, the 4,604-pair difference exactly as
#: derived from the separation profile. The four runs at 4472be00 that
#: measured one arm twice are superseded (``PAIR_SOURCE_WAS_NOT_CONSUMED``).
#:
#: **Both bounded arms are indistinguishable from the probe.** Both
#: unbounded arms are markedly worse. **No cell is fired here** -- the
#: contrasts are evaluated by the contrast task under PLAN 4.3.
COHORT_ARMS_OBSERVED = {
    "observed": "2026-09-01, four runs at sha 44ad1f7e",
    "provenance": (
        "**per-seed PCCs supplied  from the runs' own "
        "outputs; means and sds COMPUTED here from those five values "
        "per arm.** The run directories are CLUSTER-ONLY -- `runs/keeper` "
        "does not exist on the laptop -- so the artifacts could not be "
        "read directly. This is the same provenance every other banked "
        "figure in this project carries, and it is stated rather than "
        "implied"
    ),
    "arms": {
        "r_all_bounded": {
            "per_seed": [0.2794, 0.2384, 0.2432, 0.2388, 0.2497],
            "mean": 0.2499, "sd": 0.0171,
        },
        "r_clear_bounded": {
            "per_seed": [0.2762, 0.2367, 0.2421, 0.2387, 0.2509],
            "mean": 0.2489, "sd": 0.0162,
        },
        "r_all_unbounded": {
            "per_seed": [0.1645, 0.1037, 0.1442, 0.0833, 0.1514],
            "mean": 0.1294, "sd": 0.0344,
        },
        "r_clear_unbounded": {
            "per_seed": [0.1854, 0.0929, 0.1431, 0.0730, 0.1574],
            "mean": 0.1304, "sd": 0.0464,
        },
    },
    "the_census_confirms_distinct_pair_sets": (
        "**R-all 25,621 pairs trained, R-clear 21,017** -- a difference "
        "of 4,604, which is exactly the below-SE_diff non-tied count "
        "from the separation profile (step 1 = 4,604). The two arms are "
        "finally different arms, and the count is in the record rather "
        "than only in the configs"
    ),
    "what_the_point_estimates_show": (
        "**bounded: 0.2499 and 0.2489 against the probe's 0.2520** -- "
        "deltas -0.0021 and -0.0031, an order of magnitude inside the "
        "0.04-0.10 band this cohort cannot resolve. **Unbounded: 0.1294 "
        "and 0.1304**, deltas -0.1226 and -0.1216 -- large, and still "
        "short of the 0.1386 that is the smallest delta this project "
        "has ever resolved. **No verdict is read here**; the contrast "
        "task applies the two-condition criterion"
    ),
    "the_seed_spread_differs_between_the_heads": (
        "**bounded sd 0.0171 / 0.0162; unbounded sd 0.0344 / 0.0464** -- "
        "roughly two to three times wider. Reported as an observation, "
        "not a finding: an unbounded score has no label scale pinning "
        "it, so wider seed spread is unsurprising, and nothing was "
        "registered in advance about it"
    ),
    "r_all_versus_r_clear_is_tiny_at_both_bounds": (
        "**per-seed mean difference +0.0010 bounded, -0.0009 "
        "unbounded** -- and the per-seed signs are mixed in both "
        "(+/+/+/+/- and -/+/+/+/-). The noise-pair question looks "
        "headed for unresolved at both bounds, which is what the "
        "record predicted for every contrast on this cohort"
    ),
    "tag": "[MEASURED]",
}


#: **[RECORDED 2026-09-01] Only SEVEN of the fifteen contrasts can be
#: evaluated; the other eight wait on R-syn.**
#:
#: The two R-syn arms are unrun, blocked on settings 9 and 10
#: (``DECLARED_SETTINGS_22``) and, since the settings sweep, on R-syn's
#: pair source being unimplemented.
FAMILY_EVALUABLE_SUBSET = {
    "recorded": "2026-09-01, after the four cohort arms ran",
    "evaluable_now": {
        "four_primaries": (
            "R-all-bounded, R-clear-bounded, R-all-unbounded and "
            "R-clear-unbounded, each against the 0.2520 probe"
        ),
        "two_head_contrasts": (
            "R-all-bounded vs R-all-unbounded, and R-clear-bounded vs "
            "R-clear-unbounded. **The third head contrast (R-syn) "
            "waits** -- so the head question is answered on the cohort "
            "arms only, and any reading of it must say so"
        ),
        "two_noise_pair_contrasts": (
            "R-all vs R-clear at bounded, and at unbounded. **Both "
            "members exist, so this question is fully evaluable** -- it "
            "is the only one of the three that is"
        ),
        "count": 8,
    },
    "the_count_is_eight_not_seven": (
        "**four primaries + two head + two noise-pair = 8.** Stated "
        "because the request enumerated 'the four primaries, the two "
        "head contrasts, and the noise-pair contrast at each bounding', "
        "which is the same set and sums to eight rather than seven"
    ),
    "waiting_on_r_syn": (
        "**SEVEN**: R-syn's two primaries, its one head contrast, and "
        "all four axis contrasts. **The axis question is entirely "
        "unevaluable**, since every one of its four contrasts has R-syn "
        "on one side"
    ),
    "nothing_is_added_or_dropped": (
        "the family stays at FIFTEEN (CONTRAST_FAMILY). Eight are "
        "evaluated now and seven are DEFERRED, not withdrawn -- a "
        "deferred contrast still owes a verdict, and criterion 8 "
        "requires one for all fifteen before the phase closes"
    ),
    "tag": "[RECORDED]",
}


#: **[AUDITED 2026-09-01] What the post-run configs RECONSTRUCTED rather
#: than read.**
#:
#: The four run-directory paths were built from the
#: ``<stem>__<sha8>__<job-id>`` contract and were wrong: every real name
#: ends ``-2``, which no contract predicts. **Twelve entries across three
#: configs pointed at directories that do not exist.**
#:
#: Same class as the invented input references and the invented
#: ``PooledSource`` call: **a plausible construction standing in for a
#: read one.** So the rest of the file was audited in the same pass,
#: before the declare rather than after.
POSTRUN_RECONSTRUCTION_AUDIT = {
    "audited": "2026-09-01, every value in the three post-run configs",
    "the_defect": (
        "**run-directory paths CONSTRUCTED from the naming contract.** "
        "The job-id fragment is whatever the maintainer set at launch -- "
        "here `p22-r-all-bounded-2`, with a `-2` suffix nothing in the "
        "repo predicts. **A fact only the run knows, and the generator "
        "guessed it.** Twelve entries, all wrong, none detectable on "
        "the laptop because the directory is cluster-only"
    ),
    "also_reconstructed_and_now_derived": {
        "seeds": (
            "typed as [1337, 2024, 7, 99, 12345]. **Right, and not "
            "read.** Now read from the four arm configs, and required "
            "to agree across all four"
        ),
        "n_patients": (
            "typed as 237. Now read from the arm configs' "
            "expect_patients, likewise required to agree"
        ),
        "winner_sd": (
            "typed as 0.0148. Now read from "
            "ladder.TRADE_OFF_PAIR['result']['vit_sd']"
        ),
        "the_probe_stem": (
            "**the one that would have mattered most.** "
            "`p7_d1_vit_b16_imagenet_g1` was TYPED as the probe's run "
            "directory. It is correct -- ladder.TRADE_OFF_PAIR's "
            "best_arm.stem, recorded_pcc 0.252 -- but it was asserted, "
            "not derived, and **had it been wrong every primary "
            "contrast would have compared against the wrong "
            "baseline**. Now read from ladder"
        ),
        "the_kappa_thresholds": (
            "typed as 0.7012 / 0.5729 / 0.0956 / -0.0743. The task "
            "already checked them against phase21 at run time, so a "
            "drifted value would have stopped the phase -- but the "
            "config can now simply not be wrong. Read from "
            "phase21.ARM_B_OBSERVED and "
            "ARM_A_THE_SHRINKAGE_CONTROL['kappa_measured']"
        ),
        "the_csv_stem": (
            "typed as 'predictions', copied from the donor's "
            "convention. **The first correction attempt read it from "
            "the donor and found TWO stems** -- 'predictions' and "
            "'identity_predictions' -- so there was no single "
            "convention to copy. Now derived from the WRITER: "
            "phase3.write_outputs emits `<prefix>predictions.csv`, "
            "checked in its source"
        ),
    },
    "reconstructed_and_still_reconstructed": {
        "run_root": (
            "`runs/keeper/p22` was built from the tier-and-phase "
            "convention. **It happens to be right** and it remains "
            "constructed: the four supplied names are leaf names, so a "
            "root is still needed. Named here rather than left implicit"
        ),
        "group_label": (
            "`p22_ranking` is **INVENTED**. Nothing derives it and "
            "nothing checks it -- it is a grouping key in the report. "
            "Recorded as invented rather than dressed as read"
        ),
    },
    "how_such_names_are_obtained_in_future": (
        "**SUPPLIED FROM AN `ls`, recorded verbatim in the "
        "generator's RUN_DIRECTORIES table.** The laptop cannot list "
        "runs/keeper -- it is cluster-only -- so there is no honest way "
        "to derive them here, and the convenient way is the one that "
        "failed. The generator now REFUSES to emit an arm whose name is "
        "not in that table, rather than falling back to the contract"
    ),
    "tag": "[MEASURED] -- traced value by value, not recalled",
}


#: **[REPORTED 2026-09-01] Can a wrong run-dir path be caught locally?
#: Partly.**
RUN_PATH_CHECK_FEASIBILITY = {
    "reported": "2026-09-01",
    "what_is_impossible": (
        "**a load-time existence check.** A config declaring "
        "`runs/keeper/...` names a CLUSTER-ONLY directory; the laptop "
        "cannot stat it, so `schema.validate` cannot verify it exists. "
        "This is not a gap that better checking closes -- the "
        "information is not here"
    ),
    "the_real_guard_is_guard_3": (
        "**at launch, on the cluster.** Guard 3 resolves every declared "
        "input and refuses a missing one. That is what would have "
        "caught these twelve paths, and it is the only check that can "
        "see the truth. Stated plainly rather than implying the repo "
        "layer could have caught it"
    ),
    "what_IS_checkable_locally_and_is_now_checked": (
        "**that the generator did not CONSTRUCT the name.** It emits "
        "only leaf names present in RUN_DIRECTORIES and raises "
        "otherwise, and a test asserts no run-directory name is built "
        "by string interpolation anywhere in the generator. **A "
        "constructed path cannot be verified; a constructed path can be "
        "PREVENTED**, and that is the half that lives on this machine"
    ),
    "the_residual_risk": (
        "a name supplied wrongly -- mistyped in the `ls`, or from the "
        "wrong sha -- still reaches the config and is caught only by "
        "guard 3. **The check moves the failure from 'plausible "
        "invention' to 'transcription of something real', which is a "
        "smaller class, not an empty one**"
    ),
    "tag": "[REASONED]",
}


#: **[OBSERVED 2026-09-02, sha 43ad4a00] Shrinkage on the four cohort
#: arms -- and the registered prediction is REFUTED.**
#:
#: The prediction (``BOUNDED_HEAD_PREDICTION_REGISTERED``) was that a
#: bounded ranking arm shrinks toward the label mean like the other 63,
#: and an unbounded one does not. **The second half held. The first was
#: wrong, and wrong in a direction nobody registered.**
SHRINKAGE_OBSERVED_AND_PREDICTION_REFUTED = {
    "observed": "2026-09-02, sha 43ad4a00",
    "provenance": "read from the run's own outputs",
    "measured": {
        "r_all_bounded": {"ratio": 1.0385, "sd": 0.0071},
        "r_clear_bounded": {"ratio": 1.0799, "sd": 0.0107},
        "r_all_unbounded": {"ratio": 4.8239, "sd": 2.6011},
        "r_clear_unbounded": {"ratio": 5.2569, "sd": 2.6003},
    },
    "the_prediction_is_refuted": (
        "**the record predicted a BOUNDED ranking arm would shrink toward "
        "the label mean like the other 63. It does not.** 1.0385 and "
        "1.0799 against the probe's 0.4947 and the ladder median "
        "0.3413 -- **not shrunk, CALIBRATED.** A ratio of ~1.04 means "
        "the arm's prediction spread matches the label spread almost "
        "exactly, which **no arm in the ladder achieves**. Recorded in "
        "the direction it actually went rather than as a near-miss: "
        "bounding did not reintroduce shrinkage, it produced near-"
        "perfect scale calibration"
    ),
    "the_half_that_held": (
        "the UNBOUNDED arms do not shrink -- 4.8239 and 5.2569, five "
        "times the label spread. That was predicted, and it is the "
        "less interesting half: an unbounded score has no label scale "
        "to sit on, so its spread is arbitrary"
    ),
    "the_widest_seed_variation_in_the_project": (
        "**the unbounded arms' per-seed ratios run 2.59 to 8.96, sd "
        "~2.60.** Nothing else in this record varies that far across "
        "seeds. Reported as an observation with no cell attached: "
        "nothing was registered about seed spread, and a figure this "
        "large invites a story it has not earned. It is consistent "
        "with an unpinned scale, which is a reading and not a finding"
    ),
    "it_fired_no_cell_and_gated_nothing": (
        "**the prediction was registered TO BE REFUTED and was.** It "
        "carried no criterion, decided nothing, and its being wrong "
        "changes no verdict. What it bought is that the "
        "bounded-versus-unbounded question was settled by measurement "
        "rather than by the expectation -- which is exactly the "
        "ground the ruling was six arms on"
    ),
    "tag": "[MEASURED]",
}


#: **[RECORDED 2026-09-02] Five arms now do not shrink, and none of them
#: beats the probe.**
#:
#: The pattern in its strongest available form.
FIVE_ARMS_DO_NOT_SHRINK = {
    "recorded": "2026-09-02, after the four cohort arms' shrinkage landed",
    "the_five": {
        "p17_arm_a": {"shrinkage": 2.1730, "pcc": 0.2334},
        "r_all_bounded": {"shrinkage": 1.0385, "pcc": 0.2499},
        "r_clear_bounded": {"shrinkage": 1.0799, "pcc": 0.2489},
        "r_all_unbounded": {"shrinkage": 4.8239, "pcc": 0.1294},
        "r_clear_unbounded": {"shrinkage": 5.2569, "pcc": 0.1304},
    },
    "the_probe": {"shrinkage": 0.4947, "pcc": 0.2520},
    "what_it_establishes": (
        "**the probe SHRINKS to 0.4947 and scores 0.2520 -- above all "
        "five.** Two of the five are at near-perfect calibration "
        "(1.0385, 1.0799) and still do not beat it; two are at five "
        "times the label spread and score half as well; arm A expands "
        "and matches nobody. **NOT SHRINKING BUYS NOTHING ON PCC**, "
        "and this is the strongest form the record can put it in: not "
        "an invariance argument, not one exceptional arm, but five "
        "arms spanning calibration ratios from 1.04 to 5.26, none of "
        "which reaches a shrinking probe"
    ),
    "why_the_calibrated_pair_is_the_strongest_case": (
        "**a reader could dismiss arm A and the unbounded arms as "
        "degenerate** -- one has no label mean, two have no scale at "
        "all. The bounded ranking arms are neither: they sit on the "
        "1-5 scale, they are calibrated better than anything in the "
        "ladder, and they land 0.002 and 0.003 below the probe. "
        "**There is no degeneracy left to blame**"
    ),
    "what_it_does_not_establish": (
        "**not that calibration is worthless** -- calibration is not "
        "PCC, and an arm whose spread matches the labels may be worth "
        "more to a clinician than one that does not. The claim is "
        "narrow and is exactly the registered one: shrinkage does not "
        "explain the PCC ceiling, and removing it does not raise PCC"
    ),
    "tag": "[MEASURED] -- five arms, one probe, one conclusion",
}


#: **[OBSERVED 2026-09-02] The feature-difference diagnostic: all eight
#: cells `kappa_between`, and the SPLIT BY HEAD is the phase's answer.**
DIAGNOSTIC_OBSERVED = {
    "observed": "2026-09-02, sha 43ad4a00",
    "provenance": "read from the run's own outputs",
    "measured": {
        "residual_sign": {
            "r_all_bounded": 0.4398, "r_clear_bounded": 0.4134,
            "r_all_unbounded": 0.1598, "r_clear_unbounded": 0.1213,
        },
        "worst_quartile": {
            "r_all_bounded": 0.2578, "r_clear_bounded": 0.2294,
            "r_all_unbounded": 0.0070, "r_clear_unbounded": -0.0062,
        },
    },
    "every_cell_is_between": (
        "**all eight fired `kappa_between`**, the cell that says report "
        "the value and claim neither. The registered band is wide "
        "(0.0956 to 0.7012 on residual sign) and was left wide "
        "deliberately -- those are the two levels the project has "
        "measured, and inventing a midpoint would have been the choice "
        "the derivation avoided. **The cell is honoured: no claim is "
        "read off any single value**"
    ),
    "but_the_split_by_head_is_the_answer": (
        "**bounded arms partly share the 63's error structure** -- "
        "0.4398 / 0.4134 on residual sign, 0.2578 / 0.2294 on worst "
        "quartile. **Unbounded arms barely share it at all** -- 0.1598 "
        "/ 0.1213, and on worst quartile **0.0070 and -0.0062, sitting "
        "at arm A's control level.** The split is between the HEADS, "
        "not between the pair sets, and it is large: the bounded arms "
        "are 3x the unbounded on residual sign and ~37x on worst "
        "quartile"
    ),
    "what_it_establishes_against_the_hypothesis": (
        "**UNBOUNDED RANKING DID FIND DIFFERENT FEATURES, AND THE "
        "DIFFERENT FEATURES ARE WORSE.** The phase's narrowed "
        "hypothesis was that ordering-based training MAY FIND "
        "DIFFERENT FEATURES (PHASE_22_RECKONING). The unbounded arms "
        "did: their worst-quartile agreement with the 63 is "
        "indistinguishable from arm A's chance level. **And they score "
        "0.1294 and 0.1304 against the probe's 0.2520.** Different is "
        "not better. **This is a direct answer to the hypothesis as "
        "registered, and it is NEGATIVE**"
    ),
    "the_bounded_arms_answer_it_the_other_way": (
        "they partly share the error structure AND match the probe's "
        "PCC to within 0.003. **So the head that scores well is the "
        "head that makes similar errors**, which is the same finding "
        "from the other side: on this cohort, agreeing with the "
        "shrinking arms and scoring like them go together"
    ),
    "the_cells_fired_are_post_hoc_free": (
        "the thresholds were read from phase21 at run time and matched "
        "their declared values; no cell was chosen after the numbers, "
        "and the between-cell was pre-registered to claim nothing"
    ),
    "tag": "[MEASURED]",
}


#: **[OBSERVED 2026-09-02] The eight evaluable contrasts: EIGHT
#: UNRESOLVED.**
CONTRASTS_OBSERVED = {
    "observed": "2026-09-02, sha 43ad4a00; re-run after the condition fix",
    "provenance": (
        "deltas read from the run's own outputs. **The "
        "per-contrast condition_1/condition_2 flags come from the "
        "RE-RUN**, which populates them; the deltas are unchanged by "
        "that fix, which touches only what is recorded"
    ),
    "deltas": {
        "primary__r_all_bounded_vs_probe": -0.0021,
        "primary__r_all_unbounded_vs_probe": -0.1226,
        "primary__r_clear_bounded_vs_probe": -0.0031,
        "primary__r_clear_unbounded_vs_probe": -0.1217,
        "secondary__head__r_all": +0.1205,
        "secondary__head__r_clear": +0.1186,
        "secondary__noise_pairs__bounded": +0.0010,
        "secondary__noise_pairs__unbounded": -0.0009,
    },
    "verdict": "**EIGHT UNRESOLVED. Nothing claimable.**",
    "the_noise_pair_result_is_the_phases_cleanest_null": (
        "**+0.0010 bounded and -0.0009 unbounded.** The R-all/R-clear "
        "contrast is the one question in the family isolated by a "
        "SINGLE controlled factor, and **dropping 4,604 "
        "uncertain-order pairs changed nothing at either head.** Not "
        "'too small to resolve' in the usual sense -- these are two "
        "orders of magnitude below the resolvable floor, at a "
        "thousandth of a PCC point. The naive construction and the "
        "careful one are the same arm to within measurement"
    ),
    "the_four_largest_deltas_are_all_about_0_12_and_all_failed": (
        "**[CORRECTED 2026-09-02 before banking.** The first draft read "
        "'+0.1205 and +0.1186 are the largest deltas in the family'. "
        "**They are not**: |-0.1226| and |-0.1217|, the two unbounded "
        "primaries, are larger. The head contrasts are the largest "
        "SECONDARIES and the largest POSITIVE deltas. Caught by a test "
        "asserting the maximum rather than by re-reading the sentence "
        "-- the same shape as the |-0.8470| > |+0.7521| correction.**\n"
        "By magnitude: **0.1226, 0.1217 (unbounded primaries), 0.1205, "
        "0.1186 (head contrasts)**, then a gap to 0.0031. **All four "
        "sit just under the 0.1386 this cohort has ever resolved** "
        "(ladder.SMALLEST_RESOLVABLE_DIFFERENCE) and all four are "
        "UNRESOLVED. **Every real effect the phase produced fell in "
        "the band the cohort cannot see** -- the limit doing exactly "
        "what the record predicted"
    ),
    "the_four_are_one_effect_seen_twice": (
        "the two unbounded primaries measure unbounded-against-probe; "
        "the two head contrasts measure bounded-against-unbounded, and "
        "the bounded arms sit on the probe. **So all four are the same "
        "gap -- the unbounded head's cost -- read from two directions**, "
        "which is why they cluster at 0.12 rather than four "
        "independent findings landing together"
    ),
    "the_primaries": (
        "bounded arms -0.0021 and -0.0031 against the probe: "
        "indistinguishable. Unbounded arms -0.1226 and -0.1217: "
        "markedly worse and, like the head contrasts, just short of "
        "the resolvable floor. **A loss this project cannot claim is "
        "still not a loss it may report as nothing**"
    ),
    "tag": "[MEASURED]",
}


#: **[OBSERVED 2026-09-02, RE-RUN] The eight verdicts with both
#: conditions populated.**
#:
#: The re-run after the condition fix. **Eight UNRESOLVED**, and the
#: split within them is the point: **four fail condition 1 while PASSING
#: condition 2**, four fail both.
CONTRAST_VERDICTS_OBSERVED = {
    "observed": "2026-09-02, the re-run after the condition fix",
    "provenance": (
        "supplied  from the re-run's own outputs. **The "
        "deltas are unchanged from the earlier run** -- the fix touched "
        "only what is recorded, not what is computed"
    ),
    "verdict": "**ALL EIGHT UNRESOLVED. Nothing claimable.**",
    "condition_2_pass_condition_1_fail": {
        "primary__r_all_unbounded_vs_probe": -0.1226,
        "primary__r_clear_unbounded_vs_probe": -0.1217,
        "secondary__head__r_all": +0.1205,
        "secondary__head__r_clear": +0.1186,
    },
    "fail_both_conditions": {
        "primary__r_all_bounded_vs_probe": -0.0021,
        "primary__r_clear_bounded_vs_probe": -0.0031,
        "secondary__noise_pairs__bounded": +0.0010,
        "secondary__noise_pairs__unbounded": -0.0009,
    },
    "the_split_is_the_four_largest_against_the_four_smallest": (
        "the four that pass condition 2 are exactly the four ~0.12 "
        "deltas; the four that fail both are the four at or below "
        "0.0031. **No contrast sits between** -- the gap runs from "
        "0.1186 down to 0.0031, a factor of nearly forty"
    ),
    "why_this_is_the_clearest_case_for_BOTH_conditions": (
        "**deltas three to seven times their own thresholds, failing "
        "on 1 of 5 and 2 of 5 seeds.** A single-condition criterion "
        "would have called all four claimable on a margin that looks "
        "commanding. **Condition 1 is what refuses them**, and it "
        "refuses them on the only ground that matters: the effect is "
        "not there in most of the seeds. The record has eight prior "
        "instances of this shape and **none with a margin this "
        "large** -- the previous maximum is 5.1x (ledger index 34, "
        "p17-c-vs-probe), and the previous seed counts run 0, 2 and 3 "
        "of 5. **A 1-of-5 failure at seven times threshold is the "
        "sharpest illustration in the project that a large mean delta "
        "and a real effect are different claims**"
    ),
    "tag": "[MEASURED]",
}


#: **[RECORDED 2026-09-02] Twelve and eight are not a contradiction.**
#:
#: The condition-2-pass/condition-1-fail phenomenon now has **TWELVE**
#: project-wide instances. ``results_ledger`` entry 38 says **EIGHT**,
#: and **entry 38 is still correct**.
CONDITION_PHENOMENON_COUNT_RECONCILED = {
    "recorded": "2026-09-02, at the Phase 22 close-out",
    "the_two_numbers": (
        "**TWELVE instances of the phenomenon project-wide. EIGHT rows "
        "in the ledger.** Both are right, and a reader meeting them a "
        "year apart could easily read them as a contradiction -- which "
        "is why the distinction is written down rather than left to be "
        "re-derived"
    ),
    "what_entry_38_counts": (
        "**LEDGER ROWS.** Entry 38's own words: 'derived from the "
        "ledger's own condition fields, EIGHT rows have condition 1 "
        "FALSE and condition 2 TRUE -- indices 25, 26, 27, 28, 31, 32, "
        "34, 35'. **Verified independently at this close-out** by "
        "filtering the ledger's condition fields: exactly those eight "
        "indices, no more. Entry 38 stands unchanged and needs no "
        "correction"
    ),
    "why_phase_22_adds_none": (
        "**Phase 22's contrasts are DESCRIPTIVE and registered NO "
        "LEDGER ROWS.** The family was declared with no ledger row "
        "from the start; the ledger is at 38 entries before and after "
        "this phase. So the four new instances are instances of the "
        "PHENOMENON without being rows in the LEDGER, and entry 38's "
        "count is untouched by them"
    ),
    "the_arithmetic": (
        "8 ledger rows + 4 Phase 22 contrasts = **12 instances**. The "
        "eight: indices 25, 26, 27, 28 (Phases 10-12), 31, 32, 34, 35 "
        "(Phases 16-17). The four: both unbounded primaries and both "
        "head contrasts"
    ),
    "how_to_quote_either": (
        "**say which you mean.** 'Eight rows in the ledger' and "
        "'twelve instances in the project' are both quotable and "
        "neither may be written as the other. A future write-up "
        "wanting one number should take TWELVE for the phenomenon and "
        "EIGHT for the ledger, and say so in the sentence"
    ),
    "tag": "[MEASURED] -- both counts verified at source",
}


#: **[CLOSED 2026-09-02] PHASE_22_CLOSING.**
PHASE_22_CLOSING = {
    "closed": (
        "2026-09-02, four cohort arms run, two R-syn arms blocked. "
        "**The verdicts cited here are the RE-RUN's** "
        "(CONTRAST_VERDICTS_OBSERVED), not the earlier contrast run"
    ),
    "which_run_the_verdicts_come_from": (
        "**the re-run, after the condition fix.** The earlier contrast "
        "run is **SUPERSEDED FOR THE VERDICTS ONLY**: it recorded "
        "`condition_1: None` on all eight, so its verdict rows cannot "
        "be read. **Its DELTAS were correct and are unchanged** -- the "
        "fix altered what was recorded, not what was computed, and the "
        "re-run reproduces every delta exactly. Superseded is not void: "
        "nothing measured there was wrong"
    ),
    "the_finding": (
        "**BOUNDED RANKING MATCHES THE PROBE TO WITHIN 0.003 AND "
        "CALIBRATES BETTER THAN ANY ARM IN THE LADDER, AND BUYS "
        "NOTHING MEASURABLE ON PCC. UNBOUNDED RANKING FINDS DIFFERENT "
        "FEATURES AND THEY ARE WORSE. THE NOISE-PAIR QUESTION IS A "
        "CLEAN NULL.** Three sentences, all negative or null, and all "
        "three answer questions that were registered before any arm "
        "ran"
    ),
    "against_the_narrowed_hypothesis": (
        "the phase was restated on the weaker claim that "
        "ordering-based training MAY FIND DIFFERENT FEATURES, after "
        "SCALE_INVARIANCE_PROHIBITION and arm A closed the shrinkage "
        "mechanism. **The diagnostic answered it directly: the "
        "unbounded arms did find different features -- worst-quartile "
        "agreement at arm A's chance level -- and scored half as "
        "well.** Different is not better. The hypothesis was tested "
        "and it failed"
    ),
    "criterion_walk": {
        "1_reckoning_at_the_top": (
            "PHASE_22_RECKONING and "
            "THE_STRONGER_MECHANISM_ASSERTED are in the record, "
            "with the refuted mechanism and the order of events stated"
        ),
        "2_six_arms_built": (
            "**PARTIAL, and stated as such.** Six built, FOUR RUN. "
            "The two R-syn arms are blocked on settings 9 and 10 and "
            "on their pair source being unimplemented"
        ),
        "3_one_controlled_factor": (
            "MET, and verified in code after it failed once: "
            "pair_source reached only a log line, making R-all and "
            "R-clear the same arm. The census now proves distinct pair "
            "sets -- 25,621 against 21,017"
        ),
        "4_threshold_recomputed": (
            "MET: se_diff is recomputed from sd_obs and "
            "RELIABILITY_237 at run time and refused on mismatch"
        ),
        "5_diagnostic_on_every_arm": (
            "MET for the four that ran; both binarisations, reported "
            "separately"
        ),
        "6_thresholds_from_phase21": (
            "MET: read at run time and checked against the config's "
            "declared values"
        ),
        "7_fifteen_and_no_more": (
            "MET: the family is fifteen, eight evaluated, seven "
            "deferred. Nothing added"
        ),
        "8_every_contrast_gets_a_verdict": (
            "**NOT MET, and it is the reason the phase does not close "
            "completely.** Eight have verdicts; **SEVEN ARE STILL "
            "OWED ONE** and are DEFERRED, not withdrawn"
        ),
        "9_axis_assumption_travels": (
            "vacuous so far -- no R-syn result exists to attach it to. "
            "It remains binding for when one does"
        ),
        "10_no_gain_attributed_to_shrinkage": (
            "MET, and reinforced: FIVE_ARMS_DO_NOT_SHRINK puts the "
            "negative on measured ground rather than on the "
            "invariance alone"
        ),
        "11_the_expected_null_reported_as_a_null": (
            "MET. Eight unresolved, reported as eight unresolved, with "
            "the point estimates and no claim -- and **with both PLAN "
            "4.3 conditions on every row**, so a reader can see WHICH "
            "condition failed rather than inferring it. Four fail "
            "condition 1 while passing condition 2 "
            "(CONTRAST_VERDICTS_OBSERVED)"
        ),
        "12_nothing_added_after_the_lock": (
            "MET. No arm, contrast, reading, threshold or setting was "
            "introduced once numbers existed"
        ),
        "13_open_settings_ruled_before_the_run": (
            "**HELD AS A BLOCK, which is the criterion working.** "
            "Settings 9 and 10 were never ruled, so the R-syn arms "
            "refuse at load. The criterion stopped them rather than "
            "letting them default"
        ),
        "14_unbounded_limitation_travels": (
            "MET: MONITOR_BIND_RESOLVED rides in every unbounded arm's "
            "metrics.json, and the bounded/unbounded PCC difference is "
            "reported without attributing it to the selection effect"
        ),
    },
    "the_phase_does_not_fully_close": (
        "**criterion 8 is unmet: seven contrasts are deferred and "
        "still owe a verdict.** Phase 22 closes on the COHORT arms and "
        "stays open on the axis question. Saying it closed would be "
        "the shape this project has corrected repeatedly -- a "
        "criterion reported as met because the part that ran went well"
    ),
    "what_was_void_and_stays_void": (
        "PAIR_SOURCE_WAS_NOT_CONSUMED: the four runs at 4472be00 "
        "measured one arm twice and are VOID for the R-all/R-clear "
        "contrast. Superseded by 44ad1f7e and 43ad4a00, and kept on "
        "record so no later turn reads a 0.2497 pair as agreement"
    ),
    "the_defects_this_phase_cost": (
        "**six, all in one task, all caught after a launch**: an "
        "invented input reference pair, a constructor called against "
        "an assumed signature, a gate-3 claim that outran its code, a "
        "declared setting that reached only a log line, four "
        "reconstructed run-dir paths, and a computed condition that "
        "never reached the record. **Each produced a check that now "
        "runs in the suite.** Recorded here because a phase's cost is "
        "part of its result"
    ),
    "tag": "[CLOSED, PARTIAL]",
}


#: The six arms, as the configs name them.
ARMS = (
    "r_syn_bounded", "r_syn_unbounded",
    "r_all_bounded", "r_all_unbounded",
    "r_clear_bounded", "r_clear_unbounded",
)

#: The probe every primary is measured against (``ladder.TRADE_OFF_PAIR``).
PROBE = "vit_paired_mean"


def contrast_family() -> list[dict]:
    """The fifteen contrasts, enumerated. **No more and no fewer.**

    Derived from ``CONTRAST_FAMILY`` rather than typed out, so the count
    cannot drift from the record that fixes it: six primaries, then the
    three secondary groups. **No contrast crosses the bounding except
    the three that exist to measure it**, which is what keeps every
    other one to a single controlled factor.
    """
    contrasts = [
        {"key": f"primary__{arm}_vs_probe", "kind": "primary",
         "question": "does ordering-based training raise PCC",
         "a": arm, "b": PROBE, "varies": "objective"}
        for arm in ARMS
    ]
    for source in ("r_syn", "r_all", "r_clear"):
        contrasts.append({
            "key": f"secondary__head__{source}",
            "kind": "secondary", "question": "the head question",
            "a": f"{source}_bounded", "b": f"{source}_unbounded",
            "varies": "bounding",
        })
    for head in ("bounded", "unbounded"):
        contrasts.append({
            "key": f"secondary__noise_pairs__{head}",
            "kind": "secondary", "question": "the noise-pair question",
            "a": f"r_all_{head}", "b": f"r_clear_{head}",
            "varies": "pair_set",
        })
    for head in ("bounded", "unbounded"):
        for cohort in ("r_all", "r_clear"):
            contrasts.append({
                "key": f"secondary__axis__{cohort}_{head}",
                "kind": "secondary", "question": "the axis question",
                "a": f"r_syn_{head}", "b": f"{cohort}_{head}",
                "varies": "pairing_axis",
            })
    return contrasts


def summary() -> dict:
    """The phase's restate, importable as one object."""
    return {
        "reckoning": PHASE_22_RECKONING,
        "the_stronger_mechanism_asserted": (
            THE_STRONGER_MECHANISM_ASSERTED
        ),
        "coverage": COVERAGE_NOTHING_TESTS_ORDERING,
        "arm_r_syn": ARM_R_SYN_REGISTERED,
        "arm_r_all": ARM_R_ALL_REGISTERED,
        "arm_r_clear": ARM_R_CLEAR_REGISTERED,
        "r_all_r_clear_ground": THE_R_ALL_R_CLEAR_GROUND,
        "pairing_axis_assumption": PAIRING_AXIS_ASSUMPTION,
        "diagnostic": FEATURE_DIFFERENCE_DIAGNOSTIC,
        "contrast_family": CONTRAST_FAMILY,
        "readings": READINGS_BOTH_WAYS,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        "config_surface": CONFIG_SURFACE_REPORT,
        # [2026-09-01] The five rulings, the lock, and what they settled.
        "loss_ruled": LOSS_RULED,
        "pair_construction_ruled": PAIR_CONSTRUCTION_RULED,
        "head_ruled": HEAD_RULED,
        "axis_ruled": AXIS_RULED,
        "head_bounding_ruled": HEAD_BOUNDING_RULED,
        "bounded_head_prediction": BOUNDED_HEAD_PREDICTION_REGISTERED,
        "monitor_bind": MONITOR_BIND_RESOLVED,
        "rescaling_note": RESCALING_NOTE,
        "declared_settings": DECLARED_SETTINGS_22,
        "exit_criteria": EXIT_CRITERIA,
        # [2026-09-01] The target map, the finding that shaped it, and
        # the measurement it is conditional on.
        "target_map_ruled": TARGET_MAP_RULED,
        "attractor_finding": ATTRACTOR_FINDING,
        "tie_fraction_measurement": TIE_FRACTION_MEASUREMENT_DESIGNED,
        "tie_comparison_rule": TIE_COMPARISON_RULE_DECLARED,
        # [2026-09-01] The measurement landed; the tie rule is ruled.
        "tie_fraction_observed": TIE_FRACTION_OBSERVED,
        "se_diff_knife_edge": SE_DIFF_SITS_ON_A_KNIFE_EDGE,
        "tie_rule_ruled": TIE_RULE_RULED,
        "gate_3_claim_corrected": GATE_3_CLAIM_CORRECTED,
        "pair_source_was_not_consumed": PAIR_SOURCE_WAS_NOT_CONSUMED,
        "settings_consumption_sweep": SETTINGS_CONSUMPTION_SWEEP,
        "cohort_arms_observed": COHORT_ARMS_OBSERVED,
        "family_evaluable_subset": FAMILY_EVALUABLE_SUBSET,
        "postrun_reconstruction_audit": POSTRUN_RECONSTRUCTION_AUDIT,
        "run_path_check_feasibility": RUN_PATH_CHECK_FEASIBILITY,
        "shrinkage_observed": SHRINKAGE_OBSERVED_AND_PREDICTION_REFUTED,
        "five_arms_do_not_shrink": FIVE_ARMS_DO_NOT_SHRINK,
        "diagnostic_observed": DIAGNOSTIC_OBSERVED,
        "contrasts_observed": CONTRASTS_OBSERVED,
        "contrast_verdicts_observed": CONTRAST_VERDICTS_OBSERVED,
        "condition_phenomenon_count": CONDITION_PHENOMENON_COUNT_RECONCILED,
        "closing": PHASE_22_CLOSING,
    }
