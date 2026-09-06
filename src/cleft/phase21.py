"""Phase 21: the ensemble probe and error-consistency diagnosis.

Opened 2026-08-31. **This module is the REGISTRATION and nothing else**
-- no task, no schema kind, no config, no run. Two things go to the maintainer
before anything is built: the **arm-set rule** for Arm A (the ruling
was requested in the registration message and the request came back
without it -- see ``ARM_SET_RULE_PROPOSED``) and **Arm B's consistency
threshold** (``CONSISTENCY_THRESHOLD_PROPOSED``).

Both arms run entirely on **cached out-of-fold predictions**: no GPU, no
new artifact, no new hash. The inputs are the 68 locked arms'
``seed_<n>__predictions.csv`` files (``phase18.ARM_LIST_LOCKED``) and the
manifest.
"""

from __future__ import annotations


#: **[RECKONED 2026-08-31, EVERY FIGURE VERIFIED AT SOURCE] The opening
#: reckoning.**
#:
#: **What the record already holds**, checked rather than recalled:
#:
#: * ``ladder.SMALLEST_RESOLVABLE_DIFFERENCE`` -- this cohort **cannot
#:   resolve PCC differences of 0.04 to 0.10** between arms; the smallest
#:   it has ever resolved is **0.1386**.
#: * The probe: **0.2520, sd 0.0148**
#:   (``ladder.TRADE_OFF_PAIR["result"]``).
#: * ``phase20.SEED_STABILITY_EVIDENCE`` -- seed sd falls monotonically
#:   as real structure is added: **0.1109 / 0.0802 / 0.0222 / 0.0148**.
#: * ``phase18.ARM_LIST_LOCKED`` -- **68 arms**, counted at source across
#:   ten groups, every one of them holding per-seed OOF CSVs (the
#:   ``no_per_seed_oof_csvs`` exclusion exists precisely to guarantee
#:   this).
#:
#: **(A) Does combining existing arms' predictions exceed any of them
#: individually? NOT IN THE RECORD.**
#:
#: The near-miss is ``phase7b.ENSEMBLE_COMBINES`` --
#: ``concat``/``concat_standardised``/``concat_l2norm``. That is
#: **FEATURE concatenation with a learned head**, and Phase 7B's own note
#: says why it is not averaging: *"Averaging is unavailable and that is a
#: measured fact, not an omission: models.factory.BACKBONES gives
#: ViT-B/16 768 dims and Swin-B 1024, so there is nothing to average
#: without a projection, and a projection is a learned layer rather than
#: an ensemble."*
#:
#: **That reason cannot apply to Arm A.** Predictions are SCALARS. The
#: dimensional obstruction that ruled averaging out at the feature layer
#: does not exist at the prediction layer, and no record considers it.
#: Phase 7B ruled out a different operation for a reason specific to that
#: operation.
#:
#: **(B) Do architecturally diverse arms mispredict the SAME patients?
#: NOT IN THE RECORD, AND NOTHING IS CLOSE.**
#:
#: A mechanism search finds no error-consistency work of any kind: no
#: ``geirhos``, no ``error_consistency``, no per-patient residual
#: agreement. The nearest thing is
#: ``run._band_error_by_grade``, which groups **reconstruction** band
#: error **by grade** -- not model residuals, and not by patient identity
#: across arms.
#:
#: **Why the gap exists and is structural, not accidental**: every
#: analysis to date compares arms by **aggregate PCC**. PLAN 4.3's
#: criterion, the ladder, the ledger, the paired BCa -- all of them
#: reduce an arm to one number per seed. **A per-patient error structure
#: is invisible to every instrument this project has built.**
PHASE_21_RECKONING = {
    "reckoned": "2026-08-31, every figure verified at source",

    "resolution_floor": (
        "ladder.SMALLEST_RESOLVABLE_DIFFERENCE: this cohort CANNOT "
        "RESOLVE PCC differences of 0.04 to 0.10 between arms; the "
        "smallest ever resolved is 0.1386"
    ),
    "the_probe": (
        "0.2520, sd 0.0148 (ladder.TRADE_OFF_PAIR['result']: "
        "vit_paired_mean 0.2520, vit_sd 0.0148)"
    ),
    "seed_stability": (
        "phase20.SEED_STABILITY_EVIDENCE: seed sd falls monotonically as "
        "real structure is added -- P 0.1109, S 0.0802, C 0.0222, probe "
        "0.0148"
    ),
    "the_arm_list": (
        "phase18.ARM_LIST_LOCKED: 68 arms, COUNTED AT SOURCE across ten "
        "groups (12+18+1+4+3+2+3+10+12+3), every one holding per-seed "
        "OOF CSVs -- the 'no_per_seed_oof_csvs' exclusion exists to "
        "guarantee exactly that"
    ),

    "a_is_not_covered": (
        "**combining arms' PREDICTIONS is not in the record.** The "
        "near-miss is phase7b.ENSEMBLE_COMBINES (concat / "
        "concat_standardised / concat_l2norm), which is FEATURE "
        "concatenation with a learned head. Phase 7B's own note says why "
        "it is not averaging: 'Averaging is unavailable ... ViT-B/16 768 "
        "dims and Swin-B 1024, so there is nothing to average without a "
        "projection, and a projection is a learned layer rather than an "
        "ensemble'"
    ),
    "why_that_reason_cannot_apply_to_arm_a": (
        "**PREDICTIONS ARE SCALARS.** The dimensional obstruction that "
        "ruled averaging out at the FEATURE layer does not exist at the "
        "PREDICTION layer. Phase 7B ruled out a different operation for "
        "a reason specific to that operation, and no record considers "
        "this one"
    ),

    "b_is_not_covered": (
        "**no error-consistency work of any kind exists.** A mechanism "
        "search finds no 'geirhos', no 'error_consistency', no "
        "per-patient residual agreement. The nearest is "
        "run._band_error_by_grade, which groups RECONSTRUCTION band "
        "error BY GRADE -- not model residuals, and not by patient "
        "identity across arms"
    ),
    "why_the_gap_is_structural": (
        "**every analysis to date compares arms by AGGREGATE PCC.** PLAN "
        "4.3's criterion, the ladder, the ledger, the paired BCa -- all "
        "reduce an arm to one number per seed. A per-patient error "
        "structure is INVISIBLE TO EVERY INSTRUMENT THIS PROJECT HAS "
        "BUILT, which is why the gap is structural rather than an "
        "oversight"
    ),
    "coverage_conceded_where_it_exists": (
        "phase7b's concat arms DO answer 'does combining REPRESENTATIONS "
        "help' and answered it: arm 4's concat was withdrawn at +0.0073 "
        "(ladder's 7D concat precedent). Arm A must be reported against "
        "that, not as though no combination had ever been tried"
    ),
}


#: **[MEASURED 2026-08-31, AT SOURCE -- A CONSTRAINT THE REGISTRATION
#: MESSAGE DID NOT CARRY] THE 68 ARMS ARE NOT HOMOGENEOUS.**
#:
#: Two facts fall out of counting ``ARM_LIST_LOCKED`` rather than quoting
#: its total, and both bear on Arm A and Arm B directly:
#:
#: **1. FOUR ARMS ARE ON A DIFFERENT COHORT.** ``p12_view_ablation``
#: carries ``n_patients: 236``; the other nine groups (64 arms) carry
#: 237. Averaging predictions across them would average over
#: **misaligned patient sets**, and Phase 12's own record already refuses
#: the analogous move: ``PAIRED_CLAIM_COVERAGE["excluded"]
#: ["any_pair_against_the_237_ladder"]`` -- *"different cohort (236 vs
#: 237) and different fold populations"*.
#:
#: **2. THIRTY ARMS HAVE TEN SEEDS, THIRTY-EIGHT HAVE FIVE.**
#: ``p7_graph`` (18) and ``roadb_resolution_graph`` (12) run the graph
#: regime's ten seeds; the rest run five. The ten-seed set is a strict
#: SUPERSET beginning with the same five (1337, 2024, 7, 99, 12345), so a
#: shared five-seed basis exists -- but it has to be **declared**, not
#: assumed, or "five seeds" would mean two different things.
#:
#: Neither is a blocker. Both are **decisions that must be made before
#: any averaging**, and both are the R2 shape waiting to happen:
#: different quantities under one name.
COHORT_AND_SEED_HETEROGENEITY = {
    "measured": "2026-08-31, by counting ARM_LIST_LOCKED at source",
    "four_arms_are_on_236_not_237": (
        "p12_view_ablation carries n_patients: 236; the other nine "
        "groups (64 arms) carry 237. Averaging across them would average "
        "over MISALIGNED PATIENT SETS. Phase 12 already refuses the "
        "analogous move: PAIRED_CLAIM_COVERAGE['excluded']"
        "['any_pair_against_the_237_ladder'] -- 'different cohort (236 "
        "vs 237) and different fold populations'"
    ),
    "thirty_arms_have_ten_seeds": (
        "p7_graph (18) and roadb_resolution_graph (12) run the graph "
        "regime's ten seeds (ladder.SEEDS_BY_KIND); the other 38 run "
        "five. The ten-seed set is a strict SUPERSET beginning with the "
        "same five (1337, 2024, 7, 99, 12345), so a shared five-seed "
        "basis EXISTS -- but it must be DECLARED, not assumed, or 'five "
        "seeds' would mean two different things"
    ),
    "neither_is_a_blocker": (
        "both are DECISIONS TO BE MADE BEFORE ANY AVERAGING, and both "
        "are the R2 shape waiting to happen: different quantities under "
        "one name"
    ),
    "what_the_registration_proposes": (
        "the shared five seeds for every arm, and the 236/237 question "
        "settled by the arm-set rule rather than silently -- see "
        "ARM_SET_RULE_PROPOSED, which is to be ruled"
    ),
}


#: **[AWAITING A RULING -- REQUESTED AND NOT SUPPLIED] THE ARM-SET
#: RULE.**
#:
#: **The registration message asked for a ruling on the selection
#: problem and stated the constraint it had to satisfy**, that the arm
#: set is declared by a rule which never consults OOF PCC and that the
#: rule is recorded before any averaging. **The ruling itself was left
#: blank in the message and has not been supplied since.** The
#: constraint is recorded. The rule is not, because inventing one and
#: attributing it to the maintainer would be worse than leaving it open.
#:
#: **The constraint, which IS ruled and is recorded as binding**: the arm
#: set is declared by a rule that **never consults OOF PCC**, and the
#: rule is recorded **before any averaging**.
#:
#: **Proposed for ruling, three candidates**, each satisfying the
#: constraint and none consulting a result:
#:
#: 1. **ALL 68**, minus nothing. Maximally rule-free. Includes the four
#:    236-patient arms, so it forces the cohort question
#:    (``COHORT_AND_SEED_HETEROGENEITY``) to be answered by intersection
#:    -- ensemble on the 236 all arms share.
#: 2. **ALL ARMS ON THE 237 COHORT (64)**. One exclusion, on a DATA
#:    PROPERTY that is nothing to do with performance. This is the
#:    registration's preference: it removes the misalignment without
#:    consulting a single PCC.
#: 3. **ONE PER ARCHITECTURE FAMILY**, chosen by a declared
#:    non-performance key (e.g. the alphabetically first run stem in each
#:    of the ten groups). Maximises architectural diversity, which is
#:    what Arm B is about -- but "one per family" needs a family
#:    definition, and the group names are a scheduling artifact rather
#:    than an architecture taxonomy.
#:
#: **Whatever is ruled, the all-68 (or all-64) variant is reported
#: DESCRIPTIVELY beside it**, as the message requires.
ARM_SET_RULE_PROPOSED = {
    "status": (
        "**AWAITING A RULING.** It was marked '[insert the "
        "ruling]' in the registration message and arrived EMPTY. The "
        "CONSTRAINT is recorded as binding; the RULE is not, because "
        "inventing one and attributing it would be worse than "
        "leaving it open"
    ),
    "the_constraint_is_ruled_and_binds": (
        "the arm set is declared by a rule that NEVER CONSULTS OOF PCC, "
        "and the rule is recorded BEFORE ANY AVERAGING"
    ),
    "candidate_1_all_68": (
        "maximally rule-free; includes the four 236-patient arms, so it "
        "forces the cohort question to be answered by INTERSECTION -- "
        "ensemble on the 236 every arm shares"
    ),
    "candidate_2_all_237_cohort_arms_64": (
        "one exclusion, on a DATA PROPERTY with nothing to do with "
        "performance. **The registration's preference**: it removes the "
        "misalignment without consulting a single PCC"
    ),
    "candidate_3_one_per_architecture_family": (
        "chosen by a declared non-performance key (e.g. the "
        "alphabetically first run stem in each of the ten groups). "
        "Maximises architectural diversity, which is what Arm B is about "
        "-- but 'one per family' needs a family definition, and the "
        "group names are a SCHEDULING ARTIFACT rather than an "
        "architecture taxonomy"
    ),
    "the_all_variant_is_reported_either_way": (
        "whatever is ruled, the all-68 (or all-64) variant is reported "
        "DESCRIPTIVELY beside it, as the message requires"
    ),
    "the_seed_basis_proposed": (
        "the shared five seeds (1337, 2024, 7, 99, 12345) for EVERY arm, "
        "including the thirty that have ten. Declared, not assumed "
        "(COHORT_AND_SEED_HETEROGENEITY)"
    ),
    # [RULED 2026-09-01, the maintainer] Candidate 2. The proposal above is
    # preserved as written; ARM_SET_RULED is the locked rule.
    "ruled_2026_09_01": "ARM_SET_RULED",
}


#: **[RULED 2026-09-01] THE ARM SET, LOCKED: all 64 arms on the
#: 237 cohort.** Candidate 2, as proposed.
#:
#: **EXACTLY ONE EXCLUSION.** The four ``p12_view_ablation`` arms
#: (``p12_arm_a_frontal``, ``p12_arm_b_basal``, ``p12_arm_c_concat``,
#: ``p12_arm_d_capacity``) are excluded because they are on **236
#: patients, not 237**. Averaging predictions across them would average
#: over misaligned patient sets -- the move Phase 12's own record already
#: refuses for the analogous case (``PAIRED_CLAIM_COVERAGE["excluded"]
#: ["any_pair_against_the_237_ladder"]``: *"different cohort (236 vs 237)
#: and different fold populations"*).
#:
#: **The exclusion is on a DATA PROPERTY with no relation to
#: performance**, and the rule was **declared before any averaging**.
#: ``SELECTION_ON_EVALUATION_DATA_PROHIBITION`` forbids selecting by OOF
#: PCC; **this rule does not consult it.** Cohort size is knowable from
#: ``ARM_LIST_LOCKED`` without opening a single prediction file.
#:
#: **Counted at source, not asserted**: ``phase18.locked_arm_entries()``
#: yields 68 rows; filtering ``n_patients == 237`` yields **64**, and the
#: four dropped are exactly the p12 group. 63 distinct run directories
#: (the p16 run carries two prediction sets).
ARM_SET_RULED = {
    "ruled": "2026-09-01 -- candidate 2, as proposed",
    "the_set": "all 64 arms on the 237 cohort",
    "exactly_one_exclusion": (
        "the four p12_view_ablation arms (p12_arm_a_frontal, "
        "p12_arm_b_basal, p12_arm_c_concat, p12_arm_d_capacity), "
        "because they are on 236 PATIENTS, NOT 237. Averaging across "
        "them would average over MISALIGNED PATIENT SETS -- the move "
        "Phase 12's own record already refuses for the analogous case"
    ),
    "it_is_a_data_property_not_a_performance_one": (
        "**cohort size is knowable from ARM_LIST_LOCKED without opening "
        "a single prediction file.** SELECTION_ON_EVALUATION_DATA_"
        "PROHIBITION forbids selecting by OOF PCC; this rule does not "
        "consult it, and could not have"
    ),
    "declared_before_any_averaging": (
        "the rule is recorded here, in the registration, before the arms "
        "were built -- which is the constraint the maintainer bound at "
        "registration"
    ),
    "counted_at_source": (
        "phase18.locked_arm_entries() yields 68 rows; filtering "
        "n_patients == 237 yields 64, and the four dropped are exactly "
        "the p12 group. 63 distinct run directories -- the p16 run "
        "carries two prediction sets (loop and identity)"
    ),
}


#: **[REJECTED 2026-09-01 -- recorded per the
#: unregistered-variants pattern] "ARMS SHARING THE PROBE'S RECIPE".**
#:
#: **The reason it was rejected, and it binds on any future rule**: it is
#: **a rule chosen because its members were expected to perform well**,
#: which is **the selection problem in a different form**. It never names
#: a PCC, so it would pass a literal reading of
#: ``SELECTION_ON_EVALUATION_DATA_PROHIBITION`` -- and it selects on
#: expected performance all the same. **A proxy for the forbidden
#: quantity is the forbidden quantity.**
#:
#: **[PROVENANCE NOTE 2026-09-01, stated because it matters where a
#: rejected proposal came from.]** This rule is **not among the three
#: candidates written in ``ARM_SET_RULE_PROPOSED``**, which were: all 68;
#: all 237-cohort arms (64); and one per architecture family by a
#: declared non-performance key. So it was either raised in conversation
#: outside the registration, or is a paraphrase of candidate 3 -- and
#: candidate 3 as written selects by alphabetical stem, not by recipe or
#: expected performance, so the two are not the same rule.
#:
#: **The rejection is recorded regardless of where the candidate came
#: from, because the REASON is what has force**, and the reason applies
#: to any rule of that shape whoever proposes it.
ARM_SET_CANDIDATE_REJECTED = {
    "rejected": "2026-09-01",
    "the_rule": "restricting to 'arms sharing the probe's recipe'",
    "the_reason": (
        "**a rule chosen because its members were expected to perform "
        "well** -- the selection problem in a different form. It never "
        "names a PCC, so it would pass a LITERAL reading of "
        "SELECTION_ON_EVALUATION_DATA_PROHIBITION, and it selects on "
        "expected performance all the same. **A PROXY FOR THE FORBIDDEN "
        "QUANTITY IS THE FORBIDDEN QUANTITY**"
    ),
    "provenance_note_2026_09_01": (
        "**this rule is NOT among the three candidates written in "
        "ARM_SET_RULE_PROPOSED** (all 68; all 237-cohort arms; one per "
        "architecture family by a declared non-performance key). It was "
        "either raised in conversation outside the registration or is a "
        "paraphrase of candidate 3 -- and candidate 3 as written selects "
        "by ALPHABETICAL STEM, not by recipe or expected performance, so "
        "the two are not the same rule. Stated because where a rejected "
        "proposal came from is part of the record"
    ),
    "why_it_is_recorded_anyway": (
        "**the REASON is what has force**, and it applies to any rule of "
        "that shape whoever proposes it. Recorded per the "
        "unregistered-variants pattern: a considered-and-rejected option "
        "is part of the design record, not a loose end"
    ),
}


#: **[RULED 2026-09-01] THE SEED BASIS: the shared five.**
#:
#: Every arm contributes seeds **1337, 2024, 7, 99, 12345**, including
#: the thirty that have ten. The ten-seed set is a strict superset
#: beginning with the same five, so **"five seeds" means one thing
#: throughout** the phase.
#:
#: **The alternative not taken**: use each arm's own full seed set, so
#: the thirty graph arms contribute ten predictions each and the
#: thirty-four transformer arms contribute five. That would weight the
#: graph arms **twice as heavily in every ensemble average**, purely
#: because of a scheduling decision made in another phase
#: (``ladder.SEEDS_BY_KIND``) -- an unequal weighting nobody chose,
#: arriving through the back door. Rejected for that reason, and recorded
#: so the choice is visible rather than implicit.
SEED_BASIS_RULED = {
    "ruled": "2026-09-01",
    "the_basis": (
        "the shared five seeds (1337, 2024, 7, 99, 12345) for EVERY arm, "
        "including the thirty that have ten"
    ),
    "why_it_works": (
        "the ten-seed set is a STRICT SUPERSET beginning with the same "
        "five, so 'five seeds' means ONE THING throughout the phase"
    ),
    "the_alternative_not_taken": (
        "use each arm's own full seed set -- thirty graph arms "
        "contributing ten predictions each against thirty-four "
        "contributing five. That would weight the graph arms TWICE AS "
        "HEAVILY in every ensemble average, purely because of a "
        "scheduling decision made in another phase (ladder.SEEDS_BY_KIND) "
        "-- an unequal weighting NOBODY CHOSE, arriving through the back "
        "door. Rejected, and recorded so the choice is visible rather "
        "than implicit"
    ),
}


#: **[REGISTERED 2026-08-31] THE PROHIBITION -- selecting arms by their
#: OOF PCC is selection on the evaluation data. A tested literal**,
#: sibling to ``ladder.DETECTION_FLOOR_PROHIBITION``,
#: ``phase10_annex.ANNEX_PROHIBITION`` and
#: ``phase20.RESIDUAL_PROHIBITION``.
SELECTION_ON_EVALUATION_DATA_PROHIBITION = (
    "ARMS ARE NEVER SELECTED FOR THE ENSEMBLE BY THEIR OOF PCC. The "
    "out-of-fold predictions are the EVALUATION DATA: choosing which "
    "arms to combine by how well they scored on it, and then scoring the "
    "combination on the same data, is selection on the evaluation set. "
    "The ensemble's number would be biased upward by an amount nobody "
    "can estimate, and it would beat the probe for that reason alone. "
    "THIS IS THE SWEEP-REPORTING-ITS-BEST FAILURE IN A NEW COSTUME -- "
    "the same shape as phase18's 'it ranks a near-constant predictor "
    "first in a 68-arm ladder', arriving through a different door. The "
    "arm set is declared by a rule that never consults a result, and the "
    "rule is recorded before any averaging (ARM_SET_RULE_PROPOSED). A "
    "top-k-by-PCC ensemble is not a weaker version of this arm; it is a "
    "different and invalid measurement, and it is not run, not reported, "
    "and not quoted as a sensitivity check."
)


#: **[REGISTERED 2026-08-31] ARM A -- THE ENSEMBLE PROBE.**
#:
#: **The question**: does combining existing arms' predictions exceed any
#: of them individually?
#:
#: **The design**: **simple mean of per-patient predictions within each
#: seed**, evaluated **out-of-fold**, **five seeds**, contrasted against
#: the probe under **the full criterion** -- paired BCa over patients,
#: BOTH of PLAN 4.3's conditions.
#:
#: **Simple mean, not a learned combination.** A learned weighting fitted
#: on the OOF predictions is the prohibition again with extra steps: the
#: weights would be fitted on the evaluation data. The unweighted mean
#: has no free parameters and therefore nothing to fit.
#:
#: **Runs on cached CSVs.** ``seed_<n>__predictions.csv`` per arm, column
#: ``prediction``, keyed by ``patient_id``
#: (``phase18.ARM_LIST_LOCKED``). No GPU, no new artifact, no new hash.
ARM_A_REGISTERED = {
    "registered": "2026-08-31",
    "the_question": (
        "does combining existing arms' PREDICTIONS exceed any of them "
        "individually?"
    ),
    "design": (
        "SIMPLE MEAN of per-patient predictions WITHIN EACH SEED, "
        "evaluated OUT-OF-FOLD, five seeds, contrasted against the probe "
        "under the FULL criterion -- paired BCa over patients, BOTH of "
        "PLAN 4.3's conditions"
    ),
    "why_a_simple_mean_and_not_a_learned_one": (
        "**a learned weighting fitted on the OOF predictions is the "
        "prohibition again with extra steps** -- the weights would be "
        "fitted on the evaluation data. The unweighted mean has NO FREE "
        "PARAMETERS and therefore nothing to fit"
    ),
    "the_arm_set": "ARM_SET_RULE_PROPOSED -- to be ruled",
    "the_prohibition": "SELECTION_ON_EVALUATION_DATA_PROHIBITION",
    "inputs": (
        "cached seed_<n>__predictions.csv per arm, column 'prediction', "
        "keyed by patient_id (phase18.ARM_LIST_LOCKED). No GPU, no new "
        "artifact, no new hash"
    ),
    "also_reported_descriptively": (
        "the all-68 (or all-64) variant, beside whichever set the ruling "
        "declares"
    ),
    "the_prior": (
        "**registered as MODEST, before the number.** The 7D concat "
        "precedent combined two representations and was WITHDRAWN at "
        "+0.0073; phase20.SEED_STABILITY_EVIDENCE shows the probe is "
        "already the stable end of the range. An ensemble gain large "
        "enough to clear 0.1386 would be a surprise, and the registered "
        "expectation is reading 2"
    ),
}


#: **[REGISTERED 2026-08-31] ARM B -- ERROR CONSISTENCY.**
#:
#: **The question**: do architecturally diverse arms mispredict the SAME
#: patients?
#:
#: **The method**: Geirhos, Meding & Wichmann (NeurIPS 2020) error
#: consistency, **adapted to regression**. Their kappa is
#: ``(c_obs - c_exp) / (1 - c_exp)`` over a binary error indicator, where
#: ``c_exp = p1*p2 + (1-p1)*(1-p2)`` from the two models' own error
#: rates. Regression has no "error" without a rule, so **two
#: binarisations are registered, both reported**:
#:
#: * **residual SIGN** -- over- vs under-prediction. Error rate ~0.5 by
#:   construction for a centred model, so this measures whether arms err
#:   in the same DIRECTION on the same patients.
#: * **absolute-residual RANK**, binarised at the **worst quartile**.
#:   Error rate 0.25 by construction. This measures whether the same
#:   patients are the hard ones.
#:
#: The two answer different questions and are **not averaged together**.
#:
#: **Deliverables**: the pairwise consistency matrix; the per-patient
#: **hardness vector** (how many arms mispredict each patient); and the
#: cross-reference -- **hardness against per-patient inter-rater
#: disagreement**.
#:
#: **DESCRIPTIVE. No ledger row.**
ARM_B_REGISTERED = {
    "registered": "2026-08-31",
    "the_question": (
        "do architecturally diverse arms mispredict the SAME patients?"
    ),
    "the_method": (
        "Geirhos, Meding & Wichmann (NeurIPS 2020) error consistency, "
        "ADAPTED TO REGRESSION. kappa = (c_obs - c_exp) / (1 - c_exp) "
        "over a binary error indicator, with c_exp = p1*p2 + "
        "(1-p1)*(1-p2) from the two arms' own error rates"
    ),
    "two_binarisations_both_reported": {
        "residual_sign": (
            "over- vs under-prediction; error rate ~0.5 by construction "
            "for a centred model. Measures whether arms err in the same "
            "DIRECTION on the same patients"
        ),
        "worst_quartile_absolute_residual": (
            "|residual| ranked, binarised at the worst quartile; error "
            "rate 0.25 BY CONSTRUCTION. Measures whether the same "
            "patients are the HARD ones"
        ),
        "not_averaged_together": (
            "the two answer DIFFERENT QUESTIONS and are reported "
            "separately. Collapsing them to one number would be the R2 "
            "shape"
        ),
    },
    "deliverables": (
        "1. the pairwise consistency matrix over the declared arm set; "
        "2. the per-patient HARDNESS VECTOR -- how many arms mispredict "
        "each patient; "
        "3. the cross-reference: per-patient hardness against "
        "per-patient INTER-RATER DISAGREEMENT"
    ),
    "status": "DESCRIPTIVE, no ledger row",
    "threshold": "CONSISTENCY_THRESHOLD_PROPOSED -- to be ruled",
    # [2026-09-01] The source is now banked, and it SANCTIONS the two
    # binarisations rather than merely permitting them: the metric is
    # defined for binary correct/incorrect, and continuous outputs "must
    # be thresholded and binarized ... before computing this metric".
    # The registration reached this design independently.
    "binarisation_is_sanctioned_2026_09_01": (
        "literature.GEIRHOS_ERROR_CONSISTENCY -- the paper REQUIRES "
        "binarisation of a continuous output; residual sign and "
        "worst-quartile membership are the sanctioned adaptation, not an "
        "invented one"
    ),
}


#: **[MEASURED 2026-08-31] THE INTER-RATER DISAGREEMENT IS EXACTLY
#: RECOVERABLE -- and it takes at most 26 distinct values.**
#:
#: **It is NOT a manifest column.** The registration message said the
#: five raters' sd is "available in the manifest"; that is true in the
#: **derivable** sense and false as a column. ``MANIFEST_COLUMNS`` has no
#: sd. What it has is ``soft_1..soft_5`` -- the fraction of five raters
#: awarding each grade -- and ``data.softlabels`` records that *"soft
#: labels are fractions of five raters, so every value is a multiple of
#: 0.2"*. So ``soft_k * 5`` is an exact integer count, the full rater
#: multiset is recoverable, and the per-patient sd follows exactly.
#:
#: **Verified, not argued**: over 20,000 random five-rater panels, the
#: maximum discrepancy between the sd of the original grades and the sd
#: of the multiset rebuilt from soft labels is **4.44e-16** -- float
#: noise.
#:
#: **The constraint that matters for the cross-reference**: a five-rater
#: panel on a 1-5 scale admits **only 26 distinct sd values**, from 0.0
#: to 2.1909. Across 237 patients the disagreement variable is
#: **heavily tied**, which rules out a naive Pearson correlation and is
#: registered here **before the run** rather than discovered in it.
#:
#: **A second, independent route exists** and should be used as a
#: cross-check, not a substitute: ``run.phase10_rater_grades`` reads the
#: five rater columns from the score sheet directly. It needs the
#: scoresheet artifact declared; the soft-label route needs only the
#: manifest. **The label join in this project has already produced two
#: defects**, so computing it both ways and asserting equality is worth
#: the few lines.
INTER_RATER_DISAGREEMENT_MEASURED = {
    "measured": "2026-08-31",
    "it_is_not_a_manifest_column": (
        "the registration message said the five raters' sd is 'available "
        "in the manifest'; that is true in the DERIVABLE sense and false "
        "as a column. MANIFEST_COLUMNS has no sd"
    ),
    "how_it_is_recovered": (
        "soft_1..soft_5 are the fraction of five raters awarding each "
        "grade, and data.softlabels records that 'soft labels are "
        "fractions of five raters, so every value is a multiple of 0.2'. "
        "soft_k * 5 is an exact integer count, the full rater multiset "
        "is recoverable, and the per-patient sd follows EXACTLY"
    ),
    "verified_not_argued": (
        "over 20,000 random five-rater panels the maximum discrepancy "
        "between the sd of the original grades and the sd of the "
        "multiset rebuilt from soft labels is 4.44e-16 -- float noise"
    ),
    "only_26_distinct_values": (
        "**the constraint that matters for the cross-reference.** A "
        "five-rater panel on a 1-5 scale admits only 26 DISTINCT SD "
        "VALUES, from 0.0 to 2.1909. Across 237 patients the "
        "disagreement variable is HEAVILY TIED, which rules out a naive "
        "Pearson correlation -- registered BEFORE the run rather than "
        "discovered in it"
    ),
    "the_cross_reference_must_handle_ties": (
        "Spearman with a tie correction, or a grouped comparison of "
        "mean hardness across disagreement levels. Declared before the "
        "run; whichever is used is reported with the tie structure "
        "beside it"
    ),
    "a_second_route_for_cross_check": (
        "run.phase10_rater_grades reads the five rater columns from the "
        "SCORE SHEET directly. It needs the scoresheet artifact "
        "declared; the soft-label route needs only the manifest. **The "
        "label join in this project has already produced two defects**, "
        "so computing it both ways and asserting equality is worth the "
        "few lines"
    ),
}


#: **[FOUND AND CORRECTED 2026-09-01, WHILE BUILDING THIS PHASE] A
#: DEFECT IN ``phase11.kendall_tau_b``.**
#:
#: **The defect.** tau-b's denominator is
#: ``sqrt((n0 - n1) * (n0 - n2))`` -- all pairs, minus those tied in
#: ``a``, times all pairs minus those tied in ``b``. A pair tied in
#: **both** appears in ``n1`` and in ``n2``, so it is removed from both
#: factors and contributes to neither. The shipped form **added such
#: pairs into both factors instead**, inflating the denominator.
#:
#: **The decisive symptom**: a vector compared against ITSELF is perfect
#: agreement and must give exactly 1.0. It gave **0.6667** for
#: ``[1, 1, 2]``, **0.8000** for ``[1, 1, 2, 2, 3]``, **0.8214** for
#: ``[0, 0, 0, 1, 1, 2, 3, 3]``.
#:
#: **Verified against ``scipy.stats.kendalltau``**, which matches the
#: corrected form exactly on every case tried and matches the previous
#: form only when no pair is tied in both. Over 300 untied random pairs
#: the two forms agreed to **0.00e+00** -- which is why it survived.
#:
#: **Why it survived**: ``tests/test_phase11.py`` exercised only UNTIED
#: vectors, where the previous form was exact. The missing test -- a
#: vector against itself, WITH ties -- is now there.
#:
#: **THE ERROR IS ONE-DIRECTIONAL.** It could only shrink ``|tau|``,
#: never grow it. **Every tau this project has banked is a LOWER BOUND
#: on its true value.**
#:
#: **DOWNSTREAM, AND THIS IS TO BE RULED.** Phase 18 banked six
#: tau figures through the defective form: tau(PCC, macro F1) 0.5180,
#: tau(PCC, IEM) 0.3784, tau(macro F1, QWK) 0.7024, tau(PCC, QWK) 0.6295,
#: tau(accuracy, QWK) 0.3392663. **One of them carries a pre-registered
#: verdict** -- D3 HELD iff tau(accuracy, QWK) > 0.7024, and it was ruled
#: WRONG at 0.3393.
#:
#: **What can be said without the artifacts**: the affected quantity is
#: the count of pairs tied in BOTH metrics over the 68 arms. Accuracy is
#: ``k/237`` and ties readily; QWK, PCC, macro F1 and IEM are continuous
#: and tie rarely, so doubly-tied pairs are probably few and the
#: attenuation small. **Probably is not measured.** Every figure above
#: should be re-derived from its run artifact before being quoted again,
#: and the direction of any change is known: **upward, or not at all.**
#: Phase 18 is closed; nothing there is rewritten by this record.
TAU_B_DEFECT_CORRECTED = {
    "found": "2026-09-01, while building this phase",
    "the_defect": (
        "tau-b's denominator is sqrt((n0 - n1) * (n0 - n2)). A pair tied "
        "in BOTH variables appears in n1 AND n2, so it is removed from "
        "both factors and contributes to neither. The shipped form ADDED "
        "such pairs INTO both factors instead, inflating the denominator"
    ),
    "the_decisive_symptom": (
        "a vector compared against ITSELF is perfect agreement and must "
        "give exactly 1.0. It gave 0.6667 for [1, 1, 2], 0.8000 for "
        "[1, 1, 2, 2, 3], 0.8214 for [0, 0, 0, 1, 1, 2, 3, 3]"
    ),
    "verified_against_scipy": (
        "scipy.stats.kendalltau matches the CORRECTED form exactly on "
        "every case tried, and matches the previous form only when no "
        "pair is tied in both. Over 300 untied random pairs the two "
        "forms agreed to 0.00e+00 -- which is why it survived"
    ),
    "why_it_survived": (
        "tests/test_phase11.py exercised only UNTIED vectors, where the "
        "previous form was exact. The missing test -- a vector against "
        "itself, WITH ties -- is now there"
    ),
    "the_error_is_one_directional": (
        "**it could only shrink |tau|, never grow it. Every tau this "
        "project has banked is a LOWER BOUND on its true value**"
    ),
    # [SUPERSEDED 2026-09-01 -- the original is preserved below. "Six
    # banked figures came through it" was right about which function
    # they called and WRONG about which the defect could reach: five of
    # the six go through run.agreement, which passes TIE-FREE RANK
    # POSITIONS (permutations of 0..67), so the doubly-tied term is zero
    # and the defect is INERT for them. Exactly ONE figure was affected.
    # See phase18.TAU_B_CORRECTION_IMPACT and phase18.IEM_TAU_ORIENTATION.
    "downstream_and_to_be_ruled": (
        "[SUPERSEDED 2026-09-01 -- see corrected_scope_2026_09_01] "
        "Phase 18 banked six tau figures through the defective form: "
        "tau(PCC, macro F1) 0.5180, tau(PCC, IEM) 0.3784, tau(macro F1, "
        "QWK) 0.7024, tau(PCC, QWK) 0.6295, tau(accuracy, QWK) "
        "0.3392663. ONE CARRIES A PRE-REGISTERED VERDICT -- D3 HELD iff "
        "tau(accuracy, QWK) > 0.7024, ruled WRONG at 0.3393"
    ),
    "corrected_scope_2026_09_01": (
        "**EXACTLY ONE banked figure was affected: tau(accuracy, QWK), "
        "0.3392663 -> 0.3400134 (+0.0007471), on 5 pairs tied in both "
        "columns by duplicate runs.** The other five could not have "
        "moved: they are computed by run.agreement, which passes RANK "
        "POSITIONS -- permutations of 0..67, TIE-FREE BY CONSTRUCTION -- "
        "so the doubly-tied term is zero and the two denominators are "
        "identical (measured: max |old - corrected| = 0.000e+00 over 300 "
        "trials at that shape). Only tau_accuracy_vs_qwk computes on RAW "
        "POOLED VALUES, which is why it alone could tie and alone "
        "moved. D3's WRONG stands by 0.3624"
    ),
    "what_can_be_said_without_the_artifacts": (
        "the affected quantity is the count of pairs tied in BOTH "
        "metrics over the 68 arms. Accuracy is k/237 and ties readily; "
        "QWK, PCC, macro F1 and IEM are continuous and tie rarely, so "
        "doubly-tied pairs are PROBABLY few and the attenuation small. "
        "**Probably is not measured.** Every figure should be re-derived "
        "from its run artifact before being quoted again, and the "
        "direction of any change is known: UPWARD, OR NOT AT ALL. Phase "
        "18 is closed; nothing there is rewritten by this record"
    ),
    # [2026-09-01] The downstream size is now MEASURED for one of the
    # six figures and the one-directional argument is now a PROOF -- see
    # phase18.TAU_B_CORRECTION_IMPACT. tau(accuracy, QWK) rose by
    # +0.0007471 to 0.3400134 and D3's WRONG stands by 0.3624. The two
    # IEM figures never moved: their apparent flips were an ORIENTATION
    # artifact in a conversational probe, not defect drift
    # (phase18.IEM_TAU_ORIENTATION, phase18.TAU_RECOMPUTATION_PROVENANCE).
    "downstream_measured_2026_09_01": "phase18.TAU_B_CORRECTION_IMPACT",
    "why_it_was_fixed_rather_than_worked_around": (
        "Arm B's cross-reference is EXACTLY this shape -- a 26-level "
        "disagreement vector against an integer hardness vector, which "
        "is dense in doubly-tied pairs. Building a new phase on a "
        "statistic just proven wrong for its own case was not an option, "
        "and a second corrected copy in phase21 would have been the "
        "two-implementations defect this project keeps finding"
    ),
}


#: **[PROPOSED 2026-08-31 -- NOT LOCKED] ARM B'S CONSISTENCY THRESHOLD,
#: WITH ITS DERIVATION.**
#:
#: **The LOW threshold is DERIVED from the chance correction.** Simulated
#: at n = 237 over 20,000 independent pairs, with the shipped kappa:
#:
#:     residual sign (p = 0.50)      mean -0.0003  sd 0.0651  p99 +0.149
#:     worst quartile (p = 0.25)     mean -0.0005  sd 0.0649  p99 +0.152
#:
#: The two binarisations give **the same null** to three decimals, which
#: is itself worth knowing: the threshold does not depend on which one is
#: used. A **single pair** at kappa = 0.10 is unremarkable; the **mean
#: over many pairs** is far tighter (50 independent pairs: sd 0.0091,
#: p99 +0.022).
#:
#: **Proposed LOW: mean pairwise kappa <= 0.05.** Above the mean-of-pairs
#: null p99 (~0.022) by more than a factor of two, so calling this
#: "indistinguishable from chance" is **conservative** in the direction
#: that matters -- it makes LOW harder to declare, not easier.
#:
#: **Proposed HIGH: mean pairwise kappa >= 0.50.** kappa is normalised so
#: that 1.0 is all achievable beyond-chance agreement and 0.0 is none, so
#: 0.50 is **the midpoint of its own scale**: half the achievable
#: agreement realised.
#:
#: **HONEST NOTE ON THE TWO THRESHOLDS, and they are not equally
#: grounded.** LOW is **derived** -- it comes from a measured null
#: distribution. HIGH is **principled but not measured**: it is the
#: midpoint of a normalised scale, not a quantity anyone computed. It is
#: NOT the Landis-Koch convention and does not borrow its authority. This
#: asymmetry is stated rather than smoothed over, and HIGH is the one
#: most in need of a ruling.
#:
#: **BETWEEN (0.05, 0.50) is deliberately wide** -- partial is the honest
#: cell, and narrowing it pushes borderline outcomes into a confident
#: reading. Same principle as ``phase20``'s diagnostic band.
CONSISTENCY_THRESHOLD_PROPOSED = {
    "status": "PROPOSED -- NOT LOCKED until it is ruled",
    "proposed": "2026-08-31",
    "the_measured_null": (
        "simulated at n = 237 over 20,000 independent pairs with the "
        "shipped kappa: residual sign (p=0.50) mean -0.0003, sd 0.0651, "
        "p99 +0.149; worst quartile (p=0.25) mean -0.0005, sd 0.0649, "
        "p99 +0.152. **The two binarisations give the same null to three "
        "decimals** -- the threshold does not depend on which is used"
    ),
    "the_mean_over_pairs_is_far_tighter": (
        "50 independent pairs: sd 0.0091, p99 +0.022. A SINGLE pair at "
        "kappa = 0.10 is unremarkable; the MEAN over many pairs at 0.10 "
        "would not be"
    ),
    "proposed_low": (
        "mean pairwise kappa <= 0.05 -- above the mean-of-pairs null p99 "
        "(~0.022) by more than a factor of two, so calling this "
        "'indistinguishable from chance' is CONSERVATIVE in the "
        "direction that matters: it makes LOW harder to declare, not "
        "easier"
    ),
    "proposed_high": (
        "mean pairwise kappa >= 0.50 -- kappa is normalised so 1.0 is "
        "all achievable beyond-chance agreement and 0.0 is none, so 0.50 "
        "is THE MIDPOINT OF ITS OWN SCALE: half the achievable agreement "
        "realised"
    ),
    "the_two_are_not_equally_grounded": (
        "**stated rather than smoothed over.** LOW is DERIVED -- from a "
        "measured null distribution. HIGH is PRINCIPLED BUT NOT MEASURED "
        "-- the midpoint of a normalised scale, not a quantity anyone "
        "computed. It is NOT the Landis-Koch convention and does not "
        "borrow its authority. HIGH is the one most in need of El "
        "the maintainer's ruling"
    ),
    "between_is_deliberately_wide": (
        "(0.05, 0.50): partial is the honest cell, and narrowing it "
        "pushes borderline outcomes into a confident reading -- the same "
        "principle as phase20's diagnostic band"
    ),
    # [RULED 2026-09-01, the maintainer] Both as proposed, asymmetry preserved.
    "ruled_2026_09_01": "CONSISTENCY_THRESHOLDS_RULED",
}


#: **[RULED 2026-09-01] ARM B'S THRESHOLDS: LOW <= 0.05,
#: HIGH >= 0.50, and the asymmetry between them is PART OF THE RULING.**
#:
#: **LOW <= 0.05 is DERIVED.** Simulated at n = 237 over 20,000
#: independent pairs; the mean-of-pairs null sits at p99 = **+0.022**, so
#: 0.05 is above it by more than a factor of two. That is conservative
#: **in the direction that matters**: it makes LOW *harder* to declare,
#: not easier, so "the arms err independently" is the claim that has to
#: work for its result.
#:
#: **HIGH >= 0.50 is CONVENTION, not measurement**, and is recorded as
#: the less-grounded half. kappa is normalised so 1.0 is all achievable
#: beyond-chance agreement and 0.0 is none; 0.50 is the midpoint of that
#: scale. **It does not borrow Landis-Koch's authority** and is not that
#: convention.
#:
#: **WHY NO MEASURED HIGH THRESHOLD EXISTS, stated plainly.** A derived
#: HIGH would need a null for *"genuinely shared hard cases"* -- a
#: simulation of what kappa looks like when the shared-ceiling account is
#: TRUE. **That cannot be built without assuming the answer**: the
#: simulation would need a shared-hardness parameter, and whatever value
#: it was given would determine the threshold, so the threshold would
#: encode the assumption rather than test it. The LOW threshold has a
#: null because **independence is a specific, simulable hypothesis**;
#: "shared" is a family of hypotheses indexed by how shared. This is not
#: a gap to be closed later -- it is a property of the question.
#:
#: **The band between is deliberately wide.**
CONSISTENCY_THRESHOLDS_RULED = {
    "ruled": "2026-09-01 -- both as proposed",
    "low": "mean pairwise kappa <= 0.05",
    "high": "mean pairwise kappa >= 0.50",
    # The numeric home. The loader compares against THESE, never against
    # the sentences above -- values are not parsed out of prose, and a
    # re-wording must not be able to move a threshold.
    "low_value": 0.05,
    "high_value": 0.50,
    "low_is_derived": (
        "simulated at n = 237 over 20,000 independent pairs; the "
        "mean-of-pairs null sits at p99 = +0.022, so 0.05 is above it by "
        "more than a factor of two. Conservative IN THE DIRECTION THAT "
        "MATTERS: it makes LOW harder to declare, so 'the arms err "
        "independently' is the claim that must work for its result"
    ),
    "high_is_convention_not_measurement": (
        "kappa is normalised so 1.0 is all achievable beyond-chance "
        "agreement and 0.0 is none; 0.50 is the MIDPOINT OF THAT SCALE. "
        "**It does not borrow Landis-Koch's authority and is not that "
        "convention.** Recorded as the LESS-GROUNDED HALF"
    ),
    "why_no_measured_high_threshold_exists": (
        "**a derived HIGH would need a null for 'genuinely shared hard "
        "cases' -- a simulation of what kappa looks like when the "
        "shared-ceiling account is TRUE. That cannot be built without "
        "assuming the answer**: the simulation needs a shared-hardness "
        "parameter, and whatever value it was given would determine the "
        "threshold, so the threshold would ENCODE the assumption rather "
        "than test it. LOW has a null because INDEPENDENCE IS A "
        "SPECIFIC, SIMULABLE HYPOTHESIS; 'shared' is a FAMILY of "
        "hypotheses indexed by how shared. **This is not a gap to be "
        "closed later -- it is a property of the question**"
    ),
    "the_asymmetry_is_part_of_the_ruling": (
        "the two thresholds are not equally grounded and the record says "
        "so wherever either is quoted. A reader who takes HIGH for a "
        "measured boundary has been misled by the record, not by the "
        "number"
    ),
    "between_is_deliberately_wide": (
        "(0.05, 0.50): partial is the honest cell, and narrowing it "
        "pushes borderline outcomes into a confident reading"
    ),
    # [2026-09-01] Geirhos' own measured bands are now banked
    # (literature.GEIRHOS_ERROR_CONSISTENCY): human-to-human 0.32-0.48,
    # CNN-to-CNN 0.62-0.79. The locked 0.50 sits 0.02 above the human
    # ceiling and 0.12 below the CNN floor. **CONTEXT TO REPORT BESIDE
    # THE THRESHOLD, NOT AN AMENDMENT TO IT** -- the band is a
    # different quantity on different models and tasks, and moving a
    # locked threshold to meet it would be choosing from outside the
    # registration. The lock is untouched.
    "published_band_for_context_2026_09_01": (
        "literature.GEIRHOS_ERROR_CONSISTENCY -- reported BESIDE the "
        "result, never used to move this threshold"
    ),
}


#: **[PROPOSED 2026-09-01 -- A RULING IS OWED] THE DISAGREEMENT
#: CROSS-REFERENCE STATISTIC, with its derivation and its null.**
#:
#: **The statistic: Kendall's tau-b**, ``phase11.kendall_tau_b`` -- the
#: shipped one, corrected today (``TAU_B_DEFECT_CORRECTED``). Chosen
#: because it is **defined for ties by construction** rather than
#: patched for them: its denominator excludes tied pairs from the count
#: of comparable pairs, so heavy tying reduces the statistic's precision
#: without biasing it.
#:
#: **Why not Pearson**: the disagreement variable takes at most **26
#: distinct values** and hardness at most 65, so both are dense in ties;
#: Pearson would treat the sd scale's spacing as meaningful when only its
#: ordering is.
#:
#: **Why not Spearman**: Spearman's tie correction exists, but its null
#: distribution under heavy tying is approximated rather than exact, and
#: this project has a shipped tau-b and no shipped Spearman. One
#: implementation, not two.
#:
#: **The null: a PERMUTATION test at the OBSERVED tie structure.**
#: Shuffle hardness against disagreement 20,000 times and take tau-b each
#: time. Permutation **preserves both marginal tie structures exactly**,
#: because it only reassigns the pairing -- so the null is computed at
#: the tie structure the data actually has, not at an assumed one. No
#: distributional approximation enters.
#:
#: **Validated before proposing, not after**: at a plausible tie
#: structure (13 distinct disagreement values over 237, largest tie group
#: 51 patients = 22%), the permutation null is centred at **-0.0001**
#: with sd **0.0461**, and the test's **type-I error is 0.040 against a
#: nominal 0.050** over 300 independent trials -- correctly calibrated,
#: slightly conservative.
#:
#: **Declared now rather than chosen at analysis time**, which is the
#: whole point: a statistic picked after seeing the scatter is a
#: statistic picked to make a story.
DISAGREEMENT_STATISTIC_PROPOSED = {
    "status": "PROPOSED 2026-09-01 -- to be ruled",
    "the_statistic": (
        "Kendall's tau-b via phase11.kendall_tau_b -- the shipped one, "
        "corrected today (TAU_B_DEFECT_CORRECTED)"
    ),
    "why_tau_b": (
        "**defined for ties by construction** rather than patched for "
        "them: its denominator excludes tied pairs from the count of "
        "comparable pairs, so heavy tying reduces PRECISION without "
        "introducing BIAS"
    ),
    "why_not_pearson": (
        "the disagreement variable takes at most 26 distinct values and "
        "hardness at most 65, so both are dense in ties; Pearson would "
        "treat the sd scale's SPACING as meaningful when only its "
        "ORDERING is"
    ),
    "why_not_spearman": (
        "Spearman's tie correction exists, but its null under heavy "
        "tying is APPROXIMATED rather than exact -- and this project has "
        "a shipped tau-b and no shipped Spearman. One implementation, "
        "not two"
    ),
    "the_null": (
        "**a PERMUTATION test at the OBSERVED tie structure**: shuffle "
        "hardness against disagreement 20,000 times, tau-b each time. "
        "Permutation PRESERVES BOTH MARGINAL TIE STRUCTURES EXACTLY -- it "
        "only reassigns the pairing -- so the null is computed at the tie "
        "structure the data actually has, not at an assumed one. No "
        "distributional approximation enters"
    ),
    "validated_before_proposing": (
        "at a plausible tie structure (13 distinct disagreement values "
        "over 237, largest tie group 51 patients = 22%), the permutation "
        "null is centred at -0.0001 with sd 0.0461, and the test's "
        "TYPE-I ERROR IS 0.040 against a nominal 0.050 over 300 "
        "independent trials -- correctly calibrated, slightly "
        "conservative"
    ),
    "declared_now_not_at_analysis_time": (
        "**a statistic picked after seeing the scatter is a statistic "
        "picked to make a story.** This is why the proposal lands in the "
        "same cycle as the build rather than with the results"
    ),
    "the_sd_is_cross_checked_by_two_routes": (
        "the derived sd (soft_k * 5 recovers the multiset) is asserted "
        "equal to run.phase10_rater_grades' direct read of the five "
        "rater columns. **The label join has produced two defects "
        "before**, and one line of assertion costs nothing"
    ),
}


#: **[COMMITTED 2026-08-31, BEFORE ANY NUMBER] THE READINGS.**
READINGS_COMMITTED = {
    "committed": "2026-08-31, before any number exists",

    # ---- Arm A ------------------------------------------------------
    "a_above_the_probe_and_claimable": (
        "**arms make partly independent errors, and combination recovers "
        "signal none of them holds alone.** Both of PLAN 4.3's "
        "conditions met against the probe. This would be the phase's "
        "only positive result and it is the LEAST expected of the three"
    ),
    "a_above_but_unresolved": (
        "**THE REGISTERED EXPECTATION**, per ladder.COHORT_CANNOT_RESOLVE "
        "and the 7D concat precedent (+0.0073, withdrawn). Reported as a "
        "POINT ESTIMATE WITH ITS INTERVAL AND NO CLAIM -- the gain is "
        "stated, its failure to clear the criterion is stated, and "
        "neither is dressed as the other"
    ),
    "a_at_or_below_the_probe": (
        "**the arms are not independent enough to combine** -- averaging "
        "cannot help when the things averaged agree. This is ITSELF "
        "EVIDENCE FOR THE SHARED-CEILING ACCOUNT and feeds directly into "
        "Arm B: it predicts HIGH consistency"
    ),

    # ---- Arm B ------------------------------------------------------
    "b_high_consistency": (
        "**the same patients are hard for every architecture.** The "
        "ceiling HAS A LOCUS and is not a capacity problem: adding "
        "parameters or changing architecture has been failing because "
        "the difficulty is in specific patients, not in the models"
    ),
    "b_low_consistency": (
        "**arms make different errors yet reach the same PCC.** The "
        "ceiling is NOT a shared-hard-cases story -- and Arm A SHOULD "
        "HAVE WORKED. If it did not, that pairing is a contradiction "
        "requiring explanation (see the combination readings)"
    ),
    "b_hardness_correlates_with_disagreement": (
        "**the residual is LABEL NOISE** -- the strongest available "
        "explanation for the ceiling. The patients the models cannot "
        "predict are the patients the RATERS cannot agree on, and no "
        "model can predict a quantity its target does not determine"
    ),
    "b_hardness_does_not_correlate_with_disagreement": (
        "**hard patients are hard for an IMAGE reason nobody has "
        "identified**, and that becomes a NAMED OPEN QUESTION carried "
        "forward -- not a loose end. The obvious candidates (staging "
        "geometry, the confound statistics, grade extremity) are "
        "checkable and would be the next phase's material"
    ),

    # ---- the combinations, because the arms are NOT independent -----
    "combination_a_fails_and_b_high": (
        "**THE COHERENT SHARED-CEILING ACCOUNT.** Averaging cannot help "
        "because the arms agree; they agree because the same patients "
        "are hard for all of them. Two arms, one story, and each "
        "predicts the other -- which is what makes it an account rather "
        "than two observations"
    ),
    "combination_a_fails_and_b_low": (
        "**A CONTRADICTION REQUIRING EXPLANATION, NOT A STORY.** "
        "Independent errors that do not combine is not a coherent "
        "position: if the arms err on different patients, averaging "
        "should reduce error. Registered in advance as a CONTRADICTION "
        "so it cannot be narrated away after the fact. What would have "
        "to be checked, named now: (i) whether the residuals are "
        "independent but BIASED in the same direction, which sign "
        "consistency would catch and quartile consistency would not; "
        "(ii) whether the averaging is dominated by a few extreme arms; "
        "(iii) whether PCC is insensitive to the error reduction that "
        "actually occurred -- a variance change with no correlation "
        "change"
    ),
    "combination_a_succeeds_and_b_high": (
        "also incoherent in the other direction, and named for the same "
        "reason: if the same patients are hard for every arm, there is "
        "little for a mean to recover. Would require the same kind of "
        "explanation"
    ),
    "no_reading_is_invented_after": (
        "whichever lands, no reading is added once numbers exist; a "
        "pattern outside these cells gets a dated OBSERVATION, on "
        "phase17.UNPREDICTED_PATTERN's precedent -- which Phase 20's Arm "
        "S required, so the precedent is live and not theoretical"
    ),
}


#: **[BOUND 2026-08-31, BEFORE EITHER ARM RUNS] WHAT NEITHER ARM
#: LICENSES.**
#:
#: **Neither arm licenses any claim about what the 0.2520 is made of.**
#: That is Phase 20's territory, it is BOUNDED, and it stays bounded.
#: ``phase20.RESIDUAL_PROHIBITION`` is unaffected by anything measured
#: here.
#:
#: The distinction is sharp and worth stating because Arm B will be
#: tempting: **"the residual is label noise" is a claim about what the
#: MODEL CANNOT REACH, not a claim about what the 0.2520 CONTAINS.**
#: Learning that hard patients are the high-disagreement patients says
#: something about the ceiling's locus; it says nothing about how much of
#: the achieved correlation is cleft-specific rather than confound
#: structure. Phase 20 measured that and bounded it at partial-and-
#: variable survival. **Arm B cannot unbound it, in either direction.**
PHASE_21_BINDING = {
    "bound": "2026-08-31, before either arm runs",
    "the_binding": (
        "**neither arm licenses any claim about what the 0.2520 is made "
        "of.** That is Phase 20's territory, it is BOUNDED, and it stays "
        "bounded. phase20.RESIDUAL_PROHIBITION is unaffected by anything "
        "measured here"
    ),
    "the_distinction_that_will_be_tempting": (
        "**'the residual is label noise' is a claim about what the MODEL "
        "CANNOT REACH, not a claim about what the 0.2520 CONTAINS.** "
        "Learning that hard patients are the high-disagreement patients "
        "says something about the CEILING'S LOCUS; it says nothing about "
        "how much of the achieved correlation is cleft-specific rather "
        "than confound structure"
    ),
    "arm_b_cannot_unbound_phase_20": (
        "Phase 20 measured the partition and bounded it at "
        "partial-and-variable confound survival (60.2% retained, 37.6 "
        "point spread). Arm B cannot unbound that IN EITHER DIRECTION"
    ),
    "and_neither_can_arm_a": (
        "an ensemble that beat the probe would raise the achieved "
        "correlation; it would not say what the correlation is made of. "
        "The residual prohibition rides on any figure this phase "
        "produces"
    ),
}


#: **[DECIDED 2026-08-31] THE SIXTH SEQUENCE AMENDMENT -- the
#: second that renumbers nothing.**
#:
#: Phase 21 is appended after 20. **19 remains the write-up**, and the
#: rule ``PHASE_SEQUENCE_EXTENDED_5`` established holds: **THE WRITE-UP
#: RUNS LAST REGARDLESS OF ITS NUMBER.** This is the first amendment
#: written under that rule rather than establishing it, and it is the
#: test of it: a second appended phase would have forced a second
#: renumber under the old assumption, and forces none under this one.
PHASE_SEQUENCE_EXTENDED_6 = {
    "decided": (
        "2026-08-31 -- the SIXTH sequence amendment, and the "
        "second that renumbers nothing"
    ),
    "first_amendment": "phase11.PHASE_SEQUENCE_RENUMBERED (view ablation to 12)",
    "second_amendment": "phase12.PHASE_SEQUENCE_RENUMBERED_2 (LDL to 14, "
                        "second beauty dataset to 15, TSTR to 16)",
    "third_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_3 (metric-space "
                       "ablation to 16, TSTR to 17, write-up to 18)",
    "fourth_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_4 (anchor loop "
                        "to 16, metric-space ablation to 18, write-up to 19)",
    "fifth_amendment": "phase20.PHASE_SEQUENCE_EXTENDED_5 (permutation "
                       "control appended as 20; renumbered nothing)",
    "was": {
        "20": "the permutation control (closed 2026-08-31)",
        "21": "**NOTHING -- the chain stopped at 20**",
    },
    "becomes": {
        "19": "write-up (UNCHANGED in place and content)",
        "20": "the permutation control (UNCHANGED, closed)",
        "21": "THE ENSEMBLE PROBE AND ERROR-CONSISTENCY DIAGNOSIS",
    },
    "status_changes": {
        "write_up": "Phase 19 -> Phase 19 (unchanged)",
        "ensemble_and_consistency": "unnumbered -> PHASE 21",
    },
    # [2026-09-01] A SEVENTH amendment schedules 22 (ranking and
    # pairwise losses), 23 (the statistical instruments), 24 (all five
    # raters) and 25 (foundation-model features) as
    # SCHEDULED-NOT-REGISTERED. It renumbers nothing, and leaves the
    # write-up's number OPEN rather than resolving it
    # (phase21.WRITE_UP_NUMBER_OPEN).
    "seventh_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_7",
    "it_is_the_test_of_the_fifth_amendments_rule": (
        "**the first amendment written UNDER the write-up-runs-last rule "
        "rather than establishing it.** A second appended phase would "
        "have forced a second renumber of the write-up under the old "
        "assumption, and forces none under this one. The rule is paying "
        "for itself already"
    ),
    "module_names_never_change": (
        "unchanged: a module file is named for the number its phase "
        "HOLDS, and a record KEY keeps the number it was WRITTEN with. "
        "Nothing is renamed because nothing moved"
    ),
    "eighth_amendment": (
        "phase25.PHASE_SEQUENCE_EXTENDED_8, 2026-09-05. Calibration "
        "ablation as 26, anchor set as a training set as 27, fine "
        "tuning as 28 with small backbones as an arm inside it. "
        "Renumbered nothing"
    ),
}


#: **[DRAFT 2026-08-31 -- NOT LOCKED] EXIT CRITERIA.** The lock waits on
#: the ruling on the **arm-set rule** (``ARM_SET_RULE_PROPOSED``,
#: whose placeholder arrived empty) and **Arm B's consistency threshold**
#: (``CONSISTENCY_THRESHOLD_PROPOSED``, whose HIGH half is the weaker of
#: the two).
EXIT_CRITERIA_DRAFT = {
    "status": (
        "DRAFT -- not locked until it is ruled the ARM-SET RULE and "
        "ARM B's CONSISTENCY THRESHOLD"
    ),
    "drafted": "2026-08-31",
    "criteria": (
        "1. ARM A: five seeds on the declared shared seed basis, the "
        "ensemble's per-seed OOF PCC, and the paired contrast against "
        "the probe under BOTH of PLAN 4.3's conditions with the margin "
        "reported",
        "2. THE ARM SET DECLARED BEFORE ANY AVERAGING, by a rule that "
        "never consults OOF PCC, with the rule recorded and the "
        "SELECTION_ON_EVALUATION_DATA_PROHIBITION carried on every "
        "figure",
        "3. the all-68 (or all-64) variant reported DESCRIPTIVELY beside "
        "the declared set",
        "4. THE COHORT QUESTION ANSWERED EXPLICITLY: the four "
        "236-patient arms either excluded by the rule or handled by "
        "intersection, with the choice recorded and never made silently "
        "(COHORT_AND_SEED_HETEROGENEITY)",
        "5. ARM B: the pairwise consistency matrix on BOTH binarisations "
        "reported separately, never averaged together",
        "6. the per-patient HARDNESS VECTOR, and the cross-reference "
        "against inter-rater disagreement computed with a TIE-AWARE "
        "method declared in advance (only 26 distinct sd values exist)",
        "7. the inter-rater sd computed by BOTH routes -- soft labels "
        "and run.phase10_rater_grades -- and asserted equal, because the "
        "label join has already produced two defects",
        "8. one committed reading applied verbatim per arm, PLUS the "
        "applicable combination reading; no reading invented after the "
        "numbers; a pattern outside the cells gets a dated observation",
        "9. NO LEDGER ROW for Arm B (descriptive by registration); Arm "
        "A's row is to be ruled and is NOT assumed here",
        "10. THE BINDING carried on every figure: neither arm licenses a "
        "claim about what the 0.2520 is made of "
        "(phase20.RESIDUAL_PROHIBITION)",
        "11. suite green",
    ),
    "what_is_not_yet_settled": (
        "the arm-set rule (the ruling arrived as an empty "
        "placeholder), Arm B's HIGH consistency threshold (the weaker of "
        "the two proposed), and whether Arm A carries a ledger row -- "
        "all the maintainer's"
    ),
    # [2026-09-01] All settled and the criteria LOCKED; this draft is
    # preserved as written. See EXIT_CRITERIA.
    "superseded_2026_09_01": "EXIT_CRITERIA",
}


#: **[LOCKED 2026-09-01] THE EXIT CRITERIA. The three rulings are closed
#: and NOTHING IS ADDED AFTER THIS RECORD.**
#:
#: **The three, closed at the lock:**
#:
#:     1. the arm set        RULED -- ARM_SET_RULED (64 arms, 237 cohort)
#:     2. the seed basis     RULED -- SEED_BASIS_RULED (the shared five)
#:     3. Arm B's thresholds RULED -- CONSISTENCY_THRESHOLDS_RULED
#:
#: **Nothing is added after this record.** A criterion discovered missing
#: later is a **limitation of the lock, recorded as such** -- never a
#: retro-fitted entry. Phase 20 wrote this clause, then honoured it
#: within a day when its own stratification check turned out to be
#: missing (``phase20.LOCK_LIMITATION_STRATIFICATION_UNVERIFIED``), so
#: the clause is live and has already cost something.
EXIT_CRITERIA = {
    "locked": (
        "2026-09-01 -- **nothing is added after this record**; a "
        "criterion discovered missing later is a limitation of the lock, "
        "recorded as such, never a retro-fitted entry"
    ),
    "the_three_rulings_closed": {
        "arm_set": "RULED -- ARM_SET_RULED, 64 arms on the 237 cohort",
        "seed_basis": "RULED -- SEED_BASIS_RULED, the shared five",
        "thresholds": (
            "RULED -- CONSISTENCY_THRESHOLDS_RULED, LOW <= 0.05 derived "
            "and HIGH >= 0.50 by convention, asymmetry preserved"
        ),
    },
    "criteria": (
        "1. ARM A: the per-seed simple mean over the 64 declared arms' "
        "per-patient OOF predictions, five shared seeds, with the "
        "ensemble's per-seed and mean OOF PCC reported",
        "2. ARM A CONTRASTED AGAINST BOTH ANCHORS: the probe (0.2520) "
        "under the FULL criterion -- paired BCa over patients, both of "
        "PLAN 4.3's conditions, margin reported -- AND the withdrawn "
        "Phase 7B concat result (+0.0073), per the conceded coverage. An "
        "ensemble gain quoted without the concat precedent beside it "
        "would read as though combination had never been tried",
        "3. THE ARM SET AS RULED, declared before any averaging, with "
        "SELECTION_ON_EVALUATION_DATA_PROHIBITION carried on every "
        "figure and the ONE exclusion named with its data-property "
        "reason",
        "4. THE ALL-68 VARIANT IS NOT RUN -- it would cross cohort sizes "
        "(64 arms on 237, 4 on 236). Its absence is RECORDED WITH THE "
        "REASON, not left as a missing deliverable",
        "5. ARM B: the pairwise consistency matrix on BOTH binarisations "
        "-- residual sign and worst-quartile membership -- reported "
        "SEPARATELY and never averaged together",
        "6. the per-patient HARDNESS VECTOR, and the cell that fires "
        "read against CONSISTENCY_THRESHOLDS_RULED with the asymmetry "
        "between LOW and HIGH stated wherever either is quoted",
        "7. THE CROSS-REFERENCE: hardness against inter-rater "
        "disagreement by the declared tie-aware statistic "
        "(DISAGREEMENT_STATISTIC_PROPOSED) with its permutation null at "
        "the OBSERVED tie structure, and the tie structure reported "
        "beside the result",
        "8. THE RATER SD COMPUTED BY BOTH ROUTES -- soft labels and "
        "run.phase10_rater_grades -- and asserted equal, because the "
        "label join has already produced two defects",
        "9. one committed reading applied verbatim per arm, PLUS the "
        "applicable combination reading; no reading invented after the "
        "numbers; a pattern outside the cells gets a dated observation",
        "10. NO LEDGER ROW for Arm B (descriptive by registration); Arm "
        "A's row is to be ruled and is NOT assumed here",
        "11. THE BINDING carried on every figure: neither arm licenses a "
        "claim about what the 0.2520 is made of "
        "(phase20.RESIDUAL_PROHIBITION)",
        "12. suite green",
    ),
    "nothing_added_after": (
        "Phase 20 wrote this clause and then HONOURED IT within a day, "
        "when its own stratification check turned out to be missing "
        "(phase20.LOCK_LIMITATION_STRATIFICATION_UNVERIFIED). The clause "
        "is live and has already cost something"
    ),
    "what_this_lock_does_not_cover": (
        "**TAU_B_DEFECT_CORRECTED's downstream question.** Whether Phase "
        "18's six banked tau figures need re-deriving is a ruling owed, is "
        "not a Phase 21 criterion, and is not folded in here -- naming "
        "it inside the lock would be exactly the retro-fit the clause "
        "above forbids"
    ),
}


#: **[CORRECTED 2026-09-02] The 17 was never counted.**
#:
#: ``PHASE_SEQUENCE_EXTENDED_7``'s Phase 23 motivation read **"17
#: unresolved ledger rows"**. The ledger holds **8
#: ``UNRESOLVED-WITHDRAWN`` and 11 ``WITHDRAWN`` = 19 rows**, and 17
#: matches no grouping of it.
#:
#: **It originated in conversation with the record, was repeated across many
#: turns, and entered the amendment as a phase's STATED MOTIVATION
#: without ever being counted.** Filed beside the other named the record
#: errors.
THE_SEVENTEEN_WAS_NEVER_COUNTED = {
    "corrected": "2026-09-02, at the Phase 23 verification",
    "the_original": (
        "**PRESERVED VERBATIM**: 'and by **17 unresolved ledger rows** "
        "that a ROPE would restate as QUANTIFIED EQUIVALENCE rather "
        "than failure to reject'"
    ),
    "the_count": (
        "**19.** Filtering the ledger's own status field: 8 "
        "UNRESOLVED-WITHDRAWN (indices 8, 29, 30, 31, 32, 34, 35, 36) "
        "and 11 WITHDRAWN (1, 4, 5, 7, 10, 15, 16, 25, 26, 27, 28). "
        "**17 is neither 8, nor 11, nor 19**, and no subset of the "
        "statuses produces it"
    ),
    "whose_error": (
        "**the record's.** It originated in conversation, was repeated "
        "across many turns without challenge, and was written into the "
        "seventh amendment as the motivation for a scheduled phase. "
        "**No turn ever counted it.** The same shape as the 0.477 "
        "misattribution and the 'largest deltas' error: a figure "
        "asserted fluently enough that nobody asked where it came from"
    ),
    "what_it_cost": (
        "**nothing measured, and one phase's motivation misstated.** "
        "Phase 23 was scheduled partly on this number. The motivation "
        "SURVIVES the correction -- 19 rows is more than 17, so the "
        "case for a ROPE is if anything stronger -- which is why it "
        "went unchallenged: **a wrong number in the direction that "
        "does not change the decision is the hardest kind to notice**"
    ),
    "a_plausible_origin_offered_as_a_HYPOTHESIS": (
        "**roadb.PHASE_7_PAIRED_RESULTS records 'withdrawn: 17'** -- "
        "Road B's 20 paired contrasts, 3 survived, 17 withdrawn. A real "
        "17 about a different quantity, sitting one module away. "
        "**Offered as a hypothesis for where the digits came from, NOT "
        "as an established provenance**: nothing in the record connects "
        "them, and the honest statement is that the number's origin is "
        "unknown"
    ),
    "the_unit_the_count_needs": (
        "**ROWS AGGREGATE CONTRASTS**, so a row count and a contrast "
        "count are different numbers and neither may be written as the "
        "other. Ledger row 5 (`p7c-nine-verdicts`) is **nine verdicts "
        "in one row**; row 7 is two contrasts. So **19 ROWS** and **29 "
        "WITHDRAWN CONTRASTS** (ladder.COHORT_CANNOT_RESOLVE: 30 "
        "tested, 1 survived, 29 withdrawn) are both correct about "
        "different things -- **exactly the eight-versus-twelve "
        "distinction** entry 38 and "
        "phase22.CONDITION_PHENOMENON_COUNT_RECONCILED already draw"
    ),
    "the_sweep": (
        "**ONE occurrence in the ledger-row sense**, corrected here. "
        "The only other numeric 17 in the record is "
        "roadb.PHASE_7_PAIRED_RESULTS['withdrawn'], which is Road B's "
        "20-pair count and is CORRECT AS IT STANDS -- not touched"
    ),
    "tag": "[MEASURED] -- the count verified at source, the origin not",
}


def summary() -> dict:
    """The phase's registration, importable as one object."""
    return {
        "reckoning": PHASE_21_RECKONING,
        "heterogeneity": COHORT_AND_SEED_HETEROGENEITY,
        "arm_set_rule": ARM_SET_RULE_PROPOSED,
        "selection_prohibition": SELECTION_ON_EVALUATION_DATA_PROHIBITION,
        "arm_a": ARM_A_REGISTERED,
        "arm_b": ARM_B_REGISTERED,
        "inter_rater_disagreement": INTER_RATER_DISAGREEMENT_MEASURED,
        "consistency_threshold": CONSISTENCY_THRESHOLD_PROPOSED,
        "readings": READINGS_COMMITTED,
        "binding": PHASE_21_BINDING,
        "sequence_extended_6": PHASE_SEQUENCE_EXTENDED_6,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        # [2026-09-01] The three rulings, the lock, the statistic
        # proposal, and the defect found while building.
        "arm_set_ruled": ARM_SET_RULED,
        "arm_set_candidate_rejected": ARM_SET_CANDIDATE_REJECTED,
        "seed_basis_ruled": SEED_BASIS_RULED,
        "consistency_thresholds_ruled": CONSISTENCY_THRESHOLDS_RULED,
        "disagreement_statistic": DISAGREEMENT_STATISTIC_PROPOSED,
        "tau_b_defect": TAU_B_DEFECT_CORRECTED,
        "exit_criteria": EXIT_CRITERIA,
        # [2026-09-01] The close-out.
        "arm_a_observed": ARM_A_OBSERVED,
        "arm_b_observed": ARM_B_OBSERVED,
        "post_hoc": POST_HOC_ANALYSES,
        "shrinkage_mechanism": SHRINKAGE_MECHANISM,
        "scale_invariance_prohibition": SCALE_INVARIANCE_PROHIBITION,
        "shrinkage_reading_error": SHRINKAGE_READING_ERROR_PROVENANCE,
        "arm_a_explained_by_arm_b": ARM_A_EXPLAINED_BY_ARM_B,
        "ranking_losses_noted": RANKING_LOSSES_NOTED,
        "closing": PHASE_21_CLOSING,
        # [2026-09-01] The seventh amendment, a dated addendum to a
        # closed phase: schedules 22-25 and records 19 as OPEN.
        "sequence_extended_7": PHASE_SEQUENCE_EXTENDED_7,
        "the_seventeen_was_never_counted": THE_SEVENTEEN_WAS_NEVER_COUNTED,
        "write_up_number_open": WRITE_UP_NUMBER_OPEN,
        # [2026-09-01] Three items before Phase 22 is scoped.
        "write_up_number_ruled": WRITE_UP_NUMBER_RULED,
        "shrinkage_figure_misattributed": SHRINKAGE_FIGURE_MISATTRIBUTED,
        "cohort_pair_separation": COHORT_PAIR_SEPARATION_DESIGNED,
        "pairing_axis_assumption": PAIRING_AXIS_ASSUMPTION,
        # [2026-09-01] Two reckoning inputs for Phase 22, built and
        # configured; neither is Phase 22 machinery.
        "cohort_pair_separation_built": COHORT_PAIR_SEPARATION_BUILT,
        "cross_arm_shrinkage": CROSS_ARM_SHRINKAGE_REGISTERED,
        # [2026-09-01] The dispersion statistic ruled, with the
        # rejected range version and the measurement that killed it.
        "shrinkage_dispersion_ruled": SHRINKAGE_DISPERSION_STATISTIC_RULED,
        # [2026-09-01] Both reckoning measurements ran; the arm A
        # control and what it forces on Phase 22.
        "cohort_pair_separation_observed": COHORT_PAIR_SEPARATION_OBSERVED,
        "cross_arm_shrinkage_observed": CROSS_ARM_SHRINKAGE_OBSERVED,
        "arm_a_shrinkage_control": ARM_A_THE_SHRINKAGE_CONTROL,
        "phase_22_consequence": PHASE_22_CONSEQUENCE_RECORDED,
    }


# ==========================================================================
# [2026-09-01] THE CLOSE-OUT. Both arms ran at sha feb53133, finalized
# single-attempt, CPU-only on cached predictions.
#
# PROVENANCE, once and bindingly: every figure below reached this record
# SECOND-HAND from the runs' own outputs. No keeper run directory is
# reachable from this machine. What this session verified is the
# ARITHMETIC INSIDE the figures, recorded per entry.
#
# NO PATIENT IDS ANYWHERE. Patient-keyed values are CLUSTER-ONLY;
# only counts and aggregate statistics appear here.
# ==========================================================================


#: **[OBSERVED 2026-09-01] ARM A -- the ensemble probe. Cell
#: ``a_above_but_unresolved`` FIRED: the registered expectation, as
#: written.**
#:
#: 64 arms x 5 seeds. Per-seed ensemble PCC **0.2827 / 0.2842 / 0.3039 /
#: 0.2398 / 0.2668**, mean **0.2755** -- **the highest figure the project
#: has produced**, above the best single arm (``p11_mse_control``
#: 0.2552) and above the probe (0.2520).
#:
#: **And it is UNRESOLVED.** Per-seed paired deltas **+0.0480 / +0.0332 /
#: +0.0320 / -0.0020 / +0.0060**, every interval spanning zero, **0 of
#: 5**, one seed negative. Mean gain **+0.0234**, against the withdrawn
#: Phase 7B concat precedent of +0.0073 -- 3.2x that precedent and still
#: not claimable.
#:
#: **OBSERVATION, not a registered reading: THE ENSEMBLE IS LESS STABLE
#: THAN THE PROBE.** Seed spread 0.2398-0.3039, **sd 0.0239 against the
#: probe's 0.0148 -- 1.61x wider.** Averaging 64 arms produced a *less*
#: stable estimate, which is the opposite of what ensembling normally
#: buys. Recorded as an observation on ``phase17.UNPREDICTED_PATTERN``'s
#: precedent; no cell anticipated it.
#:
#: **THE DUPLICATE CAVEAT, stated rather than buried**: four of the 64
#: are byte-identical to four others
#: (``phase18.DUPLICATION_PROVEN_AT_DEPTH``), so the mean
#: **double-weights those four**. Every one is a legitimate member under
#: ``ARM_SET_RULED``'s declared rule -- the rule selects on cohort size
#: and never on identity -- but the mean is not an equal-weight average
#: of 64 distinct predictors, and that is said here rather than left for
#: a reader to discover.
ARM_A_OBSERVED = {
    "observed": "2026-09-01, run at sha feb53133, finalized single-attempt",
    "provenance": (
        "read from the run's own outputs; no keeper run "
        "directory is reachable from this machine. The arithmetic below "
        "was re-derived here"
    ),
    "pcc_by_seed": (0.2827, 0.2842, 0.3039, 0.2398, 0.2668),
    "pcc_mean": 0.2755,
    "pcc_sd": 0.0239,
    "n_arms": 64,
    "highest_figure_the_project_has_produced": (
        "0.2755, above the best single arm (p11_mse_control 0.2552) and "
        "above the probe (0.2520)"
    ),
    "paired_deltas_by_seed": (0.0480, 0.0332, 0.0320, -0.0020, 0.0060),
    "n_excluding_zero": "0 of 5, one seed negative",
    "mean_gain": 0.0234,
    "cell_fired": "READINGS_COMMITTED['a_above_but_unresolved']",
    "against_the_concat_precedent": (
        "+0.0234 against Phase 7B's withdrawn +0.0073 -- 3.2x that "
        "precedent and STILL NOT CLAIMABLE. Quoted together, as "
        "EXIT_CRITERIA criterion 2 requires"
    ),
    "observation_the_ensemble_is_less_stable": (
        "**not a registered reading -- an OBSERVATION.** Seed spread "
        "0.2398-0.3039, sd 0.0239 against the probe's 0.0148: **1.61x "
        "WIDER**. Averaging 64 arms produced a LESS stable estimate, the "
        "opposite of what ensembling normally buys. No cell anticipated "
        "it; recorded on phase17.UNPREDICTED_PATTERN's precedent"
    ),
    "the_duplicate_caveat": (
        "four of the 64 are byte-identical to four others "
        "(phase18.DUPLICATION_PROVEN_AT_DEPTH), so **the mean "
        "DOUBLE-WEIGHTS those four**. Every one is a legitimate member "
        "under ARM_SET_RULED -- the rule selects on cohort size and "
        "never on identity -- but the mean is NOT an equal-weight "
        "average of 64 distinct predictors, and that is said here rather "
        "than left for a reader to discover"
    ),
    "arithmetic_checked_here": (
        "the five per-seed values give mean 0.275480 and sd 0.023897; "
        "the five paired deltas give mean 0.023440; 0.2755 - 0.2520 = "
        "0.0235, agreeing with the paired mean to rounding"
    ),
}


#: **[OBSERVED 2026-09-01] ARM B -- error consistency. Both binarisations
#: fire ``b_high_consistency``.**
#:
#: **Residual sign: mean pairwise kappa 0.7012 (sd 0.2236) over 2016
#: pairs. Worst quartile: 0.5729 (sd 0.1741).** Both clear the ruled
#: 0.50, so ``B_HIGH_CONSISTENCY`` fires on both -- reported separately,
#: never averaged.
#:
#: **GEIRHOS'S BANDS AS REGISTERED CONTEXT, NOT A THRESHOLD -- and the
#: comparison is stated as measured, not as summarised.** Human-to-human
#: 0.32-0.48; CNN-to-CNN across architectures 0.62-0.79. Measured against
#: those:
#:
#:     residual sign  0.7012  INSIDE the CNN band, 0.0888 BELOW its top
#:     worst quartile 0.5729  BELOW the CNN floor by 0.0471
#:     both           above the human ceiling (0.48) by +0.2212 / +0.0929
#:
#: **So neither binarisation sits "at or above the top" of the CNN-to-CNN
#: range**: one is inside it and one is below it. Both are well above
#: anything measured between humans. Recorded as measured because a band
#: comparison is exactly the kind of sentence that drifts.
#:
#: **THE CROSS-REFERENCE FIRED AS REGISTERED, AND IT RULES OUT THE
#: COMFORTABLE ANSWER.** tau-b(hardness, rater disagreement) =
#: **+0.0727** (permutation p 0.1226) on residual sign and **-0.0811**
#: (p 0.0835) on worst quartile -- **small, insignificant, and OPPOSITE
#: IN SIGN**. Hardness does not track rater disagreement.
#:
#: The committed cell selects:
#: ``b_hardness_does_not_correlate_with_disagreement``. **Hard patients
#: are not the patients the panel argued about.**
#:
#: **The statistic's own preconditions, measured**: the rater sd agreed
#: to **2.22e-16** across the two derivation routes (soft labels and
#: ``run.phase10_rater_grades``), and disagreement carries **18 distinct
#: values with a largest tie group of 59** -- which is why tau-b with a
#: permutation null was the registered statistic and not Pearson.
ARM_B_OBSERVED = {
    "observed": "2026-09-01, run at sha feb53133, finalized single-attempt",
    "provenance": "read from the run's own outputs",
    "residual_sign": {"mean_kappa": 0.7012, "sd": 0.2236, "n_pairs": 2016},
    "worst_quartile": {"mean_kappa": 0.5729, "sd": 0.1741, "n_pairs": 2016},
    "cell_fired": "READINGS_COMMITTED['b_high_consistency'] -- BOTH",
    "reported_separately_never_averaged": (
        "as ARM_B_REGISTERED requires: the two answer different "
        "questions and collapsing them would be the R2 shape"
    ),
    "geirhos_bands_measured_against": (
        "**stated as MEASURED, not as summarised.** residual sign 0.7012 "
        "is INSIDE the CNN-to-CNN band (0.62-0.79), 0.0888 BELOW its "
        "top; worst quartile 0.5729 is BELOW the CNN floor by 0.0471. "
        "Both are above the human-to-human ceiling (0.48) by +0.2212 and "
        "+0.0929. **So neither sits 'at or above the top' of the "
        "CNN-to-CNN range** -- one is inside it and one below it. "
        "Recorded precisely because a band comparison is exactly the "
        "kind of sentence that drifts"
    ),
    "context_not_a_threshold": (
        "literature.GEIRHOS_ERROR_CONSISTENCY -- reported BESIDE the "
        "result; the locked 0.50 was not moved to meet it"
    ),
    "cross_reference": {
        "residual_sign": {"tau_b": 0.0727, "permutation_p": 0.1226},
        "worst_quartile": {"tau_b": -0.0811, "permutation_p": 0.0835},
        "reading": (
            "**small, insignificant, and OPPOSITE IN SIGN. Hardness does "
            "NOT track rater disagreement**"
        ),
        "cell_fired": (
            "READINGS_COMMITTED"
            "['b_hardness_does_not_correlate_with_disagreement']"
        ),
        "what_it_rules_out": (
            "**the comfortable answer.** 'The residual is label noise' "
            "was the strongest available explanation for the ceiling and "
            "it is refuted: hard patients are NOT the patients the panel "
            "argued about"
        ),
    },
    "the_statistics_preconditions_measured": (
        "the rater sd agreed to **2.22e-16** across the two derivation "
        "routes (soft labels and run.phase10_rater_grades), and "
        "disagreement carries **18 distinct values with a largest tie "
        "group of 59** -- which is why tau-b with a permutation null was "
        "the registered statistic and not Pearson "
        "(DISAGREEMENT_STATISTIC_PROPOSED)"
    ),
    "status": "DESCRIPTIVE, no ledger row",
}


#: **[POST-HOC 2026-09-01, ON THE CLUSTER -- NOT REGISTERED IN
#: ADVANCE] The four analyses that explain the arms.**
#:
#: **Tagged POST-HOC and kept apart from the committed readings.** These
#: were run after the numbers existed. They **explain** the arms; they do
#: not **test** anything committed, and no cell fires on them. A post-hoc
#: analysis that changed a verdict would be the sweep-reporting-its-best
#: failure; these change no verdict and are recorded for what they show.
#:
#: **NO PATIENT IDS.** Patient-keyed values are CLUSTER-ONLY.
POST_HOC_ANALYSES = {
    "tag": "[POST-HOC]",
    "run": "2026-09-01, on the cluster",
    "not_registered_in_advance": (
        "**these were run AFTER the numbers existed.** They EXPLAIN the "
        "arms; they do not TEST anything committed, and NO CELL FIRES ON "
        "THEM. A post-hoc analysis that changed a verdict would be the "
        "sweep-reporting-its-best failure; these change no verdict"
    ),
    "no_patient_ids": (
        "patient-keyed values are CLUSTER-ONLY and appear nowhere here; "
        "only counts and aggregate statistics"
    ),

    "1_hardness_is_bimodal": (
        "**117 of 237 patients sit in the worst quartile of FEWER THAN 4 "
        "of 64 arms; 36 sit in MORE THAN 48; 84 lie between.** Maximum "
        "62.8 of 64. The distribution is bimodal, not a gradient -- most "
        "patients are hard for almost nobody or hard for almost everybody"
    ),
    "2_the_hard_and_easy_sets_differ_in_SPREAD_not_LEVEL": (
        "**near-identical panel means (hard 2.7500 vs easy 2.8017, a gap "
        "of 0.0517) and sharply different spreads (sd 1.2086 vs 0.3077, "
        "3.93x).** The hard set carries **17.6x the grade-1 rater mass** "
        "and **9.7x the grade-5 mass**; the easy set holds over half its "
        "mass on grade 3. The hard patients are the EXTREME ones, not "
        "the mis-centred ones"
    ),
    "3_worst_quartile_hardness_vs_distance_from_the_centre": (
        "**tau-b(worst-quartile hardness, |panel mean - 2.7544|) = "
        "+0.7521** -- the largest POSITIVE association measured in this "
        "project"
    ),
    "4_residual_sign_hardness_vs_signed_distance": (
        "**tau-b(residual-sign hardness, signed distance from 2.7544) = "
        "-0.8470**"
    ),
    "which_is_actually_the_largest": (
        "**-0.8470 is the largest BY MAGNITUDE**, larger than the "
        "+0.7521 above it; +0.7521 is the largest POSITIVE one. Stated "
        "because 'largest association' applied to the smaller of two "
        "adjacent figures is the kind of slip this record exists to "
        "prevent. For scale, Phase 18's largest banked tau is 0.7024"
    ),
}


#: **[MEASURED 2026-09-01] THE FINDING: the arms agree because they all
#: SHRINK, not because they share a representational limitation.**
#:
#: **Both binarisations reduce to one mechanism.** Patients above 2.7544
#: are under-predicted by nearly every arm and patients below are
#: over-predicted (**tau -0.8470**); the worst-quartile errors land on
#: the patients furthest from the centre (**tau +0.7521**). **Error
#: consistency at the top of the CNN-to-CNN band is a SHRINKAGE
#: ARTIFACT.**
#:
#: **THE MECHANISM WAS MEASURED TWICE BEFORE AND NEVER CONNECTED TO THE
#: ERROR STRUCTURE.**
#:
#: * **Phase 20** measured head shrinkage at **0.477 on scrambled
#:   labels** -- a head that fits nothing still compresses toward the
#:   mean.
#: * **Phase 3's gate** carries the interpretation verbatim: *"a head fit
#:   under MSE on 152 samples shrinks toward the label mean (see
#:   shrinkage) and a shrunk predictor keeps its correlation while its
#:   RMSE approaches a constant's."* ``phase3.sanity_report`` computes
#:   ``shrinkage = sd(predictions) / sd(truth)`` on every arm and has
#:   since 2026-07-28.
#:
#: **So the project has been reporting the mechanism on every arm since
#: Phase 3 and never asked what it implied about WHICH PATIENTS the arms
#: get wrong.** That is what Arm B measured, and the answer was already
#: implicit in a number printed beside every run.
SHRINKAGE_MECHANISM = {
    "measured": "2026-09-01",
    "the_finding": (
        "**both binarisations reduce to ONE MECHANISM: the arms agree "
        "because they all SHRINK TOWARD THE LABEL MEAN, not because they "
        "share a representational limitation.** Patients above 2.7544 "
        "are under-predicted by nearly every arm and patients below are "
        "over-predicted (tau -0.8470); the worst-quartile errors land on "
        "the patients furthest from the centre (tau +0.7521). **Error "
        "consistency at the top of the CNN-to-CNN band is a SHRINKAGE "
        "ARTIFACT**"
    ),
    "measured_twice_before_never_connected": (
        "[MIS-ATTRIBUTED -- corrected 2026-09-01, see "
        "SHRINKAGE_FIGURE_MISATTRIBUTED: this is a RUN OUTPUT "
        "(p20_permutation_plain__dc4605bf, seed 1337), NOT a figure in "
        "Phase 20's record] "
        "**Phase 20** measured head shrinkage at **0.477 on SCRAMBLED "
        "labels** -- a head that fits nothing still compresses toward "
        "the mean. **Phase 3's gate** carries the interpretation "
        "verbatim: 'a head fit under MSE on 152 samples shrinks toward "
        "the label mean (see shrinkage) and a shrunk predictor keeps its "
        "correlation while its RMSE approaches a constant's'. "
        "phase3.sanity_report computes shrinkage = sd(predictions) /"
        "sd(truth) on EVERY ARM and has since 2026-07-28"
    ),
    "what_that_means_about_the_project": (
        "**the mechanism has been printed beside every run since Phase 3 "
        "and nobody asked what it implied about WHICH PATIENTS the arms "
        "get wrong.** Arm B measured that, and the answer was already "
        "implicit in a number the gate reports on every arm"
    ),
    "the_ceilings_locus_is_characterised": (
        "SHRINKAGE, not representation. The ceiling has a locus and it "
        "is not an unidentified image property"
    ),
}


#: **[REGISTERED 2026-09-01] THE CONSEQUENCE THAT MUST NOT BE MISREAD --
#: a tested literal**, sibling to
#: ``ladder.DETECTION_FLOOR_PROHIBITION``,
#: ``phase10_annex.ANNEX_PROHIBITION``,
#: ``phase20.RESIDUAL_PROHIBITION`` and
#: ``SELECTION_ON_EVALUATION_DATA_PROHIBITION``.
SCALE_INVARIANCE_PROHIBITION = (
    "SHRINKAGE EXPLAINS THE ERROR STRUCTURE AND CANNOT EXPLAIN THE PCC "
    "CEILING. Nobody may read this phase as 'fix the shrinkage and the "
    "correlation rises'. PEARSON IS SCALE-INVARIANT: rescaling a shrunk "
    "prediction vector about the cohort mean by any positive factor "
    "changes RMSE and leaves PCC EXACTLY UNCHANGED -- verified here to "
    "0.00e+00 across rescalings of 1.5x, 2.0x and 2.5x, and to 1.11e-16 "
    "at the largest, which is float representation and not a change. So "
    "un-shrinking the predictions would improve calibration and move the "
    "project's primary metric NOT AT ALL. The ceiling remains "
    "unexplained; what this phase eliminated is one candidate "
    "explanation for it, which is a narrowing and not an answer. No "
    "phase is gated on any of this."
)


#: **[RECORDED 2026-09-01, DATED] WHERE THE WRONG READING OF ARM B CAME
#: FROM.** Filed beside ``ladder.THE_ERROR_PROVENANCE``,
#: ``phase20.CROSS_TARGET_ERROR_PROVENANCE``,
#: ``phase20.S_DESCRIPTION_ERROR_PROVENANCE``,
#: ``phase12.TWO_VIEW_CLAIM_PROVENANCE``,
#: ``literature.NADEAU_BENGIO_DOES_NOT_APPLY`` and
#: ``phase18.TAU_RECOMPUTATION_PROVENANCE``.
#:
#: **The error.** In conversation, the high consistency was read as
#: *"the same patients defeat every representation -- an unidentified
#: image property"*, and **stated it before running the extremity
#: cross-reference that refutes it.**
#:
#: **Why it was wrong.** The arms do not share a representational
#: limitation; they share an ESTIMATOR property. Every one of them
#: shrinks toward the label mean, so every one of them errs on the
#: extremes -- which is agreement about the ESTIMATOR, not about the
#: IMAGES.
#:
#: **How close the right answer was.** **One command.** The extremity
#: cross-reference that settles it was suggested only AFTER the reading
#: was given. The reading was available, the refutation was cheap, and
#: the order was wrong.
#:
#: **The pattern this is the seventh instance of**: a mechanism proposed
#: from a plausible account before the cheap measurement that would
#: decide it. ``ladder.BASAL_RATIONALE_UNSUPPORTED``'s lesson, in a new
#: costume -- **an interpretation stated in a measured voice is the one
#: nobody re-checks.**
SHRINKAGE_READING_ERROR_PROVENANCE = {
    "recorded": "2026-09-01",
    "the_error": (
        "in conversation the high consistency was read as 'the same "
        "patients defeat every representation -- an unidentified image "
        "property', and **stated it BEFORE running the extremity "
        "cross-reference that refutes it**"
    ),
    "why_it_was_wrong": (
        "the arms do not share a REPRESENTATIONAL limitation; they share "
        "an ESTIMATOR property. Every one shrinks toward the label mean, "
        "so every one errs on the extremes -- **agreement about the "
        "ESTIMATOR, not about the IMAGES**"
    ),
    "how_close_the_right_answer_was": (
        "**one command.** The extremity cross-reference that settles it "
        "was suggested only AFTER the reading was given. The reading was "
        "available, the refutation was cheap, and the order was wrong"
    ),
    "the_pattern": (
        "**the seventh instance**: a mechanism proposed from a plausible "
        "account before the cheap measurement that would decide it. "
        "ladder.BASAL_RATIONALE_UNSUPPORTED's lesson in a new costume -- "
        "an interpretation stated in a measured voice is the one nobody "
        "re-checks"
    ),
    "filed_beside": (
        "ladder.THE_ERROR_PROVENANCE, "
        "phase20.CROSS_TARGET_ERROR_PROVENANCE, "
        "phase20.S_DESCRIPTION_ERROR_PROVENANCE, "
        "phase12.TWO_VIEW_CLAIM_PROVENANCE, "
        "literature.NADEAU_BENGIO_DOES_NOT_APPLY, "
        "phase18.TAU_RECOMPUTATION_PROVENANCE"
    ),
}


#: **[2026-09-01] ARM A EXPLAINED BY ARM B -- the coherent combination
#: cell, FIRED, with the mechanism named.**
#:
#: ``READINGS_COMMITTED["combination_a_fails_and_b_high"]`` registered
#: this as *"the coherent shared-ceiling account ... two arms, one story,
#: and each predicts the other"*. **It fired, and the story now has a
#: mechanism.**
#:
#: **The mean of shrunk predictors is a shrunk predictor.** Averaging 64
#: arms that shrink toward the same mean cannot recover the extremes,
#: because none of them reaches the extremes to begin with. **+0.0234 is
#: what remains** once the shared shrinkage is averaged over -- the
#: residual disagreement between arms, and nothing more.
#:
#: **This is why Arm A was unresolved rather than merely small.** An
#: ensemble recovers signal when its members err INDEPENDENTLY; these
#: members err IDENTICALLY in the direction that matters.
ARM_A_EXPLAINED_BY_ARM_B = {
    "recorded": "2026-09-01",
    "the_cell_fired": (
        "READINGS_COMMITTED['combination_a_fails_and_b_high'] -- 'the "
        "coherent shared-ceiling account ... two arms, one story, and "
        "each predicts the other'. **It fired, and the story now has a "
        "MECHANISM**"
    ),
    "the_mechanism": (
        "**the mean of shrunk predictors is a shrunk predictor.** "
        "Averaging 64 arms that shrink toward the same mean cannot "
        "recover the extremes, because none of them reaches the extremes "
        "to begin with. **+0.0234 is what remains** once the shared "
        "shrinkage is averaged over -- the residual disagreement between "
        "arms, and nothing more"
    ),
    "why_unresolved_rather_than_merely_small": (
        "an ensemble recovers signal when its members err "
        "INDEPENDENTLY; these members err IDENTICALLY in the direction "
        "that matters"
    ),
    "and_it_explains_the_seed_spread_too": (
        "[REASONED, not measured] the ensemble's wider seed spread "
        "(0.0239 vs the probe's 0.0148) is consistent with averaging "
        "over 64 arms whose shared component is fixed and whose "
        "residual disagreement is what varies -- but this was NOT "
        "measured and is offered as a candidate, not a finding"
    ),
}


#: **[NOTED 2026-09-01 -- NOT COMMITTED, NOT REGISTERED] The forward
#: lead.**
#:
#: **Ranking and pairwise losses optimise ORDERING and have no mean to
#: shrink toward.** If the error structure is a shrinkage artifact of an
#: MSE objective, an objective with no central tendency to collapse
#: toward would not produce it.
#:
#: **Noted, not committed.** It is a different objective, which the
#: standing clause fixes at MSE across every banked arm
#: (``phase18.DELIVERABLES_REGISTERED["standing_clause_mse_objective"]``);
#: changing it is a new phase with its own registration, not a follow-on.
#: **And it is NOT a fix for the ceiling** -- ``SCALE_INVARIANCE_
#: PROHIBITION`` applies to it as to everything else in this phase.
RANKING_LOSSES_NOTED = {
    "status": "NOTED, NOT COMMITTED, NOT REGISTERED -- 2026-09-01",
    "the_lead": (
        "**ranking and pairwise losses optimise ORDERING and have no "
        "mean to shrink toward.** If the error structure is a shrinkage "
        "artifact of an MSE objective, an objective with no central "
        "tendency to collapse toward would not produce it"
    ),
    "why_not_committed": (
        "it is a DIFFERENT OBJECTIVE, which the standing clause fixes at "
        "MSE across every banked arm "
        "(phase18.DELIVERABLES_REGISTERED['standing_clause_mse_objective'"
        "]). Changing it is a NEW PHASE with its own registration, not a "
        "follow-on"
    ),
    "it_is_not_a_fix_for_the_ceiling": (
        "SCALE_INVARIANCE_PROHIBITION applies to it as to everything "
        "else here. A ranking loss might improve calibration; nothing "
        "measured says it would raise PCC"
    ),
    "the_nearest_literature": (
        "literature.SOURCES_BANKED['rankiqa'] -- and its flagged "
        "limitation applies: their ranking signal comes from distortions "
        "applied to ONE image, where the pair's order is known by "
        "construction. Cleft grade ordering between two DIFFERENT "
        "patients is exactly what is uncertain here"
    ),
}


#: **[CLOSED 2026-09-01] PHASE 21. THE ENSEMBLE PROBE AND
#: ERROR-CONSISTENCY DIAGNOSIS.**
#:
#: Two runs at sha ``feb53133``, finalized single-attempt, CPU-only on
#: cached predictions. **No new artifact, no new hash, no GPU, no ledger
#: row.**
#:
#: **THE HEADLINE: the ceiling's locus is SHRINKAGE, not
#: representation -- and naming it does not lift it.**
PHASE_21_CLOSING = {
    "closed": (
        "2026-09-01 -- two runs at sha feb53133, finalized "
        "single-attempt, CPU-only on cached predictions"
    ),

    "criterion_1_arm_a": (
        "MET -- ARM_A_OBSERVED. Five seeds on the declared shared basis, "
        "per-seed ensemble OOF PCC reported, mean 0.2755"
    ),
    "criterion_2_both_anchors": (
        "MET -- the probe (0.2520) under the FULL criterion: per-seed "
        "paired deltas, 0 of 5 excluding zero, mean +0.0234, UNRESOLVED; "
        "and the withdrawn Phase 7B concat precedent (+0.0073) quoted "
        "beside it"
    ),
    "criterion_3_the_arm_set_as_ruled": (
        "MET -- 64 arms on the 237 cohort, declared before any "
        "averaging, SELECTION_ON_EVALUATION_DATA_PROHIBITION carried, "
        "the one exclusion named with its data-property reason"
    ),
    "criterion_4_all_68_not_run": (
        "MET -- not run, and the reason recorded: it would cross cohort "
        "sizes"
    ),
    "criterion_5_both_binarisations_separately": (
        "MET -- residual sign 0.7012 (sd 0.2236) and worst quartile "
        "0.5729 (sd 0.1741) over 2016 pairs, reported separately and "
        "never averaged"
    ),
    "criterion_6_hardness_and_the_asymmetry": (
        "MET -- the hardness vector reported; both cells read against "
        "CONSISTENCY_THRESHOLDS_RULED with the LOW-derived / "
        "HIGH-convention asymmetry stated"
    ),
    "criterion_7_the_cross_reference": (
        "MET -- tau-b with a permutation null at the OBSERVED tie "
        "structure (18 distinct values, largest tie group 59), reported "
        "with that structure beside it"
    ),
    "criterion_8_rater_sd_by_both_routes": (
        "MET -- the two routes agreed to 2.22e-16"
    ),
    "criterion_9_readings_applied_verbatim": (
        "MET -- a_above_but_unresolved, b_high_consistency (both "
        "binarisations), "
        "b_hardness_does_not_correlate_with_disagreement, and the "
        "combination cell combination_a_fails_and_b_high. **One "
        "OBSERVATION recorded beside them**: the ensemble's wider seed "
        "spread, which no cell anticipated"
    ),
    "criterion_10_no_ledger_row": (
        "MET -- Arm B descriptive by registration; Arm A carries no row "
        "either, and none was assumed. The ledger stands at 38"
    ),
    "criterion_11_the_binding": (
        "MET -- PHASE_21_BINDING carried: neither arm licenses a claim "
        "about what the 0.2520 is made of, and "
        "phase20.RESIDUAL_PROHIBITION is untouched"
    ),
    "criterion_12_suite_green": "MET -- reported with the close-out",

    "the_headline": (
        "**THE CEILING'S LOCUS IS SHRINKAGE, NOT REPRESENTATION -- AND "
        "NAMING IT DOES NOT LIFT IT.** The arms agree about which "
        "patients they get wrong because they all compress toward the "
        "label mean, not because they share a representational limit. "
        "The extremes are where every arm fails, and the mean of shrunk "
        "predictors is a shrunk predictor, which is why the ensemble "
        "recovered +0.0234 and no more"
    ),
    "the_open_question_restated_honestly": (
        "**the PCC ceiling REMAINS UNEXPLAINED, with ONE CANDIDATE "
        "MECHANISM ELIMINATED.** Shrinkage explains the ERROR STRUCTURE "
        "and cannot explain the CEILING, because Pearson is "
        "scale-invariant (SCALE_INVARIANCE_PROHIBITION). This phase "
        "narrowed the question; it did not answer it"
    ),
    "what_was_eliminated": (
        "two things, both by measurement: **label noise** -- hardness "
        "does not track rater disagreement -- and **a shared "
        "representational limitation** -- the agreement is an estimator "
        "property, not an image property"
    ),
    "the_forward_lead": (
        "RANKING_LOSSES_NOTED -- noted, NOT committed, NOT registered"
    ),
    "corrections_recorded_in_this_phase": (
        "two, both the record's: phase18.TAU_RECOMPUTATION_PROVENANCE (the "
        "raw-versus-rank probe, carried here because it was found while "
        "building this phase) and "
        "SHRINKAGE_READING_ERROR_PROVENANCE (the representational "
        "reading, stated before the measurement that refutes it)"
    ),
    "post_hoc_kept_apart": (
        "POST_HOC_ANALYSES are tagged as such and fire no cell. They "
        "explain the arms; they test nothing committed"
    ),
}


#: **[DECIDED 2026-09-01] THE SEVENTH SEQUENCE AMENDMENT --
#: schedules 22, 23, 24 and 25. SCHEDULED, NOT REGISTERED.**
#:
#: **It lives here, in a closed phase, deliberately.** The established
#: pattern is that an amendment lands in the module current when it is
#: decided (1 -> phase11, 2 -> phase12, 3 and 4 -> phase15, 5 -> phase20,
#: 6 -> phase21). Phase 21 has closed and 22 has not opened, so this is
#: the current module; it is a **dated addendum to a closed phase**, on
#: the ``phase18`` precedent, and it changes nothing Phase 21 measured.
#:
#: **THE DISTINCTION THIS RECORD DRAWS, and the record already draws it
#: twice.** ``phase15.PHASE_16_SCHEDULED`` is **scheduled-not-scoped**:
#: *"the scope is proposed at the phase's RESTATE, not at scheduling ...
#: A scope written before the phase reads the record is a scope written
#: from memory."* ``phase15.ANCHOR_LOOP_REGISTERED`` is
#: **registered-not-built**: a full design, no code.
#:
#: **These four are SCHEDULED-NOT-REGISTERED -- one step earlier than
#: either.** A place in the sequence and a motivation, and nothing else.
#:
#: **THE BINDING, three clauses:**
#:
#: 1. **Each phase's scope, exit criteria and readings are written at its
#:    own restate, NEVER HERE.** Motivation is not scope.
#: 2. **A scheduled phase that does not run is recorded as CANCELLED WITH
#:    A REASON, never silently dropped.** A schedule that quietly loses
#:    entries is indistinguishable from one that never had them.
#: 3. **Each may be REORDERED OR CANCELLED on evidence from an earlier
#:    one.** Phase 22's result may change what 24 should measure --
#:    **which is exactly why no readings are written now.**
PHASE_SEQUENCE_EXTENDED_7 = {
    "decided": (
        "2026-09-01 -- the SEVENTH sequence amendment; "
        "schedules 22-25, SCHEDULED-NOT-REGISTERED"
    ),
    "why_it_lives_in_a_closed_phase": (
        "the established pattern puts an amendment in the module CURRENT "
        "WHEN DECIDED (1 -> phase11, 2 -> phase12, 3 and 4 -> phase15, "
        "5 -> phase20, 6 -> phase21). Phase 21 has closed and 22 has not "
        "opened, so this is the current module. A DATED ADDENDUM to a "
        "closed phase, on the phase18 precedent; it changes nothing "
        "Phase 21 measured"
    ),

    # ---- the full backward chain, all six ----------------------------
    "first_amendment": "phase11.PHASE_SEQUENCE_RENUMBERED (view ablation to 12)",
    "second_amendment": "phase12.PHASE_SEQUENCE_RENUMBERED_2 (LDL to 14, "
                        "second beauty dataset to 15, TSTR to 16)",
    "third_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_3 (metric-space "
                       "ablation to 16, TSTR to 17, write-up to 18)",
    "fourth_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_4 (anchor loop "
                        "to 16, metric-space ablation to 18, write-up to 19)",
    "fifth_amendment": "phase20.PHASE_SEQUENCE_EXTENDED_5 (permutation "
                       "control appended as 20; renumbered nothing; "
                       "adopted the write-up-runs-last rule)",
    "sixth_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_6 (ensemble probe "
                       "and error-consistency diagnosis appended as 21; "
                       "renumbered nothing)",

    "was": {
        "21": "the ensemble probe and error-consistency diagnosis (closed)",
        "22": "**NOTHING -- the chain stopped at 21**",
    },
    "becomes": {
        "22": "RANKING AND PAIRWISE LOSSES",
        "23": "THE STATISTICAL INSTRUMENTS",
        "24": "ALL FIVE RATERS",
        "25": "FOUNDATION-MODEL FEATURES (DINOv2)",
    },
    "status_changes": {
        "write_up": (
            "UNRESOLVED at this amendment -- see WRITE_UP_NUMBER_OPEN. "
            "[RULED (a) 2026-09-01: Phase 19 -> Phase 19 (unchanged); "
            "WRITE_UP_NUMBER_RULED]"
        ),
        "ranking_losses": "noted (phase21.RANKING_LOSSES_NOTED) -> PHASE 22",
        "statistical_instruments": (
            "noted (literature.SPLIT_RANDOMISED_ARM_NOTED, "
            "literature.SOURCES_BANKED['benavoli_rope']) -> PHASE 23"
        ),
        "all_five_raters": "unnumbered -> PHASE 24",
        "foundation_features": "unnumbered -> PHASE 25",
    },

    # ---- what each is, and no more -----------------------------------
    "phase_22_ranking_and_pairwise_losses": {
        "status": "SCHEDULED, NOT REGISTERED",
        "motivated_by": (
            "**Phase 21's measurement: every arm shrinks toward the "
            "label mean.** [run output, not a phase20 record figure -- "
            "SHRINKAGE_FIGURE_MISATTRIBUTED] phase20 measured head "
            "shrinkage at 0.477 on "
            "SCRAMBLED labels, and phase21 measured tau -0.847 between "
            "residual-sign hardness and signed distance from the cohort "
            "mean. **Ranking objectives have no mean to shrink toward**"
        ),
        "pair_source_ruled": (
            "**BOTH** -- the ruling: synthetic TPS ordering, known "
            "BY CONSTRUCTION, and cohort pairs from the panel mean; "
            "following RankIQA's rank-pretrain-then-fine-tune design"
        ),
        "the_caveat_that_travels": (
            "literature.SOURCES_BANKED['rankiqa']: **RankIQA's pair order "
            "is CERTAIN and ours is the UNCERTAIN THING.** Their ranking "
            "signal comes from distortions applied to ONE image; cleft "
            "grade ordering between two DIFFERENT patients is exactly "
            "what is in question. The synthetic half inherits their "
            "certainty; the cohort half does not"
        ),
        "what_is_not_decided_here": (
            "the loss, the pair-sampling rule, the pretrain/fine-tune "
            "split, the seeds, the arms, the criteria and every reading"
        ),
    },

    "phase_23_the_statistical_instruments": {
        "status": "SCHEDULED, NOT REGISTERED",
        "what": (
            "Bouthillier's split-randomised P(A>B) and Benavoli's ROPE"
        ),
        "motivated_by": (
            "**the banked limitation that five seeds on a fixed split is "
            "Bouthillier's BIASED ESTIMATOR** "
            "(literature.BOUTHILLIER_SAMPLE_SIZE: 'severe "
            "underestimations of standard error'), and by **19 "
            "unresolved-or-withdrawn ledger ROWS** that a ROPE would "
            "restate as QUANTIFIED EQUIVALENCE rather than failure to "
            "reject. **[CORRECTED 2026-09-02: this read '17 unresolved "
            "ledger rows'. The ledger holds 8 UNRESOLVED-WITHDRAWN and "
            "11 WITHDRAWN = 19; 17 matches no grouping. "
            "phase21.THE_SEVENTEEN_WAS_NEVER_COUNTED.]** ROWS is the "
            "unit and is now said: rows aggregate contrasts, so 19 rows "
            "and 29 withdrawn contrasts are both correct about "
            "different things"
        ),
        "neither_raises_pcc": (
            "**both change what can be SAID about every banked figure; "
            "neither changes a figure.** And both are cheap on cached "
            "embeddings"
        ),
        "what_is_not_decided_here": (
            "whether banked contrasts are re-expressed or left alone, "
            "the ROPE's width, N for the re-splitting, and every reading"
        ),
    },

    "phase_24_all_five_raters": {
        "status": "SCHEDULED, NOT REGISTERED",
        "what": (
            "label-distribution head, rater random effects, "
            "agreement-weighted training, disagreement as an auxiliary "
            "target"
        ),
        "sharpened_by_phase_21": (
            "**hardness does NOT track disagreement** "
            "(tau +0.0727 / -0.0811, both insignificant), **so the "
            "extremes are not the contested cases.** A rater-aware "
            "design must therefore be motivated by something other than "
            "'the hard patients are the disputed ones' -- which Phase 21 "
            "refuted"
        ),
        "what_is_not_decided_here": (
            "which of the four designs is built, whether any is, the "
            "target, the criteria and every reading"
        ),
    },

    "phase_25_foundation_model_features": {
        "status": "SCHEDULED, NOT REGISTERED",
        "what": "DINOv2 -- one extraction pass through the existing probe",
        "why_last_of_the_compute_phases": (
            "**the project's own evidence says feature source has moved "
            "little**: MEBeauty 0.2537 against ImageNet 0.2520. A new "
            "backbone is the most expensive intervention with the "
            "weakest prior, so it is sequenced last"
        ),
        "what_is_not_decided_here": (
            "the variant, the extraction geometry, whether it is one arm "
            "or several, the criteria and every reading"
        ),
    },

    # ---- the binding -------------------------------------------------
    "binding_1_scope_at_the_restate": (
        "**each phase's scope, exit criteria and readings are written at "
        "ITS OWN RESTATE, NEVER HERE.** Motivation is not scope. The "
        "precedent is phase15.PHASE_16_SCHEDULED: 'a scope written "
        "before the phase reads the record is a scope written from "
        "memory'"
    ),
    "binding_2_cancellation_is_recorded": (
        "**a scheduled phase that does not run is recorded as CANCELLED "
        "WITH A REASON, never silently dropped.** A schedule that "
        "quietly loses entries is indistinguishable from one that never "
        "had them"
    ),
    "binding_3_reorder_on_evidence": (
        "**each may be REORDERED OR CANCELLED on evidence from an "
        "earlier one. Phase 22's result may change what 24 should "
        "measure -- which is exactly why no readings are written now.** "
        "Readings written before the evidence that shapes the question "
        "are readings written from memory"
    ),
    "scheduled_not_registered": (
        "**one step earlier than either existing precedent.** "
        "phase15.PHASE_16_SCHEDULED is SCHEDULED-NOT-SCOPED; "
        "phase15.ANCHOR_LOOP_REGISTERED is REGISTERED-NOT-BUILT (a full "
        "design, no code). These four have **a place in the sequence and "
        "a motivation, and nothing else**"
    ),
    "module_names_never_change": (
        "unchanged: a module file is named for the number its phase "
        "HOLDS, and a record KEY keeps the number it was WRITTEN with. "
        "Nothing is renamed because nothing moved"
    ),
    "nothing_silently_renumbered": (
        "nothing moved -- 19, 20 and 21 are untouched by this amendment. "
        "The write-up's NUMBER is a separate open question and is NOT "
        "resolved here (WRITE_UP_NUMBER_OPEN)"
    ),
    "eighth_amendment": (
        "phase25.PHASE_SEQUENCE_EXTENDED_8, 2026-09-05. Calibration "
        "ablation as 26, anchor set as a training set as 27, fine "
        "tuning as 28 with small backbones as an arm inside it. "
        "Renumbered nothing"
    ),
}


#: **[OPEN 2026-09-01 -- REPORTED, NOT DECIDED] THE STATUS OF 19.**
#:
#: **What the chain records, quoted verbatim.**
#:
#: ``phase15.PHASE_SEQUENCE_RENUMBERED_4`` (the maintainer, 2026-08-24):
#: ``becomes["19"] = 'write-up'`` and
#: ``status_changes["write_up"] = 'Phase 18 -> Phase 19'``.
#:
#: ``phase20.PHASE_SEQUENCE_EXTENDED_5`` (the maintainer, 2026-08-31):
#: ``becomes["19"] = 'write-up (UNCHANGED in place and content)'`` and
#: ``status_changes["write_up"] = 'Phase 19 -> Phase 19 (unchanged)'``.
#:
#: ``phase21.PHASE_SEQUENCE_EXTENDED_6``: ``becomes["19"] = 'write-up
#: (UNCHANGED in place and content)'``.
#:
#: **The rule's own words**: *"THE WRITE-UP RUNS LAST REGARDLESS OF ITS
#: NUMBER, because it is defined as the phase that consumes all the
#: others."*
#:
#: **THE TWO CANDIDATE RULINGS:**
#:
#: **(a) 19 remains the write-up's registered number and simply runs
#: last.** This is what the write-up-runs-last rule was adopted to
#: permit.
#:
#: **(b) 19 is retired and the write-up takes a fresh number when it
#: opens.** the maintainer has indicated this -- *"19 is void and never used"*.
#:
#: **WHAT THE CHAIN'S OWN WORDING SUPPORTS: (a). Four things point that
#: way and none points the other.**
#:
#: 1. **Three explicit assignments of 19 to the write-up**, two of them
#:    the maintainer's own amendments.
#: 2. **"regardless of ITS number" presupposes the write-up HAS one.**
#:    The rule says the number does not determine execution order; it
#:    does not say the number is void.
#: 3. **The rejected alternative was exactly "the write-up takes a
#:    different number".** The fifth amendment considered moving the
#:    write-up to 20 and rejected it because *"the write-up would move on
#:    every future addition, unboundedly. It has already moved twice
#:    (18 -> 19 across two amendments) without ever running."* Retiring
#:    19 and assigning a fresh number at opening is that same move,
#:    deferred -- a third relocation of a phase whose repeated relocation
#:    was the stated reason for the rule.
#: 4. **The stated cost presupposes 19 is live**: *"a reader must not
#:    infer execution order from 19 vs 20."* There is no such cost if 19
#:    is void.
#:
#: **THIS IS NOT A RECOMMENDATION AGAINST (b).** the maintainer may rule (b);
#: it is the project sequence. **What this record establishes is that (b)
#: would
#: be a REVERSAL of the fourth and fifth amendments, not a clarification
#: of them, and must be recorded as one** -- with the write-up's number
#: moving a third time, and the fifth amendment's stated reason for
#: refusing exactly that addressed.
#:
#: **A number ruled  in the fourth amendment is not voided on
#: a paraphrase**, which is why this is reported and not applied.
WRITE_UP_NUMBER_OPEN = {
    "status": (
        "[RULED 2026-09-01 -- candidate (a), see WRITE_UP_NUMBER_RULED] "
        "OPEN 2026-09-01 -- REPORTED, NOT DECIDED. the ruling is"
    ),
    "what_the_chain_records": (
        "phase15.PHASE_SEQUENCE_RENUMBERED_4: becomes['19'] = 'write-up', "
        "status_changes['write_up'] = 'Phase 18 -> Phase 19'. "
        "phase20.PHASE_SEQUENCE_EXTENDED_5: becomes['19'] = 'write-up "
        "(UNCHANGED in place and content)', status_changes['write_up'] = "
        "'Phase 19 -> Phase 19 (unchanged)'. "
        "phase21.PHASE_SEQUENCE_EXTENDED_6: becomes['19'] = 'write-up "
        "(UNCHANGED in place and content)'"
    ),
    "the_rules_own_words": (
        "THE WRITE-UP RUNS LAST REGARDLESS OF ITS NUMBER, because it is "
        "defined as the phase that consumes all the others"
    ),
    "candidate_a": (
        "19 REMAINS the write-up's registered number and simply runs "
        "last -- what the write-up-runs-last rule was adopted to permit"
    ),
    "candidate_b": (
        "19 is RETIRED and the write-up takes a fresh number when it "
        "opens. the maintainer has indicated this -- '19 is void and never "
        "used'"
    ),
    "what_the_chains_wording_supports": (
        "**(a). Four things point that way and none points the other.** "
        "(1) THREE explicit assignments of 19 to the write-up, two of "
        "them the maintainer's own amendments. (2) 'regardless of ITS number' "
        "PRESUPPOSES the write-up has one -- the rule says the number "
        "does not determine execution order, not that it is void. "
        "(3) The rejected alternative was EXACTLY 'the write-up takes a "
        "different number': the fifth amendment refused moving it to 20 "
        "because 'the write-up would move on every future addition, "
        "unboundedly. It has already moved twice (18 -> 19 across two "
        "amendments) without ever running.' Retiring 19 is that same "
        "move, deferred. (4) The stated cost -- 'a reader must not infer "
        "execution order from 19 vs 20' -- PRESUPPOSES 19 is live; there "
        "is no such cost if it is void"
    ),
    "this_is_not_a_recommendation_against_b": (
        "**(b) may be ruled; it is the project sequence.** What this record "
        "establishes is that (b) would be a **REVERSAL of the fourth and "
        "fifth amendments, not a clarification of them**, and must be "
        "recorded as one -- with the write-up's number moving a THIRD "
        "time, and the fifth amendment's stated reason for refusing "
        "exactly that addressed"
    ),
    "why_it_is_reported_and_not_applied": (
        "**a number ruled  in the fourth amendment is not "
        "voided on a paraphrase.** The seventh amendment "
        "therefore leaves 19 untouched and records the question as open"
    ),
}


#: **[RULED 2026-09-01] THE WRITE-UP'S NUMBER: (a). 19 remains
#: the write-up's registered number and runs last.**
#:
#: ``WRITE_UP_NUMBER_OPEN`` presented two candidates and refused to
#: choose. the maintainer has ruled **(a)**: 19 stays, and the
#: write-up-runs-last rule
#: (``phase20.PHASE_SEQUENCE_EXTENDED_5``) does the work it was adopted
#: to do.
#:
#: **The chain's four supporting points, as reported:**
#:
#: 1. **Three explicit assignments of 19 to the write-up** --
#:    ``PHASE_SEQUENCE_RENUMBERED_4["becomes"]["19"] = 'write-up'``,
#:    ``PHASE_SEQUENCE_EXTENDED_5``'s ``'write-up (UNCHANGED in place and
#:    content)'``, and ``PHASE_SEQUENCE_EXTENDED_6`` repeating it. Two
#:    are the maintainer's own amendments.
#: 2. **"regardless of ITS number" presupposes the write-up HAS one.**
#:    The rule governs execution order, not numbering.
#: 3. **The fifth amendment already considered and rejected relocating
#:    the write-up** -- to 20 -- *"because the write-up would move on
#:    every future addition, unboundedly. It has already moved twice
#:    (18 -> 19 across two amendments) without ever running."*
#: 4. **The stated cost presupposes 19 is live**: *"a reader must not
#:    infer execution order from 19 vs 20."* There is no such cost if 19
#:    is void.
#:
#: **(b) WAS NOT TAKEN, and would have been a REVERSAL of the fourth and
#: fifth amendments rather than a clarification of them** -- moving the
#: write-up's number a third time, for the reason the fifth amendment
#: gave for refusing exactly that.
WRITE_UP_NUMBER_RULED = {
    "ruled": "2026-09-01 -- candidate (a)",
    "the_ruling": (
        "**19 REMAINS the write-up's registered number and runs last** "
        "under the write-up-runs-last rule "
        "(phase20.PHASE_SEQUENCE_EXTENDED_5)"
    ),
    "supporting_point_1_three_assignments": (
        "PHASE_SEQUENCE_RENUMBERED_4['becomes']['19'] = 'write-up'; "
        "PHASE_SEQUENCE_EXTENDED_5's 'write-up (UNCHANGED in place and "
        "content)'; PHASE_SEQUENCE_EXTENDED_6 repeating it. **Two are El "
        "the maintainer's own amendments**"
    ),
    "supporting_point_2_regardless_of_its_number": (
        "'THE WRITE-UP RUNS LAST REGARDLESS OF ITS NUMBER' **presupposes "
        "the write-up HAS one**. The rule governs EXECUTION ORDER, not "
        "numbering"
    ),
    "supporting_point_3_relocation_already_rejected": (
        "the fifth amendment considered moving the write-up to 20 and "
        "REFUSED, 'because the write-up would move on every future "
        "addition, unboundedly. It has already moved twice (18 -> 19 "
        "across two amendments) without ever running'"
    ),
    "supporting_point_4_the_stated_cost": (
        "'a reader must not infer execution order from 19 vs 20' -- "
        "**there is no such cost if 19 is void**"
    ),
    "b_was_not_taken": (
        "**(b) -- retiring 19 and assigning a fresh number at opening -- "
        "WOULD HAVE BEEN A REVERSAL of the fourth and fifth amendments, "
        "not a clarification of them**, moving the write-up's number a "
        "THIRD time for the very reason the fifth amendment gave for "
        "refusing exactly that. It was not taken"
    ),
    "supersedes": "WRITE_UP_NUMBER_OPEN, which stands as written",
}


#: **[CORRECTED 2026-09-01, DATED -- ORIGINAL PRESERVED] THE 0.477
#: ATTRIBUTION WAS WRONG, AND IT WAS THE RECORD'S.**
#:
#: ``SHRINKAGE_MECHANISM`` says *"**Phase 20** measured head shrinkage at
#: 0.477 on SCRAMBLED labels"*. **Phase 20 measured no such thing in its
#: record**: ``phase20.py`` contains no occurrence of ``0.477``,
#: ``shrinkage``, ``sanity_report`` or ``prediction_sd``. Swept across
#: ``src/``, ``tests/``, ``docs/`` and ``configs/``, the figure appears
#: **only inside this module and its tests**.
#:
#: **The correct attribution**: a RUN OUTPUT -- Arm P's ``metrics.json``,
#: run ``p20_permutation_plain__dc4605bf``, seed 1337's
#: ``sanity.shrinkage`` -- **supplied in conversation**. The run
#: directory is CLUSTER-ONLY and unreachable from this machine, so the
#: value is **[SUPPLIED, NOT VERIFIED HERE]**.
#:
#: **Whose error it was**: **the record's.** The mis-attribution originated
#: in conversation and was carried into the record at the record's
#: dictation. Filed beside ``ladder.THE_ERROR_PROVENANCE``,
#: ``phase20.CROSS_TARGET_ERROR_PROVENANCE``,
#: ``phase20.S_DESCRIPTION_ERROR_PROVENANCE``,
#: ``phase12.TWO_VIEW_CLAIM_PROVENANCE``,
#: ``literature.NADEAU_BENGIO_DOES_NOT_APPLY``,
#: ``phase18.TAU_RECOMPUTATION_PROVENANCE`` and
#: ``SHRINKAGE_READING_ERROR_PROVENANCE``.
#:
#: **The shape it repeats**: a figure attributed to a PHASE when it came
#: from a RUN. The distinction is the one the project keeps enforcing --
#: a phase's record is auditable in the repo; a run output is not, and
#: saying "Phase 20 measured" invites a reader to look for it where it
#: is not.
SHRINKAGE_FIGURE_MISATTRIBUTED = {
    "corrected": "2026-09-01, dated; the original stands in place",
    "what_the_record_said": (
        "SHRINKAGE_MECHANISM: '**Phase 20** measured head shrinkage at "
        "0.477 on SCRAMBLED labels'"
    ),
    "what_is_true": (
        "**phase20.py contains no occurrence of 0.477, shrinkage, "
        "sanity_report or prediction_sd.** Swept across src/, tests/, "
        "docs/ and configs/, the figure appears ONLY inside this module "
        "and its tests"
    ),
    "the_correct_attribution": (
        "a RUN OUTPUT: Arm P's metrics.json, run "
        "p20_permutation_plain__dc4605bf, seed 1337's sanity.shrinkage, "
        "**supplied in conversation**"
    ),
    "verification_status": (
        "**[SUPPLIED, NOT VERIFIED HERE]** -- the run directory is "
        "CLUSTER-ONLY and unreachable from this machine, so the figure "
        "could not be checked against its own outputs"
    ),
    "whose_error": (
        "**the record's.** The mis-attribution originated in conversation "
        "and was carried into the record at the record's dictation"
    ),
    "the_shape_it_repeats": (
        "**a figure attributed to a PHASE when it came from a RUN.** A "
        "phase's record is auditable in the repo; a run output is not, "
        "and saying 'Phase 20 measured' invites a reader to look for it "
        "where it is not"
    ),
    "filed_beside": (
        "ladder.THE_ERROR_PROVENANCE, "
        "phase20.CROSS_TARGET_ERROR_PROVENANCE, "
        "phase20.S_DESCRIPTION_ERROR_PROVENANCE, "
        "phase12.TWO_VIEW_CLAIM_PROVENANCE, "
        "literature.NADEAU_BENGIO_DOES_NOT_APPLY, "
        "phase18.TAU_RECOMPUTATION_PROVENANCE, "
        "SHRINKAGE_READING_ERROR_PROVENANCE"
    ),

    # ---- is it load-bearing elsewhere? ------------------------------
    "where_else_it_is_load_bearing": (
        "**nowhere outside this module.** Three occurrences, all in "
        "phase21.py (SHRINKAGE_MECHANISM twice, "
        "PHASE_SEQUENCE_EXTENDED_7's Phase 22 motivation once), plus "
        "three test assertions. No ledger row, no config, no other "
        "phase, and no verdict turns on it"
    ),
    "does_any_record_treat_shrinkage_as_general": (
        "**yes, and that is the more serious half.** SHRINKAGE_MECHANISM "
        "says 'the arms agree because they all SHRINK toward the label "
        "mean' -- a claim about EVERY ARM -- while the only numeric "
        "support offered is ONE SEED OF ONE ARM ON SCRAMBLED LABELS. The "
        "general claim is supported by phase3's INTERPRETATION (which is "
        "an argument about MSE heads, not a measurement over arms) and "
        "by phase21's own tau -0.8470, which IS a cross-arm measurement "
        "of the error structure. **The 0.477 was never the load-bearing "
        "evidence; it was an illustration presented as a measurement**"
    ),
    "a_proper_cross_arm_figure_is_cheap_and_reachable": (
        "**sanity_report has printed shrinkage on every arm since "
        "2026-07-28, and it does not need to be read from metrics.json "
        "at all: shrinkage = sd(prediction) / sd(truth) is recomputable "
        "from the cached per-seed CSVs**, whose columns are "
        "(patient_id, truth, prediction, fold) -- the same files Phase "
        "18 and Phase 21 already declare. A cross-arm shrinkage figure "
        "over the locked 68 therefore needs NO new artifact and NO new "
        "hash, exactly as Phase 21's arms did"
    ),
    "what_it_would_take": (
        "one pass of the phase21 reader over the locked arm list, "
        "computing sd(prediction)/sd(truth) per arm per seed. **NOT "
        "BUILT, NOT REGISTERED, and not proposed as a phase** -- "
        "recorded so that the general claim can be measured rather than "
        "illustrated, whenever the maintainer wants it"
    ),
}


#: **[DESIGNED 2026-09-01 -- NOT RUN. READINGS COMMITTED BEFORE IT
#: RUNS.] THE COHORT-PAIR SEPARATION MEASUREMENT.**
#:
#: **Why it exists**: nothing in the record measures how often two
#: patients' panel means differ by less than the panel's own noise, and
#: **Phase 22's cohort arm trains on exactly those pairs.** A pair
#: separated by less than the noise is ordered by chance, and a ranking
#: loss cannot distinguish a coin flip from a signal.
#:
#: **THE QUANTITY.** Over all **C(237, 2) = 27,966** patient pairs: the
#: distribution of ``|mean_i - mean_j|``, and the fraction falling below
#: a noise scale derived from banked reliability rather than chosen.
#:
#: **THE DERIVATION -- from banked figures, and it is a DERIVATION, not a
#: record.**
#:
#:     MEAN_R_237        = 0.4696   (mean inter-rater r)
#:     RELIABILITY_237   = 0.8158   (Spearman-Brown, k = 5; recomputes
#:                                   to 0.8157 from MEAN_R_237)
#:     error fraction    = 1 - 0.8158 = 0.1842
#:     SE(one panel mean)      = sd_obs * sqrt(0.1842) = sd_obs * 0.4292
#:     SE(difference of two)   = sqrt(2) * that        = sd_obs * 0.6070
#:
#: ``sd_obs`` is the **observed between-patient sd of the panel mean**,
#: computed from the manifest at run time -- **not** taken from the
#: docstring figure 0.628, which is prose and is not a banked constant.
#:
#: **THE sqrt(2) IS STATED BECAUSE IT MATTERS.** A pair's ORDERING turns
#: on the DIFFERENCE of two means, so the proper scale for "ordered by
#: chance" is ``SE_diff``, which is **1.41x larger** than the
#: single-mean SE. The profile below is reported against ``SE_single``
#: as instructed, **and against ``SE_diff`` beside it**, so the stricter
#: reading is not hidden by the choice of scale.
#:
#: **THE PROFILE, NOT A THRESHOLD.** The fraction of pairs separated by
#: less than **1x, 2x and 3x** each scale -- six numbers -- because a
#: single threshold would be a choice and the profile is not.
#:
#: **THE PRIOR, REGISTERED BEFORE THE NUMBER** and tagged as an
#: expectation, not a result: under a normal approximation (which the
#: panel mean is NOT -- it is a mean of five integers, so it lands on
#: multiples of 0.2 and is bounded), the expected fractions are **0.24 /
#: 0.46 / 0.64** against ``SE_single`` and **0.33 / 0.61 / 0.80**
#: against ``SE_diff``. **If the measurement lands far from these, the
#: discreteness is doing something and that is itself worth reporting.**
COHORT_PAIR_SEPARATION_DESIGNED = {
    "designed": "2026-09-01 -- DESIGNED, NOT RUN",
    "why_it_exists": (
        "**nothing in the record measures how often two patients' panel "
        "means differ by less than the panel's own noise, and Phase 22's "
        "cohort arm trains on exactly those pairs.** A pair separated by "
        "less than the noise is ORDERED BY CHANCE, and a ranking loss "
        "cannot distinguish a coin flip from a signal"
    ),
    "the_quantity": (
        "over all C(237, 2) = 27,966 patient pairs: the distribution of "
        "|mean_i - mean_j|, and the fraction below a noise scale DERIVED "
        "from banked reliability rather than chosen"
    ),
    "the_derivation": (
        "MEAN_R_237 = 0.4696; RELIABILITY_237 = 0.8158 (Spearman-Brown "
        "at k = 5, which recomputes to 0.8157 from MEAN_R_237); error "
        "fraction = 1 - 0.8158 = 0.1842; **SE(one panel mean) = sd_obs * "
        "sqrt(0.1842) = sd_obs * 0.4292**; **SE(difference of two) = "
        "sqrt(2) * that = sd_obs * 0.6070**"
    ),
    "it_is_a_derivation_not_a_record": (
        "classical test theory applied to two banked reliability "
        "figures. No record states either SE; both follow from "
        "RELIABILITY_237 and the observed sd"
    ),
    "sd_obs_is_computed_not_quoted": (
        "the observed between-patient sd of the panel mean, computed "
        "from the manifest AT RUN TIME -- **not** the docstring figure "
        "0.628, which is prose and not a banked constant. (Note the "
        "adjacent trap: reliability.item_total's docstring also carries "
        "0.628, as an ITEM-TOTAL CORRELATION -- a different quantity, "
        "coincidentally the same digits)"
    ),
    "the_sqrt_2_matters": (
        "**a pair's ORDERING turns on the DIFFERENCE of two means**, so "
        "the proper scale for 'ordered by chance' is SE_diff, 1.41x "
        "larger than the single-mean SE. The profile is reported against "
        "BOTH, so the stricter reading is not hidden by the choice of "
        "scale"
    ),
    "the_profile_not_a_threshold": (
        "the fraction of pairs separated by less than 1x, 2x and 3x each "
        "scale -- six numbers -- **because a single threshold would be a "
        "choice and the profile is not**"
    ),
    "the_prior_registered_before_the_number": (
        "[REASONED, an expectation and not a result] under a NORMAL "
        "approximation -- which the panel mean is NOT, being a mean of "
        "five integers, so it lands on multiples of 0.2 and is bounded "
        "-- the expected fractions are 0.24 / 0.46 / 0.64 against "
        "SE_single and 0.33 / 0.61 / 0.80 against SE_diff. **If the "
        "measurement lands far from these, the discreteness is doing "
        "something and that is itself worth reporting**"
    ),

    # ---- the two readings, committed -------------------------------
    "reading_most_pairs_clear_the_noise": (
        "**the cohort ranking arm has real ordering to learn from, and "
        "Phase 22's cohort half is VIABLE.** The pair set is mostly "
        "signal, and a ranking loss over it is training on order that "
        "exists"
    ),
    "reading_most_pairs_do_not_clear_it": (
        "**the cohort arm would be training substantially on COIN "
        "FLIPS.** This is a SCOPING FACT that must be known before the "
        "arm is built, not discovered in its result -- a ranking arm "
        "that underperforms because most of its supervision was noise "
        "would be indistinguishable, after the fact, from one that "
        "underperformed because ordering does not help. **What it would "
        "then license is a restricted pair set, declared in advance, "
        "with the restriction rule recorded before any training -- not a "
        "post-hoc filter chosen to make the arm work**"
    ),
    "no_reading_is_invented_after": (
        "a pattern outside these two gets a dated OBSERVATION, on "
        "phase17.UNPREDICTED_PATTERN's precedent"
    ),
    "it_bears_on_the_cohort_arm_only": (
        "**the synthetic arm's ordering is EXACT BY CONSTRUCTION** -- "
        "magnitudes are a designed series, not an estimate, so no "
        "separation question arises there. This measurement constrains "
        "the cohort half and says nothing about the synthetic half"
    ),
    "not_run_here": (
        "the manifest is CLUSTER-SIDE. Designed and registered; not run, "
        "and no machinery built"
    ),
}


#: **[REGISTERED 2026-09-01 -- NAMED, NOT SCOPED] THE PAIRING-AXIS
#: ASSUMPTION.**
#:
#: **The synthetic magnitudes order THE SAME FACE AGAINST ITSELF. The
#: panel mean orders DIFFERENT PATIENTS. These are two different pairing
#: axes, and a rank-pretrain-then-fine-tune design BRIDGES them.**
#:
#: ``scut.synthesis``'s series is four magnitudes (0.0, 0.015, 0.025,
#: 0.035) applied to each face, read per-face from ``faces.json`` by
#: ``run._synth_index``. *"Larger means more deformed"* is a statement
#: about **one face at two magnitudes**; it says nothing about face X
#: against face Y. Phase 17's Siamese pairs are intra-face too -- left
#: versus right view of a single face.
#:
#: **THE BRIDGE IS AN UNDECLARED ASSUMPTION OF THE SAME FAMILY AS
#: MAGNITUDE-TO-GRADE.** That one says a displacement does not map to a
#: grade without evidence; this one says an ordering learned within faces
#: transfers to an ordering between patients. **Neither is measured, and
#: the second is not yet written down anywhere** -- which is why it is
#: named here.
#:
#: **NAMED, NOT SCOPED. Phase 22's restate must ADDRESS it rather than
#: inherit it silently** -- by declaring it, by measuring it, or by
#: designing around it. Which of those is the restate's business, not
#: this record's.
PAIRING_AXIS_ASSUMPTION = {
    "registered": "2026-09-01 -- NAMED, NOT SCOPED",
    "the_assumption": (
        "**the synthetic magnitudes order THE SAME FACE AGAINST ITSELF; "
        "the panel mean orders DIFFERENT PATIENTS. Two different pairing "
        "axes, and a rank-pretrain-then-fine-tune design BRIDGES them**"
    ),
    "what_the_artifact_actually_provides": (
        "four magnitudes (0.0, 0.015, 0.025, 0.035) applied to EACH "
        "face, read per-face from faces.json by run._synth_index. "
        "'Larger means more deformed' is a statement about ONE FACE AT "
        "TWO MAGNITUDES; it says nothing about face X against face Y"
    ),
    "phase_17s_pairs_are_intra_face_too": (
        "left versus right view of a SINGLE face, with the readout "
        "mapping that one face's left-right distance to a grade "
        "(round_1_plus_4_min_d_over_margin). Nothing built so far pairs "
        "two different patients"
    ),
    "it_is_the_same_family_as_magnitude_to_grade": (
        "magnitude-to-grade says a displacement does not map to a grade "
        "without evidence; **this one says an ordering learned WITHIN "
        "faces transfers to an ordering BETWEEN patients**. Neither is "
        "measured, and **the second is not yet written down anywhere** -- "
        "which is why it is named here"
    ),
    "named_not_scoped": (
        "**Phase 22's restate must ADDRESS it rather than inherit it "
        "silently** -- by declaring it, by measuring it, or by designing "
        "around it. Which of those is the restate's business, not this "
        "record's"
    ),
}


#: **[BUILT 2026-09-01 -- NOT RUN] THE COHORT-PAIR SEPARATION
#: MEASUREMENT.** The design is ``COHORT_PAIR_SEPARATION_DESIGNED``;
#: this records that it is built, configured, and what it guards.
COHORT_PAIR_SEPARATION_BUILT = {
    "built": "2026-09-01 -- built and configured, NOT RUN",
    "status": (
        "DESCRIPTIVE reckoning input for Phase 22's restate; no ledger "
        "row, no lock touched, and not Phase 22 machinery"
    ),
    "the_design": (
        "COHORT_PAIR_SEPARATION_DESIGNED -- unchanged; this record does "
        "not restate it, and the readings it committed are the ones the "
        "task applies"
    ),
    "the_task": "run.task_cohort_pair_separation, kind cohort_pair_separation",
    "sd_obs_is_computed_at_run_time": (
        "np.std(means) with ddof=0, matching phase3.sanity_report's own "
        "np.std(truth) -- the project's convention for this cohort's "
        "label spread. The multipliers 0.4292 and 0.6070 are DERIVED in "
        "the task from reliability.RELIABILITY_237, not pasted"
    ),
    "the_collision_guarded": (
        "**the docstring figure 0.628 is not used anywhere in the task, "
        "and a test asserts the string does not appear in it.** "
        "reliability.item_total's docstring carries 0.628 as an "
        "ITEM-TOTAL CORRELATION -- a different quantity with the same "
        "digits, and exactly the R2 shape this project keeps catching"
    ),
    "the_cell_is_declared_in_the_config": (
        "primary_scale and primary_multiple are config fields, so which "
        "of the six fractions the reading turns on is fixed BEFORE the "
        "numbers exist. Declared as se_diff at 1x -- the proper scale, "
        "because a pair's ordering turns on the DIFFERENCE of two means"
    ),
    "what_it_reports": (
        "all six fractions, the derivation printed beside them, the "
        "separation quantiles, the count of distinct separations (the "
        "panel mean lands on multiples of 0.2, so discreteness may move "
        "the result away from the normal-approximation prior), and the "
        "registered prior quoted beside what landed"
    ),
}


#: **[REGISTERED 2026-09-01 -- READINGS COMMITTED BEFORE IT RUNS] THE
#: CROSS-ARM SHRINKAGE MEASUREMENT.**
#:
#: **Why it exists.** ``SHRINKAGE_MECHANISM`` claims *"the arms agree
#: because they all SHRINK toward the label mean"* -- a claim about
#: every arm -- while its only numeric support was **one seed of one arm
#: on scrambled labels** (``SHRINKAGE_FIGURE_MISATTRIBUTED``). This
#: measures the claim instead of illustrating it.
#:
#: **The quantity**: ``sd(prediction) / sd(truth)`` per arm per seed,
#: recomputed from the cached ``PREDICTIONS_COLUMNS`` CSVs through the
#: Phase 21 reader -- **not a second implementation**. The truth column
#: is already in every CSV, so nothing new is read and no hash enters.
#:
#: **THE DENOMINATOR IS A CONSTANT, AND THAT IS VERIFIED.** On the 237
#: cohort every arm scores the same truth vector, so ``sd(truth)`` is
#: fixed and the ratio is driven **entirely by prediction spread**. The
#: task asserts it across every arm and seed, and against the manifest's
#: own ``mean`` column, rather than assuming it.
CROSS_ARM_SHRINKAGE_REGISTERED = {
    "registered": "2026-09-01, readings committed BEFORE it runs",
    "status": (
        "DESCRIPTIVE reckoning input for Phase 22's restate; no ledger "
        "row, no lock touched, and not Phase 22 machinery"
    ),
    "why_it_exists": (
        "**SHRINKAGE_MECHANISM claims 'the arms agree because they all "
        "SHRINK toward the label mean' -- a claim about EVERY ARM -- "
        "while its only numeric support was ONE SEED OF ONE ARM on "
        "scrambled labels** (SHRINKAGE_FIGURE_MISATTRIBUTED). This "
        "measures the claim instead of illustrating it"
    ),
    "the_quantity": (
        "sd(prediction) / sd(truth) per arm per seed, ddof=0 to match "
        "phase3.sanity_report, recomputed from the cached "
        "PREDICTIONS_COLUMNS CSVs through run._p21_load_arm_predictions "
        "-- **not a second implementation**. The truth column is already "
        "in every CSV, so nothing new is read and no hash enters"
    ),
    "the_denominator_is_a_constant": (
        "**on the 237 cohort every arm scores the same truth vector, so "
        "sd(truth) is fixed and the ratio is driven ENTIRELY by "
        "prediction spread.** VERIFIED, not assumed: the task asserts "
        "the truth vector is identical across every arm and seed AND "
        "equal to the manifest's own mean column, and refuses to "
        "continue otherwise"
    ),
    "the_arm_set_and_why_it_is_64_not_68": (
        "**the instruction said 'the locked 68'; this covers the 64-arm "
        "ruled set, and the deviation is stated rather than absorbed.** "
        "The four p12 arms are on 236 patients, so they carry their own "
        "truth vector and their own denominator; including them would "
        "require relaxing _p21_load_arm_predictions' cohort guard, which "
        "exists precisely to prevent the misalignment ARM_SET_RULED was "
        "written to avoid. Their shrinkage is computable separately "
        "against their own 236 truth, and is not computed here"
    ),
    "what_it_reports": (
        "the distribution across arms (min, median, max, spread, with "
        "the arms at each extreme named), the per-group breakdown so a "
        "regime difference is visible, the probe's own value, and any "
        "arm at or above the declared fails-to-shrink line"
    ),

    "readings": {
        "every_arm_shrinks_substantially": (
            "**SHRINKAGE_MECHANISM's general claim is MEASURED rather "
            "than illustrated**, and the 0.477 becomes one instance of a "
            "measured distribution rather than the evidence for it"
        ),
        "shrinkage_varies_widely": (
            "**the 'they all shrink identically' account NEEDS "
            "QUALIFYING**, and the error-consistency explanation is "
            "WEAKER THAN STATED: arms that shrink by different amounts "
            "do not err identically on the extremes, so some of the "
            "measured agreement is something else"
        ),
        "some_arm_does_not_shrink": (
            "**that arm is the interesting one and Phase 21's account "
            "has an EXCEPTION TO NAME.** An arm that does not shrink yet "
            "shares the error structure would refute the mechanism; one "
            "that does not shrink and does not share it would be the "
            "control the account never had"
        ),
    },
    "precedence": (
        "**declared, so two cells cannot both fire**: "
        "some_arm_does_not_shrink first (any arm at or above "
        "fails_to_shrink_at) -- a single non-shrinking arm is the more "
        "consequential fact -- then shrinkage_varies_widely (the SAMPLE "
        "SD of the per-arm means, ddof=1, at or above varies_widely_at "
        "[RULED 2026-09-01; a RANGE statistic was rejected on evidence, "
        "SHRINKAGE_DISPERSION_STATISTIC_RULED]), then "
        "every_arm_shrinks_substantially "
        "(max at or below shrinks_substantially_at). If none fires, NO "
        "CELL IS STRETCHED -- a dated OBSERVATION is recorded on "
        "phase17.UNPREDICTED_PATTERN's precedent"
    ),
    # [2026-09-01] The original wording is preserved below; the
    # thresholds are now RULED, and the dispersion statistic changed
    # from a range at 0.30 to a sample SD at 0.10.
    "the_thresholds_are_not_ruled": (
        "[SUPERSEDED 2026-09-01 -- see the_thresholds_are_ruled] "
        "**fails_to_shrink_at 0.90, shrinks_substantially_at 0.75, "
        "varies_widely_at 0.30 are the choice made when it was written, not the maintainer's "
        "ruling.** They are declared in the config so the cell cannot be "
        "chosen after the numbers, and they are flagged here so they can "
        "be overruled before the run rather than argued about after it"
    ),
    "the_thresholds_are_ruled": (
        "**RULED 2026-09-01 .** fails_to_shrink_at 0.90 and "
        "shrinks_substantially_at 0.75 STAND AS BUILT, both on the MAX "
        "of the per-arm means, with the precedence order fails -> "
        "varies -> shrinks. varies_widely_at is now the SAMPLE SD "
        "(ddof=1) of the per-arm means AT 0.10, replacing a range "
        "statistic at 0.30 that was built, simulated and REJECTED ON "
        "EVIDENCE -- SHRINKAGE_DISPERSION_STATISTIC_RULED. **The "
        "flagging worked: they were declared as the choice made when it was written, put "
        "to the maintainer before the run, and one of the three was changed "
        "on a measurement rather than argued about after a result**"
    ),
    "what_it_does_not_touch": (
        "SCALE_INVARIANCE_PROHIBITION is unaffected whatever this "
        "measures. Establishing that every arm shrinks says nothing "
        "about the PCC ceiling, because Pearson is scale-invariant -- "
        "the prohibition applies to this measurement's result as to "
        "everything else in the phase"
    ),
}


#: **[RULED 2026-09-01] THE DISPERSION STATISTIC: SAMPLE SD AT
#: 0.10. A RANGE AT 0.30 WAS BUILT, SIMULATED AND REJECTED ON
#: EVIDENCE.**
#:
#: The rejected version is kept **with the measurement that killed it**,
#: not overwritten. **A rejected calibration with its numbers is more
#: informative than a threshold that was merely chosen well** -- it shows
#: what the alternative would have done, which no surviving threshold can.
#:
#: **WHAT THE RANGE DID, MEASURED.** Simulated at fixed sigma = 0.08,
#: 40,000 draws, against the exact chi-square result:
#:
#:     n     mean s   P(s>=0.10)   exact    mean range   P(R>=0.30)
#:     8     0.0773     0.1439     0.1414     0.2282       0.1423
#:     16    0.0787     0.0761     0.0753     0.2826       0.3632
#:     32    0.0793     0.0232     0.0239     0.3309       0.6961
#:     64    0.0797     0.0033     0.0029     0.3750       0.9465
#:
#: **Four grounds, each measured:**
#:
#: 1. **The range's ESTIMAND grows with arm count** -- mean range 0.2282
#:    at n = 8 to 0.3750 at n = 64, at *fixed* dispersion. So the cell
#:    tests arm count as much as heterogeneity.
#: 2. **At n = 64 the range fires 94.65% of the time on genuinely
#:    homogeneous arms**, where the SD fires **0.33%** -- cross-checked
#:    against the exact chi-square, which agrees to simulation error at
#:    every n.
#: 3. **The range is decided by exactly TWO of 64 arms.** The SD uses all
#:    of them, which is what *"do the arms shrink identically"* actually
#:    asks.
#: 4. **The SD's false-fire rate FALLS with n** (0.144 -> 0.003) rather
#:    than rising -- the safe direction. A larger arm set makes the cell
#:    harder to trip by accident, not nearly certain to.
#:
#: **IT IS NOT A RE-EXPRESSION, AND THAT IS ACCEPTED DELIBERATELY.** At
#: n = 64 the range/SD ratio is 4.70 (the d2 constant), so **SD 0.10
#: corresponds to range 0.47** -- the ruled cell is **substantially
#: stricter** than the one it replaces. The widely-fires region shrinks;
#: the substantially cell and the dead zone correspondingly enlarge.
#:
#: **THE GROUND FOR ACCEPTING THAT**: the widely cell is the one
#: that **weakens Phase 21's account**, so a stricter bar makes the
#: project's existing account **harder to overturn, not easier**.
#: Strictness in the direction that guards against a comfortable reading
#: rather than toward one.
SHRINKAGE_DISPERSION_STATISTIC_RULED = {
    "ruled": "2026-09-01 -- sample SD at 0.10; range at 0.30 rejected",
    "the_statistic": (
        "the SAMPLE SD (ddof=1) of the 64 per-arm mean shrinkage ratios, "
        "tested at >= 0.10"
    ),
    "the_rejected_version": (
        "the RANGE (max - min) of those same per-arm means, tested at "
        ">= 0.30. **Built, configured and simulated before it was "
        "rejected** -- it is kept here with its numbers, not overwritten"
    ),
    "why_the_rejected_version_is_kept": (
        "**a rejected calibration with its numbers is more informative "
        "than a threshold that was merely chosen well.** It shows what "
        "the alternative would have done, which no surviving threshold "
        "can"
    ),

    "the_simulation": (
        "fixed sigma = 0.08, 40,000 draws, cross-checked against the "
        "exact chi-square ((n-1)s^2/sigma^2 ~ chi2(n-1)): "
        "n=8 mean s 0.0773, P(s>=0.10) 0.1439 (exact 0.1414), mean range "
        "0.2282, P(R>=0.30) 0.1423; "
        "n=16 0.0787 / 0.0761 (0.0753) / 0.2826 / 0.3632; "
        "n=32 0.0793 / 0.0232 (0.0239) / 0.3309 / 0.6961; "
        "n=64 0.0797 / 0.0033 (0.0029) / 0.3750 / 0.9465"
    ),
    "ground_1_the_estimand_grows_with_n": (
        "**mean range 0.2282 at n = 8 to 0.3750 at n = 64, at FIXED "
        "dispersion.** The cell tests ARM COUNT as much as heterogeneity"
    ),
    "ground_2_it_fires_on_homogeneous_arms": (
        "**at n = 64, range >= 0.30 fires 94.65% of the time on "
        "genuinely homogeneous arms; SD >= 0.10 fires 0.33%** -- "
        "cross-checked against the exact chi-square, which agrees to "
        "simulation error at every n"
    ),
    "ground_3_two_arms_versus_all_of_them": (
        "**the range is decided by exactly TWO of 64 arms.** The SD uses "
        "all of them, which is what 'do the arms shrink identically' "
        "actually asks"
    ),
    "ground_4_the_error_rate_falls_with_n": (
        "**the SD's false-fire rate FALLS with n (0.144 -> 0.003) rather "
        "than rising** -- the safe direction. A larger arm set makes the "
        "cell harder to trip by accident, not nearly certain to"
    ),

    "it_is_not_a_re_expression": (
        "**at n = 64 the range/SD ratio is 4.70 (the d2 constant), so SD "
        "0.10 corresponds to RANGE 0.47.** The ruled cell is "
        "SUBSTANTIALLY STRICTER than the one it replaces: the "
        "widely-fires region shrinks, and the substantially cell and the "
        "dead zone correspondingly enlarge"
    ),
    "the_ground_for_accepting_the_strictness": (
        "**the widely cell is the one that WEAKENS Phase 21's account, "
        "so a stricter bar makes the project's existing account HARDER "
        "TO OVERTURN, NOT EASIER.** Strictness in the direction that "
        "guards against a comfortable reading rather than toward one"
    ),

    "the_discriminators_measured_profile": (
        "**on the record rather than in a chat.** At n = 64 the cell is "
        "a COIN FLIP at underlying SD 0.1005, fires 90% of the time at "
        "0.1133, 99% at 0.1257, and essentially never below 0.09 "
        "(P = 0.0996 at 0.09, P = 0.0029 at 0.08, P = 0.0000 at 0.06). "
        "**The cell says 'heterogeneity above 0.10 ratio units' and "
        "means it**"
    ),
    "the_other_two_thresholds_stand_as_built": (
        "fails_to_shrink_at 0.90 and shrinks_substantially_at 0.75, both "
        "on the MAX of the per-arm means, and the precedence order fails "
        "-> varies -> shrinks, **since a single non-shrinking arm is the "
        "more consequential fact**"
    ),
    "the_range_is_still_reported": (
        "computed and written to metrics.json beside the SD as context. "
        "**It decides nothing**"
    ),
    "the_dead_zone_stays_named": (
        "**no cell fires when 0.75 < max < 0.90 with a tight spread** -- "
        "arms shrinking moderately and consistently, which is a "
        "PLAUSIBLE outcome, not an exotic one. The ruled SD widens this "
        "zone relative to the range version. The dated-observation "
        "precedent (phase17.UNPREDICTED_PATTERN) handles it, and **NO "
        "CELL MAY BE STRETCHED TO COVER IT**"
    ),
}


#: **[OBSERVED 2026-09-01] COHORT-PAIR SEPARATION -- the cohort arm is
#: VIABLE.** Run ``p21_cohort_pair_separation__48a0c697``.
#:
#: ``sd_obs`` **0.657279**, computed from the manifest as designed;
#: **SE_single 0.282095**, **SE_diff 0.398942**; **27,966** pairs taking
#: **17 distinct separation values**.
#:
#:     below      1x        2x        3x
#:     SE_single  0.2485    0.4018    0.6558
#:     SE_diff    0.2485    0.5373    0.7557
#:
#: **The declared primary scale is SE_diff at 1x: 0.2485.**
#: ``reading_most_pairs_clear_the_noise`` **FIRES** -- roughly
#: **three-quarters of pairs carry ordering the panel can distinguish**,
#: and Phase 22's cohort half has real order to learn from.
#:
#: **THE PRIOR'S PERFORMANCE, RECORDED HONESTLY.** The registered
#: expectation was **0.33** on SE_diff; the measurement came in at
#: **0.2485** -- **lower, meaning MORE usable pairs than predicted**. The
#: registration named discreteness as the candidate reason in advance,
#: and the outturn is consistent with it: the panel mean lands on
#: multiples of 0.2, and the separations take only 17 distinct values
#: against a continuum. **The prior was wrong in the favourable
#: direction, which is exactly the direction that would have been easy
#: to leave unremarked.**
#:
#: **Bears on the COHORT arm only.** The synthetic ordering is exact by
#: construction.
COHORT_PAIR_SEPARATION_OBSERVED = {
    "observed": "2026-09-01, run p21_cohort_pair_separation__48a0c697",
    "sd_obs": 0.657279,
    "se_single": 0.282095,
    "se_diff": 0.398942,
    "n_pairs": 27966,
    "n_distinct_separations": 17,
    "fraction_below": {
        "se_single_x1": 0.2485, "se_single_x2": 0.4018,
        "se_single_x3": 0.6558,
        "se_diff_x1": 0.2485, "se_diff_x2": 0.5373, "se_diff_x3": 0.7557,
    },
    "primary_scale_declared": "se_diff_x1",
    "cell_fired": (
        "COHORT_PAIR_SEPARATION_DESIGNED"
        "['reading_most_pairs_clear_the_noise']"
    ),
    "the_verdict": (
        "**0.2485 of pairs fall below the declared scale, so roughly "
        "THREE-QUARTERS carry ordering the panel can distinguish. Phase "
        "22's cohort half has real order to learn from, and the arm is "
        "VIABLE**"
    ),
    "the_priors_performance": (
        "**the registered expectation was 0.33 on SE_diff; the "
        "measurement came in at 0.2485 -- LOWER, meaning MORE USABLE "
        "PAIRS THAN PREDICTED.** The registration named discreteness as "
        "the candidate reason IN ADVANCE and the outturn is consistent "
        "with it: the panel mean lands on multiples of 0.2, and the "
        "separations take only 17 distinct values against a continuum. "
        "**The prior was wrong in the FAVOURABLE direction, which is "
        "exactly the direction that would have been easy to leave "
        "unremarked**"
    ),
    "it_bears_on_the_cohort_arm_only": (
        "the synthetic ordering is EXACT BY CONSTRUCTION; this "
        "constrains the cohort half and says nothing about the other"
    ),
}


#: **[OBSERVED 2026-09-01] CROSS-ARM SHRINKAGE -- the general claim is
#: MEASURED, and it has an EXCEPTION.** Run
#: ``p21_cross_arm_shrinkage__48a0c697``.
#:
#: **The denominator was verified constant** at ``sd(truth) = 0.657279``
#: across all **64 arms x 5 seeds** and equal to the manifest, so the
#: ratio is **prediction spread alone**.
#:
#:     min     0.0656   p16 identity baseline
#:     median  0.3289
#:     max     2.1730   p17_arm_a
#:     probe   0.4947
#:     SD      0.2704   (the ruled dispersion statistic)
#:
#: **The fired cell is ``some_arm_does_not_shrink``** -- one arm at or
#: above the declared 0.90 line: **p17_arm_a**. Precedence held: the
#: observed SD 0.2704 would also have cleared the 0.10 widely line, and
#: the non-shrinking arm takes priority as ruled, because a single
#: non-shrinking arm is the more consequential fact.
#:
#: **WHAT THIS REPLACES.** ``SHRINKAGE_MECHANISM``'s general claim rested
#: on **one seed of one arm supplied in conversation**
#: (``SHRINKAGE_FIGURE_MISATTRIBUTED``). It now rests on **64 arms
#: measured from cached CSVs**, with **63 of them shrinking** at a median
#: of **0.33**, and **one exception named**. The illustration is
#: replaced by a measurement, which is what the record said it needed.
CROSS_ARM_SHRINKAGE_OBSERVED = {
    "observed": "2026-09-01, run p21_cross_arm_shrinkage__48a0c697",
    "denominator_verified": (
        "sd(truth) = 0.657279, constant across all 64 arms x 5 seeds and "
        "equal to the manifest's own mean column, so **the ratio is "
        "PREDICTION SPREAD ALONE**"
    ),
    "across_arms": {
        "min": 0.0656, "min_arm": "p16 identity baseline",
        "median": 0.3289,
        "max": 2.1730, "max_arm": "p17_arm_a",
        "sd": 0.2704,
    },
    "probe": 0.4947,
    "cell_fired": (
        "CROSS_ARM_SHRINKAGE_REGISTERED['readings']"
        "['some_arm_does_not_shrink']"
    ),
    "precedence_held": (
        "the observed SD 0.2704 would ALSO have cleared the 0.10 widely "
        "line; the non-shrinking arm takes priority as ruled, **because "
        "a single non-shrinking arm is the more consequential fact** "
        "(SHRINKAGE_DISPERSION_STATISTIC_RULED)"
    ),
    "what_it_replaces": (
        "**SHRINKAGE_MECHANISM's general claim rested on ONE SEED OF ONE "
        "ARM supplied in conversation** "
        "(SHRINKAGE_FIGURE_MISATTRIBUTED). **It now rests on 64 ARMS "
        "MEASURED FROM CACHED CSVs**, with 63 of them shrinking at a "
        "median of 0.33 and one exception named. The illustration is "
        "replaced by a measurement, which is what the record said it "
        "needed"
    ),
    "the_exception": "p17_arm_a -- see ARM_A_THE_SHRINKAGE_CONTROL",
}


#: **[2026-09-01] ARM A: THE CONTROL THE ACCOUNT NEVER HAD. The
#: strongest causal evidence in the project.**
#:
#: **Arm A does not shrink -- it EXPANDS, at 2.1730 -- and the reason is
#: STRUCTURAL.** It trained on magnitude-mapped synthetic labels with
#: **no exposure to the cohort's distribution**, so **it has no label
#: mean to shrink toward.** Not an anomaly: a model that never saw the
#: target distribution cannot regress to its centre.
#:
#: **ONE PROPERTY EXPLAINS THREE BANKED FIGURES PHASE 18 NEVER LINKED:**
#:
#:     prediction mean  1.878   against truth 2.7544
#:     prediction sd    1.4348  against truth 0.6587
#:     IEM              1.61    against every other arm's 0.51-0.62
#:
#: Mis-centred, over-spread, and scaled wrongly on an error metric --
#: all three are the same fact, and the record held them separately for
#: a phase and a half.
#:
#: **THE CONTROL.** From ``p21_error_consistency__feb53133``'s own
#: matrix, arm A's mean kappa against the other 63 is **+0.0956**
#: (residual sign) and **-0.0743** (worst quartile), against all-pairs
#: means of **+0.7012** and **+0.5729**. **That is chance level** --
#: comparable to Geirhos's CNN-to-human out-of-distribution figure of
#: **~0.07**, which that work treats as indicating **completely different
#: processing strategies**.
#:
#: **WHAT IT ESTABLISHES -- the registered reading's own words fire.**
#: ``some_arm_does_not_shrink`` said: *"An arm that does not shrink yet
#: shares the error structure would refute the mechanism; one that does
#: not shrink AND does not share it would be the control the account
#: never had."* **The second branch is what happened.**
#:
#: **The shrinkage mechanism is no longer an inference from two
#: correlations. It has a natural-experiment control that the project ran
#: three phases ago without knowing what it was for.**
#:
#: **THE ARGUMENT, CLOSED AT BOTH ENDS:**
#:
#: * 63 arms shrink, and agree at **0.7012 / 0.5729**;
#: * error DIRECTION tracks signed distance from the centre at
#:   **tau -0.8470**;
#: * WORST-QUARTILE errors track absolute distance at **tau +0.7521**;
#: * the ONE non-shrinking arm agrees at **chance**.
ARM_A_THE_SHRINKAGE_CONTROL = {
    "recorded": "2026-09-01",
    "arm_a_expands": (
        "**2.1730 -- it does not shrink, it EXPANDS -- and the reason is "
        "STRUCTURAL: it trained on magnitude-mapped synthetic labels "
        "with NO EXPOSURE to the cohort's distribution, so it has NO "
        "LABEL MEAN TO SHRINK TOWARD.** A model that never saw the "
        "target distribution cannot regress to its centre"
    ),
    "one_property_explains_three_banked_figures": (
        "prediction mean 1.878 against truth 2.7544; prediction sd "
        "1.4348 against 0.6587; IEM 1.61 against every other arm's "
        "0.51-0.62. **Mis-centred, over-spread, and scaled wrongly on an "
        "error metric are the SAME FACT**, and the record held them "
        "separately for a phase and a half"
    ),
    "the_control": (
        "from p21_error_consistency__feb53133's own matrix, arm A's mean "
        "kappa against the other 63 is **+0.0956 (residual sign)** and "
        "**-0.0743 (worst quartile)**, against all-pairs means of "
        "+0.7012 and +0.5729. **That is CHANCE LEVEL** -- comparable to "
        "Geirhos's CNN-to-human out-of-distribution figure of ~0.07, "
        "which that work treats as indicating COMPLETELY DIFFERENT "
        "PROCESSING "
        "STRATEGIES"
    ),
    "the_registered_readings_own_words_fire": (
        "some_arm_does_not_shrink said: 'An arm that does not shrink yet "
        "shares the error structure would refute the mechanism; one that "
        "does not shrink AND does not share it would be **the control "
        "the account never had**.' **The second branch is what "
        "happened**"
    ),
    "what_it_establishes": (
        "**the shrinkage mechanism is no longer an inference from two "
        "correlations. It has a NATURAL-EXPERIMENT CONTROL that the "
        "project ran three phases ago without knowing what it was for**"
    ),
    "the_argument_closed_at_both_ends": (
        "63 arms shrink and agree at 0.7012 / 0.5729; error DIRECTION "
        "tracks signed distance at tau -0.8470; WORST-QUARTILE errors "
        "track absolute distance at tau +0.7521; **the one "
        "non-shrinking arm agrees at CHANCE**"
    ),
    #: **[STRUCTURED MIRROR ADDED 2026-09-01]** The same two figures the
    #: prose above carries, in a form code can read. Phase 22's
    #: diagnostic takes its LOW thresholds from here, and the standing
    #: rule forbids regexing values out of prose -- so the figure is
    #: mirrored once, with a test asserting the mirror and the sentence
    #: agree. **No new measurement: these are the banked ones.**
    "kappa_measured": {"residual_sign": 0.0956, "worst_quartile": -0.0743},
    "the_kappa_figures_are_post_hoc": (
        "**[POST-HOC]** -- computed  from the banked matrix "
        "AFTER the phase closed, firing NO committed cell, on the same "
        "footing as POST_HOC_ANALYSES. They explain the arms; they test "
        "nothing committed"
    ),
}


#: **[2026-09-01] THE CONSEQUENCE FOR PHASE 22, recorded BEFORE its
#: restate so it cannot be discovered afterwards.**
#:
#: **Arm A is an existence proof that a model can avoid shrinkage on this
#: task -- and it scored 0.2334**, statistically indistinguishable from
#: the probe's 0.2520 (``phase17.ARM_MEANS``: A 0.2334 sd 0.0044; the
#: p17-a-vs-probe contrast is UNRESOLVED).
#:
#: **SO NOT SHRINKING BUYS NOTHING ON PCC BY ITSELF** -- consistent with
#: ``SCALE_INVARIANCE_PROHIBITION``, and now demonstrated by a measured
#: case rather than argued from scale invariance alone.
#:
#: **PHASE 22'S HONEST HYPOTHESIS NARROWS ACCORDINGLY**: a ranking loss
#: would have to **find different FEATURES**, not merely **avoid the
#: MEAN** -- and **the record now holds a measured case showing those two
#: things are separable.**
#:
#: **THIS MUST APPEAR IN PHASE 22'S RECKONING. It may not be discovered
#: afterwards.**
PHASE_22_CONSEQUENCE_RECORDED = {
    "recorded": "2026-09-01, BEFORE Phase 22's restate",
    "arm_a_is_an_existence_proof": (
        "**a model CAN avoid shrinkage on this task -- arm A expands at "
        "2.1730 -- and it scored 0.2334**, statistically "
        "indistinguishable from the probe's 0.2520 (phase17.ARM_MEANS: A "
        "0.2334 sd 0.0044; p17-a-vs-probe UNRESOLVED)"
    ),
    "not_shrinking_buys_nothing_on_pcc": (
        "**consistent with SCALE_INVARIANCE_PROHIBITION, and now "
        "DEMONSTRATED BY A MEASURED CASE rather than argued from scale "
        "invariance alone**"
    ),
    "the_hypothesis_narrows": (
        "**a ranking loss would have to find DIFFERENT FEATURES, not "
        "merely AVOID THE MEAN -- and the record now holds a measured "
        "case showing those two things are SEPARABLE**"
    ),
    "it_must_appear_in_the_reckoning": (
        "**this goes in Phase 22's reckoning BEFORE any reading is "
        "written. It may not be discovered afterwards** -- a phase with "
        "a positive hypothesis is exactly where a constraint found late "
        "gets softened"
    ),
    "and_the_test_is_harder_than_the_hypothesis": (
        "a PCC gain from a ranking arm would additionally have to clear "
        "ORDINARY ARM-TO-ARM VARIATION on a cohort that "
        "ladder.COHORT_CANNOT_RESOLVE says cannot resolve differences of "
        "0.04 to 0.10. The hypothesis is weaker AND harder to test than "
        "the shrinkage-removal one it replaces"
    ),
}
