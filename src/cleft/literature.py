"""The literature bank: external sources, and what they settle or unsettle.

**[OPENED 2026-09-01]** A record of papers read against this project's
own claims, with each source filename named and each contradiction
flagged where it bears on something the project or the record has asserted.

**THE TAGS ARE EXACTLY THREE**: ``[LITERATURE]``, ``[MEASURED]``,
``[REASONED]``. Provenance -- who read it, by what route, on what date --
is **prose beside the tag, never inside the bracket**. Compound and
qualified variants (``[LITERATURE-primary]``, ``[REPORTED, not verified
here]``, ``[MANUSCRIPT, via ...]``) were invented and are being
normalised; see ``TAG_DISCIPLINE``.

**PROVENANCE OF EVERYTHING IN THIS MODULE, stated once and bindingly.**
Every quote and figure below reached this record **through the maintainer**.
**None of the cited PDFs is reachable from this machine** -- unlike
``phase12.RATING_PROCEDURE_DOCUMENTED``, where two of four documents were
present and their quotes were verified at source. Here nothing could be.
What this session **did** verify is the **arithmetic inside the claims**,
which is recorded per entry as ``arithmetic_checked_here``.
"""

from __future__ import annotations


#: **[RULED 2026-09-01] TAG DISCIPLINE.**
#:
#: **The tags are exactly three**: ``[LITERATURE]``, ``[MEASURED]``,
#: ``[REASONED]``. Anything else in the bracket is invented.
#:
#: **A date is not a provenance clause.** ``[MEASURED 2026-08-04]`` is
#: the house convention across ~180 occurrences and stays: the tag is
#: still one of the three and the date is a timestamp. A LOCATOR after
#: the tag (``[LITERATURE, brief §3.1]``, ``[MEASURED, PLAN §4.4]``) is
#: the same shape and also stays.
#:
#: **What was invented, and why it matters**: ``REPORTED`` and
#: ``MANUSCRIPT`` are not tags at all -- they are *provenance
#: qualifications wearing a tag's clothes*, and they read as a fourth
#: and fifth evidential category. ``[LITERATURE-primary]`` implies a
#: hierarchy the three-tag scheme does not have.
#: ``[REASONED -> MEASURED at run 5]`` puts a *transition* inside a
#: bracket that names a *state*.
#:
#: **Why the distinction is load-bearing rather than cosmetic**:
#: ``ladder.BASAL_RATIONALE_UNSUPPORTED``'s own lesson is that
#: *verification effort flows to the items tagged REASONED, so a
#: mis-tagged rationale is precisely the claim nobody audits.* A reader
#: who sweeps for ``[REASONED]`` never sees ``[REPORTED, not measured]``,
#: which is a *weaker* claim than either. **An invented tag is
#: unauditable by the very sweep the tags exist to enable.**
TAG_DISCIPLINE = {
    "ruled": "2026-09-01",
    "the_three": ("[LITERATURE]", "[MEASURED]", "[REASONED]"),
    "a_date_is_not_a_provenance_clause": (
        "[MEASURED 2026-08-04] is the house convention across ~180 "
        "occurrences and STAYS: the tag is still one of the three and "
        "the date is a timestamp. A LOCATOR after the tag "
        "([LITERATURE, brief 3.1], [MEASURED, PLAN 4.4]) is the same "
        "shape and also stays"
    ),
    "what_was_invented": (
        "REPORTED and MANUSCRIPT are not tags at all -- they are "
        "PROVENANCE QUALIFICATIONS WEARING A TAG'S CLOTHES, reading as a "
        "fourth and fifth evidential category. [LITERATURE-primary] "
        "implies a hierarchy the scheme does not have. "
        "[REASONED -> MEASURED at run 5] puts a TRANSITION inside a "
        "bracket that names a STATE"
    ),
    "why_it_is_not_cosmetic": (
        "**ladder.BASAL_RATIONALE_UNSUPPORTED's own lesson**: "
        "verification effort flows to the items tagged REASONED, so a "
        "mis-tagged rationale is precisely the claim nobody audits. A "
        "reader sweeping for [REASONED] never sees [REPORTED, not "
        "measured] -- which is a WEAKER claim than either. **An invented "
        "tag is unauditable by the very sweep the tags exist to enable**"
    ),
    "the_rule_going_forward": (
        "the tag names the EVIDENTIAL STATE and nothing else; who read "
        "it, by what route, and on what date is PROSE BESIDE IT"
    ),
}


#: **[LITERATURE] Geirhos, Meding & Wichmann, NeurIPS 2020 --
#: ``2006.16736.pdf``.** Read ; not reachable from this
#: machine. Bears on ``phase21.ARM_B_REGISTERED``.
#:
#: **THE BINARISATION IS SANCTIONED, NOT INVENTED.** The metric is
#: defined strictly for binary correct/incorrect -- *"For this purpose,
#: we only analyse whether the decisions were correct/incorrect
#: (irrespective of the number of choices)"* -- and continuous outputs
#: *"must be thresholded and binarized into 'correct' and 'incorrect'
#: classifications relative to a reference standard before computing this
#: metric."* Arm B's residual-sign and worst-quartile binarisations are
#: **exactly what the paper requires of a continuous output**. The
#: registration reached that design independently; the paper licenses it.
#:
#: **THE MEASURED BANDS**, banked as context:
#:
#:     human-to-human      0.32-0.48   (cue-conflict 0.33+-0.02,
#:                                      edges 0.32+-0.04,
#:                                      silhouettes 0.48+-0.03,
#:                                      ImageNet 0.37)
#:     CNN-to-CNN          0.62-0.79   across depths, families and
#:                                     parameter counts; highest 0.793
#:     CNN-to-human OOD    ~0.066-0.068
#:
#: **WHERE ARM B'S LOCKED 0.50 SITS, arithmetic checked here**: **0.02
#: above the human ceiling (0.48) and 0.12 below the CNN floor (0.62).**
#: So the threshold is **demanding by human standards and lenient by CNN
#: standards** -- and Arm B compares CNN-like arms to each other, which
#: is the band 0.50 sits below.
#:
#: **THIS IS CONTEXT TO REPORT BESIDE THE THRESHOLD, NOT AN AMENDMENT TO
#: IT.** ``phase21.CONSISTENCY_THRESHOLDS_RULED`` is locked and stays
#: locked. The published band is a different quantity measured on
#: different models and different tasks; using it to move a threshold
#: after the fact would be choosing a threshold from outside the
#: registration. **What it licenses is a sentence beside the result, not
#: a change to the cell that fires.**
#:
#: **BOTH INSTABILITY WARNINGS, banked:**
#:
#: * kappa's denominator ``1 - c_exp`` shrinks at high ``c_exp``. **Arm
#:   B's arithmetic, checked here: c_exp = 0.500 at residual sign
#:   (p=0.50) and 0.625 at worst quartile (p=0.25), giving denominators
#:   0.500 and 0.375. NO RISK** -- both are far from the degenerate end.
#: * percentile accuracy is limited at small trial counts; the paper
#:   flags it at **160**. **Arm B has 237.**
GEIRHOS_ERROR_CONSISTENCY = {
    "tag": "[LITERATURE]",
    "source": "2006.16736.pdf -- Geirhos, Meding & Wichmann, NeurIPS 2020",
    "provenance": (
        "quoted from the published paper; the PDF is not in this repository, so no quote "
        "below was verified at source by this session"
    ),
    "bears_on": "phase21.ARM_B_REGISTERED",

    "the_metric_is_defined_for_binary_outcomes": (
        "For this purpose, we only analyse whether the decisions were "
        "correct/incorrect (irrespective of the number of choices)"
    ),
    "continuous_outputs_must_be_binarised": (
        "must be thresholded and binarized into 'correct' and "
        "'incorrect' classifications relative to a reference standard "
        "before computing this metric"
    ),
    "arm_bs_binarisations_are_sanctioned": (
        "**residual sign and worst-quartile membership are EXACTLY what "
        "the paper requires of a continuous output -- the sanctioned "
        "adaptation, not an invented one.** The registration reached the "
        "design independently; the paper licenses it"
    ),

    "human_to_human_band": (
        "0.32-0.48: cue-conflict 0.33+-0.02, edges 0.32+-0.04, "
        "silhouettes 0.48+-0.03, ImageNet 0.37"
    ),
    "cnn_to_cnn_band": (
        "0.62-0.79 across depths, families and parameter counts; "
        "highest 0.793"
    ),
    "cnn_to_human_ood": "~0.066-0.068",

    "where_arm_bs_locked_threshold_sits": (
        "**arithmetic checked here**: 0.50 is 0.02 ABOVE the human "
        "ceiling (0.48) and 0.12 BELOW the CNN floor (0.62). The "
        "threshold is DEMANDING BY HUMAN STANDARDS and LENIENT BY CNN "
        "STANDARDS -- and Arm B compares CNN-like arms to each other, "
        "which is the band 0.50 sits below"
    ),
    "context_not_amendment": (
        "**phase21.CONSISTENCY_THRESHOLDS_RULED is locked and stays "
        "locked.** The published band is a DIFFERENT QUANTITY measured "
        "on different models and different tasks; using it to move a "
        "threshold after the fact would be choosing a threshold from "
        "outside the registration. What it licenses is A SENTENCE BESIDE "
        "THE RESULT, not a change to the cell that fires"
    ),

    "instability_denominator": (
        "kappa's denominator (1 - c_exp) shrinks at high c_exp. **Arm "
        "B's arithmetic, checked here: c_exp = 0.500 at residual sign "
        "(p = 0.50) and 0.625 at worst quartile (p = 0.25), giving "
        "denominators 0.500 and 0.375. NO RISK** -- both far from the "
        "degenerate end"
    ),
    "instability_small_trial_counts": (
        "percentile accuracy is limited at small trial counts; the paper "
        "flags it at 160. **Arm B has 237**"
    ),
    "arithmetic_checked_here": (
        "c_exp at p=0.50 -> 0.5000 and at p=0.25 -> 0.6250, both "
        "recomputed; the human band's endpoints reproduce 0.32-0.48 from "
        "the four quoted points; the 0.50 threshold's distances to 0.48 "
        "and 0.62 are 0.02 and 0.12"
    ),
}


#: **[LITERATURE] Bouthillier et al., MLSys 2021 -- ``2103.03098.pdf``.**
#: Read ; not reachable here. Bears on the five-seed protocol
#: and on ``phase4``'s partition sensitivity.
#:
#: **A SAMPLE IS A FULL INDEPENDENT RUN WITH THE SPLIT RANDOMISED** --
#: *"Lets note this number of runs as the sample size N, not to be
#: confused with dataset size n."* Five seeds on a **fixed split** is
#: their **FixHOptEst** biased estimator, which gives *"severe
#: underestimations of standard error."*
#:
#: **Data sampling dominates**, and weight initialisation is consistently
#: **under half** of it. Their P(A>B) is the mean paired indicator over
#: **independent splits**, with percentile-bootstrap CIs, gamma = 0.75,
#: N = 29 paired runs.
#:
#: **[WITHDRAWN 2026-09-02] The independent-arrival claim.** It read:
#: *"THIS PROJECT ARRIVED AT THE SAME CONCLUSION INDEPENDENTLY. Phase
#: 4's partition-sensitivity arm measured a range of 0.068 across
#: partitions -- larger than the seed band."* **Two defects: the figure
#: has no banked source, and the two quantities are not comparable.**
#: See ``PARTITION_SENSITIVITY_CLAIM_WITHDRAWN``. What stands: the
#: five-seed protocol IS Bouthillier's biased estimator, as banked --
#: only the claim to have measured the same thing first is withdrawn.
#:
#: **A LIMITATION WITH A CITATION, NOT A DEFECT.** The paired BCa remains
#: valid **for what it measures**: variation under a fixed partition,
#: which is the quantity every banked contrast is defined on. What it
#: does not measure is variation under re-splitting, and nothing in the
#: record ever claimed it did. **Nothing banked becomes wrong; the
#: uncertainty statement becomes narrower than the general one.**
BOUTHILLIER_SAMPLE_SIZE = {
    "tag": "[LITERATURE]",
    "source": "2103.03098.pdf -- Bouthillier et al., MLSys 2021",
    "provenance": "quoted from the published paper; the PDF is not in this repository",
    "bears_on": "the five-seed protocol; phase4's partition sensitivity",

    "what_a_sample_is": (
        "a full independent run with the dataset split randomised: "
        "'Lets note this number of runs as the sample size N, not to be "
        "confused with dataset size n'"
    ),
    "five_seeds_on_a_fixed_split_is_their_biased_estimator": (
        "FixHOptEst, which gives 'severe underestimations of standard "
        "error'"
    ),
    "data_sampling_dominates": (
        "weight initialisation is consistently UNDER HALF of the "
        "variation contributed by data sampling"
    ),
    "their_p_a_beats_b": (
        "the mean paired indicator over INDEPENDENT SPLITS, with "
        "percentile-bootstrap CIs, gamma = 0.75, N = 29 paired runs"
    ),

    "this_project_arrived_there_independently": (
        "**WITHDRAWN 2026-09-02. ORIGINAL, PRESERVED**: 'phase4's "
        "partition-sensitivity arm measured a range of 0.068 across "
        "partitions -- LARGER THAN THE SEED BAND. The five-seed "
        "protocol therefore VARIES THE SMALLER SOURCE OF VARIATION "
        "WHILE HOLDING THE LARGER ONE FIXED, and the record can now say "
        "so with an external citation rather than only its own "
        "measurement.' **See PARTITION_SENSITIVITY_CLAIM_WITHDRAWN for "
        "both reasons.**"
    ),
    "what_replaces_it": (
        "**the project has NOT independently measured split-induced "
        "variance.** That is now a GAP the record states rather than a "
        "claim it makes. Bouthillier's finding stands as banked and "
        "carries itself without corroboration from here"
    ),
    "a_limitation_not_a_defect": (
        "**the paired BCa remains VALID FOR WHAT IT MEASURES**: "
        "variation under a FIXED partition, which is the quantity every "
        "banked contrast is defined on. What it does not measure is "
        "variation under re-splitting, and nothing in the record ever "
        "claimed it did. **Nothing banked becomes wrong; the uncertainty "
        "statement becomes NARROWER than the general one**"
    ),
}


#: **[REGISTERED 2026-09-01 -- NOTED, NOT COMMITTED] A split-randomised
#: P(A>B) arm.**
#:
#: **What it would be**: Bouthillier's estimator on this cohort --
#: independent re-splits rather than seeds on a fixed partition, the mean
#: paired indicator, percentile-bootstrap CIs.
#:
#: **Why it is noted and not committed**: it is a **different and much
#: larger measurement** than anything the project has run. Every banked
#: contrast is defined on ``cleft_v1``'s fixed folds; a re-split arm
#: measures a different quantity and would need its own registration,
#: its own criteria, and a ruling on whether existing contrasts are
#: re-expressed against it or left alone. **Noting it costs nothing;
#: committing to it without that ruling would be scope arriving through
#: a citation.**
SPLIT_RANDOMISED_ARM_NOTED = {
    "status": "NOTED, NOT COMMITTED -- 2026-09-01",
    "what_it_would_be": (
        "Bouthillier's estimator on this cohort: INDEPENDENT RE-SPLITS "
        "rather than seeds on a fixed partition, the mean paired "
        "indicator, percentile-bootstrap CIs"
    ),
    "why_not_committed": (
        "a DIFFERENT AND MUCH LARGER measurement than anything the "
        "project has run. Every banked contrast is defined on cleft_v1's "
        "FIXED FOLDS; a re-split arm measures a different quantity and "
        "would need its own registration, its own criteria, and a ruling "
        "on whether existing contrasts are re-expressed against it or "
        "left alone. **Noting it costs nothing; committing without that "
        "ruling would be scope arriving through a citation**"
    ),
    "what_would_have_to_be_ruled_first": (
        "whether the 68 locked arms are re-run under re-splitting (they "
        "would have to be, for a paired indicator), which is a cluster "
        "cost no citation by itself justifies"
    ),
}


#: **[LITERATURE] Nadeau & Bengio (2003) -- A CORRECTION TO AN
#: ASSERTION IN THIS RECORD.**
#:
#: **The corrected resampled t-test DOES NOT APPLY TO CORRELATION.** It
#: requires a metric expressible as **an average of per-example losses**.
#: A correlation coefficient is **non-linear and non-additive** -- it is
#: not computable on a single test observation, so there is no per-example
#: loss to average. **Smallest validated n = 200.** It **would** apply to
#: MSE or MAE.
#:
#: **THE ERROR, AND WHERE IT CAME FROM.** It was proposed in discussion for this
#: project's **PCC** comparisons in conversation on **2026-08-31**. The
#: source contradicts that. Named on the same footing as
#: ``ladder.THE_ERROR_PROVENANCE`` (the 0.12-0.14 resolution figure that
#: was never measured -- the phrase itself is prohibited by
#: ``ladder.DETECTION_FLOOR_PROHIBITION`` and is not written here),
#: ``phase20.CROSS_TARGET_ERROR_PROVENANCE``,
#: ``phase20.S_DESCRIPTION_ERROR_PROVENANCE`` and
#: ``phase12.TWO_VIEW_CLAIM_PROVENANCE``.
#:
#: **WHAT IT DID NOT TOUCH, and this half matters.** **Nothing measured.**
#: The project has never applied NB: ``docs/PLAN.md``'s 2026-07-28
#: amendment RETRACTED the NB requirement from §4.3 on the ground that it
#: *"corrects a test this design does not run"*, and
#: ``docs/FROZEN_KNOWN_STALE.md`` entry 1 records the surviving docstring
#: mention in frozen ``eval/metrics.py`` as known-stale, with the same
#: reasoning. **The record was already right, twice over, and the
#: conversational recommendation contradicted the record as well as the
#: source.**
NADEAU_BENGIO_DOES_NOT_APPLY = {
    "tag": "[LITERATURE]",
    "source": "Nadeau & Bengio (2003), the corrected resampled t-test",
    "provenance": "quoted from the published paper; the PDF is not in this repository",

    "it_does_not_apply_to_correlation": (
        "the corrected resampled t-test requires a metric expressible as "
        "AN AVERAGE OF PER-EXAMPLE LOSSES. A correlation coefficient is "
        "NON-LINEAR AND NON-ADDITIVE -- not computable on a single test "
        "observation, so there is no per-example loss to average"
    ),
    "smallest_validated_n": 200,
    "what_it_would_apply_to": "MSE or MAE",

    "the_error_and_where_it_came_from": (
        "**It was proposed in discussion for this project's PCC comparisons in "
        "conversation on 2026-08-31.** The source contradicts that. "
        "Named on the same footing as ladder.THE_ERROR_PROVENANCE, "
        "phase20.CROSS_TARGET_ERROR_PROVENANCE, "
        "phase20.S_DESCRIPTION_ERROR_PROVENANCE and "
        "phase12.TWO_VIEW_CLAIM_PROVENANCE"
    ),
    "what_it_did_not_touch": (
        "**nothing measured.** The project has NEVER applied NB: "
        "docs/PLAN.md's 2026-07-28 amendment RETRACTED the NB "
        "requirement from 4.3 because it 'corrects a test this design "
        "does not run', and docs/FROZEN_KNOWN_STALE.md entry 1 records "
        "the surviving docstring mention in frozen eval/metrics.py as "
        "known-stale with the same reasoning"
    ),
    "the_sharper_point": (
        "**the record was already right, TWICE OVER.** The "
        "conversational recommendation contradicted the RECORD as well "
        "as the SOURCE -- so this was not a gap in the project's "
        "knowledge but a failure to consult it"
    ),
}


#: **[LITERATURE] The remaining sources, banked with what each
#: **[WITHDRAWN 2026-09-02] The 0.068 partition-sensitivity claim.**
#:
#: ``BOUTHILLIER_SAMPLE_SIZE`` claimed this project *"arrived at the same
#: conclusion independently"*, citing a partition-sensitivity range of
#: **0.068, larger than the seed band**. **Two defects, either of which
#: alone withdraws it.**
PARTITION_SENSITIVITY_CLAIM_WITHDRAWN = {
    "withdrawn": "2026-09-02, at the Phase 23 verification",
    "the_original": (
        "**PRESERVED VERBATIM** in the entry itself and in its "
        "docstring: 'THIS PROJECT ARRIVED AT THE SAME CONCLUSION "
        "INDEPENDENTLY. Phase 4's partition-sensitivity arm measured a "
        "range of 0.068 across partitions -- larger than the seed "
        "band.'"
    ),
    "defect_1_the_figure_has_no_source": (
        "**0.068 appears NOWHERE in the record except the entry "
        "asserting it.** There is no phase4 module and no banked "
        "outturn for the partition-sensitivity arm. The arm itself is "
        "real -- run.task_partition_sensitivity and "
        "configs/p4_partition.yaml both exist -- but **its result was "
        "never banked**, so the figure is a claim about a measurement "
        "rather than a measurement"
    ),
    "defect_2_the_quantities_are_not_comparable": (
        "**task_partition_sensitivity's OWN DOCSTRING**: 'NOT "
        "fold-assignment variance -- the frozen generator is "
        "deterministic, so that cannot be measured here at all.' It "
        "varies **fold COUNTS** (5, 6, 10) and LOPO, all using the "
        "frozen generator unchanged. **Bouthillier's estimator requires "
        "the dataset SPLIT RANDOMISED at fixed k.** Sensitivity to k "
        "and sensitivity to re-splitting are different quantities, so "
        "even a banked 0.068 would not have supported the claim"
    ),
    "what_remains_true_so_the_withdrawal_does_not_overshoot": (
        "**Bouthillier's finding stands as banked** -- data sampling "
        "dominating weight initialisation is theirs, cited, and needs "
        "no corroboration from here. **The five-seeds-on-a-fixed-split "
        "limitation stands**: it is FixHOptEst, it gives severe "
        "underestimations of standard error, and that is true of this "
        "protocol whatever this project has or has not measured. "
        "**Only the claim to have arrived there independently is "
        "withdrawn**"
    ),
    "and_what_is_now_a_stated_gap": (
        "**the project has NOT independently measured split-induced "
        "variance.** Previously the record CLAIMED it had; now it "
        "states it has not. That is a gap Phase 23 may choose to close "
        "or to leave, and it may no longer be treated as already closed"
    ),
    "tag": "[MEASURED] -- both defects verified at source",
}


#: **[NAMED 2026-09-02] Same-digit collisions are a recurring hazard in
#: this record, not a coincidence. Third instance.**
SAME_DIGIT_COLLISIONS = {
    "named": "2026-09-02, on the third instance",
    "the_three": {
        "0_628": (
            "``reliability.item_total``'s docstring carries the digits "
            "of an ITEM-TOTAL CORRELATION; the same digits were nearly "
            "read as the label sd. Caught, and "
            "configs/p21_cohort_pair_separation.yaml warns about it in "
            "its own header"
        ),
        "0_1386": (
            "the smallest resolvable delta, which also appears as a "
            "CLASSIFICATION margin in classification.py ('above it by "
            "0.1386 at the lower bound') -- a different quantity, same "
            "digits"
        ),
        "0_068": (
            "**this one.** ``ladder.py`` carries 0.068 as **SHA DRIFT "
            "on the same nominal arm** ('the void ladder's hazard was "
            "0.068 PCC drift across SHAs'); the literature entry "
            "carried it as **PARTITION SENSITIVITY**. Two quantities, "
            "identical digits, one module apart -- and the second had "
            "no source of its own"
        ),
    },
    "why_it_is_a_hazard_and_not_a_coincidence": (
        "**PCC differences on this cohort live in a narrow range**, so "
        "a handful of two- and three-digit values recur across "
        "unrelated quantities. A figure recalled rather than read is "
        "therefore MORE likely than usual to land on a real number "
        "belonging to something else -- **which reads as corroboration "
        "instead of as an error.** The 0.068 did exactly that: it "
        "looked like a measurement because a measurement with those "
        "digits exists"
    ),
    "the_defence_that_works": (
        "**quote figures from their structured source, never from "
        "prose or memory** -- the standing rule that produced "
        "ARM_A_THE_SHRINKAGE_CONTROL's `kappa_measured` mirror. A "
        "figure read from a named field cannot collide with a "
        "same-digit figure elsewhere, because the field says which "
        "quantity it is"
    ),
    "tag": "[REASONED] -- three instances, one mechanism",
}


#: **[FOUND AT VERIFICATION 2026-09-02] Split randomisation is not
#: reachable by a parameter. A scoping constraint on Phase 23, recorded
#: WITHOUT scoping the phase.**
SPLIT_RANDOMISATION_IS_NOT_REACHABLE = {
    "found": "2026-09-02, measured during the Phase 23 verification",
    "the_measurement": (
        "``folds.generate`` at seed 1337 against seed 2024, 60 "
        "patients: **60 of 60 keep their fold. The partition is "
        "byte-identical.** The seed does not move the assignment at all"
    ),
    "why_the_frozen_module_already_says_so": (
        "``data/folds.py``, which is FROZEN APPARATUS, documents it: "
        "'**seed is recorded but does not affect the partition.** The "
        "splitter runs with shuffle=False because that is what "
        "stratifies well... **If a phase wants repeated CV, that needs "
        "a different splitter, not a seed here.**' It also records WHY: "
        "on the real 88/119/30 distribution shuffle=False balances "
        "classes better, and five seeds gave byte-identical membership"
    ),
    "so_the_answer_is_sharper_than_the_question": (
        "the question was whether StratifiedGroupKFold with a seeded "
        "shuffle IS Bouthillier's re-splitting or something else. "
        "**There is no seeded shuffle.** Split randomisation is not a "
        "different thing the machinery does -- it is absent, and "
        "**a new splitter in FROZEN apparatus would be required**"
    ),
    "the_cost_from_the_banks_own_N": (
        "Bouthillier's N = 29 paired runs. **One contrast = 29 runs of "
        "each arm = 58 runs.** ``SPLIT_RANDOMISED_ARM_NOTED`` already "
        "states that a paired indicator would need the 68 locked arms "
        "re-run: **68 x 29 = 1,972 runs**, which it calls 'a cluster "
        "cost no citation by itself justifies'"
    ),
    "what_this_record_does_and_does_not_do": (
        "**it does NOT scope Phase 23** -- no arm, no criterion, no "
        "ruling, and no judgement on whether the phase should proceed. "
        "**It records a constraint discovered at verification so the "
        "restate must ADDRESS it rather than INHERIT it.** A restate "
        "that opens on 'split-randomised P(A>B)' without saying which "
        "splitter and at what cost would be scoping past a known "
        "blocker"
    ),
    "tag": "[MEASURED] -- run on this machine, and confirmed at source",
}


#: contradicts or qualifies.** All read ; none reachable from
#: this machine.
SOURCES_BANKED = {
    "provenance": (
        "all read ; NONE reachable from this machine, so no "
        "figure below was verified at source by this session"
    ),

    "benavoli_rope": {
        "tag": "[LITERATURE]",
    #: **[BANKED 2026-09-02] The verbatim passages**, each with its
    #: section or equation number, so every Phase 23 parameter cites a
    #: QUOTE rather than a relayed summary. Read  against the
    #: source PDF (arXiv 1606.04316); **the paper is NOT reachable from
    #: this machine**, so these are transcriptions and the record says
    #: so, as the entry's own provenance line does.
    #:
    #: **NINE passages, not eight.** The request called them eight and
    #: listed nine; banked as nine and the count is stated here so the
    #: discrepancy is not silently absorbed.
    "passages": {
        "provenance": (
            "read  against 1606.04316.pdf via NotebookLM; "
            "NOT reachable from this machine, so these are "
            "transcriptions rather than reads"
        ),
        "1_prior_section_3": (
            "the Normal-Gamma specification p(mu,nu|mu_0,k_0,a,b) = "
            "N(mu; mu_0, k_0/nu) G(nu; a,b), and the matching-prior "
            "values: **'If we choose the prior parameters {mu_0 = 0, "
            "k_0 -> infinity, a = -1/2, b = 0} (matching prior)...'**"
        ),
        "2_posterior_eq_6_section_3": (
            "**St(mu; n-1, xbar, (1/n + rho/(1-rho)) sigmahat^2)**, "
            "with xbar and sigmahat^2 defined there"
        ),
        "3_x_as_differences_eq_4_section_3": (
            "**'x = (x_1, x_2, ..., x_n) is the vector of differences "
            "of accuracy.'**"
        ),
        "4_rho_unidentifiable_section_3": (
            "**'The likelihood (5) does not allow to estimate rho from "
            "data, since the maximum likelihood estimate of rho is "
            "rho-hat = 0 regardless the observations... thus the "
            "Bayesian correlated t-test adopts the same heuristic rho = "
            "n_te/n_tot suggested by Nadeau and Bengio (2003).'**"
        ),
        "5_rho_qualification_footnote_2": (
            "**'Nadeau and Bengio (2003) considered the case in which "
            "random training and test sets are drawn... This is "
            "slightly different from k-fold cross-validation, in which "
            "the folds are designed not to overlap. However the "
            "correlation heuristic... has since become commonly "
            "used.'**"
        ),
        "6_rope_and_the_one_percent_default_section_3_1_footnote_3": (
            "the [-0.01, 0.01] definition, and **'In classification 1% "
            "seems to be a reasonable choice. However, in other domains "
            "a different value could be more suitable.'**"
        ),
        "7_p_rope_as_integral_section_3_2": (
            "**'namely the integral of the posterior over the rope "
            "interval.'**"
        ),
        "8_their_n_and_rho_sections_2_and_3": (
            "**'n = 100 (10 runs of 10-fold cross-validation)'** and "
            "**'rho = 1/10, n = 100.'**"
        ),
        "9_the_loss_matrix_eq_7_section_3_2": (
            "the matrix with its entries, and **'Since 0.05 * 20 = 1, "
            "this leads to the same decision rule discussed previously "
            "(P(.) > 0.95).'** **Introduced as 'Consider for instance "
            "the following loss matrix' and 'For instance we can "
            "decide...' -- an EXAMPLE, which is why the threshold is "
            "ruled by this project rather than cited"
        ),
    },

        "source": (
            "Benavoli, Corani, Demsar & Zaffalon, JMLR 2017 -- "
            "'Time for a Change', arXiv 1606.04316. ROPE and the "
            "Bayesian correlated t-test. [arXiv id added 2026-09-02 "
            "with the passages]"
        ),
        "what_it_offers": (
            "a region of practical equivalence and a Bayesian correlated "
            "t-test, which report P(difference is negligible) rather "
            "than only rejecting a null"
        ),
        "how_it_bears": (
            "**it names the thing this cohort keeps producing.** "
            "ladder.COHORT_CANNOT_RESOLVE's 0.04-0.10 band IS a region "
            "of practical equivalence discovered empirically and without "
            "the vocabulary. Thirty withdrawn contrasts would read "
            "differently as 'probably equivalent' than as 'not "
            "claimable' -- the same data, a less misleading sentence"
        ),
        "what_it_does_not_do": (
            "it does not RESOLVE anything the cohort cannot resolve. A "
            "ROPE analysis would restate the limit in better language, "
            "not lift it"
        ),
    },

    "rankiqa": {
        "tag": "[LITERATURE]",
        "source": "RankIQA",
        "pair_formation_and_loss": (
            "ranked pairs generated by applying known distortions at "
            "known strengths, trained with a pairwise ranking loss, then "
            "fine-tuned for the scalar target"
        ),
        "reported_gains_and_smallest_dataset": (
            "as reported by the authors, on their datasets; the smallest "
            "dataset they use is the figure that matters here"
        ),
        "how_it_bears": (
            "the synthetic-distortion premise does not transfer: their "
            "ranking signal comes from distortions applied to a SINGLE "
            "image, so the pair's order is known BY CONSTRUCTION. Cleft "
            "grade ordering between two DIFFERENT patients is exactly "
            "what is uncertain here -- the thing their method assumes "
            "free is this project's whole difficulty"
        ),
        "flagged": (
            "**a gain reported on their smallest dataset is not a "
            "prediction for n=237 on a different task**, and quoting one "
            "as encouragement would be the comparator error this project "
            "has made before"
        ),
    },

    "zimmerman_williams": {
        "tag": "[LITERATURE]",
        "source": "Zimmerman & Williams",
        "what_it_states": (
            "the exact conditions under which a reliability coefficient "
            "is valid, and the sample sizes those conditions require"
        ),
        "how_it_bears": (
            "directly on data.reliability's MEAN_R_237 = 0.4696 and the "
            "item-total weighting: a reliability figure quoted without "
            "its conditions is the same shape as a tau quoted without "
            "its tie structure"
        ),
        "flagged": (
            "**to be checked against this cohort's n and rater count "
            "before any reliability figure is quoted in the write-up.** "
            "Not checked here -- the source is not reachable, and "
            "asserting the conditions are met would be the mis-tagging "
            "error again"
        ),
    },

    "watson_wright": {
        "tag": "[LITERATURE]",
        "source": "Watson & Wright -- conditional predictive impact",
        "requirements_and_small_n_behaviour": (
            "CPI's stated requirements, and how it behaves at small n"
        ),
        "how_it_bears": (
            "CPI is the principled version of what phase13's confound "
            "ceiling approximates: it asks what a feature adds GIVEN the "
            "others, rather than what a feature set predicts alone"
        ),
        "flagged": (
            "**its small-n behaviour is the reason it was not used and "
            "must be stated if it is ever proposed.** n=237 with five "
            "confound statistics is not obviously inside its validated "
            "range, and the ceiling arm answered the simpler question it "
            "could actually support"
        ),
    },

    "swayamdipta_cartography": {
        "tag": "[LITERATURE]",
        "source": "Swayamdipta et al. -- dataset cartography",
        "definitions": (
            "CONFIDENCE is the mean probability assigned to the gold "
            "label across training epochs; VARIABILITY is its standard "
            "deviation across those epochs"
        ),
        "whether_regression_is_addressed": (
            "**NO.** Both quantities are defined on the probability of a "
            "GOLD LABEL, which a regression target does not have"
        ),
        "how_it_bears": (
            "**it is the nearest published relative of phase21 Arm B's "
            "hardness vector, and it is NOT the same thing.** Cartography "
            "measures across TRAINING EPOCHS within one model; Arm B "
            "measures across ARMS at convergence. Neither is a "
            "substitute for the other, and calling Arm B's hardness "
            "'dataset cartography' would be the different-quantities-"
            "under-one-name error"
        ),
        "flagged": (
            "if a cartography-style analysis is ever wanted, it needs a "
            "regression adaptation of its own, registered as such -- not "
            "borrowed by analogy from Arm B"
        ),
    },
}


#: **[LITERATURE] THE COMPARATOR RECORD -- Guan and Heinrich.** Read by
#: the maintainer; not reachable here. What the published comparators actually
#: did, methodologically, beside what this project does.
COMPARATORS_BANKED = {
    "provenance": "quoted from the published paper; the PDF is not in this repository",

    "guan": {
        "tag": "[LITERATURE]",
        "labelling": (
            "adjudicated consensus -- three neurosurgeons plus a FOURTH "
            "ARBITRATOR"
        ),
        "rater_agreement": (
            "Fleiss kappa 0.31 on classic House-Brackmann; 27.3% "
            "identical grades; pairwise Cohen's kappa 0.23 / 0.37 / 0.33"
        ),
        "reported_correlation": (
            "Spearman 0.892 / 0.890 on the training cohort and 0.857 / "
            "0.875 on the prospective seven-centre cohort"
        ),
        "what_it_does_not_report": (
            "**no confidence intervals, deterministic single run.** So "
            "the figures are point estimates with no uncertainty "
            "statement of any kind -- which is the comparison this "
            "project's every-figure-with-its-spread discipline should be "
            "read against"
        ),
        "the_n_274_basis_is_an_inference": {
            "tag": "[REASONED]",
            "the_inference": (
                "the correlation is taken to rest on n = 274, INFERRED "
                "from P < 0.001 being unattainable on 5-6 aggregated "
                "points -- **not stated by the authors**"
            ),
            "the_arithmetic": (
                "the exact minimum two-tailed p for a rank correlation "
                "is 2/n!: 0.0167 at n=5, 0.00278 at n=6, 0.000397 at "
                "n=7. So P < 0.001 is unreachable below n = 7"
            ),
            "verified_here": (
                "**the arithmetic is checked, the inference is not.** "
                "2/5! = 0.016667, 2/6! = 0.002778, 2/7! = 0.000397 all "
                "reproduce exactly, and a brute-force permutation null "
                "at n=5 and n=6 confirms 2/120 and 2/720. What remains "
                "[REASONED] is the step from 'P<0.001 needs n>=7' to "
                "'the basis was n=274' -- that is an inference about "
                "what the authors did, and it is recorded AS an "
                "inference"
            ),
        },
        "label_collapsing_supports_three_not_five": (
            "**merging neighbouring grades raised consistency from 27.3% "
            "to 48.2% (eyelid) and Fleiss kappa from 0.31 to 0.53** -- "
            "a 1.77x and 1.71x improvement, checked here. **External "
            "support for classification.THREE_NOT_FIVE**: an independent "
            "group, on a different clinical scale, found the same thing "
            "this project found -- that a five-point scale carries more "
            "categories than raters can reliably separate"
        ),
    },

    "heinrich": {
        "tag": "[LITERATURE]",
        "design": "retrospective, Stennert Index, single examiner per time point",
        "reliability": (
            "'intra- and inter-examiner reliability could not be "
            "formally assessed' -- the authors' own words"
        ),
        "what_it_lacks": (
            "no split; MediaPipe's 478 landmarks validated ONLY "
            "QUALITATIVELY; single centre; single run"
        ),
        "how_it_bears": (
            "it is a comparator whose methodology is weaker than this "
            "project's on every axis that has been contested here -- "
            "splitting, reliability, landmark validation, seed "
            "variation. **That is context for the write-up, not a "
            "licence**: being more careful than a weak comparator is not "
            "evidence that a number is right"
        ),
    },

    "the_pearson_vs_spearman_hazard": (
        "**Guan reports SPEARMAN; this project's headline 0.2520 is "
        "PEARSON.** They are different quantities and are not "
        "comparable as stated. A comparator table placing 0.892 beside "
        "0.2520 would be comparing a rank correlation on an adjudicated "
        "consensus label against a linear correlation on a panel mean, "
        "across different anatomies, cohorts and tasks. **Any comparator "
        "table must name which coefficient each figure is**, and this is "
        "a tested note rather than a hope"
    ),
}


def summary() -> dict:
    """The literature bank, importable as one object."""
    return {
        "tag_discipline": TAG_DISCIPLINE,
        "geirhos": GEIRHOS_ERROR_CONSISTENCY,
        "bouthillier": BOUTHILLIER_SAMPLE_SIZE,
        "split_randomised_arm": SPLIT_RANDOMISED_ARM_NOTED,
        "nadeau_bengio": NADEAU_BENGIO_DOES_NOT_APPLY,
        "sources": SOURCES_BANKED,
        "comparators": COMPARATORS_BANKED,
    }
