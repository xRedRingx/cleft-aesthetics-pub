"""Phase 23: the statistical instruments -- the ROPE half only.

Opened 2026-09-02 per the seventh amendment. **This module is the
REGISTRATION and nothing else** -- no task, no schema kind, no config,
no run.

Two instruments were scheduled. **One is built and one is not**, and the
reason the second is not is measured rather than argued
(``SPLIT_RANDOMISED_HALF_NOT_BUILT``).

**What this phase does: it changes what can be SAID about figures already
banked. It changes no figure, and it resolves nothing this cohort cannot
resolve.** Both halves of that sentence are load-bearing and the second
is quoted from the bank rather than softened.
"""

from __future__ import annotations


#: **[RECKONED 2026-09-02, EVERY FIGURE VERIFIED AT SOURCE] The opening
#: reckoning.**
#:
#: **What this phase is for.** ``ladder.COHORT_CANNOT_RESOLVE`` records
#: that this cohort cannot resolve PCC differences of 0.04 to 0.10 --
#: *"30 tested, 1 survived, 29 withdrawn"* -- and
#: ``SMALLEST_RESOLVABLE_DIFFERENCE`` puts the smallest delta ever to
#: satisfy both PLAN 4.3 conditions at **0.1386**. Nineteen ledger rows
#: sit unresolved or withdrawn as a result.
#:
#: **A withdrawn row currently says "not claimable".** That is a
#: statement about the criterion, not about the arms: it does not
#: distinguish *"these two arms are the same"* from *"we could not
#: tell"*. A ROPE analysis reports **P(the difference is negligible)**,
#: which separates them.
#:
#: **AND IT RESOLVES NOTHING.** The bank's own limit, quoted:
#: ``SOURCES_BANKED['benavoli_rope']["what_it_does_not_do"]`` -- *"it
#: does not RESOLVE anything the cohort cannot resolve. A ROPE analysis
#: would restate the limit in better language, not lift it."* This phase
#: is a change of vocabulary applied to fixed data, and the reckoning
#: says so before any reading is registered.
PHASE_23_RECKONING = {
    "reckoned": "2026-09-02, every figure verified at source",
    "what_it_does": (
        "**it changes what can be SAID about banked figures.** A "
        "withdrawn row says 'not claimable', which is a statement about "
        "the CRITERION: it does not distinguish 'these two arms are the "
        "same' from 'we could not tell'. **A ROPE reports P(the "
        "difference is negligible), which separates them** -- and "
        "practical equivalence is a conclusion NHST cannot reach at all"
    ),
    "what_it_does_not_do": (
        "**it changes NO FIGURE and resolves NOTHING the cohort cannot "
        "resolve.** Quoted from the bank rather than paraphrased: 'it "
        "does not RESOLVE anything the cohort cannot resolve. A ROPE "
        "analysis would restate the limit in better language, not lift "
        "it' (SOURCES_BANKED['benavoli_rope']). **Every delta, "
        "threshold, margin and verdict already banked stands exactly as "
        "computed**"
    ),
    "what_the_record_holds": (
        "**ladder.COHORT_CANNOT_RESOLVE**: 'this cohort cannot resolve "
        "PCC differences of 0.04 to 0.10 between arms' -- 30 tested, 1 "
        "survived, 29 withdrawn. **SMALLEST_RESOLVABLE_DIFFERENCE**: "
        "the smallest delta ever to satisfy both PLAN 4.3 conditions is "
        "**0.1386**. **Nineteen ledger rows** unresolved or withdrawn: "
        "8 UNRESOLVED-WITHDRAWN and 11 WITHDRAWN, counted at source "
        "(phase21.THE_SEVENTEEN_WAS_NEVER_COUNTED)"
    ),
    "coverage_nothing_to_concede": (
        "**nothing in the record computes a P(A>B) or any "
        "practical-equivalence statement.** Swept 2026-09-02: the only "
        "occurrences of either are the literature bank and the "
        "amendment scheduling this phase. **And `n_excluding_zero` is "
        "NOT P(A>B)** -- it is a count of SEEDS on a FIXED SPLIT, not a "
        "paired indicator over INDEPENDENT SPLITS. The two differ in "
        "what is resampled, which is the whole of Bouthillier's point"
    ),
    "the_honest_form_of_the_audit": (
        "stated because a coverage audit that CAN come back positive is "
        "the only kind worth running -- Phase 21's did, with "
        "phase7b.ENSEMBLE_COMBINES. Here it does not"
    ),
    "tag": "[REASONED]; every cited figure [MEASURED] at its source",
}


#: **[NOTED, NOT COMMITTED 2026-09-02] The split-randomised P(A>B) half
#: is NOT BUILT.**
#:
#: The amendment scheduled two instruments. **This one is refused for a
#: measured reason, and it is registered here so it cannot arrive later
#: as a fresh idea** (Phase 14's pattern).
SPLIT_RANDOMISED_HALF_NOT_BUILT = {
    "noted_not_committed": "2026-09-02, at the restate",
    "the_measured_reason": (
        "**split randomisation is UNREACHABLE BY PARAMETER.** "
        "``data/folds.py`` runs ``shuffle=False``; the seed is recorded "
        "but inert. **Measured on this machine: ``folds.generate`` at "
        "seed 1337 against seed 2024 leaves 60 of 60 patients in their "
        "fold -- the partition is byte-identical.** The frozen module "
        "says it itself: 'seed is recorded but does not affect the "
        "partition... if a phase wants repeated CV, that needs a "
        "DIFFERENT SPLITTER, not a seed here'"
    ),
    "so_it_would_require_frozen_apparatus": (
        "**a new splitter inside FROZEN apparatus.** Not a config "
        "field, not a flag -- the capability is absent, and "
        "``data/folds.py`` is on the frozen list"
    ),
    "and_the_cost_from_the_banks_own_N": (
        "Bouthillier's **N = 29 paired runs**. One contrast = 29 runs "
        "of each arm = **58 runs**; the 68 locked arms re-run for a "
        "paired indicator = **1,972 runs**. "
        "``literature.SPLIT_RANDOMISED_ARM_NOTED`` already calls that "
        "'a cluster cost no citation by itself justifies'"
    ),
    "what_is_therefore_true_and_stays_true": (
        "**the five-seed protocol IS Bouthillier's biased estimator** "
        "(FixHOptEst, 'severe underestimations of standard error'), and "
        "**this phase does not fix that.** It restates fixed-split "
        "results in better language while the fixed-split limitation "
        "stands. Saying otherwise would be the phase claiming a "
        "correction it does not make"
    ),
    "registered_so_it_cannot_return_as_a_fresh_idea": (
        "**EXPLICITLY UNREGISTERED -- no fishing.** The arm is "
        "described, its cost is counted, and it is not taken. A later "
        "turn proposing 'split-randomised P(A>B)' is proposing THIS, "
        "and must answer the splitter and the 1,972 runs"
    ),
    "tag": "[MEASURED] for the reason; [NOTED] for the arm",
}


#: **[CORRECTED 2026-09-02] Five settings were attributed to the paper
#: with nothing in the record behind them.**
#:
#: ``THE_METHOD_REGISTERED`` recorded the prior, rho, the posterior form,
#: P(rope)-as-integral and the 0.95 threshold as *"taken from the
#: paper"*. **The bank held none of them.** Its ``benavoli_rope`` entry
#: had four summary fields whose only relevant content was the phrase
#: *"correlated t-test"* -- no prior, no rho, no threshold, no loss
#: matrix, no posterior form, no n.
#:
#: **The attributions were true but uncheckable, which is the defect.**
THE_FIVE_SOURCE_ATTRIBUTIONS_WERE_UNVERIFIED = {
    "corrected": "2026-09-02, before the lock",
    "whose_error": (
        "**the record's.** The five entered from a CONVERSATIONAL RELAY of "
        "NotebookLM answers and were written into the registration as "
        "source-attributed **with no citable entry behind them**. A "
        "reader could not have checked one of them without the paper, "
        "which the record says is unreachable from this machine"
    ),
    "the_shape_it_repeats": (
        "**the same as the 0.068 and the 17**: a figure asserted "
        "fluently, carrying the authority of a citation, with no "
        "banked source to check it against. **Third instance of "
        "citation-without-entry, and the second in this phase's own "
        "preparation.** Filed beside the other named errors"
    ),
    "what_caught_it": (
        "**a test, not a re-reading.** The scoping cycle asked whether "
        "any of the five was 'a choice dressed as a citation' and the "
        "check ran against the bank's own fields rather than against "
        "the sentence asserting them. The bank answered no to all five"
    ),
    "what_it_cost": (
        "**nothing measured** -- no posterior had been computed, and "
        "the parameters turn out to be right. **What it cost was the "
        "record's ability to verify its own method section**, which is "
        "the whole point of banking a source"
    ),
    "the_remedy_applied": (
        "**nine verbatim passages banked** with section and equation "
        "numbers (literature.SOURCES_BANKED['benavoli_rope']"
        "['passages']), so every Phase 23 parameter now cites a quote. "
        "**And one attribution moved**: P > 0.95 is OURS, because the "
        "paper introduces its loss matrix as an example"
    ),
    "tag": "[MEASURED] -- the gap tested against the bank, not recalled",
}


#: **[REGISTERED 2026-09-02] The method, every parameter supplied by the
#: source and cited.**
#:
#: Benavoli, Corani, Demsar & Zaffalon (JMLR 2017), the **single-dataset
#: Bayesian correlated t-test**. Nothing here is chosen by this project
#: except the ROPE width, which is ruled separately and marked as ours.
THE_METHOD_REGISTERED = {
    "registered": "2026-09-02, before any posterior is computed",
    "the_test": (
        "**the single-dataset Bayesian correlated t-test.** Not the "
        "hierarchical model -- see "
        "THE_HIERARCHICAL_MODEL_IS_EXCLUDED"
    ),
    "the_prior_from_the_paper": (
        "**the matching prior: mu_0 = 0, k_0 -> infinity, a = -1/2, b = "
        "0.** Taken from the paper, not chosen here. It yields the "
        "closed-form Student posterior **St(mu; n-1, xbar, (1/n + "
        "rho/(1-rho)) * sigmahat^2)**, which is what makes the analysis "
        "a formula rather than a sampler"
    ),
    #: **[QUOTED 2026-09-02]** rho is SOURCE-attributed and is not a
    #: choice: the paper records that it **cannot be estimated** --
    #: 'the maximum likelihood estimate of rho is rho-hat = 0
    #: regardless the observations' -- so the heuristic is adopted from
    #: NECESSITY, not preference.
    "rho_is_not_a_choice_it_is_unidentifiable": (
        "**quoted, section 3**: 'The likelihood (5) does not allow to "
        "estimate rho from data, since the maximum likelihood estimate "
        "of rho is rho-hat = 0 regardless the observations... thus the "
        "Bayesian correlated t-test adopts the same heuristic rho = "
        "n_te/n_tot suggested by Nadeau and Bengio (2003).' **There is "
        "no version of this analysis in which rho is fitted**, so "
        "adopting the heuristic is the method, not a decision within it"
    ),
    "and_its_approximation_travels_with_it": (
        "**quoted, footnote 2**: 'Nadeau and Bengio (2003) considered "
        "the case in which random training and test sets are drawn... "
        "**This is slightly different from k-fold cross-validation, in "
        "which the folds are designed not to overlap.** However the "
        "correlation heuristic... has since become commonly used.' "
        "**Ours are non-overlapping k-fold folds**, so the heuristic is "
        "applied outside the case it was derived for -- by the paper's "
        "own admission and its own practice. **This qualification "
        "travels with every rho = 0.2 in this phase**"
    ),
    "rho_from_the_fold_ratio": (
        "**rho = 0.2**, the 5-fold ratio (1/k with k = 5). This follows "
        "**the paper's own practice of holding rho at the SINGLE-RUN "
        "fold ratio while pooling multiple runs into one vector**: they "
        "use rho = 1/10 with n = 100, from 10 runs of 10-fold. Ours is "
        "rho = 1/5 with n = 25, from 5 runs of 5-fold -- the same "
        "construction at a smaller scale"
    ),
    "the_observation_vector": (
        "**x_i = PER-FOLD PCC DIFFERENCES, 25 per contrast** (5 folds x "
        "5 seeds). Recoverable because the banked CSVs carry "
        "`patient_id, truth, prediction, fold`, so a per-fold PCC is "
        "computable"
    ),
    "the_probability": (
        "**P(rope) by integrating the posterior over the interval**, "
        "and P(left) / P(right) over the two tails. Three probabilities "
        "summing to one"
    ),
    "the_decision_rule_and_its_derivation": (
        "**[CORRECTED 2026-09-02 -- THIS IS OURS, NOT THE PAPER'S.** "
        "ORIGINAL: 'P(.) > 0.95. Derived in the paper from a loss "
        "matrix -- wrong decision 20, no decision 1, so 0.05 x 20 = 1 "
        "and the threshold is where the two losses balance. Recorded as "
        "TAKEN FROM THE PAPER rather than chosen, because a decision "
        "threshold picked by this project would be another undeclared "
        "constant.' **The paper introduces the matrix as an EXAMPLE** "
        "-- 'Consider for instance the following loss matrix', 'For "
        "instance we can decide...' -- so the 20:1 ratio is a choice it "
        "illustrates, not a value it prescribes.** See "
        "THE_DECISION_THRESHOLD_RULED"
    ),
    "what_this_project_supplies": (
        "**[CORRECTED 2026-09-02: TWO, not one.** ORIGINAL: 'the ROPE "
        "width, and nothing else. Every other parameter above is the "
        "source's.'** **The ROPE width (0.018346) AND the decision "
        "threshold (0.95)**, the second because the paper's loss matrix "
        "is an example rather than a prescription. **Four remain the "
        "source's**: the prior, the posterior form, rho, and "
        "P(rope)-as-integral -- each now carrying a banked verbatim "
        "quote (SOURCES_BANKED['benavoli_rope']['passages'])"
    ),
    "tag": "[LITERATURE] for the parameters; [REGISTERED] for the design",
}


#: **[CORRECTED 2026-09-02] The width was a rounded display quoted as
#: the quantity.**
#:
#: The config declared **0.0183**; the formula gives **0.018346**. The
#: task refused at launch -- **criterion 2's guard working exactly as
#: designed, before any posterior existed.**
THE_WIDTH_WAS_A_ROUNDED_DISPLAY = {
    "corrected": "2026-09-02, at the refused launch",
    "what_happened": (
        "**the task recomputed 0.0183460 against a declared 0.0183000 "
        "and refused.** No posterior was computed, no result was "
        "produced, and nothing had to be withdrawn -- **the guard "
        "caught it at the only moment where catching it is free**"
    ),
    "whose_error": (
        "**the record's.** 0.0183 was a FOUR-DECIMAL DISPLAY from an "
        "earlier report, quoted forward into the ruling, the records "
        "and the config **as if it were the value**. The ruling's "
        "substance was always the QUANTITY -- 'closer than the "
        "criterion can distinguish at five seeds' -- and that quantity "
        "is 0.018346"
    ),
    "the_family_it_belongs_to": (
        "**the other citation-precision slips**: the 0.068 with no "
        "source, the 17 never counted, the five unverified "
        "attributions. Same shape -- a number carried forward from a "
        "display rather than read from its source. **This one differs "
        "in how it was caught: by MACHINERY, not by reading.** The "
        "guard existed because criterion 2 required it, and it fired "
        "on its first real use"
    ),
    "the_ruling_is_unchanged_only_its_expression": (
        "**The ruling was a QUANTITY, not a numeral.** "
        "combined_claimable_delta's arm_means_95 at the probe's sd and "
        "seed count IS the ruled width; 0.0183 was a lossy rendering of "
        "it. Correcting the digits does not reopen the ruling"
    ),
    "the_alternative_rejected": (
        "**declaring the width literally 0.0183 and comparing with a "
        "tolerance.** Rejected: **the guard's entire purpose is "
        "refusing a width nobody ruled, and slack in it defeats it.** A "
        "tolerance wide enough to accept 0.0183 would also accept any "
        "other value within 5e-5 -- including a genuinely wrong one. "
        "Recorded because a later reader will wonder why the config "
        "carries seven digits, and the answer is that the guard is "
        "exact on both sides"
    ),
    "tag": "[MEASURED] -- caught by the guard, not by a re-reading",
}


#: **[RULED 2026-09-02] The ROPE: +/- 0.018346.**
#:
#: From ``combined_claimable_delta``'s ``arm_means_95`` at sd 0.0148, n
#: = 5 -- the probe's own seed sd at the standing five seeds. Verified:
#: ``1.96 * sqrt(0.0148^2/5 + 0.0148^2/5) = 0.0183``.
#:
#: **The ground:** two arms are practically equivalent if their
#: difference is smaller than what the criterion can distinguish at five
#: seeds. **"Closer than we can measure" is a defensible definition of
#: equivalent** -- and it ties the ROPE to the project's own
#: two-condition machinery rather than to a number from outside it.
THE_ROPE_RULED = {
    "ruled": "2026-09-02",
    "the_width": (
        "**+/- 0.018346**, from combined_claimable_delta's "
        "`arm_means_95` at sd 0.0148, n = 5: 1.96 * sqrt(sd^2/n + "
        "sd^2/n), at FULL PRECISION. Recomputed at run time rather "
        "than pinned as a literal. **[CORRECTED 2026-09-02. ORIGINAL: "
        "'+/- 0.0183'** -- the form the ruling was ISSUED in, and a "
        "rounded DISPLAY rather than the quantity. See "
        "THE_WIDTH_WAS_A_ROUNDED_DISPLAY.]"
    ),
    "the_ground": (
        "**two arms are practically equivalent if their difference is "
        "smaller than what the criterion can DISTINGUISH at five "
        "seeds.** 'Closer than we can measure' is a defensible "
        "definition of equivalent, and it ties the ROPE to this "
        "project's own condition-2 machinery rather than importing a "
        "width from outside it"
    ),
    "the_four_rejected_candidates_unregistered": (
        "**EXPLICITLY UNREGISTERED -- no fishing** (Phase 14's "
        "pattern), each named with its value so none can return as a "
        "fresh idea. **(a) 0.0137**, MEASURED_SEED_BAND's sd at ten "
        "seeds -- a spread, not a distinguishability threshold. **(b) "
        "0.0410**, combined_claimable_delta's `single_run_95` -- the "
        "conservative companion, which describes a difference between "
        "two SINGLE runs and so is wider than the quantity being "
        "tested. **(c) the 0.04-0.10 band** "
        "(ladder.COHORT_CANNOT_RESOLVE) -- an empirically discovered "
        "region of practical equivalence, and the most tempting, but it "
        "is a RANGE rather than a width and choosing either end would "
        "be this project choosing. **(d) 0.1386**, the smallest "
        "resolved difference -- far too wide: it would call almost "
        "every contrast equivalent, including ones that differ by more "
        "than the phase's own headline"
    ),
    "why_not_a_per_contrast_width": (
        "**combined_claimable_delta is PER-CONTRAST, and using it that "
        "way would give EIGHTEEN DIFFERENT ROPEs.** The method assumes "
        "a **fixed domain-level width** -- a region of practical "
        "equivalence is a statement about the DOMAIN, not about a "
        "particular pair's seed spread. Eighteen widths would also make "
        "the rows incomparable with each other, which is most of what "
        "the analysis is for"
    ),
    "the_paper_gives_no_guidance_and_that_is_ours_to_carry": (
        "**the source addresses neither how to choose the width, nor "
        "post-hoc selection, nor multiple widths.** Verified against "
        "the bank: SOURCES_BANKED['benavoli_rope'] holds four summary "
        "fields and none covers it. **So the discipline here is "
        "ENTIRELY this project's: the width is FIXED BEFORE ANY "
        "POSTERIOR IS COMPUTED**, one width for all eighteen rows, and "
        "moving it afterwards would be choosing a method after seeing "
        "data"
    ),
    "tag": "[RULED] -- the one parameter this project supplies",
}


#: **[DECLARED 2026-09-02] What this application EXTENDS beyond the
#: source, and what the source never addresses.**
#:
#: Each named rather than absorbed, so a reader can weigh the analysis
#: against what its method actually licenses.
DECLARED_EXTENSIONS_AND_SILENCES = {
    "declared": "2026-09-02, at the restate",
    "1_pcc_is_a_non_additive_metric": (
        "**the paper never addresses non-additive metrics, so applying "
        "the test to PCC is an EXTENSION, not precedent.** Two things "
        "make it defensible and neither makes it licensed: the test "
        "**places no restriction on how each fold's number was "
        "produced** -- it models a vector of correlated observations -- "
        "and unlike **Nadeau & Bengio** it does **not** derive its "
        "variance on observation-wise losses, so the additivity that "
        "matters there does not enter here. **Recorded as an extension "
        "carried by this project**"
    ),
    "2_normality_is_assumed_and_never_tested": (
        "**the paper assumes multivariate normality of the fold "
        "differences and never checks it.** Neither do we, and "
        "NORMALITY_DESCRIBED_NOT_TESTED says what we do instead"
    ),
    "3_small_n_is_never_discussed": (
        "**the paper never discusses behaviour, reliability or a "
        "minimum sample size at small n.** Their examples use **n = "
        "100** (10 runs of 10-fold); **ours is 25**. The Student "
        "posterior has n - 1 = 24 degrees of freedom, which is "
        "workable, but **the source offers no assurance at this size "
        "and none is claimed on its behalf**"
    ),
    "4_extreme_rho_is_not_warned_about": (
        "the paper gives no warning on extreme rho. **Ours is benign**: "
        "at rho = 0.2 the correction factor rho/(1-rho) = 0.25, well "
        "away from the pole at rho -> 1. Recorded because a parameter "
        "that is safe here could be unsafe at a different fold count"
    ),
    "tag": "[REASONED] -- four gaps between the source and this use",
}


#: **[EXCLUDED 2026-09-02] The hierarchical model.**
#:
#: Benavoli's hierarchical test exists and is **not used here.**
THE_HIERARCHICAL_MODEL_IS_EXCLUDED = {
    "excluded": "2026-09-02, at the restate",
    "the_assumption_quoted": (
        "**its unit of replication is the DATASET**, with datasets "
        "drawn as **independent exchangeable units** from a population "
        "of problems. That is what the hierarchy is over"
    ),
    "why_it_does_not_apply": (
        "**eighteen contrasts on ONE cohort are not eighteen "
        "datasets.** They share 237 patients, one manifest, one fold "
        "structure and largely one backbone -- the opposite of "
        "independent exchangeable units. Using it would put a "
        "population-level claim on a single cohort"
    ),
    "the_paper_does_not_prohibit_the_misuse": (
        "**and that is a reason for caution rather than permission.** "
        "Nothing in the source refuses a caller who passes correlated "
        "within-dataset contrasts as if they were datasets; it simply "
        "does not contemplate it. **A method that does not forbid a "
        "misuse has not licensed it**"
    ),
    "tag": "[REASONED] -- excluded with its assumption named",
}


#: **[RULED 2026-09-02] The decision threshold: P(.) > 0.95,
#: and it is OURS.**
#:
#: The paper's loss matrix is introduced as an example -- *"Consider for
#: instance the following loss matrix"*, *"For instance we can
#: decide..."*. **The threshold follows necessarily from the matrix; the
#: matrix is a choice.** So the 20:1 ratio is ruled here.
THE_DECISION_THRESHOLD_RULED = {
    "ruled": "2026-09-02",
    "the_ruling": (
        "**the 20:1 loss ratio, and therefore P(.) > 0.95.** Since 0.05 "
        "* 20 = 1, the two losses balance at 0.95 -- the arithmetic is "
        "the paper's, the ratio is ours"
    ),
    "the_ground": (
        "**a wrong conclusion is twenty times worse than no "
        "conclusion**, which matches a project whose criterion has "
        "admitted **one result in thirty** "
        "(ladder.COHORT_CANNOT_RESOLVE: 30 tested, 1 survived). A "
        "record this reluctant to claim should be equally reluctant to "
        "declare equivalence"
    ),
    "and_the_direction_that_flatters_is_named": (
        "**a more lenient ratio would declare equivalence more "
        "readily** -- more rows resolved, a fuller-looking result, and "
        "the flattering direction. Choosing the strict end is choosing "
        "against the phase's own interest, which is why the ground is "
        "recorded rather than the number alone"
    ),
    "the_papers_matrix_is_its_basis_not_its_authority": (
        "**the source supplies the FORM and the arithmetic; this "
        "project supplies the RATIO.** Recorded this way because the "
        "first registration called the whole thing 'taken from the "
        "paper', which the paper's own 'for instance' does not support "
        "(THE_FIVE_SOURCE_ATTRIBUTIONS_WERE_UNVERIFIED)"
    ),
    "tag": "[RULED] -- the second of two settings this project supplies",
}


#: **[EXCLUDED 2026-09-02] The IEM row.**
#:
#: ``p11-iem-loss-improves-iem-withdrawn`` is a contrast on **IEM**,
#: where **lower is better** and the scale is a scaled error. The ROPE
#: of +/- 0.0183 is a **PCC** width, derived from the probe's PCC seed
#: sd. **Applying it would be a different-quantities error** -- PLAN's
#: R2 shape, in the phase whose whole purpose is to say precisely what a
#: number means.
THE_IEM_ROW_EXCLUDED = {
    "excluded": "2026-09-02",
    "the_row": (
        "**p11-iem-loss-improves-iem-withdrawn**: 'the same arm is "
        "BETTER on the metric it optimises -- IEM 0.5825 against the "
        "control's 0.6130' -- paired d -0.0305, **lower is better**, "
        "condition 1 FALSE at 2 of 5 seeds"
    ),
    "why_the_rope_cannot_be_applied": (
        "**+/- 0.0183 is a PCC width**, derived from the probe's PCC "
        "seed sd through combined_claimable_delta. **IEM is a scaled "
        "error on a different scale with the opposite direction of "
        "good.** A region of practical equivalence in PCC units says "
        "nothing about an IEM difference -- **two quantities under one "
        "width**, which is the R2 error"
    ),
    "alternative_1_a_second_iem_derived_width_NOT_TAKEN": (
        "**it would break the fixed-domain-width ruling for exactly the "
        "reason per-contrast widths were refused** (THE_ROPE_RULED): "
        "the method assumes ONE domain-level width, and a second width "
        "makes the rows incomparable. Refusing per-contrast widths and "
        "then granting a per-metric one would be the same relaxation "
        "under a different name"
    ),
    "alternative_2_deriving_one_now_NOT_TAKEN": (
        "**no banked IEM seed-variance figure exists to derive a width "
        "from.** One would have to be produced now -- **a fresh "
        "derivation made for one row, at the moment that row was "
        "inconvenient to leave out.** That is a shape worth avoiding "
        "whatever the number came out at: a threshold derived to admit "
        "a specific case is not a threshold"
    ),
    "the_door_left_open": (
        "**with an INDEPENDENTLY DERIVED IEM width, declared before any "
        "posterior, the row could be restated in a later phase.** The "
        "paper's own footnote 3 supports domain-specific widths -- 'In "
        "classification 1% seems to be a reasonable choice. However, in "
        "other domains a different value could be more suitable.' "
        "**Excluded from THIS phase, not from the record**"
    ),
    "what_it_costs": (
        "**one contrast and ZERO declares.** The other p11 row "
        "(p11-iem-loss-costs-pcc-withdrawn) is on PCC, stays in scope, "
        "and reads the SAME two run directories through the same "
        "p11_paired.yaml -- so excluding the IEM row removes no input "
        "from the declare cycle. Measured, not assumed"
    ),
    "tag": "[RULED] -- excluded on units, with the door named",
}


#: **[SCOPED 2026-09-02; RE-SCOPED THE SAME DAY] Seventeen rows, 45
#: contrasts, 77 run directories.**
#:
#: Nineteen ledger rows are unresolved or withdrawn. **Two are
#: excluded** -- ``p8-framing-not-anatomy`` in principle, and the IEM
#: row on units (``THE_IEM_ROW_EXCLUDED``) -- leaving seventeen.
THE_ROWS_SCOPED = {
    "scoped": "2026-09-02, after the recovery cycle",
    "re_scoped": (
        "**[2026-09-02, same day] 18 -> 17 rows.** The original read "
        "'Eighteen rows... leaving eighteen'; the IEM exclusion "
        "followed. **Recomputed from the record rather than taken from "
        "the ruling message, and all three figures agreed**"
    ),
    "the_three_figures_recomputed": (
        "**17 rows, 45 contrasts, 77 distinct run directories.** Rows "
        "and contrasts differ because **rows aggregate contrasts** -- "
        "roadb-resolution-withdrawn alone is 17 ('the other 17 of 20 "
        "pairs') and p7c-nine-verdicts is 9. **The eighteen-posterior "
        "framing was wrong by a factor of two and a half**, and it is "
        "the same rows-versus-contrasts distinction the record already "
        "draws at 8-versus-12 and 19-versus-29"
    ),
    "the_run_directory_count_is_unchanged_by_the_exclusion": (
        "**77 before and 77 after.** The excluded IEM row shares "
        "p11_paired.yaml with the PCC row that stays, so it contributed "
        "no directory of its own. **Measured by differencing the two "
        "sets, not assumed** -- and 77 is above Phase 18's 68, the "
        "largest declare cycle so far"
    ),
    "an_earlier_count_of_91_was_wrong": (
        "**a first pass returned 91** by treating any `a__b__c` "
        "directory as a run. **Embedding artifacts share that shape** "
        "(`vit_b16__imagenet__g1`), so the filter was corrected to "
        "require `/runs/` in the path. Recorded because the wrong "
        "number would have over-planned the declare cycle by fourteen"
    ),
    "fifteen_with_recorded_run_dirs": (
        "fifteen of the nineteen carry `run_dirs` in the ledger and "
        "need no recovery"
    ),
    "three_recovered_by_reading_or_measurement_never_by_matching": {
        "p7c_nine_verdicts": (
            "**seven arms x five seeds at sha 26be5779**, READ from "
            "``configs/p7c_paired_selected30.yaml``, whose own header "
            "names the sha and states 'Phase 7C reported all nine "
            "verdicts from condition 2 alone. This computes condition "
            "1.' **Three p7c shas exist and the config named one** -- "
            "not chosen by plausibility. Its 35 declared inputs are the "
            "per-seed CSVs themselves"
        ),
        "roadb_region_crop": (
            "**three arms x five seeds at sha 52875413**, READ from "
            "``configs/roadb_p7c_paired_regioncrop.yaml``'s 15 declared "
            "inputs. **Three and not four because the fourth arm, "
            "anatomy_concat_srgnn, runs TEN seeds** -- "
            "roadb.REGION_CROP_ARMS_OBSERVED states 'the paired BCa "
            "decides; three contrasts are computable'. The VOID first "
            "five-arm set is untouched"
        ),
        "p7b_claimably_worse": (
            "**winner ``p7b_search__eb4477d9__p7b-search-3``, "
            "IDENTIFIED BY MEASUREMENT**: its own metrics.json carries "
            "`mean_delta -0.05428245764369638` while the other two "
            "search runs have no metrics at all. **Not matched by sha "
            "or suffix** -- the run was found by the figure it "
            "contains. Baseline: the probe's five seed CSVs, declared "
            "in ``p7b_search.yaml`` as `baseline_oof_seed_*` from "
            "``p7_d1_vit_b16_imagenet_g1__3f71a6a9__p7-d1-vit-imagenet``"
        ),
    },
    "one_excluded_in_principle_and_one_on_units": (
        "**TWO exclusions.** The IEM row on units "
        "(THE_IEM_ROW_EXCLUDED), and p8 in principle, below."
    ),
    "p8_excluded_in_principle": (
        "**p8-framing-not-anatomy. NOT restatable by a ROPE, and not "
        "because its runs are missing.** Four reasons, each "
        "sufficient: **no per-seed PCC vector exists** for it; **the "
        "ledger row carries no condition_1 or condition_2**; **the "
        "quantity is SPATIAL MASS over n = 15 patients** "
        "(gradcam_sheet.out_of_content, mass_median 0.4051 against "
        "expected_median 0.4386); and **it was REFUTED BY "
        "RE-MEASUREMENT**, not withdrawn on the criterion -- a stronger "
        "settlement than a restatement would offer. **Its runs "
        "surviving is beside the point**"
    ),
    "the_count": "19 - 2 = **17 rows in scope**, 45 contrasts",
    "tag": "[MEASURED] -- each recovery read or measured at source",
}


#: **[ESTABLISHED BY CONSTRUCTION 2026-09-02] The pairing precondition.**
#:
#: Benavoli's x_i is a difference between two arms **on the same fold**,
#: which requires identical fold assignment across the arms compared.
#: **It holds, and not by coincidence.**
PAIRING_ESTABLISHED_BY_CONSTRUCTION = {
    "established": "2026-09-02, read in code",
    "the_mechanism": (
        "**the fold assignment is READ FROM THE MANIFEST'S `fold` "
        "COLUMN, not generated per run.** "
        "``phase3.load_inputs`` line 214 **[CORRECTED 2026-09-02: this cited line 209. Phase 25 registered two backbones ABOVE it in the same file and the expression moved. Nothing about the mechanism changed -- **a record that cites a LINE NUMBER is invalidated by any edit above it**, and the suite now derives the number from the source rather than trusting it]**: `assignments = "
        "{int(r['patient_id']): int(r['fold']) for r in rows}`. The "
        "other CSV writers read the identical expression. **So two arms "
        "declaring the same `manifest_v1` hash have BYTE-IDENTICAL "
        "folds by construction** -- stronger than 'the generator is "
        "deterministic', because no arm generates anything"
    ),
    "qualification_1_ten_seed_arms": (
        "**arms at ten seeds pair only on the SHARED FIVE.** SEED_POOL "
        "is ten and the standing five are its first five, so the other "
        "five have no counterpart. The 64 arms in the shipped "
        "diagnostic config all declare exactly the five, so this bites "
        "less often than it might"
    ),
    "qualification_2_the_236_cohort": (
        "**236-cohort arms align on shared patients but one fold has a "
        "DIFFERENT MEMBER SET.** ``data/views.py`` carries the fold "
        "column VERBATIM rather than re-stratifying -- 'Re-running "
        "StratifiedKFold on 236 patients would reshuffle every "
        "assignment' -- so labels match, but the fold holding patient "
        "238 is one member short there. **A per-fold PCC across that "
        "boundary is over slightly different patient sets**, and "
        "whether that is still a paired observation is a judgement this "
        "record does not make"
    ),
    "qualification_3_partition_sensitivity": (
        "**``task_partition_sensitivity`` uses `assignments_override` "
        "and CANNOT be paired** against anything reading manifest "
        "folds. It is the only caller that does"
    ),
    "tag": "[MEASURED] -- read in code, with three qualifications named",
}


#: **[COMMITTED BEFORE ANY NUMBER 2026-09-02] The readings.**
READINGS_COMMITTED = {
    "committed": "2026-09-02, before any posterior is computed",
    "p_rope_above_0_95": (
        "**PRACTICALLY EQUIVALENT.** The two arms differ by less than "
        "the criterion can distinguish at five seeds. **This is a "
        "conclusion NHST CANNOT REACH** -- a failure to reject is not "
        "evidence of equivalence, and this is. It is the one thing this "
        "phase can add that the existing machinery cannot"
    ),
    "p_left_or_p_right_above_0_95": (
        "**PRACTICALLY DIFFERENT.** And **what it would mean is "
        "registered now rather than argued later**: a row WITHDRAWN "
        "under PLAN 4.3 would be called practically different by a "
        "Bayesian analysis **on the same data**. That is not a "
        "contradiction -- the two ask different questions, one about "
        "every seed's interval and one about a posterior over fold "
        "differences -- but it IS a disagreement between instruments, "
        "and it would have to be reported as one, with both verdicts "
        "standing and neither overturning the other. **The banked "
        "verdict does not change**"
    ),
    "none_above_0_95": (
        "**NO DECISION**, and **all three probabilities are reported "
        "regardless** -- the paper's own practice, whose Table 9 "
        "publishes triples for undecided comparisons. A row returning "
        "no decision is reported with its P(left), P(rope), P(right)"
    ),
    "the_registered_expectation": (
        "**most rows are expected to return NO DECISION.** With a "
        "strict +/- 0.0183 ROPE and only 25 observations the posterior "
        "will often be too wide to put 0.95 anywhere. **Their own "
        "figure: 22% of NHST's failed rejections cleared the ROPE at n "
        "= 100.** Ours is n = 25 with a tighter width, so a lower rate "
        "should be expected. **Registered so that a mostly-undecided "
        "outturn is the predicted result and not a disappointment "
        "reframed**"
    ),
    "no_decision_is_NOT_equivalence": (
        "**and the paper distinguishes them explicitly.** A WIDE "
        "posterior means UNCERTAINTY; a NARROW posterior sitting inside "
        "the ROPE means SIMILARITY. Reporting 'no decision' as 'the "
        "arms are equivalent' would be the exact error this phase "
        "exists to stop the record making -- **it would reintroduce, in "
        "Bayesian clothing, the confusion between 'not claimable' and "
        "'the same' that motivated the phase**"
    ),
    "tag": "[REGISTERED] -- readings before numbers",
}


#: **[DECLARED 2026-09-02] Normality: DESCRIBED, not tested, gating
#: nothing.**
NORMALITY_DESCRIBED_NOT_TESTED = {
    "declared": "2026-09-02",
    "what_will_be_reported": (
        "**the per-fold difference distributions alongside the "
        "results** -- mean, sd, min, max, and anything visibly "
        "pathological such as a fold with a degenerate PCC. So the "
        "record can say what the observations look like **where the "
        "paper assumed and never checked**"
    ),
    "it_gates_nothing": (
        "**no threshold, no test, no branch.** The analysis runs the "
        "same way whatever the description shows"
    ),
    "why_it_may_not_gate_anything": (
        "**branching on it after the fact would be choosing a method "
        "after seeing data.** If the distributions were allowed to "
        "select between the correlated t-test and something else, the "
        "choice would be made on the same numbers the conclusion is "
        "drawn from. **Describing is honest; branching is not**, and "
        "the difference is declared here rather than left to judgement "
        "on the day"
    ),
    "a_caution_worth_recording_in_advance": (
        "at roughly 47 patients per fold a fold-level PCC is a noisy "
        "statistic, and a degenerate fold is a live possibility rather "
        "than a hypothetical -- phase10_annex recorded 65.3% degenerate "
        "cells in a related setting. **Named in advance so its "
        "appearance is not a surprise that invites a mid-analysis "
        "decision**"
    ),
    "tag": "[REGISTERED] -- description, explicitly not a gate",
}


#: **[LOCKED 2026-09-02] Exit criteria.**
#:
#: the ruling was all eleven as drafted and the six settings as
#: corrected. ``EXIT_CRITERIA_DRAFT`` is preserved unchanged beside
#: this.
EXIT_CRITERIA = {
    "locked": "2026-09-02, all eleven as drafted, settings as corrected",
    "supersedes": "EXIT_CRITERIA_DRAFT, preserved unchanged",
    "scope": (
        "**17 rows, 45 contrasts, 77 run directories** "
        "(THE_ROWS_SCOPED). Rows aggregate contrasts; the two numbers "
        "are not interchangeable"
    ),
    "1_the_reckoning_leads_with_the_limit": (
        "PHASE_23_RECKONING states that the phase changes no figure and "
        "resolves nothing the cohort cannot resolve, with the bank's "
        "own words quoted, before any result is reported"
    ),
    "2_the_rope_is_fixed_before_any_posterior": (
        "+/- 0.018346, RECOMPUTED from sd 0.0148 and n = 5 at run time "
        "rather than pinned, ONE width for all rows. **If it moves for "
        "any reason after a posterior exists, EVERY RESULT IS "
        "WITHDRAWN.** The strongest protection on the width, and it is "
        "TESTED rather than merely stated: the task recomputes and "
        "refuses on mismatch"
    ),
    "3_every_parameter_is_attributed": (
        "**TWO ours** -- width 0.018346 and threshold P > 0.95 from the "
        "20:1 ratio. **FOUR source** -- rho = 0.2 with its "
        "approximation travelling, the prior, the posterior form, and "
        "P(rope) as integral, each citing a banked verbatim passage. A "
        "reader must see which is which without the paper"
    ),
    "4_the_pairing_qualifications_travel": (
        "any row involving a ten-seed arm, a 236-cohort arm, or "
        "partition-sensitivity carries the relevant qualification from "
        "PAIRING_ESTABLISHED_BY_CONSTRUCTION with its result"
    ),
    "5_all_three_probabilities_always": (
        "P(left), P(rope), P(right) for every contrast, decided or not, "
        "**and n with them** -- a contrast at n != 25 has different "
        "degrees of freedom and that must be visible, not buried"
    ),
    "6_no_decision_is_never_written_as_equivalence": (
        "checked against the write-up text, not only the metrics, and "
        "**encoded as a tested literal in the task**"
    ),
    "7_the_banked_verdicts_are_unchanged": (
        "**no ledger row is edited, no delta recomputed, no verdict "
        "overturned.** The task reads per-fold PCCs and produces "
        "posteriors; the ledger is untouched. A disagreement between "
        "instruments is reported as a disagreement"
    ),
    "8_seventeen_rows_and_no_more": (
        "p8-framing-not-anatomy stays excluded in principle and the IEM "
        "row on units; nothing is added once numbers exist"
    ),
    "9_normality_is_described_and_gates_nothing": (
        "per-contrast xbar, sigmahat, min, max and any degenerate fold "
        "reported alongside the result; no branch depends on them"
    ),
    "10_the_extensions_are_stated_wherever_results_are": (
        "PCC as an extension, n = 25 against their n = 100, rho's "
        "non-overlap approximation, and the hierarchical exclusion "
        "travel with the results rather than living only here"
    ),
    "11_nothing_added_after_the_lock": (
        "no row, no contrast, no width, no reading, no parameter "
        "introduced once numbers exist. Anything discovered afterwards "
        "is [POST-HOC] and fires nothing"
    ),
    "the_settings_as_ruled": {
        "ours_1_rope_half_width": (
            "0.018346 at full precision, recomputed not pinned "
            "[corrected 2026-09-02 from the rounded 0.0183]"
        ),
        "ours_2_decision_threshold": "P(.) > 0.95, from the ruled 20:1 ratio",
        "source_1_rho": "0.2, with the non-overlap approximation travelling",
        "source_2_prior": "mu_0 = 0, k_0 -> infinity, a = -1/2, b = 0",
        "source_3_posterior": "St(mu; n-1, xbar, (1/n + rho/(1-rho)) sigmahat^2)",
        "source_4_p_rope": "the integral of the posterior over the interval",
    },
    "tag": "[LOCKED]",
}


#: **[DRAFT 2026-09-02, NOT LOCKED] Exit criteria.**
EXIT_CRITERIA_DRAFT = {
    "drafted": "2026-09-02, NOT LOCKED -- proposed for the amendment",
    "1_the_reckoning_leads_with_the_limit": (
        "PHASE_23_RECKONING states that the phase changes no figure and "
        "resolves nothing the cohort cannot resolve, with the bank's "
        "own words quoted, before any result is reported"
    ),
    "2_the_rope_is_fixed_before_any_posterior": (
        "+/- 0.0183, recomputed from sd 0.0148 and n = 5 at run time "
        "rather than pinned, ONE width for all eighteen rows. **If it "
        "moves for any reason after a posterior exists, every result is "
        "withdrawn**"
    ),
    "3_every_parameter_is_attributed": (
        "prior, rho, decision rule and posterior form are reported as "
        "the SOURCE's; the ROPE width as THIS PROJECT's. A reader must "
        "be able to see which is which without consulting the paper"
    ),
    "4_the_pairing_qualifications_travel": (
        "any row involving a ten-seed arm, a 236-cohort arm, or "
        "partition-sensitivity carries the relevant qualification from "
        "PAIRING_ESTABLISHED_BY_CONSTRUCTION with its result"
    ),
    "5_all_three_probabilities_always": (
        "P(left), P(rope), P(right) for every row, decided or not"
    ),
    "6_no_decision_is_never_written_as_equivalence": (
        "checked against the write-up text, not only the metrics"
    ),
    "7_the_banked_verdicts_are_unchanged": (
        "**no ledger row is edited, no delta recomputed, no verdict "
        "overturned.** A disagreement between instruments is reported "
        "as a disagreement"
    ),
    "8_eighteen_rows_and_no_more": (
        "p8-framing-not-anatomy stays excluded; nothing is added once "
        "numbers exist"
    ),
    "9_normality_is_described_and_gates_nothing": (
        "the distributions are reported and no branch depends on them"
    ),
    "10_the_extensions_are_stated_wherever_results_are": (
        "PCC as an extension, n = 25 against their n = 100, and the "
        "hierarchical exclusion travel with the results rather than "
        "living only here"
    ),
    "11_nothing_added_after_the_lock": (
        "no row, no width, no reading, no parameter introduced once "
        "numbers exist. Anything discovered afterwards is [POST-HOC] "
        "and fires nothing"
    ),
    "tag": "[DRAFT] -- not locked, not binding until it is fixed",
}


#: **[ENUMERATED 2026-09-02] The settings a config would declare.**
SETTINGS_TO_DECLARE = {
    "enumerated": "2026-09-02, at the restate",
    "the_standing_clause": (
        "as phase17.DECLARED_SETTINGS_17: declared before the first "
        "run, **NEVER tuned across runs**, movement is a dated "
        "amendment with a reason"
    ),
    "1_rope_half_width": "**0.0183**, recomputed from sd and n, not pinned",
    "2_rho": "**0.2**, the fold ratio 1/k at k = 5",
    "3_decision_threshold": "**0.95**, the paper's, from its loss matrix",
    "4_observation_unit": (
        "**per-fold PCC differences**, 25 per contrast (5 folds x 5 "
        "seeds)"
    ),
    "5_the_row_list": (
        "the eighteen, by ledger id, with each row's arm run "
        "directories -- the three recovered ones included"
    ),
    "6_prior_parameters": (
        "mu_0 = 0, k_0 -> infinity, a = -1/2, b = 0. Written even "
        "though they are the paper's, so a reader need not infer them"
    ),
    "what_is_NOT_a_setting": (
        "the posterior form and the loss matrix -- both follow from the "
        "prior and the threshold, and a config field for either would "
        "invite editing a derivation"
    ),
    "tag": "[REGISTERED] -- six settings, one of them ours",
}


#: **[SWEPT 2026-09-02] Where else is a rounded display declared as the
#: quantity? One more, and it is NOT to be changed.**
ROUNDED_QUOTATION_SWEEP = {
    "swept": "2026-09-02, over every config field the repo can recompute",
    "the_method_and_its_limit": (
        "**only a quantity with a RECOMPUTATION PATH in this repo is "
        "checkable.** A declared number with no formula behind it "
        "cannot be swept at all, so this is a targeted check and not an "
        "exhaustive one -- stated so the clean result is not read as "
        "proof of absence"
    ),
    "p23_expect_rope_half_width": (
        "**declared 0.018346, recomputes to 0.018346 -- IDENTICAL.** "
        "Corrected this cycle; the guard is exact"
    ),
    "p22_tie_fraction_boundary": (
        "**declared 0.12425, and 0.2485 / 2 = 0.12425 -- IDENTICAL.** "
        "Its INPUT is a rounded fraction, which "
        "phase22.TIE_FRACTION_MEASUREMENT_DESIGNED already records as "
        "'the record holds the FRACTION ROUNDED TO FOUR PLACES'. No "
        "hidden precision to lose"
    ),
    "p22_se_diff_threshold_THE_ONE_FINDING": (
        "**declared 0.398942 in both R-clear configs; recomputes from "
        "sd_obs to 0.3989418.** A difference of 2e-7 -- **the same "
        "shape as the ROPE width: a rounded display declared as the "
        "value.** It is absorbed by the task's 5e-6 tolerance, which is "
        "why it never refused a launch"
    ),
    "and_it_is_NOT_being_changed": (
        "**the Phase 22 arms have already RUN against this config.** "
        "Editing a declared threshold now would change a config whose "
        "runs are banked, so the declared value and the runs would no "
        "longer describe each other. **A cosmetic precision fix is not "
        "worth breaking that correspondence**, and the 2e-7 changes no "
        "pair's membership -- separations are multiples of 0.2, so "
        "nothing sits within 2e-7 of the threshold. **Reported as a "
        "STANDING ITEM for the maintainer, not acted on**"
    ),
    "why_p23s_guard_is_exact_and_p22s_is_not": (
        "**p22's tolerance exists because its threshold is derived from "
        "sd_obs, itself a run-produced figure** -- so an exact "
        "comparison there would refuse on float noise between runs. "
        "**p23's width is derived from two CONSTANTS** (0.0148, 5), so "
        "there is no noise to accommodate and any slack is pure "
        "weakness. The two guards differ because the two quantities do"
    ),
    "tag": "[MEASURED] -- targeted sweep, one finding, not acted on",
}


#: **[OBSERVED 2026-09-02] Run ``p23_rope__9065e59e``, finalized,
#: single attempt. Thirty contrasts, all at n = 25, df 24.**
ROPE_OBSERVED = {
    "observed": "2026-09-02, run p23_rope__9065e59e, single attempt",
    "provenance": (
        "**read from the run's own outputs.** The run "
        "directory is CLUSTER-ONLY -- `runs/keeper` does not exist on "
        "the laptop -- so the figures could not be read from metrics.json "
        "here. **What WAS verified on this machine**: the verdict counts "
        "sum to thirty, the width recomputes to 0.018346, and every "
        "contrast key named below exists in the config's own contrast "
        "list. Stated rather than implied"
    ),
    "verdicts": {
        "practically equivalent": 0,
        "practically different": 2,
        "no decision": 28,
    },
    "every_contrast_at_the_full_form": (
        "**all thirty at n = 25, df 24.** No contrast reduced to shared "
        "seeds and none hit a differing cohort -- the reduction cases "
        "PAIRING_ESTABLISHED_BY_CONSTRUCTION registered did not arise "
        "in this subset"
    ),
    "the_parameters_as_run": (
        "**ROPE +/- 0.018346 RECOMPUTED and matching the declared "
        "value**, rho = 0.2, threshold P > 0.95. The attribution line "
        "(width and threshold OURS; rho, prior, posterior form and "
        "P(rope)-as-integral the SOURCE'S) and **rho's travelling "
        "approximation** are both in the run's own output, not only in "
        "this record"
    ),
    "tag": "[MEASURED]",
}


#: **[CONFIRMED 2026-09-02] The registered expectation fired.**
THE_EXPECTATION_HELD = {
    "confirmed": "2026-09-02",
    "what_was_registered": (
        "**before any number**: 'most rows are expected to return NO "
        "DECISION... their own figure: 22% of NHST's failed rejections "
        "cleared the ROPE at n = 100. Ours is n = 25 with a tighter "
        "width, so a lower rate should be expected' "
        "(READINGS_COMMITTED)"
    ),
    "what_landed": (
        "**28 of 30 no decision, and ZERO cleared the ROPE** -- a lower "
        "rate than Benavoli's 22%, as predicted. Theirs at n = 100 with "
        "a 1% width; ours at n = 25 with a width **tighter relative to "
        "the quantity** being measured"
    ),
    "why_this_matters_procedurally": (
        "**a mostly-undecided outturn is the PREDICTED RESULT, not a "
        "disappointment reframed.** The prediction was registered "
        "before the numbers and its direction was specific -- lower "
        "than 22%, with a reason. **A phase that predicts its own null "
        "and then gets it has learned something about its instrument**; "
        "one that discovers the null and then explains it has not"
    ),
    "tag": "[MEASURED] -- registered before, fired after",
}


#: **[THE PHASE'S ANSWER 2026-09-02] The withdrawn rows were genuinely
#: uncertain, not quietly equivalent.**
#:
#: This is what Phase 23 existed to establish, and it is established.
THE_SUBSTANTIVE_FINDING = {
    "found": "2026-09-02",
    "the_question_the_phase_existed_to_answer": (
        "**a withdrawn row says 'not claimable', which does not "
        "distinguish 'these two arms are the same' from 'we could not "
        "tell'.** NHST cannot separate them: the absence of a rejection "
        "is compatible with both. A ROPE can"
    ),
    "the_answer": (
        "**NOT ONE of the thirty contrasts is demonstrably the same.** "
        "No contrast places 95% of its posterior inside +/- 0.018346, "
        "and **the highest P(rope) anywhere is 0.3271** (srgnn imagenet "
        "512->768). Nothing is close"
    ),
    "what_it_establishes": (
        "**the 0.04-0.10 unresolvable band is a region of IGNORANCE, "
        "not a region of SAMENESS.** The withdrawn rows were genuinely "
        "uncertain. **This is a statement the project could not "
        "previously make** -- the absence of a rejection could not tell "
        "the two apart, and now a posterior can"
    ),
    "what_it_does_NOT_establish": (
        "**no contrast is shown to be NON-equivalent either, except the "
        "two decided ones.** Twenty-eight contrasts remain undecided in "
        "both directions. 'Not demonstrably the same' is not 'shown to "
        "differ', and writing it as the latter would be the error this "
        "phase exists to prevent"
    ),
    "EXTENDED_2026_09_02_to_the_forty_five": (
        "**[The paragraphs above are the THIRTY's and stand as "
        "written.]** On all forty-five: **still not one contrast is "
        "demonstrably the same**, and the highest P(rope) anywhere is "
        "**0.3452** (p7c ``region_awareness_photometric``), which "
        "replaces 0.3271 as the maximum over the larger family. "
        "**Twenty-eight undecided becomes forty-one.** The finding does "
        "not merely survive the fifteen -- it is now the family's "
        "statement rather than a subset's "
        "(THE_FORTY_FIVE_OUTTURN)"
    ),
    "the_shape_of_the_contribution": (
        "**a negative that was previously unavailable rather than "
        "unstated.** The record could always say 'not claimable'; it "
        "could not say 'and not equivalent either'. It can now, for "
        "these thirty"
    ),
    "tag": "[MEASURED]",
}


#: **[OBSERVED 2026-09-02] The two decisions, and the two near-misses.**
THE_TWO_DECISIONS = {
    "observed": "2026-09-02",
    "1_vit_b16_imagenet_224_to_768": {
        "key": (
            "roadb-resolution-withdrawn__resolution__vit_b16__imagenet__"
            "224_to_768"
        ),
        "p_right": 0.9993,
        "verdict": "practically different",
        "what_it_means": (
            "**THE REGISTERED INSTRUMENTS-DISAGREE CASE.** Withdrawn "
            "under the paired criterion; practically different under "
            "the Bayesian analysis **on the same data**. "
            "READINGS_COMMITTED registered this outcome before any "
            "number: **both verdicts stand, the banked verdict is "
            "UNCHANGED, and the disagreement IS the finding** -- the "
            "two ask different questions, one about every seed's "
            "interval and one about a posterior over fold differences"
        ),
        "it_aligns_with_what_the_record_holds": (
            "the frozen ViT falling off a cliff on leaving 224 is "
            "already in the record; this contrast is that effect, and "
            "the Bayesian analysis resolves what the seed-wise "
            "criterion could not"
        ),
    },
    "2_p10_replication_vs_probe": {
        "key": (
            "p10-replication-vs-best-arm-withdrawn__p10__"
            "replication_vs_0p2520"
        ),
        "p_left": 0.9817,
        "verdict": "practically different",
        "what_it_means": (
            "**CleftGNN's replication is practically WORSE than the "
            "probe, decided.** Withdrawn under the criterion at 3 of 5 "
            "seeds. The direction is recorded because P(left) is a "
            "signed statement and 'different' alone would lose it"
        ),
    },
    "the_two_near_misses_as_observations": (
        "**anatomy_vs_random P(left) 0.9328** and **swin_b imagenet "
        "512->768 P(right) 0.9132** -- decided at 0.90, not at the "
        "ruled 0.95. **Reported because the threshold is OURS**: a "
        "reader should be able to see what a more lenient loss ratio "
        "would have changed, which is two more decisions. Neither is a "
        "verdict; both are observations beside one"
    ),
    "tag": "[MEASURED]",
}


#: **[LIMITATION 2026-09-02] Thirty of forty-five, and the reason is the
#: RECORD'S ENUMERABILITY rather than the method.**
THE_FIFTEEN_NOT_ENUMERABLE = {
    "recorded": "2026-09-02",
    "the_scope_run": (
        "**thirty of the forty-five contrasts, across seventeen rows.** "
        "The IEM row is excluded on units (THE_IEM_ROW_EXCLUDED); the "
        "other fifteen could not be assembled"
    ),
    "the_fifteen_by_row": {
        "p7c-nine-verdicts": (
            "**9.** phase7c.paired_matrix COMPUTES the matrix from "
            "loaded predictions; **which nine of the twenty-one "
            "possible pairs are the verdicts is enumerated nowhere**"
        ),
        "p17": "**4.** phase17 exposes no pair list",
        "p16-anchor-loop-unresolved": (
            "**1.** its pair is named in prose, not in a readable list"
        ),
        "p7b-claimably-worse": (
            "**1.** likewise, and its winner is a search run's own "
            "output rather than a declared arm"
        ),
    },
    "CORRECTED_2026_09_02_thirteen_WERE_enumerable": (
        "**[CORRECTION. The claim above is FALSE for thirteen of the "
        "fifteen, and it is preserved as written.]** "
        "**p7c's nine ARE constructed by readable code**: "
        "``paired_matrix``'s own docstring says 'All nine comparisons. "
        "Six against identity and the three declared between-arm "
        "pairs' -- six against ``IDENTITY_ARM`` plus the three "
        "``comparisons()`` returns, **not a filter over twenty-one**. "
        "**p17's four are a LITERAL TUPLE** in "
        "``run.task_tstr_family_analysis``. **p16's one is stated "
        "structurally** in ``phase16.PRIMARY_CONTRAST_REGISTERED``, "
        "naming both arms. **Only p7b's winner was genuinely "
        "unrecorded**, and it was recovered by MEASUREMENT -- its own "
        "metrics.json -- rather than by reading"
    ),
    "the_provenance_of_the_wrong_claim": (
        "**the record's, and the mechanism is specific: stopping at "
        "'computes from loaded predictions' without reading the "
        "docstring ONE LINE ABOVE that answers the question.** The "
        "sentence that made the recovery obvious was already in the "
        "file being quoted from. **Same family as the five unverified "
        "attributions**: a claim about a source made without reading "
        "far enough into it"
    ),
    "what_the_correction_costs": (
        "**nothing measured** -- the thirty that ran are unaffected and "
        "recompute identically inside the forty-five. What it cost was "
        "one cycle, and a limitation recorded that was not one"
    ),
    "it_is_a_limitation_of_the_record_not_the_method": (
        "**the ROPE applies perfectly well to all fifteen.** What is "
        "missing is a machine-readable statement of WHICH ARMS each "
        "contrast pairs. The method is not the constraint; the "
        "record's enumerability is"
    ),
    "they_remain_recoverable": (
        "**if their pairs can be read from RUN ARTIFACTS rather than "
        "from code** -- a paired run's own metrics.json names the "
        "contrasts it computed -- the fifteen return to scope without "
        "any change to this phase's method or width. **Deferred, not "
        "abandoned**"
    ),
    "tag": "[MEASURED] -- the boundary is the record's, not the test's",
}


#: **[DESCRIBED 2026-09-02, GATING NOTHING] The observations.**
NORMALITY_AS_OBSERVED = {
    "described": "2026-09-02, per criterion 9",
    "what_the_run_reports": (
        "**per contrast: n, mean, sd, min, max and the count of "
        "degenerate folds dropped**, alongside every result. All thirty "
        "at n = 25 with no fold dropped -- **no degenerate fold was "
        "encountered**, though the risk was named in advance "
        "(phase10_annex's 65.3% degenerate cells in a related setting)"
    ),
    "no_branch_depended_on_them": (
        "**criterion 9 held.** The description is reported and nothing "
        "read it: the same posterior, width and threshold were applied "
        "to every contrast whatever its shape. **Branching on it would "
        "have been choosing a method after seeing data**, and the "
        "record said so before the run"
    ),
    "what_the_record_can_now_say": (
        "the paper assumes multivariate normality and never checks it; "
        "**this record can say what ITS observations looked like**, "
        "which is more than the source does, without having tested "
        "anything"
    ),
    "tag": "[MEASURED] -- described, explicitly not a gate",
}


#: **[CLOSED 2026-09-02] PHASE_23_CLOSING.**
PHASE_23_CLOSING = {
    "closed": (
        "2026-09-02, run p23_rope__9065e59e (thirty contrasts); "
        "**COMPLETED 2026-09-02 by run p23_rope__8bbb9a1b__p23-rope-45**, "
        "which REPLACES it -- the thirty recomputed identically "
        "inside the forty-five. **[COMPLETE 2026-09-02 on the RERUN "
        "p23_rope__402a8477__p23-rope-45-2, which is THE CITABLE "
        "RUN and supersedes the first forty-five FOR OBSERVATIONS "
        "ONLY -- its numbers were correct and are unchanged; "
        "THE_RERUN_IS_THE_CITABLE_RUN.]**"
    ),
    "EXTENDED_2026_09_02_the_finding_on_forty_five": (
        "**0 equivalent, 4 practically different, 41 no decision.** The "
        "finding is unchanged and now rests on the whole family: not "
        "one of forty-five contrasts is demonstrably the same, highest "
        "P(rope) 0.3452. **Two further decisions**, both p17 and both "
        "at n=24: ``p17-c-vs-probe`` P(left) 0.9990, and "
        "``p17-a-vs-c`` P(right) 0.9910, **which decides under this "
        "instrument the scheme attribution Phase 17 declined to claim** "
        "-- an instruments-disagree case, Phase 17's verdict standing "
        "unchanged (THE_TWO_FURTHER_DECISIONS). **The thirty recomputed "
        "identically, both original decisions to four decimals.** The "
        "observations on the three n=24 contrasts are being refilled by "
        "a rerun (THE_DROP_WAS_COUNTED_NOT_RECORDED)"
    ),
    "COMPLETED_2026_09_02_the_rerun_landed": (
        "**the record is complete.** "
        "``p23_rope__402a8477__p23-rope-45-2`` reproduced every verdict "
        "and all four decisions to four decimals, and the one "
        "degenerate fold now carries its arm, seed, fold and cause: "
        "``p17-c-vs-probe``, seed 1337, fold 1, ``p17_arm_c``, constant "
        "prediction across 47 patients. **24 + 1 + 0 = 25.** "
        "THE_RERUN_IS_THE_CITABLE_RUN, THE_DEGENERATE_FOLD"
    ),
    "THE_PHASE_IS_COMPLETE_AND_WHAT_STAYS_OPEN": (
        "**complete, and complete is not comprehensive.** Criterion 8 "
        "is walked at **45 of 45 contrasts across seventeen rows**, and "
        "**three reductions stand unchanged and unaddressed**: the "
        "**split-randomised half is NOTED, NOT COMMITTED and UNBUILT** "
        "(SPLIT_RANDOMISED_HALF_NOT_BUILT); the **IEM row stays "
        "EXCLUDED on units** (THE_IEM_ROW_EXCLUDED); and the "
        "**five-seeds-on-a-FIXED-SPLIT limitation stands -- this phase "
        "did not address it and could not have**, since every arm it "
        "read was trained on the one split ``data/folds.py`` produces "
        "with ``shuffle=False``. **A posterior over fold differences "
        "says nothing about a split it never varied**"
    ),
    "the_finding": (
        "**THE WITHDRAWN ROWS WERE GENUINELY UNCERTAIN, NOT QUIETLY "
        "EQUIVALENT.** Of thirty contrasts, none places 95% of its "
        "posterior inside the ROPE and the highest P(rope) is 0.3271. "
        "**The 0.04-0.10 band is a region of ignorance, not of "
        "sameness** -- a statement this project could not previously "
        "make. Two contrasts are decided as practically DIFFERENT, one "
        "of them disagreeing with its own banked verdict, which was a "
        "registered possibility and is reported as a disagreement"
    ),
    "and_what_it_did_not_do": (
        "**it changed no figure and resolved nothing the cohort cannot "
        "resolve**, exactly as the reckoning said before any result "
        "existed. Twenty-eight contrasts remain undecided in both "
        "directions"
    ),
    "criterion_walk": {
        "1_reckoning_leads_with_the_limit": (
            "MET. The reckoning stated 'changes no figure, resolves "
            "nothing' before any result, with the bank's words quoted"
        ),
        "2_the_rope_is_fixed_before_any_posterior": (
            "**MET, AND THE GUARD FIRED IN REAL USE.** It refused the "
            "first launch: the width recomputed to 0.0183460 against a "
            "declared 0.0183000, a rounded display quoted as the "
            "quantity. **Caught BEFORE any posterior existed, so "
            "nothing had to be withdrawn.** Recorded as the criterion "
            "WORKING, not as an incident -- the withdrawal clause never "
            "had to be invoked because the guard made it unnecessary"
        ),
        "3_every_parameter_is_attributed": (
            "MET. The attribution line is in the run's own output: two "
            "ours, four source, with rho's approximation travelling"
        ),
        "4_the_pairing_qualifications_travel": (
            "MET, and **vacuous in this subset**: all thirty ran at n = "
            "25, so no ten-seed or 236-cohort reduction arose. The "
            "criterion is untested rather than unmet. "
            "**[UPDATED 2026-09-02 on the forty-five: NO LONGER "
            "VACUOUS, AND NOT MET ON ITS FIRST LIVE TEST.** Three "
            "contrasts ran at n=24 and the qualification did NOT "
            "travel: the reduced n reached metrics.json with no reason "
            "beside it. The criterion is the one that caught its own "
            "gap. The fix is built and the criterion is **PENDING THE "
            "RERUN**, which is the maintainer's; "
            "THE_DROP_WAS_COUNTED_NOT_RECORDED.]** "
            "**[UPDATED AGAIN 2026-09-02 on the rerun "
            "p23_rope__402a8477__p23-rope-45-2: MET.** The "
            "qualification travels: the one reduced n carries its arm, "
            "seed, fold and cause, and the accounting closes at 24 + 1 "
            "+ 0 = 25. **THREE STATES, and all three stand as "
            "written**: VACUOUS at thirty because every contrast ran at "
            "n=25 and the criterion had no opportunity to fail; NOT MET "
            "at forty-five, its first live test, which it failed; MET "
            "on the rerun. **A criterion that was never exercised is "
            "not a criterion that passed**, and the middle state is the "
            "one that earned the fix; THE_DEGENERATE_FOLD.]**"
        ),
        "5_all_three_probabilities_always": (
            "MET. P(left), P(rope), P(right) and n for every contrast, "
            "decided or not"
        ),
        "6_no_decision_is_never_written_as_equivalence": (
            "**MET, and with no opportunity to fail**: zero contrasts "
            "were equivalent, so nothing could be misreported as one. "
            "**The guard is tested regardless** -- verdict() is swept "
            "across P(rope) up to and including 0.95 and never returns "
            "equivalence. A criterion that holds only because the case "
            "did not arise is worth marking as such"
        ),
        "7_the_banked_verdicts_are_unchanged": (
            "**MET. No ledger row edited, no delta recomputed, no "
            "verdict overturned; the ledger stands at 38.** The one "
            "contrast disagreeing with its banked verdict is reported "
            "as a disagreement, with both standing"
        ),
        "8_seventeen_rows_and_no_more": (
            "**COMPLETE: 45 of 45 contrasts across all SEVENTEEN rows** "
            "(run p23_rope__8bbb9a1b__p23-rope-45, 2026-09-02). The "
            "scope criterion is fully walked, and **the two standing "
            "reductions are unchanged**: the split-randomised half is "
            "NOTED, NOT COMMITTED (SPLIT_RANDOMISED_HALF_NOT_BUILT) and "
            "the IEM row stays EXCLUDED on units "
            "(THE_IEM_ROW_EXCLUDED). "
            "**[The paragraph below is the first run's and stands as "
            "written.]** **PARTIAL AT THE FIRST RUN, and completed "
            "since.** "
            "Seventeen rows were in scope; **thirty of forty-five "
            "contrasts ran**. **[CORRECTED 2026-09-02: the fifteen were "
            "described as 'not enumerable from any readable record'. "
            "That was FALSE for thirteen of them** -- p7c's nine are "
            "constructed by paired_matrix, p17's four are a literal "
            "tuple, p16's one is stated structurally. Only p7b's winner "
            "was genuinely unrecorded. The completion to forty-five is "
            "built and awaiting one declare; see "
            "THE_COMPLETION_TO_FORTY_FIVE.]**"
        ),
        "9_normality_is_described_and_gates_nothing": (
            "MET. Descriptions reported, no degenerate fold "
            "encountered, no branch depended on them"
        ),
        "10_the_extensions_are_stated_wherever_results_are": (
            "MET. PCC as an extension, n = 25 against their n = 100, "
            "rho's non-overlap approximation and the hierarchical "
            "exclusion all travel with the results"
        ),
        "11_nothing_added_after_the_lock": (
            "MET. No row, contrast, width, reading or parameter was "
            "introduced once numbers existed"
        ),
    },
    "the_half_that_stays_unbuilt": (
        "**the split-randomised P(A>B) remains NOTED-NOT-COMMITTED and "
        "unbuilt** (SPLIT_RANDOMISED_HALF_NOT_BUILT). Split "
        "randomisation is unreachable by parameter, would need a new "
        "splitter in frozen apparatus, and costs 1,972 runs for the "
        "locked ladder. **The five-seeds-on-a-fixed-split limitation "
        "therefore stands**, and this phase did not address it"
    ),
    "no_ledger_row": (
        "**this phase RESTATES; it does not CLAIM.** No row is "
        "registered, and the ledger is unchanged at 38 entries"
    ),
    "what_the_phase_cost": (
        "**one refused launch and five record corrections found in "
        "preparation** -- the 17 never counted, the 0.068 with no "
        "source, five unverified source attributions, a rounded width, "
        "and a declared field that reached no code. **Four were caught "
        "by tests or guards this phase wrote; one by a sweep it ran.** "
        "Recorded because a phase's cost is part of its result. "
        "**[EXTENDED 2026-09-02: SEVEN, and the two added were found "
        "AFTER results existed** -- the null observations on three "
        "n=24 contrasts (THE_DROP_WAS_COUNTED_NOT_RECORDED), found by "
        "reading the run's own metrics.json rather than by any test, "
        "and the record probing ``dropped`` for a field named "
        "``dropped_folds``. **The first five were caught in "
        "preparation; these two were not, and that is the difference "
        "worth recording.]**"
    ),
    "tag": (
        "[CLOSED, COMPLETE] -- **[2026-09-02: was [CLOSED, PARTIAL] "
        "while thirty of forty-five contrasts had run. 45 of 45 across "
        "seventeen rows on p23_rope__402a8477__p23-rope-45-2. Complete "
        "on its OWN scope; the three standing reductions are unchanged "
        "and named in THE_PHASE_IS_COMPLETE_AND_WHAT_STAYS_OPEN.]**"
    ),
}


#: **[BUILT 2026-09-02, NOT RUN] The completion to forty-five.**
#:
#: The fifteen were recoverable after all -- thirteen by reading, one by
#: measurement, and one (p16's) whose directory was declared in other
#: configs all along. **Completion, not addition**, and criterion 11 is
#: the point on which that stands or falls.
THE_COMPLETION_TO_FORTY_FIVE = {
    "built": "2026-09-02, config regenerated; NOT run",
    "the_criterion_11_grounds": (
        "**criterion 11 forbids introducing a contrast 'once numbers "
        "exist'. The fifteen are FULLY DETERMINED, NOT CHOSEN** -- they "
        "are exactly the complement of the thirty within the locked "
        "forty-five, so **no judgement can enter which ones get "
        "added**, and the fishing shape the criterion exists to prevent "
        "is unavailable. **Completion, not addition**"
    ),
    "the_condition_stated_explicitly": (
        "**had ANY of the fifteen required a decision -- which pairs, "
        "which run, which arms -- criterion 11 WOULD bite**, because "
        "that decision would be made with thirty results already "
        "visible. **None does.** p7c's nine are built by construction "
        "from paired_matrix's own rule; p17's four are a literal tuple; "
        "p16's is stated structurally; p7b's winner was fixed by "
        "measurement BEFORE this cycle. The condition is recorded so a "
        "later addition cannot borrow this precedent without meeting it"
    ),
    "criterion_8s_row_count_corrected": (
        "**[CORRECTED 2026-09-02] it was cited criterion 8 as "
        "'eighteen rows'. It says SEVENTEEN.** Verbatim: "
        "'p8-framing-not-anatomy stays excluded in principle and the "
        "IEM row on units; nothing is added once numbers exist.' The "
        "count is in the criterion's own name, "
        "`8_seventeen_rows_and_no_more`"
    ),
    "the_fifteen_and_their_sources": {
        "p7c_nine": (
            "**BY CONSTRUCTION**, never a literal: six against "
            "``phase7c.IDENTITY_ARM`` plus the three "
            "``phase7c.comparisons()`` returns. The generator REFUSES "
            "if the construction does not yield exactly nine"
        ),
        "p17_four": (
            "the literal family tuple in "
            "``run.task_tstr_family_analysis``: primary_a_vs_probe, "
            "primary_c_vs_probe, secondary_a_vs_c, secondary_b_vs_c. "
            "**The fifth member, primary_b_vs_probe, is correctly "
            "EXCLUDED** -- it is ledger row 33, status CLAIMABLE, "
            "outside the unresolved/withdrawn nineteen. **Its exclusion "
            "confirms the mapping rather than leaving it inferred**"
        ),
        "p16_one": (
            "``phase16.PRIMARY_CONTRAST_REGISTERED``, both arms named: "
            "'anchor-loop OOF predictions vs the 0.2520 probe "
            "(p7_d1_vit_b16_imagenet_g1)'"
        ),
        "p7b_one": (
            "winner ``p7b_search__eb4477d9__p7b-search-3``, identified "
            "by MEASUREMENT last cycle; baseline the probe. **The only "
            "one not recoverable by reading**"
        ),
    },
    "the_p7c_naming_reconciliation": (
        "**a mismatch here MISSES SILENTLY rather than failing.** "
        "``IDENTITY_ARM`` is the short key ``0_identity`` while "
        "``comparisons()`` returns full names like "
        "``p7c_2_geometric``, and the run-directory stems carry the "
        "``p7c_`` prefix. The generator reconciles once and **raises if "
        "the identity arm is not among the arm list** -- because a "
        "silent miss would drop six of the nine and still produce a "
        "plausible-looking config"
    ),
    "the_config_delta_measured": (
        "**45 contrasts, 46 run directories, +12 -- not +11.** The "
        "twelfth is ``p16_anchor_loop__f342fed9__p16-anchor-loop``: it "
        "is declared in p18_metric_space and both p21 configs, but was "
        "NOT in this config's thirty-four. **An earlier delta report "
        "measured p16 against p16_anchor_loop.yaml's own inputs, which "
        "declare only the probe, and reported '0 new'.** Corrected by "
        "differencing the actual directory sets"
    ),
    "one_placeholder_only": (
        "**``p7b_search`` is the single all-zero hash**, because a "
        "run's own output is never one of its inputs and nothing has "
        "ever declared it. Every other entry carries a hash that "
        "already exists in a shipped config, asserted byte-identical"
    ),
    "the_rerun_replaces_rather_than_supplements": (
        "**the forty-five REPLACE the thirty; they do not add to "
        "them.** The thirty are recomputed identically inside the "
        "forty-five, **which is itself a check**: same arms, same "
        "width, same rho, same threshold, so any drift in the thirty "
        "would mean something changed that should not have"
    ),
    "tag": "[BUILT] -- not run; the declare and the launch are the maintainer's",
}


#: **[OBSERVED 2026-09-02] The forty-five ran. The outturn.**
THE_FORTY_FIVE_OUTTURN = {
    "run": "p23_rope__8bbb9a1b__p23-rope-45",
    "observed": "2026-09-02",
    "the_counts": (
        "**0 practically equivalent, 4 practically different, 41 no "
        "decision**, over forty-five contrasts across seventeen rows. "
        "The thirty-contrast outturn was 0 / 2 / 28; the fifteen added "
        "two decisions and thirteen non-decisions"
    ),
    "the_ns": (
        "**42 contrasts at n=25 and THREE at n=24** -- every p17 "
        "contrast involving arm C. This is the first time criterion 4's "
        "pairing qualification was live rather than vacuous, and **it "
        "exposed a defect on its first use** "
        "(THE_DROP_WAS_COUNTED_NOT_RECORDED)"
    ),
    "the_highest_p_rope_anywhere": (
        "**0.3452**, p7c ``region_awareness_photometric``. It replaces "
        "the thirty's 0.3271 as the maximum over the larger family, and "
        "**it is still nowhere near 0.95**. Fifteen more contrasts "
        "raised the ceiling by 0.018 and changed nothing about the "
        "conclusion"
    ),
    "the_thirty_recomputed_IDENTICALLY": (
        "**a free check, and it passed.** The forty-five REPLACE the "
        "thirty rather than adding to them, so all thirty were "
        "recomputed from the same arms at the same width, rho and "
        "threshold. **Every one reproduced, including both original "
        "decisions to four decimals.** Nothing drifted between the two "
        "runs, which is what a completion should look like and is not "
        "guaranteed merely by intending it"
    ),
    "what_the_check_would_have_caught": (
        "any change to the width, rho, threshold, arm resolution or "
        "pairing rule between the two runs. It cost nothing to have -- "
        "the thirty had to be recomputed anyway -- and it is banked as "
        "a check rather than as a coincidence"
    ),
    "tag": "[MEASURED]",
}


#: **[DEFECT 2026-09-02] The drop was COUNTED and never RECORDED.**
#:
#: Three contrasts ran at ``n=24`` and the record said only that. The
#: fields a reader would look in -- ``observations.n_dropped``,
#: ``observations.dropped_folds`` -- **did not exist under those names
#: at all**, so both read as null.
THE_DROP_WAS_COUNTED_NOT_RECORDED = {
    "found": "2026-09-02, by reading the forty-five run's metrics.json",
    "the_symptom": (
        "**a reduced n with no reason beside it.** Three of forty-five "
        "contrasts at n=24, and **two of the phase's four decisions "
        "rest on those vectors** (``p17-c-vs-probe``, ``p17-a-vs-c``). "
        "A reader could see that a fold was missing and had no way to "
        "learn which arm, which seed, which fold, or why"
    ),
    "the_mechanism_never_populated": (
        "**NEVER POPULATED, and the sharper statement is that the two "
        "fields never existed.** ``rope.describe`` emitted exactly one "
        "drop field, ``degenerate_folds_dropped``, an ``int``. "
        "``n_dropped`` and ``dropped_folds`` were never keys, so "
        "reading them returned null -- not a null that was written, a "
        "null from an absent key. **Not overwritten and not "
        "path-dependent**: there is one call site and it always passed "
        "the count"
    ),
    "where_the_drop_HAPPENS": (
        "``rope.per_fold_pcc`` returns NaN for a fold whose truth or "
        "prediction is constant -- deliberately, so a degenerate fold "
        "cannot be absorbed into a correlation of zero. "
        "``paired_fold_differences`` then skips any cell that is NaN on "
        "either side and does ``dropped += 1``. **The cause is known "
        "only inside ``per_fold_pcc``, which discards it in the same "
        "expression that detects it**, and by the time the count "
        "reaches the record there is nothing left to name"
    ),
    "the_SECOND_mechanism_that_was_invisible": (
        "**a short n had TWO possible causes and the record could "
        "distinguish neither.** ``paired_fold_differences`` walked the "
        "INTERSECTION of seeds and folds, so a fold present in one arm "
        "and absent from the other was **excluded silently and counted "
        "nowhere** -- it did not even increment the count. n=24 was "
        "therefore consistent with a degeneracy (counted) or an absence "
        "(not counted), and **which one it is cannot be settled from "
        "the shipped metrics.json**. The rerun settles it"
    ),
    "the_fix": (
        "**``per_fold_detail`` records the cause where it is known** -- "
        "per fold: ``pcc``, ``n_patients``, ``truth_sd``, "
        "``prediction_sd`` and ``undefined_because`` (one of constant "
        "truth / constant prediction / both / fewer than two patients). "
        "``per_fold_pcc`` now READS it, so the number and the cause "
        "cannot disagree about which folds are degenerate. "
        "``paired_fold_report`` walks the UNION and returns "
        "``dropped`` and ``unpaired`` as RECORDS -- arm, seed, fold, "
        "reason, and the two standard deviations that establish it -- "
        "and ``describe`` carries both lists plus ``n_dropped`` and "
        "``n_unpaired`` into metrics.json"
    ),
    "the_refusals_that_replace_the_silence": (
        "**two, because a null must not be reachable.** (i) A "
        "degenerate cell whose cause is not in the detail RAISES rather "
        "than being reported with a null reason. (ii) "
        "``run.p23_contrast_record`` requires ``n + n_dropped + "
        "n_unpaired == full_n`` and refuses otherwise, naming each "
        "arm's absent seeds -- **so a reduced n that the record has no "
        "words for stops the run instead of being written**"
    ),
    "no_default_on_either_account": (
        "``describe(x, dropped, unpaired)`` takes both as REQUIRED "
        "arguments. The caller that wrote nulls did so by having "
        "nothing to pass; a default would let the next one do it "
        "silently again. Test-enforced, as the ruled parameters are"
    ),
    "what_it_costs_the_numbers": (
        "**nothing.** No posterior, probability or verdict changes -- "
        "the pairing rule is unchanged for every cell that was already "
        "paired, and ``paired_fold_differences`` is retained and "
        "asserted to agree with the reporting form. **Only the "
        "observations change**"
    ),
    "CORRECTED_2026_09_02_the_field_is_dropped_folds_not_dropped": (
        "**[A LINE FOR THE READER WHO SEARCHES FOR THE WRONG KEY.]** "
        "The metrics.json field is **``observations.dropped_folds``** "
        "(with ``dropped_folds``'s count in ``n_dropped``, and "
        "``unpaired_folds`` / ``n_unpaired`` beside them). **``dropped`` "
        "is the ARGUMENT NAME of ``rope.describe``, not a key in the "
        "record**, and the probe used ``dropped`` and read the list as "
        "ABSENT when it was present. **Same class as the defect it was "
        "probing** -- a claim about what a record holds, made without "
        "checking the name it holds it under -- and small enough to be "
        "worth writing down precisely because it cost only a moment"
    ),
    "the_error_class": (
        "**code -> record, the third instance.** A value is computed, "
        "is correct, and does not reach the record. Its siblings: "
        "``pair_source`` reaching only a log line (Phase 22), the null "
        "``condition_1``/``condition_2`` computed and discarded (Phase "
        "22 close-out). **The distinguishing feature here is that the "
        "field APPEARED to arrive** -- ``observations`` was in "
        "metrics.json, empty of the thing that mattered"
    ),
    "tag": "[FIXED, NOT RERUN] -- the rerun is the maintainer's",
}


#: **[DEFECT 2026-09-02] The sweep asserted PRESENCE, not CONTENT.**
THE_SWEEP_CHECKED_PRESENCE_NOT_CONTENT = {
    "found": "2026-09-02",
    "what_the_sweep_asserted": (
        "``test_everything_the_additions_compute_reaches_the_record`` "
        "read ``task_p23_rope``'s SOURCE and asserted that "
        "``\"observations\"`` appears among the per-contrast fields. "
        "**It does. It arrived carrying nulls**"
    ),
    "why_that_is_not_enough": (
        "**reaching the record is not carrying the value.** A "
        "source-text check can only see that a key is written; it "
        "cannot see what the key holds, because it never runs the "
        "code. Every field the defect concerned passed the sweep"
    ),
    "why_it_was_written_that_way": (
        "**because the task could not be executed without a cluster.** "
        "``task_p23_rope`` needs a ctx, declared inputs and real "
        "prediction CSVs, none of which exist on the laptop, so the "
        "sweep read the source instead. **The constraint was real; the "
        "conclusion drawn from it was too weak** -- the answer was to "
        "make the assembly executable, not to settle for reading it"
    ),
    "the_fix": (
        "**``run.p23_contrast_record`` is extracted to module level** "
        "and takes two arms' per-fold structures rather than a ctx, so "
        "the sweep now BUILDS a fixture with a known degenerate fold, "
        "RUNS the assembly, and asserts what each field carries: no "
        "value null, the drop record naming arm, seed, fold and cause, "
        "and the refusal firing on an unaccounted shortfall"
    ),
    "the_rule_this_leaves": (
        "**a code -> record sweep must execute the assembly, not read "
        "it.** Presence of a key is a weaker claim than presence of its "
        "content, and the two are worth different amounts. Where "
        "execution needs a cluster, the assembly is extracted until it "
        "does not"
    ),
    "tag": "[FIXED]",
}


#: **[OBSERVED 2026-09-02] The two further decisions, both from p17.**
THE_TWO_FURTHER_DECISIONS = {
    "observed": "2026-09-02, run p23_rope__8bbb9a1b__p23-rope-45",
    "both_rest_on_n_24_vectors": (
        "**both involve arm C, and both therefore ran at n=24.** The "
        "decisions are reported with that on their face; the fix to the "
        "observations does not move them "
        "(THE_DROP_WAS_COUNTED_NOT_RECORDED)"
    ),
    "3_p17_c_vs_probe": {
        "key": "p17-c-vs-probe",
        "p_left": 0.9990,
        "n": 24,
        "verdict": "practically different",
        "what_it_means": (
            "**arm C is practically WORSE than the probe, decided.** "
            "The paired criterion recorded 4 of 5 seeds and left it "
            "UNRESOLVED. The direction is recorded because P(left) is a "
            "signed statement"
        ),
    },
    "4_p17_a_vs_c": {
        "key": "p17-a-vs-c",
        "p_right": 0.9910,
        "n": 24,
        "verdict": "practically different",
        "what_it_means": (
            "**THE SUBSTANTIVE ONE. The scheme attribution Phase 17 "
            "declined to claim is decidable under this instrument, and "
            "it holds.** P(right) 0.9910: **the magnitude-labelled "
            "regression (A) genuinely beats the contrastive scheme "
            "(C)** -- and in the direction Phase 17 called opposite to "
            "expectation, since A is the arm built ON the "
            "magnitude-to-grade assumption that C avoids"
        ),
        "what_phase_17_recorded": (
            "verbatim, ``THE_OBSERVED_PATTERN``'s companion: 'The A-C "
            "secondary that would attribute the pattern to the scheme "
            "is UNRESOLVED (p17-a-vs-c: +0.2518 at 4.9x threshold, "
            "condition 1 broken by one seed), so the attribution is not "
            "claimed'"
        ),
        "it_is_an_INSTRUMENTS_DISAGREE_CASE": (
            "**the same handling as the other two, and no other.** "
            "Phase 17's verdict STANDS UNCHANGED -- unresolved under "
            "the paired criterion, which is what that criterion says on "
            "this data. **The disagreement IS the finding**: one "
            "instrument asks whether every seed's interval excludes "
            "zero and one asks where a posterior over fold differences "
            "sits, and here they part. **Phase 17's cells are not "
            "amended and nothing is retrofitted**; the ledger is "
            "untouched and no row is added"
        ),
        "what_it_does_NOT_license": (
            "**it does not make the scheme attribution a CLAIM.** It "
            "makes it decidable UNDER THIS INSTRUMENT, and this "
            "instrument is descriptive: no ledger row, no figure "
            "changed. Writing 'the scheme is attributed' unqualified "
            "would drop the instrument it depends on"
        ),
    },
    "tag": "[MEASURED]",
}


#: **[BANKED 2026-09-02] The rerun. THE CITABLE RUN for Phase 23.**
THE_RERUN_IS_THE_CITABLE_RUN = {
    "run": "p23_rope__402a8477__p23-rope-45-2",
    "banked": "2026-09-02",
    "supersedes": "p23_rope__8bbb9a1b__p23-rope-45",
    "what_it_reproduced": (
        "**verdicts IDENTICAL: 0 practically equivalent, 4 practically "
        "different, 41 no decision**, and all four decisions matching "
        "to four decimals. The fix touched what the record CARRIES and "
        "nothing the arithmetic computes, and the rerun is the "
        "measurement that says so rather than the intention"
    ),
    "SUPERSEDED_FOR_OBSERVATIONS_ONLY_NOT_VOID": (
        "**the first forty-five run's NUMBERS WERE CORRECT AND ARE "
        "UNCHANGED.** Every posterior, probability, n and verdict in it "
        "stands; **only its drop records were empty**. It is superseded "
        "because a later run says strictly more, not because it says "
        "anything wrong -- **nothing is withdrawn and nothing computed "
        "from it needs revisiting**"
    ),
    "the_distinction_from_VOID_and_why_it_matters": (
        "**a VOID run's numbers are WRONG**, which is what makes a "
        "verification against the recorded band able to catch it "
        "(ladder.SIBLING_RUNS_AUDIT: 'earlier ambiguities were "
        "VOID-versus-good, where wrong numbers give the verification "
        "something to catch'). This is a third shape, beside that "
        "audit's PARTIAL: **right numbers, INCOMPLETE RECORD.** "
        "Collapsing it into 'void' would overstate the defect and "
        "invite re-deriving results that never moved; collapsing it "
        "into 'fine' would lose why the rerun happened. It is neither"
    ),
    "which_run_a_later_reader_cites": (
        "**``p23_rope__402a8477__p23-rope-45-2``, and only it.** The "
        "first forty-five run is cited for nothing on its own -- not "
        "because it is unsound but because the citable run contains it "
        "entire and says more"
    ),
    "tag": "[MEASURED]",
}


#: **[MEASURED 2026-09-02] The degenerate fold, with its cause.**
#:
#: One cell, and it is now readable: which arm, which seed, which fold,
#: and why. It is the whole of the n=24 shortfall.
THE_DEGENERATE_FOLD = {
    "measured": "2026-09-02, run p23_rope__402a8477__p23-rope-45-2",
    "the_cell": (
        "**contrast ``p17-c-vs-probe``, seed 1337, fold 1, arm "
        "``p17_arm_c``.** Cause: **'constant prediction within the "
        "fold'**, with ``n_patients`` 47, ``prediction_sd`` **exactly "
        "0.0** and ``truth_sd`` 0.6159. The truth varied; the "
        "prediction did not"
    ),
    "the_accounting_closes": (
        "**24 + 1 dropped + 0 unpaired = 25.** The whole shortfall is "
        "this one cell -- no second cause, and nothing left over. "
        "``p23_contrast_record`` would have REFUSED the run had it not "
        "closed, so the identity is enforced rather than observed"
    ),
    "what_it_says_about_the_ARM_not_the_arithmetic": (
        "**arm C's readout rounds a distance to a grade**, and in this "
        "fold the rounding collapsed all 47 patients onto ONE grade. "
        "The arm scored **-0.0185 overall** (phase17: 'both contrastive "
        "arms sit at zero, B -0.0044, C -0.0185'), so a fold in which "
        "it emits a single value is **the arm behaving as its score "
        "already said it does** -- and this is the first time the "
        "record carries a NUMBER for that behaviour rather than an "
        "aggregate consistent with it. **A property of the arm, "
        "evidenced**"
    ),
    "it_is_not_a_defect_in_the_instrument": (
        "the ROPE analysis did the right thing throughout: a constant "
        "prediction has NO correlation, and reporting it as NaN rather "
        "than as zero is what let the cause survive to be named. **The "
        "defect was only ever that the naming stopped short of the "
        "record**"
    ),
    "the_mechanism_was_DISTINGUISHED_BEFORE_THE_RERUN": (
        "**and the rerun confirmed rather than discovered it.** The "
        "first run's own metrics.json carried "
        "``degenerate_folds_dropped``, which read 1 -- so the shortfall "
        "was a DEGENERACY and not an unpaired fold, and that was "
        "settled from the shipped artifact without relaunching "
        "anything. **What the rerun added is the cause, not the "
        "class.** Worth recording as measurement preceding the run "
        "rather than waiting on it"
    ),
    "tag": "[MEASURED]",
}


def summary() -> dict:
    """The phase's registration, importable as one object."""
    return {
        "reckoning": PHASE_23_RECKONING,
        "split_randomised_not_built": SPLIT_RANDOMISED_HALF_NOT_BUILT,
        "method": THE_METHOD_REGISTERED,
        "rope": THE_ROPE_RULED,
        "width_precision_corrected": THE_WIDTH_WAS_A_ROUNDED_DISPLAY,
        "rounded_quotation_sweep": ROUNDED_QUOTATION_SWEEP,
        "observed": ROPE_OBSERVED,
        "expectation_held": THE_EXPECTATION_HELD,
        "substantive_finding": THE_SUBSTANTIVE_FINDING,
        "two_decisions": THE_TWO_DECISIONS,
        "fifteen_not_enumerable": THE_FIFTEEN_NOT_ENUMERABLE,
        "normality_observed": NORMALITY_AS_OBSERVED,
        "completion_to_forty_five": THE_COMPLETION_TO_FORTY_FIVE,
        "forty_five_outturn": THE_FORTY_FIVE_OUTTURN,
        "drop_counted_not_recorded": THE_DROP_WAS_COUNTED_NOT_RECORDED,
        "sweep_presence_not_content": THE_SWEEP_CHECKED_PRESENCE_NOT_CONTENT,
        "two_further_decisions": THE_TWO_FURTHER_DECISIONS,
        "rerun_citable": THE_RERUN_IS_THE_CITABLE_RUN,
        "degenerate_fold": THE_DEGENERATE_FOLD,
        "closing": PHASE_23_CLOSING,
        "extensions_and_silences": DECLARED_EXTENSIONS_AND_SILENCES,
        "hierarchical_excluded": THE_HIERARCHICAL_MODEL_IS_EXCLUDED,
        "rows_scoped": THE_ROWS_SCOPED,
        "pairing": PAIRING_ESTABLISHED_BY_CONSTRUCTION,
        "decision_threshold": THE_DECISION_THRESHOLD_RULED,
        "iem_row_excluded": THE_IEM_ROW_EXCLUDED,
        "attributions_corrected": THE_FIVE_SOURCE_ATTRIBUTIONS_WERE_UNVERIFIED,
        "readings": READINGS_COMMITTED,
        "normality": NORMALITY_DESCRIBED_NOT_TESTED,
        "exit_criteria": EXIT_CRITERIA,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        "settings": SETTINGS_TO_DECLARE,
    }
