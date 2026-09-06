"""Phase 26: the calibration ablation. RESTATED 2026-09-05, not locked.

Scheduled by ``phase25.PHASE_SEQUENCE_EXTENDED_8`` as SCHEDULED, NOT
REGISTERED. This module is the restate the amendment's first binding
clause requires, which is where scope lives and where the amendment
deliberately did not put it.

**Nothing here is locked and nothing is built.** The exit criteria below
are a DRAFT. No config exists, no task is registered, and the ledger
holds no row for this phase.

The phase reports six readouts together on one arm and one recipe, which
the verification established has never been done. It varies two training
settings, which the verification established no phase has ever done.
"""

from __future__ import annotations

from . import phase21

# --------------------------------------------------------------------------
# the reckoning
# --------------------------------------------------------------------------

#: **[RECKONED 2026-09-05] What the record already holds, stated before
#: the phase's own case.**
#:
#: The counter-evidence is the phase's own, so it is held to the standard
#: the record holds its cases to.
THE_RECKONING = {
    "reckoned": "2026-09-05, before any scope was written",

    "the_gap_is_real_and_it_is_the_union": (
        "**no record in this repository reports all six readouts for the "
        "same arm.** An exhaustive scan of every module-level record "
        "returns a maximum co-occurrence of THREE, at "
        "``train.phase3.GATE1_REFERENCE``, which carries PCC, MAE and "
        "RMSE. No shipped task emits the six either. The union is the "
        "deliverable, and it does not exist"
    ),
    "and_that_one_row_is_a_different_quantity": (
        "**the three-metric row is not the probe.** "
        "``GATE1_REFERENCE`` is ONE SEED, seed 1337, and its PCC is "
        "0.2719366, against the probe's five-seed 0.2520 at sd 0.0148. "
        "Same recipe, different population, similar name. Quoting its "
        "MAE and RMSE as the probe's would be the R2 error, so the "
        "phase measures rather than borrows"
    ),
    "what_IS_banked_for_the_probe": (
        "**three of the six.** PCC 0.2520 sd 0.0148 over five seeds "
        "(``ladder.BEST_ARM``), shrinkage 0.4947 "
        "(``phase21.CROSS_ARM_SHRINKAGE_OBSERVED['probe']``, mirrored at "
        "``phase22.FIVE_ARMS_DO_NOT_SHRINK['the_probe']``), and "
        "three-class accuracy 0.5181 sd 0.0159. RMSE and MAE exist only "
        "at one seed on the row above. **IEM does not exist for the "
        "probe at all**, although Phase 18 computed it on all 68 locked "
        "arms and banked only the ordering facts"
    ),
    "no_training_setting_has_ever_been_swept": (
        "**this would be the first.** The only pre-registered search is "
        "Phase 7B, 24 trials over pooling, head and ensemble, which are "
        "representation, architecture and feature combination. Its own "
        "out-of-budget list names resolution, graph ensembling and "
        "augmentation, so a training setting was never considered rather "
        "than considered and dropped. Everything else the repository "
        "calls a sweep sweeps geometry, seeds, a preprocessing threshold "
        "or record text"
    ),
    "and_the_record_says_WHY_they_are_fixed": (
        "**on purpose, so they cannot confound an arm comparison.** The "
        "reasoning is banked twice, that a setting tuned per backbone "
        "would confound the init ladder's backbone comparison with "
        "tuning effort, and that holding learning rate, weight decay and "
        "seeding identical is what makes a label the varying factor. "
        "**Phase 26 does not overturn that. It turns the held setting "
        "into the varying one on a SINGLE arm, where there is no "
        "backbone comparison to confound**"
    ),
    "the_two_factors_have_opposite_provenance": (
        "**weight decay was never chosen and has never moved.** Its only "
        "provenance statement in the record calls it the project "
        "default for AdamW, it is the schema default, it appears 180 "
        "times across the configs at 0.01 with no exception, and the "
        "governing document never uses the words. Patience 5 is stated "
        "in every config and reasoned nowhere. **Neither value has an "
        "argument behind it, which is the whole opportunity and also "
        "the whole risk**"
    ),
    "tag": "[RECKONED] -- the counter-evidence first, and it is substantial",
}


# --------------------------------------------------------------------------
# what the phase may not claim
# --------------------------------------------------------------------------

#: **[CARRIED 2026-09-05] The scale-invariance prohibition, BY REFERENCE.**
#:
#: Carried whole rather than restated, on the ``results_ledger``
#: MARGIN_TABLE precedent, because a second copy is a second thing to
#: drift. It is a registered tested literal and it binds this phase
#: harder than any other, since a calibration sweep that reports PCC and
#: shrinkage side by side invites exactly the reading it forbids.
THE_PROHIBITION = phase21.SCALE_INVARIANCE_PROHIBITION

#: **[BOUND 2026-09-05] What the prohibition forbids this phase to say.**
WHAT_THIS_PHASE_MAY_NOT_CLAIM = {
    "bound": "2026-09-05, before any number exists",
    "carries": "phase21.SCALE_INVARIANCE_PROHIBITION, whole, at THE_PROHIBITION",

    "1_no_reading_toward_the_ceiling": (
        "**a calibration result may not be read as bearing on the PCC "
        "ceiling.** The prohibition states that rescaling a shrunk "
        "prediction vector about the cohort mean changes RMSE and leaves "
        "PCC exactly unchanged, verified to 0.00e+00 at three "
        "rescalings. So no cell of this grid can explain the ceiling, "
        "whatever it does to calibration"
    ),
    "2_no_attribution_of_PCC_movement": (
        "**if PCC moves across the grid, the movement may not be "
        "attributed to the calibration change.** The two are "
        "arithmetically independent under rescaling, so a coincidence of "
        "direction is a coincidence until something else explains it"
    ),
    "3_the_reverse_is_also_forbidden": (
        "**a cell that improves calibration at unchanged PCC is not a "
        "failure**, and a cell that improves PCC at unchanged "
        "calibration is not a calibration result. The phase reports six "
        "readouts precisely so neither can be read as the other"
    ),
    "4_the_IEM_defects_travel": (
        "**every IEM figure this phase produces carries the three "
        "measured defects**, value inversion below absolute d of "
        "0.19753, gradient inversion below absolute d of 0.0719, and "
        "convention A's narrow-predictor favouritism at 1.685x "
        "inverted-region occupancy. They bite hardest here: **a sweep "
        "that reduces error moves arms toward the region where the "
        "metric misorders**"
    ),
    "5_and_never_beside_the_published_tables": (
        "``phase18.DELIVERABLES_REGISTERED['cleftgnn_iem_prohibition']`` "
        "governs unchanged. No IEM value from this phase is placed "
        "beside the comparator's published figures, because IEM is a "
        "scaled error whose magnitude is governed by the label's spread "
        "and the two designs do not share a label"
    ),
}


# --------------------------------------------------------------------------
# ruling one: the second factor
# --------------------------------------------------------------------------

#: **[RULED 2026-09-05] THE TWO FACTORS ARE WEIGHT DECAY AND PATIENCE.
#: The amendment named epoch budget. Budget was CONSIDERED AND REJECTED
#: on a banked measurement, not dropped.**
THE_SECOND_FACTOR_RULED = {
    "ruled": "2026-09-05, at the restate",
    "the_amendment_said": (
        "``phase25.PHASE_SEQUENCE_EXTENDED_8`` names the factors as "
        "weight decay and epoch budget. This restate replaces the second "
        "one, which is what a restate is for"
    ),

    "why_budget_is_inert_HERE": (
        "**it is measured inert on this harness, and the measurement is "
        "banked.** ``phase7c.ROUND_2_IS_ROUND_1_TRUNCATED`` states the "
        "mechanism: the head resets on a fixed seed at a constant "
        "learning rate with no schedule keyed to max_epochs, and the "
        "fold runner uses patience only to break, **so epoch n's fit is "
        "independent of the budget**. Its own figures: "
        "``identical_fits`` 23 of ``total_fits`` 35 across the two "
        "rounds. What max_epochs varies is the CHECKPOINT SELECTION "
        "WINDOW, not the amount of training"
    ),
    "and_it_was_already_tried_on_this_exact_arm": (
        "**arm 0 of Phase 7C is the probe's recipe with identity "
        "augmentation, and it did not move.** The record: 'Arm 0 held at "
        "0.2520 sd 0.0148 -- the gate survived the budget change, as "
        "predicted: with no augmentation the features are identical "
        "every epoch, so best-checkpoint selection picks the same epoch "
        "whether the loop stops at 6 or runs to 30.' It held under both "
        "``EPOCH_POLICY`` at 30 and ``MATCHED_EPOCH_POLICY`` at 3, "
        "stopping at epoch 1 in every fold. **A two-point budget "
        "variation on this phase's own arm therefore already exists in "
        "the record and moved nothing**"
    ),
    "REJECTED_NOT_DROPPED": (
        "**recorded so nobody revisits it.** Budget is not absent from "
        "the design because it was forgotten or because the phase "
        "shrank. It is absent because the record measured it inert under "
        "this harness on this arm, and a factor whose null is already "
        "banked is not a factor. Anyone proposing to add it back has to "
        "overturn ``ROUND_2_IS_ROUND_1_TRUNCATED`` first"
    ),
    "and_it_was_an_R2_hazard_besides": (
        "**'epoch budget' named two quantities.** The amendment's "
        "phrase and the record's phrase are the same words over "
        "training length in one and selection window in the other. "
        "Replacing the factor removes the collision rather than "
        "carrying it into a registration"
    ),

    "what_patience_IS": (
        "**the setting that actually determines how much fitting "
        "happens.** ``phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION`` records "
        "the precedent: 'a 769-parameter head over frozen embeddings "
        "converges immediately, inner-val never improves, patience "
        "terminates at epoch 1'. At patience 5 the fit stops at epoch 1. "
        "So patience, not budget, is the knob between one epoch and many"
    ),
    "the_far_end_is_already_shipped": (
        "**and reachable through config alone.** "
        "``phase7c.EPOCH_POLICY`` is max_epochs 30, patience 30, "
        "``early_stopping_can_fire`` False, monitor inner_val_mse, "
        "selection by best checkpoint. That is the far end of this axis "
        "and it needs no new code. The near end is the probe's own "
        "patience 5"
    ),
    "budget_is_HELD_and_here_is_the_value": (
        "**max_epochs is held at 30 for every cell, and 30 does not "
        "bind.** It does not bind because ``EPOCH_POLICY`` sets patience "
        "to 30 at the far end, so at the widest cell the budget and the "
        "patience are equal and early stopping cannot fire before the "
        "budget ends. At every narrower cell patience fires first by "
        "construction. **A held setting that could bind would confound "
        "the axis with a ceiling**, which is why the value is stated "
        "here rather than left to the config"
    ),
    "tag": "[RULED] -- factor replaced, rejection recorded, held value stated",
}


# --------------------------------------------------------------------------
# ruling two: the design stays two factor
# --------------------------------------------------------------------------

#: **[RULED 2026-09-05] THE EPOCH AXIS IS RESPECIFIED, NOT CONCEDED.**
THE_AXIS_IS_RESPECIFIED = {
    "ruled": "2026-09-05",
    "the_ruling": (
        "**with patience as the second factor the design remains two "
        "factor.** The epoch dimension is not removed from the phase. "
        "It is measured through the setting that moves it rather than "
        "through the setting that does not"
    ),
    "the_alternative_that_was_NOT_taken": (
        "**a single-factor weight-decay sweep**, dropping the epoch "
        "dimension entirely. It was available and it was not taken, "
        "because the record's own account of why the fit stops at epoch "
        "1 is a statement about patience, and a calibration phase that "
        "declined to test it would be leaving the more likely of its two "
        "knobs untouched. Recorded so the choice is visible as a choice"
    ),
    "what_a_concession_would_have_looked_like": (
        "**it would have been recorded as one.** The amendment's second "
        "binding clause says a scheduled phase that does not run is "
        "cancelled with a reason and never silently dropped. The same "
        "discipline applies inside a phase: a dropped factor is either a "
        "recorded rejection, as budget is above, or a concession. This "
        "is neither. It is a respecification"
    ),
}


# --------------------------------------------------------------------------
# ruling three: the IEM floor
# --------------------------------------------------------------------------

#: **[RULED 2026-09-05] THE FLOOR IS RECOMPUTED IN THE RUN, AND THE PHASE
#: 18 VALUE IS RECOVERED AS A CROSS-CHECK.**
THE_FLOOR_RULED = {
    "ruled": "2026-09-05",
    "the_problem": (
        "**the constant-predictor floor exists as a FUNCTION and not as "
        "a figure.** ``phase18.constant_predictor_iem`` fills a constant "
        "at the panel mean and passes it through the same scorer the "
        "arms go through. It is called once, and its one output went to "
        "the metrics file of a cluster-only run. **No numeric value for "
        "it exists anywhere in this repository**, while "
        "``DELIVERABLES_REGISTERED['d5_iem_third_family']`` requires a "
        "'floor beside every figure'"
    ),
    "1_recompute_in_the_run": (
        "**each cell computes its own floor from its own execution.** "
        "The floor depends only on the truth vector, so it is constant "
        "across cells by construction, and computing it per cell is "
        "therefore also a self-consistency check on the run. A figure "
        "that carries a floor transcribed from somewhere else is a "
        "figure whose floor nobody re-derived"
    ),
    "2_recover_the_phase_18_value_separately": (
        "**and compare the two.** The Phase 18 value is recoverable from "
        "that run's own metrics file without relaunching anything, the "
        "same shape as the Spearman figure Phase 9 computed and never "
        "banked. Recovery is a read, not a run"
    ),
    "why_BOTH": (
        "**two independent computations agreeing is stronger than one "
        "transcribed number, and a disagreement is itself a finding.** "
        "If they agree, the floor is established twice by different "
        "paths. If they disagree, then either the truth vector differs "
        "between the two phases or the scorer does, and both of those "
        "are worth knowing before any IEM figure is quoted. **The "
        "disagreement case is not a failure of this phase**"
    ),
    "what_it_does_not_do": (
        "**it does not settle the identity question.** "
        "``phase11.CASE_IEM_IDENTITY_OPEN`` remains open and this phase "
        "does not touch it. The floor makes this project's IEM figures "
        "readable against each other, not against anyone else's"
    ),
}


# --------------------------------------------------------------------------
# ruling four: timing
# --------------------------------------------------------------------------

#: **[RULED 2026-09-05] THE PHASE REPORTS PER-FIT WALL CLOCK.**
TIMING_IS_REPORTED = {
    "ruled": "2026-09-05",
    "why": (
        "**no banked runtime exists for a frozen-head refit on cached "
        "embeddings.** The amendment says this phase runs on cached "
        "embeddings and the ordering ruling says it runs in minutes. "
        "The record's own position on that class of claim is "
        "``phase20.COMPUTE_GATE_DESIGNED``, which reasons "
        "'a 769-parameter head over 5 folds x 5 seeds. Expectation: "
        "seconds' and immediately answers itself with 'the expectation "
        "is not the measurement'. **So the cost claim behind this "
        "phase's own scheduling is unmeasured**"
    ),
    "the_instrumentation_already_exists_and_already_runs": (
        "``run.task_train_cv`` times every seed and writes "
        "``summary['wall_clock_seconds_by_seed']``. Phases 20, 22 and 25 "
        "all ran through it. **The measurement has been taken many "
        "times and never banked**, which is why this is a reporting "
        "ruling rather than a build"
    ),
    "what_gets_banked": (
        "per-fit wall clock for every cell, so the grid's cost is a "
        "measurement rather than an expectation, and so the next phase "
        "that wants to size a head-refit sweep can quote a figure. **The "
        "unit is the fold-fit**, since a cell is five seeds by five "
        "folds and a per-cell total hides which of the two is doing the "
        "work"
    ),
    "and_it_prices_the_scheduling_claim": (
        "**the ordering ruling put this phase before a supervision "
        "meeting on the ground that it is cheap.** Reporting the timing "
        "turns that ground into something checkable after the fact, "
        "which is the only way an estimate becomes a figure"
    ),
    "DISCHARGED_2026_09_06": (
        "**the ruling is met and the timing is no longer reported.** "
        "The wall clock was read from the sixteen cell logs and banked "
        "at THE_RUNTIME_MEASURED in the unit this ruling fixed, and the "
        "scheduling claim it wanted priced is priced at "
        "THE_SCHEDULING_CLAIM_SETTLED. **This entry keeps its name and "
        "its reasoning because the ruling was made before the numbers "
        "existed and the record does not rewrite what it ruled**, but "
        "no reader should take the name as the current state"
    ),
}


# --------------------------------------------------------------------------
# the grid, as a design
# --------------------------------------------------------------------------

#: **[DESIGNED 2026-09-05, NOT SIZED] The grid's SHAPE. The cell count is
#: not chosen here.**
THE_GRID_AS_A_DESIGN = {
    "designed": "2026-09-05",
    "shape": (
        "**two factors, fully crossed, on one arm and one recipe.** "
        "Factor one is weight decay, whose current value 0.01 is the "
        "project default and has never moved. Factor two is patience, "
        "whose current value 5 is stated everywhere and reasoned "
        "nowhere, with the shipped far end at 30. Every other setting is "
        "the probe's, carried rather than retyped"
    ),
    "the_count_is_NOT_chosen_here": (
        "**levels are chosen when the phase locks, not at the "
        "restate.** What is fixed here is the shape and the cost law: "
        "W levels of weight decay by P levels of patience gives W times "
        "P cells, each five seeds by five folds, so 25 times W times P "
        "fold-fits. Naming W and P now would be sizing a grid before "
        "anyone has argued for a level"
    ),
    "one_config_per_cell_is_forced": (
        "**the config surface has no list-valued fit knob.** Of "
        "``train_cv``'s fields exactly two take a list, the "
        "concatenation artifacts and the seeds, and the seeds one is "
        "documented as 'give a list to sweep in one job'. weight_decay "
        "is a scalar float and patience is a scalar integer. So seeds "
        "sweep inside a job and the two factors do not, which means "
        "W times P configs emitted by a generator"
    ),
    "the_generator_precedent": (
        "``scripts/generate_phase22_configs.py`` emits a three by two "
        "two-factor design as six one-cell configs with ``--check`` "
        "drift detection. That shape transfers directly and is the "
        "reason no new machinery is needed"
    ),
    "the_lattice_discipline_still_applies": (
        "**cells are a grid, contrasts are not.** "
        "``ladder.comparisons()`` asserts every comparison differs in "
        "exactly its declared field, so a two-factor grid is a legal "
        "set of cells whose CONTRASTS must each stay one-factor. A "
        "diagonal comparison across both factors is not a contrast this "
        "project can make"
    ),
    "the_trap_that_would_make_it_vacuous": (
        "**if patience does not move the selected epoch, the grid "
        "measures nothing on that axis**, exactly as budget does not. "
        "The record predicts the opposite, since patience 5 is what "
        "terminates at epoch 1 and patience 30 is what the shipped "
        "policy uses to reach 30. But it is a prediction, and the exit "
        "criteria draft below requires selected_epoch to be reported "
        "per cell so the axis is shown to have moved before anything is "
        "read off it"
    ),
}


#: **[DECLARED 2026-09-05] Settings this phase must declare rather than
#: inherit silently.**
SETTINGS_NEEDING_DECLARATION = {
    "declared": "2026-09-05",
    "max_epochs": (
        "**30, held, and stated because a held setting that could bind "
        "would confound the axis.** See "
        "THE_SECOND_FACTOR_RULED['budget_is_HELD_and_here_is_the_value']"
    ),
    "monitor": (
        "**inner_val_mse, carried from the probe.** "
        "``phase7c.EPOCH_POLICY`` notes that its own deviation from "
        "inner_val_pcc was reasoned on patience firing on noise, a "
        "reason that cannot apply when patience cannot fire. **Here "
        "patience CAN fire and is the factor**, so the monitor is a "
        "held setting that interacts with the axis and must be declared "
        "rather than defaulted"
    ),
    "batch_size": (
        "**32, the probe's own, declared explicitly, and the reason is "
        "this phase rather than the last one.** Phase 26 varies "
        "patience, which does change the fit, so this is a phase where "
        "a silently defaulted setting would matter, and declaring one "
        "costs nothing. [CORRECTED 2026-09-06: this read that two Phase "
        "25 configs taking the schema default was a live instance of "
        "the hazard. The divergence there is real and it is INERT, "
        "because the head is fit full batch and batch size never "
        "reaches it, so the original wording implied a consequence that "
        "does not exist. See phase25.PHASE_25_CLOSING criterion 2.]"
    ),
    "seeds": (
        "**the shared five, carried.** Seeds sweep inside a job, so the "
        "five are a list on every cell rather than five configs"
    ),
    "learning_rate": "0.001, the probe's, carried and declared",
    "trainable": "head, carried. The backbone is frozen and stays frozen",
    "embeddings": (
        "the probe's cached set, declared by role with its rollup, so "
        "guard 3 verifies the features are the ones the 0.2520 figure "
        "was measured on"
    ),
}


#: **[MEASURED 2026-09-05] What the six-metric union needs that the
#: training path does not emit.**
WHAT_THE_UNION_REQUIRES = {
    "measured": "2026-09-05, by reading the shipped path",
    "four_come_free": (
        "**PCC, RMSE, MAE and shrinkage.** The first three come from "
        "``harness.CVResult.metrics``, and shrinkage from "
        "``train.phase3.sanity_report``, which has computed "
        "sd(prediction) over sd(truth) on every arm since 2026-07-28 "
        "and ships its interpretation string beside it"
    ),
    "three_class_accuracy_exists_outside_the_train_path": (
        "``classification.prf_report(...)['accuracy']`` computes it, and "
        "``eval.metrics.to_3class`` supplies the binning. It is not part "
        "of what a train_cv run emits, so it is a consolidation step "
        "rather than new arithmetic"
    ),
    "IEM_IS_THE_ONE_GAP": (
        "**the training path does not emit IEM at all.** It exists as "
        "``phase11.iem``, wrapped as ``phase18.iem_score`` under "
        "convention A, and it is reached only by the metric-space task. "
        "So the union needs IEM and its floor computed at the point the "
        "predictions exist, which is the one piece of assembly this "
        "phase requires"
    ),
    "and_no_frozen_module_needs_touching": (
        "**this is the part that makes the phase cheap.** "
        "``eval/metrics.py``, ``train/harness.py`` and ``data/folds.py`` "
        "are all frozen, and the eval tree is frozen at its current "
        "members so a sibling file cannot be added either. Three of the "
        "six are already inside the frozen metrics module. The other "
        "three live outside it. **The consolidation happens in the task, "
        "which is not frozen, and the two knobs are config schema "
        "fields, which are not frozen either**"
    ),
}


# --------------------------------------------------------------------------
# exit criteria, DRAFT
# --------------------------------------------------------------------------

#: **[DRAFT 2026-09-05, NOT LOCKED] Exit criteria.**
#:
#: A draft, on the ``phase22.EXIT_CRITERIA_DRAFT`` precedent, which
#: became a locked ``EXIT_CRITERIA`` only at the lock. Readings are NOT
#: written here and are not written until the criteria lock, so that no
#: reading is written after a number exists.
EXIT_CRITERIA_DRAFT = {
    "drafted": "2026-09-05, NOT LOCKED",
    "status": "DRAFT -- the lock is a separate act and has not happened",

    "1_all_six_for_every_cell": (
        "PCC, shrinkage, RMSE, MAE, three-class accuracy and IEM "
        "reported together for every cell, from the same predictions, "
        "or the phase has not produced the union it exists for"
    ),
    "2_the_floor_beside_every_IEM_figure": (
        "the constant-predictor IEM recomputed in the run and carried "
        "beside each cell's IEM, plus the recovered Phase 18 value and "
        "the comparison of the two, agreement or disagreement stated "
        "either way"
    ),
    "3_the_axis_is_shown_to_have_moved": (
        "selected_epoch reported per cell per fold. **If patience does "
        "not change the selected epoch, the epoch axis is inert here as "
        "budget is, and that is the phase's result on that axis rather "
        "than a defect to work around**"
    ),
    "4_the_prohibition_travels": (
        "``phase21.SCALE_INVARIANCE_PROHIBITION`` carried whole beside "
        "any figure that reports PCC and a calibration readout together, "
        "and no sentence anywhere attributing PCC movement to a "
        "calibration change"
    ),
    "5_the_IEM_defects_travel": (
        "the three measured defects cited by their own names beside "
        "every IEM figure, and the comparator prohibition observed"
    ),
    "6_per_fit_wall_clock_banked": (
        "wall clock per fold-fit for every cell, so the cost claim "
        "behind this phase's scheduling becomes a measurement"
    ),
    "7_one_factor_contrasts_only": (
        "every contrast varies exactly one factor, per "
        "``ladder.comparisons()``. Diagonal comparisons are not made"
    ),
    "8_a_ledger_row_only_if_claimable": (
        "the criterion is PLAN 4.3 unchanged. **The reckoning predicts "
        "no claimable row**, since the cohort has not resolved a "
        "difference of this kind before, and a phase that predicts its "
        "own null says so before it runs"
    ),
    "what_is_NOT_here": (
        "**readings.** They are written at the lock, not at the "
        "restate, so that each possible outcome has a meaning attached "
        "before any number exists rather than after"
    ),
}


# --------------------------------------------------------------------------
# the lock
# --------------------------------------------------------------------------

#: **[LOCKED 2026-09-06] SIXTEEN CELLS. Four levels of weight decay by
#: four of patience, fully crossed on the probe's arm.**
#:
#: Every level carries its reason, and every reason is written here,
#: before any number exists.
THE_GRID_LOCKED = {
    "locked": "2026-09-06",
    "shape": "4 weight decay x 4 patience = 16 cells, fully crossed",
    "held": (
        "max_epochs 30 on every cell, and every other setting the "
        "probe's. See SETTINGS_NEEDING_DECLARATION for why 30 is the "
        "held value and why it does not bind"
    ),

    "weight_decay_levels": (0.0, 0.001, 0.01, 0.1),
    "why_each_weight_decay_level": {
        "0.0": (
            "**the absence of the mechanism, not merely a small value.** "
            "It is the only level that can answer whether the default is "
            "doing anything at all. A default nobody chose might be "
            "regularising, might be inert, and the record cannot say "
            "which because the value has never moved"
        ),
        "0.001": (
            "**one decade below the default**, and the ruled requirement "
            "that the span reach below 0.01. It separates 'less decay' "
            "from 'no decay', so if 0.0 and 0.001 agree the effect is a "
            "threshold and if they differ it is graded"
        ),
        "0.01": (
            "**the current value, and the anchor.** The grid contains the "
            "probe's own recipe at patience 5, so one cell must reproduce "
            "the banked 0.2520 at sd 0.0148. **That cell is the run's "
            "end-to-end check**, the same role arm 0 played in Phase 7C"
        ),
        "0.1": (
            "**one decade above, which is the ruled 'well above'.** The "
            "question is whether the current value over regularises, and "
            "that is only answerable if the grid contains a level that "
            "certainly over regularises. If more decay shrinks the "
            "predictions further, this is the cell where it shows"
        ),
        "and_why_log_spacing": (
            "**a regularisation strength is a scale, not an offset.** "
            "Linear steps around 0.01 would put three levels inside one "
            "decade and none outside it. The zero is not part of the log "
            "ladder and is included because absence is a different "
            "question from smallness"
        ),
    },

    "patience_levels": (5, 10, 20, 30),
    "why_each_patience_level": {
        "5": (
            "**the current value, and the anchor.** The record measured "
            "what it does: at patience 5 a small head over frozen "
            "embeddings converges immediately and terminates at epoch 1"
        ),
        "10": (
            "**the first level at which early stopping could plausibly "
            "stop binding.** If the fit still terminates at epoch 1 here, "
            "the transition is not near the current value and the "
            "remaining two levels are what test whether it exists at all"
        ),
        "20": (
            "**the interior level that separates a graded effect from a "
            "step.** Between 10 and 30 it is the only point that can "
            "distinguish a selected epoch drifting upward from one that "
            "jumps when patience stops binding"
        ),
        "30": (
            "**the shipped far end, where early stopping CANNOT fire.** "
            "At patience 30 against a held budget of 30 the loop runs to "
            "the budget and selection is best checkpoint over the whole "
            "range. This is ``phase7c.EPOCH_POLICY`` exactly, reachable "
            "through config alone, and it is the far end the restate "
            "named"
        ),
        "and_why_not_log_spacing_here": (
            "**patience is a count of epochs, not a scale.** The "
            "quantity that matters is where it stops binding, and that "
            "is a location on a linear axis bounded above by the budget. "
            "Log spacing would put two levels below 10 and none between "
            "10 and 30, which is the region the record's prediction "
            "makes interesting"
        ),
    },

    "the_anchor_cell_is_the_check": (
        "**weight decay 0.01 with patience 5 IS the probe.** It must "
        "reproduce 0.2520 at sd 0.0148 over the five shared seeds. If it "
        "does not, the run is wrong and no other cell may be read. This "
        "is a gate on the path rather than a result of the phase"
    ),
    "and_the_anchor_differs_from_the_probe_in_ONE_held_field": (
        "**the budget, 30 here against the probe's 40, and the record "
        "says why that does not weaken the anchor.** The budget is held "
        "at 30 across the grid so patience is never bounded by it at the "
        "far end. At the anchor's patience of 5 the fit terminates at "
        "epoch 1, so the selection window never reaches either bound, "
        "and ``phase7c.ROUND_2_IS_ROUND_1_TRUNCATED`` measured that a "
        "given epoch's fit is independent of the budget. Phase 7C arm 0 "
        "is the same case and held at the same figure under budgets of "
        "30 and 3. **Stated because a reader comparing the two configs "
        "will see 30 against 40 and is entitled to know it was "
        "considered**"
    ),
    "sixteen_is_not_a_multiplicity_correction_problem": (
        "**the family is the grid and it is fixed here.** No cell is "
        "added once numbers exist, nothing is selected, and no contrast "
        "is promoted on a result. The grid is DESCRIPTIVE unless a "
        "contrast passes both conditions of the criterion, which the "
        "reckoning predicts none will"
    ),
    "cost": (
        "16 cells x 5 seeds x 5 folds = **400 fold fits**, each a "
        "769-parameter head over cached embeddings. No banked runtime "
        "exists for one, which is why the phase emits per fit wall clock "
        "(TIMING_IS_REPORTED)"
    ),
}


#: **[READINGS WRITTEN AT THE LOCK 2026-09-06] What each outcome would
#: mean, written before any number exists.**
#:
#: The restate deliberately carried none. They are written here because
#: this is the lock, and a reading written after a number is not a
#: reading.
READINGS = {
    "written": "2026-09-06, at the lock, before any cell has run",

    "weight_decay_moves_calibration_and_not_PCC": (
        "**the registered expectation, and it is what the prohibition "
        "predicts.** Shrinkage, RMSE, MAE and IEM move across the decay "
        "levels while PCC does not. Reading: the default is doing "
        "regularisation work that the primary metric cannot see, and the "
        "project has been choosing a calibration without knowing it. "
        "**This does not touch the ceiling** and may not be reported as "
        "though it did"
    ),
    "weight_decay_moves_nothing": (
        "**the null, and it is a real answer.** Reading: 0.01 is inert "
        "on this arm, the default was harmless, and one of the two "
        "settings the project never chose turns out not to matter. The "
        "phase would then have converted an unexamined default into a "
        "measured non-issue, which is the same shape as the feature "
        "source axis closing"
    ),
    "weight_decay_moves_PCC_too": (
        "**the surprise, and it is the one that needs the most care.** "
        "Reading: PCC is not invariant to this because decay changes the "
        "FIT and not only the scale, which is a different mechanism from "
        "rescaling and is not forbidden by the prohibition. It would "
        "still be UNRESOLVED unless it passes both conditions, and the "
        "reckoning says the cohort has not resolved a difference of this "
        "kind before"
    ),

    "patience_moves_the_selected_epoch": (
        "**the axis is live.** Reading: the record's account of why the "
        "fit stops at epoch 1 is a statement about patience, and raising "
        "it buys more fitting. Whatever the downstream metrics do is "
        "then a result about training length on this cohort"
    ),
    "patience_does_NOT_move_the_selected_epoch": (
        "**the axis is inert here exactly as the budget is, and that is "
        "the phase's result on that axis rather than a defect.** Reading: "
        "inner validation never improves after epoch 1, so no patience "
        "reachable under a 30 epoch budget can change the selection. "
        "**This would extend ``phase7c.ROUND_2_IS_ROUND_1_TRUNCATED`` "
        "from the budget to patience**, and it is worth having because "
        "it closes the epoch dimension by measurement rather than by "
        "assumption"
    ),
    "patience_moves_the_epoch_but_no_metric": (
        "**the fit gets longer and nothing improves.** Reading: the head "
        "converges and then sits, so extra epochs are extra compute. "
        "This is the outcome the record's own account predicts most "
        "strongly, since it says inner validation never improves"
    ),

    "the_two_factors_interact": (
        "**decay changes where patience bites, or the reverse.** "
        "Reading: reported as an interaction and NOT decomposed, because "
        "a two factor grid can show that an interaction exists and "
        "cannot attribute it. Any attribution would need a third factor "
        "the grid does not have"
    ),
    "the_anchor_cell_disagrees_with_the_banked_probe": (
        "**the run is wrong and nothing else is read.** Reading: not a "
        "finding about calibration, a failure of the path. The cell that "
        "is the probe must reproduce the probe"
    ),
    "and_what_NO_outcome_licenses": (
        "**none of the sixteen cells, in any combination, says anything "
        "about the PCC ceiling.** WHAT_THIS_PHASE_MAY_NOT_CLAIM governs "
        "every reading above, and the prohibition is carried whole at "
        "THE_PROHIBITION"
    ),
}


#: **[LOCKED 2026-09-06] Exit criteria. The draft above is superseded by
#: this and preserved in place.**
EXIT_CRITERIA = {
    "locked": "2026-09-06",
    "supersedes": "EXIT_CRITERIA_DRAFT, preserved unchanged",

    "1_all_six_for_every_cell": (
        "PCC, shrinkage, RMSE, MAE, three-class accuracy and IEM "
        "reported together for all sixteen cells, from the same "
        "predictions, or the phase has not produced the union it exists "
        "for"
    ),
    "2_the_floor_beside_every_IEM_figure": (
        "the constant-predictor IEM recomputed in the run and carried "
        "beside each cell's IEM, plus the recovered Phase 18 value and "
        "the comparison of the two, agreement or disagreement stated "
        "either way"
    ),
    "3_the_axis_is_shown_to_have_moved": (
        "selected_epoch reported per cell per fold. If patience does not "
        "change it, the epoch axis is inert here as the budget is, and "
        "that is the result on that axis"
    ),
    "4_the_prohibition_travels": (
        "``phase21.SCALE_INVARIANCE_PROHIBITION`` carried whole beside "
        "any figure reporting PCC and a calibration readout together, "
        "and no sentence attributing PCC movement to a calibration "
        "change"
    ),
    "5_the_IEM_defects_travel": (
        "the three measured defects cited by their own names beside "
        "every IEM figure, and the comparator prohibition observed"
    ),
    "6_per_fit_wall_clock_banked": (
        "wall clock per fold fit for every cell, so the cost claim "
        "behind this phase's scheduling becomes a measurement"
    ),
    "7_one_factor_contrasts_only": (
        "every contrast varies exactly one factor. Diagonal comparisons "
        "across both are not made"
    ),
    "8_a_ledger_row_only_if_claimable": (
        "PLAN 4.3 unchanged. The reckoning predicts no claimable row"
    ),
    "9_the_anchor_cell_reproduces_the_probe": (
        "**[ADDED AT THE LOCK]** weight decay 0.01 with patience 5 "
        "reproduces 0.2520 at sd 0.0148, or the run is wrong and no cell "
        "is read. The restate had no such gate because it had no levels"
    ),
    "10_the_readings_were_written_at_the_lock": (
        "**[ADDED AT THE LOCK]** READINGS is dated 2026-09-06 and "
        "predates every number. A reading written after a number is not "
        "a reading, and this criterion is what makes that checkable"
    ),
}


# --------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------

#: **[MEASURED 2026-09-06, run p26_calibration_table at sha 58c7e847]
#: THE SIXTEEN CELLS.**
#:
#: Read from the run's own metrics file, not from a message. Means over
#: five seeds, six digits, as the file carries them.
THE_SIXTEEN_CELLS_OBSERVED = {
    "measured": "2026-09-06, run p26_calibration_table at sha 58c7e847",
    "both_refusals_stayed_quiet": (
        "the declared cell set matched the locked grid and the anchor "
        "reproduced the probe, so neither of the task's two refusal "
        "conditions fired"
    ),

    "means_by_weight_decay": {
        "0.0": {
            "pcc": 0.252041, "shrinkage": 0.494805, "rmse": 0.656003,
            "mae": 0.532013, "acc3": 0.518987, "iem": 0.528256,
        },
        "0.001": {
            "pcc": 0.252041, "shrinkage": 0.494798, "rmse": 0.656002,
            "mae": 0.532013, "acc3": 0.518987, "iem": 0.528255,
        },
        "0.01": {
            "pcc": 0.252046, "shrinkage": 0.494740, "rmse": 0.655991,
            "mae": 0.532007, "acc3": 0.518143, "iem": 0.528248,
        },
        "0.1": {
            "pcc": 0.252092, "shrinkage": 0.494156, "rmse": 0.655879,
            "mae": 0.531952, "acc3": 0.518143, "iem": 0.528183,
        },
    },
    "why_the_table_has_four_rows_and_not_sixteen": (
        "**within every weight decay group the four patience levels are "
        "BIT IDENTICAL**, not flat to some number of decimals. Every "
        "per-seed value and every mean agrees exactly, so four cells of "
        "the grid are one measurement repeated. Sixteen rows would "
        "report the same four numbers four times each"
    ),

    "spread_across_the_whole_grid": {
        "acc3": 0.0008438819,
        "shrinkage": 0.0006488915,
        "rmse": 0.0001239189,
        "iem": 0.0000729411,
        "mae": 0.0000613519,
        "pcc": 0.0000513947,
    },
    "the_largest_movement_is_ONE_PATIENT": (
        "**the biggest number anywhere in the grid is three-class "
        "accuracy at 0.0008438819, and it is the metric's own "
        "granularity.** One patient reclassified in one seed out of 237 "
        "patients by five seeds is 1/(237*5) = 0.000843881856540, which "
        "equals the observed spread to within 1e-12. So the largest "
        "effect a hundredfold change in weight decay produces, across "
        "every readout, is a single patient crossing a class boundary in "
        "a single seed"
    ),
    "and_it_is_NOT_shrinkage": (
        "**a correction to the shape of the summary, recorded because "
        "the difference matters.** Shrinkage moves 0.0006488915 between "
        "weight decay 0.0 and 0.1, which is the six-ten-thousandths "
        "figure. It is the second largest movement, not the largest. "
        "Three-class accuracy moves more. Read from the file rather than "
        "carried forward"
    ),
    "every_movement_runs_the_same_way": (
        "raising weight decay from 0.0 to 0.1 lowers shrinkage, RMSE, "
        "MAE and IEM and raises PCC, all by amounts in the fourth "
        "decimal or beyond. The directions are consistent and the "
        "magnitudes are not resolvable"
    ),
    "tag": "[MEASURED] -- sixteen cells, six readouts, nothing claimable",
}


#: **[MEASURED 2026-09-06] THE MECHANISM, which criterion 3 was written
#: to establish rather than assume.**
THE_MECHANISM_MEASURED = {
    "measured": "2026-09-06, from the run",

    "patience_reached_the_fit_and_changed_nothing": (
        "**and the evidence is identity, not similarity.** All four "
        "patience levels, 5 through the shipped far end of 30, produce "
        "BIT IDENTICAL results at every weight decay: the same five "
        "per-seed values on all six readouts, to full double precision. "
        "Two fits that differ in when they are allowed to stop, and "
        "agree exactly, selected the same checkpoint"
    ),
    "why_that_happens": (
        "**the best checkpoint is the first epoch, so every patience "
        "level selects the same model.** Inner validation never improves "
        "after epoch 1 on a small head over frozen embeddings, which "
        "``phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION`` measured before this "
        "phase existed, so raising the ceiling on how long the loop may "
        "wait cannot change what it picks"
    ),
    "the_selected_epoch_itself_is_REPORTED_not_read": (
        "**[REPORTED] the selected epoch is 1 in every fold of all four "
        "grid corners.** That reached this record through a message and "
        "**the run's metrics file does not carry it**, because the "
        "consolidation task never emitted it. The bit identity above IS "
        "read from the file and is the stronger evidence for the same "
        "conclusion, but it is not the same statement, and criterion 3 "
        "asked for the epoch. See WHAT_CRITERION_3_DID_NOT_GET"
    ),
    "SUPERSEDED_2026_09_06_the_epoch_is_now_READ": (
        "**the entry above is preserved and its [REPORTED] tag no "
        "longer holds.** The selected epoch was extracted from each "
        "cell run's own per-seed metrics and read here: **400 fold "
        "observations, all four grid corners and the twelve cells "
        "between them, every one at epoch 1** "
        "(THE_SELECTED_EPOCHS_OBSERVED). It is no longer four corners "
        "on a message, it is the whole grid from the runs"
    ),
    "and_patience_is_INERT_IN_EFFECT_rather_than_UNREACHED": (
        "**[CORRECTED 2026-09-06] the distinction this record was "
        "missing, and it separates patience from the budget.** The "
        "budget does not reach the fit at all: "
        "``phase7c.ROUND_2_IS_ROUND_1_TRUNCATED`` measured that a given "
        "epoch's fit is independent of it, so raising it changes "
        "nothing that happens. **Patience is different. It reaches the "
        "fit and the fit obeys it**, running longer at higher levels, "
        "and the selection is unmoved only because the best checkpoint "
        "stays at the first epoch whatever comes after it. So patience "
        "is not inert as an INPUT. It is inert in EFFECT, and it is "
        "paid for. The evidence for the longer running is the wall "
        "clock, which is [REPORTED] and not yet read: see "
        "WHAT_CRITERION_3_DID_NOT_GET"
    ),
    "and_it_extends_the_budget_finding_to_patience": (
        "``phase7c.ROUND_2_IS_ROUND_1_TRUNCATED`` measured that a given "
        "epoch's fit is independent of the BUDGET, with 23 of 35 fits "
        "identical across two rounds. **This measures the same of "
        "PATIENCE, and more sharply: 16 of 16 cells are identical within "
        "their group rather than most of them.** The epoch dimension is "
        "now closed on both settings by measurement"
    ),
    "which_is_why_weight_decay_moves_so_little": (
        "**it has one epoch in which to act.** A hundredfold change, "
        "absence included, is applied to a single optimiser step over "
        "769 parameters. The wonder is not that the movement is small "
        "but that it is detectable at all"
    ),
}


#: **[MEASURED 2026-09-06] THE SELECTED EPOCH, EVERY CELL, EVERY SEED,
#: EVERY FOLD. Criterion 3 closes.**
#:
#: Read from ``p26_selected_epochs.json``, extracted from each cell run's
#: own per-seed metrics under its out-of-fold selected-epochs field. Not
#: from a message: the earlier record carried this as REPORTED and that
#: tag is now superseded.
THE_SELECTED_EPOCHS_OBSERVED = {
    "measured": "2026-09-06, from each cell run's own per-seed metrics",
    "shape": (
        "16 cells x 5 seeds x 5 folds = **400 fold observations**, every "
        "cell carrying all five seeds and every seed all five folds"
    ),
    "the_result": (
        "**every one of the 400 is epoch 1.** The set of distinct values "
        "across the whole grid has one member. Not mostly, not a "
        "majority, not a mean of 1.0 with spread. Uniform"
    ),
    "why_it_matters_that_it_is_UNIFORM": (
        "**a majority at epoch 1 would leave the mechanism open**, since "
        "a few late selections would mean patience sometimes bites and "
        "the identity of the readouts would need another explanation. "
        "One value everywhere closes it: the best checkpoint is the "
        "first epoch under every setting tested, so every patience level "
        "selects the same model and the bit identity of the readouts "
        "follows"
    ),
    "and_it_is_the_direct_form_of_what_was_already_shown": (
        "the bit identity was read from the run at the close and is "
        "still the stronger evidence in one sense, because it shows the "
        "outputs agree rather than that one input matched. **This is the "
        "quantity criterion 3 asked for**, and having both means the "
        "conclusion rests on the selection and on its consequence rather "
        "than on either alone"
    ),
    "tag": "[MEASURED] -- 400 of 400 at epoch 1, criterion 3 met",
}


#: **[MEASURED 2026-09-06] IEM AGAINST ITS FLOOR.**
#:
#: The three measured defects ride every figure here, as criterion 5
#: requires. See WHAT_THIS_PHASE_MAY_NOT_CLAIM key 4.
THE_IEM_RESULT = {
    "measured": "2026-09-06",
    "floor": 0.5371587111538849,
    "best_cell": 0.5281826341729179,
    "worst_cell": 0.5282555752558875,
    "every_cell_is_below_the_floor": (
        "**all sixteen beat the constant predictor**, by 0.0089760770 at "
        "the best cell. That is the one thing in this grid that is not "
        "small, and it says the head is doing something rather than "
        "predicting the mean"
    ),
    "and_the_grid_moves_almost_none_of_it": (
        "**the entire IEM spread across the grid is 0.0000729411, which "
        "is 0.8 percent of the distance from the best cell to the "
        "floor.** Tuning both settings across their full ranges moves "
        "IEM by under one hundredth of the gap the head has already "
        "closed"
    ),
    "the_cross_check_agrees_exactly": (
        "**absolute difference 0.0.** The floor recomputed in this run "
        "and the value read from the Phase 18 run are identical to "
        "sixteen digits, 0.5371587111538849. Two independent "
        "computations of the same quantity by different paths, which is "
        "what ``THE_FLOOR_RULED`` asked for. **The floor now exists as a "
        "banked figure**, which it did not before this phase"
    ),
}


#: **[MEASURED 2026-09-06] THE ANCHOR GATE.**
THE_ANCHOR_HELD = {
    "measured": "2026-09-06",
    "cell": "p26_cell_w2_p0",
    "declared": 0.2520,
    "observed": 0.2520460456337007,
    "verdict": (
        "**inside two of the probe's own seed sd.** The difference is "
        "0.0000460456 against a tolerance of 0.0296, so the cell that IS "
        "the probe reproduced the probe and the rest of the grid may be "
        "read"
    ),
    "and_one_seed_reproduces_the_banked_single_seed_row": (
        "**to eight decimals, not exactly, and the difference is "
        "recorded rather than rounded away.** The anchor's highest seed "
        "is 0.2719366102796185 against "
        "``train.phase3.GATE1_REFERENCE['pcc']`` at 0.2719366015264466, "
        "an absolute difference of 8.753e-09. Close enough to identify "
        "the same fit, far enough to say reproduces rather than equals"
    ),
}


#: **[CORRECTED 2026-09-06] PATIENCE IS INERT IN EFFECT, NOT UNREACHED,
#: AND THE TWO ARE NOT THE SAME FINDING.**
#:
#: The close read patience as inert in the way the budget is inert. It
#: is not, and the difference is worth more than the sameness.
PATIENCE_IS_INERT_IN_EFFECT_NOT_UNREACHED = {
    "corrected": "2026-09-06",
    "what_the_close_said": (
        "that the epoch axis is inert here exactly as the budget is, in "
        "the readings, the mechanism and the criterion walk. **The "
        "conclusion stands and the equivalence does not**"
    ),
    "how_the_budget_is_inert": (
        "**it never reaches the fit.** "
        "``phase7c.ROUND_2_IS_ROUND_1_TRUNCATED`` measured that the head "
        "resets on a fixed seed at a constant learning rate with no "
        "schedule keyed to max_epochs, so a given epoch's fit is "
        "independent of the budget. Raising it changes nothing that "
        "happens, and it costs nothing, because the loop stops on "
        "patience long before the budget matters"
    ),
    "how_patience_is_inert": (
        "**it reaches the fit and the fit obeys it.** A higher patience "
        "means the loop genuinely runs longer before it gives up, and it "
        "does. What does not move is the SELECTION, because the best "
        "checkpoint is the first epoch at every level "
        "(THE_SELECTED_EPOCHS_OBSERVED, 400 of 400). So the extra epochs "
        "are computed, evaluated, and then not chosen"
    ),
    "the_distinction_in_one_line": (
        "**the budget is inert because nothing happens. Patience is "
        "inert because what happens is not selected.** One is free and "
        "the other is paid for"
    ),
    "what_it_is_paid_for_is_REPORTED_not_read": (
        "**[REPORTED] the wall clock rises with patience, roughly "
        "fourfold from the lowest level to the highest within a weight "
        "decay group, while the predictions stay bit identical.** That "
        "reached this record through a message. **The seconds are not "
        "banked and criterion 6 is still open** "
        "(WHAT_CRITERION_3_DID_NOT_GET). The distinction above does not "
        "depend on the exact ratio, only on the direction, but a "
        "quantity this record states as its own must be read"
    ),
    "SUPERSEDED_2026_09_06_the_seconds_are_now_READ": (
        "**the entry above is preserved and its [REPORTED] tag no "
        "longer holds, and neither does the one in "
        "and_patience_is_INERT_IN_EFFECT_rather_than_UNREACHED.** The "
        "wall clock was read from the sixteen cell logs. **The "
        "roughly fourfold rise survives the reading at 4.395 on "
        "average**, and it holds in all four weight decay groups, so "
        "the reported direction and the reported size were both right "
        "(THE_RUNTIME_MEASURED). **What the reading adds is that the "
        "cost is linear in the epochs the loop runs rather than in "
        "patience itself**, which is the thing a later phase raising "
        "patience needs: it is buying epochs at a fixed price per "
        "epoch (THE_COST_OF_PATIENCE_MEASURED). The distinction this "
        "record drew never depended on the ratio, and it now rests on "
        "a measurement rather than on a direction"
    ),
    "why_the_correction_is_worth_making_before_the_seconds_arrive": (
        "**because the flattened version licenses a wrong inference.** "
        "'Inert like the budget' invites a reader to conclude that "
        "raising patience is free, and to raise it by default in a later "
        "phase. If the reported direction holds, that would buy four "
        "times the compute for a bit-identical result. **A phase that "
        "measures a null should not leave behind a reason to pay for "
        "it**"
    ),
    "and_it_sharpens_what_Phase_26_actually_established": (
        "the epoch dimension is closed on both settings, and for two "
        "DIFFERENT reasons, which is a better result than closing it on "
        "one reason twice. Anything that wants more fitting on this arm "
        "has to change what the selection sees rather than how long the "
        "loop waits"
    ),
}


#: **[MEASURED 2026-09-06] THE RUNTIME. Criterion 6 closes, and this is
#: the phase's own contribution, since no banked runtime for this shape
#: existed before it.**
#:
#: Read from ``p26_wall_clock.json``, sixteen cells by five seeds,
#: extracted from each cell run's own log where the training path writes
#: it. **The unit is SECONDS PER SEED at the top level and SECONDS PER
#: FOLD FIT below it**, five folds to a seed, and it is stated by
#: patience level rather than as one number because a single figure
#: hides the only thing the timing has to say.
THE_RUNTIME_MEASURED = {
    "measured": "2026-09-06, from the sixteen cell logs",
    "unit": (
        "**seconds per seed**, each seed being a five-fold "
        "cross-validated fit of a 769-parameter head over cached "
        "embeddings. The per fold fit figure is that divided by five"
    ),

    "seconds_per_seed_by_patience": {
        "5": 3.843, "10": 6.791, "20": 8.115, "30": 16.806,
    },
    "seed_sd_by_patience": {
        "5": 1.239, "10": 1.230, "20": 2.834, "30": 1.440,
    },
    "seconds_per_fold_fit_by_patience": {
        "5": 0.7686, "10": 1.3581, "20": 1.6230, "30": 3.3612,
    },
    "n_per_level": "20 seed runs, four weight decay groups by five seeds",

    "and_the_grid_as_a_whole": (
        "**711.1 seconds over 80 seed runs, which is 11.85 minutes of "
        "compute for the entire sixteen-cell design.** A single cell at "
        "five seeds ranges from 17.9 to 86.5 seconds"
    ),
    "tag": "[MEASURED] -- criterion 6 met, and the first runtime of this shape",
}


#: **[MEASURED 2026-09-06] WHAT THE TIMING ACTUALLY SHOWS, and it is not
#: what was reported.**
THE_COST_OF_PATIENCE_MEASURED = {
    "measured": "2026-09-06",

    "the_fourfold_figure_SURVIVES_and_is_promoted": (
        "**[MEASURED, was REPORTED] 4.395 on average, and it holds in "
        "every group.** Lowest patience to highest: 4.467 at weight "
        "decay 0.0, 3.913 at 0.001, 4.755 at 0.01, 4.444 at 0.1. The "
        "reported roughly fourfold was right"
    ),
    "but_it_is_NEITHER_a_ramp_NOR_a_step_in_patience": (
        "**the cost is linear in the epochs the loop actually runs, and "
        "patience sets those non-linearly.** With the best checkpoint at "
        "epoch 1 and the budget held at 30, a loop with patience p stops "
        "at min(1 + p, 30), so the four levels are 6, 11, 21 and 30 "
        "epochs. Fitting seconds against epochs on the two levels every "
        "group agrees on gives **seconds = 0.6023 + 0.5401 * epochs**, "
        "and that predicts **11 of the 16 cells to within 7 percent, "
        "12 within 10 and 13 within 15**. The three it misses are the "
        "three named below, and it misses them by half"
    ),
    "so_the_answer_to_smooth_or_stepped": (
        "**smooth in epochs, uneven in patience, and the unevenness is "
        "arithmetic rather than a threshold.** Nothing special happens "
        "at any patience level. Doubling patience from 5 to 10 costs 1.8 "
        "times because it nearly doubles the epochs, and 20 to 30 costs "
        "less than the gap suggests because the budget caps the loop at "
        "30 rather than 31. **A later phase raising patience is buying "
        "epochs at a fixed price per epoch, and should price it that "
        "way**"
    ),
    "the_fixed_overhead_is_small_and_real": (
        "0.6023 seconds before any epoch runs, which is 16 percent of "
        "the cheapest cell and 3.6 percent of the dearest. It is why the "
        "fourfold ratio sits below the 5.0 the epoch counts alone would "
        "give"
    ),
    "weight_decay_does_NOT_affect_the_cost": (
        "**at three of the four patience levels the spread across the "
        "whole hundredfold decay range is inside the seed noise.** "
        "Highest over lowest is 1.204 at patience 5, 1.079 at 10 and "
        "1.081 at 30, against a median within-cell relative seed sd of "
        "0.159. The one exception is patience 20 at 1.888, and that is "
        "the anomaly below rather than an effect of decay"
    ),

    "THE_ANOMALY_THREE_CELLS_COST_HALF_WHAT_THE_MODEL_PREDICTS": (
        "**stated as an anomaly rather than explained away.** The three "
        "patience-20 cells at non-zero weight decay cost 6.791, 6.567 "
        "and 6.701 seconds per seed against a predicted 11.945, ratios "
        "of 0.569, 0.550 and 0.561. **They match the ELEVEN-epoch "
        "prediction of 6.544 almost exactly**, at ratios 1.038, 1.004 "
        "and 1.024. The patience-20 cell at weight decay 0.0 fits the "
        "21-epoch prediction at 1.038 and is the only one that does"
    ),
    "and_it_is_not_seed_noise": (
        "**every seed in each of those three cells is near 6 seconds**, "
        "with within-cell sd around 1.0, while the 0.0 cell at the same "
        "patience is near 12 on every seed. The cells differ "
        "consistently rather than noisily"
    ),
    "what_it_would_mean_and_what_is_NOT_established": (
        "consistent with those three loops having stopped at epoch 11 "
        "rather than 21, which is to say having behaved as though "
        "patience were 10. **Whether that is what happened is not "
        "established here.** What is established is a discrepancy "
        "between declared patience and apparent epochs run, and settling "
        "it needs the per-epoch counts from those three runs, which this "
        "record does not have"
    ),
    "what_it_does_and_does_not_touch": (
        "**it touches no finding.** All 400 folds selected epoch 1 "
        "(THE_SELECTED_EPOCHS_OBSERVED) and every readout is bit "
        "identical within its weight decay group, so the result is the "
        "same whether those three cells ran 11 epochs or 21. **What it "
        "does touch is the completeness of the axis**: if the "
        "discrepancy is real then three of four groups tested patience "
        "at 5, 10, 10 and 30 rather than 5, 10, 20 and 30, and the "
        "20 level was exercised once rather than four times"
    ),
    "why_it_is_recorded_at_the_close_rather_than_chased": (
        "**the phase's conclusion does not depend on it and the record "
        "should not pretend the table is tidier than it is.** A reader "
        "who fits a line through these sixteen points will find three "
        "that miss, and the record naming them first is worth more than "
        "a record that averages them away"
    ),
}


#: **[MEASURED 2026-09-06] THE SCHEDULING CLAIM IS NOW MEASURED, AND IT
#: WAS GENEROUS.**
THE_SCHEDULING_CLAIM_SETTLED = {
    "measured": "2026-09-06",
    "what_was_claimed": (
        "the eighth amendment scheduled this phase on the ground that it "
        "runs on cached embeddings, and the ordering ruling put it first "
        "because it runs in minutes. The restate recorded that as an "
        "EXPECTATION and quoted the record's own position that an "
        "expectation is not a measurement"
    ),
    "what_is_measured": (
        "**a cell runs in SECONDS, not minutes.** Five seeds of the "
        "cheapest cell take 17.9 seconds and of the dearest 86.5. **The "
        "whole sixteen-cell grid is 11.85 minutes of compute.** So the "
        "claim is true of the design and generous by a factor of about "
        "ten for a cell"
    ),
    "the_expectation_it_settles": (
        "``phase20.COMPUTE_GATE_DESIGNED`` reasoned 'a 769-parameter "
        "head over 5 folds x 5 seeds. Expectation: seconds' and answered "
        "itself with 'the expectation is not the measurement'. **The "
        "expectation was right and it is now a measurement.** At the "
        "probe's own patience of 5 a seed costs 3.843 seconds and a fold "
        "fit 0.7686"
    ),
    "and_it_is_the_first_of_its_shape": (
        "**no banked runtime for a frozen head refit on cached "
        "embeddings existed in this record before.** The only banked per "
        "fit figure was a CleftGNN replication of 29,286,981 parameters, "
        "a different quantity under a similar name. This is the phase's "
        "own contribution and it outlives the null it was collected "
        "beside"
    ),
}


# --------------------------------------------------------------------------
# what the criteria did not get
# --------------------------------------------------------------------------

#: **[NOT MET 2026-09-06] TWO CRITERIA WERE NOT SATISFIED, AND BOTH ARE
#: DEFECTS IN THE CONSOLIDATION TASK RATHER THAN IN THE RUN.**
WHAT_CRITERION_3_DID_NOT_GET = {
    "recorded": "2026-09-06, at the close",

    "criterion_3_selected_epoch": (
        "**NOT MET. The task never emitted it.** Criterion 3 asks for "
        "``selected_epoch`` per cell per fold, and the metrics file "
        "carries no such key. What the file carries instead is the bit "
        "identity of every patience group, which supports the same "
        "conclusion more strongly and is not the thing that was asked "
        "for. **The criterion is recorded as unmet rather than as "
        "satisfied by a substitute**"
    ),
    "criterion_3_CLOSED_2026_09_06": (
        "**MET.** The entry above is preserved as the state at the "
        "close. The selected epoch was recovered from each cell run's "
        "own per-seed metrics rather than by re-running anything, and "
        "all 400 fold observations are epoch 1. **The task still does "
        "not emit it**, so the recovery was a read of the cell runs and "
        "not a fix to the consolidation. That remains owed"
    ),
    "criterion_6_wall_clock": (
        "**NOT MET, and the reason is worse than an omission.** The "
        "task reads ``summary.json`` from each run directory. **Nothing "
        "in this repository writes a file by that name.** The only "
        "occurrence of the string is the reader itself, so all sixteen "
        "entries came back null and the reader recorded null rather "
        "than raising"
    ),
    "and_the_restate_was_wrong_about_the_instrumentation": (
        "**the restate said the instrumentation exists and already "
        "runs, and that the measurement had been taken many times and "
        "never banked.** The first half is true and the second is "
        "understated. ``task_train_cv`` computes "
        "``wall_clock_seconds_by_seed`` and assigns it into a summary "
        "object that is rendered to the LOG and never written to a JSON "
        "artifact. **It is not merely unbanked, it is not readable from "
        "any file**, so no consolidation task could have recovered it. "
        "The claim in TIMING_IS_REPORTED is corrected by this entry"
    ),
    "the_shape_it_belongs_to": (
        "**a reused component called with something it does not "
        "produce**, which is the Phase 25 defect class exactly, and the "
        "silent-null branch is what let it pass unnoticed. A reader that "
        "records absence as a value rather than as a failure reports a "
        "clean sweep it did not run. The same lesson as "
        "``record_audit.THE_GREP_THAT_DID_NOT_RUN``"
    ),
    "criterion_6_STILL_OPEN_2026_09_06": (
        "**the wall clock did not arrive and criterion 6 is not "
        "closed.** It is in each cell run's LOG under "
        "``wall_clock_seconds_by_seed``, which is where the training "
        "path renders it and is exactly why the reader found nothing: "
        "it looked for a file this repository does not write. "
        "Recovering it is a read of sixteen logs and needs no re-run. "
        "**Until those seconds are read, the phase has no runtime of "
        "its own to bank**, and the fourfold rise with patience is "
        "[REPORTED] at PATIENCE_IS_INERT_IN_EFFECT_NOT_UNREACHED "
        "rather than measured here"
    ),
    "and_the_form_it_must_take_when_it_arrives": (
        "**seconds per seed BY PATIENCE LEVEL, not one number.** A "
        "single figure for the phase would hide the only thing the "
        "timing has to say, which is that the cost rises with a "
        "setting whose output does not move. The unit stays the fold "
        "fit, since a per-cell total hides whether seeds or folds "
        "carry it"
    ),
    "what_it_costs_and_what_it_does_not": (
        "**no finding depends on either.** The result rests on the "
        "sixteen cells and their identity structure, both read from the "
        "file. What is lost is the timing figure this phase existed "
        "partly to produce, so **no banked runtime for a frozen head "
        "refit on cached embeddings exists even now**, and the "
        "scheduling claim that this phase runs in minutes remains an "
        "expectation"
    ),
    "criterion_6_CLOSED_2026_09_06": (
        "**the wall clock arrived and criterion 6 is closed, so the "
        "three entries above are the state at the close and not the "
        "state now.** It was extracted from each cell run's own log "
        "under ``wall_clock_seconds_by_seed``, which is where this "
        "record said it would be, and it needed no re-run, which is "
        "what this record said as well. It is banked at "
        "THE_RUNTIME_MEASURED in the form this record fixed: **seconds "
        "per seed by patience level, never one number, with the fold "
        "fit beside it and the unit stated**"
    ),
    "and_what_it_COSTS_is_corrected_2026_09_06": (
        "**what_it_costs_and_what_it_does_not said no banked runtime "
        "for a frozen head refit on cached embeddings exists even now, "
        "and that the scheduling claim remains an expectation. Neither "
        "holds.** The runtime exists and is this phase's own "
        "contribution, the first of its shape in this record, and the "
        "scheduling claim is measured at THE_SCHEDULING_CLAIM_SETTLED. "
        "**The first half of that entry is untouched: no finding "
        "depended on the timing then and none depends on it now**, "
        "which is why the runtime is a contribution rather than a "
        "rescue"
    ),
    "and_the_defect_that_caused_it_is_NOT_fixed": (
        "**the reader still looks for ``summary.json`` and the "
        "training path still writes the timing only to the log.** The "
        "criterion was closed by reading the logs, not by repairing "
        "the task, exactly as criterion 3 was. **Two reads stand in "
        "for two fixes and both fixes remain owed**, and a later phase "
        "using this task will hit the same silent null unless it "
        "checks"
    ),
}


# --------------------------------------------------------------------------
# the closing
# --------------------------------------------------------------------------

#: **[CLOSED 2026-09-06] PHASE 26. The calibration this phase set out to
#: move is not reachable through these two settings on this arm.**
PHASE_26_CLOSING = {
    "closed": "2026-09-06, run p26_calibration_table at sha 58c7e847",

    "the_finding_CORRECTED_2026_09_06": (
        "**the sentence below said patience moves nothing at all, "
        "which flattens a distinction that matters.** Patience moves "
        "no OUTPUT. It does reach the fit, and the fit obeys it. See "
        "PATIENCE_IS_INERT_IN_EFFECT_NOT_UNREACHED"
    ),
    "the_finding": (
        "**NEITHER SETTING MOVES ANY READOUT BY AN AMOUNT THIS COHORT "
        "COULD RESOLVE.** Patience moves nothing at all, exactly, at "
        "every weight decay. Weight decay across a hundredfold range "
        "including its own absence moves the six readouts by at most "
        "0.0008438819, which is one patient in one seed. **The "
        "calibration this phase set out to move is not reachable "
        "through weight decay or patience on this arm**"
    ),
    "what_it_answers": (
        "**the standing request that IEM be as small as possible, "
        "directly and negatively.** IEM cannot be lowered by tuning "
        "these two settings. The grid's whole IEM range is 0.0000729411 "
        "against a distance to the constant-predictor floor of "
        "0.0089760770. **Anything that lowers IEM has to change the fit "
        "itself rather than its settings**, and this phase measured that "
        "rather than supposing it"
    ),
    "and_the_reason_is_measured_not_supposed": (
        "the mechanism is in the record and it is checkable: every "
        "patience level selects the same checkpoint, demonstrated by bit "
        "identity across all four levels at all four decay values, so "
        "the epoch axis cannot move and weight decay acts in one epoch "
        "(THE_MECHANISM_MEASURED)"
    ),

    "readings_that_fired": {
        "weight_decay_moves_nothing": (
            "**FIRED, and it was registered as a real answer rather than "
            "a failure.** The reading, written at the lock: 'the null, "
            "and it is a real answer. 0.01 is inert on this arm, the "
            "default was harmless, and one of the two settings the "
            "project never chose turns out not to matter'. The phase "
            "converted an unexamined default into a measured non-issue"
        ),
        "patience_does_NOT_move_the_selected_epoch": (
            "[The reading fired as written. Its phrase 'inert here "
            "exactly as the budget is' is corrected at "
            "PATIENCE_IS_INERT_IN_EFFECT_NOT_UNREACHED: the "
            "conclusion holds and the equivalence does not.] "
            "**FIRED, and it was written in advance as the phase's "
            "result on that axis rather than as a defect to work "
            "around.** The reading, written at the lock: 'the axis is "
            "inert here exactly as the budget is, and that is the "
            "phase's result on that axis rather than a defect. This "
            "would extend ROUND_2_IS_ROUND_1_TRUNCATED from the budget "
            "to patience, and it is worth having because it closes the "
            "epoch dimension by measurement rather than by assumption'. "
            "**That is what happened**"
        ),
        "and_which_did_NOT_fire": (
            "``weight_decay_moves_calibration_and_not_PCC`` did not "
            "fire, because nothing moved. "
            "``weight_decay_moves_PCC_too`` did not fire. "
            "``the_two_factors_interact`` did not fire, because one "
            "factor is exactly inert and cannot interact with anything. "
            "``the_anchor_cell_disagrees_with_the_banked_probe`` did not "
            "fire"
        ),
    },

    "criterion_walk": {
        "1_all_six_for_every_cell": (
            "MET. All sixteen cells carry PCC, shrinkage, RMSE, MAE, "
            "three-class accuracy and IEM, per seed and as means, from "
            "the same predictions. **This is the union no record in the "
            "project held before**, where the previous maximum "
            "co-occurrence was three at one seed"
        ),
        "2_the_floor_beside_every_IEM_figure": (
            "MET. The floor is carried on every cell and the cross-check "
            "against the recovered Phase 18 value agrees at an absolute "
            "difference of 0.0"
        ),
        "3_the_axis_is_shown_to_have_moved": (
            "**MET, from 2026-09-06.** [It read NOT MET at the close, "
            "because the consolidation task emitted no selected epoch "
            "and the bit identity was not the quantity asked for. The "
            "epoch was then extracted from each cell run's own per-seed "
            "metrics and read.] **400 fold observations across all "
            "sixteen cells, every one at epoch 1** "
            "(THE_SELECTED_EPOCHS_OBSERVED). The axis is shown not to "
            "have moved the selection, which the criterion recorded in "
            "advance as the phase's result on that axis"
        ),
        "4_the_prohibition_travels": (
            "MET. ``phase21.SCALE_INVARIANCE_PROHIBITION`` is carried "
            "whole in the run's own metrics file, and no reading here "
            "attributes PCC movement to a calibration change. There is "
            "no PCC movement to attribute"
        ),
        "5_the_IEM_defects_travel": (
            "MET. The three measured defects ride every IEM figure, and "
            "the comparator prohibition is observed. No IEM value from "
            "this phase is placed beside a published table"
        ),
        "6_per_fit_wall_clock_banked": (
            "**MET, from 2026-09-06.** [It read NOT MET at the close: "
            "all sixteen entries were null because the reader looked for "
            "a file nothing writes, while the training path renders the "
            "timing to the log.] The wall clock was read from the "
            "sixteen cell logs and banked by patience level in seconds "
            "per seed and per fold fit (THE_RUNTIME_MEASURED), with the "
            "relationship it shows at THE_COST_OF_PATIENCE_MEASURED"
        ),
        "7_one_factor_contrasts_only": (
            "MET, and vacuous in outcome. No contrast was computed, "
            "because no pair of cells differs by an amount worth "
            "testing. Nothing diagonal was compared"
        ),
        "8_a_ledger_row_only_if_claimable": (
            "MET, and vacuous. Nothing is claimable, so no row. **The "
            "reckoning predicted exactly this** and said so before the "
            "grid ran"
        ),
        "9_the_anchor_cell_reproduces_the_probe": (
            "MET. 0.2520460456337007 against the declared 0.2520, inside "
            "two of the probe's own seed sd (THE_ANCHOR_HELD)"
        ),
        "10_the_readings_were_written_at_the_lock": (
            "MET. ``READINGS`` is dated 2026-09-06 at the lock and the "
            "run is dated 2026-09-06 at sha 58c7e847, which is a later "
            "commit than the lock. Both readings that fired are quoted "
            "above in the words they were written in"
        ),
    },

    "no_ledger_row": (
        "**nothing here is claimable and none was expected.** The "
        "criterion is PLAN 4.3 unchanged, no contrast passes it, and the "
        "ledger stays at 38"
    ),
    "what_this_phase_does_NOT_close": (
        "**calibration as a question.** It closes two settings, not the "
        "subject. The head still shrinks to roughly half the truth's "
        "spread, which the grid moved by 0.0006488915 and did not fix. "
        "What changes calibration on this arm, if anything does, is a "
        "different fit rather than a different setting for the same one"
    ),
    "tag": "[CLOSED] -- two settings ruled out by measurement, no ledger row",
}


def summary() -> dict:
    """Every record in this module, for the restate sweep."""
    return {
        "reckoning": THE_RECKONING,
        "prohibition": THE_PROHIBITION,
        "may_not_claim": WHAT_THIS_PHASE_MAY_NOT_CLAIM,
        "second_factor": THE_SECOND_FACTOR_RULED,
        "axis_respecified": THE_AXIS_IS_RESPECIFIED,
        "floor": THE_FLOOR_RULED,
        "timing": TIMING_IS_REPORTED,
        "grid": THE_GRID_AS_A_DESIGN,
        "settings": SETTINGS_NEEDING_DECLARATION,
        "union": WHAT_THE_UNION_REQUIRES,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        "grid_locked": THE_GRID_LOCKED,
        "readings": READINGS,
        "exit_criteria": EXIT_CRITERIA,
        "cells_observed": THE_SIXTEEN_CELLS_OBSERVED,
        "mechanism": THE_MECHANISM_MEASURED,
        "iem_result": THE_IEM_RESULT,
        "anchor": THE_ANCHOR_HELD,
        "criteria_not_met": WHAT_CRITERION_3_DID_NOT_GET,
        "selected_epochs": THE_SELECTED_EPOCHS_OBSERVED,
        "patience_inert_in_effect": PATIENCE_IS_INERT_IN_EFFECT_NOT_UNREACHED,
        "runtime": THE_RUNTIME_MEASURED,
        "cost_of_patience": THE_COST_OF_PATIENCE_MEASURED,
        "scheduling_claim": THE_SCHEDULING_CLAIM_SETTLED,
        "closing": PHASE_26_CLOSING,
    }
