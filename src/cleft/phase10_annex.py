"""Phase 10 annex -- the split-distribution replication.

Opened 2026-08-30 , who ruled the design; this module is the
registration and nothing else. **Nothing is built this turn**: no task,
no schema change, no config, no run. The compute gate (a one-split x
one-rater full-fit job, designed below) must land and the maintainer must rule
on the draw count N and the normalisation before the exit criteria lock.

Pointers both ways: ``phase10.PHASE_10_CLOSING["annex_2026_08_30"]``
points here; every record below cites the Phase 10 record it rests on.
Every figure in this module is QUOTED from a named record, never
retyped; where a figure had to be derived (153) or arrived from outside
the repo (the manuscript's panel wording, the statement given at supervision about the 28),
the record says so explicitly.
"""

from __future__ import annotations


#: **[REGISTERED 2026-08-30] THE ANNEX QUESTION.**
#:
#: Phase 10 measured CleftGNN's architecture under OUR protocol:
#: ``phase10.PAIRED_BCA_OBSERVED["arms_as_declared"]["p10_cleftgnn"]``
#: is ``{"mean": 0.0242, "sd": 0.0676}``, and the closing quotes the
#: same figure ("PCC +0.0242 (seed sd 0.0676) over 25 folds",
#: ``PHASE_10_CLOSING["cites_only_what_is_recorded"]["manuscript_cell"]``).
#: The run-level record ``ROUTE_3_OBSERVED`` carries 0.0241 from
#: ``p10-cleftgnn-5``; the 0.0001 is itself recorded
#: (``PAIRED_BCA_OBSERVED["the_0_0001"]``) and both figures are cited
#: here so neither is silently preferred.
#:
#: Phase 10 never reproduced THEIR protocol -- the faithful arm ran
#: their protocol's shape once (single run, their form,
#: ``FAITHFUL_ARM_REGISTERED``) and closed not-claimable by
#: construction. The annex fixes label, metric, architecture, and
#: recipe to theirs and varies ONLY the split:
#:
#:     **does a single 153/28 split identify a model's performance at
#:     this cohort size?**
ANNEX_QUESTION_REGISTERED = {
    "registered": "2026-08-30 -- design ruled, registration only",
    "question": (
        "does a single 153/28 split identify a model's performance at "
        "this cohort size?"
    ),
    "what_phase_10_measured": (
        "CleftGNN's architecture under OUR protocol: 0.0242 (sd 0.0676), "
        "quoted from PAIRED_BCA_OBSERVED['arms_as_declared']"
        "['p10_cleftgnn'] and from PHASE_10_CLOSING's "
        "cites_only_what_is_recorded. ROUTE_3_OBSERVED recorded 0.0241 "
        "from p10-cleftgnn-5; the 0.0001 is recorded in "
        "PAIRED_BCA_OBSERVED['the_0_0001'], both figures carried here"
    ),
    "what_phase_10_never_did": (
        "reproduce their protocol: the faithful arm ran their form ONCE "
        "(FAITHFUL_ARM_REGISTERED, single run, no intervals -- their "
        "form, recorded not repaired) and closed not-claimable by "
        "construction (FAITHFUL_ARM_CLOSING)"
    ),
    "what_varies": "the split, and ONLY the split -- everything else theirs",
}


#: **[VERIFICATION NOTES 2026-08-30 -- what was checked against which
#: record before it entered this registration, and what could NOT be.]**
VERIFICATION_NOTES = {
    "noted": "2026-08-30, every figure verified at source before entry",
    "quoted_clean": (
        "0.0242/0.0676 (PAIRED_BCA_OBSERVED, PHASE_10_CLOSING); 0.0241 "
        "(ROUTE_3_OBSERVED via the_0_0001); '0.440 to -0.162, mean "
        "~0.14' (CLEFTGNN_COMPARATOR_TABLES['study_test_set'], Table 1, "
        "n=28, each rater's own labels); 36-vs-27 regions "
        "(REGION_COUNT_BELIEF_VS_EXECUTION); 'LR = 0.001; "
        "optim.Adam(trainable_params, lr=LR); BATCH_SIZE = 16; EPOCHS = "
        "5; nn.CrossEntropyLoss()' (NOTEBOOK_RECIPE_IS_ADAM, notebook "
        "cell 9); CIFAR-10 statistics as a defect "
        "(NOTEBOOK_BUDGET_AND_NORMALISATION); the frozen-backbone fact "
        "(BACKBONE_IS_FROZEN); the config census "
        "(phase15.CRITERION_1_AMENDED); our five rater columns "
        "(data/scoresheet.py RATERS)"
    ),
    "derived_not_recorded": (
        "**153 appears nowhere in the record.** It is arithmetic: the "
        "cohort their split divides is 181 images "
        "(THIN_CLASS_SMOOTHING['manuscript_silence']: 'their 181-image "
        "85:15 split') and their Study Test Set is n=28 "
        "(CLEFTGNN_COMPARATOR_TABLES, Table 1), so train = 181 - 28 = "
        "153. 85:15 of 181 is 153.85/27.15; the recorded test n=28 "
        "fixes the rounding. Stated as a derivation, not a quote"
    ),
    "arrived_from_outside_the_repo": (
        "TWO items are in no repo record and enter tagged: (1) the "
        "manuscript's panel wording 'surgeon + two orthodontists + SLT "
        "+ psychologist' -- [MANUSCRIPT, via the maintainer 2026-08-30]; (2) "
        "the statement given at supervision that the 28 test images come from the cohort "
        "-- [REPORTED, not measured] (SUPERVISION_28_DEPENDENCY). Their A-E "
        "rater naming IS recorded (FAITHFUL_ARM_REGISTERED's 'rater-E "
        "model', the Phase 10 handoff notes' Rater A..E table)"
    ),
    "census_remeasured": (
        "the record's census (CRITERION_1_AMENDED, 2026-08-24): head "
        "(81), graph_layers (34), classifier (2), classifier_adabn (2) "
        "= 119, zero 'full'. Re-measured 2026-08-30 over "
        "configs/*.yaml: 314 files, 143 'trainable:' occurrences -- "
        "head 102, graph_layers 35, classifier 3, classifier_adabn 3, "
        "**still zero 'full'**. Both counts carried; the claim rests on "
        "today's measurement, the record is its history"
    ),
    # [CORRECTED 2026-08-30, second cycle -- the original above is
    # PRESERVED as written, and it is WRONG in a specific way worth
    # keeping visible.] The two censuses are DIFFERENT QUANTITIES, and
    # putting them side by side as "both counts carried" implied they
    # measured the same thing. MEASURED cause: my regex matched
    # 'trainable:' ANYWHERE in a file, so it counted 24 HEADER-COMMENT
    # lines that MENTION a setting ("`trainable: head` is unchanged")
    # as though they SET one. The record's rule -- a line beginning
    # with two spaces and 'trainable: ' -- counts what configs SET, and
    # it is the correct instrument.
    #
    # Re-measured under the record's own rule, 2026-08-30: head 81,
    # graph_layers 34, classifier 2, classifier_adabn 2 -- IDENTICAL to
    # the recorded census -- plus 'full' 1, the annex gate's own config
    # and the first in the project's history. So the record's census
    # never drifted; only my instrument was loose.
    #
    # Error class: comparing two quantities under one name (R2), the
    # tenth instance in this project's ledger of them, and the same
    # shape as PROBE_RECONSTRUCTED_THE_PIPELINE -- a second
    # implementation of one measurement that silently disagreed with
    # the first. The fix is the same: ONE rule, used by both the record
    # and the test (tests/test_phase15.py and tests/test_phase10_annex.py
    # now share it).
    "census_correction_2026_08_30": (
        "the 143/102/35/3/3 figures above are a DIFFERENT QUANTITY from "
        "the record's 119: that regex counted 24 header-COMMENT "
        "mentions as settings. Under the record's own rule (a line "
        "starting '  trainable: ') the census is head 81, graph_layers "
        "34, classifier 2, classifier_adabn 2 -- identical to the "
        "recorded one -- plus 'full' 1, the annex gate. The record "
        "never drifted; the instrument was loose. R2's tenth instance"
    ),
    "no_stratification_is_a_silence": (
        "'their paper describes no stratification' is verified as a "
        "SILENCE, not a sentence: the record has the manuscript 'SILENT "
        "on thin classes despite their 181-image 85:15 split facing the "
        "same problem -- five grades over 181 images, with the extremes "
        "necessarily thin, split 85:15 with no mention of what happens "
        "to a class that lands entirely on one side' "
        "(THIN_CLASS_SMOOTHING and its doc-comment). The faithful arm's "
        "stratification was OUR registered addition "
        "(FAITHFUL_ARM_REGISTERED: '85:15 stratified on the rater's own "
        "grade, seeded 1337'), not theirs -- so the annex draws PLAIN "
        "random splits and reports what that yields"
    ),
}


#: **[REGISTERED 2026-08-30] DESIGN, FIXED TO THEIRS -- each fidelity
#: choice cited to its record.**
#:
#: **Regions: 36.** What their code ran and what produced every
#: published number: "EVERY result in the paper was produced with 36
#: regions; the reported architecture has never been run"
#: (``phase10.REGION_COUNT_BELIEF_VS_EXECUTION`` -- 27 stated in three
#: places, 36 executed with a printed warning at every construction).
#: **27 is explicitly UNREGISTERED** with that reason: an enumeration
#: yielding 27 exists nowhere in their artifacts and produced none of
#: their results.
#:
#: **Label: each rater's own integer grade, five separate models per
#: split** -- their design (``FAITHFUL_ARM_REGISTERED["protocol"]``:
#: "rater-specific, one per rater, five models").
#:
#: **Splits: 153/28 at random, plain.** 153 derived, 28 quoted
#: (VERIFICATION_NOTES). No stratification, because their paper
#: describes none (verified as a silence, VERIFICATION_NOTES).
#: **Degenerate draws are a finding, never a discard**: a draw whose
#: test side is missing one or more of the five grades for that rater
#: breaks 5-class macro-averaging over five classes
#: (``phase10.top1_macro_prf``'s own convention: "a class with no
#: truths is EXCLUDED from the macro average"). How often the design
#: yields an unusable split is reported as a per-rater FREQUENCY, and
#: metrics on such draws are reported under the stated convention
#: rather than the draw being silently redrawn.
DESIGN_FIXED_TO_THEIRS = {
    "registered": "2026-08-30",
    "regions": {
        "value": 36,
        "cited": (
            "REGION_COUNT_BELIEF_VS_EXECUTION: 'EVERY result in the "
            "paper was produced with 36 regions; the reported "
            "architecture has never been run'"
        ),
        "unregistered_27": (
            "27 is explicitly UNREGISTERED: stated in three places, "
            "executed nowhere, produced none of their results"
        ),
    },
    "label": {
        "value": "each rater's own integer grade, 1-5",
        "models": "five separate models per split, one per rater",
        "cited": (
            "FAITHFUL_ARM_REGISTERED['protocol']['models']: "
            "'rater-specific, one per rater, five models'"
        ),
    },
    "splits": {
        "value": "153/28 at random, plain -- no stratification",
        "derivation": "181 - 28 = 153 (VERIFICATION_NOTES)",
        "degenerate_draws": (
            "REPORTED AS A PER-RATER FREQUENCY, NEVER DISCARDED: a "
            "test side missing a grade breaks macro-averaging over "
            "five classes, and how often the design yields an "
            "unusable split is itself a finding. Metrics on such "
            "draws follow top1_macro_prf's stated convention (truth-"
            "absent classes excluded, stated because it changes the "
            "number)"
        ),
    },
    "metrics": {
        "value": (
            "5-class Top-1 macro precision / recall / F1, per-rater "
            "PCC, IEM"
        ),
        "cited": (
            "FAITHFUL_ARM_REGISTERED['protocol']['metrics'] ('Top-1 "
            "macro precision, recall, F1, and PCC'); IEM through "
            "phase11.iem under convention A "
            "(phase11.IEM_DIRECTION_ANSWERED)"
        ),
        "five_class_legitimacy": "FIVE_CLASS_SCOPED_EXCEPTION",
    },
    "floors": "FLOORS_REGISTERED",
}


#: **[REGISTERED 2026-08-30] WHY 5-CLASS IS LEGITIMATE HERE, WHERE
#: ``classification.THREE_NOT_FIVE`` FORBIDS IT EVERYWHERE ELSE.**
#:
#: THREE_NOT_FIVE's ground is that our arms predict a CONTINUOUS value
#: and a 4-threshold rule for collapsing it to five classes is "NOWHERE
#: IN THE RECORD" -- using one would mean inventing thresholds and
#: reporting numbers from them in the same run. **The annex does not
#: touch that ground**: its target is an integer 1-5 grade, its loss is
#: cross-entropy over five classes (the notebook's own
#: ``nn.CrossEntropyLoss()``, NOTEBOOK_RECIPE_IS_ADAM), and argmax
#: yields the class DIRECTLY -- no threshold is invented anywhere.
#: This is the one context where our 5-class figures are constructed
#: the same way theirs are.
#:
#: The scoped exception is DATED BESIDE THREE_NOT_FIVE
#: (``classification.THREE_NOT_FIVE["scoped_exception_2026_08_30"]``),
#: and it does not reopen the declined candidate: the 2026-08-30
#: decline (a 4-threshold rule for continuous predictions,
#: ``phase18.PHASE_18_RULINGS["three_not_five_declined"]``) stands
#: untouched, because no continuous prediction is collapsed here.
FIVE_CLASS_SCOPED_EXCEPTION = {
    "registered": "2026-08-30",
    "why_legitimate_here": (
        "the target is an integer 1-5 grade and cross-entropy yields "
        "an argmax directly -- no thresholds invented; the one context "
        "where our 5-class figures are constructed the same way theirs "
        "are"
    ),
    "why_forbidden_elsewhere": (
        "THREE_NOT_FIVE: collapsing a CONTINUOUS prediction to five "
        "classes needs a 4-threshold rule that is 'NOWHERE IN THE "
        "RECORD' -- a different situation, untouched by this exception"
    ),
    "pointer_beside_the_rule": (
        "classification.THREE_NOT_FIVE['scoped_exception_2026_08_30']"
    ),
    "the_decline_stands": (
        "phase18.PHASE_18_RULINGS['three_not_five_declined'] is not "
        "reopened: no continuous prediction is collapsed in this annex"
    ),
}


#: **[RECORDED 2026-08-30] THE RATER PANEL IS THE EIGHTH
#: MANUSCRIPT-VS-ARTIFACT DISCREPANCY.**
#:
#: ``PHASE_10_CLOSING`` closed on "seven manuscript-vs-artifact
#: discrepancies plus one blocking ambiguity" (the list's eighth ENTRY
#: is the ambiguity, since answered -- convention A). This is the
#: eighth DISCREPANCY of that class, recorded here dated:
#:
#:     manuscript   surgeon + two orthodontists + SLT + psychologist
#:                  [MANUSCRIPT, via the maintainer 2026-08-30 -- this
#:                  wording is in NO repo record]
#:     our sheet    Rater 7 Cleft patient / Rater 8 Orthodontist /
#:                  Rater 9 Speech and language therapist / Rater 10
#:                  Plastic surgeon / Rater 11 Psychologist
#:                  (data/scoresheet.py RATERS, verified across all
#:                  1,255 cells)
#:
#: The compositions do not agree (two orthodontists vs one; no cleft
#: patient in theirs), so **rater identities cannot be matched to
#: their A-E** (their naming: CLEFTGNN_COMPARATOR_TABLES' Study Test
#: Set "each rater's own", FAITHFUL_ARM_REGISTERED's "rater-E model",
#: the Phase 10 handoff notes' Rater A..E table). The annex therefore
#: reports **five unmatched raters, stated plainly** -- per-rater
#: figures carry sheet identities (Rater 7..11), never a claimed
#: correspondence to their A-E, and the caveat rides on every
#: per-rater quote.
RATER_PANEL_DISCREPANCY_EIGHTH = {
    "recorded": "2026-08-30",
    "class": (
        "the eighth manuscript-vs-artifact discrepancy, extending "
        "PHASE_10_CLOSING's 'seven manuscript-vs-artifact "
        "discrepancies plus one blocking ambiguity'"
    ),
    "manuscript_panel": (
        "surgeon + two orthodontists + SLT + psychologist "
        "[MANUSCRIPT, via the maintainer 2026-08-30 -- in no repo record]"
    ),
    "our_sheet_panel": (
        "cleft patient / orthodontist / speech and language therapist "
        "/ plastic surgeon / psychologist (data/scoresheet.py RATERS)"
    ),
    "consequence": (
        "rater identities CANNOT be matched to their A-E; the annex "
        "reports five UNMATCHED raters, stated plainly, and the "
        "caveat rides on every per-rater quote"
    ),
    # [PROVENANCE NOTE 2026-08-30, the maintainer] The label "discrepancy
    # #8" originates in the LITERATURE RECORD
    # (LITERATURE_RECORD_2026-08-23.md section 1, CleftGNN entry:
    # "Panel description conflict (discrepancy #8)") -- a numbering
    # INDEPENDENT of PHASE_10_CLOSING's seven-plus-ambiguity count.
    # The two schemes agree on the substance and happen to agree on
    # the ordinal; neither derives from the other, and this note
    # exists so they cannot be read as conflicting. The "eighth
    # discrepancy of that class" framing above stands unchanged.
    # [VERIFICATION 2026-08-30: that file was NOT FOUND on this
    # machine -- the handoff directory outside the repository, the repo
    # itself, and docs/ all searched -- so
    # the section-1 wording is quoted SECOND-HAND rather than
    # verified at source, tagged accordingly.]
    "number_provenance_2026_08_30": (
        "the '#8' label is the literature record's "
        "(LITERATURE_RECORD_2026-08-23.md section 1, CleftGNN entry: "
        "'Panel description conflict (discrepancy #8)' [via the maintainer "
        "-- file not found on this machine]), a numbering independent "
        "of PHASE_10_CLOSING's seven-plus-ambiguity count; the two "
        "schemes agree on substance and coincide on the ordinal"
    ),
}


#: **[REGISTERED 2026-08-30] FLOORS BESIDE EVERY METRIC.**
FLOORS_REGISTERED = {
    "registered": "2026-08-30",
    "classification_floor": (
        "the majority-class predictor, computed on each draw's own "
        "training side and evaluated on its test side, reported beside "
        "every accuracy / macro precision / recall / F1 figure -- the "
        "project's standing floor discipline "
        "(classification.CLASSIFICATION_METRICS_SECONDARY['the_floor'])"
    ),
    "iem_floor": (
        "the constant-predictor floor, computed through phase11.iem "
        "under convention A (phase11.IEM_DIRECTION_ANSWERED), reported "
        "beside every IEM figure"
    ),
    "pcc_floor": (
        "a constant predictor has NO PCC (zero variance -- undefined, "
        "not zero); recorded as such rather than a number being "
        "invented for the column"
    ),
}


#: **[RULED 2026-08-30] RULING (a): ALL FIVE RATERS, FIVE
#: MODELS PER SPLIT.**
#:
#: The across-rater spread is PART OF WHAT IS CHARACTERISED, not noise
#: to average away: their own per-rater Study-Set PCCs run "0.440 to
#: -0.162, mean ~0.14"
#: (``CLEFTGNN_COMPARATOR_TABLES["study_test_set"]["range"]``, Table
#: 1, n=28, each rater's own labels -- verified at the record).
RULING_A_ALL_FIVE_RATERS = {
    "ruled": "2026-08-30",
    "ruling": "all five raters, five models per split",
    "why": (
        "the across-rater spread is part of what is characterised: "
        "their own per-rater PCCs run '0.440 to -0.162, mean ~0.14' "
        "(CLEFTGNN_COMPARATOR_TABLES['study_test_set'], Table 1, n=28)"
    ),
}


#: **[RULED 2026-08-30] RULING (b): FULL TRAINABLE RESNET-50
#: -- A SCOPED DEPARTURE, NOT A GENERAL UNLOCK.**
#:
#: The facts around the ruling, each at its record:
#:
#: * **Zero configs use ``trainable: full``.** The record:
#:   ``phase15.CRITERION_1_AMENDED`` -- "no shipped config in this
#:   project has ever set ``trainable: full``": head (81),
#:   graph_layers (34), classifier (2), classifier_adabn (2) = 119.
#:   Re-measured 2026-08-30: 314 configs, 143 ``trainable:``
#:   occurrences, still zero ``full`` (VERIFICATION_NOTES).
#: * **Criterion 1 withdrew fine-tuning as never-operationalised**
#:   (the same record, ruled 2026-08-24 : "the fine-tuning
#:   sense was never operationalised ANYWHERE").
#: * **The notebook freezes its backbone** (``BACKBONE_IS_FROZEN``:
#:   "the notebook, the only executable artifact the group ran,
#:   freezes its backbone; lr 0.01 belongs to that regime"). The
#:   annex KNOWINGLY departs from the notebook on trainability -- the
#:   departure is the ruling, recorded with its reason, not an
#:   oversight.
#: * **The manuscript states ResNet-50** (Table 1) where the notebook
#:   executed ViT-B/16 -- discrepancy 4 of
#:   ``PHASE_10_CLOSING["criterion_1_discrepancies"]``. The ruling
#:   takes the stated ResNet-50.
#:
#: **The record this ruling writes**: fine-tuning is operationalised
#: IN THIS ANNEX ONLY, for fidelity, and remains unregistered
#: everywhere else. **Reason (the maintainer's)**: a frozen backbone would
#: leave "we froze it" as a permanent alternative explanation for any
#: gap, defeating the annex's purpose.
#:
#: **The schema/config surface consequences** -- a ``trainable: full``
#: vocabulary scoped to the annex kind -- are REPORTED AS DESIGN and
#: are NOT built this turn.
RULING_B_FULL_TRAINABLE = {
    "ruled": "2026-08-30",
    "ruling": (
        "full trainable ResNet-50 -- a scoped departure, not a "
        "general unlock"
    ),
    "scope": (
        "fine-tuning is operationalised in this annex ONLY, for "
        "fidelity, and remains unregistered everywhere else"
    ),
    "reason": (
        "a frozen backbone would leave 'we froze it' as a permanent "
        "alternative explanation for any gap, defeating the annex's "
        "purpose"
    ),
    "census": (
        "zero of 119 shipped configs used trainable: full at the "
        "record's census (phase15.CRITERION_1_AMENDED: head (81), "
        "graph_layers (34), classifier (2), classifier_adabn (2)); "
        "re-measured 2026-08-30, still zero of the current 143 "
        "occurrences across 314 configs"
    ),
    "withdrawal_respected": (
        "CRITERION_1_AMENDED withdrew the fine-tuning sense as "
        "never-operationalised (2026-08-24, the maintainer); this ruling "
        "operationalises it in one named scope rather than reopening "
        "it generally"
    ),
    "notebook_departure_owned": (
        "BACKBONE_IS_FROZEN: the notebook freezes its backbone. The "
        "annex departs from the notebook HERE, knowingly, by ruling"
    ),
    "backbone_choice": (
        "the manuscript's stated ResNet-50 (Table 1), where the "
        "notebook executed ViT-B/16 -- discrepancy 4 of "
        "PHASE_10_CLOSING['criterion_1_discrepancies']; taken as ruled"
    ),
    "schema_surface": (
        "a 'trainable: full' vocabulary scoped to the annex kind -- "
        "REPORTED AS DESIGN, NOT BUILT THIS TURN"
    ),
    # [BUILT 2026-08-30, second cycle, the instruction -- the
    # design sentence above is preserved as written.] The annex kind
    # p10x_gate_fullfit exists with trainable choices ("full",) --
    # full REQUIRED there, and the model refuses full on any other
    # recipe (models/cleftgnn.py). MEASURED CORRECTION to the
    # instruction's premise, recorded rather than smoothed: "full"
    # was ALREADY admitted (though never set by any config) in three
    # historical vocabularies -- train_cv, partition_sensitivity,
    # probe_mebeauty -- so "refused by every other kind" was never
    # the schema's state; the structural pin is the ENUMERATED
    # admits-full set in tests/test_phase10_annex.py, which any new
    # kind admitting 'full' fires.
    "schema_surface_built_2026_08_30": (
        "p10x_gate_fullfit shipped with trainable choices ('full',); "
        "the admits-full vocabulary set is pinned by enumeration "
        "(train_cv, partition_sensitivity, probe_mebeauty admitted it "
        "historically, zero configs ever set it before the gate's own)"
    ),
}


#: **[REGISTERED 2026-08-30] RECIPE: WHAT THEIR CODE RAN.**
#:
#: Quoted from ``NOTEBOOK_RECIPE_IS_ADAM`` (verified 2026-08-17,
#: notebook cell 9): "LR = 0.001; optim.Adam(trainable_params, lr=LR);
#: BATCH_SIZE = 16; EPOCHS = 5; nn.CrossEntropyLoss()" -- and the
#: budget's own shape from ``NOTEBOOK_BUDGET_AND_NORMALISATION``:
#: "EPOCHS = 5, NO validation split, no early stopping, no checkpoint
#: selection -- the notebook's last epoch IS its model". The annex
#: takes the whole of it, on the same grounds as the 36 regions: what
#: their code ran is what produced every published number.
#:
#: **The manuscript's SGD 0.01 is explicitly UNREGISTERED** (stated,
#: never executed -- discrepancy 2, "SGD 0.01 stated, Adam 0.001
#: executed").
#:
#: Because the notebook budget is taken whole, none of Phase 10's
#: registered deviations (early stopping, Laplace CE init, LayerNorm
#: fixes -- ``REGISTERED_DEVIATIONS``) rides into the annex: they
#: belonged to the manuscript cell and to arms that needed to start
#: calibrated. The annex is the notebook's recipe, run as written.
RECIPE_FIXED_TO_NOTEBOOK = {
    "registered": "2026-08-30",
    "recipe_quoted": (
        "LR = 0.001; optim.Adam(trainable_params, lr=LR); BATCH_SIZE "
        "= 16; EPOCHS = 5; nn.CrossEntropyLoss() "
        "(NOTEBOOK_RECIPE_IS_ADAM, notebook cell 9)"
    ),
    "budget_quoted": (
        "EPOCHS = 5, NO validation split, no early stopping, no "
        "checkpoint selection -- the notebook's last epoch IS its "
        "model (NOTEBOOK_BUDGET_AND_NORMALISATION)"
    ),
    "grounds": (
        "the same as the 36 regions: what their code ran is what "
        "produced every published number"
    ),
    "unregistered_sgd": (
        "the manuscript's SGD 0.01 is explicitly UNREGISTERED -- "
        "'SGD 0.01 stated, Adam 0.001 executed' (discrepancy 2, "
        "PHASE_10_CLOSING['criterion_1_discrepancies'])"
    ),
    "phase_10_deviations_do_not_ride": (
        "REGISTERED_DEVIATIONS belonged to the manuscript cell; the "
        "annex takes the notebook budget whole, so none of them "
        "apply here"
    ),
}


#: **[PROPOSED 2026-08-30 -- NOT LOCKED, FLAGGED FOR A RULING]
#: NORMALISATION: REPRODUCE THE CIFAR-10 STATISTICS, AS A DELIBERATELY
#: REPLICATED DEFECT.**
#:
#: The record characterises it as a DEFECT, not a recipe choice --
#: ``NOTEBOOK_BUDGET_AND_NORMALISATION["why_we_call_it_a_defect"]``
#: ("Nothing is gained by it and it is discussed nowhere"), and
#: ``models/cleftgnn.py``'s own docstring: "those are a defect in the
#: notebook, not a recipe choice". Phase 10's notebook cell
#: deliberately did NOT replicate it, for that stated reason.
#:
#: The annex's fidelity principle argues the other way: the published
#: numbers were produced UNDER the defect, so reproducing their
#: protocol means reproducing it, recorded as a deliberately
#: replicated defect. **The two positions are both on the record and
#: point in opposite directions, which is exactly why this is a maintainer
#: decision**: PROPOSED here, flagged in the registration report,
#: and it does not enter the exit-criteria lock until it is confirmed.
NORMALISATION_PROPOSED_NOT_LOCKED = {
    "proposed": "2026-08-30 -- NOT LOCKED, awaiting confirmation",
    "proposal": (
        "transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, "
        "0.1994, 0.2010)) -- the notebook's CIFAR-10 statistics, "
        "quoted from NOTEBOOK_BUDGET_AND_NORMALISATION, reproduced as "
        "a DELIBERATELY REPLICATED DEFECT per the fidelity principle"
    ),
    "the_record_calls_it_a_defect": (
        "NOTEBOOK_BUDGET_AND_NORMALISATION['why_we_call_it_a_defect'] "
        "-- 'Nothing is gained by it and it is discussed nowhere'; "
        "models/cleftgnn.py: 'those are a defect in the notebook, not "
        "a recipe choice'"
    ),
    "the_tension_stated": (
        "Phase 10's notebook cell deliberately did NOT replicate it; "
        "the annex's fidelity principle argues for replicating it. "
        "Both positions recorded; the choice is the maintainer's, before "
        "the lock"
    ),
    # [RULED 2026-08-30, the maintainer, same day -- the proposal above is
    # PRESERVED as written; the ruling is NORMALISATION_RULED.]
    "ruled_2026_08_30": "NORMALISATION_RULED",
}


#: **[RULED 2026-08-30] NORMALISATION: THE ANNEX REPRODUCES
#: THE NOTEBOOK'S CIFAR-10 STATISTICS, RECORDED AS A DELIBERATELY
#: REPLICATED DEFECT.**
#:
#: **The grounds, the maintainer's**: the annex's question is whether the
#: SPLIT explains their numbers, so every other factor must be what
#: actually produced those numbers -- defects included. Correct
#: ImageNet statistics would open "our normalisation differed" as a
#: permanent alternative explanation for any gap -- the same failure
#: mode ruling (b) closed for the backbone.
#:
#: The defect record's own characterisation rides beside the ruling,
#: unchanged: "a defect in the notebook, not a recipe choice"
#: (``models/cleftgnn.py``;
#: ``NOTEBOOK_BUDGET_AND_NORMALISATION["why_we_call_it_a_defect"]``).
#: **The write-up sentence is "replicated knowingly," never "a
#: recipe."**
NORMALISATION_RULED = {
    "ruled": "2026-08-30",
    "ruling": (
        "the annex reproduces the notebook's CIFAR-10 statistics -- "
        "(0.4914, 0.4822, 0.4465) / (0.2023, 0.1994, 0.2010), quoted "
        "from NOTEBOOK_BUDGET_AND_NORMALISATION -- recorded as a "
        "DELIBERATELY REPLICATED DEFECT"
    ),
    "grounds": (
        "the annex's question is whether the split explains their "
        "numbers, so every other factor must be what actually "
        "produced those numbers -- defects included; correct ImageNet "
        "stats would open 'our normalisation differed' as a permanent "
        "alternative explanation for any gap, the same failure mode "
        "ruling (b) closed for the backbone"
    ),
    "characterisation_rides": (
        "'a defect in the notebook, not a recipe choice' -- the "
        "defect record's own words travel beside every quote of the "
        "annex's normalisation"
    ),
    "write_up_sentence": "replicated knowingly -- never 'a recipe'",
}


#: **[COMMITTED 2026-08-30, BEFORE ANY NUMBER] THE TWO READINGS.**
READINGS_COMMITTED = {
    "committed": "2026-08-30, before any number exists",
    "published_inside_the_distribution": (
        "single-split reporting at n=28 does not identify performance "
        "-- the strongest form of the small-n argument the project "
        "has, measured on their own design"
    ),
    "published_outside_and_above": (
        "the split does not explain the gap; the replication is "
        "INCOMPLETE -- unmatched raters "
        "(RATER_PANEL_DISCREPANCY_EIGHTH), 181-vs-237 "
        "(SUPERVISION_28_DEPENDENCY), or a recipe detail become candidates, "
        "and the annex says so rather than concluding anything about "
        "their model"
    ),
    "no_third_reading": (
        "whichever lands, no reading is invented after the numbers"
    ),
}


#: **[REGISTERED 2026-08-30] THE PROHIBITION -- the CleftGNN-IEM
#: prohibition's sibling
#: (``phase18.DELIVERABLES_REGISTERED["cleftgnn_iem_prohibition"]``),
#: same form: a literal, tested, so no later turn reaches for the
#: comparison.**
ANNEX_PROHIBITION = (
    "THESE FIGURES CHARACTERISE OUR REPLICATION UNDER THEIR PROTOCOL "
    "-- NEVER A RE-COMPUTATION OF THEIR RESULTS, NEVER PRESENTED AS "
    "CORRECTING THEIR PUBLISHED NUMBERS. Ours: our cohort's images "
    "and sheet (five unmatched raters), our split draws, our fits. "
    "Theirs: their images, their raters A-E, their one split, their "
    "fits. A distribution from the former brackets nothing about the "
    "latter; it characterises the DESIGN, and the annex says only "
    "that. The existing IEM prohibition stands beside this one "
    "unchanged: our IEM values are never placed beside CleftGNN's "
    "Table 2/4/6 figures."
)


#: **[REGISTERED 2026-08-30] THE OPEN DEPENDENCY ON SUPERVISION.**
SUPERVISION_28_DEPENDENCY = {
    "registered": "2026-08-30",
    "statement": (
        "supervision has said the 28 test images come from the cohort -- "
        "[REPORTED, not measured]. The statement is in no repo "
        "record; it enters here via the maintainer, tagged"
    ),
    "if_ids_become_recoverable": (
        "an EXACT-SPLIT arm (their 28 as the test set) is a SEPARATE "
        "REGISTERED ADDITION -- not a silent extension of this design"
    ),
    "the_assumption_not_made": (
        "181-subset-of-237 is NOT assumed without IDs: which 181 of "
        "our 237 (or whether all 181 are among them) is unknowable "
        "from the record, and the annex draws its splits from OUR "
        "cohort, saying so"
    ),
}


#: **[DESIGNED 2026-08-30 -- NOT RUN] THE COMPUTE GATE. Measure before
#: declaring, the Phase 17 lesson.**
#:
#: One split x one rater x one full fit, run BEFORE any draw count N
#: is declared. The job form, for the maintainer to launch:
#:
#:     stem        p10x_gate_fullfit  (proposed)
#:     what        one 153/28 plain random draw (named seed), one
#:                 rater's integer grades, CleftGNN with full
#:                 trainable ResNet-50 (RULING_B_FULL_TRAINABLE), 36
#:                 regions, notebook recipe (RECIPE_FIXED_TO_NOTEBOOK)
#:     where       CLUSTER-ONLY throughout -- patient images; the
#:                 pinned image, per the standing keeper discipline
#:                 (phase18.PHASE_18_RULINGS['compute_keeper_pinned'])
#:     run dir     <stem>__<git sha8>__<job-id>, the standing contract
#:     reports     per-fit wall-clock (train + eval), so the sizing
#:                 arithmetic has a measured input
#:
#: **The sizing arithmetic that follows it**: total fits = N x 5
#: raters (five models per split, RULING_A); total cost ~ N x 5 x the
#: measured per-fit wall-clock, divided by whatever parallelism the
#: cluster grants. **N is declared only after the measured cost**, by
#: the ruling -- not estimated, not defaulted.
COMPUTE_GATE_DESIGNED = {
    "designed": "2026-08-30 -- designed, NOT run; the maintainer launches",
    "job": (
        "one split draw x one rater x one full fit: 153/28 plain "
        "random (named seed), CleftGNN with full trainable ResNet-50, "
        "36 regions, notebook recipe; proposed stem p10x_gate_fullfit; "
        "cluster-only, pinned image; run dir <stem>__<git "
        "sha8>__<job-id>; reports per-fit wall-clock (train + eval)"
    ),
    "sizing_arithmetic": (
        "total fits = N x 5 raters; total cost ~ N x 5 x measured "
        "per-fit wall-clock / cluster parallelism"
    ),
    "n_is_gated": (
        "N is declared only after the measured cost, 's "
        "ruling -- the Phase 17 lesson: measure before declaring"
    ),
    "phase_17_lesson": "measure before declaring",
    # [BUILT 2026-08-30, second cycle -- designed-not-run above is
    # preserved; this key records the build.] Schema kind
    # p10x_gate_fullfit, task in run.py, config
    # configs/p10x_gate_fullfit.yaml (inputs CARRIED from
    # p10_cleftgnn_faithful.yaml's verified entries), fixture test
    # end-to-end with the degenerate-occupancy branch. STILL NOT RUN
    # -- the launch is the maintainer's, and N stays undeclared until the
    # measured cost lands.
    "built_2026_08_30": (
        "schema kind + task + config + tests shipped; NOT run; no "
        "N-draw machinery anywhere -- the negative space holds until "
        "the measured cost lands and the ruling is N"
    ),
}


#: **[MEASURED 2026-08-31, ``p10x_gate_fullfit__81a171cc__p10x-gate-
#: fullfit``] THE COMPUTE GATE RAN. THE COST IS MEASURED AND RULING (b)
#: IS CONFIRMED STRUCTURALLY IN A REAL RUN.**
#:
#: **The run's SHA is this repo's own.** ``81a171cc`` is the commit that
#: shipped the gate (schema, model recipe, task, config), so the code
#: that produced these figures IS the code recorded here -- checked with
#: ``git cat-file``, not assumed. That matters for the distribution arm,
#: which reuses this exact fit path rather than reimplementing it.
#:
#: **Verified here, by measurement rather than transcription:**
#:
#: * **29,286,981 of 29,286,981 parameters trainable.** The annex model
#:   was rebuilt locally and counted: total 29,286,981, trainable
#:   29,286,981, of which the ResNet-50 backbone is 23,508,032. An EXACT
#:   match, and it is ruling (b) holding in a real run rather than in a
#:   schema: nothing is frozen, so "we froze it" is not available as an
#:   explanation of anything this annex measures.
#: * **The majority floor 0.1111 reproduces exactly** through the
#:   shipped ``phase10.top1_macro_prf`` from the reported occupancy --
#:   and it identifies which floor ran. A constant grade-2 predictor
#:   gives 0.111111; grade 3 -- the TEST side's own majority -- gives
#:   0.131579. The reported figure is the grade-2 one, which confirms
#:   the floor was taken from the TRAIN side, as registered. A
#:   test-side-derived floor would have been the flattering mistake and
#:   is measurably not what happened.
#: * The occupancy sums to 28, matching the declared ``test_size``.
#:
#: **[MEASURED]** -- by the run, NOT on this machine. [TAG NORMALISED
#: 2026-09-01 from '[REPORTED, not verified on this machine]': the tags
#: are exactly three, and the provenance belongs beside the tag, not
#: inside it.] loss 44.1067 -> 1.4430
#: over five epochs, train 2.5 s, eval 0.0 s, arm macro F1 0.2505. The
#: run directory is CLUSTER-ONLY and did not arrive as a paste, so these
#: four are recorded on the report, tagged, exactly as the
#: literature record's wording is.
#:
#: **The arm sits above its floor at 0.2505 against 0.1111 -- and
#: NOTHING follows from that.** One draw, n=28, no interval; it is the
#: quantity the whole annex exists to put a distribution around, and
#: quoting it alone would be the single-split reporting the annex is
#: measuring the unreliability of.
GATE_MEASURED = {
    "measured": "2026-08-31, p10x_gate_fullfit__81a171cc__p10x-gate-fullfit",
    "run_sha_is_ours": (
        "81a171cc is this repo's own commit -- the one that shipped the "
        "gate -- so the code that produced these figures IS the code "
        "recorded here (checked with git cat-file, not assumed)"
    ),
    "verified_locally": {
        "parameters": (
            "29,286,981 of 29,286,981 trainable -- rebuilt and counted "
            "here, EXACT match; backbone 23,508,032 of that total. "
            "Ruling (b) confirmed structurally in a real run"
        ),
        "majority_floor": (
            "0.1111 reproduces exactly through phase10.top1_macro_prf "
            "from the reported occupancy with a constant GRADE-2 "
            "predictor (0.111111). The test side's own majority is "
            "grade 3 and would give 0.131579 -- so the figure confirms "
            "the floor came from the TRAIN side as registered, not from "
            "the test side"
        ),
        "occupancy_sums": "4+8+10+6+0 = 28, the declared test_size",
    },
    "reported_not_verified_here": (
        "loss 44.1067 -> 1.4430 over five epochs; train 2.5 s; eval "
        "0.0 s; arm macro F1 0.2505. The run directory is CLUSTER-ONLY "
        "and did not arrive as a paste -- recorded on the report "
        "and tagged, never presented as verified at source"
    ),
    "the_arm_figure_licenses_nothing": (
        "0.2505 against a 0.1111 floor is ONE DRAW at n=28 with no "
        "interval. It is precisely the quantity this annex exists to "
        "put a distribution around; quoting it alone would BE the "
        "single-split reporting whose unreliability is the subject"
    ),
}


#: **[OBSERVED 2026-08-31, DATED -- the gate's first substantive
#: observation] DRAW 1 WAS DEGENERATE ON THE TEST SIDE, AND IT WAS THE
#: FIRST DRAW.**
#:
#: At ``split_seed`` 1337 the test side's occupancy is
#: ``{1: 4, 2: 8, 3: 10, 4: 6, 5: 0}`` -- **grade 5 absent**, so the
#: 5-class macro table reports on four classes, not five.
#:
#: **This is an OBSERVATION, not the frequency.** n = 1. The registered
#: deliverable is a per-rater FREQUENCY over the N draws
#: (``DESIGN_FIXED_TO_THEIRS["splits"]["degenerate_draws"]``), and one
#: draw cannot estimate it. Recorded now because it happened on the
#: first draw and because it bears directly on the annex's question:
#: **whether a 153/28 design reliably produces a test set containing
#: every class its 5-class macro table reports on.** If it does not,
#: then a published 5-class macro figure at n=28 may be an average over
#: a different set of classes from one split to the next -- which is a
#: statement about the DESIGN, and the frequency is what will support
#: or refuse it.
#:
#: **Not a criticism of their table, and not yet anything at all.** The
#: annex says only what it measures; a single degenerate draw is an
#: anecdote until the frequency exists.
FIRST_DRAW_DEGENERATE = {
    "observed": "2026-08-31, the gate's draw at split_seed 1337",
    "occupancy": {"1": 4, "2": 8, "3": 10, "4": 6, "5": 0},
    "what_happened": (
        "grade 5 ABSENT from the test side on the FIRST draw, so the "
        "5-class macro table averaged over four classes"
    ),
    "status": (
        "an OBSERVATION feeding the registered degenerate-frequency "
        "deliverable, NOT the frequency itself -- n = 1"
    ),
    "why_it_is_recorded_now": (
        "it bears on the annex's question: whether a 153/28 design "
        "reliably produces a test set containing every class its "
        "5-class macro table reports on. If it does not, a published "
        "5-class macro figure at n=28 may average over a different set "
        "of classes from split to split -- a statement about the "
        "DESIGN, which the frequency will support or refuse"
    ),
    "not_yet_a_claim": (
        "a single degenerate draw is an anecdote until the frequency "
        "exists; the annex says only what it measures"
    ),
}


#: **[RULED 2026-08-31] N = 500.**
#:
#: **Declared AFTER the measured cost, which is the whole point of the
#: gate** (``COMPUTE_GATE_DESIGNED["n_is_gated"]``, the Phase 17
#: lesson). The arithmetic, derived from the gate's measurement rather
#: than estimated:
#:
#:     fits          500 draws x 5 raters            = 2,500
#:     per fit       ~5 s end-to-end (measured)
#:     sequential    2,500 x 5 s = 12,500 s          = 3.47 GPU-hours
#:
#: -- recomputed here and agreeing with the ruling's ~3.5 GPU-hours;
#: less when parallelised.
#:
#: **Why 5 s and not 2.5 s.** The gate measured train 2.5 s + eval 0.0 s
#: but ~5 s end-to-end; the difference is per-fit setup (model
#: construction, weight init) plus the job's one-time data load. Sizing
#: on the END-TO-END figure is the conservative choice, and it makes
#: 3.47 GPU-hours an **upper bound**: a job running many fits pays the
#: data load once rather than per fit, so the real total is between
#: 2,500 x 2.5 s (1.74 h) and this. Recorded as derived, not measured --
#: the distribution run will measure it.
N_RULED = {
    "ruled": "2026-08-31 -- after the measured cost, as gated",
    "n_draws": 500,
    "raters": 5,
    "fits": 2500,
    "sizing": (
        "500 draws x 5 raters = 2,500 fits; at the gate's measured ~5 s "
        "end-to-end that is 12,500 s = 3.47 GPU-hours sequential "
        "(recomputed here, agreeing with the ruling's ~3.5), less "
        "parallelised"
    ),
    "derived_from_measurement": (
        "the cost is the GATE'S, not an estimate -- which is what "
        "measure-then-declare means here (COMPUTE_GATE_DESIGNED)"
    ),
    "why_the_end_to_end_figure": (
        "train 2.5 s + eval 0.0 s but ~5 s end-to-end; the difference "
        "is per-fit setup plus the job's one-time data load. Sizing on "
        "5 s is conservative and makes 3.47 GPU-hours an UPPER BOUND -- "
        "a job running many fits pays the data load once, so the true "
        "total lies between 1.74 h and 3.47 h. [DERIVED, and the "
        "distribution run will measure it]"
    ),
}


#: **[DRAFT 2026-08-30 -- SUPERSEDED BY ``EXIT_CRITERIA``, LOCKED
#: 2026-08-31. Preserved as written.]** The lock waited for the compute
#: gate's measured cost and the rulings on N and the
#: normalisation; all three landed.
EXIT_CRITERIA_DRAFT = {
    "status": (
        "DRAFT -- not locked until the compute gate lands and El "
        "the maintainer rules on N and the normalisation"
    ),
    "drafted": "2026-08-30",
    "criteria": (
        "1. N split draws x 5 raters, every draw accounted for: "
        "completed fits reported, degenerate draws reported as a "
        "per-rater frequency, nothing discarded or redrawn",
        "2. every metric quoted with its floor beside it "
        "(FLOORS_REGISTERED): majority-class beside the "
        "classification metrics, phase11.iem constant-predictor "
        "beside IEM, PCC's floor recorded as undefined",
        "3. the published Study-Set values (quoted from "
        "CLEFTGNN_COMPARATOR_TABLES, never retyped) located within "
        "the per-rater distributions; one of the two committed "
        "readings (READINGS_COMMITTED) applied verbatim; no third "
        "reading invented after the numbers",
        "4. ANNEX_PROHIBITION on every deliverable, and the "
        "unmatched-raters caveat (RATER_PANEL_DISCREPANCY_EIGHTH) on "
        "every per-rater quote",
        "5. the supervision dependency reported as registered "
        "(SUPERVISION_28_DEPENDENCY): [REPORTED, not measured] riding, no "
        "181-subset-of-237 assumption, exact-split only as a "
        "separate registered addition",
        "6. no ledger row from this annex without its own "
        "registration: the annex characterises a DESIGN; any claim "
        "would need the standing two-condition criterion and a "
        "registration of its own",
        "7. suite green",
    ),
    # [2026-08-30, second cycle] Of the three pre-lock items named in
    # the status line, the NORMALISATION is now ruled
    # (NORMALISATION_RULED); the status wording is preserved as
    # written. Still pending before the lock: the compute gate's
    # MEASURED COST (built, not run) and the ruling on N.
    "normalisation_ruled_2026_08_30": (
        "NORMALISATION_RULED -- one of the three pre-lock items is "
        "ruled; the gate run and N remain"
    ),
    # [2026-08-31, third cycle] All three closed; superseded by the
    # lock, which is EXIT_CRITERIA.
    "superseded_2026_08_31": "EXIT_CRITERIA",
}


#: **[LOCKED 2026-08-31] THE EXIT CRITERIA. All three pre-lock items are
#: closed and NOTHING WAS ADDED WHILE LOCKING.**
#:
#: **The three the draft named, each resolved -- and there is no
#: fourth:**
#:
#:     1. the compute gate's measured cost   RAN (GATE_MEASURED)
#:     2. N                                  RULED at 500 (N_RULED)
#:     3. the normalisation                  RULED (NORMALISATION_RULED)
#:
#: The draft's status line named exactly those three ("not locked until
#: the compute gate lands and the ruling is on N and the
#: normalisation"). No other item was ever held for the lock, so the
#: answer to "is a third outstanding" is: **there is none** -- the third
#: was the normalisation and it was ruled on 2026-08-30.
#:
#: **The criteria are the drafted seven, unchanged in substance.** They
#: are restated here with the now-known figures written in (N = 500;
#: 2,500 fits) and with the degenerate-draw handling made explicit,
#: because the lock is where a criterion stops being able to move. **No
#: criterion was added, dropped, weakened, or reordered at the lock** --
#: the draft is preserved above and the two can be read side by side.
EXIT_CRITERIA = {
    "locked": "2026-08-31",
    "supersedes": "EXIT_CRITERIA_DRAFT, preserved as written",
    "pre_lock_items_all_closed": {
        "1_gate_cost": "RAN -- GATE_MEASURED",
        "2_n": "RULED at 500 -- N_RULED",
        "3_normalisation": "RULED -- NORMALISATION_RULED (2026-08-30)",
        "is_a_fourth_outstanding": (
            "NO. The draft's status line named exactly three, and the "
            "third WAS the normalisation. None remains"
        ),
    },
    "nothing_added_at_the_lock": (
        "the seven criteria are the drafted seven, unchanged in "
        "substance -- restated only to write in the now-known figures "
        "(N = 500, 2,500 fits) and to make the degenerate-draw handling "
        "explicit. None added, dropped, weakened or reordered; the "
        "draft is preserved so the two can be read side by side"
    ),
    "criteria": (
        "1. 500 draws x 5 raters = 2,500 fits, every draw accounted "
        "for: completed fits reported, degenerate draws reported as a "
        "per-rater FREQUENCY, nothing discarded or redrawn",
        "2. every metric quoted with its floor beside it "
        "(FLOORS_REGISTERED): majority-class beside the classification "
        "metrics, phase11.iem constant-predictor beside IEM, PCC's "
        "floor recorded as undefined",
        "3. the published Study-Set values (quoted from "
        "CLEFTGNN_COMPARATOR_TABLES, never retyped) located within the "
        "per-rater distributions by percentile; one of the two "
        "committed readings (READINGS_COMMITTED) applied verbatim; no "
        "third reading invented -- a MIXED pattern across raters gets a "
        "dated OBSERVATION instead (MIXED_PATTERN_REGISTERED)",
        "4. ANNEX_PROHIBITION on every deliverable, and the "
        "unmatched-raters caveat (RATER_PANEL_DISCREPANCY_EIGHTH) on "
        "every per-rater quote",
        "5. the supervision dependency reported as registered "
        "(SUPERVISION_28_DEPENDENCY): [REPORTED, not measured] riding, no "
        "181-subset-of-237 assumption, exact-split only as a separate "
        "registered addition",
        "6. no ledger row from this annex without its own "
        "registration: the annex characterises a DESIGN; any claim "
        "would need the standing two-condition criterion and a "
        "registration of its own",
        "7. suite green",
    ),
    "degenerate_handling_explicit": (
        "made explicit AT THE LOCK rather than left to the run: "
        "phase10.top1_macro_prf EXCLUDES a truth-absent class from the "
        "macro average, so a degenerate draw's macro figure is an "
        "average over FEWER classes and is NOT term-by-term comparable "
        "with a five-class one. Degenerate draws are therefore reported "
        "SEPARATELY -- their own distribution and count -- as well as "
        "inside the all-draws distribution, never silently averaged in "
        "and never dropped"
    ),
}


#: **[REGISTERED 2026-08-31, BEFORE ANY NUMBER] THE MIXED CASE IS AN
#: OBSERVATION, NOT A THIRD READING.**
#:
#: ``READINGS_COMMITTED`` fixes two readings and the annex invents no
#: third. But the readings are stated for "the published values" and
#: there are five per-rater distributions with FOUR cited published
#: reference points, so the pattern can land inside for some pairings
#: and outside for others. That case is registered now, before the
#: numbers, with its handling fixed: **it is recorded as a dated
#: OBSERVATION describing which pairings fell where, and it concludes
#: nothing.**
#:
#: The precedent is Phase 17's unfired combination cells
#: (``phase17.UNPREDICTED_PATTERN``): the committed cells were
#: preserved unfired and the pattern that did occur was described
#: rather than promoted to a finding. Same discipline here -- a mixed
#: result is not a licence to pick whichever reading the mixture
#: happens to favour.
MIXED_PATTERN_REGISTERED = {
    "registered": "2026-08-31, before any number",
    "the_case": (
        "the published values land inside some per-rater distributions "
        "and outside others -- possible because there are five "
        "distributions and four cited reference points"
    ),
    # [NAMED 2026-08-31, before the run] A gap in the two readings,
    # stated rather than discovered afterwards: READINGS_COMMITTED
    # covers INSIDE and OUTSIDE-AND-ABOVE. A published value landing
    # BELOW our whole distribution is covered by NEITHER -- and it is a
    # reachable outcome, since their Study minimum is -0.162 and a
    # distribution of 500 draws could sit entirely above it. That case
    # takes the same handling as the mixed one: a dated observation
    # that concludes nothing. Naming it now is cheaper than deciding
    # what it means once a number exists.
    "the_below_gap": (
        "a published value BELOW the whole distribution is covered by "
        "NEITHER committed reading (they cover inside and "
        "outside-and-above). Reachable -- their Study minimum is "
        "-0.162. Same handling: a dated observation concluding nothing. "
        "Named before the run rather than decided after"
    ),
    "handling": (
        "a DATED OBSERVATION naming which pairings fell where, "
        "concluding NOTHING. Not a third reading, and not a licence to "
        "adopt whichever of the two the mixture happens to favour"
    ),
    "precedent": (
        "phase17.UNPREDICTED_PATTERN -- committed cells preserved "
        "unfired, the pattern that did occur described rather than "
        "promoted to a finding"
    ),
}


#: **[REGISTERED 2026-08-31] THE PUBLISHED REFERENCE POINTS, and there
#: are FOUR of them -- not five.**
#:
#: The annex locates published values inside our distributions, so
#: exactly which values are published has to be pinned before the run.
#: **Their Study Test Set (Table 1, n=28, each rater's own labels) is
#: recorded in this project as a RANGE and a mean, not as five
#: per-rater cells** (``CLEFTGNN_COMPARATOR_TABLES["study_test_set"]``:
#: "0.440 to -0.162, mean ~0.14"), plus one named cell -- rater E's
#: 0.283 (``FAITHFUL_ARM_REGISTERED["their_598_cell"]
#: ["same_model_on_their_study_set"]``).
#:
#: So four reference points are citable and a fifth is NOT: the full
#: five-cell Study vector is not in our record, and inventing it to
#: make a tidy rater-by-rater comparison is exactly the fabrication the
#: provenance rule forbids. **If the maintainer supplies Table 1's five cells
#: from the manuscript, the arm gains them as a dated addition** -- it
#: does not need them to run.
#:
#: Each point is located by percentile within EACH of our five
#: distributions, and the pairing is stated as UNMATCHED every time: a
#: position within a distribution, never a matched rater-to-rater
#: comparison (``RATER_PANEL_DISCREPANCY_EIGHTH``).
PUBLISHED_REFERENCE_POINTS = {
    "registered": "2026-08-31, before the run",
    "points": {
        "study_max": 0.440,
        "study_rater_e": 0.283,
        "study_mean_approx": 0.14,
        "study_min": -0.162,
    },
    "provenance": (
        "0.440 / -0.162 / ~0.14 from CLEFTGNN_COMPARATOR_TABLES"
        "['study_test_set']['range'] (Table 1, n=28, each rater's own "
        "labels); 0.283 from FAITHFUL_ARM_REGISTERED['their_598_cell']"
        "['same_model_on_their_study_set']"
    ),
    "why_four_and_not_five": (
        "our record holds Table 1 as a RANGE plus a mean, plus one "
        "named cell -- not five per-rater cells. Inventing the missing "
        "vector to make a tidy rater-by-rater comparison is the "
        "fabrication the provenance rule forbids"
    ),
    "if_the_five_cells_arrive": (
        "a DATED ADDITION if the maintainer supplies Table 1's five cells "
        "from the manuscript; the arm does not need them to run"
    ),
    "pairing_is_unmatched": (
        "the pairing is UNMATCHED: each point is located by percentile "
        "within EACH of our five distributions, and every quote says "
        "so -- a position within a distribution, never a matched "
        "rater-to-rater comparison"
    ),
}


#: **[REGISTERED 2026-08-31] THE DISTRIBUTION ARM: 500 DRAWS x 5
#: RATERS, THROUGH THE GATE'S OWN FIT PATH.**
#:
#: **One implementation, and it is the gate's.** The fit is factored
#: into ``run._p10x_one_fit`` and BOTH tasks call it -- the gate
#: unchanged, the distribution 2,500 times. This is the Phase 18 IEM
#: lesson applied before it can be repeated
#: (``phase10.PROBE_RECONSTRUCTED_THE_PIPELINE`` is the same shape): a
#: second fit path would let the gate's verified numbers detach from
#: the distribution's, silently, and nothing downstream would disagree.
#: The refactor is behaviour-preserving and the suite asserts the gate
#: still reports what it reported.
#:
#: **Draws are enumerated, never globbed.** 500 split seeds are DERIVED
#: from one declared root seed by a stated rule and then WRITTEN OUT in
#: the config, so the draw set is reproducible from the root and
#: auditable without running anything. The task re-derives them from
#: the root and REFUSES if the enumerated list disagrees -- the
#: generator-carries-filled-values pattern, applied to seeds.
#:
#: **Degenerate draws are accounted, not dropped.** Per-draw per-rater
#: class occupancy is recorded for all 2,500 cells; the frequency is
#: reported per rater; and because
#: ``phase10.top1_macro_prf`` excludes a truth-absent class from the
#: macro average, degenerate draws are reported as their own
#: distribution BESIDE the all-draws one rather than silently averaged
#: in (``EXIT_CRITERIA["degenerate_handling_explicit"]``).
#:
#: **Compute shape**: 2,500 fits at the gate's measured ~5 s is ~3.5
#: GPU-hours as one job -- modest, so **one job is the shipped default
#: and the recommendation**, with a DECLARED shard rule available
#: rather than improvised if a queue's wall-clock cap requires it:
#: ``shard: {count: C, index: I}`` selects draws where
#: ``draw_index % C == I``. Sharding by DRAW rather than by rater is
#: the better split -- every shard then yields all five raters on its
#: subset, so a lost shard costs draws rather than an entire rater --
#: and the shards partition the 500 exactly, asserted by test. **The
#: merge of sharded outputs into one distribution is NOT BUILT**: it is
#: needed only if the maintainer shards, and it is named here rather than
#: improvised later.
DISTRIBUTION_ARM_REGISTERED = {
    "registered": "2026-08-31",
    "shape": "500 draws x 5 raters = 2,500 fits (N_RULED)",
    "one_implementation": (
        "the fit is factored into run._p10x_one_fit and BOTH tasks call "
        "it -- the gate unchanged, the distribution 2,500 times. A "
        "second fit path would let the gate's verified numbers detach "
        "from the distribution's silently (the Phase 18 IEM lesson; "
        "phase10.PROBE_RECONSTRUCTED_THE_PIPELINE is the same shape). "
        "The refactor is behaviour-preserving and the suite asserts it"
    ),
    "draws_enumerated": (
        "500 split seeds DERIVED from one declared root seed by a "
        "stated rule and WRITTEN OUT in the config; the task re-derives "
        "them and REFUSES if the list disagrees. Reproducible from the "
        "root, auditable without running anything, never a runtime glob"
    ),
    "degenerate_accounting": (
        "per-draw per-rater occupancy for all 2,500 cells; frequency "
        "reported PER RATER; degenerate draws reported as their own "
        "distribution beside the all-draws one, never silently averaged "
        "in and never discarded"
    ),
    "outputs": (
        "per-rater distributions of 5-class macro precision/recall/F1, "
        "per-rater PCC and IEM, each with its floor beside "
        "(FLOORS_REGISTERED); percentile position of the four cited "
        "published reference points (PUBLISHED_REFERENCE_POINTS) within "
        "each distribution, pairing stated UNMATCHED every time"
    ),
    "compute_shape": (
        "~3.5 GPU-hours as ONE JOB at the gate's measured cost -- "
        "modest, so one job is the shipped default and the "
        "recommendation. A DECLARED shard rule is available rather than "
        "improvised: shard {count: C, index: I} takes draws where "
        "draw_index % C == I. Sharding by DRAW beats by rater -- every "
        "shard yields all five raters on its subset, so a lost shard "
        "costs draws, not a whole rater -- and the shards partition the "
        "500 exactly (asserted by test)"
    ),
    "merge_not_built": (
        "merging sharded outputs into one distribution is NOT BUILT: "
        "needed only if the maintainer shards, and named here rather than "
        "improvised later"
    ),
    "wall_clock_caveat": (
        "whether one job fits the queue's wall-clock cap is NOT known "
        "here -- the cap is not in the record. If it does not, the "
        "shard rule is why that is a launch decision rather than a "
        "rebuild"
    ),
}


#: **[OBSERVED 2026-08-31, ``p10x_distribution__63b0f421__p10x-
#: distribution``] THE DISTRIBUTION RAN. 2,500 FITS, FINALIZED, SINGLE
#: ATTEMPT.**
#:
#: **The run's SHA is this repo's own** -- ``63b0f421`` is the commit
#: that shipped the arm, checked with ``git cat-file``, so the code
#: that produced these figures is the code recorded here. As with the
#: gate, the run directory is CLUSTER-ONLY and did not arrive on this
#: machine, so the figures below are **[REPORTED, verified against
#: metrics.json ]** rather than read here. What could be
#: checked locally was checked, and is marked.
#:
#: **Cost**: 5,385.4 s for 2,500 fits = **2.154 s/fit** (recomputed
#: here: 5385.4 / 2500 = 2.1542, agreeing).
#:
#: **TWO THINGS THIS RUN CANNOT DO, stated so they are not assumed:**
#:
#: 1. **It does not corroborate the gate's tagged figures.** The gate
#:    ran at ``split_seed`` 1337; the distribution's 500 seeds are
#:    ``SeedSequence(1337)`` state, and **1337 is not among them**
#:    (checked). The gate's draw is never re-run, so its loss
#:    trajectory and F1 0.2505 stay ``[REPORTED, not verified here]``.
#: 2. **It does not carry a parameter report.** The arm's summary omits
#:    it, so the 29,286,981 count is corroborated by the local rebuild
#:    (``GATE_MEASURED``) and by nothing in this run. Recorded as a
#:    small gap in the arm's outputs rather than papered over.
DISTRIBUTION_OBSERVED = {
    "observed": (
        "2026-08-31, p10x_distribution__63b0f421__p10x-distribution, "
        "finalized, single attempt"
    ),
    "run_sha_is_ours": (
        "63b0f421 is this repo's own commit -- the one that shipped the "
        "distribution arm (checked with git cat-file)"
    ),
    "fits": 2500,
    "wall_clock_seconds": 5385.4,
    "per_fit_seconds": 2.154,
    "arithmetic_checked_here": (
        "5385.4 / 2500 = 2.1542 s/fit, agreeing with the reported 2.154"
    ),
    "provenance_of_the_figures": (
        "[REPORTED, verified against metrics.json ] -- the "
        "run directory is CLUSTER-ONLY and did not reach this machine, "
        "so nothing below was read from the file here. Every check that "
        "COULD be made locally was made and is marked as such"
    ),
    "does_not_corroborate_the_gate": (
        "the gate ran at split_seed 1337 and 1337 is NOT among the 500 "
        "SeedSequence(1337) draws (checked here), so the gate's loss "
        "trajectory and F1 0.2505 stay [REPORTED, not verified here] -- "
        "this run never repeats that draw"
    ),
    "no_parameter_report_in_this_arm": (
        "the distribution summary omits parameter_report, so the "
        "29,286,981 count rests on the local rebuild (GATE_MEASURED) "
        "and on nothing in this run. A small gap in the arm's outputs, "
        "recorded rather than papered over"
    ),
}


#: **[FIRED 2026-08-31] READING 1. THE PUBLISHED VALUES FALL INSIDE THE
#: DISTRIBUTION -- ALL FOUR, IN ALL FIVE RATERS.**
#:
#: The committed sentence, quoted verbatim from
#: ``READINGS_COMMITTED["published_inside_the_distribution"]``:
#:
#:     **single-split reporting at n=28 does not identify performance
#:     -- the strongest form of the small-n argument the project has,
#:     measured on their own design.**
#:
#: **The percentile table** (position of each published point within
#: each rater's PCC distribution; the spread is across the five
#: raters):
#:
#:     published point        percentile range across raters
#:     0.440  (Study max)              97.74 - 99.09
#:     0.283  (their rater E)          88.81 - 93.14
#:     0.14   (Study mean)             68.86 - 79.43
#:     -0.162 (Study min)              13.87 - 26.00
#:
#: **No point landed outside any distribution.** Every one of the 20
#: (point x rater) cells is ``inside``, which is what makes reading 1
#: fire mechanically rather than by choice -- so **neither the
#: incomplete-replication reading nor the pre-named below-distribution
#: gap applies**, and no third reading was invented
#: (``MIXED_PATTERN_REGISTERED`` did not fire either).
#:
#: **Checked locally**: every quoted percentile lies strictly inside
#: (0, 100), which is consistent with ``where == "inside"`` on all 20
#: cells and inconsistent with any ``above_all`` / ``below_all``.
#:
#: **THE EXHIBIT FOR THE WRITE-UP**: *the published range is
#: reproducible by re-drawing the split alone.* Their reported spread
#: from 0.440 down to -0.162 is spanned by our own draws with the
#: label, metric, architecture and recipe all held fixed at theirs --
#: and **0.91-2.26% of our draws EXCEED their best published Study
#: value** (derived here from the 0.440 row). Nothing about their model
#: follows from this; it is a statement about what a single 153/28
#: split can report.
READING_FIRED = {
    "fired": "2026-08-31, published_inside_the_distribution",
    "sentence_verbatim": READINGS_COMMITTED[
        "published_inside_the_distribution"
    ],
    "percentiles_across_raters": {
        "0.440": (97.74, 99.09),
        "0.283": (88.81, 93.14),
        "0.14": (68.86, 79.43),
        "-0.162": (13.87, 26.00),
    },
    "nothing_landed_outside": (
        "all 20 (point x rater) cells are INSIDE, so reading 1 fires "
        "mechanically rather than by choice; neither the "
        "incomplete-replication reading nor the pre-named "
        "below-distribution gap applies, and MIXED_PATTERN_REGISTERED "
        "did not fire. No third reading invented"
    ),
    "checked_here": (
        "every quoted percentile lies strictly inside (0, 100), "
        "consistent with where == 'inside' on all 20 cells and "
        "inconsistent with any above_all / below_all"
    ),
    "the_exhibit": (
        "**the published range is reproducible by RE-DRAWING THE SPLIT "
        "ALONE.** Their 0.440 to -0.162 spread is spanned by our draws "
        "with label, metric, architecture and recipe fixed at theirs, "
        "and 0.91-2.26% of our draws EXCEED their best published Study "
        "value (derived from the 0.440 row). Nothing about their MODEL "
        "follows -- it is a statement about what one 153/28 split can "
        "report"
    ),
    "prohibition_rides": ANNEX_PROHIBITION,
}


#: **[MEASURED 2026-08-31] THE DESIGN PATHOLOGIES. The registered
#: degenerate-frequency deliverable, now a measurement rather than an
#: observation.**
#:
#: ``FIRST_DRAW_DEGENERATE`` recorded one degenerate draw and refused
#: to call it a frequency at n=1. Here is the frequency, at n=500 per
#: rater:
#:
#: * **Degenerate draws: 173, 273, 306, 380 and 500 of 500** across the
#:   five raters -- **1,632 of 2,500 cells, 65.3%** (summed here). One
#:   rater, the SLT, is degenerate in **every single draw**.
#: * **Undefined PCC: 62-101 draws per rater**, where the model
#:   predicted a single constant grade across all 28 test images, so
#:   the correlation has no denominator.
#: * **Per-rater PCC span ~1.1 at sd ~0.19-0.21.**
#:
#: **The bearing, which is the annex's question answered in the
#: negative**: a 153/28 design **frequently cannot produce a test set
#: containing every class its 5-class macro table reports on**, and
#: **sometimes produces a model with no computable correlation at
#: all**. A published 5-class macro figure at this design is therefore
#: liable to be an average over a different set of classes from one
#: split to the next -- exactly the possibility
#: ``FIRST_DRAW_DEGENERATE`` named before the frequency existed.
#:
#: ``degenerate_note``'s incomparability clause rides every macro
#: figure quoted from this run: a truth-absent class is EXCLUDED from
#: the macro, so a degenerate draw's figure averages over FEWER classes
#: and is not term-by-term comparable with a five-class one.
#:
#: **[OPEN -- an attribution this record deliberately does not make]**
#: The five degenerate counts arrived as an ASCENDING list, so it is
#: not sheet order, and only the 500 is attributed by name (the SLT).
#: **Which rater carries 173, 273, 306 or 380 is NOT determinable from
#: what reached this machine**, and is left unassigned rather than
#: guessed. One line from the run's own ``per_rater`` block settles it.
DESIGN_PATHOLOGIES_MEASURED = {
    "measured": "2026-08-31, run 500 -- the registered deliverable",
    "supersedes_as_frequency": (
        "FIRST_DRAW_DEGENERATE, which recorded ONE degenerate draw and "
        "refused to call it a frequency at n=1. This is the frequency"
    ),
    "degenerate_draws_of_500": (173, 273, 306, 380, 500),
    "degenerate_overall": (
        "1,632 of 2,500 cells = 65.3% (summed here from the five "
        "counts) -- and one rater, the SLT, is degenerate in EVERY draw"
    ),
    "undefined_pcc_draws_per_rater": "62-101",
    "undefined_pcc_mechanism": (
        "the model predicted a single constant grade across all 28 test "
        "images, so the correlation has no denominator"
    ),
    "pcc_span_and_sd": "span ~1.1 per rater at sd ~0.19-0.21",
    "the_bearing": (
        "a 153/28 design FREQUENTLY cannot produce a test set "
        "containing every class its 5-class macro table reports on, and "
        "SOMETIMES produces a model with no computable correlation at "
        "all. A published 5-class macro figure at this design is "
        "therefore liable to average over a different set of classes "
        "from one split to the next -- the possibility "
        "FIRST_DRAW_DEGENERATE named before the frequency existed"
    ),
    "incomparability_rides": (
        "degenerate_note's clause travels with every macro figure: a "
        "truth-absent class is EXCLUDED from the macro, so a degenerate "
        "draw's figure averages over FEWER classes and is not "
        "term-by-term comparable with a five-class one"
    ),
    # [PRESERVED as written -- this is why the counts were a bare tuple
    # first, and the reason is worth keeping visible now that the names
    # exist: the attribution was UNAVAILABLE, not merely unwritten, and
    # guessing it from the ascending order would have been wrong four
    # times out of five.]
    "open_attribution": (
        "the five counts arrived as an ASCENDING list -- so NOT sheet "
        "order -- and only the 500 is attributed by name (the SLT). "
        "Which rater carries 173, 273, 306 or 380 is NOT determinable "
        "from what reached this machine and is left UNASSIGNED rather "
        "than guessed; one line from the run's per_rater block settles "
        "it"
    ),
    # [RESOLVED 2026-08-31] It did settle it. The ascending order was
    # NOT sheet order, and the mapping below confirms guessing would
    # have mis-assigned four of the five.
    "attribution_resolved_2026_08_31": "DEGENERATE_BY_RATER",
    "open_distribution_table": (
        "the close-out message referred to 'the per-rater distribution "
        "table above' but no such table was included in it. The "
        "per-rater min/median/max of each metric is therefore NOT "
        "recorded here -- named as missing rather than reconstructed "
        "from the percentile rows"
    ),
    # [PARTIALLY RESOLVED 2026-08-31] The degenerate column arrived by
    # name (DEGENERATE_BY_RATER) and is recorded. The PCC statistics
    # -- min/median/max/sd and non-null n -- were referred to a second
    # time ("the five rows above, verbatim") and again did not arrive
    # in the message. Still named as missing, still not reconstructed:
    # non-null n is derivable from an undefined-PCC count, but only the
    # 62-101 RANGE reached here, not per-rater values.
    "distribution_table_still_partial_2026_08_31": (
        "PER_RATER_TABLE -- the degenerate column is filled by name; "
        "the PCC min/median/max/sd and non-null n columns were "
        "referred to twice and arrived neither time, and are recorded "
        "as PENDING rather than reconstructed"
    ),
}


#: **[ATTRIBUTED 2026-08-31] THE DEGENERATE COUNTS, BY RATER.**
#:
#: **[MEASURED]** -- by run 500; the extractor output over its
#: [TAG NORMALISED 2026-09-01 from '[REPORTED -- ...]']
#: ``per_rater`` block, pasted; not read from metrics.json here]**, the
#: same provenance as every other figure from this run.
#:
#:     Rater 10 - Plastic surgeon                    173 / 500
#:     Rater 11 - Psychologist                       273 / 500
#:     Rater 7  - Cleft patient                      306 / 500
#:     Rater 8  - Orthodontist                       380 / 500
#:     Rater 9  - Speech and language therapist      500 / 500
#:
#: **Checked here**: the five values are the same multiset as the
#: already-recorded ``degenerate_draws_of_500``; every key is a real
#: ``scoresheet.RATERS`` column and all five are present; and the SLT
#: is the 500, agreeing with the separate statement that one rater is
#: degenerate in every draw.
#:
#: **Why the tuple came first, and why that was right**: the counts
#: originally arrived ascending, which is not sheet order. Reading them
#: as sheet order would have assigned Cleft patient 173, Orthodontist
#: 273, SLT 306, Plastic surgeon 380 and Psychologist 500 -- **wrong
#: for ALL FIVE**, and wrong about which rater is the every-draw case.
#: The bare tuple was the honest structure while the mapping was
#: unavailable (``open_attribution``, preserved).
#:
#: **[CORRECTED 2026-08-31, same turn]** This record first said "wrong
#: for four of the five". It is five: the sheet-order reading agrees
#: with the true mapping on no rater at all. Caught by the test written
#: to check the claim -- which is the point of writing the check as an
#: arithmetic re-derivation rather than a restatement.
#:
#: **The caveat rides**: these are sheet identities, UNMATCHED to
#: CleftGNN's A-E (``RATER_PANEL_DISCREPANCY_EIGHTH``). That the SLT is
#: degenerate in every draw is a fact about OUR sheet's SLT column, not
#: about any rater of theirs.
DEGENERATE_BY_RATER = {
    "attributed": "2026-08-31",
    "provenance": (
        "[MEASURED] -- by run 500, NOT read from metrics.json here: El "
        "the maintainer's extractor output over its "
        "per_rater block, pasted; not read from metrics.json here. "
        "[TAG NORMALISED 2026-09-01 from '[REPORTED -- the maintainer's "
        "extractor output over run 500's per_rater block ...]': the "
        "tags are exactly three; the provenance is prose now]"
    ),
    "counts_of_500": {
        "Rater 7 - Cleft patient": 306,
        "Rater 8 - Orthodontist": 380,
        "Rater 9 - Speech and language therapist": 500,
        "Rater 10 - Plastic surgeon": 173,
        "Rater 11 - Psychologist": 273,
    },
    "checked_here": (
        "the same multiset as the recorded degenerate_draws_of_500; "
        "every key a real scoresheet.RATERS column, all five present; "
        "and the SLT is the 500, agreeing with the separate statement "
        "that one rater is degenerate in every draw"
    ),
    "why_the_tuple_came_first": (
        "the counts arrived ASCENDING, which is not sheet order. "
        "Reading them as sheet order would have assigned Cleft patient "
        "173, Orthodontist 273, SLT 306, Plastic surgeon 380 and "
        "Psychologist 500 -- WRONG for ALL FIVE, and wrong about which "
        "rater is the every-draw case. The bare tuple was the honest "
        "structure while the mapping was unavailable"
    ),
    "corrected_2026_08_31": (
        "this record first said 'wrong for four of the five'. It is "
        "FIVE -- the sheet-order reading agrees on no rater at all. "
        "Caught by the test written to re-derive the claim rather than "
        "restate it"
    ),
    "unmatched_caveat_rides": (
        "sheet identities, UNMATCHED to CleftGNN's A-E "
        "(RATER_PANEL_DISCREPANCY_EIGHTH). That the SLT is degenerate "
        "in every draw is a fact about OUR sheet's SLT column, not "
        "about any rater of theirs"
    ),
}


#: **[RECORDED 2026-08-31, PARTIAL] THE ANNEX'S PER-RATER TABLE.**
#:
#: **[MEASURED]** -- by run 500; the extractor output over its
#: [TAG NORMALISED 2026-09-01 from '[REPORTED -- ...]']
#: ``per_rater`` block; not read from metrics.json here]** for the
#: degenerate column. **The PCC statistics columns did not arrive.**
#:
#:     rater                        degenerate   undefined PCC   PCC stats
#:     Plastic surgeon                 173/500       (62-101)*    PENDING
#:     Psychologist                    273/500       (62-101)*    PENDING
#:     Cleft patient                   306/500       (62-101)*    PENDING
#:     Orthodontist                    380/500       (62-101)*    PENDING
#:     Speech and language therapist   500/500       (62-101)*    PENDING
#:
#: \* the undefined-PCC figure reached this machine only as a RANGE
#: across raters (62-101), never per rater, so no row carries its own
#: value and none is invented for it. ``PCC stats`` covers min, median,
#: max, sd and non-null n -- referred to twice ("the per-rater
#: distribution table above", then "the five rows above, verbatim") and
#: arriving neither time. Non-null n is derivable from a per-rater
#: undefined count, which is exactly the figure that is missing.
#:
#: **THE STRUCTURAL DETAIL THIS TABLE MAKES VISIBLE, and it is verified
#: rather than asserted**: undefined-PCC draws are counted SEPARATELY
#: from degenerate ones, because the two conditions are independent.
#: Constructed from the shipped definitions and checked:
#:
#: * a model predicting ONE class for all 28 test images has **no PCC**
#:   (zero-variance prediction) but a perfectly **computable macro
#:   F1** over the truth-present classes;
#: * a test side missing a grade is **degenerate** while the model's
#:   predictions may vary freely, leaving **PCC defined**;
#: * and both can occur in the same draw.
#:
#: So the two pathologies **overlap without coinciding**, and a count
#: of one is not a count of the other. Reporting them as one number
#: would understate the damage: a draw can fail the macro table and the
#: correlation independently, for unrelated reasons.
PER_RATER_TABLE = {
    "recorded": "2026-08-31, PARTIAL",
    "provenance": (
        "[MEASURED] -- by run 500, NOT read from metrics.json here: El "
        "the maintainer's extractor output over its "
        "per_rater block; not read from metrics.json here. "
        "[TAG NORMALISED 2026-09-01 from '[REPORTED -- the maintainer's "
        "extractor output over run 500's per_rater block ...]': the "
        "tags are exactly three; the provenance is prose now]"
    ),
    "degenerate_of_500": DEGENERATE_BY_RATER["counts_of_500"],
    "undefined_pcc": (
        "62-101 across raters -- a RANGE only. It never reached this "
        "machine per rater, so no row carries its own value and none "
        "is invented"
    ),
    "pcc_stats_pending": (
        "min / median / max / sd and non-null n are PENDING: referred "
        "to twice ('the per-rater distribution table above', then 'the "
        "five rows above, verbatim') and arriving neither time. "
        "Non-null n is derivable from a per-rater undefined count -- "
        "exactly the figure that is missing -- so the column is left "
        "empty rather than reconstructed"
    ),
    # [FILLED 2026-08-31, third asking -- the note above is PRESERVED
    # as written, because it was true when written and records that the
    # column was left empty rather than reconstructed for two rounds.]
    "filled_2026_08_31": "PCC_BY_RATER, F1_BY_RATER, IEM_BY_RATER",
    "the_structural_detail": (
        "undefined-PCC draws are counted SEPARATELY from degenerate "
        "ones because the two conditions are INDEPENDENT: a model "
        "predicting one class for all 28 has NO PCC but a computable "
        "macro F1; a test side missing a grade is degenerate while PCC "
        "stays defined; and both can occur together. The two "
        "pathologies OVERLAP WITHOUT COINCIDING, so a count of one is "
        "not a count of the other -- reporting them as one number "
        "would understate the damage"
    ),
    "verified_here": (
        "the independence is CONSTRUCTED from the shipped definitions "
        "and checked, not asserted: all three cases exhibited "
        "(undefined-not-degenerate, degenerate-not-undefined, both)"
    ),
}


#: **[FILLED 2026-08-31] THE PCC DISTRIBUTIONS, PER RATER.**
#:
#: **[MEASURED]** -- by run 500; the extractor output over its
#: [TAG NORMALISED 2026-09-01 from '[REPORTED -- ...]']
#: ``per_rater`` block; not read from metrics.json here]**, the same
#: tag the attribution carries.
#:
#:     rater             min     p05     med     p95     max      sd    n
#:     Plastic surg.  -0.4910 -0.3009  0.0274  0.3385  0.5352  0.1974  432
#:     Psychologist   -0.5306 -0.3161  0.0000  0.3581  0.5502  0.1982  438
#:     Cleft patient  -0.5602 -0.3246 -0.0314  0.3156  0.6187  0.1956  423
#:     Orthodontist   -0.5967 -0.3466 -0.0263  0.3148  0.5735  0.2076  399
#:     SLT            -0.4596 -0.2963  0.0403  0.3661  0.5092  0.1909  411
#:
#: ``n`` is non-null draws; the balance of 500 is the undefined-PCC
#: count (68 / 62 / 77 / 101 / 89 respectively).
#:
#: **THE OBSERVATION THIS TABLE MAKES, and it is descriptive**: every
#: median sits between **-0.0314 and +0.0403**. The TYPICAL re-drawn
#: split produces essentially **no correlation at all**, while their
#: published values sit from our 14th to our 99th percentile. That is
#: the same finding ``READING_FIRED`` already carries, seen from the
#: other side: it is not that our replication is weak and theirs strong
#: -- it is that this design's output is dominated by which 28 images
#: the split happened to choose. **No claim about their model follows**
#: (``ANNEX_PROHIBITION``).
PCC_BY_RATER = {
    "filled": "2026-08-31",
    "provenance": (
        "[MEASURED] -- by run 500, NOT read from metrics.json here: El "
        "the maintainer's extractor output over its "
        "per_rater block; not read from metrics.json here. "
        "[TAG NORMALISED 2026-09-01 from '[REPORTED -- the maintainer's "
        "extractor output over run 500's per_rater block ...]': the "
        "tags are exactly three; the provenance is prose now]"
    ),
    "rows": {
        "Rater 7 - Cleft patient": {
            "min": -0.5602, "p05": -0.3246, "med": -0.0314,
            "p95": 0.3156, "max": 0.6187, "sd": 0.1956,
            "n_nonnull": 423, "undefined": 77,
        },
        "Rater 8 - Orthodontist": {
            "min": -0.5967, "p05": -0.3466, "med": -0.0263,
            "p95": 0.3148, "max": 0.5735, "sd": 0.2076,
            "n_nonnull": 399, "undefined": 101,
        },
        "Rater 9 - Speech and language therapist": {
            "min": -0.4596, "p05": -0.2963, "med": 0.0403,
            "p95": 0.3661, "max": 0.5092, "sd": 0.1909,
            "n_nonnull": 411, "undefined": 89,
        },
        "Rater 10 - Plastic surgeon": {
            "min": -0.4910, "p05": -0.3009, "med": 0.0274,
            "p95": 0.3385, "max": 0.5352, "sd": 0.1974,
            "n_nonnull": 432, "undefined": 68,
        },
        "Rater 11 - Psychologist": {
            "min": -0.5306, "p05": -0.3161, "med": 0.0000,
            "p95": 0.3581, "max": 0.5502, "sd": 0.1982,
            "n_nonnull": 438, "undefined": 62,
        },
    },
    "the_median_observation": (
        "every median sits between -0.0314 and +0.0403: the TYPICAL "
        "re-drawn split produces essentially NO CORRELATION AT ALL, "
        "while their published values sit from our 14th to our 99th "
        "percentile. The same finding READING_FIRED carries, from the "
        "other side -- not that our replication is weak and theirs "
        "strong, but that this design's output is dominated by which "
        "28 images the split happened to choose. NO claim about their "
        "model follows (ANNEX_PROHIBITION)"
    ),
}


#: **[FILLED 2026-08-31] MACRO F1 AND IEM, PER RATER. Both defined on
#: every draw, so all rows are n=500** -- unlike PCC, which is
#: undefined wherever the model predicted one class for all 28.
#:
#:     rater             min     p05     med     p95     max      sd
#:     -- macro F1 --
#:     Plastic surg.  0.0267  0.0706  0.1366  0.2516  0.4346  0.0602
#:     Psychologist   0.0370  0.0754  0.1460  0.2917  0.3859  0.0667
#:     Cleft patient  0.0138  0.0706  0.1479  0.2649  0.4109  0.0609
#:     Orthodontist   0.0317  0.1000  0.1860  0.3422  0.5010  0.0746
#:     SLT            0.0185  0.0968  0.1767  0.3088  0.4444  0.0664
#:     -- IEM (convention A; an ERROR, so LOWER is better) --
#:     Plastic surg.  0.4522  0.6006  0.8835  1.5375  2.5679  0.3084
#:     Psychologist   0.5138  0.6276  0.8814  1.4521  2.0191  0.2658
#:     Cleft patient  0.3714  0.6163  0.9368  1.3905  2.0915  0.2502
#:     Orthodontist   0.2143  0.3789  0.6737  1.1138  1.7342  0.2346
#:     SLT            0.3286  0.4951  0.7075  1.1465  2.1299  0.2240
#:
#: **THE F1 ROWS ARE NOT COMPARABLE ACROSS RATERS, and the reason is
#: measured rather than cautionary.** These are ALL-DRAWS
#: distributions, and each rater's draws are a different mixture of
#: degenerate and complete: **35% degenerate for the plastic surgeon,
#: 100% for the SLT** (``DEGENERATE_BY_RATER``). A degenerate draw's
#: macro averages over FEWER classes
#: (``EXIT_CRITERIA["degenerate_handling_explicit"]``), so a rater with
#: more degenerate draws is being scored on a systematically different
#: quantity. **The SLT's entire F1 distribution is degenerate draws --
#: not one of its 500 rows is a five-class macro.** Reading
#: "Orthodontist 0.1860 beats Plastic surgeon 0.1366" off this table
#: would be the different-quantities error under a column heading.
#:
#: **IEM carries the standing prohibition**
#: (``phase18.DELIVERABLES_REGISTERED["cleftgnn_iem_prohibition"]``):
#: these values are NEVER placed beside CleftGNN's Table 2/4/6
#: figures. And IEM is an ERROR under convention A, so a lower median
#: is a better one -- stated because a column of numbers beside F1
#: invites reading it the same direction.
F1_BY_RATER = {
    "filled": "2026-08-31",
    "provenance": PCC_BY_RATER["provenance"],
    "n_per_row": 500,
    "rows": {
        "Rater 7 - Cleft patient": {
            "min": 0.0138, "p05": 0.0706, "med": 0.1479,
            "p95": 0.2649, "max": 0.4109, "sd": 0.0609,
        },
        "Rater 8 - Orthodontist": {
            "min": 0.0317, "p05": 0.1000, "med": 0.1860,
            "p95": 0.3422, "max": 0.5010, "sd": 0.0746,
        },
        "Rater 9 - Speech and language therapist": {
            "min": 0.0185, "p05": 0.0968, "med": 0.1767,
            "p95": 0.3088, "max": 0.4444, "sd": 0.0664,
        },
        "Rater 10 - Plastic surgeon": {
            "min": 0.0267, "p05": 0.0706, "med": 0.1366,
            "p95": 0.2516, "max": 0.4346, "sd": 0.0602,
        },
        "Rater 11 - Psychologist": {
            "min": 0.0370, "p05": 0.0754, "med": 0.1460,
            "p95": 0.2917, "max": 0.3859, "sd": 0.0667,
        },
    },
    "not_comparable_across_raters": (
        "these are ALL-DRAWS distributions and each rater's draws are "
        "a DIFFERENT MIXTURE of degenerate and complete -- 35% "
        "degenerate for the plastic surgeon, 100% for the SLT "
        "(DEGENERATE_BY_RATER). A degenerate draw's macro averages "
        "over FEWER classes, so a rater with more of them is scored on "
        "a systematically different quantity. The SLT's ENTIRE F1 "
        "distribution is degenerate draws -- not one of its 500 rows "
        "is a five-class macro. Reading 'Orthodontist 0.1860 beats "
        "Plastic surgeon 0.1366' off this table would be the "
        "different-quantities error under a column heading"
    ),
}

IEM_BY_RATER = {
    "filled": "2026-08-31",
    "provenance": PCC_BY_RATER["provenance"],
    "n_per_row": 500,
    "convention": "A (phase11.IEM_DIRECTION_ANSWERED)",
    "direction": (
        "an ERROR -- LOWER is better. Stated because a column of "
        "numbers beside F1 invites reading it the same direction"
    ),
    "rows": {
        "Rater 7 - Cleft patient": {
            "min": 0.3714, "p05": 0.6163, "med": 0.9368,
            "p95": 1.3905, "max": 2.0915, "sd": 0.2502,
        },
        "Rater 8 - Orthodontist": {
            "min": 0.2143, "p05": 0.3789, "med": 0.6737,
            "p95": 1.1138, "max": 1.7342, "sd": 0.2346,
        },
        "Rater 9 - Speech and language therapist": {
            "min": 0.3286, "p05": 0.4951, "med": 0.7075,
            "p95": 1.1465, "max": 2.1299, "sd": 0.2240,
        },
        "Rater 10 - Plastic surgeon": {
            "min": 0.4522, "p05": 0.6006, "med": 0.8835,
            "p95": 1.5375, "max": 2.5679, "sd": 0.3084,
        },
        "Rater 11 - Psychologist": {
            "min": 0.5138, "p05": 0.6276, "med": 0.8814,
            "p95": 1.4521, "max": 2.0191, "sd": 0.2658,
        },
    },
    "prohibition_rides": (
        "phase18.DELIVERABLES_REGISTERED['cleftgnn_iem_prohibition'] -- "
        "these values are NEVER placed beside CleftGNN's Table 2/4/6 "
        "figures"
    ),
}


#: **[CHECKED 2026-08-31, BEFORE THE TABLE WAS RECORDED] The two
#: required re-derivations, and six more the table made possible.**
#:
#: Both required checks PASS:
#:
#: 1. ``n_nonnull + pcc_undefined == 500`` on every row -- 432+68,
#:    438+62, 423+77, 399+101, 411+89.
#: 2. Every ``pcc_undefined`` inside the record's existing 62-101 --
#:    and the range is **exact at both ends**: 62 is the minimum and
#:    101 the maximum, so the previously recorded range was the true
#:    one rather than a rounding of it.
#:
#: Six further checks, run because the table made them available:
#:
#: 3. The degenerate column equals ``DEGENERATE_BY_RATER`` rater for
#:    rater -- two separately-pasted tables agreeing.
#: 4. Every quantile row is monotone (min <= p05 <= med <= p95 <= max)
#:    across PCC, F1 and IEM -- fifteen rows.
#: 5. PCC sd 0.1909-0.2076 and span 0.9688-1.1789 corroborate the
#:    already-recorded "sd ~0.19-0.21" and "span ~1.1".
#: 6. **The min/max columns independently confirm READING_FIRED**:
#:    every one of the four published points lies strictly between min
#:    and max for all five raters, which is ``inside`` on all twenty
#:    cells -- established from a DIFFERENT COLUMN than the percentiles
#:    that fired the reading.
#: 7. **Every percentile row is consistent with the quantile columns**:
#:    0.440 above every p95 (reported 97.74-99.09 vs an implied
#:    95-100); 0.283 and 0.14 between median and p95 (reported
#:    88.81-93.14 and 68.86-79.43 vs an implied 50-95); -0.162 between
#:    p05 and median (reported 13.87-26.00 vs an implied 5-50).
#: 8. The medians run -0.0314 to +0.0403 -- the observation
#:    ``PCC_BY_RATER["the_median_observation"]`` rests on.
#:
#: **Checks 6 and 7 matter more than the arithmetic ones.** The
#: percentile table and the reading were recorded a turn earlier from a
#: separate paste; this table reaches the same verdict through columns
#: that were not used to produce it. Two independent routes to one
#: conclusion is the corroboration this project asks for, and it is
#: what makes the exhibit safe to quote.
TABLE_CHECKS = {
    "checked": "2026-08-31, before the table was recorded",
    "required_1_row_sums": (
        "PASS -- n_nonnull + pcc_undefined == 500 on every row "
        "(432+68, 438+62, 423+77, 399+101, 411+89)"
    ),
    "required_2_undefined_in_range": (
        "PASS -- every pcc_undefined inside the recorded 62-101, and "
        "the range is EXACT at both ends (min 62, max 101), so the "
        "recorded range was the true one rather than a rounding"
    ),
    "3_degenerate_column_agrees": (
        "PASS -- equals DEGENERATE_BY_RATER rater for rater; two "
        "separately-pasted tables agreeing"
    ),
    "4_quantiles_monotone": (
        "PASS -- min <= p05 <= med <= p95 <= max on all fifteen rows "
        "across PCC, F1 and IEM"
    ),
    "5_corroborates_recorded_spread": (
        "PASS -- sd 0.1909-0.2076 and span 0.9688-1.1789 against the "
        "recorded 'sd ~0.19-0.21' and 'span ~1.1'"
    ),
    "6_minmax_confirms_the_reading": (
        "PASS -- every published point strictly between min and max "
        "for all five raters = INSIDE on all twenty cells, established "
        "from a DIFFERENT COLUMN than the percentiles that fired "
        "READING_FIRED"
    ),
    "7_percentiles_consistent_with_quantiles": (
        "PASS -- 0.440 above every p95 (97.74-99.09 vs implied "
        "95-100); 0.283 and 0.14 between median and p95 (88.81-93.14, "
        "68.86-79.43 vs implied 50-95); -0.162 between p05 and median "
        "(13.87-26.00 vs implied 5-50)"
    ),
    "8_medians_near_zero": (
        "-0.0314 to +0.0403 -- the observation "
        "PCC_BY_RATER['the_median_observation'] rests on"
    ),
    "why_6_and_7_matter_most": (
        "the percentile table and the reading were recorded a turn "
        "earlier from a SEPARATE paste; this table reaches the same "
        "verdict through columns that were not used to produce it. Two "
        "independent routes to one conclusion is the corroboration "
        "this project asks for, and it is what makes the exhibit safe "
        "to quote"
    ),
}


#: **[CORRECTED 2026-08-31] THE SIZING BAND MISSED, AND IT MISSED LOW.**
#:
#: ``N_RULED`` predicted the run would fall between **1.74 h and 3.47
#: h**, calling 3.47 h an upper bound because a batched job pays its
#: data load once. The outturn is **1.496 h (5,385.4 s)** -- **below
#: the band's LOWER bound by 13.8%** (computed here).
#:
#: So the band was wrong in a direction the framing did not anticipate.
#: The upper bound reasoning was right and insufficient: I bounded the
#: saving at the data load, when the per-fit TRAIN cost also fell --
#: the gate measured 2.5 s of training for a fit that costs 2.154 s
#: end-to-end inside a batched job.
#:
#: **[REASONED, not measured]** The likely cause is that the gate's
#: single fit paid one-time GPU costs -- CUDA context creation, cuDNN
#: autotuning, allocator warm-up -- which a 2,500-fit job pays once and
#: amortises to nothing. Plausible and untested; the way to settle it
#: is the per-fit series in ``draws.csv``, which would show the first
#: fit dearer than the rest. **Recorded as a hypothesis, not a
#: finding.**
#:
#: **Why this is recorded at all**: the estimate was declared before
#: the run, so it is answerable, and a prediction that missed is worth
#: the same treatment as one that held (``PAIRED_BCA_OBSERVED`` is the
#: pattern). Nothing rests on the band -- it sized a job that has now
#: run -- but a sizing rule that quietly over-charges by 2.3x would
#: mis-scale the next phase that reuses it.
SIZING_OUTTURN = {
    "corrected": "2026-08-31, against N_RULED's declared band",
    "predicted": "1.74 h to 3.47 h (6,250-12,500 s)",
    "outturn": "1.496 h (5,385.4 s) -- BELOW the lower bound by 13.8%",
    "what_the_framing_got_wrong": (
        "the upper-bound reasoning was right and INSUFFICIENT: the "
        "saving was bounded at the data load, but the per-fit TRAIN "
        "cost also fell -- the gate measured 2.5 s of training for a "
        "fit that costs 2.154 s end-to-end inside a batched job"
    ),
    "likely_cause": (
        "[REASONED, not measured] the gate's single fit paid one-time "
        "GPU costs -- CUDA context creation, cuDNN autotuning, "
        "allocator warm-up -- that a 2,500-fit job pays once and "
        "amortises. Settled by the per-fit series in draws.csv, which "
        "would show the first fit dearer than the rest. A hypothesis"
    ),
    "why_recorded": (
        "the estimate was declared before the run, so it is answerable, "
        "and a prediction that missed gets the treatment one that held "
        "gets (PAIRED_BCA_OBSERVED is the pattern). Nothing rests on "
        "the band, but a sizing rule that over-charges by 2.3x would "
        "mis-scale the next phase that reuses it"
    ),
}


#: **[PROPOSED 2026-08-31 -- A RULING IS OWED] NO LEDGER ROW.**
#:
#: **The proposal**: this annex banks nothing in
#: ``results_ledger.ENTRIES``.
#:
#: **The grounds, and they are Phase 18's D2 ruling applied**:
#: ``D2_HOME_RULED`` recorded NO ledger rows for the metric-dependence
#: finding on the reasoning that the phase characterised the CRITERION
#: rather than making a claim about an arm under it. The same holds
#: here, twice over:
#:
#: * **This characterises a DESIGN, not one of our arms.** The subject
#:   is what a 153/28 split can report, measured on their protocol. No
#:   arm of ours is being asserted better or worse than another.
#: * **The two-condition criterion was never applied.** There is no
#:   paired BCa, no per-seed interval, no ``combined_claimable_delta``
#:   threshold -- because none of those is defined for this quantity.
#:   A ledger row implies a verdict the criterion produced, and the
#:   criterion did not run.
#:
#: The findings are not diminished by having no row: the exhibit
#: (``READING_FIRED["the_exhibit"]``) and the pathologies
#: (``DESIGN_PATHOLOGIES_MEASURED``) are write-up material and are
#: recorded in full here. **A row would misrepresent them as claims of
#: a kind this project reserves for the criterion.**
#:
#: **The ruling is recorded either way**, and the ledger is
#: unchanged at 37 entries until it lands.
LEDGER_PROPOSAL = {
    "proposed": "2026-08-31 -- the proposal, the ruling is",
    "proposal": "NO ledger row from the Phase 10 annex",
    "grounds": (
        "Phase 18's D2 ruling applied, in its own words: "
        "D2_HOME_RULED gave the flip table NO ledger rows because it "
        "'measures the criterion, not the arms' and because 'a ledger "
        "row is the unit of ARM-CLAIM'. Both hold here, twice over: "
        "(1) this characterises a DESIGN -- what a 153/28 split can "
        "report, measured on their protocol -- and no arm of ours is "
        "asserted better or worse than another; (2) the two-condition "
        "criterion was NEVER APPLIED: no paired BCa, no per-seed "
        "interval, no claimable-delta threshold, because none is "
        "defined for this quantity. A row implies a verdict the "
        "criterion produced, and the criterion did not run"
    ),
    "not_a_diminishment": (
        "the exhibit and the pathologies are write-up material and are "
        "recorded in full; a row would misrepresent them as claims of "
        "the kind this project reserves for the criterion"
    ),
    # [RULED 2026-08-31, the maintainer] The proposal is TAKEN, on the
    # grounds as drafted -- D2's own language, quoted rather than
    # paraphrased. The ledger stands at 37 and the annex banks nothing.
    "ruling": (
        "RULED 2026-08-31, the maintainer: NO ROW, on D2's quoted grounds as "
        "drafted. The ledger stands at 37; the Phase 10 annex banks "
        "nothing. Its findings live in the phase record -- the exhibit "
        "in READING_FIRED, the pathologies in "
        "DESIGN_PATHOLOGIES_MEASURED -- which is where a "
        "characterisation of a DESIGN belongs"
    ),
}


#: **[CLOSED 2026-08-31] PHASE 10 ANNEX. The split-distribution
#: replication.**
#:
#: **The question, answered**: *does a single 153/28 split identify a
#: model's performance at this cohort size?* **No.** Their whole
#: published Study range is reproducible by re-drawing the split alone,
#: with label, metric, architecture and recipe held at theirs.
#:
#: **The seven LOCKED criteria, walked:**
#:
#: 1. **2,500 fits, every draw accounted for** -- MET. 500 x 5, single
#:    attempt, finalized; degenerate draws reported as a per-rater
#:    frequency (``DESIGN_PATHOLOGIES_MEASURED``), none discarded or
#:    redrawn.
#: 2. **Every metric with its floor** -- MET by construction: the
#:    majority-class floor is computed per draw from the TRAIN side and
#:    the IEM constant-predictor floor through ``phase11.iem``; PCC's
#:    floor rides as *undefined* in every rater's block.
#: 3. **Published values located, one committed reading applied** --
#:    MET. All four points inside all five distributions;
#:    ``published_inside_the_distribution`` applied verbatim; no third
#:    reading, and the mixed/below cases did not fire.
#: 4. **Prohibitions on every deliverable** -- MET: the annex
#:    prohibition, five unmatched raters, the knowingly-replicated
#:    CIFAR-10 normalisation, and the CleftGNN-IEM prohibition all ride
#:    in the run's own summary.
#: 5. **The supervision dependency as registered** -- MET: ``[REPORTED, not
#:    measured]`` on the 28-from-cohort statement, no
#:    181-subset-of-237 assumption, draws taken from OUR cohort.
#: 6. **No ledger row without its own registration** -- MET, and
#:    ``LEDGER_PROPOSAL`` puts the question to the maintainer rather than
#:    settling it here.
#: 7. **Suite green** -- MET.
#:
#: **What the annex adds to the write-up**: one exhibit and one
#: pathology record. The exhibit is that the published range is
#: reproducible by re-drawing the split alone. The pathology is that
#: 65.3% of draw-rater cells cannot report a complete 5-class macro and
#: 62-101 draws per rater have no computable PCC at all.
#:
#: **What it does NOT say, and the prohibition is a tested literal**:
#: nothing here corrects, re-computes or re-scores CleftGNN's published
#: numbers. Those figures characterise OUR replication under THEIR
#: protocol.
#:
#: **Open, non-blocking, carried by name**: the per-rater attribution
#: of four degenerate counts; the per-rater distribution table, which
#: the close-out message referenced but did not carry; the maintainer's
#: ledger ruling; and the standing exact-split addition if the 28 IDs
#: ever become recoverable (``SUPERVISION_28_DEPENDENCY``).
PHASE_10_ANNEX_CLOSING = {
    "closed": "2026-08-31",
    "question": (
        "does a single 153/28 split identify a model's performance at "
        "this cohort size?"
    ),
    "answer": (
        "NO -- their whole published Study range is reproducible by "
        "re-drawing the split alone, with label, metric, architecture "
        "and recipe held at theirs (READING_FIRED)"
    ),
    "exit_criteria_walked": {
        "1_every_draw_accounted": (
            "MET -- 500 x 5 = 2,500 fits, single attempt, finalized; "
            "degenerate draws reported as a per-rater frequency, none "
            "discarded or redrawn (DESIGN_PATHOLOGIES_MEASURED)"
        ),
        "2_floors_beside_every_metric": (
            "MET by construction -- majority-class floor per draw from "
            "the TRAIN side, IEM constant-predictor floor through "
            "phase11.iem, PCC's floor riding as UNDEFINED in every "
            "rater's block"
        ),
        "3_published_located_reading_applied": (
            "MET -- all four points inside all five distributions; "
            "published_inside_the_distribution applied verbatim; no "
            "third reading, and neither the mixed nor the below case "
            "fired (READING_FIRED)"
        ),
        "4_prohibitions_on_every_deliverable": (
            "MET -- the annex prohibition, five unmatched raters, the "
            "knowingly-replicated CIFAR-10 normalisation and the "
            "CleftGNN-IEM prohibition all ride in the run's summary"
        ),
        "5_supervision_dependency_as_registered": (
            "MET -- [REPORTED, not measured] on the 28-from-cohort "
            "statement, no 181-subset-of-237 assumption, draws from OUR "
            "cohort"
        ),
        "6_no_ledger_row_unregistered": (
            "MET -- and LEDGER_PROPOSAL puts the question to the maintainer "
            "rather than settling it here"
        ),
        "7_suite_green": "MET",
    },
    "cost_outturn": (
        "2.154 s/fit, 5,385.4 s total -- BELOW N_RULED's 1.74-3.47 h "
        "band, recorded as a missed prediction with its cause "
        "hypothesised rather than asserted (SIZING_OUTTURN)"
    ),
    "gate_figures_still_tagged": (
        "the gate's loss trajectory and F1 0.2505 remain [REPORTED, not "
        "verified here]: run 500 never repeats split_seed 1337 "
        "(checked), and the arm's summary carries no parameter report, "
        "so neither is corroborated by it (DISTRIBUTION_OBSERVED)"
    ),
    "the_write_up_exhibit": (
        "**the published range is reproducible by re-drawing the split "
        "alone** -- with a pathology record beside it: 65.3% of "
        "draw-rater cells cannot report a complete 5-class macro, and "
        "62-101 draws per rater have no computable PCC at all"
    ),
    "what_it_does_not_say": (
        "nothing here corrects, re-computes or re-scores CleftGNN's "
        "published numbers -- ANNEX_PROHIBITION, a tested literal. "
        "These figures characterise OUR replication under THEIR protocol"
    ),
    "open_non_blocking": (
        "the per-rater attribution of four degenerate counts "
        "(DESIGN_PATHOLOGIES_MEASURED['open_attribution'])",
        "the per-rater distribution table, referenced by the close-out "
        "message but not carried in it",
        "the maintainer's ledger ruling (LEDGER_PROPOSAL)",
        "the exact-split addition if the 28 IDs become recoverable "
        "(SUPERVISION_28_DEPENDENCY)",
    ),
    # [UPDATED 2026-08-31, after the close-out] Three of the four moved;
    # the list above is preserved as the closing wrote it.
    "open_items_updated_2026_08_31": (
        "CLOSED: the rater attribution -- DEGENERATE_BY_RATER, by name",
        "CLOSED: the ledger question -- RULED, no row, ledger stands "
        "at 37 (LEDGER_PROPOSAL['ruling'])",
        "CLOSED (2026-08-31, third asking): the per-rater table -- "
        "PCC_BY_RATER, F1_BY_RATER and IEM_BY_RATER complete, both "
        "required re-derivations passing plus six more (TABLE_CHECKS)",
        "OPEN, unchanged: the exact-split addition if the 28 IDs "
        "become recoverable (SUPERVISION_28_DEPENDENCY)",
    ),
}


def summary() -> dict:
    """The annex's records, importable as one object."""
    return {
        "question": ANNEX_QUESTION_REGISTERED,
        "verification": VERIFICATION_NOTES,
        "design": DESIGN_FIXED_TO_THEIRS,
        "five_class_exception": FIVE_CLASS_SCOPED_EXCEPTION,
        "rater_panel_discrepancy": RATER_PANEL_DISCREPANCY_EIGHTH,
        "floors": FLOORS_REGISTERED,
        "ruling_a": RULING_A_ALL_FIVE_RATERS,
        "ruling_b": RULING_B_FULL_TRAINABLE,
        "recipe": RECIPE_FIXED_TO_NOTEBOOK,
        "normalisation_proposed": NORMALISATION_PROPOSED_NOT_LOCKED,
        "readings": READINGS_COMMITTED,
        "prohibition": ANNEX_PROHIBITION,
        "supervision_dependency": SUPERVISION_28_DEPENDENCY,
        "compute_gate": COMPUTE_GATE_DESIGNED,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        # [2026-08-30, second cycle] The pin fired as designed when
        # this key joined; updated dated.
        "normalisation_ruled": NORMALISATION_RULED,
        # [2026-08-31, third cycle] The gate ran, N was ruled, the
        # criteria locked, the distribution arm registered. The pin
        # fired again; updated dated.
        "gate_measured": GATE_MEASURED,
        "first_draw_degenerate": FIRST_DRAW_DEGENERATE,
        "n_ruled": N_RULED,
        "exit_criteria": EXIT_CRITERIA,
        "mixed_pattern": MIXED_PATTERN_REGISTERED,
        "published_reference_points": PUBLISHED_REFERENCE_POINTS,
        "distribution_arm": DISTRIBUTION_ARM_REGISTERED,
        # [2026-08-31, close-out] The run, the fired reading, the
        # measured pathologies, the missed sizing band, the ledger
        # proposal and the closing. The pin fired; updated dated.
        "distribution_observed": DISTRIBUTION_OBSERVED,
        "reading_fired": READING_FIRED,
        "design_pathologies": DESIGN_PATHOLOGIES_MEASURED,
        "sizing_outturn": SIZING_OUTTURN,
        "ledger_proposal": LEDGER_PROPOSAL,
        "closing": PHASE_10_ANNEX_CLOSING,
        # [2026-08-31, after the close-out] The attribution and the
        # partial table. The pin fired; updated dated.
        "degenerate_by_rater": DEGENERATE_BY_RATER,
        "per_rater_table": PER_RATER_TABLE,
        # [2026-08-31, third asking] The table's columns arrived and
        # the PENDING keys are filled. The pin fired; updated dated.
        "pcc_by_rater": PCC_BY_RATER,
        "f1_by_rater": F1_BY_RATER,
        "iem_by_rater": IEM_BY_RATER,
        "table_checks": TABLE_CHECKS,
    }
