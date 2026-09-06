"""Classification metrics on a regression arm's predictions.

**[REGISTERED 2026-08-24, post-Phase-15-closing] Why this module exists.**

The project reports PCC. the maintainer will be asked why it does not report
precision, recall and F1. The answer is on record -- a continuous
panel-mean target, ordinal structure, macro-averaging over a
5/89/110/30/3 distribution, a Spearman-Brown ceiling that bounds a
correlation, and a pre-registered choice -- but **the metrics themselves
have never been computed**, and "we did not compute them" is a weaker
answer than "we computed them, and here is why they are secondary".

So they get computed. Once, on the best arm's existing predictions, as
pod arithmetic: no re-training and no new inference.

----------------------------------------------------------------------------
WHAT IS REUSED RATHER THAN REWRITTEN
----------------------------------------------------------------------------
* the discretisation is ``eval.metrics.to_3class`` -- FROZEN, and the
  same function ``data.labels.class3`` calls to build the manifest
  column. The 2.5/3.5 rule is not restated here.
* the macro precision/recall/F1 convention is ``phase10.top1_macro_prf``,
  which already exists for exactly this reason (``eval/metrics.py`` is
  frozen and carries no classification metric). This module computes its
  own from the confusion matrix and then **asserts agreement with
  phase10's**, so there is one convention and a second implementation
  that checks it rather than a second convention.
* the CSV reader is ``cluster_csv.read_cluster_csv``, whose docstring
  records the three times the tier marker cost a round.

What is genuinely new here: weighted averaging, the per-class support
vector, the confusion matrix, and the majority-class floor.
"""

from __future__ import annotations

import numpy as np

from . import phase10
from .eval import metrics as frozen_metrics


class ClassificationError(ValueError):
    """A refusal specific to this addendum."""


#: **[REGISTERED 2026-08-24, BEFORE ANY NUMBER] The readings.**
#:
#: Registered before the run, in the config header and here, so no
#: number from this addendum can be read without them.
CLASSIFICATION_METRICS_SECONDARY = {
    "registered": "2026-08-24, before any number exists",
    "what_this_is": (
        "**SECONDARY and DESCRIPTIVE.** Precision, recall and F1 on the "
        "best arm's existing out-of-fold predictions "
        "(p7_d1_vit_b16_imagenet_g1, five seeds), computed so the "
        "question can be answered with numbers rather than with a "
        "reason for not having them"
    ),
    "what_this_is_not": (
        "**NEVER A CLAIM. NEVER A LEDGER ROW.** No verdict rests on "
        "these figures, no comparison between arms is made from them, "
        "and nothing here amends a banked PCC. The arm's result is its "
        "PCC; this is a second description of the same predictions"
    ),
    "no_retraining": (
        "pod arithmetic over CSVs the arm already wrote. No training, "
        "no inference, no GPU. The predictions are the SAME numbers the "
        "banked PCC was computed from -- which is the point: two "
        "metrics, one set of predictions"
    ),

    # ---- the three sentences that ride with every quote --------------
    "caveat_a_support": (
        "**(a) THE CLASS-SUPPORT VECTOR TRAVELS WITH EVERY QUOTE.** The "
        "cohort's median-grade distribution is 5/89/110/30/3 over 237; "
        "the 3-class supports are MEASURED by this task and printed "
        "beside every average. A macro F1 without its supports is a "
        "number whose denominator is hidden"
    ),
    "caveat_b_macro_averaging": (
        "**(b) MACRO-AVERAGING WEIGHTS A 3-PATIENT CLASS EQUALLY WITH A "
        "110-PATIENT ONE.** That is what macro-averaging means, and at "
        "these class counts it is the dominant fact about the number: "
        "one or two patients moving in the smallest class swing the "
        "macro average by more than a systematic change across the "
        "largest. The weighted average is reported beside it for that "
        "reason -- not because either is correct, but because their "
        "DIFFERENCE is the informative quantity"
    ),
    "caveat_c_discretisation_is_reporting": (
        "**(c) THE DISCRETISATION IS A REPORTING STEP, NOT THE TRAINING "
        "OBJECTIVE.** The arm was trained on the CONTINUOUS panel mean "
        "with an MSE loss and selected on inner-validation MSE. "
        "Collapsing its output to classes afterwards measures something "
        "the arm was never optimised for. A model trained to maximise "
        "F1 on these classes would be a different model, and its F1 is "
        "not predicted by this one's"
    ),
    "the_floor": (
        "**the MAJORITY-CLASS BASELINE is computed on the same "
        "discretisation and reported beside every figure**, so no "
        "accuracy or F1 is read without its floor. phase9 banked that "
        "floor as 0.502 for class3 and this task ASSERTS it "
        "reproduces -- a mismatch means the truth column is not the "
        "one this addendum thinks it is, and the task refuses rather "
        "than reporting against the wrong baseline"
    ),
    "why_pcc_stays_primary": (
        "unchanged, and NOT re-argued from these numbers: the target is "
        "continuous, the grades are ordinal, the Spearman-Brown "
        "ceiling bounds a CORRELATION rather than an F1, and PCC was "
        "PRE-REGISTERED. This addendum adds a measurement to that "
        "answer; it does not replace the answer, and if the numbers "
        "were flattering they would still be secondary"
    ),
}


#: **[ANSWERED 2026-08-24, BEFORE THE RUN] Three classes, not five.**
THREE_NOT_FIVE = {
    "the_question": (
        "the instruction: if the 2.5/3.5 rule collapses to three "
        "classes rather than five, say so plainly, and report both if "
        "both are defined"
    ),
    "the_answer": (
        "**IT COLLAPSES TO THREE, AND ONLY THREE ARE DEFINED.** Two "
        "thresholds cut a line into three parts. The manifest's own "
        "column says so in its schema -- ``class3``, '3-class collapse "
        "of the mean at fixed thresholds 2.5/3.5' -- and the frozen "
        "``eval.metrics.to_3class`` implements it. 1,2 -> low; 3 -> "
        "mid; 4,5 -> high"
    ),
    "why_five_is_not_reported": (
        "a 5-class family would need FOUR thresholds. Only two are "
        "registered. Cutting predictions at 1.5/2.5/3.5/4.5 -- "
        "rounding to the nearest integer -- is the obvious rule and is "
        "**NOWHERE IN THE RECORD**, so using it would mean choosing "
        "thresholds tonight and reporting numbers from them in the same "
        "run. That is the shape this project refuses. Reported as "
        "UNDEFINED rather than invented"
    ),
    # [2026-08-24, the fourth sequence amendment] "Phase 16" below
    # means the metric-space ablation, which is PHASE 18 now
    # (phase15.PHASE_SEQUENCE_RENUMBERED_4). Preserved as written.
    # [DECLINED 2026-08-30, the maintainer, at the Phase-18 rulings] The
    # candidate below is DECLINED, not taken: the tails have not
    # changed (g1=8, g5=4), a 4-threshold rule would produce numbers
    # this record calls undefined, and it would MANUFACTURE the
    # CleftGNN 5-class comparability the different-quantities rule
    # forbids. REOPEN CONDITION: a label source with sufficient support
    # in the extreme grades -- a decision, not a permanent closure
    # (phase18.PHASE_18_RULINGS['three_not_five_declined']).
    "declined_2026_08_30": "phase18.PHASE_18_RULINGS",
    # [SCOPED EXCEPTION 2026-08-30, the maintainer, at the Phase-10-annex
    # registration] The annex's 5-class figures are LEGITIMATE where
    # this record forbids them elsewhere, because they do not touch
    # this record's ground: the annex target is an integer 1-5 grade
    # trained with cross-entropy, and argmax yields the class
    # DIRECTLY -- no 4-threshold rule, no continuous prediction
    # collapsed. The decline above stands untouched.
    "scoped_exception_2026_08_30": (
        "phase10_annex.FIVE_CLASS_SCOPED_EXCEPTION"
    ),
    "what_would_make_it_defined": (
        "a registered 4-threshold rule for collapsing a continuous "
        "prediction to the 1-5 grade scale, declared before any number "
        "is computed from it. The truth side already has one -- the "
        "median grade is an integer -- so it is only the PREDICTION "
        "side that is missing. Phase 16 is where that registration "
        "belongs if it is wanted"
    ),
}


#: **[BANKED 2026-08-24] THE ADDENDUM RAN. SECONDARY AND DESCRIPTIVE.**
CLASSIFICATION_METRICS_BANKED = {
    "banked": (
        "2026-08-24, p7_d1_classification_metrics__b677f53c, single "
        "clean attempt"
    ),
    "what_these_are": (
        "**SECONDARY AND DESCRIPTIVE, exactly as registered before the "
        "run** (CLASSIFICATION_METRICS_SECONDARY). No claim, no ledger "
        "row, no arm comparison. The arm's result is still its PCC; "
        "this is a second description of the same predictions"
    ),

    # ---- the floor check, which is what makes the rest readable ------
    "the_floor_check_passed": (
        "majority 0.5021 against phase9's banked 0.5020 -- the truth "
        "column IS the mean-collapsed class3 the addendum was "
        "registered against, so the figures are quoted against the "
        "right floor. Had it been the MEDIAN collapse the task would "
        "have refused, and that refusal was the point of declaring the "
        "floor in the config"
    ),
    "the_support": (
        "**[88, 119, 30] over 237** -- measured, and it travels with "
        "every quote (caveat a). **NOT the median-grade distribution**: "
        "5/89/110/30/3 collapses to 94/110/33, and these are the "
        "MEAN-collapsed supports. The 30 here is a coincidence of "
        "arithmetic, not the same 30"
    ),
    "the_floors_own_disagreement": (
        "the majority-class predictor scores accuracy 0.5021 and macro "
        "F1 0.2228. **A trivial constant model already splits the two "
        "metrics by a factor of two before any arm is involved** -- the "
        "disagreement is a property of the class distribution, not "
        "something the arm produced"
    ),

    # ---- the numbers -------------------------------------------------
    "pooled": (
        "over five seeds: accuracy 0.5181 (sd 0.0159), macro F1 0.3760 "
        "(sd 0.0166). Per-seed accuracy 0.4937-0.5359, macro F1 "
        "0.3509-0.3956. Pooled means the mean and five-seed SD OF THE "
        "PER-SEED METRICS -- the shape the arm's banked PCC carries"
    ),

    # ---- the headline it was built to produce ------------------------
    "the_headline": (
        "**against the majority floor the SAME PREDICTIONS gain +0.0160 "
        "by accuracy and +0.1532 by macro F1 -- a factor of 9.58, an "
        "order of magnitude apart on an identical model.** That is the "
        "measured form of 'the metric choice changes the impression "
        "more than the model does', and it is the sentence this "
        "addendum was built to be able to say with numbers"
    ),
    "the_intervals_sharpen_it": (
        "**measured, and it makes the headline stronger rather than "
        "weaker.** Each arm's own 95% on the mean of five seeds: "
        "accuracy [0.5042, 0.5320] against a floor of 0.5021 -- above "
        "it by 0.0021 at the lower bound, BARELY separable from a "
        "constant predictor; macro F1 [0.3614, 0.3906] against a floor "
        "of 0.2228 -- above it by 0.1386 at the lower bound, "
        "unambiguously separable. **The two metrics do not merely "
        "differ in size; they differ in whether the arm is "
        "distinguishable from guessing at all**"
    ),
    "what_this_does_not_say": (
        "it does NOT say macro F1 is the better metric, nor that the "
        "arm is better than its PCC suggests. A metric that separates "
        "the arm from a constant predictor is not thereby the right "
        "metric for a continuous ordinal target -- and macro F1 "
        "separates them partly BECAUSE it rewards any prediction "
        "outside the majority class, which is a property of the "
        "average, not evidence about the model"
    ),

    # ---- the three caveats, attached ---------------------------------
    "caveats_attached": (
        "all three ride in the run's own metrics.json and in every "
        "quote: (a) the support vector [88, 119, 30]; (b) macro "
        "averaging weights the 30-patient class equally with the "
        "119-patient one; (c) the discretisation is a REPORTING step "
        "applied after training on the continuous panel mean -- not the "
        "training objective. An arm trained to maximise F1 would be a "
        "different arm"
    ),

    # ---- what cannot be compared to what -----------------------------
    "the_five_class_family_is_still_undefined": (
        "unchanged by the run (THREE_NOT_FIVE): two registered "
        "thresholds cut three parts, and a FOUR-threshold prediction "
        "rule is not in the record. Nothing here is a 5-class number"
    ),
    "not_comparable_with_cleftgnns_tables": (
        "**and this is a concrete trap, not a formality.** Phase 10's "
        "faithful arm reports macro F1 over FIVE classes "
        "(phase10.top1_macro_prf's default; FAITHFUL_ARM_CLOSING: "
        "0.077-0.154 for four raters, 0.3352 study / 0.2366 benchmark "
        "for the fifth). **0.3760 and 0.3352 look adjacent and are "
        "different quantities on every axis**: 3 classes vs 5, panel "
        "mean vs rater-specific labels, 237 all-out-of-fold vs an 85:15 "
        "single split, five seeds vs one run with no intervals. Neither "
        "may be quoted beside the other"
    ),
    "the_rule_now_has_a_second_instance": (
        "the same DIFFERENT-QUANTITIES rule that governs the PCC "
        "comparison (phase10.CLEFTGNN_COMPARATOR_TABLES, "
        "'quantity_distinct_from_0_2520') now has a second instance in "
        "a second metric family. **The rule is not about PCC** -- it is "
        "about two numbers sharing a NAME while measuring different "
        "things, and a metric family that was added to answer a "
        "question about comparability turns out to need the same "
        "warning the first one did"
    ),
    "what_it_settles": (
        "the question that prompted it: the metrics EXIST now, and the "
        "answer to 'why not precision, recall and F1' is no longer 'we "
        "did not compute them'. It is: they were computed, they are "
        "secondary for the pre-registered reasons, and computing them "
        "shows why the choice matters more than it looks -- a factor of "
        "9.58 between two descriptions of one model"
    ),
}


def confusion(truth, predicted, n_classes: int) -> np.ndarray:
    """Rows are truth, columns are prediction. Counts, no normalisation.

    Normalisation is left to the reader because the two normalisations
    answer different questions and neither is 'the' confusion matrix --
    row-normalised is recall per class, column-normalised is precision.
    """
    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if truth.shape != predicted.shape:
        raise ClassificationError(
            f"{truth.shape} truths against {predicted.shape} predictions"
        )
    if truth.size == 0:
        raise ClassificationError("no rows: nothing to tabulate")
    for name, array in (("truth", truth), ("predicted", predicted)):
        if array.min() < 0 or array.max() >= n_classes:
            raise ClassificationError(
                f"{name} carries a class outside 0..{n_classes - 1}; the "
                "discretisation and the class count disagree"
            )
    table = np.zeros((n_classes, n_classes), dtype=int)
    for actual, guess in zip(truth, predicted):
        table[actual, guess] += 1
    return table


def majority_baseline(truth, n_classes: int) -> dict:
    """The floor every figure is quoted against.

    Its macro F1 is reported too, and it is LOW by construction: a
    predictor that always answers one class has zero recall on every
    other, so macro F1 punishes it where accuracy does not. That
    contrast is the most useful single thing in this addendum.
    """
    truth = np.asarray(truth, dtype=int)
    support = np.bincount(truth, minlength=n_classes)
    majority_class = int(np.argmax(support))
    always = np.full(truth.shape, majority_class, dtype=int)
    report = prf_report(truth, always, n_classes, _check_phase10=False)
    return {
        "majority_class": majority_class,
        "accuracy": float(support.max() / support.sum()),
        "chance": 1.0 / int(np.count_nonzero(support)),
        "f1_macro": report["f1_macro"],
        "f1_weighted": report["f1_weighted"],
        "what_this_shows": (
            "a constant predictor scores its accuracy floor but a POOR "
            "macro F1 -- the two metrics disagree about the same "
            "trivial model, which is the disagreement this addendum "
            "exists to make visible"
        ),
    }


def prf_report(truth, predicted, n_classes: int,
               *, _check_phase10: bool = True) -> dict:
    """Per-class, macro and weighted precision / recall / F1.

    **The macro convention is phase10's, not a new one**, and the last
    step of this function asserts the two agree. Its two documented
    choices are inherited: a class with no predictions has precision 0,
    and a class with NO TRUTHS is excluded from the macro average --
    averaging over a class that cannot occur would report the model's
    silence as a score.

    Weighted averaging uses truth support as the weight, so an absent
    class contributes nothing and the exclusion question does not arise.
    """
    table = confusion(truth, predicted, n_classes)
    support = table.sum(axis=1)
    predicted_positive = table.sum(axis=0)
    true_positive = np.diag(table)

    per_class = {}
    for index in range(n_classes):
        precision = (
            float(true_positive[index] / predicted_positive[index])
            if predicted_positive[index] else 0.0
        )
        recall = (
            float(true_positive[index] / support[index])
            if support[index] else 0.0
        )
        f1 = (
            0.0 if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
        per_class[index] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[index]),
            "predicted": int(predicted_positive[index]),
        }

    present = [i for i in range(n_classes) if support[i]]
    absent = [i for i in range(n_classes) if not support[i]]

    def macro(key: str) -> float:
        return (
            float(np.mean([per_class[i][key] for i in present]))
            if present else 0.0
        )

    total = int(support.sum())

    def weighted(key: str) -> float:
        return float(
            sum(per_class[i][key] * support[i] for i in present) / total
        )

    report = {
        "n": total,
        "accuracy": float(true_positive.sum() / total),
        "precision_macro": macro("precision"),
        "recall_macro": macro("recall"),
        "f1_macro": macro("f1"),
        "precision_weighted": weighted("precision"),
        "recall_weighted": weighted("recall"),
        "f1_weighted": weighted("f1"),
        "support": [int(value) for value in support],
        "predicted_counts": [int(value) for value in predicted_positive],
        "per_class": {str(k): v for k, v in per_class.items()},
        "confusion": table.tolist(),
        "confusion_layout": "rows are TRUTH, columns are PREDICTION",
        "classes_present": present,
        "classes_absent_excluded_from_macro": absent,
    }

    if _check_phase10:
        # The cross-check: phase10 iterates grades 1..n, so 0-indexed
        # classes are shifted into its range. A divergence here means
        # two macro conventions exist in the repo, which is the defect
        # this call is placed to catch.
        theirs = phase10.top1_macro_prf(
            np.asarray(truth, dtype=int) + 1,
            np.asarray(predicted, dtype=int) + 1,
            n_classes=n_classes,
        )
        for mine, their_key in (
            ("precision_macro", "precision_macro"),
            ("recall_macro", "recall_macro"),
            ("f1_macro", "f1_macro"),
            ("accuracy", "accuracy"),
        ):
            if abs(report[mine] - theirs[their_key]) > 1e-12:
                raise ClassificationError(
                    f"{mine} is {report[mine]!r} here and "
                    f"{theirs[their_key]!r} in phase10.top1_macro_prf. "
                    "Two macro conventions in one repository; the "
                    "figures cannot both be quoted"
                )
        report["macro_convention"] = (
            "phase10.top1_macro_prf, asserted equal at every macro key"
        )
    return report


def discretise(values) -> np.ndarray:
    """The FROZEN 2.5/3.5 collapse, not a local copy of it."""
    return frozen_metrics.to_3class(np.asarray(values, dtype=float))


def summary() -> dict:
    """This addendum's records, importable as one object."""
    return {
        "registered": CLASSIFICATION_METRICS_SECONDARY,
        "three_not_five": THREE_NOT_FIVE,
        "banked": CLASSIFICATION_METRICS_BANKED,
    }
