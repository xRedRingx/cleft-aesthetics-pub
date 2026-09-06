"""Phase 11 -- the asymmetric loss (the "penalise optimism" request).

Opened 2026-08-17 under ``PLAN_AMENDMENT_2026-08-13`` section 6, whose
commitment is two things in sequence and not one:

1. **The pre-step**, free: "computing the asymmetric error on the
   predictions already on disk before building anything -- that costs
   nothing and says whether the ranking moves at all."
2. **The loss build**, which the amendment is explicit about: "it is a
   loss-function change, so it needs its own arms rather than a
   re-scoring of existing ones."

**Only the pre-step is registered here.** The loss build is unregistered
and BLOCKED, downstream of a direction question the group has not
answered (``phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION``).

The metric's constants come from ``phase10.IEM_CARRIED_FOR_PHASE_11``,
which took them from the manuscript's equation (16) at Phase 10's
opening -- before this phase existed and before any number here was
computed.
"""

from __future__ import annotations

#: Equation (16)'s crossover, exact: the magnitude at which the two
#: branches cross. ``1.2 |d|^1.12 == 0.8 |d|^0.87`` reduces to
#: ``|d|^0.25 == 0.8/1.2``, so ``|d| == (2/3)**4``.
IEM_CROSSOVER = (0.8 / 1.2) ** 4

#: The pre-step's two conventions. "A" is the manuscript's equation (16)
#: read literally on the GRADE axis; "B" swaps the two branches' constant
#: pairs, which is what the opposite direction convention amounts to.
CONVENTIONS = ("a_manuscript_literal", "b_swapped")


class Phase11Error(RuntimeError):
    """A Phase 11 contract is not usable."""


def iem(residual, convention: str = "a_manuscript_literal"):
    """Equation (16), for one convention, elementwise.

    ``residual`` is ``y_hat - G``. Convention A is the manuscript's own
    branch assignment (``phase10.IEM_CARRIED_FOR_PHASE_11``): the heavier
    ``1.2 * |d|**1.12`` falls on ``d < 0``, a prediction LOWER than the
    grade, which on 1=Excellent..5=Very Poor is the flattering one.
    Convention B is the same equation with the two constant pairs
    exchanged, which is exactly what the opposite reading of the
    direction amounts to -- **no third form is implemented, because no
    third form is documented.**
    """
    import numpy as np

    if convention not in CONVENTIONS:
        raise Phase11Error(
            f"convention {convention!r} is not one of {CONVENTIONS}; a "
            "third would need a third documented reading"
        )
    residual = np.asarray(residual, dtype=float)
    magnitude = np.abs(residual)
    heavy = 1.2 * magnitude ** 1.12
    light = 0.8 * magnitude ** 0.87
    below = residual < 0.0
    if convention == "b_swapped":
        below = ~below
    return np.where(below, heavy, light)


#: **[FINDING 2026-08-17] EQUATION (16) INVERTS ITS OWN STATED INTENT FOR
#: ERRORS BELOW |d| = 0.19753.**
#:
#: Computed from the constants ``phase10.IEM_CARRIED_FOR_PHASE_11``
#: recorded at Phase 10's opening, not from anything measured here:
#:
#:        |d|   optimistic   pessimistic    ratio
#:     0.0500       0.0419        0.0590    0.709
#:     0.1000       0.0910        0.1079    0.844
#:     0.1975       0.1951        0.1951    1.000   <- crossover
#:     0.2500       0.2540        0.2395    1.061
#:     1.0000       1.2000        0.8000    1.500
#:     4.0000       5.6688        2.6723    2.121
#:
#: The two branches cross at ``|d| = (0.8/1.2)**4 = 0.197531`` exactly,
#: and **below it the branch the metric calls more dangerous is penalised
#: LESS**. The metric is continuous at zero -- both branches vanish -- so
#: nothing is discontinuous; the asymmetry simply reverses.
#:
#: **Why it happens**: the heavier branch carries the larger weight (1.2 >
#: 0.8) but also the larger exponent (1.12 > 0.87), and for ``|d| < 1``
#: a larger exponent SHRINKS the value. Below the crossing the exponent
#: wins. The stated intent -- "optimistic errors are clinically more
#: dangerous and are penalised more heavily" -- therefore holds only for
#: errors above roughly a fifth of a grade.
#:
#: **Why it matters here, concretely**: the asymmetry the phase exists to
#: study is absent-to-reversed exactly where a well-calibrated model
#: spends most of its predictions. Any pre-step number must be read with
#: the share of ``|d| < 0.19753`` beside it, which is why exit criterion
#: 2 carries it with every figure rather than in a footnote.
#:
#: **supervision material.** This is a property of the published equation, not of
#: our implementation of it, and it is stated as an observation rather
#: than a correction: whether the crossing is intended, an artifact of
#: fitting the exponents, or something they would fix on revision is
#: theirs to say.
IEM_CROSSOVER_INVERTS = {
    "finding": "2026-08-17",
    "crossover": IEM_CROSSOVER,
    "closed_form": "|d| = (0.8/1.2)**4, from 1.2|d|^1.12 == 0.8|d|^0.87",
    "below_it": (
        "the branch the metric calls more dangerous is penalised LESS -- "
        "at |d| = 0.1 the optimistic branch scores 0.0910 against the "
        "pessimistic 0.1079"
    ),
    "continuous_at_zero": (
        "both branches vanish at d = 0, so nothing is discontinuous; the "
        "asymmetry reverses rather than jumping"
    ),
    "why": (
        "the heavier branch carries the larger weight (1.2 > 0.8) AND the "
        "larger exponent (1.12 > 0.87), and for |d| < 1 a larger exponent "
        "SHRINKS the value. Below the crossing the exponent wins"
    ),
    "so_the_stated_intent": (
        "'optimistic errors are clinically more dangerous and are "
        "penalised more heavily' holds only above roughly a fifth of a "
        "grade"
    ),
    "why_it_matters_here": (
        "the asymmetry this phase studies is absent-to-reversed exactly "
        "where a well-calibrated model spends most of its predictions, so "
        "every pre-step number is reported with the share of |d| below "
        "the crossover beside it (exit criterion 2)"
    ),
    "supervisor_material": (
        "a property of the PUBLISHED equation, not of our implementation. "
        "Stated as an observation, not a correction: whether the crossing "
        "is intended, an artifact of fitting the exponents, or something "
        "they would fix on revision is theirs to say"
    ),
}


#: **[CORRECTION 2026-08-17] THE PAPER'S [0, 4] IS A DOMAIN, NOT A
#: RANGE.**
#:
#: ``phase10.IEM_CARRIED_FOR_PHASE_11`` records "bounds noted in the
#: paper: [0, 4]". Checked against the equation: at ``|d| = 4`` the
#: branches give **5.6688** and **2.6723**, so IEM itself leaves [0, 4]
#: on the optimistic branch. What [0, 4] bounds is ``|y_hat - G|`` -- the
#: largest possible error on a 1..5 grade scale.
#:
#: **The consequence for the pre-step, and it is exit criterion 5**: the
#: domain bound holds by construction only if every arm's predictions
#: lie in [1, 5]. Softmax-expected grades do by definition; a regression
#: head's outputs do not necessarily, and no repo record establishes that
#: the ladder's heads clip. So it is ASSERTED per arm and any breach is
#: reported as a finding rather than silently clipped -- clipping would
#: change the numbers, and hiding a breach would let a metric run outside
#: the domain its own paper states.
#:
#: **supervision material**, mildly: if [0, 4] was meant as the metric's range,
#: the constants and the range disagree.
IEM_BOUNDS_ARE_A_DOMAIN = {
    "correction": "2026-08-17",
    "recorded_as": "phase10.IEM_CARRIED_FOR_PHASE_11 bounds (0, 4)",
    "measured": (
        "at |d| = 4 the branches give 5.6688 and 2.6723, so IEM leaves "
        "[0,4] on the optimistic branch"
    ),
    "what_it_bounds": (
        "|y_hat - G|, the largest possible error on a 1..5 grade scale -- "
        "a DOMAIN, not a range"
    ),
    "consequence": (
        "the bound holds by construction only if every arm's predictions "
        "lie in [1,5]. Softmax-expected grades do by definition; a "
        "regression head's outputs do not necessarily, and no repo record "
        "establishes that the ladder's heads clip. ASSERTED per arm, "
        "breaches reported as a finding rather than silently clipped"
    ),
    "supervisor_material": (
        "if [0,4] was meant as the metric's range, the constants and the "
        "range disagree"
    ),
}


#: **[OPEN IDENTITY QUESTION 2026-08-17, TAGGED REASONED] THREE NAMES ARE
#: TREATED AS ONE METRIC BY ASSUMPTION.**
#:
#: * the amendment: "**CASE**" -- ``PLAN_AMENDMENT_2026-08-13`` section 1's
#:   request table and section 6, "The shared deck's CASE metric";
#: * the manuscript: "**IEM**", equation (16), the only one of the three
#:   with an equation attached;
#: * the deck, slides 10-12: no name at all -- "Metric that penalized
#:   underprediction: if prediction gives always higher score than it
#:   should be, then penalize".
#:
#: Nothing in the repository establishes that these denote one metric.
#: ``phase10.IEM_IS_THEIRS_DIRECTION_AMBIGUOUS`` treats them as one, and
#: that is an ASSUMPTION carried forward, not a measurement -- tagged
#: **REASONED** so a later reader does not mistake it for something
#: checked.
#:
#: **The pre-step proceeds on equation (16)** because it is the only
#: artifact of the three with an equation, and a metric without an
#: equation cannot be computed. If CASE turns out to be a different
#: metric, the pre-step measured IEM and said so, which is recoverable;
#: guessing at CASE would not be.
#:
#: **On the supervision material's ask list.**
CASE_IEM_IDENTITY_OPEN = {
    "opened": "2026-08-17",
    "tag": "REASONED",
    "three_names": (
        "the amendment's CASE (section 1's table, section 6's 'the shared "
        "deck's CASE metric')",
        "the manuscript's IEM, equation (16) -- the only one of the three "
        "with an equation attached",
        "the deck's unnamed 'Metric that penalized underprediction' "
        "(slides 10-12)",
    ),
    "the_assumption": (
        "phase10.IEM_IS_THEIRS_DIRECTION_AMBIGUOUS treats them as ONE "
        "metric; nothing in the repository establishes it"
    ),
    "the_pre_step_proceeds_on": (
        "equation (16), the only artifact with an equation -- a metric "
        "without an equation cannot be computed. If CASE is a different "
        "metric, the pre-step measured IEM and said so, which is "
        "recoverable; guessing at CASE would not be"
    ),
    "for_supervisor": "is CASE the same metric as the manuscript's eq (16) IEM?",
}


#: **[INFERENCE 2026-08-17, RECORDED AS AN INFERENCE] THE QUALITY-AXIS
#: READING RECONCILES ALL THREE SOURCES AND PREDICTS CONVENTION A.**
#:
#: The direction block (``phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION``)
#: stands. This is not a discharge of it; it is a prediction, registered
#: so that the supervision material's eventual sentence CONFIRMS OR REFUTES something rather
#: than landing in a vacuum.
#:
#: A third voice, found while restating the amendment: **"The shared
#: deck's CASE metric weights overprediction more heavily than
#: underprediction -- a model that flatters a result is penalised more
#: than one that is harsh."** Those two clauses agree only on the QUALITY
#: axis: flattering means predicting better quality, which on
#: 1=Excellent..5=Very Poor is a LOWER number, i.e. ``y_hat - G < 0``.
#:
#: On that reading all three sources say the same thing:
#:
#:     manuscript eq (16)   1.2 on y_hat - G < 0        flattering
#:     deck slides 10-12    "higher score than it should be"  flattering
#:     amendment section 6  "a model that flatters"     flattering
#:
#: and equation (16) as printed already implements it -- which is
#: **convention A**.
#:
#: **Why this does NOT unblock the loss build.** It is an argument about
#: what words refer to, and the registered failure mode is precisely that
#: the wrong choice leaves every number plausible. An inference of that
#: shape is what the block exists to refuse acting on. The pre-step runs
#: BOTH conventions regardless, and neither ranking is quotable alone.
IEM_DIRECTION_INFERENCE = {
    "recorded": "2026-08-17, an INFERENCE, not a resolution",
    "third_voice": (
        "PLAN_AMENDMENT_2026-08-13 section 6: 'weights overprediction "
        "more heavily than underprediction -- a model that flatters a "
        "result is penalised more than one that is harsh'"
    ),
    "the_reconciliation": (
        "the two clauses agree only on the QUALITY axis: flattering means "
        "predicting better quality, a LOWER number on "
        "1=Excellent..5=Very Poor, i.e. y_hat - G < 0. On that reading "
        "the manuscript, the deck and the amendment all say the same "
        "thing, and eq (16) as printed already implements it"
    ),
    "predicts": "convention A (a_manuscript_literal)",
    "does_not_unblock": (
        "an argument about what words refer to, and the registered "
        "failure mode is that the wrong choice leaves every number "
        "plausible -- exactly the shape the block exists to refuse acting "
        "on. Both conventions run regardless; neither ranking is quotable "
        "alone"
    ),
    "why_recorded_now": (
        "so the sentence given at supervision confirms or refutes a PREDICTION rather than "
        "landing in a vacuum"
    ),
    # [CONFIRMED 2026-08-17] The prediction was registered before the
    # answer arrived and the answer matches it. IEM_DIRECTION_ANSWERED.
    "confirmed": (
        "2026-08-17 -- convention A, exactly as predicted. The prediction "
        "was registered BEFORE the answer and is dated so; see "
        "IEM_DIRECTION_ANSWERED"
    ),
}


#: **[ANSWERED 2026-08-17, supervision via the maintainer] CONVENTION A. THE
#: BLOCK IS LIFTED.**
#:
#: **The meaning, in the terms given at supervision**: if the model says a repair looks
#: great when it actually looks poor, a clinician might not investigate.
#: The reverse error is safer. On the 1 = Excellent .. 5 = Very Poor scale
#: that is a prediction LOWER than the truth -- ``y_hat - G < 0`` -- which
#: takes the heavier ``1.2 * |d|**1.12`` branch.
#:
#: **The manuscript's equation as printed implements the intent.** No
#: correction to eq (16)'s branch assignment is needed, and none is made.
#:
#: **The prediction was registered before the answer.**
#: ``IEM_DIRECTION_INFERENCE`` reasoned from the amendment's third voice
#: that the three sources reconcile on the QUALITY axis and that this
#: predicts convention A. Dated before, confirmed after. That is the point
#: of having written it down: the answer confirms something rather than
#: filling a vacuum.
#:
#: **The deck's wording is the outlier, and is recorded RESOLVED-AGAINST
#: rather than deleted.** Slides 10-12 say "Metric that penalized
#: underprediction: if prediction gives always higher score than it should
#: be, then penalize". Under the settled reading, "score" means QUALITY,
#: so "higher score than it should be" is a flattering prediction and
#: agrees; "penalized underprediction" reads on the GRADE axis and does
#: not. The slide mixes the two axes within one sentence. **It is kept
#: because it is what the artifact says**, and a future reader meeting
#: that slide should find it already reconciled rather than rediscover the
#: ambiguity.
IEM_DIRECTION_ANSWERED = {
    "answered": "2026-08-17, supervision via the maintainer",
    "answer": "convention A",
    "supervisor_reasoning": (
        "if the model says a repair looks great when it actually looks "
        "poor, a clinician might not investigate; the reverse error is "
        "safer"
    ),
    "on_the_scale": (
        "1 = Excellent .. 5 = Very Poor, so that is a prediction LOWER "
        "than the truth, y_hat - G < 0, which takes the heavier "
        "1.2 * |d|**1.12 branch"
    ),
    "the_equation_as_printed_is_right": (
        "the manuscript's eq (16) implements the intent; no correction to "
        "its branch assignment is needed and none is made"
    ),
    "predicted_before_answered": (
        "IEM_DIRECTION_INFERENCE reasoned from the amendment's third "
        "voice that the three sources reconcile on the QUALITY axis and "
        "that this predicts convention A -- dated before, confirmed "
        "after"
    ),
    "the_deck_is_resolved_against_not_deleted": (
        "slides 10-12's 'penalized underprediction / higher score than it "
        "should be' mixes the two axes in one sentence: 'score' as "
        "QUALITY agrees, 'underprediction' as GRADE does not. Kept "
        "because it is what the artifact says, so a future reader meeting "
        "that slide finds it already reconciled rather than rediscovering "
        "the ambiguity"
    ),
    "block_lifted": (
        "phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION; the loss build is no "
        "longer blocked on the direction, and the question is off the supervision "
        "ask list"
    ),
}


#: **[REGISTERED 2026-08-17, BEFORE ANY ARM IS BUILT] CONVENTION A IS THE
#: DIRECTION THAT FAVOURS HEDGING -- AND THE LOSS INHERITS THAT PULL.**
#:
#: The direction is settled, and the pre-step already measured what this
#: particular direction does. **Registering it now, before an arm exists,
#: is the whole point**: the same measurement arriving after a loss arm
#: scored well would be an excuse.
#:
#: **What is already measured** (``A_FAVOURS_NARROW_PREDICTORS``): under
#: convention A, across 68 arms, the winners are narrow (median span
#: **1.07** against the ladder's **2.26**), narrow arms land in the
#: inverted region **1.685x** as often as the ladder median, and the
#: metric crowns an arm at **PCC 0.024**. A loss built on eq (16) under
#: convention A therefore carries a **pull toward hedging just under the
#: modal grade**.
#:
#: **And the pull exists in the GRADIENT, not only in the score.**
#: Measured from the published constants:
#:
#:     d/dd [0.8 |d|^0.87] = 0.696 |d|^-0.13     (pessimistic branch)
#:     d/dd [1.2 |d|^1.12] = 1.344 |d|^0.12      (optimistic branch)
#:
#: These cross at **|d| = (0.696/1.344)**4 = 0.0719**. Below that, the
#: PESSIMISTIC branch has the STRONGER gradient: at |d| = 0.01 it is
#: 1.2665 against 0.7734. So for small errors the loss pushes harder to
#: remove an over-prediction than an under-prediction -- **it actively
#: drives predictions downward toward optimism near the truth** -- and the
#: optimistic branch's exponent (1.12 > 1) means its gradient VANISHES at
#: zero, so nothing pushes back. The value-level inversion below 0.198 and
#: the gradient-level inversion below 0.072 are the same defect seen from
#: the scoring and the training side.
#:
#: **THE READING, COMMITTED NOW.** An arm that improves on IEM **while its
#: prediction span narrows and its PCC falls** is **the degenerate optimum
#: realised, not a result.** It must be reported as **the metric's
#: behaviour, not the model's** -- the loss did what it rewards, which is
#: a finding about eq (16) and not a finding about the architecture. No
#: such arm may be quoted as an improvement, ledgered, or compared
#: favourably against the ladder.
#:
#: **THE DIAGNOSTIC.** The maintainer's expectation -- prediction span and
#: below-crossover share per seed alongside the loss -- is right in
#: content and I would change one thing and add two:
#:
#: 1. **Per EPOCH, not per seed.** The harness already computes a per-epoch
#:    curve and Phase 10 already writes it; adding span, below-crossover
#:    share and inner-val PCC to that file costs no extra forward. Per-seed
#:    endpoints answer "is this arm degenerate?"; per-epoch answers "did
#:    the loss PULL it there?", and only the second distinguishes the
#:    metric's behaviour from the model's. The degeneracy is a trajectory.
#: 2. **A pre-registered joint trigger**, so three columns are not
#:    eyeballed. Per fold, the degenerate signature FIRES if, between
#:    epoch 0 and the selected epoch, span DECREASES **and**
#:    below-crossover share INCREASES **and** inner-val PCC DECREASES --
#:    all three, same fold. Report the count out of the fold total, with
#:    the threshold registered before the run rather than read off it.
#: 3. **What the diagnostic cannot do alone**: distinguish "the IEM loss
#:    pulled this arm toward hedging" from "this arm hedges under any
#:    loss". That needs a **matched control arm trained on MSE under
#:    identical conditions**, and whether one runs is a build silence
#:    (``PHASE_11_BUILD_SCOPE``) rather than something to decide here.
DEGENERACY_PULL_REGISTERED = {
    "registered": "2026-08-17, before any arm is built",
    "why_now": (
        "the same measurement arriving after a loss arm scored well would "
        "be an excuse"
    ),
    "already_measured": (
        "under convention A across 68 arms the winners are narrow (median "
        "span 1.07 against the ladder's 2.26), narrow arms land in the "
        "inverted region 1.685x as often, and the metric crowns an arm at "
        "PCC 0.024 (A_FAVOURS_NARROW_PREDICTORS)"
    ),
    "the_gradient_pull": {
        "pessimistic": "d/dd [0.8|d|^0.87] = 0.696 |d|^-0.13",
        "optimistic": "d/dd [1.2|d|^1.12] = 1.344 |d|^0.12",
        "gradient_crossover": 0.0719,
        "below_it": (
            "the PESSIMISTIC branch has the STRONGER gradient -- at |d| = "
            "0.01, 1.2665 against 0.7734 -- so for small errors the loss "
            "pushes harder to remove an over-prediction than an "
            "under-prediction, driving predictions downward toward "
            "optimism near the truth"
        ),
        "and_nothing_pushes_back": (
            "the optimistic branch's exponent 1.12 > 1 means its gradient "
            "VANISHES at zero"
        ),
        "same_defect_twice": (
            "the value-level inversion below 0.198 and the gradient-level "
            "inversion below 0.072 are the same defect seen from the "
            "scoring and the training side"
        ),
    },
    "reading_committed": (
        "an arm that improves on IEM WHILE its prediction span narrows "
        "and its PCC falls is THE DEGENERATE OPTIMUM REALISED, NOT A "
        "RESULT. It is reported as the METRIC's behaviour, not the "
        "model's -- the loss did what it rewards, which is a finding "
        "about eq (16) and not about the architecture. Such an arm may "
        "not be quoted as an improvement, ledgered, or compared "
        "favourably against the ladder"
    ),
    "diagnostic": {
        "per_epoch_not_per_seed": (
            "the harness already computes a per-epoch curve and Phase 10 "
            "already writes it; adding span, below-crossover share and "
            "inner-val PCC costs no extra forward. Per-seed endpoints "
            "answer 'is this arm degenerate?'; per-epoch answers 'did the "
            "loss PULL it there?', and only the second separates the "
            "metric's behaviour from the model's. The degeneracy is a "
            "TRAJECTORY"
        ),
        "joint_trigger": (
            "per fold the signature FIRES if, between epoch 0 and the "
            "selected epoch, span DECREASES and below-crossover share "
            "INCREASES and inner-val PCC DECREASES -- all three, same "
            "fold. Report the count out of the fold total, with the "
            "threshold registered before the run rather than read off it"
        ),
        "what_it_cannot_do_alone": (
            "distinguish 'the IEM loss pulled this arm toward hedging' "
            "from 'this arm hedges under any loss'. That needs a MATCHED "
            "CONTROL ARM trained on MSE under identical conditions, and "
            "whether one runs is a build silence rather than something to "
            "decide here"
        ),
        "the_expectation": (
            "span and below-crossover share per seed alongside the loss "
            "-- right in content; the change is per-epoch, and the "
            "additions are the joint trigger and the control-arm caveat"
        ),
    },
}


#: **[REGISTERED 2026-08-17, BEFORE ANY NUMBER] THE PRE-STEP.**
#:
#: Every decision below was taken before the task existed. The silences
#: they close were listed at the phase's opening and are named here with
#: their resolutions, so a reader can see which were decided rather than
#: defaulted.
PRE_STEP_REGISTERED = {
    "registered": "2026-08-17, before any number",
    "question": (
        "the amendment's: does every existing arm's RANKING move under "
        "the asymmetric error?"
    ),
    "metric": (
        "manuscript eq (16), constants from "
        "phase10.IEM_CARRIED_FOR_PHASE_11 -- recorded at Phase 10's "
        "opening, before this phase existed"
    ),
    "both_conventions": (
        "A = eq (16) literal (heavier branch on y_hat - G < 0); B = the "
        "two constant pairs exchanged. Both computed, both reported, "
        "NEITHER quotable alone"
    ),
    "truth_is_the_median_grade": {
        "decision": (
            "the score sheet's Median column, not the panel mean -- read "
            "by scoresheet.load_median and verified row by row against "
            "the recomputed median of the five rater cells"
        ),
        "why": (
            "IEM is defined against a consensus GRADE G; the median is "
            "what the sheet carries and what Phase 10 trained on; and "
            "scoring a grade-defined metric against a continuous mean "
            "would MANUFACTURE small |d| precisely where the crossover "
            "inverts the metric (IEM_CROSSOVER_INVERTS)"
        ),
        "the_csv_truth_column_is_not_G": (
            "the declared vectors carry the panel mean in their truth "
            "column; the pre-step uses it ONLY as the loader's "
            "cross-arm patient-and-label consistency check, never as G"
        ),
    },
    "coverage": (
        "ALL declared arms, no subset -- the amendment asks whether EVERY "
        "existing arm's ranking moves"
    ),
    "seed_bands": (
        "each arm pooled over its OWN band (five seeds or ten), with the "
        "shared-five figures reported beside them so neither choice is "
        "silent"
    ),
    "ranking_rule": (
        "per-patient IEM averaged within a seed, then pooled over seeds "
        "-- the way PCC is pooled -- and ranked by IEM alone. IEM-vs-PCC "
        "rank disagreement is a SEPARATE column, because that is the "
        "amendment's actual question"
    ),
    "the_mae_control_is_ours": (
        "ADDED, and flagged as ours: mean |d| against the same median "
        "grade, ranked too. Without it the IEM-vs-PCC disagreement "
        "conflates TWO changes -- the target (panel mean -> median grade) "
        "and the metric family (correlation -> asymmetric error). "
        "PCC-vs-MAE isolates the first, MAE-vs-IEM the second, which is "
        "the asymmetry this phase is actually about"
    ),
    "status": "DESCRIPTIVE ONLY -- no claim, no ledger row for a ranking",
    "why_descriptive": (
        "the pre-step tells us whether to care, not what is true. A "
        "ranking has no interval and PLAN 4.3 does not admit one"
    ),
    "deferred": (
        "FGCM (eqs. 14-15) and NDCG@K (eqs. 17-18) -- available, recorded, "
        "NOT computed. One metric with a known defect is enough to "
        "interpret in one pass"
    ),
    "blocked_downstream": (
        "the loss build is UNREGISTERED and BLOCKED on the direction "
        "question (phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION); no arm is "
        "built in this phase until it is answered"
    ),
}


#: **[REGISTERED 2026-08-17, BEFORE THE NUMBERS] THE READINGS, AND THE
#: POWER CHECK THAT DECIDES WHICH ONE APPLIES.**
#:
#: **The rule for "the ranking moves", fixed now so it cannot be chosen
#: later**: the ranking MOVES if the **top-5 set differs** between the two
#: conventions, **or** Kendall tau-b between the two rankings is
#: **< 0.90**. Both are reported exactly whichever way they fall.
#:
#: **Reading 1 -- same ranking under both**: the ambiguity does not bite
#: for the ranking question, and the pre-step's conclusion holds
#: regardless of the supervision answer. **Conditional on the power check below.**
#:
#: **Reading 2 -- different rankings**: the measured argument that the
#: direction must be settled before any loss is built. The build stays
#: blocked with a number to show for it rather than a shrug.
#:
#: **THE POWER CHECK, and it is why reading 1 is conditional.** The two
#: conventions differ only by exchanging the branch constants, so for an
#: arm whose residuals are SYMMETRIC about zero they give near-identical
#: totals **by construction**. A same-ranking result would then mean "our
#: arms are too symmetric for the ambiguity to bite here", which is not
#: the same statement as "the ambiguity does not bite" and does not
#: license reading 1.
#:
#: So, registered now: if
#: ``max over arms of |IEM_A - IEM_B| / IEM_A < 0.01``,
#: the test had **NO POWER** and a same-ranking result is recorded as
#: **UNINFORMATIVE**, not as reading 1. Residual skew and the
#: below-crossover share are reported per arm as the explanation either
#: way.
PRE_STEP_READINGS = {
    "registered": "2026-08-17, before the numbers",
    "ranking_moves_if": (
        "the top-5 SET differs between conventions, OR Kendall tau-b "
        "between the two rankings is < 0.90. Both reported exactly, "
        "whichever way they fall"
    ),
    "reading_1_same_ranking": (
        "the ambiguity does not bite for the ranking question and the "
        "pre-step's conclusion holds regardless of the supervision answer -- "
        "CONDITIONAL on the power check"
    ),
    "reading_2_different_rankings": (
        "the measured argument that the direction must be settled before "
        "any loss is built; the build stays blocked with a number to show "
        "for it rather than a shrug"
    ),
    "power_check": {
        "why": (
            "the two conventions differ only by exchanging branch "
            "constants, so for an arm with residuals SYMMETRIC about zero "
            "they agree by construction. A same-ranking result would then "
            "mean 'our arms are too symmetric for it to bite here', which "
            "is not 'the ambiguity does not bite'"
        ),
        "rule": (
            "if max over arms of |IEM_A - IEM_B| / IEM_A < 0.01 the test "
            "had NO POWER and a same-ranking result is UNINFORMATIVE, not "
            "reading 1"
        ),
        "threshold": 0.01,
        "explanatory_columns": (
            "residual skew and the below-crossover share, per arm, "
            "reported either way"
        ),
    },
}


#: **[REGISTERED 2026-08-17, BEFORE ANY WORK -- the Phase 9 lesson, and
#: Phase 10's silence, applied] PHASE 11'S EXIT CRITERIA.**
#:
#: Phase 10 opened with "exit criteria -- none exist; they need writing
#: before the phase's work starts" among its silences, and closed against
#: six criteria written at its registration turn. These are written
#: before the pre-step's task exists, let alone runs.
PHASE_11_EXIT_CRITERIA = {
    "registered": "2026-08-17, before any work",
    "criteria": (
        "1. BOTH conventions computed over every declared arm, both "
        "rankings reported, and NEITHER quotable alone -- both travel "
        "together or neither is quoted",
        "2. the crossover defect (IEM_CROSSOVER_INVERTS) recorded as a "
        "finding and CARRIED WITH EVERY NUMBER, including the "
        "below-crossover share per arm -- not a footnote",
        "3. residual skew reported per arm, and the registered POWER "
        "CHECK applied, so a same-ranking result is read as informative "
        "or UNINFORMATIVE by the rule rather than by preference",
        "4. the ranking-movement question answered DESCRIPTIVELY -- no "
        "claim, no ledger row for a ranking",
        "5. every arm's predictions ASSERTED within [1, 5] and any breach "
        "reported as a finding (IEM_BOUNDS_ARE_A_DOMAIN)",
        "6. the loss build still BLOCKED on the direction question, and "
        "recorded as blocked rather than quietly not done",
        "7. suite green",
    ),
    "expected_count": 7,
    "not_criteria": (
        "a ranking that moves is not a result to defend and a ranking "
        "that holds is not a success -- criterion 4 is what keeps either "
        "from becoming one",
        "FGCM and NDCG@K are deferred, not missing (PRE_STEP_REGISTERED)",
    ),
}


#: **[RESTATED 2026-08-17 from the amendment, silences LISTED then
#: RESOLVED -- the t-SNE discipline] Phase 11's scope, and what the
#: opening record did not fix.**
PHASE_11_SCOPE_AND_SILENCES = {
    "restated": "2026-08-17, from PLAN_AMENDMENT_2026-08-13 section 6",
    "commits": {
        "pre_step": (
            "'computing the asymmetric error on the predictions already "
            "on disk before building anything -- that costs nothing and "
            "says whether the ranking moves at all'"
        ),
        "loss_build": (
            "'it is a loss-function change, so it needs its own arms "
            "rather than a re-scoring of existing ones'"
        ),
        "motivation": "'clinically well-motivated'",
    },
    "exists_to_build_from": (
        "68 arms and 490 hash-verified per-seed OOF prediction vectors "
        "across the shipped configs -- Phase 7 D/D1/E, 7C augmentation, "
        "7D multi-scale, Road B resolution and region-crop, and "
        "p10_cleftgnn's five",
        "phase7c.load_oof_vectors, with its cross-file truth check to "
        "1e-9 -- no new loader is needed",
        "scoresheet.load_median, which verifies the published Median "
        "against the recomputed median of the five rater cells",
        "eq (16)'s constants, recorded at Phase 10's opening",
    ),
    # Listed at the opening, resolved by the maintainer on the same day.
    # Kept in one place with their resolutions so "decided" is
    # distinguishable from "defaulted".
    "silences_resolved": {
        "exit_criteria": "PHASE_11_EXIT_CRITERIA, seven, written first",
        "case_vs_iem": "open identity question, REASONED, on the supervision material's list",
        "truth": "the median grade, with the crossover interaction as the reason",
        "coverage": "all 68 arms, 490 vectors, no subset",
        "seed_bands": "own band, with shared-five beside",
        "ranking_rule": "IEM alone; IEM-vs-PCC disagreement a separate column",
        "status": "DESCRIPTIVE only",
        "domain": "asserted per arm, breaches are findings",
        "config_shape": "ONE config -- see PRE_STEP_CONFIG_SHAPE",
        "fgcm_ndcg": "available, deferred",
        "loss_build": "unregistered and blocked",
        "direction": "both conventions; the inference recorded as an inference",
    },
    "still_open": (
        "the direction question itself -- supervision, one sentence "
        "(phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION)",
        "the CASE/IEM identity -- supervision (CASE_IEM_IDENTITY_OPEN)",
        "everything downstream of the loss build",
    ),
    # [2026-08-17] The first of the three closed.
    "still_open_resolved": (
        "2026-08-17: the direction question is ANSWERED -- convention A "
        "(IEM_DIRECTION_ANSWERED). The CASE/IEM identity stays open, and "
        "the loss build is now scoped rather than blocked "
        "(PHASE_11_BUILD_SCOPE)"
    ),
}


#: **[DECIDED 2026-08-17] ONE CONFIG, NOT SIX -- and the reason is the
#: deliverable, not the file size.**
#:
#: 492 declared inputs (490 vectors, the manifest, the score sheet) is
#: unprecedented here; the largest paired config declares 40. Per-scope
#: configs were the safer-looking shape and are the wrong one.
#:
#: **The ranking is a single cross-arm object.** Splitting the run into
#: six scopes produces six PARTIAL rankings that must then be merged by
#: hand, outside any declared artifact, to answer the question the phase
#: was opened to answer. That is precisely the computed-then-reassembled
#: gap this project has now hit three times -- the probe that rebuilt the
#: pipeline, the 25 discarded curves, and the U-shape reconstructed from a
#: run that had already measured it. The merge would be the deliverable,
#: and it would live nowhere.
#:
#: **The cost of one config is file size, not risk.** All 490 hashes are
#: already verified in shipped configs, so the generator carries every one
#: by path and **no new declaration is needed**. Guard 3 verifies 490
#: files instead of 40, which is the same operation more times.
#:
#: **What one config does cost, stated**: any single drift blocks the
#: whole pre-step rather than one scope of it. Accepted -- a pre-step that
#: runs on five sixths of the ladder answers a different question.
PRE_STEP_CONFIG_SHAPE = {
    "decided": "2026-08-17",
    "shape": "ONE config, configs/p11_iem_prestep.yaml, 492 declared inputs",
    "rejected": "per-scope configs",
    "why": (
        "the ranking is a single cross-arm object; six scopes produce six "
        "PARTIAL rankings that must be merged by hand outside any "
        "declared artifact -- the computed-then-reassembled gap this "
        "project has hit three times, with the merge as the deliverable "
        "and nowhere to live"
    ),
    "cost_is_size_not_risk": (
        "all 490 hashes are already verified in shipped configs, so the "
        "generator carries every one by path and NO new declaration is "
        "needed; guard 3 verifies 490 files instead of 40, the same "
        "operation more times"
    ),
    "what_it_does_cost": (
        "any single drift blocks the whole pre-step rather than one scope "
        "of it. Accepted: a pre-step that runs on five sixths of the "
        "ladder answers a different question"
    ),
}


#: **[OBSERVED 2026-08-17, ``p11_iem_prestep__b0de1791__p11-iem-prestep``]
#: READING 2 ON BOTH BANDS: THE RANKING MOVES.**
#:
#: ``tau-b`` between the two conventions **0.5917** (own bands) and
#: **0.5487** (shared five) -- both far below the registered 0.90 -- with
#: the top-5 sets **disjoint** on both. The power check passes: the
#: maximum relative gap between the conventions is **0.1187**, twelve
#: times the registered 0.01 floor, so this is reading 2 and not the
#: uninformative branch.
#:
#: **The two conventions name opposite winners.**
#:
#:     A (eq 16 literal)   p10_cleftgnn, p7_d1_srgnn_scut_original_g1_
#:                         native, p7_d_srgnn_scut_masked_g2_native,
#:                         p7_e_srgnn_scut_masked_g2_grid,
#:                         p7_e_srgnn_scut_masked_g2_random
#:     B (swapped)         0_identity, p7_d1_vit_b16_imagenet_g1,
#:                         roadb_p7_arm_vit_b16_imagenet_224,
#:                         roadb_p7c_arm_anatomy_concat_vit, + one
#:                         band-specific
#:
#: **Convention A ranks ``p10_cleftgnn`` FIRST** -- the arm whose PCC is
#: 0.024 and whose predictions span 2.5-3.0. Convention B puts the
#: project's best arm (0.2520) second. A metric whose direction is
#: unsettled therefore does not merely reorder the ladder; it inverts
#: which end of it is good.
#:
#: **THE REGISTERED READING, APPLIED**: the direction must be settled
#: before any loss is built, and the build stays blocked **with a number
#: to show for it** rather than a shrug. That is exactly the outcome
#: ``PRE_STEP_READINGS`` registered before the run.
PRE_STEP_OBSERVED = {
    "observed": "2026-08-17, p11_iem_prestep__b0de1791__p11-iem-prestep",
    "verdict": "READING 2 on both bands -- the ranking MOVES",
    "tau_b_between_conventions": {"own": 0.5917, "shared5": 0.5487},
    "registered_threshold": 0.90,
    "top5_disjoint": True,
    "power": {
        "max_relative_gap": 0.1187,
        "floor": 0.01,
        "sufficient": True,
        "reading": (
            "twelve times the registered floor, so this is reading 2 and "
            "NOT the uninformative branch"
        ),
    },
    "top5_a": (
        "p10_cleftgnn", "p7_d1_srgnn_scut_original_g1_native",
        "p7_d_srgnn_scut_masked_g2_native",
        "p7_e_srgnn_scut_masked_g2_grid",
        "p7_e_srgnn_scut_masked_g2_random",
    ),
    "top5_b": (
        "0_identity", "p7_d1_vit_b16_imagenet_g1",
        "roadb_p7_arm_vit_b16_imagenet_224",
        "roadb_p7c_arm_anatomy_concat_vit",
        "one band-specific",
    ),
    "the_inversion": (
        "convention A ranks p10_cleftgnn FIRST -- PCC 0.024, predictions "
        "spanning 2.5-3.0 -- while convention B puts the project's best "
        "arm second. An unsettled direction does not merely reorder the "
        "ladder; it inverts which end of it is good"
    ),
    "registered_reading_applied": (
        "the direction must be settled before any loss is built, and the "
        "build stays blocked WITH A NUMBER TO SHOW FOR IT rather than a "
        "shrug -- the outcome PRE_STEP_READINGS registered before the run"
    ),
}


#: **[OBSERVED 2026-08-17] THE MAE CONTROL WORKED: MOST OF THE
#: IEM-vs-PCC DISAGREEMENT IS NOT THE ASYMMETRY.**
#:
#: The control column existed because ``IEM(median grade)`` differs from
#: ``PCC(panel mean)`` in TWO ways at once -- the target and the error
#: family -- and the phase is about only the second. Measured:
#:
#:     tau(MAE, PCC)      0.232    target + error-family change
#:     tau(IEM_A, MAE)    0.845    the ASYMMETRY alone
#:     tau(IEM_A, PCC)    0.153 / 0.169   both together
#:
#: **The decomposition**: moving from a correlation against the panel
#: mean to an absolute error against the median grade costs most of the
#: agreement (tau falls to 0.232). Adding the asymmetry on top of that
#: costs comparatively little (0.845). So the amendment's question --
#: "does the ranking move under the asymmetric error?" -- has a two-part
#: answer that would have been one indistinguishable number without the
#: control: **the ranking moves a great deal, and mostly for reasons that
#: are not the asymmetry.**
#:
#: **A consistency check that came free.** ``tau(A, MAE) = 0.845`` is much
#: higher than ``tau(A, B) = 0.5917``, which is what a symmetric metric
#: sitting BETWEEN two mirror-image conventions should look like. Had MAE
#: landed outside that interval the implementation would have been wrong
#: somewhere, and nothing else in the run would have said so.
MAE_CONTROL_DECOMPOSITION = {
    "observed": "2026-08-17",
    "tau_mae_vs_pcc": 0.232,
    "tau_iem_a_vs_mae": 0.845,
    "tau_iem_a_vs_pcc": {"own": 0.153, "shared5": 0.169},
    "decomposition": (
        "moving from a correlation against the panel mean to an absolute "
        "error against the median grade costs most of the agreement (tau "
        "0.232); adding the asymmetry on top costs comparatively little "
        "(0.845). The ranking moves a great deal, and MOSTLY FOR REASONS "
        "THAT ARE NOT THE ASYMMETRY"
    ),
    "why_the_control_was_needed": (
        "without it the two changes would have arrived as one "
        "indistinguishable number, and the amendment's question would "
        "have been answered with the confound inside the answer"
    ),
    "free_consistency_check": (
        "tau(A, MAE) = 0.845 exceeds tau(A, B) = 0.5917, which is what a "
        "symmetric metric sitting BETWEEN two mirror-image conventions "
        "should look like. Had MAE landed outside that interval the "
        "implementation would have been wrong somewhere and nothing else "
        "in the run would have said so"
    ),
}


#: **[MEASURED 2026-08-17, AND IT CORRECTS THE PROPOSED MECHANISM] WHY
#: CONVENTION A REWARDS HEDGING -- IT IS NOT THE CROSSOVER.**
#:
#: The proposed explanation was: a near-constant predictor makes small
#: errors, a large share of which fall below the ``|d| = 0.198`` crossover
#: where eq (16) is inverted, so A rewards hedging. **Tested on the
#: measured median-grade distribution (5, 89, 110, 30, 3 over the 237) and
#: not supported.**
#:
#: **Refutation 1 -- the crossover is not where the mass is.** For a
#: predictor constant at an integer grade, residuals against integer
#: grades are either exactly 0 or at least 1: **nothing lands in the
#: crossover region at all**. For a continuous near-constant predictor at
#: ~2.75 the residuals cluster at -0.25 and +0.75, and the measured
#: below-crossover share is **~0.19**, which is not above what
#: full-spread predictors show (~0.19-0.31 across the synthetic range).
#: The share does not separate the narrow arms from the ladder.
#:
#: **Refutation 2 -- hedging is rewarded by symmetric metrics too.** MAE
#: also ranks a constant at the mode above moderately-signalled wide
#: arms. That is the ordinary fact that error metrics reward hedging when
#: a model is weak, and it is not a property of eq (16).
#:
#: **WHAT IS ACTUALLY SPECIFIC TO EQUATION (16), MEASURED**: the branch
#: asymmetry acting on a **left-heavy label distribution**. A constant at
#: 3 overstates severity for **94** patients and understates for **33**.
#: Under convention A the 94 get the LIGHT branch (0.8, 0.87), so A
#: rewards predicting HIGH. The optimal constant predictor is therefore
#:
#:     under A     3.000   (the MODE, where 110 of 237 patients sit)
#:     under B     2.112
#:     under MAE   3.000
#:
#: **A full grade apart between the two conventions.** That is the
#: mechanism behind the disjoint top-5 sets: the conventions do not
#: disagree about how much to penalise error, they disagree about **which
#: direction to hedge in** -- and a metric whose direction is unsettled
#: has an unsettled optimal hedge.
#:
#: **What this means for the "degenerate optimum" claim**: in the strong
#: form -- a constant predictor is the optimum -- it is **NOT
#: established**, and it is shared with MAE where it does appear, so it
#: is not a defect of eq (16). The defensible finding is narrower and
#: better: **eq (16)'s direction convention moves the optimal hedge by a
#: full grade on this cohort's label distribution.** That is supervision material
#: of the same class as the crossover and the domain finding, and it does
#: not require the crossover to be involved at all.
#:
#: **Provenance**: computed from the measured label distribution and the
#: published constants; the synthetic arms are illustrative and are
#: labelled as such. The arm-level half of the claim is registered as a
#: falsifier below rather than asserted from these.
HEDGING_MECHANISM_MEASURED = {
    "measured": "2026-08-17, offline, on the measured label distribution",
    "proposed_explanation": (
        "a near-constant predictor makes small errors, a large share "
        "below the |d| = 0.198 crossover where eq (16) is inverted, so A "
        "rewards hedging"
    ),
    "not_supported": (
        "for a predictor constant at an integer grade the residuals "
        "against integer grades are exactly 0 or at least 1 -- NOTHING "
        "lands in the crossover region; for a continuous near-constant "
        "predictor the below-crossover share is ~0.19, not above the "
        "~0.19-0.31 that full-spread predictors show",
        "MAE also ranks a constant at the mode above moderately-signalled "
        "wide arms, so rewarding hedging is not a property of eq (16)",
    ),
    "what_is_specific": (
        "the branch asymmetry acting on a LEFT-HEAVY label distribution: "
        "a constant at 3 overstates severity for 94 patients and "
        "understates for 33, and under A those 94 take the LIGHT branch, "
        "so A rewards predicting HIGH"
    ),
    "optimal_constant": {"a": 3.000, "b": 2.112, "mae": 3.000},
    "the_finding": (
        "eq (16)'s DIRECTION CONVENTION MOVES THE OPTIMAL HEDGE BY A FULL "
        "GRADE on this cohort's label distribution. The conventions do "
        "not disagree about how much to penalise error, they disagree "
        "about WHICH DIRECTION TO HEDGE IN -- which is the mechanism "
        "behind the disjoint top-5 sets, and it does not require the "
        "crossover to be involved at all"
    ),
    "degenerate_optimum_not_established": (
        "in the strong form -- a constant predictor is THE optimum -- it "
        "is not established here, and where it does appear it is shared "
        "with MAE, so it is not a defect of eq (16). The narrower finding "
        "above is the defensible one"
    ),
    "supervisor_material": (
        "of the same class as IEM_CROSSOVER_INVERTS and "
        "IEM_BOUNDS_ARE_A_DOMAIN"
    ),
    "provenance": (
        "computed from the measured median-grade distribution (5, 89, "
        "110, 30, 3 over 237) and the published constants. The synthetic "
        "arms are illustrative and labelled as such; the arm-level half "
        "is a registered falsifier, not an assertion from these"
    ),
    # **[CORRECTED 2026-08-17, by the registered falsifier it carried]**
    # Both refutations above FAIL against real per-arm data, and the
    # replacement finding fails with them. HEDGING_CHECK_OBSERVED has the
    # figures; A_FAVOURS_NARROW_PREDICTORS is the mechanism as measured.
    # Kept in place rather than rewritten: this record is what the
    # falsifier was registered against, and deleting it would delete the
    # thing that was tested.
    "corrected": (
        "2026-08-17: BOTH refutations fail. Check 1 HELD (p10_cleftgnn's "
        "below-crossover share 0.3072 against a ladder median of 0.1823, "
        "ratio 1.685 against the 1.5 threshold), so the crossover IS the "
        "route. Check 2 REFUTED (A's top-5 predict 2.8154, B's 2.7785, "
        "gap +0.0369 against the +0.3 threshold), so the "
        "direction-of-hedge finding -- the optimal constants 3.000 "
        "against 2.112 -- does NOT carry to real arms. See "
        "A_FAVOURS_NARROW_PREDICTORS and FOUR_ERRORS_ONE_PLACE"
    ),
}


#: **[OBSERVED 2026-08-17, the falsifier run] ALL THREE CHECKS, AGAINST
#: THEIR REGISTERED PREDICTIONS.**
#:
#:     check                    predicted   outcome    figures
#:     1 crossover              REFUTED     HELD       0.3072 vs 0.1823,
#:                                                     ratio 1.685 (>= 1.5)
#:     2 direction of hedge     HELD        REFUTED    2.8154 vs 2.7785,
#:                                                     gap +0.0369 (< +0.3)
#:     3 narrowest spanning     none        HELD       1.0715 vs 2.2611
#:
#: **Two of two predictions inverted.** The registered consequence
#: applies without interpretation: check 1 holding means
#: ``HEDGING_MECHANISM_MEASURED`` takes a dated correction, which it has.
#:
#: **Check 3 is the sharpest result and carried no prediction**, which is
#: the argument for having registered it without one rather than leaving
#: it out. A's top-5 have a median span of **1.07 grades against the
#: ladder's 2.26** -- less than half. That is the cleanest single
#: statement of what convention A rewards, and nothing in the offline
#: work anticipated it.
HEDGING_CHECK_OBSERVED = {
    "observed": "2026-08-17, scripts/p11_prestep_check.py, band own",
    "checks": {
        "1_crossover": {
            "predicted": "REFUTED", "outcome": "HELD",
            "figures": (
                "p10_cleftgnn 0.3072 against a ladder median of 0.1823, "
                "ratio 1.685 against the 1.5 threshold"
            ),
        },
        "2_direction_of_hedge": {
            "predicted": "HELD", "outcome": "REFUTED",
            "figures": (
                "A's top-5 predict 2.8154, B's 2.7785, gap +0.0369 "
                "against the +0.3 threshold"
            ),
        },
        "3_narrowest_spanning": {
            "predicted": "none registered", "outcome": "HELD",
            "figures": (
                "A's top-5 median span 1.0715 against the ladder's 2.2611 "
                "-- less than half"
            ),
        },
    },
    "both_predictions_inverted": True,
    "consequence_applied": (
        "check 1 holding means HEDGING_MECHANISM_MEASURED takes a dated "
        "correction, which it has -- the registered consequence, applied "
        "without interpretation"
    ),
    "check_3_is_the_sharpest": (
        "and it carried no prediction, which is the argument for having "
        "registered it without one rather than leaving it out. Nothing in "
        "the offline work anticipated it"
    ),
}


#: **[FINDING 2026-08-17, MEASURED ON REAL PER-ARM DATA] CONVENTION A
#: SYSTEMATICALLY FAVOURS NARROW PREDICTORS, VIA THE CROSSOVER.**
#:
#: The three checks together give one mechanism, and it is the maintainer's
#: original account with the reasoning corrected:
#:
#: 1. **A's winners are narrow.** Median span **1.0715** against the
#:    ladder's **2.2611**.
#: 2. **Narrow arms sit in the inverted region far more often.**
#:    ``p10_cleftgnn``'s below-crossover share is **0.3072** against a
#:    ladder median of **0.1823** -- **1.685x**.
#: 3. **In that region convention A charges LESS for optimism.** At
#:    ``|d| = 0.1`` the optimistic branch costs 0.09103 against the
#:    pessimistic 0.10792 (``IEM_CROSSOVER_INVERTS``). A narrow arm
#:    predicting just BELOW the modal grade produces exactly those
#:    residuals: small, negative, and cheap.
#:
#: **Which is why A crowns an arm with PCC 0.024.** ``p10_cleftgnn``
#: predicts 2.5-3.0 against a grade distribution with 110 of 237 patients
#: at grade 3. Its errors on nearly half the cohort are small and
#: optimistic, and eq (16) as published discounts precisely those.
#:
#: **Reproduced offline AFTER the fact, and the reproduction is what
#: exposes the original error**: a predictor uniform on [2.5, 3.0] gives a
#: below-crossover share of 0.2574, one concentrated near 3.0 gives
#: 0.3460, and the measured arm sits between them at 0.3072 -- while a
#: full-range predictor gives 0.2278 and the ladder median is 0.1823. The
#: first offline attempt used a tanh-shaped arm tightly centred at 2.75,
#: which put its residuals at -0.25, just OUTSIDE the inverted region, and
#: produced 0.186. **The mechanism was always there; the synthetic missed
#: it.**
#:
#: **supervision material, and the strongest of the four.** Equation (16) as
#: published, under the reading its own equation prints, ranks a
#: near-constant predictor above every genuinely predictive arm in a
#: 68-arm ladder. That is a statement about the metric, not about our
#: arms, and it is measured.
A_FAVOURS_NARROW_PREDICTORS = {
    "finding": "2026-08-17, measured on real per-arm data",
    "mechanism": (
        "A's winners are narrow (median span 1.0715 against the ladder's "
        "2.2611); narrow arms land in the inverted region far more often "
        "(0.3072 against 0.1823, 1.685x); and in that region convention A "
        "charges LESS for optimism (at |d| = 0.1, 0.09103 against "
        "0.10792). A narrow arm predicting just below the modal grade "
        "produces exactly those residuals: small, negative, and cheap"
    ),
    "why_it_crowns_pcc_0_024": (
        "p10_cleftgnn predicts 2.5-3.0 against a distribution with 110 of "
        "237 patients at grade 3, so its errors on nearly half the cohort "
        "are small and optimistic -- and eq (16) discounts precisely those"
    ),
    "reproduced_after_the_fact": (
        "uniform on [2.5,3.0] gives 0.2574, concentrated near 3.0 gives "
        "0.3460, and the measured arm sits between at 0.3072; a "
        "full-range predictor gives 0.2278 against a ladder median of "
        "0.1823. The first offline attempt used a tanh arm tightly "
        "centred at 2.75, putting residuals at -0.25 just OUTSIDE the "
        "inverted region, and produced 0.186. The mechanism was always "
        "there; the synthetic missed it"
    ),
    "supervisor_material": (
        "the strongest of the four: eq (16) as published, under the "
        "reading its own equation prints, ranks a near-constant predictor "
        "above every genuinely predictive arm in a 68-arm ladder. A "
        "statement about the METRIC, not about our arms, and measured"
    ),
    "corrects": "HEDGING_MECHANISM_MEASURED",
}


#: **[RECORDED 2026-08-17] FOUR ERRORS, ONE PLACE.**
#:
#: Two the maintainer's, two mine, on the same question within one turn of
#: each other. Kept together because the pattern is only visible together.
#:
#: **Operator, 1 -- the crossover account: RIGHT IN SUBSTANCE, WRONG IN
#: ITS REASONING.** The proposed route was "a near-constant predictor
#: makes small errors, a large share of which fall below the crossover".
#: The conclusion is confirmed at 1.685x. The reasoning offered for it --
#: that the arm's errors are small *because* it is near-constant -- is
#: incomplete: what matters is that its predictions sit just below the
#: MODAL grade, where 46% of the cohort is. A narrow arm centred anywhere
#: else would not show it.
#:
#: **Operator, 2 -- the breach direction: WRONG, and corrected.** Reported
#: as "all overshooting the upper edge (maxima 3.85-4.74)"; all twelve are
#: ``min < 1.0``. The ``max`` field was read where ``min`` fired.
#:
#: **Mine, 1 -- the offline refutation: WRONG.** I refuted the crossover
#: account using a synthetic arm and treated its residual distribution as
#: representative of the real one. It was not: my tanh-shaped arm centred
#: at 2.75 put residuals at -0.25, just outside the inverted region, where
#: the real arm's land inside it.
#:
#: **Mine, 2 -- the direction-of-hedge prediction: WRONG.** I registered
#: it at "HELD, gap near 0.9" from the optimal-constant result (3.000
#: against 2.112). Measured gap **+0.0369**. Optimal constants are a
#: property of constants; the real arms are not constants, and I
#: extrapolated across that gap without saying I was.
#:
#: **THE LESSON, and it is one lesson covering both of mine**: *an offline
#: idealisation of a real arm's residual distribution is not a measurement
#: of it.* Both my errors are the same move -- building a plausible stand-in
#: for data I could not read, then reasoning from the stand-in as though it
#: were the data. The registered falsifier caught both, which is what it
#: was for; the failure mode is not the error but the confidence, and the
#: fix is not better synthetics but registering them as predictions.
#:
#: **What worked**: every one of the four was caught by something written
#: down before the numbers -- the falsifier's thresholds, and the check
#: that a reported description matches its own reported figures.
FOUR_ERRORS_ONE_PLACE = {
    "recorded": "2026-08-17",
    "policy_1_crossover_account": (
        "RIGHT IN SUBSTANCE, WRONG IN ITS REASONING: confirmed at 1.685x, "
        "but the route is not 'small errors because near-constant' -- it "
        "is that the predictions sit just below the MODAL grade, where "
        "46% of the cohort is. A narrow arm centred elsewhere would not "
        "show it"
    ),
    "policy_2_violation_direction": (
        "WRONG, and corrected: reported as overshooting the upper edge; "
        "all twelve are min < 1.0. The max field was read where min fired"
    ),
    "mine_1_offline_refutation": (
        "WRONG: I refuted the crossover account using a synthetic arm and "
        "treated its residual distribution as representative. My "
        "tanh-shaped arm centred at 2.75 put residuals at -0.25, just "
        "outside the inverted region, where the real arm's land inside it"
    ),
    "mine_2_direction_of_hedge": (
        "WRONG: registered at 'HELD, gap near 0.9' from the "
        "optimal-constant result (3.000 against 2.112); measured "
        "+0.0369. Optimal constants are a property of CONSTANTS, the real "
        "arms are not constants, and I extrapolated across that gap "
        "without saying I was"
    ),
    "the_lesson": (
        "AN OFFLINE IDEALISATION OF A REAL ARM'S RESIDUAL DISTRIBUTION IS "
        "NOT A MEASUREMENT OF IT. Both of mine are the same move -- "
        "building a plausible stand-in for data I could not read, then "
        "reasoning from the stand-in as though it were the data. The "
        "failure mode is not the error but the confidence, and the fix is "
        "not better synthetics but registering them as predictions"
    ),
    "what_worked": (
        "every one of the four was caught by something written down "
        "before the numbers -- the falsifier's thresholds, and the check "
        "that a reported description matches its own reported figures"
    ),
}


#: **[REGISTERED 2026-08-17, BEFORE THE TABLE IS READ] THE ARM-LEVEL
#: FALSIFIER FOR THE HEDGING MECHANISM.**
#:
#: ``per_arm.csv`` in the pre-step's run directory already carries every
#: column this needs -- ``below_crossover_share``, ``prediction_min``,
#: ``prediction_max`` -- and it has not been read. Three predictions,
#: registered with their thresholds and their expected outcomes, so the
#: measurement confirms or refutes rather than illustrates.
#: ``scripts/p11_prestep_check.py`` computes all three.
#:
#: 1. **The crossover explanation** holds if ``p10_cleftgnn``'s
#:    below-crossover share is at least **1.5x the ladder's median
#:    share**. **PREDICTED: REFUTED** -- offline it lands near parity.
#: 2. **The direction-of-hedge explanation** holds if A's top-5 arms
#:    predict HIGHER on average than B's top-5, by at least **0.3
#:    grades**. **PREDICTED: HELD**, with a gap near 0.9 if the optimal
#:    constants (3.000 against 2.112) carry over to real arms.
#: 3. **"A's winners are the narrowest-spanning"** holds if the median
#:    span (``max - min``) of A's top-5 is below the ladder's median span.
#:    **NO PREDICTION REGISTERED** -- the offline work does not bear on
#:    it, and inventing one would be decoration.
#:
#: **If 1 refutes and 2 holds, ``HEDGING_MECHANISM_MEASURED`` stands as
#: written.** If 1 holds, the crossover is involved after all and that
#: record needs a dated correction. If 2 refutes, the mechanism is
#: something neither of us has named and the finding is withdrawn to a
#: question.
HEDGING_CHECK_REGISTERED = {
    "registered": "2026-08-17, before per_arm.csv is read",
    "data": (
        "per_arm.csv already carries below_crossover_share, "
        "prediction_min and prediction_max; it has not been read"
    ),
    "checks": {
        "1_crossover": {
            "holds_if": (
                "p10_cleftgnn's below-crossover share >= 1.5x the "
                "ladder's median share"
            ),
            "predicted": "REFUTED -- offline it lands near parity",
        },
        "2_direction_of_hedge": {
            "holds_if": (
                "A's top-5 predict HIGHER on average than B's top-5 by at "
                "least 0.3 grades"
            ),
            "predicted": (
                "HELD, with a gap near 0.9 if the optimal constants "
                "(3.000 against 2.112) carry over to real arms"
            ),
        },
        "3_narrowest_spanning": {
            "holds_if": (
                "the median span (max - min) of A's top-5 is below the "
                "ladder's median span"
            ),
            "predicted": (
                "NONE registered -- the offline work does not bear on it, "
                "and inventing one would be decoration"
            ),
        },
    },
    "consequences": (
        "1 refutes and 2 holds -> HEDGING_MECHANISM_MEASURED stands",
        "1 holds -> the crossover is involved after all, and that record "
        "takes a dated correction",
        "2 refutes -> the mechanism is one neither of us has named, and "
        "the finding is withdrawn to a question",
    ),
    "script": "scripts/p11_prestep_check.py",
}


#: **[OBSERVED 2026-08-17, WITH A DISCREPANCY THAT MUST BE RESOLVED
#: BEFORE IT IS RECORDED AS A FINDING] THE DOMAIN ASSERTION FIRED ON 12
#: ARM-BANDS.**
#:
#: Exit criterion 5 worked: the breaches were reported, not clipped. Twelve
#: arm-bands, one or two patients each, concentrated in the augmentation
#: and G2 graph arms.
#:
#: **The reported description does not match the reported numbers, and
#: this is flagged rather than filed.** The breaches were described as
#: "all overshooting the upper edge (maxima 3.85-4.74)". **3.85 to 4.74
#: are inside [1, 5]**, so they cannot be what tripped the assertion. The
#: check fires on ``min < 1.0 or max > 5.0``, and since no reported
#: maximum exceeds 5, the trigger must be ``min < 1.0`` -- the ``max``
#: field was read where ``min`` is what fired.
#:
#: **Why this matters beyond bookkeeping.** A prediction below 1 is
#: better-than-Excellent: it is an *optimistic* out-of-domain value, and
#: optimism is the entire subject of this phase. Under convention A those
#: patients take the heavy branch; under B the light one. So the breach
#: direction is not incidental to the phase, and recording it backwards
#: would put a phase-relevant fact in the record wrong.
#:
#: **Needed**: the ``min`` column for the 12 flagged arm-bands from
#: ``metrics.json``'s ``domain_breaches``, which carries both fields.
#: Until then the count and the concentration stand; the DIRECTION does
#: not.
DOMAIN_BREACHES_OBSERVED = {
    "observed": "2026-08-17",
    "criterion_5_worked": "reported, not clipped -- as registered",
    "count": 12,
    "unit": "arm-bands, one or two patients each",
    "concentration": "the augmentation and G2 graph arms",
    "discrepancy": (
        "described as 'all overshooting the upper edge (maxima "
        "3.85-4.74)', but 3.85-4.74 are INSIDE [1,5] and cannot have "
        "tripped the assertion. The check fires on min < 1.0 or max > "
        "5.0, so the trigger must be min < 1.0 -- the max field was read "
        "where min is what fired"
    ),
    "why_it_matters": (
        "a prediction below 1 is better-than-Excellent: an OPTIMISTIC "
        "out-of-domain value, and optimism is this phase's entire "
        "subject. Under A those patients take the heavy branch, under B "
        "the light one, so the breach direction is not incidental and "
        "recording it backwards would put a phase-relevant fact in the "
        "record wrong"
    ),
    "needed": (
        "the min column for the 12 flagged arm-bands from metrics.json's "
        "domain_breaches, which carries both fields"
    ),
    "status": (
        "the count and the concentration STAND; the DIRECTION does not, "
        "until the min values are read"
    ),
    # **[RESOLVED 2026-08-17, the min column read]** The inference was
    # right: all twelve fired on min < 1.0, none on the upper edge.
    "resolved": {
        "read": "2026-08-17, metrics.json domain_breaches",
        "count": "12 arm-bands over 6 distinct arms",
        "all_lower_edge": "every one is min < 1.0; none exceeds 5.0",
        "minima": (0.2142, 0.9707),
        "extreme": "p7_d_agnet_imagenet_g2_native at 0.2142",
        "what_they_are": (
            "predictions BETTER THAN EXCELLENT -- optimistic "
            "out-of-domain values, which is the phase's own subject "
            "rather than an incidental range violation"
        ),
        "which_branch": (
            "under convention A they take the HEAVY branch, under B the "
            "LIGHT one -- so the breaches are scored oppositely by the "
            "two conventions the pre-step could not choose between"
        ),
        "criterion_5_complete": (
            "the assertion fired, reported rather than clipped, and the "
            "finding now has its direction"
        ),
    },
}


#: **[WALKED 2026-08-17] THE SEVEN EXIT CRITERIA, AND WHETHER THE
#: PRE-STEP CAN CLOSE.**
#:
#: 1. **Both conventions, both rankings, neither quoted alone -- MET.**
#:    Both computed over all 68 arms on both bands, both reported, and
#:    every figure in ``PRE_STEP_OBSERVED`` carries its pair.
#: 2. **The crossover carried with every number -- MET, and then some.**
#:    ``below_crossover_share`` is a column of ``per_arm.csv``, and the
#:    crossover was not merely carried but TESTED as the proposed
#:    mechanism and found not to be it (``HEDGING_MECHANISM_MEASURED``).
#: 3. **Skew per arm and the power check applied -- MET.** Max relative
#:    gap 0.1187 against the registered 0.01 floor, applied by the rule
#:    rather than by preference; reading 2 fired because the rule said so.
#: 4. **Descriptive only -- MET.** No claim, no ledger row, and the task
#:    imports no ledger.
#: 5. **Domain asserted, breaches reported -- MET; the FINDING is
#:    incomplete.** The assertion fired on 12 arm-bands and reported
#:    rather than clipped, which is what the criterion asks. But the
#:    breach DIRECTION does not survive its own numbers
#:    (``DOMAIN_BREACHES_OBSERVED``) and is pending one column of a file
#:    already on disk.
#: 6. **The loss build still blocked and recorded as blocked -- MET**, and
#:    now with a measured reason rather than a caution.
#: 7. **Suite green -- MET.**
#:
#: **CAN IT CLOSE? Yes -- but not this turn.** Two READS are outstanding,
#: both of artifacts already written, neither needing a run:
#:
#: * the ``min`` column for the 12 flagged arm-bands, which settles
#:   criterion 5's finding;
#: * ``scripts/p11_prestep_check.py`` over ``per_arm.csv``, which settles
#:   the three registered hedging checks and therefore whether
#:   ``HEDGING_MECHANISM_MEASURED`` stands, corrects, or withdraws.
#:
#: Closing before either would file a finding whose direction is unread
#: and a mechanism whose falsifier is unrun, with both files sitting in
#: the run directory. That is the gap this project has now paid for three
#: times.
PRESTEP_EXIT_WALK = {
    "walked": "2026-08-17",
    "criteria": {
        "1_both_conventions": (
            "MET -- both computed over all 68 arms on both bands, both "
            "reported, every figure carrying its pair"
        ),
        "2_crossover_carried": (
            "MET, and then some: below_crossover_share is a column of "
            "per_arm.csv, and the crossover was TESTED as the proposed "
            "mechanism and found not to be it"
        ),
        "3_skew_and_power": (
            "MET -- max relative gap 0.1187 against the registered 0.01 "
            "floor, applied by the rule; reading 2 fired because the rule "
            "said so"
        ),
        "4_descriptive": (
            "MET -- no claim, no ledger row, and the task imports no "
            "ledger"
        ),
        "5_domain": (
            "MET as a criterion -- the assertion fired on 12 arm-bands "
            "and reported rather than clipped -- but the FINDING is "
            "incomplete: the breach direction does not survive its own "
            "numbers and is pending one column"
        ),
        "6_build_blocked": (
            "MET, and now with a measured reason rather than a caution"
        ),
        "7_suite": "MET",
    },
    "can_it_close": "YES, but not this turn",
    "outstanding": (
        "the min column for the 12 flagged arm-bands, which settles "
        "criterion 5's finding",
        "scripts/p11_prestep_check.py over per_arm.csv, which settles the "
        "three registered hedging checks and therefore whether "
        "HEDGING_MECHANISM_MEASURED stands, corrects or withdraws",
    ),
    "both_are_reads": (
        "of artifacts already written; neither needs a run, a GPU or a "
        "declaration"
    ),
    "why_not_close_first": (
        "it would file a finding whose direction is unread and a "
        "mechanism whose falsifier is unrun, with both files sitting in "
        "the run directory -- the gap this project has paid for three "
        "times"
    ),
    # [RESOLVED 2026-08-17, same day] Both reads happened. Both inverted
    # a prediction, which is the argument for having waited: closing
    # first would have filed HEDGING_MECHANISM_MEASURED uncorrected and
    # the breach direction backwards. See PRESTEP_CLOSED.
    "resolved": (
        "2026-08-17: both reads done, both inverted a prediction -- the "
        "min column confirmed all twelve breaches are min < 1.0, and the "
        "falsifier corrected HEDGING_MECHANISM_MEASURED. Closing first "
        "would have filed one uncorrected and the other backwards. "
        "PRESTEP_CLOSED"
    ),
}


#: **[PROPOSED 2026-08-17] WHAT IS WORTH DOING IN PHASE 11 WHILE THE
#: DIRECTION QUESTION WAITS -- and what is not.**
#:
#: **Worth doing, and it is the highest-value item available: consolidate
#: the ask for supervision.** Seven questions have accumulated across three
#: phases, and they are currently scattered across three phase files:
#:
#:     1. the IEM DIRECTION (phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION)
#:     2. is CASE the same metric as eq (16)? (CASE_IEM_IDENTITY_OPEN)
#:     3. eq (16) inverts below |d| = 0.198 (IEM_CROSSOVER_INVERTS)
#:     4. [0, 4] is a domain, not a range (IEM_BOUNDS_ARE_A_DOMAIN)
#:     5. 27 stated against 36 executed
#:        (phase10.REGION_COUNT_BELIEF_VS_EXECUTION)
#:     6. slide 9's cleft numbers and the crop sets
#:        (phase10.SLIDE_9_CLEFT_NUMBERS_REQUESTED)
#:     7. unanimity on the anchor grades (carried from PHASE_9_CLOSING)
#:
#: **Question 1 now has a number attached** -- tau 0.59, disjoint top-5
#: sets, and the two conventions naming opposite winners -- which makes it
#: a concrete finding to answer rather than a definitional quibble to
#: postpone. That is the strongest position this project has been in to
#: get it answered, and it argues for asking now rather than at the next
#: meeting.
#:
#: **NOT worth doing, and each for a stated reason:**
#:
#: * **building the loss "provisionally under convention A"** -- this is
#:   the blocked thing, and the pre-step just produced the strongest
#:   possible evidence for why: the conventions invert which end of the
#:   ladder is good.
#: * **FGCM and NDCG@K** -- deferred at registration, and the pre-step
#:   has just shown that one metric with a known defect is already
#:   demanding to read. Adding two more before the first is settled would
#:   multiply the interpretation, not the evidence.
#: * **a third, symmetric variant of eq (16)** (both branches at weight 1)
#:   to separate the weights from the exponents. Tempting, and declined:
#:   the registration admits no third form without a third documented
#:   reading, and MAE already plays that role in the control column.
#:
#: **And the honest structural answer**: the amendment's sequence puts
#: **Phase 12** next, and Phase 12 is not blocked. Phase 11 should close
#: its pre-step, stay open with the build blocked, and the project should
#: move on -- rather than manufacture Phase 11 work to fill the wait. A
#: blocked phase that keeps finding things to do is how a block stops
#: being visible.
PHASE_11_WHILE_BLOCKED = {
    "proposed": "2026-08-17",
    "worth_doing": {
        "consolidate_the_ask_for_supervisor": (
            "seven questions across three phases, currently scattered "
            "across three phase files: the IEM direction; CASE-vs-eq(16) "
            "identity; the crossover inversion; [0,4] domain-not-range; "
            "27 stated against 36 executed; slide 9's cleft numbers and "
            "crop sets; unanimity on the anchor grades"
        ),
        "why_now": (
            "question 1 now has a NUMBER attached -- tau 0.59, disjoint "
            "top-5 sets, opposite winners -- which makes it a concrete "
            "finding to answer rather than a definitional quibble to "
            "postpone. The strongest position this project has been in to "
            "get it answered"
        ),
    },
    "not_worth_doing": {
        "provisional_loss_under_a": (
            "this is the blocked thing, and the pre-step just produced "
            "the strongest evidence for why: the conventions invert which "
            "end of the ladder is good"
        ),
        "fgcm_and_ndcg": (
            "deferred at registration, and one metric with a known defect "
            "is already demanding to read -- two more before the first is "
            "settled multiplies the interpretation, not the evidence"
        ),
        "a_third_symmetric_variant": (
            "tempting and declined: the registration admits no third form "
            "without a third documented reading, and MAE already plays "
            "that role in the control column"
        ),
    },
    "the_structural_answer": (
        "the amendment's sequence puts PHASE 12 next and Phase 12 is not "
        "blocked. Phase 11 should close its pre-step, stay open with the "
        "build blocked, and the project should move on rather than "
        "manufacture Phase 11 work to fill the wait. A blocked phase that "
        "keeps finding things to do is how a block stops being visible"
    ),
}


#: **[WRITTEN 2026-08-17] THE CONSOLIDATED ASK FOR SUPERVISION -- seven
#: questions, three phases, one place.**
#:
#: They were scattered across ``phase9``, ``phase10`` and ``phase11``,
#: each recorded where it was found. Scattered is where a question goes
#: unasked, so they are gathered here in the order they are worth asking:
#: the one that blocks work first, then the ones that change what the
#: paper claims, then the ones that would give us numbers we do not have.
#:
#: **1. THE IEM DIRECTION -- BLOCKING, and it now carries a number.**
#: Does the heavier 1.2 weight fall on predictions that FLATTER (a lower
#: grade than truth, ``y_hat - G < 0``, which is what eq (16) prints), or
#: the other way? One sentence settles it; a worked example settles it
#: just as well.
#: **Why it is not a quibble**: measured on 68 arms, the two readings give
#: Kendall tau **0.5917**, **disjoint top-5 sets**, and **opposite
#: winners** -- convention A crowns an arm with PCC **0.024** via a
#: below-crossover share **1.7x** the ladder's, while B puts the project's
#: best arm second. The loss build cannot start until this is answered.
#: (``phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION``, ``PRE_STEP_OBSERVED``,
#: ``A_FAVOURS_NARROW_PREDICTORS``)
#:
#: **2. Is CASE the same metric as equation (16)?** The amendment says
#: CASE, the manuscript prints IEM, the deck names neither. We have
#: treated them as one by assumption and tagged it. (``CASE_IEM_IDENTITY_
#: OPEN``)
#:
#: **3. Equation (16) inverts below |d| = 0.19753.** The branches cross
#: there, and beneath it the branch called more dangerous is penalised
#: less. Intended, an artifact of fitting the exponents, or something to
#: fix on revision? (``IEM_CROSSOVER_INVERTS``)
#:
#: **4. The paper's [0, 4] is a domain, not a range.** At |d| = 4 the
#: branches give 5.6688 and 2.6723. If [0, 4] was meant as the range, the
#: constants and the range disagree. (``IEM_BOUNDS_ARE_A_DOMAIN``)
#:
#: **5. 27 regions stated, 36 executed.** The manuscript, the deck and the
#: notebook's own argument all say 27; the notebook's generator produces
#: 36 and prints a warning at every construction. Every result in the
#: paper was produced with 36. Correct the count, or re-run under an
#: enumeration that yields 27?
#: (``phase10.REGION_COUNT_BELIEF_VS_EXECUTION``)
#:
#: **6. Slide 9's cleft numbers, and the crop sets.** The three methods
#: are compared on the CLEFT dataset; the tables are images. The numbers,
#: the metric, the test set and its n, any intervals -- and the crop
#: coordinates, which are not derivable from "bottom half" and "bottom
#: third". (``phase10.SLIDE_9_CLEFT_NUMBERS_REQUESTED``)
#:
#: **7. Unanimity on the anchor grades.** Carried from Phase 9's closing;
#: until it is confirmed, the provenance caveat travels with every
#: classifier quotation. (``phase9.PHASE_9_CLOSING``)
#:
#: **Only question 1 blocks anything.** The rest change what the write-up
#: can say, not what it can do.
SUPERVISOR_CONSOLIDATED_ASK = {
    "written": "2026-08-17",
    "why_consolidated": (
        "they were scattered across three phase files, each recorded "
        "where it was found -- and scattered is where a question goes "
        "unasked"
    ),
    "order": (
        "the one that blocks work first, then the ones that change what "
        "the paper claims, then the ones that would give us numbers we do "
        "not have"
    ),
    # **[2026-08-17] Question 1 is ANSWERED and has left the ask.** It is
    # recorded below rather than deleted: the numbers gathered to make it
    # askable are what got it answered, and a reader should be able to see
    # that a question left this list by being answered rather than by
    # being dropped. The remaining six keep their original numbers, so a
    # reference to "question 5" still means question 5.
    "answered_and_removed": {
        "n": 1,
        "was": (
            "does the heavier 1.2 weight fall on predictions that FLATTER "
            "(y_hat - G < 0, which is what eq (16) prints), or the other "
            "way?"
        ),
        "answered": "2026-08-17 -- convention A, supervision via the maintainer",
        "why_it_was_askable": (
            "it carried its number: Kendall tau 0.5917 on 68 arms, "
            "DISJOINT top-5 sets, OPPOSITE winners, A crowning an arm at "
            "PCC 0.024 via a below-crossover share 1.7x the ladder's"
        ),
        "records": "IEM_DIRECTION_ANSWERED",
    },
    "questions": (
        {
            "n": 2, "blocking": False,
            "ask": "is CASE the same metric as equation (16)?",
            "records": "CASE_IEM_IDENTITY_OPEN",
        },
        {
            "n": 3, "blocking": False,
            "ask": (
                "equation (16) inverts below |d| = 0.19753 -- intended, "
                "an artifact of fitting the exponents, or to fix on "
                "revision?"
            ),
            "records": "IEM_CROSSOVER_INVERTS",
        },
        {
            "n": 4, "blocking": False,
            "ask": (
                "the paper's [0,4] is a domain, not a range: at |d| = 4 "
                "the branches give 5.6688 and 2.6723"
            ),
            "records": "IEM_BOUNDS_ARE_A_DOMAIN",
        },
        {
            "n": 5, "blocking": False,
            "ask": (
                "27 regions stated in three places, 36 executed with a "
                "printed warning -- correct the count, or re-run under an "
                "enumeration that yields 27?"
            ),
            "records": "phase10.REGION_COUNT_BELIEF_VS_EXECUTION",
        },
        {
            "n": 6, "blocking": False,
            "ask": (
                "slide 9's cleft numbers -- the figures, the metric, the "
                "test set and its n, any intervals -- and the crop "
                "coordinates, which are not derivable from 'bottom half' "
                "and 'bottom third'"
            ),
            "records": "phase10.SLIDE_9_CLEFT_NUMBERS_REQUESTED",
        },
        {
            "n": 7, "blocking": False,
            "ask": "unanimity on the anchor grades",
            "records": "phase9.PHASE_9_CLOSING",
        },
        # [ADDED 2026-08-23] The rating-protocol question -- the
        # HIGHEST-VALUE item on this list despite carrying the last
        # number (numbers are stable identifiers, not priority).
        {
            "n": 8, "blocking": False,
            "priority": (
                "HIGHEST-VALUE as of 2026-08-23 -- ask FIRST despite the "
                "number: one sentence decides which experiment Phase 12 is"
            ),
            "ask": (
                "When the five raters scored these 251 patients, were "
                "they shown the frontal photograph only, or the frontal "
                "and submental together, with one score recorded under "
                "the frontal's ID?"
            ),
            "why": (
                "the sheet is keyed to frontal IDs and no basal ID "
                "receives a score; the instrument literature describes "
                "frontal-based rating; whether the basal was nonetheless "
                "SHOWN is exactly what the sheet cannot distinguish. The "
                "answer separates 'restore evidence the label already "
                "contains' from 'measure what a view the raters never "
                "saw adds'"
            ),
            "records": "ladder.BASAL_RATIONALE_UNSUPPORTED",
        },
        # [APPENDED 2026-08-29, Phase 17's restate] The landmark
        # boundary. Kept on the numbered list per its own rule: numbers
        # are stable identifiers, not priority.
        {
            "n": 9, "blocking": False,
            "ask": (
                "I used SCUT's shipped landmark files for training-side "
                "synthesis, no detector anywhere -- tell me if that "
                "crosses the line you drew"
            ),
            "why": (
                "the constraint's recorded basis is a rejection of "
                "DETECTION, not of landmark data: 'no landmark detector "
                "is available and the supervision material rejected the one from its own "
                "group' (geometry/mirror.py's own docstring). The "
                "project already reads SHIPPED landmarks in Phase 5 "
                "(SCUT's 86-point .pts files) and Phase 15 (MEBeauty's "
                "landmarks.csv); Phase 17's arms warp SCUT faces by "
                "those shipped points on the training side only -- no "
                "detector runs, and nothing touches a patient image"
            ),
            "records": "phase17.PHASE_17_RULINGS",
        },
    ),
    # [2026-08-17] Was "question 1. The rest change what the write-up can
    # SAY, not what it can DO." Question 1 is answered, so NOTHING on this
    # list blocks anything now.
    "nothing_blocks_now": (
        "2026-08-17: the only blocking question was 1 and it is answered. "
        "The remaining six change what the write-up can SAY, not what it "
        "can DO -- none of them holds up a build"
    ),
    # **[2026-08-23] Question 8 joined the tuple above, and it is the
    # HIGHEST-VALUE question on the list despite carrying the last
    # number** -- numbers are stable identifiers here, not priority, so
    # questions 2-7 keep theirs and 8's priority note travels with it.
    "highest_value_as_of_2026_08_23": (
        "question 8 -- ask it first despite its number: one sentence "
        "decides which experiment Phase 12 is"
    ),
}


#: **[CLOSED 2026-08-17] PHASE 11'S PRE-STEP CLOSES. THE BUILD STAYS
#: BLOCKED.**
#:
#: **The seven exit criteria, all MET, criterion 5's finding now
#: complete:**
#:
#: 1. **Both conventions, both rankings, neither quoted alone** -- 68
#:    arms, both bands, every figure carrying its pair.
#: 2. **The crossover carried with every number** -- and it turned out to
#:    be the mechanism, at 1.685x the ladder's share.
#: 3. **Skew and the power check applied** -- max relative gap 0.1187
#:    against the 0.01 floor; reading 2 fired by the rule.
#: 4. **Descriptive only** -- no claim, no ledger row, no ledger import.
#: 5. **Domain asserted, breaches reported** -- 12 arm-bands over 6 arms,
#:    all ``min < 1.0``, minima 0.2142-0.9707, reported not clipped. The
#:    finding has its direction.
#: 6. **The loss build still blocked**, now with a measured reason.
#: 7. **Suite green.**
#:
#: **WHAT THE PRE-STEP ANSWERED.** The amendment asked whether every
#: existing arm's ranking moves under the asymmetric error. It does --
#: tau 0.153/0.169 against PCC -- **and the MAE control shows most of that
#: is the target and error-family change, not the asymmetry** (tau(MAE,
#: PCC) 0.232, tau(IEM_A, MAE) 0.845). The asymmetry's own contribution is
#: real but smaller than the headline number suggests, and without the
#: control the two would have arrived as one figure.
#:
#: **WHAT IT ANSWERED THAT IT WAS NOT ASKED.** Equation (16) as published,
#: under the reading its own equation prints, **ranks a near-constant
#: predictor first in a 68-arm ladder** -- PCC 0.024, span 1.07 against
#: the ladder's 2.26 -- because narrow predictions just below the modal
#: grade land in the region where the metric charges less for optimism.
#: That is a finding about the metric and it is the phase's most
#: significant result.
#:
#: **NOTHING IS CLAIMED.** No ledger row. A ranking has no interval, and
#: PLAN 4.3 does not admit one.
#:
#: **CARRIED FORWARD**: the loss build, unregistered and blocked on
#: question 1 of ``SUPERVISOR_CONSOLIDATED_ASK``; FGCM and NDCG@K, available and
#: deferred; and the four metric findings, which are supervision material whether
#: or not the build ever runs.
#:
#: **NEXT**: Phase 12, which the amendment sequences here and which is not
#: blocked. ``PHASE_11_WHILE_BLOCKED`` says why the phase should not
#: manufacture work to fill the wait.
PRESTEP_CLOSED = {
    "closed": "2026-08-17",
    "criteria": {
        "1_both_conventions": "MET -- 68 arms, both bands, every figure paired",
        "2_crossover_carried": (
            "MET -- and it turned out to be the mechanism, at 1.685x the "
            "ladder's share"
        ),
        "3_skew_and_power": (
            "MET -- max relative gap 0.1187 against the 0.01 floor; "
            "reading 2 fired by the rule"
        ),
        "4_descriptive": "MET -- no claim, no ledger row, no ledger import",
        "5_domain": (
            "MET, finding COMPLETE -- 12 arm-bands over 6 arms, all min < "
            "1.0, minima 0.2142-0.9707, reported not clipped"
        ),
        "6_build_blocked": "MET -- now with a measured reason",
        "7_suite": "MET",
    },
    "what_it_answered": (
        "the amendment's question: every existing arm's ranking DOES move "
        "under the asymmetric error (tau 0.153/0.169 against PCC) -- and "
        "the MAE control shows most of that is the target and "
        "error-family change, not the asymmetry (tau(MAE,PCC) 0.232, "
        "tau(IEM_A,MAE) 0.845). Without the control the two would have "
        "arrived as one figure"
    ),
    "what_it_answered_unasked": (
        "equation (16) as published, under the reading its own equation "
        "prints, ranks a NEAR-CONSTANT PREDICTOR FIRST in a 68-arm ladder "
        "-- PCC 0.024, span 1.07 against the ladder's 2.26 -- because "
        "narrow predictions just below the modal grade land where the "
        "metric charges less for optimism. A finding about the METRIC, "
        "and the phase's most significant result"
    ),
    "nothing_is_claimed": (
        "no ledger row. A ranking has no interval and PLAN 4.3 does not "
        "admit one"
    ),
    "carried_forward": (
        "the loss build -- unregistered and BLOCKED on question 1 of "
        "SUPERVISOR_CONSOLIDATED_ASK",
        "FGCM (eqs. 14-15) and NDCG@K (eqs. 17-18) -- available, deferred",
        "the four metric findings -- supervision material whether or not the "
        "build ever runs",
    ),
    "next": (
        "Phase 12, which the amendment sequences here and which is not "
        "blocked; PHASE_11_WHILE_BLOCKED says why this phase should not "
        "manufacture work to fill the wait"
    ),
    # [UNBLOCKED 2026-08-17, same day] The wait ended sooner than the
    # closing assumed: supervision answered, convention A. The pre-step's closing
    # stands unchanged -- nothing in it depended on the direction being
    # open -- and the build is now scoped rather than blocked.
    "unblocked": (
        "2026-08-17: the direction is ANSWERED (convention A, "
        "IEM_DIRECTION_ANSWERED), so the loss build is no longer blocked. "
        "Nothing in this closing depended on the direction being open. "
        "The build's scope and its unresolved silences are "
        "PHASE_11_BUILD_SCOPE; the consequence of THIS direction in "
        "particular is registered in advance at DEGENERACY_PULL_REGISTERED"
    ),
}


#: **[RESTATED 2026-08-17, silences LISTED not filled -- the t-SNE
#: discipline] PHASE 11'S BUILD SCOPE, now that the direction is
#: answered.**
#:
#: **WHAT THE AMENDMENT COMMITS THE LOSS ARMS TO** (section 6, and the
#: request table in section 1, quoted):
#:
#: * the request is **"Penalise optimism (CASE)"**, and what it changes is
#:   **"the loss function"**;
#: * **"Clinically well-motivated"**;
#: * **"it is a loss-function change, so it needs its own arms rather than
#:   a re-scoring of existing ones"** -- the pre-step was the re-scoring,
#:   and it is done; this is the part it was explicitly not a substitute
#:   for;
#: * what already exists for it: **"the training path; predictions on
#:   disk"**.
#:
#: The amendment says nothing else about the loss arms. It does not name a
#: backbone, a cell, a comparison, a seed band or a criterion.
#:
#: **WHAT EXISTS TO BUILD FROM**, all verified and none of it new:
#:
#: * the frozen harness -- ``run_cv``, the inner-val split, early
#:   stopping, and gates 3, 5 and 6;
#: * the Phase 7 arm path: ``EmbeddingHeadBackbone`` over declared
#:   embeddings, which is how every ladder arm was fitted;
#: * ``phase11.iem`` -- the metric itself, already implemented, already
#:   exercised over 490 vectors, and now with its direction settled;
#: * the per-epoch curve machinery Phase 10 built
#:   (``phase10.CURVES_AND_STAGES_WRITTEN``,
#:   ``phase10.ALL_EPOCHS_RECORDED``), which is where
#:   ``DEGENERACY_PULL_REGISTERED``'s diagnostic would ride at no extra
#:   forward;
#: * **the pre-step's own table** -- every existing arm's IEM under
#:   convention A, so a loss arm has a measured comparison population
#:   rather than one number to beat;
#: * ``run.median_by_patient``, if the loss trains against the grade.
#:
#: **REGISTERED SILENCES -- the build stops on these until they are
#: resolved. Listed, not filled:**
PHASE_11_BUILD_SCOPE = {
    "restated": "2026-08-17, from PLAN_AMENDMENT_2026-08-13 sections 1 and 6",
    "commits": {
        "what_changes": "the loss function",
        "its_own_arms": (
            "'it is a loss-function change, so it needs its own arms "
            "rather than a re-scoring of existing ones' -- the pre-step "
            "WAS the re-scoring, and this is the part it was explicitly "
            "not a substitute for"
        ),
        "motivation": "'clinically well-motivated'",
        "what_exists": "'the training path; predictions on disk'",
        "and_nothing_else": (
            "the amendment names no backbone, cell, comparison, seed band "
            "or criterion for the loss arms"
        ),
    },
    "exists_to_build_from": (
        "the frozen harness -- run_cv, the inner-val split, early "
        "stopping, gates 3, 5 and 6",
        "the Phase 7 arm path: EmbeddingHeadBackbone over declared "
        "embeddings, which is how every ladder arm was fitted",
        "phase11.iem -- implemented, exercised over 490 vectors, "
        "direction now settled",
        "the per-epoch curve machinery from Phase 10, where "
        "DEGENERACY_PULL_REGISTERED's diagnostic rides at no extra forward",
        "the pre-step's table -- every existing arm's IEM under "
        "convention A, so a loss arm has a measured comparison POPULATION "
        "rather than one number to beat",
        "run.median_by_patient, if the loss trains against the grade",
    ),
    "silences": (
        "EXIT CRITERIA for the build -- none exist. The seven that closed "
        "were the PRE-STEP's; a build's are unwritten, and Phase 10's "
        "silence list said this is written before the work starts",
        "WHICH TARGET the loss optimises: the median GRADE, which is what "
        "eq (16) is defined against and what the pre-step scored, or the "
        "panel MEAN, which is what every arm is evaluated on. If they "
        "differ the arm optimises something it is not scored on, and that "
        "is a choice with a consequence rather than a detail",
        "WHICH CELL(S) get an IEM-trained twin -- one arm, a family, or "
        "the 0.2520 arm alone; and at which geometry, init and seed band",
        "WHETHER A MATCHED MSE CONTROL ARM RUNS beside it. Without one "
        "the degeneracy diagnostic cannot separate 'the loss pulled it' "
        "from 'this arm hedges anyway' (DEGENERACY_PULL_REGISTERED)",
        "EQ (16) VERBATIM AS A LOSS, or a smoothed variant. The "
        "pessimistic branch's gradient is 0.696|d|^-0.13, which is "
        "nonzero as d -> 0 -- the loss never stops pushing -- while the "
        "optimistic branch's vanishes. Verbatim is the fidelity choice "
        "and it is also the choice that keeps the measured degeneracy in "
        "the training signal",
        "WHAT THE ARM IS REPORTED ON: PCC, the standing criterion it "
        "would likely lose on, or IEM, the thing it optimises. Reporting "
        "only IEM is marking its own homework, and reporting only PCC "
        "hides what the loss was for",
        "WHETHER ANYTHING IS CLAIMABLE -- PLAN 4.3's two conditions "
        "against what, given the comparison population is now an IEM "
        "distribution over 68 arms rather than a single contrast",
        "THE CROSSOVER DEFECT: left in as published, or corrected. "
        "Leaving it in is faithful and keeps the pathology; correcting it "
        "measures a metric the group has not published. Neither is "
        "obviously right and neither is chosen here",
        "FGCM and NDCG@K -- still available, still deferred",
    ),
    "not_a_silence": (
        "the DIRECTION -- answered, convention A (IEM_DIRECTION_ANSWERED)",
        "the metric's implementation -- phase11.iem, already exercised",
        "the degeneracy reading -- committed in advance "
        "(DEGENERACY_PULL_REGISTERED)",
    ),
}


#: The numerical floor inside the power, and NOT a smoothing of the
#: metric. ``|d|**0.87`` has an infinite derivative at exactly zero, so
#: autograd returns ``nan`` there; clamping the magnitude before the power
#: makes the gradient finite. Measured effect at 1e-8: the loss VALUE
#: changes by under 1e-7 (0.8 * 1e-8**0.87 = 8.7e-8) and the pessimistic
#: gradient is capped at 7.63. Recorded in ``IEM_LOSS_VERBATIM`` as the
#: one departure from verbatim, with its size, because "verbatim" and
#: "nan" are not alternatives.
IEM_GRADIENT_FLOOR = 1e-8


def iem_loss(prediction, target, *, convention: str = "a_manuscript_literal",
             floor: float = IEM_GRADIENT_FLOOR):
    """Equation (16) as a differentiable loss, verbatim (``IEM_LOSS_VERBATIM``).

    Same branch assignment as ``iem``: the heavier ``1.2 |d|^1.12`` on
    ``prediction - target < 0``, which under convention A is the
    flattering direction (``IEM_DIRECTION_ANSWERED``). Returns the mean
    over the batch.

    **Not smoothed.** ``floor`` clamps the magnitude inside the power only,
    which leaves the value unchanged to seven decimal places and exists
    because the alternative at exactly-zero error is ``nan``, not a
    different metric.
    """
    import torch

    if convention not in CONVENTIONS:
        raise Phase11Error(
            f"convention {convention!r} is not one of {CONVENTIONS}"
        )
    residual = prediction - target
    magnitude = residual.abs().clamp_min(floor)
    heavy = 1.2 * magnitude ** 1.12
    light = 0.8 * magnitude ** 0.87
    below = residual < 0
    if convention == "b_swapped":
        below = ~below
    return torch.where(below, heavy, light).mean()


#: **[DEFECT 2026-08-17, FOUND AND FIXED] THE ARTIFACT CARRIED BOTH
#: METRICS AND THE LOG PRINTED ONE.**
#:
#: ``p11-paired``'s ``metrics.json`` held both contrasts under
#: ``by_metric`` exactly as designed, and the run log printed only the
#: PCC line. A reader of the log alone got half the result -- and the
#: half they got was the one the phase's own exit criterion says must
#: never travel alone.
#:
#: **It is the same family as a computed-then-discarded curve**, which
#: this project has now paid for four times: the probe that rebuilt the
#: pipeline rather than reading it, the 25 curves the arm threw away, the
#: U-shape reconstructed from a run that had already measured it, and
#: now this. The shape is constant: **the computation is right, the
#: channel a human actually reads is not, and nothing fails.** The
#: artifact was correct throughout, which is exactly why it took a reader
#: comparing the log against the file to notice.
#:
#: **Why the criterion did not catch it either.** Exit criterion 3 says
#: both metrics are reported and the pair travels together. It was
#: satisfied -- in the artifact. The criterion did not distinguish
#: *reported* from *reported in every channel*, and that distinction is
#: what the defect lives in.
#:
#: **The fix**: the verdict line loops over ``by_metric`` and prints one
#: line per metric with its own delta, exclusion count, both conditions
#: and its direction. **A third metric would print itself without another
#: fix**, which is the property the first version lacked.
THE_LOG_WAS_PARTIAL = {
    "defect": "2026-08-17, found and fixed",
    "what": (
        "p11-paired's metrics.json held both contrasts under by_metric as "
        "designed, and the log printed only the PCC line -- a reader of "
        "the log alone got half the result, and the half they got was the "
        "one the phase's own exit criterion says must never travel alone"
    ),
    "the_family": (
        "computed-then-discarded, the fourth instance: the probe that "
        "rebuilt the pipeline rather than reading it, the 25 curves the "
        "arm threw away, the U-shape reconstructed from a run that had "
        "already measured it, and this. The shape is constant -- THE "
        "COMPUTATION IS RIGHT, THE CHANNEL A HUMAN READS IS NOT, AND "
        "NOTHING FAILS"
    ),
    "why_the_criterion_missed_it": (
        "exit criterion 3 says both metrics are reported and the pair "
        "travels together, and it WAS satisfied -- in the artifact. The "
        "criterion did not distinguish REPORTED from REPORTED IN EVERY "
        "CHANNEL, and the defect lives in that distinction"
    ),
    "fix": (
        "the verdict line loops over by_metric and prints one line per "
        "metric with its own delta, exclusion count, both conditions and "
        "its direction. A THIRD metric would print itself without another "
        "fix, which is the property the first version lacked"
    ),
}


#: **[OBSERVED 2026-08-17, ``p11_paired__75e9dd8a__p11-paired``] BOTH
#: CONTRASTS WITHDRAWN ON CONDITION 1. THE REGISTERED PREDICTION HELD ON
#: BOTH.**
#:
#:     metric  d        excl 0  cond 1  cond 2  margin  verdict
#:     PCC     -0.0697  3 of 5  False   True    2.57x   WITHDRAWN
#:     IEM     -0.0305  2 of 5  False   True    2.68x   WITHDRAWN
#:
#: The IEM contrast carries ``lower_is_better: true`` and a recomputed
#: baseline sd of **0.00744**. Both figures come from ``metrics.json``'s
#: ``by_metric`` rather than from the log, which is the defect at
#: ``THE_LOG_WAS_PARTIAL``.
#:
#: **The prediction held on both, and as at Phase 10 the arithmetic was
#: never what was at risk.** The deltas and thresholds were derived from
#: ``LOSS_ARMS_OBSERVED``, so predicting them was arithmetic. What was at
#: risk was the VERDICT -- and both margins sit in the region where the
#: margin does not order the outcome, just above the 2.55x line below
#: which nothing in this project has ever passed. The registration said
#: condition 1 would decide, and on both metrics it did.
#:
#: **THE DESCRIPTIVE FINDING STANDS UNCHANGED** (``LOSS_ARMS_OBSERVED``):
#: the IEM loss improves IEM on every seed (0.5825 against 0.6130) and
#: damages PCC on every seed (0.1855 against 0.2552, a fall of 0.0697),
#: ten of ten seed-wise comparisons in the expected direction with
#: disjoint ranges on both metrics; and the matched control at 0.2552
#: against the 0.2520 arm shows the target change costs nothing, so the
#: fall belongs to the LOSS.
#:
#: **AND NEITHER CONTRAST IS CLAIMABLE.** A fifth independent arrival at
#: ``COHORT_CANNOT_RESOLVE``: ten of ten seed-wise comparisons agreeing,
#: with disjoint ranges, and the per-patient intervals still will not
#: exclude zero on more than three of five seeds. **The descriptive
#: picture is unambiguous and the criterion refuses it anyway** -- which
#: is the criterion working, not failing.
PAIRED_LOSS_OBSERVED = {
    "observed": "2026-08-17, p11_paired__75e9dd8a__p11-paired",
    "by_metric": {
        "pcc": {
            "d": -0.0697, "excludes_zero": "3 of 5",
            "condition_1": False, "condition_2": True, "margin": 2.57,
            "verdict": "WITHDRAWN",
        },
        "iem": {
            "d": -0.0305, "excludes_zero": "2 of 5",
            "condition_1": False, "condition_2": True, "margin": 2.68,
            "verdict": "WITHDRAWN", "lower_is_better": True,
            "baseline_sd_recomputed": 0.00744,
        },
    },
    "figures_are_from_the_artifact": (
        "metrics.json's by_metric, not the log -- which printed only the "
        "PCC contrast (THE_LOG_WAS_PARTIAL)"
    ),
    "prediction_held": (
        "on both metrics, and as at Phase 10 the arithmetic was never "
        "what was at risk: the deltas and thresholds were DERIVED from "
        "LOSS_ARMS_OBSERVED. What was at risk was the VERDICT, and both "
        "margins sit just above the 2.55x line where the margin does not "
        "order the outcome. The registration said condition 1 would "
        "decide, and on both metrics it did"
    ),
    "descriptive_finding_stands": (
        "the IEM loss improves IEM on every seed (0.5825 against 0.6130) "
        "and damages PCC on every seed (0.1855 against 0.2552, -0.0697), "
        "ten of ten seed-wise comparisons in the expected direction with "
        "disjoint ranges on both metrics; the matched control at 0.2552 "
        "against the 0.2520 arm shows the target change costs nothing, so "
        "the fall belongs to the LOSS"
    ),
    "fifth_arrival": (
        "COHORT_CANNOT_RESOLVE, a fifth independent time: ten of ten "
        "seed-wise comparisons agreeing with disjoint ranges, and the "
        "per-patient intervals still will not exclude zero on more than "
        "three of five seeds. The descriptive picture is unambiguous and "
        "the criterion refuses it anyway -- which is the criterion "
        "WORKING, not failing"
    ),
}


#: **[CLOSED 2026-08-17] PHASE 11 CLOSES.**
#:
#: The phase asked what the "penalise optimism" request costs. It has an
#: answer, and the answer is not claimable -- which is itself the phase's
#: fifth arrival at the same wall.
#:
#: **THE SIX BUILD EXIT CRITERIA, ALL MET:**
#:
#: 1. **The loss arm built and run with the per-epoch diagnostic and its
#:    joint trigger** -- 25 folds, signature computed per fold from
#:    epoch 0 (gate 3's own prediction) to the selected epoch.
#: 2. **A matched MSE control under identical conditions** -- same cell,
#:    seeds, folds and target, differing in the loss and nothing else,
#:    from one config renderer so it could not drift.
#: 3. **Both arms reported on PCC and IEM, the pair travelling together**
#:    -- met in the artifacts throughout, and met in the LOG only after
#:    ``THE_LOG_WAS_PARTIAL`` was found and fixed.
#: 4. **The degeneracy reading applied by rule** -- 0 of 25 in both arms,
#:    condition 1 unmet, condition 2 moot, third registered outcome
#:    applied verbatim.
#: 5. **The crossover and gradient defects carried with every number** --
#:    in the arm configs, in both runs' metrics, in the paired config and
#:    in both ledger rows.
#: 6. **Suite green.**
#:
#: **WHAT THE PHASE ESTABLISHED.**
#:
#: * **The pre-step** (68 arms, 490 vectors, DESCRIPTIVE): the ranking
#:   moves under the asymmetric error, tau 0.153/0.169 against PCC -- but
#:   the MAE control showed most of that is the target and error-family
#:   change, not the asymmetry (tau(MAE,PCC) 0.232 against tau(IEM_A,MAE)
#:   0.845). The amendment's question, answered with its confound
#:   separated.
#: * **Equation (16) has two measured defects**: its value inverts below
#:   |d| = 0.19753 and its gradient below |d| = 0.0719, so the branch it
#:   calls more dangerous is both scored and trained more leniently near
#:   the truth. Under convention A it ranks a near-constant predictor
#:   first in a 68-arm ladder.
#: * **The loss does what it was asked, and the bill is correlation**:
#:   IEM 0.5825 against 0.6130 and PCC 0.1855 against 0.2552, ten of ten
#:   seed-wise comparisons in the expected direction with disjoint ranges,
#:   and the matched control showing the target change costs nothing.
#: * **And none of it is claimable.** Both contrasts withdrawn on
#:   condition 1 -- **a fifth arrival at ``COHORT_CANNOT_RESOLVE``.**
#:
#: **THE PREDICTED DEGENERACY DID NOT MATERIALISE, and that is a result
#: too.** The pre-step measured a pull toward hedging in the RANKING;
#: registered in advance, it was then looked for in TRAINING and was not
#: there -- 0 of 25 in both arms. The reading committed before the run is
#: what makes that a finding rather than a shrug.
#:
#: **NOTHING NEW IS CLAIMED HERE.** Two ledger rows, both WITHDRAWN, both
#: carrying their conditions' figures and the defects. The descriptive
#: findings were recorded when they were measured.
PHASE_11_CLOSING = {
    "closed": "2026-08-17",
    "question": "what does the 'penalise optimism' request cost?",
    "answer": (
        "measured, and NOT CLAIMABLE -- which is itself the phase's fifth "
        "arrival at the same wall"
    ),
    "exit_criteria": {
        "1_arm_and_diagnostic": (
            "MET -- 25 folds, the signature computed per fold from epoch "
            "0 (gate 3's own prediction) to the selected epoch"
        ),
        "2_matched_control": (
            "MET -- same cell, seeds, folds and target, differing in the "
            "loss and nothing else, from ONE config renderer so it could "
            "not drift"
        ),
        "3_both_metrics": (
            "MET -- in the artifacts throughout, and in the LOG only "
            "after THE_LOG_WAS_PARTIAL was found and fixed"
        ),
        "4_reading_by_rule": (
            "MET -- 0 of 25 in both arms, condition 1 unmet, condition 2 "
            "moot, third registered outcome applied verbatim"
        ),
        "5_defects_carried": (
            "MET -- in the arm configs, both runs' metrics, the paired "
            "config and both ledger rows"
        ),
        "6_suite": "MET",
    },
    "established": {
        "pre_step": (
            "68 arms, 490 vectors, DESCRIPTIVE: the ranking moves under "
            "the asymmetric error (tau 0.153/0.169 against PCC), and the "
            "MAE control showed most of that is the target and "
            "error-family change rather than the asymmetry -- tau(MAE, "
            "PCC) 0.232 against tau(IEM_A, MAE) 0.845"
        ),
        "the_metric_has_two_defects": (
            "eq (16)'s value inverts below |d| = 0.19753 and its gradient "
            "below |d| = 0.0719, so the branch it calls more dangerous is "
            "both SCORED and TRAINED more leniently near the truth; under "
            "convention A it ranks a near-constant predictor first in a "
            "68-arm ladder"
        ),
        "the_loss_works_and_the_bill_is_correlation": (
            "IEM 0.5825 against 0.6130 and PCC 0.1855 against 0.2552, ten "
            "of ten seed-wise comparisons in the expected direction with "
            "disjoint ranges, and the matched control showing the target "
            "change costs nothing"
        ),
        "and_none_of_it_is_claimable": (
            "both contrasts WITHDRAWN on condition 1 -- a FIFTH arrival "
            "at COHORT_CANNOT_RESOLVE"
        ),
    },
    "the_predicted_degeneracy_did_not_materialise": (
        "the pre-step measured a pull toward hedging in the RANKING; "
        "registered in advance, it was then looked for in TRAINING and "
        "was not there -- 0 of 25 in both arms. The reading committed "
        "before the run is what makes that a finding rather than a shrug"
    ),
    "nothing_new_is_claimed": (
        "two ledger rows, both WITHDRAWN, both carrying their conditions' "
        "figures and the defects; the descriptive findings were recorded "
        "when they were measured"
    ),
    "ledger": (
        "entry 27 p11-iem-loss-costs-pcc-withdrawn, entry 28 "
        "p11-iem-loss-improves-iem-withdrawn -- and they travel together "
        "by registration"
    ),
    # ---- open, non-blocking, carried forward BY NAME ------------------
    "carried_forward_open": (
        "the FOUR metric findings -- IEM_CROSSOVER_INVERTS (the value "
        "inversion), IEM_BOUNDS_ARE_A_DOMAIN ([0,4] is a domain not a "
        "range), A_FAVOURS_NARROW_PREDICTORS (convention A crowns a "
        "near-constant predictor), and the GRADIENT inversion at "
        "DEGENERACY_PULL_REGISTERED. All four are about the PUBLISHED "
        "metric and stand whether or not another arm ever runs",
        "FGCM (eqs. 14-15) and NDCG@K (eqs. 17-18) -- available, still "
        "DEFERRED; one metric with known defects was enough to interpret "
        "in one pass",
        "the CORRECTED-CROSSOVER variant -- a FUTURE ARM requiring its "
        "own registration, never a mid-phase repair "
        "(PHASE_11_BUILD_REGISTERED)",
        "the remaining SIX supervision questions -- SUPERVISOR_CONSOLIDATED_ASK, none of "
        "which blocks anything now that question 1 is answered",
    ),
    "next": (
        "Phase 12, the decoder/reconstruction build, which the amendment "
        "sequences after this one"
    ),
    # [RENUMBERED 2026-08-23] The line above is what the closing said when
    # it closed, and it stays visible. the maintainer re-sequenced afterwards:
    # Phase 12 is now the VIEW ABLATION, the decoder moves to 13, the
    # write-up to 14 (PHASE_SEQUENCE_RENUMBERED).
    "renumbered": (
        "2026-08-23: Phase 12 is now the view ablation; the decoder moves "
        "to Phase 13 and the write-up to Phase 14 -- "
        "PHASE_SEQUENCE_RENUMBERED. The line above is preserved as what "
        "the closing said when it closed"
    ),
}


#: **[DECIDED 2026-08-23] THE SEQUENCE AFTER PHASE 11 IS
#: RENUMBERED -- a dated amendment to the amendment.**
#:
#: ``PLAN_AMENDMENT_2026-08-13`` sequences Phase 12 as the
#: decoder/reconstruction and Phase 13 as the write-up. That changes:
#:
#:     was                          becomes
#:     Phase 12  decoder            Phase 12  VIEW ABLATION
#:     Phase 13  write-up           Phase 13  decoder/reconstruction
#:                                  Phase 14  write-up
#:
#: **The reason, as decided**: the view ablation tests a limitation the
#: record carried as CORE, while the decoder answers a visualisation
#: request -- the ablation is the more consequential of the two, and its
#: inputs already exist.
#:
#: **Nothing already written is renumbered.** Every existing record that
#: says "Phase 12" meaning the decoder -- the amendment itself,
#: ``PHASE_11_WHILE_BLOCKED``, ``PHASE_11_CLOSING`` -- stays as written,
#: with dated pointers beside the ones a reader will hit. The original
#: sequence remains visible; this record is the pointer's source.
#:
#: **And the honest interaction with the correction that landed the same
#: day**: the reason above cites the CORE tag that
#: ``ladder.BASAL_RATIONALE_UNSUPPORTED`` downgrades. The renumbering
#: was the maintainer's decision on the record as it stood, and it stands --
#: but the correction changes what the ablation CAN TEST: with the
#: premise unsupported, Phase 12 is either "restore evidence the label
#: already contains" or "measure what a view the raters never saw adds",
#: and the supervision answer to the rating-protocol question decides which. The
#: renumbering fixes the ablation's PLACE; the correction reopens its
#: QUESTION.
PHASE_SEQUENCE_RENUMBERED = {
    "decided": "2026-08-23",
    # [2026-08-23, later the same day] A SECOND amendment sequences
    # everything after 13: 14 = LDL, 15 = the second beauty dataset,
    # 16 = TSTR, 17 = write-up (phase12.PHASE_SEQUENCE_RENUMBERED_2).
    # This record's "14: write-up" is preserved as what it was.
    "second_amendment": "phase12.PHASE_SEQUENCE_RENUMBERED_2",
    # [2026-08-31] A FIFTH amendment APPENDS Phase 20 (the permutation
    # control); it renumbers nothing and 19 remains the write-up. Named
    # PHASE_SEQUENCE_EXTENDED_5, not RENUMBERED_5, because nothing moved.
    "fifth_amendment": "phase20.PHASE_SEQUENCE_EXTENDED_5",
    # [2026-09-01] A SEVENTH amendment schedules 22 (ranking and
    # pairwise losses), 23 (the statistical instruments), 24 (all five
    # raters) and 25 (foundation-model features) as
    # SCHEDULED-NOT-REGISTERED. It renumbers nothing, and leaves the
    # write-up's number OPEN rather than resolving it
    # (phase21.WRITE_UP_NUMBER_OPEN).
    "seventh_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_7",
    # [2026-08-31] A SIXTH amendment APPENDS Phase 21 (the ensemble
    # probe and error-consistency diagnosis). It renumbers nothing --
    # the first amendment written UNDER the write-up-runs-last rule
    # rather than establishing it.
    "sixth_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_6",
    "amends": "PLAN_AMENDMENT_2026-08-13's sequence, after Phase 11",
    "was": {"12": "decoder/reconstruction", "13": "write-up"},
    "becomes": {
        "12": "the VIEW ABLATION (frontal / basal / both)",
        "13": "decoder/reconstruction",
        "14": "write-up",
    },
    "reason": (
        "the view ablation tests a limitation the record carried as CORE, "
        "while the decoder answers a visualisation request -- the "
        "ablation is the more consequential of the two, and its inputs "
        "already exist"
    ),
    "nothing_silently_renumbered": (
        "every existing record that says 'Phase 12' meaning the decoder "
        "stays as written, with dated pointers beside the ones a reader "
        "will hit; the original sequence remains visible and this record "
        "is the pointer's source"
    ),
    "interaction_with_the_correction": (
        "the reason cites the CORE tag that "
        "ladder.BASAL_RATIONALE_UNSUPPORTED downgrades the same day. The "
        "renumbering stands -- it fixes the ablation's PLACE -- and the "
        "correction reopens its QUESTION: restore evidence the label "
        "already contains, or measure what a view the raters never saw "
        "adds. the supervision material's rating-protocol answer decides which"
    ),
    "eighth_amendment": (
        "phase25.PHASE_SEQUENCE_EXTENDED_8, 2026-09-05. Calibration "
        "ablation as 26, anchor set as a training set as 27, fine "
        "tuning as 28 with small backbones as an arm inside it. "
        "Renumbered nothing"
    ),
}


def kendall_tau_b(rank_a, rank_b) -> float:
    """Kendall's tau-b between two rankings, ties handled.

    Written here rather than imported because ``eval/metrics.py`` is
    FROZEN and carries no rank statistic -- the same reason
    ``phase10.top1_macro_prf`` lives in its own phase file.

    **[CORRECTED 2026-09-01 -- DOUBLY-TIED PAIRS WERE COUNTED INTO BOTH
    DENOMINATOR FACTORS.** Found while building Phase 21, whose
    cross-reference is exactly this shape; see
    ``phase21.TAU_B_DEFECT_CORRECTED`` for the verification and the
    downstream impact.]

    tau-b's denominator is ``sqrt((n0 - n1) * (n0 - n2))`` where ``n0``
    is all pairs, ``n1`` the pairs tied in ``a`` and ``n2`` the pairs
    tied in ``b``. A pair tied in BOTH is counted in ``n1`` and in
    ``n2``, so it is removed from BOTH factors -- it contributes to
    neither. The previous form added such pairs INTO both factors
    instead, inflating the denominator and attenuating tau toward zero.

    **The decisive symptom**: a vector compared against ITSELF is perfect
    agreement and must give exactly 1.0. With ties it gave 0.6667 for
    ``[1, 1, 2]``. Verified against ``scipy.stats.kendalltau``, which
    agrees with the corrected form on every case tried and with the
    previous form only when no pair is tied in both.

    **The error was one-directional**: it could only shrink ``|tau|``,
    never grow it, so every figure produced by the previous form is a
    LOWER BOUND on the true value.
    """
    import numpy as np

    a = np.asarray(rank_a, dtype=float)
    b = np.asarray(rank_b, dtype=float)
    if a.shape != b.shape:
        raise Phase11Error(f"{a.shape} against {b.shape}: not the same arms")
    n = len(a)
    concordant = discordant = tied_a_only = tied_b_only = tied_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                # Tied in BOTH: excluded from both factors, not added to
                # them. This is the line the correction turns on.
                tied_both += 1
                continue
            if da == 0:
                tied_a_only += 1
                continue
            if db == 0:
                tied_b_only += 1
                continue
            if da * db > 0:
                concordant += 1
            else:
                discordant += 1
    # n0 - n1 = pairs NOT tied in a = C + D + (tied in b only)
    # n0 - n2 = pairs NOT tied in b = C + D + (tied in a only)
    untied_in_a = concordant + discordant + tied_b_only
    untied_in_b = concordant + discordant + tied_a_only
    denominator = (untied_in_a * untied_in_b) ** 0.5
    if denominator == 0:
        return float("nan")
    return (concordant - discordant) / denominator


def ranks(values) -> list:
    """Ascending competition ranks (1 = smallest), ties sharing a rank.

    IEM and MAE are ERRORS -- smaller is better -- so ascending is the
    good-first order. PCC is inverted by the caller rather than here, so
    the direction of every ranking is visible at its call site.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    out = [0] * len(values)
    position = 0
    while position < len(order):
        run = [order[position]]
        while (
            position + len(run) < len(order)
            and values[order[position + len(run)]] == values[order[position]]
        ):
            run.append(order[position + len(run)])
        for index in run:
            out[index] = position + 1
        position += len(run)
    return out


#: **[REGISTERED 2026-08-17, BEFORE THE ARM EXISTS] THE LOSS BUILD.**
#:
#: Nine decisions, each closing a silence listed at
#: ``PHASE_11_BUILD_SCOPE`` and each taken before any code ran.
#:
#: **The cell**: ONE -- the 0.2520 arm (frozen ViT-B/16, ImageNet, G1,
#: whole image), the project's best and the anchor of every comparison. A
#: single twin answers the amendment's question; a ladder of twins answers
#: nothing extra until the first shows the loss does anything.
#:
#: **The target**: the **median grade**. Equation (16) is defined against
#: a consensus grade G, and training against a different quantity than the
#: loss's own definition would confound the arm. **Evaluation stays PCC
#: against the panel mean**, as everywhere else in the project -- so the
#: arm optimises one thing and is judged on another, deliberately and on
#: the record.
#:
#: **The control**: a **matched MSE arm** -- same cell, same seeds, same
#: folds, same target, differing in the loss and in nothing else.
#: **Registered as part of the arm, not an optional extra**: without it
#: "was pulled there" and "hedges under any loss" are indistinguishable,
#: and the degeneracy reading cannot be applied at all.
#:
#: **The loss**: eq (16) **verbatim, not smoothed**
#: (``IEM_LOSS_VERBATIM``).
#:
#: **The reporting**: **both PCC and IEM, always, travelling together.**
#: IEM alone is marking its own homework; PCC alone hides whether the loss
#: did what it was asked.
#:
#: **Claimability**: the **IEM-vs-MSE contrast may go to paired BCa** under
#: PLAN 4.3's two conditions -- it is a like-for-like two-arm comparison on
#: shared folds. The IEM arm's PCC **against the ladder is DESCRIPTIVE
#: only**: a different loss and a different target make it not a
#: like-for-like ladder entry.
#:
#: **The crossover defect stays in**, per the verbatim decision. The
#: corrected variant is registered as a **future arm requiring its own
#: registration** -- never as a mid-phase repair, which is the shape this
#: project has refused at ``phase10.MONITOR_ARM_REGISTRABILITY`` and at
#: ``phase10.GATE_3_ROUTES_PROPOSED``.
#:
#: **FGCM and NDCG@K**: still deferred.
PHASE_11_BUILD_REGISTERED = {
    "registered": "2026-08-17, before the arm exists",
    "cell": (
        "ONE -- the 0.2520 arm (frozen ViT-B/16, ImageNet, G1, whole "
        "image), the project's best and the anchor of every comparison. A "
        "single twin answers the amendment's question; a ladder of twins "
        "answers nothing extra until the first shows the loss does "
        "anything"
    ),
    "target": (
        "the MEDIAN GRADE -- eq (16) is defined against a consensus grade "
        "G, and training against a different quantity than the loss's own "
        "definition would confound the arm"
    ),
    "evaluation": (
        "PCC against the PANEL MEAN, as everywhere else -- so the arm "
        "optimises one thing and is judged on another, deliberately and "
        "on the record"
    ),
    "control": (
        "a MATCHED MSE arm: same cell, same seeds, same folds, same "
        "target, differing in the loss and nothing else. Registered as "
        "PART OF THE ARM, not an optional extra -- without it 'was pulled "
        "there' and 'hedges under any loss' are indistinguishable and the "
        "degeneracy reading cannot be applied at all"
    ),
    "loss_form": "eq (16) verbatim, not smoothed (IEM_LOSS_VERBATIM)",
    "reporting": (
        "BOTH PCC and IEM, always, travelling together. IEM alone is "
        "marking its own homework; PCC alone hides whether the loss did "
        "what it was asked"
    ),
    "claimability": {
        "iem_vs_mse": (
            "MAY go to paired BCa under PLAN 4.3's two conditions -- a "
            "like-for-like two-arm comparison on shared folds"
        ),
        "iem_vs_the_ladder": (
            "DESCRIPTIVE only -- a different loss and a different target "
            "make it not a like-for-like ladder entry"
        ),
    },
    "crossover_stays_in": (
        "per the verbatim decision. The corrected variant is a FUTURE ARM "
        "requiring its own registration, never a mid-phase repair -- the "
        "shape refused at phase10.MONITOR_ARM_REGISTRABILITY and "
        "phase10.GATE_3_ROUTES_PROPOSED"
    ),
    "deferred": "FGCM (eqs. 14-15) and NDCG@K (eqs. 17-18)",
}


#: **[REGISTERED 2026-08-17] EQ (16) VERBATIM AS A LOSS -- AND THE ONE
#: DEPARTURE, WITH ITS SIZE.**
#:
#: The loss is the published equation with the published constants and the
#: published branch assignment. It is **not smoothed**, because smoothing
#: would measure a metric the group has not published, and because the
#: defects are part of what this phase reports:
#:
#: * the **value** inversion below ``|d| = 0.19753``
#:   (``IEM_CROSSOVER_INVERTS``);
#: * the **gradient** inversion below ``|d| = 0.0719``, where the
#:   pessimistic branch pushes harder than the optimistic one
#:   (``DEGENERACY_PULL_REGISTERED``);
#: * the pessimistic branch's gradient ``0.696 |d|^-0.13``, which is
#:   **nonzero as d -> 0** -- the loss never stops pushing.
#:
#: **THE ONE DEPARTURE, and it is numerical rather than a choice of
#: metric.** ``|d|**0.87`` has an infinite derivative at exactly zero, so
#: autograd returns ``nan`` there and the run dies. The magnitude is
#: clamped to ``1e-8`` INSIDE THE POWER ONLY. Measured effect: the loss
#: value changes by under **1e-7** (``0.8 * 1e-8**0.87 = 8.7e-8``) and the
#: pessimistic gradient is capped at **7.63** instead of diverging.
#:
#: **Stated as a departure rather than defended as neutral**, because it
#: is one: the function is unchanged to seven decimal places, but it is
#: not literally the printed function everywhere. "Verbatim" and "nan"
#: were not alternatives, and pretending the clamp is free is how a
#: silent change enters a faithful build.
IEM_LOSS_VERBATIM = {
    "registered": "2026-08-17",
    "form": (
        "the published equation, constants and branch assignment; NOT "
        "smoothed"
    ),
    "why_not_smoothed": (
        "smoothing would measure a metric the group has not published, "
        "and the defects are part of what this phase reports"
    ),
    "defects_carried": (
        "the VALUE inversion below |d| = 0.19753",
        "the GRADIENT inversion below |d| = 0.0719",
        "the pessimistic gradient 0.696|d|^-0.13, nonzero as d -> 0 -- the "
        "loss never stops pushing",
    ),
    "the_one_departure": {
        "what": (
            "|d|**0.87 has an infinite derivative at exactly zero, so "
            "autograd returns nan and the run dies. The magnitude is "
            "clamped to 1e-8 INSIDE THE POWER ONLY"
        ),
        "measured_effect": (
            "the loss value changes by under 1e-7 (0.8 * 1e-8**0.87 = "
            "8.7e-8) and the pessimistic gradient is capped at 7.63 "
            "instead of diverging"
        ),
        "stated_not_defended": (
            "it IS a departure: the function is unchanged to seven "
            "decimal places but is not literally the printed function "
            "everywhere. 'Verbatim' and 'nan' were not alternatives, and "
            "pretending the clamp is free is how a silent change enters a "
            "faithful build"
        ),
    },
}


#: **[REGISTERED 2026-08-17, THRESHOLDS BEFORE THE RUN] HOW THE
#: DEGENERACY READING IS APPLIED -- BY RULE, NOT BY EYE.**
#:
#: ``DEGENERACY_PULL_REGISTERED`` committed the reading: an arm that
#: improves on IEM while its span narrows and its PCC falls is the
#: degenerate optimum realised, not a result. This fixes what counts as
#: that, before any number exists.
#:
#: **The per-fold signature** (``degeneracy_signature``) FIRES when all
#: three move the degenerate way between **epoch 0** -- captured by gate
#: 3's own prediction -- and the fold's **selected epoch**: span DOWN,
#: below-crossover share UP, inner-val PCC DOWN. Any one alone is ordinary
#: training and does not fire.
#:
#: **The two conditions, both required, both fixed now.** Across the 25
#: folds (five seeds x five folds):
#:
#: 1. **the IEM arm fires on at least 13 of 25** -- a simple majority; and
#: 2. **the IEM arm's count exceeds the MSE control's by at least 5
#:    folds.**
#:
#: Condition 2 is the one that does the work. Condition 1 alone would
#: confirm the reading for an arm that hedges under any loss; only the gap
#: against a matched control separates *the loss pulled it there* from
#: *this cell does that anyway*.
#:
#: **If both hold** -> the degenerate optimum is realised. The arm's IEM
#: improvement is reported as **the metric's behaviour, not the model's**,
#: and is not quotable as an improvement, ledgerable, or comparable
#: against the ladder.
#:
#: **If 1 holds and 2 does not** -> the cell hedges under any loss. That is
#: a finding about the CELL, not about eq (16), and the IEM arm's result
#: is read on its ordinary merits with the caveat attached.
#:
#: **If neither holds** -> the pull the pre-step measured in the RANKING
#: did not materialise in TRAINING. That is a real and reportable
#: negative, and it would mean the loss can be evaluated on its merits --
#: which is the outcome that would make the phase's remaining question
#: interesting rather than settled.
#:
#: **THE STATED LIMITATION.** The signature is a direction-of-travel test
#: over two points, epoch 0 and the selected epoch. It cannot distinguish a
#: monotone slide from a trajectory that wandered and happened to end
#: lower, and the per-epoch table is written precisely so a reader can
#: look. It also cannot see a degeneracy that was present at
#: initialisation, because epoch 0 is its own baseline.
DEGENERACY_APPLICATION_RULE = {
    "registered": "2026-08-17, thresholds before the run",
    "per_fold_signature": (
        "fires when ALL THREE move the degenerate way between epoch 0 "
        "(gate 3's own prediction) and the fold's selected epoch: span "
        "DOWN, below-crossover share UP, inner-val PCC DOWN. Any one "
        "alone is ordinary training"
    ),
    "conditions": {
        "1_majority": "the IEM arm fires on at least 13 of 25 folds",
        "2_gap_over_control": (
            "the IEM arm's count exceeds the MSE control's by at least 5 "
            "folds"
        ),
    },
    "condition_2_does_the_work": (
        "condition 1 alone would confirm the reading for an arm that "
        "hedges under any loss; only the gap against a MATCHED control "
        "separates 'the loss pulled it there' from 'this cell does that "
        "anyway'"
    ),
    "outcomes": {
        "both_hold": (
            "the degenerate optimum is realised: the IEM improvement is "
            "the METRIC's behaviour, not the model's, and is not quotable "
            "as an improvement, ledgerable, or comparable against the "
            "ladder"
        ),
        "one_only": (
            "the cell hedges under any loss -- a finding about the CELL, "
            "not about eq (16); the IEM arm is read on its ordinary "
            "merits with the caveat attached"
        ),
        "neither": (
            "the pull the pre-step measured in the RANKING did not "
            "materialise in TRAINING -- a real and reportable negative, "
            "and the outcome that would let the loss be evaluated on its "
            "merits"
        ),
    },
    "stated_limitation": (
        "a direction-of-travel test over TWO POINTS, epoch 0 and the "
        "selected epoch. It cannot distinguish a monotone slide from a "
        "trajectory that wandered and ended lower -- the per-epoch table "
        "is written so a reader can look -- and it cannot see a "
        "degeneracy present at initialisation, because epoch 0 is its own "
        "baseline"
    ),
}


#: **[REGISTERED 2026-08-17, BEFORE WORK STARTS] THE BUILD'S EXIT
#: CRITERIA.** The seven that closed were the PRE-STEP's; these are the
#: build's, written before the task existed.
PHASE_11_BUILD_EXIT_CRITERIA = {
    "registered": "2026-08-17, before work starts",
    "criteria": (
        "1. the loss arm built and run with the per-epoch diagnostic and "
        "its joint trigger",
        "2. a matched MSE control run under identical conditions -- same "
        "cell, seeds, folds and target, differing only in the loss",
        "3. both arms reported on PCC and on IEM, the pair travelling "
        "together",
        "4. the degeneracy reading applied BY RULE "
        "(DEGENERACY_APPLICATION_RULE), not by eye",
        "5. the crossover and gradient defects carried with every number",
        "6. suite green",
    ),
    "expected_count": 6,
    "not_criteria": (
        "an IEM improvement is not a result until criterion 4 has been "
        "applied to it",
        "the paired BCa against the control is admissible but not "
        "required to close (PHASE_11_BUILD_REGISTERED claimability)",
    ),
}


#: **[OBSERVED 2026-08-17, both arms at SHA 27a18e04] THE REGISTERED
#: NEGATIVE OUTCOME FIRED: THE PULL DID NOT MATERIALISE, AND THE LOSS IS
#: JUDGED ON ITS MERITS.**
#:
#: **The degeneracy signature fired on 0 of 25 folds in BOTH arms.**
#: Condition 1 (>= 13 of 25) is not met; condition 2 is moot at 0 against
#: 0. ``DEGENERACY_APPLICATION_RULE``'s third outcome applies verbatim:
#: the pull the pre-step measured in the RANKING did not materialise in
#: TRAINING, which is a real and reportable negative -- and it is the
#: outcome that lets the loss be evaluated on its merits rather than
#: dismissed as the metric's artifact.
#:
#: **THE MERITS, MEASURED. The IEM loss does exactly what it was asked,
#: and the bill is correlation.**
#:
#:     metric   IEM arm            MSE control        delta
#:     PCC      0.1855 (sd 0.0281) 0.2552 (sd 0.0129) -0.0697
#:     IEM      0.5825 (sd 0.0106) 0.6130 (sd 0.0075) -0.0305
#:
#: **Ten of ten seed-wise comparisons run in the expected direction**, and
#: the two arms' ranges do not overlap on either metric:
#:
#:     PCC   IEM arm [0.1511, 0.2118]   control [0.2382, 0.2711]
#:     IEM   IEM arm [0.5706, 0.5937]   control [0.6024, 0.6205]
#:
#: **Penalising optimism is not free, and its cost is correlation.** The
#: loss improves the thing it optimises on every seed and damages the
#: project's standing criterion on every seed, by about 0.07 -- roughly a
#: quarter of the best arm's whole correlation.
#:
#: **THE INCIDENTAL, and it is what makes the attribution clean**: the MSE
#: control scores **0.2552** against the 0.2520 arm's **0.2520** -- a
#: difference of +0.0032, inside the arm's own seed sd of 0.0148 --
#: **despite training on the median grade rather than the panel mean.**
#: So the target change alone costs nothing measurable, and the PCC fall
#: is attributable to the LOSS rather than to the target. Without the
#: matched control this would have been two changes and one number.
#:
#: **Both defects continue to travel**: the value inversion below
#: |d| = 0.19753 and the gradient inversion below |d| = 0.0719
#: (``IEM_CROSSOVER_INVERTS``, ``DEGENERACY_PULL_REGISTERED``). The loss
#: was run verbatim with them in, which is what the phase reports.
LOSS_ARMS_OBSERVED = {
    "observed": "2026-08-17, both arms at SHA 27a18e04",
    "per_seed": {
        "iem_arm_pcc": (0.1594, 0.2066, 0.1511, 0.2118, 0.1984),
        "mse_control_pcc": (0.2588, 0.2613, 0.2382, 0.2711, 0.2464),
        "iem_arm_iem": (0.5861, 0.5937, 0.5719, 0.5706, 0.5902),
        "mse_control_iem": (0.6024, 0.6205, 0.6136, 0.6091, 0.6192),
    },
    "degeneracy": {
        "fired": "0 of 25 folds in BOTH arms",
        "condition_1": "NOT MET (needs >= 13 of 25)",
        "condition_2": "MOOT at 0 against 0",
        "registered_outcome_applied": (
            "DEGENERACY_APPLICATION_RULE's third outcome, verbatim: the "
            "pull measured in the RANKING did not materialise in "
            "TRAINING -- a real and reportable negative, and the outcome "
            "that lets the loss be judged on its merits"
        ),
    },
    "merits": {
        "pcc": {"arm": 0.1855, "control": 0.2552, "delta": -0.0697},
        "iem": {"arm": 0.5825, "control": 0.6130, "delta": -0.0305},
        "ten_of_ten": (
            "every seed-wise comparison runs in the expected direction on "
            "both metrics, and the arms' ranges do not overlap on either"
        ),
        "ranges": {
            "pcc": ((0.1511, 0.2118), (0.2382, 0.2711)),
            "iem": ((0.5706, 0.5937), (0.6024, 0.6205)),
        },
        "the_sentence": (
            "penalising optimism is NOT FREE, and its cost is "
            "CORRELATION: the loss improves what it optimises on every "
            "seed and damages the project's standing criterion on every "
            "seed, by about 0.07 -- roughly a quarter of the best arm's "
            "whole correlation"
        ),
    },
    "the_incidental": (
        "the MSE control scores 0.2552 against the 0.2520 arm's 0.2520 -- "
        "+0.0032, inside that arm's own seed sd of 0.0148 -- DESPITE "
        "training on the median grade rather than the panel mean. The "
        "target change alone costs nothing measurable, so the PCC fall is "
        "attributable to the LOSS and not to the target. Without the "
        "matched control this would have been two changes and one number"
    ),
    "defects_still_travel": (
        "the value inversion below |d| = 0.19753 and the gradient "
        "inversion below |d| = 0.0719; the loss was run verbatim with "
        "them in, which is what the phase reports"
    ),
}


#: The two arms' stems, named once so the pair enumeration, the generator
#: and the records cannot drift.
PAIRED_ARM_STEM = "p11_iem_arm"
PAIRED_CONTROL_STEM = "p11_mse_control"

#: What Phase 11's paired scope covers, and what it refuses.
PAIRED_CLAIM_COVERAGE = {
    "covers": (
        "one pair: the IEM arm against its MATCHED MSE control -- same "
        "cell, same seeds, same folds, same target, differing in the loss "
        "and nothing else, on shared patients"
    ),
    "on_two_metrics": (
        "PCC against the panel mean, and IEM against the median grade the "
        "arms trained on. Both, always, because IEM alone is marking its "
        "own homework and PCC alone hides whether the loss did what it "
        "was asked (PHASE_11_BUILD_REGISTERED)"
    ),
    "excluded": {
        "the_ladder": (
            "the IEM arm's PCC against any ladder arm is DESCRIPTIVE "
            "only: a different loss and a different training target make "
            "it not a like-for-like entry, and pairing it would dress a "
            "non-comparison in an interval"
        ),
        "the_prestep": (
            "a ranking has no interval; the pre-step was DESCRIPTIVE by "
            "registration and stays so"
        ),
    },
}


def paired_claim_pairs(scope: str = "p11_loss") -> list[dict]:
    """Phase 11's paired scope: the loss contrast, shaped like the
    ladder's and Road B's so the ONE paired implementation serves it too.

    Recorded figures are DERIVED from ``LOSS_ARMS_OBSERVED`` rather than
    typed, so a correction to the measurement lands here automatically.
    ``b`` is the IEM arm, so a NEGATIVE delta means the IEM arm scores
    lower -- which is worse on PCC and better on IEM. **The direction is
    recorded per metric rather than assumed**, because the two run
    opposite ways and a single sign convention would mislead on one of
    them.
    """
    if scope != "p11_loss":
        raise Phase11Error(f"unknown Phase 11 paired-claim scope {scope!r}")

    import numpy as np

    from . import ladder
    from .train.phase3 import combined_claimable_delta

    observed = LOSS_ARMS_OBSERVED["per_seed"]
    seeds = list(ladder.SEED_POOL[:5])
    recorded = {}
    for metric in ("pcc", "iem"):
        arm = np.array(observed[f"iem_arm_{metric}"], dtype=float)
        control = np.array(observed[f"mse_control_{metric}"], dtype=float)
        delta = float(arm.mean() - control.mean())
        threshold = combined_claimable_delta(
            float(arm.std(ddof=1)), len(seeds),
            float(control.std(ddof=1)), len(seeds),
        )["arm_means_95"]
        recorded[metric] = {
            "delta_of_means": round(delta, 4),
            "threshold": round(threshold, 4),
            "margin": round(abs(delta) / threshold, 2),
            "lower_is_better": metric == "iem",
            "source": "LOSS_ARMS_OBSERVED; DESCRIPTIVE until the run",
        }
    return [{
        "key": "p11__iem_vs_mse_control",
        "question": "loss",
        "varies": "loss",
        "a": PAIRED_CONTROL_STEM,
        "b": PAIRED_ARM_STEM,
        "seeds": seeds,
        # **The metric is part of the ENUMERATION, not a branch in the
        # task.** Every other scope names none and gets the single PCC it
        # always ran; this one names two, and the paired path loops rather
        # than branching -- which is what keeps ONE call site computing a
        # paired comparison.
        "metrics": (
            {"name": "pcc", "truth": "panel_mean"},
            {"name": "iem", "truth": "median_grade", "lower_is_better": True},
        ),
        "recorded": recorded,
    }]


LOSSES = ("iem", "mse")


def make_head(*, loss: str, learning_rate: float, weight_decay: float,
              seed: int):
    """The arm's head: the ladder's own ``EmbeddingHeadBackbone``, with the
    loss swapped and a per-epoch diagnostic recorder attached.

    **The ladder's arm path is not modified.** This subclasses it, so the
    IEM arm and the MSE control differ from every existing ladder arm in
    exactly the loss and in nothing else -- and differ from EACH OTHER in
    the loss alone, which is what makes the control matched
    (``PHASE_11_BUILD_REGISTERED``).

    **The recorder costs no extra forward.** ``_epoch`` counts in
    ``train_epoch``, and the first ``predict`` after each epoch is the
    harness's own inner-val call -- the Phase 10 trick
    (``phase10.CURVES_AND_STAGES_WRITTEN``), reused rather than
    reinvented. Epoch 0 is captured by gate 3's prediction, which is what
    the joint trigger measures from.
    """
    from dataclasses import dataclass, field

    from .train.torch_backbone import EmbeddingHeadBackbone

    if loss not in LOSSES:
        raise Phase11Error(f"loss {loss!r} is not one of {LOSSES}")

    @dataclass
    class Head(EmbeddingHeadBackbone):
        loss: str = "mse"
        by_epoch: dict = field(default_factory=dict)
        _epoch: int = 0

        def reset(self, train_labels):
            super().reset(train_labels)
            self._epoch = 0
            self.by_epoch = {}

        def train_epoch(self, features, labels):
            import numpy as np
            import torch

            if self.loss == "mse":
                self._epoch += 1
                return super().train_epoch(features, labels)

            self._ensure(features.shape[1])
            x = torch.as_tensor(
                np.asarray(features, dtype=np.float32), device=self._device
            )
            y = torch.as_tensor(
                np.asarray(labels, dtype=np.float32), device=self._device
            )
            value = 0.0
            for _ in range(self.max_steps):
                self._optimizer.zero_grad(set_to_none=True)
                objective = iem_loss(x @ self._weights + self._bias, y)
                objective.backward()
                self._optimizer.step()
                value = float(objective.item())
            self._epoch += 1
            return value

        def predict(self, features):
            out = super().predict(features)
            if self._epoch not in self.by_epoch:
                self.by_epoch[self._epoch] = out.copy()
            return out

    return Head(
        loss=loss, learning_rate=learning_rate, weight_decay=weight_decay,
        seed=seed,
    )


def degeneracy_signature(by_epoch: dict, inner_labels, selected_epoch: int,
                         crossover: float = IEM_CROSSOVER) -> dict:
    """The joint trigger for one fold (``DEGENERACY_APPLICATION_RULE``).

    Fires only if ALL THREE move the degenerate way between epoch 0 and
    the fold's selected epoch: span DOWN, below-crossover share UP,
    inner-val PCC DOWN. Any one alone is ordinary training.
    """
    import numpy as np

    from .eval.metrics import pcc

    inner_labels = np.asarray(inner_labels, dtype=float)

    def row(epoch):
        predictions = np.asarray(by_epoch[epoch], dtype=float)
        residual = predictions - inner_labels
        return {
            "epoch": int(epoch),
            "span": float(predictions.max() - predictions.min()),
            "below_crossover_share": float(
                np.mean(np.abs(residual) < crossover)
            ),
            "inner_val_pcc": float(pcc(inner_labels, predictions)),
        }

    if 0 not in by_epoch or selected_epoch not in by_epoch:
        return {"fired": None, "reason": "epoch 0 or the selected epoch was not recorded"}
    start, end = row(0), row(selected_epoch)
    moves = {
        "span_narrowed": bool(end["span"] < start["span"]),
        "share_rose": bool(
            end["below_crossover_share"] > start["below_crossover_share"]
        ),
        # A non-finite PCC at either end does NOT count as a fall: an
        # untrained head predicts a constant, and metrics.pcc is nan on a
        # constant vector for reasons unrelated to the loss
        # (phase10.SD_ZERO_HAS_TWO_CAUSES).
        "pcc_fell": bool(
            np.isfinite(start["inner_val_pcc"])
            and np.isfinite(end["inner_val_pcc"])
            and end["inner_val_pcc"] < start["inner_val_pcc"]
        ),
    }
    return {
        "fired": bool(all(moves.values())),
        "moves": moves,
        "epoch_0": start,
        "selected": end,
    }


def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "scope": PHASE_11_SCOPE_AND_SILENCES,
        "exit_criteria": PHASE_11_EXIT_CRITERIA,
        "pre_step": PRE_STEP_REGISTERED,
        "readings": PRE_STEP_READINGS,
        "config_shape": PRE_STEP_CONFIG_SHAPE,
        "crossover": IEM_CROSSOVER_INVERTS,
        "bounds": IEM_BOUNDS_ARE_A_DOMAIN,
        "case_identity": CASE_IEM_IDENTITY_OPEN,
        "direction_inference": IEM_DIRECTION_INFERENCE,
        "observed": PRE_STEP_OBSERVED,
        "mae_decomposition": MAE_CONTROL_DECOMPOSITION,
        "hedging_mechanism": HEDGING_MECHANISM_MEASURED,
        "hedging_check": HEDGING_CHECK_REGISTERED,
        "domain_breaches": DOMAIN_BREACHES_OBSERVED,
        "exit_walk": PRESTEP_EXIT_WALK,
        "while_blocked": PHASE_11_WHILE_BLOCKED,
        "hedging_check_observed": HEDGING_CHECK_OBSERVED,
        "narrow_predictors": A_FAVOURS_NARROW_PREDICTORS,
        "errors": FOUR_ERRORS_ONE_PLACE,
        "supervisor_ask": SUPERVISOR_CONSOLIDATED_ASK,
        "closed": PRESTEP_CLOSED,
        "direction_answered": IEM_DIRECTION_ANSWERED,
        "degeneracy_pull": DEGENERACY_PULL_REGISTERED,
        "build_scope": PHASE_11_BUILD_SCOPE,
        "build_registered": PHASE_11_BUILD_REGISTERED,
        "loss_verbatim": IEM_LOSS_VERBATIM,
        "degeneracy_rule": DEGENERACY_APPLICATION_RULE,
        "build_exit_criteria": PHASE_11_BUILD_EXIT_CRITERIA,
        "arms_observed": LOSS_ARMS_OBSERVED,
        "paired_coverage": PAIRED_CLAIM_COVERAGE,
        "paired_observed": PAIRED_LOSS_OBSERVED,
        "log_defect": THE_LOG_WAS_PARTIAL,
        "closing": PHASE_11_CLOSING,
        "sequence_renumbered": PHASE_SEQUENCE_RENUMBERED,
    }
