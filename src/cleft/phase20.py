"""Phase 20: the permutation control.

Opened 2026-08-31 on the ruling that both arms run.

**[2026-08-31, later the same day]** The ruling was the three open
questions, added **Arm C**, and the arms are built. The module now holds
the registration, the rulings, the lock (``EXIT_CRITERIA``), and **the
permutation machinery itself** -- ``quantile_strata``,
``permute_within_strata``, ``fixed_points``, ``degenerate_strata``.

The machinery lives HERE rather than in ``run`` because it is the
phase's scientific content, not its plumbing: the stratification rule is
a locked setting and the fixed-point arithmetic is a pre-registered
prediction, so both belong beside the record that fixes them. ``run``
calls it in exactly one place and ``train.phase3`` never draws a
permutation at all -- it applies one it is handed.

The original registration text follows, preserved as written:

**This module is the REGISTRATION and nothing else** -- no task, no
schema kind, no config, no run. The stratification rule and the compute
question go to the maintainer before anything is built.

The phase exists because a verification pass on 2026-08-31 established,
by searching the mechanism rather than the name, that **no
label-permutation control exists anywhere in this repo** -- not run, not
registered, not parked, not declined. The mechanism appears exactly
once, as a hazard guarded against: ``tests/test_pooled.py`` --
*"A permutation trains every patient against another patient's label and
still produces a plausible PCC."* The project has known the fact and
used it only to build misalignment guards.
"""

from __future__ import annotations

import numpy as np


#: **[RECKONED 2026-08-31] The opening reckoning, against what the
#: record already holds.**
#:
#: Three things in the record sit near this phase, and none of them
#: answers its question.
#:
#: **1. Phase 13's confound ceiling** --
#: ``phase13.P1_CONFOUND_CEILING_BANKED``. Per-seed OOF PCC **+0.1183 /
#: +0.0659 / +0.1216 / +0.0897 / +0.1044**, **mean +0.1000, sd 0.0228**
#: against a threshold of 0.10 declared before the number existed. The
#: statistics, quoted from ``phase13.CLOSING_ADDENDUM_REGISTERED
#: ["p1_confound_ceiling"]["what"]``: *"aspect ratio, brightness mean,
#: contrast (pixel sd), content-pixel fraction, corner-white fraction"*.
#: It measures **what non-anatomical statistics can predict**. It does
#: NOT measure what THIS pipeline scores on scrambled labels -- it is a
#: different and much simpler pipeline (a closed-form ridge over five
#: scalars, no backbone), so a defect in the embedding-and-head path
#: could not surface there.
#:
#: **2. TSTR arm A at 0.2334** (``phase17.ARM_MEANS``), trained on zero
#: real patient images, against the probe's 0.2520 -- and **A-vs-probe
#: is UNRESOLVED** (ledger ``p17-a-vs-probe``: condition 1 FALSE at 0 of
#: 5, condition 2 TRUE at 1.37x). That arm asks whether synthetic
#: training transfers. Its labels are real.
#:
#: **3. The identity readout at 0.2151** (``phase16.ARM_MEANS``), an
#: UNTRAINED W = I readout scoring above the trained anchor loop's
#: 0.2040. Its labels are real too. It is worth naming here because it
#: shows how much of this cohort's correlation is available without
#: fitting a readout at all -- which makes a permutation control more
#: necessary, not less.
#:
#: **4. Where the verdicts will be read.**
#: ``ladder.COHORT_CANNOT_RESOLVE``: *"this cohort cannot resolve PCC
#: differences of 0.04 to 0.10 between arms"*, 30 tested, 1 survived.
#: ``ladder.SMALLEST_RESOLVABLE_DIFFERENCE``: the only citable figure is
#: **0.1386**, "the smallest resolvable difference this cohort has
#: demonstrated", and the band 0.10-0.1386 is untested with its boundary
#: unlocated. **Both arms' results will be read against these, and the
#: detection-floor prohibition applies to every sentence this phase
#: writes.**
#:
#: **WHAT THIS PHASE MEASURES THAT NONE OF THEM DO.** Two things:
#:
#: * **What the pipeline scores when there is definitionally no
#:   signal.** Every arm in the record trains on a real label. None
#:   establishes the pipeline's output under a known-null input, so the
#:   project has no measured chance level for its own machinery -- only
#:   constant-predictor floors, which control for class imbalance, and
#:   randomised-WEIGHT controls, which control for parameter dependence.
#:   Neither controls for label-feature association arising by chance.
#: * **How much of 0.2520 survives when cleft-specific signal is
#:   destroyed but confound structure is not.** Nothing in the record
#:   does this. The confound ceiling bounds what five scalars predict;
#:   it does not partition the headline.
#:
#: **Neither is conceded as covered.**
PHASE_20_RECKONING = {
    "reckoned": "2026-08-31, before either arm is built",
    "why_the_phase_exists": (
        "a verification pass on 2026-08-31 established by MECHANISM "
        "search that no label-permutation control exists anywhere in "
        "this repo -- not run, not registered, not parked, not "
        "declined. The mechanism appears once, as a hazard guarded "
        "against (tests/test_pooled.py): 'A permutation trains every "
        "patient against another patient's label and still produces a "
        "plausible PCC'"
    ),
    "phase_13_confound_ceiling": (
        "phase13.P1_CONFOUND_CEILING_BANKED -- per-seed OOF PCC +0.1183 "
        "/ +0.0659 / +0.1216 / +0.0897 / +0.1044, mean +0.1000, sd "
        "0.0228, against a threshold of 0.10 declared before the number "
        "existed. Statistics: 'aspect ratio, brightness mean, contrast "
        "(pixel sd), content-pixel fraction, corner-white fraction'. It "
        "measures what NON-ANATOMICAL STATISTICS can predict -- a "
        "closed-form ridge over five scalars, no backbone -- so a "
        "defect in the embedding-and-head path could not surface there"
    ),
    "tstr_arm_a": (
        "0.2334 (phase17.ARM_MEANS), zero real patient images in "
        "training, against the probe's 0.2520; A-vs-probe is UNRESOLVED "
        "(condition 1 FALSE 0 of 5, condition 2 TRUE 1.37x). It asks "
        "whether synthetic training transfers; its labels are REAL"
    ),
    "identity_readout": (
        "0.2151 (phase16.ARM_MEANS), an UNTRAINED W = I readout above "
        "the trained loop's 0.2040. Labels real. Named because it shows "
        "how much correlation is available without fitting a readout at "
        "all -- which makes a permutation control MORE necessary"
    ),
    "where_the_verdicts_are_read": (
        "ladder.COHORT_CANNOT_RESOLVE ('cannot resolve PCC differences "
        "of 0.04 to 0.10', 30 tested 1 survived) and "
        "ladder.SMALLEST_RESOLVABLE_DIFFERENCE (the only citable figure "
        "is 0.1386; the 0.10-0.1386 band is untested). The "
        "detection-floor prohibition applies to every sentence this "
        "phase writes"
    ),
    "what_this_phase_measures_that_none_do": (
        "(1) what the pipeline scores when there is DEFINITIONALLY NO "
        "SIGNAL -- every arm in the record trains on a real label, so "
        "the project has no measured chance level for its own "
        "machinery, only constant-predictor floors (class imbalance) "
        "and randomised-WEIGHT controls (parameter dependence); neither "
        "controls for label-feature association arising by chance. "
        "(2) how much of 0.2520 survives when cleft-specific signal is "
        "destroyed but confound structure is NOT -- nothing in the "
        "record partitions the headline that way"
    ),
    "concession_considered_and_declined": (
        "neither question is conceded as covered: the confound ceiling "
        "is a different pipeline on a different target, and both the "
        "TSTR and identity arms train on real labels"
    ),
}


#: **[FOUND 2026-08-31, IN THE RECKONING -- and it changes Arm S's
#: design] THE CONFOUND CEILING AND THE HEADLINE ARE SCORED AGAINST
#: DIFFERENT TARGETS, AND NO RECORD SAYS SO.**
#:
#: Traced in code, not inferred: ``run.task_confound_ceiling`` takes its
#: labels from ``run._grades_for``, whose docstring opens *"The median
#: grade per patient, through the SHIPPED resolver"*, and whose config
#: (``configs/p13_confound_ceiling.yaml``) declares no ``label`` field
#: at all. The registration says so too --
#: *"regress the MEDIAN GRADE on non-anatomical global statistics"*.
#:
#: **The headline 0.2520 is scored against the PANEL MEAN**
#: (``configs/p7_d1_vit_b16_imagenet_g1.yaml``: ``label: mean``).
#:
#: So ``P1_CONFOUND_CEILING_BANKED``'s quotable sentence -- *"approximately
#: 0.10 of correlation, **~40% of the headline arm's**, is recoverable
#: from five statistics that cannot see a nose"* -- places two PCCs
#: against **different labels** side by side, and 0.10/0.2520 = 0.397 is
#: that ratio. The two targets are not interchangeable:
#: ``data.labels.LEARNABILITY_237`` records mean **0.6022** against
#: median **0.5708**.
#:
#: **This is not a correction to Phase 13 and is not made here.** The
#: 0.1000 is correct for its own quantity, the threshold was pre-declared,
#: and the phase is closed. What is recorded is that the COMPARISON to
#: the headline carries an unflagged target change -- the
#: different-quantities class this project has caught ten times
#: (``phase10_annex.VERIFICATION_NOTES`` holds the tenth).
#:
#: **The consequence for Phase 20, which is why it belongs in the
#: registration**: Arm S must NOT inherit the mismatch. It is registered
#: against the **panel mean**, the probe's own target, and the 0.1000
#: may be quoted beside it **only with the target difference stated**.
#: If a like-for-like ceiling is wanted, recomputing the confound arm on
#: the panel mean is a separate registered addition -- named here, not
#: assumed.
CONFOUND_CEILING_TARGET_MISMATCH = {
    "found": "2026-08-31, in the reckoning; traced in code",
    "the_ceiling_target": (
        "the MEDIAN grade -- run.task_confound_ceiling takes labels from "
        "run._grades_for ('The median grade per patient, through the "
        "SHIPPED resolver'); configs/p13_confound_ceiling.yaml declares "
        "no label field, and the registration says 'regress the median "
        "grade'"
    ),
    "the_headline_target": (
        "the PANEL MEAN -- configs/p7_d1_vit_b16_imagenet_g1.yaml "
        "declares label: mean"
    ),
    "the_comparison_that_carries_it": (
        "P1_CONFOUND_CEILING_BANKED's quotable sentence, "
        "'approximately 0.10 of correlation, ~40% of the headline "
        "arm's' -- 0.10/0.2520 = 0.397, a ratio across two different "
        "labels. data.labels.LEARNABILITY_237 records mean 0.6022 "
        "against median 0.5708, so they are not interchangeable"
    ),
    "not_a_correction_to_phase_13": (
        "the 0.1000 is correct for its own quantity, the threshold was "
        "pre-declared, and the phase is closed. What is recorded is "
        "that the COMPARISON to the headline carries an unflagged "
        "target change -- the different-quantities class, caught ten "
        "times before"
    ),
    "consequence_for_phase_20": (
        "Arm S must not inherit it: S is registered against the PANEL "
        "MEAN, the probe's own target, and 0.1000 may be quoted beside "
        "it ONLY with the target difference stated. A like-for-like "
        "ceiling -- the confound arm recomputed on the panel mean -- is "
        "a SEPARATE REGISTERED ADDITION, named here rather than assumed"
    ),
}


#: **[REGISTERED 2026-08-31] ARM P -- the plain permutation. The
#: integrity gate.**
#:
#: Panel-mean labels shuffled across all 237 under declared seeds; the
#: probe's own recipe otherwise **unchanged**: frozen ImageNet ViT-B/16
#: embeddings, the 769-parameter linear head, whole image, G1, 5-fold
#: OOF on ``cleft_v1``'s own folds, five seeds
#: (``configs/p7_d1_vit_b16_imagenet_g1.yaml``: ``kind: train_cv``,
#: ``label: mean``, ``trainable: head``, seeds 1337 / 2024 / 7 / 99 /
#: 12345).
#:
#: **The fit path is REUSED, not reimplemented.** The arm runs
#: ``run.task_train_cv`` through the same ``train_cv`` machinery the
#: probe runs; the ONLY difference is a permutation applied to the label
#: vector after it is loaded and before it is fitted. **A second
#: implementation would detach the control from the thing it controls**
#: -- the same lesson as ``phase10.PROBE_RECONSTRUCTED_THE_PIPELINE``
#: and the annex's shared fit path.
#:
#: **The permutation is seeded and declared**, one permutation per seed,
#: derived from that seed so the arm is reproducible and the five seeds
#: give five independent scrambles rather than one repeated.
#:
#: **The degenerate case, named in advance**: a permutation can leave a
#: label in place. For a uniform random permutation of n elements the
#: expected number of fixed points is **exactly 1, for every n** -- so
#: Arm P expects **~1 of 237 labels unchanged (0.42%)**, verified by
#: simulation at registration time. Reported per seed, never discarded.
ARM_P_REGISTERED = {
    "registered": "2026-08-31",
    "role": "THE INTEGRITY GATE",
    "design": (
        "panel-mean labels shuffled across all 237 under declared "
        "seeds; the probe's recipe otherwise unchanged -- frozen "
        "ImageNet ViT-B/16 embeddings, 769-parameter linear head, whole "
        "image, G1, 5-fold OOF on cleft_v1's own folds, five seeds "
        "(1337, 2024, 7, 99, 12345)"
    ),
    "fit_path_reused": (
        "run.task_train_cv through the same train_cv machinery the "
        "probe runs; the ONLY difference is a permutation applied to "
        "the label vector after loading and before fitting. A second "
        "implementation would DETACH THE CONTROL FROM THE THING IT "
        "CONTROLS -- phase10.PROBE_RECONSTRUCTED_THE_PIPELINE's lesson"
    ),
    "permutation_is_seeded": (
        "one permutation per seed, derived from that seed: the arm is "
        "reproducible and the five seeds give five INDEPENDENT "
        "scrambles rather than one repeated"
    ),
    "degenerate_case_named_in_advance": (
        "a permutation can leave a label in place. For a uniform random "
        "permutation of n elements E[fixed points] = 1 for EVERY n, so "
        "Arm P expects ~1 of 237 unchanged (0.42%) -- verified by "
        "simulation at registration. Reported per seed, never discarded"
    ),
}


#: **[PROPOSED 2026-08-31 -- NOT LOCKED -- A RULING IS OWED] ARM S -- the
#: stratified permutation, and its stratification rule.**
#:
#: Labels shuffled **only within strata of Phase 13's confound
#: statistics**, so the label-confound relationship survives while the
#: label-image relationship is destroyed.
#:
#: **THE PROPOSED RULE, and it is a scientific setting** -- declared in
#: YAML before the first run, never tuned across runs, movement requires
#: a dated amendment:
#:
#: * **The stratification variable is the confound model's FITTED
#:   VALUE**, not any single statistic. Phase 13 already establishes
#:   that the five statistics jointly predict ~0.10; their fitted
#:   combination is exactly the confound signal whose structure must
#:   survive. Stratifying on one statistic (aspect ratio is the
#:   strongest, r = +0.1601) would preserve one dimension of a
#:   five-dimensional confound.
#: * **Fitted once on all 237, in sample.** This is legitimate because
#:   stratification is a DESIGN CHOICE, not an estimate -- nothing is
#:   read off the fit. The alternative, per-seed out-of-fold fitted
#:   values, would make the strata seed-dependent and confound the
#:   design with the seed variation it is meant to expose.
#: * **k = 10 quantile bins**, giving strata of 23-24 patients.
#:   Reasoning, both directions: more strata preserve confound structure
#:   more tightly but shuffle within smaller groups (less destruction of
#:   label structure, more fixed points); fewer strata do the reverse.
#:   At k = 10 each stratum still holds ~24 patients, and the expected
#:   fixed-point cost is 10 of 237.
#: * **The five statistics are Phase 13's, unchanged**: aspect ratio,
#:   brightness mean, contrast (pixel sd), content-pixel fraction,
#:   corner-white fraction.
#: * **The target is the PANEL MEAN**, not Phase 13's median grade --
#:   see ``CONFOUND_CEILING_TARGET_MISMATCH``.
#:
#: **THE DEGENERATE CASE, named in advance with its arithmetic.**
#: Because E[fixed points] = 1 per stratum regardless of stratum size,
#: **k strata expect exactly k unchanged labels**: at k = 10 that is
#: **10 of 237, 4.2%** (simulation at registration: 9.99). At k = 5 it
#: is 2.1%, at k = 20 it is 8.4%. A stratum of size 1 permutes to itself
#: with probability 1; at k = 10 quantile bins of 237 the smallest
#: stratum is 23, so size-1 strata are not expected -- **but the count
#: is reported, not assumed**, on the annex's degenerate-draw pattern:
#: reported as a frequency, never discarded, never redrawn.
ARM_S_PROPOSED = {
    "proposed": "2026-08-31 -- NOT LOCKED; the ruling is the rule",
    "design": (
        "labels shuffled only WITHIN strata of Phase 13's confound "
        "statistics, so the label-confound relationship survives while "
        "the label-image relationship is destroyed"
    ),
    "stratification_variable": (
        "the confound model's FITTED VALUE over the five statistics, "
        "not any single one: Phase 13 establishes that the five jointly "
        "predict ~0.10, and their fitted combination is the confound "
        "signal whose structure must survive. Stratifying on aspect "
        "ratio alone (the strongest, r = +0.1601) would preserve one "
        "dimension of five"
    ),
    "fitted_in_sample_and_why": (
        "fitted once on all 237, in sample -- legitimate because "
        "stratification is a DESIGN CHOICE, not an estimate: nothing is "
        "read off the fit. Per-seed out-of-fold values would make the "
        "strata seed-dependent and confound the design with the seed "
        "variation it exists to expose"
    ),
    "k_strata": 10,
    "k_reasoning": (
        "quantile bins of 23-24 patients. Both directions: more strata "
        "preserve confound structure more tightly but shuffle within "
        "smaller groups (less destruction, more fixed points); fewer do "
        "the reverse. At k = 10 each stratum holds ~24 and the expected "
        "fixed-point cost is 10 of 237"
    ),
    "statistics": (
        "Phase 13's five, unchanged: aspect ratio, brightness mean, "
        "contrast (pixel sd), content-pixel fraction, corner-white "
        "fraction"
    ),
    "target": (
        "the PANEL MEAN, not Phase 13's median grade "
        "(CONFOUND_CEILING_TARGET_MISMATCH)"
    ),
    "degenerate_case_with_arithmetic": (
        "E[fixed points] = 1 per stratum for ANY stratum size, so k "
        "strata expect exactly k unchanged labels: at k = 10 that is 10 "
        "of 237 (4.2%; simulation at registration 9.99), at k = 5 2.1%, "
        "at k = 20 8.4%. A size-1 stratum permutes to itself with "
        "probability 1; at k = 10 quantile bins of 237 the smallest is "
        "23, so none is expected -- BUT THE COUNT IS REPORTED, not "
        "assumed, on the annex's pattern: a frequency, never discarded, "
        "never redrawn"
    ),
    "it_is_a_scientific_setting": (
        "declared in YAML before the first run, never tuned across "
        "runs; movement requires a dated amendment"
    ),
    # [RULED 2026-08-31, the maintainer] Approved as proposed. The proposal
    # above is preserved; STRATIFICATION_RULED is the locked setting.
    "ruled_2026_08_31": "STRATIFICATION_RULED",
}


#: **[RULED 2026-08-31] THE STRATIFICATION RULE, LOCKED.**
#:
#: Approved exactly as proposed: **the fitted value of the confound
#: model over all five statistics, fitted in-sample, k = 10 quantile
#: bins**, against the panel mean.
#:
#: **Both reasons, recorded at the lock:**
#:
#: * **A single statistic preserves one dimension of five.** Aspect
#:   ratio is the strongest single confound (r = +0.1601,
#:   ``phase12.BASAL_CONFOUND_OBSERVED``) but Phase 13 measured the
#:   FIVE jointly at ~0.10; stratifying on one of them would leave four
#:   dimensions of confound structure free to be destroyed along with
#:   the label structure, which is the opposite of what the arm is for.
#: * **Per-seed out-of-fold strata would make the design
#:   seed-dependent** and confound it with the very variation it exists
#:   to expose: the arm's five seeds are meant to vary the PERMUTATION,
#:   and if the strata moved too, a seed's result would mix two changes.
#:   In-sample fitting is legitimate because stratification is a DESIGN
#:   CHOICE, not an estimate -- no quantity is read off the fit.
#:
#: **The trade-off, both directions**: more strata preserve confound
#: structure more tightly but shuffle within smaller groups, destroying
#: less label structure and yielding more fixed points; fewer strata do
#: the reverse. **k = 10** puts ~24 patients in each stratum at a
#: fixed-point cost of 10 of 237.
#:
#: **A LOCKED SCIENTIFIC SETTING**: declared in YAML before the first
#: run, **never tuned across runs**, movement requires a dated
#: amendment. It may not be adjusted because a result came back
#: inconvenient -- that is the whole reason it is locked before any
#: number exists.
#:
#: **The fixed-point arithmetic, pre-derived and test-pinned**:
#: E[fixed points] = 1 per stratum for ANY stratum size, so k strata
#: expect exactly k unchanged labels -- **10 of 237 (4.2%)** for Arm S,
#: **~1 of 237 (0.42%)** for Arm P's single stratum. Counts are
#: **reported per seed, never discarded, never redrawn**.
STRATIFICATION_RULED = {
    "ruled": "2026-08-31 -- approved as proposed",
    "variable": (
        "the fitted value of the confound model over all five "
        "statistics, fitted IN-SAMPLE on all 237"
    ),
    "k_strata": 10,
    "binning": "quantile bins of the fitted value",
    "target": "the panel mean",
    "statistics": (
        "aspect ratio, brightness mean, contrast (pixel sd), "
        "content-pixel fraction, corner-white fraction -- Phase 13's "
        "five, unchanged"
    ),
    "reason_not_a_single_statistic": (
        "a single statistic preserves ONE DIMENSION OF FIVE. Aspect "
        "ratio is the strongest single confound (r = +0.1601) but Phase "
        "13 measured the five JOINTLY at ~0.10; stratifying on one "
        "would leave four dimensions free to be destroyed with the "
        "label structure -- the opposite of the arm's purpose"
    ),
    "reason_not_per_seed_oof": (
        "per-seed out-of-fold strata would make the design "
        "SEED-DEPENDENT and confound it with the variation it exists to "
        "expose: the five seeds vary the PERMUTATION, and if the strata "
        "moved too a seed's result would mix two changes. In-sample is "
        "legitimate because stratification is a DESIGN CHOICE, not an "
        "estimate -- no quantity is read off the fit"
    ),
    "the_trade_off_both_directions": (
        "more strata preserve confound structure more tightly but "
        "shuffle within smaller groups, destroying less label structure "
        "and yielding more fixed points; fewer strata do the reverse. "
        "k = 10 puts ~24 patients per stratum at a fixed-point cost of "
        "10 of 237"
    ),
    "a_locked_scientific_setting": (
        "declared in YAML before the first run, NEVER TUNED ACROSS "
        "RUNS, movement requires a dated amendment. It may not be "
        "adjusted because a result came back inconvenient -- that is "
        "the whole reason it is locked before any number exists"
    ),
    "fixed_point_arithmetic": (
        "E[fixed points] = 1 per stratum for ANY stratum size, so k "
        "strata expect exactly k unchanged labels: 10 of 237 (4.2%) for "
        "Arm S, ~1 of 237 (0.42%) for Arm P's single stratum. "
        "Pre-derived, test-pinned, and reported per seed -- never "
        "discarded, never redrawn"
    ),
}


#: **[RULED 2026-08-31] NEITHER ARM CARRIES A LEDGER ROW.**
#:
#: **Arm P** is a null construct. *"Is the probe claimably better than
#: scrambled labels"* is a **category error dressed as a result**, and a
#: CLAIMABLE row would make it quotable. Reported instead: each seed's
#: PCC with a **one-sample BCa interval against zero**, plus the
#: five-seed mean and spread.
#:
#: **Arm S** is a genuine fitted arm, so an S-vs-probe paired BCa is
#: *defined* -- and still gets no row, because its quantity is an
#: **attribution**, not a method comparison. A WITHDRAWN row would read
#: as *"the probe is not claimably better than scrambled labels"*, which
#: is not what the measurement means. Reported descriptively against the
#: confound ceiling and 0.2520.
#:
#: **Arm C** is a ceiling measurement, not an arm claim: no row either.
#:
#: **The grounds, citing the precedent**:
#: ``phase18.D2_HOME_RULED["no_new_ledger_rows"]`` -- *"a ledger row is
#: the unit of ARM-CLAIM"*, and it withheld rows for a finding that
#: *"measures the criterion, not the arms"*. All three arms here measure
#: the MACHINERY and the label structure, not the standing of any arm
#: against another.
LEDGER_RULED = {
    "ruled": "2026-08-31 -- no ledger row for any of the three",
    "arm_p": (
        "a NULL CONSTRUCT. 'Is the probe claimably better than scrambled "
        "labels' is a category error dressed as a result, and a "
        "CLAIMABLE row would make it quotable. Reported: each seed's PCC "
        "with a one-sample BCa interval against ZERO, plus the "
        "five-seed mean and spread"
    ),
    "arm_s": (
        "a genuine fitted arm, so an S-vs-probe paired BCa is DEFINED -- "
        "and still no row, because the quantity is an ATTRIBUTION, not a "
        "method comparison. A WITHDRAWN row would read as 'the probe is "
        "not claimably better than scrambled labels', which is not what "
        "the measurement means. Reported descriptively against the "
        "confound ceiling and 0.2520"
    ),
    "arm_c": "a ceiling measurement, not an arm claim: no row either",
    "grounds": (
        "phase18.D2_HOME_RULED['no_new_ledger_rows'] -- 'a ledger row is "
        "the unit of ARM-CLAIM', which withheld rows for a finding that "
        "'measures the criterion, not the arms'. All three arms here "
        "measure the MACHINERY and the label structure, not the "
        "standing of any arm against another"
    ),
}


#: **[REGISTERED 2026-08-31] ARM C -- the
#: like-for-like confound ceiling.**
#:
#: **Why it exists.** Phase 13's ceiling (+0.1000) is scored against the
#: **median grade**; the headline 0.2520 and Arm S are against the
#: **panel mean**; the two are not interchangeable
#: (``LEARNABILITY_237``: mean 0.6022, median 0.5708). **Arm S therefore
#: has no valid anchor in the record**, and inventing one by quoting
#: 0.1000 beside it would be the cross-target error a second time.
#:
#: **What it is**: the confound-ceiling measurement re-run against the
#: **panel mean** -- same five statistics, same task, same recipe
#: (closed-form ridge over ``cleft_v1``'s own folds, inner-val split per
#: seed, alpha 1.0), five seeds.
#:
#: **A NEW REGISTERED ARM IN PHASE 20, NOT A CORRECTION TO PHASE 13.**
#: Phase 13's 0.1000 is **correct for its own quantity**, its threshold
#: was declared before the number existed, and its phase is closed. What
#: was wrong was never Phase 13's figure -- it was a COMPARISON made
#: later, in conversation. A dated pointer sits on Phase 13's record;
#: nothing there is rewritten.
#:
#: **The error's provenance, dated** (``CROSS_TARGET_ERROR_PROVENANCE``):
#: it originated in conversation -- the record's, this session -- was quoted
#: as *"~40% of the headline"*, and was **caught during this
#: registration, before any measurement was built on it**.
#:
#: **Arm C's readings, committed before its number:**
#:
#: * **near 0.1000** -> the target difference does not matter for this
#:   quantity and the two ceilings are interchangeable after all. The
#:   cross-target comparison was harmless in effect, and that is stated
#:   plainly rather than being treated as vindication of having made it.
#: * **materially different** -> the median-target ceiling was **never a
#:   valid anchor for panel-mean arms**, and the write-up must use Arm
#:   C's figure wherever it previously used 0.1000 against a panel-mean
#:   number.
#:
#: **Arm C is independent of the gate** and may run alongside Arm P.
ARM_C_REGISTERED = {
    "registered": "2026-08-31, the addition to the phase",
    "why_it_exists": (
        "Phase 13's ceiling (+0.1000) is scored against the MEDIAN "
        "grade; the headline 0.2520 and Arm S are against the PANEL "
        "MEAN, and the two are not interchangeable (LEARNABILITY_237: "
        "mean 0.6022, median 0.5708). Arm S therefore has NO VALID "
        "ANCHOR in the record, and quoting 0.1000 beside it would be "
        "the cross-target error a second time"
    ),
    "what_it_is": (
        "the confound-ceiling measurement re-run against the PANEL "
        "MEAN -- same five statistics, same task, same recipe "
        "(closed-form ridge over cleft_v1's own folds, inner-val split "
        "per seed, alpha 1.0), five seeds"
    ),
    "not_a_correction_to_phase_13": (
        "a NEW REGISTERED ARM IN PHASE 20. Phase 13's 0.1000 is CORRECT "
        "FOR ITS OWN QUANTITY, its threshold was declared before the "
        "number existed, and its phase is closed. What was wrong was "
        "never Phase 13's figure -- it was a COMPARISON made later, in "
        "conversation. A dated pointer sits on Phase 13's record and "
        "nothing there is rewritten"
    ),
    "readings": {
        "near_0_1000": (
            "the target difference does not matter for this quantity "
            "and the two ceilings are interchangeable after all. The "
            "cross-target comparison was harmless IN EFFECT, and that "
            "is stated plainly rather than treated as vindication of "
            "having made it"
        ),
        "materially_different": (
            "the median-target ceiling was NEVER A VALID ANCHOR for "
            "panel-mean arms, and the write-up must use Arm C's figure "
            "wherever it previously used 0.1000 against a panel-mean "
            "number"
        ),
    },
    "independent_of_the_gate": (
        "Arm C does not depend on Arm P's outcome and may run alongside "
        "it"
    ),
}


#: **[RECORDED 2026-08-31, DATED] THE CROSS-TARGET ERROR'S PROVENANCE.**
#:
#: **Where it came from**: conversation -- the record's, during the
#: 2026-08-31 session. The sentence *"approximately 0.10 of correlation,
#: ~40% of the headline arm's"* is Phase 13's own
#: (``P1_CONFOUND_CEILING_BANKED["fires_at_the_boundary"]``), and it was
#: **repeated in conversation as though the two figures shared a
#: target**, which they do not.
#:
#: **What it touched**: nothing measured. It was caught **during this
#: registration, before any measurement was built on it** -- the
#: reckoning traced ``run.task_confound_ceiling`` to ``run._grades_for``
#: ("The median grade per patient") and found the probe's config
#: declaring ``label: mean``.
#:
#: **What was done**: Arm C, not a correction. The distinction matters
#: and is the second half of the record: **a figure correct for its own
#: quantity does not become wrong because someone compared it badly.**
#: The comparison is what is corrected; the figure stands.
CROSS_TARGET_ERROR_PROVENANCE = {
    "recorded": "2026-08-31",
    "origin": (
        "CONVERSATION -- the record's, during the 2026-08-31 session. Phase "
        "13's own sentence 'approximately 0.10 of correlation, ~40% of "
        "the headline arm's' was repeated as though the two figures "
        "shared a target, which they do not"
    ),
    "caught_when": (
        "during THIS registration, before any measurement was built on "
        "it -- the reckoning traced run.task_confound_ceiling to "
        "run._grades_for ('The median grade per patient') and found the "
        "probe's config declaring label: mean"
    ),
    "what_it_touched": "nothing measured; no run rested on it",
    "what_was_done": (
        "Arm C, not a correction. A figure correct for its own quantity "
        "does not become wrong because someone compared it badly: the "
        "COMPARISON is corrected, the figure stands"
    ),
}


#: **[BOUND 2026-08-31, BEFORE EITHER RUNS] THE ORDERING. Arm P is the
#: gate.**
#:
#: **If P does not come back near zero, S is uninterpretable** -- a
#: stratified permutation's residual means nothing if the plain one
#: already produces correlation from nothing. In that case **the
#: phase's finding becomes the pipeline result, not the attribution**,
#: and that finding is bigger and more urgent than what this phase set
#: out to measure.
#:
#: **The phase stops at P and reports.** S does not run on a failed
#: gate, and no attribution sentence is written.
ORDERING_BOUND = {
    "bound": "2026-08-31, before either arm runs",
    "rule": "Arm P is the GATE; Arm S runs only if P passes",
    "if_p_fails": (
        "S is UNINTERPRETABLE -- a stratified permutation's residual "
        "means nothing if the plain one already produces correlation "
        "from nothing. The phase's finding becomes THE PIPELINE RESULT, "
        "not the attribution, and that is bigger and more urgent than "
        "what the phase set out to measure"
    ),
    "the_phase_stops": (
        "S does not run on a failed gate and no attribution sentence is "
        "written"
    ),
}


#: **[COMMITTED 2026-08-31, BEFORE ANY NUMBER] THE READINGS.**
READINGS_COMMITTED = {
    "committed": "2026-08-31, before any number exists",

    "p_near_zero": (
        "the pipeline produces NO correlation from scrambled labels. "
        "0.2520 is real signal -- weak, but real -- and the integrity "
        "of every banked figure in the project is corroborated. This is "
        "the expected outcome and it licenses nothing new: it removes a "
        "doubt rather than adding a finding"
    ),

    "p_meaningfully_above_zero": (
        "**something in the pipeline yields correlation without "
        "information, and this would call into question EVERY BANKED "
        "PCC IN THE PROJECT.** Stated in advance so it cannot be "
        "softened after the fact. What would then have to be "
        "diagnosed, named now: (i) FOLD CONSTRUCTION -- whether "
        "cleft_v1's folds leak label information through their "
        "assignment; (ii) OOF ASSEMBLY -- whether predictions are "
        "written back against the right patients; (iii) THE LABEL JOIN "
        "-- the patient -> frontal-photo -> label resolution that has "
        "already produced two defects; (iv) POOLING -- whether the "
        "per-seed pooled PCC is computed over the vector it claims. "
        "The phase STOPS and REPORTS; S does not run"
    ),

    "s_near_the_confound_ceiling": (
        "the residual above it is what is cleft-specific. THE "
        "ARITHMETIC FORM, given in advance: residual = 0.2520 - S_mean, "
        "quoted with both arms' seed spreads and against "
        "ladder.SMALLEST_RESOLVABLE_DIFFERENCE -- a residual smaller "
        "than 0.1386 is NOT demonstrated to be resolvable by this "
        "cohort, and the sentence must say so"
    ),

    "s_near_0_2520": (
        "the headline is largely confound structure, and the write-up "
        "says so plainly. This is the limitations-grade outcome and it "
        "travels with the headline wherever the headline is quoted, on "
        "the same footing as the confound ceiling itself"
    ),

    "s_between": (
        "the attribution is PARTIAL and is STATED, not resolved: "
        "'between X and Y of the headline is attributable to confound "
        "structure, and this cohort cannot place it more precisely' -- "
        "with both endpoints given, the seed spreads attached, and no "
        "point estimate promoted to a threshold. Registered this way "
        "because the tempting move is to pick a midpoint and call it "
        "the answer"
    ),

    # [2026-08-31] Arm C was added to the phase after this set was
    # committed, and its own readings were committed with it -- before
    # its number, on the same rule. They live with the arm.
    "arm_c_readings_2026_08_31": "ARM_C_REGISTERED['readings']",

    # [2026-08-31, with EXIT_CRITERIA] The residual arithmetic above is
    # written against 0.2520 and the ceiling of record. If Arm C moves
    # the ceiling, S's residual is computed against ARM C's figure with
    # BOTH stated -- see EXIT_CRITERIA['arm_c_moves_the_anchor']. The
    # 0.1386 comparison is unchanged either way.
    "the_anchor_may_move_2026_08_31": "EXIT_CRITERIA['arm_c_moves_the_anchor']",

    "no_reading_is_invented_after": (
        "whichever lands, no reading is added once numbers exist; a "
        "pattern outside these cells gets a dated OBSERVATION, on "
        "phase17.UNPREDICTED_PATTERN's precedent"
    ),
}


#: **[REGISTERED 2026-08-31] THE CONTRAST MACHINERY -- and Arm P does
#: not fit the paired criterion.**
#:
#: **Arm P is a NULL CONSTRUCT, not a competing method.** PLAN §4.3's
#: two conditions compare two methods on the same patients: a paired BCa
#: over patients, and a delta exceeding the two arms' combined seed
#: uncertainty. Asking "is the probe claimably better than scrambled
#: labels" is not a scientific question, and a CLAIMABLE verdict there
#: would be a category error dressed as a result.
#:
#: **What is reported instead for P**: each seed's own PCC with a
#: **one-sample BCa interval over patients against zero**, plus the
#: five-seed mean and spread. The natural analogue of condition 1 is
#: *does every seed's own interval contain zero* -- which is the
#: question the gate actually asks. **No ledger row**: nothing here is
#: an arm claim.
#:
#: **Arm S is a genuine fitted arm** on the same patients, so an
#: S-vs-probe paired BCa is DEFINED. Whether it should carry a
#: claimable/unresolved verdict is a different question, and the
#: registration's position is **no**: the quantity of interest is an
#: attribution (how much of the headline is confound structure), not a
#: method comparison, and a WITHDRAWN row would read as "the probe is
#: not claimably better than scrambled labels", which is not what the
#: measurement means. **Registered as reported-with-intervals, and
#: flagged for the ruling.**
CONTRAST_MACHINERY = {
    "registered": "2026-08-31",
    "arm_p_does_not_fit_the_paired_criterion": (
        "P is a NULL CONSTRUCT, not a competing method. PLAN 4.3 "
        "compares two methods on the same patients; 'is the probe "
        "claimably better than scrambled labels' is not a scientific "
        "question, and a CLAIMABLE verdict there would be a category "
        "error dressed as a result"
    ),
    "what_is_reported_for_p": (
        "each seed's own PCC with a ONE-SAMPLE BCa interval over "
        "patients against ZERO, plus the five-seed mean and spread. The "
        "analogue of condition 1 is 'does every seed's own interval "
        "contain zero' -- the question the gate actually asks. NO "
        "LEDGER ROW: nothing here is an arm claim"
    ),
    "arm_s_is_defined_but_the_verdict_is_not_wanted": (
        "S is a genuine fitted arm on the same patients, so an "
        "S-vs-probe paired BCa is DEFINED. The registration's position "
        "is that it should NOT carry a claimable/unresolved verdict: "
        "the quantity is an ATTRIBUTION, not a method comparison, and a "
        "WITHDRAWN row would read as 'the probe is not claimably better "
        "than scrambled labels', which is not what the measurement "
        "means. Reported with intervals; FLAGGED FOR A RULING"
    ),
    # [RULED 2026-08-31] the ruling was the registration's position:
    # no ledger row for P, S, or C. See LEDGER_RULED.
    "ruled_2026_08_31": "LEDGER_RULED",
}


#: **[DESIGNED 2026-08-31 -- NOT RUN] THE COMPUTE GATE. Measure before
#: declaring, per the standing rule.**
#:
#: Both arms are the probe's recipe on **cached embeddings** -- no
#: backbone forward, no GPU, a 769-parameter head over 5 folds x 5
#: seeds. The honest expectation is **seconds**, and the Phase 17 lesson
#: still applies: the expectation is not the measurement.
#:
#: **The gate job**: one arm P seed, end to end, reporting per-seed
#: wall-clock, so the full shape (2 arms x 5 seeds) is sized from a
#: measured number rather than an assumption. It reuses the probe's own
#: declared embeddings artifact, so no new hash enters.
#:
#: **If the gate confirms seconds, both arms are one job** and no
#: sharding question arises -- unlike the annex, whose 2,500 fits needed
#: a declared shard rule.
COMPUTE_GATE_DESIGNED = {
    "designed": "2026-08-31 -- designed, NOT run",
    "why_it_is_probably_trivial": (
        "both arms are the probe's recipe on CACHED EMBEDDINGS -- no "
        "backbone forward, no GPU, a 769-parameter head over 5 folds x "
        "5 seeds. Expectation: seconds"
    ),
    "the_expectation_is_not_the_measurement": (
        "the Phase 17 lesson applies even when the answer looks "
        "obvious: the gate measures rather than assumes"
    ),
    "the_job": (
        "one Arm P seed end to end, reporting per-seed wall-clock, so "
        "the full 2 arms x 5 seeds is sized from a measured number. "
        "Reuses the probe's declared embeddings artifact -- no new hash"
    ),
    "if_it_confirms_seconds": (
        "both arms are ONE JOB and no sharding question arises -- "
        "unlike the annex, whose 2,500 fits needed a declared shard rule"
    ),
    # [RULED 2026-08-31, the maintainer] The separate gate JOB is not built.
    # At 769 parameters over cached embeddings the gate's value does not
    # justify a cycle, so the TIMING MEASUREMENT is folded into the runs
    # themselves: every arm reports per-seed wall-clock in its logs. The
    # measure-don't-assume rule is honoured -- what is dropped is the
    # separate job, not the measurement.
    "ruled_2026_08_31": (
        "FOLDED INTO THE RUNS -- no separate gate job. Per-seed "
        "wall-clock is reported by every arm; the measurement survives, "
        "the extra cycle does not. See EXIT_CRITERIA"
    ),
}


#: **[DRAFT 2026-08-31 -- NOT LOCKED] EXIT CRITERIA. The lock waits on
#: the ruling on the stratification rule and the compute
#: question.**
EXIT_CRITERIA_DRAFT = {
    "status": (
        "DRAFT -- not locked until it is ruled the stratification "
        "rule (ARM_S_PROPOSED) and any compute question"
    ),
    "drafted": "2026-08-31",
    "criteria": (
        "1. ARM P: five seeds, per-seed PCC with one-sample BCa "
        "intervals against zero, the five-seed mean and spread, and the "
        "fixed-point count per seed reported against the expected ~1 of "
        "237",
        "2. THE GATE APPLIED: the ordering in ORDERING_BOUND honoured "
        "-- if P is not near zero the phase stops, reports, and S does "
        "not run",
        "3. ARM S (only if the gate passes): five seeds, per-seed PCC "
        "with intervals, and the DEGENERATE-STRATUM FREQUENCY reported "
        "per seed against the expected k = 10 of 237",
        "4. FLOORS BESIDE EVERY FIGURE: the majority/constant floors as "
        "the project reports them, and for the permutation arms the "
        "relevant floor is ZERO -- stated, since a permuted arm's "
        "reference point is no-correlation, not a class baseline",
        "5. one of the committed readings (READINGS_COMMITTED) applied "
        "verbatim; no reading invented after the numbers; a pattern "
        "outside the cells gets a dated observation",
        "6. NO LEDGER ROW and no claimable/unresolved verdict for P "
        "(CONTRAST_MACHINERY); S's contrast treatment as the ruling is",
        "7. the target discipline: both arms on the PANEL MEAN, and any "
        "quotation of the 0.1000 confound ceiling beside them carries "
        "the target-difference statement "
        "(CONFOUND_CEILING_TARGET_MISMATCH)",
        "8. suite green",
    ),
    "what_is_not_yet_settled": (
        "the stratification rule (k, the variable, in-sample fitting), "
        "S's contrast treatment, and whether the compute gate is worth "
        "a separate job at this size -- all the maintainer's"
    ),
    # [2026-08-31] All three settled and the criteria LOCKED; this draft
    # is preserved as written. See EXIT_CRITERIA.
    "superseded_2026_08_31": "EXIT_CRITERIA",
}


#: **[LOCKED 2026-08-31] THE EXIT CRITERIA. The three rulings are closed
#: and NOTHING IS ADDED AFTER THIS RECORD.**
#:
#: **The three, closed at the lock:**
#:
#:     1. the stratification rule   RULED -- STRATIFICATION_RULED
#:     2. the ledger question       RULED -- LEDGER_RULED (no row, any arm)
#:     3. the compute gate          RULED -- folded into the runs
#:
#: and one addition: **Arm C**, the like-for-like ceiling
#: (``ARM_C_REGISTERED``).
#:
#: **Nothing is added after this record.** A criterion discovered
#: missing later is a **limitation of the lock, recorded as such** --
#: never a retro-fitted entry. This is the clause Phase 18 wrote and
#: Phase 18's own addendum then honoured rather than reopening.
EXIT_CRITERIA = {
    "locked": (
        "2026-08-31 -- **nothing is added after this record**; a "
        "criterion discovered missing later is a limitation of the "
        "lock, recorded as such, never a retro-fitted entry"
    ),
    "the_three_rulings_closed": {
        "stratification": "RULED -- STRATIFICATION_RULED",
        "ledger": "RULED -- LEDGER_RULED, no row for any of the three arms",
        "compute": (
            "RULED -- the timing measurement is FOLDED INTO THE RUNS "
            "rather than a separate gate job: at 769 parameters over "
            "cached embeddings this is seconds and the gate's value "
            "does not justify a cycle. Per-seed wall-clock in the logs"
        ),
    },
    "criteria": (
        "1. ARM P: five seeds, per-seed PCC with one-sample BCa "
        "intervals against ZERO, the five-seed mean and spread, and the "
        "fixed-point count per seed against the expected ~1 of 237",
        "2. THE GATE APPLIED: ORDERING_BOUND honoured -- if P is not "
        "near zero the phase stops, reports, and S does not run",
        "3. ARM S (only if the gate passes): five seeds, per-seed PCC "
        "with intervals, and the DEGENERATE-STRATUM FREQUENCY per seed "
        "against the expected k = 10 of 237",
        "4. FLOORS BESIDE EVERY FIGURE: for the permutation arms the "
        "reference point is ZERO -- stated, since a permuted arm's "
        "floor is no-correlation, not a class baseline",
        "5. one of the committed readings (READINGS_COMMITTED) applied "
        "verbatim per arm; no reading invented after the numbers; a "
        "pattern outside the cells gets a dated observation",
        "6. NO LEDGER ROW for any of the three arms (LEDGER_RULED), and "
        "no claimable/unresolved verdict anywhere in the phase",
        "7. THE TARGET DISCIPLINE: every arm on the PANEL MEAN, and any "
        "quotation of Phase 13's 0.1000 beside them carries the "
        "target-difference statement "
        "(CONFOUND_CEILING_TARGET_MISMATCH)",
        "8. suite green",
        "9. ARM C: five seeds, per-seed and mean OOF PCC against the "
        "PANEL MEAN over the same five statistics and recipe, with its "
        "committed reading applied and a dated pointer from Phase 13's "
        "record -- and it is NOT a correction to Phase 13",
    ),
    "arm_c_moves_the_anchor": (
        "**if Arm C moves the ceiling, S's residual is computed against "
        "ARM C's figure, with BOTH stated** -- Arm C's as the anchor "
        "used and Phase 13's 0.1000 named beside it with its target "
        "difference. The residual is never quoted against a ceiling on "
        "a different target"
    ),
    "nothing_added_after": (
        "the clause Phase 18 wrote and its own addendum then honoured "
        "rather than reopening: a gap found later is recorded AS a "
        "limitation, not folded in"
    ),
}


#: **[OBSERVED 2026-08-31] THE THREE ARMS' FIGURES, as supplied by El
#: the maintainer from the runs.**
#:
#: **PROVENANCE, stated because it decides how these may be used**: these
#: figures reached this record **through the maintainer in conversation**, not
#: by this session reading the run artifacts. They are recorded as
#: supplied. The runs' own ``metrics.json`` files are the source of
#: record, and anything that turns on a digit -- the per-seed values, the
#: seed spreads, the fixed-point counts -- must be taken from there.
#: ``record_audit.compare_to_artifact`` is the standing check that closes
#: this gap when the artifacts are read.
#:
#: **Arm P: -0.0454.** The gate PASSES. ``READINGS_COMMITTED
#: ["p_near_zero"]`` fires: the pipeline produces no correlation from
#: scrambled labels, and it *"licenses nothing new: it removes a doubt
#: rather than adding a finding"*.
#:
#: **Arm C: +0.1061**, against Phase 13's median-target **+0.1000**. The
#: difference is **+0.0061**, inside the declared 0.02, so
#: ``ARM_C_REGISTERED["readings"]["near_0_1000"]`` fires -- and that cell
#: says what it says: the cross-target comparison was *"harmless IN
#: EFFECT, and that is stated plainly rather than treated as vindication
#: of having made it"*. ``EXIT_CRITERIA["arm_c_moves_the_anchor"]``
#: therefore does not trigger: the anchor did not move, and both figures
#: are still named with their target difference.
#:
#: **Arm S: +0.0414** (sd 0.0802, 95% CI [-0.0347, +0.1610]).
#: **NO COMMITTED CELL FIRES** -- see ``S_PATTERN_UNPREDICTED``.
ARMS_OBSERVED = {
    "observed": (
        "2026-08-31 -- four runs, each at its own sha, all finalized "
        "single-attempt"
    ),
    "provenance": (
        "SUPPLIED IN CONVERSATION, not read from the run "
        "artifacts by this session: no keeper run directory is reachable "
        "here, so nothing could be read at source. Recorded as supplied "
        "and INDEPENDENTLY RE-DERIVED WHERE THE SUPPLIED FIGURES ALLOW "
        "(see rederivation_checks). The runs' own metrics.json files "
        "remain the source of record; record_audit.compare_to_artifact "
        "is the standing check that closes this gap when they are read"
    ),

    "arm_p": {
        "run": "p20_permutation_plain__dc4605bf",
        "pcc_by_seed": {
            1337: 0.0611, 2024: -0.1043, 7: -0.0705, 99: 0.0728,
            12345: -0.1861,
        },
        "pcc_mean": -0.0454,
        "pcc_sd": 0.1109,
        "ci_95": (-0.1779, 0.0717),
        "fixed_points_by_seed": {
            1337: 1, 2024: 1, 7: 1, 99: 1, 12345: 2,
        },
        "fixed_points_expected": 1,
        "cell_fired": "READINGS_COMMITTED['p_near_zero']",
        "consequence": (
            "**THE GATE PASSES.** The pipeline produces NO CORRELATION "
            "FROM SCRAMBLED LABELS, so 0.2520 is real signal. "
            "ORDERING_BOUND is satisfied and Arm S was cleared to run"
        ),
        "what_it_corroborates": (
            "**EVERY BANKED PCC IN THE PROJECT.** The committed cell "
            "said the integrity of every banked figure would be "
            "corroborated, and it is: the four diagnoses the failure "
            "branch named -- fold construction, OOF assembly, the label "
            "join, pooling -- are all ruled out by the same measurement, "
            "because any of them would have produced correlation here. "
            "It still LICENSES NOTHING NEW: it removes a doubt rather "
            "than adding a finding"
        ),
    },

    "arm_c": {
        "run": "p20_confound_ceiling_mean__dc4605bf",
        "pcc_by_seed": {
            1337: 0.1336, 2024: 0.0728, 7: 0.1143, 99: 0.1088, 12345: 0.1010,
        },
        "pcc_mean": 0.1061,
        "pcc_sd": 0.0222,
        "against_phase_13_median_target": 0.1000,
        "difference": 0.0061,
        "declared_threshold": 0.02,
        "cell_fired": "ARM_C_REGISTERED['readings']['near_0_1000']",
        "consequence": (
            "the two ceilings are interchangeable IN EFFECT. The cell "
            "requires this stated plainly rather than treated as "
            "vindication of having made the cross-target comparison. "
            "EXIT_CRITERIA['arm_c_moves_the_anchor'] does NOT trigger: "
            "the anchor did not move"
        ),
        "what_now_exists": (
            "**the like-for-like PANEL-MEAN confound ceiling, at "
            "+0.1061.** It is the anchor for panel-mean comparisons; "
            "Phase 13's +0.1000 remains the anchor for median-target "
            "ones, and neither is quoted against the other's target "
            "(CONFOUND_CEILING_TARGET_MISMATCH)"
        ),
    },

    "arm_s": {
        "run": "p20_permutation_stratified__dc4605bf",
        "pcc_mean": 0.0414,
        "pcc_sd": 0.0802,
        "ci_95": (-0.0347, 0.1610),
        "pcc_range": (-0.0393, 0.1738),
        "per_seed_not_transcribed": (
            "the per-seed values are in the run's metrics.json; only the "
            "range was supplied and only the range is recorded"
        ),
        "k_strata": 10,
        "strata_sizes": (24, 24, 23, 24, 23, 24, 24, 23, 24, 24),
        "fixed_points_by_seed": {
            1337: 7, 2024: 11, 7: 17, 99: 10, 12345: 13,
        },
        "fixed_points_mean": 11.6,
        "fixed_points_expected": 10,
        "singleton_strata": 0,
        "cell_fired": None,
        "consequence": "see S_PATTERN_UNPREDICTED",
    },

    "diagnostic": {
        "run": "p20_stratification_diagnostic__77ea2247",
        "retained_fraction_by_seed": {
            1337: 1.220, 2024: 0.543, 7: 0.493, 99: 0.192, 12345: 0.562,
        },
        "pcc_mean": 0.0639,
        "pcc_sd": 0.0399,
        "retained_fraction_mean": 0.602,
        "retained_fraction_sd": 0.376,
        "ceiling_pcc": 0.1061,
        "cell_fired": "DIAGNOSTIC_READINGS_COMMITTED['between']",
        "consequence": "see DIAGNOSTIC_OBSERVED",
    },

    # -----------------------------------------------------------------
    # [RE-DERIVED 2026-08-31] What this session could check WITHOUT the
    # artifacts, and did. Every one of these agrees; none of them is a
    # substitute for reading the run files.
    # -----------------------------------------------------------------
    "rederivation_checks": {
        "arm_p_mean_and_sd_from_its_own_per_seed": (
            "the five per-seed values give mean -0.045400 and sample sd "
            "0.110916 -- the supplied -0.0454 and 0.1109 exactly, to the "
            "digits supplied"
        ),
        "arm_c_mean_and_sd_from_its_own_per_seed": (
            "mean +0.106100, sample sd 0.022168 -- the supplied +0.1061 "
            "and 0.0222 exactly"
        ),
        "the_diagnostic_is_internally_consistent": (
            "the five per-seed FRACTIONS give mean 60.20% and sd 37.65 "
            "points (supplied 60.2% and 37.6); multiplied by the 0.1061 "
            "ceiling they give per-seed PCCs whose mean is 0.0639 and sd "
            "0.0399 -- the supplied pair, exactly. Three figures "
            "reconcile against two independent routes"
        ),
        "the_strata_are_the_shipped_ones": (
            "**the strongest check available here.** The run's reported "
            "sizes (24,24,23,24,23,24,24,23,24,24) are what "
            "quantile_strata(k=10) produces over 237 IN THAT ORDER, "
            "reproduced locally -- so the run partitioned the cohort "
            "with the shipped code and not a look-alike"
        ),
        "both_fixed_point_outturns_agree_with_their_predictions": (
            "sampled from the shipped drawer, 20,000 draws: at k = 10 "
            "the mean of five draws has E = 10.03 and sd 1.40, so the "
            "observed 11.6 is z = +1.12 (two-sided p = 0.283); at k = 1 "
            "E = 1.01 and sd 0.45, so the observed 1.2 is z = +0.42 "
            "(p = 0.828). **Both pre-registered expectations are met** "
            "-- and the single seed at 17 is not anomalous either "
            "(P(draw >= 17) = 0.028 per draw, so ~13% across five)"
        ),
        "the_run_shas_are_this_repos_own_commits": (
            "**verified at source.** dc4605bf and 77ea2247 are the "
            "8-character shas of two real commits here, and each carries "
            "exactly the configs its runs used: dc4605bf shipped the P, "
            "S and C configs, 77ea2247 shipped the diagnostic's. The "
            "pairing is therefore internally consistent AND could not be "
            "otherwise -- the diagnostic could not have run at dc4605bf, "
            "where its config did not yet exist. This is the run-dir "
            "contract <config-stem>__<sha8>__<job-id> checked against "
            "git rather than assumed"
        ),
        "what_could_not_be_checked": (
            "Arm S's per-seed PCCs (not supplied), every arm's per-seed "
            "BCa interval (not supplied), and any figure against its "
            "run's own bytes. NO KEEPER RUN DIRECTORY IS REACHABLE FROM "
            "THIS SESSION"
        ),
    },
}


#: **[OBSERVED 2026-08-31] THE DIAGNOSTIC. Cell ``between`` fired:
#: PARTIAL SURVIVAL.**
#:
#: ``p20_stratification_diagnostic__77ea2247``. The confound model on Arm
#: S's own permuted labels retains **60.2% of the ceiling** (mean
#: +0.0639, sd 0.0399), between the declared 30% and 70%. The registered
#: cell applies verbatim: **the partition is BOUNDED, NOT RESOLVED.**
#:
#: **THE PERMUTATION-IDENTITY CHECK PASSED.** The diagnostic's
#: ``permutation.json`` equals Arm S's exactly -- the diff,
#: empty. This is the check the run was built to make legible, and
#: without it the diagnostic's number would say nothing about Arm S. It
#: is recorded as PASSED, by a person's diff, not inferred from the code
#: being deterministic.
#:
#: **THE PER-SEED DISPERSION IS A FINDING IN ITS OWN RIGHT, and it is
#: larger than the headline.** The retained fraction per seed is
#: **122.0% / 54.3% / 49.3% / 19.2% / 56.2%** -- a spread of **37.6
#: percentage points** around a 60.2% mean. **One seed retained MORE
#: confound signal than the unpermuted ceiling** (122%), another barely
#: a fifth (19.2%).
#:
#: **What that means**: how much confound structure survives depends
#: heavily on **WHICH PERMUTATION WAS DRAWN**. 60.2% is therefore a
#: **mean over a highly variable quantity, not a stable property of the
#: design** -- and a design whose confound-preservation varies by a
#: factor of six across seeds cannot support a precise partition. This
#: is why ``between`` says bounded rather than resolved, and it is the
#: reason the bound is wide rather than an incidental detail of it.
DIAGNOSTIC_OBSERVED = {
    "observed": "2026-08-31, run p20_stratification_diagnostic__77ea2247",
    "cell_fired": "DIAGNOSTIC_READINGS_COMMITTED['between']",
    "the_figure": (
        "the confound model on Arm S's OWN permuted labels retains 60.2% "
        "of the +0.1061 ceiling (mean +0.0639, sd 0.0399), between the "
        "declared 30% and 70%"
    ),
    "what_the_cell_says": (
        "PARTIAL SURVIVAL: the fraction retained is reported and the "
        "partition is stated as BOUNDED RATHER THAN RESOLVED, with no "
        "point estimate promoted to an attribution"
    ),
    "permutation_identity_check": (
        "**PASSED.** The diagnostic's permutation.json equals Arm S's "
        "EXACTLY -- the diff, empty. Recorded as passed BY A "
        "PERSON'S DIFF, not inferred from the code being deterministic. "
        "Without it the diagnostic's number would say nothing about Arm S"
    ),
    "the_per_seed_dispersion_is_its_own_finding": (
        "**and it is larger than the headline.** Retained per seed: "
        "122.0% / 54.3% / 49.3% / 19.2% / 56.2% -- a spread of 37.6 "
        "percentage points around a 60.2% mean. ONE SEED RETAINED MORE "
        "CONFOUND SIGNAL THAN THE UNPERMUTED CEILING (122%); another "
        "barely a fifth (19.2%)"
    ),
    "what_the_dispersion_means": (
        "how much confound structure survives depends heavily on WHICH "
        "PERMUTATION WAS DRAWN. 60.2% is a MEAN OVER A HIGHLY VARIABLE "
        "QUANTITY, NOT A STABLE PROPERTY OF THE DESIGN -- and a design "
        "whose confound-preservation varies by a factor of six across "
        "seeds cannot support a precise partition. This is WHY the bound "
        "is wide, not an incidental detail of it"
    ),
    "it_does_not_lift_the_binding": (
        "DIAGNOSTIC_BINDING applies on this branch as on every other: S "
        "is partially interpretable, which is not the same as S being an "
        "answer. See RESIDUAL_PROHIBITION"
    ),
}


#: **[OBSERVED 2026-08-31 -- honesty over retrofit] ARM S'S PATTERN. No
#: committed cell fired.**
#:
#: Recorded on ``phase17.UNPREDICTED_PATTERN``'s precedent and the maintainer's
#: instruction: *record the observed pattern beside the unfired cells; do
#: not stretch a cell to fit.*
#:
#: **Why none of the four fires.** ``s_near_the_confound_ceiling``
#: expected S near +0.1061 -- it is +0.0414, less than half.
#: ``s_near_0_2520`` expected the headline to be largely confound
#: structure -- it is not. ``s_between`` was written for a value
#: **between the ceiling and 0.2520**, and +0.0414 is BELOW the ceiling,
#: outside the interval that cell describes. The cells stand
#: **PRESERVED AND UNFIRED in READINGS_COMMITTED, byte-unchanged.**
#:
#: **The observed pattern, MEASURED -- and it does not match the summary
#: phrasing the registration arrived with.** S = **+0.0414** sits
#: **0.0647 from the ceiling** (C, +0.1061) and **0.0868 from the null**
#: (P, -0.0454): **closer to the CEILING**, at **39.0% of the ceiling**
#: and **57.3% of the way from P to C**. Its 95% interval
#: [-0.0347, +0.1610] **contains zero** and **contains the ceiling**, but
#: does **NOT** contain P's point estimate, which sits **0.0107 below the
#: lower bound**.
#:
#: **What holds on every metric**: S is **not distinguishable from
#: either**. The interval is **0.1957 wide** on a cohort that
#: ``ladder.COHORT_CANNOT_RESOLVE`` says cannot resolve differences of
#: 0.04 to 0.10 between arms at all. Whether S and P are distinguishable
#: **cannot be settled from this record** -- that needs Arm P's own
#: spread, which is not held here.
#:
#: **See ``divergence_from_the_phrasing_2026_08_31``**: the instruction
#: described S as *"near the null, not near the confound ceiling"* with
#: an interval that *"spans both"*. Measured, the positional claim goes
#: the OTHER WAY and the interval misses P. Recorded as measured, because
#: *record the observed pattern* means the pattern the numbers show.
#: **The diagnostic is unaffected** -- the within-bin question is real
#: wherever S sits.
#:
#: **What this is and is not.** An OBSERVATION, not a registered reading.
#: The cells are not amended: a reading written after the number is not a
#: reading. And the observation is not itself an answer -- what it raises
#: is a DESIGN question that S's own number cannot settle, which is why
#: ``STRATIFICATION_DIAGNOSTIC`` exists.
S_PATTERN_UNPREDICTED = {
    "observed": (
        "2026-08-31, the instruction: record the observed pattern "
        "beside the unfired cells, do not stretch a cell to fit"
    ),
    "no_cell_fired": (
        "**none of the four committed S cells describes the outcome.** "
        "s_near_the_confound_ceiling expected ~+0.1061 and S is +0.0414; "
        "s_near_0_2520 expected the headline to be largely confound "
        "structure and it is not; s_between was written for a value "
        "BETWEEN the ceiling and 0.2520, and +0.0414 is BELOW the "
        "ceiling -- outside the interval that cell describes. The cells "
        "stand PRESERVED AND UNFIRED in READINGS_COMMITTED, "
        "byte-unchanged"
    ),
    "the_observed_pattern": (
        "MEASURED: S = +0.0414 sits 0.0647 FROM THE CEILING (Arm C, "
        "+0.1061) and 0.0868 FROM THE NULL (Arm P, -0.0454) -- CLOSER TO "
        "THE CEILING, at 39.0% of the ceiling and 57.3% of the way from "
        "P to C. Its 95% interval [-0.0347, +0.1610] CONTAINS ZERO and "
        "CONTAINS THE CEILING but does NOT contain P's point estimate, "
        "which sits 0.0107 below the lower bound. What holds on every "
        "metric: S IS NOT DISTINGUISHABLE FROM EITHER -- the interval is "
        "0.1957 wide on a cohort ladder.COHORT_CANNOT_RESOLVE says "
        "cannot resolve 0.04 to 0.10 between arms at all"
    ),
    # [OPEN 2026-08-31, at the observation] "whether S and P are
    # DISTINGUISHABLE cannot be settled here: that needs Arm P's own
    # spread, which this record does not hold -- ARMS_OBSERVED carried
    # P's mean only." Preserved as written; RESOLVED below at the
    # close-out, when P's spread arrived.
    "what_this_record_cannot_settle": (
        "[OPEN 2026-08-31, RESOLVED the same day at the close-out -- see "
        "s_and_p_are_not_distinguishable_2026_08_31] whether S and P are "
        "DISTINGUISHABLE. That needs Arm P's own spread, which this "
        "record did not hold at the time of the observation: "
        "ARMS_OBSERVED carried P's mean only"
    ),
    "s_and_p_are_not_distinguishable_2026_08_31": (
        "**RESOLVED at the close-out, when Arm P's spread arrived.** "
        "S's 95% interval [-0.0347, +0.1610] and P's [-0.1779, +0.0717] "
        "OVERLAP on [-0.0347, +0.0717] -- a 0.1064-wide common region, "
        "more than half of S's own interval. S is NOT distinguishable "
        "from the plain permutation. **Stated with its limit**: this is "
        "an OVERLAP OF MARGINAL INTERVALS, not a paired test. The "
        "per-patient paired comparison PLAN 4.3 would use was "
        "deliberately not run here (CONTRAST_MACHINERY, LEDGER_RULED), "
        "so the finding is 'the intervals do not separate', which is "
        "weaker than 'a paired test found no difference' and is not "
        "quoted as the stronger one"
    ),
    # -----------------------------------------------------------------
    # [MEASURED 2026-08-31, recorded rather than smoothed]
    #
    # The registration instruction described S as "near the null (P
    # -0.0454), not near the confound ceiling (C +0.1061)" with an
    # interval that "spans both". Measured against the figures supplied
    # in the same sentence, BOTH descriptive claims go the other way:
    #
    #   |S - C| = 0.0647  <  |S - P| = 0.0868   -> nearer the CEILING
    #   interval contains C (+0.1061)           -> yes
    #   interval contains P (-0.0454)           -> NO, by 0.0107
    #
    # Recorded as measured. "Record the observed pattern" means the
    # pattern the numbers show, and encoding a description the numbers
    # contradict would be the opposite of what the instruction asked for.
    #
    # ONE SENSE OF THE PHRASING SURVIVES AND IS WORTH KEEPING: S's
    # interval CONTAINS ZERO, so "S's near-zero result" -- the phrase the
    # diagnostic's own cells use -- is true in the sense of NOT
    # DISTINGUISHABLE FROM ZERO, even though S's point estimate is
    # nearer the ceiling than the null. The cells are not weakened by
    # this note.
    #
    # THE DIAGNOSTIC IS UNAFFECTED. The within-bin question is real
    # wherever S sits, and this changes nothing about what it measures
    # or what its three cells say.
    # -----------------------------------------------------------------
    "divergence_from_the_phrasing_2026_08_31": (
        "the instruction described S as 'near the null, not near the "
        "confound ceiling' with an interval that 'spans both'. Measured "
        "against the figures supplied in the same sentence, BOTH claims "
        "go the other way: |S-C| = 0.0647 < |S-P| = 0.0868, and the "
        "interval contains C but misses P by 0.0107. Recorded as "
        "measured. ONE SENSE SURVIVES: the interval CONTAINS ZERO, so "
        "'S's near-zero result' -- the diagnostic cells' own phrase -- is "
        "true in the sense of NOT DISTINGUISHABLE FROM ZERO, and the "
        "cells are not weakened. THE DIAGNOSTIC IS UNAFFECTED: the "
        "within-bin question is real wherever S sits"
    ),
    "what_this_is_and_is_not": (
        "an OBSERVATION, not a registered reading. The cells are not "
        "amended, per the nothing-retrofitted rule: a reading written "
        "after the number is not a reading. And the observation is not "
        "itself an answer -- what it raises is a DESIGN question S's own "
        "number cannot settle (STRATIFICATION_DIAGNOSTIC)"
    ),
}


#: **[RECORDED 2026-08-31] A LIMITATION OF THE LOCK, recorded as the lock
#: itself requires -- NOT folded into it.**
#:
#: ``EXIT_CRITERIA`` says: *"nothing is added after this record; a
#: criterion discovered missing later is a limitation of the lock,
#: recorded as such, never a retro-fitted entry."* One was discovered
#: missing. This is that record.
#:
#: **The gap.** The nine criteria require Arm S to report its
#: degenerate-stratum frequency and its fixed-point counts -- both of
#: which describe **how much the permutation moved**. **Not one of them
#: checks whether the stratification PRESERVED THE THING IT EXISTS TO
#: PRESERVE.** Stratifying on the confound model's fitted value into 10
#: quantile bins preserves confound structure **only between bins**. If
#: the five statistics' predictive power lives substantially WITHIN bin
#: at k = 10, S destroyed the confound signal along with the cleft
#: signal, and S measures **both-destroyed** rather than
#: **cleft-destroyed**.
#:
#: **Why this matters more than a missing check usually does**: without
#: it, S's near-null result reads as *"confounds contribute nothing"*
#: when it may mean *"the design could not tell"*. Those are opposite
#: conclusions from the same number, and the criteria could not separate
#: them.
#:
#: **The lock is NOT reopened.** ``EXIT_CRITERIA["criteria"]`` keeps its
#: nine, byte-unchanged. The diagnostic is registered as a NEW
#: measurement (``STRATIFICATION_DIAGNOSTIC``) with its own readings
#: committed before it runs -- not as a tenth criterion. The lock landed
#: the same day this gap was found, which sharpens rather than softens
#: the point: **a lock is not made valid by the calendar.**
LOCK_LIMITATION_STRATIFICATION_UNVERIFIED = {
    "recorded": "2026-08-31, as EXIT_CRITERIA's own clause requires",
    "the_gap": (
        "the nine criteria require Arm S's degenerate-stratum frequency "
        "and fixed-point counts -- both describing HOW MUCH THE "
        "PERMUTATION MOVED. NOT ONE checks whether the stratification "
        "PRESERVED WHAT IT EXISTS TO PRESERVE. Stratifying on the "
        "fitted value into 10 quantile bins preserves confound structure "
        "ONLY BETWEEN BINS; if the five statistics' predictive power "
        "lives substantially WITHIN bin at k = 10, S destroyed the "
        "confound signal along with the cleft signal and measures "
        "BOTH-DESTROYED rather than CLEFT-DESTROYED"
    ),
    "why_it_matters": (
        "without the check, S's near-null result reads as 'confounds "
        "contribute nothing' when it may mean 'the design could not "
        "tell'. Those are OPPOSITE CONCLUSIONS FROM THE SAME NUMBER, and "
        "the criteria could not separate them"
    ),
    "the_lock_is_not_reopened": (
        "EXIT_CRITERIA['criteria'] keeps its nine, byte-unchanged. The "
        "diagnostic is a NEW REGISTERED MEASUREMENT with its own "
        "readings committed before it runs -- not a tenth criterion. The "
        "lock landed the same day this gap was found, which sharpens "
        "rather than softens the point: A LOCK IS NOT MADE VALID BY THE "
        "CALENDAR"
    ),
}


#: **[REGISTERED 2026-08-31 -- READINGS COMMITTED BEFORE IT RUNS] THE
#: STRATIFICATION DIAGNOSTIC.**
#:
#: **The question, which S's own number cannot answer.** Did the k = 10
#: stratification preserve the confound structure it exists to preserve?
#: The answer decides whether S measured **cleft-destroyed** (the arm's
#: purpose) or **both-destroyed** (a design that could not tell).
#:
#: **The measurement.** Fit the five-statistic confound model against
#: **S's own permuted labels**, per seed, **same recipe as Arm C**:
#: closed-form ridge over ``cleft_v1``'s own folds, inner-val split per
#: seed, alpha 1.0, the same five statistics. Same strata, same
#: permutations, same five seeds -- the labels the diagnostic scores are
#: byte-identically the ones Arm S trained on.
#:
#: **Cheap by construction**: no new artifact, no new hash, no backbone.
#: It reads the same manifest and staged pixels Arm S read.
#:
#: **The floor and the ceiling are both known in advance**, which is what
#: makes the fraction meaningful: a plain permutation destroys everything
#: (floor 0 by construction), and the intact confound signal is Arm C's
#: **+0.1061** (``ARMS_OBSERVED``). The diagnostic reports **the fraction
#: of the ceiling retained**, per seed and in the mean.
STRATIFICATION_DIAGNOSTIC = {
    "registered": "2026-08-31, readings committed BEFORE it runs",
    "the_question": (
        "did the k = 10 stratification preserve the confound structure "
        "it exists to preserve? The answer decides whether Arm S "
        "measured CLEFT-DESTROYED (the arm's purpose) or BOTH-DESTROYED "
        "(a design that could not tell)"
    ),
    "why_s_cannot_answer_it": (
        "S's number is compatible with both. A near-null S means "
        "'confounds contribute nothing' if the confounds survived the "
        "permutation and 'the design could not tell' if they did not, "
        "and nothing in S distinguishes the two"
    ),
    "the_measurement": (
        "fit the five-statistic confound model against S'S OWN PERMUTED "
        "LABELS, per seed, SAME RECIPE AS ARM C: closed-form ridge over "
        "cleft_v1's own folds, inner-val split per seed, alpha 1.0, the "
        "same five statistics. Same strata, same permutations, same five "
        "seeds -- the labels scored are byte-identically the ones Arm S "
        "trained on"
    ),
    "cheap_by_construction": (
        "no new artifact, no new hash, no backbone -- the same manifest "
        "and staged pixels Arm S read"
    ),
    "floor_and_ceiling_known_in_advance": (
        "a plain permutation destroys everything, so the floor is 0 BY "
        "CONSTRUCTION; the intact confound signal is Arm C's +0.1061 "
        "(ARMS_OBSERVED). The reported quantity is THE FRACTION OF THE "
        "CEILING RETAINED, per seed and in the mean"
    ),
}


#: **[COMMITTED 2026-08-31, BEFORE ANY NUMBER] THE DIAGNOSTIC'S THREE
#: READINGS.**
#:
#: The thresholds live in the CONFIG (``survived_fraction``,
#: ``destroyed_fraction``), declared before the run, so the cell that
#: fires cannot be chosen after the number is seen.
DIAGNOSTIC_READINGS_COMMITTED = {
    "committed": "2026-08-31, before any number exists",

    "near_the_ceiling": (
        "**confound structure SURVIVED the stratified permutation.** "
        "S's near-zero result is genuine SIGNAL-DESTRUCTION, and the "
        "partition reads: THE CONFOUND CONTRIBUTION IS NOT RECOVERABLE "
        "BY THE PROBE WHEN CLEFT STRUCTURE IS ABSENT. The residual "
        "arithmetic then applies -- subject to THE BINDING below, which "
        "no branch lifts"
    ),

    "near_zero": (
        "**the stratification destroyed the confounds too.** Arm S "
        "CANNOT ANSWER THE QUESTION IT WAS BUILT FOR and is recorded as "
        "UNINFORMATIVE BY CONSTRUCTION. The design flaw is NAMED -- "
        "10 quantile bins preserve confound structure only BETWEEN "
        "bins, and the five statistics' predictive power lived "
        "substantially WITHIN bin. What a corrected design would need is "
        "STATED WITHOUT BEING BUILT: finer strata, or MATCHING rather "
        "than binning. **The residual 0.2520 - 0.0414 = 0.2106 IS NOT TO "
        "BE QUOTED AS CLEFT-SPECIFIC UNDER THIS BRANCH**"
    ),

    "between": (
        "**partial survival.** The fraction retained is REPORTED, and "
        "the partition is stated as BOUNDED RATHER THAN RESOLVED -- with "
        "the fraction given, its per-seed spread attached, and no point "
        "estimate promoted to an attribution"
    ),

    "no_reading_is_invented_after": (
        "whichever lands, no reading is added once the number exists; a "
        "pattern outside these three gets a dated OBSERVATION, on "
        "phase17.UNPREDICTED_PATTERN's precedent -- which is exactly what "
        "Arm S itself required (S_PATTERN_UNPREDICTED)"
    ),

    "the_thresholds_are_declared_in_the_config": (
        "survived_fraction and destroyed_fraction are config fields, "
        "declared before the run, so the cell that fires cannot be "
        "chosen after the number is seen -- the substantial_pcc pattern"
    ),
}


#: **[BOUND 2026-08-31, BEFORE THE DIAGNOSTIC RUNS] WHAT NO BRANCH
#: LICENSES.**
#:
#: **Under no branch does Arm S's number alone license a claim about how
#: much of 0.2520 is cleft-specific.** The diagnostic decides whether S
#: is INTERPRETABLE AT ALL -- it does not convert S into an attribution.
#:
#: Written down before the number because the tempting move, on the
#: favourable branch, is to treat "S is interpretable" as "S is the
#: answer". It is not. Even at full confound survival, S is one arm at
#: +0.0414 with **sd 0.0802 and a 95% interval spanning both the null
#: and the ceiling** -- and a residual computed against it inherits that
#: spread. ``ladder.SMALLEST_RESOLVABLE_DIFFERENCE`` still governs:
#: **this cohort cannot resolve PCC differences of 0.04 to 0.10 between
#: arms**, and the smallest difference it has ever resolved is 0.1386.
DIAGNOSTIC_BINDING = {
    "bound": "2026-08-31, before the diagnostic runs",
    "the_binding": (
        "**under NO BRANCH does Arm S's number alone license a claim "
        "about how much of 0.2520 is cleft-specific.** The diagnostic "
        "decides whether S is INTERPRETABLE AT ALL; it does not convert "
        "S into an attribution"
    ),
    "why_it_is_written_before_the_number": (
        "the tempting move on the favourable branch is to treat 'S is "
        "interpretable' as 'S is the answer'. It is not"
    ),
    "what_stands_in_the_way_even_then": (
        "even at full confound survival, S is ONE ARM at +0.0414 with sd "
        "0.0802 and a 95% interval spanning BOTH the null and the "
        "ceiling; a residual computed against it inherits that spread. "
        "ladder.SMALLEST_RESOLVABLE_DIFFERENCE still governs: this "
        "cohort cannot resolve PCC differences of 0.04 to 0.10 between "
        "arms, and the smallest it has ever resolved is 0.1386"
    ),
}


#: **[REGISTERED 2026-08-31] THE RESIDUAL PROHIBITION -- a tested
#: literal, sibling to ``ladder.DETECTION_FLOOR_PROHIBITION`` and
#: ``phase10_annex.ANNEX_PROHIBITION``.**
#:
#: The arithmetic 0.2520 - 0.0414 = 0.2106 is correct and the sentence it
#: tempts is not. It rides with any quotation of Arm S.
RESIDUAL_PROHIBITION = (
    "THE RESIDUAL 0.2520 - 0.0414 = 0.2106 IS NOT CLEFT-SPECIFIC AND IS "
    "NEVER QUOTED AS SUCH. Arm S's number, under NO branch of the "
    "stratification diagnostic, licenses a claim about how much of "
    "0.2520 is cleft-specific. The diagnostic decided only whether Arm S "
    "is INTERPRETABLE, and it came back PARTIAL: 60.2% of the confound "
    "ceiling survives the stratified permutation, with a per-seed spread "
    "of 37.6 percentage points -- one seed above the unpermuted ceiling "
    "and one at a fifth of it -- so confound structure survives only "
    "PARTIALLY AND VARIABLY and the partition is BOUNDED, NOT RESOLVED. "
    "Three further facts stand in the way of the subtraction and each is "
    "sufficient alone: Arm S is ONE ARM at +0.0414 with sd 0.0802 whose "
    "95% interval [-0.0347, +0.1610] contains BOTH zero and the +0.1061 "
    "ceiling; a residual computed against it inherits that spread; and "
    "SMALLEST_RESOLVABLE_DIFFERENCE governs -- this cohort cannot "
    "resolve PCC differences of 0.04 to 0.10 between arms, and the "
    "smallest it has ever resolved is 0.1386. No phase is gated on any "
    "of this."
)


#: **[MEASURED + REASONED 2026-08-31] SEED STABILITY AS AN INDEPENDENT
#: LINE OF EVIDENCE THAT 0.2520 IS REAL.**
#:
#: **New, and not previously anywhere as a claim.**
#:
#: **The measurement [MEASURED]** -- seed SD falls **monotonically** as
#: real structure is added to the labels:
#:
#:     Arm P   scrambled labels      sd 0.1109   p20_permutation_plain__dc4605bf
#:     Arm S   stratified shuffle    sd 0.0802   p20_permutation_stratified__dc4605bf
#:     Arm C   confounds only        sd 0.0222   p20_confound_ceiling_mean__dc4605bf
#:     probe   real labels           sd 0.0148   ladder.TRADE_OFF_PAIR['result']['vit_sd']
#:
#: A model on scrambled labels varies **7.5x more across seeds** than one
#: on real labels (0.1109 / 0.0148 = 7.49).
#:
#: **The mechanism [REASONED, and tagged as such]**: with no signal to
#: fit, each seed's head chases a different accident of its own
#: inner-validation split, so the arms disagree; with signal, every seed
#: finds the same structure and they converge. **This is an argument, not
#: a measurement.** What is measured is the ordering; that this ordering
#: is CAUSED by signal availability is inference, and the four sds would
#: be equally consistent with some other mechanism nobody has ruled out.
#:
#: **Why it is worth having**: it rests on **STABILITY rather than
#: MAGNITUDE**, so it does **not depend on comparing 0.2520 against any
#: ceiling** -- not the confound ceiling, not the null, not a resolution
#: floor. The magnitude argument and this one can fail independently,
#: which is what makes them two lines rather than one restated.
#:
#: **What it does NOT license**: nothing about SIZE. A stable small
#: number is still small, and this says only that the probe's number
#: behaves like a model fitting something rather than noise.
#: ``RESIDUAL_PROHIBITION`` is untouched by it.
SEED_STABILITY_EVIDENCE = {
    "recorded": "2026-08-31 -- new, not previously anywhere as a claim",
    "the_measurement": "[MEASURED] seed SD falls monotonically as real "
                       "structure is added to the labels",
    "sds_by_arm": {
        "arm_p_scrambled": {
            "sd": 0.1109, "run": "p20_permutation_plain__dc4605bf",
            "labels": "scrambled across all 237",
        },
        "arm_s_stratified": {
            "sd": 0.0802, "run": "p20_permutation_stratified__dc4605bf",
            "labels": "shuffled within 10 confound strata",
        },
        "arm_c_confounds": {
            "sd": 0.0222, "run": "p20_confound_ceiling_mean__dc4605bf",
            "labels": "real, but only five image statistics to fit them",
        },
        "probe_real": {
            "sd": 0.0148,
            "run": "p7_d1_vit_b16_imagenet_g1 -- banked at "
                   "ladder.TRADE_OFF_PAIR['result']['vit_sd']",
            "labels": "real",
        },
    },
    "the_ratio": "[MEASURED] 0.1109 / 0.0148 = 7.49x -- a model on "
                 "scrambled labels varies 7.5x more across seeds than "
                 "one on real labels",
    "the_mechanism": (
        "[REASONED -- an argument, not a measurement] with no signal to "
        "fit, each seed's head chases a different accident of its own "
        "inner-validation split, so the arms disagree; with signal every "
        "seed finds the same structure and they converge. What is "
        "MEASURED is the ORDERING; that the ordering is CAUSED by signal "
        "availability is INFERENCE, and the four sds would be equally "
        "consistent with some other mechanism nobody has ruled out"
    ),
    "why_it_is_a_second_line": (
        "it rests on STABILITY rather than MAGNITUDE, so it does NOT "
        "depend on comparing 0.2520 against any ceiling -- not the "
        "confound ceiling, not the null, not a resolution floor. The two "
        "arguments can fail independently, which is what makes them two "
        "lines rather than one restated"
    ),
    "what_it_does_not_license": (
        "**nothing about SIZE.** A stable small number is still small. "
        "This says only that the probe's number behaves like a model "
        "fitting something rather than noise. RESIDUAL_PROHIBITION is "
        "untouched by it"
    ),
    "caveat_on_the_comparison": (
        "the four arms are not one experiment: Arm C is a five-feature "
        "ridge and the other three are 769-parameter heads over ViT "
        "embeddings, so the C-to-probe step changes the MODEL as well as "
        "the labels. The monotonicity across all four is the observation; "
        "the clean within-design comparison is P -> S -> probe, which is "
        "monotone on its own (0.1109 > 0.0802 > 0.0148)"
    ),
}


#: **[RECORDED 2026-08-31, DATED] WHERE THE WRONG DESCRIPTION OF ARM S
#: CAME FROM.** Named on the same footing as
#: ``CROSS_TARGET_ERROR_PROVENANCE`` and
#: ``ladder.THE_ERROR_PROVENANCE``.
#:
#: **The error.** Arm S was described as *"near the null, not near the
#: ceiling, its interval spans both"*. Measured against the figures
#: supplied in the same sentence, **both halves are false**:
#: |S - C| = 0.0647 < |S - P| = 0.0868, so S is nearer the CEILING; and
#: S's interval **excludes P by 0.0107**. S sits at **39.0% of the
#: ceiling** and **57.3% of the way from P to C**.
#:
#: **What survives**: S's interval **contains zero**, so "near-zero" in
#: the sense of *not distinguishable from zero* is true, and the
#: diagnostic's cells -- which use that phrase -- are not weakened.
#:
#: **Where it came from**: **conversation -- the record's, this session.** It
#: was written into the first draft of ``S_PATTERN_UNPREDICTED`` and into
#: a test asserting it. **The test failed**, which is how it was caught:
#: the assertion was written against the arithmetic rather than the
#: sentence, and the arithmetic refused it.
#:
#: **What it touched**: nothing measured, and no reading. It was a
#: DESCRIPTION of a result, corrected before the close-out; the cells,
#: the diagnostic and every figure are unaffected.
S_DESCRIPTION_ERROR_PROVENANCE = {
    "recorded": "2026-08-31",
    "the_error": (
        "Arm S described as 'near the null, not near the ceiling, its "
        "interval spans both'. Measured: |S-C| = 0.0647 < |S-P| = "
        "0.0868, so S is nearer the CEILING, and S's interval EXCLUDES "
        "P by 0.0107. S sits at 39.0% of the ceiling and 57.3% of the "
        "way from P to C. BOTH HALVES FALSE"
    ),
    "what_survives": (
        "S's interval CONTAINS ZERO, so 'near-zero' in the sense of NOT "
        "DISTINGUISHABLE FROM ZERO is true, and the diagnostic's cells -- "
        "which use that phrase -- are not weakened"
    ),
    "origin": (
        "CONVERSATION -- the record's, this session. Written into the first "
        "draft of S_PATTERN_UNPREDICTED and into a test asserting it"
    ),
    "how_it_was_caught": (
        "**THE TEST FAILED.** The assertion was written against the "
        "ARITHMETIC rather than the sentence, and the arithmetic refused "
        "it. The record moved to the numbers; the numbers were not "
        "moved to the record"
    ),
    "what_it_touched": (
        "nothing measured, and no reading. A DESCRIPTION of a result, "
        "corrected before the close-out; the cells, the diagnostic and "
        "every figure are unaffected"
    ),
}


# --------------------------------------------------------------------------
# The machinery. Four functions, and no fifth: every permutation in this
# project is drawn here, and `run` calls it in exactly one place.
# --------------------------------------------------------------------------


def quantile_strata(values, k: int) -> np.ndarray:
    """``k`` quantile bins of ``values``, as a stratum id per row.

    **Ties are never split.** The bin is a function of the VALUE alone
    (``searchsorted`` on the interior quantile edges), so two patients
    with the same fitted confound value always share a stratum. The
    alternative -- splitting a tie to even the bin sizes -- would treat
    two identically-confounded patients as differently confounded, which
    is precisely the structure stratification exists to preserve.

    ``k = 1`` returns all zeros, which is Arm P: **the plain arm is the
    stratified arm at one stratum**, not a second implementation
    (``ARM_P_REGISTERED``, ``STRATIFICATION_RULED``).
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"expected a 1-D array of values, got {array.shape}")
    if k < 1:
        raise ValueError(f"k_strata must be >= 1, got {k!r}")
    if k == 1:
        return np.zeros(len(array), dtype=int)
    edges = np.quantile(array, np.linspace(0.0, 1.0, k + 1)[1:-1])
    return np.searchsorted(edges, array, side="right").astype(int)


def permute_within_strata(strata, seed: int) -> np.ndarray:
    """An index permutation that never crosses a stratum boundary.

    Returns ``perm`` such that ``labels[perm]`` is the permuted label
    vector and ``strata[perm] == strata`` everywhere. Drawn from
    ``np.random.default_rng(seed)`` over strata in sorted id order, so
    the draw is reproducible from the seed alone.

    **A singleton stratum is a fixed point at every seed.** That is not
    a defect to be redrawn around -- see ``degenerate_strata``.
    """
    labels = np.asarray(strata, dtype=int)
    rng = np.random.default_rng(int(seed))
    perm = np.arange(len(labels), dtype=int)
    for stratum in np.unique(labels):
        rows = np.flatnonzero(labels == stratum)
        perm[rows] = rows[rng.permutation(len(rows))]
    return perm


def fixed_points(permutation) -> int:
    """How many rows kept their own label.

    Pre-registered arithmetic (``STRATIFICATION_RULED``): the expectation
    is **1 per stratum for any stratum size**, so k strata expect k. The
    count is REPORTED, never used to accept or reject a draw.
    """
    array = np.asarray(permutation, dtype=int)
    return int(np.sum(array == np.arange(len(array))))


def degenerate_strata(strata) -> dict:
    """Which strata cannot move, and the policy that governs them.

    A stratum of size 1 has exactly one permutation -- the identity -- so
    its patient trains against its own label in every seed. This is
    **counted and reported, never redrawn**: redrawing to avoid fixed
    points would make the permutation non-uniform and quietly bias the
    control toward zero, which is the direction that would make the arm
    look good.
    """
    labels = np.asarray(strata, dtype=int)
    ids, sizes = np.unique(labels, return_counts=True)
    singletons = [int(i) for i, size in zip(ids, sizes) if size == 1]
    return {
        "n_strata": int(len(ids)),
        "sizes": {int(i): int(size) for i, size in zip(ids, sizes)},
        "smallest": int(sizes.min()) if len(sizes) else 0,
        "singleton_strata": singletons,
        "n_immovable": len(singletons),
        "policy": (
            "counted and reported, NEVER REDRAWN: redrawing to avoid "
            "fixed points would make the permutation non-uniform and "
            "bias the control toward zero -- the direction that would "
            "make the arm look good"
        ),
    }


#: **[DECIDED 2026-08-31] THE FIFTH SEQUENCE AMENDMENT -- and
#: the first that RENUMBERS NOTHING.**
#:
#: **Why it is needed, verified at source.**
#: ``phase15.PHASE_SEQUENCE_RENUMBERED_4["becomes"]`` maps 16, 17, 18 and
#: **19 = write-up**. It stops there. **The chain is SILENT on 20**, and
#: no fifth amendment existed.
#:
#: **Where 20 came from.** The registration message opened
#: *"Phase 20: the permutation control"*, so the NUMBER was never
#: an invention. What was missing is the AMENDMENT: the standing
#: pattern is that every sequence change gets a record carrying the full
#: backward chain with dated pointers, and ``phase20.py`` was created
#: without one. **The number was authorised; the record was not
#: written.** This is that record.
#:
#: **It takes a different name deliberately.** The four before it are
#: ``PHASE_SEQUENCE_RENUMBERED_*`` because each MOVED an existing phase.
#: This one moves nothing -- 19 remains the write-up, and 20 is appended.
#: Calling it RENUMBERED would be R2's shape exactly: two different
#: quantities under one name.
#:
#: **The write-up is no longer the highest number, and that is
#: deliberate.** See ``the_write_up_is_last_by_rule_not_by_number``.
PHASE_SEQUENCE_EXTENDED_5 = {
    "decided": (
        "2026-08-31 -- the FIFTH sequence amendment, and "
        "the first that renumbers nothing"
    ),
    "why_the_name_differs": (
        "the four before it are PHASE_SEQUENCE_RENUMBERED_* because each "
        "MOVED an existing phase. This one moves nothing: 19 remains the "
        "write-up and 20 is APPENDED. Calling it RENUMBERED would be "
        "R2's shape -- two different quantities under one name"
    ),

    # ---- the full backward chain, carried ----------------------------
    "first_amendment": "phase11.PHASE_SEQUENCE_RENUMBERED (view ablation to 12)",
    "second_amendment": "phase12.PHASE_SEQUENCE_RENUMBERED_2 (LDL to 14, "
                        "second beauty dataset to 15, TSTR to 16)",
    "third_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_3 (metric-space "
                       "ablation to 16, TSTR to 17, write-up to 18)",
    "fourth_amendment": "phase15.PHASE_SEQUENCE_RENUMBERED_4 (anchor loop "
                        "promoted to 16, metric-space ablation to 18, "
                        "write-up to 19)",

    "was": {
        "19": "write-up",
        "20": "**NOTHING -- the chain stopped at 19**",
    },
    "becomes": {
        "19": "write-up (UNCHANGED in place and content)",
        "20": "THE PERMUTATION CONTROL -- Arms P, S and C plus the "
              "stratification diagnostic; opened, locked and closed "
              "2026-08-31",
    },
    "status_changes": {
        "write_up": "Phase 19 -> Phase 19 (unchanged)",
        "permutation_control": "unnumbered -> PHASE 20",
    },

    "what_was_verified_at_source": (
        "phase15.PHASE_SEQUENCE_RENUMBERED_4['becomes'] maps 16/17/18/19 "
        "and STOPS. No key for 20 exists in any of the four amendments; "
        "no fifth amendment existed. The gap is real and this closes it"
    ),
    "whose_say_so_the_number_was": (
        "**The registration message** opened 'Phase 20: "
        "the permutation control', so the NUMBER was never the record's "
        "invention. What was missing is the AMENDMENT RECORD the "
        "standing pattern requires. The number was authorised; the "
        "record was not written"
    ),

    # [2026-08-31] A SIXTH amendment APPENDS Phase 21, renumbering
    # nothing -- the first written UNDER this record's
    # write-up-runs-last rule rather than establishing it, and the test
    # of it: it forced no renumber where the old assumption would have.
    "sixth_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_6",
    # [2026-09-01] A SEVENTH amendment schedules 22 (ranking and
    # pairwise losses), 23 (the statistical instruments), 24 (all five
    # raters) and 25 (foundation-model features) as
    # SCHEDULED-NOT-REGISTERED. It renumbers nothing, and leaves the
    # write-up's number OPEN rather than resolving it
    # (phase21.WRITE_UP_NUMBER_OPEN).
    "seventh_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_7",
    "module_names_never_change": (
        "the standing pattern, unchanged here: a module file is named "
        "for the number its phase HOLDS, and a record KEY keeps the "
        "number it was WRITTEN with -- which is why "
        "phase15.PHASE_16_SCHEDULED describes what is now Phase 18. "
        "Nothing is renamed by this amendment because nothing moved"
    ),
    "nothing_silently_renumbered": (
        "vacuously true here -- no phase moved -- and stated anyway, "
        "because an amendment that says nothing about it reads as one "
        "that forgot to check"
    ),

    # ---- the recommendation, made rather than deferred ---------------
    "the_write_up_is_last_by_rule_not_by_number": (
        "**19 < 20 means the write-up is no longer the highest number, "
        "and a RENUMBER WAS CONSIDERED AND REJECTED.** Moving the "
        "permutation control to 19 and the write-up to 20 would restore "
        "the ordering ONCE -- and the next measurement phase would break "
        "it again, so the write-up would move on every future addition, "
        "unboundedly. It has already moved twice (18 -> 19 across two "
        "amendments) without ever running. The stable rule instead: "
        "**THE WRITE-UP RUNS LAST REGARDLESS OF ITS NUMBER**, because it "
        "is defined as the phase that consumes all the others. Stated "
        "once here rather than re-established by renumbering forever. "
        "The cost is honest and small: a reader must not infer execution "
        "order from 19 vs 20"
    ),
    "eighth_amendment": (
        "phase25.PHASE_SEQUENCE_EXTENDED_8, 2026-09-05. Calibration "
        "ablation as 26, anchor set as a training set as 27, fine "
        "tuning as 28 with small backbones as an arm inside it. "
        "Renumbered nothing"
    ),
}


#: **[CLOSED 2026-08-31] PHASE 20. THE PERMUTATION CONTROL.**
#:
#: Four runs, four shas, all finalized single-attempt. **The phase closes
#: without growing**: nine locked criteria walked, one limitation carried
#: as a limitation, no ledger row, and nothing added to the lock.
#:
#: **THE HEADLINE, named for the write-up: THE GATE, NOT THE
#: PARTITION.**
#:
#: * **The pipeline invents nothing.** Scrambled labels give -0.0454 --
#:   no correlation from no information. Every banked PCC in the project
#:   is corroborated, and the four failure modes the committed cell named
#:   are ruled out together.
#: * **0.2520 is stable where a null is not.** Seed sd 0.0148 against Arm
#:   P's 0.1109 -- a 7.5x difference. An independent line of evidence
#:   resting on stability rather than magnitude
#:   (``SEED_STABILITY_EVIDENCE``).
#: * **The partition is BOUNDED, not resolved**, because confound
#:   structure survives stratification **only partially and variably**:
#:   60.2% retained, with a 37.6-point per-seed spread running from 19.2%
#:   to 122%.
#:
#: **What the phase does NOT deliver**: an attribution.
#: ``RESIDUAL_PROHIBITION`` rides with every quotation of Arm S.
PHASE_20_CLOSING = {
    "closed": (
        "2026-08-31 -- four runs, each at its own sha, all finalized "
        "single-attempt: p20_permutation_plain__dc4605bf, "
        "p20_permutation_stratified__dc4605bf, "
        "p20_confound_ceiling_mean__dc4605bf, "
        "p20_stratification_diagnostic__77ea2247"
    ),

    "criterion_1_arm_p": (
        "MET -- ARMS_OBSERVED['arm_p']. Five seeds "
        "(+0.0611/-0.1043/-0.0705/+0.0728/-0.1861), mean -0.0454, sd "
        "0.1109, 95% interval [-0.1779, +0.0717] containing zero; "
        "fixed points 1/1/1/1/2 against the pre-registered 1 (mean 1.2, "
        "z = +0.42, p = 0.828 against the shipped drawer's own sampling "
        "distribution). The per-seed BCa intervals are in the run's "
        "metrics.json and are NOT transcribed here"
    ),
    "criterion_2_the_gate_applied": (
        "MET -- ORDERING_BOUND honoured. P is near zero, the gate "
        "CLEARED, and Arm S ran. Had it failed, S would not have"
    ),
    "criterion_3_arm_s": (
        "MET -- ARMS_OBSERVED['arm_s']. Five seeds, mean +0.0414, sd "
        "0.0802, 95% interval [-0.0347, +0.1610], range -0.0393 to "
        "+0.1738. DEGENERATE-STRATUM FREQUENCY: strata "
        "24/24/23/24/23/24/24/23/24/24 -- **ZERO singleton strata**, so "
        "nothing was immovable by construction. Fixed points 7/11/17/10/13 "
        "against the pre-registered 10 (mean 11.6, z = +1.12, p = 0.283). "
        "The two quantities are reported SEPARATELY: a degenerate stratum "
        "and an unchanged label are different things"
    ),
    "criterion_4_floors_beside_every_figure": (
        "MET -- the reference point for the permutation arms is ZERO, "
        "stated: a permuted arm's floor is no-correlation, not a class "
        "baseline. Arm P's interval is read against zero and contains "
        "it; the diagnostic's floor is 0 by construction and its ceiling "
        "Arm C's +0.1061, which is what makes its fraction meaningful"
    ),
    "criterion_5_one_committed_reading_per_arm": (
        "MET, including where it required recording a MISS. P fired "
        "p_near_zero; C fired near_0_1000; the diagnostic fired "
        "between -- each applied VERBATIM. **Arm S fired NOTHING**, and "
        "the four cells stand PRESERVED AND UNAMENDED with the outcome "
        "recorded as a dated OBSERVATION (S_PATTERN_UNPREDICTED) on "
        "phase17.UNPREDICTED_PATTERN's precedent. No reading was "
        "invented after any number"
    ),
    "criterion_6_no_ledger_row": (
        "MET -- LEDGER_RULED. The ledger stands at 38 entries, unchanged "
        "by this phase; no claimable/unresolved verdict anywhere in it. "
        "All four measurements are DESCRIPTIVE"
    ),
    "criterion_7_target_discipline": (
        "MET -- every arm on the PANEL MEAN. Arm C establishes the "
        "like-for-like panel-mean ceiling at +0.1061, which is now the "
        "anchor for panel-mean comparisons; Phase 13's +0.1000 stays the "
        "median-target anchor and neither is quoted against the other's "
        "target (CONFOUND_CEILING_TARGET_MISMATCH)"
    ),
    "criterion_8_suite_green": "MET -- reported with the close-out",
    "criterion_9_arm_c": (
        "MET -- ARMS_OBSERVED['arm_c']. Five seeds "
        "(0.1336/0.0728/0.1143/0.1088/0.1010), mean +0.1061, sd 0.0222 "
        "against the PANEL MEAN over the same five statistics and "
        "recipe. Its committed reading near_0_1000 applied: +0.0061 from "
        "Phase 13's median-target figure against a 0.02 threshold "
        "declared before the number, so arm_c_moves_the_anchor did NOT "
        "trigger. Phase 13 carries its dated pointer "
        "(P1_CONFOUND_CEILING_BANKED) and IS NOT CORRECTED"
    ),

    "the_limitation_carried_as_one": (
        "LOCK_LIMITATION_STRATIFICATION_UNVERIFIED. The nine criteria "
        "never checked that the stratification preserved what it exists "
        "to preserve. It is carried AS A LIMITATION OF THE LOCK, not "
        "folded into it: EXIT_CRITERIA['criteria'] keeps its nine "
        "byte-unchanged, and the diagnostic that closed the gap is a "
        "separate registered measurement, not a tenth criterion. **The "
        "diagnostic's answer -- partial, variable survival -- is exactly "
        "why the limitation mattered**: without it, S's number would "
        "have read as an attribution"
    ),

    "the_headline_for_the_write_up": (
        "**THE GATE, NOT THE PARTITION.** (i) The pipeline INVENTS "
        "NOTHING -- scrambled labels give -0.0454, and every banked PCC "
        "in the project is corroborated. (ii) 0.2520 IS STABLE WHERE A "
        "NULL IS NOT -- sd 0.0148 against 0.1109, a 7.5x difference, an "
        "independent line resting on stability rather than magnitude. "
        "(iii) THE PARTITION IS BOUNDED, NOT RESOLVED -- confound "
        "structure survives stratification only PARTIALLY AND VARIABLY "
        "(60.2% retained, 37.6-point spread, 19.2% to 122%)"
    ),
    "what_the_phase_does_not_deliver": (
        "AN ATTRIBUTION. RESIDUAL_PROHIBITION rides with every quotation "
        "of Arm S, on every branch"
    ),
    "corrections_recorded_in_this_phase": (
        "two, both the record's and both named as such: "
        "CROSS_TARGET_ERROR_PROVENANCE (the median-vs-panel-mean "
        "comparison, caught at registration before any measurement "
        "rested on it) and S_DESCRIPTION_ERROR_PROVENANCE (Arm S's "
        "position, caught by a test written against the arithmetic "
        "rather than the sentence)"
    ),
    "the_phase_closed_without_growing": (
        "nine criteria walked, one limitation carried as a limitation, "
        "no ledger row, nothing added to the lock, and the fifth "
        "sequence amendment written for a number that had been "
        "authorised without one (PHASE_SEQUENCE_EXTENDED_5)"
    ),
}


def summary() -> dict:
    """The phase's registration, importable as one object."""
    return {
        "reckoning": PHASE_20_RECKONING,
        "target_mismatch": CONFOUND_CEILING_TARGET_MISMATCH,
        "arm_p": ARM_P_REGISTERED,
        "arm_s": ARM_S_PROPOSED,
        "ordering": ORDERING_BOUND,
        "readings": READINGS_COMMITTED,
        "contrast_machinery": CONTRAST_MACHINERY,
        "compute_gate": COMPUTE_GATE_DESIGNED,
        "exit_criteria_draft": EXIT_CRITERIA_DRAFT,
        # [2026-08-31] The rulings, the addition, and the lock.
        "stratification_ruled": STRATIFICATION_RULED,
        "ledger_ruled": LEDGER_RULED,
        "arm_c": ARM_C_REGISTERED,
        "cross_target_error_provenance": CROSS_TARGET_ERROR_PROVENANCE,
        "exit_criteria": EXIT_CRITERIA,
        # [2026-08-31, after the arms ran] What was observed, the cell
        # that did not fire, the gap the lock left, and the diagnostic
        # registered to close it.
        "arms_observed": ARMS_OBSERVED,
        "s_pattern_unpredicted": S_PATTERN_UNPREDICTED,
        "lock_limitation": LOCK_LIMITATION_STRATIFICATION_UNVERIFIED,
        "diagnostic": STRATIFICATION_DIAGNOSTIC,
        "diagnostic_readings": DIAGNOSTIC_READINGS_COMMITTED,
        "diagnostic_binding": DIAGNOSTIC_BINDING,
        # [2026-08-31] The close-out.
        "diagnostic_observed": DIAGNOSTIC_OBSERVED,
        "residual_prohibition": RESIDUAL_PROHIBITION,
        "seed_stability": SEED_STABILITY_EVIDENCE,
        "s_description_error_provenance": S_DESCRIPTION_ERROR_PROVENANCE,
        "sequence_extended_5": PHASE_SEQUENCE_EXTENDED_5,
        "closing": PHASE_20_CLOSING,
    }
