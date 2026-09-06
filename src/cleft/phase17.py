"""Phase 17 -- TSTR (train on synthetic, test on real).

Scheduled by the fourth sequence amendment
(``phase15.PHASE_SEQUENCE_RENUMBERED_4``: 17 = TSTR). Opens under the
2026-08-29 literature verification and its amendment on the parked
record (``scut.synthesis.PARKED["arm_amended_2026_08_29"]``): the parked
design is a TSTR design in the Rosero FAMILY, not Rosero's -- Rosero
warps by piecewise affine over landmark triangulation and trains binary
contrastive; the parked arm warps by TPS and trains on magnitude-ordered
labels. The word "replication" is withdrawn there and never used of arm
A here.

This module holds the rulings and the compute-gate measurement design.
**The restate -- exit criteria, per-arm readings, the completed
five-contrast readings, YAML settings -- waits for the measurement
result. No arm is built yet.**
"""

from __future__ import annotations

#: The project's pinned image, full digest on one line per the
#: workflow-hygiene rule (a truncated digest is a digest that cannot be
#: pasted). Same value as the models' MEASURED_IN_IMAGE constants.
PINNED_IMAGE = "redring/cleft-aesthetics@sha256:2135e27b5d82b28cb5e2059c606aadf5736df80e50cb4e52fc669cc8ba33d2ab"  # noqa: E501


#: **[RULED 2026-08-29] THE FOUR PHASE-17 DECISIONS.**
PHASE_17_RULINGS = {
    "ruled": "2026-08-29, all four open items",
    "consumes": (
        "the amendment on the parked record, "
        "synthesis.PARKED['arm_amended_2026_08_29'] -- applied the same "
        "day, original preserved, with the adoption-time flag "
        "(LIMITATIONS['magnitude_to_grade_is_an_assumption']) cited by "
        "name and quoted rather than paraphrased"
    ),

    # ---- the three arms ----------------------------------------------
    "arm_a": (
        "**A -- the parked TPS design, as parked**, post-amendment "
        "wording: a TSTR design in the Rosero family that TESTS the "
        "magnitude-to-grade assumption Rosero's construction avoids. "
        "The four parked limitations travel with it unchanged, "
        "including the advance prediction that it is not predicted to "
        "succeed. **A is the stronger-assumption cousin**"
    ),
    "arm_b": (
        "**B -- method replication with two declared adaptations, on a "
        "harder label.** Piecewise-affine warp over SCUT's SHIPPED "
        "86-point landmarks (no detector anywhere); binary contrastive "
        "on symmetric/asymmetric pairs; fixed-midline left/right "
        "split; distance readout normalized-and-rounded to the grade "
        "scale. **The two adaptations are DECLARED, not discovered**: "
        "(i) shipped landmarks in place of detection, (ii) fixed "
        "midline in place of a landmark-localised split. **The claim "
        "wording is bound here: never 'we replicated Rosero'**"
    ),
    "arm_c": (
        "**C -- the hybrid**: TPS synthesis (A's transformation), B's "
        "training and readout. What it buys is in the contrast family: "
        "with A it shares synthesis and varies the training "
        "scheme/label assumption; with B it shares training and varies "
        "the transformation family"
    ),

    # ---- frozen branches, firmly --------------------------------------
    "frozen_branches": (
        "**frozen ViT-B/16 branches, trained linear projection, "
        "contrastive loss, distance readout** -- the ruling, "
        "firmly. **Dated scoping note, 2026-08-29: 'transfer' in Phase "
        "17 means THIS.** The criterion-1 amendment's sentence -- "
        "frozen-backbone linear-probe transfer and nothing else -- "
        "gains a sibling: frozen-backbone linear-PROJECTION contrastive "
        "transfer. Full-branch training remains NEVER-OPERATIONALISED "
        "in this project (no shipped config has ever set trainable: "
        "full; measured at the criterion-1 amendment)"
    ),
    "trainable_branches_unexercised": (
        "**trainable branches are the UNEXERCISED ALTERNATIVE, recorded "
        "with the reason**: B already introduces a new loss AND a new "
        "readout relative to everything banked; adding trainable "
        "branches would be THREE NOVELTIES AT ONCE, and a result could "
        "not be attributed to any one of them. The Phase-14 pattern: "
        "named now so it cannot arrive later as a fresh idea"
    ),

    # ---- the landmark boundary, on the supervision list ------------------------
    "landmark_boundary_asked": (
        "added to the consolidated ask as **question 9** "
        "(phase11.SUPERVISOR_CONSOLIDATED_ASK): 'I used SCUT's shipped "
        "landmark files for training-side synthesis, no detector "
        "anywhere -- tell me if that crosses the line you drew.' The "
        "constraint's recorded basis quoted beside it: 'no landmark "
        "detector is available and the supervision material rejected the one from its own "
        "group' (geometry/mirror.py) -- a rejection of DETECTION, not "
        "of landmark data, and the project already reads shipped "
        "landmarks in Phases 5 and 15. NOT blocking: the question "
        "changes what the write-up can say, not what the phase can do"
    ),

    # ---- seeds and the contrast family ---------------------------------
    "five_seeds_per_arm": (
        "the standing five seeds per arm -- which pairs naturally with "
        "phase7b.paired_comparison's by-seed machinery: five "
        "synthetic-trained models per arm, each producing an all-237 "
        "prediction vector (zero real images in training, so all 237 "
        "are test), paired per patient per seed against the probe's "
        "OOF vectors"
    ),
    "the_five_contrast_family": (
        "**REGISTERED IN ADVANCE, count = FIVE, nothing added after "
        "the restate locks.** Three PRIMARIES: A vs the 0.2520 probe, "
        "B vs the probe, C vs the probe -- each under the full "
        "criterion. Two SECONDARIES: **A-C** isolates the "
        "training-scheme/label assumption on SHARED synthesis (same "
        "TPS faces, magnitude-ordered regression vs binary "
        "contrastive); **B-C** isolates the transformation family on "
        "SHARED training (same contrastive recipe, piecewise-affine vs "
        "TPS)"
    ),
    "combination_readings_skeleton": (
        "**a SKELETON, to be completed at the restate -- the arms are "
        "deliberately non-independent, so patterns across the family "
        "carry the meaning, and the pattern readings must be written "
        "before the numbers.** The cells to fill: (i) all three "
        "primaries null -- what that says about TSTR on this cohort as "
        "a class; (ii) B above A with C between -- the label "
        "assumption is the damage, the Rosero construction's avoidance "
        "of it vindicated; (iii) B above C -- the transformation "
        "family matters beyond the training scheme; (iv) any primary "
        "positive -- which caveats bound it (no scar, shipped-landmark "
        "boundary, magnitude assumption for A/C); (v) discordant "
        "secondaries with concordant primaries -- what the family can "
        "and cannot attribute. EACH cell gets its committed sentence "
        "at the restate; none is filled tonight"
    ),
}


#: **[DESIGNED 2026-08-29, NOT RUN] THE COMPUTE-GATE MEASUREMENT.**
#:
#: Measure before sizing; test the recorded diagnosis
#: (``synthesis.CLUSTER_PERFORMANCE_UNRESOLVED`` lineage: ~1,400
#: CPU-seconds/face on the cluster against 0.30 s/face locally, ~185x
#: wall, with profiling pointing at BLAS thread-pool spin -- 99.98%
#: overhead, the signature of spin-waiting, not slow work).
COMPUTE_GATE_MEASUREMENT = {
    "designed": "2026-08-29 -- the job is designed here, the maintainer launches",
    "what_it_tests": (
        "the recorded leading hypothesis: BLAS thread-pool spin under "
        "node contention. The A/B is the SAME shipped config launched "
        "twice, identical in everything except job env -- run 2 pins "
        "every BLAS pool to one thread"
    ),
    "timing_observability": (
        "**MEASURED IN SOURCE: the shipped sheet-only config "
        "(p5_asymmetry_synthesis, n_faces 12, write_artifact false) "
        "logs NOTHING per face** -- the only per-face log line "
        "('checkpoint at face N/M') sits inside `if write_artifact:` "
        "and never fires in sheet-only mode. ctx.log timestamps every "
        "line at one-second resolution, so total wall for the 12-face "
        "loop is coarsely derivable from the surrounding lines -- "
        "sufficient at ~1,400 s/face, USELESS if run 2 is fast (12 "
        "faces at ~0.3 s would land inside two timestamps)"
    ),
    "smallest_honest_change_reported_not_applied": (
        "**a report, per the ruling -- it touches task code, so it is "
        "not applied this turn.** One unconditional log line in "
        "task_asymmetry_synthesis's per-face loop: elapsed = "
        "time.perf_counter() around the face's work, then "
        "ctx.log(f'face {index+1}/{n_faces} in {elapsed:.2f}s'). A log "
        "line, not a behaviour change; carrying the elapsed IN the "
        "message is what makes it resolution-independent, because the "
        "1-second timestamp granularity cannot resolve a fast run. "
        "Twelve lines at n_faces=12"
    ),
    # [APPLIED 2026-08-29, the maintainer's go] The line above went in exactly
    # as reported -- perf_counter at the face's start, the unconditional
    # log after per_face.append and BEFORE the write_artifact block, so
    # a checkpoint flush cannot spike one face's number. Nothing else in
    # the task changed. (First application landed in task_masked_scut,
    # whose loop tail is near-identical; caught the same minute by
    # asking which function the line was in, reverted, re-applied inside
    # task_asymmetry_synthesis's own span -- the scoped-anchor lesson,
    # again.)
    "timing_line_applied": "2026-08-29, exactly as reported",

    # ---- the two job forms, corrected --------------------------------
    # [CORRECTED 2026-08-29, the maintainer] The report's job forms named the
    # BASE image (pytorch/pytorch:2.8.0...), inferred from the
    # Dockerfile FROM line. No recorded reason supported that choice --
    # it was an inference error. The launch image is the
    # project's pinned image (PLAN 2.1, and the models' own
    # MEASURED_IN_IMAGE constants), and the A/B must hold environment
    # fixed: THE PATHOLOGY WAS OBSERVED UNDER THE PINNED IMAGE, so both
    # runs use it.
    "job_forms": {
        "image_both_runs": (
            PINNED_IMAGE + " -- the project's pinned image, under "
            "which the pathology was observed"
        ),
        "command_both_runs": (
            "scripts/entrypoint.sh --sha <launch sha> --config "
            "configs/p5_asymmetry_synthesis.yaml -- the standing "
            "cluster launch; the image contains no application code"
        ),
        "env_both_runs": (
            "the standard four -- CLEFT_REPO, CLEFT_OUT, "
            "CLEFT_IMAGE_DIGEST, CLEFT_JOB_ID -- plus the one root "
            "this config declares: CLEFT_SCUT_ROOT="
            "/home/user/codex/scut/SCUT-FBP5500_v2, verified against "
            "the declared rollup 1041ceed... at load"
        ),
        "run_2_adds_only": (
            "OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, "
            "OPENBLAS_NUM_THREADS=1 -- the three BLAS pins, and "
            "NOTHING else differs between the two jobs"
        ),
    },

    # ---- what each outcome means, before the numbers exist -----------
    "reading_if_the_gap_collapses": (
        "**the 185x was ENVIRONMENTAL** -- futex contention between "
        "BLAS pools, the recorded diagnosis CONFIRMED by measurement. "
        "CLUSTER_PERFORMANCE_UNRESOLVED is discharged by measurement, "
        "and the full synthesis set is unconstrained by compute"
    ),
    "reading_if_the_gap_persists": (
        "the pathology is recorded UNRESOLVED-AND-NOT-GATING -- it "
        "does not block the phase; the synthesis set is SIZED TO THE "
        "MEASURED COST, and the sizing arithmetic uses the measured "
        "per-face number, not the local one"
    ),
    "gpus_idle_is_expected": (
        "**stated in advance so it is not read as a defect**: TPS "
        "warping is scipy/numpy CPU work; GPUs are idle throughout "
        "both measurement runs. The constraint under test is THREAD "
        "BEHAVIOUR, not hardware"
    ),
    # ---- [MEASURED 2026-08-29] the outcome: the THIRD possibility ----
    "outcome_2026_08_29": (
        "**the THIRD outcome, which neither committed reading "
        "anticipated: the baseline itself was fast.** From the two dev "
        "runs (p5_asymmetry_synthesis__377d769d__p17-synth-cost-"
        "baseline / __p17-synth-cost-pinned): baseline cold 0.55-1.17 "
        "s/face; baseline warm (resume attempt 1 after a benign "
        "preemption -- no FAILED line) 0.21-0.59; pinned 0.23-0.53. "
        "**Pinning ~ warm baseline; no pathology present to diagnose**"
    ),
    "cluster_performance_closed": (
        "**CLUSTER_PERFORMANCE_UNRESOLVED is CLOSED as not-gating**: "
        "the pathology was NOT reproduced at re-measurement 2026-08-29 "
        "(0.2-1.2 s/face against the recorded 56 s/face, same pinned "
        "image, clean tree at 377d769). The futex hypothesis was "
        "never tested because the disease did not show; candidate "
        "explanations for the original observation -- node, contention "
        "at the time, code path differences -- are recorded "
        "UNADJUDICATED, and the original observation is preserved as "
        "history in the synthesis module"
    ),
    "set_size": (
        "**unconstrained -- full SCUT.** Both numbers exist; the "
        "sizing rule is discharged"
    ),
    "dev_tier_mechanism": (
        "the dev-tier placement of both measurement runs matched all "
        "historical runs of this config because the config's own "
        "``tier: dev`` field routes it -- the tier is a config field "
        "read at run start, not a diversion applied at launch"
    ),
    "no_size_until_both_numbers": (
        "**no synthesis set size is declared until both timings "
        "exist.** A size chosen from the local number assumes the "
        "collapse; a size chosen from the old cluster number assumes "
        "the persistence; both are the measure-don't-reason failure"
    ),
}



#: **[MEASURED 2026-08-30]** -- REPORT-ONLY, resume semantics untouched.
#: [TAG NORMALISED 2026-09-01 from '[REPORTED 2026-08-30, REPORT-ONLY
#: -- resume semantics untouched]'; the qualifier is prose now.]
#: The completed-artifact-plus-deterministic-post-write-crash pattern,
#: and one corrected claim inside its own report.**
RESUME_PATTERN_REPORTED = {
    "reported": "2026-08-30, from run p17-synth-pwa's attempts 1-4",
    "the_pattern": (
        "a completed artifact plus a deterministic post-write crash can "
        "never succeed on resume: the rename deletes .inprogress and the "
        "checkpoint lived inside it, so every retry meets the "
        "immutability guard with no working state -- four attempts of "
        "pure waste. The task CAN cheaply distinguish 'this run's own "
        "earlier attempt published it' (ctx.attempt > 0 plus the run's "
        "own log carrying the artifact: line)"
    ),
    "smallest_honest_option": (
        "a CLEARER REFUSAL in the guard's branch when both signals "
        "hold -- 'this run's earlier attempt published the artifact and "
        "then crashed post-write; a retry cannot succeed' -- no resume "
        "semantics change. The larger option (a summary-only resume) "
        "exists but changes resume semantics; a maintainer decision, not "
        "applied"
    ),
    # [CORRECTED 2026-08-30, same day as first recorded] The report as
    # DELIVERED said the summary-only resume was feasible because
    # "faces.jsonl inside the published artifact holds every record the
    # summary needs". **Wrong name, wrong format, about an artifact**:
    # the published artifact carries faces.json -- ONE JSON document, a
    # list -- and faces.jsonl is the resume journal the finalizer
    # DELETES before the rename. The same wrong model of the artifact
    # shipped in both arm readers and crashed arm A at run.py's index
    # read (FileNotFoundError, all attempts identical); the feasibility
    # claim itself survives the correction -- faces.json holds the same
    # records -- but under the right name and a whole-document parse.
    "corrected_2026_08_30": (
        "the report's 'faces.jsonl inside the published artifact' is "
        "WRONG: the artifact holds faces.json (one JSON list); the "
        ".jsonl journal is deleted at finalization. The wrong model, "
        "not a typo -- it also shipped in both arm readers "
        "(run._synth_index's docstring carries the crash)"
    ),
}


#: **[LOCKED 2026-08-29] THE EXIT CRITERIA. Nothing is added after this
#: record.**
EXIT_CRITERIA = {
    "locked": (
        "2026-08-29 -- **nothing is added after this record**; a "
        "criterion discovered missing later is a limitation of the "
        "lock, recorded as such, never a retro-fitted entry"
    ),
    "criteria": (
        "the synthesis artifact: a full-SCUT deformation set at G1 -- "
        "the probe's geometry, like-for-like with the 0.2520 bar; G2 "
        "recorded as unregistered for this phase to keep the family at "
        "five contrasts -- magnitudes as shipped (0.0, 0.015, 0.025, "
        "0.035), side-balanced, declared inputs, artifact hash "
        "recorded",
        "three arms, five seeds each, all trained on zero real patient "
        "images, evaluated on all 237 as pure test: A (frozen ViT-B/16 "
        "embeddings, linear head, magnitude-mapped labels), B "
        "(piecewise-affine warps over shipped SCUT 86-point landmarks, "
        "frozen branches + trained linear projection, contrastive "
        "loss, fixed-midline split, distance readout "
        "normalized-and-rounded), C (TPS synthesis, B's training and "
        "readout)",
        "the five-contrast family and nothing beyond it: three "
        "primaries (each arm vs the probe, paired BCa by shared seed, "
        "both conditions), two secondaries (A-C, B-C). Every outcome "
        "recorded claimable/withdrawn/unresolved",
        "registered metrics only: PCC and Spearman vs the mean; "
        "3-class accuracy vs class3 at 2.5/3.5 with 0.333/0.502 "
        "beside",
        "per-arm and combination readings committed before any "
        "synthesis-set number exists (READINGS_COMMITTED, same turn as "
        "this lock)",
        "LOCKED as of 2026-08-29, this record's date",
    ),
}


#: **[COMMITTED 2026-08-29, PRE-RUN] The readings -- per arm, both
#: ways, and the five combination cells with their sentences.**
READINGS_COMMITTED = {
    "committed": (
        "2026-08-29, before any synthesis-set number exists -- the "
        "skeleton filled, the family locked at five"
    ),
    "arm_a_carried": (
        "carried from the parked record, unchanged: **not predicted to "
        "succeed**, and the reason is specific -- it cannot deform the "
        "philtrum, ranked first of 22 at |Spearman| 0.151 "
        "(synthesis.LIMITATIONS['the_arm_is_not_predicted_to_succeed'])"
    ),
    "arm_b_if_null": (
        "**the method that reached 0.31 on CARS lip symmetry does not "
        "transfer to the composite panel label** -- consistent BOTH "
        "with the label being more than symmetry AND with the two "
        "declared adaptations costing signal, unattributable between "
        "them and stated so"
    ),
    "arm_b_if_positive": (
        "claimably above zero -> **a symmetry-only, zero-real-image "
        "method carries measurable signal into a composite label**, "
        "bounded by the caveats: no scar, and CARS-vs-composite -- the "
        "label Rosero's method was built for is not this label"
    ),
    "arm_c_if_null": (
        "consistent with A's parked prediction extending to the "
        "training scheme: the anchored-philtrum synthesis fails to "
        "carry grade signal whether the labels are magnitude-mapped or "
        "contrastive"
    ),
    "arm_c_if_positive": (
        "**the scheme rescues what magnitude-labels could not, on "
        "identical images** -- the strongest available evidence that "
        "the magnitude-to-grade assumption, not the synthesis, is "
        "where A's signal dies"
    ),
    "combination_cells": {
        "all_null": (
            "**the family-level SIXTH convergent null: TSTR fails on "
            "this label regardless of scheme or transformation.** Not "
            "three separate nulls -- one family-level statement, "
            "because the arms share components by design"
        ),
        "b_above_a_c_near_b": (
            "**the training scheme carries and A's deficit attributes "
            "to the magnitude-to-grade assumption -- the family's most "
            "informative outcome**: the one pattern that localises the "
            "damage to a single named assumption"
        ),
        "b_above_c": (
            "**the transformation family matters**: piecewise-affine "
            "landmark configuration vs TPS with the anchored philtrum "
            "-- the deformation geometry itself, beyond the training "
            "scheme, decides what signal survives"
        ),
        "any_positive": (
            "noteworthy per the B/C positive readings, **never worded "
            "as beating the probe unless the primary contrast says "
            "so** -- a TSTR arm above zero and a TSTR arm above 0.2520 "
            "are different sentences with different evidence"
        ),
        "discordant_secondaries": (
            "**interactions the one-factor design cannot attribute, "
            "recorded unresolved** -- the family isolates one factor "
            "per secondary only when the factors do not interact, and "
            "discordance is the measurement saying they do"
        ),
    },
}


#: **[DECLARED 2026-08-29] Settings without precedent -- tau-style:
#: declared with reasoning, never tuned across runs, movement is a
#: dated amendment.**
DECLARED_SETTINGS_17 = {
    "magnitude_to_grade_map": (
        "**A's labels: grade = 1 + 4 * (magnitude / 0.035)** over the "
        "shipped series -> (1.0, 2.714, 3.857, 5.0). Endpoints "
        "anchored -- undeformed maps to the best grade, the largest "
        "shipped magnitude to the worst -- and LINEAR in between, "
        "because inventing curvature would smuggle in a calibration "
        "the record says does not exist. **The map IS the assumption "
        "under test** (the parked record's ordered-not-graded flag), "
        "and declaring one concrete form is what makes it testable"
    ),
    "margin": (
        "**margin = 1.0.** The projection space has no prior unit; the "
        "margin DEFINES the unit -- every distance in the loss and the "
        "readout is measured relative to it -- so its absolute value "
        "is a units choice, exactly tau's reasoning in Phase 16"
    ),
    "readout_normalization": (
        "**grade = round(1 + 4 * min(d / margin, 1))** -- integers 1..5. "
        "Normalized by the margin because that is the loss's own unit "
        "of 'fully asymmetric'; no second scale is invented, and the "
        "readout is dimensionless by the same argument as the softmax "
        "temperature. Rounded per the ruling's wording "
        "('normalized-and-rounded to the grade scale')"
    ),
    "projection_dim": (
        "**768 -> 768, full width.** Mirrors Phase 16's full 768x768 "
        "precedent, and for the same reason the low-rank W was "
        "unregistered there: a bottleneck width is a rank knob with no "
        "registered value"
    ),
    "never_tuned": (
        "all four are scientific settings under the standing clause: "
        "declared in the YAML (or, for the map, in this record and the "
        "generator) before the first run, NEVER tuned across runs, "
        "movement is a dated amendment with a reason"
    ),
}


#: **[RECORDED 2026-08-29] Every setting's provenance, per value.**
SETTINGS_PROVENANCE_17 = {
    "rule": (
        "mirror the closest probe-training precedent where one exists; "
        "declare with reasoning where none does"
    ),
    "max_epochs": "40 -- p7_d1_vit_b16_imagenet_g1.yaml max_epochs",
    "learning_rate": "0.001 -- p7_d1_vit_b16_imagenet_g1.yaml learning_rate",
    "batch_size": "32 -- p7_d1_vit_b16_imagenet_g1.yaml batch_size",
    "inner_val_frac": "0.2 -- p7_d1_vit_b16_imagenet_g1.yaml inner_val_frac",
    "seeds": (
        "[1337, 2024, 7, 99, 12345] -- p7_d1_vit_b16_imagenet_g1.yaml "
        "seeds, so every contrast pairs by shared seed"
    ),
    "monitor": "inner_val_mse -- p7_d1_vit_b16_imagenet_g1.yaml monitor",
    "optimiser": "AdamW -- the repo's sole training optimiser",
    "patience": (
        "ABSENT BY RULING (fixed budget, best inner-val checkpoint), "
        "absent from both new schemas -- writing it is refused as an "
        "unknown key"
    ),
    "margin": "1.0 -- NO PRECEDENT; declared (DECLARED_SETTINGS_17)",
    "readout_normalization": (
        "round(1 + 4*min(d/margin, 1)) -- NO PRECEDENT; declared "
        "(DECLARED_SETTINGS_17)"
    ),
    "magnitude_to_grade_map": (
        "1 + 4*(m/0.035) -- NO PRECEDENT; declared "
        "(DECLARED_SETTINGS_17); the assumption under test, made "
        "concrete"
    ),
    "projection_dim": (
        "768 -- mirrors phase16's full 768x768 W (the no-rank-knob "
        "reasoning carries)"
    ),
}


# --------------------------------------------------------------------------
# the machinery
# --------------------------------------------------------------------------


def magnitude_to_grade(magnitudes):
    """DECLARED_SETTINGS_17's map: grade = 1 + 4 * (m / 0.035)."""
    import numpy as np

    values = np.asarray(magnitudes, dtype=float)
    if values.min() < 0:
        raise ValueError("magnitudes are non-negative by construction")
    return 1.0 + 4.0 * (values / 0.035)


def distance_to_grade(distance: float, *, margin: float) -> int:
    """DECLARED_SETTINGS_17's readout: round(1 + 4*min(d/margin, 1))."""
    if margin <= 0:
        raise ValueError("margin must be positive")
    return int(round(1.0 + 4.0 * min(float(distance) / margin, 1.0)))


def left_right_views(image):
    """The fixed-midline split (declared adaptation ii).

    Each view is a FULL staged square with the off half filled with
    staging white (255), so the frozen extractor's input contract is
    unchanged; the right view is mirrored so anatomy aligns across the
    pair. A perfectly symmetric image yields identical views -- the
    property the contrastive loss trains toward on symmetric samples.
    """
    import numpy as np

    array = np.asarray(image)
    width = array.shape[1]
    half = width // 2
    left = array.copy()
    left[:, half:] = 255
    mirrored = array[:, ::-1].copy()
    right = mirrored.copy()
    right[:, half:] = 255
    return left, right


#: **[OBSERVED 2026-08-30 -- honesty over retrofit] THE UNPREDICTED
#: PATTERN. No committed combination cell fired.**
UNPREDICTED_PATTERN = {
    "observed": (
        "2026-08-30, from run p17-family-analysis -- the ruling: "
        "record what happened beside what was expected, amend nothing"
    ),
    "no_cell_fired": (
        "**none of the five committed combination cells describes the "
        "outcome.** Not all-null (A is 0.019 from the probe); not "
        "B-above-A (B is the claimable NEGATIVE); not B-above-C (that "
        "contrast shows nothing); the any-positive cell has no "
        "claimable positive to bound; the discordant-secondaries cell "
        "assumed concordant primaries. The cells stand PRESERVED AND "
        "UNFIRED in READINGS_COMMITTED, byte-unchanged"
    ),
    "the_observed_pattern": (
        "**the training-scheme axis dominates, in the direction "
        "OPPOSITE to expectation**: magnitude-labeled regression (A, "
        "0.2334) transfers to within 0.019 of the probe while both "
        "contrastive arms sit at zero (B -0.0044, C -0.0185). The "
        "cells were written expecting B -- the Rosero-construction "
        "scheme that avoids the magnitude-to-grade assumption -- to "
        "carry, and it did not; the arm carrying is the one built ON "
        "that assumption"
    ),
    "what_this_is_and_is_not": (
        "**an OBSERVATION, not a registered reading.** The A-C "
        "secondary that would attribute the pattern to the scheme is "
        "UNRESOLVED (p17-a-vs-c: +0.2518 at 4.9x threshold, condition "
        "1 broken by one seed), so the attribution is not claimed. The "
        "cells are NOT amended, per the nothing-retrofitted rule: a "
        "reading written after the number is not a reading"
    ),
}


#: **[CLOSED 2026-08-30] PHASE 17. TSTR.**
PHASE_17_CLOSING = {
    "closed": (
        "2026-08-30 -- run p17_family_analysis__3d0ca56c__"
        "p17-family-analysis, finalized, single attempt; all six locked "
        "criteria walked"
    ),
    "criterion_1_synthesis_artifacts": (
        "MET: two full-SCUT G1 sets, magnitudes (0.0, 0.015, 0.025, "
        "0.035), side-balanced, declared -- TPS 855c13fa..., "
        "piecewise-affine 878626ee.... **The PWA artifact is the "
        "DELETE-AND-RELAUNCH set**: the first PWA run completed all "
        "5,499 faces and published, then crashed in the summary on the "
        "record-contract defect; the ruling was the orphaned artifact "
        "deleted and the fixed task relaunched, and the declared hash "
        "is the relaunch's"
    ),
    "criterion_2_three_arms_zero_real": (
        "MET: three arms x five seeds, zero real patient images or "
        "labels in training (structurally tested), all 237 pure test. "
        "Per-arm means: A 0.2334 (sd 0.0044), B -0.0044 (sd 0.0317), "
        "C -0.0185 (sd 0.0586)"
    ),
    # [2026-08-31] The three per-arm means stated in prose above now
    # have a structured home: ARM_MEANS. The prose is unedited.
    "arm_means_structured_2026_08_31": "ARM_MEANS",
    "criterion_3_five_contrasts": (
        "MET, and nothing beyond them: five rows appended in "
        "registration order (p17-a-vs-probe, p17-b-vs-probe, "
        "p17-c-vs-probe, p17-a-vs-c, p17-b-vs-c) -- one CLAIMABLE "
        "(negative), four UNRESOLVED, every outcome recorded under the "
        "standing vocabulary"
    ),
    "criterion_4_registered_metrics": (
        "MET: PCC and Spearman vs the mean, 3-class accuracy at "
        "2.5/3.5 with 0.333/0.502 beside, confirmed present in all "
        "three arm runs' metrics.json -- nothing beyond the registered "
        "set"
    ),
    "criterion_5_readings_committed": (
        "MET on the committed side, and the outcome fell OUTSIDE the "
        "committed cells: per-arm readings attached to their rows "
        "(B's verbatim on the claimable negative); the five "
        "combination cells preserved unfired with the observed pattern "
        "recorded beside them, dated (UNPREDICTED_PATTERN)"
    ),
    "criterion_6_locked": (
        "the lock held: the family stayed at five, nothing was added "
        "after 2026-08-29, and the unpredicted outcome was recorded as "
        "an observation rather than retrofitted into a reading"
    ),

    # ---- the phase's operational history ------------------------------
    "build_cycle_defects": (
        "recorded as operational history, each with the test it "
        "produced: (1) the PWA record-contract crash -- piecewise "
        "records lacked identity/deformation keys the summary consumes; "
        "fixed at the contract, family-parameterised end-to-end smoke "
        "added (piecewise.deformation_summary carries the story). (2) "
        "the writer->reader filename break -- both arm readers opened "
        "faces.jsonl, the resume journal the finalizer deletes, instead "
        "of faces.json; fixed through one shared reader "
        "(run._synth_index), chained writer->reader test added. Three "
        "instrument defects, zero data defects, in this phase"
    ),

    # ---- what the phase found -----------------------------------------
    "the_headline": (
        "**a zero-real-image arm reached 0.2334 on the real 237 -- "
        "~93% of the trained probe's 0.2520, statistically "
        "indistinguishable (0/5 intervals exclude zero) -- and it was "
        "the arm NOBODY predicted would succeed.** Parity, not "
        "success: the point estimate runs slightly against A and the "
        "cohort cannot resolve it. The contrastive arms carried "
        "nothing; B's failure is the phase's one claimable result, "
        "with its committed reading attached"
    ),
    "caveats_carried_forward": (
        "no scar -- now with the measured irony that the scar-free "
        "synthesis transferred anyway; the magnitude-to-grade "
        "assumption -- now the NAMED CANDIDATE for why A works, "
        "flagged [REASONED] and not measured; the anchored philtrum; "
        "and the supervision landmark-boundary question, still open "
        "(question 9)"
    ),
    "prediction_reckoned": (
        "the parked module's advance prediction is reckoned with a "
        "dated note beside it, original byte-preserved "
        "(synthesis.LIMITATIONS"
        "['prediction_reckoned_2026_08_30']); the ruling on "
        "record: the old prediction was wrong"
    ),
}


#: **[BANKED 2026-08-31] THE THREE TSTR ARM MEANS, STRUCTURED.**
#:
#: These closed Phase 17 and have lived only in prose since --
#: ``PHASE_17_CLOSING["criterion_2_three_arms_zero_real"]``: "Per-arm
#: means: A 0.2334 (sd 0.0044), B -0.0044 (sd 0.0317), C -0.0185 (sd
#: 0.0586)". The standing record-artifact check (``record_audit``) could
#: not see them, so three of the 68 locked arms were compared by
#: nothing. **The prose is unedited**; this is a second expression of the
#: same numbers, with a dated pointer beside the original.
#:
#: **Transcribed from that closing, not regex-parsed** -- and every
#: figure is checked against a quantity banked elsewhere, so a
#: transcription slip fails rather than passing quietly:
#:
#:     arm   pcc      - probe 0.2520 = delta   ledger's paired delta
#:     A     0.2334   -0.0186                  -0.0186   exact
#:     C    -0.0185   -0.2705                  -0.2705   exact
#:     B    -0.0044   -0.2564                  -0.2558   differs by
#:                                                       0.0006
#:
#: **A and C reproduce the ledger's mean paired delta exactly.** B does
#: not, and that is expected rather than a discrepancy: the ledger's
#: figure is the per-patient PAIRED estimate, not the difference of
#: pooled means -- the same distinction ``phase10.PAIRED_BCA_OBSERVED``
#: recorded when its own d differed from the pooled difference by
#: 0.0001. The two agree wherever the paired and pooled quantities
#: coincide and separate where they do not.
#:
#: **The sds are checked the same way**, through the thresholds they
#: generate: ``phase3.combined_claimable_delta(sd, 5, 0.0148, 5)``
#: returns 0.0307 for B and 0.0530 for C -- exactly the ledger's
#: thresholds -- and 0.0135 for A against the ledger's 0.0136, a
#: last-digit difference from computing on a 4-dp sd rather than the
#: full-precision one. That single 0.0001 is the same precision story
#: the record-artifact check itself measures
#: (``record_audit.RECORD_ARTIFACT_CHECK``).
ARM_MEANS = {
    "banked": "2026-08-31",
    "provenance": (
        "transcribed from PHASE_17_CLOSING"
        "['criterion_2_three_arms_zero_real'], NOT regex-parsed; every "
        "figure cross-checked against a quantity banked elsewhere "
        "(see checked_against)"
    ),
    "arms": {
        "p17_arm_a": {"pcc": 0.2334, "sd": 0.0044, "n_seeds": 5},
        "p17_arm_b": {"pcc": -0.0044, "sd": 0.0317, "n_seeds": 5},
        "p17_arm_c": {"pcc": -0.0185, "sd": 0.0586, "n_seeds": 5},
    },
    "checked_against": (
        "PCC: A and C reproduce the ledger's mean paired delta against "
        "the 0.2520 probe EXACTLY (-0.0186, -0.2705). B's pooled "
        "difference -0.2564 differs from the ledger's paired -0.2558 by "
        "0.0006 -- expected, the paired-vs-pooled distinction "
        "phase10.PAIRED_BCA_OBSERVED recorded at 0.0001. SD: "
        "combined_claimable_delta(sd, 5, 0.0148, 5) returns the "
        "ledger's thresholds 0.0307 (B) and 0.0530 (C) exactly, and "
        "0.0135 for A against a banked 0.0136 -- a 4-dp rounding, the "
        "same precision story record_audit measures"
    ),
    "the_prose_is_unedited": (
        "PHASE_17_CLOSING is untouched; it carries a dated pointer here"
    ),
}


def summary() -> dict:
    """The phase's records so far, importable as one object."""
    return {
        "rulings": PHASE_17_RULINGS,
        "compute_gate": COMPUTE_GATE_MEASUREMENT,
        "resume_pattern": RESUME_PATTERN_REPORTED,
        "exit_criteria": EXIT_CRITERIA,
        "readings": READINGS_COMMITTED,
        "declared_settings": DECLARED_SETTINGS_17,
        "settings_provenance": SETTINGS_PROVENANCE_17,
        "unpredicted_pattern": UNPREDICTED_PATTERN,
        "closing": PHASE_17_CLOSING,
        # [2026-08-31] The three per-arm means, given a structured home.
        "arm_means": ARM_MEANS,
    }
