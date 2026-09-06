"""Phase 18 -- the metric-space ablation.

Scheduled 2026-08-24 as Phase 16 (``phase15.PHASE_16_SCHEDULED``, whose
keys keep the number they were written under), renumbered to 18 by the
fourth sequence amendment. The registered question: **what do
classification metrics measure on this cohort that correlation does
not, and where do the two disagree?**

This module holds the rulings (2026-08-30), the five registered
deliverables with their pre-committed readings, the reckoning against
the pre-registered metric decision, and the DRAFT arm inventory -- the
approval artifact. **The exit-criteria lock waits for the maintainer's
approval of the enumerated arm list. Nothing is built.**
"""

from __future__ import annotations


#: **[RULED 2026-08-30] THE FOUR OPEN DECISIONS.**
PHASE_18_RULINGS = {
    "ruled": "2026-08-30, all four -- plus a fifth deliverable",
    "registration": (
        "phase15.PHASE_16_SCHEDULED, standing unchanged; these rulings "
        "supply the scope its scope_is_not_set_here clause deferred"
    ),

    "arm_set_enumerated_never_glob": (
        "**the arm set is a CLOSED ENUMERATED LIST, never a glob** -- a "
        "glob would silently change the phase's scope whenever a new "
        "run lands. BOTH directions enumerated: included arms by name, "
        "and EXCLUDED arms by name with reasons (void runs, "
        "non-current-generation, CSVs not surviving) -- so an exclusion "
        "is distinguishable from an oversight. The list is "
        "ARM_INVENTORY_DRAFT until the maintainer approves it; the lock "
        "follows the approval"
    ),
    "three_not_five_declined": (
        "**THREE_NOT_FIVE's named candidate is DECLINED, dated "
        "2026-08-30, beside the pointer** "
        "(classification.THREE_NOT_FIVE['declined_2026_08_30']): the "
        "tails have not changed (g1=8, g5=4 patients at the extreme "
        "grades), registering a 4-threshold rule would produce numbers "
        "the record calls undefined, and it would MANUFACTURE the "
        "CleftGNN 5-class comparability the different-quantities rule "
        "forbids. **Reopen condition stated**: a label source with "
        "sufficient support in the extreme grades -- a decision, not a "
        "permanent closure, per the declined-items pattern"
    ),
    "contrast_set_the_named_seven": (
        "**the criterion-under-F1 examination runs on the NAMED SEVEN "
        "contrasts**: p16-anchor-loop-unresolved, p17-a-vs-probe (the "
        "two condition-split rows), plus the five Phase 17 family "
        "contrasts (p17-a-vs-probe, p17-b-vs-probe, p17-c-vs-probe, "
        "p17-a-vs-c, p17-b-vs-c). Stated explicitly: **the deliverable "
        "is the CRITERION'S BEHAVIOUR, not an exhaustive re-audit** -- "
        "the blind audit exists for exhaustiveness"
    ),
    "compute_keeper_pinned": (
        "**a keeper job in the pinned image.** Pure CSV/record "
        "arithmetic costs seconds; there is no reason to accept a "
        "provenance hole for the phase whose subject is the "
        "measurement machinery itself"
    ),
}


#: **[REGISTERED 2026-08-30, READINGS PRE-COMMITTED] THE FIVE
#: DELIVERABLES.**
DELIVERABLES_REGISTERED = {
    "registered": "2026-08-30, before any table exists",

    # ---- D1: the two-family ladder re-scoring -------------------------
    "d1_two_family_rescoring": (
        "the enumerated arms re-scored under BOTH families: PCC and "
        "Spearman beside accuracy, macro F1, per-class F1, and "
        "QWK-3cat, floors beside every figure; Kendall tau and top-5 "
        "overlap between the two rankings"
    ),
    "d1_reading_concordant": (
        "concordant rankings -> **metric choice does not drive "
        "conclusions on this cohort** and PCC-primary survives "
        "measurement rather than resting on registration alone"
    ),
    "d1_reading_discordant_with_mechanism": (
        "discordant WITH the registered mechanism -> the movers are "
        "the NEAR-MEAN PREDICTORS: tolerable PCC, collapsed "
        "minority-class F1 -- **the addendum's 9.58x mechanism, named "
        "in advance as the expectation** "
        "(classification.CLASSIFICATION_METRICS_BANKED's floor split)"
    ),
    "d1_reading_discordant_without_mechanism": (
        "discordant WITHOUT that mechanism -> **the registered "
        "mechanism has FAILED and is recorded as failed, never "
        "adjusted to fit** -- the maintainer's strengthening, the "
        "REPAIR_WITHOUT_TRANSFER discipline applied prospectively"
    ),

    # ---- D2: the criterion under F1 -----------------------------------
    "d2_criterion_under_f1": (
        "the two-condition criterion re-evaluated under macro F1 on "
        "the named seven contrasts; a VERDICT-FLIP TABLE (per "
        "contrast: PCC verdict, F1 verdict, flipped or held). "
        "**Reading, committed**: flips are statements about the "
        "CRITERION'S METRIC-DEPENDENCE -- the banked PCC verdicts "
        "STAND, this measures the instrument, not the arms"
    ),

    # ---- D3: QWK's answer ----------------------------------------------
    "d3_qwk_answer": (
        "the set-aside closed BY MEASUREMENT: QWK-3cat "
        "(eval.metrics.qwk_3cat, frozen, unused by any banked verdict) "
        "computed across the enumerated arms; its ranking against both "
        "families; and a REGISTERED POSITION on what QWK adds over "
        "macro F1 at three classes, written from the arithmetic before "
        "the table"
    ),

    # ---- D4: the macro-averaging floor, derived not chosen -------------
    "d4_macro_floor_derived": (
        "**the macro-averaging floor is DERIVED, not chosen** -- El "
        "the maintainer's strengthening: computed from the class counts BEFORE "
        "the table exists. The worst-case single-patient perturbation "
        "at support [88, 119, 30], propagated to macro F1 "
        "(phase18.macro_f1_single_patient_floor -- the derivation in "
        "code, its value printed with this registration). **Bound: any "
        "observed macro-F1 difference below the floor is "
        "UNINTERPRETABLE regardless of what the criterion says** -- "
        "itself a finding about the metric at these class counts"
    ),

    # ---- D5: IEM as the third family -----------------------------------
    "d5_iem_third_family": (
        "**IEM as the third metric family -- the addition, "
        "motivated on record: the supervisor stated direct interest and "
        "wants IEM as small as possible; the phase answers directly.** "
        "[CORRECTED 2026-09-05: read 'Prof. the supervision material "
        "stated direct interest'. A name was replaced with a phrase and "
        "the sentence was never re-read, the same slip corrected at "
        "docs/PLAN.md under the CONTRADICTED block. The request itself "
        "is unchanged.] "
        "IEM computed across the same enumerated arms from the same "
        "banked CSVs; ranking against both families (tau, top-5); "
        "floor beside every figure -- the CONSTANT-PREDICTOR IEM at "
        "the panel mean, computed identically to the arms'"
    ),
    "d5_reference_point": (
        "Phase 11's banked pair enters as the reference, QUOTED from "
        "the record: 'IEM 0.5825 against 0.6130 and PCC 0.1855 against "
        "0.2552, ten of ten seed-wise comparisons in the expected "
        "direction with disjoint ranges' -- and 'both contrasts "
        "WITHDRAWN on condition 1' "
        "(phase11.PHASE_11_CLOSING lineage, the "
        "the_loss_works_and_the_bill_is_correlation wording)"
    ),
    "d5_defect_caveats_travel": (
        "**the measured metric defects travel as caveats on every IEM "
        "figure.** Quoted from the record "
        "(phase11's the_metric_has_two_defects): 'eq (16)'s value "
        "inverts below |d| = 0.19753 and its gradient below |d| = "
        "0.0719, so the branch it calls more dangerous is both SCORED "
        "and TRAINED more leniently near the truth; under convention A "
        "it ranks a near-constant predictor first in a 68-arm ladder'. "
        "**A wording note, kept honest**: the record's own phrase is "
        "TWO defects (value inversion 0.19753, gradient inversion "
        "0.0719); the narrow-predictor favouritism is recorded "
        "separately as A_FAVOURS_NARROW_PREDICTORS (inverted-region "
        "occupancy 1.685x, an arm crowned at PCC 0.024) -- three "
        "measured defects in total, cited by their own names rather "
        "than renumbered"
    ),
    "d5_readings": (
        "concordant with PCC -> **IEM adds no ordering information "
        "over correlation on this cohort**, said plainly. Discordant "
        "-> characterised by WHICH arms move, against the defect "
        "thresholds: **arms near |d| = 0.07 sit inside the measured "
        "gradient-inversion region and their IEM values may not order "
        "sensibly at all** -- the characterisation names the region "
        "before the numbers exist"
    ),

    # ---- the CleftGNN prohibition --------------------------------------
    "cleftgnn_iem_prohibition": (
        "OUR IEM VALUES ARE NEVER PLACED BESIDE CLEFTGNN'S TABLE "
        "2/4/6 FIGURES. Theirs: 5-class grade, their consensus label, "
        "28 images, single split, averaged over three consensus "
        "standards. Ours: continuous panel mean, 237 patients, 5-fold "
        "OOF, five seeds. IEM is a scaled error whose magnitude is "
        "governed by the label's spread; a side-by-side would be the "
        "different-quantities error the record has caught eight times"
    ),

    # ---- the standing clause -------------------------------------------
    "standing_clause_mse_objective": (
        "**on every deliverable: all arms were trained on MSE. This "
        "phase measures METRICS ON A FIXED OBJECTIVE and licenses no "
        "claim about F1-trained or IEM-trained arms.** The "
        "registration's inherited metric-vs-objective question is "
        "answered by this clause: the phase measures the metric, and "
        "says so on every figure"
    ),
}


#: **[RECKONED 2026-08-30] The opening reckoning, per the Phase-14
#: discipline the registration demands.**
PHASE_18_RECKONING = {
    "reckoned": "2026-08-30, against the pre-registered metric decision",
    "the_position_reckoned_against": (
        "quoted from the registration's own demand: the PRE-REGISTERED "
        "decision -- 'PCC is primary, QWK is not' -- with qwk_3cat "
        "frozen in eval/metrics.py and unused by any banked verdict"
    ),
    "conceded_covered": (
        "**the one-arm gap question is CONCEDED AS COVERED**: 'what "
        "are the classification numbers for the best arm, and why are "
        "they secondary' was asked and answered by the classification "
        "addendum (CLASSIFICATION_METRICS_BANKED: accuracy 0.5181 vs "
        "floor 0.5021, macro F1 0.3760 vs floor 0.2228, the 9.58x "
        "headline). Phase 18 does not re-ask it"
    ),
    "new_measurement": (
        "everything else is NEW: the ladder REORDERING under a second "
        "family (the addendum explicitly declined to compare arms); "
        "the criterion's own behaviour under F1; QWK's set-aside "
        "closed by measurement; the macro floor derived from the class "
        "counts; and the IEM family, which no phase has computed "
        "across arms. PCC's primacy is NOT reopened -- every banked "
        "verdict stands; the phase measures the instruments beside it"
    ),
}


# --------------------------------------------------------------------------
# D4's derivation, in code
# --------------------------------------------------------------------------


def macro_f1_single_patient_floor(support) -> dict:
    """The macro-F1 floor at these class counts, DERIVED not chosen.

    The worst-case effect of ONE patient's prediction changing, on a
    3-class macro F1, at the given truth supports. A single flip moves
    one unit between two cells of one row of the confusion matrix: it
    changes the moving class's recall and both affected classes'
    precision. The extremal configurations are degenerate -- the
    sharpest is a class whose ONLY correctly-predicted patient flips
    away, taking that class's F1 from 2/(s+1) to 0 in one move -- so
    the search walks structured degenerate states plus deterministic
    hill-climbing from them, and the analytic component
    max_c 2/(s_c+1) / n_classes is asserted as a lower bound on what
    the search must find.
    """
    import itertools

    import numpy as np

    support = [int(s) for s in support]
    n = len(support)

    def macro_f1(matrix):
        scores = []
        for c in range(n):
            tp = matrix[c][c]
            fp = sum(matrix[r][c] for r in range(n)) - tp
            fn = sum(matrix[c]) - tp
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            scores.append(
                0.0 if precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            )
        return float(np.mean(scores))

    def single_moves(matrix):
        for row in range(n):
            for src, dst in itertools.permutations(range(n), 2):
                if matrix[row][src] > 0:
                    moved = [list(r) for r in matrix]
                    moved[row][src] -= 1
                    moved[row][dst] += 1
                    yield moved

    # Structured starts: every assignment of each row's mass between a
    # "main" column and up to `spill` units in another column, with the
    # sharp states (0, 1, 2 correct) explicit.
    starts = []
    for diag in itertools.product(*[
        sorted({0, 1, 2, s // 2, s}) for s in support
    ]):
        matrix = [[0] * n for _ in range(n)]
        for c, correct in enumerate(diag):
            matrix[c][c] = min(correct, support[c])
            matrix[c][(c + 1) % n] = support[c] - matrix[c][c]
            starts.append([list(r) for r in matrix])

    best = {"delta": 0.0}
    for start in starts:
        frontier = [start]
        for _ in range(3):  # deterministic hill-climb, bounded depth
            state = frontier[-1]
            base = macro_f1(state)
            step = max(
                single_moves(state),
                key=lambda m: abs(macro_f1(m) - base),
            )
            delta = abs(macro_f1(step) - base)
            if delta > best["delta"]:
                best = {"delta": delta}
            frontier.append(step)

    analytic = max(2.0 / (s + 1) for s in support) / n
    if best["delta"] + 1e-12 < analytic:
        raise ValueError(
            f"search found {best['delta']:.6f} below the analytic "
            f"component {analytic:.6f}; the derivation is broken"
        )
    return {
        "support": support,
        "floor": round(best["delta"], 6),
        "analytic_component": round(analytic, 6),
        "meaning": (
            "one patient's prediction changing can move macro F1 by up "
            "to this much at these supports; any observed difference "
            "below it is uninterpretable at n=1 resolution"
        ),
    }


#: D4's registration carries the derived value, printed at import from
#: the derivation above -- never a hand-typed constant.
MACRO_F1_FLOOR = macro_f1_single_patient_floor([88, 119, 30])


#: **[DRAFT 2026-08-30 -- THE APPROVAL ARTIFACT. NOT LOCKED.]**
#:
#: The enumerated arm list awaiting the maintainer's approval. Verification
#: basis is stated per group because the run directories live on the
#: cluster: "writer" = the task kind's per-seed CSV writing verified in
#: source this turn; "read" = a completed cluster run has already read
#: the CSVs (operational proof they survive); "ls" = the listing at
#: approval is the outstanding check.
ARM_INVENTORY_DRAFT = {
    # [2026-08-30, later] Approved with the four cells ruled; the lock
    # is ARM_LIST_LOCKED. This draft is preserved as the approval
    # artifact it was.
    "status": "DRAFT FOR APPROVAL -- the lock is pending",
    "included_candidates": {
        "ladder_p7": (
            "the 36 arms of ladder.distinct_runs(), by name -- 5-seed "
            "transformer and 10-seed graph regimes; writer verified "
            "(train_cv/train_graph_cv both route phase3.write_outputs "
            "with seed prefixes); CSV survival: 'read' for the arms "
            "phase7b comparisons consumed, 'ls' for the rest"
        ),
        "p12_view_ablation": (
            "p12_arm_a_frontal__0dc7c79b, p12_arm_d_capacity__0dc7c79b, "
            "p12_arm_b_basal__0fb33b6a, p12_arm_c_concat__0fb33b6a -- "
            "4 arms, 5 seeds; 'read' (the paired run consumed them)"
        ),
        "p15_mebeauty_probes": (
            "the three probe_mebeauty runs (masked g1, masked g2, "
            "original) -- 5 seeds, phase3.write_outputs; 'ls' for "
            "survival; run dirs are in the phase-15 declarations"
        ),
        "p16_anchor_loop": (
            "p16_anchor_loop__f342fed9 -- loop AND identity-baseline "
            "CSVs, 5 seeds; 'read' would require no new check if the "
            "identity CSVs are included: both were written by the "
            "closing-verified run"
        ),
        "p17_arms": (
            "p17_arm_a__bf09bd45__p17-arm-a-2, "
            "p17_arm_b__4894169c__p17-arm-b, "
            "p17_arm_c__bf09bd45__p17-arm-c -- 5 seeds; 'read' (the "
            "family analysis consumed all three)"
        ),
    },
    "excluded_by_name": {
        "void_runs": (
            "the three CleftGNN launches ledgered VOID "
            "(void-cleftgnn-second-launch, void-cleftgnn-third-launch "
            "lineage) and p17_arm_a's first launch at eb5a6887 (the "
            "pre-fix crash; no surviving CSVs)"
        ),
        "protocol_mismatched": (
            "phase10's faithful CleftGNN arm: rater-specific models, "
            "85:15 single split, single run -- not per-seed OOF on the "
            "237, so it cannot enter a table whose columns assume that "
            "shape"
        ),
        "not_mse_trained": (
            "phase11's IEM-trained arm: the standing clause fixes the "
            "objective at MSE, and an IEM-trained arm in an IEM table "
            "would conflate metric with objective -- the exact "
            "confusion the clause exists to prevent"
        ),
        "no_per_seed_oof_csvs": (
            "p9's prototype classifier (grades CSV, not per-seed OOF "
            "predictions), p4's partition run (partitioning "
            "sensitivity, its own protocol), all p6/p15 pretraining "
            "runs (source-side, no 237 OOF)"
        ),
    },
    "a_ruling_decides": {
        "p11_mse_control": (
            "MSE-trained, 10 seeds, OOF CSVs -- but one half of a "
            "WITHDRAWN pair; including it re-surfaces a withdrawn "
            "context in a new table"
        ),
        "p13_probes": (
            "the reconstruction/occlusion decodability probes -- "
            "phase3-shaped CSVs exist, but they score RECONSTRUCTIONS, "
            "not label-model arms; arguably a different population"
        ),
        "roadb_annex_arms": (
            "Road B / region-crop arms -- current-generation keeper "
            "runs with CSVs, but annex-framed (ROAD_B_IS_THE_ANNEX); "
            "including them mixes the annex into a main-line table"
        ),
        "p7_g_label_variants": (
            "the ladder's median/ldl-target arms (p7_g*, 6 of the 36) "
            "-- MSE objective but a DIFFERENT TARGET; re-scoring them "
            "against the mean truth scores a label mismatch as model "
            "error. Flagged rather than silently included in group 1"
        ),
    },
}


#: Seed lists, by regime -- the ladder's own decision (5 transformer,
#: 10 graph; ladder.py header), read off the shipped configs.
SEEDS_5 = (1337, 2024, 7, 99, 12345)
SEEDS_10 = (1337, 2024, 7, 99, 12345, 42, 271828, 314159, 161803, 777)


#: **[LOCKED 2026-08-30, ON THE FOUR RULINGS] THE ARM LIST.**
#:
#: A CLOSED ENUMERATED LIST, never a glob -- every entry is
#: (arm_name, run_directory, seeds, csv_stem). The three p15 probe run
#: directories are not in the local record (their runs were terminal --
#: nothing ever declared them) and carry PENDING_ paths for the maintainer's
#: pass-0 paste; everything else is real. The generator derives the
#: config's input list FROM THIS CONSTANT, so the config cannot drift
#: from the lock.
ARM_LIST_LOCKED = {
    "locked": "2026-08-30 -- the four flagged cells ruled, the list closed",
    # [2026-08-31] Two hygiene gaps found by audit, both recorded WITHOUT
    # reopening this lock: seven arms with banked PCCs that the 68 does
    # not contain and that no exclusion names (ARM_LIST_ADDENDUM), and
    # the p7_g0 note below. The locked 68 and every Phase 18 figure over
    # them are unchanged.
    "addendum_2026_08_31": "ARM_LIST_ADDENDUM",
    "ruled_cells": (
        "p11 MSE control INCLUDED (its banked pairing is withdrawn, its "
        "predictions are not -- 'withdrawn' attaches to the contrast, "
        "not the run); p13 decodability probes EXCLUDED (they score "
        "reconstructions, not label-predicting arms -- a different "
        "object of measurement); Road B annex arms INCLUDED (annex "
        "framing is narrative placement, not a data property; noted as "
        "annex-framed on their rows); the six p7_g label-variant arms "
        "EXCLUDED from the ranking (re-scoring a median/LDL-target arm "
        "against the mean truth scores a label mismatch as model error "
        "-- the different-quantities trap, dated 2026-08-30 beside the "
        "exclusion)"
    ),
    "included": {
        "p7_transformer": {
            "n_patients": 237,
            "seeds": SEEDS_5,
            "runs": (
                ("p7_c0_vit_b16_scut_original_g1", "runs/keeper/p7/p7_c0_vit_b16_scut_original_g1__b954df1e__p7-c0-vit-original-g1"),
                ("p7_c_vit_b16_scut_masked_g1", "runs/keeper/p7/p7_c_vit_b16_scut_masked_g1__4cb62c05__p7-c-vit-g1"),
                ("p7_c_swin_b_scut_masked_g1", "runs/keeper/p7/p7_c_swin_b_scut_masked_g1__4cb62c05__p7-c-swin-g1"),
                ("p7_d1_vit_b16_imagenet_g1", "runs/keeper/p7/p7_d1_vit_b16_imagenet_g1__3f71a6a9__p7-d1-vit-imagenet"),
                ("p7_d1_swin_b_imagenet_g1", "runs/keeper/p7/p7_d1_swin_b_imagenet_g1__3f71a6a9__p7-d1-swin-imagenet"),
                ("p7_d1_swin_b_scut_original_g1", "runs/keeper/p7/p7_d1_swin_b_scut_original_g1__3f71a6a9__p7-d1-swin-original"),
                ("p7_d_vit_b16_imagenet_g2", "runs/keeper/p7/p7_d_vit_b16_imagenet_g2__4cb62c05__p7-d-vit-imagenet"),
                ("p7_d_vit_b16_scut_original_g2", "runs/keeper/p7/p7_d_vit_b16_scut_original_g2__4cb62c05__p7-d-vit-original"),
                ("p7_d_vit_b16_scut_masked_g2", "runs/keeper/p7/p7_d_vit_b16_scut_masked_g2__4cb62c05__p7-d-vit-masked"),
                ("p7_d_swin_b_imagenet_g2", "runs/keeper/p7/p7_d_swin_b_imagenet_g2__4cb62c05__p7-d-swin-imagenet"),
                ("p7_d_swin_b_scut_original_g2", "runs/keeper/p7/p7_d_swin_b_scut_original_g2__4cb62c05__p7-d-swin-original"),
                ("p7_d_swin_b_scut_masked_g2", "runs/keeper/p7/p7_d_swin_b_scut_masked_g2__4cb62c05__p7-d-swin-masked"),
            ),
        },
        "p7_graph": {
            "n_patients": 237,
            "seeds": SEEDS_10,
            "runs": (
                ("p7_d1_srgnn_imagenet_g1_native", "runs/keeper/p7/p7_d1_srgnn_imagenet_g1_native__3f71a6a9__p7-d1-srgnn-imagenet"),
                ("p7_d1_srgnn_scut_original_g1_native", "runs/keeper/p7/p7_d1_srgnn_scut_original_g1_native__3f71a6a9__p7-d1-srgnn-original"),
                ("p7_d1_srgnn_scut_masked_g1_native", "runs/keeper/p7/p7_d1_srgnn_scut_masked_g1_native__3f71a6a9__p7-d1-srgnn-masked"),
                ("p7_d1_agnet_imagenet_g1_native", "runs/keeper/p7/p7_d1_agnet_imagenet_g1_native__3f71a6a9__p7-d1-agnet-imagenet"),
                ("p7_d1_agnet_scut_original_g1_native", "runs/keeper/p7/p7_d1_agnet_scut_original_g1_native__3f71a6a9__p7-d1-agnet-original"),
                ("p7_d1_agnet_scut_masked_g1_native", "runs/keeper/p7/p7_d1_agnet_scut_masked_g1_native__3f71a6a9__p7-d1-agnet-masked"),
                ("p7_d_srgnn_imagenet_g2_native", "runs/keeper/p7/p7_d_srgnn_imagenet_g2_native__4cb62c05__p7-d-srgnn-imagenet"),
                ("p7_d_srgnn_scut_original_g2_native", "runs/keeper/p7/p7_d_srgnn_scut_original_g2_native__4cb62c05__p7-d-srgnn-original"),
                ("p7_d_srgnn_scut_masked_g2_native", "runs/keeper/p7/p7_d_srgnn_scut_masked_g2_native__4cb62c05__p7-d-srgnn-masked"),
                ("p7_d_agnet_imagenet_g2_native", "runs/keeper/p7/p7_d_agnet_imagenet_g2_native__39871f12__p7-d-agnet-imagenet-2"),
                ("p7_d_agnet_scut_original_g2_native", "runs/keeper/p7/p7_d_agnet_scut_original_g2_native__39871f12__p7-d-agnet-original-2"),
                ("p7_d_agnet_scut_masked_g2_native", "runs/keeper/p7/p7_d_agnet_scut_masked_g2_native__39871f12__p7-d-agnet-masked-2"),
                ("p7_e_srgnn_scut_masked_g2_grid", "runs/keeper/p7/p7_e_srgnn_scut_masked_g2_grid__4cb62c05__p7-e-srgnn-grid"),
                ("p7_e_srgnn_scut_masked_g2_anatomy", "runs/keeper/p7/p7_e_srgnn_scut_masked_g2_anatomy__4cb62c05__p7-e-srgnn-anatomy"),
                ("p7_e_srgnn_scut_masked_g2_random", "runs/keeper/p7/p7_e_srgnn_scut_masked_g2_random__f46226fb__p7-e-srgnn-random-2"),
                ("p7_e0_srgnn_imagenet_g2_grid", "runs/keeper/p7/p7_e0_srgnn_imagenet_g2_grid__48e3d710__p7-e0-srgnn-grid"),
                ("p7_e0_srgnn_imagenet_g2_anatomy", "runs/keeper/p7/p7_e0_srgnn_imagenet_g2_anatomy__48e3d710__p7-e0-srgnn-anatomy"),
                ("p7_e0_srgnn_imagenet_g2_random", "runs/keeper/p7/p7_e0_srgnn_imagenet_g2_random__48e3d710__p7-e0-srgnn-random"),
            ),
        },
        "p11_mse_control": {
            "n_patients": 237,
            "seeds": SEEDS_5,
            "note": (
                "included by ruling: the banked pairing is withdrawn, "
                "the predictions are not"
            ),
            "runs": (
                ("p11_mse_control", "runs/keeper/p11/p11_mse_control__27a18e04__p11-mse-control"),
            ),
        },
        "p12_view_ablation": {
            # [VERIFIED 2026-08-30, after run p18-metric-space-2's
            # row-count guard fired] The p12 arms' DOCUMENTED shape is
            # 236, not 237: phase12.STOP_1_MANIFEST derives the view
            # cohort as "cleft_v1 -> drop folder 238 -> 236 rows
            # verbatim" (folder 238 is the single-image patient, no
            # basal), and the arms record says arm A is "the baseline
            # RE-RUN on the 236 cohort. Not reused from the 237 runs --
            # the cohort changed, so the bar must be re-measured on
            # it". All four arms share the 236 both-views cohort. The
            # guard was right to fire; the EXPECTATION was wrong.
            "n_patients": 236,
            "seeds": SEEDS_5,
            "runs": (
                ("p12_arm_a_frontal", "runs/keeper/p12/p12_arm_a_frontal__0dc7c79b__p12-arm-a-frontal"),
                ("p12_arm_b_basal", "runs/keeper/p12/p12_arm_b_basal__0fb33b6a__p12-arm-b-basal"),
                ("p12_arm_c_concat", "runs/keeper/p12/p12_arm_c_concat__0fb33b6a__p12-arm-c-concat"),
                ("p12_arm_d_capacity", "runs/keeper/p12/p12_arm_d_capacity__0dc7c79b__p12-arm-d-capacity"),
            ),
        },
        "p15_mebeauty_probes": {
            "n_patients": 237,
            "seeds": SEEDS_5,
            # [PASS 0, 2026-08-30] the listing of runs/keeper/p15/:
            # one candidate per cell, no ambiguity -- the pretrain_ and
            # extract_ runs at other shas are different tasks. All
            # three share e57a8dea, the commit's sha8 as always.
            "note": (
                "run directories pasted 2026-08-30 (the probe runs were "
                "terminal; nothing had declared them before this phase)"
            ),
            "runs": (
                ("p15_probe_mebeauty_g1", "runs/keeper/p15/p15_probe_mebeauty_g1__e57a8dea__p15-probe-mebeauty-g1"),
                ("p15_probe_mebeauty_g2", "runs/keeper/p15/p15_probe_mebeauty_g2__e57a8dea__p15-probe-mebeauty-g2"),
                ("p15_probe_mebeauty_original", "runs/keeper/p15/p15_probe_mebeauty_original__e57a8dea__p15-probe-mebeauty-original"),
            ),
        },
        "p16_anchor_loop": {
            "n_patients": 237,
            "seeds": SEEDS_5,
            "note": (
                "ONE run, TWO prediction sets: the trained loop "
                "(seed_<n>__predictions.csv) and the identity baseline "
                "(seed_<n>__identity_predictions.csv, deterministic "
                "across seeds by construction -- its own run's "
                "documented shape)"
            ),
            "runs": (
                ("p16_anchor_loop", "runs/keeper/p16/p16_anchor_loop__f342fed9__p16-anchor-loop"),
                ("p16_identity_baseline", "runs/keeper/p16/p16_anchor_loop__f342fed9__p16-anchor-loop"),
            ),
        },
        "p17_tstr": {
            "n_patients": 237,
            "seeds": SEEDS_5,
            "runs": (
                ("p17_arm_a", "runs/keeper/p17/p17_arm_a__bf09bd45__p17-arm-a-2"),
                ("p17_arm_b", "runs/keeper/p17/p17_arm_b__4894169c__p17-arm-b"),
                ("p17_arm_c", "runs/keeper/p17/p17_arm_c__bf09bd45__p17-arm-c"),
            ),
        },
        # [CORRECTED 2026-08-30, after run p18-metric-space-3's seed
        # guard fired] The inventory said "10 seeds" for the WHOLE
        # resolution group -- an over-claim whose source was the record
        # generalising the agnet arm's ten-CSV listing in
        # roadb_p7c_paired_regioncrop.yaml to every Road B arm. The
        # ground truth (the listing, and every arm's own shipped
        # config) is the ladder's seeds-by-regime rule, which Road B
        # follows: vit/swin FIVE seeds, srgnn/agnet TEN. Disk and
        # configs agree arm by arm -- a lock correction, not an
        # artifact-completeness problem. The group is split so each
        # half carries its documented list.
        "roadb_resolution_transformer": {
            "n_patients": 237,
            "seeds": SEEDS_5,
            "note": "annex-framed (ROAD_B_IS_THE_ANNEX) -- noted per row",
            "runs": (
                ("roadb_vit_b16_imagenet_224", "runs/keeper/roadb_p7/roadb_p7_arm_vit_b16_imagenet_224__30cfbfe2__roadb-p7-arm-vit-b16-imagenet-224"),
                ("roadb_vit_b16_imagenet_512", "runs/keeper/roadb_p7/roadb_p7_arm_vit_b16_imagenet_512__30cfbfe2__roadb-p7-arm-vit-b16-imagenet-512"),
                ("roadb_vit_b16_imagenet_768", "runs/keeper/roadb_p7/roadb_p7_arm_vit_b16_imagenet_768__30cfbfe2__roadb-p7-arm-vit-b16-imagenet-768"),
                ("roadb_vit_b16_masked_224", "runs/keeper/roadb_p7/roadb_p7_arm_vit_b16_masked_224__30cfbfe2__roadb-p7-arm-vit-b16-masked-224"),
                ("roadb_vit_b16_masked_512", "runs/keeper/roadb_p7/roadb_p7_arm_vit_b16_masked_512__30cfbfe2__roadb-p7-arm-vit-b16-masked-512"),
                ("roadb_swin_b_imagenet_224", "runs/keeper/roadb_p7/roadb_p7_arm_swin_b_imagenet_224__30cfbfe2__roadb-p7-arm-swin-b-imagenet-224"),
                ("roadb_swin_b_imagenet_512", "runs/keeper/roadb_p7/roadb_p7_arm_swin_b_imagenet_512__30cfbfe2__roadb-p7-arm-swin-b-imagenet-512"),
                ("roadb_swin_b_imagenet_768", "runs/keeper/roadb_p7/roadb_p7_arm_swin_b_imagenet_768__30cfbfe2__roadb-p7-arm-swin-b-imagenet-768"),
                ("roadb_swin_b_masked_224", "runs/keeper/roadb_p7/roadb_p7_arm_swin_b_masked_224__30cfbfe2__roadb-p7-arm-swin-b-masked-224"),
                ("roadb_swin_b_masked_512", "runs/keeper/roadb_p7/roadb_p7_arm_swin_b_masked_512__30cfbfe2__roadb-p7-arm-swin-b-masked-512"),
            ),
        },
        "roadb_resolution_graph": {
            "n_patients": 237,
            "seeds": SEEDS_10,
            "note": "annex-framed -- noted per row",
            "runs": (
                ("roadb_srgnn_imagenet_224", "runs/keeper/roadb_p7/roadb_p7_arm_srgnn_imagenet_224__a54cdfae__roadb-p7-arm-srgnn-imagenet-224"),
                ("roadb_srgnn_imagenet_512", "runs/keeper/roadb_p7/roadb_p7_arm_srgnn_imagenet_512__a54cdfae__roadb-p7-arm-srgnn-imagenet-512"),
                ("roadb_srgnn_imagenet_768", "runs/keeper/roadb_p7/roadb_p7_arm_srgnn_imagenet_768__a54cdfae__roadb-p7-arm-srgnn-imagenet-768"),
                ("roadb_srgnn_masked_224", "runs/keeper/roadb_p7/roadb_p7_arm_srgnn_masked_224__a54cdfae__roadb-p7-arm-srgnn-masked-224"),
                ("roadb_srgnn_masked_512", "runs/keeper/roadb_p7/roadb_p7_arm_srgnn_masked_512__a54cdfae__roadb-p7-arm-srgnn-masked-512"),
                ("roadb_srgnn_masked_768", "runs/keeper/roadb_p7/roadb_p7_arm_srgnn_masked_768__a54cdfae__roadb-p7-arm-srgnn-masked-768"),
                ("roadb_agnet_imagenet_224", "runs/keeper/roadb_p7/roadb_p7_arm_agnet_imagenet_224__a54cdfae__roadb-p7-arm-agnet-imagenet-224"),
                ("roadb_agnet_imagenet_512", "runs/keeper/roadb_p7/roadb_p7_arm_agnet_imagenet_512__a54cdfae__roadb-p7-arm-agnet-imagenet-512"),
                ("roadb_agnet_imagenet_768", "runs/keeper/roadb_p7/roadb_p7_arm_agnet_imagenet_768__a54cdfae__roadb-p7-arm-agnet-imagenet-768"),
                ("roadb_agnet_masked_224", "runs/keeper/roadb_p7/roadb_p7_arm_agnet_masked_224__a54cdfae__roadb-p7-arm-agnet-masked-224"),
                ("roadb_agnet_masked_512", "runs/keeper/roadb_p7/roadb_p7_arm_agnet_masked_512__a54cdfae__roadb-p7-arm-agnet-masked-512"),
                ("roadb_agnet_masked_768", "runs/keeper/roadb_p7/roadb_p7_arm_agnet_masked_768__a54cdfae__roadb-p7-arm-agnet-masked-768"),
            ),
        },
        "roadb_regioncrop": {
            "n_patients": 237,
            "seeds": SEEDS_5,
            "note": "annex-framed -- noted per row",
            "runs": (
                ("roadb_rc_anatomy_concat_vit", "runs/keeper/roadb_p7c/roadb_p7c_arm_anatomy_concat_vit__52875413__roadb-p7c-arm-anatomy-concat-vit-2"),
                ("roadb_rc_random_concat_vit", "runs/keeper/roadb_p7c/roadb_p7c_arm_random_concat_vit__52875413__roadb-p7c-arm-random-concat-vit-2"),
                ("roadb_rc_control_whole_vit", "runs/keeper/roadb_p7c/roadb_p7c_arm_control_whole_vit__52875413__roadb-p7c-arm-control-whole-vit-2"),
            ),
        },
    },
    "excluded": {
        "p7_g_label_variants": (
            "the six median/LDL-target arms (p7_g_vit_b16_scut_masked_g2"
            "_median/_ldl, p7_g1_vit_b16_imagenet_g1_median/_ldl, "
            "p7_g0_vit_b16_imagenet_g2_median/_ldl) -- MSE objective but "
            "a DIFFERENT TARGET; re-scoring them against the mean truth "
            "scores a label mismatch as model error. The "
            "different-quantities trap, dated 2026-08-30"
        ),
        # [NOTED 2026-08-31, original above preserved] The label-mismatch
        # reason is CORRECT for the four G-stage arms that ran. For the
        # two p7_g0_* arms it is not the operative fact: they never ran.
        # `ladder.UNRUN_STAGES == ("G0",)` -- verified at source -- and
        # ladder records "Stage G0 was built and never run". So there
        # are no vectors to re-score, and the prior reason for their
        # absence is ABSENCE, not mismatch. Excluding them for the label
        # is true but secondary, and a reader who took it as the whole
        # reason would believe vectors exist that do not.
        "p7_g0_the_prior_fact_is_absence_2026_08_31": (
            "the two p7_g0_vit_b16_imagenet_g2_median/_ldl arms NEVER "
            "RAN -- ladder.UNRUN_STAGES == ('G0',), verified at source. "
            "The label-mismatch reason above is correct for the FOUR "
            "G-stage arms that ran; for these two the operative fact is "
            "that no vectors exist to re-score. Original reason "
            "preserved; this note is additive"
        ),
        "p13_decodability_probes": (
            "ruled EXCLUDED: they score RECONSTRUCTIONS, not "
            "label-predicting arms -- a different object of measurement"
        ),
        "void_runs": (
            "the three CleftGNN launches ledgered VOID, and p17_arm_a's "
            "first launch at eb5a6887 (pre-fix crash, no surviving CSVs)"
        ),
        "protocol_mismatched": (
            "phase10's faithful CleftGNN arm -- rater-specific models, "
            "85:15 single split, single run; not per-seed OOF on the 237"
        ),
        "not_mse_trained": (
            "p11's IEM-trained arm -- the standing clause fixes the "
            "objective at MSE"
        ),
        "no_per_seed_oof_csvs": (
            "p9 prototype classifier, p4 partition run, all p6/p15/"
            "roadb_p6 pretraining runs"
        ),
        "no_run_in_the_record": (
            "roadb vit_b16/swin_b masked_768 arms -- configs shipped but "
            "their roadb_p6 768 pretrain dependencies were still "
            "PENDING at the last shipped state; no run directory exists "
            "in the record. If the runs exist, adding them is a dated "
            "amendment with the maintainer's paste"
        ),
    },
}


#: **[ADDENDUM 2026-08-31 -- THE LOCK IS NOT
#: REOPENED] Seven arms with banked PCCs that the locked 68 does not
#: contain.**
#:
#: **WHAT THIS IS NOT, first, because it is the part that can be
#: misread.** These arms were **NOT in the locked 68 and the Phase 18
#: analysis did not cover them**. Every Phase 18 figure is a **68-arm
#: object and remains one**: tau(PCC, macro F1) 0.5180, tau(PCC, IEM)
#: 0.3784, tau(macro F1, QWK) 0.7024, the top-5 sets, and D4's 66-of-67
#: adjacent pairs were all computed over 68 arms and none is restated,
#: recomputed or extended here. **This addendum adds VISIBILITY, not
#: coverage.** A reader who quotes a Phase 18 tau as covering 75 arms
#: has misread it; ``NOT_COVERED_BY_PHASE_18`` is the literal that says
#: so.
#:
#: **The seven, with their banked figures and homes:**
#:
#:     arm                     PCC      sd       seeds  home
#:     vit_b32                 0.2519   0.0313   5      ladder.PHASE_7D_OBSERVED
#:     vit_b16 (control)       0.2520   --       5      ladder.PHASE_7D_OBSERVED
#:     vit_b8                  0.0432   0.0340   5      ladder.PHASE_7D_OBSERVED
#:     concat                  0.2594   0.0208   5      ladder.PHASE_7D_OBSERVED
#:     vit_b32_512             0.2280   0.0217   5      ladder.PHASE_7D_OBSERVED
#:     mvitv2_b                0.1844   0.0410   5      ladder.PHASE_7D_OBSERVED
#:     anatomy_concat_srgnn    0.1717   0.0222   10     roadb.REGION_CROP_ARMS_OBSERVED
#:
#: **One of the seven is not a distinct arm, and the audit that found
#: them did not catch this.** ``vit_b16`` is 7D's CONTROL, and
#: ``ladder.PHASE_7D_PATCH_AXIS_REGISTERED["arms"]["vit_b16_224"]``
#: records it as ``"control": "EXISTING -- p7_d1_vit_b16_imagenet_g1 at
#: 0.2520"``. That arm **IS in the locked 68**. So the number of arms
#: genuinely outside the lock is **SIX**, not seven, and it carries no
#: sd of its own because no new run was made for it.
#:
#: **Why the lock was not reopened** -- the ruling, 2026-08-31:
#: a dated addendum, not a reopened lock. ``EXIT_CRITERIA["locked"]``
#: says *"nothing is added after this record; a criterion discovered
#: missing later is a limitation of the lock, recorded as such, never a
#: retro-fitted entry"*, and **that clause is not being broken here**:
#: no criterion, arm or figure enters the Phase 18 analysis. Reopening
#: would mean re-running the analysis and dated-updating every
#: ranking-shaped result in a CLOSED phase -- and the rule that nothing
#: is added after a lock is precisely what stops results being shaped
#: once the numbers exist. The gap is recorded as a limitation of the
#: lock, which is what the clause prescribes.
#:
#: **The gap's origin: NO REASON IS RECORDED.** Verified 2026-08-31 --
#: ``phase18.py`` mentions ``7D``, ``p7d`` and ``anatomy_concat_srgnn``
#: **zero times**, and none of ``ARM_LIST_LOCKED["excluded"]``'s seven
#: keys names any of them. **The absence appears to be oversight**, and
#: it is recorded as oversight rather than given a reason it never had.
#:
#: **They were eligible on their face**, which is what makes it
#: oversight rather than a silent judgement: the five real 7D arms have
#: per-seed OOF prediction CSVs declared in ``configs/p7d_paired.yaml``
#: (40 declared inputs across eight stems, five seeds each), MSE
#: objective, the mean target, the 237 cohort -- the lock's own
#: inclusion shape.
#:
#: **[REASONED, not recorded anywhere] A mechanical explanation for the
#: srgnn one**, offered as a hypothesis and labelled: the lock's
#: ``roadb_regioncrop`` group declares ``seeds: SEEDS_5`` and holds the
#: three ViT arms; ``anatomy_concat_srgnn`` runs **ten** seeds, so it
#: could not have joined that group without a group of its own. That
#: would explain the omission without excusing it, and **no record says
#: this** -- it is inference from the group's shape.
ARM_LIST_ADDENDUM = {
    "added": "2026-08-31, the ruling -- an addendum, not a reopened lock",
    "points_at": "ARM_LIST_LOCKED",
    "what_this_is_not": (
        "these arms were NOT in the locked 68 and the Phase 18 analysis "
        "did NOT cover them. Every Phase 18 figure -- tau(PCC, macroF1) "
        "0.5180, tau(PCC, IEM) 0.3784, tau(macroF1, QWK) 0.7024, the "
        "top-5 sets, D4's 66-of-67 adjacent pairs -- is a 68-ARM OBJECT "
        "and remains one. The addendum adds VISIBILITY, NOT COVERAGE"
    ),
    "arms": {
        "vit_b32": {
            "pcc": 0.2519, "sd": 0.0313, "n_seeds": 5,
            "home": "ladder.PHASE_7D_OBSERVED['arms']['vit_b32']",
        },
        "vit_b16": {
            "pcc": 0.2520, "sd": None, "n_seeds": 5,
            "home": "ladder.PHASE_7D_OBSERVED['arms']['vit_b16']",
            "not_a_distinct_arm": (
                "7D's CONTROL. PHASE_7D_PATCH_AXIS_REGISTERED records it "
                "as 'EXISTING -- p7_d1_vit_b16_imagenet_g1 at 0.2520', "
                "and THAT arm IS in the locked 68. No new run, so no sd "
                "of its own"
            ),
        },
        "vit_b8": {
            "pcc": 0.0432, "sd": 0.0340, "n_seeds": 5,
            "home": "ladder.PHASE_7D_OBSERVED['arms']['vit_b8']",
        },
        "concat": {
            "pcc": 0.2594, "sd": 0.0208, "n_seeds": 5,
            "home": "ladder.PHASE_7D_OBSERVED['arms']['concat']",
        },
        "vit_b32_512": {
            "pcc": 0.2280, "sd": 0.0217, "n_seeds": 5,
            "home": "ladder.PHASE_7D_OBSERVED['arms']['vit_b32_512']",
        },
        "mvitv2_b": {
            "pcc": 0.1844, "sd": 0.0410, "n_seeds": 5,
            "home": "ladder.PHASE_7D_OBSERVED['arms']['mvitv2_b']",
        },
        "anatomy_concat_srgnn": {
            "pcc": 0.1717, "sd": 0.0222, "n_seeds": 10,
            "home": (
                "roadb.REGION_CROP_ARMS_OBSERVED['values']"
                "['anatomy_concat_srgnn']"
            ),
        },
    },
    "genuinely_outside_the_lock": (
        "SIX, not seven -- vit_b16 is 7D's control and is the already-"
        "locked p7_d1_vit_b16_imagenet_g1 under another label"
    ),
    "why_the_lock_was_not_reopened": (
        "the ruling, 2026-08-31: a dated addendum, not a reopened "
        "lock. EXIT_CRITERIA's 'nothing is added after this record' "
        "clause is NOT broken -- no criterion, arm or figure enters the "
        "analysis. Reopening would mean re-running the analysis and "
        "dated-updating every ranking-shaped result in a CLOSED phase, "
        "and the nothing-added rule is precisely what stops results "
        "being shaped once the numbers exist. The gap is recorded as a "
        "LIMITATION OF THE LOCK, which is what that clause prescribes"
    ),
    "the_gaps_origin": (
        "NO REASON IS RECORDED. Verified 2026-08-31: phase18.py mentions "
        "'7D', 'p7d' and 'anatomy_concat_srgnn' ZERO times, and none of "
        "ARM_LIST_LOCKED['excluded']'s seven keys names any of them. The "
        "absence APPEARS TO BE OVERSIGHT and is recorded as oversight "
        "rather than given a reason it never had"
    ),
    "they_were_eligible_on_their_face": (
        "the five real 7D arms have per-seed OOF prediction CSVs "
        "declared in configs/p7d_paired.yaml (40 inputs across eight "
        "stems, five seeds each), MSE objective, the mean target, the "
        "237 cohort -- the lock's own inclusion shape"
    ),
    "a_mechanical_hypothesis_for_the_srgnn_one": (
        "[REASONED, not recorded anywhere] the lock's roadb_regioncrop "
        "group declares seeds SEEDS_5 and holds the three ViT arms; "
        "anatomy_concat_srgnn runs TEN seeds, so it could not have "
        "joined that group without a group of its own. Explains the "
        "omission without excusing it; NO RECORD SAYS THIS"
    ),
    "the_srgnn_arms_run_dir": (
        "its config exists (configs/roadb_p7c_arm_anatomy_concat_srgnn"
        ".yaml, ten seeds) and the run happened -- the value is banked "
        "-- but NO RUN DIRECTORY for it is declared anywhere in src/ or "
        "configs/, and it sits outside REGION_CROP_CONTRASTS, whose "
        "three contrasts are all ViT. Recorded as found"
    ),
}


#: **[2026-08-31] The literal that stops a 68-arm figure being quoted as
#: covering 75.** A tested string, on the pattern of
#: ``DELIVERABLES_REGISTERED["cleftgnn_iem_prohibition"]``.
NOT_COVERED_BY_PHASE_18 = (
    "THE SEVEN ARMS IN ARM_LIST_ADDENDUM WERE NOT ANALYSED BY PHASE 18. "
    "Every figure the phase produced -- tau(PCC, macro F1) 0.5180, "
    "tau(PCC, IEM) 0.3784, tau(macro F1, QWK) 0.7024, the top-5 "
    "overlaps, D4's 66-of-67 adjacent pairs, D5's IEM ladder -- was "
    "computed over the LOCKED 68 and describes those 68 only. None of "
    "them is a statement about 75 arms, and none may be requoted as "
    "one. The addendum records that these arms exist and what they "
    "scored; it does not extend a single ranking, correlation or "
    "overlap to them."
)


def locked_arm_entries():
    """The lock flattened: (name, run_dir, seeds, csv_stem, group)."""
    entries = []
    for group, block in ARM_LIST_LOCKED["included"].items():
        for name, run_dir in block["runs"]:
            stem = ("identity_predictions"
                    if name == "p16_identity_baseline" else "predictions")
            entries.append({
                "name": name, "run_dir": run_dir,
                "seeds": list(block["seeds"]), "csv": stem, "group": group,
                "n_patients": block["n_patients"],
            })
    return entries


#: **[LOCKED 2026-08-30] THE EXIT CRITERIA. Nothing is added after this
#: record.**
EXIT_CRITERIA = {
    "locked": (
        "2026-08-30 -- **nothing is added after this record**; a "
        "criterion discovered missing later is a limitation of the "
        "lock, recorded as such, never a retro-fitted entry"
    ),
    "criteria": (
        "D1: the locked arms scored under both families from banked "
        "per-seed CSVs -- PCC/Spearman; accuracy, macro F1, per-class "
        "F1, QWK-3cat, floors beside; Kendall tau and top-5 overlap "
        "between family rankings; verdict against the three "
        "pre-committed readings including the mechanism-failed third "
        "arm",
        "D2: the criterion re-run with macro F1 on the named seven; "
        "verdict-flip table; banked PCC verdicts untouched",
        "D3: QWK across the locked arms; ranking vs both families; the "
        "registered position on QWK-over-macro-F1 at three classes",
        "D4: the derived floor 0.022908 printed with the table; every "
        "observed difference below it marked uninterpretable",
        "D5: IEM across the locked arms with the constant-predictor "
        "floor beside every figure; tau/top-5 vs both families; the "
        "three defect caveats attached; the CleftGNN prohibition "
        "standing",
        "the all-MSE standing clause on every output",
        "LOCKED as of 2026-08-30, this record's date",
    ),
    "named_seven_note": (
        "the named seven are SEVEN NAMES over SIX unique contrasts: "
        "p17-a-vs-probe is both a condition-split row and a family "
        "member. The flip table has six rows; the seventh name is the "
        "same pair counted from the other list"
    ),
}


#: The six unique contrasts of the named seven, with their banked PCC
#: verdicts -- carried here so the flip table's PCC column is the
#: ledger's, never recomputed.
NAMED_CONTRASTS = (
    {"name": "p16-anchor-loop-vs-probe", "winner": "p16_anchor_loop",
     "baseline": "p7_d1_vit_b16_imagenet_g1", "pcc_verdict": "unresolved"},
    {"name": "p17-a-vs-probe", "winner": "p17_arm_a",
     "baseline": "p7_d1_vit_b16_imagenet_g1", "pcc_verdict": "unresolved"},
    {"name": "p17-b-vs-probe", "winner": "p17_arm_b",
     "baseline": "p7_d1_vit_b16_imagenet_g1",
     "pcc_verdict": "claimable_negative"},
    {"name": "p17-c-vs-probe", "winner": "p17_arm_c",
     "baseline": "p7_d1_vit_b16_imagenet_g1", "pcc_verdict": "unresolved"},
    {"name": "p17-a-vs-c", "winner": "p17_arm_a", "baseline": "p17_arm_c",
     "pcc_verdict": "unresolved"},
    {"name": "p17-b-vs-c", "winner": "p17_arm_b", "baseline": "p17_arm_c",
     "pcc_verdict": "unresolved"},
)


# --------------------------------------------------------------------------
# the metric implementations the task shares
# --------------------------------------------------------------------------


def macro_f1_statistic(truth, predicted) -> float:
    """Macro F1 as a paired_comparison statistic: both sides collapsed
    at the frozen 2.5/3.5 (predictions clipped into range first). The
    phase10 cross-check is asserted ONCE by the suite, not inside the
    10,000-resample loop."""
    from . import classification
    from .eval.metrics import to_3class

    report = classification.prf_report(
        to_3class(truth), to_3class(predicted, clip=True), 3,
        _check_phase10=False,
    )
    return report["f1_macro"]


def iem_score(truth, predicted) -> float:
    """Mean IEM under convention A -- **phase11's own implementation, "
    reused never reimplemented**: the three measured defects were
    measured on that code, and a fresh implementation would detach the
    caveats from the figures they qualify."""
    import numpy as np

    from . import phase11

    residual = np.asarray(predicted, dtype=float) - np.asarray(
        truth, dtype=float
    )
    return float(np.mean(phase11.iem(residual)))


def constant_predictor_iem(truth) -> float:
    """D5's floor: the constant predictor at the panel mean, through
    the SAME function as the arms."""
    import numpy as np

    truth = np.asarray(truth, dtype=float)
    constant = np.full(truth.shape, float(truth.mean()))
    return iem_score(truth, constant)


#: **[COMPLETED 2026-08-30 -- this completes the lock, adds nothing to
#: it] D3's missing half: the QWK ranking and the registered position.**
D3_COMPLETED = {
    "completed": (
        "2026-08-30 -- exit criterion 3 promised 'ranking vs both "
        "families; the registered position'; the first task build "
        "computed qwk_3cat per arm but no ordering, no tau, no top-5, "
        "and no position. **This COMPLETES the locked criterion; it "
        "adds nothing to the lock**"
    ),
    "registered_position_qwk_over_macro_f1": (
        "**Written from what QWK is, BEFORE looking at any tau value "
        "-- marked as such.** QWK is agreement corrected for chance "
        "with quadratic distance weighting: at three classes the "
        "weight matrix has exactly two off-diagonal magnitudes "
        "((i-j)^2/(k-1)^2 with k=3: adjacent errors cost 1/4, "
        "low<->high errors cost 1), so QWK PRICES HOW FAR a "
        "misclassification lands where macro F1 prices only WHETHER. "
        "What it adds over macro F1 here: (i) the distance term -- but "
        "at supports [88, 119, 30] with the middle class dominant, "
        "extreme low<->high confusions have little mass to act on, so "
        "the term is expected mostly idle; (ii) CHANCE CORRECTION "
        "against the marginals, so a majority-class predictor scores "
        "~0 rather than inheriting the 0.502 floor; (iii) it weights "
        "through the marginals rather than averaging classes equally, "
        "so it does NOT inherit macro F1's "
        "3-patient-equals-110-patient weighting and the D4 "
        "single-patient floor does not govern it. **The position: at "
        "these class counts QWK's practical addition over macro F1 is "
        "the chance correction, not the distance pricing -- so it is "
        "expected to order arms closer to accuracy than macro F1 "
        "does.** If the tau values contradict this, the position was "
        "wrong and is recorded as wrong, never adjusted"
    ),
}


#: **[RULED 2026-08-30] D2's HOME: the table, not the ledger.**
D2_HOME_RULED = {
    "ruled": "2026-08-30, grounds recorded",
    "no_new_ledger_rows": (
        "**the flip table gets NO ledger rows.** Grounds: (i) the flip "
        "table is an INSTRUMENT finding -- it measures the criterion, "
        "not the arms; (ii) a ledger row is the unit of ARM-CLAIM, and "
        "a row would create a quotable 'claimable' stripped of its "
        "metric qualifier; (iii) the exit criteria promised a TABLE; "
        "(iv) the chain pins through 37 make entry edits impossible "
        "anyway, so the banked PCC verdicts could not absorb a "
        "qualifier even if one were wanted"
    ),
    "cross_reference_one_directional": (
        "from THIS record to the six ledger entries the flip table "
        "reads against, by name -- p16-anchor-loop-unresolved, "
        "p17-a-vs-probe, p17-b-vs-probe, p17-c-vs-probe, p17-a-vs-c, "
        "p17-b-vs-c -- and never the other direction: the ledger does "
        "not point at the instrument measurement"
    ),
}


#: **[VERIFIED 2026-08-30, IN CODE -- the CSV recomputation lands with
#: run 5] The p17 IEM band's computation path.**
P17_IEM_PATH_VERIFIED = {
    "verified": (
        "2026-08-30, every code step; the direct one-seed CSV "
        "recomputation needs the cluster CSV and lands with run "
        "p18-metric-space-5's new per-arm prediction-stats columns"
    ),
    "the_path": (
        "iem_score(arm_truth_mean, predicted) -> residual = predicted "
        "- truth (y_hat - G, the manuscript's own direction) in GRADE "
        "SPACE via phase11.iem, convention A -- the implementation the "
        "defects were measured on. The CSV is "
        "seed_<n>__predictions.csv, the column is 'prediction', the "
        "truth is the manifest mean sliced to the arm's documented "
        "rows. No unit conversion anywhere in the chain"
    ),
    "the_mechanism_writable": (
        "**writable if run 5's prediction stats confirm it "
        "[REASONED -> MEASURED at run 5]**: arm A trains on the "
        "magnitude-mapped ANCHOR-GRADE scale (labels 1.0/2.714/"
        "3.857/5.0, mean 3.14, spread ~1.5) while the panel mean sits "
        "near 2.96 with spread ~0.55 -- so A can CORRELATE (0.2334) "
        "while sitting on the wrong SCALE, and IEM, a scaled error, "
        "prices the scale mismatch the correlation ignores. B/C's "
        "rounded distance readout piles mass into high grades "
        "(distance >= margin -> grade 5). The per-arm "
        "prediction_mean/sd columns added this turn are what turns "
        "this from reasoning into measurement"
    ),
}


#: **[MEASURED 2026-08-30, run p18_metric_space__9411267e__
#: p18-metric-space-5 -- the citable run; runs 1-4 are the defect audit
#: trail] THE FIVE DELIVERABLES' VERDICTS.**
D1_VERDICT = {
    "verdict": (
        "**DISCORDANT, and the registered mechanism PARTIALLY held -- "
        "recorded with the uncovered kind named, not stretched to "
        "fit.** tau(PCC, macro F1) 0.5180, top-5 overlap 4/5"
    ),
    "covered_movers": (
        "the p16 pair moved exactly as the mechanism predicted: "
        "near-mean predictors with tolerable PCC and collapsed "
        "minority-class F1 -- the identity baseline's macro F1 equals "
        "the 0.2228 constant floor EXACTLY, the addendum's own "
        "arithmetic realised at ladder scale"
    ),
    "uncovered_kind_named": (
        "**p17 B/C at macro F1 0.0785 -- BELOW the constant "
        "predictor's floor** -- are near-constant MINORITY-CLASS "
        "predictors (per-class F1 [0, ~0, 0.225]): mass piled into the "
        "smallest class scores WORSE than predicting the majority, a "
        "collapse mode the registered mechanism (near-MEAN predictors) "
        "did not anticipate. Named per the pre-committed honesty rule: "
        "the mechanism is recorded as partially failed, never adjusted"
    ),
}


D2_MEASURED = {
    "verdict": (
        "**3 of 6 contrasts flip, all in ONE direction -- "
        "PCC-unresolved to F1-claimable**: p16-anchor-loop, "
        "p17-c-vs-probe, and p17-a-vs-c (the latter at 26x its "
        "F1-threshold). **The criterion is METRIC-DEPENDENT on this "
        "cohort** -- what it cannot resolve under the pre-registered "
        "metric it resolves under another"
    ),
    "six_not_seven": (
        "the ruled 'named seven' listed p17-a-vs-probe under both "
        "headings (condition-split row AND family contrast); the "
        "distinct set is SIX, and every figure here counts it once"
    ),
    "home_stands": (
        "D2_HOME_RULED stands as ruled: no ledger rows, the "
        "one-directional cross-reference, banked PCC verdicts "
        "untouched -- the flips are statements about the instrument"
    ),
}


D3_JUDGED = {
    "measured": (
        "tau(PCC, QWK) 0.6295; tau(macro F1, QWK) 0.7024; top-5 "
        "overlap with the F1 ranking 3 of 5"
    ),
    "the_positions_testable_clause": (
        "the registered position's exact wording: 'expected to order "
        "arms closer to accuracy than macro F1 does' -- **judged by "
        "tau(accuracy, QWK) versus tau(macro F1, QWK) = 0.7024, "
        "nothing else**. tau(accuracy, QWK) was not among the task's "
        "computed agreements, so it is DERIVED from run 5's own table "
        "(the pooled accuracy and qwk_3cat means over the 68 rows) via "
        "phase18.tau_accuracy_vs_qwk -- cited as derived-from-run-5"
    ),
    "judgement_rule_pre_committed": (
        "HELD iff tau(accuracy, QWK) > 0.7024; WRONG otherwise, and "
        "wrong-recorded-as-wrong. **The one number awaits the "
        "derivation against run 5's metrics.json (cluster-side); the "
        "rule is fixed here first so the verdict is mechanical when "
        "the number lands**"
    ),
    # [JUDGED 2026-08-30, the maintainer's derivation against run 5's table]
    # The rule above applied mechanically; everything above preserved.
    "judged_wrong_2026_08_30": (
        "**tau(accuracy, QWK) = 0.3392663 (n_arms 68, "
        "derived-from-run-5). By the fixed rule -- HELD iff > 0.7024 "
        "-- THE POSITION IS WRONG, and is recorded as wrong.** The "
        "measured characterisation: QWK sits closer to macro F1 "
        "(0.7024) and to PCC (0.6295) than to accuracy (0.3393) -- "
        "the chance correction against the 0.502-majority marginal "
        "DECOUPLES it from raw accuracy, and the distance term was "
        "NOT idle, inverting the position's expectation on both "
        "counts. **This is also the property behind QWK crowning arm "
        "A** (the exhibit's first verdict): an instrument decoupled "
        "from the majority marginal rewards exactly the arm the "
        "accuracy-shaped instruments do not"
    ),
    "unpredicted_observation": (
        "**recorded as OBSERVATION, not reading: QWK crowns arm A "
        "first of 68** -- the same arm PCC calls "
        "probe-indistinguishable (p17-a-vs-probe UNRESOLVED), macro "
        "F1 ranks mid-field, and IEM ranks LAST. Three instruments, "
        "three verdicts, one arm: the phase's subject in a single row"
    ),
}


D4_MEASURED = {
    "verdict": (
        "**66 of 67 adjacent macro-F1 pairs sit below the derived "
        "0.022908 floor** -- nearly the entire F1 ordering is "
        "uninterpretable at n=1 resolution, and the floor was derived "
        "from the class counts BEFORE the table existed "
        "(MACRO_F1_FLOOR). The D1/D3 tau values over that ordering "
        "carry this bound"
    ),
}


D5_CLOSED = {
    "defect_characterisation_fired": (
        "**the pre-committed characterisation fired on real data: IEM "
        "crowns the IDENTITY BASELINE first of 68** -- "
        "A_FAVOURS_NARROW_PREDICTORS demonstrated at ladder scale, on "
        "the very arm that trains nothing. tau(PCC, IEM) 0.3784 with "
        "top-5 overlap 1; tau(macro F1, IEM) -0.0211. IEM adds no "
        "ordering information over correlation here, and what it adds "
        "instead is the narrow-predictor preference the defects "
        "predicted"
    ),
    "prohibitions_ride": (
        "the CleftGNN prohibition and the three defect caveats ride "
        "every IEM figure, as registered"
    ),
}



def tau_accuracy_vs_qwk(table: dict) -> dict:
    """D3's derived agreement, cited as derived-from-run-5: Kendall
    tau-b between the pooled-accuracy and pooled-QWK orderings of the
    table's arms, through phase11's own tau (no second implementation).
    Run against run 5's metrics.json table block; the judgement rule in
    D3_JUDGED then applies mechanically."""
    from . import phase11

    names = sorted(table)
    accuracy = [table[n]["pooled"]["accuracy"]["mean"] for n in names]
    qwk = [table[n]["pooled"]["qwk_3cat"]["mean"] for n in names]
    tau = phase11.kendall_tau_b(accuracy, qwk)
    return {"tau_accuracy_vs_qwk": float(tau), "n_arms": len(names),
            "source": "derived-from-run-5 table values"}


#: **[MEASURED 2026-08-30 -- P17_IEM_PATH_VERIFIED's mechanism key is
#: preserved as written; this is its measured resolution, and one
#: conflation in it is the NINTH different-quantities catch.]**
P17_IEM_MEASURED = {
    "measured": (
        "run 5's prediction-stats columns landed. **The registered "
        "SPECIFIC mechanism FAILED**: the [REASONED] text predicted "
        "arm A near the 3.14 anchor-scale mean; the measurement is "
        "**1.878 +/- 1.435 against truth 2.7544 +/- 0.6587** -- "
        "UNDER-prediction on a too-wide scale, priced by IEM's "
        "measured heavy under-prediction branch, not the predicted "
        "over-shoot. **The GENERAL claim is measured**: gross scale "
        "miscalibration invisible to PCC, priced by IEM. B/C at "
        "**4.91 +/- 0.30** -- near-constant top-grade, as reasoned. "
        "Failure dated, original preserved, per the standing pattern"
    ),
    "the_ninth_different_quantities_catch": (
        "**the [REASONED] text also said 'the panel mean sits near "
        "2.96' -- a CONFLATION**: 2.96 is the ANCHOR-GRADE mean "
        "(74/25, the 25 references' spread); the cohort's panel-mean "
        "truth is 2.7544. They are different quantities, and the "
        "identity baseline's prediction mean of 2.935 sits near the "
        "ANCHOR mean, not the truth mean -- reading 2.935 as "
        "'well-calibrated to the cohort' would have been the "
        "different-quantities error, caught here for the NINTH time. "
        "Recorded where the 2.935 is discussed so the write-up cannot "
        "inherit the conflation"
    ),
}


#: **[CLOSED 2026-08-30] PHASE 18. THE METRIC-SPACE ABLATION.**
PHASE_18_CLOSING = {
    "closed": (
        "2026-08-30 -- run p18_metric_space__9411267e__"
        "p18-metric-space-5, the CITABLE run; runs 1-4 stand as the "
        "defect audit trail"
    ),
    "criterion_1_d1": (
        "MET: 68 locked arms scored under both families from banked "
        "CSVs, floors beside, tau and top-5 between rankings; verdict "
        "DISCORDANT with the mechanism PARTIALLY held and the "
        "uncovered kind named (D1_VERDICT)"
    ),
    "criterion_2_d2": (
        "MET: the flip table on the six distinct contrasts -- 3 flip, "
        "one direction; the criterion is metric-dependent; banked PCC "
        "verdicts untouched, no ledger rows (D2_MEASURED, "
        "D2_HOME_RULED)"
    ),
    "criterion_3_d3": (
        "MET: QWK ordering, tau vs both families, the pre-committed "
        "position judged by its exact wording -- verdict mechanical on "
        "tau(accuracy, QWK) via the derived-from-run-5 rule "
        "(D3_JUDGED); the three-instruments observation recorded as "
        "observation"
    ),
    "criterion_4_d4": (
        "MET: 66 of 67 adjacent macro-F1 pairs below the "
        "derived-before-the-table 0.022908 floor -- the ordering "
        "uninterpretable at n=1 resolution, marked (D4_MEASURED)"
    ),
    "criterion_5_d5": (
        "MET: IEM across the 68 with the constant-predictor floor, "
        "defect caveats and the CleftGNN prohibition riding; the "
        "characterisation fired -- IEM crowns the identity baseline "
        "(D5_CLOSED); the p17 band resolved by measurement "
        "(P17_IEM_MEASURED)"
    ),
    "criterion_6_all_mse": (
        "MET: the standing clause on every output -- metrics on a "
        "fixed MSE objective; no claim about F1- or IEM-trained arms"
    ),
    "criterion_7_locked": (
        "the lock held: D3's first-build gap was COMPLETED (not "
        "added), nothing else moved after 2026-08-30's lock"
    ),

    "operational_history": (
        "five launches: three load-path defects, each with its "
        "permanent test -- the QWK grade-space contract (the frozen "
        "guard firing on a bad caller), p12's documented 236 (the "
        "expectation fixed, never the guard), the Road B seed-regime "
        "over-claim (the lock corrected, the glob never existed) -- "
        "then run 4 clean but superseded by run 5 for D3-completeness, "
        "noted as such. Three instrument defects, zero data defects, "
        "again"
    ),
    "tau_caveats": (
        "the n_patients caveat rides every tau (four sets rank on "
        "236); the D4 floor bounds every F1-ordering read"
    ),
    "the_exhibit": (
        "**named for the write-up: ONE ARM, THREE INSTRUMENTS, THREE "
        "VERDICTS** -- arm A is QWK's first of 68, PCC's "
        "probe-indistinguishable, macro F1's mid-field, IEM's last. "
        "The registered question answered in a single row: the "
        "instruments do not disagree about quality by degree, they "
        "disagree about WHAT QUALITY IS"
    ),
}


def summary() -> dict:
    """The phase's records so far, importable as one object."""
    return {
        "rulings": PHASE_18_RULINGS,
        "deliverables": DELIVERABLES_REGISTERED,
        "reckoning": PHASE_18_RECKONING,
        "macro_f1_floor": MACRO_F1_FLOOR,
        "arm_inventory_draft": ARM_INVENTORY_DRAFT,
        "arm_list_locked": ARM_LIST_LOCKED,
        "exit_criteria": EXIT_CRITERIA,
        "d3_completed": D3_COMPLETED,
        "d2_home": D2_HOME_RULED,
        "p17_iem_path": P17_IEM_PATH_VERIFIED,
        # [2026-08-31] The lock's addendum -- visibility, not coverage.
        "arm_list_addendum": ARM_LIST_ADDENDUM,
        "d1_verdict": D1_VERDICT,
        "d2_measured": D2_MEASURED,
        "d3_judged": D3_JUDGED,
        "d4_measured": D4_MEASURED,
        "d5_closed": D5_CLOSED,
        "p17_iem_measured": P17_IEM_MEASURED,
        "closing": PHASE_18_CLOSING,
        # [2026-09-01] Three addenda, none of which reopens the phase or
        # moves a figure: the tau orientation the records never stated,
        # the probe that fell into that gap, and the correction's
        # measured downstream impact.
        "iem_tau_orientation": IEM_TAU_ORIENTATION,
        "tau_recomputation_provenance": TAU_RECOMPUTATION_PROVENANCE,
        "tau_b_correction_impact": TAU_B_CORRECTION_IMPACT,
        # [2026-09-01] The locked 68 covers 63 distinct prediction
        # sets -- a limitation of the lock, not a reopening.
        "duplicate_prediction_sets": DUPLICATE_PREDICTION_SETS,
        # [2026-09-01, by hash] Four pairs, not five; the agnet
        # divergence investigated and explained by a 2026-07-31 record.
        "duplicate_count_corrected": DUPLICATE_COUNT_CORRECTED,
        "agnet_divergence": AGNET_DIVERGENCE_INVESTIGATED,
        "d4_remeasured_at_four": D4_REMEASURED_AT_FOUR,
        "duplicate_dependencies": DUPLICATE_DEPENDENCIES,
        # [2026-09-01] The thread closed: duplication proven at every
        # seed, AG-Net differing at every seed, D4 resolved, and the
        # limitation stated for the write-up.
        "duplication_proven_at_depth": DUPLICATION_PROVEN_AT_DEPTH,
        "agnet_differs_on_all_ten": AGNET_DIFFERS_ON_ALL_TEN,
        "duplication_limitation": DUPLICATION_LIMITATION_FOR_THE_WRITE_UP,
    }


#: **[NOTE 2026-09-01 -- HYGIENE, NOT A CORRECTION. NO FIGURE CHANGES,
#: AND THE PHASE IS NOT REOPENED.] IEM'S ORIENTATION IN THE TAU
#: COMPUTATION.**
#:
#: **All four rankings are BEST-FIRST.** ``run.task_metric_space_analysis``
#: sorts descending for the three SCORES and ascending for IEM, which is
#: an ERROR (``run.py:5980``)::
#:
#:     by_pcc = sorted(names, key=lambda n: -table[n]["pooled"]["pcc"]["mean"])
#:     by_f1  = sorted(names, key=lambda n: -table[n]["pooled"]["macro_f1"]["mean"])
#:     by_iem = sorted(names, key=lambda n:  table[n]["pooled"]["iem"]["mean"])
#:     by_qwk = sorted(names, key=lambda n: -table[n]["pooled"]["qwk_3cat"]["mean"])
#:
#: **The minus sign is present on the three scores and absent on IEM.**
#: That absence IS the orientation.
#:
#: **Tau is then computed on RANK POSITION, never on raw values**
#: (``run.py:5989``)::
#:
#:     def agreement(order_a, order_b):
#:         rank_a = {n: i for i, n in enumerate(order_a)}
#:         rank_b = {n: i for i, n in enumerate(order_b)}
#:         tau = kendall_tau_b([rank_a[n] for n in names],
#:                             [rank_b[n] for n in names])
#:
#: **THE PROJECT'S STATED PRINCIPLE**, quoted from ``phase11.ranks``'s
#: docstring: *"IEM and MAE are ERRORS -- smaller is better -- so
#: ascending is the good-first order. PCC is inverted by the caller
#: rather than here, so the direction of every ranking is visible at its
#: call site."* This task honours it exactly: the direction is visible at
#: its call site, as one minus sign.
#:
#: **BOTH TAU PATHS CARRY THE CONVENTION INDEPENDENTLY.** ``agreement``
#: here, and ``run._rank_block`` (the Phase 11 IEM comparison), which
#: ranks IEM and MAE directly and negates PCC with the comment *"PCC is a
#: similarity: negate so every ranking here is best-first."* Two
#: implementations written at different times, same orientation.
#:
#: **THE CONSEQUENCE, STATED PLAINLY: recomputing any IEM tau from the
#: banked table WITHOUT orienting it produces an EXACT SIGN FLIP** -- not
#: an approximate one. Measured on constructed data: the task convention
#: and a raw-value computation differ by ``tau_task + tau_raw = 0.00e+00``
#: for both IEM pairs, and are IDENTICAL for PCC-vs-macro-F1, where there
#: is no orientation to lose.
#:
#: **Why this note exists**: the convention lives in the CODE -- this
#: task's call site and ``phase11.ranks``'s docstring -- and in **none of
#: this phase's banked records**. A reader recomputing from
#: ``metrics.json`` has nothing in the record telling them to orient it.
#: That gap was found, not hypothesised: see
#: ``TAU_RECOMPUTATION_PROVENANCE``.
IEM_TAU_ORIENTATION = {
    "noted": (
        "2026-09-01 -- HYGIENE, NOT A CORRECTION. No figure changes and "
        "the phase is not reopened; this is an addendum on the "
        "ARM_LIST_ADDENDUM pattern"
    ),
    "all_four_rankings_are_best_first": (
        "run.task_metric_space_analysis sorts DESCENDING for the three "
        "SCORES (PCC, macro F1, QWK) and ASCENDING for IEM, which is an "
        "ERROR (run.py:5980). **The minus sign is present on the three "
        "scores and absent on IEM; that absence IS the orientation**"
    ),
    "tau_is_computed_on_rank_position": (
        "run.py:5989 -- `agreement` builds {name: index} over each "
        "best-first order and passes those POSITIONS to kendall_tau_b. "
        "Raw values never reach the statistic"
    ),
    "the_projects_stated_principle": (
        "phase11.ranks's docstring: 'IEM and MAE are ERRORS -- smaller "
        "is better -- so ascending is the good-first order. PCC is "
        "inverted by the caller rather than here, so the direction of "
        "every ranking is visible at its call site.' This task honours "
        "it exactly: the direction is visible at its call site, as one "
        "minus sign"
    ),
    "both_tau_paths_carry_it_independently": (
        "`agreement` here, and run._rank_block (the Phase 11 IEM "
        "comparison), which ranks IEM and MAE directly and negates PCC "
        "with the comment 'PCC is a similarity: negate so every ranking "
        "here is best-first'. Two implementations written at different "
        "times, same orientation"
    ),
    "the_consequence": (
        "**recomputing any IEM tau from the banked table WITHOUT "
        "orienting it produces an EXACT SIGN FLIP** -- not an "
        "approximate one. Measured on constructed data: tau_task + "
        "tau_raw = 0.00e+00 for both IEM pairs, and the two methods are "
        "IDENTICAL for PCC-vs-macro-F1, where there is no orientation to "
        "lose"
    ),
    "why_this_note_exists": (
        "the convention lives in the CODE -- this task's call site and "
        "phase11.ranks's docstring -- and in NONE of this phase's banked "
        "records. A reader recomputing from metrics.json has nothing in "
        "the record telling them to orient it. The gap was FOUND, not "
        "hypothesised: TAU_RECOMPUTATION_PROVENANCE"
    ),
    "no_figure_changes": (
        "tau(PCC, IEM) +0.3784 and tau(macro F1, IEM) -0.0211 stand as "
        "banked, and D5_CLOSED's reading -- 'IEM adds no ordering "
        "information over correlation here' -- is unaffected"
    ),
}


#: **[RECORDED 2026-09-01, DATED] THE TAU RECOMPUTATION PROBE, and what
#: it got wrong.** Filed beside ``ladder.THE_ERROR_PROVENANCE``,
#: ``phase20.CROSS_TARGET_ERROR_PROVENANCE``,
#: ``phase20.S_DESCRIPTION_ERROR_PROVENANCE``,
#: ``phase12.TWO_VIEW_CLAIM_PROVENANCE`` and
#: ``literature.NADEAU_BENGIO_DOES_NOT_APPLY``.
#:
#: **What happened.** On **2026-08-31** a one-liner was written to
#: recompute this phase's five tau values on the corrected
#: ``kendall_tau_b``. It read raw ``pooled['iem']['mean']`` in ascending
#: order and **omitted the orientation the task encodes**, producing two
#: apparent sign flips -- tau(PCC, IEM) +0.3784 -> -0.377309 and
#: tau(macro F1, IEM) -0.0211 -> +0.022432 -- which were then **briefly
#: read as a possible finding**.
#:
#: **THE BANKED VALUES WERE CORRECT THROUGHOUT.** Nothing needed
#: correcting; the probe needed orienting.
#:
#: **The defect in the probe, named**: it was **a second computation
#: path built in conversation rather than a reuse of the shipped one**.
#: That is **the two-implementations defect in a new place** -- the same
#: shape as ``median_by_patient``'s *"a second copy of a label resolution
#: is how two runs quietly measure against different labels"*, and the
#: shape ``tau_accuracy_vs_qwk`` was deliberately written to avoid
#: (*"through phase11's own tau -- no second implementation"*). A
#: one-liner in a chat window is a second implementation with no
#: docstring, no test and no review.
#:
#: **The underlying gap, which is not the probe's fault**: the
#: orientation convention was **absent from the record the probe read**.
#: It is in the code and in ``phase11.ranks``'s docstring; it was in no
#: banked Phase 18 record until ``IEM_TAU_ORIENTATION``.
#:
#: **What it touched**: nothing measured, and no verdict. It was caught
#: by reading the task before accepting the number.
TAU_RECOMPUTATION_PROVENANCE = {
    "recorded": "2026-09-01",
    "what_happened": (
        "on 2026-08-31 a one-liner was written recomputing this "
        "phase's five tau values on the corrected kendall_tau_b. It read "
        "raw pooled['iem']['mean'] in ascending order and OMITTED THE "
        "ORIENTATION THE TASK ENCODES, producing two apparent sign flips "
        "-- tau(PCC, IEM) +0.3784 -> -0.377309 and tau(macro F1, IEM) "
        "-0.0211 -> +0.022432 -- which were then briefly read as a "
        "possible finding"
    ),
    "the_banked_values_were_correct_throughout": (
        "nothing needed correcting; the probe needed orienting"
    ),
    "the_defect_in_the_probe": (
        "**a second computation path built in conversation rather than a "
        "reuse of the shipped one -- the two-implementations defect in a "
        "new place.** The same shape as median_by_patient's 'a second "
        "copy of a label resolution is how two runs quietly measure "
        "against different labels', and the shape tau_accuracy_vs_qwk "
        "was deliberately written to avoid ('through phase11's own tau "
        "-- no second implementation'). **A one-liner in a chat window "
        "is a second implementation with no docstring, no test and no "
        "review**"
    ),
    "the_underlying_gap": (
        "the orientation convention was ABSENT FROM THE RECORD THE PROBE "
        "READ. It is in the code and in phase11.ranks's docstring; it "
        "was in no banked Phase 18 record until IEM_TAU_ORIENTATION. "
        "**Not the probe's fault, and the reason the note exists**"
    ),
    "what_it_touched": (
        "nothing measured, and no verdict. Caught by reading the task "
        "before accepting the number"
    ),
    # -----------------------------------------------------------------
    # [2026-09-01, later the same day] THE SECOND SYMPTOM.
    #
    # The probe's three non-IEM "moves" were not defect drift either.
    # `agreement` computes on RANK POSITIONS; the probe computed on RAW
    # VALUES. That single wrong premise produced BOTH symptoms at once.
    # -----------------------------------------------------------------
    "one_wrong_premise_two_symptoms": (
        "**ONE written for the check one-liner, ONE wrong premise -- raw "
        "values where the task uses ranks -- and TWO symptoms.** (i) A "
        "SIGN FLIP on the oriented IEM column, because raw IEM runs "
        "low-is-good against raw PCC's high-is-good. (ii) PHANTOM DRIFT "
        "on the three score columns, because raw values carry ties that "
        "rank positions cannot, so the corrected function moved numbers "
        "the shipped task would never have moved. **Neither symptom was "
        "a finding; both were the same mistake, made twice in one "
        "command**"
    ),
    "both_are_the_same_defect_in_a_new_place": (
        "the two-implementations defect appearing IN A CHAT MESSAGE "
        "rather than in code. The shipped path was available and "
        "correct; a one-liner reproduced it from memory and got the "
        "input representation wrong. **That is what a second "
        "implementation costs, and it costs it whether or not the "
        "second one is committed**"
    ),
    "filed_beside": (
        "ladder.THE_ERROR_PROVENANCE, "
        "phase20.CROSS_TARGET_ERROR_PROVENANCE, "
        "phase20.S_DESCRIPTION_ERROR_PROVENANCE, "
        "phase12.TWO_VIEW_CLAIM_PROVENANCE, "
        "literature.NADEAU_BENGIO_DOES_NOT_APPLY"
    ),
}


#: **[MEASURED 2026-09-01] THE TAU-B CORRECTION'S DOWNSTREAM IMPACT ON
#: THIS PHASE -- and one item that does NOT reconcile.**
#:
#: ``phase21.TAU_B_DEFECT_CORRECTED`` recorded that the correction can
#: only grow ``|tau|`` and that Phase 18's figures were therefore lower
#: bounds, with the size of the gap unmeasured. **The size is now
#: measured for one figure and REPORTED-BUT-UNRECONCILED for three.**
#:
#: **THE DIRECTION IS PROVEN, not argued.** With ``x = C + D``, ``a`` the
#: pairs tied in A only, ``b`` tied in B only and ``t`` tied in BOTH::
#:
#:     old denominator^2  = (x + t + a)(x + t + b)
#:     true denominator^2 = (x + b)(x + a)
#:     difference         = t * (2x + t + a + b)  >= 0
#:
#: So the old denominator is **always** >= the true one, and ``|tau|``
#: can only **grow** under correction. Brute-forced over 69,120
#: combinations with no counterexample, and measured on 400 tied draws:
#: **365 grew, 0 shrank, 35 unchanged.**
#:
#: **RECONCILES**: tau(accuracy, QWK) **0.3392663 -> 0.3400134**, a rise
#: of +0.0007471. Consistent with the proof.
#:
#: **DOES NOT RECONCILE AS STATED**: the three supplied as moves of
#: **-0.001176** (PCC/macroF1), **-0.000954** (PCC/QWK) and **-0.000684**
#: (macroF1/QWK). Applied as written they make ``|tau|`` SHRINK, which
#: the proof forbids. **Read instead as (banked - corrected) -- i.e. the
#: corrected values are HIGHER by those amounts -- all three become
#: consistent**: 0.519176, 0.630454, 0.703084.
#:
#: **The three are therefore NOT BANKED as corrected values here.** Only
#: their magnitudes and the sign question are recorded, pending
#: confirmation of the convention. Banking a number that contradicts a
#: proven property would be the error this whole cycle exists to catch.
#:
#: **NO VERDICT CHANGES, EITHER WAY.** D3's rule is *HELD iff
#: tau(accuracy, QWK) > 0.7024*; the corrected 0.3400134 misses it by
#: **0.3624**, so **WRONG stands** and no plausible reading of the other
#: three moves it. The two IEM figures' apparent moves were the
#: orientation artifact (``IEM_TAU_ORIENTATION``), not drift.
TAU_B_CORRECTION_IMPACT = {
    "measured": "2026-09-01",
    "what_phase_21_recorded": (
        "phase21.TAU_B_DEFECT_CORRECTED: the correction can only GROW "
        "|tau|, so Phase 18's figures were LOWER BOUNDS, with the size "
        "of the gap unmeasured -- 'probably few and the attenuation "
        "small. Probably is not measured'"
    ),
    "the_direction_is_proven": (
        "with x = C+D, a = tied in A only, b = tied in B only, t = tied "
        "in BOTH: old denominator^2 = (x+t+a)(x+t+b), true = (x+b)(x+a), "
        "difference = t*(2x + t + a + b) >= 0. The old denominator is "
        "ALWAYS >= the true one, so |tau| can only GROW. Brute-forced "
        "over 69,120 combinations with no counterexample; measured on "
        "400 tied draws: 365 grew, 0 shrank, 35 unchanged"
    ),
    "reconciles": {
        "tau_accuracy_vs_qwk": {
            "banked": 0.3392663,
            "corrected": 0.3400134,
            "move": 0.0007471,
            "consistent_with_the_proof": True,
        },
    },
    # -----------------------------------------------------------------
    # [SUPERSEDED 2026-09-01, later the same day -- REFUTED, and the
    # original is preserved below exactly as written.]
    #
    # This record first offered the (banked - corrected) reinterpretation
    # as the only reading that made the printed numbers consistent with
    # the proof. **That reinterpretation is REFUTED.** The three non-IEM
    # figures could not have moved IN EITHER DIRECTION -- see
    # the_three_could_not_have_moved. The numbers themselves came from a
    # probe with a wrong premise, which is why no reading of them was
    # ever going to be right.
    #
    # **No number was banked from it; only the reasoning is withdrawn.**
    # -----------------------------------------------------------------
    "does_not_reconcile_as_stated": (
        "[SUPERSEDED 2026-09-01 -- REFUTED, see "
        "the_three_could_not_have_moved] **the three supplied as moves "
        "of -0.001176 (PCC/macroF1), -0.000954 (PCC/QWK) and -0.000684 "
        "(macroF1/QWK). Applied as written they make |tau| SHRINK, which "
        "the proof forbids.** Read instead as (banked - corrected) -- "
        "the corrected values HIGHER by those amounts -- all three "
        "become consistent: 0.519176, 0.630454, 0.703084"
    ),
    "why_that_reinterpretation_was_offered_and_why_it_was_wrong": (
        "**it was the only reading that made the printed numbers "
        "consistent with the proof.** It was wrong because THE NUMBERS "
        "THEMSELVES CAME FROM A PROBE WITH A WRONG PREMISE, so no "
        "reading of them could have been right. The lesson: when a "
        "figure contradicts a proof, reconciling the FIGURE is the "
        "second move -- the first is asking where the figure came from"
    ),
    "the_three_could_not_have_moved": (
        "**`agreement` passes RANK POSITIONS -- permutations of 0..67, "
        "TIE-FREE BY CONSTRUCTION -- so the doubly-tied term is zero and "
        "the old and true denominators are IDENTICAL.** Measured at that "
        "exact shape: max |old - corrected| = 0.000e+00 over 300 trials, "
        "with tie counts a-only 0, b-only 0, BOTH 0. The defect is INERT "
        "for every tau `agreement` computes, which is five of the six"
    ),
    "accuracy_ties_alone_are_insufficient": (
        "**the defect fires only on pairs tied in BOTH columns.** "
        "Measured: 33 accuracy-only ties with ZERO doubly-tied moved the "
        "value by exactly 0.0000000. This is why the raw-value tau moved "
        "at all -- duplicate runs tie EVERY metric simultaneously"
    ),
    "the_three_are_not_banked_here": (
        "only their magnitudes and the sign question are recorded, "
        "pending confirmation of the convention. **Banking a number that "
        "contradicts a proven property would be the error this cycle "
        "exists to catch** [2026-09-01: and the caution was right for "
        "the wrong reason -- there was no convention to confirm, because "
        "the three never moved]"
    ),
    "no_verdict_changes_either_way": (
        "D3's rule is HELD iff tau(accuracy, QWK) > 0.7024; the "
        "corrected 0.3400134 misses it by 0.3624, so **WRONG STANDS** "
        "and no plausible reading of the other three moves it. The two "
        "IEM figures' apparent moves were the ORIENTATION ARTIFACT "
        "(IEM_TAU_ORIENTATION), not drift"
    ),
    "what_this_upgrades": (
        "phase21.TAU_B_DEFECT_CORRECTED's 'probably few and the "
        "attenuation small' becomes MEASURED for tau(accuracy, QWK) at "
        "+0.0007471, and the one-directional argument becomes a PROOF "
        "rather than an argument. [2026-09-01: the closing clause 'the "
        "other three await the sign convention' is SUPERSEDED -- they "
        "await nothing; they never moved]"
    ),
    "the_defect_affected_exactly_one_banked_figure": (
        "**tau(accuracy, QWK), 0.3392663 -> 0.3400134, an increase of "
        "+0.0007471 in the direction the proof requires. THE OTHER FIVE "
        "COULD NOT HAVE MOVED.** The mechanism is which vectors reach "
        "the statistic: tau_accuracy_vs_qwk computes on RAW POOLED "
        "VALUES; `agreement` computes on RANK POSITIONS"
    ),
    "the_doubly_tied_count": (
        "**t = 5 pairs tied in BOTH accuracy and QWK.** Uniquely pinned "
        "by inverting the observed ratio^2 = 1.004409 against n0 = 2278: "
        "t = 4 reaches at most 1.004151 and t = 6 begins at 1.005289, so "
        "no other value can produce the move at any number of "
        "accuracy-only ties. Consistent with DUPLICATE RUNS, which tie "
        "every metric simultaneously"
    ),
}


#: **[MEASURED 2026-09-01, ON THE CLUSTER -- A LIMITATION OF THE LOCK,
#: RECORDED AS SUCH. NOT A REOPENING, AND NO FIGURE MOVES.] THE LOCKED
#: 68 CONTAINS FIVE DOUBLED PREDICTION SETS.**
#:
#: **The count is now MEASURED, not inferred.** The inversion in
#: ``TAU_B_CORRECTION_IMPACT`` predicted **t = 5** pairs tied in both
#: accuracy and QWK, uniquely, from the observed ratio alone. **A direct
#: count on the cluster returned exactly 5 and named them** -- the
#: prediction was confirmed independently, by a different route, on the
#: real table.
#:
#: **The five pairs, each a Road A g1 arm and a Road B 224 arm:**
#:
#:     p7_c_swin_b_scut_masked_g1      | roadb_swin_b_masked_224
#:     p7_d1_agnet_imagenet_g1_native  | roadb_agnet_imagenet_224
#:     p7_d1_srgnn_imagenet_g1_native  | roadb_srgnn_imagenet_224
#:     p7_d1_swin_b_imagenet_g1        | roadb_swin_b_imagenet_224
#:     p7_d1_vit_b16_imagenet_g1       | roadb_vit_b16_imagenet_224
#:
#: **THE BANKED RECORDS CORROBORATE, four exactly and one to rounding.**
#: ``ladder.STAGE_D1_AT_G1["cells"]`` against
#: ``roadb.PHASE_7_TWENTY_TWO_ARMS["values"][...][224]``::
#:
#:     swin_b scut_masked   0.1327 (sd 0.0267) | 0.1327 (sd 0.0270)  MATCH
#:     agnet  imagenet      0.0613 (sd 0.0426) | 0.0612 (sd 0.0430)  0.0001
#:     srgnn  imagenet      0.1719 (sd 0.0630) | 0.1719 (sd 0.0630)  MATCH
#:     swin_b imagenet      0.1076 (sd 0.0217) | 0.1076 (sd 0.0220)  MATCH
#:     vit_b16 imagenet     0.2520 (sd 0.0148) | 0.2520 (sd 0.0150)  MATCH
#:
#: The single 0.0001 (agnet) is consistent with two records rounding one
#: underlying value, and the pair ties EXACTLY on accuracy and QWK in the
#: run table, which is the measurement that matters here.
#:
#: **WHAT IS NOT SETTLED, and it is the actual question.** Whether these
#: are the SAME PREDICTIONS under two names is settled exactly by
#: comparing ``predictions_sha256`` per seed across each pair. **That was
#: not done here** -- the run artifacts are CLUSTER-ONLY and unreachable
#: from this machine. Equal PCC to four decimals is strong and is not
#: proof. **One command on the cluster would settle it.**
#:
#: **THE IMPACT, MEASURED -- AND IT IS NOT UNIFORMLY NEGLIGIBLE.**
#: Duplicates hold identical values, so a stable sort places them
#: ADJACENTLY in every ordering, in ``names`` order. Simulated at the
#: real shape (63 distinct + 5 doubled vs 63 distinct), 300 draws:
#:
#: * **D4 IS AFFECTED, DETERMINISTICALLY.** Each duplicate pair
#:   contributes one adjacent pair with a gap of **exactly zero**, which
#:   is trivially below any floor. Difference in the below-floor count:
#:   **exactly +5, in every one of 300 draws.** So D4's *"66 of 67
#:   adjacent macro-F1 pairs sit below the floor"* is **61 of 62 among
#:   distinct arms**. **The figure does not change and the reading
#:   survives overwhelmingly** -- but five of the sixty-six are
#:   uninterpretable TRIVIALLY (one run counted twice) rather than
#:   INFORMATIVELY (two arms too close to separate), and that distinction
#:   belongs beside the figure.
#: * **The tau values: small in expectation, NOT provably negligible.**
#:   tau(68 with duplicates) - tau(63 distinct) has mean **+0.00231**,
#:   sd 0.01979, and reaches **0.05503** across draws. Each duplicate
#:   pair is concordant in every ranking by construction, biasing tau
#:   upward. **The actual effect on the banked taus cannot be stated
#:   without recomputing on the real table**, and is not asserted here.
#: * **Top-5 overlaps can shift by up to 2** (mean -0.073, range -2 to
#:   +1). D5 reports top-5 overlap **1**; if a duplicate pair occupies
#:   two slots, that top five covers **four distinct prediction sets**.
#: * **D5's ladder position is unaffected**: IEM crowns the identity
#:   baseline, which is not one of the ten arms involved.
#:
#: **WHY THIS IS A LIMITATION AND NOT A DEFECT.** The lock enumerated
#: RUNS, and every one of the 68 is a real run with real CSVs. Nothing
#: was double-counted by mistake; two roads legitimately produced the
#: same configuration and the lock recorded both. **What was never
#: stated is that the 68 covers 63 distinct prediction sets**, and a
#: reader treating 68 as 68 independent measurements is reading more
#: independence than the set contains.
DUPLICATE_PREDICTION_SETS = {
    "measured": (
        "2026-09-01, on the cluster -- a LIMITATION OF THE LOCK, "
        "recorded as such. NOT a reopening, and no figure moves"
    ),
    "the_count_is_measured_not_inferred": (
        "[t = 5 STANDS as a count of accuracy/QWK ties; the DUPLICATION "
        "reading it carried is corrected 2026-09-01 -- see "
        "DUPLICATE_COUNT_CORRECTED: four pairs, not five] "
        "**the inversion in TAU_B_CORRECTION_IMPACT predicted t = 5 "
        "uniquely, from the observed ratio alone. A direct count on the "
        "cluster returned exactly 5 and named them.** The prediction was "
        "CONFIRMED INDEPENDENTLY, by a different route, on the real "
        "table"
    ),
    "the_five_pairs": (
        ("p7_c_swin_b_scut_masked_g1", "roadb_swin_b_masked_224"),
        ("p7_d1_agnet_imagenet_g1_native", "roadb_agnet_imagenet_224"),
        ("p7_d1_srgnn_imagenet_g1_native", "roadb_srgnn_imagenet_224"),
        ("p7_d1_swin_b_imagenet_g1", "roadb_swin_b_imagenet_224"),
        ("p7_d1_vit_b16_imagenet_g1", "roadb_vit_b16_imagenet_224"),
    ),
    "each_is_a_road_a_g1_arm_and_a_road_b_224_arm": (
        "the pattern is systematic, not coincidental: Road B's 224 "
        "resolution cell and Road A's G1 cell are the same "
        "configuration reached down two roads"
    ),
    "the_banked_records_corroborate": (
        "[SUPERSEDED IN PART 2026-09-01: the 0.0001 agnet gap was REAL, "
        "not rounding -- DUPLICATE_COUNT_CORRECTED] "
        "ladder.STAGE_D1_AT_G1['cells'] against "
        "roadb.PHASE_7_TWENTY_TWO_ARMS['values'][...][224]: FOUR match "
        "EXACTLY at the banked precision (swin_b masked 0.1327, srgnn "
        "imagenet 0.1719, swin_b imagenet 0.1076, vit_b16 imagenet "
        "0.2520, sds agreeing to rounding), and agnet imagenet differs "
        "by 0.0001 (0.0613 vs 0.0612) -- consistent with two records "
        "rounding one underlying value. The pair ties EXACTLY on "
        "accuracy and QWK in the run table, which is the measurement "
        "that matters"
    ),
    "what_is_not_settled": (
        "[SETTLED 2026-09-01 at seed 1337 -- DUPLICATE_COUNT_CORRECTED: "
        "four identical, agnet DIFFERENT. Seeds 2024/7/99/12345 remain "
        "uncompared -- DUPLICATE_DEPENDENCIES] "
        "**whether these are the SAME PREDICTIONS under two names.** "
        "That is settled exactly by comparing predictions_sha256 PER "
        "SEED across each pair. **NOT DONE HERE** -- the run artifacts "
        "are CLUSTER-ONLY and unreachable from this machine. Equal PCC "
        "to four decimals is strong and is NOT PROOF. One command on the "
        "cluster would settle it"
    ),
    "impact_d4_is_affected_deterministically": (
        "[RE-MEASURED 2026-09-01 at four duplicates -- D4_REMEASURED_AT_FOUR] "
        "**each duplicate pair contributes one adjacent pair with a gap "
        "of EXACTLY ZERO, trivially below any floor. Measured "
        "difference: +5 in every one of 300 draws.** So D4's '66 of 67 "
        "adjacent macro-F1 pairs sit below the floor' is 61 OF 62 AMONG "
        "DISTINCT ARMS. **The figure does not change and the reading "
        "survives overwhelmingly** -- but five of the sixty-six are "
        "uninterpretable TRIVIALLY (one run counted twice) rather than "
        "INFORMATIVELY (two arms too close to separate)"
    ),
    "impact_on_tau_is_small_but_not_provably_negligible": (
        "tau(68 with duplicates) - tau(63 distinct): mean +0.00231, sd "
        "0.01979, reaching 0.05503 across 300 draws. Each duplicate pair "
        "is CONCORDANT IN EVERY RANKING by construction, biasing tau "
        "upward. **The actual effect on the banked taus cannot be stated "
        "without recomputing on the real table, and is NOT ASSERTED "
        "HERE** -- the expectation was that duplicates would be "
        "immaterial, and the measurement does not support saying so "
        "flatly"
    ),
    "impact_on_top_5_overlaps": (
        "can shift by up to 2 (mean -0.073, range -2 to +1). D5 reports "
        "top-5 overlap 1; if a duplicate pair occupies two slots, that "
        "top five covers FOUR DISTINCT PREDICTION SETS"
    ),
    "impact_on_d5_ladder_position": (
        "unaffected: IEM crowns the IDENTITY BASELINE, which is not one "
        "of the ten arms involved"
    ),
    "why_it_is_a_limitation_not_a_defect": (
        "[63 -> 64 distinct sets, corrected 2026-09-01] "
        "**the lock enumerated RUNS, and every one of the 68 is a real "
        "run with real CSVs.** Nothing was double-counted by mistake; "
        "two roads legitimately produced the same configuration and the "
        "lock recorded both. What was NEVER STATED is that the 68 covers "
        "**63 distinct prediction sets**, and a reader treating 68 as 68 "
        "independent measurements is reading more independence than the "
        "set contains"
    ),
    "figures_computed_over_the_68_with_duplicates_included": (
        "the three tau matrices (D1, D3, D5), the top-5 overlaps, D4's "
        "66-of-67 adjacent pairs, and D5's IEM ladder position. All "
        "remain as banked; this record states what they were computed "
        "over, not a new value for any of them"
    ),
    "not_a_reopening": (
        "on the ARM_LIST_ADDENDUM pattern -- visibility, not coverage. "
        "EXIT_CRITERIA is untouched, no figure moves, and no ledger row "
        "is added"
    ),
}


#: **[CORRECTED 2026-09-01, BY HASH COMPARISON -- the count was FIVE and
#: is FOUR.]** ``DUPLICATE_PREDICTION_SETS``'s reading is corrected in
#: place below; the original stands with it.
#:
#: **Measured, seed 1337, ``fingerprint.predictions_sha256``:**
#:
#:     vit_b16 imagenet   IDENTICAL
#:     swin_b  masked     IDENTICAL
#:     swin_b  imagenet   IDENTICAL
#:     srgnn   imagenet   IDENTICAL
#:     agnet   imagenet   **DIFFERENT** (fc5443061acdb181... vs
#:                        987aece1ab0303f3...)
#:
#: **THE LOCK'S 68 COVERS 64 DISTINCT PREDICTION SETS, not 63.**
#:
#: **THE 0.0001 WAS REAL, AND I READ IT AS ROUNDING.** The previous
#: record said the agnet gap (0.0613 vs 0.0612) was *"consistent with two
#: records rounding one underlying value"*. It was a genuine difference
#: between two genuinely different prediction sets. **The benign reading
#: was available and I took it, and the one pair that broke the pattern
#: was the one I explained away.**
DUPLICATE_COUNT_CORRECTED = {
    "corrected": "2026-09-01, by hash comparison at seed 1337",
    "method": "fingerprint.predictions_sha256, per pair",
    "result": {
        "vit_b16_imagenet": "IDENTICAL",
        "swin_b_masked": "IDENTICAL",
        "swin_b_imagenet": "IDENTICAL",
        "srgnn_imagenet": "IDENTICAL",
        "agnet_imagenet": (
            "DIFFERENT -- fc5443061acdb181... vs 987aece1ab0303f3..."
        ),
    },
    "the_count": (
        "**FOUR proven byte-identical pairs, not five. The lock's 68 "
        "covers 64 DISTINCT PREDICTION SETS**"
    ),
    "the_0_0001_was_real": (
        "**the previous record read the agnet gap (0.0613 vs 0.0612) as "
        "'consistent with two records rounding one underlying value'. It "
        "was a genuine difference between two genuinely different "
        "prediction sets.** The benign reading was available and I took "
        "it, and the one pair that broke the pattern was the one I "
        "explained away"
    ),
    "the_t_5_count_is_untouched": (
        "**t = 5 pairs tied in accuracy and QWK STANDS AS MEASURED.** It "
        "was never a claim about duplication -- it is a claim about "
        "ties in two coarse columns, and the agnet pair ties there "
        "whether or not its predictions match. See "
        "why_the_agnet_pair_ties_on_coarse_metrics"
    ),
    "why_the_agnet_pair_ties_on_coarse_metrics": (
        "**measured**: perturbing predictions at agnet's declared "
        "~2.3e-05 reproducibility bound changed the 3-class assignment "
        "of 0.000 of 237 patients (max 0 over 400 draws). CLASS-BASED "
        "metrics -- accuracy, QWK, macro F1 -- tie through a "
        "perturbation that size; CONTINUOUS metrics -- PCC, IEM -- do "
        "not. That is exactly the observed pattern: the pair ties on "
        "accuracy and QWK and differs on PCC at the fourth decimal"
    ),
}


#: **[INVESTIGATED 2026-09-01] THE AG-NET DIVERGENCE: a legitimate
#: configuration difference, predicted by a record from 2026-07-31.**
#:
#: **What actually differs, measured by diffing the configs.** For ALL
#: FOUR compared pairs the task blocks are **identical -- zero keys
#: differing** -- and the top-level seed is 1337 in both. The only
#: input that differs is ``embeddings``, and **it differs in all four
#: pairs**, including the three that reproduce byte-identically. So a
#: different embeddings rollup does NOT explain the divergence; the
#: artifacts are different objects whose vectors are evidently equal for
#: vit/swin/srgnn.
#:
#: **What does explain it, and it is one field**::
#:
#:     p7_d1_agnet_imagenet_g1_native   deterministic: false
#:     roadb_p7_arm_agnet_imagenet_224  deterministic: false
#:     p7_d1_srgnn_imagenet_g1_native   (absent -- defaults deterministic)
#:     roadb_p7_arm_srgnn_imagenet_224  (absent -- defaults deterministic)
#:
#: **AG-Net IS DECLARED NON-DETERMINISTIC AND HAS BEEN SINCE
#: 2026-07-31.** ``models.agnet.TRAINING_DETERMINISM``:
#: ``bitwise_flag_off: False``; mechanism *"torchvision::roi_align CUDA
#: backward uses atomicAdd"*; consequence *"same-seed reproducibility and
#: resume identity bounded at ~2.3e-05"*; decision *"the twelve
#: pretraining configs run deterministic: false"*.
#:
#: **So the two AG-Net runs are two independent draws from a declared
#: non-deterministic path, not a failed reproduction.** They are
#: legitimately two arms. The lock is fine, and this is **not a new
#: determinism finding** -- it is the existing record predicting exactly
#: what was observed.
#:
#: **The magnitude is CONSISTENT BUT NOT RECONCILED.** A single
#: perturbation at the declared 2.3e-05 bound shifts PCC by ~1.2e-06
#: (measured, 400 draws); the observed gap is 1.0e-04, about eighty times
#: larger. That is unsurprising across 40 epochs x 5 folds x 10 seeds of
#: accumulation, but **it is not a computed reconciliation and is not
#: claimed as one.**
AGNET_DIVERGENCE_INVESTIGATED = {
    "investigated": "2026-09-01",
    "verdict": (
        "**A LEGITIMATE CONFIGURATION DIFFERENCE.** The two AG-Net runs "
        "are two independent draws from a DECLARED non-deterministic "
        "path. They are legitimately two arms; the lock is fine; and "
        "this is NOT a new determinism finding"
    ),
    "what_the_config_diff_shows": (
        "for ALL FOUR compared pairs the task blocks are IDENTICAL -- "
        "zero keys differing -- and the top-level seed is 1337 in both. "
        "The only differing input is `embeddings`, and it differs in ALL "
        "FOUR pairs INCLUDING the three that reproduce byte-identically. "
        "**A different embeddings rollup does not explain the "
        "divergence**"
    ),
    "the_one_field_that_does": (
        "agnet's configs carry `deterministic: false` on BOTH roads; "
        "srgnn's carry no such field and default to deterministic. That "
        "is the whole difference between the pair that diverged and the "
        "pair at the same a54cdfae sha that did not"
    ),
    "the_record_that_predicted_it": (
        "models.agnet.TRAINING_DETERMINISM, measured 2026-07-31: "
        "bitwise_flag_off False; mechanism 'torchvision::roi_align CUDA "
        "backward uses atomicAdd'; consequence 'same-seed "
        "reproducibility and resume identity bounded at ~2.3e-05'; "
        "decision 'the twelve pretraining configs run deterministic: "
        "false'"
    ),
    "magnitude_consistent_but_not_reconciled": (
        "a single perturbation at the declared 2.3e-05 bound shifts PCC "
        "by ~1.2e-06 (measured, 400 draws); the observed gap is 1.0e-04, "
        "about eighty times larger. Unsurprising across 40 epochs x 5 "
        "folds x 10 seeds of accumulation, but **NOT a computed "
        "reconciliation and not claimed as one**"
    ),
    "the_asymmetry_it_sits_beside": (
        "[STRENGTHENED 2026-09-01: all ten seeds differ, not just 1337, "
        "and SR-GNN reproduces on all ten -- AGNET_DIFFERS_ON_ALL_TEN] "
        "SR-GNN and AG-Net are both ten-seed graph arms and both ran at "
        "a54cdfae on Road B; only AG-Net carries the non-determinism. "
        "The existing provenance asymmetry between the two now has a "
        "second observable consequence: **AG-Net cannot be checked for "
        "duplication by hash at all**, because its own path does not "
        "guarantee byte-identity even against itself"
    ),
}


#: **[MEASURED 2026-09-01] D4 RE-MEASURED AT FOUR DUPLICATES -- and the
#: answer depends on a column that was not counted.**
#:
#: **The macro-F1 tie question is OPEN.** The t = 5 count came from
#: **accuracy and QWK**. D4 is a **macro-F1** statistic, and macro F1 is
#: a separate column that was not part of that count. Whether the agnet
#: pair ties there is **not measured** -- it needs the run table's
#: ``pooled['macro_f1']['mean']`` for the two arms, which is CLUSTER-ONLY
#: and was not read.
#:
#: **Both answers, so neither has to be guessed later**:
#:
#:     if agnet DOES tie on macro F1 (5 zero gaps)  -> 61 of 62 distinct
#:     if agnet does NOT tie      (4 zero gaps)     -> 62 of 63 distinct
#:
#: **The prediction, tagged as one**: agnet's two runs differ by a
#: perturbation that changed **0.000 of 237** 3-class assignments in 400
#: measured draws, and macro F1 is a 3-class statistic -- so it is
#: **likely to tie**, making 61 of 62 the expected answer. **Likely is
#: not measured**, and both are recorded.
#:
#: **D4's banked figure is unchanged either way**: 66 of 67 is correct as
#: computed over the locked 68.
D4_REMEASURED_AT_FOUR = {
    "measured": "2026-09-01",
    "the_open_column": (
        "**the t = 5 count came from ACCURACY and QWK. D4 is a MACRO-F1 "
        "statistic**, and macro F1 is a separate column that was not "
        "part of that count. Whether the agnet pair ties there is NOT "
        "MEASURED -- it needs the run table's pooled['macro_f1']['mean'] "
        "for the two arms, which is CLUSTER-ONLY and was not read"
    ),
    # -----------------------------------------------------------------
    # [RESOLVED 2026-09-01 BY MEASUREMENT -- the two-branch form below is
    # preserved, and the branch that fired is the predicted one.]
    #
    # The agnet pair ties EXACTLY on macro F1 (0.2382392139111266 both),
    # and on accuracy and QWK. It differs only on PCC
    # (0.061312890026951984 vs 0.06121527011532292). So FIVE adjacent
    # pairs carry a zero gap and the corrected figure is 61 OF 62.
    # -----------------------------------------------------------------
    "resolved_by_measurement": (
        "**the agnet pair ties EXACTLY on macro F1 -- "
        "0.2382392139111266 both -- and on accuracy and QWK, differing "
        "only on PCC (0.061312890026951984 vs 0.06121527011532292, a gap "
        "of 9.76e-05). FIVE adjacent pairs carry a zero gap, and the "
        "corrected figure is 61 OF 62 below the floor among distinct "
        "arms**"
    ),
    "the_mechanism_is_confirmed_not_merely_consistent": (
        "**the prediction was made before the measurement and the "
        "measurement matched it.** models.agnet.TRAINING_DETERMINISM "
        "bounds AG-Net's same-seed reproducibility at ~2.3e-05; "
        "perturbing predictions at that bound flipped 0 of 237 3-class "
        "assignments across 400 draws, so class-based metrics were "
        "predicted to tie and continuous ones to differ. **Macro F1, "
        "accuracy and QWK tie exactly; PCC differs at 9.76e-05.** That "
        "is the predicted pattern, observed -- the mechanism is "
        "CONFIRMED, not merely consistent"
    ),
    "both_answers": {
        "if_agnet_ties_on_macro_f1": (
            "[THIS ONE FIRED] 61 of 62 among distinct arms (5 zero gaps)"
        ),
        "if_agnet_does_not_tie": "62 of 63 among distinct arms (4 zero gaps)",
    },
    "the_prediction_tagged_as_one": (
        "[REASONED 2026-09-01, CONFIRMED the same day -- see "
        "resolved_by_measurement] agnet's two runs differ by a "
        "perturbation that changed 0.000 of 237 3-class assignments in "
        "400 measured draws, and macro F1 is a 3-CLASS statistic -- so "
        "it is LIKELY to tie, making 61 of 62 the expected answer. "
        "**Likely is not measured**"
    ),
    "d4s_banked_figure_is_unchanged": (
        "66 of 67 is correct as computed over the locked 68, either way"
    ),
    "what_would_settle_it": (
        "one comparison of pooled['macro_f1']['mean'] between "
        "p7_d1_agnet_imagenet_g1_native and roadb_agnet_imagenet_224 in "
        "run 5's table"
    ),
}


#: **[2026-09-01] WHAT DEPENDS ON THE AGNET PAIR BEING THE SAME:
#: NOTHING.**
#:
#: **No banked figure treats them as one arm.** The lock enumerates two
#: runs and every Phase 18 figure was computed over 68 rows; the
#: duplication was a property discovered afterwards, not an assumption
#: any figure rests on. So the count moving from five to four **changes
#: no banked value** -- it changes only ``DUPLICATE_PREDICTION_SETS``'s
#: own reading, corrected above.
#:
#: **THE FOUR "IDENTICAL" PAIRS ARE PROVEN AT ONE SEED, NOT FIVE.** Seed
#: 1337 only. **One seed proves DIFFERENCE conclusively -- agnet is
#: settled -- but sameness at one seed is not sameness.** The other four
#: seeds (2024, 7, 99, 12345, plus the graph regime's further five for
#: srgnn) were not compared; that comparison is CLUSTER-ONLY and was not
#: run. Until it is, "four byte-identical pairs" is **proven at seed 1337
#: and asserted nowhere else**.
DUPLICATE_DEPENDENCIES = {
    "recorded": "2026-09-01",
    "nothing_depends_on_it": (
        "**no banked figure treats the agnet arms as one.** The lock "
        "enumerates two runs and every Phase 18 figure was computed over "
        "68 rows; the duplication was discovered afterwards, not assumed "
        "by any figure. The count moving from five to four changes NO "
        "BANKED VALUE -- only DUPLICATE_PREDICTION_SETS' own reading"
    ),
    "proven_at_one_seed_only": (
        "[LIFTED 2026-09-01 -- all seeds compared, see "
        "DUPLICATION_PROVEN_AT_DEPTH] **seed 1337 only.** One seed proves DIFFERENCE conclusively, so "
        "agnet is settled -- but **sameness at one seed is not "
        "sameness**. Seeds 2024, 7, 99, 12345 (and the graph regime's "
        "further five for srgnn) were NOT compared; that comparison is "
        "CLUSTER-ONLY and was not run"
    ),
    "what_the_claim_is_therefore_limited_to": (
        "[SUPERSEDED 2026-09-01 -- proven at every seed present] "
        "'four byte-identical pairs' is PROVEN AT SEED 1337 and asserted "
        "nowhere else. A pair identical at 1337 and divergent at 2024 "
        "would be a second AG-Net-shaped finding, and nothing here rules "
        "it out"
    ),
    "what_would_settle_it": (
        "predictions_sha256 for the remaining four seeds on each of the "
        "four pairs -- twenty comparisons, one command"
    ),
}


#: **[MEASURED 2026-09-01, AT FULL DEPTH] DUPLICATION PROVEN ON EVERY
#: SEED PRESENT -- sameness now rests on measurement, not on one seed.**
#:
#: ``DUPLICATE_DEPENDENCIES`` limited the claim to seed 1337 and said
#: *"sameness at one seed is not sameness"*. **The remaining seeds have
#: now been compared and the limit is lifted.**
#:
#: **Four pairs, byte-identical on every seed each carries:**
#:
#:     vit_b16 imagenet     5 seeds   1337 2024 7 99 12345
#:     swin_b  masked       5 seeds   1337 2024 7 99 12345
#:     swin_b  imagenet     5 seeds   1337 2024 7 99 12345
#:     srgnn   imagenet    10 seeds   the five, plus 42 271828 314159
#:                                    161803 777
#:
#: **The comparison count is 25, not 30.** Derived from
#: ``ARM_LIST_LOCKED``: 5 + 5 + 5 + 10. The closing message said 30;
#: recorded as 25 because that is what the locked seed lists give, and
#: the difference is stated rather than absorbed. Adding the AG-Net
#: pair's ten differing comparisons gives 35 for all five pairs.
#:
#: **What this upgrades**: sameness was previously PROVEN AT ONE SEED and
#: asserted nowhere else. It is now proven **at every seed the pairs
#: carry**, including the graph regime's further five -- so a
#: "identical at 1337, divergent at 2024" surprise is ruled out by
#: measurement rather than by hope.
DUPLICATION_PROVEN_AT_DEPTH = {
    "measured": "2026-09-01, at full depth",
    "what_it_lifts": (
        "DUPLICATE_DEPENDENCIES limited the claim to seed 1337 and said "
        "'sameness at one seed is not sameness'. **The remaining seeds "
        "have now been compared and the limit is lifted**"
    ),
    "four_pairs_identical_on_every_seed": {
        "vit_b16_imagenet": 5,
        "swin_b_masked": 5,
        "swin_b_imagenet": 5,
        "srgnn_imagenet": 10,
    },
    "the_graph_regimes_further_five": (42, 271828, 314159, 161803, 777),
    "the_comparison_count": (
        "**25, not 30.** Derived from ARM_LIST_LOCKED: 5 + 5 + 5 + 10. "
        "The closing message said 30; recorded as 25 because that is "
        "what the locked seed lists give, and the difference is STATED "
        "rather than absorbed. All five pairs together are 35 "
        "comparisons, the AG-Net pair contributing ten differing ones"
    ),
    "what_it_upgrades": (
        "sameness was PROVEN AT ONE SEED and asserted nowhere else. It "
        "is now proven AT EVERY SEED THE PAIRS CARRY, including the "
        "graph regime's further five -- so an 'identical at 1337, "
        "divergent at 2024' surprise is ruled out BY MEASUREMENT rather "
        "than by hope"
    ),
}


#: **[MEASURED 2026-09-01] AG-NET DIFFERS ON ALL TEN SEEDS -- the
#: stronger form of an existing finding.**
#:
#: Seed 1337 showed one divergence. **All ten show it.** So the
#: non-determinism in ``models.agnet.TRAINING_DETERMINISM`` is a
#: **property of the path, not a one-seed accident**, and **AG-Net cannot
#: be checked for duplication by hash at any depth** -- not because the
#: check is hard, but because its own path does not guarantee
#: byte-identity even against itself.
#:
#: **THE CONTROL IS SR-GNN, AND IT IS EXACT.** Same graph regime, same
#: ten seeds, same ``a54cdfae`` commit on Road B, same
#: ``region_scheme: native``, same ``trainable: graph_layers`` -- and it
#: reproduces byte-identically on all ten. **So the difference is
#: AG-Net's declared ``deterministic: false``, not the graph regime**,
#: and not the commit gap. One field, isolated by a control that differs
#: in nothing else.
AGNET_DIFFERS_ON_ALL_TEN = {
    "measured": "2026-09-01",
    "the_finding": (
        "**AG-Net's two runs differ on ALL TEN SEEDS, not just 1337.** "
        "The non-determinism is a PROPERTY OF THE PATH, not a one-seed "
        "accident"
    ),
    "the_consequence": (
        "**AG-Net cannot be checked for duplication by hash AT ANY "
        "DEPTH** -- not because the check is hard, but because its own "
        "path does not guarantee byte-identity even against itself"
    ),
    "the_control_is_srgnn_and_it_is_exact": (
        "same graph regime, same ten seeds, same a54cdfae commit on Road "
        "B, same region_scheme native, same trainable graph_layers -- "
        "and SR-GNN reproduces byte-identically on ALL TEN. **So the "
        "difference is AG-Net's declared deterministic: false, NOT the "
        "graph regime and NOT the commit gap.** One field, isolated by a "
        "control that differs in nothing else"
    ),
    "filed_beside": (
        "the SR-GNN / AG-Net provenance asymmetry, and "
        "models.agnet.TRAINING_DETERMINISM (measured 2026-07-31), which "
        "predicted exactly this: bitwise_flag_off False, mechanism "
        "'torchvision::roi_align CUDA backward uses atomicAdd'"
    ),
    "what_it_does_not_make_agnet": (
        "**not a defective arm.** Its non-determinism is declared, "
        "bounded at ~2.3e-05, and was measured and decided in 2026-07-31 "
        "before any of these runs existed. Two AG-Net runs of one config "
        "are two legitimate draws, not a failed reproduction"
    ),
}


#: **[2026-09-01] THE LIMITATION, STATED FOR THE WRITE-UP.**
#:
#: **The locked 68 comprises 64 DISTINCT PREDICTION SETS: four exact
#: duplicates, plus one pair that differs only through declared
#: non-determinism.**
#:
#: **Every Phase 18 figure was computed over 68 rows and remains correct
#: as computed.** Nothing is withdrawn, no value moves, and the lock is
#: not reopened.
#:
#: **What a reader must not do**: treat 68 as 68 independent
#: measurements. **That reads more independence than the set contains.**
#: Four pairs are one measurement counted twice; the fifth is two draws
#: from one non-deterministic configuration, which is closer to two
#: measurements than the others but is not two independent arms either.
#:
#: **Where it bites, measured**: D4's *"66 of 67 adjacent macro-F1 pairs
#: below the floor"* is **61 of 62 among distinct arms** -- five of the
#: sixty-six are zero-gap pairs, uninterpretable TRIVIALLY rather than
#: INFORMATIVELY. The tau values and top-5 overlaps carry a smaller,
#: unquantified upward bias (``DUPLICATE_PREDICTION_SETS``).
#:
#: **A LIMITATION OF THE LOCK, RECORDED AS SUCH -- not a reopening.**
DUPLICATION_LIMITATION_FOR_THE_WRITE_UP = {
    "recorded": "2026-09-01",
    "the_statement": (
        "**the locked 68 comprises 64 DISTINCT PREDICTION SETS: four "
        "exact duplicates, plus one pair that differs only through "
        "declared non-determinism**"
    ),
    "every_figure_remains_correct_as_computed": (
        "every Phase 18 figure was computed over 68 rows and remains "
        "correct as computed. Nothing is withdrawn, no value moves, and "
        "the lock is NOT reopened"
    ),
    "what_a_reader_must_not_do": (
        "**treat 68 as 68 independent measurements -- that reads more "
        "independence than the set contains.** Four pairs are ONE "
        "measurement counted twice; the fifth is two draws from one "
        "non-deterministic configuration, which is closer to two "
        "measurements than the others but is not two independent arms "
        "either"
    ),
    "where_it_bites_measured": (
        "D4's '66 of 67 adjacent macro-F1 pairs below the floor' is **61 "
        "OF 62 among distinct arms** -- five of the sixty-six are "
        "zero-gap pairs, uninterpretable TRIVIALLY rather than "
        "INFORMATIVELY. The tau values and top-5 overlaps carry a "
        "smaller, unquantified upward bias "
        "(DUPLICATE_PREDICTION_SETS)"
    ),
    "status": (
        "A LIMITATION OF THE LOCK, RECORDED AS SUCH -- not a reopening, "
        "on the ARM_LIST_ADDENDUM pattern: visibility, not coverage"
    ),
}
