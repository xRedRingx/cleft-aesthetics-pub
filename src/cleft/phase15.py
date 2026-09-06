"""Phase 15: the second beauty dataset -- Road B Branch 2, unparked.

The scheduling registration's one mandatory term
(``phase12.PHASE_SEQUENCE_RENUMBERED_2``): the exit criteria MUST include
a comparative verdict against SCUT, because Phase 16 consumes "the best
beauty dataset" as a MEASURED answer. The record's own constraint on
"best" (``ladder``, the scheme-axis block): SCUT test PCC turned out not
to predict cleft transfer at all -- so best is judged by CLEFT-SIDE
transfer, never source-side fit.
"""

from __future__ import annotations

#: **[REGISTERED 2026-08-24, FROM VERIFIED SOURCES] THE CANDIDATE SURVEY
#: VERDICT** -- the phase's first deliverable; acquisition is the maintainer's
#: act and follows separately.
CANDIDATE_SURVEY_REGISTERED = {
    "registered": (
        "2026-08-24, from verified sources; the maintainer's acquisition gate "
        "follows"
    ),
    "lead_candidate": {
        "name": "MEBeauty",
        "source": (
            "Lebedeva, Guo & Ying 2021 (Neural Computing and "
            "Applications); github.com/fbplab/MEBeauty-database"
        ),
        "size": "2,550 in-the-wild images",
        "demographics": (
            "SIX ethnicities, varied ages (31% of males under 20) -- "
            "against SCUT's young East-Asian faces, the maximal "
            "demographic difference among the acquirable candidates"
        ),
        "labels": "1-10 attractiveness scores averaged over ~300 raters",
        "landmarks": (
            "landmarks.csv SHIPPED -- no detector needed, so the "
            "avoid-detectors verdict (MEDIAPIPE_VERDICT_ENCODED) stays "
            "dormant"
        ),
        "license": (
            "non-commercial research license -- compatible; CITATION "
            "REQUIRED, which the write-up must carry"
        ),
    },
    "mechanisms": {
        "i_demographics": (
            "ALIVE AND STRONG: MEBeauty is the maximal demographic "
            "difference from SCUT among acquirable candidates -- the "
            "un-measured axis the SCUT negative cannot speak to, "
            "checkable by construction"
        ),
        "ii_scale": (
            "DEAD AND INVERTED: MEBeauty (2,550) is roughly HALF "
            "SCUT's size (5,500), so SIZE CONFOUNDS ANY TRANSFER "
            "DIFFERENCE IN SCUT'S FAVOUR. The asymmetry is binding on "
            "every reading: a MEBeauty WIN is STRONG evidence (it won "
            "despite the size handicap); a MEBeauty LOSS is AMBIGUOUS "
            "(demographics or size -- not attributable). No reading "
            "may be registered that ignores this"
        ),
        "iii_ranking_deliverable": (
            "ALIVE REGARDLESS: Phase 16 consumes the measured ranking "
            "even if both sources transfer poorly"
        ),
    },
    "alternates_named_not_pursued": (
        "SCUT-FBP500: DOMINATED (same lab, same demographic, smaller "
        "-- no mechanism survives). CelebA-attractiveness: WRONG LABEL "
        "TYPE (binary attribute, not a rating). Chicago Face Database "
        "and 10k-US-Adult-Faces: UNVERIFIED ALTERNATES, held only in "
        "case acquisition fails. HotOrNot-era sets: availability "
        "doubtful. None is registered; naming them here is what stops "
        "a later turn from presenting one as fresh"
    ),
    "acquisition": (
        "THE ACQUISITION ACT: clone github.com/fbplab/MEBeauty-database to "
        "the cluster, declare via the standing two-pass. Nothing "
        "downloads by code in this repo"
    ),
}


#: **[REGISTERED 2026-08-24, BEFORE ANYTHING ELSE RUNS] STOP 1'S FIRST
#: MEASUREMENT: THE FRONTAL FRACTION.**
FRONTAL_FRACTION_FIRST = {
    "registered": "2026-08-24, before any staging or pretraining exists",
    "why_first": (
        "MEBeauty is IN-THE-WILD and includes non-frontal poses; the "
        "cleft pipeline stages frontal faces. The usable subset, not "
        "the 2,550, is the real size -- and mechanism ii's inversion "
        "is already binding at 2,550, so a small frontal subset "
        "WORSENS the scale asymmetry and the registration must say BY "
        "HOW MUCH before any transfer number exists"
    ),
    "the_screen": (
        "pose screened by the SHIPPED landmarks (landmarks.csv), no "
        "detector; the threshold is DECLARED IN THE CONFIG before the "
        "screen runs -- the P1 discipline, a threshold chosen after "
        "seeing the distribution is not a threshold"
    ),
    "first_banked_number": (
        "the usable-subset count is the phase's FIRST banked number, "
        "and every later registration states the scale asymmetry with "
        "the screened count, not the headline 2,550"
    ),
}


#: **[ENCODED 2026-08-24, one line as instructed] THE MEDIAPIPE VERDICT**
#: -- recorded in the repo at last, sourced to where it actually lived.
MEDIAPIPE_VERDICT_ENCODED = {
    "encoded": "2026-08-24, sourced to the July session handoff (not the repo)",
    "verdict": (
        "mediapipe on this cluster failed on GL libraries, no-root "
        "installation and wrong-version conflicts (July session "
        "handoff); verdict AVOID DETECTORS -- masking rides shipped "
        "landmarks or does not happen. Dormant for MEBeauty, which "
        "ships landmarks.csv"
    ),
}


#: **[CORRECTED 2026-08-24, at acquisition] THE LANDING.**
LANDING_CORRECTED = {
    "corrected": "2026-08-24, the maintainer, at acquisition",
    "the_correction": (
        "MEBeauty is a SOURCE dataset and lives OUTSIDE the repo at "
        "/home/user/codex/mebeauty/MEBeauty-database -- the SCUT/bch "
        "sibling pattern. The proposed data/mebeauty/raw_v1 landing is "
        "REJECTED: data/ is for DERIVED artifacts this pipeline "
        "produces, not for sources it consumes"
    ),
    "why": (
        "the source-vs-derived boundary: the audit reads a tree where "
        "everything under data/ was made by a declared run, and a "
        "source dataset inside it would break that reading; and a "
        "651M clone inside the repo is a gitignore hazard waiting"
    ),
    "portability": (
        "configs declare ${CLEFT_MEBEAUTY_ROOT} on the "
        "${CLEFT_SCUT_ROOT} precedent -- pod export plus the job-form "
        "env var whenever a config declares it; the declared "
        "rollup_sha256 verifies whatever the env resolves"
    ),
}


#: **[BANKED 2026-08-24, FROM THE CLONE] WHAT ACTUALLY ARRIVED** -- 651M,
#: 9,808 files -- and the two survey items its contents add.
CLONE_CONTENTS = {
    "banked": "2026-08-24, from the run outputs's clone listing",
    "contents": (
        "original_images/, cropped_images/, landmarks.csv, scores/ (a "
        "DIRECTORY -- its files are inventoried, never assumed to be "
        "one CSV), geometric_features.csv, FaceNet_512_features/, "
        "MEBeauty_creation_cleaning/, notebooks and pytorch scripts, "
        "README.md, face_crop_align.py -- 651M, 9,808 files"
    ),
    "survey_item_scores": (
        "the task inventories scores/ and STATES which file(s) carry "
        "the 1-10 means and whether per-rater or distributional data "
        "exists -- bearing on nothing yet, but BANKED, so a later "
        "phase reaching for a distribution knows where it stands "
        "without a second dig"
    ),
    "survey_item_cropped_images": (
        "cropped_images/ is a SHIPPED ALIGNMENT the phase may COMPARE "
        "against but does NOT consume by default: staging decisions "
        "come from OUR OWN pipeline (exit criterion 3), and any use "
        "of their crops is a DATED DECISION, never a drift"
    ),
    # [RECONCILED 2026-08-24, at the declare pass] The clone listing
    # above and the declared artifact carry DIFFERENT numbers, and both
    # are right about what they measured.
    "declare_reconciliation": (
        "2026-08-24: the ls-level clone report said 9,808 files / 651M; "
        "the declare's walk sees 9,806 files / 242,062,060 bytes "
        "(rollup 6719959b). The differences are the EXCLUDED .git "
        "internals (204 MiB of pack history) and the standing basename "
        "exclusions -- the rollup hashes the DATASET, not the clone's "
        "version-control plumbing, so the artifact's numbers explain "
        "themselves and a future re-declare matching 9,806/242M is "
        "AGREEMENT, not drift"
    ),
}


#: **[AGREED 2026-08-24, all seven as drafted] THE EXIT
#: CRITERIA.**
PHASE_15_EXIT_CRITERIA = {
    "agreed": "2026-08-24 -- all seven as drafted",
    "criteria": (
        "1. BEST is defined as CLEFT-SIDE TRANSFER, measured in both "
        "registered senses -- masked-pretraining transfer and frozen "
        "linear-probe transfer, on the 237/cleft_v1 folds, A's recipe "
        "byte-identical -- never source-side fit (the ladder's 0.7893 "
        "lesson)",
        "2. the FRONTAL SCREEN banks first, threshold pre-declared; "
        "every later registration uses the SCREENED count, not the "
        "headline 2,550",
        "3. staging/masking INHERITS levelling-OFF and the CLEFT_AR "
        "parity sampling, or departs DATED, per Phase 5's decisions; "
        "parity checks re-run on the new artifact",
        "4. checkpoints are GEOMETRY-BOUND per VARIANT_FOR_INIT; the "
        "phantom-findings trap honoured -- no comparison moves "
        "geometry across a new-dataset checkpoint undeclared",
        "5. SCUT'S SIDE IS THE BANKED Phase 6 figures (0.2520 / "
        "0.1952 / 0.0830); re-run only if the recipe must change for "
        "MEBeauty, and then both sides re-run together",
        "6. the comparative verdict's statistic and threshold are "
        "DECLARED BEFORE ANY NUMBER -- deltas against the banked "
        "cells thresholded from both arms' own seed spreads (PLAN "
        "4.12.1), DESCRIPTIVE ranking, PLAN 4.3 machinery only if a "
        "gap clears the paired bar -- and EVERY reading carries the "
        "mechanism-ii asymmetry sentence (a MEBeauty win is strong, "
        "a loss is ambiguous)",
        "7. readings pre-committed and applied through "
        "READING_COUNT_GUARD; no ledger row without the paired bar; "
        "the license citation carried; suite green",
    ),
    "expected_count": 7,
}


#: **[BUILT 2026-08-24, STOP 1 -- NOT LAUNCHED] THE SURVEY TASK.**
STOP_1_BUILT = {
    "built": "2026-08-24, NOT launched",
    "task": (
        "survey_mebeauty, against ${CLEFT_MEBEAUTY_ROOT}: inventory "
        "against the published counts and the banked clone listing; "
        "scores/ inventoried file by file with the means-carrier "
        "stated (CLONE_CONTENTS' first survey item); the frontal "
        "screen by the SHIPPED landmarks with the threshold declared "
        "in the YAML; the usable count banked with the "
        "scale-asymmetry restatement; SHAREABLE accept/reject sheets "
        "for the visual check. NO GATE -- public data, no patient "
        "pixel anywhere near it"
    ),
    "the_pose_statistic": (
        "the NORMALISED CENTROID OFFSET: |mean(x) - midpoint(min_x, "
        "max_x)| / width over each image's landmark x-coordinates. "
        "Chosen because it is LANDMARK-SEMANTICS-FREE: the file's "
        "point indexing is unverified, and a statistic that needs no "
        "semantic pairs cannot be wrong about them. Frontal faces "
        "centre their landmark mass; profiles skew it. The threshold "
        "(frontal_max_offset) is DECLARED IN THE CONFIG before the "
        "screen runs (FRONTAL_FRACTION_FIRST)"
    ),
    "format_discipline": (
        "landmarks.csv's layout is UNVERIFIED offline, so the task "
        "asserts its shape rather than assuming it: the first column "
        "is the image key and the remaining numeric columns pair as "
        "(x, y) -- an ODD numeric count REFUSES with the measured "
        "header in the message, so a format surprise arrives as "
        "evidence, not as silently wrong geometry"
    ),
    "not_consumed": (
        "cropped_images/, geometric_features.csv and "
        "FaceNet_512_features/ are inventoried and NOT consumed -- "
        "the second survey item's boundary, in code"
    ),
}


#: **[BANKED 2026-08-24] STOP 1'S FIRST RUN: HALF BANKED, HALF REFUSED
#: WITH EVIDENCE -- as built.**
STOP_1_FIRST_RUN = {
    "banked": "2026-08-24, p15-survey first run",
    "inventory_banked": (
        "2,539 images under original_images/ against the published "
        "2,550 -- ELEVEN SHORT, and the measured count is the one that "
        "counts. scores/ classified in full: MEAN CARRIERS are "
        "train_universal_scores.csv + test_universal_scores.csv and "
        "train_crop.csv + test_crop.csv (TWO FAMILIES); *_all.xlsx "
        "binaries are per-rater candidates, UNREAD; the train/val/test "
        ".txt splits carry no score column"
    ),
    "the_refusal": (
        "the landmark parser refused with the measured header "
        "['', 'image', 'score', 'landmarks'] -- coordinates PACKED in "
        "one cell, not spread columns. The refusal-with-evidence "
        "discipline did its job: the format surprise arrived as a "
        "header in an error message, not as silently wrong geometry"
    ),
    "the_measured_format": (
        "measured from the source before extending the parser: 136 "
        "comma-separated integers per cell = 68 (x, y) pairs (the "
        "dlib-68 convention), a trailing comma inside the quoted cell, "
        "the image column carrying the AUTHORS' absolute paths (joined "
        "by basename), and a per-row score -- a THIRD SCORE SURFACE. "
        "The parser now refuses any cell that does not parse to "
        "exactly 136 numerics, with the cell's first 60 characters as "
        "evidence"
    ),
    "families_unresolved_readme_silent": (
        "the README does NOT distinguish universal from crop score "
        "files (verified against the repository's README, which "
        "documents a different 2022-era file set). The distinction is "
        "therefore MEASURED, not guessed: the rerun reports which "
        "image directory each family's keys resolve against "
        "(original_images vs cropped_images, by basename) -- the "
        "label-source decision waits for that measurement"
    ),
    "third_score_surface": (
        "landmarks.csv's own score column is a third surface beside "
        "the two carrier families; the rerun reports its per-family "
        "join count and mean/max absolute difference -- REPORT, NEVER "
        "GATE, a consistency check and not a label source"
    ),
    "boundary_refined": (
        "the family measurement LISTS cropped_images/ basenames -- "
        "metadata, for the key-resolution report -- and still opens NO "
        "crop pixel: render.load_image touches original_images only, "
        "test-pinned. The not-consumed-by-default boundary "
        "(CLONE_CONTENTS) is about pixels and staging, and it holds"
    ),
    "rerun": "p15-survey-2, config unchanged -- the fix is code-side only",
}


#: **[BANKED 2026-08-24] STOP 1 COMPLETE: p15-survey-2, single clean
#: attempt -- the funnel, the label source, the eye review.**
STOP_1_BANKED = {
    "banked": "2026-08-24, p15-survey-2, single clean attempt",
    "the_funnel": (
        "recorded AS A FUNNEL, each step's loss named: 2,550 published "
        "-> 2,539 on disk (eleven never shipped) -> 2,445 "
        "landmark-joined (~94 images have no landmarks.csv row) -> "
        "1,519 USABLE (62.1% pass at the declared 0.08 centroid-offset "
        "threshold). The phase's first banked number is 1,519"
    ),
    "scale_asymmetry_restated": (
        "at the SCREENED count: 0.28x SCUT (1,519 vs 5,500), down from "
        "the 0.46x headline. Mechanism ii's binding sentence now "
        "carries the real figure: a MEBeauty win is strong evidence AT "
        "0.28x THE DATA; a loss is ambiguous "
        "(CANDIDATE_SURVEY_REGISTERED)"
    ),
    "label_source_decided": (
        "BY THE KEY-RESOLUTION MEASUREMENT, not the silent README: the "
        "UNIVERSAL family (train/test_universal_scores.csv, 2,198 "
        "rows) resolves both image directories near-identically -- the "
        "general score set -- and the crop family is its cropped-keyed "
        "subset. The universal family is the phase's label source"
    ),
    "third_surface_banked": (
        "mean |diff| ~0.07 on the 1-10 scale across all four families "
        "-- the score surfaces agree to under 1% of range. BANKED, "
        "report-never-gate; no surface contradicts another"
    ),
    "eye_review_PASS": (
        "the visual check, a second reading joined (public data): PASS at 0.08, "
        "threshold UNCHANGED. Accepts are uniformly stageable-frontal "
        "with the demographic spread visible -- mechanism i in "
        "pictures. Rejects are dominated by genuine "
        "three-quarter/profile poses"
    ),
    "threshold_trade_recorded": (
        "the 0.08-0.12 band contains recoverable frontal faces "
        "(cap-wearers, near-frontals at 0.10-0.12): the threshold "
        "trades ~5% usable count for pose purity, ACCEPTED BY REVIEW, "
        "no post-hoc adjustment -- a threshold moved after seeing the "
        "distribution is not a threshold"
    ),
    "occlusion_note": (
        "minor occlusions on some accepts (hands, hats, flowers) -- "
        "SCUT-precedented and largely crop-removed; noted, not "
        "screened"
    ),
}


#: **[PROPOSED 2026-08-24 -- NOT BUILT, the ruling first] STOP
#: 2'S STAGING PLAN.** Two sub-stops, because the landmark mapping is a
#: dated decision that must be MEASURED before any batch run.
STOP_2_PLAN_PROPOSED = {
    "proposed": (
        "2026-08-24, for the ruling on the mapping and the "
        "split; NOTHING built"
    ),
    "structure": (
        "stop 2a: the dlib-68 index VERIFICATION (pod, landmarks only, "
        "no pixels) plus mapping SHEETS (a seeded sample staged "
        "through the proposed mapping, SHAREABLE, eye-gate); stop 2b: "
        "the full 1,519 staged at G1 AND G2, parity checks re-run, "
        "staged sheets eye-gated BEFORE any embedding"
    ),
    "mapping_proposed_not_trusted": (
        "the dlib-68 equivalents of placement's 86-point anchors, "
        "PROPOSED from the standard convention and VERIFIED "
        "numerically before use -- the SCUT discipline exactly, whose "
        "own indices were verified over 300-500 faces and whose "
        "record opens with 'guessing landmark indices produces "
        "geometry that looks plausible and is wrong': EYE_LEFT 36-41, "
        "EYE_RIGHT 42-47 (dlib's subject-right eye is image-left); "
        "midline = eye midpoint + 33 (subnasale) + 51 (cupid's-bow "
        "dip); vertical span = brows 17-26 to outer mouth 48-59, "
        "LOWER_MARGIN 0.06 inherited. The verification re-runs SCUT's "
        "own checks on the 1,519: mirror-pair |x| of 33 and 51 near "
        "zero, eye-group centres symmetric, 48/54 the widest "
        "symmetric mouth pair -- MEASURED, then sheets, then batch"
    ),
    "split_proposed": (
        "the SHIPPED universal-family split "
        "(train_universal_scores.csv / test_universal_scores.csv), "
        "restricted to the usable 1,519 -- the SCUT precedent (the "
        "official split as shipped, never our own re-cut), and the "
        "split whose files ARE the decided label source. The "
        "usable-side counts per split half are MEASURED at stop 2a "
        "and declared into stop 2b's config -- the two-pass spirit "
        "applied to counts"
    ),
    "inheritances": (
        "levelling OFF (SCUT_HEAD_TILT_LEVELLING, the matched-property "
        "argument inherited unchanged); CLEFT_AR parity sampling from "
        "the 237 observed ratios (CLEFT_AR, piecewise-linear inverse "
        "CDF); parity checks re-run on the new artifact; G1 AND G2 "
        "per the SCUT convention; VARIANT_FOR_INIT applies to any "
        "later checkpoint"
    ),
    "configs_planned": (
        "p15_verify_mapping.yaml (stop 2a: verification + sheets, "
        "mebeauty_root carried, no new declares) and "
        "p15_stage_mebeauty.yaml (stop 2b: two-pass on the counts "
        "measured at 2a; writes data/mebeauty_staged/... as the "
        "DERIVED artifact, which belongs under data/ exactly where "
        "the source clone does not)"
    ),
    "eye_gates": (
        "two: the mapping sheets (2a) and the staged sheets (2b), "
        "both SHAREABLE, both before any embedding exists"
    ),
}


#: **[RULED 2026-08-24: PROCEED AS PROPOSED] THE MAPPING AND
#: THE SPLIT.**
STOP_2_RULINGS = {
    "ruled": "2026-08-24 -- both as proposed",
    "mapping_is_literature_not_measured": (
        "the dlib-68 indices are the STANDARD CONVENTION and are "
        "hereby tagged [LITERATURE], NOT measured. 2a verifies them "
        "numerically over the 1,519 by SCUT's own three checks: "
        "mirror-pair |x| of 33 and 51 near zero; eye-group centres "
        "symmetric; 48/54 the widest symmetric pair"
    ),
    "the_direction_of_accommodation": (
        "**IF ANY CHECK FAILS THE MAPPING REOPENS -- THE CHECK IS "
        "NEVER ADJUSTED TO FIT THE INDICES. That direction of "
        "accommodation IS the failure mode.** The thresholds are "
        "declared in the YAML before the run so the question cannot "
        "be settled after the fact"
    ),
    "the_image_left_trap": (
        "dlib's 'left eye' is the SUBJECT's left, which appears on the "
        "IMAGE's RIGHT; scut.placement's EYE_LEFT means IMAGE-left. So "
        "dlib 36-41 is the IMAGE-LEFT group. Named in "
        "mebeauty.EYE_IMAGE_LEFT and ASSERTED in code (check 2's "
        "image_left_fraction), because getting it backwards mirrors "
        "every derived quantity while leaving every shape plausible"
    ),
    "order_is_binding": (
        "MEASURED, then SHEETS, then BATCH -- in that order. 2a "
        "verifies and renders; the visual-check gates; only then does "
        "2b stage a pixel"
    ),
    "split_approved_with_its_reason": (
        "the SHIPPED universal train/test files restricted to the "
        "usable 1,519, on the SCUT precedent -- and the maintainer's added "
        "reason for the record: A HOME-MADE PARTITION INSIDE A "
        "CROSS-DATASET COMPARISON IS A CONFOUND THE PHASE CAN AVOID "
        "FOR FREE. Per-half usable counts are measured at 2a and "
        "declared into 2b's config"
    ),
    "inherited_unchanged": (
        "LOWER_MARGIN 0.06; levelling OFF; CLEFT_AR parity sampling; "
        "parity checks re-run on the new artifact; G1 and G2; "
        "VARIANT_FOR_INIT for any later checkpoint; both eye gates "
        "before any embedding; the derived artifact under data/"
    ),
}


#: **[BUILT 2026-08-24, STOP 2a -- NOT LAUNCHED] THE MAPPING
#: VERIFICATION.**
STOP_2A_BUILT = {
    "built": "2026-08-24, NOT launched",
    "task": (
        "verify_mebeauty_mapping: the three checks over every usable "
        "face (landmarks only -- NO PIXEL is read for the checks), the "
        "per-half usable split counts measured for 2b's config, and "
        "mapping SHEETS -- a seeded sample with the trapezium drawn "
        "from the proposed anchors, SHAREABLE, for the visual check"
    ),
    "checks_own_the_mapping": (
        "mebeauty.verify_mapping reports what it MEASURED and whether "
        "each check cleared the threshold it was GIVEN; it tunes "
        "nothing. The task refuses to proceed to the sheets' verdict "
        "if all_pass is false -- it still writes the report, because a "
        "failed verification is a finding and its numbers are how the "
        "mapping gets reopened"
    ),
    "sheets": (
        "accept-side faces with the proposed trapezium and the midline "
        "drawn on, so the eye checks the MAPPING and not merely the "
        "pose screen; SHAREABLE (public data)"
    ),
    "no_pixel_for_the_verdict": (
        "the three checks read landmarks.csv only. Pixels are read "
        "ONLY to render the sheets, and the verdict does not depend on "
        "them -- so a sheet-rendering failure cannot corrupt a "
        "measurement"
    ),
}


#: **[BANKED 2026-08-24] STOP 2a: THE MAPPING PASSES, BY NUMBERS AND BY
#: EYE.**
STOP_2A_BANKED = {
    "banked": "2026-08-24, p15-verify-mapping, one pass",
    "usable_reproduced": "1,519 -- identical to stop 1's banked count",
    "check_1_midline": (
        "landmark 33 median |x| 0.01302 (p95 0.05562); landmark 51 "
        "median 0.01294 (p95 0.06653) -- both against the declared 0.02"
    ),
    "check_2_eye_symmetry": (
        "median imbalance 0.01667 against the declared 0.02; "
        "image-left fraction 1.0000 -- THE SUBJECT-RIGHT TRAP IS CLOSED "
        "IN EVERY FACE"
    ),
    "check_3_corners": (
        "48/54 the widest symmetric pair in 1.0000 of faces"
    ),
    "verdict": (
        "ALL CHECKS PASS. The [LITERATURE] dlib-68 mapping is confirmed "
        "on this dataset, and no threshold was touched to get there"
    ),
    "split_counts_measured": (
        "train 1,538 rows / 932 usable; test 660 rows / 394 usable"
    ),
    "the_count_gap_stated": (
        "**TWO COUNTS, STATED TOGETHER RATHER THAN LEFT TO COEXIST**: "
        "the STAGEABLE set is 1,519; the SPLIT-COVERED set is 932 + "
        "394 = 1,326. The shipped universal split files DO NOT COVER "
        "EVERY USABLE FACE -- 193 usable faces appear in no split half"
    ),
    "proposed_ruling_for_2b": (
        "PROPOSED, for the maintainer's agreement or amendment: STAGE ALL "
        "surviving faces (the extras cost nothing and may serve later) "
        "but PRETRAIN ONLY on the split-covered subset, with the scale "
        "asymmetry restated at that number -- 1,326 vs 5,500 ~ 0.24x, "
        "and it falls again after the quality screen"
    ),
}


#: **[REVIEWED 2026-08-24] THE MAPPING SHEETS: PASS -- and a defect the
#: three checks cannot see.**
MAPPING_SHEETS_REVIEWED = {
    "reviewed": (
        "2026-08-24, the visual check, a second reading joined (public data)"
    ),
    "verdict": (
        "**MAPPING PASSES BY EYE -- DO NOT REOPEN IT.** On well-formed "
        "faces the anatomy is right (brows, eye contours, subnasale, "
        "mouth outline), the red midline tracks nose-philtrum-chin, and "
        "the subject-left/image-right convention is visibly correct -- "
        "confirming check 2's 1.0000 by eye"
    ),
    "the_defect_the_checks_cannot_see": (
        "LANDMARK QUALITY on individual faces. Within the accepted "
        "1,519: girl-3956612 has the whole constellation floating off "
        "the face (brow points in hair, mouth points on cheek); "
        "man-1868320 has mouth points on occluding hands; 119.jpg "
        "renders with NO landmarks and NO midline at all yet was "
        "counted usable; alieza-rizvie, farhan-shaikh and ankush-minda "
        "show sparse or partly-drifted sets"
    ),
    "prevalence_EYE_IMPRESSION": (
        "**EYE-IMPRESSION FROM TWO REVIEWERS, NOT A COUNT**: roughly "
        "3-5 of 32 sampled. The figure may be quoted as an impression "
        "and never as a rate"
    ),
    "withdrawal": (
        "**ONE EARLIER CALL WITHDRAWN, on the review**: "
        "man-2785071's points are CORRECT -- the face is tilted back "
        "and bearded, and this was misread the pose. Recorded because "
        "EYE IMPRESSIONS GET THE SAME RETRACTION DISCIPLINE AS "
        "NUMBERS: a withdrawn observation is withdrawn by name, not "
        "quietly dropped"
    ),
    "cause": (
        "the authors' detector failed or half-failed on some images, "
        "and the centroid-offset pose screen is BLIND to it: a "
        "misplaced-but-SYMMETRIC cloud passes happily. The screen was "
        "never wrong about pose; it was answering a different question"
    ),
}


#: **[BUILT 2026-08-24, STOP 2a-ii -- NOT LAUNCHED] THE
#: LANDMARK-QUALITY SCREEN**, before any staging.
STOP_2A_II_BUILT = {
    "built": "2026-08-24, NOT launched",
    "the_five_checks": (
        "from facts the mapping already trusts, thresholds declared in "
        "the YAML before the run: (1) DEGENERATE -- empty, non-finite "
        "or collapsed rows, a hard refusal that short-circuits; (2) "
        "BOUNDS -- every point on the image it claims; (3) SPAN -- the "
        "face bbox a plausible fraction of image side; (4) ORDERING -- "
        "brows above eyes above nose-base above mouth; (5) "
        "INTEROCULAR -- eye-centre distance over face span, "
        "SCALE-FREE, so it tests the constellation's shape and not the "
        "image's size"
    ),
    "measured_against_bad_cases": (
        "a screen that cannot fail verifies nothing, so all six were "
        "MEASURED before the checks were trusted: a good face passes; "
        "an all-zero row fails DEGENERATE; a foreign coordinate frame "
        "fails BOUNDS+SPAN (65 of 68 points outside); a collapsed tiny "
        "cloud fails SPAN at 0.0399 -- **and its POSE offset is 0.0006, "
        "which would sail through the 0.08 screen, which is exactly the "
        "blindness this stop exists for**; a scattered set fails "
        "ORDERING; fused eyes fail INTEROCULAR at 0.0252"
    ),
    "the_named_diagnosis": (
        "119.jpg must EXPLAIN ITSELF BY NAME in the report -- row "
        "present? cell parsed? coordinates in what range against the "
        "image's own dimensions? which checks fail? -- because it "
        "passed a screen it should never have reached, and WHATEVER "
        "LET IT THROUGH MAY NOT BE UNIQUE TO IT. The diagnosis is a "
        "DECLARED LIST in the config (119.jpg, girl-3956612, "
        "man-1868320), structural rather than a one-off print"
    ),
    "the_119_hypothesis_not_a_finding": (
        "**HYPOTHESIS, TO BE SETTLED BY THE RUN**: its landmarks lie "
        "OUTSIDE the image's bounds -- the sheet renderer skips "
        "out-of-bounds points and skips the midline column when it "
        "falls outside, which would produce exactly 'no landmarks and "
        "no midline' while a symmetric cloud passed the pose screen. A "
        "foreign coordinate frame (a differently-sized copy of the "
        "same basename) is the mechanism that would explain it. NOT "
        "MEASURED: a remote lookup of the row returned a mismatched "
        "record and was discarded rather than banked"
    ),
    "the_falling_count_is_the_price": (
        "the surviving count REPLACES 1,519 in every later "
        "registration and its fall is recorded as THE PRICE OF "
        "QUALITY -- exactly as the 0.08 band's ~5% was, not a defect "
        "to be argued down"
    ),
    "verdict_decodes_no_pixel": (
        "the screen reads landmark geometry and image DIMENSIONS (PIL "
        "headers); pixels are decoded solely for the sheets, so a "
        "rendering problem cannot corrupt a measurement -- 2a's "
        "property, kept"
    ),
    "second_eye_pass": (
        "fresh sheets of SURVIVORS and NEWLY-REJECTED (rejection "
        "reasons in the labels) go to the eye before any staging; 2b's "
        "split declaration waits for the post-screen numbers"
    ),
}


#: **[BANKED 2026-08-24] STOP 2a-ii: THE QUALITY SCREEN'S VERDICT** --
#: and a failure mode that turned out to be MONOLITHIC.
STOP_2A_II_BANKED = {
    "banked": "2026-08-24, p15-screen-landmarks, single clean attempt",
    "the_screen": (
        "1,464 survive of 1,519 (96.4%); 55 rejected"
    ),
    "monolithic_failure_mode": (
        "**ALL 55 REJECTIONS ARE ON BOUNDS. ZERO on degenerate, span, "
        "ordering or interocular** -- the authors' detector's failure "
        "mode here is MONOLITHIC, not a spread across quality modes. "
        "Leading suspect: a foreign coordinate frame (an "
        "original-vs-cropped resolution mismatch), UNVERIFIED. The "
        "other four checks found nothing, which is itself the finding: "
        "the sets that survive are geometrically sound"
    ),
    "scale_asymmetry_restated": (
        "0.27x SCUT (1,464 vs 5,500), from 0.28x at 1,519 and 0.46x at "
        "the 2,550 headline. The fall is THE PRICE OF QUALITY, the "
        "0.08 band's precedent applied"
    ),
    "diagnoses_two_of_three_unexpected": (
        "the three named diagnoses did NOT come back as expected, and "
        "both surprises were defects in OUR OWN sheet renderer rather "
        "than in the data (SHEET_OVERLAY_DEFECT, SHEET_LABEL_DEFECT)"
    ),
}


#: **[REFUTED 2026-08-24, BY NAME] THE 119.jpg HYPOTHESIS.**
THE_119_HYPOTHESIS_REFUTED = {
    "refuted": "2026-08-24, by the run's own named diagnosis",
    "what_was_registered": (
        "STOP_2A_II_BUILT's hypothesis: 119.jpg's landmarks lie OUTSIDE "
        "the image's bounds -- a foreign coordinate frame -- which "
        "would explain a blank overlay"
    ),
    "what_was_measured": (
        "row present, image present, pose offset 0.0129, quality_failed "
        "EMPTY, coordinates x[1167, 3101] y[2892, 4743] inside a 4000 x "
        "6000 image. **THE LANDMARKS ARE CORRECT AND IN BOUNDS.** "
        "Nothing about the data explains the blank overlay"
    ),
    "the_hypothesis_is_refuted": (
        "REFUTED, recorded by name rather than quietly replaced. The "
        "registered mechanism was wrong, and it was wrong in the "
        "direction that blames the data -- which is the direction to "
        "distrust when the instrument is one's own"
    ),
    "where_the_defect_actually_was": (
        "the MAPPING-SHEET RENDERER (SHEET_OVERLAY_DEFECT), not the "
        "landmarks. A sheet that can silently omit an overlay can hide "
        "a real defect on a later pass; here it MANUFACTURED A FALSE "
        "ALARM, which is the same failure wearing the friendlier face"
    ),
}


#: **[DEFECT 2026-08-24, DIAGNOSED AND FIXED -- MEASURED] THE SHEET
#: OVERLAY VANISHED ON LARGE IMAGES.**
SHEET_OVERLAY_DEFECT = {
    "defect": (
        "2026-08-24: mapping-sheet panels drew the landmark marks and "
        "the midline at SOURCE resolution and THEN decimated the panel "
        "to 224 with nearest-neighbour sampling. The overlay's survival "
        "therefore depended on the source image's SIZE"
    ),
    "the_mechanism_measured": (
        "at 119.jpg's 4000 x 6000 the sampling stride is 17.9 x 26.8 "
        "pixels per sample: of ~1,539 marked pixels FOUR survived, and "
        "the 3-pixel midline column survived only where a sample column "
        "landed on it -- 16.8% of positions. 119.jpg's midline (~2134) "
        "fell between the sampled columns 2125 and 2142 and vanished "
        "entirely. At a typical 500 x 600 the SAME code left 281 pixels "
        "and looked perfectly fine"
    ),
    "why_it_read_as_a_data_defect": (
        "the failure is SCALE-DEPENDENT, so it struck exactly the "
        "largest images -- and a blank overlay on a large photograph "
        "looks precisely like a detector that failed on a hard image. "
        "The renderer's defect wore the data's clothes"
    ),
    "the_fix": (
        "ONE helper, run._overlay_panel: resize FIRST, then draw in "
        "DISPLAY space, so a mark is a constant fraction of the panel "
        "at any source resolution. MEASURED: 1,539 green and 630 red "
        "pixels at BOTH 4000x6000 and 500x600, against 4 green and a "
        "vanished midline before. Both sheet call sites use it"
    ),
    "the_evidence": (
        "the fixed task renders a DIAGNOSIS SHEET of exactly the named "
        "faces, so 119.jpg's panel comes back as the fix's own evidence "
        "rather than as an assurance"
    ),
}


#: **[DEFECT 2026-08-24, DIAGNOSED AND FIXED] SHEET LABELS WERE
#: SILENTLY CUT, AND THE CUT NAMES WERE THEN LOOKED UP.**
SHEET_LABEL_DEFECT = {
    "defect": (
        "2026-08-24: two of the three named diagnoses came back "
        "row=False image=False -- NO SUCH FILENAMES. The names had been "
        "read off sheet labels that were visually cut mid-name"
    ),
    "the_mechanism": (
        "``render.save_sheet`` never truncated at all. It drew each "
        "label at its cell's left edge, so a label wider than the cell "
        "OVERFLOWED into the next column -- and because labels are "
        "drawn in index order, the next column's label OVERDREW its "
        "tail. The last column simply ran off the sheet and was "
        "clipped. The result reads as a truncated name with nothing to "
        "say it was truncated"
    ),
    "the_consequence": (
        "the two flagged faces (girl-3956612, man-1868320) were "
        "named from cut labels, so BOTH DIAGNOSES ARE INCONCLUSIVE, "
        "not clean -- they answered questions about filenames that "
        "never existed. Recorded as inconclusive rather than as "
        "negative findings"
    ),
    "the_fix": (
        "two parts, because trimming alone does not restore a name: "
        "(1) ``render.fit_label`` trims to the cell and MARKS the trim "
        "with an ellipsis, so the loss is visible instead of silent; "
        "(2) every Phase 15 sheet writes a SIDECAR CSV of the FULL "
        "labels beside it, so a name in a panel can always be looked "
        "up. And the diagnosis path now matches by SUBSTRING, so even "
        "a cut label resolves to the face it names"
    ),
    "re_diagnosis_pending": (
        "girl-3956612 and man-1868320 are re-diagnosed BY THE RERUN "
        "through substring matching; if either is among the 55 "
        "bounds-rejects, that closes the loop between the eye's "
        "impression and the screen's verdict. Not claimable until then"
    ),
    "the_general_lesson": (
        "a review artifact is an INSTRUMENT and gets an instrument's "
        "scrutiny: this one could silently omit an overlay AND "
        "silently cut a name, and both defects produced confident "
        "wrong conclusions about the DATA. The sheets wait for the "
        "second eye pass until the renderer is trustworthy"
    ),
}


#: **[BANKED 2026-08-24, FROM THE FIXED DIAGNOSIS SHEET]
#: ALL THREE NAMED FACES, SETTLED.**
DIAGNOSIS_VERDICTS = {
    "banked": (
        "2026-08-24, the visual check on the fixed diagnosis sheet -- the "
        "renderer's own evidence, reviewed"
    ),
    "119_jpg_renders": (
        "**119.jpg RENDERS ITS OVERLAY.** The renderer fix is VERIFIED "
        "BY ITS OWN EVIDENCE, which is the point of rendering the named "
        "faces rather than asserting the fix. The false-alarm chain "
        "closes end to end: the decimation defect MANUFACTURED the "
        "alarm, the display-space fix DISSOLVED it, and THE DATA WAS "
        "CORRECT THROUGHOUT (SHEET_OVERLAY_DEFECT, "
        "THE_119_HYPOTHESIS_REFUTED)"
    ),
    "girl_3956612_withdrawn": (
        "**girl-3956612_1920: WITHDRAWN.** the flag was a false "
        "alarm from the PRE-FIX sheet: label overflow made "
        "panel-to-name attribution unreliable, so the observation was "
        "very likely attached to the WRONG PANEL. Withdrawn by name, "
        "not quietly dropped"
    ),
    "man_1868320_real": (
        "**man-1868320_1920: REAL.** The points sit slightly off the "
        "face (occluding hands) while the screen passes it on EVERY "
        "check: pose 0.0307, in bounds, span fine, ordering fine, "
        "interocular fine"
    ),
    "the_scoreboard": (
        "of three named faces: one renderer defect (119), one "
        "withdrawn false alarm (girl-3956612), one real observation "
        "the screen cannot see (man-1868320). TWO OF THE THREE ORIGINAL "
        "FLAGS WERE ARTEFACTS OF OUR OWN INSTRUMENT"
    ),
}


#: **[WITHDRAWALS 2026-08-24, RECORDED TOGETHER] TWO EYE IMPRESSIONS
#: RETRACTED BY NAME** -- the discipline's cost, and its point.
EYE_IMPRESSION_WITHDRAWALS = {
    "recorded": (
        "2026-08-24, together, deliberately -- two at first, and a "
        "THIRD arrived the same day"
    ),
    "first": (
        "man-2785071 (MAPPING_SHEETS_REVIEWED): the check flagged the "
        "landmarks as drifted; the review found them CORRECT -- "
        "the face is tilted back and bearded, and this was misread the "
        "pose"
    ),
    "second": (
        "girl-3956612_1920 (DIAGNOSIS_VERDICTS): the check flagged the "
        "constellation as floating off the face; the flag came from a "
        "PRE-FIX sheet whose label overflow made panel-to-name "
        "attribution unreliable, so it was very likely attached to the "
        "wrong panel"
    ),
    "third": (
        "the record's THIRD-AXIS FRAMING claim "
        "(FRAMING_VARIES_LIMITATION): proposed as a MEBeauty-vs-SCUT "
        "confound; the maintainer's correction is that SCUT varies the same "
        "way, so it is a shared property of both beauty sources and "
        "not a confound between them"
    ),
    "fourth_and_of_a_different_kind": (
        "**A numeric over-call, withdrawn by name "
        "2026-08-24**: 'the masked arms are below zero -- "
        "anti-correlated with grade'. TRUE of masked@g1 ([-0.0914, "
        "-0.0472], excludes zero); NOT TRUE of masked@g2 ([-0.0665, "
        "+0.0067], spans zero and is indistinguishable from no "
        "signal). Two findings, not one, and 'at or below zero' is the "
        "correct banked wording (READING_1_MASKED_ARMS_AT_ZERO)"
    ),
    "the_tally_by_kind": (
        "THREE withdrawn eye impressions, all the record's; ONE withdrawn "
        "numeric over-call, the maintainer's. **Kept distinct because they "
        "fail differently**: an eye impression is withdrawn when "
        "someone looks again, a numeric claim when someone computes "
        "the interval. The discipline is the same and the evidence "
        "that overturns them is not"
    ),
    "the_cost": (
        "THREE of a second reading impressions are now withdrawn by name, "
        "and the withdrawals are as findable as the observations were. "
        "That is the COST: the record carries recorded errors as "
        "prominently as its readings"
    ),
    "the_point": (
        "and that is also the POINT. An eye impression that cannot be "
        "withdrawn is an eye impression that gets cited forever. The "
        "same retraction discipline applies to impressions as to "
        "numbers -- established when the first was recorded, honoured "
        "when the second arrived, and it has now caught TWO false "
        "alarms that would otherwise have travelled into the write-up "
        "as evidence of a data problem -- and a third that would have "
        "invented a confound between the two beauty sources"
    ),
    "what_survived_it": (
        "man-1868320 -- the one flag that survived both the renderer "
        "fix and the review is the one that became a registered "
        "limitation (ANATOMICALLY_BLIND_LIMITATION). The discipline "
        "did not suppress the real observation; it removed the two "
        "that were not"
    ),
}


#: **[REGISTERED 2026-08-24, A MEASURED LIMITATION -- NOT A DEFECT TO
#: FIX] THE QUALITY SCREEN IS GEOMETRICALLY COMPLETE AND ANATOMICALLY
#: BLIND.**
ANATOMICALLY_BLIND_LIMITATION = {
    "registered": "2026-08-24, from man-1868320's confirmed case",
    "the_limitation": (
        "landmarks can be VALID ON EVERY MEASURABLE AXIS and still sit "
        "slightly wrong on the face. man-1868320_1920 passes pose "
        "(0.0307), bounds, span, ordering and interocular while its "
        "points rest on occluding hands. The five checks are "
        "GEOMETRICALLY COMPLETE -- they test the constellation's shape, "
        "extent, placement and internal consistency -- and "
        "ANATOMICALLY BLIND: nothing in them asks whether the points "
        "are on the ANATOMY they name"
    ),
    "not_building_a_detector": (
        "**WE ARE NOT BUILDING AN ANATOMICAL-CORRECTNESS DETECTOR.** It "
        "would be a NEW UNVALIDATED INSTRUMENT measuring itself -- the "
        "scar-trace precedent from Phase 13 "
        "(phase13.STOP_3_REGISTERED['e_scar_trace_not_testable']), "
        "where a detector with unbounded errors was declined for "
        "exactly this reason. A screen that flags 'landmark not on "
        "anatomy' would need its own validation set, and none exists"
    ),
    "the_rate_is_unquantified": (
        "ONE CONFIRMED INSTANCE out of THREE NAMED FACES -- and the "
        "three were not a sample, they were the faces the record happened "
        "to flag, two of which turned out to be instrument artefacts. "
        "**THE RATE IS UNQUANTIFIED**, and no rate may be quoted from "
        "this"
    ),
    "the_catch_net": (
        "the STAGED-SHEET EYE GATE at stop 2b is the catch-net: staged "
        "crops make a mis-placed trapezium obvious in a way a landmark "
        "overlay does not, and the eye reviews them before any "
        "embedding exists"
    ),
    "the_trigger_to_requantify": (
        "**IF THE STAGED SHEETS SHOW THIS CLASS AT ANY NOTICEABLE "
        "RATE, THE LIMITATION GETS REQUANTIFIED BEFORE PRETRAINING** "
        "-- written now, so the decision is not made after seeing "
        "numbers one would rather not act on"
    ),
}


#: **[REVIEWED 2026-08-24] THE SECOND EYE PASS** -- survivors pass, and
#: the eye disputes the rejects.
SECOND_EYE_PASS = {
    "reviewed": (
        "2026-08-24, the maintainer and the record, public data -- on the FIXED "
        "sheets, with honest labels and their sidecar CSVs"
    ),
    "survivors_PASS": (
        "**PASS FOR STAGING.** Anchors on anatomy across both sheets, "
        "including glasses, beards and 119.jpg"
    ),
    "two_mild_instances_EYE_IMPRESSION": (
        "jon-ly-ADBOC3UP4eQ (constellation low-left on a tilted face) "
        "and girl-3447599 (slightly off) -- the man-1868320 class, "
        "**EYE-IMPRESSION not measurement**, consistent with "
        "ANATOMICALLY_BLIND_LIMITATION at a LOW, UNQUANTIFIED rate. NO "
        "NEW INSTRUMENT is built for them"
    ),
    "rejects_disputed": (
        "**THE EYE SAYS MOST OF THE 55 DO NOT DESERVE REJECTION**: "
        "pexels-cottonbro, models-2158971, baby-5925923, man-2785071 "
        "and most of the rest are frontal with anatomically correct "
        "landmarks"
    ),
    "the_suspected_mechanism": (
        "all 55 failed on BOUNDS, so the likely mechanism is TIGHT "
        "CROPS where a legitimately-correct landmark set extends to or "
        "past the image edge (jaw or brow at the border, or detector "
        "extrapolation a pixel or two out) -- **a threshold problem in "
        "OUR CHECK, not a detection problem in the data**. Suspected, "
        "not established: the distribution decides"
    ),
    "man_2785071_returns": (
        "worth noting by name: man-2785071 -- the face whose landmarks "
        "the record first flagged and the maintainer cleared "
        "(EYE_IMPRESSION_WITHDRAWALS) -- is among the disputed "
        "rejects. The same face has now survived a false flag AND a "
        "suspected false rejection"
    ),
}


#: **[REGISTERED 2026-08-24, BEFORE THE NUMBER EXISTS] HOW THE BOUNDS
#: VIOLATION WILL BE READ.**
#:
#: The eye has spoken and the check disagrees with it. That is exactly
#: the situation in which a threshold gets quietly moved until the two
#: agree, so the readings are written FIRST and the measurement is built
#: to report a DISTRIBUTION rather than a verdict.
BOUNDS_VIOLATION_READINGS = {
    "registered": (
        "2026-08-24, before any violation number exists -- after the "
        "eye pass, which is why the provenance below is stated plainly"
    ),
    "what_is_measured": (
        "for all 55 bounds-rejects: how many of the 68 points fall "
        "outside, and by how much -- max and median overshoot, in "
        "PIXELS and as a FRACTION of the image dimension. Reported as a "
        "histogram and quantiles (mebeauty.bounds_overshoot), never as "
        "a pass/fail"
    ),
    "reading_if_marginal": (
        "IF THE MASS SITS AT 1-2 POINTS OVERSHOOTING BY A SMALL "
        "FRACTION of the image -> the check is TOO STRICT and gets a "
        "declared tolerance (points-outside count and/or overshoot "
        "fraction), both declared in the YAML BEFORE any re-screen"
    ),
    "reading_if_split": (
        "IF A SUBSET OVERSHOOTS BY LARGE MARGINS (the "
        "foreign-coordinate-frame suspects) -> THOSE STAY REJECTED, "
        "and the two populations are REPORTED SEPARATELY rather than "
        "averaged into one tolerance"
    ),
    "either_way": (
        "the surviving count moves and the scale asymmetry RESTATES "
        "AGAIN -- 0.27x is not final"
    ),
    "the_provenance_stated_plainly": (
        "**THIS TOLERANCE WOULD BE CHOSEN FROM A MEASURED DISTRIBUTION "
        "AFTER AN EYE PASS. That is written here, in the record, "
        "rather than hidden in a config diff.** The project's standing "
        "rule is that a threshold moved after seeing the data is not a "
        "threshold; the exception is argued, not assumed"
    ),
    "why_this_differs_from_the_0_08_precedent": (
        "the 0.08 pose threshold was NOT moved when the eye found "
        "recoverable faces in the 0.08-0.12 band: it was doing exactly "
        "what it was designed to do, and the ~5% loss was recorded as "
        "the price (STOP_1_BANKED). The bounds check is different in "
        "kind -- it may be measuring something OTHER than what it was "
        "designed to measure: 'the landmark set belongs to this image' "
        "was the intent, and 'no point touches the border' may be a "
        "strict-equality artefact of that intent. Fixing an "
        "instrument that measures the wrong thing is not the same act "
        "as loosening one that measures the right thing"
    ),
    "the_value_is_not_proposed_here": (
        "**NO TOLERANCE VALUE IS PROPOSED IN THIS RECORD.** The "
        "mechanism is built (bounds_max_points_outside beside the "
        "existing bounds_margin, BOTH strict by default so the shipped "
        "behaviour is unchanged); the VALUE waits for the distribution, "
        "because a number chosen now would be tuned to the eye's "
        "verdict -- which is the thing this record exists to prevent"
    ),
}


#: **[BANKED 2026-08-24] THE BOUNDS DISTRIBUTION: BRANCH 1 FIRES,
#: BRANCH 2 HAS NO INSTANCE.**
BOUNDS_DISTRIBUTION_BANKED = {
    "banked": "2026-08-24, measured over all 55 bounds-rejects",
    "points_outside_histogram": (
        "1:7, 2:9, 3:22, 4:6, 5:3, 6-10:8, 11-20:0, 21-68:0 -- median "
        "3, p90 7, MAX 9 of 68"
    ),
    "overshoot": (
        "max overshoot per face: median 7.0 px, p90 24.8 px, max 39 px; "
        "as a fraction of the image dimension, median 1.6%, p90 5.7%, "
        "MAX 7.8%"
    ),
    "branch_1_fires": (
        "**THE CHECK IS TOO STRICT**, as registered "
        "(BOUNDS_VIOLATION_READINGS['reading_if_marginal']). Every "
        "reject is a TIGHT-CROP case -- a few jaw or brow points a few "
        "pixels past the border"
    ),
    "branch_2_has_no_instance": (
        "**NO SECOND POPULATION.** The foreign-coordinate-frame family "
        "the bounds check was built against has NO INSTANCE in this "
        "dataset: the synthetic case that motivated it (65 points "
        "outside, 169% overshoot) has no counterpart anywhere in the "
        "55. The registered split-reading does not fire, and saying so "
        "is part of reading it"
    ),
    "the_suspicion_was_wrong": (
        "STOP_2A_II_BANKED named an original-vs-cropped resolution "
        "mismatch as the leading suspect for the monolithic bounds "
        "failure. The distribution REFUTES it: a foreign frame would "
        "put most of the 68 points far outside, and the observed mass "
        "is 1-4 points a few pixels out. Recorded rather than quietly "
        "dropped -- the suspicion was the instrument's, not the data's"
    ),
}


#: **[RULED 2026-08-24] THE GENEROUS TOLERANCE**, with its
#: provenance in the open.
BOUNDS_TOLERANCE_RULED = {
    "ruled": "2026-08-24 -- the GENEROUS tolerance",
    "the_values": (
        "bounds_max_points_outside = 10 and overshoot cap "
        "(bounds_margin) = 0.10, both JUST ABOVE the observed maxima "
        "of 9 points and 7.8%. Declared in the YAML as a VISIBLE CONFIG "
        "DIFF from the shipped strict 0 / 0.0"
    ),
    "the_provenance_in_the_open": (
        "**THE VALUE WAS CHOSEN FROM A MEASURED DISTRIBUTION AFTER AN "
        "EYE PASS.** That sentence is in the record, in the config "
        "header, and in this field -- not hidden in a diff. The "
        "project's standing rule is that a threshold moved after "
        "seeing the data is not a threshold; this is the argued "
        "exception, and it is argued where it can be found"
    ),
    "why_generous_rather_than_tight": (
        "the distribution is a SINGLE CONTINUOUS MODE with no "
        "bimodality to cut at, so a conservative cut inside that mode "
        "would have NO MEASURED BASIS -- it would be a number chosen "
        "for the comfort of looking strict. The eye said the 6-10 "
        "bucket looks fine and the numbers say it is not a different "
        "population; the two agree, and the tolerance follows the "
        "measurement rather than splitting the difference"
    ),
    "what_stays_untouched": (
        "the 0.08 pose threshold, per the distinction registered "
        "before this measurement existed "
        "(BOUNDS_VIOLATION_READINGS['why_this_differs_from_the_0_08_"
        "precedent']): that check was doing what it was designed to "
        "do, and its ~5% cost stands as the price"
    ),
}


#: **[STATED PLAINLY 2026-08-24] THE CONSEQUENCE, NOT GLOSSED.**
TOLERANCE_CONSEQUENCE = {
    "stated": "2026-08-24, plainly, because it would be easy to gloss",
    "the_consequence": (
        "**AT THIS TOLERANCE THE QUALITY SCREEN REJECTS NOTHING ON "
        "MEBEAUTY.** The usable count returns to 1,519, the scale "
        "asymmetry to 0.28x, and the screen's value ON THIS DATASET is "
        "PURELY PROTECTIVE"
    ),
    "still_protective": (
        "it still FAILS the foreign-frame case it was built for, and "
        "would still catch collapsed clouds, broken vertical ordering "
        "and degenerate rows -- all measured against synthetic cases "
        "(STOP_2A_II_BUILT). A guard that catches nothing on the data "
        "it guards is not thereby a guard that catches nothing"
    ),
    "what_the_exercise_produced": (
        "it is NOT nothing: (1) **the authors' landmarks are SOUND "
        "WITHIN THE USABLE SET** -- itself a banked finding about the "
        "source, and one no amount of reasoning would have "
        "established; (2) the screen's sheets caught TWO RENDERER "
        "DEFECTS (overlay decimation, label overflow) and ONE REAL "
        "ANATOMICAL CASE (man-1868320)"
    ),
    "the_tally": (
        "**THREE INSTRUMENT DEFECTS AGAINST ONE DATA DEFECT, THIS "
        "PHASE.** Recorded because it is the honest ratio and because "
        "it predicts where the next defect will be: in what we built "
        "to look with, not in what we are looking at"
    ),
}


#: **[BUILT 2026-08-24, STOP 2b -- NOT LAUNCHED] THE STAGING.**
STOP_2B_BUILT = {
    "built": "2026-08-24, NOT launched -- the staging run is the maintainer's",
    "composition_is_scuts": (
        "masked.build_one VERBATIM: crop, arrival mask, staging, G2 "
        "unwarp. The ONLY MEBeauty-specific step is the content box -- "
        "dlib-68 ANCHORS (mebeauty.vertical_span, mebeauty.midline_x) "
        "with the SHARED formula (placement.box_from_anchors, extracted "
        "so the two datasets cannot be framed differently while both "
        "configs say the same thing). build_one gained an optional "
        "`box`, defaulting to None, so every SCUT call stays "
        "byte-identical to what the bit-for-bit test pins"
    ),
    "inherited_not_restated": (
        "levelling OFF (SCUT_HEAD_TILT_LEVELLING, the matched-property "
        "argument); CLEFT_AR parity sampling from the 237 observed "
        "ratios; LOWER_MARGIN 0.06 imported, never redefined; the "
        "frozen staging and trapezium. Parity is RE-RUN on the new "
        "artifact and reported as gaps, never as a verdict"
    ),
    "reproduced_not_redecided": (
        "both screens are DECLARED in the config and the surviving "
        "count is ASSERTED (expect_survivors 1519), so 2b stages the "
        "set 2a-ii measured rather than a set it decides for itself; "
        "the split halves are asserted too (932 / 394)"
    ),
    "stage_all_split_recorded": (
        "ALL 1,519 survivors are staged and each carries its split "
        "(train / test / UNCOVERED), so pretraining selects the "
        "split-covered 1,326 without a second staging pass. This is "
        "the stage-all-pretrain-covered proposal from STOP_2A_BANKED, "
        "taken as agreed; if that reading of 'as agreed' is wrong, the "
        "split column makes the correction a config change and not a "
        "re-run"
    ),
    "the_eye_gate": (
        "staged sheets at BOTH geometries go to the visual check BEFORE "
        "any embedding -- the catch-net for "
        "ANATOMICALLY_BLIND_LIMITATION, whose requantify trigger was "
        "written before these numbers existed"
    ),
    "artifact": (
        "data/mebeauty_staged/mebeauty_masked_v1 -- built in "
        "<version>.inprogress and renamed on completion, memmapped one "
        "face resident, with faces.csv (stem, split, realised AR, pad "
        "and white fractions, box-inside-frame), metadata.json and "
        "MANIFEST.json. A DERIVED artifact, so it lives under data/ -- "
        "the mirror of the source-clone correction"
    ),
}


#: **[DIAGNOSED 2026-08-24 -- THE HYPOTHESIS IS REFUTED] THE PARITY
#: None: AN ASYMMETRIC REFERENCE SCHEMA, NOT AN EMPTY GROUP.**
PARITY_NONE_DIAGNOSED = {
    "defect": (
        "2026-08-24: p15-stage-mebeauty failed all four attempts (+3 "
        "retries) at the parity summary -- TypeError: unsupported "
        "format string passed to NoneType.__format__. The STAGING "
        "succeeded every time: 1,519 survivors reproduced, split "
        "asserted 932/394/193, 1,519 faces staged in ~20s"
    ),
    "the_measured_cause": (
        "**REPRODUCED LOCALLY through real build_one records.** "
        "``staging.CLEFT_STAGED_GEOMETRY`` carries DIFFERENT statistics "
        "for different quantities: aspect_ratio has min/median/max (NO "
        "mean), pad_fraction has min/mean/max (NO median). "
        "``parity_report`` handles that correctly and by design -- it "
        "returns None for the statistic the reference does not carry. "
        "So aspect_ratio.median_gap = +0.0265 and mean_gap = None; "
        "pad_fraction.mean_gap = -0.0458 and median_gap = None. **The "
        "crash site was the TASK'S LOG LINE**, which formatted "
        "median_gap for EVERY key"
    ),
    "why_nothing_printed": (
        "the line was built with ``', '.join(...)``, which is fully "
        "evaluated BEFORE ctx.log is called -- so the exception fired "
        "during string construction and not one parity character "
        "reached the log. That is exactly what the run showed: no "
        "parity, gap or median line in any of the four attempts"
    ),
    "the_hypothesis_refuted": (
        "**IT IS NOT AN EMPTY GROUP AND NOT AN EMPTY AR BIN.** The "
        "registered suspicion was a cleft-AR bin with no MEBeauty face "
        "in it, which would have been a measured COVERAGE GAP and a "
        "finding. It is refuted: both gaps were computed over all "
        "1,519 records and the None marks a statistic the REFERENCE "
        "never recorded. **No coverage finding is warranted on this "
        "evidence, and none is registered** -- inventing one from a "
        "formatting crash would be the opposite of the discipline that "
        "asked for the diagnosis"
    ),
    "where_the_coverage_question_does_live": (
        "the question is real and already has a home in the report: "
        "``n_box_outside_frame`` and the REQUESTED-vs-REALISED aspect "
        "ratios, both of which parity_report already carries. In the "
        "local reproduction every box was inside its frame "
        "(n_box_outside_frame = 0) and the requested/realised "
        "divergence was at most 0.003 -- INTEGER PIXEL ROUNDING on a "
        "~270px box, not clipping. The real answer comes from the "
        "re-run, on real images"
    ),
    "inprogress_worked_as_designed": (
        "**THE .inprogress CONVENTION DID ITS JOB.** The directory "
        "survives at mebeauty_masked_v1.inprogress holding exactly "
        "staged_g1.npy and staged_g2.npy (437M) -- pixels with no ids, "
        "no split column, no MANIFEST. The rename gate WITHHELD, "
        "because the artifact genuinely was incomplete: the metadata "
        "that makes the tensors meaningful had not been written. A "
        "convention that only pays off on failure paid off"
    ),
    "inprogress_disposition": (
        "**NOT REUSABLE, and the code REMOVES IT AUTOMATICALLY.** "
        "Nothing in the directory says which row is which face, so no "
        "resume is possible even in principle; restaging costs ~20s, "
        "so inventing a resume for it would be machinery earning "
        "nothing. The next run logs the removal LOUDLY with the reason "
        "before deleting -- it is hundreds of megabytes, and a silent "
        "rmtree of that size is not something a reader should have to "
        "infer"
    ),
    "the_three_fixes": (
        "(1) THE REPORTER REFUSES AND REPORTS: parity_report now "
        "carries an `undefined` map naming each missing statistic, the "
        "reference keys that do exist, and the member count -- so a "
        "None can never again be mistaken for an empty group. The "
        "reading-count guard's principle applied to summary "
        "statistics. (2) THE TASK reports every DEFINED gap and names "
        "the undefined ones, per key, so no single missing statistic "
        "can suppress the whole line. (3) WORK-COMPLETED-REPORT-FAILED "
        "announces itself: the run logs STAGING COMPLETE before any "
        "reporting begins, and a reporting failure logs 'WORK "
        "COMPLETED, REPORT FAILED' with the surviving directory and "
        "its disposition before re-raising"
    ),
}


#: **[FIXED 2026-08-24] A REGISTERED SENTENCE FIRED WHERE ITS CONDITION
#: DID NOT HOLD** -- small, and the same shape as a large defect.
PRICE_SENTENCE_FIRED_ON_NO_LOSS = {
    "defect": (
        "2026-08-24: p15-screen-landmarks-4 printed the "
        "price-of-quality sentence ('the falling count...') at a run "
        "where 1,519 of 1,519 survived and NOTHING FELL"
    ),
    "why_it_matters_despite_being_cosmetic": (
        "it is the auto-applied-reading defect in miniature "
        "(phase13.READING_COUNT_GUARD): a sentence registered for one "
        "condition, printed where that condition does not hold. Here "
        "it cost nothing; the same shape one level up applied a "
        "registered reading to a computation over nobody"
    ),
    "the_fix": (
        "the clause is CONDITIONAL on an actual loss and states the "
        "loss when there is one; where nothing fell the run says so "
        "explicitly rather than staying silent, so the absence is "
        "visible too"
    ),
}


#: **[BANKED 2026-08-24] STOP 2b: THE ARTIFACT EXISTS.**
STOP_2B_BANKED = {
    "banked": "2026-08-24, p15-stage-mebeauty-2, single clean attempt",
    "the_run": (
        "the incomplete .inprogress was removed with its reason logged; "
        "1,519 staged at two geometries; splits asserted 932 / 394 / "
        "193; STAGING COMPLETE printed BEFORE reporting; artifact "
        "mebeauty_masked_v1 renamed. Rollup 195d80ab"
    ),
    "parity": (
        "aspect_ratio median gap -0.0007, pad_fraction mean gap "
        "+0.0031 -- both NEAR ZERO against CLEFT_STAGED_GEOMETRY. The "
        "two undefined statistics were named with their reference keys "
        "and their non-empty member count: the `undefined` map working "
        "as built (PARITY_NONE_DIAGNOSED)"
    ),
    "the_fixes_held": (
        "every fix from the failed attempt did its job on the first "
        "try: the loud removal, the completion announcement before "
        "reporting, the per-key gap reporting, and the undefined map"
    ),
}


#: **[MEASURED PROPERTY 2026-08-24 -- NOT A LOG LINE] EVERY MEBEAUTY
#: CROP IS PADDED.**
PADDING_IS_AN_ARTIFACT_PROPERTY = {
    "measured": (
        "2026-08-24: the content box is inside the source frame for **0 "
        "of 1,519 faces**, and the visual check confirms it literally -- "
        "visible white padding on essentially every crop, at BOTH "
        "geometries"
    ),
    "the_cause": (
        "MEBeauty is TIGHT IN-THE-WILD PORTRAITURE, so a trapezium "
        "spanning brow to below-mouth AT CLEFT PROPORTIONS runs past "
        "the image edge almost always. The box is not wrong; the source "
        "photographs simply do not extend as far as the cleft framing "
        "asks"
    ),
    "consistent_with_the_parity": (
        "and it is CONSISTENT with the near-zero pad_fraction gap "
        "(+0.0031): the cleft crops are padded too, so a padded "
        "MEBeauty crop is closer to the cleft distribution than an "
        "unpadded one would be. The two facts agree rather than "
        "conflict"
    ),
    "it_travels": (
        "**REGISTERED AS AN ARTIFACT PROPERTY, NOT AN OBSERVATION**: it "
        "travels with ANY MEBeauty result in this phase and the "
        "write-up. A reader comparing MEBeauty and SCUT pretraining is "
        "comparing one source whose every crop is padded against one "
        "whose crops mostly are not, and that belongs in the sentence, "
        "not in a footnote"
    ),
    # [CORRECTED 2026-08-24 -- the text above is preserved as written
    # and is WRONG in its number and in its comparison. See below.]
    "corrected_the_number_was_a_null_read": (
        "**THE 0/1,519 WAS NOT A MEASUREMENT.** The v1 staging task "
        "read ``record.get(\"inside_frame\")`` while ``build_one`` "
        "writes ``box_inside_frame`` -- so every row carried None, and "
        "``sum(1 for r in rows if r[...])`` counted None as falsy. "
        "0 of 1,519 was a null read wearing a statistic's clothes. The "
        "v2 task reads the real key and reports **1,516 of 1,519 boxes "
        "INSIDE the frame**. The two runs never disagreed about the "
        "data; one of them was not reading it"
    ),
    "corrected_two_different_questions": (
        "and the eye was right at the same time, because **'box inside "
        "the frame' and 'crop carries pad' ARE DIFFERENT QUESTIONS**. "
        "The masked crop is cut at a CLEFT ASPECT RATIO -- non-square "
        "by construction -- and the frozen ``stage`` then pads it to "
        "square. So essentially every masked crop carries pad WHETHER "
        "OR NOT its box was inside the frame. Both observations were "
        "true; only the number was wrong"
    ),
    "corrected_the_property_restated": (
        "**STATED IN A QUANTITY THAT IS ACTUALLY MEASURED**: the "
        "masked variants' PAD FRACTION, which the artifact carries. "
        "The parity gap is +0.0031 against the cleft cohort's own mean "
        "of 0.2534 (staging.CLEFT_STAGED_GEOMETRY), so the MEBeauty "
        "masked crops average about **0.2565 pad** -- essentially the "
        "cleft cohort's own figure. That is the AR-parity design "
        "WORKING, not a defect"
    ),
    # [ACCEPTED AS BINDING 2026-08-24, the maintainer, with one amendment:
    # the flag below travels AS PART OF the caveat, not beside it.]
    "accepted_as_binding": (
        "2026-08-24: the corrected caveat is BINDING. The masked arms' "
        "~0.2565 against the cohort's 0.2534 (gap +0.0031) is the "
        "AR-parity design working; the original arm's ~0.0019 with 10 "
        "of 1,519 padded is negligible. Both are quoted from MEASURED "
        "distributions"
    ),
    "travels_as_shared_construction": (
        "**AMENDED: the caveat travels as a property of the "
        "COMPARISON'S SHARED CONSTRUCTION, not as a MEBeauty-specific "
        "hazard.** SCUT's masked arm goes through the same build_one "
        "path, so padding is not a MEBeauty-vs-SCUT differentiator and "
        "must not be quoted as one"
    ),
    "the_unmeasured_half_named": (
        "**SCUT'S OWN MASKED PAD FIGURE IS NOT RE-MEASURED HERE.** The "
        "claim rests on CONSTRUCTION IDENTITY -- the same build_one, "
        "the same cleft-AR crop, the same square staging -- and not on "
        "a comparison of two numbers. **IF ANY READING EVER TURNS ON "
        "THE PAD COMPARISON, THAT FIGURE MUST BE MEASURED FIRST RATHER "
        "THAN INFERRED.** Written here so a later turn reaching for the "
        "comparison finds the condition attached to it"
    ),
    "corrected_it_is_not_a_differentiator": (
        "and therefore it is **NOT a MEBeauty-vs-SCUT caveat**: SCUT's "
        "masked arm goes through the SAME build_one path -- cut at a "
        "cleft-sampled AR, staged to square -- so it is padded by the "
        "same construction. The padding is a property of the MASKED "
        "PIPELINE, shared by both beauty sources and matched to the "
        "cohort on purpose. (SCUT's own pad figure is not re-measured "
        "here; the mechanism is identical by construction, and that is "
        "what the claim rests on.) The framing-variance correction had "
        "exactly this shape: a property proposed as a difference "
        "between the two sources, which both of them share"
    ),
}


#: **[REVIEWED 2026-08-24] THE STAGED SHEETS: PASS.**
STAGED_SHEETS_REVIEWED = {
    "reviewed": "2026-08-24, reviewed -- PASS",
    "verdict": (
        "framing is ANATOMICALLY CORRECT and recognisably THE SAME "
        "INSTRUMENT: nose and mouth centred, brow at the top, the "
        "trapezium narrowing upward, G1 and G2 behaving as expected. "
        "The demographic spread is visible throughout -- mechanism i, "
        "in staged pixels"
    ),
    "the_catch_net_caught_nothing_new": (
        "ANATOMICALLY_BLIND_LIMITATION's requantify trigger did NOT "
        "fire: the staged sheets show no noticeable rate of the "
        "landmarks-valid-but-slightly-wrong class, so the limitation "
        "stands as registered and pretraining is not blocked on it"
    ),
}


#: **[LIMITATION 2026-08-24 -- THE GOVERNING READING, AND THE
#: OVER-CALL IT WITHDRAWS] IN-THE-WILD FRAMING VARIES, AND IT IS NOT A CONFOUND
#: BETWEEN THE TWO BEAUTY SOURCES.**
FRAMING_VARIES_LIMITATION = {
    "registered": "2026-08-24, the maintainer's reading, and it is the operative one",
    "what_was_proposed": (
        "the check flagged MEBeauty's vertical framing as VARYING (some "
        "crops eyes-and-all, others nose-to-chin) and proposed it as a "
        "THIRD AXIS OF UNATTRIBUTABILITY beside demographics and size"
    ),
    "the_correction_is_the_reading": (
        "**SCUT VARIES THE SAME WAY.** The cleft cohort's crops all "
        "show the same anatomy because those photographs were taken to "
        "a CLINICAL PROTOCOL; SCUT and MEBeauty are IN-THE-WILD "
        "photographs where face size in frame varies, so a "
        "landmark-anchored trapezium lands on different anatomy per "
        "image -- some show eyes, some chin, some a cigarette. It is "
        "therefore **NOT a MEBeauty-vs-SCUT confound**: it is a SHARED "
        "PROPERTY OF BOTH BEAUTY SOURCES and a DIFFERENCE BETWEEN BOTH "
        "OF THEM AND THE CLEFT COHORT"
    ),
    "the_over_call_withdrawn": (
        "**WITHDRAWN BY NAME**: the third-axis framing was an OVER-CALL "
        "-- it would have added a confound to the MEBeauty-vs-SCUT "
        "comparison that does not exist there, and would have made the "
        "comparison look weaker than it is for a reason that applies "
        "equally to both arms. Third eye-impression withdrawal this "
        "phase, after man-2785071 and girl-3956612"
    ),
    "nothing_can_be_done": (
        "no fix is available: the variation is a property of "
        "photographs taken without a protocol, and levelling or "
        "re-anchoring would impose a uniformity neither beauty source "
        "has. STATED ONCE, and it travels with every transfer result "
        "in this phase and the write-up"
    ),
    "a_measurement_declined": (
        "an EYES-FRACTION measurement (what proportion of staged crops "
        "include the eyes) was CONSIDERED AND DECLINED as "
        "DECISION-IRRELEVANT: no value it could return would change "
        "the pretraining, the comparison, or this limitation's "
        "wording. Recorded because a declined measurement is a "
        "decision, and an undeclared one looks like an oversight"
    ),
}


#: **[PROPOSED 2026-08-24 -- NOTHING BUILT] STOP 3: THE PRETRAINING AND
#: THE COMPARATIVE VERDICT.**
STOP_3_PROPOSED = {
    "proposed": "2026-08-24, for the maintainer's agreement or amendment",
    "the_arms": (
        "masked pretraining on the SPLIT-COVERED 1,326 (train 932 fits, "
        "test 394 held out -- the shipped split, never our own re-cut) "
        "at BOTH geometries, the Phase 6 recipe BYTE-IDENTICAL. The "
        "193 uncovered faces are staged and carry split='uncovered'; "
        "they are NOT pretrained on"
    ),
    "checkpoint_discipline": (
        "VARIANT_FOR_INIT applies: a masked init is GEOMETRY-BOUND, so "
        "MEBeauty produces ONE CHECKPOINT PER GEOMETRY and every "
        "downstream comparison declares which. The phantom-findings "
        "trap is the reason -- a single weak artifact bound by a rule "
        "to one level of a factor is indistinguishable from an effect "
        "of that factor"
    ),
    "the_comparators_are_banked_and_geometry_specific": (
        "Phase 6's own cells, verified from the record: at G1 "
        "vit_b16 imagenet 0.2520 (sd 0.0148), scut_original 0.1952, "
        "**scut_masked 0.0830**; at G2 **scut_masked 0.2001** (sd "
        "0.0142). **THE MASKED COMPARATOR DIFFERS BY GEOMETRY BY MORE "
        "THAN THE EFFECT ANYONE EXPECTS TO MEASURE** (0.0830 vs "
        "0.2001), so the verdict is made WITHIN geometry and NEVER "
        "pooled across it -- like against like, mebeauty_masked@G1 "
        "against scut_masked@G1"
    ),
    "the_verdict_statistic_proposed": (
        "the delta against the banked SCUT cell in the SAME geometry, "
        "thresholded from BOTH arms' own five-seed SDs (PLAN 4.12.1, "
        "inherited never), DECLARED IN THE CONFIG before any number. "
        "DESCRIPTIVE ranking; PLAN 4.3 machinery only if a gap clears "
        "the paired bar -- and COHORT_CANNOT_RESOLVE says to expect it "
        "will not"
    ),
    "readings_to_register_before_any_number": (
        "MEBEAUTY BETTER -> a demographically broader source transfers "
        "better to this cohort than SCUT, AT 0.28x THE DATA, which is "
        "the strong direction of the registered asymmetry. MEBEAUTY "
        "WORSE -> ambiguous between demographics and size and NOT "
        "attributable, per mechanism ii. UNRESOLVABLE -> the ranking "
        "is reported with both spreads and Phase 16 consumes it as "
        "'no measured difference', which is still a measured answer"
    ),
    "the_asymmetry_sentence_is_bound": (
        "**EVERY reading carries the mechanism-ii sentence** (a "
        "MEBeauty win is STRONG evidence, a loss is AMBIGUOUS), and "
        "PADDING_IS_AN_ARTIFACT_PROPERTY and "
        "FRAMING_VARIES_LIMITATION travel with the result too -- three "
        "sentences, bound to the number before the number exists"
    ),
    "nothing_built": (
        "no config, no task, no arm. The plan is for agreement first"
    ),
}


#: **[AGREED 2026-08-24 -- FOUR AMENDMENTS] STOP 3, BUILT.**
STOP_3_AMENDED = {
    "agreed": "2026-08-24 -- the plan with four amendments",
    "amendment_1_within_geometry": (
        "**BINDING, AND ASSERTED IN CODE** (mebeauty.verdict_delta "
        "refuses a cross-geometry pair): masked SCUT sits at 0.0830 "
        "(G1) and 0.2001 (G2), a **0.117 SPREAD LARGER THAN ANY "
        "PLAUSIBLE DATASET EFFECT**, so a pooled verdict would report "
        "a GEOMETRY difference wearing a DATASET's name. "
        "mebeauty_masked@G1 vs scut_masked@G1, @G2 vs @G2, never across"
    ),
    "amendment_2_original_arm": (
        "MEBeauty needs an ORIGINAL (uncropped) arm: the plan as "
        "proposed filled two of the three cells SCUT occupies. Full "
        "uncropped faces from the same covered split, staged EXACTLY "
        "as SCUT's original arm stages its faces -- the frozen "
        "``stage`` (pad-to-square then resize), no landmarks, no "
        "trapezium, no content box -- which is what "
        "``pretrain.load_original_features`` does. **Why it matters**: "
        "scut_original at 0.1952 is SCUT's BEST transfer cell, so if "
        "MEBeauty's demographic breadth helps anywhere, the record "
        "predicts the uncropped arm is where it shows"
    ),
    "amendment_3_imagenet_anchor": (
        "**THE IMAGENET ANCHOR RIDES IN EVERY READING.** The phase's "
        "registered question is whether a second beauty dataset "
        "transfers better than SCUT, but the DECISION-RELEVANT bar is "
        "ImageNet's 0.2520: the ladder's finding is that SCUT "
        "pretraining is WORSE THAN NO BEAUTY PRETRAINING AT ALL. Each "
        "reading states BOTH comparisons -- against its same-named "
        "SCUT cell (the ranking Phase 16 consumes) and against "
        "ImageNet. **An arm that beats SCUT while still losing to "
        "ImageNet is the most likely outcome and must read as such, "
        "not as a win**"
    ),
    "amendment_4_caveats_bound": (
        "FOUR caveats bound to any masked-arm number, THREE to the "
        "original arm: mechanism-ii asymmetry (0.28x the data -- a win "
        "is strong evidence, a loss unattributable), the padding "
        "property (0/1,519 boxes inside frame), the framing-variance "
        "limitation (shared with SCUT, distinct from the "
        "protocol-photographed cohort), and anatomical blindness "
        "(man-1868320's class, unquantified rate, staged-sheet gate "
        "passed)"
    ),
    "the_build": (
        "three pretraining runs -- masked_g1, masked_g2, "
        "masked_original -- on the covered 1,326 (932 fit / 394 held "
        "out, the shipped universal split), the Phase 6 recipe "
        "byte-identical, VARIANT_FOR_INIT giving ONE checkpoint per "
        "source. MEBeauty reaches the frozen recipe through TWO "
        "default-preserving seams: the artifact carries the SCUT "
        "masked-artifact FORMAT so load_masked_features reads it "
        "unchanged, and `labels_by_stem` supplies MEBeauty's split. "
        "Every existing SCUT call is byte-identical"
    ),
    "the_verdict_statistic": (
        "delta against the SAME-NAMED banked cell, thresholded from "
        "BOTH arms' own five-seed SDs (PLAN 4.12.1, inherited never), "
        "DECLARED in the config before any number. DESCRIPTIVE "
        "ranking; PLAN 4.3 machinery only if a gap clears the paired "
        "bar, with COHORT_CANNOT_RESOLVE as the registered expectation"
    ),
}


#: **[REFINEMENT 2026-08-24 -- MEASURED] THE ORIGINAL ARM PADS TOO, AND
#: SCUT'S DOES NOT.** Amendment 2 held that the uncropped arm sidesteps
#: the padding caveat. Measured, that is not quite right.
ORIGINAL_ARM_PADS_TOO = {
    "the_amendment_said": (
        "the original arm sidesteps two of the four caveats -- no "
        "padding property, no framing variance -- since the whole face "
        "is present either way"
    ),
    "what_is_measured": (
        "FRAMING VARIANCE: correct, and it drops. **PADDING: NOT "
        "QUITE.** The frozen ``stage`` pads to square BEFORE resizing. "
        "SCUT's sources are 350x350 SQUARE, so stage() is a pure "
        "resize there and pad_fraction is 0.0 -- measured. MEBeauty's "
        "are in-the-wild and NON-SQUARE (119.jpg is 4000x6000), so "
        "stage() pads: a 500x600 source measures **0.1652 pad "
        "fraction**"
    ),
    "the_correction": (
        "so mebeauty_original does not lose the padding caveat -- it "
        "SWAPS one padding story for another. The masked arms are "
        "padded because the trapezium runs past the image edge; the "
        "original arm is padded because the image is not square and "
        "SCUT's is. **The caveat set is still SMALLER (framing "
        "variance drops) but it is not padding-free**, and the "
        "difference is against the very cell it compares to"
    ),
    "how_it_is_handled": (
        "the staging run MEASURES the original variant's pad fraction "
        "per face and reports mean, median, max and how many faces "
        "carry any pad. The caveat is then quoted from the measured "
        "number rather than from either assumption -- and if the pad "
        "fraction comes back at ~0, the amendment's reading stands and "
        "this record says so"
    ),
    # [CORRECTED 2026-08-24, BY THE MEASUREMENT THIS RECORD ASKED FOR --
    # the text above is preserved as written.]
    "measured_and_withdrawn": (
        "**THE PAD FRACTION CAME BACK AT ~0, SO THE AMENDMENT'S "
        "READING STANDS AND THIS RECORD SAYS SO** -- as it promised it "
        "would. Measured over the staged 1,519: mean 0.0019, median "
        "0.0000, max 0.4196, with only **10 of 1,519 faces carrying "
        "ANY pad**. MEBeauty's originals are NEAR-SQUARE in practice, "
        "so stage's non-square padding almost never fires. **The "
        "swap-story is WITHDRAWN**: the original arm's caveat set is "
        "genuinely smaller, exactly as amendment 2 claimed, and the "
        "difference from SCUT's 350x350 squares is real but NEGLIGIBLE "
        "(10 faces)"
    ),
    "whose_reading_won": (
        "the maintainer's, not the record's. The amendment said the original arm "
        "sidesteps the padding caveat; it was refined to a swap on "
        "the strength of a synthetic case. The distribution says the "
        "amendment was right"
    ),
}


#: **[BANKED 2026-08-24] THE THREE EXTRACTIONS, AND THE TWO-DIGEST
#: TRAP SPOTTED IN THEIR LOGS.**
STOP_4A_BANKED = {
    "banked": "2026-08-24, three single clean attempts",
    "the_assertion_held": (
        "``assert_run_is_the_declared_arm`` green on all three: each "
        "run's RECORDED source agreed with the config's declaration "
        "(masked_g1 / masked_g2 / masked_original). The check that "
        "could not be run on the laptop ran where the checkpoint was "
        "consumed, which is where it was put"
    ),
    "the_sets": (
        "(237, 768) each -- the full cohort, MEBeauty namespace, "
        "CLUSTER-ONLY. Declared rollups ac3b607b / 7e20eaf8 / f0a71930 "
        "(3 files; 731,283 / 731,283 / 731,321 bytes). The original "
        "set is 38 bytes larger, which is its longer metadata strings, "
        "not a different row count"
    ),
    "provenance_not_verdict": (
        "source-side PCCs 0.6905 / 0.6685 / 0.7186 logged as "
        "PROVENANCE. **The verdict is not here**: it is the probe on "
        "these sets, against the banked SCUT cells, at the threshold "
        "already declared in the probe configs"
    ),
}


#: **[FOUND 2026-08-24, IN THE EXTRACTION LOGS] TWO DIGESTS
#: OVER TWO SCOPES, AND ONLY ONE OF THEM IS DECLARABLE.**
TWO_DIGESTS_ONE_WORD = {
    "found_by": (
        "the maintainer, reading the extraction logs: the printed 'rollup' "
        "(71082846 / 86e30fb8 / fa7bf516) is NOT a prefix of the "
        "declared input hash. It was not filled from that value -- it "
        "was named"
    ),
    "what_each_covers": (
        "**payload rollup** = ``values.npy`` + ``metadata.json``, "
        "returned by ``mebeauty.save_set`` and recorded inside "
        "``MANIFEST.json`` as ``payload_rollup``. **Input rollup** = "
        "the directory AS IT STANDS, ``MANIFEST.json`` included -- "
        "what ``declare_inputs.py`` measures, what a config declares, "
        "and what guard 3 verifies"
    ),
    "why_they_cannot_coincide": (
        "**not a defect but a structural consequence**: MANIFEST.json "
        "CONTAINS the payload digest, so hashing the directory again "
        "necessarily gives a different value. A writer cannot record "
        "its own final hash inside the thing being hashed. MEASURED "
        "rather than argued: a set built on the laptop gave payload "
        "9dcb54dc over 2 files and input 2bc9c01e over 3, with the "
        "manifest carrying 9dcb54dc"
    ),
    "the_harm_it_invites": (
        "a fill taken from the log is **a wrong value that looks "
        "exactly like a right one** -- 64 hex characters, correct "
        "shape, correct provenance story. Guard 3 would refuse the run "
        "with a hash mismatch, and the mismatch would be unexplainable "
        "from either artifact, because both digests are correct for "
        "the scope each covers"
    ),
    "the_scope_measured_not_assumed": (
        "**exactly two writers have this shape**, and the "
        "discriminator was measured on the laptop, not grepped: a "
        "writer that emits MANIFEST.json AFTER hashing the payload and "
        "returns the payload digest. ``mebeauty.save_set`` and "
        "``decoder.save_scut_embeddings`` do. The ladder's own "
        "``embeddings.save`` writes NO manifest and returns no rollup "
        "key, so the five tasks built on it compute ``hash_dir`` over "
        "the finished directory themselves and already print a "
        "DECLARABLE digest -- untouched"
    ),
    "an_earlier_static_survey_was_wrong": (
        "worth recording: a first pass asked 'does this function "
        "mention MANIFEST.json' and answered NO for the MEBeauty "
        "extraction -- because the manifest is written by the CALLEE. "
        "The property is about the writer, not the caller, and only "
        "running the writer settled it. Instance N of "
        "``cleft-verification-habits``: measure, do not reason"
    ),
    "the_fix_was_already_in_the_repo": (
        "**not a new convention -- an existing one the new writer "
        "failed to follow.** ``task_masked_scut`` already emits "
        "``payload_rollup`` and ``rollup_sha256_for_configs`` side by "
        "side with a note; ``task_stage_mebeauty`` already recomputes "
        "after the manifest and logs the declarable digest. Both "
        "extraction logs now print BOTH under those names, the "
        "declarable one marked ``(DECLARE THIS)``, and both metrics "
        "files carry both keys plus a ``two_digests`` note. The SCUT "
        "line was the same trap still armed; it was fixed with the "
        "MEBeauty one and is reported, not slipped in"
    ),
    "and_the_fix_had_its_own_defect": (
        "the new ``declare = hash_dir(out_dir)`` line referenced a "
        "name neither function imported -- ``run.py`` imports "
        "``hash_dir`` per-function, and eleven other tasks do. That is "
        "a **NameError after the artifact is written**, the exact "
        "shape of ``phase13.WRITER_DIED_AFTER_SAVE``. Caught by "
        "checking scope rather than assuming it, and a repo-wide AST "
        "guard now asserts the invariant for every function"
    ),
}


#: **[BANKED 2026-08-24] THE THREE PROBES. THE PHASE'S NUMBERS.**
STOP_4B_BANKED = {
    "banked": "2026-08-24, three single clean attempts",
    "cells": {
        "mebeauty_masked@g1": {
            "pcc": -0.0693, "sd": 0.0252, "n": 5,
            "vs_scut": -0.1523, "threshold": 0.0309, "clears": True,
            "survives_single_run_95": True,
            "vs_imagenet": -0.3213, "imagenet_threshold": 0.0256,
            "own_95": [-0.0914, -0.0472],
        },
        "mebeauty_masked@g2": {
            "pcc": -0.0299, "sd": 0.0417, "n": 5,
            "vs_scut": -0.2300, "threshold": 0.0386, "clears": True,
            "survives_single_run_95": True,
            "vs_imagenet": -0.2819, "imagenet_threshold": 0.0388,
            "own_95": [-0.0665, 0.0067],
        },
        "mebeauty_original": {
            "pcc": 0.2537, "sd": 0.0222, "n": 5,
            "vs_scut": 0.0585, "threshold": 0.0313, "clears": True,
            "survives_single_run_95": False,
            "vs_imagenet": 0.0017, "imagenet_threshold": 0.0234,
            "own_95": [0.2342, 0.2732],
        },
    },
    "the_thresholds_reproduce": (
        "**every declared threshold was recomputed on the laptop from "
        "the two arms' own five-seed SDs** and matched to four places: "
        "0.0309 / 0.0386 / 0.0313 (phase3.combined_claimable_delta, "
        "PLAN 4.12.1). The deltas reproduce too. Nothing here is "
        "quoted from a run's own arithmetic without being checked "
        "against the formula the plan names"
    ),
    "all_three_clear": (
        "all three deltas clear their pre-declared thresholds. **The "
        "two losses also survive ``single_run_95``** (0.0692, 0.0863) "
        "-- the conservative bar that ignores averaging -- while the "
        "one win does NOT (0.0585 against 0.0700). By the function's "
        "own docstring that is worth saying out loud: **the two "
        "negative findings are the stronger statements of the three**"
    ),
}


#: **[READING 1, pre-registered as a delta-against-banked-cells reading;
#: applied 2026-08-24] THE MASKED ARMS ARE AT OR BELOW ZERO.**
READING_1_MASKED_ARMS_AT_ZERO = {
    "the_sentence": (
        "**MEBeauty's masked arms do not merely lose to SCUT's -- they "
        "carry no usable grade signal at all.** masked@g1 is -0.0693 "
        "and masked@g2 is -0.0299, against SCUT's 0.0830 and 0.2001"
    ),
    "why_a_loss_at_negative_pcc_is_a_stronger_statement": (
        "**a claimable loss that lands BELOW ZERO says more than a "
        "small positive one.** A cell at +0.05 is a weak predictor; a "
        "cell at -0.07 is a representation whose linear read-out is "
        "ANTI-CORRELATED with grade -- worse than uninformative, and "
        "not something a larger sample would be expected to rescue. "
        "The distance from SCUT is the same arithmetic either way; the "
        "STANDING of the arm is not"
    ),
    "one_correction_to_that_sentence": (
        "**measured before banking, and it changes half the claim.** "
        "Only g1 is CLAIMABLY below zero: its own 95% interval is "
        "[-0.0914, -0.0472] and excludes zero. **g2's is [-0.0665, "
        "+0.0067] and SPANS zero** -- that arm is indistinguishable "
        "from no signal, not claimably anti-correlated. Recorded as "
        "'at or below zero' rather than 'below zero' for that reason: "
        "one destroyed representation and one anti-correlated one are "
        "different findings, and only one of them is g2's"
    ),
    "both_clear_both_bars": (
        "both deltas clear their pre-declared thresholds against SCUT "
        "(-0.1523 vs 0.0309; -0.2300 vs 0.0386) AND survive "
        "single_run_95. Both are also claimably below ImageNet "
        "(-0.3213, -0.2819). The loss is not in doubt; only g2's "
        "position relative to zero is"
    ),
    "the_caveat_that_binds_here": (
        "**mechanism-ii: a MEBeauty LOSS is UNATTRIBUTABLE** between "
        "demographics and the 0.28x data. Taken alone these two cells "
        "cannot say whether masking or size did it. What resolves that "
        "is READING_3's within-source contrast, not these cells"
    ),
}


#: **[READING 2, same registration] THE ORIGINAL ARM REACHES IMAGENET
#: PARITY -- THE FIRST ARM IN THIS PROJECT TO DO SO.**
READING_2_ORIGINAL_REACHES_PARITY = {
    "the_sentence": (
        "**mebeauty_original is 0.2537 against ImageNet's 0.2520: a "
        "delta of +0.0017 inside a threshold of 0.0234. That is "
        "PARITY, not victory -- and it is the FIRST arm in this "
        "project to reach ImageNet rather than lose to it.**"
    ),
    "it_also_beats_its_scut_comparator_claimably": (
        "+0.0585 against scut_original@g1's 0.1952, clearing 0.0313. "
        "**Same geometry, same recipe, same cohort, same folds** -- a "
        "within-geometry comparison, asserted in code"
    ),
    "the_asymmetry_runs_in_its_favour": (
        "**per the registered mechanism-ii asymmetry, a MEBeauty WIN "
        "is STRONG evidence**, because the 0.28x data handicap ran "
        "AGAINST it. This arm reached parity with a quarter of SCUT's "
        "faces. The caveat that weakens the masked arms' losses "
        "strengthens this cell"
    ),
    "the_not_a_win_sentence_fires_correctly": (
        "the registered guard says an arm that beats its SCUT cell "
        "while losing to ImageNet is NOT a win. It fires here and it "
        "is right to: **no cell in this phase beats ImageNet.** But "
        "'not a win' must NOT be read as 'not a result' -- the guard "
        "was written to stop a SCUT-relative win being sold as an "
        "absolute one, and it does that without saying anything about "
        "what this cell IS"
    ),
    "what_this_cell_is": (
        "**parity, not victory, and the first of its kind here.** "
        "Every previously banked pretrained arm at G1 loses to "
        "ImageNet -- scut_original -0.0568 (claimable), scut_masked "
        "-0.1690. This one does not lose. The honest ceiling: it does "
        "not beat ImageNet either, and its win over SCUT does not "
        "survive single_run_95 (0.0585 against 0.0700), so it is a "
        "claim about ARMS and not about single runs"
    ),
}


#: **[READING 3, same registration] BEAUTY PRETRAINING IS NOT THE
#: PROBLEM. MASKING IS.**
READING_3_MASKING_NOT_BEAUTY = {
    "the_sentence": (
        "**read together the three cells say the damage is done by the "
        "crop, not by the pretraining task.** The SAME 1,519 faces, "
        "the SAME beauty objective, the SAME backbone and probe: "
        "whole-face gives 0.2537 (ImageNet parity), through the "
        "trapezium gives -0.0693 and -0.0299 (no signal). The "
        "difference between them is +0.3230 and +0.2836, both clearing "
        "their thresholds and both surviving single_run_95"
    ),
    "why_this_contrast_escapes_the_mechanism_ii_caveat": (
        "**this is the load-bearing point.** The mechanism-ii "
        "asymmetry makes a MEBeauty-vs-SCUT loss unattributable "
        "between demographics and size. It does NOT apply here, "
        "because this contrast is WITHIN MEBeauty: source, size, "
        "demographics, screen and staging pipeline are held constant "
        "across the three arms, and only the crop differs. Two "
        "unattributable losses become one attributable finding by "
        "being compared to their own source's whole-face arm rather "
        "than to SCUT's"
    ),
    "what_the_contrast_is_still_confounded_by": (
        "**everything else the trapezium changes**, which is not only "
        "'masking': the crop also changes effective face resolution, "
        "the padded fraction, and how much context survives. This "
        "reading attributes the damage to THE CROP AS A WHOLE, not to "
        "occlusion of the surrounding face specifically. Separating "
        "those would need a crop-geometry sweep, which this phase did "
        "not run"
    ),
    "the_boundary_of_the_claim": (
        "**'beauty pretraining is not the problem' is true of "
        "MEBEAUTY and NOT of SCUT.** scut_original still loses "
        "claimably to ImageNet (-0.0568, threshold 0.0278). So the "
        "sentence is SOURCE-DEPENDENT, and stating it unqualified "
        "would overclaim across a disagreement this phase actually "
        "measured (PHASE_6_REFRAMED)"
    ),
}


#: **[REFRAMING 2026-08-24] WHAT PHASE 6's 0.0830 NOW READS AS.
#: A REFRAMING OF A BANKED RESULT -- NOT A NEW CLAIM ABOUT IT.**
PHASE_6_REFRAMED = {
    "registered_as": (
        "**a REFRAMING, not a re-measurement.** Phase 6's cells are "
        "untouched: 0.2520 / 0.1952 / 0.0830 stand exactly as banked, "
        "no number moves, no arm re-runs. What changes is the sentence "
        "the 0.0830 is read with"
    ),
    "what_it_read_as": (
        "scut_masked@g1 = 0.0830, far below ImageNet's 0.2520: **read "
        "as beauty pretraining being useless or harmful for this "
        "task**"
    ),
    "what_it_now_reads_as": (
        "**the CROPPING did the damage, and the pretraining was "
        "carrying more than the cell showed.** MEBeauty runs the same "
        "objective through the same trapezium and lands at -0.0693; "
        "run whole-face it reaches 0.2537. A cell that looked like a "
        "verdict on the TASK now looks like a verdict on the "
        "PIPELINE"
    ),
    "why_this_is_not_a_new_claim_about_the_banked_cell": (
        "no MEBeauty measurement can change what SCUT's masked arm "
        "scored, and this phase re-ran neither side. The reframing is "
        "about which of two explanations the banked number is "
        "evidence for -- and it is offered as the better-supported "
        "reading, not as a correction of the record"
    ),
    "what_would_test_it": (
        "**the two sources DISAGREE at the original geometry, and that "
        "disagreement is the phase's genuine source-level finding.** "
        "scut_original is 0.1952 and loses claimably to ImageNet "
        "(-0.0568, threshold 0.0278); mebeauty_original is 0.2537 and "
        "reaches parity (+0.0017). Same geometry, same recipe, same "
        "cohort. If cropping were the whole story, SCUT's whole-face "
        "arm should have reached parity too, and it does not. **The "
        "test is a crop-geometry sweep within EACH source** -- "
        "whole-face, trapezium, and intermediate crops -- which would "
        "separate 'the crop damages every source' from 'these two "
        "sources differ in what survives the crop'"
    ),
    "the_caveats_bound_to_it": (
        "the four in phase15.caveats_for apply to this reframing "
        "unchanged -- mechanism-ii asymmetry, framing variance, "
        "anatomical blindness, and the padding caveat in whichever "
        "form the arm takes. **They are not weakened by the "
        "within-source contrast**: that contrast removes the SIZE "
        "confound from the masking finding, and touches none of the "
        "other three"
    ),
}


#: **[WALKED 2026-08-24] THE SEVEN EXIT CRITERIA, ONE BY ONE.**
PHASE_15_EXIT_WALK = {
    "walked": "2026-08-24, against PHASE_15_EXIT_CRITERIA as agreed",
    "1_best_is_cleft_side_transfer": (
        "**MET AS AMENDED 2026-08-24** (CRITERION_1_AMENDED): the second "
        "sense was withdrawn as never-operationalised, so the criterion "
        "reads on the sense actually run -- frozen-backbone linear-probe "
        "transfer, and nothing else. The flag that prompted it, preserved: "
        "**one ambiguity flagged rather than resolved by "
        "the record.** All three arms were measured as cleft-side "
        "transfer on the 237/cleft_v1 folds with A's recipe checked "
        "byte-identical by difference-set test, and NO source-side "
        "figure enters any verdict -- the three source-side PCCs are "
        "recorded as provenance and named as such. The ambiguity: the "
        "criterion says 'both registered senses -- masked-pretraining "
        "transfer and frozen linear-probe transfer'. A's recipe is "
        "``trainable: head``, so the ladder's comparator cells and "
        "these three arms are the SAME sense -- a frozen backbone with "
        "a fitted head. If a genuinely distinct second sense was "
        "meant (fine-tuning rather than a frozen head), **it has not "
        "been run**, and that is a maintainer decision, not the record's"
    ),
    "2_frontal_screen_banks_first": (
        "**MET.** The screen banked first at a pre-declared 0.08 "
        "centroid-offset threshold; 1,519 usable of 2,446 screened, "
        "and every later registration in this phase uses 1,519 -- the "
        "headline 2,550 appears nowhere in a count"
    ),
    "3_staging_inherits_or_departs_dated": (
        "**MET.** Levelling OFF and CLEFT_AR parity sampling "
        "inherited, parity checks re-run on the new artifact, and the "
        "one departure -- the padding caveat -- is dated in place with "
        "the original preserved and CORRECTED BY MEASUREMENT rather "
        "than by argument"
    ),
    "4_checkpoints_geometry_bound": (
        "**MET, and asserted in code rather than promised.** "
        "VARIANT_FOR_MEBEAUTY_INIT binds each init to its geometry; "
        "the writer REFUSES a crossed pair; the extraction refuses a "
        "run whose recorded source is not the declared arm; and "
        "mebeauty.verdict_delta refuses a cross-geometry comparison. "
        "The phantom-findings trap is honoured by keeping the two "
        "lattices separate -- embeddings.INITS and the 24/12 "
        "arithmetic are untouched"
    ),
    "5_scuts_side_is_the_banked_figures": (
        "**MET.** 0.2520 / 0.1952 / 0.0830 used exactly as banked, "
        "read from ladder.STAGE_D1_AT_G1 (and 0.2001 from "
        "STAGE_D_AT_G2) by test rather than retyped. Nothing on "
        "SCUT's side was re-run, because the recipe did not change"
    ),
    "6_statistic_and_threshold_declared_first": (
        "**MET.** Comparator, its SD, n, and the threshold formula "
        "were written into each probe config BEFORE the runs; the "
        "thresholds reproduce from the formula on the laptop; the "
        "ranking is DESCRIPTIVE; every reading carries the "
        "mechanism-ii sentence and the ImageNet anchor"
    ),
    "7_readings_pre_committed_no_ledger_row": (
        "**MET.** Three readings pre-registered as deltas against "
        "banked cells and applied to exactly that; **no ledger row** "
        "-- the paired bar is absent and COHORT_CANNOT_RESOLVE remains "
        "the standing expectation; the MEBeauty license citation is "
        "carried in the clone record; suite green"
    ),
    "the_phase_is_closed": (
        "**all seven are met**, six outright and criterion 1 as "
        "amended. the ruling was the second sense withdrawn rather "
        "than deferred, so nothing is left open (PHASE_15_CLOSING)"
    ),
}


#: **[PROPOSED 2026-08-24, NOT RULED] THE COMPARATIVE VERDICT FOR
#: PHASE 16.**
PHASE_16_VERDICT_PROPOSED = {
    "proposed": "2026-08-24 -- the proposal; the ruling is",
    "the_definition_it_answers": (
        "'BEST' = CLEFT-SIDE TRANSFER, per exit criterion 1. Not "
        "source-side fit, where MEBeauty's arms score 0.6905 / 0.6685 "
        "/ 0.7186 and say nothing about this question -- the ladder's "
        "0.7893 lesson"
    ),
    "stated_per_cell_not_pooled": (
        "**there is no single 'best dataset', and pooling would "
        "manufacture one.** The two sources swap places depending on "
        "the crop, which is exactly the interaction a pooled ranking "
        "would hide:"
    ),
    "the_cells": (
        "AT THE ORIGINAL GEOMETRY: **MEBeauty is better** -- 0.2537 vs "
        "SCUT's 0.1952, +0.0585, clears 0.0313. MEBeauty reaches "
        "ImageNet parity; SCUT loses to it claimably. "
        "AT G1 MASKED: **SCUT is better** -- 0.0830 vs -0.0693, "
        "+0.1523, clears 0.0309. "
        "AT G2 MASKED: **SCUT is better** -- 0.2001 vs -0.0299, "
        "+0.2300, clears 0.0386. "
        "Three cells, three claimable differences, and they do not "
        "agree on a winner"
    ),
    "the_masking_finding_attached": (
        "**and the reason they disagree is the finding.** MEBeauty "
        "wins where the face is whole and loses where it is cropped, "
        "and its own whole-face-vs-cropped contrast (+0.3230, +0.2836, "
        "both surviving single_run_95) says the crop is what costs it. "
        "So the per-cell answer is not three unrelated results: "
        "**MEBeauty is the better source and the trapezium is where it "
        "is lost**, while SCUT degrades less through the same crop"
    ),
    "what_phase_16_should_carry_forward": (
        "if a single init must be picked for cleft-side work, the "
        "candidate this phase produces is **mebeauty_original** -- the "
        "only pretrained arm at ImageNet parity. The honest statement "
        "beside it: it does not BEAT ImageNet, so ImageNet remains "
        "the arm to justify departing from, and its SCUT win does not "
        "survive single_run_95"
    ),
    "the_caveats_that_travel_with_it": (
        "all four from phase15.caveats_for, unchanged, plus the "
        "single-backbone limit -- **every cell here is ViT-B/16.** "
        "Phase 6 showed masking's answer is backbone-dependent "
        "(claimably negative in ViT, positive in AG-Net at G1), so a "
        "one-backbone verdict is a one-backbone verdict"
    ),
    "what_would_overturn_it": (
        "a crop-geometry sweep within each source, or a second "
        "backbone at the original geometry. Both are cheap relative to "
        "pretraining -- the checkpoints exist -- and either could move "
        "the per-cell picture"
    ),
}


#: **[AMENDED 2026-08-24] CRITERION 1's SECOND SENSE IS
#: WITHDRAWN AS NEVER-OPERATIONALISED.**
CRITERION_1_AMENDED = {
    "ruled": "2026-08-24 -- amend, do not defer",
    "what_was_withdrawn": (
        "**the phrase 'both registered senses -- masked-pretraining "
        "transfer and frozen linear-probe transfer' was written "
        "LOOSELY AT SCHEDULING.** All three MEBeauty arms and every "
        "comparator cell are the SAME sense: a frozen backbone with a "
        "fitted head (A's recipe is ``trainable: head``). There were "
        "never two senses to measure"
    ),
    "the_reason": (
        "a fine-tuning sense is a GENUINELY DIFFERENT EXPERIMENT, not "
        "a second reading of this one: fine-tuning at n=237 memorises, "
        "and Phase 6's fine-tuned variants went nowhere"
    ),
    "the_provenance_of_that_reason": (
        "**stated as the maintainer's recollection, and labelled as such**: "
        "no result of that kind was found a banked fine-tuning result in "
        "ladder.py or the phase records. What the record CAN measure "
        "supports the ruling independently -- **no shipped config in "
        "this project has ever set ``trainable: full``**: across all "
        "shipped configs the values are head (81), graph_layers (34), "
        "classifier (2), classifier_adabn (2). The fine-tuning sense "
        "was never operationalised ANYWHERE, so it cannot have been an "
        "exit condition for this phase"
    ),
    "what_criterion_1_now_reads_as": (
        "**MET on the sense actually run.** Cleft-side transfer means "
        "FROZEN-BACKBONE LINEAR-PROBE TRANSFER and nothing else -- "
        "237 patients, cleft_v1 folds, A's recipe checked "
        "byte-identical, five seeds. No source-side figure enters any "
        "verdict"
    ),
    "the_shape_of_this_amendment": (
        "a criterion was narrowed AFTER the numbers existed, which is "
        "the shape the project treats with suspicion. What makes it "
        "admissible: the narrowing REMOVES an unmeasured sense rather "
        "than adjusting a threshold, changes no cell and no verdict, "
        "and is dated in place with the original wording preserved in "
        "PHASE_15_EXIT_CRITERIA. Had the second sense been run and "
        "disagreed, this would not be admissible"
    ),
    # [SCOPED EXCEPTION 2026-08-30, the maintainer, at the Phase-10-annex
    # gate build] The withdrawal above is UNCHANGED for everything it
    # was about, and the supporting measurement is re-stated rather
    # than weakened: no arm, ladder cell, probe or comparator in this
    # project fine-tunes, and the census of every OTHER trainable value
    # is identical to the one recorded here (head 81, graph_layers 34,
    # classifier 2, classifier_adabn 2). What changed is that ONE
    # config now sets trainable: full -- the annex's compute gate,
    # where fine-tuning is operationalised BY RULING for fidelity to
    # the group's stated ResNet-50, and remains unregistered everywhere
    # else. A named exception, not a counter-example: the pinned test
    # fired on the fill, and this record was revisited rather than the
    # test relaxed.
    "scoped_exception_2026_08_30": (
        "phase10_annex.RULING_B_FULL_TRAINABLE"
    ),
}


#: **[CLOSED 2026-08-24] PHASE 15. THE SECOND BEAUTY DATASET.**
PHASE_15_CLOSING = {
    "closed": "2026-08-24, all seven exit criteria met, criterion 1 as amended",
    "what_the_phase_did": (
        "acquired MEBeauty, screened it, verified its landmark "
        "mapping, staged it at both cleft geometries and uncropped, "
        "pretrained three arms on it, and probed all three on the "
        "cleft cohort against Phase 6's banked cells"
    ),

    # ---- the funnel -------------------------------------------------
    "the_funnel": (
        "**2,550 published -> 2,539 on disk (eleven never shipped) -> "
        "2,445 landmark-joined (~94 images have no landmarks.csv row) "
        "-> 1,519 USABLE (62.1% at the pre-declared 0.08 "
        "centroid-offset threshold).** Every step's loss is named. "
        "1,519 is the phase's first banked number and every later "
        "registration uses it -- the headline 2,550 appears in no "
        "count. Staged splits 932 / 394 / 193. The scale asymmetry "
        "restated at the screened count: **0.28x SCUT** (1,519 vs "
        "5,500), down from the 0.46x headline"
    ),

    # ---- the one threshold that moved -------------------------------
    "the_tolerance_ruling_and_its_provenance": (
        "bounds tolerance ruled GENEROUS -- 10 points outside, 0.10 "
        "overshoot -- **JUST ABOVE the observed maxima of 9 points and "
        "7.8%**, declared as a visible config diff from the shipped "
        "strict 0 / 0.0. **The value was chosen from a measured "
        "distribution after an eye pass, and that sentence is in the "
        "record, in the config header, and in the ruling** rather "
        "than hidden in a diff. The argued basis: the distribution is "
        "a single continuous mode with no bimodality to cut at, so a "
        "tighter cut would have had no measured basis. The 0.08 pose "
        "threshold stayed untouched, per a distinction registered "
        "BEFORE that measurement existed"
    ),

    # ---- what the guards found --------------------------------------
    "eight_guards_seven_unfired": (
        "**EIGHT guards built -- five quality checks, three mapping "
        "checks -- and SEVEN NEVER FIRED on the real data.** The "
        "eighth, bounds, fired 55 times at the strict setting and "
        "rejects nothing at the ruled tolerance, so at the shipped "
        "configuration none of the eight rejects a face. An unfired "
        "guard is EVIDENCE ABOUT THE DATASET: MEBeauty's shipped "
        "landmarks are sound within the usable set. The count itself "
        "was first stated as SIX and corrected the same day"
    ),

    # ---- what was withdrawn -----------------------------------------
    "four_withdrawals_by_name": (
        "**three of a second reading impressions**: man-2785071 (the face "
        "is tilted back and bearded; this was misread the pose), "
        "girl-3956612_1920 (the flag came from a pre-fix sheet whose "
        "label overflow made panel attribution unreliable), and "
        "the record's THIRD-AXIS FRAMING claim (SCUT varies the same way, "
        "so it is shared, not a confound). **And one of the maintainer's "
        "numeric over-calls**: 'the masked arms are below zero' -- "
        "true of g1, not of g2. The withdrawals are as findable as the "
        "observations were"
    ),

    # ---- the two method lessons -------------------------------------
    "mechanism_is_not_rate": (
        "**a synthetic measures a MECHANISM, not a PREVALENCE.** "
        "Registered as a phase-level lesson and flagged for the "
        "write-up's methods chapter, then applied BACKWARDS across "
        "every synthetic-motivated caveat in the phase. It changed "
        "one: the original arm's padding caveat, whose synthetic "
        "0.1652 became a measured mean of 0.0019 over 10 of 1,519"
    ),
    "artifact_differences_are_pipeline_differences": (
        "three properties proposed as SOURCE differences between "
        "MEBeauty and SCUT turned out to be SHARED, because both go "
        "through the same construction. A property of the pipeline "
        "cannot discriminate between two datasets fed through it -- "
        "the shape behind the padding correction and the withdrawn "
        "framing claim alike"
    ),
    "the_two_digest_trap": (
        "the maintainer found the extraction logs printing a 'rollup' that "
        "is not the declarable one. **Payload rollup** covers "
        "values.npy + metadata.json and is what MANIFEST.json records; "
        "**rollup_sha256_for_configs** covers the directory with "
        "MANIFEST.json in it. They cannot coincide -- the manifest "
        "contains the payload digest -- so a fill from the log is a "
        "wrong value that looks exactly like a right one. Measured, "
        "not argued; found in exactly two writers by running them; "
        "both log lines and both metrics files now name which is "
        "which, following an idiom the repo already had"
    ),

    # ---- the numbers ------------------------------------------------
    "the_three_probe_verdicts": (
        "**masked@g1 = -0.0693, interval [-0.0914, -0.0472]** -- "
        "claimably ANTI-CORRELATED with grade; vs scut_masked@g1 "
        "-0.1523, clears 0.0309. "
        "**masked@g2 = -0.0299, interval [-0.0665, +0.0067]** -- SPANS "
        "ZERO, indistinguishable from no signal; vs scut_masked@g2 "
        "-0.2300, clears 0.0386. Two findings, not one. "
        "**original = +0.2537, interval [+0.2342, +0.2732]** -- vs "
        "scut_original@g1 +0.0585, clears 0.0313; vs ImageNet +0.0017 "
        "inside 0.0234, which is PARITY, the first arm in this project "
        "to reach ImageNet rather than lose to it"
    ),
    "the_single_run_comparison": (
        "**both losses survive single_run_95 (0.0692, 0.0863); the win "
        "does not (0.0585 against 0.0700).** The negative findings are "
        "the phase's MOST ROBUST results, and the original-arm result "
        "must be quoted with that beside it -- it is a claim about "
        "ARMS, not about single runs"
    ),
    "the_within_source_contrast": (
        "original vs its own masked arms: **+0.3230 and +0.2836, both "
        "clearing their thresholds and both surviving single_run_95.** "
        "This contrast ESCAPES the mechanism-ii asymmetry, because "
        "source, size, demographics and pipeline are held constant and "
        "only the crop varies -- which is what turns two "
        "unattributable losses into one attributable finding. Bounded: "
        "it attributes damage to THE CROP AS A WHOLE, and 'beauty "
        "pretraining is not the problem' is true of MEBeauty and NOT "
        "of SCUT, whose original arm still loses claimably to ImageNet "
        "(-0.0568, threshold 0.0278)"
    ),
    "the_reframing": (
        "Phase 6's scut_masked 0.0830 read as beauty pretraining being "
        "useless; it now reads as the CROPPING doing the damage. "
        "**Registered as a reframing of a banked result, not a new "
        "claim about it** -- no cell moves, neither side re-run. Its "
        "test is named: a crop-geometry sweep within EACH source, "
        "because the two sources DISAGREE at the original geometry and "
        "that disagreement is the phase's genuine source-level finding"
    ),

    # ---- the verdict ------------------------------------------------
    "the_per_cell_verdict_for_phase_16": (
        "**there is no single best dataset, and pooling would "
        "manufacture one.** ORIGINAL geometry: MEBeauty (0.2537 vs "
        "0.1952). G1 MASKED: SCUT (0.0830 vs -0.0693). G2 MASKED: SCUT "
        "(0.2001 vs -0.0299). Three claimable differences that do not "
        "agree on a winner -- and the disagreement is the finding: "
        "MEBeauty is the better source and the trapezium is where it "
        "is lost. Candidate init carried forward: **mebeauty_original**, "
        "with the honest statement that it does NOT beat ImageNet, so "
        "ImageNet remains the arm to justify departing from. "
        "ONE-BACKBONE BOUND: every cell here is ViT-B/16, and masking's "
        "answer is backbone-dependent"
    ),
    "what_transfer_means_here": (
        "**stated plainly, per the criterion-1 amendment: transfer in "
        "this phase means FROZEN-BACKBONE LINEAR-PROBE TRANSFER and "
        "nothing else.** No arm was fine-tuned; no shipped config in "
        "this project ever has been"
    ),

    # ---- what survives ----------------------------------------------
    "carried_forward_by_name": (
        "``mebeauty_original`` as the candidate init, with its three "
        "checkpoints and three CLUSTER-ONLY embedding sets; "
        "``phase15.caveats_for`` -- the four caveats travel with any "
        "number quoted from this phase; ``MECHANISM_IS_NOT_RATE`` and "
        "``ARTIFACT_DIFFERENCES_ARE_PIPELINE_DIFFERENCES`` for the "
        "write-up's methods chapter; ``PHASE_6_REFRAMED`` with the "
        "crop-geometry sweep as its named test; the two-digest "
        "labelling now in both extraction tasks; and the AST guard "
        "that asserts every run.py function imports the names it uses"
    ),
    "the_tally": (
        "**three instrument defects against one data defect** in this "
        "phase: the record's own machinery failed more often than "
        "MEBeauty did"
    ),
}


#: **[DECIDED 2026-08-24] THE FOURTH SEQUENCE AMENDMENT.**
#:
#: Same pattern as the three before it: the previous mapping stays
#: visible, nothing is silently renumbered, and this record is the
#: pointer's source.
PHASE_SEQUENCE_RENUMBERED_4 = {
    "decided": "2026-08-24 -- the fourth sequence amendment",
    "first_amendment": "phase11.PHASE_SEQUENCE_RENUMBERED (view ablation to 12)",
    "second_amendment": "phase12.PHASE_SEQUENCE_RENUMBERED_2 (LDL to 14, "
                        "second beauty dataset to 15, TSTR to 16)",
    "third_amendment": "PHASE_SEQUENCE_RENUMBERED_3 (metric-space ablation "
                       "to 16, TSTR to 17, write-up to 18)",
    # [2026-08-31] A FIFTH amendment APPENDS Phase 20 (the permutation
    # control). This record's "19: write-up" STANDS -- the fifth moves
    # nothing, which is why it is EXTENDED_5 rather than RENUMBERED_5.
    # It also records that the write-up is no longer the highest number
    # and that this is deliberate: the write-up runs last BY RULE, not
    # by numeric position, so it need not move again.
    "fifth_amendment": "phase20.PHASE_SEQUENCE_EXTENDED_5",
    # [2026-08-31] A SIXTH amendment APPENDS Phase 21 (the ensemble probe
    # and error-consistency diagnosis), renumbering nothing. It is the
    # first written UNDER the write-up-runs-last rule rather than
    # establishing it -- and the test of that rule: a second appended
    # phase would have forced a second renumber of the write-up under
    # the old assumption, and forces none under this one.
    "sixth_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_6",
    # [2026-09-01] A SEVENTH amendment schedules 22 (ranking and
    # pairwise losses), 23 (the statistical instruments), 24 (all five
    # raters) and 25 (foundation-model features) as
    # SCHEDULED-NOT-REGISTERED. It renumbers nothing, and leaves the
    # write-up's number OPEN rather than resolving it
    # (phase21.WRITE_UP_NUMBER_OPEN).
    "seventh_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_7",
    "was": {
        "16": "the METRIC-SPACE ABLATION (working title)",
        "17": "TSTR on the best from 15",
        "18": "write-up",
    },
    "becomes": {
        "16": "the ANCHOR LOOP -- the 25-image design, PROMOTED from "
              "noted_5_the_anchor_loop to a phase in its own right, "
              "with ANCHOR_LOOP_REGISTERED standing unchanged as its "
              "registration",
        "17": "TSTR on the best from 15 (unchanged in place and content)",
        "18": "the METRIC-SPACE ABLATION (formerly 16; "
              "PHASE_16_SCHEDULED is its scheduling record, its keys "
              "named for the number it held when written)",
        "19": "write-up",
    },
    "status_changes": {
        "anchor_loop": "noted item under PHASE_16_SCHEDULED -> PHASE 16",
        "tstr": "Phase 17 -> Phase 17 (unchanged)",
        "metric_space_ablation": "Phase 16 -> Phase 18",
        "write_up": "Phase 18 -> Phase 19",
    },

    # ---- the dependency, stated rather than left to be discovered ----
    "the_apparent_inversion": (
        "**Phase 16 is a classification-shaped method and Phase 18 "
        "decides how classification metrics are used on this cohort, so "
        "the order LOOKS inverted. It is not.** Phase 16 inherits pass "
        "zero's ALREADY-REGISTERED evaluation "
        "(phase9.PROTOTYPE_CLASSIFIER_REGISTERED: PCC and Spearman "
        "against the panel mean, 3-class accuracy against class3 at the "
        "fixed 2.5/3.5 thresholds, chance 0.333 and majority 0.502 "
        "printed beside). **It chooses no metrics fresh and therefore "
        "does not wait on 18** -- the loop is scored exactly as its "
        "own pass zero was, which is also what makes the two passes "
        "comparable"
    ),
    "the_compensating_gain": (
        "18 arriving LAST lets it examine criterion behaviour across "
        "ALL results -- 16 and 17 included -- rather than a snapshot "
        "that goes stale the moment the next phase reports. A "
        "metric-space ablation run before the anchor loop would have "
        "had to be re-run after it or silently exclude it"
    ),

    "nothing_silently_renumbered": (
        "every existing record that says 'Phase 16' meaning the "
        "metric-space ablation or TSTR stays AS WRITTEN, with dated "
        "pointers beside the ones a reader will hit -- "
        "PHASE_SEQUENCE_RENUMBERED_3, PHASE_16_SCHEDULED, "
        "ANCHOR_LOOP_REGISTERED, phase12, phase14, and "
        "classification.THREE_NOT_FIVE all name old numbers and all "
        "now carry one. The previous mapping stays visible; this "
        "record is the pointer's source"
    ),
    "eighth_amendment": (
        "phase25.PHASE_SEQUENCE_EXTENDED_8, 2026-09-05. Calibration "
        "ablation as 26, anchor set as a training set as 27, fine "
        "tuning as 28 with small backbones as an arm inside it. "
        "Renumbered nothing"
    ),
}


#: **[DECIDED 2026-08-24] THE THIRD SEQUENCE AMENDMENT.**
#:
#: Same pattern as the two before it: the previous mapping stays
#: visible, nothing is silently renumbered, and this record is the
#: pointer's source.
PHASE_SEQUENCE_RENUMBERED_3 = {
    "decided": "2026-08-24 -- the third sequence amendment",
    # [2026-08-24, later the same day] A FOURTH amendment promotes the
    # anchor loop to 16, moving the metric-space ablation to 18 and the
    # write-up to 19; TSTR stays at 17 (PHASE_SEQUENCE_RENUMBERED_4).
    # This record's "16: metric-space ablation" is preserved as what it
    # was.
    "fourth_amendment": "PHASE_SEQUENCE_RENUMBERED_4",
    # [2026-08-31] And a FIFTH appends Phase 20, the permutation
    # control, renumbering nothing (phase20.PHASE_SEQUENCE_EXTENDED_5).
    "fifth_amendment": "phase20.PHASE_SEQUENCE_EXTENDED_5",
    # [2026-08-31] A SIXTH amendment APPENDS Phase 21 (the ensemble probe
    # and error-consistency diagnosis), renumbering nothing. It is the
    # first written UNDER the write-up-runs-last rule rather than
    # establishing it -- and the test of that rule: a second appended
    # phase would have forced a second renumber of the write-up under
    # the old assumption, and forces none under this one.
    "sixth_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_6",
    # [2026-09-01] A SEVENTH amendment schedules 22 (ranking and
    # pairwise losses), 23 (the statistical instruments), 24 (all five
    # raters) and 25 (foundation-model features) as
    # SCHEDULED-NOT-REGISTERED. It renumbers nothing, and leaves the
    # write-up's number OPEN rather than resolving it
    # (phase21.WRITE_UP_NUMBER_OPEN).
    "seventh_amendment": "phase21.PHASE_SEQUENCE_EXTENDED_7",
    "first_amendment": "phase11.PHASE_SEQUENCE_RENUMBERED (view ablation to 12)",
    "second_amendment": "phase12.PHASE_SEQUENCE_RENUMBERED_2 (LDL to 14, "
                        "second beauty dataset to 15, TSTR to 16)",
    "was": {"16": "TSTR on the best from 15", "17": "write-up"},
    "becomes": {
        "16": "the METRIC-SPACE ABLATION (working title) -- what do "
              "classification metrics measure on this cohort that "
              "correlation does not, and where do the two disagree?",
        # [2026-08-29] "Rosero-design" is amended wording
        # (synthesis.PARKED["arm_amended_2026_08_29"]): Rosero FAMILY,
        # not Rosero's design. Preserved as written.
        "17": "TSTR on the best from 15 (unchanged in content; the "
              "Rosero-design arm, unparked from PLAN Part 6)",
        "18": "write-up",
    },
    "status_changes": {
        "metric_space_ablation": "not on the sequence -> SCHEDULED (Phase 16)",
        "tstr": "Phase 16 -> Phase 17",
        "write_up": "Phase 17 -> Phase 18",
    },
    "nothing_silently_renumbered": (
        "every existing record that says 'Phase 16' meaning TSTR stays "
        "AS WRITTEN, with dated pointers beside the ones a reader will "
        "hit -- phase12.PHASE_SEQUENCE_RENUMBERED_2 and "
        "phase14.PHASE_14_CONCEDED_COVERED both name Phase 16 in its "
        "old sense and both now carry one. The previous mapping stays "
        "visible; this record is the pointer's source"
    ),
    "what_15_hands_to_17": (
        "unchanged by the insertion: Phase 15 closed with a per-cell "
        "comparative verdict and ``mebeauty_original`` carried forward "
        "as the candidate init, which is what TSTR consumes. Phase 16 "
        "does not touch that hand-off -- it re-reads arms that already "
        "exist"
    ),
    "eighth_amendment": (
        "phase25.PHASE_SEQUENCE_EXTENDED_8, 2026-09-05. Calibration "
        "ablation as 26, anchor set as a training set as 27, fine "
        "tuning as 28 with small backbones as an arm inside it. "
        "Renumbered nothing"
    ),
}


#: **[REGISTERED 2026-08-24, BUILD NOTHING TONIGHT] THE ANCHOR LOOP --
#: the maintainer's design, extending the measured anchor classifier.**
#:
#: Registration only. It runs in Phase 16, under whatever scope that
#: phase's restate proposes; this record fixes the design and its
#: committed readings so neither is written after the numbers.
ANCHOR_LOOP_REGISTERED = {
    "registered": "2026-08-24, design the maintainer's, build nothing tonight",

    # ---- the verified premise: pass zero is already measured ---------
    "pass_zero_is_measured": (
        "**the un-looped version of this design already ran and is "
        "banked**: phase9.PROTOTYPE_CLASSIFIER_OBSERVED / ledger "
        "p9-anchor-classifier-convergence. All eight cells within 0.03 "
        "of chance (3-class accuracy 0.333-0.363 against chance 0.333, "
        "majority 0.502; the banked wording is precise -- two cosine "
        "cells sit marginally ABOVE chance, so 'at chance' is the "
        "rounded form). The anchors' own leave-one-out self-consistency "
        "FAILED first: 4/25 euclidean, 3/25 cosine, against ~4.75/25 "
        "expected by chance from the 3/7/6/6/3 grade counts. **The "
        "extension's value therefore lives entirely in what pass zero "
        "lacks** -- a learned metric and a per-patient explanation -- "
        "and nothing in this registration re-litigates the measured "
        "null"
    ),

    # ---- (1) the anchors ---------------------------------------------
    "anchors": (
        "the 25 Deall images, grade spread 3/7/6/6/3, **each anchor its "
        "own prototype** -- sub-grade level. A patient may be nearest "
        "an anchor of one grade while sitting nearer another grade's "
        "anchor than that grade's remaining anchors; assignment is "
        "PER-ANCHOR, and aggregation to grade happens only at readout. "
        "**Anchors are FIXED for all time**: surgeon-chosen, and "
        "therefore immune to the medoid instability the record "
        "measured (bootstrap persistence 0.394/0.427/0.414 -- a "
        "data-derived prototype changes identity three bootstraps in "
        "five; a surgeon-chosen one cannot)"
    ),

    # ---- (2) the loop ------------------------------------------------
    "the_loop": (
        "a LEARNED METRIC CORRECTION: a linear transform of the frozen "
        "768-d space is the default (alternatives NOTED, not "
        "committed), trained to pull each training-fold patient toward "
        "anchors of their own grade. Trained on TRAINING FOLDS ONLY, "
        "evaluated out-of-fold on cleft_v1, standard five seeds"
    ),
    "the_forbidden_version_named": (
        "**iterating to correctness on all 237 without folds is "
        "MEMORISATION and is forbidden.** A loop that runs until every "
        "patient lands on the right anchor has learned the roster, not "
        "the grade; its accuracy is 1.0 by construction and means "
        "nothing. The criterion's shape applies UNCHANGED: out-of-fold "
        "numbers, five seeds, both arms' spreads in any threshold"
    ),

    # ---- (3) what the probe cannot produce ----------------------------
    "the_explainability_deliverable": (
        "**per-patient mismatch records**: assigned anchor vs the "
        "anchors of the ID-card grade, WITH DISTANCES -- case-based "
        "explanations ('this patient sits nearest anchor 14, a grade-4 "
        "reference; their ID card says grade 2; the nearest grade-2 "
        "anchor is 2.3x further'). **This is the contribution the "
        "linear probe cannot produce and the reason the extension "
        "exists**: the probe outputs a scalar with no case to point "
        "to, and the registration says so in advance so the "
        "deliverable is not retrofitted as the goal after a null"
    ),

    # ---- (4) the prohibition, cited ------------------------------------
    "prohibition_no_decoded_prototypes": (
        "**no decoding of averaged or refined embeddings into 'ideal "
        "president' faces.** Cited, not asserted: "
        "phase13.SCUT_NORMAL_PRIOR (the decoder pulls "
        "structurally-unusual features toward its SCUT-normal average) "
        "and the asymmetry probe (asymmetry ENCODED at +0.4856, "
        "RENDERED at +0.0082 -- the decoder discards per-patient "
        "geometry). A decoded prototype would therefore be an "
        "explanation that LIES: it would show the prior, not the "
        "patient. Presidents remain REAL images; any group visual is "
        "its nearest real members"
    ),

    # ---- (5) readings committed both ways ------------------------------
    "reading_if_null": (
        "**the expectation, from four convergent nulls** (t-SNE 0.409, "
        "medoid 0.257, persistence ~0.4, anchor classifier at chance) "
        "and the probe's 0.2520: the loop MATCHES OR LANDS BELOW the "
        "probe, with COHORT_CANNOT_RESOLVE predicting the difference "
        "unclaimable. That outcome is the FIFTH convergent measurement, "
        "and the deliverable still stands: parity-with-explanations is "
        "the registered success, not victory"
    ),
    "reading_if_above": (
        "if it lands CLAIMABLY ABOVE the probe -- both arms' five-seed "
        "spreads in the threshold, the standing machinery -- **that is "
        "the surprise that matters more**: the frozen space carries "
        "grade structure that a free linear direction missed and a "
        "metric shaped by anchors found. The same both-ways commitment "
        "the prototype classifier carried, for the same reason"
    ),
    "success_is_defined_before_the_run": (
        "**parity-with-explanations, not victory.** An anchor loop at "
        "the probe's level with per-patient mismatch records is a "
        "success; an anchor loop 0.01 above it with no explanations "
        "would not be. Written down tonight so tomorrow cannot redefine "
        "it"
    ),

    # ---- (6) caveats bound ---------------------------------------------
    "caveats_bound": (
        "(i) anchor grades are TRUSTED SINGLE GRADES from the survey "
        "lineage, not verified-unanimous -- pass zero's banked caveat, "
        "inherited verbatim; supervision can confirm (it is question 5's "
        "neighbour). (ii) one backbone, frozen space -- nothing here "
        "generalises past ViT-B/16 arm A. (iii) the ImageNet anchor "
        "rides in every reading, as everywhere"
    ),

    # [2026-08-29] The restate rulings, the pre-run self-consistency
    # readings, the primary contrast, and the measured compute shape
    # live in the phase's OWN module -- the registration material
    # outgrew this closed phase's file (phase16.ANCHOR_LOOP_RULINGS).
    "restate_rulings": "phase16.ANCHOR_LOOP_RULINGS",
    # [2026-08-24, the fourth amendment] PROMOTED: the anchor loop IS
    # Phase 16 now, a phase in its own right, this record standing
    # unchanged as its registration (PHASE_SEQUENCE_RENUMBERED_4). The
    # field below is preserved as written; only its "noted item"
    # framing is superseded -- the number happens to be the same.
    "promoted": "PHASE_SEQUENCE_RENUMBERED_4",
    "where_it_runs": (
        "Phase 16, as a noted item under PHASE_16_SCHEDULED -- "
        "REGISTERED-NOT-BUILT tonight, so the true sentence tomorrow "
        "is: 'pass zero is measured at chance; the extension is "
        "registered with its readings committed; it runs in Phase 16'"
    ),
}


#: **[SCHEDULED 2026-08-24] PHASE 16 -- THE METRIC-SPACE ABLATION.**
#:
#: Scheduled, not opened. Its SCOPE is proposed at its restate, in the
#: phase's own words, against whatever the record says then.
PHASE_16_SCHEDULED = {
    "scheduled": "2026-08-24 (PHASE_SEQUENCE_RENUMBERED_3)",
    # [2026-08-24, the fourth amendment] This phase is now PHASE 18
    # (PHASE_SEQUENCE_RENUMBERED_4); the record and its keys keep the
    # number they were written under, per the standing pattern.
    "renumbered_to_18": "PHASE_SEQUENCE_RENUMBERED_4",
    # [2026-08-30] The scope this record deferred to the restate now
    # exists: phase18.PHASE_18_RULINGS (arm set, contrasts, compute),
    # DELIVERABLES_REGISTERED (five, readings pre-committed), and the
    # reckoning this record demanded (phase18.PHASE_18_RECKONING).
    "rulings_2026_08_30": "phase18.PHASE_18_RULINGS",
    "working_title": "metric-space ablation",
    "the_registered_question": (
        "**what do classification metrics measure on this cohort that "
        "correlation does not, and where do the two disagree?**"
    ),
    "scope_is_not_set_here": (
        "**deliberately.** The scope is proposed at the phase's "
        "RESTATE, not at scheduling -- the same discipline that caught "
        "criterion 1's loose wording in Phase 15, where two 'registered "
        "senses' written at scheduling turned out to be one sense "
        "written twice (CRITERION_1_AMENDED). A scope written before "
        "the phase reads the record is a scope written from memory"
    ),

    # ---- the obvious content, NOTED not committed --------------------
    "noted_content_not_a_scope": (
        "four things a reader would expect the phase to consider, "
        "recorded so they are not lost between now and the restate -- "
        "**none of them committed, none of them exhaustive**"
    ),
    "noted_1_the_full_ladder_under_both_families": (
        "the arm ladder read under both metric families rather than one "
        "arm under two. A disagreement visible in a single cell is an "
        "anecdote; a disagreement that changes the RANKING is a finding"
    ),
    "noted_2_the_criterions_two_conditions": (
        "whether the criterion's two conditions behave differently on "
        "F1 than on PCC. **This is not idle**: Phase 11 already found "
        "one metric-versus-metric trade-off, so the question has a "
        "prior instance rather than a hypothetical one"
    ),
    "noted_3_the_qwk_question": (
        "QWK, which the record SET ASIDE rather than answered -- and "
        "``eval.metrics.qwk_3cat`` already exists, frozen, unused by "
        "any banked verdict. The phase inherits both the function and "
        "the decision not to have used it"
    ),
    "noted_4_what_macro_averaging_can_support": (
        "a REGISTERED POSITION on what macro-averaging can support at "
        "these class counts -- the cohort's median-grade distribution "
        "is 5/89/110/30/3 over 237, and a 3-patient class carries the "
        "same weight as a 110-patient one. The position is registered "
        "BEFORE the numbers, not read off them"
    ),

    "noted_5_the_anchor_loop": (
        "**[ADDED 2026-08-24, REGISTERED-NOT-BUILT]** the anchor-loop "
        "extension of the measured anchor classifier -- the maintainer's "
        "design, registered in full with both readings committed "
        "(ANCHOR_LOOP_REGISTERED). Unlike the other noted items this "
        "one is REGISTERED, not merely noted: its design, its "
        "forbidden variant, its prohibition, its success definition "
        "and its caveats are fixed tonight; only the BUILD belongs to "
        "Phase 16"
    ),

    # ---- the discipline it must open with ----------------------------
    "must_open_with_a_reckoning": (
        "**the same reckoning discipline Phase 14 used** "
        "(phase14.PHASE_14_CONCEDED_COVERED): position against the "
        "PRE-REGISTERED decision -- 'PCC is primary, QWK is not' -- and "
        "say what NEW thing this phase measures that the earlier "
        "decision did not cover. Phase 14 met that bar by CONCEDING "
        "coverage, which is a legitimate outcome and the one to expect "
        "if no new measurement is on offer. Scheduling is not "
        "reopening; opening without meeting the prior decision would be"
    ),
    "what_it_already_has_in_hand": (
        "``classification.CLASSIFICATION_METRICS_SECONDARY`` -- the "
        "post-closing addendum registered the same day, which computes "
        "precision/recall/F1 on ONE arm's existing predictions. That "
        "addendum is a measurement, not a phase: it answers 'what are "
        "the numbers' for a single cell and explicitly declines to "
        "compare arms. Phase 16 is where comparing them would be "
        "argued for"
    ),
    "the_bound_it_inherits": (
        "the addendum's three caveats apply to anything Phase 16 builds "
        "on it: the class-support vector travels with every quote, "
        "macro-averaging weights the smallest class equally with the "
        "largest, and the discretisation is a REPORTING step applied "
        "after training on a continuous target -- not the training "
        "objective. An arm trained to maximise F1 would be a different "
        "arm, and Phase 16 must say whether it is measuring the metric "
        "or the objective"
    ),
}


def caveats_for(source: str) -> dict:
    """The caveats bound to a number from this arm -- FOUR for the
    masked arms, THREE for the original (phase15.STOP_3_AMENDED).

    Returned as data rather than prose so a reading cannot quote a
    number without them: the caller writes them into its own metrics.
    """
    caveats = {
        "mechanism_ii_asymmetry": (
            "MEBeauty is 0.28x SCUT's data (1,519 vs 5,500): a "
            "MEBeauty WIN is STRONG evidence, a LOSS is UNATTRIBUTABLE "
            "between demographics and size"
        ),
        "framing_variance": (
            "in-the-wild framing varies within BOTH beauty sources -- "
            "shared with SCUT, and distinct from the "
            "protocol-photographed cleft cohort "
            "(FRAMING_VARIES_LIMITATION)"
        ),
        "anatomical_blindness": (
            "landmarks can be valid on every measurable axis and still "
            "sit slightly wrong (man-1868320's class); the rate is "
            "UNQUANTIFIED and the staged-sheet gate passed "
            "(ANATOMICALLY_BLIND_LIMITATION)"
        ),
    }
    # [CORRECTED 2026-08-24] Both padding clauses were wrong: the
    # masked one rested on a null read (0/1,519 was never measured),
    # and the original one on a synthetic that measured a mechanism
    # rather than a rate. Both now quote MEASURED distributions.
    if source != "masked_original":
        caveats["padding_matched_by_design"] = (
            "the masked crops average ~0.2565 pad fraction against the "
            "cleft cohort's own 0.2534 (parity gap +0.0031) -- they "
            "are padded BY CONSTRUCTION (cut at a cleft aspect ratio, "
            "staged to square) and MATCHED to the cohort on purpose. "
            "SCUT's masked arm shares the same construction, so this "
            "is NOT a differentiator between the two sources "
            "(PADDING_IS_AN_ARTIFACT_PROPERTY, corrected)"
        )
    else:
        caveats["padding_negligible_measured"] = (
            "the original arm carries essentially no pad: mean 0.0019, "
            "median 0.0000, with 10 of 1,519 faces carrying any at all "
            "-- MEBeauty's originals are near-square in practice, so "
            "the difference from SCUT's 350x350 squares is real but "
            "negligible (ORIGINAL_ARM_PADS_TOO, corrected)"
        )
    return caveats


#: **[PATTERN 2026-08-24] DIFFERENCES BETWEEN OUR ARTIFACTS ARE
#: USUALLY DIFFERENCES IN OUR PIPELINE, NOT IN THE SOURCES.**
ARTIFACT_DIFFERENCES_ARE_PIPELINE_DIFFERENCES = {
    "recorded": (
        "2026-08-24, on the third instance -- a pattern, not a "
        "coincidence"
    ),
    "the_three": (
        "(1) FRAMING VARIANCE: proposed by the record as a third axis of "
        "unattributability between MEBeauty and SCUT; SCUT varies the "
        "same way, and the real contrast is between both beauty "
        "sources and the PROTOCOL-PHOTOGRAPHED cohort "
        "(FRAMING_VARIES_LIMITATION). (2) PADDING: registered as 'every "
        "MEBeauty crop is padded, unlike SCUT'; both masked arms are "
        "padded by the SAME build_one construction, and the fraction "
        "is matched to the cohort ON PURPOSE "
        "(PADDING_IS_AN_ARTIFACT_PROPERTY, corrected). (3) THE ASPECT "
        "RATIO ITSELF: MEBeauty's crops are cut at cleft-sampled "
        "ratios because CLEFT_AR parity is INHERITED -- the AR "
        "distribution is a property of our sampler, not of MEBeauty"
    ),
    "the_pattern": (
        "**DIFFERENCES BETWEEN OUR ARTIFACTS ARE USUALLY DIFFERENCES "
        "IN OUR PIPELINE, NOT IN THE SOURCES.** Three times this "
        "phase, a property of the staging was read as a property of "
        "the dataset -- and each time it was shared, because the "
        "pipeline is shared BY DESIGN. The parity machinery exists "
        "precisely to make the two artifacts alike, so its successes "
        "keep arriving disguised as MEBeauty's peculiarities"
    ),
    "the_check_it_implies": (
        "before recording any property as a source difference, ask "
        "whether SCUT's arm goes through the same code. If it does, "
        "the property is the PIPELINE's and the caveat is about the "
        "comparison's construction rather than about either dataset"
    ),
    "what_survives_as_a_real_difference": (
        "the ones that are NOT pipeline artifacts: DEMOGRAPHICS "
        "(mechanism i, the whole reason MEBeauty was chosen) and SIZE "
        "(mechanism ii, 0.28x -- dead and inverted). Those are "
        "properties of the sources themselves, and they are the two "
        "the phase's readings actually turn on"
    ),
}


#: **[LESSON 2026-08-24] MECHANISM IS NOT RATE** -- the family this
#: belongs to, recorded because the same shape recurs.
MECHANISM_IS_NOT_RATE = {
    "recorded": "2026-08-24, from the original-arm padding refinement",
    "what_happened": (
        "a SYNTHETIC case (a 500x600 source) measured a mechanism "
        "CORRECTLY -- the frozen ``stage`` does pad non-square sources, "
        "0.1652 on that input -- and a PREVALENCE was read off it: "
        "that MEBeauty's originals would therefore be padded. The "
        "distribution says 10 of 1,519"
    ),
    "the_lesson": (
        "**A SYNTHETIC CASE MEASURES A MECHANISM AND TELLS YOU NOTHING "
        "ABOUT ITS PREVALENCE.** Mechanism is not rate. The synthetic "
        "was not wrong -- it was answering 'can this happen', and it "
        "was quoted as if it had answered 'how often does it'"
    ),
    "the_family": (
        "this is the third form of one habit this phase: the "
        "collapsed-cloud synthetic proved the pose screen COULD be "
        "fooled (and the real data then showed 55 tight-crop cases and "
        "no collapsed clouds at all); the foreign-frame synthetic "
        "proved the bounds check COULD fire (and had NO instance in "
        "the data); and now the padding synthetic. Each synthetic "
        "earned its place as a guard; none of them measured a rate"
    ),
    "the_discipline": (
        "when a synthetic motivates a caveat, the caveat is PROVISIONAL "
        "until the distribution is measured -- and the record that "
        "carries it should say which of the two it rests on"
    ),
    # [ACCEPTED 2026-08-24 as a PHASE-LEVEL lesson, and flagged beyond
    # the record.]
    "for_the_writeup_methods_chapter": (
        "**FLAGGED FOR THE WRITE-UP'S METHODS CHAPTER, not only the "
        "record**: three synthetics this phase measured mechanisms "
        "correctly and told us nothing about prevalence -- collapsed "
        "cloud (0 real instances), foreign frame (0), padding (10 of "
        "1,519). A methods section that describes guards without "
        "saying which ones ever fired describes an instrument nobody "
        "can calibrate"
    ),
    "applied_retroactively": (
        "the discipline applies BACKWARDS: every synthetic-motivated "
        "caveat in this phase is swept and marked mechanism-only or "
        "rate-measured (SYNTHETIC_CAVEAT_SWEEP)"
    ),
}


#: **[SWEPT 2026-08-24, RETROACTIVELY] EVERY SYNTHETIC-MOTIVATED GUARD
#: IN THIS PHASE, MARKED.** MECHANISM-ONLY means the synthetic proved
#: the failure CAN happen and the data never showed it; RATE-MEASURED
#: means the distribution was measured on the real set.
SYNTHETIC_CAVEAT_SWEEP = {
    "swept": "2026-08-24, applying MECHANISM_IS_NOT_RATE backwards",
    "quality_screen_degenerate": (
        "MECHANISM-ONLY. Synthetic: an all-zero row is refused. Real "
        "instances: **0 of 1,519** -- no MEBeauty row was empty, "
        "non-finite or collapsed"
    ),
    "quality_screen_bounds": (
        "**BOTH, and the split matters.** The check FIRED on the real "
        "data (55 of 1,519) -- RATE-MEASURED. But the family it was "
        "BUILT for, the foreign coordinate frame, has **0 real "
        "instances**: every one of the 55 was a tight crop, and the "
        "synthetic's 65-points-outside/169% signature never appeared. "
        "So the guard is rate-measured; its motivating story is "
        "mechanism-only"
    ),
    "quality_screen_span": (
        "MECHANISM-ONLY. Synthetic: a collapsed cloud fails at 0.0399 "
        "while its POSE offset (0.0006) sails through the 0.08 screen "
        "-- the blindness the stop existed for. Real instances: **0**"
    ),
    "quality_screen_ordering": (
        "MECHANISM-ONLY. Synthetic: brows below the mouth are refused. "
        "Real instances: **0**"
    ),
    "quality_screen_interocular": (
        "MECHANISM-ONLY. Synthetic: fused eyes fail at 0.0252. Real "
        "instances: **0**"
    ),
    "mapping_verification": (
        "MECHANISM-ONLY for the three wrong-mapping families "
        "(eye-swap, off-midline landmark, wrong corners -- each "
        "refused synthetically), and RATE-MEASURED for the mapping "
        "itself: verified over all 1,519 with image-left fraction "
        "1.0000 and corner-widest 1.0000. The guards never fired "
        "because the mapping was right"
    ),
    "original_arm_padding": (
        "**RATE-MEASURED, and it changed the caveat.** Synthetic: "
        "0.1652 on a 500x600 source. Real: mean 0.0019, 10 of 1,519. "
        "The synthetic's mechanism was correct and its prevalence "
        "reading was wrong (ORIGINAL_ARM_PADS_TOO)"
    ),
    "masked_arm_padding": (
        "RATE-MEASURED: pad fraction ~0.2565 against the cohort's "
        "0.2534, from the artifact's own parity report -- never "
        "synthetic-motivated, and the 0/1,519 figure it briefly rested "
        "on was a null read, not a synthetic"
    ),
    "not_synthetic_motivated_at_all": (
        "two caveats are motivated by REAL observations and are marked "
        "separately: ANATOMICAL BLINDNESS (man-1868320, one confirmed "
        "instance, rate UNQUANTIFIED and declared so) and FRAMING "
        "VARIANCE (seen by eye in both sources; an eyes-fraction rate "
        "was considered and DECLINED as decision-irrelevant). Neither "
        "is provisional in the synthetic sense -- but neither carries "
        "a rate, and both say so"
    ),
    "mechanism_ii_is_neither": (
        "the 0.28x asymmetry is a COUNT RATIO (1,519 vs 5,500), "
        "measured by construction -- no synthetic, no distribution, "
        "nothing provisional about it"
    ),
    "what_the_sweep_shows": (
        "**EIGHT GUARDS were built -- five quality checks and three "
        "mapping checks -- and SEVEN NEVER FIRED on the real data.** "
        "The eighth, bounds, fired 55 times at the strict setting and "
        "rejects NOTHING at the ruled tolerance, so at the shipped "
        "configuration none of the eight rejects a face. That is not "
        "waste: an unfired guard is EVIDENCE ABOUT THE DATASET -- here, "
        "that MEBeauty's shipped landmarks are sound within the usable "
        "set -- and the sweep is what turns 'we checked' into 'we "
        "checked, and here is what was there'"
    ),
    "the_count_is_stated_because_it_was_first_stated_wrong": (
        "this field said SIX in its first draft, from counting the "
        "three mapping checks as one guard. Corrected the same hour by "
        "recounting: 5 quality + 3 mapping = 8. A sweep whose own "
        "arithmetic is loose is not a sweep"
    ),
}





#: **[DEFECT 2026-08-24, MINE] A CONFIG/TASK KEY MISMATCH REACHED THE
#: CLUSTER, AND THE SMOKE TESTS DID NOT COVER IT.**
KEY_MISMATCH_DIAGNOSED = {
    "defect": (
        "2026-08-24: p15-stage-mebeauty-v2 failed all five attempts (+4 "
        "retries) with KeyError: 'geometries', raised immediately after "
        "the incomplete-removal line and BEFORE any staging"
    ),
    "the_cause_is_mine": (
        "**A COPY-PASTE.** The patch that renamed ``geometries`` to "
        "``variants`` inserted the new line but LEFT THE OLD ONE IN the "
        "replacement text, so the task read both -- and one of them was "
        "a key the config no longer had. The generator wrote "
        "``variants``, the schema required ``variants``, and the task "
        "still asked for ``geometries``"
    ),
    "why_the_schema_could_not_catch_it": (
        "the schema validates what the config DECLARES, not what the "
        "task READS. A config can be perfectly valid and a task can "
        "still subscript a key nobody declared -- the two halves were "
        "never connected on the laptop, so the mismatch travelled"
    ),
    "the_coverage_answer": (
        "**NO TEST EXERCISED A SHIPPED CONFIG THROUGH ITS TASK'S ACTUAL "
        "KEY ACCESS.** The smoke tests checked hashes, placeholders, "
        "portability and artifact-writing modes; none of them asked "
        "whether a task's reads are satisfied by the config that feeds "
        "it. Added: "
        "``test_every_shipped_config_carries_every_key_its_task_"
        "subscripts`` walks EVERY shipped config, resolves its task, "
        "and asserts every key the task SUBSCRIPTS is present. "
        "Subscripts only -- ``.get`` is optional by construction and "
        "cannot raise"
    ),
    "the_test_was_verified_to_fail": (
        "a check that cannot fail verifies nothing, so the helper was "
        "run against the task AS IT WAS when the cluster failed: it "
        "reports 'geometries' and ignores both the ``.get`` read and "
        "the f-string key. It is AST-based rather than regex-based "
        "because the first version matched the word inside the COMMENT "
        "that explains the rename -- the word-matching trap, biting "
        "the very helper written to close a defect"
    ),
    "the_second_run_time_only_defect_this_week": (
        "**AND THAT IS THE PATTERN WORTH NAMING.** The shutil/io "
        "NameError was caught at build time by checking the module's "
        "imports; this one was not caught because nothing checked the "
        "config-to-task contract. Two run-time-only defects in a week, "
        "both in code that compiles and passes every existing test. "
        "The cluster is not a test environment, and each of these "
        "costs retries on the record"
    ),
    "the_fix": (
        "the stale read is gone; and EVERY config read in the staging "
        "task now happens up front, BEFORE anything is created or "
        "destroyed -- so a key mismatch fails before any side effect"
    ),
}


#: **[CORRECTIONS 2026-08-24, RECORDED AS SENT]** Two
#: corrections to what was sent earlier in the same exchange.
V1_SURVIVES_CORRECTION = {
    "corrected": "2026-08-24, the maintainer, correcting an earlier message",
    "a_v1_survives": (
        "**mebeauty_masked_v1 SURVIVES.** The removal targets ONLY "
        "``mebeauty_masked_v2.inprogress`` -- the task's own incomplete "
        "directory, named from its own out_version. The earlier claim "
        "that the eye-passed v1 artifact had been destroyed was WRONG "
        "and is WITHDRAWN"
    ),
    "b_ordering_is_hygiene": (
        "the validate-then-remove ordering is therefore HYGIENE, not "
        "damage control -- still worth doing (validate the config fully "
        "before deleting anything), but NO ARTIFACT WAS LOST and none "
        "was ever at risk"
    ),
    "why_it_is_recorded": (
        "a withdrawn claim about data loss is worth as much record as a "
        "confirmed one: the next reader of the .inprogress machinery "
        "should find the correction beside it, not the alarm alone. "
        "The same retraction discipline the eye impressions get"
    ),
}


#: **[BANKED 2026-08-24] THE THREE PRETRAINING ARMS.** Source-side
#: only: this is PROVENANCE, not the verdict -- SCUT test PCC turned
#: out not to predict cleft transfer at all (ladder).
STOP_3_PRETRAINING_BANKED = {
    "banked": (
        "2026-08-24, three arms, all single clean attempts"
    ),
    "figures_as_reported": (
        "masked G1 -- selected epoch 26, source-side test PCC 0.6905; "
        "masked G2 -- epoch 6, 0.6685; original -- epoch 7, 0.7186"
    ),
    "the_mapping_is_not_confirmed_here": (
        "**THE ARM-TO-RUN MAPPING CANNOT BE CONFIRMED HERE: there is no "
        "cluster access, and the run directories are the only place it "
        "is written.** The figures above are recorded AS REPORTED, not "
        "as verified. Each run's metrics.json carries its own "
        "``source`` and ``variant`` fields (task_pretrain_mebeauty "
        "writes them), so the mapping is machine-checkable -- and the "
        "EXTRACTION TASK ASSERTS IT: it reads each declared run's "
        "metrics.json and refuses if the recorded source is not the "
        "arm the config names. A swapped mapping fails there rather "
        "than propagating into a verdict"
    ),
    "observation_1_harder_on_its_own_task": (
        "MEBeauty is HARDER on its own task in every cell. Against "
        "SCUT's corresponding source-side figures: masked G1 0.6905 vs "
        "**0.7893** (-0.0988) and masked G2 0.6685 vs **0.8306** "
        "(-0.1621), both banked in ladder.MASKED_G1_ARTEFACT; "
        "original 0.7186 vs 0.8914 (-0.1728), **that comparator "
        "OPERATOR-SUPPLIED and NOT in the repo** -- a distinction kept "
        "because two of the three are checkable here and one is not. "
        "Expected for in-the-wild multi-ethnic data, consistent with "
        "the authors' own framing, and PROVENANCE rather than verdict"
    ),
    "observation_2_the_geometry_ordering_inverts": (
        "**RECORDED NOW, SOURCE-SIDE ONLY, SO IT CANNOT BE RETROFITTED "
        "AS AN EXPLANATION ONCE TRANSFER NUMBERS EXIST.** SCUT: G2 "
        "0.8306 > G1 0.7893. MEBeauty: G1 0.6905 > G2 0.6685. The "
        "ordering INVERTS between the sources. No reading is attached "
        "to it and none may be attached retrospectively; if a transfer "
        "difference later runs the same way, this record is what makes "
        "the coincidence checkable rather than persuasive"
    ),
    "observation_3_the_arms_memorised_their_fit_set": (
        "train loss 0.002-0.03 on 932 fit images: the arms MEMORISED "
        "the fit set. **The 0.28x asymmetry is visible in the training "
        "dynamics**, and the epoch selection shows it -- G2 and "
        "original peaked at epochs 6-7 and then overfit, and only G1 "
        "improved late (epoch 26). The FIXED-BUDGET / BEST-CHECKPOINT "
        "discipline working exactly as designed: a patience rule would "
        "have stopped the three arms at different places for reasons "
        "that track noise, and the comparison would have been between "
        "run lengths"
    ),
    "the_verdict_is_elsewhere": (
        "none of this is the phase's answer. The cleft-side probe "
        "decides, and the source-side numbers are recorded so the "
        "checkpoints have provenance -- the ladder's own finding is "
        "that SCUT test PCC did not predict cleft transfer"
    ),
}


#: **[RULED 2026-08-24 -- THE NAMESPACE] AND THE EXTRACTION
#: STOP PROPOSED ON THAT BASIS.**
STOP_4_PROPOSED = {
    "proposed": "2026-08-24, the ruling first; nothing built yet",
    "the_ruling": (
        "**the cleft-side extraction gets its OWN registered set "
        "namespace and its OWN count arithmetic. ``embeddings.INITS`` "
        "and Phase 6's 24/12 lattice stay FROZEN AND UNTOUCHED, and "
        "the two lattices are NEVER POOLED.** Adding MEBeauty inits to "
        "a closed phase's constant is the PHANTOM-FINDINGS SHAPE, "
        "where an undeclared artifact swap manufactures factor effects "
        "-- ladder.MASKED_G1_ARTEFACT is that shape's fourth instance, "
        "and this would have been its fifth"
    ),
    "what_the_ruling_forbids_concretely": (
        "no MEBeauty entry in embeddings.INITS or VARIANT_FOR_INIT; no "
        "change to expected_set_count's 24/12 arithmetic; no MEBeauty "
        "row in embedding_plan's lattice; and no comparison that puts "
        "a MEBeauty cell and a ladder cell in one table without saying "
        "which lattice each came from"
    ),
    "the_namespace": (
        "``mebeauty.MEBEAUTY_INITS`` = (mebeauty_masked, "
        "mebeauty_original) with its own VARIANT_FOR_MEBEAUTY_INIT -- "
        "masked is GEOMETRY-BOUND exactly as SCUT's is, so "
        "mebeauty_masked@g1 and @g2 are different variants and "
        "different checkpoints. Its own count arithmetic: 3 sets (2 "
        "geometries x masked, plus original), from 3 checkpoints, over "
        "1 backbone -- stated in the MEBeauty namespace and never "
        "added to the ladder's"
    ),
    "the_mapping_gets_asserted_not_trusted": (
        "**the extraction task READS each declared pretraining run's "
        "metrics.json and REFUSES if its recorded ``source`` is not "
        "the arm the config names.** this machine cannot confirm the "
        "arm-to-run mapping (no cluster access, "
        "STOP_3_PRETRAINING_BANKED), so the confirmation is moved into "
        "the code that consumes it: a swapped mapping fails at "
        "extraction rather than propagating into a verdict. This is "
        "the checkpoint-identity lesson applied at the one place it "
        "can still be caught"
    ),
    "the_arms": (
        "three MEBeauty inits x the 237-patient cleft cohort: "
        "mebeauty_masked@g1, mebeauty_masked@g2, mebeauty_original. "
        "Each extracts through its own geometry-bound checkpoint "
        "(VARIANT_FOR_INIT discipline, MEBeauty's own table), writes "
        "its own CLUSTER-ONLY set, and feeds one probe"
    ),
    "the_probes": (
        "A's recipe BYTE-IDENTICAL -- five seeds, cleft_v1 folds, the "
        "same head and knobs, derived from the shipped p7 config so "
        "identity is checked rather than claimed"
    ),
    "the_verdict": (
        "within-geometry ASSERTED (mebeauty.verdict_delta refuses a "
        "crossed pair), the ImageNet anchor in every reading, the four "
        "caveats bound via phase15.caveats_for, and the verdict "
        "statistic -- delta against the same-named banked cell, "
        "thresholded from both arms' own five-seed SDs -- DECLARED IN "
        "THE CONFIG before any number exists"
    ),
    "compute_shape": (
        "3 extraction jobs (237 faces through a frozen ViT-B/16 each "
        "-- minutes, GPU) and 3 probe runs (5 seeds x 5 folds on "
        "cached 768-d vectors -- the ladder's own cheapest arm shape). "
        "Far below the pretraining stretch that preceded it"
    ),
    "one_item_for_a_ruling": (
        "**THE PROBE'S CONFIG LAYER.** The ruling names the "
        "EXTRACTION's namespace; the probe raises the same question "
        "one layer up. Reusing ``train_cv`` would mean widening its "
        "``init`` vocabulary to accept MEBeauty inits -- which puts a "
        "MEBeauty arm inside the ladder's own arm kind, exactly the "
        "pooling the ruling forbids at the set layer. The alternative "
        "is a separate kind (``probe_mebeauty``) that calls the SAME "
        "phase3 entry point with the SAME derived recipe, keeping the "
        "two lattices apart at the config layer too, at the cost of a "
        "second call site into phase3. **the recommendation: the "
        "separate kind, because the ruling's principle plainly extends "
        "-- but it is a config-layer decision the ruling does not "
        "state, and six configs would be generated on it, so it is "
        "put rather than assumed**"
    ),
}


#: **[VERIFIED 2026-08-24, ON THE CLUSTER] THE ARM MAPPING, AND ONE
#: WITHDRAWN NUMBER.**
MAPPING_VERIFIED_AND_A_WITHDRAWAL = {
    "verified": (
        "2026-08-24, the maintainer, from each run's OWN source field -- not "
        "from tail order"
    ),
    "the_mapping": (
        "p15_pretrain_mebeauty_g1 -> masked_g1 (epoch 26, "
        "source_side_test_pcc 0.6905); _g2 -> masked_g2 (epoch 6, "
        "0.6685); _original -> masked_original (epoch 7, 0.7186). "
        "the record's assert_run_is_the_declared_arm now has a VERIFIED "
        "BASELINE to assert against"
    ),
    "the_original_arms_source_string": (
        "**RECORDED SO NO LATER TURN MISREADS IT**: the original arm's "
        "source string is ``masked_original``. It is the UNCROPPED "
        "variant, carried in the masked artifact's FORMAT -- the name "
        "says where it lives, not what was done to it. That arm has no "
        "trapezium, no landmarks and no content box"
    ),
    "the_artifacts_carry_their_own_caveats": (
        "each run persisted ``caveats_bound`` and "
        "``the_verdict_is_not_here`` beside its figures, both working "
        "as intended -- the caveats travel with the number in the "
        "artifact, not only in the record"
    ),
    "the_0_8914_withdrawal": (
        "**0.8914 IS WITHDRAWN -- the maintainer's error, recorded as such.** "
        "Confirmed ABSENT from ladder.py by grep. **The original "
        "cell's source-side comparison is therefore UNAVAILABLE**, and "
        "it is stated that way with the reason so a later turn does "
        "not reach for a remembered number. If a Phase 6 original-arm "
        "run directory carries an equivalent figure (the p6 metrics "
        "use DIFFERENT KEYS from p15's ``source_side_test_pcc`` -- "
        "check ``p6_pretrain_vit_b16_original__*/metrics.json``), it "
        "may be measured and recorded WITH ITS PROVENANCE; until then "
        "UNAVAILABLE stands. this machine cannot check it: no cluster access"
    ),
    "what_the_withdrawal_costs": (
        "only the original arm's SOURCE-SIDE comparison, which was "
        "provenance rather than verdict in any case. The TRANSFER "
        "comparator for that arm -- scut_original 0.1952 (sd 0.0280) "
        "-- is banked and unaffected, so stop 4's verdict machinery "
        "needs nothing from the withdrawn figure"
    ),
    "the_verified_comparators": (
        "source-side SCUT masked 0.7893 (g1) / 0.8306 (g2) at "
        "ladder.py:242-243; transfer cells 0.2520 (sd 0.0148), 0.1952 "
        "(sd 0.0280), 0.0830 (sd 0.0247), 0.2001 (sd 0.0142) -- all "
        "verified present, and the four SDs are what the verdict "
        "threshold is computed from"
    ),
}


#: **[RULED 2026-08-24] THE PROBE GETS ITS OWN KIND.**
PROBE_LAYER_RULED = {
    "ruled": "2026-08-24 -- build the separate kind",
    "the_reason_given": (
        "**the principle extends for the same reason it holds at the "
        "set layer: a SHARED VOCABULARY is what lets an undeclared "
        "swap manufacture a factor effect, and a shared ARM KIND has "
        "that property at the config layer too.** So ``train_cv``'s "
        "vocabulary stays CLOSED"
    ),
    "what_was_built": (
        "``probe_mebeauty``: the SAME phase3 entry point, the SAME "
        "derived recipe, a SECOND call site. Recipe identity is "
        "asserted by the difference-set test used for the occlusion "
        "probes, so 'byte-identical' stays CHECKED rather than claimed"
    ),
    "how_the_features_reach_it": (
        "through ``phase3.run``'s existing ``features_override`` seam "
        "-- no new seam was needed, and the MEBeauty set is loaded by "
        "MEBeauty's own reader, which re-asserts row order with the "
        "LADDER's ``assert_row_order``. The vocabulary is separate; "
        "the property that matters is shared"
    ),
}


#: **[CAUGHT 2026-08-24, BY THE SUITE] A GUARD FIRED ON STOP 4's OWN
#: SCHEMA, AND THE GUARD ITSELF NEEDED NARROWING.**
TRAINABLE_GUARD_FIRED = {
    "what_fired": (
        "``test_every_trainable_field_constrains_its_choices`` refused "
        "the new ``probe_mebeauty`` kind: its ``trainable`` field was "
        "shipped with ``choices=None``, so a typo would have been "
        "absorbed and trained an arm nobody configured. **A real "
        "defect in code written this stop, caught before launch** -- "
        "instance 5 of the shape that guard was built for"
    ),
    "the_fix": (
        "constrained to A's own vocabulary ``('head', 'full')`` -- the "
        "probe DERIVES its recipe from the shipped ladder arm, so it "
        "takes that arm's vocabulary and no wider one -- and pinned "
        "separately in the test's ``expected`` map, so widening it "
        "later costs a deliberate edit"
    ),
    "why_the_difference_set_test_did_not_catch_it": (
        "**worth recording**: the difference-set test compares the "
        "probe's knobs to whatever A's file says, so it would have "
        "passed a MISSPELLED value only if A were misspelled too. It "
        "checks AGREEMENT, not admissibility. The schema check is the "
        "one that bounds the vocabulary, and the two are not "
        "substitutes for each other"
    ),
    "then_the_guard_itself_was_narrowed": (
        "the same guard also demanded ``default in choices``. The new "
        "field is REQUIRED with ``default=None``, and the loader "
        "raises on an absent required key at schema.py:2150 BEFORE the "
        "default is read -- so the default is dead code there and the "
        "guard's own stated harm ('on every run that omits the field') "
        "cannot occur. **Narrowed to optional fields, and the "
        "narrowing is MEASURED, not argued**: a new test loads the "
        "shipped probe config, then omits ``trainable`` and asserts "
        "the loader refuses it. If required/default handling ever "
        "changes, the exemption fails with it"
    ),
    "the_rule_this_did_not_break": (
        "the field was made to fit the guard, not the guard to fit the "
        "field. The narrowing came SECOND, was argued from the guard's "
        "own comment, and was then checked against the loader -- the "
        "order that keeps 'never tune a constant until a check passes' "
        "intact"
    ),
}


#: **[BUILT 2026-08-24, STOP 4 -- NOT LAUNCHED] EXTRACTION AND PROBES.**
STOP_4_BUILT = {
    "built": "2026-08-24, six configs, NOT launched",
    "the_extraction": (
        "three arms: mebeauty_masked@g1, mebeauty_masked@g2, "
        "mebeauty_original. Each ASSERTS the arm-to-run mapping from "
        "the run's own metrics.json, asserts the geometry-bound "
        "variant, extracts the 237-patient cohort through that "
        "checkpoint, and writes a CLUSTER-ONLY set in MEBeauty's own "
        "namespace"
    ),
    "the_probes": (
        "three probes, A's recipe DERIVED from the shipped p7 arm and "
        "verified identical on all eleven knobs; five seeds; cleft_v1 "
        "folds; the verdict computed against a comparator DECLARED IN "
        "THE CONFIG with its own five-seed SD, thresholded by "
        "phase3.combined_claimable_delta (PLAN 4.12.1) -- the same "
        "function that reproduces the ladder's own 0.0297/0.0664"
    ),
    "what_is_asserted_in_code": (
        "the arm-to-run mapping; the geometry-bound variant, at both "
        "write and read; row order against cleft_v1, with the ladder's "
        "own checker; the within-geometry comparison "
        "(mebeauty.verdict_delta refuses a crossed pair); and that the "
        "set's init/geometry match the arm the probe config names"
    ),
    "what_rides_in_every_reading": (
        "the ImageNet anchor (0.2520) with its sentence, and the four "
        "caveats from phase15.caveats_for -- written into each probe's "
        "own metrics.json, so a number cannot be read out of the "
        "artifact without them"
    ),
    "the_lattices_stay_apart": (
        "verified: embeddings.INITS is still the ladder's three, "
        "expected_set_count still reports 24/12, MEBeauty's own "
        "arithmetic reports 3 sets from 3 runs, and the two init "
        "vocabularies are disjoint"
    ),
}


def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "survey": CANDIDATE_SURVEY_REGISTERED,
        "frontal_first": FRONTAL_FRACTION_FIRST,
        "mediapipe": MEDIAPIPE_VERDICT_ENCODED,
        "landing": LANDING_CORRECTED,
        "clone_contents": CLONE_CONTENTS,
        "exit_criteria": PHASE_15_EXIT_CRITERIA,
        "stop_1": STOP_1_BUILT,
        "stop_1_first_run": STOP_1_FIRST_RUN,
        "stop_1_banked": STOP_1_BANKED,
        "stop_2_plan": STOP_2_PLAN_PROPOSED,
        "stop_2_rulings": STOP_2_RULINGS,
        "stop_2a": STOP_2A_BUILT,
        "stop_2a_banked": STOP_2A_BANKED,
        "mapping_sheets": MAPPING_SHEETS_REVIEWED,
        "stop_2a_ii": STOP_2A_II_BUILT,
        "stop_2a_ii_banked": STOP_2A_II_BANKED,
        "the_119_refuted": THE_119_HYPOTHESIS_REFUTED,
        "overlay_defect": SHEET_OVERLAY_DEFECT,
        "label_defect": SHEET_LABEL_DEFECT,
        "diagnosis_verdicts": DIAGNOSIS_VERDICTS,
        "withdrawals": EYE_IMPRESSION_WITHDRAWALS,
        "anatomically_blind": ANATOMICALLY_BLIND_LIMITATION,
        "second_eye_pass": SECOND_EYE_PASS,
        "bounds_readings": BOUNDS_VIOLATION_READINGS,
        "bounds_distribution": BOUNDS_DISTRIBUTION_BANKED,
        "bounds_tolerance": BOUNDS_TOLERANCE_RULED,
        "tolerance_consequence": TOLERANCE_CONSEQUENCE,
        "stop_2b": STOP_2B_BUILT,
        "parity_none": PARITY_NONE_DIAGNOSED,
        "price_sentence": PRICE_SENTENCE_FIRED_ON_NO_LOSS,
        "mechanism_not_rate": MECHANISM_IS_NOT_RATE,
        "synthetic_sweep": SYNTHETIC_CAVEAT_SWEEP,
        "pipeline_not_sources": ARTIFACT_DIFFERENCES_ARE_PIPELINE_DIFFERENCES,
        "pretraining_banked": STOP_3_PRETRAINING_BANKED,
        "stop_4_proposed": STOP_4_PROPOSED,
        "mapping_verified": MAPPING_VERIFIED_AND_A_WITHDRAWAL,
        "probe_layer_ruled": PROBE_LAYER_RULED,
        "trainable_guard_fired": TRAINABLE_GUARD_FIRED,
        "stop_4": STOP_4_BUILT,
        "stop_4a_banked": STOP_4A_BANKED,
        "two_digests": TWO_DIGESTS_ONE_WORD,
        "stop_4b_banked": STOP_4B_BANKED,
        "reading_1": READING_1_MASKED_ARMS_AT_ZERO,
        "reading_2": READING_2_ORIGINAL_REACHES_PARITY,
        "reading_3": READING_3_MASKING_NOT_BEAUTY,
        "phase_6_reframed": PHASE_6_REFRAMED,
        "exit_walk": PHASE_15_EXIT_WALK,
        "phase_16_verdict": PHASE_16_VERDICT_PROPOSED,
        "criterion_1_amended": CRITERION_1_AMENDED,
        "closing": PHASE_15_CLOSING,
        "sequence_renumbered_3": PHASE_SEQUENCE_RENUMBERED_3,
        "phase_16_scheduled": PHASE_16_SCHEDULED,
        "anchor_loop": ANCHOR_LOOP_REGISTERED,
        "sequence_renumbered_4": PHASE_SEQUENCE_RENUMBERED_4,
        "stop_2b_banked": STOP_2B_BANKED,
        "padding_property": PADDING_IS_AN_ARTIFACT_PROPERTY,
        "staged_sheets": STAGED_SHEETS_REVIEWED,
        "framing_varies": FRAMING_VARIES_LIMITATION,
        "stop_3_proposed": STOP_3_PROPOSED,
        "stop_3_amended": STOP_3_AMENDED,
        "original_arm_pads": ORIGINAL_ARM_PADS_TOO,
        "key_mismatch": KEY_MISMATCH_DIAGNOSED,
        "v1_survives": V1_SURVIVES_CORRECTION,
    }
