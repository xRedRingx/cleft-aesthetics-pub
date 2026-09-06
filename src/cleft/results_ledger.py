"""The results ledger: every claim the project has made, in one place.

Registered 2026-08-16 (Phase 9, PLAN_AMENDMENT_2026-08-13 section 4's
"running results ledger that Phases 10-12 append to, so the write-up
assembles from a maintained record rather than being reconstructed from
twelve phases of run directories").

**One ledger, never two.** The main-project/additional-ideas distinction
(``phase9.ROAD_B_IS_THE_ANNEX``) is a framing FIELD on each entry.

**What an entry is.** One CLAIM -- an evidential statement with a status
-- not one record and not one finding. Mechanism findings (the flat
softmax, map-existence dependence, ViT pretraining nondeterminism, ...)
stay in their phase records, where their full context lives; the ledger
indexes what the write-up will assert, withdraw, or void. That boundary
is stated here so a thin ledger is read as a scope decision, not an
omission.

**APPEND-ONLY.** Entries are never edited and never deleted. A correction
or a status change is a NEW dated entry whose ``corrects`` field names
its target id. The enforcement is not this docstring: the suite pins the
cumulative checksum of the born population, so a silent rewrite of any
existing entry fails the build (``tests/test_phase9.py``). Appending
extends the pin; rewriting breaks it.

Born 2026-08-16 with every claim through Phase 8, main line and annex
alike, the voids, and the margin table (carried by reference below --
one source of truth, ``roadb.CONDITION_1_MARGIN_STRUCTURE``).
"""

from __future__ import annotations

import hashlib
import json

from . import roadb

LEDGER_BORN = "2026-08-16"

STATUSES = (
    "CLAIMABLE", "WITHDRAWN", "UNRESOLVED-WITHDRAWN", "VOID", "DESCRIPTIVE",
)
FRAMINGS = ("main", "additional")

#: The condition-2 margin table, carried whole BY REFERENCE -- restating
#: it would create a second copy to drift.
MARGIN_TABLE = roadb.CONDITION_1_MARGIN_STRUCTURE

#: **[CHAIN RE-DERIVED 2026-09-05] THE PREFIX CHECKSUMS WERE RE-PINNED
#: FOR ANONYMISATION, AND FOR NOTHING ELSE.**
#:
#: The append-only rule is enforced by prefix checksums pinned in the
#: suite, and the rule's whole purpose is that no RESULT is quietly
#: altered. An anonymisation alters no result -- and leaving personal
#: names in the most-read artifact in a repository that is being made
#: public is the higher cost. So the rows were rewritten, the chain
#: re-derived, and this record written in the convention every other
#: correction here follows: in place, dated, original preserved, reason
#: stated. The consequence is stated too, and it is not free.
CHAIN_RE_DERIVED = {
    "date": "2026-09-05",
    "what_changed": (
        "**nine party mentions across six rows.** ``void-deall-first-"
        "launch`` (claim), ``p9-anchor-classifier-convergence`` (claim "
        "and caveats), ``p10-rater-screen-mixed`` (caveats), "
        "``p16-anchor-loop-unresolved`` (caveats), and "
        "``ledger-condition-split-count-corrected`` (caveats). Each "
        "rewrite keeps the MECHANISM and drops only the party: a share "
        "layout that differed from the cluster copy is still a dated "
        "staging error; a method that lands near chance is still the "
        "method that was proposed at supervision; an orthodontist "
        "premise still fails a second time after Phase 1 measured it "
        "unsupported; 'parity' is still a stipulated framing rather "
        "than a derived fact; the 2026-08-31 ruling still carries its "
        "date and its two grounds"
    ),
    "what_did_not_change": (
        "**no claim, status, framing, phase, record pointer, condition, "
        "run_dirs, date, correction target, figure, threshold or "
        "ordering.** Stated as a MEASUREMENT, not an assurance -- see "
        "``how_that_is_known``"
    ),
    "how_that_is_known": (
        "the ledger was dumped row by row before and after and diffed "
        "field by field. (i) the id sequence is identical position by "
        "position, so the ordering did not move; (ii) all 38 rows x 9 "
        "non-prose fields -- status, framing, phase, record, "
        "condition_1, condition_2, run_dirs, date, corrects -- are "
        "byte-identical, **0 differences**; (iii) every numeric literal "
        "in every field of every row, in order, is unchanged: **970 "
        "figures compared, 0 moved**; (iv) the only text differences "
        "are in claim/caveats on the six rows named above, and every "
        "changed span is a party name or the grammar around one. The "
        "diff is the evidence; git history holds both sides of it"
    ),
    "the_born_prefix_is_untouched": (
        "**the first anonymised row sits at position 19, so prefixes "
        "1-18 hash exactly as they did on 2026-08-16.** "
        "``BORN_CHECKSUM`` (n=17, c46fd950...) and ``CHECKSUM_18`` are "
        "unchanged, and the 'never touch this one' instruction beside "
        "the born pin was never tested by this pass"
    ),
    "the_superseded_checksums": {
        19: "eb40d3712f87f78149cf57bacada5144fcf6f58a64962609e6c5a06a410bde4b",
        20: "8fdc8f71d7d48f787ce33b56dad32e96ca1bd15a19c739d3ff1efe06cfd2e4c8",
        21: "88e652afc2349be619a758fccba3efc42936b7d00fb7618f03525b58d62b630d",
        22: "08fe50558ad8a58d1a464f9d32c7dc5e21e4e2b3171fac9c47baf75fb01891c2",
        24: "a1897e84a93b1e1392a96267decf7c8b282c5813937daf618c682014d5612477",
        25: "e513758781465346e371b96a63b45e4caca6c93c44a0d627861fc2d9a81aa394",
        26: "98bdb3d433e6b3f22c4f3bb003daf9560da0692b74b686ee4030dd0610cb9d53",
        28: "ff2225a9f37713a5a24bc9fb0c67120200406feedeff818a222f875559fbe1c7",
        31: "3a65dc92e3a15b196c8e3ce719558dbc504ee3d2fbf450edf98e9fb0f7ca22c3",
        32: "028e41897e869991c515c3033257e70bb89a446a3d68795f914af08f778f7fdb",
        37: "889fb62fac03260dd6262e7ada3d9f74ce08ce2d81cebbf51f0aced40d9ec557",
        38: "f01228a7199b44dadfcdb33fea127f24e2dd44c861a0dc3925620db25a569a3b",
    },
    "the_superseded_values_are_kept_here": (
        "**the originals are preserved, as every correction here "
        "preserves its original.** Twelve pinned lengths moved; the "
        "values they held until 2026-09-05 are in the dict above, so "
        "the pre-anonymisation chain is re-checkable from inside the "
        "record and not only from git. Prefix lengths not listed did "
        "not move"
    ),
    "why_it_was_done": (
        "the repository is being made public. **A record that names "
        "people beside fifteen refuted premises is the wrong thing to "
        "publish, whether or not each refutation is correct** -- the "
        "refutations are of PREMISES and stand on the measurements, "
        "and attaching a name to them adds nothing evidential while "
        "costing a person something real. The mechanism is what the "
        "record is for; the party never was"
    ),
    "the_honest_consequence": (
        "**the chain now proves that nothing has been edited SINCE "
        "2026-09-05, not that nothing was ever edited.** That is a "
        "strictly weaker guarantee and it is stated rather than "
        "glossed. What replaces the lost span is ordinary and "
        "checkable: the diff above, and a git history in which this "
        "commit's changes to the ledger are visible line by line. A "
        "reader who does not want to take this record's word for it "
        "can read the two versions"
    ),
    "what_would_not_have_justified_this": (
        "a figure that read better, a status that aged badly, a "
        "precedence claim that turned out wrong. **Those append; they "
        "never rewrite** -- ``ledger-condition-split-count-corrected`` "
        "is what that looks like, and it is one of the rows this pass "
        "touched WITHOUT touching its verdict. The narrowness is the "
        "point: re-deriving the chain is defensible exactly once, for "
        "a change that carries no evidential content"
    ),
}

ENTRIES: tuple[dict, ...] = (
    {
        "id": "p7-best-arm-descriptive",
        "claim": (
            "p7_d1_vit_b16_imagenet_g1 is the highest-scoring arm: PCC "
            "0.2520 (sd 0.0148, 5 seeds), reproducing phase-3 gate 2's "
            "0.2529 through the live path"
        ),
        "status": "DESCRIPTIVE",
        "framing": "main",
        "phase": "p7",
        "record": "ladder.BEST_ARM",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("p7_d1_vit_b16_imagenet_g1__3f71a6a9__p7-d1-vit-imagenet",),
        "date": "2026-08-02",
        "caveats": (
            "an ordering, not a comparative claim -- the comparative claim "
            "beside it is withdrawn (p7-best-arm-vs-challenger)",
        ),
        "corrects": None,
    },
    {
        "id": "p7-best-arm-vs-challenger",
        "claim": (
            "0.2520 is claimably ahead of the nearest challenger "
            "(swin_b scut_original g2, 0.2092; delta 0.0428)"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p7",
        "record": "ladder.BEST_ARM.nearest_challenger",
        "condition_1": {"n_excluding_zero": 0, "n_seeds": 5},
        "condition_2": {"margin": 1.05, "threshold": 0.0406},
        "run_dirs": ("p7_paired_ladder",),
        "date": "withdrawn 2026-08-04",
        "caveats": (
            "'nothing beats it' overstates a 1.05x; the descriptive "
            "ordering stands",
        ),
        "corrects": None,
    },
    {
        "id": "p7-q1-swin-g2",
        "claim": (
            "SCUT-original pretraining beats ImageNet for swin_b at G2: "
            "delta +0.2027 (threshold 0.0529)"
        ),
        "status": "CLAIMABLE",
        "framing": "main",
        "phase": "p7",
        "record": "ladder.STAGE_D_AT_G2.q1.rederived.swin_b",
        "condition_1": {"n_excluding_zero": 5, "n_seeds": 5},
        "condition_2": {"margin": 3.83},
        "run_dirs": ("p7_paired_ladder",),
        "date": "2026-08-04",
        "caveats": (
            "the baseline is 0.0065 -- Swin's ImageNet G2 cell barely "
            "correlates",
            "the same comparison at G1 is 0.44x at 0/5 -- never quoted "
            "without both facts",
        ),
        "corrects": None,
    },
    {
        "id": "p7-cohort-cannot-resolve",
        "claim": (
            "this cohort cannot resolve PCC differences of 0.04-0.10 "
            "between arms: 29 of 30 paired comparisons fail condition 1"
        ),
        "status": "DESCRIPTIVE",
        "framing": "main",
        "phase": "p7",
        "record": "ladder.COHORT_CANNOT_RESOLVE; ladder.LADDER_PAIRED_AUDIT",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("p7_paired_ladder",),
        "date": "2026-08-04",
        "caveats": (
            "the project's governing measurement: per-seed BCa intervals "
            "~0.28 wide at n=237",
        ),
        "corrects": None,
    },
    {
        "id": "p7b-claimably-worse",
        "claim": (
            "the searched configuration is claimably worse than the "
            "baseline (delta -0.0542)"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p7b",
        "record": (
            "phase7b.SEARCH_AXIS_VERDICTS.outcome.against_baseline; "
            "phase7b.CLAIMABLY_WORSE_WITHDRAWN"
        ),
        "condition_1": {"n_excluding_zero": 1, "n_seeds": 5},
        "condition_2": {"margin": 2.37},
        "run_dirs": (),
        "date": "withdrawn 2026-08-04",
        "caveats": (
            "condition 2 only; condition 1 sat UNRESOLVED in the run's "
            "own metrics the whole time",
        ),
        "corrects": None,
    },
    {
        "id": "p7c-nine-verdicts",
        "claim": (
            "the nine augmentation verdicts (deltas to -0.1005), "
            "withdrawn as a phase-level outcome"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p7c",
        "record": "ladder.COHORT_CANNOT_RESOLVE.tested (phase-level outcome)",
        "condition_1": {"n_excluding_zero": 0, "n_seeds": 45},
        "condition_2": {"margin": 3.62},
        "run_dirs": (),
        "date": "withdrawn 2026-08-04",
        "caveats": (),
        "corrects": None,
    },
    {
        "id": "p7d-grid-density",
        "claim": (
            "grid density is the implicated variable: densifying destroys "
            "(patch8 vs b16: -0.2089) and coarsening at 512 recovers "
            "(b32@512 vs patch16@512: +0.1386); the interpolation "
            "mechanism is retired jointly"
        ),
        "status": "CLAIMABLE",
        "framing": "main",
        "phase": "p7d",
        "record": "ladder.PHASE_7D_CLOSING",
        "condition_1": {
            "patch8_vs_b16": {"n_excluding_zero": 5, "n_seeds": 5},
            "b32_512_vs_patch16_512": {"n_excluding_zero": 5, "n_seeds": 5},
        },
        "condition_2": {
            "patch8_vs_b16": {"margin": 6.42},
            "b32_512_vs_patch16_512": {"margin": 5.32},
        },
        "run_dirs": (
            "the five p7d arms at SHA c3a8dabd and the p7d paired run "
            "(declared in configs/p7d_paired.yaml)",
        ),
        "date": "2026-08-15",
        "caveats": (
            "tokens and per-patch fraction are deterministically linked: "
            "one variable, not two",
        ),
        "corrects": None,
    },
    {
        "id": "p7d-null-and-concat",
        "claim": (
            "patch32 vs b16 (-0.0001) and concat vs b16 (+0.0073) -- the "
            "7B concat prior neither confirmed nor overturned (sign "
            "disagreement, both unclaimable)"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p7d",
        "record": "ladder.PHASE_7D_OBSERVED; ladder.PHASE_7D_CLOSING",
        "condition_1": {"n_excluding_zero": 0, "n_seeds": 5},
        "condition_2": None,
        "run_dirs": ("declared in configs/p7d_paired.yaml",),
        "date": "2026-08-15",
        "caveats": (),
        "corrects": None,
    },
    {
        "id": "p7d-mvitv2",
        "claim": (
            "both MViTv2 contrasts (vs b16, vs swin) -- unresolved and "
            "withdrawn"
        ),
        "status": "UNRESOLVED-WITHDRAWN",
        "framing": "main",
        "phase": "p7d",
        "record": "ladder.PHASE_7D_CLOSING",
        "condition_1": {"n_excluding_zero": 0, "n_seeds": 5},
        "condition_2": {"margins": (1.77, 1.89)},
        "run_dirs": ("declared in configs/p7d_paired.yaml",),
        "date": "2026-08-15",
        "caveats": (
            "'hierarchy alone does not determine' is FORBIDDEN as a "
            "claim -- downgraded in place, dated",
        ),
        "corrects": None,
    },
    {
        "id": "p8-parameter-dependence",
        "claim": (
            "arm A's Grad-CAM maps depend on the model's parameters: "
            "separability -0.1061, 95% CI [-0.2047, -0.0020], the "
            "criterion's own null at P = 0.0037"
        ),
        "status": "CLAIMABLE",
        "framing": "main",
        "phase": "p8",
        "record": (
            "phase8.PARAMETER_DEPENDENCE_ESTABLISHED; phase8.PHASE_8_CLOSING"
        ),
        "condition_1": {"interval_95": (-0.2047, -0.0020), "excludes_zero": True},
        "condition_2": None,
        "run_dirs": (),
        "date": "2026-08-04",
        "caveats": (
            "evidenced under its own pre-registered criterion, not the "
            "PLAN 4.3 two-condition arm comparison",
        ),
        "corrects": None,
    },
    {
        "id": "p8-framing-not-anatomy",
        "claim": (
            "the maps attend substantially to framing rather than anatomy "
            "(out-of-content mass 1.60x uniform, 13 of 15)"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p8",
        "record": "phase8.FRAMING_NOT_ANATOMY; phase8.PHASE_8_CLOSING.refuted",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": (),
        "date": "refuted 2026-08-08",
        "caveats": (
            "re-measured against each patient's OWN expectation: median "
            "ratio 1.0652, 10 of 15 above their own, where 8 is coin "
            "flips -- refuted, not weakened",
        ),
        "corrects": None,
    },
    {
        "id": "p8-ranking-contrast",
        "claim": (
            "Grad-CAM produces a ranking (0.659); the gated graph "
            "attention never produced one at all"
        ),
        "status": "DESCRIPTIVE",
        "framing": "main",
        "phase": "p8",
        "record": "phase8.A_VS_B_COMPARISON_STATEMENT",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": (
            "p8_node_weights__7be67a24__p8-node-weights",
        ),
        "date": "2026-08-15",
        "caveats": (
            "the graph-head contribution confound is attached "
            "(phase8.GRAPH_HEAD_CONTRIBUTION_CANDIDATE) and travels with "
            "any quotation",
        ),
        "corrects": None,
    },
    {
        "id": "void-first-ladder",
        "claim": (
            "the first ladder's runs are VOID, superseded by the rebuilt "
            "ladder"
        ),
        "status": "VOID",
        "framing": "main",
        "phase": "p7",
        "record": "ladder.SIBLING_RUNS_AUDIT (the void-versus-good history)",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": (),
        "date": "superseded before 2026-08-02",
        "caveats": (
            "the measured hazard: the same nominal arm differing by 0.068 "
            "PCC between void and good runs",
        ),
        "corrects": None,
    },
    {
        "id": "void-scut-animation-first-launch",
        "claim": (
            "the first SCUT animation launch (SHA e40f09de) is VOID: 6-7 "
            "attempts each on pre-fix code"
        ),
        "status": "VOID",
        "framing": "main",
        "phase": "p8",
        "record": (
            "phase8.SCUT_ANIMATION_FIRST_LAUNCH; "
            "phase8.SCUT_ANIMATION_CLOSING.void_runs"
        ),
        "condition_1": None,
        "condition_2": None,
        "run_dirs": (),
        "date": "2026-08-16",
        "caveats": (
            "superseded by the -2 runs at SHA a2f7c8c3 -- the citable "
            "pair",
        ),
        "corrects": None,
    },
    {
        "id": "roadb-resolution-fall",
        "claim": (
            "ViT falls from 224 as resolution rises -- the first coherent "
            "survivor set: masked 224->512 -0.1848; imagenet 224->512 "
            "-0.1626; imagenet 224->768 -0.1583"
        ),
        "status": "CLAIMABLE",
        "framing": "additional",
        "phase": "roadb-p7",
        "record": "roadb.PHASE_7_PAIRED_RESULTS",
        "condition_1": {
            "masked_224_512": {"n_excluding_zero": 5, "n_seeds": 5},
            "imagenet_224_512": {"n_excluding_zero": 5, "n_seeds": 5},
            "imagenet_224_768": {"n_excluding_zero": 5, "n_seeds": 5},
        },
        "condition_2": {
            "masked_224_512": {"margin": 8.57},
            "imagenet_224_512": {"margin": 7.38},
            "imagenet_224_768": {"margin": 6.91},
        },
        "run_dirs": ("roadb_p7_paired_resolution",),
        "date": "2026-08-12",
        "caveats": (
            "the masked survivor is confounded by the pretraining draw "
            "(ViT-specific nondeterminism) and is quoted that way",
            "mechanism attribution refined by ladder.PHASE_7D_CLOSING: "
            "grid density, with the interpolation mechanism retired",
        ),
        "corrects": None,
    },
    {
        "id": "roadb-resolution-withdrawn",
        "claim": "the other 17 of 20 Road B resolution pairs",
        "status": "WITHDRAWN",
        "framing": "additional",
        "phase": "roadb-p7",
        "record": "roadb.PHASE_7_PAIRED_RESULTS",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("roadb_p7_paired_resolution",),
        "date": "2026-08-12",
        "caveats": (),
        "corrects": None,
    },
    {
        "id": "roadb-region-crop",
        "claim": (
            "Branch 3 region-crop: the first five-arm set VOID on five "
            "defects; the corrected set ran and closed with nothing "
            "claimable"
        ),
        "status": "WITHDRAWN",
        "framing": "additional",
        "phase": "roadb-branch3",
        "record": (
            "roadb.REGION_CROP_STRUCTURE; roadb.REGION_CROP_ARMS_OBSERVED"
        ),
        "condition_1": None,
        "condition_2": None,
        "run_dirs": (),
        "date": "2026-08-14",
        "caveats": ("the defective first set is VOID, recorded in place",),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-16, after the p9_prototypes run. Entries are
    # appended at the END, always: the author of this ledger first placed
    # this entry mid-tuple, which shifted the born prefix -- exactly the
    # rewrite the pinned checksum exists to catch. Moved before it shipped;
    # the lesson stays here in prose.
    {
        "id": "p9-prototypes-descriptive",
        "claim": (
            "three class3 medoids (patients 183/147/148) stand as "
            "descriptive artifacts -- three real faces per grade; "
            "representativeness claims are dead: nearest-medoid accuracy "
            "0.257 sits below chance 0.333, bootstrap persistence ~0.4 "
            "on all three"
        ),
        "status": "DESCRIPTIVE",
        "framing": "main",
        "phase": "p9",
        "record": "phase9.PROTOTYPES_OBSERVED",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("the p9_prototypes keeper run (dir per the run listing)",),
        "date": "2026-08-16",
        "caveats": (
            "the at-ceiling LOO figures (0.992, 0.967) are the "
            "single-deletion-robustness artifact and may never be quoted "
            "as stability",
            "corroborates the t-SNE companion (0.409 vs majority 0.502) "
            "exactly as pre-committed in PROTOTYPES_REGISTERED",
        ),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-16, after the 25-set's first launch ------------
    {
        "id": "void-deall-first-launch",
        "claim": (
            "the first 25-set launch is VOID: the labels CSV sat at bch "
            "root on the cluster copy rather than inside the images "
            "folder (the share layout as received vs the cluster "
            "copy -- a dated staging error), and the pinned-filename "
            "read "
            "refused; four instant attempts, nothing scored"
        ),
        "status": "VOID",
        "framing": "main",
        "phase": "p9",
        "record": "phase9.DEALL_REFERENCE_READS (the dated layout correction)",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("p9_deall_reference__d67248c9__p9-deall-reference-2",),
        "date": "2026-08-16",
        "caveats": (
            "fixed by restoring the canonical layout on the cluster; the "
            "re-declared rollup covers 76 files and the earlier 75-vs-76 "
            "flag resolved as exactly this file",
            "the four instant attempts are data points on "
            "phase8.RETRY_LIMIT_IS_NOT_HOLDING",
        ),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-16, after the prototype classifier ran ---------
    {
        "id": "p9-anchor-classifier-convergence",
        "claim": (
            "the prototype classifier -- a method proposed at "
            "supervision -- "
            "lands near chance: 3-class accuracy 0.333-0.363 across all "
            "eight cells (chance 0.333, majority 0.502), PCC 0.15-0.18 "
            "everywhere, under the trained head's 0.2520; and the "
            "anchors themselves do not neighbour by grade "
            "(self-consistency 4/25 Euclidean, 3/25 cosine vs ~4.75/25 "
            "by chance), so the method's premise fails among the trusted "
            "references. The FOURTH convergent measurement that arm A's "
            "feature-space neighbourhoods do not carry clinical grade: "
            "t-SNE companion 0.409, medoid companion 0.257, bootstrap "
            "persistence ~0.4, and the anchor classifier"
        ),
        "status": "DESCRIPTIVE",
        "framing": "main",
        "phase": "p9",
        "record": "phase9.PROTOTYPE_CLASSIFIER_OBSERVED",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": (
            "p9_prototype_classifier__18d0d9fb__p9-prototype-classifier",
        ),
        "date": "2026-08-16",
        "caveats": (
            "anchor grades are trusted single grades from the survey "
            "lineage, not verified-unanimous (confirmable at source, "
            "not yet confirmed) -- travels with every quotation",
            "the most persuasive form the result can take: the "
            "method measured is the one that was proposed",
        ),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-17, after the first CleftGNN launch ------------
    {
        "id": "void-cleftgnn-first-launch",
        "claim": (
            "the first CleftGNN launch is VOID: the label reader searched "
            "for a 'consensus' header the primary sheet does not carry "
            "(its columns are RanaPhotoID, five raters, Average, Median), "
            "so four attempts died deterministically before any training; "
            "the training label is the sheet's Median column"
        ),
        "status": "VOID",
        "framing": "main",
        "phase": "p10",
        "record": (
            "phase10.CLEFTGNN_FIRST_LAUNCH_VOID; "
            "phase10.CONSENSUS_LABEL_IS_THE_MEDIAN"
        ),
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("p10_cleftgnn__9a28f8a8__p10-cleftgnn",),
        "date": "2026-08-17",
        "caveats": (
            "the sheet's layout was in data/scoresheet.py since Phase 1 -- "
            "an R10 failure (a new reader written beside one that already "
            "knew the answer), not a data problem",
            "three more retries for phase8.RETRY_LIMIT_IS_NOT_HOLDING: "
            "four attempts against a stated limit of one",
        ),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-17, after the second CleftGNN launch ----------
    {
        "id": "void-cleftgnn-second-launch",
        "claim": (
            "the second CleftGNN launch is VOID: four attempts died six "
            "seconds in on non-finite predictions at the first pcc call. "
            "Measured cause -- NOT the thin-class log(0) hypothesis, which "
            "is refuted (the bias is smoothed and finite at -23.65) -- but "
            "divergence in the first optimiser step, from 36-region summed "
            "features of magnitude 21.5 entering a zero-initialised "
            "classifier at lr 0.01"
        ),
        "status": "VOID",
        "framing": "main",
        "phase": "p10",
        "record": (
            "phase10.CLEFTGNN_SECOND_LAUNCH_VOID; "
            "phase10.CLEFTGNN_NAN_MEASURED"
        ),
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("p10_cleftgnn__32519178__p10-cleftgnn-2",),
        "date": "2026-08-17",
        "caveats": (
            "the label correction held: the Median column verified on all "
            "237, counts 1:5, 2:89, 3:110, 4:30, 5:3",
            "two apparatus findings recorded: gate 3 cannot catch a "
            "non-finite epoch 0, and this model's forward has no CPU "
            "kernel so no laptop test can execute it",
            "four more retries for phase8.RETRY_LIMIT_IS_NOT_HOLDING",
        ),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-17, the rater screen and the third void -------
    {
        "id": "p10-rater-screen-mixed",
        "claim": (
            "rater-specific modelling does not beat the panel mean on this "
            "cohort: pooled OOF PCC per rater is -0.0098 (cleft patient), "
            "0.1606 (orthodontist), 0.2150 (SLT), 0.2703 (plastic "
            "surgeon), 0.1362 (psychologist) against the arm's 0.2520 -- "
            "four below, one nominally above and INSIDE the arm's seed "
            "band. The registered MIXED reading fired; the Phase 1 "
            "learnability prior (mean 0.6022 vs orthodontist 0.4944) is "
            "confirmed by an independent route, and no rater approaches "
            "the literature's per-rater figures"
        ),
        "status": "DESCRIPTIVE",
        "framing": "main",
        "phase": "p10",
        "record": "phase10.RATER_SCREEN_OBSERVED; phase10.RATER_SCREEN_REGISTERED",
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("the p10_rater_screen keeper run",),
        "date": "2026-08-17",
        "caveats": (
            "each rater is a DIFFERENT target, so these are not paired "
            "against the 0.2520 arm's vectors -- the reading is level "
            "against the arm and its seed sd 0.0148",
            "the plastic surgeon's +0.0183 is 1.24 sd, inside the "
            "registered two-sd refutation threshold: the prior is not "
            "refuted and the 400-run ladder is not triggered",
            "the orthodontist premise raised at supervision fails a "
            "second time -- fourth of five here, after Phase 1 "
            "measured it unsupported on reliability",
        ),
        "corrects": None,
    },
    {
        "id": "void-cleftgnn-third-launch",
        "claim": (
            "the third CleftGNN protocol-arm launch is VOID: non-finite "
            "again at ~6s under the frozen backbone, four attempts. The "
            "backbone-gradient diagnosis is REFUTED (freezing removed "
            "23.5M parameters from the optimiser and the divergence "
            "survived); the surviving hypothesis is feature scale. The "
            "faithful arm ran to completion but every PCC is nan on a "
            "CONSTANT predictor, confounded with the same divergence"
        ),
        "status": "VOID",
        "framing": "main",
        "phase": "p10",
        "record": (
            "phase10.CLEFTGNN_THIRD_LAUNCH_VOID; "
            "phase10.CLEFTGNN_SCALE_MEASURED; phase10.FAITHFUL_ARM_OBSERVED"
        ),
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("the frozen p10_cleftgnn relaunch; the p10_cleftgnn_faithful run",),
        "date": "2026-08-17",
        "caveats": (
            "the faithful arm's numbers are NOT a finding about their "
            "regime: our own bug explains the collapse, and the regime "
            "question stays open until a model that trains runs it",
            "six more retries for phase8.RETRY_LIMIT_IS_NOT_HOLDING",
        ),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-17: banked under the name of what was actually
    # measured. The CleftGNN row is deliberately NOT written -- it stays
    # unclaimed and available for a build whose GNN branch contributes.
    {
        "id": "p10-resnet50-sabm-head",
        "claim": (
            "a FROZEN ImageNet ResNet-50 with CleftGNN's SABM attention "
            "head, trained by cross-entropy on the score sheet's median "
            "grade under the group's recipe and evaluated against the "
            "panel mean, scores pooled-OOF PCC 0.0241 with a seed sd of "
            "0.0676 across five seeds (per seed 0.0069, 0.0680, -0.0138, "
            "-0.0555, 0.1151) -- indistinguishable from zero, the seed sd "
            "being itself the size of a single correlation's sampling "
            "error at n=237. Its predictions span 20.9% of the label sd "
            "on average and 33.7% at best"
        ),
        "status": "DESCRIPTIVE",
        "framing": "main",
        "phase": "p10",
        "record": (
            "phase10.ROUTE_3_OBSERVED; phase10.THIRD_READING_PROPOSED; "
            "phase10.COLLAPSE_DIAGNOSED"
        ),
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("p10-cleftgnn-5",),
        "date": "2026-08-17",
        "caveats": (
            "ATTACHED FACT, not a footnote: the GNN branch's measured "
            "image-dependence is 0.0068, so what varies in this model "
            "comes from the SABM branch alone -- this row is NOT a "
            "CleftGNN measurement, and the CleftGNN row is deliberately "
            "left unclaimed for a build whose GNN branch contributes",
            "criterion (i) is UNREPORTED: --stages was never taken for "
            "this run, so whether the fused ratio cleared 0.4 on the "
            "arm's own fold is unknown and the probe's figure must not be "
            "assumed to transfer (phase10.CRITERION_I_UNREPORTED)",
            "the training label (the sheet's median grade) differs from "
            "the evaluation label (the panel mean) by design fidelity -- "
            "the standing comparability caveat",
            "five registered deviations from the group's recipe "
            "(phase10.REGISTERED_DEVIATIONS), and a 6.33x spread in "
            "prediction sd across seeds that stopping behaviour does not "
            "explain (phase10.EPOCHS_ANSWER_NEITHER_READING)",
        ),
        "corrects": None,
    },
    {
        "id": "p10-replication-vs-best-arm-withdrawn",
        "claim": (
            "the 0.2520 arm being ahead of the CleftGNN replication "
            "(0.0242, sd 0.0676) is NOT CLAIMABLE: paired d +0.2279 with "
            "condition 1 FALSE (3 of 5 seeds exclude zero) and condition "
            "2 TRUE at 3.76x. Under PLAN 4.3 both must hold, so the "
            "comparison is WITHDRAWN -- unresolvable at this cohort's "
            "resolution, the FOURTH independent arrival at "
            "COHORT_CANNOT_RESOLVE"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p10",
        "record": (
            "phase10.PAIRED_BCA_OBSERVED; phase10.PHASE_10_PAIRED_"
            "REGISTERED; phase10.PHASE_10_CLOSING"
        ),
        "condition_1": (
            "FALSE -- 3 of 5 seeds exclude zero in the paired BCa over "
            "patients"
        ),
        "condition_2": (
            "TRUE at 3.76x -- delta 0.2279 against a combined-seed "
            "threshold of 0.0607, inside the mixed 3.6x-4.7x band where "
            "3.83x passed and 4.34x and 4.69x failed, so the margin does "
            "not order it and condition 1 decides"
        ),
        "run_dirs": ("p10-paired", "p10-cleftgnn-6", "p7-d1-vit-imagenet"),
        "date": "2026-08-17",
        "caveats": (
            "THE CAUSE IS THE LOSER'S INSTABILITY, and it is attached "
            "rather than footnoted: the replication's seed sd is 0.0676 "
            "against the arm's 0.0148 -- 4.6x -- and the threshold is "
            "driven by the noisier arm. An arm that wobbles by 0.0676 "
            "cannot be claimably beaten however far behind it is",
            "this withdrawal does NOT say the two architectures are "
            "comparable: it says this cohort cannot resolve the "
            "difference at this criterion. The replication does not span "
            "the label range and the descriptive gap is 0.2279",
            "the prediction was registered before the run and held "
            "exactly -- delta 0.2279, threshold 0.0607, margin 3.76x, "
            "condition 1 as the decider. Delta and threshold were DERIVED "
            "from recorded figures, so the arithmetic was not at risk; "
            "the verdict was",
            "this run FITS NOTHING -- every vector was written by a "
            "keeper run that already happened",
        ),
        "corrects": None,
    },
    {
        "id": "p11-iem-loss-costs-pcc-withdrawn",
        "claim": (
            "training the 0.2520 cell on the manuscript's eq (16) instead "
            "of MSE COSTS correlation -- PCC 0.1855 against the matched "
            "control's 0.2552, a fall of 0.0697 -- is NOT CLAIMABLE: "
            "paired d -0.0697 with condition 1 FALSE (3 of 5 seeds "
            "exclude zero) and condition 2 TRUE at 2.57x. Both must hold, "
            "so the comparison is WITHDRAWN"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p11",
        "record": (
            "phase11.PAIRED_LOSS_OBSERVED; phase11.LOSS_ARMS_OBSERVED; "
            "phase11.PHASE_11_CLOSING"
        ),
        "condition_1": "FALSE -- 3 of 5 seeds exclude zero",
        "condition_2": (
            "TRUE at 2.57x -- delta 0.0697 against a combined-seed "
            "threshold of 0.0271, just above the 2.55x line below which "
            "nothing in this project has ever passed, so the margin does "
            "not order it and condition 1 decides"
        ),
        "run_dirs": ("p11-paired", "p11-iem-arm", "p11-mse-control"),
        "date": "2026-08-17",
        "caveats": (
            "THE DESCRIPTIVE PICTURE IS UNAMBIGUOUS AND IS WITHDRAWN "
            "ANYWAY: ten of ten seed-wise comparisons run in the expected "
            "direction with DISJOINT ranges (arm [0.1511, 0.2118], "
            "control [0.2382, 0.2711]), and the per-patient intervals "
            "still exclude zero on only three of five seeds. A fifth "
            "independent arrival at COHORT_CANNOT_RESOLVE",
            "the fall belongs to the LOSS, not the target: the matched "
            "control scores 0.2552 against the 0.2520 arm's 0.2520 -- "
            "inside its own seed sd of 0.0148 -- despite training on the "
            "median grade rather than the panel mean",
            "the loss ran eq (16) VERBATIM, carrying its two measured "
            "defects: the value inversion below |d| = 0.19753 and the "
            "gradient inversion below |d| = 0.0719 "
            "(phase11.IEM_CROSSOVER_INVERTS)",
            "the arm and the control differ in the LOSS and nothing else "
            "-- same cell, seeds, folds and target -- which is what makes "
            "this pair admissible where the IEM arm against the LADDER is "
            "descriptive only",
        ),
        "corrects": None,
    },
    {
        "id": "p11-iem-loss-improves-iem-withdrawn",
        "claim": (
            "the same arm is BETTER on the metric it optimises -- IEM "
            "0.5825 against the control's 0.6130 -- and that is NOT "
            "CLAIMABLE either: paired d -0.0305 (lower is better) with "
            "condition 1 FALSE (2 of 5 seeds exclude zero) and condition "
            "2 TRUE at 2.68x. WITHDRAWN"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p11",
        "record": (
            "phase11.PAIRED_LOSS_OBSERVED; phase11.LOSS_ARMS_OBSERVED; "
            "phase11.PHASE_11_CLOSING"
        ),
        "condition_1": "FALSE -- 2 of 5 seeds exclude zero",
        "condition_2": (
            "TRUE at 2.68x -- delta 0.0305 against a combined-seed "
            "threshold of 0.0114, baseline sd recomputed 0.00744; also "
            "just above the 2.55x line, so condition 1 decides"
        ),
        "run_dirs": ("p11-paired", "p11-iem-arm", "p11-mse-control"),
        "date": "2026-08-17",
        "caveats": (
            "LOWER IS BETTER on this metric, so the negative delta is the "
            "IEM arm winning -- the direction is recorded per metric "
            "because IEM and PCC run opposite ways and one sign "
            "convention would mislead on one of them",
            "this row and its sibling travel TOGETHER by registration: "
            "IEM alone is the arm marking its own homework and PCC alone "
            "hides whether the loss did what it was asked "
            "(phase11.PHASE_11_BUILD_REGISTERED)",
            "IEM here is scored against the sheet's MEDIAN GRADE, the "
            "target both arms trained on; the PCC row is scored against "
            "the PANEL MEAN, the project's standing evaluation label",
            "the same two eq (16) defects travel with this number as with "
            "its sibling",
        ),
        "corrects": None,
    },
    {
        "id": "p12-basal-vs-frontal-bar-withdrawn",
        "claim": (
            "the frontal bar being ahead of a basal-only arm -- A 0.2505 "
            "(sd 0.0077) against B 0.1880 (sd 0.0388), a 0.0625 gap whose "
            "DIRECTION NOBODY DOUBTS -- is NOT CLAIMABLE: paired d "
            "+0.0625 with condition 1 FALSE (0 of 5 seeds exclude zero) "
            "and condition 2 TRUE at 1.8x. WITHDRAWN -- the SIXTH "
            "independent arrival at COHORT_CANNOT_RESOLVE, and the first "
            "where the direction of the difference was never in question"
        ),
        "status": "WITHDRAWN",
        "framing": "main",
        "phase": "p12",
        "record": (
            "phase12.PAIRED_OBSERVED; phase12.ARMS_B_C_OBSERVED; "
            "phase12.PHASE_12_CLOSING"
        ),
        "condition_1": "FALSE -- 0 of 5 seeds exclude zero",
        "condition_2": (
            "TRUE at 1.8x -- delta 0.0625 against a combined-seed "
            "threshold of 0.0347, below the 2.55x floor beneath which "
            "nothing in this project has ever passed condition 1"
        ),
        "run_dirs": ("p12-paired", "p12-arm-a-frontal", "p12-arm-b-basal"),
        "date": "2026-08-23",
        "caveats": (
            "THE NOISIER ARM SETS THE THRESHOLD: B's seed sd 0.0388 is "
            "5x A's 0.0077, and every per-seed interval includes zero -- "
            "the cohort cannot resolve even a gap whose direction is "
            "certain",
            "the staging-geometry caveat travels: r(AR, mean) = +0.1601 "
            "(phase12.BASAL_CONFOUND_OBSERVED), and the report cannot "
            "distinguish real anatomy from crop-geometry artifact",
            "B still carries REAL frontal-independent signal -- 0.1880 "
            "is ~5x A's seed sd above zero -- and that descriptive fact "
            "is unchallenged by this withdrawal",
            "236-patient both-views cohort, folds carried verbatim from "
            "cleft_v1; patient 238 excluded from BOTH arms",
        ),
        "corrects": None,
    },
    {
        "id": "p12-concat-vs-frontal-bar-unresolved",
        "claim": (
            "adding the basal view to the frontal bar -- C 0.2640 (sd "
            "0.0321), descriptively the project's highest arm mean, "
            "against A 0.2505 (sd 0.0077) -- is UNRESOLVED: paired d "
            "+0.0135 with condition 1 FALSE (0 of 5) and condition 2 "
            "FALSE at 0.47x. Reading 2, the registered prior, fired "
            "verbatim: descriptively ahead, not claimable"
        ),
        "status": "UNRESOLVED-WITHDRAWN",
        "framing": "main",
        "phase": "p12",
        "record": (
            "phase12.PAIRED_OBSERVED; phase12.PHASE_12_READINGS; "
            "phase12.PHASE_12_CLOSING"
        ),
        "condition_1": "FALSE -- 0 of 5 seeds exclude zero",
        "condition_2": (
            "FALSE at 0.47x -- delta 0.0135 against a threshold of "
            "0.0289 driven by C's seed sd (0.0321, inherited from B's "
            "0.0388 through the basal channel, ~4x A's)"
        ),
        "run_dirs": ("p12-paired", "p12-arm-a-frontal", "p12-arm-c-concat"),
        "date": "2026-08-23",
        "caveats": (
            "the per-seed direction was 3/5 with seeds 99 and 12345 "
            "negative -- exactly B's two worst seeds, the basal channel's "
            "noise carried into the concatenation",
            "the staging-geometry caveat travels: r(AR, mean) = +0.1601 "
            "(phase12.BASAL_CONFOUND_OBSERVED)",
            "the pre-written interpretation sentences (restores-evidence "
            "vs raters-never-saw) stay UNUSED: their applies-when clause "
            "requires reading 1, which did not fire",
            "the missing-basal-view explanation of the r~0.3 plateau "
            "stays dead (ladder.BASAL_RATIONALE_UNSUPPORTED): the view "
            "adds no claimable improvement even measured directly",
        ),
        "corrects": None,
    },
    {
        "id": "p12-concat-vs-capacity-unresolved",
        "claim": (
            "the concat beating CAPACITY -- C 0.2640 against D "
            "(frontal(+)frontal) 0.2534 (sd 0.0105) -- is UNRESOLVED: "
            "paired d +0.0106 with condition 1 FALSE (0 of 5) and "
            "condition 2 FALSE at 0.36x. The pair that would have made a "
            "C-vs-A gain attributable to the VIEW cannot order the two "
            "arms"
        ),
        "status": "UNRESOLVED-WITHDRAWN",
        "framing": "main",
        "phase": "p12",
        "record": (
            "phase12.PAIRED_OBSERVED; phase12.ARMS_A_D_OBSERVED; "
            "phase12.PHASE_12_CLOSING"
        ),
        "condition_1": "FALSE -- 0 of 5 seeds exclude zero",
        "condition_2": "FALSE at 0.36x -- delta 0.0106 against 0.0296",
        "run_dirs": ("p12-paired", "p12-arm-c-concat", "p12-arm-d-capacity"),
        "date": "2026-08-23",
        "caveats": (
            "D itself is the measured capacity null: doubling the head "
            "on the SAME information moved A 0.2505 to 0.2534, within "
            "noise -- the control did its job, and what it controls for "
            "is exactly what this pair could not resolve",
            "the staging-geometry caveat travels on C: r(AR, mean) = "
            "+0.1601 (phase12.BASAL_CONFOUND_OBSERVED)",
            "this row and its two siblings travel together: the three "
            "contrasts are one design (A/B/C/D, one factor), and quoting "
            "any one alone would drop the structure that makes it "
            "readable",
        ),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-29, Phase 16's primary contrast ----------------
    {
        "id": "p16-anchor-loop-unresolved",
        "claim": (
            "the anchor loop vs the best probe is UNRESOLVED: loop mean "
            "0.2040 (sd 0.0345) against probe 0.2520 (sd 0.0148), mean "
            "paired delta -0.0481. Condition 1 FALSE (0 of 5 seed BCa "
            "intervals exclude zero, directions mixed -- seed 7 at "
            "+0.0085); condition 2 TRUE (|-0.0481| > 0.0329, 1.46x). "
            "The point estimate runs AGAINST the loop and the cohort "
            "cannot resolve it"
        ),
        "status": "UNRESOLVED-WITHDRAWN",
        "framing": "additional",
        "phase": "p16",
        "record": (
            "phase16.PHASE_16_CLOSING; phase16.REPAIR_WITHOUT_TRANSFER; "
            "phase15.ANCHOR_LOOP_REGISTERED"
        ),
        "condition_1": "FALSE -- 0 of 5 seeds exclude zero, directions mixed",
        "condition_2": "TRUE at 1.46x -- delta -0.0481 against 0.0329",
        "run_dirs": ["p16-anchor-loop"],
        "date": "2026-08-29",
        "caveats": (
            "**the first case in this ledger of condition 2 passing "
            "while condition 1 fails** -- a mean delta larger than the "
            "combined-means band whose per-seed paired intervals all "
            "span zero, directions mixed. The clearest demonstration "
            "yet of why BOTH conditions are required: either alone "
            "would have called this resolved, in opposite directions",
            "'parity' is the CRITERION'S verdict, not a claim the two "
            "are equal (the framing adopted): the registered success "
            "condition -- parity-with-explanations -- is met on its "
            "own terms, the mismatch records exist for all 237, AND "
            "the negative point estimate is not smoothed away",
            "the pre-committed null reading attaches: the FIFTH "
            "convergent measurement that arm A's feature-space "
            "neighbourhoods do not carry clinical grade, exactly as "
            "COHORT_CANNOT_RESOLVE predicted "
            "(phase16.SELF_CONSISTENCY_READINGS registered both ways)",
            "anchor grades are trusted single grades from the survey "
            "lineage, not verified-unanimous; one backbone, frozen "
            "space; the ImageNet anchor rides",
        ),
        "corrects": None,
    },
    # ---- APPENDED 2026-08-30, Phase 17's five-contrast family ------------
    {
        "id": "p17-a-vs-probe",
        "claim": (
            "TSTR arm A vs the best probe is UNRESOLVED: A 0.2334 (sd "
            "0.0044) against probe 0.2520 (sd 0.0148), mean paired delta "
            "-0.0186. Condition 1 FALSE (0 of 5 intervals exclude zero); "
            "condition 2 TRUE (|-0.0186| > 0.0136, 1.37x)"
        ),
        "status": "UNRESOLVED-WITHDRAWN",
        "framing": "additional",
        "phase": "p17",
        "record": "phase17.PHASE_17_CLOSING; phase17.READINGS_COMMITTED",
        "condition_1": "FALSE -- 0 of 5 seeds exclude zero",
        "condition_2": "TRUE at 1.37x -- delta -0.0186 against 0.0136",
        "run_dirs": ["p17-family-analysis"],
        "date": "2026-08-30",
        "caveats": (
            "**the SECOND condition-2-pass/condition-1-fail instance in "
            "this ledger** (after p16-anchor-loop-unresolved) -- the "
            "pattern that demonstrates why both conditions are required "
            "now has a pair",
            "**A's sd 0.0044 is ~3.4x tighter than the probe's**: the "
            "22k-synthetic-image training regime nearly eliminates seed "
            "variance, which SHRINKS the threshold -- the tiny 0.0136 is "
            "a property of the arms' variances, not a strengthened claim",
            "zero real patient images or labels in training; all 237 "
            "pure test; the parked limitations travel (no scar, "
            "magnitude-to-grade assumption, anchored philtrum)",
        ),
        "corrects": None,
    },
    {
        "id": "p17-b-vs-probe",
        "claim": (
            "TSTR arm B vs the best probe is CLAIMABLE, NEGATIVE: B "
            "-0.0044 (sd 0.0317) against probe 0.2520, mean paired delta "
            "-0.2558; 5 of 5 intervals exclude zero in one direction and "
            "the delta is 8.3x the 0.0307 threshold"
        ),
        "status": "CLAIMABLE",
        "framing": "additional",
        "phase": "p17",
        "record": "phase17.PHASE_17_CLOSING; phase17.READINGS_COMMITTED",
        "condition_1": "TRUE -- 5 of 5, one direction",
        "condition_2": "TRUE at 8.3x -- delta -0.2558 against 0.0307",
        "run_dirs": ["p17-family-analysis"],
        "date": "2026-08-30",
        "caveats": (
            "**the committed B-null reading attaches verbatim**: the "
            "method that reached 0.31 on CARS lip symmetry does not "
            "transfer to the composite panel label -- consistent BOTH "
            "with the label being more than symmetry AND with the two "
            "declared adaptations costing signal, unattributable "
            "between them and stated so "
            "(phase17.READINGS_COMMITTED['arm_b_if_null'])",
            "never 'we replicated Rosero': method replication with two "
            "declared adaptations, on a harder label",
        ),
        "corrects": None,
    },
    {
        "id": "p17-c-vs-probe",
        "claim": (
            "TSTR arm C vs the best probe is UNRESOLVED: C -0.0185 (sd "
            "0.0586) against probe 0.2520, mean paired delta -0.2705; "
            "4 of 5 intervals exclude zero -- seed 1337's reaches "
            "+0.0016 -- so condition 1 fails on one seed"
        ),
        "status": "UNRESOLVED-WITHDRAWN",
        "framing": "additional",
        "phase": "p17",
        "record": "phase17.PHASE_17_CLOSING; phase17.READINGS_COMMITTED",
        "condition_1": "FALSE -- 4 of 5; seed 1337's interval spans zero",
        "condition_2": "TRUE at 5.1x -- delta -0.2705 against 0.0530",
        "run_dirs": ["p17-family-analysis"],
        "date": "2026-08-30",
        "caveats": (
            "the committed C-null reading attaches: consistent with A's "
            "parked prediction extending to the training scheme "
            "(phase17.READINGS_COMMITTED['arm_c_if_null']) -- though the "
            "family's observed pattern complicates that reading "
            "(phase17.UNPREDICTED_PATTERN)",
        ),
        "corrects": None,
    },
    {
        "id": "p17-a-vs-c",
        "claim": (
            "the A-C secondary (training scheme on shared synthesis) is "
            "UNRESOLVED: mean paired delta +0.2518, 4 of 5 intervals "
            "exclude zero -- the same seed-1337 C outlier at 0.0803 "
            "breaks condition 1"
        ),
        "status": "UNRESOLVED-WITHDRAWN",
        "framing": "additional",
        "phase": "p17",
        "record": "phase17.PHASE_17_CLOSING; phase17.UNPREDICTED_PATTERN",
        "condition_1": "FALSE -- 4 of 5; seed 1337",
        "condition_2": "TRUE at 4.9x -- delta +0.2518 against 0.0515",
        "run_dirs": ["p17-family-analysis"],
        "date": "2026-08-30",
        "caveats": (
            "**the scheme attribution is NOT claimed**: a delta 18x the "
            "Phase 16 effect size still failed condition 1 -- the "
            "margin-table pattern continuing, and the reason the "
            "family's most interesting axis stays an observation "
            "(phase17.UNPREDICTED_PATTERN) rather than a claim",
        ),
        "corrects": None,
    },
    {
        "id": "p17-b-vs-c",
        "claim": (
            "the B-C secondary (transformation family on shared "
            "training) is UNRESOLVED: mean paired delta +0.0147, 2 of 5 "
            "intervals exclude zero, directions MIXED -- the only "
            "contrast in the family failing BOTH conditions. The "
            "transformation family shows nothing"
        ),
        "status": "UNRESOLVED-WITHDRAWN",
        "framing": "additional",
        "phase": "p17",
        "record": "phase17.PHASE_17_CLOSING",
        "condition_1": "FALSE -- 2 of 5, directions mixed",
        "condition_2": "FALSE -- delta +0.0147 against 0.0584",
        "run_dirs": ["p17-family-analysis"],
        "date": "2026-08-30",
        "caveats": (
            "piecewise-affine vs TPS, byte-identical training and "
            "readout -- the axis B-C isolates is the one axis the "
            "family can say did not matter here",
        ),
        "corrects": None,
    },
    # ---------------------------------------------------------------
    # [2026-08-31] THE LEDGER'S FIRST CORRECTION, and its first-ever use
    # of the `corrects` field. The mechanism has existed since the
    # ledger was born at 17 entries and has never been exercised until
    # now; validate() has always enforced that a correction targets an
    # EARLIER entry, and that rule is what this entry runs through.
    #
    # It corrects a claim about the LEDGER'S OWN HISTORY -- not a
    # measurement. Both corrected rows' figures stand untouched; only
    # their sentences about precedence were wrong.
    # ---------------------------------------------------------------
    {
        "id": "ledger-condition-split-count-corrected",
        "claim": (
            "the precedence claims on p16-anchor-loop-unresolved ('the "
            "first case in this ledger of condition 2 passing while "
            "condition 1 fails') and p17-a-vs-probe ('the SECOND "
            "condition-2-pass/condition-1-fail instance in this "
            "ledger') are WRONG AS WRITTEN. Derived from the ledger's "
            "own condition fields, EIGHT rows have condition 1 FALSE "
            "and condition 2 TRUE -- indices 25, 26, 27, 28, 31, 32, "
            "34, 35. p16 (index 31) is the FIFTH, not the first; "
            "p17-a-vs-probe (index 32) is the SIXTH, not the second. "
            "No measurement changes: every delta, threshold, margin and "
            "verdict on both rows stands exactly as computed"
        ),
        "status": "DESCRIPTIVE",
        "framing": "main",
        "phase": "p16",
        "record": (
            "ladder.SMALLEST_RESOLVABLE_DIFFERENCE (the sibling "
            "correction of the same session); phase16.PHASE_16_CLOSING; "
            "phase17.PHASE_17_CLOSING"
        ),
        "condition_1": None,
        "condition_2": None,
        "run_dirs": [],
        "date": "2026-08-31",
        "caveats": (
            # -- the eight rows, reproduced FROM THE LEDGER'S OWN FIELDS
            # (index | id | date | condition 1 | condition 2), not
            # retyped from any document. A test re-derives this tuple
            # from ENTRIES and asserts equality, so it cannot drift.
            "THE EIGHT ROWS, from the ledger's own condition fields: "
            "25 | p10-replication-vs-best-arm-withdrawn | 2026-08-17 | "
            "condition 1 3 of 5 seeds exclude zero in the paired BCa "
            "over patients | condition 2 3.76x || "
            "26 | p11-iem-loss-costs-pcc-withdrawn | 2026-08-17 | "
            "condition 1 3 of 5 seeds exclude zero | condition 2 2.57x || "
            "27 | p11-iem-loss-improves-iem-withdrawn | 2026-08-17 | "
            "condition 1 2 of 5 seeds exclude zero | condition 2 2.68x || "
            "28 | p12-basal-vs-frontal-bar-withdrawn | 2026-08-23 | "
            "condition 1 0 of 5 seeds exclude zero | condition 2 1.8x || "
            "31 | p16-anchor-loop-unresolved | 2026-08-29 | condition 1 "
            "0 of 5 seeds exclude zero, directions mixed | condition 2 "
            "1.46x || "
            "32 | p17-a-vs-probe | 2026-08-30 | condition 1 0 of 5 "
            "seeds exclude zero | condition 2 1.37x || "
            "34 | p17-c-vs-probe | 2026-08-30 | condition 1 4 of 5; "
            "seed 1337's interval spans zero | condition 2 5.1x || "
            "35 | p17-a-vs-c | 2026-08-30 | condition 1 4 of 5; seed "
            "1337 | condition 2 4.9x",

            # -- what IS true of p16, preserved because it was measured
            "**p16 IS verifiably the first of a NARROWER kind, and that "
            "is preserved rather than lost**: the first row where ALL "
            "per-seed intervals span zero AND directions are mixed. "
            "Index 28 (p12-basal) is the only earlier row with all "
            "intervals spanning zero, and its directions are NOT mixed "
            "-- phase12.PAIRED_OBSERVED['verdicts']['b_vs_a'] records a "
            "single direction 'A ahead', re-derived from the arms' "
            "per-seed means as +0.0507 / +0.0321 / +0.0326 / +0.1121 / "
            "+0.0849, all one sign. That narrower claim is true and "
            "measured; it is simply not the claim the row's bolded "
            "sentence makes",

            # -- p17 fails under BOTH readings
            "**p17-a-vs-probe fails under the narrow reading too, and "
            "not merely by omission**: its condition-1 field records "
            "'0 of 5 seeds exclude zero' with NO direction statement, "
            "and no phase-17 record states its per-seed sign pattern. "
            "Its membership in the narrower kind is therefore "
            "UNESTABLISHED, not just unclaimed -- so no reading rescues "
            "'the SECOND'",

            # -- the ruling and its grounds
            "**THE RULING, 2026-08-31**: the DESCRIPTIVE reading "
            "of p16's em-dash clause governs -- the bolded head "
            "sentence is the claim, the appositive describes that "
            "instance rather than qualifying the claim. Grounds "
            "recorded with it: (i) p17 restates the property with the "
            "qualifier DROPPED, which is how the author read it a day "
            "later; (ii) a record that requires an appositive to be "
            "load-bearing will mislead a reader who does not parse it "
            "that way. Under this reading both claims are wrong as "
            "written",

            # -- the generalisable lesson
            "**THE MECHANISM, and it is the generalisable part**: the "
            "error was a claim about the LEDGER'S OWN HISTORY written "
            "into a row AT APPEND TIME, when the ledger itself could "
            "have been queried. A precedence claim ('the first', 'the "
            "Nth') is derivable from the fields of the rows already "
            "present; asserting one from memory while appending is "
            "writing a claim next to the data that refutes it",

            "NO MEASUREMENT CHANGES. Every delta, threshold, margin, "
            "condition and verdict on p16-anchor-loop-unresolved and "
            "p17-a-vs-probe stands exactly as computed; the rows are "
            "append-only and untouched. What is corrected is two "
            "sentences about precedence",

            "**THE LEDGER'S FIRST USE OF `corrects`.** The field has "
            "existed since the born population of 17 and had never been "
            "exercised; validate() has always required a correction to "
            "target an EARLIER entry, and this entry is the first to "
            "run through that rule. A second target -- p17-a-vs-probe "
            "-- is named in the claim rather than in the field, which "
            "holds one id",
        ),
        "corrects": "p16-anchor-loop-unresolved",
    },
    # ---- APPENDED 2026-09-06, after Phase 27 closed --------------------
    {
        "id": "p27-anchor-train-descriptive",
        "claim": (
            "a linear head fit on the 25 Deall anchor GRADES and "
            "evaluated on all 237 as pure test scores PCC "
            "0.18270409137534543 at the declared primary epoch, against "
            "the probe's 0.2520 -- a difference of 0.0693, INSIDE the "
            "0.04-0.10 band this cohort cannot resolve, and far below "
            "the 0.1386 it has demonstrably resolved once. The "
            "correlation itself CLEARS the n=237 threshold 0.1281, so "
            "it is distinguishable from zero and not from the probe. "
            "The registered degenerate signature did NOT fire: "
            "prediction sd 0.8727 against the cohort truth's 0.6587, "
            "and prediction mean 2.2108 rather than the anchor grade "
            "mean 2.96 the head started on. Seed sd 0.0 at five seeds, "
            "deterministic by construction. **The first time anything "
            "in this record was FIT to the anchor Score**"
        ),
        "status": "DESCRIPTIVE",
        "framing": "additional",
        "phase": "p27",
        "record": (
            "phase27.PHASE_27_CLOSING; phase27.THE_RESULT_OBSERVED; "
            "phase27.THE_LEDGER_DISPOSITION_RULED"
        ),
        # Both None, as ruled. Condition 2 IS computable and passes at
        # 5.34x, and is recorded at phase27.THE_CRITERION_NOT_RESOLVED
        # rather than here, because a row carrying one condition and not
        # the other reads as a contrast that was run under the criterion.
        # It was not: condition 1 is UNCOMPUTED, which is neither TRUE
        # nor FALSE and must not be written as either.
        "condition_1": None,
        "condition_2": None,
        "run_dirs": ("p27_anchor_train__0fe38121__p27-anchor-train",),
        "date": "2026-09-06",
        "caveats": (
            "anchor grades are trusted single grades from the survey "
            "lineage, not verified-unanimous (supervision ask 7, open "
            "since 2026-08-16) -- travels with every quotation",
            "**the two targets are a DIFFERENT QUANTITY on scale and "
            "construction**: trained on an integer 1-5 Score, evaluated "
            "against a 0.2-grid panel mean. Only PCC and Spearman cross "
            "cleanly. RMSE 1.1312, MAE 0.8981 and 3-class accuracy "
            "0.4388 carry a MEASURED offset of -0.5436 and are NOT "
            "comparable to any cohort-trained arm's "
            "(phase27.THE_VALUE_METRICS_UNDER_THE_CROSSING)",
            "**condition 1 is UNCOMPUTED, not FALSE.** The per-seed "
            "paired BCa over the 237 needs the CLUSTER-ONLY prediction "
            "CSVs and no re-run. Until it exists the contrast is "
            "unresolved and this row claims a descriptive ordering only "
            "(phase27.THE_CRITERION_NOT_RESOLVED)",
            "**a positive result here is registered UNINTERPRETABLE in "
            "advance**: nobody knows how the 25 grades were made, so the "
            "record cannot attribute a correlation above the zero "
            "threshold to label quality (phase27.THE_ASYMMETRY_"
            "REGISTERED). It licenses NO sentence about cleaner labels",
            "the declared target offset +0.2056 was CORRECTED by "
            "measurement to -0.5436, opposite in sign: the first "
            "measurement that the anchor images and the cohort images "
            "occupy different regions of the frozen space "
            "(phase27.THE_OFFSET_IS_CORRECTED)",
            "one backbone, one frozen space, one geometry; the ImageNet "
            "anchor rides",
        ),
        "corrects": None,
    },
)


def validate(entries: tuple = ENTRIES) -> None:
    """Structural integrity: unique ids, known statuses and framings,
    every correction targeting an EARLIER entry. Raises on the first
    violation; the suite calls this on the shipped ledger."""
    seen: set = set()
    for position, entry in enumerate(entries):
        if entry["id"] in seen:
            raise ValueError(f"duplicate ledger id {entry['id']!r}")
        seen.add(entry["id"])
        if entry["status"] not in STATUSES:
            raise ValueError(
                f"{entry['id']}: unknown status {entry['status']!r}"
            )
        if entry["framing"] not in FRAMINGS:
            raise ValueError(
                f"{entry['id']}: unknown framing {entry['framing']!r}"
            )
        target = entry.get("corrects")
        if target is not None:
            earlier = {e["id"] for e in entries[:position]}
            if target not in earlier:
                raise ValueError(
                    f"{entry['id']}: corrects {target!r}, which is not an "
                    "earlier entry -- corrections append, they never reach "
                    "forward"
                )


def cumulative_checksum(n: int | None = None) -> str:
    """SHA-256 over the canonical JSON of the first ``n`` entries.

    The suite pins this for the born population: appending entries leaves
    every earlier prefix checksum unchanged, while rewriting any existing
    entry breaks the pin -- the append-only rule with teeth."""
    subset = ENTRIES if n is None else ENTRIES[:n]
    canonical = json.dumps(
        [dict(sorted(entry.items())) for entry in subset],
        sort_keys=True, ensure_ascii=True, separators=(",", ":"),
        default=list,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()
