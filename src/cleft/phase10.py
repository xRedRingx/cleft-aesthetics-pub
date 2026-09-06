"""Phase 10: the CleftGNN replication under this project's criterion.

Opened 2026-08-16 (PLAN_AMENDMENT_2026-08-13 section 5), with three notes
carried from the manuscript read (``HANDOFF_PHASE10_NOTES_2026-08-16.md``,
outside the repository)
recorded before anything is built, the registered free first step -- the
paper-vs-code ROI check -- executed the same turn, and the phase's
registered silences listed t-SNE-style rather than filled. Nothing else
is built this turn.
"""

from __future__ import annotations

#: **[RECORDED 2026-08-16, from the tracked-changes manuscript read -- the
#: comparator correction, dated] CleftGNN's published numbers are TWO
#: TABLES, two quantities, and neither is citable without the other.**
#:
#: The amendment's section-5 sentence ("rater-specific PCCs run 0.440 to
#: -0.162, mean ~0.14") describes the **Study Test Set only** (Table 1,
#: n=28, each rater's own labels). The manuscript's second table, which
#: the record did not carry until now: **Benchmark Test Set** (Table 5,
#: n=25 -- the 25 of 76 highest-inter-rater-agreement images from a
#: separate study of 27 cleft surgeons, single consensus label; the SAME
#: 25 images as this project's ``deall_set``). CleftGNN's five
#: rater-trained models there: -0.273, 0.323, -0.092, -0.121, 0.598
#: (mean ~0.087), and the paper's own text notes its summed PCC score
#: trails PMRC on that table.
#:
#: **On the 0.598 cell**: single cell, single run, single split, no
#: interval anywhere in the paper. Fisher 95% CI at n=25, RECOMPUTED at
#: recording time (arctanh, se = 1/sqrt(22)): **[0.2656, 0.8033]** --
#: before selection effects, and it is the maximum of 15 table cells
#: (3 backbones x 5 raters) after backbone and module selection. The
#: same architecture on the same benchmark scores -0.273 when trained on
#: Rater A -- a spread that is the signature of sampling noise at n=25.
#: Plausibility bound: this cohort's mean inter-rater r is 0.4696, so a
#: model at 0.598 on one rater would predict that rater better than the
#: other four clinicians do.
#:
#: **Quantity-distinct from this project's 0.2520**: panel mean vs
#: consensus class; 237 all-out-of-fold vs 25 single-split; five seeds
#: with paired BCa and both conditions vs one run with no interval;
#: pre-registered selection (29/30 comparisons withdrawn on the main
#: line) vs best-of-15 cells. And this project's own external-reference
#: **0.2512 on the same 25 images** (Phase 9, ledger; never pooled) now
#: sits beside that table -- a different quantity again, and recorded as
#: such.
CLEFTGNN_COMPARATOR_TABLES = {
    "recorded": "2026-08-16, from the tracked-changes manuscript",
    "corrects": (
        "the amendment section-5 range was the Study Test Set ONLY; the "
        "Benchmark Test Set was missing from the record"
    ),
    "study_test_set": {
        "table": "Table 1", "n": 28,
        "labels": "each rater's own",
        "range": "0.440 to -0.162, mean ~0.14",
    },
    "benchmark_test_set": {
        "table": "Table 5", "n": 25,
        "labels": (
            "single consensus; the 25-of-76 highest-agreement images "
            "from a separate 27-surgeon study -- the SAME 25 images as "
            "deall_set"
        ),
        "per_rater_model": (-0.273, 0.323, -0.092, -0.121, 0.598),
        "mean": "~0.087",
        "papers_own_text": "CleftGNN's summed PCC score trails PMRC there",
    },
    "the_598_cell": {
        "fisher_ci_95_recomputed": (0.2656, 0.8033),
        "selection": "maximum of 15 cells after backbone+module selection",
        "same_architecture_same_benchmark": "-0.273 trained on Rater A",
        "plausibility": (
            "cohort mean inter-rater r is 0.4696; 0.598 would out-predict "
            "the other four clinicians"
        ),
        "quantity_distinct_from_0_2520": (
            "panel mean / 237 out-of-fold / 5 seeds paired BCa both "
            "conditions / pre-registered -- vs consensus class / 25 "
            "single-split / one run no interval / best-of-15"
        ),
    },
    "our_number_beside_it": (
        "0.2512 on the same 25 images (Phase 9 external reference, "
        "ledger) -- a different quantity, never pooled"
    ),
    "rule": "neither table is citable as 'CleftGNN's performance' without the other",
}


#: **[MEASURED 2026-08-16, the registered free first step, executed the
#: turn the phase opened] THE PAPER-VS-CODE ROI CHECK: the sets do NOT
#: differ -- the notebook's grid scheme IS a sliding-window enumeration
#: under exactly one parameterization, and the manuscript's text
#: UNDERDETERMINES that parameterization.**
#:
#: The manuscript (read directly from the tracked-changes docx, not from
#: a paraphrase): "These regions are extracted from the high-level
#: convolutional feature map using a sliding window approach with varying
#: aspect ratios and scales" -- no window sizes, no stride, no region
#: count, anywhere in the passage; N is left generic.
#:
#: The code (``srgnn.grid_rois``, the notebook scheme, already ported and
#: verified): 26 boxes on the 42-map -- every rectangle on the 3x3-cell
#: lattice with at least one extent >= 2 cells, whole image excluded then
#: appended as region 27. Composition by (width, height) in cells:
#: (1,2):6, (1,3):3, (2,1):6, (2,2):4, (2,3):2, (3,1):3, (3,2):2.
#:
#: **The identity, verified by enumeration rather than argued**: a
#: sliding window with window sizes {1,2,3} x {1,2,3} cells EXCLUDING
#: 1x1, stride one cell (14 px), plus the whole map, reproduces the
#: code's set EXACTLY (set equality, measured True). So there is no
#: contradiction between paper and code -- "varying aspect ratios and
#: scales" is satisfied -- but the paper's sentence admits infinitely
#: many region sets, of which the shipped code is one.
#:
#: **Consequences**: (1) the replication builds THE CODE'S SCHEME --
#: ``grid_rois``, already ported, verified, laptop-tested -- the notebook
#: is ground truth; (2) for supervision, worth reporting either way as promised:
#: the manuscript revision can pin section 2.10 to the code with one
#: sentence ("window sizes of one to three grid cells per dimension,
#: excluding 1x1, stride one cell, plus the whole map -- 27 regions").
ROI_CHECK_MEASURED = {
    "measured": "2026-08-16, the phase's registered free first step",
    "manuscript_text": (
        "'a sliding window approach with varying aspect ratios and "
        "scales' -- no sizes, no stride, no count given"
    ),
    "code_set": {
        "boxes": 26,
        "plus": "the whole image, appended as region 27",
        "composition_by_cells": {
            "1x2": 6, "1x3": 3, "2x1": 6, "2x2": 4, "2x3": 2,
            "3x1": 3, "3x2": 2,
        },
        "rule": "all lattice rectangles with at least one extent >= 2 cells",
    },
    "identity": (
        "sliding window, sizes {1..3}x{1..3} minus 1x1, stride one cell, "
        "reproduces the code set EXACTLY -- set equality measured True"
    ),
    "verdict": (
        "no set contradiction; the text UNDERDETERMINES the "
        "parameterization; the code's set is the buildable ground truth"
    ),
    "replication_builds": "srgnn.grid_rois -- the notebook scheme, 26+1",
    "for_supervisor": (
        "one sentence pins section 2.10 to the code: window sizes one to "
        "three cells per dimension excluding 1x1, stride one cell, plus "
        "the whole map -- 27 regions"
    ),
    # **[CORRECTED 2026-08-16, same day, on reading the notebook's OWN
    # generator]** The check above compared the manuscript against
    # ``grid_rois`` while ATTRIBUTING grid_rois to the notebook -- the
    # amendment's claim, taken on trust. The notebook's actual
    # ``generate_ran_rois`` has NO minimum-size filter: it enumerates ALL
    # spans of the 3x3 grid INCLUDING the nine 1x1 cells, appends the
    # whole image, and de-duplicates -- measured: 37 raw -> 36 unique
    # (3*(224/3) == 224.0 exactly in IEEE double, so the appended whole
    # deduplicates cleanly; no float accident). The class's
    # ``region_count=27`` is only a DEFAULT, self-adjusted to 36 at
    # runtime with a printed warning. So: grid_rois (26+1) is the SR-GNN
    # PAPER's rule (Sec III-B, min extent 2), NOT the notebook's; the
    # notebook the group ran uses 36 regions. The sliding-window identity
    # above still holds for grid_rois; the notebook's 36 is ALSO a
    # sliding-window family -- the COMPLETE one, sizes {1..3}x{1..3},
    # stride one cell, nothing excluded. NOTEBOOK_ROI_SET_MEASURED is the
    # first finding's corrected form.
    "corrected": (
        "2026-08-16: grid_rois is the SR-GNN paper's rule, not the "
        "notebook's; the notebook's own generator yields 36 (all spans "
        "incl. 1x1) -- NOTEBOOK_ROI_SET_MEASURED"
    ),
}


#: **[RECORDED 2026-08-16, FOR PHASE 11 -- record only, no action] IEM,
#: the manuscript's equation (16) verbatim**, so the Phase 11 pre-step
#: (re-scoring every existing arm's on-disk predictions BEFORE any loss
#: build) has its constants from the record rather than from memory:
#:
#: - optimistic (underestimating severity, y_hat - G < 0):
#:   IEM = 1.2 * |y_hat - G| ** 1.12
#: - pessimistic (overestimating severity, y_hat - G >= 0):
#:   IEM = 0.8 * |y_hat - G| ** 0.87
#: - bounds noted in the paper: [0, 4]; the framing: optimistic errors
#:   are clinically more dangerous and are penalised more heavily.
#:
#: Also available from the same manuscript if Phase 11 wants them: FGCM
#: (eqs. 14-15, softmax-weighted expected class vs consensus) and NDCG@K
#: (eqs. 17-18, relevance = max(0, 4 - |i - G|)).
IEM_CARRIED_FOR_PHASE_11 = {
    "recorded": "2026-08-16, record only, no action",
    "equation": "manuscript eq. (16), verbatim constants",
    "optimistic": {"condition": "y_hat - G < 0", "form": "1.2 * |d|**1.12"},
    "pessimistic": {"condition": "y_hat - G >= 0", "form": "0.8 * |d|**0.87"},
    "bounds": (0, 4),
    "pre_step": (
        "re-score every existing arm's on-disk predictions with IEM "
        "before any loss build -- does the ranking move at all?"
    ),
    "also_available": (
        "FGCM (eqs. 14-15); NDCG@K (eqs. 17-18, rel = max(0, 4-|i-G|))"
    ),
}


#: **[RESTATED 2026-08-16 from the amendment, silences LISTED not filled
#: -- the t-SNE discipline] Phase 10's registered scope, and what the
#: record does not fix.**
#:
#: **What the amendment commits** (section 5, restated): build from the
#: architecture in both manuscript drafts; the APPNP configuration
#: (alpha=0.3, K=1) is IDENTICAL to this project's verified SR-GNN port;
#: region generation resolved by the notebook (now measured:
#: ROI_CHECK_MEASURED). The NEW code is the Spatial Attention Branch --
#: the manuscript's stage 3, "vertical mask-guided aggregation", a
#: vertical-position weighting over region features -- plus their crop
#: scheme; everything else is assembly of verified parts. Training:
#: pretraining through the existing Phase 6 pipeline, a ladder through
#: the existing Phase 7 one; the amendment's estimate is DAYS. The run
#: here is under THIS project's criterion: 237 patients, the panel mean
#: (the standing primary label), five seeds, paired BCa, both conditions
#: -- "a genuine contribution, and the kind a viva rewards."
#:
#: **Comparison target**: the amendment names the LEDGER ("10 needs 9's
#: ledger to compare against") and the published tables
#: (CLEFTGNN_COMPARATOR_TABLES); the natural paired contrast is the
#: 0.2520 arm (ladder.BEST_ARM) -- evident, but its registration as the
#: named contrast belongs to the phase's registration turn, so it is
#: listed below rather than assumed.
#:
#: **Registered silences -- the build stops on these until they are
#: resolved:**
PHASE_10_SCOPE_AND_SILENCES = {
    "restated": "2026-08-16, from PLAN_AMENDMENT_2026-08-13 section 5",
    "commits": {
        "from_existing": (
            "architecture from both drafts; APPNP alpha=0.3 K=1 == the "
            "verified SR-GNN port; regions == the notebook scheme "
            "(ROI_CHECK_MEASURED: grid_rois, 26+1)"
        ),
        "new_code": (
            "the Spatial Attention Branch (stage 3, vertical mask-guided "
            "aggregation) plus their crop scheme"
        ),
        "training": (
            "pretraining through Phase 6's pipeline; ladder through "
            "Phase 7's; estimate DAYS"
        ),
        "criterion": (
            "237 patients, panel mean, five seeds, paired BCa, both "
            "conditions -- this project's, unchanged"
        ),
    },
    "silences": (
        "the SABM's exact form -- the vertical masks' number, shape and "
        "parameterization live in the manuscript's equations, not in any "
        "repo record; needs registering from the manuscript before code",
        "'their crop scheme' -- named by the amendment, never specified "
        "in the repo",
        "the Phase 6 pretraining cell for the replication (SCUT original "
        "vs masked; which geometry)",
        "the ladder placement: geometry, init axis, which arms run",
        "rater-specific training (their five models) vs panel-mean only "
        "-- 'their architecture under our criterion' implies panel mean; "
        "whether rater-specific replicas also run is undecided",
        "the paired contrast's registration (vs the 0.2520 arm) -- "
        "evident, unregistered",
        "exit criteria -- none exist; they need writing before the "
        "phase's work starts (the Phase 9 lesson)",
    ),
}


#: **[MEASURED 2026-08-16 -- PHASE 10'S FIRST FINDING, corrected form,
#: FLAGGED FOR THE THURSDAY AGENDA] The paper, the notebook, and the
#: repo's port hold THREE different region-set answers.**
#:
#: 1. **Manuscript** (2.10, read from the docx): "a sliding window
#:    approach with varying aspect ratios and scales" -- no sizes, no
#:    stride, no count; N generic. UNPARAMETERIZED -- unimplementable
#:    from text.
#: 2. **The shared notebook** (``generate_ran_rois``, what the group
#:    ran): ALL spans of the 3x3 grid including the nine 1x1 cells, whole
#:    image appended, float de-dup -- **36 unique regions** (37 raw; the
#:    whole deduplicates exactly). The class's ``region_count=27``
#:    default is self-adjusted to 36 at runtime with a printed warning --
#:    the notebook contains BOTH numbers, and only 36 is what runs.
#: 3. **This repo's port** (``srgnn.grid_rois``): the SR-GNN paper's
#:    Sec III-B rule (min extent two cells) -- 26 + whole = **27**.
#:
#: Both code sets are sliding-window instantiations (the notebook's is
#: the complete size family; the port excludes 1x1), so neither
#: contradicts the manuscript text -- the text underdetermines both. The
#: amendment's "the notebook resolves how 27 regions arise" is REFUTED
#: by the notebook's own code: 27 was the class default, never the
#: generated set.
NOTEBOOK_ROI_SET_MEASURED = {
    "measured": "2026-08-16, from generate_ran_rois re-enumerated float-exact",
    "manuscript": "sliding window, unparameterized -- unimplementable from text",
    "notebook": {
        "regions": 36, "raw": 37,
        "rule": "ALL 3x3 spans incl. nine 1x1 cells + whole, de-duplicated",
        "region_count_27_default": (
            "self-adjusted to 36 at runtime with a printed warning"
        ),
        "float_check": "3*(224/3) == 224.0 exactly; the whole dedups cleanly",
    },
    "repo_port": {"regions": 27, "rule": "SR-GNN paper Sec III-B, min extent 2"},
    "amendment_claim_refuted": (
        "'the notebook resolves how 27 regions arise' -- 27 was the "
        "class default, not the generated set"
    ),
    "thursday_agenda": (
        "the paper's text cannot reproduce either set; the notebook and "
        "the SR-GNN paper rule disagree by the nine 1x1 cells; which set "
        "is CleftGNN's is a question only the group can answer"
    ),
    # [EXTENDED 2026-08-17, from the deck] Slide 3 says "Checked with 27
    # crops", and the notebook PASSES REGION_COUNT = 27 before its
    # generator self-adjusts to 36. The group's stated understanding is
    # 27; their code runs 36 -- belief-vs-execution, not just
    # paper-vs-code (CIFAR_CONFLATION_CONFIRMED_AT_SOURCE).
    "deck_also_says_27": (
        "2026-08-17: slide 3 'Checked with 27 crops', and the notebook "
        "passes REGION_COUNT = 27 -- so the group states 27 while their "
        "code runs 36: belief-vs-execution"
    ),
}


#: **[REGISTERED 2026-08-16] Phase 10's five decisions, the SABM's
#: published form, the exit criteria -- and the choices that BLOCK the
#: build, t-SNE-style, after the same-day measurements.**
#:
#: **(1) Region scheme -- DECIDED "the notebook's", PREMISE REFUTED the
#: same turn, RE-BLOCKED.** The decision's rationale ("the notebook is
#: what the group ran") was given believing notebook = 27; the notebook's
#: own generator yields 36 (``NOTEBOOK_ROI_SET_MEASURED``). The rationale
#: points at 36; the believed content was 27; the builder chooses
#: neither. Re-decision needed with the measured facts.
#:
#: **(2) Output reading -- BOTH, two cells, neither selected after the
#: fact.** Primary for the headline: the softmax-expected value (their
#: own eq. 14 form) -- continuous, like-for-like PCC against the panel
#: mean beside 0.2520. Top-1 class (their published PCC form, eq. 13)
#: reported beside. Committed now.
#:
#: **(3) Label -- train as they trained, evaluate as we evaluate.**
#: Cross-entropy on the discrete consensus grade; predictions evaluated
#: against the panel mean under the standing criterion. The
#: COMPARABILITY CAVEAT travels with every report of the number: the
#: training label differs from the evaluation label by design fidelity.
#: **Interpretation flagged**: no manifest column is named "consensus";
#: the five-grade consensus reading is the MODE column (majority of five,
#: ties toward the lower grade) -- recorded as this build's
#: interpretation, awaiting confirmation.
#:
#: **(4) Training regime -- their recipe, one deliberate divergence.**
#: Cross-entropy + SGD at lr 0.01 (the manuscript's stated recipe,
#: literal). The manuscript states NO epoch budget anywhere -- measured,
#: zero mentions in the docx -- so inner-val early stopping FILLS a
#: silence rather than contradicts a statement; it is still recorded as
#: the one deliberate divergence from any fixed-budget reading, with the
#: Phase-2-era measurement as its reason: fixed budgets measured
#: post-collapse endpoints on this cohort; the peaks sat at epochs 2-11
#: (the standing cleft-ladder early-stopping policy). Momentum, batch
#: size and any schedule are ALSO unstated in the paper -- silences
#: listed below, not filled. Everything else at this project's
#: standards: 5-fold OOF, five seeds, paired BCa against the 0.2520 arm,
#: both conditions.
#:
#: **(5) Backbone -- ResNet-50 ImageNet**, their best per Table 1,
#: one-factor faithful. Components built to the notebook where it
#: disambiguates the text, conflicts recorded per component:
#:
#: - **gated attention pooling**: notebook DISAMBIGUATES --
#:   sigmoid(W1 f) * (W2 f), summed over regions. Consistent with the
#:   text.
#: - **self-attention (ACM)**: notebook implements Q/K/V with tanh then
#:   softmax, per-region softmax weighting, sigmoid output.
#: - **SABM / vertical masks**: ABSENT from the notebook entirely -- the
#:   notebook is the SR-GNN(ViT) family member, not CleftGNN. The
#:   manuscript alone specifies it: eq (7) m_i proportional to
#:   VerticalCenter(r_i), lower regions weighted higher; eq (8)
#:   v_a = sum_i(F_att,i * m_i); eq (9) v_r = f_t + v_a, additive into
#:   the classifier. The notebook's own f_t + f_t*v (multiplicative ACM
#:   residual) is the OLD design -- the notebook cannot disambiguate a
#:   component it does not contain, so the manuscript's eq (9) governs.
#: - **backbone**: the notebook's is a FROZEN ViT-B/16 adapter -- a
#:   different family member; usable as a mechanics donor (roi_align
#:   semantics, pooling, ACM), never as the CleftGNN backbone.
#: - **crop scheme**: resolved by reading -- the manuscript's "crops"
#:   are the DATASET's standardised cropped frontal photographs, the
#:   same standard our staged pipeline already carries; no model code
#:   exists or is needed.
#:
#: **Exit criteria, registered before the work**: (1) the discrepancy
#: finding recorded (NOTEBOOK_ROI_SET_MEASURED -- done, and on the
#: Thursday agenda); (2) the replication built to the group's scheme
#: with deviations enumerated -- expect exactly one, early stopping;
#: (3) five seeds run; (4) paired BCa against 0.2520 delivered with
#: both output readings; (5) the result ledgered beside
#: CLEFTGNN_COMPARATOR_TABLES (both tables, both n's); (6) suite green.
#: Days-not-weeks stands.
PHASE_10_REGISTERED = {
    "registered": "2026-08-16",
    "region_scheme": {
        "decided": "the notebook's -- what the group ran",
        "premise_refuted_same_turn": (
            "the decision believed notebook = 27; the generator yields "
            "36 (NOTEBOOK_ROI_SET_MEASURED); re-decision needed -- the "
            "builder picks neither"
        ),
    },
    "output_readings": {
        "primary": "softmax-expected value (eq. 14 form) -- continuous, vs mean",
        "beside": "Top-1 class (eq. 13, the published PCC form)",
        "rule": "two cells, committed now, neither selected after the fact",
    },
    "label": {
        "train": "cross-entropy on the discrete consensus grade",
        "evaluate": "panel mean, standing criterion",
        "caveat": (
            "the training label differs from the evaluation label by "
            "design fidelity -- stated in every report of the number"
        ),
        "consensus_column_interpretation": (
            "the manifest's MODE (majority of five, ties toward lower) "
            "-- no column is named consensus; awaiting confirmation"
        ),
        # [REFUTED BY MEASUREMENT 2026-08-17] The mode reading is wrong and
        # so was the premise under it: the sheet has no consensus column at
        # all. The label is the sheet's MEDIAN column
        # (CONSENSUS_LABEL_IS_THE_MEDIAN). The line above stands as what was
        # believed on 2026-08-16.
        "consensus_column_corrected": (
            "2026-08-17: MODE REFUTED; the label is the sheet's Median "
            "column -- CONSENSUS_LABEL_IS_THE_MEDIAN"
        ),
    },
    "regime": {
        "recipe": "cross-entropy, SGD lr 0.01 -- the manuscript, literal",
        "epochs": (
            "the paper states NONE (measured, zero mentions); inner-val "
            "early stopping fills the silence -- the one deliberate "
            "divergence, reasoned by the Phase-2-era measurement (fixed "
            "budgets read post-collapse endpoints; peaks at epochs 2-11; "
            "the standing cleft-ladder early-stopping policy)"
        ),
        "ours": "5-fold OOF, five seeds, paired BCa vs 0.2520, both conditions",
    },
    "backbone": "resnet50, ImageNet init -- Table 1 best, one-factor faithful",
    "sabm_published_form": {
        "eq7": "m_i proportional to VerticalCenter(r_i), lower = larger",
        "eq8": "v_a = sum_i(F_att,i * m_i)",
        "eq9": "v_r = f_t + v_a, additive -- governs over the notebook's "
               "f_t + f_t*v, which is the OLD design",
    },
    "exit_criteria": (
        "discrepancy finding recorded (done, Thursday agenda)",
        "built to the group's scheme, deviations enumerated (expect one)",
        "five seeds run",
        "paired BCa vs 0.2520, both output readings",
        "ledgered beside CLEFTGNN_COMPARATOR_TABLES",
        "suite green",
    ),
    "estimate": "days, not weeks -- stands",
}


#: **[BLOCKED 2026-08-16, t-SNE-style -- the build stops here until these
#: are resolved; no config exists to generate and none is]**
PHASE_10_BUILD_BLOCKED_ON = {
    "blocked": "2026-08-16",
    "choices": (
        "the region set: notebook-36 (what ran, incl. nine 1x1 cells) vs "
        "port-27 (the SR-GNN paper rule) -- the decision's premise was "
        "refuted by measurement the same turn it was made",
        "eq (7)'s proportionality: m_i 'proportional to' "
        "VerticalCenter(r_i) with no constant or normalisation stated -- "
        "proposal awaiting sign-off: m_i = y_center_i / 224, unit-height "
        "normalised, larger downward, committed blind if accepted",
        "SGD momentum, batch size, any LR schedule -- unstated in the "
        "manuscript; literal reading is momentum 0, but literal-vs-"
        "customary is a maintainer decision",
        "the consensus column: MODE is this build's flagged "
        "interpretation, awaiting confirmation",
    ),
    "consequence": (
        "no cleftgnn model module, task, or config is built this turn -- "
        "building 36 would contradict what the maintainer believed, "
        "building 27 would contradict what the maintainer said; the "
        "scope-not-guess precedent (roadb.GRAPH_NODE_PATH_IS_NOT_WIRED) "
        "governs"
    ),
    # [RESOLVED 2026-08-16, same day -- PHASE_10_UNBLOCKED carries the
    # four answers; the build followed.]
    "resolved": "2026-08-16 -- see PHASE_10_UNBLOCKED",
}



#: **[UNBLOCKED 2026-08-16, the maintainer's four answers -- and the
#: correcting note for cc89d27, recorded here rather than by amending.]**
#:
#: **The correcting note first**: commit cc89d27's message says
#: "cleftgnn replication built to notebook"; that commit contains the
#: REGISTRATION and the BLOCK, not a build -- the build is THIS commit's.
#: Recorded so a git-log reader is not misled; history unamended.
#:
#: **(1) Region set: NOTEBOOK-36.** The "what the group ran" rationale
#: survives the measurement; the 27 belief does not. The ROI finding
#: stays Thursday-flagged.
#: **(2) m_i = y_center/224** -- accepted as proposed, committed blind.
#: **(3) Literal recipe**: momentum 0, plain SGD lr 0.01. The notebook
#: states NO batch size (measured: every 'batch' hit is a tensor-shape
#: variable), so the smallest defensible default is the project's
#: standing cleft batch size, 16, stated. The underdetermination is
#: recorded and the NO-MOMENTUM CAVEAT travels with any reported number.
#: **(4) MODE ties-lower -- VERIFIED, not adopted**: the score sheet
#: carries its own consensus column; the task asserts
#: computed-mode-equals-sheet-column on all 237 and, on ANY
#: disagreement, the SHEET'S column is the label. The sheet's column is
#: located by normalised header containing 'consensus' and the task
#: refuses zero or multiple matches.
PHASE_10_UNBLOCKED = {
    "unblocked": "2026-08-16",
    "cc89d27_note": (
        "its message ('built to notebook') overstates: that commit holds "
        "the registration and the block; the build is this commit's"
    ),
    "answers": {
        "region_set": "notebook-36; the rationale survives, the belief did not",
        "m_i": "y_center/224, committed blind",
        "recipe": (
            "literal: momentum 0, plain SGD lr 0.01; batch 16 -- the "
            "standing cleft default, the notebook states none; "
            "no-momentum caveat travels"
        ),
        "consensus": (
            "MODE ties-lower verified against the sheet's own consensus "
            "column on all 237; the sheet wins any disagreement"
        ),
        # [CORRECTED 2026-08-17] Both halves of that answer were wrong in
        # the same way: there is no consensus column to verify against, and
        # the mechanism is not the mode. The answer's SHAPE survives -- read
        # the label from the declared primary sheet, cross-check it against
        # the rater cells, let a disagreement be fatal -- and only the
        # column's name and the statistic change.
        "consensus_corrected": (
            "2026-08-17: the sheet's MEDIAN column is the label; the "
            "cross-check is the recomputed median of the five rater cells, "
            "disagreement fatal (CONSENSUS_LABEL_IS_THE_MEDIAN)"
        ),
    },
}


#: **[BUILT 2026-08-16, not launched] CleftGNN, the whole chain -- model,
#: harness backbone, task, schema, generator, config.**
#:
#: **The model** (``models/cleftgnn.py``): ResNet-50 (ImageNet, FULL
#: fine-tune) -> roi_align over the notebook's 36 regions (pool 7, GAP)
#: -> Linear 2048->512 -> MLP 512->1024->1024 -> APPNP K=1 alpha=0.3
#: over the complete graph (one step == the node mean, the port's
#: verified reading) -> sigmoid -> gated pooling sigmoid(W1 f)*(W2 f)
#: summed -> f_t; SABM: softmax(tanh(QK^T/sqrt(512)))*V -> F_att,
#: v_a = sum_i(m_i * F_att,i) with m_i = y_center/224, v_r = f_t + v_a
#: ADDITIVE (eq 9); Linear 1024->5.
#:
#: **The recipe** (``CleftGNNBackbone``, the frozen harness's protocol):
#: cross-entropy on the resolved consensus grade; plain SGD lr 0.01
#: momentum 0; batch 16; CALIBRATED CE INIT -- zero classifier weights,
#: bias = log train-fold class frequencies, so the epoch-0 expected
#: value equals the train-fold label mean (gate 3's policy translated
#: to CE). Early stopping through the frozen harness: inner_val_frac
#: 0.2, max_epochs 40, patience 5, monitor inner_val_mse -- monitored on
#: the EXPECTED-VALUE prediction against the CONSENSUS label, because
#: training never sees the mean (train-as-they-trained); registered
#: here, before any number.
#:
#: **The two readings** (registered cells, neither selected after the
#: fact): ``predict`` returns the softmax-EXPECTED grade -- the primary,
#: continuous, like-for-like against the panel mean; Top-1 rides beside
#: through the captured fold backbones. Per-seed OOF expected-value
#: files land in the baseline's exact CSV layout (``cluster_csv``), so
#: the standing paired-BCa tooling consumes them against the 0.2520
#: arm's; Top-1 grades land beside under their own name.
#:
#: **The caveats that travel with every number**: the training label
#: differs from the evaluation label by design fidelity; momentum 0 is
#: the literal reading of an underdetermined recipe; batch 16 is a
#: stated default the notebook does not carry.
CLEFTGNN_BUILT = {
    "built": "2026-08-16, not launched",
    "model": "models/cleftgnn.py -- provenance per component in its docstring",
    "regions": 36,
    "recipe": (
        "CE on resolved consensus; SGD lr 0.01 momentum 0; batch 16; "
        "calibrated CE init (bias = log class frequencies)"
    ),
    # [CORRECTED 2026-08-17] "resolved consensus" is the sheet's MEDIAN
    # column, read by scoresheet.load_median and verified against the five
    # rater cells row by row. No rounding step exists anywhere in the model.
    "label_corrected": (
        "2026-08-17: the sheet's Median column; no rounding step exists "
        "(CONSENSUS_LABEL_IS_THE_MEDIAN)"
    ),
    "regime_corrected": (
        "2026-08-17: the backbone is FROZEN (BACKBONE_IS_FROZEN) and the "
        "CE init smooths by Laplace (THIN_CLASS_SMOOTHING); the full "
        "fine-tune was an inference, not their recipe"
    ),
    "early_stopping": (
        "frozen harness: inner_val_frac 0.2, max_epochs 40, patience 5, "
        "monitor inner_val_mse on expected-value vs consensus -- the one "
        "registered deviation from any fixed-budget reading"
    ),
    "readings": {
        "primary": "softmax-expected grade, per-seed OOF CSVs in the "
                   "baseline layout for the paired BCa vs 0.2520",
        "beside": "Top-1 grades, own files",
    },
    "caveats_travel": (
        "train-label differs from eval-label by design fidelity",
        "momentum 0 is the literal reading",
        "batch 16 is a stated default; the notebook carries none",
    ),
}



#: **[MEASURED 2026-08-17, from the declared primary sheet] THE TRAINING
#: LABEL IS THE SHEET'S ``Median`` COLUMN. The mode interpretation is
#: REFUTED, and the premise under it -- that a "consensus" column exists
#: -- is refuted with it.**
#:
#: **What the sheet actually holds** (``${CLEFT_SCORESHEET}``, the
#: input this run already declares):
#: ``RanaPhotoID``, the five named rater columns, ``Average``, ``Median``.
#: There is no column named "consensus" anywhere in it. The DERIVED
#: files' consensus column IS this sheet's ``Median`` -- verified by the
#: maintainer on rows 241 (raters 2,3,2,1,3 -> 2), 243 (3,3,3,3,4 -> 3) and
#: 245 (3,3,2,4,4 -> 3), each matching APScores' eighth column.
#:
#: **The repo knew the layout the whole time, and the builder did not
#: read it.** ``data/scoresheet.py`` has carried ``ID_COLUMN =
#: "RanaPhotoID"``, the five ``RATERS`` and ``DERIVED_COLUMNS =
#: ("Average", "Median")`` since Phase 1, with a docstring line saying the
#: Average and Median columns exist and are not read. A new reader
#: searching for a "consensus" header was written beside a module that
#: already answered the question -- the R10 failure (read the existing
#: resolver before writing one) in its plainest form. The ad-hoc reader is
#: deleted; ``scoresheet.load_median`` is where this now lives.
#:
#: **The verification, not the trust.** That same docstring gives the
#: reason the column was ignored: "a precomputed column that has drifted
#: from the cells beside it disagrees silently." So the column is read AND
#: recomputed -- the median of the five rater cells, which for five
#: integers is the third of the sorted five and has no tie rule to get
#: wrong -- and ANY row where the two disagree raises. Two internal
#: stories about one row is a fault to stop on.
#:
#: **On "round-half-up of the mean": no record ever carried that
#: statement.** Searched: it appears in no record in this repository. What
#: did exist was a rounding STEP in the model's own code
#: (``np.round`` on incoming labels) -- and ``np.round`` is half-to-EVEN,
#: so it was not the half-up rule either. It is now gone: ``_as_grades``
#: asserts integrality and refuses anything else, because the median of
#: five integers is an integer and a rounding call could only ever have
#: hidden a label that arrived as something other than this one.
#:
#: **The label's standing**: one of the three raised at supervision approved targets, and its
#: learnability is measured -- ``labels.LEARNABILITY_237["median"] =
#: 0.5708``. Quote the 237 figure, never 0.5561: that is
#: ``LEARNABILITY_251``, a different population, and mixing the two is the
#: exact trap ``labels.py`` was written to prevent.
#:
#: **The design-fidelity consequence, which travels with the result**: the
#: median is measured LESS learnable than the mean (0.5708 against
#: 0.6022), and the ladder found the median claimably worse at masked/G2.
#: So a CleftGNN number below 0.2520 is partly the training target and not
#: only the architecture -- the comparability caveat now has a measured
#: size, not just a direction.
CONSENSUS_LABEL_IS_THE_MEDIAN = {
    "measured": "2026-08-17, from the declared primary sheet",
    "columns": (
        "RanaPhotoID, five named raters, Average, Median -- NO consensus "
        "column exists"
    ),
    "label": "the sheet's Median column",
    "derived_files": (
        "their consensus column IS this Median -- verified on rows 241 "
        "(2,3,2,1,3 -> 2), 243 (3,3,3,3,4 -> 3), 245 (3,3,2,4,4 -> 3) "
        "against APScores' eighth column"
    ),
    "mode_refuted": (
        "the 2026-08-16 MODE interpretation is refuted by measurement, "
        "and so is the premise that a consensus column exists"
    ),
    "round_half_up_statement": (
        "no record in this repository ever carried it; what existed was a "
        "rounding STEP in the model code (np.round, which is half-to-EVEN, "
        "not half-up) -- now replaced by an integrality assertion"
    ),
    "reader": (
        "scoresheet.load_median -- the module that already knew this "
        "sheet's identity; the ad-hoc reader beside it was the R10 failure "
        "and is deleted"
    ),
    "verification": (
        "the median of the five rater cells is recomputed for every row "
        "and any disagreement with the published column RAISES"
    ),
    "learnability_237": 0.5708,
    "learnability_population_trap": (
        "0.5708 is LEARNABILITY_237; 0.5561 is LEARNABILITY_251, a "
        "different population -- never quote them together"
    ),
    "design_fidelity_consequence": (
        "the median is measured less learnable than the mean (0.5708 vs "
        "0.6022) and the ladder found it claimably worse at masked/G2, so "
        "a result below 0.2520 is partly the training target and not only "
        "the architecture"
    ),
}


#: **[VOID 2026-08-17] The first CleftGNN launch, on this cause.** Run
#: ``p10_cleftgnn__9a28f8a8__p10-cleftgnn``: four failed attempts, all on
#: the reader searching for a "consensus" header that the sheet does not
#: carry -- a deterministic failure, four times. Nothing was trained,
#: nothing is citable. The retry-limit standing item collects THREE more
#: data points (four attempts, three retries, against a stated limit of
#: one): ``phase8.RETRY_LIMIT_IS_NOT_HOLDING``.
CLEFTGNN_FIRST_LAUNCH_VOID = {
    "void": "2026-08-17",
    "run": "p10_cleftgnn__9a28f8a8__p10-cleftgnn",
    "attempts": 4,
    "cause": (
        "the label reader searched for a 'consensus' header the sheet does "
        "not carry -- deterministic, four times"
    ),
    "fixed_by": "CONSENSUS_LABEL_IS_THE_MEDIAN; scoresheet.load_median",
    "citable": "nothing -- no training reached",
    "retry_item": "+3 retries on phase8.RETRY_LIMIT_IS_NOT_HOLDING",
}



#: **[MEASURED 2026-08-17] THE log(0) HYPOTHESIS IS REFUTED, AND THE
#: ACTUAL NON-FINITE SOURCE IS DIVERGENCE IN THE FIRST OPTIMISER STEP --
#: driven by feature SCALE, not by the thin class.**
#:
#: **1. The bias is finite even with a class wholly absent.** ``reset``
#: already divides by a smoothed denominator -- ``(counts + 1e-8) /
#: (counts.sum() + 5e-8)`` -- so a train fold holding zero grade-5
#: patients gives ``log`` of 1e-8/183, i.e. **-23.65: finite**, never
#: -inf. Measured on the exact arithmetic (counts [4,71,88,24,0]), and
#: again on a constructed model whose fit labels contain no grade 5:
#: classifier bias finite, weight all-zero as designed, normalisation std
#: with no zero, **every parameter finite**.
#:
#: **2. Epoch 0 is clean.** With the classifier weight at zero the logits
#: ARE the bias, measured exactly so, and the epoch-0 prediction is the
#: frequency-weighted expected grade (2.4167 in the probe's fold). So the
#: run did not start non-finite.
#:
#: **3. The first SGD step overshoots, and the loss explodes.** Under the
#: registered recipe (plain SGD, lr 0.01, momentum 0, full fine-tune):
#: epoch-1 mean loss **16.42**, epoch 2 **108.26**, epoch 3 **29,519.17**;
#: max|grad| 1.6e1 -> 3.3e1 -> 2.4e3. The initial batch's loss is ~0.9 by
#: the bias arithmetic (softmax over the measured bias gives 0.083 /
#: 0.417 / 0.500 on the three present grades), so an epoch mean of 16.42
#: over 8+4 samples implies the SECOND batch scored ~47 -- after exactly
#: one step. The trajectory is a divergence, not a plateau.
#:
#: **4. Why: the 36-region SUM.** Gated pooling sums 36 region vectors
#: and the SABM sums 36 more, so what reaches the classifier has
#: magnitude **|f_t| 20.7, |v_a| 8.4, |f_t + v_a| 21.5** at init. A
#: zero-initialised classifier receiving features of that size takes a
#: first step of order lr x |feature| across 1024 dimensions; the softmax
#: saturates and cross-entropy explodes. Both sums are FAITHFUL to the
#: notebook -- but the notebook trains them with a FROZEN backbone, and
#: this build fine-tunes ResNet-50 end to end at their stated lr. The
#: divergence lives in that combination, which no single source
#: prescribed.
#:
#: **Measurement caveats, stated**: the forward ran with a DECLARED
#: SUBSTITUTE for ``roi_align`` (per-box adaptive average pooling),
#: ``pretrained=False``, synthetic images, 12 samples. So this measures
#: the mechanism CLASS -- scale into a zero classifier at lr 0.01 -- not
#: the pod's exact numbers. The pod's own instrumented probe would settle
#: the arithmetic; nothing here needs it to refute log(0).
#:
#: **Two apparatus findings, recorded because they explain the six
#: seconds:**
#:
#: * **Gate 3 cannot catch a non-finite epoch 0.** ``assert_epoch0_
#:   calibrated`` compares ``abs(pred_mean - train_mean) > tol``, and
#:   every comparison against NaN is False, so a NaN epoch-0 passes it
#:   silently. The first thing that WOULD catch it is ``metrics.pcc`` --
#:   which is exactly where the run died, at harness line 307, AFTER
#:   epoch 1. harness.py is FROZEN, so this is recorded, not fixed; it
#:   also means the log alone cannot prove epoch 0 was clean, which is
#:   why measurement 2 above was necessary.
#: * **This model's forward cannot run on the laptop at all.**
#:   ``torchvision::roi_align`` has no CPU kernel in the local build
#:   (CUDA-only). No laptop test can execute CleftGNN's forward, so no
#:   suite run could have caught this before the cluster. That is a
#:   testability property of the port, and it belongs in the record
#:   beside the port.
CLEFTGNN_NAN_MEASURED = {
    "measured": "2026-08-17, offline",
    "hypothesis_refuted": (
        "log(0) in the CE bias: REFUTED -- reset already smooths by 1e-8, "
        "so an absent class gives bias -23.65, finite; every parameter "
        "finite at init"
    ),
    "epoch_0_clean": (
        "weight is zero so logits ARE the bias; epoch-0 prediction is the "
        "frequency-weighted expected grade (2.4167 in the probe)"
    ),
    "actual_source": (
        "divergence in the FIRST optimiser step: loss 16.42 -> 108.26 -> "
        "29,519.17 across three epochs, max|grad| 1.6e1 -> 2.4e3; the "
        "initial batch is ~0.9, so one step took the next to ~47"
    ),
    "mechanism": (
        "gated pooling sums 36 regions and the SABM sums 36 more, so "
        "|f_t + v_a| is 21.5 at init; a zero-initialised classifier at "
        "lr 0.01 over 1024 dims overshoots and the softmax saturates"
    ),
    "scale_at_init": {"f_t": 20.7, "v_a": 8.4, "sum": 21.5},
    "the_combination_no_source_prescribed": (
        "both sums are faithful to the notebook, but the notebook FREEZES "
        "its backbone; this build fine-tunes ResNet-50 end to end at "
        "their stated lr"
    ),
    "caveats": (
        "declared substitute for roi_align (per-box adaptive average "
        "pooling), pretrained=False, synthetic images, 12 samples -- the "
        "mechanism class, not the pod's arithmetic",
    ),
    "apparatus_findings": {
        "gate_3_misses_non_finite": (
            "abs(NaN - x) > tol is False, so a NaN epoch 0 passes gate 3 "
            "silently; metrics.pcc at harness line 307 is the first "
            "catch. harness.py is FROZEN -- recorded, not fixed"
        ),
        "not_laptop_testable": (
            "torchvision::roi_align has no CPU kernel in the local build, "
            "so no suite test can execute this model's forward -- which "
            "is why nothing caught this before the cluster"
        ),
    },
    "thin_class_artifact_still_open": (
        "the 1e-8 epsilon is a poor smoother: an absent class gets bias "
        "-23.65, i.e. probability 1e-10, so that grade is effectively "
        "unreachable and the expected-value reading is biased. NOT the "
        "crash cause, but a fidelity question the fix must answer"
    ),
}


#: **[VOID 2026-08-17] The second CleftGNN launch.** Run
#: ``p10_cleftgnn__32519178__p10-cleftgnn-2``: four attempts, each dying
#: six seconds in at the first ``metrics.pcc`` call -- after epoch 1, per
#: the harness's own order -- on non-finite predictions
#: (``CLEFTGNN_NAN_MEASURED``). The label resolution worked: the Median
#: column verified against all 237 rows, grade counts 1:5, 2:89, 3:110,
#: 4:30, 5:3. Nothing trained to completion; nothing is citable. The
#: retry-limit item collects FOUR more.
CLEFTGNN_SECOND_LAUNCH_VOID = {
    "void": "2026-08-17",
    "run": "p10_cleftgnn__32519178__p10-cleftgnn-2",
    "attempts": 4,
    "cause": (
        "non-finite predictions at the first metrics.pcc call, after "
        "epoch 1 -- training divergence, CLEFTGNN_NAN_MEASURED"
    ),
    "label_worked": (
        "the Median column verified on all 237; counts 1:5, 2:89, 3:110, "
        "4:30, 5:3 -- the 2026-08-17 label correction holds"
    ),
    "citable": "nothing",
    "retry_item": "+4 on phase8.RETRY_LIMIT_IS_NOT_HOLDING",
    "fix": "NOT BUILT -- a registration question, proposed not picked",
}



#: **[FIDELITY CORRECTION 2026-08-17 -- not a fix] THE BACKBONE IS
#: FROZEN.** The full fine-tune was an INFERENCE from the manuscript's
#: "ResNet-50 backbone", never their recipe. The notebook -- the only
#: executable artifact the group ran -- freezes its backbone, and their
#: stated lr 0.01 belongs to that regime. Every alternative on the table
#: (gradient clipping, discriminative learning rates, replacing the
#: notebook-disambiguated sum with a mean) would have added a mechanism
#: the group never states in order to rescue an inference of mine. Moving
#: to the frozen backbone removes the inference instead.
#:
#: **The benefit, and it is not incidental**: the contrast against the
#: 0.2520 arm becomes LIKE-FOR-LIKE -- frozen backbone plus trained head
#: on both sides -- so the comparison is of architecture without the
#: training regime confounding it. The earlier build compared CleftGNN
#: full-fine-tuned against a frozen-backbone ViT arm; any gap would have
#: carried both differences at once.
#:
#: **Frozen means eval() too.** ResNet-50 carries BatchNorm, so a frozen
#: backbone left in train mode would still drift its running statistics
#: and would compute batch statistics -- making the "frozen"
#: representation depend on the batch a face arrived in. The void
#: ladder's own defect was missing BatchNorm running statistics; leaving
#: this implicit is how that returns. Asserted at reset: the backbone has
#: zero trainable parameters, everything outside it trains, and the
#: optimiser is handed only what trains.
BACKBONE_IS_FROZEN = {
    "corrected": "2026-08-17, a fidelity correction, not a fix",
    "was": "full fine-tune -- an inference from 'ResNet-50 backbone'",
    "now": "frozen: requires_grad False AND eval() during training",
    "why": (
        "the notebook, the only executable artifact the group ran, "
        "freezes its backbone; lr 0.01 belongs to that regime"
    ),
    "alternatives_rejected": (
        "clipping, discriminative lr, mean-not-sum -- each adds a "
        "mechanism the group never states to rescue my inference"
    ),
    "benefit": (
        "like-for-like against 0.2520: frozen backbone + trained head on "
        "both sides, so architecture is compared without the training "
        "regime confounding it"
    ),
    "batchnorm": (
        "eval() is part of frozen -- running statistics must not drift "
        "and batch statistics would make the representation depend on "
        "the batch; the void ladder's defect was exactly missing BN "
        "running statistics"
    ),
    "asserted_at_reset": (
        "backbone trainable count is zero; trainable equals total minus "
        "backbone; the optimiser receives only trainable parameters"
    ),
}


#: **[DEVIATION REGISTERED 2026-08-17] THIN-CLASS SMOOTHING: Laplace,
#: not epsilon.** The calibrated CE init divided by ``(counts + 1e-8) /
#: (N + 5e-8)``, which puts a class absent from a training fold at
#: probability ~1e-10 -- bias -23.65 -- so that grade is effectively
#: unreachable and every expected-value reading is biased downward
#: wherever grade 5 belongs. This is INDEPENDENT of the crash
#: (``CLEFTGNN_NAN_MEASURED`` refuted the log(0) hypothesis: the epsilon
#: kept it finite). The replacement is the standard add-one estimator,
#: ``(count + 1) / (N + K)``, giving ~-5.26 for an absent class -- a
#: grade the model can still reach.
#:
#: **A stated deviation with its reason**, because the group's recipe
#: prescribes no init at all: the manuscript is SILENT on thin classes,
#: and their own 181-image split faced the same problem -- five grades
#: over 181 images, with the extremes necessarily thin, split 85:15 with
#: no mention of what happens to a class that lands entirely on one side.
#: **Thursday-flagged**: it is a question their revision should answer,
#: and our own answer is registered here rather than defaulted.
THIN_CLASS_SMOOTHING = {
    "registered": "2026-08-17",
    "was": "(counts + 1e-8) / (N + 5e-8) -- absent class at p ~ 1e-10, bias -23.65",
    "now": "(count + 1) / (N + K) -- absent class at bias ~ -5.26, reachable",
    "why": (
        "the epsilon made a grade unreachable and biased the "
        "expected-value reading, independently of the crash"
    ),
    "not_the_crash_cause": "CLEFTGNN_NAN_MEASURED refuted log(0)",
    "manuscript_silence": (
        "silent on thin classes despite their 181-image 85:15 split "
        "facing the same problem -- Thursday-flagged"
    ),
}


#: **[REGISTERED 2026-08-17] EVERY DEVIATION FROM THE GROUP'S RECIPE, in
#: one place.** Exit criterion 2 says "deviations enumerated"; this is
#: that list, and it is exhaustive by construction -- anything not here
#: is either theirs or a silence their sources never fill.
#:
#: 1. **Inner-val early stopping** (0.2 / 40 / 5, monitor inner_val_mse).
#:    Fills a SILENCE rather than contradicting a statement: the
#:    manuscript states no epoch budget anywhere (measured, zero
#:    mentions). Reason: fixed budgets measured post-collapse endpoints
#:    on this cohort, peaks at epochs 2-11.
#: 2. **Laplace smoothing in the CE init** (``THIN_CLASS_SMOOTHING``).
#:    Also fills a silence -- they prescribe no init -- and the reason is
#:    reachability of thin grades.
#:
#: **NOT deviations, recorded here so the list is not misread**: the
#: frozen backbone is a fidelity CORRECTION toward their own artifact
#: (``BACKBONE_IS_FROZEN``); the 36-region set is the notebook's own
#: enumeration; batch 16 and momentum 0 are stated defaults filling
#: silences, carried with their caveats.
REGISTERED_DEVIATIONS = {
    "registered": "2026-08-17",
    # [EXTENDED 2026-08-17, after the divergence was measured] Two more,
    # each with its reason and its fidelity cost. Both fill silences; both
    # were chosen over alternatives that would have overwritten something
    # the group DOES state (lr 0.01) or added a mechanism they never
    # mention (clipping).
    "deviations": (
        "inner-val early stopping (0.2/40/5, inner_val_mse) -- fills the "
        "epoch silence; reason: post-collapse endpoints, peaks 2-11; "
        "fidelity cost: LOW, the manuscript states no budget at all",
        "Laplace (count+1)/(N+K) in the CE init -- fills the init "
        "silence; reason: thin-grade reachability; fidelity cost: LOW, "
        "they prescribe no init",
        "LayerNorm on the fused feature before the classifier "
        "(FUSED_NORM_AND_STANDARD_INIT fix a) -- reason: |f_t + v_a| ~ 21 "
        "drove the first lr-0.01 step to blow up; fidelity cost: LOWEST "
        "AVAILABLE, it leaves the notebook's torch.sum intact and "
        "normalises only what happens after it, which no source specifies",
        "the classifier's weight returns to nn.Linear's own init, Laplace "
        "bias kept (fix b) -- reason: the zero-weight init was MINE, a "
        "gate-3 translation, and it made epoch-0 predictions constant by "
        "construction; fidelity cost: NEGATIVE, it withdraws an addition "
        "of mine rather than departing from them",
        "LayerNorm on the GNN branch before eq. (9)'s sum, affine-free "
        "(GNN_BRANCH_NORM, route 3) -- reason: the branch measured 5.8x "
        "larger than the SABM branch while carrying ratio 0.0068, so the "
        "plain sum drowned the only informative path (0.7794 in, 0.1315 "
        "out); fidelity cost: MEDIUM, eq. (9) states a plain sum and this "
        "changes its balance -- chosen over dropping the notebook's own "
        "sigmoid (HIGH) and over an outcome-chosen init knob",
    ),
    "expected_count": 5,
    "untouched_deliberately": (
        "lr 0.01 -- their one stated hyperparameter; lowering it would "
        "overwrite the single number their recipe actually gives",
        "the 36-region sum -- the notebook disambiguates it",
        "no gradient clipping -- it fills no silence",
    ),
    # [CORRECTED 2026-08-17, on reading the notebook's EXECUTION cell]
    # "batch 16 -- the notebook states none" is WRONG: cell 9 states
    # BATCH_SIZE = 16 plainly. The value is right by coincidence; the
    # claim about the artifact was made from one cell of it
    # (NOTEBOOK_RECIPE_IS_ADAM). And the epoch silence is the
    # MANUSCRIPT'S: the notebook states EPOCHS = 5 with no validation
    # split, so early stopping departs from the notebook even while
    # filling the manuscript's silence.
    "corrected": (
        "2026-08-17: the notebook DOES state BATCH_SIZE = 16 (cell 9), "
        "and EPOCHS = 5 -- the epoch silence is the manuscript's alone; "
        "see NOTEBOOK_RECIPE_IS_ADAM",
        "2026-08-17: the early-stopping deviation's fidelity cost is "
        "HIGHER than 'LOW' as recorded. Against the manuscript it is "
        "still low -- no budget is stated anywhere, measured. Against the "
        "notebook it is REAL: EPOCHS = 5 with no validation split is "
        "explicit, and we depart from it. Kept anyway, defect-cited, on "
        "both cells (NOTEBOOK_BUDGET_AND_NORMALISATION, "
        "NOTEBOOK_CELL_DEVIATIONS)",
    ),
    "not_deviations": (
        "the frozen backbone -- a fidelity correction toward the "
        "notebook (BACKBONE_IS_FROZEN)",
        "the 36-region set -- the notebook's own enumeration",
        "batch 16 and momentum 0 -- stated defaults filling silences, "
        "carried with their caveats",
    ),
}


#: **[REGISTERED 2026-08-17] THE FAITHFUL ARM -- the same architecture
#: under THEIR protocol, as its own cell beside the protocol arm.**
#:
#: **The purpose, stated before any number**: the protocol arm
#: (``p10_cleftgnn``: 237 patients, median label, 5-fold OOF, five seeds,
#: paired BCa, both readings) and this arm differ ONLY in protocol, so
#: any gap between them is attributable to PROTOCOL rather than
#: architecture. That is the direct answer to "why are we at 0.252 while
#: they reach 0.598".
#:
#: **Their protocol, as our cohort allows**: rater-specific models -- one
#: per rater, five models, each trained on that rater's own grades; an
#: 85:15 stratified split (seeded, registered); their metrics -- Top-1
#: macro precision, recall, F1, and PCC; a SINGLE run with NO intervals,
#: which is their form and is recorded as such rather than repaired.
#: Evaluated on both of their test-set shapes: the held-out 15% (their
#: Study Test Set shape) and the 25-image benchmark against its Score
#: (their Benchmark Test Set -- literally the same 25 images).
#:
#: **The arithmetic on their 0.598 cell, carried with this arm**: one of
#: FIFTEEN cells (3 backbones x 5 raters); n=25; no interval published
#: anywhere; Fisher 95% CI ~[0.27, 0.80] recomputed at recording time.
#: And the decisive one -- **the same rater-E model scored 0.283 on
#: their n=28 Study set**. A doubling between two test sets of the same
#: task, for one model, is the signature of sampling noise, not of a
#: capability that transfers.
FAITHFUL_ARM_REGISTERED = {
    "registered": "2026-08-17",
    "purpose": (
        "the protocol arm and this arm differ ONLY in protocol, so any "
        "gap is attributable to protocol rather than architecture -- the "
        "direct answer to the 0.598 question"
    ),
    "protocol": {
        "models": "rater-specific, one per rater, five models",
        "split": "85:15 stratified on the rater's own grade, seeded 1337",
        "metrics": "Top-1 macro precision, recall, F1, and PCC",
        "runs": "single, NO intervals -- their form, recorded not repaired",
        "test_shapes": (
            "the held-out 15% (their Study Test Set shape) and the "
            "25-image benchmark against Score (their Benchmark shape, "
            "the same 25 images)"
        ),
    },
    "their_598_cell": {
        "one_of": 15,
        "n": 25,
        "interval": "none published",
        "fisher_95_recomputed": (0.2656, 0.8033),
        "same_model_on_their_study_set": 0.283,
        "reading": (
            "a doubling between two test sets of the same task, for one "
            "model, is the signature of sampling noise"
        ),
    },
}


#: **[REGISTERED 2026-08-17, PRIOR COMMITTED BOTH WAYS BEFORE ANY
#: NUMBER] THE RATER-SPECIFIC SCREEN.** The current best arm -- frozen
#: ViT-B/16, ImageNet, G1, whole image, the 0.2520 arm -- run once per
#: rater: five cells, five seeds each, the standard criterion (5-fold
#: OOF, per-seed pooled PCC).
#:
#: **The measured prior**: Phase 1 learnability puts the panel mean at
#: **0.6022** and the single-rater orthodontist target at **0.4944**
#: (``labels.LEARNABILITY_237``). Averaging five raters cancels noise
#: that a single rater carries, so the registered prediction is that
#: **every rater-specific cell lands BELOW 0.2520**.
#:
#: **Both readings, committed now:**
#:
#: * **All five below** -> rater-specific modelling is measured as WORSE
#:   on this cohort: a direct, intervalled test of the literature's
#:   central premise, and the strongest form the question has.
#: * **Any cell meaningfully above** -- beyond the 0.2520 arm's own seed
#:   band (sd 0.0148, so beyond roughly 0.2520 + 2 sd) -> the prior is
#:   REFUTED, and the full rater ladder is justified immediately.
#:
#: The screen is cheap because the 0.2520 arm is frozen-backbone: only
#: the head refits, on embeddings already extracted and declared.
RATER_SCREEN_REGISTERED = {
    "registered": "2026-08-17, before any number",
    "arm": "the 0.2520 arm -- frozen ViT-B/16, imagenet, G1, whole image",
    "cells": "five raters x five seeds, 5-fold OOF, standard criterion",
    "prior": {
        "panel_mean_learnability": 0.6022,
        "orthodontist_learnability": 0.4944,
        "prediction": "every rater-specific cell lands BELOW 0.2520",
        "mechanism": "averaging five raters cancels noise a single rater carries",
    },
    "readings": {
        "all_below": (
            "rater-specific modelling measured WORSE on this cohort -- a "
            "direct intervalled test of the literature's central premise"
        ),
        "any_meaningfully_above": (
            "beyond the 0.2520 arm's seed band (sd 0.0148): the prior is "
            "refuted and the full rater ladder is justified immediately"
        ),
    },
    "cost": "cheap -- frozen backbone, only the head refits on declared embeddings",
}


#: **[REGISTERED 2026-08-17, CONDITIONAL -- NOT BUILT] The full rater
#: ladder.** 5 raters x 4 backbones x 2 geometries x 2 inits x 5 seeds =
#: **400 runs**. Built if the screen refutes its prior, or on the
#: maintainer's word regardless. Registered now so that a later decision to
#: build it is a decision about scope and not about design.
RATER_LADDER_CONDITIONAL = {
    "registered": "2026-08-17, conditional, not built",
    "shape": "5 raters x 4 backbones x 2 geometries x 2 inits x 5 seeds",
    "runs": 400,
    "trigger": (
        "the screen refuting its prior, or the maintainer's word regardless"
    ),
}



#: **[OBSERVED 2026-08-17] THE RATER SCREEN: the registered MIXED reading
#: fired, and the Phase 1 prior is confirmed by an independent route.**
#:
#: Pooled OOF PCC per rater, five seeds each, against the arm's 0.2520:
#: cleft patient **-0.0098** (sd 0.0207), orthodontist **0.1606**
#: (0.0207), speech and language therapist **0.2150** (0.0149), plastic
#: surgeon **0.2703** (0.0196), psychologist **0.1362** (0.0283).
#:
#: **Four below, one nominally above and INSIDE the band.** The plastic
#: surgeon's 0.2703 exceeds 0.2520 by 0.0183, which is 1.24 of the arm's
#: own seed sd (0.0148) and well inside the registered refutation
#: threshold of two sd. So the registered MIXED reading fired: no cell is
#: beyond the band, and not all five are below. The prior is not refuted
#: and the full ladder is not triggered by this screen.
#:
#: **What it establishes**: the panel mean is a better target than any
#: single rater, now measured by a SECOND independent route -- Phase 1
#: measured it as learnability (mean 0.6022 vs orthodontist 0.4944)
#: before any arm ran; this measures it as trained-model performance,
#: on the same cohort, under the standard criterion. Different data path,
#: same conclusion.
#:
#: **And no rater approaches the literature's per-rater figures.** The
#: best cell here is 0.2703; CleftGNN's Study-Set range runs to 0.440 and
#: its Benchmark cell to 0.598. **the orthodontist premise raised at supervision fails
#: again**: the orthodontist -- proposed as the most reliable rater -- is
#: fourth of five at 0.1606, after Phase 1 already measured that premise
#: unsupported on reliability (SLT 0.654 vs 0.628). Two different
#: instruments, the same answer.
RATER_SCREEN_OBSERVED = {
    "observed": "2026-08-17",
    "per_rater": {
        "cleft patient": {"pcc": -0.0098, "sd": 0.0207},
        "orthodontist": {"pcc": 0.1606, "sd": 0.0207},
        "speech and language therapist": {"pcc": 0.2150, "sd": 0.0149},
        "plastic surgeon": {"pcc": 0.2703, "sd": 0.0196},
        "psychologist": {"pcc": 0.1362, "sd": 0.0283},
    },
    "arm": 0.2520,
    "verdict": "MIXED -- four below, one nominally above and inside the band",
    "inside_the_band": (
        "the plastic surgeon's +0.0183 is 1.24 of the arm's own seed sd "
        "(0.0148), inside the registered two-sd refutation threshold"
    ),
    "prior_confirmed_independently": (
        "Phase 1 measured the mean as more learnable than any single "
        "rater (0.6022 vs 0.4944) BEFORE any arm ran; this measures the "
        "same ordering as trained-model performance -- different data "
        "path, same conclusion"
    ),
    "no_rater_approaches_the_literature": (
        "best cell 0.2703 against CleftGNN's Study range to 0.440 and "
        "its Benchmark cell 0.598"
    ),
    "orthodontist_premise_fails_again": (
        "fourth of five at 0.1606; Phase 1 already measured the premise "
        "unsupported on reliability (SLT 0.654 vs 0.628)"
    ),
    "ladder_not_triggered": "no cell beyond the band; RATER_LADDER_CONDITIONAL stands",
}


#: **[OBSERVED + DIAGNOSED 2026-08-17] THE FAITHFUL ARM'S NaN PCCs ARE A
#: CONSTANT PREDICTION VECTOR -- and they are CONFOUNDED with our own
#: divergence, so they say nothing yet about their regime.**
#:
#: **The mechanism, exactly.** ``metrics.pcc`` RETURNS nan when either
#: side is constant (``denom == 0``) and RAISES on non-finite input.
#: Every faithful-arm PCC is nan and none raised, so the predictions were
#: finite and CONSTANT: one Top-1 class for every image. The F1 range
#: confirms it arithmetically -- a constant Top-1 predictor scores macro
#: F1 = (1/K) x 2p/(p+1) at the modal class's prevalence p, which is
#: **0.088 on the benchmark's 3/7/6/6/3 and ~0.112 on a 36-image study
#: split**: the observed 0.04-0.13, with no room left over.
#:
#: **Every model stopped at epoch 1**, and one inner-val MSE was itself
#: nan. That is the same signature the protocol arm shows: the first
#: epoch destroys the model, so epoch 1 is the best the early stopper
#: ever sees, and everything after is worse.
#:
#: **THE CONFOUND, and it governs what may be said.** Both arms are the
#: SAME model at the SAME lr with the SAME 36-region summed features. A
#: measured collapse in the faithful arm is therefore explained by our
#: divergence bug (``CLEFTGNN_SCALE_MEASURED``) before it is explained by
#: anything about n=25-36 or about their protocol. **So this run does NOT
#: show that their evaluation regime is fragile on our cohort** -- it
#: shows that a model broken in a known way produces a constant predictor
#: whose PCC is undefined. The regime question is genuinely interesting
#: and remains OPEN; it becomes answerable only when the faithful arm
#: runs a model that trains.
#:
#: **What it does and does not show, stated so neither is misread:**
#:
#: * It does NOT refute their published numbers on their data. Nothing
#:   here touches their cohort, their splits, or their training.
#: * It does NOT yet show the regime's fragility on ours -- our own bug
#:   explains the collapse.
#: * It DOES show that a single-run, no-interval, n=25-36 protocol
#:   reports F1 values (0.04-0.13) for a predictor that has learned
#:   nothing, and reports nan rather than a warning for its PCC. That is
#:   a property of the REPORTING regime, visible regardless of our bug,
#:   and it is the part that stands.
FAITHFUL_ARM_OBSERVED = {
    "observed": "2026-08-17",
    "figures": {
        "pcc": "nan on both test sets, all five raters",
        "f1_macro": "0.04-0.13",
        "stopping": "epoch 1 for every model; one inner-val MSE itself nan",
    },
    "mechanism": (
        "pcc RETURNS nan on a constant vector and RAISES on non-finite; "
        "every cell returned nan, so predictions were finite and CONSTANT "
        "-- one Top-1 class for every image"
    ),
    "f1_arithmetic_confirms": (
        "a constant Top-1 predictor scores (1/K) x 2p/(p+1): 0.088 on the "
        "benchmark's 3/7/6/6/3, ~0.112 on a 36-image study split -- the "
        "observed range with nothing left over"
    ),
    "confound": (
        "the same model, lr and summed features as the protocol arm, so "
        "our divergence explains the collapse before their protocol does"
    ),
    "does_not_show": (
        "it does NOT refute their published numbers on their data",
        "it does NOT yet show the regime's fragility on ours -- our own "
        "bug explains the collapse",
    ),
    "does_show": (
        "a single-run, no-interval, n=25-36 protocol reports F1 0.04-0.13 "
        "for a predictor that learned nothing, and reports nan rather "
        "than a warning for its PCC -- a property of the REPORTING "
        "regime, visible regardless of our bug"
    ),
    "regime_question": "OPEN until the faithful arm runs a model that trains",
    "not_ledgered_as_a_claim": (
        "confounded; the ledger carries the void, not a regime finding"
    ),
}


#: **[MEASURED 2026-08-17] THE BACKBONE-GRADIENT DIAGNOSIS IS REFUTED,
#: AND THE SCALE HYPOTHESIS SURVIVES ITS FIRST TEST -- the frozen build
#: still diverges, with only the 5.8M head training.**
#:
#: Measured offline on the frozen build (declared substitute for
#: ``roi_align``; ``pretrained=False``; synthetic images -- the mechanism
#: class, not the pod's arithmetic):
#:
#: * **frozen 23,508,032 parameters, trainable 5,778,949** -- the freeze
#:   is real and the optimiser sees only the head.
#: * **Epoch-0 predictions are CONSTANT BY CONSTRUCTION**: unique values
#:   1, sd 0.000000. The calibrated CE init sets the classifier weight to
#:   zero, so every image gets the same logits -- the bias -- and the
#:   same expected grade. **A PCC against that vector is nan for reasons
#:   that have nothing to do with training**, which is worth knowing
#:   before any epoch-0 metric is quoted.
#: * **First-batch loss 1.2975** -- sane -- against an **epoch-1 mean of
#:   15.0116**. Divergence inside the first epoch, with the backbone
#:   frozen. max|grad| at the first batch 1.83.
#: * Predictions after epoch 1: still finite here, still constant
#:   (unique 1). On the pod, with real pretrained features, the same
#:   trajectory reaches non-finite -- which is why the protocol arm
#:   RAISES where the faithful arm merely returns nan.
#:
#: **So**: freezing removed 23.5M parameters from the optimiser and the
#: divergence survived. The remaining registered suspect is SCALE -- the
#: head takes lr-0.01 steps against features measured at |f_t + v_a| ~
#: 21.5, summed over 36 regions twice. That is now the hypothesis to
#: measure on the pod, NOT to adopt: ``P10_SCALE_PROBE`` states exactly
#: what it prints.
CLEFTGNN_SCALE_MEASURED = {
    "measured": "2026-08-17, offline, frozen build",
    "backbone_gradient_diagnosis": "REFUTED -- the freeze did not fix it",
    "parameters": {"frozen": 23508032, "trainable": 5778949},
    "epoch_0_predictions_constant_by_construction": (
        "zero classifier weight means every image gets the bias, so the "
        "epoch-0 prediction vector has sd 0.000000 and any PCC against "
        "it is nan for reasons unrelated to training"
    ),
    "divergence_inside_epoch_1": {
        "first_batch_loss": 1.2975,
        "epoch_1_mean_loss": 15.0116,
        "first_batch_max_grad": 1.83,
        "predictions_after": "finite here, constant (unique 1)",
    },
    "why_the_two_arms_differ": (
        "with real pretrained features the same trajectory reaches "
        "non-finite, so the protocol arm RAISES where the faithful arm "
        "returns nan on a constant vector"
    ),
    "surviving_hypothesis": (
        "SCALE: the head takes lr-0.01 steps against |f_t + v_a| ~ 21.5, "
        "summed over 36 regions twice -- to be MEASURED on the pod"
    ),
    "caveats": (
        "declared substitute for roi_align, pretrained=False, synthetic "
        "images: the mechanism class, not the pod's arithmetic"
    ),
}


#: **[REGISTERED 2026-08-17] THE POD PROBE, and what it prints.**
#: ``scripts/p10_scale_probe.py`` -- no RunContext, no artifacts, the
#: ``p8b_softmax_probe`` precedent. For the protocol arm's FIRST FOLD and
#: for one faithful-arm rater split, it prints:
#:
#: 1. per-fold first-batch loss and max|grad| at the first batch;
#: 2. whether logits are finite at epoch 0 and again after step 1;
#: 3. the prediction vector's sd at epoch 0, after step 1, and after
#:    epoch 1 -- a zero sd is the collapse, a nan is the blow-up, and the
#:    two are different findings;
#: 4. |f_t| and |v_a| on real features, so the 21.5 measured offline is
#:    confirmed or corrected on the pod's own data;
#: 5. the same five numbers for the faithful arm's split, so
#:    "do the arms share the mechanism" is answered rather than argued.
P10_SCALE_PROBE = {
    "registered": "2026-08-17",
    "script": "scripts/p10_scale_probe.py",
    "prints": (
        "per-fold first-batch loss and max|grad|",
        "logits finite at epoch 0 and after step 1",
        "prediction sd at epoch 0 / after step 1 / after epoch 1 -- zero "
        "sd is collapse, nan is blow-up, and they are different findings",
        "|f_t| and |v_a| on real features, against the offline 21.5",
        "the same five for a faithful-arm split, answering whether the "
        "arms share the mechanism",
    ),
    "no_artifacts": "a probe, not a run -- the p8b_softmax_probe precedent",
}


#: **[VOID 2026-08-17] The third protocol-arm launch**, under the freeze:
#: still non-finite at ~6 seconds. Four attempts. Recorded with the
#: faithful arm's own run, whose numbers exist but are confounded
#: (``FAITHFUL_ARM_OBSERVED``). The retry-limit item collects SIX more.
CLEFTGNN_THIRD_LAUNCH_VOID = {
    "void": "2026-08-17",
    "run": "the p10_cleftgnn relaunch under the freeze",
    "attempts": 4,
    "cause": "non-finite predictions again -- CLEFTGNN_SCALE_MEASURED",
    "faithful_arm": (
        "ran to completion but every PCC is nan on a constant predictor; "
        "not citable, and confounded with this divergence"
    ),
    "retry_item": "+6 on phase8.RETRY_LIMIT_IS_NOT_HOLDING",
}



#: **[SETTLED 2026-08-17, by the pod probe] THE TWO ARMS SHARE THE
#: MECHANISM EXACTLY -- so the faithful arm's collapse is OURS, and that
#: run demonstrates nothing about their protocol.**
#:
#: The pod's figures, on real features: **|f_t| 21.1 in BOTH arms**
#: (validating the 21.5 measured offline through a substitute), and the
#: first lr-0.01 step taking the logits from **3.89 to 12.75** in the
#: protocol arm and **3.94 to 18.21** in the faithful arm. Epoch-1 mean
#: loss **4,581** and **1.3e16**. One mechanism, two severities.
#:
#: **The attribution correction is therefore CONFIRMED BY MEASUREMENT,
#: not by argument.** ``FAITHFUL_ARM_OBSERVED`` held that our divergence
#: explained the faithful arm's constant predictor before their protocol
#: did; the shared signature settles it. **Nothing about their evaluation
#: regime is demonstrated by that run.** The regime question stays OPEN
#: and stays out of the ledger until a model that trains runs it.
ARMS_SHARE_THE_MECHANISM = {
    "settled": "2026-08-17, by the pod probe",
    "f_t_real_features": {"protocol": 21.1, "faithful": 21.1},
    "offline_estimate_validated": 21.5,
    "logits_after_one_step": {
        "protocol": (3.89, 12.75), "faithful": (3.94, 18.21),
    },
    "epoch_1_mean_loss": {"protocol": 4581.0, "faithful": 1.3e16},
    "conclusion": (
        "one mechanism, two severities -- the faithful arm's collapse is "
        "ours; that run demonstrates nothing about their protocol"
    ),
    "regime_question": "OPEN, unledgered, until a model that trains runs it",
}


#: **[RECORDED 2026-08-17] sd 0.000000 HAS TWO CAUSES, and they are
#: different findings wearing the same number.**
#:
#: 1. **Epoch 0: constant BY CONSTRUCTION.** The calibrated CE init set
#:    the classifier WEIGHT to zero, so every image received identical
#:    logits -- the bias -- and therefore an identical expected grade.
#:    Nothing about the data or the training produced this; it was a
#:    property of my init, and any PCC against that vector was nan for
#:    reasons unrelated to learning. **Fix (b) removes this cause
#:    entirely**: with the framework's small-random weights the epoch-0
#:    predictions vary (measured sd 0.0015, 48 unique of 48).
#: 2. **Epoch 1: SATURATION.** After the first oversized step the softmax
#:    saturates onto one class, so every image again predicts the same
#:    grade -- this time because the model has been destroyed. Fix (a)
#:    addresses this cause.
#:
#: **Why the distinction earns its record**: a reader meeting sd 0.0 in a
#: metrics file cannot tell "the head has not been trained yet" from "the
#: head has been ruined", and the two demand opposite responses. The
#: project has an exact precedent for the confusion -- ``metrics.pcc``
#: returns nan for BOTH, while ``_pair`` raises only for the non-finite
#: case, which is precisely how the faithful arm (nan) and the protocol
#: arm (raise) came to look like different bugs.
SD_ZERO_HAS_TWO_CAUSES = {
    "recorded": "2026-08-17",
    "cause_1_epoch_0": (
        "constant BY CONSTRUCTION -- zero classifier weight gives every "
        "image the bias; a property of my init, not of the data. Fix (b) "
        "removes it: measured sd 0.0015, 48 unique of 48"
    ),
    "cause_2_epoch_1": (
        "SATURATION -- after the oversized first step the softmax "
        "collapses onto one class; the model has been destroyed. Fix (a) "
        "addresses it"
    ),
    "why_it_matters": (
        "sd 0.0 cannot distinguish 'not trained yet' from 'ruined', and "
        "the two demand opposite responses; pcc returns nan for both "
        "while raising only on non-finite -- which is how the two arms "
        "came to look like different bugs"
    ),
}


#: **[FIX APPLIED 2026-08-17, chosen on the figures, measured before
#: launch] (a) LayerNorm on the fused feature; (b) the framework's own
#: classifier weights, Laplace bias kept.**
#:
#: **(a)** ``LayerNorm(1024)`` immediately before the classifier. It
#: leaves the notebook's ``torch.sum`` intact and normalises only what
#: happens AFTER the sum, which no source specifies -- the lowest
#: fidelity cost available for the measured cause (|f_t + v_a| ~ 21).
#: **(b)** The zero-weight classifier init is dropped for ``nn.Linear``'s
#: own kaiming-uniform (bound 1/sqrt(1024) ~ 0.031); the Laplace bias
#: stays. That init was MINE -- gate 3 translated into cross-entropy --
#: so it is mine to give up, and dropping it removes the epoch-0 constant
#: artifact (``SD_ZERO_HAS_TWO_CAUSES``) as well.
#:
#: **UNTOUCHED, deliberately**: lr **0.01**, their one stated
#: hyperparameter, and the 36-region **sum**, which the notebook
#: disambiguates. Neither clipping nor a discriminative learning rate was
#: added: clipping fills no silence, and lowering lr would overwrite the
#: single number their recipe actually states.
#:
#: **MEASURED UNDER THE FIX, before any launch** (offline, declared
#: substitute for ``roi_align``, ``pretrained=False``, synthetic images,
#: n=48 -- the mechanism class, as before):
#:
#:     shape        logits after step 1   epoch-1 mean loss   predictions
#:     protocol     3.65 -> 4.40          3.868               sd 0.0017, 48 unique
#:     faithful     3.14 -> 6.51          3.480               sd 0.0024, 48 unique
#:
#: Against the pod's pre-fix 3.89 -> 12.75 / 3.94 -> 18.21 and epoch-1
#: means of 4,581 / 1.3e16. **The divergence is gone**: logits stay in
#: single figures, losses stay in single figures across epochs 1-3
#: (protocol 3.87 / 2.33 / 3.81; faithful 3.48 / 3.76 / 3.59), and no
#: prediction vector collapses.
#:
#: **The caveat, stated rather than glossed**: those losses sit ABOVE
#: ln(5) = 1.609, the uniform-guess level, and oscillate rather than
#: descend. On synthetic noise with no signal that is expected -- there is
#: nothing to learn, so a loss cannot fall -- but it means this
#: measurement establishes only that the mechanism class is FIXED, not
#: that the arm trains well. The pod's probe on real features is the test
#: of that, and it runs before the arm does.
FUSED_NORM_AND_STANDARD_INIT = {
    "applied": "2026-08-17, chosen on the figures",
    "fix_a": (
        "LayerNorm(1024) on the fused feature immediately before the "
        "classifier -- leaves the notebook's torch.sum intact, normalises "
        "only what no source specifies"
    ),
    "fix_b": (
        "the classifier's weight keeps nn.Linear's own kaiming-uniform "
        "(bound ~0.031); the Laplace bias stays. The zero-weight init was "
        "mine to give up, and dropping it removes the epoch-0 constant "
        "artifact"
    ),
    "untouched": (
        "lr 0.01 -- their one stated hyperparameter",
        "the 36-region sum -- the notebook disambiguates it",
        "no clipping (fills no silence), no discriminative lr",
    ),
    "measured_under_the_fix": {
        "protocol": {
            "logits_after_step_1": (3.65, 4.40),
            "epoch_1_mean_loss": 3.868,
            "predictions": "sd 0.0017, 48 unique of 48",
            "epochs_1_to_3": (3.868, 2.330, 3.813),
        },
        "faithful": {
            "logits_after_step_1": (3.14, 6.51),
            "epoch_1_mean_loss": 3.480,
            "predictions": "sd 0.0024, 48 unique of 48",
            "epochs_1_to_3": (3.480, 3.759, 3.587),
        },
        "against_pre_fix_pod": "3.89->12.75 / 3.94->18.21; 4,581 / 1.3e16",
    },
    "verdict": "the divergence is gone; nothing collapses; nothing is non-finite",
    "caveat": (
        "the losses sit ABOVE ln(5)=1.609 and oscillate rather than "
        "descend -- expected on synthetic noise where nothing can be "
        "learned, but it means this establishes the mechanism class is "
        "fixed, NOT that the arm trains well. The pod probe on real "
        "features is that test, and it runs before the arm does"
    ),
    "parameters": {"trainable_now": 5780997, "added_by_layernorm": 2048},
}



#: **[OBSERVED 2026-08-17] BOTH ARMS COMPLETE UNDER THE FIX -- AND THE
#: NUMBERS ARE NOT AN ARCHITECTURE RESULT.**
#:
#: **Protocol arm** (``p10_cleftgnn__a3f9b627__p10-cleftgnn-4``):
#: expected-value PCC by seed **0.0071, 0.0637, -0.0269, -0.0327,
#: 0.0041**; top-1 similar. **Faithful arm** (``...faithful-2``): nan PCC
#: on every rater again -- constant test predictions -- with selected
#: epochs 4/40/2/40/40 and F1 0.077-0.154.
#:
#: **The prediction spread refutes reading those PCCs as capacity.**
#: Per-seed prediction sd **0.0658 / 0.1300 / 0.4216 / 0.2120 / 0.0320**
#: against the labels' sd **0.6587** (labels 1.4-4.6). The predictions
#: never leave ~2.5-3.7 and the worst seed is pinned inside
#: **2.907-2.999**. A **13x sd difference across seeds of the same
#: config** means the fit is initialisation-dominated, not
#: data-determined.
#:
#: **FORBIDDEN, and this is the point of the record**: "CleftGNN scores
#: zero on this cohort" may NOT be said, written, or implied from these
#: figures. A correlation computed between the labels and a predictor
#: that spans a twentieth of their range is undefined in practice -- the
#: statistic is arithmetically defined but measures the residual jitter
#: of a collapsed predictor, not the architecture's capacity. The honest
#: sentence is: **the replication does not yet produce a predictor whose
#: outputs vary with the input, so its correlation with anything is not
#: interpretable.**
#:
#: **What the run DOES establish**: the divergence fix holds -- both arms
#: train to completion, nothing goes non-finite, no loss explodes. That
#: was the previous blocker and it is cleared.
ARMS_COMPLETE_BUT_COLLAPSED = {
    "observed": "2026-08-17",
    "protocol_arm": {
        "run": "p10_cleftgnn__a3f9b627__p10-cleftgnn-4",
        "pcc_by_seed": (0.0071, 0.0637, -0.0269, -0.0327, 0.0041),
        "prediction_sd_by_seed": (0.0658, 0.1300, 0.4216, 0.2120, 0.0320),
        "label_sd": 0.6587,
        "label_range": (1.4, 4.6),
        "prediction_range": "~2.5-3.7; the worst seed pinned at 2.907-2.999",
        "seed_sd_ratio": "13x across seeds of the same config",
    },
    "faithful_arm": {
        "run": "p10_cleftgnn_faithful-2",
        "pcc": "nan on every rater -- constant test predictions",
        "selected_epochs": (4, 40, 2, 40, 40),
        "f1_macro": (0.077, 0.154),
    },
    "forbidden": (
        "'CleftGNN scores zero on this cohort' may NOT be said, written "
        "or implied from these figures: a correlation against a predictor "
        "spanning a twentieth of the label range measures the residual "
        "jitter of a collapsed predictor, not the architecture's capacity"
    ),
    "honest_sentence": (
        "the replication does not yet produce a predictor whose outputs "
        "vary with the input, so its correlation with anything is not "
        "interpretable"
    ),
    "what_it_does_establish": (
        "the divergence fix holds: both arms train to completion, nothing "
        "non-finite, no loss explodes -- the previous blocker is cleared"
    ),
    "ledger_status": "HELD UNLEDGERED -- see LEDGER_HOLD_REASONS",
}


#: **[DECIDED 2026-08-17, the builder's call with reasons] THE COLLAPSED
#: ARMS ARE HELD UNLEDGERED.**
#:
#: The ledger's own charter says an entry is "one CLAIM -- an evidential
#: statement with a status", indexing "what the write-up will assert,
#: withdraw, or void". These runs are none of those. They are the
#: diagnostic state of an unfinished build.
#:
#: **Three reasons, in order of weight:**
#:
#: 1. **A ledger row is a citable object.** Entering "PCC 0.002" with
#:    caveats still creates a line whose headline number invites exactly
#:    the reading ``ARMS_COMPLETE_BUT_COLLAPSED`` forbids. Caveats have
#:    lost that fight before in this project -- ``BEST_ARM``'s 1.05x
#:    margin needed a withdrawal, not a footnote.
#: 2. **Nothing is being asserted or withdrawn.** There is no claim to
#:    withdraw, because none was made; and the runs are not VOID -- they
#:    completed, they are diagnostically valuable, and calling them void
#:    would discard the very measurement that identified the collapse.
#: 3. **Precedent.** ``FAITHFUL_ARM_OBSERVED``'s regime finding was held
#:    unledgered for the same structural reason (confounded), and
#:    consistency is what makes the ledger readable.
#:
#: **What WOULD be ledgered, and when**: a replication whose predictions
#: span the label range -- claimable or withdrawn on the standard
#: criterion, whichever the numbers give; or a VOID entry if the build is
#: abandoned. Until one of those, the phase records carry the state and
#: the ledger stays clean.
LEDGER_HOLD_REASONS = {
    "decided": "2026-08-17, the builder's call",
    "status": "the collapsed arms are HELD UNLEDGERED",
    "reasons": (
        "a ledger row is a citable object, and a headline 0.002 invites "
        "the reading the record forbids -- caveats have lost that fight "
        "before (BEST_ARM's 1.05x needed a withdrawal, not a footnote)",
        "nothing is asserted or withdrawn, and the runs are not VOID: "
        "they completed and they carry the measurement that found the "
        "collapse",
        "precedent -- the faithful arm's regime finding is held "
        "unledgered for the same structural reason",
    ),
    "what_would_be_ledgered": (
        "a replication whose predictions span the label range (claimable "
        "or withdrawn on the standard criterion), or a VOID if the build "
        "is abandoned"
    ),
}


#: **[MEASURED 2026-08-17] THE COLLAPSE IS THE HEAD, NOT THE DATA -- and
#: the image-to-image variation dies at ONE identifiable stage.**
#:
#: Four candidates were put up; three are answered offline (declared
#: substitute for ``roi_align``, ``pretrained=False``, synthetic images --
#: but the decisive control is a WITHIN-PROBE contrast, so the caveats
#: fall on both sides equally).
#:
#: **(d) THE LINEAR-HEAD CONTROL -- the cleanest one, and it is
#: decisive.** On a target that IS learnable (grade encoded in pixel
#: brightness, the ``test_pretrain`` fixture trick), with the SAME frozen
#: ResNet-50 features and the SAME recipe:
#:
#:     head          epoch 1 -> 4 loss        prediction sd   classes used
#:     plain linear  1.5382 -> 0.9512 (falls) 0.169 -> 0.351  all five
#:     CleftGNN      2.7821 -> 4.2938 (swings) 0.010 -> 0.037 ONE, swinging
#:
#: A plain linear head on the same features **learns**; the CleftGNN head
#: does not. Its predictions barely vary across images (sd ~0.02) while
#: the whole batch swings between single classes epoch to epoch -- and it
#: behaves **almost identically on pure noise** (loss 2.81/7.09/5.61/5.38,
#: same single-class swings). Its output is essentially input-independent.
#:
#: **(a) CLASS IMBALANCE IS NOT THE DRIVER.** The control above ran a
#: PERFECTLY BALANCED 8/8/8/8/8 target and collapsed identically.
#: Imbalance may worsen the picture on real data -- the raw prior's
#: expected grade is 2.7342 and the Laplace prior's 2.7397, both near
#: where the pod's worst seed pinned -- but it is not necessary for the
#: collapse, so it cannot be the cause.
#:
#: **WHERE THE SIGNAL DIES**, measured as across-image sd relative to
#: each stage's own magnitude (ratio; higher = more image-dependence):
#:
#:     input pixels                    1.178
#:     frozen ResNet-50 feature map    1.171
#:     region vectors (36 x 2048)      0.899
#:     region_proposer -> 512          0.807
#:     gnn_mlp -> 1024                 0.689
#:     APPNP (0.3 self + 0.7 nodemean) 0.648
#:     sigmoid -> f_hat                0.017   <-- 39x collapse
#:     gated pooling, summed over 36   0.020
#:     SABM v_a, summed over 36        0.786   <-- this path survives
#:     fused f_t + v_a                 0.348
#:     after LayerNorm                 0.342
#:     logits                          0.063
#:
#: **The sigmoid at K == 1 is where it goes.** The APPNP output has
#: magnitude ~0.05 -- close to zero -- and ``sigmoid(x) ~ 0.5 + x/4``
#: there, so the branch emits a near-constant 0.5 plus a twentieth of the
#: variation it received. Summing 36 of those makes ``f_t`` a LARGE
#: NEARLY-CONSTANT vector (this is also why |f_t| measured ~21 on the
#: pod), which then dilutes the one path that did preserve
#: image-dependence -- the SABM's ``v_a`` at 0.786. The fused feature
#: ends at 0.348 and the logits at 0.063: the expected grade spans
#: 3.07-3.41 for images whose true grades span 1-5.
#:
#: **The sigmoid is THEIRS** -- ``if self.propagate.K == 1: Y =
#: torch.sigmoid(Y)`` is the notebook's own line -- so this is a finding
#: about the architecture as shared, not a porting error. Whether their
#: own pre-activations sit near zero depends on their feature scale
#: (frozen ViT, different magnitudes), which is exactly what the pod
#: probe's stage table now measures on real pretrained features.
#: **Thursday-flagged.**
#:
#: **(b) and (c) remain OPEN**: the post-LayerNorm scale is no longer a
#: divergence risk but the branch imbalance above is a scale finding of
#: its own; and under-training vs over-regularisation is readable only
#: from the pod's inner-val curves -- the faithful arm's 4/40/2/40/40
#: selections suggest both regimes occur across raters.
COLLAPSE_DIAGNOSED = {
    "measured": "2026-08-17, offline, declared substitute",
    "d_linear_head_control": {
        "verdict": "DECISIVE -- a plain linear head learns where this head does not",
        "linear": {
            "loss_1_to_4": (1.5382, 1.2740, 1.1108, 0.9512),
            "prediction_sd": (0.169, 0.230, 0.238, 0.351),
            "classes_used": "all five by epoch 3",
        },
        "cleftgnn": {
            "loss_1_to_4": (2.7821, 6.9921, 5.5268, 4.2938),
            "prediction_sd": (0.019, 0.025, 0.037, 0.005),
            "classes_used": "ONE at a time, swinging 1/1/5/1",
        },
        "on_pure_noise": (
            "almost identical -- 2.81/7.09/5.61/5.38, same single-class "
            "swings: the head's output is essentially input-independent"
        ),
        "why_the_caveats_cancel": (
            "a WITHIN-PROBE contrast: both heads see the same frozen "
            "features under the same substitute and the same recipe"
        ),
    },
    "a_class_imbalance_not_the_driver": (
        "the control ran a perfectly balanced 8/8/8/8/8 target and "
        "collapsed identically; imbalance may worsen it (raw prior "
        "expected grade 2.7342, Laplace 2.7397, near the pod's pinned "
        "worst seed) but is not necessary for it"
    ),
    "stage_variance_ratio": {
        "input": 1.178, "resnet_map": 1.171, "region_vectors": 0.899,
        "region_proposer": 0.807, "gnn_mlp": 0.689, "appnp": 0.648,
        "sigmoid_f_hat": 0.017, "gated_pooling_f_t": 0.020,
        "sabm_v_a": 0.786, "fused": 0.348, "after_layernorm": 0.342,
        "logits": 0.063,
    },
    "where_it_dies": (
        "the K==1 sigmoid: APPNP output magnitude ~0.05, where "
        "sigmoid(x) ~ 0.5 + x/4, so the branch emits a near-constant 0.5 "
        "and a twentieth of the variation; summing 36 makes f_t a large "
        "nearly-constant vector that dilutes the SABM path (0.786), which "
        "is the only one preserving image-dependence"
    ),
    "the_sigmoid_is_theirs": (
        "'if self.propagate.K == 1: Y = torch.sigmoid(Y)' is the "
        "notebook's own line -- a finding about the architecture as "
        "shared, not a porting error; Thursday-flagged"
    ),
    "b_and_c_open": (
        "post-LayerNorm scale is no longer a divergence risk but the "
        "branch imbalance is a scale finding of its own; under-training "
        "vs over-regularisation needs the pod's inner-val curves -- the "
        "faithful arm's 4/40/2/40/40 suggests both regimes occur"
    ),
}


#: **[PROPOSED 2026-08-17, NOT PICKED] Four routes out of the collapse,
#: with fidelity costs -- and the measurement each one needs first.**
#:
#: 1. **Confirm on the pod before anything.** The stage table above was
#:    measured with a randomly-initialised backbone through a substitute
#:    op; their pre-activation scale depends on real pretrained features.
#:    ``scripts/p10_scale_probe.py`` now prints the same table on real
#:    data. **Cost: none. Nothing should be changed before this runs.**
#: 2. **Drop or replace the K==1 sigmoid.** Fidelity cost: **HIGH** --
#:    it is the notebook's own line, one of the few places their code
#:    disambiguates the paper.
#: 3. **Normalise the GNN branch** (LayerNorm on ``f_hat`` or on ``f_t``
#:    before fusion). Fidelity cost: **MEDIUM** -- it leaves their
#:    sigmoid and their sum intact and normalises between them, but it
#:    changes the branch balance their eq. (9) states as a plain sum.
#: 4. **Widen the pre-activation** (scale ``gnn_mlp``'s output init so
#:    the sigmoid operates off its flat midpoint). Fidelity cost:
#:    **LOW** -- no source specifies an init -- but it is a knob chosen
#:    to produce an outcome, which needs registering as such.
#:
#: Not proposed: lowering lr 0.01 (their one stated hyperparameter) or
#: clipping (fills no silence) -- both remain out for the reasons already
#: registered in ``REGISTERED_DEVIATIONS``.
COLLAPSE_ROUTES_PROPOSED = {
    "proposed": "2026-08-17, not picked",
    "routes": (
        "confirm the stage table on the pod's real features first -- cost "
        "NONE, and nothing should change before it runs",
        "drop or replace the K==1 sigmoid -- fidelity cost HIGH, it is "
        "the notebook's own line",
        "normalise the GNN branch before fusion -- fidelity cost MEDIUM, "
        "leaves the sigmoid and the sum intact but changes the branch "
        "balance eq. (9) states as a plain sum",
        "widen the pre-activation via the gnn_mlp init -- fidelity cost "
        "LOW (no source specifies an init) but it is a knob chosen for an "
        "outcome and must be registered as one",
    ),
    "still_out": "lr 0.01 and clipping, for the reasons already registered",
}



#: **[CONFIRMED 2026-08-17 on real pretrained features] The stage table
#: holds -- sharper than offline -- and it makes a SECOND finding
#: explicit.**
#:
#:     input                            0.259
#:     frozen backbone feature map      2.580
#:     region vectors                   1.483
#:     region_proposer -> 512           1.018
#:     gnn_mlp -> 1024                  0.660
#:     APPNP  (|mean| 0.0233)           0.5025
#:     sigmoid -> f_hat                 0.0058   <-- 87x kill
#:       (|mean| 0.4998, sd 0.0029)
#:     gated pooling, summed over 36    0.0068
#:       (|mean| 4.2731)
#:     SABM v_a, summed over 36         0.7794
#:       (|mean| 0.7323)
#:     fused f_t + v_a                  0.1315
#:     logits                           0.0214
#:
#: **Finding 1, confirmed**: the K==1 sigmoid is the point of failure.
#: APPNP's output has |mean| **0.0233** -- the sigmoid sits at its linear
#: midpoint, emitting 0.4998 +/- 0.0029, and 87x of the branch's relative
#: image-dependence is destroyed in one operation.
#:
#: **Finding 2, which only the magnitudes make visible**: the GNN branch
#: is not merely uninformative, it is **~5.8x LARGER than the SABM
#: branch** (|f_t| 4.2731 against |v_a| 0.7323). Equation (9) adds them
#: plainly, so a nearly-constant vector six times the size of the
#: informative one **drowns it**: v_a arrives carrying ratio 0.7794 and
#: the fused feature leaves at 0.1315. The dilution, not the compression,
#: is what the classifier actually sees -- which is why the route out is
#: aimed at the sum rather than at the sigmoid.
STAGE_TABLE_CONFIRMED = {
    "confirmed": "2026-08-17, real pretrained features, pod",
    "ratios": {
        "input": 0.259, "backbone_map": 2.580, "region_vectors": 1.483,
        "region_proposer": 1.018, "gnn_mlp": 0.660, "appnp": 0.5025,
        "sigmoid_f_hat": 0.0058, "gated_pooling_f_t": 0.0068,
        "sabm_v_a": 0.7794, "fused": 0.1315, "logits": 0.0214,
    },
    "magnitudes": {
        "appnp_mean": 0.0233, "f_hat_mean": 0.4998, "f_hat_sd": 0.0029,
        "f_t_mean": 4.2731, "v_a_mean": 0.7323,
    },
    "finding_1_sigmoid": (
        "87x kill: APPNP's |mean| 0.0233 puts the sigmoid at its linear "
        "midpoint, so the branch emits 0.4998 +/- 0.0029"
    ),
    "finding_2_imbalance": (
        "the GNN branch is ~5.8x LARGER than the SABM branch (4.2731 vs "
        "0.7323) and eq. (9) adds them plainly, so a nearly-constant "
        "vector drowns the informative one: 0.7794 in, 0.1315 out"
    ),
    "why_it_directs_the_fix": (
        "the dilution, not the compression, is what the classifier sees "
        "-- so the route out is aimed at the sum, not at the sigmoid"
    ),
}


#: **[REGISTERED DEVIATION 2026-08-17, route 3, chosen on the
#: measurement] Normalise the GNN branch immediately before eq. (9)'s
#: sum.**
#:
#: **The form, derived from the table rather than defaulted**:
#: ``nn.LayerNorm(1024, elementwise_affine=False)`` applied to ``f_t``,
#: with ``v_a`` untouched. Four properties, each answering something the
#: table says:
#:
#: 1. **Per-sample, not per-batch.** The failure is a magnitude
#:    imbalance, which a BatchNorm would also fix -- but BatchNorm
#:    carries running statistics, and missing BN running statistics is
#:    the defect that produced two void ladders in this project. A
#:    per-sample norm has no such state.
#: 2. **On ``f_t``, after the sum, not on ``f_hat`` before it.** The
#:    measured imbalance exists at the branch OUTPUT (4.2731 vs 0.7323);
#:    normalising per-region would change what ``torch.sum`` sums, and
#:    the sum is the notebook's own line.
#: 3. **No learnable affine.** A learnable scale could restore precisely
#:    the imbalance being corrected, and no source asks for one here.
#:    The correction is meant to be structural, not something training
#:    can undo.
#: 4. **``v_a`` is NOT normalised.** It is the branch that works (0.7794)
#:    and there is no measured justification for touching it. One factor
#:    moves.
#:
#: **What it does NOT do, stated so the prediction is honest**: it does
#: not fix the sigmoid. The GNN branch's own ratio stays ~0.006-0.02 --
#: the compression is untouched. This fix addresses the DILUTION only.
#:
#: **Fidelity cost: MEDIUM.** Equation (9) states a plain sum, and this
#: changes the balance of that sum. Weighed against the alternatives:
#: dropping the sigmoid would overwrite the notebook's own line (HIGH),
#: and widening the pre-activation would be a knob chosen for an outcome
#: (LOW cost but unprincipled). The sigmoid line, ``torch.sum``, lr 0.01
#: and the region set are all untouched.
#:
#: **MEASURED OFFLINE UNDER THE FIX** (declared substitute,
#: ``pretrained=False`` -- the mechanism class):
#:
#:     stage                     before    after
#:     gnn branch |mean|         4.2702    0.7937   <- equalised
#:     gnn branch ratio          0.0202    0.0202   <- unchanged, as predicted
#:     fused ratio               0.3477    0.7391
#:     logits ratio              0.0634    0.1616
#:     expected-grade sd         0.128     0.246
#:
#: The branch ratio is unchanged and the fused ratio doubles to v_a's own
#: 0.786 -- exactly the signature of a dilution fix rather than a
#: compression fix.
#:
#: **PRE-REGISTERED EXPECTATION ON THE POD.** The pod's imbalance is
#: WORSE than offline (5.8x against 2.1x), so equalising should help
#: MORE there: the fused ratio should land in **0.4-0.7** (from 0.1315),
#: the logits ratio should rise several-fold from 0.0214, and the GNN
#: branch's own ratio should stay near 0.0068. If the fused ratio does
#: not clear ~0.4, the dilution was not the binding constraint and the
#: sigmoid is.
GNN_BRANCH_NORM = {
    "registered": "2026-08-17, route 3, chosen on the measurement",
    "form": "nn.LayerNorm(1024, elementwise_affine=False) on f_t, before eq. (9)",
    "why_this_form": (
        "per-sample, not per-batch: a BatchNorm would also equalise but "
        "carries running statistics, the defect behind two void ladders",
        "on f_t after the sum, not on f_hat before it: the measured "
        "imbalance is at the branch output, and per-region normalisation "
        "would change what the notebook's torch.sum sums",
        "no learnable affine: a learnable scale could restore the very "
        "imbalance being corrected, and no source asks for one",
        "v_a is NOT normalised -- it is the branch that works, and one "
        "factor moves",
    ),
    "what_it_does_not_do": (
        "it does not fix the sigmoid; the GNN branch's own ratio stays "
        "~0.006-0.02. This addresses the DILUTION only"
    ),
    "fidelity_cost": (
        "MEDIUM -- eq. (9) states a plain sum and this changes its "
        "balance; weighed against dropping the sigmoid (HIGH, the "
        "notebook's own line) and widening the pre-activation (LOW cost "
        "but a knob chosen for an outcome). Sigmoid, torch.sum, lr 0.01 "
        "and the region set are untouched"
    ),
    "measured_offline": {
        "gnn_branch_magnitude": (4.2702, 0.7937),
        "gnn_branch_ratio": (0.0202, 0.0202),
        "fused_ratio": (0.3477, 0.7391),
        "logits_ratio": (0.0634, 0.1616),
        "expected_grade_sd": (0.128, 0.246),
    },
    "pre_registered_expectation": (
        "the pod's imbalance is worse (5.8x vs 2.1x), so equalising "
        "should help more: fused ratio 0.4-0.7 from 0.1315, logits "
        "several-fold above 0.0214, GNN branch ratio still ~0.0068. If "
        "fused does not clear ~0.4, the dilution was not the binding "
        "constraint and the sigmoid is"
    ),
    "success_criteria": (
        "post-fix stage ratios with fused and logits in the SAME ORDER as "
        "SABM's 0.78 -- not 0.13 and 0.02",
        "prediction sd on a real fold in the order of the labels' 0.6587 "
        "-- not 0.03-0.42",
    ),
}


#: **[COMMITTED BOTH WAYS 2026-08-17, before the run] What the normalised
#: arm's result means, either way.**
#:
#: **If the predictions span the label range and the PCC is still ~0**:
#: that IS an architecture measurement. A predictor that varies with its
#: input and still fails to correlate with the panel mean has been given
#: its chance on this cohort, and the number becomes **LEDGERABLE** under
#: the standard criterion -- claimable or withdrawn as the paired BCa
#: gives it. This is the outcome that would let Phase 10 answer the
#: question it was opened to answer.
#:
#: **If it still collapses**: then the replication cannot produce an
#: interpretable predictor on this cohort, and **that is the finding** --
#: with the stage table as its mechanism, the K==1 sigmoid named as the
#: point of failure, and the 5.8x branch imbalance as what made it fatal.
#: It would be reported as a replication limit with a measured cause, not
#: as a performance figure, and the forbidden sentence
#: (``ARMS_COMPLETE_BUT_COLLAPSED``) would still stand.
#:
#: Both readings are committed now, before the run, so neither can be
#: chosen after the numbers arrive. The current collapsed runs stay
#: UNLEDGERED either way (``LEDGER_HOLD_REASONS``).
NORMALISED_ARM_READINGS = {
    "committed": "2026-08-17, before the run",
    "if_spans_and_pcc_near_zero": (
        "an ARCHITECTURE MEASUREMENT -- a predictor that varies with its "
        "input and still does not correlate has had its chance on this "
        "cohort; LEDGERABLE under the standard criterion, claimable or "
        "withdrawn as the paired BCa gives it"
    ),
    "if_still_collapses": (
        "the replication cannot produce an interpretable predictor on "
        "this cohort, and THAT is the finding -- with the stage table as "
        "its mechanism, the K==1 sigmoid named, and the 5.8x imbalance as "
        "what made it fatal. A replication limit with a measured cause, "
        "never a performance figure"
    ),
    "either_way": (
        "the current collapsed runs stay unledgered (LEDGER_HOLD_REASONS) "
        "and the forbidden sentence stands"
    ),
}



#: **[DEFECT 2026-08-17, found by the maintainer in the pod's run-5 output]
#: THE PROBE REBUILT THE PIPELINE BY HAND, AND THE REBUILD FELL BEHIND
#: THE MODEL.**
#:
#: ``stage_variance`` computed ``f_t`` itself and fused with
#: ``f_t + v_a``, never calling ``gnn_norm`` -- which the model had gained
#: at line 179 and applies at line 234. So pod run 5, the run whose whole
#: purpose was to measure route 3, **printed figures identical to run 4 to
#: four decimals** (fused ratio 0.1315, unchanged). The instrument was
#: describing an architecture the model was not running.
#:
#: **The defect class is one this project has recorded before**: two
#: implementations of one computation, drifting apart silently. It is the
#: same shape as ``ctx.write_metrics``, ``extractor.preprocess`` and the
#: ``phase3.run_cv`` call that never existed -- and the same shape as the
#: 36-vs-27 region count, where a second statement of the same fact
#: disagreed with the first.
#:
#: **The fix removes the second implementation rather than syncing it.**
#: The model now records its own intermediates when handed a dict
#: (``forward(x, stages=...)``, observation-only, default None -- the
#: ``on_step``/``on_epoch`` precedent), and the probe READS them. A stage
#: renamed in the model now fails the probe loudly instead of vanishing
#: from the table.
#:
#: **VERIFICATION, because the diagnosis rested on the broken
#: instrument**: every stage of the old reconstruction was compared
#: against the model's own forward, same inputs, same substitute op.
#:
#:     input, backbone_map, region_vectors, region_proposer, gnn_mlp,
#:     appnp, sigmoid_f_hat, gated_pooling_f_t, sabm_v_a
#:         -> max |difference| 0.000e+00, ALL NINE BITWISE IDENTICAL
#:     fused, after_layernorm, logits
#:         -> DIVERGED (1.698e+01, 2.728e+00, 1.074e+00)
#:
#: **So the diagnosis stands.** The 87x sigmoid kill and the 5.8x branch
#: imbalance are measured on ``appnp``, ``sigmoid_f_hat``,
#: ``gated_pooling_f_t`` and ``sabm_v_a`` -- four stages the
#: reconstruction computed EXACTLY. The three that diverged are precisely
#: the three downstream of the missing call, and they diverged by exactly
#: what ``gnn_norm`` does (|f_t| 4.2704 against |gnn_norm(f_t)| 0.7937).
#: One defect, three affected rows, nothing else touched. Route 3 rests
#: on arithmetic that was right.
PROBE_RECONSTRUCTED_THE_PIPELINE = {
    "found": "2026-08-17, by the maintainer, in the pod's run-5 output",
    "defect": (
        "stage_variance rebuilt the pipeline by hand and never called "
        "gnn_norm, so run 5 printed run 4's figures to four decimals "
        "(fused ratio 0.1315 unchanged) -- the instrument described an "
        "architecture the model was not running"
    ),
    "defect_class": (
        "two implementations of one computation, drifting silently -- the "
        "ctx.write_metrics / extractor.preprocess / phase3.run_cv shape, "
        "and the 36-vs-27 region count's shape"
    ),
    "fix": (
        "the second implementation is REMOVED, not synced: the model "
        "records its own intermediates via forward(x, stages=...) "
        "(observation-only, default None -- the on_step/on_epoch "
        "precedent) and the probe reads them; a renamed stage now fails "
        "loudly instead of vanishing from the table"
    ),
    "verification": {
        "bitwise_identical": (
            "input", "backbone_map", "region_vectors", "region_proposer",
            "gnn_mlp", "appnp", "sigmoid_f_hat", "gated_pooling_f_t",
            "sabm_v_a",
        ),
        "diverged": {
            "fused": 1.698e01, "after_layernorm": 2.728e00,
            "logits": 1.074e00,
        },
        "tolerance": 1e-5,
    },
    "diagnosis_stands": (
        "the 87x sigmoid kill and the 5.8x imbalance are measured on four "
        "stages the reconstruction computed BITWISE exactly; the three "
        "that diverged are exactly the three downstream of the missing "
        "call, differing by exactly what gnn_norm does (|f_t| 4.2704 vs "
        "|gnn_norm(f_t)| 0.7937). One defect, three rows, nothing else"
    ),
    "cost": "one pod run (run 5) spent measuring the wrong architecture",
}


#: **[EXPECTED 2026-08-17, handed back for the pod] The post-fix stage
#: table's SHAPE, so the next run is read against a stated prediction
#: rather than an impression.**
#:
#: With the instrument reading the model's own forward, the table gains a
#: ``gnn_norm_f_t`` row between ``gated_pooling_f_t`` and ``sabm_v_a``.
#: Everything upstream of the fusion is UNCHANGED by route 3 -- it must
#: reprint within noise of run 4, and a departure there would mean
#: something else moved:
#:
#:     input                 0.259     unchanged
#:     backbone_map          2.580     unchanged
#:     region_vectors        1.483     unchanged
#:     region_proposer       1.018     unchanged
#:     gnn_mlp               0.660     unchanged
#:     appnp                 0.5025    unchanged   (|mean| 0.0233)
#:     sigmoid_f_hat         0.0058    unchanged   (the 87x kill stands)
#:     gated_pooling_f_t     0.0068    unchanged   (|mean| 4.2731)
#:     gnn_norm_f_t          ~0.0068   NEW ROW, |mean| ~0.79
#:     sabm_v_a              0.7794    unchanged   (|mean| 0.7323)
#:     fused                 0.45-0.60 was 0.1315  <-- the test
#:     after_layernorm       ~fused's  was 0.1315-ish
#:     logits                0.05-0.15 was 0.0214
#:
#: **The fused band is computed, not guessed**: with the GNN branch
#: normalised to |mean| ~0.79 it no longer outweighs ``v_a``'s 0.7323, so
#: the fused across-image sd is essentially ``v_a``'s own (0.7794 x
#: 0.7323 = 0.571) over a fused magnitude near 1.0-1.2 -- a ratio of
#: roughly 0.48-0.57, inside the pre-registered 0.4-0.7.
#:
#: **The falsifier is unchanged**: if fused does not clear ~0.4, the
#: dilution was not the binding constraint and the sigmoid is -- and
#: route 3 should be reported as measured-and-insufficient rather than
#: quietly followed by route 2.
EXPECTED_POST_FIX_TABLE = {
    "expected": "2026-08-17, handed back for the pod",
    "unchanged_upstream": {
        "input": 0.259, "backbone_map": 2.580, "region_vectors": 1.483,
        "region_proposer": 1.018, "gnn_mlp": 0.660, "appnp": 0.5025,
        "sigmoid_f_hat": 0.0058, "gated_pooling_f_t": 0.0068,
        "sabm_v_a": 0.7794,
    },
    "new_row": {"gnn_norm_f_t": "ratio ~0.0068, |mean| ~0.79"},
    "predicted": {
        "fused": (0.45, 0.60), "logits": (0.05, 0.15),
    },
    "how_the_band_was_computed": (
        "normalised the GNN branch sits at |mean| ~0.79 against v_a's "
        "0.7323, so fused across-image sd is essentially v_a's own "
        "(0.7794 x 0.7323 = 0.571) over a fused magnitude near 1.0-1.2"
    ),
    "falsifier": (
        "if fused does not clear ~0.4 the dilution was not binding and "
        "the sigmoid is -- report route 3 as measured-and-insufficient "
        "rather than quietly moving to route 2"
    ),
    "also_a_check_on_the_instrument": (
        "every upstream row must reprint within noise of run 4; a "
        "departure there means something other than route 3 moved"
    ),
}



#: **[OBSERVED 2026-08-17, route 3] NEITHER REGISTERED READING APPLIES,
#: and saying so is the first honest act.**
#:
#: ``NORMALISED_ARM_READINGS`` committed two branches: predictions
#: SPANNING the label range with PCC ~0 (an architecture measurement,
#: ledgerable), or STILL COLLAPSING (a replication limit). The run is
#: neither, so **neither reading is applied verbatim** -- a third reading
#: is proposed below and labelled as post-hoc, which is what it is.
#:
#: **Primary arm** (``p10-cleftgnn-5``): expected-value PCC by seed
#: 0.0069, 0.0680, -0.0138, -0.0555, 0.1151 -- **mean +0.0241 with a
#: seed sd of 0.0676**. Prediction sd 0.1024, 0.2222, 0.2034, 0.1267,
#: 0.0351 against the labels' 0.6587; ranges from [2.865, 3.013] (seed
#: 99, width 0.148) to [1.865, 3.548] (seed 1337, width 1.683).
#:
#: **The registered success criterion (ii) FAILS, measurably.** Spread
#: "in the order of the labels' 0.6587" is not reached: the best seed is
#: **33.7%** of label sd and the mean seed is **20.9%**.
#:
#: **Criterion (i) IS UNREPORTED.** The stage table for this run was not
#: pasted, so whether ``fused`` cleared the pre-registered 0.4-0.7 is
#: unknown -- and that is exactly the falsifier
#: (``EXPECTED_POST_FIX_TABLE``) that decides whether the dilution was
#: the binding constraint. **The arm's own numbers cannot answer it**;
#: ``p10_scale_probe.py --stages`` can, in seconds, and should be read
#: before Phase 10 concludes anything about route 3's mechanism.
#:
#: **What route 3 measurably did to the arm** -- and it is not what the
#: fix was aimed at:
#:
#:     run 4 -> run 5    mean prediction sd   0.1723 -> 0.1380  (-19.9%)
#:                       seed spread ratio    13.17x -> 6.33x
#:                       widest seed sd       0.4216 -> 0.2222
#:
#: The mean spread went DOWN by a fifth while the seed-to-seed ratio
#: halved. So the normalisation did not buy predictive range; it bought
#: **consistency between seeds** -- a real change, in a quantity nobody
#: registered a criterion for, and one that must not be reported as
#: success against the criterion that failed.
#:
#: **Faithful arm** (``...faithful-3``): four raters still nan on
#: constant predictions with F1 0.077-0.154; **Rater 11 - Psychologist
#: broke free at epoch 33** -- F1 0.3352 study / 0.2366 benchmark, PCC
#: -0.1287 study / +0.0663 benchmark
#: (``PSYCHOLOGIST_CELL_IS_THE_INSTABILITY``).
ROUTE_3_OBSERVED = {
    "observed": "2026-08-17",
    "run": "p10-cleftgnn-5",
    "neither_reading_applies": (
        "the run is neither spanning nor collapsed, so NORMALISED_ARM_"
        "READINGS' two branches are both withheld; the third reading is "
        "post-hoc and labelled so"
    ),
    "pcc_by_seed": (0.0069, 0.0680, -0.0138, -0.0555, 0.1151),
    "pcc_mean": 0.0241,
    "pcc_seed_sd": 0.0676,
    "prediction_sd_by_seed": (0.1024, 0.2222, 0.2034, 0.1267, 0.0351),
    "ranges": {
        "narrowest": "seed 99, [2.865, 3.013], width 0.148",
        "widest": "seed 1337, [1.865, 3.548], width 1.683",
    },
    "criterion_ii_fails": (
        "best seed 33.7% of the labels' 0.6587, mean seed 20.9% -- not "
        "'in the order of' it"
    ),
    "criterion_i_unreported": (
        "the stage table for this run was not pasted, so whether fused "
        "cleared the pre-registered 0.4-0.7 is UNKNOWN -- and that is the "
        "falsifier. The arm's numbers cannot answer it; --stages can, in "
        "seconds"
    ),
    "what_route_3_did": {
        "mean_prediction_sd": (0.1723, 0.1380),
        "change": "-19.9%",
        "seed_spread_ratio": (13.17, 6.33),
        "widest_seed_sd": (0.4216, 0.2222),
        "reading": (
            "the normalisation bought CONSISTENCY BETWEEN SEEDS, not "
            "predictive range -- a real change in a quantity nobody "
            "registered a criterion for, and not to be reported as "
            "success against the criterion that failed"
        ),
    },
}


#: **[PROPOSED 2026-08-17, POST HOC AND LABELLED AS SUCH] The honest
#: third reading -- and why it is NOT ledgered yet.**
#:
#: **The reading**: the replication now trains stably and produces
#: varying predictions on some seeds and one rater, but **predictive
#: spread stays well below label spread** (20.9% of it on average, 33.7%
#: at best), **the PCC is indistinguishable from zero in every
#: configuration** (mean +0.0241 against a seed sd of 0.0676 -- and that
#: seed sd is itself the size of a single correlation's sampling error at
#: n=237, ~0.065), and **the GNN branch still contributes nothing**
#: (image-dependence 0.0068), so whatever varies comes from the SABM
#: branch alone.
#:
#: **NOT LEDGERED. Three reasons, and the third is the one that decides
#: it:**
#:
#: 1. **Neither registered trigger fired.** Ledgering on a reading
#:    composed after the numbers arrived is precisely the post-hoc
#:    selection the phase discipline exists to prevent -- the move this
#:    project corrected at ``CLAIMABLY_WORSE_WITHDRAWN`` and again at the
#:    8b criterion switch.
#: 2. **The mechanism is known-incomplete.** The sigmoid still kills the
#:    GNN branch by 87x. A "replication limit" claim quotes a number from
#:    a build whose diagnosed defect is unaddressed; if route 2 were run
#:    and the result held, THAT would be a limit worth ledgering.
#: 3. **The object would be misnamed.** With the GNN branch at 0.0068,
#:    this arm measures a frozen ResNet-50 plus the SABM attention head
#:    -- not CleftGNN. A ledger row saying "CleftGNN" would name a thing
#:    the run did not test, and the ledger's rows are what the write-up
#:    quotes.
#:
#: **What would make it ledgerable**, in order of cheapness: the stage
#: table confirming route 3's mechanism (criterion i); then either a
#: build whose GNN branch carries image-dependence -- so the name is
#: earned -- or an explicit decision to report "ResNet-50 + SABM head on
#: this cohort" under its own honest name, which IS ledgerable as it
#: stands.
THIRD_READING_PROPOSED = {
    "proposed": "2026-08-17, post hoc and labelled as such",
    "reading": (
        "trains stably; varying predictions on some seeds and one rater; "
        "spread 20.9% of label sd on average (33.7% at best); PCC "
        "indistinguishable from zero everywhere (mean +0.0241, seed sd "
        "0.0676, which is itself the size of a single correlation's "
        "sampling error at n=237); the GNN branch contributes 0.0068, so "
        "what varies comes from SABM alone"
    ),
    "ledgered": False,
    "reasons": (
        "neither registered trigger fired -- ledgering a reading composed "
        "after the numbers is the post-hoc selection corrected at "
        "CLAIMABLY_WORSE_WITHDRAWN and at the 8b criterion switch",
        "the mechanism is known-incomplete: the sigmoid still kills the "
        "GNN branch 87x, so a 'replication limit' would quote a build "
        "whose diagnosed defect is unaddressed",
        "the object would be MISNAMED: at 0.0068 the GNN branch is inert, "
        "so this measures a frozen ResNet-50 plus the SABM head, not "
        "CleftGNN -- and ledger rows are what the write-up quotes",
    ),
    "what_would_make_it_ledgerable": (
        "the stage table confirming route 3's mechanism (criterion i)",
        "then either a build whose GNN branch carries image-dependence, "
        "so the name is earned, or an explicit decision to report "
        "'ResNet-50 + SABM head on this cohort' under its own honest "
        "name -- which IS ledgerable as it stands",
    ),
}


#: **[MEASURED 2026-08-17] THE PSYCHOLOGIST CELL IS THE 0.598
#: INSTABILITY, MEASURED ON OUR OWN COHORT.**
#:
#: Rater 11 - Psychologist broke free of the collapse at epoch 33 and
#: produced, from ONE model: **PCC -0.1287 on the 36-image study set and
#: +0.0663 on the 25-image benchmark** -- a swing of **0.195** between
#: two test sets of the same task, with F1 0.3352 and 0.2366.
#:
#: **This is the phenomenon the comparator analysis predicted, now
#: observed rather than argued.** ``CLEFTGNN_COMPARATOR_TABLES`` records
#: their Rater E at 0.598 on the benchmark against 0.283 on the study set
#: -- a swing of 0.315 from one model. Ours swings 0.195. **In both
#: cases the benchmark number is the HIGHER one**, which is suggestive
#: and no more: one rater each is an anecdote, not a pattern, and the
#: direction could be chance.
#:
#: **What it does establish**: at n=25-36, a single model's PCC is not a
#: stable quantity, and a table reporting one such number per cell
#: without an interval is reporting something that moves by 0.2-0.3 when
#: the test set changes. That is the measured context for the 0.598 cell,
#: and it now comes from our own run rather than from Fisher arithmetic
#: alone.
#:
#: **What it does NOT establish**: nothing about their cohort, their
#: splits, or the correctness of their published figures. And our two
#: numbers are both near noise, from a model that barely escaped
#: collapse -- which is why this is recorded as an instance of the
#: regime's behaviour, not as a result about either model.
PSYCHOLOGIST_CELL_IS_THE_INSTABILITY = {
    "measured": "2026-08-17",
    "cell": "Rater 11 - Psychologist, broke free at epoch 33",
    "figures": {
        "pcc_study_n36": -0.1287, "pcc_benchmark_n25": 0.0663,
        "swing": 0.1950,
        "f1_study": 0.3352, "f1_benchmark": 0.2366,
    },
    "their_rater_e": {"benchmark": 0.598, "study": 0.283, "swing": 0.3150},
    "same_direction": (
        "benchmark higher in both cases -- suggestive and no more; one "
        "rater each is an anecdote and the direction could be chance"
    ),
    "establishes": (
        "at n=25-36 a single model's PCC is not a stable quantity: one "
        "model moved 0.195 between two test sets of the same task, so a "
        "table of one such number per cell without an interval reports "
        "something that moves by 0.2-0.3 when the test set changes"
    ),
    "does_not_establish": (
        "nothing about their cohort, splits, or the correctness of their "
        "published figures; and both our numbers are near noise from a "
        "model that barely escaped collapse -- an instance of the "
        "regime's behaviour, not a result about either model"
    ),
}


#: **[PROPOSED 2026-08-17] IS THE SEED INSTABILITY ITSELF THE FINDING --
#: and the one diagnostic that needs NO new run.**
#:
#: The 6.33x spread in prediction sd across seeds of one config is a
#: candidate finding: it says the fit is initialisation-dominated, which
#: with the GNN branch inert means **the SABM path's random init decides
#: how much the model varies at all**. But "initialisation-dominated" and
#: "stopped too early on some seeds" produce the same signature from the
#: outside, and they have opposite consequences -- one is a property of
#: the architecture, the other a property of our early-stopping
#: deviation.
#:
#: **The diagnostic is a READ, not a run.** The completed run already
#: wrote what settles it: ``metrics.json`` carries ``selected_epochs``
#: per fold per seed, and ``curves.csv`` carries the inner-val
#: trajectories. Nothing needs relaunching. **Both readings committed
#: before the numbers:**
#:
#: * **If the narrow seeds stopped early (epochs 1-3) and seed 1337 ran
#:   long**: the early-stopping deviation is implicated, not the
#:   architecture -- the monitor (``inner_val_mse`` on the
#:   expected-value readout) may be flat while the head is still learning
#:   to differentiate. That is a registered-deviation question, and it is
#:   answerable without touching lr or the sigmoid.
#: * **If every seed ran to a similar epoch**: the difference is pure
#:   initialisation, the 6.33x IS the finding, and it says the arm's
#:   behaviour is decided by the SABM branch's random draw.
#:
#: Note the faithful arm already gives a hint in the second direction's
#: favour: its psychologist cell broke free **at epoch 33**, deep into
#: training, while the other four never did -- the same config, the same
#: budget, a different draw.
SEED_INSTABILITY_NEXT_DIAGNOSTIC = {
    "proposed": "2026-08-17",
    "the_question": (
        "6.33x spread in prediction sd across seeds of one config: "
        "initialisation-dominated fitting, or early stopping cutting some "
        "seeds short? The signatures look identical from outside and the "
        "consequences are opposite"
    ),
    "needs_no_new_run": (
        "metrics.json already carries selected_epochs per fold per seed "
        "and curves.csv the inner-val trajectories -- the diagnostic is a "
        "READ of artifacts the completed run already wrote"
    ),
    "if_narrow_seeds_stopped_early": (
        "the early-stopping DEVIATION is implicated, not the "
        "architecture: inner_val_mse on the expected-value readout may be "
        "flat while the head is still learning to differentiate. "
        "Answerable without touching lr or the sigmoid"
    ),
    "if_all_seeds_ran_similar": (
        "pure initialisation: the 6.33x IS the finding, and the arm's "
        "behaviour is decided by the SABM branch's random draw"
    ),
    "existing_hint": (
        "the faithful arm's psychologist cell broke free at EPOCH 33, "
        "deep into training, while four others never did -- same config, "
        "same budget, different draw"
    ),
}



#: **[MEASURED 2026-08-17] THE EPOCHS ANSWER NEITHER COMMITTED READING --
#: and they refute the first one's simple form outright.**
#:
#: ``selected_epochs`` per fold, by seed:
#:
#:     seed     7   [8, 1, 4, 1, 4]     mean 3.60   max  8
#:     seed    99   [1, 2, 1, 26, 2]    mean 6.40   max 26
#:     seed  1337   [6, 6, 5, 3, 14]    mean 6.80   max 14
#:     seed  2024   [6, 2, 2, 3, 9]     mean 4.40   max  9
#:     seed 12345   [7, 4, 4, 20, 1]    mean 7.20   max 20
#:
#: **Reading 1 (narrow seeds stopped early, so the early-stopping
#: deviation is implicated): REFUTED.** The narrowest seed -- 99, sd
#: 0.0351 -- holds **the longest fold in the entire run, 26 epochs**. A
#: fold that trained 26 epochs and still produced the run's flattest
#: predictions was not cut short by the monitor. The best-PCC seed
#: (12345, 0.1151) meanwhile has folds at 1 and 4. The association the
#: reading predicted is absent, and where it appears it points the wrong
#: way.
#:
#: **Reading 2 (all seeds ran to similar epochs, so the spread is pure
#: initialisation): ALSO NOT SUPPORTED.** The epochs are nowhere near
#: similar -- seed 99 alone spans 1 to 26.
#:
#: **What the data does say, and it is cleaner than either**: the
#: stopping epoch varies **16.8x more WITHIN seeds than BETWEEN them**
#: (mean within-seed variance 42.26 against a between-seed variance of
#: seed means of 2.51). A seed-level quantity -- the 6.33x spread in
#: prediction sd -- cannot be explained by a variable whose variation is
#: overwhelmingly fold-level. **Stopping behaviour and the seed spread
#: are not coupled**, so the seed spread stays unexplained and the
#: monitor stays un-exonerated: both questions survive, and neither is
#: answerable from epochs alone.
#:
#: **A count corrected**: 15 of the 25 fold-runs stop at epochs 1-4, not
#: 14 (11 stop at 1-3; 16 at 1-5). It does not change the reading.
#:
#: **What keeps the monitor question alive**: with patience 5, a fold
#: selecting epoch 1 means epochs 2-6 were all worse -- and that happened
#: in a large minority of folds. Against that sits the faithful arm's
#: psychologist cell, **the only cell in either arm that learned to
#: differentiate, and it did so at epoch 33** -- deep past where most
#: folds stopped. One instance is not evidence that the monitor stops
#: runs which would have learned, but it is exactly the shape such
#: evidence would take, and it is the reason the question is worth one
#: more measurement rather than a shrug.
EPOCHS_ANSWER_NEITHER_READING = {
    "measured": "2026-08-17, from the completed run's selected_epochs",
    "epochs_by_seed": {
        7: (8, 1, 4, 1, 4), 99: (1, 2, 1, 26, 2), 1337: (6, 6, 5, 3, 14),
        2024: (6, 2, 2, 3, 9), 12345: (7, 4, 4, 20, 1),
    },
    "reading_1_refuted": (
        "the NARROWEST seed (99, sd 0.0351) holds the LONGEST fold in the "
        "run (26 epochs); the best-PCC seed (12345) has folds at 1 and 4. "
        "The predicted association is absent and points the wrong way "
        "where it appears"
    ),
    "reading_2_not_supported": (
        "the epochs are not similar at all -- seed 99 alone spans 1 to 26"
    ),
    "the_actual_structure": {
        "within_seed_variance": 42.26,
        "between_seed_variance_of_means": 2.51,
        "ratio": 16.8,
        "consequence": (
            "stopping epoch varies 16.8x more within seeds than between "
            "them, so it cannot explain a SEED-level 6.33x spread -- "
            "stopping behaviour and the seed spread are not coupled"
        ),
    },
    "count_corrected": "15 of 25 stop at epochs 1-4, not 14 (11 at 1-3, 16 at 1-5)",
    "monitor_question_still_live": (
        "with patience 5, selecting epoch 1 means epochs 2-6 were all "
        "worse, and that happened in a large minority of folds; against "
        "it, the psychologist cell -- the only cell in either arm that "
        "learned to differentiate -- did so at EPOCH 33, deep past where "
        "most folds stopped"
    ),
}


#: **[PROPOSED 2026-08-17, NOT RUN] The cheapest measurement that would
#: settle the monitor question -- and the arm is already throwing the
#: data away.**
#:
#: **The measurement**: the inner-val trajectory per fold. Flat
#: ``inner_val_mse`` after the selected epoch means the monitor could not
#: see improvement that may still have been coming; RISING means the
#: model genuinely degraded and stopping was right. The two have opposite
#: consequences for Phase 10's wording, and nothing else distinguishes
#: them.
#:
#: **The cost is almost nothing, because the numbers already exist.**
#: ``harness.run_cv`` computes ``train_loss``, ``inner_val_mse`` and
#: ``inner_val_pcc`` for EVERY epoch of EVERY fold and returns them in
#: ``FoldRun.curve`` (harness.py line 309). ``task_cleftgnn_cv`` never
#: writes them -- they are computed and discarded. The change is roughly
#: five lines: write ``result.folds[*].curve`` to a per-seed CSV beside
#: the predictions, in the pretraining path's own ``curves.csv`` layout.
#: **No new computation per epoch, no new hyperparameter, no touch to lr,
#: the sigmoid, or route 2.**
#:
#: **But it needs one more run**, because the completed run's curves are
#: gone -- which is the whole point: a diagnostic that was computed
#: twenty-five times and kept zero times.
#:
#: **Worth it before Phase 10 concludes? YES, on three grounds:**
#:
#: 1. It closes the last alternative explanation for a headline finding
#:    that goes to supervision. "The replication scores ~0 under our
#:    protocol" is a weaker sentence than "...and we can show it is not
#:    our stopping rule".
#: 2. It costs one arm run and five lines against a question that would
#:    otherwise be answered with a shrug in the write-up.
#: 3. **The instrument gap is worth closing regardless.** This is the
#:    same family as the probe that rebuilt the pipeline: information the
#:    system already produced, thrown away, leaving a question
#:    unanswerable that the data could have answered. Writing the curves
#:    is worth doing even if the trajectories turn out uninformative.
#:
#: If the answer is "flat", the monitor is implicated and the remedy is a
#: registered-deviation question (the monitor watches ``inner_val_mse``
#: on the expected-value readout -- a quantity that barely moves when
#: predictions sit near the prior). If "rising", stopping was correct and
#: the seed spread is the finding.
CURVE_WRITING_PROPOSED = {
    "proposed": "2026-08-17, not run",
    "measurement": (
        "the inner-val trajectory per fold: FLAT after the selected epoch "
        "means the monitor could not see improvement still coming; RISING "
        "means the model degraded and stopping was right"
    ),
    "the_data_already_exists": (
        "harness.run_cv computes train_loss, inner_val_mse and "
        "inner_val_pcc every epoch of every fold and returns them in "
        "FoldRun.curve (harness.py:309); task_cleftgnn_cv never writes "
        "them -- computed 25 times, kept zero times"
    ),
    "cost": (
        "~5 lines to write result.folds[*].curve to a per-seed CSV in the "
        "pretraining path's own layout; NO new computation, no new "
        "hyperparameter, no touch to lr, the sigmoid, or route 2 -- plus "
        "one arm run, because the completed run's curves are gone"
    ),
    "worth_it": (
        "closes the last alternative explanation for a headline finding "
        "going to supervision",
        "one run and five lines against a question otherwise answered "
        "with a shrug in the write-up",
        "the instrument gap is worth closing regardless -- the same "
        "family as the probe that rebuilt the pipeline: data the system "
        "produced and discarded",
    ),
    "readings": (
        "flat -> the monitor is implicated; the remedy is a "
        "registered-deviation question (inner_val_mse on the "
        "expected-value readout barely moves near the prior)",
        "rising -> stopping was correct and the seed spread is the finding",
    ),
    # [2026-08-17, same day] Approved and BUILT -- CURVES_AND_STAGES_WRITTEN.
    "built": "2026-08-17 -- see CURVES_AND_STAGES_WRITTEN",
}


#: **[OPEN CHECK 2026-08-17, the maintainer's own note] Criterion (i) for
#: run 5 was never taken.**
#:
#: ``--stages`` was not requested for the arm's own run, so **whether the
#: fused ratio cleared 0.4 on the arm's own fold is unreported**. The
#: probe's figure comes from a separate invocation on a separate split,
#: and **it must not be assumed to transfer**: the probe reads one batch
#: of one fold at initialisation, while the arm's folds differ in their
#: training sets and their heads move during training. Recorded as an
#: OPEN CHECK, not as an assumption, and it is the cheapest of the
#: outstanding items -- seconds on the pod.
CRITERION_I_UNREPORTED = {
    "recorded": "2026-08-17, the maintainer's own note",
    "gap": (
        "--stages was never requested for run 5, so whether fused cleared "
        "0.4 on the ARM's own fold is unreported"
    ),
    "why_it_does_not_transfer": (
        "the probe reads one batch of one fold at initialisation; the "
        "arm's folds differ in training set and their heads move during "
        "training"
    ),
    "status": "OPEN CHECK, the cheapest outstanding item -- seconds on the pod",
    # [2026-08-17, same day] The arm now records its own stage table at
    # each fold's SELECTED epoch, so this is answered on the arm's folds
    # rather than assumed from the probe -- CURVES_AND_STAGES_WRITTEN.
    "answered_by": "CURVES_AND_STAGES_WRITTEN -- the arm's own folds, next run",
}



#: **[BUILT 2026-08-17, not launched] The arm now writes its curves AND
#: its stage table -- one run answers both open items.**
#:
#: **The curves.** ``harness.run_cv`` already computed a per-epoch
#: ``train_loss``/``inner_val_mse``/``inner_val_pcc`` for every fold and
#: returned it in ``FoldRun.curve``; the task discarded all twenty-five.
#: It now writes ``seed_<n>__curves.csv`` in the pretraining path's own
#: layout with one added column -- ``fold`` -- because the arm has five
#: folds per seed where the pretraining path has one run. No new
#: computation, no hyperparameter touched.
#:
#: **The stage table, at the epoch the fold SELECTS -- which turned out
#: to be cheaply possible after all.** The obstacle looked structural:
#: the frozen harness does not restore best weights, so at the end of a
#: fold the model sits at ``best + patience``, not at its selection. But
#: the harness calls ``train_epoch`` exactly once per epoch, immediately
#: before that epoch's inner-val prediction, so the BACKBONE can count
#: its own epochs and record the table during a forward that already
#: happens. The ratios are therefore indexed by the true epoch number and
#: the selected epoch is a lookup.
#:
#: **It costs no extra forward.** The capture rides the first batch of
#: the first prediction after each epoch's training; a second prediction
#: in the same epoch does not overwrite it. Verified offline: epoch 0
#: recorded at the gate-3 prediction, epochs 1-3 after their training,
#: every stage present at every epoch, recording OFF by default.
#:
#: **What lands**: ``seed_<n>__stage_ratios.csv`` with one row per stage
#: per fold at that fold's selected epoch, and
#: ``fused_ratio_at_selected_epoch`` per seed in metrics.json -- so
#: criterion (i) is answered **on the arm's own folds** rather than
#: assumed from the probe's single initialisation batch. A fold selecting
#: epoch 0 writes ``NOT_RECORDED`` rather than borrowing another epoch's
#: table.
#:
#: **The ratio arithmetic moved into the model** (``cleftgnn.STAGE_ORDER``
#: and ``cleftgnn.stage_ratios``), and the probe now calls it. That is
#: the ``PROBE_RECONSTRUCTED_THE_PIPELINE`` lesson applied before it could
#: be repeated: the arm and the probe cannot now disagree about what a
#: stage table contains.
#:
#: **Both readings stand as committed**: flat inner-val after the
#: selected epoch implicates the monitor (a registered-deviation
#: question); rising means stopping was right and the seed spread is the
#: finding.
CURVES_AND_STAGES_WRITTEN = {
    "built": "2026-08-17, not launched",
    "curves": (
        "seed_<n>__curves.csv in the pretraining layout plus a fold "
        "column; the harness already computed them and the task "
        "discarded all 25"
    ),
    "stage_table_at_selection": (
        "the frozen harness does not restore best weights, but it calls "
        "train_epoch exactly once per epoch immediately before that "
        "epoch's inner-val prediction -- so the backbone counts its own "
        "epochs, records during a forward that already happens, and the "
        "selected epoch becomes a lookup"
    ),
    "costs_no_extra_forward": (
        "the capture rides the first batch of the first prediction after "
        "each epoch's training; a second prediction in the same epoch "
        "does not overwrite"
    ),
    "verified_offline": (
        "epoch 0 recorded at the gate-3 prediction, epochs 1-3 after "
        "their training, every stage present at every epoch, recording "
        "OFF by default"
    ),
    "artifacts": (
        "seed_<n>__curves.csv",
        "seed_<n>__stage_ratios.csv (one row per stage per fold, at that "
        "fold's selected epoch; NOT_RECORDED if the fold selected epoch 0)",
        "fused_ratio_at_selected_epoch per seed in metrics.json",
    ),
    "one_implementation": (
        "cleftgnn.STAGE_ORDER and cleftgnn.stage_ratios now live in the "
        "model and the probe calls them -- the "
        "PROBE_RECONSTRUCTED_THE_PIPELINE lesson applied before it could "
        "be repeated"
    ),
    "readings_unchanged": (
        "flat after the selected epoch -> the monitor is implicated, a "
        "registered-deviation question; rising -> stopping was right and "
        "the seed spread is the finding"
    ),
    "config_unchanged": (
        "no task field moved, so configs/p10_cleftgnn.yaml is untouched "
        "and its declarations still stand"
    ),
}



#: **[MEASURED 2026-08-17, p10-cleftgnn-6] CRITERION (i) FAILS ON THE
#: ARM'S OWN FOLDS -- and the reason is not the sigmoid.**
#:
#: Fused ratio at the selected epochs: **0.162-0.288 in 24 of 25 folds**
#: (one fold at 0.6009), against the registered >= 0.4 and the probe's
#: initialisation figure of 0.5256. **The probe's number did not
#: transfer**, exactly as ``CRITERION_I_UNREPORTED`` warned it might not:
#: the probe reads one batch at initialisation, the arm's folds are
#: measured after training.
#:
#: **The mechanism, from the table**: the GNN branch is UNCHANGED and
#: still dead -- 0.0029-0.0036 at the selected epochs, if anything worse
#: than its 0.0058 at init. What falls is the SABM branch:
#: **``sabm_v_a`` 0.7794 at init to 0.230-0.267 at the selected epochs**,
#: and ``fused`` follows it down.
#:
#: **So route 3's benefit is real at initialisation and does not survive
#: training.** Normalising the GNN branch stopped it drowning SABM -- the
#: init figures confirmed that -- but the branch it was protecting
#: degrades during training, and a fused ratio built on SABM falls with
#: it. **The failure is SABM degradation, not the sigmoid**: the sigmoid's
#: 87x kill is still there, still measured, and still not what moved.
#:
#: Route 3 is therefore recorded as **measured-and-insufficient**, which
#: is the disposition ``EXPECTED_POST_FIX_TABLE`` registered for exactly
#: this outcome -- and NOT as a licence to proceed to route 2, because
#: the cause the table names is upstream of the sigmoid question.
CRITERION_I_FAILED_ON_THE_ARM = {
    "measured": "2026-08-17, p10-cleftgnn-6",
    "fused_at_selected_epochs": "0.162-0.288 in 24 of 25 folds; one at 0.6009",
    "registered_threshold": 0.4,
    "probe_init_figure": 0.5256,
    "verdict": "FAILS -- and the probe's figure did not transfer, as warned",
    "gnn_branch": (
        "UNCHANGED and still dead: 0.0029-0.0036 at the selected epochs, "
        "worse than its 0.0058 at init"
    ),
    "sabm_branch": (
        "0.7794 at init -> 0.230-0.267 at the selected epochs; fused "
        "follows it down"
    ),
    "reading": (
        "route 3's benefit is real at initialisation and does not survive "
        "training: the branch it protects degrades, so a fused ratio "
        "built on SABM falls with it. The failure is SABM DEGRADATION, "
        "not the sigmoid"
    ),
    "disposition": (
        "measured-and-insufficient, the disposition registered in "
        "EXPECTED_POST_FIX_TABLE for this outcome -- NOT a licence for "
        "route 2, whose target is upstream of the named cause"
    ),
}


#: **[MEASURED ON ONE FOLD 2026-08-17, VERIFICATION REGISTERED AND
#: PENDING] THE MONITOR MAY BE SELECTING AGAINST THE THING WE MEASURE.**
#:
#: Fold 0, seed 1337: ``inner_val_mse`` 0.683 / 0.591 / 0.532 / 0.594 /
#: 0.820 / 0.415 / 0.790 selects epoch 6, while ``inner_val_pcc`` runs
#: -0.198 / -0.209 / -0.193 / -0.170 / -0.082 / -0.181 / -0.066.
#: **r(MSE, PCC) = +0.756** on that fold -- low MSE goes with LOW PCC --
#: and the selected epoch ranks **4th of 7** by PCC while the PCC-best
#: epoch (7) carries nearly the worst MSE.
#:
#: **The mechanism is arithmetic, not mysterious.** A model that predicts
#: near the label mean scores WELL on MSE precisely by not varying. So a
#: stopping rule that minimises inner-val MSE prefers the collapsed
#: solution, and the arm's near-zero PCC would then be partly a
#: measurement of **our own registered deviation** rather than of the
#: architecture.
#:
#: **This is one fold. The verification is registered BEFORE the
#: numbers**: ``scripts/p10_monitor_check.py`` reads the arm's own
#: ``seed_*__curves.csv`` and reports, per fold, r(MSE, PCC), the
#: MSE-selected epoch, the PCC-best epoch and the selected epoch's PCC
#: rank. **The pattern "holds generally" if r > 0 in at least 18 of 25
#: folds AND the MSE-selected epoch differs from the PCC-best epoch in at
#: least 18 of 25.** Fewer than that in either is a fold-level curiosity
#: and the monitor is not implicated. The tool reads artifacts only -- no
#: torch, no run -- and its arithmetic is checked against this fold in
#: the suite.
MONITOR_SELECTS_AGAINST_PCC = {
    "measured": "2026-08-17, one fold; verification registered and pending",
    "fold_0_seed_1337": {
        "inner_val_mse": (0.683, 0.591, 0.532, 0.594, 0.820, 0.415, 0.790),
        "inner_val_pcc": (-0.198, -0.209, -0.193, -0.170, -0.082, -0.181, -0.066),
        "r_mse_pcc": 0.7559,
        "mse_selected_epoch": 6,
        "pcc_best_epoch": 7,
        "pcc_rank_of_selected": "4 of 7",
    },
    "mechanism": (
        "a model predicting near the label mean scores WELL on MSE by not "
        "varying, so minimising inner-val MSE prefers the collapsed "
        "solution -- and the arm's near-zero PCC would be partly a "
        "measurement of our own registered deviation"
    ),
    "verification_tool": "scripts/p10_monitor_check.py -- artifacts only",
    "registered_threshold": (
        "holds generally iff r > 0 in >= 18 of 25 folds AND the "
        "MSE-selected epoch differs from the PCC-best in >= 18 of 25; "
        "fewer in either is a fold-level curiosity"
    ),
    "status": "PENDING the 25-fold check; one fold is not a pattern",
    # [WITHDRAWN 2026-08-17, same day] The check ran: r>0 in 13 of 25
    # against a required 18, binomial p = 1.000. The mechanism story is
    # WITHDRAWN as unsupported and fold 0 was a fold-level curiosity --
    # MONITOR_EXONERATED. The record stands as what was suspected and how
    # it was tested.
    "withdrawn": (
        "2026-08-17: 13 of 25, p = 1.000 -- unsupported; see "
        "MONITOR_EXONERATED"
    ),
}


#: **[HYPOTHESIS 2026-08-17, testable with data the arm can already
#: write] THE TWO FINDINGS MAY BE ONE MECHANISM.**
#:
#: Line the numbers up by epoch:
#:
#:     epoch 0 (init)        sabm 0.7794   -- highest
#:     epochs 1-6 (typical
#:       selection)          sabm 0.230-0.267, fused 0.162-0.288
#:     epoch 14 (fold 4,
#:       the longest)        sabm 0.436, fused 0.288, logits 0.200
#:                           against 0.03-0.07 logits elsewhere
#:
#: SABM does not fall monotonically with training: it **drops sharply in
#: the first epochs and is partway recovered by epoch 14**. And the fold
#: that trained longest is the one with the most image-dependence
#: surviving at every stage.
#:
#: Now add fold 0's curve: its PCC is WORST in the early epochs and best
#: at epoch 7, the last -- while its MSE is best at epoch 6. **The same
#: story from two instruments**: early training collapses the model
#: toward the mean, which is where inner-val MSE is minimised and where
#: SABM's image-dependence is lowest; more training begins to recover
#: both. **MSE-based early stopping stops the model in the trough.**
#:
#: If that is right, "SABM degrades during training" and "the monitor
#: selects collapsed solutions" are not two findings but one, seen from
#: the stage table and from the curve.
#:
#: **The test, and it needs one small change**: the arm currently writes
#: the stage table only at each fold's SELECTED epoch -- my own design
#: choice, and it is the wrong one for this question. Writing EVERY
#: epoch's table (the data is already in ``stage_ratios_by_epoch``, held
#: in memory for the whole fold) would show directly whether ``sabm_v_a``
#: is U-shaped in epoch. Cost: a longer CSV, no extra computation.
#: Proposed, not built -- and it belongs to whatever run answers the
#: monitor question, not to a run of its own.
THE_TWO_FINDINGS_MAY_BE_ONE = {
    "hypothesis": "2026-08-17, testable with data the arm can already write",
    "the_alignment": (
        "sabm 0.7794 at init; 0.230-0.267 at typical selections (epochs "
        "1-6); 0.436 at epoch 14 in the longest fold, which also carries "
        "the best fused (0.288) and logits (0.200 against 0.03-0.07)"
    ),
    "not_monotonic": (
        "SABM drops sharply early and is partway recovered by epoch 14 -- "
        "and fold 0's PCC is worst early and best at its last epoch while "
        "its MSE is best at epoch 6"
    ),
    "the_synthesis": (
        "early training collapses the model toward the mean, which is "
        "where inner-val MSE is minimised and SABM's image-dependence is "
        "lowest; more training recovers both. MSE-based early stopping "
        "stops the model in the trough -- so the two findings may be one "
        "mechanism seen from the stage table and from the curve"
    ),
    "the_test": (
        "write EVERY epoch's stage table, not only the selected epoch's: "
        "the data already sits in stage_ratios_by_epoch for the whole "
        "fold, so the cost is a longer CSV and no extra computation. It "
        "would show directly whether sabm_v_a is U-shaped in epoch"
    ),
    "my_design_error": (
        "recording only the selected epoch's table was my choice and it "
        "is the wrong one for this question"
    ),
    "not_built": "proposed; it belongs to whatever run answers the monitor question",
    # [HALF WITHDRAWN 2026-08-17, same day] The synthesis joined two
    # findings; the monitor half is now unsupported (MONITOR_EXONERATED),
    # so the SYNTHESIS as stated is withdrawn. The U-shape observation is
    # untouched -- it never depended on the monitor -- and stands on its
    # own in U_SHAPE_BELONGS_LATER.
    "half_withdrawn": (
        "2026-08-17: the monitor half is unsupported, so the synthesis as "
        "stated falls; the U-shape stands alone (U_SHAPE_BELONGS_LATER)"
    ),
}


#: **[PROPOSED 2026-08-17] WHAT THIS IMPLIES FOR PHASE 10'S CONCLUSION,
#: AND WHERE A MONITOR-ON-PCC ARM BELONGS.**
#:
#: **The conclusion cannot be "the architecture scores zero on this
#: cohort."** If the monitor check holds, the arm's near-zero PCC is
#: partly produced by our own stopping rule, and the defensible sentence
#: becomes: *under this protocol -- including an MSE-stopped training
#: rule we have measured to select against PCC -- the replication
#: produces predictions that do not span the label range and a
#: correlation indistinguishable from zero.* The limitation is ours to
#: state, not a property of their architecture.
#:
#: **The ledger consequence**: ``p10-resnet50-sabm-head`` is banked with
#: its figures, and the ledger is append-only. If the check holds, the
#: correct action is a NEW dated entry whose ``corrects`` field names it
#: and states the confound -- not an edit. That entry should not be
#: written before the check runs.
#:
#: **Is a monitor-on-PCC arm registrable as a Phase 10 cell? YES, with
#: conditions -- and the distinction that licenses it is precise.**
#:
#: Changing the statistic a result is REPORTED with, after seeing the
#: result, is the move this project has corrected twice
#: (``phase7b.CLAIMABLY_WORSE_WITHDRAWN``; the 8b criterion switch). That
#: is forbidden and stays forbidden. But the monitor is not a reporting
#: statistic -- it is a TRAINING protocol, our own deviation filling a
#: silence in their recipe, and it has now been measured defective for
#: this readout. Fixing a training protocol on a measured diagnosis is
#: what the early-stopping deviation ITSELF was: fixed budgets were
#: replaced because they were measured to read post-collapse endpoints.
#: The precedent points the same way, and only because the diagnosis is
#: measured rather than the result disliked.
#:
#: **The conditions, all five required:**
#:
#: 1. The 25-fold check must HOLD first, at its registered threshold.
#: 2. It is a SECOND CELL, never a replacement: the MSE-stopped arm's
#:    numbers stand and are reported beside it.
#: 3. Both readings committed before launch -- including the one where
#:    the PCC-stopped arm also lands near zero, which would strengthen
#:    rather than weaken the phase's finding.
#: 4. The EVALUATION criterion is untouched: panel mean, paired BCa,
#:    both output readings, both conditions.
#: 5. Registered as a dated amendment to ``REGISTERED_DEVIATIONS`` with
#:    the original standing, and **no third monitor without a new
#:    measured defect** -- the guard against trying monitors until one
#:    flatters the arm.
#:
#: **If any condition cannot be met, it belongs to a later phase**, where
#: it would be registered before any run rather than after this one.
MONITOR_ARM_REGISTRABILITY = {
    "proposed": "2026-08-17",
    "conclusion_cannot_be": (
        "'the architecture scores zero on this cohort' -- if the check "
        "holds, the near-zero PCC is partly produced by our stopping rule"
    ),
    "defensible_sentence": (
        "under this protocol, including an MSE-stopped training rule we "
        "have measured to select against PCC, the replication produces "
        "predictions that do not span the label range and a correlation "
        "indistinguishable from zero"
    ),
    "ledger_consequence": (
        "p10-resnet50-sabm-head is banked and the ledger is append-only: "
        "if the check holds, a NEW dated entry with corrects= names the "
        "confound. Not before the check runs"
    ),
    "registrable": True,
    "why_it_is_not_the_forbidden_move": (
        "changing a REPORTING statistic after seeing the result is "
        "forbidden and stays forbidden (CLAIMABLY_WORSE_WITHDRAWN, the 8b "
        "criterion switch). The monitor is a TRAINING protocol -- our own "
        "deviation filling their silence -- measured defective for this "
        "readout. Fixing it on a measured diagnosis is what the "
        "early-stopping deviation itself was"
    ),
    "conditions": (
        "the 25-fold check holds at its registered threshold",
        "a SECOND cell, never a replacement -- the MSE arm's numbers "
        "stand beside it",
        "both readings committed before launch, including the one where "
        "the PCC-stopped arm also lands near zero",
        "the evaluation criterion untouched: panel mean, paired BCa, both "
        "output readings, both conditions",
        "a dated amendment to REGISTERED_DEVIATIONS with the original "
        "standing, and NO third monitor without a new measured defect",
    ),
    "otherwise": "if any condition cannot be met it belongs to a later phase",
    # [CONDITION FAILED 2026-08-17, same day] Condition 1 -- "the 25-fold
    # check holds at its registered threshold" -- did NOT hold. So there
    # is NO monitor-on-PCC cell in Phase 10, and none is licensed
    # elsewhere on this evidence. The reasoning above stands as the
    # standard any future monitor change must meet.
    "outcome": (
        "NO CELL: condition 1 failed (MONITOR_EXONERATED). The five "
        "conditions stand as the standard a future monitor change must "
        "meet, on new evidence"
    ),
}



#: **[MEASURED 2026-08-17, 25 folds] THE MONITOR IS EXONERATED, AND THE
#: PRE-REGISTERED THRESHOLD CAUGHT MY ERROR.**
#:
#: The check: r(MSE, PCC) > 0 in **13 of 25** folds against a required 18
#: -- median r **+0.0927**, range **-0.946 to +0.729**. A two-sided
#: binomial on 13 of 25 gives **p = 1.000**: indistinguishable from a
#: coin. The mismatch condition passed alone (19 of 25) but it is the
#: weaker leg, and the sign condition is the load-bearing one.
#:
#: **The mismatch leg, read against its OWN null, points the other way.**
#: Two criteria picking independently from n epochs agree with
#: probability 1/n, so with 6-31 epochs per fold independence predicts
#: agreement between 3.2% and 16.7%. The observed agreement is **24%
#: (6 of 25) -- higher than chance for every epoch count in the range**.
#: MSE-selection and PCC-selection agree MORE often than independence
#: would give, which is evidence against anti-selection, not for it. The
#: leg that "passed" was passing in the wrong direction, and computing
#: its null is what shows that.
#:
#: **Fold 0 seed 1337 (r = +0.756) was a fold-level curiosity.** With a
#: per-fold range spanning -0.946 to +0.729, a single fold at +0.756 is
#: an unremarkable draw from that spread. **My one-fold reading was the
#: error**, and the threshold registered before the numbers is what
#: caught it -- which is the entire purpose of registering thresholds
#: before numbers, working exactly as designed.
#:
#: **WITHDRAWN, explicitly and as unsupported**: the mechanism story that
#: "early stopping on inner-val MSE systematically selects collapsed
#: solutions". It has no support at 25 folds. It should not be repeated,
#: quoted, or carried into the write-up in any form.
MONITOR_EXONERATED = {
    "measured": "2026-08-17, 25 folds",
    "sign_condition": {
        "positive_r": "13 of 25 against a required 18",
        "binomial_p": 1.000,
        "median_r": 0.0927,
        "range": (-0.946, 0.729),
        "verdict": "indistinguishable from a coin",
    },
    "mismatch_condition": {
        "observed": "19 of 25 differ, i.e. 6 of 25 agree (24%)",
        "its_own_null": (
            "two criteria picking independently from n epochs agree with "
            "probability 1/n; with 6-31 epochs that is 3.2%-16.7%"
        ),
        "reading": (
            "the observed 24% EXCEEDS chance for every epoch count in "
            "range -- the two criteria agree MORE than independence "
            "predicts, which is evidence AGAINST anti-selection. The leg "
            "that passed was passing in the wrong direction"
        ),
    },
    "fold_0_was_a_curiosity": (
        "at +0.756 against a per-fold range of -0.946 to +0.729, it is an "
        "unremarkable draw from the spread"
    ),
    "my_error": (
        "reading one fold as a mechanism; the threshold registered before "
        "the numbers caught it, which is exactly what registering "
        "thresholds before numbers is for"
    ),
    "withdrawn_as_unsupported": (
        "'early stopping on inner-val MSE systematically selects "
        "collapsed solutions' -- no support at 25 folds; not to be "
        "repeated, quoted, or carried into the write-up"
    ),
}


#: **[UNQUALIFIED 2026-08-17] PHASE 10'S DEFENSIBLE SENTENCE, with the
#: qualifier removed because the confound was measured away.**
#:
#: *Under this protocol the replication produces predictions that do not
#: span the label range and a correlation indistinguishable from zero.*
#:
#: No stopping-rule caveat attaches: the monitor was suspected, tested at
#: 25 folds, and exonerated (``MONITOR_EXONERATED``). The mechanism is
#: measured and stands on its own:
#:
#: * the **K == 1 sigmoid** kills the GNN branch -- 0.003 at the selected
#:   epochs, against 0.0058 at initialisation and 0.5025 one stage
#:   upstream;
#: * **route 3's initialisation-time benefit does not survive training**
#:   -- SABM falls 0.78 to ~0.25 and the fused ratio follows, so the
#:   normalisation that was right at init is not what the trained model
#:   carries.
#:
#: The sentence is about **this protocol and this build**, which is the
#: honest scope: it is not a claim about CleftGNN's architecture in the
#: group's own hands, on their cohort, under their splits.
PHASE_10_CONCLUSION_UNQUALIFIED = {
    "unqualified": "2026-08-17",
    "sentence": (
        "under this protocol the replication produces predictions that do "
        "not span the label range and a correlation indistinguishable "
        "from zero"
    ),
    "no_stopping_caveat": (
        "the monitor was suspected, tested at 25 folds and exonerated "
        "(MONITOR_EXONERATED)"
    ),
    "mechanism": (
        "the K==1 sigmoid kills the GNN branch (0.003 at the selected "
        "epochs against 0.0058 at init and 0.5025 one stage upstream)",
        "route 3's init-time benefit does not survive training: SABM "
        "falls 0.78 to ~0.25 and fused follows",
    ),
    "scope": (
        "this protocol and this build -- NOT a claim about CleftGNN in "
        "the group's hands, on their cohort, under their splits"
    ),
}


#: **[PROPOSED 2026-08-17] WHAT REMAINS BEFORE PHASE 10 CAN CLOSE.**
#:
#: Against the six registered exit criteria
#: (``PHASE_10_REGISTERED['exit_criteria']``):
#:
#:     1 discrepancy finding recorded        DONE (NOTEBOOK_ROI_SET_MEASURED,
#:                                           Thursday-flagged)
#:     2 built, deviations enumerated        BUILT -- but the criterion said
#:                                           "expect exactly one" and there are
#:                                           FIVE. That expectation needs a
#:                                           dated correction in the closing,
#:                                           not a quiet pass.
#:     3 five seeds run                      DONE (runs 4, 5, 6)
#:     4 paired BCa vs 0.2520, both readings **NOT DONE -- the one real
#:                                           outstanding item**
#:     5 ledgered beside the comparator      DONE (entry 25)
#:     6 suite green                         DONE
#:
#: **The paired BCa, with its reading registered before it runs.** The
#: per-seed OOF CSVs already exist, so this is a p7d-shaped two-pass
#: config (run dir, then declare) and no new training.
#:
#: **Predicted margin, computed now**: delta 0.2520 - 0.0241 = **0.2279**
#: against ``arm_means_95 = 1.96*sqrt(sd_a^2/n_a + sd_b^2/n_b)`` =
#: 1.96*sqrt(0.0148^2/5 + 0.0676^2/5) = **0.0607**, i.e. **3.76x**. That
#: lands **inside the margin table's MIXED region (3.6x-4.7x)**, where
#: 3.83 passed while 4.34 and 4.69 failed -- so **the margin does not
#: order it and condition 1 decides**. A delta this large yielding only
#: 3.76x is entirely due to the replication's seed sd being 4.6x arm A's:
#: an unstable arm is hard to beat claimably even when it is far behind.
#:
#: **Both readings committed now**: if it clears both conditions, Phase 10
#: can say the 0.2520 arm is claimably ahead of the replication under this
#: protocol -- a statement about our protocol's outcome, NOT evidence of
#: architectural superiority, since the loser does not span the label
#: range. If condition 1 fails, Phase 10 **cannot even claim that**, and
#: the honest closing is that the comparison the phase was opened to make
#: is unresolvable at this cohort's resolution -- which would be the
#: fourth independent arrival at ``COHORT_CANNOT_RESOLVE``.
#:
#: **Also needing a disposition, not a criterion but not silent either**:
#: the faithful arm under the fixed model still returns nan on four of
#: five raters (``faithful-3``). It should close with its own sentence
#: rather than be absorbed into the protocol arm's.
WHAT_REMAINS_BEFORE_CLOSING = {
    "proposed": "2026-08-17",
    "criteria_status": {
        "1_discrepancy": "DONE",
        "2_deviations_enumerated": (
            "BUILT, but the criterion expected ONE deviation and there are "
            "FIVE -- a dated correction in the closing, not a quiet pass"
        ),
        "3_five_seeds": "DONE",
        "4_paired_bca": "NOT DONE -- the one real outstanding item",
        "5_ledgered": "DONE (entry 25)",
        "6_suite": "DONE",
    },
    "paired_bca": {
        "cost": "no new training -- the per-seed OOF CSVs exist; a "
                "p7d-shaped two-pass config",
        "predicted_delta": 0.2279,
        "predicted_threshold": 0.0607,
        "predicted_margin": 3.76,
        "where_that_lands": (
            "INSIDE the mixed region 3.6x-4.7x (3.83 passed; 4.34 and 4.69 "
            "failed) -- the margin does not order it, condition 1 decides"
        ),
        "why_so_low_for_so_large_a_delta": (
            "the replication's seed sd is 4.6x arm A's: an unstable arm is "
            "hard to beat claimably even when far behind"
        ),
        "if_claimable": (
            "the 0.2520 arm is claimably ahead under this protocol -- a "
            "statement about our protocol's outcome, NOT architectural "
            "superiority, since the loser does not span the label range"
        ),
        "if_not_claimable": (
            "Phase 10 cannot even claim that, and closes on the comparison "
            "being unresolvable at this cohort's resolution -- a fourth "
            "independent arrival at COHORT_CANNOT_RESOLVE"
        ),
    },
    "faithful_arm_disposition": (
        "still nan on four of five raters under the fixed model "
        "(faithful-3); it needs its own closing sentence rather than "
        "absorption into the protocol arm's"
    ),
    # [AMENDED 2026-08-17, after the maintainer approved the notebook cell]
    # A SIXTH exit item, added openly rather than absorbed: the phase now
    # closes on two cells, not one. The amendment is recorded here because
    # a closing criterion that grows silently is how a phase closes on
    # what it happened to finish rather than on what it set out to do.
    "amended_2026_08_17": {
        "new_item": (
            "the NOTEBOOK-RECIPE cell (NOTEBOOK_RECIPE_CELL_REGISTERED) -- "
            "built, not launched; four outcomes committed in advance, "
            "including divergence, which would be recorded rather than "
            "repaired"
        ),
        "why_it_belongs_in_this_phase": (
            "outcome 2 -- 'also collapses' -- would let Phase 10 close on "
            "the ARCHITECTURE's behaviour rather than on our "
            "implementation of it, which is the stronger form of the "
            "conclusion the phase already reached. That is a Phase 10 "
            "question, answered with a Phase 10 run"
        ),
        "does_not_change": (
            "the manuscript cell's numbers, its registration, or the "
            "unqualified conclusion at PHASE_10_CONCLUSION_UNQUALIFIED -- "
            "and the outstanding paired BCa (criterion 4) remains the "
            "MANUSCRIPT cell's against 0.2520, unaffected"
        ),
        # [RESOLVED 2026-08-17, same day] The item is DISPOSED, not
        # outstanding: the cell was refused entry at gate 3 fold 0 and
        # yields a FINDING rather than a number
        # (NOTEBOOK_CELL_CANNOT_START_CALIBRATED), closed against
        # reopening by NOTEBOOK_CELL_DISPOSITION_CLOSED. The full walk of
        # all six criteria with this included is PHASE_10_EXIT_WALK.
        "resolved": (
            "2026-08-17 -- DISPOSED, not outstanding: a finding, not a "
            "number; see NOTEBOOK_CELL_CANNOT_START_CALIBRATED and "
            "PHASE_10_EXIT_WALK"
        ),
    },
}


#: **[DEFERRED 2026-08-17] THE U-SHAPE BELONGS TO A LATER PHASE -- but
#: the recording change belongs here.**
#:
#: The observation survives the monitor's exoneration untouched, because
#: it never depended on it: SABM measures 0.7794 at init, 0.230-0.267 at
#: typical selections, and **0.436 at epoch 14** in the longest fold,
#: which also carries that run's best fused (0.288) and logits (0.200
#: against 0.03-0.07). Non-monotonic, and unexplained.
#:
#: **It does not belong in Phase 10**, for three reasons:
#:
#: 1. **It cannot change the conclusion.** Phase 10 asks whether the
#:    replication reproduces their numbers under our criterion. That is
#:    answered, unqualified, with a measured mechanism. Whether SABM
#:    recovers by epoch 40 is a question about training dynamics, and no
#:    answer to it would alter the sentence.
#: 2. **It needs a run the phase does not otherwise need.** Every
#:    remaining exit criterion is satisfiable from artifacts already on
#:    disk. Adding a training run for a question that cannot move the
#:    conclusion is how a phase fails to close.
#: 3. **It is a lead for a build, not a result.** If a later phase
#:    revisits this architecture, "does the informative branch recover
#:    with longer training?" is the first thing to measure -- and it
#:    would be registered before that run rather than after this one.
#:
#: **What DOES belong here: the recording change.** The arm writes only
#: the selected epoch's stage table; writing all of
#: ``stage_ratios_by_epoch`` costs a longer CSV and no computation. Making
#: it now means any future run captures the U-shape for free, instead of
#: rediscovering that the data was computed and discarded -- the gap this
#: phase has now hit twice (the probe's rebuild, the missing curves).
#: **Proposed as a five-line change with no run attached.**
U_SHAPE_BELONGS_LATER = {
    "deferred": "2026-08-17",
    "observation_survives": (
        "SABM 0.7794 at init, 0.230-0.267 at typical selections, 0.436 at "
        "epoch 14 in the longest fold (best fused 0.288, best logits "
        "0.200) -- non-monotonic and unexplained; it never depended on "
        "the monitor story"
    ),
    "not_in_phase_10": (
        "it cannot change the conclusion, which is answered and "
        "unqualified with a measured mechanism",
        "it needs a training run the phase does not otherwise need -- "
        "every remaining criterion is satisfiable from artifacts on disk",
        "it is a lead for a future build, and would be registered before "
        "that run rather than after this one",
    ),
    "what_belongs_here": (
        "the RECORDING change: write all of stage_ratios_by_epoch, not "
        "only the selected epoch's row. A longer CSV, no computation, and "
        "any future run then captures the U-shape for free instead of "
        "rediscovering that the data was computed and discarded -- the "
        "gap this phase has hit twice"
    ),
    "proposed_not_built": "a five-line change with no run attached",
}



#: **[VERIFIED AT SOURCE 2026-08-17, both new artifacts read] THE 90.94%
#: IS CIFAR-10, DOCUMENTED IN THE GROUP'S OWN DECK.**
#:
#: Deck slide 3, verbatim: "Experiments on Benchmark -- CIFAR10 (various
#: RGB colours, scenes, classes: bird, dog, etc.)", with the table
#: "ViT-B16 with SRGNN **90.94%**, ViT-B32 with SRGNN **88.40%**" and the
#: note "Table accuracies provided below are within few epochs of
#: fine-tuning". The notebook backs every part: ``NUM_CLASSES = 10``,
#: ``datasets.CIFAR10``, ``EPOCHS = 5``.
#:
#: The project's conflation finding -- that the supervision 0.9 target
#: is a CIFAR-10 number, not a cleft correlation -- was previously an
#: INFERENCE from shared documents. It is now **confirmed from the
#: primary artifacts**, and the confirmation is stronger than the
#: inference: the deck states the dataset, the accuracy and the epoch
#: budget on one slide.
#:
#: **A fourth voice on the region count, found while verifying**: the
#: same slide says "Checked with **27 crops** in the framework selected
#: automatically with Regional Attention Network". So the deck says 27,
#: the notebook PASSES ``REGION_COUNT = 27``, and the notebook's own
#: generator produces **36** and self-adjusts with a printed warning
#: (``NOTEBOOK_ROI_SET_MEASURED``). The group's stated understanding is
#: 27; their code runs 36. That sharpens the Thursday item from
#: paper-vs-code to **belief-vs-execution**, which only they can settle.
CIFAR_CONFLATION_CONFIRMED_AT_SOURCE = {
    "verified": "2026-08-17, deck slide 3 + notebook cell 9",
    "deck_slide_3": (
        "'CIFAR10 (various RGB colours, scenes, classes: bird, dog, "
        "etc.)'; 'ViT-B16 with SRGNN 90.94%', 'ViT-B32 with SRGNN "
        "88.40%'; 'within few epochs of fine-tuning'"
    ),
    "notebook_backing": "NUM_CLASSES = 10, datasets.CIFAR10, EPOCHS = 5",
    "status_change": (
        "the conflation finding was an INFERENCE from shared documents; "
        "it is now confirmed from the primary artifacts"
    ),
    "fourth_voice_on_regions": (
        "the same slide says 'Checked with 27 crops'; the notebook PASSES "
        "REGION_COUNT = 27 and its generator produces 36 with a printed "
        "warning -- the group's stated understanding is 27, their code "
        "runs 36: belief-vs-execution, and only they can settle it"
    ),
    "thursday": True,
}


#: **[VERIFIED 2026-08-17, and it corrects a record of mine] THE
#: NOTEBOOK'S RECIPE IS ADAM AT lr 0.001, NOT SGD AT 0.01.**
#:
#: Notebook cell 9, the execution cell: ``LR = 0.001``;
#: ``optimizer = optim.Adam(trainable_params, lr=LR)``;
#: ``BATCH_SIZE = 16``; ``EPOCHS = 5``;
#: ``criterion = nn.CrossEntropyLoss()``. The manuscript says SGD at 0.01
#: and we implemented the manuscript -- **a third discrepancy of the same
#: class as the ROI one**.
#:
#: **MY RECORD WAS WRONG, and the maintainer's diagnosis of why is
#: correct.** ``REGISTERED_DEVIATIONS`` carried "batch 16 -- the standing
#: cleft default, the notebook states none". The notebook states
#: ``BATCH_SIZE = 16`` plainly in cell 9. I read the model-class cell (7)
#: and never opened the execution cell, then wrote a claim about the
#: whole notebook from one cell of it. **The value we chose is right by
#: coincidence** -- 16 either way -- which is exactly how such an error
#: survives: nothing downstream disagreed.
#:
#: **The hypothesis, and it is a hypothesis until measured**: Adam's
#: per-parameter adaptation would absorb the |f_t| ~ 21 feature scale
#: that plain SGD could not, which plausibly explains the entire
#: divergence saga (measured pre-fix: loss 16.4 -> 108.3 -> 29,519).
#: Adam normalises each parameter's step by its own gradient history, so
#: a large-magnitude, near-constant feature does not translate into a
#: large step. **Not measured here. It would be measured by the
#: notebook-recipe cell, not asserted.**
#:
#: **What else the whole-notebook read found** (the earlier pass covered
#: cell 7 only):
#:
#: * ``EPOCHS = 5`` with **no validation split and no early stopping** --
#:   the notebook's last epoch IS its model. Our early-stopping deviation
#:   fills a silence in the MANUSCRIPT, which remains true, but it
#:   **departs from the notebook's explicit budget**; the deviation's
#:   fidelity cost against the notebook is higher than recorded.
#: * ``transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
#:   0.2010))`` -- **CIFAR-10 statistics, not ImageNet**, under an
#:   ImageNet-pretrained ViT.
#: * ``transforms.Resize(224)`` on CIFAR's 32x32 -- a 7x upscale.
#: * ``REGION_COUNT = 27`` passed explicitly, then self-adjusted to 36.
#: * The optimiser is handed ``filter(lambda p: p.requires_grad, ...)``
#:   -- consistent with the frozen backbone.
#: * Cells 0-6 are CUDA setup, a pip install of torch-geometric, and
#:   imports. Nothing else substantive was missed.
NOTEBOOK_RECIPE_IS_ADAM = {
    "verified": "2026-08-17, notebook cell 9",
    "recipe": (
        "LR = 0.001; optim.Adam(trainable_params, lr=LR); BATCH_SIZE = "
        "16; EPOCHS = 5; nn.CrossEntropyLoss()"
    ),
    "manuscript_says": "SGD at 0.01 -- and we implemented the manuscript",
    "discrepancy_class": "the third of the same class as the ROI one",
    "my_record_was_wrong": (
        "REGISTERED_DEVIATIONS said 'the notebook states none' of batch "
        "size; cell 9 states BATCH_SIZE = 16 plainly. I read the "
        "model-class cell and wrote a claim about the whole notebook. The "
        "value we chose is right by coincidence -- 16 either way -- which "
        "is how such an error survives"
    ),
    "adam_hypothesis": (
        "Adam's per-parameter adaptation would absorb the |f_t| ~ 21 "
        "scale that plain SGD could not, plausibly explaining the whole "
        "divergence saga (loss 16.4 -> 108.3 -> 29,519 pre-fix). NOT "
        "measured; it would be measured by the notebook-recipe cell"
    ),
    "whole_read_also_found": (
        "EPOCHS = 5 with NO validation split and NO early stopping -- the "
        "notebook's last epoch is its model, so our early-stopping "
        "deviation fills a MANUSCRIPT silence but departs from the "
        "notebook's explicit budget",
        "CIFAR-10 normalisation statistics under an ImageNet-pretrained "
        "ViT, not ImageNet statistics",
        "transforms.Resize(224) on CIFAR's 32x32 -- a 7x upscale",
        "REGION_COUNT = 27 passed explicitly, then self-adjusted to 36",
        "the optimiser receives only requires_grad parameters",
        "cells 0-6 are CUDA setup, a torch-geometric install and imports "
        "-- nothing else substantive was missed",
    ),
}


#: **[VERIFIED 2026-08-17] THE NOTEBOOK'S FUSION IS MULTIPLICATIVE, AND
#: IT BEARS ON THE MEASURED DILUTION.**
#:
#: Notebook cell 7: ``f_t_bar = f_t_batch + f_t_batch * v_batch`` -- that
#: is **f_t * (1 + v)** -- with ``v`` squashed to (0, 1) by the ACM's
#: final ``torch.sigmoid``. We built the manuscript's eq (9),
#: ``v_r = f_t + v_a``, additive. **A third manuscript-vs-code
#: discrepancy**, and unlike the first two it has a measured consequence
#: waiting for it.
#:
#: **Why it bears on the dilution.** Our measurement: ``f_t`` is
#: near-constant across images (ratio 0.0068) and 5.8x larger than the
#: attention branch, so **addition drowns** the informative branch --
#: fused 0.1315 against v_a's 0.7794. Under multiplication the constant
#: branch is SCALED by an image-dependent factor instead of having a
#: small image-dependent vector added to it, so the attention's relative
#: variation survives the fusion rather than being diluted by a
#: magnitude ratio.
#:
#: **Estimated, not measured**: with v in (0, 1), the gate spans
#: 1 + v in (1, 2), so the fused relative variation is roughly
#: sd(v) / (1 + mean v) -- of order 0.2-0.3 on these magnitudes, against
#: the 0.1315 measured for addition and the 0.7794 the branch carries
#: alone. **Better than addition, not a restoration.** Marked an estimate
#: because it is arithmetic on measured magnitudes, not a run.
#:
#: **What this does NOT do**: it does not rehabilitate route 3 or
#: retrospectively fix anything. The additive form we built is the
#: manuscript's, faithfully; the multiplicative form is the notebook's,
#: equally faithfully. They are two artifacts disagreeing, and the
#: disagreement is now recorded with its measured stake.
NOTEBOOK_FUSION_IS_MULTIPLICATIVE = {
    "verified": "2026-08-17, notebook cell 7",
    "notebook": "f_t_bar = f_t_batch + f_t_batch * v_batch, i.e. f_t * (1 + v)",
    "v_is_bounded": "the ACM ends in torch.sigmoid, so v is in (0, 1)",
    "manuscript": "eq (9) v_r = f_t + v_a, additive -- what we built",
    "discrepancy_class": "the third manuscript-vs-code discrepancy",
    "why_it_matters": (
        "f_t is near-constant (0.0068) and 5.8x larger than the attention "
        "branch, so ADDITION drowns it (fused 0.1315 against v_a's "
        "0.7794); MULTIPLICATION scales the constant branch by an "
        "image-dependent factor, so the attention's relative variation "
        "survives the fusion"
    ),
    "estimate_not_measurement": (
        "with v in (0,1) the gate spans 1+v in (1,2), so fused relative "
        "variation is roughly sd(v)/(1+mean v) -- order 0.2-0.3 on these "
        "magnitudes, against 0.1315 for addition and 0.7794 for the "
        "branch alone. Better than addition, NOT a restoration; "
        "arithmetic on measured magnitudes, not a run"
    ),
    "does_not": (
        "rehabilitate route 3 or retrospectively fix anything -- the "
        "additive form is the manuscript's, faithfully built; the "
        "multiplicative is the notebook's. Two artifacts disagreeing, now "
        "recorded with the stake measured"
    ),
}


#: **[VERIFIED 2026-08-17] THE NOTEBOOK'S BACKBONE IS A FROZEN
#: ViT-B/16 -- the freeze decision is confirmed by the artifact.**
#:
#: ``ViTFeatureExtractorAndAdapter``: ``timm.create_model('vit_base_
#: patch16_224', pretrained=True, num_classes=0)``, then
#: ``for param in self.model.parameters(): param.requires_grad = False``,
#: and ``with torch.no_grad():`` wrapping the forward. Frozen twice over.
#:
#: ``BACKBONE_IS_FROZEN`` was decided on the reasoning that the notebook
#: freezes and that lr belongs to that regime; the artifact now states it
#: outright. **The backbone IDENTITY still differs**: the notebook's is
#: ViT-B/16, ours is ResNet-50 on the manuscript's Table-1 basis -- a
#: difference we chose deliberately and which the notebook-recipe cell
#: would close.
#:
#: **The ACM runs PARALLEL to the GNN, not downstream**:
#: ``self._attentional_context_modeling(batched_graph.x, ...)`` takes the
#: PRE-GNN region features. **Our build already does this** -- the ACM
#: reads ``regions``, the ``region_proposer`` output, not ``f_hat`` --
#: so this is a confirmation rather than a correction, and it is worth
#: recording as confirmed because the alternative would have been a
#: silent structural error.
NOTEBOOK_BACKBONE_AND_ACM_CONFIRMED = {
    "verified": "2026-08-17, notebook cell 7",
    "frozen_twice": (
        "requires_grad = False on every parameter AND torch.no_grad() in "
        "the forward"
    ),
    "confirms": "BACKBONE_IS_FROZEN, previously decided on reasoning",
    "identity_still_differs": (
        "the notebook's backbone is ViT-B/16; ours is ResNet-50 on the "
        "manuscript's Table-1 basis -- deliberate, and the "
        "notebook-recipe cell would close it"
    ),
    "acm_is_parallel": (
        "_attentional_context_modeling(batched_graph.x, ...) takes the "
        "PRE-GNN region features; our build already reads region_proposer "
        "output rather than f_hat, so this is CONFIRMED, not corrected"
    ),
}


#: **[VERIFIED 2026-08-17] A THIRD, NON-IDENTICAL DESCRIPTION OF THE
#: ATTENTION IDEA -- and the deck's is about WHICH REGIONS EXIST.**
#:
#: Deck slides 5-9, verbatim: "Two Modification Strategies: 1.
#: Modification on the crops in GNN, 2. Exploration on the attention
#: part"; previous work selects crops "implicitly and automatically";
#: "Our strategy **scopes the region selection**... **Bottom half case ->
#: 5 crops**" (slide 7) and "**Bottom third case -> 10 crops**" (slide
#: 8); slide 9 compares "Method-1: prev. work", "Method-2: bottom-half",
#: "Method-3: bottom-third".
#:
#: So the three artifacts describe three different things:
#:
#:     manuscript   vertical-position WEIGHTING of all regions (eq 7-8)
#:     deck         SCOPING -- restricting which ROIs exist (5 or 10)
#:     notebook     NO SABM at all; 36 unweighted regions
#:
#: These are not three phrasings of one mechanism. Weighting keeps every
#: region and scales it; scoping deletes regions. A model built to one is
#: not the model built to another.
#:
#: **The exact crop sets are NOT derivable from the deck** -- it gives
#: counts (5, 10) and a region of the face ("bottom half", "bottom
#: third") but no coordinates, no grid, no rule. **A question for supervision**,
#: and one only supervision can answer.
#:
#: **Also on slide 9, and worth its own line**: the three methods are
#: compared **on the CLEFT dataset**, not on CIFAR. So the group HAS
#: cleft results for bottom-half and bottom-third crops. We do not have
#: those numbers -- the deck's tables are images, and the extracted text
#: carries only the method labels. **Worth asking for**: they are the
#: closest thing to a like-for-like comparator this project has.
THREE_DESCRIPTIONS_OF_THE_ATTENTION = {
    "verified": "2026-08-17, deck slides 5-9",
    "manuscript": "vertical-position WEIGHTING of all regions (eq 7-8)",
    "deck": "SCOPING -- restricting which ROIs exist: 5 (bottom half), 10 (bottom third)",
    "notebook": "NO SABM at all; 36 unweighted regions",
    "not_three_phrasings": (
        "weighting keeps every region and scales it; scoping deletes "
        "regions. A model built to one is not the model built to another"
    ),
    "not_derivable": (
        "the deck gives counts and a face region but no coordinates, grid "
        "or rule -- a question for supervision, and only supervision can answer it"
    ),
    "slide_9_has_cleft_results": (
        "the three methods are compared ON THE CLEFT DATASET, so the "
        "group has cleft numbers for bottom-half and bottom-third crops. "
        "We do not have them -- the deck's tables are images. Worth "
        "asking for: the closest thing to a like-for-like comparator "
        "this project has"
    ),
}


#: **[RECORDED 2026-08-17, FOR PHASE 11 -- record only] The
#: penalise-optimism metric is theirs and central to their own
#: comparison -- and its DIRECTION is ambiguous across artifacts.**
#:
#: Deck slides 10, 11 and 12 each carry the same line beside a different
#: method: "Metric that penalized underprediction: if prediction gives
#: always higher score than it should be, then penalize". It appears on
#: all three method slides, so the metric is how they compare their own
#: methods -- confirming ``IEM_CARRIED_FOR_PHASE_11``'s premise.
#:
#: **But the direction does not read cleanly, and Phase 11 depends on
#: it.** The slide's label says "penalized UNDERprediction" while its
#: description says "gives always HIGHER score than it should be" --
#: which is overprediction, unless "score" runs opposite to the grade.
#: On the grade scale 1 = Excellent to 5 = Very Poor, a HIGHER number is
#: a WORSE outcome; on a quality "score", higher is better. The
#: manuscript's eq (16) is unambiguous in its own terms -- the heavier
#: 1.2 weight goes to ``y_hat - G < 0``, predicting a LOWER number, i.e.
#: flattering the result -- so the manuscript and the deck agree only if
#: "score" means quality rather than grade.
#:
#: **Recorded as an ambiguity to resolve before Phase 11 implements
#: anything**, because implementing the asymmetry backwards would invert
#: the metric while leaving every number plausible. One sentence from
#: supervision settles it.
IEM_IS_THEIRS_DIRECTION_AMBIGUOUS = {
    "recorded": "2026-08-17, deck slides 10-12, record only",
    "confirms": (
        "the penalise-optimism metric is theirs and central: the same "
        "line appears beside all three methods"
    ),
    "the_ambiguity": (
        "the label says 'penalized UNDERprediction' while the description "
        "says 'gives always HIGHER score than it should be' -- "
        "overprediction, unless 'score' runs opposite to the grade. On "
        "the 1=Excellent..5=Very Poor scale a higher number is worse; on "
        "a quality score higher is better"
    ),
    "the_manuscript_is_unambiguous": (
        "eq (16) gives the heavier 1.2 weight to y_hat - G < 0, "
        "predicting a LOWER number, i.e. flattering the result -- so the "
        "two agree only if 'score' means quality rather than grade"
    ),
    "why_it_must_be_settled_first": (
        "implementing the asymmetry backwards would invert the metric "
        "while leaving every number plausible. One sentence from supervision "
        "settles it"
    ),
}


#: **[PROPOSED 2026-08-17, NOT PICKED] A NOTEBOOK-RECIPE CELL BESIDE THE
#: MANUSCRIPT-RECIPE CELL.**
#:
#: **The cell**: Adam at lr 0.001, multiplicative fusion f_t*(1+v),
#: ViT-B/16 frozen backbone, 36 regions, no SABM -- the notebook as it
#: stands. Everything else this project's: 237 patients, the median
#: label, 5-fold OOF, five seeds, the panel mean, paired BCa, both output
#: readings.
#:
#: **For it**: the notebook is the ONLY EXECUTABLE artifact the group has
#: shared, and three of our five deviations exist purely because the
#: manuscript is silent where the notebook speaks -- optimiser, batch
#: size, epoch budget. Two of our measured problems point straight at it:
#: the divergence saga (Adam would plausibly absorb the scale that broke
#: SGD) and the dilution (multiplication preserves what addition drowns).
#: A replication that never runs the group's own executable recipe has
#: tested the paper, not the work.
#:
#: **Against it**: the manuscript is the publication under revision --
#: the artifact being replicated for the viva and the paper -- and the
#: existing cell is faithful to it. And adding a recipe after the first
#: recipe's result is known has the SHAPE of the move that failed its
#: condition at ``MONITOR_ARM_REGISTRABILITY``.
#:
#: **Why the shape is not the same, stated so the distinction can be
#: checked rather than trusted**: the monitor case proposed changing OUR
#: OWN deviation to chase a better number, on a one-fold diagnosis that
#: then failed at 25. This proposes running a SECOND INDEPENDENTLY
#: DOCUMENTED recipe -- the group's own -- where both cells are faithful
#: to different artifacts BY THE GROUP. It is a fidelity question, not a
#: tuning question. But that distinction is exactly what the maintainer
#: should test rather than accept.
#:
#: **THE READINGS, committed before any run:**
#:
#: 1. **Spanning predictor** -> the manuscript's stated recipe is not
#:    what produces their results, and the discrepancy list stops being
#:    bookkeeping and becomes a finding about the paper.
#: 2. **Also collapses** -> the architecture's behaviour on this cohort
#:    is RECIPE-INDEPENDENT, which is the stronger finding and the one
#:    that would let Phase 10 close on the architecture rather than on
#:    our implementation of it.
#: 3. **Spans but PCC ~ 0** -> the ``NORMALISED_ARM_READINGS`` branch
#:    fires: an architecture measurement, and LEDGERABLE under the
#:    standard criterion.
#:
#: **Conditions, mirroring the monitor arm's**: a SECOND cell with the
#: manuscript cell's registration and numbers untouched; the evaluation
#: criterion unchanged; its own deviations enumerated (it has FEWER
#: against its own artifact -- no optimiser, batch or epoch deviation,
#: though early stopping still departs from the notebook's fixed 5); and
#: no third recipe without a new documented artifact.
#:
#: **The scope caveat that must travel with it**: the notebook's recipe
#: is demonstrated on **CIFAR-10**, not on cleft. Running it on this
#: cohort tests "their executable recipe on our data", not "the recipe
#: that produced their cleft numbers" -- which we have never seen. Slide
#: 9 says cleft comparisons exist; we do not have them.
NOTEBOOK_RECIPE_CELL_PROPOSED = {
    "proposed": "2026-08-17, not picked",
    "the_cell": (
        "Adam lr 0.001, multiplicative fusion f_t*(1+v), frozen ViT-B/16, "
        "36 regions, no SABM -- the notebook as it stands; everything "
        "else this project's criterion"
    ),
    "for": (
        "the notebook is the only EXECUTABLE artifact the group shared",
        "three of our five deviations exist only because the manuscript "
        "is silent where the notebook speaks -- optimiser, batch, epochs",
        "two measured problems point at it: Adam would plausibly absorb "
        "the scale that broke SGD, and multiplication preserves what "
        "addition drowns",
        "a replication that never runs the group's executable recipe has "
        "tested the paper, not the work",
    ),
    "against": (
        "the manuscript is the publication under revision, and the "
        "existing cell is faithful to it",
        "adding a recipe after the first recipe's result is known has the "
        "SHAPE of the move that failed at MONITOR_ARM_REGISTRABILITY",
    ),
    "why_the_shape_differs": (
        "the monitor case changed OUR OWN deviation to chase a number, on "
        "a one-fold diagnosis that failed at 25. This runs a SECOND "
        "INDEPENDENTLY DOCUMENTED recipe, the group's own, with both "
        "cells faithful to different artifacts BY THE GROUP -- a fidelity "
        "question, not a tuning question. The maintainer should test that "
        "distinction rather than accept it"
    ),
    "readings_committed": (
        "spanning predictor -> the manuscript's stated recipe is not what "
        "produces their results; the discrepancy list becomes a finding "
        "about the paper",
        "also collapses -> the architecture's behaviour on this cohort is "
        "RECIPE-INDEPENDENT, the stronger finding, and it lets Phase 10 "
        "close on the architecture rather than on our implementation",
        "spans but PCC ~ 0 -> the NORMALISED_ARM_READINGS branch fires: "
        "an architecture measurement, LEDGERABLE under the criterion",
    ),
    "conditions": (
        "a SECOND cell; the manuscript cell's registration and numbers "
        "untouched",
        "the evaluation criterion unchanged",
        "its own deviations enumerated -- FEWER against its own artifact, "
        "though early stopping still departs from the notebook's fixed 5",
        "no third recipe without a new documented artifact",
    ),
    "scope_caveat": (
        "the notebook's recipe is demonstrated on CIFAR-10, not cleft. "
        "Running it here tests 'their executable recipe on our data', NOT "
        "'the recipe that produced their cleft numbers', which we have "
        "never seen -- slide 9 says those comparisons exist; we do not "
        "have them"
    ),
    # [2026-08-17, same day] APPROVED and built. The maintainer accepted the
    # fidelity-not-tuning distinction ON THIS TEST, on the stated basis
    # that the cell would be run BELIEVING it might collapse, because
    # "recipe-independent" is the stronger finding. The registration that
    # governs the built cell is NOTEBOOK_RECIPE_CELL_REGISTERED; this
    # record stays as the proposal it was.
    "approved": (
        "2026-08-17 -- see NOTEBOOK_RECIPE_CELL_REGISTERED for the "
        "registration the built cell runs under"
    ),
}


#: **[REGISTERED 2026-08-17, APPROVED, BUILT, NOT LAUNCHED] THE
#: NOTEBOOK-RECIPE CELL.**
#:
#: ``configs/p10_cleftgnn_notebook.yaml``, beside ``p10_cleftgnn.yaml``,
#: which is **untouched** -- its registration, its recipe, its numbers and
#: its record all stand exactly as they were.
#:
#: **The four differences, and there are only four.** Each is verified in
#: the notebook's own source, and each is the notebook disagreeing with
#: the manuscript rather than us disagreeing with either:
#:
#: 1. **Adam at lr 0.001**, betas and eps at torch's defaults, no weight
#:    decay -- ``optim.Adam(trainable_params, lr=LR)``, cell 9. The
#:    manuscript says SGD at 0.01.
#: 2. **Multiplicative fusion**, ``f_t + f_t * v`` kept in the notebook's
#:    literal form. The manuscript's eq (9) is additive.
#: 3. **A frozen ViT-B/16** read through the notebook's own extraction
#:    path, including its omission of the final ``model.norm``. The
#:    manuscript's Table-1 best is ResNet-50.
#: 4. **Their attention head, not SABM**: a learned ``acm_w_beta`` softmax
#:    over regions where the manuscript has eq (7)'s fixed vertical mask,
#:    and a closing sigmoid that bounds v into (0, 1) because it is about
#:    to multiply rather than be added. There is no ``m_vertical`` on this
#:    path.
#:
#: **And NEITHER LayerNorm.** ``fused_norm`` and ``gnn_norm`` were
#: measured repairs for the additive path's scale problems; the notebook
#: normalises nowhere. Carrying ours here would make a spanning result
#: unreadable -- their recipe or our repairs, and no way to tell. **This
#: makes the cell harder, not easier**: with v in (0, 1) the multiplicative
#: fused feature is LARGER than the additive one that broke SGD, so
#: whether Adam absorbs that scale is not a hope the cell rests on, it is
#: the hypothesis the cell measures.
#:
#: **Batch 16** is not a difference. Both artifacts state it.
#:
#: **THE FOUR OUTCOMES, committed before the run.** The first three are
#: the readings registered at proposal time and carried unchanged; the
#: fourth is the numerical failure the honest version of this cell has to
#: name in advance, because it is a live possibility and naming it
#: afterwards would be choosing it:
#:
#: 1. **Spanning predictor** -> the manuscript's stated recipe is not what
#:    produces their results, and the discrepancy list stops being
#:    bookkeeping and becomes a finding about the paper.
#: 2. **Also collapses** -> the architecture's behaviour on this cohort is
#:    RECIPE-INDEPENDENT. The stronger finding, and the one that lets
#:    Phase 10 close on the architecture rather than on our implementation
#:    of it.
#: 3. **Spans but PCC ~ 0** -> the ``NORMALISED_ARM_READINGS`` branch
#:    fires: an architecture measurement, LEDGERABLE under the standard
#:    criterion.
#: 4. **Diverges** (loss blows up, or non-finite) -> Adam does not absorb
#:    the scale either, which makes the fused-norm deviation NECESSARY
#:    rather than optional and is a result about the architecture in its
#:    own right. **It will be recorded, not repaired.** Adding our
#:    LayerNorms to this cell after seeing it diverge is precisely the
#:    move that failed its condition at ``MONITOR_ARM_REGISTRABILITY``.
#:
#: **The conditions, as registered and as met by the build**: a SECOND
#: cell (a second config, a second run directory); the manuscript cell's
#: registration and numbers untouched; the evaluation criterion unchanged
#: -- 237 patients, the sheet's Median as the training label, the panel
#: mean as the target, 5-fold OOF, five seeds, expected-value primary with
#: Top-1 beside, per-seed CSVs in the baseline layout; its own deviations
#: enumerated (``NOTEBOOK_CELL_DEVIATIONS``); and **no third recipe
#: without a third documented artifact by the group**.
#:
#: **The scope caveat travels with every number this cell produces**: the
#: notebook's recipe is demonstrated on **CIFAR-10**. Running it here
#: tests "their executable recipe on our data", NOT "the recipe that
#: produced their cleft numbers" -- which we have never seen. Slide 9 says
#: cleft comparisons exist (``SLIDE_9_CLEFT_NUMBERS_REQUESTED``); we do
#: not have them.
#:
#: **Why this is not the monitor move**, stated once more because it is
#: the load-bearing distinction: the monitor case changed OUR OWN
#: registered deviation to chase a better number, on a one-fold diagnosis
#: that then failed at 25 folds. This runs a second **independently
#: documented** recipe, the group's own, with both cells faithful to
#: different artifacts BY THE GROUP. The maintainer tested that distinction
#: and accepted it on this test, on the explicit basis that the cell is
#: run believing it might collapse.
NOTEBOOK_RECIPE_CELL_REGISTERED = {
    "registered": "2026-08-17, approved and built, NOT launched",
    "config": "configs/p10_cleftgnn_notebook.yaml",
    "manuscript_cell_untouched": (
        "p10_cleftgnn.yaml -- registration, recipe, numbers and record all "
        "stand exactly as they were"
    ),
    "four_differences": (
        "Adam lr 0.001, torch-default betas/eps, no weight decay "
        "(notebook cell 9) -- the manuscript says SGD 0.01",
        "multiplicative fusion f_t + f_t*v, the notebook's literal form -- "
        "the manuscript's eq (9) is additive",
        "a frozen ViT-B/16 by the notebook's own extraction path, "
        "including its omission of the final model.norm -- the "
        "manuscript's Table-1 best is ResNet-50",
        "their attention head, not SABM: a learned acm_w_beta softmax over "
        "regions in place of eq (7)'s fixed vertical mask, plus the "
        "closing sigmoid that bounds v into (0,1); no m_vertical exists "
        "on this path",
    ),
    "batch_16_is_not_a_difference": "both artifacts state it",
    "neither_layernorm": (
        "fused_norm and gnn_norm were measured repairs for the ADDITIVE "
        "path; the notebook normalises nowhere, and carrying ours would "
        "make a spanning result unreadable -- their recipe or our repairs, "
        "indistinguishable"
    ),
    "the_cell_is_harder_not_easier": (
        "with v in (0,1) the multiplicative fused feature is LARGER than "
        "the additive one that broke SGD, so whether Adam absorbs that "
        "scale is the hypothesis under test, not an assumption the cell "
        "rests on"
    ),
    "outcomes_committed": (
        "spanning predictor -> the manuscript's stated recipe is not what "
        "produces their results; the discrepancy list becomes a finding "
        "about the paper",
        "also collapses -> RECIPE-INDEPENDENT behaviour on this cohort, "
        "the stronger finding, and it lets Phase 10 close on the "
        "architecture rather than on our implementation",
        "spans but PCC ~ 0 -> the NORMALISED_ARM_READINGS branch fires: an "
        "architecture measurement, LEDGERABLE under the criterion",
        "DIVERGES -> Adam does not absorb the scale either, which makes "
        "the fused-norm deviation NECESSARY rather than optional. "
        "RECORDED, NOT REPAIRED: adding our LayerNorms after seeing it "
        "diverge is exactly the move that failed at "
        "MONITOR_ARM_REGISTRABILITY",
    ),
    "conditions": (
        "a SECOND cell -- second config, second run directory",
        "the manuscript cell's registration and numbers untouched",
        "the evaluation criterion unchanged: 237 patients, the sheet's "
        "Median as training label, the panel mean as target, 5-fold OOF, "
        "five seeds, expected-value primary with Top-1 beside, per-seed "
        "CSVs in the baseline layout",
        "its own deviations enumerated -- NOTEBOOK_CELL_DEVIATIONS",
        "NO third recipe without a third documented artifact by the group",
    ),
    "scope_caveat": (
        "the notebook's recipe is demonstrated on CIFAR-10. This tests "
        "'their executable recipe on our data', NOT 'the recipe that "
        "produced their cleft numbers', which we have never seen"
    ),
    "why_not_the_monitor_move": (
        "the monitor case changed OUR OWN registered deviation to chase a "
        "number, on a one-fold diagnosis that failed at 25 folds. This "
        "runs a second INDEPENDENTLY DOCUMENTED recipe, the group's own, "
        "both cells faithful to different artifacts BY THE GROUP. The "
        "maintainer tested the distinction and accepted it on this test, on "
        "the explicit basis that the cell is run believing it might "
        "collapse"
    ),
}


#: **[REGISTERED 2026-08-17] THE NOTEBOOK CELL'S OWN DEVIATIONS -- THREE,
#: each against the notebook rather than against the manuscript.**
#:
#: The manuscript cell carries five (``REGISTERED_DEVIATIONS``). This cell
#: carries three, and two of them are decisions the maintainer took toward
#: fidelity-to-the-artifact rather than away from it:
#:
#: 1. **Early stopping (0.2 / 40 / 5, monitor ``inner_val_mse``) instead
#:    of the notebook's fixed ``EPOCHS = 5``.** Kept deliberately. The
#:    notebook's budget is defect-cited: five epochs with **no validation
#:    split at all**, tuned on 50,000 CIFAR-10 images, is not transferable
#:    to 190 training patients, and this project measured post-collapse
#:    endpoints on fixed budgets for this cohort (peaks at epochs 2-11).
#:    **This is the ONE deviation this cell also carries from the
#:    manuscript cell** -- the two arms early-stop identically, which is
#:    also what keeps them comparable. Fidelity cost against the notebook:
#:    REAL and stated, because the notebook is explicit here where the
#:    manuscript is silent.
#: 2. **This project's normalisation, not the notebook's.** The notebook
#:    normalises with **CIFAR-10 statistics** (0.4914, 0.4822, 0.4465) /
#:    (0.2023, 0.1994, 0.2010) under an **ImageNet-pretrained** ViT. That
#:    is a **defect in the notebook, not a recipe choice**: it mis-scales
#:    the input to frozen weights that were fitted under different
#:    statistics, and replicating it would corrupt exactly the frozen
#:    features this cell depends on. The statistics come from the
#:    backbone's own ``timm`` data config, as everywhere else in this
#:    project. Fidelity cost: NONE -- it declines to replicate a bug.
#: 3. **The Laplace-biased classifier bias is held constant across both
#:    cells.** The notebook uses ``nn.Linear``'s default init throughout.
#:    Keeping our bias here is not fidelity to the notebook; it is holding
#:    OUR apparatus fixed so that the difference between the two cells is
#:    the group's four, not ours as well. Stated as a deviation rather
#:    than defended as neutral, because it does change the trajectory.
#:
#: **What is NOT a deviation**: the data, the label, the folds, the seeds
#: and the criterion. Those are the project's, on both cells, by design --
#: that is what "evaluate as we evaluate" means, and it is the point of
#: the comparison rather than a departure from it.
NOTEBOOK_CELL_DEVIATIONS = {
    "registered": "2026-08-17",
    "expected_count": 3,
    "deviations": (
        "inner-val early stopping (0.2/40/5, inner_val_mse) instead of the "
        "notebook's fixed EPOCHS = 5 -- the notebook's budget is "
        "defect-cited: five epochs with NO validation split, tuned on "
        "50,000 CIFAR-10 images, is not transferable to 190 training "
        "patients, and fixed budgets measured post-collapse endpoints on "
        "this cohort (peaks 2-11). THE ONE DEVIATION THIS CELL ALSO "
        "CARRIES from the manuscript cell, which is also what keeps the "
        "two comparable. Fidelity cost against the notebook: REAL, and "
        "stated -- the notebook is explicit here where the manuscript is "
        "silent",
        "this project's normalisation (the backbone's own timm data "
        "config) instead of the notebook's CIFAR-10 statistics under an "
        "ImageNet-pretrained ViT -- a DEFECT in the notebook rather than a "
        "recipe choice: it mis-scales the input to frozen weights fitted "
        "under different statistics, and replicating it would corrupt the "
        "very frozen features this cell depends on. Fidelity cost: NONE, "
        "it declines to replicate a bug",
        "the Laplace-biased classifier bias held CONSTANT across both "
        "cells, where the notebook uses nn.Linear's default init -- not "
        "fidelity to the notebook but holding OUR apparatus fixed so the "
        "difference between the cells is the group's four and not ours as "
        "well. Stated as a deviation rather than defended as neutral, "
        "because it does change the trajectory",
    ),
    "not_deviations": (
        "the data, the label, the folds, the seeds and the criterion -- "
        "the project's on both cells by design; that is what 'evaluate as "
        "we evaluate' means, and it is the point of the comparison rather "
        "than a departure from it"
    ),
    "against_the_manuscript_cells_five": (
        "fewer, and against its own artifact: this cell has no optimiser, "
        "batch, fusion or backbone deviation because the notebook states "
        "all four"
    ),
}


#: **[FINDING 2026-08-17] THE REGION COUNT IS NOT A PAPER-VS-CODE
#: DISCREPANCY. IT IS BELIEF VERSUS EXECUTION, AND THE GROUP HOLDS BOTH
#: SIDES OF IT.**
#:
#: Four voices, three of them the group's own:
#:
#:     manuscript   "27 regions" (the count in the text)
#:     deck slide 3 "Checked with 27 crops in the framework selected
#:                  automatically with Regional Attention Network"
#:     notebook     ``SRGNN(..., region_count=27)`` -- passed explicitly
#:     notebook     ``generate_ran_rois`` returns **36**, and the model
#:                  self-adjusts: "Warning: Adjusted region count R to 36
#:                  based on unique generated ROIs."
#:
#: So the group states 27 in three places and runs 36 in the only place
#: that executes. Their own code **prints the correction at every
#: construction** and it has not propagated back into the text or the
#: slides.
#:
#: **Why the reframing matters.** A paper-vs-code discrepancy is a
#: documentation slip: the text describes one thing, the implementation
#: does another, and the fix is to correct whichever is wrong. This is
#: different, and worse for the paper: the ROI enumeration in the
#: manuscript is presented as a design choice, and the number attached to
#: it is not the number that produced any of their results. **Every result
#: in the paper was produced with 36 regions.** The reported architecture
#: has never been run.
#:
#: **Phrased for a manuscript under revision** -- what the authors need,
#: not what we found: the enumeration in the text produces 36 unique
#: regions, not 27, and the released code detects this at run time and
#: overrides the passed value with a printed warning. The revision needs
#: **one** of two things: the count in the text corrected to 36 with the
#: enumeration left as it is, **or** an enumeration that actually yields
#: 27 together with a re-run of every result under it. The first is a
#: correction; the second is a different paper. **Only the authors can say
#: which was intended**, and that is exactly why this is a question rather
#: than a finding we can settle.
#:
#: **This is the sharpest Thursday item**: it is short, it is verifiable
#: from their own artifacts in under a minute, and it changes what the
#: paper claims to have run.
REGION_COUNT_BELIEF_VS_EXECUTION = {
    "finding": "2026-08-17",
    "four_voices": (
        "manuscript: 27 regions in the text",
        "deck slide 3: 'Checked with 27 crops in the framework selected "
        "automatically with Regional Attention Network'",
        "notebook: SRGNN(..., region_count=27), passed explicitly",
        "notebook: generate_ran_rois returns 36 and the model self-adjusts "
        "with a printed warning at every construction",
    ),
    "reframing": (
        "not paper-vs-code, which is a documentation slip, but "
        "BELIEF-vs-EXECUTION: the ROI enumeration is presented as a design "
        "choice and the number attached to it is not the number that "
        "produced any of their results. EVERY result in the paper was "
        "produced with 36 regions; the reported architecture has never "
        "been run"
    ),
    "for_the_revision": (
        "the enumeration in the text produces 36 unique regions, not 27, "
        "and the released code detects this at run time and overrides the "
        "passed value with a printed warning. The revision needs ONE of: "
        "the count corrected to 36 with the enumeration left as it is (a "
        "correction), OR an enumeration that actually yields 27 together "
        "with a re-run of every result under it (a different paper). Only "
        "the authors can say which was intended"
    ),
    "sharpest_thursday_item": (
        "short, verifiable from their own artifacts in under a minute, and "
        "it changes what the paper claims to have run"
    ),
    "thursday": True,
}


#: **[BLOCKING 2026-08-17] PHASE 11 CANNOT IMPLEMENT THE PENALISE-OPTIMISM
#: METRIC UNTIL SUPERVISION SETTLES ITS DIRECTION.**
#:
#: The ambiguity is recorded in full at
#: ``IEM_IS_THEIRS_DIRECTION_AMBIGUOUS``: the deck's label says "penalized
#: **under**prediction" while its own text says "gives always **higher**
#: score than it should be", and on the 1 = Excellent .. 5 = Very Poor
#: grade scale those are opposite. The manuscript's eq (16) is internally
#: unambiguous -- the heavier 1.2 weight falls on ``y_hat - G < 0``, a
#: LOWER predicted number, i.e. a flattering prediction -- so the two
#: sources agree only if the deck's "score" means quality rather than
#: grade.
#:
#: **Recorded as BLOCKING, not as a caveat.** The distinction matters
#: because a caveat travels beside a number and a block prevents the
#: number existing. Implementing the asymmetry backwards would invert the
#: metric while leaving **every number plausible**: the magnitudes would
#: be in range, the ordering would look sensible, and nothing downstream
#: would disagree. That is the failure mode with no internal detector --
#: the same shape as the batch-size claim made from one cell, and as the
#: probe that reported a pipeline it was not running. Neither was caught
#: by anything except going back to the source.
#:
#: **What unblocks it**: one sentence from supervision. "Higher grade = worse, and
#: we penalise predictions that are too LOW" (or its opposite) settles it
#: completely. Failing that, an unambiguous worked example -- one true
#: grade, one prediction, the resulting penalty -- settles it just as well.
#:
#: **What is NOT blocked**: everything else in Phase 11. This blocks the
#: IEM implementation specifically, and it is registered now so that the
#: block is inherited rather than rediscovered.
PHASE_11_BLOCKED_ON_IEM_DIRECTION = {
    "blocking": "2026-08-17",
    "blocks": "the Phase 11 implementation of the penalise-optimism metric",
    "detail": "IEM_IS_THEIRS_DIRECTION_AMBIGUOUS carries the full reading",
    "why_blocking_and_not_a_caveat": (
        "a caveat travels beside a number; a block prevents the number "
        "existing. Implementing the asymmetry backwards would invert the "
        "metric while leaving EVERY number plausible -- magnitudes in "
        "range, ordering sensible, nothing downstream disagreeing. The "
        "failure mode with no internal detector, the same shape as the "
        "batch-size claim from one cell and the probe that reported a "
        "pipeline it was not running"
    ),
    "what_unblocks_it": (
        "one sentence from supervision -- 'higher grade = worse, and we penalise "
        "predictions that are too LOW', or its opposite. Failing that, one "
        "worked example: a true grade, a prediction, the resulting penalty"
    ),
    "not_blocked": (
        "everything else in Phase 11; this blocks the IEM implementation "
        "specifically, registered now so the block is inherited rather "
        "than rediscovered"
    ),
    # **[LIFTED 2026-08-17] Answered by supervision via the maintainer: convention
    # A.** If the model says a repair looks great when it actually looks
    # poor, a clinician might not investigate; the reverse error is safer.
    # On 1=Excellent..5=Very Poor that is y_hat - G < 0, the heavier
    # branch -- so eq (16) as printed implements the intent, and
    # phase11.IEM_DIRECTION_INFERENCE's registered prediction is
    # confirmed. The record stays because the block was real and its
    # reasoning is what made the answer worth asking for.
    "lifted": (
        "2026-08-17 -- convention A, answered by supervision via the maintainer. "
        "eq (16) as printed implements the intent; "
        "phase11.IEM_DIRECTION_ANSWERED. The loss build is no longer "
        "blocked on the direction"
    ),
}


#: **[REQUEST 2026-08-17] SLIDE 9'S CLEFT COMPARISON -- ASK FOR THE
#: NUMBERS.**
#:
#: Deck slide 9 compares "Method-1: prev. work", "Method-2: bottom-half"
#: and "Method-3: bottom-third" **on the CLEFT dataset**, not on CIFAR-10.
#: The group therefore holds cleft results for the two crop-scoping
#: strategies, evaluated against the same data this project works on.
#: **We do not have them**: the deck's tables are images, and the
#: extracted text carries only the method labels.
#:
#: **Why they are worth asking for, specifically.** Every comparator this
#: project has had to work with so far is either (a) their CIFAR-10
#: accuracy, which is a different dataset and a different task, or (b)
#: their published 0.598 cell, which is one of fifteen, n=25, with no
#: interval and a same-model companion at 0.283
#: (``FAITHFUL_ARM_REGISTERED``). Slide 9's numbers are **cleft, and
#: internally comparable to each other** -- three methods, one dataset,
#: one protocol. That makes them the closest thing to a like-for-like
#: comparator this project has been offered.
#:
#: **What to ask for, precisely**: the numbers behind slide 9's tables,
#: the metric they are in, the test set they were computed on and its n,
#: and whether any of them carries an interval. **And the crop sets
#: themselves** -- the deck gives counts (5, 10) and a face region
#: ("bottom half", "bottom third") but no coordinates, grid or rule, so
#: the strategies are not reproducible from what we hold
#: (``THREE_DESCRIPTIONS_OF_THE_ATTENTION``).
#:
#: **Recorded as a request, not a finding.** Nothing is claimed from a
#: table we have not read.
SLIDE_9_CLEFT_NUMBERS_REQUESTED = {
    "request": "2026-08-17",
    "what_the_slide_shows": (
        "Method-1 prev. work, Method-2 bottom-half, Method-3 bottom-third, "
        "compared ON THE CLEFT DATASET rather than on CIFAR-10"
    ),
    "we_do_not_have_them": (
        "the deck's tables are images; the extracted text carries only the "
        "method labels"
    ),
    "why_worth_asking": (
        "every comparator so far is either their CIFAR-10 accuracy (a "
        "different dataset and task) or their 0.598 cell (one of fifteen, "
        "n=25, no interval, same model 0.283 on their other test set). "
        "Slide 9's numbers are cleft AND internally comparable -- three "
        "methods, one dataset, one protocol: the closest thing to a "
        "like-for-like comparator this project has been offered"
    ),
    "ask_precisely": (
        "the numbers behind slide 9's tables",
        "the metric they are in",
        "the test set and its n",
        "whether any carries an interval",
        "the crop sets themselves -- coordinates, grid or rule; the deck "
        "gives counts and a face region only, so the strategies are not "
        "reproducible from what we hold",
    ),
    "not_a_finding": "nothing is claimed from a table we have not read",
    "thursday": True,
}


#: **[ARTIFACT FACTS 2026-08-17] THE NOTEBOOK'S TRAINING BUDGET AND ITS
#: NORMALISATION, recorded as what they are.**
#:
#: **The budget**: ``EPOCHS = 5``, **no validation split**, no early
#: stopping, no checkpoint selection. The notebook's last epoch IS its
#: model. Combined with ``NUM_CLASSES = 10`` and ``datasets.CIFAR10``,
#: that budget was chosen for 50,000 training images and a ten-class
#: problem, and the deck's own words for it are "within few epochs of
#: fine-tuning".
#:
#: **Its consequence for our record, stated because it makes our position
#: worse rather than better**: the early-stopping deviation was registered
#: as filling a silence, and against the **manuscript** that is still
#: exactly true -- the paper states no epoch budget anywhere, measured.
#: But the notebook is not silent, and our deviation **departs from it**.
#: The fidelity cost of that deviation is therefore higher than
#: ``REGISTERED_DEVIATIONS`` recorded, and it is corrected there in place.
#:
#: **The normalisation**: ``transforms.Normalize((0.4914, 0.4822,
#: 0.4465), (0.2023, 0.1994, 0.2010))`` -- **CIFAR-10 statistics** --
#: applied to input for an **ImageNet-pretrained** ViT-B/16, on images
#: brought to 224 by ``transforms.Resize(224)`` from CIFAR's 32x32, a 7x
#: upscale.
#:
#: **This one is a defect, and saying so is a judgement we own.** A frozen
#: backbone's weights were fitted under the statistics of their own
#: pretraining distribution; feeding it inputs standardised by a different
#: dataset's statistics shifts every activation in a way the frozen
#: weights cannot adapt to. It is not a recipe choice because nothing is
#: gained by it and it is not discussed anywhere. On CIFAR-10 at 90.94%
#: the model evidently tolerates it; on a frozen-feature cleft task where
#: the features are the whole model, it is the kind of thing that quietly
#: costs everything. **The notebook cell therefore does not replicate
#: it** (``NOTEBOOK_CELL_DEVIATIONS``), and the reason is recorded rather
#: than left as a silent difference.
NOTEBOOK_BUDGET_AND_NORMALISATION = {
    "recorded": "2026-08-17, notebook cell 9",
    "budget": (
        "EPOCHS = 5, NO validation split, no early stopping, no checkpoint "
        "selection -- the notebook's last epoch IS its model; chosen for "
        "50,000 CIFAR-10 images and ten classes, and the deck calls it "
        "'within few epochs of fine-tuning'"
    ),
    "consequence_for_our_record": (
        "the early-stopping deviation fills a silence in the MANUSCRIPT, "
        "which remains exactly true -- the paper states no budget "
        "anywhere, measured. But the notebook is NOT silent, and our "
        "deviation departs from it: the fidelity cost is higher than "
        "REGISTERED_DEVIATIONS recorded, corrected there in place"
    ),
    "normalisation": (
        "transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, "
        "0.2010)) -- CIFAR-10 statistics -- under an ImageNet-pretrained "
        "ViT-B/16, on images brought to 224 from 32x32 by Resize(224)"
    ),
    "why_we_call_it_a_defect": (
        "a frozen backbone's weights were fitted under its own "
        "pretraining statistics; standardising the input by a different "
        "dataset's statistics shifts every activation in a way frozen "
        "weights cannot adapt to. Nothing is gained by it and it is "
        "discussed nowhere. CIFAR-10 at 90.94% evidently tolerates it; a "
        "frozen-feature cleft task, where the features ARE the model, is "
        "where it would quietly cost everything. A judgement we own, and "
        "the reason the notebook cell does not replicate it"
    ),
}



#: **[FINDING 2026-08-17, DATED, RECORDED BEFORE ANY ROUTE IS CHOSEN]
#: THE NOTEBOOK CELL FAILS GATE 3, AND THE FAILURE IS A PER-SEED LOTTERY
#: RATHER THAN A SYSTEMATIC BIAS.**
#:
#: ``p10-cleftgnn-notebook``, fold 0: epoch-0 predicted mean **2.0030**
#: against the training-fold mean **2.7566** -- 0.7536 out, **0.95 SD**,
#: limit 0.5. Gate 3's HEAD-POSITION check; the dispersion check did not
#: fire. Nothing trained. Nothing is citable.
#:
#: **THE OPERATOR'S READING, TESTED RATHER THAN ADOPTED.** It was: the
#: multiplicative fusion with the ACM's closing sigmoid makes the fused
#: feature systematically enlarged and non-zero-centred, so a
#: small-but-nonzero classifier weight drags the epoch-0 expected value
#: off the fold mean, where the manuscript cell's near-zero weights made
#: epoch 0 equal the bias by construction.
#:
#: **CONFIRMED: the cause.** The head computes ``softmax(W . f + b)``. The
#: bias alone is exactly calibrated -- ``softmax(b)`` IS the Laplace
#: frequency vector, so the zero-weight prediction is
#: ``(N * mean + 15) / (N + 5)``, a closed form needing no forward. On a
#: 189-patient fold at mean 2.7302 that is **2.7371, 0.009 SD out**, and
#: the numeric control reproduces it to four decimals with prediction sd
#: **0.000000**. So every part of the departure is ``W . f`` and nothing
#: else. And ``W . f`` scales with what the classifier receives:
#: ``sd((W . f)_k) = sd(W) * ||f||``, with ``sd(W) = 0.01804`` from
#: ``nn.Linear``'s own kaiming-uniform bound. Measured, offline:
#:
#:     path         rms(f)      ||f||     sd(W.f)      p_max
#:     manuscript   1.0000      32.00     0.36-1.00    0.39-0.70
#:     notebook     7.72-8.40   247-269   2.43-5.39    0.59-0.999
#:
#: The manuscript path's rms is **exactly 1 by construction** -- that is
#: what ``fused_norm`` does -- so its analytic perturbation is 0.577 nats
#: against the notebook's ~4.8. **An eightfold larger logit
#: perturbation**, against a Laplace bias spanning only ~3.3 nats. The
#: bias loses.
#:
#: **CORRECTED: the character.** It is NOT a systematic drag. Across five
#: seeds the notebook cell's epoch-0 mean is **3.7739, 2.9994, 2.4195,
#: 1.0256, 2.3076** -- it departs in BOTH directions and spans nearly the
#: whole grade range. ``W`` is a random draw; there is no downward force.
#: The observed 2.0030 is one draw, not a tendency.
#:
#: **AND IT IS FIXED PER SEED, NOT PER FOLD.** ``reset`` seeds torch and
#: rebuilds, so every fold of a seed gets the SAME ``W``; only the bias
#: and the images differ, and both are swamped. Measured across five
#: folds per seed, the epoch-0 mean moves by **0.0001 to 0.0548** while
#: the fold's own label mean moves ten times that:
#:
#:     seed 1337  -> 3.737-3.792   FAIL on all five folds
#:     seed 2024  -> 2.9994-2.9995 pass on all five
#:     seed    7  -> 2.386-2.438   pass on four, fail on one
#:     seed   99  -> 1.015-1.061   FAIL on all five
#:     seed 12345 -> 2.299-2.322   FAIL on all five
#:
#: **The epoch-0 prediction is a function of the random seed, not of the
#: training fold's labels.** That is the exact negation of what gate 3
#: asks for, and it means the arm's viability is decided by five draws
#: rather than twenty-five.
#:
#: **A LIMITATION OF GATE 3, RECORDED AS SUCH -- NOT AS A REASON TO
#: CHANGE IT.** Seed 2024 puts **99.94%** of its softmax mass on one class
#: and predicts a constant **2.9995** for every image, on every fold. Both
#: of gate 3's checks pass: the head position is 0.31-0.38 SD out, and the
#: dispersion check passes precisely BECAUSE the predictions are constant,
#: which is the behaviour it exists to require. A maximally collapsed,
#: label-independent head passes gate 3 whenever its saturated class
#: happens to sit near the fold mean. The gate is not wrong -- it was
#: built against a different defect (missing BatchNorm running
#: statistics) and it caught this one on three seeds in five. But its
#: coverage of this failure mode is partial, and that belongs in the
#: record rather than in a wider tolerance.
#:
#: **PROVENANCE OF THESE NUMBERS.** Offline: ``pretrained=False``, random
#: images, and a CPU stand-in for ``roi_align`` because this laptop's
#: torchvision registers it for CUDA only. The MAGNITUDES are therefore
#: not the fold's own. Three things do not depend on the environment and
#: are exact: the closed-form zero-weight prediction, the manuscript
#: path's rms of exactly 1, and ``sd(W) = 0.01804``. The structure -- an
#: offset fixed per seed, sign-random, scaling with ``||f||`` -- is
#: confirmable on the pod with ``p10_scale_probe.py --gate3``, which now
#: prints exactly this decomposition for whichever cell the config names.
#:
#: **THE FOUR REGISTERED OUTCOMES STAND AS WRITTEN.** This is not one of
#: them. The cell never reached epoch 1, so it has not spanned, collapsed,
#: scored, or diverged -- it was refused entry by our harness. Reading it
#: as outcome 2 ("also collapses") would be reading a gate failure as a
#: result, and outcome 2 is about what the architecture does when trained.
GATE_3_FAILED_ON_THE_NOTEBOOK_CELL = {
    "finding": "2026-08-17",
    "run": "p10-cleftgnn-notebook, fold 0",
    "failure": (
        "epoch-0 predicted mean 2.0030 against the training-fold mean "
        "2.7566 -- 0.95 SD, limit 0.5; gate 3's HEAD-POSITION check, the "
        "dispersion check did not fire. Nothing trained, nothing citable"
    ),
    "reading_confirmed": (
        "the cause is exactly W . f. softmax(bias) IS the Laplace "
        "frequency vector, so the zero-weight prediction is the closed "
        "form (N*mean + 15)/(N + 5) = 2.7371 on a 189-fold at mean "
        "2.7302, 0.009 SD out, reproduced numerically to four decimals "
        "with prediction sd 0.000000. Every part of the departure is the "
        "classifier weight times the fused feature"
    ),
    "reading_corrected": (
        "NOT a systematic drag. Across five seeds the epoch-0 mean is "
        "3.7739, 2.9994, 2.4195, 1.0256, 2.3076 -- both directions, "
        "nearly the whole grade range. W is a random draw and there is no "
        "downward force; 2.0030 is one draw, not a tendency"
    ),
    "magnitudes_measured": {
        "manuscript": {
            "rms": 1.0000, "norm2": 32.00,
            "sd_w_dot_f": (0.356, 0.998), "p_max": (0.3905, 0.7020),
        },
        "notebook": {
            "rms": (7.7230, 8.4017), "norm2": (247.14, 268.85),
            "sd_w_dot_f": (2.430, 5.394), "p_max": (0.5885, 0.9994),
        },
        "analytic": (
            "sd((W.f)_k) = sd(W) * ||f||, sd(W) = 0.01804 from nn.Linear's "
            "own bound: 0.577 nats on the manuscript path against ~4.8 on "
            "the notebook path -- EIGHTFOLD -- against a Laplace bias "
            "spanning only ~3.3 nats. The bias loses"
        ),
        "manuscript_rms_is_exact": (
            "1.0000 by construction -- that is what fused_norm does"
        ),
    },
    "fixed_per_seed_not_per_fold": {
        "why": (
            "reset seeds torch and rebuilds, so every fold of a seed gets "
            "the SAME W; only the bias and the images differ, and both are "
            "swamped"
        ),
        "spread_across_five_folds_of_a_seed": (0.0001, 0.0548),
        "per_seed": {
            1337: "3.737-3.792, FAIL on all five folds",
            2024: "2.9994-2.9995, pass on all five",
            7: "2.386-2.438, pass on four, fail on one",
            99: "1.015-1.061, FAIL on all five",
            12345: "2.299-2.322, FAIL on all five",
        },
        "reading": (
            "the epoch-0 prediction is a function of the random SEED, not "
            "of the training fold's labels -- the exact negation of what "
            "gate 3 asks for. The arm's viability is decided by five draws "
            "rather than twenty-five"
        ),
    },
    "gate_3_coverage_limit": (
        "seed 2024 puts 99.94% of its mass on one class and predicts a "
        "constant 2.9995 on every fold, and PASSES both checks -- the "
        "dispersion check passes precisely BECAUSE the predictions are "
        "constant, which is what it exists to require. A maximally "
        "collapsed, label-independent head passes gate 3 whenever its "
        "saturated class sits near the fold mean. RECORDED AS A "
        "LIMITATION, NOT AS A REASON TO WIDEN THE TOLERANCE: the gate was "
        "built against missing BatchNorm running statistics and it caught "
        "this on three seeds in five"
    ),
    # [UPDATED 2026-08-17] It WAS confirmed: see confirmed_on_real_features.
    "provenance": (
        "the table above is OFFLINE (pretrained=False, random images, a "
        "CPU stand-in for roi_align, which this torchvision registers for "
        "CUDA only), so its magnitudes are not the fold's own. Exact and "
        "environment-independent: the closed-form zero-weight prediction, "
        "the manuscript path's rms of exactly 1, and sd(W) = 0.01804. "
        "CONFIRMED on the pod the same day -- confirmed_on_real_features"
    ),
    "the_four_outcomes_stand": (
        "this is NOT one of them. The cell never reached epoch 1, so it "
        "has not spanned, collapsed, scored or diverged -- it was refused "
        "entry by our harness. Reading it as outcome 2 would be reading a "
        "gate failure as a result, and outcome 2 is about what the "
        "architecture does when TRAINED"
    ),
    # **[CONFIRMED ON REAL PRETRAINED FEATURES 2026-08-17,
    # p10_scale_probe --gate3]** Every structural claim holds and the
    # magnitudes are larger, not smaller. Train fold n 190, mean 2.7263,
    # sd 0.7670, gate-3 limit 0.3835.
    "confirmed_on_real_features": {
        "probe": "p10_scale_probe.py --gate3, real pretrained features",
        "fold": "n 190, mean 2.7263, sd 0.7670, limit 0.3835",
        "rms": (7.9966, 8.1894),
        "norm2": (255.9, 262.1),
        "sd_w_dot_f": (2.74, 8.26),
        "p_max": (0.5510, 0.9997),
        "predicted_means": (2.0031, 2.0015, 3.9997, 3.6031, 4.0099),
        "verdict": "ALL FIVE FAIL at 0.94-1.67 SD -- none passes",
        "zero_weight_control": (
            "2.7333 on EVERY seed, 0.009 SD out -- the closed form "
            "(190*2.7263 + 15)/195 verified on real features, so every "
            "departure is W.f alone"
        ),
        "saturation_signature": (
            "p_max reaching 0.9997 with predicted means pinned at 2.0015, "
            "2.0031, 3.9997 and 4.0099 -- the softmax has PICKED A CLASS "
            "before any training step, and the predicted mean is that "
            "class. The mechanism's visible signature"
        ),
        "caveat_lifted": (
            "the ACROSS-SEED spread was the one clause resting on offline "
            "measurement alone; it is now measured at 2.00-4.01 on real "
            "features. The provenance caveat is lifted FROM THAT CLAUSE "
            "and stays on everything else offline-derived"
        ),
    },
}


#: **[PROPOSED 2026-08-17, NOTHING PICKED] WHAT TO DO ABOUT GATE 3 ON THE
#: NOTEBOOK CELL.**
#:
#: Route (d) -- relax gate 3 -- is CLOSED by the maintainer before the
#: options were written, and correctly: loosening a guard because a run
#: failed it is the shape this project has recorded as a defect every
#: time it appeared. It is listed below only so that its absence is
#: visible rather than assumed.
#:
#: **(a) RECORD AS AN OUTCOME. No number.** Phase 10 reports that the
#: notebook recipe cannot start calibrated under this harness, with the
#: mechanism and the per-seed lottery as the finding.
#: *For*: it is true, it is measured, and it is a real result about the
#: interaction between their architecture and our protocol -- the
#: architecture has no normalisation before its classifier, so its
#: untrained output is a property of the seed. It costs nothing and risks
#: nothing.
#: *Against*: it answers none of the four registered outcomes. The cell
#: was built to measure whether the notebook's recipe spans where the
#: manuscript's collapsed, and a gate failure at epoch 0 leaves that
#: question exactly where it was.
#:
#: **(b) ZERO THE CLASSIFIER WEIGHT ON THIS PATH.** Epoch 0 then predicts
#: the bias regardless of feature scale; registered as this cell's fourth
#: deviation, on the argument that a calibrated start is a HARNESS
#: requirement rather than a recipe property.
#: *For*: the argument is sound. Gate 3 is ours, not theirs; the notebook
#: has no notion of epoch-0 calibration, and its own default init would
#: fail just as surely. The zero-weight init touches nothing the notebook
#: specifies about its architecture, and it is the exact limit of "make
#: ``W . f`` small enough that the bias governs epoch 0".
#: *Against, and it is measured rather than hypothetical*: **it
#: reintroduces the artifact FIX (b) was made to remove.** With ``W = 0``
#: every image receives identical logits, so the epoch-0 prediction
#: vector is CONSTANT BY CONSTRUCTION -- measured sd 0.000000 -- and any
#: epoch-0 PCC against it is nan for a reason unrelated to training
#: (``SD_ZERO_HAS_TWO_CAUSES``). Training is unaffected, since the monitor
#: is ``inner_val_mse``; the curves carry a construction-nan at epoch 0.
#: And it is asymmetric: we would be putting back on one cell exactly what
#: we withdrew from the other.
#:
#: **(c-i) DO (a) REGARDLESS, and treat (b) as a SEPARATE second
#: decision.** The incompatibility is a finding whether or not a number
#: follows it. Framing this as a choice between (a) and (b) loses the
#: finding in the case where (b) is taken -- and (b) does not make the
#: gate-3 failure untrue, it works around it.
#:
#: **(c-ii) SCALE-MATCHED INIT.** Initialise the classifier weight to
#: ``nn.Linear``'s own draw scaled by ``32 / ||f||`` measured on the first
#: training batch, so ``W . f`` has the same 0.577-nat spread the
#: manuscript cell has, and epoch 0 is calibrated to the same tolerance.
#: *For*: it keeps epoch-0 predictions NON-constant, so no construction-nan
#: and no reintroduced artifact; it makes the two cells' starting
#: conditions matched rather than merely both passing.
#: *Against*: more machinery than (b) and more OURS than (b) -- a
#: data-dependent init nobody's artifact prescribes, requiring a forward
#: before training. (b) is one line; this is a mechanism.
#:
#: **(c-iii) NAMED IN ORDER TO BE REJECTED: zero the weight for the gate
#: only, then restore it.** This makes gate 3 pass while the model that
#: trains is the one that would have failed. It is worse than relaxing the
#: gate, because it hides exactly what the gate exists to detect. Listed
#: because it is the cheapest-looking option and someone will think of it.
#:
#: **(c-iv) NAMED IN ORDER TO BE REJECTED: give the notebook path
#: ``fused_norm``.** It would work -- rms 1, perturbation 0.577, epoch 0
#: calibrated. It is ruled out by the cell's OWN registration, in its own
#: words: neither LayerNorm, because carrying ours would make a spanning
#: result unreadable -- their recipe or our repairs, and no way to tell.
#: Ruled out by the registration, not by taste.
#:
#: **(d) CLOSED: relax gate 3.** Not proposed. Recorded as closed.
#:
#: **What is NOT in doubt**: the four registered outcomes stand as
#: written, and this failure is recorded as its own dated finding
#: (``GATE_3_FAILED_ON_THE_NOTEBOOK_CELL``) whichever route is taken.
GATE_3_ROUTES_PROPOSED = {
    "proposed": "2026-08-17, nothing picked",
    "routes": {
        "a_record_as_outcome": {
            "what": (
                "Phase 10 reports that the notebook recipe cannot start "
                "calibrated under this harness; the mechanism and the "
                "per-seed lottery are the finding. No number"
            ),
            "for": (
                "true, measured, and a real result about the interaction "
                "between their architecture and our protocol -- no "
                "normalisation before the classifier means the untrained "
                "output is a property of the seed. Costs and risks nothing"
            ),
            "against": (
                "answers none of the four registered outcomes; the cell "
                "was built to measure whether the notebook's recipe spans "
                "where the manuscript's collapsed, and a gate failure at "
                "epoch 0 leaves that exactly where it was"
            ),
        },
        "b_zero_the_classifier_weight": {
            "what": (
                "epoch 0 predicts the bias regardless of feature scale; "
                "this cell's fourth deviation, on the argument that a "
                "calibrated start is a HARNESS requirement rather than a "
                "recipe property"
            ),
            "for": (
                "the argument is sound: gate 3 is ours, the notebook has "
                "no notion of epoch-0 calibration and its own default init "
                "would fail just as surely. It touches nothing the "
                "notebook specifies, and is the exact limit of 'make W.f "
                "small enough that the bias governs epoch 0'"
            ),
            "against_measured": (
                "it REINTRODUCES the artifact FIX (b) removed: with W = 0 "
                "every image gets identical logits, so epoch-0 predictions "
                "are constant by construction (measured sd 0.000000) and "
                "epoch-0 PCC is nan for a non-training reason "
                "(SD_ZERO_HAS_TWO_CAUSES). Training is unaffected -- the "
                "monitor is inner_val_mse -- but the curves carry a "
                "construction-nan, and it is asymmetric: putting back on "
                "one cell what we withdrew from the other"
            ),
        },
        "c_i_do_a_regardless": (
            "the incompatibility is a finding whether or not a number "
            "follows. Framing this as a choice between (a) and (b) loses "
            "the finding in the case where (b) is taken -- and (b) does "
            "not make the gate-3 failure untrue, it works around it"
        ),
        "c_ii_scale_matched_init": {
            "what": (
                "nn.Linear's own draw scaled by 32 / ||f|| measured on the "
                "first training batch, so W.f carries the manuscript "
                "cell's 0.577-nat spread and epoch 0 is calibrated to the "
                "same tolerance"
            ),
            "for": (
                "epoch-0 predictions stay NON-constant -- no "
                "construction-nan, no reintroduced artifact -- and the two "
                "cells' starting conditions are matched rather than merely "
                "both passing"
            ),
            "against": (
                "more machinery and more OURS than (b): a data-dependent "
                "init nobody's artifact prescribes, needing a forward "
                "before training. (b) is one line; this is a mechanism"
            ),
        },
        "c_iii_rejected_zero_for_the_gate_only": (
            "makes gate 3 pass while the model that TRAINS is the one that "
            "would have failed -- worse than relaxing the gate, because it "
            "hides exactly what the gate exists to detect. Named because "
            "it is the cheapest-looking option"
        ),
        "c_iv_rejected_give_it_fused_norm": (
            "it would work -- rms 1, perturbation 0.577, epoch 0 "
            "calibrated -- and is ruled out by the cell's OWN "
            "registration, in its own words: neither LayerNorm, because "
            "carrying ours would make a spanning result unreadable. Ruled "
            "out by the registration, not by taste"
        ),
        "d_closed_relax_gate_3": (
            "CLOSED by the maintainer before the options were written, and "
            "correctly: loosening a guard because a run failed it is the "
            "shape this project has recorded as a defect every time. "
            "Listed so its absence is visible rather than assumed"
        ),
    },
    "not_in_doubt": (
        "the four registered outcomes stand as written, and this failure "
        "is recorded as its own dated finding whichever route is taken"
    ),
}


#: **[FINDING 2026-08-17, TAKEN UNCONDITIONALLY -- route (c-i), the
#: maintainer's decision] THE NOTEBOOK RECIPE CANNOT START CALIBRATED UNDER
#: THIS PROTOCOL. THE CELL YIELDS A FINDING, NOT A NUMBER.**
#:
#: **The mechanism, measured.** The notebook's multiplicative fusion sends
#: the classifier a feature of roughly **eight times** the manuscript
#: path's magnitude -- rms **7.72-8.40** against **exactly 1.0**, which is
#: what ``fused_norm`` produces by construction -- into a random
#: ``nn.Linear`` draw. The resulting class-to-class spread of ``W . f``
#: reaches **2.43-5.39 nats** against a Laplace bias spanning only about
#: **3.3**. The bias loses, and the epoch-0 prediction becomes a function
#: of the SEED rather than of the fold's labels: **1.026 to 3.774 across
#: seeds, 0.0001 to 0.055 within a seed's five folds.**
#:
#: **The control, in closed form.** ``softmax(bias)`` IS the Laplace
#: frequency vector, so a zero-weight head predicts
#: ``(N * mean + 15) / (N + 5)`` exactly -- **2.7371** on a 189-patient
#: fold at mean 2.7302, **0.009 SD** out, verified numerically to four
#: decimals with prediction sd **0.000000**. Every part of the observed
#: departure is ``W . f`` and nothing else.
#:
#: **Provenance, attached and travelling.** OFFLINE: ``pretrained=False``,
#: random images, a CPU stand-in for ``roi_align``. The magnitudes are not
#: the fold's own. What does NOT depend on the environment: the closed
#: form, the manuscript path's rms of exactly 1, ``sd(W) = 0.01804`` from
#: ``nn.Linear``'s own bound, and the per-seed constancy, which follows
#: from ``reset`` re-seeding torch and rebuilding so that every fold of a
#: seed receives the same ``W``. The real-feature datum is the failing run
#: itself: 2.0030 against 2.7566, 0.95 SD.
#:
#: **THIS IS NONE OF THE FOUR REGISTERED OUTCOMES, and reading it as one
#: is forbidden.** The cell was refused entry at gate 3, fold 0, before a
#: single optimiser step. It has not spanned, it has not collapsed, it has
#: not scored, and it has not diverged -- those describe what an
#: architecture does when TRAINED, and this one never trained. Reading a
#: gate failure as a collapse result would let a harness refusal
#: masquerade as a measurement of their architecture.
#:
#: **What the finding IS.** A result about the interaction between their
#: architecture and our protocol, and a real one: an architecture with no
#: normalisation before its classifier has an untrained output that is a
#: property of its initialisation, not of its data. Our harness requires
#: the opposite and says so at gate 3. Both are defensible; they are
#: incompatible. That is the finding, and it needed no epoch to establish.
NOTEBOOK_CELL_CANNOT_START_CALIBRATED = {
    "finding": "2026-08-17, route (c-i) taken unconditionally",
    "statement": (
        "the notebook recipe cannot start calibrated under this protocol; "
        "the cell yields a FINDING, not a number"
    ),
    "mechanism": (
        "the multiplicative fusion sends the classifier a feature of ~8x "
        "the manuscript path's magnitude -- rms 7.72-8.40 against exactly "
        "1.0 by fused_norm -- into a random nn.Linear draw, so sd(W.f) "
        "reaches 2.43-5.39 nats against a Laplace bias spanning ~3.3. The "
        "bias loses"
    ),
    "epoch_0_is_a_seed_function": {
        # [2026-08-17] CONFIRMED on real pretrained features: the
        # across-seed spread is 2.00-4.01 there, so this clause no longer
        # carries the offline caveat. The offline figures stay beside it
        # as what was predicted before the pod ran.
        "across_seeds_real_features": (2.0015, 4.0099),
        "across_seeds_offline": (1.026, 3.774),
        "within_a_seeds_five_folds": (0.0001, 0.055),
        "reading": (
            "the epoch-0 prediction is a function of the SEED, not of the "
            "fold's labels -- the exact negation of what gate 3 requires"
        ),
        "and_on_real_features_none_passes": (
            "all five seeds fail at 0.94-1.67 SD; the offline pass had one "
            "passing seed and it did not survive real features"
        ),
    },
    "closed_form_control": {
        "identity": "(N * mean + 15) / (N + 5)",
        "why": "softmax(bias) IS the Laplace frequency vector",
        "value": 2.7371,
        "fold": "189 patients at mean 2.7302",
        "sd_out": 0.009,
        "verified": (
            "numerically to four decimals, prediction sd 0.000000 -- so "
            "every part of the observed departure is W.f and nothing else"
        ),
    },
    # [UPDATED 2026-08-17, after the pod probe] The finding was recorded
    # from offline measurement with the caveat attached. The pod has now
    # measured it on real pretrained features and every structural claim
    # held, with LARGER magnitudes: rms 7.9966-8.1894, sd(W.f) 2.74-8.26,
    # p_max to 0.9997, and ALL FIVE seeds failing at 0.94-1.67 SD where
    # the offline pass had one passing. The zero-weight control returned
    # 2.7333 on every seed, 0.009 SD out -- the closed form verified on
    # real features.
    "provenance": (
        "CONFIRMED ON REAL PRETRAINED FEATURES (p10_scale_probe --gate3): "
        "rms 7.9966-8.1894, sd(W.f) 2.74-8.26, p_max to 0.9997, all five "
        "seeds failing at 0.94-1.67 SD, zero-weight control 2.7333 on "
        "every seed at 0.009 SD. The original record was OFFLINE "
        "(pretrained=False, random images, a CPU stand-in for roi_align) "
        "with the caveat attached; the pod pass kept every structural "
        "claim and made the magnitudes larger, not smaller"
    ),
    "none_of_the_four_outcomes": (
        "FORBIDDEN to read as one. The cell was refused entry at gate 3 "
        "fold 0, before a single optimiser step: it has not spanned, "
        "collapsed, scored or diverged, and those describe what an "
        "architecture does when TRAINED. Reading a gate failure as a "
        "collapse result would let a harness refusal masquerade as a "
        "measurement of their architecture"
    ),
    "what_it_is": (
        "a result about the interaction between their architecture and "
        "our protocol: an architecture with no normalisation before its "
        "classifier has an untrained output that is a property of its "
        "initialisation, not of its data. Our harness requires the "
        "opposite and says so at gate 3. Both are defensible; they are "
        "incompatible -- and establishing that needed no epoch"
    ),
}


#: **[CLOSED 2026-08-17, BY OPERATOR DECISION] THE NOTEBOOK CELL'S
#: DISPOSITION IS SETTLED, AND PREFERENCE CANNOT REOPEN IT.**
#:
#: Registered as a closure rather than a conclusion so that a later turn
#: cannot revisit it by wanting a different answer. **The notebook cell
#: yields a finding, not a number.** Every route that would have produced
#: a number was considered and answered, each on a stated ground:
#:
#: * **(b) zero the classifier weight** -- **NO**, on the measured cost.
#:   It reintroduces precisely what ``FUSED_NORM_AND_STANDARD_INIT``'s fix
#:   (b) removed: constant epoch-0 predictions, sd 0.000000, a
#:   construction-nan in the curves (``SD_ZERO_HAS_TWO_CAUSES``). It would
#:   put back on one cell what we withdrew from the other, in order to
#:   satisfy a gate the notebook has no notion of.
#: * **(c-ii) scale-matched init** -- **NO**: more machinery than (b) and
#:   more OURS than (b), for the same purpose.
#: * **(c-iii) zero the weight for the gate only** -- **NO**: it hides
#:   what the gate exists to detect.
#: * **(c-iv) give the notebook path ``fused_norm``** -- **NO**: ruled out
#:   by the cell's own registration, which forbids carrying our
#:   normalisation onto this path.
#: * **(d) relax gate 3** -- **NO**, closed before the options were
#:   written.
#:
#: **THE BAR FOR REOPENING, stated now so it cannot be lowered later**: a
#: future attempt to make this cell run must arrive with **a new measured
#: reason** -- a fact about the architecture, the harness or the data that
#: is not in this record -- and not with a wish for a PCC. "We would like
#: a number from this cell" is not a reason; it is the thing the closure
#: exists to refuse. A new documented artifact from the group would
#: qualify. A better mood would not.
NOTEBOOK_CELL_DISPOSITION_CLOSED = {
    "closed": "2026-08-17, maintainer decision",
    "disposition": "the notebook cell yields a FINDING, not a number",
    "routes_answered_no": {
        "b_zero_the_classifier_weight": (
            "the measured cost: it reintroduces what fix (b) removed -- "
            "constant epoch-0 predictions, sd 0.000000, a construction-nan "
            "in the curves -- putting back on one cell what we withdrew "
            "from the other, to satisfy a gate the notebook has no notion "
            "of"
        ),
        "c_ii_scale_matched_init": (
            "more machinery than (b) and more OURS than (b), for the same "
            "purpose"
        ),
        "c_iii_zero_for_the_gate_only": (
            "it hides what the gate exists to detect"
        ),
        "c_iv_give_it_fused_norm": (
            "ruled out by the cell's own registration, which forbids "
            "carrying our normalisation onto this path"
        ),
        "d_relax_gate_3": "closed before the options were written",
    },
    "bar_for_reopening": (
        "a NEW MEASURED REASON -- a fact about the architecture, the "
        "harness or the data that is not in this record. Not a wish for a "
        "PCC: 'we would like a number from this cell' is not a reason, it "
        "is the thing this closure exists to refuse. A new documented "
        "artifact from the group would qualify; a better mood would not"
    ),
}


#: **[STANDING FINDING 2026-08-17] GATE 3 DOES NOT DETECT A COLLAPSED
#: HEAD WHOSE SATURATED CLASS SITS NEAR THE FOLD MEAN.**
#:
#: Recorded in its own right, for whoever next relies on gate 3 as
#: evidence that a run started calibrated.
#:
#: **The measurement.** Seed 2024 on the notebook cell puts **99.94%** of
#: its softmax mass on a single class and predicts a constant **2.9995**
#: for every image, on every one of five folds -- a head that has not
#: looked at its input. It **passes both of gate 3's checks**:
#:
#: * head position, 0.31-0.38 SD out against a 0.5 limit; and
#: * dispersion, which passes *precisely because* the predictions are
#:   constant -- emitting one value is the behaviour that check exists to
#:   require.
#:
#: **Why the gate is not wrong.** It was built against a specific,
#: measured defect: ImageNet SR-GNN arms starting at ``inner_val_mse``
#: 1.12-8.63 for want of BatchNorm running statistics, which produced two
#: void ladders. Against that defect it works, and against this one it
#: fired on three seeds in five. A guard that catches most instances of an
#: unanticipated failure mode is a good guard.
#:
#: **What it means in practice.** *Gate 3 passing is evidence that the
#: head is positioned near the fold mean. It is NOT evidence that the head
#: depends on its input.* Those come apart exactly when a saturated class
#: happens to sit near the mean, and on a 1-5 grade scale with a cohort
#: centred near 2.7-2.8 that is grade 3 -- a coincidence available on
#: every fold of this project.
#:
#: **The tolerances are untouched, deliberately.** Widening or narrowing
#: them would not address this: the failure passes *inside* the band. What
#: would address it is a separate check on prediction sd against something
#: other than a constant baseline -- and that is a proposal for a phase
#: that is not closing, made by someone who has measured what it would
#: cost. **Recorded, not fixed.**
GATE_3_COVERAGE_LIMIT = {
    "standing_finding": "2026-08-17",
    "for": "whoever next relies on gate 3 as evidence of a calibrated start",
    "measurement_offline": (
        "seed 2024 on the notebook cell puts 99.94% of its softmax mass on "
        "one class and predicts a constant 2.9995 on every one of five "
        "folds -- a head that has not looked at its input -- and PASSES "
        "both checks: head position 0.31-0.38 SD against a 0.5 limit, and "
        "dispersion, which passes precisely BECAUSE the predictions are "
        "constant. OFFLINE ONLY; see `corrected`"
    ),
    # [CORRECTED 2026-08-17, on the real-feature probe] "three of five"
    # was the OFFLINE pass. On real pretrained features ALL FIVE seeds
    # fail (0.94-1.67 SD), so this cell does not exemplify the coverage
    # limit on real data at all. The limitation itself stands, and now
    # stands on better ground -- see `limitation_is_structural`.
    "corrected": (
        "2026-08-17: 'three of five' was the OFFLINE pass. On real "
        "pretrained features ALL FIVE seeds FAIL (0.94-1.67 SD), so this "
        "cell does NOT exemplify the coverage limit on real data. The "
        "saturated classes drawn were 2 and 4; grade 3 -- 0.27 from the "
        "fold mean, 0.36 SD, inside the 0.5 limit -- WOULD have passed "
        "and simply did not come up in five draws"
    ),
    "the_gate_is_not_wrong": (
        "it was built against a measured defect -- ImageNet SR-GNN arms at "
        "inner_val_mse 1.12-8.63 for want of BatchNorm running statistics, "
        "which produced two void ladders -- and it caught this "
        "unanticipated one on EVERY real-feature seed"
    ),
    # The limitation survives losing its exemplar, because it never
    # needed one: it is provable from gate 3's own arithmetic.
    "limitation_is_structural": (
        "the dispersion check computes expected = var(inner) + "
        "(mean(inner) - pred_mean)^2, which for a CONSTANT predictor is "
        "the exact MSE -- so the ratio is 1 and it passes BY IDENTITY -- "
        "and the head-position check tests position alone. A constant "
        "predictor whose value sits within 0.5 SD of the fold mean "
        "therefore passes both by arithmetic rather than by luck. The "
        "offline seed-2024 row was an INSTANCE, now known not to occur on "
        "real features for this cell; the property does not depend on it"
    ),
    "what_it_means": (
        "gate 3 passing is evidence that the head is POSITIONED near the "
        "fold mean. It is NOT evidence that the head DEPENDS on its input. "
        "The two come apart exactly when a saturated class sits near the "
        "mean -- on a 1-5 scale with a cohort centred near 2.7-2.8 that is "
        "grade 3, a coincidence available on every fold of this project"
    ),
    "tolerances_untouched": (
        "deliberately: the failure passes INSIDE the band, so widening or "
        "narrowing would not address it. What would is a separate check on "
        "prediction sd against a non-constant baseline -- a proposal for a "
        "phase that is not closing, from someone who has measured what it "
        "costs. RECORDED, NOT FIXED"
    ),
}


#: **[WALKED 2026-08-17] WHERE PHASE 10 STANDS AGAINST ITS SIX EXIT
#: CRITERIA, with the notebook cell's disposition included.**
#:
#: 1. **Discrepancy list -- DONE, and it is the phase's largest
#:    deliverable.** It began as one item (the ROI count) and now runs to
#:    seven manuscript-vs-artifact discrepancies plus one ambiguity, every
#:    one verified at source: the region count (now BELIEF-vs-EXECUTION --
#:    three statements of 27 against code that runs 36); the optimiser
#:    (SGD 0.01 against Adam 0.001); the fusion (additive eq (9) against
#:    ``f_t + f_t*v``); the backbone (ResNet-50 against ViT-B/16); SABM
#:    (in the manuscript, absent from the notebook, and a THIRD
#:    description -- crop scoping -- in the deck); the 90.94% (CIFAR-10,
#:    confirmed at source); the epoch budget (the notebook states 5, the
#:    manuscript states none); and the IEM's direction, which is
#:    ambiguous and blocks Phase 11.
#: 2. **Deviations enumerated -- DONE, with the dated correction the
#:    closing must carry.** The criterion anticipated ONE. The manuscript
#:    cell carries FIVE, and one of those five had its fidelity cost
#:    corrected UPWARD on reading the notebook. The notebook cell carries
#:    THREE of its own, registered and now unexercised. A closing that
#:    reported "the deviations were enumerated" without that arithmetic
#:    would be true and misleading.
#: 3. **Five seeds -- DONE** on the manuscript cell: 25 folds, with the
#:    per-fold curves and the selected-epoch stage tables now written
#:    rather than discarded.
#: 4. **Paired BCa vs 0.2520 -- NOT DONE. The only outstanding RUN.**
#: 5. **Ledgered -- DONE (entry 25).** The notebook cell's finding is NOT
#:    a ledger entry: the ledger takes intervalled measurements and this
#:    is a structural finding with no interval. It lives here and in the
#:    closing.
#: 6. **Suite -- DONE**, green throughout.
#:
#: **The amendment item (the notebook cell) -- DISPOSED**, not
#: outstanding: ``NOTEBOOK_CELL_CANNOT_START_CALIBRATED``, closed against
#: reopening by ``NOTEBOOK_CELL_DISPOSITION_CLOSED``.
#:
#: **IS THE PAIRED BCa THE ONLY OUTSTANDING RUN? YES** -- and it needs no
#: training: the per-seed OOF CSVs exist, so it is a p7d-shaped two-pass
#: config over artifacts already on disk. Everything else remaining is
#: WRITING, not running:
#:
#: * the faithful arm's own closing sentence (still nan on four of five
#:   raters; it must not be absorbed into the protocol arm's);
#: * the closing record itself, carrying criterion 2's arithmetic;
#: * the five-line recording change from ``U_SHAPE_BELONGS_LATER`` -- all
#:   epochs' stage tables rather than only the selected epoch's. No run
#:   attached; it makes the next run capture the U-shape for free.
#:
#: **ONE OPTIONAL RUN, named rather than assumed.**
#: ``p10_scale_probe.py --gate3`` on the notebook config would confirm the
#: gate-3 magnitudes on real pretrained features. It is NOT required:
#: three of the finding's four supports are environment-independent, and
#: the fourth is the failing run itself. What it would firm up is the one
#: clause that rests on offline measurement alone -- the ACROSS-SEED
#: spread of 1.026 to 3.774. Cheap, no training, no artifact. The
#: maintainer's call, and the finding stands either way with its provenance
#: caveat attached.
PHASE_10_EXIT_WALK = {
    "walked": "2026-08-17",
    "criteria": {
        "1_discrepancy": (
            "DONE, and the phase's largest deliverable: seven "
            "manuscript-vs-artifact discrepancies plus one ambiguity, all "
            "verified at source -- region count (belief-vs-execution), "
            "optimiser, fusion, backbone, SABM (three descriptions), the "
            "90.94% (CIFAR-10), the epoch budget, and the IEM's direction"
        ),
        "2_deviations_enumerated": (
            "DONE with a dated correction the closing must carry: the "
            "criterion anticipated ONE, the manuscript cell carries FIVE "
            "(one with its fidelity cost corrected upward on reading the "
            "notebook), and the notebook cell carries THREE of its own, "
            "registered and now unexercised"
        ),
        "3_five_seeds": (
            "DONE on the manuscript cell -- 25 folds, per-fold curves and "
            "selected-epoch stage tables written rather than discarded"
        ),
        # [RESOLVED 2026-08-17, same day] It ran, and it WITHDREW.
        "4_paired_bca": (
            "DONE 2026-08-17 -- and WITHDRAWN: d +0.2279, condition 1 "
            "FALSE, condition 2 TRUE at 3.76x (PAIRED_BCA_OBSERVED). "
            "Recorded here as it stood when the walk was made: 'NOT DONE "
            "-- the only outstanding RUN'"
        ),
        "5_ledgered": (
            "DONE (entry 25). The notebook cell's finding is NOT a ledger "
            "entry: the ledger takes intervalled measurements and this is "
            "a structural finding with no interval"
        ),
        "6_suite": "DONE, green throughout",
    },
    "amendment_item_notebook_cell": (
        "DISPOSED, not outstanding -- "
        "NOTEBOOK_CELL_CANNOT_START_CALIBRATED, closed against reopening "
        "by NOTEBOOK_CELL_DISPOSITION_CLOSED"
    ),
    "paired_bca_is_the_only_outstanding_run": True,
    "and_it_needs_no_training": (
        "the per-seed OOF CSVs exist; a p7d-shaped two-pass config over "
        "artifacts already on disk"
    ),
    "remaining_is_writing_not_running": (
        "the faithful arm's own closing sentence -- still nan on four of "
        "five raters, and it must not be absorbed into the protocol arm's",
        "the closing record itself, carrying criterion 2's arithmetic",
        "the five-line recording change from U_SHAPE_BELONGS_LATER: all "
        "epochs' stage tables rather than only the selected epoch's, no "
        "run attached",
    ),
    "one_optional_run": (
        "p10_scale_probe.py --gate3 on the notebook config would confirm "
        "the gate-3 magnitudes on real pretrained features. NOT required: "
        "three of the finding's four supports are "
        "environment-independent and the fourth is the failing run "
        "itself. It would firm up the one clause resting on offline "
        "measurement alone -- the ACROSS-SEED spread of 1.026 to 3.774. "
        "Cheap, no training, no artifact; a maintainer decision, and the "
        "finding stands either way with its provenance caveat"
    ),
}


#: What Phase 10's paired scope covers, and what it does not.
#:
#: **ONE pair.** The replication arm against the 0.2520 arm, on the five
#: shared seeds, over the same 237 patients and the same fold column. It
#: is the phase's fourth exit criterion and the last outstanding run.
#:
#: **What it cannot cover, and why each is excluded rather than missing:**
#:
#: * **the notebook cell** -- it has no vectors. It was refused entry at
#:   gate 3 and yields a finding, not a number
#:   (``NOTEBOOK_CELL_CANNOT_START_CALIBRATED``).
#: * **the faithful arm** -- a single 85:15 run with no seed band and no
#:   OOF vectors over the cohort. Its form is theirs, deliberately, and
#:   nothing in it is claimable by construction
#:   (``FAITHFUL_ARM_REGISTERED``).
#: * **the rater screen** -- each rater is a DIFFERENT target, so there is
#:   no common truth vector to pair on. The same exclusion the ladder
#:   applies to its label question.
PAIRED_CLAIM_COVERAGE = {
    "covers": (
        "one pair: the replication arm against the 0.2520 arm, five "
        "shared seeds, same 237 patients and same fold column"
    ),
    "excluded": {
        "notebook_cell": (
            "no vectors -- refused entry at gate 3, yields a finding not "
            "a number (NOTEBOOK_CELL_CANNOT_START_CALIBRATED)"
        ),
        "faithful_arm": (
            "a single 85:15 run with no seed band and no OOF vectors over "
            "the cohort; nothing in it is claimable by construction"
        ),
        "rater_screen": (
            "each rater is a DIFFERENT target, so there is no common "
            "truth vector to pair on -- the exclusion the ladder applies "
            "to its label question"
        ),
    },
}

#: The stem the replication arm's keeper run carries, and the seeds both
#: sides share. Named here so the generator and the record cannot drift.
PAIRED_ARM_STEM = "p10_cleftgnn"
PAIRED_BASELINE_STEM = "p7_d1_vit_b16_imagenet_g1"


def paired_claim_pairs(scope: str = "p10") -> list[dict]:
    """Phase 10's paired scope, shaped exactly like the ladder's and Road
    B's so the ONE paired implementation serves all three.

    ``task_paired_claims`` walks these, ``ladder``'s three derivations
    take them as an argument, and ``phase7b.paired_comparison`` does the
    arithmetic. Nothing here re-implements a comparison -- the Road B
    precedent, applied.

    **Every recorded figure is DERIVED, not typed.** The delta comes from
    ``ROUTE_3_OBSERVED`` and ``ladder.STAGE_D1_AT_G1``, and the threshold
    from the frozen ``combined_claimable_delta`` -- so a later correction
    to either measurement lands here automatically rather than in one
    record and not the other. That is the same reason the 7D scope
    derives its deltas from ``PHASE_7D_OBSERVED``.

    **``STAGE_D1_AT_G1``, not ``CLEAN_GEOMETRY_MEASUREMENTS``**, and the
    difference is 0.0001 rather than nothing: the arm has TWO recorded
    figures -- 0.2520/0.0148 through the D1 artifact path and
    0.25206/0.01483 live through ``p3_train_cv``. **The vectors this pair
    declares are the D1 run's**, so the D1 figures are the ones that
    describe them. Sourcing the live pair would have made the recorded
    delta 0.2280 for vectors that produced 0.2279 -- a discrepancy too
    small to notice and exactly the kind this project has paid for.
    """
    if scope != "p10":
        raise ValueError(f"unknown Phase 10 paired-claim scope {scope!r}")

    from . import ladder
    from .train.phase3 import combined_claimable_delta

    baseline_mean = ladder.STAGE_D1_AT_G1["cells"]["vit_b16"][0]
    baseline_sd = ladder.STAGE_D1_AT_G1["sd"]["vit_b16"]["imagenet"]
    seeds = list(ladder.SEED_POOL[:5])
    arm_mean = ROUTE_3_OBSERVED["pcc_mean"]
    arm_sd = ROUTE_3_OBSERVED["pcc_seed_sd"]
    threshold = combined_claimable_delta(
        baseline_sd, len(seeds), arm_sd, len(seeds)
    )["arm_means_95"]
    delta = baseline_mean - arm_mean
    return [{
        "key": "p10__replication_vs_0p2520",
        "question": "replication",
        "varies": "architecture+recipe",
        # `b` is the 0.2520 arm, so a POSITIVE delta means the baseline is
        # ahead -- the headline scope's convention, kept.
        "a": PAIRED_ARM_STEM,
        "b": PAIRED_BASELINE_STEM,
        "seeds": seeds,
        "recorded": {
            "delta_of_means": round(delta, 4),
            "threshold": round(threshold, 4),
            "margin": round(delta / threshold, 2),
            "source": (
                "ROUTE_3_OBSERVED and ladder.STAGE_D1_AT_G1 (the D1 "
                "artifact path, which is what the declared vectors are); "
                "DESCRIPTIVE, not claimable -- the run computes the real "
                "one"
            ),
        },
    }]


#: **[REGISTERED 2026-08-17, BUILT, NOT LAUNCHED] THE PAIRED BCa -- PHASE
#: 10'S FOURTH EXIT CRITERION AND ITS LAST OUTSTANDING RUN.**
#:
#: ``configs/p10_paired.yaml``. **It fits nothing**: every vector it reads
#: was written by a ``cleftgnn_cv`` keeper run that already happened, and
#: by the 0.2520 arm's keeper run from Phase 7. No GPU, no training, no
#: re-extraction.
#:
#: **THE PREDICTION, REGISTERED BEFORE THE RUN, WITH ITS CAUSE.**
#: Delta **0.2279**, threshold **0.0607**, margin **3.76x** -- derived,
#: not typed, from the two arms' recorded means and seed SDs.
#:
#: **Where 3.76x lands, from the margin table**: INSIDE the mixed region
#: 3.6x-4.7x, where 3.83x passed while 4.34x and 4.69x failed. **The
#: margin does not order it.** Condition 1 decides, which is the whole
#: point of running it rather than reasoning about it.
#:
#: **THE CAUSE OF SO SMALL A MARGIN FOR SO LARGE A DELTA, and it is the
#: interesting part**: the replication's seed SD is **0.0676 against the
#: baseline's 0.0148 -- 4.6x** -- and the threshold is driven by the
#: noisier arm. **An unstable arm is hard to beat CLAIMABLY even when it
#: is far behind.** A delta of 0.2279 on a 0-1 scale would be
#: overwhelming between two stable arms; here it clears its threshold only
#: 3.76-fold, entirely because the loser wobbles.
#:
#: **BOTH READINGS COMMITTED NOW:**
#:
#: * **claimable** -> the 0.2520 arm is claimably ahead **under this
#:   protocol**. That is a statement about our protocol's outcome, NOT
#:   about architectural superiority -- the loser does not span the label
#:   range, so "ahead of an arm that barely predicts" is the honest
#:   reading and the one that must travel.
#: * **not claimable** -> Phase 10 cannot claim even that, and closes on
#:   the comparison being unresolvable at this cohort's resolution -- a
#:   **fourth independent arrival** at ``COHORT_CANNOT_RESOLVE``.
#:
#: **Two passes, the standing flow.** The 0.2520 arm's five vectors carry
#: real hashes from ``p7d_paired.yaml`` -- a path's contents are
#: immutable, so one hash serves every config declaring it. The
#: replication's five are ``PENDING_p10_cleftgnn`` with all-zero
#: PLACEHOLDER hashes: a run directory carries a SHA and a job id, neither
#: derivable on a laptop. Paste the run directory from the listing, then
#: ``declare_inputs.py``. **Guard 3 refuses the run until both are real.**
#:
#: **The vectors are p10-cleftgnn-6's**, the citable run. Its training
#: path is identical to p10-cleftgnn-5's, whose per-seed PCCs the
#: prediction is derived from: the stage recording added between them
#: rides a forward that already happened and consumes no RNG. If the
#: declared vectors disagree with that, the prediction moves and the
#: config does not -- which is why the recorded figures are marked
#: DESCRIPTIVE.
PHASE_10_PAIRED_REGISTERED = {
    "registered": "2026-08-17, built, NOT launched",
    "config": "configs/p10_paired.yaml",
    "fits_nothing": (
        "every vector was written by a keeper run that already happened; "
        "no GPU, no training, no re-extraction"
    ),
    "prediction": {
        "delta": 0.2279,
        "threshold": 0.0607,
        "margin": 3.76,
        "derived_from": (
            "ROUTE_3_OBSERVED's pcc_mean/pcc_seed_sd and "
            "ladder.STAGE_D1_AT_G1's 0.2520/0.0148, through the frozen "
            "combined_claimable_delta -- not typed"
        ),
        "which_baseline_figure": (
            "the D1 ARTIFACT path's 0.2520/0.0148, not p3_train_cv's live "
            "0.25206/0.01483 -- the declared vectors are the D1 run's, so "
            "the D1 figures are the ones that describe them. The live "
            "pair would have recorded 0.2280 for vectors that produced "
            "0.2279: a discrepancy too small to notice, and exactly the "
            "kind this project has paid for"
        ),
    },
    "where_the_margin_lands": (
        "INSIDE the mixed region 3.6x-4.7x, where 3.83x passed while "
        "4.34x and 4.69x failed. The margin does not order it; condition "
        "1 decides, which is why it is run rather than reasoned about"
    ),
    "cause_of_the_small_margin": (
        "the replication's seed SD is 0.0676 against the baseline's "
        "0.0148 -- 4.6x -- and the threshold is driven by the noisier "
        "arm. AN UNSTABLE ARM IS HARD TO BEAT CLAIMABLY EVEN WHEN FAR "
        "BEHIND: a delta of 0.2279 would be overwhelming between two "
        "stable arms, and clears its threshold only 3.76-fold here "
        "entirely because the loser wobbles"
    ),
    "readings_committed": {
        "claimable": (
            "the 0.2520 arm is claimably ahead UNDER THIS PROTOCOL -- a "
            "statement about our protocol's outcome, NOT architectural "
            "superiority, since the loser does not span the label range. "
            "'Ahead of an arm that barely predicts' is the honest reading "
            "and must travel"
        ),
        "not_claimable": (
            "Phase 10 cannot claim even that, and closes on the "
            "comparison being unresolvable at this cohort's resolution -- "
            "a FOURTH independent arrival at COHORT_CANNOT_RESOLVE"
        ),
    },
    "two_passes": (
        "the 0.2520 arm's five vectors carry real hashes from "
        "p7d_paired.yaml (a path's contents are immutable, so one hash "
        "serves every config declaring it); the replication's five are "
        "PENDING_p10_cleftgnn with all-zero PLACEHOLDER hashes, because a "
        "run directory carries a SHA and a job id and neither is "
        "derivable on a laptop. Paste the run directory, then "
        "declare_inputs.py. Guard 3 refuses the run until both are real"
    ),
    "which_run": (
        "p10-cleftgnn-6, the citable one. Its training path is identical "
        "to p10-cleftgnn-5's, whose per-seed PCCs the prediction derives "
        "from -- the stage recording added between them rides a forward "
        "that already happened and consumes no RNG. If the declared "
        "vectors disagree, the prediction moves and the config does not, "
        "which is why the recorded figures are marked DESCRIPTIVE"
    ),
}


#: **[BUILT 2026-08-17, NO RUN ATTACHED] ALL EPOCHS' STAGE TABLES, NOT
#: ONLY THE SELECTED EPOCH'S.**
#:
#: The change ``U_SHAPE_BELONGS_LATER`` proposed and deferred nothing
#: about. ``seed_<n>__stage_ratios.csv`` gains an ``epoch`` column, a
#: ``selected`` flag and every epoch's rows; ``selected_epoch`` stays, so
#: the old reading is one filter away and nothing that consumed the file
#: loses its answer.
#:
#: **It costs nothing.** ``stage_ratios_by_epoch`` is already populated
#: for every epoch -- the capture rides the inner-val prediction the
#: harness already makes -- so the task was computing all of it and
#: writing one row of it. A longer CSV, no forward, no training, no run.
#:
#: **Why it belongs in the closing rather than the next phase.** This
#: phase discarded computed data twice and paid for it twice: the probe
#: that rebuilt the pipeline rather than reading it, and the 25 curves the
#: task threw away, which is what made the monitor question unanswerable
#: until they were rewritten. The U-shape is the third instance queued up
#: -- SABM at 0.7794 initially, 0.230-0.267 at typical selections, 0.436
#: at epoch 14 in the longest fold -- and it was reconstructed by hand
#: from a run that had already measured it. Making the change now means
#: the next run captures it for free instead of rediscovering the gap.
ALL_EPOCHS_RECORDED = {
    "built": "2026-08-17, no run attached",
    "change": (
        "seed_<n>__stage_ratios.csv gains an epoch column, a selected "
        "flag and every epoch's rows; selected_epoch stays, so the old "
        "reading is one filter away"
    ),
    "costs_nothing": (
        "stage_ratios_by_epoch is already populated for every epoch -- "
        "the capture rides the inner-val prediction the harness already "
        "makes -- so the task was computing all of it and writing one row "
        "of it"
    ),
    "why_now": (
        "this phase discarded computed data twice and paid twice: the "
        "probe that rebuilt the pipeline rather than reading it, and the "
        "25 curves the task threw away, which made the monitor question "
        "unanswerable until they were rewritten. The U-shape is the third "
        "instance queued up and was reconstructed by hand from a run that "
        "had already measured it"
    ),
}


#: **[CLOSING SENTENCE 2026-08-17] THE FAITHFUL ARM, IN ITS OWN WORDS --
#: not absorbed into the protocol arm's.**
#:
#: **The sentence**: under the group's own protocol -- rater-specific
#: models, an 85:15 stratified split, their Top-1 metrics, a single run
#: with no intervals -- CleftGNN on this cohort produced **constant
#: predictions for four of five raters**, with macro F1 **0.077-0.154**
#: and PCC undefined on those four. The fifth, ``Rater 11 -
#: Psychologist``, **broke free at epoch 33** and produced varying
#: predictions: F1 0.3352 study / 0.2366 benchmark, PCC **-0.1287**
#: study / **+0.0663** benchmark. **Nothing in this arm is claimable, by
#: construction, and that is what the arm was for.**
#:
#: **Why it gets its own sentence.** Absorbing it into the protocol arm's
#: closing would say "the replication scored near zero under both
#: protocols" -- true, and it would hide the two things this arm alone
#: establishes:
#:
#: 1. **The gap is not protocol.** The two arms differ ONLY in protocol,
#:    which is what the arm was built to isolate
#:    (``FAITHFUL_ARM_REGISTERED``). Running their protocol did not
#:    recover their numbers. So "we are at 0.252 while they report 0.598"
#:    is not answered by "they evaluated differently" -- we evaluated
#:    their way and it did not help.
#: 2. **The one cell that moved is the instability, not a result.** The
#:    psychologist cell's two PCCs have OPPOSITE SIGNS on the two test
#:    shapes of one model -- -0.1287 and +0.0663
#:    (``PSYCHOLOGIST_CELL_IS_THE_INSTABILITY``). That is the same
#:    signature as their own 0.598/0.283 pair on one rater-E model, and
#:    reading either as a capability is the error this project has
#:    documented on both sides of the comparison.
#:
#: **Its disposition**: recorded, not repaired, not ledgered, and not
#: quoted as a measurement of anything. A single run with no intervals is
#: not evidence under this project's criterion, and the arm was built
#: knowing that -- the point was the CONTRAST, and the contrast is what it
#: delivered.
FAITHFUL_ARM_CLOSING = {
    "closed": "2026-08-17",
    "sentence": (
        "under the group's own protocol -- rater-specific models, an "
        "85:15 stratified split, their Top-1 metrics, a single run with "
        "no intervals -- CleftGNN on this cohort produced CONSTANT "
        "predictions for four of five raters (macro F1 0.077-0.154, PCC "
        "undefined); the fifth, Rater 11 - Psychologist, broke free at "
        "epoch 33 with F1 0.3352 study / 0.2366 benchmark and PCC -0.1287 "
        "study / +0.0663 benchmark. Nothing here is claimable, by "
        "construction, and that is what the arm was for"
    ),
    "why_its_own_sentence": (
        "THE GAP IS NOT PROTOCOL: the two arms differ only in protocol, "
        "and running theirs did not recover their numbers -- so 'we are "
        "at 0.252 while they report 0.598' is not answered by 'they "
        "evaluated differently'",
        "THE ONE CELL THAT MOVED IS THE INSTABILITY, NOT A RESULT: the "
        "psychologist cell's two PCCs have OPPOSITE SIGNS on the two test "
        "shapes of one model, the same signature as their own 0.598/0.283 "
        "pair on one rater-E model",
    ),
    "disposition": (
        "recorded, not repaired, not ledgered, not quoted as a "
        "measurement of anything. A single run with no intervals is not "
        "evidence under this project's criterion, and the arm was built "
        "knowing that -- the point was the CONTRAST"
    ),
}


#: **[CLOSING 2026-08-17] PHASE 10 CLOSES.**
#:
#: The phase asked whether CleftGNN, replicated from the group's own
#: artifacts, reproduces their reported numbers under this project's
#: criterion. **It does not**, and the conclusion is unqualified
#: (``PHASE_10_CONCLUSION_UNQUALIFIED``) with a measured mechanism behind
#: it.
#:
#: **CRITERION 2'S ARITHMETIC, STATED RATHER THAN PASSED.** The exit
#: criterion anticipated **one** deviation from the published recipe. The
#: manuscript cell carries **FIVE** -- early stopping, Laplace smoothing
#: in the CE init, LayerNorm on the fused feature, the classifier weight
#: returning to the framework's own init, and LayerNorm on the GNN branch
#: -- and one of those five had its fidelity cost **corrected upward**
#: when the notebook turned out to state an epoch budget the manuscript
#: does not. The notebook cell carries **THREE more of its own**,
#: registered and **permanently unexercised**. A closing that reported
#: "the deviations were enumerated" and stopped would be true and
#: misleading; **eight registered deviations across two cells against a
#: criterion that anticipated one** is the honest form, and it is itself a
#: finding about how much the published description leaves unspecified.
#:
#: **CRITERION 1 IS THE PHASE'S LARGEST DELIVERABLE.** It began as a
#: single item -- the ROI count -- and closes as **seven
#: manuscript-vs-artifact discrepancies plus one blocking ambiguity**,
#: every one verified at source in the group's own documents:
#:
#:     1. region count      27 stated in three places, 36 executed
#:     2. optimiser         SGD 0.01 stated, Adam 0.001 executed
#:     3. fusion            additive eq (9) stated, f_t + f_t*v executed
#:     4. backbone          ResNet-50 (Table 1) stated, ViT-B/16 executed
#:     5. SABM              in the manuscript, absent from the notebook,
#:                          and a THIRD description (crop scoping) in the
#:                          deck -- weighting, deletion, and nothing
#:     6. the 90.94%        CIFAR-10, confirmed on the group's own slide
#:     7. epoch budget      the notebook states 5, the manuscript none
#:     8. the IEM direction ambiguous, and BLOCKING for Phase 11
#:
#: **Item 1 is the sharpest**: the group states 27 and their code runs 36,
#: printing the correction at every construction. **Every result in the
#: paper was produced with 36 regions; the reported architecture has never
#: been run** (``REGION_COUNT_BELIEF_VS_EXECUTION``).
#:
#: **THE FOUR CELLS AND WHAT EACH YIELDED:**
#:
#: * **protocol arm** -- five seeds, 25 folds, PCC +0.0241 (seed sd
#:   0.0676). Trains stably after three fixes; does not span the label
#:   range. Ledger entry 25 under its honest name.
#: * **faithful arm** -- their protocol, their metrics; four of five
#:   raters constant, the fifth an instability with opposite-signed PCCs
#:   (``FAITHFUL_ARM_CLOSING``). Not claimable by construction, and the
#:   gap is therefore **not protocol**.
#: * **rater screen** -- every rater-specific cell below 0.2520; the
#:   registered prior held.
#: * **notebook cell** -- refused entry at gate 3; a finding, not a number
#:   (``NOTEBOOK_CELL_CANNOT_START_CALIBRATED``), closed against reopening.
#:
#: **WHAT THE PHASE ESTABLISHED THAT NO SINGLE ARM SHOWS.** Four
#: independent routes -- the protocol arm, the faithful arm, the rater
#: screen, and the notebook cell's refusal -- and none of them recovers
#: the reported performance. The mechanism is measured rather than
#: asserted: the GNN branch carries image-dependence 0.0068 and the K=1
#: sigmoid kills it by 87x, so what varies comes from the SABM branch
#: alone, and that branch **degrades under training** (0.7794 at init to
#: 0.230-0.267 at the selected epochs).
#:
#: **STILL OPEN, and it is one run**: the paired BCa
#: (``PHASE_10_PAIRED_REGISTERED``), which fits nothing and whose two
#: readings are already committed.
PHASE_10_CLOSING = {
    "closed": "2026-08-17",
    "question": (
        "does CleftGNN, replicated from the group's own artifacts, "
        "reproduce their reported numbers under this project's criterion?"
    ),
    "answer": (
        "NO, unqualified (PHASE_10_CONCLUSION_UNQUALIFIED), with a "
        "measured mechanism behind it"
    ),
    "criterion_2_arithmetic": {
        "anticipated": 1,
        "manuscript_cell": 5,
        "notebook_cell": 3,
        "total": 8,
        "statement": (
            "EIGHT registered deviations across two cells against a "
            "criterion that anticipated ONE -- and one of the five had "
            "its fidelity cost CORRECTED UPWARD when the notebook turned "
            "out to state an epoch budget the manuscript does not. A "
            "closing that reported 'the deviations were enumerated' and "
            "stopped would be true and misleading"
        ),
        "is_itself_a_finding": (
            "eight-against-one measures how much the published "
            "description leaves unspecified"
        ),
        "notebook_cells_three": "registered and PERMANENTLY UNEXERCISED",
    },
    "criterion_1_discrepancies": (
        "region count: 27 stated in three places, 36 executed",
        "optimiser: SGD 0.01 stated, Adam 0.001 executed",
        "fusion: additive eq (9) stated, f_t + f_t*v executed",
        "backbone: ResNet-50 (Table 1) stated, ViT-B/16 executed",
        "SABM: in the manuscript, absent from the notebook, and a THIRD "
        "description (crop scoping) in the deck -- weighting, deletion "
        "and nothing",
        "the 90.94%: CIFAR-10, confirmed on the group's own slide",
        "epoch budget: the notebook states 5, the manuscript none",
        "the IEM direction: ambiguous, and BLOCKING for Phase 11",
    ),
    "sharpest": (
        "the region count -- the group states 27 and their code runs 36, "
        "printing the correction at every construction. EVERY result in "
        "the paper was produced with 36 regions; the reported "
        "architecture has never been run"
    ),
    "four_cells": {
        "protocol_arm": (
            "five seeds, 25 folds, PCC +0.0241 (seed sd 0.0676); trains "
            "stably after three fixes, does not span the label range; "
            "ledger entry 25 under its honest name"
        ),
        "faithful_arm": (
            "their protocol and metrics; four of five raters constant, "
            "the fifth an instability with opposite-signed PCCs. Not "
            "claimable by construction -- and the gap is therefore NOT "
            "protocol (FAITHFUL_ARM_CLOSING)"
        ),
        "rater_screen": (
            "every rater-specific cell below 0.2520; the registered prior "
            "held"
        ),
        "notebook_cell": (
            "refused entry at gate 3; a FINDING, not a number, closed "
            "against reopening"
        ),
    },
    "what_no_single_arm_shows": (
        "FOUR independent routes -- protocol arm, faithful arm, rater "
        "screen, and the notebook cell's refusal -- and none recovers the "
        "reported performance. The mechanism is measured: the GNN branch "
        "carries image-dependence 0.0068 and the K=1 sigmoid kills it by "
        "87x, so what varies comes from SABM alone, and SABM DEGRADES "
        "under training (0.7794 at init to 0.230-0.267 at selection)"
    ),
    # [RESOLVED 2026-08-17, same day] The run happened and the registered
    # not-claimable reading applied verbatim (PAIRED_BCA_OBSERVED).
    "still_open": (
        "NOTHING. The paired BCa ran: d +0.2279, condition 1 FALSE (3 of "
        "5 seeds), condition 2 TRUE at 3.76x -> WITHDRAWN, ledger entry "
        "26. PHASE_10_PAIRED_REGISTERED's not-claimable reading applied "
        "verbatim (PAIRED_BCA_OBSERVED)"
    ),
    # ---- the six exit criteria, walked at the close -------------------
    "exit_criteria": {
        "1_discrepancy": (
            "MET -- seven manuscript-vs-artifact discrepancies plus one "
            "blocking ambiguity, all verified at source in the group's "
            "own documents (criterion_1_discrepancies)"
        ),
        "2_deviations_enumerated": (
            "MET, with the arithmetic stated rather than passed: EIGHT "
            "across two cells against a criterion that anticipated ONE "
            "(criterion_2_arithmetic)"
        ),
        "3_five_seeds": (
            "MET -- 25 folds on the manuscript cell, with per-fold curves "
            "and every epoch's stage table now written rather than "
            "discarded (CURVES_AND_STAGES_WRITTEN, ALL_EPOCHS_RECORDED)"
        ),
        "4_paired_bca": (
            "MET -- and WITHDRAWN. d +0.2279, condition 1 FALSE, "
            "condition 2 TRUE at 3.76x; the registered reading applied "
            "verbatim (PAIRED_BCA_OBSERVED)"
        ),
        "5_ledgered": (
            "MET -- entry 25 banks the frozen ResNet-50 + SABM head under "
            "its own honest name, entry 26 the withdrawal with both "
            "conditions' figures and the cause attached. The born pin "
            "still verifies, so nothing earlier was rewritten"
        ),
        "6_suite": "MET -- green throughout",
    },
    # ---- what the closing rests on, all of it already recorded --------
    "cites_only_what_is_recorded": {
        "manuscript_cell": (
            "PCC +0.0242 (seed sd 0.0676) over 25 folds; mechanism "
            "measured, not asserted -- the K=1 sigmoid kills the GNN "
            "branch by 87x, so what varies comes from SABM alone, and "
            "SABM DEGRADES under training from 0.7794 at init to "
            "0.230-0.267 at the selected epochs "
            "(COLLAPSE_DIAGNOSED, CRITERION_I_FAILED_ON_THE_ARM)"
        ),
        "notebook_cell": (
            "refused entry at gate 3 fold 0; the epoch-0 prediction is a "
            "function of the SEED rather than the fold's labels, "
            "confirmed on real pretrained features where all five seeds "
            "fail at 0.94-1.67 SD and the zero-weight control returns the "
            "closed form on every one "
            "(NOTEBOOK_CELL_CANNOT_START_CALIBRATED). A finding, not a "
            "number, closed against reopening"
        ),
        "faithful_arm": (
            "its own sentence, not absorbed: four of five raters "
            "constant, the fifth an instability with opposite-signed "
            "PCCs -- so THE GAP IS NOT PROTOCOL (FAITHFUL_ARM_CLOSING)"
        ),
        "gate_3_coverage_limit": (
            "recorded as a standing finding and PROVED from gate 3's own "
            "arithmetic rather than from an instance: for a constant "
            "predictor the dispersion check's expected MSE is exact, so "
            "the ratio is 1 and it passes by identity, while the "
            "head-position check tests position alone (GATE_3_COVERAGE_"
            "LIMIT). Tolerances untouched"
        ),
        "ledger": (
            "entry 25 (p10-resnet50-sabm-head, DESCRIPTIVE) and entry 26 "
            "(p10-replication-vs-best-arm-withdrawn, WITHDRAWN)"
        ),
    },
    "no_new_claims": (
        "everything cited above was recorded when it happened; the "
        "closing adds nothing and asserts nothing that was not already "
        "measured and dated"
    ),
    # ---- open, non-blocking, carried forward BY NAME ------------------
    "carried_forward_open": (
        "the IEM direction question -- PHASE_11_BLOCKED_ON_IEM_DIRECTION; "
        "BLOCKING for Phase 11's implementation, unblocked by one "
        "sentence from supervision",
        "slide 9's cleft numbers -- SLIDE_9_CLEFT_NUMBERS_REQUESTED; the "
        "closest like-for-like comparator this project has been offered, "
        "and we do not have it",
        "the U-shape lead -- U_SHAPE_BELONGS_LATER; deferred to whichever "
        "phase next revisits this architecture, and now captured for free "
        "by ALL_EPOCHS_RECORDED",
        "the unanimity confirmation on the anchor grades -- carried "
        "from PHASE_9_CLOSING; the provenance caveat on every classifier "
        "quotation until then",
        "phase8.RETRY_LIMIT_IS_NOT_HOLDING -- the standing infrastructure "
        "item, with this phase's launches added to its evidence",
        "phase9.PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE -- to consider, "
        "deliberately not built",
    ),
    # [RESOLVED 2026-08-17] One of the six carried items is closed. The
    # list stays as written -- it is what the phase closed on -- and the
    # resolution is dated beside it.
    "carried_forward_resolved": (
        "2026-08-17: item 1, the IEM direction question, is ANSWERED -- "
        "convention A (phase11.IEM_DIRECTION_ANSWERED). It was the only "
        "one of the six that blocked anything, and nothing is blocked "
        "now. The other five stand"
    ),
    # [ANNEX OPENED 2026-08-30, the maintainer] The split-distribution
    # replication: label, metric, architecture and recipe fixed to
    # theirs, ONLY the split varied. Registration in phase10_annex
    # (pointers both ways); this closing is unchanged by it -- the
    # annex asks what Phase 10 never asked, it reopens nothing here.
    "annex_2026_08_30": "phase10_annex.ANNEX_QUESTION_REGISTERED",
}


#: **[OBSERVED 2026-08-17, ``p10_paired__27b895b6__p10-paired``] THE
#: PAIRED BCa: WITHDRAWN -- AND THE PRE-REGISTERED PREDICTION HELD
#: EXACTLY.**
#:
#: **The verdict.** ``p10__replication_vs_0p2520``: d **+0.2279**,
#: **3 of 5** seeds excluding zero, **condition 1 FALSE**, condition 2
#: TRUE at **3.76x** -> **WITHDRAWN**. Arms as declared: ``p10_cleftgnn``
#: 0.0242 (sd 0.0676), ``p7_d1_vit_b16_imagenet_g1`` 0.2520 (sd 0.0148).
#:
#: **THE PREDICTION HELD, AND THE ARITHMETIC IS WORTH STATING.**
#: ``PHASE_10_PAIRED_REGISTERED`` was written before the run and every
#: figure in it landed:
#:
#:     predicted delta      0.2279     observed d          +0.2279
#:     predicted threshold  0.0607     condition 2          3.76x
#:     predicted margin     3.76x      band                 mixed 3.6-4.7
#:     predicted decider    condition 1                     condition 1
#:     stated cause         4.6x seed sd    0.0676 vs 0.0148
#:
#: This is not a coincidence and it is not a success. The delta and the
#: threshold were DERIVED from figures already recorded, so predicting
#: them was arithmetic, not foresight. What was genuinely at risk was the
#: VERDICT -- 3.76x sits inside the band where 3.83x passed and 4.34x and
#: 4.69x failed, so the margin did not order it. **The registration said
#: condition 1 would decide, and condition 1 decided.**
#:
#: **One figure moved by 0.0001 and it is recorded rather than smoothed**:
#: the arm's declared pooled mean is 0.0242 where ``ROUTE_3_OBSERVED``
#: recorded 0.0241 from ``p10-cleftgnn-5``. The prediction derived from
#: 0.0241; the run's d is the PAIRED per-patient estimate, not the
#: difference of pooled means (0.2520 - 0.0242 = 0.2278), and the two
#: agree to the recorded precision. Nothing turns on it; a 0.0001 that
#: goes unmentioned is how a larger one later goes unnoticed.
#:
#: **THE REGISTERED NOT-CLAIMABLE READING, APPLIED VERBATIM**: Phase 10
#: cannot claim even that the 0.2520 arm is ahead, and closes on the
#: comparison being unresolvable at this cohort's resolution -- **a
#: FOURTH independent arrival at ``COHORT_CANNOT_RESOLVE``**.
#:
#: **What that sentence means, stated so it cannot be softened later.**
#: Under this project's criterion, the difference between the
#: best-scoring arm in the project and a replication scoring **0.024** is
#: **not claimable** -- because the replication's own instability sets
#: the threshold. An arm that wobbles by 0.0676 across seeds cannot be
#: claimably beaten, however far behind it is. That is a statement about
#: what this cohort can resolve, not about the two architectures.
PAIRED_BCA_OBSERVED = {
    "observed": "2026-08-17, p10_paired__27b895b6__p10-paired",
    "key": "p10__replication_vs_0p2520",
    "d": 0.2279,
    "seeds_excluding_zero": "3 of 5",
    "condition_1": False,
    "condition_2": True,
    "condition_2_margin": 3.76,
    "verdict": "WITHDRAWN",
    "arms_as_declared": {
        "p10_cleftgnn": {"mean": 0.0242, "sd": 0.0676},
        "p7_d1_vit_b16_imagenet_g1": {"mean": 0.2520, "sd": 0.0148},
    },
    "prediction_held": {
        "delta": (0.2279, 0.2279),
        "threshold_and_margin": (0.0607, 3.76),
        "band": "mixed 3.6-4.7, where 3.83x passed and 4.34x/4.69x failed",
        "decider": "condition 1, as registered",
        "cause": "the 4.6x seed sd -- 0.0676 against 0.0148",
        "honest_reading": (
            "NOT foresight: delta and threshold were DERIVED from figures "
            "already recorded, so predicting them was arithmetic. What was "
            "genuinely at risk was the VERDICT, because the margin sits "
            "inside the band that does not order it -- and the "
            "registration named condition 1 as the decider, which it was"
        ),
    },
    "the_0_0001": (
        "the arm's declared pooled mean is 0.0242 where ROUTE_3_OBSERVED "
        "recorded 0.0241 from p10-cleftgnn-5. The prediction derived from "
        "0.0241; the run's d is the PAIRED per-patient estimate rather "
        "than the difference of pooled means (0.2278), and the two agree "
        "to the recorded precision. Nothing turns on it -- a 0.0001 that "
        "goes unmentioned is how a larger one later goes unnoticed"
    ),
    "registered_reading_applied": (
        "NOT CLAIMABLE, verbatim: Phase 10 cannot claim even that the "
        "0.2520 arm is ahead, and closes on the comparison being "
        "unresolvable at this cohort's resolution -- a FOURTH independent "
        "arrival at COHORT_CANNOT_RESOLVE"
    ),
    "what_that_means": (
        "under this project's criterion the difference between the "
        "best-scoring arm in the project and a replication scoring 0.024 "
        "is NOT CLAIMABLE, because the replication's own instability sets "
        "the threshold. An arm that wobbles by 0.0676 across seeds cannot "
        "be claimably beaten however far behind it is. A statement about "
        "what this cohort can resolve, not about the two architectures"
    ),
}


def top1_macro_prf(truth, predicted, n_classes: int = 5) -> dict:
    """Macro precision / recall / F1 over the grade classes -- their
    metrics (``FAITHFUL_ARM_REGISTERED``), computed here because
    ``eval/metrics.py`` is FROZEN and carries none of them.

    A class with no predictions has precision 0 by convention and a class
    with no truths is EXCLUDED from the macro average -- averaging over a
    class that cannot occur would report the model's silence as a score.
    Both choices are stated because both change the number.
    """
    import numpy as np

    truth = np.asarray(truth, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    precisions, recalls, f1s, present = [], [], [], []
    for grade in range(1, n_classes + 1):
        true_positive = int(np.sum((predicted == grade) & (truth == grade)))
        predicted_positive = int(np.sum(predicted == grade))
        actual_positive = int(np.sum(truth == grade))
        if not actual_positive:
            continue
        present.append(grade)
        precision = (
            true_positive / predicted_positive if predicted_positive else 0.0
        )
        recall = true_positive / actual_positive
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(
            0.0 if precision + recall == 0
            else 2 * precision * recall / (precision + recall)
        )
    return {
        "precision_macro": float(np.mean(precisions)) if precisions else 0.0,
        "recall_macro": float(np.mean(recalls)) if recalls else 0.0,
        "f1_macro": float(np.mean(f1s)) if f1s else 0.0,
        "accuracy": float(np.mean(truth == predicted)) if len(truth) else 0.0,
        "classes_present": tuple(present),
        "classes_absent_excluded": tuple(
            g for g in range(1, n_classes + 1) if g not in present
        ),
    }



def summary() -> dict:
    """The phase's records, importable as one object."""
    return {
        "comparator": CLEFTGNN_COMPARATOR_TABLES,
        "roi_check": ROI_CHECK_MEASURED,
        "notebook_roi": NOTEBOOK_ROI_SET_MEASURED,
        "iem_for_phase_11": IEM_CARRIED_FOR_PHASE_11,
        "scope": PHASE_10_SCOPE_AND_SILENCES,
        "registered": PHASE_10_REGISTERED,
        "blocked": PHASE_10_BUILD_BLOCKED_ON,
        "unblocked": PHASE_10_UNBLOCKED,
        "built": CLEFTGNN_BUILT,
        "label": CONSENSUS_LABEL_IS_THE_MEDIAN,
        "first_launch_void": CLEFTGNN_FIRST_LAUNCH_VOID,
        "nan_measured": CLEFTGNN_NAN_MEASURED,
        "second_launch_void": CLEFTGNN_SECOND_LAUNCH_VOID,
        "frozen": BACKBONE_IS_FROZEN,
        "smoothing": THIN_CLASS_SMOOTHING,
        "deviations": REGISTERED_DEVIATIONS,
        "faithful_arm": FAITHFUL_ARM_REGISTERED,
        "rater_screen": RATER_SCREEN_REGISTERED,
        "rater_ladder": RATER_LADDER_CONDITIONAL,
        "screen_observed": RATER_SCREEN_OBSERVED,
        "faithful_observed": FAITHFUL_ARM_OBSERVED,
        "scale_measured": CLEFTGNN_SCALE_MEASURED,
        "pod_probe": P10_SCALE_PROBE,
        "third_launch_void": CLEFTGNN_THIRD_LAUNCH_VOID,
        "shared_mechanism": ARMS_SHARE_THE_MECHANISM,
        "sd_zero": SD_ZERO_HAS_TWO_CAUSES,
        "fix": FUSED_NORM_AND_STANDARD_INIT,
        "arms_collapsed": ARMS_COMPLETE_BUT_COLLAPSED,
        "ledger_hold": LEDGER_HOLD_REASONS,
        "collapse_diagnosed": COLLAPSE_DIAGNOSED,
        "routes": COLLAPSE_ROUTES_PROPOSED,
        "stage_table_confirmed": STAGE_TABLE_CONFIRMED,
        "gnn_branch_norm": GNN_BRANCH_NORM,
        "normalised_arm_readings": NORMALISED_ARM_READINGS,
        "probe_defect": PROBE_RECONSTRUCTED_THE_PIPELINE,
        "expected_table": EXPECTED_POST_FIX_TABLE,
        "route_3_observed": ROUTE_3_OBSERVED,
        "third_reading": THIRD_READING_PROPOSED,
        "psychologist_cell": PSYCHOLOGIST_CELL_IS_THE_INSTABILITY,
        "seed_diagnostic": SEED_INSTABILITY_NEXT_DIAGNOSTIC,
        "epochs_answer_neither": EPOCHS_ANSWER_NEITHER_READING,
        "curve_writing": CURVE_WRITING_PROPOSED,
        "criterion_i_open": CRITERION_I_UNREPORTED,
        "curves_and_stages": CURVES_AND_STAGES_WRITTEN,
        "criterion_i_failed": CRITERION_I_FAILED_ON_THE_ARM,
        "monitor": MONITOR_SELECTS_AGAINST_PCC,
        "synthesis": THE_TWO_FINDINGS_MAY_BE_ONE,
        "monitor_arm": MONITOR_ARM_REGISTRABILITY,
        "monitor_exonerated": MONITOR_EXONERATED,
        "conclusion": PHASE_10_CONCLUSION_UNQUALIFIED,
        "remains": WHAT_REMAINS_BEFORE_CLOSING,
        "u_shape": U_SHAPE_BELONGS_LATER,
        "cifar_confirmed": CIFAR_CONFLATION_CONFIRMED_AT_SOURCE,
        "notebook_recipe": NOTEBOOK_RECIPE_IS_ADAM,
        "notebook_fusion": NOTEBOOK_FUSION_IS_MULTIPLICATIVE,
        "notebook_backbone": NOTEBOOK_BACKBONE_AND_ACM_CONFIRMED,
        "three_descriptions": THREE_DESCRIPTIONS_OF_THE_ATTENTION,
        "iem_direction": IEM_IS_THEIRS_DIRECTION_AMBIGUOUS,
        "notebook_cell_proposed": NOTEBOOK_RECIPE_CELL_PROPOSED,
        "notebook_cell": NOTEBOOK_RECIPE_CELL_REGISTERED,
        "notebook_cell_deviations": NOTEBOOK_CELL_DEVIATIONS,
        "region_count_belief": REGION_COUNT_BELIEF_VS_EXECUTION,
        "iem_blocks_phase_11": PHASE_11_BLOCKED_ON_IEM_DIRECTION,
        "slide_9_request": SLIDE_9_CLEFT_NUMBERS_REQUESTED,
        "notebook_budget": NOTEBOOK_BUDGET_AND_NORMALISATION,
        "gate_3_failed": GATE_3_FAILED_ON_THE_NOTEBOOK_CELL,
        "gate_3_routes": GATE_3_ROUTES_PROPOSED,
        "notebook_cell_finding": NOTEBOOK_CELL_CANNOT_START_CALIBRATED,
        "notebook_cell_closed": NOTEBOOK_CELL_DISPOSITION_CLOSED,
        "gate_3_coverage": GATE_3_COVERAGE_LIMIT,
        "exit_walk": PHASE_10_EXIT_WALK,
        "paired_coverage": PAIRED_CLAIM_COVERAGE,
        "paired": PHASE_10_PAIRED_REGISTERED,
        "paired_observed": PAIRED_BCA_OBSERVED,
        "all_epochs": ALL_EPOCHS_RECORDED,
        "faithful_closing": FAITHFUL_ARM_CLOSING,
        "closing": PHASE_10_CLOSING,
    }
