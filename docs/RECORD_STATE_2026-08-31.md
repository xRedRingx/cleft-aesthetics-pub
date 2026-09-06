# Record state — verified 2026-08-31

**Generated from the repo, not written from memory.** Every figure below was
read from its record at the commit named here, or computed from it. Nothing is
carried over from a summary, a handoff, or a conversation. Where a fact that
was expected to exist does not exist in the repo, this document says
**NOT IN REPO** rather than filling the gap.

| | |
|---|---|
| commit | `073bed28bcea5d8ff439daea48ab8fc170281b07` (2026-08-31, "p10x annex: close — reading fired, pathologies measured, per-rater table, no ledger row") |
| working tree | clean at generation time |
| suite | **2722 passed, 17 skipped** (`.\scripts\test.ps1`, pinned venv Python 3.11.0) |
| ledger | 37 entries, `validate()` returns `None` (clean), all 10 checksum pins recompute *(→ **38 entries / 11 pins** as of 2026-08-31, §2.3.1 — the state below is as generated)* |
| modules indexed | 36 `.py` under `src/cleft`, 85 test files, 316 configs |

**How to regenerate.** Re-run `.\scripts\test.ps1`; re-read the records named
below by their exact symbol names. The document is a view over the records, not
a second copy of them — where this document and a record disagree, **the record
wins** (precedence: code > record > this document > memory).

**Four things this verification found that the request's own framing did not
have right.** They are flagged in place and collected here because a future
session will otherwise inherit them:

1. The phase numbering drifted **four** times by amendment, not twice — plus a
   fifth, earlier shift in `docs/PLAN.md` before any amendment record existed.
2. The ledger contains **eight** condition-2-pass / condition-1-fail rows, not
   two. Two of them carry caveats claiming to be the first and second such
   cases; four earlier rows already matched. See §2.3. **[RESOLVED
   2026-08-31 — §2.3.1: both claims ruled wrong as written and corrected by
   `results_ledger.ENTRIES[37]`, the ledger's first use of `corrects`. The
   ledger now holds 38 entries.]**
3. The "detection floor ΔPCC ≈ 0.12–0.14" is **NOT IN REPO** in that framing.
   What is recorded is different. See §4.3.
4. The ledger chain is pinned at **ten** depths (17/18/19/20/21/22/24/25/32/37),
   not six.

---

## 1. The authoritative phase map

### 1.1 What each number is NOW

| # | Subject | Closing record | Status |
|---|---|---|---|
| 0 | Foundation: provenance contract, guards, pinned image | **NOT IN REPO** as a record — `docs/PLAN.md` L1084 prose only | closed (prose) |
| 1 | Data: manifest, labels, reliability, folds | **NOT IN REPO** — no closing record, no `PHASE_1_*` symbol | closed by implication |
| 2 | Geometry and patches: staging, G1/G2, generators | **NOT IN REPO** (nearest: `roadb.PHASE_2_REQUIREMENTS`, a requirements record) | no closing record |
| 3 | Gates and freeze (six gates) | **NOT IN REPO** as a record — `docs/PLAN.md` L1101 prose, tag `phase3-freeze` | closed (prose) |
| 4 | Baselines: mirror-difference symmetry, SymNose, patch probes | **NOT IN REPO** | no closing record |
| 5 | SCUT preparation: masked SCUT at G1/G2 | **NOT IN REPO** as a record — `docs/PLAN.md` L1567 prose | closed (prose) |
| 6 | Pretraining (12 cells / 24 embedding sets) | **NOT IN REPO** as a closing. Structure/results: `roadb.PHASE_6_STRUCTURE`, `roadb.PHASE_6_TWELVE_CELLS`, `roadb.PHASE_6_RESOLUTION_RESULTS`; later reframed by `phase15.PHASE_6_REFRAMED` | no closing; one cell reframed |
| 7 | The ladder (Road A) / resolution+arm road (Road B) | No whole-phase-7 closing on Road A. `ladder.PHASE_7D_CLOSING` (sub-phase). Road B: `roadb.PHASE_7_CLOSING` (the decision) and `roadb.ROAD_B_PHASE_7_CLOSING` (the statement). **7B and 7C have no `_CLOSING` record at all** | closed piecewise |
| 8 | Explainability: Grad-CAM, node attention, t-SNE, clinical delivery | `phase8.PHASE_8_CLOSING` (partial) then `phase8.PHASE_8_COMPLETE`. Sub-closings `phase8.PHASE_8B_CLOSING`, `phase8c.PHASE_8C_CLOSING`, `phase8.TSNE_CLOSING`, `phase8.SCUT_ANIMATION_CLOSING` | closed on **amended scope** |
| 9 | Prototypes, results ledger, external 25-image reference | `phase9.PHASE_9_CLOSING` | closed 2026-08-16 |
| 10 | CleftGNN replication under this project's criterion | `phase10.PHASE_10_CLOSING` | closed — answer "NO, unqualified" |
| 11 | The asymmetric loss (IEM / "penalise optimism") | `phase11.PHASE_11_CLOSING` | closed |
| 12 | **View ablation** (frontal / basal / both) | `phase12.PHASE_12_CLOSING` | closed — "descriptively yes, claimably no" |
| 13 | Decoder / reconstruction as a measurement instrument | `phase13.PHASE_13_CLOSING` | closed 2026-08-24 |
| 14 | Label Distribution Learning (soft_1..soft_5) | `phase14.PHASE_14_CONCEDED_COVERED` | **conceded** — nothing built |
| 15 | Second beauty dataset (MEBeauty) — Road B Branch 2 unparked | `phase15.PHASE_15_CLOSING` | closed 2026-08-24 |
| 16 | **Anchor loop** (25-image design, promoted) | `phase16.PHASE_16_CLOSING` | closed 2026-08-29 |
| 17 | TSTR — train on synthetic, test on real | `phase17.PHASE_17_CLOSING` | closed 2026-08-30 |
| 18 | Metric-space ablation | `phase18.PHASE_18_CLOSING` | closed 2026-08-30 |
| 19 | Write-up | **NOT IN REPO** — no `phase19.py`, no record | not opened |
| — | **Phase 10 annex** — split-distribution replication | `phase10_annex.PHASE_10_ANNEX_CLOSING` | closed 2026-08-31; takes no number |

### 1.2 The numbering drift — where a number means something different

**This is the trap most likely to mislead a future session.** Five shifts, in
order:

**(0) Before any amendment record.** `docs/PLAN.md` Part 5 ends at
"**Phase 10 — Write-up**". `PLAN_AMENDMENT_2026-08-13` (the file itself is at
`PLAN_AMENDMENT_2026-08-13.md`, **NOT IN REPO**; cited in `phase9.py:3`,
`phase10.py:3`, `phase11.py:3`) replaced that with 9 = prototypes,
10 = CleftGNN replication, 11 = asymmetric loss, 12 = decoder, 13 = write-up.
**So PLAN.md's "Phase 10" is the write-up; the repo's Phase 10 is the
replication.** PLAN.md is the older document and is wrong about this.

**(1) `phase11.PHASE_SEQUENCE_RENUMBERED`** (2026-08-23) — was
`{12: decoder, 13: write-up}` → becomes `{12: VIEW ABLATION, 13: decoder,
14: write-up}`.

**(2) `phase12.PHASE_SEQUENCE_RENUMBERED_2`** (2026-08-23, same day) — was
`{13: decoder, 14: write-up}` → becomes `13 decoder, 14 LDL, 15 second beauty
dataset, 16 TSTR, 17 write-up`.

**(3) `phase15.PHASE_SEQUENCE_RENUMBERED_3`** (2026-08-24) — was
`{16: TSTR, 17: write-up}` → becomes `16 metric-space ablation, 17 TSTR,
18 write-up`.

**(4) `phase15.PHASE_SEQUENCE_RENUMBERED_4`** (2026-08-24, later same day) —
was `{16: metric-space ablation, 17: TSTR, 18: write-up}` → becomes
`16 ANCHOR LOOP, 17 TSTR, 18 metric-space ablation, 19 write-up`.

Note `_4` is defined **above** `_3` in `phase15.py` (line 2424 vs 2494):
reading order does not match amendment order.

### 1.3 Keys whose name carries the wrong number

| Key | Home | Name says | Actually is | Dated pointer? |
|---|---|---|---|---|
| `PHASE_16_SCHEDULED` | `phase15.py` | 16 | **Phase 18** (metric-space ablation) | yes — `renumbered_to_18` |
| `ANCHOR_LOOP_REGISTERED` | `phase15.py` | (hosted in 15) | **Phase 16's registration** | yes — `promoted` |
| `PHASE_16_VERDICT_PROPOSED` | `phase15.py` | 16 | the verdict consumed by **TSTR = 17** | **NO pointer** — unpatched drifted name |
| `PHASE_11_CLOSING["next"]` | `phase11.py` | "Phase 12, the decoder" | decoder is **13** | yes |
| `PHASE_14_CONCEDED_COVERED["sequence_advances"]` | `phase14.py` | "Phase 16 consumes the best" | that is **17** | yes (inline) |
| `classification.THREE_NOT_FIVE` | `classification.py` | comment "Phase 16" | means **18** | yes |

Not drift, just cross-module homes: `phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION`
(genuinely Phase 11), `phase15.PHASE_6_REFRAMED` (genuinely Phase 6).

**Name collision to avoid:** `phase15.py`'s `"branch_2_has_no_instance"` is a
bounds-check reading branch, **not** Road B Branch 2.

### 1.4 Road B

**Framing:** `phase9.ROAD_B_IS_THE_ANNEX` (2026-08-16, the maintainer) — "Road A is
the main project; Road B is the annex of additional and alternative ideas".
The convergence deliverable is `"DROPPED -- no fork, no convergence"`. Every
Road B measurement stands unchanged; only the framing changed. `roadb_`
prefixes stay in paths. The superseded prose survives as history at
`roadb.py:3` ("Road A is not superseded. It is the control arm.") with the
dated reframing note beneath it.

| Branch | Subject | Record | Status |
|---|---|---|---|
| 1 | recover the 10.8× discarded pixels via resolution 224/512/768 | `roadb.BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER` (no `_CLOSING`) | **refuted**, not closed |
| 2 | second beauty dataset | no `ROAD_B_BRANCH_2` record; parked at `phase8.FLAGGED_FOR_LATER`, unparked 2026-08-23 as Phase 15 | closed **as Phase 15** |
| 3 | anatomy region-crop input | `roadb.ROAD_B_BRANCH_3_CLOSING`; module `roadb_regioncrop.py` | closed, **nothing claimable** |

Road B Phase 7 (its own sequence, distinct from the branches):
`roadb.PHASE_7_CLOSING` is the run-and-report **decision**;
`roadb.ROAD_B_PHASE_7_CLOSING` is the closing **statement** — the record itself
states the distinction. "22 of 24 arms run, 20 contrasts tested, 3 survive."

---

## 2. The ledger — all 37 rows

`validate()` clean. Statuses: `CLAIMABLE`, `WITHDRAWN`, `UNRESOLVED-WITHDRAWN`,
`VOID`, `DESCRIPTIVE`. Framings: `main`, `additional`.

### 2.1 The rows

| # | id | phase | status | statistic / arms |
|---|---|---|---|---|
| 0 | `p7-best-arm-descriptive` | p7 | DESCRIPTIVE | `p7_d1_vit_b16_imagenet_g1` PCC 0.2520 (sd 0.0148, 5 seeds) |
| 1 | `p7-best-arm-vs-challenger` | p7 | WITHDRAWN | vs swin_b scut_original g2 0.2092; δ 0.0428, 1.05×, 0/5 |
| 2 | `p7-q1-swin-g2` | p7 | **CLAIMABLE** | SCUT-original > ImageNet, swin_b @G2: +0.2027, 3.83×, 5/5 |
| 3 | `p7-cohort-cannot-resolve` | p7 | DESCRIPTIVE | 29 of 30 paired comparisons fail condition 1 |
| 4 | `p7b-claimably-worse` | p7b | WITHDRAWN | δ −0.0542, 2.37×, 1/5 |
| 5 | `p7c-nine-verdicts` | p7c | WITHDRAWN | nine augmentation verdicts to −0.1005, 0/45 |
| 6 | `p7d-grid-density` | p7d | **CLAIMABLE** | patch8 vs b16 −0.2089 (6.42×, 5/5); b32@512 vs patch16@512 +0.1386 (5.32×, 5/5) |
| 7 | `p7d-null-and-concat` | p7d | WITHDRAWN | patch32 −0.0001, concat +0.0073, 0/5 |
| 8 | `p7d-mvitv2` | p7d | UNRESOLVED-WITHDRAWN | margins 1.77×, 1.89× |
| 9 | `p8-parameter-dependence` | p8 | **CLAIMABLE** | Grad-CAM separability −0.1061, CI [−0.2047, −0.0020], P = 0.0037 |
| 10 | `p8-framing-not-anatomy` | p8 | WITHDRAWN | **refuted** 2026-08-08 |
| 11 | `p8-ranking-contrast` | p8 | DESCRIPTIVE | Grad-CAM ranks (0.659); gated graph attention never ranked |
| 12 | `void-first-ladder` | p7 | VOID | the pre-restart ladder |
| 13 | `void-scut-animation-first-launch` | p8 | VOID | SHA e40f09de, 6–7 attempts on pre-fix code |
| 14 | `roadb-resolution-fall` | roadb-p7 | **CLAIMABLE** | ViT falls from 224: −0.1848 (8.57×), −0.1626 (7.38×), −0.1583 (6.91×), all 5/5 |
| 15 | `roadb-resolution-withdrawn` | roadb-p7 | WITHDRAWN | the other 17 of 20 pairs |
| 16 | `roadb-region-crop` | roadb-branch3 | WITHDRAWN | first five-arm set VOID on five defects; corrected set nothing claimable |
| 17 | `p9-prototypes-descriptive` | p9 | DESCRIPTIVE | medoids 183/147/148; nearest-medoid 0.257 < chance 0.333 |
| 18 | `void-deall-first-launch` | p9 | VOID | labels CSV at bch root, not in the images folder |
| 19 | `p9-anchor-classifier-convergence` | p9 | DESCRIPTIVE | 3-class acc 0.333–0.363 across 8 cells; anchors self-consistency 4/25, 3/25 |
| 20 | `void-cleftgnn-first-launch` | p10 | VOID | label reader sought a "consensus" header that does not exist |
| 21 | `void-cleftgnn-second-launch` | p10 | VOID | first-step divergence; log(0) hypothesis refuted |
| 22 | `p10-rater-screen-mixed` | p10 | DESCRIPTIVE | per-rater −0.0098 / 0.1606 / 0.2150 / 0.2703 / 0.1362 vs 0.2520 |
| 23 | `void-cleftgnn-third-launch` | p10 | VOID | non-finite under frozen backbone; backbone-gradient diagnosis refuted |
| 24 | `p10-resnet50-sabm-head` | p10 | DESCRIPTIVE | frozen ResNet-50 + SABM head: PCC 0.0241, seed sd 0.0676 |
| 25 | `p10-replication-vs-best-arm-withdrawn` | p10 | WITHDRAWN | d +0.2279; **c1 FALSE (3/5), c2 TRUE 3.76×** |
| 26 | `p11-iem-loss-costs-pcc-withdrawn` | p11 | WITHDRAWN | IEM-loss 0.1855 vs control 0.2552; **c1 FALSE (3/5), c2 TRUE 2.57×** |
| 27 | `p11-iem-loss-improves-iem-withdrawn` | p11 | WITHDRAWN | IEM 0.5825 vs 0.6130; **c1 FALSE (2/5), c2 TRUE 2.68×** |
| 28 | `p12-basal-vs-frontal-bar-withdrawn` | p12 | WITHDRAWN | A 0.2505 vs B 0.1880; **c1 FALSE (0/5), c2 TRUE 1.8×** |
| 29 | `p12-concat-vs-frontal-bar-unresolved` | p12 | UNRESOLVED-WITHDRAWN | C 0.2640 vs A 0.2505; c1 0/5, c2 0.47× |
| 30 | `p12-concat-vs-capacity-unresolved` | p12 | UNRESOLVED-WITHDRAWN | C vs D 0.2534; c1 0/5, c2 0.36× |
| 31 | `p16-anchor-loop-unresolved` | p16 | UNRESOLVED-WITHDRAWN | loop 0.2040 vs probe 0.2520; **c1 FALSE (0/5, mixed), c2 TRUE 1.46×** |
| 32 | `p17-a-vs-probe` | p17 | UNRESOLVED-WITHDRAWN | A 0.2334 (sd 0.0044) vs 0.2520; **c1 FALSE (0/5), c2 TRUE 1.37×** |
| 33 | `p17-b-vs-probe` | p17 | **CLAIMABLE (negative)** | B −0.0044 vs 0.2520; δ −0.2558, 5/5, 8.3× |
| 34 | `p17-c-vs-probe` | p17 | UNRESOLVED-WITHDRAWN | C −0.0185; **c1 FALSE (4/5), c2 TRUE 5.1×** |
| 35 | `p17-a-vs-c` | p17 | UNRESOLVED-WITHDRAWN | δ +0.2518; **c1 FALSE (4/5), c2 TRUE 4.9×** |
| 36 | `p17-b-vs-c` | p17 | UNRESOLVED-WITHDRAWN | δ +0.0147; c1 2/5 mixed, c2 FALSE — the only row failing both |

**Totals:** ~~5 CLAIMABLE (one of them negative), 10 WITHDRAWN,
8 UNRESOLVED-WITHDRAWN, 6 VOID, 6 DESCRIPTIVE, plus 2 rows whose conditions are
structured dicts rather than verdict strings.~~

> **[CORRECTED 2026-09-05 — wrong at this document's own commit, not stale.]** The struck
> line is preserved. Over the 37 rows this section describes,
> `collections.Counter(e["status"] for e in results_ledger.ENTRIES[:37])` gives
> **5 CLAIMABLE, 11 WITHDRAWN, 8 UNRESOLVED-WITHDRAWN, 6 VOID, 7 DESCRIPTIVE = 37**. The
> struck totals sum to 35 and then add "plus 2 rows" to reach 37 — but those two rows already
> carry a status and are already counted, so the clause double-counts them. Structured-dict
> conditions are a property of a row, not a sixth status.

### 2.2 Chain pins — all verified

Pinned in `tests/test_phase9.py` at depths **17, 18, 19, 20, 21, 22, 24, 25,
32, 37** — ten pins, not the six commonly quoted. Every one recomputes against
the live ledger:

```
BORN_CHECKSUM  n=17  c46fd950…   CHECKSUM_22  n=22  08fe5055…
CHECKSUM_18    n=18  b476ca7f…   CHECKSUM_24  n=24  a1897e84…
CHECKSUM_19    n=19  eb40d371…   CHECKSUM_25  n=25  e5137587…
CHECKSUM_20    n=20  8fdc8f71…   CHECKSUM_32  n=32  028e4189…
CHECKSUM_21    n=21  88e652af…   CHECKSUM_37  n=37  889fb62f…
```

The final pin covers all 37 entries.

*[2026-08-31] An eleventh pin, `CHECKSUM_38`
(`f01228a7…`), was added when the correcting entry appended (§2.3.1). All ten
above are unchanged and still recompute — which is the evidence that the
corrected rows were not edited.*

### 2.3 ⚠ DISCREPANCY — the condition-2-pass / condition-1-fail count

**The record is internally inconsistent here, and it should be ruled on rather
than quoted as-is.**

`p16-anchor-loop-unresolved` carries: *"the first case in this ledger of
condition 2 passing while condition 1 fails"*, and `p17-a-vs-probe` carries:
*"the SECOND condition-2-pass/condition-1-fail instance in this ledger (after
p16-anchor-loop-unresolved)"*.

Computed over the ledger's own `condition_1` / `condition_2` fields, **eight**
rows have condition 1 FALSE and condition 2 TRUE, and **four of them were
appended before p16**, by both index and date:

| index | id | date | margin |
|---|---|---|---|
| 25 | `p10-replication-vs-best-arm-withdrawn` | 2026-08-17 | 3.76× |
| 26 | `p11-iem-loss-costs-pcc-withdrawn` | 2026-08-17 | 2.57× |
| 27 | `p11-iem-loss-improves-iem-withdrawn` | 2026-08-17 | 2.68× |
| 28 | `p12-basal-vs-frontal-bar-withdrawn` | 2026-08-23 | 1.8× |
| 31 | `p16-anchor-loop-unresolved` | 2026-08-29 | 1.46× |
| 32 | `p17-a-vs-probe` | 2026-08-30 | 1.37× |
| 34 | `p17-c-vs-probe` | 2026-08-30 | 5.1× |
| 35 | `p17-a-vs-c` | 2026-08-30 | 4.9× |

The narrowest defensible reading of p16's sentence is its own qualifier — *all
five per-seed intervals spanning zero **with directions mixed*** — which p12's
0/5 row may not satisfy. But that reading does not rescue `p17-a-vs-probe`,
whose condition 1 reads "0 of 5" with no mixed-direction qualifier.

**Recommended handling:** treat the eight-row set as the fact (it is computed
from the fields), and treat the two "first/second" sentences as prose needing a
dated correction. Nothing downstream depends on the count — no verdict changes
either way. **Not corrected here: this document changes nothing.**

### 2.3.1 RESOLVED — 2026-08-31

The discrepancy raised above was verified in full and then corrected. **the maintainer ruled the descriptive reading** of p16's em-dash clause: the bolded head
sentence is the claim, and the appositive describes that instance rather than
qualifying it. The grounds recorded: p17 restates the property with the qualifier
dropped, which is how the author read it a day later; and a record that needs
an appositive to be load-bearing will mislead a reader who does not parse it
that way. **Under that reading both claims are wrong as written.**

- **p16-anchor-loop-unresolved** is the **fifth**, not the first.
- **p17-a-vs-probe** is the **sixth**, not the second — and it fails the
  narrower reading too: its condition-1 field records "0 of 5" with no
  direction statement, and no phase-17 record states its sign pattern, so its
  membership in the narrow kind is *unestablished*, not merely unclaimed.
- **What was true is preserved**: p16 *is* verifiably the first row with all
  per-seed intervals spanning zero **and** directions mixed. ~~Index 28
  (p12-basal) is the only earlier all-span-zero row~~, and its direction is
  single — `phase12.PAIRED_OBSERVED["verdicts"]["b_vs_a"]` records "A ahead",
  re-derived from per-seed means +0.0507 / +0.0321 / +0.0326 / +0.1121 /
  +0.0849, all one sign.

> **[CORRECTED 2026-09-05 — wrong at this document's own commit, and wrong in the same way as
> the error this section exists to correct.]** Index 28 is **not** the only earlier
> all-span-zero row: indices **29 and 30** also record 0 of 5 intervals excluding zero, and
> **this document's own ledger table twelve lines above lists all three consecutively**. So the
> claim is refutable from the page it is written on.
>
> **The mechanism is the point.** This section corrects `ledger-condition-split-count-corrected`
> (entry 38), whose whole lesson is that *a precedence claim — "the first", "the only", "the
> Nth" — is derivable from the rows already present, and asserting one from memory while
> writing is putting a claim next to the data that refutes it*. The sentence correcting that
> error committed it again, one paragraph later.
>
> **What survives is the narrower fact, and it is unaffected**: index 28's direction is single
> ("A ahead", the five per-seed means all one sign), so p16 remains the first row with all
> intervals spanning zero **and** directions mixed. Only the "only earlier" clause is withdrawn.

**The correction is `results_ledger.ENTRIES[37]`,
`ledger-condition-split-count-corrected`** — the ledger's **first-ever use of
the `corrects` field**, which had existed unused since the born population of
17. The corrected rows are append-only and untouched: `CHECKSUM_37` and all
nine earlier pins still hold, and `CHECKSUM_38` is pinned beside them. **No
measurement changed** — every delta, threshold, margin and verdict stands as
computed; two sentences about precedence were wrong.

The generalisable lesson, recorded in the entry: the error was **a claim about
the ledger's own history written into a row at append time, when the ledger
itself could have been queried**.

---

## 3. The arm table

**Authoritative inventory:** `phase18.ARM_LIST_LOCKED` (locked 2026-08-30) —
**68 arms**, flattened by `phase18.locked_arm_entries()`; each entry is
`(arm_name, run_directory)`.

**Seed regime:** `ladder.SEEDS_BY_KIND = {"transformer": 5, "graph": 10}`.
Materialised as `phase18.SEEDS_5 = (1337, 2024, 7, 99, 12345)` and
`SEEDS_10 = SEEDS_5 + (42, 271828, 314159, 161803, 777)`.

### 3.1 Road A ladder — transformers (5 seeds, n=237)

| arm | PCC (sd) | run dir suffix |
|---|---|---|
| `p7_c0_vit_b16_scut_original_g1` | 0.1952 (0.0280) | `__b954df1e__p7-c0-vit-original-g1` |
| `p7_c_vit_b16_scut_masked_g1` | 0.0830 (0.0247) | `__4cb62c05__p7-c-vit-g1` |
| `p7_c_swin_b_scut_masked_g1` | 0.1327 (0.0267) | `__4cb62c05__p7-c-swin-g1` |
| **`p7_d1_vit_b16_imagenet_g1`** | **0.2520 (0.0148)** | `__3f71a6a9__p7-d1-vit-imagenet` |
| `p7_d1_swin_b_imagenet_g1` | 0.1076 (0.0217) | `__3f71a6a9__p7-d1-swin-imagenet` |
| `p7_d1_swin_b_scut_original_g1` | 0.1190 (0.0201) | `__3f71a6a9__p7-d1-swin-original` |
| `p7_d_vit_b16_imagenet_g2` | 0.1347 (0.0193) | `__4cb62c05__p7-d-vit-imagenet` |
| `p7_d_vit_b16_scut_original_g2` | 0.1406 (0.0333) | `__4cb62c05__p7-d-vit-original` |
| `p7_d_vit_b16_scut_masked_g2` | 0.2001 (0.0142) | `__4cb62c05__p7-d-vit-masked` |
| `p7_d_swin_b_imagenet_g2` | 0.0065 (0.0414) | `__4cb62c05__p7-d-swin-imagenet` |
| `p7_d_swin_b_scut_original_g2` | 0.2092 (0.0439) | `__4cb62c05__p7-d-swin-original` |
| `p7_d_swin_b_scut_masked_g2` | 0.1025 (0.0414) | `__4cb62c05__p7-d-swin-masked` |

Two arms span zero (`ladder.STAGE_D_AT_G2["arms_spanning_zero"]`):
`swin_b__imagenet__g2` (t = 0.35) and `agnet__imagenet__g2` (t = 1.23).
`p7_d_swin_b_scut_original_g2` is `BEST_ARM["nearest_challenger"]` — δ 0.0428,
margin 1.05×, condition 1 **0 of 5**, WITHDRAWN 2026-08-04.

### 3.2 Road A ladder — graphs (10 seeds, n=237)

| arm | PCC (sd) | run dir suffix |
|---|---|---|
| `p7_d1_srgnn_imagenet_g1_native` | 0.1719 (0.0630) | `__3f71a6a9__p7-d1-srgnn-imagenet` |
| `p7_d1_srgnn_scut_original_g1_native` | 0.1198 (0.0399) | `__3f71a6a9__p7-d1-srgnn-original` |
| `p7_d1_srgnn_scut_masked_g1_native` | 0.1201 (0.0211) | `__3f71a6a9__p7-d1-srgnn-masked` |
| `p7_d1_agnet_imagenet_g1_native` | 0.0613 (0.0426) | `__3f71a6a9__p7-d1-agnet-imagenet` |
| `p7_d1_agnet_scut_original_g1_native` | 0.0209 (0.0363) | `__3f71a6a9__p7-d1-agnet-original` |
| `p7_d1_agnet_scut_masked_g1_native` | 0.0634 (0.0278) | `__3f71a6a9__p7-d1-agnet-masked` |
| `p7_d_srgnn_imagenet_g2_native` | 0.0926 (0.0350) | `__4cb62c05__p7-d-srgnn-imagenet` |
| `p7_d_srgnn_scut_original_g2_native` | 0.0674 (0.0402) | `__4cb62c05__p7-d-srgnn-original` |
| `p7_d_srgnn_scut_masked_g2_native` | 0.1507 (0.0251) | `__4cb62c05__p7-d-srgnn-masked` |
| `p7_d_agnet_imagenet_g2_native` | 0.0221 (0.0567) | `__39871f12__p7-d-agnet-imagenet-2` |
| `p7_d_agnet_scut_original_g2_native` | 0.0530 (0.0457) | `__39871f12__p7-d-agnet-original-2` |
| `p7_d_agnet_scut_masked_g2_native` | 0.1300 (0.0301) | `__39871f12__p7-d-agnet-masked-2` |
| `p7_e_srgnn_scut_masked_g2_grid` | **per-arm value NOT IN REPO** | `__4cb62c05__p7-e-srgnn-grid` |
| `p7_e_srgnn_scut_masked_g2_anatomy` | **per-arm value NOT IN REPO** | `__4cb62c05__p7-e-srgnn-anatomy` |
| `p7_e_srgnn_scut_masked_g2_random` | 0.1354 (0.0276) | `__f46226fb__p7-e-srgnn-random-2` |
| `p7_e0_srgnn_imagenet_g2_grid` | 0.0778 (sd not recorded) | `__48e3d710__p7-e0-srgnn-grid` |
| `p7_e0_srgnn_imagenet_g2_anatomy` | 0.0765 (sd not recorded) | `__48e3d710__p7-e0-srgnn-anatomy` |
| `p7_e0_srgnn_imagenet_g2_random` | 0.0794 (sd not recorded) | `__48e3d710__p7-e0-srgnn-random` |

Stage E grid/anatomy: only the **range** is banked —
`graph_cleft.PRETRAINING_DOES_NOT_PREDICT_TRANSFER` records
"Stage E (masked-G2 …): 0.0554 – 0.1507". `STAGE_E_COMPLETE` banks only the
random arm.

> **[SUPERSEDED since 2026-08-31 — true as generated, not rewritten.]** The endpoints are now
> banked per arm at `train/graph_cleft.STAGE_E_PER_ARM`, and
> `PRETRAINING_DOES_NOT_PREDICT_TRANSFER["endpoints_now_traceable_2026_08_31"]` points at it.
> The statement above was accurate at commit `073bed28` and is left standing: a snapshot that
> is edited to stay current is no longer a snapshot.

### 3.3 Main line, non-ladder

| arm | PCC (sd) | seeds | run dir |
|---|---|---|---|
| `p11_mse_control` | 0.2552 | 5 | `p11_mse_control__27a18e04__p11-mse-control` |
| `p12_arm_a_frontal` | 0.2505 (0.0077) | 5 (**n=236**) | `__0dc7c79b__p12-arm-a-frontal` |
| `p12_arm_b_basal` | 0.1880 (0.0388) | 5 (236) | `__0fb33b6a__p12-arm-b-basal` |
| `p12_arm_c_concat` | 0.2640 (0.0321) | 5 (236) | `__0fb33b6a__p12-arm-c-concat` |
| `p12_arm_d_capacity` | 0.2534 (0.0105) | 5 (236) | `__0dc7c79b__p12-arm-d-capacity` |
| `p15_probe_mebeauty_g1` | −0.0693 (0.0252) | 5 | `__e57a8dea__p15-probe-mebeauty-g1` |
| `p15_probe_mebeauty_g2` | −0.0299 (0.0417) | 5 | `__e57a8dea__p15-probe-mebeauty-g2` |
| `p15_probe_mebeauty_original` | 0.2537 (0.0222) | 5 | `__e57a8dea__p15-probe-mebeauty-original` |
| `p16_anchor_loop` | 0.2040 (0.0345) | 5 | `p16_anchor_loop__f342fed9__p16-anchor-loop` |
| `p16_identity_baseline` | 0.2151 (deterministic, sd ~3e-17) | 5 | same dir, `identity_predictions` stem |
| `p17_arm_a` | 0.2334 (0.0044) | 5 | `p17_arm_a__bf09bd45__p17-arm-a-2` |
| `p17_arm_b` | −0.0044 (0.0317) | 5 | `p17_arm_b__4894169c__p17-arm-b` |
| `p17_arm_c` | −0.0185 (0.0586) | 5 | `p17_arm_c__bf09bd45__p17-arm-c` |

**The p12 group is n=236, not 237** — corrected in the lock after run
`p18-metric-space-2`'s row-count guard fired. Patient 238 is excluded from
~~both arms~~ **all four arms**.

> **[CORRECTED 2026-09-05 — wrong at this document's own commit.]** Phase 12 has FOUR arms,
> not two. `phase18.ARM_LIST_LOCKED["included"]["p12_view_ablation"]` carries a
> [VERIFIED 2026-08-30] comment reading "All four arms share the 236 both-views cohort", and
> `phase12.STOP_1_MANIFEST` records the derived counts as 236/236/236 with exactly one patient
> dropped. The exclusion is right; the arm count is not.

### 3.4 Road B (annex-framed)

`roadb_resolution_transformer` — 5 seeds, all at sha `30cfbfe2`:
vit imagenet 224/512/768 = **0.2520** (0.015) / 0.0894 (0.020) / 0.0937
(0.022); vit masked 224/512 = 0.0846 (0.015) / −0.1002 (0.019); swin imagenet
224/512/768 = 0.1076 (0.022) / **0.2050** (0.021) / 0.1537 (0.030); swin masked
224/512 = 0.1327 (0.027) / 0.0301 (0.027).

`roadb_resolution_graph` — 10 seeds, all at sha `a54cdfae`: srgnn imagenet
0.1719 / 0.1345 / 0.1176; srgnn masked 0.1249 / 0.1135 / 0.0498; agnet imagenet
0.0612 / 0.0309 / 0.0929; agnet masked 0.0678 / 0.0410 / 0.0020.

`roadb_regioncrop` — 5 seeds, sha `52875413`: anatomy_concat_vit 0.2428
(0.0154), control_whole_vit 0.2307 (0.0223), random_concat_vit 0.1655 (0.0310).

**Seed-regime correction on record:** the lock originally said 10 seeds for the
whole Road B resolution group; run `p18-metric-space-3`'s seed guard fired and
the group was split. The dated note calls the error "the record generalising the
agnet arm's ten-CSV listing … to every Road B arm."

### 3.5 VOID and SUPERSEDED — do not cite these numbers

| what | reason | record |
|---|---|---|
| **The pre-restart ladder** (two void ladders) | missing BatchNorm running statistics / head-init defect. Same nominal arm differed by **0.068 PCC** between void and good runs | `ladder.SIBLING_RUNS_AUDIT`; ledger `void-first-ladder` |
| **p17 arm A's first launch, SHA `eb5a6887`** | pre-fix crash (piecewise records lacked identity/deformation keys), **no surviving CSVs**. Survivor is the `-2` run at `bf09bd45` | `phase18.ARM_LIST_LOCKED["excluded"]["void_runs"]`; `phase17.PHASE_17_CLOSING["build_cycle_defects"]` |
| **Six p7_g label variants** | MSE objective but a **different target**; re-scoring against the mean truth would score a label mismatch as model error | `phase18.ARM_LIST_LOCKED["excluded"]["p7_g_label_variants"]` |
| **Phase 7C's seven augmentation arms** | "not superseded — none of them ran the policy its config named"; selected epoch mostly 1 | `phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION["arms_are_void"]` |
| **Phase 7C v1 and v2** | every arm stopped at epoch 1 | `scripts/generate_phase7c_configs.py` |
| **Road B Branch 3, first five-arm set** | five defects; corrected set closed with nothing claimable | ledger `roadb-region-crop` |
| **The partial `p7-e-srgnn-random`** (4 seeds, interrupted) | short seed count — "a new shape": right numbers, easier bar for condition 1 | `ladder.LADDER_JOB_IDS` pins `-2` |
| **`p7b-search-2`** | same trial scores, died writing its summary; re-run not recovered | `phase7b.SEARCH_AXIS_VERDICTS["superseded"]` |
| **Eight pre-policy SCUT pretraining runs** | superseded by the FIXED-budget policy (patience terminated on noise) | `config/schema.py` |
| **`per_region` embedding kind** | superseded for graph extraction | `embeddings.py` |
| **Phase 18 runs 1–4** | run 5 is the citable one; 1–4 are the defect audit trail | `phase18.PHASE_18_CLOSING` |
| **roadb vit/swin masked_768** | never ran — configs shipped, roadb_p6 768 pretrain dependencies still PENDING | `ARM_LIST_LOCKED["excluded"]["no_run_in_the_record"]` |
| **`ladder.MASKED_G1_ARTEFACT`** | not void, but the closest thing to a phantom: a checkpoint at SCUT 0.7893 manufactured **three phantom findings, each withdrawn** | `ladder.MASKED_G1_ARTEFACT` |

### 3.6 ⚠ Two gaps in the locked list

1. **The six Phase 7D arms are neither in the lock nor named in its
   exclusions** — `vit_b32` 0.2519 (0.0313), `vit_b16` 0.2520 (control, no new
   run), `vit_b8` 0.0432 (0.0340), `concat` 0.2594 (0.0208), `vit_b32_512`
   0.2280 (0.0217), `mvitv2_b` 0.1844 (0.0410), all 5 seeds
   (`ladder.PHASE_7D_OBSERVED["arms"]`). The lock's own ruling says exclusions
   must be **by name**; these are simply absent. Same for
   `roadb anatomy_concat_srgnn` 0.1717 (0.0222), n=10
   (`roadb.REGION_CROP_ARMS_OBSERVED`).
2. **Two of the six excluded p7_g arms never ran at all.** `p7_g0_*_median`
   and `p7_g0_*_ldl` are excluded for label mismatch, but
   `ladder.UNRUN_STAGES = ("G0",)` — they have no vectors. The operative fact
   is absence, not mismatch.

Neither gap changes any verdict. Both are list hygiene, flagged for a ruling.

**[RESOLVED 2026-08-31]** The ruling was a **dated addendum, not a reopened
lock** — `EXIT_CRITERIA`'s nothing-added clause stands and is not broken, since
no criterion, arm or figure enters the analysis.

- **Gap 1** → `phase18.ARM_LIST_ADDENDUM`, with `NOT_COVERED_BY_PHASE_18` as a
  tested literal so a 68-arm τ is never requoted as covering 75. Verified at
  source: `phase18.py` mentions `7D`, `p7d` and `anatomy_concat_srgnn` **zero
  times** and no exclusion key names them — **no reason is recorded, and the
  absence is recorded as oversight** rather than given one it never had.
  Correction to this section: **`vit_b16` is not a distinct arm** — it is 7D's
  control, recorded as "EXISTING — p7_d1_vit_b16_imagenet_g1 at 0.2520", which
  *is* in the locked 68. So **six** arms sit genuinely outside the lock, not
  seven.
- **Gap 2** → a dated note beside the p7_g exclusion, original preserved:
  `ladder.UNRUN_STAGES == ("G0",)` verified at source, so for the two `p7_g0_*`
  arms the operative fact is **absence, not mismatch**; the label reason remains
  correct for the four G-stage arms that ran.

---

## 4. Specific figures, verified at source

### 4.1 Phase 13 — the confound ceiling

**Record:** `phase13.P1_CONFOUND_CEILING_BANKED` (banked 2026-08-24, run
`p13-confound-ceiling`, single clean attempt, 237/237).

**The five statistics** (from `phase13.CLOSING_ADDENDUM_REGISTERED
["p1_confound_ceiling"]["what"]`) — "NON-ANATOMICAL GLOBAL STATISTICS alone":

> aspect ratio · brightness mean · contrast (pixel sd) · content-pixel fraction · corner-white fraction

**The figures:** per-seed OOF PCC **+0.1183 / +0.0659 / +0.1216 / +0.0897 /
+0.1044** (seeds 1337, 2024, 7, 99, 12345); **mean +0.1000, sd 0.0228**,
against a threshold of 0.10 declared *before* the number existed.

**The quotable sentence, as recorded:**

> "approximately 0.10 of correlation, ~40% of the headline arm's, is
> recoverable from five statistics that cannot see a nose" — STATED WITH THE
> SEED SPREAD, never as a razor-edge pass

It fires **exactly on** the threshold; the record says so itself and calls that
part of the verdict. Status: DESCRIPTIVE, no ladder entry, **no ledger row**.
It subsumes `phase12.BASAL_CONFOUND_OBSERVED`'s single-variable
r(AR, mean) = +0.1601.

Companion, `phase13.P2_BAND_OCCLUSION_BANKED`: eyes-occluded 0.2143
[0.194, 0.236], nose-occluded 0.1684 [0.144, 0.180], lips-occluded 0.0885
[0.068, 0.125] — drops of −0.036 / −0.082 / −0.162 against a "floor ~0.10
image statistics".

**No JPEG or file-size statistic is used** — NOT IN REPO as a confound-ceiling
predictor.

### 4.2 Road B Branch 1 and Phase 7D grid density

**Record:** `roadb.BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER`.

```
means: {224: 0.2319, 512: 0.0857, 768: 0.0898}
sds:   {224: 0.012436, 512: 0.021425, 768: 0.016739}
shape: "a cliff at the pretraining boundary, not a decline"
plateau: delta 0.0041 against arm_means_95 0.017
```

Mechanism as recorded: *"a cliff then a plateau says the damage is done by
leaving the pretraining resolution at all"*.

**Phase 7D retired that mechanism.** `ladder.PHASE_7D_CLOSING` /
`PHASE_7D_OBSERVED`: patch8 vs b16 **−0.2089** (5/5, 6.42×) and b32@512 vs
patch16@512 **+0.1386** (5/5, 5.32×), both CLAIMABLE. The implicated variable
is **grid density**, ordering:

> 49 tokens 0.2519 · 196 tokens 0.2520 · 256 tokens 0.2280 | 784 tokens 0.0432 · 1024 tokens 0.0857 · 2304 tokens 0.0898

> "arm 3 collapsed with ZERO interpolation and arm 5 survived under the
> IDENTICAL 2.286x stretch — jointly, the Road B cliff's mechanism was never
> interpolation"

Tokens and per-patch fraction are deterministically linked: **one variable,
not two**.

### 4.3 ⚠ The detection floor — NOT IN REPO as stated

**There is no record stating a detection floor of ΔPCC ≈ 0.12–0.14.** What is
recorded:

- `ladder.COHORT_CANNOT_RESOLVE` (2026-08-04): *"this cohort cannot resolve PCC
  differences of **0.04 to 0.10** between arms"*; at scale, **30 tested, 1
  survived, 29 withdrawn**.
- `roadb.ROAD_B_PHASE_7_CLOSING`: *"this cohort resolves effects of
  ViT-collapse size and nothing smaller"* — those deltas are −0.185, −0.163,
  −0.158.
- **Smallest delta ever to pass both conditions: +0.1386** (`p7d-grid-density`).
- `roadb.CONDITION_1_MARGIN_STRUCTURE` records the floor as a **margin**, not a
  delta: *"nothing at or below 2.55x has ever passed"*; *"everything at or
  above 5.32x has passed — five for five"*.

A floor somewhere between 0.10 (unresolvable) and 0.1386 (smallest pass) is a
**defensible inference**, but it is an inference — it is not written anywhere,
and it should be labelled `[REASONED]` if used.

### 4.4 Seed bands

| band | value | record |
|---|---|---|
| linear probe / transformer (Road A) | n=10, mean 0.2529, **sd 0.0137**, range 0.0373; claimable δ **0.038 / 0.017 / 0.012** at 1 / 5 / 10 seeds | `train/phase3.MEASURED_SEED_BAND` |
| graph regime (Road A) | n=10, mean 0.15067, **sd 0.025053**; claimable δ 0.069442 / 0.031056 / 0.021960 | `train/graph_cleft.MEASURED_GRAPH_SEED_BAND` |
| graph regime (Road B, planning) | **sd 0.041452** (ratio 1.6546 to Road A, F CI [0.823, 3.326] contains 1 → one band, the larger); claimable δ 0.115 / 0.051 / 0.036 | `roadb.PHASE_7_GRAPH_BAND_RESOLVED` |
| transformer (Road B, planning) | **0.021425**; claimable δ 0.059 / 0.027 / 0.019 | `roadb.SEED_BAND_RESOLVED` |
| Phase 17 arm A | **sd 0.0044** (B 0.0317, C 0.0586) | `phase17.PHASE_17_CLOSING` |

Arm A's 0.0044 is ~3.4× tighter than the probe's 0.0148; the ledger records
that this **shrinks** the threshold and is a property of the arms' variances,
not a strengthened claim.

`arm_means_95` is a **derived** value returned by
`phase3.combined_claimable_delta`, not a stored constant (one literal appears
in `roadb.BRANCH_1_PREMISE_REFUTED_FOR_FROZEN_TRANSFER["plateau"]` = 0.017).

### 4.5 The ceiling 0.9032

**Record:** `data/reliability.PCC_CEILING_237 = 0.9032`, mirrored as
`data/labels.CEILING_MEAN_OF_FIVE`.

**Confidence interval: NOT IN REPO.** No CI, BCa or bootstrap interval is
attached to 0.9032 anywhere.

What it is: reliability is var(true)/var(observed); the maximum correlation any
predictor can reach against the *observed* panel mean is its **square root**.

```
MEAN_R_237 = 0.4696   RELIABILITY_237 = 0.8158   PCC_CEILING_237 = 0.9032
MEAN_R_251 = 0.4560   RELIABILITY_251 = 0.8073   PCC_CEILING_251 = 0.8985
FLEISS_237 = 0.1662   QWK_237 = 0.4276          MANIFEST_POPULATION = 237
CEILING_SINGLE_RATER = 0.685  (= sqrt(0.4696))
```

The 237 figures are the ones that matter — the patients with photographs. The
251-row figures exist "only so that a number computed on the other cohort is
recognisable rather than mistaken for an error."

**Three registered traps:**
1. The lineage's projected **panel reliability 0.90/0.902 is NOT** this
   correlation ceiling 0.9032 — "different quantities, coincidentally adjacent,
   never placed in proximity unqualified."
2. Comparing across populations "is exactly how the 0.903 ceiling came to be
   'corrected' to 0.807" — an R2 instance.
3. `mean_pairwise_qwk` = 0.4173 is **never a ceiling**; it is an agreement
   index.

### 4.6 The human bar (per-rater item-total, 237)

> **[SUPERSEDED since 2026-08-31 — the population tag in this heading is not safe to rely
> on.]** The per-rater item-total values reachable from `docs/BRIEF_phase1.md` §1.6 are the
> **251**-row population, and `data/reliability.item_total`'s own docstring cites "§1.6"
> without naming a population. Read the population off the constant you are quoting
> (`MEAN_R_237` / `MEAN_R_251` and their siblings), never off this heading. Left as generated.

`data/reliability.item_total` — each rater against the mean of the other four;
`r` Pearson, `rho` Spearman. **Only two of the five values are in
`src/cleft/`:** SLT highest at **0.654**, orthodontist **0.628** — "which
contradicts a premise raised at supervision."

**The full five-rater table is NOT IN `src/` — it is in
`docs/BRIEF_phase1.md` §1.6:**

| rater | r | ρ | mean QWK |
|---|---|---|---|
| Cleft patient | 0.575 | 0.564 | 0.420 |
| Orthodontist | 0.628 | 0.603 | 0.449 |
| Speech & language therapist | **0.654** | **0.654** | 0.447 |
| Plastic surgeon | 0.545 | 0.540 | 0.394 |
| Psychologist | 0.547 | 0.549 | 0.377 |

**Do not merge with** the *trained-model* per-rater PCCs
(`phase10.RATER_SCREEN_OBSERVED`): cleft patient −0.0098, orthodontist 0.1606,
SLT 0.2150, plastic surgeon 0.2703, psychologist 0.1362. Different quantity.
And **0.4696 is not an item-total figure** — it is the mean inter-rater r.

### 4.7 One line each

- **Phase 6 masking** — `ladder.STAGE_D1_AT_G1["q2"]`: *"no reproduction of the
  G2 conflict: one claimable negative, one claimable positive, two null.
  Masking's answer is operating-point dependent"* (ViT −0.1122, swin +0.0137,
  srgnn +0.0003, agnet +0.0425). Road B cousin
  `roadb.PHASE_6_RESOLUTION_RESULTS`: *"FLAT where readable"*.
- **Phase 4 partition sensitivity** — **the measured result is NOT IN REPO.**
  Only the instrument and its declared gap: `train/partition.NOT_MEASURED` —
  *"sensitivity to partitioning — a BOUND on it, not a measurement of it, and
  confounded with training-set size"*, `captured_by_neither: "fold assignment"`.
  No per-partitioning PCC is recorded anywhere.
- **Phase 9 prototypes** — `phase9.PROTOTYPES_OBSERVED`: *"medoids stand as
  descriptive artifacts; representativeness claims are dead"*. Medoids
  183/147/148; persistence 0.394/0.427/0.414; companion 0.257 vs majority
  0.502. The LOO figures 0.992/0.967 **may never be quoted as stability**.
- **Phase 11 IEM defects — three named:** value inversion below
  **|d| = 0.197531** (`IEM_CROSSOVER_INVERTS`); gradient inversion below
  **0.0719** (`DEGENERACY_PULL_REGISTERED`) — "the same defect seen from the
  scoring and the training side"; and `IEM_BOUNDS_ARE_A_DOMAIN` — at |d| = 4
  the branches give 5.6688 and 2.6723, so **[0,4] is a domain, not a range**.
  A fourth, recorded separately: `A_FAVOURS_NARROW_PREDICTORS`.
- **Phase 12 view ablation** — `phase12.PAIRED_OBSERVED["sharpest_finding"]`:
  *"even B-vs-A is WITHDRAWN … a contrast whose DIRECTION NOBODY DOUBTS …
  the SIXTH arrival at COHORT_CANNOT_RESOLVE, and the first where the direction
  was never in question"*. A 0.2505, B 0.1880, C 0.2640, D 0.2534.
- **Phase 14 concession** — `phase14.PHASE_14_CONCEDED_COVERED`:
  *"CONCEDED-COVERED. Nothing was built"*. Prior measurement carried: ldl
  0.2336 vs mean 0.2520 (δ −0.0184, inside its 0.0297 threshold); 0.1583 vs
  0.2001 at masked/G2, **claimably harmful**. No ledger entry.
- **Phase 15 MEBeauty** — `phase15.PHASE_15_CLOSING`: *"**there is no single
  best dataset, and pooling would manufacture one.** ORIGINAL geometry:
  MEBeauty (0.2537 vs 0.1952). G1 MASKED: SCUT (0.0830 vs −0.0693). G2 MASKED:
  SCUT (0.2001 vs −0.0299). Three claimable differences that do not agree on a
  winner"*.

---

## 5. Is there a label-permutation control?

## **NO. NO LABEL-PERMUTATION CONTROL EXISTS IN THE REPO.**

> **[SUPERSEDED since 2026-08-31 — and this is the one that turned over fastest.]** Phase 20
> IS the label-permutation control: `phase20.py` holds the permutation machinery
> (`quantile_strata`, `permute_within_strata`) and `phase20.ARMS_OBSERVED` records four runs
> dated **2026-08-31**, the same day this document was generated. The answer above was correct
> when the search was run and is left standing as the reason the phase exists.

Searched by mechanism across `src/`, `tests/`, `configs/`, `docs/`, `scripts/`:
`permut*`, `shuffl*`, `np.random.permutation`, `rng.shuffle`, "null label",
"label control", "randomised label", "scrambled", "chance baseline", "negative
control", "y_shuffle", "random target", "permutation test". **Zero hits fit or
refit a model on shuffled target labels.**

Every hit classified:

| what it is | where | what it actually controls |
|---|---|---|
| **row/batch index permutation** | `run.py` (split seeds), `harness.py`, `torch_backbone.py`, `folds.py` | splitting and batching — not a control |
| **randomised-WEIGHT backbone** | `phase8.THE_RANDOMISED_BACKBONE_COLLAPSES`, `extract.RANDOMISED_BACKBONE` | parameter dependence — labels untouched |
| randomised finals (Grad-CAM) | `node_weights.py` | recorded 2026-08-15 as **UNINFORMATIVE** — permutation symmetry over the region axis is architectural |
| bootstrap resampling | paired BCa throughout | interval estimation, not a null |
| **majority / constant-predictor floor** | `classification.majority_baseline` (acc 0.5021, macro F1 0.2228), `phase10_annex.FLOORS_REGISTERED`, `phase18.constant_predictor_iem` | class imbalance only |
| random-**placement** crops | `configs/roadb_p7c_arm_random_concat_vit.yaml` | placement — real labels |
| fixture shuffles | `tests/test_folds.py`, `test_report.py` | build skewed fixtures; no model fit |

**The sharpest fact.** The mechanism appears in the repo **only as a hazard
guarded against, never as a control that is run**:

> `tests/test_pooled.py:142` — "A permutation trains every patient against
> another patient's label and still produces a plausible PCC."

> `scripts/compare_live_and_stored_features.py:32` — "a permutation trains
> every patient against another's label and still fits."

So the project knows precisely that fitting on permuted labels yields a
plausible score, and has used that knowledge exclusively to build misalignment
guards. **The chance level of the headline PCC has never been established by
permutation.** Given §4.3 (the cohort cannot resolve 0.04–0.10) and §4.1
(≈0.10 recoverable from five non-anatomical statistics), this is the most
consequential gap in the record.

---

## 6. The standing corrections list

### 6.1 VOID runs (7 in the ledger + 3 outside)

| id | record | date | replaced by |
|---|---|---|---|
| `void-first-ladder` | `ladder.SIBLING_RUNS_AUDIT` | before 2026-08-02 | rebuilt ladder; hazard measured at **0.068 PCC** drift |
| `void-scut-animation-first-launch` | `phase8.SCUT_ANIMATION_FIRST_LAUNCH` | 2026-08-16 | the `-2` runs at SHA a2f7c8c3 |
| `void-deall-first-launch` | `phase9.DEALL_REFERENCE_READS` | 2026-08-16 | canonical layout restored; 76-file rollup |
| `void-cleftgnn-first-launch` | `phase10.CLEFTGNN_FIRST_LAUNCH_VOID` | 2026-08-17 | `CONSENSUS_LABEL_IS_THE_MEDIAN` |
| `void-cleftgnn-second-launch` | `phase10.CLEFTGNN_SECOND_LAUNCH_VOID` | 2026-08-17 | `CLEFTGNN_NAN_MEASURED` |
| `void-cleftgnn-third-launch` | `phase10.CLEFTGNN_THIRD_LAUNCH_VOID` | 2026-08-17 | `CLEFTGNN_SCALE_MEASURED` |
| `roadb-region-crop` (first set) | `roadb.REGION_CROP_STRUCTURE` | 2026-08-14 | corrected set — nothing claimable |

Outside the ledger: **Phase 7C v1 and v2** both VOID ("every arm stopped at
epoch 1", `scripts/generate_phase7c_configs.py:291`); **p17 arm A's first
launch at SHA eb5a6887** (pre-fix crash, no surviving CSVs, `phase18.py:642`).

### 6.2 Named corrections outside the ledger (selection)

| record | home | date | replacement |
|---|---|---|---|
| `MECHANISM_CLAIM_WITHDRAWN` | `phase7c.py` | 2026-08-03 | refuted by its own data (selected epoch 1.00 → 7.13) |
| `CLAIMABLY_WORSE_WITHDRAWN` | `phase7b.py` | 2026-08-04 | downgraded to UNRESOLVED |
| `FRAMING_NOT_ANATOMY` | `phase8.py` | refuted 2026-08-08 | per-patient own-expectation re-measure 1.0652, 10/15 |
| `BASAL_RATIONALE_UNSUPPORTED` | `ladder.py` | 2026-08-23 | MEASURED → UNSUPPORTED; **sixth of its error class** |
| `READING_APPLIED_TO_NOBODY` | `phase13.py` | 2026-08-24 | "THE SAMPLING SENTENCE IS WITHDRAWN BY NAME" |
| `THE_119_HYPOTHESIS_REFUTED` | `phase15.py` | 2026-08-24 | landmarks correct; overlay defect elsewhere |
| `EYE_IMPRESSION_WITHDRAWALS` | `phase15.py` | 2026-08-24 | three by-name retractions |
| `FRAMING_VARIES_LIMITATION` | `phase15.py` | 2026-08-24 | third-axis over-call withdrawn |
| `CRITERION_1_AMENDED` | `phase15.py` | 2026-08-24 | second sense withdrawn as never-operationalised |
| `MAPPING_VERIFIED_AND_A_WITHDRAWAL` | `phase15.py` | 2026-08-24 | **"0.8914 IS WITHDRAWN — the maintainer's error, recorded as such"** |
| `THREE_NOT_FIVE` candidate | `classification.py` → `phase18.PHASE_18_RULINGS` | declined 2026-08-30 | reopen condition stated |
| `G2_WHITE_DEFECT` | `scut/masked.py`, `docs/FROZEN_KNOWN_STALE.md` | resolved | a **correct** conclusion had been wrongly retracted, then restored |

`docs/FROZEN_KNOWN_STALE.md` carries four registered stale-prose items inside
frozen modules (Nadeau–Bengio reference; two dead DataLoader helpers;
trapezium "corners cease to exist"; the Dockerfile CUBLAS claim). Nothing there
changes a number — that is the file's own admission rule.

### 6.3 The mechanism failures

**(a) Phase 17 IEM anchor-scale story — the SPECIFIC mechanism FAILED.**
Registered in `phase18.P17_IEM_PATH_VERIFIED` (predicting training on the
anchor-grade scale, mean 3.14, against a panel mean near 2.96). Resolved in
`phase18.P17_IEM_MEASURED`: measured **1.878 ± 1.435 against truth
2.7544 ± 0.6587** — *under*-prediction on a too-wide scale, **not** the
predicted over-shoot. The GENERAL claim (scale miscalibration invisible to PCC,
priced by IEM) **is** measured. "Failure dated, original preserved."

**(b) D1 — mechanism PARTIALLY held.** `phase18.D1_VERDICT`: verdict
DISCORDANT, tau(PCC, macro F1) 0.5180, top-5 overlap 4/5. Covered: the p16 pair
moved as predicted; the identity baseline's macro F1 equals the 0.2228 constant
floor exactly. **Uncovered kind named:** p17 B/C at macro F1 0.0785, *below the
constant predictor's floor* — near-constant **minority**-class collapse, which
the registered near-mean mechanism did not anticipate. "Recorded as partially
failed, never adjusted."

**(c) D3 — the registered position judged WRONG.** Registered before any tau
was seen (`phase18.D3_COMPLETED`), with the pre-committed rule: HELD iff
tau(accuracy, QWK) > 0.7024. Measured **tau = 0.3392663** over 68 arms →
`phase18.D3_JUDGED`: *"THE POSITION IS WRONG, and is recorded as wrong."* Both
counts inverted — the chance correction *decouples* QWK from accuracy, and the
distance term was not idle. Commit `e1077fa`.

Also: `phase18.D2_MEASURED["six_not_seven"]` — the ruled "named seven"
double-counted `p17-a-vs-probe`; the distinct set is **six**, corrected in
place.

### 6.4 The different-quantities catches (error class R2) — about ten

**There is no single consolidated numbered list — NOT IN REPO.** The ordinals
are asserted piecemeal inside individual records. Reconstructed in order:

| # | where the ordinal is asserted | what was conflated |
|---|---|---|
| 1 | `docs/PLAN.md:147` | QWK read as a correlation ceiling |
| 2 | `docs/PLAN.md:147`, `data/labels.py` | the 0.9032 ceiling "corrected" to 0.807 — reliability vs ceiling, across populations |
| 3 | `docs/FROZEN_KNOWN_STALE.md:44` | Nadeau–Bengio: a fold-level correction attached to a patient-level bootstrap |
| 4 | `docs/PLAN.md:237` | `separability` — a check not covering what its name implied |
| 5 | `docs/PLAN.md:1509` | SymNose: PCC quoted against a Spearman-based literature finding |
| 6 | **no record asserts this ordinal** | the κ rows over two populations under one heading; and a `du -sb` "difference" that was the directory inode |
| 7 | `phase12.py:1623` | "Asher-McDade composite" (4 components) vs our single gestalt 1–5 |
| 8 | `phase18.py:184`, `classification.py:280` | 3-class vs 5-class macro F1 (0.3760 vs 0.3352); panel mean vs rater labels; 237 OOF vs 28 single split |
| 9 | `phase18.P17_IEM_MEASURED` | anchor-grade mean 2.96 vs cohort panel-mean truth 2.7544 |
| 10 | `phase10_annex.VERIFICATION_NOTES["census_correction_2026_08_30"]` | the `trainable:` census — a loose regex counted 24 header **comments** as settings |

**Gap:** instance **#6 has no record asserting its ordinal**, so the sequence
1–10 cannot be fully reconstructed from the repo alone.

Unnumbered instances carrying the same class: `roadb.py` 36-vs-37 region count;
`count_names.py` (its own rule **R11**, two instances); `augment_sheet.py`
`moved_outside`; `phase10_annex.F1_BY_RATER` (degenerate-draw mixture — "the
different-quantities error under a column heading");
`phase18.ARM_LIST_LOCKED` (the six excluded p7_g arms); and further instances in
`scut/synthesis.py`, `scut/dataset.py`, `train/ldl.py`, `train/phase3.py`,
`models/agnet.py`.

---

## 7. Open items across all phases

### 7.1 The supervision question list — `phase11.SUPERVISOR_CONSOLIDATED_ASK`

Written 2026-08-17 with seven; q1 answered the same day (convention A) and moved
to `answered_and_removed`; two appended since. **Eight remain open, none
blocking:**

| # | subject | record |
|---|---|---|
| 2 | is CASE the same metric as equation (16)? | `CASE_IEM_IDENTITY_OPEN` |
| 3 | eq (16) inverts below \|d\| = 0.19753 — intended, artifact, or fix on revision? | `IEM_CROSSOVER_INVERTS` |
| 4 | the paper's [0,4] is a domain, not a range | `IEM_BOUNDS_ARE_A_DOMAIN` |
| 5 | 27 regions stated, 36 executed — correct the count or re-run? | `phase10.REGION_COUNT_BELIEF_VS_EXECUTION` |
| 6 | slide 9's cleft numbers, metric, test-set n, intervals, crop coordinates | `phase10.SLIDE_9_CLEFT_NUMBERS_REQUESTED` |
| 7 | unanimity on the anchor grades | `phase9.PHASE_9_CLOSING` |
| **8** | were raters shown frontal only, or frontal+submental with one score? | `ladder.BASAL_RATIONALE_UNSUPPORTED` |
| 9 | does using SCUT's shipped landmark files cross the no-detector line? | `phase17.PHASE_17_RULINGS` |

Q8 carries `"priority": "HIGHEST-VALUE as of 2026-08-23 — ask FIRST despite the
number"`.

⚠ **Stale text to correct:** `SUPERVISOR_CONSOLIDATED_ASK["nothing_blocks_now"]` and
`phase11.PHASE_11_CLOSING["carried_forward_open"]` both still say "the remaining
**SIX** supervision questions" — written before q8 and q9 were appended. It is eight.

### 7.2 The supervision 28-ID dependency — `phase10_annex.SUPERVISION_28_DEPENDENCY`

"supervision has said the 28 test images come from the cohort — **[REPORTED, not
measured]**", entering via the maintainer, in no repo record. If the IDs become
recoverable, an exact-split arm is a **separate registered addition**. The
181-⊂-237 assumption is explicitly not made. This is the **only** item still
open in the annex — rater attribution, the ledger ruling and the per-rater table
all closed 2026-08-31.

### 7.3 ⚠ The blind audit — essentially NOT IN REPO

**There is exactly one mention in the entire repository**, plus its test
assertion: `phase18.PHASE_18_RULINGS["contrast_set_the_named_seven"]` —

> "the deliverable is the CRITERION'S BEHAVIOUR, not an exhaustive re-audit —
> **the blind audit exists for exhaustiveness**"

There is **no brief, no registration record, no design, no config, no run
directory, no ledger entry, and no `BLIND_AUDIT` constant anywhere.** A
case-insensitive grep for `blind[ _-]?audit` across the repo returns only that
line and its test. The `*_AUDIT` constants that do exist
(`ladder.SIBLING_RUNS_AUDIT`, `ladder.LADDER_PAIRED_AUDIT`,
`phase8.SCUT_ANIMATION_AUDIT`) are all different things.

**How far behind is it?** *Unmeasurable from the repo, because it was never
registered.* What can be said precisely: it is invoked as the thing that
*would* carry exhaustiveness, Phase 18 then closed having deliberately measured
only the **six distinct contrasts**
(`phase18.D2_MEASURED["six_not_seven"]` — the ruled "named seven"
double-counted `p17-a-vs-probe`), and the two most recent closures added **no
ledger rows**. So the exhaustive pass is outstanding against a ledger of 37
entries spanning p7–p17. **If the blind audit is real, its brief lives outside
the repo and needs importing before it can be tracked.**

### 7.4 Registered but NOT BUILT

| item | home | status |
|---|---|---|
| Stage G0 label arms | `ladder.UNRUN_STAGES`, `STAGE_G_LABEL_FORMULATION` | built, never run — the G/G1 disagreement is "unattributable until Stage G0 runs" |
| Stage B, the view ablation | `ladder.STAGE_B_NOT_RUN` | not run, deliberately; blocker was basal staging. Its premise is now UNSUPPORTED |
| The 400-run full rater ladder | `phase10.RATER_LADDER_CONDITIONAL` | conditional, **not triggered** (surgeon's +0.0183 = 1.24 sd, inside the 2-sd threshold) |
| P3 occlusion-on-reconstructions probe | `phase13` `p3_conditional_not_built` | runs only if P2 shows a substantial middle+bottom drop |
| The CleftGNN divergence fix | `phase10.CLEFTGNN_SECOND_LAUNCH_VOID["fix"]` | "NOT BUILT — a registration question, proposed not picked" |
| Shard-merge for the annex distribution | `phase10_annex` `merge_not_built` | needed only if the run is sharded |
| Phase 8 exit items 4, 5, 6 | `phase8.py` | node weights for arms B and C; A-vs-B region interval; t-SNE with companion |
| `PRE_CONTEXT_REFUSALS_LEAVE_NO_TRACE` | `phase9`, carried by p10/p12 | "to consider, deliberately not built" |

### 7.5 Unregistered variants, named so they cannot return as fresh ideas

`phase16.ANCHOR_LOOP_RULINGS["unregistered_not_deferred"]` — "naming them now is
what stops any of them being presented later as a fresh idea": (i) a 3-class
pull-loss variant; (ii) diagonal `W`; (iii) low-rank `W`. Plus
`classification.THREE_NOT_FIVE["declined_2026_08_30"]` — the 4-threshold rule
**declined**, with a stated reopen condition (a label source with support in the
extreme grades).

### 7.6 Deferred / parked

FGCM (eqs. 14–15) and NDCG@K (eqs. 17–18) — "available, still DEFERRED"
(`phase11`). The IEM loss build — unregistered. The corrected-crossover variant
— "a FUTURE ARM requiring its own registration, never a mid-phase repair".
`phase10.U_SHAPE_BELONGS_LATER`. `scut.synthesis.PARKED` — the TPS arm,
unparked into Phase 17 arm A with `arm_amended_2026_08_29` (**never call arm A
a replication**).

**Standing infrastructure item, explicitly not closed:**
`phase8.RETRY_LIMIT_IS_NOT_HOLDING` — launches ran 4–7 attempts against a stated
backoff limit of 1, incremented by every phase since (p10 +4/+6, p12 +7/+3,
p13 +2/+2/+3). The real retry field has still not been identified.

### 7.7 Caveats a closed phase carried forward

- `phase12.PHASE_12_CLOSING`: the AR confound r(AR, mean) = **+0.1601**
  "travels with ANY future use of the basal view"; the whole-image-G1-only
  staging boundary, which any region/patch or G2 basal use **reopens**.
- `phase13.PHASE_13_CLOSING`: the AR caveat still riding arms B and C; the
  **fractional-GPU note** — p13-asymmetry-2 OOM'd on a 13.04 GiB device, so the
  standard job template does not guarantee a full card.
- `phase17.PHASE_17_CLOSING`: no scar; the magnitude-to-grade assumption, the
  named candidate for why arm A works, **[REASONED] and not measured**; the
  anchored philtrum.
- `phase15.ANATOMICALLY_BLIND_LIMITATION` — standing and **unquantified**.
- `phase18`: both mechanism failures recorded as failures and **not adjusted**.

---

## 8. What a future session should not do

1. **Do not quote a "detection floor" of 0.12–0.14.** §4.3 — not in the record.
2. **Do not cite `docs/PLAN.md`'s phase numbers.** §1.2 — PLAN.md's Phase 10 is
   the write-up; the repo's Phase 10 is the CleftGNN replication.
3. **Do not read `PHASE_16_SCHEDULED` as Phase 16.** §1.3 — it is Phase 18.
4. **Do not quote 0.8914, the 0.903→0.807 "correction", the three withdrawn eye
   impressions, or `p8-framing-not-anatomy`.** §6.
5. **Do not quote Phase 9's LOO figures (0.992, 0.967) as stability.** §4.7.
6. **Do not place our IEM values beside CleftGNN's Table 2/4/6.** Tested literal
   in `phase18.DELIVERABLES_REGISTERED["cleftgnn_iem_prohibition"]`.
7. **Do not present the Phase 10 annex figures as correcting CleftGNN's
   published numbers.** Tested literal in `phase10_annex.ANNEX_PROHIBITION`.
8. **Do not call Phase 17 arm A a replication of Rosero.** `scut.synthesis`
   `arm_amended_2026_08_29`.
9. **Do not assume a chance baseline exists.** §5 — no label-permutation control
   has ever been run.
