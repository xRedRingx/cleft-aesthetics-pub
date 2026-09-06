# BRIEF — PHASE 1: MANIFEST, LABELS, FOLDS

**Repo:** `cleft-aesthetics`
**Governing document:** `docs/PLAN.md` **v2** — Part 4.1 (data), 4.2 (label), 4.3 (ceilings).
**Supersedes:** any earlier Phase 1 brief. An earlier draft stated the ceiling as 0.807 and told you
0.903 indicates a bug. **That draft was wrong.** See §1.5.

---

## PROVENANCE TAGGING

Facts below are tagged **[MEASURED]** (computed from project data), **[LITERATURE]** (published), or
**[REASONED]** (inferred, may be wrong). Only [MEASURED] facts become assertions in code. If you
find yourself about to encode a [REASONED] item as a hard invariant, stop and flag it instead.

---

## 0. THE BOUNDARY THAT MATTERS MOST

**You cannot see the clinical data, and must not try.** The cohort never leaves EHU infrastructure.
There is no copy on the laptop and there must never be one.

| where | what |
|---|---|
| **laptop (you)** | write modules and tests, run against **synthetic fixtures** |
| **cluster (user)** | run the real build as a keeper job, report aggregates only |

Every test constructs its own fake cohort. The real inputs are described precisely below; build
fixtures matching that shape, including both known exceptions.

**Tier discipline:** patient-keyed outputs are `CLUSTER-ONLY`. Aggregate scalars are `SHAREABLE`.

---

## 1. THE VERIFIED FACTS — these become tests before any code

### 1.1 Structure [MEASURED]

| fact | value |
|---|---|
| patient folders | **238 directories, all present**; folder **52 exists and is empty**, so **237** hold photographs (corrected 2026-07-27) |
| images per folder | 2, except folder 238 which has 1 |
| total images | 473 |
| frontal images | 237 (every one scored) |
| basal images | 236 (none scored) |
| score-sheet rows | 251 |
| score rows with no photograph | **14** — IDs 716, 718 … 742 (even, contiguous) |

`237 + 14 = 251`. Assert this explicitly.

### 1.2 The frontal rule and its two exceptions [MEASURED]

Folders 1–171: the **odd** image ID is frontal. Folders 172–237: the **even** image ID is frontal.

- **Folder 143** — images 523, 524. The rule predicts 523. **524 is the frontal.**
- **Folder 238** — single image **581**, odd despite being past 171. Frontal. No basal.

235 of 237 follow the rule.

**THE CRITICAL DESIGN POINT.** Assign frontal by **score-sheet membership** — the scored ID *is* the
frontal. Use the odd/even rule **only as a cross-check**, asserting it disagrees in **exactly**
folders 143 and 238. A third disagreement is a hard failure.

Building from the rule would silently swap patient 143's views, and nothing would error. That is the
failure this phase exists to prevent.

### 1.3 The score sheet [MEASURED]

Columns: `RanaPhotoID`, five raters, `Average`, `Median`. One composite 1–5 grade per rater — **not**
four Asher-McDade components.

Raters: `Rater 7 - Cleft patient`, `Rater 8 - Orthodontist`, `Rater 9 - Speech and language
therapist`, `Rater 10 - Plastic surgeon`, `Rater 11 - Psychologist`. Some headers have doubled
spaces; normalise rather than matching exactly.

Two workbooks exist, **identical in content** — one text, one integer. Mapping verified across all
1,255 cells, zero missing:

```
Excellent = 1   Good = 2   Fair = 3   Poor = 4   Very poor = 5
```

Note the lowercase "poor" in "Very poor". The loader accepts either form and produces the same result.

### 1.4 Reliability — must reproduce exactly [MEASURED]

Computed on the **237 patients with photographs**:

| statistic | value |
|---|---|
| mean inter-rater r | **0.4696** |
| Spearman-Brown reliability (k=5) | **0.8158** |
| **PCC ceiling = sqrt(reliability)** | **0.9032** |

On all 251 scored rows: r=0.4560, SB=0.8073, ceiling 0.8985.

| statistic | value |
|---|---|
| Fleiss κ | **0.1605** |
| mean pairwise QWK | **0.4173** |

### 1.5 THE CEILING — read carefully [MEASURED]

**Reliability and a correlation ceiling are different quantities.**

- Reliability ρ = var(true) / var(observed) = **0.8158**
- Max correlation of any predictor with the **observed** panel mean = √ρ = **0.9032**

**The ceiling is 0.903.** An earlier brief said 0.807 and instructed you to treat 0.903 as a bug.
That instruction was wrong — it came from computing reliability on 251 rows and comparing it against
a ceiling computed on 237.

Encode **both** numbers with their names, so the distinction cannot collapse again:
`reliability_237 = 0.8158` and `pcc_ceiling_237 = 0.9032`. A test asserting √reliability = ceiling
is worth having, because it makes the relationship explicit in code rather than in a comment.

### 1.6 Per-rater reliability [MEASURED] — against the mean of the other four

| rater | r | ρ | mean QWK |
|---|---|---|---|
| Cleft patient | 0.575 | 0.564 | 0.420 |
| Orthodontist | 0.628 | 0.603 | 0.449 |
| **Speech & language therapist** | **0.654** | **0.654** | 0.447 |
| Plastic surgeon | 0.545 | 0.540 | 0.394 |
| Psychologist | 0.547 | 0.549 | 0.377 |

The SLT is most reliable on r and ρ. This contradicts a premise raised at supervision. Assert it.

### 1.7 Label learnability [MEASURED]

| target | learnability |
|---|---|
| Reliability-weighted mean | 0.5926 |
| **Mean of 5 — PRIMARY** | **0.5900** |
| Median of 5 | 0.5561 |
| Mode of 5 | 0.4880 |
| Orthodontist | 0.4811 |

The mean is ~23% more learnable than the orthodontist column. Weighted beats plain by 0.0026 —
negligible, so plain mean is primary.

---

## 2. WHAT TO BUILD

### 2.1 `src/cleft/data/scoresheet.py`
Load either workbook form. Normalise headers. Map text→integer. Assert 251 rows, 5 raters, zero
missing. Return a structure keyed by `RanaPhotoID`.

### 2.2 `src/cleft/data/labels.py`
Produce `mean`, `median`, `mode`, `weighted_mean` (weights = corrected item-total correlations),
`orthodontist`, and `soft_1..soft_5` (fraction of raters per grade — these exist in a derived CSV
and must match). Plus the 3-class collapse at **fixed 2.5 / 3.5**, a priori, never tuned.

### 2.3 `src/cleft/data/reliability.py`
Fleiss κ, pairwise QWK, Cronbach α, mean inter-rater r, Spearman-Brown, **the ceiling as √SB**,
per-rater item-total correlations, and the learnability computation. Every number in §1.4, §1.6 and
§1.7 reproducible from here.

### 2.4 `src/cleft/data/manifest.py`
Inputs: patient-folder directory + loaded score sheet. Rows of:

```
patient_id, frontal_id, basal_id, mean, median, mode, weighted_mean, orthodontist,
soft_1..soft_5, class3
```

`basal_id` null for patient 238. Assert every invariant in §1.1 and §1.2. Emit the 14 photo-less
scored IDs to a **separate record** — explicit, logged, counted. Never a silent drop.

### 2.5 `src/cleft/data/folds.py`
`StratifiedGroupKFold`, 5 folds, groups = `patient_id`, stratified on the 3-class label. Generated
**once** and written as a versioned artifact. Assert: every patient appears in exactly one test
fold; every fold contains all three classes; the same patient never spans folds.

**Folds are now in scope.** The earlier brief deferred them pending a design decision; that decision
resolved — CV and TSTR answer different questions, so folds are needed either way. See PLAN v2 §4.9.

### 2.6 `configs/p1_build_manifest.yaml`
A keeper config the user runs on the cluster. Declares inputs by path and hash; writes to
`data/manifests/cleft_v1/` with `MANIFEST.json` per PLAN §2.6.

---

## 3. FIXTURES

A synthetic cohort matching the real shape: 237 folders with the same numbering and gap at 52, two
images each with the real ID arithmetic, folder 238 with one, **folder 143 inverted**, a 251-row
score sheet including 14 photo-less IDs, fixed-seed random grades.

The generator must also produce **broken** variants: a third rule violation, a missing frontal, a
duplicate scored ID, a folder with three images, a missing cell.

Reliability functions are tested separately against small hand-computable matrices — the fixture's
random grades will not reproduce 0.1605.

---

## 4. EXIT CRITERIA

**Laptop:**
1. `bash scripts/test.sh` green, still under 60s.
2. Synthetic cohort builds a manifest with correct counts and both exceptions flagged.
3. Every broken variant rejected with a message naming the specific violation.
4. Reliability matches hand-computed values on small matrices, including √SB = ceiling.
5. Folds satisfy their invariants on the synthetic cohort.

**Cluster (user runs, reports aggregates only):**
6. Real manifest builds: 237 rows, 237 frontal, 236 basal, 14 exclusions logged.
7. Rule cross-check flags exactly folders 143 and 238.
8. Reliability reproduces, **on the 237** (corrected 2026-07-27): **Fleiss 0.1662**,
   **mean pairwise QWK 0.4276**, **SB 0.8158**, **ceiling 0.9032**.
   The 0.1605 / 0.4173 originally quoted here are the **251-row** values from
   `summary.json` — the same population trap as the learnability figures. Both are
   named separately in `reliability.py` as `FLEISS_237`/`FLEISS_251` and
   `QWK_237`/`QWK_251`; compare a cluster run against the 237 column.
9. Learnability reproduces the §1.7 ordering — mean ~0.590, orthodontist ~0.481.
10. Folds generated, hashed, and every patient in exactly one test fold.

---

## 5. OUT OF SCOPE

- Images, staging, geometry, regions — Phase 2.
- The 25-image consensus set — Phase 9.
- Any modelling use of the basal view. Phase 1 records only that it exists and which image it is.
- Synthesis, TSTR — Phase 5.

---

## 6. TRAPS

- **Do not build the manifest from the odd/even rule.** Lookup first, rule as cross-check.
- **Do not silently drop the 14 photo-less IDs.**
- **Do not tune the 3-class thresholds.** 2.5 and 3.5, fixed.
- **The ceiling is 0.903, not 0.807.** They are different quantities — reliability and its square
  root. Encode both under distinct names.
- **Do not assume the orthodontist is the best rater or the best target.** Measurement says
  otherwise on both.
- **Do not attempt to read the clinical data.**
