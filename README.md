# cleft-aesthetics

Automated aesthetic assessment of surgical outcome in cleft lip and palate
repair — and a record of what could and could not be shown on a 237-patient
cohort.

The short version: **no model tried here predicts panel-assigned aesthetic
grade well, and the reason is measured rather than guessed.** The best arm
reaches a Pearson correlation of **0.2520** against a five-rater panel mean
whose own reliability caps any predictor somewhere in **[0.8983, 0.9065]**.
The gap is not the ceiling. Twenty-seven phases of work narrowed where it
might be, and this repository is the audit trail.

---

## What is here

A research codebase built around one constraint: **the cohort is small and the
labels are noisy, so most differences between models are not resolvable.**

**The panel is five raters from five different backgrounds** — a cleft patient,
an orthodontist, a speech and language therapist, a plastic surgeon and a
psychologist (`data/scoresheet.py` `RATERS`). It is not five clinicians and not
five surgeons: one member is a cleft patient, which is a deliberate feature of
the panel's design and not an approximation to be rounded away. Each rater gave
one composite grade per patient on a five-point ordinal scale (1 excellent → 5
very poor). They agreed weakly: **mean pairwise inter-rater Pearson r = 0.4696**
across 237 patients.

Almost every design decision follows from that. Comparisons are pre-registered
before the numbers exist; results are judged against a two-condition criterion
that most comparisons fail; and failures are recorded as failures rather than
re-analysed until they pass.

## What it found

**Nothing moved the number, and that is the result.**

**The feature-source axis, in measurements.** The unit below is one *measurement*
— one recorded comparison of an alternative feature source against the 0.2520
baseline, as enumerated in `phase25.THE_ELEVEN_AND_THE_SHARPEST`:

| feature source varied | measurements | outcome |
|---|---|---|
| Pretraining dataset (ImageNet / SCUT / MEBeauty, whole and masked) | 5 | best case parity (+0.0017) |
| Architecture (ViT patch 32 / 8, MViTv2, Swin) | 4 | parity (0.2519) to collapse (0.0432) |
| Input resolution (512 and 768 against 224) | 1 | a cliff outside 224 (−0.1583) |
| Pretraining objective (DINO, DINOv2 against supervised) | 2 | both at or below the probe |
| **strict feature-source total** | **12** | **none beat the baseline claimably** |

**Two counts exist and the record keeps both, deliberately**
(`phase25.THE_AXIS_CLOSES_DERIVED`). **Twelve** is the strict count above.
**Thirteen** is the count the project's registered sentence uses, because it
also counts one measurement on the adjacent *training-objective* axis, which
the record labels "not feature SOURCE, but the same lesson on the adjacent
axis". The difference is one measurement and one convention, not a disagreement
about any figure. Twelve is used throughout this README.

**Two further axes were varied, and neither is a feature source:**

| axis varied | unit | outcome |
|---|---|---|
| Training objective (regression / ranking / distribution) | Phase 22, **15 contrasts** | bounded ranking matches the probe to within 0.003; nothing better |
| Label construction | Stage G: **4 alternatives** (median and distribution, at two operating points); the rater screen: **5 single-rater targets** | no label alternative beats the mean anywhere; the rater screen is MIXED |

The best alternative feature source in the whole project is **parity**: MEBeauty
at +0.0017 and ViT-B/32 at −0.0001 against the baseline's 0.2520.

**Other findings worth naming:**

- **Rater-specific modelling does not beat the panel mean, measured twice.**
  Training one model per rater scores −0.0098 (the cleft patient) to 0.2703
  (the plastic surgeon) against the panel mean's 0.2520. The record's reading
  is **MIXED**, not "worse": four of the five are below, and the fifth is
  0.0183 above — 1.24 of the arm's own seed sd, inside the band that would
  have refuted the prior. Phase 1 predicted the ordering from label
  learnability before any model ran (mean 0.6022 against the single-rater
  target's 0.4944, the lowest entry in that table).
- **Hard cases are not the disputed ones.** Per-patient model difficulty does
  not track inter-rater disagreement (τ_b = **+0.0727** on residual sign and
  **−0.0811** on the worst quartile, both insignificant, opposite signs), which
  refutes the most attractive explanation for the ceiling.
- **The unresolved results are genuinely uncertain, not quietly equivalent.**
  A Bayesian ROPE analysis over **45 contrasts** found **0 equivalent, 4
  practically different, 41 no decision**: not one contrast is demonstrably the
  same, and the highest posterior mass inside the equivalence region is
  **0.3452**.
- **Systematic rater offset exists and is small** — the first measurement of
  rater *bias* in the project, since every earlier rater statistic (item-total
  correlation, mean inter-rater r, quadratic-weighted kappa) is invariant to an
  additive shift and could not have detected one.

**About the ceiling.** It is a range because reliability is, and the record
reports all three derivations rather than picking the flattering one
(`record_audit.THE_PRIMARY_REPORTING_FORM`): **ICC(2,k) = 0.8069** is primary
(two-way ANOVA, raters random, offset penalised) and gives the lower endpoint
√0.8069 = **0.8983**; **ICC(3,k) = 0.8218** (raters fixed, offset removed) gives
the upper endpoint √0.8218 = **0.9065**; and Spearman–Brown on the mean pairwise
r, **0.8158**, gives √0.8158 = **0.9032**, which sits between them because
Pearson r ignores offset rather than removing or penalising it. All three are
ceilings on what a predictor *could* reach, and 0.2520 sits far below every one
of them.

## How the record works

Two devices carry the discipline, and both are enforced by the test suite.

**Pre-registration.** Every phase writes its question, its arms, its exit
criteria and its *readings* — what each possible outcome would mean — before any
number exists. When a result arrives, the pre-written reading fires. A reading
written after the number is not a reading, and the record says so wherever it
happened.

**The results ledger** (`src/cleft/results_ledger.py`) is append-only with a
SHA-256 chain: each entry hashes the one before it, so an edit to any earlier
row invalidates every checksum after it, and the checksums are pinned in tests.
It holds **39 entries**: 11 `WITHDRAWN`, 9 `DESCRIPTIVE`, 8
`UNRESOLVED-WITHDRAWN`, 6 `VOID` — and **only 5 `CLAIMABLE`**. A claim is
recorded
as claimable only when it passes both conditions of the project's criterion:

1. every seed's paired bootstrap interval excludes zero in one direction, **and**
2. the effect exceeds the 95% uncertainty on the *difference of the two
   arms' seed means*, `1.96·√(sd²ₐ/nₐ + sd²ᵦ/nᵦ)`.

The second condition has to name which uncertainty it means: PLAN §4.3
records that "the combined seed uncertainty" admits two readings that
differ by about 2.5×, and the criterion is the narrower one above, not the
single-run band `1.96·√(sd²ₐ + sd²ᵦ)`.

Most comparisons fail condition 1. That is the cohort speaking, and the ledger
records it rather than working around it.

The chain has been re-derived once, on 2026-09-05, to remove personal names from
six rows before this repository was made public. That re-derivation is itself a
dated record (`results_ledger.CHAIN_RE_DERIVED`) carrying the superseded
checksums, a field-by-field diff showing no claim, status, figure, threshold or
ordering moved, and the honest consequence: **the chain now proves nothing has
been edited since that date, rather than nothing ever.** Prefixes through entry
18 were unaffected and still hash as they did when the ledger was born.

**Corrections are dated in place and originals are preserved.** Where a figure
was wrong, the record keeps the wrong version, marks it, and says how the error
arose — a mis-read, a value recalled rather than checked, a claim repeated
without verification. Those entries are deliberate: the failure modes are part
of the result.

## Layout

```
src/cleft/          phase records, the ledger, and the analysis code
  phase*.py         one module per phase: registrations, rulings, findings
  results_ledger.py the append-only claim ledger
  data/             manifest, score sheet, reliability, label construction
  train/            the frozen-backbone harness, heads, folds
  geometry/         staging, cropping, landmark geometry
  eval/             metrics
configs/            every run is a declared config; nothing is a flag
scripts/            config generators, each with a --check drift mode
tests/              3,000+ test functions, including the record's own pins
docs/PLAN.md        the governing document
```

**Precedence is code > `docs/PLAN.md` > anything else.** If the code and the
document disagree, the code is right and the document is fixed.

## Running it

```bash
pip install -e ".[test]"   # or, to run without testing:
                           #   pip install -e .
                           # or no install at all:
                           #   export PYTHONPATH="$PWD/src"
python -m cleft.run --config configs/smoke.yaml
```

`configs/smoke.yaml` runs end to end on synthetic data and needs no cohort. The
test suite runs the same way:

```bash
.\scripts\test.ps1        # Windows: builds the pinned 3.11 venv for you
pytest                    # elsewhere, after pip install -e ".[test]"
```

The `[test]` extra is what pins the numerical stack (numpy 1.26.4, scipy
1.13.1, scikit-learn 1.5.0). Installing without it and running `pytest`
fails `tests/test_environment.py` on purpose: a green suite in the wrong
environment is worse than a red one. Python 3.11 is required — numpy
1.26.4 publishes no wheels for 3.13.

**Nothing is a command-line flag.** Every run is a config file that declares its
inputs by path *and* by SHA-256 rollup hash; a run aborts before starting if a
declared hash does not match what is on disk. That is why the configs are
verbose, and why `scripts/generate_*.py --check` exists — a config that drifts
from its generator is a config nobody re-derived.

## The data boundary

**No patient data is in this repository, and none can be.** `data/` is
git-ignored; images, the manifest and the score sheet live only on the compute
cluster. Analysis code, configs, hashes and findings live here.

Practically, this means:

- Any run touching the cohort fails on an authoring machine with a missing
  declared input. That is the intended behaviour, not a setup problem.
- Figures in the record were produced on the cluster and transcribed here with
  their run directory named, so each is traceable to the run that produced it.
- Some records note that a check could not be performed locally. Those notes are
  accurate and deliberate.

**Filenames and column headers are the source data's own** and are preserved
verbatim — including personal names inside them. The score sheet's column
names are matched against the workbook at load time, so renaming a column
breaks the load outright.

Declared paths are a weaker case, and the record should say so rather than
overstate it: a path is the *location* a run resolves, and the rollup hash
covers the file's bytes — for a directory, the paths of the files inside it
relative to that directory. **The declared path itself never enters any
hash.** Renaming or parameterising one breaks the lookup, not the hash guard.

Line endings are likewise not normalised (`.gitattributes` sets `* -text`), for
the same reason: a rollup hash is a SHA-256 over bytes, and a checkout that
rewrote line endings would make the same artifact hash differently on Linux and
Windows.

## What this is not

It is not a deployable clinical tool, and the record is explicit about that. The
reliability ceiling is a bound on what any predictor could reach against this
target; **it says nothing about what a given model would reach**, and the
project's own evidence is that the distance to the ceiling — not the ceiling —
is the binding constraint.

Nor is the negative result a claim that the task is impossible. It is a claim
about what these methods achieved on this cohort at this label quality, with the
uncertainty stated.

## Reproducibility

Runs execute in a pinned container, referenced **by full digest, never by a
tag** — a tag can be repointed, and then two runs at the same git SHA are no
longer the same experiment.

```
redring/cleft-aesthetics@sha256:2135e27b5d82b28cb5e2059c606aadf5736df80e50cb4e52fc669cc8ba33d2ab
```

Python 3.11.13, torch 2.8.0+cu128, numpy 1.26.4, timm 1.0.7. Base image
`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385`.

A hygiene test in the suite fails the build if any file references a `:latest`
tag, if the base image is referenced by tag rather than digest, if a
requirement floats, or if a digest is abbreviated. It is the cheapest test here
and it guards the most expensive failure this project has had: a queued job
picking up different code than the one that was tested.

Every run also writes its own provenance — git SHA, dirty flag, image digest,
Python version, GPU, package versions, host — beside its outputs, so a result
carries the environment that produced it.
