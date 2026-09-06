# Known-stale content inside frozen modules

**Status: live register. Additive — this file never changes a frozen module.**

The measurement apparatus was frozen at `phase3-freeze` (`eef8f56`) and
`tests/test_frozen_apparatus.py` fails if any of it moves. That guard has a
consequence worth naming: **a comment inside a frozen module cannot be corrected
in place**, because correcting it is byte-identical, to the guard, with changing
the code.

That is the rule working, not a flaw in it. Re-running Phase 3 to fix a docstring
would be an absurd trade, and the escalation rule working correctly means *not*
escalating for this. But a reader who lands on the stale text needs somewhere to
go, so it lands here.

**How to use this file.** If you are reading a frozen module and something looks
wrong, check here first. If it is listed, the correction is below and the
governing record is named. If it is not listed, it is either right or newly
discovered — add it.

**What does not belong here.** Anything that changes a *number*. This register is
for prose that misdescribes correct behaviour. A frozen module computing the wrong
value is an escalation, not an entry.

---

## 1. `src/cleft/eval/metrics.py:244` — the Nadeau–Bengio reference

**The text, in `paired_delta_bca`'s docstring:**

> The delta is claimed only if the interval excludes 0. For CV-based comparisons
> the Nadeau-Bengio correction is applied on top of this; that belongs with the
> fold machinery, not here.

**Why it is wrong.** The correction is not applied anywhere, was never
implemented, and **should not be**. NB corrects a *t-test over fold-level
estimates*, where overlapping training sets across folds inflate apparent
precision. This project does not run that test: it pools out-of-fold predictions
into one 237-vector per arm and bootstraps over patients. There is no set of
fold-level estimates to correct.

The sentence therefore describes a step that does not exist and would be a
category error if it did — a statistic imported for the right-sounding reason and
attached to the wrong quantity. It is the third instance of the pattern R2 exists
for, after QWK-as-ceiling and the 0.903 "correction".

**The correction, and where it is recorded.** PLAN §4.3. The claim criterion is:

1. paired BCa CI over patients (10k) excluding 0, **and**
2. the delta exceeds the combined seed uncertainty of the two arms compared.

**How it was found.** Phase 4's exit criteria required a paired BCa delta for
every arm, which turned "the plan says NB applies" into something that had to
actually exist. It had survived in the governing document and in this docstring
until something depended on it.

**Recorded:** 2026-07-28.

---

## 2. `src/cleft/train/determinism.py` — two DataLoader helpers that nothing can call

**The functions:** `worker_init_fn` (line 78) and `make_generator` (line 91).

**Why they are stale.** Their only possible use is being passed to
`torch.utils.data.DataLoader` as `worker_init_fn=` and `generator=`. **There is no
`DataLoader` anywhere in the repo** — `torch_backbone.py` hand-rolls batching in
`_batches`. Neither function has a single call site in `src/` or `tests/`.

`worker_init_fn`'s docstring warns that without it *"workers share the parent
seed and their augmentation streams are correlated but not reproducible across
runs with different worker counts"*. That hazard is real for the code path it
describes, and **that code path was never built**, so the warning describes a risk
the project does not currently run.

**Why it is not a correction to make.** The behaviour is right — gate 1 passes on
`configure()`, which *is* called (`phase3.py`), and the seeding and cuDNN flags it
sets are what determinism actually rests on. Nothing computes a wrong number.
Removing dead helpers from a frozen module would invalidate every result measured
with it, for tidiness.

**What to do instead.** If Phase 6's pretraining loop introduces a `DataLoader` —
which is the likely moment, since it is the first arm with real batching — wire
these two in *then*, and note that doing so is a change to a frozen module and
therefore an escalation. Until then they are inert.

**A related observation, in an unfrozen file, recorded here because it was found
with them:** `torch_backbone.py`'s `_batches` seeds its permutation from
`self.seed`, which is fixed on the instance, so every epoch draws the *same*
shuffle rather than a fresh one. That affects only the `trainable: full` path,
which no current arm runs. It is not frozen, so it can be fixed when that path is
next used — but it should be fixed deliberately, because changing it changes
results for any arm that uses it.

**Recorded:** 2026-07-29.

**How it was found.** A reachability audit looking for registered-but-unreachable
code paths, prompted by `feature_relevance` shipping with no config. Both carry
`# pragma: no cover - needs torch`, which makes their zero coverage read as an
environment limitation rather than an absent call site.

---

## 3. `src/cleft/geometry/trapezium.py` — "the corners cease to exist" under G2, and `unwarp`'s stated precondition

**The text.** The module docstring: *"**G2** resamples to a canonical trapezium and
then stretches each row to full width, so the corners cease to exist."* And
`unwarp`'s docstring: *"The image is assumed already resampled to the canonical
trapezium."*

**Why it is wrong.** The frozen caller does not satisfy that assumption.
`stage_build.py:77-80` — also frozen, and the code that produced `staged_v1` —
runs `stage(image)` (pad-square-then-resize, white pad) and then
`unwarp(base.image)` on the **padded square**. At any aspect ratio other than 1
the staged square is not the canonical trapezium: the baked-in trapezium is
normalised to the **crop content**, while `unwarp` samples per the trapezium
normalised to the **full square**. The pad and part of the baked corners
therefore lie inside the sampled span and are stretched **into** G2.

**Measured (2026-07-30), by replaying the frozen composition on synthetic
crops with the canonical baked trapezium:** G2 white ≈ the pad fraction —
0.0026 at AR 1.0, 0.259 at the cohort median 0.7404, 0.446 at AR 0.5531. The
corners cease to exist *as corners*; their pixels do not leave the image.
`staged_v1`'s G2 arrays were built this way, and every result measured on them
carries this property uniformly.

**What made it look true.** `p2-stage-1` reports `white_fraction_max: 0.0` for
`grid_g2` / `anatomy_g2` / `random_g2`. That number is `patches.summarise`:
`1 − min(patch coverage)`, coverage measured **geometrically** against the
generating mask — and at G2 the generating mask is `contact.G2_TRAPEZIUM`, the
full square, so every coverage is 1.0 and the figure is 0.0 **by construction**.
No pixel is consulted. It is a patch-box quantity that cannot fail, misread in
Phase 5 as a pixel fact about the arrays — which produced the (now-removed)
"G2 must have zero white" invariant and the wrongful retraction of the
"matched padding" conclusion in `scut/masked.py`. R2: right number, wrong
quantity.

**Why it is not a correction to make.** No recorded number is wrong —
`white_fraction_max: 0.0` is correct *for its quantity*, and no pixel-white
figure for the G2 arrays was ever recorded anywhere. The composition is the
same for every patient and both domains (cleft and masked SCUT now match it
bit-for-bit), so nothing measured on `staged_v1` is invalidated. Changing the
frozen path to deliver the docstring's intention would be an escalation that
invalidates every G2 result for prose.

**The correction, and where it is recorded.** The governing record is
`scut/masked.py::G2_WHITE_DEFECT` (status RESOLVED, with the measured table)
and the parity expectations in `masked.parity_report`: G1 white ≈
`1 − 0.802·(1 − pad_fraction)`, G2 white ≈ `pad_fraction`. The masked-SCUT
build reproduces the frozen composition exactly — arrival mask baked
content-relative, then `stage`, then `unwarp` — asserted bit-identical by
`test_build_one_is_the_frozen_cleft_composition_exactly`.

**A second stale sentence found with it, same mechanism.** `resize_nearest`'s
floor-based index mapping is left-biased by half a source pixel, so staged and
unwarped arrays carry a small left/right white asymmetry (≤ 0.9% per
half-frame on the cohort ARs) even at even pad splits and at AR 1.0. Nothing
in the frozen prose claims otherwise, but Phase 5 briefly attributed the gap
to odd pad splits; the measured cause is the resize. Matched behaviour across
domains, recorded in `G2_WHITE_DEFECT["second_defect_resolution"]`.

**Recorded:** 2026-07-30.

---

## 4. `src/cleft/train/determinism.py:23` — "also set in the Dockerfile"

**The text, above `CUBLAS_WORKSPACE_CONFIG`:**

> cuBLAS needs this set BEFORE the CUDA context is created, so it is exported
> here and also set in the Dockerfile

**Why it is wrong.** The Dockerfile sets no such variable — `grep CUBLAS
docker/` finds nothing, and never has (the claim predates the freeze). The
constraint the comment states is real and the module's own export satisfies it
for every caller that goes through `configure()`; the Dockerfile half of the
sentence describes a belt that does not exist.

**How it was found [MEASURED 2026-07-31, in the image].** Two new call sites —
`train/pretrain.py`'s deterministic mode and
`scripts/verify_train_determinism.py` — set
`torch.use_deterministic_algorithms(True)` directly instead of calling
`configure()`, implicitly trusting that the environment variable was already
set "in the Dockerfile". Every backbone then raised *"not deterministic
because it uses CuBLAS"* in the image. Had the Dockerfile claim been true,
the bypass would have worked by accident; because it was false, the bypass
failed loudly — the stale comment and the duplicated-setup defect exposed
each other.

**The correction, and where it is recorded.** `configure()` is the ONE
deterministic setup path and every flag-enabling call site now routes through
it (`pretrain.py`, `verify_train_determinism.py`; `phase3.py` always did).
The duplicated setup is the same class as the two `is_absolute_path`
predicates and the `output_rows`/`step` pair, and is recorded as such in both
fixed files.

**Recorded:** 2026-07-31.

---

## 5. Nothing else, yet

The other corrections made at and after the freeze all landed in modules that are
**not** frozen, so they were fixed in place rather than registered here:

| corrected | where | how |
|---|---|---|
| "seeds differ in head initialisation and nothing else" | `train/phase3.py`, `train/torch_backbone.py`, `configs/p3_train_cv.yaml`, PLAN §4.12 | fixed in place — the band is initialisation **and** inner-val split variance |
| `trainable_parameters` reporting the frozen extraction pass | `train/phase3.py`, `train/torch_backbone.py` | fixed in place, before the freeze |
| the backwards learning-rate comment | `configs/p3_train_cv.yaml` | fixed in place, before the freeze |
| `lateral_orbit` coverage quoted at 87% | PLAN §4.4 | fixed in place — 76.4% at the frozen `scale 1.80`; `generators.ACCEPTED_ANATOMY` had the right value all along |
| determinism gated on `backbone != "stub"` | `train/phase3.py` | fixed in place — a positive `TORCH_BACKBONES` list |

That the list of *registered* staleness is four entries long while everything
else has been fixed in place is the intended ratio. The frozen set is
deliberately small, and most of the code is not in it.
