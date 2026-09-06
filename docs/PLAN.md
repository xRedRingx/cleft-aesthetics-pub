# CLEFT AESTHETIC ASSESSMENT — WORKFLOW AND PHASE PLAN v3

**Written:** 2026-07-27 (supersedes v2, same day; v1 of 2026-07-26)
**Amended:** 2026-07-28 at the Phase 3 freeze — not a version bump, additions only.
**Status:** governing document. The anchor to return to when things get confusing.

**Rule of precedence:** code > this document > memory.

**What the 2026-07-28 amendment added.** Mostly [MEASURED], from the three Phase 3 runs: the seed
band and the claimable-delta rule (§4.12, new); the trainable-parameter policy and its real count
(§4.7); why `beats_constant` is a calibration check and not the primary one (§4.3); `JOB_UUID`
surviving a pause, and the fact that **training does not** (§2.7); Phase 3's gate table and the
freeze, and Phase 4's shape (Part 5).

**Two things it retracts** — both wrong, both caught by Phase 4 asking what they meant in practice:

- the **Nadeau–Bengio** requirement in the claim criterion (§4.3). It corrects a test this design
  does not run. Replaced by the combined-seed-uncertainty condition, and the fold-assignment gap it
  was masking is now measured in Phase 4.
- the **87%** `lateral_orbit` figure in §4.4, which was the pre-sweep value at `scale 1.30`. The
  frozen scheme is at `scale 1.80`, where it is 76.4%.

---

## PROVENANCE TAGGING — read this first

| tag | meaning |
|---|---|
| **[MEASURED]** | computed from this project's data, reproducible on demand |
| **[LITERATURE]** | established in published work, citation given |
| **[REASONED]** | inferred, not verified. **May be wrong.** |

**A [REASONED] item must not gate a phase** until checked or explicitly accepted as an assumption.

### What changed from v2

**§4.5 region scheme, corrected.** the supervision material's "27 regions" is a multi-scale **patch generation
procedure**, not a list of anatomical structures. this was read backwards. The multi-scale grid is
now the primary design; the anatomy-anchored regions become an ablation arm. Q-c is resolved.

The misreading, recorded so it isn't repeated: supervision wrote *"Split the image in 3x3 grid, and extract
different patches with different aspect ratios or size, it can be horizontal or vertical or
different combinations and all together you get 27."* That is a construction with an arithmetic
result. the record took the anatomical examples given at supervision, eye corners, nose, mouth and upper lip, as the
specification, and treated the grid description as loose. It is the reverse: the anatomy terms name
what the generated patches must **cover**, and the grid is the specification.

Also: the mirror-pair rationale is weakened (§4.5), Phase 2's entry gate is replaced by a design
task (§5), and the arm list is adjusted (§6).

---

# PART 1 — WHY THE LAST ATTEMPT FAILED [MEASURED]

Three causal chains, all workflow.

**Chain 1 — slow builds destroyed provenance.** The image baked the code in (`COPY . .`; CI on
`'**.py'`), so pushes rebuilt gigabytes, so commits were avoided, so runs went out dirty.
`dirty_tree: true` on every run ever produced — so when the same arm gave 0.218 at one SHA and
0.150 at another, code drift could never be separated from non-determinism.

**Chain 2 — mutable data directories destroyed lineage.** The SCUT mask was rebuilt in place at the
same path. "Which mask made this checkpoint?" has no answer in the artifacts.

**Chain 3 — flag-based commands drifted.** The command that produced a result is not recoverable
from the result.

**Phase 0 closed all three.**

---

# PART 2 — THE WORKFLOW [MEASURED — verified end to end 2026-07-27]

## 2.1 Separation

| layer | contains | changes | lives |
|---|---|---|---|
| image | python, torch, timm, deps | ~never | DockerHub, pinned by digest |
| code | everything we write | many times a day | NFS, via git |
| data | images, manifests, checkpoints | append-only | NFS, versioned dirs |
| runs | outputs | append-only | NFS, immutable |

Pinned: `redring/cleft-aesthetics@sha256:2135e27b5d82b28cb5e2059c606aadf5736df80e50cb4e52fc669cc8ba33d2ab`
Base: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime@sha256:417bd75d…`
Python **3.11.13**, torch **2.8.0+cu128**, numpy **1.26.4**, timm **1.0.7**, openpyxl **3.1.5**.

## 2.2 The loop

`edit → commit → push → CI tests (~40s) → git pull on NFS → launch job`. No image rebuild.

## 2.3 Configs, never flags

`python -m cleft.run --config configs/<name>.yaml`. Two CLI args only. The config **is** the
provenance record.

## 2.4 Run directory contract

```
runs/{keeper|dev}/<phase>/<config>__<sha8>__<job-id>/
  config.yaml  env.json  inputs.json  metrics.json  curves.csv
  predictions.csv [CLUSTER-ONLY]  code/ (worktree at the SHA)  log.txt
```

## 2.5 Guards — mechanical

1. Dirty tree → forced to `runs/dev/`; keeper aborts.
2. Git unavailable → aborts at **every** tier. Unknown provenance is worse than dirty.
3. Run directory exists → abort, unless the ID came from a job ID, then it is a resume check.
4. Input hash mismatch → abort.

## 2.6 Immutable data artifacts

`data/<name>/v<N>/` + `MANIFEST.json` (per-file hashes, rollup, generating config, SHA). Create
`v5`; never modify `v4`.

## 2.7 Cluster specifics [MEASURED]

- The pinned image has **no Jupyter** — it cannot back a workspace. All citable runs are **training
  workloads**.
- Runs as **root, `HOME=/root`**, no SSH credentials. **Jobs must not run git operations.** Code
  reaches NFS via the workspace; the entrypoint *verifies* a SHA is present rather than fetching it.
- Set `GIT_CONFIG_GLOBAL=/home/user/codex/.ssh-cleft/gitconfig` (NFS, survives container recreation).
- Use `PYTHONPATH=src`, not `pip install -e .`.
- Workspace containers are ephemeral; credentials live on NFS at `/home/user/codex/.ssh-cleft/`.
- Run:AI resumes by recreating the pod; run IDs derive from the job ID. **[MEASURED 2026-07-28]**
  `JOB_UUID` **survives a pause.** The gate-2 sweep was paused mid-run and resumed; `JOB_UUID` was
  `8e7c7452-568b-4c15-b4a9-cd0809572ca7` at both attempts, the new pod mapped back to the same run
  directory, and `resumes.json` carries attempt 1 with `resumed: true`. `CLEFT_JOB_ID` is now
  belt-and-braces rather than load-bearing. `docs/VERIFY_resume.md` is closed and Phase 6 is
  unblocked.
- **[MEASURED 2026-07-28] Run identity survives a pause; training does not.** The same resume redid
  all ten seeds in the second pod and nothing noticed, because the sweep takes a minute. A Phase 6
  job paused at hour three would silently restart at hour zero with **no symptom but wall time** —
  the resume mechanism reattaches the directory, not the work. Checkpointing is a Phase 5/6
  prerequisite, not a Phase 7 nicety.
- Compute: 5 × RTX PRO 6000 Blackwell. Fractional allocation is recorded in `env.json`.
- **[MEASURED 2026-07-28]** A ten-seed sweep of the frozen linear probe ran in **about a minute on a
  0.16 GPU fraction**. Frozen-backbone arms are effectively free, which changes what is worth
  sweeping: seed counts should be chosen from the band in §4.12, not from a compute budget.

---

# PART 3 — GOVERNING RULES

**R1** Measure before asserting.
**R2** When a number looks wrong, check whether it is the **same quantity** before calling it an
error. Both major mistakes here — QWK-as-ceiling, and the 0.903 "correction" — were quantity
confusions. So were Nadeau–Bengio (§4.3) and the seed band's name (§4.12).

> **Worked example, and the clearest instance so far — segmentation `separability`, 2026-07-28.**
> The number was **real, computed correctly, and measured a different contrast than its name
> implied.** It was Otsu's η on the lip discriminant over the whole staged frame, designed as the
> *strongest* signal in the diagnostic: "if this is low, the discriminant is not bimodal and no
> threshold could work". It scored **0.888** on real crops and was read as evidence.
>
> But the staged crop is padded white, so the dominant contrast in the frame is **face against
> background**, and that is what 0.888 measured. On synthetic fixtures a face with **no lips at all**
> scores **0.992** — *higher* than one with lips at 0.806, because lip pixels muddy the skin-versus-pad
> split. So the value was not merely uninformative, it ran **backwards**: 0.888 was
> **anti-informative**, and it was read as supporting a conclusion it could not speak to.
>
> Nothing was wrong with Otsu, the implementation, or the data. The name promised lip-versus-skin
> and the computation delivered face-versus-background, and no amount of checking the arithmetic
> would have found it. What found it was asking *which contrast is this actually measuring* — R2's
> question — and the fix was to compute it inside the face mask and **report both side by side**, so
> the correction is visible in the artifact rather than asserted in prose.
>
> The general lesson, worth carrying into every metric this project reports: **a number can be
> correct and still answer the wrong question, and its name is not evidence about which question it
> answers.**
> **[MEASURED 2026-07-31] Seventh instance of a quantity compared against the wrong reference, and the
> THIRD where the check rather than the code was the defect.** The APPNP parity script compared a
> **float32** absolute difference against a `1e-10` constant — a threshold only meaningful for
> float64 — and printed `DIFFERS` on agreement that was exact to machine precision. The per-case
> checks were dtype-aware and all read OK; only the summary was wrong. It measures **ULPs** now, and
> the answer is 0.92 ULP worst case.
>
> The running split is worth keeping visible: of the seven, three were the *check* and not the thing
> checked — the `separability` metric, the zero-white G2 invariant, and this. A wrong check is more
> expensive than a wrong result, because it spends the time of whoever chases the phantom.
>
> **[MEASURED 2026-08-03] Eighth instance, and the FOURTH where the check was the defect — Phase
> 7C's `moved_outside`.** The augmentation contact sheet reports whether augmentation moves protected
> pixels less than unprotected ones. It averaged the change over *every* unprotected pixel, and a
> staged crop is pad-square-then-resize with a **white** pad — 0.2534 of the frame on average
> across the cohort (min 0.0179, max 0.4464; §4.4). *[CORRECTED 2026-09-05: this read "at the
> median aspect ratio". 0.2534 is banked as the cohort MEAN
> (`geometry.staging.PAD_FRACTION_MEAN`); no median pad fraction is banked anywhere. The
> argument below is unaffected.]* Flat white barely moves under photometric jitter and skin moves a lot, so the
> outside average was diluted by pixels that *cannot* move, and the check reported protection
> **inverted on 7 of 12 patients whose masks were correct**.
>
> The name is the whole error, and it is R2's exact shape: **`moved_outside` is named for the
> policy's effect and computes the average change over a region defined by geometry.** Those are
> different quantities, and the name is not evidence about which one it is.
>
> **The tell was patient 5** — 47.3 inside against 47.5 outside, nearly equal where the others looked
> inverted. That is the darker-skinned patient from the Phase 4 fairness work, whose crop carries the
> least white margin relative to face: least pad, least dilution. The same patient who exposed the
> fairness limitation exposed this one, for the same underlying reason.
>
> Both figures are reported now — content-restricted and whole-frame side by side — rather than the
> diluted one being removed, which is the treatment `separability` got once it turned out to be
> measuring face-against-background. `train/augment_sheet.MOVED_OUTSIDE_WAS_DILUTED_BY_THE_PAD`.
>
> **[PROCESS] Its failure mode was a false alarm rather than a silent pass**, which is the safer
> direction for a diagnostic and is worth distinguishing in this tally: instances 1–5 reported
> success on their own failure mode, and this reported failure on a success. It still cost a review
> round, and a check that cries wolf is a check people learn to overrule — which is how the *next*
> one gets ignored when it is right.

**R3** Assert invariants at runtime.
**R4** Tests encode verified facts, written before the code they guard.
**R5** Freeze before the ladder.
**R6** One change at a time.
**R7** A verifier passing means only that the thing it tests is right.

> **[STANDING RULE, 2026-07-29] Suite runtime is advisory, not a criterion.**
> The 60-second target is a comfort, not a constraint. **Correctness over speed.**
> Tests are never deleted, shortened or weakened to save time. If CI is slow, that is
> acceptable; if the suite grows because coverage grows, that is *correct*.
>
> A coverage reduction is reversible only if it was **genuinely redundant** — meaning
> the assertion is proven elsewhere, **not** that the test was slow.
>
> And the apparatus is never unfrozen for speed. Invalidating the measurement
> apparatus to make a test run faster inverts the freeze's entire purpose.
>
> **[MEASURED] The runtime target was also unmeasurable.** Four back-to-back runs of
> identical code on the laptop: 57.5, 61.1, 89.9, 69.3 seconds. The error bar is larger
> than the target, so "is it under 60?" was never a question this machine could answer —
> and the effort spent trying to answer it produced the defect below.

> **Worked example — a check keyed to one exact string, 2026-07-28.** The
> parameter/policy check was written after `trainable_parameters` reported 85,798,656 on a 769-
> parameter head. It keyed off `report["features"] == "frozen_backbone_embeddings"`. Phase 4's patch
> arms report `frozen_backbone_patch_embeddings`, so **every patch arm silently skipped the check
> that existed because that exact reporting had been wrong once already.** It is a set now.
>
> **This is the fourth instance of the same pattern** — after QWK-as-ceiling, the 0.903
> "correction", Nadeau–Bengio, and `separability`. A check, like a metric, can be correct and still
> not cover what its name implies it covers.
>
> **[PROCESS] It was found by building a new arm, not by review.** Nobody reading that line would
> have noticed; it took an arm whose feature kind differed. That is an argument for **adding arms
> early and one at a time** rather than in bulk at Phase 7: each new arm exercises the shared path
> from a slightly different angle, and the cheap ones find what review does not. Phase 4 has now
> done this twice — the patch arm found this, and the mirror arm found that determinism was gated on
> `backbone != "stub"`.

> **Worked example — a test that passes on no data, 2026-07-29. THE FIFTH INSTANCE of a check
> reporting success on its own failure mode.**
>
> Two fairness tests had their patient counts cut from four to two while chasing the 60-second
> target. `pass_rate_by_brightness` returns `{"insufficient": True}` with **no quartiles at all**
> below four patients, so `for quartile in fairness["quartiles"].values():` iterated an empty dict.
> Both tests **passed while asserting nothing**, and the suite went green.
>
> The running tally of this pattern:
>
> | # | the check | how it reported success on its own failure |
> |---|---|---|
> | 1 | `separability` | measured face-versus-background; a face with **no lips** scored *higher* than one with them |
> | 2 | the spatial prior | made `centroid_y` and `area` satisfiable **by construction** — band-filling passed every criterion |
> | 3 | the fissure detector | fired on every mask, and a halved mask satisfies `fraction_above_split` **by construction** |
> | 4 | the parameter/policy check | keyed to one exact feature string, so it silently skipped every patch arm |
> | 5 | the fairness tests | iterated an empty dict and asserted nothing |
>
> **[MEASURED 2026-08-03] Tenth instance — Phase 7C's augmentation arms, and the first that was
> invisible in the results themselves.**
>
> All seven arms early-stopped at **epoch 1**. A 769-parameter head over frozen embeddings converges
> immediately, inner-val never improves again, and patience terminates on the first epoch — the same
> signature as Phase 6's scheme runs before the gate-4 amendment. So the model saw each training
> image **once, under a single random transform**, which is not augmentation: augmentation works by
> accumulating many transformed views so the model learns what is invariant across them, and one view
> is not a sample of a distribution.
>
> **What makes this the sharpest instance so far is that nothing looked wrong.** The seven arms
> produced a plausible, monotone ordering — identity best, mild photometric next, heavy geometric
> worst — which is exactly what *"how much does one random distortion degrade a one-epoch fit"* looks
> like. Every previous instance had a tell in a quantity somebody was reading: a metric that ran
> backwards, a delta that flipped sign, seven arms returning byte-identical predictions. This one had
> its tell in `selected_epochs`, a bookkeeping field nobody compares between arms **because it is not
> the answer to anything**.
>
> The rule that follows: **when an experiment varies a training-time regulariser, the epoch counts
> are part of the result, not part of the log.** An arm that stops at epoch 1 has not run the
> procedure its config names, whatever number it reports.
>
> The fix needed no harness change. `run_fold` already keeps the best epoch's predictions and uses
> `patience` only to break, so `patience >= max_epochs` makes the break unreachable and leaves a
> fixed budget with best-checkpoint selection — gate 4's amendment through config alone.
> `phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION`.
>
> **[MEASURED 2026-08-03] Eleventh instance, and a new shape — a "second round" that was
> two-thirds the first round's arithmetic.** Phase 7C ran seven augmentation arms at a
> pre-registered 30-epoch budget, then re-ran them at 3 epochs to remove a selection confound. The
> second round was about to be nominated the primary reported result, because it changed a verdict.
>
> It is not a second measurement. The augmentation view stream is `rng_for(seed, fold, epoch)` —
> derived from the coordinates alone, deliberately, so a resumed run reproduces its views with no RNG
> state to save — and the head trains at a constant rate from a fixed seed with no schedule keyed to
> the budget. **So epoch *n*'s fit does not depend on `max_epochs`**, and the short round is the long
> round's first three epochs. It changes exactly one thing: the candidate set the best-checkpoint
> argmin runs over. **23 of the 35 fits are the same fit** — every fold where the long round already
> selected an epoch ≤ 3 — so "the two rounds agree" was two-thirds arithmetic identity, and the
> corroboration it appeared to give did not exist.
>
> Worse, the truncation point came from *"the curves put divergence at about epoch 3"* — the long
> round's own curves, read after its verdicts were known. A data-dependent restriction of a
> model-selection candidate set, decided post hoc, which moved a verdict in the direction the
> analysis preferred. **The rule that follows: a re-analysis that shares fits with the analysis it
> corrects is a sensitivity analysis, and its selection window must be fixed before the first result
> is seen or it is a forking path.**
>
> What saved it was the phase's own pre-registration. `MATCHED_EPOCH_POLICY` had committed, before
> either round ran, to *"if they disagree, the disagreement is the result and the selection procedure
> is the factor."* They disagreed on one arm. The pre-registered output is that photometric
> augmentation is **unresolved**, not that the narrower window wins — which is the opposite of what
> the draft conclusion said, and the pre-registration is the only thing that made the difference
> visible. `phase7c.ROUND_2_IS_ROUND_1_TRUNCATED`.
>
> **[MEASURED] Twelfth instance, caught before it was quoted — the same phase's mechanism tell.**
> The claim underneath the verdict is *"if augmentation were regularising, the augmented arms'
> inner-val optimum would arrive LATER than identity's, and it does not."* In the 3-epoch round
> **"every optimum is at epoch 1–3" is true by construction** — the budget forbids anything else.
> Only the 30-epoch round can evidence it, and there it is a measurement: a late optimum was
> reachable and mostly did not happen (18 of 30 augmented folds chose epoch ≤ 3, against identity's
> 5 of 5 at epoch 1). Same shape as `white_fraction_max: 0.0` (§4.4), as `centroid_y` under a
> bottom-band prior, and as `fraction_above_fissure` on a halved mask — a criterion satisfied by the
> design rather than by the data. `phase7c.MECHANISM_NEEDS_THE_LONG_BUDGET`.
>
> *[CORRECTED 2026-09-05 — the paragraph above stands as written, and the claim it rests on
> does not. `phase7c.MECHANISM_NEEDS_THE_LONG_BUDGET["status"]` reads **"WITHDRAWN — see
> MECHANISM_CLAIM_WITHDRAWN"**, withdrawn 2026-08-03. The two counts are right — recomputing
> from `phase7c.STAGE_7C_RESULTS["rounds"]["selected30"]` gives identity 5/5 at epoch 1 and
> 18 of 30 augmented folds at epoch ≤ 3 — but the reading is not. `MECHANISM_CLAIM_WITHDRAWN`
> records the refutation: **augmented mean selected epoch 7.13 against identity's 1.0, and
> 18/30 folds LATER than epoch 1, 0/30 earlier.** The distribution did shift. The withdrawal
> names the mechanism exactly: *"it asserted 'most folds select epoch ≤ 3' (18 of 30, true)
> and read that as 'not later'. A majority being early is a different quantity from the
> distribution not shifting."* **This sentence repeats that misreading**, so it is withdrawn
> with the claim rather than repaired.]*
>
> **[PROCESS] Both were found by an adversarial review whose brief was to refute the closing
> position, not to check it.** The numbers had already been re-derived twice and the arithmetic was
> right both times; what was wrong was what the numbers were taken to *be*. Re-deriving a delta
> cannot catch a round that is not independent, and no amount of checking the verdicts would have
> found it — the question that found it was *"is this a second measurement?"*, which nobody asks of
> a result that agrees with the first one.
>
> **[MEASURED 2026-08-04] Thirteenth instance, and the most expensive — a claim criterion with two
> conditions, of which the phase computed one.**
>
> §4.3 claims a delta only if **both** hold: (1) the paired BCa CI over patients excludes zero, and
> (2) the delta exceeds combined seed uncertainty. Phase 7C formed all nine of its verdicts from
> condition 2 alone. Condition 1 was pre-registered here, promised again in
> `phase7c.NOT_A_SEARCH["carried_over"]`, and required by the phase's own exit criterion 6 — and was
> not computed until every verdict had been written up, defended, and twice corrected. **When it was
> finally computed it withdrew all nine.** Not one of the 45 per-seed intervals excludes zero. The
> data never changed; the phase had not applied its own rule.
>
> **This is not a check that passed on its own failure mode. It is a check that was never run, while
> a second check standing next to it made the verdicts look tested.** Every verdict carried a
> threshold, a delta and a `claimable: true` — the visible apparatus of a criterion being applied.
> The half that was missing left no gap in the output to notice.
>
> **The phase also predicted the wrong direction, in writing.**
> `OUTSTANDING_BEFORE_THE_NULLS_ARE_REPORTABLE` recorded the paired test as *"conservative; they
> survive"* for the negative verdicts and dangerous only for the nulls. Exactly backwards: the nulls
> were already null and stayed null; every negative fell. The reasoning — that an unpaired threshold
> overstates variance — is correct **about condition 2**, which was never the binding constraint on
> those six. Reasoning confidently about the direction of a test's effect is not a substitute for
> running a test that needs no new fits.
>
> The rule that follows: **a claim criterion with N conditions is not partially applicable.** A
> verdict computed from a subset of them is not a weaker verdict, it is not a verdict. Where a
> condition is deferred, the deferral belongs in the verdict field itself rather than in a
> neighbouring record, so the gap is visible wherever the claim is quoted.
> `phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT`.
>
> **[MEASURED 2026-08-04] The audit that follows from it: 43 claimable verdicts across `ladder.py`,
> `phase7b.py` and `phase7c.py`. Not one carries an interval.** The whole of Phase 7 has been
> reporting condition 2 as though it were the criterion — the Q1/Q2 verdicts, the geometry
> measurements, the label-formulation triples, and the headline comparison in `ladder.BEST_ARM`.
>
> **The condition-2 margin does not predict condition 1, and 7C proves it.** The tempting reading is
> that thin claims are exposed and comfortable ones are safe. 7C's arm 2 cleared its threshold by
> **3.62×** and still failed. Every claim in the audit sits at 5.51× or below. Interval width is set
> by patient-level disagreement between the two prediction vectors; the ratio of two seed-level
> summaries carries no information about it.
>
> The thinnest is the project's headline — `BEST_ARM.nearest_challenger` at **1.05×**. If condition 1
> fails there, *"0.2520 is claimably ahead of everything else"* becomes *"it is the highest-scoring
> arm, not claimably the highest"*. The PCC is untouched; the comparative sentence is not.
>
> **Uncomputed, not failed** — 7C's arms are augmented and need not generalise — and **checkable
> without a single new fit**, since every ladder arm is a `train_cv` keeper run that wrote
> `seed_<n>__predictions.csv`. `ladder.CLAIMS_REST_ON_HALF_THE_CRITERION`.
>
> **[MEASURED 2026-08-04] Fourteenth instance, found by reading the headline's own output — a
> field that reports agreement on an empty set.** `paired_comparison` computed `same_direction` as
> `len(directions) <= 1` over the *excluding* subset, so when no interval excluded zero the set was
> empty and the field returned `True`. The ladder headline came back `n_excluding_zero: 0,
> same_direction: true`, which reads as five seeds agreeing — while the per-seed deltas ran −0.0027
> to +0.1040 and **the sign flipped**. The verdict was never affected (`all_exclude` requires
> `len(directions) == 1`, which an empty set fails), so nothing was ever claimed on it; what it
> corrupted was the *reading*. It is now `None` when nothing excludes zero — "no seed excluded zero"
> is a different statement from "the excluding seeds disagreed" — with `same_direction_all_seeds`
> added for the question a reader actually has.
>
> **[OPEN] The same gap may sit in Phase 7B's headline.** `phase7b.SEARCH_AXIS_VERDICTS` records
> `delta`, `threshold`, `claimable` and `single_run_95` for winner-versus-baseline — all condition-2
> quantities, no interval. So *"the tuned arm is claimably WORSE"* is, in the record, the same shape
> of verdict 7C just lost nine of. **"Nothing beat 0.2520" does not depend on it** — that needs only
> that the winner is not claimably *better* — but the sharper sentence does. `task_phase7b_search`
> computes the paired comparison whenever the `baseline_oof_seed_<seed>` inputs are declared and the
> shipped config declares all five, so this is likely a read of an existing `metrics.json`, not a
> run. `phase7c.PHASE_7B_MAY_CARRY_THE_SAME_GAP`.
>
> **[PROCESS] Phase 7B was immune, and by accident.** Its fits are closed form — one solve, no
> iteration, no epoch to stop early from — so its configs carry no `max_epochs` or `patience` and its
> results carry no `selected_epochs`. That choice was made for an unrelated reason: `run_fold`
> computes the held-out fold's predictions as part of its job, so using it per trial would have meant
> computing 24 out-of-fold vectors and declining to read them (`phase7b.SEARCH_HEAD_PROCEDURE`). A
> decision taken to protect the selection discipline incidentally protected against a defect nobody
> had thought of. Worth recording because it is the inverse of everything else in this tally: usually
> a choice made for one good reason turns out to have a cost elsewhere.
>
> **[PROCESS] Suite optimisation is a common way to introduce checks that pass on empty input.**
> Shrinking a fixture, cutting a count, narrowing a range — each can drop the input below the
> threshold at which the code under test still produces something to assert against, and the
> resulting green is indistinguishable from a real one. **A test that can pass with no data is not a
> test.** The guard is to assert the input is non-empty *before* iterating it, which is now done at
> `test_symnose.py`'s fairness tests.
>
> That the defect was **caused by chasing a runtime target** is why runtime is now advisory (R7's
> standing rule above).
**R8** Nothing is added to a phase after its exit criteria are written.
**R9** When reading a supervision's instruction, separate the **procedure** from the **examples**.
The region misreading came from treating illustrative anatomy as the specification.
**R10** **A call to something that does not exist appears where no laptop test can execute the
line.** Not R7 — R7 is a check that passes on its own failure mode; this is the absence of any
check, on a surface that can be enumerated in advance.

> **[MEASURED] Three instances, one position.**
>
> | # | the call | where |
> |---|---|---|
> | 1 | `ctx.write_metrics` — `RunContext` has no such method | `task_phase7b_search`; died on the cluster after 24 trials |
> | 2 | `extractor.preprocess` — `FrozenExtractor` had neither that nor `model` | `gradcam.token_activations_and_gradients` |
> | 3 | `phase3.run_cv(..., keep_heads=True)` — wrong module, invented parameter, invented return field | `phase8_refit_heads` |
>
> All three sit on code that **needs torch and a GPU**, so the suite cannot reach them however
> thoroughly it is written. Nothing was wrong with the tests; the line was simply outside what any
> of them could run.
>
> **The surface is finite and countable.** [MEASURED 2026-08-04] 46 functions in `src/cleft` import
> torch or timm locally — 14 in `torch_backbone.py`, 7 in `graph_cleft.py`, 6 in `pretrain.py`, 4
> each in `extract.py` and `ldl.py`, the rest scattered. **The fourth instance will be in one of
> those 46**, and that is a useful thing to know before it happens rather than after.
>
> **[MEASURED 2026-08-08] The rule's value is now measured rather than argued: checking the API
> first changed the plan FOUR times in one session.** `phase3.run_cv(keep_heads=True)` did not exist,
> and the real path was `harness.run_cv(make_backbone=...)` with heads captured through the factory;
> `extractor.model` and `extractor.preprocess` did not exist; `load_manifest` refused `geometry.csv`
> where `phase3.load_geometry_rows` was the reader; and `stage_build.build` took no size argument
> while `staging.stage` already did — which shrank a planned new builder to six cells of frozen code
> at a different argument plus one new function. **Three of the four would have failed on the cluster
> after the compute.** The fourth would have been a large module written against an API that already
> existed in simpler form.
>
> **[MEASURED 2026-08-09] A fifth instance, and it widens the rule: the surface is not only
> torch-only code. It is any line whose correctness depends on state that exists on the cluster and
> not on the laptop.**
>
> `roadb_tasks.source_image` built `<root>/<frontal_id>.jpg`. The layout is
> `<root>/<patient_id>/<frontal_id>.jpg` — it skipped the patient folder. Both declared inputs
> resolved and hashed, so **guard 3 was satisfied and only the join was wrong**.
>
> **A fixture cannot catch this, and that is the point.** A test builds whatever layout it invents
> and then passes against it, so it confirms the code agrees with the fixture rather than with the
> data. The real layout only exists on the cluster.
>
> **The guard for this variant is not name-checking — it is refusing to write the resolver at all.**
> `geometry.contact.find_image` already existed, is what `stage_build` uses, and already carries what
> a fourth implementation would have had to rediscover: AppleDouble `._<name>.jpg` and `Thumbs.db`
> excluded **by basename rather than extension** (and `Thumbs.db` regenerates on Windows browsing, so
> it must stay out of input hash walks); folder 143's exception where 524 is the frontal, not 523;
> and folder 238's single image. It matches `str(image_id) in path.stem` rather than an exact
> filename, which is what tolerates all three without a per-patient special case.
>
> So the rule gains a second clause: **before writing anything that resolves a path, a name, or an
> id, look for the existing resolver.** Five for five this session, checking first would have saved a
> run.
>
> **The guard is static name-checking over exactly that surface**, which is the only kind of check
> that works on a line it cannot run: parse the source, collect every attribute accessed on an
> object whose class is importable, and assert it exists. `test_workflow_hygiene` does this for
> every `ctx.<name>` in every task; `test_gradcam.py` does it for `extractor.<name>`,
> `harness.<name>`, `backbone.<name>` and `TrainConfig`'s fields. **Each new torch-only call site
> needs one, and writing the code without it is what produced all three.**

---

# PART 4 — DECISIONS REGISTER

## 4.1 Data [MEASURED]

| fact | value |
|---|---|
| patient folders | **238 directories, all present**; folder **52 exists and is empty**, so **237** hold photographs (corrected 2026-07-27 — v3 and earlier said "52 absent") |
| filesystem clutter | every image has a 4,096-byte macOS AppleDouble sidecar `._<name>.jpg`; three folders also carry `Thumbs.db` and `._Thumbs.db`. 952 files total = 473 images + 473 sidecars + 3 + 3. Excluded **by basename**, since the sidecar has a `.jpg` extension and the image id in its stem |
| total images | 473 |
| frontal (all scored) | **237** |
| basal (unscored) | **236** |
| score rows | 251 = 237 + **14 with no photos** (716–742 even) |
| frontal rule | 1–171 odd; 172–237 even — holds 235/237 |
| **exception A** | folder **143** → **524** frontal, not 523 |
| **exception B** | folder **238** → single image **581**, frontal, no basal |
| raters | 5, single composite 1–5 grade each |
| encoding | Excellent=1 … Very poor=5, verified across 1,255 cells, zero missing |
| soft labels | `soft_1..soft_5` present |
| 25-set | 25 unique, 3/7/6/6/3, with `-lips` and `-nose` crops |
| un-masked originals | **do not exist** |
| laterality | **not recorded anywhere** |

**Manifest by score-sheet lookup, never by rule.** The rule is a cross-check that must flag exactly
143 and 238.

## 4.2 Label [MEASURED]

| target | learnability | decision |
|---|---|---|
| Reliability-weighted mean | 0.5926 | reported |
| **Mean of 5** | **0.5900** | **PRIMARY** |
| Median of 5 | 0.5561 | ordinal secondary (supervision approved) |
| Mode of 5 | 0.4880 | rejected |
| Orthodontist | 0.4811 | **rejected — worst target** |

the supervision material's premise that the orthodontist is most reliable is **not supported**: SLT 0.654 vs 0.628.
3-class collapse `{1,2}/{3}/{4,5}` at fixed 2.5/3.5.

## 4.3 Metrics and ceilings [MEASURED]

> ### [FINDING — MEASURED 2026-08-04] This cohort cannot resolve PCC differences of 0.04 to 0.10 between arms.
>
> **A result in its own right, not a caveat attached to each withdrawal.** Four independent
> comparisons have now been tested against this section's full criterion, and **all four failed
> condition 1**:
>
> | comparison | delta | condition-2 margin | intervals excluding 0 |
> |---|---|---|---|
> | Phase 7B, hyperparameter search | −0.0542 | 2.37× | 1 of 5 |
> | Phase 7C, augmentation (arm 2) | −0.0891 | 3.62× | 0 of 5 |
> | Phase 7C, all nine verdicts | to −0.1005 | to 3.62× | 0 of 45 |
> | The ladder headline | +0.0428 | 1.05× | 0 of 5 |
>
> The failures are not clustered in one method: a hyperparameter search, an augmentation family and
> a backbone-plus-init contrast are **three unrelated ways of changing a model**, and they give one
> answer.
>
> **[CONFIRMED AT SCALE 2026-08-04] The full ladder audit ran all 26 remaining pairs. One survived.**
> Across Q1 and Q2 at both geometries, both region schemes, the geometry contrasts, the search, the
> augmentation family and the headline — **29 of 30 comparisons fail condition 1**.
>
> The survivor is `Q1__swin_b` at 5 of 5, margin 3.83×, and it must not be quoted plainly: its
> +0.2027 is large because the **baseline is 0.0065** (`ladder.SWIN_G2_IMAGENET_CELL`), and **the
> same comparison at G1 gives +0.0114 at 0.44× with 0 of 5**. Both facts are recorded beside the
> claim itself rather than one entry away.
>
> **And the condition-2 margin does not predict condition 1 — measured across 26 pairs, not inferred
> from one.** The two *highest* margins both fail (4.69× at 2 of 5; 4.34× at 4 of 10) while the
> survivor ranks only third at 3.83×. Ranking claims by how comfortably they clear the seed threshold
> orders them by something that does not govern whether they hold. `ladder.LADDER_PAIRED_AUDIT`.
>
> **The mechanism is interval width, and it is a property of the cohort.** A per-seed paired BCa over
> the 237 is roughly **0.28 wide** at the headline. A 0.043 mean delta sits inside that many times
> over, so the two arms are indistinguishable on every seed taken individually — and at the headline
> **the sign flips between seeds**, which no summary statistic shows. **More seeds cannot fix it**:
> seeds narrow the estimate of a *mean*, condition 1 asks about *patients*, and it is a universal
> over the seeds present so more of them is a harder bar.
>
> **What this licenses.** Descriptive orderings — one arm *scored higher* than another, which is a
> fact about the measurements. **What it forbids:** any claim that an arm was shown to be better or
> worse. Every "claimably" sentence in the write-up becomes a "scored higher" sentence.
>
> **This is a stronger argument for the full-face data than any thin claimed win would have been.**
> A 1.05× victory would have been a fragile sentence inviting scepticism. *"We measured what this
> cohort can resolve, and it is coarser than every effect we set out to detect"* is an argument from
> evidence, and it generalises — it says what a next dataset has to be big enough to do.
> `ladder.COHORT_CANNOT_RESOLVE`.

**Reliability and a correlation ceiling are different quantities.** On the **237 with photographs**:

| quantity | value |
|---|---|
| mean inter-rater r | 0.4696 |
| Spearman-Brown reliability | 0.8158 |
| **PCC ceiling = √0.8158** | **0.9032** |

**The ceiling is 0.903 and always was.** v1 wrongly declared it miscalculated. Do not repeat that.

Primary metric **PCC**. Co-reported: QWK (3-cat quadratic), MAE, RMSE, Spearman. Single-rater bar
0.654 (SLT) / 0.628 (orthodontist). Fleiss κ **0.1662** is **the research problem**. Mean pairwise
QWK **0.4276** is **never** a ceiling.

> **[CORRECTED 2026-08-02] These were the 251-row figures, quoted under a heading that says 237.**
> This section previously gave Fleiss κ 0.1605 and QWK 0.4173. Those are the values from
> `summary.json` over all **251 score rows**; the manifest population is the **237 with
> photographs**, where they are **0.1662** and **0.4276**. `data/reliability.py` names both
> deliberately — `FLEISS_237`/`FLEISS_251`, `QWK_237`/`QWK_251` — and its own docstring records
> that the population trap had been walked into three times when it was written. **This is the
> fourth, and it was in the governing document**, which is where the other three were corrected
> *to*. Found while checking the Phase 7B brief, which quotes κ to a clinical audience as the scale
> of the research problem and had it right.
>
> R2 in its plainest form: the same statistic over two populations, one heading, and no way to tell
> from the number which one it is. The ceiling, SB and mean-r above were always the 237 column —
> only these two rows were mixed.

**Claim criterion — amended 2026-07-28.** Two conditions, both required:

1. **Paired BCa CI over patients** (10k resamples) excluding 0.
2. **The delta exceeds the combined seed uncertainty of the two arms being compared** — each arm's
   own measured SD, not a band inherited from another arm. See §4.12.

   That phrase admits two readings and they differ by ~2.5×, so `phase3.combined_claimable_delta`
   computes both and names them. **`arm_means_95` = 1.96·√(sd²ₐ/nₐ + sd²ᵦ/nᵦ) is the criterion** —
   arms are reported as means over seeds, so this is the uncertainty on the difference of those
   means. `single_run_95` = 1.96·√(sd²ₐ + sd²ᵦ) is the conservative companion: the spread of a
   difference between two *single* runs, ignoring the averaging. Quote it when a difference survives
   even without averaging, which is a stronger statement worth making when true. Say which one a
   number is.

**The Nadeau–Bengio correction is removed, and this is why.** v3 required it, citing the literature
for CV comparisons. NB corrects a *t-test over fold-level estimates*, where the folds share
overlapping training sets and the resulting correlation inflates apparent precision. **This project
does not run that test.** It pools out-of-fold predictions into one 237-vector per arm and
bootstraps over patients. There is no set of fold-level estimates to correct, so implementing NB
would mean building a correction for a procedure nobody uses.

**A third instance of the pattern R2 exists for** — after QWK-as-ceiling and the 0.903 "correction",
a statistic imported for the right-sounding reason and attached to the wrong quantity. It survived
into the governing document and into a docstring in `eval/metrics.py`, and was caught only when
Phase 4's exit criteria turned it into something that had to actually exist.

**What the design does and does not capture:**

| source of variability | captured by |
|---|---|
| patient sampling | paired BCa over patients |
| training procedure | the measured seed band (§4.12) — **empirically**, which is what NB approximates analytically. Measured beats approximated |
| **fold assignment** | **neither. This is the real gap, and NB would not have closed it either** |

Folds are fixed across seeds, so nothing in the current design sees fold-assignment variability.

> **[LIMITATION — the gap is bounded, not closed.]** Phase 4 was to close this empirically by
> regenerating folds at several fold-seeds. **That is not possible.** `data.folds.generate` runs the
> splitter with `shuffle=False` and its docstring is explicit that the seed is *recorded but does not
> affect the partition* — so five seeds return one fold set five times and a spread of exactly zero,
> a clean number answering no question.
>
> What Phase 4 measures instead is **sensitivity to partitioning**: the same arm at 5, 6 and 10
> folds plus leave-one-patient-out, all with `shuffle=False` intact and nothing touching the frozen
> module. That is a **bound** on the gap, not a measurement of it, and it is **confounded with
> training-set size** — 5-fold trains on ~190, 10-fold on ~213, leave-one-out on 236.
>
> **Fold-assignment variance remains un-measured, and the design cannot see it**: `cleft_v1`'s folds
> are fixed, every arm uses them, and the generator is deterministic, so no arm varies fold
> assignment while holding everything else constant. Nadeau–Bengio was removed from this criterion
> partly because it addressed a test this design does not run — and **the gap it was masking is
> exactly this one**. It belongs in limitations as a stated gap in the claim criterion.

**[MEASURED 2026-07-28] RMSE-against-a-constant is a calibration check, not a learning check.** The
`beats_constant` sanity test — pooled OOF RMSE against the label SD — was written as gate 2's
blocker after the first Phase 3 run lost to a constant predictor, and it was right to. But across
the ten-seed sweep it was **false in 3 of 10 seeds while pooled OOF PCC was 0.23–0.27 in all ten**.
That is not noise. A head fit under MSE on 152 samples compresses its predictions toward the label
mean, and a shrunk predictor keeps its correlation while its RMSE approaches a constant's.

Two quantities, two questions — R2 again. In a **PCC-primary** project the one that decides whether
an arm learned is PCC against zero. Both are reported, with `shrinkage` = sd(pred)/sd(truth) beside
them because that is the quantity that explains a divergence, and with the interpretation attached
in `metrics.json` so a reader meeting `beats_constant: false` does not rediscover this from scratch.
**Nothing in the ladder gates on `beats_constant`.**

## 4.4 Geometry

**[LITERATURE]** The nasolabial crop is clinically justified — Asher-McDade (1991) masked the face
because judges are influenced by general facial attractiveness; Schwirtz (2018) confirmed cropped
and full-face differ at p<0.001.

**[MEASURED]** Crops are not square: AR 0.553–1.099, median 0.740. Staging is pad-square-then-resize,
white pad, to 224.

**[MEASURED]** Pixel-level background removal is impossible — alpha is dropped at `convert('RGB')`.
Two PNGs, one "background removed", were bit-identical: 0 of 36,278 pixels differed.

**Two geometry arms:**
- **G1 trapezium + computational exclusion** — corners stay in pixels, excluded from computation.
- **G2 unwarp-to-square** — resample to a canonical trapezium, stretch each row to full width so the
  corners cease to exist.

> **[MEASURED 2026-07-30] "The corners cease to exist" is true of the corners, not of white.** The
> frozen composition (`stage_build.py`: `stage`, then `unwarp` the padded square) stretches the white
> pad — and part of the content-relative baked corners — **into** G2 at every AR ≠ 1: G2 white ≈ the
> pad fraction (0.259 at the median AR, 0.0026 at AR 1.0, 0.446 at AR 0.553). `p2-stage-1`'s
> `white_fraction_max: 0.0` for the `*_g2` patch sets is a **patch-geometry constant** — coverage
> against the full-square `G2_TRAPEZIUM`, zero by construction, no pixels consulted — and Phase 5
> briefly misread it as a pixel fact (R2), deriving a "G2 must have zero white" invariant and
> wrongly retracting a correct "matched padding" conclusion on its authority. `staged_v1` is
> internally consistent and uniform across arms; nothing is invalidated. The full record:
> `docs/FROZEN_KNOWN_STALE.md` §3 and `scut/masked.py::G2_WHITE_DEFECT` (RESOLVED). Masked SCUT now
> reproduces the frozen composition bit-for-bit, asserted by test.

**[REASONED]** that unwarping preserves asymmetry. Each row scales by a different factor; the
stretch is left-right symmetric so it *should* survive, but this is unverified. **Check before
building G2:** synthesise a known asymmetry, unwarp, confirm it survives.

> ### [CORRECTED 2026-08-08] "Verified to machine epsilon" is true of the MAP and false of the staged pixels.
>
> `asymmetry_is_preserved` passes at a **1e-9** tolerance, and that is a real result — but it takes
> **no size argument**. It evaluates `half_width_at` and `unwarp_x`, both fractions of the content
> box, so it is a statement about the analytic map and returns the identical answer at every
> resolution. **It has never touched a pixel.**
>
> The pixel path, `trapezium.unwarp`, was measured for the first time on 2026-08-08 by unwarping a
> mark at a known offset and comparing its recovered centre against `unwarp_x`:
>
> | size | residual |
> |---|---|
> | 224 | 3.29e-03 |
> | 512 | 1.40e-03 |
> | 768 | 9.14e-04 |
>
> **Six orders of magnitude looser than 1e-9**, shrinking as ~1/size — resampling quantisation, not
> a defect. Nothing built on G2 is invalidated: the map is right, and interpolation error at this
> scale is expected.
>
> **What is wrong is the sentence.** "Road A verified G2 to machine epsilon" reads as a claim about
> the staged images, and a reader will take it that way. It is a claim about the coordinate
> transform. Quote it with the distinction or not at all.
>
> Road B's Phase 2 gates the pixel residual per setting, and gates it **on the ordering** — the
> residual must shrink as resolution rises, because a per-cell tolerance each setting passes can
> still hide a path that gets worse where it should get better.
> `roadb.ASYMMETRY_GATE_IS_RESOLUTION_BLIND`, `roadb_staging.assert_residual_monotone`.

**New argument for G2 [REASONED]:** a 3×3 grid over a trapezium has corner cells falling partly
outside the mask, so patch generation needs coverage rules, clipping, or inward shifts. Under G2 the
mask *is* the full square and the grid construction is clean. That is a real simplification, and it
means the geometry choice and the patch scheme interact — worth testing together rather than
assuming independence.

**Second concrete instance [MEASURED, 2026-07-28 contact sheet]:** under G1 the grid carries
substantial background — 10 of 27 patches contain white, up to 50% of a patch — while the anatomy
regions contain **none at all** (worst coverage 1.000). The anatomy regions are smaller and placed
inward, so they never reach the mask edge.

So at G1, **anatomy-vs-grid and G1-vs-G2 are confounded**: a grid-vs-anatomy difference measured
there is partly a difference in how much background each scheme feeds the model, not only in where
the patches sit. **The region-scheme ablation must be run at G2**, where both schemes see zero
background. This is no longer a reasoned worry about interaction; it is a measured one, and it
constrains the arm list.

**Third instance, sharper [MEASURED, 2026-07-28 sweep 2; corrected at the freeze]:** at the chosen
spread (`anatomy_scale_x = 1.60`, selected on G2 panels) the `lateral_orbit` regions sit 0.384 from
the midline, where the G1 trapezium half-width is only 0.315. They are inside the frame — so no
clamp fires — but most of each falls outside the mask. At G1, **4 of the 27 anatomy regions carry
background at all**, and the two `lateral_orbit` regions are the extreme: coverage **0.2356, so
76.4% outside the mask**, at the frozen `scale 1.80`.

> **Correction.** This paragraph previously said 87%, which is the figure at `scale 1.30` — measured
> before the scale sweep concluded and left behind when the scheme froze at 1.80. Coverage
> **improves** as the boxes grow, because a larger box extends inward across the mask edge faster
> than it extends outward. Same quantity, different scale (R2). The conclusion is unaffected either
> way, but 76.4% at `scale 1.80` is the number that describes the frozen scheme, and it is the one
> that goes in a write-up.

Running the anatomy scheme at G1 would therefore feed two of its 27 nodes mostly background. Anatomy
is a **G2-only** scheme as currently parameterised; running it at G1 needs a smaller spread, and
that would be a different scheme, not the same one at a different geometry.

## 4.5 Region and patch scheme — **CORRECTED IN v3**

### 4.5.1 The primary design: multi-scale grid patches [the specification given at supervision]

1. **Mask first**, so background never enters the graph.
2. **Generate patches only inside the masked area.**
3. **3×3 grid positions**, crossed with different sizes and aspect ratios — horizontal, vertical,
   and combinations — giving ~27.
4. **Each patch is a graph node.**
5. Features per node from a backbone.
6. **SR-GNN message passing**, so regions compare and exchange information, and the model learns
   which areas drive the judgement.

The anatomical terms supervision mentioned — eye corners, nose, mouth, upper lip — describe what the
generated patches must **cover**, not regions to hand-define.

### 4.5.2 The construction is the maintainer's to determine

supervision gave the idea and left the specifics: *"You can think about the number of patches, resolution,
and regions."* That makes patch count, scale set, aspect-ratio set and overlap **design variables to
investigate**, not parameters to guess once.

**[REASONED] — the most direct reading of "27":** 3×3 = 9 positions × 3 shape variants (square,
horizontal, vertical) = 27. This matches the supervision wording most closely. Alternatives that also reach 27:
9 positions × 3 scales; or 3 scales × 3 aspect ratios at a single centre plus grid offsets. **The
notes do not determine it.** Pick one, document the reasoning, and treat the count as tunable.

**Resolution follows the thirds specification [the specification given at supervision]:** top row coarse, middle normal, bottom
finest. This maps onto grid rows directly.

**Open sub-questions, all the maintainer's to settle:** patch count; scale and aspect-ratio sets; overlap
between neighbouring patches; how to handle patches straddling the mask boundary under G1; whether
resolution is per-row or per-patch.

### 4.5.3 The anatomy-anchored regions become an ablation

The 27 anatomy regions (alar base, philtral column, vermillion border, commissure, etc.) are
**not** equivalent to the supervision material's patches and must not replace them. They become a separate arm testing a
different hypothesis:

| | hypothesis |
|---|---|
| **Grid patches (primary)** | data-driven multi-scale patches → graph learning discovers which regional relationships matter |
| **Anatomy regions (ablation)** | clinically predefined regions inject prior anatomical knowledge |

**[LITERATURE]** The evidence favours the primary. Charm (CVPR 2025) found random patch selection
beat saliency, entropy, frequency and gradient selection on small aesthetic datasets, and noted that
aesthetic assessment needs attention to both foreground and background. Prior-knowledge region
selection is a genuine bet, not a safe default.

### 4.5.3 A structural limitation of the ablation [MEASURED, 2026-07-28]

The anatomy scheme is settled after three contact-sheet sweeps: `v_offset 0.04`, `scale 1.80`,
`scale_x 1.60`, `box_scale_x 1.00`. At those values it is 27 nodes, zero clamping, redundancy 1.392,
53 of 351 pairs overlapping, max pairwise IoU 0.282, covering 70.2% of the frame.

**Node count is matched to the grid at 27. Node AREA is not, and cannot be.** Anatomy regions
average **0.527 of the grid's mean patch area**. Closing that gap needs `scale ≈ 2.48`, and `scale
2.4` already clamps the commissures and lateral orbits against the frame — so matched-capacity
anatomy does not exist within this region layout.

This is a **stated limitation of the ablation, not a defect to fix later.** The honest framing:

- if **anatomy wins** at roughly half the per-node area, the result is strong — prior anatomical
  knowledge overcame a capacity disadvantage;
- if **anatomy loses**, capacity is a live alternative explanation and the comparison cannot
  separate placement from area.

Either way the write-up must say so. The alternative — enlarging regions until areas match — would
trade a known confound for a hidden one, since the enlarged regions clamp and would no longer be the
structures they are named after.

### 4.5.4 Mirror symmetry — weakened [REASONED]

the record previously argued paired regions were "the asymmetry instrument". A 3×3 grid is mirror-
symmetric as a set anyway — column 1 mirrors column 3, the middle column self-mirrors — so the
property survives as a *consequence* of the grid rather than as a designed-in feature, and the
pairing is coarser than in the anatomy scheme. Whether pairwise differences survive SR-GNN's
permutation-invariant SumPool was **never checked** and remains open.

## 4.6 Architecture

**[LITERATURE]** SR-GNN and AG-Net were validated only at ≥2,040 images on categorical fine-grained
classification, with no small-data or asymmetry evidence — **upper-capacity comparators**, not
expected winners. Task-matched SOTA is a Siamese contrastive symmetry model (Rosero 2025). The
landmark baseline is from the supervision material's own group (Bakaki, r≈0.236 automated / 0.457 human).

**[LITERATURE] Mandatory baselines:** landmark geometry / SymNose reimplementation; random-patch;
whole-image.

**[MEASURED]** Two views carry the label — raters saw frontal and basal together.

> ### [CORRECTED 2026-08-31] The line above is wrong, and its tag was wrong before that.
>
> **The tag was never earned.** `[MEASURED]` cited no source and nothing was measured: the claim was
> reasoned from a plausible account of how the score sheet must have worked, and was never checked
> against an artifact on disk. Governing record: `ladder.BASAL_RATIONALE_UNSUPPORTED`.
>
> **Downgraded 2026-08-23 to UNSUPPORTED**, on the sheet (251 rows keyed to frontal IDs, no basal ID
> scored; folder 143's basal 523 absent; only 2 adjacent-ID pairs present, coincidence not per-view
> scoring — the PLAN's own §2 inventory already said "basal (unscored)") and on the three instrument
> primaries, read at source, which use frontal and **lateral**, not basal.
>
> **Upgraded 2026-08-31 to CONTRADICTED.** The supervisor's own manuscripts describe the procedure
> directly: each rater independently scored **one cropped Frontal-Eye-View** image per patient, and no
> basal, submental, profile or lateral view was shown to anyone. Quoted verbatim with each source
> document named in `phase12.RATING_PROCEDURE_DOCUMENTED`.
>
> **Why the downgrade stopped at unsupported and this does not.** The sheet and the primaries could
> establish only that *nothing distinguished* frontal-only rating from both-shown. The manuscripts
> describe what the raters were given. Both bodies of evidence stand together.
>
> **Nothing measured changes.** No arm was re-run and no figure moves — Phase 12 pre-wrote both
> interpretation sentences precisely so that this answer could not move a number
> (`phase12.SENTENCE_SELECTED`). Provenance of the error, including that it survived here eight days
> after being downgraded everywhere else: `phase12.TWO_VIEW_CLAIM_PROVENANCE`.

**[REASONED]** The two-view architecture shape (two branches fused before one head). Alternatives
never evaluated: shared encoder with a view embedding, late fusion, single model on a stacked image.
**[REASONED]** that the missing basal view explains the r≈0.3 plateau.

> ### [CORRECTED 2026-08-31] The plateau inference above is VOID, not merely unsupported.
>
> Its premise is the two-view claim corrected immediately above, which is now **contradicted** by the
> primary manuscripts. An inference from a false premise is void: the missing basal view cannot
> explain the r≈0.3 plateau, because the raters never saw a basal view.
>
> **This was marked in `STAGE_B_NOT_RUN` on 2026-08-23 and the PLAN line was never touched** — the
> same eight-day gap as the line above, and the reason both corrections are being made here now.
> Governing record: `ladder.BASAL_RATIONALE_UNSUPPORTED["linked_item_corrected"]`.

**[REASONED]** Separate lip and nose heads. A finding about *rater* reliability is not automatically
a finding about model architecture.

## 4.7 Training protocol

**[MEASURED]** Regression head, scalar, MSE, raw 1–5. Not CORAL/CORN — tails too sparse.
**[MEASURED]** Inner-val early stopping, identical patience. The old 40-epoch budget measured
post-collapse endpoints (peaks at epochs 2–11, decay to ~0 in 20/20 fold-runs).
**[MEASURED + LITERATURE]** BN re-estimation for SR-GNN/AG-Net — the root cause of the old
"head-init defect". ViT/Swin use LayerNorm and were clean 20/20. AdaBN is the established technique.
**Do not touch the head path.**
**[LITERATURE + MEASURED]** No horizontal flip — not label-preserving, and laterality is unrecorded.
**[REASONED]** Augmentation order: background exclusion → photometric → geometric with boxes
transformed in lockstep.
**[MEASURED 2026-07-28]** Seed count — **derived, see §4.12.** No longer an assumption.

**[MEASURED 2026-07-28] Trainable-parameter policy.** `trainable: head` — the backbone is frozen and
only the regression head trains, on embeddings extracted once and reused across every fold and seed.
Full fine-tuning of ViT-B/16's ~86M parameters on 237 images gave PCC −0.013 and QWK exactly 0.0,
worse than a constant predictor: it damaged the pretrained representation rather than adapting it.
The head is **769 parameters** — 768 embedding weights and a bias — and `metrics.json` reports that
count, checked at runtime against the declared policy.

## 4.8 Explainability [LITERATURE]

Grad-CAM passes Adebayo's sanity checks; **Guided Grad-CAM fails and must not be used**. Run the
model-parameter randomisation test. t-SNE is illustration only, always with a quantitative companion
(k-NN accuracy, silhouette); report perplexity and seed; never tune on it. Prototype interpretability
is contested — medoids must be validated, not assumed.

Note the fit with §4.5.1: once patches are graph nodes, **node attention is the explanation** —
per-region, no CAM upsampling artefact. the supervision material's "voting" description is exactly this.

## 4.9 SOTA comparison, stated honestly

Rosero 2025: r=0.31, n=146, CARS symmetry labels, TSTR. ~~This project's void result: 0.297,
n=237, Asher-McDade composite, 5-fold CV.~~ **"Same range on a related task" is defensible;
"at parity" is not.**

> **[WITHDRAWN 2026-09-05 — 0.297 traces to nothing.]** The struck figure above is preserved
> as written. It is withdrawn because **it has no source anywhere in this repository**: a
> repo-wide search finds `0.297` exactly once, on that line, and `git log -S"0.297" --all`
> returns only the commit that introduced this document. No run directory, no phase record,
> no ledger row and no test produces it.
>
> **The provenance is the finding: a figure entered the governing document underived and was
> then quoted as this project's own result.** It is the same class the record has been
> correcting throughout — a value asserted fluently enough that nobody asked where it came
> from — and it is the sharpest instance, because the governing document has precedence over
> everything but code.
>
> **What replaces it, and it is banked**: the best arm is **PCC 0.2520** (sd 0.0148, 5 seeds,
> `results_ledger` row `p7-best-arm-descriptive`, status DESCRIPTIVE). The comparison the
> struck sentence was making cannot be made from 0.2520 either — different cohort, different
> labels, different protocol — so the honest form is that **no comparison to Rosero 2025 is
> available**, which is stronger than the sentence it replaces and is what the surrounding
> paragraph already argues.

## 4.10 CV versus TSTR — resolved [LITERATURE]

Different questions, not competing designs. CV estimates *the procedure* (5 models, ~152 images
each); TSTR estimates *one artifact* on a 237-patient test set. The literature pattern is both.

**CV is primary for the ladder** (Q1–Q4 all need cleft fine-tuning); **TSTR is one additional arm**,
the Rosero replication. **[REASONED] caveat:** Rosero's CARS label is symmetry-weighted so synthetic
deformation maps onto it plausibly; Asher-McDade is a composite, so the assumption is stronger here.

> **[AMENDED 2026-07-30] The TSTR arm is parked and Stage F is struck** — see Part 6. The caveat
> above is part of why: the arm's interpretability rested on the synthetic deformation mapping onto
> the label, and the synthesis' own recorded limitations say it cannot reach the region that best
> tracks the grade. **The CV-versus-TSTR resolution in this section still stands as a resolution** —
> they answer different questions and the literature pattern is both. What changed is not the
> reasoning but whether this project has a TSTR arm worth running, and the answer is currently no.

**Folds are needed either way** — generate them in Phase 1.

## 4.11 Open questions

| # | question | status |
|---|---|---|
| Q-a | Separate lip and nose heads? | [REASONED], open |
| Q-b | How to deform a SCUT face into plausible cleft asymmetry | design + supervision |
| **Q-e** | **How the nose tip enters the pipeline** | **partly resolved 2026-07-30 — see below** |
| ~~Q-c~~ | ~~27 regions: grid or anatomy?~~ | **resolved — grid primary, see §4.5** |
| Q-d | Basal patch scheme | design |
| **Q-f** | **The patch construction — count, scales, aspect ratios, overlap** | **The maintainer's to determine, §4.5.2** |

### 4.11.1 Q-e — the nose tip [MEASURED 2026-07-30, the univariate relevance diagnostic]

**One of three readings is answered; two are untouched.** Ranked by |Spearman| over the 22
mirror-difference features against the mean label, n=237. Thresholds are computed
(`relevance.significance_threshold`): **0.128 uncorrected, 0.199 Bonferroni across 22.**

| rank | feature | \|Spearman\| |
|---|---|---|
| 1 | `philtrum` | **0.151** |
| 2 | `labial_tubercle` | **0.138** |
| 3 | `philtral_column` | 0.122 |
| 4 | `subnasale` | — |
| 19 | `alar_base` | ~0 |
| 20 | `commissure` | ~0 |
| **22** | **`nasal_tip`** | **0.002** |

**Nothing clears the corrected bar**, so this remains a ranking that informs a design decision, as
declared before the run.

**`nasal_tip` is last of 22, so the suggestion raised at supervision is not supported by this measurement — and the
scope is the whole of what makes that usable.** What was measured is *pixel-level mirror asymmetry
within a box placed by anatomy fractions*. Two other readings are untested and remain live:

- **tip deviation from the midline** — a *displacement*, not a regional pixel difference. A tip
  displaced bodily off the midline can leave the mirror difference inside its own box nearly
  unchanged, because the box travels with the anatomy fractions rather than with the face's midline.
  It is also the quantity the Phase 2 cleft sheet's tilt gradient pointed at.
- **the tip as an alignment anchor** — a *use*, not a feature. A zero correlation of its regional
  asymmetry says nothing either way.

**Do not record this as "the nose tip does not matter."** That is the R2 error in a new place: a
correct number attached to a question it cannot answer. `relevance.NOSE_TIP_DECISION` holds the
record.

**The top four are all midline, all lower face, all philtral and upper-lip complex** — which coheres
with the group's own finding that raters are more reliable scoring lips alone. Two independent lines
of evidence pointing at the same anatomy.

> **[MEASURED] A methodological note, and it generalises.** `mad_whole` scores **0.004** while
> individual regions carry **three to thirty times more**, and the band features are near zero. So
> the mirror-difference arm's **PCC 0.158 comes from the ridge combining many weak regional signals**,
> not from any single site and not from the aggregate. **An aggregate that averages heterogeneous
> signals reports none of them** — the same lesson as the fairness quartiles, where a cohort-level
> pass rate hid a brightness-linked gradient, and as `separability`, where a whole-frame statistic
> measured face-versus-background. Third instance, three different parts of the project. The rule:
> **report the per-unit breakdown beside any aggregate whose units might differ from one another**,
> and read an aggregate near zero as uninformative about its parts rather than as evidence they are
> zero. `relevance.AGGREGATION_NOTE`.

**A consequence for the synthesis, recorded and not acted on.** The TPS **anchors the philtrum** —
the top-ranked region — because the midline landmarks are pinned to keep the mirror-difference
measurement's midline stable, so the deformation cannot touch the area that correlates best with the
grade. And of its four targets, `philtral_column` ranks 3rd while `alar_base` (19th) and
`commissure` (20th) are essentially zero. **Not decisive, and the reason is R2:** the TPS displaces a
*landmark* while the index measures *pixel asymmetry in a box*, and nothing here measures the former.
**A decision is needed before Phase 7**; changing targets now to chase a ranking that clears no
corrected threshold would be fitting the instrument to a diagnostic. Recorded in
`synthesis.LIMITATIONS["targets_are_misaligned_with_measured_regional_relevance"]`.

## 4.12 The seed band — what a claim has to beat [MEASURED 2026-07-28]

Gate 2. Ten seeds in one job, one environment, one `env.json`. Frozen ViT-B/16 embeddings and a
linear head, G1, mean label, 5-fold CV over the Phase 1 folds.

**What the seed actually changes here — corrected 2026-07-28.** Two things, not one:

1. **Head initialisation.** The embeddings are extracted once and reused, so the *features* are
   byte-identical across seeds and contribute nothing to the spread. That part of the original
   claim holds.
2. **The inner-val split.** `harness.inner_val_split` seeds off `config.seed`, so the seed also
   decides **which patients are held out of the training fold**. Every arm in this project inherits
   that, whatever its model.

So **0.0137 is initialisation *and* split variance combined**, and this sweep cannot separate them.
Earlier text — this section, `phase3.prepare_features`, and the Phase 3 config — said seeds differed
in "head initialisation and nothing else". That was wrong, and it is the name the number carries
into the write-up, so it is corrected here rather than left to be inferred.

Decomposing it would need a dedicated sweep holding one fixed while varying the other. Nothing needs
that yet; the combined figure is the right threshold for comparing arms, because a real arm varies
in both.

| quantity | value |
|---|---|
| mean pooled OOF PCC | **0.2529** |
| SD | **0.0137** |
| min | 0.2347 (seed 7) |
| max | 0.2719 (seed 1337) |
| observed range | 0.0373 |

**The claimable delta.** Two arm means at *n* seeds each differ by a standard error of
`sd·√(2/n)`; at 95% that is 1.96 of them:

```
claimable delta = 1.96 · sd · √(2/n)
```

| seeds per arm | smallest claimable PCC delta |
|---|---|
| **1** | **0.038** |
| **5** | **0.017** |
| **10** | **0.012** |

**One seed resolves nothing this project is trying to measure** — 0.038 is essentially the entire
observed range. The void ladder's 0.068 swing across SHAs was comparable to the effect it was meant
to measure; this is the same failure mode quantified in advance instead of discovered afterwards.

A formula rather than three memorised numbers, so a re-measured SD gives a re-derived band. It lives
in `train/phase3.py` as `claimable_delta`, and `seed_variance.json` reports the band **derived from
whatever sweep produced it**, not the constant below.

> **Caveat, and it is a real one.** This band is for the **frozen linear probe**, and it is
> initialisation-plus-split. Any arm with a substantially different training procedure — an unfrozen
> backbone, the patch path, SR-GNN, augmentation — has more sources of seed sensitivity and **must
> have its own band measured**. Carrying this one across would be the QWK-as-ceiling mistake in a new
> place: the right number for the wrong quantity (R2).

### 4.12.1 Every arm reports its own SD [DECIDED 2026-07-28]

The band above is **not a project-wide constant**, and quoting it for an arm that did not produce it
is the error it exists to prevent.

- **Each arm measures and reports its own seed SD.** Inherited, never.
- **A delta between arms A and B carries uncertainty from both**, so the threshold is the combined
  figure, not either arm's band alone and not 0.0137.
- **Two arms can both report "seed SD" and mean different things.** This is not hypothetical, it is
  the two arms that exist:

  | arm | what its seed varies | what its SD measures |
  |---|---|---|
  | frozen ViT probe | head initialisation **and** the inner-val split | init + split, combined |
  | mirror-difference ridge | the inner-val split only — the solver is closed-form and deterministic | split alone |

  Neither is wrong; they are different quantities wearing the same label. **Say which one a number
  is** whenever it is quoted, and never compare an SD from one against a band from the other as
  though the smaller one were the more precise arm.
- **Where an arm's SD really is zero, say so plainly** rather than quoting a band that does not
  apply. A zero is not a very tight band — it is a statement that the seed is not a source of
  variation for that arm at all, and the comparison still has to carry the *other* arm's spread.
- **Every arm inherits split variance**, whatever its model, because `inner_val_split` seeds off
  `config.seed`. An arm with genuinely zero seed SD would therefore be one that never consults the
  split — worth checking rather than assuming, since assuming it is what produced the correction
  above.
- Seed count per arm follows from what that arm needs to claim, not from a habit of five.

---

# PART 5 — PHASES

**Phase 0 — Foundation. ✅ COMPLETE 2026-07-27.** Provenance contract, guards, ~196 tests in ~42s,
env-only image pinned by digest, verified on the cluster.

**Phase 1 — Data.** Manifest (lookup, not rule), labels, reliability, folds.
*Exit:* 237/236/14 asserted; rule cross-check flags exactly 143 and 238; reliability reproduces
**on the 237** Fleiss **0.1662**, QWK **0.4276**, SB 0.8158, ceiling **0.9032**; learnability
reproduces §4.2. (Corrected 2026-08-02 — see §4.3. The 0.1605 / 0.4173 previously here are the
251-row values, and `BRIEF_phase1.md` §8 has said so since 2026-07-27.)

**Phase 2 — Geometry and patches.** Staging; G1 and G2; the **multi-scale patch generator**; the
anatomy region set as a second, separate generator; per-image mapping.
*Design task (Q-f):* fix the patch construction, document the reasoning, make count and scale set
configurable so they can be investigated rather than assumed.
*Exit:* patches lie inside the mask for all 237 under both geometries; the generator is
deterministic given a config; the §4.4 asymmetry-preservation check passes for G2; anatomy and grid
generators produce comparable node counts so the ablation is fair.

**Phase 3 — Gates and freeze. ✅ COMPLETE 2026-07-28, tagged `phase3-freeze`.**

| gate | state |
|---|---|
| 1 determinism | **green** — three runs byte-identical, PCC 0.2719366015264466. `use_deterministic_algorithms(True)` raised on nothing |
| 2 seed variance | **green** — §4.12 |
| 3 epoch-0 calibration | **green** on real data |
| 4 epoch policy | **frozen as-is, never stressed at the freeze** — stressed 2026-07-31, a phase early; see the amendment below |
| 5 metric verification | **green** on real data |
| 6 OOF reconstruction | **green** on real data |

**Gate 4 is recorded as unstressed, not as passing.** Every fold stops at epoch 1–4, because a
768-dim head on 152 samples overfits immediately. The policy is right and has nothing to do on this
arm; it will be stressed by Phase 7's arms, which is where a failure would mean something. Freezing
it in that state is deliberate — the alternative is inventing an arm to exercise it and freezing on
a result nobody needs.

> **[AMENDED 2026-07-31] Gate 4 was stressed a phase early — by Phase 6's SR-GNN scheme
> pretraining runs — and the policy failed there in a way the freeze-time reasoning did not
> cover.** The failure mode was **patience terminating on noise in a flat region**, not the
> monitored quantity: anatomy's epoch 5 gave inner_val_mse 0.155406 and epoch 7 gave 0.155429 —
> 2.3e-5 apart — so patience fired on whichever side of the noise a value landed. Run length
> tracked luck rather than scheme (native 31 / grid 30 / anatomy 13 / random 13 epochs, the
> scheme "ranking" tracking epochs run), and **none of the eight completed pretraining runs had
> converged**: inner-val PCC was still climbing at every stopping point. A replay of six
> candidate policies over all eight curves (`scripts/replay_stopping.py`) showed every run
> changing selection under a PCC monitor, with six running out of recorded curve — switching
> the monitored quantity only moves where the noise bites.
>
> **[DECIDED 2026-07-31] Pretraining uses a fixed 30-epoch budget with best-checkpoint
> selection on inner_val_pcc — no early stopping** (`train/pretrain.py`,
> `FIXED_BUDGET_POLICY`). Comparability across the thirty arms is by construction and the cost
> is knowable in advance. **This does not contradict this gate.** The void ladder's fixed
> 40-epoch budget measured post-collapse endpoints on 152 cleft samples — inner-val PCC peaked
> at epochs 2–11 and decayed to ~0 in 20 of 20 fold-runs; on SCUT's 3,300 there is no collapse,
> PCC climbs monotonically in all eight curves. The rejection of fixed budgets was a property
> of the cleft data size, and **early stopping stays for the cleft ladder, where collapse is
> the documented behaviour.** Selection on inner_val_pcc matches the primary metric and is
> legitimate here because inner-val is carved from the train side and the test set is disjoint,
> touched once.
>
> **The eight pre-policy pretraining runs are SUPERSEDED, not deleted** — four backbone
> originals (Swin-B 0.9122, SR-GNN 0.9024, ViT-B/16 0.8904, AG-Net 0.8858) and four SR-GNN
> scheme originals (native 0.8350, grid 0.8331, anatomy 0.8232, random 0.8099, confounded with
> epochs run). Their curves are the evidence for this decision and belong in the write-up; the
> whole lattice is redone under the fixed budget.

**Gate 1 will go void again at Phase 7.** It tests the *harness*, and the patch path changes the
harness. Re-run it against `GATE1_REFERENCE` when that lands.

*Exit met:* all six accounted for → committed, tagged, **frozen**. After the freeze the code does
not change; anything discovered goes in limitations, because arms run at different code states are
not comparable.

**Phase 4 — Baselines.** Mirror-difference symmetry index (primary geometric baseline, G2);
SymNose reimplementation *conditional on a segmentation-feasibility diagnostic*; patch-fed frozen
probes for grid / anatomy / random at G2; a geometry probe of random patches at G1 vs G2. All arms
use frozen features, so none needs pretraining, fine-tuning or checkpointing. The bar is the
whole-image probe at 0.2529.

### Phase 4 results so far [MEASURED 2026-07-28]

**Mirror-difference index — complete.** Five seeds, G2, ridge over 22 features: **PCC 0.158, SD
0.019** (0.140, 0.142, 0.150, 0.175, 0.183). Against the probe's 0.2529 the delta is **−0.095**,
clearing `arm_means_95` of 0.019 by 5.1× and even `single_run_95` of 0.046 by 2.1×.

**The learned embedding genuinely beats direct geometric asymmetry measurement**, and that is now a
measurement rather than an assumption. Shrinkage 0.39–0.46, which is exactly why `beats_constant`
read false while PCC stayed healthy — §4.3's interpretation string did the job it was added for.

For the write-up: **22 hand-defined features and 23 trained parameters reach 62% of what an
86M-parameter ImageNet representation achieves**, with no learned representation at all. Against
Bakaki's automated plane-of-symmetry at r≈0.236 and human raters at r≈0.457 on related data, that
places a landmark-free, segmentation-free index in a defensible band. Its SD is **split variance
only** — the ridge is closed-form (§4.12.1).

### ✅ Segmentation gate CLOSED — SymNose is IN [DECIDED 2026-07-28, run 5]

**The screen said `out`; the sheet said in; the sheet decided.** `upper_lip` was plausible on 4 of 9
by the criteria and **8 of 9 usable on review** — patients 1, 7, 11, 4 and 18 near-perfect upper lip,
3 and 2 slightly under-covering, 17 halved as predicted. Patient 5 now captures the upper lip with
some spread at the extremes rather than escaping to the chin: the split fixed most of the skin-tone
problem as a side effect.

**The centroid split is validated.** Errors against the detected fissures of **4, −1, 2, −2, 2, −1,
4 pixels** — max 4px, **3.1% of mask height**. The fixed-fraction rule is four times worse at −7.1px
mean and systematically biased. Centroid was the default, so **nothing was tuned**. Geometry
predicts the anatomy accurately, and that is the finding.

**Patient 17 is the predicted cost**, not a surprise: the crop excludes the lower lip, the
always-fires rule halves a correct mask, and no shape statistic can detect it. One in nine, known
mechanism, recorded.

> ### [LIMITATION] `area_min` is mis-calibrated for the upper-lip target — and is NOT being changed
>
> `upper_lip` mean area is **0.0230** against an `area_min` of **0.02**. That floor was calibrated
> when the instrument segmented *both* lips, so a correct upper-lip mask sits right on it and about
> half fall through. That is the whole of the screen's failure, and the cause is visible.
>
> **It stays wrong.** `area_max` was re-expressed after run 1, `fraction_above_fissure` was added
> before run 3 and renamed at run 5, and now `area_min` is mis-calibrated. **Three amendments across
> six runs is the shape of a criterion set being fitted to the data.** Amending it a fourth time
> would make that fit tighter and the criteria no more meaningful.
>
> **What it means is that the screen never decided anything.** The human review of the contact sheet
> was the operative decision procedure at every one of the five gate runs — including this one, where
> screen and sheet disagreed outright. That belongs in the write-up **as stated, not repaired**: the
> honest description of this gate is *pre-declared criteria that flagged candidates, and a clinician's
> eye that decided*, and pretending otherwise would overstate how much of the decision was automated.

**Segmentation gate — run 2 [DECIDED 2026-07-28].** Both fixes worked. Otsu and
Chan-Vese trace the lips near-perfectly on nearly every patient, and Otsu picks up the **surgical
scar** in some — which on a post-repair cleft is much of what the aesthetic grade is about. Colour
separation is unchanged from run 1 apart from being confined to the band, and is out.

**The instrument is Chan-Vese; Otsu is the comparator.** They find the same region, which is why
Otsu is kept — and their agreement is what distinguishes a failing image from a failing method. But
**SymNose mirrors a traced boundary and superimposes it**, so boundary noise enters the asymmetry
measurement *directly, as spurious asymmetry*. Otsu thresholds per pixel and its edge is pixelated;
Chan-Vese carries a curvature term and its edge is smooth. A jagged edge would manufacture the very
signal being measured. Boundary methods for boundary problems.

**One fixed relaxation.** The masks sit just inside the vermillion border and want a little more on
the upper lip — the contour converges where between-class separation is maximised, which is not the
anatomical boundary. The relaxation is expressed as a fraction of **each image's own class gap**
rather than in pixels, so it is one rule with one parameter landing in the same place relative to
every image's separation. A fixed pixel dilation would have been a different relaxation on each
patient. **SymNose measures the upper lip**, so over-relaxing crosses the oral fissure into the lower
lip and silently changes what is measured; the unrelaxed contour is rendered beside the relaxed one
and the per-patient area growth is reported, so the magnitude is judged on the sheet.

**Run 3 — fits 8 of 9, and two findings.**

**The masks cover both lips, and the relaxation is not the cause.** Run 2 already crossed on 3 of 9
at relaxation 0; the relaxation made a pre-existing failure *universal* rather than creating it, so
**reducing it returns to 3 of 9, not to 0**. The mechanism: Chan-Vese segments regions of similar
intensity and the two lips *are* similar, so the fissure is a thin dark line between two lookalike
regions and an intensity-based method has no reason to stop there. Whether it crosses depends only
on how pronounced the fissure is in that image.

**A both-lips mask is a different instrument.** SymNose measures the upper lip; a mask spanning both
carries lower-lip variation into the mirrored asymmetry, which is not part of the construct, and the
resulting index would not be comparable with SymNose's. So the **upper-lip constraint is the fix,
not an addition to a smaller relaxation** — any intensity-based segmenter needs an explicit
anatomical rule to stop at the fissure, because intensity does not encode one. The relaxation is
then re-tuned against the upper lip alone, where 0.15 may well be right.

**New pre-declared criterion:** `fraction_above_fissure ≥ 0.80`, measured on the *unconstrained*
contour because the instrument satisfies it by construction. It is **expected to fail** run-2
relaxed on 9 of 9 and run-2 unrelaxed on 3 of 9, and both are verified as a regression — a criterion
that passes everything on arrival has not been shown to test anything.

> **A defect the tests caught while building it.** The first fissure detector fired on *every* mask
> and halved correct upper-lip-only segmentations — which would have read as **success** on the
> screen, since a halved mask satisfies the new criterion by construction. Two causes, both real:
> the row profile was smoothed with zero padding, so the end rows came out at two thirds of their
> value and the darkest row was always an end of the profile; and nothing required the dip to be
> deep enough to be a dark *line* rather than the darkest row of something uniform. Both are now
> asserted directly.

**Run 4 — the constraint is correct where it fires, and rarely fires.** `n_fissure_found` was 2 of
9; only 3 of 9 masks are upper-lip only on the sheet, and two of those for incidental reasons — one
sits above the vermillion line on the moustache (skin tone, not fissure) and one has no lower lip in
the crop. Six still span both lips, **passing the criterion trivially because no fissure was found**.

The threshold was too strict: 0.08 separated the synthetic fixtures cleanly (0.0000 against 0.1373),
and real fissures are shallower than synthetic ones. The diagnostic now reports the **measurement
rather than found/not-found** — per-patient dip depth, dip sharpness in standard deviations of the
row profile, a sweep of 0.02/0.03/0.05/0.08 from the same run, and a floor count.

> **Detection rate is not the outcome on its own.** A threshold low enough to fire on 9 of 9 while
> landing on arbitrary dark rows is *worse* than one that fires on 2, because a false fissure halves
> a correct mask and the halved mask then satisfies `fraction_above_fissure` by construction — the
> same false positive the smoothing defect produced. Depth alone cannot tell a shallow dark *line*
> from the darkest row of something uniform; at 0.03 they are identical. **Sharpness** can, so the
> sweep reports `n_sharp` and `n_noise` beside `n_found`, and every candidate threshold is
> regression-tested against a no-fissure mask.

### [LIMITATION — MEASURED 2026-07-28, run 4 sweep] The oral fissure is not recoverable

**In seven of nine patients the oral fissure cannot be recovered from these images.** The sweep
settled it: seven of nine dip depths sit below 0.05, and the only real gap in the distribution is
between **0.047 and 0.114** — there is no cut that separates signal from noise. Lowering the
threshold to 0.03 buys two more sharp detections *and one noise detection*, which is precisely the
false positive that halves a correct mask while satisfying the criterion by construction.

The cause is the imagery, not the method. In a **closed relaxed mouth** the oral fissure is one or
two pixels of shadow, and these are compressed JPEGs. There is nothing in the pixels to find.

**This is a limitation of intensity-based upper-lip segmentation on this data, and it belongs in the
write-up beside the fairness point below.** Both are constraints inherited from the method and the
imagery rather than defects to fix: one from what the algorithm family can see, one from whose faces
it was calibrated on.

### The geometric split, and its own trade [DECIDED 2026-07-28, run 5]

With no dark line to find, the mask is divided at a position derived from **its own shape** — the
row of its centroid, or a fixed fraction of its vertical extent (0.5, declared a priori and **not
fitted**). No fissure needed, applies uniformly, and it is what the detection was proxying for.

**The two detected fissures are the validation set**, and they are the whole point of run 4: where a
fissure was found on evidence it is ground truth for where the division actually is. If the
shape-derived line lands in the same place, the geometry predicts the anatomy and the split is a
defensible substitute on the seven patients where nothing could be detected. **If it does not, the
geometry does not predict the anatomy and this route fails too — which is a result, not a setback.**

Two things about that validation, both mechanical in the code:

- **The validation set is selected on sharpness (≥ 1.4), not on the operating depth threshold.**
  That is deliberate and it is *not* the criterion loosened: "where do I trust the ground truth" and
  "what do I run on every patient" are different questions, and a shallow but sharp detection is
  trustworthy evidence even where firing on it automatically would be too risky.
- **The disagreement is reported, not minimised.** Errors are signed, in pixels. Tuning
  `SPLIT_FRACTION` against them would fit the line to the handful of patients with a detected
  fissure and say nothing about the seven the split exists to serve.

> **The trade, stated up front.** The fissure rule was correct wherever it fired and **inert on 7 of
> 9**. The geometric split **always fires** — it has no not-found case — so it halves a contour that
> was *already* upper-lip only, which happens when a patient's lower lip is outside the crop (patient
> 17 in run 4). `split_removed_fraction` is ~0.5 for a both-lips mask and ~0.5 for an upper-lip-only
> one, so **no shape statistic can distinguish them**: the two masks look the same. Only the sheet
> can say which patient is which, and the per-patient split rows are reported so it can.

**A fairness limitation, not an image-quality one [MEASURED 2026-07-28].** Patient 5 is of Indian
ethnicity; the other eight are white British. Every method here is intensity- or redness-based, so
**all are implicitly calibrated on lighter skin** — on darker skin the lip-versus-skin contrast
compresses in both hue and value. The signature is exact: median saturation **0.824** against
0.331–0.492, all four methods taking 44–99% of the band, and the mask escaping onto the chin and
cheeks well outside the vermillion lines.

The `genuine_difficulty` attribution **reached the right verdict for the wrong reason**: it reads as
"hard image", and the cause is "the method assumes a skin tone this patient does not have". Those
are recorded differently, and the attribution note now says so.

**SymNose was developed and validated on a British cleft cohort, so an automated reimplementation
inherits its population assumptions along with its algorithm.** That belongs in the write-up
whatever the screen says.

Three consequences: the nine-patient sample is **not representative on this axis** — one non-white
patient, and it is the only failure, so any success rate carries that caveat; per-image
normalisation of the discriminant is the hypothesis to test, as a **separate run** (R6); and
**nothing is tuned on patient 5** — the fix must be a rule applying identically to everyone, or it
fits one image, hides the fairness problem, and leaves the next darker-skinned patient exactly where
this one is. A second pre-declared criterion requires the instrument to succeed on the
darkest-skinned patient in the sample; it is currently failing.

**Failures are attributed, because a limitation and a defect are recorded differently.** If the
instrument and the comparator both fail but *agree*, two independent methods found the same non-lip
thing and the cause is the image — poor repair with no clear vermillion border, an open mouth,
flattening light. That is a **limitation**. If they *disagree*, at least one is breaking down: a
**defect**. Failing patients are named in `segmentation_per_patient.json` (CLUSTER-ONLY);
`metrics.json` carries counts only.

*Run 1, for the record:* colour separation worked on 5 of 9, two near-perfect; the rest picked lips
*and nose* and over-selected; 3 of 9 collapsed to the whole face. Otsu and Chan-Vese returned
whole-face masks on all nine.

> **The `separability` metric was measuring the wrong contrast, and that is a defect in this
> project's diagnostic, not a finding about the data.** It scored 0.888 and was designed as the
> *stronger* signal — "the discriminant is not bimodal, so no threshold could work". But the staged
> crop is padded white, so the dominant contrast in the frame is **face against background**, and
> 0.888 measured that. Verified on synthetic fixtures: a face with **no lips at all** scores
> **0.992** whole-frame, *higher* than one with lips (0.806), because lip pixels muddy the skin-vs-pad
> split. The number is not merely uninformative, it runs the wrong way. **Run 1's separability
> cannot be cited for or against the protocol's compression claim in either direction.** It is now
> computed inside the face mask, where the same fixtures give 0.0 and 0.992 the right way round.

Two diagnosed causes, two targeted fixes, both built and ready to re-run: a **face mask** so the
search is not over the padding — which also makes Otsu a real per-image lip-vs-skin cut instead of a
fixed constant that 3 of 9 skin tones sat on — and a **spatial prior** restricting the search to the
Phase 2 bottom band, since lips and nose are both red and only position separates them.

**The criteria are a screen, not a gate.** This wants saying plainly because the ceremony around
"pre-declared criteria" implies otherwise. The segmentation question has been decided twice, both
times by reading the contact sheet — run 1's screen said *proceed* and the sheet said *undecided*.
The criteria narrow what is worth looking at and fix what counts as a candidate before the
candidates are seen, which is worth doing and worth declaring in advance. They have never decided
anything on their own. `metrics.json` records `role: screen` and `decided_by: human review of the
contact sheet`, and the write-up should describe them that way.

**The area criterion now reads at two denominators [AMENDED 2026-07-28, post-hoc].** What was
declared was a set of shape and position facts — *a plausible lip is a modest fraction of what was
searched, sits low, near the midline, one blob*. Under a whole-frame search, `area ≤ 0.35 of frame`
expressed that. **The number was the instantiation; the sentence was the criterion.** Adding the
prior changed the search area and broke the instantiation, so each end now takes the denominator its
invariant implies:

| bound | invariant | scales with search area? | read against |
|---|---|---|---|
| `area_max` | a modest fraction of **what was searched** | yes | the **search region** |
| `area_min` | not a speckle — a lip is a real structure | no | the **frame** |

Both are necessary, and that is measured rather than argued. Band-filling is 0.168 of the frame —
inside `[0.02, 0.35]`, so frame-relative cannot catch it — and 1.000 of the region. A 0.4%-of-frame
speckle is 0.023 of the region — above the 0.02 floor, so region-relative cannot catch it — and
`centroid_y`, `centroid_x` and `largest_component` all pass a single small midline blob, so nothing
else would have.

> **No number changed, and the amendment is post-hoc.** Nobody wrote "relative to the search region"
> before run 1; it is the right reading, arrived at after the literal one broke. Recorded as such in
> `segmentation.CRITERIA_PROVENANCE` and carried into every `metrics.json`. With no search region
> the two denominators coincide exactly, so this reduces to the pre-declared form and run 1 is
> unaffected — and both quantities are reported on every mask, so runs with and without a prior stay
> comparable on the original frame-relative number.

`centroid_y` is satisfied **by construction** under a bottom-band prior and stops being evidence
there; it is kept because it is exactly the criterion that matters with the prior off.
`saturates_region` remains a separate reported flag, since `area_max` and it can disagree at the
margin.

### ❌ SymNose (§3.2) — OUT, on a scope boundary [DECIDED 2026-07-28]

**The reason is the manual instrument's dependence on landmark placement, not any result produced
here.** Every SymNose face measure depends on **user-placed roundels at named anatomical points** —
canthi, alar bases, mouth corners, philtrum centre, nose tip, mid-columella. Strange's proposal was
to automate the *tracing*; the roundels are **separate manual work and equally essential** to every
assessment.

So automating SymNose requires **landmark detection**, and that method class is outside this
project's remit by supervisory decision: the supervision material rejected Bakaki's landmark work because it is not deep
learning and the project is directed at deep-learning methods. **A scope boundary, not a failure.**

**No faithful reimplementation is being attempted.**

#### What was built was not SymNose [MEASURED from the app bundle's help files]

| | |
|---|---|
| **SymNose** | levels the image on the **intercanthal line** first; outlines the bottom of the **nose *and*** the upper lip; four face-view assessments, each on a different anatomically-derived fold axis — registering the extreme lateral points to maximise overlap, superimposing alar bases and mouth corners, a philtrum roundel, the mid-columella line; plus lip dehiscence, three user-outlined scar types, three base-view measures |
| **What was built** | an auto-segmented **upper lip only**, mirrored about the **fixed image midline**. No rotation correction. No registration. One axis. |

Those are not the same instrument, and the differences are not refinements to add later. The index
carries **head tilt, crop centring and lateral displacement**, any of which plausibly **dominates**
the clinical asymmetry it was meant to measure — and none is separable from it afterwards.

#### The null, recorded as what it is

**PCC −0.066 (SD 0.051) over five seeds; Spearman −0.003.** Recorded as a **null result for a
non-faithful reimplementation**. It is **not** evidence about upper-lip asymmetry, **not** evidence
about SymNose, and **not** evidence for the scope decision above — that argument stands on the
landmark dependence alone and would stand identically had this index scored 0.4. The two must not be
presented as though one supported the other.

A modest positive was predicted before the run and zero was measured, but the pairing no longer means
what it was set up to mean: the prediction was about a faithful instrument's construct overlap, and
this did not test it.

> **Superseded framing, removed.** An earlier draft of this section read the zero as an informative
> comparison against the mirror-difference index — a targeted instrument finding nothing where a
> cruder one found +0.158. That framing was wrong: a confounded instrument scoring zero supports no
> comparison at all. It is recorded here as removed rather than silently deleted, because it is the
> kind of reading that returns.

#### The segmentation code is PARKED, not deleted [DECIDED 2026-07-28]

Its only consumer was the retired index, and nothing else in the arm list segments anything. **It is
kept because it is the evidence behind a write-up chapter** — deleting it would leave that chapter
describing work with no artefact. Four results stand on their own regardless of what consumed them:
§2's feasibility answer (lips *can* be segmented from compressed JPEG, against a protocol that said
otherwise), the fissure floor, the geometric split validated at max 4px, and the fairness finding.

**Its tests keep running.** A parked module whose suite is skipped quietly stops working, and then
the chapter's artefact cannot be re-derived when someone asks. **Do not extend it** — if a future
phase needs segmentation, that is a decision to take deliberately, with a consumer named first.

#### The audit still runs — for the fairness number [DECIDED 2026-07-28]

Not to attribute the null, which is now uninterpretable either way. **"Unsupervised colour
segmentation is implicitly calibrated on lighter skin" rests on one patient in nine** — a real
observation with a mechanism behind it, and an indefensible sample for the strongest ethical finding
in the project.

The per-patient records already span all 237, so `fairness_by_brightness` splits the pass rate by
`median_value_in_region` quartile. A monotone gradient with the darkest quartile failing most is
what the mechanism predicts, and the per-criterion breakdown says *how* — patient 5's signature was
over-selection, so an excess of `area_max` in the darkest quartile is the specific pattern.

> **Proxy, not skin tone.** `median_value_in_region` also carries lighting, exposure and how much of
> the frame the face fills. A gradient is evidence of a **brightness-linked failure pattern**;
> calling it a skin-tone gradient would need ethnicity data this cohort does not record.
>
> **A flat result is also a finding.** It would mean the n=1 observation did not generalise and the
> claim must be **narrowed to that patient**, not quietly kept at cohort scale.

#### `normalisation: rank` — dropped [DECIDED 2026-07-28]

It was the fairness *fix* for the segmentation. With no downstream metric there is no way to tell
whether it helped: it would change masks that nothing consumes, and "did it work" would have no
answer. The fairness **finding** is being measured instead; the fix is not being attempted.

**§3.2 — the retired index, as built.** The protocol's algorithm with **automatic segmentation
replacing the manual tracing**, which is the gap this lineage has been stuck on since Strange's
proposal. Mirror the segmented upper lip, superimpose, quantify the mismatch. Primary index is the
mean boundary-to-mirrored-boundary distance over √area — a boundary quantity because SymNose
superimposes boundaries, normalised so it does not track crop size; 95th percentile rather than
maximum, since one stray pixel should not define an index; magnitudes not signs, because laterality
is unrecorded (§4.1).

**The image midline is the primary axis, and the reason is anatomical.** It preserves **lateral
displacement**, which in a unilateral cleft is genuine asymmetry and often its largest component.
Mirroring about the mask's own centroid scores a purely translated lip at exactly zero — which is
why it cannot be primary. Both are reported with `centroid_offset_px`; where offsets are large, the
gap between the indices *is* the lateral-displacement component, separating *the lip is shifted*
from *the lip is misshapen*. The offset is **not purely a miscentring measure** — an asymmetric lip
moves its own centroid — so it cannot distinguish the two causes alone.

**[LITERATURE] Spearman is the comparable statistic.** The protocol's finding is that human and
SymNose scores correlated well *when ranked best to worst*. That is a rank correlation, so quoting
PCC against it would be the fifth instance of the R2 error.

> **Stated before the run, not after: a modest correlation is the expected outcome.** SymNose
> measures upper-lip asymmetry; the label is an Asher-McDade composite over **nasal form, nasal
> symmetry, nasolabial profile and vermillion border**. Only one of four components is the thing this
> index measures. An index that tracked the whole composite would be the surprising result and would
> warrant asking what else it had picked up. Writing this down in advance is what stops a modest
> number being read as a failure of the method.

**An automated SymNose is a different instrument, not the manual one with a caveat attached.** Its
properties include an unrecoverable oral fissure, an always-fires geometric split, and a discriminant
calibrated on lighter skin. **That framing must survive into the write-up intact** — these are
characteristics of the instrument being reported, not disclaimers about a component of it.

**Patch-fed probes — built, not yet run.** Three schemes at G2 plus the geometry probe, each five
seeds, each differing from its neighbour in exactly one field (asserted by test). Patch extraction
reuses Phase 2's mapping unchanged, and the per-image mapping is applied at **both** geometries —
at G2 the trapezium bbox is the whole square but the *content* box is not, because aspect ratios
span a factor of two.

> **The pooling limit, and what it does *not* limit.** Mean pooling discards every relationship
> between regions, which is half of what the patch scheme exists to test — SR-GNN replaces pooling
> with message passing in Phase 7. So the **absolute numbers** are a lower bound on what patches can
> do, and a pooled arm losing to the whole-image probe shows that *mean-pooled patch embeddings* do
> not beat whole-image embeddings, not that region structure does not help.
>
> **That limit does not extend to the scheme comparison.** Pooling is identical across grid, anatomy
> and random, so a difference between them is a difference in **placement**. Q4 has a placement half
> and a relational half; these arms answer the placement half cleanly and defer the relational half.
> Do not discount the scheme comparison along with the absolute numbers.

Three things Phase 4 owes beyond its arms:

- **The freeze guard.** `tests/test_frozen_apparatus.py` hashes the frozen modules against their
  state at `phase3-freeze` and fails naming any that moved. Touching the apparatus becomes
  deliberate rather than accidental.
- **A determinism re-run for the patch path.** Patch extraction and pooling are new code between the
  images and the harness, and pooling is exactly where a nondeterministic reduction hides. One
  patch-fed arm, run twice at the same seed, byte-identical, ~~compared against
  `GATE1_REFERENCE`~~ — **compared on `predictions_sha256` between the two runs**.
  `harness.py` asked for this itself; the arms simply arrived earlier than Phase 7.

  *[CORRECTED 2026-09-05 — the struck phrasing is preserved above and is **unexecutable**.
  `train/phase3.GATE1_REVERIFIED["still_owed"]` states it: "the patch path — PLAN asks for it
  and its phrasing is unexecutable (`reference_arm` is False for any patch arm, so no
  comparison happens); the honest test is two same-seed runs on `predictions_sha256`." The
  obligation is unchanged and still owed; only the comparison it names is corrected.]*
- **Sensitivity to partitioning** — `configs/p4_partition.yaml`, the last Phase 4 item. **Not**
  fold-assignment variance: see §4.3 for why that cannot be measured here, and why the seed route
  would have returned a silent zero.

  The same arm at **5, 6 and 10 folds plus leave-one-patient-out**, all through the frozen generator
  with `shuffle=False` intact — genuinely different partitions, deterministic stratification, no
  balance degradation, and `shuffle=False` not re-litigated. LOPO is the reference point: 237 folds,
  deterministic, **no assignment choice at all**, so `lopo_minus_5fold` bounds how much the quoted
  number depends on partitioning, from one run of minutes over frozen embeddings.
  - **The confound is reported, not hidden.** Fold count changes training-set size too (~190 / ~213
    / 236), so the spread mixes partitioning with how much each model sees. `training_sizes` is
    reported per partitioning so it is visible which is moving.
  - **Read `range` against the seed band first** (§4.12: SD 0.0137, claimable delta 0.017 at five
    seeds). A spread inside the band is not a finding; one comparable to it matters for every result
    in the project, since each is quoted from a single partitioning.
  - **This is additive.** `cleft_v1`'s folds are unchanged on disk and remain the ones every ladder
    arm uses; the alternative partitionings exist only for this measurement.

**Phase 5 — SCUT preparation. ✅ COMPLETE 2026-07-30.**

| deliverable | state |
|---|---|
| masked SCUT at G1 and G2 | **done** — parity measured at 5,499 faces, sheet reviewed and approved |
| checkpointing | **done** — verified against a real Run:AI pause, the Phase 6 entry gate |
| univariate relevance diagnostic | **done** — §4.11.1 |
| asymmetry synthesis | ~~**built, verified, PARKED — not run.**~~ **RAN 2026-08-30 as Phase 17 arm A** — see the correction at Stage F below |
| SCUT copy verification | **done** — exit criterion 9, byte-identical on both machines |

*Exit:* met. **Nothing in Phase 6 depends on the synthesis**, which is why parking it closes the
phase rather than holding it open.

> **[KNOWN UNRESOLVED — performance, not correctness] The full synthesis build is pathologically slow
> on the cluster and the cause is not established.** `p5-synth-1` managed fewer than 250 faces in 233
> minutes while sustaining ~25 cores — about **1,400 CPU-seconds per face against 0.30 s/face on
> identical code locally**, roughly 185× in wall time and 4,600× in CPU. Memory was flat at 789 MB, so
> nothing accumulates.
>
> What profiling settled: the build is **flat in n** — nothing scales with faces already processed,
> and the observed throughput is equally consistent with a constant per-face cost. The hot path is
> **single-threaded elementwise numpy over a 33 MB intermediate** inside `apply_tps` (broadcast 28%,
> reduce 34%, kernel 35%), *not* linear algebra: `np.linalg.solve` is 7% of the build and the GEMMs
> are 2.9% of `apply_tps`. That is a memory-bandwidth-bound workload, which is what degrades on a node
> at load 406.
>
> What it did not settle: 99.98% of the cluster's CPU is overhead, which is the signature of
> spin-waiting rather than slow work, and that could not be reproduced on 12 uncontended cores. The
> leading hypothesis is BLAS thread-pool spin — ~33 BLAS calls per face waking a 64-thread pool — and
> the untried decisive tests are `OPENBLAS_NUM_THREADS=1` and reading `utime` versus `stime` from
> `/proc/<pid>/stat`, which separates userspace spinning from kernel time.
>
> **Recorded as a known issue rather than a blocker**, because the only arm that consumes the output
> is parked. If Stage F is ever revived this is the first thing to resolve.

**Original Phase 5 scope, for the record.** Masked variants at both geometries; asymmetry synthesis.

**✅ Exit criterion 9 met [MEASURED 2026-07-30]: the cluster and laptop SCUT copies are
byte-identical.** 5,500 files and 168,214,846 bytes of `Images/`, 5,500 and 3,805,312 of
`facial landmark/`, plus four MD5s — all recorded in `scut.dataset.VERIFIED_COPIES` and **recomputed
on the laptop copy before being written down**, because a pasted hash nobody recomputed is what §7
warns about. `dataset.verify_against_record` re-derives the criterion rather than recalling it.
Cluster copy at `/home/user/codex/scut/SCUT-FBP5500_v2`.

> **Two further independent confirmations that `CM152.pts` carries `count = 0`**, after the `od` read
> and the parse failure — four in total, failing in different ways. Its MD5
> `f1d3ff8443297732862df21dc4e57262` is the MD5 of four zero bytes. And an 86-point `.pts` is
> `4 + 86·8 = 692` bytes, so `5499·692 + 4 = 3,805,312` — **exactly** the recorded landmark total,
> which means the byte count is only consistent with 5,499 complete point sets and exactly one short
> file. The two are not the same statement: the MD5 pins *that* file's contents, the arithmetic pins
> that every *other* file is complete and no second truncated file is hiding in the set.

> **[R2] The first comparison reported a 135,168-byte difference in both directories, and it was not
> data.** `du -sb` counts the directory itself; a files-only sum does not. The tell was that the
> discrepancy was **identical across two directories of unrelated content**, which file bytes cannot
> do. Harmless, and recorded because it is another instance of comparing two different quantities —
> and because tooling should not invite it. `dataset.BYTE_SUM_COMMANDS` pins the POSIX and PowerShell
> commands that produce the *same* quantity and names `du -sb` as the one not to use.

> **[MEASURED 2026-07-30] Three defects found in Phase 5 code, all of the same shape: a check that
> could not fail.** Recorded here because the pattern is now six instances deep and the tally in R7
> is where it is tracked.
>
> 1. **The "G2 has zero white" invariant was a patch-geometry constant.** See §4.4's note and
>    `docs/FROZEN_KNOWN_STALE.md` §3. The real defect was the mask's coordinate frame — applied
>    square-relative after staging where cleft bakes it content-relative before. Masked SCUT now
>    reproduces the frozen composition bit-for-bit, asserted by test.
> 2. **`placement.midline_x` was displaced off the facial midline on every face.** It averaged the
>    eye midpoint with `mean(x)` of nose indices 60–65, described as running "down the bridge …
>    close to the midline". **The nose contour is a trace, not a span**: 60–65 descend the
>    image-left flank, 0.117–0.238 of crop width from the midline. The estimate sat **−0.0915 crop
>    widths image-left on 300 of 300 faces** (~10.6px on a 115px crop). Now `(eye + 66 + 76) / 3`,
>    chosen on **held-out** structures — the first ranking was circular, since the mirror-pair
>    centre minimises its own score — and 6.6× better, winning on 98.8% of faces. **The fixture
>    `a_face` endorsed the broken version** (0.010 against the fix's 0.077) because it alternates
>    sides within each group, so the left-flank bias cancels by construction. `a_realistic_face`,
>    built from the measured trace, is now used for midline tests.
> 3. **The TPS deformation was neither local nor unilateral, and its locality check said it was.**
>    The anchor set was ten points; the thin-plate kernel is globally supported, so **48 of 86
>    landmarks moved, including both eyes**, while the check reported 1e-10 — because every point it
>    measured was an anchor. Fixed by anchoring every non-target landmark, and locality is now
>    measured **in the pixels**, where a by-construction pass is unavailable. A coordinate-frame slip
>    was found with it: the warp was applied ~(x0, y0) from the anatomy it was fitted to.
>
> Also measured, and recorded as a bound rather than fixed: **the usable deformation magnitude is
> capped by anchor spacing** at ~0.038, because the cupid's bow peak is pinned ~0.09 crop widths from
> the anchored midline dip. Whether to unanchor the midline — a real unilateral cleft deviates the
> philtrum — is an **open decision**, since it changes what the instrument measures.

*Also here, and not in the original list:* **univariate feature relevance**
(`configs/p5_feature_relevance.yaml`). Each of the mirror-difference index's 22 per-region asymmetry
features correlated against the mean label **on its own**, ranked, Spearman and Pearson. No fitting,
no seeds, nothing tuned on it.

It exists to decide **how the nose tip enters the pipeline**, which converges three ways: supervision
suggested considering it; the Phase 2 cleft contact sheet shows apparent tilt worst at the nose tips,
less at the eyes, least at the lips — a gradient head roll cannot produce, so probably nasal
deviation; and Asher-McDade scores nasal form and nasal symmetry as two of its four components. It
also matches SymNose's nasion and nose-tip roundels. **Alignment anchor, weighted region, or TPS
deformation target are three different builds**, and this says whether the tip's asymmetry tracks the
grade at all before any of them is started.

**Univariate rather than ridge coefficients**, because the region features are strongly correlated
and correlated predictors make individual coefficients unstable — they can flip sign on a resample
while predictions barely move. **No multiple-comparison correction is applied**: with 22 features
some will correlate by chance, so it is a ranking that informs a design decision, not a set of
claims, and anything acted on needs its own arm under §4.3.
*Also here:* **checkpointing** (§2.7). Run-directory identity survives a pause; training does not.
This is a prerequisite for Phase 6, not a Phase 7 nicety.

**Phase 6 — Pretraining.** *Entry gate:* ✅ `docs/VERIFY_resume.md` completed 2026-07-28 — `JOB_UUID`
survives a pause. *Second entry gate:* checkpointing from Phase 5, without which a long job that
pauses silently restarts at epoch 0.

**Phase 7 — The ladder.**

**Phase 8 — Explainability.** Node attention as the primary instrument (§4.8).

**Phase 9 — Prototypes.** Medoids per grade, 3-class primary, validated by leave-one-out and
bootstrap stability.

~~**Phase 10 — Write-up.**~~ Work stream 2 of the programme: SymNose (Pigott) → Strange → Bakaki → this.

> **[CORRECTED 2026-09-05 — the heading above is superseded; the work-stream sentence still
> stands.]** **Phase 10 is the CleftGNN replication** (`phase10.py`, opened 2026-08-16 under
> `PLAN_AMENDMENT_2026-08-13`), and the write-up is not Phase 10. Part 5 was never updated
> when the amendment chain replaced it — seven sequence amendments have run since
> (`phase11.PHASE_SEQUENCE_RENUMBERED` through the seventh), moving the write-up to 14, then
> 17, and onward. **The durable statement, and the one that does not age: the write-up runs
> LAST BY RULE, not at a fixed number** (`phase15.py`, the fourth amendment's note). Quoting a
> number here is what made this stale in the first place.

---

# PART 6 — ARM LIST [REASONED]

| stage | arms | varies | answers |
|---|---|---|---|
| A baselines | 4 | ~~landmark~~, ~~SymNose~~, mirror-difference, random-patch, whole-image | the bar |
| B view | 3 | frontal / basal / both | cost of the missing view |
| C geometry | 2 | G1 vs G2 | does the fill matter |
| D init ladder | 12 | 4 backbones × 3 inits | Q1, Q2 |
| **E patch scheme** | **3** | **grid vs anatomy vs random** | **prior knowledge vs data-driven** |
| ~~F training set~~ | ~~2~~ | ~~TSTR vs CV-on-237~~ | **struck — see below** |
| G label | 3 | mean vs median vs LDL | label formulation |
| **total** | **~29 configs** | | |

~~**Stage F is struck [DECIDED 2026-07-30]. The TSTR arm is parked: built, verified, not
run.**~~

> **[CORRECTED 2026-09-05 — the arm was unparked and it RAN.]** The 2026-07-30 decision is
> preserved above as the state at its date. The parked design ran on **2026-08-30** as Phase
> 17 arm A — `phase17.PHASE_17_RULINGS["arm_a"]` is "**the parked TPS design, as parked**" —
> in run `p17_family_analysis__3d0ca56c__p17-family-analysis`, three arms × five seeds, zero
> real patient images in training, all 237 pure test (`phase17.PHASE_17_CLOSING`).

It depended on the synthetic asymmetry set, and **that arm's own record said in advance that it was
not predicted to succeed**, for four reasons recorded before anything ran (`scut/synthesis.py`,
`LIMITATIONS`): no scar, the magnitude-to-grade mapping is an assumption rather than a measurement,
the midline is anchored, and — the specific one — the deformation **cannot reach the philtrum**, which
the univariate relevance diagnostic ranks first of 22 at |Spearman| 0.151.

~~**So an interpretable result was never available in either direction.**~~ A null would have been
consistent with all four limitations and would not have distinguished between them; a positive would
have needed explaining against a prediction that said it should not happen. An arm whose two
outcomes are both uninterpretable does not earn cluster time, and — the sharper point — **that was
knowable before the run, which is exactly why the prediction was written down first.** The value of
recording it in advance is realised here, in a decision not to spend the time, rather than in a
post-hoc reading of a number.

> **[CORRECTED 2026-09-05 — the argument above is preserved and it was overtaken by a
> measurement.]** The arm ran on 2026-08-30, and **the contradiction is the finding**.
> `scut/synthesis.py` `LIMITATIONS["prediction_reckoned_2026_08_30"]` records it:
> *"the arm ran 2026-08-30 and the outcome contradicts the prediction in substance — the
> philtrum-anchored, scar-free synthesis transferred to PCC 0.2334 on the real 237, ~93% of
> the probe's 0.2520, statistically indistinguishable (0/5 intervals exclude zero; ledger
> `p17-a-vs-probe`)."*
>
> Three things survive the correction and one does not. **Survives:** the reading discipline —
> the record words this **PARITY, NOT SUCCESS**, because the point estimate runs slightly
> against A and the cohort cannot resolve it. **Survives:** the pre-commitment's value, but
> inverted — it is realised in a contradiction that could be *recognised as one*, not in a
> decision to save the time. **Survives:** the honesty note the record attaches, that
> "not predicted to succeed" was never given a numeric definition, so the contradiction is
> **substantive rather than formal**. **Does not survive:** "an interpretable result was never
> available in either direction." It was available, it was interpretable, and it went against
> the prediction this module wrote down first.

**The code is kept, not deleted**, in the same posture as the parked segmentation module (Part 5
§3.2): its tests keep running (a parked module whose suite is skipped quietly stops working), and if
time allows near the end it can be run without rebuilding anything. What is dropped is the *claim*
that the ladder includes a TSTR replication — leaving it in the arm list while it is unbuilt is the
inconsistency the Stage A amendment already had to correct once.

**Stage A shed two arms, both on the same supervisory decision [AMENDED 2026-07-29].** `landmark`
and `SymNose` are struck: automating SymNose requires **landmark detection**, and that method class
is outside the project's remit because the supervision material rejected Bakaki's landmark work for not being deep
learning (Part 5, §3.2). The landmark baseline was the same method class, so ruling out one ruled out
both — but the arm list kept counting them for another 170 lines. **An arm list promising something
the scope excludes is exactly the inconsistency that survives into a write-up.**

What Phase 4 actually delivered in their place is the **mirror-difference index**: landmark-free,
segmentation-free, and the only symmetry baseline that could be built under the remit. Six shipped
configs, four distinct baseline arms.

Stage E is now the anatomy-versus-grid comparison rather than an assumed design. Random-patch is the
Charm-motivated control.

**[REASONED]** One-factor-at-a-time controls cost but **cannot detect interactions** — and §4.4
identifies a likely one between geometry and patch scheme. Revisit before Phase 7 commits compute.

**Multiply by the seed count, now measured (§4.12).** Five seeds resolve 0.017 PCC and ten resolve
0.012; one resolves 0.038, which is the whole observed spread. Choose per arm from what that arm
needs to claim, and re-measure the band for any arm whose training procedure differs substantially
from the frozen probe. Compute is not the constraint it was assumed to be — a ten-seed sweep of a
frozen-backbone arm costs about a minute on a 0.16 GPU fraction (§2.7).

---

# PART 7 — WHEN THINGS GO WRONG

1. Is the tree clean and the run in `runs/keeper/`?
2. Does the run directory have `config.yaml`, `env.json`, `inputs.json`, `code/`?
3. Did the runtime assertions fire? One that did not fire is not evidence it passed.
4. Does the comparison differ in exactly one factor?
5. Is the delta larger than the seed band from Phase 3?
6. **Is the surprising number the same quantity** as what you compare it against?
7. Has the assumption been measured, or only reasoned about? Check its tag.
8. **Are you reading a specification or an example?** (R9)
9. If defects have accumulated with unclear provenance, stop patching and rebuild from a green gate.
10. **Does a comment inside a frozen module look wrong?** Check
    [`docs/FROZEN_KNOWN_STALE.md`](FROZEN_KNOWN_STALE.md) before doing anything. Frozen prose cannot
    be corrected in place — that is byte-identical, to the guard, with changing the code — so known
    errors are registered there with the correction. Add an entry rather than editing the module.
