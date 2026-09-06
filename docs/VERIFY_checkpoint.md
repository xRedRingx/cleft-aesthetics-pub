# Verifying that checkpointing survives a real Run:AI pause

**Status: ✅ VERIFIED 2026-07-30. Phase 6's entry gate is closed.**

| | state |
|---|---|
| the mechanism, on a stub loop | verified — `tests/test_checkpoint.py`, 19 tests |
| the mechanism, on the real build pipeline | verified — byte-identical, 232.9 MiB, two interruptions |
| **a real Run:AI pause** | **VERIFIED on the re-run — all nine artifact files byte-identical** |

**Gate run 2 passed:** an uninterrupted build against one paused from the Run:AI
console and resumed produced **all nine artifact files byte-identical**, including
`faces.json` — the file that failed run 1. Reported by the maintainer and recorded
in the Phase 6 brief's entry-gate section.

> **The figures for run 2 are recorded here as reported, not as observed from the
> run directory.** If the comparison output or the `RESUMING …` log line is still
> available it belongs in this file, in the pattern the rest of this document
> uses: cluster figures arrive by paste, and a number that only exists in a
> message is one nobody can re-derive.

---

## Gate run 1 — FAILED, and it is worth reading before the re-run

**Eight of nine files byte-identical. `faces.json` was not.** 211 rows for a
200-face build, 11 duplicate stems, `n_measured` 634 against 600 and
`n_deformations_measured` 633 against 600.

**Every array was correct.** The resume rolled the arrays back properly and the
reprocessed faces overwrote the same array positions — so the pixels were right
and *the index beside them was wrong*. That is the worst shape this class of
defect can take: an artifact **misreporting its own contents**, claiming 211 faces
where 200 exist, with nothing failing loudly. Anything reading `faces.json` to
enumerate the set, index into the arrays, or check side balance would have worked
from a corrupted record. The array files being identical is *not* the same
statement as the artifact being correct.

`n_measured` and `n_deformations_measured` were downstream symptoms, not separate
defects: both are recomputed from the journals on resume, so the bad journal
carried straight into `metrics.json`.

### The fix: derive, do not record twice

The rollback truncated the journals to `output_rows[...]`, a count saved *beside*
`step`. The two are meant to be equal, and in the failed run they were not — by
exactly the 11 faces between the last checkpoint and the pause.

**Recording the same fact twice and trusting both is the error.** There is one
truth here — face *N*'s row is row *N* — so the journal length is now computed
from `step` alone, and the recorded counters are used only to *report* a
disagreement, never to act on one. A short journal is refused outright rather than
resumed into.

`synthesis.assert_index_describes_arrays` then runs **before the rename**: row
count matches the face count, no duplicate stems, survival rows equal
faces × deformed magnitudes. Cheap, total, true of every build, and it converts
this class from a silent corruption into a refusal to publish.

### What could not be reproduced, and what was proven instead

**The failure did not reproduce locally**, even with two interruptions leaving 15
and 17 journal rows past their checkpoints — including a second kill rolling back
a journal a previous resume had already rewritten. So how `output_rows` came to
disagree with `step` is still unknown.

Rather than guess, the *state* was forced directly: a checkpoint with `step=80`
and `output_rows.faces=91`, which is the only shape that yields the observed
numbers (`final = keep + (n − step)`, so `keep = step + 11`). Under the old rule
that publishes **131 rows for a 120-face build** — the same +11 signature. Under
the new rule it published **120 rows, 120 unique**.

So the fix is proven against the observed failure shape without the mechanism
being known, and the new resume log now prints both counts and flags a
disagreement, which will name the mechanism if it recurs:

```
RESUMING at face N of M; journals rolled back to X face and Y survival rows
  [recorded output_rows.faces=Z disagreed with step=N; the derived count wins]
```

**If you still have gate run 1's `log.txt`, its `RESUMING …` line would settle
it** — the old form recorded `(K face records durable)`, and `K` against `N` is
the whole story.

This file closes what [`VERIFY_resume.md`](VERIFY_resume.md) deliberately left
open. That one settled that `JOB_UUID` survives a pause, and named what it did
not settle:

> Run-directory identity survived the pause. The training did not. […] **A Phase 6
> pretraining job paused at hour three would silently restart at hour zero, with
> no symptom but wall time.**

Checkpointing was built for that. It has never met a real pause.

---

## The specific failure this tests for

**Not "does the job finish".** A job with no checkpointing also finishes after a
pause — it just does the whole thing again. Guard 2 recognises the resume and
reattaches the directory, so every visible signal looks right while the work is
silently redone. Wall time is the only symptom, and on a long job nobody is
watching a clock.

**The pass condition is therefore byte-identity, not completion.** A resumed build
must produce the same artifact as an uninterrupted one, and it must reach it
without redoing the faces it had already written.

---

## What has been verified locally, and how

`scripts/verify_resume_synthesis.py`, on the real synthesis pipeline — real
images, TPS warps, staging, memmaps and the JSONL journals. Not the stub loop.

1. build 120 faces uninterrupted → artifact A
2. build 120 faces, **kill the process** once its own checkpoint passes face 20
3. resume to completion → artifact B
4. compare every file in A and B byte for byte

**Result 2026-07-30 (after the gate-1 fix): PASS, with TWO interruptions.**
Killed at the checkpoint for face 100 with **115 journal rows — 15 beyond the
checkpoint** — then killed *again* during the resume at face 150 with **167 rows —
17 beyond**, so the second rollback operated on a journal a previous resume had
already rewritten. Both artifacts' indexes came out **200 rows, 200 unique**, and
all 9 files were identical.

**Earlier result, kept because it shows what a weaker test misses:** a single kill
at face 100 with 106 journal rows also passed — and passed the gate-1 defect
straight through, because it never produced a disagreeing counter. The partial
`resume_paused.inprogress/` survived carrying `checkpoint.npz`, both journals and
the eight memmaps; the **final** artifact path did not exist, so a half-built
artifact could not be mistaken for a complete one; and after the resume all 9
files were identical — **232.9 MiB compared**.

It runs at the **same shape as the cluster gate** — 200 faces, four magnitudes,
both geometries, `checkpoint_every: 50` — so the two differ only in how the
interruption arrives.

Two details that decide whether this proves anything:

- **A process kill, not an exception.** A real pause deletes the pod; an exception
  inside the process would unwind cleanly and prove less.
- **The kill waits a few seconds after the checkpoint appears**, and the script
  **refuses to continue** unless the journal has more rows than the checkpoint
  claims. Killing the instant a checkpoint lands leaves the journals with exactly
  the checkpointed count, so the resume would have nothing to truncate and the
  rollback — the part that can actually be wrong — would never run. The test would
  pass without testing it.

```bash
python scripts/verify_resume_synthesis.py
```

---

## What is still open: the real pause

Everything above ran on one machine, killing a local process. It does not test
Run:AI recreating a pod, NFS visibility of a partially written memmap, or the
entrypoint's path back into the same run directory.

### Procedure — the gate runs SMALL and FIRST, then the artifact is built once

**Not two full builds.** Verifying the mechanism with two 90-minute builds would
cost three hours to test a mechanism, and — the stronger objection — it would put
the verification **after** the long run. A 90-minute job on a preemptible queue
may be paused whether or not anyone plans it, so checkpointing should be proven
before that happens rather than discovered during it.

| step | config | ~time |
|---|---|---|
| 1. reference | `p5_synthesis_gate_a.yaml` (200 faces) | ~3 min |
| 2. paused and resumed | `p5_synthesis_gate_b.yaml` (200 faces) | ~3 min |
| 3. the artifact | `p5_asymmetry_synthesis.yaml` (5,499 faces) | ~90 min |

~96 minutes for the same evidence, with the gate closed before the long job
starts.

1. **Run `p5_synthesis_gate_a.yaml` to completion, uninterrupted.** This is the
   reference. The two gate configs are **identical except for `out_version`**, and
   a test asserts it — if they differed in anything else, a byte difference
   between their artifacts would have a second possible cause and the gate would
   not isolate the pause.

   > **Both runs on the cluster, in the pinned image.** The reference cannot be
   > built on the laptop and compared against a cluster resume: the TPS fit goes
   > through `np.linalg.solve` and therefore whatever LAPACK the platform ships, so
   > a Windows-versus-Linux byte difference would be a platform difference
   > **wearing a checkpoint failure's costume**, and the gate could not tell them
   > apart. Same machine, same image, one variable: the pause.

2. **Run `p5_synthesis_gate_b.yaml` and pause it from the Run:AI console**
   mid-build — after at least one `checkpoint at face N` line in the log, and
   before it publishes. `checkpoint_every` is **50** here against the real build's
   250: at 250 with 200 faces there would be one checkpoint and the journal
   truncation would barely be exercised; at 50 there are four, so a pause between
   any two of them tests the path the 5,499-face build uses. **That is a
   configuration difference, not a code difference** — the gate and the artifact
   run identical code.

3. **Resume.** The log must show `RESUMING at face N of 200`.

4. **Compare.** `synth_gate_b` must hash identically to `synth_gate_a`, file for
   file.

5. **Only then build the artifact**: `p5_asymmetry_synthesis.yaml`, 5,499 faces,
   uninterrupted, to `synth_v1`.

   **[MEASURED 2026-07-30, CORRECTED]** 40 faces spread through the set time at
   **0.301 s/face** on the laptop, so the full build projects to **~28 minutes**
   with a checkpoint every ~1.3 minutes at `checkpoint_every: 250`.

   > **An earlier figure of 0.975 s/face in this file was wrong, and the ~90
   > minute projection built on it.** It was measured while the test suite ran in
   > the background. Clean re-measurement gives 0.301 s/face spread and 0.324 on
   > the first 40 — and every SCUT image is 350×350, so a claim made here that
   > early faces are cheaper than the cohort mean was also wrong. Correcting it
   > matters because the contaminated number made the cluster's per-face cost look
   > three times closer to the laptop's than it is.

   **On the cluster the full build is far slower than either figure and the cause
   is unresolved** — see `synthesis.CLUSTER_PERFORMANCE_UNRESOLVED`. That does not
   affect this gate, which passed at 200 faces, and the arm consuming the full
   build is parked.

### What each outcome means

- **Byte-identical, and the log shows the resume.** The gate is closed and Phase 6
  can run a long job.
- **Byte-identical, but the log shows no resume.** The pause landed before the
  first checkpoint, or the working directory did not survive. Nothing is proven —
  re-run with a later pause.
- **Not byte-identical.** Do not proceed to Phase 6. Compare `faces.json` first: a
  duplicated or missing face means the journal truncation is wrong; identical
  `faces.json` with differing `.npy` means the memmap flush is not ordered before
  the checkpoint.
- **The resumed run aborts on the artifact-exists guard.** The rename published
  the artifact before the pause, so the pause was too late to be a test.
- **The log shows a resume but the journals had nothing to roll back.** If the
  pause landed within a second or two of a checkpoint, the journals may already
  have matched it exactly, so the truncation ran on nothing. This is the same trap
  the local script guards against by waiting before killing. Check
  `RESUMING at face N (M face records durable)` — if `M` equals `N` and the pause
  was that tight, the rollback was not exercised; pause further from a checkpoint
  and repeat.

### Record the result here, in the pattern of `VERIFY_resume.md`

State the outcome, the job it was checked on, and — if it passed — what it did
*not* verify. That last part is what made `VERIFY_resume.md` useful: it is the
reason checkpointing exists at all.

---

## Design notes worth knowing before reading a failure

**The artifact is built in `<version>.inprogress` and renamed on completion.** The
final path never exists half-built, so the immutability guard on the final name
stays exactly as strict as it was while a resume can still reuse partial work.

**Journals are truncated to the checkpointed row count on resume.** `faces.jsonl`
and `survival.jsonl` are appended per face, so a kill leaves rows the checkpoint
does not know about. Those faces are about to be produced again; keeping the rows
would duplicate them in the artifact.

**Arrays are flushed before the checkpoint is written**, never after — otherwise a
checkpoint could claim faces whose pixels had not reached disk, and the resume
would skip them.

**This build consumes no streaming randomness.** Aspect ratios and sides are drawn
for every face up front from `config.seed`, so face *N*'s work depends only on
`(stem, ratio, side)`. The RNG state is checkpointed anyway because the contract
is uniform and a streamed version would need it — and because recording a constant
costs nothing next to discovering later that it was not one.
