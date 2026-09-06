# Verifying that the resume design actually holds

**Status: VERIFIED 2026-07-28. Outcome A.** The assumption held:

> The job identifier in `JOB_UUID` is the same value before and after a Run:AI
> pause/resume.

It was checked on the gate-2 seed sweep rather than on `smoke_keeper.yaml` — the
sweep was already running, already long enough to interrupt, and a real workload
is better evidence than a contrived one. `JOB_UUID` was
`8e7c7452-568b-4c15-b4a9-cd0809572ca7` at **both** attempts. The recreated pod
mapped back to the same run directory, `env.json` stayed the first attempt's
record, and `resumes.json` carries attempt 1 with `resumed: true`.

So `CLEFT_JOB_ID` is belt-and-braces rather than load-bearing, and Phase 6 is
unblocked on this count.

---

## What this did NOT verify — read before the first long job

**Run-directory identity survived the pause. The training did not.**

The second pod re-ran all ten seeds from the beginning, and nothing noticed,
because the sweep takes about a minute. Guard 2 did exactly what it was designed
to do — it recognised the resume and did not abort — but there was no checkpoint
for the run to pick up, so "resume" meant "start over in the right directory".

At a minute that is invisible. **A Phase 6 pretraining job paused at hour three
would silently restart at hour zero, with no symptom but wall time.** That is the
original failure this whole design exists to prevent, displaced one layer down:
the directory is now right and the work is still lost.

Checkpointing is therefore a **Phase 5/6 prerequisite** (PLAN §2.7, Part 5), and
it is the thing to build before a long job — not this file's assumption, which is
now settled.

---

## What is known

Measured inside a running workload on 2026-07-27:

| variable | value | note |
|---|---|---|
| `JOB_UUID` | `6fb7f0fa-…` | a real UUID |
| `jobUUID` | same | camelCase duplicate |
| `RUNAI_JOB_NAME` | `test3` | a workload **name**, reusable |
| `JOB_NAME`, `jobName` | `test3` | same |
| `RUNAI_JOB_ID` | **absent** | does not exist on this cluster |
| `RUNAI_JOB_UUID` | **absent** | does not exist on this cluster |
| `RUNAI_NUM_OF_GPUS` | `0.11` | a GPU *fraction*, not a device |

`src/cleft/provenance/context.py` prefers UUIDs over names, because a workload
name can be reused across workloads and a UUID cannot. A run id built from a name
emits a provenance warning into `env.json`.

## The test

It needs one pause and one resume, and the answer is visible in the filesystem.

1. Launch the tracked keeper config on the cluster:

   ```bash
   scripts/entrypoint.sh --sha <sha> --config configs/smoke_keeper.yaml
   ```

2. While it runs, **pause** the workload in the Run:AI console. Wait for the pod
   to terminate.

3. **Resume** it. Wait for the new pod to start and finish.

4. Look at the run root:

   ```bash
   ls -d runs/keeper/p0/smoke_keeper__*
   ```

### Outcome A — one directory (the assumption holds)

```
runs/keeper/p0/smoke_keeper__<sha8>__6fb7f0fa-…/
  env.json        attempt 0
  resumes.json    attempt 1
```

`resumes.json` exists and contains a second attempt whose `env` may name a
different node or GPU. Nothing else to do: change the status line at the top of
this file to VERIFIED and record the date.

### Outcome B — two directories (the assumption is false)

```
runs/keeper/p0/smoke_keeper__<sha8>__6fb7f0fa-…/
runs/keeper/p0/smoke_keeper__<sha8>__9c2e14bb-…/
```

The identifier did not survive. **Stop and revisit the design** — do not start a
long job. The fix is to supply a stable id from the workload spec instead, which
is part of the pod template and therefore survives pod recreation by
construction:

```yaml
env:
  - name: CLEFT_JOB_ID
    value: p6-swin-masked-seed1337
```

`CLEFT_JOB_ID` is first in the lookup order, so it overrides everything else. Its
one hazard is copy-paste: cloning a workload without changing the value would
make the clone look like a resume of the original. Guard 2 catches the dangerous
half of that — a resume whose config or SHA differs from the recorded one aborts
— but two genuinely identical runs (the Phase 3 determinism gate) would collide.
Give each a distinct value.

## Keeping this procedure honest

The procedure above stays in the repo because it is worth re-running whenever the
cluster's Run:AI version changes — the variable names it depends on are not
contractual, and `RUNAI_JOB_ID` and `RUNAI_JOB_UUID` already vanished once.

A standing caveat, unchanged by the result: the test suite **fakes** the
environment variable, so it proves the *code* does the right thing with a stable
id. It never proved the id was stable. Only the cluster could, and now has.
