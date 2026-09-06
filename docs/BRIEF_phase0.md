# BRIEF — PHASE 0: FOUNDATION

**Repo:** `cleft-aesthetics` (NEW, empty, separate from everything that exists)
**Governing document:** `CLEFT_PIPELINE_PLAN_v1.md` — read Parts 1, 2 and 3 before starting.
**Phase:** 0 only. Do not start Phase 1. Do not write any model, training, or data code.

---

## 0. ROLE AND BOUNDARIES

- Laptop implementation only. **No commits, no pushes, no cluster launches, no cleft data access.**
  The user does all of those.
- **Port nothing from the old repository.** Not one file. Knowledge carries over only as tests.
  If you find yourself opening the old repo to copy something, stop — write the test instead.
- The clinical cohort never leaves EHU infrastructure. Nothing in Phase 0 touches it; Phase 0 runs
  entirely on synthetic fixtures.

---

## 1. WHY THIS PHASE EXISTS

The previous attempt failed for workflow reasons, not modelling reasons. The Docker image baked
the code in → builds were slow → commits were avoided → every run had `dirty_tree: true` → when
the same arm produced PCC 0.218 at one SHA and 0.150 at another, code drift could not be
separated from non-determinism. Permanently, for every result in the project.

Phase 0 makes that class of failure impossible. Nothing else is built until it holds.

---

## 2. FIRST TASK — DETERMINE THE ENVIRONMENT (report before building)

Before writing the Dockerfile, find out what the current working image actually contains, and
report back:

- python version, torch version, CUDA version, timm version
- whether the existing image runs on RTX Blackwell 6000 (it does — the user has been training on
  it, so whatever is in there works)
- the existing Dockerfile: **does it `COPY` the code in?** Report the exact `COPY` lines and the
  `WORKDIR`. This confirms or refutes the root-cause diagnosis.
- the contents of `.dockerignore` (it once silently excluded a Python package)

**If the existing image already has a working environment, we pin its digest and never build one.**
That would remove image builds from this project entirely. Report before assuming a rebuild is
needed.

---

## 3. REPO SKELETON

```
cleft-aesthetics/
  src/cleft/
    __init__.py
    provenance/          run context, hashing, guards
    config/              schema + loader
    data/                (empty in Phase 0)
    geometry/            (empty in Phase 0)
    models/              (empty in Phase 0)
    train/               (empty in Phase 0)
    eval/                metrics only, in Phase 0
    run.py               entry point
  configs/
    _schema.py or schema.yaml
    smoke.yaml           trivial config that exercises the whole path
  tests/
  scripts/
    entrypoint.sh
  docker/
    Dockerfile
    requirements.txt
  .github/workflows/
    tests.yml
    build-image.yml      workflow_dispatch only
  docs/
    PLAN.md              copy of CLEFT_PIPELINE_PLAN_v1.md
  .dockerignore
  .gitignore
  pyproject.toml
  README.md
```

---

## 4. WRITE THE TESTS FIRST

These encode facts already paid for. Write them before the code they guard; they should fail
first, then pass.

**`tests/test_guards.py`**
- dirty tree + `runs/keeper/` target → raises
- dirty tree + `runs/dev/` target → allowed, and `env.json` records `dirty_tree: true`
- existing run directory → raises (no clobber, ever)
- declared input hash ≠ actual hash → raises
- clean tree + `runs/keeper/` → succeeds

**`tests/test_provenance.py`**
- run directory contains all of: `config.yaml`, `env.json`, `inputs.json`, `log.txt`, `code/`
- `config.yaml` in the run directory is byte-identical to the source config
- `env.json` captures git SHA, dirty flag, python/torch versions, GPU name, hostname, timestamp
- hashing is deterministic: same directory hashed twice gives the same rollup
- hashing is order-independent but content-sensitive: changing one byte changes the rollup

**`tests/test_config.py`**
- unknown key → raises (a typo must not be silently ignored)
- missing required key → raises
- a config round-trips: load → dump → load gives an identical object
- **no scientific setting may be overridden by a CLI flag** — assert `run.py` accepts only
  `--config` and `--out`

**`tests/test_metrics.py`** (metrics module only, no models)
- PCC matches `scipy.stats.pearsonr` on random data
- Spearman matches `scipy.stats.spearmanr`
- QWK 3-cat quadratic matches `sklearn.metrics.cohen_kappa_score(weights='quadratic')`
- MAE, RMSE match closed form
- BCa bootstrap: reproducible under a fixed seed; CI brackets the point estimate; a paired delta
  of a variable with itself gives a CI containing 0

**`tests/test_normalization.py`** — guards a known Stage-1 bug
- normalization values come from the model factory, never hardcoded ImageNet constants
- a test that greps the source tree for the literal ImageNet mean/std triple and fails if it
  appears outside the factory

**`tests/fixtures/`** — synthetic only. A tiny fake git repo, a fake data directory with known
hashes, small random arrays for the metrics. No real data of any kind.

---

## 5. THE PROVENANCE MODULE — the heart of Phase 0

`src/cleft/provenance/context.py`, roughly:

```python
class RunContext:
    """Created once at run start. Owns the run directory and all guards."""
    def __init__(self, config_path: Path, out_root: Path, tier: str):  # tier: "keeper" | "dev"
        ...
```

Responsibilities, in order:

1. Load and validate the config (unknown keys are fatal).
2. Capture environment: `git rev-parse HEAD`, `git status --porcelain` → dirty flag, python,
   torch, timm, CUDA, GPU name, hostname, UTC timestamp, image digest if available from env.
3. **Guard 1** — if dirty and `tier == "keeper"`: raise. If dirty, force tier to `dev`.
4. Compute the run directory name: `<config-stem>__<sha8>__<UTC timestamp>`.
5. **Guard 2** — if that directory exists: raise.
6. Hash every input artifact the config declares. **Guard 3** — mismatch against declared hashes:
   raise.
7. Create the directory, copy the config verbatim, write `env.json` and `inputs.json`.
8. Create `code/` as a git worktree at the captured SHA. If the tree is dirty, skip the worktree
   and record why in `env.json` — dev runs do not get a worktree.
9. Provide `ctx.path(name)` for writing outputs, and a `SHAREABLE` / `CLUSTER-ONLY` tier marker on
   each written file.

Hashing: SHA-256 per file, sorted by relative path, rolled up into a single digest. Record file
count and total bytes alongside, so a truncated directory is visible.

---

## 6. DOCKER AND CI

**`docker/Dockerfile`** — environment only:
- `COPY docker/requirements.txt .` and **nothing else**. No application code, ever.
- `WORKDIR` is the NFS project path on the cluster.
- Pin versions in `requirements.txt`; no floating `latest`.

**`.github/workflows/tests.yml`** — runs `pytest` on push and PR. **Does not build an image.**
Target: under 60 seconds.

**`.github/workflows/build-image.yml`** — `workflow_dispatch` only, tagged with the hash of
`requirements.txt`. Never on push.

**`scripts/entrypoint.sh`** — takes `--sha` and `--config`; fetches, verifies the SHA exists,
creates a worktree, runs from it. Fails loudly if the SHA is unknown.

---

## 7. THE SMOKE CONFIG

`configs/smoke.yaml` must exercise the entire path without any real work — declare a synthetic
input, "compute" a trivial metric, write outputs. This is what proves the contract holds, and
it becomes the template every real config follows.

---

## 8. EXIT CRITERIA — all must hold

1. `python -m cleft.run --config configs/smoke.yaml` produces a fully compliant run directory.
2. Running it twice produces a second directory and **never** overwrites the first.
3. With a dirty tree, `runs/keeper/` is refused and `runs/dev/` is used with the flag recorded.
4. With a clean tree, `runs/keeper/` succeeds and `code/` is a valid worktree at the right SHA.
5. Corrupting one byte of a declared input makes the next run refuse to start.
6. `pytest` is green and completes in under 60 seconds.
7. The Dockerfile contains no application code.

---

## 9. REPORT BACK

When Phase 0 is done, report:
- the environment findings from §2, especially whether the old Dockerfile copied code in
- whether an image rebuild is needed at all, or the existing digest can be pinned
- anything in the contract that proved impractical, with what you did instead
- the pytest runtime

Then stop. Phase 1 (manifest and labels) is a separate brief, and it depends on decisions the
user has not yet made.

---

## 10. THINGS THAT WILL BE TEMPTING AND ARE WRONG

- Adding a `--lr` or `--epochs` flag "just for convenience". Configs only. This is how the last
  pipeline lost the ability to say what produced a result.
- Making the dirty-tree guard a warning instead of a hard failure.
- Copying a metrics or dataset file from the old repo because it "already works". It may; the
  point is that we can no longer tell, and the test suite is how we regain that.
- Building the image on every push because it is easier to configure. That is the exact root cause
  of the previous failure.
