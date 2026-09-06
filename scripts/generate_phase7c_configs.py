"""Write the Phase 7C arm configs from the pre-registration.

    PYTHONPATH=src python scripts/generate_phase7c_configs.py [--check]

Seven configs, one per arm in ``cleft.phase7c.arms()``. Generated rather than
hand-written for the reason ``generate_ladder_configs.py`` gives: the arms are
DERIVED from the factors that define them, so the configs cannot drift from
the comparisons the tests assert -- and ``--check`` re-derives and diffs, so a
hand edit is caught rather than silently kept.

**The strengths are not written into the configs.** They live in
``phase7c.PHOTOMETRIC``, ``GEOMETRIC``, ``REGION_AWARE`` and ``BLEND``, where
the suite checks them. A config carries the four policy FLAGS and nothing
else, because a pre-registration a config could edit is not one.

Hashes are copied from the arm this phase tunes -- ``p7_d1_vit_b16_imagenet_g1``
-- since every Phase 7C arm reads the same manifest and staged images. It reads
no embedding artifact at all: the whole point is that features are recomputed
from augmented pixels each epoch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from cleft import phase7c  # noqa: E402

CLUSTER_ROOT = "/home/user/codex/cleft-aesthetics"

#: Read from the arm this phase tunes, not retyped. Both are declared in every
#: shipped config and verified on the cluster; the cross-config invariant in
#: tests/test_smoke_run.py makes a mistyped one fail on the laptop.
SOURCE_CONFIG = "p7_d1_vit_b16_imagenet_g1.yaml"
SHARED_INPUTS = ("manifest_v1", "staged_v1")


def shared_inputs() -> list[dict]:
    payload = yaml.safe_load(
        (REPO / "configs" / SOURCE_CONFIG).read_text(encoding="utf-8")
    )
    by_name = {entry["name"]: entry for entry in payload["inputs"]}
    missing = [name for name in SHARED_INPUTS if name not in by_name]
    if missing:
        raise SystemExit(f"{SOURCE_CONFIG} no longer declares {missing}")
    return [dict(by_name[name]) for name in SHARED_INPUTS]


def header(arm: dict, policy: dict, prefix: str) -> str:
    families = [
        name for name in ("photometric", "geometric", "rotation", "region_aware")
        if arm[name]
    ]
    gate = (
        "#\n"
        "# **THIS ARM IS A GATE, NOT A RESULT.** It must reproduce the 0.2520\n"
        "# baseline within tolerance. If it does not, stop -- nothing\n"
        "# downstream is interpretable.\n"
        "#\n"
        "# The fixed budget does not change that. With no augmentation the\n"
        "# features are identical every epoch, so the trajectory is\n"
        "# deterministic and best-checkpoint selection picks the same epoch\n"
        "# whether the loop stops at 6 or runs to 30. The budget is INERT here\n"
        "# and load-bearing in the other six.\n"
        "#\n"
        "# What it tests is NARROW: the augmenting backbone at identity\n"
        "# policy. Live-versus-artifact extraction is already settled at this\n"
        "# exact cell -- 0.25206 (p3_train_cv, live) against 0.2520 (the D1\n"
        "# artifact arm), four decimals on the mean AND the SD. This is a gate\n"
        "# on new code, not on a question this project answered twice.\n"
        if arm["index"] == 0 else ""
    )
    ten_seed = (
        "#\n"
        "# ##################################################################\n"
        "# **DO NOT RUN. [DECIDED 2026-08-04] This round cannot change any\n"
        "# verdict, and the phase is closed as unresolved.**\n"
        "#\n"
        "# The paired BCa withdrew all nine of Phase 7C's verdicts: not one\n"
        "# of the 45 per-seed intervals excludes zero\n"
        "# (phase7c.PAIRED_BCA_WITHDREW_EVERY_VERDICT).\n"
        "#\n"
        "# Condition 1 is a UNIVERSAL over the seeds present -- every\n"
        "# interval must exclude zero -- so it is anti-monotone in the seed\n"
        "# set: failing at five FORCES failure at ten. And the five failing\n"
        "# intervals reappear unchanged inside a ten-seed run, because a\n"
        "# seed's fit is count-independent and the bootstrap is seeded from\n"
        "# the seed. Ten seeds fails BY CONSTRUCTION, whatever the five new\n"
        "# seeds do. phase7c.TEN_SEED_ROUND_NOT_RUN.\n"
        "#\n"
        "# Kept rather than deleted because the design reasoning below is\n"
        "# still right about seed counts generally. It is not right about\n"
        "# this phase, which needs a larger cohort and not more seeds.\n"
        "# ##################################################################\n"
        "#\n"
        "# **TEN SEEDS, not five.** PLAN §4.12: the seed count follows from\n"
        "# what the arm needs to claim. Arms 0, 1 and 4 are the only arms in\n"
        "# the two contrasts whose verdict turns on n --\n"
        "# photometric-vs-identity and region-photometric (1 vs 4). The\n"
        "# geometric arms clear the five-seed threshold by 3-4x and are NOT\n"
        "# raised.\n"
        "#\n"
        "# **The first five seeds must come back BIT-IDENTICAL** to the five\n"
        "# already on disk: the fits are budget- and count-independent, so\n"
        "# this is a determinism re-verification obtained for free. If they\n"
        "# differ, stop — nothing in this phase is interpretable.\n"
        "#\n"
        "# Ten in ONE config, never five merged with the existing five: one\n"
        "# SD over ten seeds computed where it is reported, and no chance of\n"
        "# pairing a v3 vector with a v4 one. phase7c.TEN_SEED_ROUND.\n"
        if "s10" in prefix else ""
    )
    matched = (
        "#\n"
        "# **ROUND 2 — THE MATCHED-EPOCH VARIANT.** Three epochs, not thirty.\n"
        "#\n"
        "# Best-checkpoint selection under augmentation partly selects the\n"
        "# luckiest DRAW rather than the best-converged model: each epoch is a\n"
        "# different perturbation, so a fold's inner-val score depends on model\n"
        "# quality AND on how kind that draw was to 38 patients.\n"
        "#\n"
        "# That confound runs AGAINST the augmented arms — a late epoch wins on\n"
        "# inner-val-specific luck, and the model there is heavily overfit, so\n"
        "# the reported OOF is degraded. Only augmented arms can select late;\n"
        "# arm 0 is deterministic and always selects epoch 1. So it could be\n"
        "# inflating a negative result, and cannot be waved through.\n"
        "#\n"
        "# Three epochs puts every arm before divergence (the curves put it at\n"
        "# about epoch 3) and shrinks selection from best-of-thirty to\n"
        "# best-of-three. **A reduction, not an elimination** — the frozen\n"
        "# run_fold always keeps the best epoch, so no-selection is not\n"
        "# reachable through config. phase7c.SELECTION_CONFLATES_DRAW_QUALITY.\n"
        if "m3" in prefix else ""
    )
    return (
        f"# PHASE 7C — ARM {arm['index']}: {config_name(arm, prefix)}\n"
        "#\n"
        "# GENERATED by scripts/generate_phase7c_configs.py from cleft.phase7c.\n"
        "# Do not hand-edit the task block: --check re-derives and diffs, and a\n"
        "# hand edit would break the one-factor property the comparisons rest on.\n"
        "#\n"
        f"# policy: {', '.join(families) if families else 'identity (no augmentation)'}\n"
        "#\n"
        f"# FIXED {policy['max_epochs']}-EPOCH BUDGET, no early stopping: patience\n"
        "# equals max_epochs, so the harness's break is unreachable and what\n"
        "# remains is a fixed budget with best-checkpoint selection — gate 4's\n"
        "# amendment, through config alone. Identical across all seven arms so\n"
        "# convergence speed cannot confound the comparison.\n"
        "#\n"
        "# [MEASURED 2026-08-03] The FIRST run early-stopped at epoch 1 in every\n"
        "# arm, so the model saw each image once under one random transform. That\n"
        "# is not augmentation, and it produced a plausible monotone ordering\n"
        "# measuring something else entirely. phase7c.ONE_EPOCH_IS_NOT_AUGMENTATION.\n"
        + ten_seed
        + matched
        + gate +
        "#\n"
        "# The STRENGTHS are not here. They are pre-registered in cleft.phase7c\n"
        "# and checked by the suite -- a pre-registration a config could edit is\n"
        "# not one. This file carries the four policy flags and nothing else.\n"
        "#\n"
        "# No embeddings_artifact: features are recomputed from augmented pixels\n"
        "# every epoch, which is the whole point. `trainable: head` is unchanged\n"
        "# -- the backbone is still frozen and the head is still 769 parameters.\n"
        "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{config_name(arm, prefix)}.yaml\"\n\n"
    )


def render(arm: dict, inputs: list[dict], policy: dict, prefix: str) -> str:
    task = {
        "kind": "train_cv",
        "manifest_artifact": "manifest_v1",
        "staged_artifact": "staged_v1",
        "geometry": arm["geometry"],
        "label": arm["label"],
        "backbone": arm["backbone"],
        "trainable": "head",
        # Fixed budget, best-checkpoint selection, no early stopping.
        # `patience >= max_epochs` makes the harness's break unreachable, so
        # this is gate 4's amendment through config alone, with no change to
        # the frozen loop. See phase7c.EPOCH_POLICY / MATCHED_EPOCH_POLICY.
        "max_epochs": policy["max_epochs"],
        "patience": policy["patience"],
        "inner_val_frac": 0.2,
        "monitor": policy["monitor"],
        "seeds": arm["seeds"],
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "batch_size": 32,
        "augmentation": {
            "photometric": arm["photometric"],
            "geometric": arm["geometric"],
            "rotation": arm["rotation"],
            "region_aware": arm["region_aware"],
        },
    }
    payload = {
        "schema_version": 1,
        "phase": "p7c",
        "tier": "keeper",
        "seed": arm["seeds"][0],
        "inputs": inputs,
        "task": task,
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return header(arm, policy, prefix) + body


#: The two rounds. The second exists because best-checkpoint selection under
#: augmentation partly selects the luckiest draw
#: (``phase7c.SELECTION_CONFLATES_DRAW_QUALITY``), and that confound runs
#: AGAINST the augmented arms -- so a negative result needs to survive both.
#:
#: The third and fourth are the TEN-SEED round, arms 0/1/4 only
#: (``phase7c.TEN_SEED_ROUND``). Both rounds get them, because the two
#: contrasts that turn on the seed count are contrasts BETWEEN the rounds, and
#: raising only the one whose nulls flip would promote the sensitivity
#: analysis by a second route.
#:
#: **Imported, not restated.** ``phase7c.CONFIG_ROUNDS`` is the record the
#: suite checks the shipped directory against; a second copy here would be a
#: second place for it to drift.
ROUNDS = phase7c.CONFIG_ROUNDS


#: The naming rule lives beside the round list it names, in ``phase7c``.
config_name = phase7c.config_name


#: Where a keeper run lands on the cluster. A run directory is
#: ``<config-stem>__<sha8>__<job-id>``, and neither the SHA nor the job id is
#: derivable on the laptop -- hence the globs below.
RUNS_ROOT = f"{CLUSTER_ROOT}/runs/keeper/p7c"

#: The round the paired BCa closes exit criterion 6 for. ``selected30`` is the
#: PRIMARY round (``phase7c.ROUND_2_IS_ROUND_1_TRUNCATED``); ``matched3`` is a
#: sensitivity analysis whose verdicts are not reported, so it does not need a
#: condition-1 pass. Change this to emit the twin.
PAIRED_ROUND = "selected30"
PAIRED_PREFIX = ""


def paired_config_name() -> str:
    return f"p7c_paired_{PAIRED_ROUND}"


def resolved_paired_inputs() -> dict:
    """What the shipped paired config already declares, keyed by input name.

    **Regeneration must not wipe a resolved glob or a pasted hash.** Those
    arrive from the cluster and cannot be re-derived here, so an unconditional
    render would silently revert them and ``--check`` would report drift
    forever after. Values are carried forward by NAME; the input LIST is still
    derived, so a new arm or seed shows up as a fresh glob with a placeholder.
    """
    path = REPO / "configs" / f"{paired_config_name()}.yaml"
    if not path.is_file():
        return {}
    # **Resolved entries only.** [MEASURED 2026-08-04, in the sibling script]
    # carrying placeholders forward too means a generator change cannot reach
    # an already-written config -- a pinned job id produced no change because
    # the old unpinned glob was preserved. A placeholder is re-derivable by
    # definition; preserving it freezes whatever rule wrote it.
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        entry["name"]: entry
        for entry in payload.get("inputs", [])
        if entry.get("rollup_sha256") != "0" * 64
    }


def paired_inputs() -> list[dict]:
    """Thirty-five OOF vectors, one per arm per seed, as GLOBS.

    **The globs cannot be resolved here and must not be guessed.** A run
    directory carries a job id; ``declare_ladder_inputs.py`` resolves them on
    the cluster and prints the real paths with their hashes.

    **Only the SHA is globbed. The job id is pinned.**
    [MEASURED 2026-08-03] The first version globbed both, and
    ``declare_ladder_inputs.py`` refused all 35: three rounds of these arms
    are on disk under the same config stems, and v1 (the augmenter never ran)
    and v2 (every arm stopped at epoch 1) are both VOID. Their prediction
    files would have loaded, paired and yielded plausible intervals.

    ``phase7c.V3_JOB_IDS`` names the round, so each glob now resolves to
    exactly one directory. The SHA stays a wildcard because it identifies the
    CODE rather than the round, and pinning it would need an edit whenever a
    future round ran at a different commit.
    """
    existing = resolved_paired_inputs()
    entries = []
    for arm in phase7c.arms():
        stem = config_name(arm, PAIRED_PREFIX)
        key = phase7c.result_key(arm["name"])
        job = phase7c.V3_JOB_IDS[arm["index"]]
        for seed in arm["seeds"]:
            name = (
                f"{phase7c.OOF_INPUT_PREFIX}{key}"
                f"{phase7c.OOF_SEED_SEPARATOR}{seed}"
            )
            entries.append(existing.get(name) or {
                "name": name,
                "path": (
                    f"{RUNS_ROOT}/{stem}__*__{job}"
                    f"/seed_{seed}__predictions.csv"
                ),
                "rollup_sha256": "0" * 64,
            })
    return entries


def paired_provenance(inputs: list[dict]) -> str:
    """The paragraph describing where these vectors came from.

    **Conditional on whether the inputs are resolved**, because a header that
    announces placeholders over 35 real hashes is worse than no header: it is
    the kind of stale comment a reader trusts and then has to un-learn. The
    two states say different true things, so the generator writes whichever
    one is true when it runs.
    """
    unresolved = [e for e in inputs if e["rollup_sha256"] == "0" * 64]
    if unresolved:
        return (
            f"# **THE {len(inputs)} HASHES BELOW ARE ALL-ZERO PLACEHOLDERS, and\n"
            "# the paths are GLOBS.** A run directory carries a job id no\n"
            "# laptop can derive, so neither can be filled in here. Guard 3\n"
            "# will refuse this run until they are real, which is the\n"
            "# placeholder working rather than a mistake. Resolve them on the\n"
            "# cluster with declare_ladder_inputs.py and paste the real paths\n"
            "# with their hashes.\n"
        )
    shas = sorted({e["path"].split("/")[-2].split("__")[1] for e in inputs})
    return (
        f"# **RESOLVED 2026-08-03.** All {len(inputs)} vectors are declared at\n"
        f"# real paths with verified hashes, under SHA {', '.join(shas)} with\n"
        "# the v3 job ids (phase7c.V3_JOB_IDS). Guard 3 hash-verifies every\n"
        "# one of them before the task reads a single row.\n"
    )


def render_paired() -> str:
    inputs = paired_inputs()
    payload = {
        "schema_version": 1,
        "phase": "p7c",
        "tier": "keeper",
        "seed": phase7c.SEEDS[0],
        "inputs": inputs,
        "task": {
            "kind": "phase7c_paired",
            "round": PAIRED_ROUND,
            "seeds": list(phase7c.SEEDS),
            "n_boot": 10000,
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return (
        f"# PHASE 7C — EXIT CRITERION 6: the per-seed paired BCa\n"
        "#\n"
        "# GENERATED by scripts/generate_phase7c_configs.py from cleft.phase7c.\n"
        "#\n"
        "# **THIS RUN FITS NOTHING.** Every vector it reads is already on disk.\n"
        "# PLAN §4.3 claims a delta only if BOTH conditions hold: (1) the paired\n"
        "# BCa CI over patients excludes zero, and (2) the delta exceeds combined\n"
        "# seed uncertainty. Phase 7C reported all nine verdicts from condition 2\n"
        "# alone. This computes condition 1, which\n"
        "# phase7c.NOT_A_SEARCH['carried_over'] promised in writing.\n"
        "#\n"
        "# The conditions are ANDed, so this can only WITHDRAW claims. It cannot\n"
        "# rescue the three nulls and is not permitted to try — what it settles\n"
        "# for them is whether they are underpowered or empty.\n"
        "# phase7c.PAIRED_BCA_IS_THE_MISSING_CONDITION.\n"
        "#\n"
        f"# ROUND: {PAIRED_ROUND} — the PRIMARY round. Round 2 (matched3) is a\n"
        "# sensitivity analysis whose verdicts are not reported, and 23 of its 35\n"
        "# fits ARE these fits. phase7c.ROUND_2_IS_ROUND_1_TRUNCATED.\n"
        "#\n"
        + paired_provenance(inputs) +
        "#\n"
        "# **Three rounds of these arms share these config stems**, and v1 and v2\n"
        "# are both VOID. Their prediction files load, pair, and produce\n"
        "# entirely plausible intervals, so which run each path names is the\n"
        "# load-bearing fact in this file.\n"
        "#\n"
        "# [MEASURED 2026-08-03] The first version of this config globbed the\n"
        "# job id as well as the SHA. Every one of the 35 matched three runs,\n"
        "# and declare_ladder_inputs.py refused all 35 rather than choosing --\n"
        "# had it picked, it would have paired against a run that never applied\n"
        "# its policy. The job ids were then pinned from phase7c.V3_JOB_IDS,\n"
        "# leaving only the SHA to glob, and all 35 resolved to one run each.\n"
        "#\n"
        "# The remaining guard is in the task, not here: it refuses vectors\n"
        "# that do not reproduce the recorded round on BOTH mean and SD, so a\n"
        "# path edited to a void run fails before any interval is computed.\n"
        "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{paired_config_name()}.yaml\"\n\n"
    ) + body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    inputs = shared_inputs()
    drifted, written, expected = [], 0, 0
    emitted: list[tuple[Path, str]] = []
    for round_ in ROUNDS:
        for arm in phase7c.arms_at(round_["n_seeds"], only=round_["only"]):
            emitted.append((
                REPO / "configs" / f"{config_name(arm, round_['prefix'])}.yaml",
                render(arm, inputs, round_["policy"], round_["prefix"]),
            ))
    emitted.append((
        REPO / "configs" / f"{paired_config_name()}.yaml", render_paired()
    ))

    for path, text in emitted:
        expected += 1
        if args.check:
            if not path.is_file():
                drifted.append(f"{path.name}: missing")
            elif path.read_text(encoding="utf-8") != text:
                drifted.append(f"{path.name}: differs from the derived arm")
        else:
            path.write_text(text, encoding="utf-8")
            written += 1

    if args.check:
        for line in drifted:
            print(f"  DRIFT {line}")
        print(f"{expected} configs checked, {len(drifted)} drifted")
        return 1 if drifted else 0

    fits = sum(
        len(arm["seeds"])
        for round_ in ROUNDS
        for arm in phase7c.arms_at(round_["n_seeds"], only=round_["only"])
    )
    print(f"wrote {written} configs -- {fits} fits across {len(ROUNDS)} rounds")
    for round_ in ROUNDS:
        arms_here = phase7c.arms_at(round_["n_seeds"], only=round_["only"])
        print(
            f"  {round_['prefix'] or '(round 1)':10} {len(arms_here)} arms x "
            f"{round_['n_seeds']} seeds = "
            f"{len(arms_here) * round_['n_seeds']} fits"
        )
    print(
        f"  paired BCa: {paired_config_name()}.yaml, "
        f"{len(paired_inputs())} declared vectors, fits nothing"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
