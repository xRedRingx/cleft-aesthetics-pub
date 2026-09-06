"""Write the Phase 7 ladder configs from the derived arm list.

    PYTHONPATH=src python scripts/generate_ladder_configs.py [--check]

Nineteen configs, one per distinct run (``cleft.ladder.distinct_runs``). The
arm list is derived from the stages that define it, so the configs cannot
drift from the lattice the tests assert -- and ``--check`` re-derives them and
diffs, so a hand edit to a generated config is caught rather than silently
kept.

----------------------------------------------------------------------------
HASHES: REAL WHERE ALREADY VERIFIED, PLACEHOLDER OTHERWISE
----------------------------------------------------------------------------
A path is immutable, so the same artifact must declare the same rollup in
every config (``tests/test_smoke_run.py``). This therefore **reuses hashes
already verified in shipped configs** rather than emitting placeholders it
would then have to be told about: ``cleft_v1`` and ``staged_v1`` are declared
everywhere, and the SR-GNN native masked-G2 embedding set and its checkpoint
were verified for the Phase 6 seed band. Whatever is genuinely unknown gets
the all-zeros placeholder, and guard 3 refuses those runs until they are
filled from ``scripts/declare_ladder_inputs.py``.

**Checkpoint paths cannot be derived here.** A pretraining run directory
carries its job id, which this machine has never seen, so inventing one would
put a fabricated path where a reader expects provenance. Unresolved
checkpoints are written as an obvious ``UNRESOLVED_GLOB`` marker holding the
glob that resolves them; the batch script does the resolution on the cluster,
where the directories exist.

This never writes a hash it did not read from an existing config -- for the
reason ``declare_inputs.py`` gives at length: a tool that fills in whatever is
on disk makes guard 3 agree with reality by definition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from cleft import embedding_plan, ladder  # noqa: E402
from cleft.embeddings import expected_variant  # noqa: E402
from cleft.train.extract import checkpoint_input_name  # noqa: E402

PLACEHOLDER = "0" * 64
CLUSTER_ROOT = "/home/user/codex/cleft-aesthetics"
EMBEDDINGS_VERSION = "embeddings_v1"

#: Written into the path of any checkpoint whose run directory this machine
#: cannot know. Deliberately not a plausible path.
UNRESOLVED = "UNRESOLVED_GLOB"


def known_declarations() -> tuple[dict, dict]:
    """What the shipped configs already declare, keyed two ways.

    ``by_path`` -- path -> rollup. The cross-config invariant makes this safe
    and necessary: a path's contents cannot differ, so a hash verified once is
    the hash everywhere, and emitting a placeholder beside an already-verified
    value would put the repo in the exact disagreeing state that invariant
    catches.

    ``by_name`` -- input name -> (path, rollup), for the ``ckpt_*`` inputs.
    A checkpoint's run directory carries a job id this machine cannot derive,
    but the naming convention is fixed, so an already-declared checkpoint can
    be resolved by NAME even though its path could never have been guessed.
    ``p6_extract_embeddings.yaml`` declares all thirteen, verified on the
    cluster, which is why almost nothing here needs resolving at all.
    """
    by_path: dict[str, str] = {}
    by_name: dict[str, tuple] = {}
    for path in sorted((REPO / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in payload.get("inputs") or []:
            rollup = str(entry.get("rollup_sha256", ""))
            declared = str(entry["path"])
            if len(rollup) != 64 or rollup == PLACEHOLDER:
                continue
            by_path.setdefault(declared, rollup)
            name = str(entry["name"])
            if name.startswith("ckpt_") and UNRESOLVED not in declared:
                by_name.setdefault(name, (declared, rollup))
    return by_path, by_name


def inputs_for(arm: dict, known: dict, checkpoints: dict) -> list[dict]:
    def declare(name: str, path: str) -> dict:
        return {
            "name": name,
            "path": path,
            "rollup_sha256": known.get(path, PLACEHOLDER),
        }

    # The module's rule, not a copy of it: the same string decides which
    # artifact the arm reads and which registry entry declares its hash, and
    # two implementations of it would eventually name two different sets.
    set_name = ladder.embedding_set_name(arm)
    version = arm.get("embeddings_version", EMBEDDINGS_VERSION)
    embeddings_path = (
        f"{CLUSTER_ROOT}/data/embeddings/{version}/{set_name}"
    )
    # Two sources for an embedding rollup: the registry (pasted once from the
    # cluster) and any config that already declared it. Where both know a set
    # they MUST agree -- a path is immutable, so two different hashes for it
    # means one is wrong and neither can be preferred silently.
    registered = ladder.DECLARED_EMBEDDING_HASHES.get(set_name)
    from_configs = known.get(embeddings_path)
    if registered and from_configs and registered != from_configs:
        raise SystemExit(
            f"CONFLICT for {set_name}: the registry says {registered[:12]}... "
            f"and a shipped config says {from_configs[:12]}.... A path's "
            "contents cannot differ; resolve it before generating."
        )

    entries = [
        declare("manifest_v1", f"{CLUSTER_ROOT}/data/manifests/cleft_v1"),
        declare("staged_v1", f"{CLUSTER_ROOT}/data/staged/staged_v1"),
        {
            "name": "embeddings",
            "path": embeddings_path,
            "rollup_sha256": registered or from_configs or PLACEHOLDER,
        },
    ]
    variant = expected_variant(arm["init"], arm["geometry"])
    if variant is not None:
        scheme = (
            arm["region_scheme"] if arm["backbone_kind"] == "graph" else None
        )
        stem = f"p6_pretrain_{arm['backbone']}"
        if scheme:
            stem += f"_{scheme}"
        stem += f"_{variant}"
        name = checkpoint_input_name({
            "backbone": arm["backbone"],
            "variant": variant,
            "pretrain_scheme": scheme,
        })
        # Resolved by NAME from an already-verified declaration where one
        # exists -- the path could never have been derived here, but the
        # naming convention is fixed and p6_extract_embeddings.yaml declares
        # all thirteen. Only a checkpoint nobody has declared falls back to
        # the glob marker.
        if name in checkpoints:
            resolved, rollup = checkpoints[name]
            entries.append(
                {"name": name, "path": resolved, "rollup_sha256": rollup}
            )
        else:
            entries.append(
                declare(
                    name,
                    f"{CLUSTER_ROOT}/runs/keeper/p6/{UNRESOLVED}/"
                    f"{stem}__*/pretrained.npz",
                )
            )
    return entries


def task_for(arm: dict, entries: list[dict]) -> dict:
    by_role = {entry["name"] for entry in entries}
    checkpoint = next(
        (name for name in by_role if name.startswith("ckpt_")), None
    )
    common = {
        "manifest_artifact": "manifest_v1",
        "staged_artifact": "staged_v1",
        "embeddings_artifact": "embeddings",
        "geometry": arm["geometry"],
        "label": arm["label"],
        "max_epochs": 40,
        "patience": 5,
        "inner_val_frac": 0.2,
        "monitor": "inner_val_mse",
        "seeds": arm["seed_list"],
    }
    if checkpoint:
        common["checkpoint"] = checkpoint

    if arm["task"] == "train_graph_cv":
        return {
            "kind": "train_graph_cv",
            **common,
            "backbone": arm["backbone"],
            "init": arm["init"],
            "region_scheme": arm["region_scheme"],
            "trainable": arm["trainable"],
            # Emitted only where it must be declared -- AG-Net. A field
            # present on every graph config would read as a knob anyone may
            # turn, when for three arms it records a measured constraint and
            # for the rest it is not a question.
            **(
                {"deterministic": arm["deterministic"]}
                if arm["deterministic"] is not None
                else {}
            ),
            # The 12M-stack rate, as the Phase 6 graph arms used.
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "batch_size": 16,
        }
    return {
        "kind": "train_cv",
        **common,
        "backbone": arm["backbone"],
        "init": arm["init"],
        "trainable": arm["trainable"],
        # The head rate the 0.2529 bar was measured with.
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "batch_size": 32,
    }


def header(arm: dict, entries: list[dict]) -> str:
    """The provenance preamble, describing THIS config's actual state.

    **The two notes below are conditional on the entries, and that is a fix
    rather than a refinement.** They used to be emitted from the arm alone --
    the unresolved-checkpoint note for every pretrained arm and the
    placeholder note for every arm -- so a config carrying a verified
    checkpoint hash still said its path was an UNRESOLVED_GLOB placeholder.
    A config IS the provenance record (PLAN §2.3); one that misdescribes its
    own inputs is the record disagreeing with itself, and a reader who checks
    and finds a real hash learns to skip the preamble.
    """
    unresolved = "" if not any(UNRESOLVED in e["path"] for e in entries) else (
        "#\n"
        "# The checkpoint path is an UNRESOLVED_GLOB PLACEHOLDER: a pretraining\n"
        "# run directory carries its job id, which the generating machine has\n"
        "# never seen. scripts/declare_ladder_inputs.py resolves it on the\n"
        "# cluster and prints the real path with its hash.\n"
    )
    placeholder = "" if not any(
        e["rollup_sha256"] == PLACEHOLDER for e in entries
    ) else (
        "#\n"
        "# An all-zeros rollup below is a PLACEHOLDER and guard 3 will refuse\n"
        "# the run until it is filled from scripts/declare_ladder_inputs.py.\n"
    )
    return (
        f"# PHASE 7 — STAGE {arm['stage']}: {arm['name']}\n"
        "#\n"
        f"# GENERATED by scripts/generate_ladder_configs.py from cleft.ladder.\n"
        "# Do not hand-edit the task block: --check re-derives and diffs, and a\n"
        "# hand edit would break the one-factor property the lattice test\n"
        "# asserts over every comparison this arm takes part in.\n"
        "#\n"
        f"# backbone {arm['backbone']} ({arm['backbone_kind']}), init "
        f"{arm['init']}, geometry {arm['geometry']}"
        + (f", scheme {arm['region_scheme']}" if arm["region_scheme"] else "")
        + f", label {arm['label']}\n"
        f"# {arm['seeds']} seeds — the {arm['backbone_kind']} regime\n"
        "# (ladder.SEEDS_BY_KIND). This arm reports its OWN seed SD, and every\n"
        "# comparison it takes part in is thresholded from the two arms' own\n"
        "# spreads — PLAN §4.12.1, inherited never. No band is quoted here: the\n"
        "# twelve measured Stage D arms span SD 0.0142 to 0.0567.\n"
        + unresolved
        + placeholder +
        "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{arm['name']}.yaml\"\n\n"
    )


def render(arm: dict, known: dict, checkpoints: dict) -> str:
    entries = inputs_for(arm, known, checkpoints)
    payload = {
        "schema_version": 1,
        "phase": "p7",
        "tier": "keeper",
        "seed": arm["seed_list"][0],
        "inputs": entries,
        "task": task_for(arm, entries),
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return header(arm, entries) + body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="verify the shipped configs match what would be generated",
    )
    args = parser.parse_args(argv)

    known, checkpoints = known_declarations()
    drifted, written = [], 0
    for arm in ladder.distinct_runs():
        path = REPO / "configs" / f"{arm['name']}.yaml"
        text = render(arm, known, checkpoints)
        if args.check:
            if not path.is_file():
                drifted.append(f"{path.name}: missing")
            elif path.read_text(encoding="utf-8") != text:
                drifted.append(f"{path.name}: differs from the derived arm")
        else:
            path.write_text(text, encoding="utf-8")
            written += 1

    summary = ladder.summary()
    if args.check:
        for line in drifted:
            print(f"  DRIFT {line}")
        print(
            f"{summary['distinct_runs']} configs checked, {len(drifted)} drifted"
        )
        return 1 if drifted else 0

    filled = sum(
        1
        for arm in ladder.distinct_runs()
        for entry in inputs_for(arm, known, checkpoints)
        if entry["rollup_sha256"] != PLACEHOLDER
    )
    total = sum(
        len(inputs_for(arm, known, checkpoints)) for arm in ladder.distinct_runs()
    )
    print(
        f"wrote {written} configs ({summary['cells']} cells, "
        f"{summary['total_fits']} fits)\n"
        f"  {filled}/{total} input hashes already verified; "
        f"{total - filled} need scripts/declare_ladder_inputs.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
