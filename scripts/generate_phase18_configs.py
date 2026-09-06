"""Phase 18 config -- the metric-space analysis over the locked arms.

ONE config, derived from ``phase18.ARM_LIST_LOCKED`` itself -- the
input list and the task's arm list come from the same constant, so the
config cannot drift from the lock and no glob exists anywhere in the
resolution chain. 67 run-directory inputs (64 with real paths, 3
PENDING on the maintainer's p15-probe pastes) plus the resolved manifest.

    python scripts/generate_phase18_configs.py           # write
    python scripts/generate_phase18_configs.py --check    # drift only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CLUSTER_ROOT = "/home/user/codex/cleft-aesthetics"
PLACEHOLDER = "0" * 64

sys.path.insert(0, str(REPO / "src"))
from cleft import phase18  # noqa: E402
from cleft.phase18 import NAMED_CONTRASTS  # noqa: E402

NAME = "p18_metric_space.yaml"
PROBE_CONFIG = "p7_d1_vit_b16_imagenet_g1.yaml"


def _named(payload: dict, name: str) -> dict:
    return dict(next(e for e in payload["inputs"] if e["name"] == name))


def _known_hashes() -> dict:
    """Every verified hash any shipped config declares, keyed by path.

    A path is immutable, so a hash verified anywhere is THE hash --
    the cross-config agreement guard
    (test_the_same_artifact_declares_the_same_hash_in_every_config)
    enforces exactly this, and fired when the first version of this
    generator emitted placeholders for run directories the p17 family
    and addendum configs had already declared. Carrying the known
    values is not a convenience: emitting a placeholder for a KNOWN
    hash is a disagreement."""
    known: dict = {}
    for shipped in sorted((REPO / "configs").glob("*.yaml")):
        if shipped.name == NAME:
            continue
        payload = yaml.safe_load(
            shipped.read_text(encoding="utf-8")
        ) or {}
        for candidate in payload.get("inputs") or []:
            rollup = str(candidate.get("rollup_sha256") or "")
            if len(rollup) == 64 and set(rollup) != {"0"}:
                known.setdefault(str(candidate.get("path")), rollup)
    return known


def _carried(config_name: str, input_name: str, path: str,
             known: dict | None = None) -> dict:
    """The path-pinned carry, extended cross-config: this config's own
    shipped copy first, then any sibling's verified hash for the same
    immutable path."""
    entry = {"name": input_name, "path": path, "rollup_sha256": PLACEHOLDER}
    shipped = REPO / "configs" / config_name
    if shipped.is_file():
        payload = yaml.safe_load(shipped.read_text(encoding="utf-8")) or {}
        for candidate in payload.get("inputs") or []:
            if candidate.get("name") != input_name:
                continue
            if str(candidate.get("path")) != path:
                continue
            rollup = str(candidate.get("rollup_sha256") or "")
            if len(rollup) == 64 and set(rollup) != {"0"}:
                entry["rollup_sha256"] = rollup
    if set(entry["rollup_sha256"]) == {"0"} and known:
        entry["rollup_sha256"] = known.get(path, entry["rollup_sha256"])
    return entry


def render() -> str:
    probe = yaml.safe_load(
        (REPO / "configs" / PROBE_CONFIG).read_text(encoding="utf-8")
    )
    entries = phase18.locked_arm_entries()

    # One input per distinct run directory (the p16 pair shares one).
    known = _known_hashes()
    seen: dict[str, str] = {}
    inputs = []
    for entry in entries:
        if entry["run_dir"] in seen:
            continue
        seen[entry["run_dir"]] = entry["name"]
        inputs.append(_carried(
            NAME, entry["name"], f"{CLUSTER_ROOT}/{entry['run_dir']}",
            known,
        ))
    inputs.append(_named(probe, "manifest_v1"))

    arms = [
        {
            "name": entry["name"],
            "input": seen[entry["run_dir"]],
            "seeds": entry["seeds"],
            "csv": entry["csv"],
            "group": entry["group"],
            "n_patients": entry["n_patients"],
        }
        for entry in entries
    ]
    payload = {
        "schema_version": 1, "phase": "p18", "tier": "keeper",
        "seed": 1337, "inputs": inputs,
        "task": {
            "kind": "metric_space_analysis",
            "manifest_artifact": "manifest_v1",
            "arms": arms,
            "contrasts": [dict(c) for c in NAMED_CONTRASTS],
            "n_boot": 10000,
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    pending = sum(1 for e in inputs if "/PENDING_" in e["path"])
    unresolved = sum(
        1 for e in inputs if set(e["rollup_sha256"]) == {"0"}
    )
    if pending:
        provenance = (
            f"# **{pending} PATH(S) PENDING (the p15 probe runs -- their\n"
            "# directories were never declared by anything) and\n"
            f"# {unresolved} ALL-ZERO PLACEHOLDER HASH(ES).** Pass 0\n"
            "# pastes the three probe run directories; ONE pass-1\n"
            "# declare then fills every hash at once. Guard 3 refuses\n"
            "# until all are real.\n"
        )
    elif unresolved:
        provenance = (
            f"# **{unresolved} ALL-ZERO PLACEHOLDER HASH(ES) REMAIN.**\n"
            "# Every path is real; one declare pass fills them all and\n"
            "# guard 3 refuses until it does.\n"
        )
    else:
        provenance = (
            "# **RESOLVED.** Every input declared at a real path with a\n"
            "# verified hash.\n"
        )
    return (
        "# PHASE 18 -- THE METRIC-SPACE ANALYSIS OVER THE LOCKED ARMS.\n"
        "# phase18.EXIT_CRITERIA (locked 2026-08-30);\n"
        "# phase18.ARM_LIST_LOCKED is the single source of this file's\n"
        "# input list AND the task's arm list -- generated from the\n"
        "# constant, so the config cannot drift from the lock and NO\n"
        "# GLOB exists anywhere in the arm resolution.\n"
        "#\n"
        "# GENERATED by scripts/generate_phase18_configs.py.\n"
        "#\n"
        f"# {len(inputs) - 1} run-directory inputs + the manifest; "
        f"{len(arms)} arm rows\n"
        "# (the p16 run carries two prediction sets: loop and identity).\n"
        "#\n"
        "# Pure CPU arithmetic over banked per-seed CSVs -- no patient\n"
        "# images, no training, seconds of compute; keeper tier in the\n"
        "# pinned image so the phase that measures the instruments has\n"
        "# no provenance hole.\n"
        "#\n"
        "# ALL-MSE STANDING CLAUSE: every arm here was trained on MSE;\n"
        "# this run measures metrics on a fixed objective and licenses\n"
        "# no claim about F1- or IEM-trained arms.\n"
        "#\n"
        "# The CleftGNN-IEM prohibition stands on every output.\n"
        "#\n"
        + provenance
        + "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{NAME}\"\n\n"
    ) + body


CONFIGS = {NAME: render}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    drifted = 0
    for name, builder in CONFIGS.items():
        path = REPO / "configs" / name
        text = builder()
        if args.check:
            if not path.is_file():
                print(f"  DRIFT {name}: missing")
                drifted += 1
            elif path.read_text(encoding="utf-8") != text:
                print(f"  DRIFT {name}: differs from the derived config")
                drifted += 1
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {name}")
    if args.check:
        print(f"{len(CONFIGS)} configs checked, {drifted} drifted")
        return 1 if drifted else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
