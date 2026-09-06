"""Phase 17 configs -- TSTR (phase17.EXIT_CRITERIA, locked 2026-08-29).

Six configs: two full-SCUT G1 synthesis sets (TPS for A/C, piecewise
affine for B), three arms, and the five-contrast family analysis. The
mirrored settings are READ FROM the probe's shipped config
(phase17.SETTINGS_PROVENANCE_17); the declared scientific settings --
margin, pair rule, readout normalization, projection dim -- live here as
constants under the never-tuned clause, and any silent edit is DRIFT
under --check.

    python scripts/generate_phase17_configs.py           # write
    python scripts/generate_phase17_configs.py --check    # drift only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CLUSTER_ROOT = "/home/user/codex/cleft-aesthetics"
PLACEHOLDER = "0" * 64

PROBE_CONFIG = "p7_d1_vit_b16_imagenet_g1.yaml"
P5_SYNTH_CONFIG = "p5_asymmetry_synthesis.yaml"
ADDENDUM_CONFIG = "p7_d1_classification_metrics.yaml"

SYNTH_TPS_VERSION = "synth_tps_g1_v1"
SYNTH_PWA_VERSION = "synth_pwa_g1_v1"

#: **PASTED 2026-08-30, the maintainer.** The three finalized arm runs, per
#: the run-dir contract <stem>__<git sha8>__<job id>. Arm A's job id is
#: ``-2``: the first launch (eb5a6887) was the pre-fix faces.jsonl
#: FileNotFoundError crash and is NOT the input. A and C share bf09bd45
#: (launched from one commit); B ran at 4894169c, the commit its own
#: declaration landed in -- the sha8 is the COMMIT's, not the config's.
ARM_RUN_DIRS = {
    "arm_a_run": "p17_arm_a__bf09bd45__p17-arm-a-2",
    "arm_b_run": "p17_arm_b__4894169c__p17-arm-b",
    "arm_c_run": "p17_arm_c__bf09bd45__p17-arm-c",
}

#: The declared scientific settings (phase17.DECLARED_SETTINGS_17):
#: never tuned across runs; movement is a dated amendment with a reason.
MARGIN = 1.0
PAIR_RULE = "symmetric_if_zero_magnitude"
READOUT_NORMALIZATION = "round_1_plus_4_min_d_over_margin"
PROJECTION_DIM = 768
MAGNITUDES = [0.0, 0.015, 0.025, 0.035]


def source_config(name: str) -> dict:
    return yaml.safe_load(
        (REPO / "configs" / name).read_text(encoding="utf-8")
    )


def _named(payload: dict, name: str) -> dict:
    return dict(next(e for e in payload["inputs"] if e["name"] == name))


def _carried(config_name: str, input_name: str, path: str) -> dict:
    """The path-pinned carry (phase 15's ``_carried_named``)."""
    entry = {"name": input_name, "path": path, "rollup_sha256": PLACEHOLDER}
    shipped = REPO / "configs" / config_name
    if not shipped.is_file():
        return entry
    payload = yaml.safe_load(shipped.read_text(encoding="utf-8")) or {}
    for candidate in payload.get("inputs") or []:
        if candidate.get("name") != input_name:
            continue
        if str(candidate.get("path")) != path:
            continue
        rollup = str(candidate.get("rollup_sha256") or "")
        if len(rollup) == 64 and set(rollup) != {"0"}:
            entry["rollup_sha256"] = rollup
    return entry


def _provenance(inputs: list) -> str:
    pending = sum(1 for e in inputs if "/PENDING_" in e["path"])
    unresolved = sum(1 for e in inputs if set(e["rollup_sha256"]) == {"0"})
    if pending:
        return (
            f"# **{pending} PATH(S) PENDING and {unresolved} ALL-ZERO\n"
            "# PLACEHOLDER HASH(ES).** Two passes: paste the run\n"
            "# directory, then declare_inputs.py. Guard 3 refuses until\n"
            "# both are real.\n"
        )
    if unresolved:
        return (
            f"# **{unresolved} ALL-ZERO PLACEHOLDER HASH(ES) REMAIN.**\n"
            "# Every path is real; declare_inputs.py fills the rest and\n"
            "# guard 3 refuses until it does.\n"
        )
    return (
        "# **RESOLVED.** Every input declared at a real path with a\n"
        "# verified hash.\n"
    )


def _launch(name: str) -> str:
    return (
        "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{name}\"\n\n"
    )


def render_synth(version: str, warp_family: str, name: str):
    def render() -> str:
        p5 = source_config(P5_SYNTH_CONFIG)
        inputs = [_named(p5, "scut_root")]
        payload = {
            "schema_version": 1, "phase": "p17", "tier": "keeper",
            "seed": 1337, "inputs": inputs,
            "task": {
                "kind": "asymmetry_synthesis",
                "scut_root": "scut_root",
                "geometries": ["g1"],
                "n_faces": 0,
                "n_sheet": 6,
                "magnitudes": MAGNITUDES,
                "warp_family": warp_family,
                "write_artifact": True,
                "out_version": version,
            },
        }
        body = yaml.safe_dump(
            payload, sort_keys=False, default_flow_style=False
        )
        family = ("the parked module's TPS" if warp_family == "tps"
                  else "arm B's piecewise affine (scut.piecewise)")
        return (
            f"# PHASE 17 -- FULL-SCUT SYNTHESIS SET AT G1, {warp_family.upper()}.\n"
            "# phase17.EXIT_CRITERIA criterion 1.\n"
            "#\n"
            "# GENERATED by scripts/generate_phase17_configs.py.\n"
            "#\n"
            f"# Warp family: {family}; SAME control points, directions\n"
            "# and magnitudes either way, so B-vs-C isolates the\n"
            "# transformation family. G1 only -- the probe's geometry,\n"
            "# like-for-like with the 0.2520 bar; G2 is UNREGISTERED for\n"
            "# this phase. n_faces 0 = every SCUT face; sides balanced\n"
            "# deterministically from the seed.\n"
            "#\n"
            "# Set size ruled UNCONSTRAINED by the compute-gate\n"
            "# measurement (COMPUTE_GATE_MEASUREMENT['outcome_2026_08_29']:\n"
            "# 0.2-1.2 s/face, no pathology). Per-face timing lines are\n"
            "# in the task and stay.\n"
            "#\n"
            "# CPU work; GPUs idle -- expected, not a defect.\n"
            "#\n"
            + _provenance(inputs) + _launch(name)
        ) + body

    return render


def _recipe():
    probe = source_config(PROBE_CONFIG)["task"]
    return {
        "seeds": list(probe["seeds"]),
        "max_epochs": probe["max_epochs"],
        "learning_rate": probe["learning_rate"],
        "batch_size": probe["batch_size"],
        "inner_val_frac": probe["inner_val_frac"],
    }


def render_arm_a() -> str:
    probe = source_config(PROBE_CONFIG)
    inputs = [
        _carried("p17_arm_a.yaml", "synth_set",
                 f"{CLUSTER_ROOT}/data/scut/{SYNTH_TPS_VERSION}"),
        _named(probe, "manifest_v1"),
        _named(probe, "embeddings"),
    ]
    payload = {
        "schema_version": 1, "phase": "p17", "tier": "keeper",
        "seed": 1337, "inputs": inputs,
        "task": {
            "kind": "tstr_regression",
            "arm": "p17_arm_a",
            "synth_set": "synth_set",
            "manifest_artifact": "manifest_v1",
            "embeddings_artifact": "embeddings",
            "backbone": "vit_b16",
            **_recipe(),
            "monitor": "inner_val_mse",
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return (
        "# PHASE 17 ARM A -- the parked TPS design, as parked\n"
        "# (post-amendment wording: a TSTR design in the Rosero FAMILY,\n"
        "# NOT a replication -- synthesis.PARKED['arm_amended_2026_08_29']).\n"
        "#\n"
        "# GENERATED by scripts/generate_phase17_configs.py; recipe\n"
        f"# values COPIED FROM {PROBE_CONFIG}.\n"
        "#\n"
        "# Frozen ViT-B/16 embeddings, linear head, MAGNITUDE-MAPPED\n"
        "# labels: grade = 1 + 4*(m/0.035) -- the declared map, the\n"
        "# assumption under test (DECLARED_SETTINGS_17). Zero real\n"
        "# patient images or labels in training; all 237 pure test.\n"
        "#\n"
        "# CARRIED READING: not predicted to succeed (the philtrum\n"
        "# reason). NO patience -- fixed budget, best inner-val\n"
        "# checkpoint.\n"
        "#\n"
        + _provenance(inputs) + _launch("p17_arm_a.yaml")
    ) + body


def render_siamese(arm: str, version: str, name: str):
    def render() -> str:
        probe = source_config(PROBE_CONFIG)
        inputs = [
            _carried(name, "synth_set",
                     f"{CLUSTER_ROOT}/data/scut/{version}"),
            _named(probe, "manifest_v1"),
            _named(probe, "staged_v1"),
        ]
        payload = {
            "schema_version": 1, "phase": "p17", "tier": "keeper",
            "seed": 1337, "inputs": inputs,
            "task": {
                "kind": "siamese_contrastive",
                "arm": arm,
                "synth_set": "synth_set",
                "manifest_artifact": "manifest_v1",
                "staged_artifact": "staged_v1",
                "geometry": "g1",
                "backbone": "vit_b16",
                "branch_trainability": "frozen",
                "margin": MARGIN,
                "pair_rule": PAIR_RULE,
                "readout_normalization": READOUT_NORMALIZATION,
                "projection_dim": PROJECTION_DIM,
                **_recipe(),
                "monitor": "inner_val_loss",
            },
        }
        body = yaml.safe_dump(
            payload, sort_keys=False, default_flow_style=False
        )
        letter = arm[-1].upper()
        family = ("piecewise-affine (arm B's declared set)" if letter == "B"
                  else "TPS (shared with arm A -- the hybrid)")
        return (
            f"# PHASE 17 ARM {letter} -- frozen branches, trained linear\n"
            "# projection, contrastive loss, fixed-midline split,\n"
            "# distance readout normalized-and-rounded.\n"
            "#\n"
            "# GENERATED by scripts/generate_phase17_configs.py; recipe\n"
            f"# values COPIED FROM {PROBE_CONFIG}; margin, pair rule,\n"
            "# readout and projection dim are DECLARED scientific\n"
            "# settings (phase17.DECLARED_SETTINGS_17) -- never tuned;\n"
            "# movement is a dated amendment.\n"
            "#\n"
            f"# Synthesis set: {family}. The training and readout are\n"
            "# byte-identical between B and C -- only the declared set\n"
            "# differs, which is what makes B-C isolate the\n"
            "# transformation family.\n"
            "#\n"
            "# branch_trainability: frozen -- the ONLY admissible value;\n"
            "# 'trainable' is refused by the schema (the unexercised\n"
            "# regime: three novelties at once, unattributable).\n"
            "#\n"
            "# CLAIM WORDING BOUND: never 'we replicated Rosero'. Two\n"
            "# declared adaptations: shipped landmarks for detection,\n"
            "# fixed midline for a landmark-localised split.\n"
            "#\n"
            "# Zero real patient images or labels in training; all 237\n"
            "# pure test. NO patience.\n"
            "#\n"
            + _provenance(inputs) + _launch(name)
        ) + body

    return render


def render_family() -> str:
    probe = source_config(PROBE_CONFIG)
    addendum = source_config(ADDENDUM_CONFIG)
    name = "p17_family_analysis.yaml"
    inputs = [
        _carried(name, input_name,
                 f"{CLUSTER_ROOT}/runs/keeper/p17/{directory}")
        for input_name, directory in ARM_RUN_DIRS.items()
    ] + [
        _named(addendum, "arm_run") | {"name": "probe_run"},
        _named(probe, "manifest_v1"),
    ]
    payload = {
        "schema_version": 1, "phase": "p17", "tier": "keeper",
        "seed": 1337, "inputs": inputs,
        "task": {
            "kind": "tstr_family_analysis",
            "arm_a_run": "arm_a_run",
            "arm_b_run": "arm_b_run",
            "arm_c_run": "arm_c_run",
            "probe_run": "probe_run",
            "manifest_artifact": "manifest_v1",
            "seeds": _recipe()["seeds"],
            "n_boot": 10000,
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return (
        "# PHASE 17 -- THE FIVE-CONTRAST FAMILY, AND NOTHING BEYOND IT.\n"
        "# phase17.EXIT_CRITERIA criterion 3; READINGS_COMMITTED holds\n"
        "# the five combination cells, committed before any number.\n"
        "#\n"
        "# GENERATED by scripts/generate_phase17_configs.py.\n"
        "#\n"
        "# Three primaries (each arm vs the probe), two secondaries\n"
        "# (A-C: training scheme on shared synthesis; B-C:\n"
        "# transformation family on shared training). Paired BCa\n"
        "# 10,000 by shared seed, both criterion conditions, every\n"
        "# outcome recorded claimable/withdrawn/unresolved.\n"
        "#\n"
        "# Pod arithmetic over prediction CSVs; no training here.\n"
        "#\n"
        + _provenance(inputs) + _launch(name)
    ) + body


CONFIGS = {
    "p17_synth_tps_g1.yaml": render_synth(
        SYNTH_TPS_VERSION, "tps", "p17_synth_tps_g1.yaml"
    ),
    "p17_synth_pwa_g1.yaml": render_synth(
        SYNTH_PWA_VERSION, "piecewise_affine", "p17_synth_pwa_g1.yaml"
    ),
    "p17_arm_a.yaml": render_arm_a,
    "p17_arm_b.yaml": render_siamese(
        "p17_arm_b", SYNTH_PWA_VERSION, "p17_arm_b.yaml"
    ),
    "p17_arm_c.yaml": render_siamese(
        "p17_arm_c", SYNTH_TPS_VERSION, "p17_arm_c.yaml"
    ),
    "p17_family_analysis.yaml": render_family,
}


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
