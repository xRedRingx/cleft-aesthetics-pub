"""Write Phase 11's pre-step config from the shipped configs' own vectors.

    PYTHONPATH=src python scripts/generate_phase11_configs.py [--check]

ONE config (``phase11.PRE_STEP_CONFIG_SHAPE``): the ranking the amendment
asks about is a single cross-arm object, and six per-scope configs would
produce six partial rankings to be merged by hand outside any declared
artifact.

**No hash is new.** Every OOF vector this declares is already declared,
at the same path, with a verified rollup, in a config this repository has
shipped -- a path's contents are immutable, so one hash serves every
config declaring it. The generator carries them all by path, which is why
a run over 490 vectors needs no cluster round trip at all.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from cleft import ladder, phase11  # noqa: E402

CLUSTER_ROOT = "/home/user/codex/cleft-aesthetics"
PLACEHOLDER = "0" * 64

#: Arms this phase created, which are NOT part of the pre-step's coverage.
#: The pre-step measured the ladder as it stood and is closed; see
#: ``declared_vectors``.
EXCLUDED_FROM_THE_PRESTEP = (
    phase11.PAIRED_ARM_STEM, phase11.PAIRED_CONTROL_STEM,
    # [2026-08-23] Phase 12's four view-ablation arms, on the same
    # ground as Phase 11's own: the pre-step measured the ladder as it
    # stood on 2026-08-17 and is CLOSED; arms created after a
    # measurement are not retroactively part of what it measured. These
    # also run a DIFFERENT COHORT (236, not 237), so their vectors could
    # not even pair against the pre-step's truth column.
    "p12_arm_a_frontal", "p12_arm_b_basal",
    "p12_arm_c_concat", "p12_arm_d_capacity",
)
CONFIG_NAME = "p11_iem_prestep.yaml"
P1_CONFIG = "p1_build_manifest.yaml"
SOURCE_CONFIG = "p7_d1_vit_b16_imagenet_g1.yaml"


def source(name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / name).read_text(encoding="utf-8"))


def declared_vectors() -> list[dict]:
    """Every verified OOF vector any shipped config declares, deduplicated
    by name and sorted.

    A vector qualifies only if its path is a real run directory and its
    rollup is a verified hash: a PENDING path or an all-zero placeholder
    is by definition not yet a declaration, and carrying one here would
    put an unverifiable input in front of guard 3.
    """
    by_name: dict[str, dict] = {}
    unnamed: dict[str, str] = {}
    for path in sorted((REPO / "configs").glob("*.yaml")):
        if path.name == CONFIG_NAME:
            continue
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in payload.get("inputs") or []:
            name = str(entry.get("name") or "")
            location = str(entry.get("path") or "")
            rollup = str(entry.get("rollup_sha256") or "")
            if not name.startswith("oof_") or not location.endswith(
                "__predictions.csv"
            ):
                continue
            if "*" in location or "/PENDING_" in location:
                continue
            if len(rollup) != 64 or set(rollup) == {"0"}:
                continue
            # **[2026-08-17] The pre-step's coverage is CLOSED.** Its 68
            # arms and 490 vectors are part of a recorded result
            # (phase11.PRESTEP_CLOSED, PRE_STEP_OBSERVED: tau 0.5917 over
            # exactly those arms). Phase 11's own loss arms were declared
            # after it, and letting them in would silently regrow the
            # input set of a measurement that has already been read and
            # closed. Arms created after a measurement are not
            # retroactively part of what it measured.
            if any(
                name.startswith(f"oof_{stem}_seed_")
                for stem in EXCLUDED_FROM_THE_PRESTEP
            ):
                continue
            # **Not every `oof_` input names an arm.** p8c_sheet declares
            # five as bare `oof_seed_<n>`, because that renderer reads one
            # arm and needs no arm axis. They cannot be parsed into
            # (arm, seed), and phase7c.oof_paths_from_inputs would refuse
            # them at run time. They are set aside here rather than
            # dropped -- see the coverage check below, which refuses any
            # whose FILE is not already carried under a proper name.
            stem, separator, seed = name[len("oof_"):].rpartition("_seed_")
            if not separator or not stem or not seed.isdigit():
                unnamed[location] = name
                continue
            existing = by_name.get(name)
            if existing is not None and existing["path"] != location:
                raise SystemExit(
                    f"{name} is declared at two different paths:\n"
                    f"  {existing['path']}\n  {location}\n"
                    "One name must mean one file, or the pre-step would "
                    "score two runs under one arm."
                )
            by_name[name] = {
                "name": name, "path": location, "rollup_sha256": rollup,
            }

    # An unnamed vector whose file is ALSO declared under a proper
    # arm_seed name costs nothing to skip -- the same file is already in.
    # One that is not would be a whole arm dropped in silence, so it stops
    # the generator instead.
    carried = {entry["path"] for entry in by_name.values()}
    orphans = sorted(set(unnamed) - carried)
    if orphans:
        raise SystemExit(
            f"{len(orphans)} OOF vectors are declared only under names "
            "that carry no arm, e.g.\n  "
            + "\n  ".join(f"{unnamed[p]} -> {p}" for p in orphans[:3])
            + "\nSkipping them would drop an arm from the pre-step in "
            "silence. Give them oof_<arm>_seed_<n> names, or exclude them "
            "here deliberately with a recorded reason."
        )
    return [by_name[name] for name in sorted(by_name)]


def render() -> str:
    vectors = declared_vectors()
    p1 = {e["name"]: e for e in source(P1_CONFIG)["inputs"]}
    arm = source(SOURCE_CONFIG)
    inputs_by_name = {e["name"]: e for e in arm["inputs"]}
    shared = [int(s) for s in ladder.SEED_POOL[:5]]

    arms: dict[str, set] = {}
    for entry in vectors:
        stem, _, seed = entry["name"][len("oof_"):].rpartition("_seed_")
        arms.setdefault(stem, set()).add(int(seed))
    missing = sorted(
        stem for stem, seeds in arms.items() if not set(shared) <= seeds
    )
    if missing:
        raise SystemExit(
            f"{len(missing)} arms lack the shared band {shared}, e.g. "
            f"{missing[:3]}. The shared-five column cannot be reported "
            "for them, so the config would promise a column it cannot fill."
        )
    bands: dict[int, int] = {}
    for seeds in arms.values():
        bands[len(seeds)] = bands.get(len(seeds), 0) + 1

    payload = {
        "schema_version": 1,
        "phase": "p11",
        "tier": "keeper",
        "seed": shared[0],
        "inputs": (
            [dict(inputs_by_name["manifest_v1"]), dict(p1["scoresheet_primary"])]
            + vectors
        ),
        "task": {
            "kind": "iem_prestep",
            "manifest_artifact": "manifest_v1",
            "scoresheet_artifact": "scoresheet_primary",
            "shared_seeds": shared,
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    band_text = ", ".join(
        f"{count} arms x {size} seeds" for size, count in sorted(bands.items())
    )
    criteria = phase11.PHASE_11_EXIT_CRITERIA["criteria"]
    return (
        "# PHASE 11 — THE PRE-STEP: THE ASYMMETRIC ERROR ON PREDICTIONS\n"
        "# ALREADY ON DISK. phase11.PRE_STEP_REGISTERED.\n"
        "#\n"
        "# GENERATED by scripts/generate_phase11_configs.py from the OOF\n"
        "# vectors the shipped configs already declare.\n"
        "#\n"
        "# **THIS RUN FITS NOTHING.** Every vector was written by a keeper\n"
        "# run that already happened. No GPU, no training, no re-scoring\n"
        "# of any model -- arithmetic over files on disk.\n"
        "#\n"
        f"# **COVERAGE: {len(arms)} arms, {len(vectors)} vectors** "
        f"({band_text}).\n"
        "# No subset: the amendment asks whether EVERY existing arm's\n"
        "# ranking moves. Each arm is pooled over its OWN band with the\n"
        "# shared-five figures beside it, so neither choice is silent.\n"
        "#\n"
        "# **ONE CONFIG, NOT SIX** (phase11.PRE_STEP_CONFIG_SHAPE): the\n"
        "# ranking is a single cross-arm object, and per-scope configs\n"
        "# would produce partial rankings to be merged by hand outside any\n"
        "# declared artifact -- the computed-then-reassembled gap this\n"
        "# project has hit three times. The cost is file size, not risk.\n"
        "#\n"
        "# **G IS THE SHEET'S MEDIAN GRADE**, which is why the manifest and\n"
        "# the score sheet are declared: IEM is defined against a consensus\n"
        "# GRADE, the median is what the sheet carries and what Phase 10\n"
        "# trained on, and scoring a grade-defined metric against the\n"
        "# continuous panel mean would manufacture small |d| precisely\n"
        "# where the metric inverts. The CSVs' panel-mean truth column is\n"
        "# used ONLY as the cross-arm consistency check, never as G.\n"
        "#\n"
        "# **BOTH DIRECTION CONVENTIONS RUN, NEITHER RANKING IS QUOTABLE\n"
        "# ALONE.** The direction question is unanswered\n"
        "# (phase10.PHASE_11_BLOCKED_ON_IEM_DIRECTION) and the loss build\n"
        "# stays blocked on it; this measures whether the ambiguity\n"
        "# changes the ranking rather than waiting to find out.\n"
        "#\n"
        "# **THE METRIC HAS A MEASURED DEFECT THAT TRAVELS WITH EVERY\n"
        "# NUMBER**: eq (16)'s branches cross at |d| = 0.19753 and below\n"
        "# it the branch called more dangerous is penalised LESS\n"
        "# (phase11.IEM_CROSSOVER_INVERTS). The below-crossover share is\n"
        "# reported per arm. And the paper's [0,4] is a DOMAIN, not a\n"
        "# range, so predictions are asserted within [1,5] and breaches\n"
        "# are reported rather than clipped.\n"
        "#\n"
        "# **DESCRIPTIVE ONLY -- no claim, no ledger row for a ranking.**\n"
        "#\n"
        "# EXIT CRITERIA (phase11.PHASE_11_EXIT_CRITERIA, registered\n"
        "# before this config existed):\n"
        + "".join(f"#   {line}\n" for line in criteria)
        + "#\n"
        "# **RESOLVED.** Every hash is carried from a shipped config at the\n"
        "# same path -- a path's contents are immutable, so one verified\n"
        "# hash serves every config declaring it. NOTHING NEW IS DECLARED.\n"
        "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{CONFIG_NAME}\"\n\n"
    ) + body


ARM_NAME = "p11_iem_arm.yaml"
CONTROL_NAME = "p11_mse_control.yaml"


def render_arm(loss: str) -> str:
    """The loss arm, or its matched control -- ONE renderer, so the two
    configs cannot drift apart in anything but ``loss``.

    That is not tidiness: ``PHASE_11_BUILD_REGISTERED`` requires the
    control to differ in the loss and in NOTHING else, and two renderers
    would make that a claim to check rather than a fact by construction.
    """
    arm = source(SOURCE_CONFIG)
    inputs_by_name = {e["name"]: e for e in arm["inputs"]}
    p1 = {e["name"]: e for e in source(P1_CONFIG)["inputs"]}
    task = arm["task"]
    payload = {
        "schema_version": 1,
        "phase": "p11",
        "tier": "keeper",
        "seed": task["seeds"][0],
        "inputs": [
            dict(inputs_by_name["manifest_v1"]),
            dict(inputs_by_name["staged_v1"]),
            dict(inputs_by_name["embeddings"]),
            dict(p1["scoresheet_primary"]),
        ],
        "task": {
            "kind": "iem_arm",
            "loss": loss,
            "manifest_artifact": "manifest_v1",
            "staged_artifact": "staged_v1",
            "embeddings_artifact": "embeddings",
            "scoresheet_artifact": "scoresheet_primary",
            "geometry": task["geometry"],
            "label": task["label"],
            "seeds": task["seeds"],
            "inner_val_frac": task["inner_val_frac"],
            "max_epochs": task["max_epochs"],
            "patience": task["patience"],
            "monitor": task["monitor"],
            "learning_rate": task["learning_rate"],
            "weight_decay": task["weight_decay"],
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    name = ARM_NAME if loss == "iem" else CONTROL_NAME
    role = (
        "THE LOSS ARM: eq (16) VERBATIM as the training objective"
        if loss == "iem" else
        "THE MATCHED CONTROL: mean squared error, and nothing else changed"
    )
    return (
        f"# PHASE 11 — {role}.\n"
        "# phase11.PHASE_11_BUILD_REGISTERED.\n"
        "#\n"
        "# GENERATED by scripts/generate_phase11_configs.py. The arm and\n"
        "# its control come from ONE renderer, so they cannot differ in\n"
        "# anything but `loss` -- which the registration requires, and\n"
        "# which two renderers would make a claim to check rather than a\n"
        "# fact by construction.\n"
        "#\n"
        f"# **THE CELL, copied from {SOURCE_CONFIG}**: the 0.2520 arm --\n"
        "# frozen ViT-B/16, ImageNet, G1, whole image; the project's best\n"
        "# and the anchor of every comparison. ONE twin, not a ladder of\n"
        "# them: a ladder answers nothing extra until the first shows the\n"
        "# loss does anything.\n"
        "#\n"
        "# **TARGET: the sheet's MEDIAN GRADE.** Equation (16) is defined\n"
        "# against a consensus grade G, and training against a different\n"
        "# quantity than the loss's own definition would confound the arm.\n"
        "# **EVALUATION: PCC against the PANEL MEAN**, as everywhere else\n"
        "# -- so the arm optimises one thing and is judged on another,\n"
        "# deliberately and on the record.\n"
        "#\n"
        "# **BOTH READINGS TRAVEL TOGETHER.** IEM alone is marking its own\n"
        "# homework; PCC alone hides whether the loss did what it was\n"
        "# asked. Neither is reported without the other.\n"
        "#\n"
        + (
            "# **THE LOSS IS VERBATIM, NOT SMOOTHED**\n"
            "# (phase11.IEM_LOSS_VERBATIM). It carries two measured\n"
            "# defects, and they carry with every number it produces: the\n"
            "# VALUE inversion below |d| = 0.19753, where the branch\n"
            "# called more dangerous is penalised less, and the GRADIENT\n"
            "# inversion below |d| = 0.0719, where the loss pushes harder\n"
            "# to remove an over-prediction than an under-prediction. The\n"
            "# one departure from verbatim is numerical: |d|**0.87 has an\n"
            "# infinite derivative at zero, so the magnitude is clamped to\n"
            "# 1e-8 inside the power only -- the value changes by under\n"
            "# 1e-7 and the gradient is capped at 7.63.\n"
            "#\n"
            "# **THE DEGENERACY READING IS REGISTERED IN ADVANCE**\n"
            "# (phase11.DEGENERACY_APPLICATION_RULE). An arm that improves\n"
            "# on IEM while its span narrows and its PCC falls is the\n"
            "# degenerate optimum realised, NOT a result, and is reported\n"
            "# as the metric's behaviour rather than the model's. The\n"
            "# per-fold signature and its two thresholds -- 13 of 25, and\n"
            "# a gap of 5 folds over this control -- were fixed before\n"
            "# either arm ran.\n"
            if loss == "iem" else
            "# **THIS CONTROL IS PART OF THE ARM, NOT AN OPTIONAL EXTRA.**\n"
            "# Without it, 'the loss pulled this arm toward hedging' and\n"
            "# 'this cell hedges under any loss' are indistinguishable,\n"
            "# and the degeneracy reading cannot be applied at all. It is\n"
            "# condition 2 of phase11.DEGENERACY_APPLICATION_RULE, and it\n"
            "# is the condition that does the work.\n"
        )
        + "#\n"
        "# **RESOLVED.** Every hash is carried from a shipped config at\n"
        "# the same path. NOTHING NEW IS DECLARED.\n"
        "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{name}\"\n\n"
    ) + body


PAIRED_NAME = "p11_paired.yaml"


def render_paired() -> str:
    """PLAN 4.3 condition 1 for the loss contrast. Fits nothing.

    Same task, same BCa, same loader as every other scope -- only the
    pair enumeration differs, and it lives in ``phase11``. The one thing
    that is new anywhere: this scope reports TWO metrics, because IEM
    alone would be marking its own homework
    (``phase11.PAIRED_CLAIM_COVERAGE``).
    """
    from cleft import ladder

    arm = source(SOURCE_CONFIG)
    inputs_by_name = {e["name"]: e for e in arm["inputs"]}
    p1 = {e["name"]: e for e in source(P1_CONFIG)["inputs"]}
    pairs = phase11.paired_claim_pairs("p11_loss")
    vectors = ladder.paired_claim_vectors("p11_loss", pairs=pairs)
    carried = {}
    path = REPO / "configs" / PAIRED_NAME
    if path.is_file():
        carried = {
            e["name"]: e
            for e in (yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            .get("inputs") or []
            if "/PENDING_" not in e["path"]
        }

    inputs = [
        dict(inputs_by_name["manifest_v1"]),
        dict(p1["scoresheet_primary"]),
    ]
    for stem, seed in vectors:
        name = f"oof_{stem}_seed_{seed}"
        inputs.append(carried.get(name) or {
            "name": name,
            "path": (
                f"{CLUSTER_ROOT}/runs/keeper/p11/PENDING_{stem}/"
                f"seed_{seed}__predictions.csv"
            ),
            "rollup_sha256": PLACEHOLDER,
        })
    unresolved = sum(1 for e in inputs if e["rollup_sha256"] == PLACEHOLDER)
    pending = sum(1 for e in inputs if "/PENDING_" in e["path"])
    recorded = pairs[0]["recorded"]

    if pending:
        provenance = (
            f"# **{unresolved} HASHES ARE ALL-ZERO PLACEHOLDERS and "
            f"{pending} paths are\n"
            "# PENDING_<stem>.** Both arms ran at SHA 27a18e04, but a run\n"
            "# directory also carries a job id, which is not derivable on a\n"
            "# laptop. Two passes: paste the run directories, then\n"
            "# declare_inputs.py. Guard 3 refuses until both are real.\n"
        )
    elif unresolved:
        provenance = (
            "# **PATHS RESOLVED, HASHES PENDING.** Every path is a real run\n"
            f"# directory (each one passed run_names.check_run_dir), and the\n"
            f"# {unresolved} remaining hashes are all-zero PLACEHOLDERS until\n"
            "# declare_inputs.py verifies the files -- the second pass of the\n"
            "# standing two-pass flow. Guard 3 refuses the run until they are\n"
            "# real, which is the placeholder working rather than a mistake.\n"
        )
    else:
        provenance = (
            f"# **RESOLVED.** All {len(inputs)} inputs declared at real paths\n"
            "# with verified hashes.\n"
        )

    payload = {
        "schema_version": 1,
        "phase": "p11",
        "tier": "keeper",
        "seed": pairs[0]["seeds"][0],
        "inputs": inputs,
        "task": {
            "kind": "paired_claims",
            "scope": "p11_loss",
            "n_boot": 10000,
            "manifest_artifact": "manifest_v1",
            "scoresheet_artifact": "scoresheet_primary",
        },
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return (
        "# PHASE 11 — PLAN §4.3 CONDITION 1 FOR THE LOSS CONTRAST:\n"
        "# THE IEM ARM AGAINST ITS MATCHED MSE CONTROL.\n"
        "# phase11.PAIRED_CLAIM_COVERAGE, phase11.LOSS_ARMS_OBSERVED.\n"
        "#\n"
        "# GENERATED by scripts/generate_phase11_configs.py from\n"
        "# phase11.paired_claim_pairs('p11_loss').\n"
        "#\n"
        "# **THIS RUN FITS NOTHING.** Both vectors were written by keeper\n"
        "# runs that already happened.\n"
        "#\n"
        "# **LIKE-FOR-LIKE, and that is why it is admissible**: same cell,\n"
        "# same seeds, same folds, same target, differing in the LOSS and\n"
        "# nothing else, on shared patients. The IEM arm's PCC against the\n"
        "# LADDER stays DESCRIPTIVE -- a different loss and a different\n"
        "# training target make it not a like-for-like entry, and pairing\n"
        "# it would dress a non-comparison in an interval.\n"
        "#\n"
        "# **TWO METRICS, AND THIS IS THE ONLY SCOPE THAT REPORTS TWO.**\n"
        "# PCC against the panel mean (the CSVs' own truth column) and IEM\n"
        "# against the sheet's MEDIAN GRADE -- which is why the manifest\n"
        "# and the score sheet are declared here and nowhere else in a\n"
        "# paired config. IEM alone would be marking its own homework;\n"
        "# PCC alone would hide whether the loss did what it was asked.\n"
        "#\n"
        "# The frozen paired_delta_bca was ALREADY metric-generic; only\n"
        "# phase7b.paired_comparison hardcoded pcc, and it now takes the\n"
        "# statistic. No second paired implementation was written, and\n"
        "# every other scope's output is byte-identical.\n"
        "#\n"
        "# **THE PREDICTION, DERIVED FROM LOSS_ARMS_OBSERVED**:\n"
        f"#   PCC  delta {recorded['pcc']['delta_of_means']:+.4f}  "
        f"threshold {recorded['pcc']['threshold']:.4f}  "
        f"margin {recorded['pcc']['margin']}x\n"
        f"#   IEM  delta {recorded['iem']['delta_of_means']:+.4f}  "
        f"threshold {recorded['iem']['threshold']:.4f}  "
        f"margin {recorded['iem']['margin']}x\n"
        "# Delta is (IEM arm minus MSE control), so NEGATIVE means the IEM\n"
        "# arm scores lower -- WORSE on PCC, BETTER on IEM. The direction\n"
        "# is recorded per metric because the two run opposite ways.\n"
        "# Both margins sit just above the 2.55x line below which nothing\n"
        "# in this project has ever passed, so condition 2 is marginal on\n"
        "# both and condition 1 decides.\n"
        "#\n"
        "# **BOTH DEFECTS TRAVEL WITH EVERY NUMBER**: the value inversion\n"
        "# below |d| = 0.19753 and the gradient inversion below\n"
        "# |d| = 0.0719. The loss ran verbatim with them in.\n"
        "#\n"
        + provenance
        + "#\n"
        f"#   bash -lc \"cd {CLUSTER_ROOT} && PYTHONPATH=src \\\n"
        f"#     python -m cleft.run --config configs/{PAIRED_NAME}\"\n\n"
    ) + body


CONFIGS = {
    CONFIG_NAME: render,
    ARM_NAME: lambda: render_arm("iem"),
    CONTROL_NAME: lambda: render_arm("mse"),
    PAIRED_NAME: render_paired,
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
    print("every vector carried from a shipped config; nothing new declared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
