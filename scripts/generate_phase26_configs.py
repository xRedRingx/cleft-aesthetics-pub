"""Write Phase 26's sixteen cell configs and its table config.

    PYTHONPATH=src python scripts/generate_phase26_configs.py [--check]

**The Phase 22 precedent**: a two factor design is emitted as one config
per cell by a nested loop over the levels, each cell a one factor
neighbour of its row and column, with ``--check`` catching drift.

**Every recipe field is copied from the probe over the UNION of its task
block, not over a hand maintained list.** Phase 25's arms diverged on
batch size because a field nobody added to the copied list is invisible
to a check that iterates the list. Here the copy is everything the probe
declares, minus the fields this grid varies and the fields that name an
artifact, so an omission is impossible rather than unlikely.

Nothing is launched. The configs are written and the runs are the
maintainer's.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CONFIGS = REPO / "configs"
PROBE = "p7_d1_vit_b16_imagenet_g1"
CLUSTER_ROOT = "/home/user/codex/cleft-aesthetics"
PLACEHOLDER = "0" * 64

#: The two factors, from the locked record rather than retyped here. A
#: level list in two places is a level list that can disagree.
from cleft import phase26  # noqa: E402

WEIGHT_DECAYS = phase26.THE_GRID_LOCKED["weight_decay_levels"]
PATIENCES = phase26.THE_GRID_LOCKED["patience_levels"]

#: Held at 30 on every cell so patience is never bounded by the budget
#: at the far end (``phase26.THE_SECOND_FACTOR_RULED``).
MAX_EPOCHS = 30

#: **[SUPPLIED 2026-09-06] The sixteen run directories, as given.**
#:
#: Every cell ran at sha ``1df28d6f`` with a job id matching its own
#: name. **These are supplied rather than constructed**: the run-path
#: rule exists because job ids abbreviate unpredictably, so a
#: directory derived from a stem is a directory nobody declared. The
#: pattern is written out here because it was stated, not because it
#: was inferred.
RUN_SHA = "1df28d6f"
RUN_DIRS = {
    f"p26_cell_w{w}_p{p}":
        f"p26_cell_w{w}_p{p}__{RUN_SHA}__p26-cell-w{w}-p{p}"
    for w in range(4) for p in range(4)
}

#: **[DECLARED 2026-09-06 from scripts/declare_inputs.py]** The sixteen
#: run-directory rollups, transcribed once.
#:
#: **Every one must be distinct, and the generator refuses if any two
#: agree.** The cells differ only in two scalar settings, so their
#: directories are nearly identical in size and shape. A repeated hash
#: therefore reads as two cells that happen to agree, when what it
#: actually means is that one was pasted twice. The check is cheap and
#: the failure it catches is silent.
RUN_ROLLUPS = {
    "p26_cell_w0_p0":
        "82dd875bd94f03363ab210f6f880b0c796ad6fc35c501cac0a0982db84963b18",
    "p26_cell_w0_p1":
        "3c95221e35ee0fcc3131401f0c3a11668cd9be9632b6c71effb8e26881d3a3b7",
    "p26_cell_w0_p2":
        "67232a7d5ce5b8cf1d535f7794bca4a7dfdb8df801514d9580550fdb6f08e07d",
    "p26_cell_w0_p3":
        "84b13c350dfa16f4d52b6a2fb65573abef0a843b5143abc93067b4505c5ced3e",
    "p26_cell_w1_p0":
        "5fb2df0d34ce3f7673b4f1f32681e6b4c83257ce0acddd87b52189e454e793c1",
    "p26_cell_w1_p1":
        "285f7d7b9ada2a501022b90371c0a9de0331700d77d822af016161bfde50bc15",
    "p26_cell_w1_p2":
        "5d83261800ee74f036ab681aaca5cca331edb0c16ad81d6dd30a93d7ec93e624",
    "p26_cell_w1_p3":
        "8a9e138ad8981c68b2f2fcfe991a9ea412ed80be6126b37b330ade0b59a0f5d5",
    "p26_cell_w2_p0":
        "3597d8fa8fabf0fb679cbeade0e5e09917ee2b3c16e45ded950f6b357fa8746a",
    "p26_cell_w2_p1":
        "84497989d0fab56a53ab535be6eada62b35df695aa7e820ba2524bb44ed79ad7",
    "p26_cell_w2_p2":
        "4f08be85cdb2a6f4d8aa288e60279a1f9f028d10591e9f73319896709e5d606a",
    "p26_cell_w2_p3":
        "e34e0e87f05f4890f6f1ea346752effc620189ee760beeede74ab5edecfe21fb",
    "p26_cell_w3_p0":
        "f1d9f6dfdbd79269c0e31a949223f4c3d281692a9013add101dec66182c0f212",
    "p26_cell_w3_p1":
        "779c425ad4624238a11d1e6f26e3b5509015d98599d42e4531af431a0762ad86",
    "p26_cell_w3_p2":
        "bc4ef5afde6012f9ef3cbc18618e0ae8b7a04b8b78e179a237e7b277b884c9c3",
    "p26_cell_w3_p3":
        "098f542f03d08ad9e2f4d1a93ddfe7494ba6677ad8e983437933c99ba7f765ba",
}


#: **[READ 2026-09-06 from the Phase 18 run's metrics.json]** The
#: constant-predictor IEM under ``floors.iem_constant_predictor``.
#: **Read, never recomputed here**: the point of the cross-check is
#: that two independent computations agree, and a value this script
#: derived would be the same computation twice
#: (``phase26.THE_FLOOR_RULED``).
PHASE18_FLOOR_IEM = 0.5371587111538849

#: What the grid varies, and what names an artifact rather than a
#: recipe. Everything else in the probe's task block is copied.
VARIED = {"weight_decay", "patience", "max_epochs"}
STRUCTURAL = {
    "kind", "manifest_artifact", "staged_artifact", "embeddings_artifact",
}


class DriftError(RuntimeError):
    """The generator cannot build a config it can stand behind."""


def _load(stem: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{stem}.yaml").read_text(encoding="utf-8"))


def _stem(index_w: int, index_p: int) -> str:
    return f"p26_cell_w{index_w}_p{index_p}"


def _cell_configs() -> dict:
    probe = _load(PROBE)
    task = probe["task"]
    carried = {
        key: value for key, value in task.items()
        if key not in VARIED and key not in STRUCTURAL
    }
    if "batch_size" not in carried or "weight_decay" in carried:
        raise DriftError(
            "the carried recipe is wrong: batch_size must be carried and "
            "weight_decay must be varied"
        )

    texts = {}
    for index_w, decay in enumerate(WEIGHT_DECAYS):
        for index_p, patience in enumerate(PATIENCES):
            stem = _stem(index_w, index_p)
            anchor = decay == 0.01 and patience == 5
            head = (
                f"# PHASE 26 -- the calibration ablation, cell "
                f"w{index_w}p{index_p}.\n"
                f"#\n"
                f"# weight_decay {decay}, patience {patience}. Every other\n"
                f"# field is the probe's own, carried from\n"
                f"# configs/{PROBE}.yaml over the UNION of its task block\n"
                f"# rather than over a list, so a field cannot be omitted\n"
                f"# silently (phase25.PHASE_25_CLOSING criterion 2).\n"
                f"#\n"
                f"# max_epochs is held at {MAX_EPOCHS} on every cell and does\n"
                f"# not bind: at the far end patience equals the budget, so\n"
                f"# early stopping cannot fire before it\n"
                f"# (phase26.THE_SECOND_FACTOR_RULED).\n"
            )
            if anchor:
                head += (
                    "#\n"
                    "# **THIS IS THE ANCHOR CELL.** It is the probe's own\n"
                    "# recipe and must reproduce PCC 0.2520 at sd 0.0148.\n"
                    "# If it does not, the run is wrong and no other cell\n"
                    "# may be read (phase26.EXIT_CRITERIA criterion 9).\n"
                )
            head += (
                "#\n"
                "# GENERATED by scripts/generate_phase26_configs.py. Do not\n"
                "# hand-edit: --check fails on drift.\n\n"
            )
            body = {
                "schema_version": 1,
                "phase": "p26",
                "tier": "keeper",
                "seed": probe["seed"],
                "inputs": probe["inputs"],
                "task": {
                    "kind": "train_cv",
                    "manifest_artifact": task["manifest_artifact"],
                    "staged_artifact": task["staged_artifact"],
                    "embeddings_artifact": task["embeddings_artifact"],
                    **carried,
                    "max_epochs": MAX_EPOCHS,
                    "patience": patience,
                    "weight_decay": decay,
                },
            }
            texts[stem] = head + yaml.safe_dump(body, sort_keys=False)
    return texts


def _table_config() -> dict:
    # **Sixteen cells, sixteen distinct rollups.** Two directories of
    # nearly identical content hashing the same is not agreement, it
    # is one of them pasted twice.
    repeated = {
        digest for digest in RUN_ROLLUPS.values()
        if list(RUN_ROLLUPS.values()).count(digest) > 1
    }
    if repeated:
        raise DriftError(
            f"{len(repeated)} rollup(s) appear more than once: "
            f"{sorted(repeated)}. Sixteen cells differing only in two "
            "scalar settings produce directories of nearly identical "
            "size, so a repeat means a paste error rather than two "
            "cells that agree."
        )
    for stem, digest in RUN_ROLLUPS.items():
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise DriftError(f"{stem}: {digest!r} is not a sha256 digest")
    probe = _load(PROBE)
    manifest = next(
        entry for entry in probe["inputs"] if entry["name"] == "manifest_v1"
    )
    inputs = [dict(manifest)]
    arms = []
    for index_w, decay in enumerate(WEIGHT_DECAYS):
        for index_p, patience in enumerate(PATIENCES):
            stem = _stem(index_w, index_p)
            if stem not in RUN_DIRS:
                raise DriftError(
                    f"no supplied run directory for {stem}. Names are "
                    "supplied, never constructed from a stem."
                )
            if stem not in RUN_ROLLUPS:
                raise DriftError(
                    f"no declared rollup for {stem}. Hashes come from "
                    "scripts/declare_inputs.py on the cluster and are "
                    "transcribed, never derived here."
                )
            inputs.append({
                "name": stem,
                "path": f"{CLUSTER_ROOT}/runs/keeper/p26/{RUN_DIRS[stem]}",
                "rollup_sha256": RUN_ROLLUPS[stem],
            })
            arms.append({
                "name": stem,
                "input": stem,
                "seeds": list(probe["task"]["seeds"]),
                "n_patients": 237,
                "csv": "predictions",
                "weight_decay": decay,
                "patience": patience,
            })

    head = (
        "# PHASE 26 -- the six-metric union over the sixteen cells.\n"
        "#\n"
        "# Reads each cell's banked predictions and reports PCC, shrinkage,\n"
        "# RMSE, MAE, three-class accuracy and IEM together, with the\n"
        "# constant-predictor IEM floor recomputed here and carried beside\n"
        "# every IEM figure (phase26.EXIT_CRITERIA criteria 1 and 2).\n"
        "#\n"
        "# FULLY DECLARED 2026-09-06. Paths and rollups are both filled,\n"
        "# so guard 3 is live on all sixteen cells and the table will\n"
        "# refuse if any directory has moved since it was hashed.\n"
        "#\n"
        "# The run directories are SUPPLIED names, not names derived from\n"
        "# the cell stems, because job ids abbreviate unpredictably and a\n"
        "# constructed directory is one nobody declared. The rollups are\n"
        "# transcribed from scripts/declare_inputs.py on the cluster,\n"
        "# never derived here.\n"
        "#\n"
        "# **All sixteen rollups are distinct, and the generator refuses\n"
        "# if any two agree.** The cells differ only in weight decay and\n"
        "# patience, so their directories are nearly identical in size\n"
        "# and a repeated hash means one cell was pasted twice rather\n"
        "# than two cells that happen to match.\n"
        "#\n"
        "# The Phase 18 constant-predictor IEM is declared for the\n"
        "# cross-check criterion 2 requires. It was READ from that run's\n"
        "# metrics.json, never recomputed here, because two independent\n"
        "# computations are the point (phase26.THE_FLOOR_RULED).\n"
        "#\n"
        "# GENERATED by scripts/generate_phase26_configs.py.\n\n"
    )
    body = {
        "schema_version": 1,
        "phase": "p26",
        "tier": "keeper",
        "seed": probe["seed"],
        "inputs": inputs,
        "task": {
            "kind": "p26_calibration_table",
            "manifest_artifact": "manifest_v1",
            "arms": arms,
            "seeds": list(probe["task"]["seeds"]),
            "expect_cells": len(WEIGHT_DECAYS) * len(PATIENCES),
            "expect_patients": 237,
            "anchor_cell": _stem(
                WEIGHT_DECAYS.index(0.01), PATIENCES.index(5)
            ),
            "anchor_pcc": 0.2520,
            "anchor_sd": 0.0148,
            "phase18_floor_iem": PHASE18_FLOOR_IEM,
        },
    }
    return {"p26_calibration_table": head + yaml.safe_dump(body, sort_keys=False)}


def build() -> dict:
    texts = _cell_configs()
    texts.update(_table_config())
    return texts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    texts = build()
    if args.check:
        drifted = []
        for stem, text in sorted(texts.items()):
            path = CONFIGS / f"{stem}.yaml"
            if not path.is_file():
                drifted.append(f"MISSING {path.name}")
            elif path.read_text(encoding="utf-8") != text:
                drifted.append(f"DRIFT {path.name}: differs from the derived config")
        for line in drifted:
            print(f"  {line}")
        print(f"{len(texts)} configs checked, {len(drifted)} drifted")
        return 1 if drifted else 0

    for stem, text in sorted(texts.items()):
        (CONFIGS / f"{stem}.yaml").write_text(text, encoding="utf-8")
        print(f"wrote {stem}.yaml")
    print(
        f"{len(WEIGHT_DECAYS)} x {len(PATIENCES)} cells plus the table. "
        "Run directories are PENDING and the launches are the maintainer's."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
