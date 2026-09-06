"""Replay early-stopping policies over completed runs' curves.csv.

**Why [MEASURED 2026-07-31, the four SR-GNN scheme originals].** The scheme
ranking tracked epochs run, not scheme: anatomy and random both selected
epoch 5 and stopped at 13 because inner_val_mse never beat its epoch-5 value
-- for anatomy by a near-tie of 2.3e-05 -- while inner_val_pcc was still
climbing at every stopping point. Phase 3 gate 4 (the epoch policy) was
frozen UNSTRESSED because the frozen probe stopped at epochs 1-4; these are
the first runs to stress it, a phase early.

Before any policy change, the CONSISTENCY question: would the already-
completed runs have selected a different epoch under the candidate policy?
If yes, they are not comparable with runs made under the new policy and must
be re-run -- the same problem the `deterministic` field raised. This tool
answers that from each run's own curves.csv, replaying the loop's exact
semantics (``score < best - max(min_delta, 1e-12)``; selection = best epoch
at stop; patience counts non-improving epochs).

**The tool checks itself before it checks anything else.** When metrics.json
sits beside a curves.csv, the shipped-policy replay must reproduce that run's
recorded ``selected_epoch`` and ``epochs_run`` exactly. A replay that cannot
reproduce what actually happened has wrong semantics, and its counterfactual
columns would be fiction (a wrong check costs more than a wrong result).

Usage, from the repo root (works on laptop or cluster, no torch)::

    PYTHONPATH=src python scripts/replay_stopping.py runs/keeper/p6/*/curves.csv
    PYTHONPATH=src python scripts/replay_stopping.py --patience 8 <files...>

Columns: selected epoch, stopping epoch, and whether the run was cut by
patience or ran to its recorded end. CHANGED marks any policy whose SELECTED
epoch differs from the shipped policy's -- the checkpoint would hold
different weights, which is what decides re-running.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

#: The exact tie-breaker the loop uses: an improvement must beat the best by
#: more than this to reset patience. min_delta=0 replays shipped behaviour.
LOOP_EPSILON = 1e-12

#: The candidate grid. (inner_val_mse, 0.0) is the SHIPPED policy; min_delta
#: rows harden the monitor against near-ties; inner_val_pcc rows switch the
#: monitored quantity to the project's primary metric.
POLICIES = (
    ("inner_val_mse", 0.0),
    ("inner_val_mse", 1e-4),
    ("inner_val_mse", 1e-3),
    ("inner_val_pcc", 0.0),
    ("inner_val_pcc", 1e-4),
    ("inner_val_pcc", 1e-3),
)


def read_curves(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path}: no epochs recorded")
    return [
        {
            "epoch": int(row["epoch"]),
            "inner_val_mse": float(row["inner_val_mse"]),
            "inner_val_pcc": float(row["inner_val_pcc"]),
        }
        for row in rows
    ]


def replay(
    rows: list[dict], monitor: str, patience: int, min_delta: float
) -> dict:
    """The loop's early stopping, re-run over recorded epochs."""
    best = None
    best_epoch = 0
    since = 0
    for row in rows:
        raw = row[monitor] if monitor == "inner_val_mse" else -row[monitor]
        score = math.inf if math.isnan(raw) else raw
        if best is None or score < best - max(min_delta, LOOP_EPSILON):
            best, best_epoch = score, row["epoch"]
            since = 0
        else:
            since += 1
            if since >= patience:
                return {
                    "selected": best_epoch,
                    "stopped": row["epoch"],
                    "by": "patience",
                }
    return {"selected": best_epoch, "stopped": rows[-1]["epoch"], "by": "end"}


def self_check(path: Path, rows: list[dict], patience: int) -> str:
    """Reproduce the run's own recorded selection, or say why not."""
    metrics_path = path.parent / "metrics.json"
    if not metrics_path.is_file():
        return "no metrics.json beside the curves; shipped selection unverified"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    shipped_monitor = metrics.get("train_config", {}).get("monitor")
    shipped_patience = metrics.get("train_config", {}).get("patience", patience)
    outcome = replay(rows, shipped_monitor, shipped_patience, 0.0)
    recorded = (metrics.get("selected_epoch"), metrics.get("epochs_run"))
    replayed = (outcome["selected"], outcome["stopped"])
    if recorded == replayed:
        return (
            f"SELF-CHECK OK: replay reproduces the recorded selection "
            f"{recorded[0]} and stop {recorded[1]} under "
            f"({shipped_monitor}, patience {shipped_patience})"
        )
    return (
        f"SELF-CHECK FAILED: replay says {replayed}, metrics.json says "
        f"{recorded}. The replay's semantics do not match the loop that "
        "produced this run -- fix the tool before reading any other column."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("curves", nargs="+", type=Path)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    failed_self_check = False
    for path in args.curves:
        rows = read_curves(path)
        print(f"\n{path}")
        verdict = self_check(path, rows, args.patience)
        print(f"  {verdict}")
        failed_self_check |= verdict.startswith("SELF-CHECK FAILED")

        shipped = replay(rows, "inner_val_mse", args.patience, 0.0)
        print(f"  {'policy':<32}{'selected':>9}{'stopped':>9}{'by':>10}{'CHANGED':>9}")
        for monitor, min_delta in POLICIES:
            outcome = replay(rows, monitor, args.patience, min_delta)
            changed = outcome["selected"] != shipped["selected"]
            print(
                f"  {monitor + f', min_delta {min_delta:g}':<32}"
                f"{outcome['selected']:>9}{outcome['stopped']:>9}"
                f"{outcome['by']:>10}{'YES' if changed else '-':>9}"
            )
    return 1 if failed_self_check else 0


if __name__ == "__main__":
    sys.exit(main())
