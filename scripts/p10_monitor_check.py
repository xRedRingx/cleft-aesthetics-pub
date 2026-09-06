"""Does inner-val MSE select against inner-val PCC, across all 25 folds?

    PYTHONPATH=src python scripts/p10_monitor_check.py \
        --run runs/keeper/p10/p10_cleftgnn__<sha>__p10-cleftgnn-6

Reads the per-seed ``curves.csv`` files the arm now writes and answers,
per fold: the correlation between per-epoch ``inner_val_mse`` and
``inner_val_pcc``, which epoch each would select, and where the
MSE-selected epoch ranks by PCC.

**Why it matters** (``phase10.MONITOR_SELECTS_AGAINST_PCC``): the arm
stops on ``inner_val_mse``, our own registered deviation. A model that
predicts near the label mean scores well on MSE precisely by NOT
varying, so if MSE and PCC move together across epochs, the stopping
rule systematically prefers the collapsed solution -- and the arm's
near-zero PCC would then be partly a measurement of our stopping rule
rather than of the architecture.

**Reads artifacts only** -- no torch, no model, no run. It works on the
laptop the moment the CSVs are in hand.

**The threshold is registered before the numbers**
(``phase10.MONITOR_SELECTS_AGAINST_PCC``): the pattern "holds generally"
if r(MSE, PCC) > 0 in at least 18 of 25 folds AND the MSE-selected epoch
differs from the PCC-best epoch in at least 18 of 25. Fewer than that in
either is a fold-level curiosity, not a systematic bias.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

#: Registered before the numbers -- see the module docstring.
MAJORITY_OF_25 = 18


def read_curves(run_dir: Path) -> dict:
    """``{(seed, fold): [(epoch, mse, pcc), ...]}`` from the arm's CSVs."""
    curves: dict = defaultdict(list)
    files = sorted(run_dir.glob("seed_*__curves.csv"))
    if not files:
        raise SystemExit(
            f"no seed_*__curves.csv in {run_dir}. Only runs from 2026-08-17 "
            "onward write them (phase10.CURVES_AND_STAGES_WRITTEN); before "
            "that the harness computed them and the task discarded them."
        )
    for path in files:
        seed = path.name.split("__")[0].removeprefix("seed_")
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                curves[(seed, int(row["fold"]))].append((
                    int(row["epoch"]),
                    float(row["inner_val_mse"]),
                    float(row["inner_val_pcc"]),
                ))
    return {key: sorted(rows) for key, rows in curves.items()}


def analyse_fold(rows) -> dict:
    """One fold: how MSE-selection and PCC-selection relate."""
    epochs = np.array([r[0] for r in rows])
    mse = np.array([r[1] for r in rows])
    pcc = np.array([r[2] for r in rows])
    finite = np.isfinite(mse) & np.isfinite(pcc)
    correlation = (
        float(np.corrcoef(mse[finite], pcc[finite])[0, 1])
        if finite.sum() > 2 and mse[finite].std() > 0 and pcc[finite].std() > 0
        else float("nan")
    )
    selected = int(epochs[np.argmin(mse)])
    pcc_best = int(epochs[np.nanargmax(pcc)])
    ranking = list(epochs[np.argsort(-pcc)])
    return {
        "n_epochs": len(rows),
        "r_mse_pcc": correlation,
        "mse_selected_epoch": selected,
        "pcc_best_epoch": pcc_best,
        "differ": selected != pcc_best,
        "pcc_rank_of_selected": ranking.index(selected) + 1,
        "pcc_at_selected": float(pcc[epochs == selected][0]),
        "pcc_best_value": float(np.nanmax(pcc)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="the arm's run directory")
    args = parser.parse_args(argv)

    curves = read_curves(Path(args.run))
    print(f"{'seed':>7} {'fold':>4} {'epochs':>6} {'r(MSE,PCC)':>11} "
          f"{'MSE-sel':>7} {'PCC-best':>8} {'rank':>5}")
    print("-" * 60)
    results = []
    for (seed, fold), rows in sorted(curves.items()):
        result = analyse_fold(rows)
        results.append(result)
        print(f"{seed:>7} {fold:>4} {result['n_epochs']:>6} "
              f"{result['r_mse_pcc']:>+11.4f} {result['mse_selected_epoch']:>7} "
              f"{result['pcc_best_epoch']:>8} "
              f"{result['pcc_rank_of_selected']:>3}/{result['n_epochs']}")

    correlations = np.array([r["r_mse_pcc"] for r in results])
    usable = correlations[np.isfinite(correlations)]
    positive = int((usable > 0).sum())
    differing = sum(1 for r in results if r["differ"])
    print()
    print(f"folds analysed          {len(results)}")
    print(f"r(MSE,PCC) computable   {len(usable)}")
    print(f"  median r              {float(np.median(usable)):+.4f}")
    print(f"  positive (low MSE <-> low PCC)  {positive} of {len(usable)}")
    print(f"MSE-selected != PCC-best          {differing} of {len(results)}")
    print()
    holds = positive >= MAJORITY_OF_25 and differing >= MAJORITY_OF_25
    print(
        f"REGISTERED THRESHOLD: both counts >= {MAJORITY_OF_25} of 25 -> "
        f"{'HOLDS GENERALLY' if holds else 'DOES NOT HOLD'}"
    )
    print(
        "If it holds, early stopping on inner_val_mse systematically "
        "selects collapsed solutions, and the arm's near-zero PCC is "
        "partly a measurement of our own registered deviation."
        if holds else
        "If it does not hold, fold 0 seed 1337 (r=+0.756) is a fold-level "
        "curiosity and the monitor is not implicated generally."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
