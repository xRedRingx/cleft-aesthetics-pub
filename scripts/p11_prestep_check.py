"""Answer the three registered hedging checks from the pre-step's own table.

    PYTHONPATH=src python scripts/p11_prestep_check.py \
        --per-arm runs/keeper/p11/<run-dir>/per_arm.csv

**A checker, not a run**: no RunContext, no artifacts, nothing declared or
claimed. It reads ``per_arm.csv`` -- which the pre-step already wrote --
and applies ``phase11.HEDGING_CHECK_REGISTERED``'s three thresholds,
every one of them fixed before this file existed.

The predictions being tested are the ones registered there: check 1
predicted REFUTED, check 2 predicted HELD, check 3 has no registered
prediction because the offline work does not bear on it. Printing the
prediction beside the outcome is the point -- a checker that only prints
the outcome lets a refutation read as a result.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cleft import phase11  # noqa: E402

BAND = "own"
ARM_OF_INTEREST = "p10_cleftgnn"


def read_per_arm(path: Path, band: str = BAND) -> dict:
    rows = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["band"] != band:
                continue
            rows[row["arm"]] = {
                "iem_a": float(row["iem_a"]),
                "iem_b": float(row["iem_b"]),
                "below_crossover_share": float(row["below_crossover_share"]),
                "low": float(row["prediction_min"]),
                "high": float(row["prediction_max"]),
            }
    if not rows:
        raise SystemExit(f"no rows for band {band!r} in {path}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-arm", required=True)
    parser.add_argument("--band", default=BAND)
    args = parser.parse_args(argv)

    rows = read_per_arm(Path(args.per_arm), args.band)
    spec = phase11.HEDGING_CHECK_REGISTERED["checks"]
    top5_a = [a for a, _ in sorted(rows.items(), key=lambda kv: kv[1]["iem_a"])][:5]
    top5_b = [a for a, _ in sorted(rows.items(), key=lambda kv: kv[1]["iem_b"])][:5]
    print(f"{len(rows)} arms, band {args.band!r}")
    print(f"top-5 under A: {top5_a}")
    print(f"top-5 under B: {top5_b}")
    print(f"disjoint: {not (set(top5_a) & set(top5_b))}")

    # ---- 1. the crossover explanation ---------------------------------
    shares = [cell["below_crossover_share"] for cell in rows.values()]
    ladder = statistics.median(shares)
    if ARM_OF_INTEREST not in rows:
        raise SystemExit(f"{ARM_OF_INTEREST} is not in the table")
    mine = rows[ARM_OF_INTEREST]["below_crossover_share"]
    ratio = mine / ladder if ladder else float("inf")
    held_1 = ratio >= 1.5
    print("\n1. CROSSOVER EXPLANATION")
    print(f"   {ARM_OF_INTEREST} below-crossover share {mine:.4f}")
    print(f"   ladder median {ladder:.4f}  ratio {ratio:.3f}  "
          f"(threshold 1.5)")
    print(f"   predicted {spec['1_crossover']['predicted']}")
    print(f"   OUTCOME: {'HELD' if held_1 else 'REFUTED'}")

    # ---- 2. the direction-of-hedge explanation ------------------------
    centre = lambda arm: (rows[arm]["low"] + rows[arm]["high"]) / 2.0
    mean_a = statistics.fmean(centre(a) for a in top5_a)
    mean_b = statistics.fmean(centre(a) for a in top5_b)
    gap = mean_a - mean_b
    held_2 = gap >= 0.3
    print("\n2. DIRECTION-OF-HEDGE EXPLANATION")
    print(f"   A's top-5 predict {mean_a:.4f} on average, B's {mean_b:.4f}")
    print(f"   gap {gap:+.4f}  (threshold +0.3)")
    print(f"   predicted {spec['2_direction_of_hedge']['predicted']}")
    print(f"   OUTCOME: {'HELD' if held_2 else 'REFUTED'}")

    # ---- 3. narrowest-spanning ----------------------------------------
    span = lambda arm: rows[arm]["high"] - rows[arm]["low"]
    spans = [span(a) for a in rows]
    top5_span = statistics.median(span(a) for a in top5_a)
    ladder_span = statistics.median(spans)
    held_3 = top5_span < ladder_span
    print("\n3. NARROWEST-SPANNING")
    print(f"   A's top-5 median span {top5_span:.4f}, ladder median "
          f"{ladder_span:.4f}")
    print("   predicted: NONE registered")
    print(f"   OUTCOME: {'HELD' if held_3 else 'REFUTED'}")

    print("\nCONSEQUENCE, per phase11.HEDGING_CHECK_REGISTERED:")
    if not held_1 and held_2:
        print("   HEDGING_MECHANISM_MEASURED stands as written")
    elif held_1:
        print("   the crossover IS involved -- "
              "HEDGING_MECHANISM_MEASURED takes a dated correction")
    else:
        print("   check 2 refuted -- the mechanism is one neither account "
              "named; the finding withdraws to a question")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
