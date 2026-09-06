"""Phase 24: print the reliability curve, its inverse and the checks.

**Runs on the laptop and reads NOTHING.** No manifest, no score sheet,
no embeddings, no declared inputs -- the whole computation is
``reliability.spearman_brown`` applied to the banked constant
``MEAN_R_237 = 0.4696`` (phase24.THE_SETTINGS_RULED). There is no run
directory because there is nothing to declare.

    python scripts/phase24_reliability_curve.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from cleft import phase24  # noqa: E402


def main() -> int:
    print(phase24.render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
