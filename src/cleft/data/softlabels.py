"""Cross-check our soft labels against the ones someone else already computed.

``..._per_image_labels.csv`` carries ``soft_1..soft_5`` produced by different code
from the same workbook. ``labels.soft_labels`` recomputes them from the raw
grades.

**The policy, decided before seeing the data:**

* The **raw grades are the source of truth.** The CSV is a cross-check, in the
  same way the workbook's ``Average`` column is not read.
* A disagreement is **a finding to investigate and report** — not something to
  resolve by silently preferring either side. So it **stops the run** and names
  which patients disagree and by how much.

Deciding this in advance matters because the tempting move on the day is to add a
tolerance until it passes, and a tolerance chosen after seeing the disagreement
is a number fitted to make a problem disappear. If the two genuinely differ, that
means one of the two computations is wrong about real patients, and which one is
a question worth an hour rather than a shrug.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Soft labels are fractions of five raters, so every value is a multiple of 0.2
#: and an exact comparison is reasonable. The tolerance covers decimal formatting
#: in the CSV ("0.2" vs "0.20000000001"), nothing more. It is NOT a knob.
TOLERANCE = 1e-6

SOFT_COLUMNS = ("soft_1", "soft_2", "soft_3", "soft_4", "soft_5")
ID_CANDIDATES = ("RanaPhotoID", "photo_id", "image_id", "id")


class SoftLabelError(ValueError):
    """The reference file could not be read."""


class SoftLabelMismatch(ValueError):
    """Our soft labels and the reference disagree. A finding, not a nuisance."""


@dataclass(frozen=True)
class Disagreement:
    photo_id: int
    computed: tuple[float, ...]
    reference: tuple[float, ...]

    @property
    def max_abs_diff(self) -> float:
        return max(abs(a - b) for a, b in zip(self.computed, self.reference))


def load_reference(path: str | Path) -> dict[int, tuple[float, ...]]:
    """Read ``soft_1..soft_5`` keyed by photo id."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SoftLabelError(f"{path} has no rows")

    header = {name.strip(): name for name in rows[0]}
    id_column = next((header[c] for c in ID_CANDIDATES if c in header), None)
    if id_column is None:
        raise SoftLabelError(
            f"{path}: no id column. Looked for {ID_CANDIDATES}, found "
            f"{sorted(header)[:12]}"
        )
    missing = [c for c in SOFT_COLUMNS if c not in header]
    if missing:
        raise SoftLabelError(f"{path}: missing column(s) {missing}")

    out: dict[int, tuple[float, ...]] = {}
    for line, row in enumerate(rows, start=2):
        try:
            photo_id = int(str(row[id_column]).strip())
        except (TypeError, ValueError):
            raise SoftLabelError(
                f"{path} line {line}: id {row[id_column]!r} is not an integer"
            ) from None
        if photo_id in out:
            raise SoftLabelError(f"{path} line {line}: duplicate id {photo_id}")
        try:
            out[photo_id] = tuple(float(row[header[c]]) for c in SOFT_COLUMNS)
        except (TypeError, ValueError):
            raise SoftLabelError(
                f"{path} line {line} (id {photo_id}): a soft label is not a number"
            ) from None
    return out


def crosscheck(
    computed: dict[int, np.ndarray],
    reference: dict[int, tuple[float, ...]],
    *,
    tolerance: float = TOLERANCE,
    require_full_coverage: bool = True,
) -> None:
    """Raise if the two disagree, naming who and by how much."""
    shared = sorted(set(computed) & set(reference))
    if not shared:
        raise SoftLabelMismatch(
            "our soft labels and the reference file share no photo ids at all. "
            f"We have {len(computed)}, the reference has {len(reference)}. That is "
            "a keying problem, not a rounding one."
        )

    if require_full_coverage:
        only_ours = sorted(set(computed) - set(reference))
        only_theirs = sorted(set(reference) - set(computed))
        if only_ours or only_theirs:
            raise SoftLabelMismatch(
                "coverage differs.\n"
                f"  {len(only_ours)} id(s) only in ours, e.g. {only_ours[:8]}\n"
                f"  {len(only_theirs)} id(s) only in the reference, e.g. {only_theirs[:8]}\n"
                "Investigate before proceeding: the two files describe different "
                "populations, which is exactly the kind of mismatch that makes a "
                "number look wrong when it is merely computed over something else."
            )

    disagreements = []
    for photo_id in shared:
        ours = tuple(float(v) for v in np.asarray(computed[photo_id]).ravel())
        theirs = reference[photo_id]
        if len(ours) != len(theirs):
            raise SoftLabelMismatch(
                f"photo {photo_id}: {len(ours)} soft values against {len(theirs)}"
            )
        if any(abs(a - b) > tolerance for a, b in zip(ours, theirs)):
            disagreements.append(Disagreement(photo_id, ours, theirs))

    if not disagreements:
        return

    worst = sorted(disagreements, key=lambda d: d.max_abs_diff, reverse=True)
    lines = [
        f"    photo {d.photo_id}: ours {['%.3f' % v for v in d.computed]} vs "
        f"reference {['%.3f' % v for v in d.reference]} (max diff {d.max_abs_diff:.4f})"
        for d in worst[:10]
    ]
    raise SoftLabelMismatch(
        f"soft labels disagree for {len(disagreements)} of {len(shared)} patients "
        f"(tolerance {tolerance}).\n"
        + "\n".join(lines)
        + (f"\n    ... and {len(disagreements) - 10} more" if len(disagreements) > 10 else "")
        + "\n\n  The raw grades are the source of truth and this file is the "
        "cross-check, so do NOT resolve this by preferring either side. One of "
        "the two computations is wrong about real patients; find out which. Do "
        "not widen the tolerance to make this pass -- a tolerance chosen after "
        "seeing the disagreement is fitted to hide it."
    )
