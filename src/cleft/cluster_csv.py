"""Reading the CSVs this project writes, which carry a tier marker.

Every CLUSTER-ONLY CSV here opens with a comment line naming its tier::

    # CLUSTER-ONLY: patient-keyed
    patient_id,truth,prediction,fold
    1,2.200000,2.492088,0

``csv.DictReader`` on the raw file takes **the marker** as the header, so every
column name comes out wrong and ``row["patient_id"]`` raises ``KeyError`` --
or worse, does not, and the caller reads a column that is not the one it
named.

----------------------------------------------------------------------------
WHY THIS IS A MODULE AND NOT A THIRD LOCAL SKIP
----------------------------------------------------------------------------
**[MEASURED 2026-08-02] This is the third time the marker has cost a round.**
The Phase 1 manifest found it first; ``phase3.load_geometry_rows`` met it
again in Phase 5; and ``phase7b.load_baseline_predictions`` met it in Phase 7B
reading the baseline's per-seed predictions. Before this module there were
**three independent implementations of one skip** -- ``load_manifest``
filtering ``#`` lines, and ``lines[1:]`` twice in ``phase3`` -- and a fourth
was about to be written.

That is the shape of the two ``is_absolute_path`` copies and of the
``check_pairing`` clause that would have been updated in one place only: one
rule, several implementations, and the drift arrives when one of them is
corrected. **``lines[1:]`` is already the weaker version** -- it skips exactly
one line whether or not that line is a marker, so a file with two comment
lines silently loses a data row and a file with none silently loses its
header.

The marker itself stays. It is a safety feature: a file that says CLUSTER-ONLY
on its first line cannot be pasted into a shareable artifact by accident.
"""

from __future__ import annotations

import csv
from pathlib import Path

#: A line beginning with this is metadata, not data. No data row can begin
#: with it -- every CSV here is keyed by a numeric id in the first column.
COMMENT_PREFIX = "#"

#: The layout ``phase3.write_outputs`` writes for a CV arm's out-of-fold
#: predictions. Recorded here so a second producer can match it exactly rather
#: than approximately -- the comparison in Phase 7B reads the baseline's file
#: and the winner's, and two layouts would pair different columns.
#:
#: **``phase3.write_outputs`` is NOT changed to use this.** Gate 1 compares
#: ``predictions.csv`` byte for byte across runs (``phase3.GATE1_REFERENCE``),
#: so touching the writer that produced the reference would invalidate the
#: determinism evidence to tidy a duplicate. The constants below mirror it and
#: a test asserts they still agree.
PREDICTIONS_MARKER = "# CLUSTER-ONLY: patient-keyed"
PREDICTIONS_COLUMNS = ("patient_id", "truth", "prediction", "fold")


class ClusterCsvError(ValueError):
    """A tier-marked CSV could not be read as one."""


def data_lines(text: str) -> list[str]:
    """The lines of ``text`` that are not tier markers, in order.

    Filters EVERY comment line rather than dropping a fixed count, which is
    the difference between this and the ``lines[1:]`` it replaces: a file that
    gains a second marker line still reads, and one that has none does not
    silently lose its header row.
    """
    return [
        line for line in text.splitlines()
        if line.strip() and not line.startswith(COMMENT_PREFIX)
    ]


def read_cluster_csv(path: str | Path, *, expect: tuple | None = None) -> list[dict]:
    """Read a tier-marked CSV into dicts, skipping the marker line(s).

    ``expect`` names the columns the caller depends on. Checked here rather
    than at the first ``row[...]``, because a header mismatch surfaces as a
    ``KeyError`` deep in a loop where it reads as a missing patient rather
    than as a file in the wrong format.
    """
    path = Path(path)
    rows = data_lines(path.read_text(encoding="utf-8"))
    if not rows:
        raise ClusterCsvError(f"{path} has no rows after its tier marker(s)")

    parsed = list(csv.DictReader(rows))
    if expect is not None:
        header = tuple(parsed[0]) if parsed else tuple()
        missing = [name for name in expect if name not in header]
        if missing:
            raise ClusterCsvError(
                f"{path} is missing column(s) {missing}; its header is "
                f"{list(header)}. If that header looks like a comment line, "
                "the file was read without skipping its tier marker."
            )
    return parsed


def write_predictions(path: str | Path, rows) -> None:
    """Write out-of-fold predictions in ``phase3.write_outputs``' layout.

    ``rows`` yields ``(patient_id, truth, prediction, fold)``. Six decimal
    places, the same as the frozen writer, so a file from either producer
    reads the same way and ``predictions_digest`` computes over the same
    precision.
    """
    lines = [PREDICTIONS_MARKER, ",".join(PREDICTIONS_COLUMNS)]
    for patient_id, truth, prediction, fold in rows:
        lines.append(
            f"{patient_id},{float(truth):.6f},{float(prediction):.6f},{fold}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
