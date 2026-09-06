"""The tier marker, and the trap it sets for ``csv.DictReader``.

**[MEASURED 2026-08-02] Three rounds lost to one line.** The Phase 1 manifest
met it first, ``phase3.load_geometry_rows`` met it again in Phase 5, and
``phase7b.load_baseline_predictions`` met it in Phase 7B. Phase 1's lesson was
recorded as *a test that conforms to a defect rather than catching it* --
those tests were written around the offset with ``lines[1:]`` instead of
asserting what a naive read does.

So the trap is documented here by something that FAILS rather than by a
comment: the naive reader is exercised directly and its wrongness asserted.
"""

from __future__ import annotations

import csv

import pytest

from cleft.cluster_csv import (
    PREDICTIONS_COLUMNS,
    PREDICTIONS_MARKER,
    ClusterCsvError,
    data_lines,
    read_cluster_csv,
    write_predictions,
)

SAMPLE = (
    "# CLUSTER-ONLY: patient-keyed\n"
    "patient_id,truth,prediction,fold\n"
    "1,2.200000,2.492088,0\n"
    "2,3.000000,2.881000,1\n"
)


def test_a_naive_dictreader_misreads_the_marker_as_the_header(tmp_path):
    """**The trap, asserted rather than described.**

    ``csv.DictReader`` on the raw file takes the tier marker as the header, so
    the only key is the comment text and ``row["patient_id"]`` raises. This is
    the failure that cost three rounds, and a test that merely worked around
    it would be the Phase 1 mistake repeated.
    """
    path = tmp_path / "predictions.csv"
    path.write_text(SAMPLE, encoding="utf-8")

    with open(path, newline="", encoding="utf-8") as handle:
        naive = list(csv.DictReader(handle))

    # The marker becomes the ONLY named column; every real column falls into
    # DictReader's restkey, unnamed.
    assert PREDICTIONS_MARKER in naive[0], (
        "the naive reader no longer misreads the marker, so this file no "
        "longer demonstrates the trap the module exists for"
    )
    assert "patient_id" not in naive[0]
    with pytest.raises(KeyError):
        naive[0]["patient_id"]

    # And the shared reader gets it right.
    rows = read_cluster_csv(path, expect=PREDICTIONS_COLUMNS)
    assert [r["patient_id"] for r in rows] == ["1", "2"]
    assert rows[0]["prediction"] == "2.492088"


def test_the_skip_survives_more_than_one_marker_and_none_at_all(tmp_path):
    """``lines[1:]`` -- the version this replaced -- drops exactly one line
    whether or not it is a marker. So a second comment line silently loses a
    DATA ROW, and a file with no marker silently loses its HEADER. Both are
    wrong in ways that produce a plausible table rather than an error."""
    two_markers = tmp_path / "two.csv"
    two_markers.write_text("# CLUSTER-ONLY: x\n# regenerated\n" + SAMPLE.split("\n", 1)[1],
                           encoding="utf-8")
    def weak(path):
        """What `lines[1:]` did -- drop exactly one line, marker or not."""
        return list(csv.DictReader(
            path.read_text(encoding="utf-8").splitlines()[1:]
        ))

    assert [r["patient_id"] for r in read_cluster_csv(two_markers)] == ["1", "2"]
    # The weak version keeps the SECOND marker as the header.
    assert list(weak(two_markers)[0])[0] == "# regenerated"
    assert "patient_id" not in weak(two_markers)[0]

    unmarked = tmp_path / "none.csv"
    unmarked.write_text(SAMPLE.split("\n", 1)[1], encoding="utf-8")
    assert [r["patient_id"] for r in read_cluster_csv(unmarked)] == ["1", "2"]
    # The weak version eats the header and reads row 1 as the column names.
    assert list(weak(unmarked)[0]) == ["1", "2.200000", "2.492088", "0"]


def test_a_missing_column_is_refused_where_the_file_is_read(tmp_path):
    """Not at the first ``row[...]`` deep in a loop, where a header mismatch
    reads as a missing patient rather than as a file in the wrong format."""
    path = tmp_path / "wrong.csv"
    path.write_text("# CLUSTER-ONLY: x\nid,value\n1,2\n", encoding="utf-8")
    with pytest.raises(ClusterCsvError, match="missing column"):
        read_cluster_csv(path, expect=PREDICTIONS_COLUMNS)

    empty = tmp_path / "empty.csv"
    empty.write_text("# CLUSTER-ONLY: x\n", encoding="utf-8")
    with pytest.raises(ClusterCsvError, match="no rows after its tier marker"):
        read_cluster_csv(empty)


def test_data_lines_keeps_order_and_drops_blanks():
    assert data_lines("# a\nx,y\n1,2\n\n3,4\n") == ["x,y", "1,2", "3,4"]


def test_the_writer_round_trips_through_the_reader(tmp_path):
    """Two producers write these files -- the frozen ``phase3.write_outputs``
    and Phase 7B's search -- so what one writes the other must read."""
    path = tmp_path / "out.csv"
    write_predictions(path, [(1, 2.2, 2.492088, 0), (2, 3.0, 2.881, 1)])

    rows = read_cluster_csv(path, expect=PREDICTIONS_COLUMNS)
    assert [r["patient_id"] for r in rows] == ["1", "2"]
    assert rows[0]["truth"] == "2.200000"
    assert rows[0]["prediction"] == "2.492088"
    assert rows[1]["fold"] == "1"
    assert path.read_text(encoding="utf-8").startswith(PREDICTIONS_MARKER + "\n")


def test_the_mirrored_layout_still_matches_the_frozen_writer(repo_root):
    """**``cluster_csv`` mirrors ``phase3.write_outputs``; it does not replace
    it.** Gate 1 compares ``predictions.csv`` byte for byte across runs, so
    changing the writer that produced ``GATE1_REFERENCE`` would invalidate the
    determinism evidence in order to remove a duplicate.

    Mirroring is only safe while the two agree, which is what this asserts --
    against the frozen module's source, so a change there fails here.
    """
    source = (repo_root / "src" / "cleft" / "train" / "phase3.py").read_text(
        encoding="utf-8"
    )
    assert f'rows = ["{PREDICTIONS_MARKER}", "{",".join(PREDICTIONS_COLUMNS)}"]' in source, (
        "phase3.write_outputs no longer writes the layout cluster_csv mirrors; "
        "the two producers would emit different files"
    )
    # And the six-decimal precision, which predictions_digest also assumes.
    assert '{truth:.6f},{prediction:.6f}' in source
