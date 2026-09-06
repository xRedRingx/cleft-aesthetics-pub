"""The soft-label cross-check, and the policy for when it disagrees."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from cleft.data import labels as L
from cleft.data import softlabels as SL


def write_reference(path, rows, columns=SL.SOFT_COLUMNS, id_column="RanaPhotoID"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([id_column, *columns])
        for photo_id, values in rows.items():
            writer.writerow([photo_id, *values])
    return path


@pytest.fixture
def grades():
    return np.array([[1, 1, 3, 5, 5], [2, 2, 2, 2, 3], [4, 4, 4, 5, 5]])


@pytest.fixture
def computed(grades):
    soft = L.soft_labels(grades)
    return {101: soft[0], 102: soft[1], 103: soft[2]}


def test_agreement_passes(tmp_path, computed):
    reference = write_reference(
        tmp_path / "ref.csv", {pid: list(v) for pid, v in computed.items()}
    )
    SL.crosscheck(computed, SL.load_reference(reference))


def test_a_disagreement_names_the_patients_and_the_size(tmp_path, computed):
    rows = {pid: list(v) for pid, v in computed.items()}
    rows[102] = [0.0, 0.6, 0.4, 0.0, 0.0]  # ours is [0, 0.8, 0.2, 0, 0]
    reference = write_reference(tmp_path / "ref.csv", rows)

    with pytest.raises(SL.SoftLabelMismatch) as excinfo:
        SL.crosscheck(computed, SL.load_reference(reference))

    message = str(excinfo.value)
    assert "102" in message, "the message must name which patient disagrees"
    assert "0.2000" in message, "and by how much"
    assert "1 of 3" in message


def test_the_message_states_the_policy(tmp_path, computed):
    rows = {pid: list(v) for pid, v in computed.items()}
    rows[101] = [1.0, 0.0, 0.0, 0.0, 0.0]
    reference = write_reference(tmp_path / "ref.csv", rows)

    with pytest.raises(SL.SoftLabelMismatch) as excinfo:
        SL.crosscheck(computed, SL.load_reference(reference))

    message = str(excinfo.value)
    assert "source of truth" in message
    assert "do NOT resolve this by preferring either side" in message
    assert "widen the tolerance" in message, (
        "the message must forbid the tempting fix, because a tolerance chosen "
        "after seeing the disagreement is fitted to hide it"
    )


def test_a_disagreement_stops_the_run_rather_than_choosing_a_winner(tmp_path, computed):
    """It raises. It does not return 'ours' with a warning."""
    rows = {pid: list(v) for pid, v in computed.items()}
    rows[103] = [0.2, 0.2, 0.2, 0.2, 0.2]
    reference = write_reference(tmp_path / "ref.csv", rows)
    with pytest.raises(SL.SoftLabelMismatch):
        SL.crosscheck(computed, SL.load_reference(reference))


def test_a_totally_different_keying_is_reported_as_such(tmp_path, computed):
    reference = write_reference(tmp_path / "ref.csv", {9001: [0.2] * 5})
    with pytest.raises(SL.SoftLabelMismatch, match="share no photo ids"):
        SL.crosscheck(computed, SL.load_reference(reference))


def test_partial_coverage_is_reported(tmp_path, computed):
    rows = {pid: list(v) for pid, v in computed.items()}
    rows.pop(103)
    reference = write_reference(tmp_path / "ref.csv", rows)
    with pytest.raises(SL.SoftLabelMismatch, match="coverage differs"):
        SL.crosscheck(computed, SL.load_reference(reference))


def test_tolerance_absorbs_decimal_formatting_only(tmp_path, computed):
    rows = {pid: [f"{x:.10f}" for x in v] for pid, v in computed.items()}
    reference = write_reference(tmp_path / "ref.csv", rows)
    SL.crosscheck(computed, SL.load_reference(reference))

    # One fifth of a rater is far outside it, which is the smallest real error.
    assert SL.TOLERANCE < 0.2


def test_missing_soft_columns_are_reported(tmp_path, computed):
    reference = write_reference(
        tmp_path / "ref.csv", {101: [0.2, 0.2, 0.2]}, columns=("soft_1", "soft_2", "soft_3")
    )
    with pytest.raises(SL.SoftLabelError, match="missing column"):
        SL.load_reference(reference)


def test_an_unrecognised_id_column_is_reported(tmp_path):
    reference = write_reference(
        tmp_path / "ref.csv", {101: [0.2] * 5}, id_column="patientNumber"
    )
    with pytest.raises(SL.SoftLabelError, match="no id column"):
        SL.load_reference(reference)


def test_a_duplicate_id_in_the_reference_is_rejected(tmp_path):
    path = tmp_path / "ref.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["RanaPhotoID", *SL.SOFT_COLUMNS])
        writer.writerow([101, 0.2, 0.2, 0.2, 0.2, 0.2])
        writer.writerow([101, 0.2, 0.2, 0.2, 0.2, 0.2])
    with pytest.raises(SL.SoftLabelError, match="duplicate"):
        SL.load_reference(path)


def test_a_bom_prefixed_header_is_handled(tmp_path, computed):
    """Excel exports commonly carry a UTF-8 BOM."""
    path = tmp_path / "bom.csv"
    body = "RanaPhotoID," + ",".join(SL.SOFT_COLUMNS) + "\n"
    for pid, values in computed.items():
        body += f"{pid}," + ",".join(f"{v}" for v in values) + "\n"
    path.write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
    SL.crosscheck(computed, SL.load_reference(path))
