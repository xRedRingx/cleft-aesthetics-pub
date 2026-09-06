"""Loading the score sheet.

Phase 1 brief §1.3. Two workbooks exist for the real data, **identical in
content** — one spells the grades out, one stores integers. The mapping was
verified across all 1,255 cells with zero missing:

    Excellent = 1   Good = 2   Fair = 3   Poor = 4   Very poor = 5

Note the lowercase "poor" in "Very poor". Some rater headers contain doubled
spaces. Both are the kind of detail that makes exact string matching fail on the
real file and pass on a tidy fixture, so the fixture is deliberately untidy.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.data import scoresheet as S

from fixtures import cohort as C


# Module-scoped: a 251-row cohort and an .xlsx round trip are not free, and every
# test below only reads them. The defect tests write their own workbooks into the
# same directory under distinct names, which is safe because none of them mutate
# the shared ones.
@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return C.build_cohort(tmp_path_factory.mktemp("cohort"))


@pytest.fixture(scope="module")
def text_sheet(built):
    return C.write_scoresheet(built, built.scoresheet_text, form="text")


@pytest.fixture(scope="module")
def integer_sheet(built):
    return C.write_scoresheet(built, built.scoresheet_integer, form="integer")


# --------------------------------------------------------------------------
# the two forms must agree
# --------------------------------------------------------------------------


def test_text_and_integer_workbooks_load_identically(text_sheet, integer_sheet):
    """The central claim of §1.3: the two workbooks are the same data."""
    from_text = S.load(text_sheet)
    from_integer = S.load(integer_sheet)

    assert from_text.rows == from_integer.rows
    assert np.array_equal(from_text.grade_matrix(), from_integer.grade_matrix())


def test_grades_are_integers_on_the_1_to_5_scale(text_sheet):
    sheet = S.load(text_sheet)
    matrix = sheet.grade_matrix()
    assert matrix.dtype.kind in "iu"
    assert matrix.min() >= 1 and matrix.max() <= 5


def test_the_word_mapping_is_the_verified_one():
    assert S.parse_grade("Excellent") == 1
    assert S.parse_grade("Good") == 2
    assert S.parse_grade("Fair") == 3
    assert S.parse_grade("Poor") == 4
    assert S.parse_grade("Very poor") == 5


def test_very_poor_is_accepted_in_either_capitalisation(built):
    """§1.3: 'The loader accepts either form and produces the same result.'"""
    lower = C.write_scoresheet(built, built.root / "lower.xlsx", form="text")
    upper = C.write_scoresheet(
        built, built.root / "upper.xlsx", form="text", very_poor_capital=True
    )
    assert S.load(lower).rows == S.load(upper).rows
    assert S.parse_grade("Very poor") == S.parse_grade("Very Poor") == 5


def test_integers_are_accepted_as_written(integer_sheet):
    assert S.parse_grade(3) == 3
    assert S.parse_grade("3") == 3
    assert S.load(integer_sheet).n_cells == 251 * 5


# --------------------------------------------------------------------------
# header normalisation
# --------------------------------------------------------------------------


def test_doubled_spaces_in_headers_are_normalised(text_sheet):
    """The fixture writes 'Rater 8 -  Orthodontist' with a doubled space."""
    sheet = S.load(text_sheet)
    assert sheet.raters == C.RATERS, (
        "rater columns must be normalised back to their canonical names"
    )


def test_surrounding_whitespace_in_headers_is_stripped(text_sheet):
    sheet = S.load(text_sheet)
    assert all(name == name.strip() for name in sheet.raters)


def test_header_normalisation_is_idempotent():
    once = S.normalise_header("Rater 9  -   Speech and language therapist ")
    assert S.normalise_header(once) == once
    assert once == "Rater 9 - Speech and language therapist"


def test_a_missing_rater_column_is_rejected(built):
    path = C.write_scoresheet(
        built, built.root / "short.xlsx", form="text", defect="missing_rater"
    )
    with pytest.raises(S.ScoreSheetError, match="rater"):
        S.load(path)


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def test_row_and_cell_counts(text_sheet):
    """251 rows x 5 raters = 1,255 cells, zero missing."""
    sheet = S.load(text_sheet)
    assert len(sheet.rows) == 251
    assert len(sheet.raters) == 5
    assert sheet.n_cells == 1255
    assert sheet.n_missing == 0


def test_declared_row_count_is_asserted(text_sheet):
    S.load(text_sheet, expected_rows=251)
    with pytest.raises(S.ScoreSheetError, match="251|row"):
        S.load(text_sheet, expected_rows=250)


def test_rows_are_keyed_by_photo_id(text_sheet, built):
    sheet = S.load(text_sheet)
    assert set(sheet.rows) == set(built.scored_ids)
    assert all(isinstance(key, int) for key in sheet.rows)


def test_grade_matrix_row_order_matches_photo_id_order(text_sheet):
    sheet = S.load(text_sheet)
    matrix = sheet.grade_matrix()
    assert matrix.shape == (251, 5)
    for index, photo_id in enumerate(sheet.photo_ids()):
        assert list(matrix[index]) == list(sheet.rows[photo_id].grades)


def test_photo_ids_are_sorted(text_sheet):
    sheet = S.load(text_sheet)
    assert sheet.photo_ids() == sorted(sheet.photo_ids())


# --------------------------------------------------------------------------
# defects — every one must be named, never absorbed
# --------------------------------------------------------------------------


def test_a_missing_cell_is_rejected(built):
    """Zero missing is a measured fact; a blank cell means the file changed."""
    path = C.write_scoresheet(
        built, built.root / "gap.xlsx", form="text", defect="missing_cell"
    )
    with pytest.raises(S.ScoreSheetError) as excinfo:
        S.load(path)
    message = str(excinfo.value)
    assert "missing" in message.lower()
    assert "Rater 9" in message, "the message must name the offending rater column"


def test_an_unknown_grade_word_is_rejected(built):
    path = C.write_scoresheet(
        built, built.root / "word.xlsx", form="text", defect="bad_word"
    )
    with pytest.raises(S.ScoreSheetError, match="Adequate"):
        S.load(path)


def test_an_out_of_range_grade_is_rejected(built):
    path = C.write_scoresheet(
        built, built.root / "range.xlsx", form="integer", defect="out_of_range"
    )
    with pytest.raises(S.ScoreSheetError, match="6|range"):
        S.load(path)


def test_a_duplicate_photo_id_is_rejected(built):
    path = C.write_scoresheet(
        built, built.root / "dupe.xlsx", form="text", defect="duplicate_id"
    )
    with pytest.raises(S.ScoreSheetError, match="duplicate"):
        S.load(path)


def test_parse_grade_rejects_nonsense():
    for value in (None, "", "   ", "excellent-ish", 0, 6, 2.5):
        with pytest.raises(S.ScoreSheetError):
            S.parse_grade(value)


def test_the_average_column_is_not_trusted(text_sheet):
    """The sheet carries Average and Median columns; we recompute, never read.

    A stale precomputed column is exactly the kind of thing that silently
    disagrees with the raw grades.
    """
    sheet = S.load(text_sheet)
    assert not hasattr(next(iter(sheet.rows.values())), "average")
    assert "Average" not in sheet.raters
    assert "Median" not in sheet.raters
