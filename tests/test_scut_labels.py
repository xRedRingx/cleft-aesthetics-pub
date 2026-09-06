"""SCUT's label and official split (Phase 6).

The structure was MEASURED before being coded against. These tests pin what was
measured, on synthetic files -- the real copy is checked by
``scripts/verify_scut_labels.py``, which needs the 5,500 images.
"""

from __future__ import annotations

import pytest

from cleft.scut import dataset, labels


def write_split(root, train, test, all_labels=None):
    directory = root / dataset.SPLIT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    for side, rows in (("train", train), ("test", test)):
        (directory / f"{side}.txt").write_text(
            "".join(f"{stem}.jpg {value}\n" for stem, value in rows), encoding="utf-8"
        )
    if all_labels is not None:
        (root / labels.ALL_LABELS_FILE).parent.mkdir(parents=True, exist_ok=True)
        (root / labels.ALL_LABELS_FILE).write_text(
            "".join(f"{s}.jpg {v}\n" for s, v in all_labels), encoding="utf-8"
        )


def test_the_measured_structure_is_recorded():
    """**[MEASURED 2026-07-31] Confirmed, not assumed.** All_Ratings.xlsx is long
    format: 330,000 rows = 5,500 images x exactly 60 raters, uniform. The label
    is the mean of the 60 -- established by it agreeing with the shipped labels
    to 3.33e-07, which is the six-decimal rounding in the text files."""
    assert labels.N_RATING_ROWS == labels.N_IMAGES * labels.N_RATERS == 330_000
    assert labels.RATING_VALUES == (1, 2, 3, 4, 5)
    assert labels.LABEL_VS_RATING_MEAN_MAX_DIFF < 1e-6
    assert labels.N_TRAIN + labels.N_TEST == labels.N_IMAGES


def test_the_original_rating_column_is_recorded_as_not_the_label():
    """Populated for 7 raters x 5,500 = 38,500 rows and empty for the other 53.
    A column filled for one ninth of the data is exactly what someone later
    mistakes for the label."""
    assert labels.N_ORIGINAL_RATING_ROWS == labels.N_RATERS_WITH_ORIGINAL * 5_500
    assert "uses ``Rating``" in labels.__doc__


def test_the_cross_validation_partition_is_named_as_unused():
    """SCUT also ships a 5-fold CV split. The brief specifies 60/40, and mixing
    them would make runs incomparable -- so it is named rather than left as an
    unmarked directory someone reaches into."""
    assert "5_folders_cross_validations_files" in labels.UNUSED_CV_DIR
    assert "NOT used" in labels.__doc__ or "not** what this project uses" in labels.__doc__


def test_the_split_is_read_with_its_own_labels(tmp_path):
    """The split files carry ``filename rating``, so this is one read rather
    than a join -- and a join is where a key-mismatch defect would live."""
    write_split(tmp_path, [("AF1", 2.5), ("AF2", 3.0)], [("CF1", 4.1)])
    train = labels.read_labelled_split(tmp_path, "train")
    assert train == {"AF1": 2.5, "AF2": 3.0}
    assert labels.read_labelled_split(tmp_path, "test") == {"CF1": 4.1}


def test_cm152_is_excluded_and_it_is_on_the_test_side(tmp_path):
    """**[MEASURED] Third independent confirmation** of what Phase 5 recorded:
    CM152 sits in TEST at 1.816667, so excluding it leaves the pretraining
    TRAINING set untouched."""
    write_split(tmp_path, [("AF1", 2.5)], [("CM152", 1.816667), ("CF1", 4.1)])
    assert "CM152" not in labels.read_labelled_split(tmp_path, "test")
    assert labels.read_labelled_split(tmp_path, "test", exclude=False)["CM152"]

    report = labels.split_report.__doc__
    assert "asserted" in report
    assert dataset.CM152_DISPOSITION["official_split_side"] == "test"
    assert dataset.CM152_DISPOSITION["rating"] == 1.816667


def test_an_overlapping_split_is_refused(tmp_path):
    """Training on test images would inflate the very correlation the §3.1
    sanity check reads, and nothing downstream would look wrong."""
    write_split(tmp_path, [("AF1", 2.5)], [("AF1", 2.5)])
    with pytest.raises(dataset.DatasetError, match="BOTH sides"):
        labels.split_report(tmp_path)


def test_a_split_of_the_wrong_size_is_refused(tmp_path):
    """3,300/2,200 or it is not the shipped split, and a generated one would
    make every number incomparable with the published band."""
    write_split(tmp_path, [("AF1", 2.5)], [("CF1", 4.1)])
    with pytest.raises(dataset.DatasetError, match="not the shipped 60/40 split"):
        labels.split_report(tmp_path)


def test_a_missing_split_file_says_why_it_matters(tmp_path):
    with pytest.raises(dataset.DatasetError, match="READ, never"):
        labels.read_labelled_split(tmp_path, "train")


def test_a_malformed_line_is_refused_with_its_line_number(tmp_path):
    directory = tmp_path / dataset.SPLIT_DIR
    directory.mkdir(parents=True)
    (directory / "train.txt").write_text("AF1.jpg 2.5\nAF2.jpg\n", encoding="utf-8")
    with pytest.raises(dataset.DatasetError, match="train.txt:2"):
        labels.read_labelled_split(tmp_path, "train")


def test_an_unknown_side_is_refused():
    with pytest.raises(dataset.DatasetError, match="unknown split side"):
        labels.read_labelled_split(".", "validation")
