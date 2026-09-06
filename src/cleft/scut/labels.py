"""SCUT-FBP5500's ratings, its label, and its official split. All measured.

**[MEASURED 2026-07-31] Nothing here is assumed from the dataset's description.**
The structure was read off the files before any of it was coded against, because
"presumably the mean" is exactly the kind of premise this project has been wrong
about before.

----------------------------------------------------------------------------
WHAT THE FILES ACTUALLY CONTAIN
----------------------------------------------------------------------------
``All_Ratings.xlsx`` is **long format** -- one row per (rater, image), columns
``Rater, Filename, Rating, original Rating`` -- not a wide matrix:

* **330,000 rating rows = 5,500 images x exactly 60 raters.** Uniform: every
  image has 60, no image has 59 or 61.
* Ratings are **integers 1-5**.
* The label is the **mean of the 60**, confirmed rather than assumed: it agrees
  with the shipped labels to **3.33e-07**, which is the six-decimal rounding in
  the text files and nothing else.
* Label range **1.0167 to 4.7500**, mean 2.9909, SD 0.6880 over the 5,500.

**``original Rating`` is populated for exactly 38,500 rows = 7 raters x 5,500
images**, and empty for the other 53. Seven of the sixty appear to have had their
scores adjusted with the original preserved. **This project uses ``Rating``**;
the observation is recorded because a column that is populated for one ninth of
the data is the kind of thing someone later mistakes for the label.

----------------------------------------------------------------------------
THE SPLIT IS READ, NEVER GENERATED
----------------------------------------------------------------------------
``train_test_files/split_of_60%training and 40%testing/{train,test}.txt`` --
**3,300 and 2,200**, disjoint, and their union is exactly the 5,500 in
``All_labels.txt``. Each line is ``filename rating``, so **the split files carry
the label themselves** and are the single source for both.

Generating a new split would make every number incomparable with the published
band the sanity check in PLAN Part 5 §3.1 relies on.

``5_folders_cross_validations_files/`` also ships a 5-fold CV partition. It is
**not** what this project uses -- the brief specifies the 60/40 split -- and it is
named here so nobody reaches for it by accident.

----------------------------------------------------------------------------
CM152, FOR THE THIRD TIME
----------------------------------------------------------------------------
Excluded from every variant (its landmark file is shipped empty). **[MEASURED]**
It is on the **test** side with label **1.816667**, so excluding it leaves
**3,300 train and 2,199 test** -- the pretraining *training* set is untouched.

That is an independent confirmation of ``dataset.CM152_DISPOSITION``, which
recorded the same rating and the same side in Phase 5 from a different reading.
"""

from __future__ import annotations

from pathlib import Path

from .dataset import EXCLUDED, DatasetError, SPLIT_DIR

#: [MEASURED] Structure of ``All_Ratings.xlsx``.
N_IMAGES = 5_500
N_RATERS = 60
N_RATING_ROWS = N_IMAGES * N_RATERS  # 330,000
RATING_VALUES = (1, 2, 3, 4, 5)

#: [MEASURED] Rows carrying a non-empty ``original Rating``: 7 raters x 5,500.
#: **Not the label.** Recorded so it is not mistaken for one.
N_ORIGINAL_RATING_ROWS = 38_500
N_RATERS_WITH_ORIGINAL = 7

#: [MEASURED] The label, over all 5,500.
LABEL_MIN = 1.016667
LABEL_MAX = 4.75
LABEL_MEAN = 2.9909
LABEL_SD = 0.6880

#: [MEASURED] Largest disagreement between the shipped labels and the mean of the
#: 60 ratings. Six-decimal rounding in the text files, nothing more -- which is
#: what establishes that the label IS that mean.
LABEL_VS_RATING_MEAN_MAX_DIFF = 3.33e-07

#: [MEASURED] The official 60/40 split.
N_TRAIN = 3_300
N_TEST = 2_200
N_TEST_AFTER_EXCLUSION = 2_199

ALL_LABELS_FILE = "train_test_files/All_labels.txt"

#: Shipped but NOT used: the brief specifies the 60/40 split, and mixing the two
#: would make runs incomparable. Named so it is not reached for by accident.
UNUSED_CV_DIR = "train_test_files/5_folders_cross_validations_files"


def read_labelled_split(root, side: str, exclude: bool = True) -> dict[str, float]:
    """``{stem: label}`` for one side of the official split.

    The split file carries the label, so this is one read rather than a join --
    and a join is where a row-order or key-mismatch defect would live.
    """
    if side not in ("train", "test"):
        raise DatasetError(f"unknown split side {side!r}; expected train or test")
    path = Path(root) / SPLIT_DIR / f"{side}.txt"
    if not path.is_file():
        raise DatasetError(
            f"{path} does not exist. The official split is READ, never "
            "generated: a new split would make every number incomparable with "
            "the published band."
        )

    labels: dict[str, float] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise DatasetError(f"{path}:{number}: expected 'filename rating', got {line!r}")
        stem = Path(parts[0]).stem
        if exclude and stem in EXCLUDED:
            continue
        labels[stem] = float(parts[1])
    return labels


def read_all_labels(root, exclude: bool = True) -> dict[str, float]:
    """``{stem: label}`` for all 5,500, from ``All_labels.txt``."""
    path = Path(root) / ALL_LABELS_FILE
    if not path.is_file():
        raise DatasetError(f"{path} does not exist")
    labels: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            stem, value = line.split()
            stem = Path(stem).stem
            if exclude and stem in EXCLUDED:
                continue
            labels[stem] = float(value)
    return labels


def split_report(root) -> dict:
    """The split as it will be used, with the counts asserted.

    Every number here was measured before it was coded against; this recomputes
    them from the files so a different SCUT copy cannot quietly differ.
    """
    train = read_labelled_split(root, "train")
    test = read_labelled_split(root, "test")
    train_all = read_labelled_split(root, "train", exclude=False)
    test_all = read_labelled_split(root, "test", exclude=False)

    overlap = set(train_all) & set(test_all)
    if overlap:
        raise DatasetError(
            f"{len(overlap)} image(s) appear in BOTH sides of the official "
            f"split, e.g. {sorted(overlap)[:5]}. Training on test images would "
            "inflate the correlation the sanity check in PLAN Part 5 §3.1 reads."
        )
    if len(train_all) != N_TRAIN or len(test_all) != N_TEST:
        raise DatasetError(
            f"official split is {len(train_all)}/{len(test_all)}, expected "
            f"{N_TRAIN}/{N_TEST}. This is not the shipped 60/40 split."
        )

    excluded_side = "test" if set(EXCLUDED) & set(test_all) else "train"
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_train_before_exclusion": len(train_all),
        "n_test_before_exclusion": len(test_all),
        "excluded": list(EXCLUDED),
        "excluded_side": excluded_side,
        "label_source": "the split files themselves (filename rating per line)",
        "note": (
            "The official 60/40 split, READ not generated. The split files carry "
            "the label, so this is one read rather than a join. CM152 is on the "
            "TEST side, so excluding it leaves the pretraining TRAINING set "
            "untouched at 3,300."
        ),
    }
