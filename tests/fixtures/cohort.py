"""A synthetic cohort with the real shape and none of the real data.

The clinical cohort never leaves EHU infrastructure and is never read here. What
is reproduced is the *structure* described in the Phase 1 brief §1.1-§1.2, so the
invariants can be tested without the data:

* 237 folders numbered 1-238 with **52 absent**
* two images per folder, except folder 238 which has one
* folders 1-171: the odd image id is frontal; 172-237: the even id is frontal
* **folder 143** holds images 523 and 524, and 524 is the frontal -- the rule
  predicts 523
* **folder 238** holds a single image 581, odd despite being past 171
* a 251-row score sheet: 237 frontal ids plus 14 scored ids with no photograph

The two exception folders use their REAL image ids, because those are measured
facts and the whole phase turns on them. Every other id is synthetic and
deliberately drawn from a disjoint range: the real numbering has gaps that are
not documented anywhere, and inventing an arithmetic for it would be encoding a
guess as if it were measured. Nothing in the manifest builder may depend on id
arithmetic anyway -- that is the point of building by lookup.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# measured facts (Phase 1 brief §1.1, §1.2)
# --------------------------------------------------------------------------

FIRST_FOLDER, LAST_FOLDER = 1, 238

#: Folder 52 EXISTS and is EMPTY. It is not missing -- every one of the 238
#: numeric directories is present. The consequence is the same (237 patients have
#: photographs) but the condition is different, and code that infers absence from
#: a missing directory would be wrong about the actual tree.
EMPTY_FOLDER = 52

N_FOLDERS_WITH_PHOTOS = 237
N_DIRECTORIES = 238
N_IMAGES = 473

#: Measured census of the real tree, 2026-07-27:
#:   952 files = 473 images + 473 AppleDouble sidecars + 3 Thumbs.db + 3 ._Thumbs.db
#: The sidecars are exactly 4,096 bytes and carry a .jpg extension, so only a
#: basename rule can exclude them.
APPLEDOUBLE_BYTES = 4096
N_THUMBS_DB = 3
N_TOTAL_FILES = 952

#: Folders at or below this take the odd image as frontal; above it, the even.
RULE_SPLIT = 171

#: Exception A. The rule predicts 523; the frontal is 524.
FOLDER_143 = 143
FOLDER_143_IMAGES = (523, 524)
FOLDER_143_FRONTAL = 524
FOLDER_143_RULE_PREDICTS = 523

#: Exception B. One image, odd, despite being past the split. No basal.
FOLDER_238 = 238
FOLDER_238_IMAGE = 581

#: 14 scored ids with no photograph: 716, 718 ... 742, even and contiguous.
PHOTOLESS_IDS = tuple(range(716, 743, 2))

#: Synthetic ids start here, clear of 523/524, 581 and the photo-less block.
SYNTHETIC_ID_BASE = 1001

RATERS = (
    "Rater 7 - Cleft patient",
    "Rater 8 - Orthodontist",
    "Rater 9 - Speech and language therapist",
    "Rater 10 - Plastic surgeon",
    "Rater 11 - Psychologist",
)

GRADE_WORDS = {1: "Excellent", 2: "Good", 3: "Fair", 4: "Poor", 5: "Very poor"}


def numeric_folders() -> list[int]:
    """All 238 directories, including the empty folder 52."""
    return list(range(FIRST_FOLDER, LAST_FOLDER + 1))


def photo_folders() -> list[int]:
    """The 237 directories that actually hold photographs."""
    return [n for n in numeric_folders() if n != EMPTY_FOLDER]


@dataclass
class Cohort:
    """A built synthetic cohort."""

    root: Path
    folders: dict[int, list[int]] = field(default_factory=dict)
    frontal: dict[int, int] = field(default_factory=dict)
    basal: dict[int, int | None] = field(default_factory=dict)
    scored_ids: list[int] = field(default_factory=list)
    grades: dict[int, list[int]] = field(default_factory=dict)

    @property
    def images_dir(self) -> Path:
        return self.root / "patients"

    @property
    def scoresheet_csv(self) -> Path:
        return self.root / "scores.csv"

    @property
    def scoresheet_text(self) -> Path:
        return self.root / "scores_text.xlsx"

    @property
    def scoresheet_integer(self) -> Path:
        return self.root / "scores_integer.xlsx"


def _assign_images() -> tuple[dict[int, list[int]], dict[int, int]]:
    """Image ids per folder, and which one is frontal."""
    folders: dict[int, list[int]] = {}
    frontal: dict[int, int] = {}

    next_id = SYNTHETIC_ID_BASE
    for folder in photo_folders():
        if folder == FOLDER_143:
            folders[folder] = list(FOLDER_143_IMAGES)
            frontal[folder] = FOLDER_143_FRONTAL
            continue
        if folder == FOLDER_238:
            folders[folder] = [FOLDER_238_IMAGE]
            frontal[folder] = FOLDER_238_IMAGE
            continue

        # Consecutive (odd, even) pair, so the parity rule is well defined.
        if next_id % 2 == 0:
            next_id += 1
        pair = [next_id, next_id + 1]
        next_id += 2
        folders[folder] = pair
        frontal[folder] = pair[0] if folder <= RULE_SPLIT else pair[1]

    return folders, frontal


def build_cohort(
    root: Path,
    seed: int = 1337,
    broken: str | None = None,
    write_images: bool = False,
) -> Cohort:
    """Build a synthetic cohort at ``root``.

    ``write_images`` defaults to False. The cohort is 473 images, and a test that
    only exercises manifest logic works from the returned dicts and never touches
    the filesystem -- writing the tree for each of a dozen tests would cost
    thousands of file operations against a 60-second suite budget. Pass True only
    in the tests that genuinely scan the directory.

    ``broken`` injects one specific defect, for the tests that assert each is
    rejected with a message naming it:

    ``third_rule_violation``  a folder other than 143/238 disagrees with the rule
    ``missing_frontal``       a folder whose images contain no scored id
    ``duplicate_scored_id``   one image id scored twice
    ``three_images``          a folder with three images
    ``missing_cell``          a score-sheet row with a blank rater cell
    """
    root = Path(root)
    folders, frontal = _assign_images()

    if broken == "third_rule_violation":
        # Folder 7 is well below the split, so swap its frontal to the even id.
        folders[7] = list(folders[7])
        frontal[7] = folders[7][1]
    if broken == "three_images":
        folders[9] = folders[9] + [max(max(v) for v in folders.values()) + 1]

    # Folder 52 exists and holds nothing. It appears in the mapping with an empty
    # list, so downstream code sees "present and empty" rather than inferring
    # absence from a gap in the numbering.
    folders[EMPTY_FOLDER] = []

    basal = {
        folder: next((i for i in ids if i != frontal[folder]), None)
        for folder, ids in folders.items()
        if ids
    }

    scored = [frontal[f] for f in photo_folders()]
    if broken == "missing_frontal":
        # Folder 11's frontal is simply not in the score sheet.
        scored = [s for s in scored if s != frontal[11]]
    if broken == "duplicate_scored_id":
        scored.append(frontal[13])

    all_scored = scored + list(PHOTOLESS_IDS)

    # Grades from raters who PARTLY AGREE, via a latent per-patient truth plus
    # per-rater noise. Independent uniform grades would be easier, and were how
    # this fixture started, but they make every rater uncorrelated with every
    # other: item-total correlations sit at zero, and a reliability-weighted mean
    # is then genuinely undefined rather than merely unusual. The real panel has
    # a mean inter-rater r of 0.4696, and a fixture that cannot exercise the
    # weighted target is not testing the pipeline the cluster will run.
    rng = np.random.default_rng(seed)
    grades = {}
    for image_id in all_scored:
        truth = rng.uniform(1.0, 5.0)
        row = np.clip(
            np.rint(truth + rng.normal(0.0, 0.9, size=len(RATERS))), 1, 5
        ).astype(int)
        grades[image_id] = row.tolist()

    # --- write the folder tree -------------------------------------------
    root.mkdir(parents=True, exist_ok=True)
    if write_images:
        images = root / "patients"
        images.mkdir(parents=True, exist_ok=True)

        # Every numeric directory exists, including the empty folder 52.
        for folder in numeric_folders():
            (images / str(folder)).mkdir(exist_ok=True)

        for folder, ids in folders.items():
            folder_dir = images / str(folder)
            for image_id in ids:
                # Content, not real pixels: nothing here decodes an image.
                (folder_dir / f"{image_id}.jpg").write_bytes(
                    f"synthetic image {image_id}".encode("utf-8")
                )
                # The AppleDouble sidecar the real tree carries beside every
                # image: 4,096 bytes, .jpg extension, id in the stem. Only a
                # basename rule excludes it; extension filtering admits it and
                # stem parsing then sees a duplicate id.
                (folder_dir / f"._{image_id}.jpg").write_bytes(
                    b"\x00\x05\x16\x07" + b"\x00" * (APPLEDOUBLE_BYTES - 4)
                )

        # Three folders carry Thumbs.db and its sidecar. Windows regenerates
        # Thumbs.db on browse, so a rollup including it would be unstable.
        for folder in list(folders)[:N_THUMBS_DB]:
            (images / str(folder) / "Thumbs.db").write_bytes(b"thumbs")
            (images / str(folder) / "._Thumbs.db").write_bytes(b"\x00" * 82)

    # --- write the score sheet -------------------------------------------
    with (root / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["RanaPhotoID", *RATERS, "Average", "Median"])
        for image_id in all_scored:
            row_grades = list(grades[image_id])
            cells = [GRADE_WORDS[g] for g in row_grades]
            if broken == "missing_cell" and image_id == all_scored[3]:
                cells[2] = ""
            writer.writerow(
                [
                    image_id,
                    *cells,
                    f"{sum(row_grades) / len(row_grades):.4f}",
                    sorted(row_grades)[len(row_grades) // 2],
                ]
            )

    return Cohort(
        root=root,
        folders=folders,
        frontal=frontal,
        basal=basal,
        scored_ids=all_scored,
        grades=grades,
    )


# --------------------------------------------------------------------------
# score-sheet workbooks
# --------------------------------------------------------------------------

#: Two workbooks exist for the real data, identical in content: one spells the
#: grades out, the other stores integers. The loader must produce the same result
#: from either (brief §1.3).
FORMS = ("text", "integer")


def write_scoresheet(
    cohort: Cohort,
    path: Path,
    form: str = "text",
    *,
    doubled_spaces: bool = True,
    very_poor_capital: bool = False,
    defect: str | None = None,
) -> Path:
    """Write a synthetic score sheet as .xlsx, the way the real one ships.

    ``doubled_spaces`` reproduces the real header quirk -- some rater headers
    contain runs of whitespace -- so the loader is forced to normalise rather
    than match exact strings.

    ``defect`` injects one fault: ``missing_cell``, ``bad_word``,
    ``out_of_range``, ``duplicate_id``, ``missing_rater``.
    """
    from openpyxl import Workbook

    if form not in FORMS:  # pragma: no cover - programmer error
        raise ValueError(f"unknown form {form!r}")

    headers = list(RATERS)
    if doubled_spaces:
        # e.g. "Rater 8 -  Orthodontist"
        headers[1] = headers[1].replace(" - ", " -  ")
        headers[3] = f" {headers[3]} "
    if defect == "missing_rater":
        headers = headers[:-1]

    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["RanaPhotoID", *headers, "Average", "Median"])

    rows = list(cohort.scored_ids)
    if defect == "duplicate_id":
        rows.append(rows[0])

    for index, image_id in enumerate(rows):
        grades = list(cohort.grades[image_id])[: len(headers)]
        if form == "text":
            word = "Very Poor" if very_poor_capital else "Very poor"
            words = {**GRADE_WORDS, 5: word}
            cells = [words[g] for g in grades]
        else:
            cells = list(grades)

        if index == 3:
            if defect == "missing_cell":
                cells[2] = None
            elif defect == "bad_word":
                cells[2] = "Adequate"
            elif defect == "out_of_range":
                # 6 in either form. The text vocabulary only spans 1-5, so an
                # out-of-range value can only arrive as a number -- which is
                # exactly what a permissive loader would wave through.
                cells[2] = 6

        numeric = [g for g in grades]
        sheet.append(
            [
                image_id,
                *cells,
                sum(numeric) / len(numeric),
                sorted(numeric)[len(numeric) // 2],
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return path
