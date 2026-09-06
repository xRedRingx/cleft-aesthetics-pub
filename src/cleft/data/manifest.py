"""Build the patient manifest.

**Frontal is assigned by score-sheet membership. The scored id IS the frontal.**
The odd/even rule is a cross-check and nothing else.

Phase 1 brief §1.2: the rule holds for 235 of 237 folders and is wrong for two.
Folder 143 holds images 523 and 524; the rule predicts 523 and the frontal is
524. Building from the rule would exchange that patient's two views -- the basal
scored as frontal and the frontal as basal -- and nothing would error, no count
would change, and every downstream number would be computed on the wrong image.

So the rule runs anyway, on every folder, purely to be disagreed with. It must
disagree in exactly folders 143 and 238. A third disagreement means either the
data or an assumption has changed, and the build stops.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

#: Folders at or below this take the odd image as frontal; above it, the even.
RULE_SPLIT = 171

#: The only folders where the rule may disagree with the score sheet. Held here
#: as the *expected disagreement set*, never consulted by ``frontal_by_rule`` --
#: a cross-check that knows the answers is not a check.
KNOWN_RULE_EXCEPTIONS = frozenset({143, 238})

#: Folder 238 is the single-image patient; every other folder holds two.
SINGLE_IMAGE_FOLDER = 238
IMAGES_PER_FOLDER = 2


#: manifest.csv opens with a tier-marker comment line, so ``csv.DictReader`` on
#: the raw file takes THAT as the header and every column name comes out wrong.
#: The marker is a safety feature and stays; ``load_manifest`` below is how the
#: file is read, and MANIFEST.json records the layout so a consumer never has to
#: discover this by experiment.
COMMENT_PREFIX = "#"

MANIFEST_COLUMNS: tuple[tuple[str, str], ...] = (
    ("patient_id", "folder number, 1-238"),
    ("frontal_id", "image id of the frontal view, from score-sheet lookup"),
    ("basal_id", "image id of the basal view; EMPTY for patient 238"),
    ("mean", "mean of 5 raters on the raw 1-5 scale -- the PRIMARY target"),
    ("median", "median of 5 raters -- ordinal secondary"),
    ("mode", "mode of 5 raters, ties broken toward the lower grade"),
    ("weighted_mean", "reliability-weighted mean; weights are item-total r"),
    ("orthodontist", "the orthodontist's own grade -- single-rater target"),
    ("soft_1", "fraction of raters awarding grade 1"),
    ("soft_2", "fraction of raters awarding grade 2"),
    ("soft_3", "fraction of raters awarding grade 3"),
    ("soft_4", "fraction of raters awarding grade 4"),
    ("soft_5", "fraction of raters awarding grade 5"),
    ("class3", "3-class collapse of the mean at fixed thresholds 2.5/3.5"),
    ("fold", "cross-validation test fold, 0-4"),
)


class ManifestError(ValueError):
    """A manifest invariant failed. Always fatal; never a warning."""


def manifest_schema() -> dict:
    """The layout of manifest.csv, for recording inside the artifact."""
    return {
        "file": "manifest.csv",
        "tier": "CLUSTER-ONLY",
        "comment_lines_before_header": 1,
        "note": (
            "The first line is a tier marker beginning with '#'. csv.DictReader on "
            "the raw file would take it as the header. Use "
            "cleft.data.manifest.load_manifest(), or skip leading '#' lines."
        ),
        "columns": [{"name": n, "description": d} for n, d in MANIFEST_COLUMNS],
        "fold_column": "fold",
    }


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read manifest.csv, skipping the tier-marker comment line(s).

    The skip itself lives in ``cleft.cluster_csv``, shared with the other
    tier-marked CSVs this project writes -- there were three implementations
    of it before that module and the marker had cost three rounds. The column
    check below stays here, because it is about the manifest specifically.
    """
    from ..cluster_csv import ClusterCsvError, read_cluster_csv

    try:
        parsed = read_cluster_csv(path)
    except ClusterCsvError as error:
        raise ManifestError(str(error)) from error
    expected = [name for name, _ in MANIFEST_COLUMNS]
    if parsed and list(parsed[0]) != expected:
        raise ManifestError(
            f"{path} columns are {list(parsed[0])}, expected {expected}. If the "
            "first column looks like a comment, the tier marker is being read as "
            "the header."
        )
    return parsed


@dataclass(frozen=True)
class Row:
    patient_id: int
    frontal_id: int
    basal_id: int | None


@dataclass(frozen=True)
class Counts:
    """The measured cohort shape, asserted when a build declares it."""

    patients: int
    frontal: int
    basal: int
    photoless: int


#: Phase 1 brief §1.1. 237 + 14 = 251.
REAL_COHORT = Counts(patients=237, frontal=237, basal=236, photoless=14)


@dataclass
class Manifest:
    rows: list[Row] = field(default_factory=list)
    photoless_ids: list[int] = field(default_factory=list)
    rule_disagreements: set[int] = field(default_factory=set)
    n_images: int = 0
    #: Directories that exist but hold no photograph. Folder 52 is one, and it is
    #: PRESENT AND EMPTY rather than missing -- recorded, never inferred from a
    #: gap in the numbering.
    empty_folders: list[int] = field(default_factory=list)

    @property
    def n_scored_rows(self) -> int:
        return len(self.rows) + len(self.photoless_ids)


# --------------------------------------------------------------------------
# the two assignment methods
# --------------------------------------------------------------------------


def frontal_by_rule(folder_id: int, image_ids: list[int]) -> int | None:
    """The odd/even rule. **Never used to build the manifest.**

    Deliberately naive: it knows nothing about folders 143 or 238. Special-casing
    them here would make it agree with the lookup by construction and destroy its
    value as an independent check. Returns None when no image has the required
    parity, which is itself a disagreement.
    """
    want_odd = folder_id <= RULE_SPLIT
    for image_id in sorted(image_ids):
        if (image_id % 2 == 1) == want_odd:
            return image_id
    return None


def frontal_by_lookup(image_ids: list[int], scored_ids: set[int]) -> int:
    """The real method: the image that appears in the score sheet is the frontal."""
    matches = [i for i in sorted(image_ids) if i in scored_ids]
    if not matches:
        raise ManifestError(
            f"no scored image among {sorted(image_ids)}: this folder has no frontal"
        )
    if len(matches) > 1:
        raise ManifestError(
            f"more than one scored image in one folder: {matches}. Exactly one "
            "image per patient is scored, and it is the frontal."
        )
    return matches[0]


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


def build(
    folders: dict[int, list[int]],
    scored_ids: set[int],
    *,
    scored_sequence: list[int] | None = None,
    expected: Counts | None = None,
) -> Manifest:
    """Build the manifest from the folder tree and the scored ids.

    ``scored_sequence`` is the score-sheet ids *in order, with duplicates*, so a
    repeated id can be detected -- a set has already lost that information.
    ``expected`` asserts the measured cohort shape; pass ``REAL_COHORT`` for the
    real build.
    """
    if scored_sequence is not None:
        _reject_duplicates(scored_sequence)

    rows: list[Row] = []
    disagreements: set[int] = set()
    matched: set[int] = set()
    n_images = 0

    empty_folders: list[int] = []

    for patient_id in sorted(folders):
        image_ids = list(folders[patient_id])
        n_images += len(image_ids)

        # A directory that exists with no photograph is not a patient. Folder 52
        # is exactly this. It is recorded rather than skipped silently, because
        # "empty" and "missing" are different facts and only one of them is true.
        if not image_ids:
            empty_folders.append(patient_id)
            continue

        _check_image_count(patient_id, image_ids)

        try:
            frontal = frontal_by_lookup(image_ids, scored_ids)
        except ManifestError as exc:
            raise ManifestError(f"folder {patient_id}: {exc}") from None

        matched.add(frontal)
        others = [i for i in image_ids if i != frontal]
        basal = others[0] if others else None

        if frontal_by_rule(patient_id, image_ids) != frontal:
            disagreements.add(patient_id)

        rows.append(Row(patient_id=patient_id, frontal_id=frontal, basal_id=basal))

    _check_rule_disagreements(disagreements)

    photoless = sorted(scored_ids - matched)
    result = Manifest(
        rows=rows,
        photoless_ids=photoless,
        rule_disagreements=disagreements,
        n_images=n_images,
        empty_folders=empty_folders,
    )

    if expected is not None:
        _check_counts(result, expected)
    return result


def _reject_duplicates(scored_sequence: list[int]) -> None:
    seen: set[int] = set()
    duplicates = sorted({i for i in scored_sequence if i in seen or seen.add(i)})
    if duplicates:
        raise ManifestError(
            f"duplicate scored id(s) in the score sheet: {duplicates}. Each image "
            "is scored exactly once; a duplicate means two rows claim one patient."
        )


def _check_image_count(patient_id: int, image_ids: list[int]) -> None:
    if len(set(image_ids)) != len(image_ids):
        raise ManifestError(f"folder {patient_id}: repeated image id in {image_ids}")

    expected = 1 if patient_id == SINGLE_IMAGE_FOLDER else IMAGES_PER_FOLDER
    if len(image_ids) != expected:
        raise ManifestError(
            f"folder {patient_id}: found {len(image_ids)} images {sorted(image_ids)}, "
            f"expected {expected}. Every folder holds {IMAGES_PER_FOLDER} images "
            f"except folder {SINGLE_IMAGE_FOLDER}, which holds 1."
        )


def _check_rule_disagreements(disagreements: set[int]) -> None:
    if disagreements == set(KNOWN_RULE_EXCEPTIONS):
        return

    unexpected = sorted(disagreements - KNOWN_RULE_EXCEPTIONS)
    missing = sorted(KNOWN_RULE_EXCEPTIONS - disagreements)
    detail = []
    if unexpected:
        detail.append(f"unexpected disagreement(s) in folder(s) {unexpected}")
    if missing:
        detail.append(f"expected disagreement(s) absent in folder(s) {missing}")

    raise ManifestError(
        "the odd/even frontal rule cross-check failed: "
        + "; ".join(detail)
        + f". The rule must disagree with the score sheet in exactly folders "
        f"{sorted(KNOWN_RULE_EXCEPTIONS)} (143 -> 524 not 523; 238 -> single odd "
        "image 581). A third disagreement means the data or an assumption has "
        "changed, and the manifest must not be built until it is understood."
    )


def _check_counts(result: Manifest, expected: Counts) -> None:
    actual = Counts(
        patients=len(result.rows),
        frontal=sum(1 for r in result.rows if r.frontal_id is not None),
        basal=sum(1 for r in result.rows if r.basal_id is not None),
        photoless=len(result.photoless_ids),
    )
    if actual != expected:
        raise ManifestError(
            f"cohort shape does not match the measured facts.\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual}"
        )
    total = actual.patients + actual.photoless
    if total != expected.patients + expected.photoless:  # pragma: no cover - arithmetic
        raise ManifestError(f"score-sheet row count is {total}")
