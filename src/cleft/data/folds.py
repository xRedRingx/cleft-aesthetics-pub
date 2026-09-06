"""Cross-validation folds, generated once and reused forever.

Brief §2.5: ``StratifiedGroupKFold``, 5 folds, groups = patient, stratified on the
3-class label. PLAN §4.9 settles why they are needed at all -- CV is primary for
the ladder, and TSTR is one additional arm, so folds are required either way.

**Grouping is a no-op today and must be here anyway.** The Phase 1 manifest holds
one row per patient, so groups are 1:1 with rows and ``StratifiedGroupKFold``
reduces to ``StratifiedKFold``. Phase 2 adds the basal view and Phase 7 adds
patches, at which point one patient owns many rows and a patient spanning the
train/test boundary would leak. Writing the grouping in now means that change
does not require remembering to add it.

The fold file is consumed by every arm in the ladder, so a defect in it
contaminates every result at once rather than one of them. That is why
``verify()`` exists separately from ``generate()`` and is re-run on load: the
generator being correct is not evidence that the file on disk is (R7).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

N_FOLDS = 5
DEFAULT_SEED = 1337

#: The 3-class collapse: {1,2} -> 0, {3} -> 1, {4,5} -> 2.
CLASSES = (0, 1, 2)

#: Largest acceptable difference, across folds, in the count of any one class.
#:
#: A correct stratified split differs by at most 1 -- some folds get the ceiling
#: and some the floor. 2 leaves room for an awkward remainder without admitting a
#: genuinely unstratified split.
#:
#: This exists because "every fold contains all three classes" was the only check
#: here, and it passed happily on a split whose class-1 counts ranged 22..26 on
#: the real cohort. A presence check cannot see a balance failure.
MAX_CLASS_SPREAD = 2


class FoldError(ValueError):
    """A fold invariant failed. Always fatal."""


@dataclass
class Folds:
    n_folds: int
    seed: int
    #: patient id -> the fold in which that patient is a TEST case.
    assignments: dict[int, int] = field(default_factory=dict)
    class_counts: dict[int, dict[int, int]] = field(default_factory=dict)

    @property
    def n_patients(self) -> int:
        return len(self.assignments)

    def test_ids(self, fold: int) -> list[int]:
        return sorted(pid for pid, f in self.assignments.items() if f == fold)

    def train_ids(self, fold: int) -> list[int]:
        return sorted(pid for pid, f in self.assignments.items() if f != fold)

    def summary(self) -> dict:
        """Aggregates only -- SHAREABLE. No patient id appears here."""
        return {
            "n_folds": self.n_folds,
            "seed": self.seed,
            "n_patients": self.n_patients,
            "fold_sizes": [len(self.test_ids(f)) for f in range(self.n_folds)],
            "class_counts": {
                str(fold): dict(sorted(counts.items()))
                for fold, counts in sorted(self.class_counts.items())
            },
        }

    def as_dict(self) -> dict:
        """The full artifact. Patient-keyed, therefore CLUSTER-ONLY."""
        return {
            **self.summary(),
            "assignments": {str(k): v for k, v in sorted(self.assignments.items())},
        }


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def _validate_inputs(
    patient_ids: list[int], labels: np.ndarray, groups: list[int] | None
) -> np.ndarray:
    array = np.asarray(labels)
    if len(patient_ids) != array.shape[0]:
        raise FoldError(
            f"length mismatch: {len(patient_ids)} patient ids, {array.shape[0]} labels"
        )
    if array.ndim != 1:
        raise FoldError(f"labels must be one-dimensional, got shape {array.shape}")

    unknown = sorted(set(np.unique(array).tolist()) - set(CLASSES))
    if unknown:
        raise FoldError(
            f"labels outside the 3-class collapse {CLASSES}: {unknown}. Stratify on "
            "class3, not on the raw 1-5 grade."
        )

    if groups is None:
        duplicates = [pid for pid, n in Counter(patient_ids).items() if n > 1]
        if duplicates:
            raise FoldError(
                f"duplicate patient id(s) {sorted(duplicates)[:5]} with no groups "
                "given. Pass groups= when one patient owns several rows, or the "
                "same patient will land on both sides of a split."
            )
    else:
        if len(groups) != len(patient_ids):
            raise FoldError("groups must be the same length as patient_ids")
        by_group: dict[int, set[int]] = defaultdict(set)
        for group, label in zip(groups, array.tolist()):
            by_group[group].add(label)
        inconsistent = sorted(g for g, values in by_group.items() if len(values) > 1)
        if inconsistent:
            raise FoldError(
                f"inconsistent labels within group(s) {inconsistent[:5]}. One patient "
                "cannot belong to two classes; stratification would be meaningless."
            )

    counts = Counter(array.tolist())
    for cls in CLASSES:
        if counts.get(cls, 0) == 0:
            raise FoldError(f"class {cls} has no members; it cannot appear in any fold")
    return array


def generate(
    patient_ids: list[int],
    labels,
    *,
    n_folds: int = N_FOLDS,
    seed: int = DEFAULT_SEED,
    groups: list[int] | None = None,
) -> Folds:
    """Assign every patient to exactly one test fold.

    ``seed`` is **recorded but does not affect the partition.** The splitter runs
    with ``shuffle=False`` because that is what stratifies well, and it is then a
    deterministic function of (groups, labels, n_folds). The seed is kept in the
    artifact for provenance and because later phases seed other things from the
    same config; do not read it as fold control. If folds ever need to *vary* --
    repeated CV, say -- that needs a different splitter, not a seed here, and the
    balance cost has to be measured rather than assumed.
    """
    array = _validate_inputs(list(patient_ids), labels, groups)
    effective_groups = list(groups) if groups is not None else list(patient_ids)

    counts = Counter(array.tolist())
    too_small = {c: n for c, n in counts.items() if n < n_folds}
    if too_small:
        detail = ", ".join(f"class {c} has {n}" for c, n in sorted(too_small.items()))
        raise FoldError(
            f"too few members to place one in every fold: {detail}, but there are "
            f"{n_folds} folds. Either reduce n_folds or collapse the class; do not "
            "emit folds with an empty class."
        )

    # Sort by patient id first so a differently ordered manifest cannot produce a
    # different split.
    order = np.argsort(np.asarray(effective_groups), kind="stable")
    ids_sorted = [patient_ids[i] for i in order]
    groups_sorted = [effective_groups[i] for i in order]
    labels_sorted = array[order]

    # shuffle=False, DELIBERATELY. This splitter orders groups by the standard
    # deviation of their class counts and assigns greedily to the emptiest fold;
    # that ordering IS the optimisation, and shuffle=True discards it.
    #
    # Measured on the real distribution (88/119/30, 5 folds): shuffle=False gives
    # a per-class spread of [1, 1, 0] -- folds of 18/24/6, essentially ideal.
    # shuffle=True gives [3, 4, 3], with class 1 ranging 22..26. The shuffle was
    # added for "better class balance" and did the opposite.
    #
    # It also bought nothing: with shuffle=False the partition is a deterministic
    # function of (groups, labels, n_splits). Permuting the input first was tested
    # and produced byte-identical membership for five different seeds, so the seed
    # cannot vary the split even indirectly. See ``seed`` in the docstring.
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=False)
    features = np.zeros((len(ids_sorted), 1))

    assignments: dict[int, int] = {}
    class_counts: dict[int, dict[int, int]] = {f: {c: 0 for c in CLASSES} for f in range(n_folds)}

    for fold, (_, test_index) in enumerate(
        splitter.split(features, labels_sorted, groups=groups_sorted)
    ):
        for i in test_index:
            assignments[ids_sorted[i]] = fold
            class_counts[fold][int(labels_sorted[i])] += 1

    result = Folds(
        n_folds=n_folds, seed=seed, assignments=assignments, class_counts=class_counts
    )
    verify(result, ids_sorted, labels_sorted, groups=groups_sorted)
    return result


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------


def verify(
    result: Folds,
    patient_ids: list[int],
    labels,
    *,
    groups: list[int] | None = None,
) -> None:
    """Re-check every invariant. Run at generation AND at load."""
    array = np.asarray(labels)
    unique_ids = sorted(set(patient_ids))

    missing = [pid for pid in unique_ids if pid not in result.assignments]
    if missing:
        raise FoldError(
            f"{len(missing)} patient(s) missing from the fold assignment, e.g. "
            f"{missing[:5]}. Every patient must appear in exactly one test fold."
        )

    extra = sorted(set(result.assignments) - set(unique_ids))
    if extra:
        raise FoldError(f"fold file names patients not in the cohort: {extra[:5]}")

    out_of_range = {
        pid: fold
        for pid, fold in result.assignments.items()
        if not 0 <= fold < result.n_folds
    }
    if out_of_range:
        raise FoldError(
            f"fold index out of range for {list(out_of_range.items())[:5]}; expected "
            f"0..{result.n_folds - 1}"
        )

    # Group integrity: every row of a patient must carry the same fold.
    if groups is not None:
        group_folds: dict[int, set[int]] = defaultdict(set)
        for group in groups:
            if group in result.assignments:
                group_folds[group].add(result.assignments[group])
        spanning = sorted(g for g, folds in group_folds.items() if len(folds) > 1)
        if spanning:
            raise FoldError(f"patient(s) {spanning[:5]} span more than one fold")

    # Recompute the class counts rather than trusting the recorded ones.
    recomputed: dict[int, Counter] = {f: Counter() for f in range(result.n_folds)}
    for pid, label in zip(patient_ids, array.tolist()):
        recomputed[result.assignments[pid]][int(label)] += 1

    for fold in range(result.n_folds):
        present = {c for c, n in recomputed[fold].items() if n > 0}
        if present != set(CLASSES):
            raise FoldError(
                f"fold {fold} does not contain all three classes: has {sorted(present)}"
            )
        recorded = {c: n for c, n in result.class_counts.get(fold, {}).items() if n > 0}
        if recorded != dict(recomputed[fold]):
            raise FoldError(
                f"fold {fold} class counts disagree with the assignment: recorded "
                f"{recorded}, actual {dict(recomputed[fold])}"
            )

    # Stratification actually took effect, not merely "every class is present".
    for cls in CLASSES:
        per_fold = [recomputed[f].get(cls, 0) for f in range(result.n_folds)]
        spread = max(per_fold) - min(per_fold)
        if spread > MAX_CLASS_SPREAD:
            raise FoldError(
                f"class {cls} is not stratified across folds: counts {per_fold}, "
                f"spread {spread} > {MAX_CLASS_SPREAD}. A correct stratified split "
                "differs by at most one. Check that y and groups reach the "
                "splitter, and that shuffle is off."
            )


# --------------------------------------------------------------------------
# the artifact
# --------------------------------------------------------------------------


def save(result: Folds, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load(path: str | Path) -> Folds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return Folds(
        n_folds=payload["n_folds"],
        seed=payload["seed"],
        assignments={int(k): int(v) for k, v in payload["assignments"].items()},
        class_counts={
            int(fold): {int(c): int(n) for c, n in counts.items()}
            for fold, counts in payload["class_counts"].items()
        },
    )
