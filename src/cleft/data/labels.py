"""Label targets, the 3-class collapse, and how learnable each target is.

Phase 1 brief §2.2 and §1.7.

**Learnability** is leave-one-rater-out mean Pearson. Hold out rater j, rebuild
the target from the remaining four, correlate rater j's column against it, and
average over the five held-out raters. For a single-rater target there is nothing
to rebuild: correlate each of the other four raters against that rater's column
and average those four.

Measured on the 237 photographed patients, all to four decimal places:

    weighted mean  0.5926      mean  0.5900      median  0.5561
    mode           0.4880      orthodontist      0.4811

**A documented asymmetry, not a flaw.** Aggregate targets are measured against a
mean-of-four; the orthodontist target is a single column. A mean of four is
inherently less noisy, so part of the gap *is* exactly that. Lower noise is what
makes a target more learnable, so this does not invalidate the ordering -- but it
is not a like-for-like comparison of estimators either, and the write-up must say
so. ``LEARNABILITY_CAVEAT`` carries that sentence so it travels with the numbers.

The **attenuation ceiling** is the principled companion: sqrt(reliability of the
target), which has no asymmetry because it is the same construction for every
target. Mean of five -> sqrt(0.8158) = 0.9032; a single rater -> sqrt(0.4696) =
0.685. Same ordering, and it connects directly to PLAN §4.3 rather than being a
bespoke index. Both are reported; **the ordering is what is asserted.**

Learnability lives here rather than in ``reliability`` (brief §2.3) because it
depends on target construction, and pointing the dependency the other way would
make the two modules circular.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eval import metrics
from . import reliability as R

#: Aggregate targets, plus the single-rater one, in reporting order.
TARGETS: tuple[str, ...] = ("weighted_mean", "mean", "median", "mode", "orthodontist")

#: Targets built from every available rater column.
AGGREGATE_TARGETS = frozenset({"weighted_mean", "mean", "median", "mode"})

#: Measured on **all 251 scored rows** -- these are the values quoted in brief
#: §1.7. They are NOT the values the manifest will reproduce.
LEARNABILITY_251 = {
    "weighted_mean": 0.5926,
    "mean": 0.5900,
    "median": 0.5561,
    "mode": 0.4880,
    "orthodontist": 0.4811,
}

#: Measured on the **237 patients with photographs** -- the cohort the manifest
#: holds, and therefore what a cluster run actually produces. About 0.012 higher
#: throughout.
#:
#: Both populations are named separately for the same reason ``reliability_237``
#: and ``pcc_ceiling_237`` are: comparing a value computed on one population
#: against a value computed on another is exactly how the 0.903 ceiling came to
#: be "corrected" to 0.807. A number that looks wrong is usually a different
#: quantity or a different population before it is an error (R2).
#: CORRECTED 2026-07-27 after the first cluster run. The weighted-mean figure was
#: 0.6052, computed with the item-total weights derived ONCE from the full
#: five-rater panel and then a column dropped inside each fold. That leaks the
#: held-out rater into its own target through the weights. Recomputing the
#: weights from the remaining four inside each fold -- which is what
#: ``item_total_weights`` does -- gives **0.6043**, and is correct.
#:
#: The gap is small, which is exactly why it would have survived unnoticed.
LEARNABILITY_237 = {
    "weighted_mean": 0.6043,
    "mean": 0.6022,
    "median": 0.5708,
    "mode": 0.4956,
    "orthodontist": 0.4944,
}

#: The ordering is identical on both populations, so the conclusion -- that the
#: mean is the best available target and the orthodontist the worst -- does not
#: depend on which one you compute it over.
EXPECTED_ORDERING = ("weighted_mean", "mean", "median", "mode", "orthodontist")

LEARNABILITY_BY_POPULATION = {237: LEARNABILITY_237, 251: LEARNABILITY_251}

#: The manifest holds the photographed patients, so a cluster run reproduces the
#: 237 column. Anything comparing a run against §1.7's quoted numbers is
#: comparing populations, not checking a computation.
MANIFEST_POPULATION = 237


def expected_learnability(population: int = MANIFEST_POPULATION) -> dict[str, float]:
    """The measured values for a population. Never mix the two."""
    if population not in LEARNABILITY_BY_POPULATION:
        raise LabelError(
            f"no measured learnability for a population of {population}; known: "
            f"{sorted(LEARNABILITY_BY_POPULATION)}"
        )
    return LEARNABILITY_BY_POPULATION[population]

#: Attenuation ceilings on the 237, sqrt of the target's reliability.
CEILING_MEAN_OF_FIVE = 0.9032
CEILING_SINGLE_RATER = 0.685

LEARNABILITY_CAVEAT = (
    "Aggregate targets are measured against a mean-of-four while the "
    "single-rater target is a single column. A mean of four is inherently less "
    "noisy, so part of the gap is that difference in noise rather than a "
    "difference in the estimator. Lower noise is what makes a target more "
    "learnable, so the ordering stands, but this is not a like-for-like "
    "comparison and the write-up must say so. The attenuation ceiling is the "
    "symmetric companion measure."
)


class LabelError(ValueError):
    """A label could not be constructed."""


# --------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------


def item_total_weights(matrix: np.ndarray) -> np.ndarray:
    """Corrected item-total Pearson correlations, used as rater weights.

    Recomputed from whatever columns are passed, so inside a leave-one-out fold
    the weights come from the remaining four raters only -- never from the full
    panel, which would leak the held-out rater into its own target.
    """
    array = np.asarray(matrix, dtype=float)
    weights = np.array([stat.r for stat in R.item_total(array.astype(np.int64))])
    if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
        raise LabelError(
            f"item-total weights are unusable: {weights}. A weighted mean needs "
            "at least one rater positively correlated with the others."
        )
    return weights


def _mode(array: np.ndarray) -> np.ndarray:
    """Row-wise mode, ties broken toward the LOWER grade.

    Ties are the normal case with four or five raters, so the rule is fixed and
    stated rather than left to whichever library is in use. 1 is the best outcome,
    so breaking downward is the optimistic direction; what matters is that it is
    deterministic and documented.
    """
    grades = np.asarray(array, dtype=np.int64)
    out = np.empty(grades.shape[0], dtype=np.int64)
    for index, row in enumerate(grades):
        values, counts = np.unique(row, return_counts=True)
        out[index] = values[counts == counts.max()].min()
    return out


def build_target(matrix, kind: str) -> np.ndarray:
    """Build an aggregate target from every column of ``matrix``."""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[1] < 1:
        raise LabelError(f"expected a 2-D grade matrix, got shape {array.shape}")

    if kind == "mean":
        return array.mean(axis=1)
    if kind == "median":
        return np.median(array, axis=1)
    if kind == "mode":
        return _mode(array).astype(float)
    if kind == "weighted_mean":
        weights = item_total_weights(array)
        return (array * weights).sum(axis=1) / weights.sum()
    raise LabelError(
        f"unknown aggregate target {kind!r}; expected one of {sorted(AGGREGATE_TARGETS)}"
    )


def soft_labels(matrix) -> np.ndarray:
    """(n, 5) fraction of raters awarding each grade 1-5. Rows sum to 1."""
    array = np.asarray(matrix, dtype=np.int64)
    counts = np.stack(
        [(array == grade).sum(axis=1) for grade in R.CATEGORIES], axis=1
    ).astype(float)
    return counts / array.shape[1]


def class3(values) -> np.ndarray:
    """The 3-class collapse at the fixed a priori thresholds 2.5 / 3.5."""
    return metrics.to_3class(np.asarray(values, dtype=float))


# --------------------------------------------------------------------------
# learnability
# --------------------------------------------------------------------------


def learnability(matrix, kind: str, *, rater_index: int | None = None) -> float:
    """Leave-one-rater-out mean Pearson for one target."""
    array = np.asarray(matrix, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] < 3:
        raise LabelError(f"need at least 3 raters, got shape {array.shape}")

    if kind in AGGREGATE_TARGETS:
        scores = []
        for held in range(array.shape[1]):
            rest = np.delete(array, held, axis=1)
            scores.append(metrics.pcc(array[:, held], build_target(rest, kind)))
        return float(np.mean(scores))

    if kind == "orthodontist":
        if rater_index is None:
            raise LabelError("a single-rater target needs rater_index")
        # Nothing to rebuild: the target IS one column, so every other rater is
        # correlated against it directly.
        target = array[:, rater_index]
        others = np.delete(array, rater_index, axis=1)
        scores = [metrics.pcc(others[:, j], target) for j in range(others.shape[1])]
        return float(np.mean(scores))

    raise LabelError(f"unknown target {kind!r}; expected one of {list(TARGETS)}")


def attenuation_ceiling(matrix, k: int) -> float:
    """sqrt(reliability of a k-rater target). The symmetric companion measure.

    k=5 gives sqrt(0.8158) = 0.9032; k=1 gives sqrt(0.4696) = 0.685.
    """
    return R.pcc_ceiling(R.spearman_brown(R.mean_inter_rater_r(matrix), k))


@dataclass(frozen=True)
class LearnabilityTable:
    scores: dict[str, float]
    ceilings: dict[str, float]
    caveat: str = LEARNABILITY_CAVEAT

    def ordering(self) -> list[str]:
        return sorted(self.scores, key=lambda k: self.scores[k], reverse=True)

    def as_dict(self) -> dict:
        return {
            "learnability": dict(self.scores),
            "attenuation_ceiling": dict(self.ceilings),
            "ordering": self.ordering(),
            "caveat": self.caveat,
        }


def learnability_table(matrix, *, orthodontist_index: int) -> LearnabilityTable:
    """Every target's learnability and its attenuation ceiling."""
    array = np.asarray(matrix, dtype=np.int64)
    scores = {
        kind: learnability(array, kind, rater_index=orthodontist_index)
        for kind in TARGETS
    }
    ceilings = {
        kind: attenuation_ceiling(array, 1 if kind == "orthodontist" else array.shape[1])
        for kind in TARGETS
    }
    return LearnabilityTable(scores=scores, ceilings=ceilings)
