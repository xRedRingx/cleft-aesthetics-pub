"""Rater reliability, and the ceiling that follows from it.

Phase 1 brief §1.4-§1.6.

**Reliability and a correlation ceiling are different quantities**, and confusing
them has already cost this project once. Reliability is var(true)/var(observed).
The maximum correlation any predictor can reach against the *observed* panel mean
is its **square root**. Both are named here, separately, and the relationship
between them is asserted in a test rather than left in a comment:

    reliability_237 = 0.8158        sqrt -> pcc_ceiling_237 = 0.9032
    reliability_251 = 0.8073        sqrt -> pcc_ceiling_251 = 0.8985

The 237 figures are the ones that matter -- they are the patients with
photographs. The 251-row figures exist only so that a number computed on the
other cohort is recognisable rather than mistaken for an error.

Everything is computed from the raw grade matrix. The workbook's own `Average`
and `Median` columns are never read.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..eval import metrics

#: Grades run 1-5. Passed explicitly everywhere a confusion matrix is built: if
#: the categories were inferred from the data instead, a rater pair that never
#: used grade 5 would silently get a 4x4 matrix and a different kappa.
CATEGORIES: tuple[int, ...] = (1, 2, 3, 4, 5)

N_RATERS = 5

# --------------------------------------------------------------------------
# measured values (brief §1.4) -- targets, not inputs
# --------------------------------------------------------------------------

MEAN_R_237 = 0.4696
RELIABILITY_237 = 0.8158
# [2026-08-23] Provenance split, from the primaries
# (phase12.PRIMARY_SOURCES_BANKED): Spearman-Brown is the LINEAGE'S own
# machinery (1991 names the formula; Part 4 sizes its panel on the
# projections), but the sqrt(reliability) CEILING below is standard
# psychometrics applied on top -- neither paper does attenuation-ceiling
# reasoning. And the trap: their projected PANEL RELIABILITY 0.90/0.902
# is NOT this correlation ceiling 0.9032 -- different quantities,
# coincidentally adjacent, never placed in proximity unqualified.
PCC_CEILING_237 = 0.9032

MEAN_R_251 = 0.4560
RELIABILITY_251 = 0.8073
PCC_CEILING_251 = 0.8985

#: Agreement indices, split by population for the same reason the reliabilities
#: and the learnabilities are. The brief quotes the 251-row values from
#: summary.json; the manifest holds 237, and a cluster run reproduces THOSE.
#: Comparing a run against the 251 figures looks like a module defect and is a
#: population mismatch -- the third time this exact trap has been walked into.
#: **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- it scores 1-vs-2 as
#: identically wrong to 1-vs-5, which is the wrong loss for an ordinal
#: scale. The distance-aware figures on the same matrix are QWK 0.4276
#: and mean inter-rater r 0.4696. Nothing below is edited:
#: ``record_audit.THE_KAPPA_LIMITATION``.
FLEISS_251 = 0.1605
FLEISS_237 = 0.1662

QWK_251 = 0.4173
QWK_237 = 0.4276

#: The manifest population. Exit criterion 8 compares against the 237 column.
MANIFEST_POPULATION = 237

FLEISS_BY_POPULATION = {237: FLEISS_237, 251: FLEISS_251}
QWK_BY_POPULATION = {237: QWK_237, 251: QWK_251}


class ReliabilityError(ValueError):
    """The grade matrix is not usable for a reliability computation."""


def _check(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ReliabilityError(f"expected a 2-D grade matrix, got shape {array.shape}")
    if array.shape[0] < 2:
        raise ReliabilityError(f"need at least 2 subjects, got {array.shape[0]}")
    if array.shape[1] < 2:
        raise ReliabilityError(f"need at least 2 raters, got {array.shape[1]}")
    if not np.all(np.isin(array, CATEGORIES)):
        outside = sorted(set(np.unique(array)) - set(CATEGORIES))
        raise ReliabilityError(f"grades outside {CATEGORIES}: {outside}")
    return array.astype(np.int64)


# --------------------------------------------------------------------------
# correlation-based
# --------------------------------------------------------------------------


def pairwise_r(matrix) -> list[float]:
    """Pearson r for every rater pair. k=5 gives 10 pairs."""
    array = _check(matrix)
    return [
        metrics.pcc(array[:, i], array[:, j])
        for i, j in itertools.combinations(range(array.shape[1]), 2)
    ]


def mean_inter_rater_r(matrix) -> float:
    values = [r for r in pairwise_r(matrix) if not math.isnan(r)]
    if not values:
        raise ReliabilityError("every rater pair was degenerate; no correlation exists")
    return float(np.mean(values))


def spearman_brown(mean_r: float, k: int = N_RATERS) -> float:
    """Reliability of a k-rater mean, from the mean inter-rater correlation.

        rho = k*r / (1 + (k-1)*r)

    Verified against the measured values: 0.4696 -> 0.8158, 0.4560 -> 0.8073.

    **[PRECISION NOTE 2026-09-05] Neither mapping reproduces from the ROUNDED
    r, and the line above is left as written because the banked figures are
    right.** Recomputing this formula on the 4-dp values gives 0.8157 (not
    0.8158) and 0.8074 (not 0.8073) -- one unit in the last place each, in
    OPPOSITE directions, which is the signature of rounding rather than of an
    arithmetic error. ``RELIABILITY_237`` was computed from the unrounded mean
    r; back-solving 0.8158 gives r = 0.46971, a hair above the banked 0.4696.

    The note exists because **"Verified" is the strongest word this record
    has**, and it was asserting an equality that does not hold at the precision
    it is written to. A reader who checks it on the printed numbers finds it
    fails and has no way to know why. Anyone reproducing these must start from
    the unrounded correlations, not from the 4-dp constants.

    ``k=1`` is meaningful and returns r itself: the reliability of a single rater
    is its average correlation with a parallel rater. That is what gives the
    single-rater attenuation ceiling sqrt(0.4696) = 0.685.
    """
    if k < 1:
        raise ReliabilityError(f"k must be at least 1, got {k}")
    denominator = 1.0 + (k - 1) * mean_r
    if denominator == 0:
        raise ReliabilityError("Spearman-Brown is undefined for this mean correlation")
    return float(k * mean_r / denominator)


#: **[ADDED 2026-09-02] The two-way ANOVA behind the ICCs.**
#:
#: ``spearman_brown`` averages PAIRWISE PEARSON correlations, and Pearson
#: is invariant to an additive offset -- **a rater who is consistently
#: harsh but perfectly ordered contributes fully to r and nothing to
#: absolute agreement**. The ICCs decompose the variance instead, so that
#: offset lands in a term rather than vanishing
#: (``record_audit.ICC_PREDICTION_REGISTERED``).
#:
#: **Both forms are returned and NEITHER is chosen here.** (2,k) treats
#: the raters as a random sample of possible raters and penalises their
#: between-rater variance; (3,k) treats these five as the population of
#: interest and removes it. **ICC(3,k) >= ICC(2,k) exactly when MSC >=
#: MSE** -- not always; it reverses by sampling noise on a panel with no
#: offset (corrected 2026-09-03). Which one a
#: write-up wants depends on what it claims
#: (``record_audit.ICC_FORM_DISTINCTION``).
def icc_two_way(matrix) -> dict:
    """Shrout-Fleiss ICCs from a fully-crossed subjects x raters matrix.

    Every rater rates every subject -- which ``_check`` already requires,
    and which the 237 x 5 panel satisfies. Returns the three mean squares
    and the four ICCs, so a reader can see where each figure comes from
    rather than taking a single number on trust.
    """
    return _icc_two_way_anova(_check(matrix).astype(float))


def _icc_two_way_anova(array) -> dict:
    """The arithmetic, WITHOUT the 1-5 category guard.

    Split out so the ANOVA can be validated against the published
    Shrout-Fleiss worked example, whose grades run to 10 and which
    ``_check`` therefore refuses. **The guard is not weakened** -- the
    public function still applies it; this only separates "is this our
    panel" from "is this arithmetic right".
    """
    array = np.asarray(array, dtype=float)
    n, k = array.shape
    if n < 2 or k < 2:
        raise ReliabilityError(
            f"a two-way ICC needs at least 2 subjects and 2 raters; "
            f"got {n} x {k}"
        )

    grand = array.mean()
    row_means = array.mean(axis=1)
    col_means = array.mean(axis=0)

    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((array - grand) ** 2))
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    if ms_rows == 0:
        raise ReliabilityError(
            "between-subject variance is zero; every ICC is undefined"
        )

    icc_2_1_denominator = (
        ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    )
    icc_2_k_denominator = ms_rows + (ms_cols - ms_error) / n
    return {
        "n_subjects": int(n),
        "n_raters": int(k),
        "ms_rows": ms_rows,
        "ms_cols": ms_cols,
        "ms_error": ms_error,
        #: Single-rater forms, for completeness beside the k-rater ones.
        "icc_2_1": float((ms_rows - ms_error) / icc_2_1_denominator),
        "icc_3_1": float(
            (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error)
        ),
        #: **The two the panel's reliability could be quoted as.**
        "icc_2_k": float((ms_rows - ms_error) / icc_2_k_denominator),
        "icc_3_k": float((ms_rows - ms_error) / ms_rows),
        #: The rater term itself -- what Pearson r cannot see. Zero when
        #: no rater carries a systematic offset.
        "between_rater_ms": ms_cols,
    }


def pcc_ceiling(reliability: float) -> float:
    """The maximum correlation a predictor can reach against the observed mean.

    **This is the square root of reliability, not reliability.** Read §1.5 of the
    brief before changing anything here.
    """
    if reliability < 0:
        raise ReliabilityError(f"reliability {reliability} is negative; no real ceiling")
    return float(math.sqrt(reliability))


def cronbach_alpha(matrix) -> float:
    """Raw alpha, from item and total variances.

    Distinct from the standardised form, which is Spearman-Brown applied to the
    mean inter-item correlation. The brief lists both because they differ.
    """
    array = _check(matrix).astype(float)
    k = array.shape[1]
    item_variance = array.var(axis=0, ddof=1).sum()
    total_variance = array.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        raise ReliabilityError("total score has zero variance; alpha is undefined")
    return float((k / (k - 1)) * (1.0 - item_variance / total_variance))


# --------------------------------------------------------------------------
# agreement-based
# --------------------------------------------------------------------------


# **[2026-09-02] Fleiss kappa is UNWEIGHTED** -- 1-vs-2 scores as
# identically wrong to 1-vs-5. Distance-aware beside it: QWK 0.4276,
# mean inter-rater r 0.4696. record_audit.THE_KAPPA_LIMITATION.
def fleiss_kappa(matrix, categories: tuple[int, ...] = CATEGORIES) -> float:
    """Fleiss' kappa for m raters over n subjects.

    Measured value on this cohort: 0.1605. That is **the research problem, not a
    data defect** -- raters genuinely disagree about aesthetic outcome.
    """
    array = _check(matrix)
    n_subjects, n_raters = array.shape

    counts = np.zeros((n_subjects, len(categories)), dtype=np.int64)
    for index, category in enumerate(categories):
        counts[:, index] = (array == category).sum(axis=1)

    agreement = (np.square(counts).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(agreement.mean())
    proportions = counts.sum(axis=0) / (n_subjects * n_raters)
    p_expected = float(np.square(proportions).sum())

    if p_expected == 1.0:
        raise ReliabilityError(
            "every rating is in one category; expected agreement is 1 and kappa "
            "is undefined"
        )
    return (p_bar - p_expected) / (1.0 - p_expected)


def _qwk(a: np.ndarray, b: np.ndarray, categories: tuple[int, ...]) -> float:
    n = len(categories)
    index = {c: i for i, c in enumerate(categories)}
    observed = np.zeros((n, n), dtype=float)
    for x, y in zip(a, b):
        observed[index[x], index[y]] += 1.0
    observed /= observed.sum()

    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)

    grid = np.arange(n, dtype=float)
    weights = (grid[:, None] - grid[None, :]) ** 2 / (n - 1) ** 2

    denominator = float((weights * expected).sum())
    if denominator == 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


def pairwise_qwk(matrix, categories: tuple[int, ...] = CATEGORIES) -> list[float]:
    array = _check(matrix)
    return [
        _qwk(array[:, i], array[:, j], categories)
        for i, j in itertools.combinations(range(array.shape[1]), 2)
    ]


def mean_pairwise_qwk(matrix, categories: tuple[int, ...] = CATEGORIES) -> float:
    """Measured value: 0.4173. **Never a ceiling** -- it is an agreement index."""
    values = [v for v in pairwise_qwk(matrix, categories) if not math.isnan(v)]
    if not values:
        raise ReliabilityError("no rater pair produced a defined kappa")
    return float(np.mean(values))


# --------------------------------------------------------------------------
# per-rater
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RaterStats:
    index: int
    r: float
    rho: float
    mean_qwk: float


def item_total(matrix, categories: tuple[int, ...] = CATEGORIES) -> list[RaterStats]:
    """Each rater against the mean of the other four (corrected item-total).

    ``r`` is Pearson, ``rho`` is Spearman. Measured (§1.6): the speech and
    language therapist is highest on both at 0.654, against the orthodontist's
    0.628 -- which contradicts a premise raised at supervision.
    """
    array = _check(matrix)
    n_raters = array.shape[1]

    stats = []
    for i in range(n_raters):
        others = np.delete(array, i, axis=1)
        rest = others.mean(axis=1)
        pairwise = [
            _qwk(array[:, i], others[:, j], categories) for j in range(others.shape[1])
        ]
        defined = [v for v in pairwise if not math.isnan(v)]
        stats.append(
            RaterStats(
                index=i,
                r=metrics.pcc(array[:, i], rest),
                rho=metrics.spearman(array[:, i], rest),
                mean_qwk=float(np.mean(defined)) if defined else float("nan"),
            )
        )
    return stats


# --------------------------------------------------------------------------
# the bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    n_subjects: int
    n_raters: int
    mean_inter_rater_r: float
    reliability: float
    pcc_ceiling: float
    cronbach_alpha: float
    fleiss_kappa: float
    mean_pairwise_qwk: float
    per_rater: tuple[RaterStats, ...]

    def as_dict(self) -> dict:
        """Aggregate scalars only -- SHAREABLE. No patient is identifiable here."""
        return {
            "n_subjects": self.n_subjects,
            "n_raters": self.n_raters,
            "mean_inter_rater_r": self.mean_inter_rater_r,
            "reliability_spearman_brown": self.reliability,
            "pcc_ceiling": self.pcc_ceiling,
            "cronbach_alpha": self.cronbach_alpha,
            "fleiss_kappa": self.fleiss_kappa,
            "mean_pairwise_qwk": self.mean_pairwise_qwk,
            "per_rater": [
                {"index": s.index, "r": s.r, "rho": s.rho, "mean_qwk": s.mean_qwk}
                for s in self.per_rater
            ],
        }


def summarise(matrix, categories: tuple[int, ...] = CATEGORIES) -> Summary:
    array = _check(matrix)
    mean_r = mean_inter_rater_r(array)
    reliability = spearman_brown(mean_r, array.shape[1])
    return Summary(
        n_subjects=array.shape[0],
        n_raters=array.shape[1],
        mean_inter_rater_r=mean_r,
        reliability=reliability,
        pcc_ceiling=pcc_ceiling(reliability),
        cronbach_alpha=cronbach_alpha(array),
        fleiss_kappa=fleiss_kappa(array, categories),
        mean_pairwise_qwk=mean_pairwise_qwk(array, categories),
        per_rater=tuple(item_total(array, categories)),
    )
