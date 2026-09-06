"""Metrics for the aesthetic scale.

PCC is the primary metric, not QWK (Part 4.3). Everything here is implemented in
numpy rather than delegating to scipy or sklearn, so that the tests comparing
against those libraries are genuine independent checks rather than a tautology.

The one exception is the normal quantile function used by the BCa interval,
taken from ``scipy.special``: there the test is reproducibility and bracketing,
not agreement with scipy.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import ndtr, ndtri

#: The 3-class collapse for reporting, fixed a priori (Part 4.2). Grades
#: {1,2} -> 0, {3} -> 1, {4,5} -> 2. Never tuned, never per-table.
CLASS_THRESHOLDS: tuple[float, float] = (2.5, 3.5)

#: The raw label scale: Excellent=1 ... Very poor=5.
SCALE_MIN, SCALE_MAX = 1.0, 5.0

#: Smallest sample the BCa jackknife acceleration is meaningful on.
MIN_BCA_N = 10


def _as_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains nan or inf. A silent nan metric has previously looked "
            "like a model that had not learned."
        )
    return array


def _pair(a, b) -> tuple[np.ndarray, np.ndarray]:
    x = _as_array(a, "first argument")
    y = _as_array(b, "second argument")
    if x.size != y.size:
        raise ValueError(f"length mismatch: {x.size} vs {y.size}")
    if x.size < 2:
        raise ValueError(f"need at least 2 observations, got n={x.size}")
    return x, y


# --------------------------------------------------------------------------
# correlation
# --------------------------------------------------------------------------


def pcc(truth, pred) -> float:
    """Pearson correlation. Returns nan when either side is constant."""
    x, y = _pair(truth, pred)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc**2).sum()) * np.sqrt((yc**2).sum())
    if denom == 0.0:
        # Undefined against a constant. Saying so beats returning 0, which reads
        # as "measured, and no relationship".
        return float("nan")
    return float((xc * yc).sum() / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, matching scipy's default method."""
    order = np.argsort(values, kind="mergesort")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(values.size)
    sorted_values = values[order]
    is_new = np.r_[True, sorted_values[1:] != sorted_values[:-1]]
    dense = is_new.cumsum()[inverse]
    boundaries = np.r_[np.nonzero(is_new)[0], values.size]
    return 0.5 * (boundaries[dense] + boundaries[dense - 1] + 1)


def spearman(truth, pred) -> float:
    """Spearman rank correlation, ties averaged. Grades tie constantly."""
    x, y = _pair(truth, pred)
    return pcc(_rankdata(x), _rankdata(y))


# --------------------------------------------------------------------------
# error
# --------------------------------------------------------------------------


def mae(truth, pred) -> float:
    x, y = _pair(truth, pred)
    return float(np.abs(x - y).mean())


def rmse(truth, pred) -> float:
    x, y = _pair(truth, pred)
    return float(np.sqrt(((x - y) ** 2).mean()))


# --------------------------------------------------------------------------
# the 3-class collapse
# --------------------------------------------------------------------------


def to_3class(values, clip: bool = False) -> np.ndarray:
    """Collapse the 1-5 scale to 3 reporting classes at the fixed thresholds.

    A value exactly on a threshold goes to the higher class. Out-of-range input
    raises unless ``clip`` is set: a model whose head is unclipped, or labels on
    a 0-based or 0-1 scale, must surface as an error rather than as quietly
    saturated classes. Pass ``clip=True`` only where clipping is the intended
    modelling decision.
    """
    array = _as_array(values, "values")
    if clip:
        array = np.clip(array, SCALE_MIN, SCALE_MAX)
    elif array.min() < SCALE_MIN or array.max() > SCALE_MAX:
        raise ValueError(
            f"values outside the {SCALE_MIN}-{SCALE_MAX} label range: "
            f"[{array.min():.4g}, {array.max():.4g}]"
        )
    return np.digitize(array, CLASS_THRESHOLDS).astype(int)


def qwk_3cat(truth, pred, clip_pred: bool = True) -> float:
    """Quadratic weighted kappa on the 3-class collapse.

    Predictions are clipped to the label range by default, since a regression
    head may legitimately overshoot; the truth is never clipped.
    """
    x, y = _pair(truth, pred)
    a = to_3class(x, clip=False)
    b = to_3class(y, clip=clip_pred)
    n_classes = 3

    observed = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(observed, (a, b), 1.0)
    observed /= observed.sum()

    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)

    index = np.arange(n_classes, dtype=float)
    weights = (index[:, None] - index[None, :]) ** 2 / (n_classes - 1) ** 2

    denominator = float((weights * expected).sum())
    if denominator == 0.0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


# --------------------------------------------------------------------------
# BCa bootstrap
# --------------------------------------------------------------------------


def bca_ci(
    statistic: Callable[..., float],
    *samples,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI for ``statistic``.

    All ``samples`` are resampled with one shared index vector, so pairing over
    patients is preserved. Part 4.3: a delta is claimed only if its paired BCa CI
    over patients excludes 0; comparing two arms' separate CIs is invalid because
    they share the same patients.
    """
    arrays = [_as_array(s, f"sample {i}") for i, s in enumerate(samples)]
    n = arrays[0].size
    if any(a.size != n for a in arrays):
        raise ValueError("length mismatch between samples")
    if n < MIN_BCA_N:
        raise ValueError(
            f"sample too small for a BCa interval: n={n}, need at least {MIN_BCA_N}"
        )

    theta_hat = float(statistic(*arrays))

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = statistic(*(a[idx] for a in arrays))

    finite = boot[np.isfinite(boot)]
    if finite.size < 0.9 * n_boot:
        raise ValueError(
            f"only {finite.size}/{n_boot} bootstrap replicates were finite; the "
            "statistic is undefined on this sample"
        )

    # Degenerate but legitimate: comparing something with itself.
    if np.allclose(finite, theta_hat):
        return theta_hat, theta_hat

    # Bias correction.
    proportion = float((finite < theta_hat).mean())
    proportion = min(max(proportion, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = float(ndtri(proportion))

    # Acceleration, from the jackknife.
    jack = np.array(
        [
            statistic(*(np.delete(a, i) for a in arrays))
            for i in range(n)
        ],
        dtype=float,
    )
    jack = jack[np.isfinite(jack)]
    deviation = jack.mean() - jack
    denominator = 6.0 * (deviation**2).sum() ** 1.5
    acceleration = float((deviation**3).sum() / denominator) if denominator != 0 else 0.0

    def adjusted(z_alpha: float) -> float:
        numerator = z0 + z_alpha
        return float(ndtr(z0 + numerator / (1.0 - acceleration * numerator)))

    lo_q = adjusted(float(ndtri(alpha / 2.0)))
    hi_q = adjusted(float(ndtri(1.0 - alpha / 2.0)))
    lo, hi = np.percentile(finite, [100.0 * lo_q, 100.0 * hi_q])
    return float(lo), float(hi)


def paired_delta_bca(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    truth,
    pred_a,
    pred_b,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 1337,
) -> tuple[float, float, float]:
    """Paired BCa interval for ``statistic(truth, a) - statistic(truth, b)``.

    Returns ``(delta, lo, hi)``. The delta is claimed only if the interval
    excludes 0. For CV-based comparisons the Nadeau-Bengio correction is applied
    on top of this; that belongs with the fold machinery, not here.
    """

    def delta(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float(statistic(t, a)) - float(statistic(t, b))

    point = delta(
        _as_array(truth, "truth"), _as_array(pred_a, "pred_a"), _as_array(pred_b, "pred_b")
    )
    lo, hi = bca_ci(
        delta, truth, pred_a, pred_b, n_boot=n_boot, alpha=alpha, seed=seed
    )
    return point, lo, hi
