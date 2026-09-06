"""Benavoli's Bayesian correlated t-test: the arithmetic (Phase 23).

The **single-dataset** test, not the hierarchical model
(``phase23.THE_HIERARCHICAL_MODEL_IS_EXCLUDED``).

**Every parameter is either the source's or ruled, and none is chosen
here.** The functions below take them as arguments with no defaults, so
a caller cannot silently inherit a value this project never ruled:

* **rho** -- the source's, and not a choice: the paper records that rho
  *"does not allow to estimate rho from data, since the maximum
  likelihood estimate of rho is rho-hat = 0 regardless the
  observations"*, so the heuristic ``rho = n_te/n_tot`` is adopted from
  necessity. **Its approximation travels with it**: it was derived for
  random resamples and ours are non-overlapping k-fold folds, which the
  paper's own footnote 2 concedes.
* **the posterior** -- the source's, equation 6:
  ``St(mu; n-1, xbar, (1/n + rho/(1-rho)) sigmahat^2)`` under the
  matching prior ``{mu_0 = 0, k_0 -> inf, a = -1/2, b = 0}``.
* **P(rope)** -- the source's, section 3.2: *"the integral of the
  posterior over the rope interval"*.
* **the half-width and the threshold** -- **OURS**
  (``phase23.THE_ROPE_RULED``, ``THE_DECISION_THRESHOLD_RULED``).

**This module computes nothing that is banked.** It reads per-fold PCCs
and produces posteriors; no delta, threshold or verdict already in the
ledger is recomputed here.
"""

from __future__ import annotations

import math

import numpy as np

#: The three verdicts, exhaustive and mutually exclusive. **"no
#: decision" is a verdict in its own right and is never equivalence**
#: (``phase23.EXIT_CRITERIA`` criterion 6).
VERDICTS = (
    "practically equivalent",
    "practically different",
    "no decision",
)


def per_fold_detail(rows) -> dict:
    """Per fold: the PCC **and WHY it is undefined when it is**.

    **[ADDED 2026-09-02.]** ``per_fold_pcc`` returns the number alone,
    which is everything the arithmetic needs and less than the record
    needs. A NaN there says THAT a fold is degenerate and never WHICH
    quantity was constant -- so a dropped fold reached metrics.json as
    a reduced ``n`` with no reason beside it. The cause is available
    exactly here, where the fold's two columns are in hand, and
    nowhere downstream.

    Each fold maps to ``pcc``, ``n_patients``, ``truth_sd``,
    ``prediction_sd`` and ``undefined_because`` (``None`` when the PCC
    is defined). **The degeneracy rule is this function's alone**;
    ``per_fold_pcc`` reads its answer rather than repeating the test.
    """
    by_fold: dict[int, list[tuple[float, float]]] = {}
    for row in rows:
        fold = int(row["fold"])
        by_fold.setdefault(fold, []).append(
            (float(row["truth"]), float(row["prediction"]))
        )
    out = {}
    for fold, pairs in by_fold.items():
        truth = np.array([t for t, _ in pairs], dtype=float)
        prediction = np.array([p for _, p in pairs], dtype=float)
        truth_sd = float(truth.std())
        prediction_sd = float(prediction.std())
        if len(truth) < 2:
            reason = "fewer than two patients in the fold"
        elif truth_sd == 0.0 and prediction_sd == 0.0:
            reason = "constant truth AND constant prediction within the fold"
        elif truth_sd == 0.0:
            reason = "constant truth within the fold"
        elif prediction_sd == 0.0:
            reason = "constant prediction within the fold"
        else:
            reason = None
        out[fold] = {
            "pcc": (
                float("nan") if reason is not None
                else float(np.corrcoef(truth, prediction)[0, 1])
            ),
            "n_patients": int(len(truth)),
            "truth_sd": truth_sd,
            "prediction_sd": prediction_sd,
            "undefined_because": reason,
        }
    return out


def per_fold_pcc(rows) -> dict:
    """PCC within each fold, from one arm's prediction CSV rows.

    **A fold whose truth or prediction is constant has an undefined
    PCC and is returned as NaN**, not as zero. ``phase23`` named
    degenerate folds as a live risk in advance, so they surface rather
    than being absorbed into a correlation of zero -- which would be a
    number where there is no number.

    **[2026-09-02] The number only.** The cause travels in
    ``per_fold_detail``, which this now reads so that the two cannot
    disagree about which folds are degenerate.
    """
    return {
        fold: detail["pcc"] for fold, detail in per_fold_detail(rows).items()
    }


def paired_fold_differences(arm_a: dict, arm_b: dict):
    """``x_i`` -- the vector of per-fold PCC differences, a minus b.

    ``arm_a``/``arm_b`` map ``seed -> {fold: pcc}``. **Only seeds and
    folds present in BOTH arms are paired**, which is how a ten-seed
    arm against a five-seed one reduces to the shared five
    (``phase23.PAIRING_ESTABLISHED_BY_CONSTRUCTION``). A NaN on either
    side drops that fold and is counted.

    Returns ``(x, n, dropped)`` -- **the count only**; the reporting
    form is ``paired_fold_report``. Deterministic order: seeds then
    folds, both sorted.
    """
    values, dropped = [], 0
    for _seed, _fold, a, b, status in _paired_cells(arm_a, arm_b):
        if status == "paired":
            values.append(a - b)
        elif status == "degenerate":
            dropped += 1
    return np.array(values, dtype=float), len(values), dropped


def _paired_cells(arm_a: dict, arm_b: dict):
    """Every ``(seed, fold)`` cell of the two arms, with its status.

    **[ADDED 2026-09-02] ONE pairing rule, serving both the counting
    form and the reporting form**, so a fix to one cannot leave the
    other behind. Yields ``(seed, fold, a, b, status)`` with status in
    ``paired`` / ``degenerate`` / ``unpaired``.

    It walks the UNION rather than the intersection, because the
    intersection makes an unpaired cell invisible: a fold present in
    one arm and absent from the other left no trace at all, and was
    indistinguishable in the record from a degenerate one. Both
    shorten ``n``; only one of them is a degeneracy.
    """
    for seed in sorted(set(arm_a) | set(arm_b)):
        folds_a = arm_a.get(seed, {})
        folds_b = arm_b.get(seed, {})
        for fold in sorted(set(folds_a) | set(folds_b)):
            if fold not in folds_a or fold not in folds_b:
                yield seed, fold, None, None, "unpaired"
                continue
            a, b = folds_a[fold], folds_b[fold]
            if math.isnan(a) or math.isnan(b):
                yield seed, fold, a, b, "degenerate"
                continue
            yield seed, fold, a, b, "paired"


def paired_fold_report(
    name_a: str, arm_a: dict, detail_a: dict,
    name_b: str, arm_b: dict, detail_b: dict,
) -> dict:
    """``x`` **and the account of every cell that is not in it**.

    **[ADDED 2026-09-02.]** ``paired_fold_differences`` returns a bare
    count, and a count is not an account: three contrasts ran at
    ``n=24`` and nothing in the record said which arm, which seed,
    which fold, or why. This returns ``x``, ``n``, ``dropped`` and
    ``unpaired``, the last two as records rather than integers.

    **A degenerate cell whose cause is not in ``detail`` RAISES**
    rather than being reported with a null reason -- writing the null
    is the defect being fixed, so it must not be reachable from here.
    """
    values, dropped, unpaired = [], [], []
    for seed, fold, a, b, status in _paired_cells(arm_a, arm_b):
        if status == "paired":
            values.append(a - b)
            continue
        if status == "unpaired":
            present = name_a if fold in arm_a.get(seed, {}) else name_b
            absent = name_b if present == name_a else name_a
            unpaired.append({
                "seed": int(seed), "fold": int(fold),
                "present_in": present, "absent_from": absent,
            })
            continue
        causes = []
        for name, value, detail in (
            (name_a, a, detail_a), (name_b, b, detail_b)
        ):
            if not math.isnan(value):
                continue
            cell = (detail.get(seed) or {}).get(fold)
            if cell is None:
                raise ValueError(
                    f"arm {name!r} seed {seed} fold {fold} has an "
                    "undefined PCC and no per-fold detail to say why. "
                    "Refusing to report a dropped fold with a null "
                    "reason: a reduced n with no reason beside it is "
                    "the defect this function exists to prevent."
                )
            causes.append({
                "arm": name,
                "reason": cell["undefined_because"],
                "n_patients": cell["n_patients"],
                "truth_sd": cell["truth_sd"],
                "prediction_sd": cell["prediction_sd"],
            })
        dropped.append({
            "seed": int(seed), "fold": int(fold),
            "arms": [cause["arm"] for cause in causes],
            "causes": causes,
        })
    return {
        "x": np.array(values, dtype=float),
        "n": len(values),
        "dropped": dropped,
        "unpaired": unpaired,
    }


def posterior(x: np.ndarray, rho: float):
    """``(df, loc, scale)`` of the source's equation-6 Student posterior.

    ``St(mu; n-1, xbar, (1/n + rho/(1-rho)) sigmahat^2)``. The variance
    is the sample variance with ``ddof=1``, which is the ``sigmahat^2``
    the paper defines beside the formula.
    """
    values = np.asarray(x, dtype=float)
    n = len(values)
    if n < 2:
        raise ValueError(
            f"the posterior needs at least two observations; got {n}. A "
            "contrast this thin has no degrees of freedom and must be "
            "reported as unavailable rather than given a posterior."
        )
    variance = float(np.var(values, ddof=1))
    scale = math.sqrt((1.0 / n + rho / (1.0 - rho)) * variance)
    return n - 1, float(np.mean(values)), scale


def probabilities(x: np.ndarray, rho: float, half_width: float) -> dict:
    """P(left), P(rope), P(right) -- the integral of the posterior.

    The rope is ``[-half_width, +half_width]``; left and right are the
    two tails outside it. The three sum to one by construction.
    """
    from scipy.stats import t as student

    df, loc, scale = posterior(x, rho)
    if scale == 0.0:
        # Every difference identical: the posterior is a point mass.
        inside = abs(loc) <= half_width
        return {
            "left": 0.0 if inside or loc > 0 else 1.0,
            "rope": 1.0 if inside else 0.0,
            "right": 0.0 if inside or loc < 0 else 1.0,
        }
    left = float(student.cdf(-half_width, df, loc, scale))
    right = float(student.sf(half_width, df, loc, scale))
    return {"left": left, "rope": 1.0 - left - right, "right": right}


def verdict(probs: dict, threshold: float) -> str:
    """One of ``VERDICTS``, at ``P(.) > threshold``.

    **The inequality is STRICT**, as the paper writes it, so a
    probability sitting exactly at the threshold does not decide. And
    **"no decision" is returned as itself** -- there is no path in this
    function by which an undecided contrast is called equivalent
    (``phase23.EXIT_CRITERIA`` criterion 6).
    """
    if probs["rope"] > threshold:
        return "practically equivalent"
    if probs["left"] > threshold or probs["right"] > threshold:
        return "practically different"
    return "no decision"


def describe(x: np.ndarray, dropped: list, unpaired: list) -> dict:
    """What the observations look like. **Gates nothing.**

    The paper assumes multivariate normality and never checks it; this
    reports the shape so the record can say what ours looked like.
    **No threshold, no test, no branch** -- branching on this after the
    fact would be choosing a method after seeing data
    (``phase23.NORMALITY_DESCRIBED_NOT_TESTED``).

    **[CHANGED 2026-09-02: ``dropped_folds: int`` -> ``dropped: list``,
    ``unpaired: list``.]** The old signature could only carry HOW MANY
    cells were missing, so the record said ``n=24`` and stopped. Both
    arguments are REQUIRED and neither has a default: the caller that
    wrote nulls did so by having nothing to pass, and a default would
    have let the next one do it silently again
    (``phase23.THE_DROP_WAS_COUNTED_NOT_RECORDED``).

    **The emitted keys are ``dropped_folds`` and ``unpaired_folds``,
    with counts in ``n_dropped`` and ``n_unpaired``.** The arguments
    are named ``dropped`` and ``unpaired``; **neither is a key in the
    record**, and a reader probing for one finds a present list absent.
    Named here because that mistake has already been made once.
    """
    values = np.asarray(x, dtype=float)
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else float("nan"),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "min": float(np.min(values)) if len(values) else float("nan"),
        "max": float(np.max(values)) if len(values) else float("nan"),
        # The count kept under its shipped name so the two runs'
        # metrics.json compare directly; equal to n_dropped by
        # construction, and asserted so.
        "degenerate_folds_dropped": int(len(dropped)),
        "n_dropped": int(len(dropped)),
        "dropped_folds": list(dropped),
        "n_unpaired": int(len(unpaired)),
        "unpaired_folds": list(unpaired),
    }


def rope_half_width(sd: float, n_seeds: int) -> float:
    """The ruled width, RECOMPUTED from its inputs.

    ``combined_claimable_delta``'s ``arm_means_95`` at the probe's own
    seed sd -- **not read from a config constant**. Criterion 2 makes
    the config's declared value a CHECK against this, so a drifted
    constant refuses the run rather than being used.
    """
    from .train.phase3 import combined_claimable_delta

    return float(
        combined_claimable_delta(sd, n_seeds, sd, n_seeds)["arm_means_95"]
    )
