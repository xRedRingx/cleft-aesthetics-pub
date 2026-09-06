"""The cleft cohort's aspect-ratio distribution, and how to sample from it.

**[MEASURED] From ``data/staged/staged_v1/geometry.csv``, n=237.**

Recorded here rather than in ``geometry/staging.py`` because that module is
**frozen** -- it produced ``staged_v1``, so adding to it would invalidate every
result measured with it. This module is additive, and
``test_the_distribution_agrees_with_the_frozen_record`` asserts its summary
statistics against ``staging.CLEFT_STAGED_GEOMETRY`` so the two cannot drift.

**Why percentiles and not just min/median/max.** The frozen record carries three
order statistics, and sampling uniformly over that range is wrong by nearly one
standard deviation:

* true cohort mean **0.7475**, sd 0.0887
* uniform over [0.5531, 1.0986] gives mean **0.8259**

The IQR is only 0.691-0.792 wide and the range is dominated by two thin tails --
**236 of 237 crops are portrait**, with exactly one landscape case at 1.0986. A
uniform draw puts far too much mass in a region the cohort barely occupies, so a
masked SCUT set built that way would present the pretrained model with a framing
distribution the clinical data does not have.

**A caution when reading ``geometry.csv``:** line 1 is a ``# CLUSTER-ONLY``
comment and the header is line 2. ``phase3.load_geometry_rows`` already accounts
for this; a naive ``csv.DictReader`` would silently take the comment as the
header, which Phase 1 hit once already.
"""

from __future__ import annotations

import numpy as np

#: Percentiles 0, 5, 10, ... 100 of the 237 cleft crops' aspect ratio (w/h).
#: [MEASURED] from the p2-stage-1 artifact.
AR_PERCENTILES = (
    0.5531, 0.6093, 0.6441, 0.6585, 0.6785, 0.6907, 0.7010, 0.7079, 0.7172,
    0.7305, 0.7404, 0.7469, 0.7549, 0.7686, 0.7775, 0.7922, 0.8151, 0.8360,
    0.8794, 0.9173, 1.0986,
)

#: The quantile positions ``AR_PERCENTILES`` are measured at, as fractions.
AR_QUANTILES = tuple(round(q / 100.0, 2) for q in range(0, 101, 5))

#: [MEASURED] Summary statistics, for cross-checking and for reporting parity.
CLEFT_AR = {
    "source": "data/staged/staged_v1/geometry.csv (p2-stage-1)",
    "n": 237,
    "mean": 0.7475,
    "median": 0.7404,
    "sd": 0.0887,
    "min": 0.5531,
    "max": 1.0986,
    "iqr": (0.6907, 0.7922),
    "n_portrait": 236,
    "n_landscape": 1,
    "uniform_range_mean": 0.8259,
    "note": (
        "Sample by piecewise-linear inverse CDF over AR_PERCENTILES, NOT "
        "uniformly over [min, max]. The range is dominated by two thin tails -- "
        "236 of 237 crops are portrait and exactly one is landscape at 1.0986 -- "
        "so a uniform draw gives mean 0.8259 against the true 0.7475, nearly one "
        "SD high, and would hand the pretrained model a framing distribution the "
        "clinical data does not have."
    ),
}

#: **[MEASURED 2026-07-29] What the percentile grid cannot reproduce.** Drawing
#: 20,000 samples through ``sample_empirical`` against the cohort:
#:
#: | statistic | cohort | empirical | uniform |
#: |---|---|---|---|
#: | mean | 0.7475 | **0.7490** | 0.8242 |
#: | median | 0.7404 | **0.7391** | 0.8224 |
#: | sd | 0.0887 | 0.0966 | 0.1577 |
#: | portrait fraction | 0.9958 | **0.9729** | 0.8229 |
#:
#: Mean and median land within 0.0015. **The portrait fraction does not**: 2.7%
#: of draws come out landscape against the cohort's 0.42%.
#:
#: The cause is structural, not a bug. The cohort's single landscape crop sits in
#: the top 5% bin, so linear interpolation from p95=0.9173 to p100=1.0986 smears
#: one outlier across that whole bin -- about 2.7% of draws land above 1.0. It
#: also inflates the sd for the same reason.
#:
#: **Not corrected, deliberately.** Truncating or reweighting the top bin until
#: the landscape fraction matched would be tuning the sampler to a statistic, and
#: the percentile grid is what the available data supports. ``describe`` reports
#: ``fraction_portrait`` beside the cohort's so the gap is visible in every run
#: rather than assumed away. If it matters, the fix is the 237 raw ratios, not a
#: correction factor.
EMPIRICAL_SAMPLING_LIMITATION = {
    "measured": "2026-07-29, 20000 draws",
    "mean": {"cohort": 0.7475, "empirical": 0.7490, "uniform": 0.8242},
    "median": {"cohort": 0.7404, "empirical": 0.7391, "uniform": 0.8224},
    "sd": {"cohort": 0.0887, "empirical": 0.0966, "uniform": 0.1577},
    "fraction_portrait": {"cohort": 0.9958, "empirical": 0.9729, "uniform": 0.8229},
    "cause": (
        "the cohort's single landscape crop sits in the top 5% bin, so linear "
        "interpolation from p95=0.9173 to p100=1.0986 smears one outlier across "
        "that bin -- about 2.7% of draws land above 1.0 against the cohort's 0.42%"
    ),
    "not_corrected_because": (
        "truncating or reweighting the top bin until the landscape fraction "
        "matched would be tuning the sampler to a statistic. The percentile grid "
        "is what the data supports; the real fix is the 237 raw ratios."
    ),
}

#: **[MEASURED 2026-07-30, 5,499 faces] The predicted defect, confirmed at scale
#: and diagnosed exactly.** The full masked build realised
#: ``fraction_portrait`` **0.9725** against the cohort's 0.9958 -- 2.75% landscape
#: against 0.42%, so **151 landscape faces where the cohort rate implies ~23**.
#:
#: The arithmetic closes completely, which is what makes this a diagnosis rather
#: than an observation. Linear interpolation spreads the top 5% of mass uniformly
#: across ``[p95, p100] = [0.9173, 1.0986]``, so the predicted fraction above 1.0
#: is::
#:
#:     (1.0986 - 1.0) / (1.0986 - 0.9173) * 0.05 = 0.02719
#:
#: against an observed **0.0275**. The cohort has ONE point at 1.0986 and nothing
#: between 0.9173 and 1.0986; the interpolation invents a uniform tail across a
#: gap the data says is empty.
#:
#: ``EMPIRICAL_SAMPLING_LIMITATION`` predicted this from 20,000 draws before the
#: build and named the fix -- "the real fix is the 237 raw ratios, not a
#: correction factor". That is now being taken, as ``sample_observed``.
#:
#: **The alternatives are worse and are recorded so they are not revisited.**
#: A piecewise-CONSTANT inverse CDF over the same grid puts 1/20 of the mass on
#: ``p100`` exactly, giving a 5% landscape rate -- worse than the 2.75% it would
#: replace. Truncating or reweighting the top bin is tuning the sampler to a
#: statistic, which the limitation note already ruled out.
AR_TAIL_DEFECT = {
    "measured": "2026-07-30, full build, 5499 faces",
    "realised_fraction_portrait": 0.9725,
    "cohort_fraction_portrait": 0.9958,
    "n_landscape_realised": 151,
    "n_landscape_implied_by_cohort": 23,
    "predicted_fraction_above_1": 0.02719,
    "observed_fraction_above_1": 0.0275,
    "cause": (
        "piecewise-linear interpolation across [p95, p100] = [0.9173, 1.0986], a "
        "wide sparse bin holding a single observation at its top. The "
        "interpolation invents a uniform tail across a gap the data says is empty."
    ),
    "fix": "sample_observed -- draw with replacement from the 237 observed ratios",
    "rejected_alternatives": {
        "piecewise_constant_cdf": "puts 1/20 of the mass on p100: 5% landscape, worse",
        "truncate_top_bin": "tuning the sampler to a statistic; ruled out already",
    },
}

#: **[MEASURED] The 237 observed cleft aspect ratios**, sorted, extracted from
#: ``data/staged/staged_v1/geometry.csv`` on the cluster 2026-07-30.
#:
#: **Tier: SHAREABLE [DECIDED 2026-07-30].** No patient identifier, and sorted
#: they carry no ordering that could reintroduce one -- the same class as the 21
#: percentiles already committed here, at finer granularity. Mapping a ratio back
#: to a patient would need the images, which are cluster-only. Taken as a
#: deliberate decision rather than by analogy, because it is a much larger
#: constant than the percentile grid it replaces.
#:
#: **Verified against the constants already recorded here** rather than trusted
#: as pasted -- see ``test_the_observed_ratios_reproduce_every_recorded_constant``,
#: which checks all 21 percentiles, the mean, median, sd, IQR and both counts.
#: The endpoints alone would not have caught a transcription error in the middle.
#:
#: The landscape rate is the point: **exactly one value above 1.0** (1.098584,
#: with the next at 0.981038), so a bootstrap gives 1/237 = 0.42% -- the cohort's
#: own rate, where interpolation across that empty gap gave 2.75%.
AR_OBSERVED: tuple[float, ...] = (
    0.553067, 0.572143, 0.573626, 0.582266, 0.590031, 0.590489, 0.592098, 0.594203,
    0.602273, 0.604619, 0.605968, 0.607153, 0.609879, 0.613149, 0.625072, 0.626405,
    0.631319, 0.638179, 0.638554, 0.638554, 0.639706, 0.640814, 0.642303, 0.642857,
    0.644999, 0.645957, 0.646149, 0.648463, 0.649300, 0.650015, 0.650209, 0.650898,
    0.651184, 0.652500, 0.655308, 0.658086, 0.659084, 0.660687, 0.662298, 0.664183,
    0.667519, 0.668410, 0.673798, 0.675264, 0.676947, 0.677570, 0.677922, 0.678419,
    0.679063, 0.679562, 0.679688, 0.679688, 0.681191, 0.681932, 0.683387, 0.687204,
    0.688530, 0.689185, 0.689754, 0.690718, 0.693227, 0.693934, 0.694428, 0.695013,
    0.696251, 0.697145, 0.697816, 0.698795, 0.699388, 0.700726, 0.700968, 0.700968,
    0.701362, 0.701420, 0.701647, 0.702637, 0.703009, 0.703009, 0.703236, 0.706303,
    0.706360, 0.707222, 0.707587, 0.708048, 0.709239, 0.709239, 0.709239, 0.709667,
    0.711172, 0.712459, 0.712927, 0.713407, 0.714098, 0.714333, 0.716925, 0.717635,
    0.720381, 0.721448, 0.721992, 0.721992, 0.722569, 0.722647, 0.723994, 0.725785,
    0.727729, 0.729560, 0.730511, 0.730666, 0.731157, 0.733964, 0.734956, 0.736111,
    0.736976, 0.738362, 0.738864, 0.739116, 0.739619, 0.740375, 0.740426, 0.740968,
    0.742145, 0.742145, 0.742564, 0.742906, 0.743923, 0.745710, 0.745966, 0.746223,
    0.746479, 0.746781, 0.746992, 0.747059, 0.748225, 0.748387, 0.748794, 0.749310,
    0.749354, 0.749569, 0.751643, 0.752163, 0.753729, 0.754226, 0.755427, 0.758508,
    0.758721, 0.758994, 0.760140, 0.761654, 0.762500, 0.765396, 0.765410, 0.766182,
    0.766761, 0.768116, 0.769204, 0.769476, 0.770294, 0.770567, 0.772210, 0.773115,
    0.773310, 0.773860, 0.774964, 0.775321, 0.776349, 0.777480, 0.777480, 0.780924,
    0.781655, 0.783345, 0.783345, 0.786464, 0.786749, 0.787111, 0.787890, 0.790757,
    0.791937, 0.792198, 0.793645, 0.795388, 0.796846, 0.797139, 0.800958, 0.801254,
    0.807507, 0.810821, 0.811124, 0.812336, 0.812640, 0.815691, 0.817225, 0.817840,
    0.819382, 0.822483, 0.825294, 0.828757, 0.829389, 0.830340, 0.831210, 0.833206,
    0.834869, 0.836735, 0.840944, 0.844868, 0.851214, 0.853496, 0.853831, 0.854839,
    0.859913, 0.860253, 0.863672, 0.876917, 0.879401, 0.879401, 0.892036, 0.892402,
    0.895344, 0.897934, 0.899793, 0.901286, 0.903159, 0.904663, 0.905417, 0.912642,
    0.916491, 0.920373, 0.920763, 0.921935, 0.926257, 0.927443, 0.927839, 0.934624,
    0.938256, 0.941508, 0.961326, 0.981038, 1.098584,
)

#: [MEASURED] Pad fraction over the same 237 crops. Confirms the figure the
#: Phase 5 brief quotes and the one in the frozen record.
CLEFT_PAD_FRACTION = {
    "mean": 0.2534,
    "median": 0.2589,
    "min": 0.0179,
    "max": 0.4464,
}


class DistributionError(RuntimeError):
    """The distribution could not be sampled."""


def inverse_cdf(quantiles) -> np.ndarray:
    """Aspect ratios at the given quantiles, piecewise-linear between percentiles.

    ``np.interp`` over the measured percentile grid. Linear interpolation is what
    a percentile grid supports: it assumes the ratio varies smoothly *between*
    measured points and adds no shape the data does not show. Fitting a normal or
    a beta to 21 order statistics would be inventing a distribution.
    """
    array = np.asarray(quantiles, dtype=float)
    if array.size and (array.min() < 0.0 or array.max() > 1.0):
        raise DistributionError(
            f"quantiles must lie in [0, 1]; got [{array.min()}, {array.max()}]"
        )
    return np.interp(array, np.asarray(AR_QUANTILES, dtype=float), AR_PERCENTILES)


def sample_empirical(n: int, seed: int) -> np.ndarray:
    """``n`` aspect ratios drawn from the cohort's own distribution shape.

    **Superseded by ``sample_observed`` where the raw ratios are available.** This
    interpolates, and interpolation across ``[p95, p100]`` invents a uniform tail
    the cohort does not have -- 2.75% landscape against 0.42%, diagnosed exactly in
    ``AR_TAIL_DEFECT``. Kept so earlier runs can be re-derived.
    """
    if n < 0:
        raise DistributionError(f"n must be non-negative, got {n}")
    draws = np.random.default_rng(seed).uniform(0.0, 1.0, size=n)
    return inverse_cdf(draws)


def sample_observed(n: int, seed: int, observed=None) -> np.ndarray:
    """``n`` aspect ratios drawn WITH REPLACEMENT from the observed ratios.

    **The correct sampler, and it is simpler than the one it replaces.** A
    bootstrap over the raw values reproduces the empirical distribution exactly --
    including a sparse tail, which is where interpolation fails -- and it removes
    the parameterisation step rather than adding one. There is no grid, no
    interpolation rule, and nothing between the data and the draw.

    The cost is the constant: 237 values instead of 21 percentiles. That is the
    trade, and it is the right way round -- a bigger constant buys a sampler with
    no invented structure in it.

    Refuses rather than falling back when ``AR_OBSERVED`` is empty: a silent
    fallback to ``sample_empirical`` would reintroduce the tail defect while the
    config still recorded ``observed``, which is a run whose metrics.json
    misdescribes what produced it.
    """
    if n < 0:
        raise DistributionError(f"n must be non-negative, got {n}")
    values = AR_OBSERVED if observed is None else tuple(observed)
    if not values:
        source = "AR_OBSERVED" if observed is None else "the observed= argument"
        raise DistributionError(
            f"{source} is empty, so the observed-ratio sampler has nothing to "
            "draw from. The 237 cleft aspect ratios come from "
            "data/staged/staged_v1/geometry.csv, which is CLUSTER-ONLY. Refusing "
            "rather than falling back to sample_empirical: that would "
            "reintroduce the interpolated tail (AR_TAIL_DEFECT) while the run "
            "still recorded 'observed'."
        )
    array = np.asarray(values, dtype=float)
    return np.random.default_rng(seed).choice(array, size=n, replace=True)


def describe(realised) -> dict:
    """Compare a realised set of ratios against the cohort. SHAREABLE."""
    array = np.asarray(realised, dtype=float)
    if array.size == 0:
        raise DistributionError("no ratios to describe")
    return {
        "n": int(array.size),
        "mean": round(float(array.mean()), 6),
        "median": round(float(np.median(array)), 6),
        "sd": round(float(array.std(ddof=1)), 6) if array.size > 1 else 0.0,
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "fraction_portrait": round(float((array < 1.0).mean()), 4),
        "cleft": dict(CLEFT_AR),
        "mean_gap": round(float(array.mean()) - CLEFT_AR["mean"], 6),
        "median_gap": round(float(np.median(array)) - CLEFT_AR["median"], 6),
        "cleft_fraction_portrait": round(
            CLEFT_AR["n_portrait"] / CLEFT_AR["n"], 4
        ),
    }
