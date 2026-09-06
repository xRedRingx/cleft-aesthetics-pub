"""The cleft aspect-ratio distribution and its sampler (Phase 5).

The percentiles are [MEASURED] from ``staged_v1/geometry.csv``, n=237. These
tests check the sampler reproduces the cohort's *shape*, that it agrees with the
frozen record, and that the one thing it cannot reproduce is recorded rather than
hidden.
"""

from __future__ import annotations

import numpy as np
import pytest

from cleft.geometry.staging import CLEFT_STAGED_GEOMETRY
from cleft.scut import ar_distribution
from cleft.scut import ar_distribution as AD


# --------------------------------------------------------------------------
# the interpolated tail, and the bootstrap that replaces it
# --------------------------------------------------------------------------


def test_the_interpolated_tail_defect_arithmetic_closes():
    """**[MEASURED 2026-07-30, 5,499 faces]** The realised landscape fraction is
    predicted exactly by the interpolation, which is what makes this a diagnosis
    rather than an observation: linear interpolation spreads the top 5% of mass
    uniformly across [p95, p100] = [0.9173, 1.0986], a bin holding one
    observation at its very top."""
    defect = ar_distribution.AR_TAIL_DEFECT
    p95, p100 = 0.9173, 1.0986
    predicted = (p100 - 1.0) / (p100 - p95) * 0.05

    assert predicted == pytest.approx(defect["predicted_fraction_above_1"], abs=1e-5)
    assert defect["observed_fraction_above_1"] == pytest.approx(predicted, abs=0.0005)
    assert defect["n_landscape_realised"] == 151
    assert "invents a uniform tail" in defect["cause"]


def test_the_defect_is_reproducible_from_the_sampler_itself():
    """Not taken from the build's report: drawn here, from the shipped sampler."""
    drawn = ar_distribution.sample_empirical(40_000, seed=20260730)
    fraction_above_one = float((drawn > 1.0).mean())
    assert fraction_above_one == pytest.approx(
        ar_distribution.AR_TAIL_DEFECT["predicted_fraction_above_1"], abs=0.003
    )
    # And it is far from the cohort's own rate, which is the whole problem.
    cohort = 1.0 - ar_distribution.CLEFT_AR["n_portrait"] / ar_distribution.CLEFT_AR["n"]
    assert fraction_above_one > 5 * cohort


def test_the_rejected_alternatives_are_recorded_with_their_numbers():
    """A piecewise-constant CDF is WORSE (1/20 of the mass lands on p100 = 5%
    landscape), and truncating the top bin is tuning the sampler to a statistic.
    Recorded so neither is revisited as though it were untried."""
    rejected = ar_distribution.AR_TAIL_DEFECT["rejected_alternatives"]
    assert "5% landscape, worse" in rejected["piecewise_constant_cdf"]
    assert "tuning the sampler" in rejected["truncate_top_bin"]


def test_the_bootstrap_sampler_reproduces_its_input_exactly():
    """**The fix.** Drawing with replacement from the observed values reproduces
    the empirical distribution including a sparse tail, because it never invents
    a value between two observations."""
    observed = (0.60, 0.70, 0.72, 0.74, 0.75, 0.78, 0.80, 1.0986)
    drawn = ar_distribution.sample_observed(50_000, seed=7, observed=observed)

    # Every draw is an observed value: no interpolation, so nothing in the gap.
    assert set(np.unique(drawn)).issubset(set(observed))
    assert not ((drawn > 0.80) & (drawn < 1.0986)).any(), (
        "a value appeared in the empty gap, which is the interpolation defect"
    )
    # And the sparse tail lands at its true rate rather than a smeared one.
    assert float((drawn > 1.0).mean()) == pytest.approx(1 / 8, abs=0.01)


def test_the_bootstrap_is_reproducible_from_the_seed():
    observed = (0.6, 0.7, 0.8, 0.9)
    first = ar_distribution.sample_observed(200, seed=11, observed=observed)
    assert np.array_equal(first, ar_distribution.sample_observed(200, 11, observed))
    assert not np.array_equal(
        first, ar_distribution.sample_observed(200, 12, observed)
    )


def test_the_bootstrap_refuses_rather_than_falling_back():
    """**A silent fallback would be the worst outcome**: the interpolated tail
    would return while metrics.json still recorded 'observed', so the run would
    misdescribe what produced it.

    Forced through the ``observed=`` argument rather than by asserting the module
    constant is empty. The earlier form asserted ``AR_OBSERVED == ()``, which made
    the refusal **untestable the moment the ratios were recorded** -- the branch
    would still exist and nothing would exercise it.
    """
    with pytest.raises(ar_distribution.DistributionError, match="nothing to"):
        ar_distribution.sample_observed(10, seed=1, observed=())


def test_the_observed_ratios_reproduce_every_recorded_constant():
    """**The paste is verified against the whole prior record, not its endpoints.**

    Min and max agreeing would say nothing about a transcription slip in the
    middle of 237 values. Every one of the 21 percentiles, the mean, median, sd,
    IQR and both counts are independent checks against constants recorded before
    these values existed -- so the two records can only agree if the data is the
    same data.
    """
    values = np.asarray(ar_distribution.AR_OBSERVED, dtype=float)

    assert values.size == ar_distribution.CLEFT_AR["n"] == 237
    assert np.all(np.diff(values) >= 0), "recorded sorted, so ties are visible"

    # All 21 percentiles, at the tolerance of the 4-decimal grid they are stored at.
    computed = np.percentile(values, np.arange(0, 101, 5))
    for pct, got, recorded in zip(
        range(0, 101, 5), computed, ar_distribution.AR_PERCENTILES
    ):
        assert got == pytest.approx(recorded, abs=5e-5), f"p{pct} disagrees"

    cohort = ar_distribution.CLEFT_AR
    assert values.mean() == pytest.approx(cohort["mean"], abs=5e-5)
    assert np.median(values) == pytest.approx(cohort["median"], abs=5e-5)
    assert values.std(ddof=1) == pytest.approx(cohort["sd"], abs=5e-5)
    assert np.percentile(values, 25) == pytest.approx(cohort["iqr"][0], abs=5e-5)
    assert np.percentile(values, 75) == pytest.approx(cohort["iqr"][1], abs=5e-5)

    assert int((values < 1.0).sum()) == cohort["n_portrait"] == 236
    assert int((values > 1.0).sum()) == cohort["n_landscape"] == 1


def test_the_bootstrap_lands_the_landscape_rate_on_the_cohort_exactly():
    """**The whole point of the change.** One value above 1.0 in 237, so a
    bootstrap gives 1/237 = 0.42% -- the cohort's own rate. Interpolation gave
    2.75% by smearing that single observation across an empty bin."""
    cohort_rate = 1.0 - ar_distribution.CLEFT_AR["n_portrait"] / ar_distribution.CLEFT_AR["n"]

    drawn = ar_distribution.sample_observed(60_000, seed=20260730)
    assert float((drawn > 1.0).mean()) == pytest.approx(cohort_rate, abs=0.002)

    # And nothing lands in the gap the interpolation used to fill.
    assert not ((drawn > 0.981038) & (drawn < 1.098584)).any(), (
        "a draw appeared between the top two observed ratios, which is the "
        "interpolated tail returning"
    )

    interpolated = ar_distribution.sample_empirical(60_000, seed=20260730)
    assert float((interpolated > 1.0).mean()) > 5 * cohort_rate, (
        "the sampler being replaced must still show the defect, or this "
        "comparison is not measuring the fix"
    )


def test_the_default_sampler_tracks_whether_the_ratios_are_available():
    """The constant and the default must move together. If AR_OBSERVED is pasted
    and the default is left on the interpolating sampler, the fix is present and
    unused -- which looks exactly like the fix being applied."""
    from cleft.scut import masked

    if ar_distribution.AR_OBSERVED:
        assert masked.DEFAULT_AR_SAMPLING == "observed", (
            "the 237 ratios are recorded, so the default sampler must be "
            "'observed'; the interpolated tail is a known defect"
        )
    else:
        assert masked.DEFAULT_AR_SAMPLING == "empirical"
    assert "observed" in masked.AR_SAMPLING


def test_the_distribution_agrees_with_the_frozen_record():
    """**The percentiles live outside ``staging.py`` because it is frozen**, so
    the two records could drift. This is what stops them."""
    frozen = CLEFT_STAGED_GEOMETRY["aspect_ratio"]
    assert AD.AR_PERCENTILES[0] == frozen["min"]
    assert AD.AR_PERCENTILES[10] == frozen["median"]
    assert AD.AR_PERCENTILES[-1] == frozen["max"]
    assert AD.CLEFT_AR["min"] == frozen["min"]
    assert AD.CLEFT_AR["median"] == frozen["median"]
    assert AD.CLEFT_AR["max"] == frozen["max"]
    assert AD.CLEFT_PAD_FRACTION["mean"] == CLEFT_STAGED_GEOMETRY["pad_fraction"]["mean"]


def test_the_percentile_grid_is_complete_and_monotone():
    assert len(AD.AR_PERCENTILES) == len(AD.AR_QUANTILES) == 21
    assert list(AD.AR_PERCENTILES) == sorted(AD.AR_PERCENTILES)
    assert AD.AR_QUANTILES[0] == 0.0 and AD.AR_QUANTILES[-1] == 1.0


def test_the_inverse_cdf_returns_the_measured_percentiles_at_their_quantiles():
    recovered = AD.inverse_cdf(AD.AR_QUANTILES)
    assert recovered == pytest.approx(AD.AR_PERCENTILES, abs=1e-9)


def test_the_inverse_cdf_interpolates_between_measured_points():
    """Linear between percentiles: it assumes the ratio varies smoothly and adds
    no shape the data does not show."""
    midpoint = AD.inverse_cdf([0.025])[0]
    assert AD.AR_PERCENTILES[0] < midpoint < AD.AR_PERCENTILES[1]


def test_a_quantile_outside_the_unit_interval_is_refused():
    with pytest.raises(AD.DistributionError, match=r"\[0, 1\]"):
        AD.inverse_cdf([1.5])


def test_sampling_reproduces_the_cohort_mean_and_median():
    """**The point of the whole module.** Uniform over the range gives 0.826
    against the cohort's 0.7475, nearly one SD high, because the range is
    dominated by two thin tails."""
    drawn = AD.sample_empirical(20000, seed=1337)
    assert drawn.mean() == pytest.approx(AD.CLEFT_AR["mean"], abs=0.01)
    assert np.median(drawn) == pytest.approx(AD.CLEFT_AR["median"], abs=0.01)


def test_sampling_beats_a_uniform_draw_on_the_cohort_mean():
    """Asserted as a comparison, not a bare tolerance -- the whole reason the
    percentiles were fetched."""
    empirical = AD.sample_empirical(20000, seed=1337)
    uniform = np.random.default_rng(1337).uniform(
        AD.CLEFT_AR["min"], AD.CLEFT_AR["max"], size=20000
    )
    true_mean = AD.CLEFT_AR["mean"]
    assert abs(empirical.mean() - true_mean) < abs(uniform.mean() - true_mean) / 5


def test_sampling_stays_inside_the_measured_range():
    drawn = AD.sample_empirical(5000, seed=7)
    assert drawn.min() >= AD.CLEFT_AR["min"]
    assert drawn.max() <= AD.CLEFT_AR["max"]


def test_sampling_is_reproducible_from_the_seed():
    assert np.array_equal(
        AD.sample_empirical(100, seed=7), AD.sample_empirical(100, seed=7)
    )
    assert not np.array_equal(
        AD.sample_empirical(100, seed=7), AD.sample_empirical(100, seed=8)
    )


def test_a_negative_count_is_refused():
    with pytest.raises(AD.DistributionError, match="non-negative"):
        AD.sample_empirical(-1, seed=1)


# --------------------------------------------------------------------------
# what the percentile grid cannot reproduce, recorded rather than hidden
# --------------------------------------------------------------------------


def test_the_portrait_fraction_is_a_known_shortfall():
    """**[MEASURED] 2.7% of draws come out landscape against the cohort's 0.42%.**

    Structural, not a bug: the cohort's single landscape crop sits in the top 5%
    bin, so linear interpolation from p95=0.9173 to p100=1.0986 smears one
    outlier across that whole bin.

    Not corrected -- truncating the top bin until the fraction matched would be
    tuning the sampler to a statistic. Recorded so it is visible in every run.
    """
    limitation = AD.EMPIRICAL_SAMPLING_LIMITATION
    assert limitation["fraction_portrait"]["cohort"] == 0.9958
    assert limitation["fraction_portrait"]["empirical"] < 0.9958
    assert "smears one outlier" in limitation["cause"]
    assert "tuning the sampler to a statistic" in limitation["not_corrected_because"]

    drawn = AD.sample_empirical(20000, seed=1337)
    assert (drawn < 1.0).mean() == pytest.approx(
        limitation["fraction_portrait"]["empirical"], abs=0.01
    )


def test_describe_reports_the_gap_against_the_cohort():
    described = AD.describe(AD.sample_empirical(2000, seed=1337))
    assert abs(described["mean_gap"]) < 0.02
    assert described["cleft"]["mean"] == AD.CLEFT_AR["mean"]
    # Both fractions side by side, so the known shortfall is visible.
    assert "fraction_portrait" in described
    assert "cleft_fraction_portrait" in described


def test_describe_refuses_an_empty_set():
    with pytest.raises(AD.DistributionError, match="no ratios"):
        AD.describe([])
