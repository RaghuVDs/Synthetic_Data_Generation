from __future__ import annotations

import numpy as np
import pytest

from synthgen.config import ProfilingSettings
from synthgen.profiling.fitting import fit_counts, fit_distributions
from synthgen.profiling.numeric_shape import (
    benford_pvalue,
    constant_step,
    modes,
    normality,
    numeric_shape,
)

RNG = np.random.default_rng(7)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (RNG.lognormal(3.0, 0.6, 2_000), "lognorm"),
        (RNG.normal(50.0, 10.0, 2_000), "norm"),
        (RNG.exponential(4.0, 2_000), "expon"),
    ],
)
def test_fits_recover_the_generating_family(data: np.ndarray, expected: str) -> None:
    fits = fit_distributions(data, full=True)
    assert fits[0].name == expected
    assert [f.aic for f in fits] == sorted(f.aic for f in fits)
    assert fits[0].ks_pvalue > 0.01
    light = fit_distributions(data, full=False)
    assert len(light) == 1 and light[0].name == expected


def test_too_few_distinct_values_are_not_fitted() -> None:
    assert fit_distributions(np.array([1.0, 2.0, 3.0] * 50), full=True) == []


def test_constant_step() -> None:
    assert constant_step(count=100, distinct=100, minimum=1, maximum=100) == 1
    assert constant_step(count=5, distinct=5, minimum=10, maximum=50) == 10
    assert constant_step(count=100, distinct=99, minimum=1, maximum=100) is None
    assert constant_step(count=3, distinct=3, minimum=0, maximum=1) is None


def test_bimodal_data_has_two_modes() -> None:
    data = np.concatenate([RNG.normal(10, 1, 1_000), RNG.normal(30, 1, 1_000)])
    found = modes(data, 512)
    assert len(found) == 2
    assert found[0] == pytest.approx(10, abs=1) and found[1] == pytest.approx(30, abs=1)


def test_outlier_rates(profiling_settings: ProfilingSettings) -> None:
    data = np.concatenate([RNG.normal(0, 1, 990), np.full(10, 50.0)])
    shape = numeric_shape(data, data, None, profiling_settings, full=True)
    for rate in (shape.iqr_outlier_rate, shape.zscore_outlier_rate, shape.modified_z_outlier_rate):
        assert rate >= 0.01
    assert shape.skew > 1


def test_round_rates(profiling_settings: ProfilingSettings) -> None:
    data = np.array([10.0, 20.0, 25.0, 5.0, 3.5, 1000.0])
    rates = dict(numeric_shape(data, data, None, profiling_settings, full=False).round_rates)
    assert rates[1] == pytest.approx(5 / 6)
    assert rates[10] == pytest.approx(3 / 6)
    assert rates[1000] == pytest.approx(1 / 6)


def test_benford() -> None:
    benford_like = 10 ** RNG.uniform(0, 4, 5_000)
    fitting = benford_pvalue(benford_like)
    assert fitting is not None and fitting > 0.01
    flat = RNG.uniform(100, 999, 5_000)
    assert benford_pvalue(10 ** RNG.uniform(0, 4, 50)) is None
    assert benford_pvalue(flat) is None  # spans under two decades
    uniform_digits = (
        np.concatenate([RNG.uniform(d * 10, d * 10 + 9, 600) for d in range(1, 10)]) * 100
    )
    flat_digits = benford_pvalue(np.concatenate([uniform_digits, [1.0, 100_000.0]]))
    assert flat_digits is not None and flat_digits < 1e-6


def test_normality(profiling_settings: ProfilingSettings) -> None:
    normal = normality(RNG.normal(0, 1, 2_000), profiling_settings)
    skewed = normality(RNG.lognormal(0, 1, 2_000), profiling_settings)
    assert normal is not None and skewed is not None
    assert normal.shapiro_pvalue > 0.01 and normal.jarque_bera_pvalue > 0.01
    assert skewed.shapiro_pvalue < 1e-3 and skewed.anderson_pvalue < 0.05
    assert normality(np.full(100, 3.0), profiling_settings) is None


def test_count_fits() -> None:
    fits = fit_counts(RNG.poisson(3.0, 5_000).astype(float))
    assert fits is not None and fits.poisson_lambda == pytest.approx(3.0, abs=0.1)
    overdispersed = fit_counts(RNG.negative_binomial(2, 0.3, 5_000).astype(float))
    assert overdispersed is not None and overdispersed.nbinom_aic is not None
    assert overdispersed.nbinom_aic < overdispersed.poisson_aic
    assert fit_counts(np.array([-1.0, 2.0])) is None
    assert fit_counts(np.array([1.5, 2.0])) is None
