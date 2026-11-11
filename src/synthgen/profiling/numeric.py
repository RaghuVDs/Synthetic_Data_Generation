"""Core numeric statistics (SPEC 8.3 C). Moments come from SQL; shape comes from the sample."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synthgen.config import ProfilingSettings
from synthgen.profiling.fitting import fit_counts, fit_distributions
from synthgen.profiling.models import (
    CountFits,
    DistributionFit,
    Histogram,
    Measured,
    NormalityTests,
    NumericProfile,
    NumericShape,
    StatSource,
)
from synthgen.profiling.numeric_shape import constant_step, normality, numeric_shape

MAX_DECIMALS = 6


@dataclass(frozen=True)
class SqlMoments:
    """Exact aggregates for one numeric column."""

    count: int
    minimum: float
    maximum: float
    total: float
    sum_squares: float


def _sql(value: float) -> Measured[float]:
    return Measured[float](value=float(value), source=StatSource.SQL)


def _sample(value: float) -> Measured[float]:
    return Measured[float](value=float(value), source=StatSource.SAMPLE)


def decimal_places(values: NDArray[np.float64]) -> int:
    """Fewest decimals (up to six) that represent every value exactly."""
    for places in range(MAX_DECIMALS + 1):
        if np.allclose(values, np.round(values, places), rtol=0.0, atol=1e-9):
            return places
    return MAX_DECIMALS


def numeric_profile(
    moments: SqlMoments,
    sample: NDArray[np.float64],
    settings: ProfilingSettings,
    *,
    distinct: int,
    full_fit: bool,
) -> NumericProfile:
    """Numeric statistics from exact moments and a non-empty, null-free sample.

    `distinct` is the exact distinct count. `full_fit` adds iterative distribution fits, KDE modes
    and normality tests; tables with more than `max_fit_columns` numeric columns skip them.
    """
    subsample = sample[: settings.fit_sample_size]
    step = constant_step(moments.count, distinct, moments.minimum, moments.maximum)
    tests = normality(subsample, settings) if full_fit else None
    counts_fit = fit_counts(subsample)
    n = moments.count
    mean = moments.total / n
    variance = max(0.0, (moments.sum_squares - n * mean * mean) / (n - 1)) if n > 1 else 0.0
    percentiles = np.percentile(sample, settings.percentiles)
    counts, edges = np.histogram(sample, bins=settings.histogram_bins)
    derived = StatSource.DERIVED
    return NumericProfile(
        minimum=_sql(moments.minimum),
        maximum=_sql(moments.maximum),
        total=_sql(moments.total),
        mean=Measured[float](value=mean, source=derived),
        std=Measured[float](value=float(np.sqrt(variance)), source=derived),
        variance=Measured[float](value=variance, source=derived),
        percentiles=Measured[list[tuple[float, float]]](
            value=[
                (float(p), float(v)) for p, v in zip(settings.percentiles, percentiles, strict=True)
            ],
            source=StatSource.SAMPLE,
        ),
        quantiles=Measured[list[float]](
            value=[
                float(q) for q in np.quantile(sample, np.linspace(0, 1, settings.quantile_points))
            ],
            source=StatSource.SAMPLE,
        ),
        histogram=Measured[Histogram](
            value=Histogram(edges=[float(e) for e in edges], counts=[int(c) for c in counts]),
            source=StatSource.SAMPLE,
        ),
        zero_rate=_sample(np.mean(sample == 0)),
        negative_rate=_sample(np.mean(sample < 0)),
        is_integer=Measured[bool](
            value=bool(np.all(np.mod(sample, 1) == 0)), source=StatSource.SAMPLE
        ),
        decimal_places=Measured[int](value=decimal_places(sample), source=StatSource.SAMPLE),
        shape=Measured[NumericShape](
            value=numeric_shape(sample, subsample, step, settings, full=full_fit),
            source=StatSource.SAMPLE,
        ),
        fits=Measured[list[DistributionFit]](
            value=fit_distributions(subsample, full=full_fit), source=StatSource.SAMPLE
        ),
        normality=None
        if tests is None
        else Measured[NormalityTests](value=tests, source=StatSource.SAMPLE),
        counts=None
        if counts_fit is None
        else Measured[CountFits](value=counts_fit, source=StatSource.SAMPLE),
    )
