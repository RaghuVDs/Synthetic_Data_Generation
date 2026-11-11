"""Date and time series statistics (SPEC 8.3 E): granularity, trend, seasonality, gaps, growth.

Pure functions over second-resolution datetimes from the sample.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal, stats

from synthgen.config import ProfilingSettings
from synthgen.profiling.behaviour import spread
from synthgen.profiling.shape_models import Gaps, Granularity, Growth, Seasonality, Trend

Times = NDArray[np.datetime64]
SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
SECONDS_PER_MINUTE = 60
MIN_TREND_DAYS = 3


def granularity(times: Times) -> Granularity:
    """Finest unit in which the values vary: every value at midnight means day or month."""
    seconds = times.astype(np.int64)
    if np.all(seconds % SECONDS_PER_DAY == 0):
        first_of_month = times.astype("datetime64[D]") == times.astype("datetime64[M]")
        return Granularity.MONTH if bool(np.all(first_of_month)) else Granularity.DAY
    if np.all(seconds % SECONDS_PER_HOUR == 0):
        return Granularity.HOUR
    if np.all(seconds % SECONDS_PER_MINUTE == 0):
        return Granularity.MINUTE
    return Granularity.SECOND


def daily_counts(times: Times) -> NDArray[np.int64]:
    """Values per calendar day from the first to the last day, empty days included."""
    days = times.astype("datetime64[D]").astype(np.int64)
    return np.bincount(days - np.min(days)).astype(np.int64)


def trend(series: NDArray[np.int64]) -> Trend | None:
    """Least-squares slope of the daily counts, relative to their mean."""
    if series.size < MIN_TREND_DAYS or not np.std(series):
        return None
    fit = stats.linregress(np.arange(series.size, dtype=np.float64), series.astype(np.float64))
    return Trend(
        relative_slope_per_day=float(fit.slope) / float(np.mean(series)),
        pvalue=float(fit.pvalue),
        r_squared=float(fit.rvalue) ** 2,
    )


def seasonality(series: NDArray[np.int64], min_days: int) -> Seasonality | None:
    """Period of the strongest periodogram peak, ignoring the zero frequency."""
    if series.size < min_days or not np.std(series):
        return None
    frequencies, power = signal.periodogram(series.astype(np.float64) - float(np.mean(series)))
    frequencies, power = frequencies[1:], power[1:]
    total = float(np.sum(power))
    if not total:
        return None
    peak = int(np.argmax(power))
    return Seasonality(
        period_days=1.0 / float(frequencies[peak]), power_share=float(power[peak]) / total
    )


def gaps(times: Times) -> Gaps:
    """Spread of the gaps between consecutive distinct values, and the empty-day share."""
    distinct = np.unique(times.astype(np.int64))
    days = np.unique(times.astype("datetime64[D]").astype(np.int64))
    calendar_days = int(days[-1] - days[0]) + 1
    return Gaps(
        seconds=spread(np.diff(distinct).astype(np.float64)),
        empty_day_rate=1.0 - days.size / calendar_days,
    )


def growth(times: Times, count: int, span_days: float, settings: ProfilingSettings) -> Growth:
    """Rows per day over the range, and sampled counts per month (per year for long ranges)."""
    months = times.astype("datetime64[M]")
    span_months = int((np.max(months) - np.min(months)).astype(np.int64)) + 1
    unit = "month" if span_months <= settings.max_series_months else "year"
    labels, counts = np.unique(
        times.astype("datetime64[M]" if unit == "month" else "datetime64[Y]"), return_counts=True
    )
    return Growth(
        rows_per_day=count / max(span_days, 1.0),
        unit=unit,
        periods=[(str(label), int(n)) for label, n in zip(labels, counts, strict=True)],
    )
