"""Date and time statistics (SPEC 8.3 E). Dates are ISO-8601 text in SQLite."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from synthgen.config import ProfilingSettings
from synthgen.profiling.models import Measured, StatSource, TemporalProfile
from synthgen.profiling.shape_models import Granularity, TemporalShape
from synthgen.profiling.temporal_series import (
    daily_counts,
    gaps,
    granularity,
    growth,
    seasonality,
    trend,
)

MONTHS = 12
WEEKDAYS = 7
DAYS_OF_MONTH = 31
HOURS = 24
SECONDS_PER_DAY = 86_400
DATE_ONLY = (Granularity.DAY, Granularity.MONTH)


def parse_times(texts: Sequence[str]) -> NDArray[np.datetime64]:
    """ISO dates or timestamps (space or `T` separator) as second-resolution datetimes."""
    return np.array([t.replace(" ", "T") for t in texts], dtype="datetime64[s]")


def epoch_seconds(values: Sequence[object]) -> NDArray[np.float64]:
    """Seconds since 1970 of ISO-8601 values; NaN for nulls and values that do not parse."""
    texts = [None if v is None else str(v) for v in values]
    stamps = pd.to_datetime(pd.Series(texts, dtype=object), errors="coerce", format="ISO8601")
    seconds = stamps.to_numpy("datetime64[s]").astype(np.int64).astype(np.float64)
    return np.where(stamps.isna().to_numpy(), np.nan, seconds)


def temporal_profile(  # noqa: PLR0913 - exact SQL facts, the sample and the reference time
    minimum: str,
    maximum: str,
    texts: Sequence[str],
    *,
    count: int,
    now: datetime,
    settings: ProfilingSettings,
) -> TemporalProfile:
    """Range from exact SQL min/max and count; calendar mix and shape from a non-empty sample."""
    times = parse_times(texts)
    seconds = times.astype(np.int64)
    days = seconds // SECONDS_PER_DAY
    months = times.astype("datetime64[M]").astype(np.int64) % MONTHS
    weekdays = (days + 3) % WEEKDAYS  # 1970-01-01 was a Thursday; Monday = 0
    day_of_month = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(np.int64)
    hours = (seconds % SECONDS_PER_DAY) // 3600
    span = float((parse_times([maximum])[0] - parse_times([minimum])[0]).astype(np.int64))
    span_days = span / SECONDS_PER_DAY
    sample = StatSource.SAMPLE
    return TemporalProfile(
        minimum=Measured[str](value=minimum, source=StatSource.SQL),
        maximum=Measured[str](value=maximum, source=StatSource.SQL),
        span_days=Measured[float](value=span_days, source=StatSource.DERIVED),
        month_counts=Measured[list[int]](value=_bincount(months, MONTHS), source=sample),
        weekday_counts=Measured[list[int]](value=_bincount(weekdays, WEEKDAYS), source=sample),
        day_of_month_counts=Measured[list[int]](
            value=_bincount(day_of_month, DAYS_OF_MONTH), source=sample
        ),
        hour_counts=Measured[list[int]](value=_bincount(hours, HOURS), source=sample),
        shape=Measured[TemporalShape](
            value=_shape(
                times, weekdays, hours, count=count, span_days=span_days, now=now, settings=settings
            ),
            source=sample,
        ),
    )


def _shape(  # noqa: PLR0913 - calendar parts computed once by the caller
    times: NDArray[np.datetime64],
    weekdays: NDArray[np.int64],
    hours: NDArray[np.int64],
    *,
    count: int,
    span_days: float,
    now: datetime,
    settings: ProfilingSettings,
) -> TemporalShape:
    unit = granularity(times)
    weekend = np.isin(weekdays, settings.weekend_days)
    start, end = settings.business_hours
    sentinel = np.isin(
        times.astype("datetime64[D]"), np.array(settings.sentinel_dates, dtype="datetime64[D]")
    )
    regular = times[~sentinel] if not np.all(sentinel) else times
    series = _daily_series(regular, settings.max_series_days)
    return TemporalShape(
        granularity=unit,
        trend=None if series is None else trend(series),
        seasonality=None if series is None else seasonality(series, settings.min_series_days),
        gaps=gaps(regular),
        growth=growth(regular, count, span_days, settings),
        future_rate=float(np.mean(times > _naive_utc(now))),
        sentinel_rate=float(np.mean(sentinel)),
        weekend_share=float(np.mean(weekend)),
        business_hours_share=None
        if unit in DATE_ONLY
        else float(np.mean(~weekend & (hours >= start) & (hours < end))),
    )


def _daily_series(times: NDArray[np.datetime64], max_days: int) -> NDArray[np.int64] | None:
    """Daily counts without sentinel dates; None when the range is too long to model."""
    days = times.astype("datetime64[D]")
    if int((np.max(days) - np.min(days)).astype(np.int64)) >= max_days:
        return None
    return daily_counts(times)


def _naive_utc(moment: datetime) -> np.datetime64:
    """SQLite stores naive UTC text; an aware reference time is converted before comparing."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(UTC).replace(tzinfo=None)
    return np.datetime64(moment, "s")


def _bincount(values: NDArray[np.int64], size: int) -> list[int]:
    return [int(c) for c in np.bincount(values, minlength=size)[:size]]
