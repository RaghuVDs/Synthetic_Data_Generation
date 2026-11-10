"""Core date and time statistics (SPEC 8.3 E). Dates are ISO-8601 text in SQLite."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from synthgen.profiling.models import Measured, StatSource, TemporalProfile

MONTHS = 12
WEEKDAYS = 7
HOURS = 24
SECONDS_PER_DAY = 86_400


def parse_times(texts: Sequence[str]) -> NDArray[np.datetime64]:
    """ISO dates or timestamps (space or `T` separator) as second-resolution datetimes."""
    return np.array([t.replace(" ", "T") for t in texts], dtype="datetime64[s]")


def temporal_profile(minimum: str, maximum: str, texts: Sequence[str]) -> TemporalProfile:
    """Range from exact SQL min/max; month, weekday and hour mix from the sample."""
    times = parse_times(texts)
    seconds = times.astype(np.int64)
    days = seconds // SECONDS_PER_DAY
    months = times.astype("datetime64[M]").astype(np.int64) % MONTHS
    weekdays = (days + 3) % WEEKDAYS  # 1970-01-01 was a Thursday; Monday = 0
    hours = (seconds % SECONDS_PER_DAY) // 3600
    span = (parse_times([maximum])[0] - parse_times([minimum])[0]).astype(np.int64)
    sample = StatSource.SAMPLE
    return TemporalProfile(
        minimum=Measured[str](value=minimum, source=StatSource.SQL),
        maximum=Measured[str](value=maximum, source=StatSource.SQL),
        span_days=Measured[float](value=float(span) / SECONDS_PER_DAY, source=StatSource.DERIVED),
        month_counts=Measured[list[int]](value=_bincount(months, MONTHS), source=sample),
        weekday_counts=Measured[list[int]](value=_bincount(weekdays, WEEKDAYS), source=sample),
        hour_counts=Measured[list[int]](value=_bincount(hours, HOURS), source=sample),
    )


def _bincount(values: NDArray[np.int64], size: int) -> list[int]:
    return [int(c) for c in np.bincount(values, minlength=size)[:size]]
