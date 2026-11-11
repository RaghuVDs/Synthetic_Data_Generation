from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from synthgen.config import ProfilingSettings
from synthgen.profiling.models import Granularity, TemporalShape
from synthgen.profiling.temporal import parse_times, temporal_profile
from synthgen.profiling.temporal_series import (
    daily_counts,
    gaps,
    granularity,
    growth,
    seasonality,
    trend,
)

START = datetime(2024, 1, 1)  # a Monday
NOW = datetime(2030, 1, 1, tzinfo=UTC)


def _texts(moments: list[datetime]) -> list[str]:
    return [m.strftime("%Y-%m-%d %H:%M:%S") for m in moments]


def _profile(texts: list[str], settings: ProfilingSettings, now: datetime = NOW) -> TemporalShape:
    ordered = sorted(texts)
    return temporal_profile(
        ordered[0], ordered[-1], texts, count=len(texts), now=now, settings=settings
    ).shape.value


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["2024-01-01", "2024-02-01", "2024-07-01"], Granularity.MONTH),
        (["2024-01-01", "2024-01-15"], Granularity.DAY),
        (["2024-01-01 10:00:00", "2024-01-01 13:00:00"], Granularity.HOUR),
        (["2024-01-01 10:05:00", "2024-01-01 13:00:00"], Granularity.MINUTE),
        (["2024-01-01 10:05:07", "2024-01-01 13:00:00"], Granularity.SECOND),
    ],
)
def test_granularity(texts: list[str], expected: Granularity) -> None:
    assert granularity(parse_times(texts)) is expected


def test_trend_recovers_growing_daily_counts() -> None:
    series = np.arange(10, 110, dtype=np.int64)  # +1 a day from a mean of 59.5
    found = trend(series)
    assert found is not None
    assert found.relative_slope_per_day == pytest.approx(1 / 59.5)
    assert found.r_squared == pytest.approx(1.0) and found.pvalue < 1e-6
    assert trend(np.full(30, 5, dtype=np.int64)) is None


def test_weekly_seasonality() -> None:
    weeks = np.tile(np.array([10, 10, 10, 10, 10, 2, 2], dtype=np.int64), 16)
    found = seasonality(weeks, 14)
    assert found is not None
    assert found.period_days == pytest.approx(7, abs=0.1)
    assert seasonality(weeks[:10], 14) is None


def test_gaps_and_empty_days() -> None:
    times = parse_times(["2024-01-01", "2024-01-02", "2024-01-05"])
    assert daily_counts(times).tolist() == [1, 1, 0, 0, 1]
    found = gaps(times)
    assert found.seconds.max == 3 * 86_400
    assert found.empty_day_rate == pytest.approx(2 / 5)


def test_growth_switches_to_years_for_long_ranges(profiling_settings: ProfilingSettings) -> None:
    short = growth(
        parse_times(["2024-01-05", "2024-01-20", "2024-03-01"]), 30, 56, profiling_settings
    )
    assert short.unit == "month"
    assert short.periods == [("2024-01", 2), ("2024-03", 1)]
    assert short.rows_per_day == pytest.approx(30 / 56)
    long = growth(parse_times(["1950-06-01", "2020-06-01"]), 2, 25_567, profiling_settings)
    assert long.unit == "year" and long.periods == [("1950", 1), ("2020", 1)]


def test_calendar_shares_and_sentinels(profiling_settings: ProfilingSettings) -> None:
    moments = [START + timedelta(days=d, hours=h) for d in range(14) for h in (3, 10, 15)]
    shape = _profile(_texts(moments), profiling_settings)
    assert shape.weekend_share == pytest.approx(4 / 14)
    assert shape.business_hours_share == pytest.approx((10 / 14) * (2 / 3))
    assert shape.granularity is Granularity.HOUR
    dates = ["1900-01-01", "2024-01-01", "2024-01-02", "9999-12-31"]
    dated = _profile(dates, profiling_settings)
    assert dated.sentinel_rate == 0.5 and dated.business_hours_share is None
    assert dated.future_rate == 0.25
    assert dated.growth.periods == [("2024-01", 2)]  # sentinels stay out of the series


def test_long_ranges_skip_trend_and_seasonality(profiling_settings: ProfilingSettings) -> None:
    yearly = [f"{year}-03-01" for year in range(1950, 2020)]
    shape = _profile(yearly, profiling_settings)
    assert shape.trend is None and shape.seasonality is None
    assert shape.growth.unit == "year"
    weekly = [(START + timedelta(days=d)).strftime("%Y-%m-%d") for d in range(70) if d % 7 < 5]
    assert _profile(weekly, profiling_settings).seasonality is not None


def test_future_rate_uses_an_aware_reference_in_utc(profiling_settings: ProfilingSettings) -> None:
    texts = ["2024-01-01 10:00:00", "2024-01-01 12:00:00"]
    ahead = timezone(timedelta(hours=2))
    shape = _profile(texts, profiling_settings, now=datetime(2024, 1, 1, 13, tzinfo=ahead))
    assert shape.future_rate == 0.5  # 13:00 at +02:00 is 11:00 UTC
