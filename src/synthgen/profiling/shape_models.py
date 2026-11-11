"""Detailed shape models for numeric, text and date columns (SPEC 8.3 C, D, E)."""

from __future__ import annotations

from enum import StrEnum

from synthgen.domain.base import StrictModel


class DistributionFit(StrictModel):
    """One fitted candidate: SciPy name and parameters, AIC and KS goodness of fit."""

    name: str
    params: list[float]
    aic: float
    ks_statistic: float
    ks_pvalue: float


class CountFits(StrictModel):
    """Poisson and negative-binomial fits for non-negative integer data."""

    poisson_lambda: float
    poisson_aic: float
    nbinom_n: float | None
    nbinom_p: float | None
    nbinom_aic: float | None


class NumericShape(StrictModel):
    """Spread, shape, round numbers, outliers, modes, step and first digits of a numeric column."""

    range: float
    iqr: float
    mad: float
    variation: float | None
    skew: float
    kurtosis: float
    round_rates: list[tuple[float, float]]
    iqr_outlier_rate: float
    zscore_outlier_rate: float
    modified_z_outlier_rate: float
    modes: list[float]
    constant_step: float | None
    benford_pvalue: float | None


class NormalityTests(StrictModel):
    """p-values of the normality tests (Shapiro on at most `shapiro_max_rows` values)."""

    shapiro_pvalue: float
    normaltest_pvalue: float
    anderson_statistic: float
    anderson_pvalue: float
    jarque_bera_pvalue: float


class Spread(StrictModel):
    """Median, 90th and 99th percentile, and maximum of a quantity."""

    p50: float
    p90: float
    p99: float
    max: float


class TextDetails(StrictModel):
    """Value-free text statistics: lengths, character classes, case, spaces, words, validity."""

    length_percentiles: list[tuple[float, float]]
    char_classes: list[tuple[str, float]]
    case_styles: list[tuple[str, float]]
    leading_space_rate: float
    trailing_space_rate: float
    word_count: Spread
    validity: list[tuple[str, float]]


class TextFormat(StrictModel):
    """Fixed parts of values. Never stored for sensitive columns: a fixed part is a value."""

    common_prefix: str
    common_suffix: str
    fixed_positions: list[tuple[int, str]]
    """Positions holding the same character in nearly every value of the dominant mask."""


class Granularity(StrEnum):
    """Finest unit in which a date or time column varies."""

    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


class Trend(StrictModel):
    """Linear trend of daily counts; slope is relative to the mean daily count."""

    relative_slope_per_day: float
    pvalue: float
    r_squared: float


class Seasonality(StrictModel):
    """Strongest periodogram peak of the daily counts."""

    period_days: float
    power_share: float


class Gaps(StrictModel):
    """Gaps between consecutive distinct values, in seconds, and the share of empty days."""

    seconds: Spread
    empty_day_rate: float


class Growth(StrictModel):
    """Rows per day over the range, and sampled counts per month or per year."""

    rows_per_day: float
    unit: str
    periods: list[tuple[str, int]]


class TemporalShape(StrictModel):
    """Trend, seasonality, gaps, growth and calendar shares (SPEC 8.3 E)."""

    granularity: Granularity
    trend: Trend | None
    seasonality: Seasonality | None
    gaps: Gaps
    growth: Growth
    future_rate: float
    sentinel_rate: float
    weekend_share: float
    business_hours_share: float | None
