"""Profile models (SPEC 8.3). Every statistic records where it came from: SQL, sample or derived.

Profiles stay in the profile store and never reach an LLM; agents only see `LlmSafeProfile`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from synthgen.domain.base import StrictModel
from synthgen.domain.metadata import Sensitivity

T = TypeVar("T")


class StatSource(StrEnum):
    """Where a statistic came from."""

    SQL = "sql"
    SAMPLE = "sample"
    DERIVED = "derived"


class Measured(StrictModel, Generic[T]):
    """A value with its source."""

    value: T
    source: StatSource


class InferredType(StrEnum):
    """Real type of a column, inferred from the sample because SQLite typing is loose."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    TEXT = "text"
    DATE = "date"
    DATETIME = "datetime"
    EMPTY = "empty"


class ValueCount(StrictModel):
    """One value and how many rows hold it."""

    value: str | int | float
    count: int


class MaskShare(StrictModel):
    """A character-class mask (letters `A`, digits `9`, punctuation kept) and its share of rows."""

    mask: str
    share: float


class Histogram(StrictModel):
    """Equal-width histogram: `edges` has one more entry than `counts`."""

    edges: list[float]
    counts: list[int]


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


class NumericProfile(StrictModel):
    """Numeric statistics (SPEC 8.3 C)."""

    minimum: Measured[float]
    maximum: Measured[float]
    total: Measured[float]
    mean: Measured[float]
    std: Measured[float]
    variance: Measured[float]
    percentiles: Measured[list[tuple[float, float]]]
    quantiles: Measured[list[float]]
    histogram: Measured[Histogram]
    zero_rate: Measured[float]
    negative_rate: Measured[float]
    is_integer: Measured[bool]
    decimal_places: Measured[int]
    shape: Measured[NumericShape]
    fits: Measured[list[DistributionFit]]
    normality: Measured[NormalityTests] | None
    counts: Measured[CountFits] | None

    @property
    def best_fit(self) -> str | None:
        """Name of the lowest-AIC distribution, if any fitted."""
        return self.fits.value[0].name if self.fits.value else None


class CategoricalProfile(StrictModel):
    """Core categorical statistics (SPEC 8.3 D). `other_count`: rows beyond the stored values."""

    frequencies: Measured[list[ValueCount]]
    other_count: Measured[int]
    entropy: Measured[float]
    normalized_entropy: Measured[float]
    rare_count: Measured[int]
    zipf_exponent: Measured[float] | None


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


class TextShape(StrictModel):
    """Lengths, masks and details: value-free, so it is all a sensitive column keeps."""

    length_min: Measured[int]
    length_max: Measured[int]
    length_mean: Measured[float]
    masks: Measured[list[MaskShare]]
    details: Measured[TextDetails]


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


class TemporalProfile(StrictModel):
    """Date and time statistics (SPEC 8.3 E). Counts are indexed by month, weekday, day, hour."""

    minimum: Measured[str]
    maximum: Measured[str]
    span_days: Measured[float]
    month_counts: Measured[list[int]]
    weekday_counts: Measured[list[int]]
    day_of_month_counts: Measured[list[int]]
    hour_counts: Measured[list[int]]
    shape: Measured[TemporalShape]


class ColumnProfile(StrictModel):
    """Every-column statistics (SPEC 8.3 B) plus the type-specific group that applies."""

    name: str
    declared_type: str
    inferred_type: InferredType
    sensitivity: Sensitivity
    semantic_type: str | None
    null_count: Measured[int]
    null_rate: Measured[float]
    distinct_count: Measured[int]
    cardinality_ratio: Measured[float]
    is_unique: Measured[bool]
    is_constant: Measured[bool]
    top_value: Measured[str | int | float] | None
    top_share: Measured[float] | None
    empty_rate: Measured[float]
    numeric: NumericProfile | None
    categorical: CategoricalProfile | None
    text: TextShape | None
    text_format: Measured[TextFormat] | None
    temporal: TemporalProfile | None


class ChildrenPerParent(StrictModel):
    """How many child rows each parent row has, as (children, parents) pairs including zero."""

    child_column: str
    parent_table: str
    parent_column: str
    counts: Measured[list[tuple[int, int]]]
    orphan_rows: Measured[int]


class TableProfile(StrictModel):
    """A whole table (SPEC 8.3 A) with its columns and relationship shapes."""

    source: str
    table: str
    schema_hash: str
    row_count: Measured[int]
    primary_key_unique: Measured[bool] | None
    duplicate_row_rate: Measured[float]
    completeness: Measured[float]
    sample_rows: int
    columns: list[ColumnProfile]
    relationships: list[ChildrenPerParent]

    def column(self, name: str) -> ColumnProfile | None:
        """One column's profile, or None when the table has no such column."""
        return next((c for c in self.columns if c.name == name), None)
