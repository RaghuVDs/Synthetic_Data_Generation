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


class NumericProfile(StrictModel):
    """Core numeric statistics (SPEC 8.3 C)."""

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


class CategoricalProfile(StrictModel):
    """Core categorical statistics (SPEC 8.3 D). `other_count`: rows beyond the stored values."""

    frequencies: Measured[list[ValueCount]]
    other_count: Measured[int]
    entropy: Measured[float]
    normalized_entropy: Measured[float]
    rare_count: Measured[int]


class TextShape(StrictModel):
    """Lengths and masks: value-free, so it is all a sensitive column keeps."""

    length_min: Measured[int]
    length_max: Measured[int]
    length_mean: Measured[float]
    masks: Measured[list[MaskShare]]


class TemporalProfile(StrictModel):
    """Core date and time statistics (SPEC 8.3 E). Counts are indexed by month, weekday, hour."""

    minimum: Measured[str]
    maximum: Measured[str]
    span_days: Measured[float]
    month_counts: Measured[list[int]]
    weekday_counts: Measured[list[int]]
    hour_counts: Measured[list[int]]


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
