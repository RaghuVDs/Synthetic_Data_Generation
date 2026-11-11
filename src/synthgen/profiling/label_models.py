"""Label-stratified statistics (SPEC 8.3 I): class balance, separation and per-class summaries."""

from __future__ import annotations

from synthgen.domain.base import StrictModel
from synthgen.profiling.shape_models import DistributionFit


class ClassBalance(StrictModel):
    """One class of a label column, counted exactly through SQL."""

    value: str
    count: int
    share: float


class ColumnSeparation(StrictModel):
    """How differently one column behaves in the minority class and the rest.

    KS and Wasserstein apply to numbers and dates; Wasserstein is in units of the pooled
    standard deviation. Jensen-Shannon (base 2) applies to every kind. All run from 0 (same).
    A date-time column is scored on its hour of day (kind `hour_of_day`) when that separates more.
    """

    column: str
    kind: str
    ks: float | None
    ks_pvalue: float | None
    wasserstein: float | None
    jensen_shannon: float
    score: float


class ClassNumbers(StrictModel):
    """A numeric column within one class."""

    count: int
    mean: float
    std: float
    percentiles: list[tuple[float, float]]
    best_fit: DistributionFit | None


class ClassTimes(StrictModel):
    """A date or time column within one class: weekday (Monday first) and hour counts."""

    weekday_counts: list[int]
    hour_counts: list[int]


class ClassColumn(StrictModel):
    """One strongly separating column summarised per class (groups C, D, E)."""

    column: str
    value: str
    numbers: ClassNumbers | None
    times: ClassTimes | None
    shares: list[tuple[str, float]] | None


class StratifiedProfile(StrictModel):
    """Group I for one label column."""

    label_column: str
    minority: str
    balance: list[ClassBalance]
    sample_rows: int
    separation: list[ColumnSeparation]
    """Columns ranked by score, capped at `max_stratified_columns`."""
    per_class: list[ClassColumn]
