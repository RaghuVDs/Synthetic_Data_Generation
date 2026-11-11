"""Cross-column (SPEC 8.3 F), cross-table (G) and behaviour (H) statistics.

Only strong pairs are stored, so wide tables stay small.
"""

from __future__ import annotations

from enum import StrEnum

from synthgen.domain.base import StrictModel
from synthgen.profiling.shape_models import CountFits, DistributionFit, Spread


class NumericPair(StrictModel):
    """Correlations of two numeric columns on rows where both are present."""

    a: str
    b: str
    pearson: float
    spearman: float
    kendall: float


class CategoricalPair(StrictModel):
    """Association of two categorical columns. Theil's U is the uncertainty of one explained."""

    a: str
    b: str
    cramers_v: float
    chi2_pvalue: float
    theils_u_b_given_a: float
    theils_u_a_given_b: float


class LevelSummary(StrictModel):
    """A numeric column within one level of a categorical column."""

    level: str
    share: float
    mean: float
    median: float
    p90: float


class LevelShares(StrictModel):
    """A categorical column's top values within one level of another categorical column."""

    level: str
    share: float
    values: list[tuple[str, float]]


class MixedPair(StrictModel):
    """A numeric column against a categorical one, with per-level summaries."""

    numeric: str
    categorical: str
    correlation_ratio: float
    anova_pvalue: float
    kruskal_pvalue: float
    point_biserial: float | None
    """Set when the categorical column has exactly two levels."""
    levels: list[LevelSummary]


class ConditionalShares(StrictModel):
    """Distribution of `dependent` within the top levels of a strongly associated `given`."""

    given: str
    dependent: str
    levels: list[LevelShares]


class RuleKind(StrEnum):
    """Deterministic relations found in the sample."""

    DETERMINES = "determines"
    """Each value of `a` maps to one value of `b` (functional dependency)."""
    LESS_EQUAL = "less_equal"
    """a <= b on every row where both are present."""
    SUM = "sum"
    """result = a + b."""
    PRODUCT = "product"
    """result = a * b."""


class ColumnRule(StrictModel):
    """One rule, the share of checked rows on which it holds, and how many rows were checked."""

    kind: RuleKind
    a: str
    b: str
    result: str | None
    holds: float
    rows: int


class NullPair(StrictModel):
    """Two columns whose nulls fall on the same rows (Jaccard similarity of null masks)."""

    a: str
    b: str
    jaccard: float
    both_null_rate: float


class CrossColumnProfile(StrictModel):
    """Group F for one table: the Spearman matrix feeding the copula, strong pairs and rules."""

    columns: list[str]
    """Columns that entered pairwise statistics, after the `max_pair_columns` cap."""
    truncated: bool
    rows: int
    spearman_columns: list[str]
    spearman: list[list[float]]
    numeric_pairs: list[NumericPair]
    categorical_pairs: list[CategoricalPair]
    mixed_pairs: list[MixedPair]
    conditionals: list[ConditionalShares]
    rules: list[ColumnRule]
    null_pairs: list[NullPair]


class CountSummary(StrictModel):
    """Children per parent row, parents without children included."""

    minimum: int
    maximum: int
    mean: float
    percentiles: list[tuple[float, float]]


class RelationshipShape(StrictModel):
    """Coverage, children-per-parent fits and key skew of one relationship (exact, via SQL)."""

    child_rows: int
    """Child rows whose key is not null."""
    orphan_rate: float
    parent_coverage: float
    """Share of parent rows with at least one child."""
    children: CountSummary
    count_fits: CountFits | None
    lognorm: DistributionFit | None
    """Log-normal fit of the counts of parents that have children."""
    zipf_exponent: float | None
    top_share: float
    """Share of child rows held by the top 1% of parents."""
    gini: float


class TemporalOrder(StrictModel):
    """Child date against its parent's date: share on or after, and the lag in seconds."""

    parent_column: str
    child_column: str
    rows: int
    after_share: float
    lag_seconds: Spread


class ParentLevel(StrictModel):
    """Parents in one category and their children."""

    level: str
    parent_share: float
    child_share: float
    mean_children: float


class ParentCategory(StrictModel):
    """Children per parent within each level of a parent's categorical column."""

    parent_column: str
    levels: list[ParentLevel]


class BehaviourStats(StrictModel):
    """How entities act over time."""

    entities: int
    events: int
    events_per_entity: Spread
    inter_arrival_hours: Spread | None
    burst_1h: Spread
    burst_24h: Spread
    value_to_entity_median: Spread | None


class TransitionMatrix(StrictModel):
    """Probability of each next state given the current one, within each entity's history."""

    column: str
    states: list[str]
    probabilities: list[list[float]]


class Counterparties(StrictModel):
    """Distinct values of another key per entity, such as merchants per account."""

    column: str
    per_entity: Spread


class BehaviourProfile(StrictModel):
    """Behaviour of one entity key over the table's main timestamp (SPEC 8.3 H).

    Computed on the full histories of a seeded sample of entities, so gaps are not stretched
    by row sampling.
    """

    entity_column: str
    time_column: str
    measure_column: str | None
    stats: BehaviourStats
    per_active_day: Spread
    per_active_week: Spread
    inter_arrival_fits: list[DistributionFit]
    """Exponential, gamma and log-normal fits of the gaps in hours, best first."""
    gap_autocorrelation: float | None
    value_autocorrelation: float | None
    lifetime_days: Spread
    counterparties: list[Counterparties]
    transitions: list[TransitionMatrix]
