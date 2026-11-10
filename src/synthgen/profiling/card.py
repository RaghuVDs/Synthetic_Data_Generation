"""LlmSafeProfile: the only view of a profile an agent may receive (SPEC 8.3, CLAUDE.md).

Sensitive columns show a mask and a length range, never a value. Internal columns show only
values held by at least `internal_min_count` rows.
"""

from __future__ import annotations

from collections.abc import Sequence

from synthgen.config import ProfilingSettings
from synthgen.domain.base import StrictModel
from synthgen.domain.metadata import Relationship, Sensitivity, TableMeta
from synthgen.profiling.models import ColumnProfile, InferredType, TableProfile

RATE_DIGITS = 4
FIVE_NUMBER_PERCENTILES = (25.0, 50.0, 75.0)


class TopValue(StrictModel):
    """A frequent value and its share of non-null rows."""

    value: str | int | float
    share: float


class CardColumn(StrictModel):
    """One column as an agent may see it."""

    name: str
    description: str
    type: InferredType
    semantic_type: str | None
    sensitivity: Sensitivity
    null_rate: float
    distinct_count: int
    five_number: tuple[float, float, float, float, float] | None
    date_range: tuple[str, str] | None
    top_values: list[TopValue] | None
    mask: str | None
    length_range: tuple[int, int] | None


class LlmSafeProfile(StrictModel):
    """A profile card: about 300 to 600 tokens per table (SPEC 4.4)."""

    table: str
    description: str
    row_count: int
    relationships: list[str]
    columns: list[CardColumn]


def build_card(
    profile: TableProfile,
    meta: TableMeta,
    relationships: Sequence[Relationship],
    settings: ProfilingSettings,
) -> LlmSafeProfile:
    """Reduce a full profile to its LLM-safe card."""
    return LlmSafeProfile(
        table=profile.table,
        description=meta.description,
        row_count=profile.row_count.value,
        relationships=[
            r.label() for r in relationships if profile.table in (r.child_table, r.parent_table)
        ],
        columns=[_card_column(c, meta, settings) for c in profile.columns],
    )


def _card_column(column: ColumnProfile, meta: TableMeta, settings: ProfilingSettings) -> CardColumn:
    sensitive = column.sensitivity is Sensitivity.SENSITIVE
    col_meta = meta.column(column.name)
    shape = column.text
    return CardColumn(
        name=column.name,
        description=col_meta.description if col_meta else "",
        type=column.inferred_type,
        semantic_type=column.semantic_type,
        sensitivity=column.sensitivity,
        null_rate=round(column.null_rate.value, RATE_DIGITS),
        distinct_count=column.distinct_count.value,
        five_number=None if sensitive else _five_number(column),
        date_range=None
        if sensitive or column.temporal is None
        else (column.temporal.minimum.value, column.temporal.maximum.value),
        top_values=None if sensitive else _top_values(column, settings),
        mask=shape.masks.value[0].mask if sensitive and shape and shape.masks.value else None,
        length_range=(shape.length_min.value, shape.length_max.value)
        if sensitive and shape
        else None,
    )


def _five_number(column: ColumnProfile) -> tuple[float, float, float, float, float] | None:
    numeric = column.numeric
    if numeric is None:
        return None
    at = dict(numeric.percentiles.value)
    p25, p50, p75 = (round(at[p], RATE_DIGITS) for p in FIVE_NUMBER_PERCENTILES)
    return (numeric.minimum.value, p25, p50, p75, numeric.maximum.value)


def _top_values(column: ColumnProfile, settings: ProfilingSettings) -> list[TopValue] | None:
    categorical = column.categorical
    if categorical is None:
        return None
    total = sum(v.count for v in categorical.frequencies.value) + categorical.other_count.value
    shown = [
        v
        for v in categorical.frequencies.value
        if column.sensitivity is not Sensitivity.INTERNAL or v.count >= settings.internal_min_count
    ]
    return [
        TopValue(value=v.value, share=round(v.count / total, RATE_DIGITS))
        for v in shown[: settings.card_top_values]
    ]
