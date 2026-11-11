"""LlmSafeProfile: the only view of a profile an agent may receive (SPEC 8.3, CLAUDE.md).

Sensitive columns show a mask and a length range, never a value. Internal columns show only
values held by at least `internal_min_count` rows. Profiles already fold rarer values; the card
checks again because it is the boundary the LLM sees.
"""

from __future__ import annotations

from collections.abc import Sequence

from synthgen.config import ProfilingSettings
from synthgen.domain.base import StrictModel
from synthgen.domain.metadata import Relationship, Sensitivity, TableMeta
from synthgen.profiling.models import ColumnProfile, InferredType, TableProfile
from synthgen.profiling.table_stats import identifier_columns

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
    best_fit: str | None
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
        columns=[
            _card_column(c, meta, settings, keys=identifier_columns(meta, relationships))
            for c in profile.columns
        ],
    )


def _card_column(
    column: ColumnProfile, meta: TableMeta, settings: ProfilingSettings, *, keys: set[str]
) -> CardColumn:
    """One column's card line; keys get no distribution fit, which would mean nothing for them."""
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
        best_fit=None
        if sensitive or column.numeric is None or column.name in keys
        else column.numeric.best_fit,
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


WHOLE_NUMBER_FROM = 10_000
SMALL_SHARE = 0.01


def _number(value: float) -> str:
    """Whole numbers from ten thousand up, otherwise four significant digits."""
    return f"{value:,.0f}" if abs(value) >= WHOLE_NUMBER_FROM else f"{value:,.4g}"


def format_share(share: float) -> str:
    """A percentage; shares below 1% keep two decimals so rare values do not read as 0%."""
    return f"{share:.2%}" if 0 < share < SMALL_SHARE else f"{share:.0%}"


def _column_line(column: CardColumn, role: str | None) -> str:
    parts = [f"- {column.name} {column.type}"]
    if role:
        parts[0] += f" [{role}]"
    if column.semantic_type and column.semantic_type != column.type:
        parts.append(column.semantic_type)
    if column.sensitivity is not Sensitivity.PUBLIC:
        parts.append(str(column.sensitivity))
    if column.null_rate:
        parts.append(f"nulls {column.null_rate:.1%}")
    parts.append(f"{column.distinct_count:,} distinct")
    if column.mask is not None:
        low, high = column.length_range or (0, 0)
        parts.append(f"mask {column.mask} length {low}-{high}")
    elif column.five_number is not None:
        lo, p25, p50, p75, hi = column.five_number
        named = zip(("min", "p25", "p50", "p75", "max"), (lo, p25, p50, p75, hi), strict=True)
        parts.append(" ".join(f"{label} {_number(v)}" for label, v in named))
        if column.best_fit:
            parts.append(f"fit {column.best_fit}")
    elif column.date_range is not None:
        parts.append(f"{column.date_range[0]} .. {column.date_range[1]}")
    elif column.top_values:
        parts.append(
            "top " + ", ".join(f"{v.value} {format_share(v.share)}" for v in column.top_values)
        )
    line = " | ".join(parts)
    return f"{line} : {column.description}" if column.description else line


def card_text(card: LlmSafeProfile, roles: dict[str, str] | None = None) -> str:
    """A compact rendering for prompts: one line per column, only fields that carry information."""
    head = f"table {card.table} ({card.row_count:,} rows)"
    lines = [f"{head}: {card.description}" if card.description else head]
    if card.relationships:
        lines.append("relationships: " + "; ".join(card.relationships))
    lines += [_column_line(c, (roles or {}).get(c.name)) for c in card.columns]
    return "\n".join(lines)


KEY_ROLES = frozenset(
    {"identifier", "entity_key", "foreign_key", "event_time", "label", "status", "category"}
)
"""Roles whose columns keep full card lines in a wide-table summary."""


def card_slice(card: LlmSafeProfile, names: set[str]) -> LlmSafeProfile:
    """The card with only the named columns."""
    return card.model_copy(update={"columns": [c for c in card.columns if c.name in names]})


def wide_card_text(card: LlmSafeProfile, roles: dict[str, str], keep: set[str], limit: int) -> str:
    """A wide table: full lines for kept and key-role columns (up to `limit`), names for the rest.

    Statistics of any listed column are one `get_column_profile` call away for the planner.
    """
    full = [c for c in card.columns if c.name in keep or roles.get(c.name) in KEY_ROLES][:limit]
    shown = {c.name for c in full}
    others: dict[str, list[str]] = {}
    for column in card.columns:
        if column.name not in shown:
            others.setdefault(roles.get(column.name, "unmapped"), []).append(column.name)
    lines = [card_text(card_slice(card, shown), roles)]
    lines += [
        f"{role} columns ({len(names)}): {', '.join(names)}"
        for role, names in sorted(others.items())
    ]
    return "\n".join(lines)
