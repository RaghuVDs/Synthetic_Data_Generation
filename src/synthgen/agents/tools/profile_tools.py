"""Planner tools over stored profiles: one column, one relationship (SPEC 6)."""

from __future__ import annotations

from typing import Any

import numpy as np

from synthgen.agents.tools.context import ToolContext
from synthgen.domain.base import StrictModel
from synthgen.domain.errors import ToolArgumentError
from synthgen.domain.metadata import Sensitivity
from synthgen.profiling.behaviour import spread
from synthgen.profiling.models import ColumnProfile

TOP_VALUES = 20
HISTOGRAM_BINS = 10
TOP_PARENT_SHARE = 0.01
Result = dict[str, Any]  # Any: JSON-compatible tool output


class ColumnArgs(StrictModel):
    """Arguments of get_column_profile."""

    table: str
    column: str


class RelationshipArgs(StrictModel):
    """Arguments of get_relationship_profile."""

    child: str
    parent: str


def get_column_profile(ctx: ToolContext, args: ColumnArgs) -> Result:
    """Full LLM-safe statistics of one column; a sensitive column shows masks and lengths only."""
    column = ctx.column(args.table, args.column)
    out: Result = {
        "table": args.table,
        "column": column.name,
        "type": column.inferred_type,
        "semantic_type": column.semantic_type,
        "sensitivity": column.sensitivity,
        "null_rate": column.null_rate.value,
        "distinct": column.distinct_count.value,
        "unique": column.is_unique.value,
    }
    if column.sensitivity is Sensitivity.SENSITIVE:
        if column.text:
            out["masks"] = [m.model_dump() for m in column.text.masks.value[:5]]
            out["length"] = [column.text.length_min.value, column.text.length_max.value]
        return out
    return out | _numeric(column) | _categorical(column, ctx) | _temporal(column)


def _numeric(column: ColumnProfile) -> Result:
    n = column.numeric
    if n is None:
        return {}
    edges = np.linspace(n.minimum.value, n.maximum.value, HISTOGRAM_BINS + 1)
    fine = n.histogram.value
    centres = (np.array(fine.edges[:-1]) + np.array(fine.edges[1:])) / 2
    coarse, _ = np.histogram(centres, bins=edges, weights=fine.counts)
    return {
        "min": n.minimum.value,
        "max": n.maximum.value,
        "mean": n.mean.value,
        "std": n.std.value,
        "percentiles": {f"p{p:g}": v for p, v in n.percentiles.value},
        "zero_rate": n.zero_rate.value,
        "negative_rate": n.negative_rate.value,
        "integer": n.is_integer.value,
        "decimals": n.decimal_places.value,
        "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in coarse]},
    }


def _categorical(column: ColumnProfile, ctx: ToolContext) -> Result:
    c = column.categorical
    if c is None:
        return {}
    total = sum(v.count for v in c.frequencies.value) + c.other_count.value
    minimum = ctx.settings.internal_min_count if column.sensitivity is Sensitivity.INTERNAL else 0
    shown = [v for v in c.frequencies.value if v.count >= minimum][:TOP_VALUES]
    return {
        "top_values": [{"value": v.value, "share": v.count / total} for v in shown],
        "rare_values": c.rare_count.value,
        "normalized_entropy": c.normalized_entropy.value,
    }


def _temporal(column: ColumnProfile) -> Result:
    t = column.temporal
    if t is None:
        return {}
    return {
        "min": t.minimum.value,
        "max": t.maximum.value,
        "span_days": t.span_days.value,
        "month_counts": t.month_counts.value,
        "weekday_counts_mon_first": t.weekday_counts.value,
        "hour_counts": t.hour_counts.value,
    }


def get_relationship_profile(ctx: ToolContext, args: RelationshipArgs) -> Result:
    """Children per parent, key skew and orphan rate for one relationship."""
    child = ctx.table(args.child)
    ctx.table(args.parent)
    shape = next((s for s in child.relationships if s.parent_table == args.parent), None)
    if shape is None:
        raise ToolArgumentError(f"{args.child} has no foreign key to {args.parent}")
    per_parent = np.repeat(
        np.array([c for c, _ in shape.counts.value], dtype=np.float64),
        [n for _, n in shape.counts.value],
    )
    ordered = np.sort(per_parent)[::-1]
    total = float(ordered.sum())
    top = max(1, round(TOP_PARENT_SHARE * ordered.size))
    rows = child.row_count.value
    return {
        "child": args.child,
        "parent": args.parent,
        "key": shape.child_column,
        "children_per_parent": spread(per_parent).model_dump() | {"mean": float(per_parent.mean())},
        "parents_without_children": float(np.mean(per_parent == 0)),
        "top_1pct_parent_share": float(ordered[:top].sum() / total) if total else 0.0,
        "gini": _gini(ordered[::-1]),
        "orphan_rate": shape.orphan_rows.value / rows if rows else 0.0,
    }


def _gini(ascending: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Gini coefficient of non-negative values sorted ascending."""
    total = float(ascending.sum())
    if not total:
        return 0.0
    n = ascending.size
    ranks = np.arange(1, n + 1)
    return float((2 * np.sum(ranks * ascending) / (n * total)) - (n + 1) / n)
