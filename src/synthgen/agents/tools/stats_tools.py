"""Planner tools that compute on demand from SQL or the seeded sample (SPEC 6)."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pyarrow as pa

from synthgen.agents.tools.context import ToolContext
from synthgen.domain.base import StrictModel
from synthgen.domain.errors import ToolArgumentError
from synthgen.domain.metadata import Sensitivity
from synthgen.domain.query import AggExpr, AggFunc
from synthgen.profiling.behaviour import behaviour_stats
from synthgen.profiling.models import ColumnProfile, InferredType
from synthgen.profiling.stratified import class_separation
from synthgen.profiling.temporal import parse_times

TOP_GROUPS = 20
TOP_SEPARATING = 10
NUMERIC = (InferredType.INTEGER, InferredType.FLOAT)
TEMPORAL = (InferredType.DATE, InferredType.DATETIME)
SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
Result = dict[str, Any]  # Any: JSON-compatible tool output


class ConditionalArgs(StrictModel):
    """Arguments of get_conditional_stats."""

    table: str
    target: str
    by: str


class BehaviourArgs(StrictModel):
    """Arguments of get_behaviour_profile."""

    table: str
    entity_key: str
    time_column: str
    measure: str


class SeparationArgs(StrictModel):
    """Arguments of get_class_separation."""

    table: str
    label: str


def get_conditional_stats(ctx: ToolContext, args: ConditionalArgs) -> Result:
    """The target summarised per category of `by`: the 20 largest groups, exact via SQL."""
    target = ctx.column(args.table, args.target, values=True)
    by = ctx.column(args.table, args.by, values=True)
    exprs = [AggExpr(func=AggFunc.COUNT, column=None, alias="rows")]
    if target.inferred_type in NUMERIC:
        exprs += [
            AggExpr(func=f, column=args.target, alias=str(f))
            for f in (AggFunc.AVG, AggFunc.MIN, AggFunc.MAX)
        ]
        groups = ctx.source.connector.aggregate(args.table, exprs, group_by=[args.by])
    else:
        groups = ctx.source.connector.aggregate(args.table, exprs, group_by=[args.by, args.target])
    minimum = ctx.settings.internal_min_count if by.sensitivity is Sensitivity.INTERNAL else 1
    kept = [g for g in groups if g["rows"] >= minimum]
    kept.sort(key=lambda g: -g["rows"])
    return {
        "table": args.table,
        "target": args.target,
        "by": args.by,
        "groups": kept[:TOP_GROUPS],
        "groups_not_shown": len(groups) - min(len(kept), TOP_GROUPS),
    }


def get_behaviour_profile(ctx: ToolContext, args: BehaviourArgs) -> Result:
    """Velocity, inter-arrival and bursts per entity, from the seeded sample."""
    ctx.column(args.table, args.entity_key)
    time = ctx.column(args.table, args.time_column, values=True)
    measure = ctx.column(args.table, args.measure, values=True)
    if time.inferred_type not in (InferredType.DATE, InferredType.DATETIME):
        raise ToolArgumentError(f"{args.table}.{args.time_column} is not a date or time column")
    if measure.inferred_type not in NUMERIC:
        raise ToolArgumentError(f"{args.table}.{args.measure} is not numeric")
    sample = ctx.source.connector.sample(
        args.table,
        [args.entity_key, args.time_column, args.measure],
        ctx.settings.sample_size,
        ctx.settings.sample_seed,
    )
    rows = [r for r in sample.to_pylist() if None not in r.values()]
    _, entity = np.unique([str(r[args.entity_key]) for r in rows], return_inverse=True)
    seconds = parse_times([str(r[args.time_column]) for r in rows]).astype(np.int64)
    values = np.array([float(r[args.measure]) for r in rows])
    stats = behaviour_stats(entity.astype(np.int64), seconds, values)
    return {"table": args.table, "entity_key": args.entity_key, **stats.model_dump()}


def get_class_separation(ctx: ToolContext, args: SeparationArgs) -> Result:
    """Columns ranked by how differently they behave in the rarer label class."""
    label = ctx.column(args.table, args.label, values=True)
    profile = ctx.table(args.table)
    keys = {c.name for c in ctx.source.metadata.table(args.table).columns if c.primary_key}
    keys |= {r.child_column for r in profile.relationships}
    usable = [
        c
        for c in profile.columns
        if c.name not in keys | {label.name}
        and c.sensitivity is not Sensitivity.SENSITIVE
        and (c.inferred_type in NUMERIC or c.inferred_type in TEMPORAL or c.categorical is not None)
    ]
    sample = ctx.source.connector.sample(
        args.table,
        [label.name, *(c.name for c in usable)],
        ctx.settings.sample_size,
        ctx.settings.sample_seed,
    )
    labels = sample.column(label.name).to_pylist()
    counts = Counter(v for v in labels if v is not None)
    if len(counts) < 2:  # noqa: PLR2004 - two classes are needed to compare
        raise ToolArgumentError(f"{args.table}.{args.label} has fewer than two classes")
    positive_value = min(counts.most_common(2), key=lambda vc: vc[1])[0]
    positive = np.array([v == positive_value for v in labels])
    numeric, categorical = _features(usable, sample)
    ranked = class_separation(positive, numeric, categorical)
    return {
        "table": args.table,
        "label": args.label,
        "positive_class": positive_value,
        "positive_share": float(positive.mean()),
        "separation": [s.model_dump() for s in ranked[:TOP_SEPARATING]],
    }


def _features(
    usable: list[ColumnProfile], sample: pa.Table
) -> tuple[dict[str, np.ndarray[Any, np.dtype[np.float64]]], dict[str, list[Any]]]:
    """Numeric features (numbers, hours, days) and categorical value lists from the sample."""
    numeric = {
        c.name: np.array(
            [np.nan if v is None else float(v) for v in sample.column(c.name).to_pylist()]
        )
        for c in usable
        if c.inferred_type in NUMERIC
    }
    numeric |= {
        _temporal_label(c.inferred_type, c.name): _temporal_feature(
            c.inferred_type, sample.column(c.name).to_pylist()
        )
        for c in usable
        if c.inferred_type in TEMPORAL
    }
    categorical = {
        c.name: sample.column(c.name).to_pylist()
        for c in usable
        if c.inferred_type not in NUMERIC and c.inferred_type not in TEMPORAL
    }
    return numeric, categorical


def _temporal_label(kind: InferredType, name: str) -> str:
    return f"{name} (hour)" if kind is InferredType.DATETIME else f"{name} (day)"


def _temporal_feature(
    kind: InferredType, values: list[Any]
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Hour of day for timestamps, day number for dates; NaN where missing."""
    present = [v for v in values if v is not None]
    seconds = iter(parse_times([str(v) for v in present]).astype(np.int64).tolist())
    out = []
    for v in values:
        if v is None:
            out.append(np.nan)
            continue
        s = next(seconds)
        out.append(
            (s % SECONDS_PER_DAY) // SECONDS_PER_HOUR
            if kind is InferredType.DATETIME
            else s // SECONDS_PER_DAY
        )
    return np.array(out, dtype=np.float64)
