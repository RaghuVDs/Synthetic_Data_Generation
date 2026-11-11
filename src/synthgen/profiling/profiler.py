"""Profile one table: exact aggregates through SQL, shape statistics from a seeded sample.

Sensitive columns keep lengths and masks only (SPEC 8.3); no value of theirs enters the profile.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa

from synthgen.config import ProfilingSettings
from synthgen.domain.metadata import ColumnMeta, Relationship, Sensitivity, TableMeta
from synthgen.domain.ports import Connector
from synthgen.domain.query import AggExpr, AggFunc
from synthgen.profiling.batching import batched_aggregate, batched_sample, sample_rows
from synthgen.profiling.categorical import categorical_profile, sample_counts, text_shape
from synthgen.profiling.inference import Value, infer_type, semantic_type, sensitivity_for
from synthgen.profiling.models import (
    CategoricalProfile,
    ChildrenPerParent,
    ColumnProfile,
    InferredType,
    Measured,
    StatSource,
    TableProfile,
)
from synthgen.profiling.numeric import SqlMoments, numeric_profile
from synthgen.profiling.temporal import temporal_profile

NUMERIC = (InferredType.INTEGER, InferredType.FLOAT)
TEMPORAL = (InferredType.DATE, InferredType.DATETIME)
CATEGORICAL = (InferredType.TEXT, InferredType.BOOLEAN)


@dataclass(frozen=True)
class ColumnFacts:
    """One column's metadata, sampled non-null values and exact aggregates."""

    meta: ColumnMeta
    inferred: InferredType
    values: list[Value]
    agg: dict[str, Any]  # Any: SQLite aggregate cells (int, float, str or None)


def schema_hash(meta: TableMeta) -> str:
    """Hash of column names and declared types; a change invalidates cached profiles."""
    shape = [(c.name, c.declared_type, c.nullable, c.primary_key, c.unique) for c in meta.columns]
    return hashlib.sha256(json.dumps([meta.name, shape]).encode()).hexdigest()[:16]


def profile_table(  # noqa: PLR0913 - the full context of one profiling run
    source: str,
    connector: Connector,
    meta: TableMeta,
    relationships: Sequence[Relationship],
    *,
    settings: ProfilingSettings,
    seed: int,
) -> TableProfile:
    """Compute groups A to E (core) and children-per-parent shapes for one table."""
    rows = connector.row_count(meta.name)
    names = [c.name for c in meta.columns]
    batch = settings.max_select_columns
    sample = batched_sample(
        connector,
        meta.name,
        names,
        rows=sample_rows(len(names), settings),
        seed=seed,
        batch=batch,
    )
    facts = _facts(connector, meta, sample, batch)
    numeric_columns = sum(f.inferred in NUMERIC for f in facts)
    ctx = TableContext(
        connector, meta.name, rows, settings, numeric_columns <= settings.max_fit_columns
    )
    columns = [_column(ctx, f) for f in facts]
    return TableProfile(
        source=source,
        table=meta.name,
        schema_hash=schema_hash(meta),
        row_count=Measured[int](value=rows, source=StatSource.SQL),
        primary_key_unique=_primary_key_unique(meta, columns, rows),
        duplicate_row_rate=Measured[float](
            value=_duplicate_rate(sample, [c.name for c in meta.columns if not c.primary_key]),
            source=StatSource.SAMPLE,
        ),
        completeness=Measured[float](
            value=1.0 - float(np.mean([c.null_rate.value for c in columns])),
            source=StatSource.DERIVED,
        ),
        sample_rows=sample.num_rows,
        columns=columns,
        relationships=[
            _children_per_parent(connector, r)
            for r in relationships
            if r.child_table == meta.name and not r.is_self_reference
        ],
    )


def _facts(
    connector: Connector, meta: TableMeta, sample: pa.Table, batch: int
) -> list[ColumnFacts]:
    """Sample values and inferred types, then batched SQL passes for every column's aggregates."""
    values = {
        c.name: [v for v in sample.column(c.name).to_pylist() if v is not None]
        for c in meta.columns
    }
    types = {name: infer_type(vals) for name, vals in values.items()}
    exprs: list[AggExpr] = []
    for i, col in enumerate(meta.columns):
        funcs = [AggFunc.COUNT, AggFunc.COUNT_DISTINCT, AggFunc.MIN, AggFunc.MAX]
        if types[col.name] in NUMERIC:
            funcs += [AggFunc.SUM, AggFunc.SUM_SQ]
        exprs += [AggExpr(func=f, column=col.name, alias=f"c{i}_{f}") for f in funcs]
    row = batched_aggregate(connector, meta.name, exprs, batch)
    return [
        ColumnFacts(
            meta=col,
            inferred=types[col.name],
            values=values[col.name],
            agg={k.split("_", 1)[1]: v for k, v in row.items() if k.startswith(f"c{i}_")},
        )
        for i, col in enumerate(meta.columns)
    ]


@dataclass(frozen=True)
class TableContext:
    """What every column of one table is profiled with."""

    connector: Connector
    table: str
    rows: int
    settings: ProfilingSettings
    full_fit: bool


def _column(ctx: TableContext, facts: ColumnFacts) -> ColumnProfile:
    connector, table, rows, settings = ctx.connector, ctx.table, ctx.rows, ctx.settings
    count, distinct = int(facts.agg["count"]), int(facts.agg["count_distinct"])
    semantic = semantic_type(facts.values, facts.inferred)
    sensitivity = sensitivity_for(facts.meta.sensitivity, semantic)
    sensitive = sensitivity is Sensitivity.SENSITIVE
    texts = [str(v) for v in facts.values]
    top = None if sensitive or not facts.values else Counter(facts.values).most_common(1)[0]
    sql, derived, sampled = StatSource.SQL, StatSource.DERIVED, StatSource.SAMPLE
    usable = not sensitive and bool(facts.values)
    return ColumnProfile(
        name=facts.meta.name,
        declared_type=facts.meta.declared_type,
        inferred_type=facts.inferred,
        sensitivity=sensitivity,
        semantic_type=semantic,
        null_count=Measured[int](value=rows - count, source=sql),
        null_rate=Measured[float](value=(rows - count) / rows if rows else 0.0, source=derived),
        distinct_count=Measured[int](value=distinct, source=sql),
        cardinality_ratio=Measured[float](value=distinct / count if count else 0.0, source=derived),
        is_unique=Measured[bool](value=count > 0 and distinct == count, source=derived),
        is_constant=Measured[bool](value=distinct == 1, source=derived),
        top_value=None if top is None else Measured(value=_scalar(top[0]), source=sampled),
        top_share=None
        if top is None
        else Measured[float](value=top[1] / len(facts.values), source=sampled),
        empty_rate=Measured[float](
            value=sum(not t.strip() for t in texts) / len(texts) if texts else 0.0, source=sampled
        ),
        numeric=numeric_profile(
            _moments(facts.agg),
            np.asarray(facts.values, dtype=np.float64),
            settings,
            distinct=distinct,
            full_fit=ctx.full_fit,
        )
        if usable and facts.inferred in NUMERIC
        else None,
        categorical=_categorical(
            connector, table, facts, count=count, distinct=distinct, settings=settings
        )
        if usable and facts.inferred in CATEGORICAL
        else None,
        text=text_shape(texts) if sensitive or facts.inferred is InferredType.TEXT else None,
        temporal=temporal_profile(str(facts.agg["min"]), str(facts.agg["max"]), texts)
        if usable and facts.inferred in TEMPORAL
        else None,
    )


def _moments(agg: dict[str, Any]) -> SqlMoments:  # Any: SQLite aggregate cells
    return SqlMoments(
        count=int(agg["count"]),
        minimum=float(agg["min"]),
        maximum=float(agg["max"]),
        total=float(agg["sum"]),
        sum_squares=float(agg["sum_sq"]),
    )


def _categorical(  # noqa: PLR0913 - column context plus the two exact counts
    connector: Connector,
    table: str,
    facts: ColumnFacts,
    *,
    count: int,
    distinct: int,
    settings: ProfilingSettings,
) -> CategoricalProfile:
    """Exact SQL frequencies for up to `max_full_frequencies` values, else sampled top values."""
    if distinct <= settings.max_full_frequencies:
        grouped = connector.aggregate(
            table, [AggExpr(func=AggFunc.COUNT, column=None, alias="n")], group_by=[facts.meta.name]
        )
        counts = [
            (_scalar(r[facts.meta.name]), int(r["n"]))
            for r in grouped
            if r[facts.meta.name] is not None
        ]
        return categorical_profile(counts, count, StatSource.SQL, settings)
    sampled = [_scalar(v) for v in facts.values]
    return categorical_profile(sample_counts(sampled), len(sampled), StatSource.SAMPLE, settings)


def _scalar(value: Value) -> str | int | float:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, bytes):
        return value.hex()
    return value


def _primary_key_unique(
    meta: TableMeta, columns: Sequence[ColumnProfile], rows: int
) -> Measured[bool] | None:
    keys = [c for c in columns if c.name in meta.primary_key]
    if len(keys) != 1:
        return None
    return Measured[bool](value=keys[0].distinct_count.value == rows, source=StatSource.DERIVED)


def _duplicate_rate(sample: pa.Table, columns: list[str]) -> float:
    if not columns or sample.num_rows == 0:
        return 0.0
    rows = list(zip(*(sample.column(c).to_pylist() for c in columns), strict=True))
    return 1.0 - len(set(rows)) / len(rows)


def _children_per_parent(connector: Connector, rel: Relationship) -> ChildrenPerParent:
    """Children per parent row (including parents with none) and orphan child rows, via SQL."""
    n = AggExpr(func=AggFunc.COUNT, column=None, alias="n")
    children = {
        r[rel.child_column]: int(r["n"])
        for r in connector.aggregate(rel.child_table, [n], group_by=[rel.child_column])
        if r[rel.child_column] is not None
    }
    parents = [
        r[rel.parent_column]
        for r in connector.aggregate(rel.parent_table, [n], group_by=[rel.parent_column])
    ]
    known = set(parents)
    per_parent = Counter(children.get(key, 0) for key in parents)
    return ChildrenPerParent(
        child_column=rel.child_column,
        parent_table=rel.parent_table,
        parent_column=rel.parent_column,
        counts=Measured[list[tuple[int, int]]](
            value=sorted(per_parent.items()), source=StatSource.SQL
        ),
        orphan_rows=Measured[int](
            value=sum(c for key, c in children.items() if key not in known), source=StatSource.SQL
        ),
    )
