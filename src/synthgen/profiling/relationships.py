"""Cross-table statistics per relationship (SPEC 8.3 G).

Children per parent, coverage and parent categories come from exact SQL group-bys; the order of
child and parent dates comes from the child table's seeded sample. Sensitive parent and child
columns never take part.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from synthgen.config import ProfilingSettings
from synthgen.domain.metadata import Relationship, Sensitivity, TableMeta
from synthgen.domain.ports import Connector
from synthgen.domain.query import AggExpr, AggFunc
from synthgen.profiling.batching import batched_sample
from synthgen.profiling.behaviour import spread
from synthgen.profiling.categorical import zipf_exponent
from synthgen.profiling.fitting import fit_counts, lognorm_fit
from synthgen.profiling.inference import infer_type, semantic_type, sensitivity_for
from synthgen.profiling.models import (
    ChildrenPerParent,
    ColumnProfile,
    InferredType,
    Measured,
    StatSource,
)
from synthgen.profiling.relation_models import (
    CountSummary,
    ParentCategory,
    ParentLevel,
    RelationshipShape,
    TemporalOrder,
)
from synthgen.profiling.temporal import epoch_seconds

TEMPORAL = (InferredType.DATE, InferredType.DATETIME)
CATEGORICAL = (InferredType.TEXT, InferredType.BOOLEAN)
TOP_PARENT_SHARE = 0.01
PERCENTILES = (50.0, 90.0, 99.0)
OTHER = "other"
NULL_LEVEL = "(null)"


@dataclass(frozen=True)
class RelationshipInputs:
    """The child table's sample and column profiles, and the metadata of every parent."""

    connector: Connector
    sample: pa.Table
    child_columns: Sequence[ColumnProfile]
    parents: Mapping[str, TableMeta]
    settings: ProfilingSettings
    seed: int


@dataclass(frozen=True)
class KeyCounts:
    """Child rows per parent key, and every parent key."""

    children: dict[object, int]
    parents: list[object]


def _key_counts(connector: Connector, rel: Relationship) -> KeyCounts:
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
    return KeyCounts(children, parents)


def relationship_profile(inputs: RelationshipInputs, rel: Relationship) -> ChildrenPerParent:
    """Children per parent with fits and skew, date order and parent-category summaries."""
    keys = _key_counts(inputs.connector, rel)
    known = set(keys.parents)
    orphans = sum(c for key, c in keys.children.items() if key not in known)
    per_parent = np.array([keys.children.get(k, 0) for k in keys.parents], dtype=np.int64)
    dates, categories = _parent_columns(inputs, inputs.parents[rel.parent_table])
    return ChildrenPerParent(
        child_column=rel.child_column,
        parent_table=rel.parent_table,
        parent_column=rel.parent_column,
        counts=Measured[list[tuple[int, int]]](
            value=sorted(Counter(per_parent.tolist()).items()), source=StatSource.SQL
        ),
        orphan_rows=Measured[int](value=orphans, source=StatSource.SQL),
        shape=Measured[RelationshipShape](
            value=relationship_shape(per_parent, sum(keys.children.values()), orphans),
            source=StatSource.SQL,
        ),
        temporal_orders=Measured[list[TemporalOrder]](
            value=_temporal_orders(inputs, rel, dates), source=StatSource.SAMPLE
        ),
        by_parent_category=Measured[list[ParentCategory]](
            value=[_parent_category(inputs, rel, keys, c) for c in categories],
            source=StatSource.SQL,
        ),
    )


def gini(counts: NDArray[np.int64]) -> float:
    """Gini coefficient of child counts: 0 for equal counts, near 1 when one parent has all."""
    total = int(counts.sum())
    if not total:
        return 0.0
    ordered = np.sort(counts).astype(np.float64)
    n = ordered.size
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(2 * np.sum(ranks * ordered) / (n * total) - (n + 1) / n)


def relationship_shape(
    per_parent: NDArray[np.int64], child_rows: int, orphans: int
) -> RelationshipShape:
    """Coverage, summary, fits and skew of children per parent (parents without children count)."""
    counts = per_parent.astype(np.float64)
    if not counts.size:
        counts = np.zeros(1)
    positive = counts[counts > 0]
    ordered = np.sort(per_parent)[::-1]
    top = max(1, math.ceil(TOP_PARENT_SHARE * ordered.size))
    total = int(per_parent.sum())
    return RelationshipShape(
        child_rows=child_rows,
        orphan_rate=orphans / child_rows if child_rows else 0.0,
        parent_coverage=float(np.mean(counts > 0)),
        children=CountSummary(
            minimum=int(np.min(counts)),
            maximum=int(np.max(counts)),
            mean=float(counts.mean()),
            percentiles=[(p, float(np.percentile(counts, p))) for p in PERCENTILES],
        ),
        count_fits=fit_counts(counts),
        lognorm=lognorm_fit(positive) if positive.size else None,
        zipf_exponent=zipf_exponent([int(c) for c in ordered if c > 0]),
        top_share=int(ordered[:top].sum()) / total if total else 0.0,
        gini=gini(per_parent),
    )


def _parent_columns(inputs: RelationshipInputs, meta: TableMeta) -> tuple[list[str], list[str]]:
    """Non-sensitive date columns and low-cardinality categorical columns of a parent table."""
    settings = inputs.settings
    columns = [c for c in meta.columns if not c.primary_key]
    if not columns or not settings.max_parent_columns:
        return [], []
    sample = batched_sample(
        inputs.connector,
        meta.name,
        [c.name for c in columns],
        rows=settings.parent_sample_rows,
        seed=inputs.seed,
        batch=settings.max_select_columns,
    )
    dates: list[str] = []
    categories: list[str] = []
    for column in columns:
        values = [v for v in sample.column(column.name).to_pylist() if v is not None]
        inferred = infer_type(values)
        semantic = semantic_type(values, inferred)
        if sensitivity_for(column.sensitivity, semantic) is Sensitivity.SENSITIVE:
            continue
        if inferred in TEMPORAL:
            dates.append(column.name)
        elif inferred in CATEGORICAL and 1 < len(set(values)) <= settings.max_pair_levels:
            categories.append(column.name)
    return dates[: settings.max_parent_columns], categories[: settings.max_parent_columns]


def _temporal_orders(
    inputs: RelationshipInputs, rel: Relationship, parent_dates: list[str]
) -> list[TemporalOrder]:
    child_dates = [
        c.name
        for c in inputs.child_columns
        if c.inferred_type in TEMPORAL and c.sensitivity is not Sensitivity.SENSITIVE
    ][: inputs.settings.max_parent_columns]
    keys = inputs.sample.column(rel.child_column).to_pylist()
    found: list[TemporalOrder] = []
    for parent_date in parent_dates:
        earliest = AggExpr(func=AggFunc.MIN, column=parent_date, alias="d")
        lookup = {
            r[rel.parent_column]: r["d"]
            for r in inputs.connector.aggregate(
                rel.parent_table, [earliest], group_by=[rel.parent_column]
            )
        }
        parent_seconds = epoch_seconds([lookup.get(k) for k in keys])
        for child_date in child_dates:
            lag = epoch_seconds(inputs.sample.column(child_date).to_pylist()) - parent_seconds
            lag = lag[~np.isnan(lag)]
            if lag.size:
                found.append(
                    TemporalOrder(
                        parent_column=parent_date,
                        child_column=child_date,
                        rows=int(lag.size),
                        after_share=float(np.mean(lag >= 0)),
                        lag_seconds=spread(lag),
                    )
                )
    return found


def _parent_category(
    inputs: RelationshipInputs, rel: Relationship, keys: KeyCounts, column: str
) -> ParentCategory:
    """Parents and children per level; small levels of internal columns fold into `other`."""
    n = AggExpr(func=AggFunc.COUNT, column=None, alias="n")
    rows = inputs.connector.aggregate(rel.parent_table, [n], group_by=[rel.parent_column, column])
    parents: Counter[str] = Counter()
    children: Counter[str] = Counter()
    for row in rows:
        level = NULL_LEVEL if row[column] is None else str(row[column])
        parents[level] += int(row["n"])
        children[level] += keys.children.get(row[rel.parent_column], 0) * int(row["n"])
    meta = inputs.parents[rel.parent_table].column(column)
    folds = meta is not None and meta.sensitivity is Sensitivity.INTERNAL
    settings = inputs.settings
    kept = [
        level
        for level, count in parents.most_common(settings.max_conditional_levels)
        if not folds or count >= settings.internal_min_count
    ]
    total_parents, total_children = sum(parents.values()), sum(children.values())
    levels = [(level, parents[level], children[level]) for level in kept]
    rest_parents = total_parents - sum(p for _, p, _ in levels)
    if rest_parents:
        levels.append((OTHER, rest_parents, total_children - sum(c for _, _, c in levels)))
    return ParentCategory(
        parent_column=column,
        levels=[
            ParentLevel(
                level=level,
                parent_share=p / total_parents,
                child_share=c / total_children if total_children else 0.0,
                mean_children=c / p,
            )
            for level, p, c in levels
        ],
    )
