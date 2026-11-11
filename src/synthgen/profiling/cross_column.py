"""Cross-column profile of one table (SPEC 8.3 F) from the seeded sample.

Wide tables would need millions of pairs, so at most `max_pair_columns` columns take part:
dates and booleans first (they carry ordering rules and labels), then the most variable columns.
Sensitive columns, keys, identifiers and constants never take part, except in null
co-occurrence, which looks only at null masks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray

from synthgen.config import ProfilingSettings
from synthgen.domain.metadata import Sensitivity
from synthgen.profiling.associations import (
    categorical_pair,
    conditional_shares,
    correlation_ratio,
    cramers_v,
    mixed_pair,
    numeric_pair,
)
from synthgen.profiling.models import ColumnProfile, InferredType
from synthgen.profiling.relation_models import (
    CategoricalPair,
    ConditionalShares,
    CrossColumnProfile,
    MixedPair,
)
from synthgen.profiling.rules import (
    arithmetic_rules,
    functional_dependencies,
    null_pairs,
    ordering_rules,
)

NUMERIC = (InferredType.INTEGER, InferredType.FLOAT)
TEMPORAL = (InferredType.DATE, InferredType.DATETIME)
CATEGORICAL = (InferredType.TEXT, InferredType.BOOLEAN)
IDENTIFIER_TYPES = (InferredType.INTEGER, InferredType.TEXT)
FIRST = (InferredType.BOOLEAN, *TEMPORAL)


def _score(column: ColumnProfile) -> float:
    """Variability in [0, 1]: coefficient of variation for numbers, normalised entropy otherwise."""
    if column.numeric is not None:
        mean, std = column.numeric.mean.value, column.numeric.std.value
        return min(1.0, abs(std / mean)) if mean else 1.0
    if column.categorical is not None:
        return column.categorical.normalized_entropy.value
    return 1.0


def _eligible(column: ColumnProfile, keys: set[str], max_levels: int) -> bool:
    if column.name in keys or column.sensitivity is Sensitivity.SENSITIVE:
        return False
    if column.is_constant.value or column.inferred_type is InferredType.EMPTY:
        return False
    if column.is_unique.value and column.inferred_type in IDENTIFIER_TYPES:
        return False
    return column.inferred_type not in CATEGORICAL or column.distinct_count.value <= max_levels


def select_columns(
    columns: Sequence[ColumnProfile], keys: set[str], settings: ProfilingSettings
) -> tuple[list[ColumnProfile], bool]:
    """Columns for pairwise statistics in table order, and whether the cap removed any."""
    eligible = [c for c in columns if _eligible(c, keys, settings.max_pair_levels)]
    ranked = sorted(
        range(len(eligible)),
        key=lambda i: (eligible[i].inferred_type not in FIRST, -_score(eligible[i]), i),
    )
    keep = sorted(ranked[: settings.max_pair_columns])
    return [eligible[i] for i in keep], len(eligible) > settings.max_pair_columns


@dataclass(frozen=True)
class Arrays:
    """Aligned sample arrays of the selected columns, by kind."""

    numeric: dict[str, NDArray[np.float64]]
    temporal: dict[str, NDArray[np.float64]]
    codes: dict[str, NDArray[np.int64]]
    """Every selected column factorised; nulls are code -1."""
    labels: dict[str, list[str]]


def _arrays(sample: pa.Table, columns: Sequence[ColumnProfile]) -> Arrays:
    numeric: dict[str, NDArray[np.float64]] = {}
    temporal: dict[str, NDArray[np.float64]] = {}
    codes: dict[str, NDArray[np.int64]] = {}
    labels: dict[str, list[str]] = {}
    for column in columns:
        series = pd.Series(sample.column(column.name).to_pylist(), dtype=object)
        if column.inferred_type in NUMERIC:
            numeric[column.name] = pd.to_numeric(series, errors="coerce").to_numpy(np.float64)
        if column.inferred_type in TEMPORAL:
            stamps = pd.to_datetime(series, errors="coerce", format="ISO8601")
            seconds = stamps.to_numpy("datetime64[s]").astype(np.int64).astype(np.float64)
            temporal[column.name] = np.where(stamps.isna().to_numpy(), np.nan, seconds)
        found, uniques = pd.factorize(series.where(series.isna(), series.astype(str)), sort=True)
        codes[column.name] = found.astype(np.int64)
        labels[column.name] = [str(u) for u in uniques]
    return Arrays(numeric, temporal, codes, labels)


def _null_masks(
    sample: pa.Table, columns: Sequence[ColumnProfile], limit: int
) -> dict[str, NDArray[np.bool_]]:
    """Null masks of the first `limit` sometimes-null columns; masks hold no values."""
    partial = [c.name for c in columns if 0 < c.null_rate.value < 1][:limit]
    return {name: sample.column(name).is_null().to_numpy(zero_copy_only=False) for name in partial}


def _spearman(numeric: dict[str, NDArray[np.float64]]) -> list[list[float]]:
    if not numeric:
        return []
    matrix = pd.DataFrame(numeric).corr(method="spearman").fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(matrix, 1.0)
    return [[float(v) for v in row] for row in matrix]


def cross_column_profile(
    sample: pa.Table,
    columns: Sequence[ColumnProfile],
    keys: set[str],
    settings: ProfilingSettings,
) -> CrossColumnProfile:
    """Spearman matrix, strong pairs, conditional distributions, rules and null co-occurrence."""
    selected, truncated = select_columns(columns, keys, settings)
    rows = sample.slice(0, settings.pair_sample_size)
    arrays = _arrays(rows, selected)
    strong = settings.strong_association
    spearman = _spearman(arrays.numeric)
    names = list(arrays.numeric)
    numeric_pairs = [
        numeric_pair(names[i], names[j], *_both(arrays.numeric[names[i]], arrays.numeric[names[j]]))
        for i, j in combinations(range(len(names)), 2)
        if abs(spearman[i][j]) >= strong
    ]
    categorical = [c.name for c in selected if c.inferred_type in CATEGORICAL]
    categorical_pairs = _categorical_pairs(arrays, categorical, strong)
    decimals = {c.name: c.numeric.decimal_places.value for c in selected if c.numeric is not None}
    arithmetic = dict(list(arrays.numeric.items())[: settings.max_arithmetic_columns])
    return CrossColumnProfile(
        columns=[c.name for c in selected],
        truncated=truncated,
        rows=rows.num_rows,
        spearman_columns=names,
        spearman=spearman,
        numeric_pairs=numeric_pairs,
        categorical_pairs=categorical_pairs,
        mixed_pairs=_mixed_pairs(arrays, categorical, settings),
        conditionals=[
            _conditional(arrays, given, dependent, settings.max_conditional_levels)
            for pair in categorical_pairs
            for given, dependent in ((pair.a, pair.b), (pair.b, pair.a))
        ],
        rules=[
            *functional_dependencies(arrays.codes, settings.rule_min_share),
            *ordering_rules(arrays.numeric | arrays.temporal, settings.rule_min_share),
            *arithmetic_rules(
                arithmetic,
                {n: 0.5 * 10.0 ** -decimals.get(n, 0) + 1e-9 for n in arithmetic},
                settings.rule_min_share,
            ),
        ],
        null_pairs=null_pairs(_null_masks(rows, columns, settings.max_pair_columns), strong),
    )


def _both(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    keep = ~np.isnan(x) & ~np.isnan(y)
    return x[keep], y[keep]


def _present(
    arrays: Arrays, a: str, b: str
) -> tuple[NDArray[np.int64], list[str], NDArray[np.int64], list[str]]:
    keep = (arrays.codes[a] >= 0) & (arrays.codes[b] >= 0)
    return arrays.codes[a][keep], arrays.labels[a], arrays.codes[b][keep], arrays.labels[b]


def _conditional(arrays: Arrays, given: str, dependent: str, limit: int) -> ConditionalShares:
    codes_g, labels_g, codes_d, labels_d = _present(arrays, given, dependent)
    return conditional_shares(
        given,
        dependent,
        given_codes=codes_g,
        given_labels=labels_g,
        dependent_codes=codes_d,
        dependent_labels=labels_d,
        level_limit=limit,
    )


def _categorical_pairs(arrays: Arrays, names: list[str], strong: float) -> list[CategoricalPair]:
    found: list[CategoricalPair] = []
    for a, b in combinations(names, 2):
        x, _, y, _ = _present(arrays, a, b)
        if x.size and cramers_v(x, y) >= strong:
            found.append(categorical_pair(a, b, x, y))
    return found


def _mixed_pairs(
    arrays: Arrays, categorical: list[str], settings: ProfilingSettings
) -> list[MixedPair]:
    found: list[MixedPair] = []
    for numeric, values in arrays.numeric.items():
        for name in categorical:
            keep = ~np.isnan(values) & (arrays.codes[name] >= 0)
            if not keep.any():
                continue
            codes = arrays.codes[name][keep]
            if correlation_ratio(values[keep], codes) >= settings.strong_association:
                found.append(
                    mixed_pair(
                        numeric,
                        name,
                        values[keep],
                        codes,
                        labels=arrays.labels[name],
                        level_limit=settings.max_conditional_levels,
                    )
                )
    return found
