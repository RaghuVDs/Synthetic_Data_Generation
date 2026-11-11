"""Label-stratified profile of one table (SPEC 8.3 I).

The semantic mapper names the label later in the graph, so the profiler profiles label
candidates chosen by code: binary, non-key, non-sensitive columns, boolean flags first and then
the rarest minority class, at most `max_label_columns`. Any other label is served on demand by
`get_class_separation`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray

from synthgen.config import ProfilingSettings
from synthgen.domain.metadata import Sensitivity
from synthgen.profiling.fitting import fit_distributions
from synthgen.profiling.label_models import (
    ClassBalance,
    ClassColumn,
    ClassNumbers,
    ClassTimes,
    ColumnSeparation,
    StratifiedProfile,
)
from synthgen.profiling.models import ColumnProfile, InferredType
from synthgen.profiling.stratified import categorical_separation, numeric_separation
from synthgen.profiling.temporal import epoch_seconds

NUMERIC = (InferredType.INTEGER, InferredType.FLOAT)
TEMPORAL = (InferredType.DATE, InferredType.DATETIME)
BINARY = 2
CLASS_PERCENTILES = (5.0, 25.0, 50.0, 75.0, 95.0)
SECONDS_PER_DAY = 86_400
WEEKDAYS = 7
HOURS = 24
OTHER = "other"
HOUR_KIND = "hour_of_day"
"""Separation kind of a date-time column whose hour of day separates better than its value."""


def _minority_share(column: ColumnProfile) -> float:
    counts: list[int] = []
    if column.categorical is not None:
        counts = [v.count for v in column.categorical.frequencies.value]
    total = sum(counts)
    return min(counts) / total if len(counts) == BINARY and total else 1.0


def label_candidates(
    columns: Sequence[ColumnProfile], keys: set[str], settings: ProfilingSettings
) -> list[ColumnProfile]:
    """Binary, non-key, non-sensitive columns: boolean flags first, then rarest minority class."""
    binary = [
        c
        for c in columns
        if c.name not in keys
        and c.sensitivity is not Sensitivity.SENSITIVE
        and c.categorical is not None
        and c.distinct_count.value == BINARY
    ]
    ranked = sorted(
        binary, key=lambda c: (c.inferred_type is not InferredType.BOOLEAN, _minority_share(c))
    )
    return ranked[: settings.max_label_columns]


def _texts(sample: pa.Table, name: str) -> pd.Series:
    values = pd.Series(sample.column(name).to_pylist(), dtype=object)
    return values.where(values.isna(), values.astype(str))


def _features(
    column: ColumnProfile, sample: pa.Table
) -> tuple[str, NDArray[np.float64] | None, pd.Series]:
    """Kind, numeric values (numbers and dates as epoch seconds) and text values of a column."""
    raw = sample.column(column.name).to_pylist()
    if column.inferred_type in NUMERIC:
        numbers = pd.to_numeric(pd.Series(raw, dtype=object), errors="coerce")
        return "numeric", numbers.to_numpy(np.float64), _texts(sample, column.name)
    if column.inferred_type in TEMPORAL:
        return "temporal", epoch_seconds(raw), _texts(sample, column.name)
    return "categorical", None, _texts(sample, column.name)


def _separation(
    column: ColumnProfile, sample: pa.Table, positive: NDArray[np.bool_], bins: int
) -> ColumnSeparation | None:
    kind, numbers, texts = _features(column, sample)
    if numbers is not None:
        present = ~np.isnan(numbers)
        pos, neg = numbers[positive & present], numbers[~positive & present]
        if not pos.size or not neg.size:
            return None
        by_value = numeric_separation(column.name, kind, pos, neg, bins)
        if column.inferred_type is not InferredType.DATETIME:
            return by_value
        by_hour = numeric_separation(column.name, HOUR_KIND, _hours(pos), _hours(neg), HOURS)
        return max(by_value, by_hour, key=lambda s: s.score)
    found, levels = pd.factorize(texts)
    codes = found.astype(np.int64)
    present = codes >= 0
    in_class, rest = codes[positive & present], codes[~positive & present]
    if not in_class.size or not rest.size:
        return None
    return categorical_separation(column.name, in_class, rest, len(levels))


def _hours(seconds: NDArray[np.float64]) -> NDArray[np.float64]:
    return ((seconds.astype(np.int64) % SECONDS_PER_DAY) // 3600).astype(np.float64)


def _numbers(values: NDArray[np.float64]) -> ClassNumbers:
    fits = fit_distributions(values, full=False)
    return ClassNumbers(
        count=int(values.size),
        mean=float(np.mean(values)),
        std=float(np.std(values)),
        percentiles=[(p, float(np.percentile(values, p))) for p in CLASS_PERCENTILES],
        best_fit=fits[0] if fits else None,
    )


def _times(seconds: NDArray[np.float64]) -> ClassTimes:
    whole = seconds.astype(np.int64)
    days = whole // SECONDS_PER_DAY
    weekdays = (days + 3) % WEEKDAYS  # 1970-01-01 was a Thursday; Monday = 0
    return ClassTimes(
        weekday_counts=np.bincount(weekdays, minlength=WEEKDAYS).tolist(),
        hour_counts=np.bincount(_hours(seconds).astype(np.int64), minlength=HOURS).tolist(),
    )


def _shares(
    column: ColumnProfile, texts: pd.Series, settings: ProfilingSettings
) -> list[tuple[str, float]]:
    """Top values within one class; internal columns fold values under the row minimum."""
    counts = Counter(texts.dropna().tolist())
    total = sum(counts.values())
    if column.sensitivity is Sensitivity.INTERNAL:
        rare = [v for v, n in counts.items() if n < settings.internal_min_count]
        for value in rare:
            counts[OTHER] += counts.pop(value)
    return [(str(v), n / total) for v, n in counts.most_common(settings.max_conditional_levels)]


def _class_column(
    column: ColumnProfile,
    sample: pa.Table,
    members: NDArray[np.bool_],
    value: str,
    settings: ProfilingSettings,
) -> ClassColumn:
    kind, numbers, texts = _features(column, sample)
    inside = None if numbers is None else numbers[members & ~np.isnan(numbers)]
    return ClassColumn(
        column=column.name,
        value=value,
        numbers=_numbers(inside)
        if kind == "numeric" and inside is not None and inside.size
        else None,
        times=_times(inside) if kind == "temporal" and inside is not None and inside.size else None,
        shares=_shares(column, texts[members], settings) if kind == "categorical" else None,
    )


def stratified_profile(
    sample: pa.Table,
    columns: Sequence[ColumnProfile],
    label: ColumnProfile,
    keys: set[str],
    settings: ProfilingSettings,
) -> StratifiedProfile:
    """Class balance, columns ranked by separation, and per-class summaries of the top ones."""
    frequencies = label.categorical.frequencies.value if label.categorical else []
    total = sum(v.count for v in frequencies)
    balance = [
        ClassBalance(value=str(v.value), count=v.count, share=v.count / total) for v in frequencies
    ]
    minority = min(balance, key=lambda b: (b.count, b.value)).value
    labels = _texts(sample, label.name)
    positive = (labels == minority).to_numpy(bool)
    usable = [
        c
        for c in columns
        if c.name not in keys | {label.name}
        and c.sensitivity is not Sensitivity.SENSITIVE
        and (c.inferred_type in NUMERIC or c.inferred_type in TEMPORAL or c.categorical is not None)
    ]
    found = [
        s
        for c in usable
        if (s := _separation(c, sample, positive, settings.separation_bins)) is not None
    ]
    ranked = sorted(found, key=lambda s: (-s.score, s.column))[: settings.max_stratified_columns]
    by_name = {c.name: c for c in usable}
    known = labels.notna().to_numpy(bool)
    return StratifiedProfile(
        label_column=label.name,
        minority=minority,
        balance=balance,
        sample_rows=sample.num_rows,
        separation=ranked,
        per_class=[
            _class_column(by_name[s.column], sample, members, b.value, settings)
            for s in ranked
            for b in balance
            if (members := known & (labels == b.value).to_numpy(bool)).any()
        ],
    )
