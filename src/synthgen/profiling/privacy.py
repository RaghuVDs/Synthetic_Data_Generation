"""Sensitivity classes, value folding, and the guard every stored profile passes (SPEC 8.3).

- public: full statistics and frequencies.
- internal: full statistics; values held by fewer than `internal_min_count` rows fold into
  "other".
- sensitive: masks, lengths and validity rates; no values.

Metadata tags override the rule that recognised personal formats are sensitive.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

import pandas as pd

from synthgen.config import ProfilingSettings
from synthgen.domain.errors import PrivacyError
from synthgen.domain.metadata import Sensitivity
from synthgen.profiling.models import ColumnProfile, TableProfile

SENSITIVE_SEMANTICS = frozenset({"email", "phone", "card_number"})
OTHER = "other"
TOLERANCE = 1e-9
VALUE_FIELDS = ("top_value", "top_share", "numeric", "categorical", "temporal", "text_format")
"""Column profile fields that hold values; a sensitive column stores none of them."""


def classify(tag: Sensitivity | None, semantic: str | None) -> Sensitivity:
    """Metadata tags win; otherwise recognised personal formats are sensitive."""
    if tag is not None:
        return tag
    return Sensitivity.SENSITIVE if semantic in SENSITIVE_SEMANTICS else Sensitivity.PUBLIC


def min_count(sensitivity: Sensitivity, settings: ProfilingSettings) -> int:
    """Fewest rows a value needs before it may be stored: internal columns fold rarer ones."""
    return settings.internal_min_count if sensitivity is Sensitivity.INTERNAL else 0


def fold_counts(counts: Mapping[str, int], threshold: int) -> Counter[str]:
    """Counts with every value under the threshold merged into `other`."""
    folded: Counter[str] = Counter()
    for value, count in counts.items():
        folded[value if count >= threshold else OTHER] += count
    return folded


def fold_labels(values: pd.Series, threshold: int) -> pd.Series:
    """Text values with those seen fewer than `threshold` times replaced by `other`."""
    if threshold <= 0:
        return values
    counts = values.value_counts()
    rare = set(counts[counts < threshold].index)
    return values.where(~values.isin(rare), OTHER)


def _column_violations(column: ColumnProfile, rows: int, settings: ProfilingSettings) -> list[str]:
    if column.sensitivity is Sensitivity.SENSITIVE:
        return [
            f"{column.name}: sensitive column stores {field}"
            for field in VALUE_FIELDS
            if getattr(column, field) is not None
        ]
    threshold = min_count(column.sensitivity, settings)
    found: list[str] = []
    if column.categorical is not None and any(
        v.count < threshold for v in column.categorical.frequencies.value
    ):
        found.append(f"{column.name}: internal values under {threshold} rows are stored")
    non_null = rows - column.null_count.value
    if column.top_share is not None and column.top_share.value * non_null < threshold - TOLERANCE:
        found.append(f"{column.name}: internal top value under {threshold} rows is stored")
    return found


def _columns_used(profile: TableProfile) -> list[tuple[str, str]]:
    """Columns whose values feed group F, H or I statistics, with where they are used."""
    used = [(name, "cross-column statistics") for name in profile.cross_column.columns]
    for behaviour in profile.behaviour:
        used += [(behaviour.time_column, "behaviour")]
        used += [(behaviour.measure_column, "behaviour")] if behaviour.measure_column else []
        used += [(t.column, "behaviour states") for t in behaviour.transitions]
    for stratified in profile.stratified:
        used += [(stratified.label_column, "label profile")]
        used += [(s.column, "label separation") for s in stratified.separation]
    return used


def privacy_violations(profile: TableProfile, settings: ProfilingSettings) -> list[str]:
    """Every place a profile stores a value its column's class forbids."""
    rows = profile.row_count.value
    found = [v for c in profile.columns for v in _column_violations(c, rows, settings)]
    sensitive = {c.name for c in profile.columns if c.sensitivity is Sensitivity.SENSITIVE}
    found += [
        f"{name}: sensitive column used in {where}"
        for name, where in _columns_used(profile)
        if name in sensitive
    ]
    return found


def check_privacy(profile: TableProfile, settings: ProfilingSettings) -> TableProfile:
    """The profile unchanged, or a PrivacyError naming every violation."""
    found = privacy_violations(profile, settings)
    if found:
        raise PrivacyError(
            f"profile of {profile.table} breaks its privacy classes: {'; '.join(found)}"
        )
    return profile
