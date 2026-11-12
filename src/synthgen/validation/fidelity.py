"""Fidelity checks (SPEC 8.6): typical rows against the source profile, as effect sizes.

Numbers: KS and Wasserstein/IQR against the stored quantiles. Categories: Jensen-Shannon over the
stored frequencies. Dates: Jensen-Shannon over month, weekday and hour mixes. Every column: the
null-rate difference. Tables with fewer typical rows than `min_fidelity_rows` report these checks
as skipped (PLAN Q6); the row count is always checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synthgen.config import ValidationSettings
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.keys import expected_children
from synthgen.engine.runner import RunOptions
from synthgen.engine.writer import Manifest
from synthgen.profiling.models import ColumnProfile, TableProfile
from synthgen.spec.evaluator import to_internal
from synthgen.spec.expressions import ExprType
from synthgen.spec.models import ColumnSpec, ColumnType, GenerationSpec, TableSpec
from synthgen.validation.output import Cells, OutputData, ValidationInputs
from synthgen.validation.stats import (
    category_js,
    js_distance,
    ks_statistic,
    time_mixes,
    wasserstein_over_iqr,
)

QUANTILE_GENERATORS = ("empirical", "distribution", "derived", "conditional")
_NUMBERS = (ColumnType.INTEGER, ColumnType.FLOAT)
_TIMES = (ColumnType.DATE, ColumnType.DATETIME)


@dataclass(frozen=True)
class Scope:
    """Where a check runs and whether the sample is too small to judge (PLAN Q6)."""

    table: str
    column: str | None
    small: str | None
    """Why distribution checks are skipped here, or None."""


def result(  # noqa: PLR0913 - the fields of one report row
    scope: Scope, check: str, metric: str, value: float | None, threshold: float, *, upper: bool
) -> CheckResult:
    """A fidelity row. `upper`: the value must stay at or below the threshold, else at or above."""
    if scope.small or value is None:
        status, detail = CheckStatus.SKIPPED, scope.small or "not measurable on this column"
    else:
        ok = value <= threshold if upper else value >= threshold
        status, detail = (CheckStatus.PASS if ok else CheckStatus.FAIL), None
    return CheckResult(
        family=CheckFamily.FIDELITY,
        check=check,
        table=scope.table,
        column=scope.column,
        metric=metric,
        value=value,
        threshold=threshold,
        status=status,
        detail=detail,
    )


def check_row_counts(
    spec: GenerationSpec, manifest: Manifest, options: RunOptions, settings: ValidationSettings
) -> list[CheckResult]:
    """Each table's row count against the target the spec implies, within the tolerance."""
    actual = {t.name: t.rows for t in manifest.tables}
    results: list[CheckResult] = []
    for table in spec.tables:
        rule = table.row_rule
        if rule.kind == "fixed":
            cap = options.max_root_rows
            target = min(rule.rows, cap) if cap else rule.rows
        else:
            parents = actual.get(rule.parent_table, 0)
            target = expected_children(rule, parents, use_target=not options.is_sample)
        rows = actual.get(table.name, 0)
        error = abs(rows - target) / max(target, 1)
        row = result(
            Scope(table.name, None, None),
            "row_count",
            "relative_error",
            error,
            settings.row_count_tolerance,
            upper=True,
        )
        results.append(row.model_copy(update={"detail": f"{rows} rows, target {target}"}))
    return results


def column_checks(
    inputs: ValidationInputs, data: OutputData, table: TableSpec, profile: TableProfile
) -> list[CheckResult]:
    """Null rate and distribution checks for every column the profile describes."""
    typical = data.typical(table.name)
    small = too_small(int(typical.sum()), profile.sample_rows, inputs.settings)
    rows = _sample(np.flatnonzero(typical), inputs.settings.fidelity_sample_size)
    results: list[CheckResult] = []
    for column in table.columns:
        stats = profile.column(column.name)
        if stats is None or column.primary_key:
            continue
        values = data.column(table.name, column.name)[rows]
        scope = Scope(table.name, column.name, small)
        results += _column(
            inputs.settings, scope, column, stats, values, is_key=_is_key(table, column)
        )
    return results


def too_small(synthetic: int, profiled: int, settings: ValidationSettings) -> str | None:
    """Why the sample is too small to judge (PLAN Q6, both sides), or None when it is not."""
    minimum = settings.min_fidelity_rows
    if synthetic < minimum:
        return f"only {synthetic} typical rows; fidelity needs {minimum}"
    if profiled < minimum:
        return f"the profile rests on only {profiled} source rows; fidelity needs {minimum}"
    return None


def _sample(rows: np.ndarray, limit: int) -> np.ndarray:
    """At most `limit` rows, chosen with a fixed seed so reports are reproducible."""
    if rows.size <= limit:
        return rows
    return np.sort(np.random.default_rng(0).choice(rows, size=limit, replace=False))


def _is_key(table: TableSpec, column: ColumnSpec) -> bool:
    rule = table.row_rule
    driving = rule.kind == "per_parent" and rule.child_column == column.name
    return driving or column.generator.type in ("foreign_key", "sequence", "uuid")


def _column(  # noqa: PLR0913 - one column's full context
    settings: ValidationSettings,
    scope: Scope,
    column: ColumnSpec,
    stats: ColumnProfile,
    values: Cells,
    *,
    is_key: bool,
) -> list[CheckResult]:
    present = np.array([v is not None for v in values], dtype=bool)
    null_rate = float(1.0 - present.mean()) if values.size else 0.0
    found = [
        result(
            scope,
            "null_rate",
            "abs_difference",
            abs(null_rate - stats.null_rate.value),
            settings.null_rate_diff_max,
            upper=True,
        )
    ]
    if is_key or column.sensitive:
        return found
    kept = values[present]
    gen = column.generator.type
    if column.type in _NUMBERS and stats.numeric is not None and gen in QUANTILE_GENERATORS:
        found += _numeric(settings, scope, kept.astype(np.float64), stats)
    elif gen == "categorical" and stats.categorical is not None:
        reference = {str(v.value): float(v.count) for v in stats.categorical.frequencies.value}
        value = category_js(kept.tolist(), reference) if reference else None
        found.append(
            result(
                scope, "category_mix", "js_distance", value, settings.js_category_max, upper=True
            )
        )
    elif column.type in _TIMES and stats.temporal is not None and gen == "datetime":
        found += _temporal(settings, scope, column, kept, stats)
    return found


def _numeric(
    settings: ValidationSettings, scope: Scope, values: np.ndarray, stats: ColumnProfile
) -> list[CheckResult]:
    assert stats.numeric is not None  # noqa: S101 - checked by the caller
    reference = np.asarray(stats.numeric.quantiles.value, dtype=np.float64)
    ks = ks_statistic(values, reference) if values.size else None
    distance = wasserstein_over_iqr(values, reference) if values.size else None
    return [
        result(scope, "distribution", "ks_statistic", ks, settings.ks_max, upper=True),
        result(
            scope,
            "distribution",
            "wasserstein_iqr",
            distance,
            settings.wasserstein_iqr_max,
            upper=True,
        ),
    ]


def _temporal(
    settings: ValidationSettings,
    scope: Scope,
    column: ColumnSpec,
    values: np.ndarray,
    stats: ColumnProfile,
) -> list[CheckResult]:
    assert stats.temporal is not None  # noqa: S101 - checked by the caller
    kind = ExprType.DATETIME if column.type is ColumnType.DATETIME else ExprType.DATE
    seconds = to_internal(values, kind)
    mixes = time_mixes(seconds[~np.isnan(seconds)], with_hours=kind is ExprType.DATETIME)
    expected = {
        "month": stats.temporal.month_counts.value,
        "weekday": stats.temporal.weekday_counts.value,
        "hour": stats.temporal.hour_counts.value,
    }
    return [
        result(
            scope,
            f"{name}_mix",
            "js_distance",
            js_distance(counts, expected[name]) if values.size else None,
            settings.js_temporal_max,
            upper=True,
        )
        for name, counts in mixes.items()
    ]
