"""Table-level fidelity: Spearman correlations and children per parent, on typical rows."""

from __future__ import annotations

from collections import Counter

import numpy as np

from synthgen.domain.report import CheckResult
from synthgen.engine.schema import SCENARIO_COLUMN
from synthgen.profiling.models import TableProfile
from synthgen.spec.models import TYPICAL, TableSpec
from synthgen.validation.fidelity import Scope, result, too_small
from synthgen.validation.output import OutputData, ValidationInputs
from synthgen.validation.stats import discrete_ks, largest_spearman_gap


def spearman_check(
    inputs: ValidationInputs, data: OutputData, table: TableSpec, profile: TableProfile
) -> list[CheckResult]:
    """Largest gap between the synthetic and profiled Spearman matrices of the copula columns."""
    if table.correlation is None:
        return []
    names = [n for n in table.correlation.columns if n in profile.cross_column.spearman_columns]
    if len(names) < 2:  # noqa: PLR2004 - a correlation needs two columns
        return []
    typical = data.typical(table.name)
    positions = [profile.cross_column.spearman_columns.index(n) for n in names]
    reference = [[profile.cross_column.spearman[i][j] for j in positions] for i in positions]
    columns = [data.column(table.name, n)[typical] for n in names]
    complete = np.all([np.not_equal(c, None) for c in columns], axis=0)
    sample = np.column_stack([c[complete].astype(np.float64) for c in columns])
    small = too_small(int(complete.sum()), profile.cross_column.rows, inputs.settings)
    scope = Scope(table.name, ",".join(names), small)
    gap = largest_spearman_gap(sample, reference) if sample.size else None
    return [
        result(
            scope,
            "correlation",
            "max_spearman_difference",
            gap,
            inputs.settings.spearman_diff_max,
            upper=True,
        )
    ]


def children_check(
    inputs: ValidationInputs, data: OutputData, table: TableSpec, profile: TableProfile
) -> list[CheckResult]:
    """KS between children per typical parent and the profiled distribution."""
    rule = table.row_rule
    if rule.kind != "per_parent":
        return []
    shape = next((r for r in profile.relationships if r.child_column == rule.child_column), None)
    if shape is None:
        return []
    parents = data.column(rule.parent_table, rule.parent_column)
    typical = data.label(rule.parent_table, SCENARIO_COLUMN) == TYPICAL
    children = Counter(data.column(table.name, rule.child_column).tolist())
    sample = [children.get(key, 0) for key in parents[typical].tolist()]
    profiled = int(sum(n for _, n in shape.counts.value))
    scope = Scope(table.name, rule.child_column, too_small(len(sample), profiled, inputs.settings))
    if rule.target_rows is not None and not inputs.options.is_sample:
        scope = Scope(table.name, rule.child_column, "a row target rescales children per parent")
    reference = {int(c): float(n) for c, n in shape.counts.value}
    value = discrete_ks(sample, reference) if sample else None
    return [
        result(
            scope,
            "children_per_parent",
            "ks_statistic",
            value,
            inputs.settings.ks_children_max,
            upper=True,
        )
    ]
