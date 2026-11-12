"""Scenario checks (SPEC 8.6): presence, share, signatures, separation, near misses and labels.

Each scenario must be present at its approved share; its rows satisfy its signature and stand
apart from typical rows; near misses never match a target; labels agree with the manifest.
Every result names its scenario, so failures route to repair.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from synthgen.domain.plan import Assertion, PlannedScenario, ScenarioCategory
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.scenarios import scenario_counts
from synthgen.engine.schema import SCENARIO_COLUMN
from synthgen.spec.evaluator import to_internal
from synthgen.spec.expressions import TIMES
from synthgen.spec.models import (
    EXPR_TYPES,
    TYPICAL,
    ColumnSpec,
    ColumnType,
    ScenarioSpec,
    TableSpec,
)
from synthgen.validation.output import Cells, OutputData, ValidationInputs
from synthgen.validation.signature import ROW_METRICS, AssertionRows, outcomes
from synthgen.validation.stats import ks_statistic


def _result(  # noqa: PLR0913, PLR0917 - the fields of one report row
    scenario: str | None,
    where: tuple[str, str | None],
    check: str,
    metric: str,
    value: float | None,
    threshold: float,
    *,
    upper: bool,
) -> CheckResult:
    if value is None:
        status = CheckStatus.SKIPPED
    else:
        ok = value <= threshold if upper else value >= threshold
        status = CheckStatus.PASS if ok else CheckStatus.FAIL
    return CheckResult(
        family=CheckFamily.SCENARIO,
        check=check,
        table=where[0],
        column=where[1],
        metric=metric,
        value=value,
        threshold=threshold,
        status=status,
        scenario=scenario,
    )


def check_scenarios(inputs: ValidationInputs, data: OutputData) -> list[CheckResult]:
    """Every scenario check for every scenario in the spec."""
    planned = {s.name: s for s in inputs.plan.scenarios} if inputs.plan else {}
    results: list[CheckResult] = []
    for scenario in inputs.spec.scenarios:
        results += _share(inputs, data, scenario)
        results += _labels(inputs, data, scenario)
        plan = planned.get(scenario.name)
        if plan is not None:
            for assertion in plan.signature:
                results += _signature(inputs, data, scenario, assertion)
    results += _near_misses(inputs, data, list(planned.values()))
    results += _manifest_counts(inputs, data)
    return results


def _share(inputs: ValidationInputs, data: OutputData, scenario: ScenarioSpec) -> list[CheckResult]:
    if data.entry(scenario.entity) is None:
        return []
    labels = data.label(scenario.entity, SCENARIO_COLUMN)
    own = {s.name for s in inputs.spec.scenarios if s.entity == scenario.entity}
    eligible = int(np.isin(labels, [TYPICAL, *own]).sum())
    actual = int((labels == scenario.name).sum())
    expected = scenario_counts([scenario.share], eligible)[0] if eligible else 0
    gap = abs(actual - expected)
    tolerance = inputs.settings.scenario_share_rel_tolerance
    error = 0.0 if gap <= 1 else gap / max(expected, 1)
    where = (scenario.entity, None)
    return [
        _result(scenario.name, where, "present", "rows", float(actual), 1.0, upper=False),
        _result(scenario.name, where, "share", "relative_error", error, tolerance, upper=True),
    ]


def _labels(
    inputs: ValidationInputs, data: OutputData, scenario: ScenarioSpec
) -> list[CheckResult]:
    del inputs
    results: list[CheckResult] = []
    for label in scenario.labels:
        rows = data.label(label.table, SCENARIO_COLUMN) == scenario.name
        values = data.column(label.table, label.column)[rows]
        wanted = float(label.value) if isinstance(label.value, bool | int | float) else label.value
        wrong = sum(1 for v in values.tolist() if _as_label(v) != wanted)
        where = (label.table, label.column)
        results.append(
            _result(
                scenario.name, where, "labels", "mismatched_rows", float(wrong), 0.0, upper=True
            )
        )
    return results


def _as_label(value: object) -> object:
    return float(value) if isinstance(value, bool | int | float) else value


def _rows(
    inputs: ValidationInputs, data: OutputData, assertion: Assertion, scenario: str
) -> AssertionRows | None:
    """The assertion column on the scenario's rows, with entity keys and event times."""
    table = inputs.spec.table(assertion.table)
    column = table.column(assertion.column) if table else None
    if table is None or column is None or data.entry(table.name) is None:
        return None
    picked = data.label(table.name, SCENARIO_COLUMN) == scenario
    owner = next(s for s in inputs.spec.scenarios if s.name == scenario)
    entities = _entity_keys(inputs, data, table, owner.entity, picked)
    time = _time_column(inputs, table, scenario)
    times = None
    if time is not None:
        kind = EXPR_TYPES[time.type]
        times = to_internal(data.column(table.name, time.name)[picked], kind)
    return AssertionRows(
        values=data.column(table.name, column.name)[picked],
        kind=EXPR_TYPES[column.type],
        entities=entities,
        times=times,
    )


def _entity_keys(
    inputs: ValidationInputs, data: OutputData, table: TableSpec, entity: str, picked: Cells
) -> Cells | None:
    """Key of the scenario entity each picked row belongs to, following driving keys upward.

    None when the rows are the entity themselves (each row is its own entity).
    """
    rule = table.row_rule
    if table.name == entity or rule.kind != "per_parent":
        return None
    keys = data.column(table.name, rule.child_column)[picked]
    while rule.parent_table != entity:
        parent = inputs.spec.table(rule.parent_table)
        if parent is None or parent.row_rule.kind != "per_parent":
            return keys
        upward = dict(
            zip(
                data.column(parent.name, rule.parent_column).tolist(),
                data.column(parent.name, parent.row_rule.child_column).tolist(),
                strict=True,
            )
        )
        keys = np.array([upward.get(k) for k in keys.tolist()], dtype=object)
        rule = parent.row_rule
    return keys


def _time_column(inputs: ValidationInputs, table: TableSpec, scenario: str) -> ColumnSpec | None:
    """The scenario's sequence time column, else the table's first date-time, else first date."""
    spec = next((s for s in inputs.spec.scenarios if s.name == scenario), None)
    sequence = next((q for q in spec.sequences if q.table == table.name), None) if spec else None
    if sequence is not None:
        return table.column(sequence.time_column)
    ordered = sorted(
        (c for c in table.columns if c.type in (ColumnType.DATETIME, ColumnType.DATE)),
        key=lambda c: c.type is not ColumnType.DATETIME,
    )
    return ordered[0] if ordered else None


def _signature(
    inputs: ValidationInputs, data: OutputData, scenario: ScenarioSpec, assertion: Assertion
) -> list[CheckResult]:
    rows = _rows(inputs, data, assertion, scenario.name)
    where = (assertion.table, assertion.column)
    found = outcomes(assertion, rows) if rows is not None and rows.values.size else None
    rate = float(found.mean()) if found is not None and found.size else None
    results = [
        _result(
            scenario.name,
            where,
            "signature",
            "pass_rate",
            rate,
            inputs.settings.signature_pass_min,
            upper=False,
        )
    ]
    labelled = (assertion.table, assertion.column) in {
        (lb.table, lb.column) for lb in scenario.labels
    }
    if (
        rows is not None
        and not labelled  # a label is set, not generated; it need not differ from typical rows
        and assertion.metric in ROW_METRICS
        and rows.kind is not EXPR_TYPES[ColumnType.TEXT]
    ):
        results.append(_separation(inputs, data, scenario.name, assertion, rows=rows))
    return results


def _separation(
    inputs: ValidationInputs,
    data: OutputData,
    scenario: str,
    assertion: Assertion,
    rows: AssertionRows,
) -> CheckResult:
    """KS between the scenario's and the typical rows on a signature column."""
    typical = data.typical(assertion.table)
    others = to_internal(data.column(assertion.table, assertion.column)[typical], rows.kind)
    mine = to_internal(rows.values, rows.kind)
    others, mine = others[~np.isnan(others)], mine[~np.isnan(mine)]
    if assertion.metric.value == "hour_of_day" and rows.kind in TIMES:
        others, mine = (np.floor((x % 86_400) / 3_600) for x in (others, mine))
    value = ks_statistic(mine, others) if mine.size and others.size else None
    return _result(
        scenario,
        (assertion.table, assertion.column),
        "separation",
        "ks_statistic",
        value,
        inputs.settings.scenario_ks_min,
        upper=False,
    )


def _near_misses(
    inputs: ValidationInputs, data: OutputData, planned: list[PlannedScenario]
) -> list[CheckResult]:
    """Rows of near-miss scenarios that satisfy every row-level assertion of a target scenario."""
    targets = [p for p in planned if p.category is ScenarioCategory.TARGET_SCENARIO]
    results: list[CheckResult] = []
    for near in (p for p in planned if p.category is ScenarioCategory.NEAR_MISS):
        for target in targets:
            by_table: dict[str, list[Assertion]] = {}
            for a in target.signature:
                if a.metric in ROW_METRICS:
                    by_table.setdefault(a.table, []).append(a)
            for table, assertions in by_table.items():
                matching = None
                for assertion in assertions:
                    rows = _rows(inputs, data, assertion, near.name)
                    passed = outcomes(assertion, rows) if rows is not None else None
                    if passed is None:
                        break
                    matching = passed if matching is None else matching & passed
                count = None if matching is None else float(matching.sum())
                results.append(
                    _result(
                        near.name,
                        (table, None),
                        "near_miss",
                        f"rows_matching_{target.name}",
                        count,
                        0.0,
                        upper=True,
                    )
                )
    return results


def _manifest_counts(inputs: ValidationInputs, data: OutputData) -> list[CheckResult]:
    """Rows per scenario in the data must equal the manifest's counts."""
    results: list[CheckResult] = []
    for table in inputs.spec.tables:
        entry = data.entry(table.name)
        if entry is None:
            continue
        actual = Counter(data.label(table.name, SCENARIO_COLUMN).tolist())
        recorded = {c.scenario: c.rows for c in entry.scenario_counts}
        wrong = sum(abs(actual.get(k, 0) - recorded.get(k, 0)) for k in set(actual) | set(recorded))
        results.append(
            _result(
                TYPICAL,
                (table.name, None),
                "label_counts",
                "rows_off_manifest",
                float(wrong),
                0.0,
                upper=True,
            )
        )
    return results
