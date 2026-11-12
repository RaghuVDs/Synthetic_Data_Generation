"""Evaluate scenario signature assertions (SPEC 5) on a scenario's rows.

Row metrics (`value`, `hour_of_day`) give one outcome per row. Entity metrics (`count`, `sum`,
`mean`, `min`, `max`, `distinct_count`, `inter_arrival_hours`) group rows by their scenario entity
key and give one outcome per group; with `window_hours` the count, sum and distinct count are
the largest inside any window of that length.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.plan import Assertion, AssertionMetric, AssertionOperator
from synthgen.spec.evaluator import to_internal
from synthgen.spec.expressions import TIMES, ExprType
from synthgen.validation.output import Cells

Bools = NDArray[np.bool_]
Floats = NDArray[np.float64]
SECONDS_PER_HOUR = 3_600
HOURS_PER_DAY = 24
ROW_METRICS = (AssertionMetric.VALUE, AssertionMetric.HOUR_OF_DAY)
M = AssertionMetric


@dataclass(frozen=True)
class AssertionRows:
    """A scenario's rows in the assertion's table: the column, its type, entities and times."""

    values: Cells
    kind: ExprType
    entities: Cells | None
    """Scenario entity key per row; None when each row is an entity of its own."""
    times: Floats | None
    """Event time per row in seconds, for windows and gaps; None when the table has none."""


def outcomes(assertion: Assertion, rows: AssertionRows) -> Bools | None:
    """Pass or fail per row (row metrics) or per entity; None when the metric cannot apply."""
    if assertion.metric is M.VALUE:
        return compare(
            assertion.operator, _comparable(rows.values, rows.kind), assertion.value, rows.kind
        )
    if assertion.metric is M.HOUR_OF_DAY:
        if rows.kind not in TIMES:
            return None
        seconds = to_internal(rows.values, rows.kind)
        hours = np.floor((seconds % 86_400) / SECONDS_PER_HOUR)
        return compare(assertion.operator, hours, assertion.value, ExprType.NUMBER)
    measured = _per_entity(assertion, rows)
    if measured is None:
        return None
    if assertion.metric is M.INTER_ARRIVAL_HOURS:
        return np.array(
            [
                bool(np.all(compare(assertion.operator, g, assertion.value, ExprType.NUMBER)))
                for g in measured
            ],
            dtype=bool,
        )
    return compare(
        assertion.operator, np.asarray(measured, dtype=np.float64), assertion.value, ExprType.NUMBER
    )


def _comparable(values: Cells, kind: ExprType) -> Cells:
    if kind is ExprType.TEXT:
        return np.array([None if v is None else str(v) for v in values], dtype=object)
    return to_internal(values, kind)


def compare(
    op: AssertionOperator, observed: Cells, expected: Any, kind: ExprType
) -> Bools:  # Any: scalar or list
    """Row-wise comparison; a missing value never passes."""
    wanted = (
        [_expected(v, kind) for v in expected]
        if isinstance(expected, list)
        else _expected(expected, kind)
    )
    present = np.array(
        [v is not None and not (isinstance(v, float) and np.isnan(v)) for v in observed], dtype=bool
    )
    out = np.zeros(observed.size, dtype=bool)
    rows = np.flatnonzero(present)
    seen = observed[rows]
    if op in (AssertionOperator.IN, AssertionOperator.NOT_IN):
        members = set(wanted) if isinstance(wanted, list) else {wanted}
        hit = np.array([v in members for v in seen.tolist()], dtype=bool)
        out[rows] = hit if op is AssertionOperator.IN else ~hit
        return out
    target = wanted[0] if isinstance(wanted, list) and wanted else wanted
    out[rows] = np.array([_OPS[op](v, target) for v in seen.tolist()], dtype=bool)
    return out


def _expected(value: Any, kind: ExprType) -> Any:  # Any: a JSON scalar from the plan
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if kind in TIMES and isinstance(value, str):
        return float(to_internal(np.array([value], dtype=object), kind)[0])
    if kind is ExprType.TEXT:
        return str(value)
    return float(value) if isinstance(value, int | float) else value


def _safe(compare_: Callable[[Any, Any], bool]) -> Callable[[Any, Any], bool]:
    def run(a: Any, b: Any) -> bool:  # Any: comparable scalars
        try:
            return bool(compare_(a, b))
        except TypeError:
            return False  # a text value compared with a number never passes

    return run


_OPS: dict[AssertionOperator, Callable[[Any, Any], bool]] = {
    AssertionOperator.EQ: _safe(lambda a, b: a == b),
    AssertionOperator.NE: _safe(lambda a, b: a != b),
    AssertionOperator.LT: _safe(lambda a, b: a < b),
    AssertionOperator.LE: _safe(lambda a, b: a <= b),
    AssertionOperator.GT: _safe(lambda a, b: a > b),
    AssertionOperator.GE: _safe(lambda a, b: a >= b),
}


def _per_entity(
    assertion: Assertion, rows: AssertionRows
) -> list[Any] | None:  # Any: a number or gaps
    entities = rows.entities if rows.entities is not None else np.arange(rows.values.size)
    groups: dict[Any, list[int]] = defaultdict(list)
    for index, key in enumerate(entities.tolist()):
        groups[key].append(index)
    if assertion.metric is M.INTER_ARRIVAL_HOURS:
        if rows.times is None:
            return None
        gaps = [
            np.diff(np.sort(rows.times[g])) / SECONDS_PER_HOUR
            for g in groups.values()
            if len(g) > 1
        ]
        return gaps
    numbers = _comparable(rows.values, rows.kind) if rows.kind is not ExprType.TEXT else rows.values
    return [_aggregate(assertion, numbers, rows.times, g) for g in groups.values()]


def _aggregate(
    assertion: Assertion, values: Cells, times: Floats | None, members: Sequence[int]
) -> float:
    picked = np.asarray(members, dtype=np.int64)
    if (
        assertion.window_hours is not None
        and times is not None
        and assertion.metric in (M.COUNT, M.SUM, M.DISTINCT_COUNT)
    ):
        return _best_window(assertion, values, times, picked)
    return _summary(assertion.metric, values[picked])


def _summary(metric: AssertionMetric, values: Cells) -> float:
    present = [
        v for v in values.tolist() if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    if metric is M.COUNT:
        return float(values.size)
    if metric is M.DISTINCT_COUNT:
        return float(len(set(present)))
    if not present:
        return float("nan")
    numbers = np.asarray(present, dtype=np.float64)
    return float(_REDUCERS[metric](numbers))


_REDUCERS: dict[AssertionMetric, Callable[[Floats], Any]] = {  # Any: a NumPy scalar
    M.SUM: np.sum,
    M.MEAN: np.mean,
    M.MIN: np.min,
    M.MAX: np.max,
}


def _best_window(
    assertion: Assertion, values: Cells, times: Floats, members: NDArray[np.int64]
) -> float:
    """The largest count, sum or distinct count over `window_hours` windows starting at an event."""
    order = members[np.argsort(times[members], kind="stable")]
    span = float(assertion.window_hours or 0) * SECONDS_PER_HOUR
    best = 0.0
    end = 0
    for start in range(order.size):
        while end < order.size and times[order[end]] - times[order[start]] <= span:
            end += 1
        best = max(best, _summary(assertion.metric, values[order[start:end]]))
    return best
