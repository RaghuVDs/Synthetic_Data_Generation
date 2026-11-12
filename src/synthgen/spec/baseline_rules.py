"""Profiled rules turned into baseline constructs: order constraints and derived columns.

`a <= b` rules become order constraints; a child date that follows its driving parent's date
becomes an order constraint on `parent.column` and is drawn as that date plus the profiled lag;
`c = a + b` and `c = a * b` make `c` derived.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from synthgen.profiling.models import TableProfile
from synthgen.profiling.relation_models import RuleKind, TemporalOrder
from synthgen.spec.constraints import Constraint, OrderConstraint
from synthgen.spec.generators import DateLag, DatetimeGen, DerivedGen
from synthgen.spec.models import ColumnSpec, ColumnType, RowRule

_NAME = re.compile(r"^[A-Za-z_]\w*$")
_OPERATORS = {RuleKind.SUM: "+", RuleKind.PRODUCT: "*"}
_TIMES = (ColumnType.DATE, ColumnType.DATETIME)
_NUMBERS = (ColumnType.INTEGER, ColumnType.FLOAT)


def derived_columns(profile: TableProfile, columns: Sequence[ColumnSpec]) -> list[ColumnSpec]:
    """Columns whose profile shows `result = a op b` get a derived generator; others unchanged."""
    by_name = {c.name: c for c in columns}
    formulas: dict[str, str] = {}
    for rule in profile.cross_column.rules:
        result = by_name.get(rule.result or "")
        if rule.kind not in _OPERATORS or result is None or result.name in formulas:
            continue
        parts = [by_name.get(rule.a), by_name.get(rule.b)]
        if not _plain_number(result) or not all(p is not None and _plain_number(p) for p in parts):
            continue
        if {rule.a, rule.b} & set(formulas):
            continue  # keep derivations one level deep, so they never form a cycle
        formulas[result.name] = f"{rule.a} {_OPERATORS[rule.kind]} {rule.b}"
    return [
        c.model_copy(update={"generator": DerivedGen(type="derived", expression=formulas[c.name])})
        if c.name in formulas
        else c
        for c in columns
    ]


def order_constraints(
    profile: TableProfile, columns: Sequence[ColumnSpec], rule: RowRule, min_share: float
) -> list[Constraint]:
    """Within-table `a <= b` rules and child-after-driving-parent date orders."""
    by_name = {c.name: c for c in columns}
    found: list[Constraint] = [
        OrderConstraint(kind="order", lower=r.a, upper=r.b)
        for r in profile.cross_column.rules
        if r.kind is RuleKind.LESS_EQUAL and _orderable(by_name.get(r.a), by_name.get(r.b))
    ]
    for lower, order in _parent_orders(profile, columns, rule, min_share):
        found.append(OrderConstraint(kind="order", lower=lower, upper=order.child_column))
    return found


def parent_lagged(
    profile: TableProfile, columns: Sequence[ColumnSpec], rule: RowRule, min_share: float
) -> list[ColumnSpec]:
    """Child dates that follow a driving-parent date are drawn as that date plus the lag.

    A child following several parent dates follows the one with the shortest median lag.
    """
    tightest: dict[str, tuple[str, TemporalOrder]] = {}
    for lower, order in _parent_orders(profile, columns, rule, min_share):
        held = tightest.get(order.child_column)
        if held is None or order.lag_seconds.p50 < held[1].lag_seconds.p50:
            tightest[order.child_column] = (lower, order)
    out: list[ColumnSpec] = []
    for column in columns:
        found = tightest.get(column.name)
        if found is None or not isinstance(column.generator, DatetimeGen):
            out.append(column)
            continue
        lag = DateLag(column=found[0], lag_quantiles=[max(0.0, q) for q in found[1].lag_quantiles])
        after = column.generator.model_copy(update={"after": lag})
        out.append(column.model_copy(update={"generator": after}))
    return out


def _parent_orders(
    profile: TableProfile, columns: Sequence[ColumnSpec], rule: RowRule, min_share: float
) -> list[tuple[str, TemporalOrder]]:
    """Child-after-driving-parent date orders that hold on enough rows, with `parent.column`."""
    if rule.kind != "per_parent":
        return []
    by_name = {c.name: c for c in columns}
    found: list[tuple[str, TemporalOrder]] = []
    for shape in profile.relationships:
        if shape.parent_table != rule.parent_table or shape.child_column != rule.child_column:
            continue
        for order in shape.temporal_orders.value:
            child = by_name.get(order.child_column)
            if order.after_share >= min_share and child is not None and child.type in _TIMES:
                found.append((f"{rule.parent_table}.{order.parent_column}", order))
    return found


def _plain_number(column: ColumnSpec | None) -> bool:
    return (
        column is not None
        and column.type in _NUMBERS
        and not (column.primary_key or column.unique or column.sensitive)
        and column.generator.type in ("empirical", "distribution")
        and bool(_NAME.match(column.name))
    )


def _orderable(a: ColumnSpec | None, b: ColumnSpec | None) -> bool:
    if a is None or b is None or a.primary_key or b.primary_key:
        return False
    if a.type in _TIMES and b.type in _TIMES:
        return True
    return a.type in _NUMBERS and b.type in _NUMBERS and not (a.sensitive or b.sensitive)
