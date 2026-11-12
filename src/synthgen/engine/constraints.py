"""Enforce a table's constraints on drawn values (SPEC 8.5, after derived columns, before nulls).

order: rows where lower > upper swap the two values (both columns keep their values as a set);
when `lower` is a parent column, upper is reflected above it instead. range: values are clipped.
unique: rows repeating an earlier combination redraw the last listed column. expression: rows
where it is false redraw the free columns it reads. A constraint still broken after
`MAX_REDRAWS` rounds is an engine error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.engine.generators import Values
from synthgen.spec.constraints import (
    Constraint,
    ExpressionConstraint,
    OrderConstraint,
    RangeConstraint,
    UniqueConstraint,
)
from synthgen.spec.evaluator import Column, evaluate, to_internal, to_output
from synthgen.spec.expressions import ExprType, parse, references

MAX_REDRAWS = 20
Rows = NDArray[np.int64]
Redraw = Callable[[str, Rows], None]
"""Draws fresh values for one column on some rows, in place."""


def enforce(
    constraints: Sequence[Constraint],
    values: dict[str, Values],
    types: Mapping[str, ExprType],
    parents: Mapping[str, Column],
    redraw: Redraw,
) -> None:
    """Apply every constraint in order; `redraw` replaces a column's values on some rows."""
    for constraint in constraints:
        if isinstance(constraint, OrderConstraint):
            _order(constraint, values, types, parents)
        elif isinstance(constraint, RangeConstraint):
            _range(constraint, values, types)
        elif isinstance(constraint, UniqueConstraint):
            _unique(constraint, values, redraw)
        elif isinstance(constraint, ExpressionConstraint):
            _expression(constraint, values, types, parents, redraw)


def _order(
    rule: OrderConstraint,
    values: dict[str, Values],
    types: Mapping[str, ExprType],
    parents: Mapping[str, Column],
) -> None:
    kind = types[rule.upper]
    upper = to_internal(values[rule.upper], kind)
    if rule.lower in parents:
        lower_kind, lower_values = parents[rule.lower]
        lower = to_internal(lower_values, lower_kind)
    else:
        lower = to_internal(values[rule.lower], types[rule.lower])
    broken = np.flatnonzero(lower > upper)  # NaN compares false: nulls never break an order
    if broken.size == 0:
        return
    fixed = upper.copy()
    if rule.lower in parents:
        fixed[broken] = 2 * lower[broken] - upper[broken]
    else:
        fixed[broken] = lower[broken]
        swapped = lower.copy()
        swapped[broken] = upper[broken]
        values[rule.lower] = _back(values[rule.lower], swapped, types[rule.lower], broken)
    values[rule.upper] = _back(values[rule.upper], fixed, kind, broken)


def _back(original: Values, internal: NDArray[np.float64], kind: ExprType, rows: Rows) -> Values:
    """Write changed rows back in the engine's representation of the column type."""
    out = original.astype(object)
    out[rows] = to_output(internal[rows], kind)
    return out


def _range(rule: RangeConstraint, values: dict[str, Values], types: Mapping[str, ExprType]) -> None:
    kind = types[rule.column]
    current = to_internal(values[rule.column], kind)
    bounds = [
        to_internal(np.array([b], dtype=object), kind)[0] if b is not None else np.nan
        for b in (rule.minimum, rule.maximum)
    ]
    low = -np.inf if np.isnan(bounds[0]) else bounds[0]
    high = np.inf if np.isnan(bounds[1]) else bounds[1]
    broken = np.flatnonzero((current < low) | (current > high))
    if broken.size:
        clipped = np.clip(current, low, high)
        values[rule.column] = _back(values[rule.column], clipped, kind, broken)


def _repeats(values: dict[str, Values], columns: Sequence[str]) -> Rows:
    keys = np.array(
        ["\x1f".join(map(str, row)) for row in zip(*(values[c] for c in columns), strict=True)]
    )
    _, first = np.unique(keys, return_index=True)
    return np.setdiff1d(np.arange(keys.size), first).astype(np.int64)


def _unique(rule: UniqueConstraint, values: dict[str, Values], redraw: Redraw) -> None:
    for _ in range(MAX_REDRAWS):
        repeats = _repeats(values, rule.columns)
        if repeats.size == 0:
            return
        redraw(rule.columns[-1], repeats)
    raise EngineError(f"could not make {', '.join(rule.columns)} unique")


def _expression(
    rule: ExpressionConstraint,
    values: dict[str, Values],
    types: Mapping[str, ExprType],
    parents: Mapping[str, Column],
    redraw: Redraw,
) -> None:
    tree = parse(rule.expression)
    own = sorted(name for name in references(tree) if name in values)
    for _ in range(MAX_REDRAWS):
        frame = {**parents, **{k: (types[k], values[k]) for k in own}}
        n = next(iter(values.values())).size
        broken = np.flatnonzero(evaluate(tree, frame, n) == 0)
        if broken.size == 0:
            return
        for name in own:
            redraw(name, broken)
    raise EngineError(f"rows still break the constraint {rule.expression}")
