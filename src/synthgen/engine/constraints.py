"""Enforce a table's constraints on drawn values (SPEC 8.5, after derived columns, before nulls).

order: rows where lower > upper swap the two values (both columns keep their values as a set);
when `lower` is a parent column, upper is redrawn, and rows that still break are lifted to the
parent value (dates keep their time of day). range: values are clipped.
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
SECONDS_PER_DAY = 86_400
Rows = NDArray[np.int64]
Floats = NDArray[np.float64]
Redraw = Callable[[str, Rows], None]
"""Draws fresh values for one column on some rows, in place."""


def enforce(
    constraints: Sequence[Constraint],
    values: dict[str, Values],
    types: Mapping[str, ExprType],
    parents: Mapping[str, Column],
    redraw: Redraw,
) -> None:
    """Apply the constraints until a full pass changes nothing; `redraw` replaces some rows.

    Fixing one constraint can break an earlier one (a parent-date repair can undo an order
    between two child dates), so passes repeat, at most `MAX_REDRAWS` times.
    """
    for _ in range(MAX_REDRAWS):
        changed = 0
        for constraint in constraints:
            if isinstance(constraint, OrderConstraint):
                changed += _order(constraint, values, types, parents, redraw)
            elif isinstance(constraint, RangeConstraint):
                changed += _range(constraint, values, types)
            elif isinstance(constraint, UniqueConstraint):
                changed += _unique(constraint, values, redraw)
            elif isinstance(constraint, ExpressionConstraint):
                changed += _expression(constraint, values, types, parents, redraw)
        if not changed:
            return
    raise EngineError("the table's constraints contradict each other; no values satisfy all")


def _order(
    rule: OrderConstraint,
    values: dict[str, Values],
    types: Mapping[str, ExprType],
    parents: Mapping[str, Column],
    redraw: Redraw,
) -> int:
    if rule.lower in parents:
        return _parent_order(rule, values, types[rule.upper], parents[rule.lower], redraw)
    kind = types[rule.upper]
    upper = to_internal(values[rule.upper], kind)
    lower = to_internal(values[rule.lower], types[rule.lower])
    broken = np.flatnonzero(lower > upper)  # NaN compares false: nulls never break an order
    if broken.size == 0:
        return 0
    swapped = lower.copy()
    swapped[broken] = upper[broken]
    values[rule.lower] = _back(values[rule.lower], swapped, types[rule.lower], broken)
    values[rule.upper] = _back(values[rule.upper], lower, kind, broken)
    return int(broken.size)


def _parent_order(
    rule: OrderConstraint,
    values: dict[str, Values],
    kind: ExprType,
    parent: Column,
    redraw: Redraw,
) -> int:
    """Child below its parent value: redraw it from its own generator, then lift what is left.

    Redrawing keeps the child inside its own range and distribution; the lift handles rows whose
    parent value leaves little room.
    """
    lower = to_internal(parent[1], parent[0])
    changed = 0
    for _ in range(MAX_REDRAWS):
        broken = np.flatnonzero(lower > to_internal(values[rule.upper], kind))
        if broken.size == 0:
            return changed
        changed += int(broken.size)
        before = values[rule.upper]
        redraw(rule.upper, broken)
        if values[rule.upper] is before:
            break  # the column cannot be redrawn (a key or a computed column)
    upper = to_internal(values[rule.upper], kind)
    broken = np.flatnonzero(lower > upper)
    if broken.size:
        fixed = upper.copy()
        fixed[broken] = _lift(lower[broken], upper[broken], kind)
        values[rule.upper] = _back(values[rule.upper], fixed, kind, broken)
    return changed + int(broken.size)


def _lift(lower: NDArray[np.float64], upper: NDArray[np.float64], kind: ExprType) -> Floats:
    """The smallest value at or above `lower`; dates keep their own time of day.

    Only rows that redrawing could not fix get here, so the small pile-up at the parent's value
    stays inside the parent's range instead of being pushed past the source's dates.
    """
    if kind not in (ExprType.DATE, ExprType.DATETIME):
        return lower.copy()
    lifted = np.floor(lower / SECONDS_PER_DAY) * SECONDS_PER_DAY + upper % SECONDS_PER_DAY
    return np.asarray(np.where(lifted < lower, lifted + SECONDS_PER_DAY, lifted), dtype=np.float64)


def _back(original: Values, internal: NDArray[np.float64], kind: ExprType, rows: Rows) -> Values:
    """Write changed rows back in the engine's representation of the column type."""
    out = original.astype(object)
    out[rows] = to_output(internal[rows], kind)
    return out


def _range(rule: RangeConstraint, values: dict[str, Values], types: Mapping[str, ExprType]) -> int:
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
    return int(broken.size)


def _repeats(values: dict[str, Values], columns: Sequence[str]) -> Rows:
    keys = np.array(
        ["\x1f".join(map(str, row)) for row in zip(*(values[c] for c in columns), strict=True)]
    )
    _, first = np.unique(keys, return_index=True)
    return np.setdiff1d(np.arange(keys.size), first).astype(np.int64)


def _unique(rule: UniqueConstraint, values: dict[str, Values], redraw: Redraw) -> int:
    redrawn = 0
    for _ in range(MAX_REDRAWS):
        repeats = _repeats(values, rule.columns)
        if repeats.size == 0:
            return redrawn
        redraw(rule.columns[-1], repeats)
        redrawn += int(repeats.size)
    raise EngineError(f"could not make {', '.join(rule.columns)} unique")


def _expression(
    rule: ExpressionConstraint,
    values: dict[str, Values],
    types: Mapping[str, ExprType],
    parents: Mapping[str, Column],
    redraw: Redraw,
) -> int:
    tree = parse(rule.expression)
    own = sorted(name for name in references(tree) if name in values)
    redrawn = 0
    for _ in range(MAX_REDRAWS):
        frame = {**parents, **{k: (types[k], values[k]) for k in own}}
        n = next(iter(values.values())).size
        broken = np.flatnonzero(evaluate(tree, frame, n) == 0)
        if broken.size == 0:
            return redrawn
        redrawn += int(broken.size)
        for name in own:
            redraw(name, broken)
    raise EngineError(f"rows still break the constraint {rule.expression}")
