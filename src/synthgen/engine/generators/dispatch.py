"""`generate`: one drawing function per generator type.

Also holds the generators that read other columns of the same row (`derived`, `conditional`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from synthgen.engine.generators import basic, foreign, temporal, text
from synthgen.engine.generators.context import GenContext, Values
from synthgen.spec.evaluator import evaluate
from synthgen.spec.expressions import parse
from synthgen.spec.generators import ConditionalGen, DerivedGen, Generator

Drawer = Callable[[Any, int, np.random.Generator, GenContext], Values]  # Any: one generator model


def generate(gen: Generator, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    """Draw n values for one column."""
    return _DRAWERS[gen.type](gen, n, rng, ctx)


def _derived(gen: DerivedGen, n: int, _rng: np.random.Generator, ctx: GenContext) -> Values:
    return evaluate(parse(gen.expression), ctx.frame, n)


def condition_key(value: Any) -> float | str | None:  # Any: a value of any column type
    """Comparable form of a value: numbers and booleans as floats, everything else as text."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        return float(value)
    return str(value)


def _conditional(gen: ConditionalGen, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    keys = [condition_key(v) for v in ctx.frame[gen.column][1]]
    out = np.empty(n, dtype=object)
    free = np.ones(n, dtype=bool)
    choices: list[tuple[set[float | str | None] | None, Generator]] = [
        ({condition_key(v) for v in case.values}, case.generator) for case in gen.cases
    ]
    for wanted, case_gen in [*choices, (None, gen.otherwise)]:
        match = free if wanted is None else free & np.array([k in wanted for k in keys], bool)
        rows = np.flatnonzero(match)
        if rows.size:
            out[rows] = generate(case_gen, rows.size, rng, ctx.subset(rows))
            free[rows] = False
    return out


_DRAWERS: dict[str, Drawer] = {
    "sequence": basic.sequence,
    "uuid": basic.uuids,
    "foreign_key": foreign.foreign_keys,
    "empirical": basic.quantile_draw,
    "distribution": basic.quantile_draw,
    "categorical": basic.categorical,
    "datetime": temporal.datetimes,
    "pattern": text.patterns,
    "semantic": text.semantic,
    "text_pool": basic.uniform_pick,
    "boundary": basic.uniform_pick,
    "derived": _derived,
    "conditional": _conditional,
    "constant": basic.constant,
}
