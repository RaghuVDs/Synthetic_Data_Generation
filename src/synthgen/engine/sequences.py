"""Sequences and state paths (SPEC 7.2): each entity's events rewritten as ordered steps.

For every entity of a scenario, its events in the sequence table are ordered by time. The first
event keeps its time; each later event follows the one before it by a gap drawn from its step's
range. A step can also set columns, so a status column can walk through a path of states.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from synthgen.engine.generators import Values
from synthgen.spec.evaluator import to_internal, to_output
from synthgen.spec.expressions import ExprType
from synthgen.spec.generators import Generator
from synthgen.spec.shapes import SequenceSpec

Rows = NDArray[np.int64]
Draw = Callable[[Generator, Rows], Values]
"""Draws a generator's values for some rows of the table."""


def step_of_rank(sequence: SequenceSpec, ranks: Rows) -> Rows:
    """The step of each event from its position in its entity's ordered events.

    Events past the last step (an entity with more events than the steps list) stay in the last.
    """
    bounds = np.cumsum([s.count for s in sequence.steps])
    found = np.searchsorted(bounds, ranks, side="right")
    return np.minimum(found, len(sequence.steps) - 1).astype(np.int64)


def order_events(entities: Values, times: NDArray[np.float64]) -> tuple[Rows, Rows]:
    """Positions sorted by entity, then time, and each sorted event's rank within its entity."""
    keys = np.unique(entities.astype(str), return_inverse=True)[1]
    order = np.lexsort((times, keys)).astype(np.int64)
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    first = np.repeat(starts, np.diff(np.r_[starts, order.size]))
    return order, (np.arange(order.size) - first).astype(np.int64)


@dataclass(frozen=True)
class SequenceDraw:
    """What applying a sequence needs besides the data: the time column's type and the stream."""

    time_type: ExprType
    rng: np.random.Generator
    draw: Draw


def sequence_times(
    sequence: SequenceSpec, entities: Values, times: NDArray[np.float64], rng: np.random.Generator
) -> tuple[Rows, Rows, NDArray[np.float64]]:
    """Sorted positions, the step of each sorted event, and its new time (seconds)."""
    known = times[~np.isnan(times)]
    times = np.where(np.isnan(times), float(np.min(known)) if known.size else 0.0, times)
    order, ranks = order_events(entities, times)
    steps = step_of_rank(sequence, ranks)
    low = np.array([s.gap_min_seconds for s in sequence.steps])[steps]
    high = np.array([s.gap_max_seconds for s in sequence.steps])[steps]
    gaps = np.where(ranks == 0, 0.0, rng.uniform(low, high))
    elapsed = np.cumsum(gaps)
    first = np.arange(ranks.size) - ranks
    return order, steps, times[order][first] + elapsed - elapsed[first]


def apply_sequence(
    sequence: SequenceSpec, values: dict[str, Values], rows: Rows, how: SequenceDraw
) -> set[str]:
    """Rewrite event times and step columns on `rows`; returns the columns it changed."""
    if rows.size == 0:
        return set()
    times = to_internal(values[sequence.time_column][rows], how.time_type)
    entities = values[sequence.entity_key][rows]
    order, steps, new_times = sequence_times(sequence, entities, times, how.rng)
    placed = rows[order]
    stamped = to_output(new_times, how.time_type)
    values[sequence.time_column] = _set(values[sequence.time_column], placed, stamped)
    changed = {sequence.time_column}
    for index, step in enumerate(sequence.steps):
        members = placed[steps == index]
        for setting in step.columns:
            if members.size:
                drawn = how.draw(setting.generator, members)
                values[setting.column] = _set(values[setting.column], members, drawn)
            changed.add(setting.column)
    return changed


def _set(values: Values, rows: Rows, new: Values) -> Values:
    out = values.astype(object)
    out[rows] = new
    return out
