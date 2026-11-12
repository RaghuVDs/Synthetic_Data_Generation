"""Key structure: how many child rows each parent row gets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.spec.models import PerParentRows, ScenarioSpec
from synthgen.spec.shapes import ChildCount

Counts = NDArray[np.int64]


def expected_children(rule: PerParentRows, parents: int, *, use_target: bool) -> int:
    """Rows the child table should get: the target when used, else parents x mean, rounded."""
    if use_target and rule.target_rows is not None:
        return rule.target_rows
    return round(parents * rule.mean_children)


def stratified_counts(
    counts: Sequence[ChildCount], parents: int, total: int, rng: np.random.Generator
) -> Counts:
    """Stratified draws from a children-per-parent distribution, nudged to an exact total."""
    if parents == 0:
        if total:
            raise EngineError(f"{total} child rows requested for no parent rows")
        return np.zeros(0, dtype=np.int64)
    children = np.array([c.children for c in counts], dtype=np.int64)
    weights = np.array([c.weight for c in counts], dtype=np.float64)
    cumulative = np.cumsum(weights / weights.sum())
    strata = (rng.permutation(parents) + rng.random(parents)) / parents
    drawn = children[
        np.minimum(np.searchsorted(cumulative, strata, side="right"), children.size - 1)
    ]
    diff = total - int(drawn.sum())
    if diff > 0:
        np.add.at(drawn, rng.choice(parents, size=diff, replace=diff > parents), 1)
    elif diff < 0:
        owners = np.repeat(np.arange(parents), drawn)
        np.subtract.at(drawn, owners[rng.choice(owners.size, size=-diff, replace=False)], 1)
    return drawn


def children_per_parent(
    rule: PerParentRows, parents: int, rng: np.random.Generator, *, use_target: bool
) -> Counts:
    """Children of every parent from the profiled distribution, totalling the expected rows."""
    total = expected_children(rule, parents, use_target=use_target)
    if parents == 0 and total:
        raise EngineError(f"{total} child rows requested but {rule.parent_table} is empty")
    return stratified_counts(rule.counts, parents, total, rng)


def scenario_shape(scenario: ScenarioSpec, table: str) -> list[ChildCount] | None:
    """Children per parent a scenario sets for `table`: its cardinality, else its sequence."""
    explicit = next((c.counts for c in scenario.cardinality if c.table == table), None)
    if explicit is not None:
        return explicit
    sequence = next((s for s in scenario.sequences if s.table == table), None)
    return None if sequence is None else [ChildCount(children=sequence.events, weight=1.0)]


def plan_children(  # noqa: PLR0913 - the row plan needs every one of these
    rule: PerParentRows,
    table: str,
    parent_scenarios: NDArray[np.object_],
    scenarios: Sequence[ScenarioSpec],
    rng: np.random.Generator,
    *,
    use_target: bool,
) -> Counts:
    """Children of every parent. Scenario parents follow their scenario's shape for this table.

    The other parents absorb the difference, so the table still gets the expected total.
    """
    parents = parent_scenarios.size
    shapes = [(s.name, shape) for s in scenarios if (shape := scenario_shape(s, table))]
    if not shapes:
        return children_per_parent(rule, parents, rng, use_target=use_target)
    counts = np.zeros(parents, dtype=np.int64)
    shaped = np.zeros(parents, dtype=bool)
    for name, shape in shapes:
        rows = np.flatnonzero(parent_scenarios == name)
        expected = sum(c.children * c.weight for c in shape) / sum(c.weight for c in shape)
        counts[rows] = stratified_counts(shape, rows.size, round(rows.size * expected), rng)
        shaped[rows] = True
    rest = np.flatnonzero(~shaped)
    total = expected_children(rule, parents, use_target=use_target) - int(counts.sum())
    if rest.size:
        counts[rest] = stratified_counts(rule.counts, rest.size, max(total, 0), rng)
    return counts
