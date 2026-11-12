"""Scenario assignment and overlays (SPEC 8.5, PLAN A2).

Each scenario gets exactly max(1, round(share x eligible)) entities per table; the chunk plan
splits that total over chunks. Inside a chunk the rows are chosen by a seeded hash rank of the
entity key. Rows already carrying a scenario (inherited from a parent) are not eligible again.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.engine.generators import Values
from synthgen.engine.hashing import entity_hash
from synthgen.spec.models import TYPICAL, ScenarioSpec

ObjectArray = NDArray[np.object_]


def scenario_counts(shares: Sequence[float], eligible: int) -> list[int]:
    """Rows per scenario: at least one for any positive share; never more than are eligible."""
    counts = [max(1, round(share * eligible)) if share > 0 else 0 for share in shares]
    if sum(counts) > eligible:
        raise EngineError(f"scenarios need {sum(counts)} rows but only {eligible} are eligible")
    return counts


def assign_scenarios(
    keys: Values,
    current: ObjectArray,
    scenarios: Sequence[ScenarioSpec],
    seed: int,
    counts: Sequence[int],
) -> ObjectArray:
    """Give `counts[i]` eligible rows to scenario i, lowest hash first, in spec order.

    The counts come from the chunk plan, which splits each scenario's exact total over chunks.
    """
    result = current.copy()
    if not scenarios:
        return result
    eligible = np.flatnonzero(current == TYPICAL)
    if sum(counts) > eligible.size:
        raise EngineError(
            f"scenarios need {sum(counts)} rows but only {eligible.size} are eligible"
        )
    ranked = eligible[np.argsort(entity_hash(keys[eligible], seed), kind="stable")]
    start = 0
    for scenario, count in zip(scenarios, counts, strict=True):
        result[ranked[start : start + count]] = scenario.name
        start += count
    return result


def categories(assigned: ObjectArray, scenarios: Sequence[ScenarioSpec]) -> ObjectArray:
    """`_category` for each row: the scenario's category, or typical."""
    lookup = {s.name: str(s.category) for s in scenarios}
    return np.array([lookup.get(name, TYPICAL) for name in assigned], dtype=object)
