"""Key structure: children per parent row."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.spec.models import PerParentRows


def expected_children(rule: PerParentRows, parents: int, *, use_target: bool) -> int:
    """Rows the child table should get: the target when used, else parents x mean, rounded."""
    if use_target and rule.target_rows is not None:
        return rule.target_rows
    return round(parents * rule.mean_children)


def children_per_parent(
    rule: PerParentRows, parents: int, rng: np.random.Generator, *, use_target: bool
) -> NDArray[np.int64]:
    """Stratified draws from the profiled distribution, then nudged to the exact expected total."""
    target = expected_children(rule, parents, use_target=use_target)
    if parents == 0:
        if target:
            raise EngineError(f"{target} child rows requested but {rule.parent_table} is empty")
        return np.zeros(0, dtype=np.int64)
    children = np.array([c.children for c in rule.counts], dtype=np.int64)
    weights = np.array([c.weight for c in rule.counts], dtype=np.float64)
    cumulative = np.cumsum(weights / weights.sum())
    strata = (rng.permutation(parents) + rng.random(parents)) / parents
    counts = children[
        np.minimum(np.searchsorted(cumulative, strata, side="right"), children.size - 1)
    ]
    diff = target - int(counts.sum())
    if diff > 0:
        np.add.at(counts, rng.choice(parents, size=diff, replace=diff > parents), 1)
    elif diff < 0:
        owners = np.repeat(np.arange(parents), counts)
        np.subtract.at(counts, owners[rng.choice(owners.size, size=-diff, replace=False)], 1)
    return counts
