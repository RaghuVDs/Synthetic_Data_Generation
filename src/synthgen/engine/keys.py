"""Key structure: children per parent and the seeded entity hash used for scenario assignment."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.engine.generators import Values
from synthgen.spec.models import PerParentRows

_MIX_1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


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


def _splitmix(values: NDArray[np.uint64]) -> NDArray[np.uint64]:
    with np.errstate(over="ignore"):
        z = values + _GOLDEN
        z = (z ^ (z >> np.uint64(30))) * _MIX_1
        z = (z ^ (z >> np.uint64(27))) * _MIX_2
        return z ^ (z >> np.uint64(31))


def entity_hash(keys: Values, seed: int) -> NDArray[np.uint64]:
    """A seeded 64-bit hash per key, stable across runs, chunks and worker counts."""
    salt = _splitmix(np.array([seed], dtype=np.uint64))[0]
    if np.issubdtype(keys.dtype, np.integer):
        return _splitmix(keys.astype(np.int64).view(np.uint64) ^ salt)
    seed_bytes = int(seed).to_bytes(8, "little")
    return np.array(
        [
            int.from_bytes(
                hashlib.blake2b(str(k).encode(), digest_size=8, key=seed_bytes).digest(), "little"
            )
            for k in keys
        ],
        dtype=np.uint64,
    )
