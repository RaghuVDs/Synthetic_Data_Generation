"""Foreign keys drawn from the parent's generated keys: uniform, Zipf, hot keys or a shared group.

Which keys are hot (or form the shared group) is fixed by a seeded hash of the keys, so every
chunk and worker agrees on them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.errors import EngineError
from synthgen.engine.generators.context import GenContext, Values
from synthgen.engine.hashing import derived_seed, ranked
from synthgen.spec.generators import ForeignKeyGen


def foreign_keys(gen: ForeignKeyGen, n: int, rng: np.random.Generator, ctx: GenContext) -> Values:
    """N keys of the parent table, following the generator's distribution."""
    keys = ctx.parent_keys(gen.parent_table, gen.parent_column)
    if keys.size == 0:
        raise EngineError(f"no keys in {gen.parent_table}.{gen.parent_column} to reference")
    return keys[_positions(gen, keys, n, rng, ctx.seed)]


def _positions(
    gen: ForeignKeyGen, keys: Values, n: int, rng: np.random.Generator, seed: int
) -> NDArray[np.int64]:
    size = keys.size
    if gen.distribution == "uniform":
        return rng.integers(0, size, n)
    order = ranked(keys, derived_seed(seed, "keys", gen.parent_table, gen.distribution))
    if gen.distribution == "zipf" and gen.exponent is not None:
        weights = 1.0 / np.arange(1, size + 1, dtype=np.float64) ** gen.exponent
        cumulative = np.cumsum(weights / weights.sum())
        picks = np.searchsorted(cumulative, rng.random(n), side="right")
        return order[np.minimum(picks, size - 1)]
    if gen.distribution == "weighted" and gen.hot_keys is not None and gen.hot_share is not None:
        hot = order[: min(gen.hot_keys, size)]
        is_hot = rng.random(n) < gen.hot_share
        return np.where(is_hot, hot[rng.integers(0, hot.size, n)], rng.integers(0, size, n))
    if gen.distribution == "shared_group" and gen.group_size is not None:
        group = order[: min(gen.group_size, size)]
        return group[rng.integers(0, group.size, n)]
    raise EngineError(f"{gen.distribution} foreign key is missing its parameters")
