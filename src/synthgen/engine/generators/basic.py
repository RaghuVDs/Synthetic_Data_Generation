"""Generators that need only their spec and the random stream."""

from __future__ import annotations

import numpy as np
from scipy import stats

from synthgen.engine.generators.context import GenContext, Values
from synthgen.spec.distributions import DistributionGen, Family
from synthgen.spec.generators import (
    BoundaryGen,
    CategoricalGen,
    ConstantGen,
    EmpiricalGen,
    SequenceGen,
    TextPoolGen,
    UuidGen,
)

UUID_BYTES = 16
PPF_EPSILON = 1e-9
"""Uniforms are kept this far from 0 and 1 so unbounded families never return infinity."""
FAMILIES: dict[Family, stats.rv_continuous | stats.rv_discrete] = {
    Family.NORM: stats.norm,
    Family.UNIFORM: stats.uniform,
    Family.LOGNORM: stats.lognorm,
    Family.EXPON: stats.expon,
    Family.GAMMA: stats.gamma,
    Family.WEIBULL_MIN: stats.weibull_min,
    Family.BETA: stats.beta,
    Family.PARETO: stats.pareto,
    Family.T: stats.t,
    Family.POISSON: stats.poisson,
    Family.NBINOM: stats.nbinom,
}


def sequence(gen: SequenceGen, n: int, _rng: np.random.Generator, ctx: GenContext) -> Values:
    """Start + step x row number, counting from the chunk's row offset."""
    return gen.start + gen.step * np.arange(ctx.row_offset, ctx.row_offset + n, dtype=np.int64)


def uuids(_gen: UuidGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    """Version-4 UUID text from the seeded stream."""
    raw = rng.integers(0, 256, (n, UUID_BYTES), dtype=np.uint8)
    raw[:, 6] = (raw[:, 6] & 0x0F) | 0x40  # version 4
    raw[:, 8] = (raw[:, 8] & 0x3F) | 0x80  # RFC 4122 variant
    hexed = [row.tobytes().hex() for row in raw]
    return np.array(
        [f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}" for h in hexed], dtype=object
    )


def from_uniform(gen: EmpiricalGen | DistributionGen, u: Values) -> Values:
    """Map uniforms in [0, 1] to values: interpolation on quantiles, or the family's inverse CDF.

    The copula feeds correlated uniforms through here, so marginals stay as the spec says.
    """
    if isinstance(gen, EmpiricalGen):
        grid = np.linspace(0.0, 1.0, len(gen.quantiles))
        values = np.interp(u, grid, np.asarray(gen.quantiles, dtype=np.float64))
    else:
        clipped = np.clip(u, PPF_EPSILON, 1.0 - PPF_EPSILON)
        values = np.asarray(FAMILIES[gen.family].ppf(clipped, *gen.params), dtype=np.float64)
        low = -np.inf if gen.minimum is None else gen.minimum
        high = np.inf if gen.maximum is None else gen.maximum
        values = np.clip(values, low, high)
    return values if gen.decimals is None else np.round(values, gen.decimals)


def quantile_draw(
    gen: EmpiricalGen | DistributionGen, n: int, rng: np.random.Generator, _ctx: GenContext
) -> Values:
    """Independent draws through `from_uniform`."""
    return from_uniform(gen, rng.random(n))


def categorical(gen: CategoricalGen, n: int, rng: np.random.Generator, _ctx: GenContext) -> Values:
    """Listed values drawn with their weights."""
    picks = rng.choice(len(gen.values), size=n, p=np.asarray(gen.weights) / sum(gen.weights))
    return np.asarray(gen.values, dtype=object)[picks]


def uniform_pick(
    gen: BoundaryGen | TextPoolGen, n: int, rng: np.random.Generator, _ctx: GenContext
) -> Values:
    """Boundary points or text-pool values, each equally likely."""
    listed = gen.points if isinstance(gen, BoundaryGen) else gen.values
    return np.asarray(listed, dtype=object)[rng.integers(0, len(listed), n)]


def constant(gen: ConstantGen, n: int, _rng: np.random.Generator, _ctx: GenContext) -> Values:
    """The same value on every row."""
    return np.full(n, gen.value, dtype=object)
