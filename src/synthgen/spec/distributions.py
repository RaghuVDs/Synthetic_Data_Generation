"""The `distribution` generator: a fitted SciPy family sampled through its inverse CDF."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from synthgen.domain.base import StrictModel


class Family(StrEnum):
    """Fitted families (SciPy parameter order: shapes, then location and scale)."""

    NORM = "norm"
    UNIFORM = "uniform"
    LOGNORM = "lognorm"
    EXPON = "expon"
    GAMMA = "gamma"
    WEIBULL_MIN = "weibull_min"
    BETA = "beta"
    PARETO = "pareto"
    T = "t"
    POISSON = "poisson"
    NBINOM = "nbinom"


PARAMETER_COUNTS = {
    Family.NORM: 2,
    Family.UNIFORM: 2,
    Family.LOGNORM: 3,
    Family.EXPON: 2,
    Family.GAMMA: 3,
    Family.WEIBULL_MIN: 3,
    Family.BETA: 4,
    Family.PARETO: 3,
    Family.T: 3,
    Family.POISSON: 1,
    Family.NBINOM: 2,
}


class DistributionGen(StrictModel):
    """A fitted family sampled by its inverse CDF, clipped to [minimum, maximum], then rounded."""

    type: Literal["distribution"]
    family: Family
    params: list[float]
    minimum: float | None
    maximum: float | None
    decimals: int | None = Field(ge=0)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.params) != PARAMETER_COUNTS[self.family]:
            want = PARAMETER_COUNTS[self.family]
            raise ValueError(f"{self.family} takes {want} parameters, got {len(self.params)}")
        low, high = self.minimum, self.maximum
        if low is not None and high is not None and high < low:
            raise ValueError("distribution maximum is below its minimum")
        return self
