"""Distribution fitting (SPEC 8.3 C): candidates ranked by AIC, each checked with a KS test.

Closed-form maximum-likelihood fits (normal, log-normal, exponential, uniform) always run.
The numerical fits (gamma, Weibull, beta, Pareto, t) run when `full` is set; wide tables skip
them because thousands of columns times nine iterative fits would dominate profiling time.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from synthgen.profiling.models import CountFits, DistributionFit

FloatArray = NDArray[np.float64]
Frozen = stats.rv_continuous  # the distribution family; params come separately
MIN_DISTINCT_TO_FIT = 10


def _aic(dist: Frozen, params: tuple[float, ...], x: FloatArray) -> float | None:
    """Akaike information criterion; None when the likelihood is not finite."""
    loglik = float(np.sum(dist.logpdf(x, *params)))
    return 2 * len(params) - 2 * loglik if math.isfinite(loglik) else None


def _with_ks(
    name: str, dist: Frozen, params: tuple[float, ...], aic: float, x: FloatArray
) -> DistributionFit:
    ks = stats.kstest(x, lambda v: dist.cdf(v, *params))
    return DistributionFit(
        name=name,
        params=[float(p) for p in params],
        aic=aic,
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
    )


def _closed_form(x: FloatArray) -> list[tuple[str, Frozen, tuple[float, ...]]]:
    lo, hi = float(np.min(x)), float(np.max(x))
    found: list[tuple[str, Frozen, tuple[float, ...]]] = [
        ("norm", stats.norm, (float(np.mean(x)), float(np.std(x)) or 1e-12)),
        ("uniform", stats.uniform, (lo, (hi - lo) or 1e-12)),
    ]
    if lo > 0:
        logs = np.log(x)
        found.append(
            (
                "lognorm",
                stats.lognorm,
                (float(np.std(logs)) or 1e-12, 0.0, float(np.exp(np.mean(logs)))),
            )
        )
    if lo >= 0:
        found.append(("expon", stats.expon, (0.0, float(np.mean(x)) or 1e-12)))
    return found


Fixed = tuple[float | None, float | None]
NUMERICAL: dict[str, tuple[Frozen, Callable[[FloatArray], bool], Fixed]] = {
    "gamma": (stats.gamma, lambda x: bool(np.min(x) > 0), (0.0, None)),
    "weibull_min": (stats.weibull_min, lambda x: bool(np.min(x) > 0), (0.0, None)),
    "beta": (stats.beta, lambda x: bool(np.min(x) > 0 and np.max(x) < 1), (0.0, 1.0)),
    "pareto": (stats.pareto, lambda x: bool(np.min(x) > 0), (0.0, None)),
    "t": (stats.t, lambda x: True, (None, None)),
}
"""Iteratively fitted candidates: family, the support it needs, fixed (location, scale)."""


def _fit(dist: Frozen, x: FloatArray, fixed: Fixed) -> tuple[float, ...]:
    floc, fscale = fixed
    if floc is None:
        params = dist.fit(x)
    elif fscale is None:
        params = dist.fit(x, floc=floc)
    else:
        params = dist.fit(x, floc=floc, fscale=fscale)
    return tuple(float(p) for p in params)


def _numerical(x: FloatArray) -> list[tuple[str, Frozen, tuple[float, ...]]]:
    found: list[tuple[str, Frozen, tuple[float, ...]]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # SciPy warns on slow convergence; the AIC ranks the fit
        for name, (dist, supports, fixed) in NUMERICAL.items():
            if not supports(x):
                continue
            try:
                params = _fit(dist, x, fixed)
            except (RuntimeError, ValueError):
                continue  # this family cannot describe the data; the other candidates still rank
            found.append((name, dist, params))
    return found


def fit_distributions(x: FloatArray, *, full: bool) -> list[DistributionFit]:
    """Candidates ranked by AIC, each with a KS test. Fewer than ten distinct values: none.

    Without `full` only closed-form candidates are fitted and only the best one is KS-tested
    and returned, which keeps wide tables fast.
    """
    if np.unique(x).size < MIN_DISTINCT_TO_FIT:
        return []
    candidates = _closed_form(x) + (_numerical(x) if full else [])
    scored = [
        (aic, name, dist, params)
        for name, dist, params in candidates
        if (aic := _aic(dist, params, x)) is not None
    ]
    scored.sort(key=lambda item: item[0])
    keep = scored if full else scored[:1]
    return [_with_ks(name, dist, params, aic, x) for aic, name, dist, params in keep]


def fit_counts(x: FloatArray) -> CountFits | None:
    """Poisson and method-of-moments negative binomial, for non-negative integers only."""
    if np.min(x) < 0 or not np.all(np.mod(x, 1) == 0):
        return None
    mean, var = float(np.mean(x)), float(np.var(x))
    poisson_ll = float(np.sum(stats.poisson.logpmf(x, mean))) if mean > 0 else 0.0
    nbinom: tuple[float | None, float | None, float | None] = (None, None, None)
    if var > mean > 0:
        p = mean / var
        n = mean * p / (1 - p)
        nbinom = (n, p, 4.0 - 2 * float(np.sum(stats.nbinom.logpmf(x, n, p))))
    return CountFits(
        poisson_lambda=mean,
        poisson_aic=2.0 - 2 * poisson_ll,
        nbinom_n=nbinom[0],
        nbinom_p=nbinom[1],
        nbinom_aic=nbinom[2],
    )
