"""Effect-size statistics for validation (SPEC 8.6). Pure: arrays in, one number out."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial.distance import jensenshannon

Floats = NDArray[np.float64]
MONTHS, WEEKDAYS, HOURS = 12, 7, 24
WASSERSTEIN_POINTS = 10_000
"""Points at which the two quantile functions are compared."""
EPOCH_WEEKDAY = 3
"""1970-01-01 was a Thursday; Monday is 0."""


def ks_statistic(sample: Floats, reference: Floats) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (largest gap between the two CDFs)."""
    return float(stats.ks_2samp(sample, reference).statistic)


def wasserstein_over_iqr(sample: Floats, quantiles: Floats) -> float | None:
    """Earth mover's distance to the stored quantiles' distribution, over its IQR; None if IQR is 0.

    The quantiles define a distribution that is linear between them (what the generator samples),
    so the distance is the mean gap between the two quantile functions. Treating the quantiles as
    point masses would misplace the tail mass: a source fails against its own quantiles that way.
    """
    grid = np.linspace(0.0, 1.0, quantiles.size)
    q1, q3 = np.interp([0.25, 0.75], grid, quantiles)
    iqr = float(q3 - q1)
    if iqr <= 0 or sample.size == 0:
        return None
    u = (np.arange(WASSERSTEIN_POINTS) + 0.5) / WASSERSTEIN_POINTS
    gap = np.abs(np.quantile(sample, u) - np.interp(u, grid, quantiles))
    return float(gap.mean()) / iqr


def js_distance(observed: Sequence[float], expected: Sequence[float]) -> float:
    """Jensen-Shannon distance (base 2, 0 to 1) between two count or share vectors."""
    p = np.asarray(observed, dtype=np.float64)
    q = np.asarray(expected, dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return 0.0 if p.sum() == q.sum() else 1.0
    return float(np.nan_to_num(jensenshannon(p / p.sum(), q / q.sum(), base=2)))


def category_js(values: Sequence[Any], reference: Mapping[str, float]) -> float:  # Any: cells
    """JS distance over the reference's categories; unseen synthetic values form one more bucket."""
    counts = Counter(_label(v) for v in values)
    keys = list(reference)
    observed = [counts.get(k, 0) for k in keys] + [
        sum(c for k, c in counts.items() if k not in reference)
    ]
    return js_distance(observed, [reference[k] for k in keys] + [0.0])


def _label(value: Any) -> str:  # Any: a cell of any type
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def discrete_ks(sample: Sequence[int], reference: Mapping[int, float]) -> float:
    """KS statistic between an integer sample and a weighted integer distribution."""
    support = sorted(set(reference) | set(sample))
    counts = Counter(sample)
    total_s = max(len(sample), 1)
    total_r = sum(reference.values()) or 1.0
    cdf_s = np.cumsum([counts.get(k, 0) / total_s for k in support])
    cdf_r = np.cumsum([reference.get(k, 0.0) / total_r for k in support])
    return float(np.max(np.abs(cdf_s - cdf_r))) if support else 0.0


def time_mixes(seconds: Floats, with_hours: bool) -> dict[str, list[int]]:
    """Counts by month, weekday and (for date-times) hour of day."""
    stamps = seconds.astype("datetime64[s]")
    months = (stamps.astype("datetime64[M]").astype(np.int64) % MONTHS).astype(np.int64)
    days = np.floor(seconds / 86_400).astype(np.int64)
    found = {
        "month": np.bincount(months, minlength=MONTHS).tolist(),
        "weekday": np.bincount((days + EPOCH_WEEKDAY) % WEEKDAYS, minlength=WEEKDAYS).tolist(),
    }
    if with_hours:
        hours = ((seconds % 86_400) // 3_600).astype(np.int64)
        found["hour"] = np.bincount(hours, minlength=HOURS).tolist()
    return found


def largest_spearman_gap(sample: Floats, reference: Sequence[Sequence[float]]) -> float:
    """Largest absolute difference between the sample's Spearman matrix and the reference one.

    `sample` has one column per matrix row; constant columns count as zero correlation.
    """
    if sample.shape[0] < 3 or sample.shape[1] < 2:  # noqa: PLR2004 - too few rows to correlate
        return 0.0
    rho = np.nan_to_num(np.asarray(stats.spearmanr(sample).statistic, dtype=np.float64), nan=0.0)
    observed = rho if rho.ndim == 2 else np.array([[1.0, float(rho)], [float(rho), 1.0]])  # noqa: PLR2004
    expected = np.nan_to_num(np.asarray(reference, dtype=np.float64), nan=0.0)
    return float(np.max(np.abs(observed - expected)))
