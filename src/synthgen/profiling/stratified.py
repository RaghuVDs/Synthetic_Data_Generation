"""Label-stratified statistics (SPEC 8.3 I): class separation per column.

`class_separation` serves the planner's on-demand tool with the KS statistic and total
variation distance. `numeric_separation` and `categorical_separation` serve the stored profile
with `ks_2samp`, `wasserstein_distance` and `jensenshannon`. All run from 0 (same).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.spatial import distance

from synthgen.domain.base import StrictModel
from synthgen.profiling.label_models import ColumnSeparation


class Separation(StrictModel):
    """How differently one column behaves for the two classes."""

    column: str
    measure: str
    score: float


def ks_statistic(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Largest distance between the two empirical CDFs; 0 when either side is empty."""
    if a.size == 0 or b.size == 0:
        return 0.0
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def total_variation(a: Sequence[object], b: Sequence[object]) -> float:
    """Half the L1 distance between two category mixes."""
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    return 0.5 * sum(abs(ca[k] / len(a) - cb[k] / len(b)) for k in set(ca) | set(cb))


def class_separation(
    positive: NDArray[np.bool_],
    numeric: Mapping[str, NDArray[np.float64]],
    categorical: Mapping[str, Sequence[object]],
) -> list[Separation]:
    """Columns ranked by separation between the positive class and the rest."""
    scores = [
        Separation(
            column=name,
            measure="ks",
            score=ks_statistic(
                values[positive & ~np.isnan(values)], values[~positive & ~np.isnan(values)]
            ),
        )
        for name, values in numeric.items()
    ]
    for name, values in categorical.items():
        pos = [v for v, p in zip(values, positive.tolist(), strict=True) if p]
        neg = [v for v, p in zip(values, positive.tolist(), strict=True) if not p]
        scores.append(
            Separation(column=name, measure="total_variation", score=total_variation(pos, neg))
        )
    return sorted(scores, key=lambda s: (-s.score, s.column))


def jensen_shannon(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Base-2 Jensen-Shannon distance of two count vectors; 0 when either is empty."""
    if not p.sum() or not q.sum():
        return 0.0
    value = float(distance.jensenshannon(p, q, base=2))
    return value if np.isfinite(value) else 0.0


def numeric_separation(
    column: str, kind: str, positive: NDArray[np.float64], negative: NDArray[np.float64], bins: int
) -> ColumnSeparation:
    """KS, scaled Wasserstein and histogram Jensen-Shannon between two non-empty samples."""
    pooled = np.concatenate([positive, negative])
    edges = np.histogram_bin_edges(pooled, bins=bins)
    js = jensen_shannon(
        np.histogram(positive, edges)[0].astype(np.float64),
        np.histogram(negative, edges)[0].astype(np.float64),
    )
    ks = stats.ks_2samp(positive, negative)
    scale = float(np.std(pooled))
    wasserstein = float(stats.wasserstein_distance(positive, negative)) / scale if scale else 0.0
    return ColumnSeparation(
        column=column,
        kind=kind,
        ks=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        wasserstein=wasserstein,
        jensen_shannon=js,
        score=max(float(ks.statistic), js),
    )


def categorical_separation(
    column: str, positive: NDArray[np.int64], negative: NDArray[np.int64], levels: int
) -> ColumnSeparation:
    """Jensen-Shannon distance between the category mixes of two code arrays."""
    js = jensen_shannon(
        np.bincount(positive, minlength=levels).astype(np.float64),
        np.bincount(negative, minlength=levels).astype(np.float64),
    )
    return ColumnSeparation(
        column=column,
        kind="categorical",
        ks=None,
        ks_pvalue=None,
        wasserstein=None,
        jensen_shannon=js,
        score=js,
    )
