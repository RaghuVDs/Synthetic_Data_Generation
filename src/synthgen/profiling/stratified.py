"""Label-stratified statistics (SPEC 8.3 I): class balance and class separation per column.

Numeric columns use the two-sample KS statistic; categorical columns use total variation
distance between the class mixes. Both run from 0 (same) to 1 (fully separated).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from synthgen.domain.base import StrictModel


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
