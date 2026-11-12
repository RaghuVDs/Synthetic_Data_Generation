"""Gaussian copula (SPEC 8.5): correlated uniforms whose Spearman matrix is the one asked for.

Spearman's rho becomes the normal-scale correlation 2 sin(pi rho / 6); correlated normals are
mapped to uniforms with the normal CDF, and each column's own generator maps its uniforms to
values, so every column keeps its marginal distribution.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from synthgen.spec.correlation import as_matrix, nearest_correlation, spearman_to_pearson


def correlated_uniforms(
    spearman: Sequence[Sequence[float]], n: int, rng: np.random.Generator
) -> NDArray[np.float64]:
    """N rows of uniforms, one column per matrix row, with the given Spearman correlations."""
    pearson = nearest_correlation(spearman_to_pearson(as_matrix(spearman)))
    values, vectors = np.linalg.eigh(pearson)
    factor = vectors * np.sqrt(np.maximum(values, 0.0))
    normals = rng.standard_normal((n, pearson.shape[0])) @ factor.T
    return np.asarray(stats.norm.cdf(normals), dtype=np.float64)
