"""Correlation matrices for the Gaussian copula: checks and repair to the nearest valid matrix.

A valid matrix is symmetric, has a unit diagonal and no negative eigenvalue. Repair uses
Higham's alternating projections (2002), which keeps the result as close as possible to the
requested matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float64]
EIGEN_TOLERANCE = 1e-9
"""Smallest eigenvalue still accepted as non-negative (rounding noise)."""
REPAIR_ITERATIONS = 200
REPAIR_TOLERANCE = 1e-10


def as_matrix(rows: Sequence[Sequence[float]]) -> Matrix:
    """A float matrix from nested lists; missing correlations (NaN) count as zero."""
    return np.nan_to_num(np.asarray(rows, dtype=np.float64), nan=0.0)


def is_valid(matrix: Matrix) -> bool:
    """Symmetric, unit diagonal and positive semi-definite."""
    return (
        bool(np.allclose(matrix, matrix.T))
        and bool(np.allclose(np.diag(matrix), 1.0))
        and float(np.min(np.linalg.eigvalsh(matrix))) >= -EIGEN_TOLERANCE
    )


def nearest_correlation(matrix: Matrix) -> Matrix:
    """The nearest valid correlation matrix (Frobenius norm); valid input comes back unchanged."""
    if is_valid(matrix):
        return matrix
    target = (matrix + matrix.T) / 2.0
    y = target.copy()
    correction = np.zeros_like(target)
    for _ in range(REPAIR_ITERATIONS):
        r = y - correction
        values, vectors = np.linalg.eigh(r)
        x = (vectors * np.maximum(values, 0.0)) @ vectors.T
        correction = x - r
        previous = y
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        if np.linalg.norm(y - previous) <= REPAIR_TOLERANCE * np.linalg.norm(y):
            break
    values, vectors = np.linalg.eigh((y + y.T) / 2.0)
    y = (vectors * np.maximum(values, EIGEN_TOLERANCE)) @ vectors.T
    scale = 1.0 / np.sqrt(np.diag(y))
    repaired = np.asarray(y * np.outer(scale, scale), dtype=np.float64)
    np.fill_diagonal(repaired, 1.0)
    return (repaired + repaired.T) / 2.0


def spearman_to_pearson(spearman: Matrix) -> Matrix:
    """The normal-scale correlation that gives this Spearman correlation: 2 sin(pi rho / 6)."""
    return np.asarray(2.0 * np.sin(np.pi * spearman / 6.0), dtype=np.float64)
