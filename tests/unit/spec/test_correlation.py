from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from synthgen.spec.correlation import (
    as_matrix,
    is_valid,
    nearest_correlation,
    spearman_to_pearson,
)
from synthgen.spec.models import CorrelationSpec

INVALID = [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]


def test_valid_matrices_are_kept() -> None:
    valid = as_matrix([[1.0, 0.5], [0.5, 1.0]])
    assert is_valid(valid)
    assert nearest_correlation(valid) is valid


def test_invalid_matrix_is_repaired_to_a_close_valid_one() -> None:
    matrix = as_matrix(INVALID)
    assert not is_valid(matrix)
    repaired = nearest_correlation(matrix)
    assert is_valid(repaired)
    assert np.allclose(np.diag(repaired), 1.0)
    assert np.allclose(repaired, repaired.T)
    assert np.sign(repaired[0, 1]) == 1 and np.sign(repaired[0, 2]) == -1
    assert np.abs(repaired - matrix).max() < 0.5


def test_missing_correlations_count_as_zero() -> None:
    assert as_matrix([[1.0, float("nan")], [float("nan"), 1.0]]).tolist() == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]


def test_spearman_to_pearson_keeps_ends_and_sign() -> None:
    out = spearman_to_pearson(np.array([0.0, 1.0, -1.0, 0.5]))
    assert out[:3].tolist() == pytest.approx([0.0, 1.0, -1.0])
    assert out[3] == pytest.approx(2 * np.sin(np.pi / 12))


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"columns": ["a", "a"], "spearman": [[1, 0], [0, 1]]}, "repeat"),
        ({"columns": ["a", "b"], "spearman": [[1, 0]]}, "2 x 2"),
        ({"columns": ["a", "b"], "spearman": [[0.9, 0], [0, 1]]}, "diagonal"),
        ({"columns": ["a", "b"], "spearman": [[1, 0.2], [0.3, 1]]}, "symmetric"),
        ({"columns": ["a", "b"], "spearman": [[1, 1.5], [1.5, 1]]}, "between -1 and 1"),
    ],
)
def test_matrix_shape_is_checked(data: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CorrelationSpec.model_validate(data)
