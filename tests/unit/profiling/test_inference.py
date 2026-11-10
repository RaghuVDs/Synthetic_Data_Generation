from __future__ import annotations

import pytest

from synthgen.domain.metadata import Sensitivity
from synthgen.profiling.inference import (
    infer_type,
    luhn_valid,
    mask,
    mask_shares,
    semantic_type,
    sensitivity_for,
)
from synthgen.profiling.models import InferredType


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], InferredType.EMPTY),
        ([0, 1, 1, 0], InferredType.BOOLEAN),
        ([True, False], InferredType.BOOLEAN),
        ([1, 2, 3], InferredType.INTEGER),
        ([1, 0], InferredType.BOOLEAN),
        ([1, 2.5], InferredType.FLOAT),
        (["2024-01-31", "1999-12-01"], InferredType.DATE),
        (["2024-01-31 10:00:00", "2024-01-31T11:30"], InferredType.DATETIME),
        (["2024-01-31", "2024-01-31 10:00:00"], InferredType.TEXT),
        (["a", 1], InferredType.TEXT),
        (["hello"], InferredType.TEXT),
    ],
)
def test_infer_type(values: list[object], expected: InferredType) -> None:
    assert infer_type(values) is expected  # type: ignore[arg-type]


def test_luhn() -> None:
    assert luhn_valid("79927398713")
    assert not luhn_valid("79927398710")


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["a.b@x.com", "c@y.org"], "email"),
        (["4111111111111111", "5500005555555559"], "card_number"),
        (["+1-555-123-4567", "+44 20 7946 0958"], "phone"),
        (["hello", "world"], None),
    ],
)
def test_semantic_type(values: list[str], expected: str | None) -> None:
    assert semantic_type(values, InferredType.TEXT) == expected


def test_semantic_type_for_dates_and_numbers() -> None:
    assert semantic_type(["2024-01-01"], InferredType.DATE) == "date"
    assert semantic_type([1, 2], InferredType.INTEGER) is None


def test_sensitivity_rules() -> None:
    assert sensitivity_for(Sensitivity.PUBLIC, "email") is Sensitivity.PUBLIC
    assert sensitivity_for(None, "email") is Sensitivity.SENSITIVE
    assert sensitivity_for(None, None) is Sensitivity.PUBLIC


def test_masks() -> None:
    assert mask("Ab-12 x.") == "AA-99 A."
    shares = mask_shares(["ab1", "cd2", "e-3"], limit=5)
    assert [(s.mask, round(s.share, 2)) for s in shares] == [("AA9", 0.67), ("A-9", 0.33)]
    assert mask_shares([], limit=5) == []
