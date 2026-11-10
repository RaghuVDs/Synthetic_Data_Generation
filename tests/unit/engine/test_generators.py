from __future__ import annotations

import re
import uuid
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter

from synthgen.domain.errors import EngineError
from synthgen.engine.generators import GenContext, Values, generate
from synthgen.spec.generators import Generator

PARENT = np.array([10, 20, 30], dtype=np.int64)


def _lookup(table: str, column: str) -> Values:
    return PARENT if (table, column) == ("parents", "id") else np.array([], dtype=np.int64)


CTX = GenContext(row_offset=5, parent_keys=_lookup)


def draw(spec: dict[str, Any], n: int = 2_000, seed: int = 1) -> Values:
    gen = TypeAdapter(Generator).validate_python(spec)
    return generate(gen, n, np.random.default_rng(seed), CTX)


def test_sequence_uses_offset_and_step() -> None:
    assert draw({"type": "sequence", "start": 100, "step": 2}, n=3).tolist() == [110, 112, 114]


def test_uuids_are_valid_version_4() -> None:
    values = draw({"type": "uuid"}, n=50)
    parsed = [uuid.UUID(v) for v in values]
    assert all(u.version == 4 for u in parsed)
    assert len(set(values)) == 50


def test_foreign_key_draws_parent_keys() -> None:
    values = draw(
        {
            "type": "foreign_key",
            "parent_table": "parents",
            "parent_column": "id",
            "distribution": "uniform",
        }
    )
    assert set(values.tolist()) == {10, 20, 30}


def test_foreign_key_without_parents_fails() -> None:
    with pytest.raises(EngineError):
        draw(
            {
                "type": "foreign_key",
                "parent_table": "x",
                "parent_column": "id",
                "distribution": "uniform",
            }
        )


def test_empirical_follows_quantiles() -> None:
    values = draw({"type": "empirical", "quantiles": [0.0, 10.0, 100.0], "decimals": 1}, n=20_000)
    assert np.min(values) >= 0.0 and np.max(values) <= 100.0
    assert np.median(values) == pytest.approx(10.0, abs=1.0)
    assert np.allclose(values, np.round(values, 1))


def test_categorical_follows_weights() -> None:
    values = draw({"type": "categorical", "values": ["a", "b"], "weights": [0.8, 0.2]}, n=20_000)
    assert np.mean(values == "a") == pytest.approx(0.8, abs=0.02)


def test_dates_respect_range_and_weekdays() -> None:
    values = draw(
        {
            "type": "datetime",
            "start": "2024-01-01",
            "end": "2024-01-31",
            "granularity": "date",
            "weekday_weights": [0, 0, 0, 0, 0, 0, 1.0],
            "hour_weights": None,
        },
        n=200,
    )
    assert set(values.tolist()) == {"2024-01-07", "2024-01-14", "2024-01-21", "2024-01-28"}


def test_datetimes_respect_hours_and_bounds() -> None:
    hours = [0.0] * 24
    hours[13] = 1.0
    values = draw(
        {
            "type": "datetime",
            "start": "2024-01-01 00:00:00",
            "end": "2024-01-02 23:59:59",
            "granularity": "datetime",
            "weekday_weights": None,
            "hour_weights": hours,
        },
        n=100,
    )
    assert all(re.fullmatch(r"2024-01-0[12] 13:\d\d:\d\d", v) for v in values)


def test_patterns_follow_masks() -> None:
    values = draw({"type": "pattern", "masks": ["AA-99", ""], "weights": [0.9, 0.1]}, n=500)
    filled = [v for v in values if v]
    assert all(re.fullmatch(r"[a-z]{2}-\d{2}", v) for v in filled)
    assert 0 < sum(v == "" for v in values) < 100


def test_constant_including_null() -> None:
    assert draw({"type": "constant", "value": 3}, n=2).tolist() == [3, 3]
    assert draw({"type": "constant", "value": None}, n=2).tolist() == [None, None]


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "uuid"},
        {"type": "empirical", "quantiles": [1.0, 5.0], "decimals": None},
        {"type": "pattern", "masks": ["A9A"], "weights": [1.0]},
    ],
)
def test_same_seed_same_values(spec: dict[str, Any]) -> None:
    assert draw(spec, seed=4).tolist() == draw(spec, seed=4).tolist()
    assert draw(spec, seed=4).tolist() != draw(spec, seed=5).tolist()
