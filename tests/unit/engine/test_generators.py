from __future__ import annotations

import re
import uuid
from datetime import date
from typing import Any

import numpy as np
import pytest
from pydantic import TypeAdapter

from synthgen.domain.errors import EngineError
from synthgen.engine.generators import GenContext, SemanticPools, Values, generate
from synthgen.engine.generators.text import with_counter
from synthgen.spec.evaluator import Column
from synthgen.spec.expressions import ExprType
from synthgen.spec.generators import Generator, SemanticKind

PARENT = np.array([10, 20, 30], dtype=np.int64)


def _lookup(table: str, column: str) -> Values:
    return PARENT if (table, column) == ("parents", "id") else np.array([], dtype=np.int64)


POOLS = SemanticPools(7, 200, "en_US", (date(1950, 1, 1), date(2000, 12, 31)))
CTX = GenContext(row_offset=5, seed=7, parent_keys=_lookup, pools=POOLS, frame={})


def draw(
    spec: dict[str, Any], n: int = 2_000, seed: int = 1, frame: dict[str, Column] | None = None
) -> Values:
    gen = TypeAdapter(Generator).validate_python(spec)
    ctx = CTX if frame is None else GenContext(5, 7, _lookup, POOLS, frame)
    return generate(gen, n, np.random.default_rng(seed), ctx)


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
            "exponent": None,
            "hot_keys": None,
            "hot_share": None,
            "group_size": None,
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
                "exponent": None,
                "hot_keys": None,
                "hot_share": None,
                "group_size": None,
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
    values = draw(
        {"type": "pattern", "masks": ["AA-99", ""], "weights": [0.9, 0.1], "check_digit": None},
        n=500,
    )
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
        {"type": "pattern", "masks": ["A9A"], "weights": [1.0], "check_digit": None},
    ],
)
def test_same_seed_same_values(spec: dict[str, Any]) -> None:
    assert draw(spec, seed=4).tolist() == draw(spec, seed=4).tolist()
    assert draw(spec, seed=4).tolist() != draw(spec, seed=5).tolist()


def _fk(distribution: str, n: int = 20_000, **params: Any) -> Values:
    keys = {"exponent": None, "hot_keys": None, "hot_share": None, "group_size": None} | params
    many = np.arange(1, 1_001, dtype=np.int64)
    gen = TypeAdapter(Generator).validate_python(
        {
            "type": "foreign_key",
            "parent_table": "p",
            "parent_column": "id",
            "distribution": distribution,
            **keys,
        }
    )
    ctx = GenContext(0, 7, lambda _t, _c: many, POOLS, {})
    return generate(gen, n, np.random.default_rng(3), ctx)


def test_zipf_foreign_keys_concentrate_on_a_few_parents() -> None:
    _, counts = np.unique(_fk("zipf", exponent=1.2), return_counts=True)
    top = np.sort(counts)[::-1]
    assert top[:10].sum() > 0.4 * 20_000
    uniform_top = np.sort(np.unique(_fk("uniform"), return_counts=True)[1])[::-1]
    assert uniform_top[:10].sum() < 0.03 * 20_000


def test_hot_keys_take_their_share_and_are_stable_across_streams() -> None:
    values = _fk("weighted", hot_keys=3, hot_share=0.5)
    keys, counts = np.unique(values, return_counts=True)
    hot = set(keys[np.argsort(counts)[::-1][:3]].tolist())
    assert np.isin(values, list(hot)).mean() == pytest.approx(0.5, abs=0.02)
    again = _fk("weighted", n=500, hot_keys=3, hot_share=1.0)
    assert set(again.tolist()) == hot


def test_shared_group_uses_only_its_keys() -> None:
    assert len(set(_fk("shared_group", group_size=4).tolist())) == 4


@pytest.mark.parametrize(
    ("distribution", "params"),
    [("uniform", {"exponent": 1.0}), ("zipf", {}), ("weighted", {"hot_keys": 2})],
)
def test_foreign_key_parameters_match_the_distribution(
    distribution: str, params: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="foreign key"):
        _fk(distribution, **params)


def test_distribution_follows_the_family_and_clips() -> None:
    values = draw(
        {
            "type": "distribution",
            "family": "lognorm",
            "params": [0.5, 0.0, 20.0],
            "minimum": None,
            "maximum": 60.0,
            "decimals": 2,
        },
        n=20_000,
    )
    assert np.median(values) == pytest.approx(20.0, rel=0.05)
    assert values.max() == 60.0
    assert values.min() > 0


def test_count_families_give_whole_numbers() -> None:
    values = draw(
        {
            "type": "distribution",
            "family": "poisson",
            "params": [3.0],
            "minimum": None,
            "maximum": None,
            "decimals": 0,
        },
        n=20_000,
    )
    assert np.all(values == np.round(values))
    assert values.mean() == pytest.approx(3.0, abs=0.1)


def test_distribution_parameter_count_is_checked() -> None:
    with pytest.raises(ValueError, match="takes 2 parameters"):
        draw(
            {
                "type": "distribution",
                "family": "norm",
                "params": [1.0],
                "minimum": None,
                "maximum": None,
                "decimals": None,
            }
        )


def test_luhn_check_digit_makes_valid_numbers() -> None:
    values = draw(
        {
            "type": "pattern",
            "masks": ["9999 9999 9999 9999"],
            "weights": [1.0],
            "check_digit": "luhn",
        },
        n=300,
    )

    def valid(text: str) -> bool:
        digits = [int(c) for c in text if c.isdigit()][::-1]
        total = sum(
            d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits)
        )
        return total % 10 == 0

    assert all(valid(v) for v in values)


def test_semantic_values_come_from_a_seeded_pool() -> None:
    emails = draw({"type": "semantic", "kind": "email"}, n=500)
    assert all(re.fullmatch(r"[^@\s]+@[^@\s]+\.\w+", v) for v in emails)
    births = draw({"type": "semantic", "kind": "date_of_birth"}, n=500)
    assert all("1950-01-01" <= v <= "2000-12-31" for v in births)
    other = SemanticPools(7, 200, "en_US", (date(1950, 1, 1), date(2000, 12, 31)))
    same = other.pool(SemanticKind.LAST_NAME).tolist()
    assert same == POOLS.pool(SemanticKind.LAST_NAME).tolist()
    changed = SemanticPools(8, 200, "en_US", (date(1950, 1, 1), date(2000, 12, 31)))
    assert changed.pool(SemanticKind.LAST_NAME).tolist() != same


def test_text_pool_and_boundary_pick_listed_values() -> None:
    assert set(draw({"type": "text_pool", "values": ["late", "lost"]}, n=200).tolist()) == {
        "late",
        "lost",
    }
    assert set(draw({"type": "boundary", "points": [0.0, 9999.99]}, n=200).tolist()) == {
        0.0,
        9999.99,
    }


def test_text_pool_rejects_long_values() -> None:
    with pytest.raises(ValueError, match="at most"):
        draw({"type": "text_pool", "values": ["x" * 1001]})


FRAME: dict[str, Column] = {
    "tier": (ExprType.TEXT, np.array(["gold", "basic", "gold", None], dtype=object)),
    "amount": (ExprType.NUMBER, np.array([10.0, 20.0, 30.0, 40.0])),
}


def test_derived_reads_the_frame() -> None:
    values = draw({"type": "derived", "expression": "amount * 2 + 1"}, n=4, frame=FRAME)
    assert values.tolist() == [21.0, 41.0, 61.0, 81.0]


def test_derived_expression_must_parse() -> None:
    with pytest.raises(ValueError, match="does not parse"):
        draw({"type": "derived", "expression": "amount +"})


def test_conditional_picks_a_generator_per_row() -> None:
    values = draw(
        {
            "type": "conditional",
            "column": "tier",
            "cases": [
                {"values": ["gold"], "generator": {"type": "derived", "expression": "amount * 10"}}
            ],
            "otherwise": {"type": "constant", "value": 0},
        },
        n=4,
        frame=FRAME,
    )
    assert values.tolist() == [100.0, 0, 300.0, 0]


def test_unique_semantic_values_get_a_counter() -> None:
    values = np.array(["a@x.com", "a@x.com", "Ann Lee"], dtype=object)
    out = with_counter(values, 100, np.random.default_rng(1)).tolist()
    assert len(set(out[:2])) == 2
    assert re.fullmatch(r"Ann Lee 10[0-2]", out[2])
    assert all(re.fullmatch(r"a\.10[0-2]@x\.com", v) for v in out[:2])
