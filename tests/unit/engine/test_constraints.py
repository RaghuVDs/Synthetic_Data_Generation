from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from pydantic import TypeAdapter

from synthgen.domain.errors import EngineError
from synthgen.engine.constraints import enforce
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.spec.constraints import Constraint
from synthgen.spec.expressions import ExprType
from synthgen.spec.models import GenerationSpec
from synthgen.spec.validator import validate_spec

from .test_runner import GENERATION, _col, make_spec

N, D = ExprType.NUMBER, ExprType.DATE


def _apply(constraints: list[dict[str, Any]], values: dict[str, Any], **kw: Any) -> dict[str, Any]:
    parsed = [TypeAdapter(Constraint).validate_python(c) for c in constraints]
    arrays = {k: np.array(v, dtype=object) for k, v in values.items()}
    types = kw.get("types", dict.fromkeys(values, N))
    redraws: list[tuple[str, list[int]]] = []

    def redraw(name: str, rows: Any) -> None:
        redraws.append((name, rows.tolist()))
        if kw.get("frozen"):
            return  # a column the engine may not redraw (a key or a computed column)
        updated = arrays[name].copy()
        updated[rows] = kw.get("fresh", 0)
        arrays[name] = updated

    enforce(parsed, arrays, types, kw.get("parents", {}), redraw)
    return {k: v.tolist() for k, v in arrays.items()} | {"redraws": redraws}


def test_order_swaps_broken_rows() -> None:
    out = _apply(
        [{"kind": "order", "lower": "a", "upper": "b"}],
        {"a": [1.0, 5.0, None], "b": [2.0, 3.0, 1.0]},
    )
    assert out["a"] == [1.0, 3.0, None]
    assert out["b"] == [2.0, 5.0, 1.0]


def test_order_against_a_parent_redraws_the_child() -> None:
    parents = {"p.opened": (D, np.array(["2024-01-10", "2024-01-10"], dtype=object))}
    out = _apply(
        [{"kind": "order", "lower": "p.opened", "upper": "d"}],
        {"d": ["2024-01-05", "2024-01-20"]},
        types={"d": D},
        parents=parents,
        fresh="2024-03-01",
    )
    assert out["d"] == ["2024-03-01", "2024-01-20"]
    assert out["redraws"] == [("d", [0])]


def test_order_against_a_parent_lifts_what_cannot_be_redrawn() -> None:
    parents = {"p.opened": (D, np.array(["2024-01-10", "2024-01-10"], dtype=object))}
    out = _apply(
        [{"kind": "order", "lower": "p.opened", "upper": "d"}],
        {"d": ["2024-01-05", "2024-01-20"]},
        types={"d": D},
        parents=parents,
        frozen=True,
    )
    assert out["d"] == ["2024-01-10", "2024-01-20"]
    stamps = _apply(
        [{"kind": "order", "lower": "p.opened", "upper": "t"}],
        {"t": ["2024-01-05 08:30:00"]},
        types={"t": ExprType.DATETIME},
        parents={"p.opened": (ExprType.DATETIME, np.array(["2024-01-10 12:00:00"], dtype=object))},
        frozen=True,
    )
    assert stamps["t"] == ["2024-01-11 08:30:00"]


def test_range_clips_numbers_and_dates() -> None:
    out = _apply(
        [
            {"kind": "range", "column": "a", "minimum": 0.0, "maximum": 10.0},
            {"kind": "range", "column": "d", "minimum": None, "maximum": "2024-12-31"},
        ],
        {"a": [-5.0, 3.0, 50.0], "d": ["2025-06-01", "2024-01-01", None]},
        types={"a": N, "d": D},
    )
    assert out["a"] == [0.0, 3.0, 10.0]
    assert out["d"] == ["2024-12-31", "2024-01-01", None]


def test_unique_redraws_repeated_combinations() -> None:
    out = _apply(
        [{"kind": "unique", "columns": ["a", "b"]}],
        {"a": [1, 1, 2], "b": [7, 7, 7]},
        fresh=8,
    )
    assert out["redraws"] == [("b", [1])]
    assert out["b"] == [7, 8, 7]


def test_expression_redraws_until_true_or_fails() -> None:
    out = _apply(
        [{"kind": "expression", "expression": "a + b <= 10"}],
        {"a": [1.0, 9.0], "b": [2.0, 9.0]},
    )
    assert out["a"] == [1.0, 0] and out["b"] == [2.0, 0]
    with pytest.raises(EngineError, match="still break"):
        _apply([{"kind": "expression", "expression": "a > 100"}], {"a": [1.0]})
    with pytest.raises(EngineError, match="could not make"):
        _apply([{"kind": "unique", "columns": ["a"]}], {"a": [1, 1]}, fresh=1)


def _spec_with(constraints: list[dict[str, Any]], **orders: Any) -> GenerationSpec:
    data = make_spec(customers=3_000).model_dump()
    orders_table = data["tables"][1]
    orders_table["columns"] += [
        _col("low", "float", {"type": "empirical", "quantiles": [0.0, 100.0], "decimals": 2}),
        _col("high", "float", {"type": "empirical", "quantiles": [0.0, 100.0], "decimals": 2}),
        _col(
            "note",
            "text",
            {"type": "constant", "value": "x"},
            nullable=True,
            null_rate=0.5,
        ),
    ]
    orders_table["constraints"] = constraints
    return GenerationSpec.model_validate(data | orders)


def test_engine_enforces_constraints_before_nulls(tmp_path: Path) -> None:
    spec = _spec_with(
        [
            {"kind": "order", "lower": "low", "upper": "high"},
            {"kind": "not_null", "column": "note"},
            {"kind": "range", "column": "amount", "minimum": 10.0, "maximum": None},
        ]
    )
    assert validate_spec(spec, GENERATION) == []
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    rows = pq.read_table(tmp_path / "data" / "orders" / "part-00000.parquet").to_pylist()
    assert all(r["low"] <= r["high"] for r in rows)
    assert all(r["note"] == "x" for r in rows)
    assert all(r["amount"] >= 10.0 for r in rows)
    assert {r["_defect"] for r in rows} == {None}


@pytest.mark.parametrize(
    ("constraint", "fragment"),
    [
        ({"kind": "order", "lower": "low", "upper": "ghost"}, "unknown column ghost"),
        ({"kind": "order", "lower": "customers.code", "upper": "high"}, "cannot be ordered"),
        ({"kind": "order", "lower": "high", "upper": "customers.id"}, "upper side"),
        ({"kind": "range", "column": "amount", "minimum": "low", "maximum": None}, "range bound"),
        ({"kind": "range", "column": "ghost", "minimum": None, "maximum": None}, "unknown column"),
        ({"kind": "unique", "columns": ["low", "ghost"]}, "unknown column orders.ghost"),
        ({"kind": "not_null", "column": "ghost"}, "unknown column"),
        ({"kind": "expression", "expression": "low + 1"}, "true/false"),
        ({"kind": "expression", "expression": "ghost > 1"}, "unknown column ghost"),
        ({"kind": "order", "lower": "customers.id", "upper": "high"}, ""),
    ],
)
def test_constraint_problems_are_baseline_errors(constraint: dict[str, Any], fragment: str) -> None:
    errors = validate_spec(_spec_with([constraint]), GENERATION)
    if not fragment:
        assert errors == []
        return
    assert [e.origin for e in errors] == ["baseline"]
    assert fragment in errors[0].message


def test_interacting_constraints_settle() -> None:
    parents = {"p.opened": (D, np.array(["2024-01-10", "2024-01-10"], dtype=object))}
    out = _apply(
        [
            {"kind": "order", "lower": "issued", "upper": "expiry"},
            {"kind": "order", "lower": "p.opened", "upper": "issued"},
        ],
        {"issued": ["2024-01-01", "2024-02-01"], "expiry": ["2024-01-12", "2024-03-01"]},
        types={"issued": D, "expiry": D},
        parents=parents,
        frozen=True,
    )
    issued, expiry = out["issued"], out["expiry"]
    assert all("2024-01-10" <= i <= e for i, e in zip(issued, expiry, strict=True))


def test_contradictory_constraints_fail() -> None:
    with pytest.raises(EngineError, match="contradict"):
        _apply(
            [
                {"kind": "range", "column": "a", "minimum": 5.0, "maximum": None},
                {"kind": "range", "column": "b", "minimum": None, "maximum": 1.0},
                {"kind": "order", "lower": "a", "upper": "b"},
            ],
            {"a": [6.0], "b": [0.0]},
        )
