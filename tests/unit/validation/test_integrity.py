from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from synthgen.domain.report import CheckResult, CheckStatus
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import part_path, read_manifest
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import GenerationSpec, ScenarioSpec
from synthgen.validation.integrity import check_integrity
from synthgen.validation.integrity_rules import luhn_valid
from synthgen.validation.output import OutputData

from ..engine.test_runner import BIG, GENERATION, _col, make_spec


def _spec(constraints: list[dict[str, Any]] | None = None) -> GenerationSpec:
    data = make_spec(customers=300).model_dump()
    customers, orders = data["tables"]
    customers["columns"] += [
        _col(
            "card",
            "text",
            {
                "type": "pattern",
                "masks": ["9999 9999 9999 9999"],
                "weights": [1.0],
                "check_digit": "luhn",
            },
        ),
        _col("email", "text", {"type": "semantic", "kind": "email"}),
        _col(
            "joined",
            "date",
            {
                "type": "datetime",
                "start": "2023-01-01",
                "end": "2023-12-31",
                "granularity": "date",
                "weekday_weights": None,
                "hour_weights": None,
                "after": None,
            },
        ),
    ]
    orders["columns"] += [
        _col(
            "placed",
            "date",
            {
                "type": "datetime",
                "start": "2024-01-01",
                "end": "2024-12-31",
                "granularity": "date",
                "weekday_weights": None,
                "hour_weights": None,
                "after": None,
            },
        ),
        _col("low", "float", {"type": "empirical", "quantiles": [0.0, 10.0], "decimals": 2}),
        _col("high", "float", {"type": "empirical", "quantiles": [20.0, 30.0], "decimals": 2}),
    ]
    orders["constraints"] = constraints or [
        {"kind": "order", "lower": "customers.joined", "upper": "placed"},
        {"kind": "order", "lower": "low", "upper": "high"},
        {"kind": "range", "column": "amount", "minimum": 1.0, "maximum": 100.0},
        {"kind": "unique", "columns": ["customer_id", "id"]},
        {"kind": "not_null", "column": "flag"},
        {"kind": "expression", "expression": "high - low >= 10"},
    ]
    return GenerationSpec.model_validate(data)


def _run(spec: GenerationSpec, out: Path) -> list[CheckResult]:
    return check_integrity(spec, OutputData(out, read_manifest(out)))


def _failures(spec: GenerationSpec, out: Path) -> dict[tuple[str, str | None], CheckResult]:
    return {(r.check, r.column): r for r in _run(spec, out) if r.status is CheckStatus.FAIL}


def _edit(out: Path, table: str, column: str, change: Any) -> None:
    path = part_path(out, table, 0)
    data = pq.read_table(path)
    values = data.column(column).to_pylist()
    values[0] = change(values[0]) if callable(change) else change
    index = data.schema.get_field_index(column)
    data = data.set_column(index, column, pa.array(values, type=data.schema.field(column).type))
    pq.write_table(data, path, compression="zstd")


def test_luhn() -> None:
    assert luhn_valid("4539 1488 0343 6467")
    assert not luhn_valid("4539 1488 0343 6468")
    assert not luhn_valid("")


def test_clean_output_passes_every_rule(tmp_path: Path) -> None:
    spec = _spec()
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    results = _run(spec, tmp_path)
    assert [r for r in results if r.status is CheckStatus.FAIL] == []
    checks = {r.check for r in results}
    assert {"format", "constraint_order", "constraint_range", "constraint_unique"} <= checks
    assert {"constraint_not_null", "constraint_expression"} <= checks


@pytest.mark.parametrize(
    ("table", "column", "change", "key"),
    [
        ("customers", "code", "not-a-code", ("format", "code")),
        ("customers", "card", lambda v: v[:-1] + str((int(v[-1]) + 1) % 10), ("format", "card")),
        ("customers", "email", "nobody", ("format", "email")),
        ("orders", "placed", "04/03/2024", ("format", "placed")),
        ("orders", "placed", "2000-01-01", ("constraint_order", "customers.joined,placed")),
        ("orders", "low", 50.0, ("constraint_order", "high,low")),
        ("orders", "amount", 500.0, ("constraint_range", "amount")),
        ("orders", "flag", None, ("constraint_not_null", "flag")),
        ("orders", "high", 25.0, ("constraint_expression", "high,low")),
    ],
)
def test_each_rule_catches_a_broken_value(
    tmp_path: Path, table: str, column: str, change: Any, key: tuple[str, str]
) -> None:
    spec = _spec()
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    if key == ("constraint_expression", "high,low"):
        _edit(tmp_path, "orders", "low", 20.0)
    _edit(tmp_path, table, column, change)
    failures = _failures(spec, tmp_path)
    assert key in failures, sorted(failures)
    assert failures[key].value == 1.0


def test_unique_constraint_over_columns(tmp_path: Path) -> None:
    generate_dataset(_spec(), tmp_path, RunOptions(), GENERATION)
    strict = _spec([{"kind": "unique", "columns": ["customer_id"]}])
    assert ("constraint_unique", "customer_id") in _failures(strict, tmp_path)


def test_overlay_rows_follow_their_own_format(tmp_path: Path) -> None:
    override = {
        "table": "customers",
        "column": "code",
        "generator": {"type": "pattern", "masks": ["99"], "weights": [1.0], "check_digit": None},
    }
    scenario = ScenarioSpec.model_validate({**BIG, "labels": [], "overrides": [override]})
    spec = merge_overlays(_spec(), [scenario])
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    assert ("format", "code") not in _failures(spec, tmp_path)


def test_broken_scenario_rows_are_traced(tmp_path: Path) -> None:
    override = {
        "table": "orders",
        "column": "amount",
        "generator": {"type": "constant", "value": 0.0},
    }
    scenario = ScenarioSpec.model_validate({**BIG, "labels": [], "overrides": [override]})
    spec = merge_overlays(_spec(), [scenario])
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    failure = _failures(spec, tmp_path)[("constraint_range", "amount")]
    assert failure.scenario == "big"
