from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest

from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import read_manifest
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import GenerationSpec, ScenarioSpec
from synthgen.spec.validator import validate_spec
from synthgen.validation.integrity import check_integrity

from .test_runner import BIG, GENERATION, _col, make_spec


def _spec(*defects: dict[str, Any]) -> GenerationSpec:
    data = make_spec(customers=2_000).model_dump()
    data["tables"][1]["columns"].append(
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
            },
        )
    )
    overlay = ScenarioSpec.model_validate({**BIG, "defects": list(defects)})
    return merge_overlays(GenerationSpec.model_validate(data), [overlay])


def _defect(table: str, column: str, kind: str, rate: float = 0.5) -> dict[str, Any]:
    return {"table": table, "column": column, "kind": kind, "rate": rate}


ALL = [
    _defect("customers", "code", "null"),
    _defect("customers", "id", "duplicate_key", 0.1),
    _defect("orders", "amount", "out_of_range"),
    _defect("orders", "placed", "bad_format"),
    _defect("orders", "customer_id", "orphan_key", 0.2),
]


def test_defects_break_values_and_label_them(tmp_path: Path) -> None:
    spec = _spec(*ALL)
    assert validate_spec(spec, GENERATION, allow_defects=True) == []
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    customers = pq.read_table(tmp_path / "data" / "customers" / "part-00000.parquet").to_pylist()
    orders = pq.read_table(tmp_path / "data" / "orders" / "part-00000.parquet").to_pylist()
    labelled = [r for r in customers if r["_defect"]]
    assert labelled and all(r["_scenario"] == "big" for r in labelled)
    assert all(r["code"] is None for r in customers if "null:code" in (r["_defect"] or ""))
    ids = [r["id"] for r in customers]
    assert len(ids) > len(set(ids))
    assert all(r["code"] is not None for r in customers if not r["_defect"])
    broken_amounts = [r for r in orders if "out_of_range:amount" in (r["_defect"] or "")]
    assert broken_amounts and all(r["amount"] > 1_000 for r in broken_amounts)
    for row in orders:
        if "bad_format:placed" in (row["_defect"] or ""):
            assert re.fullmatch(r"\d{2}/\d{2}/\d{4}", row["placed"])
    orphans = {r["customer_id"] for r in orders if "orphan_key:customer_id" in (r["_defect"] or "")}
    assert orphans and not orphans & set(ids)
    assert all(r["_defect"] is None for r in orders if r["_scenario"] == "typical")
    failures = [
        c for c in check_integrity(spec, tmp_path, read_manifest(tmp_path)) if c.status == "fail"
    ]
    assert failures == []  # labelled defect rows are exempt from integrity


def test_defects_are_opt_in() -> None:
    errors = validate_spec(_spec(ALL[0]), GENERATION)
    assert [e.origin for e in errors] == ["overlay"]
    assert "include_invalid" in errors[0].message


@pytest.mark.parametrize(
    ("defect", "fragment"),
    [
        (_defect("orders", "amount", "bad_format"), "does not suit"),
        (_defect("customers", "tier", "out_of_range"), "does not suit"),
        (_defect("orders", "amount", "duplicate_key"), "key or unique"),
        (_defect("orders", "amount", "orphan_key"), "foreign key"),
        (_defect("orders", "ghost", "null"), "unknown column"),
        (_defect("ghosts", "x", "null"), "unknown table"),
    ],
)
def test_defect_problems(defect: dict[str, Any], fragment: str) -> None:
    errors = validate_spec(_spec(defect), GENERATION, allow_defects=True)
    assert errors and fragment in errors[0].message


def test_defects_must_sit_on_or_below_the_entity() -> None:
    spec = _spec(_defect("customers", "code", "null"))
    moved = spec.model_copy(
        update={"scenarios": [spec.scenarios[0].model_copy(update={"entity": "orders"})]}
    )
    errors = validate_spec(moved, GENERATION, allow_defects=True)
    assert any("neither the scenario entity" in e.message for e in errors)
