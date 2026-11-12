from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from synthgen.config import load_settings
from synthgen.domain.plan import Assertion, ScenarioPlan
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import part_path
from synthgen.spec.expressions import ExprType
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import ScenarioSpec
from synthgen.validation.output import ValidationInputs
from synthgen.validation.report import validate_output
from synthgen.validation.signature import AssertionRows, outcomes

from ..engine.test_runner import BIG, GENERATION, ROOT, make_spec

VALIDATION = load_settings(ROOT / "config" / "settings.yaml", {}).validation
N, T = ExprType.NUMBER, ExprType.DATETIME


def _assert(metric: str, op: str, value: Any, window: float | None = None) -> Assertion:
    return Assertion(
        table="t", column="c", metric=metric, operator=op, value=value, window_hours=window
    )


def _rows(values: list[Any], kind: ExprType = N, **kw: Any) -> AssertionRows:
    return AssertionRows(
        values=np.array(values, dtype=object),
        kind=kind,
        entities=None if kw.get("entities") is None else np.array(kw["entities"], dtype=object),
        times=None if kw.get("times") is None else np.array(kw["times"], dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("assertion", "rows", "expected"),
    [
        (_assert("value", "ge", 5), _rows([4, 5, None]), [False, True, False]),
        (_assert("value", "in", [1, "x"]), _rows([1, 2]), [True, False]),
        (_assert("value", "not_in", [1.0]), _rows([1, 2]), [False, True]),
        (_assert("value", "eq", True), _rows([1, 0]), [True, False]),
        (
            _assert("value", "lt", "2024-06-01 00:00:00"),
            _rows(["2024-05-31 23:00:00", "2024-06-02 00:00:00"], T),
            [True, False],
        ),
        (_assert("value", "eq", "a"), _rows(["a", "b"], ExprType.TEXT), [True, False]),
        (_assert("value", "gt", 3), _rows(["a", "1"], ExprType.TEXT), [True, False]),  # as text
        (
            _assert("hour_of_day", "lt", 6),
            _rows(["2024-01-01 03:10:00", "2024-01-01 13:00:00"], T),
            [True, False],
        ),
        (_assert("count", "ge", 2), _rows([1, 1, 1], entities=["a", "a", "b"]), [True, False]),
        (_assert("sum", "eq", 3), _rows([1, 2, 5], entities=["a", "a", "b"]), [True, False]),
        (_assert("mean", "le", 2), _rows([1, 2, 5], entities=["a", "a", "b"]), [True, False]),
        (_assert("max", "ge", 5), _rows([1, 2, 5], entities=["a", "a", "b"]), [False, True]),
        (_assert("min", "le", 1), _rows([1, 2, 5], entities=["a", "a", "b"]), [True, False]),
        (_assert("distinct_count", "eq", 1), _rows([7, 7, 5]), [True, True, True]),
        (
            _assert("count", "ge", 3, window=1.0),
            _rows([1] * 4, entities=["a"] * 4, times=[0, 600, 1200, 99_000]),
            [True],
        ),
        (
            _assert("count", "ge", 3, window=0.1),
            _rows([1] * 4, entities=["a"] * 4, times=[0, 600, 1200, 99_000]),
            [False],
        ),
        (
            _assert("inter_arrival_hours", "le", 0.5),
            _rows([1, 1, 1], entities=["a", "a", "b"], times=[0, 900, 0]),
            [True],
        ),
    ],
)
def test_outcomes(assertion: Assertion, rows: AssertionRows, expected: list[bool]) -> None:
    found = outcomes(assertion, rows)
    assert found is not None and found.tolist() == expected


def test_metrics_that_cannot_apply() -> None:
    assert outcomes(_assert("hour_of_day", "lt", 6), _rows([1])) is None
    assert outcomes(_assert("inter_arrival_hours", "le", 1), _rows([1], entities=["a"])) is None


def _planned(name: str, category: str, signature: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "description": "d",
        "rationale": "r",
        "share": 0.1,
        "unit": "entity",
        "tables": ["customers", "orders"],
        "primitives": ["value_shift"],
        "levers": [],
        "signature": [
            {"table": "orders", "column": "amount", "window_hours": None} | a for a in signature
        ],
        "template_ref": None,
    }


TARGET = [{"metric": "value", "operator": "ge", "value": 1_000.0}]


def _run(
    tmp_path: Path, scenarios: list[dict[str, Any]], plan: list[dict[str, Any]]
) -> list[CheckResult]:
    spec = merge_overlays(
        make_spec(customers=600), [ScenarioSpec.model_validate(s) for s in scenarios]
    )
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    inputs = ValidationInputs(
        spec,
        RunOptions(),
        VALIDATION,
        profiles={},
        plan=ScenarioPlan.model_validate({"scenarios": plan, "assumptions": []}),
        source=None,
    )
    return [
        r for r in validate_output(inputs, tmp_path).results if r.family is CheckFamily.SCENARIO
    ]


def _status(results: list[CheckResult], check: str, scenario: str | None) -> CheckStatus:
    return next(r.status for r in results if r.check == check and r.scenario == scenario)


def test_a_good_scenario_passes_every_check(tmp_path: Path) -> None:
    results = _run(tmp_path, [BIG], [_planned("big", "target_scenario", TARGET)])
    assert [r for r in results if r.status is CheckStatus.FAIL] == []
    checks = {r.check for r in results if r.status is CheckStatus.PASS}
    assert checks == {"present", "share", "labels", "signature", "separation", "label_counts"}


def test_unmet_signatures_and_weak_separation_fail(tmp_path: Path) -> None:
    weak = {
        **BIG,
        "overrides": [
            {
                "table": "orders",
                "column": "amount",
                "generator": {"type": "empirical", "quantiles": [1.0, 100.0], "decimals": 2},
            }
        ],
    }
    results = _run(tmp_path, [weak], [_planned("big", "target_scenario", TARGET)])
    assert _status(results, "signature", "big") is CheckStatus.FAIL
    assert _status(results, "separation", "big") is CheckStatus.FAIL


def test_wrong_labels_fail(tmp_path: Path) -> None:
    assert _status(_run(tmp_path, [BIG], []), "labels", "big") is CheckStatus.PASS
    path = part_path(tmp_path, "orders", 0)
    table = pq.read_table(path)
    index = table.schema.get_field_index("flag")
    table = table.set_column(index, "flag", pa.array([0] * table.num_rows, type=pa.int64()))
    pq.write_table(table, path)
    spec = merge_overlays(make_spec(customers=600), [ScenarioSpec.model_validate(BIG)])
    inputs = ValidationInputs(spec, RunOptions(), VALIDATION, {}, None, None)
    assert _status(list(validate_output(inputs, tmp_path).results), "labels", "big") is (
        CheckStatus.FAIL
    )


def test_near_misses_must_not_match_a_target(tmp_path: Path) -> None:
    near = {**BIG, "name": "near", "category": "near_miss", "share": 0.05, "labels": []}
    results = _run(
        tmp_path,
        [BIG, near],
        [_planned("big", "target_scenario", TARGET), _planned("near", "near_miss", [])],
    )
    failing = [r for r in results if r.check == "near_miss"]
    assert failing and failing[0].status is CheckStatus.FAIL
    assert failing[0].scenario == "near" and (failing[0].value or 0) > 0
