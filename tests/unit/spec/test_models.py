from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from synthgen.spec.generators import Generator
from synthgen.spec.models import (
    ChildCount,
    GenerationSpec,
    OverlaySet,
    PerParentRows,
)


def _spec() -> dict[str, Any]:
    return {
        "version": 1,
        "seed": 7,
        "profile_versions": [{"table": "people", "version": "abc"}],
        "tables": [
            {
                "name": "people",
                "row_rule": {"kind": "fixed", "rows": 10},
                "columns": [
                    {
                        "name": "id",
                        "type": "integer",
                        "nullable": False,
                        "null_rate": 0.0,
                        "primary_key": True,
                        "unique": False,
                        "sensitive": False,
                        "generator": {"type": "sequence", "start": 1, "step": 1},
                    }
                ],
            }
        ],
        "scenarios": [],
        "output": {"compression": "zstd", "chunk_rows": 1000},
    }


def test_spec_round_trip_and_stable_hash() -> None:
    spec = GenerationSpec.model_validate(_spec())
    again = GenerationSpec.model_validate_json(spec.model_dump_json())
    assert again == spec
    assert again.spec_hash() == spec.spec_hash()
    assert spec.table("people") is not None
    assert spec.table("absent") is None
    people = spec.table("people")
    assert people is not None and people.primary_key is not None


@pytest.mark.parametrize(
    "generator",
    [
        {"type": "categorical", "values": ["a", "b"], "weights": [0.5, 0.4]},
        {"type": "categorical", "values": ["a"], "weights": [0.5, 0.5]},
        {"type": "categorical", "values": ["a", "b"], "weights": [1.5, -0.5]},
        {"type": "empirical", "quantiles": [3.0, 1.0], "decimals": 0},
        {"type": "empirical", "quantiles": [1.0], "decimals": 0},
        {"type": "pattern", "masks": ["AA"], "weights": [0.9], "check_digit": None},
        {"type": "sequence", "start": 1, "step": 0},
        {
            "type": "datetime",
            "start": "2024-02-01",
            "end": "2024-01-01",
            "granularity": "date",
            "weekday_weights": None,
            "hour_weights": None,
        },
        {
            "type": "datetime",
            "start": "2024-01-01",
            "end": "2024-02-01",
            "granularity": "date",
            "weekday_weights": [1.0],
            "hour_weights": None,
        },
        {"type": "unknown"},
    ],
)
def test_invalid_generators_are_rejected(generator: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(Generator).validate_python(generator)


def test_generator_union_picks_by_type() -> None:
    gen = TypeAdapter(Generator).validate_python({"type": "constant", "value": None})
    assert gen.type == "constant"


def test_mean_children() -> None:
    rule = PerParentRows(
        kind="per_parent",
        parent_table="p",
        parent_column="id",
        child_column="p_id",
        counts=[ChildCount(children=0, weight=1), ChildCount(children=3, weight=3)],
        target_rows=None,
    )
    assert rule.mean_children == pytest.approx(2.25)


def test_overlay_schema_is_strict() -> None:
    schema = json.dumps(OverlaySet.model_json_schema())
    assert '"default"' not in schema
    assert '"additionalProperties": true' not in schema
