from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from synthgen.config import ProfilingSettings, load_settings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.metadata import Sensitivity
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import part_path
from synthgen.metadata.graph import build_table_set
from synthgen.metadata.provider import ColumnDoc, SourceDoc, TableDoc, YamlMetadataProvider
from synthgen.profiling.profiler import profile_table
from synthgen.spec.baseline import BaselineInputs, compile_baseline
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import GenerationSpec, ScenarioSpec
from synthgen.validation.output import ValidationInputs
from synthgen.validation.report import validate_output

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = load_settings(ROOT / "config" / "settings.yaml", {})
NAMES = ["Ann", "Bob", "Cy", "Di", "Ed"]


@pytest.fixture(scope="module")
def people_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("privacy") / "hr.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            "CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE,"
            " first_name TEXT, city TEXT, note TEXT, score REAL)"
        )
        conn.executemany(
            "INSERT INTO people VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    i,
                    f"secret{i}@corp.test",
                    NAMES[i % 5],
                    "Rare" if i < 5 else ("A" if i % 2 else "B"),
                    ["ok", "fine", "late"][i % 3],
                    float(i % 97),
                )
                for i in range(500)
            ],
        )
        conn.commit()
    return path


@pytest.fixture(scope="module")
def source(people_db: Path) -> Iterator[tuple[SqliteConnector, YamlMetadataProvider]]:
    doc = SourceDoc(
        source="hr",
        tables={
            "people": TableDoc(
                columns={
                    "email": ColumnDoc(sensitivity=Sensitivity.SENSITIVE),
                    "first_name": ColumnDoc(sensitivity=Sensitivity.SENSITIVE),
                    "city": ColumnDoc(sensitivity=Sensitivity.INTERNAL),
                }
            )
        },
    )
    connector = SqliteConnector(people_db)
    yield connector, YamlMetadataProvider(doc, connector)
    connector.close()


def _setup(
    source: tuple[SqliteConnector, YamlMetadataProvider],
    profiling: ProfilingSettings,
    out: Path,
    scenarios: list[dict[str, Any]] | None = None,
) -> tuple[GenerationSpec, ValidationInputs]:
    connector, provider = source
    profile = profile_table(
        "hr",
        connector,
        provider.table("people"),
        [],
        parents={},
        settings=profiling,
        seed=profiling.sample_seed,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    spec = compile_baseline(
        BaselineInputs(
            table_set=build_table_set(["people"], ["people"], []),
            tables={"people": provider.table("people")},
            profiles={"people": profile},
            profile_versions={"people": "v"},
            semantic_map=None,
            row_targets={"people": 400},
            seed=2,
        ),
        SETTINGS.generation,
    )
    spec = merge_overlays(spec, [ScenarioSpec.model_validate(s) for s in scenarios or []])
    generate_dataset(spec, out, RunOptions(), SETTINGS.generation)
    inputs = ValidationInputs(
        spec,
        RunOptions(),
        SETTINGS.validation,
        {"people": profile},
        plan=None,
        source=connector,
        profile_seed=profiling.sample_seed,
        min_value_count=profiling.internal_min_count,
    )
    return spec, inputs


def _privacy(inputs: ValidationInputs, out: Path) -> dict[tuple[str, str | None], CheckResult]:
    results = validate_output(inputs, out).results
    return {(r.check, r.column): r for r in results if r.family is CheckFamily.PRIVACY}


def _overwrite(out: Path, values: dict[str, Any]) -> None:
    path = part_path(out, "people", 0)
    table = pq.read_table(path)
    for column, value in values.items():
        cells = table.column(column).to_pylist()
        cells[0] = value
        index = table.schema.get_field_index(column)
        table = table.set_column(
            index, column, pa.array(cells, type=table.schema.field(column).type)
        )
    pq.write_table(table, path)


def test_clean_output_passes(
    source: tuple[SqliteConnector, YamlMetadataProvider],
    profiling_settings: ProfilingSettings,
    tmp_path: Path,
) -> None:
    _, inputs = _setup(source, profiling_settings, tmp_path)
    found = _privacy(inputs, tmp_path)
    assert found[("exact_match", "email")].status is CheckStatus.PASS
    assert found[("exact_match", "first_name")].status is CheckStatus.INFO  # small vocabulary
    assert found[("suppressed_values", "city")].status is CheckStatus.PASS
    assert found[("row_hash", None)].status is CheckStatus.PASS


@pytest.mark.parametrize(
    ("change", "key"),
    [
        ({"email": "secret3@corp.test"}, ("exact_match", "email")),
        ({"city": "Rare"}, ("suppressed_values", "city")),
        (
            {
                "email": "secret8@corp.test",
                "first_name": "Di",
                "city": "B",
                "note": "late",
                "score": 8.0,
            },
            ("row_hash", None),
        ),
    ],
)
def test_leaks_fail(
    source: tuple[SqliteConnector, YamlMetadataProvider],
    profiling_settings: ProfilingSettings,
    tmp_path: Path,
    change: dict[str, Any],
    key: tuple[str, str | None],
) -> None:
    _, inputs = _setup(source, profiling_settings, tmp_path)
    _overwrite(tmp_path, change)
    found = _privacy(inputs, tmp_path)
    assert found[key].status is CheckStatus.FAIL
    assert found[key].value == 1.0


def test_text_pool_values_are_screened(
    source: tuple[SqliteConnector, YamlMetadataProvider],
    profiling_settings: ProfilingSettings,
    tmp_path: Path,
) -> None:
    leaky = {
        "name": "chatty",
        "category": "target_scenario",
        "share": 0.1,
        "unit": "entity",
        "entity": "people",
        "labels": [],
        "overrides": [
            {
                "table": "people",
                "column": "note",
                "generator": {"type": "text_pool", "values": ["hello", "secret1@corp.test"]},
            }
        ],
        "correlations": [],
        "cardinality": [],
        "sequences": [],
        "defects": [],
    }
    _, inputs = _setup(source, profiling_settings, tmp_path, [leaky])
    result = _privacy(inputs, tmp_path)[("text_pool_values", None)]
    assert result.status is CheckStatus.FAIL
    assert result.scenario == "chatty"


def test_without_a_source_privacy_is_info(
    source: tuple[SqliteConnector, YamlMetadataProvider],
    profiling_settings: ProfilingSettings,
    tmp_path: Path,
) -> None:
    _, inputs = _setup(source, profiling_settings, tmp_path)
    found = _privacy(
        ValidationInputs(inputs.spec, inputs.options, inputs.settings, inputs.profiles, None, None),
        tmp_path,
    )
    assert [r.status for r in found.values()] == [CheckStatus.INFO]


def test_leaks_in_scenario_rows_are_traced(
    source: tuple[SqliteConnector, YamlMetadataProvider],
    profiling_settings: ProfilingSettings,
    tmp_path: Path,
) -> None:
    everyone = {
        "name": "all",
        "category": "rare",
        "share": 1.0,
        "unit": "entity",
        "entity": "people",
        "labels": [],
        "overrides": [],
        "correlations": [],
        "cardinality": [],
        "sequences": [],
        "defects": [],
    }
    _, inputs = _setup(source, profiling_settings, tmp_path, [everyone])
    _overwrite(tmp_path, {"email": "secret3@corp.test"})
    result = _privacy(inputs, tmp_path)[("exact_match", "email")]
    assert (result.status, result.scenario) == (CheckStatus.FAIL, "all")
