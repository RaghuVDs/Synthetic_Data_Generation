from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from synthgen.config import ValidationSettings, load_settings
from synthgen.domain.report import CheckFamily, CheckStatus
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import part_path
from synthgen.validation.report import REPORT_FILE, validate_output

from ..engine.test_runner import BIG, make_spec

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def settings() -> ValidationSettings:
    return load_settings(ROOT / "config" / "settings.yaml", {}).validation


def _rewrite(out_dir: Path, table: str, column: str, values: list[object]) -> None:
    path = part_path(out_dir, table, 0)
    data = pq.read_table(path)
    index = data.schema.get_field_index(column)
    data = data.set_column(index, column, pa.array(values, type=data.schema.field(column).type))
    pq.write_table(data, path, compression="zstd")


def test_clean_output_passes(tmp_path: Path, settings: ValidationSettings) -> None:
    spec = make_spec(scenarios=[BIG])
    generate_dataset(spec, tmp_path, RunOptions())
    report = validate_output(spec, tmp_path, RunOptions(), settings)
    assert report.passed, report.failures()
    assert (tmp_path / REPORT_FILE).is_file()
    families = {r.family for r in report.results}
    assert families == {CheckFamily.INTEGRITY, CheckFamily.FIDELITY}


def test_duplicate_primary_key_fails(tmp_path: Path, settings: ValidationSettings) -> None:
    spec = make_spec()
    generate_dataset(spec, tmp_path, RunOptions())
    ids = pq.read_table(part_path(tmp_path, "customers", 0)).column("id").to_pylist()
    _rewrite(tmp_path, "customers", "id", [ids[0], *ids[:-1]])
    failures = validate_output(spec, tmp_path, RunOptions(), settings).failures()
    checks = {(f.check, f.table) for f in failures}
    assert ("primary_key_unique", "customers") in checks
    assert ("foreign_key_resolves", "orders") in checks


def test_failure_in_scenario_rows_is_traced(tmp_path: Path, settings: ValidationSettings) -> None:
    spec = make_spec(scenarios=[BIG])
    generate_dataset(spec, tmp_path, RunOptions())
    orders = pq.read_table(part_path(tmp_path, "orders", 0))
    broken = [
        None if s == "big" else a
        for s, a in zip(
            orders.column("_scenario").to_pylist(), orders.column("amount").to_pylist(), strict=True
        )
    ]
    _rewrite(tmp_path, "orders", "amount", broken)
    [failure] = validate_output(spec, tmp_path, RunOptions(), settings).failures()
    assert failure.check == "not_null"
    assert failure.scenario == "big"


def test_failure_in_typical_rows_is_not_traced(
    tmp_path: Path, settings: ValidationSettings
) -> None:
    spec = make_spec()
    generate_dataset(spec, tmp_path, RunOptions())
    codes = pq.read_table(part_path(tmp_path, "customers", 0)).column("code").to_pylist()
    _rewrite(tmp_path, "customers", "code", [codes[1], *codes[1:]])
    [failure] = validate_output(spec, tmp_path, RunOptions(), settings).failures()
    assert (failure.check, failure.scenario) == ("unique", None)


def test_row_count_failure(tmp_path: Path, settings: ValidationSettings) -> None:
    spec = make_spec(customers=100)
    generate_dataset(spec, tmp_path, RunOptions())
    bigger = make_spec(customers=150)
    report = validate_output(bigger, tmp_path, RunOptions(), settings)
    failures = [f for f in report.failures() if f.check == "row_count"]
    assert [f.table for f in failures] == ["customers"]
    assert failures[0].status is CheckStatus.FAIL


def test_dry_run_targets_use_the_cap(tmp_path: Path, settings: ValidationSettings) -> None:
    spec = make_spec(customers=10_000)
    options = RunOptions(max_root_rows=40)
    generate_dataset(spec, tmp_path, options)
    assert validate_output(spec, tmp_path, options, settings).passed
