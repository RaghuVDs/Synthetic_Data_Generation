"""The committed golden set is well formed and names only real tables and columns (T7.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from synthgen.domain.errors import ConfigError
from synthgen.domain.intent import TestType
from synthgen.evaluation.golden import GoldenSet, load_golden

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden"
METADATA = ROOT / "config" / "metadata"


@pytest.fixture(scope="module")
def golden() -> GoldenSet:
    return load_golden(GOLDEN)


def _columns(source: str) -> dict[str, set[str]]:
    doc = yaml.safe_load((METADATA / f"{source}.yaml").read_text())
    return {name: set(table["columns"]) for name, table in doc["tables"].items()}


def test_size_sources_and_test_types(golden: GoldenSet) -> None:
    assert 15 <= len(golden.requests) <= 20
    assert {r.source for r in golden.requests} == {"finance", "orders"}
    assert {r.expected.test_type for r in golden.requests} == set(TestType)
    for source in ("finance", "orders"):
        assert {r.expected.test_type for r in golden.requests if r.source == source} == set(
            TestType
        )


def test_requests_name_real_tables_and_columns(golden: GoldenSet) -> None:
    for request in golden.requests:
        schema = _columns(request.source)
        assert set(request.tables) <= set(schema), request.name
        assert request.target_table in request.tables, request.name
        for must in request.expected.must_have:
            assert set(must.tables) <= set(request.tables), (request.name, must.what)
            reachable = set().union(*(schema[t] for t in must.tables or request.tables))
            assert set(must.columns) <= reachable, (request.name, must.what)


def test_roles_label_every_column(golden: GoldenSet) -> None:
    for source in ("finance", "orders"):
        labels = golden.roles[source]
        assert {t: set(c) for t, c in labels.items()} == _columns(source)
        assert all(roles for table in labels.values() for roles in table.values())


def test_select_filters_and_rejects_unknown_names(golden: GoldenSet) -> None:
    orders = golden.select([], ["orders"])
    assert {r.source for r in orders.requests} == {"orders"}
    one = golden.select(["orders_churn_model"], [])
    assert [r.name for r in one.requests] == ["orders_churn_model"]
    with pytest.raises(ConfigError, match="no golden request named"):
        golden.select(["nope"], [])
    with pytest.raises(ConfigError, match="matches the filter"):
        golden.select(["orders_churn_model"], ["finance"])


def test_empty_or_malformed_sets_are_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no golden requests"):
        load_golden(tmp_path)
    (tmp_path / "requests").mkdir()
    (tmp_path / "requests" / "bad.yaml").write_text("- just a list\n")
    with pytest.raises(ConfigError, match="not a YAML mapping"):
        load_golden(tmp_path)
