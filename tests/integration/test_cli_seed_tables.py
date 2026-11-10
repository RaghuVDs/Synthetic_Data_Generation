from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from synthgen.app import build_application
from synthgen.cli import app
from synthgen.domain.errors import MetadataError, NotFoundError

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ROOT / "config" / "settings.yaml"
WIDE = {"COLUMNS": "200"}


def env_for(data_dir: Path, **extra: str) -> dict[str, str]:
    return {"SYNTHGEN_PATHS__DATA_DIR": str(data_dir), **WIDE, **extra}


def test_seed_builds_both_sources(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["seed"], env=env_for(tmp_path, SYNTHGEN_SEED__SCALE="0.01"))
    assert result.exit_code == 0, result.output
    assert (tmp_path / "finance.db").is_file()
    assert (tmp_path / "orders.db").is_file()
    assert "transactions" in result.output
    assert "2,000" in result.output  # 1% of 200,000 transactions


def test_tables_lists_tables_and_relationships(seeded_dir: Path) -> None:
    result = CliRunner().invoke(app, ["tables", "--source", "finance"], env=env_for(seeded_dir))
    assert result.exit_code == 0, result.output
    for name in ("customers", "accounts", "cards", "merchants", "transactions"):
        assert name in result.output
    assert "transactions.merchant_id" in result.output
    assert "merchants.merchant_id" in result.output


def test_tables_shows_relationship_found_in_database(seeded_dir: Path) -> None:
    result = CliRunner().invoke(app, ["tables", "--source", "orders"], env=env_for(seeded_dir))
    assert result.exit_code == 0, result.output
    line = next(ln for ln in result.output.splitlines() if "order_items.product_id" in ln)
    assert "pragma" in line


def test_unknown_source_exits_with_error(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["tables", "--source", "nope"], env=env_for(tmp_path))
    assert result.exit_code == 1
    assert "run `python main.py seed`" in result.output


def test_list_tables_counts_rows(seeded_dir: Path) -> None:
    application = build_application({"SYNTHGEN_PATHS__DATA_DIR": str(seeded_dir)}, SETTINGS, None)
    overview = application.list_tables("finance")
    rows = {t.name: t.row_count for t in overview.tables}
    assert rows["transactions"] == 4_000
    assert overview.description.startswith("Retail banking")
    assert len(overview.relationships) == 4


def test_source_without_metadata_uses_database_keys(seeded_dir: Path, tmp_path: Path) -> None:
    shutil.copy(seeded_dir / "orders.db", tmp_path / "shop.db")
    application = build_application({"SYNTHGEN_PATHS__DATA_DIR": str(tmp_path)}, SETTINGS, None)
    overview = application.list_tables("shop")
    assert overview.description == ""
    assert {r.origin for r in overview.relationships} == {"pragma"}
    assert len(overview.relationships) == 4


def test_metadata_for_another_source_is_rejected(seeded_dir: Path, tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    shutil.copy(seeded_dir / "orders.db", tmp_path / "data" / "orders.db")
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "orders.yaml").write_text("source: finance\n", encoding="utf-8")
    env = {
        "SYNTHGEN_PATHS__DATA_DIR": str(tmp_path / "data"),
        "SYNTHGEN_PATHS__METADATA_DIR": str(tmp_path / "meta"),
    }
    with pytest.raises(MetadataError):
        build_application(env, SETTINGS, None).list_tables("orders")


@pytest.mark.parametrize("source", ["../finance", "bad name", ""])
def test_invalid_source_names_are_rejected(seeded_dir: Path, source: str) -> None:
    application = build_application({"SYNTHGEN_PATHS__DATA_DIR": str(seeded_dir)}, SETTINGS, None)
    with pytest.raises(NotFoundError):
        application.list_tables(source)
