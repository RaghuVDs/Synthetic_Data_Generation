"""1,000,000 transactions from the hand-written finance spec, twice, byte for byte (slow)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.compute as pc
import pytest

from synthgen.config import load_settings
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.engine.writer import read_table
from synthgen.spec.models import GenerationSpec
from synthgen.spec.validator import validate_spec

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "tests" / "fixtures" / "specs" / "finance_1m.json"
pytestmark = pytest.mark.slow


def _digests(path: Path) -> dict[str, str]:
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def test_one_million_rows_twice_byte_identical(tmp_path: Path) -> None:
    settings = load_settings(ROOT / "config" / "settings.yaml", {}).generation
    spec = GenerationSpec.model_validate_json(SPEC.read_text(encoding="utf-8"))
    assert validate_spec(spec, settings) == []
    first = generate_dataset(spec, tmp_path / "first", RunOptions(), settings, workers=2)
    transactions = first.table("transactions")
    assert transactions is not None and transactions.rows == 1_000_000
    assert len(transactions.files) >= 4
    counts = {c.scenario: c.rows for c in transactions.scenario_counts}
    assert counts["large_night_purchases"] > 0 and counts["card_testing_bursts"] > 0
    data = read_table(tmp_path / "first", transactions, ["transaction_id", "amount", "fee"])
    assert pc.count_distinct(data.column("transaction_id")).as_py() == 1_000_000
    assert pc.max(data.column("amount")).as_py() <= 15_000.0
    second = generate_dataset(spec, tmp_path / "second", RunOptions(), settings, workers=1)
    assert first == second
    assert _digests(tmp_path / "first") == _digests(tmp_path / "second")
