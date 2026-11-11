from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.models import TableProfile
from synthgen.profiling.profiler import profile_table
from synthgen.profiling.stratified import (
    categorical_separation,
    jensen_shannon,
    numeric_separation,
)

ROWS = 4_000
FRAUD_RATE = 0.05


@pytest.fixture(scope="module")
def fraud_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(21)
    path = tmp_path_factory.mktemp("label") / "fraud.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE cards (id INTEGER PRIMARY KEY);"
        "CREATE TABLE txns (id INTEGER PRIMARY KEY, card_id INTEGER REFERENCES cards(id),"
        " at TEXT, amount REAL, channel TEXT, email TEXT, promo INTEGER, is_fraud INTEGER);"
    )
    con.executemany("INSERT INTO cards VALUES (?)", [(c,) for c in range(200)])
    start = datetime(2024, 3, 1)
    rows = []
    for i in range(ROWS):
        fraud = rng.random() < FRAUD_RATE
        hour = int(rng.integers(0, 5)) if fraud else int(rng.integers(8, 22))
        moment = start + timedelta(days=int(rng.integers(0, 60)), hours=hour)
        amount = float(rng.lognormal(6 if fraud else 3, 0.4))
        channel = str(rng.choice(["web", "pos"], p=[0.9, 0.1] if fraud else [0.3, 0.7]))
        rows.append(
            (
                i, int(rng.integers(0, 200)), moment.isoformat(sep=" "), round(amount, 2),
                channel, f"u{i}@mail.test", int(rng.random() < 0.5), int(fraud),
            )
        )  # fmt: skip
    con.executemany("INSERT INTO txns VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return path


def _profile(path: Path, settings: ProfilingSettings) -> TableProfile:
    connector = SqliteConnector(path)
    try:
        provider = YamlMetadataProvider(SourceDoc(source="fraud"), connector)
        return profile_table(
            "fraud",
            connector,
            provider.table("txns"),
            provider.relationships(),
            parents={m.name: m for m in provider.tables()},
            settings=settings,
            seed=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
    finally:
        connector.close()


def test_rarest_binary_column_comes_first(
    fraud_db: Path, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(fraud_db, profiling_settings)
    assert [s.label_column for s in profile.stratified] == ["is_fraud", "promo"]
    fraud = profile.stratified[0]
    assert fraud.minority == "1"
    balance = {b.value: b for b in fraud.balance}
    assert balance["1"].share == pytest.approx(FRAUD_RATE, abs=0.01)
    assert balance["0"].count + balance["1"].count == ROWS


def test_separation_ranks_the_changed_columns(
    fraud_db: Path, profiling_settings: ProfilingSettings
) -> None:
    fraud = _profile(fraud_db, profiling_settings).stratified[0]
    scores = {s.column: s for s in fraud.separation}
    assert "email" not in scores and "card_id" not in scores and "is_fraud" not in scores
    assert {fraud.separation[0].column, fraud.separation[1].column} == {"amount", "at"}
    amount = scores["amount"]
    assert amount.ks is not None and amount.ks > 0.9 and amount.wasserstein is not None
    assert scores["channel"].jensen_shannon > 0.3
    assert scores["promo"].score < 0.2


def test_per_class_summaries(fraud_db: Path, profiling_settings: ProfilingSettings) -> None:
    fraud = _profile(fraud_db, profiling_settings).stratified[0]
    per = {(c.column, c.value): c for c in fraud.per_class}
    high, low = per[("amount", "1")].numbers, per[("amount", "0")].numbers
    assert high is not None and low is not None and high.mean > 10 * low.mean
    assert high.best_fit is not None and high.best_fit.name == "lognorm"
    night = per[("at", "1")].times
    assert night is not None and sum(night.hour_counts[:5]) == sum(night.hour_counts)
    shares = per[("channel", "1")].shares
    assert shares is not None and dict(shares)["web"] > 0.8


def test_behaviour_per_class(fraud_db: Path, profiling_settings: ProfilingSettings) -> None:
    (cards,) = _profile(fraud_db, profiling_settings).behaviour
    by_class = {(c.label_column, c.value): c.stats for c in cards.by_class}
    assert set(by_class) == {("is_fraud", "0"), ("is_fraud", "1"), ("promo", "0"), ("promo", "1")}
    assert by_class[("is_fraud", "1")].events < by_class[("is_fraud", "0")].events


def test_label_cap(fraud_db: Path, profiling_settings: ProfilingSettings) -> None:
    none = profiling_settings.model_copy(update={"max_label_columns": 0})
    profile = _profile(fraud_db, none)
    assert profile.stratified == [] and profile.behaviour[0].by_class == []


def test_separation_measures() -> None:
    same = numeric_separation("x", "numeric", np.arange(100.0), np.arange(100.0), 10)
    assert same.ks == 0.0 and same.jensen_shannon == 0.0 and same.wasserstein == 0.0
    apart = numeric_separation("x", "numeric", np.zeros(50), np.ones(50), 10)
    assert apart.ks == 1.0 and apart.jensen_shannon == pytest.approx(1.0)
    flat = numeric_separation("x", "numeric", np.ones(5), np.ones(5), 10)
    assert flat.wasserstein == 0.0
    split = categorical_separation("c", np.zeros(5, dtype=np.int64), np.ones(5, dtype=np.int64), 2)
    assert split.score == pytest.approx(1.0) and split.ks is None
    assert jensen_shannon(np.zeros(3), np.ones(3)) == 0.0
