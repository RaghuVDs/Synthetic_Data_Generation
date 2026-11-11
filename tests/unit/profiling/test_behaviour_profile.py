from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.behaviour import (
    distinct_per_entity,
    lag1_autocorrelation,
    lifetimes_days,
    per_period,
    transitions,
)
from synthgen.profiling.models import TableProfile
from synthgen.profiling.profiler import profile_table

ACCOUNTS = 150
EVENTS = 40
MEAN_GAP_HOURS = 10.0
NEXT_STATE = {
    "ok": (["ok", "review"], [0.9, 0.1]),
    "review": (["ok", "blocked"], [0.5, 0.5]),
    "blocked": (["blocked"], [1.0]),
}


@pytest.fixture(scope="module")
def events_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(11)
    path = tmp_path_factory.mktemp("events") / "events.db"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE accounts (id INTEGER PRIMARY KEY, opened TEXT);"
        "CREATE TABLE merchants (id INTEGER PRIMARY KEY, name TEXT);"
        "CREATE TABLE payments (id INTEGER PRIMARY KEY,"
        " account_id INTEGER REFERENCES accounts(id),"
        " merchant_id INTEGER REFERENCES merchants(id), paid_at TEXT, amount REAL, status TEXT);"
    )
    con.executemany("INSERT INTO merchants VALUES (?, ?)", [(m, f"m{m}") for m in range(60)])
    payment = 0
    for account in range(ACCOUNTS):
        con.execute("INSERT INTO accounts VALUES (?, ?)", (account, "2023-12-01"))
        moment = datetime(2024, 1, 1) + timedelta(hours=float(rng.uniform(0, 48)))
        scale = float(rng.lognormal(3, 1))
        merchants = rng.choice(60, size=3, replace=False)
        state = "ok"
        for _ in range(EVENTS):
            moment += timedelta(hours=float(rng.exponential(MEAN_GAP_HOURS)))
            row = (
                payment, account, int(rng.choice(merchants)), moment.isoformat(sep=" "),
                round(scale * float(rng.lognormal(0, 0.2)), 2), state,
            )  # fmt: skip
            con.execute("INSERT INTO payments VALUES (?, ?, ?, ?, ?, ?)", row)
            payment += 1
            options, weights = NEXT_STATE[state]
            state = str(rng.choice(options, p=weights))
    con.commit()
    con.close()
    return path


def _profile(path: Path, table: str, settings: ProfilingSettings) -> TableProfile:
    connector = SqliteConnector(path)
    try:
        provider = YamlMetadataProvider(SourceDoc(source="ev"), connector)
        return profile_table(
            "ev",
            connector,
            provider.table(table),
            provider.relationships(),
            parents={m.name: m for m in provider.tables()},
            settings=settings,
            seed=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
    finally:
        connector.close()


def test_behaviour_of_accounts(events_db: Path, profiling_settings: ProfilingSettings) -> None:
    profile = _profile(events_db, "payments", profiling_settings)
    by_key = {b.entity_column: b for b in profile.behaviour}
    assert set(by_key) == {"account_id", "merchant_id"}
    account = by_key["account_id"]
    assert (account.time_column, account.measure_column) == ("paid_at", "amount")
    assert account.stats.entities == ACCOUNTS
    assert account.stats.events_per_entity.p50 == EVENTS
    assert account.stats.inter_arrival_hours is not None
    assert account.stats.inter_arrival_hours.p50 == pytest.approx(
        MEAN_GAP_HOURS * np.log(2), rel=0.15
    )
    assert account.inter_arrival_fits[0].name in {"expon", "gamma"}
    assert account.stats.value_to_entity_median is not None
    assert account.stats.value_to_entity_median.p50 == pytest.approx(1.0, abs=0.05)
    assert account.lifetime_days.p50 == pytest.approx(EVENTS * MEAN_GAP_HOURS / 24, rel=0.25)
    (merchants,) = account.counterparties
    assert merchants.column == "merchant_id" and merchants.per_entity.max == 3
    (status,) = account.transitions
    matrix = dict(zip(status.states, status.probabilities, strict=True))
    blocked = status.states.index("blocked")
    assert matrix["blocked"][blocked] == 1.0
    assert matrix["ok"][status.states.index("ok")] == pytest.approx(0.9, abs=0.05)
    assert account.gap_autocorrelation is not None and abs(account.gap_autocorrelation) < 0.15


def test_tables_without_events_have_no_behaviour(
    events_db: Path, profiling_settings: ProfilingSettings
) -> None:
    assert _profile(events_db, "accounts", profiling_settings).behaviour == []


def test_entity_sample_is_capped(events_db: Path, profiling_settings: ProfilingSettings) -> None:
    capped = profiling_settings.model_copy(
        update={"behaviour_entities": 20, "behaviour_key_batch": 7}
    )
    profile = _profile(events_db, "payments", capped)
    account = next(b for b in profile.behaviour if b.entity_column == "account_id")
    assert account.stats.entities == 20 and account.stats.events == 20 * EVENTS


def test_pure_behaviour_functions() -> None:
    entity = np.array([0, 0, 0, 1, 1])
    day = 86_400
    time = np.array([0, 100, 2 * day, 0, 9 * day])
    assert per_period(entity, time, day).max == 2
    assert lifetimes_days(entity, time).max == 9
    assert lag1_autocorrelation(entity, np.array([1.0, 2.0, 3.0, 5.0, np.nan])) is None
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert lag1_autocorrelation(np.zeros(6, dtype=np.int64), values) == pytest.approx(1.0)
    assert distinct_per_entity(entity, np.array([4, 4, 5, -1, -1])).max == 2
    matrix = transitions("s", entity, np.array([0, 1, 1, 1, 0]), ["a", "b"])
    assert matrix.probabilities == [[0.0, 1.0], [0.5, 0.5]]  # b->b in 0, b->a in 1
    assert per_period(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), day).max == 0
