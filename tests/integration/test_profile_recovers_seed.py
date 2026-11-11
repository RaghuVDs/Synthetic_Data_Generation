"""The full profiler recovers the parameters the finance seed generated its data from (T4.9)."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from synthgen.config import ProfilingSettings, load_settings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.metadata.provider import YamlMetadataProvider, load_source_doc
from synthgen.profiling.models import ColumnProfile, TableProfile
from synthgen.profiling.privacy import privacy_violations
from synthgen.profiling.profiler import profile_table
from synthgen.profiling.relation_models import RuleKind
from synthgen.seed import finance_activity
from synthgen.seed.finance import build_finance

ROOT = Path(__file__).resolve().parents[2]
SCALE = 0.1
TRANSACTIONS = 20_000
SEED = 7


@pytest.fixture(scope="module")
def settings() -> ProfilingSettings:
    return load_settings(ROOT / "config" / "settings.yaml", {}).profiling


@pytest.fixture(scope="module")
def profiles(
    tmp_path_factory: pytest.TempPathFactory, settings: ProfilingSettings
) -> Iterator[dict[str, TableProfile]]:
    path = tmp_path_factory.mktemp("recover") / "finance.db"
    counts = build_finance(path, SEED, SCALE)
    assert counts["transactions"] == TRANSACTIONS
    connector = SqliteConnector(path)
    provider = YamlMetadataProvider(
        load_source_doc(ROOT / "config/metadata/finance.yaml"), connector
    )
    metas = {m.name: m for m in provider.tables()}
    yield {
        name: profile_table(
            "finance",
            connector,
            metas[name],
            provider.relationships(),
            parents=metas,
            settings=settings,
            seed=0,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
        for name in ("customers", "accounts", "cards", "transactions")
    }
    connector.close()


def _column(profile: TableProfile, name: str) -> ColumnProfile:
    column = profile.column(name)
    assert column is not None
    return column


def _total_variation(counts: list[int], weights: list[float]) -> float:
    observed = np.asarray(counts, dtype=np.float64)
    expected = np.asarray(weights, dtype=np.float64)
    return float(0.5 * np.abs(observed / observed.sum() - expected / expected.sum()).sum())


def test_money_columns_are_log_normal(profiles: dict[str, TableProfile]) -> None:
    for table, column in (("transactions", "amount"), ("customers", "annual_income")):
        numeric = _column(profiles[table], column).numeric
        assert numeric is not None and numeric.best_fit == "lognorm"
    income = _column(profiles["customers"], "annual_income").numeric
    assert income is not None
    sigma, _, scale = income.fits.value[0].params
    assert 0.5 <= sigma <= 0.75  # 0.5 within a segment, wider across the three segments
    assert np.log(scale) == pytest.approx(10.8 + 0.18 * 0.7 + 0.07 * 1.1, abs=0.1)


def test_keys_are_zipf_skewed(profiles: dict[str, TableProfile]) -> None:
    shapes = {r.child_column: r.shape.value for r in profiles["transactions"].relationships}
    merchant, card = shapes["merchant_id"], shapes["card_id"]
    assert merchant.zipf_exponent == pytest.approx(0.9, abs=0.2)
    assert card.zipf_exponent == pytest.approx(0.7, abs=0.2)
    assert merchant.gini > 0.4 and merchant.orphan_rate == 0.0
    city = _column(profiles["customers"], "city").categorical
    assert city is not None and city.zipf_exponent is not None
    assert city.zipf_exponent.value == pytest.approx(0.9, abs=0.2)


def test_hour_and_weekday_mix(profiles: dict[str, TableProfile]) -> None:
    temporal = _column(profiles["transactions"], "txn_timestamp").temporal
    assert temporal is not None
    hours = _total_variation(temporal.hour_counts.value, finance_activity.HOUR_WEIGHTS)
    weekdays = _total_variation(temporal.weekday_counts.value, finance_activity.WEEKDAY_WEIGHTS)
    assert hours < 0.03 and weekdays < 0.03


def test_fraud_rate_and_its_signature(profiles: dict[str, TableProfile]) -> None:
    fraud = profiles["transactions"].stratified[0]
    assert fraud.label_column == "is_fraud" and fraud.minority == "1"
    balance = {b.value: b.count for b in fraud.balance}
    assert balance["1"] == round(finance_activity.FRAUD_RATE * TRANSACTIONS)
    assert fraud.separation[0].column == "amount"
    amounts = {c.value: c.numbers for c in fraud.per_class if c.column == "amount"}
    high, low = amounts["1"], amounts["0"]
    assert high is not None and low is not None and high.mean > 1.5 * low.mean


def test_correlations(profiles: dict[str, TableProfile]) -> None:
    cross = profiles["customers"].cross_column
    i = cross.spearman_columns.index("annual_income")
    j = cross.spearman_columns.index("credit_score")
    assert cross.spearman[i][j] > 0.7
    segment = next(
        p for p in cross.mixed_pairs if (p.numeric, p.categorical) == ("annual_income", "segment")
    )
    means = {level.level: level.mean for level in segment.levels}
    assert means["business"] > means["premium"] > means["retail"]
    rules = {(r.kind, r.a, r.b) for r in cross.rules}
    assert (RuleKind.DETERMINES, "city", "country") in rules


def test_date_order_rules(profiles: dict[str, TableProfile]) -> None:
    rules = {(r.kind, r.a, r.b) for r in profiles["cards"].cross_column.rules}
    assert (RuleKind.LESS_EQUAL, "issued_date", "expiry_date") in rules
    orders = {
        (o.parent_column, o.child_column): o.after_share
        for table in ("accounts", "transactions")
        for r in profiles[table].relationships
        for o in r.temporal_orders.value
    }
    assert orders[("signup_date", "opened_date")] == 1.0
    assert orders[("issued_date", "txn_timestamp")] == 1.0


def test_profiles_respect_privacy(
    profiles: dict[str, TableProfile], settings: ProfilingSettings
) -> None:
    for profile in profiles.values():
        assert privacy_violations(profile, settings) == []
    for name in ("email", "phone"):
        assert _column(profiles["customers"], name).categorical is None
    assert _column(profiles["cards"], "card_number").top_value is None
