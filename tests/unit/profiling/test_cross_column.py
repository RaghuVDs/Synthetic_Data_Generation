from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.associations import (
    categorical_pair,
    correlation_ratio,
    cramers_v,
    mixed_pair,
    theils_u,
)
from synthgen.profiling.models import TableProfile
from synthgen.profiling.profiler import profile_table
from synthgen.profiling.relation_models import RuleKind
from synthgen.profiling.rules import functional_dependencies, mode_share, null_pairs

ROWS = 3_000
CITY_REGION = {"Leeds": "North", "York": "North", "Bath": "South", "Kent": "South"}


@pytest.fixture(scope="module")
def related_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = np.random.default_rng(3)
    path = tmp_path_factory.mktemp("cross") / "related.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE sales (id INTEGER PRIMARY KEY, x REAL, y REAL, noise REAL, city TEXT,"
        " region TEXT, tier TEXT, amount REAL, flag INTEGER, price REAL, fee REAL, total REAL,"
        " qty INTEGER, line REAL, opened TEXT, closed TEXT, note_a TEXT, note_b TEXT)"
    )
    start = datetime(2024, 1, 1)
    rows = []
    for i in range(ROWS):
        x = float(rng.normal())
        city = str(rng.choice(list(CITY_REGION)))
        tier = str(rng.choice(["gold", "silver", "bronze"]))
        amount = round(float(rng.lognormal({"gold": 6, "silver": 4, "bronze": 2}[tier], 0.3)), 2)
        price, fee = round(float(rng.uniform(5, 50)), 2), round(float(rng.uniform(0, 3)), 2)
        qty = int(rng.integers(1, 6))
        opened = start + timedelta(days=int(rng.integers(0, 300)), hours=int(rng.integers(0, 24)))
        closed = opened + timedelta(days=int(rng.integers(0, 30)))
        note = None if rng.random() < 0.2 else "n"
        rows.append(
            (
                i, x, 2 * x + float(rng.normal(0, 0.1)), float(rng.normal()), city,
                CITY_REGION[city], tier, amount, int(amount > 100), price, fee,
                round(price + fee, 2), qty, round(qty * price, 2),
                opened.isoformat(sep=" "), closed.isoformat(sep=" "), note, note,
            )
        )  # fmt: skip
    con.executemany(f"INSERT INTO sales VALUES ({','.join('?' * 18)})", rows)
    con.commit()
    con.close()
    return path


def _profile(path: Path, settings: ProfilingSettings) -> TableProfile:
    connector = SqliteConnector(path)
    try:
        provider = YamlMetadataProvider(SourceDoc(source="rel"), connector)
        return profile_table(
            "rel",
            connector,
            provider.table("sales"),
            [],
            settings=settings,
            seed=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
    finally:
        connector.close()


def _rules(profile: TableProfile, kind: RuleKind) -> set[tuple[str, str, str | None]]:
    return {(r.a, r.b, r.result) for r in profile.cross_column.rules if r.kind is kind}


def test_profile_recovers_known_relations(
    related_db: Path, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(related_db, profiling_settings)
    cross = profile.cross_column
    assert "id" not in cross.columns and not cross.truncated
    xy = next(p for p in cross.numeric_pairs if {p.a, p.b} == {"x", "y"})
    assert xy.pearson > 0.99 and xy.spearman > 0.99 and xy.kendall > 0.9
    i, j = cross.spearman_columns.index("x"), cross.spearman_columns.index("noise")
    assert abs(cross.spearman[i][j]) < 0.1
    assert ("city", "region", None) in _rules(profile, RuleKind.DETERMINES)
    assert ("region", "city", None) not in _rules(profile, RuleKind.DETERMINES)
    assert ("opened", "closed", None) in _rules(profile, RuleKind.LESS_EQUAL)
    assert ("price", "fee", "total") in _rules(profile, RuleKind.SUM)
    assert ("price", "qty", "line") in _rules(profile, RuleKind.PRODUCT)
    notes = next(p for p in cross.null_pairs if {p.a, p.b} == {"note_a", "note_b"})
    assert notes.jaccard == 1.0


def test_profile_recovers_label_and_category_effects(
    related_db: Path, profiling_settings: ProfilingSettings
) -> None:
    cross = _profile(related_db, profiling_settings).cross_column
    tier = next(p for p in cross.mixed_pairs if (p.numeric, p.categorical) == ("amount", "tier"))
    assert tier.correlation_ratio > 0.5 and tier.kruskal_pvalue < 1e-6
    means = {level.level: level.mean for level in tier.levels}
    assert means["gold"] > means["silver"] > means["bronze"]
    flag = next(p for p in cross.mixed_pairs if (p.numeric, p.categorical) == ("amount", "flag"))
    assert flag.point_biserial is not None and flag.point_biserial > 0.5
    city = next(p for p in cross.categorical_pairs if {p.a, p.b} == {"city", "region"})
    assert city.cramers_v > 0.9
    given = next(c for c in cross.conditionals if (c.given, c.dependent) == ("city", "region"))
    assert all(len(level.values) == 1 for level in given.levels)


def test_pair_columns_are_capped(related_db: Path, profiling_settings: ProfilingSettings) -> None:
    capped = profiling_settings.model_copy(update={"max_pair_columns": 4})
    cross = _profile(related_db, capped).cross_column
    assert cross.truncated and len(cross.columns) == 4
    assert {"opened", "closed", "flag"} <= set(cross.columns)  # dates and booleans come first


def test_theils_u_and_cramers_v() -> None:
    a = np.array([0, 0, 1, 1, 2, 2] * 10)
    b = np.array([0, 0, 0, 0, 1, 1] * 10)
    assert theils_u(a, b) == pytest.approx(1.0)
    assert theils_u(b, a) < 1.0
    assert cramers_v(a, a) == pytest.approx(1.0)
    assert cramers_v(a, np.zeros_like(a)) == 0.0
    assert categorical_pair("a", "z", a, np.zeros_like(a)).chi2_pvalue == 1.0


def test_correlation_ratio_and_degenerate_mixed_pair() -> None:
    codes = np.array([0, 0, 1, 1])
    assert correlation_ratio(np.array([1.0, 1.0, 5.0, 5.0]), codes) == pytest.approx(1.0)
    assert correlation_ratio(np.array([2.0, 2.0, 2.0, 2.0]), codes) == 0.0
    constant = mixed_pair("v", "c", np.full(4, 2.0), codes, labels=["p", "q"], level_limit=5)
    assert constant.kruskal_pvalue == 1.0 and constant.point_biserial is None
    single = mixed_pair(
        "v", "c", np.array([1.0, 2.0]), np.array([0, 0]), labels=["p"], level_limit=5
    )
    assert single.anova_pvalue == 1.0


def test_mode_share_and_dependencies() -> None:
    assert mode_share(np.array([0, 0, 1, 1]), np.array([5, 5, 6, 7])) == pytest.approx(0.75)
    codes = {"a": np.array([0, 0, 1, 1] * 5), "b": np.array([1, 1, 0, 0] * 5)}
    rules = functional_dependencies(codes, 0.999)
    assert [(r.a, r.b) for r in rules] == [("a", "b")]  # one-to-one: one direction only


def test_null_pairs_need_partial_nulls() -> None:
    always = np.ones(4, dtype=bool)
    assert null_pairs({"a": always, "b": always}, 0.3) == []
    partial = np.array([True, False, True, False])
    assert null_pairs({"a": partial, "b": ~partial}, 0.3) == []
