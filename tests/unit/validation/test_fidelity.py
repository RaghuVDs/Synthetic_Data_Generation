from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from synthgen.config import GenerationSettings, ProfilingSettings, ValidationSettings, load_settings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.report import CheckFamily, CheckResult, CheckStatus
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.metadata.graph import build_table_set
from synthgen.metadata.provider import SourceDoc, YamlMetadataProvider
from synthgen.profiling.models import TableProfile
from synthgen.profiling.profiler import profile_table
from synthgen.spec.baseline import BaselineInputs, compile_baseline
from synthgen.spec.models import GenerationSpec
from synthgen.validation.output import ValidationInputs
from synthgen.validation.report import validate_output
from synthgen.validation.stats import (
    category_js,
    discrete_ks,
    js_distance,
    ks_statistic,
    largest_spearman_gap,
    time_mixes,
    wasserstein_over_iqr,
)

ROOT = Path(__file__).resolve().parents[3]
SETTINGS = load_settings(ROOT / "config" / "settings.yaml", {})
GENERATION: GenerationSettings = SETTINGS.generation
VALIDATION: ValidationSettings = SETTINGS.validation


def test_ks_and_wasserstein_against_stored_quantiles() -> None:
    rng = np.random.default_rng(1)
    source = rng.lognormal(3, 1.0, 100_000)
    quantiles = np.quantile(source, np.linspace(0, 1, 1001))
    same = rng.lognormal(3, 1.0, 50_000)
    shifted = same * 1.5
    assert ks_statistic(same, quantiles) < 0.02
    assert (wasserstein_over_iqr(source, quantiles) or 1) < 0.1  # a source passes its own profile
    assert (wasserstein_over_iqr(same, quantiles) or 1) < 0.1
    assert ks_statistic(shifted, quantiles) > 0.1
    assert (wasserstein_over_iqr(shifted, quantiles) or 0) > 0.1
    assert wasserstein_over_iqr(same, np.full(5, 2.0)) is None


def test_mix_distances() -> None:
    assert js_distance([10, 10], [1, 1]) == pytest.approx(0.0)
    assert js_distance([1, 0], [0, 1]) == pytest.approx(1.0)
    assert js_distance([0, 0], [0, 0]) == 0.0
    assert js_distance([0, 0], [1, 0]) == 1.0
    assert category_js(["a", "b", 1.0], {"a": 1.0, "b": 1.0, "1": 1.0}) == pytest.approx(0.0)
    assert category_js(["z"] * 10, {"a": 1.0}) == pytest.approx(1.0)


def test_discrete_ks_and_time_mixes() -> None:
    assert discrete_ks([0, 1, 1, 2], {0: 1, 1: 2, 2: 1}) == pytest.approx(0.0)
    assert discrete_ks([3, 3], {0: 1}) == pytest.approx(1.0)
    stamps = (
        np.array([np.datetime64("2024-01-01T13:00:00"), np.datetime64("2024-03-02T02:00:00")])
        .astype(np.int64)
        .astype(np.float64)
    )
    mixes = time_mixes(stamps, with_hours=True)
    assert mixes["month"][0] == 1 and mixes["month"][2] == 1
    assert mixes["weekday"][0] == 1 and mixes["weekday"][5] == 1  # Monday, Saturday
    assert mixes["hour"][13] == 1 and mixes["hour"][2] == 1
    assert "hour" not in time_mixes(stamps, with_hours=False)


def test_spearman_gap() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=2_000)
    sample = np.column_stack([a, a + rng.normal(scale=0.1, size=2_000)])
    assert largest_spearman_gap(sample, [[1, 0.99], [0.99, 1]]) < 0.05
    assert largest_spearman_gap(sample, [[1, 0.0], [0.0, 1]]) > 0.9
    assert largest_spearman_gap(sample[:2], [[1, 0.0], [0.0, 1]]) == 0.0


CUSTOMERS = 3_000


@pytest.fixture(scope="module")
def big_shop_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The shop schema with enough rows for fidelity to judge (PLAN Q6)."""
    path = tmp_path_factory.mktemp("fidelity") / "shop.db"
    rng = np.random.default_rng(0)
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(
            """
            CREATE TABLE customers (
                id INTEGER PRIMARY KEY, email TEXT NOT NULL, tier TEXT, joined TEXT, score REAL
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(id),
                amount REAL NOT NULL,
                placed TEXT NOT NULL,
                paid INTEGER NOT NULL
            );
            """
        )
        conn.executemany(
            "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
            [
                (
                    i,
                    f"person{i}@mail.example",
                    ["gold", "silver", "bronze"][i % 3] if i % 10 else None,
                    f"2024-{i % 12 + 1:02d}-{i % 28 + 1:02d}",
                    round(float(rng.normal(50, 10)), 2),
                )
                for i in range(1, CUSTOMERS + 1)
            ],
        )
        owners = [(cid, k) for cid in range(1, CUSTOMERS + 1) for k in range(cid % 4)]
        conn.executemany(
            "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
            [
                (
                    n + 1,
                    cid,
                    round(float(rng.lognormal(3, 0.5)), 2),
                    f"2024-03-{k + 4:02d} {(n * 7) % 24:02d}:30:00",
                    k % 2,
                )
                for n, (cid, k) in enumerate(owners)
            ],
        )
        conn.commit()
    return path


@pytest.fixture(scope="module")
def shop(
    big_shop_db: Path, profiling_settings: ProfilingSettings
) -> tuple[GenerationSpec, dict[str, TableProfile]]:
    connector = SqliteConnector(big_shop_db)
    provider = YamlMetadataProvider(SourceDoc(source="shop"), connector)
    rels = provider.relationships()
    table_set = build_table_set(["orders"], [t.name for t in provider.tables()], rels)
    profiles = {
        t: profile_table(
            "shop",
            connector,
            provider.table(t),
            rels,
            parents={m.name: m for m in provider.tables()},
            settings=profiling_settings,
            seed=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
        for t in table_set.order
    }
    spec = compile_baseline(
        BaselineInputs(
            table_set=table_set,
            tables={t: provider.table(t) for t in table_set.order},
            profiles=profiles,
            profile_versions=dict.fromkeys(table_set.order, "v"),
            semantic_map=None,
            row_targets={"customers": 4_000},
            seed=3,
        ),
        GENERATION,
    )
    connector.close()
    return spec, profiles


def _fidelity(
    spec: GenerationSpec, profiles: dict[str, TableProfile], out: Path, **options: Any
) -> list[CheckResult]:
    run = RunOptions(**options)
    generate_dataset(spec, out, run, GENERATION)
    inputs = ValidationInputs(spec, run, VALIDATION, profiles, plan=None, source=None)
    return [r for r in validate_output(inputs, out).results if r.family is CheckFamily.FIDELITY]


def _named(results: list[CheckResult], check: str, column: str | None) -> CheckResult:
    return next(r for r in results if r.check == check and r.column == column)


def test_typical_output_passes_fidelity(
    shop: tuple[GenerationSpec, dict[str, TableProfile]], tmp_path: Path
) -> None:
    spec, profiles = shop
    results = _fidelity(spec, profiles, tmp_path)
    assert [r for r in results if r.status is CheckStatus.FAIL] == []
    checks = {(r.check, r.table, r.column) for r in results if r.status is CheckStatus.PASS}
    assert ("distribution", "customers", "score") in checks
    assert ("category_mix", "customers", "tier") in checks
    assert ("hour_mix", "orders", "placed") in checks
    assert ("weekday_mix", "customers", "joined") in checks
    assert ("null_rate", "customers", "tier") in checks
    assert ("children_per_parent", "orders", "customer_id") in checks
    assert ("row_count", "orders", None) in checks


def _with_generator(
    spec: GenerationSpec, table: str, column: str, gen: dict[str, Any]
) -> GenerationSpec:
    data = spec.model_dump()
    index = [t["name"] for t in data["tables"]].index(table)
    for col in data["tables"][index]["columns"]:
        if col["name"] == column:
            col["generator"] = gen
    return GenerationSpec.model_validate(data)


def test_shifted_columns_fail(
    shop: tuple[GenerationSpec, dict[str, TableProfile]], tmp_path: Path
) -> None:
    spec, profiles = shop
    broken = _with_generator(
        spec, "customers", "score", {"type": "empirical", "quantiles": [80.0, 120.0], "decimals": 2}
    )
    broken = _with_generator(
        broken,
        "customers",
        "tier",
        {"type": "categorical", "values": ["gold", "platinum"], "weights": [0.5, 0.5]},
    )
    results = _fidelity(broken, profiles, tmp_path)
    failed = {(r.check, r.column, r.metric) for r in results if r.status is CheckStatus.FAIL}
    assert ("distribution", "score", "ks_statistic") in failed
    assert ("distribution", "score", "wasserstein_iqr") in failed
    assert ("category_mix", "tier", "js_distance") in failed


def test_small_samples_skip_distribution_checks(
    shop: tuple[GenerationSpec, dict[str, TableProfile]], tmp_path: Path
) -> None:
    spec, profiles = shop
    results = _fidelity(spec, profiles, tmp_path, max_root_rows=500)
    score = _named(results, "distribution", "score")
    assert score.status is CheckStatus.SKIPPED
    assert score.detail is not None and "typical rows" in score.detail
    assert _named(results, "row_count", None).status is CheckStatus.PASS


def test_row_targets_skip_the_children_check(
    shop: tuple[GenerationSpec, dict[str, TableProfile]], tmp_path: Path
) -> None:
    spec, profiles = shop
    data = spec.model_dump()
    data["tables"][1]["row_rule"]["target_rows"] = 9_000
    results = _fidelity(GenerationSpec.model_validate(data), profiles, tmp_path)
    children = _named(results, "children_per_parent", "customer_id")
    assert children.status is CheckStatus.SKIPPED
    assert children.detail == "a row target rescales children per parent"


def test_a_small_profile_also_skips(
    shop: tuple[GenerationSpec, dict[str, TableProfile]], tmp_path: Path
) -> None:
    spec, profiles = shop
    thin = {
        name: p.model_copy(update={"sample_rows": 200}) if name == "customers" else p
        for name, p in profiles.items()
    }
    score = _named(_fidelity(spec, thin, tmp_path), "distribution", "score")
    assert score.status is CheckStatus.SKIPPED
    assert score.detail is not None and "200 source rows" in score.detail
