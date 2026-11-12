from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytest
from scipy import stats

from synthgen.engine.copula import correlated_uniforms
from synthgen.engine.runner import RunOptions, generate_dataset
from synthgen.spec.correlation import as_matrix
from synthgen.spec.merge import merge_overlays
from synthgen.spec.models import GenerationSpec, ScenarioSpec
from synthgen.spec.validator import repair_correlations, validate_spec

from .test_runner import BIG, GENERATION, _col, make_spec

TARGET = [[1.0, 0.7, -0.4], [0.7, 1.0, -0.2], [-0.4, -0.2, 1.0]]


def _spearman(columns: list[Any]) -> np.ndarray[Any, Any]:
    return np.asarray(stats.spearmanr(np.column_stack(columns)).statistic, dtype=np.float64)


def test_uniforms_recover_the_spearman_matrix() -> None:
    u = correlated_uniforms(TARGET, 50_000, np.random.default_rng(1))
    assert np.min(u) > 0 and np.max(u) < 1
    assert np.abs(u.mean(axis=0) - 0.5).max() < 0.01
    assert np.allclose(_spearman([u[:, 0], u[:, 1], u[:, 2]]), TARGET, atol=0.02)


def test_invalid_matrices_still_draw() -> None:
    bad = [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]
    assert correlated_uniforms(bad, 10, np.random.default_rng(1)).shape == (10, 3)


def _correlated_spec(scenarios: list[dict[str, Any]] | None = None) -> GenerationSpec:
    data = make_spec(customers=20_000, scenarios=scenarios).model_dump()
    customers = data["tables"][0]
    customers["columns"] += [
        _col(
            "income",
            "float",
            {"type": "empirical", "quantiles": [10.0, 50.0, 400.0], "decimals": 2},
        ),
        _col(
            "score",
            "integer",
            {
                "type": "distribution",
                "family": "norm",
                "params": [600.0, 80.0],
                "minimum": 300.0,
                "maximum": 850.0,
                "decimals": 0,
            },
        ),
    ]
    customers["correlation"] = {"columns": ["income", "score"], "spearman": [[1, 0.8], [0.8, 1]]}
    return GenerationSpec.model_validate(data)


def _customers(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path / "data" / "customers" / "part-00000.parquet").to_pylist()


def test_engine_correlates_columns_and_keeps_marginals(tmp_path: Path) -> None:
    spec = _correlated_spec()
    assert validate_spec(spec, GENERATION) == []
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    rows = _customers(tmp_path)
    income = np.array([r["income"] for r in rows])
    score = np.array([r["score"] for r in rows])
    assert stats.spearmanr(income, score).statistic == pytest.approx(0.8, abs=0.02)
    assert np.median(income) == pytest.approx(50.0, rel=0.05)
    assert score.min() >= 300 and score.max() <= 850
    assert score.mean() == pytest.approx(600.0, abs=3.0)


def test_scenario_correlation_applies_to_its_rows_only(tmp_path: Path) -> None:
    flipped = {
        **BIG,
        "entity": "customers",
        "share": 0.3,
        "labels": [],
        "overrides": [],
        "correlations": [
            {
                "table": "customers",
                "columns": ["income", "score"],
                "spearman": [[1, -0.8], [-0.8, 1]],
            }
        ],
    }
    spec = _correlated_spec(scenarios=[flipped])
    assert validate_spec(spec, GENERATION) == []
    generate_dataset(spec, tmp_path, RunOptions(), GENERATION)
    rows = _customers(tmp_path)

    def rho(name: str) -> float:
        picked = [r for r in rows if r["_scenario"] == name]
        return float(
            stats.spearmanr([r["income"] for r in picked], [r["score"] for r in picked]).statistic
        )

    assert rho("big") == pytest.approx(-0.8, abs=0.03)
    assert rho("typical") == pytest.approx(0.8, abs=0.03)


def test_validator_reports_columns_a_copula_cannot_carry() -> None:
    data = _correlated_spec().model_dump()
    data["tables"][0]["correlation"] = {
        "columns": ["tier", "score"],
        "spearman": [[1, 0.5], [0.5, 1]],
    }
    errors = validate_spec(GenerationSpec.model_validate(data), GENERATION)
    assert [e.origin for e in errors] == ["baseline"]
    assert "empirical or distribution" in errors[0].message
    data["tables"][0]["correlation"] = {
        "columns": ["ghost", "score"],
        "spearman": [[1, 0.5], [0.5, 1]],
    }
    assert (
        "unknown column"
        in validate_spec(GenerationSpec.model_validate(data), GENERATION)[0].message
    )


def test_validator_caps_copula_width() -> None:
    narrow = GENERATION.model_copy(update={"max_copula_columns": 2})
    data = _correlated_spec().model_dump()
    data["tables"][0]["correlation"] = {
        "columns": ["income", "score", "id"],
        "spearman": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    }
    errors = validate_spec(GenerationSpec.model_validate(data), narrow)
    assert any("at most 2 columns" in e.message for e in errors)


def test_overlay_correlation_errors_name_the_scenario() -> None:
    bad = {
        **BIG,
        "correlations": [
            {"table": "orders", "columns": ["amount", "flag"], "spearman": [[1, 0.5], [0.5, 1]]}
        ],
    }
    errors = validate_spec(
        merge_overlays(make_spec(), [ScenarioSpec.model_validate(bad)]), GENERATION
    )
    assert [(e.origin, e.scenario) for e in errors] == [("overlay", "big")]
    assert "flag" in errors[0].message


def test_repair_replaces_only_invalid_matrices() -> None:
    data = _correlated_spec().model_dump()
    data["tables"][0]["columns"].append(
        _col("debt", "float", {"type": "empirical", "quantiles": [0.0, 1.0], "decimals": None})
    )
    data["tables"][0]["correlation"] = {
        "columns": ["income", "score", "debt"],
        "spearman": [[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]],
    }
    spec = GenerationSpec.model_validate(data)
    repaired = repair_correlations(spec)
    correlation = repaired.tables[0].correlation
    assert correlation is not None
    assert np.min(np.linalg.eigvalsh(as_matrix(correlation.spearman))) >= -1e-9
    assert repaired.tables[1] == spec.tables[1]
    valid = _correlated_spec()
    assert repair_correlations(valid) == valid
