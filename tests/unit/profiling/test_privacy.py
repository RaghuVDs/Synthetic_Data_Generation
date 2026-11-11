from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from synthgen.config import ProfilingSettings
from synthgen.connectors.sqlite import SqliteConnector
from synthgen.domain.errors import PrivacyError
from synthgen.domain.metadata import Sensitivity
from synthgen.metadata.provider import ColumnDoc, SourceDoc, TableDoc, YamlMetadataProvider
from synthgen.profiling.models import Measured, StatSource, TableProfile
from synthgen.profiling.privacy import (
    OTHER,
    check_privacy,
    classify,
    fold_counts,
    fold_labels,
    min_count,
    privacy_violations,
)
from synthgen.profiling.profiler import profile_table


@pytest.fixture(scope="module")
def tagged_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Customers with an untagged email, an internal city with rare values, a tagged salary."""
    path = tmp_path_factory.mktemp("privacy") / "people.db"
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE people (id INTEGER PRIMARY KEY, email TEXT, city TEXT, salary REAL,"
        " joined TEXT, vip INTEGER)"
    )
    cities = ["Leeds"] * 60 + ["York"] * 30 + ["Hull"] * 5 + ["Ely"] * 3 + ["Ripon"] * 2
    con.executemany(
        "INSERT INTO people VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                i,
                f"secret{i}@corp.test",
                city,
                30_000 + 517 * i,
                f"2024-02-{i % 28 + 1:02d}",
                i % 7 == 0,
            )
            for i, city in enumerate(cities)
        ],
    )
    con.commit()
    con.close()
    return path


def _profile(path: Path, settings: ProfilingSettings) -> TableProfile:
    doc = SourceDoc(
        source="hr",
        tables={
            "people": TableDoc(
                columns={
                    "city": ColumnDoc(sensitivity=Sensitivity.INTERNAL),
                    "salary": ColumnDoc(sensitivity=Sensitivity.SENSITIVE),
                }
            )
        },
    )
    connector = SqliteConnector(path)
    try:
        provider = YamlMetadataProvider(doc, connector)
        return profile_table(
            "hr",
            connector,
            provider.table("people"),
            [],
            parents={},
            settings=settings,
            seed=1,
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )
    finally:
        connector.close()


def _leaves(node: object) -> list[object]:
    if isinstance(node, dict):
        return [leaf for value in node.values() for leaf in _leaves(value)]
    if isinstance(node, list):
        return [leaf for value in node for leaf in _leaves(value)]
    return [node]


def test_stored_profile_holds_no_sensitive_value(
    tagged_db: Path, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(tagged_db, profiling_settings)
    leaves = _leaves(profile.model_dump(mode="json"))
    assert not any("@corp.test" in str(leaf) for leaf in leaves)
    salaries = {float(30_000 + 517 * i) for i in range(100)}
    assert not salaries & {leaf for leaf in leaves if isinstance(leaf, int | float)}
    email, salary = profile.column("email"), profile.column("salary")
    assert email is not None and email.sensitivity is Sensitivity.SENSITIVE
    assert salary is not None and salary.sensitivity is Sensitivity.SENSITIVE
    assert salary.numeric is None and email.text is not None
    assert privacy_violations(profile, profiling_settings) == []


def test_internal_values_under_the_minimum_fold(
    tagged_db: Path, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(tagged_db, profiling_settings)
    city = profile.column("city")
    assert city is not None and city.categorical is not None
    stored = {v.value: v.count for v in city.categorical.frequencies.value}
    assert stored == {"Leeds": 60, "York": 30}
    assert city.categorical.other_count.value == 10
    assert city.categorical.entropy.value > 0
    cross = profile.cross_column
    for conditional in cross.conditionals:
        if conditional.given == "city":
            assert {level.level for level in conditional.levels} <= {"Leeds", "York", OTHER}


def test_violations_are_reported_and_raised(
    tagged_db: Path, profiling_settings: ProfilingSettings
) -> None:
    profile = _profile(tagged_db, profiling_settings)
    email = profile.column("email")
    city = profile.column("city")
    assert email is not None and city is not None and city.categorical is not None
    leaked = email.model_copy(
        update={"top_value": Measured[str | int | float](value="x", source=StatSource.SAMPLE)}
    )
    rare = city.categorical.frequencies.value[0].model_copy(update={"count": 2})
    frequencies = city.categorical.frequencies.model_copy(update={"value": [rare]})
    folded = city.model_copy(
        update={"categorical": city.categorical.model_copy(update={"frequencies": frequencies})}
    )
    cross = profile.cross_column.model_copy(update={"columns": ["email"]})
    broken = profile.model_copy(update={"columns": [leaked, folded], "cross_column": cross})
    found = privacy_violations(broken, profiling_settings)
    assert any("email: sensitive column stores top_value" in v for v in found)
    assert any("city: internal values under 10 rows" in v for v in found)
    assert any("email: sensitive column used in cross-column" in v for v in found)
    with pytest.raises(PrivacyError, match="people"):
        check_privacy(broken, profiling_settings)


def test_rare_internal_top_value_is_hidden(
    tagged_db: Path, profiling_settings: ProfilingSettings
) -> None:
    strict = profiling_settings.model_copy(update={"internal_min_count": 70})
    city = _profile(tagged_db, strict).column("city")
    assert city is not None and city.top_value is None and city.categorical is not None
    assert city.categorical.frequencies.value == []


def test_classes_and_folding_helpers(profiling_settings: ProfilingSettings) -> None:
    assert classify(None, "phone") is Sensitivity.SENSITIVE
    assert classify(Sensitivity.INTERNAL, "email") is Sensitivity.INTERNAL
    assert min_count(Sensitivity.PUBLIC, profiling_settings) == 0
    assert (
        min_count(Sensitivity.INTERNAL, profiling_settings) == profiling_settings.internal_min_count
    )
    assert fold_counts({"a": 12, "b": 3, "c": 2}, 10) == {"a": 12, OTHER: 5}
    labels = pd.Series(["a"] * 3 + ["b", None], dtype=object)
    assert fold_labels(labels, 2).tolist()[:4] == ["a", "a", "a", OTHER]
    assert fold_labels(labels, 0) is labels
