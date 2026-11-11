from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from synthgen.app import Application
from synthgen.domain.intent import TestIntent
from synthgen.domain.job import ArtifactKind, JobStatus, UserRequest
from synthgen.domain.semantic import SemanticMap

from .helpers import fixture_set, skeleton

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]


def request(**changes: Any) -> UserRequest:
    fields: dict[str, Any] = {
        "source": "finance",
        "tables": ("customers", "accounts"),
        "text": "customers with unusually high balances",
        "auto_approve": True,
    }
    return UserRequest.model_validate(fields | changes)


def _one_table(name: str) -> dict[str, Any]:
    full = skeleton("SemanticMap")
    return {"tables": [t for t in full["tables"] if t["table"] == name], "business_rules": []}


def test_mapper_runs_in_batches_above_the_limit(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    fixtures = fixture_set(
        tmp_path / "fx",
        SemanticMap=[
            ("table customers (", _one_table("customers")),
            ("table accounts (", _one_table("accounts")),
        ],
    )
    app = make_app(job_env(fixtures, SYNTHGEN_PROFILING__MAPPER_BATCH_TABLES="1"))
    view = app.start_job(request())
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    mapper_calls = [c for c in app._llm_log.for_job(view.job.job_id) if c.node == "semantic_mapper"]
    assert len(mapper_calls) == 2
    merged = app._artifacts.get(view.job.job_id, ArtifactKind.SEMANTIC_MAP, SemanticMap)
    assert [t.table for t in merged.tables] == ["customers", "accounts"]


def test_include_invalid_flag_is_enforced_by_code(
    make_app: AppFactory, job_env: EnvFactory
) -> None:
    app = make_app(job_env())
    view = app.start_job(request(include_invalid=True))
    intent = app._artifacts.get(view.job.job_id, ArtifactKind.INTENT, TestIntent)
    assert intent.include_invalid is True


def test_volume_hint_above_the_limit_needs_review(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    huge = skeleton("TestIntent") | {"volume_hint": 50_000_000}
    app = make_app(job_env(fixture_set(tmp_path / "fx", TestIntent=[(None, huge)])))
    view = app.start_job(request())
    assert view.job.status is JobStatus.NEEDS_REVIEW
    assert view.job.error is not None and "exceeds the limit" in view.job.error
