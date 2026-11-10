from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from synthgen.app import build_application
from synthgen.domain.job import ArtifactKind, JobStatus, UserRequest

from .helpers import SETTINGS_FILE, fixture_set, skeleton

EnvFactory = Callable[..., dict[str, str]]
REQUEST = UserRequest(
    source="finance", tables=("customers", "accounts"), text="high balances", auto_approve=True
)


def _overlay(column: str, generator: dict[str, Any]) -> dict[str, Any]:
    overlays = skeleton("OverlaySet")
    overlays["scenarios"][0]["overrides"] = [
        {"table": "accounts", "column": column, "generator": generator}
    ]
    return overlays


BROKEN_FK = _overlay("customer_id", {"type": "constant", "value": 999_999})


def _run(env: dict[str, str]) -> tuple[str, JobStatus, str | None]:
    app = build_application(env, SETTINGS_FILE, None)
    view = app.start_job(REQUEST)
    return view.job.job_id, view.job.status, view.job.error


def test_overlay_errors_stop_at_the_repair_cap(job_env: EnvFactory, tmp_path: Path) -> None:
    unknown = _overlay("no_such_column", {"type": "constant", "value": 1})
    _, status, error = _run(job_env(fixture_set(tmp_path / "fx", OverlaySet=[(None, unknown)])))
    assert status is JobStatus.NEEDS_REVIEW
    assert error == "spec still invalid after 3 repairs"


def test_invalid_llm_output_needs_review(job_env: EnvFactory, tmp_path: Path) -> None:
    broken = fixture_set(tmp_path / "fx", SemanticMap=[(None, {"tables": "not a list"})])
    _, status, error = _run(job_env(broken))
    assert status is JobStatus.NEEDS_REVIEW
    assert error is not None and "SemanticMap" in error


def test_baseline_error_fails_the_job(job_env: EnvFactory) -> None:
    _, status, error = _run(job_env(SYNTHGEN_GENERATION__MAX_ROWS_PER_TABLE="10"))
    assert status is JobStatus.FAILED
    assert error is not None and error.startswith("baseline spec invalid")


def test_dry_run_failure_is_repaired(job_env: EnvFactory, tmp_path: Path) -> None:
    patch = {
        "patches": [
            {
                "path": "/scenarios/0/overrides/0/column",
                "value_json": '"balance"',
                "reason": "wrong column",
            },
            {
                "path": "/scenarios/0/overrides/0/generator",
                "value_json": json.dumps(
                    {"type": "empirical", "quantiles": [50000.0, 90000.0], "decimals": 2}
                ),
                "reason": "balances, not keys",
            },
        ],
        "plan_change_required": False,
        "explanation": "The overlay changed the key instead of the balance.",
    }
    env = job_env(
        fixture_set(tmp_path / "fx", OverlaySet=[(None, BROKEN_FK)], SpecPatch=[(None, patch)])
    )
    job_id, status, error = _run(env)
    assert status is JobStatus.COMPLETED, error
    app = build_application(env, SETTINGS_FILE, None)
    assert len(app._artifacts.versions(job_id, ArtifactKind.SPEC)) >= 2
    dry_run_reports = app._artifacts.versions(job_id, ArtifactKind.DRY_RUN_REPORT)
    assert len(dry_run_reports) == 2


def test_unrepaired_dry_run_stops_at_the_cap(job_env: EnvFactory, tmp_path: Path) -> None:
    _, status, error = _run(job_env(fixture_set(tmp_path / "fx", OverlaySet=[(None, BROKEN_FK)])))
    assert status is JobStatus.NEEDS_REVIEW
    assert error == "dry run still failing after 3 repairs"


def test_plan_change_request_returns_to_approval(job_env: EnvFactory, tmp_path: Path) -> None:
    replan = {
        "patches": [],
        "plan_change_required": True,
        "explanation": "The scenario needs another table.",
    }
    env = job_env(
        fixture_set(tmp_path / "fx", OverlaySet=[(None, BROKEN_FK)], SpecPatch=[(None, replan)])
    )
    job_id, status, error = _run(env)
    assert status is JobStatus.NEEDS_REVIEW
    assert error == "dry run still failing after 3 repairs"
    approvals = build_application(env, SETTINGS_FILE, None)._jobs.approvals(job_id)
    assert len(approvals) == 4  # the first approval, then one per plan-change request
