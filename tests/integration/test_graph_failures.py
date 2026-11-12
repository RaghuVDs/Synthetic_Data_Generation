from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from synthgen.app import Application
from synthgen.domain.job import ArtifactKind, JobStatus, UserRequest

from .helpers import fixture_set, skeleton

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]
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


def _run(make_app: AppFactory, env: dict[str, str]) -> tuple[str, JobStatus, str | None]:
    app = make_app(env)
    view = app.start_job(REQUEST)
    return view.job.job_id, view.job.status, view.job.error


def test_overlay_errors_stop_at_the_repair_cap(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    wrong_type = _overlay(
        "opened_date", {"type": "pattern", "masks": ["AAA"], "weights": [1.0], "check_digit": None}
    )
    _, status, error = _run(
        make_app, job_env(fixture_set(tmp_path / "fx", OverlaySet=[(None, wrong_type)]))
    )
    assert status is JobStatus.NEEDS_REVIEW
    assert error == "spec still invalid after 3 repairs"


def test_unknown_column_is_reprompted_until_the_cap(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    unknown = _overlay("no_such_column", {"type": "constant", "value": 1})
    env = job_env(fixture_set(tmp_path / "fx", OverlaySet=[(None, unknown)]))
    job_id, status, error = _run(make_app, env)
    assert status is JobStatus.NEEDS_REVIEW
    assert error is not None and error.startswith("OverlaySet failed after 3 attempts")
    assert "unknown column accounts.no_such_column" in error
    calls = make_app(env)._llm_log.for_job(job_id)
    overlay_calls = [c for c in calls if c.node == "overlay_writer"]
    assert [c.status for c in overlay_calls] == ["invalid"] * 3


def test_invalid_llm_output_needs_review(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    broken = fixture_set(tmp_path / "fx", SemanticMap=[(None, {"tables": "not a list"})])
    _, status, error = _run(make_app, job_env(broken))
    assert status is JobStatus.NEEDS_REVIEW
    assert error is not None and "SemanticMap" in error


def test_baseline_error_fails_the_job(make_app: AppFactory, job_env: EnvFactory) -> None:
    _, status, error = _run(make_app, job_env(SYNTHGEN_GENERATION__MAX_ROWS_PER_TABLE="10"))
    assert status is JobStatus.FAILED
    assert error is not None and error.startswith("baseline spec invalid")


def test_dry_run_failure_is_repaired(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    patch = {
        "patches": [
            {
                "path": "/scenarios/0/overrides/0/generator/value",
                "value_json": "1",
                "reason": "point scenario accounts at an existing customer",
            }
        ],
        "plan_change_required": False,
        "explanation": "The overlay pointed accounts at a customer that does not exist.",
    }
    env = job_env(
        fixture_set(tmp_path / "fx", OverlaySet=[(None, BROKEN_FK)], SpecPatch=[(None, patch)])
    )
    job_id, status, error = _run(make_app, env)
    assert status is JobStatus.COMPLETED, error
    app = make_app(env)
    assert len(app._artifacts.versions(job_id, ArtifactKind.SPEC)) >= 2
    dry_run_reports = app._artifacts.versions(job_id, ArtifactKind.DRY_RUN_REPORT)
    assert len(dry_run_reports) == 2


def test_unrepaired_dry_run_stops_at_the_cap(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    useless = {
        "patches": [
            {
                "path": "/scenarios/0/overrides/0/generator/value",
                "value_json": "999998",
                "reason": "r",
            }
        ],
        "plan_change_required": False,
        "explanation": "Try another customer.",
    }
    env = job_env(
        fixture_set(tmp_path / "fx", OverlaySet=[(None, BROKEN_FK)], SpecPatch=[(None, useless)])
    )
    _, status, error = _run(make_app, env)
    assert status is JobStatus.NEEDS_REVIEW
    assert error == "dry run still failing after 3 repairs"


def test_disallowed_patch_path_needs_review(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    sneaky = {
        "patches": [{"path": "/scenarios/0/share", "value_json": "0.5", "reason": "r"}],
        "plan_change_required": False,
        "explanation": "Change the share.",
    }
    env = job_env(
        fixture_set(tmp_path / "fx", OverlaySet=[(None, BROKEN_FK)], SpecPatch=[(None, sneaky)])
    )
    _, status, error = _run(make_app, env)
    assert status is JobStatus.NEEDS_REVIEW
    assert error is not None and "/scenarios/0/share is not patchable" in error


def test_plan_change_request_returns_to_approval(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    replan = {
        "patches": [],
        "plan_change_required": True,
        "explanation": "The scenario needs another table.",
    }
    env = job_env(
        fixture_set(tmp_path / "fx", OverlaySet=[(None, BROKEN_FK)], SpecPatch=[(None, replan)])
    )
    job_id, status, error = _run(make_app, env)
    assert status is JobStatus.NEEDS_REVIEW
    assert error == "dry run still failing after 3 repairs"
    approvals = make_app(env)._jobs.approvals(job_id)
    assert len(approvals) == 4  # the first approval, then one per plan-change request
