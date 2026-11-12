"""The validation repair loop end to end on the stub LLM (SPEC 3.3 rows 9-12)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from synthgen.app import Application
from synthgen.domain.job import ArtifactKind, JobStatus, UserRequest
from synthgen.domain.report import CheckFamily, ValidationReport

from .helpers import fixture_set, skeleton

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]
REQUEST = UserRequest(
    source="finance", tables=("customers", "accounts"), text="high balances", auto_approve=True
)


def _overlay(column: str, generator: dict[str, Any], **extra: Any) -> dict[str, Any]:
    overlays = skeleton("OverlaySet")
    scenario = overlays["scenarios"][0]
    scenario["overrides"] = [o for o in scenario["overrides"] if o["column"] != column] + [
        {"table": "accounts", "column": column, "generator": generator}
    ]
    scenario.update(extra)
    return overlays


def _patch(path: str, value: Any) -> dict[str, Any]:
    return {
        "patches": [{"path": path, "value_json": json.dumps(value), "reason": "r"}],
        "plan_change_required": False,
        "explanation": "Move the generator to where the checks expect it.",
    }


def _reports(app: Application, job_id: str) -> list[ValidationReport]:
    versions = app._artifacts.versions(job_id, ArtifactKind.DRY_RUN_REPORT)
    return [
        app._artifacts.get(job_id, ArtifactKind.DRY_RUN_REPORT, ValidationReport, v)
        for v in versions
    ]


LOW = {"type": "empirical", "quantiles": [100.0, 200.0], "decimals": 2}
HIGH = [50_000.0, 90_000.0]


def test_a_scenario_failure_is_repaired(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    fixtures = fixture_set(
        tmp_path / "fx",
        OverlaySet=[(None, _overlay("balance", LOW))],
        SpecPatch=[(None, _patch("/scenarios/0/overrides/0/generator/quantiles", HIGH))],
    )
    env = job_env(fixtures)
    app = make_app(env)
    view = app.start_job(REQUEST)
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    first, second = _reports(app, view.job.job_id)
    failed = {(f.family, f.check, f.scenario) for f in first.failures()}
    assert (CheckFamily.SCENARIO, "signature", "high_balance") in failed
    assert second.passed
    calls = [c.node for c in app._llm_log.for_job(view.job.job_id)]
    assert calls.count("repair_analyst") == 1
    out = Path(env["SYNTHGEN_PATHS__OUTPUT_DIR"]) / "jobs" / view.job.job_id
    written = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    assert written["passed"] is True


def test_an_overlay_that_breaks_integrity_is_caught_and_repaired(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    too_early = {
        "type": "datetime",
        "start": "1990-01-01",
        "end": "1990-12-31",
        "granularity": "date",
        "weekday_weights": None,
        "hour_weights": None,
        "after": None,
    }
    patch = _patch("/scenarios/0/overrides/1/generator/start", "2025-07-01")
    patch["patches"].append(
        {
            "path": "/scenarios/0/overrides/1/generator/end",
            "value_json": '"2025-12-31"',
            "reason": "r",
        }
    )
    fixtures = fixture_set(
        tmp_path / "fx",
        OverlaySet=[(None, _overlay("opened_date", too_early))],
        SpecPatch=[(None, patch)],
    )
    app = make_app(job_env(fixtures))
    view = app.start_job(REQUEST)
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    first = _reports(app, view.job.job_id)[0]
    broken = first.failures(CheckFamily.INTEGRITY)
    assert broken and all(f.scenario == "high_balance" for f in broken)
    assert any(f.check == "constraint_order" for f in broken)


def test_requested_defects_are_labelled_and_pass(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    defects = [{"table": "accounts", "column": "status", "kind": "null", "rate": 0.5}]
    overlays = skeleton("OverlaySet")
    overlays["scenarios"][0]["defects"] = defects
    env = job_env(fixture_set(tmp_path / "fx", OverlaySet=[(None, overlays)]))
    app = make_app(env)
    view = app.start_job(REQUEST.model_copy(update={"include_invalid": True}))
    assert view.job.status is JobStatus.COMPLETED, view.job.error
    out = Path(env["SYNTHGEN_PATHS__OUTPUT_DIR"]) / "jobs" / view.job.job_id / "data" / "accounts"
    rows = pq.read_table(out).to_pylist()
    broken = [r for r in rows if r["_defect"]]
    assert broken and all(r["_defect"] == "null:status" and r["status"] is None for r in broken)


def test_defects_without_permission_are_spec_errors(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    overlays = skeleton("OverlaySet")
    overlays["scenarios"][0]["defects"] = [
        {"table": "accounts", "column": "status", "kind": "null", "rate": 0.5}
    ]
    app = make_app(job_env(fixture_set(tmp_path / "fx", OverlaySet=[(None, overlays)])))
    view = app.start_job(REQUEST)
    assert view.job.status is JobStatus.NEEDS_REVIEW
    assert view.job.error is not None and "spec still invalid" in view.job.error


def test_an_unrepairable_scenario_stops_at_the_cap(
    make_app: AppFactory, job_env: EnvFactory, tmp_path: Path
) -> None:
    fixtures = fixture_set(
        tmp_path / "fx",
        OverlaySet=[(None, _overlay("balance", LOW))],
        SpecPatch=[(None, _patch("/scenarios/0/overrides/0/generator/quantiles", [150.0, 250.0]))],
    )
    app = make_app(job_env(fixtures))
    view = app.start_job(REQUEST)
    assert view.job.status is JobStatus.NEEDS_REVIEW
    assert view.job.error == "dry run still failing after 3 repairs"
    assert len(_reports(app, view.job.job_id)) == 4
