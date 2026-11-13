"""Use-case results shaped as JSON for the page: plain dicts, no domain objects leak out."""

from __future__ import annotations

from typing import Any

import pyarrow as pa

from synthgen.app import JobReport, JobView
from synthgen.domain.metadata import SourceOverview
from synthgen.domain.pending import PendingApproval, PendingClarification
from synthgen.domain.report import CheckStatus

Json = dict[str, Any]  # Any: JSON values (str, number, bool, None, lists, dicts)


def source_json(overview: SourceOverview) -> Json:
    """A source's tables (rows, columns, description) and relationships."""
    return {
        "name": overview.source,
        "description": overview.description,
        "tables": [
            {
                "name": t.name,
                "rows": t.row_count,
                "columns": t.column_count,
                "description": t.description,
            }
            for t in overview.tables
        ],
        "relationships": [r.label() for r in overview.relationships],
    }


def job_json(view: JobView, *, busy: bool, failure: str | None) -> Json:
    """Status, error and, when the job is paused, its questions or plan."""
    job = view.job
    out: Json = {
        "job_id": job.job_id,
        "status": job.status.value,
        "terminal": job.status.is_terminal,
        "error": job.error or failure,
        "busy": busy,
        "request": job.request.model_dump(mode="json"),
        "pending": None,
    }
    pending = None if busy else view.pending
    if isinstance(pending, PendingClarification):
        out["pending"] = {
            "kind": "clarification",
            "questions": [q.model_dump(mode="json") for q in pending.questions],
        }
    elif isinstance(pending, PendingApproval):
        out["pending"] = _plan_json(pending)
    return out


def _plan_json(pending: PendingApproval) -> Json:
    feasibility = {f.scenario: f for f in pending.feasibility}
    scenarios = []
    for s in pending.plan.scenarios:
        check = feasibility.get(s.name)
        scenarios.append(
            {
                "name": s.name,
                "category": s.category.value,
                "share": s.share,
                "description": s.description,
                "feasibility": check.status.value if check else None,
                "feasibility_note": check.reason if check else None,
                "signature": [
                    f"{a.table}.{a.column} {a.metric.value} {a.operator.value} {a.value}"
                    for a in s.signature
                ],
                "source": s.template_ref or "model-proposed",
            }
        )
    return {
        "kind": "approval",
        "plan_version": pending.plan_version,
        "scenarios": scenarios,
        "typical_share": max(0.0, 1.0 - pending.plan.total_share),
        "assumptions": pending.plan.assumptions,
        "errors": pending.errors,
        "explanation": pending.explanation,
    }


def report_json(report: JobReport) -> Json:
    """Summary, rows per table and scenario, and the failed checks of a finished job."""
    final = report.final_report
    failures = [r for r in final.results if r.status is CheckStatus.FAIL] if final else []
    return {
        "summary": report.summary.text if report.summary else None,
        "caveats": report.summary.caveats if report.summary else [],
        "tables": [
            {
                "name": t.name,
                "rows": t.rows,
                "scenarios": {c.scenario: c.rows for c in t.scenario_counts},
            }
            for t in (report.manifest.tables if report.manifest else [])
        ],
        "checks_run": len(final.results) if final else 0,
        "failures": [
            {
                "family": f.family.value,
                "check": f.check,
                "where": f"{f.table}.{f.column}" if f.column else f.table,
                "value": f.value,
                "threshold": f.threshold,
                "scenario": f.scenario,
            }
            for f in failures
        ],
    }


def preview_json(table: pa.Table) -> Json:
    """Column names and rows as lists, values made JSON-safe."""
    rows = [[_cell(v) for v in row.values()] for row in table.to_pylist()]
    return {"columns": table.column_names, "rows": rows}


def _cell(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)
