"""The `eval` command: run the golden set and print its metrics (SPEC 9, 11)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from synthgen.cli.commands import split_names
from synthgen.cli.runtime import use_app
from synthgen.evaluation.run import EvalReport
from synthgen.evaluation.scoring import EvalSummary, JobScore

_METRICS = (
    ("intent accuracy", "intent_accuracy", "share"),
    ("role accuracy", "role_accuracy", "share"),
    ("plan coverage", "plan_coverage", "share"),
    ("first-pass spec validity", "first_pass_spec_validity", "share"),
    ("repair loops per job", "repair_loops_per_job", "number"),
    ("dry-run pass rate", "dry_run_pass_rate", "share"),
    ("tokens per job", "tokens_per_job", "whole"),
    ("LLM latency per job (s)", "llm_latency_ms_per_job", "ms"),
    ("time per job (s)", "seconds_per_job", "number"),
)


def evaluate(
    volumes: Annotated[
        str, typer.Option("--volumes", help="Comma-separated row counts; default from settings.")
    ] = "",
    only: Annotated[str, typer.Option("--only", help="Comma-separated request names.")] = "",
    source: Annotated[str, typer.Option("--source", help="Only requests on these sources.")] = "",
) -> None:
    """Run the golden set on both sources at each volume and print the metrics."""
    console = Console()
    sizes = [_volume(v) for v in split_names(volumes)]

    def show(score: JobScore) -> None:
        console.print(
            f"{score.name} @ {score.volume:,} rows: {score.status.value}"
            + (f" ({score.error})" if score.error else "")
        )

    report, path = use_app(
        lambda app: app.run_eval(sizes, split_names(only), split_names(source), show)
    )
    render_eval(console, report, path)


def _volume(text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        raise typer.BadParameter(f"{text!r} is not a row count", param_hint="--volumes")
    return int(text)


def render_eval(console: Console, report: EvalReport, path: Path) -> None:
    """One row per job, then the metrics overall and per volume."""
    jobs = Table(title="golden jobs", title_justify="left")
    for heading in ("request", "rows", "status", "intent", "roles", "coverage", "repairs"):
        jobs.add_column(heading)
    for s in report.scores:
        jobs.add_row(
            s.name,
            f"{s.volume:,}",
            s.status.value,
            _cell(s.intent_correct, "share"),
            _cell(s.role_accuracy, "share"),
            _cell(s.plan_coverage, "share"),
            str(s.repair_loops),
        )
    console.print(jobs)
    columns = {"all": report.summary} | {f"{v:,} rows": m for v, m in report.by_volume.items()}
    console.print(_metrics(columns))
    console.print(f"report saved to {path}")


def _metrics(columns: dict[str, EvalSummary]) -> Table:
    table = Table(title="metrics", title_justify="left")
    table.add_column("metric")
    for name in columns:
        table.add_column(name, justify="right")
    table.add_row("jobs completed", *(f"{m.completed}/{m.jobs}" for m in columns.values()))
    for label, field, kind in _METRICS:
        table.add_row(label, *(_cell(getattr(m, field), kind) for m in columns.values()))
    return table


def _cell(value: float | bool | None, kind: str) -> str:
    if value is None:
        return "-"
    number = float(value)
    if kind == "share":
        return f"{number:.0%}"
    if kind == "whole":
        return f"{number:,.0f}"
    if kind == "ms":
        return f"{number / 1000:.1f}"
    return f"{number:.1f}"
