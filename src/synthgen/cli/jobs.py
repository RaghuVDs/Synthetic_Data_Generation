"""Job commands: run, resume, status, report and preview."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from synthgen.app import Application, JobView
from synthgen.cli.commands import SourceOption, TablesOption, split_names
from synthgen.cli.prompts import ask_answer, ask_decision
from synthgen.cli.render import (
    render_job,
    render_plan,
    render_preview,
    render_questions,
    render_report,
)
from synthgen.cli.runtime import use_app
from synthgen.domain.job import RowTarget, UserRequest
from synthgen.domain.pending import PendingApproval, PendingClarification

JobArgument = Annotated[str, typer.Argument(help="Job ID printed by `run`.")]


def parse_rows(values: list[str]) -> tuple[RowTarget, ...]:
    """`table=N` options -> row targets."""
    targets = []
    for value in values:
        table, sep, rows = value.partition("=")
        if not sep or not rows.strip().isdigit():
            raise typer.BadParameter(f"{value!r} is not table=N", param_hint="--rows")
        targets.append(RowTarget(table=table.strip(), rows=int(rows)))
    return tuple(targets)


def drive(app: Application, view: JobView, console: Console) -> JobView:
    """Answer every pause in the terminal until the job ends."""
    while view.pending is not None:
        pending = view.pending
        if isinstance(pending, PendingClarification):
            render_questions(console, pending)
            view = app.answer_clarification(view.job.job_id, ask_answer())
        elif isinstance(pending, PendingApproval):
            render_plan(console, pending)
            view = app.decide_plan(view.job.job_id, ask_decision())
    return view


def run(  # noqa: PLR0913 - one option per flag in SPEC 9
    *,
    source: SourceOption,
    tables: TablesOption,
    request: Annotated[str, typer.Option("--request", help="What to test, in plain language.")],
    rows: Annotated[list[str] | None, typer.Option("--rows", help="Row target, table=N.")] = None,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 0,
    include_invalid: Annotated[bool, typer.Option("--include-invalid")] = False,
    auto_approve: Annotated[bool, typer.Option("--auto-approve")] = False,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Start a job and see it through, pausing for questions and plan approval."""
    console = Console()
    job_request = UserRequest(
        source=source,
        tables=split_names(tables),
        text=request,
        row_targets=parse_rows(rows or []),
        seed=seed,
        include_invalid=include_invalid,
        auto_approve=auto_approve,
        workers=workers,
    )

    def action(app: Application) -> JobView:
        job = app.create_job(job_request)
        console.print(f"job [bold]{job.job_id}[/] started")
        return drive(app, app.run_job(job.job_id), console)

    render_job(console, use_app(action))


def resume(job_id: JobArgument) -> None:
    """Continue a paused or interrupted job from its last checkpoint."""
    console = Console()
    render_job(console, use_app(lambda app: drive(app, app.resume_job(job_id), console)))


def status(job_id: JobArgument) -> None:
    """Show a job's status."""
    render_job(Console(), use_app(lambda app: app.view(job_id)))


def report(job_id: JobArgument) -> None:
    """Show a job's summary, row counts and failed checks."""
    render_report(Console(), use_app(lambda app: app.get_report(job_id)))


def preview(
    job_id: JobArgument,
    table: Annotated[str, typer.Option("--table", help="Table to show.")],
    rows: Annotated[int, typer.Option("--rows", min=1)] = 10,
) -> None:
    """Show the first rows of a generated table."""
    render_preview(Console(), use_app(lambda app: app.get_preview(job_id, table, rows)))
