"""Rich rendering of use-case results. The only module that formats output for people."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa
from rich.console import Console
from rich.table import Table

from synthgen.app import JobReport, JobView
from synthgen.domain.metadata import SourceOverview
from synthgen.domain.pending import PendingApproval, PendingClarification
from synthgen.profiling.card import CardColumn, LlmSafeProfile, format_share


def render_seed(console: Console, counts: dict[str, dict[str, int]]) -> None:
    """Show the row counts of each seeded source."""
    for source, tables in counts.items():
        table = Table(title=f"{source}.db", title_justify="left")
        table.add_column("table")
        table.add_column("rows", justify="right")
        for name, rows in tables.items():
            table.add_row(name, f"{rows:,}")
        console.print(table)


def render_overview(console: Console, overview: SourceOverview) -> None:
    """Show a source's tables and relationships."""
    tables = Table(title=f"{overview.source}: {overview.description}", title_justify="left")
    for heading in ("table", "rows", "columns", "description"):
        tables.add_column(heading, justify="right" if heading in {"rows", "columns"} else "left")
    for t in overview.tables:
        tables.add_row(t.name, f"{t.row_count:,}", str(t.column_count), t.description)
    console.print(tables)

    links = Table(title="relationships", title_justify="left")
    for heading in ("child", "parent", "nullable", "sizes child", "from"):
        links.add_column(heading)
    for r in overview.relationships:
        links.add_row(
            f"{r.child_table}.{r.child_column}",
            f"{r.parent_table}.{r.parent_column}",
            "yes" if r.nullable else "no",
            "yes" if r.drives_cardinality else "",
            r.origin,
        )
    console.print(links)


def render_error(console: Console, message: str) -> None:
    """Show a failure in a way that stands out."""
    console.print(f"[bold red]error:[/] {message}", highlight=False)


def _fmt(value: float) -> str:
    return f"{value:,.4g}"


def render_cards(console: Console, cards: Sequence[LlmSafeProfile]) -> None:
    """One table per profile card; sensitive columns show only their mask and lengths."""
    for card in cards:
        table = Table(title=f"{card.table} ({card.row_count:,} rows)", title_justify="left")
        headings = ("column", "type", "semantic", "class", "nulls", "distinct", "summary", "fit")
        for heading in headings:
            table.add_column(heading)
        for c in card.columns:
            table.add_row(
                c.name,
                c.type,
                c.semantic_type or "",
                c.sensitivity,
                f"{c.null_rate:.1%}",
                f"{c.distinct_count:,}",
                _summary(c),
                c.best_fit or "",
            )
        console.print(table)


def _summary(column: CardColumn) -> str:
    if column.mask is not None:
        low, high = column.length_range or (0, 0)
        return f"mask {column.mask}, length {low}-{high}"
    if column.five_number is not None:
        return " | ".join(_fmt(v) for v in column.five_number)
    if column.date_range is not None:
        return f"{column.date_range[0]} .. {column.date_range[1]}"
    if column.top_values:
        return ", ".join(f"{v.value} {format_share(v.share)}" for v in column.top_values)
    return ""


def render_questions(console: Console, pending: PendingClarification) -> None:
    """The intent analyst's blocking questions."""
    console.print("[bold]The request leaves these questions open:[/]")
    for q in pending.questions:
        options = f" (options: {', '.join(q.options)})" if q.options else ""
        console.print(f"  - {q.question}{options}", highlight=False)


def render_plan(console: Console, pending: PendingApproval) -> None:
    """The plan as SPEC 9 describes it, plus any problems with the last edit or repair."""
    table = Table(title=f"plan v{pending.plan_version}", title_justify="left")
    for heading in ("scenario", "category", "share", "feasibility", "signature", "source"):
        table.add_column(heading)
    for s in pending.plan.scenarios:
        signature = "; ".join(
            f"{a.table}.{a.column} {a.metric} {a.operator} {a.value}" for a in s.signature
        )
        table.add_row(
            s.name, s.category, f"{s.share:.1%}", "-", signature, s.template_ref or "model-proposed"
        )
    console.print(table)
    console.print(f"typical rows: {1 - pending.plan.total_share:.1%}")
    for check in pending.feasibility:
        if check.reason:
            hint = f" (try: {check.alternative})" if check.alternative else ""
            console.print(
                f"[yellow]{check.scenario} is {check.status}:[/] {check.reason}{hint}",
                highlight=False,
            )
    for assumption in pending.plan.assumptions:
        console.print(f"assumption: {assumption}", highlight=False)
    if pending.explanation:
        console.print(
            f"[yellow]repair asks for a plan change:[/] {pending.explanation}", highlight=False
        )
    for error in pending.errors:
        console.print(f"[red]previous edit rejected:[/] {error}", highlight=False)


def render_job(console: Console, view: JobView) -> None:
    """A job's ID, status and, when it stopped early, why."""
    job = view.job
    style = (
        "green" if job.status == "completed" else "yellow" if not job.status.is_terminal else "red"
    )
    console.print(f"job {job.job_id}: [{style}]{job.status}[/]")
    if job.error:
        console.print(f"reason: {job.error}", highlight=False)
    if not job.status.is_terminal and view.pending is None:
        console.print(f"continue with: python main.py resume {job.job_id}")


def render_report(console: Console, report: JobReport) -> None:
    """Summary, rows per table and scenario, and every failed check."""
    render_job(console, JobView(job=report.job, pending=None))
    if report.summary:
        console.print(report.summary.text, highlight=False)
        for caveat in report.summary.caveats:
            console.print(f"caveat: {caveat}", highlight=False)
    if report.manifest:
        rows = Table(title="rows", title_justify="left")
        for heading in ("table", "rows", "scenarios"):
            rows.add_column(heading)
        for t in report.manifest.tables:
            scenarios = ", ".join(f"{c.scenario} {c.rows:,}" for c in t.scenario_counts)
            rows.add_row(t.name, f"{t.rows:,}", scenarios)
        console.print(rows)
    if report.final_report:
        failures = report.final_report.failures()
        console.print(f"checks: {len(report.final_report.results)} run, {len(failures)} failed")
        for f in failures:
            console.print(
                f"[red]{f.family} {f.check}[/] {f.table}.{f.column}: {f.detail or f.value}",
                highlight=False,
            )


def render_preview(console: Console, data: pa.Table) -> None:
    """Rows of a generated table."""
    table = Table()
    for name in data.column_names:
        table.add_column(name)
    for row in data.to_pylist():
        table.add_row(*("" if v is None else str(v) for v in row.values()))
    console.print(table)
