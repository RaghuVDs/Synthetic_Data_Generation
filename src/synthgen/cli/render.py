"""Rich rendering of use-case results. The only module that formats output for people."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from synthgen.domain.metadata import SourceOverview


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
