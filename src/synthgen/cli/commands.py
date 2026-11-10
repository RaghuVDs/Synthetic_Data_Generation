"""Source commands: seed, tables and profile."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from synthgen.cli.render import render_cards, render_overview, render_seed
from synthgen.cli.runtime import use_app

SourceOption = Annotated[str, typer.Option("--source", help="Source name, e.g. finance.")]
TablesOption = Annotated[str, typer.Option("--tables", help="Comma-separated table names.")]


def split_names(text: str) -> tuple[str, ...]:
    """`a, b,c` -> ("a", "b", "c")."""
    return tuple(name.strip() for name in text.split(",") if name.strip())


def seed() -> None:
    """Build the fake source databases (finance.db and orders.db)."""
    render_seed(Console(), use_app(lambda app: app.seed()))


def tables(source: SourceOption) -> None:
    """List a source's tables and relationships."""
    render_overview(Console(), use_app(lambda app: app.list_tables(source)))


def profile(
    source: SourceOption,
    tables: TablesOption,
    refresh: Annotated[bool, typer.Option("--refresh", help="Ignore cached profiles.")] = False,
) -> None:
    """Profile tables and print their LLM-safe profile cards."""
    cards = use_app(lambda app: app.profile_tables(source, split_names(tables), refresh))
    render_cards(Console(), cards)
