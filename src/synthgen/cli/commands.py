"""CLI commands: parse options, call one use case, render the result."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, TypeVar

import typer
from rich.console import Console

from synthgen.app import Application, build_application
from synthgen.cli.render import render_error, render_overview, render_seed
from synthgen.domain.errors import SynthgenError

SETTINGS_FILE = Path("config/settings.yaml")
ENV_FILE = Path(".env")

ResultT = TypeVar("ResultT")


def _run(action: Callable[[Application], ResultT]) -> ResultT:
    """Build the application and run one use case; typed errors become exit code 1."""
    try:
        return action(build_application(os.environ, SETTINGS_FILE, ENV_FILE))
    except SynthgenError as exc:
        render_error(Console(stderr=True), str(exc))
        raise typer.Exit(code=1) from exc


def seed() -> None:
    """Build the fake source databases (finance.db and orders.db)."""
    render_seed(Console(), _run(lambda app: app.seed()))


def tables(
    source: Annotated[str, typer.Option("--source", help="Source name, e.g. finance.")],
) -> None:
    """List a source's tables and relationships."""
    render_overview(Console(), _run(lambda app: app.list_tables(source)))
