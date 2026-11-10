"""Builds the application for a command and turns typed errors into exit code 1."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import typer
from rich.console import Console

from synthgen.app import Application, build_application
from synthgen.cli.render import render_error
from synthgen.domain.errors import SynthgenError

SETTINGS_FILE = Path("config/settings.yaml")
ENV_FILE = Path(".env")

ResultT = TypeVar("ResultT")


def use_app(action: Callable[[Application], ResultT]) -> ResultT:
    """Run one use case; a SynthgenError is shown and ends the command with exit code 1."""
    try:
        with build_application(os.environ, SETTINGS_FILE, ENV_FILE) as app:
            return action(app)
    except SynthgenError as exc:
        render_error(Console(stderr=True), str(exc))
        raise typer.Exit(code=1) from exc
