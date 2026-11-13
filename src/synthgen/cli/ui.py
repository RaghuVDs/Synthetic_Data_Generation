"""The `ui` command: serve the local web page until Ctrl+C."""

from __future__ import annotations

import os
from typing import Annotated

import typer
from rich.console import Console

from synthgen.app import Application, build_application
from synthgen.cli.runtime import ENV_FILE, SETTINGS_FILE, use_app
from synthgen.web.server import serve


def _make_app() -> Application:
    return build_application(os.environ, SETTINGS_FILE, ENV_FILE)


def ui(
    port: Annotated[int | None, typer.Option("--port", help="Port; default from settings.")] = None,
) -> None:
    """Open a local web page to pick tables, describe the test and generate data."""
    console = Console()
    conf = use_app(lambda app: app.settings.ui)
    server = serve(_make_app, conf.host, port or conf.port)
    console.print(f"open http://{conf.host}:{server.server_address[1]}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("stopped")
    finally:
        server.server_close()
