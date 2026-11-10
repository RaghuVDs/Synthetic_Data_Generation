"""Typer commands, Rich rendering and terminal prompts. The only layer that prints."""

import typer

app = typer.Typer(
    name="synthgen",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def root() -> None:
    """Synthetic Data Agent: turn a plain-language test request into labelled synthetic data."""
