"""Typer commands, Rich rendering and terminal prompts. The only layer that prints."""

import typer

from synthgen.cli import commands, evaluation, jobs

app = typer.Typer(
    name="synthgen",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)
app.command("seed")(commands.seed)
app.command("tables")(commands.tables)
app.command("profile")(commands.profile)
app.command("run")(jobs.run)
app.command("resume")(jobs.resume)
app.command("status")(jobs.status)
app.command("report")(jobs.report)
app.command("preview")(jobs.preview)
app.command("eval")(evaluation.evaluate)


@app.callback()
def root() -> None:
    """Synthetic Data Agent: turn a plain-language test request into labelled synthetic data."""
