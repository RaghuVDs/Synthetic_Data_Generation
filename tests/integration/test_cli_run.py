from __future__ import annotations

import re
from collections.abc import Callable

from typer.testing import CliRunner, Result

from synthgen.cli import app

EnvFactory = Callable[..., dict[str, str]]
RUN = ["run", "--source", "finance", "--tables", "customers,accounts", "--request", "high balances"]


def invoke(env: dict[str, str], args: list[str], answers: str = "") -> Result:
    return CliRunner().invoke(app, args, env={**env, "COLUMNS": "200"}, input=answers)


def job_id(output: str) -> str:
    found = re.search(r"job ([0-9a-f]{12})", output)
    assert found, output
    return found.group(1)


def test_run_with_auto_approve(job_env: EnvFactory) -> None:
    env = job_env()
    result = invoke(env, [*RUN, "--auto-approve", "--rows", "customers=300", "--seed", "4"])
    assert result.exit_code == 0, result.output
    assert "completed" in result.output
    job = job_id(result.output)
    report = invoke(env, ["report", job])
    assert "including 15 high-balance customers" in report.output
    assert "customers" in report.output and "300" in report.output
    status = invoke(env, ["status", job])
    assert "completed" in status.output
    preview = invoke(env, ["preview", job, "--table", "accounts", "--rows", "3"])
    assert preview.exit_code == 0 and "balance" in preview.output


def test_interactive_approval(job_env: EnvFactory) -> None:
    result = invoke(job_env(), RUN, answers="a\n")
    assert result.exit_code == 0, result.output
    assert "high_balance" in result.output and "model-proposed" in result.output
    assert "completed" in result.output


def test_interactive_edit_then_approve(job_env: EnvFactory) -> None:
    result = invoke(job_env(), RUN, answers="e\nhigh_balance=oops\n\nhigh_balance=0.2\n\n")
    assert result.exit_code == 0, result.output
    assert "Not understood" in result.output
    assert "completed" in result.output


def test_interactive_rejection_without_feedback_cancels(job_env: EnvFactory) -> None:
    result = invoke(job_env(), RUN, answers="r\n\n")
    assert "cancelled" in result.output
    assert "plan rejected without feedback" in result.output


def test_unknown_source_is_an_error(job_env: EnvFactory) -> None:
    result = invoke(job_env(), ["run", "--source", "nope", "--tables", "a", "--request", "x"])
    assert result.exit_code == 1
    assert "run `python main.py seed`" in result.output


def test_bad_rows_option(job_env: EnvFactory) -> None:
    result = invoke(job_env(), [*RUN, "--rows", "customers=lots"])
    assert result.exit_code == 2


def test_profile_prints_cards_without_sensitive_values(job_env: EnvFactory) -> None:
    result = invoke(job_env(), ["profile", "--source", "finance", "--tables", "customers"])
    assert result.exit_code == 0, result.output
    assert "mask" in result.output
    assert "@example" not in result.output


def test_profile_shows_fits_and_rare_shares(job_env: EnvFactory) -> None:
    result = invoke(job_env(), ["profile", "--source", "finance", "--tables", "transactions"])
    assert result.exit_code == 0, result.output
    assert "lognorm" in result.output
    assert "1 0.50%" in result.output  # the fraud label is not rounded away
