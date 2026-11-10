from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from rich.console import Console

from synthgen.app import Application
from synthgen.cli.render import render_plan
from synthgen.config import load_settings, read_environment
from synthgen.domain.errors import SynthgenError
from synthgen.domain.job import Approval, ApprovalDecision, JobStatus, UserRequest
from synthgen.domain.pending import PendingApproval

from .helpers import SETTINGS_FILE, fixture_set, skeleton

EnvFactory = Callable[..., dict[str, str]]
START = datetime(2026, 11, 10, 9, 0, tzinfo=UTC)
REQUEST = UserRequest(source="finance", tables=("customers", "accounts"), text="high balances")


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now


@pytest.fixture
def clocked(job_env: EnvFactory) -> Iterator[tuple[Application, Clock]]:
    clock = Clock()
    settings = load_settings(SETTINGS_FILE, read_environment(job_env(), None))
    app = Application(settings, clock=clock)
    yield app, clock
    app.close()


def approve() -> Approval:
    return Approval(decision=ApprovalDecision.APPROVED, decided_at=START)


def test_stale_plan_expires(clocked: tuple[Application, Clock]) -> None:
    app, clock = clocked
    view = app.start_job(REQUEST)
    assert isinstance(view.pending, PendingApproval)
    clock.now = START + timedelta(days=8)
    expired = app.view(view.job.job_id)
    assert expired.job.status is JobStatus.CANCELLED
    assert expired.job.error == "plan expired after 7 days without a decision"
    assert expired.pending is None
    with pytest.raises(SynthgenError, match="has ended"):
        app.decide_plan(view.job.job_id, approve())


def test_plan_within_the_window_can_be_approved(clocked: tuple[Application, Clock]) -> None:
    app, clock = clocked
    view = app.start_job(REQUEST)
    clock.now = START + timedelta(days=6)
    assert app.decide_plan(view.job.job_id, approve()).job.status is JobStatus.COMPLETED


def test_plan_table_shows_feasibility(job_env: EnvFactory, tmp_path: Path) -> None:
    plan = skeleton("ScenarioPlan")
    burst = dict(plan["scenarios"][0])
    burst.update(
        name="opening_burst",
        primitives=["sequence"],
        levers=[
            {"table": "accounts", "column": "opened_date", "primitive": "sequence", "change": "x"}
        ],
        signature=[],
        share=0.01,
    )
    plan["scenarios"].append(burst)
    settings = load_settings(
        SETTINGS_FILE, job_env(fixture_set(tmp_path / "fx", ScenarioPlan=[(None, plan)]))
    )
    with Application(settings) as app:
        pending = app.start_job(REQUEST).pending
    assert isinstance(pending, PendingApproval)
    console = Console(record=True, width=200)
    render_plan(console, pending)
    text = console.export_text()
    assert "expressible" in text and "not_expressible" in text
    assert "opening_burst is not_expressible" in text
    assert "try: use timing_shift instead of sequence" in text
