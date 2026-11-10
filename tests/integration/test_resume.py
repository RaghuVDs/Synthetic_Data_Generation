from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from synthgen.agents import graph
from synthgen.agents import routing as r
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.app import Application
from synthgen.cli import app as cli
from synthgen.domain.job import JobStatus, UserRequest

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]
REQUEST = UserRequest(
    source="finance",
    tables=("customers", "accounts"),
    text="high balances",
    seed=8,
    auto_approve=True,
)


def _crash(state: JobState, deps: NodeDeps) -> StateUpdate:
    raise RuntimeError("power cut")


def _digests(env: dict[str, str], job_id: str) -> dict[str, str]:
    data = Path(env["SYNTHGEN_PATHS__OUTPUT_DIR"]) / "jobs" / job_id / "data"
    return {
        p.relative_to(data).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in data.rglob("*.parquet")
    }


def _crashed_job(make_app: AppFactory, env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> str:
    """A job that died inside full generation, as if the process had been killed."""
    with monkeypatch.context() as patch:
        patch.setitem(graph.NODES, r.FULL, _crash)
        crashing = make_app(env)
        job = crashing.create_job(REQUEST)
        with pytest.raises(RuntimeError, match="power cut"):
            crashing.run_job(job.job_id)
    assert crashing.get_status(job.job_id).status is JobStatus.GENERATING
    return job.job_id


def test_resume_after_forced_stop_gives_the_same_output(
    make_app: AppFactory, job_env: EnvFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = job_env()
    job_id = _crashed_job(make_app, env, monkeypatch)
    fresh = make_app(env)
    assert fresh.resume_job(job_id).job.status is JobStatus.COMPLETED
    clean = make_app(env).start_job(REQUEST)
    assert clean.job.status is JobStatus.COMPLETED
    assert _digests(env, job_id) == _digests(env, clean.job.job_id)


def test_resume_command(
    make_app: AppFactory, job_env: EnvFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = job_env()
    job_id = _crashed_job(make_app, env, monkeypatch)
    status = CliRunner().invoke(cli, ["status", job_id], env=env)
    assert "generating" in status.output and f"resume {job_id}" in status.output
    result = CliRunner().invoke(cli, ["resume", job_id], env=env)
    assert result.exit_code == 0, result.output
    assert "completed" in result.output


def test_resume_of_a_finished_job_changes_nothing(
    make_app: AppFactory, job_env: EnvFactory
) -> None:
    env = job_env()
    application = make_app(env)
    done = application.start_job(REQUEST)
    assert application.resume_job(done.job.job_id).job == done.job
