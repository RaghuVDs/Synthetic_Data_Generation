"""A job killed inside any of the 15 nodes resumes to the same result (definition of done, T7.5).

The job follows the repair path (a broken overlay fixed by one patch), so every node runs,
the repair analyst included. Each case crashes one node the first time it runs, resumes from
the checkpoint in a fresh application, and compares the Parquet bytes with one clean run.
"""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from synthgen.agents import graph
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.app import Application, build_application
from synthgen.domain.job import ArtifactKind, JobStatus, UserRequest

from .helpers import SETTINGS_FILE, fixture_set
from .test_graph_failures import BROKEN_FK

EnvFactory = Callable[..., dict[str, str]]
AppFactory = Callable[[dict[str, str]], Application]
REQUEST = UserRequest(
    source="finance", tables=("customers", "accounts"), text="high balances", auto_approve=True
)
PATCH = {
    "patches": [
        {
            "path": "/scenarios/0/overrides/0/generator/value",
            "value_json": "1",
            "reason": "point scenario accounts at an existing customer",
        }
    ],
    "plan_change_required": False,
    "explanation": "The overlay pointed accounts at a customer that does not exist.",
}


def _digests(env: dict[str, str], job_id: str) -> dict[str, str]:
    data = Path(env["SYNTHGEN_PATHS__OUTPUT_DIR"]) / "jobs" / job_id / "data"
    return {
        p.relative_to(data).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(data.rglob("*.parquet"))
    }


def _crash_once(real: graph.NodeFn) -> graph.NodeFn:
    calls = {"n": 0}

    def node(state: JobState, deps: NodeDeps) -> StateUpdate:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("power cut")
        return real(state, deps)

    return node


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Stub answers for the repair path: a broken overlay, then the patch that fixes it."""
    return fixture_set(
        tmp_path_factory.mktemp("fx"), OverlaySet=[(None, BROKEN_FK)], SpecPatch=[(None, PATCH)]
    )


@pytest.fixture(scope="module")
def reference(
    fixtures: Path, seeded_dir: Path, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, dict[str, str]]:
    """The fixtures and the digests of an uninterrupted run, built once for every case."""
    root = tmp_path_factory.mktemp("reference")
    data = root / "data"
    data.mkdir()
    shutil.copy(seeded_dir / "finance.db", data / "finance.db")
    env = {
        "SYNTHGEN_PATHS__DATA_DIR": str(data),
        "SYNTHGEN_PATHS__OUTPUT_DIR": str(root / "output"),
        "LLM_PROVIDER": "stub",
        "LLM_STUB_FIXTURES": str(fixtures),
    }
    with build_application(env, SETTINGS_FILE, None) as app:
        view = app.start_job(REQUEST)
        assert view.job.status is JobStatus.COMPLETED
        assert len(app._artifacts.versions(view.job.job_id, ArtifactKind.DRY_RUN_REPORT)) == 2
    return fixtures, _digests(env, view.job.job_id)


@pytest.mark.parametrize("node", list(graph.NODES))
def test_crash_in_any_node_resumes_to_the_same_output(
    node: str,
    reference: tuple[Path, dict[str, str]],
    make_app: AppFactory,
    job_env: EnvFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixtures, digests = reference
    env = job_env(fixtures)
    with monkeypatch.context() as patch:
        patch.setitem(graph.NODES, node, _crash_once(graph.NODES[node]))
        crashing = make_app(env)
        job_id = crashing.create_job(REQUEST).job_id
        with pytest.raises(RuntimeError, match="power cut"):
            crashing.run_job(job_id)
    resumed = make_app(env).resume_job(job_id)
    assert resumed.job.status is JobStatus.COMPLETED, resumed.job.error
    assert _digests(env, job_id) == digests
