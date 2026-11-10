"""Node 14, code: run every validation family on the full output."""

from __future__ import annotations

from synthgen.agents.nodes.common import load_spec
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.job import ArtifactKind
from synthgen.engine.runner import RunOptions
from synthgen.validation.report import validate_output


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Validate the output and keep the report."""
    out = deps.job_dir(state["job_id"])
    report = validate_output(load_spec(state, deps), out, RunOptions(), deps.settings.validation)
    deps.artifacts.put(state["job_id"], ArtifactKind.FINAL_REPORT, report)
    return {"final_report": report}
