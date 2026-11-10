"""Node 11, code: run every validation family on the dry-run sample."""

from __future__ import annotations

from synthgen.agents.nodes.common import load_spec
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.nodes.dry_run import DRY_RUN_DIR, options
from synthgen.agents.state import JobState
from synthgen.domain.job import ArtifactKind
from synthgen.validation.report import validate_output


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Validate the sample and keep the report."""
    out = deps.job_dir(state["job_id"]) / DRY_RUN_DIR
    report = validate_output(load_spec(state, deps), out, options(deps), deps.settings.validation)
    deps.artifacts.put(state["job_id"], ArtifactKind.DRY_RUN_REPORT, report)
    return {"dry_run_report": report}
