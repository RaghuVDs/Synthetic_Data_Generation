"""Node 10, code: generate a sample of up to `dry_run_max_root_entities` root rows."""

from __future__ import annotations

from synthgen.agents.nodes.common import load_spec
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.engine.runner import RunOptions, generate_dataset

DRY_RUN_DIR = "dry_run"


def options(deps: NodeDeps) -> RunOptions:
    """The dry-run options shared by generation and validation."""
    return RunOptions(max_root_rows=deps.settings.generation.dry_run_max_root_entities)


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Write the sample under the job's dry_run folder."""
    out = deps.job_dir(state["job_id"]) / DRY_RUN_DIR
    generate_dataset(load_spec(state, deps), out, options(deps))
    return {}
