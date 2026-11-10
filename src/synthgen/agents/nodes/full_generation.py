"""Node 13, code: generate every row and write Parquet, the manifest and spec.json."""

from __future__ import annotations

from synthgen.agents.nodes.common import load_spec
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.engine.runner import RunOptions, generate_dataset

SPEC_FILE = "spec.json"


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Write the full output under the job folder."""
    out = deps.job_dir(state["job_id"])
    spec = load_spec(state, deps)
    generate_dataset(spec, out, RunOptions())
    (out / SPEC_FILE).write_text(spec.model_dump_json(indent=2), encoding="utf-8")
    return {}
