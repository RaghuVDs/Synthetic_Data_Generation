"""Node 15, LLM: a short summary. The model names metrics; code inserts every number."""

from __future__ import annotations

import json
import re

from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.prompts.loader import render
from synthgen.agents.state import JobState
from synthgen.domain.errors import LlmOutputError
from synthgen.domain.job import ArtifactKind
from synthgen.domain.summary import Summary
from synthgen.engine.writer import Manifest, read_manifest

ROLE = "reporter"
PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")


def metrics(state: JobState, manifest: Manifest) -> dict[str, str]:
    """Every number the summary may mention, keyed by placeholder name."""
    report = state["final_report"]
    values: dict[str, str] = {
        "total_rows": f"{sum(t.rows for t in manifest.tables):,}",
        "table_count": str(len(manifest.tables)),
        "checks_run": str(len(report.results)) if report else "0",
        "checks_failed": str(len(report.failures())) if report else "0",
    }
    for table in manifest.tables:
        values[f"rows_{table.name}"] = f"{table.rows:,}"
        for count in table.scenario_counts:
            values[f"scenario_rows_{table.name}_{count.scenario}"] = f"{count.rows:,}"
    return values


def fill(text: str, values: dict[str, str]) -> str:
    """Replace every {placeholder}; an unknown name is an output error."""
    unknown = sorted(set(PLACEHOLDER.findall(text)) - set(values))
    if unknown:
        raise LlmOutputError(f"summary uses unknown metrics: {', '.join(unknown)}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], text)


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Write the summary with real numbers in place of placeholders."""
    values = metrics(state, read_manifest(deps.job_dir(state["job_id"])))
    _, messages = render(
        ROLE,
        {
            "METRICS": json.dumps(sorted(values)),
            "INTENT": state["intent"].model_dump_json() if state["intent"] else "none",
        },
    )
    draft = deps.llm.complete(messages, Summary)
    summary = Summary(
        text=fill(draft.text, values), caveats=[fill(c, values) for c in draft.caveats]
    )
    deps.artifacts.put(state["job_id"], ArtifactKind.SUMMARY, summary)
    return {}
