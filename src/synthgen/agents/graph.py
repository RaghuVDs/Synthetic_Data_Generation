"""The 15-node LangGraph graph with a SQLite checkpointer; thread ID = job ID (SPEC 3, 2).

Each node runs inside `_tracked`, which turns typed errors into a terminal status, evaluates the
pure router for the node and stores the route in the state. Edges then follow `state.route`.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pydantic import TypeAdapter

from synthgen.agents import routing as r
from synthgen.agents.nodes import (
    approval_gate,
    baseline_compiler,
    context_builder,
    dry_run,
    final_validation,
    finish,
    full_generation,
    intent_analyst,
    overlay_writer,
    profiler,
    repair_analyst,
    reporter,
    scenario_planner,
    semantic_mapper,
    spec_validator,
    validator,
)
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState, checkpoint_types, initial_state
from synthgen.domain.errors import LlmOutputError, SynthgenError
from synthgen.domain.job import JobStatus, UserRequest
from synthgen.domain.pending import Pending

logger = logging.getLogger(__name__)
NodeFn = Callable[[JobState, NodeDeps], StateUpdate]
NODES: dict[str, NodeFn] = {
    r.INTENT: intent_analyst.run,
    r.CONTEXT: context_builder.run,
    r.PROFILER: profiler.run,
    r.MAPPER: semantic_mapper.run,
    r.PLANNER: scenario_planner.run,
    r.APPROVAL: approval_gate.run,
    r.BASELINE: baseline_compiler.run,
    r.OVERLAY: overlay_writer.run,
    r.SPEC_VALIDATOR: spec_validator.run,
    r.DRY_RUN: dry_run.run,
    r.VALIDATOR: validator.run,
    r.REPAIR: repair_analyst.run,
    r.FULL: full_generation.run,
    r.FINAL: final_validation.run,
    r.REPORTER: reporter.run,
}
GraphApp = CompiledStateGraph[JobState, None, JobState, JobState]


class StateNode(Protocol):
    """A graph node: takes the state by name, returns a partial update."""

    def __call__(self, state: JobState) -> StateUpdate:
        """Run the node."""
        ...


def _tracked(name: str, fn: NodeFn, deps: NodeDeps) -> StateNode:
    def node(state: JobState) -> StateUpdate:
        log = logging.LoggerAdapter(logger, {"job_id": state["job_id"], "node": name})
        log.info("node %s started for job %s", name, state["job_id"])
        try:
            update = fn(state, deps)
        except LlmOutputError as exc:
            update = {"status": JobStatus.NEEDS_REVIEW, "error": str(exc)}
        except SynthgenError as exc:
            update = {"status": JobStatus.FAILED, "error": str(exc)}
        merged: JobState = {**state, **update}  # type: ignore[typeddict-item]
        route = r.route_after(name, merged, deps.settings.agent.caps)
        deps.jobs.set_status(state["job_id"], route.status, route.reason)
        log.info("node %s routes to %s (%s)", name, route.next, route.status)
        return {**update, "route": route, "status": route.status}

    return node


def _next(state: JobState) -> str:
    route = state["route"]
    if route is None:
        raise SynthgenError("a node finished without a route")
    return route.next


def build_graph(deps: NodeDeps, checkpointer: SqliteSaver) -> GraphApp:
    """Wire every node, its routed edges and the finish step."""
    graph = StateGraph(JobState)
    for name, fn in NODES.items():
        graph.add_node(name, _tracked(name, fn, deps))
        graph.add_conditional_edges(name, _next, [*NODES, r.FINISH])

    def finish_node(state: JobState) -> StateUpdate:
        return finish.run(state, deps)

    graph.add_node(r.FINISH, finish_node)
    graph.add_edge(START, r.INTENT)
    graph.add_edge(r.FINISH, END)
    return graph.compile(checkpointer=checkpointer)


def open_checkpointer(path: Path) -> SqliteSaver:
    """A SQLite checkpointer that only deserialises the state's own model and enum classes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types())
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=serde)


class JobRunner:
    """Starts, resumes and inspects graph runs by job ID."""

    def __init__(self, deps: NodeDeps, checkpointer: SqliteSaver) -> None:
        """Compile the graph once for this process."""
        self._checkpointer = checkpointer
        self._graph = build_graph(deps, checkpointer)

    def close(self) -> None:
        """Close the checkpoint database connection."""
        self._checkpointer.conn.close()

    @staticmethod
    def _config(job_id: str) -> Any:  # Any: LangGraph's RunnableConfig mapping
        return {"configurable": {"thread_id": job_id}}

    def start(self, job_id: str, request: UserRequest) -> None:
        """Run a new job until it pauses or ends."""
        self._graph.invoke(initial_state(job_id, request), self._config(job_id))

    def resume(self, job_id: str, answer: Any = None) -> None:  # Any: JSON answer for the pause
        """Continue after a pause (with its answer) or after a crash (with no answer)."""
        command: Command[Any] | None = None if answer is None else Command(resume=answer)
        self._graph.invoke(command, self._config(job_id))

    def pending(self, job_id: str) -> Pending | None:
        """What the paused job waits for; None when it is not paused."""
        snapshot = self._graph.get_state(self._config(job_id))
        if not snapshot.interrupts:
            return None
        return TypeAdapter(Pending).validate_python(snapshot.interrupts[0].value)

    def has_run(self, job_id: str) -> bool:
        """True once the job has a checkpoint."""
        return bool(self._graph.get_state(self._config(job_id)).values)
