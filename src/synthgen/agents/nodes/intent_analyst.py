"""Node 1, LLM: request text to TestIntent; pauses for an answer to blocking questions."""

from __future__ import annotations

from langgraph.types import interrupt

from synthgen.agents.meaning import check_intent
from synthgen.agents.nodes.common import ask
from synthgen.agents.nodes.deps import NodeDeps, StateUpdate
from synthgen.agents.state import JobState
from synthgen.domain.intent import TestIntent
from synthgen.domain.job import ArtifactKind, JobStatus, QuestionAnswer
from synthgen.domain.pending import PendingClarification

ROLE = "intent_analyst"
ANSWER_PREFIX = "ANSWER:"


def run(state: JobState, deps: NodeDeps) -> StateUpdate:
    """Ask first when the last reading left blocking questions; then read the request."""
    clarifications = list(state["clarifications"])
    counters = state["counters"]
    previous = state["intent"]
    if state["status"] is JobStatus.AWAITING_CLARIFICATION and previous is not None:
        questions = previous.blocking_ambiguities
        pending = PendingClarification(kind="clarification", questions=questions)
        answer = interrupt(pending.model_dump(mode="json"))
        asked = " / ".join(q.question for q in questions)
        clarifications.append(QuestionAnswer(question=asked, answer=str(answer)))
        counters = counters.model_copy(update={"clarify": counters.clarify + 1})
    request = state["request"]
    answered = "\n".join(f"Q: {qa.question}\n{ANSWER_PREFIX} {qa.answer}" for qa in clarifications)
    max_rows = deps.settings.generation.max_rows_per_table
    intent = ask(
        state,
        deps,
        ROLE,
        {
            "REQUEST": request.text,
            "SELECTED_TABLES": ", ".join(request.tables),
            "CLARIFICATIONS": answered or "none",
        },
        TestIntent,
        meaning=lambda value: check_intent(value, max_rows),
    )
    if request.include_invalid and not intent.include_invalid:
        intent = intent.model_copy(update={"include_invalid": True})
    deps.artifacts.put(state["job_id"], ArtifactKind.INTENT, intent)
    return {"intent": intent, "clarifications": clarifications, "counters": counters}
