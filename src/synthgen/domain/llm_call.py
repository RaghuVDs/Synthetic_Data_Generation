"""One audited LLM call (SPEC 4.5): what was asked, what came back, cost and outcome."""

from __future__ import annotations

from enum import StrEnum

from synthgen.domain.base import StrictModel


class CallStatus(StrEnum):
    """Outcome of one call: accepted, rejected by a gate, or failed in transport."""

    OK = "ok"
    INVALID = "invalid"
    ERROR = "error"


class LlmCallRecord(StrictModel):
    """A row of the `llm_calls` table."""

    job_id: str
    node: str
    model: str
    prompt_version: str
    prompt: str
    response: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int
    status: CallStatus
