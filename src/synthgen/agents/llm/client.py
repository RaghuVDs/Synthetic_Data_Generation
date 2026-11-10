"""The LLM client protocol (SPEC 4.1). Agents depend on this, never on a concrete client.

A client handles transport and the provider's own schema gate. It returns the raw reply; the
StructuredLlm layer parses it, checks its meaning, re-prompts and logs (SPEC 4.3, 4.5).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel

from synthgen.domain.base import StrictModel


class ToolCall(StrictModel):
    """One call the model asked for; arguments are JSON text."""

    id: str
    name: str
    arguments: str


class Message(StrictModel):
    """One chat message, including assistant tool calls and tool results."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None


class ToolDef(StrictModel):
    """A tool the model may call, described by a JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]  # Any: a JSON schema document


class Usage(StrictModel):
    """Token counts reported by the provider, when it reports them."""

    prompt_tokens: int | None
    completion_tokens: int | None


class Reply(StrictModel):
    """A completion's text with its cost and timing."""

    text: str
    model: str
    usage: Usage
    latency_ms: int


class ToolTurn(StrictModel):
    """The model's reply in a tool-calling turn: calls to run, or final text."""

    tool_calls: list[ToolCall]
    content: str | None
    model: str
    usage: Usage
    latency_ms: int


class LlmClient(Protocol):
    """Structured completion and tool-calling turns."""

    def complete(
        self, messages: list[Message], schema: type[BaseModel], temperature: float
    ) -> Reply:
        """Ask for JSON matching `schema`; enforced by the provider where it can."""
        ...

    def complete_with_tools(
        self, messages: list[Message], tools: list[ToolDef], temperature: float
    ) -> ToolTurn:
        """Return the model's tool calls, or its final text."""
        ...
