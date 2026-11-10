"""The LLM client protocol (SPEC 4.1). Agents depend on this, never on a concrete client."""

from __future__ import annotations

from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from synthgen.domain.base import StrictModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class Message(StrictModel):
    """One chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ToolDef(StrictModel):
    """A tool the model may call, described by a JSON schema."""

    name: str
    description: str
    parameters: dict[str, Any]  # Any: a JSON schema document


class ToolCall(StrictModel):
    """One call the model asked for; arguments are JSON text."""

    id: str
    name: str
    arguments: str


class ToolTurn(StrictModel):
    """The model's reply in a tool-calling turn: calls to run, or final text."""

    tool_calls: list[ToolCall]
    content: str | None


class LlmClient(Protocol):
    """Structured completion and tool-calling turns."""

    def complete(self, messages: list[Message], schema: type[ModelT]) -> ModelT:
        """Return a validated instance of `schema`."""
        ...

    def complete_with_tools(self, messages: list[Message], tools: list[ToolDef]) -> ToolTurn:
        """Return the model's tool calls, or its final text."""
        ...
