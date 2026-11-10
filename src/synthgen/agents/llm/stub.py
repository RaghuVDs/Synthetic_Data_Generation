"""StubClient: recorded responses for tests and offline runs (SPEC 4.1, PLAN A9).

Fixtures live in `<dir>/<SchemaName>.json` as a list of `{"match": str | null, "response": ...}`.
The stub answers with the last entry whose `match` text appears in the prompt (null always
matches). Choosing by prompt content, not by call count, keeps answers the same when a job
resumes in a new process. A response is a JSON object, or a string sent back verbatim (used to
record malformed output). Tool turns use `ToolTurn.json` with `tool_calls` and `content`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from synthgen.agents.llm.client import Message, Reply, ToolCall, ToolDef, ToolTurn, Usage
from synthgen.domain.base import StrictModel
from synthgen.domain.errors import LlmError

STUB_MODEL = "stub"
NO_USAGE = Usage(prompt_tokens=None, completion_tokens=None)


class FixtureEntry(StrictModel):
    """One recorded response and the prompt text that selects it."""

    match: str | None
    response: dict[str, Any] | str  # Any: recorded JSON, validated by the caller's gates


class StubClient:
    """Answers from recorded fixtures; never touches the network."""

    def __init__(self, fixtures_dir: Path) -> None:
        """Read fixtures lazily from this folder."""
        self._dir = fixtures_dir
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(
        self, messages: list[Message], schema: type[BaseModel], temperature: float
    ) -> Reply:
        """The matching recorded response as text."""
        del temperature
        raw = self._pick(schema.__name__, messages)
        text = raw if isinstance(raw, str) else json.dumps(raw)
        return Reply(text=text, model=STUB_MODEL, usage=NO_USAGE, latency_ms=0)

    def complete_with_tools(
        self, messages: list[Message], tools: list[ToolDef], temperature: float
    ) -> ToolTurn:
        """The matching recorded tool turn."""
        del tools, temperature
        raw = self._pick(ToolTurn.__name__, messages)
        if isinstance(raw, str):
            raise LlmError("a recorded tool turn must be a JSON object")
        return ToolTurn(
            tool_calls=[ToolCall.model_validate(c) for c in raw.get("tool_calls", [])],
            content=raw.get("content"),
            model=STUB_MODEL,
            usage=NO_USAGE,
            latency_ms=0,
        )

    def _pick(self, name: str, messages: list[Message]) -> dict[str, Any] | str:  # Any: JSON
        self.calls.append((name, messages))
        path = self._dir / f"{name}.json"
        try:
            entries = [FixtureEntry.model_validate(e) for e in json.loads(path.read_text("utf-8"))]
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise LlmError(f"cannot read stub fixture {path}: {exc}") from exc
        prompt = "\n".join(m.content or "" for m in messages)
        chosen = [e for e in entries if e.match is None or e.match in prompt]
        if not chosen:
            raise LlmError(f"no stub response in {path} matches the prompt")
        return chosen[-1].response
