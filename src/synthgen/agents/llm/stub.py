"""StubClient: recorded responses for tests and offline runs (SPEC 4.1, PLAN A9).

Fixtures live in `<dir>/<SchemaName>.json` as a list of `{"match": str | null, "response": {...}}`.
The stub answers with the last entry whose `match` text appears in the prompt (null always
matches). Choosing by prompt content, not by call count, keeps answers the same when a job
resumes in a new process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from synthgen.agents.llm.client import Message, ModelT, ToolDef, ToolTurn
from synthgen.domain.base import StrictModel
from synthgen.domain.errors import LlmError, LlmOutputError


class FixtureEntry(StrictModel):
    """One recorded response and the prompt text that selects it."""

    match: str | None
    response: dict[str, Any]  # Any: a recorded JSON object, validated against the schema on use


class StubClient:
    """Answers from recorded fixtures; never touches the network."""

    def __init__(self, fixtures_dir: Path) -> None:
        """Read fixtures lazily from this folder."""
        self._dir = fixtures_dir
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, messages: list[Message], schema: type[ModelT]) -> ModelT:
        """The matching recorded response, validated as `schema`."""
        raw = self._pick(schema.__name__, messages)
        try:
            return schema.model_validate(raw)
        except ValidationError as exc:
            raise LlmOutputError(f"recorded {schema.__name__} fails its schema: {exc}") from exc

    def complete_with_tools(self, messages: list[Message], tools: list[ToolDef]) -> ToolTurn:
        """The matching recorded tool turn."""
        del tools
        return ToolTurn.model_validate(self._pick(ToolTurn.__name__, messages))

    def _pick(self, name: str, messages: list[Message]) -> dict[str, Any]:  # Any: recorded JSON
        self.calls.append((name, messages))
        path = self._dir / f"{name}.json"
        try:
            entries = [FixtureEntry.model_validate(e) for e in json.loads(path.read_text("utf-8"))]
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise LlmError(f"cannot read stub fixture {path}: {exc}") from exc
        prompt = "\n".join(m.content for m in messages)
        chosen = [e for e in entries if e.match is None or e.match in prompt]
        if not chosen:
            raise LlmError(f"no stub response in {path} matches the prompt")
        return chosen[-1].response
