from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthgen.agents.llm.client import Message
from synthgen.agents.llm.stub import StubClient
from synthgen.domain.errors import LlmError
from synthgen.domain.summary import Summary


def _write(folder: Path, name: str, entries: list[dict[str, object]]) -> None:
    (folder / f"{name}.json").write_text(json.dumps(entries), encoding="utf-8")


def _ask(text: str) -> list[Message]:
    return [Message(role="system", content="role"), Message(role="user", content=text)]


def test_last_matching_entry_wins(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "Summary",
        [
            {"match": None, "response": {"text": "default", "caveats": []}},
            {"match": "ANSWER:", "response": {"text": "answered", "caveats": []}},
            {"match": "never", "response": {"text": "unused", "caveats": []}},
        ],
    )
    stub = StubClient(tmp_path)
    assert json.loads(stub.complete(_ask("first"), Summary, 0.0).text)["text"] == "default"
    assert json.loads(stub.complete(_ask("ANSWER: yes"), Summary, 0.0).text)["text"] == "answered"
    assert [name for name, _ in stub.calls] == ["Summary", "Summary"]


def test_string_responses_are_sent_verbatim(tmp_path: Path) -> None:
    _write(tmp_path, "Summary", [{"match": None, "response": "not json"}])
    assert StubClient(tmp_path).complete(_ask("q"), Summary, 0.0).text == "not json"


def test_missing_or_unmatched_fixture_fails(tmp_path: Path) -> None:
    with pytest.raises(LlmError):
        StubClient(tmp_path).complete(_ask("q"), Summary, 0.0)
    _write(tmp_path, "Summary", [{"match": "zzz", "response": {"text": "x", "caveats": []}}])
    with pytest.raises(LlmError):
        StubClient(tmp_path).complete(_ask("q"), Summary, 0.0)


def test_tool_turns(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "ToolTurn",
        [
            {
                "match": None,
                "response": {
                    "tool_calls": [{"id": "1", "name": "get_column_profile", "arguments": "{}"}],
                    "content": None,
                },
            }
        ],
    )
    turn = StubClient(tmp_path).complete_with_tools(_ask("q"), [], 0.3)
    assert turn.tool_calls[0].name == "get_column_profile"
    _write(tmp_path, "ToolTurn", [{"match": None, "response": "oops"}])
    with pytest.raises(LlmError):
        StubClient(tmp_path).complete_with_tools(_ask("q"), [], 0.3)
