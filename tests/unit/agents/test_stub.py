from __future__ import annotations

import json
from pathlib import Path

import pytest

from synthgen.agents.llm.client import Message
from synthgen.agents.llm.stub import StubClient
from synthgen.domain.errors import LlmError, LlmOutputError
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
    assert stub.complete(_ask("first"), Summary).text == "default"
    assert stub.complete(_ask("ANSWER: yes"), Summary).text == "answered"
    assert stub.complete(_ask("first"), Summary).text == "default"
    assert [name for name, _ in stub.calls] == ["Summary"] * 3


def test_invalid_recorded_response_is_an_output_error(tmp_path: Path) -> None:
    _write(tmp_path, "Summary", [{"match": None, "response": {"text": "x"}}])
    with pytest.raises(LlmOutputError):
        StubClient(tmp_path).complete(_ask("q"), Summary)


def test_missing_or_unmatched_fixture_fails(tmp_path: Path) -> None:
    with pytest.raises(LlmError):
        StubClient(tmp_path).complete(_ask("q"), Summary)
    _write(tmp_path, "Summary", [{"match": "zzz", "response": {"text": "x", "caveats": []}}])
    with pytest.raises(LlmError):
        StubClient(tmp_path).complete(_ask("q"), Summary)


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
    turn = StubClient(tmp_path).complete_with_tools(_ask("q"), [])
    assert turn.tool_calls[0].name == "get_column_profile"
