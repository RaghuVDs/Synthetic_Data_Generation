from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from pydantic import BaseModel

from synthgen.agents.llm.client import Message, Reply, ToolDef, ToolTurn, Usage
from synthgen.agents.llm.retry import RateLimiter, backoff_seconds
from synthgen.agents.llm.schema import strict_schema
from synthgen.agents.llm.structured import CallContext, StructuredLlm, numbered, parse
from synthgen.config import load_settings
from synthgen.domain.errors import ConfigError, LlmError, LlmOutputError
from synthgen.domain.intent import TestIntent
from synthgen.domain.llm_call import CallStatus, LlmCallRecord
from synthgen.domain.patch import SpecPatch
from synthgen.domain.plan import Assertion, ScenarioPlan
from synthgen.domain.semantic import SemanticMap
from synthgen.domain.summary import Summary
from synthgen.spec.models import OverlaySet
from synthgen.storage.db import StateDatabase
from synthgen.storage.llm_call_log import LlmCallLog

ROOT = Path(__file__).resolve().parents[4]
CTX = CallContext(job_id="job", node="reporter", prompt_version="v1")
MESSAGES = [Message(role="system", content="s"), Message(role="user", content="u")]
GOOD = json.dumps({"text": "fine", "caveats": []})


class Scripted:
    """A client that replies with a fixed script and records what it saw."""

    def __init__(self, *texts: str | Exception) -> None:
        self.texts = list(texts)
        self.seen: list[list[Message]] = []

    def complete(self, messages: list[Message], schema: object, temperature: float) -> Reply:
        self.seen.append(list(messages))
        item = self.texts.pop(0)
        if isinstance(item, Exception):
            raise item
        return Reply(
            text=item, model="m", usage=Usage(prompt_tokens=3, completion_tokens=2), latency_ms=7
        )

    def complete_with_tools(
        self, messages: list[Message], tools: list[ToolDef], temperature: float
    ) -> ToolTurn:
        self.seen.append(list(messages))
        return ToolTurn(
            tool_calls=[],
            content="done",
            model="m",
            usage=Usage(prompt_tokens=1, completion_tokens=1),
            latency_ms=1,
        )


class MemoryLog:
    def __init__(self) -> None:
        self.entries: list[LlmCallRecord] = []

    def record(self, entry: LlmCallRecord) -> None:
        self.entries.append(entry)


def test_first_good_answer_is_returned_and_logged() -> None:
    log = MemoryLog()
    value = StructuredLlm(Scripted(GOOD), log, 3).ask(CTX, MESSAGES, Summary, 0.0)
    assert value.text == "fine"
    [entry] = log.entries
    assert (entry.status, entry.node, entry.prompt_tokens, entry.latency_ms) == (
        CallStatus.OK,
        "reporter",
        3,
        7,
    )


def test_bad_json_is_reprompted_with_numbered_errors() -> None:
    client, log = Scripted("not json", GOOD), MemoryLog()
    assert StructuredLlm(client, log, 3).ask(CTX, MESSAGES, Summary, 0.0).text == "fine"
    retry = client.seen[1]
    assert retry[-2] == Message(role="assistant", content="not json")
    assert retry[-1].content is not None and retry[-1].content.startswith(
        "Your previous answer was rejected"
    )
    assert "1. " in retry[-1].content
    assert [e.status for e in log.entries] == [CallStatus.INVALID, CallStatus.OK]


def test_meaning_failures_are_reprompted() -> None:
    client = Scripted(GOOD, json.dumps({"text": "better", "caveats": []}))
    value = StructuredLlm(client, MemoryLog(), 3).ask(
        CTX, MESSAGES, Summary, 0.0, meaning=lambda s: ["too short"] if s.text == "fine" else []
    )
    assert value.text == "better"
    assert "1. too short" in (client.seen[1][-1].content or "")


def test_cap_raises_output_error() -> None:
    log = MemoryLog()
    with pytest.raises(LlmOutputError, match="after 3 attempts"):
        StructuredLlm(Scripted("x", "y", "z"), log, 3).ask(CTX, MESSAGES, Summary, 0.0)
    assert len(log.entries) == 3


def test_transport_errors_are_logged_and_raised() -> None:
    log = MemoryLog()
    with pytest.raises(LlmError):
        StructuredLlm(Scripted(LlmError("down")), log, 3).ask(CTX, MESSAGES, Summary, 0.0)
    assert [e.status for e in log.entries] == [CallStatus.ERROR]


def test_tool_turns_are_logged() -> None:
    log = MemoryLog()
    turn = StructuredLlm(Scripted(), log, 3).tool_turn(CTX, MESSAGES, [], 0.3)
    assert turn.content == "done"
    assert json.loads(log.entries[0].response or "")["content"] == "done"


def test_numbered() -> None:
    assert numbered(["a", "b"]) == "1. a\n2. b"


@pytest.mark.parametrize(
    "model", [TestIntent, SemanticMap, ScenarioPlan, OverlaySet, SpecPatch, Summary]
)
def test_every_llm_schema_has_a_strict_form(model: type[BaseModel]) -> None:
    banned = {"oneOf", "const", "pattern", "minimum", "default", "discriminator", "minLength"}
    assert not banned & _keys(strict_schema(model))


def _keys(node: object) -> set[str]:
    if isinstance(node, dict):
        found = set()
        for key, value in node.items():
            if key not in {"properties", "$defs"}:
                found.add(key)
            found |= (
                _keys(value)
                if key not in {"properties", "$defs"}
                else set().union(*(_keys(v) for v in value.values()))
            )
        return found
    if isinstance(node, list):
        return set().union(*(_keys(item) for item in node)) if node else set()
    return set()


class WithDefault(BaseModel):
    value: int = 1


class TwoLists(BaseModel):
    model_config = {"extra": "forbid"}
    value: list[int] | list[str]


class IntOrFloat(BaseModel):
    model_config = {"extra": "forbid"}
    value: int | float


@pytest.mark.parametrize("model", [TwoLists, IntOrFloat])
def test_strict_schema_rejects_ambiguous_unions(model: type[BaseModel]) -> None:
    with pytest.raises(ConfigError, match="cannot be told apart"):
        strict_schema(model)


def test_strict_schema_rejects_models_with_optional_fields() -> None:
    with pytest.raises(ConfigError):
        strict_schema(WithDefault)


def test_backoff_and_rate_limiter() -> None:
    llm = load_settings(ROOT / "config" / "settings.yaml", {}).llm
    rng = random.Random(1)
    assert backoff_seconds(0, "3", llm, rng) == 3.0
    assert backoff_seconds(0, "9999", llm, rng) == llm.backoff_max_seconds
    delays = [backoff_seconds(a, None, llm, rng) for a in range(4)]
    assert all(0 < d <= llm.backoff_max_seconds for d in delays)
    assert delays[3] > delays[0]
    now = [0.0]
    slept: list[float] = []
    limiter = RateLimiter(60, lambda: now[0], slept.append)
    limiter.wait()
    limiter.wait()
    assert slept == [1.0]


def test_call_log_round_trip(tmp_path: Path) -> None:
    log = LlmCallLog(StateDatabase(tmp_path / "state.db"))
    entry = LlmCallRecord(
        job_id="j",
        node="n",
        model="m",
        prompt_version="v1",
        prompt="[]",
        response=None,
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=0,
        status=CallStatus.ERROR,
    )
    log.record(entry)
    assert log.for_job("j") == [entry]
    assert log.for_job("other") == []


def test_retries_carry_only_the_latest_answer() -> None:
    client = Scripted("first bad", "second bad", GOOD)
    StructuredLlm(client, MemoryLog(), 3).ask(CTX, MESSAGES, Summary, 0.0)
    third = client.seen[2]
    assert len(third) == len(MESSAGES) + 2
    assert third[-2].content == "second bad"


def test_union_errors_are_collapsed() -> None:
    bad = {"table": "t", "column": "c", "metric": "value", "operator": "in", "value": {"x": 1}}
    _, errors = parse(json.dumps(bad | {"window_hours": None}), Assertion)
    assert errors == ["value: matches none of the allowed types"]
