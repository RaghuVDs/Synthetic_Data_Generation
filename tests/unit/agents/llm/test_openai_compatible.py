from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from synthgen.agents.llm.client import Message, ToolCall, ToolDef
from synthgen.agents.llm.openai_compatible import OpenAiCompatibleClient
from synthgen.config import LlmSettings, load_settings
from synthgen.domain.errors import ConfigError, LlmError
from synthgen.domain.summary import Summary

ROOT = Path(__file__).resolve().parents[4]
KEY = "test-key-not-real"
MESSAGES = [Message(role="system", content="s"), Message(role="user", content="u")]
OK_BODY = {
    "model": "openai/gpt-oss-120b",
    "choices": [{"message": {"role": "assistant", "content": '{"text": "hi", "caveats": []}'}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 5},
}


def settings(**changes: str) -> LlmSettings:
    env = {"LLM_PROVIDER": "openai_compatible", "LLM_API_KEY": KEY, **changes}
    return load_settings(ROOT / "config" / "settings.yaml", env).llm


Handler = Callable[[httpx.Request], httpx.Response]


def client(handler: Handler, sleeps: list[float], **changes: str) -> OpenAiCompatibleClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAiCompatibleClient(
        settings(**changes), http, sleep=sleeps.append, rng=random.Random(0)
    )


def recording(responses: list[httpx.Response], seen: list[httpx.Request]) -> Handler:
    queue = iter(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return next(queue)

    return handle


def body(request: httpx.Request) -> dict[str, Any]:
    return dict(json.loads(request.content))


def test_strict_request_and_reply() -> None:
    seen: list[httpx.Request] = []
    reply = client(recording([httpx.Response(200, json=OK_BODY)], seen), []).complete(
        MESSAGES, Summary, 0.0
    )
    assert json.loads(reply.text)["text"] == "hi"
    assert (reply.usage.prompt_tokens, reply.usage.completion_tokens) == (12, 5)
    sent = body(seen[0])
    assert seen[0].url.path.endswith("/chat/completions")
    assert seen[0].headers["authorization"] == f"Bearer {KEY}"
    assert sent["model"] == "openai/gpt-oss-120b"
    assert sent["temperature"] == 0.0
    fmt = sent["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False


def test_json_object_mode_puts_schema_in_prompt() -> None:
    seen: list[httpx.Request] = []
    handler = recording([httpx.Response(200, json=OK_BODY)], seen)
    client(handler, [], SYNTHGEN_LLM__OUTPUT_MODE="json_object").complete(MESSAGES, Summary, 0.0)
    sent = body(seen[0])
    assert sent["response_format"] == {"type": "json_object"}
    assert "JSON schema" in sent["messages"][-1]["content"]


def test_prompt_mode_sends_no_response_format_and_honours_max_tokens() -> None:
    seen: list[httpx.Request] = []
    handler = recording([httpx.Response(200, json=OK_BODY)], seen)
    llm = client(handler, [], SYNTHGEN_LLM__OUTPUT_MODE="prompt", SYNTHGEN_LLM__MAX_TOKENS="4096")
    llm.complete(MESSAGES, Summary, 0.0)
    sent = body(seen[0])
    assert "response_format" not in sent
    assert sent["max_tokens"] == 4096
    assert "JSON schema" in sent["messages"][-1]["content"]


def test_tool_turn_round_trip() -> None:
    seen: list[httpx.Request] = []
    tool_reply = {
        "model": "m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "t", "arguments": '{"a": 1}'},
                        }
                    ],
                }
            }
        ],
    }
    history = [
        *MESSAGES,
        Message(
            role="assistant",
            content=None,
            tool_calls=(ToolCall(id="c0", name="t", arguments="{}"),),
        ),
        Message(role="tool", content="result", tool_call_id="c0", name="t"),
    ]
    tool = ToolDef(name="t", description="d", parameters={"type": "object", "properties": {}})
    turn = client(recording([httpx.Response(200, json=tool_reply)], seen), []).complete_with_tools(
        history, [tool], 0.3
    )
    assert turn.tool_calls == [ToolCall(id="c1", name="t", arguments='{"a": 1}')]
    sent = body(seen[0])
    assert sent["tools"][0]["function"]["name"] == "t"
    assert sent["messages"][2]["tool_calls"][0]["id"] == "c0"
    assert sent["messages"][3] == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "c0",
        "name": "t",
    }


def test_rate_limit_honours_retry_after() -> None:
    sleeps: list[float] = []
    responses = [
        httpx.Response(429, headers={"retry-after": "2"}),
        httpx.Response(200, json=OK_BODY),
    ]
    client(recording(responses, []), sleeps).complete(MESSAGES, Summary, 0.0)
    assert 2.0 in sleeps


def test_server_errors_and_timeouts_are_retried() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadTimeout("slow", request=request)
        if calls["n"] == 2:
            return httpx.Response(503)
        return httpx.Response(200, json=OK_BODY)

    client(flaky, sleeps).complete(MESSAGES, Summary, 0.0)
    assert calls["n"] == 3
    assert len([s for s in sleeps if s > 0]) >= 2


def test_gives_up_after_max_attempts() -> None:
    sleeps: list[float] = []
    responses = [httpx.Response(500) for _ in range(5)]
    with pytest.raises(LlmError, match="after 5 attempts"):
        client(recording(responses, []), sleeps).complete(MESSAGES, Summary, 0.0)


def test_client_errors_are_not_retried() -> None:
    seen: list[httpx.Request] = []
    with pytest.raises(LlmError, match="400"):
        client(recording([httpx.Response(400, text="bad schema")], seen), []).complete(
            MESSAGES, Summary, 0.0
        )
    assert len(seen) == 1


def test_missing_key_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        OpenAiCompatibleClient(settings(LLM_API_KEY=""), httpx.Client())


def test_malformed_reply_is_an_llm_error() -> None:
    with pytest.raises(LlmError):
        client(recording([httpx.Response(200, json={"choices": []})], []), []).complete(
            MESSAGES, Summary, 0.0
        )


def test_schema_rejection_returns_the_failed_generation() -> None:
    rejected = httpx.Response(
        400,
        json={
            "error": {
                "message": "Generated JSON does not match the expected schema.",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
                "failed_generation": '{"text": 5}',
            }
        },
    )
    seen: list[httpx.Request] = []
    reply = client(recording([rejected], seen), []).complete(MESSAGES, Summary, 0.0)
    assert reply.text == '{"text": 5}'
    assert len(seen) == 1


def test_other_bad_requests_stay_errors() -> None:
    other = httpx.Response(400, json={"error": {"code": "invalid_api_key", "message": "no"}})
    with pytest.raises(LlmError, match="400"):
        client(recording([other], []), []).complete(MESSAGES, Summary, 0.0)
