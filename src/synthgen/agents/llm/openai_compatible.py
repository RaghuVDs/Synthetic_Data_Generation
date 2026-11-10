"""OpenAI-compatible chat completions over HTTP (Groq in the POC; SPEC 4.1, 4.2, 4.5).

Retries rate limits, server errors and timeouts with backoff, honouring Retry-After. The API
key is sent only in the Authorization header and never logged.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

from synthgen.agents.llm.client import Message, Reply, ToolCall, ToolDef, ToolTurn, Usage
from synthgen.agents.llm.retry import RETRY_STATUSES, RateLimiter, backoff_seconds
from synthgen.agents.llm.schema import strict_schema
from synthgen.config import LlmSettings
from synthgen.domain.errors import ConfigError, LlmError

logger = logging.getLogger(__name__)
Json = dict[str, Any]  # Any: a decoded JSON object from the provider
JSON_MODE_INSTRUCTION = "Reply with one JSON object that matches this JSON schema:\n"


class OpenAiCompatibleClient:
    """Chat completions with structured output, tool calls and resilient retries."""

    def __init__(
        self,
        settings: LlmSettings,
        http: httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        """Fail early without a key; transport, sleep and jitter are injectable for tests."""
        if settings.api_key is None or not settings.api_key.get_secret_value():
            raise ConfigError("LLM_API_KEY is required when LLM_PROVIDER=openai_compatible")
        self._settings = settings
        self._http = http
        self._sleep = sleep
        self._rng = rng or random.Random()  # noqa: S311 - backoff jitter, not security
        self._limiter = RateLimiter(settings.requests_per_minute, time.monotonic, sleep)

    def complete(
        self, messages: list[Message], schema: type[BaseModel], temperature: float
    ) -> Reply:
        """Strict JSON-schema output, or JSON mode with the schema in the prompt."""
        body = self._body(messages, temperature)
        if self._settings.strict_output:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": strict_schema(schema),
                },
            }
        else:
            hint = Message(
                role="system", content=JSON_MODE_INSTRUCTION + json.dumps(strict_schema(schema))
            )
            body["messages"] = [_wire(m) for m in [*messages, hint]]
            body["response_format"] = {"type": "json_object"}
        data, latency = self._post(body)
        message = _first_message(data)
        return Reply(
            text=message.get("content") or "",
            model=str(data.get("model", "")),
            usage=_usage(data),
            latency_ms=latency,
        )

    def complete_with_tools(
        self, messages: list[Message], tools: list[ToolDef], temperature: float
    ) -> ToolTurn:
        """One tool-calling turn."""
        body = self._body(messages, temperature)
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        body["tool_choice"] = "auto"
        data, latency = self._post(body)
        message = _first_message(data)
        calls = [
            ToolCall(
                id=c["id"],
                name=c["function"]["name"],
                arguments=c["function"].get("arguments") or "{}",
            )
            for c in message.get("tool_calls") or []
        ]
        return ToolTurn(
            tool_calls=calls,
            content=message.get("content"),
            model=str(data.get("model", "")),
            usage=_usage(data),
            latency_ms=latency,
        )

    def _body(self, messages: list[Message], temperature: float) -> Json:
        return {
            "model": self._settings.model,
            "messages": [_wire(m) for m in messages],
            "temperature": temperature,
        }

    def _post(self, body: Json) -> tuple[Json, int]:
        """POST with retries; returns the decoded body and latency of the successful attempt."""
        key = self._settings.api_key.get_secret_value() if self._settings.api_key else ""
        url = self._settings.base_url.rstrip("/") + "/chat/completions"
        for attempt in range(self._settings.max_attempts):
            self._limiter.wait()
            started = time.monotonic()
            try:
                response = self._http.post(
                    url,
                    json=body,
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=self._settings.timeout_seconds,
                )
            except httpx.TransportError as exc:
                failure, retry_after = f"transport error: {type(exc).__name__}", None
            else:
                if response.status_code == httpx.codes.OK:
                    return response.json(), round((time.monotonic() - started) * 1000)
                if response.status_code not in RETRY_STATUSES:
                    raise LlmError(
                        f"LLM request rejected ({response.status_code}): {response.text[:500]}"
                    )
                failure, retry_after = (
                    f"HTTP {response.status_code}",
                    response.headers.get("retry-after"),
                )
            if attempt + 1 < self._settings.max_attempts:
                delay = backoff_seconds(attempt, retry_after, self._settings, self._rng)
                logger.warning(
                    "LLM call failed (%s); retry %d in %.1fs", failure, attempt + 1, delay
                )
                self._sleep(delay)
        raise LlmError(
            f"LLM request failed after {self._settings.max_attempts} attempts: {failure}"
        )


def _wire(message: Message) -> Json:
    """A message in the provider's wire format."""
    out: Json = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in message.tool_calls
        ]
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        out["name"] = message.name
    return out


def _first_message(data: Json) -> Json:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError(f"LLM response has no message: {str(data)[:300]}") from exc
    if not isinstance(message, dict):
        raise LlmError("LLM response message is not an object")
    return message


def _usage(data: Json) -> Usage:
    usage = data.get("usage") or {}
    return Usage(
        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens")
    )
