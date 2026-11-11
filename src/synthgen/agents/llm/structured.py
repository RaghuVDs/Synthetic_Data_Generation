"""The three gates on every LLM response, re-prompts and the audit log (SPEC 4.3, 4.5).

Gate 1 is the provider's schema (strict mode) or JSON mode. Gate 2 parses with the node's
Pydantic model. Gate 3 is the node's meaning check against metadata and plan. A failure is
sent back with the previous output and a numbered error list; after the allowed attempts the
call raises LlmOutputError and the job ends as needs_review.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from synthgen.agents.llm.client import LlmClient, Message, Reply, ToolDef, ToolTurn
from synthgen.domain.errors import LlmError, LlmOutputError
from synthgen.domain.llm_call import CallStatus, LlmCallRecord

ModelT = TypeVar("ModelT", bound=BaseModel)
MeaningCheck = Callable[[ModelT], list[str]]
RETRY_INSTRUCTION = (
    "Your previous answer was rejected for these reasons:\n{errors}\nReturn corrected JSON."
)


@dataclass(frozen=True)
class CallContext:
    """Who is asking: job, node and prompt version, for the audit log."""

    job_id: str
    node: str
    prompt_version: str


class CallLog(Protocol):
    """Where audited calls go."""

    def record(self, entry: LlmCallRecord) -> None:
        """Store one call."""
        ...


def no_meaning_errors(_value: BaseModel) -> list[str]:
    """Gate 3 for roles without semantic checks."""
    return []


def numbered(errors: list[str]) -> str:
    """`1. first` lines for a re-prompt."""
    return "\n".join(f"{i}. {e}" for i, e in enumerate(errors, start=1))


UNION_BRANCH = re.compile(r"^(float|int|str|bool|list\[.*\]|dict\[.*\]|[A-Z]\w*)$")
"""A location part that names one branch of a union type, e.g. `float` or `list[str]`."""


def parse(text: str, schema: type[ModelT]) -> tuple[ModelT | None, list[str]]:
    """Gate 2: the parsed model, or the reasons it does not parse (one line per field)."""
    try:
        return schema.model_validate_json(text), []
    except ValidationError as exc:
        return None, _readable(exc)


def _readable(exc: ValidationError) -> list[str]:
    """Collapse the per-branch errors of a union field into one line."""
    lines: dict[str, str] = {}
    for err in exc.errors():
        parts = [str(p) for p in err["loc"]]
        branch = next((i for i, p in enumerate(parts) if UNION_BRANCH.match(p)), None)
        if branch is None:
            lines.setdefault(".".join(parts) or "root", err["msg"])
        else:
            lines[".".join(parts[:branch])] = "matches none of the allowed types"
    return [f"{where}: {message}" for where, message in lines.items()]


class StructuredLlm:
    """Asks a client for validated, meaningful JSON and logs every call."""

    def __init__(self, client: LlmClient, log: CallLog, attempts: int) -> None:
        """`attempts` is the cap on tries per call (SPEC 3.4 `llm_reprompts`)."""
        self._client = client
        self._log = log
        self._attempts = max(1, attempts)

    def ask(
        self,
        ctx: CallContext,
        messages: list[Message],
        schema: type[ModelT],
        temperature: float,
        meaning: Callable[[ModelT], list[str]] = no_meaning_errors,
    ) -> ModelT:
        """A response that passed all three gates."""
        conversation = list(messages)
        errors: list[str] = []
        for _ in range(self._attempts):  # each retry carries only the latest answer and errors
            reply = self._send(
                ctx, conversation, lambda c: self._client.complete(c, schema, temperature)
            )
            value, errors = parse(reply.text, schema)
            if value is not None:
                errors = meaning(value)
            self._record(ctx, conversation, reply, CallStatus.INVALID if errors else CallStatus.OK)
            if value is not None and not errors:
                return value
            conversation = [
                *messages,
                Message(role="assistant", content=reply.text),
                Message(role="user", content=RETRY_INSTRUCTION.format(errors=numbered(errors))),
            ]
        raise LlmOutputError(
            f"{schema.__name__} failed after {self._attempts} attempts: {'; '.join(errors)}"
        )

    def tool_turn(
        self, ctx: CallContext, messages: list[Message], tools: list[ToolDef], temperature: float
    ) -> ToolTurn:
        """One audited tool-calling turn."""
        turn = self._send(
            ctx, messages, lambda c: self._client.complete_with_tools(c, tools, temperature)
        )
        text = json.dumps(
            {"tool_calls": [c.model_dump() for c in turn.tool_calls], "content": turn.content}
        )
        reply = Reply(text=text, model=turn.model, usage=turn.usage, latency_ms=turn.latency_ms)
        self._record(ctx, messages, reply, CallStatus.OK)
        return turn

    def _send(
        self, ctx: CallContext, messages: list[Message], call: Callable[[list[Message]], ReplyT]
    ) -> ReplyT:
        try:
            return call(messages)
        except LlmError:
            self._log.record(_entry(ctx, messages, None, CallStatus.ERROR))
            raise

    def _record(
        self, ctx: CallContext, messages: list[Message], reply: Reply, status: CallStatus
    ) -> None:
        self._log.record(_entry(ctx, messages, reply, status))


ReplyT = TypeVar("ReplyT", Reply, ToolTurn)


def _entry(
    ctx: CallContext, messages: list[Message], reply: Reply | None, status: CallStatus
) -> LlmCallRecord:
    return LlmCallRecord(
        job_id=ctx.job_id,
        node=ctx.node,
        model=reply.model if reply else "",
        prompt_version=ctx.prompt_version,
        prompt=json.dumps([m.model_dump(mode="json", exclude_defaults=True) for m in messages]),
        response=reply.text if reply else None,
        prompt_tokens=reply.usage.prompt_tokens if reply else None,
        completion_tokens=reply.usage.completion_tokens if reply else None,
        latency_ms=reply.latency_ms if reply else 0,
        status=status,
    )
