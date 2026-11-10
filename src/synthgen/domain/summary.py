"""Summary: the reporter's short text. Code inserts every number (SPEC 3.1, 5)."""

from __future__ import annotations

from pydantic import field_validator

from synthgen.domain.base import StrictModel

MAX_SUMMARY_WORDS = 150


class Summary(StrictModel):
    """A summary of at most 150 words, plus caveats the tester should know."""

    text: str
    caveats: list[str]

    @field_validator("text")
    @classmethod
    def _word_limit(cls, text: str) -> str:
        words = len(text.split())
        if not words:
            raise ValueError("summary text is empty")
        if words > MAX_SUMMARY_WORDS:
            raise ValueError(f"summary has {words} words, limit is {MAX_SUMMARY_WORDS}")
        return text
