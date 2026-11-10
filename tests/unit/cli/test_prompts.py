from __future__ import annotations

from collections.abc import Iterator

import pytest

from synthgen.cli import prompts
from synthgen.domain.job import ApprovalDecision


def test_parse_share_edits() -> None:
    edits = prompts.parse_share_edits("a=0.1, b = 0.25")
    assert [(e.scenario, e.share) for e in edits] == [("a", 0.1), ("b", 0.25)]
    assert prompts.parse_share_edits("") == ()


@pytest.mark.parametrize("text", ["a", "a=x", "a=2"])
def test_parse_share_edits_rejects(text: str) -> None:
    with pytest.raises(ValueError):
        prompts.parse_share_edits(text)


def _answers(monkeypatch: pytest.MonkeyPatch, *values: str) -> None:
    feed: Iterator[str] = iter(values)
    monkeypatch.setattr("synthgen.cli.prompts.typer.prompt", lambda *_a, **_k: next(feed))


def test_decisions(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, "x", "a")
    assert prompts.ask_decision().decision is ApprovalDecision.APPROVED
    _answers(monkeypatch, "r", "more fraud please")
    rejected = prompts.ask_decision()
    assert (rejected.decision, rejected.feedback) == (
        ApprovalDecision.REJECTED,
        "more fraud please",
    )
    _answers(monkeypatch, "r", "  ")
    assert prompts.ask_decision().feedback is None
    _answers(monkeypatch, "e", "s=0.3", "t")
    edited = prompts.ask_decision()
    assert edited.decision is ApprovalDecision.EDITED
    assert edited.removed_scenarios == ("t",)
    _answers(monkeypatch, "e", "", "")
    assert prompts.ask_decision().decision is ApprovalDecision.APPROVED


def test_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _answers(monkeypatch, "  p99 ")
    assert prompts.ask_answer() == "p99"
    _answers(monkeypatch, " ")
    assert prompts.ask_answer() == "no answer"
