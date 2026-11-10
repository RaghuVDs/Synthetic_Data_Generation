"""Terminal prompts for a paused job: answer questions, or approve, edit or reject a plan."""

from __future__ import annotations

from datetime import UTC, datetime

import typer
from pydantic import ValidationError

from synthgen.domain.job import Approval, ApprovalDecision, ShareEdit

CHOICES = {
    "a": ApprovalDecision.APPROVED,
    "e": ApprovalDecision.EDITED,
    "r": ApprovalDecision.REJECTED,
}


def ask_answer() -> str:
    """One answer covering every blocking question."""
    return str(typer.prompt("Your answer")).strip() or "no answer"


def parse_share_edits(text: str) -> tuple[ShareEdit, ...]:
    """`name=0.1, other=0.05` -> share edits; raises ValueError on a malformed pair."""
    edits = []
    for pair in (p.strip() for p in text.split(",") if p.strip()):
        name, sep, share = pair.partition("=")
        if not sep:
            raise ValueError(f"{pair!r} is not name=share")
        try:
            edits.append(ShareEdit(scenario=name.strip(), share=float(share)))
        except ValidationError as exc:
            raise ValueError(f"{pair!r}: share must be between 0 and 1") from exc
    return tuple(edits)


def _ask_edits() -> Approval:
    while True:
        shares = str(typer.prompt("New shares as name=share, comma-separated", default=""))
        removed = str(typer.prompt("Scenarios to remove, comma-separated", default=""))
        try:
            edits = parse_share_edits(shares)
        except ValueError as exc:
            typer.echo(f"Not understood: {exc}")
            continue
        drop = tuple(n.strip() for n in removed.split(",") if n.strip())
        if not edits and not drop:
            return Approval(decision=ApprovalDecision.APPROVED, decided_at=datetime.now(UTC))
        return Approval(
            decision=ApprovalDecision.EDITED,
            share_edits=edits,
            removed_scenarios=drop,
            decided_at=datetime.now(UTC),
        )


def ask_decision() -> Approval:
    """Approve, edit (shares and removals) or reject with feedback; blank feedback cancels."""
    while True:
        choice = str(typer.prompt("Approve, edit or reject? [a/e/r]", default="a")).strip().lower()
        if choice in CHOICES:
            break
        typer.echo("Please answer a, e or r.")
    if CHOICES[choice] is ApprovalDecision.EDITED:
        return _ask_edits()
    if CHOICES[choice] is ApprovalDecision.REJECTED:
        feedback = str(typer.prompt("Feedback for the planner (blank cancels the job)", default=""))
        return Approval(
            decision=ApprovalDecision.REJECTED,
            feedback=feedback.strip() or None,
            decided_at=datetime.now(UTC),
        )
    return Approval(decision=ApprovalDecision.APPROVED, decided_at=datetime.now(UTC))
