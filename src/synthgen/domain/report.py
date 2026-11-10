"""ValidationReport: one row per check, produced by code (SPEC 8.6)."""

from __future__ import annotations

from enum import StrEnum

from synthgen.domain.base import StrictModel


class CheckFamily(StrEnum):
    """The four validation families. Integrity and privacy are hard gates."""

    FIDELITY = "fidelity"
    INTEGRITY = "integrity"
    SCENARIO = "scenario"
    PRIVACY = "privacy"


class CheckStatus(StrEnum):
    """Outcome of one check. Skipped and info never fail a job (PLAN Q6, Q7)."""

    PASS = "pass"  # noqa: S105 - a status word, not a password
    FAIL = "fail"
    SKIPPED = "skipped"
    INFO = "info"


class CheckResult(StrictModel):
    """One report row: check, table, column, metric, value, threshold, status."""

    family: CheckFamily
    check: str
    table: str
    column: str | None = None
    metric: str
    value: float | None = None
    threshold: float | None = None
    status: CheckStatus
    scenario: str | None = None
    detail: str | None = None


class ValidationReport(StrictModel):
    """All check results for one dry run or one final validation."""

    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """True when no check failed."""
        return not self.failures()

    def failures(self, family: CheckFamily | None = None) -> list[CheckResult]:
        """Failed checks, optionally limited to one family."""
        return [
            r
            for r in self.results
            if r.status is CheckStatus.FAIL and (family is None or r.family is family)
        ]
