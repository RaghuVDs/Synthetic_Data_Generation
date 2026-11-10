"""Per-job cache of planner tool results (SPEC 6)."""

from __future__ import annotations


class ToolResultCache:
    """Tool results per job, so repeated calls in later plan rounds cost nothing."""

    def __init__(self) -> None:
        """Start empty."""
        self._results: dict[tuple[str, str, str], str] = {}

    def get(self, key: tuple[str, str, str]) -> str | None:
        """A cached result, or None."""
        return self._results.get(key)

    def put(self, key: tuple[str, str, str], result: str) -> None:
        """Remember a result."""
        self._results[key] = result
