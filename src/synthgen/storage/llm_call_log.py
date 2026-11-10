"""The `llm_calls` audit table (SPEC 4.5)."""

from __future__ import annotations

from synthgen.domain.llm_call import CallStatus, LlmCallRecord
from synthgen.storage.db import Clock, StateDatabase, utc_now


class LlmCallLog:
    """Appends one row per LLM call and reads a job's calls back."""

    def __init__(self, db: StateDatabase, clock: Clock = utc_now) -> None:
        """Use the given database; the clock is injectable for tests."""
        self._db = db
        self._clock = clock

    def record(self, entry: LlmCallRecord) -> None:
        """Store one call."""
        with self._db.connect() as conn:
            conn.execute(
                "INSERT INTO llm_calls (job_id, node, model, prompt_version, prompt, response,"
                " prompt_tokens, completion_tokens, latency_ms, status, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.job_id,
                    entry.node,
                    entry.model,
                    entry.prompt_version,
                    entry.prompt,
                    entry.response,
                    entry.prompt_tokens,
                    entry.completion_tokens,
                    entry.latency_ms,
                    entry.status,
                    self._clock().isoformat(),
                ),
            )

    def for_job(self, job_id: str) -> list[LlmCallRecord]:
        """Every call made for a job, oldest first."""
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_calls WHERE job_id = ? ORDER BY call_id", (job_id,)
            ).fetchall()
        return [
            LlmCallRecord(
                job_id=r["job_id"],
                node=r["node"],
                model=r["model"],
                prompt_version=r["prompt_version"],
                prompt=r["prompt"],
                response=r["response"],
                prompt_tokens=r["prompt_tokens"],
                completion_tokens=r["completion_tokens"],
                latency_ms=r["latency_ms"],
                status=CallStatus(r["status"]),
            )
            for r in rows
        ]
