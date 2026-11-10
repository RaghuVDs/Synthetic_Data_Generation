"""Jobs and approvals in the state database (SPEC 10)."""

from __future__ import annotations

from datetime import datetime

from synthgen.domain.errors import NotFoundError, StorageError
from synthgen.domain.job import Approval, JobRecord, JobStatus, UserRequest
from synthgen.storage.db import Clock, StateDatabase, utc_now


class JobStore:
    """Creates jobs, tracks their status and records approval decisions."""

    def __init__(self, db: StateDatabase, clock: Clock = utc_now) -> None:
        """Use the given database; the clock is injectable for tests."""
        self._db = db
        self._clock = clock

    def create(self, job_id: str, request: UserRequest) -> JobRecord:
        """Insert a new job in status queued; fails if the ID already exists."""
        now = self._clock().isoformat()
        with self._db.connect() as conn:
            exists = conn.execute("SELECT 1 FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if exists:
                raise StorageError(f"job {job_id} already exists")
            conn.execute(
                "INSERT INTO jobs (job_id, source, request_json, status, error, created_at,"
                " updated_at) VALUES (?, ?, ?, ?, NULL, ?, ?)",
                (job_id, request.source, request.model_dump_json(), JobStatus.QUEUED, now, now),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> JobRecord:
        """Load one job; raises NotFoundError when it does not exist."""
        with self._db.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"job {job_id} not found")
        return JobRecord(
            job_id=row["job_id"],
            request=UserRequest.model_validate_json(row["request_json"]),
            status=JobStatus(row["status"]),
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> JobRecord:
        """Move a job to a new status, storing the error text for failed or stopped jobs."""
        with self._db.connect() as conn:
            cursor = conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
                (status, error, self._clock().isoformat(), job_id),
            )
        if cursor.rowcount == 0:
            raise NotFoundError(f"job {job_id} not found")
        return self.get(job_id)

    def record_approval(self, job_id: str, plan_version: int, approval: Approval) -> None:
        """Append the user's decision on a plan version."""
        self.get(job_id)
        with self._db.connect() as conn:
            conn.execute(
                "INSERT INTO approvals (job_id, plan_version, decision, approval_json, decided_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    job_id,
                    plan_version,
                    approval.decision,
                    approval.model_dump_json(),
                    approval.decided_at.isoformat(),
                ),
            )

    def approvals(self, job_id: str) -> list[tuple[int, Approval]]:
        """All decisions for a job in the order they were made, with their plan version."""
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT plan_version, approval_json FROM approvals WHERE job_id = ?"
                " ORDER BY approval_id",
                (job_id,),
            ).fetchall()
        return [(r["plan_version"], Approval.model_validate_json(r["approval_json"])) for r in rows]
