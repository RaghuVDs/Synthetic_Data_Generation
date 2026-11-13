"""Runs job steps in background threads so the page stays responsive while a job works.

Each step builds its own application (its own database and checkpoint connections) and closes
it, so steps never share a connection across threads. One step per job runs at a time.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from synthgen.app import Application
from synthgen.domain.errors import SynthgenError

AppFactory = Callable[[], Application]
Step = Callable[[Application], object]
_LOG = logging.getLogger(__name__)


class JobWorkers:
    """Background steps per job, with the last unexpected failure of each."""

    def __init__(self, make_app: AppFactory) -> None:
        """Build an application per step with `make_app`."""
        self._make_app = make_app
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._failures: dict[str, str] = {}

    def busy(self, job_id: str) -> bool:
        """True while a step of this job is running."""
        with self._lock:
            thread = self._threads.get(job_id)
            return thread is not None and thread.is_alive()

    def failure(self, job_id: str) -> str | None:
        """Why the last step stopped unexpectedly, if it did."""
        with self._lock:
            return self._failures.get(job_id)

    def start(self, job_id: str, step: Step) -> None:
        """Run `step` for the job in the background; refuse while another step runs."""
        with self._lock:
            running = self._threads.get(job_id)
            if running is not None and running.is_alive():
                raise SynthgenError(f"job {job_id} is still working; wait for it to pause")
            self._failures.pop(job_id, None)
            thread = threading.Thread(target=self._run, args=(job_id, step), daemon=True)
            self._threads[job_id] = thread
        thread.start()

    def _run(self, job_id: str, step: Step) -> None:
        try:
            with self._make_app() as app:
                step(app)
        except Exception as exc:  # shown on the page; the job record keeps the job's state
            _LOG.exception("job %s step failed", job_id)
            with self._lock:
                self._failures[job_id] = f"{type(exc).__name__}: {exc}"
