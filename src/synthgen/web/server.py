"""A local web page for choosing tables and generating data (user decision after M7).

Standard-library HTTP server, bound to localhost by default. Routes:

- `GET /` the page; `GET /api/sources` every source with its tables;
- `POST /api/jobs` start a job; `GET /api/jobs/<id>` its status, questions, plan or report;
- `POST /api/jobs/<id>/answer` and `/decision` continue a paused job;
- `GET /api/jobs/<id>/preview?table=t` the first rows of a generated table.

Every input goes through the same Pydantic models and use cases as the command line.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import ValidationError

from synthgen.domain.errors import NotFoundError, SynthgenError
from synthgen.domain.job import Approval, ApprovalDecision, RowTarget, UserRequest
from synthgen.web.jobs import AppFactory, JobWorkers
from synthgen.web.views import Json, job_json, preview_json, report_json, source_json

PAGE = Path(__file__).with_name("index.html")
_JOB = re.compile(r"^/api/jobs/([0-9a-f]+)(?:/(answer|decision|preview))?$")
_LOG = logging.getLogger(__name__)


class UiServer(ThreadingHTTPServer):
    """The HTTP server with the application factory and the background job workers."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], make_app: AppFactory) -> None:
        """Listen on `address`; every request builds its own short-lived application."""
        super().__init__(address, Handler)
        self.make_app = make_app
        self.workers = JobWorkers(make_app)
        with make_app() as app:
            self.settings = app.settings


class Handler(BaseHTTPRequestHandler):
    """One HTTP request: routes it, turns typed errors into JSON error replies."""

    server: UiServer

    def do_GET(self) -> None:
        """Page, sources, job status or preview."""
        url = urlparse(self.path)
        if url.path == "/":
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        self._json_call(lambda: self._get(url.path, parse_qs(url.query)))

    def do_POST(self) -> None:
        """Start, answer or decide a job."""
        self._json_call(lambda: self._post(urlparse(self.path).path, self._body()))

    def log_message(self, format: str, *args: Any) -> None:  # Any: printf-style arguments
        """Request lines go to the debug log instead of standard error."""
        _LOG.debug(format, *args)

    def _get(self, path: str, query: dict[str, list[str]]) -> Json:
        if path == "/api/sources":
            return {"sources": self._sources()}
        job_id, action = _job_route(path)
        if action is None:
            return self._status(job_id)
        if action == "preview":
            table = (query.get("table") or [""])[0]
            rows = self.server.settings.ui.preview_rows
            with self.server.make_app() as app:
                return preview_json(app.get_preview(job_id, table, rows))
        raise NotFoundError(f"no page at {path}")

    def _post(self, path: str, body: Json) -> Json:
        if path == "/api/jobs":
            return self._start(body)
        job_id, action = _job_route(path)
        if action == "answer":
            answer = str(body.get("answer") or "").strip()
            if not answer:
                raise SynthgenError("an answer is needed")
            self.server.workers.start(job_id, lambda app: app.answer_clarification(job_id, answer))
        elif action == "decision":
            approval = _approval(body)
            self.server.workers.start(job_id, lambda app: app.decide_plan(job_id, approval))
        else:
            raise NotFoundError(f"no action at {path}")
        return {"job_id": job_id}

    def _sources(self) -> list[Json]:
        found: list[Json] = []
        with self.server.make_app() as app:
            for yaml_file in sorted(app.settings.paths.metadata_dir.glob("*.yaml")):
                try:
                    found.append(source_json(app.list_tables(yaml_file.stem)))
                except SynthgenError as exc:
                    found.append({"name": yaml_file.stem, "error": str(exc), "tables": []})
        return found

    def _start(self, body: Json) -> Json:
        rows = body.get("rows") or {}
        request = UserRequest(
            source=body.get("source", ""),
            tables=tuple(body.get("tables") or ()),
            text=str(body.get("request") or "").strip(),
            row_targets=tuple(RowTarget(table=t, rows=n) for t, n in rows.items() if n),
            seed=body.get("seed") or 0,
            include_invalid=bool(body.get("include_invalid")),
            auto_approve=bool(body.get("auto_approve")),
        )
        with self.server.make_app() as app:
            job_id = app.create_job(request).job_id
        self.server.workers.start(job_id, lambda app: app.run_job(job_id))
        return {"job_id": job_id}

    def _status(self, job_id: str) -> Json:
        workers = self.server.workers
        busy = workers.busy(job_id)
        with self.server.make_app() as app:
            out = job_json(app.view(job_id), busy=busy, failure=workers.failure(job_id))
            if out["terminal"] and not busy:
                out["report"] = report_json(app.get_report(job_id))
        return out

    def _body(self) -> Json:
        length = int(self.headers.get("Content-Length") or 0)
        if length > self.server.settings.ui.max_body_bytes:
            raise SynthgenError("request too large")
        loaded = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(loaded, dict):
            raise SynthgenError("expected a JSON object")
        return loaded

    def _json_call(self, action: Any) -> None:  # Any: a zero-argument callable returning Json
        try:
            self._send_json(HTTPStatus.OK, action())
        except NotFoundError as exc:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except ValidationError as exc:
            errors = "; ".join(e["msg"] for e in exc.errors())
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": errors})
        except (SynthgenError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def _send_json(self, status: HTTPStatus, payload: Json) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _send(self, status: HTTPStatus, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _job_route(path: str) -> tuple[str, str | None]:
    match = _JOB.match(path)
    if match is None:
        raise NotFoundError(f"no page at {path}")
    return match.group(1), match.group(2)


def _approval(body: Json) -> Approval:
    decision = ApprovalDecision(body.get("decision", ""))
    removed = tuple(body.get("removed_scenarios") or ())
    if decision is ApprovalDecision.APPROVED and removed:
        decision = ApprovalDecision.EDITED
    return Approval(
        decision=decision,
        feedback=body.get("feedback") or None,
        removed_scenarios=removed,
        decided_at=datetime.now(UTC),
    )


def serve(make_app: AppFactory, host: str, port: int) -> UiServer:
    """A server ready to `serve_forever()`; the caller prints its address and stops it."""
    return UiServer((host, port), make_app)
