"""The local web page's API end to end on the stub: start, approve, report, preview, errors."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from synthgen.app import Application, build_application
from synthgen.web.server import UiServer, serve

from .helpers import SETTINGS_FILE

EnvFactory = Callable[..., dict[str, str]]
TIMEOUT_SECONDS = 120


@pytest.fixture
def base_url(job_env: EnvFactory) -> Iterator[str]:
    env = job_env()

    def make_app() -> Application:
        return build_application(env, SETTINGS_FILE, None)

    server: UiServer = serve(make_app, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _call(url: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    request = Request(url, data=data, headers=headers)  # noqa: S310 - the local test server
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 - local test server
            raw = response.read()
            kind = response.headers.get("Content-Type", "")
            return response.status, json.loads(raw) if "json" in kind else raw.decode()
    except HTTPError as error:
        return error.code, json.loads(error.read())


def _wait(base: str, job_id: str, done: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        _, job = _call(f"{base}/api/jobs/{job_id}")
        if not job["busy"] and done(job):
            return dict(job)
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} did not settle")


def test_pick_tables_approve_and_see_the_report(base_url: str) -> None:
    status, page = _call(f"{base_url}/")
    assert status == 200 and "Synthetic Data Agent" in page
    _, found = _call(f"{base_url}/api/sources")
    finance = next(s for s in found["sources"] if s["name"] == "finance")
    assert {"customers", "accounts"} <= {t["name"] for t in finance["tables"]}

    status, started = _call(
        f"{base_url}/api/jobs",
        {
            "source": "finance",
            "tables": ["customers", "accounts"],
            "rows": {"customers": 300},
            "request": "customers with unusually high balances",
        },
    )
    assert status == 200
    job_id = started["job_id"]
    paused = _wait(base_url, job_id, lambda j: j["pending"] is not None or j["terminal"])
    assert paused["pending"]["kind"] == "approval"
    assert [s["name"] for s in paused["pending"]["scenarios"]] == ["high_balance"]

    status, _ = _call(f"{base_url}/api/jobs/{job_id}/decision", {"decision": "approved"})
    assert status == 200
    done = _wait(base_url, job_id, lambda j: j["terminal"])
    assert done["status"] == "completed", done["error"]
    report = done["report"]
    assert {t["name"]: t["rows"] for t in report["tables"]}["customers"] == 300
    assert report["checks_run"] > 0 and report["failures"] == []

    _, preview = _call(f"{base_url}/api/jobs/{job_id}/preview?table=accounts")
    assert "balance" in preview["columns"] and 0 < len(preview["rows"]) <= 20


def test_bad_requests_get_readable_errors(base_url: str) -> None:
    status, reply = _call(
        f"{base_url}/api/jobs", {"source": "finance", "tables": [], "request": "x"}
    )
    assert status == 400 and "at least 1 item" in reply["error"]
    status, reply = _call(
        f"{base_url}/api/jobs", {"source": "nowhere", "tables": ["a"], "request": "x"}
    )
    assert status == 404 and "nowhere" in reply["error"]
    status, _ = _call(f"{base_url}/api/jobs/abc123")
    assert status == 404
    status, _ = _call(f"{base_url}/api/elsewhere")
    assert status == 404
