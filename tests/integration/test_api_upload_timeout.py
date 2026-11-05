"""M10 audit finding H1: stalled or trickling uploads time out and never hold a slot.

Raw ASGI requests run on the application's own event loop (through the
TestClient portal), so an upload can genuinely stop half-way: the client sends
part of a multipart body and then nothing, or keeps trickling bytes.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

import pytest

from backend.api import uploads
from tests.fixtures.documents import make_docx
from tests.fixtures.rate_limit import CLIENT_HEADER, harness_factory
from tests.integration.test_api_analyze_concurrency import (
    _FAKES,
    FakeWorkflow,
    analyze,
    assert_busy,
    fill_gate,
    wait_for,
)
from tests.unit.test_critic_checks import JOB, RESUME

BOUNDARY = b"m10-h1-boundary"
CANARY = b"CANARY-h1-upload-5e2a"
DOCX = make_docx(paragraphs=[line for line in RESUME.splitlines() if line.strip()])
MULTIPART_ENDPOINTS = ("/api/v1/parse", "/api/v1/match", "/api/v1/analyze")
TIMEOUT_MESSAGE = "The upload was not received in time. Please try again."
_RAW_FUTURES: list = []


@pytest.fixture
def harness(make_settings, fake_groq):
    factory = harness_factory(make_settings, fake_groq)
    try:
        yield next(factory)
    finally:
        # Never leave requests blocked when a test fails: cancel raw requests that
        # are still running and unblock fake analyses before the clients close.
        for future in _RAW_FUTURES:
            future.cancel()
        _RAW_FUTURES.clear()
        for fake in _FAKES:
            fake.released.set()
        _FAKES.clear()
        next(factory, None)


@pytest.fixture
def short_deadline(monkeypatch) -> float:
    monkeypatch.setattr(uploads, "UPLOAD_READ_TIMEOUT_SECONDS", 0.5)
    return 0.5


def multipart(*, file_bytes: bytes = DOCX, job: str | None = None) -> bytes:
    parts = [
        b"--" + BOUNDARY + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="cv.docx"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" + file_bytes + b"\r\n"
    ]
    if job is not None:
        parts.append(
            b"--" + BOUNDARY + b"\r\n"
            b'Content-Disposition: form-data; name="job_description"\r\n\r\n'
            + job.encode("utf-8")
            + b"\r\n"
        )
    return b"".join(parts) + b"--" + BOUNDARY + b"--\r\n"


def partial_body() -> bytes:
    """The start of a file part (with a canary in the data), never completed."""
    return (
        b"--" + BOUNDARY + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="cv.docx"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" + CANARY
    )


@dataclass
class RawResult:
    status: int = 0
    body: bytes = b""
    chunks_sent: int = 0
    finished_at: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)


async def raw_post(
    app, path: str, ip: str, chunks: list[bytes], *, then: str, delay: float, result: RawResult
) -> None:
    """POST ``chunks`` (with ``delay`` between them), then ``stall``, ``trickle`` or ``finish``."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"multipart/form-data; boundary=" + BOUNDARY),
            (CLIENT_HEADER.encode(), ip.encode()),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
        "state": {},
    }
    responded = False

    async def receive() -> dict:
        if responded:
            return {"type": "http.disconnect"}
        if result.chunks_sent < len(chunks):
            if result.chunks_sent and delay:
                await asyncio.sleep(delay)
            chunk = chunks[result.chunks_sent]
            result.chunks_sent += 1
            return {"type": "http.request", "body": chunk, "more_body": True}
        if then == "stall":
            await asyncio.sleep(3600)  # the client goes silent
        if then == "trickle":
            await asyncio.sleep(0.05)
            result.chunks_sent += 1
            return {"type": "http.request", "body": b"x", "more_body": True}
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        nonlocal responded
        if message["type"] == "http.response.start":
            result.status = message["status"]
        elif message["type"] == "http.response.body":
            result.body += message.get("body", b"")
            if not message.get("more_body", False):
                responded = True

    try:
        await app(scope, receive, send)
    finally:
        result.finished_at = time.monotonic()
        result.done.set()


def start_raw(h, path: str, ip: str, chunks: list[bytes], *, then: str, delay: float = 0.0):
    result = RawResult()
    future = h.client.portal.start_task_soon(
        lambda: raw_post(h.wrapper, path, ip, chunks, then=then, delay=delay, result=result)
    )
    _RAW_FUTURES.append(future)
    return result, future


def assert_upload_timeout(result: RawResult) -> None:
    import json

    assert result.status == 408
    body = json.loads(result.body)
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "request_id"}
    assert body["error"]["code"] == "upload_timeout"
    assert body["error"]["message"] == TIMEOUT_MESSAGE
    assert CANARY not in result.body


# --- the deadline --------------------------------------------------------------------------


def test_the_default_deadline_is_sixty_seconds() -> None:
    assert uploads.UPLOAD_READ_TIMEOUT_SECONDS == 60.0


@pytest.mark.parametrize("path", MULTIPART_ENDPOINTS)
def test_a_stalled_upload_times_out_with_a_safe_408(harness, short_deadline, path) -> None:
    h = harness()
    started = time.monotonic()
    result, future = start_raw(h, path, "192.0.2.10", [partial_body()], then="stall")
    future.result(timeout=20)
    assert_upload_timeout(result)
    assert short_deadline <= result.finished_at - started < short_deadline + 5


def test_a_trickling_upload_cannot_extend_the_deadline(harness, short_deadline) -> None:
    h = harness()
    started = time.monotonic()
    result, future = start_raw(h, "/api/v1/parse", "192.0.2.11", [partial_body()], then="trickle")
    future.result(timeout=20)
    assert_upload_timeout(result)
    assert result.chunks_sent > 3  # data kept arriving, yet the one deadline still applied
    assert result.finished_at - started < short_deadline + 5


def test_an_upload_that_finishes_in_time_is_accepted(harness, monkeypatch) -> None:
    monkeypatch.setattr(uploads, "UPLOAD_READ_TIMEOUT_SECONDS", 5.0)
    h = harness()
    body = multipart()
    chunks = [body[i : i + 4096] for i in range(0, len(body), 4096)]
    result, future = start_raw(h, "/api/v1/parse", "192.0.2.12", chunks, then="finish", delay=0.02)
    future.result(timeout=30)
    assert result.status == 200, result.body
    assert len(chunks) > 1  # really sent in pieces, nothing truncated


def test_a_timeout_is_logged_without_content(harness, short_deadline, captured_logs) -> None:
    import logging

    logging.getLogger("backend").setLevel(logging.INFO)
    h = harness()
    result, future = start_raw(h, "/api/v1/analyze", "192.0.2.13", [partial_body()], then="stall")
    future.result(timeout=20)
    assert_upload_timeout(result)
    (event,) = [e for e in captured_logs.events("app_error") if e.get("status") == 408]
    assert event["error_code"] == "upload_timeout"
    assert CANARY.decode() not in captured_logs.text and "192.0.2.13" not in captured_logs.text


def test_size_and_validation_limits_are_unchanged(harness) -> None:
    h = harness(max_upload_bytes=2048)
    too_big = h.request("POST", "/api/v1/parse", files={"file": ("cv.docx", b"x" * 4096)})
    assert too_big.status_code == 413
    assert too_big.json()["error"]["code"] == "file_too_large"
    unsupported = h.request("POST", "/api/v1/parse", files={"file": ("cv.exe", b"MZ" * 10)})
    assert unsupported.status_code in (415, 422)
    assert h.request("POST", "/api/v1/parse", content=b"{}").status_code == 415


# --- stalled uploads and the analysis slots ----------------------------------------------------


def test_stalled_uploads_never_occupy_an_analysis_slot(harness, monkeypatch) -> None:
    monkeypatch.setattr(uploads, "UPLOAD_READ_TIMEOUT_SECONDS", 3.0)
    h = harness(rate_limit_enabled=False)
    gate = h.app.state.analyze_gate
    stalled = [
        start_raw(h, "/api/v1/analyze", f"192.0.2.{20 + i}", [partial_body()], then="stall")
        for i in range(2)
    ]
    wait_for(lambda: all(r.chunks_sent == 1 for r, _ in stalled))
    assert gate.in_flight == 0
    # The audit scenario: with two uploads stuck, a third, complete analysis still runs.
    h.app.state.analysis_workflow = FakeWorkflow(block=False)
    for i in range(3):
        assert analyze(h, f"198.51.100.{i + 1}").status_code == 504  # the fake's outcome
        assert gate.in_flight == 0
    assert all(not r.done.is_set() for r, _ in stalled)  # still stuck meanwhile
    for result, future in stalled:
        future.result(timeout=20)
        assert_upload_timeout(result)
    assert gate.in_flight == 0


def test_the_cap_holds_with_stalled_uploads_and_running_analyses(harness, monkeypatch) -> None:
    monkeypatch.setattr(uploads, "UPLOAD_READ_TIMEOUT_SECONDS", 3.0)
    h = harness(rate_limit_enabled=False)
    gate = h.app.state.analyze_gate
    peak: list[int] = []
    real_try = gate.try_acquire

    def recording_try() -> bool:
        acquired = real_try()
        peak.append(gate.in_flight)
        return acquired

    monkeypatch.setattr(gate, "try_acquire", recording_try)
    stalled = [
        start_raw(h, "/api/v1/analyze", f"192.0.2.{30 + i}", [partial_body()], then="stall")
        for i in range(3)
    ]
    wait_for(lambda: all(r.chunks_sent == 1 for r, _ in stalled))
    fake = FakeWorkflow()
    running = fill_gate(h, fake)  # two real analyses hold both slots
    assert_busy(analyze(h, "198.51.100.50"))
    fake.released.set()
    running.join()
    for result, future in stalled:
        future.result(timeout=20)
        assert_upload_timeout(result)
    assert max(peak) <= 2 and gate.in_flight == 0
    assert len(peak) == 3  # only complete uploads ever asked for a slot


def test_an_analysis_with_a_job_description_still_works_in_pieces(harness, monkeypatch) -> None:
    monkeypatch.setattr(uploads, "UPLOAD_READ_TIMEOUT_SECONDS", 5.0)
    h = harness(rate_limit_enabled=False)
    h.app.state.analysis_workflow = FakeWorkflow(block=False)
    body = multipart(job=JOB)
    chunks = [body[i : i + 2048] for i in range(0, len(body), 2048)]
    result, future = start_raw(
        h, "/api/v1/analyze", "192.0.2.40", chunks, then="finish", delay=0.01
    )
    future.result(timeout=30)
    assert result.status == 504  # reached the (fake) workflow: parsed after a complete upload
    assert h.app.state.analyze_gate.in_flight == 0


def test_openapi_documents_408_on_the_multipart_endpoints_only(harness) -> None:
    spec = harness(api_docs_enabled=True).client.get("/openapi.json").json()
    ref = {"$ref": "#/components/schemas/ErrorEnvelope"}
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            documented = "408" in operation["responses"]
            assert documented == (path in MULTIPART_ENDPOINTS), (method, path)
            if documented:
                schema = operation["responses"]["408"]["content"]["application/json"]["schema"]
                assert schema == ref
                assert "upload_timeout" in operation["responses"]["408"]["description"]
