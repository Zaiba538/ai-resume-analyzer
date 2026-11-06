"""M10 Phase 2: the /analyze concurrency gate (at most 2 analyses in flight)."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from backend.api.rate_limits import ANALYSIS_BUSY_MESSAGE
from backend.core.errors import AppError
from backend.graph.analysis import AnalysisTimeoutError
from tests.conftest import FAKE_API_KEY
from tests.fixtures.documents import make_docx
from tests.fixtures.rate_limit import harness_factory
from tests.integration.test_analysis_workflow import NO_ISSUES, scripted
from tests.integration.test_api_rate_limit import assert_rate_limited
from tests.unit.test_critic_checks import RESUME

DOCX = make_docx(paragraphs=[line for line in RESUME.splitlines() if line.strip()])


class FakeRegistry:
    """The fake workflows created by one test, released together when it ends."""

    def __init__(self) -> None:
        self._fakes: list[FakeWorkflow] = []

    def add(self, fake: FakeWorkflow) -> None:
        self._fakes.append(fake)

    def release_all(self) -> None:
        for fake in self._fakes:
            fake.released.set()


# Bound only while a test runs (the ``fake_registry`` fixture), so fakes can never
# outlive their test or be shared between tests.
_active_fakes: FakeRegistry | None = None


def _current_fakes() -> FakeRegistry:
    if _active_fakes is None:
        raise RuntimeError("FakeWorkflow needs the fake_registry fixture")
    return _active_fakes


@pytest.fixture(autouse=True)
def fake_registry():
    global _active_fakes
    registry = FakeRegistry()
    _active_fakes = registry
    try:
        yield registry
    finally:
        registry.release_all()
        _active_fakes = None


@pytest.fixture
def harness(make_settings, fake_groq, fake_registry):
    factory = harness_factory(make_settings, fake_groq)
    try:
        yield next(factory)
    finally:
        # Unblock every fake before the clients close, so that a failed assertion
        # cannot leave requests (and the test run) waiting forever.
        fake_registry.release_all()
        next(factory, None)


class FakeWorkflow:
    """Stands in for the analysis workflow: optionally blocks, then returns ``result``
    or raises ``outcome`` (default: the analysis deadline error, 504)."""

    def __init__(
        self, outcome: BaseException | None = None, *, block: bool = True, result: object = None
    ) -> None:
        self.outcome = outcome or AnalysisTimeoutError()
        self.result = result
        self.released = threading.Event()
        if not block:
            self.released.set()
        self._lock = threading.Lock()
        self.entered = 0
        _current_fakes().add(self)

    async def run(self, *, resume_text: str, job_description: str | None):
        with self._lock:
            self.entered += 1
        while not self.released.is_set():
            await asyncio.sleep(0.005)
        if self.result is not None:
            return self.result
        raise self.outcome


def analyze(h, ip: str = "192.0.2.1", data: bytes = DOCX, *, timeout: float = 20.0):
    """POST /analyze; fails (instead of hanging) if the request is queued or blocked."""
    result: list = []
    thread = threading.Thread(
        target=lambda: result.append(
            h.request("POST", "/api/v1/analyze", ip=ip, files={"file": ("cv.docx", data)})
        ),
        daemon=True,
    )
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        _current_fakes().release_all()
        thread.join(timeout)
        pytest.fail("the request did not complete: it was queued or blocked, not rejected")
    return result[0]


def assert_busy(response) -> None:
    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "code": "analysis_capacity_exceeded",
            "message": ANALYSIS_BUSY_MESSAGE,
            "request_id": response.headers["X-Request-ID"],
        }
    }
    assert response.headers["Retry-After"] == "10"
    assert response.headers["Cache-Control"] == "no-store"
    assert "Content-Security-Policy" in response.headers


def wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


class InFlight:
    """Start analyses in background threads and collect their responses."""

    def __init__(self, h) -> None:
        self.h = h
        self.threads: list[threading.Thread] = []
        self.responses: list = []

    def start(self, ip: str) -> None:
        thread = threading.Thread(
            target=lambda: self.responses.append(analyze(self.h, ip, timeout=60)), daemon=True
        )
        thread.start()
        self.threads.append(thread)

    def join(self) -> list:
        for thread in self.threads:
            thread.join(timeout=30)
        assert not any(thread.is_alive() for thread in self.threads)
        return self.responses


def fill_gate(h, fake: FakeWorkflow, ips=("192.0.2.101", "192.0.2.102")) -> InFlight:
    h.app.state.analysis_workflow = fake
    running = InFlight(h)
    for ip in ips:
        running.start(ip)
    wait_for(lambda: fake.entered == len(ips))
    assert h.app.state.analyze_gate.in_flight == len(ips)
    return running


# --- the cap ---------------------------------------------------------------------------


def test_two_analyses_run_and_the_third_is_rejected_at_once(harness) -> None:
    h = harness()
    fake = FakeWorkflow()
    running = fill_gate(h, fake)
    started = time.monotonic()
    assert_busy(analyze(h, "192.0.2.103"))
    assert time.monotonic() - started < 5  # rejected, not queued
    assert fake.entered == 2
    fake.released.set()
    assert [r.status_code for r in running.join()] == [504, 504]
    assert h.app.state.analyze_gate.in_flight == 0
    assert analyze(h, "192.0.2.103").status_code == 504  # a slot is free again


def test_a_burst_admits_exactly_two(harness) -> None:
    h = harness(rate_limit_enabled=False)
    fake = FakeWorkflow()
    h.app.state.analysis_workflow = fake
    barrier = threading.Barrier(10)
    statuses: list[int] = []

    def run(i: int) -> None:
        barrier.wait()
        statuses.append(analyze(h, f"192.0.2.{i + 1}").status_code)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(10)]
    for thread in threads:
        thread.start()
    try:
        wait_for(lambda: statuses.count(429) == 8)
        # The two admitted requests hold their slots from before the body is read,
        # so they may still be parsing the upload when the eight rejections arrive.
        wait_for(lambda: fake.entered == 2)
        assert h.app.state.analyze_gate.in_flight == 2
    finally:
        fake.released.set()  # never leave requests blocked, even when an assertion fails
        for thread in threads:
            thread.join(timeout=30)
    assert sorted(statuses) == [429] * 8 + [504] * 2
    assert h.app.state.analyze_gate.in_flight == 0


def test_gate_stays_active_when_rate_limiting_is_disabled(harness) -> None:
    h = harness(rate_limit_enabled=False)
    fake = FakeWorkflow()
    running = fill_gate(h, fake)
    assert_busy(analyze(h, "192.0.2.1"))
    fake.released.set()
    running.join()
    assert h.limiter.tracked_keys() == ()  # quota enforcement really is off


def test_max_concurrent_setting_is_honoured(harness) -> None:
    h = harness(analyze_max_concurrent=1)
    fake = FakeWorkflow()
    running = fill_gate(h, fake, ips=("192.0.2.101",))
    assert_busy(analyze(h, "192.0.2.102"))
    fake.released.set()
    running.join()


def test_busy_rejection_comes_after_the_upload_and_before_parsing(harness, monkeypatch) -> None:
    import backend.api.routes.analyze as analyze_route

    parsed: list[str] = []
    real_parse = analyze_route.parse_document

    def counting_parse(*args, **kwargs):
        parsed.append("parse")
        return real_parse(*args, **kwargs)

    monkeypatch.setattr(analyze_route, "parse_document", counting_parse)
    h = harness()
    fake = FakeWorkflow()
    running = fill_gate(h, fake)
    assert len(parsed) == 2
    before = len(h.wrapper.body_reads)
    assert_busy(analyze(h, "192.0.2.103"))
    assert h.wrapper.body_reads[before][1] >= 1  # the upload was received...
    assert len(parsed) == 2 and fake.entered == 2  # ...but nothing was parsed or analysed
    fake.released.set()
    running.join()


# --- release on every exit path ---------------------------------------------------------------


def test_slot_is_released_after_a_successful_analysis(harness) -> None:
    h = harness(groq_api_key=FAKE_API_KEY, llm_http_client=scripted(NO_ISSUES).client())
    gate = h.app.state.analyze_gate
    first = analyze(h, "192.0.2.1")  # the real workflow (one scripted analysis)
    assert first.status_code == 200, first.text
    assert gate.in_flight == 0
    store = h.app.state.report_store
    report = h.client.portal.call(store.get, first.json()["report_id"])
    h.app.state.analysis_workflow = FakeWorkflow(block=False, result=report)
    seen: list[int] = []
    h.wrapper.on_response_start = lambda: seen.append(gate.in_flight)
    for i in range(4):  # more successes than slots: a leak would show as a busy 429
        assert analyze(h, f"192.0.2.{i + 2}").status_code == 200
        assert gate.in_flight == 0
    # Released before the 200 is sent (function scope), not after it (request scope).
    assert seen == [0, 0, 0, 0]


@pytest.mark.parametrize(
    ("outcome", "status"),
    [
        (AnalysisTimeoutError(), 504),
        (AppError("Upstream failed.", code="upstream_error", status_code=502), 502),
        (AppError("Busy upstream.", code="upstream_rate_limited", status_code=503), 503),
        (RuntimeError("unexpected failure"), 500),
    ],
)
def test_slot_is_released_after_errors_and_deadlines(harness, outcome, status) -> None:
    h = harness(rate_limit_enabled=False)
    h.app.state.analysis_workflow = FakeWorkflow(outcome, block=False)
    for i in range(4):  # more requests than slots: any leak would show as a busy 429
        assert analyze(h, f"192.0.2.{i + 1}").status_code == status
    assert h.app.state.analyze_gate.in_flight == 0


def test_slot_is_released_before_the_response_is_sent(harness) -> None:
    h = harness(rate_limit_enabled=False)
    h.app.state.analysis_workflow = FakeWorkflow(block=False)
    gate = h.app.state.analyze_gate
    seen: list[int] = []
    h.wrapper.on_response_start = lambda: seen.append(gate.in_flight)
    assert analyze(h).status_code == 504
    assert analyze(h, data=b"not a document").status_code in (415, 422)
    assert seen == [0, 0]  # error responses: released before they are sent


def test_slot_is_released_when_the_upload_is_rejected(harness) -> None:
    h = harness(rate_limit_enabled=False)
    for i in range(4):
        response = analyze(h, f"192.0.2.{i + 1}", data=b"not a document")
        assert response.status_code in (415, 422)
    response = h.request(
        "POST", "/api/v1/analyze", content=b"{}", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 415
    assert h.app.state.analyze_gate.in_flight == 0


# --- interaction with the rate limit -----------------------------------------------------------


def test_busy_requests_are_counted_by_the_rate_limit(harness) -> None:
    """The rate limit runs before the upload is read, so a busy rejection has been
    counted: a client cannot retry a busy server faster than its quota allows."""
    h = harness()  # analyze: 5/minute per client, 20/minute global
    fake = FakeWorkflow()
    running = fill_gate(h, fake)
    for _ in range(5):
        assert_busy(analyze(h, "203.0.113.5"))
    before = len(h.wrapper.body_reads)
    assert_rate_limited(analyze(h, "203.0.113.5"))
    assert h.wrapper.body_reads[before][1] == 0  # rejected before its upload was read
    (pool,) = [w for k, w in h.limiter._global_windows.items() if "analyze_global" in k]
    assert len(pool.entries) == 2 + 5
    fake.released.set()
    running.join()


def test_rate_limited_requests_release_their_slot(harness) -> None:
    h = harness(rate_limit_analyze="1/minute")
    h.app.state.analysis_workflow = FakeWorkflow(block=False)
    assert analyze(h, "203.0.113.5").status_code == 504
    for _ in range(10):
        assert_rate_limited(analyze(h, "203.0.113.5"))
    assert h.app.state.analyze_gate.in_flight == 0
    fake = FakeWorkflow()
    running = fill_gate(h, fake)  # both slots are still usable
    assert_busy(analyze(h, "192.0.2.103"))
    fake.released.set()
    running.join()


def test_the_rate_limit_runs_before_the_busy_check(harness) -> None:
    h = harness(rate_limit_analyze="1/minute")
    h.app.state.analysis_workflow = FakeWorkflow(block=False)
    assert analyze(h, "203.0.113.5").status_code == 504  # client quota now used up
    fake = FakeWorkflow()
    running = fill_gate(h, fake)
    assert_rate_limited(analyze(h, "203.0.113.5"))  # rate_limited wins over busy
    fake.released.set()
    running.join()


def test_rate_limited_requests_never_touch_the_gate(harness, monkeypatch) -> None:
    h = harness(rate_limit_analyze="1/minute")
    h.app.state.analysis_workflow = FakeWorkflow(block=False)
    gate = h.app.state.analyze_gate
    attempts: list[int] = []
    real_try = gate.try_acquire
    monkeypatch.setattr(gate, "try_acquire", lambda: attempts.append(1) or real_try())
    assert analyze(h, "203.0.113.5").status_code == 504
    assert attempts == [1]
    for _ in range(5):
        assert_rate_limited(analyze(h, "203.0.113.5"))
    assert attempts == [1] and gate.in_flight == 0


def test_other_endpoints_are_not_gated(harness) -> None:
    h = harness(rate_limit_enabled=False)
    fake = FakeWorkflow()
    running = fill_gate(h, fake)
    assert h.request("POST", "/api/v1/parse", files={"file": ("cv.docx", DOCX)}).status_code == 200
    assert h.request("GET", "/health").status_code == 200
    fake.released.set()
    running.join()
