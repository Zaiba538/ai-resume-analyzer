"""Phase 3, M1: the document-processing gate (bounded concurrency, soft deadline).

Parsing runs in worker threads. A thread cannot be stopped from outside, so the
gate's deadline only ends the request: the slot stays taken until the worker
really finishes, and the limit therefore counts work that is still running.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from backend.api.document_gate import (
    DOCUMENT_BUSY_RETRY_AFTER_SECONDS,
    DocumentProcessingBusyError,
    DocumentProcessingGate,
    DocumentProcessingTimeoutError,
)
from backend.core.config import Settings
from backend.parsing.errors import UnsupportedFileTypeError

pytestmark = pytest.mark.asyncio


class Blocker:
    """A worker function that blocks until released, then returns or raises."""

    def __init__(self, result: object = "done", error: BaseException | None = None) -> None:
        self.result = result
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def __call__(self, *args: object) -> object:
        self.args = args
        self.started.set()
        try:
            if not self.release.wait(10):
                raise RuntimeError("test blocker was never released")
            if self.error is not None:
                raise self.error
            return self.result
        finally:
            self.finished.set()


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.005)


@pytest.fixture
def gates():
    created: list[DocumentProcessingGate] = []

    def make(max_concurrent: int = 1, deadline_seconds: float = 5.0) -> DocumentProcessingGate:
        gate = DocumentProcessingGate(max_concurrent, deadline_seconds)
        created.append(gate)
        return gate

    yield make
    for gate in created:
        gate.shutdown()


# --- normal operation -----------------------------------------------------------------------


async def test_work_within_the_deadline_returns_its_result_and_frees_the_slot(gates) -> None:
    gate = gates(max_concurrent=2)
    assert await gate.run(lambda a, b: a + b, 2, 3) == 5
    assert gate.in_flight == 0


async def test_worker_errors_propagate_unchanged_and_free_the_slot(gates) -> None:
    gate = gates()
    error = UnsupportedFileTypeError()

    def fail() -> None:
        raise error

    with pytest.raises(UnsupportedFileTypeError) as info:
        await gate.run(fail)
    assert info.value is error
    assert gate.in_flight == 0
    assert await gate.run(lambda: "next") == "next"


async def test_work_runs_off_the_event_loop(gates) -> None:
    gate = gates()
    loop_thread = threading.get_ident()
    assert await gate.run(threading.get_ident) != loop_thread


async def test_request_context_reaches_the_worker(gates) -> None:
    from backend.core.request_id import _request_id, get_request_id

    gate = gates()
    token = _request_id.set("req-ctx-12345678")
    try:
        assert await gate.run(get_request_id) == "req-ctx-12345678"
    finally:
        _request_id.reset(token)


# --- concurrency limit ----------------------------------------------------------------------


async def test_requests_beyond_the_limit_are_rejected_at_once(gates) -> None:
    gate = gates(max_concurrent=2)
    first, second = Blocker(), Blocker()
    running = [asyncio.create_task(gate.run(first)), asyncio.create_task(gate.run(second))]
    await wait_until(lambda: first.started.is_set() and second.started.is_set())
    assert gate.in_flight == 2

    started = time.monotonic()
    with pytest.raises(DocumentProcessingBusyError) as info:
        await gate.run(lambda: "never runs")
    assert time.monotonic() - started < 1.0  # rejected, not queued
    assert (info.value.status_code, info.value.code) == (503, "document_processing_busy")
    assert info.value.headers == {"Retry-After": str(DOCUMENT_BUSY_RETRY_AFTER_SECONDS)}

    first.release.set()
    second.release.set()
    assert await asyncio.gather(*running) == ["done", "done"]
    assert gate.in_flight == 0


async def test_a_rejected_request_never_starts_its_work(gates) -> None:
    gate = gates()
    blocker = Blocker()
    task = asyncio.create_task(gate.run(blocker))
    await wait_until(blocker.started.is_set)
    calls: list[str] = []
    with pytest.raises(DocumentProcessingBusyError):
        await gate.run(lambda: calls.append("ran"))
    blocker.release.set()
    await task
    assert calls == []


# --- deadline: the slot follows the worker, not the request -----------------------------------


async def test_deadline_returns_504_but_the_slot_stays_taken_until_the_worker_ends(
    gates,
) -> None:
    gate = gates(max_concurrent=1, deadline_seconds=0.05)
    blocker = Blocker()
    with pytest.raises(DocumentProcessingTimeoutError) as info:
        await gate.run(blocker)
    assert (info.value.status_code, info.value.code) == (504, "document_processing_timeout")

    # The request has ended, but the worker is still running: its slot is held.
    assert blocker.started.is_set() and not blocker.finished.is_set()
    assert gate.in_flight == 1
    with pytest.raises(DocumentProcessingBusyError):
        await gate.run(lambda: "rejected while the abandoned worker runs")

    blocker.release.set()
    await wait_until(lambda: gate.in_flight == 0)
    assert blocker.finished.is_set()
    assert await gate.run(lambda: "accepted again") == "accepted again"


async def test_late_result_or_error_of_an_abandoned_worker_is_discarded(gates) -> None:
    loop = asyncio.get_running_loop()
    unhandled: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        gate = gates(max_concurrent=2, deadline_seconds=0.05)
        late_error = Blocker(error=UnsupportedFileTypeError())
        late_result = Blocker()
        for blocker in (late_error, late_result):
            with pytest.raises(DocumentProcessingTimeoutError):
                await gate.run(blocker)
        late_error.release.set()
        late_result.release.set()
        await wait_until(lambda: gate.in_flight == 0)
        await asyncio.sleep(0.05)  # let any late delivery run on the loop
    finally:
        loop.set_exception_handler(None)
    assert unhandled == []  # no "exception was never retrieved"


async def test_abandoned_work_is_logged_without_claiming_it_stopped(gates, captured_logs) -> None:
    import logging

    logging.getLogger("backend").setLevel(logging.INFO)
    gate = gates(deadline_seconds=0.05)
    blocker = Blocker()
    with pytest.raises(DocumentProcessingTimeoutError):
        await gate.run(blocker)
    (abandoned,) = captured_logs.events("document_processing_deadline_exceeded")
    assert abandoned["error_code"] == "document_processing_timeout"
    assert abandoned["level"] == "WARNING"

    blocker.release.set()
    await wait_until(lambda: gate.in_flight == 0)
    await wait_until(lambda: captured_logs.events("document_processing_finished_after_deadline"))
    (finished,) = captured_logs.events("document_processing_finished_after_deadline")
    assert finished["duration_ms"] >= 0


# --- cancellation ---------------------------------------------------------------------------


async def test_cancelled_request_keeps_the_slot_until_the_worker_ends(gates) -> None:
    gate = gates(max_concurrent=1)
    blocker = Blocker()
    task = asyncio.create_task(gate.run(blocker))
    await wait_until(blocker.started.is_set)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gate.in_flight == 1  # the worker is still running
    with pytest.raises(DocumentProcessingBusyError):
        await gate.run(lambda: "rejected")

    blocker.release.set()
    await wait_until(lambda: gate.in_flight == 0)
    assert await gate.run(lambda: "free again") == "free again"


async def test_cancellation_before_the_worker_starts_still_releases_the_slot(gates) -> None:
    gate = gates(max_concurrent=1)
    started = threading.Event()

    def quick() -> str:
        started.set()
        return "ran"

    task = asyncio.create_task(gate.run(quick))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_until(lambda: gate.in_flight == 0)


# --- configuration and lifecycle ------------------------------------------------------------


@pytest.mark.parametrize(("max_concurrent", "deadline"), [(0, 1.0), (1, 0.0), (1, -1.0)])
async def test_invalid_gate_configuration_is_rejected(max_concurrent, deadline) -> None:
    with pytest.raises(ValueError):
        DocumentProcessingGate(max_concurrent, deadline)


async def test_settings_defaults_and_bounds() -> None:
    defaults = Settings(_env_file=None)
    assert defaults.parse_max_concurrent == 2
    assert defaults.parse_deadline_seconds == 30.0
    for field, bad in (
        ("parse_max_concurrent", 0),
        ("parse_max_concurrent", 9),
        ("parse_deadline_seconds", 0.5),
        ("parse_deadline_seconds", 301),
    ):
        with pytest.raises(ValueError):
            Settings(_env_file=None, **{field: bad})


async def test_shutdown_does_not_wait_for_a_stuck_worker(gates) -> None:
    gate = gates(deadline_seconds=0.05)
    blocker = Blocker()
    with pytest.raises(DocumentProcessingTimeoutError):
        await gate.run(blocker)
    started = time.monotonic()
    gate.shutdown()
    assert time.monotonic() - started < 1.0
    assert gate.closed
    blocker.release.set()
    await wait_until(blocker.finished.is_set)


# --- L1: the slot is released at the hand-off, after all of the worker's work -----------------


async def test_slot_is_still_held_when_the_worker_hands_off_its_result(gates, monkeypatch) -> None:
    gate = gates()
    loop = asyncio.get_running_loop()
    seen_at_hand_off: list[int] = []
    original = loop.call_soon_threadsafe

    def recording(callback, *args):
        seen_at_hand_off.append(gate.in_flight)  # called from the worker thread
        return original(callback, *args)

    monkeypatch.setattr(loop, "call_soon_threadsafe", recording)
    assert await gate.run(lambda: "parsed") == "parsed"
    assert seen_at_hand_off == [1]  # the worker's last step ran while it held the slot
    assert gate.in_flight == 0  # released before the waiting request resumed


async def test_abandoned_worker_holds_its_slot_while_it_logs(gates) -> None:
    import logging

    gate = gates(deadline_seconds=0.05)
    blocker = Blocker()
    seen_while_logging: list[int] = []

    class Probe(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.msg == "document_processing_finished_after_deadline":
                seen_while_logging.append(gate.in_flight)

    log = logging.getLogger("backend.api.document_gate")
    probe = Probe()
    log.addHandler(probe)
    previous = log.level
    log.setLevel(logging.INFO)
    try:
        with pytest.raises(DocumentProcessingTimeoutError):
            await gate.run(blocker)
        blocker.release.set()
        await wait_until(lambda: gate.in_flight == 0)
    finally:
        log.removeHandler(probe)
        log.setLevel(previous)
    assert seen_while_logging == [1]


async def test_back_to_back_requests_never_see_a_stale_slot(gates) -> None:
    gate = gates(max_concurrent=1)
    for i in range(100):  # a slot released after the result would make some of these 503
        assert await gate.run(lambda i=i: i) == i
    assert gate.in_flight == 0


async def test_worker_releases_its_slot_when_the_hand_off_is_impossible(gates, monkeypatch) -> None:
    gate = gates(deadline_seconds=0.2)
    loop = asyncio.get_running_loop()

    def closed_loop(callback, *args):
        raise RuntimeError("Event loop is closed")

    finished = threading.Event()

    def work() -> str:
        finished.set()
        return "result nobody can receive"

    monkeypatch.setattr(loop, "call_soon_threadsafe", closed_loop)
    with pytest.raises(DocumentProcessingTimeoutError):
        await gate.run(work)
    assert finished.is_set()
    assert gate.in_flight == 0  # released by the worker's finally path


# --- L2: a closed gate answers 503, never 500 ---------------------------------------------------


async def test_run_after_shutdown_is_busy_not_an_internal_error(gates) -> None:
    gate = gates()
    gate.shutdown()
    with pytest.raises(DocumentProcessingBusyError) as info:
        await gate.run(lambda: "never runs")
    assert info.value.status_code == 503
    assert info.value.headers == {"Retry-After": str(DOCUMENT_BUSY_RETRY_AFTER_SECONDS)}
    assert gate.in_flight == 0


async def test_shutdown_racing_a_submission_is_busy_and_frees_the_slot(gates, monkeypatch) -> None:
    gate = gates()

    def shut_down(*args, **kwargs):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(gate._executor, "submit", shut_down)
    with pytest.raises(DocumentProcessingBusyError):
        await gate.run(lambda: "never runs")
    assert gate.in_flight == 0


async def test_a_closed_gate_keeps_accounting_for_its_running_worker(gates) -> None:
    gate = gates(deadline_seconds=0.05)
    blocker = Blocker()
    with pytest.raises(DocumentProcessingTimeoutError):
        await gate.run(blocker)
    gate.shutdown()
    assert gate.in_flight == 1  # shutdown does not pretend the worker stopped
    blocker.release.set()
    await wait_until(lambda: gate.in_flight == 0)
