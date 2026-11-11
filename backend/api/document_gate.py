"""Bounded document processing (Phase 3 security audit, finding M1).

Parsing a PDF or DOCX is synchronous, CPU-bound work that runs in a worker
thread. One ``DocumentProcessingGate`` per application bounds it for every
endpoint that parses an upload (/parse, /match with a file, /analyze):

* Concurrency. At most ``max_concurrent`` documents are processed at a time,
  across all those endpoints together. A request that finds every slot taken
  is rejected at once with 503 ``document_processing_busy`` and
  ``Retry-After``; nothing waits in a queue.
* Deadline. A request waits at most ``deadline_seconds`` for its document and
  then gets 504 ``document_processing_timeout``. This is a soft deadline: Python
  cannot stop a running thread, so the worker carries on until the parser
  returns, and its result is discarded.
* Accounting. A slot belongs to the worker, not to the request. It is taken
  before the work is submitted and released only when the worker hands its
  result back to the event loop, after all of its work (parsing and logging);
  if the hand-off is impossible because the loop has closed, the worker
  releases it itself. A request that times out or is cancelled (for example
  because the client disconnected) therefore never frees a slot while its
  thread is still working, and the limit always counts work in progress.
* Lifecycle. A gate lives for one application lifespan. ``shutdown`` stops new
  work (later requests get 503 ``document_processing_busy``) without waiting
  for, or stopping, running workers. A new lifespan creates a new gate.

The workers run in the gate's own thread pool (``max_concurrent`` threads), so
document parsing cannot occupy the shared thread pool used for other work.
Waiting for the result never blocks the event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from backend.core.errors import AppError

logger = logging.getLogger("backend.api.document_gate")

DOCUMENT_BUSY_RETRY_AFTER_SECONDS = 5
DOCUMENT_BUSY_DESCRIPTION = (
    "the maximum number of documents is already being processed "
    "(`document_processing_busy`); see `Retry-After`"
)
DOCUMENT_TIMEOUT_DESCRIPTION = (
    "document processing exceeded its deadline (`document_processing_timeout`)"
)


class DocumentProcessingBusyError(AppError):
    status_code = 503
    code = "document_processing_busy"
    message = "The server is busy processing other documents. Please try again shortly."

    def __init__(self) -> None:
        super().__init__(headers={"Retry-After": str(DOCUMENT_BUSY_RETRY_AFTER_SECONDS)})


class DocumentProcessingTimeoutError(AppError):
    status_code = 504
    code = "document_processing_timeout"
    message = (
        "The document took too long to process. Please try a simpler or smaller file, "
        "or try again later."
    )


class _Job:
    """One submitted piece of work; ``abandoned`` is set when its request stops waiting."""

    __slots__ = ("abandoned", "started")

    def __init__(self) -> None:
        self.abandoned = False
        self.started = time.monotonic()


class DocumentProcessingGate:
    """Concurrency limit and soft deadline for document processing; one per lifespan."""

    def __init__(self, max_concurrent: int, deadline_seconds: float) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        if not deadline_seconds > 0:
            raise ValueError("deadline_seconds must be positive")
        self._max = max_concurrent
        self._deadline = float(deadline_seconds)
        self._in_flight = 0
        self._lock = threading.Lock()  # shared with the worker threads
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="document-processing"
        )
        self._closed = False

    @property
    def max_concurrent(self) -> int:
        return self._max

    @property
    def deadline_seconds(self) -> float:
        return self._deadline

    @property
    def in_flight(self) -> int:
        """Slots taken by work that has not finished, including abandoned work."""
        with self._lock:
            return self._in_flight

    @property
    def closed(self) -> bool:
        return self._closed

    def _try_acquire(self) -> bool:
        with self._lock:
            if self._in_flight >= self._max:
                return False
            self._in_flight += 1
            return True

    def _release(self) -> None:
        with self._lock:
            self._in_flight -= 1

    async def run[T](self, fn: Callable[..., T], /, *args: Any) -> T:
        """Run ``fn(*args)`` in a worker thread under the limit and the deadline.

        Raises ``DocumentProcessingBusyError`` (503) when no slot is free and
        ``DocumentProcessingTimeoutError`` (504) when the deadline passes; any
        exception raised by ``fn`` itself is re-raised unchanged.
        """
        # A closed gate (application shutting down) answers like a full one.
        if self._closed or not self._try_acquire():
            raise DocumentProcessingBusyError()

        loop = asyncio.get_running_loop()
        outcome: asyncio.Future[T] = loop.create_future()
        job = _Job()

        def finish(result: T | None, error: BaseException | None) -> None:
            # Runs on the event loop and is the worker's hand-off: the slot is
            # released here, after every step of the worker (including its
            # logging), and before the waiting request can resume. A request that
            # stopped waiting has cancelled ``outcome``; its late result or error
            # is dropped.
            try:
                if outcome.done():
                    return
                if error is not None:
                    outcome.set_exception(error)
                else:
                    outcome.set_result(result)  # type: ignore[arg-type]
            finally:
                self._release()

        def work() -> None:
            # Runs in a worker thread and owns the slot until ``finish`` runs.
            handed_off = False
            try:
                result: T | None = None
                error: BaseException | None = None
                try:
                    result = fn(*args)
                except Exception as exc:  # delivered to the waiting request, if any
                    error = exc
                if job.abandoned:
                    logger.info(
                        "document_processing_finished_after_deadline",
                        extra={
                            "component": "document_processing",
                            "duration_ms": round((time.monotonic() - job.started) * 1000),
                        },
                    )
                # RuntimeError: the event loop has closed, so nobody is waiting;
                # the slot is then released below instead.
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(finish, result, error)
                    handed_off = True
            finally:
                if not handed_off:
                    self._release()

        try:
            # Copy the request's context (request ID) into the worker, as
            # Starlette's run_in_threadpool does.
            self._executor.submit(contextvars.copy_context().run, work)
        except RuntimeError:
            # The pool was shut down concurrently: the work never started.
            self._release()
            raise DocumentProcessingBusyError() from None
        except BaseException:
            self._release()  # the work never started
            raise

        try:
            return await asyncio.wait_for(outcome, timeout=self._deadline)
        except TimeoutError:
            job.abandoned = True
            logger.warning(
                "document_processing_deadline_exceeded",
                extra={
                    "component": "document_processing",
                    "error_code": DocumentProcessingTimeoutError.code,
                },
            )
            raise DocumentProcessingTimeoutError() from None
        except asyncio.CancelledError:
            job.abandoned = True  # the worker keeps running and keeps its slot
            raise

    def shutdown(self) -> None:
        """Stop accepting work, without waiting for running workers to finish.

        Running workers are not stopped (Python cannot stop a thread): each one
        carries on until its parser returns and then releases its slot on this
        gate. Later calls to ``run`` get 503 ``document_processing_busy``. A
        closed gate is never reopened; a new application lifespan creates a new
        gate, so a closed gate's late workers cannot affect it.
        """
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=False)
