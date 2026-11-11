"""Phase 3, M1: document processing is bounded through the real endpoints.

/parse, /match (multipart) and /analyze share one document-processing gate:
requests beyond its limit get 503 ``document_processing_busy`` at once, and
parsing that outlives the deadline gets 504 ``document_processing_timeout``
while its slot stays taken until the worker thread really finishes.
"""

from __future__ import annotations

import threading
import time

import pytest

import backend.api.routes.analyze as analyze_route
import backend.api.routes.match as match_route
import backend.api.routes.parse as parse_route
from backend.api.document_gate import DOCUMENT_BUSY_RETRY_AFTER_SECONDS
from backend.parsing.service import parse_document as real_parse
from tests.conftest import FAKE_API_KEY
from tests.fixtures.documents import make_docx
from tests.integration.test_analysis_workflow import NO_ISSUES, scripted
from tests.unit.test_critic_checks import JOB, RESUME

DOCX = make_docx(paragraphs=[line for line in RESUME.splitlines() if line.strip()])
ROUTES = {"parse": parse_route, "match": match_route, "analyze": analyze_route}
PATHS = {"parse": "/api/v1/parse", "match": "/api/v1/match", "analyze": "/api/v1/analyze"}


class BlockingParse:
    """Replaces parse_document: blocks until released, then parses for real."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def __call__(self, *args, **kwargs):
        self.entered.set()
        try:
            if not self.release.wait(30):
                raise RuntimeError("blocking parse was never released")
            return real_parse(*args, **kwargs)
        finally:
            self.finished.set()


@pytest.fixture
def blockers():
    created: list[BlockingParse] = []

    def make() -> BlockingParse:
        blocker = BlockingParse()
        created.append(blocker)
        return blocker

    yield make
    for blocker in created:  # never leave a worker thread blocked after a test
        blocker.release.set()


def post(client, endpoint: str):
    data = {"job_description": JOB} if endpoint in ("match", "analyze") else {}
    return client.post(PATHS[endpoint], files={"file": ("cv.docx", DOCX)}, data=data)


def in_background(client, endpoint: str) -> tuple[threading.Thread, list]:
    responses: list = []
    thread = threading.Thread(target=lambda: responses.append(post(client, endpoint)), daemon=True)
    thread.start()
    return thread, responses


def wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.005)


def assert_envelope(response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    assert error["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["Cache-Control"] == "no-store"


def assert_busy(response) -> None:
    assert_envelope(response, 503, "document_processing_busy")
    assert response.headers["Retry-After"] == str(DOCUMENT_BUSY_RETRY_AFTER_SECONDS)


# --- one shared limit -----------------------------------------------------------------------


def test_one_limit_is_shared_by_parse_match_and_analyze(start_app, monkeypatch, blockers) -> None:
    client = start_app(parse_max_concurrent=1)
    gate = client.app.state.document_gate
    blocker = blockers()
    monkeypatch.setattr(parse_route, "parse_document", blocker)

    thread, responses = in_background(client, "parse")
    wait_for(blocker.entered.is_set)
    assert gate.in_flight == 1

    for endpoint in ("parse", "match", "analyze"):
        assert_busy(post(client, endpoint))
    # A /match request with JSON text parses no document and is not gated.
    text = client.post("/api/v1/match", json={"resume_text": RESUME, "job_description": JOB})
    assert text.status_code == 200, text.text

    blocker.release.set()
    thread.join(30)
    assert responses[0].status_code == 200, responses[0].text
    assert gate.in_flight == 0
    monkeypatch.setattr(parse_route, "parse_document", real_parse)
    assert post(client, "parse").status_code == 200


def test_requests_within_the_limit_run_concurrently(start_app, monkeypatch, blockers) -> None:
    client = start_app(parse_max_concurrent=2)
    gate = client.app.state.document_gate
    first, second = blockers(), blockers()
    monkeypatch.setattr(parse_route, "parse_document", first)
    monkeypatch.setattr(match_route, "parse_document", second)

    threads = [in_background(client, "parse"), in_background(client, "match")]
    wait_for(lambda: first.entered.is_set() and second.entered.is_set())
    assert gate.in_flight == 2
    assert_busy(post(client, "analyze"))  # the third is rejected, not queued

    first.release.set()
    second.release.set()
    for thread, responses in threads:
        thread.join(30)
        assert responses[0].status_code == 200, responses[0].text
    assert gate.in_flight == 0


def test_max_concurrent_setting_is_honoured(start_app) -> None:
    assert start_app(parse_max_concurrent=3).app.state.document_gate.max_concurrent == 3
    assert start_app().app.state.document_gate.max_concurrent == 2


# --- the deadline ---------------------------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["parse", "match", "analyze"])
def test_deadline_returns_504_and_the_slot_follows_the_worker(
    start_app, monkeypatch, blockers, endpoint
) -> None:
    client = start_app(parse_max_concurrent=1, parse_deadline_seconds=1.0)
    gate = client.app.state.document_gate
    blocker = blockers()
    monkeypatch.setattr(ROUTES[endpoint], "parse_document", blocker)

    started = time.monotonic()
    assert_envelope(post(client, endpoint), 504, "document_processing_timeout")
    assert 0.9 <= time.monotonic() - started < 10

    # The request has ended; the worker has not, so its slot is still taken.
    assert not blocker.finished.is_set()
    assert gate.in_flight == 1
    assert_busy(post(client, "parse"))
    if endpoint == "analyze":
        assert client.app.state.analyze_gate.in_flight == 0  # the analysis slot is freed

    blocker.release.set()
    wait_for(lambda: gate.in_flight == 0)
    assert blocker.finished.is_set()
    assert post(client, "parse").status_code == 200  # capacity is back


def test_processing_within_the_deadline_is_unchanged(start_app) -> None:
    client = start_app(parse_max_concurrent=1, parse_deadline_seconds=1.0)
    gate = client.app.state.document_gate
    for _ in range(3):  # more requests than slots: a leaked slot would show as a 503
        for endpoint in ("parse", "match"):
            response = post(client, endpoint)
            assert response.status_code == 200, response.text
            assert gate.in_flight == 0


def test_a_successful_analysis_releases_its_document_slot(start_app_with_llm) -> None:
    client = start_app_with_llm()
    response = post(client, "analyze")
    assert response.status_code == 200, response.text
    assert client.app.state.document_gate.in_flight == 0
    assert client.app.state.analyze_gate.in_flight == 0


def test_document_errors_keep_their_codes_and_free_the_slot(start_app) -> None:
    client = start_app(parse_max_concurrent=1)
    for _ in range(3):
        response = client.post(PATHS["parse"], files={"file": ("cv.txt", b"plain text")})
        assert response.status_code == 415, response.text
        assert response.json()["error"]["code"] == "unsupported_file_type"
    assert client.app.state.document_gate.in_flight == 0


# --- lifecycle and documentation ------------------------------------------------------------


def test_the_gate_is_shut_down_with_the_application(start_app) -> None:
    client = start_app()
    gate = client.app.state.document_gate
    assert not gate.closed
    client.__exit__(None, None, None)
    assert gate.closed


@pytest.mark.parametrize("endpoint", ["parse", "match", "analyze"])
def test_openapi_documents_busy_and_timeout(start_app, endpoint) -> None:
    responses = start_app().app.openapi()["paths"][PATHS[endpoint]]["post"]["responses"]
    assert "document_processing_busy" in responses["503"]["description"]
    assert "document_processing_timeout" in responses["504"]["description"]


@pytest.fixture
def start_app_with_llm(make_settings, fake_groq):
    """An app whose analysis workflow runs against a scripted fake Groq API."""
    from fastapi.testclient import TestClient

    from backend.main import create_app
    from tests.fixtures.embeddings import CountingFactory

    clients: list[TestClient] = []

    def _start(**overrides) -> TestClient:
        app = create_app(
            make_settings(groq_api_key=FAKE_API_KEY, **overrides),
            groq_http_client=fake_groq.client(),
            llm_http_client=scripted(NO_ISSUES).client(),
            embedder_factory=CountingFactory(),
        )
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield _start
    for client in clients:
        client.__exit__(None, None, None)


# --- L2: shutdown and restart -------------------------------------------------------------------


def test_a_request_meeting_a_closed_gate_gets_503_not_500(start_app) -> None:
    client = start_app()
    client.app.state.document_gate.shutdown()  # as during application shutdown
    for endpoint in ("parse", "match", "analyze"):
        assert_busy(post(client, endpoint))


def test_the_same_application_can_start_a_new_lifespan(start_app) -> None:
    from fastapi.testclient import TestClient

    first = start_app()
    old_gate = first.app.state.document_gate
    assert post(first, "parse").status_code == 200
    first.__exit__(None, None, None)
    assert old_gate.closed

    with TestClient(first.app, raise_server_exceptions=False) as second:
        new_gate = second.app.state.document_gate
        assert new_gate is not old_gate and not new_gate.closed
        for endpoint in ("parse", "match"):
            response = post(second, endpoint)
            assert response.status_code == 200, response.text
        assert new_gate.in_flight == 0
    assert new_gate.closed


def test_an_old_gate_and_its_late_worker_do_not_affect_the_new_gate(
    start_app, monkeypatch, blockers
) -> None:
    from fastapi.testclient import TestClient

    first = start_app(parse_max_concurrent=1, parse_deadline_seconds=1.0)
    old_gate = first.app.state.document_gate
    blocker = blockers()
    monkeypatch.setattr(parse_route, "parse_document", blocker)
    assert_envelope(post(first, "parse"), 504, "document_processing_timeout")
    first.__exit__(None, None, None)  # old gate closed while its worker still runs
    assert old_gate.closed and old_gate.in_flight == 1
    monkeypatch.setattr(parse_route, "parse_document", real_parse)

    with TestClient(first.app, raise_server_exceptions=False) as second:
        new_gate = second.app.state.document_gate
        assert new_gate is not old_gate and new_gate.in_flight == 0
        assert post(second, "parse").status_code == 200  # full capacity, not blocked
        blocker.release.set()  # the old worker finishes now
        wait_for(lambda: old_gate.in_flight == 0)
        assert new_gate.in_flight == 0  # its release went to the old gate only
        assert post(second, "match").status_code == 200
