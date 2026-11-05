"""M8 Phase 4: ``GET /api/v1/report/{report_id}/pdf`` and the PDF readiness check."""

from __future__ import annotations

import builtins
import io
import logging
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfReader

from backend.api.routes import report as report_route
from backend.core.logging import ALLOWED_FIELDS
from backend.main import create_app
from backend.reports import fonts, pdf
from backend.reports.errors import (
    PdfPageLimitError,
    PdfRendererUnavailableError,
    PdfRenderTimeoutError,
)
from backend.reports.fonts import FontRegistry, default_registry
from backend.reports.labels import label
from backend.reports.pdf import FOOTER_DISCLAIMER, render_report_pdf
from backend.schemas.report import AnalysisReport
from backend.schemas.resume import ResumeProfile
from tests.conftest import FAKE_API_KEY
from tests.fixtures.documents import make_pdf
from tests.fixtures.embeddings import CountingFactory
from tests.fixtures.reports import PROFILE, synthetic_report
from tests.integration.test_analysis_workflow import NO_ISSUES, scripted
from tests.unit.test_critic_checks import JOB, RESUME

REPORT = "/api/v1/report"
FILENAME = "resume-analysis-report.pdf"
DISPOSITION = f'attachment; filename="{FILENAME}"'
NOT_FOUND = (404, "report_not_found", "The report was not found or has expired.")
CANARY = "CANARYpdf7e21"


@pytest.fixture(autouse=True)
def _info_logs() -> None:
    logging.getLogger("backend").setLevel(logging.INFO)


def _store(client: TestClient, **overrides) -> str:
    store = client.app.state.report_store
    report = client.portal.call(store.create, lambda stamp: synthetic_report(stamp, **overrides))
    return str(report.report_id)


def _pdf(client: TestClient, report_id: str):
    return client.get(f"{REPORT}/{report_id}/pdf")


def _error(response) -> tuple[int, str, str]:
    body = response.json()["error"]
    assert body["request_id"] == response.headers["X-Request-ID"]
    return response.status_code, body["code"], body["message"]


def _text(data: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)


def _assert_private(response) -> None:
    assert response.headers["Cache-Control"] == "no-store"
    assert "noindex" in response.headers["X-Robots-Tag"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in response.headers


class _Blocked:
    """Stands in for the renderer: blocks the worker thread until released."""

    def __init__(self, then=None) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.then = then

    def __call__(self, report, **limits) -> bytes:
        self.started.set()
        try:
            self.release.wait(timeout=30)
            return self.then(report, **limits) if self.then else b"%PDF-late"
        finally:
            self.finished.set()


# --- download ------------------------------------------------------------------------


def test_pdf_download_is_valid_with_fixed_filename_and_content_type(start_app) -> None:
    client = start_app()
    report_id = _store(client)
    response = _pdf(client, report_id)
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/pdf"
    assert response.headers["Content-Disposition"] == DISPOSITION
    assert int(response.headers["Content-Length"]) == len(response.content)
    assert response.content.startswith(b"%PDF-") and response.content.rstrip().endswith(b"%%EOF")
    text = _text(response.content)
    for expected in (
        label("document", "title"),
        label("sections", "resume_profile"),
        label("sections", "interview_questions"),
        label("sections", "match"),
        label("sections", "critic"),
        label("sections", "security"),
        "Apache Spark",
        "How would you tune a Spark job?",
        "Minor wording.",
        FOOTER_DISCLAIMER,
    ):
        assert expected in text, expected
    assert label("ats", "not_calibrated") in text


def test_pdf_matches_the_renderer_output_for_the_stored_report(start_app) -> None:
    client = start_app()
    report_id = _store(client)
    stored = AnalysisReport.model_validate(client.get(f"{REPORT}/{report_id}").json())
    expected = render_report_pdf(stored, max_pages=100, deadline_seconds=15)
    assert _pdf(client, report_id).content == expected


def test_report_id_is_not_in_headers_filename_or_pdf(start_app) -> None:
    client = start_app()
    report_id = _store(client)
    response = _pdf(client, report_id)
    headers = "\n".join(f"{k}: {v}" for k, v in response.headers.items())
    assert report_id not in headers and report_id.replace("-", "") not in headers
    assert report_id.encode() not in response.content
    assert report_id not in _text(response.content)


def test_pdf_responses_are_no_store_and_noindex(start_app) -> None:
    client = start_app()
    response = _pdf(client, _store(client))
    assert response.status_code == 200
    _assert_private(response)
    _assert_private(_pdf(client, str(uuid.uuid4())))


# --- identical 404s ----------------------------------------------------------------------


def test_unknown_malformed_and_expired_ids_return_the_same_404(start_app) -> None:
    client = start_app()
    stored = _store(client)
    store = client.app.state.report_store
    store._clock = lambda: datetime.now(UTC) + timedelta(seconds=3601)  # past the 1 h TTL
    ids = [
        str(uuid.uuid4()),  # unknown
        "not-a-uuid",
        str(uuid.uuid1()),
        str(uuid.uuid4()).upper()[:-1] + "Z",
        "abc%20def",
        str(uuid.uuid4()) + "0",
        stored,  # expired
    ]
    responses = [_pdf(client, report_id) for report_id in ids]
    assert {_error(r) for r in responses} == {NOT_FOUND}
    for response in responses:
        _assert_private(response)
        assert set(response.json()) == {"error"} and set(response.json()["error"]) == {
            "code",
            "message",
            "request_id",
        }
    assert store._reports == {}  # the expired report was purged on access
    json_404 = client.get(f"{REPORT}/{uuid.uuid4()}")
    assert _error(json_404) == NOT_FOUND  # same as the JSON endpoint


# --- error mapping (E1, E2) -----------------------------------------------------------


def test_page_limit_maps_to_422(start_app, monkeypatch) -> None:
    client = start_app()
    report_id = _store(client)
    assert len(PdfReader(io.BytesIO(_pdf(client, report_id).content)).pages) > 1
    monkeypatch.setattr(report_route, "PDF_MAX_PAGES", 1)
    response = _pdf(client, report_id)
    assert _error(response) == (
        422,
        "pdf_page_limit_exceeded",
        "The report is too long to render as a PDF.",
    )
    assert response.headers["Content-Type"] == "application/json"
    assert "Content-Disposition" not in response.headers
    _assert_private(response)


class _JumpClock:
    """Monotonic clock that passes any deadline after ``readings`` readings."""

    def __init__(self, readings: int) -> None:
        self.left = readings

    def __call__(self) -> float:
        self.left -= 1
        return 0.0 if self.left >= 0 else 1e9


def test_renderer_deadline_maps_to_504(start_app, monkeypatch) -> None:
    client = start_app()
    report_id = _store(client)
    clock = _JumpClock(readings=4)
    monkeypatch.setattr(pdf, "_monotonic", clock)
    response = _pdf(client, report_id)
    assert clock.left < 0  # the renderer's own deadline check raised
    assert _error(response) == (
        504,
        "pdf_render_timeout",
        "Generating the PDF report took too long.",
    )
    assert "Content-Disposition" not in response.headers
    _assert_private(response)


def test_fonts_unavailable_at_startup_fail_closed_with_503(
    start_app, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(fonts, "_default_registry", FontRegistry(tmp_path / "missing"))
    client = start_app()
    ready = client.get("/ready")
    assert ready.status_code == 503 and ready.json()["status"] == "not_ready"
    assert ready.json()["checks"]["pdf_renderer_ready"] is False
    stored = _store(client)
    unavailable = (503, "pdf_renderer_unavailable", "The PDF renderer is not available.")
    for report_id in (stored, str(uuid.uuid4()), "not-a-uuid"):
        response = _pdf(client, report_id)
        assert _error(response) == unavailable  # the same for every ID
        _assert_private(response)
    assert client.get(f"{REPORT}/{stored}").status_code == 200  # JSON is unaffected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (PdfPageLimitError(100), (422, "pdf_page_limit_exceeded")),
        (PdfRenderTimeoutError(15), (504, "pdf_render_timeout")),
        (PdfRendererUnavailableError(), (503, "pdf_renderer_unavailable")),
    ],
)
def test_typed_renderer_errors_map_to_their_contract(
    start_app, monkeypatch, error, expected
) -> None:
    client = start_app()
    report_id = _store(client)

    def failing(report, **limits) -> bytes:
        raise error

    monkeypatch.setattr(report_route, "render_report_pdf", failing)
    response = _pdf(client, report_id)
    assert _error(response) == (*expected, error.message)


def test_unexpected_renderer_error_is_a_generic_500(start_app, monkeypatch, captured_logs) -> None:
    client = start_app()
    report_id = _store(client)

    def broken(report, **limits) -> bytes:
        raise RuntimeError(f"secret {CANARY}")

    monkeypatch.setattr(report_route, "render_report_pdf", broken)
    response = _pdf(client, report_id)
    assert _error(response) == (500, "internal_error", "An internal error occurred.")
    assert CANARY not in response.text and CANARY not in captured_logs.text


# --- timeout of the worker thread ---------------------------------------------------


def test_request_deadline_returns_504_while_the_worker_is_abandoned(
    start_app, monkeypatch, captured_logs
) -> None:
    client = start_app()
    report_id = _store(client)
    blocked = _Blocked()
    monkeypatch.setattr(report_route, "render_report_pdf", blocked)
    monkeypatch.setattr(report_route, "PDF_DEADLINE_SECONDS", 0.3)
    try:
        started = time.monotonic()
        response = _pdf(client, report_id)
        elapsed = time.monotonic() - started
        assert _error(response) == (
            504,
            "pdf_render_timeout",
            "Generating the PDF report took too long.",
        )
        assert "cancel" not in response.text.lower() and "terminat" not in response.text.lower()
        assert 0.3 <= elapsed < 5
        # The request returned while the worker thread is still running: it was not stopped.
        assert blocked.started.is_set() and not blocked.finished.is_set()
    finally:
        blocked.release.set()
    assert blocked.finished.wait(timeout=10)  # the worker ends; its result is discarded
    (abandoned,) = captured_logs.events("report_pdf_render_abandoned")
    assert abandoned["error_code"] == "pdf_render_timeout"
    assert abandoned["level"] == "WARNING"
    assert not captured_logs.events("report_pdf_rendered")


def test_abandoned_render_keeps_the_lock_and_fonts_consistent(start_app, monkeypatch) -> None:
    client = start_app()
    report_id = _store(client, resume_profile=_profile(skills=["हिन्दी", "क्षमता ज्ञान"]))
    expected = _pdf(client, report_id).content
    blocked = _Blocked(then=render_report_pdf)  # the real renderer, once released
    monkeypatch.setattr(report_route, "render_report_pdf", blocked)
    monkeypatch.setattr(report_route, "PDF_DEADLINE_SECONDS", 0.3)
    try:
        assert _pdf(client, report_id).status_code == 504
    finally:
        blocked.release.set()
    assert blocked.finished.wait(timeout=30)  # the abandoned render completed under the lock
    monkeypatch.undo()
    assert pdf._RENDER_LOCK.acquire(blocking=False)
    pdf._RENDER_LOCK.release()
    assert FontRegistry().register()  # original glyph and width entries unchanged
    assert _pdf(client, report_id).content == expected


def test_rendering_does_not_block_the_event_loop(start_app, monkeypatch) -> None:
    client = start_app()
    report_id = _store(client)
    blocked = _Blocked(then=render_report_pdf)
    monkeypatch.setattr(report_route, "render_report_pdf", blocked)
    results: list[int] = []
    request = threading.Thread(target=lambda: results.append(_pdf(client, report_id).status_code))
    request.start()
    others: list[int] = []

    def other_requests() -> None:
        others.append(client.get("/health").status_code)
        others.append(client.get(f"{REPORT}/{report_id}").status_code)

    try:
        assert blocked.started.wait(timeout=10)
        # Served while the worker is still rendering, not after it finishes.
        served = threading.Thread(target=other_requests, daemon=True)
        served.start()
        served.join(timeout=10)
        assert others == [200, 200] and not blocked.finished.is_set()
    finally:
        blocked.release.set()
        request.join(timeout=30)
    assert results == [200]


def test_renderer_gets_the_approved_limits(start_app, monkeypatch) -> None:
    client = start_app()
    report_id = _store(client)
    seen: list[tuple[AnalysisReport, dict]] = []
    real = report_route.render_report_pdf

    def spy(report, **limits) -> bytes:
        seen.append((report, limits))
        return real(report, **limits)

    monkeypatch.setattr(report_route, "render_report_pdf", spy)
    assert _pdf(client, report_id).status_code == 200
    ((report, limits),) = seen
    assert limits == {"max_pages": 100, "deadline_seconds": 15.0}  # M8 decision D5
    assert type(report) is AnalysisReport and str(report.report_id) == report_id


# --- readiness --------------------------------------------------------------------------


def test_readiness_reports_the_renderer_font_registry(start_app) -> None:
    client = start_app()
    checks = client.get("/ready").json()["checks"]
    assert checks["pdf_renderer_ready"] is True
    assert default_registry().ready is True
    assert client.app.state.readiness.snapshot()["pdf_renderer_ready"] is True


def test_readiness_is_set_by_the_lifespan_from_the_registry(make_settings, monkeypatch, tmp_path):
    monkeypatch.setattr(fonts, "_default_registry", FontRegistry(tmp_path / "missing"))
    app = create_app(make_settings(), embedder_factory=CountingFactory())
    assert app.state.readiness.snapshot()["pdf_renderer_ready"] is False  # not before start-up
    with TestClient(app, raise_server_exceptions=False):
        assert app.state.readiness.snapshot()["pdf_renderer_ready"] is False
        assert fonts.default_registry().error_code == "manifest_invalid"


# --- privacy, caching and persistence ------------------------------------------------


def _profile(**fields) -> ResumeProfile:
    return ResumeProfile.model_validate({**PROFILE.model_dump(), **fields})


def test_logs_contain_no_report_content_or_ids(start_app, captured_logs) -> None:
    client = start_app()
    report_id = _store(client, resume_profile=_profile(skills=[f"{CANARY}-skill"]))
    response = _pdf(client, report_id)
    assert response.status_code == 200 and CANARY in _text(response.content)
    assert CANARY not in captured_logs.text and report_id not in captured_logs.text
    assert FILENAME not in captured_logs.text
    (rendered,) = captured_logs.events("report_pdf_rendered")
    assert rendered["component"] == "report_pdf" and rendered["duration_ms"] >= 0
    for record in captured_logs.records():
        assert set(record) <= ALLOWED_FIELDS | {"ts", "level", "logger", "event"}
    routes = {r["route"] for r in captured_logs.events("request_completed")}
    assert routes == {"/api/v1/report/{report_id}/pdf"}


def test_pdf_is_rendered_per_request_and_never_cached_or_written(
    start_app, monkeypatch, tmp_path: Path
) -> None:
    client = start_app()
    report_id = _store(client)
    calls: list[int] = []
    real = report_route.render_report_pdf

    def counting(report, **limits) -> bytes:
        calls.append(1)
        return real(report, **limits)

    monkeypatch.setattr(report_route, "render_report_pdf", counting)
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"file opened for writing: {mode}")
        return real_open(file, mode, *args, **kwargs)

    def no_temp(*args, **kwargs):
        raise AssertionError("temporary file created")

    before = sorted(p.name for p in tmp_path.iterdir())
    monkeypatch.setattr(builtins, "open", guarded_open)
    for name in ("TemporaryFile", "NamedTemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, no_temp)
    first, second = _pdf(client, report_id), _pdf(client, report_id)
    monkeypatch.undo()
    assert first.status_code == second.status_code == 200
    assert first.content == second.content
    assert len(calls) == 2  # rendered afresh each time
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    stored = client.app.state.report_store._reports[uuid.UUID(report_id)]
    assert set(type(stored).model_fields) == set(AnalysisReport.model_fields)
    assert not [name for name in vars(client.app.state)["_state"] if "pdf" in name.lower()]


# --- end to end and the JSON endpoint ------------------------------------------------


@pytest.fixture
def analyze_client(make_settings, fake_groq, monkeypatch):
    import groq._base_client as base

    monkeypatch.setattr(base, "INITIAL_RETRY_DELAY", 0.001)
    monkeypatch.setattr(base, "MAX_RETRY_DELAY", 0.002)
    app = create_app(
        make_settings(groq_api_key=FAKE_API_KEY),
        groq_http_client=fake_groq.client(),
        llm_http_client=scripted(NO_ISSUES).client(),
        embedder_factory=CountingFactory(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_analyzed_report_downloads_without_raw_text_or_preview(analyze_client) -> None:
    lines = [line for line in RESUME.splitlines() if line.isascii()]
    upload = make_pdf([[*lines, f"Hobby: {CANARY}-resume"]])
    analyzed = analyze_client.post(
        "/api/v1/analyze",
        files={"file": ("resume.pdf", upload)},
        data={"job_description": f"{JOB} {CANARY}-jd", "include_preview": "true"},
    )
    assert analyzed.status_code == 200 and CANARY in analyzed.json()["preview"]
    response = _pdf(analyze_client, analyzed.json()["report_id"])
    assert response.status_code == 200
    text = _text(response.content)
    assert CANARY not in text and CANARY.encode() not in response.content
    assert label("sections", "critic") in text


def test_json_report_endpoint_is_unchanged(start_app) -> None:
    client = start_app(api_docs_enabled=True)
    report_id = _store(client)
    response = client.get(f"{REPORT}/{report_id}")
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/json"
    assert "Content-Disposition" not in response.headers
    _assert_private(response)
    report = AnalysisReport.model_validate(response.json())
    assert report.report_id == uuid.UUID(report_id)
    assert _error(client.get(f"{REPORT}/not-a-uuid")) == NOT_FOUND
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/report/{report_id}", "/api/v1/report/{report_id}/pdf"} <= set(paths)
    pdf_op = paths["/api/v1/report/{report_id}/pdf"]["get"]
    assert set(pdf_op["responses"]) == {"200", "404", "422", "429", "503", "504"}
    assert "application/pdf" in pdf_op["responses"]["200"]["content"]
    assert set(paths["/api/v1/report/{report_id}"]) == {"get"}


def test_only_get_is_allowed_on_the_pdf_route(start_app) -> None:
    client = start_app()
    report_id = _store(client)
    for method in ("post", "put", "delete", "patch"):
        assert getattr(client, method)(f"{REPORT}/{report_id}/pdf").status_code == 405
    assert client.get(f"{REPORT}/{report_id}/pdf/extra").status_code == 404
