"""Canary logging scaffold (M1 exit criterion).

Canary strings are injected through every input channel the API has so far:
path, query string, headers (including Authorization, Cookie and User-Agent),
request bodies, exception messages, and (M2) uploaded document content and
file names. None may appear in captured log output. Later milestones extend
this module with their endpoints.
"""

from __future__ import annotations

from fastapi import FastAPI

from tests.conftest import FAKE_API_KEY

CANARY = "CANARY-pii-5e1d"
CANARY_IP = "203.0.113.77"


def _canary_routes(app: FastAPI) -> None:
    @app.post("/api/v1/_test/echo/{item_id}")
    async def echo(item_id: int, payload: dict) -> dict:
        return {"ok": True}

    @app.get("/api/v1/_test/explode")
    async def explode() -> dict:
        raise RuntimeError(f"failed while processing {CANARY}")


CANARY_HEADERS = {
    "Authorization": f"Bearer {CANARY}",
    "Cookie": f"session={CANARY}",
    "User-Agent": f"agent/{CANARY}",
    "X-Forwarded-For": CANARY_IP,
    "Referer": f"https://example.invalid/{CANARY}",
}

CANARY_REQUESTS = [
    ("GET", f"/health?q={CANARY}", None),
    ("GET", f"/ready?email={CANARY}@example.invalid", None),
    ("GET", f"/api/v1/unknown/{CANARY}?name={CANARY}", None),
    ("POST", f"/api/v1/_test/echo/{CANARY}", {"resume_text": CANARY}),
    ("POST", "/api/v1/_test/echo/7", {"resume_text": CANARY, "answer": CANARY}),
    ("GET", "/api/v1/_test/explode", None),
]


def test_no_canary_reaches_logs(start_app, captured_logs) -> None:
    client = start_app(extra_routes=_canary_routes, groq_api_key=FAKE_API_KEY)
    for method, url, body in CANARY_REQUESTS:
        client.request(method, url, json=body, headers=CANARY_HEADERS)

    text = captured_logs.text
    assert captured_logs.events("request_completed"), "access log scaffold produced nothing"
    for secret in (CANARY, CANARY_IP, FAKE_API_KEY):
        assert secret not in text, f"{secret!r} leaked into logs"


def test_parse_uploads_never_log_content_or_filenames(start_app, captured_logs) -> None:
    """M2: document text, file names and rejected payloads stay out of logs."""
    from tests.fixtures.documents import make_docx, make_pdf

    client = start_app()
    uploads = [
        (f"{CANARY}.pdf", make_pdf([["Skills", f"Python {CANARY}"]]), {"include_preview": "true"}),
        (f"{CANARY}.docx", make_docx(["Skills", CANARY], header=CANARY), {}),
        (f"{CANARY}.pdf", b"%PDF-1.7\n" + CANARY.encode() * 50, {}),  # unreadable
        (f"{CANARY}.txt", CANARY.encode(), {}),  # unsupported type
    ]
    statuses = []
    for filename, data, fields in uploads:
        response = client.post(
            "/api/v1/parse", files={"file": (filename, data)}, data=fields, headers=CANARY_HEADERS
        )
        statuses.append(response.status_code)
    assert statuses == [200, 200, 422, 415]

    assert CANARY not in captured_logs.text
    parsed = captured_logs.events("document_parsed")
    assert {r["file_type"] for r in parsed} == {"pdf", "docx"}
    for record in parsed:
        assert set(record) <= {
            "ts",
            "level",
            "logger",
            "event",
            "request_id",
            "file_type",
            "file_size_bytes",
            "page_count",
            "char_count",
        }
    rejected = captured_logs.events("app_error")
    assert {r["error_code"] for r in rejected} == {"unreadable_document", "unsupported_file_type"}


def test_access_log_uses_route_templates_not_raw_paths(start_app, captured_logs) -> None:
    client = start_app(extra_routes=_canary_routes)
    client.post(f"/api/v1/_test/echo/{CANARY}", json={})
    client.post("/api/v1/_test/echo/42", json={"a": 1})
    client.get(f"/api/v1/unknown/{CANARY}")
    client.get("/health?x=1")

    routes = [
        (r["method"], r["route"], r["status"]) for r in captured_logs.events("request_completed")
    ]
    assert ("POST", "/api/v1/_test/echo/{item_id}", 422) in routes
    assert ("POST", "/api/v1/_test/echo/{item_id}", 200) in routes
    assert ("GET", "<unmatched>", 404) in routes
    assert ("GET", "/health", 200) in routes
    for record in captured_logs.events("request_completed"):
        assert set(record) <= {
            "ts",
            "level",
            "logger",
            "event",
            "request_id",
            "method",
            "route",
            "status",
            "duration_ms",
        }
        assert isinstance(record["duration_ms"], float)


def test_access_log_carries_request_id(start_app, captured_logs) -> None:
    client = start_app()
    response = client.get("/health", headers={"X-Request-ID": "trace-abcdef12"})
    (record,) = captured_logs.events("request_completed")
    assert record["request_id"] == response.headers["X-Request-ID"] == "trace-abcdef12"


def test_unhandled_exception_logged_without_message(start_app, captured_logs) -> None:
    client = start_app(extra_routes=_canary_routes)
    response = client.get("/api/v1/_test/explode")
    assert response.status_code == 500
    (record,) = captured_logs.events("unhandled_exception")
    assert record["exc_type"] == "RuntimeError"
    assert record["error_code"] == "internal_error"
    assert record["request_id"] == response.headers["X-Request-ID"]
    assert CANARY not in captured_logs.text


def test_unknown_http_methods_are_normalised(start_app, captured_logs) -> None:
    start_app().request("PROPFIND", "/health")
    (record,) = captured_logs.events("request_completed")
    assert record["method"] == "OTHER"
