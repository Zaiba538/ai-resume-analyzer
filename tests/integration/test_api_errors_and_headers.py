"""Error envelope, request IDs, security headers, docs and static mount."""

from __future__ import annotations

import re

from fastapi import FastAPI
from pydantic import BaseModel, Field

from backend.core.errors import AppError

UUID_HEX = re.compile(r"[0-9a-f]{32}")
CANARY = "CANARY-envelope-91c2"


class _Payload(BaseModel):
    name: str = Field(max_length=5)
    count: int


def _test_routes(app: FastAPI) -> None:
    @app.post("/api/v1/_test/validate")
    async def validate(payload: _Payload) -> dict:
        return {"ok": True}

    @app.get("/api/v1/_test/boom")
    async def boom() -> dict:
        raise RuntimeError(f"database exploded with {CANARY}")

    @app.get("/api/v1/_test/app-error")
    async def app_error() -> dict:
        raise AppError(
            "The upstream model timed out.",
            code="upstream_timeout",
            status_code=504,
            headers={"Retry-After": "5"},
        )


def _assert_envelope(response, status: int, code: str) -> dict:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["code"] == code
    assert isinstance(error["message"], str) and error["message"]
    assert error["request_id"] == response.headers["X-Request-ID"]
    return error


def test_unknown_route_404_envelope(start_app) -> None:
    response = start_app().get(f"/api/v1/does-not-exist/{CANARY}")
    error = _assert_envelope(response, 404, "not_found")
    assert error["message"] == "Not Found"
    assert CANARY not in response.text


def test_method_not_allowed_envelope_keeps_allow_header(start_app) -> None:
    response = start_app().post("/health")
    _assert_envelope(response, 405, "method_not_allowed")
    assert "GET" in response.headers["Allow"]


def test_validation_error_never_echoes_input(start_app) -> None:
    client = start_app(extra_routes=_test_routes)
    response = client.post("/api/v1/_test/validate", json={"name": CANARY, "count": CANARY})
    error = _assert_envelope(response, 422, "validation_error")
    assert CANARY not in response.text
    assert {tuple(d["loc"]) for d in error["details"]} == {("body", "name"), ("body", "count")}
    for detail in error["details"]:
        assert set(detail) == {"loc", "type", "msg"}


def test_malformed_json_is_validation_error(start_app) -> None:
    client = start_app(extra_routes=_test_routes)
    response = client.post(
        "/api/v1/_test/validate",
        content=b'{"name": "' + CANARY.encode() + b'",',
        headers={"Content-Type": "application/json"},
    )
    _assert_envelope(response, 422, "validation_error")
    assert CANARY not in response.text


def test_unhandled_exception_is_generic_500_with_headers(start_app) -> None:
    client = start_app(extra_routes=_test_routes)
    response = client.get("/api/v1/_test/boom")
    error = _assert_envelope(response, 500, "internal_error")
    assert error["message"] == "An internal error occurred."
    assert CANARY not in response.text
    assert "Content-Security-Policy" in response.headers
    assert response.headers["Cache-Control"] == "no-store"


def test_app_error_uses_its_code_message_and_headers(start_app) -> None:
    client = start_app(extra_routes=_test_routes)
    response = client.get("/api/v1/_test/app-error")
    error = _assert_envelope(response, 504, "upstream_timeout")
    assert error["message"] == "The upstream model timed out."
    assert response.headers["Retry-After"] == "5"


def test_request_id_generated_when_absent(start_app) -> None:
    response = start_app().get("/health")
    assert UUID_HEX.fullmatch(response.headers["X-Request-ID"])


def test_request_id_echoed_when_well_formed(start_app) -> None:
    response = start_app().get("/health", headers={"X-Request-ID": "client-trace-0001"})
    assert response.headers["X-Request-ID"] == "client-trace-0001"


def test_malformed_request_id_replaced(start_app) -> None:
    response = start_app().get("/health", headers={"X-Request-ID": "bad id <script>"})
    assert UUID_HEX.fullmatch(response.headers["X-Request-ID"])


def test_each_request_gets_distinct_id(start_app) -> None:
    client = start_app()
    ids = {client.get("/health").headers["X-Request-ID"] for _ in range(10)}
    assert len(ids) == 10


def test_security_headers_on_every_response(start_app) -> None:
    client = start_app()
    for response in (client.get("/health"), client.get("/ready"), client.get("/nope")):
        headers = response.headers
        csp = headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp and "script-src 'self'" in csp
        assert "object-src 'none'" in csp
        assert "frame-ancestors 'self' https://huggingface.co" in csp
        assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["Referrer-Policy"] == "no-referrer"
        assert "camera=()" in headers["Permissions-Policy"]
        assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
        assert "Strict-Transport-Security" not in headers


def test_api_and_health_responses_not_cached_or_indexed(start_app) -> None:
    client = start_app()
    for path in ("/health", "/ready", "/api/v1/anything"):
        response = client.get(path)
        assert response.headers["Cache-Control"] == "no-store", path
        assert response.headers["X-Robots-Tag"] == "noindex, nofollow", path


def test_hsts_when_enabled(start_app) -> None:
    response = start_app(hsts_enabled=True).get("/health")
    assert response.headers["Strict-Transport-Security"].startswith("max-age=31536000")


def test_api_docs_disabled_by_default(start_app) -> None:
    client = start_app()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404, path


def test_api_docs_when_enabled_are_csp_exempt(start_app) -> None:
    client = start_app(api_docs_enabled=True)
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "Content-Security-Policy" not in docs.headers
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "Content-Security-Policy" in schema.headers
    assert {"/health", "/ready"} <= set(schema.json()["paths"])


def test_static_frontend_mounted_with_csp(start_app, tmp_path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    client = start_app(frontend_dir=frontend)
    page = client.get("/")
    assert page.status_code == 200
    assert "<title>t</title>" in page.text
    assert "Content-Security-Policy" in page.headers
    assert "Cache-Control" not in page.headers or page.headers["Cache-Control"] != "no-store"
    # API routes still take precedence over the catch-all mount.
    assert client.get("/health").json() == {"status": "alive"}
    _assert_envelope(client.get("/missing.js"), 404, "not_found")


def test_missing_frontend_dir_is_tolerated(start_app) -> None:
    client = start_app()  # frontend_dir points at a non-existent path
    _assert_envelope(client.get("/"), 404, "not_found")
