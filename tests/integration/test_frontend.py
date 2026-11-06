"""M11 Phase 1: the browser frontend served by the real application.

These tests need only Python. They check that the existing static mount serves
frontend/ at "/", that every asset loads with the right media type and the
unchanged security headers, that the frontend calls only routes the backend
really has, that every backend error code has a user-facing message, and that
mounting the frontend leaves the API routes untouched.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.routing import APIRoute, iter_route_contexts

from backend.core.config import Settings
from backend.core.security_headers import PERMISSIONS_POLICY, build_csp
from tests.fixtures.frontend import (
    API_JS,
    FRONTEND_DIR,
    JS_DIR,
    backend_error_codes,
    error_message_codes,
    frontend_endpoints,
    object_block,
)

MEDIA_TYPES = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
}
EXPECTED_API_ROUTES = {
    ("/health", "GET"),
    ("/ready", "GET"),
    ("/api/v1/parse", "POST"),
    ("/api/v1/match", "POST"),
    ("/api/v1/analyze", "POST"),
    ("/api/v1/report/{report_id}", "GET"),
    ("/api/v1/report/{report_id}/pdf", "GET"),
    ("/api/v1/interview/questions", "POST"),
    ("/api/v1/interview/evaluate", "POST"),
}
CLIENT_ONLY_CODES = {
    "network_error",
    "client_timeout",
    "analysis_client_timeout",
    "cancelled",
    "invalid_response",
}
UNKNOWN_REPORT = "00000000-0000-4000-8000-000000000000"


@pytest.fixture
def client(start_app):
    return start_app(frontend_dir=FRONTEND_DIR)


def _api_routes(app) -> set[tuple[str, str]]:
    """(path, method) of every FastAPI route, included routers expanded as FastAPI does."""
    return {
        (context.path, method)
        for context in iter_route_contexts(app.router.routes)
        if isinstance(context.original_route, APIRoute)
        for method in context.methods or ()
    }


def _frontend_files() -> list[Path]:
    return sorted(p for p in FRONTEND_DIR.rglob("*") if p.is_file())


class _Assets(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and values.get("src"):
            self.urls.append(values["src"] or "")
        if tag == "link" and values.get("rel") in {"stylesheet", "icon"} and values.get("href"):
            self.urls.append(values["href"] or "")


# --- serving ----------------------------------------------------------------------------


def test_the_frontend_directory_is_the_default_frontend_dir() -> None:
    assert Settings(_env_file=None).frontend_dir == FRONTEND_DIR
    assert (FRONTEND_DIR / "index.html").is_file()


def test_root_serves_index_html_with_the_unchanged_security_headers(client) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert '<script type="module" src="js/app.js"></script>' in page.text
    assert page.text == (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    expected_csp = build_csp(Settings(_env_file=None).csp_frame_ancestors)
    assert page.headers["Content-Security-Policy"] == expected_csp
    assert "script-src 'self'" in expected_csp and "style-src 'self'" in expected_csp
    assert "unsafe-inline" not in expected_csp and "unsafe-eval" not in expected_csp
    assert page.headers["X-Content-Type-Options"] == "nosniff"
    assert page.headers["Referrer-Policy"] == "no-referrer"
    assert page.headers["Permissions-Policy"] == PERMISSIONS_POLICY
    assert page.headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert page.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert page.headers.get("Cache-Control") != "no-store"  # static files stay cacheable


@pytest.mark.parametrize(
    "path", [p.relative_to(FRONTEND_DIR).as_posix() for p in _frontend_files()]
)
def test_every_frontend_file_loads_with_its_media_type_and_csp(client, path: str) -> None:
    response = client.get(f"/{path}")
    assert response.status_code == 200
    media_type = MEDIA_TYPES[Path(path).suffix]
    # Module scripts are refused by browsers (with nosniff) unless served as JavaScript.
    assert response.headers["content-type"].split(";")[0] == media_type
    assert response.content == (FRONTEND_DIR / path).read_bytes()
    assert "Content-Security-Policy" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_only_html_css_and_js_files_are_shipped() -> None:
    assert {p.suffix for p in _frontend_files()} <= set(MEDIA_TYPES)


def test_every_asset_and_module_import_resolves_to_a_served_file(client) -> None:
    parser = _Assets()
    parser.feed((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))
    referenced = {url for url in parser.urls if not url.startswith("data:")}
    assert referenced == {"css/styles.css", "js/app.js"}
    for url in referenced:
        assert client.get(f"/{url}").status_code == 200
    for module in JS_DIR.glob("*.js"):
        for target in re.findall(r'from "(\./[a-z]+\.js)"', module.read_text(encoding="utf-8")):
            assert (JS_DIR / target).is_file(), f"{module.name} imports missing {target}"
            assert client.get(f"/js/{target[2:]}").status_code == 200


def test_unknown_static_paths_still_get_the_json_404_envelope(client) -> None:
    response = client.get("/js/missing.js")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# --- the API is unchanged ------------------------------------------------------------------


def test_mounting_the_frontend_leaves_the_api_route_table_unchanged(client) -> None:
    routes = _api_routes(client.app)
    assert routes == EXPECTED_API_ROUTES


def test_api_routes_keep_their_behaviour_with_the_frontend_mounted(client) -> None:
    assert client.get("/health").json() == {"status": "alive"}
    ready = client.get("/ready")
    assert ready.headers["content-type"] == "application/json"
    assert set(ready.json()["checks"]) >= {"config_valid", "embedding_model_loaded"}
    assert ready.json()["scoring"]["ats_composite"] == "not_calibrated"
    for path in ("/api/v1/does-not-exist", "/api/v1/report/not-a-uuid/extra/segments"):
        missing = client.get(path)
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "not_found"  # never index.html
    report = client.get(f"/api/v1/report/{UNKNOWN_REPORT}/pdf")
    assert report.status_code == 404
    assert report.json()["error"]["code"] == "report_not_found"
    analyze = client.post("/api/v1/analyze", json={})
    assert analyze.status_code == 415
    assert analyze.json()["error"]["code"] == "multipart_required"


# --- the frontend uses existing routes only ----------------------------------------------


def test_frontend_endpoints_are_existing_backend_routes(client) -> None:
    endpoints = frontend_endpoints()
    assert endpoints == {
        "analyze": ("POST", "/api/v1/analyze"),
        "interviewQuestions": ("POST", "/api/v1/interview/questions"),
        "interviewEvaluate": ("POST", "/api/v1/interview/evaluate"),
        "reportPdf": ("GET", "/api/v1/report/{report_id}/pdf"),
    }
    routes = _api_routes(client.app)
    for method, path in endpoints.values():
        assert (path, method) in routes


def test_api_paths_appear_only_in_the_endpoint_table() -> None:
    api_source = API_JS.read_text(encoding="utf-8")
    table = object_block(api_source, "ENDPOINTS")
    literal = re.compile(r"""["'`]/api/""")  # quoted paths only, not comments
    assert len(literal.findall(api_source)) == len(literal.findall(table)) == 4
    for path in _frontend_files():
        if path != API_JS:
            assert not literal.search(path.read_text(encoding="utf-8")), path.name
    assert len(re.findall(r"\bfetch\(", api_source)) == 1
    for module in JS_DIR.glob("*.js"):
        if module != API_JS:
            assert not re.search(
                r"\bfetch\(|XMLHttpRequest|WebSocket|EventSource",
                module.read_text(encoding="utf-8"),
            )


def test_frontend_uses_only_relative_urls() -> None:
    for path in _frontend_files():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(?:https?|wss?|ftp):", text, flags=re.IGNORECASE), path.name
        assert not re.search(r"""(?:src|href|action)\s*=\s*["']//""", text), path.name
        assert not re.search(r"""["']//[a-z0-9]""", text, flags=re.IGNORECASE), path.name
        assert "@import" not in text and "url(" not in text, path.name


# --- error messages ----------------------------------------------------------------------


def test_every_backend_error_code_has_a_user_facing_message() -> None:
    backend_codes = backend_error_codes()
    frontend_codes = error_message_codes()
    missing = backend_codes - frontend_codes
    assert not missing, f"no frontend message for: {sorted(missing)}"
    # Beyond backend codes, only the codes the browser raises itself.
    assert frontend_codes - backend_codes == CLIENT_ONLY_CODES


@pytest.mark.parametrize(
    "code",
    [
        "report_not_found",
        "upload_timeout",
        "file_too_large",
        "payload_too_large",
        "unsupported_file_type",
        "validation_error",
        "unsupported_language",
        "rate_limited",
        "analysis_capacity_exceeded",
        "upstream_error",
        "upstream_rate_limited",
        "analysis_timeout",
    ],
)
def test_audit_identified_error_codes_are_mapped(code: str) -> None:
    assert code in error_message_codes()


def test_the_expired_report_message_and_banner_explain_what_happened() -> None:
    block = object_block(API_JS.read_text(encoding="utf-8"), "ERROR_MESSAGES")
    message = re.search(r'report_not_found:\s*"([^"]+)"', block)
    assert message and "expired" in message.group(1) and "again" in message.group(1)
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    banner = re.search(r'<div id="expired-banner"[^>]*hidden>(.*?)</div>', html, flags=re.DOTALL)
    assert banner and "no longer available" in banner.group(1)
    assert 'id="expired-restart-button"' in banner.group(1)
    # The PDF download (app.js) switches to the expired state; interview.js is covered
    # behaviourally in test_frontend_behaviour.py.
    app = (JS_DIR / "app.js").read_text(encoding="utf-8")
    assert re.search(r"if \(isExpiredReport\(error\)\) \{\s*setExpired\(true\);", app)
    assert re.search(r"function setExpired\(expired\) \{[^}]*\"expired-banner\"", app)


def test_ats_composite_is_shown_as_not_calibrated_and_never_as_a_score() -> None:
    results = (JS_DIR / "results.js").read_text(encoding="utf-8")
    assert '"not_calibrated"' in results
    assert "Not calibrated. No ATS score is produced." in results
    for path in JS_DIR.glob("*.js"):
        source = path.read_text(encoding="utf-8")
        for field in (
            "calibration_version",
            "evaluation_summary",
            "ats_score.score",
            "atsScore.score",
        ):
            assert field not in source, f"{path.name} reads {field}"


def test_every_element_id_used_by_the_scripts_exists_in_index_html() -> None:
    html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(ids) == len(set(ids)), "duplicate IDs in index.html"
    used: set[str] = set()
    for module in JS_DIR.glob("*.js"):
        source = module.read_text(encoding="utf-8")
        used |= set(re.findall(r'(?:getElementById|byId)\("([a-z0-9-]+)"\)', source))
        used |= {f"view-{view}" for view in re.findall(r'"(upload|progress|results)"', source)}
    assert used, "no element lookups found"
    assert used <= set(ids), f"missing in index.html: {sorted(used - set(ids))}"
    for target in re.findall(r'aria-controls="([^"]+)"', html):
        assert target in ids
    for target in re.findall(r'(?:aria-labelledby|aria-describedby|for)="([^"]+)"', html):
        for part in target.split():
            assert part in ids, part


# --- screen-reader announcements (M11 Phase 1 audit, M-4) ---------------------------------


class _LiveRegions(HTMLParser):
    """Element stack with live-region attributes, for nesting checks."""

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[tuple[str, dict[str, str | None]]] = []
        self.ancestors: dict[str, list[dict[str, str | None]]] = {}
        self.attrs: dict[str, dict[str, str | None]] = {}
        self.status_regions: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("role") == "status":
            self.status_regions.append(values.get("id"))
        element_id = values.get("id")
        if element_id:
            self.ancestors[element_id] = [a for _, a in self.stack]
            self.attrs[element_id] = values
        if tag not in {"input", "link", "meta", "br", "img"}:
            self.stack.append((tag, values))

    def handle_endtag(self, tag: str) -> None:
        while self.stack:
            if self.stack.pop()[0] == tag:
                break


def _is_live(attrs: dict[str, str | None]) -> bool:
    return attrs.get("role") in {"status", "alert", "log"} or attrs.get("aria-live") in {
        "polite",
        "assertive",
    }


def test_frequently_changing_text_is_never_inside_a_live_region() -> None:
    parser = _LiveRegions()
    parser.feed((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))
    for element_id in ("progress-elapsed", "job-description-count"):
        assert not _is_live(parser.attrs[element_id]), element_id
        assert not any(_is_live(a) for a in parser.ancestors[element_id]), element_id
    # The counter is still read with its field, on focus.
    described = parser.attrs["job-description"].get("aria-describedby") or ""
    assert "job-description-count" in described.split()
    # Exactly one status region for short, deliberate announcements.
    assert parser.status_regions == ["app-status"]
    assert parser.attrs["app-status"].get("aria-atomic") == "true"
    assert "visually-hidden" in (parser.attrs["app-status"].get("class") or "")


# --- Node.js in CI (audit T-1) ------------------------------------------------------------


def test_ci_installs_node_and_forbids_skipping_the_node_tests() -> None:
    workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    quality = workflow.split("\n  quality:", 1)[1].split("\n  integrity:", 1)[0]
    assert re.search(r"uses: actions/setup-node@[0-9a-f]{40} # v\d", quality)  # pinned by SHA
    assert re.search(r'node-version: "\d+"', quality)
    tests_step = quality.split("- name: Tests", 1)[1]
    assert 'FRONTEND_REQUIRE_NODE: "1"' in tests_step


@pytest.mark.parametrize(("required", "skipped"), [("1", False), ("", True)])
def test_node_tests_skip_without_node_only_when_not_required(required: str, skipped: bool) -> None:
    script = (
        "from tests.fixtures.frontend import NODE, requires_node;"
        "print(NODE is None, requires_node.args[0])"
    )
    env = {**os.environ, "PATH": "", "FRONTEND_REQUIRE_NODE": required}
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned code
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["True", str(skipped)]
