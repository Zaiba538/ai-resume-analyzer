"""M10 Phase 2: rate limits on the real routes (dependency wiring, 429s, exemptions)."""

from __future__ import annotations

import threading

import pytest
from fastapi import APIRouter

from backend.api import rate_limits
from backend.api.rate_limits import ANALYSIS_BUSY_MESSAGE, RATE_LIMITED_MESSAGE
from backend.core import rate_limit
from backend.core.rate_limit import RateLimiter
from backend.main import create_app
from tests.fixtures.documents import make_docx
from tests.fixtures.embeddings import CountingFactory
from tests.fixtures.rate_limit import (
    DEFAULT_MINUTE_LIMITS,
    ENDPOINTS,
    REPORT_ID,
    harness_factory,
    rate_limit_headers,
)
from tests.unit.test_critic_checks import JOB, RESUME


@pytest.fixture
def harness(make_settings, fake_groq):
    yield from harness_factory(make_settings, fake_groq)


ALL_ONE_PER_MINUTE = {
    "rate_limit_analyze": "1/minute",
    "rate_limit_analyze_global": "500/minute",
    "rate_limit_interview_questions": "1/minute",
    "rate_limit_interview_evaluate": "1/minute",
    "rate_limit_interview_global": "500/minute",
    "rate_limit_match": "1/minute",
    "rate_limit_parse": "1/minute",
    "rate_limit_report": "1/minute",
    "rate_limit_report_pdf": "1/minute",
}


def assert_rate_limited(response, retry_after: int | None = None) -> None:
    assert response.status_code == 429
    assert response.json() == {
        "error": {
            "code": "rate_limited",
            "message": RATE_LIMITED_MESSAGE,
            "request_id": response.headers["X-Request-ID"],
        }
    }
    assert response.headers["Retry-After"].isdigit()
    assert int(response.headers["Retry-After"]) >= 1
    if retry_after is not None:
        assert response.headers["Retry-After"] == str(retry_after)
    assert response.headers["Content-Type"] == "application/json"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in response.headers


# --- per-client limits --------------------------------------------------------------------------


@pytest.mark.parametrize("policy", sorted(ENDPOINTS))
def test_each_endpoint_enforces_its_default_per_client_limit(harness, policy: str) -> None:
    h = harness()
    for i in range(DEFAULT_MINUTE_LIMITS[policy]):
        assert h.call(policy, ip="198.51.100.1").status_code != 429, (policy, i)
    assert_rate_limited(h.call(policy, ip="198.51.100.1"))
    assert h.call(policy, ip="198.51.100.99").status_code != 429  # another client


def test_limits_are_per_endpoint(harness) -> None:
    h = harness(**ALL_ONE_PER_MINUTE)
    for policy in ENDPOINTS:
        assert h.call(policy).status_code != 429, policy
    for policy in ENDPOINTS:
        assert_rate_limited(h.call(policy))


def test_retry_after_follows_the_moving_window(harness) -> None:
    h = harness(rate_limit_report="2/minute")
    clock = h.use_fake_clock()
    assert h.call(rate_limit.REPORT).status_code == 404
    clock.advance(20)
    assert h.call(rate_limit.REPORT).status_code == 404
    assert_rate_limited(h.call(rate_limit.REPORT), retry_after=40)
    clock.advance(39.5)
    assert_rate_limited(h.call(rate_limit.REPORT), retry_after=1)
    clock.advance(0.5)  # the first request leaves the window
    assert h.call(rate_limit.REPORT).status_code == 404
    assert_rate_limited(h.call(rate_limit.REPORT), retry_after=20)


def test_hour_limit_applies_across_minute_windows(harness) -> None:
    h = harness()  # analyze: 5/minute;30/hour, global 20/minute
    clock = h.use_fake_clock()
    for minute in range(6):
        for _ in range(5):
            assert h.call(rate_limit.ANALYZE).status_code != 429
        # Minutes 1-5: the minute window is full. Minute 6: the hour window is too,
        # and its oldest entry (t=0) leaves it in 3600 - 300 seconds.
        assert_rate_limited(h.call(rate_limit.ANALYZE), retry_after=60 if minute < 5 else 3300)
        clock.advance(60)
    # 30 requests in the last hour (the first at t=0); now at t=360.
    assert_rate_limited(h.call(rate_limit.ANALYZE), retry_after=3240)


def test_successful_responses_carry_no_rate_limit_headers(harness) -> None:
    h = harness()
    responses = [
        h.request("POST", "/api/v1/parse", files={"file": ("cv.docx", make_docx())}),
        h.request("POST", "/api/v1/match", json={"resume_text": RESUME, "job_description": JOB}),
        h.call(rate_limit.REPORT),
        h.call(rate_limit.INTERVIEW_QUESTIONS),
    ]
    assert [r.status_code for r in responses[:2]] == [200, 200]
    for response in responses:
        assert response.status_code != 429
        assert rate_limit_headers(response) == []


# --- global limits --------------------------------------------------------------------------------


def test_global_analysis_limit_spans_client_addresses(harness) -> None:
    h = harness()  # 20/minute across all clients
    for i in range(20):
        assert h.call(rate_limit.ANALYZE, ip=f"192.0.2.{i + 1}").status_code != 429
    assert_rate_limited(h.call(rate_limit.ANALYZE, ip="203.0.113.1"), retry_after=60)
    assert h.call(rate_limit.PARSE, ip="203.0.113.1").status_code != 429  # other policy


def test_interview_endpoints_share_one_global_pool(harness) -> None:
    h = harness(rate_limit_interview_global="4/minute")
    order = [rate_limit.INTERVIEW_QUESTIONS, rate_limit.INTERVIEW_EVALUATE] * 2
    for i, policy in enumerate(order):
        assert h.call(policy, ip=f"192.0.2.{i + 1}").status_code != 429
    assert_rate_limited(h.call(rate_limit.INTERVIEW_QUESTIONS, ip="203.0.113.1"))
    assert_rate_limited(h.call(rate_limit.INTERVIEW_EVALUATE, ip="203.0.113.2"))
    assert h.call(rate_limit.ANALYZE, ip="203.0.113.3").status_code != 429


def test_default_interview_pool_is_sixty_per_minute(harness) -> None:
    h = harness()
    for i in range(60):
        policy = (rate_limit.INTERVIEW_QUESTIONS, rate_limit.INTERVIEW_EVALUATE)[i % 2]
        assert h.call(policy, ip=f"192.0.2.{i + 1}").status_code != 429
    assert_rate_limited(h.call(rate_limit.INTERVIEW_EVALUATE, ip="203.0.113.9"))


# --- ordering: before the body and the handler -------------------------------------------------


def test_limited_requests_never_read_the_body_or_reach_the_handler(harness, monkeypatch) -> None:
    h = harness(**ALL_ONE_PER_MINUTE)
    calls: list[str] = []
    import backend.api.routes.analyze as analyze_route
    import backend.api.routes.interview as interview_route
    import backend.api.routes.match as match_route
    import backend.api.routes.parse as parse_route

    for module, name in (
        (parse_route, "read_multipart_upload"),
        (match_route, "read_multipart_upload"),
        (analyze_route, "read_multipart_upload"),
        (interview_route, "read_json_body"),
    ):
        real = getattr(module, name)

        async def spy(*args, _real=real, _name=f"{module.__name__}.{name}", **kwargs):
            calls.append(_name)
            return await _real(*args, **kwargs)

        monkeypatch.setattr(module, name, spy)
    store = h.app.state.report_store
    real_get = store.get

    async def counting_get(report_id: str):
        calls.append("report_store.get")
        return await real_get(report_id)

    monkeypatch.setattr(store, "get", counting_get)
    embeddings = h.app.state.embeddings
    real_embedder = embeddings.get
    monkeypatch.setattr(
        embeddings, "get", lambda: calls.append("embeddings.get") or real_embedder()
    )

    for policy in ENDPOINTS:  # use up each endpoint's single request
        h.call(policy)
    used = len(calls)
    assert "report_store.get" in calls and "embeddings.get" in calls  # the spies work
    reads = dict(h.wrapper.body_reads)
    assert reads["/api/v1/interview/questions"] >= 1  # and so does the read counter
    reads_before = len(h.wrapper.body_reads)
    invalid = {
        "content": b"\x00not json, not multipart" * 1000,
        "headers": {"Content-Type": "application/json"},
    }
    for policy in ENDPOINTS:
        method, url, _ = ENDPOINTS[policy]
        response = h.request(method, url, **invalid) if method == "POST" else h.request(method, url)
        assert_rate_limited(response)
    assert len(calls) == used  # no parser, store lookup or model access
    assert [count for _, count in h.wrapper.body_reads[reads_before:]] == [0] * len(ENDPOINTS)


def test_limited_upload_is_rejected_before_size_and_type_checks_run(harness) -> None:
    h = harness(rate_limit_parse="1/minute")
    assert h.call(rate_limit.PARSE).status_code == 415  # not multipart
    response = h.request("POST", "/api/v1/parse", files={"file": ("cv.exe", b"MZ" * 1000)})
    assert_rate_limited(response)


# --- concurrency -------------------------------------------------------------------------------


def _parallel(h, jobs: list[tuple[str, str]]) -> list[int]:
    barrier = threading.Barrier(len(jobs))
    statuses: list[int] = [0] * len(jobs)

    def run(i: int) -> None:
        policy, ip = jobs[i]
        barrier.wait()
        statuses[i] = h.call(policy, ip=ip).status_code

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads)
    return statuses


@pytest.mark.parametrize("round_", range(3))
def test_concurrent_requests_from_one_client_admit_exactly_the_limit(harness, round_: int) -> None:
    h = harness(rate_limit_report="10/minute")
    statuses = _parallel(h, [(rate_limit.REPORT, "192.0.2.7")] * 40)
    assert statuses.count(429) == 30
    assert statuses.count(404) == 10


@pytest.mark.parametrize("round_", range(3))
def test_concurrent_requests_share_the_interview_pool_exactly(harness, round_: int) -> None:
    h = harness(rate_limit_interview_global="7/minute")
    jobs = [
        ((rate_limit.INTERVIEW_QUESTIONS, rate_limit.INTERVIEW_EVALUATE)[i % 2], f"192.0.2.{i + 1}")
        for i in range(40)
    ]
    statuses = _parallel(h, jobs)
    assert len(statuses) - statuses.count(429) == 7


# --- exemptions and disabled enforcement -----------------------------------------------------


def test_health_readiness_docs_static_and_unknown_paths_are_exempt(harness, tmp_path) -> None:
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    h = harness(api_docs_enabled=True, frontend_dir=frontend, **ALL_ONE_PER_MINUTE)
    off = harness(api_docs_enabled=True, frontend_dir=frontend, rate_limit_enabled=False)
    paths = [
        ("GET", "/health"), ("GET", "/ready"), ("GET", "/docs"), ("GET", "/redoc"),
        ("GET", "/openapi.json"), ("GET", "/"), ("GET", "/index.html"),
        ("GET", "/api/v1/unknown"), ("POST", "/api/v1/report"), ("GET", "/api/v1/parse"),
        ("HEAD", "/health"),
    ]  # fmt: skip
    expected = {(m, p): off.request(m, p).status_code for m, p in paths}
    assert expected[("GET", "/health")] == expected[("GET", "/docs")] == 200
    assert expected[("GET", "/")] == expected[("GET", "/openapi.json")] == 200
    assert expected[("GET", "/api/v1/unknown")] == 404
    for method, path in paths:
        for _ in range(5):  # exactly as without rate limiting: never 429, never 500
            assert h.request(method, path).status_code == expected[(method, path)], (method, path)
    assert h.limiter.tracked_keys() == ()


def test_disabled_quota_enforcement_keeps_endpoint_behaviour(harness) -> None:
    enabled = harness(rate_limit_enabled=True)
    disabled = harness(rate_limit_enabled=False, **ALL_ONE_PER_MINUTE)
    expected = {policy: enabled.call(policy).status_code for policy in ENDPOINTS}
    assert 429 not in expected.values()
    for _ in range(5):
        for policy in ENDPOINTS:
            response = disabled.call(policy)
            assert response.status_code == expected[policy], policy
            assert rate_limit_headers(response) == []
    assert disabled.limiter.tracked_keys() == ()
    assert (
        disabled.request(
            "POST", "/api/v1/match", json={"resume_text": RESUME, "job_description": JOB}
        ).status_code
        == 200
    )


# --- application wiring ----------------------------------------------------------------------


def test_each_app_has_one_limiter_and_one_gate(harness) -> None:
    first, second = harness(rate_limit_report="1/minute"), harness(rate_limit_report="1/minute")
    limiter, gate = first.app.state.rate_limiter, first.app.state.analyze_gate
    assert isinstance(limiter, RateLimiter)
    assert limiter is not second.app.state.rate_limiter
    assert gate is not second.app.state.analyze_gate
    assert gate.max_concurrent == 2
    for policy in ENDPOINTS:
        first.call(policy)
    assert first.app.state.rate_limiter is limiter and first.app.state.analyze_gate is gate
    assert_rate_limited(first.call(rate_limit.REPORT))
    assert second.call(rate_limit.REPORT).status_code == 404  # separate state


def test_startup_fails_when_a_route_has_no_policy(make_settings, monkeypatch) -> None:
    monkeypatch.delitem(rate_limits.ROUTE_POLICIES, "/api/v1/match")
    with pytest.raises(RuntimeError, match="an API route has no rate-limit policy"):
        create_app(make_settings(), embedder_factory=CountingFactory())


def test_startup_fails_when_a_policy_names_no_route(make_settings, monkeypatch) -> None:
    monkeypatch.setitem(rate_limits.ROUTE_POLICIES, "/api/v1/gone", rate_limit.REPORT)
    with pytest.raises(RuntimeError, match="names an unknown route"):
        create_app(make_settings(), embedder_factory=CountingFactory())


def test_startup_fails_when_a_new_route_is_added_without_a_policy(
    make_settings, monkeypatch
) -> None:
    import backend.api.routes.report as report_module

    # The report router gains an endpoint nobody added a policy for; create_app
    # registers it as usual and must refuse to start (no list of routers to update).
    grown = APIRouter()
    grown.include_router(report_module.router)

    @grown.get("/api/v1/report-export")
    async def new_thing() -> dict:
        return {}

    monkeypatch.setattr(report_module, "router", grown)
    with pytest.raises(RuntimeError, match="an API route has no rate-limit policy"):
        create_app(make_settings(), embedder_factory=CountingFactory())


def test_route_policies_cover_exactly_the_api_endpoints() -> None:
    assert {path: policy for path, policy in rate_limits.ROUTE_POLICIES.items()} == {
        "/api/v1/parse": rate_limit.PARSE,
        "/api/v1/match": rate_limit.MATCH,
        "/api/v1/analyze": rate_limit.ANALYZE,
        "/api/v1/report/{report_id}": rate_limit.REPORT,
        "/api/v1/report/{report_id}/pdf": rate_limit.REPORT_PDF,
        "/api/v1/interview/questions": rate_limit.INTERVIEW_QUESTIONS,
        "/api/v1/interview/evaluate": rate_limit.INTERVIEW_EVALUATE,
    }


def test_openapi_documents_429_on_rate_limited_operations_only(harness) -> None:
    spec = harness(api_docs_enabled=True).client.get("/openapi.json").json()
    ref = {"$ref": "#/components/schemas/ErrorEnvelope"}
    paths = {policy: path for path, policy in rate_limits.ROUTE_POLICIES.items()}
    for policy, (method, _, _) in ENDPOINTS.items():
        response = spec["paths"][paths[policy]][method.lower()]["responses"]["429"]
        assert response["content"]["application/json"]["schema"] == ref, policy
        if policy == rate_limit.ANALYZE:
            assert "analysis_capacity_exceeded" in response["description"]
        else:
            assert response["description"] == rate_limits.RATE_LIMIT_RESPONSE_DESCRIPTION
    assert "429" not in spec["paths"]["/health"]["get"]["responses"]
    assert "429" not in spec["paths"]["/ready"]["get"]["responses"]


def test_busy_and_rate_limited_messages_are_fixed(harness) -> None:
    assert RATE_LIMITED_MESSAGE == "Too many requests. Please try again later."
    assert (
        ANALYSIS_BUSY_MESSAGE == "The server is busy with other analyses. Please try again shortly."
    )
    # A real 429 for a request that names a report never echoes the report ID, the
    # path or the client address, in the body or in any header.
    h = harness(rate_limit_report="1/minute")
    client_ip = "198.51.100.42"
    assert h.call(rate_limit.REPORT, ip=client_ip).status_code == 404
    limited = h.call(rate_limit.REPORT, ip=client_ip)
    assert_rate_limited(limited)
    exposed = limited.text + "\n".join(f"{k}: {v}" for k, v in limited.headers.items())
    for value in (REPORT_ID, client_ip, "/api/v1/report"):
        assert value not in exposed


# --- a full limiter store (fail closed) -----------------------------------------------------


def test_a_full_store_rejects_new_clients_with_the_429_envelope(harness, captured_logs) -> None:
    """Every key slot taken: a new client gets 429 before any work; known clients go on."""
    h = harness(rate_limit_max_tracked_keys=100)  # the configurable minimum
    clock = h.use_fake_clock()
    known = [f"198.51.100.{i}" for i in range(1, 101)]
    for ip in known:  # report: 60/minute, so one key per client
        assert h.call(rate_limit.REPORT, ip=ip).status_code == 404
    assert h.limiter.tracked_client_keys() == 100

    clock.advance(15)
    newcomer = "203.0.113.200"
    # Retry-After: the earliest key expires 60 s after it was created, 45 s from now.
    assert_rate_limited(h.call(rate_limit.REPORT, ip=newcomer), retry_after=45)
    reads = len(h.wrapper.body_reads)
    upload = h.request(
        "POST", "/api/v1/parse", ip=newcomer, files={"file": ("cv.docx", make_docx(["x"]))}
    )
    assert_rate_limited(upload, retry_after=45)
    assert h.wrapper.body_reads[reads] == ("/api/v1/parse", 0)  # rejected before the body
    assert h.limiter.tracked_client_keys() == 100  # nothing created, nothing evicted
    assert h.call(rate_limit.REPORT, ip=known[0]).status_code == 404  # known clients served

    clock.advance(45)  # the keys created at the start have expired
    assert h.call(rate_limit.REPORT, ip=newcomer).status_code == 404  # recovered
    assert len(captured_logs.events("rate_limit_capacity_exhausted")) == 1
    assert newcomer not in captured_logs.text and known[0] not in captured_logs.text


# --- HEAD and OPTIONS on limited routes -----------------------------------------------------


@pytest.mark.parametrize("method", ["HEAD", "OPTIONS"])
@pytest.mark.parametrize("policy", sorted(ENDPOINTS))
def test_head_and_options_on_limited_routes_are_405_without_work(
    harness, policy: str, method: str
) -> None:
    h = harness(**ALL_ONE_PER_MINUTE)
    route_method, url, _ = ENDPOINTS[policy]
    reads = len(h.wrapper.body_reads)
    for _ in range(3):
        response = h.request(method, url)
        assert response.status_code == 405
        assert response.headers["Allow"] == route_method
        assert "X-Request-ID" in response.headers and "Retry-After" not in response.headers
        if method == "OPTIONS":
            assert response.json()["error"]["code"] == "method_not_allowed"
    # Routing answers 405 before any dependency runs: the rate limit was not consulted
    # (no key, so no quota used) and the handler, which runs after it, never ran.
    assert h.limiter.tracked_client_keys() == 0
    assert all(count == 0 for _, count in h.wrapper.body_reads[reads:])
    assert h.app.state.analyze_gate.in_flight == 0
    assert h.call(policy).status_code != 429  # the one allowed request is still available
    assert_rate_limited(h.call(policy))
