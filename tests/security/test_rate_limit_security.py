"""M10 Phase 2 security: client identity, proxy headers, privacy and error separation."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from backend.core import rate_limit
from backend.main import create_app
from tests.conftest import FAKE_API_KEY
from tests.fixtures.embeddings import CountingFactory
from tests.fixtures.groq import Reply
from tests.fixtures.interview import InterviewFakeGroq
from tests.fixtures.rate_limit import ENDPOINTS, ClientAddress, harness_factory
from tests.fixtures.reports import synthetic_report
from tests.integration.test_api_rate_limit import assert_rate_limited

CANARY_V4 = "203.0.113.77"
CANARY_V6 = "2001:db8:77:1::abcd"
CANARY = "CANARYm10rl7f3"
SPOOF_HEADERS = ("X-Forwarded-For", "X-Real-IP", "Forwarded", "True-Client-IP", "CF-Connecting-IP")


@pytest.fixture
def harness(make_settings, fake_groq):
    yield from harness_factory(make_settings, fake_groq)


@contextmanager
def _backend_log_level(level: int) -> Iterator[None]:
    """Set the ``backend`` logger's level for the block, then restore it, however it ends."""
    logger = logging.getLogger("backend")
    original = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original)


@pytest.fixture
def backend_info_logs() -> Iterator[None]:
    """INFO records from ``backend`` (request_completed, app_error) for one test only."""
    with _backend_log_level(logging.INFO):
        yield


# --- identity -------------------------------------------------------------------------------


@pytest.mark.parametrize("header", SPOOF_HEADERS)
def test_forwarding_headers_cannot_change_the_client_identity(harness, header: str) -> None:
    h = harness(rate_limit_report="1/minute")
    value = "for=198.51.100.1" if header == "Forwarded" else "198.51.100.1"
    assert h.call(rate_limit.REPORT, ip="192.0.2.5", headers={header: value}).status_code == 404
    other = "for=198.51.100.2" if header == "Forwarded" else "198.51.100.2"
    assert_rate_limited(h.call(rate_limit.REPORT, ip="192.0.2.5", headers={header: other}))
    assert_rate_limited(h.call(rate_limit.REPORT, ip="192.0.2.5"))


@pytest.mark.parametrize("header", SPOOF_HEADERS)
def test_the_application_never_reads_forwarding_headers(
    harness, header, captured_logs, backend_info_logs
) -> None:
    """On every limited endpoint, the limiter's keys come from the ASGI peer only.

    Forged values, including a loopback and an IPv6 address, never select a key and
    never reach the logs, whatever the endpoint does with the request afterwards.
    """
    h = harness()
    peer = "192.0.2.5"
    forged = ("198.51.100.1", "127.0.0.1", "2001:db8:5::1", CANARY_V4)
    for policy in sorted(ENDPOINTS):
        for value in forged:
            sent = f'for="{value}"' if header == "Forwarded" else value
            assert h.call(policy, ip=peer, headers={header: sent}).status_code != 429
    keys = h.limiter.tracked_keys()
    owners = {key.rsplit(":", 1)[1] for key in keys if key.startswith("client:")}
    assert owners == {h.limiter.client_key(peer)}
    assert CANARY_V4 not in captured_logs.text


def _behind_uvicorn_proxy_headers(
    make_settings, fake_groq, **overrides
) -> tuple[TestClient, ClientAddress]:
    app = create_app(
        make_settings(rate_limit_enabled=True, **overrides),
        groq_http_client=fake_groq.client(),
        embedder_factory=CountingFactory(),
    )
    # The real uvicorn middleware with its default trust (loopback only), in front
    # of the application, exactly as ``uvicorn --factory`` installs it.
    wrapper = ClientAddress(ProxyHeadersMiddleware(app, trusted_hosts="127.0.0.1"))
    return TestClient(wrapper, raise_server_exceptions=False), wrapper


def test_forwarded_for_from_an_untrusted_peer_is_ignored(make_settings, fake_groq) -> None:
    client, _ = _behind_uvicorn_proxy_headers(
        make_settings, fake_groq, rate_limit_report="1/minute"
    )
    url = "/api/v1/report/00000000-0000-4000-8000-000000000000"
    with client:
        peer = {"x-test-client-ip": "203.0.113.5"}
        assert (
            client.get(url, headers={**peer, "X-Forwarded-For": "198.51.100.1"}).status_code == 404
        )
        assert_rate_limited(client.get(url, headers={**peer, "X-Forwarded-For": "198.51.100.2"}))


def test_trusted_proxy_forwarding_cannot_be_spoofed_by_prepending(make_settings, fake_groq) -> None:
    client, _ = _behind_uvicorn_proxy_headers(
        make_settings, fake_groq, rate_limit_report="1/minute"
    )
    url = "/api/v1/report/00000000-0000-4000-8000-000000000000"
    with client:
        proxy = {"x-test-client-ip": "127.0.0.1"}
        assert (
            client.get(url, headers={**proxy, "X-Forwarded-For": "198.51.100.1"}).status_code == 404
        )
        # Distinct real clients behind the trusted proxy get distinct buckets...
        assert (
            client.get(url, headers={**proxy, "X-Forwarded-For": "198.51.100.2"}).status_code == 404
        )
        # ...and a client cannot pick a fresh bucket by prepending a fake address:
        # uvicorn uses the right-most address the proxy appended.
        spoofed = {**proxy, "X-Forwarded-For": "192.0.2.250, 198.51.100.1"}
        assert_rate_limited(client.get(url, headers=spoofed))


# --- privacy --------------------------------------------------------------------------------


def test_client_addresses_never_reach_logs_responses_or_storage(harness, captured_logs) -> None:
    h = harness(rate_limit_report="1/minute", rate_limit_max_tracked_keys=100)
    bodies: list[str] = []
    for ip in (CANARY_V4, CANARY_V6):
        for _ in range(3):
            response = h.call(rate_limit.REPORT, ip=ip, headers={"X-Forwarded-For": CANARY_V4})
            bodies.append(response.text + str(dict(response.headers)))
    for i in range(110):  # fill the limiter to exercise the capacity warning too
        h.call(rate_limit.REPORT, ip=f"198.51.100.{i % 250}" if i < 98 else f"192.0.2.{i}")
    assert captured_logs.events("rate_limit_capacity_exhausted")
    logged = captured_logs.text
    stored = " ".join(h.limiter.tracked_keys())
    for secret in (CANARY_V4, CANARY_V6, "2001:db8:77:1", "198.51.100.", "192.0.2."):
        assert secret not in logged
        assert secret not in stored
        assert all(secret not in body for body in bodies)


def test_429s_are_logged_only_as_an_error_code(harness, captured_logs, backend_info_logs) -> None:
    h = harness(rate_limit_report="1/minute")
    h.call(rate_limit.REPORT, ip=CANARY_V4)
    assert_rate_limited(h.call(rate_limit.REPORT, ip=CANARY_V4, headers={"User-Agent": CANARY}))
    (event,) = [e for e in captured_logs.events("app_error") if e.get("status") == 429]
    assert event["error_code"] == "rate_limited"
    assert set(event) <= {"ts", "level", "logger", "event", "request_id", "error_code", "status"}
    (access,) = [e for e in captured_logs.events("request_completed") if e.get("status") == 429]
    assert access["route"] == "/api/v1/report/{report_id}"  # template, not the raw path
    assert CANARY not in captured_logs.text and CANARY_V4 not in captured_logs.text


# --- separation from upstream errors -----------------------------------------------------------


def test_upstream_503_and_its_retry_after_are_unchanged(make_settings, fake_groq) -> None:
    fake = InterviewFakeGroq().script(
        "interview", *(Reply(status=429, headers={"retry-after": "7"}) for _ in range(3))
    )
    app = create_app(
        make_settings(rate_limit_enabled=True, groq_api_key=FAKE_API_KEY, llm_max_retries=0),
        groq_http_client=fake_groq.client(),
        llm_http_client=fake.client(),
        embedder_factory=CountingFactory(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        store = app.state.report_store
        report_id = str(client.portal.call(store.create, synthetic_report).report_id)
        response = client.post("/api/v1/interview/questions", json={"report_id": report_id})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "upstream_rate_limited"
    assert response.headers["Retry-After"] == "7"  # Groq's value, passed through as before


def test_local_rate_limit_is_distinct_from_upstream_rate_limiting(harness) -> None:
    h = harness(rate_limit_interview_questions="1/minute")
    h.call(rate_limit.INTERVIEW_QUESTIONS)
    response = h.call(rate_limit.INTERVIEW_QUESTIONS)
    assert_rate_limited(response)
    assert response.json()["error"]["code"] != "upstream_rate_limited"


# --- log-level isolation (audit finding L-2) ----------------------------------------------------


@pytest.mark.parametrize("original", [logging.NOTSET, logging.ERROR])
def test_backend_log_level_is_restored_even_when_an_assertion_fails(monkeypatch, original) -> None:
    logger = logging.getLogger("backend")
    monkeypatch.setattr(logger, "level", original)  # restored by monkeypatch afterwards
    with (
        pytest.raises(AssertionError, match="simulated failure"),
        _backend_log_level(logging.INFO),
    ):
        assert logger.level == logging.INFO
        raise AssertionError("simulated failure")
    assert logger.level == original


def test_backend_info_logs_fixture_sets_info_for_the_test(backend_info_logs) -> None:
    assert logging.getLogger("backend").level == logging.INFO


@pytest.mark.parametrize(
    "test_function",
    [
        test_the_application_never_reads_forwarding_headers,
        test_429s_are_logged_only_as_an_error_code,
    ],
    ids=lambda f: f.__name__,
)
def test_log_privacy_tests_request_the_restoring_fixture(test_function) -> None:
    """Audit finding L-a: these tests must keep using ``backend_info_logs``.

    ``create_app`` already sets ``backend`` to INFO, so they would still pass without
    the fixture, but would no longer restore the level they started with. pytest
    takes fixture requests from the test function's parameters, as read here; a
    parameter with a default value is not a fixture request (audit finding F-1).
    """
    import inspect

    parameters = inspect.signature(test_function).parameters
    assert "backend_info_logs" in parameters
    assert parameters["backend_info_logs"].default is inspect.Parameter.empty


def test_no_test_here_changes_the_backend_level_without_restoring_it() -> None:
    import inspect
    import sys

    call = "." + "setLevel" + "("  # built here so that this test's text never matches
    source = inspect.getsource(sys.modules[__name__])
    helper = inspect.getsource(_backend_log_level)
    # The only level changes are the helper's two calls: set, then restore in finally.
    assert helper.count(call) == 2
    assert source.count(call) == 2
