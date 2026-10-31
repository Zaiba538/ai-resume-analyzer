"""Groq upstream monitor: probe outcomes, passive signals, lifecycle."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from pydantic import SecretStr

from backend.health.upstream import GroqUpstreamMonitor
from tests.conftest import DEFAULT_MODELS, FAKE_API_KEY, FakeGroqAPI

FIXED_NOW = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)


def _monitor(fake: FakeGroqAPI, *, key: str | None = FAKE_API_KEY, interval: float = 300.0):
    return GroqUpstreamMonitor(
        api_key=SecretStr(key) if key else None,
        base_url="https://api.groq.com/openai/v1",
        required_models=DEFAULT_MODELS,
        interval_seconds=interval,
        timeout_seconds=1.0,
        client=fake.client(),
        clock=lambda: FIXED_NOW,
    )


@pytest.mark.asyncio
async def test_initial_snapshot_is_unknown(fake_groq) -> None:
    snapshot = _monitor(fake_groq).snapshot()
    assert snapshot.status == "unknown"
    assert snapshot.configured_models_present is None
    assert snapshot.last_checked_at is None


@pytest.mark.asyncio
async def test_no_key_means_no_probe(fake_groq) -> None:
    monitor = _monitor(fake_groq, key=None)
    assert monitor.key_configured is False
    snapshot = await monitor.probe()
    await monitor.start()
    await monitor.stop()
    assert fake_groq.calls == []
    assert snapshot.status == "unknown"


@pytest.mark.asyncio
async def test_successful_probe(fake_groq) -> None:
    snapshot = await _monitor(fake_groq).probe()
    assert snapshot.status == "available"
    assert snapshot.configured_models_present is True
    assert snapshot.last_checked_at == snapshot.last_success_at == FIXED_NOW
    assert snapshot.source == "probe"
    assert snapshot.last_error_code is None

    (request,) = fake_groq.calls
    assert request.method == "GET"
    assert str(request.url) == "https://api.groq.com/openai/v1/models"
    assert request.headers["Authorization"] == f"Bearer {FAKE_API_KEY}"


@pytest.mark.asyncio
async def test_missing_configured_model_reported(fake_groq) -> None:
    fake_groq.models = ["openai/gpt-oss-120b"]
    snapshot = await _monitor(fake_groq).probe()
    assert snapshot.status == "available"
    assert snapshot.configured_models_present is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "error_code"),
    [
        (401, "auth_failed"),
        (403, "auth_failed"),
        (429, "rate_limited"),
        (500, "upstream_error"),
        (503, "upstream_error"),
        (404, "unexpected_status"),
    ],
)
async def test_http_failures(fake_groq, status_code: int, error_code: str) -> None:
    fake_groq.status_code = status_code
    snapshot = await _monitor(fake_groq).probe()
    assert snapshot.status == "unavailable"
    assert snapshot.last_error_code == error_code
    assert snapshot.last_success_at is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception", "error_code"),
    [
        (httpx.ReadTimeout("timed out"), "timeout"),
        (httpx.ConnectTimeout("timed out"), "timeout"),
        (httpx.ConnectError("refused"), "network_error"),
    ],
)
async def test_transport_failures(fake_groq, exception: Exception, error_code: str) -> None:
    fake_groq.exception = exception
    snapshot = await _monitor(fake_groq).probe()
    assert snapshot.status == "unavailable"
    assert snapshot.last_error_code == error_code


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"not json", b'{"no_data": []}', b'{"data": [{"no_id": 1}]}'])
async def test_malformed_response(fake_groq, body: bytes) -> None:
    fake_groq.raw_body = body
    snapshot = await _monitor(fake_groq).probe()
    assert snapshot.status == "unavailable"
    assert snapshot.last_error_code == "invalid_response"


@pytest.mark.asyncio
async def test_recovery_clears_error(fake_groq) -> None:
    monitor = _monitor(fake_groq)
    fake_groq.status_code = 503
    await monitor.probe()
    fake_groq.status_code = 200
    snapshot = await monitor.probe()
    assert snapshot.status == "available"
    assert snapshot.last_error_code is None


@pytest.mark.asyncio
async def test_passive_request_signals(fake_groq) -> None:
    monitor = _monitor(fake_groq)
    monitor.record_request_outcome(False, "upstream_timeout")
    failed = monitor.snapshot()
    assert (failed.status, failed.source, failed.last_error_code) == (
        "unavailable",
        "request",
        "upstream_timeout",
    )
    monitor.record_request_outcome(True)
    ok = monitor.snapshot()
    assert (ok.status, ok.source, ok.last_error_code) == ("available", "request", None)
    assert ok.last_success_at == FIXED_NOW
    assert fake_groq.calls == []


@pytest.mark.asyncio
async def test_start_probes_periodically_and_stop_cancels(fake_groq) -> None:
    monitor = _monitor(fake_groq, interval=0.01)
    await monitor.start()
    await monitor.start()  # second start is a no-op
    for _ in range(200):
        if len(fake_groq.calls) >= 3:
            break
        await asyncio.sleep(0.01)
    await monitor.stop()
    calls_after_stop = len(fake_groq.calls)
    assert calls_after_stop >= 3
    await asyncio.sleep(0.05)
    assert len(fake_groq.calls) == calls_after_stop


@pytest.mark.asyncio
async def test_probe_logs_never_contain_key(fake_groq, captured_logs) -> None:
    import logging

    logging.getLogger("backend").setLevel(logging.INFO)
    fake_groq.status_code = 401
    await _monitor(fake_groq).probe()
    (record,) = captured_logs.events("groq_probe_completed")
    assert record["upstream_status"] == "unavailable"
    assert record["error_code"] == "auth_failed"
    assert FAKE_API_KEY not in captured_logs.text
