"""/health and /ready contracts (approved Rev. 3, correction 5)."""

from __future__ import annotations

import time

from tests.conftest import FAKE_API_KEY


def _wait_for_first_probe(client, timeout: float = 5.0) -> None:
    monitor = client.app.state.upstream_monitor
    deadline = time.monotonic() + timeout
    while monitor.snapshot().status == "unknown":
        assert time.monotonic() < deadline, "upstream probe did not complete"
        time.sleep(0.01)


def test_health_is_liveness_only(start_app, fake_groq) -> None:
    client = start_app()  # no API key: app is not ready, but is alive
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}
    assert fake_groq.calls == []


def test_ready_without_api_key_is_not_ready(start_app, fake_groq) -> None:
    client = start_app()
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"] == {
        "config_valid": True,
        "groq_api_key_configured": False,
        "embedding_model_loaded": True,  # M5 component
        "report_store_ready": True,  # M7 component
        "analysis_graph_compiled": True,  # M7 component
        "pdf_renderer_ready": True,  # M8 component
    }
    assert body["scoring"] == {"ats_composite": "not_calibrated", "calibration_version": None}
    groq = body["upstream"]["groq"]
    assert groq["status"] == "unknown"
    assert groq["configured_models_present"] is None
    assert fake_groq.calls == []


def test_ready_with_key_and_available_upstream(start_app, fake_groq) -> None:
    client = start_app(groq_api_key=FAKE_API_KEY)
    _wait_for_first_probe(client)
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"] == {
        "config_valid": True,
        "groq_api_key_configured": True,
        "embedding_model_loaded": True,  # M5 component
        "report_store_ready": True,  # M7 component
        "analysis_graph_compiled": True,  # M7 component
        "pdf_renderer_ready": True,  # M8 component
    }
    assert body["scoring"] == {"ats_composite": "not_calibrated", "calibration_version": None}
    groq = body["upstream"]["groq"]
    assert groq["status"] == "available"
    assert groq["configured_models_present"] is True
    assert groq["source"] == "probe"
    assert groq["last_checked_at"] and groq["last_success_at"]
    assert FAKE_API_KEY not in response.text


def test_upstream_outage_does_not_flip_readiness(start_app, fake_groq) -> None:
    fake_groq.status_code = 503
    client = start_app(groq_api_key=FAKE_API_KEY)
    _wait_for_first_probe(client)
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["upstream"]["groq"]["status"] == "unavailable"
    assert body["upstream"]["groq"]["last_error_code"] == "upstream_error"


def test_invalid_key_reported_as_upstream_auth_failure(start_app, fake_groq) -> None:
    fake_groq.status_code = 401
    client = start_app(groq_api_key=FAKE_API_KEY)
    _wait_for_first_probe(client)
    groq = client.get("/ready").json()["upstream"]["groq"]
    assert groq["status"] == "unavailable"
    assert groq["last_error_code"] == "auth_failed"


def test_ready_makes_no_outbound_call(start_app, fake_groq) -> None:
    """Exit criterion: readiness requests never call Groq."""
    client = start_app(groq_api_key=FAKE_API_KEY, groq_probe_interval_seconds=3600)
    _wait_for_first_probe(client)
    calls_before = len(fake_groq.calls)
    assert calls_before == 1  # the single start-up probe
    for _ in range(25):
        assert client.get("/ready").status_code == 200
        assert client.get("/health").status_code == 200
    assert len(fake_groq.calls) == calls_before


def test_monitor_stops_with_application(start_app, fake_groq) -> None:
    client = start_app(groq_api_key=FAKE_API_KEY)
    _wait_for_first_probe(client)
    monitor = client.app.state.upstream_monitor
    client.__exit__(None, None, None)
    assert monitor._task is None
