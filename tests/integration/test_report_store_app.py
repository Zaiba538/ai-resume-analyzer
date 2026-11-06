"""Report store wiring: readiness, lifecycle, settings and no listing route (M7 phase 1)."""

from __future__ import annotations

from backend.storage.report_store import ReportStore


def test_store_is_created_started_and_ready(start_app) -> None:
    client = start_app(max_stored_reports=7, report_ttl_seconds=120)
    store = client.app.state.report_store
    assert isinstance(store, ReportStore)
    assert store.running
    assert client.app.state.readiness.snapshot()["report_store_ready"] is True
    assert client.get("/ready").json()["checks"]["report_store_ready"] is True
    assert store._max_reports == 7 and store._ttl.total_seconds() == 120


def test_shutdown_stops_the_sweep_and_marks_not_ready(start_app) -> None:
    client = start_app()
    app = client.app
    client.__exit__(None, None, None)
    assert not app.state.report_store.running
    assert app.state.readiness.snapshot()["report_store_ready"] is False


def test_no_report_listing_route_exists(start_app) -> None:
    client = start_app()
    paths = {getattr(route, "path", "") for route in client.app.routes}
    assert not any(path.rstrip("/").endswith("/reports") for path in paths)
    for method in ("get", "post"):
        assert getattr(client, method)("/api/v1/reports").status_code in {404, 405}
