"""M10 Phase 2: the rate-limit dependency, the /analyze gate and route coverage."""

from __future__ import annotations

import asyncio
import contextlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated

import httpx
import pytest
from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.routing import iter_route_contexts
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Host, Mount, Route, Router, WebSocketRoute
from starlette.staticfiles import StaticFiles

from backend.api import rate_limits
from backend.api.rate_limits import (
    ANALYSIS_BUSY_MESSAGE,
    RATE_LIMITED_MESSAGE,
    AnalysisSlot,
    AnalysisSlotDep,
    AnalyzeConcurrencyGate,
    ConcurrencyLimitedError,
    RateLimitedError,
    RouteCoverageError,
    analysis_slot,
    enforce_rate_limit,
    validate_rate_limit_coverage,
)
from backend.api.routes import analyze, interview, match, parse, report
from backend.core.errors import register_exception_handlers
from backend.core.rate_limit import RateLimitDecision
from backend.main import create_app
from tests.fixtures.embeddings import CountingFactory

# --- errors --------------------------------------------------------------------------------


def test_rate_limited_error() -> None:
    error = RateLimitedError(17)
    assert (error.status_code, error.code, error.message) == (
        429,
        "rate_limited",
        RATE_LIMITED_MESSAGE,
    )
    assert error.headers == {"Retry-After": "17"}
    assert error.details is None


def test_concurrency_limited_error() -> None:
    error = ConcurrencyLimitedError()
    assert (error.status_code, error.code, error.message) == (
        429,
        "analysis_capacity_exceeded",
        ANALYSIS_BUSY_MESSAGE,
    )
    assert error.headers == {"Retry-After": "10"}


# --- the gate ------------------------------------------------------------------------------


def test_gate_admits_up_to_its_maximum() -> None:
    gate = AnalyzeConcurrencyGate(2)
    assert (gate.max_concurrent, gate.in_flight) == (2, 0)
    assert gate.try_acquire() and gate.try_acquire()
    assert not gate.try_acquire()
    assert gate.in_flight == 2  # a refusal changes nothing
    gate.release()
    assert gate.in_flight == 1
    assert gate.try_acquire()
    assert not gate.try_acquire()


def test_gate_rejects_unbalanced_release_and_bad_size() -> None:
    gate = AnalyzeConcurrencyGate(1)
    with pytest.raises(RuntimeError):
        gate.release()
    assert gate.in_flight == 0
    for bad in (0, -1):
        with pytest.raises(ValueError):
            AnalyzeConcurrencyGate(bad)


def _request(**state) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(**state)))


def test_slot_acquires_and_releases_exactly_once() -> None:
    gate = AnalyzeConcurrencyGate(1)
    slot = AnalysisSlot(gate)
    assert not slot.held and gate.in_flight == 0  # creating a slot takes nothing
    slot.acquire()
    assert slot.held and gate.in_flight == 1
    with pytest.raises(RuntimeError, match="already held"):
        slot.acquire()
    assert gate.in_flight == 1
    slot.release()
    slot.release()  # idempotent: never over-releases the gate
    assert not slot.held and gate.in_flight == 0


def test_slot_acquire_fails_with_429_when_the_gate_is_full() -> None:
    gate = AnalyzeConcurrencyGate(1)
    AnalysisSlot(gate).acquire()
    refused = AnalysisSlot(gate)
    with pytest.raises(ConcurrencyLimitedError):
        refused.acquire()
    assert not refused.held and gate.in_flight == 1
    refused.release()  # releasing a refused slot changes nothing
    assert gate.in_flight == 1


@pytest.mark.parametrize("exc", [ValueError("boom"), asyncio.CancelledError(), TimeoutError()])
def test_hold_releases_whatever_ends_the_block(exc: BaseException) -> None:
    gate = AnalyzeConcurrencyGate(1)
    slot = AnalysisSlot(gate)
    with pytest.raises(type(exc)), slot.hold():
        assert gate.in_flight == 1
        raise exc
    assert gate.in_flight == 0 and not slot.held


def test_hold_does_not_release_a_slot_it_could_not_take() -> None:
    gate = AnalyzeConcurrencyGate(1)
    AnalysisSlot(gate).acquire()
    with pytest.raises(ConcurrencyLimitedError), AnalysisSlot(gate).hold():
        pytest.fail("the block must not run without a slot")
    assert gate.in_flight == 1


@pytest.mark.asyncio
async def test_dependency_yields_an_unacquired_slot_and_releases_it_on_exit() -> None:
    gate = AnalyzeConcurrencyGate(1)
    dependency = analysis_slot(_request(analyze_gate=gate))
    slot = await dependency.__anext__()
    assert isinstance(slot, AnalysisSlot) and gate.in_flight == 0
    slot.acquire()  # the endpoint takes it, then fails to give it back
    with pytest.raises(StopAsyncIteration):
        await dependency.__anext__()
    assert gate.in_flight == 0  # the dependency released it


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [ValueError("boom"), asyncio.CancelledError(), TimeoutError()])
async def test_dependency_releases_a_held_slot_when_the_endpoint_fails(exc: BaseException) -> None:
    gate = AnalyzeConcurrencyGate(1)
    dependency = analysis_slot(_request(analyze_gate=gate))
    (await dependency.__anext__()).acquire()
    with pytest.raises(type(exc)):
        await dependency.athrow(exc)
    assert gate.in_flight == 0


def _gated_app(gate: AnalyzeConcurrencyGate) -> tuple[FastAPI, asyncio.Event]:
    app = FastAPI()
    app.state.analyze_gate = gate
    register_exception_handlers(app)
    started = asyncio.Event()

    @app.get("/slow")
    async def slow(slot: AnalysisSlotDep) -> dict:
        with slot.hold():
            started.set()
            await asyncio.sleep(3600)
        return {}

    @app.get("/fail")
    async def fail(slot: AnalysisSlotDep) -> dict:
        with slot.hold():
            raise RuntimeError("endpoint failure")

    @app.get("/leak")
    async def leak(slot: AnalysisSlotDep) -> dict:
        slot.acquire()  # never released by the endpoint itself
        return {}

    return app, started


@pytest.mark.asyncio
async def test_cancelling_a_request_releases_its_slot() -> None:
    gate = AnalyzeConcurrencyGate(1)
    app, started = _gated_app(gate)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        task = asyncio.create_task(client.get("/slow"))
        await asyncio.wait_for(started.wait(), 5)
        assert gate.in_flight == 1
        busy = await client.get("/slow")
        assert busy.status_code == 429
        assert busy.json()["error"]["code"] == "analysis_capacity_exceeded"
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_unhandled_endpoint_errors_release_the_slot() -> None:
    gate = AnalyzeConcurrencyGate(1)
    app, _ = _gated_app(gate)
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            assert (await client.get("/fail")).status_code == 500
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_a_slot_the_endpoint_forgets_is_released_by_the_dependency() -> None:
    gate = AnalyzeConcurrencyGate(1)
    app, _ = _gated_app(gate)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(3):
            assert (await client.get("/leak")).status_code == 200
    assert gate.in_flight == 0


@pytest.mark.asyncio
async def test_deadline_on_the_endpoint_releases_the_slot() -> None:
    gate = AnalyzeConcurrencyGate(1)
    app, started = _gated_app(gate)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.2):
                await client.get("/slow")
    assert started.is_set() and gate.in_flight == 0


# --- the rate-limit dependency -----------------------------------------------------------------


class SpyLimiter:
    def __init__(self, decision: RateLimitDecision) -> None:
        self.decision = decision
        self.calls: list[tuple[str, str | None]] = []

    def hit(self, policy: str, host: str | None) -> RateLimitDecision:
        self.calls.append((policy, host))
        return self.decision


def _limited_request(
    path: str | None, *, enabled: bool = True, host: str | None = "192.0.2.1", decision=None
):
    limiter = SpyLimiter(decision or RateLimitDecision(True, 0, "allowed"))
    state = SimpleNamespace(
        settings=SimpleNamespace(rate_limit_enabled=enabled), rate_limiter=limiter
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=state),
        scope={"route": SimpleNamespace(path=path)} if path is not None else {},
        client=SimpleNamespace(host=host) if host is not None else None,
    )
    return request, limiter


@pytest.mark.asyncio
@pytest.mark.parametrize(("path", "policy"), sorted(rate_limits.ROUTE_POLICIES.items()))
async def test_dependency_uses_the_routes_policy_and_the_client_host(
    path: str, policy: str
) -> None:
    request, limiter = _limited_request(path)
    assert await enforce_rate_limit(request) is None
    assert limiter.calls == [(policy, "192.0.2.1")]


@pytest.mark.asyncio
async def test_dependency_passes_a_missing_client_as_none() -> None:
    request, limiter = _limited_request("/api/v1/parse", host=None)
    await enforce_rate_limit(request)
    assert limiter.calls == [("parse", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["limited", "capacity"])
async def test_dependency_raises_429_with_the_decisions_retry_after(reason: str) -> None:
    request, _ = _limited_request("/api/v1/match", decision=RateLimitDecision(False, 42, reason))
    with pytest.raises(RateLimitedError) as excinfo:
        await enforce_rate_limit(request)
    assert excinfo.value.headers == {"Retry-After": "42"}
    assert excinfo.value.code == "rate_limited"  # capacity exhaustion looks the same


@pytest.mark.asyncio
async def test_disabled_dependency_never_touches_the_limiter() -> None:
    request, limiter = _limited_request("/api/v1/match", enabled=False)
    await enforce_rate_limit(request)
    request, limiter2 = _limited_request(None, enabled=False)  # not even the route lookup
    await enforce_rate_limit(request)
    assert limiter.calls == [] and limiter2.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [None, "/api/v1/unknown", "/health"])
async def test_dependency_fails_closed_without_a_policy(path: str | None) -> None:
    request, limiter = _limited_request(path)
    with pytest.raises(RuntimeError, match="route has no rate-limit policy"):
        await enforce_rate_limit(request)
    assert limiter.calls == []


def test_dependencies_are_coroutines_so_they_run_on_the_event_loop() -> None:
    assert asyncio.iscoroutinefunction(enforce_rate_limit)
    import inspect

    assert inspect.isasyncgenfunction(analysis_slot)


# --- route coverage (validated on the assembled application) -----------------------------


def _app(make_settings) -> FastAPI:
    return create_app(make_settings(), embedder_factory=CountingFactory())


def _limited_app(*routers: APIRouter) -> FastAPI:
    """A bare app (no M10 wiring) for building route tables by hand."""
    app = FastAPI()
    for router in routers:
        app.include_router(router, dependencies=[Depends(enforce_rate_limit)])
    return app


def _api_paths(routes) -> set[str]:
    return {c.path for c in iter_route_contexts(routes) if (c.path or "").startswith("/api/")}


def test_the_application_route_table_passes(make_settings) -> None:
    app = _app(make_settings)
    validate_rate_limit_coverage(app.router)  # create_app already did; it is repeatable
    assert _api_paths(app.routes) == set(rate_limits.ROUTE_POLICIES)
    assert rate_limits.ROUTE_METHODS.keys() == rate_limits.ROUTE_POLICIES.keys()


def test_create_app_validates_its_final_route_table(make_settings, monkeypatch) -> None:
    import backend.main as main

    seen: list[set[str]] = []
    real = main.validate_rate_limit_coverage

    def spy(router) -> None:
        seen.append({c.path for c in iter_route_contexts(router.routes) if c.path})
        real(router)

    monkeypatch.setattr(main, "validate_rate_limit_coverage", spy)
    _app(make_settings)
    (paths,) = seen
    assert set(rate_limits.ROUTE_POLICIES) | {"/health", "/ready"} <= paths


def test_an_unlisted_api_router_is_detected(make_settings) -> None:
    app = _app(make_settings)
    extra = APIRouter(prefix="/api/v1")

    @extra.post("/unlisted")
    async def unlisted() -> dict:
        return {}

    app.include_router(extra)
    with pytest.raises(RouteCoverageError, match="an API route has no rate-limit policy"):
        validate_rate_limit_coverage(app.router)


def test_a_directly_registered_api_endpoint_is_detected(make_settings) -> None:
    app = _app(make_settings)

    @app.post("/api/v1/direct", dependencies=[Depends(enforce_rate_limit)])
    async def direct() -> dict:
        return {}

    with pytest.raises(RouteCoverageError, match="an API route has no rate-limit policy"):
        validate_rate_limit_coverage(app.router)


def test_a_direct_endpoint_on_an_approved_path_without_the_dependency_is_detected(
    make_settings,
) -> None:
    app = _app(make_settings)

    @app.get("/api/v1/report/{report_id}")
    async def shadow(report_id: str) -> dict:
        return {}

    with pytest.raises(RouteCoverageError, match="does not run the rate limit first"):
        validate_rate_limit_coverage(app.router)


def test_a_router_included_without_the_rate_limit_is_detected() -> None:
    app = _limited_app(parse.router, match.router, report.router, interview.router)
    app.include_router(analyze.router)  # forgot the rate limit
    with pytest.raises(RouteCoverageError, match="does not run the rate limit first"):
        validate_rate_limit_coverage(app.router)


def test_the_rate_limit_must_run_before_other_dependencies() -> None:
    async def other() -> None:
        return None

    app = _limited_app(parse.router, match.router, analyze.router, report.router)
    app.include_router(interview.router, dependencies=[Depends(other), Depends(enforce_rate_limit)])
    with pytest.raises(RouteCoverageError, match="does not run the rate limit first"):
        validate_rate_limit_coverage(app.router)


def test_an_analyze_route_without_its_concurrency_dependency_is_detected() -> None:
    app = _limited_app(parse.router, match.router, report.router, interview.router)
    bare = APIRouter(prefix="/api/v1")

    @bare.post("/analyze")
    async def unguarded_analyze() -> dict:
        return {}

    app.include_router(bare, dependencies=[Depends(enforce_rate_limit)])
    with pytest.raises(RouteCoverageError, match="lacks its concurrency dependency"):
        validate_rate_limit_coverage(app.router)


def test_a_request_scoped_concurrency_dependency_is_rejected() -> None:
    app = _limited_app(parse.router, match.router, report.router, interview.router)
    bare = APIRouter(prefix="/api/v1")

    @bare.post("/analyze")
    async def late_release(slot: Annotated[AnalysisSlot, Depends(analysis_slot)]) -> dict:
        return {}

    app.include_router(bare, dependencies=[Depends(enforce_rate_limit)])
    with pytest.raises(RouteCoverageError, match="lacks its concurrency dependency"):
        validate_rate_limit_coverage(app.router)


def test_the_concurrency_dependency_is_only_for_analyze() -> None:
    app = _limited_app(parse.router, match.router, analyze.router, interview.router)
    bare = APIRouter(prefix="/api/v1")

    @bare.get("/report/{report_id}")
    async def report_with_slot(report_id: str, slot: AnalysisSlotDep) -> dict:
        return {}

    @bare.get("/report/{report_id}/pdf")
    async def pdf(report_id: str) -> dict:
        return {}

    app.include_router(bare, dependencies=[Depends(enforce_rate_limit)])
    with pytest.raises(RouteCoverageError, match="only the analyze route"):
        validate_rate_limit_coverage(app.router)


def test_an_extra_method_on_an_approved_path_is_detected(make_settings) -> None:
    app = _app(make_settings)

    @app.delete("/api/v1/parse", dependencies=[Depends(enforce_rate_limit)])
    async def delete_parse() -> dict:
        return {}

    with pytest.raises(RouteCoverageError, match="unexpected methods"):
        validate_rate_limit_coverage(app.router)


def test_a_missing_approved_route_is_detected() -> None:
    app = _limited_app(parse.router, match.router, analyze.router, report.router)  # no interview
    with pytest.raises(RouteCoverageError, match="names an unknown route"):
        validate_rate_limit_coverage(app.router)


@pytest.mark.parametrize("kind", ["route", "mount", "websocket"])
def test_non_fastapi_routes_under_api_are_rejected(make_settings, kind: str) -> None:
    app = _app(make_settings)
    if kind == "route":
        app.router.routes.append(Route("/api/v1/raw", lambda request: PlainTextResponse("x")))
    elif kind == "mount":
        app.router.routes.append(Mount("/api/v2", app=PlainTextResponse("x")))
    else:
        app.router.routes.append(WebSocketRoute("/api/v1/ws", lambda ws: None))
    with pytest.raises(RouteCoverageError, match="not a FastAPI route"):
        validate_rate_limit_coverage(app.router)


def test_exempt_routes_need_no_protection(make_settings, tmp_path) -> None:
    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    app = create_app(
        make_settings(api_docs_enabled=True, frontend_dir=tmp_path),
        embedder_factory=CountingFactory(),
    )

    @app.get("/apiary")  # not under /api/
    async def apiary() -> dict:
        return {}

    validate_rate_limit_coverage(app.router)
    exempt = {c.path for c in iter_route_contexts(app.routes)} - set(rate_limits.ROUTE_POLICIES)
    assert {"/health", "/ready", "/docs", "/redoc", "/openapi.json", "/apiary"} <= exempt


def test_startup_fails_closed_with_a_fixed_message(make_settings, monkeypatch) -> None:
    monkeypatch.delitem(rate_limits.ROUTE_POLICIES, "/api/v1/match")
    with pytest.raises(RouteCoverageError) as excinfo:
        _app(make_settings)
    assert str(excinfo.value) == "an API route has no rate-limit policy"
    assert issubclass(RouteCoverageError, RuntimeError)


def test_a_route_at_the_bare_api_prefix_is_checked(make_settings) -> None:
    app = _app(make_settings)

    @app.get("/api")
    async def api_index() -> dict:
        return {}

    with pytest.raises(RouteCoverageError, match="an API route has no rate-limit policy"):
        validate_rate_limit_coverage(app.router)


# --- mounted applications and Host routes (audit finding M1-a) ------------------------------


def _sub_app() -> FastAPI:
    """A nested application exposing an unprotected /api/ route."""
    sub = FastAPI()

    @sub.post("/api/v1/sneaky")
    async def sneaky() -> dict:
        return {}

    return sub


@pytest.mark.parametrize("prefix", ["/", "/tools", "/static"])
def test_a_mounted_sub_application_is_rejected(make_settings, prefix: str) -> None:
    app = _app(make_settings)
    app.mount(prefix, _sub_app())
    with pytest.raises(RouteCoverageError, match="only the static frontend may be mounted"):
        validate_rate_limit_coverage(app.router)


def test_a_sub_application_at_the_root_cannot_serve_unchecked_api_routes(make_settings) -> None:
    app = _app(make_settings)
    app.router.routes.insert(0, Mount("/", app=_sub_app()))  # matched before everything else
    with pytest.raises(RouteCoverageError, match="only the static frontend may be mounted"):
        validate_rate_limit_coverage(app.router)


@pytest.mark.parametrize("inner", ["sub_app", "static", "router"])
def test_host_routes_are_rejected(make_settings, tmp_path, inner: str) -> None:
    app = _app(make_settings)
    target = {
        "sub_app": _sub_app,
        "static": lambda: StaticFiles(directory=tmp_path),
        "router": lambda: Router(routes=[Mount("/", app=_sub_app())]),
    }[inner]()
    app.router.routes.insert(0, Host("{tenant}.example.com", app=target))
    with pytest.raises(RouteCoverageError, match="host routes are not allowed"):
        validate_rate_limit_coverage(app.router)


def test_a_mount_inside_an_included_router_is_rejected(make_settings) -> None:
    app = _app(make_settings)
    router = APIRouter()
    router.mount("/nested", _sub_app())
    app.include_router(router)
    with pytest.raises(RouteCoverageError, match="only the static frontend may be mounted"):
        validate_rate_limit_coverage(app.router)


def test_the_approved_static_frontend_is_accepted(make_settings, tmp_path) -> None:
    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    app = create_app(make_settings(frontend_dir=tmp_path), embedder_factory=CountingFactory())
    (frontend,) = [r for r in app.routes if isinstance(r, Mount)]
    assert (frontend.path, frontend.name, type(frontend.app)) == ("", "frontend", StaticFiles)
    validate_rate_limit_coverage(app.router)  # create_app already accepted it


@pytest.mark.parametrize("mount", ["other_name", "other_path", "subclass", "wrapped"])
def test_only_the_exact_frontend_mount_is_accepted(make_settings, tmp_path, mount: str) -> None:
    """Each look-alike is the app's only mount, so it is judged on its own."""
    app = _app(make_settings)  # no frontend directory: no mount of its own
    assert not [r for r in app.routes if isinstance(r, Mount)]

    class LookAlike(StaticFiles):
        pass

    static = StaticFiles(directory=tmp_path)
    route = {
        "other_name": Mount("/", app=static, name="assets"),
        "other_path": Mount("/static", app=static, name="frontend"),
        "subclass": Mount("/", app=LookAlike(directory=tmp_path), name="frontend"),
        "wrapped": Mount("/", app=static, name="frontend", middleware=[Middleware(GZipMiddleware)]),
    }[mount]
    app.router.routes.append(route)
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == "only the static frontend may be mounted"


def test_the_exact_frontend_mount_alone_is_accepted(make_settings, tmp_path) -> None:
    app = _app(make_settings)
    app.router.routes.append(Mount("/", app=StaticFiles(directory=tmp_path), name="frontend"))
    validate_rate_limit_coverage(app.router)


def test_a_second_frontend_mount_is_rejected(make_settings, tmp_path) -> None:
    (tmp_path / "index.html").write_text("ok", encoding="utf-8")
    app = create_app(make_settings(frontend_dir=tmp_path), embedder_factory=CountingFactory())
    app.router.routes.append(Mount("/", app=StaticFiles(directory=tmp_path), name="frontend"))
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == "only one static frontend may be mounted"


def test_the_frontend_is_still_served_and_api_routes_still_limited(make_settings, tmp_path) -> None:
    from fastapi.testclient import TestClient

    (tmp_path / "index.html").write_text("<p>frontend</p>", encoding="utf-8")
    app = create_app(
        make_settings(frontend_dir=tmp_path, rate_limit_enabled=True, rate_limit_report="1/minute"),
        embedder_factory=CountingFactory(),
    )
    url = "/api/v1/report/00000000-0000-4000-8000-000000000000"
    with TestClient(app) as client:
        assert client.get("/").text == "<p>frontend</p>"
        assert client.get("/health").status_code == 200
        assert client.get(url).status_code == 404
        assert client.get(url).status_code == 429


# --- F1: routes judged by the paths they can match ---------------------------------------------

OUTSIDE_API = "a route outside /api can match API paths"
F1_BYPASSES = [
    "/{rest:path}",
    "/{prefix}/v1/sneaky",
    "/api{rest:path}",
    "/ap{x}/v1/sneaky",
    "/{a}{b:path}",
    "/{segment}",  # exactly /api
    "/a{x}",  # /api
    "/{x:path}/parse",
    "/{a}/{b}/{c}",
    "/{seg}/v1/parse",  # overlaps the approved /api/v1/parse
    "/{seg}/v1/future-endpoint",  # a future /api/v1/... path
    "/{seg}/{version}/report/{report_id}",
]
LEGITIMATE = [
    "/items/{item_id}",
    "/{n:int}/x",
    "/{n:int}",
    "/{x:float}/y",
    "/{item:uuid}",
    "/files/{p:path}",
    "/apix/{p:path}",
    "/api-docs",
    "/apiary",
    "/{a}x",  # a str parameter cannot contain "/", and "x" cannot end "/api"
    "/api{x}",  # /apifoo, never /api or /api/...
    "/api{n:int}",
    "/x/{a}/{b:path}",
    "/api{x}/{rest:path}",  # x needs at least one character: /apiZ/..., never /api/...
]


async def _ok() -> dict:
    return {"ok": True}


@pytest.mark.parametrize("template", F1_BYPASSES)
def test_every_f1_bypass_is_rejected(make_settings, template: str) -> None:
    app = _app(make_settings)
    app.add_api_route(template, _ok, methods=["POST"])
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == OUTSIDE_API


@pytest.mark.parametrize("template", LEGITIMATE)
def test_legitimate_non_api_routes_are_accepted(make_settings, template: str) -> None:
    app = _app(make_settings)
    app.add_api_route(template, _ok, methods=["GET"])
    validate_rate_limit_coverage(app.router)


@pytest.mark.parametrize("kind", ["starlette_route", "websocket", "fastapi_websocket", "router"])
def test_catch_alls_of_every_route_kind_are_rejected(make_settings, kind: str) -> None:
    app = _app(make_settings)
    if kind == "starlette_route":
        app.router.routes.append(Route("/{p:path}", endpoint=_sub_app(), methods=["POST"]))
    elif kind == "websocket":
        app.router.routes.append(WebSocketRoute("/{p:path}", lambda ws: None))
    elif kind == "fastapi_websocket":

        @app.websocket("/{p:path}")
        async def socket(websocket) -> None:
            return None

    else:
        router = APIRouter()
        router.add_api_route("/{p:path}", _ok, methods=["POST"])
        app.include_router(router, dependencies=[Depends(enforce_rate_limit)])
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == OUTSIDE_API


def test_a_custom_converter_is_treated_as_matching_anything(make_settings, monkeypatch) -> None:
    from starlette.convertors import CONVERTOR_TYPES, IntegerConvertor

    class DigitsOnly(IntegerConvertor):  # in fact digits only, but unknown to the analysis
        pass

    monkeypatch.setitem(CONVERTOR_TYPES, "digits", DigitsOnly())
    app = _app(make_settings)
    app.add_api_route("/{n:digits}/x", _ok, methods=["GET"])
    with pytest.raises(RouteCoverageError, match=OUTSIDE_API):
        validate_rate_limit_coverage(app.router)


def test_a_catch_all_before_the_api_routes_intercepts_them_and_is_rejected(make_settings) -> None:
    from fastapi.testclient import TestClient

    app = _app(make_settings)
    shadow = Route(
        "/{p:path}", endpoint=lambda request: PlainTextResponse("shadow"), methods=["POST"]
    )
    app.router.routes.insert(0, shadow)
    with TestClient(app) as client:  # the threat: the protected route is never reached
        assert client.post("/api/v1/parse").text == "shadow"
    with pytest.raises(RouteCoverageError, match=OUTSIDE_API):
        validate_rate_limit_coverage(app.router)


def test_a_catch_all_after_the_api_routes_serves_new_paths_and_is_rejected(make_settings) -> None:
    from fastapi.testclient import TestClient

    app = create_app(
        make_settings(rate_limit_enabled=True, rate_limit_parse="1/minute"),
        embedder_factory=CountingFactory(),
    )
    app.add_api_route("/{rest:path}", _ok, methods=["POST"])
    with TestClient(app) as client:  # the threat: unlimited /api/... paths
        assert [client.post("/api/v1/future").status_code for _ in range(3)] == [200, 200, 200]
        assert client.post("/api/v1/parse").status_code == 415  # approved route still first
        assert client.post("/api/v1/parse").status_code == 429
    with pytest.raises(RouteCoverageError, match=OUTSIDE_API):
        validate_rate_limit_coverage(app.router)


def test_create_app_refuses_a_router_with_a_catch_all(make_settings, monkeypatch) -> None:
    import backend.api.routes.report as report_module

    grown = APIRouter()
    grown.include_router(report_module.router)
    grown.add_api_route("/{rest:path}", _ok, methods=["GET"])
    monkeypatch.setattr(report_module, "router", grown)
    with pytest.raises(RouteCoverageError) as excinfo:
        _app(make_settings)
    assert str(excinfo.value) == OUTSIDE_API


def test_legitimate_parameterised_routes_still_serve_requests(make_settings) -> None:
    from fastapi.testclient import TestClient

    app = create_app(
        make_settings(rate_limit_enabled=True, rate_limit_report="1/minute"),
        embedder_factory=CountingFactory(),
    )

    @app.get("/items/{item_id}")
    async def item(item_id: str) -> dict:
        return {"item": item_id}

    @app.get("/files/{p:path}")
    async def files(p: str) -> dict:
        return {"file": p}

    validate_rate_limit_coverage(app.router)
    url = "/api/v1/report/00000000-0000-4000-8000-000000000000"
    with TestClient(app) as client:
        assert client.get("/items/42").json() == {"item": "42"}
        assert client.get("/files/a/b.txt").json() == {"file": "a/b.txt"}
        assert client.get(url).status_code == 404
        assert client.get(url).status_code == 429  # API routes are still limited


@pytest.mark.parametrize("how", ["direct", "router_included_twice", "report_twice"])
def test_duplicate_api_routes_are_rejected(make_settings, how: str) -> None:
    app = _app(make_settings)
    if how == "direct":
        app.add_api_route(
            "/api/v1/parse", _ok, methods=["POST"], dependencies=[Depends(enforce_rate_limit)]
        )
    elif how == "router_included_twice":
        app.include_router(match.router, dependencies=[Depends(enforce_rate_limit)])
    else:
        app.add_api_route(
            "/api/v1/report/{report_id}",
            _ok,
            methods=["GET"],
            dependencies=[Depends(enforce_rate_limit)],
        )
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == "an API route is registered more than once"


def test_the_analysis_never_misses_a_path_starlette_would_match() -> None:
    """Soundness against Starlette's own compiled regex, on many generated templates.

    The analysis may reject a route that could not really match an API path (it
    over-approximates unknown cases), but it must never accept one that can.
    """
    import random

    from starlette.routing import compile_path

    from backend.api.rate_limits import _can_match_api_paths

    rng = random.Random(20261104)  # noqa: S311 - a seeded, deterministic test generator
    literals = ["/", "a", "p", "i", "/api", "/ap", "x", "/v1", "-", "/api/", "api"]
    params = ["{p%d}", "{p%d:path}", "{p%d:int}", "{p%d:float}", "{p%d:uuid}"]
    probes = ["/api", "/api/", "/api/v1/parse", "/api/x", "/api/1", "/api/1.5", "/api/a/b/c",
              "/api/v1/sneaky", "/api/api", "/api/-", "/api/v1/x/v1"]  # fmt: skip
    checked = 0
    for _ in range(12000):
        parts, count = [], 0
        for _ in range(rng.randint(1, 4)):
            if rng.random() < 0.5:
                parts.append(rng.choice(literals))
            else:
                parts.append(rng.choice(params) % count)
                count += 1
        template = "".join(parts)
        if not template.startswith("/") or "//" in template:
            continue
        try:
            regex, _, convertors = compile_path(template)
        except ValueError:
            continue
        if any(regex.match(probe) for probe in probes):
            checked += 1
            assert _can_match_api_paths(template, convertors), template
    assert checked > 200  # the generator really produced matching templates


# --- N1: FastAPI's frontend() low-priority routes -----------------------------------------------

LOW_PRIORITY = "low-priority routes such as frontend() are not allowed"


def _static_dir(tmp_path) -> Path:
    (tmp_path / "index.html").write_text("<p>spa</p>", encoding="utf-8")
    (tmp_path / "v1").mkdir(exist_ok=True)
    (tmp_path / "v1" / "data.json").write_text('{"static": true}', encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("prefix", ["/api", "/", "/tools", "/api/v1", "/{x}"])
def test_app_frontend_is_rejected_at_any_prefix(make_settings, tmp_path, prefix: str) -> None:
    app = _app(make_settings)
    app.frontend(prefix, directory=_static_dir(tmp_path))
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == LOW_PRIORITY


def test_an_unvalidated_frontend_would_serve_files_under_api(make_settings, tmp_path) -> None:
    """The threat N1 closes: static files served under /api with no rate limit."""
    from fastapi.testclient import TestClient

    app = _app(make_settings)
    app.frontend("/api", directory=_static_dir(tmp_path))
    with TestClient(app) as client:
        assert client.get("/api/v1/data.json").json() == {"static": True}
    with pytest.raises(RouteCoverageError, match=re.escape(LOW_PRIORITY)):
        validate_rate_limit_coverage(app.router)


def test_create_app_refuses_a_router_carrying_a_frontend(
    make_settings, tmp_path, monkeypatch
) -> None:
    """A frontend inside an included router stops start-up, so nothing under /api is served."""
    import backend.api.routes.report as report_module

    grown = APIRouter()
    grown.include_router(report_module.router)
    grown.frontend("/api", directory=_static_dir(tmp_path))
    monkeypatch.setattr(report_module, "router", grown)
    with pytest.raises(RouteCoverageError) as excinfo:
        _app(make_settings)
    assert str(excinfo.value) == LOW_PRIORITY


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_frontend_in_nested_included_routers_is_rejected(
    make_settings, tmp_path, depth: int
) -> None:
    app = _app(make_settings)
    innermost = APIRouter(prefix="/inner")
    innermost.frontend("/files", directory=_static_dir(tmp_path))
    router = innermost
    for level in range(depth - 1):
        outer = APIRouter(prefix=f"/level{level}")
        outer.include_router(router)
        router = outer
    app.include_router(router)
    with pytest.raises(RouteCoverageError, match=re.escape(LOW_PRIORITY)):
        validate_rate_limit_coverage(app.router)


def test_several_frontend_registrations_are_rejected(make_settings, tmp_path) -> None:
    app = _app(make_settings)
    static = _static_dir(tmp_path)
    app.frontend("/a", directory=static)
    app.frontend("/b", directory=static)
    router = APIRouter()
    router.frontend("/c", directory=static)
    app.include_router(router)
    assert len(list(app.router._iter_low_priority_routes())) >= 2
    with pytest.raises(RouteCoverageError, match=re.escape(LOW_PRIORITY)):
        validate_rate_limit_coverage(app.router)


def test_the_approved_frontend_has_no_low_priority_routes(make_settings, tmp_path) -> None:
    from fastapi.testclient import TestClient

    app = create_app(
        make_settings(frontend_dir=_static_dir(tmp_path), rate_limit_enabled=True),
        embedder_factory=CountingFactory(),
    )
    assert list(app.router._iter_low_priority_routes()) == []
    validate_rate_limit_coverage(app.router)
    with TestClient(app) as client:
        assert client.get("/").text == "<p>spa</p>"  # the approved StaticFiles mount
        assert client.get("/v1/data.json").json() == {"static": True}
        assert client.get("/api/v1/data.json").status_code == 404  # nothing static under /api


def test_uninspectable_low_priority_routes_fail_closed() -> None:
    router_without_internals = SimpleNamespace(routes=[])
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(router_without_internals)  # type: ignore[arg-type]
    assert str(excinfo.value) == "low-priority routes cannot be inspected"


def test_fastapi_low_priority_internals_are_as_pinned(tmp_path) -> None:
    """Fails after a FastAPI upgrade that changes the internals N1 relies on."""
    import fastapi
    from fastapi.routing import APIRouter as Router

    assert fastapi.__version__ == "0.162.0", "re-audit N1 (frontend routes) for this FastAPI"
    router = Router()
    assert router._low_priority_routes == [] and list(router._iter_low_priority_routes()) == []
    router.frontend("/x", directory=tmp_path)
    assert len(router._low_priority_routes) == 1
    assert len(list(router._iter_low_priority_routes())) == 1
    parent = Router()
    parent.include_router(router)
    assert len(list(parent._iter_low_priority_routes())) == 1  # propagated through include


# --- N2: the router's not-found fallback ----------------------------------------------------------

FALLBACK_REPLACED = "the router's not-found fallback has been replaced"


def _fallbacks(router, tmp_path) -> dict:
    """Replacement fallbacks; none of them is Starlette's built-in, bound to `router`."""
    import functools
    from types import MethodType

    from starlette.routing import Router as StarletteRouter

    async def asgi_callable(scope, receive, send) -> None:
        return None

    def plain_function(*args) -> None:
        return None

    builtin = router.default

    @functools.wraps(builtin)
    async def decorated(scope, receive, send) -> None:
        await builtin(scope, receive, send)

    class LooksBound:  # carries exactly the attributes a bound method would have
        def __init__(self) -> None:
            self.__func__ = StarletteRouter.not_found
            self.__self__ = router

        async def __call__(self, scope, receive, send) -> None:
            return None

    async def other_not_found(self, scope, receive, send) -> None:
        return None

    async def not_found(self, scope, receive, send) -> None:  # same name, different function
        return None

    return {
        "fastapi_sub_app": _sub_app(),
        "static_files": StaticFiles(directory=tmp_path),
        "asgi_callable": asgi_callable,
        "plain_function": plain_function,
        "decorated_builtin": decorated,
        "partial_builtin": functools.partial(builtin),
        "imitation_of_a_bound_method": LooksBound(),
        "builtin_bound_to_another_router": StarletteRouter().not_found,
        "other_function_bound_to_this_router": MethodType(other_not_found, router),
        "same_name_function_bound_to_this_router": MethodType(not_found, router),
        "none": None,
        "number": 42,
    }


@pytest.mark.parametrize(
    "kind",
    [
        "fastapi_sub_app",
        "static_files",
        "asgi_callable",
        "plain_function",
        "decorated_builtin",
        "partial_builtin",
        "imitation_of_a_bound_method",
        "builtin_bound_to_another_router",
        "other_function_bound_to_this_router",
        "same_name_function_bound_to_this_router",
        "none",
        "number",
    ],
)
def test_a_replaced_fallback_is_rejected(make_settings, tmp_path, kind: str) -> None:
    app = _app(make_settings)
    app.router.default = _fallbacks(app.router, tmp_path)[kind]
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == FALLBACK_REPLACED


def test_a_missing_fallback_is_rejected(make_settings) -> None:
    app = _app(make_settings)
    del app.router.default
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == FALLBACK_REPLACED


def test_validation_never_calls_the_fallback(make_settings) -> None:
    app = _app(make_settings)
    calls: list[str] = []

    async def recording(scope, receive, send) -> None:
        calls.append("called")

    app.router.default = recording
    with pytest.raises(RouteCoverageError):
        validate_rate_limit_coverage(app.router)
    assert calls == []


def test_a_fallback_sub_app_would_serve_api_paths_and_is_rejected(make_settings) -> None:
    """The threat N2 closes: unprotected /api/... endpoints behind the fallback."""
    from fastapi.testclient import TestClient

    app = create_app(
        make_settings(rate_limit_enabled=True, rate_limit_parse="1/minute"),
        embedder_factory=CountingFactory(),
    )
    app.router.default = _sub_app()
    with TestClient(app) as client:
        assert [client.post("/api/v1/sneaky").status_code for _ in range(3)] == [200, 200, 200]
    with pytest.raises(RouteCoverageError, match=re.escape(FALLBACK_REPLACED)):
        validate_rate_limit_coverage(app.router)


def test_create_app_refuses_to_start_with_a_replaced_fallback(make_settings, monkeypatch) -> None:
    """A not_found swapped in before start-up is caught before any request is served."""
    from starlette.routing import Router as StarletteRouter

    async def serving_not_found(self, scope, receive, send) -> None:
        return None

    monkeypatch.setattr(StarletteRouter, "not_found", serving_not_found)
    with pytest.raises(RouteCoverageError) as excinfo:
        _app(make_settings)
    assert str(excinfo.value) == FALLBACK_REPLACED


def test_an_included_routers_fallback_is_not_used_and_not_rejected(make_settings, tmp_path) -> None:
    from fastapi.testclient import TestClient

    app = _app(make_settings)
    router = APIRouter()
    router.add_api_route("/apiary-status", _ok, methods=["GET"])
    router.default = _sub_app()  # never reached: the app's own fallback handles misses
    app.include_router(router)
    validate_rate_limit_coverage(app.router)
    with TestClient(app) as client:
        response = client.post("/api/v1/sneaky")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"


def test_the_builtin_fallback_is_accepted_and_still_answers_404(make_settings, tmp_path) -> None:
    from types import MethodType

    from fastapi.testclient import TestClient
    from starlette.routing import Router as StarletteRouter

    (tmp_path / "index.html").write_text("<p>approved</p>", encoding="utf-8")
    app = create_app(
        make_settings(frontend_dir=tmp_path, api_docs_enabled=True),
        embedder_factory=CountingFactory(),
    )
    fallback = app.router.default
    assert isinstance(fallback, MethodType)
    assert fallback.__func__ is StarletteRouter.not_found and fallback.__self__ is app.router
    validate_rate_limit_coverage(app.router)
    with TestClient(app) as client:
        missing = client.get("/api/v1/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "not_found"
        assert missing.json()["error"]["request_id"] == missing.headers["X-Request-ID"]
        assert client.get("/").text == "<p>approved</p>"
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code in (200, 503)
        assert client.get("/docs").status_code == 200


def test_starlette_fallback_internals_are_as_pinned() -> None:
    """Fails after a Starlette or FastAPI upgrade that changes what N2 relies on."""
    from types import MethodType

    import starlette
    from fastapi import FastAPI
    from starlette.routing import Router as StarletteRouter

    assert starlette.__version__ == "1.10.0", "re-audit N2 (router fallback) for this Starlette"
    app = FastAPI()
    assert isinstance(app.router.default, MethodType)
    assert app.router.default.__func__ is StarletteRouter.not_found
    assert app.router.default.__self__ is app.router
    assert "not_found" not in vars(type(app.router))  # FastAPI does not override it


# --- F-N2-1: a proxy imitating the built-in fallback ------------------------------------------


def _forwarding_proxy(target):
    """A proxy around a bound method: forwards every attribute (``__class__``,
    ``__func__``, ``__self__`` included) but answers requests itself."""
    from fastapi.responses import JSONResponse

    calls: list[str] = []

    class ForwardingProxy:
        def __getattribute__(self, name):
            return getattr(target, name)

        async def __call__(self, scope, receive, send) -> None:
            calls.append(scope["path"])
            response = JSONResponse({"sneaky": True}, status_code=200)
            await response(scope, receive, send)

    return ForwardingProxy(), calls


def test_a_forwarding_proxy_fools_isinstance_but_not_the_validator(make_settings) -> None:
    from types import MethodType

    from starlette.routing import Router as StarletteRouter

    app = _app(make_settings)
    proxy, calls = _forwarding_proxy(app.router.default)
    # It passes the previous isinstance()-based check...
    assert isinstance(proxy, MethodType) and type(proxy) is not MethodType
    assert proxy.__func__ is StarletteRouter.not_found and proxy.__self__ is app.router
    assert (
        isinstance(proxy, MethodType)
        and proxy.__func__ is StarletteRouter.not_found
        and proxy.__self__ is app.router
    )
    # ...but not the exact-type check, and validation never calls it.
    app.router.default = proxy
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == FALLBACK_REPLACED
    assert calls == []


def test_a_forwarding_proxy_would_serve_api_paths_and_is_refused_at_startup(
    make_settings, monkeypatch
) -> None:
    from fastapi.testclient import TestClient

    import backend.main as main

    # The vulnerable configuration, unvalidated: the proxy answers /api/v1/sneaky.
    app = create_app(
        make_settings(rate_limit_enabled=True, rate_limit_parse="1/minute"),
        embedder_factory=CountingFactory(),
    )
    proxy, calls = _forwarding_proxy(app.router.default)
    app.router.default = proxy
    with TestClient(app) as client:
        assert [client.post("/api/v1/sneaky").status_code for _ in range(3)] == [200, 200, 200]
    assert calls == ["/api/v1/sneaky"] * 3

    # The same proxy installed during create_app: start-up fails, so nothing is served.
    installed: list = []
    real_register = main.register_exception_handlers

    def register_and_install_proxy(application) -> None:
        real_register(application)
        proxy, proxy_calls = _forwarding_proxy(application.router.default)
        application.router.default = proxy
        installed.append(proxy_calls)

    monkeypatch.setattr(main, "register_exception_handlers", register_and_install_proxy)
    with pytest.raises(RouteCoverageError) as excinfo:
        _app(make_settings)
    assert str(excinfo.value) == FALLBACK_REPLACED
    (proxy_calls,) = installed
    assert proxy_calls == []  # never called: no app was returned to serve requests


# --- F-N2-2: the router's middleware stack (its dispatch) ---------------------------------------

DISPATCH_REPLACED = "the router's dispatch has been replaced"
ROUTER_MODIFIED = "the application router has been replaced or modified"


def _sneaky_asgi(original=None):
    """An ASGI app answering every HTTP request with 200 (lifespan passed on)."""
    from fastapi.responses import JSONResponse

    calls: list[str] = []

    async def app(scope, receive, send) -> None:
        if scope["type"] != "http" and original is not None:
            await original(scope, receive, send)
            return
        calls.append(scope["path"])
        await JSONResponse({"sneaky": True}, status_code=200)(scope, receive, send)

    return app, calls


def _stack_replacements(router) -> dict:
    import functools
    from types import MethodType

    from fastapi.routing import APIRouter as Router

    genuine = router.middleware_stack
    plain, _ = _sneaky_asgi(genuine)

    async def wrapper(scope, receive, send) -> None:
        await genuine(scope, receive, send)

    async def app(self, scope, receive, send) -> None:  # same name, different function
        return None

    proxy, _ = _forwarding_proxy(genuine)
    return {
        "asgi_app": plain,
        "wrapper_around_genuine": functools.wraps(genuine)(wrapper),
        "partial_of_genuine": functools.partial(genuine),
        "genuine_bound_to_another_router": Router().middleware_stack,
        "other_function_bound_to_this_router": MethodType(app, router),
        "forwarding_proxy": proxy,
        "none": None,
    }


def test_the_genuine_router_and_dispatch_are_accepted(make_settings) -> None:
    import inspect
    from types import MethodType

    from fastapi.routing import APIRouter as Router

    app = _app(make_settings)
    stack = vars(app.router)["middleware_stack"]
    assert type(app.router) is Router
    assert type(stack) is MethodType and stack.__self__ is app.router
    assert stack.__func__ is inspect.getattr_static(Router, "app")
    validate_rate_limit_coverage(app.router)


@pytest.mark.parametrize(
    "kind",
    [
        "asgi_app",
        "wrapper_around_genuine",
        "partial_of_genuine",
        "genuine_bound_to_another_router",
        "other_function_bound_to_this_router",
        "forwarding_proxy",
        "none",
    ],
)
def test_a_replaced_middleware_stack_is_rejected(make_settings, kind: str) -> None:
    app = _app(make_settings)
    app.router.middleware_stack = _stack_replacements(app.router)[kind]
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == DISPATCH_REPLACED


def test_a_missing_middleware_stack_is_rejected(make_settings) -> None:
    app = _app(make_settings)
    del app.router.middleware_stack
    with pytest.raises(RouteCoverageError, match=re.escape(DISPATCH_REPLACED)):
        validate_rate_limit_coverage(app.router)


def test_a_replaced_stack_bypasses_every_limit_and_is_refused(make_settings, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    import backend.main as main

    app = create_app(
        make_settings(rate_limit_enabled=True, rate_limit_parse="1/minute"),
        embedder_factory=CountingFactory(),
    )
    sneaky, calls = _sneaky_asgi(app.router.middleware_stack)
    app.router.middleware_stack = sneaky
    with TestClient(app) as client:  # the threat: all routing replaced, limits included
        assert [client.post("/api/v1/sneaky").status_code for _ in range(3)] == [200] * 3
        assert [client.post("/api/v1/parse").status_code for _ in range(3)] == [200] * 3
    with pytest.raises(RouteCoverageError, match=re.escape(DISPATCH_REPLACED)):
        validate_rate_limit_coverage(app.router)

    installed: list = []
    real_register = main.register_exception_handlers

    def register_and_replace_stack(application) -> None:
        real_register(application)
        stack, stack_calls = _sneaky_asgi(application.router.middleware_stack)
        application.router.middleware_stack = stack
        installed.append(stack_calls)

    monkeypatch.setattr(main, "register_exception_handlers", register_and_replace_stack)
    with pytest.raises(RouteCoverageError) as excinfo:
        _app(make_settings)
    assert str(excinfo.value) == DISPATCH_REPLACED
    assert installed == [[]]  # never called: start-up failed


def test_validation_never_calls_the_middleware_stack(make_settings) -> None:
    app = _app(make_settings)
    sneaky, calls = _sneaky_asgi()
    app.router.middleware_stack = sneaky
    with pytest.raises(RouteCoverageError):
        validate_rate_limit_coverage(app.router)
    assert calls == []


# --- F-N2-4: the router's class and its class-level dispatch -----------------------------------


def test_a_class_swapped_router_with_a_stateful_default_is_rejected(make_settings) -> None:
    """The original attack: genuine fallback for the validator, a malicious one after it."""
    from fastapi.testclient import TestClient

    app = create_app(
        make_settings(rate_limit_enabled=True, rate_limit_parse="1/minute"),
        embedder_factory=CountingFactory(),
    )
    router = app.router
    genuine = router.default
    sneaky, calls = _sneaky_asgi()
    reads: list[int] = []

    class StatefulRouter(type(router)):
        @property
        def default(self):
            reads.append(1)
            return genuine if len(reads) == 1 else sneaky

        @default.setter
        def default(self, value) -> None:
            return None

    del vars(router)["default"]
    router.__class__ = StatefulRouter
    assert router.default is genuine  # the first read: what an attribute-based check sees
    with TestClient(app) as client:  # the vulnerable configuration serves the bypass
        assert [client.post("/api/v1/sneaky").status_code for _ in range(3)] == [200] * 3
    assert calls == ["/api/v1/sneaky"] * 3
    reads.clear()
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(router)
    assert str(excinfo.value) == ROUTER_MODIFIED
    assert reads == []  # the property was never consulted


def test_a_plain_router_subclass_is_rejected(make_settings) -> None:
    app = _app(make_settings)

    class Quiet(type(app.router)):
        pass

    app.router.__class__ = Quiet
    with pytest.raises(RouteCoverageError, match=re.escape(ROUTER_MODIFIED)):
        validate_rate_limit_coverage(app.router)


@pytest.mark.parametrize(
    ("owner", "name"),
    [
        ("fastapi", "default"),
        ("fastapi", "middleware_stack"),
        ("fastapi", "app"),
        ("fastapi", "__call__"),
        ("fastapi", "__getattribute__"),
        ("starlette", "__call__"),
    ],
)
def test_patched_router_classes_are_rejected(make_settings, monkeypatch, owner, name) -> None:
    from fastapi.routing import APIRouter as Router
    from starlette.routing import Router as StarletteRouter

    app = _app(make_settings)
    target = Router if owner == "fastapi" else StarletteRouter
    if name == "__getattribute__":

        def replacement(self, attribute):
            return object.__getattribute__(self, attribute)

    elif name in ("default", "middleware_stack"):
        replacement = property(lambda self: None)
    else:

        async def replacement(self, scope, receive, send) -> None:
            return None

    monkeypatch.setattr(target, name, replacement, raising=False)  # some are new attributes
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_rate_limit_coverage(app.router)
    assert str(excinfo.value) == ROUTER_MODIFIED


def test_router_dispatch_internals_are_as_pinned() -> None:
    """Fails after a FastAPI or Starlette upgrade that changes what F-N2-2/F-N2-4 rely on."""
    import inspect
    from types import MethodType

    import fastapi
    import starlette
    from fastapi import FastAPI
    from fastapi.routing import APIRouter as Router
    from starlette.routing import Router as StarletteRouter

    assert (fastapi.__version__, starlette.__version__) == ("0.162.0", "1.10.0")
    app = FastAPI()
    assert type(app.router) is Router
    assert inspect.getattr_static(Router, "__call__") is StarletteRouter.__call__
    assert "app" in vars(Router) and "__call__" not in vars(Router)
    missing = object()
    assert inspect.getattr_static(Router, "default", missing) is missing
    assert inspect.getattr_static(Router, "middleware_stack", missing) is missing
    stack = vars(app.router)["middleware_stack"]
    assert type(stack) is MethodType and stack.__func__ is vars(Router)["app"]


# --- F-N2-3: the handler that answers an unmatched request's 404 ------------------------------


def _custom_ok_handler():
    from fastapi.responses import JSONResponse

    calls: list[str] = []

    async def handler(request, exc):
        calls.append(request.url.path)
        return JSONResponse({"sneaky": True}, status_code=200)

    return handler, calls


def test_a_custom_404_handler_answers_unmatched_paths_and_is_refused(
    make_settings, monkeypatch
) -> None:
    from fastapi.testclient import TestClient

    import backend.main as main
    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    handler, calls = _custom_ok_handler()
    app.add_exception_handler(404, handler)
    with TestClient(app) as client:  # the threat: 200 for an /api path with no route
        assert [client.post("/api/v1/sneaky").status_code for _ in range(3)] == [200] * 3
    assert calls == ["/api/v1/sneaky"] * 3

    fresh = _app(make_settings)
    fresh.add_exception_handler(404, _custom_ok_handler()[0])
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(fresh)
    assert str(excinfo.value) == "a custom 404 handler can answer unmatched requests"

    real_register = main.register_exception_handlers

    def register_and_add_404(application) -> None:
        real_register(application)
        application.add_exception_handler(404, _custom_ok_handler()[0])

    monkeypatch.setattr(main, "register_exception_handlers", register_and_add_404)
    with pytest.raises(RouteCoverageError, match="a custom 404 handler"):
        _app(make_settings)


@pytest.mark.parametrize("how", ["replaced", "removed"])
def test_a_replaced_http_exception_handler_is_rejected(make_settings, how: str) -> None:
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    if how == "replaced":
        app.add_exception_handler(StarletteHTTPException, _custom_ok_handler()[0])
    else:
        del app.exception_handlers[StarletteHTTPException]  # Starlette's plain-text default
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(app)
    assert str(excinfo.value) == "the HTTPException handler has been replaced"


def test_a_replaced_http_exception_handler_would_answer_200(make_settings) -> None:
    from fastapi.testclient import TestClient
    from starlette.exceptions import HTTPException as StarletteHTTPException

    app = _app(make_settings)
    handler, calls = _custom_ok_handler()
    app.add_exception_handler(StarletteHTTPException, handler)
    with TestClient(app) as client:
        assert client.post("/api/v1/sneaky").status_code == 200
    assert calls == ["/api/v1/sneaky"]


@pytest.mark.parametrize(
    "key",
    [
        "request_validation",
        "fastapi_http_exception_subclass",
        "app_error",
        "exception",
        "status_500",
    ],
)
def test_handlers_that_cannot_answer_the_404_are_accepted(make_settings, key: str) -> None:
    from fastapi.exceptions import HTTPException as FastAPIHTTPException
    from fastapi.exceptions import RequestValidationError
    from fastapi.testclient import TestClient

    from backend.api.rate_limits import validate_unmatched_request_handling
    from backend.core.errors import AppError

    app = _app(make_settings)
    handler, calls = _custom_ok_handler()
    exc_key = {
        "request_validation": RequestValidationError,
        "fastapi_http_exception_subclass": FastAPIHTTPException,
        "app_error": AppError,
        "exception": Exception,
        "status_500": 500,
    }[key]
    app.add_exception_handler(exc_key, handler)
    validate_unmatched_request_handling(app)
    with TestClient(app) as client:
        response = client.post("/api/v1/sneaky")
    assert response.status_code == 404 and response.json()["error"]["code"] == "not_found"
    assert calls == []  # it never saw the unmatched request's 404


def test_a_middleware_stack_built_before_validation_is_rejected(make_settings) -> None:
    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    app.middleware_stack = app.build_middleware_stack()
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(app)
    assert str(excinfo.value) == "the middleware stack was built before validation"


def test_uninspectable_exception_handlers_are_rejected(make_settings) -> None:
    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    app.exception_handlers = list(app.exception_handlers.items())  # type: ignore[assignment]
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(app)
    assert str(excinfo.value) == "exception handlers cannot be inspected"


def test_create_app_validates_unmatched_request_handling(make_settings, monkeypatch) -> None:
    import backend.main as main

    seen: list = []
    real = main.validate_unmatched_request_handling

    def spy(application) -> None:
        seen.append(application)
        real(application)

    monkeypatch.setattr(main, "validate_unmatched_request_handling", spy)
    app = _app(make_settings)
    assert seen == [app]


def test_the_real_application_still_answers_normally(make_settings, tmp_path) -> None:
    from fastapi.testclient import TestClient

    from backend.api.rate_limits import validate_unmatched_request_handling

    (tmp_path / "index.html").write_text("<p>approved</p>", encoding="utf-8")
    app = create_app(
        make_settings(frontend_dir=tmp_path, api_docs_enabled=True, rate_limit_enabled=True),
        embedder_factory=CountingFactory(),
    )
    validate_rate_limit_coverage(app.router)
    validate_unmatched_request_handling(app)
    with TestClient(app) as client:
        # With the frontend mounted at "/", an unmatched GET reaches StaticFiles, which
        # raises the 404 our handler answers (an unmatched POST gets StaticFiles' 405).
        missing = client.get("/api/v1/sneaky")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "not_found"
        assert missing.json()["error"]["request_id"] == missing.headers["X-Request-ID"]
        assert client.get("/").text == "<p>approved</p>"
        assert client.get("/health").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.post("/api/v1/parse").status_code == 415  # ordinary routes still served


# --- A1: the method that builds the application's middleware stack ---------------------------

BUILD_REPLACED = "the application's middleware construction has been replaced"


def _sneaky_build(app):
    """A build_middleware_stack replacement whose stack answers every HTTP request with 200."""
    real = app.build_middleware_stack
    calls: list[str] = []

    def build():
        calls.append("built")
        sneaky, _ = _sneaky_asgi(real())
        return sneaky

    return build, calls


def _build_replacements(app) -> dict:
    import functools
    from types import MethodType

    genuine = app.build_middleware_stack

    class ForwardingProxy:  # reports the genuine method's class, function and binding
        def __getattribute__(self, name):
            return getattr(genuine, name)

        def __call__(self):
            return genuine()

    return {
        "plain_function": _sneaky_build(app)[0],
        "wrapper_around_genuine": functools.wraps(genuine)(lambda: genuine()),
        "partial_of_genuine": functools.partial(FastAPI.build_middleware_stack, app),
        "genuine_bound_to_another_app": MethodType(FastAPI.build_middleware_stack, FastAPI()),
        "other_function_bound_to_this_app": MethodType(lambda self: genuine(), app),
        "forwarding_proxy": ForwardingProxy(),
        "none": None,
    }


def test_the_genuine_middleware_construction_is_accepted(make_settings) -> None:
    from types import MethodType

    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    assert "build_middleware_stack" not in vars(app)
    validate_unmatched_request_handling(app)
    # The genuine function bound to this very app is the same method, so it is allowed.
    app.build_middleware_stack = MethodType(FastAPI.build_middleware_stack, app)
    validate_unmatched_request_handling(app)


@pytest.mark.parametrize(
    "kind",
    [
        "plain_function",
        "wrapper_around_genuine",
        "partial_of_genuine",
        "genuine_bound_to_another_app",
        "other_function_bound_to_this_app",
        "forwarding_proxy",
        "none",
    ],
)
def test_a_replaced_middleware_construction_is_rejected(make_settings, kind: str) -> None:
    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    app.build_middleware_stack = _build_replacements(app)[kind]
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(app)
    assert str(excinfo.value) == BUILD_REPLACED


def test_a_replaced_middleware_construction_bypasses_every_limit_and_is_refused(
    make_settings, monkeypatch
) -> None:
    from fastapi.testclient import TestClient

    import backend.main as main
    from backend.api.rate_limits import validate_unmatched_request_handling

    settings = make_settings(rate_limit_enabled=True, rate_limit_parse="1/minute")
    app = create_app(settings, embedder_factory=CountingFactory())
    build, calls = _sneaky_build(app)
    app.build_middleware_stack = build
    with TestClient(app) as client:  # the threat: the whole application dispatch replaced
        assert [client.post("/api/v1/sneaky").status_code for _ in range(3)] == [200] * 3
        assert [client.post("/api/v1/parse").status_code for _ in range(3)] == [200] * 3
    assert calls == ["built"]

    fresh = create_app(settings, embedder_factory=CountingFactory())
    build, calls = _sneaky_build(fresh)
    fresh.build_middleware_stack = build
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(fresh)
    assert str(excinfo.value) == BUILD_REPLACED
    assert calls == []  # never called by the validator

    installed: list = []
    real_register = main.register_exception_handlers

    def register_and_replace_build(application) -> None:
        real_register(application)
        replacement, build_calls = _sneaky_build(application)
        application.build_middleware_stack = replacement
        installed.append(build_calls)

    monkeypatch.setattr(main, "register_exception_handlers", register_and_replace_build)
    with pytest.raises(RouteCoverageError) as excinfo:
        create_app(settings, embedder_factory=CountingFactory())
    assert str(excinfo.value) == BUILD_REPLACED
    assert installed == [[]]  # start-up failed before any request could build a stack


def test_an_application_subclass_is_rejected(make_settings) -> None:
    from backend.api.rate_limits import validate_unmatched_request_handling

    class Subclassed(FastAPI):
        def build_middleware_stack(self):
            return super().build_middleware_stack()

    app = _app(make_settings)
    app.__class__ = Subclassed
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(app)
    assert str(excinfo.value) == BUILD_REPLACED


@pytest.mark.parametrize("name", ["build_middleware_stack", "__getattribute__"])
def test_a_patched_application_class_is_rejected(make_settings, monkeypatch, name: str) -> None:
    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    original = getattr(FastAPI, name)

    def patched(self, *args):
        return original(self, *args)

    monkeypatch.setattr(FastAPI, name, patched)
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(app)
    assert str(excinfo.value) == BUILD_REPLACED


def test_the_real_application_builds_its_normal_middleware_stack(make_settings) -> None:
    from fastapi.middleware.asyncexitstack import AsyncExitStackMiddleware
    from fastapi.testclient import TestClient
    from starlette.middleware.errors import ServerErrorMiddleware
    from starlette.middleware.exceptions import ExceptionMiddleware

    from backend.core.body_limit import BodySizeLimitMiddleware
    from backend.core.errors import UnhandledExceptionMiddleware
    from backend.core.logging import AccessLogMiddleware
    from backend.core.request_id import RequestIdMiddleware
    from backend.core.security_headers import SecurityHeadersMiddleware

    app = _app(make_settings)
    assert app.middleware_stack is None
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
    layers, node = [], app.middleware_stack
    while node is not app.router:
        layers.append(type(node))
        node = node.app
    assert layers[0] is ServerErrorMiddleware
    assert layers[1].__name__ == "ExceptionTelemetryMiddleware"  # FastAPI-internal
    assert layers[2:] == [
        RequestIdMiddleware,
        SecurityHeadersMiddleware,
        AccessLogMiddleware,
        BodySizeLimitMiddleware,
        UnhandledExceptionMiddleware,
        ExceptionMiddleware,
        AsyncExitStackMiddleware,
    ]


def test_the_application_build_internals_are_as_validated() -> None:
    """Pins the FastAPI 0.162 / Starlette 1.10 behaviour the A1 check relies on."""
    import inspect

    import fastapi
    import starlette
    from starlette.applications import Starlette

    assert (fastapi.__version__, starlette.__version__) == ("0.162.0", "1.10.0")
    assert "build_middleware_stack" in vars(FastAPI)  # FastAPI's own, not Starlette's
    call = inspect.getsource(Starlette.__call__)
    assert "self.middleware_stack = self.build_middleware_stack()" in call
    assert inspect.getattr_static(FastAPI, "__getattribute__") is object.__getattribute__


# --- A3: handler keys sorted exactly as FastAPI and Starlette sort them -----------------------


class _EqualToEverything(int):
    """An int key that FastAPI's ``key in (500, Exception)`` sorts as a 500 handler."""

    def __hash__(self) -> int:
        return hash(404)

    def __eq__(self, other) -> bool:
        return True


class _EqualTo404Only(int):
    """An int key that is not 500 but that Starlette's ``.get(404)`` finds."""

    def __hash__(self) -> int:
        return hash(404)

    def __eq__(self, other) -> bool:
        return other == 404


def _answers_with_approved_404(app) -> bool:
    """Whether an unmatched request really gets the approved envelope and request ID."""
    from fastapi.testclient import TestClient

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/sneaky")
    return (
        response.status_code == 404
        and response.headers.get("content-type", "").startswith("application/json")
        and response.json()["error"]["code"] == "not_found"
        and response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
    )


def test_a_key_equal_to_500_is_sorted_as_fastapi_sorts_it(make_settings) -> None:
    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    handler, calls = _custom_ok_handler()
    app.exception_handlers[_EqualToEverything(7)] = handler
    validate_unmatched_request_handling(app)  # FastAPI sends it to ServerErrorMiddleware
    assert _answers_with_approved_404(app)
    assert calls == []


def test_a_key_equal_only_to_404_is_a_404_handler(make_settings) -> None:
    from backend.api.rate_limits import validate_unmatched_request_handling

    app = _app(make_settings)
    handler, calls = _custom_ok_handler()
    app.exception_handlers[_EqualTo404Only(7)] = handler
    with pytest.raises(RouteCoverageError) as excinfo:
        validate_unmatched_request_handling(app)
    assert str(excinfo.value) == "a custom 404 handler can answer unmatched requests"
    assert not _answers_with_approved_404(app)  # the threat is real: Starlette finds it
    assert calls == ["/api/v1/sneaky"]


@pytest.mark.parametrize(
    "config",
    [
        "approved",
        "status_404",
        "status_404_none",
        "http_exception_replaced",
        "http_exception_removed",
        "starlette_http_exception_subclass",
        "fastapi_http_exception_subclass",
        "exception",
        "status_500",
        "request_validation",
        "app_error",
        "key_equal_to_everything",
        "key_equal_only_to_404",
    ],
)
def test_the_validator_agrees_with_the_framework(make_settings, config: str) -> None:
    """The decision matches what the framework really does with the unmatched 404."""
    from fastapi.exceptions import HTTPException as FastAPIHTTPException
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    from backend.api.rate_limits import validate_unmatched_request_handling
    from backend.core.errors import AppError

    class StarletteSubclass(StarletteHTTPException):
        pass

    keys = {
        "status_404": 404,
        "status_404_none": 404,
        "http_exception_replaced": StarletteHTTPException,
        "starlette_http_exception_subclass": StarletteSubclass,
        "fastapi_http_exception_subclass": FastAPIHTTPException,
        "exception": Exception,
        "status_500": 500,
        "request_validation": RequestValidationError,
        "app_error": AppError,
        "key_equal_to_everything": _EqualToEverything(7),
        "key_equal_only_to_404": _EqualTo404Only(7),
    }
    handler, calls = _custom_ok_handler()
    validated, served = _app(make_settings), _app(make_settings)
    for app in (validated, served):
        if config == "http_exception_removed":
            del app.exception_handlers[StarletteHTTPException]
        elif config in keys:
            app.exception_handlers[keys[config]] = None if config == "status_404_none" else handler
    try:
        validate_unmatched_request_handling(validated)
        accepted = True
    except RouteCoverageError:
        accepted = False
    approved = _answers_with_approved_404(served) and calls == []
    assert accepted is approved
