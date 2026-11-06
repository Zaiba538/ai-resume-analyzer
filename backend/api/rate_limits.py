"""Rate limiting and the /analyze concurrency gate (M10).

Rate limits. ``enforce_rate_limit`` is attached in ``create_app`` with
``include_router(..., dependencies=...)``, so it runs after routing and before the
endpoint's own dependencies and handler. No endpoint declares a FastAPI body
parameter, so a request is checked before its body is read: a limited request
never reaches parsing, the report store or an LLM. The policy of the matched route
comes from ``ROUTE_POLICIES``; the app's ``RateLimiter`` checks the per-client and
global limits of the policy atomically. The client is identified only by
``request.client.host``: uvicorn has already applied its trusted-proxy rules, and
forwarding headers are never read here. A rejected request gets 429
``rate_limited`` with ``Retry-After``. ``RATE_LIMIT_ENABLED=false`` turns this
check off.

Concurrency. At most ``ANALYZE_MAX_CONCURRENT`` analyses run at a time. The
/analyze endpoint receives an ``AnalysisSlot`` from the function-scoped
``analysis_slot`` dependency, but acquires it only once the whole upload has been
received and before any document parsing or agent work, so a slow or stalled
upload never occupies a slot (audit finding H1). When no slot is free the request
is rejected at once with 429 ``analysis_capacity_exceeded`` and ``Retry-After: 10``.
The slot is released when the analysis ends, however it ends, and again (as a
no-op) when the dependency exits, which happens before the response is sent. The
gate stays active when rate limiting is disabled: it protects server resources,
not quota. The rate limit runs first, so a rate-limited request never acquires a
slot; a request rejected as busy has been counted against its quota.

Coverage. ``validate_rate_limit_coverage`` inspects the application's assembled
route table at the end of ``create_app`` (audit finding M1): every route under
``/api/`` must be a FastAPI route with a policy and the expected methods, run
``enforce_rate_limit`` as its first dependency, and /analyze (only) must have the
``analysis_slot`` dependency. Anything else fails start-up.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from string import hexdigits
from types import MethodType
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Request
from fastapi.routing import APIRoute, APIRouter, iter_route_contexts
from starlette.convertors import (
    FloatConvertor,
    IntegerConvertor,
    StringConvertor,
    UUIDConvertor,
)
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import PARAM_REGEX, BaseRoute, Host, Mount, Router
from starlette.staticfiles import StaticFiles

from backend.api.deps import get_analyze_gate, get_app_settings, get_rate_limiter
from backend.core import rate_limit
from backend.core.errors import AppError, http_exception_handler

RATE_LIMITED_MESSAGE = "Too many requests. Please try again later."
ANALYSIS_BUSY_MESSAGE = "The server is busy with other analyses. Please try again shortly."
ANALYSIS_BUSY_RETRY_AFTER_SECONDS = 10

API_PREFIX = "/api"
ANALYZE_PATH = "/api/v1/analyze"
# The only mount allowed: the static frontend at "/" (see create_app).
FRONTEND_MOUNT_NAME = "frontend"

# Route path template -> rate-limit policy. Every route under /api/ must appear
# here with exactly the methods in ROUTE_METHODS (checked at start-up). Health,
# readiness, documentation and the static frontend are outside /api/ and exempt.
ROUTE_POLICIES: dict[str, str] = {
    "/api/v1/parse": rate_limit.PARSE,
    "/api/v1/match": rate_limit.MATCH,
    ANALYZE_PATH: rate_limit.ANALYZE,
    "/api/v1/report/{report_id}": rate_limit.REPORT,
    "/api/v1/report/{report_id}/pdf": rate_limit.REPORT_PDF,
    "/api/v1/interview/questions": rate_limit.INTERVIEW_QUESTIONS,
    "/api/v1/interview/evaluate": rate_limit.INTERVIEW_EVALUATE,
}
ROUTE_METHODS: dict[str, frozenset[str]] = {
    "/api/v1/parse": frozenset({"POST"}),
    "/api/v1/match": frozenset({"POST"}),
    ANALYZE_PATH: frozenset({"POST"}),
    "/api/v1/report/{report_id}": frozenset({"GET"}),
    "/api/v1/report/{report_id}/pdf": frozenset({"GET"}),
    "/api/v1/interview/questions": frozenset({"POST"}),
    "/api/v1/interview/evaluate": frozenset({"POST"}),
}

RATE_LIMIT_RESPONSE_DESCRIPTION = "Too many requests (`rate_limited`); see `Retry-After`"
ANALYZE_429_DESCRIPTION = (
    "Too many requests (`rate_limited`), or the maximum number of concurrent analyses "
    "is running (`analysis_capacity_exceeded`); see `Retry-After`"
)


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"
    message = RATE_LIMITED_MESSAGE

    def __init__(self, retry_after_seconds: int) -> None:
        super().__init__(headers={"Retry-After": str(retry_after_seconds)})


class ConcurrencyLimitedError(AppError):
    status_code = 429
    code = "analysis_capacity_exceeded"
    message = ANALYSIS_BUSY_MESSAGE

    def __init__(self) -> None:
        super().__init__(headers={"Retry-After": str(ANALYSIS_BUSY_RETRY_AFTER_SECONDS)})


class RouteCoverageError(RuntimeError):
    """Start-up failure: an API route is not protected as required."""


class AnalyzeConcurrencyGate:
    """Counts in-flight analyses; one instance per application.

    ``try_acquire`` and ``release`` never await, so on the event loop each is
    atomic. Requests are rejected rather than queued when the gate is full.
    """

    def __init__(self, max_concurrent: int) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self._max = max_concurrent
        self._in_flight = 0

    @property
    def max_concurrent(self) -> int:
        return self._max

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def try_acquire(self) -> bool:
        if self._in_flight >= self._max:
            return False
        self._in_flight += 1
        return True

    def release(self) -> None:
        if self._in_flight <= 0:
            raise RuntimeError("release without a matching acquire")
        self._in_flight -= 1


class AnalysisSlot:
    """One request's handle on the gate: acquired by the endpoint, released once."""

    def __init__(self, gate: AnalyzeConcurrencyGate) -> None:
        self._gate = gate
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    def acquire(self) -> None:
        """Take a slot, or raise ``ConcurrencyLimitedError`` (429) when none is free."""
        if self._held:
            raise RuntimeError("analysis slot already held")
        if not self._gate.try_acquire():
            raise ConcurrencyLimitedError()
        self._held = True

    def release(self) -> None:
        """Give the slot back; does nothing when it is not held."""
        if self._held:
            self._held = False
            self._gate.release()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold a slot for the duration of the block, whatever ends it."""
        self.acquire()
        try:
            yield
        finally:
            self.release()


def _policy_for(request: Request) -> str:
    route = request.scope.get("route")
    path = getattr(route, "path", None)
    try:
        return ROUTE_POLICIES[path]  # type: ignore[index]
    except KeyError:
        # Unreachable when start-up validation passed; fail closed (500).
        raise RuntimeError("route has no rate-limit policy") from None


async def enforce_rate_limit(request: Request) -> None:
    """Count the request against its route's policy; 429 when over a limit."""
    if not get_app_settings(request).rate_limit_enabled:
        return
    policy = _policy_for(request)
    host = request.client.host if request.client else None
    decision = get_rate_limiter(request).hit(policy, host)
    if not decision.allowed:
        raise RateLimitedError(decision.retry_after_seconds)


async def analysis_slot(request: Request) -> AsyncIterator[AnalysisSlot]:
    """An unacquired slot for the endpoint; always released when the endpoint ends.

    Must be used with ``scope="function"`` so that the release happens before the
    response is sent.
    """
    slot = AnalysisSlot(get_analyze_gate(request))
    try:
        yield slot
    finally:
        slot.release()


AnalysisSlotDep = Annotated[AnalysisSlot, Depends(analysis_slot, scope="function")]


def _is_api_path(path: str) -> bool:
    return path == API_PREFIX or path.startswith(API_PREFIX + "/")


@dataclass(frozen=True)
class _Param:
    """What one path parameter can match: allowed characters and minimum length."""

    allows: Callable[[str], bool]
    min_length: int


_DIGITS = frozenset("0123456789")
_FLOAT_CHARS = _DIGITS | {"."}
_UUID_CHARS = frozenset(hexdigits) | {"-"}


def _param_for(convertor: Any) -> _Param:
    """A safe over-approximation of a Starlette converter's language.

    ``path`` and any converter not listed here (custom converters included) are
    treated as matching anything, so an unknown converter can only cause a route
    to be rejected, never let it through.
    """
    kind = type(convertor)
    if kind is StringConvertor:  # [^/]+
        return _Param(lambda c: c != "/", 1)
    if kind is IntegerConvertor:  # [0-9]+
        return _Param(lambda c: c in _DIGITS, 1)
    if kind is FloatConvertor:  # [0-9]+(\.[0-9]+)?
        return _Param(lambda c: c in _FLOAT_CHARS, 1)
    if kind is UUIDConvertor:  # hex digits and dashes
        return _Param(lambda c: c in _UUID_CHARS, 1)
    return _Param(lambda c: True, 0)  # PathConvertor (.*) and unknown converters


def _template_tokens(template: str, convertors: Mapping[str, Any]) -> list[str | _Param]:
    """Split a route template into literal text and parameters, as Starlette does."""
    tokens: list[str | _Param] = []
    position = 0
    for match in PARAM_REGEX.finditer(template):
        if match.start() > position:
            tokens.append(template[position : match.start()])
        tokens.append(_param_for(convertors.get(match.group(1))))
        position = match.end()
    if position < len(template):
        tokens.append(template[position:])
    return tokens


def _can_produce(tokens: Sequence[str | _Param], target: str, *, exact: bool) -> bool:
    """Whether some path matched by ``tokens`` equals ``target`` (``exact``) or starts with it."""
    end = len(target)

    def walk(index: int, position: int) -> bool:
        if position == end:
            if not exact:
                return True  # any remaining tokens can always match something
            return all(isinstance(t, _Param) and t.min_length == 0 for t in tokens[index:])
        if index == len(tokens):
            return False
        token = tokens[index]
        rest = target[position:]
        if isinstance(token, str):
            if len(token) <= len(rest):
                return rest.startswith(token) and walk(index + 1, position + len(token))
            return not exact and token.startswith(rest)
        for taken in range(len(rest) + 1):  # the parameter takes the next `taken` characters
            if taken and not token.allows(rest[taken - 1]):
                break
            if taken == len(rest):
                if not exact:
                    return True  # it may go on matching past the target
                if taken >= token.min_length and walk(index + 1, end):
                    return True
            elif taken >= token.min_length and walk(index + 1, position + taken):
                return True
        return False

    return walk(0, 0)


def _can_match_api_paths(template: str, convertors: Mapping[str, Any]) -> bool:
    """Whether a route with this template can match ``/api`` or any path under ``/api/``."""
    tokens = _template_tokens(template, convertors)
    return _can_produce(tokens, API_PREFIX + "/", exact=False) or _can_produce(
        tokens, API_PREFIX, exact=True
    )


def _is_frontend_mount(route: BaseRoute) -> bool:
    """The approved static frontend: ``StaticFiles`` itself (no routes, no wrapper) at "/"."""
    return (
        isinstance(route, Mount)
        and route.path == ""
        and route.name == FRONTEND_MOUNT_NAME
        and type(route.app) is StaticFiles
    )


def _low_priority_routes(router: APIRouter) -> list[Any]:
    """Every low-priority route the router would try after its ordinary routes.

    FastAPI 0.162 keeps the routes registered with ``app.frontend()`` and
    ``APIRouter.frontend()`` outside ``router.routes``: in the private
    ``_low_priority_routes`` lists, which the router reads, with those of every
    included router, through the private ``_iter_low_priority_routes()``. This
    depends on those internals (tests pin them); if they ever disappear, start-up
    fails rather than skip the check.
    """
    iterate = getattr(router, "_iter_low_priority_routes", None)
    if not callable(iterate):
        raise RouteCoverageError("low-priority routes cannot be inspected")
    return list(iterate())


# Starlette's built-in not-found fallback, captured when this module loads, so a
# later patch of ``Router.not_found`` cannot make a replacement look built-in.
_BUILTIN_NOT_FOUND = Router.not_found

# The router class FastAPI 0.162 creates for every application, and the class-level
# pieces of its request dispatch, captured when this module loads (audit findings
# F-N2-2 and F-N2-4). Starlette 1.10's ``Router.__call__`` dispatches through the
# instance attribute ``middleware_stack``, which ``Router.__init__`` sets to the
# bound ``app``; FastAPI's ``APIRouter.app`` calls the instance attribute
# ``default`` for requests no route matches. Neither class defines ``default`` or
# ``middleware_stack``, so the instance values are exactly what dispatch reads.
_MISSING = object()
_ROUTER_TYPE = APIRouter
_ROUTER_CALL = inspect.getattr_static(APIRouter, "__call__")
_ROUTER_APP = inspect.getattr_static(APIRouter, "app")
_ROUTER_GETATTRIBUTE = inspect.getattr_static(APIRouter, "__getattribute__")
_INSTANCE_ONLY_ATTRIBUTES = ("default", "middleware_stack")


def _router_class_is_genuine(router: APIRouter) -> bool:
    """Whether the router is a plain ``APIRouter`` whose dispatch has not been altered.

    The exact type is required: a subclass, or a swapped ``__class__``, could
    override dispatch or turn ``default`` into a property that answers the
    validator differently from later requests. The class-level dispatch functions
    must be the captured originals (no patching), and the class must not define
    ``default`` or ``middleware_stack``, which would shadow the instance values.
    ``inspect.getattr_static`` reads them without running any descriptor.
    """
    if type(router) is not _ROUTER_TYPE:
        return False
    static = inspect.getattr_static
    if (
        static(_ROUTER_TYPE, "__call__", _MISSING) is not _ROUTER_CALL
        or static(_ROUTER_TYPE, "app", _MISSING) is not _ROUTER_APP
        or static(_ROUTER_TYPE, "__getattribute__", _MISSING) is not _ROUTER_GETATTRIBUTE
    ):
        return False
    return all(
        static(_ROUTER_TYPE, name, _MISSING) is _MISSING for name in _INSTANCE_ONLY_ATTRIBUTES
    )


def _instance_attribute(router: APIRouter, name: str) -> Any:
    """An attribute from the instance dictionary, without dynamic attribute lookup.

    Only used once ``_router_class_is_genuine`` holds, so ``vars()`` is the
    ordinary instance ``__dict__`` and nothing on the class can intercept it.
    """
    return vars(router).get(name)


def _has_genuine_dispatch(router: APIRouter) -> bool:
    """Whether ``middleware_stack`` is still the bound ``APIRouter.app`` of this router.

    Replacing it replaces all routing, every rate limit included (audit finding
    F-N2-2). Exact ``MethodType`` (as for the fallback, F-N2-1), the captured
    function, and this very router; the stack is never called here.
    """
    stack = _instance_attribute(router, "middleware_stack")
    return type(stack) is MethodType and stack.__func__ is _ROUTER_APP and stack.__self__ is router


# FastAPI 0.162's own ``build_middleware_stack``, captured when this module loads (audit
# finding A1). On the first request Starlette 1.10's ``__call__`` runs
# ``self.middleware_stack = self.build_middleware_stack()``, so whatever that lookup
# resolves to builds the whole application dispatch, every rate limit included.
_APP_TYPE = FastAPI
_APP_BUILD = inspect.getattr_static(FastAPI, "build_middleware_stack")
_APP_GETATTRIBUTE = inspect.getattr_static(FastAPI, "__getattribute__")


def _builds_genuine_middleware_stack(app: FastAPI) -> bool:
    """Whether the first request will build the stack with FastAPI's own method.

    The exact type is required (a subclass could override the method), and the
    class-level method and attribute lookup must be the captured originals, read
    without running any descriptor. A value in the instance dictionary shadows
    the method, so it may only be the genuine function bound to this very app.
    The method is never called here.
    """
    if type(app) is not _APP_TYPE:
        return False
    static = inspect.getattr_static
    if (
        static(_APP_TYPE, "build_middleware_stack", _MISSING) is not _APP_BUILD
        or static(_APP_TYPE, "__getattribute__", _MISSING) is not _APP_GETATTRIBUTE
    ):
        return False
    override = vars(app).get("build_middleware_stack", _MISSING)
    return override is _MISSING or (
        type(override) is MethodType
        and override.__func__ is _APP_BUILD
        and override.__self__ is app
    )


def _has_builtin_fallback(router: APIRouter) -> bool:
    """Whether the router's fallback is still Starlette's own ``not_found``, bound to it.

    Starlette 1.10 sets ``router.default = self.not_found`` when no default is given
    (``FastAPI()`` offers none), and calls ``router.default`` for every request no
    route matches. Anything else there could serve /api/ paths outside the route
    table (audit finding N2). The check uses identity only and never calls the
    fallback: a bound method of exactly the captured function, bound to this very
    router.

    The type is compared exactly (``type(fallback) is MethodType``), not with
    ``isinstance()``: ``isinstance`` also consults ``__class__``, which a proxy can
    forward along with ``__func__`` and ``__self__`` while its own ``__call__`` runs
    instead (audit finding F-N2-1). ``type()`` reports the real type, and
    ``MethodType`` cannot be subclassed, so only a genuine bound method passes.
    """
    fallback = _instance_attribute(router, "default")
    return (
        type(fallback) is MethodType
        and fallback.__func__ is _BUILTIN_NOT_FOUND
        and fallback.__self__ is router
    )


def validate_rate_limit_coverage(router: APIRouter) -> None:
    """Fail unless every route the router can serve under /api/ is protected as required.

    ``router`` is an application's router (``app.router``); included routers are
    expanded to the routes FastAPI actually registers, with their combined
    dependencies, so a router or a route added any other way is checked too.

    The router's not-found fallback must be Starlette's built-in one (audit finding
    N2): a replacement could serve any path, /api/ included, without a route.

    Low-priority routes (FastAPI's ``frontend()`` feature, audit finding N1) are
    not part of the approved architecture, which serves its frontend through one
    validated ``StaticFiles`` mount; any of them, at any prefix and in any
    included router, fails start-up.

    A mounted application or a ``Host`` route could serve /api/ paths that this
    table does not show (audit finding M1-a), so neither is allowed anywhere: the
    only mount accepted is the approved static frontend.

    A route is judged by the paths it can match, not by how its template starts
    (audit finding F1): a template outside /api/ with parameters, such as
    ``/{rest:path}`` or ``/{prefix}/v1/x``, may still match /api/ paths, so it is
    rejected unless its parameters cannot produce them. Each API path and method
    may be registered only once, so no duplicate can shadow an approved route.
    """
    if _low_priority_routes(router):
        raise RouteCoverageError("low-priority routes such as frontend() are not allowed")
    if not _router_class_is_genuine(router):
        raise RouteCoverageError("the application router has been replaced or modified")
    if not _has_genuine_dispatch(router):
        raise RouteCoverageError("the router's dispatch has been replaced")
    if not _has_builtin_fallback(router):
        raise RouteCoverageError("the router's not-found fallback has been replaced")
    methods: dict[str, set[str]] = {}
    frontend_mounts = 0
    for context in iter_route_contexts(router.routes):
        original = context.original_route
        if isinstance(original, Host):
            raise RouteCoverageError("host routes are not allowed")
        path = context.path or ""
        if not _is_api_path(path):
            if isinstance(original, Mount):
                if not _is_frontend_mount(original):
                    raise RouteCoverageError("only the static frontend may be mounted")
                frontend_mounts += 1
                if frontend_mounts > 1:
                    raise RouteCoverageError("only one static frontend may be mounted")
            convertors = getattr(context, "param_convertors", None) or {}
            if _can_match_api_paths(path, convertors):
                raise RouteCoverageError("a route outside /api can match API paths")
            continue
        if not isinstance(context.original_route, APIRoute):
            raise RouteCoverageError(
                "an API route is not a FastAPI route and cannot be rate limited"
            )
        if path not in ROUTE_POLICIES:
            raise RouteCoverageError("an API route has no rate-limit policy")
        dependencies = context.dependant.dependencies
        if not dependencies or dependencies[0].call is not enforce_rate_limit:
            raise RouteCoverageError("an API route does not run the rate limit first")
        slots = [d for d in dependencies if d.call is analysis_slot]
        if path == ANALYZE_PATH:
            if len(slots) != 1 or slots[0].scope != "function":
                raise RouteCoverageError("the analyze route lacks its concurrency dependency")
        elif slots:
            raise RouteCoverageError("only the analyze route may use the concurrency dependency")
        registered = methods.setdefault(path, set())
        if registered & set(context.methods or ()):
            raise RouteCoverageError("an API route is registered more than once")
        registered.update(context.methods or ())
    if methods.keys() != ROUTE_POLICIES.keys():
        raise RouteCoverageError("a rate-limit policy names an unknown route")
    for path, seen in methods.items():
        if seen != ROUTE_METHODS[path]:
            raise RouteCoverageError("an API route has unexpected methods")


def validate_unmatched_request_handling(app: FastAPI) -> None:
    """Fail unless the 404 for an unmatched request is answered by the approved handler.

    Starlette's built-in ``Router.not_found`` raises ``HTTPException(404)``. In
    Starlette 1.10 ``ExceptionMiddleware`` answers it with the status-code handler
    for 404 if there is one, otherwise with the first handler found along
    ``type(exc).__mro__`` (``HTTPException``, ``Exception``, ...); FastAPI 0.162
    sends handlers keyed ``500`` or ``Exception`` to ``ServerErrorMiddleware``
    instead. A custom handler on either path could answer any unmatched /api/
    path, with any status (audit finding F-N2-3). This resolves the handler the
    same way and requires the approved ``http_exception_handler``. Handlers that
    cannot receive this exception (request validation, ``AppError``, subclasses
    of ``HTTPException``) are left alone.

    The middleware stack is built from ``app.exception_handlers`` on the first
    request; a stack built earlier could hold handlers this check cannot see, so
    it must not exist yet, and it must be built by FastAPI's own method, since a
    replacement could return any application at all (audit finding A1).
    """
    if not _builds_genuine_middleware_stack(app):
        raise RouteCoverageError("the application's middleware construction has been replaced")
    if app.middleware_stack is not None:
        raise RouteCoverageError("the middleware stack was built before validation")
    handlers = app.exception_handlers
    if type(handlers) is not dict:
        raise RouteCoverageError("exception handlers cannot be inspected")
    status_handlers: dict[int, Any] = {}
    class_handlers: dict[Any, Any] = {StarletteHTTPException: None}  # Starlette's default
    for key, handler in handlers.items():
        # FastAPI's and Starlette's own expressions, equality semantics included, so a
        # key that merely compares equal to 500 is sorted exactly as they sort it.
        if key in (500, Exception):
            continue  # ServerErrorMiddleware: never sees an HTTPException
        if isinstance(key, int):
            status_handlers[key] = handler
        else:
            class_handlers[key] = handler
    if status_handlers.get(404) is not None:
        raise RouteCoverageError("a custom 404 handler can answer unmatched requests")
    effective = next(
        (class_handlers[cls] for cls in StarletteHTTPException.__mro__ if cls in class_handlers),
        None,
    )
    if effective is not http_exception_handler:
        raise RouteCoverageError("the HTTPException handler has been replaced")
