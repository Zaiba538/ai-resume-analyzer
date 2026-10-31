"""Request ID propagation.

Every HTTP request gets an ID that is returned in the ``X-Request-ID`` response
header, included in error envelopes and attached to every log record emitted
while the request is handled. A client-supplied ``X-Request-ID`` is reused only
if it matches a strict character set, which prevents log and header injection.
"""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_HEADER_RAW = REQUEST_ID_HEADER.lower().encode("latin-1")
_VALID_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{7,63}")

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str | None:
    """The ID of the request being handled in the current context, if any."""
    return _request_id.get()


def new_request_id() -> str:
    return uuid.uuid4().hex


def resolve_request_id(raw: bytes | None) -> str:
    """Reuse a well-formed client-supplied ID, otherwise generate a new one."""
    if raw is not None:
        candidate = raw.decode("latin-1")
        if _VALID_REQUEST_ID.fullmatch(candidate):
            return candidate
    return new_request_id()


class RequestIdMiddleware:
    """Pure ASGI middleware: assigns the request ID and echoes it in responses."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw = next(
            (value for name, value in scope["headers"] if name == _REQUEST_ID_HEADER_RAW),
            None,
        )
        request_id = resolve_request_id(raw)
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        token = _request_id.set(request_id)
        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            _request_id.reset(token)
