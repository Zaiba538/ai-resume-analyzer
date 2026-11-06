"""Error handling: one JSON envelope for every error response.

Shape: ``{"error": {"code", "message", "request_id", "details"?}}``.

User input is never echoed:

* ``HTTPException`` responses use the standard status phrase, not ``detail``.
* Validation errors keep only ``loc``, ``type`` and ``msg``. Pydantic's
  ``input`` and ``ctx`` fields, which can contain submitted data, are dropped.
* Unhandled exceptions become a generic 500. The exception message is neither
  returned nor logged.

Application code raises ``AppError`` (or subclasses) for expected failures with
a stable machine-readable ``code`` and a safe, fixed ``message``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.core.request_id import get_request_id
from backend.schemas.common import ErrorBody, ErrorDetail, ErrorEnvelope

logger = logging.getLogger("backend.errors")

STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "upstream_error",
    503: "service_unavailable",
    504: "upstream_timeout",
}

_MAX_VALIDATION_DETAILS = 20
_MAX_LOC_PARTS = 10


class AppError(Exception):
    """Expected application failure with a stable code and a safe message."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An internal error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: Sequence[ErrorDetail] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = list(details) if details else None
        self.headers = dict(headers) if headers else None
        super().__init__(self.code)


def code_for_status(status_code: int) -> str:
    return STATUS_CODES.get(status_code, "client_error" if status_code < 500 else "server_error")


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    details: Sequence[ErrorDetail] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    envelope = ErrorEnvelope(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=get_request_id() or "unavailable",
            details=list(details) if details else None,
        )
    )
    return JSONResponse(
        envelope.model_dump(exclude_none=True),
        status_code=status_code,
        headers=dict(headers) if headers else None,
    )


def _status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Error"


async def app_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)  # noqa: S101 - registered for AppError only
    logger.info("app_error", extra={"error_code": exc.code, "status": exc.status_code})
    return error_response(
        exc.status_code, exc.code, exc.message, details=exc.details, headers=exc.headers
    )


async def http_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101
    return error_response(
        exc.status_code,
        code_for_status(exc.status_code),
        _status_phrase(exc.status_code),
        headers=exc.headers,
    )


async def validation_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101
    details = [
        ErrorDetail(
            loc=[str(part) for part in error.get("loc", ())][:_MAX_LOC_PARTS],
            type=str(error.get("type", "invalid")),
            msg=str(error.get("msg", "Invalid value")),
        )
        for error in exc.errors()[:_MAX_VALIDATION_DETAILS]
    ]
    return error_response(422, "validation_error", "Request validation failed.", details=details)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)


class UnhandledExceptionMiddleware:
    """Convert unexpected exceptions into a generic 500 envelope.

    Starlette's ServerErrorMiddleware sits outside all user middleware, so its
    500 responses would miss the request ID and security headers. This
    middleware is installed inside them instead.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def send_tracking_start(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, send_tracking_start)
        except Exception:
            logger.error(
                "unhandled_exception", exc_info=True, extra={"error_code": "internal_error"}
            )
            if response_started:
                raise
            response = error_response(500, "internal_error", "An internal error occurred.")
            await response(scope, receive, send)
