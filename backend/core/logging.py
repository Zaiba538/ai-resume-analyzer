"""Privacy-preserving structured logging.

Policy (approved architecture, privacy model):

* Output is one JSON object per line, built from an allowlist. Nothing outside
  ``ALLOWED_FIELDS`` is emitted, whatever a caller passes in ``extra``.
* Log arguments are never interpolated. The ``event`` field is the message
  template, which is a code constant, and only for loggers in
  ``MESSAGE_LOGGERS``. Other libraries' messages are dropped, because they may
  embed URLs, payloads or exception text.
* Exceptions are reduced to their type and stack frames
  (``file:line:function``). Exception messages are never logged, because they
  can carry user content.
* The access log records the route template (``/api/v1/report/{id}``), never
  the raw path, query string, client address or headers. The uvicorn access log
  is disabled, because it records client IP and full path.
* Third-party LLM tracing is forced off, because prompts contain resume content.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

from starlette.routing import Mount
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.core.request_id import get_request_id

ALLOWED_FIELDS = frozenset(
    {
        "request_id",
        "method",
        "route",
        "status",
        "duration_ms",
        "error_code",
        "component",
        "upstream_status",
        "models_present",
        "file_type",
        "file_size_bytes",
        "page_count",
        "char_count",
        "model",
        "input_tokens",
        "output_tokens",
        "node",
        "timing_ms",
        "critic_verdict",
        "security_flag_types",
    }
)

# Loggers whose message templates are code constants and safe to emit.
MESSAGE_LOGGERS = ("backend", "uvicorn.error")

# Libraries that may log request URLs, payloads or prompts at INFO/DEBUG.
QUIET_LOGGERS = (
    "httpx",
    "httpcore",
    "groq",
    "langchain",
    "langchain_core",
    "langchain_groq",
    "langgraph",
    "langsmith",
    "sentence_transformers",
    "transformers",
    "huggingface_hub",
    "urllib3",
    "filelock",
    "multipart",
    "python_multipart",
    "pypdf",
    "docx",
)

TRACING_ENV_OVERRIDES = {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}

_MAX_STRING = 128
_MAX_LIST = 20
_MAX_FRAMES = 10
_HANDLER_MARKER = "_backend_privacy_handler"
_KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
_DROP = object()

access_logger = logging.getLogger("backend.access")


def _message_allowed(logger_name: str) -> bool:
    return any(logger_name == p or logger_name.startswith(p + ".") for p in MESSAGE_LOGGERS)


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return value if len(value) <= _MAX_STRING else _DROP
    if isinstance(value, list | tuple | frozenset | set):
        items = list(value)
        if len(items) <= _MAX_LIST and all(
            isinstance(i, str) and len(i) <= _MAX_STRING for i in items
        ):
            return sorted(items) if isinstance(value, set | frozenset) else items
    return _DROP


class PrivacyJsonFormatter(logging.Formatter):
    """Allowlist-based JSON formatter (see module docstring)."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
        }
        if _message_allowed(record.name) and isinstance(record.msg, str):
            entry["event"] = record.msg

        request_id = get_request_id()
        if request_id:
            entry["request_id"] = request_id

        for key in ALLOWED_FIELDS:
            if key in record.__dict__:
                value = _safe_value(record.__dict__[key])
                if value is not _DROP:
                    entry[key] = value

        if record.exc_info and record.exc_info[0] is not None:
            exc_type, _, tb = record.exc_info
            entry["exc_type"] = exc_type.__name__
            entry["exc_frames"] = [
                f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
                for frame in traceback.extract_tb(tb)[-_MAX_FRAMES:]
            ]

        return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))


def disable_third_party_tracing() -> None:
    """Force LangSmith/LangChain tracing off, overriding any inherited setting."""
    os.environ.update(TRACING_ENV_OVERRIDES)


def configure_logging(level: str = "INFO", stream: IO[str] | None = None) -> None:
    """Install the privacy formatter on the root logger. Safe to call repeatedly."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, False):
            root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(PrivacyJsonFormatter())
    setattr(handler, _HANDLER_MARKER, True)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)

    logging.getLogger("backend").setLevel(level)

    # Route uvicorn's own loggers through the privacy formatter.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers.clear()
    uvicorn_access.propagate = False
    uvicorn_access.disabled = True

    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def route_template(scope: Scope) -> str:
    """The matched route's path template, never the raw request path."""
    route = scope.get("route")
    if isinstance(route, Mount):
        return f"{route.path}/{{path:path}}"
    path = getattr(route, "path", None)
    return path if isinstance(path, str) else "<unmatched>"


class AccessLogMiddleware:
    """Pure ASGI middleware: one privacy-safe ``request_completed`` record per request."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status = 500

        async def send_capturing_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_capturing_status)
        finally:
            method = scope.get("method", "")
            access_logger.info(
                "request_completed",
                extra={
                    "method": method if method in _KNOWN_METHODS else "OTHER",
                    "route": route_template(scope),
                    "status": status,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
