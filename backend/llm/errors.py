"""Typed errors for LLM calls, mapped from the groq SDK.

Messages are fixed and user-safe. Upstream response bodies, which may echo
prompt content or contain failed generations, are never included or logged.

``upstream_diagnostics`` extracts only metadata for the logs (Phase 3, M3): the
exception class, the upstream HTTP status and Groq's ``x-request-id`` response
header. Nothing else is read from the exception: not its message, body,
request or other headers, which can carry prompts, outputs or credentials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import groq

from backend.core.errors import AppError

# Groq request IDs are short ASCII tokens (for example "req_01j..."). Any other
# value is dropped rather than logged.
_REQUEST_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_REQUEST_ID_HEADER = "x-request-id"


class LLMError(AppError):
    status_code = 502
    code = "upstream_error"
    message = "The AI service returned an error. Please try again."


class LLMNotConfiguredError(LLMError):
    status_code = 503
    code = "llm_not_configured"
    message = "The AI service is not configured."


class UpstreamTimeoutError(LLMError):
    status_code = 504
    code = "upstream_timeout"
    message = "The AI service did not respond in time. Please try again."


class UpstreamRateLimitedError(LLMError):
    status_code = 503
    code = "upstream_rate_limited"
    message = "The AI service is busy. Please try again shortly."


class UpstreamAuthError(LLMError):
    status_code = 502
    code = "upstream_auth_failed"
    message = "The AI service rejected the server's credentials."


class InvalidLLMOutputError(LLMError):
    status_code = 502
    code = "invalid_llm_output"
    message = "The AI service returned an invalid response. Please try again."


# Groq returns HTTP 400 with this code when a strict-schema generation fails;
# it is a model output problem, so it is handled like a schema violation.
JSON_VALIDATE_FAILED = "json_validate_failed"


def is_schema_generation_failure(exc: groq.BadRequestError) -> bool:
    body = exc.body if isinstance(exc.body, dict) else {}
    error = body.get("error", body) if isinstance(body, dict) else {}
    return isinstance(error, dict) and error.get("code") == JSON_VALIDATE_FAILED


def map_groq_error(exc: groq.GroqError) -> LLMError:
    """Translate a groq SDK exception into a typed, user-safe error."""
    if isinstance(exc, groq.APITimeoutError):
        return UpstreamTimeoutError()
    if isinstance(exc, groq.RateLimitError):
        retry_after = exc.response.headers.get("retry-after")
        headers = {"Retry-After": retry_after} if retry_after and retry_after.isdigit() else None
        return UpstreamRateLimitedError(headers=headers)
    if isinstance(exc, groq.AuthenticationError | groq.PermissionDeniedError):
        return UpstreamAuthError()
    return LLMError()


@dataclass(frozen=True)
class UpstreamDiagnostics:
    """Log-safe metadata of a failed upstream call."""

    error_type: str
    http_status: int | None = None
    request_id: str | None = None

    def log_fields(self) -> dict[str, str | int | None]:
        """Fields for ``extra``; each is in the log formatter's allowlist."""
        return {
            "error_type": self.error_type,
            "upstream_http_status": self.http_status,
            "upstream_request_id": self.request_id,
        }


def upstream_diagnostics(exc: BaseException) -> UpstreamDiagnostics:
    """Metadata of an upstream failure; missing or malformed values become None.

    Never raises: diagnostics must not replace the typed error being reported.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
        status = None

    request_id: str | None = None
    try:
        headers = getattr(getattr(exc, "response", None), "headers", None)
        value = headers.get(_REQUEST_ID_HEADER) if headers is not None else None
    except Exception:  # malformed response object: report what is known
        value = None
    if isinstance(value, str) and _REQUEST_ID.fullmatch(value):
        request_id = value

    return UpstreamDiagnostics(type(exc).__name__, status, request_id)


def monitor_code(error: LLMError) -> str:
    """Error code reported to the upstream monitor as a passive signal."""
    return {
        UpstreamTimeoutError: "timeout",
        UpstreamRateLimitedError: "rate_limited",
        UpstreamAuthError: "auth_failed",
    }.get(type(error), "upstream_error")
