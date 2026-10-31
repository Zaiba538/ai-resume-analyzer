"""Typed errors for LLM calls, mapped from the groq SDK.

Messages are fixed and user-safe. Upstream response bodies, which may echo
prompt content or contain failed generations, are never included or logged.
"""

from __future__ import annotations

import groq

from backend.core.errors import AppError


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


def monitor_code(error: LLMError) -> str:
    """Error code reported to the upstream monitor as a passive signal."""
    return {
        UpstreamTimeoutError: "timeout",
        UpstreamRateLimitedError: "rate_limited",
        UpstreamAuthError: "auth_failed",
    }.get(type(error), "upstream_error")
