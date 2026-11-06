"""Privacy JSON formatter: allowlist, no interpolation, no exception messages."""

from __future__ import annotations

import io
import json
import logging
import os

import pytest

from backend.core.logging import (
    TRACING_ENV_OVERRIDES,
    PrivacyJsonFormatter,
    configure_logging,
    disable_third_party_tracing,
)

CANARY = "CANARY-7f3a-resume-text"


def _format(name: str, msg: str, *args, exc_info=None, **extra) -> dict:
    record = logging.LogRecord(name, logging.INFO, __file__, 1, msg, args, exc_info)
    record.__dict__.update(extra)
    return json.loads(PrivacyJsonFormatter().format(record))


def test_backend_event_template_emitted_without_args() -> None:
    entry = _format("backend.test", "user_uploaded %s", CANARY)
    assert entry["event"] == "user_uploaded %s"
    assert CANARY not in json.dumps(entry)
    assert entry["level"] == "INFO" and entry["logger"] == "backend.test"


def test_third_party_messages_dropped() -> None:
    entry = _format("httpx", f"HTTP Request: POST https://x/?q={CANARY}")
    assert "event" not in entry
    assert CANARY not in json.dumps(entry)


def test_only_allowlisted_extra_fields_emitted() -> None:
    entry = _format(
        "backend.test",
        "evt",
        status=200,
        route="/api/v1/report/{id}",
        resume_text=CANARY,
        client_ip="203.0.113.9",
        headers={"authorization": CANARY},
    )
    assert entry["status"] == 200
    assert entry["route"] == "/api/v1/report/{id}"
    for forbidden in ("resume_text", "client_ip", "headers"):
        assert forbidden not in entry
    assert CANARY not in json.dumps(entry)


@pytest.mark.parametrize(
    "value",
    ["x" * 129, {"nested": "dict"}, ["ok"] * 21, [1, 2], object()],
)
def test_unsafe_allowlisted_values_dropped(value) -> None:
    entry = _format("backend.test", "evt", model=value)
    assert "model" not in entry


def test_short_string_lists_kept() -> None:
    entry = _format("backend.test", "evt", security_flag_types=["heuristic", "classifier"])
    assert entry["security_flag_types"] == ["heuristic", "classifier"]


def test_exception_reduced_to_type_and_frames() -> None:
    try:
        raise ValueError(f"parse failed for {CANARY}")
    except ValueError:
        import sys

        entry = _format("backend.test", "failed", exc_info=sys.exc_info())
    assert entry["exc_type"] == "ValueError"
    assert entry["exc_frames"] and all(":" in f for f in entry["exc_frames"])
    assert CANARY not in json.dumps(entry)


def test_configure_logging_is_idempotent_and_disables_uvicorn_access() -> None:
    stream = io.StringIO()
    configure_logging("INFO", stream=stream)
    configure_logging("INFO", stream=stream)
    marked = [
        h for h in logging.getLogger().handlers if getattr(h, "_backend_privacy_handler", False)
    ]
    assert len(marked) == 1
    access = logging.getLogger("uvicorn.access")
    assert access.disabled is True and access.propagate is False
    assert logging.getLogger("httpx").level == logging.WARNING

    logging.getLogger("uvicorn.access").info('203.0.113.9 - "GET /?q=%s"', CANARY)
    logging.getLogger("backend.test").info("hello")
    output = stream.getvalue()
    assert CANARY not in output and "203.0.113.9" not in output
    assert '"event":"hello"' in output


def test_tracing_forced_off(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    disable_third_party_tracing()
    for name, value in TRACING_ENV_OVERRIDES.items():
        assert os.environ[name] == value == "false"
