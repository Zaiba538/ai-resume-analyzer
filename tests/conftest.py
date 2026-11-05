"""Shared test fixtures.

Isolation guarantees for every test not marked ``live``:

* Environment variables for all Settings fields are removed, so a developer's
  real ``GROQ_API_KEY`` can never leak into a test.
* Real outbound HTTP through httpx is blocked: any unmocked call fails the test.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.core.config import Settings
from backend.core.logging import PrivacyJsonFormatter
from backend.main import create_app
from tests.fixtures.embeddings import CountingFactory

# Fake, syntactically plausible key. Never a real credential.
FAKE_API_KEY = "gsk_test_" + "0" * 40

DEFAULT_MODELS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "meta-llama/llama-prompt-guard-2-86m",
)


@pytest.fixture(autouse=True)
def _isolate_environment(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.get_closest_marker("live") is not None:
        return

    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)

    def _blocked_sync(self: Any, req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected real network call to {req.url.host}")

    async def _blocked_async(self: Any, req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected real network call to {req.url.host}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked_sync)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _blocked_async)


class FakeGroqAPI:
    """In-process stand-in for the Groq models endpoint."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.status_code = 200
        self.models: list[str] = list(DEFAULT_MODELS)
        self.raw_body: bytes | None = None
        self.exception: Exception | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if self.exception is not None:
            raise self.exception
        if self.raw_body is not None:
            return httpx.Response(self.status_code, content=self.raw_body)
        payload = {"object": "list", "data": [{"id": m, "object": "model"} for m in self.models]}
        return httpx.Response(self.status_code, json=payload)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def fake_groq() -> FakeGroqAPI:
    return FakeGroqAPI()


@pytest.fixture
def make_settings(tmp_path: Any) -> Callable[..., Settings]:
    """Settings that ignore any .env file and the real environment.

    Rate limiting is off unless a test enables it: every TestClient request comes
    from the same host, so ordinary tests would otherwise share one bucket. The
    production default (on) is asserted in tests/unit/test_config.py.
    """

    def _make(**overrides: Any) -> Settings:
        values: dict[str, Any] = {
            "frontend_dir": tmp_path / "no-frontend",
            "rate_limit_enabled": False,
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    return _make


@pytest.fixture
def start_app(
    make_settings: Callable[..., Settings], fake_groq: FakeGroqAPI
) -> Iterator[Callable[..., TestClient]]:
    """Start an app (lifespan included) and return a TestClient for it."""
    clients: list[TestClient] = []

    def _start(
        *,
        extra_routes: Callable[[Any], None] | None = None,
        embedder_factory: Callable[[], Any] | None = None,
        **overrides: Any,
    ) -> TestClient:
        # Tests never download or load the real Sentence-Transformer model.
        app = create_app(
            make_settings(**overrides),
            groq_http_client=fake_groq.client(),
            embedder_factory=embedder_factory or CountingFactory(),
        )
        if extra_routes is not None:
            extra_routes(app)
        client = TestClient(app, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return client

    yield _start
    for client in clients:
        client.__exit__(None, None, None)


class CapturedLogs:
    def __init__(self, stream: io.StringIO) -> None:
        self._stream = stream

    @property
    def text(self) -> str:
        return self._stream.getvalue()

    def records(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.text.splitlines() if line.strip()]

    def events(self, event: str) -> list[dict[str, Any]]:
        return [r for r in self.records() if r.get("event") == event]


@pytest.fixture
def captured_logs() -> Iterator[CapturedLogs]:
    """Capture exactly what the privacy formatter would emit."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(PrivacyJsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield CapturedLogs(stream)
    finally:
        root.removeHandler(handler)
