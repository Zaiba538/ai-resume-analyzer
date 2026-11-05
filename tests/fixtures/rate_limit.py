"""Test harness for M10 rate limiting through the real application.

``ClientAddress`` wraps the app and sets the ASGI ``client`` address of each
request from a test-only header, which it removes before the app sees it, so one
TestClient (one lifespan, one event loop) can act as many clients, also from
several threads at once. It also records how many body messages each request read.
The application itself only ever sees ``request.client.host``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.core import rate_limit
from backend.core.rate_limit import RateLimiter
from backend.main import create_app
from tests.conftest import FakeGroqAPI
from tests.fixtures.embeddings import CountingFactory

CLIENT_HEADER = "x-test-client-ip"
_CLIENT_HEADER_RAW = CLIENT_HEADER.encode("latin-1")
DEFAULT_IP = "192.0.2.1"
RATE_LIMIT_HEADER_PREFIXES = ("x-ratelimit", "ratelimit")

REPORT_ID = str(uuid.UUID(int=0x1234, version=4))
# One cheap request per rate-limited endpoint. None of them needs an LLM; most are
# rejected by the endpoint itself (415, 404, 503), which is fine: the rate limit
# counts every request that reaches the route, before the body is read.
ENDPOINTS: dict[str, tuple[str, str, dict[str, Any]]] = {
    rate_limit.PARSE: ("POST", "/api/v1/parse", {}),
    rate_limit.MATCH: ("POST", "/api/v1/match", {}),
    rate_limit.ANALYZE: ("POST", "/api/v1/analyze", {}),
    rate_limit.REPORT: ("GET", f"/api/v1/report/{REPORT_ID}", {}),
    rate_limit.REPORT_PDF: ("GET", f"/api/v1/report/{REPORT_ID}/pdf", {}),
    rate_limit.INTERVIEW_QUESTIONS: (
        "POST",
        "/api/v1/interview/questions",
        {"json": {"report_id": REPORT_ID}},
    ),
    rate_limit.INTERVIEW_EVALUATE: (
        "POST",
        "/api/v1/interview/evaluate",
        {"json": {"report_id": REPORT_ID, "question_id": "beginner-1", "answer": "An answer."}},
    ),
}
# Approved per-client minute limits (defaults).
DEFAULT_MINUTE_LIMITS = {
    rate_limit.PARSE: 30,
    rate_limit.MATCH: 30,
    rate_limit.ANALYZE: 5,
    rate_limit.REPORT: 60,
    rate_limit.REPORT_PDF: 20,
    rate_limit.INTERVIEW_QUESTIONS: 10,
    rate_limit.INTERVIEW_EVALUATE: 20,
}


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class ClientAddress:
    """ASGI wrapper: the client address comes from ``CLIENT_HEADER`` (then removed)."""

    def __init__(self, app: ASGIApp, default_ip: str = DEFAULT_IP) -> None:
        self.app = app
        self.default_ip = default_ip
        self.body_reads: list[tuple[str, int]] = []
        # Called when a response starts, before its first byte reaches the client.
        self.on_response_start: Callable[[], None] | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = [(k, v) for k, v in scope["headers"] if k != _CLIENT_HEADER_RAW]
        host = next(
            (v.decode("latin-1") for k, v in scope["headers"] if k == _CLIENT_HEADER_RAW),
            self.default_ip,
        )
        scope = {**scope, "headers": headers, "client": (host, 50000)}
        index = len(self.body_reads)
        self.body_reads.append((scope["path"], 0))

        async def counting_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                path, count = self.body_reads[index]
                self.body_reads[index] = (path, count + 1)
            return message

        async def observing_send(message: Message) -> None:
            if message["type"] == "http.response.start" and self.on_response_start:
                self.on_response_start()
            await send(message)

        await self.app(scope, counting_receive, observing_send)


@dataclass
class Harness:
    app: FastAPI
    client: TestClient
    wrapper: ClientAddress
    clock: FakeClock = field(default_factory=FakeClock)

    @property
    def limiter(self) -> RateLimiter:
        return self.app.state.rate_limiter

    def call(self, policy: str, ip: str = DEFAULT_IP, **kwargs: Any):
        method, url, defaults = ENDPOINTS[policy]
        return self.request(method, url, ip=ip, **{**defaults, **kwargs})

    def request(self, method: str, url: str, *, ip: str = DEFAULT_IP, **kwargs: Any):
        headers = {**kwargs.pop("headers", {}), CLIENT_HEADER: ip}
        return self.client.request(method, url, headers=headers, **kwargs)

    def use_fake_clock(self) -> FakeClock:
        """Replace the app's limiter with one on a fake clock (same settings)."""
        self.app.state.rate_limiter = RateLimiter.from_settings(
            self.app.state.settings, clock=self.clock
        )
        return self.clock


def rate_limit_headers(response) -> list[str]:
    names = [name.lower() for name in response.headers]
    return [n for n in names if n == "retry-after" or n.startswith(RATE_LIMIT_HEADER_PREFIXES)]


def harness_factory(
    make_settings: Callable[..., Any], fake_groq: FakeGroqAPI
) -> Iterator[Callable[..., Harness]]:
    """Body of a ``harness`` fixture: ``yield from harness_factory(...)``.

    It builds apps with rate limiting ON (unless overridden), behind ``ClientAddress``.
    """
    clients: list[TestClient] = []

    def _make(*, llm_http_client: Any = None, **overrides: Any) -> Harness:
        settings = make_settings(**{"rate_limit_enabled": True, **overrides})
        app = create_app(
            settings,
            groq_http_client=fake_groq.client(),
            llm_http_client=llm_http_client,
            embedder_factory=CountingFactory(),
        )
        wrapper = ClientAddress(app)
        client = TestClient(wrapper, raise_server_exceptions=False)
        client.__enter__()
        clients.append(client)
        return Harness(app=app, client=client, wrapper=wrapper)

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
