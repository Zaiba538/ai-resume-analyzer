"""M10: rate limiting through a real uvicorn server, with uvicorn's default proxy trust.

Each test starts its own ``uvicorn.Server`` in a thread, on a socket bound to an
ephemeral loopback port, so it never touches another server or a fixed port. The
server runs the real application from ``create_app`` with uvicorn's defaults:
``proxy_headers=True`` and ``forwarded_allow_ips`` of ``127.0.0.1,::1``. The test
client connects from 127.0.0.1, which is therefore a trusted proxy here. The
limiter's clock is replaced by a fake one so that recovery needs no real waiting.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import httpx
import pytest
import uvicorn
from fastapi import FastAPI

from backend.core.rate_limit import RateLimiter
from backend.main import create_app
from tests.fixtures.embeddings import CountingFactory
from tests.fixtures.rate_limit import REPORT_ID, FakeClock
from tests.integration.test_api_rate_limit import assert_rate_limited

REPORT_URL = f"/api/v1/report/{REPORT_ID}"
STARTUP_SECONDS = 60.0
SHUTDOWN_SECONDS = 30.0
UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi")


def _uvicorn_logging_state() -> dict[str, tuple]:
    """Everything ``uvicorn.Config`` could change on uvicorn's loggers.

    pytest attaches its own capture handlers to every non-propagating logger for
    each test phase (setup, call, teardown) and removes them afterwards; those are
    left out, so that states taken in different phases can be compared.
    """
    state = {}
    for name in UVICORN_LOGGERS:
        logger = logging.getLogger(name)
        handlers = tuple(
            h for h in logger.handlers if not type(h).__module__.startswith("_pytest.")
        )
        state[name] = (
            logger.level,
            logger.disabled,
            logger.propagate,
            handlers,
            tuple(logger.filters),
        )
    return state


# The suite blocks real network calls by patching httpx's transports for every test
# (tests/conftest.py). The method is captured here, at import, before any test runs,
# and used only by ``_TestServerTransport``: requests to anything other than this
# test's own loopback server stay blocked, including any made by the application.
_UNGUARDED_HANDLE_REQUEST = httpx.HTTPTransport.handle_request


class _TestServerTransport(httpx.HTTPTransport):
    def __init__(self, port: int) -> None:
        super().__init__()
        self._port = port

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if (request.url.host, request.url.port) != ("127.0.0.1", self._port):
            raise AssertionError(f"unexpected real network call to {request.url.host}")
        return _UNGUARDED_HANDLE_REQUEST(self, request)


@dataclass
class LiveServer:
    app: FastAPI
    server: uvicorn.Server
    port: int
    clock: FakeClock

    def client(self) -> httpx.Client:
        return httpx.Client(
            base_url=f"http://127.0.0.1:{self.port}",
            transport=_TestServerTransport(self.port),
            timeout=30.0,
            trust_env=False,
        )

    def client_keys(self) -> set[str]:
        """The identities (HMACs) the limiter has per-client keys for."""
        keys = self.app.state.rate_limiter.tracked_keys()
        return {key.rsplit(":", 1)[1] for key in keys if key.startswith("client:")}

    def key_of(self, address: str) -> str:
        return self.app.state.rate_limiter.client_key(address)


@pytest.fixture
def live_server(make_settings, monkeypatch) -> Iterator[Callable[..., LiveServer]]:
    # uvicorn reads FORWARDED_ALLOW_IPS from the environment when it is not passed;
    # remove any inherited value so the default trust is what is tested.
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    started: list[tuple[uvicorn.Server, threading.Thread, socket.socket]] = []
    # Logging as it was just before the latest uvicorn.Config was created.
    baseline: list[dict[str, tuple]] = []

    def _start(*, prepare_logging: Callable[[], None] | None = None, **overrides) -> LiveServer:
        # create_app runs the application's own configure_logging (expected).
        app = create_app(
            make_settings(**{"rate_limit_enabled": True, **overrides}),
            embedder_factory=CountingFactory(),
        )
        if prepare_logging is not None:
            prepare_logging()
        baseline[:] = [_uvicorn_logging_state()]
        # Leave global logging alone: log_config=None skips uvicorn's dictConfig,
        # log_level=None skips its setLevel calls, and the default access_log=True
        # skips clearing uvicorn.access's handlers and propagation.
        config = uvicorn.Config(app, log_config=None, log_level=None, lifespan="on")
        assert _uvicorn_logging_state() == baseline[0], "uvicorn.Config changed logging"
        clock = FakeClock()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        started.append((server, thread, sock))
        thread.start()
        deadline = time.monotonic() + STARTUP_SECONDS
        while not server.started:
            assert thread.is_alive(), "the test server stopped during start-up"
            assert time.monotonic() < deadline, "the test server did not start"
            time.sleep(0.01)
        # Same settings, fake clock: replaced once the server is up and idle.
        app.state.rate_limiter = RateLimiter.from_settings(app.state.settings, clock=clock)
        return LiveServer(app, server, sock.getsockname()[1], clock)

    yield _start
    for server, thread, sock in started:
        server.should_exit = True
        thread.join(SHUTDOWN_SECONDS)
        sock.close()
        assert not thread.is_alive(), "the test server did not shut down"
    if baseline:
        assert _uvicorn_logging_state() == baseline[0], "the test server changed logging"


def test_the_server_runs_with_uvicorns_default_proxy_trust(live_server) -> None:
    live = live_server()
    assert live.server.config.proxy_headers is True
    assert live.server.config.forwarded_allow_ips == "127.0.0.1,::1"
    with live.client() as client:
        assert client.get("/health").status_code == 200
    # The transport reaches this test's server only; everything else stays blocked.
    transport = _TestServerTransport(live.port)
    for url in ("http://example.com/", f"http://localhost:{live.port}/", "http://127.0.0.1:1/"):
        with pytest.raises(AssertionError, match="unexpected real network call"):
            transport.handle_request(httpx.Request("GET", url))


def test_rate_limit_429_and_recovery_over_real_http(live_server) -> None:
    live = live_server(rate_limit_report="2/minute")
    with live.client() as client:
        first, second = client.get(REPORT_URL), client.get(REPORT_URL)
        assert (first.status_code, second.status_code) == (404, 404)
        limited = client.get(REPORT_URL)
        assert_rate_limited(limited, retry_after=60)
        request_ids = {r.headers["X-Request-ID"] for r in (first, second, limited)}
        assert len(request_ids) == 3
        live.clock.advance(59.5)
        assert_rate_limited(client.get(REPORT_URL), retry_after=1)
        live.clock.advance(0.5)  # both requests leave the window
        assert client.get(REPORT_URL).status_code == 404
    assert live.client_keys() == {live.key_of("127.0.0.1")}


def test_forwarded_for_follows_uvicorns_trust_rules(live_server) -> None:
    live = live_server(rate_limit_report="1/minute")
    with live.client() as client:
        # No header: the client is the loopback peer itself.
        assert client.get(REPORT_URL).status_code == 404
        assert_rate_limited(client.get(REPORT_URL))
        # The peer (127.0.0.1) is trusted, so uvicorn takes the client from
        # X-Forwarded-For: each forwarded client has its own bucket.
        forwarded = {"X-Forwarded-For": "198.51.100.1"}
        assert client.get(REPORT_URL, headers=forwarded).status_code == 404
        assert_rate_limited(client.get(REPORT_URL, headers=forwarded))
        other = {"X-Forwarded-For": "198.51.100.2"}
        assert client.get(REPORT_URL, headers=other).status_code == 404
        # Prepending a fake address does not give a fresh bucket: uvicorn uses the
        # right-most untrusted address, the one the trusted proxy appended.
        spoofed = {"X-Forwarded-For": "192.0.2.250, 198.51.100.1"}
        assert_rate_limited(client.get(REPORT_URL, headers=spoofed))
        # Other forwarding headers are not used by uvicorn or by the application.
        assert_rate_limited(client.get(REPORT_URL, headers={"X-Real-IP": "198.51.100.3"}))
    assert live.client_keys() == {
        live.key_of("127.0.0.1"),
        live.key_of("198.51.100.1"),
        live.key_of("198.51.100.2"),
    }


def test_health_stays_exempt_while_the_client_is_limited(live_server) -> None:
    live = live_server(rate_limit_report="1/minute")
    with live.client() as client:
        assert client.get(REPORT_URL).status_code == 404
        assert_rate_limited(client.get(REPORT_URL))
        for _ in range(100):  # more than any configured limit
            health = client.get("/health")
            assert health.status_code == 200
            assert "Retry-After" not in health.headers
        assert client.get("/ready").status_code != 429
        assert_rate_limited(client.get(REPORT_URL))  # still limited: /health used nothing
    assert live.client_keys() == {live.key_of("127.0.0.1")}


def test_the_test_server_leaves_uvicorn_access_logging_unchanged(live_server, monkeypatch) -> None:
    """Audit finding L-1: starting and using the server must not reconfigure logging.

    Once the application has configured logging (``uvicorn.access`` disabled and not
    propagating), a marker handler and level are added before the server is created.
    uvicorn's own logging setup would re-enable the logger, make it propagate, drop
    the handler or change the level.
    """
    access = logging.getLogger("uvicorn.access")
    marker = logging.NullHandler()
    before: list[dict[str, tuple]] = []

    def prepare() -> None:  # monkeypatch restores every attribute after the test
        monkeypatch.setattr(access, "handlers", [marker])
        monkeypatch.setattr(access, "level", logging.ERROR)
        # The state just before this server's uvicorn.Config is created.
        before.append(_uvicorn_logging_state())

    live = live_server(rate_limit_report="1/minute", prepare_logging=prepare)
    assert len(before) == 1
    assert _uvicorn_logging_state() == before[0]  # after start-up
    assert access.disabled is True and access.propagate is False  # set by create_app
    with live.client() as client:
        assert client.get(REPORT_URL).status_code == 404
        assert_rate_limited(client.get(REPORT_URL))
    assert _uvicorn_logging_state() == before[0]  # after requests
    assert access.disabled is True and access.propagate is False
    assert access.handlers == [marker] and access.level == logging.ERROR
