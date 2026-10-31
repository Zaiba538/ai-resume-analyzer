"""Application factory.

Run with uvicorn's factory mode so that importing this module has no side
effects (no settings loading, no logging configuration):

    uvicorn --factory backend.main:create_app --no-access-log --port 7860

Middleware order, outermost first:
RequestId -> SecurityHeaders -> AccessLog -> UnhandledException -> routes.
Rate limiting and body-size limits join this chain in later milestones.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend import __version__
from backend.api.routes import health
from backend.core.config import Settings, get_settings
from backend.core.errors import UnhandledExceptionMiddleware, register_exception_handlers
from backend.core.logging import (
    AccessLogMiddleware,
    configure_logging,
    disable_third_party_tracing,
)
from backend.core.request_id import RequestIdMiddleware
from backend.core.security_headers import SecurityHeadersMiddleware
from backend.health.readiness import CONFIG_VALID, GROQ_API_KEY_CONFIGURED, ReadinessRegistry
from backend.health.upstream import GroqUpstreamMonitor

logger = logging.getLogger("backend.main")

DOCS_URL = "/docs"
REDOC_URL = "/redoc"
OPENAPI_URL = "/openapi.json"


def create_app(
    settings: Settings | None = None,
    *,
    groq_http_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    """Build the application.

    ``groq_http_client`` lets tests inject a fake transport for the upstream
    probe. In production the monitor creates and owns its own client.
    """
    settings = settings or get_settings()
    disable_third_party_tracing()
    configure_logging(settings.log_level)

    readiness = ReadinessRegistry()
    readiness.register(CONFIG_VALID, ready=True)  # Settings validated above.
    readiness.register(GROQ_API_KEY_CONFIGURED, ready=settings.groq_api_key_configured)

    upstream_monitor = GroqUpstreamMonitor(
        api_key=settings.groq_api_key,
        base_url=settings.groq_api_base_url,
        required_models=settings.groq_models,
        interval_seconds=settings.groq_probe_interval_seconds,
        timeout_seconds=settings.groq_probe_timeout_seconds,
        client=groq_http_client,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await upstream_monitor.start()
        logger.info("application_started", extra={"component": "app"})
        try:
            yield
        finally:
            await upstream_monitor.stop()
            logger.info("application_stopped", extra={"component": "app"})

    docs = settings.api_docs_enabled
    app = FastAPI(
        title="Resume Intelligence Platform",
        version=__version__,
        lifespan=lifespan,
        docs_url=DOCS_URL if docs else None,
        redoc_url=REDOC_URL if docs else None,
        openapi_url=OPENAPI_URL if docs else None,
    )
    app.state.settings = settings
    app.state.readiness = readiness
    app.state.upstream_monitor = upstream_monitor

    register_exception_handlers(app)
    app.include_router(health.router)

    # The frontend is mounted last so that API routes always take precedence.
    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    else:
        logger.warning("frontend_dir_missing", extra={"component": "frontend"})

    # add_middleware prepends, so the last one added is the outermost.
    app.add_middleware(UnhandledExceptionMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(
        SecurityHeadersMiddleware,
        frame_ancestors=settings.csp_frame_ancestors,
        hsts_enabled=settings.hsts_enabled,
        # Swagger UI / ReDoc load assets from a CDN; they are disabled by default.
        csp_exempt_paths=(DOCS_URL, REDOC_URL) if docs else (),
    )
    app.add_middleware(RequestIdMiddleware)
    return app
