"""Application factory.

Run with uvicorn's factory mode so that importing this module has no side
effects (no settings loading, no logging configuration):

    uvicorn --factory backend.main:create_app --no-access-log --port 7860

Middleware order, outermost first:
RequestId -> SecurityHeaders -> AccessLog -> BodySizeLimit -> UnhandledException
-> routes. Rate limiting (M10) joins between AccessLog and BodySizeLimit.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend import __version__
from backend.api.routes import analyze, health, interview, match, parse, report
from backend.core.body_limit import BodySizeLimitMiddleware
from backend.core.config import Settings, get_settings
from backend.core.errors import UnhandledExceptionMiddleware, register_exception_handlers
from backend.core.logging import (
    AccessLogMiddleware,
    configure_logging,
    disable_third_party_tracing,
)
from backend.core.request_id import RequestIdMiddleware
from backend.core.security_headers import SecurityHeadersMiddleware
from backend.graph.analysis import AnalysisDeps, AnalysisWorkflow, build_prompt_guard
from backend.graph.evaluation import EvaluationDeps, EvaluationWorkflow
from backend.health.readiness import (
    ANALYSIS_GRAPH_COMPILED,
    CONFIG_VALID,
    EMBEDDING_MODEL_LOADED,
    GROQ_API_KEY_CONFIGURED,
    PDF_RENDERER_READY,
    REPORT_STORE_READY,
    ReadinessRegistry,
)
from backend.health.upstream import GroqUpstreamMonitor
from backend.llm.client import LLMClients
from backend.reports.fonts import default_registry
from backend.scoring.embeddings import Embedder, EmbeddingService, load_sentence_transformer
from backend.services.interview import InterviewQuestionService
from backend.storage.report_store import ReportStore

logger = logging.getLogger("backend.main")

DOCS_URL = "/docs"
REDOC_URL = "/redoc"
OPENAPI_URL = "/openapi.json"

# Headroom above the file limit for multipart boundaries, part headers and fields.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


def create_app(
    settings: Settings | None = None,
    *,
    groq_http_client: httpx.AsyncClient | None = None,
    llm_http_client: httpx.AsyncClient | None = None,
    embedder_factory: Callable[[], Embedder] | None = None,
) -> FastAPI:
    """Build the application.

    ``groq_http_client`` and ``llm_http_client`` let tests inject fake
    transports for the upstream probe and for LLM calls. In production each
    component creates and owns its own client. ``embedder_factory`` replaces the
    Sentence-Transformer loader (tests use a deterministic fake); it is called
    once, at start-up.
    """
    settings = settings or get_settings()
    disable_third_party_tracing()
    configure_logging(settings.log_level)

    readiness = ReadinessRegistry()
    readiness.register(CONFIG_VALID, ready=True)  # Settings validated above.
    readiness.register(GROQ_API_KEY_CONFIGURED, ready=settings.groq_api_key_configured)
    readiness.register(EMBEDDING_MODEL_LOADED)
    readiness.register(REPORT_STORE_READY)
    readiness.register(PDF_RENDERER_READY)

    report_store = ReportStore(
        ttl_seconds=settings.report_ttl_seconds,
        max_reports=settings.max_stored_reports,
        sweep_interval_seconds=settings.report_sweep_interval_seconds,
    )

    embeddings = EmbeddingService(embedder_factory or (lambda: load_sentence_transformer(settings)))

    upstream_monitor = GroqUpstreamMonitor(
        api_key=settings.groq_api_key,
        base_url=settings.groq_api_base_url,
        required_models=settings.groq_models,
        interval_seconds=settings.groq_probe_interval_seconds,
        timeout_seconds=settings.groq_probe_timeout_seconds,
        client=groq_http_client,
    )
    llm_clients = LLMClients(settings, http_client=llm_http_client, monitor=upstream_monitor)

    # M7: the analysis workflow, compiled once (no checkpointer) and shared by requests.
    analysis_workflow = AnalysisWorkflow(
        AnalysisDeps(
            settings=settings,
            llm_clients=llm_clients,
            embeddings=embeddings,
            report_store=report_store,
            prompt_guard=build_prompt_guard(settings, llm_clients, upstream_monitor),
        )
    )
    readiness.register(ANALYSIS_GRAPH_COMPILED, ready=True)

    # M9: interview questions from a stored report, and the answer-evaluation graph
    # (compiled once, no checkpointer). Both only read the report store. The
    # evaluation graph has its own flag-only Prompt Guard classifier (D5, N1).
    interview_service = InterviewQuestionService(
        settings=settings, llm_clients=llm_clients, report_store=report_store
    )
    evaluation_workflow = EvaluationWorkflow(
        EvaluationDeps(
            settings=settings,
            llm_clients=llm_clients,
            report_store=report_store,
            prompt_guard=build_prompt_guard(settings, llm_clients, upstream_monitor),
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        readiness.set_ready(EMBEDDING_MODEL_LOADED, await embeddings.load())
        # M8 D7: register the PDF fonts once per process. Registration reads and
        # verifies the font files, so it runs off the event loop; it never raises.
        fonts = default_registry()
        readiness.set_ready(PDF_RENDERER_READY, await asyncio.to_thread(fonts.register))
        await report_store.start()
        readiness.set_ready(REPORT_STORE_READY, report_store.running)
        await upstream_monitor.start()
        logger.info("application_started", extra={"component": "app"})
        try:
            yield
        finally:
            readiness.set_ready(REPORT_STORE_READY, False)
            await report_store.stop()
            await upstream_monitor.stop()
            await llm_clients.aclose()
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
    app.state.llm_clients = llm_clients
    app.state.embeddings = embeddings
    app.state.report_store = report_store
    app.state.analysis_workflow = analysis_workflow
    app.state.interview_service = interview_service
    app.state.evaluation_workflow = evaluation_workflow

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(parse.router)
    app.include_router(match.router)
    app.include_router(analyze.router)
    app.include_router(report.router)
    app.include_router(interview.router)

    # The frontend is mounted last so that API routes always take precedence.
    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    else:
        logger.warning("frontend_dir_missing", extra={"component": "frontend"})

    # add_middleware prepends, so the last one added is the outermost.
    app.add_middleware(UnhandledExceptionMiddleware)
    app.add_middleware(
        BodySizeLimitMiddleware,
        # File + job description (UTF-8, at most 4 bytes per character) + overhead.
        max_body_bytes=settings.max_upload_bytes
        + settings.max_job_description_chars * 4
        + MULTIPART_OVERHEAD_BYTES,
    )
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
