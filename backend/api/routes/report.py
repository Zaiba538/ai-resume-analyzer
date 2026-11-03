"""Stored analysis reports (approved Rev. 3; M8 decisions D5, D7, E1 and E2).

``GET /api/v1/report/{report_id}``: the report as JSON.
``GET /api/v1/report/{report_id}/pdf``: the same report rendered as a PDF on
demand. The PDF is built in memory for each request and never cached or stored.

Only canonical UUID4 IDs are looked up. Unknown, malformed and expired IDs
all return the same 404 ``report_not_found``, so they cannot be told apart.
Responses carry ``Cache-Control: no-store`` and ``X-Robots-Tag: noindex``
(security-headers middleware, all ``/api/`` paths). There is deliberately no
listing endpoint, and the access log records only the route template, never
the ID.

PDF errors use the error envelope (E1, E2): 422 ``pdf_page_limit_exceeded``,
504 ``pdf_render_timeout`` and 503 ``pdf_renderer_unavailable``. Renderer
availability is checked before the ID is looked up, so when the fonts are
unavailable every ID gets the same 503.

Rendering runs in a worker thread (``asyncio.to_thread``) under a deadline of
``PDF_DEADLINE_SECONDS``. Python cannot stop a running thread: when the request
times out, the worker is abandoned, not cancelled. It stops itself at the
renderer's next deadline check (the renderer receives the same deadline), its
result is discarded, and the renderer's lock serialises it with later renders.
"""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter
from fastapi.responses import Response

from backend.api.deps import ReportStoreDep
from backend.core.errors import AppError
from backend.reports.errors import (
    PdfPageLimitError,
    PdfRenderError,
    PdfRendererUnavailableError,
    PdfRenderTimeoutError,
)
from backend.reports.fonts import default_registry
from backend.reports.pdf import render_report_pdf
from backend.schemas.common import ErrorEnvelope
from backend.schemas.report import AnalysisReport

logger = logging.getLogger("backend.api.report")

router = APIRouter(prefix="/api/v1", tags=["report"])

# M8 decision D5.
PDF_MAX_PAGES = 100
PDF_DEADLINE_SECONDS = 15.0
PDF_FILENAME = "resume-analysis-report.pdf"
PDF_MEDIA_TYPE = "application/pdf"

# E1 and E2: typed renderer error -> HTTP status.
PDF_ERROR_STATUS: dict[type[PdfRenderError], int] = {
    PdfPageLimitError: 422,
    PdfRenderTimeoutError: 504,
    PdfRendererUnavailableError: 503,
}


class ReportNotFoundError(AppError):
    status_code = 404
    code = "report_not_found"
    message = "The report was not found or has expired."


def _pdf_error(error: PdfRenderError) -> AppError:
    return AppError(error.message, code=error.code, status_code=PDF_ERROR_STATUS[type(error)])


@router.get(
    "/report/{report_id}",
    response_model=AnalysisReport,
    responses={404: {"model": ErrorEnvelope, "description": "Unknown, malformed or expired ID"}},
)
async def get_report(report_id: str, store: ReportStoreDep) -> AnalysisReport:
    report = await store.get(report_id)  # malformed IDs are never looked up
    if report is None:
        raise ReportNotFoundError()
    return report


@router.get(
    "/report/{report_id}/pdf",
    response_class=Response,
    responses={
        200: {"content": {PDF_MEDIA_TYPE: {}}, "description": "The report as a PDF"},
        404: {"model": ErrorEnvelope, "description": "Unknown, malformed or expired ID"},
        422: {"model": ErrorEnvelope, "description": "The PDF would exceed the page limit"},
        503: {"model": ErrorEnvelope, "description": "The PDF renderer is unavailable"},
        504: {"model": ErrorEnvelope, "description": "Rendering exceeded the deadline"},
    },
)
async def get_report_pdf(report_id: str, store: ReportStoreDep) -> Response:
    if not default_registry().ready:
        raise _pdf_error(PdfRendererUnavailableError())
    report = await store.get(report_id)  # malformed IDs are never looked up
    if report is None:
        raise ReportNotFoundError()
    started = time.monotonic()
    try:
        data = await asyncio.wait_for(
            asyncio.to_thread(
                render_report_pdf,
                report,
                max_pages=PDF_MAX_PAGES,
                deadline_seconds=PDF_DEADLINE_SECONDS,
            ),
            timeout=PDF_DEADLINE_SECONDS,
        )
    except TimeoutError:
        # The worker thread is still running; it is abandoned, not cancelled.
        logger.warning(
            "report_pdf_render_abandoned",
            extra={"component": "report_pdf", "error_code": PdfRenderTimeoutError.code},
        )
        raise _pdf_error(PdfRenderTimeoutError(PDF_DEADLINE_SECONDS)) from None
    except (PdfPageLimitError, PdfRenderTimeoutError, PdfRendererUnavailableError) as exc:
        raise _pdf_error(exc) from None
    logger.info(
        "report_pdf_rendered",
        extra={
            "component": "report_pdf",
            "duration_ms": round((time.monotonic() - started) * 1000),
        },
    )
    return Response(
        content=data,
        media_type=PDF_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{PDF_FILENAME}"'},
    )
