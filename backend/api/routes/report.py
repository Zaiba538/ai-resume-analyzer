"""``GET /api/v1/report/{report_id}``: a stored analysis report (approved Rev. 3).

Only canonical UUID4 IDs are looked up. Unknown, malformed and expired IDs
all return the same 404 ``report_not_found``, so they cannot be told apart.
Responses carry ``Cache-Control: no-store`` and ``X-Robots-Tag: noindex``
(security-headers middleware, all ``/api/`` paths). There is deliberately no
listing endpoint, and the access log records only the route template, never
the ID.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend.api.deps import ReportStoreDep
from backend.core.errors import AppError
from backend.schemas.common import ErrorEnvelope
from backend.schemas.report import AnalysisReport

router = APIRouter(prefix="/api/v1", tags=["report"])


class ReportNotFoundError(AppError):
    status_code = 404
    code = "report_not_found"
    message = "The report was not found or has expired."


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
