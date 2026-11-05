"""``POST /api/v1/analyze``: run the M7 analysis workflow on an uploaded resume.

Request (approved Rev. 3): ``multipart/form-data`` with ``file`` (PDF or
DOCX; the M2 upload rules and limits apply), optional ``job_description``
and optional ``include_preview`` (boolean, default false). Other request
formats are rejected with the M2 415 ``multipart_required`` error.

Response: the stored ``AnalysisReport`` (including ``report_id``) and, only
when requested, a ``preview`` of at most 1000 characters. The preview is in
this response only; it is never stored.

Validation runs before any LLM call: multipart limits and file checks (M2),
then the workflow guard (sanitisation, length limits, embedding availability).
A job-description field that is present is validated with the existing rules;
an absent field means no job description. Failures use the existing typed
errors: 4xx for input, 502/503 for upstream problems, 503 when the report
store is full, 504 when the workflow deadline passes. No report is stored for
a failed or timed-out run. Nothing here logs document or job text.

Concurrency (M10, audit finding H1): an analysis slot is taken only after the
whole upload has been received, and held through parsing and the workflow; it is
released however they end. When no slot is free the request gets 429
``analysis_capacity_exceeded``. The rate limit runs earlier, before the body is read.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.api.deps import AnalysisWorkflowDep, SettingsDep
from backend.api.rate_limits import AnalysisSlotDep
from backend.api.uploads import parse_bool_field, read_multipart_upload
from backend.parsing.service import ParseLimits, parse_document
from backend.schemas.common import ErrorEnvelope
from backend.schemas.parse import PREVIEW_MAX_CHARS
from backend.schemas.report import AnalysisResponse

logger = logging.getLogger("backend.api.analyze")

router = APIRouter(prefix="/api/v1", tags=["analysis"])

JOB_DESCRIPTION_FIELD = "job_description"
INCLUDE_PREVIEW_FIELD = "include_preview"
UTF8_MAX_BYTES_PER_CHAR = 4

_ERRORS = {
    status: {"model": ErrorEnvelope, "description": description}
    for status, description in {
        400: "Malformed multipart body",
        413: "File or request too large",
        415: "Not multipart, or an unsupported, mismatched or protected file type",
        422: "Invalid upload or job description, unreadable document or limits exceeded",
        502: "Upstream LLM error or invalid model output",
        503: "LLM not configured, model unavailable, upstream rate limit or report store full",
        504: "Upstream timeout or analysis deadline exceeded",
    }.items()
}

_REQUEST_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": ["file"],
                "properties": {
                    "file": {"type": "string", "format": "binary"},
                    JOB_DESCRIPTION_FIELD: {"type": "string"},
                    INCLUDE_PREVIEW_FIELD: {"type": "boolean", "default": False},
                },
            }
        }
    },
}


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses=_ERRORS,
    openapi_extra={"requestBody": _REQUEST_BODY},
)
async def analyze(
    request: Request, settings: SettingsDep, workflow: AnalysisWorkflowDep, slot: AnalysisSlotDep
) -> JSONResponse:
    form = await read_multipart_upload(
        request,
        max_file_bytes=settings.max_upload_bytes,
        allowed_fields=(JOB_DESCRIPTION_FIELD, INCLUDE_PREVIEW_FIELD),
        max_field_bytes=settings.max_job_description_chars * UTF8_MAX_BYTES_PER_CHAR,
    )
    include_preview = parse_bool_field(form.fields.get(INCLUDE_PREVIEW_FIELD))
    limits = ParseLimits(
        max_bytes=settings.max_upload_bytes,
        max_pages=settings.max_pdf_pages,
        max_chars=settings.max_extracted_chars,
    )
    # The upload is complete: only now take an analysis slot (429 when none is free).
    with slot.hold():
        parsed = await run_in_threadpool(parse_document, form.filename, form.file_bytes, limits)
        report = await workflow.run(
            resume_text=parsed.text, job_description=form.fields.get(JOB_DESCRIPTION_FIELD)
        )
    response = AnalysisResponse.model_validate(
        {
            **report.model_dump(),
            "preview": parsed.text[:PREVIEW_MAX_CHARS] if include_preview else None,
        }
    )
    logger.info(
        "analyze_completed",
        extra={
            "file_type": parsed.file_type,
            "page_count": parsed.page_count,
            "char_count": len(parsed.text),
            "critic_verdict": report.critic.verdict,
        },
    )
    body = response.model_dump(mode="json", exclude=None if include_preview else {"preview"})
    return JSONResponse(body)
