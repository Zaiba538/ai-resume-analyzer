"""``POST /api/v1/match``: separate match metrics, TF-IDF baseline, ATS status.

Request forms (approved Rev. 3):

* ``multipart/form-data`` with ``file`` (PDF or DOCX, the M2 upload rules and
  limits apply) and ``job_description``;
* ``application/json`` with ``{"resume_text": ..., "job_description": ...}``.

Both texts are sanitised (M3), capped (``MAX_EXTRACTED_CHARS`` for the resume,
``MAX_JOB_DESCRIPTION_CHARS`` for the job description) and held in memory only
for the duration of the request. Nothing is stored, no text is logged, and no
LLM is called. The ATS composite is reported as ``not_calibrated``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from backend.api.deps import DocumentGateDep, EmbeddingServiceDep, SettingsDep
from backend.api.document_gate import DOCUMENT_BUSY_DESCRIPTION, DOCUMENT_TIMEOUT_DESCRIPTION
from backend.api.uploads import read_multipart_upload
from backend.core.errors import AppError
from backend.parsing.service import ParseLimits, parse_document
from backend.schemas.common import ErrorEnvelope
from backend.schemas.match import MatchResult, MatchTextRequest

# Scoring and input cleaning live in backend.scoring.match (M7, decision D8).
# They are re-exported here so existing imports from this module keep working.
from backend.scoring.match import (  # noqa: F401
    EmptyJobDescriptionError,
    EmptyResumeTextError,
    JobDescriptionTooLongError,
    MissingJobDescriptionError,
    ResumeTextTooLongError,
    clean_job_description,
    clean_resume_text,
    compute_match,
)

logger = logging.getLogger("backend.api.match")
router = APIRouter(prefix="/api/v1", tags=["match"])

JOB_DESCRIPTION_FIELD = "job_description"
UTF8_MAX_BYTES_PER_CHAR = 4


class UnsupportedMatchMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"
    message = "Send multipart/form-data (file + job_description) or application/json."


_ERRORS = {
    status: {"model": ErrorEnvelope, "description": description}
    for status, description in {
        400: "Malformed multipart body",
        413: "File or request too large",
        415: "Unsupported media type or file type",
        422: "Invalid input, unreadable document, limits exceeded or unsupported language",
        503: (
            "Semantic similarity model not available (`embedding_model_unavailable`), or "
            f"{DOCUMENT_BUSY_DESCRIPTION}"
        ),
        504: f"The document took too long: {DOCUMENT_TIMEOUT_DESCRIPTION}",
    }.items()
}
_REQUEST_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": ["file", JOB_DESCRIPTION_FIELD],
                "properties": {
                    "file": {"type": "string", "format": "binary"},
                    JOB_DESCRIPTION_FIELD: {"type": "string"},
                },
            }
        },
        "application/json": {"schema": MatchTextRequest.model_json_schema()},
    },
}


@router.post(
    "/match",
    response_model=MatchResult,
    responses=_ERRORS,
    openapi_extra={"requestBody": _REQUEST_BODY},
)
async def match(
    request: Request,
    settings: SettingsDep,
    embeddings: EmbeddingServiceDep,
    gate: DocumentGateDep,
) -> MatchResult:
    embedder = embeddings.get()  # 503 before reading the body if the model is unavailable
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()

    if media_type == "multipart/form-data":
        form = await read_multipart_upload(
            request,
            max_file_bytes=settings.max_upload_bytes,
            allowed_fields=(JOB_DESCRIPTION_FIELD,),
            max_field_bytes=settings.max_job_description_chars * UTF8_MAX_BYTES_PER_CHAR,
        )
        job = clean_job_description(form.fields.get(JOB_DESCRIPTION_FIELD), settings)
        limits = ParseLimits(
            max_bytes=settings.max_upload_bytes,
            max_pages=settings.max_pdf_pages,
            max_chars=settings.max_extracted_chars,
        )
        parsed = await gate.run(parse_document, form.filename, form.file_bytes, limits)
        resume, source = parsed.text, parsed.file_type
    elif media_type == "application/json":
        try:
            body = MatchTextRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            raise RequestValidationError(exc.errors(include_input=False)) from None
        job = clean_job_description(body.job_description, settings)
        resume, source = clean_resume_text(body.resume_text, settings), "text"
    else:
        raise UnsupportedMatchMediaTypeError()

    result = await run_in_threadpool(compute_match, embedder, resume, job)
    logger.info(
        "match_completed",
        extra={"file_type": source, "char_count": len(resume), "model": embedder.name},
    )
    return result
