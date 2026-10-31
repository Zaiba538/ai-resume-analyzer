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

from backend.api.deps import EmbeddingServiceDep, SettingsDep
from backend.api.uploads import read_multipart_upload
from backend.core.config import Settings
from backend.core.errors import AppError
from backend.parsing.service import ParseLimits, normalise_text, parse_document
from backend.schemas.common import ErrorEnvelope
from backend.schemas.match import (
    Baseline,
    MatchMetrics,
    MatchResult,
    MatchTextRequest,
    SectionCompletenessMetric,
    SemanticSimilarityMetric,
    SkillCoverageMetric,
)
from backend.scoring.ats_composite import evaluate_ats_gate
from backend.scoring.completeness import section_completeness
from backend.scoring.embeddings import Embedder
from backend.scoring.semantic import semantic_similarity
from backend.scoring.skill_coverage import skill_coverage
from backend.scoring.tfidf_baseline import tfidf_keyword_match
from backend.security.sanitize import TextTooLongError, sanitize_text

logger = logging.getLogger("backend.api.match")
router = APIRouter(prefix="/api/v1", tags=["match"])

JOB_DESCRIPTION_FIELD = "job_description"
UTF8_MAX_BYTES_PER_CHAR = 4


class UnsupportedMatchMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"
    message = "Send multipart/form-data (file + job_description) or application/json."


class MissingJobDescriptionError(AppError):
    status_code = 422
    code = "missing_job_description"
    message = "A job description is required."


class EmptyJobDescriptionError(AppError):
    status_code = 422
    code = "empty_job_description"
    message = "The job description is empty."


class JobDescriptionTooLongError(AppError):
    status_code = 422
    code = "job_description_too_long"
    message = "The job description exceeds the maximum allowed length."


class EmptyResumeTextError(AppError):
    status_code = 422
    code = "empty_resume_text"
    message = "The resume text is empty."


class ResumeTextTooLongError(AppError):
    status_code = 422
    code = "document_too_long"
    message = "The resume text exceeds the maximum allowed length."


def clean_job_description(text: str | None, settings: Settings) -> str:
    if text is None:
        raise MissingJobDescriptionError()
    try:
        cleaned = sanitize_text(text, max_chars=settings.max_job_description_chars).text
    except TextTooLongError as exc:
        raise JobDescriptionTooLongError() from exc
    cleaned = normalise_text(cleaned)
    if not cleaned:
        raise EmptyJobDescriptionError()
    return cleaned


def clean_resume_text(text: str, settings: Settings) -> str:
    cleaned = normalise_text(sanitize_text(text).text)
    if not cleaned:
        raise EmptyResumeTextError()
    if len(cleaned) > settings.max_extracted_chars:
        raise ResumeTextTooLongError()
    return cleaned


def compute_match(embedder: Embedder, resume: str, job: str) -> MatchResult:
    """All metrics for one pair of already-cleaned texts. CPU-bound; run in a thread."""
    coverage = skill_coverage(resume, job)
    completeness = section_completeness(resume)
    return MatchResult(
        metrics=MatchMetrics(
            semantic_similarity=SemanticSimilarityMetric(
                value=semantic_similarity(embedder, resume, job)
            ),
            skill_coverage=SkillCoverageMetric(
                value=coverage.value,
                matched=list(coverage.matched),
                missing=list(coverage.missing),
                jd_skills_total=coverage.jd_skills_total,
            ),
            section_completeness=SectionCompletenessMetric(
                value=completeness.value,
                present=list(completeness.present),
                absent=list(completeness.absent),
            ),
        ),
        baseline=Baseline(tfidf_keyword_match=tfidf_keyword_match(resume, job)),
        ats_score=evaluate_ats_gate(),
    )


_ERRORS = {
    status: {"model": ErrorEnvelope, "description": description}
    for status, description in {
        400: "Malformed multipart body",
        413: "File or request too large",
        415: "Unsupported media type or file type",
        422: "Invalid input, unreadable document or limits exceeded",
        503: "Semantic similarity model not available",
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
    request: Request, settings: SettingsDep, embeddings: EmbeddingServiceDep
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
        parsed = await run_in_threadpool(parse_document, form.filename, form.file_bytes, limits)
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
