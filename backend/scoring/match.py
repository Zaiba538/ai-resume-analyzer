"""Match metrics for a resume and job description (M5), framework-independent.

Extracted unchanged from ``backend.api.routes.match`` (M7, decision D8) so that
``POST /api/v1/match`` and the M7 analysis workflow share one implementation.
Nothing here depends on FastAPI request or response objects.

* ``clean_job_description`` / ``clean_resume_text``: M3 sanitisation, M2 text
  normalisation, the configured length caps and the English-only check, raising
  typed 422 errors.
* ``compute_match``: semantic similarity, skill coverage, section completeness,
  the TF-IDF baseline and the ATS gate (always ``not_calibrated``).
"""

from __future__ import annotations

from backend.core.config import Settings
from backend.core.errors import AppError
from backend.parsing.language import require_english
from backend.parsing.service import normalise_text
from backend.schemas.match import (
    Baseline,
    MatchMetrics,
    MatchResult,
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
    require_english(cleaned, "job_description")
    return cleaned


def clean_resume_text(text: str, settings: Settings) -> str:
    cleaned = normalise_text(sanitize_text(text).text)
    if not cleaned:
        raise EmptyResumeTextError()
    if len(cleaned) > settings.max_extracted_chars:
        raise ResumeTextTooLongError()
    require_english(cleaned, "resume")
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
