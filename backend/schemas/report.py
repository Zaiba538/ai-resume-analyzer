"""Persisted analysis report (approved Rev. 3, M7): derived data only.

``AnalysisReport`` holds structured agent output, match metrics, the Critic
summary, security flags and run metadata. It has no field for raw resume text,
the job description, the extracted-text preview or any prompt, and unknown
fields are rejected, so raw text cannot be stored by accident (decision D6).

``match`` and ``skill_gap`` are both present when a job description was
supplied and both null otherwise. A report expires at most
``MAX_REPORT_TTL_SECONDS`` after it was created.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import UUID4, AwareDatetime, ConfigDict, Field, StringConstraints, model_validator

from backend.core.config import MAX_REPORT_TTL_SECONDS
from backend.schemas.critic import MAX_CRITIC_ISSUES, CriticIssue
from backend.schemas.interview import InterviewQuestions
from backend.schemas.match import MatchResult
from backend.schemas.parse import PREVIEW_MAX_CHARS
from backend.schemas.resume import ResumeProfile, StrictModel
from backend.schemas.security import SecurityFlag
from backend.schemas.skill_gap import SkillGapReport

# Upper bound of the ``MAX_REVISIONS`` setting.
MAX_REVISIONS_LIMIT = 3

MetaKey = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]
MetaValue = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class _ReportModel(StrictModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


class CriticSummary(_ReportModel):
    """The Critic's final verdict for the report: ``{verdict, issues, revisions}``."""

    verdict: Literal["approve", "revise", "flagged"]
    issues: list[CriticIssue] = Field(max_length=MAX_CRITIC_ISSUES)
    revisions: int = Field(ge=0, le=MAX_REVISIONS_LIMIT)


class ReportMeta(_ReportModel):
    """Run metadata: model names, prompt versions and node timings (no content)."""

    models: dict[MetaKey, MetaValue]
    prompt_versions: dict[MetaKey, MetaValue]
    timings_ms: dict[MetaKey, Annotated[float, Field(ge=0.0)]]


class AnalysisReport(_ReportModel):
    report_id: UUID4
    created_at: AwareDatetime
    expires_at: AwareDatetime
    resume_profile: ResumeProfile
    interview_questions: InterviewQuestions
    match: MatchResult | None
    skill_gap: SkillGapReport | None
    critic: CriticSummary
    security_flags: list[SecurityFlag]
    meta: ReportMeta

    @model_validator(mode="after")
    def _check_consistency(self) -> AnalysisReport:
        lifetime = (self.expires_at - self.created_at).total_seconds()
        if not 0 < lifetime <= MAX_REPORT_TTL_SECONDS:
            raise ValueError("expires_at must be after created_at and within the maximum TTL")
        if (self.match is None) != (self.skill_gap is None):
            raise ValueError("match and skill_gap must both be present or both be null")
        return self


class AnalysisResponse(AnalysisReport):
    """``POST /api/v1/analyze`` response: the stored report plus, only when
    requested, the extracted-text preview (approved Rev. 3).

    The preview exists in this response only. The report store accepts exact
    ``AnalysisReport`` objects, so this model can never be stored.
    """

    preview: str | None = Field(default=None, max_length=PREVIEW_MAX_CHARS)
