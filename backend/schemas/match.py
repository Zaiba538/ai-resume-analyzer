"""Models for ``POST /api/v1/match`` (approved Rev. 3 ``MatchResult`` contract).

Three separate metrics, the legacy TF-IDF baseline, and the ATS status. The
ATS model admits only the disabled state in M5; widening it is an M12 change.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Percent = Annotated[float, Field(ge=0.0, le=100.0)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SemanticSimilarityMetric(_Model):
    value: Percent
    calibrated: Literal[False] = False


class SkillCoverageMetric(_Model):
    value: Percent
    matched: list[str]
    missing: list[str]
    jd_skills_total: int = Field(ge=0)


class SectionCompletenessMetric(_Model):
    value: Percent
    present: list[str]
    absent: list[str]


class MatchMetrics(_Model):
    semantic_similarity: SemanticSimilarityMetric
    skill_coverage: SkillCoverageMetric
    section_completeness: SectionCompletenessMetric


class Baseline(_Model):
    tfidf_keyword_match: Percent


class AtsScore(_Model):
    """ATS composite status. M5: only ``not_calibrated``, every other field null."""

    status: Literal["not_calibrated"] = "not_calibrated"
    score: None = None
    calibration_version: None = None
    evaluation_summary: None = None


class MatchResult(_Model):
    metrics: MatchMetrics
    baseline: Baseline
    ats_score: AtsScore


class MatchTextRequest(_Model):
    """JSON request form: resume as plain text."""

    resume_text: str
    job_description: str
