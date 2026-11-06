"""Synthetic ``AnalysisReport`` builders (no real personal data)."""

from __future__ import annotations

from typing import Any

from backend.schemas.critic import CriticIssue
from backend.schemas.interview import InterviewQuestion, InterviewQuestions
from backend.schemas.match import (
    Baseline,
    MatchMetrics,
    MatchResult,
    SectionCompletenessMetric,
    SemanticSimilarityMetric,
    SkillCoverageMetric,
)
from backend.schemas.report import AnalysisReport, CriticSummary, ReportMeta
from backend.schemas.resume import ResumeProfile
from backend.schemas.security import SecurityFlag
from backend.schemas.skill_gap import SkillGapReport
from backend.scoring.ats_composite import evaluate_ats_gate
from backend.storage.report_store import ReportStamp

PROFILE = ResumeProfile.model_validate(
    {
        "skills": ["Python", "SQL", "Apache Spark"],
        "education": [
            {
                "qualification": "B.Sc. Computer Science",
                "institution": "Example State University",
                "year": "2020",
            }
        ],
        "experience": [
            {
                "organization": "Acme Analytics (fictional)",
                "role": "Data Engineer",
                "period": "2021 - 2024",
                "highlights": ["Built batch ETL pipelines in Apache Spark."],
            }
        ],
        "projects": ["Inventory forecaster"],
        "certifications": [],
    }
)


def questions() -> InterviewQuestions:
    def q(level: str, n: int, text: str) -> InterviewQuestion:
        return InterviewQuestion(
            id=f"{level}-{n}", question=text, topic="Python", rationale="Listed skill."
        )

    return InterviewQuestions(
        beginner=[q("beginner", 1, "What is a Python list?")],
        intermediate=[q("intermediate", 1, "How would you tune a Spark job?")],
        advanced=[q("advanced", 1, "Design a fault-tolerant ETL pipeline?")],
    )


def match_result() -> MatchResult:
    return MatchResult(
        metrics=MatchMetrics(
            semantic_similarity=SemanticSimilarityMetric(value=61.5),
            skill_coverage=SkillCoverageMetric(
                value=50.0,
                matched=["Python", "SQL"],
                missing=["Kubernetes", "Airflow"],
                jd_skills_total=4,
            ),
            section_completeness=SectionCompletenessMetric(
                value=80.0,
                present=["skills", "experience", "education", "projects"],
                absent=["certifications"],
            ),
        ),
        baseline=Baseline(tfidf_keyword_match=22.0),
        ats_score=evaluate_ats_gate(),
    )


def skill_gap() -> SkillGapReport:
    return SkillGapReport(
        matching_skills=["Python", "SQL"],
        missing_skills=["Kubernetes", "Airflow"],
        recommendations=["Build and schedule a small Airflow DAG."],
    )


def report_data(stamp: ReportStamp, *, with_job: bool = True) -> dict[str, Any]:
    return {
        "report_id": stamp.report_id,
        "created_at": stamp.created_at,
        "expires_at": stamp.expires_at,
        "resume_profile": PROFILE,
        "interview_questions": questions(),
        "match": match_result() if with_job else None,
        "skill_gap": skill_gap() if with_job else None,
        "critic": CriticSummary(
            verdict="approve",
            issues=[
                CriticIssue(
                    agent="interview", field="advanced[0]", severity="low", message="Minor wording."
                )
            ],
            revisions=0,
        ),
        "security_flags": [SecurityFlag(type="instruction_override", source="resume")],
        "meta": ReportMeta(
            models={"resume": "openai/gpt-oss-120b"},
            prompt_versions={"resume": "resume@1.0.0", "critic": "critic@1.1.0"},
            timings_ms={"resume": 812.5, "critic": 1203.0},
        ),
    }


def synthetic_report(stamp: ReportStamp, *, with_job: bool = True, **overrides: Any):
    return AnalysisReport(**{**report_data(stamp, with_job=with_job), **overrides})
