"""Critic verdict (approved shape, Rev. 2/3):
``CriticVerdict {verdict, issues[{agent, field, severity, message}], revise_targets[]}``.

The verdict is derived by the server from validated issues (approved M6
decision Q1). The LLM never chooses it: its strict output contract is
``CriticReviewDraft``, a list of categorised issues. The category is used
server-side only and is not part of the published ``CriticIssue``.

An ``injection_influence`` category does not by itself make a verdict
``flagged``. Only confirmed injection influence in generated outputs does:
deterministic evidence in the output, a matching injection signal in the
inputs, and an LLM judgement on the same field (see ``critic_agent``).
Otherwise such an issue is an ordinary issue and, at medium or high severity,
leads to ``revise``. A ``revise`` still unresolved after ``MAX_REVISIONS`` also
becomes ``flagged``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from backend.schemas.resume import StrictModel

AgentName = Literal["resume", "interview", "skill_gap"]
Severity = Literal["low", "medium", "high"]
MAX_ISSUE_MESSAGE = 400


class CriticIssue(StrictModel):
    agent: AgentName
    field: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    severity: Severity
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_ISSUE_MESSAGE)
    ]


class CriticVerdict(StrictModel):
    verdict: Literal["approve", "revise", "flagged"]
    issues: list[CriticIssue] = Field(max_length=30)
    revise_targets: list[AgentName] = Field(max_length=3)


MAX_CRITIC_ISSUES = 30

ReviewCategory = Literal[
    "unsupported_claim",
    "contradiction",
    "irrelevant_recommendation",
    "irrelevant_question",
    "incomplete_output",
    "schema_violation",
    "inconsistency",
    "injection_influence",
]


class ReviewIssue(CriticIssue):
    """An issue with its category, used to derive the verdict server-side."""

    category: ReviewCategory

    def published(self) -> CriticIssue:
        return CriticIssue(
            agent=self.agent, field=self.field, severity=self.severity, message=self.message
        )


class CriticReviewDraft(StrictModel):
    """Strict LLM output contract for the Critic: issues only, never a verdict."""

    issues: list[ReviewIssue] = Field(max_length=MAX_CRITIC_ISSUES)
