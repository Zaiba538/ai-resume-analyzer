"""Interview questions produced by the Interview agent.

Preserves the legacy feature: beginner, intermediate and advanced questions.
The model generates ``QuestionDraft`` items; the server assigns stable IDs to
form the API contract ``Q = {id, question, topic, rationale}``.

The answer-evaluation schema is deferred to M4: its rubric criteria and rating
scale have not been decided.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from backend.schemas.resume import StrictModel

MAX_QUESTIONS_PER_LEVEL = 10

QuestionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=5, max_length=500)
]
Topic = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Rationale = Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]


class QuestionDraft(StrictModel):
    question: QuestionText
    topic: Topic
    rationale: Rationale


class InterviewQuestionSet(StrictModel):
    """LLM output: questions grouped by the three preserved difficulty levels."""

    beginner: list[QuestionDraft] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_LEVEL)
    intermediate: list[QuestionDraft] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_LEVEL)
    advanced: list[QuestionDraft] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_LEVEL)


class InterviewQuestion(QuestionDraft):
    """API representation with a server-assigned ID."""

    id: str = Field(pattern=r"^(beginner|intermediate|advanced)-\d{1,2}$")


def with_ids(drafts: InterviewQuestionSet) -> dict[str, list[InterviewQuestion]]:
    """Assign stable IDs such as ``intermediate-2`` (1-based within a level)."""
    return {
        level: [
            InterviewQuestion(id=f"{level}-{index}", **draft.model_dump())
            for index, draft in enumerate(getattr(drafts, level), start=1)
        ]
        for level in ("beginner", "intermediate", "advanced")
    }
