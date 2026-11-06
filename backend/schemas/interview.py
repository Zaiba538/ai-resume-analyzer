"""Interview questions and answer evaluation.

Questions preserve the legacy feature: beginner, intermediate and advanced.
The model generates ``QuestionDraft`` items; the server assigns stable IDs to
form the API contract ``Q = {id, question, topic, rationale}``.

Requested levels only (approved M4 decision B): ``question_set_model`` builds a
strict schema containing just the requested levels, each with exactly
``count_per_level`` questions, so the model is never asked for, and never
returns, levels that were not requested.

Answer evaluation (approved M4 decision A): five fixed criteria, each rated by
the model as an integer 1-5 with a one-sentence justification. The model never
produces ``overall_rating``; Python computes it as the arithmetic mean of the
five ratings, rounded to one decimal place.
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, StrictInt, StringConstraints, create_model, model_validator

from backend.schemas.resume import StrictModel

Level = Literal["beginner", "intermediate", "advanced"]
LEVELS: tuple[Level, ...] = ("beginner", "intermediate", "advanced")

MAX_QUESTIONS_PER_LEVEL = 10
DEFAULT_COUNT_PER_LEVEL = 3
MIN_COUNT_PER_LEVEL = 1
MAX_COUNT_PER_LEVEL = 5

QuestionText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=5, max_length=500)
]
Topic = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
Rationale = Annotated[str, StringConstraints(strip_whitespace=True, max_length=300)]


class QuestionDraft(StrictModel):
    question: QuestionText
    topic: Topic
    rationale: Rationale


class _QuestionSetBase(StrictModel):
    """Rejects a question repeated anywhere in the set (case-insensitive)."""

    @model_validator(mode="after")
    def _no_duplicate_questions(self) -> _QuestionSetBase:
        seen: set[str] = set()
        for level in LEVELS:
            for draft in getattr(self, level, None) or ():
                key = " ".join(draft.question.casefold().split())
                if key in seen:
                    raise ValueError("duplicate interview question")
                seen.add(key)
        return self


class InterviewQuestionSet(StrictModel):
    """All three levels (1-10 questions each). M3 model, unchanged."""

    beginner: list[QuestionDraft] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_LEVEL)
    intermediate: list[QuestionDraft] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_LEVEL)
    advanced: list[QuestionDraft] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_LEVEL)


def normalise_levels(levels: Iterable[str]) -> tuple[Level, ...]:
    """Distinct requested levels in canonical order; raises ValueError if invalid."""
    requested = set(levels)
    unknown = requested - set(LEVELS)
    if unknown or not requested:
        raise ValueError("levels must be a non-empty subset of beginner/intermediate/advanced")
    return tuple(level for level in LEVELS if level in requested)


@lru_cache(maxsize=64)
def question_set_model(levels: tuple[Level, ...], count: int) -> type[_QuestionSetBase]:
    """Strict schema with only ``levels``, each holding exactly ``count`` questions."""
    if normalise_levels(levels) != levels:
        raise ValueError("levels must be distinct and in canonical order")
    if not MIN_COUNT_PER_LEVEL <= count <= MAX_COUNT_PER_LEVEL:
        raise ValueError("count out of range")
    fields: dict[str, object] = {
        level: (list[QuestionDraft], Field(min_length=count, max_length=count)) for level in levels
    }
    name = "InterviewQuestions_" + "_".join(levels) + f"_{count}"
    return create_model(name, __base__=_QuestionSetBase, **fields)  # type: ignore[call-overload]


class InterviewQuestion(QuestionDraft):
    """API representation with a server-assigned ID."""

    id: str = Field(pattern=r"^(beginner|intermediate|advanced)-\d{1,2}$")


class InterviewQuestions(StrictModel):
    """Agent output: questions with IDs; levels that were not requested are empty."""

    beginner: list[InterviewQuestion] = Field(max_length=MAX_QUESTIONS_PER_LEVEL)
    intermediate: list[InterviewQuestion] = Field(max_length=MAX_QUESTIONS_PER_LEVEL)
    advanced: list[InterviewQuestion] = Field(max_length=MAX_QUESTIONS_PER_LEVEL)


def with_ids(drafts: StrictModel) -> dict[str, list[InterviewQuestion]]:
    """Assign stable IDs such as ``intermediate-2`` (1-based within a level).

    Levels absent from ``drafts`` map to empty lists.
    """
    return {
        level: [
            InterviewQuestion(id=f"{level}-{index}", **draft.model_dump())
            for index, draft in enumerate(getattr(drafts, level, None) or (), start=1)
        ]
        for level in LEVELS
    }


# --- Answer evaluation --------------------------------------------------------

RUBRIC_CRITERIA = ("relevance", "technical_accuracy", "depth", "clarity", "structure")
MIN_RATING = 1
MAX_RATING = 5

Justification = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=300, pattern=r"^[^\r\n]+$"),
]
FeedbackItem = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)
]


class CriterionScore(StrictModel):
    rating: StrictInt = Field(ge=MIN_RATING, le=MAX_RATING)
    justification: Justification


class Rubric(StrictModel):
    relevance: CriterionScore
    technical_accuracy: CriterionScore
    depth: CriterionScore
    clarity: CriterionScore
    structure: CriterionScore

    def ratings(self) -> tuple[int, ...]:
        return tuple(getattr(self, criterion).rating for criterion in RUBRIC_CRITERIA)


class AnswerEvaluationDraft(StrictModel):
    """What the model produces. There is deliberately no ``overall_rating``."""

    rubric: Rubric
    strengths: list[FeedbackItem] = Field(max_length=5)
    improvements: list[FeedbackItem] = Field(max_length=5)
    model_answer_outline: list[FeedbackItem] = Field(max_length=8)


def mean_rating(ratings: Iterable[int]) -> float:
    """Arithmetic mean rounded half-up to one decimal place."""
    values = list(ratings)
    if not values:
        raise ValueError("at least one rating is required")
    mean = Decimal(sum(values)) / Decimal(len(values))
    return float(mean.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


class AnswerEvaluation(AnswerEvaluationDraft):
    """API representation: the draft plus the server-computed overall rating."""

    overall_rating: float = Field(ge=MIN_RATING, le=MAX_RATING)

    @model_validator(mode="after")
    def _overall_matches_rubric(self) -> AnswerEvaluation:
        if self.overall_rating != mean_rating(self.rubric.ratings()):
            raise ValueError("overall_rating must equal the mean of the rubric ratings")
        return self

    @classmethod
    def from_draft(cls, draft: AnswerEvaluationDraft) -> AnswerEvaluation:
        return cls(**draft.model_dump(), overall_rating=mean_rating(draft.rubric.ratings()))
