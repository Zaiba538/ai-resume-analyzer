"""Interview agent: questions for the requested difficulty levels only.

Approved M4 decision B: ``count_per_level`` defaults to 3 and must be 1-5;
only the requested levels are generated (the schema sent to the model contains
just those levels, each with exactly ``count_per_level`` questions). Questions
are generated from the extracted profile, as in the legacy app.
"""

from __future__ import annotations

from collections.abc import Iterable

from backend.agents.base import AgentInputError, AgentResult, clean_profile
from backend.llm.prompts import build_messages
from backend.llm.prompts.interview import INTERVIEW_PROMPT
from backend.llm.structured import StructuredLLM
from backend.schemas.interview import (
    DEFAULT_COUNT_PER_LEVEL,
    LEVELS,
    MAX_COUNT_PER_LEVEL,
    MIN_COUNT_PER_LEVEL,
    InterviewQuestions,
    normalise_levels,
    question_set_model,
    with_ids,
)
from backend.schemas.resume import ResumeProfile


async def run_interview_agent(
    llm: StructuredLLM,
    profile: ResumeProfile,
    *,
    levels: Iterable[str] = LEVELS,
    count_per_level: int = DEFAULT_COUNT_PER_LEVEL,
) -> AgentResult[InterviewQuestions]:
    if isinstance(count_per_level, bool) or not isinstance(count_per_level, int):
        raise AgentInputError("count_per_level must be an integer.")
    if not MIN_COUNT_PER_LEVEL <= count_per_level <= MAX_COUNT_PER_LEVEL:
        raise AgentInputError(
            f"count_per_level must be between {MIN_COUNT_PER_LEVEL} and {MAX_COUNT_PER_LEVEL}."
        )
    try:
        requested = normalise_levels(levels)
    except ValueError as exc:
        raise AgentInputError("levels must be beginner, intermediate and/or advanced.") from exc

    schema = question_set_model(requested, count_per_level)
    profile_json = clean_profile(profile).model_dump_json()
    result = await llm.generate(
        "interview",
        schema,
        build_messages(
            INTERVIEW_PROMPT,
            {"profile": profile_json},
            params={"count_per_level": count_per_level},
        ),
        source_texts=[profile_json],
    )
    questions = InterviewQuestions(**with_ids(result.output))
    return AgentResult.from_structured(result, output=questions, prompt_id=INTERVIEW_PROMPT.id)
