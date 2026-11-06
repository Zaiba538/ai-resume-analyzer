"""Analysis workflow state (approved Rev. 2/3 ``AnalysisState``).

The sanitised resume and job description live in the state only for the
duration of one run; the graph has no checkpointer and the persisted report
holds derived data only. ``security_flags``, ``errors`` and ``timings_ms`` have
reducers so that parallel branches can each contribute to them.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from backend.schemas.critic import AgentName, CriticVerdict
from backend.schemas.interview import InterviewQuestions
from backend.schemas.match import MatchResult
from backend.schemas.report import AnalysisReport
from backend.schemas.resume import ResumeProfile
from backend.schemas.security import SecurityFlag
from backend.schemas.skill_gap import SkillGapReport

NodeName = Literal[
    "guard",
    "resume",
    "interview",
    "metrics",
    "skill_gap",
    "critic",
    "revise",
    "cap",
    "aggregate",
    "persist",
]


@dataclass(frozen=True)
class NodeError:
    """A non-fatal workflow notice (codes only, never content).

    Required agent and Critic failures are not recorded here: they abort the run
    (fail closed). This holds degraded but non-blocking conditions, such as the
    flag-only Prompt Guard being unavailable.
    """

    node: NodeName
    code: str


def merge_dicts[K, V](left: Mapping[K, V], right: Mapping[K, V]) -> dict[K, V]:
    """Reducer for parallel branches: later keys win, nothing is lost."""
    return {**left, **right}


class AnalysisState(TypedDict, total=False):
    request_id: str | None
    resume_text: str  # sanitised; this run only, never persisted
    job_description: str | None  # sanitised; this run only, never persisted
    sections: dict[str, str]
    security_flags: Annotated[list[SecurityFlag], operator.add]
    resume_profile: ResumeProfile | None
    interview_questions: InterviewQuestions | None
    match_result: MatchResult | None
    skill_gap: SkillGapReport | None
    critic_verdict: CriticVerdict | None
    critic_history: Annotated[list[CriticVerdict], operator.add]
    revision_count: int
    revise_targets: list[AgentName]
    errors: Annotated[list[NodeError], operator.add]
    timings_ms: Annotated[dict[str, float], merge_dicts]
    prompt_ids: Annotated[dict[str, str], merge_dicts]
    report_draft: dict[str, Any]  # derived fields only; identity and lifetime come from the store
    report_id: str | None
    report: AnalysisReport | None
