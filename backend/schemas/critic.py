"""Critic verdict (approved shape, Rev. 2/3):
``CriticVerdict {verdict, issues[{agent, field, severity, message}], revise_targets[]}``.

The Critic agent and its prompt are implemented in M6; the schema ships with
the other LLM output schemas in M3 so that output validation covers it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from backend.schemas.resume import StrictModel

AgentName = Literal["resume", "interview", "skill_gap"]


class CriticIssue(StrictModel):
    agent: AgentName
    field: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    severity: Literal["low", "medium", "high"]
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)]


class CriticVerdict(StrictModel):
    verdict: Literal["approve", "revise", "flagged"]
    issues: list[CriticIssue] = Field(max_length=30)
    revise_targets: list[AgentName] = Field(max_length=3)
