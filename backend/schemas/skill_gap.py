"""Skill-gap report produced by the Skill-Gap agent.

Preserves the legacy feature: matching skills, missing skills and
recommendations (approved ``AnalysisReport.skill_gap`` contract).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from backend.schemas.resume import ShortText, StrictModel

Recommendation = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)
]


class SkillGapReport(StrictModel):
    matching_skills: list[ShortText] = Field(max_length=100)
    missing_skills: list[ShortText] = Field(max_length=100)
    recommendations: list[Recommendation] = Field(max_length=15)
