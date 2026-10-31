"""Structured resume profile produced by the Resume agent.

Shape follows the M0 live spike (validated against Groq strict JSON-schema
output on both gpt-oss models) and covers the preserved legacy feature:
skills, education, experience, projects and certifications.

Every field is required (nullable where optional) and objects forbid extra
keys, as Groq's strict structured-output mode requires. Length and count limits
are enforced locally after generation (see ``backend.llm.structured``).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
LongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Education(StrictModel):
    qualification: ShortText
    institution: str = Field(max_length=200)
    year: str | None = Field(max_length=40)


class Experience(StrictModel):
    organization: str = Field(max_length=200)
    role: str = Field(max_length=200)
    period: str | None = Field(max_length=80)
    highlights: list[LongText] = Field(max_length=12)


class ResumeProfile(StrictModel):
    skills: list[ShortText] = Field(max_length=150)
    education: list[Education] = Field(max_length=20)
    experience: list[Experience] = Field(max_length=40)
    projects: list[LongText] = Field(max_length=40)
    certifications: list[ShortText] = Field(max_length=40)
