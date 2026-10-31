"""Skill-gap agent: matching skills, missing skills, recommendations (legacy feature).

Uses the structured profile and the job description, as approved.
"""

from __future__ import annotations

from backend.agents.base import AgentResult, clean_input, clean_profile
from backend.llm.prompts import build_messages
from backend.llm.prompts.skill_gap import SKILL_GAP_PROMPT
from backend.llm.structured import StructuredLLM
from backend.schemas.resume import ResumeProfile
from backend.schemas.skill_gap import SkillGapReport


async def run_skill_gap_agent(
    llm: StructuredLLM, profile: ResumeProfile, job_description: str
) -> AgentResult[SkillGapReport]:
    job = clean_input(job_description, field="job description")
    profile_json = clean_profile(profile).model_dump_json()
    result = await llm.generate(
        "skill_gap",
        SkillGapReport,
        build_messages(SKILL_GAP_PROMPT, {"profile": profile_json, "job_description": job}),
        source_texts=[profile_json, job],
    )
    return AgentResult.from_structured(
        result, output=result.output, prompt_id=SKILL_GAP_PROMPT.id
    )
