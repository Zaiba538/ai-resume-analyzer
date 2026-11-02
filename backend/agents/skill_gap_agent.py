"""Skill-gap agent: matching skills, missing skills, recommendations (legacy feature).

Uses the structured profile and the job description, as approved.

M7 (decision D1): optional Critic ``feedback`` for one revision. Only issues for
this agent are used, in an untrusted ``critic_feedback`` block with the revision
prompt; without them the request is exactly as before.
"""

from __future__ import annotations

from collections.abc import Sequence

from backend.agents.base import AgentResult, clean_feedback, clean_input, clean_profile
from backend.llm.prompts import build_messages
from backend.llm.prompts.skill_gap import SKILL_GAP_PROMPT, SKILL_GAP_REVISION_PROMPT
from backend.llm.structured import StructuredLLM
from backend.schemas.critic import CriticIssue
from backend.schemas.resume import ResumeProfile
from backend.schemas.skill_gap import SkillGapReport


async def run_skill_gap_agent(
    llm: StructuredLLM,
    profile: ResumeProfile,
    job_description: str,
    *,
    feedback: Sequence[CriticIssue] | None = None,
) -> AgentResult[SkillGapReport]:
    job = clean_input(job_description, field="job description")
    profile_json = clean_profile(profile).model_dump_json()
    notes = clean_feedback(feedback, "skill_gap")
    spec, untrusted = SKILL_GAP_PROMPT, {"profile": profile_json, "job_description": job}
    if notes is not None:
        spec, untrusted = SKILL_GAP_REVISION_PROMPT, {**untrusted, "critic_feedback": notes}
    # Feedback is not a source text: links in it never become "known" links.
    result = await llm.generate(
        "skill_gap",
        SkillGapReport,
        build_messages(spec, untrusted),
        source_texts=[profile_json, job],
    )
    return AgentResult.from_structured(result, output=result.output, prompt_id=spec.id)
