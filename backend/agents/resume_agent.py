"""Resume agent: extract a structured profile (preserved legacy feature).

M7 (decision D1): optional Critic ``feedback`` for one revision. Only issues for
this agent are used, in an untrusted ``critic_feedback`` block with the revision
prompt; without them the request is exactly as before.
"""

from __future__ import annotations

from collections.abc import Sequence

from backend.agents.base import AgentResult, clean_feedback, clean_input
from backend.llm.prompts import build_messages
from backend.llm.prompts.resume import RESUME_PROMPT, RESUME_REVISION_PROMPT
from backend.llm.structured import StructuredLLM
from backend.schemas.critic import CriticIssue
from backend.schemas.resume import ResumeProfile


async def run_resume_agent(
    llm: StructuredLLM,
    resume_text: str,
    *,
    feedback: Sequence[CriticIssue] | None = None,
) -> AgentResult[ResumeProfile]:
    resume = clean_input(resume_text, field="resume")
    notes = clean_feedback(feedback, "resume")
    spec, untrusted = RESUME_PROMPT, {"resume": resume}
    if notes is not None:
        spec, untrusted = RESUME_REVISION_PROMPT, {**untrusted, "critic_feedback": notes}
    # Feedback is not a source text: links in it never become "known" links.
    result = await llm.generate(
        "resume",
        ResumeProfile,
        build_messages(spec, untrusted),
        source_texts=[resume],
    )
    return AgentResult.from_structured(result, output=result.output, prompt_id=spec.id)
