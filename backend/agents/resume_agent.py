"""Resume agent: extract a structured profile (preserved legacy feature)."""

from __future__ import annotations

from backend.agents.base import AgentResult, clean_input
from backend.llm.prompts import build_messages
from backend.llm.prompts.resume import RESUME_PROMPT
from backend.llm.structured import StructuredLLM
from backend.schemas.resume import ResumeProfile


async def run_resume_agent(llm: StructuredLLM, resume_text: str) -> AgentResult[ResumeProfile]:
    resume = clean_input(resume_text, field="resume")
    result = await llm.generate(
        "resume",
        ResumeProfile,
        build_messages(RESUME_PROMPT, {"resume": resume}),
        source_texts=[resume],
    )
    return AgentResult.from_structured(result, output=result.output, prompt_id=RESUME_PROMPT.id)
