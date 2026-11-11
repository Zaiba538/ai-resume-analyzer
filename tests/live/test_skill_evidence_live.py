"""Opt-in live check of M11 Phase 2 (F1-F4) against real Groq (synthetic data only).

    GROQ_API_KEY=... pytest -m live tests/live/test_skill_evidence_live.py -s

The offline tests prove the prompts, payloads and server-side rules; only this
test shows whether the models actually follow the new instructions. Results are
probabilistic, so a single pass or failure is a signal, not proof.

Only skill names, verdicts, agents, fields and severities are printed, never
prompts or documents.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from backend.agents.critic_agent import CriticInput, run_critic_agent
from backend.agents.resume_agent import run_resume_agent
from backend.agents.skill_gap_agent import run_skill_gap_agent
from backend.core.config import Settings
from backend.llm.client import LLMClients
from backend.llm.structured import StructuredLLM
from tests.unit.test_skill_evidence import DS_JOB, DS_RESUME

pytestmark = [pytest.mark.live, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def llm():
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    clients = LLMClients(Settings(_env_file=None))
    yield StructuredLLM(clients)
    await clients.aclose()


def _mentions(skills: list[str], *words: str) -> bool:
    return any(word in skill.casefold() for skill in skills for word in words)


async def test_experience_evidence_and_preferred_skills(llm) -> None:
    profile = (await run_resume_agent(llm, DS_RESUME)).output
    gap_result = await run_skill_gap_agent(llm, profile, DS_JOB)
    gap = gap_result.output
    print(f"\nmatching={gap.matching_skills} missing={gap.missing_skills}")
    assert gap_result.prompt_id == "skill_gap@1.2.0"
    # F1: preprocessing is evidenced in Experience, so it is not a gap.
    assert not _mentions(gap.missing_skills, "preprocess", "cleaning")
    # F1: preferred skills (Tableau, Docker) are never reported as missing.
    assert not _mentions(gap.missing_skills, "tableau", "docker")

    review = await run_critic_agent(
        llm,
        CriticInput(resume_text=DS_RESUME, profile=profile, job_description=DS_JOB, skill_gap=gap),
    )
    verdict = review.output
    print(f"critic: verdict={verdict.verdict} issues="
          f"{[(i.agent, i.field, i.severity) for i in verdict.issues]}")  # fmt: skip
    assert review.prompt_id == "critic@1.5.0"
    assert verdict.verdict != "flagged"
