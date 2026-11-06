"""M7 phase 3 (decision D1): optional Critic feedback for the resume, interview and
skill-gap agents, through the real SDK against the fake Groq API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from backend.agents.base import AgentInputError
from backend.agents.interview_agent import run_interview_agent
from backend.agents.resume_agent import run_resume_agent
from backend.agents.skill_gap_agent import run_skill_gap_agent
from backend.llm.errors import InvalidLLMOutputError, UpstreamTimeoutError
from backend.llm.prompts import REVISION_GUIDANCE, UNTRUSTED_DATA_RULES, PromptSpec
from backend.llm.prompts.interview import INTERVIEW_PROMPT, INTERVIEW_REVISION_PROMPT
from backend.llm.prompts.resume import RESUME_PROMPT, RESUME_REVISION_PROMPT
from backend.llm.prompts.skill_gap import SKILL_GAP_PROMPT, SKILL_GAP_REVISION_PROMPT
from backend.llm.structured import REPAIR_INSTRUCTION
from backend.schemas.critic import CriticIssue, ReviewIssue
from tests.fixtures.groq import FakeGroqChat, Reply, json_reply
from tests.unit.test_agents import (
    JOB,
    PROFILE,
    PROFILE_PAYLOAD,
    RESUME,
    make_llm,
    questions,
    system_and_user,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _fast_sdk_backoff(monkeypatch) -> None:
    import groq._base_client as base

    monkeypatch.setattr(base, "INITIAL_RETRY_DELAY", 0.001)
    monkeypatch.setattr(base, "MAX_RETRY_DELAY", 0.002)


GAP_PAYLOAD = {
    "matching_skills": ["Python", "SQL"],
    "missing_skills": ["Kubernetes", "Airflow"],
    "recommendations": ["Build and schedule a small Airflow DAG."],
}
INTERVIEW_PAYLOAD = {lvl: questions(lvl, 3) for lvl in ("beginner", "intermediate", "advanced")}


@dataclass(frozen=True)
class AgentCase:
    name: str
    base: PromptSpec
    revision: PromptSpec
    payload: dict[str, Any]

    async def run(self, fake: FakeGroqChat, **kwargs):
        llm = make_llm(fake)
        if self.name == "resume":
            return await run_resume_agent(llm, RESUME, **kwargs)
        if self.name == "interview":
            return await run_interview_agent(llm, PROFILE, **kwargs)
        return await run_skill_gap_agent(llm, PROFILE, JOB, **kwargs)


CASES = [
    AgentCase("resume", RESUME_PROMPT, RESUME_REVISION_PROMPT, PROFILE_PAYLOAD),
    AgentCase("interview", INTERVIEW_PROMPT, INTERVIEW_REVISION_PROMPT, INTERVIEW_PAYLOAD),
    AgentCase("skill_gap", SKILL_GAP_PROMPT, SKILL_GAP_REVISION_PROMPT, GAP_PAYLOAD),
]
by_name = pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)


def issue(
    agent: str,
    field: str = "skills[0]",
    message: str = "Not supported by the data.",
    severity: str = "high",
) -> CriticIssue:
    return CriticIssue(agent=agent, field=field, severity=severity, message=message)


def feedback_block(fake: FakeGroqChat, index: int = 0) -> list[dict]:
    _, user = system_and_user(fake, index)
    return json.loads(user.split("<critic_feedback>\n", 1)[1].split("\n</critic_feedback>", 1)[0])


# --- prompt versions ------------------------------------------------------------------


async def test_revision_prompt_versions() -> None:
    assert [(c.base.id, c.revision.id) for c in CASES] == [
        ("resume@1.0.0", "resume@1.1.0"),
        ("interview@1.1.0", "interview@1.2.0"),
        ("skill_gap@1.0.0", "skill_gap@1.1.0"),
    ]


@by_name
async def test_revision_prompt_extends_the_unchanged_base(case: AgentCase) -> None:
    assert case.revision.system == f"{case.base.system}\n{REVISION_GUIDANCE}"
    assert case.revision.inputs == (*case.base.inputs, "critic_feedback")
    assert "critic_feedback" not in case.base.inputs
    assert "{" not in REVISION_GUIDANCE  # no template parameters in the guidance


# --- no feedback: exactly the previous behaviour --------------------------------------


@by_name
async def test_no_feedback_requests_are_byte_identical(case: AgentCase) -> None:
    bodies = []
    for kwargs in ({}, {"feedback": None}, {"feedback": []}, {"feedback": ()}):
        fake = FakeGroqChat().queue(json_reply(case.payload))
        result = await case.run(fake, **kwargs)
        assert result.prompt_id == case.base.id
        bodies.append(fake.requests[0].content)
    assert len(set(bodies)) == 1
    system, user = system_and_user(fake)
    assert system == f"{case.base.system}\n\n{UNTRUSTED_DATA_RULES}".replace(
        "{count_per_level}", "3"
    )
    assert "critic_feedback" not in user and REVISION_GUIDANCE not in system


@by_name
async def test_feedback_only_for_other_agents_is_ignored(case: AgentCase) -> None:
    others = [issue(name) for name in ("resume", "interview", "skill_gap") if name != case.name]
    plain, filtered = FakeGroqChat(), FakeGroqChat()
    plain.queue(json_reply(case.payload))
    filtered.queue(json_reply(case.payload))
    await case.run(plain)
    result = await case.run(filtered, feedback=others)
    assert result.prompt_id == case.base.id
    assert filtered.requests[0].content == plain.requests[0].content


# --- with feedback --------------------------------------------------------------------


@by_name
async def test_valid_feedback_uses_the_revision_prompt_and_block(case: AgentCase) -> None:
    fake = FakeGroqChat().queue(json_reply(case.payload))
    result = await case.run(fake, feedback=[issue(case.name)])
    assert result.prompt_id == case.revision.id
    system, user = system_and_user(fake)
    assert REVISION_GUIDANCE in system
    assert user.rstrip().endswith("</critic_feedback>")  # last block, after the data
    assert feedback_block(fake) == [
        {"field": "skills[0]", "severity": "high", "message": "Not supported by the data."}
    ]
    # Model, temperature, reasoning effort, token limit and schema are unchanged.
    plain = FakeGroqChat().queue(json_reply(case.payload))
    await case.run(plain)
    settings = {k: v for k, v in fake.bodies[0].items() if k != "messages"}
    assert settings == {k: v for k, v in plain.bodies[0].items() if k != "messages"}


@by_name
async def test_multiple_issues_and_fields_kept_in_order_and_filtered(case: AgentCase) -> None:
    feedback = [
        issue(case.name, field="a[0]", message="First problem.", severity="high"),
        issue("resume" if case.name != "resume" else "interview", message="Other agent."),
        issue(case.name, field="b.c[2]", message="Second problem.", severity="medium"),
        issue(case.name, field="d", message="Third problem.", severity="low"),
    ]
    fake = FakeGroqChat().queue(json_reply(case.payload))
    await case.run(fake, feedback=feedback)
    assert feedback_block(fake) == [
        {"field": "a[0]", "severity": "high", "message": "First problem."},
        {"field": "b.c[2]", "severity": "medium", "message": "Second problem."},
        {"field": "d", "severity": "low", "message": "Third problem."},
    ]


@by_name
async def test_review_issue_category_is_not_forwarded(case: AgentCase) -> None:
    review = ReviewIssue(
        agent=case.name, field="x", severity="high", message="Problem.", category="contradiction"
    )
    fake = FakeGroqChat().queue(json_reply(case.payload))
    await case.run(fake, feedback=[review])
    assert feedback_block(fake) == [{"field": "x", "severity": "high", "message": "Problem."}]
    assert "contradiction" not in system_and_user(fake)[1]


@by_name
async def test_revised_output_is_strictly_validated_and_repaired_once(case: AgentCase) -> None:
    fake = FakeGroqChat().queue(json_reply({"unexpected": True}), json_reply(case.payload))
    result = await case.run(fake, feedback=[issue(case.name)])
    assert result.attempts == 2 and result.prompt_id == case.revision.id
    repair = fake.bodies[1]["messages"]
    assert repair[-1]["content"] == REPAIR_INSTRUCTION
    assert "<critic_feedback>" in repair[1]["content"]  # repair keeps the same feedback


@by_name
async def test_revised_output_invalid_twice_is_a_typed_error(case: AgentCase) -> None:
    fake = FakeGroqChat().queue(json_reply({"bad": 1}), json_reply({"bad": 2}))
    with pytest.raises(InvalidLLMOutputError):
        await case.run(fake, feedback=[issue(case.name)])


@by_name
async def test_upstream_timeout_is_unchanged_with_feedback(case: AgentCase) -> None:
    fake = FakeGroqChat().queue(*(Reply(exception=httpx.ReadTimeout("slow")) for _ in range(3)))
    with pytest.raises(UpstreamTimeoutError):
        await case.run(fake, feedback=[issue(case.name)])


@by_name
async def test_generator_feedback_is_rejected_before_any_request(case: AgentCase) -> None:
    """An unsized iterable is not a list of issues: typed 422, never a TypeError."""
    valid = issue(case.name)
    fake = FakeGroqChat()
    with pytest.raises(AgentInputError) as info:
        await case.run(fake, feedback=(item for item in [valid]))
    # AppError's str() is its code; the guard's own message is on ``.message``.
    assert info.value.code == "invalid_agent_input"
    assert info.value.message == "Critic feedback must be a list of Critic issues."
    assert fake.requests == []


async def test_interview_revision_keeps_requested_levels_and_count() -> None:
    payload = {"advanced": questions("advanced", 2)}
    fake = FakeGroqChat().queue(json_reply(payload))
    result = await run_interview_agent(
        make_llm(fake),
        PROFILE,
        levels=["advanced"],
        count_per_level=2,
        feedback=[issue("interview", field="advanced[1]")],
    )
    assert result.prompt_id == "interview@1.2.0"
    assert len(result.output.advanced) == 2 and result.output.beginner == []
    schema = fake.bodies[0]["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"]) == ["advanced"]
    assert "exactly 2 questions" in system_and_user(fake)[0]
