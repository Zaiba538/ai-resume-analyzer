"""The four M4 agents, run through the real ChatGroq/groq SDK against a fake Groq API."""

from __future__ import annotations

import httpx
import pytest

from backend.agents.answer_evaluator_agent import run_answer_evaluator_agent
from backend.agents.base import AgentInputError
from backend.agents.interview_agent import run_interview_agent
from backend.agents.resume_agent import run_resume_agent
from backend.agents.skill_gap_agent import run_skill_gap_agent
from backend.core.config import Settings
from backend.llm.client import LLMClients
from backend.llm.errors import InvalidLLMOutputError, UpstreamTimeoutError
from backend.llm.prompts.evaluator import EVALUATOR_PROMPT
from backend.llm.prompts.interview import INTERVIEW_PROMPT
from backend.llm.prompts.resume import RESUME_PROMPT
from backend.llm.prompts.skill_gap import SKILL_GAP_PROMPT
from backend.llm.structured import REPAIR_INSTRUCTION, StructuredLLM
from backend.schemas.interview import RUBRIC_CRITERIA
from backend.schemas.resume import ResumeProfile
from tests.conftest import FAKE_API_KEY
from tests.fixtures.documents import SYNTHETIC_RESUME_LINES
from tests.fixtures.groq import FakeGroqChat, Reply, json_reply

RESUME = "\n".join(SYNTHETIC_RESUME_LINES)
JOB = "Data Engineer (fictional). Required: Python, SQL, Kubernetes, Airflow."
PROFILE = ResumeProfile.model_validate(
    {
        "skills": ["Python", "SQL", "Apache Spark"],
        "education": [
            {"qualification": "B.Sc. Computer Science", "institution": "Example U", "year": "2021"}
        ],
        "experience": [
            {
                "organization": "Acme Analytics (fictional)",
                "role": "Data Engineer",
                "period": "2021 - 2024",
                "highlights": ["Built batch ETL pipelines in Apache Spark"],
            }
        ],
        "projects": ["Inventory forecaster"],
        "certifications": [],
    }
)
PROFILE_PAYLOAD = PROFILE.model_dump()


@pytest.fixture(autouse=True)
def _fast_sdk_backoff(monkeypatch) -> None:
    import groq._base_client as base

    monkeypatch.setattr(base, "INITIAL_RETRY_DELAY", 0.001)
    monkeypatch.setattr(base, "MAX_RETRY_DELAY", 0.002)


def make_llm(fake: FakeGroqChat) -> StructuredLLM:
    settings = Settings(_env_file=None, groq_api_key=FAKE_API_KEY)
    return StructuredLLM(LLMClients(settings, http_client=fake.client()))


def questions(level: str, count: int) -> list[dict]:
    return [
        {
            "question": f"{level.title()} question number {i} about Python?",
            "topic": "Python",
            "rationale": "Python is a core skill in the profile.",
        }
        for i in range(1, count + 1)
    ]


def evaluation(*ratings: int) -> dict:
    ratings = ratings or (4, 3, 3, 4, 4)
    return {
        "rubric": {
            name: {"rating": r, "justification": "The answer covers this reasonably well."}
            for name, r in zip(RUBRIC_CRITERIA, ratings, strict=True)
        },
        "strengths": ["Correct core definition"],
        "improvements": ["Discuss memory trade-offs"],
        "model_answer_outline": ["Define generators", "Explain lazy evaluation", "Give an example"],
    }


def schema_sent(fake: FakeGroqChat, index: int = 0) -> dict:
    return fake.bodies[index]["response_format"]["json_schema"]["schema"]


def system_and_user(fake: FakeGroqChat, index: int = 0) -> tuple[str, str]:
    messages = fake.bodies[index]["messages"]
    assert [m["role"] for m in messages[:2]] == ["system", "user"]
    return messages[0]["content"], messages[1]["content"]


# --- resume agent -------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_agent_success() -> None:
    fake = FakeGroqChat().queue(json_reply(PROFILE_PAYLOAD))
    result = await run_resume_agent(make_llm(fake), RESUME)
    assert result.output == PROFILE
    assert result.prompt_id == RESUME_PROMPT.id
    assert (result.attempts, result.flags) == (1, ())
    assert fake.bodies[0]["model"] == "openai/gpt-oss-120b"
    system, user = system_and_user(fake)
    assert "Jordan Sample" in user and "Jordan Sample" not in system


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", ["", "   ", "\u200b\u200b", "\x00\x07"])
async def test_resume_agent_rejects_empty_input_without_calling_llm(empty: str) -> None:
    fake = FakeGroqChat()
    with pytest.raises(AgentInputError) as info:
        await run_resume_agent(make_llm(fake), empty)
    assert info.value.status_code == 422 and info.value.code == "invalid_agent_input"
    assert fake.requests == []


@pytest.mark.asyncio
async def test_resume_agent_repairs_then_fails() -> None:
    fake = FakeGroqChat().queue(Reply("not json"), json_reply({**PROFILE_PAYLOAD, "extra": 1}))
    with pytest.raises(InvalidLLMOutputError):
        await run_resume_agent(make_llm(fake), RESUME)
    assert len(fake.requests) == 2
    assert fake.bodies[1]["messages"][-1]["content"] == REPAIR_INSTRUCTION


# --- interview agent ----------------------------------------------------------


@pytest.mark.asyncio
async def test_interview_defaults_three_levels_three_each() -> None:
    reply = {lvl: questions(lvl, 3) for lvl in ("beginner", "intermediate", "advanced")}
    fake = FakeGroqChat().queue(json_reply(reply))
    result = await run_interview_agent(make_llm(fake), PROFILE)
    out = result.output
    assert [q.id for q in out.beginner] == ["beginner-1", "beginner-2", "beginner-3"]
    assert len(out.intermediate) == 3 and len(out.advanced) == 3
    assert result.prompt_id == INTERVIEW_PROMPT.id == "interview@1.1.0"
    assert fake.bodies[0]["model"] == "openai/gpt-oss-20b"
    assert sorted(schema_sent(fake)["required"]) == ["advanced", "beginner", "intermediate"]
    system, user = system_and_user(fake)
    assert "exactly 3 questions" in system
    assert "Acme Analytics" in user and "Acme Analytics" not in system


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested",
    [("beginner",), ("advanced",), ("beginner", "advanced"), ("intermediate", "advanced")],
)
async def test_interview_generates_only_requested_levels(requested) -> None:
    fake = FakeGroqChat().queue(json_reply({lvl: questions(lvl, 2) for lvl in requested}))
    result = await run_interview_agent(make_llm(fake), PROFILE, levels=requested, count_per_level=2)
    schema = schema_sent(fake)
    assert set(schema["required"]) == set(requested) == set(schema["properties"])
    for level in ("beginner", "intermediate", "advanced"):
        produced = getattr(result.output, level)
        assert len(produced) == (2 if level in requested else 0)


@pytest.mark.asyncio
async def test_interview_level_order_and_duplicates_normalised() -> None:
    fake = FakeGroqChat().queue(
        json_reply({"beginner": questions("beginner", 1), "advanced": questions("advanced", 1)})
    )
    await run_interview_agent(
        make_llm(fake), PROFILE, levels=["advanced", "beginner", "advanced"], count_per_level=1
    )
    assert list(schema_sent(fake)["properties"]) == ["beginner", "advanced"]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 5])
async def test_interview_count_boundaries_accepted(count: int) -> None:
    fake = FakeGroqChat().queue(json_reply({"beginner": questions("beginner", count)}))
    result = await run_interview_agent(
        make_llm(fake), PROFILE, levels=["beginner"], count_per_level=count
    )
    assert len(result.output.beginner) == count
    assert f"exactly {count} questions" in system_and_user(fake)[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 6, -1, 10, True, 2.5, "3"])
async def test_interview_count_out_of_range_rejected_without_llm(count) -> None:
    fake = FakeGroqChat()
    with pytest.raises(AgentInputError):
        await run_interview_agent(make_llm(fake), PROFILE, count_per_level=count)
    assert fake.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("levels", [[], ["expert"], ["beginner", "Beginner"], ["easy"]])
async def test_interview_invalid_levels_rejected_without_llm(levels) -> None:
    fake = FakeGroqChat()
    with pytest.raises(AgentInputError):
        await run_interview_agent(make_llm(fake), PROFILE, levels=levels)
    assert fake.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {"beginner": questions("beginner", 2)},  # wrong count (expected 3)
        {"beginner": questions("beginner", 4)},
        {
            "beginner": questions("beginner", 3),
            "advanced": questions("advanced", 3),
        },  # unrequested level
        {"beginner": [questions("beginner", 1)[0]] * 3},  # duplicates
        {"beginner": [{**q, "question": "Why?"} for q in questions("beginner", 3)]},  # too short
    ],
)
async def test_interview_invalid_output_triggers_repair(bad) -> None:
    good = {"beginner": questions("beginner", 3)}
    fake = FakeGroqChat().queue(json_reply(bad), json_reply(good))
    result = await run_interview_agent(make_llm(fake), PROFILE, levels=["beginner"])
    assert result.attempts == 2
    assert len(result.output.beginner) == 3


# --- skill-gap agent ----------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_gap_agent_success() -> None:
    report = {
        "matching_skills": ["Python", "SQL"],
        "missing_skills": ["Kubernetes", "Airflow"],
        "recommendations": ["Complete a hands-on Kubernetes project."],
    }
    fake = FakeGroqChat().queue(json_reply(report))
    result = await run_skill_gap_agent(make_llm(fake), PROFILE, JOB)
    assert result.output.missing_skills == ["Kubernetes", "Airflow"]
    assert result.prompt_id == SKILL_GAP_PROMPT.id
    system, user = system_and_user(fake)
    assert "Kubernetes" in user and "Kubernetes" not in system
    assert "<job_description>" in user and "<profile>" in user


@pytest.mark.asyncio
async def test_skill_gap_rejects_empty_job_description() -> None:
    fake = FakeGroqChat()
    with pytest.raises(AgentInputError):
        await run_skill_gap_agent(make_llm(fake), PROFILE, " \u202e ")
    assert fake.requests == []


# --- answer evaluator agent ---------------------------------------------------


@pytest.mark.asyncio
async def test_evaluator_success_computes_overall_rating() -> None:
    fake = FakeGroqChat().queue(json_reply(evaluation(4, 3, 3, 4, 4)))
    result = await run_answer_evaluator_agent(
        make_llm(fake),
        question="What is a Python generator?",
        answer="A function that yields values lazily.",
        level="beginner",
    )
    assert result.output.overall_rating == 3.6
    assert result.output.rubric.relevance.rating == 4
    assert result.prompt_id == EVALUATOR_PROMPT.id
    body = fake.bodies[0]
    assert body["model"] == "openai/gpt-oss-20b"
    wire = schema_sent(fake)
    assert "overall_rating" not in str(wire)
    assert wire["properties"]["rubric"]["required"] == list(RUBRIC_CRITERIA)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("level", "index"), [("beginner", 1), ("intermediate", 2), ("advanced", 3)]
)
async def test_evaluator_level_passed_as_integer_index(level: str, index: int) -> None:
    fake = FakeGroqChat().queue(json_reply(evaluation()))
    await run_answer_evaluator_agent(
        make_llm(fake), question="Explain joins?", answer="An answer.", level=level
    )
    system, user = system_and_user(fake)
    assert f"difficulty level is {index} (1 = beginner" in system
    assert "<question>" in user and "<answer>" in user
    assert "Explain joins?" not in system and "An answer." not in system


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "calls"),
    [
        ({"question": "Q?", "answer": "A.", "level": "expert"}, 0),
        ({"question": "", "answer": "A.", "level": "beginner"}, 0),
        ({"question": "Q?", "answer": "\u200b", "level": "beginner"}, 0),
    ],
)
async def test_evaluator_input_validation(kwargs, calls) -> None:
    fake = FakeGroqChat()
    with pytest.raises(AgentInputError):
        await run_answer_evaluator_agent(make_llm(fake), **kwargs)
    assert len(fake.requests) == calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {**evaluation(), "overall_rating": 5.0},  # model must not set it
        {
            **evaluation(),
            "rubric": {
                **evaluation()["rubric"],
                "creativity": {"rating": 5, "justification": "x y z"},
            },
        },
        evaluation(4, 3, 3, 4, 6),  # out of range
        evaluation(4, 3, 3, 4, 0),
        {**evaluation(), "strengths": ["s"] * 6},
        {**evaluation(), "model_answer_outline": ["o"] * 9},
    ],
)
async def test_evaluator_invalid_output_repaired(bad) -> None:
    fake = FakeGroqChat().queue(json_reply(bad), json_reply(evaluation(5, 5, 4, 4, 4)))
    result = await run_answer_evaluator_agent(
        make_llm(fake), question="Q here?", answer="Answer.", level="advanced"
    )
    assert result.attempts == 2
    assert result.output.overall_rating == 4.4


@pytest.mark.asyncio
async def test_evaluator_fractional_rating_repaired() -> None:
    bad = evaluation()
    bad["rubric"]["depth"]["rating"] = 3.5
    fake = FakeGroqChat().queue(json_reply(bad), json_reply(evaluation()))
    result = await run_answer_evaluator_agent(
        make_llm(fake), question="Q here?", answer="Answer.", level="beginner"
    )
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_evaluator_upstream_timeout_is_typed() -> None:
    fake = FakeGroqChat().queue(*(Reply(exception=httpx.ReadTimeout("slow")) for _ in range(3)))
    with pytest.raises(UpstreamTimeoutError):
        await run_answer_evaluator_agent(
            make_llm(fake), question="Q here?", answer="Answer.", level="beginner"
        )
    assert len(fake.requests) == 3  # 1 call + 2 SDK retries


# --- multilingual -------------------------------------------------------------

HINDI_Q = "पायथन में जनरेटर क्या होता है?"
MARATHI_A = "जनरेटर हे मूल्ये एकेक करून देणारे फंक्शन आहे; अर्\u200dय उदाहरण."


@pytest.mark.asyncio
async def test_multilingual_inputs_and_outputs_preserved() -> None:
    reply = evaluation()
    reply["strengths"] = ["संकल्पना सही समझाई", "मराठीत स्पष्ट उत्तर"]
    fake = FakeGroqChat().queue(json_reply(reply))
    result = await run_answer_evaluator_agent(
        make_llm(fake), question=HINDI_Q, answer=MARATHI_A, level="intermediate"
    )
    _, user = system_and_user(fake)
    assert HINDI_Q in user and MARATHI_A in user  # ZWJ preserved inside the prompt
    assert result.output.strengths == ["संकल्पना सही समझाई", "मराठीत स्पष्ट उत्तर"]


@pytest.mark.asyncio
async def test_multilingual_profile_and_job() -> None:
    profile = ResumeProfile.model_validate(
        {**PROFILE_PAYLOAD, "skills": ["पायथन", "डेटा विश्लेषण", "SQL"]}
    )
    reply = {
        "matching_skills": ["पायथन"],
        "missing_skills": ["कुबरनेटीस"],
        "recommendations": ["कुबरनेटीस शिका."],
    }
    fake = FakeGroqChat().queue(json_reply(reply))
    result = await run_skill_gap_agent(make_llm(fake), profile, "आवश्यक कौशल्य: पायथन, कुबरनेटीस")
    _, user = system_and_user(fake)
    assert "डेटा विश्लेषण" in user and "आवश्यक कौशल्य" in user
    assert result.output.recommendations == ["कुबरनेटीस शिका."]
