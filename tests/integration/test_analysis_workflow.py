"""M7 phase 4 end to end: real agents, Critic, structured output and Prompt Guard,
through the real SDK against a routed fake Groq API (no live calls)."""

from __future__ import annotations

import json
import logging

import pytest

from backend.core.config import Settings
from backend.graph.analysis import AnalysisDeps, AnalysisWorkflow, build_prompt_guard
from backend.llm.client import LLMClients
from backend.llm.errors import LLMError
from backend.llm.prompts import REVISION_GUIDANCE
from backend.scoring.embeddings import EmbeddingService
from backend.storage.report_store import ReportStore
from tests.conftest import FAKE_API_KEY
from tests.fixtures.embeddings import FakeEmbedder
from tests.fixtures.groq import Reply, RoutedFakeGroq, json_reply
from tests.unit.test_agents import questions
from tests.unit.test_critic_checks import JOB, RESUME, gap, profile

pytestmark = pytest.mark.asyncio

PROFILE_REPLY = json_reply(profile().model_dump())
INTERVIEW_REPLY = json_reply(
    {lvl: questions(lvl, 3) for lvl in ("beginner", "intermediate", "advanced")}
)
GAP_REPLY = json_reply(
    gap(["Python", "Kubernetes"], ["Apache Airflow"], ["Build an Airflow DAG."]).model_dump()
)
NO_ISSUES = json_reply({"issues": []})
CANARY = "CANARYworkflow3a7e"


@pytest.fixture(autouse=True)
def _fast_sdk_backoff(monkeypatch) -> None:
    import groq._base_client as base

    monkeypatch.setattr(base, "INITIAL_RETRY_DELAY", 0.001)
    monkeypatch.setattr(base, "MAX_RETRY_DELAY", 0.002)


def critic_issue(
    agent: str,
    message: str,
    severity: str = "medium",
    category: str = "irrelevant_question",
    field: str = "beginner[0]",
) -> dict:
    return {
        "agent": agent,
        "field": field,
        "severity": severity,
        "category": category,
        "message": message,
    }


async def workflow_for(fake: RoutedFakeGroq, **overrides):
    settings = Settings(_env_file=None, groq_api_key=FAKE_API_KEY, **overrides)
    clients = LLMClients(settings, http_client=fake.client())
    embeddings = EmbeddingService(FakeEmbedder)
    await embeddings.load()
    store = ReportStore(ttl_seconds=3600, max_reports=10, sweep_interval_seconds=60)
    deps = AnalysisDeps(
        settings=settings,
        llm_clients=clients,
        embeddings=embeddings,
        report_store=store,
        prompt_guard=build_prompt_guard(settings, clients),
    )
    return AnalysisWorkflow(deps), store


def scripted(
    *critic_replies: Reply,
    resume=(PROFILE_REPLY,),
    interview=(INTERVIEW_REPLY,),
    skill_gap=(GAP_REPLY,),
    guard_score: str = "0.01",
) -> RoutedFakeGroq:
    fake = RoutedFakeGroq(guard_score=guard_score)
    fake.script("resume", *resume).script("interview", *interview)
    fake.script("skill_gap", *skill_gap).script("critic", *critic_replies)
    return fake


def user_message(body: dict) -> str:
    return body["messages"][1]["content"]


async def test_approve_end_to_end() -> None:
    fake = scripted(NO_ISSUES)
    workflow, store = await workflow_for(fake)
    report = await workflow.run(resume_text=RESUME, job_description=JOB)
    assert fake.routes.count("prompt_guard") == 2  # resume and job description, flag-only
    assert sorted(r for r in fake.routes if r != "prompt_guard") == [
        "critic",
        "interview",
        "resume",
        "skill_gap",
    ]
    assert fake.routes.index("resume") < fake.routes.index("critic")
    assert (report.critic.verdict, report.critic.revisions) == ("approve", 0)
    assert report.meta.prompt_versions == {
        "resume": "resume@1.0.0",
        "interview": "interview@1.1.0",
        "skill_gap": "skill_gap@1.2.0",
        "critic": "critic@1.4.0",
    }
    assert report.match.ats_score.status == "not_calibrated"
    assert await store.get(str(report.report_id)) == report


async def test_revision_reruns_only_the_targeted_agent_with_its_feedback() -> None:
    first = json_reply(
        {"issues": [critic_issue("interview", "Question is not about the profile.")]}
    )
    fake = scripted(first, NO_ISSUES, interview=(INTERVIEW_REPLY, INTERVIEW_REPLY))
    workflow, _ = await workflow_for(fake)
    report = await workflow.run(resume_text=RESUME, job_description=JOB)
    assert [fake.routes.count(r) for r in ("resume", "interview", "skill_gap", "critic")] == [
        1,
        2,
        1,
        2,
    ]
    revised = fake.bodies_for("interview")[1]
    assert REVISION_GUIDANCE in revised["messages"][0]["content"]
    block = (
        user_message(revised).split("<critic_feedback>\n", 1)[1].split("\n</critic_feedback>")[0]
    )
    assert json.loads(block) == [
        {
            "field": "beginner[0]",
            "severity": "medium",
            "message": "Question is not about the profile.",
        }
    ]
    assert "<critic_feedback>" not in user_message(fake.bodies_for("interview")[0])
    assert (report.critic.verdict, report.critic.revisions) == ("approve", 1)
    assert report.meta.prompt_versions["interview"] == "interview@1.2.0"


async def test_revised_resume_reruns_dependents_without_foreign_feedback() -> None:
    first = json_reply(
        {
            "issues": [
                critic_issue(
                    "resume",
                    "Project wording is not in the resume.",
                    category="unsupported_claim",
                    field="projects[0]",
                )
            ]
        }
    )
    fake = scripted(
        first,
        NO_ISSUES,
        resume=(PROFILE_REPLY,) * 2,
        interview=(INTERVIEW_REPLY,) * 2,
        skill_gap=(GAP_REPLY,) * 2,
    )
    workflow, _ = await workflow_for(fake)
    report = await workflow.run(resume_text=RESUME, job_description=JOB)
    assert [fake.routes.count(r) for r in ("resume", "interview", "skill_gap", "critic")] == [
        2,
        2,
        2,
        2,
    ]
    assert "<critic_feedback>" in user_message(fake.bodies_for("resume")[1])
    for route in ("interview", "skill_gap"):
        second = user_message(fake.bodies_for(route)[1])
        assert "<critic_feedback>" not in second and "Project wording" not in second
    assert report.meta.prompt_versions["resume"] == "resume@1.1.0"
    assert report.meta.prompt_versions["interview"] == "interview@1.1.0"


async def test_unresolved_revision_is_flagged_with_its_issues() -> None:
    issue = critic_issue("interview", "Still unrelated to the profile.")
    fake = scripted(
        json_reply({"issues": [issue]}),
        json_reply({"issues": [issue]}),
        interview=(INTERVIEW_REPLY,) * 2,
    )
    workflow, store = await workflow_for(fake)
    report = await workflow.run(resume_text=RESUME, job_description=JOB)
    assert (report.critic.verdict, report.critic.revisions) == ("flagged", 1)
    assert [i.message for i in report.critic.issues] == ["Still unrelated to the profile."]
    assert fake.routes.count("critic") == 2
    assert await store.get(str(report.report_id)) == report


async def test_data_scientist_paraphrase_evidence_is_not_flagged_through_the_cap() -> None:
    """M11 Phase 2 (H-1): the manually tested Data Scientist case, end to end.

    The skill-gap agent correctly lists data preprocessing and machine learning as
    matching on described work only. A real interview issue forces one revision
    (max_revisions=1), so any deterministic contradiction on that evidence would
    survive the cap and flag the report.
    """
    from tests.unit.test_skill_evidence import DS_JOB, DS_RESUME, ds_profile

    ds_gap = json_reply(
        gap(
            ["Python", "SQL", "data preprocessing", "machine learning"],
            [],
            ["Build a Tableau dashboard from a public dataset."],
        ).model_dump()
    )
    interview_issue = critic_issue("interview", "Question is not about the profile.")
    first = json_reply({"issues": [interview_issue]})
    fake = scripted(
        first,
        NO_ISSUES,
        resume=(json_reply(ds_profile().model_dump()),),
        interview=(INTERVIEW_REPLY,) * 2,
        skill_gap=(ds_gap,) * 2,
    )
    workflow, _ = await workflow_for(fake)
    report = await workflow.run(resume_text=DS_RESUME, job_description=DS_JOB)
    assert (report.critic.verdict, report.critic.revisions) == ("approve", 1)
    assert not [i for i in report.critic.issues if i.agent == "skill_gap"]
    assert fake.routes.count("skill_gap") == 1  # never targeted for revision
    assert fake.routes.count("critic") == 2
    claims = json.loads(
        user_message(fake.bodies_for("critic")[0])
        .split("<agent_outputs>\n", 1)[1]
        .split("\n</agent_outputs>", 1)[0]
    )["claims_to_verify"]
    for n in (2, 3):
        assert {
            "agent": "skill_gap",
            "field": f"matching_skills[{n}]",
            "expectation": "resume_supports",
        } in claims
    assert report.match.ats_score.status == "not_calibrated"


async def test_prompt_guard_high_score_flags_but_never_rejects() -> None:
    fake = scripted(NO_ISSUES, guard_score="0.99")
    workflow, _ = await workflow_for(fake)
    report = await workflow.run(resume_text=RESUME, job_description=JOB)
    classifier = {(f.source, f.score) for f in report.security_flags if f.type == "classifier"}
    assert classifier == {("resume", 0.99), ("job_description", 0.99)}
    assert report.critic.verdict == "approve"


async def test_upstream_failure_fails_closed() -> None:
    fake = scripted(*(Reply(status=500) for _ in range(3)))
    workflow, store = await workflow_for(fake)
    with pytest.raises(LLMError):
        await workflow.run(resume_text=RESUME, job_description=JOB)
    assert await store.count() == 0


async def test_raw_text_reaches_agents_but_is_never_stored_or_logged(captured_logs) -> None:
    logging.getLogger("backend").setLevel(logging.INFO)
    fake = scripted(NO_ISSUES)
    workflow, store = await workflow_for(fake)
    report = await workflow.run(
        resume_text=f"{RESUME}\nHobby: {CANARY}-resume", job_description=f"{JOB} {CANARY}-jd"
    )
    assert CANARY in user_message(fake.bodies_for("resume")[0])  # the agent saw the input
    critic_user = user_message(fake.bodies_for("critic")[0])
    for word in ("semantic_similarity", "skill_coverage", "tfidf", "ats_score"):
        assert word not in critic_user  # metrics never reach the Critic
    assert CANARY not in (await store.get(str(report.report_id))).model_dump_json()
    assert CANARY not in captured_logs.text and FAKE_API_KEY not in captured_logs.text


# --- English-only check (language validation) -------------------------------------------------


@pytest.mark.parametrize(
    ("resume", "job", "loc"),
    [
        ("hindi", None, "resume"),
        ("marathi", JOB, "resume"),
        (RESUME, "hindi", "job_description"),
        (RESUME, "marathi", "job_description"),
    ],
)
async def test_non_english_input_stops_the_workflow_before_any_call(resume, job, loc) -> None:
    from backend.parsing.language import UnsupportedLanguageError
    from tests.unit.test_language import HINDI_JD, HINDI_RESUME, MARATHI_JD, MARATHI_RESUME

    resume = {"hindi": HINDI_RESUME, "marathi": MARATHI_RESUME}.get(resume, resume)
    job = {"hindi": HINDI_JD, "marathi": MARATHI_JD}.get(job, job)
    fake = scripted(NO_ISSUES)
    workflow, store = await workflow_for(fake)
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        await workflow.run(resume_text=resume, job_description=job)
    assert excinfo.value.details[0].loc == [loc]
    assert fake.routes == []  # no agent, Critic or Prompt Guard call
    embedder = workflow._deps.embeddings.get()
    assert (embedder.encode_calls, embedder.chunk_calls) == (0, 0)
    assert await store.count() == 0
