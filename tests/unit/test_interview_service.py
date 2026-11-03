"""M9 Phase 2: interview question service (stored profile, limits, deadline, no writes)."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from backend.api.routes.report import ReportNotFoundError as RouteReportNotFoundError
from backend.llm.errors import InvalidLLMOutputError, LLMError, LLMNotConfiguredError
from backend.schemas.interview_api import InterviewQuestionsRequest, InterviewQuestionsResponse
from backend.security.output_validation import LINK_PLACEHOLDER
from backend.services.errors import InterviewTimeoutError, ReportNotFoundError
from backend.services.interview import InterviewQuestionService
from tests.fixtures.groq import Reply, json_reply
from tests.fixtures.interview import (
    InterviewFakeGroq,
    clients_for,
    settings_for,
    snapshot,
    store_with_report,
)
from tests.fixtures.reports import PROFILE
from tests.unit.test_agents import questions

pytestmark = pytest.mark.asyncio

CANARY = "CANARYm9service29"


@pytest.fixture(autouse=True)
def _fast_sdk_backoff(monkeypatch) -> None:
    import groq._base_client as base

    monkeypatch.setattr(base, "INITIAL_RETRY_DELAY", 0.001)
    monkeypatch.setattr(base, "MAX_RETRY_DELAY", 0.002)


def _reply(levels: tuple[str, ...], count: int, **edit) -> Reply:
    payload = {level: questions(level, count) for level in levels}
    for level, items in edit.items():
        payload[level] = items
    return json_reply(payload)


async def _service(fake: InterviewFakeGroq, **settings):
    store, report = await store_with_report()
    config = settings_for(**settings)
    service = InterviewQuestionService(
        settings=config, llm_clients=clients_for(fake, config), report_store=store
    )
    return service, store, report


def _request(report_id, **fields) -> InterviewQuestionsRequest:
    return InterviewQuestionsRequest.model_validate({"report_id": str(report_id), **fields})


async def test_questions_come_from_the_stored_profile_for_the_requested_levels() -> None:
    fake = InterviewFakeGroq().script("interview", _reply(("beginner", "advanced"), 2))
    service, store, report = await _service(fake)
    response = await service.generate(
        _request(report.report_id, levels=["advanced", "beginner"], count_per_level=2)
    )
    assert isinstance(response, InterviewQuestionsResponse)
    assert response.levels == ["beginner", "advanced"] and response.count_per_level == 2
    assert [q.id for q in response.questions.beginner] == ["beginner-1", "beginner-2"]
    assert [q.id for q in response.questions.advanced] == ["advanced-1", "advanced-2"]
    assert response.questions.intermediate == []
    assert fake.routes == ["interview"]  # one interview call, no other agent
    (user,) = fake.user_messages("interview")
    sent = json.loads(user.split("<profile>", 1)[1].split("</profile>", 1)[0])
    assert (
        sent
        == json.loads(PROFILE.model_dump_json())
        == report.resume_profile.model_dump(mode="json")
    )
    (system,) = fake.system_messages("interview")
    assert "Produce exactly 2 questions for each difficulty level" in system
    schema = fake.bodies_for("interview")[0]["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"beginner", "advanced"}  # requested levels only


async def test_defaults_are_all_three_levels_with_three_questions() -> None:
    fake = InterviewFakeGroq().script(
        "interview", _reply(("beginner", "intermediate", "advanced"), 3)
    )
    service, _, report = await _service(fake)
    response = await service.generate(_request(report.report_id))
    assert response.levels == ["beginner", "intermediate", "advanced"]
    assert all(len(getattr(response.questions, lv)) == 3 for lv in response.levels)


async def test_generated_questions_are_never_stored_and_the_report_is_unchanged() -> None:
    fake = InterviewFakeGroq().script("interview", _reply(("intermediate",), 4))
    service, store, report = await _service(fake)
    before = snapshot(store)
    response = await service.generate(
        _request(report.report_id, levels=["intermediate"], count_per_level=4)
    )
    assert snapshot(store) == before  # same reports, same content
    stored = await store.get(str(report.report_id))
    assert stored.interview_questions == report.interview_questions
    assert stored.interview_questions != response.questions


async def test_output_sanitisation_and_link_neutralisation_still_apply() -> None:
    bad = questions("beginner", 1)
    bad[0]["question"] = f"Explain{chr(0x200B)} lists, see https://evil.example/x{chr(0x202E)} now?"
    fake = InterviewFakeGroq().script("interview", _reply(("beginner",), 1, beginner=bad))
    service, _, report = await _service(fake)
    response = await service.generate(
        _request(report.report_id, levels=["beginner"], count_per_level=1)
    )
    text = response.questions.beginner[0].question
    assert "evil.example" not in text and LINK_PLACEHOLDER in text
    assert chr(0x200B) not in text and chr(0x202E) not in text
    assert [(f.type, f.source) for f in response.security_flags] == [
        ("unexpected_url", "llm_output")
    ]


async def test_wrong_count_from_the_model_fails_closed() -> None:
    wrong = _reply(("beginner",), 2)  # asked for 1
    fake = InterviewFakeGroq().script("interview", wrong, wrong)
    service, _, report = await _service(fake)
    with pytest.raises(InvalidLLMOutputError) as raised:
        await service.generate(_request(report.report_id, levels=["beginner"], count_per_level=1))
    assert (raised.value.status_code, raised.value.code) == (502, "invalid_llm_output")


async def test_upstream_errors_keep_their_typed_mapping() -> None:
    fake = InterviewFakeGroq().script("interview", *[Reply(status=500)] * 3)
    service, _, report = await _service(fake)
    with pytest.raises(LLMError) as raised:
        await service.generate(_request(report.report_id))
    assert raised.value.status_code == 502


async def test_missing_api_key_is_503_before_any_call() -> None:
    fake = InterviewFakeGroq()
    store, report = await store_with_report()
    config = settings_for(groq_api_key=None)
    service = InterviewQuestionService(
        settings=config, llm_clients=clients_for(fake, config), report_store=store
    )
    with pytest.raises(LLMNotConfiguredError):
        await service.generate(_request(report.report_id))
    assert fake.routes == []


@pytest.mark.parametrize(
    "report_id",
    [str(uuid.uuid4()), "not-a-uuid", "", " x ", str(uuid.uuid1())],
)
async def test_unknown_and_malformed_reports_are_404_without_any_call(report_id: str) -> None:
    fake = InterviewFakeGroq()
    service, _, _ = await _service(fake)
    with pytest.raises(ReportNotFoundError) as raised:
        await service.generate(_request(report_id))
    assert (raised.value.status_code, raised.value.code) == (404, "report_not_found")
    assert fake.routes == []


async def test_expired_reports_are_404() -> None:
    fake = InterviewFakeGroq()
    service, store, report = await _service(fake)
    store._clock = lambda: datetime.now(UTC) + timedelta(seconds=3601)
    with pytest.raises(ReportNotFoundError):
        await service.generate(_request(report.report_id))
    assert fake.routes == [] and store._reports == {}


async def test_not_found_contract_matches_the_report_route() -> None:
    ours, route = ReportNotFoundError(), RouteReportNotFoundError()
    assert (ours.status_code, ours.code, ours.message) == (
        route.status_code,
        route.code,
        route.message,
    )


async def test_deadline_cancels_the_blocked_request_and_raises_typed_504() -> None:
    fake = InterviewFakeGroq(block=("interview",))
    service, store, report = await _service(fake, interview_deadline_seconds=1.0)
    before_tasks = asyncio.all_tasks()
    before = snapshot(store)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(InterviewTimeoutError) as raised:
        # Bounded here too, so a missing deadline fails the test instead of hanging it.
        await asyncio.wait_for(service.generate(_request(report.report_id)), timeout=10)
    elapsed = loop.time() - started
    assert (raised.value.status_code, raised.value.code) == (504, "interview_timeout")
    assert 0.95 <= elapsed < 5  # asyncio may fire timers up to one clock tick early
    assert fake.started == ["interview"] and fake.cancelled == ["interview"]
    for _ in range(3):
        await asyncio.sleep(0)
    assert asyncio.all_tasks() <= before_tasks  # nothing left running
    assert snapshot(store) == before


async def test_logs_contain_no_profile_question_or_report_content(captured_logs) -> None:
    logging.getLogger("backend").setLevel(logging.INFO)
    canary_profile = PROFILE.model_copy(update={"skills": [f"{CANARY}-skill"]})
    store, report = await store_with_report(resume_profile=canary_profile)
    leaked = questions("beginner", 1)
    leaked[0]["question"] = f"What is {CANARY} in practice?"
    fake = InterviewFakeGroq().script("interview", _reply(("beginner",), 1, beginner=leaked))
    config = settings_for()
    service = InterviewQuestionService(
        settings=config, llm_clients=clients_for(fake, config), report_store=store
    )
    await service.generate(_request(report.report_id, levels=["beginner"], count_per_level=1))
    assert CANARY not in captured_logs.text
    assert str(report.report_id) not in captured_logs.text
    (done,) = captured_logs.events("interview_questions_generated")
    assert done["node"] == "interview" and done["attempts"] == 1
