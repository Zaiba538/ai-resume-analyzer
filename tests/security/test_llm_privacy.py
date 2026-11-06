"""LLM privacy: keys, prompts, outputs and failed generations never reach logs or errors."""

from __future__ import annotations

import logging

import pytest

from backend.core.config import Settings
from backend.llm.client import LLMClients
from backend.llm.errors import LLMError
from backend.llm.prompts import build_messages
from backend.llm.prompts.skill_gap import SKILL_GAP_PROMPT
from backend.llm.structured import StructuredLLM
from backend.schemas.skill_gap import SkillGapReport
from tests.conftest import FAKE_API_KEY
from tests.fixtures.groq import FakeGroqChat, Reply, json_reply

RESUME_CANARY = "CANARY-resume-9d2e"
JD_CANARY = "CANARY-jd-41f0"
OUTPUT_CANARY = "CANARY-output-77aa"


@pytest.fixture(autouse=True)
def _fast_sdk_backoff(monkeypatch) -> None:
    import groq._base_client as base

    monkeypatch.setattr(base, "INITIAL_RETRY_DELAY", 0.001)
    monkeypatch.setattr(base, "MAX_RETRY_DELAY", 0.002)


@pytest.fixture
def backend_info_logs():
    logging.getLogger("backend").setLevel(logging.INFO)
    yield


def _llm(fake: FakeGroqChat) -> StructuredLLM:
    settings = Settings(_env_file=None, groq_api_key=FAKE_API_KEY)
    return StructuredLLM(LLMClients(settings, http_client=fake.client()))


def _messages():
    return build_messages(
        SKILL_GAP_PROMPT,
        {"profile": f'{{"skills": ["{RESUME_CANARY}"]}}', "job_description": f"Needs {JD_CANARY}"},
    )


async def _run(fake: FakeGroqChat):
    return await _llm(fake).generate(
        "skill_gap", SkillGapReport, _messages(), source_texts=[RESUME_CANARY, JD_CANARY]
    )


@pytest.mark.asyncio
async def test_successful_call_logs_metadata_only(captured_logs, backend_info_logs) -> None:
    output = {"matching_skills": [OUTPUT_CANARY], "missing_skills": [], "recommendations": []}
    await _run(FakeGroqChat().queue(json_reply(output)))
    (record,) = captured_logs.events("llm_call_completed")
    assert record["node"] == "skill_gap" and record["model"] == "openai/gpt-oss-120b"
    assert (record["input_tokens"], record["output_tokens"], record["attempts"]) == (120, 40, 1)
    for secret in (FAKE_API_KEY, RESUME_CANARY, JD_CANARY, OUTPUT_CANARY):
        assert secret not in captured_logs.text


@pytest.mark.asyncio
async def test_failed_generation_never_logged_or_returned(captured_logs, backend_info_logs) -> None:
    failed = Reply(status=400, error_code="json_validate_failed")
    fake = FakeGroqChat().queue(failed, failed)
    with pytest.raises(LLMError) as info:
        await _run(fake)
    text = captured_logs.text
    assert "SECRET-FAILED-GENERATION" not in text
    assert "SECRET-FAILED-GENERATION" not in info.value.message
    assert "scripted failure" not in info.value.message
    assert FAKE_API_KEY not in text


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429, 500])
async def test_upstream_errors_never_expose_key_or_body(captured_logs, status: int) -> None:
    fake = FakeGroqChat().queue(*(Reply(status=status) for _ in range(3)))
    with pytest.raises(LLMError) as info:
        await _run(fake)
    for leaked in (FAKE_API_KEY, "scripted failure", RESUME_CANARY, JD_CANARY):
        assert leaked not in info.value.message
        assert leaked not in captured_logs.text
    assert info.value.__cause__ is None  # upstream exception (with body) is not chained


def test_api_key_not_rendered_by_clients() -> None:
    clients = LLMClients(Settings(_env_file=None, groq_api_key=FAKE_API_KEY))
    model = clients.chat_model("resume")
    for rendering in (repr(model), str(model), repr(clients.settings)):
        assert FAKE_API_KEY not in rendering


def test_tracing_disabled_by_app_factory(start_app) -> None:
    import os

    start_app()
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
