"""Structured LLM calls through the real ChatGroq and groq SDK, with a fake HTTP API."""

from __future__ import annotations

import httpx
import pytest

from backend.core.config import Settings
from backend.health.upstream import GroqUpstreamMonitor
from backend.llm.client import LLMClients
from backend.llm.errors import (
    InvalidLLMOutputError,
    LLMError,
    LLMNotConfiguredError,
    UpstreamAuthError,
    UpstreamRateLimitedError,
    UpstreamTimeoutError,
)
from backend.llm.prompts import build_messages
from backend.llm.prompts.resume import RESUME_PROMPT
from backend.llm.structured import REPAIR_INSTRUCTION, StructuredLLM, wire_schema
from backend.schemas.resume import ResumeProfile
from tests.conftest import FAKE_API_KEY
from tests.fixtures.groq import FakeGroqChat, Reply, json_reply

RESUME = "Jordan Sample\nSkills: Python, SQL"
VALID = {
    "skills": ["Python", "SQL"],
    "education": [],
    "experience": [],
    "projects": [],
    "certifications": [],
}
MESSAGES = build_messages(RESUME_PROMPT, {"resume": RESUME})


@pytest.fixture(autouse=True)
def _fast_sdk_backoff(monkeypatch) -> None:
    """The SDK's exponential backoff (0.5 s base) would slow tests; behaviour is unchanged."""
    import groq._base_client as base

    monkeypatch.setattr(base, "INITIAL_RETRY_DELAY", 0.001)
    monkeypatch.setattr(base, "MAX_RETRY_DELAY", 0.002)


def _monitor() -> GroqUpstreamMonitor:
    return GroqUpstreamMonitor(
        api_key=None,
        base_url="https://x",
        required_models=(),
        interval_seconds=60,
        timeout_seconds=1,
    )


def _llm(fake: FakeGroqChat, *, monitor=None, **settings) -> StructuredLLM:
    values = {"groq_api_key": FAKE_API_KEY, "llm_max_retries": 2, **settings}
    clients = LLMClients(
        Settings(_env_file=None, **values), http_client=fake.client(), monitor=monitor
    )
    return StructuredLLM(clients)


async def _generate(llm: StructuredLLM):
    return await llm.generate("resume", ResumeProfile, MESSAGES, source_texts=[RESUME])


@pytest.mark.asyncio
async def test_success_single_call() -> None:
    fake = FakeGroqChat().queue(json_reply(VALID))
    monitor = _monitor()
    result = await _generate(_llm(fake, monitor=monitor))
    assert result.output.skills == ["Python", "SQL"]
    assert (result.attempts, result.usage.input_tokens, result.usage.output_tokens) == (1, 120, 40)
    assert len(fake.requests) == 1
    assert monitor.snapshot().status == "available"
    assert monitor.snapshot().source == "request"


@pytest.mark.asyncio
async def test_request_contract() -> None:
    fake = FakeGroqChat().queue(json_reply(VALID))
    await _generate(_llm(fake, llm_timeout_seconds=12.5, llm_max_output_tokens=2048))
    body, request = fake.bodies[0], fake.requests[0]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["response_format"]["json_schema"]["schema"] == wire_schema(ResumeProfile)
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert RESUME not in body["messages"][0]["content"]
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 2048
    assert "tools" not in body and "tool_choice" not in body
    assert request.extensions["timeout"]["read"] == 12.5
    assert request.headers["authorization"] == f"Bearer {FAKE_API_KEY}"


@pytest.mark.asyncio
async def test_reasoning_effort_only_for_gpt_oss() -> None:
    fake = FakeGroqChat().queue(json_reply(VALID))
    await _generate(_llm(fake, groq_model_resume="qwen/qwen3.8-27b"))
    assert "reasoning_effort" not in fake.bodies[0] or fake.bodies[0]["reasoning_effort"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        Reply("not json at all"),
        Reply('{"skills": ["Python"]'),  # truncated
        json_reply({**VALID, "unexpected": True}),  # extra key
        json_reply({**VALID, "skills": ["x" * 300]}),  # oversize item
        json_reply({**VALID, "skills": "Python"}),  # wrong type
        json_reply(["not", "an", "object"]),
        Reply(status=400, error_code="json_validate_failed"),
    ],
)
async def test_one_repair_then_success(bad: Reply) -> None:
    fake = FakeGroqChat().queue(bad, json_reply(VALID))
    result = await _generate(_llm(fake))
    assert result.attempts == 2
    assert len(fake.requests) == 2
    repair_messages = fake.bodies[1]["messages"]
    assert repair_messages[-1] == {"role": "user", "content": REPAIR_INSTRUCTION}
    assert [m["role"] for m in repair_messages[:2]] == ["system", "user"]


@pytest.mark.asyncio
async def test_two_invalid_outputs_raise_typed_error() -> None:
    fake = FakeGroqChat().queue(Reply("garbage"), json_reply({**VALID, "skills": 7}))
    with pytest.raises(InvalidLLMOutputError) as info:
        await _generate(_llm(fake))
    assert (info.value.status_code, info.value.code) == (502, "invalid_llm_output")
    assert len(fake.requests) == 2  # exactly one repair, no more


@pytest.mark.asyncio
async def test_transient_5xx_retried_by_sdk() -> None:
    fake = FakeGroqChat().queue(Reply(status=503), Reply(status=500), json_reply(VALID))
    result = await _generate(_llm(fake, llm_max_retries=2))
    assert result.attempts == 1 and len(fake.requests) == 3


@pytest.mark.asyncio
async def test_retries_exhausted_raise_upstream_error() -> None:
    fake = FakeGroqChat().queue(*(Reply(status=500) for _ in range(3)))
    monitor = _monitor()
    with pytest.raises(LLMError) as info:
        await _generate(_llm(fake, llm_max_retries=2, monitor=monitor))
    assert (info.value.status_code, info.value.code) == (502, "upstream_error")
    assert len(fake.requests) == 3
    assert monitor.snapshot().last_error_code == "upstream_error"


@pytest.mark.asyncio
async def test_no_retries_when_disabled() -> None:
    fake = FakeGroqChat().queue(Reply(status=500))
    with pytest.raises(LLMError):
        await _generate(_llm(fake, llm_max_retries=0))
    assert len(fake.requests) == 1


@pytest.mark.asyncio
async def test_timeout_retried_then_typed_error() -> None:
    fake = FakeGroqChat().queue(*(Reply(exception=httpx.ReadTimeout("slow")) for _ in range(3)))
    monitor = _monitor()
    with pytest.raises(UpstreamTimeoutError) as info:
        await _generate(_llm(fake, llm_max_retries=2, monitor=monitor))
    assert (info.value.status_code, info.value.code) == (504, "upstream_timeout")
    assert len(fake.requests) == 3
    assert monitor.snapshot().status == "unavailable"
    assert monitor.snapshot().last_error_code == "timeout"


@pytest.mark.asyncio
async def test_timeout_then_success() -> None:
    fake = FakeGroqChat().queue(Reply(exception=httpx.ConnectTimeout("slow")), json_reply(VALID))
    result = await _generate(_llm(fake))
    assert result.output.skills == ["Python", "SQL"]


@pytest.mark.asyncio
async def test_connection_error_mapped() -> None:
    fake = FakeGroqChat().queue(*(Reply(exception=httpx.ConnectError("refused")) for _ in range(3)))
    with pytest.raises(LLMError) as info:
        await _generate(_llm(fake))
    assert info.value.code == "upstream_error"


@pytest.mark.asyncio
async def test_rate_limit_retried_then_mapped_with_retry_after() -> None:
    limited = Reply(status=429, headers={"retry-after": "7"})
    fake = FakeGroqChat().queue(limited, limited, limited)
    monitor = _monitor()
    with pytest.raises(UpstreamRateLimitedError) as info:
        await _generate(_llm(fake, monitor=monitor))
    assert (info.value.status_code, info.value.code) == (503, "upstream_rate_limited")
    assert info.value.headers == {"Retry-After": "7"}
    assert len(fake.requests) == 3
    assert monitor.snapshot().last_error_code == "rate_limited"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failure_not_retried(status: int) -> None:
    fake = FakeGroqChat().queue(Reply(status=status))
    with pytest.raises(UpstreamAuthError) as info:
        await _generate(_llm(fake))
    assert info.value.code == "upstream_auth_failed"
    assert len(fake.requests) == 1


@pytest.mark.asyncio
async def test_other_bad_request_not_repaired() -> None:
    fake = FakeGroqChat().queue(Reply(status=400, error_code="model_decommissioned"))
    with pytest.raises(LLMError) as info:
        await _generate(_llm(fake))
    assert info.value.code == "upstream_error"
    assert len(fake.requests) == 1


@pytest.mark.asyncio
async def test_output_validation_applied_to_result() -> None:
    dirty = {
        **VALID,
        "projects": ["See https://evil.example/x for details"],
        "skills": ["Py\u202ethon"],
    }
    fake = FakeGroqChat().queue(json_reply(dirty))
    result = await _generate(_llm(fake))
    assert result.output.skills == ["Python"]
    assert "evil.example" not in result.output.projects[0]
    assert [f.type for f in result.flags] == ["unexpected_url"]


@pytest.mark.asyncio
async def test_multilingual_output_round_trips() -> None:
    hindi = {**VALID, "skills": ["पायथन", "डेटा विश्लेषण"], "projects": ["महाराष्ट्रातील प्रकल्प"]}
    fake = FakeGroqChat().queue(json_reply(hindi))
    result = await _generate(_llm(fake))
    assert result.output.skills == ["पायथन", "डेटा विश्लेषण"]
    assert result.output.projects == ["महाराष्ट्रातील प्रकल्प"]


@pytest.mark.asyncio
async def test_not_configured_raises_503_without_network() -> None:
    fake = FakeGroqChat()
    clients = LLMClients(Settings(_env_file=None), http_client=fake.client())
    assert clients.configured is False
    with pytest.raises(LLMNotConfiguredError) as info:
        await StructuredLLM(clients).generate("resume", ResumeProfile, MESSAGES, source_texts=[])
    assert info.value.status_code == 503
    assert fake.requests == []


@pytest.mark.asyncio
async def test_clients_are_cached_and_closed() -> None:
    owned = LLMClients(Settings(_env_file=None, groq_api_key=FAKE_API_KEY))
    assert owned.chat_model("resume") is owned.chat_model("resume")
    assert owned.chat_model("interview").model_name == "openai/gpt-oss-20b"
    http = owned._http_client
    assert http is not None and not http.is_closed
    await owned.aclose()
    assert http.is_closed

    shared = httpx.AsyncClient()
    injected = LLMClients(Settings(_env_file=None, groq_api_key=FAKE_API_KEY), http_client=shared)
    injected.chat_model("resume")
    await injected.aclose()
    assert not shared.is_closed  # injected clients belong to the caller
    await shared.aclose()
