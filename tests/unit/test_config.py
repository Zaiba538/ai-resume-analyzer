"""Settings validation and defaults (decisions D4, D6)."""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from backend.core.config import MAX_REPORT_TTL_SECONDS, Settings
from tests.conftest import FAKE_API_KEY


def test_defaults_match_approved_decisions(make_settings) -> None:
    settings = make_settings()
    assert settings.report_ttl_seconds == 3600
    assert settings.max_revisions == 1
    assert settings.groq_model_resume == "openai/gpt-oss-120b"
    assert settings.groq_model_skill_gap == "openai/gpt-oss-120b"
    assert settings.groq_model_critic == "openai/gpt-oss-120b"
    assert settings.groq_model_interview == "openai/gpt-oss-20b"
    assert settings.groq_model_evaluator == "openai/gpt-oss-20b"
    assert settings.groq_model_prompt_guard == "meta-llama/llama-prompt-guard-2-86m"
    assert settings.api_docs_enabled is False
    assert settings.groq_api_key is None
    assert settings.groq_api_key_configured is False


def test_values_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("REPORT_TTL_SECONDS", "1800")
    monkeypatch.setenv("MAX_REVISIONS", "0")
    monkeypatch.setenv("GROQ_MODEL_INTERVIEW", "qwen/qwen3.8-27b")
    monkeypatch.setenv("GROQ_API_KEY", FAKE_API_KEY)
    settings = Settings(_env_file=None)
    assert settings.report_ttl_seconds == 1800
    assert settings.max_revisions == 0
    assert settings.groq_model_interview == "qwen/qwen3.8-27b"
    assert settings.groq_api_key_configured is True


@pytest.mark.parametrize("ttl", [MAX_REPORT_TTL_SECONDS + 1, 59, 0, -1])
def test_report_ttl_bounded_to_one_hour(make_settings, ttl: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(report_ttl_seconds=ttl)


@pytest.mark.parametrize("value", [-1, 4])
def test_max_revisions_bounds(make_settings, value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(max_revisions=value)


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_api_key_means_not_configured(make_settings, blank: str) -> None:
    settings = make_settings(groq_api_key=blank)
    assert settings.groq_api_key is None
    assert settings.groq_api_key_configured is False


def test_api_key_is_stripped_and_never_rendered(make_settings) -> None:
    settings = make_settings(groq_api_key=f"  {FAKE_API_KEY}\n")
    assert isinstance(settings.groq_api_key, SecretStr)
    assert settings.groq_api_key.get_secret_value() == FAKE_API_KEY
    for rendering in (repr(settings), str(settings), settings.model_dump_json()):
        assert FAKE_API_KEY not in rendering


@pytest.mark.parametrize(
    "url", ["http://api.groq.com/openai/v1", "ftp://x", "https://api groq.com", ""]
)
def test_base_url_must_be_https(make_settings, url: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(groq_api_base_url=url)


def test_base_url_trailing_slash_removed(make_settings) -> None:
    settings = make_settings(groq_api_base_url="https://api.groq.com/openai/v1/")
    assert settings.groq_api_base_url == "https://api.groq.com/openai/v1"


@pytest.mark.parametrize(
    "value", ["", "'self'; script-src *", "'self'\r\nX-Injected: 1", "'self', https://x"]
)
def test_frame_ancestors_rejects_injection(make_settings, value: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(csp_frame_ancestors=value)


def test_blank_model_name_rejected(make_settings) -> None:
    with pytest.raises(ValidationError):
        make_settings(groq_model_resume="  ")


def test_groq_models_are_distinct_in_role_order(make_settings) -> None:
    assert make_settings().groq_models == (
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "meta-llama/llama-prompt-guard-2-86m",
    )


def test_settings_are_immutable(make_settings) -> None:
    settings = make_settings()
    with pytest.raises(ValidationError):
        settings.report_ttl_seconds = 10  # type: ignore[misc]
