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


# --- M10 rate limiting -----------------------------------------------------------------

RATE_LIMIT_DEFAULTS = {
    "rate_limit_analyze": "5/minute;30/hour",
    "rate_limit_analyze_global": "20/minute",
    "rate_limit_interview_questions": "10/minute;60/hour",
    "rate_limit_interview_evaluate": "20/minute;200/hour",
    "rate_limit_interview_global": "60/minute",
    "rate_limit_match": "30/minute",
    "rate_limit_parse": "30/minute",
    "rate_limit_report": "60/minute",
    "rate_limit_report_pdf": "20/minute",
}


def test_rate_limit_defaults_match_approved_decisions() -> None:
    # Real defaults, not make_settings (which turns rate limiting off for app tests).
    settings = Settings(_env_file=None)
    assert settings.rate_limit_enabled is True
    for name, value in RATE_LIMIT_DEFAULTS.items():
        assert getattr(settings, name) == value
    assert settings.rate_limit_max_tracked_keys == 10_000
    assert settings.analyze_max_concurrent == 2


def test_rate_limits_load_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("RATE_LIMIT_ANALYZE", " 2/minute; 10/hour ")
    monkeypatch.setenv("RATE_LIMIT_MAX_TRACKED_KEYS", "500")
    monkeypatch.setenv("ANALYZE_MAX_CONCURRENT", "1")
    settings = Settings(_env_file=None)
    assert settings.rate_limit_enabled is False
    assert settings.rate_limit_analyze == "2/minute; 10/hour"
    assert settings.rate_limit_max_tracked_keys == 500
    assert settings.analyze_max_concurrent == 1


@pytest.mark.parametrize("name", sorted(RATE_LIMIT_DEFAULTS))
@pytest.mark.parametrize(
    "value", ["", "often", "0/minute", "501/minute", "1/minute;2/minute", "1/2 days"]
)
def test_invalid_rate_limits_rejected(make_settings, name: str, value: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{name: value})


@pytest.mark.parametrize("value", [99, 25_001, 1_000_000, 0])
def test_rate_limit_capacity_bounds(make_settings, value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(rate_limit_max_tracked_keys=value)


@pytest.mark.parametrize("value", [0, 9])
def test_analyze_concurrency_bounds(make_settings, value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(analyze_max_concurrent=value)


@pytest.mark.parametrize("value", [100, 25_000])
def test_rate_limit_capacity_bounds_are_inclusive(make_settings, value: int) -> None:
    assert make_settings(rate_limit_max_tracked_keys=value).rate_limit_max_tracked_keys == value


def test_largest_rate_limit_amount_is_accepted(make_settings) -> None:
    assert make_settings(rate_limit_report="500/minute").rate_limit_report == "500/minute"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("501/minute", "each amount must be between 1 and 500"),
        ("often", "must use the limits notation"),
        ("1/minute;2/minute", "must not repeat a window length"),
    ],
)
def test_rate_limit_errors_use_fixed_messages(make_settings, value: str, message: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        make_settings(rate_limit_match=value)
    (error,) = excinfo.value.errors()
    assert error["loc"] == ("rate_limit_match",)
    assert error["msg"].startswith(f"Value error, {message}")
    # Pydantic reports the rejected value separately; the message itself never
    # repeats it, and limit strings carry no secrets.
    assert value not in error["msg"]


def test_app_test_settings_disable_rate_limiting_but_production_enables_it(make_settings) -> None:
    assert make_settings().rate_limit_enabled is False
    assert make_settings(rate_limit_enabled=True).rate_limit_enabled is True
    assert Settings(_env_file=None).rate_limit_enabled is True
