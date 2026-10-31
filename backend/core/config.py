"""Application settings.

Values come from environment variables and, for local development, a ``.env``
file in the working directory (see ``.env.example``). Every field can be
overridden by the environment variable of the same name in upper case.

The Groq API key is held as ``SecretStr`` so that ``repr()``, logs and
serialisation never render it. Its validator never raises, because Pydantic
validation errors echo the rejected input value.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Decision D6: derived report data may stay in server memory for at most one hour.
MAX_REPORT_TTL_SECONDS = 3600

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_MODEL_FIELDS = (
    "groq_model_resume",
    "groq_model_skill_gap",
    "groq_model_critic",
    "groq_model_interview",
    "groq_model_evaluator",
    "groq_model_prompt_guard",
)


class Settings(BaseSettings):
    """Validated, immutable runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    log_level: LogLevel = "INFO"

    # Groq: credentials, models per agent role, and the upstream availability probe.
    groq_api_key: SecretStr | None = None
    groq_api_base_url: str = "https://api.groq.com/openai/v1"
    groq_model_resume: str = "openai/gpt-oss-120b"
    groq_model_skill_gap: str = "openai/gpt-oss-120b"
    groq_model_critic: str = "openai/gpt-oss-120b"
    groq_model_interview: str = "openai/gpt-oss-20b"
    groq_model_evaluator: str = "openai/gpt-oss-20b"
    groq_model_prompt_guard: str = "meta-llama/llama-prompt-guard-2-86m"
    groq_probe_interval_seconds: float = Field(default=300.0, ge=30.0, le=3600.0)
    groq_probe_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)

    # LLM calls (generation agents).
    llm_timeout_seconds: float = Field(default=30.0, gt=0.0, le=120.0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_reasoning_effort: Literal["low", "medium", "high"] = "low"
    llm_max_output_tokens: int = Field(default=4096, ge=256, le=32768)

    # Prompt Guard (supplementary, flag-only injection classifier).
    prompt_guard_chunk_chars: int = Field(default=800, ge=200, le=2000)
    prompt_guard_max_chunks: int = Field(default=16, ge=1, le=64)
    prompt_guard_concurrency: int = Field(default=4, ge=1, le=16)
    # Uncalibrated display threshold; calibration and any enforcement are M12.
    prompt_guard_display_threshold: float = Field(default=0.9, gt=0.0, lt=1.0)

    # Reports and the Critic loop (decisions D4 and D6).
    report_ttl_seconds: int = Field(default=3600, ge=60, le=MAX_REPORT_TTL_SECONDS)
    max_revisions: int = Field(default=1, ge=0, le=3)

    # Upload validation (deterministic; the only layer that rejects requests).
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    max_pdf_pages: int = Field(default=10, ge=1, le=100)
    max_extracted_chars: int = Field(default=50_000, ge=1_000, le=500_000)

    # HTTP surface.
    frontend_dir: Path = PROJECT_ROOT / "frontend"
    api_docs_enabled: bool = False
    # Hugging Face Spaces renders the app inside an iframe on huggingface.co.
    csp_frame_ancestors: str = "'self' https://huggingface.co"
    hsts_enabled: bool = False

    @field_validator("groq_api_key", mode="before")
    @classmethod
    def _blank_api_key_is_unset(cls, value: Any) -> Any:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @field_validator(*_MODEL_FIELDS)
    @classmethod
    def _model_name_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model name must not be blank")
        return value

    @field_validator("groq_api_base_url")
    @classmethod
    def _base_url_is_https(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith("https://") or any(ch.isspace() for ch in value):
            raise ValueError("must be an https:// URL without whitespace")
        return value

    @field_validator("csp_frame_ancestors")
    @classmethod
    def _frame_ancestors_safe(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ch in value for ch in ";,\r\n"):
            raise ValueError("must be a non-empty CSP source list without ';', ',' or newlines")
        return value

    @property
    def groq_api_key_configured(self) -> bool:
        """True when a non-blank key is present. Says nothing about validity."""
        return self.groq_api_key is not None

    @property
    def groq_models(self) -> tuple[str, ...]:
        """Distinct configured Groq model IDs, in role order."""
        return tuple(dict.fromkeys(getattr(self, name) for name in _MODEL_FIELDS))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, loaded once."""
    return Settings()
