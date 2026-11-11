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

from backend.core.rate_limit import parse_rate_limits

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Decision D6: derived report data may stay in server memory for at most one hour.
MAX_REPORT_TTL_SECONDS = 3600

# M9 decision Q7: bounds of the configurable answer length (characters, after sanitisation).
MIN_ANSWER_CHARS_LIMIT = 200
MAX_ANSWER_CHARS_LIMIT = 20_000

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_MODEL_FIELDS = (
    "groq_model_resume",
    "groq_model_skill_gap",
    "groq_model_critic",
    "groq_model_interview",
    "groq_model_evaluator",
    "groq_model_prompt_guard",
)

_RATE_LIMIT_FIELDS = (
    "rate_limit_analyze",
    "rate_limit_analyze_global",
    "rate_limit_interview_questions",
    "rate_limit_interview_evaluate",
    "rate_limit_interview_global",
    "rate_limit_match",
    "rate_limit_parse",
    "rate_limit_report",
    "rate_limit_report_pdf",
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
    # M7 decision D4: overall deadline for one analysis workflow run.
    analysis_deadline_seconds: float = Field(default=600.0, ge=1.0, le=3600.0)
    # M7 report store: capacity (approved: 500) and the periodic expiry sweep.
    max_stored_reports: int = Field(default=500, ge=1, le=10_000)
    report_sweep_interval_seconds: float = Field(
        default=60.0, ge=1.0, le=float(MAX_REPORT_TTL_SECONDS)
    )

    # Upload validation (deterministic; the only layer that rejects requests).
    max_upload_bytes: int = Field(default=5 * 1024 * 1024, ge=1024, le=20 * 1024 * 1024)
    max_pdf_pages: int = Field(default=10, ge=1, le=100)
    max_extracted_chars: int = Field(default=50_000, ge=1_000, le=500_000)
    max_job_description_chars: int = Field(default=20_000, ge=500, le=100_000)

    # M9 interview API (decision Q7): answer length and the overall deadline of each
    # endpoint. A deadline bounds the whole request, including upstream retries.
    max_answer_chars: int = Field(
        default=5_000, ge=MIN_ANSWER_CHARS_LIMIT, le=MAX_ANSWER_CHARS_LIMIT
    )
    interview_deadline_seconds: float = Field(default=120.0, ge=1.0, le=600.0)
    evaluation_deadline_seconds: float = Field(default=120.0, ge=1.0, le=600.0)

    # M10 rate limiting: in memory, per process. Limits use the ``limits`` notation
    # ("5/minute;30/hour"). Groq quotas are unverified, so these are configurable
    # starting defaults. ``*_global`` limits are shared by all clients; both
    # interview endpoints draw on the same global pool.
    rate_limit_enabled: bool = True
    rate_limit_analyze: str = "5/minute;30/hour"
    rate_limit_analyze_global: str = "20/minute"
    rate_limit_interview_questions: str = "10/minute;60/hour"
    rate_limit_interview_evaluate: str = "20/minute;200/hour"
    rate_limit_interview_global: str = "60/minute"
    rate_limit_match: str = "30/minute"
    rate_limit_parse: str = "30/minute"
    rate_limit_report: str = "60/minute"
    rate_limit_report_pdf: str = "20/minute"
    # Capacity of the limiter's per-client store (one key per client, endpoint and
    # window). When full, requests needing a new key are rejected until keys expire.
    # Measured: about 1 KB per key plus 32 bytes per counted request; the upper bound
    # keeps the worst case within a constrained container.
    rate_limit_max_tracked_keys: int = Field(default=10_000, ge=100, le=25_000)
    # Concurrent /analyze requests in this process (enforced from M10 Phase 2).
    analyze_max_concurrent: int = Field(default=2, ge=1, le=8)

    # Document processing (Phase 3, M1): documents parsed at the same time by
    # /parse, /match and /analyze together, and the time a request waits for its
    # document. The deadline is soft: it ends the request, not the worker thread.
    parse_max_concurrent: int = Field(default=2, ge=1, le=8)
    parse_deadline_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    # Embeddings for semantic similarity (loaded once at start-up). The revision
    # pins exact model files on the Hugging Face Hub; remote code is never run.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

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

    @field_validator(*_RATE_LIMIT_FIELDS)
    @classmethod
    def _rate_limit_valid(cls, value: str) -> str:
        parse_rate_limits(value)  # raises ValueError with a fixed message
        return value.strip()

    @field_validator("embedding_model_name", "embedding_model_revision")
    @classmethod
    def _identifier_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ch.isspace() for ch in value):
            raise ValueError("must be non-empty without whitespace")
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
