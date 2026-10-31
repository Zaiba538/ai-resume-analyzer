"""Shared agent types and input handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.core.errors import AppError
from backend.llm.structured import StructuredResult, Usage
from backend.schemas.resume import ResumeProfile
from backend.schemas.security import SecurityFlag
from backend.security.sanitize import sanitize_text


class AgentInputError(AppError):
    """Invalid agent input, rejected before any LLM call."""

    status_code = 422
    code = "invalid_agent_input"
    message = "The input is empty or invalid."


@dataclass(frozen=True)
class AgentResult[T: BaseModel]:
    output: T
    prompt_id: str
    flags: tuple[SecurityFlag, ...] = ()
    attempts: int = 1
    usage: Usage = Usage()

    @classmethod
    def from_structured(
        cls, result: StructuredResult, *, output: T, prompt_id: str
    ) -> AgentResult[T]:
        return cls(
            output=output,
            prompt_id=prompt_id,
            flags=result.flags,
            attempts=result.attempts,
            usage=result.usage,
        )


def clean_input(text: str, *, field: str) -> str:
    """Sanitise untrusted text; reject it if nothing meaningful remains."""
    cleaned = sanitize_text(text).text.strip()
    if not cleaned:
        raise AgentInputError(f"The {field} is empty.")
    return cleaned


def _sanitize_strings(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value).text
    if isinstance(value, list):
        return [_sanitize_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_strings(item) for key, item in value.items()}
    return value


def clean_profile(profile: ResumeProfile) -> ResumeProfile:
    """Re-sanitise every string in a profile before it enters a prompt.

    A profile may not have come from the resume agent's validated output, so it
    is treated as untrusted. Only the M3 Unicode sanitisation is applied; field
    names, structure and types are unchanged, and the result is re-validated
    against the strict schema. A profile left invalid by sanitisation (for
    example, an entry made only of invisible characters) is rejected.
    """
    try:
        return ResumeProfile.model_validate(_sanitize_strings(profile.model_dump(mode="python")))
    except ValidationError as exc:
        raise AgentInputError("The profile is invalid.") from exc
