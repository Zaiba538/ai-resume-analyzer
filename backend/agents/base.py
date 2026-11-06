"""Shared agent types and input handling."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from backend.core.errors import AppError
from backend.llm.structured import StructuredResult, Usage
from backend.schemas.critic import MAX_CRITIC_ISSUES, AgentName, CriticIssue
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


def sanitize_data(value: Any) -> Any:
    """Apply M3 Unicode sanitisation to every string in JSON-like data."""
    return _sanitize_strings(value)


def sanitize_model[T: BaseModel](model: T) -> T:
    """Apply M3 Unicode sanitisation to every string of a model and re-validate it.

    Raises ``pydantic.ValidationError`` if sanitisation leaves the model invalid.
    """
    return type(model).model_validate(_sanitize_strings(model.model_dump(mode="python")))


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


def clean_feedback(feedback: Sequence[CriticIssue] | None, agent: AgentName) -> str | None:
    """Critic feedback for ``agent`` as JSON for a ``critic_feedback`` block, or ``None``.

    M7 (decision D1). ``feedback`` holds published Critic issues; only those
    whose ``agent`` is ``agent`` are kept, so one agent's feedback never reaches
    another agent. Each kept issue is rebuilt from its four published fields,
    re-sanitised (M3) and re-validated. Malformed or oversized feedback is
    rejected before any LLM call, without echoing its content. ``None`` (no
    feedback, or none for this agent) means the agent runs exactly as without
    feedback.
    """
    if not feedback:
        return None
    if isinstance(feedback, str | bytes) or not isinstance(feedback, Sequence):
        raise AgentInputError("Critic feedback must be a list of Critic issues.")
    if len(feedback) > MAX_CRITIC_ISSUES:
        raise AgentInputError(f"Critic feedback is limited to {MAX_CRITIC_ISSUES} issues.")
    relevant: list[dict[str, str]] = []
    for item in feedback:
        if not isinstance(item, CriticIssue):
            raise AgentInputError("Critic feedback must contain Critic issues only.")
        if item.agent != agent:
            continue
        fields = {
            "agent": item.agent,
            "field": item.field,
            "severity": item.severity,
            "message": item.message,
        }
        try:
            issue = CriticIssue.model_validate(_sanitize_strings(fields))
        except ValidationError as exc:
            raise AgentInputError("Critic feedback is invalid.") from exc
        relevant.append(
            {"field": issue.field, "severity": issue.severity, "message": issue.message}
        )
    if not relevant:
        return None
    return json.dumps(relevant, ensure_ascii=False)
