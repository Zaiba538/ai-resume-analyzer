"""Security flags raised by injection detection and output validation.

Flags are informational: they are shown to the user and passed to the Critic,
and they never cause a request to be rejected (approved Rev. 3, correction 4).
Flags never contain the matched text, only its category and origin.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FlagType = Literal[
    "hidden_unicode",
    "instruction_override",
    "role_impersonation",
    "prompt_exfiltration",
    "score_manipulation",
    "delimiter_injection",
    "classifier",
    "unexpected_url",
]
FlagSource = Literal["resume", "job_description", "answer", "llm_output"]


class SecurityFlag(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: FlagType
    source: FlagSource
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    calibrated: bool = False
