"""Models shared across the API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ErrorDetail(BaseModel):
    """One validation problem. Never carries the rejected input value."""

    model_config = ConfigDict(extra="forbid")

    loc: list[str]
    type: str
    msg: str


class ErrorBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    request_id: str
    details: list[ErrorDetail] | None = None


class ApiWarning(BaseModel):
    """A non-fatal issue the client should know about. Never contains user data."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class ErrorEnvelope(BaseModel):
    """The single error response shape: ``{"error": {...}}``."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorBody
