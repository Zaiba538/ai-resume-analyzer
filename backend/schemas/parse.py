"""Models for ``POST /api/v1/parse``."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from backend.schemas.common import ApiWarning

PREVIEW_MAX_CHARS = 1000


class ParseResponse(BaseModel):
    """Result of parsing an uploaded resume. Nothing is stored server-side."""

    model_config = ConfigDict(extra="forbid")

    text_length: int = Field(ge=0, description="Characters of normalised extracted text")
    page_count: int | None = Field(description="PDF page count; null for DOCX")
    sections: dict[str, bool] = Field(description="Detected standard resume sections")
    preview: str | None = Field(
        default=None,
        max_length=PREVIEW_MAX_CHARS,
        description="First 1000 characters of extracted text; only if include_preview=true",
    )
    warnings: list[ApiWarning]
