"""``POST /api/v1/parse``: validate a resume upload and extract its text.

Request: ``multipart/form-data`` with ``file`` (PDF or DOCX) and optional
``include_preview`` (boolean, default false).

Response: text length, page count (PDF only), detected sections, warnings,
and, only when requested, a preview of at most 1000 characters. The upload,
extracted text and preview are not stored, and the response is served with
``Cache-Control: no-store``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.api.deps import SettingsDep
from backend.api.uploads import parse_bool_field, read_multipart_upload
from backend.parsing.sections import detect_sections
from backend.parsing.service import ParseLimits, parse_document
from backend.schemas.common import ApiWarning, ErrorEnvelope
from backend.schemas.parse import PREVIEW_MAX_CHARS, ParseResponse

router = APIRouter(prefix="/api/v1", tags=["parse"])

INCLUDE_PREVIEW_FIELD = "include_preview"

_ERRORS = {
    status: {"model": ErrorEnvelope, "description": description}
    for status, description in {
        400: "Malformed multipart body",
        413: "File or request too large",
        415: "Unsupported, mismatched or protected file type",
        422: "Invalid upload, unreadable document, limits exceeded or unsupported language",
    }.items()
}

_REQUEST_BODY = {
    "required": True,
    "content": {
        "multipart/form-data": {
            "schema": {
                "type": "object",
                "required": ["file"],
                "properties": {
                    "file": {"type": "string", "format": "binary"},
                    INCLUDE_PREVIEW_FIELD: {"type": "boolean", "default": False},
                },
            }
        }
    },
}


@router.post(
    "/parse",
    response_model=ParseResponse,
    responses=_ERRORS,
    openapi_extra={"requestBody": _REQUEST_BODY},
)
async def parse_resume(request: Request, settings: SettingsDep) -> JSONResponse:
    form = await read_multipart_upload(
        request,
        max_file_bytes=settings.max_upload_bytes,
        allowed_fields=(INCLUDE_PREVIEW_FIELD,),
    )
    include_preview = parse_bool_field(form.fields.get(INCLUDE_PREVIEW_FIELD))

    limits = ParseLimits(
        max_bytes=settings.max_upload_bytes,
        max_pages=settings.max_pdf_pages,
        max_chars=settings.max_extracted_chars,
    )
    parsed = await run_in_threadpool(parse_document, form.filename, form.file_bytes, limits)
    sections = detect_sections(parsed.text)

    warnings = [ApiWarning(code=w.code, message=w.message) for w in parsed.warnings]
    if not any(present for name, present in sections.items() if name != "contact"):
        warnings.append(
            ApiWarning(
                code="no_sections_detected",
                message="No standard resume section headings were detected.",
            )
        )

    response = ParseResponse(
        text_length=len(parsed.text),
        page_count=parsed.page_count,
        sections=sections,
        preview=parsed.text[:PREVIEW_MAX_CHARS] if include_preview else None,
        warnings=warnings,
    )
    body = response.model_dump(mode="json", exclude={"preview"} if not include_preview else None)
    return JSONResponse(body)
