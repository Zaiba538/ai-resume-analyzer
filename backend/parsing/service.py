"""Document parsing pipeline: validate, extract, normalise, enforce caps.

CPU-bound and synchronous; API handlers run it in a worker thread.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from backend.parsing.docx import extract_docx_text
from backend.parsing.errors import DocumentTooLongError, NoExtractableTextError
from backend.parsing.pdf import extract_pdf_text
from backend.parsing.validate import FileType, validate_upload

logger = logging.getLogger("backend.parsing")

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+\n")


@dataclass(frozen=True)
class ParseLimits:
    max_bytes: int
    max_pages: int
    max_chars: int


@dataclass(frozen=True)
class DocumentWarning:
    code: str
    message: str


@dataclass(frozen=True)
class ParsedDocument:
    file_type: FileType
    text: str
    page_count: int | None
    warnings: tuple[DocumentWarning, ...] = field(default_factory=tuple)


def normalise_text(text: str) -> str:
    """Line endings to ``\\n``, NULs removed, trailing spaces and blank runs trimmed."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _TRAILING_SPACE.sub("\n", text + "\n")
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


def parse_document(filename: str, data: bytes, limits: ParseLimits) -> ParsedDocument:
    """Validate and extract text from an uploaded resume held in memory."""
    file_type = validate_upload(filename, data, max_bytes=limits.max_bytes)
    warnings: list[DocumentWarning] = []

    if file_type == "pdf":
        pdf = extract_pdf_text(data, max_pages=limits.max_pages)
        raw_text, page_count = pdf.text, pdf.page_count
        if 0 < pdf.pages_without_text < pdf.page_count:
            warnings.append(
                DocumentWarning(
                    "pages_without_text",
                    f"{pdf.pages_without_text} of {pdf.page_count} pages contained no "
                    "extractable text; they may be scanned images and were skipped.",
                )
            )
    else:
        raw_text, page_count = extract_docx_text(data), None

    text = normalise_text(raw_text)
    if not text:
        raise NoExtractableTextError()
    if len(text) > limits.max_chars:
        raise DocumentTooLongError()

    logger.info(
        "document_parsed",
        extra={
            "file_type": file_type,
            "file_size_bytes": len(data),
            "page_count": page_count,
            "char_count": len(text),
        },
    )
    return ParsedDocument(
        file_type=file_type, text=text, page_count=page_count, warnings=tuple(warnings)
    )
