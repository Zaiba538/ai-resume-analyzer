"""In-memory PDF text extraction (successor of legacy ``utils/pdf_loader.py``).

Differences from the legacy loader:

* reads from bytes, never from a file path;
* rejects encrypted PDFs and enforces a page cap before extracting text;
* joins pages with a blank line, so words no longer fuse across page breaks
  (the legacy loader concatenated pages with no separator);
* reports which pages produced no text (possible scanned images);
* maps every parser failure to a typed, user-safe error.

Per-page text is exactly ``page.extract_text()``, as in the legacy loader, so
the legacy output equals ``"".join(result.pages)``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader

from backend.parsing.errors import (
    EncryptedDocumentError,
    TooManyPagesError,
    UnreadableDocumentError,
)

PAGE_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class PdfText:
    pages: tuple[str, ...]
    page_count: int
    pages_without_text: int

    @property
    def text(self) -> str:
        return PAGE_SEPARATOR.join(page for page in self.pages if page.strip())


def extract_pdf_text(data: bytes, *, max_pages: int) -> PdfText:
    """Extract text from a PDF held in memory."""
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise EncryptedDocumentError()
        page_count = len(reader.pages)
    except EncryptedDocumentError:
        raise
    except MemoryError:
        raise
    except Exception as exc:  # untrusted input: any parser failure is "unreadable"
        raise UnreadableDocumentError() from exc

    if page_count == 0:
        raise UnreadableDocumentError()
    if page_count > max_pages:
        raise TooManyPagesError()

    pages: list[str] = []
    try:
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    except MemoryError:
        raise
    except Exception as exc:
        raise UnreadableDocumentError() from exc

    return PdfText(
        pages=tuple(pages),
        page_count=page_count,
        pages_without_text=sum(1 for page in pages if not page.strip()),
    )
