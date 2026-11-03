"""Typed PDF rendering errors (M8, decisions D5, E1 and E2).

The renderer raises these instead of returning partial output. They are plain
exceptions, independent of FastAPI and HTTP: each has a stable ``code`` and a
fixed, safe ``message`` that never contains report content. The API layer maps
them to responses (E1: page limit 422 ``pdf_page_limit_exceeded``, deadline 504
``pdf_render_timeout``; E2: renderer unavailable 503 ``pdf_renderer_unavailable``).
"""

from __future__ import annotations


class PdfRenderError(Exception):
    """Base class: no PDF was produced."""

    code: str = "pdf_render_error"
    message: str = "The PDF report could not be generated."

    def __init__(self) -> None:
        super().__init__(self.message)


class PdfPageLimitError(PdfRenderError):
    """The report needs more pages than the configured maximum (never truncated)."""

    code = "pdf_page_limit_exceeded"
    message = "The report is too long to render as a PDF."

    def __init__(self, max_pages: int) -> None:
        super().__init__()
        self.max_pages = max_pages


class PdfRenderTimeoutError(PdfRenderError):
    """Rendering did not finish before its deadline."""

    code = "pdf_render_timeout"
    message = "Generating the PDF report took too long."

    def __init__(self, deadline_seconds: float) -> None:
        super().__init__()
        self.deadline_seconds = deadline_seconds


class PdfRendererUnavailableError(PdfRenderError):
    """The PDF fonts are not registered, so the renderer cannot run."""

    code = "pdf_renderer_unavailable"
    message = "The PDF renderer is not available."
