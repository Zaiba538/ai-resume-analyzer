"""Synthetic test documents, generated in memory.

All content is fictional. No real resumes or personal data are used, and no
binary fixtures are committed.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import docx
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

FONT_DIR = Path(__file__).resolve().parents[2] / "backend" / "assets" / "fonts"
_UNICODE_FONT = "DejaVuSans"

SYNTHETIC_RESUME_LINES = (
    "Jordan Sample",
    "jordan.sample@example.invalid | +1 555 010 0199",
    "Summary",
    "Data engineer building reliable batch pipelines.",
    "Skills",
    "Python, SQL, Apache Spark, Docker",
    "Experience",
    "Acme Analytics (fictional) - Data Engineer, 2021 - 2024",
    "Built batch ETL pipelines in Apache Spark.",
    "Education",
    "B.Sc. Computer Science, Example State University, 2021",
    "Projects",
    "Inventory forecaster: Python demand forecasting prototype.",
    "Certifications",
    "Example Cloud Practitioner (fictional), 2023",
)


def _ensure_unicode_font() -> None:
    if _UNICODE_FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(_UNICODE_FONT, str(FONT_DIR / "DejaVuSans.ttf")))


def make_pdf(pages: Sequence[Sequence[str]], *, unicode_font: bool = False) -> bytes:
    """A text PDF with one entry per page, each a list of lines."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    font = "Helvetica"
    if unicode_font:
        _ensure_unicode_font()
        font = _UNICODE_FONT
    for lines in pages:
        pdf.setFont(font, 11)
        y = 800
        for line in lines:
            pdf.drawString(50, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_image_only_pdf(page_count: int = 1) -> bytes:
    """Pages containing only vector graphics: no extractable text (scan-like)."""
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    for _ in range(page_count):
        pdf.rect(50, 500, 300, 200, fill=1)
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def make_mixed_pdf() -> bytes:
    """Page 1 has text, page 2 is image-only."""
    writer = PdfWriter()
    for source in (make_pdf([SYNTHETIC_RESUME_LINES]), make_image_only_pdf()):
        writer.append(PdfReader(io.BytesIO(source)))
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def encrypt_pdf(data: bytes, *, user_password: str = "") -> bytes:
    """RC4-128 encryption (pure Python in pypdf); empty user password by default."""
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
    writer.encrypt(
        user_password=user_password,
        owner_password="owner",  # noqa: S106 - synthetic test fixture, not a credential
        algorithm="RC4-128",
    )
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_docx(
    paragraphs: Sequence[str] = SYNTHETIC_RESUME_LINES,
    *,
    tables: Sequence[Sequence[Sequence[str]]] = (),
    header: str | None = None,
    footer: str | None = None,
) -> bytes:
    """A DOCX with paragraphs, then optional tables, header and footer."""
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    for rows in tables:
        table = document.add_table(rows=len(rows), cols=len(rows[0]))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.cell(r, c).text = value
    section = document.sections[0]
    if header is not None:
        section.header.paragraphs[0].text = header
    if footer is not None:
        section.footer.paragraphs[0].text = footer
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_zip(entries: Mapping[str, bytes], *, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def docx_entries(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        return {info.filename: archive.read(info) for info in archive.infolist()}
