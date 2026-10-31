"""M0 smoke test: ReportLab renders Unicode text with the bundled DejaVu Sans font.

Checks that a PDF is produced in memory, that the TrueType font is embedded,
that Unicode text round-trips through text extraction, and that markup-like
text is rendered literally once escaped (ReportLab's Paragraph otherwise
interprets it as markup).
"""

from __future__ import annotations

import io
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
from pypdf import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate

FONT_DIR = Path(__file__).resolve().parents[2] / "backend" / "assets" / "fonts"

SAMPLES = {
    "latin_extended": "José Müller-Łukasiewicz, Ångström, naïve café, İstanbul ğş",
    "greek": "Ελληνικά δεδομένα",
    "cyrillic": "Русский текст",
    "punctuation": "€ 1 250 — “quoted” • ✓",
}
MARKUP_TEXT = "<b>not bold</b> & <a href='https://example.invalid'>not a link</a>"


@pytest.fixture(scope="module")
def font_name() -> str:
    pdfmetrics.registerFont(TTFont("DejaVuSans", str(FONT_DIR / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
    return "DejaVuSans"


@pytest.fixture(scope="module")
def pdf_bytes(font_name: str) -> bytes:
    style = ParagraphStyle("body", fontName=font_name, fontSize=11, leading=14)
    story = [Paragraph(escape(text), style) for text in SAMPLES.values()]
    story.append(Paragraph(escape(MARKUP_TEXT), style))
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title="Smoke test").build(story)
    return buffer.getvalue()


def _extracted_text(pdf: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)


def test_fonts_and_licence_are_bundled() -> None:
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "LICENSE-DejaVu.txt"):
        assert (FONT_DIR / name).is_file(), name


def test_pdf_is_generated_in_memory(pdf_bytes: bytes) -> None:
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(PdfReader(io.BytesIO(pdf_bytes)).pages) == 1


def test_truetype_font_is_embedded(pdf_bytes: bytes) -> None:
    page = PdfReader(io.BytesIO(pdf_bytes)).pages[0]
    fonts = page["/Resources"]["/Font"]
    embedded = [
        font.get_object()
        for font in fonts.values()
        if font.get_object()["/Subtype"] == "/TrueType"
        and "/FontFile2" in font.get_object()["/FontDescriptor"]
    ]
    assert embedded, "DejaVu Sans should be embedded as a TrueType subset"
    assert all("DejaVuSans" in str(font["/BaseFont"]) for font in embedded)


@pytest.mark.parametrize("script", list(SAMPLES))
def test_unicode_text_round_trips(pdf_bytes: bytes, script: str) -> None:
    extracted = _extracted_text(pdf_bytes)
    for word in SAMPLES[script].split():
        assert word in extracted, f"{script}: {word!r} missing from extracted text"


def test_escaped_markup_is_rendered_literally(pdf_bytes: bytes) -> None:
    extracted = _extracted_text(pdf_bytes)
    assert "<b>not bold</b>" in extracted
    assert "<a href='https://example.invalid'>not a link</a>" in extracted


@pytest.mark.parametrize("script", list(SAMPLES))
def test_font_has_glyphs_for_supported_scripts(font_name: str, script: str) -> None:
    face = pdfmetrics.getFont(font_name).face
    missing = {ch for ch in SAMPLES[script] if not ch.isspace() and ord(ch) not in face.charToGlyph}
    assert not missing, f"{script}: no glyph for {sorted(missing)}"


@pytest.mark.xfail(
    strict=True,
    reason="Not yet supported: DejaVu Sans has no Devanagari glyphs. Planned for M8 "
    "(pending approval): bundle Noto Sans Devanagari (OFL-1.1) and enable "
    "ReportLab's optional HarfBuzz shaping (reportlab[shaping] / uharfbuzz).",
)
def test_font_has_devanagari_glyphs(font_name: str) -> None:
    face = pdfmetrics.getFont(font_name).face
    assert all(ord(ch) in face.charToGlyph for ch in "हिन्दी")
