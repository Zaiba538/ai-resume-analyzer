"""Smoke tests: ReportLab renders Unicode text with the bundled fonts.

M0: checks that a PDF is produced in memory, that the DejaVu Sans TrueType font
is embedded, that Unicode text round-trips through text extraction, and that
markup-like text is rendered literally once escaped (ReportLab's Paragraph
otherwise interprets it as markup).

M8: checks that the bundled Noto Sans Devanagari fonts cover Devanagari and
that HarfBuzz shaping (reportlab[shaping], uharfbuzz) is available and forms
conjuncts. Shaped Devanagari may not copy or search correctly in the PDF (an
accepted limitation), so these tests check fonts and glyphs, not extracted text.
"""

from __future__ import annotations

import io
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
import uharfbuzz
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
NOTO_FONTS = {
    "NotoSansDevanagari": "NotoSansDevanagari-Regular.ttf",
    "NotoSansDevanagari-Bold": "NotoSansDevanagari-Bold.ttf",
}
DEVANAGARI_SAMPLES = {
    "hindi": "हिन्दी प्रबंधक",
    "marathi": "महाराष्ट्र विकास",
    "conjuncts": "क्षमता ज्ञान श्री",
}


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


@pytest.fixture(scope="module")
def noto_fonts() -> dict[str, TTFont]:
    for name, filename in NOTO_FONTS.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / filename)))
    return {name: pdfmetrics.getFont(name) for name in NOTO_FONTS}


def _devanagari_pdf(font_name: str, *, shaping: int) -> bytes:
    style = ParagraphStyle("deva", fontName=font_name, fontSize=12, leading=18, shaping=shaping)
    story = [Paragraph(escape(text), style) for text in DEVANAGARI_SAMPLES.values()]
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, invariant=1).build(story)
    return buffer.getvalue()


def test_dejavu_has_no_devanagari_glyphs(font_name: str) -> None:
    # Devanagari needs the Noto face; DejaVu Sans does not cover it.
    face = pdfmetrics.getFont(font_name).face
    assert not any(ord(ch) in face.charToGlyph for ch in "हिन्दी")


@pytest.mark.parametrize("font", list(NOTO_FONTS))
@pytest.mark.parametrize("script", list(DEVANAGARI_SAMPLES))
def test_noto_has_devanagari_glyphs(noto_fonts, font: str, script: str) -> None:
    face = noto_fonts[font].face
    text = DEVANAGARI_SAMPLES[script]
    missing = {ch for ch in text if not ch.isspace() and ord(ch) not in face.charToGlyph}
    assert not missing, f"{font}/{script}: no glyph for {sorted(missing)}"


@pytest.mark.parametrize("font", list(NOTO_FONTS))
def test_shaping_is_available_for_noto(noto_fonts, font: str) -> None:
    assert uharfbuzz.__version__ == "0.57.0"
    assert noto_fonts[font].shapable


@pytest.mark.parametrize("font", list(NOTO_FONTS))
def test_harfbuzz_forms_conjuncts_with_noto(font: str) -> None:
    data = (FONT_DIR / NOTO_FONTS[font]).read_bytes()
    hb_font = uharfbuzz.Font(uharfbuzz.Face(data))
    buffer = uharfbuzz.Buffer()
    buffer.add_str("क्ष")  # KA + VIRAMA + SSA: a single conjunct glyph when shaped
    buffer.guess_segment_properties()
    uharfbuzz.shape(hb_font, buffer)
    glyphs = [info.codepoint for info in buffer.glyph_infos]
    assert buffer.script == "Deva"
    assert len(glyphs) == 1
    assert 0 not in glyphs, "shaping produced .notdef"


@pytest.mark.parametrize("font", list(NOTO_FONTS))
def test_shaped_devanagari_pdf_embeds_noto(noto_fonts, font: str) -> None:
    shaped = _devanagari_pdf(font, shaping=1)
    assert shaped.startswith(b"%PDF-")
    assert shaped != _devanagari_pdf(font, shaping=0), "shaping had no effect"
    fonts = PdfReader(io.BytesIO(shaped)).pages[0]["/Resources"]["/Font"]
    embedded = [
        str(obj["/BaseFont"])
        for obj in (ref.get_object() for ref in fonts.values())
        if obj["/Subtype"] == "/TrueType" and "/FontFile2" in obj["/FontDescriptor"]
    ]
    postscript_name = Path(NOTO_FONTS[font]).stem
    assert embedded
    assert all(name.endswith(f"+{postscript_name}") for name in embedded), embedded
