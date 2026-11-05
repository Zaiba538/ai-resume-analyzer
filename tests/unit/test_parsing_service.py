"""Parsing pipeline: normalisation, caps, warnings, and the multipart boolean helper."""

from __future__ import annotations

import pytest

from backend.api.uploads import parse_bool_field
from backend.parsing.errors import (
    DocumentTooLongError,
    InvalidFieldError,
    NoExtractableTextError,
)
from backend.parsing.service import ParseLimits, normalise_text, parse_document
from tests.fixtures.documents import (
    SYNTHETIC_RESUME_LINES,
    make_docx,
    make_image_only_pdf,
    make_mixed_pdf,
    make_pdf,
)

LIMITS = ParseLimits(max_bytes=5 * 1024 * 1024, max_pages=10, max_chars=50_000)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a\r\nb\rc", "a\nb\nc"),
        ("a\x00b", "ab"),
        ("a   \nb\t\n", "a\nb"),
        ("a\n\n\n\n\nb", "a\n\nb"),
        ("\n\n  a  \n\n", "a"),
        ("", ""),
    ],
)
def test_normalise_text(raw: str, expected: str) -> None:
    assert normalise_text(raw) == expected


def test_pdf_document() -> None:
    parsed = parse_document("cv.pdf", make_pdf([list(SYNTHETIC_RESUME_LINES)]), LIMITS)
    assert parsed.file_type == "pdf"
    assert parsed.page_count == 1
    assert parsed.text.startswith("Jordan Sample")
    assert parsed.warnings == ()


def test_docx_document_has_no_page_count() -> None:
    parsed = parse_document("cv.docx", make_docx(SYNTHETIC_RESUME_LINES), LIMITS)
    assert (parsed.file_type, parsed.page_count) == ("docx", None)


def test_textless_pages_warning() -> None:
    parsed = parse_document("cv.pdf", make_mixed_pdf(), LIMITS)
    assert [w.code for w in parsed.warnings] == ["pages_without_text"]
    assert "1 of 2 pages" in parsed.warnings[0].message


def test_image_only_pdf_has_no_text() -> None:
    with pytest.raises(NoExtractableTextError):
        parse_document("scan.pdf", make_image_only_pdf(), LIMITS)


def test_whitespace_only_docx_has_no_text() -> None:
    with pytest.raises(NoExtractableTextError):
        parse_document("cv.docx", make_docx(["   ", "\t"]), LIMITS)


def test_character_cap_applies_after_normalisation() -> None:
    data = make_docx(["x" * 600, "", "", "", "y" * 399])
    assert len(parse_document("cv.docx", data, LIMITS).text) == 1001
    tight = ParseLimits(max_bytes=LIMITS.max_bytes, max_pages=10, max_chars=1001)
    parse_document("cv.docx", data, tight)
    with pytest.raises(DocumentTooLongError):
        parse_document("cv.docx", data, ParseLimits(LIMITS.max_bytes, 10, 1000))


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("true", True), (" On ", True), ("1", True), ("false", False), ("", False)],
)
def test_parse_bool_field(value: str | None, expected: bool) -> None:
    assert parse_bool_field(value) is expected


@pytest.mark.parametrize("value", ["maybe", "2", "truee"])
def test_parse_bool_field_rejects_garbage(value: str) -> None:
    with pytest.raises(InvalidFieldError):
        parse_bool_field(value)


# --- English-only check (language validation) -------------------------------------------------


def test_english_resume_with_devanagari_is_parsed_unchanged() -> None:
    from tests.unit.test_language import DEVANAGARI_NAME_ADDRESS_LINES, LANGUAGES_LINE

    lines = [*DEVANAGARI_NAME_ADDRESS_LINES, LANGUAGES_LINE]
    parsed = parse_document("cv.docx", make_docx(lines), LIMITS)
    assert parsed.text == "\n".join(lines)


@pytest.mark.parametrize("kind", ["docx", "pdf"])
@pytest.mark.parametrize("sample", ["hindi", "marathi"])
def test_hindi_and_marathi_resumes_are_rejected(kind: str, sample: str) -> None:
    from backend.parsing.language import UnsupportedLanguageError
    from tests.unit.test_language import (
        HINDI_RESUME_LINES,
        MARATHI_RESUME_LINES,
        make_devanagari_pdf,
    )

    lines = HINDI_RESUME_LINES if sample == "hindi" else MARATHI_RESUME_LINES
    data = make_docx(lines) if kind == "docx" else make_devanagari_pdf(lines)
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        parse_document(f"cv.{kind}", data, LIMITS)
    assert excinfo.value.details[0].loc == ["resume"]


def test_existing_errors_take_precedence_over_the_language_check() -> None:
    from tests.unit.test_language import HINDI_RESUME_LINES

    small = ParseLimits(max_bytes=LIMITS.max_bytes, max_pages=10, max_chars=20)
    with pytest.raises(DocumentTooLongError):
        parse_document("cv.docx", make_docx(HINDI_RESUME_LINES), small)
    with pytest.raises(NoExtractableTextError):
        parse_document("cv.pdf", make_image_only_pdf(), LIMITS)
