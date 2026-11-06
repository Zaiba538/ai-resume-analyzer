"""In-memory PDF extraction."""

from __future__ import annotations

import pytest

from backend.parsing.errors import (
    EncryptedDocumentError,
    TooManyPagesError,
    UnreadableDocumentError,
)
from backend.parsing.pdf import PAGE_SEPARATOR, extract_pdf_text
from backend.parsing.service import normalise_text
from tests.fixtures.documents import (
    encrypt_pdf,
    make_image_only_pdf,
    make_mixed_pdf,
    make_pdf,
)


def test_single_page_text() -> None:
    result = extract_pdf_text(make_pdf([["Skills", "Python and SQL"]]), max_pages=10)
    assert result.page_count == 1
    assert result.pages_without_text == 0
    assert "Skills" in result.text and "Python and SQL" in result.text


def test_pages_joined_with_separator_so_words_do_not_fuse() -> None:
    result = extract_pdf_text(make_pdf([["alpha"], ["omega"]]), max_pages=10)
    assert result.page_count == 2
    assert PAGE_SEPARATOR in result.text
    assert normalise_text(result.text) == "alpha\n\nomega"
    assert "alphaomega" not in result.text


def test_unicode_text_extracted() -> None:
    data = make_pdf([["José Müller-Łukasiewicz", "Ελληνικά Русский"]], unicode_font=True)
    text = extract_pdf_text(data, max_pages=10).text
    for word in ("José", "Müller-Łukasiewicz", "Ελληνικά", "Русский"):
        assert word in text


def test_page_cap_enforced_before_extraction() -> None:
    data = make_pdf([["p"]] * 3)
    assert extract_pdf_text(data, max_pages=3).page_count == 3
    with pytest.raises(TooManyPagesError):
        extract_pdf_text(data, max_pages=2)


def test_image_only_pages_counted() -> None:
    result = extract_pdf_text(make_image_only_pdf(2), max_pages=10)
    assert result.page_count == 2
    assert result.pages_without_text == 2
    assert result.text == ""


def test_mixed_document_reports_textless_pages() -> None:
    result = extract_pdf_text(make_mixed_pdf(), max_pages=10)
    assert (result.page_count, result.pages_without_text) == (2, 1)
    assert "Jordan Sample" in result.text


@pytest.mark.parametrize("password", ["", "secret"])
def test_encrypted_pdfs_rejected(password: str) -> None:
    data = encrypt_pdf(make_pdf([["confidential"]]), user_password=password)
    with pytest.raises(EncryptedDocumentError):
        extract_pdf_text(data, max_pages=10)


@pytest.mark.parametrize(
    "data",
    [
        b"%PDF-1.7\n",
        b"%PDF-1.7\n" + b"\x00\xff" * 500,
        b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"trailer << /Root 1 0 R >>\n%%EOF",
    ],
)
def test_corrupt_pdfs_are_unreadable_not_500(data: bytes) -> None:
    with pytest.raises(UnreadableDocumentError):
        extract_pdf_text(data, max_pages=10)


def test_truncated_pdf_is_unreadable() -> None:
    data = make_pdf([["hello"]])
    with pytest.raises(UnreadableDocumentError):
        extract_pdf_text(data[: len(data) // 3], max_pages=10)
