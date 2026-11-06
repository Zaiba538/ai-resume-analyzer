"""In-memory DOCX extraction: body, tables in order, headers, footers."""

from __future__ import annotations

import io

import docx
import pytest

from backend.parsing.docx import CELL_SEPARATOR, extract_docx_text
from backend.parsing.errors import UnreadableDocumentError, UnsupportedFileTypeError
from tests.fixtures.documents import docx_entries, make_docx, make_zip


def test_paragraphs_extracted_in_order() -> None:
    text = extract_docx_text(make_docx(["Summary", "Skills", "Python"]))
    assert text.splitlines() == ["Summary", "Skills", "Python"]


def test_tables_extracted() -> None:
    data = make_docx(["Skills"], tables=[[["Python", "SQL"], ["Docker", "Spark"]]])
    lines = extract_docx_text(data).splitlines()
    assert f"Python{CELL_SEPARATOR}SQL" in lines
    assert f"Docker{CELL_SEPARATOR}Spark" in lines


def test_tables_interleaved_in_document_order() -> None:
    document = docx.Document()
    document.add_paragraph("Before")
    document.add_table(rows=1, cols=1).cell(0, 0).text = "Inside"
    document.add_paragraph("After")
    buffer = io.BytesIO()
    document.save(buffer)
    lines = [line for line in extract_docx_text(buffer.getvalue()).splitlines() if line]
    assert lines == ["Before", "Inside", "After"]


def test_merged_cells_emitted_once() -> None:
    document = docx.Document()
    table = document.add_table(rows=1, cols=3)
    merged = table.cell(0, 0).merge(table.cell(0, 1))
    merged.text = "Merged"
    table.cell(0, 2).text = "Right"
    buffer = io.BytesIO()
    document.save(buffer)
    assert f"Merged{CELL_SEPARATOR}Right" in extract_docx_text(buffer.getvalue()).splitlines()


def test_nested_tables_extracted() -> None:
    document = docx.Document()
    outer = document.add_table(rows=1, cols=1).cell(0, 0)
    outer.paragraphs[0].text = "Outer"
    outer.add_table(rows=1, cols=1).cell(0, 0).text = "Nested"
    buffer = io.BytesIO()
    document.save(buffer)
    lines = extract_docx_text(buffer.getvalue()).splitlines()
    assert "Outer" in lines and "Nested" in lines


def test_header_first_footer_last() -> None:
    data = make_docx(["Body"], header="Jordan Sample", footer="Page footer")
    lines = [line for line in extract_docx_text(data).splitlines() if line]
    assert lines == ["Jordan Sample", "Body", "Page footer"]


def test_undefined_headers_add_nothing() -> None:
    assert extract_docx_text(make_docx(["Only body"])).strip() == "Only body"


def test_repeated_headers_across_sections_deduplicated() -> None:
    document = docx.Document()
    document.sections[0].header.paragraphs[0].text = "Shared header"
    document.add_paragraph("Section one")
    second = document.add_section()
    second.header.is_linked_to_previous = False
    second.header.paragraphs[0].text = "Shared header"
    document.add_paragraph("Section two")
    buffer = io.BytesIO()
    document.save(buffer)
    assert extract_docx_text(buffer.getvalue()).count("Shared header") == 1


def test_macro_enabled_package_rejected() -> None:
    entries = docx_entries(make_docx(["x"]))
    entries["[Content_Types].xml"] = entries["[Content_Types].xml"].replace(
        b"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
        b"application/vnd.ms-word.document.macroEnabled.main+xml",
    )
    with pytest.raises(UnsupportedFileTypeError):
        extract_docx_text(make_zip(entries))


def test_malformed_document_xml_is_unreadable() -> None:
    entries = docx_entries(make_docx(["x"]))
    entries["word/document.xml"] = b"<w:document><unclosed>"
    with pytest.raises(UnreadableDocumentError):
        extract_docx_text(make_zip(entries))
