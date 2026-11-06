"""M2 exit criterion: parity with the legacy Streamlit extraction.

The legacy loaders (``utils/pdf_loader.py``, ``utils/docx_loader.py``) are
imported unmodified and run on the same synthetic documents.

* PDF: per-page text is identical; the only difference is the page separator,
  so ``"".join(new.pages)`` must equal the legacy output exactly.
* DOCX: with only body paragraphs, the outputs are identical after
  normalisation. With tables, headers and footers, every legacy line still
  appears in the same order, and the new extractor additionally recovers the
  content the legacy loader dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.parsing.docx import extract_docx_text
from backend.parsing.pdf import extract_pdf_text
from backend.parsing.service import normalise_text
from tests.fixtures.documents import SYNTHETIC_RESUME_LINES, make_docx, make_pdf
from utils.docx_loader import extract_text_from_docx as legacy_docx
from utils.pdf_loader import extract_text_from_pdf as legacy_pdf

PDF_CASES = {
    "one_page": [list(SYNTHETIC_RESUME_LINES)],
    "three_pages": [["Summary", "alpha"], ["Skills", "beta"], ["Education", "gamma"]],
}


def _write(tmp_path: Path, name: str, data: bytes) -> str:
    path = tmp_path / name
    path.write_bytes(data)
    return str(path)


@pytest.mark.parametrize("case", list(PDF_CASES))
def test_pdf_page_text_identical_to_legacy(tmp_path: Path, case: str) -> None:
    data = make_pdf(PDF_CASES[case])
    legacy = legacy_pdf(_write(tmp_path, "cv.pdf", data))
    new = extract_pdf_text(data, max_pages=10)
    assert "".join(new.pages) == legacy


def test_pdf_new_output_only_adds_page_separators(tmp_path: Path) -> None:
    data = make_pdf(PDF_CASES["three_pages"])
    legacy = legacy_pdf(_write(tmp_path, "cv.pdf", data))
    new = extract_pdf_text(data, max_pages=10).text
    assert new.replace("\n\n", "") == legacy.replace("\n\n", "")
    assert len(new) > len(legacy)


def test_docx_body_identical_to_legacy(tmp_path: Path) -> None:
    data = make_docx(SYNTHETIC_RESUME_LINES)
    legacy = legacy_docx(_write(tmp_path, "cv.docx", data))
    assert normalise_text(extract_docx_text(data)) == normalise_text(legacy)


def test_docx_superset_of_legacy_in_order(tmp_path: Path) -> None:
    data = make_docx(
        SYNTHETIC_RESUME_LINES,
        tables=[[["Python", "SQL"], ["Spark", "Docker"]]],
        header="Jordan Sample - Data Engineer",
        footer="References available on request",
    )
    legacy_lines = [line for line in legacy_docx(_write(tmp_path, "cv.docx", data)).splitlines()]
    new_lines = extract_docx_text(data).splitlines()

    iterator = iter(new_lines)
    assert all(line in iterator for line in legacy_lines), "legacy lines missing or reordered"

    legacy_text = "\n".join(legacy_lines)
    for recovered in ("Python | SQL", "Jordan Sample - Data Engineer", "References available"):
        assert recovered not in legacy_text
        assert any(recovered in line for line in new_lines)
