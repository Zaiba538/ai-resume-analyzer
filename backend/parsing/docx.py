"""In-memory DOCX text extraction (successor of legacy ``utils/docx_loader.py``).

The legacy loader read body paragraphs only, silently dropping text in tables,
headers and footers, which many resume templates use for layout. This module
extracts, in order:

1. text of every defined header (default, first-page, even-page; de-duplicated);
2. the body in document order: paragraphs and tables interleaved, with table
   rows rendered as `` | ``-separated cells, merged cells once, and nested
   tables up to ``MAX_TABLE_DEPTH``;
3. text of every defined footer (de-duplicated).

For a document with no tables, headers or footers, the body output equals the
legacy loader's output (modulo normalisation).

Known limitation: floating text boxes are not extracted. In DOCX they are
stored twice (DrawingML plus a VML fallback), so naive extraction duplicates
text. Supporting them needs a deliberate design.

python-docx parses XML with ``resolve_entities=False``, so XML external
entities are not expanded. Archive-level safety is enforced beforehand by
``backend.parsing.validate.check_docx_archive``.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator

import docx
from docx.section import _Footer, _Header
from docx.table import Table
from docx.text.paragraph import Paragraph

from backend.parsing.errors import UnreadableDocumentError, UnsupportedFileTypeError

MAX_TABLE_DEPTH = 5
CELL_SEPARATOR = " | "


def _table_lines(table: Table, depth: int) -> Iterator[str]:
    if depth > MAX_TABLE_DEPTH:
        return
    for row in table.rows:
        cells: list[str] = []
        nested: list[str] = []
        seen: set[int] = set()
        for cell in row.cells:
            if id(cell._tc) in seen:  # horizontally merged cells repeat
                continue
            seen.add(id(cell._tc))
            parts: list[str] = []
            for item in cell.iter_inner_content():
                if isinstance(item, Paragraph):
                    if item.text.strip():
                        parts.append(item.text.strip())
                else:
                    nested.extend(_table_lines(item, depth + 1))
            if parts:
                cells.append(" ".join(parts))
        if cells:
            yield CELL_SEPARATOR.join(cells)
        yield from nested


def _block_lines(blocks: Iterable[Paragraph | Table]) -> Iterator[str]:
    for block in blocks:
        if isinstance(block, Paragraph):
            yield block.text
        else:
            yield from _table_lines(block, depth=1)


def _defined(parts: Iterable[_Header | _Footer]) -> Iterator[_Header | _Footer]:
    # A part "linked to previous" has no definition of its own; reading it would
    # make python-docx create one, so it is skipped.
    for part in parts:
        if not part.is_linked_to_previous:
            yield part


def _unique_texts(parts: Iterable[_Header | _Footer]) -> list[str]:
    texts: list[str] = []
    for part in _defined(parts):
        text = "\n".join(line for line in _block_lines(part.iter_inner_content()) if line.strip())
        if text and text not in texts:
            texts.append(text)
    return texts


def extract_docx_text(data: bytes) -> str:
    """Extract headers, body (paragraphs and tables in order) and footers."""
    try:
        document = docx.Document(io.BytesIO(data))
    except ValueError as exc:
        # python-docx raises ValueError for non-Word packages, e.g. macro-enabled .docm.
        raise UnsupportedFileTypeError() from exc
    except MemoryError:
        raise
    except Exception as exc:  # untrusted input: any parser failure is "unreadable"
        raise UnreadableDocumentError() from exc

    try:
        headers = _unique_texts(
            part
            for section in document.sections
            for part in (section.header, section.first_page_header, section.even_page_header)
        )
        footers = _unique_texts(
            part
            for section in document.sections
            for part in (section.footer, section.first_page_footer, section.even_page_footer)
        )
        body = list(_block_lines(document.iter_inner_content()))
    except MemoryError:
        raise
    except Exception as exc:
        raise UnreadableDocumentError() from exc

    return "\n".join([*headers, *body, *footers])
