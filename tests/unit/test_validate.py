"""Deterministic upload validation: size, type and magic bytes, DOCX archive safety."""

from __future__ import annotations

import zipfile

import pytest

from backend.parsing.errors import (
    EmptyFileError,
    FileContentMismatchError,
    FileTooLargeError,
    UnsafeArchiveError,
    UnsupportedFileTypeError,
)
from backend.parsing.validate import (
    DOCX_MAX_ENTRIES,
    OLE_MAGIC,
    check_docx_archive,
    detect_file_type,
    file_extension,
    validate_upload,
)
from tests.fixtures.documents import docx_entries, make_docx, make_pdf, make_zip

PDF = make_pdf([["Hello"]])
DOCX = make_docx(["Hello"])


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("resume.pdf", ".pdf"),
        ("RESUME.PDF", ".pdf"),
        ("my.resume.v2.docx", ".docx"),
        ("C:\\Users\\x\\cv.docx", ".docx"),
        ("../../etc/cv.pdf", ".pdf"),
        ("noext", ""),
        ("resume.pdf.exe", ".exe"),
    ],
)
def test_file_extension(filename: str, expected: str) -> None:
    assert file_extension(filename) == expected


def test_valid_files_detected() -> None:
    assert validate_upload("cv.pdf", PDF, max_bytes=10**7) == "pdf"
    assert validate_upload("cv.docx", DOCX, max_bytes=10**7) == "docx"


def test_pdf_header_after_leading_junk_accepted() -> None:
    assert detect_file_type("cv.pdf", b"\n\n" + PDF) == "pdf"


def test_empty_rejected() -> None:
    with pytest.raises(EmptyFileError):
        validate_upload("cv.pdf", b"", max_bytes=100)


def test_size_limit_is_inclusive() -> None:
    validate_upload("cv.pdf", PDF, max_bytes=len(PDF))
    with pytest.raises(FileTooLargeError):
        validate_upload("cv.pdf", PDF, max_bytes=len(PDF) - 1)


@pytest.mark.parametrize("filename", ["cv.doc", "cv.txt", "cv.png", "cv", "cv.pdf.exe", "cv.docm"])
def test_unsupported_extensions(filename: str) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        detect_file_type(filename, PDF)


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("cv.pdf", DOCX),
        ("cv.docx", PDF),
        ("cv.pdf", b"<html><script>alert(1)</script></html>"),
        ("cv.docx", b"MZ\x90\x00 executable"),
        ("cv.pdf", b"x" * 2000 + b"%PDF-1.7"),
    ],
)
def test_content_must_match_extension(filename: str, data: bytes) -> None:
    with pytest.raises(FileContentMismatchError):
        detect_file_type(filename, data)


def test_ole_container_rejected_as_protected_or_legacy() -> None:
    with pytest.raises(UnsupportedFileTypeError) as info:
        detect_file_type("cv.docx", OLE_MAGIC + b"\x00" * 512)
    assert "Password-protected or legacy" in info.value.message


def test_minimal_valid_archive_accepted() -> None:
    check_docx_archive(DOCX)


def test_missing_required_parts_rejected() -> None:
    entries = docx_entries(DOCX)
    del entries["word/document.xml"]
    with pytest.raises(UnsafeArchiveError):
        check_docx_archive(make_zip(entries))


def test_not_a_zip_rejected() -> None:
    with pytest.raises(UnsafeArchiveError):
        check_docx_archive(b"PK\x03\x04" + b"garbage" * 50)


def test_too_many_entries_rejected() -> None:
    entries = docx_entries(DOCX)
    entries.update({f"word/media/x{i}.bin": b"x" for i in range(DOCX_MAX_ENTRIES)})
    with pytest.raises(UnsafeArchiveError):
        check_docx_archive(make_zip(entries))


def test_duplicate_entry_names_rejected() -> None:
    entries = docx_entries(DOCX)
    data = make_zip(entries)
    import io
    import warnings

    buffer = io.BytesIO(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns about the duplicate
        with zipfile.ZipFile(buffer, "a") as archive:
            archive.writestr("word/document.xml", b"<shadow/>")
    with pytest.raises(UnsafeArchiveError):
        check_docx_archive(buffer.getvalue())


def test_high_compression_ratio_rejected() -> None:
    entries = docx_entries(DOCX)
    entries["word/media/bomb.bin"] = b"\x00" * (8 * 1024 * 1024)  # ratio ~1000:1
    with pytest.raises(UnsafeArchiveError):
        check_docx_archive(make_zip(entries))


def test_total_uncompressed_size_limit(monkeypatch) -> None:
    import os

    from backend.parsing import validate

    monkeypatch.setattr(validate, "DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES", 256 * 1024)
    entries = docx_entries(DOCX)
    # Incompressible data passes the ratio check but exceeds the total.
    for i in range(3):
        entries[f"word/media/img{i}.bin"] = os.urandom(100 * 1024)
    with pytest.raises(UnsafeArchiveError):
        check_docx_archive(make_zip(entries, compression=zipfile.ZIP_STORED))


def test_per_entry_uncompressed_size_limit(monkeypatch) -> None:
    import os

    from backend.parsing import validate

    monkeypatch.setattr(validate, "DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES", 64 * 1024)
    entries = docx_entries(DOCX)
    entries["word/media/big.bin"] = os.urandom(65 * 1024)
    with pytest.raises(UnsafeArchiveError):
        check_docx_archive(make_zip(entries, compression=zipfile.ZIP_STORED))


def test_production_archive_limits() -> None:
    from backend.parsing import validate

    assert validate.DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES == 50 * 1024 * 1024
    assert validate.DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES == 25 * 1024 * 1024
    assert validate.DOCX_MAX_COMPRESSION_RATIO == 100
