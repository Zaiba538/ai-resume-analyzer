"""Deterministic upload validation.

Runs before any parser touches the bytes:

* size: non-empty and within ``max_upload_bytes``;
* type: the extension (``.pdf`` / ``.docx``) must agree with the magic bytes;
* DOCX archive safety: entry count, total and per-entry uncompressed size,
  compression ratio (zip bombs), encrypted or duplicate entries, and the
  presence of the parts every Word document has.

The client-supplied Content-Type is ignored: it is not trustworthy.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from backend.parsing.errors import (
    EmptyFileError,
    FileContentMismatchError,
    FileTooLargeError,
    UnsafeArchiveError,
    UnsupportedFileTypeError,
)

FileType = Literal["pdf", "docx"]

EXTENSIONS: dict[str, FileType] = {".pdf": "pdf", ".docx": "docx"}

PDF_MAGIC = b"%PDF-"
# The PDF header may follow a few bytes of junk; readers search the first 1 KiB.
PDF_HEADER_SEARCH_BYTES = 1024
ZIP_MAGIC = b"PK\x03\x04"
# Password-protected Office files and legacy .doc files are OLE compound files.
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

DOCX_REQUIRED_ENTRIES = frozenset({"[Content_Types].xml", "word/document.xml"})
DOCX_MAX_ENTRIES = 1000
DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES = 25 * 1024 * 1024
DOCX_MAX_COMPRESSION_RATIO = 100
# Tiny, highly compressible XML parts legitimately exceed the ratio.
DOCX_RATIO_CHECK_MIN_BYTES = 1024 * 1024


def file_extension(filename: str) -> str:
    """Lower-case extension of the final path component, for either path style."""
    name = PureWindowsPath(PurePosixPath(filename).name).name
    return PurePosixPath(name).suffix.lower()


def check_size(data: bytes, max_bytes: int) -> None:
    if not data:
        raise EmptyFileError()
    if len(data) > max_bytes:
        raise FileTooLargeError()


def detect_file_type(filename: str, data: bytes) -> FileType:
    """Resolve the file type from the extension and confirm it with magic bytes."""
    file_type = EXTENSIONS.get(file_extension(filename))
    if file_type is None:
        raise UnsupportedFileTypeError()

    if file_type == "pdf":
        if PDF_MAGIC not in data[:PDF_HEADER_SEARCH_BYTES]:
            raise FileContentMismatchError()
    elif data.startswith(OLE_MAGIC):
        raise UnsupportedFileTypeError(
            "Password-protected or legacy .doc files are not supported. "
            "Save the document as an unprotected .docx file."
        )
    elif not data.startswith(ZIP_MAGIC):
        raise FileContentMismatchError()
    return file_type


def check_docx_archive(data: bytes) -> None:
    """Reject zip structures that are malformed, encrypted or decompression bombs."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile, ValueError, EOFError) as exc:
        raise UnsafeArchiveError() from exc

    if not entries or len(entries) > DOCX_MAX_ENTRIES:
        raise UnsafeArchiveError()

    names = [entry.filename for entry in entries]
    if len(set(names)) != len(names):
        raise UnsafeArchiveError()
    if not DOCX_REQUIRED_ENTRIES.issubset(names):
        raise UnsafeArchiveError()

    total = 0
    for entry in entries:
        if entry.flag_bits & 0x1:  # encrypted entry
            raise UnsafeArchiveError()
        if entry.file_size > DOCX_MAX_ENTRY_UNCOMPRESSED_BYTES:
            raise UnsafeArchiveError()
        if (
            entry.file_size >= DOCX_RATIO_CHECK_MIN_BYTES
            and entry.file_size > DOCX_MAX_COMPRESSION_RATIO * max(entry.compress_size, 1)
        ):
            raise UnsafeArchiveError()
        total += entry.file_size
        if total > DOCX_MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise UnsafeArchiveError()


def validate_upload(filename: str, data: bytes, *, max_bytes: int) -> FileType:
    """Run every deterministic check; return the confirmed file type."""
    check_size(data, max_bytes)
    file_type = detect_file_type(filename, data)
    if file_type == "docx":
        check_docx_archive(data)
    return file_type
