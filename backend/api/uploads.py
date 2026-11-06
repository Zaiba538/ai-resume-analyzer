"""Streaming, in-memory multipart/form-data reader for document uploads.

FastAPI's ``UploadFile`` spools any upload larger than 1 MiB to a temporary
file on disk (Starlette ``MultiPartParser.spool_max_size``). Resume uploads
must never touch disk, so uploads are parsed here with python-multipart
straight into memory, with hard limits enforced while streaming:

* exactly one file part, named ``file``, of at most ``max_file_bytes``;
* only allowlisted text fields, each at most ``max_field_bytes``;
* a bounded number of parts;
* the body must be complete, well-formed multipart;
* the whole body must arrive within ``UPLOAD_READ_TIMEOUT_SECONDS`` (M10, audit
  finding H1): a client that stalls or trickles data gets 408 ``upload_timeout``
  instead of holding the request open indefinitely. Nothing is truncated: a body
  either arrives in full in time or is rejected.

File names are used only to read the extension; they are never logged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from dataclasses import dataclass

from fastapi import Request
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, MultipartState, parse_options_header

from backend.core.errors import AppError
from backend.parsing.errors import (
    FileTooLargeError,
    InvalidFieldError,
    MalformedMultipartError,
    MissingFileError,
    MultipartRequiredError,
    MultipleFilesError,
    UnexpectedFieldError,
)

FILE_FIELD = "file"
MAX_PARTS = 8
MAX_FIELD_BYTES = 1024
_TRUE = frozenset({"true", "1", "on", "yes"})
_FALSE = frozenset({"false", "0", "off", "no", ""})

# Maximum time to receive a whole multipart body: generous for a 5 MiB upload on a
# slow connection (about 85 KB/s), short enough that stalled uploads do not pile up.
UPLOAD_READ_TIMEOUT_SECONDS = 60.0
UPLOAD_TIMEOUT_DESCRIPTION = "The upload was not received in time (`upload_timeout`)"


class UploadTimeoutError(AppError):
    status_code = 408
    code = "upload_timeout"
    message = "The upload was not received in time. Please try again."


@dataclass(frozen=True)
class UploadedForm:
    filename: str
    file_bytes: bytes
    fields: dict[str, str]


class _PartCollector:
    def __init__(
        self, *, max_file_bytes: int, max_field_bytes: int, allowed_fields: Collection[str]
    ) -> None:
        self.max_file_bytes = max_file_bytes
        self.max_field_bytes = max_field_bytes
        self.allowed_fields = frozenset(allowed_fields)
        self.parts = 0
        self.filename: str | None = None
        self.file_data = bytearray()
        self.fields: dict[str, str] = {}
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._headers: dict[bytes, bytes] = {}
        self._name: str | None = None
        self._is_file = False
        self._buffer = bytearray()

    # python-multipart callbacks -------------------------------------------------
    def on_part_begin(self) -> None:
        self.parts += 1
        if self.parts > MAX_PARTS:
            raise MalformedMultipartError()
        self._headers = {}
        self._buffer = bytearray()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field += data[start:end]

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value += data[start:end]

    def on_header_end(self) -> None:
        self._headers[bytes(self._header_field).lower()] = bytes(self._header_value)
        self._header_field = bytearray()
        self._header_value = bytearray()

    def on_headers_finished(self) -> None:
        disposition, options = parse_options_header(self._headers.get(b"content-disposition"))
        if disposition != b"form-data" or b"name" not in options:
            raise MalformedMultipartError()
        self._name = options[b"name"].decode("utf-8", "replace")
        self._is_file = b"filename" in options
        if self._name == FILE_FIELD:
            if not self._is_file:
                raise MissingFileError()
            if self.filename is not None:
                raise MultipleFilesError()
            self.filename = options[b"filename"].decode("utf-8", "replace")
        elif self._is_file:
            raise MultipleFilesError()
        elif self._name not in self.allowed_fields or self._name in self.fields:
            raise UnexpectedFieldError()

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._name == FILE_FIELD:
            if len(self.file_data) + len(chunk) > self.max_file_bytes:
                raise FileTooLargeError()
            self.file_data += chunk
        else:
            if len(self._buffer) + len(chunk) > self.max_field_bytes:
                raise InvalidFieldError()
            self._buffer += chunk

    def on_part_end(self) -> None:
        if self._name is not None and self._name != FILE_FIELD:
            try:
                self.fields[self._name] = self._buffer.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InvalidFieldError() from exc
        self._name = None


async def read_multipart_upload(
    request: Request,
    *,
    max_file_bytes: int,
    allowed_fields: Collection[str] = (),
    max_field_bytes: int = MAX_FIELD_BYTES,
) -> UploadedForm:
    """Read a single-file multipart upload entirely in memory."""
    content_type, options = parse_options_header(request.headers.get("content-type"))
    boundary = options.get(b"boundary")
    if content_type != b"multipart/form-data" or not boundary:
        raise MultipartRequiredError()

    collector = _PartCollector(
        max_file_bytes=max_file_bytes,
        max_field_bytes=max_field_bytes,
        allowed_fields=allowed_fields,
    )
    callbacks = {
        "on_part_begin": collector.on_part_begin,
        "on_part_data": collector.on_part_data,
        "on_part_end": collector.on_part_end,
        "on_header_field": collector.on_header_field,
        "on_header_value": collector.on_header_value,
        "on_header_end": collector.on_header_end,
        "on_headers_finished": collector.on_headers_finished,
    }
    try:
        parser = MultipartParser(boundary, callbacks)  # type: ignore[arg-type]
        # One deadline for the whole body, not per chunk: trickling data cannot
        # extend it. Read at call time so that tests can shorten it.
        async with asyncio.timeout(UPLOAD_READ_TIMEOUT_SECONDS):
            async for chunk in request.stream():
                if chunk:
                    parser.write(chunk)
        parser.finalize()
    except TimeoutError:
        raise UploadTimeoutError() from None
    except FormParserError as exc:
        raise MalformedMultipartError() from exc

    if parser.state != MultipartState.END:
        raise MalformedMultipartError()
    if collector.filename is None:
        raise MissingFileError()
    return UploadedForm(
        filename=collector.filename,
        file_bytes=bytes(collector.file_data),
        fields=dict(collector.fields),
    )


def parse_bool_field(value: str | None, *, default: bool = False) -> bool:
    """Parse an HTML-form style boolean; reject anything else."""
    if value is None:
        return default
    normalised = value.strip().lower()
    if normalised in _TRUE:
        return True
    if normalised in _FALSE:
        return False
    raise InvalidFieldError()
