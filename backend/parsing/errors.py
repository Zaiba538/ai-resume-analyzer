"""Upload and parsing errors.

Each error has a stable ``code`` and a fixed, user-safe message. Messages
never include file names, file content or library exception text.
"""

from __future__ import annotations

from backend.core.errors import AppError


class UploadError(AppError):
    """Base class for rejected uploads (deterministic validation layer)."""

    status_code = 422
    code = "invalid_upload"
    message = "The uploaded file could not be accepted."


class MalformedMultipartError(UploadError):
    status_code = 400
    code = "malformed_multipart"
    message = "The request body is not valid multipart/form-data."


class MultipartRequiredError(UploadError):
    status_code = 415
    code = "multipart_required"
    message = "Send the file as multipart/form-data in a field named 'file'."


class MissingFileError(UploadError):
    code = "missing_file"
    message = "No file was provided in the 'file' field."


class MultipleFilesError(UploadError):
    code = "multiple_files"
    message = "Upload exactly one file."


class UnexpectedFieldError(UploadError):
    code = "unexpected_field"
    message = "The request contains an unexpected form field."


class InvalidFieldError(UploadError):
    code = "invalid_field"
    message = "A form field has an invalid value."


class EmptyFileError(UploadError):
    code = "empty_file"
    message = "The uploaded file is empty."


class FileTooLargeError(UploadError):
    status_code = 413
    code = "file_too_large"
    message = "The uploaded file exceeds the maximum allowed size."


class UnsupportedFileTypeError(UploadError):
    status_code = 415
    code = "unsupported_file_type"
    message = "Only PDF and DOCX files are supported."


class FileContentMismatchError(UploadError):
    status_code = 415
    code = "file_content_mismatch"
    message = "The file content does not match its extension."


class UnsafeArchiveError(UploadError):
    code = "unsafe_archive"
    message = "The DOCX file structure is invalid or exceeds safety limits."


class EncryptedDocumentError(UploadError):
    code = "encrypted_document"
    message = "Password-protected or encrypted documents are not supported."


class UnreadableDocumentError(UploadError):
    code = "unreadable_document"
    message = "The document could not be read. It may be corrupt."


class TooManyPagesError(UploadError):
    code = "too_many_pages"
    message = "The document has more pages than the maximum allowed."


class DocumentTooLongError(UploadError):
    code = "document_too_long"
    message = "The document contains more text than the maximum allowed."


class NoExtractableTextError(UploadError):
    code = "no_extractable_text"
    message = (
        "No text could be extracted. Scanned or image-only documents are not supported; "
        "upload a text-based PDF or DOCX."
    )
