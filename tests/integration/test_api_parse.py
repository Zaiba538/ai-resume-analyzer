"""POST /api/v1/parse contract (approved Rev. 3)."""

from __future__ import annotations

import pytest

from tests.fixtures.documents import (
    SYNTHETIC_RESUME_LINES,
    encrypt_pdf,
    make_docx,
    make_image_only_pdf,
    make_mixed_pdf,
    make_pdf,
)

PARSE = "/api/v1/parse"
PDF = make_pdf([list(SYNTHETIC_RESUME_LINES)])
DOCX = make_docx(SYNTHETIC_RESUME_LINES, tables=[[["Python", "SQL"]]], header="Jordan Sample")
ALL_SECTIONS = {
    "contact",
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
}


def _upload(client, filename: str, data: bytes, **fields: str):
    return client.post(PARSE, files={"file": (filename, data)}, data=fields)


def _assert_error(response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


def test_pdf_contract(start_app) -> None:
    response = _upload(start_app(), "resume.pdf", PDF)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"text_length", "page_count", "sections", "warnings"}
    assert body["page_count"] == 1
    assert body["text_length"] > 100
    assert set(body["sections"]) == ALL_SECTIONS and all(body["sections"].values())
    assert body["warnings"] == []


def test_docx_contract_page_count_null(start_app) -> None:
    body = _upload(start_app(), "resume.docx", DOCX).json()
    assert body["page_count"] is None
    assert body["sections"]["skills"] is True
    assert "preview" not in body


@pytest.mark.parametrize("flag", ["true", "1", "on", "TRUE"])
def test_preview_only_when_requested(start_app, flag: str) -> None:
    body = _upload(start_app(), "resume.pdf", PDF, include_preview=flag).json()
    assert body["preview"].startswith("Jordan Sample")
    assert len(body["preview"]) <= 1000


def test_preview_capped_at_1000_chars(start_app) -> None:
    long_pdf = make_pdf([[f"Line {i} " + "x" * 80 for i in range(45)]])
    body = _upload(start_app(), "cv.pdf", long_pdf, include_preview="true").json()
    assert body["text_length"] > 1000
    assert len(body["preview"]) == 1000


@pytest.mark.parametrize("flag", ["false", "0", "off"])
def test_preview_omitted_when_false(start_app, flag: str) -> None:
    assert "preview" not in _upload(start_app(), "cv.pdf", PDF, include_preview=flag).json()


def test_invalid_preview_flag_rejected(start_app) -> None:
    _assert_error(
        _upload(start_app(), "cv.pdf", PDF, include_preview="maybe"), 422, "invalid_field"
    )


def test_response_not_cached(start_app) -> None:
    response = _upload(start_app(), "cv.pdf", PDF)
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Robots-Tag"] == "noindex, nofollow"


def test_mixed_pdf_warns_about_textless_pages(start_app) -> None:
    body = _upload(start_app(), "cv.pdf", make_mixed_pdf()).json()
    assert body["page_count"] == 2
    assert [w["code"] for w in body["warnings"]] == ["pages_without_text"]


def test_no_sections_warning(start_app) -> None:
    body = _upload(start_app(), "cv.pdf", make_pdf([["just some words here"]])).json()
    assert "no_sections_detected" in [w["code"] for w in body["warnings"]]


@pytest.mark.parametrize(
    ("filename", "data", "status", "code"),
    [
        ("cv.pdf", b"", 422, "empty_file"),
        ("cv.txt", b"plain text", 415, "unsupported_file_type"),
        ("cv.pdf", DOCX, 415, "file_content_mismatch"),
        ("cv.docx", PDF, 415, "file_content_mismatch"),
        ("cv.pdf", encrypt_pdf(PDF), 422, "encrypted_document"),
        ("cv.pdf", make_image_only_pdf(), 422, "no_extractable_text"),
        ("cv.pdf", b"%PDF-1.7\n" + b"\x00" * 100, 422, "unreadable_document"),
        ("cv.docx", b"PK\x03\x04" + b"\x00" * 100, 422, "unsafe_archive"),
    ],
)
def test_rejections(start_app, filename: str, data: bytes, status: int, code: str) -> None:
    _assert_error(_upload(start_app(), filename, data), status, code)


def test_page_limit(start_app) -> None:
    data = make_pdf([["Skills"]] * 3)
    assert _upload(start_app(max_pdf_pages=3), "cv.pdf", data).status_code == 200
    _assert_error(_upload(start_app(max_pdf_pages=2), "cv.pdf", data), 422, "too_many_pages")


def test_character_limit(start_app) -> None:
    data = make_pdf([["x" * 90] * 15])  # ~1350 characters
    _assert_error(
        _upload(start_app(max_extracted_chars=1000), "cv.pdf", data), 422, "document_too_long"
    )


def test_file_size_limit(start_app) -> None:
    client = start_app(max_upload_bytes=len(PDF))
    assert _upload(client, "cv.pdf", PDF).status_code == 200
    _assert_error(_upload(client, "cv.pdf", PDF + b" "), 413, "file_too_large")


def test_missing_file_field(start_app) -> None:
    response = start_app().post(PARSE, files={"include_preview": (None, "true")})
    _assert_error(response, 422, "missing_file")


def test_file_sent_as_text_field_rejected(start_app) -> None:
    response = start_app().post(PARSE, files={"file": (None, "not a file")})
    _assert_error(response, 422, "missing_file")


def test_two_files_rejected(start_app) -> None:
    response = start_app().post(PARSE, files=[("file", ("a.pdf", PDF)), ("file", ("b.pdf", PDF))])
    _assert_error(response, 422, "multiple_files")


def test_unexpected_field_rejected(start_app) -> None:
    response = start_app().post(
        PARSE, files={"file": ("cv.pdf", PDF)}, data={"job_description": "x"}
    )
    _assert_error(response, 422, "unexpected_field")


def test_non_multipart_rejected(start_app) -> None:
    client = start_app()
    _assert_error(client.post(PARSE, json={"file": "x"}), 415, "multipart_required")
    _assert_error(
        client.post(PARSE, content=PDF, headers={"Content-Type": "application/pdf"}),
        415,
        "multipart_required",
    )


def test_truncated_multipart_rejected(start_app) -> None:
    boundary = "XyZboundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="cv.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
    ).encode() + PDF[:200]
    response = start_app().post(
        PARSE,
        content=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    _assert_error(response, 400, "malformed_multipart")


def test_client_content_type_is_ignored(start_app) -> None:
    response = start_app().post(PARSE, files={"file": ("cv.pdf", PDF, "image/png")})
    assert response.status_code == 200


def test_parse_documented_in_openapi(start_app) -> None:
    schema = start_app(api_docs_enabled=True).get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/parse"]["post"]
    assert "multipart/form-data" in operation["requestBody"]["content"]
    assert {"200", "413", "415", "422"} <= set(operation["responses"])


# --- English-only check (language validation) -------------------------------------------------


def _language_samples():
    from tests.unit.test_language import (
        HINDI_RESUME_LINES,
        MARATHI_RESUME_LINES,
        make_devanagari_pdf,
    )

    return {
        "hindi_docx": ("cv.docx", make_docx(HINDI_RESUME_LINES)),
        "marathi_pdf": ("cv.pdf", make_devanagari_pdf(MARATHI_RESUME_LINES)),
        "hindi_pdf": ("cv.pdf", make_devanagari_pdf(HINDI_RESUME_LINES)),
        "marathi_docx": ("cv.docx", make_docx(MARATHI_RESUME_LINES)),
    }


@pytest.mark.parametrize("sample", ["hindi_docx", "marathi_pdf", "hindi_pdf", "marathi_docx"])
def test_non_english_resume_is_rejected_with_unsupported_language(start_app, sample) -> None:
    from tests.unit.test_language import RESUME_MESSAGE

    filename, data = _language_samples()[sample]
    response = _upload(start_app(), filename, data, include_preview="true")
    _assert_error(response, 422, "unsupported_language")
    error = response.json()["error"]
    assert error["message"] == RESUME_MESSAGE
    assert error["details"] == [
        {"loc": ["resume"], "type": "unsupported_language", "msg": RESUME_MESSAGE}
    ]
    assert "पायथन" not in response.text  # nothing of the document is echoed


def test_english_resume_with_devanagari_is_accepted(start_app) -> None:
    from tests.unit.test_language import DEVANAGARI_NAME_ADDRESS_LINES, LANGUAGES_LINE

    lines = [*DEVANAGARI_NAME_ADDRESS_LINES, LANGUAGES_LINE]
    body = _upload(start_app(), "cv.docx", make_docx(lines), include_preview="true").json()
    assert body["preview"] == "\n".join(lines)
    assert body["sections"]["skills"] and body["sections"]["education"]


@pytest.mark.parametrize("path", ["/api/v1/parse", "/api/v1/match", "/api/v1/analyze"])
def test_openapi_documents_the_language_error(start_app, path: str) -> None:
    responses = start_app().app.openapi()["paths"][path]["post"]["responses"]
    assert "unsupported language" in responses["422"]["description"]
