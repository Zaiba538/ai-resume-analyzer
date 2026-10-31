"""Hostile uploads: oversize bodies, zip bombs, XML entity attacks, disk spooling.

Every case must end in a typed 4xx envelope, never a 500, and never write
upload content to disk.
"""

from __future__ import annotations

import io
import tempfile
import zipfile

import pytest

from backend.parsing.validate import check_docx_archive
from tests.fixtures.documents import (
    SYNTHETIC_RESUME_LINES,
    docx_entries,
    make_docx,
    make_pdf,
    make_zip,
)

PARSE = "/api/v1/parse"
DOCX = make_docx(SYNTHETIC_RESUME_LINES)


def _upload(client, filename: str, data: bytes):
    return client.post(PARSE, files={"file": (filename, data)})


def test_declared_oversize_body_rejected_before_reading(start_app) -> None:
    client = start_app(max_upload_bytes=1024)
    response = client.post(
        PARSE,
        content=b"x" * 10,
        headers={"Content-Type": "multipart/form-data; boundary=b", "Content-Length": "999999999"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"
    assert "Content-Security-Policy" in response.headers


def test_streamed_oversize_body_rejected(start_app) -> None:
    client = start_app(max_upload_bytes=1024)

    def chunks():
        for _ in range(200):
            yield b"y" * 1024

    response = client.post(
        PARSE, content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=b"}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_body_limit_also_protects_json_routes(start_app) -> None:
    from fastapi import FastAPI

    def routes(app: FastAPI) -> None:
        @app.post("/api/v1/_test/json")
        async def echo(payload: dict) -> dict:
            return {"n": len(payload)}

    client = start_app(max_upload_bytes=1024, extra_routes=routes)

    def chunks():
        yield b'{"a": "'
        for _ in range(200):
            yield b"z" * 1024
        yield b'"}'

    response = client.post(
        "/api/v1/_test/json", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_zip_bomb_rejected(start_app) -> None:
    entries = docx_entries(DOCX)
    entries["word/media/bomb.bin"] = b"\x00" * (20 * 1024 * 1024)
    bomb = make_zip(entries)
    assert len(bomb) < 1024 * 1024  # small on the wire
    response = _upload(start_app(), "cv.docx", bomb)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsafe_archive"


_CENTRAL_DIR_SIG = b"PK\x01\x02"


def _understate_declared_size(archive: bytes, entry: str, declared: int) -> bytes:
    """Rewrite an entry's central-directory uncompressed size (offset 24)."""
    data = bytearray(archive)
    index = data.find(_CENTRAL_DIR_SIG)
    while index != -1:
        name_len = int.from_bytes(data[index + 28 : index + 30], "little")
        if bytes(data[index + 46 : index + 46 + name_len]) == entry.encode():
            data[index + 24 : index + 28] = declared.to_bytes(4, "little")
            return bytes(data)
        index = data.find(_CENTRAL_DIR_SIG, index + 4)
    raise AssertionError(f"entry not found: {entry}")


def test_lying_zip_header_cannot_inflate_beyond_declared_size() -> None:
    """A central directory that understates the payload passes the size checks,
    but zipfile stops decompressing at the declared size and fails the CRC,
    so the declared sizes really do bound memory."""
    entries = docx_entries(DOCX)
    entries["word/media/payload.bin"] = b"\x00" * (4 * 1024 * 1024)
    data = _understate_declared_size(make_zip(entries), "word/media/payload.bin", 1024)

    check_docx_archive(data)  # declared sizes look harmless
    with zipfile.ZipFile(io.BytesIO(data)) as archive, pytest.raises(zipfile.BadZipFile):
        archive.read("word/media/payload.bin")


def test_lying_zip_header_upload_is_rejected_not_500(start_app) -> None:
    entries = docx_entries(DOCX)
    entries["word/document.xml"] = entries["word/document.xml"] + b"<!--" + b"p" * 200_000 + b"-->"
    data = _understate_declared_size(make_zip(entries), "word/document.xml", 100)
    response = _upload(start_app(), "cv.docx", data)
    assert response.status_code == 422
    assert response.json()["error"]["code"] in {"unreadable_document", "unsafe_archive"}


def test_xml_external_entity_not_resolved(start_app, tmp_path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("XXE-CANARY-41a7", encoding="utf-8")
    entries = docx_entries(DOCX)
    entries["word/document.xml"] = (
        f'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY x SYSTEM "file://{secret.as_posix()}">]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Skills &x;</w:t></w:r></w:p></w:body></w:document>"
    ).encode()
    response = start_app().post(
        PARSE, files={"file": ("cv.docx", make_zip(entries))}, data={"include_preview": "true"}
    )
    assert response.status_code in {200, 422}
    assert "XXE-CANARY-41a7" not in response.text


def test_billion_laughs_bounded(start_app) -> None:
    lol = '<!ENTITY a "lollollollollollollollollollol">' + "".join(
        f'<!ENTITY {chr(98 + i)} "{("&" + chr(97 + i) + ";") * 10}">' for i in range(9)
    )
    entries = docx_entries(DOCX)
    entries["word/document.xml"] = (
        f'<?xml version="1.0"?><!DOCTYPE d [{lol}]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&j;</w:t></w:r></w:p></w:body></w:document>"
    ).encode()
    response = _upload(start_app(), "cv.docx", make_zip(entries))
    assert response.status_code in {200, 422}
    assert len(response.content) < 100_000


@pytest.mark.parametrize(
    "filename",
    [
        "../../../../etc/passwd.pdf",
        "C:\\Windows\\System32\\cv.pdf",
        "cv.pdf\x00.exe",
        "<script>.pdf",
    ],
)
def test_hostile_filenames_never_echoed(start_app, filename: str) -> None:
    response = _upload(start_app(), filename, make_pdf([["Skills"]]))
    assert response.status_code in {200, 415}
    for fragment in ("passwd", "System32", "<script>", ".exe"):
        assert fragment not in response.text


def test_upload_never_spooled_to_disk(start_app, monkeypatch) -> None:
    """A 2 MiB upload (above Starlette's 1 MiB spool threshold) stays in memory."""
    created: list[str] = []
    for factory in ("SpooledTemporaryFile", "NamedTemporaryFile", "TemporaryFile", "mkstemp"):
        original = getattr(tempfile, factory)

        def spy(*args, _original=original, _name=factory, **kwargs):
            created.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(tempfile, factory, spy)
    import starlette.formparsers

    monkeypatch.setattr(
        starlette.formparsers, "SpooledTemporaryFile", tempfile.SpooledTemporaryFile
    )

    big_pdf = make_pdf([["Skills", "Python"]]) + b"\n%" + b"p" * (2 * 1024 * 1024)
    response = _upload(start_app(), "cv.pdf", big_pdf)
    assert response.status_code == 200, response.text
    assert created == []
