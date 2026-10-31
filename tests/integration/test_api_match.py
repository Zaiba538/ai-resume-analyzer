"""POST /api/v1/match contract (approved Rev. 3 MatchResult)."""

from __future__ import annotations

import json

import pytest

from tests.conftest import FAKE_API_KEY
from tests.fixtures.documents import SYNTHETIC_RESUME_LINES, make_docx, make_pdf
from tests.fixtures.embeddings import CountingFactory

MATCH = "/api/v1/match"
RESUME = "\n".join(SYNTHETIC_RESUME_LINES)
JOB = "Data Engineer (fictional). Required: Python, SQL, Kubernetes, Apache Airflow."
ATS_DISABLED = {
    "status": "not_calibrated",
    "score": None,
    "calibration_version": None,
    "evaluation_summary": None,
}


def _json(client, resume=RESUME, job=JOB, **extra):
    return client.post(MATCH, json={"resume_text": resume, "job_description": job, **extra})


def _upload(client, filename="cv.pdf", data=None, job=JOB):
    fields = {} if job is None else {"job_description": job}
    data = make_pdf([list(SYNTHETIC_RESUME_LINES)]) if data is None else data
    return client.post(MATCH, files={"file": (filename, data)}, data=fields)


def _assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    assert error["request_id"] == response.headers["X-Request-ID"]
    return error


def _assert_contract(body: dict) -> None:
    assert set(body) == {"metrics", "baseline", "ats_score"}
    metrics = body["metrics"]
    assert set(metrics) == {"semantic_similarity", "skill_coverage", "section_completeness"}
    assert set(metrics["semantic_similarity"]) == {"value", "calibrated"}
    assert metrics["semantic_similarity"]["calibrated"] is False
    assert set(metrics["skill_coverage"]) == {"value", "matched", "missing", "jd_skills_total"}
    assert set(metrics["section_completeness"]) == {"value", "present", "absent"}
    assert set(body["baseline"]) == {"tfidf_keyword_match"}
    for value in (
        metrics["semantic_similarity"]["value"],
        metrics["skill_coverage"]["value"],
        metrics["section_completeness"]["value"],
        body["baseline"]["tfidf_keyword_match"],
    ):
        assert 0.0 <= value <= 100.0
    assert body["ats_score"] == ATS_DISABLED


def test_json_request_contract(start_app) -> None:
    response = _json(start_app())
    assert response.status_code == 200
    body = response.json()
    _assert_contract(body)
    coverage = body["metrics"]["skill_coverage"]
    assert coverage == {
        "value": 50.0,
        "matched": ["Python", "SQL"],
        "missing": ["Kubernetes", "Apache Airflow"],
        "jd_skills_total": 4,
    }
    assert body["metrics"]["section_completeness"]["value"] == 100.0
    assert body["metrics"]["semantic_similarity"]["value"] > 0
    assert body["baseline"]["tfidf_keyword_match"] > 0


def test_ats_nulls_are_serialised_not_omitted(start_app) -> None:
    raw = json.loads(_json(start_app()).text)
    assert "score" in raw["ats_score"] and raw["ats_score"]["score"] is None


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("cv.pdf", make_pdf([list(SYNTHETIC_RESUME_LINES)])),
        ("cv.docx", make_docx(SYNTHETIC_RESUME_LINES)),
    ],
)
def test_multipart_request_contract(start_app, filename: str, data: bytes) -> None:
    response = _upload(start_app(), filename, data)
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_contract(body)
    assert body["metrics"]["skill_coverage"]["matched"] == ["Python", "SQL"]


def test_json_and_multipart_agree_on_same_text(start_app) -> None:
    client = start_app()
    docx_body = _upload(client, "cv.docx", make_docx(SYNTHETIC_RESUME_LINES)).json()
    json_body = _json(client).json()
    assert docx_body == json_body


def test_response_headers(start_app) -> None:
    response = _json(start_app())
    assert response.headers["X-Request-ID"]
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Robots-Tag"] == "noindex, nofollow"
    assert "Content-Security-Policy" in response.headers


def test_model_loaded_once_and_reused_across_requests(start_app) -> None:
    factory = CountingFactory()
    client = start_app(embedder_factory=factory)
    for _ in range(4):
        assert _json(client).status_code == 200
    assert factory.calls == 1
    assert factory.embedder.encode_calls == 8  # 2 documents x 4 requests


def test_no_llm_call_required(start_app, fake_groq) -> None:
    client = start_app(groq_api_key=FAKE_API_KEY)
    probe_calls = len(fake_groq.calls)  # start-up metadata probe only
    assert _json(client).status_code == 200
    assert len(fake_groq.calls) == probe_calls
    assert client.app.state.llm_clients._chat_models == {}


def test_model_unavailable_returns_503_and_not_ready(start_app) -> None:
    client = start_app(embedder_factory=CountingFactory(fail=True))
    _assert_error(_json(client), 503, "embedding_model_unavailable")
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["checks"]["embedding_model_loaded"] is False


# --- input validation ---------------------------------------------------------------


def test_missing_job_description_multipart(start_app) -> None:
    _assert_error(_upload(start_app(), job=None), 422, "missing_job_description")


@pytest.mark.parametrize("job", ["", "   \n\t", chr(0x200B) * 3, chr(0x202E)])
def test_empty_job_description(start_app, job: str) -> None:
    _assert_error(_json(start_app(), job=job), 422, "empty_job_description")
    _assert_error(_upload(start_app(), job=job), 422, "empty_job_description")


def test_job_description_too_long(start_app) -> None:
    client = start_app(max_job_description_chars=500)
    assert _json(client, job="x " * 250).status_code == 200  # exactly 500 chars
    _assert_error(_json(client, job="x " * 251), 422, "job_description_too_long")


@pytest.mark.parametrize("resume", ["", "   ", chr(0x200B) + chr(0x2060)])
def test_empty_resume_text(start_app, resume: str) -> None:
    _assert_error(_json(start_app(), resume=resume), 422, "empty_resume_text")


def test_resume_text_too_long(start_app) -> None:
    client = start_app(max_extracted_chars=1000)
    _assert_error(_json(client, resume="y" * 1001), 422, "document_too_long")


@pytest.mark.parametrize(
    "payload",
    [
        {"resume_text": RESUME},
        {"job_description": JOB},
        {"resume_text": RESUME, "job_description": JOB, "extra": 1},
        {"resume_text": 123, "job_description": JOB},
        {"resume_text": [RESUME], "job_description": JOB},
        ["not", "an", "object"],
    ],
)
def test_invalid_json_shapes(start_app, payload) -> None:
    error = _assert_error(start_app().post(MATCH, json=payload), 422, "validation_error")
    for detail in error["details"]:
        assert set(detail) == {"loc", "type", "msg"}


def test_malformed_json(start_app) -> None:
    response = start_app().post(
        MATCH, content=b'{"resume_text": "x", ', headers={"Content-Type": "application/json"}
    )
    _assert_error(response, 422, "validation_error")


@pytest.mark.parametrize(
    "content_type", ["text/plain", "application/xml", "application/x-www-form-urlencoded", ""]
)
def test_unsupported_media_type(start_app, content_type: str) -> None:
    response = start_app().post(MATCH, content=b"resume", headers={"Content-Type": content_type})
    _assert_error(response, 415, "unsupported_media_type")


def test_unsupported_file_type(start_app) -> None:
    _assert_error(_upload(start_app(), "cv.txt", b"plain text"), 415, "unsupported_file_type")


def test_unreadable_document(start_app) -> None:
    _assert_error(
        _upload(start_app(), "cv.pdf", b"%PDF-1.7\n" + b"\x00" * 64), 422, "unreadable_document"
    )


def test_oversized_upload(start_app) -> None:
    pdf = make_pdf([list(SYNTHETIC_RESUME_LINES)])
    client = start_app(max_upload_bytes=len(pdf))
    assert _upload(client, data=pdf).status_code == 200
    _assert_error(_upload(client, data=pdf + b" "), 413, "file_too_large")


def test_page_limit_enforced(start_app) -> None:
    data = make_pdf([list(SYNTHETIC_RESUME_LINES)] * 3)
    _assert_error(_upload(start_app(max_pdf_pages=2), data=data), 422, "too_many_pages")


def test_unexpected_multipart_field(start_app) -> None:
    response = start_app().post(
        MATCH,
        files={"file": ("cv.pdf", make_pdf([["Skills"]]))},
        data={"job_description": JOB, "include_preview": "true"},
    )
    _assert_error(response, 422, "unexpected_field")


def test_documented_in_openapi(start_app) -> None:
    operation = start_app(api_docs_enabled=True).get("/openapi.json").json()["paths"][MATCH]["post"]
    assert {"multipart/form-data", "application/json"} <= set(operation["requestBody"]["content"])
    assert {"200", "415", "422", "503"} <= set(operation["responses"])
