"""/match privacy: no resume or job-description text in logs, errors or on disk."""

from __future__ import annotations

import tempfile

import pytest

from tests.fixtures.documents import make_docx

MATCH = "/api/v1/match"
RESUME_CANARY = "CANARYresume7c1d"
JOB_CANARY = "CANARYjob2b9e"
RESUME = f"Skills\nPython, SQL\n{RESUME_CANARY} built pipelines\nEducation\nB.Sc."
JOB = f"Required: Python, Kubernetes. {JOB_CANARY} team."


def test_no_canary_in_logs_json_or_multipart(start_app, captured_logs) -> None:
    client = start_app()
    assert (
        client.post(MATCH, json={"resume_text": RESUME, "job_description": JOB}).status_code == 200
    )
    upload = client.post(
        MATCH,
        files={"file": (f"{RESUME_CANARY}.docx", make_docx(RESUME.split("\n")))},
        data={"job_description": JOB},
    )
    assert upload.status_code == 200
    records = captured_logs.events("match_completed")
    assert {r["file_type"] for r in records} == {"text", "docx"}
    for record in records:
        assert set(record) <= {
            "ts",
            "level",
            "logger",
            "event",
            "request_id",
            "file_type",
            "char_count",
            "model",
        }
    for canary in (RESUME_CANARY, JOB_CANARY):
        assert canary not in captured_logs.text


@pytest.mark.parametrize(
    ("payload", "settings"),
    [
        ({"resume_text": RESUME, "job_description": JOB * 50}, {"max_job_description_chars": 500}),
        ({"resume_text": RESUME, "job_description": JOB, "note": JOB_CANARY}, {}),
        (
            {"resume_text": RESUME_CANARY * 70, "job_description": JOB},
            {"max_extracted_chars": 1000},
        ),
        ({"resume_text": 42, "job_description": JOB_CANARY}, {}),
    ],
)
def test_errors_never_echo_input(start_app, captured_logs, payload, settings) -> None:
    response = start_app(**settings).post(MATCH, json=payload)
    assert response.status_code == 422
    for canary in (RESUME_CANARY, JOB_CANARY):
        assert canary not in response.text
        assert canary not in captured_logs.text


def test_malformed_json_never_echoed(start_app) -> None:
    response = start_app().post(
        MATCH,
        content=f'{{"resume_text": "{RESUME_CANARY}", '.encode(),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert RESUME_CANARY not in response.text


def test_nothing_written_to_disk(start_app, monkeypatch, tmp_path) -> None:
    created: list[str] = []
    for factory in ("SpooledTemporaryFile", "NamedTemporaryFile", "TemporaryFile", "mkstemp"):
        original = getattr(tempfile, factory)

        def spy(*args, _original=original, _name=factory, **kwargs):
            created.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(tempfile, factory, spy)
    monkeypatch.chdir(tmp_path)
    client = start_app()
    client.post(MATCH, json={"resume_text": RESUME, "job_description": JOB})
    client.post(
        MATCH,
        files={"file": ("cv.docx", make_docx(RESUME.split("\n")))},
        data={"job_description": JOB},
    )
    assert created == []
    assert list(tmp_path.iterdir()) == []


def test_no_text_kept_on_application_state(start_app) -> None:
    client = start_app()
    client.post(MATCH, json={"resume_text": RESUME, "job_description": JOB})
    state = repr(vars(client.app.state))
    assert RESUME_CANARY not in state and JOB_CANARY not in state
