"""M9 Phase 1: interview error envelopes and request validation through the app.

The interview routes arrive in Phase 3. These tests mount test-only routes that
raise the typed errors and accept the request schemas, to check the HTTP
contract: status, code, fixed message, envelope shape, security headers, and
no submitted text in responses or logs.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI

from backend.schemas.interview_api import AnswerEvaluationRequest, InterviewQuestionsRequest
from backend.services.errors import (
    AnswerTooLongError,
    EmptyAnswerError,
    EvaluationTimeoutError,
    InterviewTimeoutError,
    QuestionNotFoundError,
)

CANARY = "CANARYm9contract73"
REPORT_ID = str(uuid.uuid4())
ERRORS = {
    "question-not-found": (QuestionNotFoundError, 404, "question_not_found"),
    "empty-answer": (EmptyAnswerError, 422, "empty_answer"),
    "answer-too-long": (AnswerTooLongError, 422, "answer_too_long"),
    "interview-timeout": (InterviewTimeoutError, 504, "interview_timeout"),
    "evaluation-timeout": (EvaluationTimeoutError, 504, "evaluation_timeout"),
}


def _routes(app: FastAPI) -> None:
    for name, (error, _, _) in ERRORS.items():

        async def raise_error(error=error) -> None:
            raise error()

        app.add_api_route(f"/api/v1/_test/interview/{name}", raise_error, methods=["POST"])

    @app.post("/api/v1/_test/interview/evaluate")
    async def evaluate(body: AnswerEvaluationRequest) -> dict:
        return {"level": body.level, "question_id": body.target_question_id}

    @app.post("/api/v1/_test/interview/questions")
    async def questions(body: InterviewQuestionsRequest) -> dict:
        return {"levels": list(body.requested_levels), "count": body.count_per_level}


def _envelope(response, status: int, code: str) -> dict:
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert error["code"] == code
    assert error["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["Cache-Control"] == "no-store"
    assert "noindex" in response.headers["X-Robots-Tag"]
    return error


@pytest.mark.parametrize("name", list(ERRORS))
def test_typed_errors_use_the_envelope(start_app, captured_logs, name: str) -> None:
    error, status, code = ERRORS[name]
    client = start_app(extra_routes=_routes)
    response = client.post(f"/api/v1/_test/interview/{name}", json={"answer": CANARY})
    body = _envelope(response, status, code)
    assert set(body) == {"code", "message", "request_id"}
    assert body["message"] == error.message
    assert CANARY not in response.text and CANARY not in captured_logs.text
    (logged,) = captured_logs.events("app_error")
    assert (logged["error_code"], logged["status"]) == (code, status)


def test_valid_requests_reach_the_route(start_app) -> None:
    client = start_app(extra_routes=_routes)
    stored = client.post(
        "/api/v1/_test/interview/evaluate",
        json={"report_id": REPORT_ID, "question_id": "advanced-3", "answer": "x"},
    )
    assert stored.json() == {"level": "advanced", "question_id": "advanced-3"}
    supplied = client.post(
        "/api/v1/_test/interview/evaluate",
        json={
            "report_id": REPORT_ID,
            "answer": "x",
            "question": {
                "id": "beginner-2",
                "question": "What is a list?",
                "topic": "Python",
                "rationale": "Listed skill.",
            },
        },
    )
    assert supplied.json() == {"level": "beginner", "question_id": "beginner-2"}
    questions = client.post(
        "/api/v1/_test/interview/questions",
        json={"report_id": REPORT_ID, "levels": ["advanced", "beginner"], "count_per_level": 2},
    )
    assert questions.json() == {"levels": ["beginner", "advanced"], "count": 2}


@pytest.mark.parametrize(
    "body",
    [
        {"report_id": REPORT_ID, "question_id": CANARY, "answer": CANARY},
        {"report_id": REPORT_ID, "question_id": "beginner-1", "level": CANARY, "answer": CANARY},
        {"report_id": REPORT_ID, "question_id": "beginner-1", "model": CANARY, "answer": "x"},
        {"report_id": REPORT_ID, "answer": CANARY},  # no question source
        {
            "report_id": REPORT_ID,
            "question_id": "beginner-1",
            "answer": CANARY,
            "question": {"id": "beginner-1", "question": CANARY, "topic": CANARY, "rationale": ""},
        },
        {
            "report_id": REPORT_ID,
            "answer": CANARY,
            "question": {"id": CANARY, "question": CANARY * 50, "topic": "", "rationale": CANARY},
        },
        {"report_id": REPORT_ID, "question_id": "beginner-1", "answer": CANARY * 3000},
        {"report_id": 12345, "question_id": "beginner-1", "answer": [CANARY]},
    ],
)
def test_invalid_evaluate_bodies_are_422_without_echo(start_app, captured_logs, body) -> None:
    client = start_app(extra_routes=_routes)
    response = client.post("/api/v1/_test/interview/evaluate", json=body)
    error = _envelope(response, 422, "validation_error")
    assert error["message"] == "Request validation failed."
    for detail in error["details"]:
        assert set(detail) == {"loc", "type", "msg"}
    assert CANARY not in response.text
    assert CANARY not in captured_logs.text


@pytest.mark.parametrize(
    "body",
    [
        {"report_id": REPORT_ID, "levels": [CANARY]},
        {"report_id": REPORT_ID, "levels": ["beginner", "beginner"]},
        {"report_id": REPORT_ID, "levels": []},
        {"report_id": REPORT_ID, "count_per_level": 6},
        {"report_id": REPORT_ID, "count_per_level": "3"},
        {"report_id": REPORT_ID, "count_per_level": True},
        {"report_id": REPORT_ID, "model": CANARY},
        {"report_id": None},
    ],
)
def test_invalid_questions_bodies_are_422_without_echo(start_app, body) -> None:
    client = start_app(extra_routes=_routes)
    response = client.post("/api/v1/_test/interview/questions", json=body)
    _envelope(response, 422, "validation_error")
    assert CANARY not in response.text


def test_malformed_json_is_422_without_echo(start_app) -> None:
    client = start_app(extra_routes=_routes)
    response = client.post(
        "/api/v1/_test/interview/evaluate",
        content=b'{"report_id": "x", "answer": "' + CANARY.encode() + b'",',
        headers={"Content-Type": "application/json"},
    )
    _envelope(response, 422, "validation_error")
    assert CANARY not in response.text
