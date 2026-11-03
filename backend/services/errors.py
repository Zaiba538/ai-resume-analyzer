"""Interview API errors (M9, approved decisions Q7 and Q8).

Each error has a stable ``code`` and a fixed, user-safe message. Messages never
include the answer, the question, the report ID or any other request content.
Unknown, malformed and expired report IDs keep the existing 404
``report_not_found``.
"""

from __future__ import annotations

from backend.core.errors import AppError


class ReportNotFoundError(AppError):
    """Same contract as ``backend.api.routes.report.ReportNotFoundError`` (status,
    code and message; a test keeps them identical). Defined here because services
    must not import route modules, which import the API dependencies."""

    status_code = 404
    code = "report_not_found"
    message = "The report was not found or has expired."


class QuestionNotFoundError(AppError):
    status_code = 404
    code = "question_not_found"
    message = "The question was not found in the report."


class EmptyAnswerError(AppError):
    status_code = 422
    code = "empty_answer"
    message = "The answer is empty."


class AnswerTooLongError(AppError):
    status_code = 422
    code = "answer_too_long"
    message = "The answer exceeds the maximum allowed length."


class InterviewTimeoutError(AppError):
    status_code = 504
    code = "interview_timeout"
    message = "Generating the interview questions took too long."


class EvaluationTimeoutError(AppError):
    status_code = 504
    code = "evaluation_timeout"
    message = "Evaluating the answer took too long."
