"""Interview API (M9 Phase 3; decisions Q1-Q9, N1-N3).

``POST /api/v1/interview/questions``: interview questions for chosen levels,
generated from a stored report's profile (``InterviewQuestionService``).

``POST /api/v1/interview/evaluate``: the evaluation of one answer to a question
in the stored report (``question_id``) or a client-supplied ``question``,
exactly one of the two (``EvaluationWorkflow``).

Requests must be ``application/json`` (otherwise 415). Bodies are parsed here
rather than by FastAPI so that duplicate object keys can be rejected (approved
Phase 1 decision 4): JSON parsers otherwise keep the last duplicate silently,
which would let two parts of a system read different values from one body.
A duplicate key at any nesting level, malformed JSON and schema violations are
all 422 ``validation_error`` through the existing handler, whose details hold
only ``loc``, ``type`` and ``msg``: submitted values, raw request content and
parser messages are never returned or logged.

Neither endpoint stores anything: the report store is only read, and answers,
questions and evaluations exist for the request only. Errors are the typed
errors of the service and graph (404, 422, 502, 503, 504). Responses carry
``Cache-Control: no-store`` and ``X-Robots-Tag: noindex`` like every ``/api/``
path, and the access log records only the route template.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from backend.api.deps import EvaluationWorkflowDep, InterviewServiceDep
from backend.core.errors import AppError
from backend.schemas.common import ErrorEnvelope
from backend.schemas.interview_api import (
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    InterviewQuestionsRequest,
    InterviewQuestionsResponse,
)

router = APIRouter(prefix="/api/v1/interview", tags=["interview"])

JSON_MEDIA_TYPE = "application/json"
DUPLICATE_KEY_ERROR = {
    "loc": ("body",),
    "type": "duplicate_key",
    "msg": "JSON objects must not contain duplicate keys.",
}
INVALID_JSON_ERROR = {"loc": ("body",), "type": "json_invalid", "msg": "JSON decode error"}


class UnsupportedInterviewMediaTypeError(AppError):
    status_code = 415
    code = "unsupported_media_type"
    message = "Send the request body as application/json."


class _DuplicateKeyError(Exception):
    """Raised by the parser hook; deliberately not a ``ValueError``."""


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook``: called for every JSON object, at every depth."""
    obj = dict(pairs)
    if len(obj) != len(pairs):
        raise _DuplicateKeyError
    return obj


async def read_json_body[T: BaseModel](request: Request, model: type[T]) -> T:
    """Parse a JSON body strictly (no duplicate keys) and validate it with ``model``."""
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != JSON_MEDIA_TYPE:
        raise UnsupportedInterviewMediaTypeError()
    raw = await request.body()
    try:
        data = json.loads(raw, object_pairs_hook=_unique_keys)
    except _DuplicateKeyError:
        raise RequestValidationError([DUPLICATE_KEY_ERROR]) from None
    except (ValueError, RecursionError):  # malformed JSON, bad encoding, excessive nesting
        raise RequestValidationError([INVALID_JSON_ERROR]) from None
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        raise RequestValidationError(
            [{**error, "loc": ("body", *error["loc"])} for error in errors]
        ) from None


def inline_schema_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` with its local ``#/$defs/...`` references replaced by their definitions.

    Pydantic keeps nested models in ``$defs`` and refers to them from inside the
    schema. Embedded in an OpenAPI document, those references would be resolved
    from the document root, where no ``$defs`` exists. The request bodies here are
    documented through ``openapi_extra``, outside FastAPI's components, so nested
    models are inlined instead. Unknown or cyclic references raise at import.
    """
    definitions = schema.get("$defs", {})

    def resolve(node: Any, active: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, active) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if ref is None:
            return {key: resolve(value, active) for key, value in node.items() if key != "$defs"}
        name = ref.removeprefix("#/$defs/")
        if ref != f"#/$defs/{name}" or name not in definitions or name in active:
            raise ValueError(f"cannot inline schema reference {ref!r}")
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        return resolve({**definitions[name], **siblings}, active | {name})

    return resolve(schema, frozenset())


def _request_body(
    model: type[BaseModel],
    *,
    description: str | None = None,
    examples: dict[str, Any] | None = None,
) -> dict[str, Any]:
    media: dict[str, Any] = {"schema": inline_schema_refs(model.model_json_schema())}
    if examples:
        media["examples"] = examples
    body: dict[str, Any] = {"required": True, "content": {JSON_MEDIA_TYPE: media}}
    if description:
        body["description"] = description
    return {"requestBody": body}


EVALUATE_BODY_DESCRIPTION = (
    "Provide exactly one question source: `question_id`, the ID of a question in "
    "the stored report, or `question`, a client-supplied question object (for "
    "example one returned by `/interview/questions`). Supplying both or neither "
    "is a 422 `validation_error`. The level is derived from the question ID and "
    "cannot be supplied separately."
)
EVALUATE_BODY_EXAMPLES = {
    "stored_question": {
        "summary": "A question in the stored report",
        "value": {
            "report_id": "3f6c1f0e-8c1a-4c55-9a43-2b1f4f6f2a10",
            "question_id": "beginner-1",
            "answer": "A list is an ordered, mutable sequence.",
        },
    },
    "client_question": {
        "summary": "A client-supplied question",
        "value": {
            "report_id": "3f6c1f0e-8c1a-4c55-9a43-2b1f4f6f2a10",
            "question": {
                "id": "advanced-2",
                "question": "How would you design an idempotent ETL step?",
                "topic": "ETL",
                "rationale": "Data engineering role.",
            },
            "answer": "Use upserts keyed on a natural key.",
        },
    },
}


def _errors(descriptions: dict[int, str]) -> dict[int | str, dict[str, Any]]:
    return {
        status: {"model": ErrorEnvelope, "description": text}
        for status, text in descriptions.items()
    }


_COMMON_ERRORS = {
    415: "Body is not application/json",
    422: "Invalid JSON, duplicate keys or schema violation",
    502: "Upstream model error or invalid model output",
    503: "LLM not configured or upstream rate limited",
}


@router.post(
    "/questions",
    response_model=InterviewQuestionsResponse,
    responses=_errors(
        {
            **_COMMON_ERRORS,
            404: "Unknown, malformed or expired report ID",
            504: "Question generation exceeded its deadline",
        }
    ),
    openapi_extra=_request_body(InterviewQuestionsRequest),
)
async def interview_questions(
    request: Request, service: InterviewServiceDep
) -> InterviewQuestionsResponse:
    body = await read_json_body(request, InterviewQuestionsRequest)
    return await service.generate(body)


@router.post(
    "/evaluate",
    response_model=AnswerEvaluationResponse,
    responses=_errors(
        {
            **_COMMON_ERRORS,
            404: "Unknown, malformed or expired report ID, or unknown question ID",
            422: "Invalid JSON, duplicate keys, schema violation, or empty or too long answer",
            504: "Evaluation exceeded its deadline",
        }
    ),
    openapi_extra=_request_body(
        AnswerEvaluationRequest,
        description=EVALUATE_BODY_DESCRIPTION,
        examples=EVALUATE_BODY_EXAMPLES,
    ),
)
async def evaluate_answer(
    request: Request, workflow: EvaluationWorkflowDep
) -> AnswerEvaluationResponse:
    body = await read_json_body(request, AnswerEvaluationRequest)
    return await workflow.run(body)
