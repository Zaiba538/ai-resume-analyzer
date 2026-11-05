"""M9 F2: the interview endpoints in the application's real OpenAPI document."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.api.routes.interview import (
    EVALUATE_BODY_EXAMPLES,
    inline_schema_refs,
)
from backend.api.routes.match import _REQUEST_BODY as MATCH_REQUEST_BODY
from backend.schemas.interview_api import (
    QUESTION_ID_PATTERN,
    AnswerEvaluationRequest,
    InterviewQuestionsRequest,
)

EVALUATE = "/api/v1/interview/evaluate"
QUESTIONS = "/api/v1/interview/questions"
# The component schemas FastAPI generated before F2; F2 must not add or change any.
COMPONENTS_BEFORE_F2 = {
    "AnalysisReport", "AnalysisResponse", "AnswerEvaluation", "AnswerEvaluationResponse",
    "ApiWarning", "AtsScore", "Baseline", "CriterionScore", "CriticIssue", "CriticSummary",
    "Education", "ErrorBody", "ErrorDetail", "ErrorEnvelope", "Experience",
    "HTTPValidationError", "HealthResponse", "InterviewQuestion", "InterviewQuestions",
    "InterviewQuestionsResponse", "MatchMetrics", "MatchResult", "ParseResponse",
    "ReadyResponse", "ReadyScoring", "ReportMeta", "ResumeProfile", "Rubric",
    "SectionCompletenessMetric", "SecurityFlag", "SemanticSimilarityMetric",
    "SkillCoverageMetric", "SkillGapReport", "UpstreamSnapshot", "ValidationError",
}  # fmt: skip


@pytest.fixture
def openapi(start_app) -> dict[str, Any]:
    client: TestClient = start_app(api_docs_enabled=True)
    return client.get("/openapi.json").json()


def _refs(node: Any) -> list[str]:
    if isinstance(node, dict):
        found = [node["$ref"]] if isinstance(node.get("$ref"), str) else []
        return found + [ref for value in node.values() for ref in _refs(value)]
    if isinstance(node, list):
        return [ref for value in node for ref in _refs(value)]
    return []


def _pointer(document: dict[str, Any], ref: str) -> Any:
    """Resolve a JSON-pointer reference from the document root (as OpenAPI does)."""
    assert ref.startswith("#/"), ref
    node: Any = document
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        assert isinstance(node, dict) and part in node, f"unresolvable reference {ref}"
        node = node[part]
    return node


def _resolve_all(document: dict[str, Any], node: Any, seen: frozenset[str] = frozenset()) -> None:
    """Follow every reference reachable from ``node``, recursively."""
    for ref in _refs(node):
        if ref not in seen:
            _resolve_all(document, _pointer(document, ref), seen | {ref})


def _evaluate_body(openapi: dict[str, Any]) -> dict[str, Any]:
    return openapi["paths"][EVALUATE]["post"]["requestBody"]


def _variants(prop: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The non-null and null alternatives of an optional property."""
    (value,) = [v for v in prop["anyOf"] if v.get("type") != "null"]
    assert {"type": "null"} in prop["anyOf"]
    return value, {"type": "null"}


# --- references ---------------------------------------------------------------------


def test_every_reference_in_the_openapi_document_resolves(openapi) -> None:
    refs = _refs(openapi)
    assert refs  # the document does use references (components)
    for ref in set(refs):
        _pointer(openapi, ref)
    assert "$defs" not in json.dumps(openapi["paths"])


def test_evaluate_request_schema_has_no_unresolved_references(openapi) -> None:
    body = _evaluate_body(openapi)
    schema = body["content"]["application/json"]["schema"]
    assert "$defs" not in json.dumps(schema)
    _resolve_all(openapi, body)  # every reference reachable from the request body resolves
    assert _refs(schema) == []  # the nested question schema is inlined


def test_client_question_fields_and_constraints_are_documented(openapi) -> None:
    schema = _evaluate_body(openapi)["content"]["application/json"]["schema"]
    question, _ = _variants(schema["properties"]["question"])
    assert question["type"] == "object" and question["title"] == "ClientQuestion"
    assert question["additionalProperties"] is False
    assert set(question["required"]) == {"id", "question", "topic", "rationale"}
    props = question["properties"]
    assert (props["id"]["pattern"], props["id"]["maxLength"]) == (QUESTION_ID_PATTERN, 15)
    assert (props["question"]["minLength"], props["question"]["maxLength"]) == (5, 500)
    assert (props["topic"]["minLength"], props["topic"]["maxLength"]) == (1, 100)
    assert props["rationale"]["maxLength"] == 300 and "minLength" not in props["rationale"]
    assert all(props[name]["type"] == "string" for name in props)
    assert "level" not in props and "level" not in schema["properties"]


def test_both_question_sources_and_the_exactly_one_rule_are_documented(openapi) -> None:
    body = _evaluate_body(openapi)
    media = body["content"]["application/json"]
    schema = media["schema"]
    assert body["required"] is True
    assert set(schema["properties"]) == {"report_id", "question_id", "question", "answer"}
    assert set(schema["required"]) == {"report_id", "answer"}
    assert schema["additionalProperties"] is False
    question_id, _ = _variants(schema["properties"]["question_id"])
    assert question_id["pattern"] == QUESTION_ID_PATTERN
    question, _ = _variants(schema["properties"]["question"])
    assert question["title"] == "ClientQuestion"
    assert schema["properties"]["answer"]["maxLength"] == 40_000
    description = body["description"]
    for phrase in ("exactly one", "`question_id`", "`question`", "derived from the question ID"):
        assert phrase in description
    assert set(media["examples"]) == {"stored_question", "client_question"}
    stored, supplied = (
        media["examples"][k]["value"] for k in ("stored_question", "client_question")
    )
    assert "question_id" in stored and "question" not in stored
    assert "question" in supplied and "question_id" not in supplied


@pytest.mark.parametrize("name", list(EVALUATE_BODY_EXAMPLES))
def test_documented_examples_are_valid_requests(name: str) -> None:
    request = AnswerEvaluationRequest.model_validate(EVALUATE_BODY_EXAMPLES[name]["value"])
    assert request.level in {"beginner", "advanced"}


# --- nothing else changed ---------------------------------------------------------------


def test_component_schemas_are_unchanged_and_not_duplicated(openapi) -> None:
    components = openapi["components"]["schemas"]
    assert set(components) == COMPONENTS_BEFORE_F2
    assert "ClientQuestion" not in components and "AnswerEvaluationRequest" not in components


def test_other_request_bodies_are_unchanged(openapi) -> None:
    questions = openapi["paths"][QUESTIONS]["post"]["requestBody"]
    assert questions == {
        "required": True,
        "content": {"application/json": {"schema": InterviewQuestionsRequest.model_json_schema()}},
    }
    assert openapi["paths"]["/api/v1/match"]["post"]["requestBody"] == MATCH_REQUEST_BODY
    evaluate = openapi["paths"][EVALUATE]["post"]
    assert set(evaluate["responses"]) == {"200", "404", "415", "422", "429", "502", "503", "504"}
    assert evaluate["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/AnswerEvaluationResponse"
    }


def test_inlining_is_exact_and_only_affects_references() -> None:
    raw = AnswerEvaluationRequest.model_json_schema()
    inlined = inline_schema_refs(raw)
    assert raw["$defs"]["ClientQuestion"] in [
        v for v in inlined["properties"]["question"]["anyOf"] if v.get("type") != "null"
    ]
    without = {k: v for k, v in raw.items() if k != "$defs"}
    assert {k: v for k, v in inlined.items() if k != "properties"} == {
        k: v for k, v in without.items() if k != "properties"
    }
    for name in ("report_id", "question_id", "answer"):
        assert inlined["properties"][name] == raw["properties"][name]
    assert "$defs" in raw  # the model's own schema is not modified
    assert inline_schema_refs(InterviewQuestionsRequest.model_json_schema()) == (
        InterviewQuestionsRequest.model_json_schema()
    )


@pytest.mark.parametrize(
    "schema",
    [
        {"$defs": {}, "properties": {"x": {"$ref": "#/$defs/Missing"}}},
        {"properties": {"x": {"$ref": "#/components/schemas/Other"}}},
        {"$defs": {"A": {"properties": {"self": {"$ref": "#/$defs/A"}}}}, "$ref": "#/$defs/A"},
    ],
)
def test_unknown_or_cyclic_references_fail_loudly(schema: dict) -> None:
    with pytest.raises(ValueError, match="cannot inline schema reference"):
        inline_schema_refs(schema)


def test_inlining_keeps_sibling_keywords_and_handles_lists() -> None:
    schema = {
        "$defs": {"B": {"type": "integer"}},
        "anyOf": [{"$ref": "#/$defs/B", "description": "kept"}, {"type": "null"}],
    }
    assert inline_schema_refs(schema) == {
        "anyOf": [{"type": "integer", "description": "kept"}, {"type": "null"}]
    }


# --- runtime validation is unaffected ---------------------------------------------------


def test_request_validation_through_the_app_is_unchanged(start_app) -> None:
    client: TestClient = start_app()
    report_id = str(uuid.uuid4())
    question = EVALUATE_BODY_EXAMPLES["client_question"]["value"]["question"]

    def details(body: Any) -> tuple[int, list[dict]]:
        response = client.post(EVALUATE, json=body)
        return response.status_code, response.json()["error"].get("details", [])

    both = {
        "report_id": report_id,
        "question_id": "beginner-1",
        "question": question,
        "answer": "x",
    }
    status, errors = details(both)
    assert status == 422 and "exactly one of question_id or question" in errors[0]["msg"]
    status, errors = details({"report_id": report_id, "answer": "x"})
    assert status == 422 and "exactly one of question_id or question" in errors[0]["msg"]
    status, errors = details({**both, "question_id": None, "level": "advanced"})
    assert status == 422 and errors[0]["type"] == "extra_forbidden"
    oversized = {**question, "topic": "t" * 201}  # F1 raw cap
    status, errors = details({"report_id": report_id, "question": oversized, "answer": "x"})
    assert status == 422 and errors[0]["type"] == "string_too_long"
    duplicate = client.post(
        EVALUATE,
        content=b'{"report_id": "x", "report_id": "y", "question_id": "beginner-1", "answer": "x"}',
        headers={"Content-Type": "application/json"},
    )
    assert duplicate.json()["error"]["details"][0]["type"] == "duplicate_key"
    valid_unknown_report = client.post(
        EVALUATE, json=EVALUATE_BODY_EXAMPLES["stored_question"]["value"]
    )
    assert valid_unknown_report.status_code == 404  # passes validation; no such report
