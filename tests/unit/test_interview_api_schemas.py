"""M9 Phase 1: interview API schemas, settings and typed errors."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from backend.core.config import MAX_ANSWER_CHARS_LIMIT, MIN_ANSWER_CHARS_LIMIT
from backend.schemas.interview import (
    RUBRIC_CRITERIA,
    AnswerEvaluation,
    AnswerEvaluationDraft,
    InterviewQuestions,
    QuestionDraft,
    question_set_model,
    with_ids,
)
from backend.schemas.interview_api import (
    ANSWER_MAX_INPUT_CHARS,
    AnswerEvaluationRequest,
    AnswerEvaluationResponse,
    ClientQuestion,
    InterviewQuestionsRequest,
    InterviewQuestionsResponse,
    level_of,
)
from backend.schemas.security import SecurityFlag
from backend.services.errors import (
    AnswerTooLongError,
    EmptyAnswerError,
    EvaluationTimeoutError,
    InterviewTimeoutError,
    QuestionNotFoundError,
)
from backend.storage.report_store import parse_report_id

REPORT_ID = str(uuid.UUID("3f6c1f0e-8c1a-4c55-9a43-2b1f4f6f2a10"))
CANARY = "CANARYm9schema41"
QUESTION = {
    "id": "intermediate-2",
    "question": "How would you tune a Spark job?",
    "topic": "Apache Spark",
    "rationale": "Listed skill.",
}


def _evaluate(**fields: Any) -> AnswerEvaluationRequest:
    body = {"report_id": REPORT_ID, "answer": "Partition the data.", **fields}
    return AnswerEvaluationRequest.model_validate(body)


def _errors(exc: pytest.ExceptionInfo[ValidationError]) -> list[tuple[tuple, str]]:
    return [(tuple(e["loc"]), e["type"]) for e in exc.value.errors()]


# --- InterviewQuestionsRequest --------------------------------------------------------


def test_questions_request_defaults_to_the_m4_defaults() -> None:
    request = InterviewQuestionsRequest.model_validate({"report_id": REPORT_ID})
    assert request.levels == ["beginner", "intermediate", "advanced"]
    assert request.count_per_level == 3
    assert request.requested_levels == ("beginner", "intermediate", "advanced")


def test_questions_request_levels_are_reported_in_canonical_order() -> None:
    request = InterviewQuestionsRequest.model_validate(
        {"report_id": REPORT_ID, "levels": ["advanced", "beginner"], "count_per_level": 5}
    )
    assert request.requested_levels == ("beginner", "advanced")
    assert request.count_per_level == 5


@pytest.mark.parametrize(
    "levels",
    [
        [],
        ["expert"],
        ["Beginner"],
        ["beginner", "beginner"],
        ["beginner", " advanced"],
        "beginner",
        ["beginner", "intermediate", "advanced", "beginner"],
        [1],
        [None],
    ],
)
def test_questions_request_rejects_invalid_levels(levels: Any) -> None:
    with pytest.raises(ValidationError):
        InterviewQuestionsRequest.model_validate({"report_id": REPORT_ID, "levels": levels})


@pytest.mark.parametrize("count", [0, 6, -1, True, False, 3.0, "3", None])
def test_questions_request_count_is_a_strict_integer_from_1_to_5(count: Any) -> None:
    with pytest.raises(ValidationError):
        InterviewQuestionsRequest.model_validate({"report_id": REPORT_ID, "count_per_level": count})


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_questions_request_accepts_counts_1_to_5(count: int) -> None:
    request = InterviewQuestionsRequest.model_validate(
        {"report_id": REPORT_ID, "count_per_level": count}
    )
    assert request.count_per_level == count


@pytest.mark.parametrize(
    "extra",
    [
        {"model": "x"},
        {"level": "advanced"},
        {"criteria": ["depth"]},
        {"temperature": 1},
        {"profile": {}},
        {"system": "x"},
        {"prompt": "x"},
    ],
)
def test_questions_request_rejects_unknown_fields(extra: dict) -> None:
    with pytest.raises(ValidationError) as raised:
        InterviewQuestionsRequest.model_validate({"report_id": REPORT_ID, **extra})
    assert {kind for _, kind in _errors(raised)} == {"extra_forbidden"}


@pytest.mark.parametrize("report_id", [None, 7, ["x"], {"id": REPORT_ID}])
def test_report_id_must_be_a_string(report_id: Any) -> None:
    with pytest.raises(ValidationError):
        InterviewQuestionsRequest.model_validate({"report_id": report_id})
    with pytest.raises(ValidationError):
        _evaluate(report_id=report_id, question_id="beginner-1")


@pytest.mark.parametrize(
    "report_id",
    ["", "not-a-uuid", f" {REPORT_ID} ", REPORT_ID.upper()[:-1] + "Z", REPORT_ID + "0", "x" * 5000],
)
def test_malformed_report_ids_are_kept_verbatim_for_the_uniform_404(report_id: str) -> None:
    # Validation never rejects or repairs a string ID; the store decides (one 404 for all).
    assert InterviewQuestionsRequest.model_validate({"report_id": report_id}).report_id == report_id
    assert _evaluate(report_id=report_id, question_id="beginner-1").report_id == report_id
    assert parse_report_id(report_id) is None


# --- InterviewQuestionsResponse ------------------------------------------------------


def _questions(levels: tuple[str, ...], count: int) -> InterviewQuestions:
    drafts = question_set_model(levels, count)(
        **{
            level: [
                QuestionDraft(
                    question=f"{level} question number {i}?", topic="Python", rationale="Skill."
                )
                for i in range(count)
            ]
            for level in levels
        }
    )
    return InterviewQuestions(**with_ids(drafts))


def test_questions_response_accepts_matching_questions() -> None:
    response = InterviewQuestionsResponse(
        levels=["beginner", "advanced"],
        count_per_level=2,
        questions=_questions(("beginner", "advanced"), 2),
        security_flags=[],
    )
    assert [q.id for q in response.questions.advanced] == ["advanced-1", "advanced-2"]
    assert response.questions.intermediate == []


@pytest.mark.parametrize(
    ("levels", "count", "built"),
    [
        (["beginner"], 2, (("beginner",), 3)),  # wrong count
        (["beginner"], 2, (("beginner", "advanced"), 2)),  # unrequested level present
        (["beginner", "advanced"], 2, (("beginner",), 2)),  # requested level missing
        (["advanced", "beginner"], 1, (("beginner", "advanced"), 1)),  # not canonical
        (["beginner", "beginner"], 1, (("beginner",), 1)),  # repeated
    ],
)
def test_questions_response_must_match_levels_and_count(levels, count, built) -> None:
    with pytest.raises(ValidationError):
        InterviewQuestionsResponse(
            levels=levels, count_per_level=count, questions=_questions(*built), security_flags=[]
        )


# --- AnswerEvaluationRequest -------------------------------------------------------------


def test_evaluate_with_a_stored_question_id() -> None:
    request = _evaluate(question_id="advanced-10")
    assert request.question is None
    assert request.target_question_id == "advanced-10"
    assert request.level == "advanced"


def test_evaluate_with_a_client_question_derives_the_level_from_its_id() -> None:
    request = _evaluate(question=QUESTION)
    assert request.question_id is None
    assert request.target_question_id == "intermediate-2"
    assert request.level == "intermediate"
    assert request.question.level == "intermediate"


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"question_id": None},
        {"question": None},
        {"question_id": None, "question": None},
        {"question_id": "beginner-1", "question": QUESTION},
        {"question_id": "intermediate-2", "question": QUESTION},  # same ID twice is still two
    ],
)
def test_exactly_one_question_source_is_required(fields: dict) -> None:
    with pytest.raises(ValidationError) as raised:
        _evaluate(**fields)
    assert "exactly one of question_id or question" in str(raised.value)


@pytest.mark.parametrize(
    "question_id",
    [
        "beginner-0",
        "beginner-01",
        "beginner-11",
        "beginner-100",
        "expert-1",
        "Beginner-1",
        "beginner_1",
        "beginner-",
        "-1",
        "beginner-1 ",
        " beginner-1",
        "beginner-1\n",
        "beginner-\uff11",  # full-width digit
        "beginner-1-2",
        "",
        3,
        True,
        b"beginner-1",  # bytes are not a string (strict)
    ],
)
def test_invalid_question_ids_are_rejected(question_id: Any) -> None:
    with pytest.raises(ValidationError):
        _evaluate(question_id=question_id)
    with pytest.raises(ValidationError):
        _evaluate(question={**QUESTION, "id": question_id})


@pytest.mark.parametrize("question_id", ["beginner-1", "intermediate-5", "advanced-10"])
def test_valid_question_ids(question_id: str) -> None:
    assert _evaluate(question_id=question_id).level == level_of(question_id)


@pytest.mark.parametrize(
    "extra",
    [
        {"level": "beginner"},
        {"model": "openai/gpt-oss-120b"},
        {"criteria": ["depth"]},
        {"rubric": {}},
        {"overall_rating": 5},
        {"temperature": 2},
        {"system": "x"},
        {"evaluation": {}},
        {"levels": ["advanced"]},
    ],
)
def test_evaluate_request_rejects_settings_and_levels_from_the_client(extra: dict) -> None:
    with pytest.raises(ValidationError) as raised:
        _evaluate(question_id="beginner-1", **extra)
    assert {kind for _, kind in _errors(raised)} == {"extra_forbidden"}


@pytest.mark.parametrize("extra", [{"level": "advanced"}, {"model": "x"}, {"answer": "x"}])
def test_client_question_rejects_unknown_fields(extra: dict) -> None:
    with pytest.raises(ValidationError) as raised:
        _evaluate(question={**QUESTION, **extra})
    assert {kind for _, kind in _errors(raised)} == {"extra_forbidden"}


@pytest.mark.parametrize(
    "changes",
    [
        {"question": "Why?"},  # shorter than 5 characters
        {"question": "x" * 501},
        {"question": ""},
        {"question": chr(0x200B) * 10},  # only invisible characters: empty once sanitised
        {"topic": ""},
        {"topic": "x" * 101},
        {"rationale": "x" * 301},
        {"question": None},
        {"question": 42},
        {"topic": ["Python"]},
    ],
)
def test_invalid_client_question_fields_are_rejected(changes: dict) -> None:
    with pytest.raises(ValidationError):
        _evaluate(question={**QUESTION, **changes})


@pytest.mark.parametrize("missing", ["id", "question", "topic", "rationale"])
def test_client_question_requires_every_field(missing: str) -> None:
    fields = {k: v for k, v in QUESTION.items() if k != missing}
    with pytest.raises(ValidationError):
        _evaluate(question=fields)


def test_client_question_fields_are_sanitised_before_validation() -> None:
    zwsp, bidi = chr(0x200B), chr(0x202E)
    request = _evaluate(
        question={
            "id": "beginner-3",
            "question": f"  What{zwsp} is a Python{bidi} list?  ",
            "topic": f"Py{zwsp}thon",
            "rationale": f"Listed{chr(0xFE0F)} skill.\r\n",
        }
    )
    question = request.question
    assert question.question == "What is a Python list?"
    assert question.topic == "Python"
    assert zwsp not in question.model_dump_json() and bidi not in question.model_dump_json()


def test_client_question_length_is_checked_after_sanitising() -> None:
    padded = "What is a list?" + chr(0x200B) * 600  # 615 raw characters, 15 once sanitised
    assert _evaluate(question={**QUESTION, "question": padded}).question.question == (
        "What is a list?"
    )


def test_client_question_id_is_never_normalised() -> None:
    # A full-width digit would become "1" under NFKC; IDs are matched exactly instead.
    with pytest.raises(ValidationError):
        _evaluate(question={**QUESTION, "id": "beginner-" + chr(0xFF11)})


@pytest.mark.parametrize("answer", [None, 5, ["text"], {"text": "x"}, True, b"x", bytearray(b"x")])
def test_answer_must_be_a_string(answer: Any) -> None:
    with pytest.raises(ValidationError):
        _evaluate(question_id="beginner-1", answer=answer)


def test_answer_is_required() -> None:
    with pytest.raises(ValidationError):
        AnswerEvaluationRequest.model_validate(
            {"report_id": REPORT_ID, "question_id": "beginner-1"}
        )


@pytest.mark.parametrize("answer", ["", "   ", chr(0x200B)])
def test_empty_answers_pass_the_schema_for_the_typed_empty_answer_error(answer: str) -> None:
    assert _evaluate(question_id="beginner-1", answer=answer).answer == answer


def test_answer_is_kept_verbatim_for_sanitisation_by_the_service() -> None:
    raw = f"  Use {chr(0x200B)}generators.\r\n"
    assert _evaluate(question_id="beginner-1", answer=raw).answer == raw


def test_raw_answer_hard_bound() -> None:
    assert ANSWER_MAX_INPUT_CHARS == 2 * MAX_ANSWER_CHARS_LIMIT == 40_000
    assert len(_evaluate(question_id="beginner-1", answer="a" * ANSWER_MAX_INPUT_CHARS).answer) == (
        ANSWER_MAX_INPUT_CHARS
    )
    with pytest.raises(ValidationError) as raised:
        _evaluate(question_id="beginner-1", answer="a" * (ANSWER_MAX_INPUT_CHARS + 1))
    assert _errors(raised) == [(("answer",), "string_too_long")]


def test_requests_are_immutable() -> None:
    request = _evaluate(question_id="beginner-1")
    with pytest.raises(ValidationError):
        request.question_id = "advanced-1"  # type: ignore[misc]


def _published(raised: pytest.ExceptionInfo[ValidationError]) -> str:
    """What the API's validation handler publishes: loc, type and msg only (M1)."""
    return json.dumps(
        [{"loc": e["loc"], "type": e["type"], "msg": e["msg"]} for e in raised.value.errors()]
    )


@pytest.mark.parametrize(
    "body",
    [
        {"report_id": CANARY, "question_id": CANARY, "answer": CANARY},
        {
            "report_id": REPORT_ID,
            "question": {**QUESTION, "id": CANARY, "topic": ""},
            "answer": "x",
        },
        {"report_id": REPORT_ID, "question": {**QUESTION, "question": CANARY * 40}, "answer": "x"},
        {"report_id": REPORT_ID, "answer": CANARY * 5000, "question_id": "beginner-1"},
        {
            "report_id": REPORT_ID,
            "question_id": "beginner-1",
            "question": QUESTION,
            "answer": CANARY,
        },
        {"report_id": [CANARY], "answer": {"text": CANARY}, "question_id": 1},
    ],
)
def test_published_validation_errors_contain_no_submitted_values(body: dict) -> None:
    with pytest.raises(ValidationError) as raised:
        AnswerEvaluationRequest.model_validate(body)
    assert CANARY not in _published(raised)


def test_unknown_field_names_appear_in_loc_as_in_every_existing_endpoint() -> None:
    # Existing M1 behaviour: the error location names the unknown key (never its value).
    with pytest.raises(ValidationError) as raised:
        _evaluate(question_id="beginner-1", **{"unexpected_key": CANARY})
    assert "unexpected_key" in _published(raised)
    assert CANARY not in _published(raised)


# --- AnswerEvaluationResponse ----------------------------------------------------------


def _evaluation() -> AnswerEvaluation:
    rubric = {c: {"rating": 4, "justification": "Covers the point well."} for c in RUBRIC_CRITERIA}
    rubric["depth"]["rating"] = 3
    draft = AnswerEvaluationDraft(
        rubric=rubric,
        strengths=["Clear."],
        improvements=["More depth."],
        model_answer_outline=["Define it."],
    )
    return AnswerEvaluation.from_draft(draft)


def test_evaluation_response_keeps_the_five_criteria_and_computed_rating() -> None:
    response = AnswerEvaluationResponse(
        question_id="intermediate-2",
        level="intermediate",
        evaluation=_evaluation(),
        security_flags=[SecurityFlag(type="instruction_override", source="answer")],
    )
    assert RUBRIC_CRITERIA == ("relevance", "technical_accuracy", "depth", "clarity", "structure")
    assert tuple(response.evaluation.rubric.model_dump()) == RUBRIC_CRITERIA
    assert response.evaluation.overall_rating == 3.8  # (4+4+3+4+4)/5, computed in Python


def test_evaluation_response_level_must_match_the_question_id() -> None:
    with pytest.raises(ValidationError):
        AnswerEvaluationResponse(
            question_id="intermediate-2",
            level="advanced",
            evaluation=_evaluation(),
            security_flags=[],
        )


def test_evaluation_response_rejects_a_model_rating_that_is_not_the_mean() -> None:
    data = _evaluation().model_dump()
    data["overall_rating"] = 5.0
    with pytest.raises(ValidationError):
        AnswerEvaluationResponse.model_validate(
            {
                "question_id": "beginner-1",
                "level": "beginner",
                "evaluation": data,
                "security_flags": [],
            }
        )


@pytest.mark.parametrize("extra", ["answer", "question", "answer_text", "raw_answer", "preview"])
def test_responses_cannot_carry_the_answer(extra: str) -> None:
    with pytest.raises(ValidationError):
        AnswerEvaluationResponse.model_validate(
            {
                "question_id": "beginner-1",
                "level": "beginner",
                "evaluation": _evaluation().model_dump(),
                "security_flags": [],
                extra: CANARY,
            }
        )
    with pytest.raises(ValidationError):
        AnswerEvaluationResponse.model_validate(
            {
                "question_id": "beginner-1",
                "level": "beginner",
                "evaluation": {**_evaluation().model_dump(), extra: CANARY},
                "security_flags": [],
            }
        )


def _property_names(schema: dict) -> set[str]:
    names: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            names.update(node.get("properties", {}))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    return names


def test_no_response_schema_has_an_answer_field() -> None:
    evaluation = _property_names(AnswerEvaluationResponse.model_json_schema())
    questions = _property_names(InterviewQuestionsResponse.model_json_schema())
    assert "answer" not in evaluation | questions
    assert not {"question", "topic", "rationale"} & evaluation  # the question is not echoed
    assert set(AnswerEvaluationResponse.model_fields) == {
        "question_id",
        "level",
        "evaluation",
        "security_flags",
    }
    assert set(InterviewQuestionsResponse.model_fields) == {
        "levels",
        "count_per_level",
        "questions",
        "security_flags",
    }


def test_level_of() -> None:
    assert [level_of(i) for i in ("beginner-1", "intermediate-9", "advanced-10")] == [
        "beginner",
        "intermediate",
        "advanced",
    ]
    with pytest.raises(ValueError):
        level_of("expert-1")


def test_client_question_model_has_no_level_field() -> None:
    assert set(ClientQuestion.model_fields) == {"id", "question", "topic", "rationale"}
    assert "level" not in AnswerEvaluationRequest.model_fields


# --- Settings (decision Q7) ------------------------------------------------------------------


def test_interview_settings_defaults(make_settings) -> None:
    settings = make_settings()
    assert settings.max_answer_chars == 5000
    assert settings.interview_deadline_seconds == 120.0
    assert settings.evaluation_deadline_seconds == 120.0
    assert (MIN_ANSWER_CHARS_LIMIT, MAX_ANSWER_CHARS_LIMIT) == (200, 20_000)


@pytest.mark.parametrize("value", [200, 5000, 20_000])
def test_answer_limit_accepts_the_approved_range(make_settings, value: int) -> None:
    assert make_settings(max_answer_chars=value).max_answer_chars == value


@pytest.mark.parametrize("value", [199, 20_001, 0, -1])
def test_answer_limit_rejects_values_outside_200_to_20000(make_settings, value: int) -> None:
    with pytest.raises(ValidationError):
        make_settings(max_answer_chars=value)


@pytest.mark.parametrize("name", ["interview_deadline_seconds", "evaluation_deadline_seconds"])
@pytest.mark.parametrize("value", [0, 0.5, -1, 601])
def test_deadlines_are_bounded(make_settings, name: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make_settings(**{name: value})


def test_interview_settings_load_from_environment(monkeypatch) -> None:
    from backend.core.config import Settings

    monkeypatch.setenv("MAX_ANSWER_CHARS", "8000")
    monkeypatch.setenv("INTERVIEW_DEADLINE_SECONDS", "90")
    monkeypatch.setenv("EVALUATION_DEADLINE_SECONDS", "60.5")
    settings = Settings(_env_file=None)
    assert settings.max_answer_chars == 8000
    assert settings.interview_deadline_seconds == 90.0
    assert settings.evaluation_deadline_seconds == 60.5


def test_answer_limit_is_documented_in_env_example() -> None:
    from backend.core.config import PROJECT_ROOT

    assert "MAX_ANSWER_CHARS=5000" in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")


# --- Typed errors (decisions Q7, Q8) -----------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (QuestionNotFoundError, 404, "question_not_found"),
        (EmptyAnswerError, 422, "empty_answer"),
        (AnswerTooLongError, 422, "answer_too_long"),
        (InterviewTimeoutError, 504, "interview_timeout"),
        (EvaluationTimeoutError, 504, "evaluation_timeout"),
    ],
)
def test_typed_errors(error, status: int, code: str) -> None:
    instance = error()
    assert (instance.status_code, instance.code) == (status, code)
    assert instance.message and instance.message.endswith(".")
    assert str(instance) == code  # AppError convention: no content in the exception text


# --- Hidden Unicode in client-supplied questions (Phase 2 correction) ----------------

HIDDEN = {
    "zero_width_space": chr(0x200B),
    "word_joiner": chr(0x2060),
    "right_to_left_override": chr(0x202E),
    "left_to_right_isolate": chr(0x2066),
    "pop_directional_formatting": chr(0x202C),
    "tag_character": chr(0xE0041),
    "variation_selector_supplement": chr(0xE0100),
}


@pytest.mark.parametrize("field", ["question", "topic", "rationale"])
@pytest.mark.parametrize("name", list(HIDDEN))
def test_hidden_unicode_is_recorded_before_sanitising(field: str, name: str) -> None:
    char = HIDDEN[name]
    original = QUESTION[field] or "Listed skill."
    raw = original[:3] + char + original[3:]
    question = _evaluate(question={**QUESTION, field: raw}).question
    assert question.hidden_unicode is True
    assert getattr(question, field) == original  # sanitised text only
    assert char not in question.model_dump_json()


def test_clean_question_has_no_hidden_unicode_finding() -> None:
    assert _evaluate(question=QUESTION).question.hidden_unicode is False


@pytest.mark.parametrize("char", [chr(0xFEFF), chr(0x00AD)])
def test_benign_format_characters_are_removed_without_a_finding(char: str) -> None:
    # Same classification as M3 sanitisation: a BOM or soft hyphen is not suspicious.
    question = _evaluate(question={**QUESTION, "topic": "Apache" + char + " Spark"}).question
    assert question.topic == "Apache Spark" and question.hidden_unicode is False


def test_the_finding_cannot_be_supplied_or_serialised() -> None:
    for key in ("hidden_unicode", "_hidden_unicode"):
        with pytest.raises(ValidationError) as raised:
            _evaluate(question={**QUESTION, key: True})
        assert {kind for _, kind in _errors(raised)} == {"extra_forbidden"}
    flagged = _evaluate(question={**QUESTION, "question": "What" + chr(0x202E) + " is a list?"})
    dumped = flagged.model_dump()
    assert (
        "hidden_unicode" not in json.dumps(dumped) and "_hidden_unicode" not in dumped["question"]
    )
    assert set(flagged.question.model_dump()) == {"id", "question", "topic", "rationale"}


def test_the_finding_survives_copies_and_revalidation_of_the_instance() -> None:
    request = _evaluate(question={**QUESTION, "rationale": "Listed" + chr(0x200B) + " skill."})
    assert request.model_copy(deep=True).question.hidden_unicode is True
    again = AnswerEvaluationRequest.model_validate(
        {"report_id": REPORT_ID, "answer": "x", "question": request.question}
    )
    assert again.question.hidden_unicode is True


def test_hidden_unicode_does_not_reject_and_limits_still_apply_after_sanitising() -> None:
    padded = "What is a list?" + chr(0x202E) * 600  # 615 raw characters, 15 once sanitised
    question = _evaluate(question={**QUESTION, "question": padded}).question
    assert question.question == "What is a list?" and question.hidden_unicode is True
    with pytest.raises(ValidationError):  # only hidden characters: empty once sanitised
        _evaluate(question={**QUESTION, "question": chr(0x200B) * 10})


# --- Raw-length caps before sanitisation (audit finding F1) -----------------------------

RAW_CAPS = {"question": 1_000, "topic": 200, "rationale": 600}
POST_LIMITS = {"question": 500, "topic": 100, "rationale": 300}


def test_raw_caps_are_twice_the_post_sanitisation_limits() -> None:
    from backend.schemas.interview import QuestionDraft
    from backend.schemas.interview_api import CLIENT_QUESTION_RAW_CAPS

    assert CLIENT_QUESTION_RAW_CAPS == RAW_CAPS
    for name, cap in RAW_CAPS.items():
        (constraints,) = QuestionDraft.model_fields[name].metadata
        assert constraints.max_length == POST_LIMITS[name] and cap == 2 * POST_LIMITS[name]
    assert ANSWER_MAX_INPUT_CHARS == 40_000  # the answer's raw bound is unchanged


@pytest.mark.parametrize("field", list(RAW_CAPS))
def test_each_field_over_its_raw_cap_is_rejected(field: str) -> None:
    raw = CANARY + "x" * (RAW_CAPS[field] + 1 - len(CANARY))
    with pytest.raises(ValidationError) as raised:
        _evaluate(question={**QUESTION, field: raw})
    errors = raised.value.errors()
    assert [(e["loc"], e["type"]) for e in errors] == [(("question",), "string_too_long")]
    assert errors[0]["msg"] == f"Field {field} should have at most {RAW_CAPS[field]} characters"
    assert CANARY not in _published(raised)


@pytest.mark.parametrize("field", list(RAW_CAPS))
def test_oversized_fields_are_rejected_before_any_sanitisation(field: str, monkeypatch) -> None:
    from backend.schemas import interview_api

    calls: list[int] = []
    real = interview_api.sanitize_text

    def spy(text: str, **kwargs):
        calls.append(len(text))
        return real(text, **kwargs)

    monkeypatch.setattr(interview_api, "sanitize_text", spy)
    with pytest.raises(ValidationError):
        _evaluate(question={**QUESTION, field: "y" * 5_000_000})
    assert calls == []  # nothing was sanitised, not even the fields within their caps
    _evaluate(question=QUESTION)
    assert sorted(calls) == sorted(len(QUESTION[name]) for name in RAW_CAPS)


@pytest.mark.parametrize("field", list(RAW_CAPS))
def test_whitespace_padding_cannot_slip_past_the_raw_cap(field: str) -> None:
    # Pydantic's strip_whitespace would shrink this below the field limit, so the
    # raw cap must reject it explicitly rather than leave it to field validation.
    padded = QUESTION[field] + " " * (RAW_CAPS[field] + 1 - len(QUESTION[field]))
    with pytest.raises(ValidationError) as raised:
        _evaluate(question={**QUESTION, field: padded})
    assert raised.value.errors()[0]["type"] == "string_too_long"


@pytest.mark.parametrize("field", list(RAW_CAPS))
def test_fields_at_the_raw_cap_are_sanitised_and_validated_normally(field: str) -> None:
    removable = QUESTION[field] + chr(0x200B) * (RAW_CAPS[field] - len(QUESTION[field]))
    assert len(removable) == RAW_CAPS[field]
    question = _evaluate(question={**QUESTION, field: removable}).question
    assert getattr(question, field) == QUESTION[field]
    assert question.hidden_unicode is True
    too_long_after = "z" * (POST_LIMITS[field] + 1)  # within the raw cap, over the field limit
    with pytest.raises(ValidationError) as raised:
        _evaluate(question={**QUESTION, field: too_long_after})
    assert raised.value.errors()[0]["loc"] == ("question", field)


@pytest.mark.parametrize("char", [chr(0x202E), chr(0x2066), chr(0x200B), chr(0xE0041)])
@pytest.mark.parametrize("field", list(RAW_CAPS))
def test_hidden_unicode_within_the_raw_cap_is_still_found(field: str, char: str) -> None:
    raw = QUESTION[field][:2] + char * 20 + QUESTION[field][2:]
    question = _evaluate(question={**QUESTION, field: raw}).question
    assert question.hidden_unicode is True and char not in question.model_dump_json()


def test_benign_characters_at_the_cap_still_give_no_finding() -> None:
    topic = "Apache Spark" + chr(0xFEFF) * (RAW_CAPS["topic"] - len("Apache Spark"))
    question = _evaluate(question={**QUESTION, "topic": topic}).question
    assert question.topic == "Apache Spark" and question.hidden_unicode is False
    assert _evaluate(question=QUESTION).question.hidden_unicode is False


def test_raw_caps_apply_only_to_strings_and_leave_type_errors_unchanged() -> None:
    for value in (None, 42, ["x" * 5000]):
        with pytest.raises(ValidationError) as raised:
            _evaluate(question={**QUESTION, "question": value})
        assert raised.value.errors()[0]["loc"] == ("question", "question")


def test_raw_cap_rejection_is_fast() -> None:
    import time

    started = time.perf_counter()
    for field in RAW_CAPS:
        with pytest.raises(ValidationError):
            _evaluate(question={**QUESTION, field: "q" * 5_000_000})
    assert time.perf_counter() - started < 0.5  # was about 5.5 s per field before F1
