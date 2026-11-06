"""Interview API request and response schemas (M9 Phase 1; decisions Q1-Q9).

``POST /api/v1/interview/questions`` generates questions for chosen levels from
a stored report's profile. ``POST /api/v1/interview/evaluate`` evaluates one
answer to a question identified either by ``question_id`` (a question in the
stored report) or by a client-supplied ``question`` object, exactly one of the
two.

Rules:

* Every model rejects unknown fields, so a client cannot supply a level, a
  model, evaluation criteria or any other setting. Strict types reject
  coercion (``"3"``, ``3.0`` or ``true`` are not a count).
* The level of the evaluated question is derived from its validated ID
  (``intermediate-2`` is intermediate), never supplied separately.
* ``report_id`` is a plain string, neither parsed nor stripped here: the report
  store accepts only canonical UUID4 strings, so malformed, unknown and expired
  IDs keep producing the same 404.
* Client-supplied question fields are sanitised (M3) and then validated with
  the same limits as generated questions. Hidden Unicode found before
  sanitising is kept as a boolean so that it can be flagged (never rejected).
  Each raw field is first capped at twice its limit, before any sanitising.
  Only the question text is used for evaluation; the topic and rationale are
  validated but never sent to a model.
* The answer is capped here at ``ANSWER_MAX_INPUT_CHARS``, a hard bound on
  untrusted input that is applied before any processing. The configured
  ``MAX_ANSWER_CHARS`` limit applies after sanitisation and is enforced by the
  service with the typed ``answer_too_long`` error. Emptiness (``empty_answer``)
  is also decided after sanitisation, so the schema accepts an empty string.
* No response model has a field for the answer, and unknown fields are
  rejected, so the answer cannot be returned through these schemas.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    StrictInt,
    StrictStr,
    StringConstraints,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from backend.core.config import MAX_ANSWER_CHARS_LIMIT
from backend.schemas.interview import (
    DEFAULT_COUNT_PER_LEVEL,
    LEVELS,
    MAX_COUNT_PER_LEVEL,
    MIN_COUNT_PER_LEVEL,
    AnswerEvaluation,
    InterviewQuestions,
    Level,
    QuestionText,
    Rationale,
    Topic,
    normalise_levels,
)
from backend.schemas.security import SecurityFlag
from backend.security.sanitize import sanitize_text

# IDs as assigned by the server: 1-based within a level, at most
# MAX_QUESTIONS_PER_LEVEL (10) per level, no leading zeros, ASCII only.
QUESTION_ID_PATTERN = r"^(beginner|intermediate|advanced)-([1-9]|10)$"
QuestionId = Annotated[StrictStr, StringConstraints(max_length=15, pattern=QUESTION_ID_PATTERN)]

# Hard bound on the raw answer. Sanitisation can shorten text (removed
# characters) or change its length (NFKC), so the raw bound is twice the
# largest configurable limit; the configured limit is checked after sanitising.
ANSWER_MAX_INPUT_CHARS = 2 * MAX_ANSWER_CHARS_LIMIT


def level_of(question_id: str) -> Level:
    """The level encoded in a validated question ID."""
    level = question_id.split("-", 1)[0]
    if level not in LEVELS:
        raise ValueError("invalid question ID")
    return level  # type: ignore[return-value]


class _RequestModel(BaseModel):
    # No str_strip_whitespace: IDs are matched exactly, never repaired.
    model_config = ConfigDict(extra="forbid", frozen=True)


class _ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- POST /api/v1/interview/questions ------------------------------------------


class InterviewQuestionsRequest(_RequestModel):
    report_id: StrictStr
    levels: list[Level] = Field(
        default_factory=lambda: list(LEVELS), min_length=1, max_length=len(LEVELS)
    )
    count_per_level: StrictInt = Field(
        default=DEFAULT_COUNT_PER_LEVEL, ge=MIN_COUNT_PER_LEVEL, le=MAX_COUNT_PER_LEVEL
    )

    @field_validator("levels")
    @classmethod
    def _levels_unique(cls, levels: list[Level]) -> list[Level]:
        if len(set(levels)) != len(levels):
            raise ValueError("levels must not repeat")
        return levels

    @property
    def requested_levels(self) -> tuple[Level, ...]:
        """The requested levels in canonical order (beginner, intermediate, advanced)."""
        return normalise_levels(self.levels)


class InterviewQuestionsResponse(_ResponseModel):
    levels: list[Level] = Field(min_length=1, max_length=len(LEVELS))
    count_per_level: StrictInt = Field(ge=MIN_COUNT_PER_LEVEL, le=MAX_COUNT_PER_LEVEL)
    questions: InterviewQuestions
    security_flags: list[SecurityFlag]

    @model_validator(mode="after")
    def _questions_match_request(self) -> InterviewQuestionsResponse:
        if tuple(self.levels) != normalise_levels(self.levels) or len(set(self.levels)) != len(
            self.levels
        ):
            raise ValueError("levels must be distinct and in canonical order")
        for level in LEVELS:
            questions = getattr(self.questions, level)
            expected = self.count_per_level if level in self.levels else 0
            if [q.id for q in questions] != [f"{level}-{i}" for i in range(1, expected + 1)]:
                raise ValueError("questions must match the requested levels and count")
        return self


# --- POST /api/v1/interview/evaluate -------------------------------------------


CLIENT_QUESTION_TEXT_FIELDS = ("question", "topic", "rationale")

# Raw-length caps, checked before any sanitisation (audit finding F1): sanitising
# is linear in the input and runs on the event loop, so an unbounded field could
# block it for seconds. Each cap is twice the field's limit after sanitisation
# (question 500, topic 100, rationale 300), the same headroom as the answer's raw
# bound, leaving room for characters that sanitisation removes.
CLIENT_QUESTION_RAW_CAPS = {"question": 1_000, "topic": 200, "rationale": 600}


class ClientQuestion(_RequestModel):
    """A question supplied by the client (for example one returned by
    ``/interview/questions``). Untrusted: every text field is sanitised (M3)
    before the same limits as generated questions are applied.

    Whether any text field contained hidden Unicode (bidirectional controls,
    zero-width or tag characters and the other M3 suspicious categories) is
    recorded before sanitising, as a boolean only, in a private attribute that
    clients cannot set and that is never serialised. The raw text is not kept.
    """

    id: QuestionId
    question: QuestionText
    topic: Topic
    rationale: Rationale
    _hidden_unicode: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def _sanitise(cls, data: Any, handler: ModelWrapValidatorHandler[ClientQuestion]) -> Any:
        if not isinstance(data, dict):
            return handler(data)  # an existing instance keeps its finding
        # Every raw length is checked before anything is sanitised. The error names
        # the field and its cap (code constants), never the submitted value.
        for name, cap in CLIENT_QUESTION_RAW_CAPS.items():
            value = data.get(name)
            if isinstance(value, str) and len(value) > cap:
                raise PydanticCustomError(
                    "string_too_long",
                    "Field {field} should have at most {max_length} characters",
                    {"field": name, "max_length": cap},
                )
        cleaned, hidden = dict(data), False
        for name in CLIENT_QUESTION_TEXT_FIELDS:
            value = data.get(name)
            if isinstance(value, str):
                sanitized = sanitize_text(value)
                cleaned[name] = sanitized.text
                hidden = hidden or sanitized.suspicious
        question = handler(cleaned)
        question._hidden_unicode = hidden
        return question

    @property
    def hidden_unicode(self) -> bool:
        """Whether a text field contained hidden Unicode before sanitisation."""
        return self._hidden_unicode

    @property
    def level(self) -> Level:
        return level_of(self.id)


class AnswerEvaluationRequest(_RequestModel):
    report_id: StrictStr
    question_id: QuestionId | None = None
    question: ClientQuestion | None = None
    answer: StrictStr = Field(max_length=ANSWER_MAX_INPUT_CHARS)

    @model_validator(mode="after")
    def _exactly_one_question_source(self) -> AnswerEvaluationRequest:
        if (self.question_id is None) == (self.question is None):
            raise ValueError("provide exactly one of question_id or question")
        return self

    @property
    def target_question_id(self) -> str:
        """The ID of the question to evaluate, whichever way it was supplied."""
        return self.question.id if self.question is not None else str(self.question_id)

    @property
    def level(self) -> Level:
        """Derived from the validated question ID; never supplied by the client."""
        return level_of(self.target_question_id)


class AnswerEvaluationResponse(_ResponseModel):
    """The evaluation of one answer. The answer itself is never included."""

    question_id: QuestionId
    level: Level
    evaluation: AnswerEvaluation
    security_flags: list[SecurityFlag]

    @model_validator(mode="after")
    def _level_matches_question(self) -> AnswerEvaluationResponse:
        if self.level != level_of(self.question_id):
            raise ValueError("level must be the level of the question ID")
        return self
