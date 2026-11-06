"""AnswerEvaluation schema: fixed rubric, rating bounds, server-computed overall rating."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from backend.schemas.interview import (
    RUBRIC_CRITERIA,
    AnswerEvaluation,
    AnswerEvaluationDraft,
    Rubric,
    mean_rating,
)


def criterion(rating: int, why: str = "Clear and correct explanation.") -> dict:
    return {"rating": rating, "justification": why}


def draft_payload(*ratings: int) -> dict:
    ratings = ratings or (4, 4, 4, 4, 4)
    return {
        "rubric": {name: criterion(r) for name, r in zip(RUBRIC_CRITERIA, ratings, strict=True)},
        "strengths": ["Accurate definition"],
        "improvements": ["Mention trade-offs"],
        "model_answer_outline": ["Define the concept", "Give an example"],
    }


def test_rubric_criteria_are_the_five_approved_in_order() -> None:
    assert RUBRIC_CRITERIA == ("relevance", "technical_accuracy", "depth", "clarity", "structure")
    assert tuple(Rubric.model_fields) == RUBRIC_CRITERIA


@pytest.mark.parametrize(
    ("ratings", "expected"),
    [
        ((5, 5, 5, 5, 5), 5.0),
        ((1, 1, 1, 1, 1), 1.0),
        ((1, 2, 3, 4, 5), 3.0),
        ((4, 4, 4, 4, 3), 3.8),
        ((3, 4, 4, 4, 4), 3.8),
        ((2, 2, 3, 3, 3), 2.6),
        ((5, 4, 5, 4, 5), 4.6),
        ((1, 1, 1, 1, 2), 1.2),
    ],
)
def test_overall_rating_is_mean_of_five(ratings, expected) -> None:
    evaluation = AnswerEvaluation.from_draft(
        AnswerEvaluationDraft.model_validate(draft_payload(*ratings))
    )
    assert evaluation.overall_rating == expected
    assert evaluation.overall_rating == round(sum(ratings) / 5, 1)


@pytest.mark.parametrize(
    ("ratings", "expected"),
    [
        ([4, 5], 4.5),
        ([1, 2, 2], 1.7),  # 1.666... rounds up
        ([1, 1, 2], 1.3),  # 1.333... rounds down
        ([2, 2, 2, 3], 2.3),  # 2.25 rounds half UP (banker's rounding would give 2.2)
        ([3, 3, 3, 4], 3.3),  # 3.25 -> 3.3
        ([1, 1, 1, 2, 2, 2, 2, 2], 1.6),  # 1.625 -> 1.6
        ([5], 5.0),
    ],
)
def test_mean_rating_rounds_half_up_to_one_decimal(ratings, expected) -> None:
    assert mean_rating(ratings) == expected


def test_mean_rating_requires_ratings() -> None:
    with pytest.raises(ValueError):
        mean_rating([])


def test_overall_rating_cannot_disagree_with_rubric() -> None:
    payload = {**draft_payload(4, 4, 4, 4, 4), "overall_rating": 5.0}
    with pytest.raises(ValidationError):
        AnswerEvaluation.model_validate(payload)
    assert AnswerEvaluation.model_validate({**payload, "overall_rating": 4.0}).overall_rating == 4.0


def test_model_cannot_supply_overall_rating() -> None:
    with pytest.raises(ValidationError):
        AnswerEvaluationDraft.model_validate({**draft_payload(), "overall_rating": 5.0})


@pytest.mark.parametrize("rating", [0, 6, -1, 10])
def test_rating_bounds(rating: int) -> None:
    payload = draft_payload()
    payload["rubric"]["depth"]["rating"] = rating
    with pytest.raises(ValidationError):
        AnswerEvaluationDraft.model_validate(payload)


@pytest.mark.parametrize("rating", [4.5, 4.0, "4", True, None])
def test_rating_must_be_a_strict_integer(rating) -> None:
    payload = draft_payload()
    payload["rubric"]["clarity"]["rating"] = rating
    with pytest.raises(ValidationError):
        AnswerEvaluationDraft.model_validate(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["rubric"].pop("structure"),
        lambda p: p["rubric"].update(creativity=criterion(4)),
        lambda p: p["rubric"]["relevance"].pop("justification"),
        lambda p: p["rubric"]["relevance"].update(justification=""),
        lambda p: p["rubric"]["relevance"].update(justification="Line one.\nLine two."),
        lambda p: p["rubric"]["relevance"].update(justification="x" * 301),
        lambda p: p["rubric"]["relevance"].update(weight=2),
        lambda p: p.update(strengths=["s"] * 6),
        lambda p: p.update(improvements=["i"] * 6),
        lambda p: p.update(model_answer_outline=["o"] * 9),
        lambda p: p.update(strengths=["x" * 301]),
        lambda p: p.update(strengths=[""]),
        lambda p: p.pop("model_answer_outline"),
        lambda p: p.update(critic={"verdict": "approve"}),
    ],
)
def test_strict_validation_rejects_malformed_drafts(mutate) -> None:
    payload = copy.deepcopy(draft_payload())
    mutate(payload)
    with pytest.raises(ValidationError):
        AnswerEvaluationDraft.model_validate(payload)


def test_list_limits_inclusive() -> None:
    payload = draft_payload()
    payload.update(strengths=["s"] * 5, improvements=["i"] * 5, model_answer_outline=["o"] * 8)
    AnswerEvaluationDraft.model_validate(payload)
    AnswerEvaluationDraft.model_validate({**draft_payload(), "strengths": [], "improvements": []})
