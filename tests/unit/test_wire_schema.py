"""Wire schemas sent to Groq strict mode: closed, fully required, ref-free."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field

from backend.llm.structured import wire_schema
from backend.schemas.critic import CriticVerdict
from backend.schemas.interview import InterviewQuestionSet
from backend.schemas.resume import ResumeProfile
from backend.schemas.skill_gap import SkillGapReport

SCHEMAS = [ResumeProfile, InterviewQuestionSet, SkillGapReport, CriticVerdict]
FORBIDDEN = {
    "maxLength",
    "minLength",
    "maxItems",
    "minItems",
    "pattern",
    "format",
    "default",
    "$ref",
    "$defs",
}


def _objects(node: Any):
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for key, value in node.items():
            if key == "properties":
                for sub in value.values():
                    yield from _objects(sub)
            else:
                yield from _objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _objects(item)


def _keywords(node: Any):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "properties":
                for sub in value.values():
                    yield from _keywords(sub)
            else:
                yield key
                yield from _keywords(value)
    elif isinstance(node, list):
        for item in node:
            yield from _keywords(item)


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
def test_every_object_closed_and_fully_required(schema) -> None:
    wire = wire_schema(schema)
    objects = list(_objects(wire))
    assert objects
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
def test_no_unsupported_keywords_or_refs(schema) -> None:
    assert not FORBIDDEN & set(_keywords(wire_schema(schema)))


def test_enums_and_nullables_survive() -> None:
    text = json.dumps(wire_schema(CriticVerdict))
    assert '"enum": ["approve", "revise", "flagged"]' in text
    assert '{"type": "null"}' in json.dumps(wire_schema(ResumeProfile))


def test_field_names_matching_keywords_are_kept() -> None:
    class Tricky(BaseModel):
        model_config = ConfigDict(extra="forbid")
        title: str = Field(max_length=5)
        format: str
        pattern: str
        default: int

    wire = wire_schema(Tricky)
    assert set(wire["properties"]) == {"title", "format", "pattern", "default"}
    assert set(wire["required"]) == {"title", "format", "pattern", "default"}


def test_objects_closed_even_without_extra_forbid() -> None:
    """The wire builder must close objects itself, not rely on model config."""

    class Inner(BaseModel):  # default config: extra fields ignored, schema open
        name: str

    class Outer(BaseModel):
        items: list[Inner]
        note: str | None = None

    assert "additionalProperties" not in Outer.model_json_schema()
    wire = wire_schema(Outer)
    for obj in _objects(wire):
        assert obj["additionalProperties"] is False
        assert sorted(obj["required"]) == sorted(obj["properties"])
    assert set(wire["required"]) == {"items", "note"}  # defaults do not make fields optional


def test_wire_schema_is_a_defensive_copy() -> None:
    first = wire_schema(ResumeProfile)
    first["properties"].clear()
    assert wire_schema(ResumeProfile)["properties"]
