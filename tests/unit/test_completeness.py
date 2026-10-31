"""Section completeness reuses the M2 section detector and expected-section data."""

from __future__ import annotations

from backend.parsing.sections import section_names
from backend.scoring.completeness import section_completeness
from tests.fixtures.documents import SYNTHETIC_RESUME_LINES

ALL = (
    "contact",
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
)


def test_uses_the_approved_seven_sections() -> None:
    assert section_names() == ALL


def test_full_resume_is_complete() -> None:
    result = section_completeness("\n".join(SYNTHETIC_RESUME_LINES))
    assert (result.value, result.present, result.absent) == (100.0, ALL, ())


def test_empty_resume_has_nothing() -> None:
    result = section_completeness("")
    assert (result.value, result.present, result.absent) == (0.0, (), ALL)


def test_partial_resume_keeps_configured_order() -> None:
    result = section_completeness("Education\nB.Sc.\nSkills\nPython\nEmail: a@example.invalid")
    assert result.present == ("contact", "skills", "education")
    assert result.absent == ("summary", "experience", "projects", "certifications")
    assert result.value == 42.86  # 3 / 7 * 100, rounded to two decimals


def test_one_section() -> None:
    assert section_completeness("Projects\nForecaster").value == 14.29
