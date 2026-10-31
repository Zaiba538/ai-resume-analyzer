"""Section detection."""

from __future__ import annotations

import time

import pytest

from backend.parsing.sections import (
    detect_sections,
    has_contact_details,
    load_sections_config,
    section_names,
)
from tests.fixtures.documents import SYNTHETIC_RESUME_LINES

ALL_SECTIONS = (
    "contact",
    "summary",
    "skills",
    "experience",
    "education",
    "projects",
    "certifications",
)


def test_config_defines_seven_sections_in_order() -> None:
    assert section_names() == ALL_SECTIONS
    assert load_sections_config().version == 1


def test_full_synthetic_resume_has_every_section() -> None:
    assert detect_sections("\n".join(SYNTHETIC_RESUME_LINES)) == dict.fromkeys(ALL_SECTIONS, True)


def test_empty_text_has_no_sections() -> None:
    assert detect_sections("") == dict.fromkeys(ALL_SECTIONS, False)


@pytest.mark.parametrize(
    ("line", "section"),
    [
        ("WORK EXPERIENCE", "experience"),
        ("Professional Experience:", "experience"),
        ("• Technical Skills", "skills"),
        ("Skills & Tools", "skills"),
        ("## Education", "education"),
        ("Licenses & Certifications", "certifications"),
        ("Skills: Python, SQL, Docker", "skills"),
        ("  Personal Projects  ", "projects"),
        ("Career Objective", "summary"),
    ],
)
def test_heading_variants(line: str, section: str) -> None:
    assert detect_sections(f"Some intro\n{line}\nbody text")[section] is True


@pytest.mark.parametrize(
    "line",
    [
        "I have extensive experience with Python and SQL across many teams",
        "Education is something I value",
        "My projects include a forecaster",
        "skillsets",
    ],
)
def test_prose_mentions_are_not_headings(line: str) -> None:
    sections = detect_sections(line)
    assert not any(present for name, present in sections.items() if name != "contact")


@pytest.mark.parametrize(
    "text",
    [
        "jordan.sample@example.invalid",
        "+1 555 010 0199",
        "(555) 010-0199 ext",
        "+91 98765 43210",
        "linkedin.com/in/jordan-sample",
        "https://github.com/jordan-sample",
    ],
)
def test_contact_details_detected(text: str) -> None:
    assert has_contact_details(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "2021 - 2024",
        "2019-2021",
        "GPA 3.85 / 4.00",
        "Version 1.2.3.4",
        "Increased revenue by 12345678 units",
        "user at example dot com",
        "user@localhost",
        "name@domain.c0m1",
    ],
)
def test_non_contact_numbers_ignored(text: str) -> None:
    assert has_contact_details(text) is False


def test_contact_heading_also_counts() -> None:
    assert detect_sections("Contact Information\nAvailable on request")["contact"] is True


@pytest.mark.parametrize(
    "hostile",
    [
        "a@" + "a." * 20_000 + "!",
        "+" + "1 " * 20_000,
        ("Skills:" * 10_000),
        "x" * 50_000,
        "x@" * 25_000,
        "a." * 25_000,
        "1" * 50_000,
        "1 " * 25_000,
        "linkedin.com/in/" * 3_000,
    ],
)
def test_detection_is_fast_on_hostile_input(hostile: str) -> None:
    started = time.perf_counter()
    detect_sections(hostile)
    assert time.perf_counter() - started < 1.0
