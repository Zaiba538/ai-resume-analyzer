"""Section completeness: share of the expected resume sections that are present.

Reuses the M2 section detector and ``backend/data/expected_sections.json``
unchanged. value = present / expected * 100, rounded to two decimals; the
present and absent lists follow the configured section order.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.parsing.sections import detect_sections


@dataclass(frozen=True)
class SectionCompleteness:
    value: float
    present: tuple[str, ...]
    absent: tuple[str, ...]


def section_completeness(resume_text: str) -> SectionCompleteness:
    sections = detect_sections(resume_text)
    present = tuple(name for name, found in sections.items() if found)
    absent = tuple(name for name, found in sections.items() if not found)
    value = round(len(present) / len(sections) * 100, 2) if sections else 0.0
    return SectionCompleteness(value=value, present=present, absent=absent)
