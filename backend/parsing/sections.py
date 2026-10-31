"""Resume section detection.

Determines which standard sections a resume contains, as ``{name: bool}`` in
the order of ``backend/data/expected_sections.json``. Used by ``/parse`` now
and by the section-completeness metric in M5.

A section is present when a line is a heading for it. A heading line is
either exactly a known heading (case, ``&``/``and``, surrounding bullets and a
trailing colon are ignored), or starts with a known heading followed by a
colon, as in ``Skills: Python, SQL``. The contact section is also detected
from an email address, phone number or profile URL anywhere in the text.

Only booleans are returned. Matched contact details are never returned or
logged.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SECTIONS_FILE = Path(__file__).resolve().parents[1] / "data" / "expected_sections.json"

MAX_HEADING_LINE_LENGTH = 60

# Every quantifier is bounded (RFC 5321 local part <= 64, DNS labels <= 63) so
# each start position does constant work: unbounded classes made the search
# quadratic on long runs of address-like characters (ReDoS).
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){1,8}")
_PROFILE_URL = re.compile(r"(?:linkedin\.com/in/|github\.com/)[A-Za-z0-9_-]{1,100}", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<![\d.])"
    r"(?:\+\d{1,3}[ .-]?)?"  # country code
    r"(?:\(\d{2,5}\)[ .-]?|\d{2,5}[ .-])?"  # area code
    r"\d{3,5}[ .-]?\d{4,5}"  # subscriber number (incl. 5+5 Indian mobiles)
    r"(?![\d.])"
)
_PHONE_MIN_DIGITS = 10
_PHONE_MAX_DIGITS = 15
_EDGE_PUNCTUATION = " \t•·▪●◦*-–—#|:>"


class SectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    headings: tuple[str, ...] = Field(min_length=1)
    detector: Literal["contact_patterns"] | None = None


class SectionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    sections: tuple[SectionSpec, ...] = Field(min_length=1)


def _normalise_heading(text: str) -> str:
    text = text.strip(_EDGE_PUNCTUATION).lower().replace("&", " and ")
    return " ".join(text.split())


@lru_cache(maxsize=1)
def load_sections_config(path: Path = SECTIONS_FILE) -> SectionsConfig:
    config = SectionsConfig.model_validate(json.loads(path.read_text(encoding="utf-8")))
    names = [spec.name for spec in config.sections]
    if len(set(names)) != len(names):
        raise ValueError("duplicate section names in sections config")
    return config


def section_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in load_sections_config().sections)


def _has_email(text: str) -> bool:
    for match in _EMAIL.finditer(text):
        top_level = match.group().rsplit(".", 1)[-1]
        if len(top_level) >= 2 and top_level.isalpha():
            return True
    return False


def has_contact_details(text: str) -> bool:
    if _has_email(text) or _PROFILE_URL.search(text):
        return True
    for match in _PHONE.finditer(text):
        digits = sum(ch.isdigit() for ch in match.group())
        if _PHONE_MIN_DIGITS <= digits <= _PHONE_MAX_DIGITS:
            return True
    return False


def _heading_index(config: SectionsConfig) -> dict[str, str]:
    return {
        _normalise_heading(heading): spec.name
        for spec in config.sections
        for heading in spec.headings
    }


def _headings_in_line(line: str, index: dict[str, str]) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if len(stripped) <= MAX_HEADING_LINE_LENGTH:
        name = index.get(_normalise_heading(stripped))
        if name is not None:
            return name
    head, sep, _ = stripped.partition(":")
    if sep and len(head) <= MAX_HEADING_LINE_LENGTH:
        return index.get(_normalise_heading(head))
    return None


def detect_sections(text: str) -> dict[str, bool]:
    """Map every configured section name to whether the resume contains it."""
    config = load_sections_config()
    index = _heading_index(config)
    found = {name for line in text.splitlines() if (name := _headings_in_line(line, index))}
    result: dict[str, bool] = {}
    for spec in config.sections:
        present = spec.name in found
        if not present and spec.detector == "contact_patterns":
            present = has_contact_details(text)
        result[spec.name] = present
    return result
