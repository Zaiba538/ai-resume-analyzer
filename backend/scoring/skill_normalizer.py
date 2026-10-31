"""Skill vocabulary, alias normalisation and deterministic skill detection.

Vocabulary: ``backend/data/skill_aliases.json``. Each skill has a canonical
display ``name``, case-insensitive ``aliases`` and optional ``case_sensitive``
aliases for short or ambiguous terms (``Go``, ``R``, ``React``, ``Excel``)
that would otherwise match ordinary words.

Matching rules:

* Text is sanitised first (M3 rules: NFKC, invisible characters removed).
* An alias matches only as a whole term: it may not be preceded or followed by
  a letter, digit, underscore, ``+`` or ``#`` (so ``Java`` does not match inside
  ``JavaScript``, ``C`` not inside ``C++``, ``SQL`` not inside ``MySQL``).
  ``R`` and ``C`` are additionally not matched before ``&`` (``R&D``).
* Spaces and hyphens inside a multi-word alias match any run of spaces or
  hyphens (``machine-learning``, ``machine  learning``).
* Overlapping matches are resolved deterministically: the earliest match wins
  and, at the same position, the longest (``SQL Server`` over ``SQL``).
* Each canonical skill is reported once, in order of first appearance.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from backend.security.sanitize import sanitize_text

SKILLS_FILE = Path(__file__).resolve().parents[1] / "data" / "skill_aliases.json"

_BEFORE = r"(?<![\w+#])"
_AFTER = r"(?![\w+#])"
_AFTER_SINGLE_LETTER = r"(?![\w+#&])"
_SEPARATOR = re.compile(r"[\s\-]+")


class SkillSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    aliases: tuple[str, ...] = Field(min_length=1)
    case_sensitive: tuple[str, ...] = ()


class SkillVocabulary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    skills: tuple[SkillSpec, ...] = Field(min_length=1)


def _alias_pattern(alias: str) -> str:
    parts = [re.escape(part) for part in _SEPARATOR.split(alias.strip()) if part]
    body = r"[\s\-]+".join(parts)
    after = _AFTER_SINGLE_LETTER if len(alias.strip()) == 1 else _AFTER
    return f"{_BEFORE}{body}{after}"


class SkillMatcher:
    def __init__(self, vocabulary: SkillVocabulary) -> None:
        names = [skill.name for skill in vocabulary.skills]
        if len(set(names)) != len(names):
            raise ValueError("duplicate canonical skill names")
        self.vocabulary = vocabulary
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        self._alias_index: dict[str, str] = {}
        for skill in vocabulary.skills:
            # Matching uses only declared aliases: adding the display name as a
            # case-insensitive pattern would let "go" match Go or "react" React.
            for alias in skill.aliases:
                self._patterns.append(
                    (skill.name, re.compile(_alias_pattern(alias), re.IGNORECASE))
                )
            for alias in (*skill.aliases, skill.name):
                self._alias_index[_SEPARATOR.sub(" ", alias.strip()).casefold()] = skill.name
            for alias in skill.case_sensitive:
                self._patterns.append((skill.name, re.compile(_alias_pattern(alias))))

    def normalize(self, term: str) -> str | None:
        """Canonical name for a skill term (case-insensitive aliases only), or None."""
        key = _SEPARATOR.sub(" ", sanitize_text(term).text.strip()).casefold()
        return self._alias_index.get(key)

    def find(self, text: str) -> list[str]:
        """Canonical skills mentioned in ``text``, de-duplicated, in order of appearance."""
        clean = sanitize_text(text).text
        spans: list[tuple[int, int, str]] = []
        for name, pattern in self._patterns:
            spans.extend((m.start(), m.end(), name) for m in pattern.finditer(clean))
        spans.sort(key=lambda span: (span[0], -(span[1] - span[0])))
        found: list[str] = []
        covered_until = -1
        for start, end, name in spans:
            if start < covered_until:
                continue
            covered_until = end
            if name not in found:
                found.append(name)
        return found


@lru_cache(maxsize=1)
def skill_matcher(path: Path = SKILLS_FILE) -> SkillMatcher:
    vocabulary = SkillVocabulary.model_validate(json.loads(path.read_text(encoding="utf-8")))
    return SkillMatcher(vocabulary)
