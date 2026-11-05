"""English-only input check for resumes and job descriptions.

The platform analyses English resumes and job descriptions only. This check
rejects text that is predominantly written in a non-Latin script (for example
a Hindi or Marathi resume in Devanagari) and accepts English text containing
occasional non-Latin words, such as a name, an address or a list of languages.

It is deterministic and uses only ``unicodedata``:

* URLs and email addresses are removed first, token by token in linear time;
  they are Latin-script whatever the language of the document.
* A word is a run of letters and combining marks, including ZWJ and ZWNJ, so
  Devanagari vowel signs and viramas stay inside their word. Numbers,
  punctuation and runs without a letter are not words.
* Each word is Latin or non-Latin by the majority script of its letters (a tie
  counts as Latin).
* Text with 20 or more words is rejected when at least half of its words are
  non-Latin. Shorter text is rejected only when at least 80% of its words, and
  at least three words, are non-Latin. Text without words is accepted.

Anything below those thresholds is accepted: a borderline document is
analysed rather than an English user being turned away. Latin-script languages
other than English are not detected. Neither the text nor the words are ever
logged or included in an error.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from backend.core.errors import AppError
from backend.schemas.common import ErrorDetail

logger = logging.getLogger("backend.parsing.language")

Source = Literal["resume", "job_description"]

SHORT_TEXT_WORDS = 20
# Shares as exact fractions (numerator, denominator), compared in integers.
SHORT_TEXT_REJECT_SHARE = (4, 5)
SHORT_TEXT_MIN_NON_LATIN_WORDS = 3
REJECT_SHARE = (1, 2)

# URL and email removal works token by token (audit finding F1). Whole-text
# patterns such as ``\b[a-z][a-z0-9+.-]*://`` backtrack quadratically on crafted
# input ("a." repeated 25,000 times took 20 s), so no regex is run across the
# text: each whitespace-delimited token is scanned in linear time, and the only
# patterns used match a fixed number of characters.
_SCHEME_SEPARATOR = "://"
_SCHEME_START = re.compile(r"[a-z]", re.IGNORECASE)
_SCHEME_CHAR = re.compile(r"[a-z0-9+.-]", re.IGNORECASE)
_WORD_CHAR = re.compile(r"\w")
_WWW = re.compile(r"\bwww\.", re.IGNORECASE)
_JOINERS = frozenset({"\u200c", "\u200d"})  # ZWNJ, ZWJ

# Latin-script letter blocks; NFKC (applied by the sanitiser) has already folded
# full-width and other compatibility forms into these.
_LATIN_RANGES = (
    (0x0041, 0x005A),
    (0x0061, 0x007A),
    (0x00C0, 0x024F),  # Latin-1 Supplement letters, Latin Extended-A and -B
    (0x0250, 0x02AF),  # IPA Extensions
    (0x1D00, 0x1DBF),  # Phonetic Extensions
    (0x1E00, 0x1EFF),  # Latin Extended Additional
    (0x2C60, 0x2C7F),  # Latin Extended-C
    (0xA720, 0xA7FF),  # Latin Extended-D
    (0xAB30, 0xAB6F),  # Latin Extended-E
)

_MESSAGES: dict[str, str] = {
    "resume": "The resume must be in English. Predominantly non-English resumes are not supported.",
    "job_description": (
        "The job description must be in English. "
        "Predominantly non-English job descriptions are not supported."
    ),
}


class UnsupportedLanguageError(AppError):
    status_code = 422
    code = "unsupported_language"
    message = "Only English content is supported."

    def __init__(self, source: Source) -> None:
        message = _MESSAGES[source]
        super().__init__(
            message,
            details=[ErrorDetail(loc=[source], type="unsupported_language", msg=message)],
        )


@dataclass(frozen=True)
class LanguageAssessment:
    latin_words: int
    non_latin_words: int

    @property
    def words(self) -> int:
        return self.latin_words + self.non_latin_words

    @property
    def predominantly_non_english(self) -> bool:
        total, non_latin = self.words, self.non_latin_words
        if total == 0:
            return False
        if total < SHORT_TEXT_WORDS:
            numerator, denominator = SHORT_TEXT_REJECT_SHARE
            return (
                non_latin >= SHORT_TEXT_MIN_NON_LATIN_WORDS
                and non_latin * denominator >= total * numerator
            )
        numerator, denominator = REJECT_SHARE
        return non_latin * denominator >= total * numerator


def _is_latin(ch: str) -> bool:
    code = ord(ch)
    return any(low <= code <= high for low, high in _LATIN_RANGES)


def _is_word_char(ch: str) -> bool:
    return ch in _JOINERS or unicodedata.category(ch)[0] in "LM"


def _url_start(token: str) -> int | None:
    """Where the first URL in ``token`` starts, or ``None``.

    A URL starts at a word boundary with ``www.``, or with a letter followed by
    scheme characters and ``://``: the same starts the earlier whole-text
    pattern found, located by walking back from each ``://``. The scheme runs
    before different ``://`` are disjoint, so the scan is linear.
    """
    starts = []
    if www := _WWW.search(token):
        starts.append(www.start())
    search_from = 0
    while (separator := token.find(_SCHEME_SEPARATOR, search_from)) != -1:
        run = separator
        while run > 0 and _SCHEME_CHAR.match(token[run - 1]):
            run -= 1
        scheme = next(
            (
                index
                for index in range(run, separator)
                if _SCHEME_START.match(token[index])
                and (index == 0 or not _WORD_CHAR.match(token[index - 1]))
            ),
            None,
        )
        if scheme is not None:
            starts.append(scheme)
            break
        search_from = separator + 1
    return min(starts) if starts else None


def _is_domain(part: str) -> bool:
    """Whether ``part`` has a dot with at least one character on each side."""
    return part.find(".", 1, len(part) - 1) != -1


def _without_emails(token: str) -> str:
    """``token`` with each ``local@domain.tld`` replaced by a space.

    Splitting at ``@`` gives the same matches as the earlier pattern
    ``[^\\s@]+@[^\\s@]+\\.[^\\s@]+``: a non-empty part, ``@``, and a following part
    with an inner dot, scanned left to right without overlap.
    """
    parts = token.split("@")
    kept: list[str] = []
    index = 0
    while index < len(parts):
        if index + 1 < len(parts) and parts[index] and _is_domain(parts[index + 1]):
            kept.append(" ")
            index += 2
        else:
            kept.append(parts[index])
            index += 1
        if index < len(parts):
            kept.append("@")
    return "".join(kept)


def _without_links(text: str) -> str:
    """``text`` with URLs (to the end of their token) and email addresses removed."""
    kept: list[str] = []
    for token in text.split():
        start = _url_start(token)
        if start is not None:
            token = token[:start]
        kept.append(_without_emails(token))
    return " ".join(kept)


def words(text: str) -> list[str]:
    """The words of ``text``: letter-and-mark runs, URLs and email addresses removed."""
    text = _without_links(text)
    found: list[str] = []
    current: list[str] = []
    for ch in text + " ":
        if _is_word_char(ch):
            current.append(ch)
            continue
        if any(unicodedata.category(c)[0] == "L" for c in current):
            found.append("".join(current))
        current = []
    return found


def _word_is_latin(word: str) -> bool:
    letters = [ch for ch in word if unicodedata.category(ch)[0] == "L"]
    latin = sum(1 for ch in letters if _is_latin(ch))
    return latin * 2 >= len(letters)


def assess_language(text: str) -> LanguageAssessment:
    latin = non_latin = 0
    for word in words(text):
        if _word_is_latin(word):
            latin += 1
        else:
            non_latin += 1
    return LanguageAssessment(latin_words=latin, non_latin_words=non_latin)


def require_english(text: str, source: Source) -> None:
    """Raise ``UnsupportedLanguageError`` if ``text`` is predominantly non-English."""
    if assess_language(text).predominantly_non_english:
        logger.info("unsupported_language", extra={"component": source})
        raise UnsupportedLanguageError(source)
