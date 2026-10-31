"""Deterministic Unicode sanitisation of untrusted text.

Applied to every resume, job description and answer before it reaches a
prompt, and to every LLM output string before it is stored or rendered.

Steps (pure, idempotent, no I/O):

1. NFKC normalisation. Compatibility forms such as fullwidth letters,
   ligatures and styled mathematical letters become their plain equivalents,
   so look-alike text cannot hide from detection. Devanagari is unaffected
   apart from canonical nukta recomposition, which renders identically.
2. Line separators (CR, CRLF, VT, FF, NEL, U+2028, U+2029) become ``\\n``.
3. Removal of characters with no legitimate place in a resume:

   * C0/C1 control characters (``\\n`` and ``\\t`` are kept);
   * bidirectional embeddings, overrides and isolates ("Trojan Source");
   * invisible formatting characters: zero-width space, word joiner,
     invisible operators and similar (soft hyphens and BOMs are removed too,
     but counted as benign "format" because word processors emit them);
   * Unicode tag characters (U+E0000-U+E007F), used for "ASCII smuggling";
   * variation selectors supplement (U+E0100-U+E01EF), used to hide data;
   * private-use, surrogate, unassigned and noncharacter code points
     (private-use characters are typically icon-font glyphs from PDFs).

4. ZERO WIDTH JOINER / NON-JOINER are kept where they are part of the writing
   system: between letters or marks when either neighbour is outside the
   Latin, Greek and Cyrillic ranges (Devanagari conjunct control, the Marathi
   eyelash-ra, Persian ZWNJ) or inside emoji sequences. Between Latin letters
   they only serve to evade detection and are removed.
5. Runs of combining marks longer than ``MAX_COMBINING_RUN`` ("Zalgo" text) are
   truncated. Real Devanagari clusters stay far below this.

Counts of removed characters are reported so that hidden content can be flagged.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from backend.core.errors import AppError

MAX_COMBINING_RUN = 8

_NEWLINE_EQUIVALENTS = {"\r\n": "\n", "\r": "\n", "\x0b": "\n", "\x0c": "\n", "\x85": "\n"}
_LINE_SEPARATORS = {" ": "\n", " ": "\n"}
_ALLOWED_CONTROLS = frozenset({"\n", "\t"})
_JOINERS = frozenset({"‌", "‍"})
# Invisible but routinely left behind by word processors and PDF exporters.
_BENIGN_FORMAT = frozenset({"­", "﻿"})
_BIDI_CONTROLS = frozenset("؜‎‏‪‫‬‭‮⁦⁧⁨⁩")
# Below this code point (Latin, IPA, Greek, Cyrillic, Armenian) joiners have no
# orthographic use.
_JOINER_SCRIPT_THRESHOLD = 0x0590

# Removal categories that indicate deliberately hidden or reordered content.
SUSPICIOUS_CATEGORIES = frozenset(
    {"bidi_control", "zero_width", "tag_character", "variation_selector_supplement", "zalgo"}
)


class TextTooLongError(AppError):
    status_code = 422
    code = "text_too_long"
    message = "The text exceeds the maximum allowed length."


@dataclass(frozen=True)
class SanitizedText:
    text: str
    removed: Counter[str] = field(default_factory=Counter)

    @property
    def suspicious(self) -> bool:
        return any(self.removed[category] for category in SUSPICIOUS_CATEGORIES)


def _removal_category(ch: str) -> str | None:
    """Why ``ch`` must be removed, or None to keep it (joiners handled separately)."""
    code = ord(ch)
    if ch in _ALLOWED_CONTROLS:
        return None
    if 0xE0000 <= code <= 0xE007F:
        return "tag_character"
    if 0xE0100 <= code <= 0xE01EF:
        return "variation_selector_supplement"
    if ch in _BIDI_CONTROLS:
        return "bidi_control"
    if ch in _BENIGN_FORMAT:
        return "format"
    if (code & 0xFFFE) == 0xFFFE or 0xFDD0 <= code <= 0xFDEF:
        return "noncharacter"
    category = unicodedata.category(ch)
    if category == "Cc":
        return "control"
    if category == "Cf":
        return "zero_width"
    if category == "Co":
        return "private_use"
    if category == "Cs":
        return "surrogate"
    if category == "Cn":
        return "unassigned"
    return None


def _joiner_allowed(previous: str | None, following: str | None) -> bool:
    if previous is None or following is None:
        return False
    categories = (unicodedata.category(previous), unicodedata.category(following))
    if not all(c[0] in "LM" or c in ("So", "Sk") for c in categories):
        return False
    if "So" in categories:  # emoji ZWJ sequences
        return True
    return max(ord(previous), ord(following)) >= _JOINER_SCRIPT_THRESHOLD


def sanitize_text(text: str, *, max_chars: int | None = None) -> SanitizedText:
    """Normalise and strip untrusted text. Raises ``TextTooLongError`` past ``max_chars``."""
    for old, new in _NEWLINE_EQUIVALENTS.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKC", text)
    for old, new in _LINE_SEPARATORS.items():
        text = text.replace(old, new)

    removed: Counter[str] = Counter()
    kept: list[str] = []
    for ch in text:
        reason = None if ch in _JOINERS else _removal_category(ch)
        if reason is None:
            kept.append(ch)
        else:
            removed[reason] += 1

    output: list[str] = []
    combining_run = 0
    for index, ch in enumerate(kept):
        if ch in _JOINERS:
            previous = output[-1] if output else None
            following = kept[index + 1] if index + 1 < len(kept) else None
            if not _joiner_allowed(previous, following):
                removed["zero_width"] += 1
                continue
            output.append(ch)
            continue
        if unicodedata.category(ch) in ("Mn", "Me"):
            combining_run += 1
            if combining_run > MAX_COMBINING_RUN:
                removed["zalgo"] += 1
                continue
        else:
            combining_run = 0
        output.append(ch)

    result = "".join(output)
    if max_chars is not None and len(result) > max_chars:
        raise TextTooLongError()
    return SanitizedText(text=result, removed=removed)
