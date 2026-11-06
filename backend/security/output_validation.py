"""Validation of LLM output before it is stored, rendered or put into a PDF.

LLM output is untrusted: the model may have been steered by injected input.
After strict schema validation (``backend.llm.structured``) every string in
the output is:

* sanitised with the same deterministic rules as user input (control, bidi,
  invisible and private-use characters removed; NFKC);
* checked for URLs. A URL that does not occur in the source text the model
  was given is replaced with ``[link removed]`` and flagged
  (``unexpected_url``). Models must not introduce links users might follow.

The cleaned data is validated against the schema again, so length and count
limits still hold after cleaning.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from backend.schemas.security import SecurityFlag
from backend.security.sanitize import sanitize_text

LINK_PLACEHOLDER = "[link removed]"
_URL = re.compile(
    r"(?:https?://|ftp://|www\.)[^\s<>\"'()\[\]{}]{1,500}|"
    r"\b[a-z0-9][a-z0-9-]{0,62}\.(?:com|net|org|io|ai|xyz|ru|cn|in|co|me|ly|gl|link|app)"
    r"(?:/[^\s<>\"'()\[\]{}]{0,500})?",
    re.IGNORECASE,
)
_TRAILING_PUNCTUATION = ".,;:!?"


def _clean_string(value: str, sources: str, unexpected: list[str]) -> str:
    cleaned = sanitize_text(value).text

    def replace(match: re.Match[str]) -> str:
        url = match.group().rstrip(_TRAILING_PUNCTUATION)
        trailing = match.group()[len(url) :]
        if url.casefold() in sources:
            return match.group()
        unexpected.append(url)
        return LINK_PLACEHOLDER + trailing

    return _URL.sub(replace, cleaned)


def _walk(value: Any, sources: str, unexpected: list[str]) -> Any:
    if isinstance(value, str):
        return _clean_string(value, sources, unexpected)
    if isinstance(value, list):
        return [_walk(item, sources, unexpected) for item in value]
    if isinstance(value, dict):
        return {key: _walk(item, sources, unexpected) for key, item in value.items()}
    return value


def validate_llm_output[T: BaseModel](
    output: T, *, source_texts: Iterable[str]
) -> tuple[T, list[SecurityFlag]]:
    """Sanitise every string in ``output`` and neutralise links absent from the sources."""
    sources = "\n".join(sanitize_text(text).text for text in source_texts).casefold()
    unexpected: list[str] = []
    cleaned = _walk(output.model_dump(mode="python"), sources, unexpected)
    validated = type(output).model_validate(cleaned)
    flags = [SecurityFlag(type="unexpected_url", source="llm_output")] if unexpected else []
    return validated, flags
