"""Validation of LLM output before it is stored, rendered or put into a PDF.

LLM output is untrusted: the model may have been steered by injected input.
After strict schema validation (``backend.llm.structured``) every string in
the output is:

* sanitised with the same deterministic rules as user input (control, bidi,
  invisible and private-use characters removed; NFKC);
* checked for links. A link that does not occur in the source text the model
  was given is replaced with ``[link removed]`` and flagged
  (``unexpected_url``). Models must not introduce links users might follow.

What counts as a link. Strong evidence only: a scheme (``https://``,
``http://``, ``ftp://``), a ``www.`` prefix, an e-mail address, or a bare
domain with a listed top-level domain (or a country second-level suffix
such as ``.co.uk`` or ``.com.au``) that is a whole token (it does not
start or end inside a longer word, so ``pd.concat``, ``React.Component`` and
``System.Collections`` are not links). A few technology names and degrees
that are also domain-shaped (``ASP.NET``, ``Socket.IO``, ``B.Com`` and so on)
are not treated as links when written bare without a path; with a scheme,
``www.`` or a path they are checked like any other link.

Comparison. A link is "known" when its normalised form (case-folded, without
scheme, ``www.`` or trailing ``/``) occurs in the normalised source text as a
whole token, so ``https://github.com/x/`` matches ``github.com/x`` in a
resume but ``x.co`` does not match ``x.com``.

Replacement always covers the whole matched token, so no word fragment is
left next to the placeholder. Only the agent name and the number of removed
links are logged, never the links themselves.

The cleaned data is validated against the schema again, so length and count
limits still hold after cleaning.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel

from backend.schemas.security import SecurityFlag
from backend.security.sanitize import sanitize_text

logger = logging.getLogger("backend.security")

LINK_PLACEHOLDER = "[link removed]"

_TLDS = r"(?:com|net|org|io|ai|xyz|ru|cn|in|co|me|ly|gl|link|app)"
# A listed TLD, or a second-level public suffix under a country code
# (``.co.uk``, ``.com.au``, ``.org.nz``); ``java.io.File`` is not one.
_SUFFIX = rf"(?:{_TLDS}|(?:co|com|net|org|ac|gov|edu)\.[a-z]{{2}})"
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_URL_CHARS = r"[^\s<>\"'()\[\]{}]"
_URL = re.compile(
    # A scheme; word characters glued in front belong to the same token.
    rf"(?P<scheme>[\w+.-]{{0,64}}(?:https?|ftp)://{_URL_CHARS}{{1,500}})"
    rf"|(?P<www>(?<![\w.-])www\.{_URL_CHARS}{{1,500}})"
    rf"|(?P<email>(?<![\w.+-])[\w.+-]{{1,64}}@(?:{_LABEL}\.)+{_SUFFIX}(?![\w-])(?!\.\w))"
    rf"|(?P<domain>(?<![\w.@-])(?:{_LABEL}\.)+{_SUFFIX}(?![\w-])(?!\.\w))"
    rf"(?P<path>[/?#]{_URL_CHARS}{{0,500}})?",
    re.IGNORECASE,
)
# Domain-shaped names that are not links when written bare, without a path.
_NOT_LINKS = frozenset(
    {"asp.net", "ado.net", "vb.net", "ml.net", "socket.io", "draw.io", "b.com", "m.com"}
)
_TRAILING_PUNCTUATION = ".,;:!?"
_SCHEME_PREFIX = re.compile(r"^(?:https?|ftp)://", re.IGNORECASE)
_SOURCE_PREFIXES = re.compile(r"(?:https?|ftp)://|(?<![\w.-])www\.", re.IGNORECASE)


def _normalise(url: str) -> str:
    """Case-folded link without scheme, ``www.`` prefix or trailing slashes."""
    url = _SCHEME_PREFIX.sub("", url.casefold())
    if url.startswith("www."):
        url = url[4:]
    return url.rstrip("/")


def _normalise_sources(texts: Iterable[str]) -> str:
    joined = "\n".join(sanitize_text(text).text for text in texts).casefold()
    return _SOURCE_PREFIXES.sub("", joined)


def _is_known(url: str, sources: str) -> bool:
    norm = _normalise(url)
    if not norm:
        return False
    pattern = rf"(?<![\w.@-]){re.escape(norm)}(?![\w-])"
    return re.search(pattern, sources) is not None


def _clean_string(value: str, sources: str, unexpected: list[str]) -> str:
    cleaned = sanitize_text(value).text

    def replace(match: re.Match[str]) -> str:
        bare = match["domain"] is not None and match["path"] is None
        if bare and match["domain"].casefold() in _NOT_LINKS:
            return match.group()
        url = match.group().rstrip(_TRAILING_PUNCTUATION)
        trailing = match.group()[len(url) :]
        if _is_known(url, sources):
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
    output: T, *, source_texts: Iterable[str], role: str | None = None
) -> tuple[T, list[SecurityFlag]]:
    """Sanitise every string in ``output`` and neutralise links absent from the sources.

    ``role`` (the agent name) is used only for the log record of removed links.
    """
    sources = _normalise_sources(source_texts)
    unexpected: list[str] = []
    cleaned = _walk(output.model_dump(mode="python"), sources, unexpected)
    validated = type(output).model_validate(cleaned)
    if not unexpected:
        return validated, []
    # Agent name and count only: links can carry personal data.
    logger.info("llm_output_links_removed", extra={"node": role, "url_findings": len(unexpected)})
    return validated, [SecurityFlag(type="unexpected_url", source="llm_output")]
