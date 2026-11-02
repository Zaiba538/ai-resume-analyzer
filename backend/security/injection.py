"""Prompt-injection detection: a supplementary, flag-only signal.

Nothing here rejects a request (approved Rev. 3, correction 4). The real
defences are deterministic validation, system/user role separation with
escaped delimiters, tool-free models and output validation. Detection only
produces ``SecurityFlag``s for the UI notice and the Critic.

Two detectors:

* Heuristics: curated phrase patterns over sanitised, case-folded text, plus a
  squeezed view (letters and digits only) that catches spaced-out or
  punctuated variants of the most common overrides. A few Hindi and Marathi
  phrases are included; coverage outside English is limited.
* Prompt Guard (``meta-llama/llama-prompt-guard-2-86m`` on Groq): text is split
  into chunks that fit its 512-token window and scored; the highest score is
  reported. The score is uncalibrated, so a flag is raised only above a
  configurable display threshold, pending calibration in M12. At most
  ``max_chunks`` chunks are scanned per text, with heuristic hits prioritised.
  Partial coverage is reported, never hidden.

Matched text is never logged, returned or stored, only categories and scores.
"""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import groq

from backend.health.upstream import GroqUpstreamMonitor
from backend.schemas.security import FlagSource, FlagType, SecurityFlag
from backend.security.sanitize import sanitize_text

logger = logging.getLogger("backend.security.injection")

_I = re.IGNORECASE
_PATTERNS: tuple[tuple[FlagType, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass|skip)\b[^.\n]{0,40}?"
            r"\b(previous|prior|above|earlier|all|any|preceding|system|your|the)\b[^.\n]{0,20}?"
            r"\b(instructions?|prompts?|rules|directions|guidelines|context)\b",
            _I,
        ),
    ),
    ("instruction_override", re.compile(r"\bnew\s+(instructions?|task|rules)\s*:", _I)),
    (
        "role_impersonation",
        re.compile(
            r"(^|\n)\s*(system|assistant|developer)\s*:|\byou\s+are\s+now\b|\bact\s+as\s+(an?\s+)?"
            r"(ai|assistant|system|recruiter|hiring\s+manager)\b",
            _I,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|print|show|repeat|output|leak)\b[^.\n]{0,30}?\b(system\s+prompt|"
            r"hidden\s+instructions?|your\s+instructions?|initial\s+prompt)\b",
            _I,
        ),
    ),
    (
        "score_manipulation",
        re.compile(
            # No trailing \b on the numeric forms: "%" is not a word character.
            r"\b(rate|score|rank|mark|evaluate)\b[^.\n]{0,40}?(\b100\s*%|\b100\s*/\s*100\b|"
            r"\b10\s*/\s*10\b|\b(perfect|highest|maximum|top)\b)|\b(this|the)\s+candidate\s+(is|must\s+be)\s+"
            r"(perfect|the\s+best|ideal|hired)\b|\b(recommend|shortlist)\s+(this|the)\s+candidate\b",
            _I,
        ),
    ),
    (
        "delimiter_injection",
        re.compile(
            r"<\s*/?\s*(resume|job_description|profile|answer|question|agent_outputs|system|"
            r"instructions?)\s*>|"
            r"<\|[^|>\n]{1,40}\|>|\[/?(INST|SYS)\]|###\s*(instruction|system)",
            _I,
        ),
    ),
    (
        "instruction_override",
        # Normalised like the text it is matched against (nukta forms differ otherwise).
        re.compile(
            unicodedata.normalize(
                "NFKC",
                r"(पिछले|पूर्व|सभी|ऊपर\s+के)\s+(सभी\s+)?निर्देशों\s+को\s+(अनदेखा|नज़रअंदाज़|नजरअंदाज)|"
                r"(मागील|आधीच्या|सर्व)\s+(सर्व\s+)?(सूचनांकडे|सूचना)\s+(दुर्लक्ष|दुर्लक्षित)",
            )
        ),
    ),
)
_SQUEEZED_PHRASES = (
    "ignorepreviousinstructions",
    "ignoreallpreviousinstructions",
    "ignoreallinstructions",
    "disregardpreviousinstructions",
    "disregardallpreviousinstructions",
    "forgetallpreviousinstructions",
    "ignoretheaboveinstructions",
    "revealyoursystemprompt",
)
_SQUEEZE = re.compile(r"[\W_]+", re.UNICODE)


def heuristic_flag_types(text: str) -> list[FlagType]:
    """Distinct flag types whose patterns match ``text`` (sanitised first)."""
    view = sanitize_text(text).text.casefold()
    found: list[FlagType] = []
    for flag_type, pattern in _PATTERNS:
        if flag_type not in found and pattern.search(view):
            found.append(flag_type)
    squeezed = _SQUEEZE.sub("", view)
    if "instruction_override" not in found and any(p in squeezed for p in _SQUEEZED_PHRASES):
        found.append("instruction_override")
    return found


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split into chunks of at most ``max_chars``, preferring whitespace boundaries."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            boundary = max(boundary, text.rfind("\n", start + max_chars // 2, end))
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


ClassifierStatus = Literal["ok", "partial", "unavailable", "skipped"]


@dataclass(frozen=True)
class ClassifierResult:
    status: ClassifierStatus
    max_score: float | None
    chunks_scanned: int
    chunks_total: int


class PromptGuardClassifier:
    """Scores text with Prompt Guard on Groq. Never raises on upstream failure."""

    def __init__(
        self,
        client: groq.AsyncGroq,
        *,
        model: str,
        chunk_chars: int,
        max_chunks: int,
        concurrency: int,
        monitor: GroqUpstreamMonitor | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._chunk_chars = chunk_chars
        self._max_chunks = max_chunks
        self._semaphore = asyncio.Semaphore(concurrency)
        self._monitor = monitor

    def select_chunks(self, text: str) -> tuple[list[str], int]:
        chunks = chunk_text(text, self._chunk_chars)
        prioritised = sorted(
            range(len(chunks)), key=lambda i: (not heuristic_flag_types(chunks[i]), i)
        )
        return [chunks[i] for i in sorted(prioritised[: self._max_chunks])], len(chunks)

    async def _score(self, chunk: str) -> float | None:
        async with self._semaphore:
            try:
                completion = await self._client.chat.completions.create(
                    model=self._model, messages=[{"role": "user", "content": chunk}]
                )
            except groq.GroqError:
                if self._monitor is not None:
                    self._monitor.record_request_outcome(False, "classifier_error")
                return None
        if self._monitor is not None:
            self._monitor.record_request_outcome(True)
        try:
            score = float((completion.choices[0].message.content or "").strip())
        except (ValueError, IndexError, AttributeError):
            return None
        return score if 0.0 <= score <= 1.0 else None

    async def classify(self, text: str) -> ClassifierResult:
        selected, total = self.select_chunks(text)
        if not selected:
            return ClassifierResult("skipped", None, 0, 0)
        scores = [
            s for s in await asyncio.gather(*(self._score(c) for c in selected)) if s is not None
        ]
        if not scores:
            status: ClassifierStatus = "unavailable"
        elif len(scores) == total:
            status = "ok"
        else:
            status = "partial"
        result = ClassifierResult(status, max(scores) if scores else None, len(scores), total)
        logger.info(
            "prompt_guard_completed",
            extra={"upstream_status": status, "model": self._model, "node": "prompt_guard"},
        )
        return result


@dataclass(frozen=True)
class InjectionScan:
    flags: tuple[SecurityFlag, ...]
    classifier: ClassifierResult


async def scan_untrusted_text(
    text: str,
    source: FlagSource,
    *,
    classifier: PromptGuardClassifier | None,
    display_threshold: float,
) -> InjectionScan:
    """Flag-only scan of one untrusted input. Never raises, never rejects."""
    sanitized = sanitize_text(text)
    flags: list[SecurityFlag] = []
    if sanitized.suspicious:
        flags.append(SecurityFlag(type="hidden_unicode", source=source))
    flags.extend(SecurityFlag(type=t, source=source) for t in heuristic_flag_types(sanitized.text))

    if classifier is None:
        result = ClassifierResult("skipped", None, 0, 0)
    else:
        result = await classifier.classify(sanitized.text)
        if result.max_score is not None and result.max_score >= display_threshold:
            flags.append(SecurityFlag(type="classifier", source=source, score=result.max_score))

    if flags:
        logger.info(
            "injection_flags_raised",
            extra={"security_flag_types": sorted({f.type for f in flags}), "component": source},
        )
    return InjectionScan(flags=tuple(flags), classifier=result)


def merge_flags(*groups: Sequence[SecurityFlag]) -> tuple[SecurityFlag, ...]:
    """De-duplicate flags across sources while keeping first-seen order."""
    seen: dict[tuple[str, str], SecurityFlag] = {}
    for group in groups:
        for flag in group:
            key = (flag.type, flag.source)
            if key not in seen or (flag.score or 0) > (seen[key].score or 0):
                seen[key] = flag
    return tuple(seen.values())
