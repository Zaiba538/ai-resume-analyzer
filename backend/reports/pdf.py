"""In-memory PDF rendering of analysis reports (approved Rev. 3, M8 Phase 3).

``render_report_pdf`` turns a stored ``AnalysisReport`` into PDF bytes with
ReportLab, entirely in memory (``BytesIO``): no temporary or persistent files,
no caching, nothing kept after the call.

Content. The report is derived data only: an exact ``AnalysisReport`` is
required, so a response carrying the extracted-text preview is rejected. The
PDF shows the resume profile, the three interview-question levels, semantic
similarity, skill coverage, section completeness, the TF-IDF baseline, the ATS
status (never a number while it is not calibrated), skill gaps and
recommendations, the Critic verdict with each issue's severity and message,
and security flag types only. Wording comes from ``backend.reports.labels``;
every page has a footer with the generation and expiry times and the fixed
disclaimer ``FOOTER_DISCLAIMER``. Content is never truncated: long text wraps
and continues on later pages. Private-use, control and surrogate characters,
which have no defined appearance, are shown as U+FFFD.

Safety. Report content is untrusted: every dynamic string is XML-escaped
before it reaches a ``Paragraph``, so markup in it is shown literally. No
images, links or other external resources are ever created or loaded.
Metadata is generic (``PDF_METADATA``) and contains no report content.

Fonts and shaping. DejaVu Sans renders all text except Devanagari, which uses
Noto Sans Devanagari with HarfBuzz shaping (D1). ReportLab shapes a whole
paragraph, and shapes each space-separated word with the font of its first
character, so only paragraphs that contain Devanagari are shaped, and in them
every word has a single font: a word mixing Devanagari with characters the
Noto face also covers (punctuation, digits, joiners) is set entirely in Noto;
if a word mixes Devanagari with characters Noto lacks (for example Latin
letters with no space between), the paragraph is not shaped. Known
limitations: shaped Devanagari may not copy or search correctly (D1), Latin
words in a shaped paragraph may use ligatures, and an unshaped Devanagari
paragraph shows letters without conjunct forms.

Limits (D5, E1). The page limit is checked as each page begins and the deadline
before every flowable and page; exceeding either raises ``PdfPageLimitError``
or ``PdfRenderTimeoutError`` and no partial PDF is returned. A check cannot
interrupt a single ReportLab step that is already running, and a caller that
runs this function in a worker thread (for example with ``asyncio.to_thread``
and a timeout) cannot cancel or terminate it either: the thread finishes or
fails at its next check. Builds are serialised by a module lock because
ReportLab's shaping adds glyph mappings to the shared font objects; time spent
waiting for the lock counts against the deadline.

Determinism. Invariant mode fixes the document ID and dates, so the same
report renders to identical bytes within a process. Devanagari conjuncts get
private-use codepoints in the order a process first renders them, so bytes
(not text or layout) can differ between processes.
"""

from __future__ import annotations

import io
import re
import threading
import time
import unicodedata
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import partial
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Flowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.reports.errors import (
    PdfPageLimitError,
    PdfRenderError,
    PdfRendererUnavailableError,
    PdfRenderTimeoutError,
)
from backend.reports.fonts import default_registry, font_name, split_script_runs
from backend.reports.labels import LABELS, label
from backend.schemas.critic import CriticIssue
from backend.schemas.interview import LEVELS, InterviewQuestions
from backend.schemas.match import MatchResult
from backend.schemas.report import AnalysisReport, CriticSummary
from backend.schemas.resume import ResumeProfile
from backend.schemas.security import SecurityFlag
from backend.schemas.skill_gap import SkillGapReport

FOOTER_DISCLAIMER = "AI-generated, unverified"
PDF_METADATA = {
    "title": label("document", "title"),
    "author": "Resume Intelligence Platform",
    "subject": "AI-generated resume analysis",
    "creator": "Resume Intelligence Platform",
    "keywords": [],
}

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 18 * mm
FOOTER_Y = 10 * mm
FRAME_WIDTH = PAGE_WIDTH - 2 * MARGIN
LABEL_COLUMN = 52 * mm

REPLACEMENT_CHARACTER = chr(0xFFFD)
BULLET = chr(0x2022)
_WORD_SEPARATORS = re.compile(r"([ \t\n\r\f\v]+)")
_RENDER_LOCK = threading.Lock()
_monotonic = time.monotonic  # replaced in tests to simulate a slow render

_REGULAR = font_name("default")
_BOLD = font_name("default", bold=True)
_BASE = ParagraphStyle(
    "body", fontName=_REGULAR, fontSize=9.5, leading=13, spaceAfter=2, bulletFontName=_REGULAR
)
_STYLES = {
    "body": _BASE,
    "title": ParagraphStyle("title", parent=_BASE, fontName=_BOLD, fontSize=18, leading=23),
    "h1": ParagraphStyle(
        "h1", parent=_BASE, fontName=_BOLD, fontSize=13.5, leading=17, spaceBefore=12, spaceAfter=4
    ),
    "h2": ParagraphStyle(
        "h2", parent=_BASE, fontName=_BOLD, fontSize=11, leading=14, spaceBefore=7, spaceAfter=2
    ),
    "h3": ParagraphStyle("h3", parent=_BASE, fontName=_BOLD, spaceBefore=3, spaceAfter=1),
    "bullet": ParagraphStyle("bullet", parent=_BASE, leftIndent=12, bulletIndent=2),
    "note": ParagraphStyle("note", parent=_BASE, fontSize=8.5, leading=11, textColor=colors.grey),
    "cell": ParagraphStyle("cell", parent=_BASE, spaceAfter=0),
    "cell_label": ParagraphStyle("cell_label", parent=_BASE, fontName=_BOLD, spaceAfter=0),
}
_SHAPED_STYLES = {
    name: ParagraphStyle(f"{name}-shaped", parent=style, shaping=1)
    for name, style in _STYLES.items()
}
_TABLE_STYLE = TableStyle(
    [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
)


def render_report_pdf(report: AnalysisReport, *, max_pages: int, deadline_seconds: float) -> bytes:
    """Render ``report`` to PDF bytes, or raise; never returns a partial PDF."""
    if type(report) is not AnalysisReport:
        raise TypeError("render_report_pdf accepts an AnalysisReport only")
    if max_pages < 1 or not deadline_seconds > 0:
        raise ValueError("max_pages and deadline_seconds must be positive")
    deadline = _monotonic() + deadline_seconds
    if not default_registry().ready:
        raise PdfRendererUnavailableError()
    story = _story(report)
    remaining = deadline - _monotonic()
    if remaining <= 0 or not _RENDER_LOCK.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
        raise PdfRenderTimeoutError(deadline_seconds)
    try:
        buffer = io.BytesIO()
        document = _ReportDocument(
            buffer, max_pages=max_pages, deadline=deadline, deadline_seconds=deadline_seconds
        )
        try:
            document.build(story, canvasmaker=partial(_FooterCanvas, footer=_footer_text(report)))
        except Exception:
            # ReportLab re-creates exceptions raised in its page hooks (annotateException),
            # so the typed error recorded by the hook is the one raised.
            if document.failure is not None:
                raise document.failure from None
            raise
        document.check_deadline()
        return buffer.getvalue()
    finally:
        _RENDER_LOCK.release()


class _ReportDocument(SimpleDocTemplate):
    """A4 document that enforces the page limit and the deadline while building."""

    def __init__(
        self, buffer: io.BytesIO, *, max_pages: int, deadline: float, deadline_seconds: float
    ) -> None:
        super().__init__(
            buffer,
            pagesize=A4,
            leftMargin=MARGIN,
            rightMargin=MARGIN,
            topMargin=MARGIN,
            bottomMargin=MARGIN,
            invariant=1,
            **PDF_METADATA,
        )
        self._max_pages = max_pages
        self._deadline = deadline
        self._deadline_seconds = deadline_seconds
        self.failure: PdfRenderError | None = None

    def _fail(self, error: PdfRenderError) -> None:
        self.failure = error
        raise error

    def check_deadline(self) -> None:
        if _monotonic() > self._deadline:
            self._fail(PdfRenderTimeoutError(self._deadline_seconds))

    def beforePage(self) -> None:  # noqa: N802 - ReportLab hook name
        if self.page > self._max_pages:
            self._fail(PdfPageLimitError(self._max_pages))
        self.check_deadline()

    def filterFlowables(self, flowables) -> None:  # noqa: N802 - ReportLab hook name
        self.check_deadline()


class _FooterCanvas(Canvas):
    """Draws the footer, with "Page n of N", once the page count is known."""

    def __init__(self, *args, footer: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._footer = footer
        self._page_states: list[dict] = []

    def showPage(self) -> None:  # noqa: N802 - ReportLab API
        self._page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        total = len(self._page_states)
        for number, state in enumerate(self._page_states, start=1):
            self.__dict__.update(state)
            self.saveState()
            self.setFont(_REGULAR, 7.5)
            self.setFillColor(colors.grey)
            self.drawString(MARGIN, FOOTER_Y, self._footer)
            page = f"{label('document', 'page')} {number} {label('document', 'of')} {total}"
            self.drawRightString(PAGE_WIDTH - MARGIN, FOOTER_Y, page)
            self.restoreState()
            super().showPage()
        super().save()


def _footer_text(report: AnalysisReport) -> str:
    return (
        f"{label('document', 'generated_at')} {_timestamp(report.created_at)}  |  "
        f"{label('document', 'expires_at')} {_timestamp(report.expires_at)}  |  "
        f"{FOOTER_DISCLAIMER}"
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")


# --- Markup -------------------------------------------------------------------


def _clean(text: str) -> str:
    """Show characters without a defined appearance as U+FFFD (never dropped)."""
    return "".join(REPLACEMENT_CHARACTER if _unrenderable(ch) else ch for ch in text)


def _unrenderable(char: str) -> bool:
    category = unicodedata.category(char)
    return category in ("Co", "Cs") or (category == "Cc" and char not in "\t\n\r")


def _run_markup(role: str, text: str) -> str:
    if role == "default":  # the paragraph style's DejaVu font
        return escape(text)
    return f'<font name="{font_name("devanagari")}">{escape(text)}</font>'


def paragraph_markup(text: str) -> tuple[str, bool]:
    """Escaped Paragraph markup for untrusted ``text`` and whether to shape it.

    Shaped markup never contains a space-separated word with two fonts.
    """
    runs = split_script_runs(text)
    if not any(run.role == "devanagari" for run in runs):
        return escape(text), False
    noto = pdfmetrics.getFont(font_name("devanagari")).face.charToGlyph
    parts = []
    for index, part in enumerate(_WORD_SEPARATORS.split(text)):
        word_runs = split_script_runs(part)
        if index % 2 or len({run.role for run in word_runs}) <= 1:
            parts.append("".join(_run_markup(run.role, run.text) for run in word_runs))
        elif all(ord(ch) in noto for ch in part):
            parts.append(_run_markup("devanagari", part))
        else:  # a word would need two fonts: shaping it would use the wrong glyphs
            return "".join(_run_markup(run.role, run.text) for run in runs), False
    return "".join(parts), True


def _para(text: str, style: str = "body", **kwargs) -> Paragraph:
    markup, shaped = paragraph_markup(_clean(text))
    return Paragraph(markup, (_SHAPED_STYLES if shaped else _STYLES)[style], **kwargs)


def _fixed(text: str, style: str, **kwargs) -> Paragraph:
    """A paragraph of fixed label text (escaped all the same)."""
    return Paragraph(escape(text), _STYLES[style], **kwargs)


def _heading(key: str) -> Paragraph:
    return _fixed(label("sections", key), "h1")


def _message(key: str) -> Paragraph:
    return _fixed(label("messages", key), "body")


def _note(group: str, key: str) -> Paragraph:
    return _fixed(label(group, key), "note")


def _value(text: str | None) -> str:
    return text if text else label("messages", "none")


def _fields(rows: Sequence[tuple[str, str]]) -> Table:
    """Label/value rows; values are short (bounded by the schema), so rows fit a page."""
    data = [[_fixed(name, "cell_label"), _para(_value(value), "cell")] for name, value in rows]
    table = Table(data, colWidths=[LABEL_COLUMN, FRAME_WIDTH - LABEL_COLUMN], hAlign="LEFT")
    table.setStyle(_TABLE_STYLE)
    return table


def _inline(items: Sequence[str]) -> Paragraph:
    return _para(", ".join(items)) if items else _message("none")


def _bullets(items: Sequence[str]) -> list[Flowable]:
    if not items:
        return [_message("none")]
    return [_para(item, "bullet", bulletText=BULLET) for item in items]


def _percent(value: float) -> str:
    return f"{value:.2f}%"


# --- Sections -----------------------------------------------------------------


def _story(report: AnalysisReport) -> list[Flowable]:
    created, expires = _timestamp(report.created_at), _timestamp(report.expires_at)
    story: list[Flowable] = [
        _fixed(label("document", "title"), "title"),
        _fixed(
            f"{label('document', 'generated_at')}: {created}    "
            f"{label('document', 'expires_at')}: {expires}",
            "note",
        ),
        _fixed(label("messages", "disclaimer"), "note"),
    ]
    story += _profile_section(report.resume_profile)
    story += _questions_section(report.interview_questions)
    story += _match_section(report.match)
    story += _skill_gap_section(report.skill_gap)
    story += _critic_section(report.critic)
    story += _security_section(report.security_flags)
    return story


def _profile_section(profile: ResumeProfile) -> list[Flowable]:
    out: list[Flowable] = [_heading("resume_profile")]
    out += [_fixed(label("profile", "skills"), "h2"), _inline(profile.skills)]
    out.append(_fixed(label("profile", "education"), "h2"))
    if not profile.education:
        out.append(_message("none"))
    for entry in profile.education:
        rows = [
            (label("profile", "qualification"), entry.qualification),
            (label("profile", "institution"), entry.institution),
            (label("profile", "year"), entry.year),
        ]
        out += [_fields(rows), Spacer(1, 4)]
    out.append(_fixed(label("profile", "experience"), "h2"))
    if not profile.experience:
        out.append(_message("none"))
    for job in profile.experience:
        rows = [
            (label("profile", "role"), job.role),
            (label("profile", "organization"), job.organization),
            (label("profile", "period"), job.period),
        ]
        out += [_fields(rows), _fixed(label("profile", "highlights"), "h3")]
        out += _bullets(job.highlights)
        out.append(Spacer(1, 4))
    out.append(_fixed(label("profile", "projects"), "h2"))
    out += _bullets(profile.projects)
    out.append(_fixed(label("profile", "certifications"), "h2"))
    out += _bullets(profile.certifications)
    return out


def _questions_section(questions: InterviewQuestions) -> list[Flowable]:
    out: list[Flowable] = [_heading("interview_questions")]
    for level in LEVELS:
        out.append(_fixed(label("interview_levels", level), "h2"))
        items = getattr(questions, level)
        if not items:
            out.append(_message("none"))
        for number, question in enumerate(items, start=1):
            out.append(_para(question.question, "bullet", bulletText=f"{number}."))
            rows = [(label("interview", "topic"), question.topic)]
            if question.rationale:
                rows.append((label("interview", "rationale"), question.rationale))
            out += [_fields(rows), Spacer(1, 3)]
    return out


def _match_section(match: MatchResult | None) -> list[Flowable]:
    out: list[Flowable] = [_heading("match")]
    if match is None:
        return [*out, _message("no_job_description")]
    metrics = match.metrics
    coverage, completeness = metrics.skill_coverage, metrics.section_completeness
    sections = LABELS["resume_sections"]
    out += [
        _fields(
            [(label("metrics", "semantic_similarity"), _percent(metrics.semantic_similarity.value))]
        ),
        _note("metrics", "semantic_similarity_note"),
        _fields(
            [
                (label("metrics", "skill_coverage"), _percent(coverage.value)),
                (label("metrics", "jd_skills_total"), str(coverage.jd_skills_total)),
            ]
        ),
        _fixed(label("metrics", "matched"), "h3"),
        _inline(coverage.matched),
        _fixed(label("metrics", "missing"), "h3"),
        _inline(coverage.missing),
        _fields(
            [
                (label("metrics", "section_completeness"), _percent(completeness.value)),
                (
                    label("metrics", "present"),
                    ", ".join(sections.get(n, n) for n in completeness.present),
                ),
                (
                    label("metrics", "absent"),
                    ", ".join(sections.get(n, n) for n in completeness.absent),
                ),
            ]
        ),
        _heading("baseline"),
        _fields(
            [
                (
                    label("baseline", "tfidf_keyword_match"),
                    _percent(match.baseline.tfidf_keyword_match),
                )
            ]
        ),
        _note("baseline", "tfidf_keyword_match_note"),
        # The ATS status only: no score is shown while it is not calibrated.
        _heading("ats_score"),
        _fields([(label("ats", "status"), label("ats", match.ats_score.status))]),
        _note("ats", "not_calibrated_note"),
    ]
    return out


def _skill_gap_section(gap: SkillGapReport | None) -> list[Flowable]:
    out: list[Flowable] = [_heading("skill_gap")]
    if gap is None:
        return [*out, _message("no_job_description")]
    out += [_fixed(label("skill_gap", "matching_skills"), "h3"), _inline(gap.matching_skills)]
    out += [_fixed(label("skill_gap", "missing_skills"), "h3"), _inline(gap.missing_skills)]
    out.append(_fixed(label("skill_gap", "recommendations"), "h3"))
    return out + _bullets(gap.recommendations)


def _critic_section(critic: CriticSummary) -> list[Flowable]:
    out: list[Flowable] = [
        _heading("critic"),
        _fields(
            [
                (label("critic", "verdict"), label("critic_verdicts", critic.verdict)),
                (label("critic", "revisions"), str(critic.revisions)),
            ]
        ),
        _fixed(label("critic", "issues"), "h3"),
    ]
    if not critic.issues:
        return [*out, _message("no_issues")]
    return [*out, _issues_table(critic.issues)]


def _issues_table(issues: Sequence[CriticIssue]) -> Table:
    header = [
        _fixed(label("critic", key), "cell_label") for key in ("severity", "agent", "message")
    ]
    rows = [
        [
            _fixed(label("severities", issue.severity), "cell"),
            _fixed(label("agents", issue.agent), "cell"),
            _para(issue.message, "cell"),
        ]
        for issue in issues
    ]
    widths = [22 * mm, 36 * mm, FRAME_WIDTH - 58 * mm]
    table = Table([header, *rows], colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(_TABLE_STYLE)
    return table


def _security_section(flags: Sequence[SecurityFlag]) -> list[Flowable]:
    out: list[Flowable] = [_heading("security")]
    types = list(dict.fromkeys(flag.type for flag in flags))  # types only, each once
    if not types:
        return [*out, _message("no_security_flags")]
    out.append(_message("security_notice"))
    for flag_type in types:
        out.append(_fixed(label("security_flags", flag_type), "bullet", bulletText=BULLET))
    return out
