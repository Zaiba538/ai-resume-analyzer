"""M8 Phase 3: in-memory PDF renderer (content, safety, limits, determinism)."""

from __future__ import annotations

import builtins
import io
import json
import re
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

import pytest
from pypdf import PdfReader
from reportlab.platypus import Paragraph
from reportlab.platypus.paragraph import _getFragWords

from backend.reports import pdf
from backend.reports.errors import (
    PdfPageLimitError,
    PdfRenderError,
    PdfRendererUnavailableError,
    PdfRenderTimeoutError,
)
from backend.reports.fonts import FontRegistry, default_registry
from backend.reports.labels import LABELS, label
from backend.reports.pdf import FOOTER_DISCLAIMER, PDF_METADATA, render_report_pdf
from backend.schemas.report import AnalysisReport, AnalysisResponse
from backend.storage.report_store import ReportStamp
from tests.fixtures.reports import report_data, synthetic_report

ROOT = Path(__file__).resolve().parents[2]
CREATED = datetime(2026, 11, 2, 10, 15, tzinfo=UTC)
STAMP = ReportStamp(
    uuid.UUID("3f6c1f0e-8c1a-4c55-9a43-2b1f4f6f2a10"), CREATED, CREATED + timedelta(hours=1)
)
LIMITS = {"max_pages": 100, "deadline_seconds": 15}
MIXED = "Python और SQL (हिन्दी) मराठी-English"


@pytest.fixture(autouse=True)
def _fonts_registered() -> None:
    assert default_registry().register()


def _report(**overrides) -> AnalysisReport:
    return synthetic_report(STAMP, **overrides)


def _with(**changes) -> AnalysisReport:
    data = report_data(STAMP)
    for key, value in changes.items():
        data[key] = value
    return AnalysisReport.model_validate(
        {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in data.items()}
    )


def _profile(**fields) -> dict:
    base = report_data(STAMP)["resume_profile"].model_dump()
    return {**base, **fields}


def _render(report: AnalysisReport, **limits) -> bytes:
    return render_report_pdf(report, **{**LIMITS, **limits})


def _pages(data: bytes) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages]


def _text(data: bytes) -> str:
    return "\n".join(_pages(data))


def _flat(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _between(text: str, start: str, end: str) -> str:
    return text.split(start, 1)[1].split(end, 1)[0]


def _font_names(data: bytes) -> set[str]:
    names = set()
    for page in PdfReader(io.BytesIO(data)).pages:
        for ref in page["/Resources"]["/Font"].values():
            names.add(str(ref.get_object()["/BaseFont"]).lstrip("/").split("+")[-1])
    return names


# --- Validity and content ---------------------------------------------------------


def test_pdf_is_valid_and_rendered_in_memory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def no_writes(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            raise AssertionError(f"renderer opened a file for writing: {file!r}")
        return real_open(file, mode, *args, **kwargs)

    def no_temp_files(*args, **kwargs):
        raise AssertionError("renderer created a temporary file")

    monkeypatch.setattr(builtins, "open", no_writes)
    for name in ("mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryFile"):
        monkeypatch.setattr(tempfile, name, no_temp_files)
    data = _render(_report())
    assert isinstance(data, bytes)
    assert data.startswith(b"%PDF-") and data.rstrip().endswith(b"%%EOF")
    assert len(PdfReader(io.BytesIO(data)).pages) >= 1
    assert list(tmp_path.iterdir()) == []


def test_every_section_is_rendered_with_its_labels() -> None:
    text = _flat(_text(_render(_report())))
    for key in LABELS["sections"]:
        assert _flat(label("sections", key)) in text, key
    expected = [
        label("document", "title"),
        label("messages", "disclaimer"),
        # resume profile
        "Python, SQL, Apache Spark",
        label("profile", "qualification"),
        "B.Sc. Computer Science",
        "Example State University",
        "2020",
        "Data Engineer",
        "Acme Analytics (fictional)",
        "2021 - 2024",
        "Built batch ETL pipelines in Apache Spark.",
        "Inventory forecaster",
        # interview questions, all three levels
        *(label("interview_levels", level) for level in ("beginner", "intermediate", "advanced")),
        "What is a Python list?",
        "How would you tune a Spark job?",
        "Design a fault-tolerant ETL pipeline?",
        label("interview", "topic"),
        label("interview", "rationale"),
        "Listed skill.",
        # match metrics, baseline and ATS status
        label("metrics", "semantic_similarity"),
        "61.50%",
        label("metrics", "semantic_similarity_note"),
        label("metrics", "skill_coverage"),
        "50.00%",
        label("metrics", "jd_skills_total"),
        "Kubernetes, Airflow",
        label("metrics", "section_completeness"),
        "80.00%",
        "Skills, Experience, Education, Projects",
        label("baseline", "tfidf_keyword_match"),
        "22.00%",
        label("ats", "not_calibrated"),
        label("ats", "not_calibrated_note"),
        # skill gap
        label("skill_gap", "matching_skills"),
        label("skill_gap", "recommendations"),
        "Build and schedule a small Airflow DAG.",
        # critic
        label("critic_verdicts", "approve"),
        label("severities", "low"),
        label("agents", "interview"),
        "Minor wording.",
        # security
        label("messages", "security_notice"),
        label("security_flags", "instruction_override"),
    ]
    for item in expected:
        assert _flat(item) in text, item


def test_footer_on_every_page_has_times_disclaimer_and_page_numbers() -> None:
    pages = _pages(_render(_report()))
    for number, page in enumerate(pages, start=1):
        assert FOOTER_DISCLAIMER in page
        assert "AI-generated, unverified" in page  # exact required wording
        assert "Generated 2026-11-02 10:15 UTC" in page
        assert "Expires 2026-11-02 11:15 UTC" in page
        assert f"Page {number} of {len(pages)}" in page


def test_long_disclaimer_label_is_kept_separately() -> None:
    text = _text(_render(_report()))
    assert label("messages", "disclaimer") in text
    assert label("messages", "disclaimer") != FOOTER_DISCLAIMER


def test_report_without_job_description() -> None:
    text = _flat(_text(_render(_report(with_job=False))))
    assert text.count(_flat(label("messages", "no_job_description"))) == 2  # match and skill gap
    for absent in (
        label("metrics", "semantic_similarity"),
        label("baseline", "tfidf_keyword_match"),
        label("sections", "ats_score"),
        label("skill_gap", "matching_skills"),  # "Recommendations" is also in a heading
    ):
        assert _flat(absent) not in text


def test_ats_section_shows_status_and_no_number() -> None:
    text = _text(_render(_report()))
    ats = _between(text, label("sections", "ats_score"), label("sections", "skill_gap"))
    assert label("ats", "not_calibrated") in ats
    assert not re.search(r"\d", ats), ats


def test_empty_values_are_shown_as_none() -> None:
    report = _with(
        resume_profile=_profile(skills=[], education=[], experience=[], projects=[]),
        critic={"verdict": "revise", "issues": [], "revisions": 2},
        security_flags=[],
    )
    text = _text(_render(report))
    assert text.count(label("messages", "none")) >= 5
    assert label("messages", "no_issues") in text
    assert label("messages", "no_security_flags") in text
    assert label("critic_verdicts", "revise") in text


def test_security_flags_are_shown_by_type_only_and_once() -> None:
    flags = [
        {"type": "hidden_unicode", "source": "resume", "score": 0.97, "calibrated": False},
        {"type": "hidden_unicode", "source": "job_description"},
        {"type": "classifier", "source": "resume", "score": 0.91},
    ]
    text = _text(_render(_with(security_flags=flags)))
    section = text.split(label("sections", "security"), 1)[1]
    assert section.count(label("security_flags", "hidden_unicode")) == 1
    assert label("security_flags", "classifier") in section
    for hidden in ("0.97", "0.91", "job_description", "resume"):
        assert hidden not in section.split("Generated")[0]


def test_critic_issues_show_severity_area_and_message() -> None:
    issues = [
        {"agent": "resume", "field": "skills", "severity": "high", "message": "Skill not in text."},
        {"agent": "skill_gap", "field": "x", "severity": "medium", "message": "Vague advice."},
    ]
    text = _text(_render(_with(critic={"verdict": "flagged", "issues": issues, "revisions": 3})))
    section = _flat(text.split(label("sections", "critic"), 1)[1])
    for item in (
        label("critic_verdicts", "flagged"),
        "3",
        label("severities", "high"),
        label("agents", "resume"),
        "Skill not in text.",
        label("severities", "medium"),
        label("agents", "skill_gap"),
        "Vague advice.",
    ):
        assert _flat(item) in section, item


# --- Raw text, preview and personal data ------------------------------------------


def test_response_with_preview_is_rejected() -> None:
    canary = "CANARYpreview91c2"
    response = AnalysisResponse.model_validate({**_report().model_dump(), "preview": canary})
    with pytest.raises(TypeError):
        render_report_pdf(response, **LIMITS)


@pytest.mark.parametrize("value", [None, {}, "report", report_data(STAMP)])
def test_non_report_input_is_rejected(value) -> None:
    with pytest.raises(TypeError):
        render_report_pdf(value, **LIMITS)


@pytest.mark.parametrize(("max_pages", "deadline"), [(0, 15), (-1, 15), (100, 0), (100, -1)])
def test_invalid_limits_are_rejected(max_pages: int, deadline: float) -> None:
    with pytest.raises(ValueError):
        render_report_pdf(_report(), max_pages=max_pages, deadline_seconds=deadline)


@pytest.mark.asyncio
async def test_raw_resume_and_job_description_never_reach_the_pdf() -> None:
    from tests.integration.test_analysis_workflow import (
        JOB,
        NO_ISSUES,
        RESUME,
        scripted,
        workflow_for,
    )

    canary = "CANARYpdf5d1e"
    workflow, _ = await workflow_for(scripted(NO_ISSUES))
    report = await workflow.run(
        resume_text=f"{RESUME}\nHobby: {canary}-resume",
        job_description=f"{JOB} {canary}-jd",
    )
    data = _render(report)
    assert canary not in _text(data)
    assert canary.encode() not in data
    assert canary not in json.dumps(dict(PdfReader(io.BytesIO(data)).metadata))


def test_metadata_is_generic() -> None:
    reader = PdfReader(io.BytesIO(_render(_report())))
    metadata = dict(reader.metadata)
    assert metadata == {
        "/Title": "Resume Analysis Report",
        "/Author": "Resume Intelligence Platform",
        "/Subject": "AI-generated resume analysis",
        "/Creator": "Resume Intelligence Platform",
        "/Producer": "ReportLab PDF Library - (opensource)",
        "/Keywords": "",
        "/CreationDate": "D:20000101000000+00'00'",
        "/ModDate": "D:20000101000000+00'00'",
        "/Trapped": "/False",
    }
    assert "/Metadata" not in reader.trailer["/Root"]  # no XMP packet
    assert PDF_METADATA["title"] == label("document", "title")


def test_metadata_and_bytes_contain_no_report_content_or_identifiers() -> None:
    data = _render(_report())
    metadata = json.dumps(dict(PdfReader(io.BytesIO(data)).metadata))
    for value in ("Example State University", "Acme", "Data Engineer", "Python", "2026-11-02"):
        assert value not in metadata
    report_id = str(STAMP.report_id)
    assert report_id not in _text(data)
    assert report_id.encode() not in data
    assert report_id.replace("-", "").encode() not in data


# --- Markup injection and external resources ---------------------------------------

PAYLOADS = [
    "<b>not bold</b> <i>not italic</i>",
    '<img src="https://example.invalid/tracker.png" width="10" height="10"/>',
    '<a href="https://example.invalid/phish">click</a>',
    '<link href="javascript:alert(1)">x</link> <a name="anchor"/>',
    '<font name="Courier" color="red" size="40">x</font>',
    "&amp; &lt; &#65; &#x42; &nbsp; &unknown; & < >",
    "]]><!-- comment --><![CDATA[x]]>",
    "<para><br/></para><onDraw name='x'/><seq id='s'/><index item='i'/>",
    "<b>unclosed <i>tags",
    "<span style='x'>हिन्दी <img src='https://example.invalid/d.png'/></span>",
]


def _injected_report(payload: str) -> AnalysisReport:
    data = report_data(STAMP)
    profile = data["resume_profile"].model_dump()
    profile["skills"] = [payload]
    profile["projects"] = [payload]
    profile["experience"][0]["highlights"] = [payload]
    questions = data["interview_questions"].model_dump()
    questions["beginner"][0]["question"] = f"Q {payload}"
    critic = data["critic"].model_dump()
    critic["issues"][0]["message"] = payload
    gap = data["skill_gap"].model_dump()
    gap["recommendations"] = [payload]
    data.update(
        resume_profile=profile,
        interview_questions=questions,
        critic=critic,
        skill_gap=gap,
    )
    return AnalysisReport.model_validate(
        {k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in data.items()}
    )


@pytest.mark.parametrize("payload", PAYLOADS)
def test_markup_in_report_content_is_rendered_literally(payload: str) -> None:
    data = _render(_injected_report(payload))
    text = _flat(_text(data))
    if "हिन्दी" in payload:  # Devanagari may not extract reliably (D1): check the markup parts
        literal = [_flat(part) for part in payload.split("हिन्दी")]
    else:
        literal = [_flat(payload)]
    for part in literal:
        # skills, project, highlight, question, critic issue and recommendation
        assert text.count(part) >= 6, part
    assert _font_names(data) <= {
        "Helvetica",  # ReportLab's initial canvas font, never used for text
        "DejaVuSans",
        "DejaVuSans-Bold",
        "NotoSansDevanagari-Regular",
    }


@pytest.mark.parametrize("payload", PAYLOADS)
def test_no_links_images_or_external_resources(payload: str) -> None:
    data = _render(_injected_report(payload))
    reader = PdfReader(io.BytesIO(data))
    for page in reader.pages:
        assert "/Annots" not in page
        assert "/XObject" not in page["/Resources"]
    # "/ImageB /ImageC /ImageI" in ReportLab's standard /ProcSet are not images.
    for marker in (b"/URI", b"/Link", b"/Launch", b"/JavaScript", b"/Subtype /Image", b"/GoToR"):
        assert marker not in data, marker
    assert b"/XObject" not in data
    assert b"example.invalid/tracker" not in data
    assert b"javascript:" not in data


def test_rendering_makes_no_network_connections(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("renderer attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for payload in PAYLOADS[1:4]:
        _render(_injected_report(payload))


def test_unrenderable_characters_are_replaced_not_dropped() -> None:
    odd = "a" + chr(0) + "b" + chr(0x1B) + "c" + chr(0xE000) + "d" + chr(0xF0000) + "e"
    text = _text(_render(_with(resume_profile=_profile(skills=[odd]))))
    replacement = pdf.REPLACEMENT_CHARACTER
    assert f"a{replacement}b{replacement}c{replacement}d{replacement}e" in text


# --- Devanagari shaping and mixed scripts --------------------------------------------


def test_devanagari_and_mixed_scripts_render_with_noto() -> None:
    profile = _profile(
        skills=["हिन्दी", "मराठी", MIXED, "क्षमता ज्ञान श्री"],
        projects=["महाराष्ट्र विकास प्रकल्प", "Pythonआधारित tool"],
    )
    data = _render(_with(resume_profile=profile))
    assert {"NotoSansDevanagari-Regular", "DejaVuSans"} <= _font_names(data)
    text = _text(data)
    for latin in ("Python", "SQL", "English", "tool"):
        assert latin in text


@pytest.mark.parametrize(
    ("text", "shaped", "fonts"),
    [
        ("Python & SQL", False, set()),
        ("हिन्दी", True, {"NotoSansDevanagari"}),
        ("(हिन्दी),", True, {"NotoSansDevanagari"}),  # punctuation stays with the word in Noto
        ("Python और SQL", True, {"NotoSansDevanagari"}),
        ("१२३ 2024", True, {"NotoSansDevanagari"}),
        ("क्" + chr(0x200D) + "ष", True, {"NotoSansDevanagari"}),
        ("मराठी-English", False, {"NotoSansDevanagari"}),  # Latin letters in the word: unshaped
        ("Pythonआधारित", False, {"NotoSansDevanagari"}),
    ],
)
def test_paragraph_markup_shaping_decision(text: str, shaped: bool, fonts: set[str]) -> None:
    markup, is_shaped = pdf.paragraph_markup(text)
    assert is_shaped is shaped
    assert set(re.findall(r'<font name="([^"]+)">', markup)) == fonts


def test_paragraph_markup_escapes_inside_devanagari_runs() -> None:
    markup, _ = pdf.paragraph_markup("हिन्दी<b>&")
    assert "<b>" not in markup and "&amp;" in markup and "&lt;b&gt;" in markup


@pytest.mark.parametrize("text", ["<हिन्दी>", "हिन्दी<>", "\"क\"<'ख'>"])
def test_paragraph_markup_escapes_words_set_entirely_in_noto(text: str) -> None:
    # Noto maps < > " ', so these words are set whole in the Devanagari font.
    markup, shaped = pdf.paragraph_markup(text)
    assert shaped is True
    assert markup == f'<font name="NotoSansDevanagari">{escape(text)}</font>'
    Paragraph(markup, pdf._SHAPED_STYLES["body"])  # parses as text, not tags


SHAPED_SAMPLES = [
    MIXED,
    "Python और SQL, हिन्दी (Hindi) — मराठी" + chr(0x200D) + "x" + chr(0x200C),
    "Senior Engineer — Bengaluru, प्रबंधक / Team Lead\tटीम\nनेतृत्व",
    "(हिन्दी) [मराठी] {१२३} हिन्दी123 क" + chr(0x301) + " ॐ" + chr(0x1CD0),
]


@pytest.mark.parametrize("text", SHAPED_SAMPLES)
def test_shaped_paragraphs_never_mix_fonts_within_a_word(text: str) -> None:
    # ReportLab shapes each word with the font of its first fragment; a word with
    # two fonts would be given the wrong glyphs (or crash), so it must not exist.
    markup, shaped = pdf.paragraph_markup(pdf._clean(text))
    style = (pdf._SHAPED_STYLES if shaped else pdf._STYLES)["body"]
    for word in _getFragWords(Paragraph(markup, style).frags, 1e6):
        fonts = {frag.fontName for frag, chars in word[1:] if chars}
        assert len(fonts) <= 1 or not shaped, (word, fonts)


def test_mixed_words_render_in_a_fresh_process() -> None:
    # Regression: with shaping, a word mixing fonts crashed ReportLab when it was
    # the first Devanagari rendered in a process.
    code = textwrap.dedent(
        f"""
        import sys, uuid
        sys.path.insert(0, {str(ROOT)!r})
        from datetime import UTC, datetime, timedelta
        from backend.reports.fonts import default_registry
        from backend.reports.pdf import render_report_pdf
        from backend.storage.report_store import ReportStamp
        from tests.fixtures.reports import report_data
        from backend.schemas.report import AnalysisReport
        assert default_registry().register()
        now = datetime(2026, 11, 2, tzinfo=UTC)
        data = report_data(ReportStamp(uuid.uuid4(), now, now + timedelta(hours=1)))
        profile = data["resume_profile"].model_dump()
        profile["skills"] = ["(हिन्दी)", "मराठी-English", "Pythonआधारित", "हिन्दी123"]
        data["resume_profile"] = profile
        report = AnalysisReport.model_validate(
            {{k: v.model_dump() if hasattr(v, "model_dump") else v for k, v in data.items()}}
        )
        out = render_report_pdf(report, max_pages=100, deadline_seconds=60)
        print(out[:5].decode())
        """
    )
    proc = subprocess.run(  # noqa: S603 - fixed interpreter and test-owned code
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "%PDF-"


# --- Limits: pages, deadline, availability ----------------------------------------------


def _long_report(entries: int) -> AnalysisReport:
    experience = [
        {
            "organization": f"Org {i}",
            "role": f"Role {i}",
            "period": "2020 - 2024",
            "highlights": [f"Token{i:02d}x{j:02d} " + "detail " * 60 for j in range(12)],
        }
        for i in range(entries)
    ]
    return _with(resume_profile=_profile(experience=experience))


def test_page_limit_raises_without_output() -> None:
    report = _long_report(10)
    pages = len(PdfReader(io.BytesIO(_render(report))).pages)
    assert pages > 3
    assert len(PdfReader(io.BytesIO(_render(report, max_pages=pages))).pages) == pages
    with pytest.raises(PdfPageLimitError) as raised:
        _render(report, max_pages=pages - 1)
    assert raised.value.max_pages == pages - 1
    assert raised.value.code == "pdf_page_limit_exceeded"


class _FakeClock:
    """Monotonic clock that advances ``step`` seconds per reading."""

    def __init__(self, step: float) -> None:
        self.now = 0.0
        self.step = step
        self.readings = 0

    def __call__(self) -> float:
        self.readings += 1
        self.now += self.step
        return self.now


def test_deadline_exceeded_while_building_raises_without_output(monkeypatch) -> None:
    clock = _FakeClock(step=0.5)
    monkeypatch.setattr(pdf, "_monotonic", clock)
    with pytest.raises(PdfRenderTimeoutError) as raised:
        _render(_long_report(4), deadline_seconds=15)
    assert raised.value.code == "pdf_render_timeout"
    assert raised.value.deadline_seconds == 15
    assert clock.readings > 4  # failed during the build, not before it


def test_deadline_is_checked_before_every_flowable_and_page(monkeypatch) -> None:
    checks: list[int] = []
    real_check = pdf._ReportDocument.check_deadline

    def counting_check(self) -> None:
        checks.append(self.page)
        real_check(self)

    monkeypatch.setattr(pdf._ReportDocument, "check_deadline", counting_check)
    report = _long_report(2)
    pages = len(PdfReader(io.BytesIO(_render(report))).pages)
    assert len(checks) >= len(pdf._story(report)) + pages
    assert set(range(1, pages + 1)) <= set(checks)  # also checked before page 1 begins


def test_waiting_for_another_render_counts_against_the_deadline() -> None:
    # The waiting render runs in another thread, so an unbounded wait for the lock
    # fails this test instead of hanging it.
    outcome: list[BaseException | bytes] = []

    def waiting_render() -> None:
        try:
            outcome.append(_render(_report(), deadline_seconds=0.2))
        except BaseException as exc:
            outcome.append(exc)

    assert pdf._RENDER_LOCK.acquire()
    try:
        waiter = threading.Thread(target=waiting_render, daemon=True)
        waiter.start()
        waiter.join(timeout=10)
        finished = not waiter.is_alive()
    finally:
        pdf._RENDER_LOCK.release()
    waiter.join(timeout=10)
    assert finished, "the render kept waiting for the lock after its deadline"
    assert len(outcome) == 1 and isinstance(outcome[0], PdfRenderTimeoutError), outcome
    assert _render(_report()).startswith(b"%PDF-")  # the lock was released


def test_render_timed_out_mid_build_leaves_shared_font_state_consistent(monkeypatch) -> None:
    report = _with(resume_profile=_profile(skills=["क्षमता ज्ञान श्री", MIXED, "हिन्दी"]))
    expected = _render(report)

    class _StopAfter:
        """Monotonic clock that passes the deadline after ``readings`` readings."""

        def __init__(self, readings: int) -> None:
            self.left = readings

        def __call__(self) -> float:
            self.left -= 1
            return 0.0 if self.left >= 0 else 1e9

    counter = _FakeClock(step=0)
    monkeypatch.setattr(pdf, "_monotonic", counter)
    _render(report)
    # Time out at every deadline check in turn: before, between and after the
    # shaped Devanagari paragraphs, before each page and after the build.
    for readings in range(2, counter.readings):
        clock = _StopAfter(readings)
        monkeypatch.setattr(pdf, "_monotonic", clock)
        with pytest.raises(PdfRenderTimeoutError):
            _render(report)
    monkeypatch.undo()

    assert pdf._RENDER_LOCK.acquire(blocking=False)  # released by the timed-out render
    pdf._RENDER_LOCK.release()
    assert FontRegistry().register()  # every original glyph and width entry unchanged
    assert _render(report) == expected


def test_unavailable_fonts_raise_without_output(monkeypatch, tmp_path: Path) -> None:
    failed = FontRegistry(tmp_path / "missing")
    assert failed.register() is False
    monkeypatch.setattr(pdf, "default_registry", lambda: failed)
    with pytest.raises(PdfRendererUnavailableError) as raised:
        _render(_report())
    assert raised.value.code == "pdf_renderer_unavailable"


def test_unregistered_fonts_raise_without_output(monkeypatch) -> None:
    monkeypatch.setattr(pdf, "default_registry", lambda: FontRegistry())  # never registered
    with pytest.raises(PdfRendererUnavailableError):
        _render(_report())


def test_errors_are_typed_plain_exceptions_with_safe_messages() -> None:
    errors = [PdfPageLimitError(100), PdfRenderTimeoutError(15), PdfRendererUnavailableError()]
    assert [e.code for e in errors] == [
        "pdf_page_limit_exceeded",
        "pdf_render_timeout",
        "pdf_renderer_unavailable",
    ]
    for error in errors:
        assert isinstance(error, PdfRenderError)
        assert str(error) == error.message
        assert not hasattr(error, "status_code")
    assert "fastapi" not in sys.modules["backend.reports.errors"].__dict__


# --- No truncation -------------------------------------------------------------------


def test_long_content_is_never_truncated() -> None:
    report = _long_report(8)
    text = _flat(_text(_render(report)))
    for i in range(8):
        assert f"Role{i}" in text and f"Org{i}" in text
        for j in range(12):
            assert f"Token{i:02d}x{j:02d}" in text
    assert text.count("detail") == 8 * 12 * 60


def test_long_unbroken_words_wrap_without_loss() -> None:
    word = "Z" * 180 + "0123456789" * 2
    text = _flat(_text(_render(_with(resume_profile=_profile(skills=[word], projects=[word * 2])))))
    assert word in text
    assert word * 2 in text


# --- Determinism, caching and concurrency ------------------------------------------------


def test_output_is_deterministic() -> None:
    latin = _report()
    devanagari = _with(resume_profile=_profile(skills=["हिन्दी", MIXED, "क्षमता"]))
    assert _render(latin) == _render(latin)
    assert _render(devanagari) == _render(devanagari)
    assert _render(latin) != _render(_report(with_job=False))


def test_each_call_renders_afresh(monkeypatch) -> None:
    builds = []
    real_story = pdf._story
    monkeypatch.setattr(pdf, "_story", lambda report: builds.append(1) or real_story(report))
    first, second = _render(_report()), _render(_report())
    assert first == second
    assert len(builds) == 2  # nothing is cached between calls


def test_concurrent_rendering_and_registry_verification() -> None:
    reports = [
        _report(),
        _with(resume_profile=_profile(skills=["हिन्दी", MIXED, "क्षमता ज्ञान श्री"])),
    ]
    expected = [_render(report) for report in reports]
    barrier = threading.Barrier(12)
    outputs: list[tuple[int, bytes]] = []
    registries: list[bool] = []
    errors: list[BaseException] = []

    def render(index: int) -> None:
        try:
            barrier.wait()
            for _ in range(3):
                outputs.append((index, _render(reports[index])))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    def verify() -> None:
        try:
            barrier.wait()
            for _ in range(3):
                registries.append(FontRegistry().register())
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=render, args=(i % 2,)) for i in range(8)]
    threads += [threading.Thread(target=verify) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    assert registries == [True] * 12
    assert len(outputs) == 24
    assert all(data == expected[index] for index, data in outputs)
