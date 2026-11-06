"""English-only check: tokenisation, script classification, thresholds and the error.

All text is synthetic. The Hindi and Marathi samples are written the way such
resumes and job descriptions usually are, with skill names left in English.
"""

from __future__ import annotations

import io
import logging
import random
import re
import time
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from backend.parsing import language
from backend.parsing.language import (
    LanguageAssessment,
    UnsupportedLanguageError,
    assess_language,
    require_english,
    words,
)
from backend.security.sanitize import sanitize_text

# --- synthetic samples (shared with the parsing, scoring, API and workflow tests) ------------

ENGLISH_RESUME_LINES = [
    "Priya Deshpande",
    "priya.deshpande@example.com | https://github.com/example",
    "Summary",
    "Backend engineer with six years of experience building Python and Go services.",
    "Experience",
    "Software Engineer, Example Corp (2019-2024): designed REST APIs and migrated jobs "
    "to Kubernetes.",
    "Skills",
    "Python, Go, PostgreSQL, Kafka, Docker, Kubernetes, AWS, CI/CD",
    "Education",
    "B.E. Computer Engineering, University of Pune",
]
ENGLISH_RESUME = "\n".join(ENGLISH_RESUME_LINES)
DEVANAGARI_NAME_ADDRESS_LINES = [
    ENGLISH_RESUME_LINES[0],
    "प्रिया देशपांडे, पुणे, महाराष्ट्र",
    *ENGLISH_RESUME_LINES[1:],
]
LANGUAGES_LINE = "Languages: English, हिंदी, मराठी"
ENGLISH_WITH_DEVANAGARI = "\n".join([*DEVANAGARI_NAME_ADDRESS_LINES, LANGUAGES_LINE])
ENGLISH_SKILLS = "Skills\nPython, SQL, Docker, Kubernetes, AWS, Terraform, React, Node.js, C++, C#"
ENGLISH_JD = (
    "We are hiring a backend engineer with Python, PostgreSQL and Kubernetes experience "
    "to build reliable payment services."
)
ENGLISH_JD_WITH_DEVANAGARI_CITY = (
    "Senior Python developer for our पुणे office; Docker and AWS required."
)

HINDI_RESUME_LINES = [
    "वरिष्ठ सॉफ्टवेयर इंजीनियर",
    "मुझे पायथन और जावा में छह साल का अनुभव है।",
    "मैंने भुगतान प्रणाली के लिए सेवाएँ बनाई हैं।",
    "कौशल: Python, Java, SQL, AWS, Docker",
]
HINDI_RESUME = "\n".join(HINDI_RESUME_LINES)
MARATHI_RESUME_LINES = [
    "वरिष्ठ सॉफ्टवेअर अभियंता",
    "मला पायथन आणि जावा मध्ये सहा वर्षांचा अनुभव आहे.",
    "मी पेमेंट प्रणालीसाठी सेवा तयार केल्या.",
    "कौशल्ये: Python, Java, SQL, AWS",
]
MARATHI_RESUME = "\n".join(MARATHI_RESUME_LINES)
HINDI_JD = (
    "हम पुणे में एक वरिष्ठ बैकएंड इंजीनियर की तलाश कर रहे हैं। उम्मीदवार को पायथन, "
    "डेटाबेस और क्लाउड सेवाओं का पाँच साल का अनुभव होना चाहिए। Python, AWS"
)
MARATHI_JD = (
    "आम्ही पुण्यात वरिष्ठ बॅकएंड अभियंत्याच्या शोधात आहोत. उमेदवाराला पायथन, डेटाबेस "
    "आणि क्लाउड सेवांचा पाच वर्षांचा अनुभव असावा. कौशल्ये: Python, AWS, Docker"
)
SHORT_HINDI_JD = "पायथन डेवलपर चाहिए"
BILINGUAL_RESUME = "\n".join([ENGLISH_RESUME, HINDI_RESUME])

RESUME_MESSAGE = (
    "The resume must be in English. Predominantly non-English resumes are not supported."
)
JD_MESSAGE = (
    "The job description must be in English. "
    "Predominantly non-English job descriptions are not supported."
)

_DEVANAGARI_FONT = "LanguageTestNotoDevanagari"
_FONT_FILE = (
    Path(language.__file__).parents[1] / "assets" / "fonts" / "NotoSansDevanagari-Regular.ttf"
)


def make_devanagari_pdf(lines: list[str]) -> bytes:
    """A text PDF; lines containing Devanagari use the bundled Noto Sans Devanagari."""
    if _DEVANAGARI_FONT not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(_DEVANAGARI_FONT, str(_FONT_FILE)))
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    y = 800
    for line in lines:
        devanagari = any("\u0900" <= ch <= "\u097f" for ch in line)
        pdf.setFont(_DEVANAGARI_FONT if devanagari else "Helvetica", 11)
        pdf.drawString(50, y, line)
        y -= 16
    pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def _clean(text: str) -> str:
    return sanitize_text(text).text  # the check always sees sanitised (NFKC) text


def _text(latin: int, non_latin: int) -> str:
    return " ".join(["engineer"] * latin + ["अभियंता"] * non_latin)


# --- the samples are what they claim to be ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "minimum_words", "share_range"),
    [
        (ENGLISH_RESUME, 20, (0, 0)),
        (ENGLISH_WITH_DEVANAGARI, 20, (0.01, 0.15)),
        (HINDI_RESUME, 20, (0.75, 0.90)),
        (MARATHI_RESUME, 20, (0.75, 0.90)),
        (HINDI_JD, 20, (0.85, 0.95)),
        (MARATHI_JD, 20, (0.80, 0.95)),
        (BILINGUAL_RESUME, 20, (0.25, 0.40)),
    ],
)
def test_samples_have_the_intended_shape(text, minimum_words, share_range) -> None:
    result = assess_language(_clean(text))
    assert result.words >= minimum_words
    low, high = share_range
    assert low <= result.non_latin_words / result.words <= high


# --- decisions -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        ENGLISH_RESUME,
        ENGLISH_SKILLS,
        ENGLISH_JD,
        ENGLISH_JD_WITH_DEVANAGARI_CITY,
        ENGLISH_WITH_DEVANAGARI,
        "\n".join(DEVANAGARI_NAME_ADDRESS_LINES),
        "\n".join([ENGLISH_RESUME, LANGUAGES_LINE]),
        BILINGUAL_RESUME,  # full English content alongside Hindi: not predominantly non-English
        "Python, SQL",
        "प्रिया",
        "प्रिया देशपांडे",  # two words: too short to judge
        "12345 !!! 2019-2024 40%",  # no letters: accepted, as before
        "",
    ],
    ids=[
        "english_resume",
        "english_skill_list",
        "english_jd",
        "english_jd_devanagari_city",
        "devanagari_name_address_languages",
        "devanagari_name_address",
        "languages_line",
        "bilingual",
        "short_skill_list",
        "one_devanagari_word",
        "two_devanagari_words",
        "no_letters",
        "empty",
    ],
)
def test_english_and_borderline_text_is_accepted(text: str) -> None:
    assert not assess_language(_clean(text)).predominantly_non_english
    require_english(_clean(text), "resume")


@pytest.mark.parametrize(
    "text",
    [HINDI_RESUME, MARATHI_RESUME, HINDI_JD, MARATHI_JD, SHORT_HINDI_JD],
    ids=["hindi_resume", "marathi_resume", "hindi_jd", "marathi_jd", "short_hindi_jd"],
)
@pytest.mark.parametrize("source", ["resume", "job_description"])
def test_hindi_and_marathi_are_rejected(text: str, source: str) -> None:
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        require_english(_clean(text), source)
    assert excinfo.value.details[0].loc == [source]


@pytest.mark.parametrize(
    "text",
    [
        "خبرة ست سنوات في تطوير خدمات بايثون وقواعد البيانات " * 3,
        "六年后端开发经验 熟悉数据库 云服务 支付系统 " * 6,
        "Шесть лет опыта разработки серверных сервисов и баз данных " * 3,
        "ছয় বছরের অভিজ্ঞতা সফটওয়্যার প্রকৌশলী ডেটাবেস ক্লাউড পরিষেবা " * 3,
        "ஆறு ஆண்டுகள் அனுபவம் மென்பொருள் பொறியாளர் தரவுத்தளம் சேவைகள் " * 3,
    ],
    ids=["arabic", "chinese", "cyrillic", "bengali", "tamil"],
)
def test_other_non_latin_scripts_are_rejected(text: str) -> None:
    assert assess_language(_clean(text)).words >= 20
    with pytest.raises(UnsupportedLanguageError):
        require_english(_clean(text), "resume")


# --- exact thresholds ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("latin", "non_latin", "rejected"),
    [
        # 20 or more words: rejected at 50%.
        (10, 10, True),
        (11, 9, False),
        (50, 50, True),
        (51, 49, False),
        (0, 20, True),
        (20, 0, False),
        # Fewer than 20 words: rejected at 80% with at least three non-Latin words.
        (3, 16, True),  # 19 words, 84%
        (4, 15, False),  # 19 words, 78.9%
        (3, 12, True),  # exactly 80%
        (4, 11, False),
        (1, 4, True),  # 5 words, exactly 80%
        (2, 3, False),  # 60%
        (0, 3, True),  # the minimum: three words, 100%
        (0, 2, False),  # 100%, but only two words
        (1, 3, False),  # 75%
        (0, 1, False),
        (0, 0, False),
    ],
)
def test_threshold_boundaries(latin: int, non_latin: int, rejected: bool) -> None:
    result = assess_language(_text(latin, non_latin))
    assert (result.latin_words, result.non_latin_words) == (latin, non_latin)
    assert result.predominantly_non_english is rejected


def test_the_short_text_limit_is_exactly_twenty_words() -> None:
    # 10 of 19 (53%) is below the short-text bar; 10 of 20 (50%) meets the long-text bar.
    assert not LanguageAssessment(latin_words=9, non_latin_words=10).predominantly_non_english
    assert LanguageAssessment(latin_words=10, non_latin_words=10).predominantly_non_english


def test_thresholds_are_the_approved_values() -> None:
    assert language.SHORT_TEXT_WORDS == 20
    assert language.SHORT_TEXT_REJECT_SHARE == (4, 5)
    assert language.SHORT_TEXT_MIN_NON_LATIN_WORDS == 3
    assert language.REJECT_SHARE == (1, 2)


# --- tokenisation ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("प्रिया देशपांडे", ["प्रिया", "देशपांडे"]),  # vowel signs and viramas stay inside
        ("महाराष्ट्र", ["महाराष्ट्र"]),
        ("अर्\u200dयमान", ["अर्\u200dयमान"]),  # Marathi eyelash ra: ZWJ kept in the word
        ("क्\u200cष", ["क्\u200cष"]),  # ZWNJ kept in the word
        ("cafe\u0301 naïve", ["cafe\u0301", "naïve"]),  # combining accent kept
        ("Python3, C++ and 2019-2024 (40%)", ["Python", "C", "and"]),
        (
            "Contact priya@example.com, https://github.com/example or www.example.org/x",
            ["Contact", "or"],
        ),
        ("ftp://files.example.com/cv and mailto:me@example.com", ["and"]),  # one email token
        ("\u0947\u094d 123", []),  # orphan marks are not a word
    ],
)
def test_words(text: str, expected: list[str]) -> None:
    assert words(text) == expected


def test_urls_and_emails_do_not_make_hindi_look_english() -> None:
    links = " ".join(f"https://example.com/p{i} user{i}@example.com" for i in range(30))
    assert assess_language(_clean(f"{HINDI_RESUME}\n{links}")).predominantly_non_english


@pytest.mark.parametrize(
    ("word", "latin"),
    [
        ("engineer", True),
        ("अभियंता", False),
        ("aब", True),  # a tie counts as Latin
        ("abकख", True),
        ("aकख", False),
        ("Ünïcödé", True),
        ("ǅemal", True),  # Latin Extended-B
        ("ɐbc", True),  # IPA Extensions
        ("ʃʊə", True),  # only IPA letters
        ("x" + chr(0x0323) + chr(0x0307), True),  # marks do not vote, even uncomposed
        ("ḍḥ", True),  # Latin Extended Additional
        ("αβγ", False),  # Greek
    ],
)
def test_words_are_classified_by_majority_script(word: str, latin: bool) -> None:
    result = assess_language(word)
    assert (result.latin_words, result.non_latin_words) == ((1, 0) if latin else (0, 1))


def test_full_width_latin_is_latin_after_sanitisation() -> None:
    full_width = "".join(chr(ord(c) + 0xFEE0) if c.isalpha() else c for c in ENGLISH_JD)
    assert assess_language(_clean(full_width)).non_latin_words == 0


def test_the_decision_is_deterministic() -> None:
    results = {assess_language(_clean(text)) for text in [MARATHI_RESUME] * 20}
    assert len(results) == 1


# --- the error -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "message"), [("resume", RESUME_MESSAGE), ("job_description", JD_MESSAGE)]
)
def test_the_error_is_a_safe_422(source: str, message: str) -> None:
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        require_english(_clean(HINDI_RESUME), source)
    error = excinfo.value
    assert (error.status_code, error.code, error.message) == (422, "unsupported_language", message)
    (detail,) = error.details
    assert detail.model_dump() == {"loc": [source], "type": "unsupported_language", "msg": message}
    assert "पायथन" not in str(error) + repr(error.details)


def test_a_rejection_is_logged_without_any_text(captured_logs) -> None:
    logging.getLogger("backend").setLevel(logging.INFO)
    with pytest.raises(UnsupportedLanguageError):
        require_english(_clean(f"{MARATHI_JD} CANARYlang7"), "job_description")
    (record,) = captured_logs.events("unsupported_language")
    assert record["component"] == "job_description"
    assert "CANARYlang7" not in captured_logs.text
    assert not any("\u0900" <= ch <= "\u097f" for ch in captured_logs.text)


def test_acceptance_logs_nothing(captured_logs) -> None:
    logging.getLogger("backend").setLevel(logging.INFO)
    require_english(_clean(ENGLISH_WITH_DEVANAGARI), "resume")
    assert captured_logs.events("unsupported_language") == []


# --- F1: URL and email removal in linear time ------------------------------------------------

# The whole-text patterns used before audit finding F1: correct, but quadratic on
# crafted input. They are the reference for the token scanner, on short texts only.
_REFERENCE_URL = re.compile(r"(?:\b[a-z][a-z0-9+.-]*://|\bwww\.)\S*", re.IGNORECASE)
_REFERENCE_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[^\s@]+")
_FUZZ_PIECES = [
    *"abWwxh.:/@+-_1 \n\t(),é",
    "\xa0",
    "\u3000",
    "हि",
    "न्",
    "\u0130",  # characters that [a-z] matches under IGNORECASE
    "\u0131",
    "\u017f",
    "\u212a",
    "://",
    "www.",
    "@x.co",
]


def test_links_are_removed_exactly_as_the_earlier_patterns_removed_them() -> None:
    rng = random.Random(20261105)  # noqa: S311 - deterministic test data
    for _ in range(30_000):
        text = "".join(rng.choice(_FUZZ_PIECES) for _ in range(rng.randint(0, 24)))
        reference = _REFERENCE_EMAIL.sub(" ", _REFERENCE_URL.sub(" ", text))
        assert language._without_links(text).split() == reference.split(), repr(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("(https://example.com/x), see", ["see"]),
        ("Portfolio:https://x.io/p", ["Portfolio"]),
        ("WWW.Example.COM/jobs and more", ["and", "more"]),
        ("Email: me@example.com.", ["Email"]),
        ("user@localhost", ["user", "localhost"]),  # no dot in the domain: not an address
        ("@handle", ["handle"]),
        ("a@b.c@d.e", ["d", "e"]),
        ("https://x.io/a\nपायथन", ["पायथन"]),  # removal stops at the end of the token
        (
            "पायथन https://example.com/a अनुभव me@example.com कौशल",
            ["पायथन", "अनुभव", "कौशल"],  # surrounding text is kept
        ),
    ],
)
def test_ordinary_urls_and_emails(text: str, expected: list[str]) -> None:
    assert words(text) == expected


def _attack(kind: str, length: int) -> str:
    half = length // 2
    return {
        "dots": "a." * half,  # the audit's 20 s input at 50,000 characters
        "dashes": "a-" * half,
        "email": "a" * (half - 1) + "@" + "b" * (half - 1),
        "invalid_schemes": "_x://" * (length // 5),
        "at_signs": "a@" * half,
    }[kind]


def _seconds(fn, *args) -> float:
    best = float("inf")
    for _ in range(3):
        started = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - started)
    return best


@pytest.mark.parametrize("kind", ["dots", "dashes", "email", "invalid_schemes", "at_signs"])
def test_crafted_input_is_processed_in_linear_time(kind: str) -> None:
    small = _seconds(assess_language, _attack(kind, 12_500))
    large = _seconds(assess_language, _attack(kind, 50_000))
    assert large < 1.0  # was 20 s for "dots" and 11 s for "email"
    assert large <= 8 * small + 0.05  # four times the input: about four times the time


@pytest.mark.parametrize("kind", ["dots", "dashes", "email"])
def test_crafted_input_at_the_input_caps(kind: str) -> None:
    from backend.core.config import Settings
    from backend.scoring.match import clean_job_description, clean_resume_text

    settings = Settings(_env_file=None)
    resume = _attack(kind, settings.max_extracted_chars)
    job = _attack(kind, settings.max_job_description_chars)
    assert len(resume) <= settings.max_extracted_chars
    assert len(job) <= settings.max_job_description_chars
    assert _seconds(clean_resume_text, resume, settings) < 1.0
    assert _seconds(clean_job_description, job, settings) < 0.5  # was 3.2 s


def test_a_normal_english_resume_at_the_cap_is_fast_and_accepted() -> None:
    from backend.core.config import Settings
    from backend.scoring.match import clean_resume_text

    settings = Settings(_env_file=None)
    resume = ((ENGLISH_RESUME + "\n") * 1000)[: settings.max_extracted_chars].strip()
    assert _seconds(clean_resume_text, resume, settings) < 1.0


def test_long_hindi_with_link_fragments_is_rejected_quickly() -> None:
    fragments = " https://example.com/p1 user@example.com www.example.org/x x://y a.b.c "
    text = _clean(((MARATHI_RESUME + fragments) * 200)[:50_000])
    assert assess_language(text).predominantly_non_english
    assert _seconds(assess_language, text) < 1.0
