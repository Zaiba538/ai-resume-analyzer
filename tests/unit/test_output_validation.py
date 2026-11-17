"""M3 exit criterion: output validation rejects malformed or oversize output."""

from __future__ import annotations

import logging
import re

import pytest
from pydantic import BaseModel, ValidationError

from backend.schemas.critic import CriticVerdict
from backend.schemas.interview import InterviewQuestionSet, with_ids
from backend.schemas.resume import ResumeProfile
from backend.schemas.skill_gap import SkillGapReport
from backend.security.output_validation import LINK_PLACEHOLDER, validate_llm_output

PROFILE = {
    "skills": ["Python", "SQL"],
    "education": [{"qualification": "B.Sc.", "institution": "Example University", "year": None}],
    "experience": [
        {
            "organization": "Acme (fictional)",
            "role": "Data Engineer",
            "period": "2021 - 2024",
            "highlights": ["Built ETL pipelines"],
        }
    ],
    "projects": ["Inventory forecaster"],
    "certifications": [],
}
QUESTION = {"question": "Explain a Python generator.", "topic": "Python", "rationale": "Core skill"}


def test_valid_profile_accepted() -> None:
    assert ResumeProfile.model_validate(PROFILE).skills == ["Python", "SQL"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(extra_field="x"),
        lambda p: p.pop("skills"),
        lambda p: p.update(skills="Python"),
        lambda p: p.update(skills=["x" * 201]),
        lambda p: p.update(skills=["Python"] * 151),
        lambda p: p.update(skills=[""]),
        lambda p: p["experience"][0].update(highlights=["h"] * 13),
        lambda p: p["experience"][0].update(highlights=["x" * 501]),
        lambda p: p["education"][0].update(qualification=""),
        lambda p: p["education"][0].update(grade="A"),
        lambda p: p.update(projects=[123]),
    ],
)
def test_malformed_or_oversize_profile_rejected(mutation) -> None:
    import copy

    data = copy.deepcopy(PROFILE)
    mutation(data)
    with pytest.raises(ValidationError):
        ResumeProfile.model_validate(data)


@pytest.mark.parametrize(
    "data",
    [
        {"beginner": [], "intermediate": [QUESTION], "advanced": [QUESTION]},
        {"beginner": [QUESTION] * 11, "intermediate": [QUESTION], "advanced": [QUESTION]},
        {"beginner": [QUESTION], "intermediate": [QUESTION]},
        {"beginner": [QUESTION], "intermediate": [QUESTION], "advanced": [QUESTION], "expert": []},
        {
            "beginner": [{**QUESTION, "question": "Why?"}],
            "intermediate": [QUESTION],
            "advanced": [QUESTION],
        },
    ],
)
def test_malformed_interview_sets_rejected(data) -> None:
    with pytest.raises(ValidationError):
        InterviewQuestionSet.model_validate(data)


def test_question_ids_assigned_by_server() -> None:
    drafts = InterviewQuestionSet.model_validate(
        {"beginner": [QUESTION, QUESTION], "intermediate": [QUESTION], "advanced": [QUESTION]}
    )
    questions = with_ids(drafts)
    assert [q.id for q in questions["beginner"]] == ["beginner-1", "beginner-2"]
    assert questions["advanced"][0].id == "advanced-1"


@pytest.mark.parametrize(
    "data",
    [
        {"verdict": "approved", "issues": [], "revise_targets": []},
        {"verdict": "revise", "issues": [], "revise_targets": ["ats"]},
        {
            "verdict": "revise",
            "issues": [{"agent": "resume", "field": "skills", "severity": "fatal", "message": "x"}],
            "revise_targets": ["resume"],
        },
    ],
)
def test_critic_enums_enforced(data) -> None:
    with pytest.raises(ValidationError):
        CriticVerdict.model_validate(data)


def test_skill_gap_limits() -> None:
    SkillGapReport.model_validate(
        {
            "matching_skills": ["Python"],
            "missing_skills": ["Kubernetes"],
            "recommendations": ["Learn"],
        }
    )
    with pytest.raises(ValidationError):
        SkillGapReport.model_validate(
            {"matching_skills": [], "missing_skills": [], "recommendations": ["r"] * 16}
        )


def test_output_strings_are_sanitised() -> None:
    dirty = {**PROFILE, "skills": ["Py\u202ethon", "S\u200bQL\x07", "\U000e0041Docker"]}
    cleaned, flags = validate_llm_output(ResumeProfile.model_validate(dirty), source_texts=[""])
    assert cleaned.skills == ["Python", "SQL", "Docker"]
    assert flags == []


def test_unexpected_links_neutralised_and_flagged() -> None:
    report = SkillGapReport.model_validate(
        {
            "matching_skills": ["Python"],
            "missing_skills": ["Kubernetes"],
            "recommendations": [
                "Take the course at https://evil.example/phish?x=1.",
                "Read the docs at www.malicious.xyz",
                "Visit bit.ly/abc123 now",
            ],
        }
    )
    cleaned, flags = validate_llm_output(report, source_texts=["Job: Python, Kubernetes"])
    assert cleaned.recommendations == [
        f"Take the course at {LINK_PLACEHOLDER}.",
        f"Read the docs at {LINK_PLACEHOLDER}",
        f"Visit {LINK_PLACEHOLDER} now",
    ]
    assert [f.type for f in flags] == ["unexpected_url"]
    assert flags[0].source == "llm_output"


def test_links_present_in_source_are_kept() -> None:
    profile = ResumeProfile.model_validate(
        {**PROFILE, "projects": ["Portfolio at https://github.com/jordan-sample/forecaster"]}
    )
    cleaned, flags = validate_llm_output(
        profile, source_texts=["Projects: https://github.com/jordan-sample/forecaster"]
    )
    assert cleaned.projects == profile.projects
    assert flags == []


def test_technology_names_are_not_links() -> None:
    profile = ResumeProfile.model_validate({**PROFILE, "skills": ["Node.js", "Vue.js", "ASP.NET"]})
    cleaned, flags = validate_llm_output(profile, source_texts=["Skills: Node.js, Vue.js, ASP.NET"])
    assert cleaned.skills == ["Node.js", "Vue.js", "ASP.NET"]
    assert flags == []


def test_cleaning_that_empties_a_required_string_is_rejected() -> None:
    # model_copy(update=...) skips validation, simulating a value that is only
    # invalid after cleaning (it is entirely invisible characters).
    profile = ResumeProfile.model_validate(PROFILE).model_copy(update={"skills": ["\u200b\u200b"]})
    with pytest.raises(ValidationError):
        validate_llm_output(profile, source_texts=[""])


def test_multilingual_output_preserved() -> None:
    profile = ResumeProfile.model_validate(
        {**PROFILE, "skills": ["पायथन", "डेटा विश्लेषण", "अर्\u200dय"]}
    )
    cleaned, _ = validate_llm_output(profile, source_texts=[""])
    assert cleaned.skills == ["पायथन", "डेटा विश्लेषण", "अर्\u200dय"]


# --- Link detection: false positives, normalisation, boundaries ---------------


class _Texts(BaseModel):
    items: list[str]


def _clean(items: list[str], source: str = "") -> tuple[list[str], list]:
    cleaned, flags = validate_llm_output(_Texts(items=items), source_texts=[source])
    return cleaned.items, flags


@pytest.mark.parametrize(
    "text",
    [
        "Practise pd.concat and df.merge in pandas.",
        "Built services with ASP.NET Core and ADO.NET.",
        "Holds a B.Com and an M.Com.",
        "Real-time chat with Socket.IO; diagrams in Draw.io.",
        "Extended React.Component and used System.Collections.Generic.",
        "Scored with sklearn.metrics and wired app.include_router.",
        "Node.js, Vue.js, Python 3.12, e.g. FastAPI.",
        "Imported java.io.File and java.net.URL; used com.google.gson.",
        "Loaded socket.io.js and config.app.py.",
    ],
)
def test_technical_terms_are_not_links(text: str) -> None:
    items, flags = _clean([text], source="Python developer")
    assert items == [text]
    assert flags == []


@pytest.mark.parametrize(
    "output",
    [
        "Portfolio: https://github.com/jordan-sample/app",
        "Portfolio: http://github.com/jordan-sample/app",
        "Portfolio: github.com/jordan-sample/app",
        "Portfolio: www.github.com/jordan-sample/app/",
        "Portfolio: HTTPS://WWW.GitHub.com/jordan-sample/app/.",
        "Portfolio: github.com/jordan-sample",  # a prefix of the source link, at a path boundary
    ],
)
@pytest.mark.parametrize(
    "source",
    [
        "Projects: github.com/jordan-sample/app",
        "Projects: https://github.com/jordan-sample/app/",
        "Projects: www.github.com/jordan-sample/app",
    ],
)
def test_known_links_match_in_any_format(output: str, source: str) -> None:
    items, flags = _clean([output], source=source)
    assert items == [output]
    assert flags == []


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("Read https://evil.example/x now.", f"Read {LINK_PLACEHOLDER} now."),
        ("Read https://evil.example/x.", f"Read {LINK_PLACEHOLDER}."),
        ("Visit bit.ly/abc today", f"Visit {LINK_PLACEHOLDER} today"),
        ("See www.malicious.xyz/path/, then stop", f"See {LINK_PLACEHOLDER}, then stop"),
        ("Free course on coursera.org", f"Free course on {LINK_PLACEHOLDER}"),
        ("Email recruiter@evil.com today", f"Email {LINK_PLACEHOLDER} today"),
        ("Go to EVIL.COM", f"Go to {LINK_PLACEHOLDER}"),
        ("Docs: socket.io/docs", f"Docs: {LINK_PLACEHOLDER}"),  # with a path it is a link
        ("Try https://draw.io", f"Try {LINK_PLACEHOLDER}"),  # with a scheme it is a link
        ("Readhttps://evil.example/x now", f"{LINK_PLACEHOLDER} now"),  # glued word removed whole
        ("(see https://evil.example/x) and more", f"(see {LINK_PLACEHOLDER}) and more"),
    ],
)
def test_unknown_links_removed_as_whole_tokens(output: str, expected: str) -> None:
    items, flags = _clean([output], source="Job: Python, github.com/jordan-sample/app")
    assert items == [expected]
    assert [(f.type, f.source) for f in flags] == [("unexpected_url", "llm_output")]


def test_link_removal_preserves_surrounding_text_and_formatting() -> None:
    text = "Line one: pd.concat\n  - Visit https://evil.example/x, then ASP.NET.\tDone"
    items, _ = _clean([text])
    assert items == [f"Line one: pd.concat\n  - Visit {LINK_PLACEHOLDER}, then ASP.NET.\tDone"]


def test_placeholder_never_touches_word_characters() -> None:
    texts = [
        "pd.concat df.merge React.Component System.Collections x.company y.coffee",
        "Readhttps://evil.example/x bit.ly/abc evil.com/path, me@evil.com.",
    ]
    items, _ = _clean(texts)
    for item in items:
        placeholder = re.escape(LINK_PLACEHOLDER)
        assert not re.search(rf"\w{placeholder}|{placeholder}\w", item)
    assert items[0] == texts[0]


def test_domain_prefix_of_a_known_domain_is_not_known() -> None:
    items, flags = _clean(["Visit x.co and notevil.com"], source="Site: x.com, evil.com")
    assert items == [f"Visit {LINK_PLACEHOLDER} and {LINK_PLACEHOLDER}"]
    assert len(flags) == 1


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("See evil.co.uk/x now", f"See {LINK_PLACEHOLDER} now"),
        ("News at bbc.co.uk.", f"News at {LINK_PLACEHOLDER}."),
        ("Log in at evil.com.au/login", f"Log in at {LINK_PLACEHOLDER}"),
        ("Apply via jobs.org.nz", f"Apply via {LINK_PLACEHOLDER}"),
        ("Study at uni.ac.in", f"Study at {LINK_PLACEHOLDER}"),
        ("Mail user@evil.co.uk today", f"Mail {LINK_PLACEHOLDER} today"),
    ],
)
def test_country_second_level_domains_are_links(output: str, expected: str) -> None:
    items, flags = _clean([output], source="Python developer")
    assert items == [expected]
    assert [(f.type, f.source) for f in flags] == [("unexpected_url", "llm_output")]


def test_known_country_second_level_domain_is_kept() -> None:
    text = "Portfolio at jordan-sample.co.uk/work"
    items, flags = _clean([text], source="Site: https://www.jordan-sample.co.uk/work/")
    assert items == [text]
    assert flags == []


def test_known_email_is_kept_and_unknown_email_removed() -> None:
    source = "Contact: jordan.sample@example.com"
    items, flags = _clean(["Reach JORDAN.SAMPLE@example.com", "or x@example.com"], source=source)
    assert items == ["Reach JORDAN.SAMPLE@example.com", f"or {LINK_PLACEHOLDER}"]
    assert len(flags) == 1


def test_full_width_link_is_normalised_and_removed() -> None:
    items, flags = _clean(["Visit ｅｖｉｌ．ｃｏｍ now"])
    assert items == [f"Visit {LINK_PLACEHOLDER} now"]
    assert len(flags) == 1


def test_removed_links_logged_as_agent_and_count_only(captured_logs, caplog) -> None:
    caplog.set_level(logging.INFO, logger="backend")  # restored automatically at teardown
    private_link = "https://evil.example/jordan-sample-private"
    validate_llm_output(
        _Texts(items=[f"See {private_link}", "and bit.ly/abc", "pd.concat"]),
        source_texts=["Resume of Jordan Sample"],
        role="skill_gap",
    )
    (record,) = captured_logs.events("llm_output_links_removed")
    assert record["node"] == "skill_gap"
    assert record["url_findings"] == 2
    for leaked in ("evil.example", "bit.ly", "jordan", "Jordan", "Resume"):
        assert leaked not in captured_logs.text


def test_no_log_record_without_removed_links(captured_logs, caplog) -> None:
    caplog.set_level(logging.INFO, logger="backend")  # restored automatically at teardown
    validate_llm_output(_Texts(items=["ASP.NET"]), source_texts=[""], role="resume")
    assert captured_logs.events("llm_output_links_removed") == []
