"""M3 exit criterion: output validation rejects malformed or oversize output."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    dirty = {**PROFILE, "skills": ["Py‮thon", "S​QL\x07", "\U000e0041Docker"]}
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
    profile = ResumeProfile.model_validate(PROFILE).model_copy(update={"skills": ["​​"]})
    with pytest.raises(ValidationError):
        validate_llm_output(profile, source_texts=[""])


def test_multilingual_output_preserved() -> None:
    profile = ResumeProfile.model_validate({**PROFILE, "skills": ["पायथन", "डेटा विश्लेषण", "अर्‍य"]})
    cleaned, _ = validate_llm_output(profile, source_texts=[""])
    assert cleaned.skills == ["पायथन", "डेटा विश्लेषण", "अर्‍य"]
