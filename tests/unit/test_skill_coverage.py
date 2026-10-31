"""Skill normalisation, alias matching and coverage."""

from __future__ import annotations

import json
import re
import time

import pytest

from backend.scoring.skill_coverage import skill_coverage
from backend.scoring.skill_normalizer import SKILLS_FILE, skill_matcher

ZWSP = chr(0x200B)


def find(text: str) -> list[str]:
    return skill_matcher().find(text)


def test_vocabulary_integrity() -> None:
    data = json.loads(SKILLS_FILE.read_text(encoding="utf-8"))
    names = [skill["name"] for skill in data["skills"]]
    assert len(names) == len(set(names))
    owner: dict[str, str] = {}
    for skill in data["skills"]:
        for alias in skill["aliases"]:
            key = " ".join(alias.casefold().replace("-", " ").split())
            assert owner.setdefault(key, skill["name"]) == skill["name"], f"alias {alias!r} reused"
    assert len(skill_matcher().vocabulary.skills) == len(names)


@pytest.mark.parametrize(
    ("term", "canonical"),
    [
        ("ML", None),  # case-sensitive aliases are not normalisation keys
        ("machine learning", "Machine Learning"),
        ("Machine-Learning", "Machine Learning"),
        ("sklearn", "scikit-learn"),
        ("K8s", "Kubernetes"),
        ("Postgres", "PostgreSQL"),
        ("NodeJS", "Node.js"),
        ("golang", "Go"),
        ("Python", "Python"),
        ("not a skill", None),
    ],
)
def test_normalize(term: str, canonical: str | None) -> None:
    assert skill_matcher().normalize(term) == canonical


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Python, SQL and Docker", ["Python", "SQL", "Docker"]),
        ("JavaScript and Java", ["JavaScript", "Java"]),
        ("C++ and C# and C", ["C++", "C#", "C"]),
        ("MySQL, PostgreSQL", ["MySQL", "PostgreSQL"]),
        ("SQL Server administration", ["SQL Server"]),
        ("machine-learning and deep  learning", ["Machine Learning", "Deep Learning"]),
        (
            "ML and NLP with LLMs",
            ["Machine Learning", "Natural Language Processing", "Large Language Models"],
        ),
        ("CI/CD with GitHub Actions on k8s", ["CI/CD", "GitHub Actions", "Kubernetes"]),
        ("Node.js, React and Vue.js", ["Node.js", "React", "Vue.js"]),
        ("Built in Go and R", ["Go", "R"]),
        ("Golang services", ["Go"]),
    ],
)
def test_detects_skills_in_order_of_appearance(text: str, expected: list[str]) -> None:
    assert find(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Let's go to the meeting and react quickly",  # verbs, not Go/React
        "We excel at teamwork in R&D",  # 'excel' verb, R&D
        "Attach your CV",
        "javascripts",  # not a whole term
        "sparkle and swiftly",
    ],
)
def test_ambiguous_words_are_not_skills(text: str) -> None:
    found = find(text)
    assert not (
        {"Go", "React", "Excel", "R", "Computer Vision", "JavaScript", "Apache Spark", "Swift"}
        & set(found)
    )


@pytest.mark.parametrize("text", ["Worked on AR/VR prototypes", "ksql streams", "pgsql dumps"])
def test_skill_names_at_the_end_of_other_words_are_ignored(text: str) -> None:
    """The leading term boundary: no longer overlapping skill masks these cases."""
    assert find(text) == []


def test_duplicates_and_aliases_counted_once() -> None:
    assert find("Python python PYTHON python3, Python 3") == ["Python"]
    assert find("k8s Kubernetes kubernetes") == ["Kubernetes"]


def test_hidden_characters_cannot_split_or_hide_skills() -> None:
    assert find(f"Kuber{ZWSP}netes and Doc{ZWSP}ker") == ["Kubernetes", "Docker"]
    assert find(chr(0xFF30) + "ython") == ["Python"]  # fullwidth P normalised by NFKC


def test_coverage_counts_and_order() -> None:
    result = skill_coverage(
        "Skills: Python, SQL, Docker, Spark",
        "Requirements: Kubernetes, Python, Airflow, SQL, Python again",
    )
    assert result.matched == ("Python", "SQL")
    assert result.missing == ("Kubernetes", "Apache Airflow")
    assert result.jd_skills_total == 4
    assert result.value == 50.0


def test_coverage_rounds_to_two_decimals() -> None:
    result = skill_coverage("Python", "Python, SQL, Docker")
    assert result.value == 33.33 and result.jd_skills_total == 3


def test_full_and_no_coverage() -> None:
    assert skill_coverage("Python SQL", "Python and SQL").value == 100.0
    none = skill_coverage("Pastry and baking", "Python and SQL")
    assert (none.value, none.matched, none.missing) == (0.0, (), ("Python", "SQL"))


@pytest.mark.parametrize("job", ["", "We value curiosity and kindness.", "   "])
def test_zero_skill_job_description(job: str) -> None:
    result = skill_coverage("Python SQL", job)
    assert (result.value, result.matched, result.missing, result.jd_skills_total) == (
        0.0,
        (),
        (),
        0,
    )


@pytest.mark.parametrize(
    "hostile",
    ["a" * 50_000, ("c++" * 15_000), ("machine " * 6_000), ("-" * 50_000), ("C#" * 20_000)],
)
def test_matching_is_fast_on_hostile_input(hostile: str) -> None:
    started = time.perf_counter()
    skill_coverage(hostile, hostile[:20_000])
    assert time.perf_counter() - started < 2.0


def test_all_alias_patterns_compile() -> None:
    for _, pattern in skill_matcher()._patterns:
        assert isinstance(pattern, re.Pattern)
