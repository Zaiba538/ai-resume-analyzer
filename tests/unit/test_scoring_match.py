"""M7 phase 2: M5 scoring extracted to ``backend.scoring.match`` (decision D8).

Golden values were recorded from the original implementation in
``backend.api.routes.match`` before the move (deterministic fake embedder).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import backend.api.routes.match as match_route
import backend.scoring.match as scoring_match
from backend.core.config import Settings
from backend.scoring.match import clean_job_description, clean_resume_text, compute_match
from tests.fixtures.documents import SYNTHETIC_RESUME_LINES
from tests.fixtures.embeddings import FakeEmbedder
from tests.unit.test_language import (
    BILINGUAL_RESUME,
    ENGLISH_JD_WITH_DEVANAGARI_CITY,
    ENGLISH_WITH_DEVANAGARI,
    HINDI_JD,
    HINDI_RESUME,
    MARATHI_JD,
    MARATHI_RESUME,
    SHORT_HINDI_JD,
)

SETTINGS = Settings(_env_file=None)
SMALL = Settings(_env_file=None, max_job_description_chars=500, max_extracted_chars=1000)
RESUME = "\n".join(SYNTHETIC_RESUME_LINES)
UNSUPPORTED = ("UnsupportedLanguageError", 422, "unsupported_language")
NOT_CALIBRATED = {
    "status": "not_calibrated",
    "score": None,
    "calibration_version": None,
    "evaluation_summary": None,
}


def metrics(resume: str, job: str) -> dict:
    cleaned = clean_resume_text(resume, SETTINGS), clean_job_description(job, SETTINGS)
    return compute_match(FakeEmbedder(), *cleaned).model_dump(mode="json")


# --- golden parity with the original implementation -----------------------------------


@pytest.mark.parametrize(
    ("resume", "job", "expected"),
    [
        (
            RESUME,
            "Required: Python, SQL, Kubernetes, Airflow, AWS.",
            {
                "semantic": 23.19,
                "coverage": (40.0, ["Python", "SQL"], ["Kubernetes", "Apache Airflow", "AWS"], 5),
                "completeness": (100.0, 7, []),
                "tfidf": 7.42,
            },
        ),
        (
            "Skills: k8s, Golang, Py, Postgres",
            "Kubernetes, Go, Python, PostgreSQL",
            {
                "semantic": 0.0,
                "coverage": (75.0, ["Kubernetes", "Go", "PostgreSQL"], ["Python"], 4),
                "completeness": (
                    14.29,
                    1,
                    ["contact", "summary", "experience", "education", "projects", "certifications"],
                ),
                "tfidf": 0.0,
            },
        ),
        (
            "कौशल्य: पायथन, डेटा विश्लेषण. Skills: Python, SQL",
            "Python, SQL, डेटा",
            {
                "semantic": 53.45,
                "coverage": (100.0, ["Python", "SQL"], [], 2),
                "completeness": (
                    0.0,
                    0,
                    [
                        "contact",
                        "summary",
                        "skills",
                        "experience",
                        "education",
                        "projects",
                        "certifications",
                    ],
                ),
                "tfidf": 44.94,
            },
        ),
    ],
    ids=["synthetic", "aliases", "multilingual"],
)
def test_compute_match_matches_original_golden_values(resume, job, expected) -> None:
    result = metrics(resume, job)
    m = result["metrics"]
    assert m["semantic_similarity"] == {"value": expected["semantic"], "calibrated": False}
    value, matched, missing, total = expected["coverage"]
    assert m["skill_coverage"] == {
        "value": value,
        "matched": matched,
        "missing": missing,
        "jd_skills_total": total,
    }
    value, present_count, absent = expected["completeness"]
    assert m["section_completeness"]["value"] == value
    assert len(m["section_completeness"]["present"]) == present_count
    assert m["section_completeness"]["absent"] == absent
    assert result["baseline"] == {"tfidf_keyword_match": expected["tfidf"]}
    assert result["ats_score"] == NOT_CALIBRATED


def test_ats_gate_is_always_not_calibrated() -> None:
    for resume, job in [(RESUME, RESUME), ("Python", "Python"), ("a b c", "x y z")]:
        assert metrics(resume, job)["ats_score"] == NOT_CALIBRATED


@pytest.mark.parametrize(
    ("fn", "text", "settings", "expected"),
    [
        (
            clean_job_description,
            None,
            SETTINGS,
            ("MissingJobDescriptionError", 422, "missing_job_description"),
        ),
        (
            clean_job_description,
            "   \n\t ",
            SETTINGS,
            ("EmptyJobDescriptionError", 422, "empty_job_description"),
        ),
        (
            clean_job_description,
            chr(0x200B) * 5,
            SETTINGS,
            ("EmptyJobDescriptionError", 422, "empty_job_description"),
        ),
        (
            clean_job_description,
            "x" * 501,
            SMALL,
            ("JobDescriptionTooLongError", 422, "job_description_too_long"),
        ),
        (clean_resume_text, "  ", SETTINGS, ("EmptyResumeTextError", 422, "empty_resume_text")),
        (
            clean_resume_text,
            "y" * 1001,
            SMALL,
            ("ResumeTextTooLongError", 422, "document_too_long"),
        ),
        # English-only check: after every existing check, so their errors keep precedence.
        (clean_resume_text, HINDI_RESUME, SETTINGS, UNSUPPORTED),
        (clean_resume_text, MARATHI_RESUME, SETTINGS, UNSUPPORTED),
        (clean_job_description, HINDI_JD, SETTINGS, UNSUPPORTED),
        (clean_job_description, MARATHI_JD, SETTINGS, UNSUPPORTED),
        (clean_job_description, SHORT_HINDI_JD, SETTINGS, UNSUPPORTED),
        (
            clean_resume_text,
            HINDI_RESUME * 20,
            SMALL,
            ("ResumeTextTooLongError", 422, "document_too_long"),
        ),
        (
            clean_job_description,
            MARATHI_JD * 5,
            SMALL,
            ("JobDescriptionTooLongError", 422, "job_description_too_long"),
        ),
    ],
)
def test_clean_helpers_raise_the_original_typed_errors(fn, text, settings, expected) -> None:
    with pytest.raises(Exception) as info:
        fn(text, settings)
    error = info.value
    assert (type(error).__name__, error.status_code, error.code) == expected


@pytest.mark.parametrize(
    ("fn", "text", "settings", "expected"),
    [
        (
            clean_job_description,
            "  Python  developer \r\n\r\n\r\n SQL ",
            SETTINGS,
            "Python  developer\n\n SQL",
        ),
        (clean_job_description, "x" * 500, SMALL, "x" * 500),  # exactly at the limit
        (clean_resume_text, "A" + chr(0xA0) + "B  \n\n\n\nC", SETTINGS, "A B\n\nC"),
        (clean_resume_text, "y" * 1000, SMALL, "y" * 1000),
        (clean_resume_text, ENGLISH_WITH_DEVANAGARI, SETTINGS, ENGLISH_WITH_DEVANAGARI),
        (clean_resume_text, BILINGUAL_RESUME, SETTINGS, BILINGUAL_RESUME),
        (
            clean_job_description,
            ENGLISH_JD_WITH_DEVANAGARI_CITY,
            SETTINGS,
            ENGLISH_JD_WITH_DEVANAGARI_CITY,
        ),
    ],
)
def test_clean_helpers_return_the_original_cleaned_text(fn, text, settings, expected) -> None:
    assert fn(text, settings) == expected


# --- backward compatibility of the route module ---------------------------------------

MOVED = (
    "MissingJobDescriptionError",
    "EmptyJobDescriptionError",
    "JobDescriptionTooLongError",
    "EmptyResumeTextError",
    "ResumeTextTooLongError",
    "clean_job_description",
    "clean_resume_text",
    "compute_match",
)


@pytest.mark.parametrize("name", MOVED)
def test_route_module_re_exports_the_same_objects(name: str) -> None:
    assert getattr(match_route, name) is getattr(scoring_match, name)


def test_route_module_no_longer_defines_scoring_logic() -> None:
    tree = ast.parse(Path(match_route.__file__).read_text(encoding="utf-8"))
    defined = {n.name for n in tree.body if isinstance(n, ast.ClassDef | ast.FunctionDef)}
    assert defined.isdisjoint(MOVED)


# --- framework independence -----------------------------------------------------------


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_extracted_module_imports_no_web_framework() -> None:
    for name in _imports(Path(scoring_match.__file__)):
        assert not name.startswith(("fastapi", "starlette", "backend.api")), name


# M5's embeddings.py uses ``starlette.concurrency.run_in_threadpool`` to load the
# model off the event loop: a concurrency helper, not a request or response object.
WEB_LAYER = ("fastapi", "backend.api", "starlette.requests", "starlette.responses")


@pytest.mark.parametrize(
    "path", sorted(Path(scoring_match.__file__).parent.glob("*.py")), ids=lambda p: p.name
)
def test_scoring_package_never_imports_request_or_response_objects(path: Path) -> None:
    for name in _imports(path):
        assert not name.startswith(WEB_LAYER), (path.name, name)


def test_scoring_functions_take_no_request_or_response_objects() -> None:
    for fn in (clean_job_description, clean_resume_text, compute_match):
        for parameter in inspect.signature(fn).parameters.values():
            annotation = str(parameter.annotation)
            assert "Request" not in annotation and "Response" not in annotation, fn.__name__


def test_the_language_error_names_the_rejected_input() -> None:
    from backend.parsing.language import UnsupportedLanguageError

    for fn, text, loc in (
        (clean_resume_text, HINDI_RESUME, ["resume"]),
        (clean_job_description, MARATHI_JD, ["job_description"]),
    ):
        with pytest.raises(UnsupportedLanguageError) as excinfo:
            fn(text, SETTINGS)
        assert excinfo.value.details[0].loc == loc
