"""M5 exit criterion: TF-IDF baseline parity with legacy ``utils/matching.py``.

The legacy module is imported unmodified and run side by side with the port.
"""

from __future__ import annotations

import random

import pytest

from backend.scoring.tfidf_baseline import calculate_match_score, tfidf_keyword_match
from tests.fixtures.documents import SYNTHETIC_RESUME_LINES
from utils.matching import calculate_match_score as legacy_calculate_match_score

RESUME = "\n".join(SYNTHETIC_RESUME_LINES)
JOB = "AI Engineer\nRequired Skills:\nPython\nMachine Learning\nSQL\nLLMs\nLangChain\nAPIs"

CASES = {
    "synthetic_resume_vs_job": (RESUME, JOB),
    "identical_texts": (RESUME, RESUME),
    "disjoint_vocabulary": ("alpha beta gamma", "delta epsilon zeta"),
    "case_differences": ("PYTHON Sql DoCkEr", "python sql docker"),
    "punctuation_and_symbols": ("C++, C#, Node.js; CI/CD!", "c++ c# node js ci cd"),
    "stop_words_are_kept": ("the and of to in a", "the of and"),
    "numbers_and_years": ("2021 2024 5 years", "5 years experience 2024"),
    "repeated_terms": ("python " * 50 + "sql", "python sql sql sql"),
    "devanagari": ("पायथन और एसक्यूएल में अनुभव", "पायथन अनुभव आवश्यक"),
    "mixed_scripts": ("Python पायथन डेटा data", "data डेटा SQL"),
    "one_side_has_no_terms": (RESUME, "a b c ! ?"),
    "whitespace_heavy": ("  python\n\n\tsql   ", "sql\npython"),
    "long_document": (RESUME * 40, JOB * 10),
}


@pytest.mark.parametrize("case", list(CASES))
def test_port_matches_legacy_exactly(case: str) -> None:
    resume, job = CASES[case]
    assert calculate_match_score(resume, job) == legacy_calculate_match_score(resume, job)


@pytest.mark.parametrize(
    ("resume", "job"),
    [("", ""), ("a b c", "d e f"), ("! ? .", "")],
    ids=["both_empty", "single_char_tokens", "punctuation_only"],
)
def test_empty_vocabulary_raises_like_legacy(resume: str, job: str) -> None:
    with pytest.raises(ValueError) as legacy:
        legacy_calculate_match_score(resume, job)
    with pytest.raises(ValueError) as port:
        calculate_match_score(resume, job)
    assert str(port.value) == str(legacy.value)
    assert "empty vocabulary" in str(port.value)


def test_randomised_parity() -> None:
    rng = random.Random(20261031)  # noqa: S311 - seeded test-data generator, not cryptography
    words = [
        "python",
        "sql",
        "docker",
        "kubernetes",
        "spark",
        "data",
        "engineer",
        "the",
        "and",
        "pipelines",
        "etl",
        "aws",
        "ml",
        "models",
        "api",
        "पायथन",
        "डेटा",
        "2024",
        "x",
        "y",
    ]
    for _ in range(300):
        resume = " ".join(rng.choices(words, k=rng.randint(0, 40)))
        job = " ".join(rng.choices(words, k=rng.randint(0, 25)))
        try:
            expected = legacy_calculate_match_score(resume, job)
        except ValueError:
            with pytest.raises(ValueError):
                calculate_match_score(resume, job)
            continue
        assert calculate_match_score(resume, job) == expected


@pytest.mark.parametrize(
    ("resume", "job", "expected"),
    [(RESUME, RESUME, 100.0), ("alpha beta", "gamma delta", 0.0), ("", "", 0.0), ("a", "b", 0.0)],
)
def test_api_wrapper(resume: str, job: str, expected: float) -> None:
    score = tfidf_keyword_match(resume, job)
    assert type(score) is float
    assert score == expected


def test_api_wrapper_value_matches_legacy_and_is_bounded() -> None:
    score = tfidf_keyword_match(RESUME, JOB)
    assert score == float(legacy_calculate_match_score(RESUME, JOB))
    assert 0.0 < score < 100.0
    assert round(score, 2) == score


def test_wrapper_does_not_hide_other_errors(monkeypatch) -> None:
    from backend.scoring import tfidf_baseline

    def boom(*_args):
        raise ValueError("something else entirely")

    monkeypatch.setattr(tfidf_baseline, "calculate_match_score", boom)
    with pytest.raises(ValueError, match="something else"):
        tfidf_baseline.tfidf_keyword_match("a", "b")
