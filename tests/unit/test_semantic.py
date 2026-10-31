"""Semantic similarity: range, determinism, chunking, degenerate input."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.scoring.semantic import ScoringError, document_vector, semantic_similarity
from tests.fixtures.embeddings import WORDS_PER_CHUNK, FakeEmbedder

RESUME = "Data engineer with Python SQL Spark pipelines and Docker deployments"
JOB = "Hiring a data engineer: Python, SQL, Spark and Airflow pipelines"
UNRELATED = "Pastry chef specialising in sourdough croissants and laminated dough"


def test_identical_texts_score_100() -> None:
    assert semantic_similarity(FakeEmbedder(), RESUME, RESUME) == 100.0


def test_related_scores_higher_than_unrelated() -> None:
    embedder = FakeEmbedder()
    related = semantic_similarity(embedder, RESUME, JOB)
    unrelated = semantic_similarity(embedder, RESUME, UNRELATED)
    assert 0.0 <= unrelated < related < 100.0


@pytest.mark.parametrize(
    ("resume", "job"),
    [("", JOB), (RESUME, ""), ("   \n\t", JOB), ("", ""), ("!!! ???", JOB)],
)
def test_empty_or_unusable_input_scores_zero(resume: str, job: str) -> None:
    score = semantic_similarity(FakeEmbedder(), resume, job)
    assert score == 0.0 and math.isfinite(score)


def test_deterministic_and_rounded() -> None:
    scores = {semantic_similarity(FakeEmbedder(), RESUME, JOB) for _ in range(5)}
    assert len(scores) == 1
    (score,) = scores
    assert round(score, 2) == score


def test_multi_chunk_documents_use_every_chunk() -> None:
    embedder = FakeEmbedder()
    long_resume = " ".join([RESUME] * 20)  # many chunks
    assert len(embedder.chunk(long_resume)) > 1
    vector = document_vector(embedder, long_resume)
    assert vector is not None and math.isclose(float(np.linalg.norm(vector)), 1.0, rel_tol=1e-9)
    assert semantic_similarity(embedder, long_resume, JOB) > 0


def test_document_vector_is_mean_of_chunks() -> None:
    embedder = FakeEmbedder()
    text = " ".join(f"w{i}" for i in range(WORDS_PER_CHUNK * 2))
    chunks = embedder.chunk(text)
    expected = embedder.encode(chunks).astype(np.float64).mean(axis=0)
    expected /= np.linalg.norm(expected)
    assert np.allclose(document_vector(embedder, text), expected)


class _Crafted:
    """Embedder returning fixed vectors per text, to test edge cases."""

    name = "crafted"

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def chunk(self, text: str) -> list[str]:
        return [text] if text.strip() else []

    def encode(self, texts):
        return np.array([self.vectors[t] for t in texts], dtype=np.float32)


def test_negative_cosine_clamped_to_zero() -> None:
    embedder = _Crafted({"a": [1.0, 0.0], "b": [-1.0, 0.0]})
    assert semantic_similarity(embedder, "a", "b") == 0.0


def test_zero_vectors_score_zero() -> None:
    embedder = _Crafted({"a": [0.0, 0.0], "b": [1.0, 0.0]})
    assert semantic_similarity(embedder, "a", "b") == 0.0


@pytest.mark.parametrize("bad", [[float("nan"), 0.0], [float("inf"), 1.0]])
def test_non_finite_model_output_raises(bad) -> None:
    embedder = _Crafted({"a": bad, "b": [1.0, 0.0]})
    with pytest.raises(ScoringError) as info:
        semantic_similarity(embedder, "a", "b")
    assert info.value.code == "scoring_failed"


def test_wrong_number_of_vectors_raises() -> None:
    class Short(_Crafted):
        def encode(self, texts):
            return np.zeros((0, 2), dtype=np.float32)

    with pytest.raises(ScoringError):
        semantic_similarity(Short({}), "a", "b")
