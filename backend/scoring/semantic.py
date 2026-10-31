"""Semantic similarity: cosine of chunked Sentence-Transformer embeddings.

Uncalibrated and never called an ATS score (approved Rev. 3, correction 1).

Method (deterministic for a given model and input):

1. Each document is split into model-window chunks (``Embedder.chunk``).
2. Chunks are embedded as L2-normalised vectors.
3. The document vector is the mean of its chunk vectors, re-normalised.
4. Similarity is the cosine of the two document vectors, clamped to [0, 1]
   (a negative cosine means "unrelated", not "worse than unrelated"), scaled
   to 0-100 and rounded to two decimals.

A document with no usable text (empty, whitespace only, or no chunks) yields
0.0. Non-finite model output raises ``ScoringError`` instead of producing a
NaN or infinite score.
"""

from __future__ import annotations

import math

import numpy as np

from backend.core.errors import AppError
from backend.scoring.embeddings import Embedder


class ScoringError(AppError):
    status_code = 500
    code = "scoring_failed"
    message = "A similarity metric could not be computed."


def document_vector(embedder: Embedder, text: str) -> np.ndarray | None:
    """Unit-length document embedding, or None when the text has no usable content."""
    if not text.strip():
        return None
    chunks = embedder.chunk(text)
    if not chunks:
        return None
    vectors = np.asarray(embedder.encode(chunks), dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] != len(chunks) or not np.all(np.isfinite(vectors)):
        raise ScoringError()
    mean = vectors.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm):
        raise ScoringError()
    if norm == 0.0:
        return None
    return mean / norm


def semantic_similarity(embedder: Embedder, resume_text: str, job_description: str) -> float:
    """Cosine similarity of the two documents on a 0-100 scale (see module docstring)."""
    resume = document_vector(embedder, resume_text)
    job = document_vector(embedder, job_description)
    if resume is None or job is None:
        return 0.0
    cosine = float(np.dot(resume, job))
    if not math.isfinite(cosine):
        raise ScoringError()
    return round(min(max(cosine, 0.0), 1.0) * 100, 2)
