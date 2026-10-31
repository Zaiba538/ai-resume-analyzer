"""Opt-in check of the real Sentence-Transformer (downloads the pinned model once).

    pytest -m live tests/live/test_embeddings_live.py -s

No Groq call and no API key needed. Synthetic text only; only scores are printed.
"""

from __future__ import annotations

import pytest

from backend.core.config import Settings
from backend.scoring.embeddings import load_sentence_transformer
from backend.scoring.semantic import semantic_similarity
from tests.fixtures.documents import SYNTHETIC_RESUME_LINES

pytestmark = pytest.mark.live

RESUME = "\n".join(SYNTHETIC_RESUME_LINES)
JOB = "Data Engineer. Required: Python, SQL, Apache Spark, building ETL pipelines."
UNRELATED = "Pastry chef needed for a bakery: sourdough, croissants and cake decoration."


@pytest.fixture(scope="module")
def embedder():
    return load_sentence_transformer(Settings(_env_file=None))


def test_real_model_scores_are_sensible(embedder) -> None:
    related = semantic_similarity(embedder, RESUME, JOB)
    unrelated = semantic_similarity(embedder, RESUME, UNRELATED)
    print(f"\nsemantic related={related} unrelated={unrelated}")
    assert 0.0 <= unrelated < related <= 100.0
    assert semantic_similarity(embedder, RESUME, JOB) == related  # deterministic


def test_real_tokenizer_chunks_fit_window_without_loss(embedder) -> None:
    long_text = " ".join([RESUME] * 30)
    chunks = embedder.chunk(long_text)
    window = embedder._window
    tokenizer = embedder._tokenizer
    lengths = [len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in chunks]
    print(f"\nchunks={len(chunks)} window={window} max_tokens={max(lengths)}")
    assert len(chunks) > 1 and max(lengths) <= window
    assert all(c in long_text for c in chunks)


def test_devanagari_text_is_handled(embedder) -> None:
    score = semantic_similarity(embedder, "पायथन डेटा अभियंता", "पायथन डेटा इंजीनियर आवश्यक")
    print(f"\ndevanagari score={score}")
    assert 0.0 <= score <= 100.0
