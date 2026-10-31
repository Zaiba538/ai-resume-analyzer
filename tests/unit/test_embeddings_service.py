"""Embedding service lifecycle and token-exact chunking (no real model needed)."""

from __future__ import annotations

import numpy as np
import pytest

from backend.scoring.embeddings import (
    EmbeddingService,
    EmbeddingUnavailableError,
    SentenceTransformerEmbedder,
)
from tests.fixtures.embeddings import CountingFactory


@pytest.mark.asyncio
async def test_model_loaded_once_and_reused() -> None:
    factory = CountingFactory()
    service = EmbeddingService(factory)
    assert service.loaded is False
    assert await service.load() is True
    assert await service.load() is True
    assert factory.calls == 1
    assert service.get() is service.get() is factory.embedder


@pytest.mark.asyncio
async def test_load_failure_leaves_service_unavailable(captured_logs) -> None:
    import logging

    logging.getLogger("backend").setLevel(logging.INFO)
    service = EmbeddingService(CountingFactory(fail=True))
    assert await service.load() is False
    assert service.loaded is False
    with pytest.raises(EmbeddingUnavailableError) as info:
        service.get()
    assert (info.value.status_code, info.value.code) == (503, "embedding_model_unavailable")
    (record,) = captured_logs.events("embedding_model_load_failed")
    assert record["error_code"] == "OSError"
    assert "simulated" not in captured_logs.text  # exception message never logged


def test_get_before_load_is_unavailable() -> None:
    with pytest.raises(EmbeddingUnavailableError):
        EmbeddingService(CountingFactory()).get()


class _FastTokenizer:
    """Whitespace tokenizer with offsets, like a Hugging Face fast tokenizer."""

    is_fast = True

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping, truncation, verbose):
        assert add_special_tokens is False and return_offsets_mapping and truncation is False
        assert verbose is False  # no "sequence longer than max length" warning per request
        offsets, start = [], None
        for index, ch in enumerate(text + " "):
            if ch.isspace():
                if start is not None:
                    offsets.append((start, index))
                    start = None
            elif start is None:
                start = index
        return {"offset_mapping": offsets}


class _Model:
    def __init__(self, max_seq_length: int = 6, fast: bool = True) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = _FastTokenizer()
        self.tokenizer.is_fast = fast
        self.encode_kwargs: dict = {}

    def encode(self, texts, **kwargs):
        self.encode_kwargs = kwargs
        return np.ones((len(texts), 3), dtype=np.float64)


def test_chunks_fit_the_model_window_and_lose_nothing() -> None:
    model = _Model(max_seq_length=6)  # window = 6 - 2 special tokens = 4 tokens
    embedder = SentenceTransformerEmbedder(model, "fake")
    text = "one two three four five six seven eight nine ten"
    chunks = embedder.chunk(text)
    assert chunks == ["one two three four", "five six seven eight", "nine ten"]
    assert all(len(c.split()) <= 4 for c in chunks)
    assert " ".join(chunks) == text
    assert all(c in text for c in chunks)


def test_chunking_empty_text() -> None:
    assert SentenceTransformerEmbedder(_Model(), "fake").chunk("   ") == []


def test_encode_requests_normalised_float32_without_progress_bar() -> None:
    model = _Model()
    vectors = SentenceTransformerEmbedder(model, "fake").encode(["a", "b"])
    assert vectors.dtype == np.float32 and vectors.shape == (2, 3)
    assert model.encode_kwargs["normalize_embeddings"] is True
    assert model.encode_kwargs["show_progress_bar"] is False


@pytest.mark.parametrize("model", [_Model(fast=False), _Model(max_seq_length=2)])
def test_unsupported_models_rejected(model) -> None:
    with pytest.raises(RuntimeError):
        SentenceTransformerEmbedder(model, "fake")


def test_library_stderr_handlers_are_detached() -> None:
    import logging

    from backend.scoring.embeddings import LIBRARY_LOGGERS, route_library_logs_through_root

    for name in LIBRARY_LOGGERS:
        logging.getLogger(name).addHandler(logging.StreamHandler())
        logging.getLogger(name).propagate = False
    route_library_logs_through_root()
    for name in LIBRARY_LOGGERS:
        library_logger = logging.getLogger(name)
        assert library_logger.handlers == [] and library_logger.propagate is True


def test_loader_uses_pinned_revision_cpu_and_no_remote_code(monkeypatch) -> None:
    import sys
    import types

    from backend.core.config import Settings
    from backend.scoring.embeddings import load_sentence_transformer

    captured = {}

    class FakeSentenceTransformer(_Model):
        def __init__(self, name, **kwargs):
            super().__init__()
            captured.update(name=name, **kwargs)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    embedder = load_sentence_transformer(Settings(_env_file=None))
    assert captured == {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "device": "cpu",
        "trust_remote_code": False,
    }
    assert embedder.name == "sentence-transformers/all-MiniLM-L6-v2"
