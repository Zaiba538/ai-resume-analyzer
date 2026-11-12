"""Sentence-Transformer embeddings: one model per process, token-exact chunking.

* The model is loaded once, at application start-up, from the configured name
  and pinned revision, on CPU, with remote code disabled. Requests never load it.
* ``chunk`` splits text with the model's own fast tokenizer so that every chunk
  fits the model's window (``max_seq_length`` minus the two special tokens).
  Nothing is silently truncated. Chunks are substrings of the input.
* ``encode`` returns L2-normalised float32 vectors. Inference is serialised by a
  lock: one model instance is shared by all requests.

Text passed here is never logged.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from starlette.concurrency import run_in_threadpool

from backend.core.config import Settings
from backend.core.errors import AppError

logger = logging.getLogger("backend.scoring.embeddings")

ENCODE_BATCH_SIZE = 4
SPECIAL_TOKENS = 2  # [CLS] and [SEP]

# Hugging Face libraries: no telemetry, no tokenizer thread pools in workers, and
# no progress bars (they write non-JSON lines to stderr, bypassing the log format).
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


class EmbeddingUnavailableError(AppError):
    status_code = 503
    code = "embedding_model_unavailable"
    message = "The semantic similarity model is not available. Please try again later."


class Embedder(Protocol):
    name: str

    def chunk(self, text: str) -> list[str]: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Adapter over a loaded ``SentenceTransformer``."""

    def __init__(self, model: object, name: str) -> None:
        self._model = model
        self.name = name
        self._tokenizer = model.tokenizer  # type: ignore[attr-defined]
        self._window = int(model.max_seq_length) - SPECIAL_TOKENS  # type: ignore[attr-defined]
        if self._window < 1 or not getattr(self._tokenizer, "is_fast", False):
            raise RuntimeError("embedding model needs a fast tokenizer and a positive window")
        self._lock = threading.Lock()

    def chunk(self, text: str) -> list[str]:
        # The full sequence is used only for offsets, never fed to the model, so
        # the tokenizer's "longer than max length" warning is suppressed.
        offsets = self._tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )["offset_mapping"]
        chunks: list[str] = []
        for start in range(0, len(offsets), self._window):
            window = offsets[start : start + self._window]
            piece = text[window[0][0] : window[-1][1]].strip()
            if piece:
                chunks.append(piece)
        return chunks

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            vectors = self._model.encode(  # type: ignore[attr-defined]
                list(texts),
                batch_size=ENCODE_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)


LIBRARY_LOGGERS = ("huggingface_hub", "transformers", "sentence_transformers")


def route_library_logs_through_root() -> None:
    """Detach the stderr handlers Hugging Face libraries install on import.

    They are created when the libraries are first imported, after the
    application configured logging, and would print unstructured text that
    bypasses the privacy formatter. Their records propagate to the root
    handler instead (where non-allowlisted messages are dropped).
    """
    for name in LIBRARY_LOGGERS:
        library_logger = logging.getLogger(name)
        library_logger.handlers.clear()
        library_logger.propagate = True
        library_logger.setLevel(logging.WARNING)


def load_sentence_transformer(settings: Settings) -> SentenceTransformerEmbedder:
    """Load the configured model (downloads it on first use). CPU only, no remote code."""
    from sentence_transformers import SentenceTransformer  # heavy import, done once

    route_library_logs_through_root()
    model = SentenceTransformer(
        settings.embedding_model_name,
        revision=settings.embedding_model_revision,
        device="cpu",
        trust_remote_code=False,
    )
    return SentenceTransformerEmbedder(model, settings.embedding_model_name)


class EmbeddingService:
    """Owns the single process-wide embedder and its readiness state."""

    def __init__(self, factory: Callable[[], Embedder]) -> None:
        self._factory = factory
        self._embedder: Embedder | None = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._embedder is not None

    def _load_sync(self) -> None:
        with self._lock:
            if self._embedder is None:
                self._embedder = self._factory()

    async def load(self) -> bool:
        """Load once; failures are logged by type only and leave the service unready."""
        try:
            await run_in_threadpool(self._load_sync)
        except Exception as exc:  # any failure: model unavailable, app stays alive
            logger.error(
                "embedding_model_load_failed",
                extra={"component": "embeddings", "error_code": type(exc).__name__},
            )
            return False
        logger.info("embedding_model_loaded", extra={"component": "embeddings"})
        return True

    def get(self) -> Embedder:
        if self._embedder is None:
            raise EmbeddingUnavailableError()
        return self._embedder
