"""Deterministic fake embedder for offline tests (no model download, no torch).

Texts are embedded as L2-normalised hashed bag-of-words vectors, so texts that
share words are similar and unrelated texts are not. ``chunk`` splits on words
with a small window to exercise multi-chunk documents. Every factory call and
encode call is counted.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

DIM = 256
WORDS_PER_CHUNK = 40
_WORD = re.compile(r"\w+", re.UNICODE)


class FakeEmbedder:
    name = "fake-hashed-bow"

    def __init__(self) -> None:
        self.encode_calls = 0
        self.chunk_calls = 0

    def chunk(self, text: str) -> list[str]:
        self.chunk_calls += 1
        words = text.split()
        return [
            " ".join(words[i : i + WORDS_PER_CHUNK]) for i in range(0, len(words), WORDS_PER_CHUNK)
        ]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self.encode_calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.casefold()):
                index = int(hashlib.sha256(word.encode()).hexdigest(), 16) % DIM
                out[row, index] += 1.0
            norm = np.linalg.norm(out[row])
            if norm:
                out[row] /= norm
        return out


class CountingFactory:
    """Embedder factory that records how many times the model is 'loaded'."""

    def __init__(self, embedder: FakeEmbedder | None = None, *, fail: bool = False) -> None:
        self.embedder = embedder or FakeEmbedder()
        self.calls = 0
        self.fail = fail

    def __call__(self) -> FakeEmbedder:
        self.calls += 1
        if self.fail:
            raise OSError("model files unavailable (simulated)")
        return self.embedder
