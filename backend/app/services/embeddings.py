"""Text embedding providers.

Two backends are supported behind one small interface so the rest of the app
never cares which is active:

* ``SentenceTransformerEmbedder`` — real semantic embeddings using
  ``sentence-transformers`` (default; downloads a model on first use).
* ``HashingEmbedder`` — a deterministic, dependency-light bag-of-words hashing
  embedder. It needs no model download and is used (a) by the automated test
  suite for fast, offline, reproducible retrieval tests, and (b) as an opt-in
  diagnostic backend (``EMBEDDING_BACKEND=hash``) so the retrieval pipeline can
  be demonstrated even where the ML stack cannot be installed.

The hashing embedder is NOT a semantic model — it rewards literal word overlap.
It is good enough to prove the plumbing, not to ship as production search.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

from app.logging_config import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class EmbeddingProvider(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


class HashingEmbedder:
    """Deterministic hashed bag-of-words embeddings with L2 normalisation."""

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _hash(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % self.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            for token in _WORD_RE.findall(text.lower()):
                vec[self._hash(token)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)
        return vectors


class SentenceTransformerEmbedder:
    """Real semantic embeddings. The model is loaded lazily on first use."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._dimension: int | None = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "sentence-transformers is not installed. Install it with "
                    "`pip install sentence-transformers`, or set "
                    "EMBEDDING_BACKEND=hash for a lightweight offline backend."
                ) from exc
            logger.info("Loading embedding model '%s' ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            # The method was renamed across sentence-transformers versions;
            # prefer the new name and fall back for older releases.
            get_dim = getattr(
                self._model,
                "get_embedding_dimension",
                self._model.get_sentence_embedding_dimension,
            )
            self._dimension = get_dim()
        return self._model

    @property
    def dimension(self) -> int:
        self._ensure_model()
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [v.tolist() for v in vectors]


def build_embedder(backend: str, model_name: str) -> EmbeddingProvider:
    """Factory that maps a config string to a concrete provider."""
    backend = backend.lower().strip()
    if backend in ("hash", "hashing"):
        logger.info("Using HashingEmbedder (lightweight, non-semantic).")
        return HashingEmbedder()
    if backend in ("sentence_transformers", "sentence-transformers", "st"):
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown EMBEDDING_BACKEND: {backend!r}")
