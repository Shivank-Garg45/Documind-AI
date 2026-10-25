"""Application dependency container.

Builds the long-lived singletons (embedder, vector store, LLM client) from
configuration and exposes them to routes via FastAPI's dependency system.
Tests override :func:`get_container` to inject fast, offline components.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.services.embeddings import EmbeddingProvider, build_embedder
from app.services.llm import BaseLLMClient, GeminiClient, OpenAIClient
from app.services.retriever import VectorStore

logger = get_logger(__name__)


def build_llm(settings: Settings) -> BaseLLMClient:
    """Pick an LLM provider from configuration."""
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        return OpenAIClient(
            settings.openai_api_key,
            settings.openai_model,
            timeout=settings.llm_timeout_seconds,
        )
    if provider == "gemini":
        return GeminiClient(
            settings.google_api_key,
            settings.gemini_model,
            timeout=settings.llm_timeout_seconds,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")


class Container:
    """Holds wired-up services for the lifetime of the process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder: EmbeddingProvider = build_embedder(
            settings.embedding_backend, settings.embedding_model
        )
        self.store = VectorStore(self.embedder, persist_dir=settings.chroma_path)
        self.llm = build_llm(settings)


@lru_cache
def get_container() -> Container:
    settings = get_settings()
    logger.info(
        "Starting DocuMind AI | embedding_backend=%s llm_enabled=%s",
        settings.embedding_backend,
        settings.llm_enabled,
    )
    return Container(settings)
