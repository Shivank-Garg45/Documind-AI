"""Shared pytest fixtures.

Key idea: tests use the deterministic ``HashingEmbedder`` and an in-memory
ChromaDB so the whole suite runs fast, offline, and without any model download
or API key. The LLM is never really called.
"""
from __future__ import annotations

import io
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.config import Settings
from app.deps import Container, get_container
from app.main import app
from app.services.embeddings import HashingEmbedder
from app.services.llm import LLMClient
from app.services.retriever import VectorStore


# --------------------------------------------------------------------------
# PDF generation helpers
# --------------------------------------------------------------------------
@pytest.fixture
def make_pdf() -> Callable[[list[str]], bytes]:
    """Return a factory that builds a real multi-page PDF from page texts."""

    def _make(pages: list[str]) -> bytes:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        for text in pages:
            y = 750
            for line in text.split("\n"):
                c.drawString(72, y, line)
                y -= 16
            c.showPage()
        c.save()
        return buf.getvalue()

    return _make


# --------------------------------------------------------------------------
# Container / app fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def test_settings() -> Settings:
    # No API key -> llm disabled; hash backend -> no model download.
    return Settings(
        OPENAI_API_KEY=None,
        EMBEDDING_BACKEND="hash",
        CHUNK_SIZE=400,
        CHUNK_OVERLAP=80,
        RETRIEVAL_TOP_K=5,
        MAX_UPLOAD_MB=10,
    )


@pytest.fixture
def container(test_settings: Settings) -> Container:
    """A fully in-memory container (no disk persistence, no network)."""
    c = Container.__new__(Container)  # bypass __init__ to avoid building the ST model
    c.settings = test_settings
    c.embedder = HashingEmbedder()
    c.store = VectorStore(c.embedder, persist_dir=None)
    c.llm = LLMClient(api_key=None, model="gpt-4o-mini")
    return c


@pytest.fixture
def client(container: Container):
    app.dependency_overrides[get_container] = lambda: container
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
