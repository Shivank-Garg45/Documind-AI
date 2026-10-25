"""Tests for the pluggable LLM provider layer (OpenAI + Gemini)."""
from __future__ import annotations

import pytest

from app.config import Settings
from app.deps import build_llm
from app.schemas import Passage
from app.services.llm import GeminiClient, OpenAIClient


def _passages():
    return [
        Passage(
            document_id="d",
            filename="f.pdf",
            page=1,
            text="The capital of France is Paris.",
            score=0.9,
        )
    ]


def test_factory_selects_gemini():
    s = Settings(LLM_PROVIDER="gemini", GOOGLE_API_KEY="k", EMBEDDING_BACKEND="hash")
    assert isinstance(build_llm(s), GeminiClient)


def test_factory_selects_openai():
    s = Settings(LLM_PROVIDER="openai", OPENAI_API_KEY="sk-x", EMBEDDING_BACKEND="hash")
    assert isinstance(build_llm(s), OpenAIClient)


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        build_llm(Settings(LLM_PROVIDER="not-a-provider"))


def test_gemini_disabled_without_key():
    assert GeminiClient(None, "gemini-2.0-flash").enabled is False
    # The .env placeholder must count as "no key".
    assert GeminiClient("your-google-api-key-here", "gemini-2.0-flash").enabled is False


def test_gemini_enabled_with_key():
    assert GeminiClient("AIza-looks-real", "gemini-2.0-flash").enabled is True


def test_gemini_retrieval_only_without_key():
    result = GeminiClient(None, "gemini-2.0-flash").answer("q?", _passages())
    assert result.used_llm is False
    assert "Paris" in result.answer  # falls back to the top passage


def test_gemini_generate_with_mocked_sdk(monkeypatch):
    class _Resp:
        text = "The capital of France is Paris [1]."

    class _Models:
        def generate_content(self, **kwargs):
            return _Resp()

    class _FakeClient:
        def __init__(self, *a, **k):
            self.models = _Models()

    from google import genai

    monkeypatch.setattr(genai, "Client", _FakeClient)

    client = GeminiClient("AIza-looks-real", "gemini-2.0-flash")
    result = client.answer("What is the capital of France?", _passages())
    assert result.used_llm is True
    assert "Paris" in result.answer
