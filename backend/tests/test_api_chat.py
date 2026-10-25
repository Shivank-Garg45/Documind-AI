"""Tests for the RAG chat/retrieve endpoints, including LLM error handling."""
from __future__ import annotations

import pytest

from app.services.llm import LLMClient, LLMError


def _upload(client, make_pdf, pages, filename="kb.pdf"):
    pdf = make_pdf(pages)
    return client.post(
        "/api/documents", files={"file": (filename, pdf, "application/pdf")}
    )


def test_retrieve_endpoint_validation(client):
    # Empty query is rejected by Pydantic -> 422.
    resp = client.post("/api/retrieve", json={"query": "   "})
    assert resp.status_code == 422


def test_chat_without_api_key_is_retrieval_only(client, make_pdf):
    _upload(client, make_pdf, ["Solar panels convert sunlight into electricity."])
    resp = client.post("/api/chat", json={"query": "What do solar panels do?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["used_llm"] is False  # no key configured
    assert body["sufficient_evidence"] is True
    assert len(body["citations"]) >= 1
    assert body["citations"][0]["page"] == 1


def test_chat_insufficient_evidence_when_no_documents(client):
    resp = client.post("/api/chat", json={"query": "anything at all"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sufficient_evidence"] is False
    assert body["citations"] == []
    assert "not contain enough information" in body["answer"].lower()


def test_chat_with_mocked_llm_success(client, container, make_pdf, monkeypatch):
    """With a (fake) key and a mocked OpenAI client, used_llm must be True."""
    _upload(client, make_pdf, ["The capital of France is Paris."])

    # Build a mock OpenAI client whose completion returns a canned answer.
    class _Msg:
        content = "The capital of France is Paris [1]."

    class _Choice:
        message = _Msg()

    class _Completion:
        choices = [_Choice()]

    class _Completions:
        def create(self, **kwargs):
            return _Completion()

    class _Chat:
        completions = _Completions()

    class _FakeOpenAI:
        def __init__(self, *a, **k):
            self.chat = _Chat()

    import openai

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    # Give the container's LLM a (fake) key so .enabled is True.
    container.llm = LLMClient(api_key="sk-test-fake-key", model="gpt-4o-mini")

    resp = client.post("/api/chat", json={"query": "What is the capital of France?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["used_llm"] is True
    assert "Paris" in body["answer"]


def test_chat_llm_failure_returns_502(client, container, make_pdf, monkeypatch):
    _upload(client, make_pdf, ["Some indexable content about turbines."])

    def _boom(self, question, passages):
        raise LLMError("The AI service is currently unavailable. Please try again later.")

    monkeypatch.setattr(LLMClient, "answer", _boom)
    container.llm = LLMClient(api_key="sk-test-fake-key", model="gpt-4o-mini")

    resp = client.post("/api/chat", json={"query": "Tell me about turbines"})
    assert resp.status_code == 502
    assert "unavailable" in resp.json()["detail"].lower()


def test_llm_client_disabled_without_key():
    client = LLMClient(api_key=None, model="gpt-4o-mini")
    assert client.enabled is False
    client2 = LLMClient(api_key="sk-your-key-here", model="gpt-4o-mini")
    assert client2.enabled is False  # placeholder is treated as "no key"


@pytest.mark.parametrize("key", ["sk-real-looking-key", "sk-abc123"])
def test_llm_client_enabled_with_key(key):
    assert LLMClient(api_key=key, model="gpt-4o-mini").enabled is True
