"""Tests for the retrieval pipeline (independent of the LLM)."""
from __future__ import annotations

from app.services.chunker import chunk_pages
from app.services.embeddings import HashingEmbedder
from app.services.pdf_parser import PageText
from app.services.retriever import VectorStore


def _store() -> VectorStore:
    return VectorStore(HashingEmbedder(), persist_dir=None)


def _index(store: VectorStore, doc_id: str, filename: str, pages: list[tuple[int, str]]):
    page_texts = [PageText(page=p, text=t) for p, t in pages]
    chunks = chunk_pages(page_texts, chunk_size=400, overlap=50)
    store.add_document(doc_id, filename, pages=len(pages), chunks=chunks)


def test_retrieve_returns_relevant_passage():
    store = _store()
    _index(
        store,
        "doc1",
        "animals.pdf",
        [
            (1, "The giraffe is the tallest living terrestrial animal."),
            (2, "Photosynthesis converts sunlight into chemical energy in plants."),
        ],
    )
    passages = store.retrieve("How tall is a giraffe?", top_k=1)
    assert len(passages) == 1
    assert "giraffe" in passages[0].text.lower()
    assert passages[0].page == 1
    assert passages[0].filename == "animals.pdf"


def test_retrieval_scoped_to_selected_document():
    store = _store()
    _index(store, "doc1", "a.pdf", [(1, "Quantum entanglement links particle states.")])
    _index(store, "doc2", "b.pdf", [(1, "The mitochondria is the powerhouse of the cell.")])

    # Restrict to doc2 -> must never return doc1's passage.
    passages = store.retrieve("quantum entanglement", document_ids=["doc2"], top_k=5)
    assert all(p.document_id == "doc2" for p in passages)


def test_empty_store_returns_no_passages():
    store = _store()
    assert store.retrieve("anything", top_k=5) == []


def test_delete_removes_vectors():
    store = _store()
    _index(store, "doc1", "a.pdf", [(1, "Deletable content about volcanoes.")])
    assert store.count_documents() == 1

    assert store.delete_document("doc1") is True
    assert store.count_documents() == 0
    # After deletion, its content must not be retrievable.
    assert store.retrieve("volcanoes", top_k=5) == []
    # Deleting again is a no-op (returns False).
    assert store.delete_document("doc1") is False


def test_scores_are_bounded():
    store = _store()
    _index(store, "doc1", "a.pdf", [(1, "Machine learning models learn from data.")])
    passages = store.retrieve("machine learning", top_k=1)
    assert 0.0 <= passages[0].score <= 1.0
