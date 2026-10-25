"""Tests for the document upload/list/delete endpoints and validation."""
from __future__ import annotations


def _upload(client, make_pdf, pages, filename="sample.pdf"):
    pdf = make_pdf(pages)
    return client.post(
        "/api/documents",
        files={"file": (filename, pdf, "application/pdf")},
    )


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_enabled"] is False
    assert body["embedding_backend"] == "hash"


def test_upload_list_delete_roundtrip(client, make_pdf):
    # Upload
    resp = _upload(client, make_pdf, ["Content about renewable energy.", "Page two."])
    assert resp.status_code == 201, resp.text
    doc = resp.json()["document"]
    assert doc["filename"] == "sample.pdf"
    assert doc["pages"] == 2
    assert doc["chunks"] >= 1
    assert doc["status"] == "ready"
    doc_id = doc["document_id"]

    # List
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert any(d["document_id"] == doc_id for d in docs)

    # Delete
    resp = client.delete(f"/api/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    # List is empty again
    assert client.get("/api/documents").json()["documents"] == []


def test_delete_unknown_document_returns_404(client):
    resp = client.delete("/api/documents/does-not-exist")
    assert resp.status_code == 404


def test_reject_non_pdf_extension(client):
    resp = client.post(
        "/api/documents",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert resp.status_code == 400


def test_reject_empty_pdf(client):
    # Valid extension but zero bytes -> parser rejects it (422).
    resp = client.post(
        "/api/documents",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 422


def test_reject_malformed_pdf(client):
    resp = client.post(
        "/api/documents",
        files={"file": ("bad.pdf", b"%PDF-1.4 not really a pdf", "application/pdf")},
    )
    assert resp.status_code == 422


def test_document_isolation(client, make_pdf):
    """A query scoped to one document must not surface another's content."""
    r1 = _upload(client, make_pdf, ["Blockchain is a distributed ledger."], "chain.pdf")
    r2 = _upload(client, make_pdf, ["The Pacific is the largest ocean."], "ocean.pdf")
    id1 = r1.json()["document"]["document_id"]
    _ = r2  # second doc just needs to exist

    resp = client.post(
        "/api/retrieve",
        json={"query": "largest ocean", "document_ids": [id1]},
    )
    assert resp.status_code == 200
    passages = resp.json()["passages"]
    assert all(p["document_id"] == id1 for p in passages)
