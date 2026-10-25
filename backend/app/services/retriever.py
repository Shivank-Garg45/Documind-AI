"""Vector storage and semantic retrieval backed by ChromaDB.

This module is the *retrieval pipeline*. It is deliberately independent of the
LLM so it can be tested on its own (see ``tests/test_retriever.py``). An
:class:`EmbeddingProvider` is injected, which also lets tests swap in a fast,
deterministic embedder.

A small JSON registry tracks document-level metadata (filename, page count,
chunk count, status) because ChromaDB stores chunks, not documents.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.logging_config import get_logger
from app.schemas import DocumentSummary, Passage
from app.services.chunker import Chunk
from app.services.embeddings import EmbeddingProvider

logger = get_logger(__name__)

_COLLECTION = "documind_chunks"


class VectorStore:
    """Chunk-level vector store + document registry."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        *,
        persist_dir: Path | str | None = None,
    ) -> None:
        self.embedder = embedder
        self._lock = threading.Lock()

        chroma_settings = ChromaSettings(anonymized_telemetry=False)
        if persist_dir is None:
            # In-memory (used by tests). ChromaDB's in-process system is shared
            # between EphemeralClient instances, so a unique collection name per
            # store is required to keep instances (and tests) isolated.
            self._client = chromadb.EphemeralClient(settings=chroma_settings)
            collection_name = f"{_COLLECTION}_{uuid.uuid4().hex}"
            self._registry_path: Path | None = None
            self._registry: dict[str, dict] = {}
        else:
            persist_dir = Path(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir), settings=chroma_settings
            )
            collection_name = _COLLECTION
            self._registry_path = persist_dir / "registry.json"
            self._registry = self._load_registry()

        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # -- registry persistence --------------------------------------------
    def _load_registry(self) -> dict[str, dict]:
        if self._registry_path and self._registry_path.exists():
            try:
                return json.loads(self._registry_path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read registry, starting fresh: %s", exc)
        return {}

    def _save_registry(self) -> None:
        if self._registry_path:
            self._registry_path.write_text(
                json.dumps(self._registry, indent=2), "utf-8"
            )

    # -- writes -----------------------------------------------------------
    def add_document(
        self, document_id: str, filename: str, pages: int, chunks: list[Chunk]
    ) -> DocumentSummary:
        """Embed and store all chunks of a document, then register it."""
        if not chunks:
            raise ValueError("Cannot index a document with no chunks.")

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)
        ids = [f"{document_id}:{c.index}" for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "filename": filename,
                "page": c.page,
            }
            for c in chunks
        ]

        with self._lock:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            summary = DocumentSummary(
                document_id=document_id,
                filename=filename,
                pages=pages,
                chunks=len(chunks),
                status="ready",
                uploaded_at=datetime.now(UTC),
            )
            self._registry[document_id] = summary.model_dump(mode="json")
            self._save_registry()

        logger.info(
            "Indexed document %s (%s): %d chunk(s).",
            document_id,
            filename,
            len(chunks),
        )
        return summary

    def delete_document(self, document_id: str) -> bool:
        """Remove a document's chunks AND its registry entry."""
        with self._lock:
            if document_id not in self._registry:
                return False
            self._collection.delete(where={"document_id": document_id})
            self._registry.pop(document_id, None)
            self._save_registry()
        logger.info("Deleted document %s and its vectors.", document_id)
        return True

    # -- reads ------------------------------------------------------------
    def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            items = list(self._registry.values())
        docs = [DocumentSummary(**item) for item in items]
        return sorted(docs, key=lambda d: d.uploaded_at, reverse=True)

    def get_document(self, document_id: str) -> DocumentSummary | None:
        item = self._registry.get(document_id)
        return DocumentSummary(**item) if item else None

    def count_documents(self) -> int:
        return len(self._registry)

    def retrieve(
        self,
        query: str,
        *,
        document_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[Passage]:
        """Return the ``top_k`` most relevant passages for ``query``.

        When ``document_ids`` is given, retrieval is scoped to those documents
        only — this is what prevents answers leaking across unrelated uploads.
        """
        if self._collection.count() == 0:
            return []

        where: dict | None = None
        if document_ids:
            where = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": document_ids}}
            )

        query_embedding = self.embedder.embed([query])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        passages: list[Passage] = []
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists, strict=False):
            # cosine distance -> similarity score in [0, 1]
            score = max(0.0, min(1.0, 1.0 - float(dist)))
            passages.append(
                Passage(
                    document_id=str(meta.get("document_id", "")),
                    filename=str(meta.get("filename", "")),
                    page=int(meta.get("page", 0)),
                    text=str(text),
                    score=round(score, 4),
                )
            )
        return passages
