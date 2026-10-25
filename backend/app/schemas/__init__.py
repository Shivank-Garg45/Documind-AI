"""Pydantic request/response models for the DocuMind AI API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------


class DocumentSummary(BaseModel):
    """Metadata about one processed document."""

    document_id: str
    filename: str
    pages: int
    chunks: int
    status: Literal["ready", "processing", "error"] = "ready"
    uploaded_at: datetime
    error: str | None = None


class UploadResponse(BaseModel):
    document: DocumentSummary
    message: str = "Document processed successfully."


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict search to these documents. None = search all.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query must not be empty.")
        return v


class Passage(BaseModel):
    """A single retrieved chunk with its provenance."""

    document_id: str
    filename: str
    page: int
    text: str
    score: float = Field(description="Similarity score in [0, 1]; higher is closer.")


class RetrieveResponse(BaseModel):
    query: str
    passages: list[Passage]
    count: int


# --------------------------------------------------------------------------
# Chat / RAG
# --------------------------------------------------------------------------


class ChatRequest(RetrieveRequest):
    """A chat request is a retrieve request that also calls the LLM."""


class Citation(BaseModel):
    document_id: str
    filename: str
    page: int
    excerpt: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    used_llm: bool = Field(
        description="True if an LLM generated the answer; "
        "False for retrieval-only / fallback responses."
    )
    sufficient_evidence: bool = Field(
        description="False when no relevant passages were found for the query."
    )


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: Literal["ok"]
    llm_enabled: bool
    llm_provider: str
    embedding_backend: str
    documents_indexed: int


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
