"""RAG endpoints: retrieval-only and full chat (retrieve + generate)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.deps import Container, get_container
from app.logging_config import get_logger
from app.schemas import (
    ChatRequest,
    ChatResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.llm import INSUFFICIENT_MSG, LLMError, passages_to_citations

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["rag"])


def _resolve_top_k(requested: int | None, default: int) -> int:
    return requested if requested else default


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(
    req: RetrieveRequest,
    container: Container = Depends(get_container),
) -> RetrieveResponse:
    """Semantic search only — no LLM call. Useful for testing retrieval."""
    top_k = _resolve_top_k(req.top_k, container.settings.retrieval_top_k)
    passages = container.store.retrieve(
        req.query, document_ids=req.document_ids, top_k=top_k
    )
    return RetrieveResponse(query=req.query, passages=passages, count=len(passages))


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    container: Container = Depends(get_container),
) -> ChatResponse:
    """Full RAG: retrieve evidence, then ask the LLM to answer from it."""
    top_k = _resolve_top_k(req.top_k, container.settings.retrieval_top_k)
    passages = container.store.retrieve(
        req.query, document_ids=req.document_ids, top_k=top_k
    )

    if not passages:
        # No evidence at all -> never fabricate an answer.
        return ChatResponse(
            query=req.query,
            answer=INSUFFICIENT_MSG,
            citations=[],
            used_llm=False,
            sufficient_evidence=False,
        )

    try:
        result = container.llm.answer(req.query, passages)
    except LLMError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc

    return ChatResponse(
        query=req.query,
        answer=result.answer,
        citations=passages_to_citations(passages),
        used_llm=result.used_llm,
        sufficient_evidence=True,
    )
