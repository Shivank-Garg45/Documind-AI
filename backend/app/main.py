"""DocuMind AI — FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.deps import Container, get_container
from app.logging_config import configure_logging, get_logger
from app.routes import chat, documents
from app.schemas import HealthResponse

configure_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="DocuMind AI",
    description=(
        "A Retrieval-Augmented Generation (RAG) API for asking natural-language "
        "questions about your PDF documents, with page-level citations."
    ),
    version="1.0.0",
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health(container: Container = Depends(get_container)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_enabled=container.llm.enabled,
        llm_provider=container.settings.llm_provider,
        embedding_backend=container.settings.embedding_backend,
        documents_indexed=container.store.count_documents(),
    )


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "DocuMind AI", "docs": "/docs", "health": "/api/health"}
