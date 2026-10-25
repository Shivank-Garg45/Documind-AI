"""Document endpoints: upload, list, delete."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from app.deps import Container, get_container
from app.logging_config import get_logger
from app.schemas import (
    DeleteResponse,
    DocumentListResponse,
    UploadResponse,
)
from app.services.chunker import chunk_pages
from app.services.pdf_parser import PdfValidationError, extract_pages

logger = get_logger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])

_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream", "application/x-pdf"}


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    container: Container = Depends(get_container),
) -> UploadResponse:
    settings = container.settings

    # --- validate filename & declared type -------------------------------
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pdf files are supported.",
        )
    if file.content_type and file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type: {file.content_type}",
        )

    data = await file.read()

    # --- validate + extract (bytes-based, never trusts the filename) -----
    try:
        pages = extract_pages(data, max_bytes=settings.max_upload_bytes)
    except PdfValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc

    chunks = chunk_pages(
        pages,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text chunks could be produced from this document.",
        )

    # Generate our own id; never derive storage identity from user input.
    document_id = uuid.uuid4().hex
    page_count = pages[-1].page if pages else 0

    try:
        summary = container.store.add_document(
            document_id=document_id,
            filename=filename,
            pages=page_count,
            chunks=chunks,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Indexing failed for %s: %s", filename, type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to index the document.",
        ) from exc

    return UploadResponse(document=summary)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    container: Container = Depends(get_container),
) -> DocumentListResponse:
    return DocumentListResponse(documents=container.store.list_documents())


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    container: Container = Depends(get_container),
) -> DeleteResponse:
    deleted = container.store.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    return DeleteResponse(
        document_id=document_id,
        deleted=True,
        message="Document and its vectors were removed.",
    )
