"""PDF validation and text extraction (page-aware).

The parser works on raw bytes so that callers never have to trust a
user-supplied filename or write to disk before validating. Text is returned
per page so that downstream chunking can preserve accurate page numbers.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.logging_config import get_logger

logger = get_logger(__name__)

PDF_MAGIC = b"%PDF-"


class PdfValidationError(ValueError):
    """Raised when an uploaded file is not a usable PDF."""


@dataclass
class PageText:
    page: int  # 1-indexed
    text: str


def looks_like_pdf(data: bytes) -> bool:
    """Cheap signature check before doing any real parsing."""
    # The %PDF- header may be preceded by a few junk bytes in the wild.
    return PDF_MAGIC in data[:1024]


def extract_pages(data: bytes, *, max_bytes: int | None = None) -> list[PageText]:
    """Validate ``data`` and return non-empty pages of extracted text.

    Raises :class:`PdfValidationError` for empty, oversized, malformed, or
    text-less (likely scanned) PDFs.
    """
    if not data:
        raise PdfValidationError("The uploaded file is empty.")

    if max_bytes is not None and len(data) > max_bytes:
        raise PdfValidationError(
            f"File is too large ({len(data) / 1_048_576:.1f} MB). "
            f"Maximum allowed is {max_bytes / 1_048_576:.0f} MB."
        )

    if not looks_like_pdf(data):
        raise PdfValidationError(
            "File does not look like a PDF (missing %PDF header)."
        )

    try:
        reader = PdfReader(io.BytesIO(data))
    except (PdfReadError, OSError, ValueError) as exc:
        raise PdfValidationError(f"Could not parse PDF: {exc}") from exc

    if reader.is_encrypted:
        # Try an empty-password decrypt; bail out if it is truly locked.
        try:
            if reader.decrypt("") == 0:  # 0 == failure
                raise PdfValidationError(
                    "PDF is password-protected and cannot be read."
                )
        except (NotImplementedError, PdfReadError) as exc:
            raise PdfValidationError(
                "PDF uses an unsupported encryption scheme."
            ) from exc

    pages: list[PageText] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to extract text from page %d: %s", idx, exc)
            raw = ""
        cleaned = _normalise_whitespace(raw)
        if cleaned:
            pages.append(PageText(page=idx, text=cleaned))

    if not pages:
        raise PdfValidationError(
            "No extractable text found. The PDF may be scanned/image-only "
            "and would require OCR, which this service does not perform."
        )

    logger.info("Extracted text from %d/%d page(s).", len(pages), len(reader.pages))
    return pages


def _normalise_whitespace(text: str) -> str:
    """Collapse runaway whitespace while keeping word boundaries."""
    # Join hyphenated line breaks: "inter-\nnationalisation" -> "internationalisation"
    text = text.replace("-\n", "")
    lines = [line.strip() for line in text.splitlines()]
    joined = " ".join(line for line in lines if line)
    return " ".join(joined.split())
