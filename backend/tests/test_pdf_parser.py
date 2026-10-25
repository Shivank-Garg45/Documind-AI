"""Tests for PDF validation and page-aware text extraction."""
from __future__ import annotations

import pytest

from app.services.pdf_parser import PdfValidationError, extract_pages, looks_like_pdf


def test_extract_pages_returns_page_numbers(make_pdf):
    pdf = make_pdf(["Hello from page one.", "Second page content here."])
    pages = extract_pages(pdf)
    assert [p.page for p in pages] == [1, 2]
    assert "page one" in pages[0].text
    assert "Second page" in pages[1].text


def test_extract_skips_blank_pages(make_pdf):
    pdf = make_pdf(["Real content.", "   ", "More content."])
    pages = extract_pages(pdf)
    # The blank middle page is dropped, but page numbers stay truthful.
    assert [p.page for p in pages] == [1, 3]


def test_empty_bytes_rejected():
    with pytest.raises(PdfValidationError, match="empty"):
        extract_pages(b"")


def test_non_pdf_rejected():
    with pytest.raises(PdfValidationError, match="%PDF"):
        extract_pages(b"this is clearly not a pdf file" * 10)


def test_malformed_pdf_rejected():
    # Has the header but is otherwise garbage.
    with pytest.raises(PdfValidationError):
        extract_pages(b"%PDF-1.4\n garbage bytes that do not form a pdf")


def test_oversized_pdf_rejected(make_pdf):
    pdf = make_pdf(["content"])
    with pytest.raises(PdfValidationError, match="too large"):
        extract_pages(pdf, max_bytes=10)


def test_looks_like_pdf():
    assert looks_like_pdf(b"%PDF-1.7 ...")
    assert not looks_like_pdf(b"not a pdf")
