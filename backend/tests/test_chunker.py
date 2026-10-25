"""Tests for chunk creation and page metadata preservation."""
from __future__ import annotations

import pytest

from app.services.chunker import chunk_pages
from app.services.pdf_parser import PageText


def test_short_page_becomes_single_chunk():
    pages = [PageText(page=1, text="A short page.")]
    chunks = chunk_pages(pages, chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].index == 0


def test_long_page_splits_with_overlap():
    text = " ".join(f"word{i}" for i in range(500))
    pages = [PageText(page=3, text=text)]
    chunks = chunk_pages(pages, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    # Every chunk must keep the correct page number.
    assert all(c.page == 3 for c in chunks)
    # Indices are contiguous and ordered.
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_page_numbers_preserved_across_pages():
    pages = [
        PageText(page=1, text="alpha " * 100),
        PageText(page=2, text="beta " * 100),
    ]
    chunks = chunk_pages(pages, chunk_size=150, overlap=30)
    pages_seen = {c.page for c in chunks}
    assert pages_seen == {1, 2}


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_pages([PageText(page=1, text="x")], chunk_size=100, overlap=100)
