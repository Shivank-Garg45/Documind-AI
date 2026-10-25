"""Split page-aware text into overlapping chunks with provenance metadata.

Each chunk records the page it came from so that citations can point users to
the exact page. Chunking happens per page, which keeps page numbers exact at
the cost of not merging text across page boundaries — a deliberate, simple
trade-off appropriate for a portfolio MVP.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.pdf_parser import PageText


@dataclass
class Chunk:
    text: str
    page: int
    index: int  # global chunk order within the document
    metadata: dict = field(default_factory=dict)


def chunk_pages(
    pages: list[PageText],
    *,
    chunk_size: int = 800,
    overlap: int = 150,
) -> list[Chunk]:
    """Turn extracted pages into overlapping character chunks.

    ``chunk_size`` and ``overlap`` are measured in characters. A word-boundary
    aware sliding window is used so chunks do not split words.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    chunks: list[Chunk] = []
    global_index = 0
    step = chunk_size - overlap

    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        if len(text) <= chunk_size:
            chunks.append(Chunk(text=text, page=page.page, index=global_index))
            global_index += 1
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Extend to the next whitespace so we don't cut a word in half.
            if end < len(text):
                nxt = text.find(" ", end)
                if nxt != -1 and nxt - end < 40:
                    end = nxt
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, page=page.page, index=global_index))
                global_index += 1
            if end >= len(text):
                break
            start += step

    return chunks
