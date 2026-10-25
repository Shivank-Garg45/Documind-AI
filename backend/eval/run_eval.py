"""Reproducible retrieval evaluation for DocuMind AI.

Builds an in-memory index from ``dataset.json`` and measures, from ACTUAL runs
(no fabricated numbers):

* retrieval hit@k   — did a passage from the expected page appear in the top-k?
* keyword coverage  — do the retrieved passages contain the expected keywords?
* unanswerable      — for questions with no supporting evidence, is the top
                       similarity below ``--threshold`` (i.e. the system should
                       recognise it cannot answer)?

Usage:
    python eval/run_eval.py                       # hash backend (default, offline)
    EMBEDDING_BACKEND=sentence_transformers \\
        python eval/run_eval.py                   # real semantic embeddings
    python eval/run_eval.py --top-k 5 --threshold 0.15

Exit code is non-zero if any answerable question misses its expected page, so
this can gate CI if desired.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make `app` importable when run from the backend/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chunker import chunk_pages  # noqa: E402
from app.services.embeddings import build_embedder  # noqa: E402
from app.services.pdf_parser import PageText  # noqa: E402
from app.services.retriever import VectorStore  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_dataset() -> dict:
    return json.loads((HERE / "dataset.json").read_text("utf-8"))


def build_store(corpus: dict, backend: str) -> VectorStore:
    embedder = build_embedder(backend, "sentence-transformers/all-MiniLM-L6-v2")
    store = VectorStore(embedder, persist_dir=None)
    pages = [PageText(page=p["page"], text=p["text"]) for p in corpus["pages"]]
    chunks = chunk_pages(pages, chunk_size=400, overlap=80)
    store.add_document(
        "eval-doc", corpus["filename"], pages=len(pages), chunks=chunks
    )
    return store


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.15,
        help="Max top-1 score for a question to count as correctly unanswerable.",
    )
    args = parser.parse_args()

    backend = os.getenv("EMBEDDING_BACKEND", "hash")
    data = load_dataset()
    store = build_store(data["corpus"], backend)

    print(f"\nDocuMind AI — retrieval evaluation (backend={backend}, k={args.top_k})")
    print("=" * 74)

    answerable_total = 0
    page_hits = 0
    keyword_hits = 0
    unanswerable_total = 0
    unanswerable_correct = 0

    for item in data["questions"]:
        passages = store.retrieve(item["q"], top_k=args.top_k)
        top_score = passages[0].score if passages else 0.0
        pages_found = [p.page for p in passages]
        joined = " ".join(p.text.lower() for p in passages)

        if item["answerable"]:
            answerable_total += 1
            hit = item["expected_page"] in pages_found
            kw_ok = all(k.lower() in joined for k in item["expect_keywords"])
            page_hits += int(hit)
            keyword_hits += int(kw_ok)
            status = "OK " if hit else "MISS"
            print(f"[{status}] q: {item['q']}")
            print(
                f"        expected page {item['expected_page']}, "
                f"retrieved pages {pages_found}, keywords_ok={kw_ok}, "
                f"top_score={top_score:.3f}"
            )
        else:
            unanswerable_total += 1
            correct = top_score < args.threshold
            unanswerable_correct += int(correct)
            status = "OK " if correct else "WEAK"
            print(f"[{status}] q (unanswerable): {item['q']}")
            print(
                f"        top_score={top_score:.3f} "
                f"(< {args.threshold} => correctly uncertain: {correct})"
            )

    print("-" * 74)
    print("Results (measured, not estimated):")
    if answerable_total:
        print(
            f"  retrieval hit@{args.top_k}: {page_hits}/{answerable_total} "
            f"({100 * page_hits / answerable_total:.0f}%)"
        )
        print(
            f"  keyword coverage:  {keyword_hits}/{answerable_total} "
            f"({100 * keyword_hits / answerable_total:.0f}%)"
        )
    if unanswerable_total:
        print(
            f"  unanswerable flagged: {unanswerable_correct}/{unanswerable_total}"
        )
    print("=" * 74)
    print(
        "Note: the 'hash' backend rewards literal word overlap, not semantics. "
        "Run with EMBEDDING_BACKEND=sentence_transformers for a semantic eval.\n"
    )

    # Fail if any answerable question missed its page.
    return 0 if page_hits == answerable_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
