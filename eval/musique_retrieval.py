"""MuSiQue per-question paragraph retrieval helpers.

MuSiQue gives each question its own candidate paragraph set. These helpers use
the same retrieval shape as the HotpotQA distractor lane: gte bi-encoder scores
the candidate paragraphs attached to the current question, then MiniLM CE
reranks the per-question candidate list.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from langchain_core.documents import Document

from rag_utils import get_embeddings, rerank_with_cross_encoder


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "datasets" / "musique"
PASSAGES_PATH = DATA_DIR / "passages.csv"


def _norm(value: Any) -> str:
    return str(value or "").strip()


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


@lru_cache(maxsize=1)
def load_musique_paragraphs() -> dict[str, list[dict[str, Any]]]:
    if not PASSAGES_PATH.exists():
        raise FileNotFoundError(
            f"MuSiQue passages not found at {PASSAGES_PATH}. "
            "Run scripts/download_musique_validation.py first."
        )
    df = pd.read_csv(PASSAGES_PATH, keep_default_na=False)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        q_id = _norm(row.get("q_id"))
        item = {
            "q_id": q_id,
            "idx": _norm(row.get("idx")),
            "para_idx": int(row.get("para_idx", 0) or 0),
            "title": _norm(row.get("title")),
            "text": _norm(row.get("text")),
            "is_supporting": _boolish(row.get("is_supporting", False)),
        }
        grouped.setdefault(q_id, []).append(item)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["para_idx"]))
    return grouped


def paragraph_text(paragraph: dict[str, Any]) -> str:
    title = _norm(paragraph.get("title"))
    text = _norm(paragraph.get("text"))
    return f"{title}\n{text}".strip() if title else text


def musique_documents_by_idx(idxs: Iterable[str]) -> list[Document]:
    wanted = [str(idx) for idx in idxs if str(idx)]
    if not wanted:
        return []
    by_idx: dict[str, dict[str, Any]] = {}
    for rows in load_musique_paragraphs().values():
        for paragraph in rows:
            by_idx.setdefault(str(paragraph["idx"]), paragraph)
    docs: list[Document] = []
    for idx in wanted:
        paragraph = by_idx.get(idx)
        if not paragraph:
            continue
        docs.append(_document_from_paragraph(paragraph))
    return docs


def _document_from_paragraph(paragraph: dict[str, Any], dense_score: float | None = None) -> Document:
    metadata = {
        "idx": _norm(paragraph.get("idx")),
        "q_id": _norm(paragraph.get("q_id")),
        "source": "musique",
        "title": _norm(paragraph.get("title")),
        "context_title": _norm(paragraph.get("title")),
        "is_supporting": bool(paragraph.get("is_supporting", False)),
        "para_idx": int(paragraph.get("para_idx", 0) or 0),
    }
    if dense_score is not None:
        metadata["dense_score"] = float(dense_score)
    return Document(page_content=paragraph_text(paragraph), metadata=metadata)


_PARAGRAPH_EMBED_CACHE: dict[tuple[str, str], np.ndarray] = {}


def _paragraph_embeddings(
    q_id: str,
    paragraphs: list[dict[str, Any]],
    embedding_model: str | None,
) -> np.ndarray:
    model_key = embedding_model or os.getenv("EMBEDDING_MODEL", "") or "default"
    cache_key = (model_key, q_id)
    cached = _PARAGRAPH_EMBED_CACHE.get(cache_key)
    if cached is not None:
        return cached
    embeddings = get_embeddings(embedding_model)
    texts = [paragraph_text(paragraph) for paragraph in paragraphs]
    matrix = np.asarray(embeddings.embed_documents(texts), dtype=np.float32)
    _PARAGRAPH_EMBED_CACHE[cache_key] = matrix
    return matrix


def retrieve_musique_documents(
    row: Any,
    queries: list[str],
    *,
    k: int = 10,
    rerank_query: str | None = None,
    embedding_model: str | None = None,
) -> list[Document]:
    """Retrieve/rerank within one MuSiQue candidate paragraph set."""
    q_id = _norm(row.get("idx", ""))
    paragraphs = load_musique_paragraphs().get(q_id, [])
    if not paragraphs:
        return []

    clean_queries = [str(query).strip() for query in queries if str(query).strip()]
    if not clean_queries:
        clean_queries = [_norm(row.get("question", ""))]
    embeddings = get_embeddings(embedding_model)
    query_matrix = np.asarray(embeddings.embed_documents(clean_queries), dtype=np.float32)
    doc_matrix = _paragraph_embeddings(q_id, paragraphs, embedding_model)
    dense_scores = np.max(query_matrix @ doc_matrix.T, axis=0)
    dense_order = np.argsort(-dense_scores)

    dense_docs = [
        _document_from_paragraph(paragraphs[int(i)], dense_score=float(dense_scores[int(i)]))
        for i in dense_order
    ]
    ce_query = str(rerank_query or clean_queries[0])
    return rerank_with_cross_encoder(ce_query, dense_docs, top_k=min(k, len(dense_docs)))
