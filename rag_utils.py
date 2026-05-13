import logging
import os
import re
import threading
import numpy as np
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")
COLLECTION_NAME = "legal_passages"

# Embedding model — configurable via env var. Default: gte-large-en-v1.5 (1024d, 8192 tokens)
DEFAULT_EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"


# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

_embeddings_instances: Dict[str, HuggingFaceEmbeddings] = {}
_embeddings_lock = threading.Lock()


def _repair_position_ids(embeddings: HuggingFaceEmbeddings) -> None:
    """Repair remote-code embedders whose non-persistent position buffer is stale.

    Alibaba GTE's `new-impl` module reads `embeddings.position_ids` for RoPE.
    On some cluster loads that buffer can contain uninitialized values, causing
    immediate index errors even for a four-token query. Rebuild it after load.
    """
    try:
        import torch

        transformer = embeddings._client[0]
        auto_model = getattr(transformer, "auto_model", None)
        emb_module = getattr(auto_model, "embeddings", None)
        if emb_module is None or not hasattr(emb_module, "position_ids"):
            return

        max_positions = int(getattr(auto_model.config, "max_position_embeddings", 0))
        if max_positions <= 0:
            max_positions = int(emb_module.position_ids.numel())
        device = emb_module.word_embeddings.weight.device
        emb_module.register_buffer(
            "position_ids",
            torch.arange(max_positions, device=device, dtype=torch.long),
            persistent=False,
        )
        print(f"[rag_utils] Reinitialized embedding position_ids={max_positions}")
    except Exception as exc:
        logger.warning("Could not repair embedding position_ids: %s", exc)


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    """Get a cached embedding model instance. Supports multiple models for A/B testing.

    Args:
        model_name: HuggingFace model ID. Defaults to EMBEDDING_MODEL env var or DEFAULT_EMBEDDING_MODEL.
    """
    if model_name is None:
        model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    if model_name in _embeddings_instances:
        return _embeddings_instances[model_name]

    with _embeddings_lock:
        if model_name not in _embeddings_instances:
            print(f"[rag_utils] Loading embedding model: {model_name}")
            device = os.getenv("EMBEDDING_DEVICE", "").strip()
            backend = os.getenv("EMBEDDING_BACKEND", "").strip()
            model_kwargs = {"trust_remote_code": True}
            if device:
                model_kwargs["device"] = device
            if backend:
                model_kwargs["backend"] = backend
            fp16_requested = os.getenv("EMBEDDING_FP16", "").strip().lower() in {"1", "true", "yes", "on"}
            if fp16_requested and (not backend or backend == "torch") and device.lower() != "cpu":
                try:
                    import torch
                    model_kwargs["model_kwargs"] = {"dtype": torch.float16}
                except Exception as exc:
                    logger.warning("Could not enable fp16 embedding load: %s", exc)
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
            _repair_position_ids(embeddings)
            # Keep online query embeddings aligned with utils/fast_embed.py,
            # which built the Chroma collections with SentenceTransformer
            # max_seq_length=512. Let callers override only for experiments.
            max_seq_raw = os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "512").strip()
            if max_seq_raw:
                try:
                    embeddings._client.max_seq_length = int(max_seq_raw)
                    print(f"[rag_utils] Embedding max_seq_length={embeddings._client.max_seq_length}")
                except Exception as exc:
                    logger.warning("Could not set embedding max_seq_length=%r: %s", max_seq_raw, exc)
            _embeddings_instances[model_name] = embeddings
    return _embeddings_instances[model_name]


_cross_encoder_instance = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder():
    """Cached cross-encoder for reranking (ms-marco-MiniLM-L-6-v2)."""
    global _cross_encoder_instance
    if _cross_encoder_instance is not None:
        return _cross_encoder_instance

    with _cross_encoder_lock:
        if _cross_encoder_instance is None:
            device = os.getenv("CROSS_ENCODER_DEVICE", "").strip()
            kwargs = {"device": device} if device else {}
            _cross_encoder_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs)
    return _cross_encoder_instance


def rerank_with_cross_encoder(
    query: str,
    docs: List[Document],
    top_k: int = 5,
) -> List[Document]:
    """Rerank documents using cross-encoder. Stores score in metadata."""
    if not docs:
        return []
    if os.getenv("DISABLE_CROSS_ENCODER", "").strip() in {"1", "true", "TRUE", "yes"}:
        result = []
        for doc in docs[:top_k]:
            doc.metadata["cross_encoder_score"] = 0.0
            result.append(doc)
        return result

    cross_encoder = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    result = []
    for doc, score in scored[:top_k]:
        doc.metadata["cross_encoder_score"] = float(score)
        result.append(doc)
    return result


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

_vectorstore_instances: Dict[str, Chroma] = {}
_vectorstore_lock = threading.Lock()

def get_vectorstore(collection_name: str = COLLECTION_NAME,
                    embedding_model: str = None) -> Chroma:
    """Returns a Chroma vector store singleton for the given collection.

    For embedding A/B testing, pass embedding_model to pair the correct
    model with the correct collection. The collection name should already
    include any model suffix (e.g., 'legal_passages__bge_m3').
    """
    # Cache key includes both collection and model to avoid mismatches
    cache_key = f"{collection_name}::{embedding_model or 'default'}"
    if cache_key in _vectorstore_instances:
        return _vectorstore_instances[cache_key]

    with _vectorstore_lock:
        if cache_key not in _vectorstore_instances:
            embeddings = get_embeddings(embedding_model)
            _vectorstore_instances[cache_key] = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=CHROMA_DB_DIR,
            )
    return _vectorstore_instances[cache_key]


def get_documents_by_idx(
    collection_name: str,
    idxs: List[str],
    embedding_model: str = None,
) -> List[Document]:
    """Fetch Chroma documents by corpus `idx`, preserving requested order.

    `utils/fast_embed.py` writes Chroma ids as `doc_{idx}` and stores the same
    value in metadata field `idx`. The direct id lookup is fast; the metadata
    fallback keeps older collections usable if their Chroma ids differ.
    """
    requested = [str(idx) for idx in idxs if str(idx) != ""]
    if not requested:
        return []

    try:
        import chromadb

        client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        collection = client.get_collection(collection_name)
    except Exception as exc:
        logger.debug("Direct Chroma client lookup failed for %s: %s", collection_name, exc)
        vectorstore = get_vectorstore(collection_name, embedding_model=embedding_model)
        collection = vectorstore._collection
    found: Dict[str, Document] = {}

    def _store_batch(batch: dict) -> None:
        ids = batch.get("ids") or []
        documents = batch.get("documents") or []
        metadatas = batch.get("metadatas") or []
        for chroma_id, text, metadata in zip(ids, documents, metadatas):
            metadata = dict(metadata or {})
            idx = str(metadata.get("idx") or str(chroma_id).removeprefix("doc_"))
            if idx and idx not in found:
                metadata.setdefault("idx", idx)
                found[idx] = Document(page_content=text or "", metadata=metadata)

    try:
        _store_batch(collection.get(ids=[f"doc_{idx}" for idx in requested], include=["documents", "metadatas"]))
    except Exception as exc:
        logger.debug("Direct Chroma id lookup failed for %s: %s", collection_name, exc)

    missing = [idx for idx in requested if idx not in found]
    if missing:
        try:
            _store_batch(collection.get(where={"idx": {"$in": missing}}, include=["documents", "metadatas"]))
        except Exception as exc:
            logger.debug("Batch metadata lookup failed for %s: %s", collection_name, exc)

    missing = [idx for idx in requested if idx not in found]
    for idx in missing:
        try:
            _store_batch(collection.get(where={"idx": idx}, include=["documents", "metadatas"]))
        except Exception as exc:
            logger.debug("Metadata lookup failed for %s idx=%s: %s", collection_name, idx, exc)

    return [found[idx] for idx in requested if idx in found]

# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

_bm25_indices: Dict[str, Dict] = {}  # keyed by collection name
_bm25_locks: Dict[str, threading.Lock] = {}
_bm25_locks_guard = threading.Lock()


def _tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric."""
    return re.findall(r"[a-z0-9]+", text.lower())


BM25_MAX_CORPUS = 1_000_000  # Skip BM25 for collections larger than this


def _get_bm25_lock(collection_name: str) -> threading.Lock:
    with _bm25_locks_guard:
        return _bm25_locks.setdefault(collection_name, threading.Lock())


def get_bm25_index(vectorstore: Chroma = None) -> Dict:
    """Build or return cached BM25 index for a specific collection."""
    vs = vectorstore or get_vectorstore()
    collection = vs._collection
    coll_name = collection.name

    if coll_name in _bm25_indices:
        return _bm25_indices[coll_name]

    with _get_bm25_lock(coll_name):
        if coll_name in _bm25_indices:
            return _bm25_indices[coll_name]

        count = collection.count()

        if count > BM25_MAX_CORPUS:
            raise RuntimeError(
                f"Collection has {count:,} docs (>{BM25_MAX_CORPUS:,}). "
                f"BM25 index skipped to avoid OOM. Using dense-only retrieval."
            )

        print(f"[rag_utils] Building BM25 index from {count} passages...")

        all_docs = []
        tokenized_corpus = []
        batch_size = 5000
        for offset in range(0, count, batch_size):
            batch = collection.get(
                offset=offset,
                limit=batch_size,
                include=["documents", "metadatas"],
            )
            for text, meta in zip(batch["documents"], batch["metadatas"]):
                doc = Document(page_content=text, metadata=meta)
                all_docs.append(doc)
                tokenized_corpus.append(_tokenize(text))

        bm25 = BM25Okapi(tokenized_corpus)
        index = {"bm25": bm25, "docs": all_docs}
        _bm25_indices[coll_name] = index
        print(f"[rag_utils] BM25 index built ({count} docs) for '{coll_name}'")
        return index


def _retrieve_bm25(query: str, k: int, exclude_ids: set = None,
                   vectorstore: Chroma = None) -> List[Document]:
    """First-stage BM25 retrieval. Returns raw candidates (no reranking)."""
    index = get_bm25_index(vectorstore)
    bm25 = index["bm25"]
    docs = index["docs"]

    scores = bm25.get_scores(_tokenize(query))
    top_indices = np.argsort(scores)[::-1]

    results = []
    for i in top_indices:
        if len(results) >= k:
            break
        doc = docs[i]
        idx = doc.metadata.get("idx", "")
        if exclude_ids and idx in exclude_ids:
            continue
        doc.metadata["bm25_score"] = float(scores[i])
        results.append(doc)
    return results


def _retrieve_dense(query: str, k: int, exclude_ids: set = None,
                    vectorstore: Chroma = None,
                    where: Dict = None) -> List[Document]:
    """First-stage bi-encoder retrieval. Returns raw candidates (no reranking).

    Args:
        where: Optional ChromaDB metadata filter, e.g. {"source": "mbe"}.
    """
    vs = vectorstore or get_vectorstore()
    search_kwargs = {"k": k}
    if where:
        search_kwargs["filter"] = where
    retriever = vs.as_retriever(search_kwargs=search_kwargs)
    candidates = retriever.invoke(query)
    if exclude_ids:
        candidates = [d for d in candidates if d.metadata.get("idx", "") not in exclude_ids]
    return candidates


def _pool_and_dedup(doc_lists: List[List[Document]]) -> List[Document]:
    """Pool multiple candidate lists, deduplicate by idx."""
    combined = []
    seen = set()
    for docs in doc_lists:
        for doc in docs:
            idx = doc.metadata.get("idx", "")
            if idx not in seen:
                seen.add(idx)
                combined.append(doc)
    return combined


# ---------------------------------------------------------------------------
# Public retrieval API (hybrid: BM25 + dense → cross-encoder rerank)
# ---------------------------------------------------------------------------

def retrieve_documents(query: str, k: int = 5, exclude_ids: set = None,
                       vectorstore: Chroma = None, use_bm25: bool = False,
                       where: Dict = None, rerank_query: str = None) -> List[Document]:
    """Hybrid retrieval: BM25 + bi-encoder candidates → cross-encoder rerank.

    Both first-stage retrievers fetch k*3 candidates each. The combined pool
    is deduplicated by idx, then the cross-encoder picks the best k.
    Falls back to dense-only if BM25 index build fails (e.g., OOM on large corpora).

    Args:
        where: Optional ChromaDB metadata filter, e.g. {"source": "mbe"}.
        rerank_query: If provided, cross-encoder reranks against this text instead of
            the dense retrieval query. Useful for HyDE: embed(hyde_passage) but rerank(question).
    """
    vs = vectorstore or get_vectorstore()
    fetch_k = k * 3

    dense_docs = _retrieve_dense(query, k=fetch_k, exclude_ids=exclude_ids, vectorstore=vs, where=where)

    bm25_docs = []
    if use_bm25:
        try:
            bm25_docs = _retrieve_bm25(query, k=fetch_k, exclude_ids=exclude_ids, vectorstore=vs)
        except Exception as e:
            logger.warning("BM25 retrieval failed (falling back to dense-only): %s", e)

    candidates = _pool_and_dedup([dense_docs, bm25_docs])
    return rerank_with_cross_encoder(rerank_query or query, candidates, top_k=k)


def retrieve_documents_multi_query(queries: List[str], k: int = 5,
                                   exclude_ids: set = None,
                                   vectorstore: Chroma = None,
                                   use_bm25: bool = False,
                                   where: Dict = None,
                                   rerank_query: str = None) -> List[Document]:
    """Multi-query hybrid retrieval: pool BM25 + dense across all query variants.

    Each query variant contributes candidates from both BM25 and bi-encoder.
    All candidates are pooled and deduplicated. The cross-encoder reranks the
    full pool against the PRIMARY query (first in list), or rerank_query if provided.

    Args:
        where: Optional ChromaDB metadata filter, e.g. {"source": "mbe"}.
        rerank_query: If provided, cross-encoder reranks against this text instead of
            queries[0]. Useful for HyDE: embed(hyde_passage) but rerank(question).
    """
    if not queries:
        return []
    if len(queries) == 1:
        return retrieve_documents(queries[0], k=k, exclude_ids=exclude_ids,
                                  vectorstore=vectorstore, use_bm25=use_bm25,
                                  where=where, rerank_query=rerank_query)

    vs = vectorstore or get_vectorstore()
    fetch_k = k * 3

    all_lists = []
    for q in queries:
        all_lists.append(_retrieve_dense(q, k=fetch_k, exclude_ids=exclude_ids, vectorstore=vs, where=where))
        if use_bm25:
            try:
                all_lists.append(_retrieve_bm25(q, k=fetch_k, exclude_ids=exclude_ids, vectorstore=vs))
            except Exception as e:
                logger.warning("BM25 multi-query failed (dense-only for this variant): %s", e)

    candidates = _pool_and_dedup(all_lists)
    return rerank_with_cross_encoder(rerank_query or queries[0], candidates, top_k=k)
