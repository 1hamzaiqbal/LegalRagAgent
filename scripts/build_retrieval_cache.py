#!/usr/bin/env python3
"""Build deterministic retrieval-id caches for top-k replay.

The output can be passed to `eval/eval_harness.py --retrieval-cache-path`.
Each row stores the ordered corpus ids returned by retrieval at a large max-k;
answer-generation runs can then slice top-k without re-running embedding search.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import BEIR_DATASETS, EvalConfig, load_questions  # noqa: E402
from hotpotqa_retrieval import retrieve_hotpotqa_documents  # noqa: E402
from eval_harness import (  # noqa: E402
    _collection_for_config,
    _fmt_intermediate,
    _gold_ids,
    _gold_reference_text,
    _is_gold_retrieved,
    _retrieval_question,
    _retrieval_where_for_row,
    _row_label,
)
from langchain_core.documents import Document  # noqa: E402
from rag_utils import get_vectorstore, rerank_with_cross_encoder, retrieve_documents_multi_query  # noqa: E402


QUERY_TYPE_TO_LABEL_PREFIX = {
    "csqe_cache": "csqe",
    "hyde_cache": "hyde",
    "raw_question": "simple",
    "hyre_cache": "snap_hyre",
    "golden_neighbors": "golden_plus_neighbors",
}

EXPECTED_GENERATION_SOURCE_MODES = {
    "csqe_cache": {"csqe"},
    "hyde_cache": {"rag_hyde"},
    "hyre_cache": {"snap_hyre", "snap_hyre_exemplar", "rag_snap_hyde_2call"},
}

PASSAGE_STYLE_VARIANT_ALIASES = {
    "": "single",
    "default": "single",
    "realpassage": "single",
    "single": "single",
    "one": "single",
    "multi": "multi3",
    "multi3": "multi3",
    "three": "multi3",
    "3": "multi3",
}


def _normalize_passage_style_variant(value: str) -> str:
    raw = str(value or "").strip().lower()
    variant = PASSAGE_STYLE_VARIANT_ALIASES.get(raw)
    if not variant:
        raise SystemExit(
            f"Unsupported passage-style exemplar variant {raw!r}; expected single or multi3"
        )
    return variant


def _generation_entry_passage_style_variant(entry: dict[str, Any], source_mode: str) -> str:
    variant = str(entry.get("passage_style_signal_variant") or "").strip().lower()
    if variant:
        return PASSAGE_STYLE_VARIANT_ALIASES.get(variant, variant)
    if source_mode in {"rag_hyde_exemplar", "snap_hyre_exemplar"}:
        return "single"
    return ""


def _load_hyre_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label")
            if not label:
                raise SystemExit(f"{path}:{line_no}: missing label")
            cache[str(label)] = row
    if not cache:
        raise SystemExit(f"{path}: no HyRE cache rows loaded")
    return cache


def _validate_generation_cache_entry(
    *,
    entry: dict[str, Any],
    path: Path,
    label: str,
    dataset: str,
    query_type: str,
    expected_provider: str = "",
    expected_passage_style_variant: str = "",
) -> None:
    violations: list[str] = []
    source_mode = str(entry.get("source_mode") or entry.get("mode") or "").strip()
    expected_modes = EXPECTED_GENERATION_SOURCE_MODES.get(query_type, set())
    if expected_modes:
        if not source_mode:
            violations.append("missing source_mode/mode")
        elif source_mode not in expected_modes:
            violations.append(f"source_mode={source_mode!r} not in {sorted(expected_modes)!r}")

    entry_dataset = str(entry.get("dataset") or "").strip()
    if entry_dataset and entry_dataset != dataset:
        violations.append(f"dataset={entry_dataset!r} != expected {dataset!r}")

    entry_provider = str(entry.get("provider") or "").strip()
    if expected_provider and entry_provider and entry_provider != expected_provider:
        violations.append(f"provider={entry_provider!r} != expected {expected_provider!r}")

    if source_mode in {"rag_hyde_exemplar", "snap_hyre_exemplar"}:
        expected_variant = _normalize_passage_style_variant(expected_passage_style_variant)
        entry_variant = _generation_entry_passage_style_variant(entry, source_mode)
        if entry_variant != expected_variant:
            violations.append(
                f"passage_style_signal_variant={entry_variant!r} != expected {expected_variant!r}"
            )

    if violations:
        raise SystemExit(
            "generation cache provenance mismatch "
            f"path={path} label={label}: " + "; ".join(violations)
        )


def _load_existing_labels(path: Path) -> set[str]:
    if not path.exists():
        return set()
    labels: set[str] = set()
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON in existing retrieval cache: {exc}") from exc
            label = row.get("label")
            if label:
                labels.add(str(label))
    return labels


def _hash_texts(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(value.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _score_from_doc(doc) -> float:
    try:
        return float(doc.metadata.get("cross_encoder_score", 0.0) or 0.0)
    except Exception:
        return 0.0


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value) != ""))


def _no_silent_fallback_enabled() -> bool:
    return os.getenv("NO_SILENT_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def _optional_retrieval_lock():
    """Serialize Chroma-backed retrieval cache builds on local machines."""
    if os.getenv("RETRIEVAL_CACHE_LOCK", "1").strip().lower() not in {"1", "true", "yes", "on"}:
        yield
        return

    lock_path = Path(os.getenv("RETRIEVAL_CACHE_LOCK_PATH", str(ROOT / ".locks" / "retrieval_cache.lock")))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock_file:
        print(f"[retrieval-lock] waiting path={lock_path}", flush=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        print(f"[retrieval-lock] acquired path={lock_path}", flush=True)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            print(f"[retrieval-lock] released path={lock_path}", flush=True)


def _direct_chroma_collection(collection_name: str):
    import chromadb

    chroma_dir = os.getenv("CHROMA_DB_DIR", str(ROOT / "chroma_db"))
    client = chromadb.PersistentClient(path=chroma_dir)
    return client.get_collection(collection_name)


def _direct_lookup_by_doc_ids(collection, idxs: list[str], include: list[str]) -> dict[str, Any]:
    unique_idxs = _dedupe(idxs)
    if not unique_idxs:
        return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
    return collection.get(
        ids=[f"doc_{idx}" for idx in unique_idxs],
        include=include,
    )


def _docs_from_chroma_result(result: dict[str, Any]) -> list[Document]:
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    return [
        Document(page_content=text or "", metadata=dict(metadata or {}))
        for text, metadata in zip(documents, metadatas)
    ]


def _golden_neighbors_from_stored_embeddings(
    *,
    collection,
    gold_text: str,
    gold_ids: list[str],
    retrieve_k: int,
    where: dict | None = None,
) -> tuple[list[Document], list[str]]:
    """Retrieve around stored gold-document embeddings without loading an embedder.

    HousingQA's 1.8M-document Chroma index is close to the local memory limit.
    For gold-neighbor cache construction, the query text is the gold corpus
    document itself, so using its persisted Chroma embedding preserves the
    retrieval intent while avoiding a second sentence-transformer resident in
    memory.
    """
    unique_gold_ids = _dedupe(gold_ids)
    batch = _direct_lookup_by_doc_ids(
        collection,
        unique_gold_ids,
        include=["embeddings", "metadatas"],
    )
    embeddings = batch.get("embeddings")
    if embeddings is None:
        embeddings = []
    metadatas = batch.get("metadatas") or []
    found_embedding_ids = [
        str((metadata or {}).get("idx") or chroma_id).removeprefix("doc_")
        for chroma_id, metadata in zip(batch.get("ids") or [], metadatas)
    ]
    missing = [idx for idx in unique_gold_ids if idx not in set(found_embedding_ids)]
    if missing and _no_silent_fallback_enabled():
        raise SystemExit(
            "NO_SILENT_FALLBACK blocked stored-gold-embedding retrieval: "
            f"missing embeddings for gold_ids={missing[:10]}"
        )
    if len(embeddings) == 0:
        raise RuntimeError("no stored gold embeddings found")

    fetch_k = retrieve_k * 3
    pooled: list[Document] = []
    seen: set[str] = set()
    for embedding in embeddings:
        result = collection.query(
            query_embeddings=[embedding],
            n_results=fetch_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        result_docs = result.get("documents") or [[]]
        result_metas = result.get("metadatas") or [[]]
        result_distances = result.get("distances") or [[]]
        for text, metadata, distance in zip(result_docs[0], result_metas[0], result_distances[0]):
            metadata = dict(metadata or {})
            idx = str(metadata.get("idx", "") or "")
            if not idx or idx in seen:
                continue
            seen.add(idx)
            metadata["dense_distance"] = float(distance)
            pooled.append(Document(page_content=text or "", metadata=metadata))

    return rerank_with_cross_encoder(gold_text, pooled, top_k=retrieve_k), found_embedding_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=[
        "barexam", "housing", "legal_rag", "legal_rag_bench", "mas_legal_bench", "legal_link_eu", "australian", "casehold",
        "musique", "hotpotqa", "legalbench_scalr", "medqa", *BEIR_DATASETS.keys(),
    ])
    parser.add_argument("--questions", default="full", help="'full' or integer N")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--query-type", required=True, choices=sorted(QUERY_TYPE_TO_LABEL_PREFIX))
    parser.add_argument("--label-prefix", help="Override cache label_prefix")
    parser.add_argument("--hyre-cache-path", type=Path, help="Required for --query-type hyde_cache or hyre_cache")
    parser.add_argument("--expected-provider", default="",
                        help="Optional provider label expected inside generation cache rows")
    parser.add_argument("--passage-style-variant", default="",
                        help="Expected exemplar style variant for exemplar generation caches: single or multi3")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--source-filter", default="")
    parser.add_argument("--housing-state-filter", action="store_true",
                        help="For HousingQA, constrain retrieval to each question's state metadata")
    parser.add_argument("--collection", help="Override dataset collection")
    parser.add_argument("--embedding-model", default="", help="Override EVAL_EMBEDDING_MODEL")
    parser.add_argument("--exclude-gold-ids", default="",
                        help="Comma/whitespace-separated gold ids to exclude from question loading")
    parser.add_argument("--exclude-gold-ids-path", default="",
                        help="JSON/TXT file of gold ids to exclude from question loading")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Append missing labels if the output already exists")
    parser.add_argument("--progress-interval", type=int, default=25,
                        help="Print progress every N newly written rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_k <= 0:
        raise SystemExit("--max-k must be positive")
    with _optional_retrieval_lock():
        _main_locked(args)


def _main_locked(args: argparse.Namespace) -> None:
    if args.embedding_model:
        os.environ["EVAL_EMBEDDING_MODEL"] = args.embedding_model
    if args.query_type in {"csqe_cache", "hyde_cache", "hyre_cache"} and not args.hyre_cache_path:
        raise SystemExit("--hyre-cache-path is required for generated-query cache types")
    if (
        args.dataset == "housing"
        and not args.housing_state_filter
        and not _env_truthy("EVAL_ALLOW_UNFILTERED_HOUSING_RETRIEVAL")
    ):
        raise SystemExit(
            "HousingQA retrieval-cache builds must pass --housing-state-filter. "
            "Set EVAL_ALLOW_UNFILTERED_HOUSING_RETRIEVAL=1 only for an explicit "
            "unfiltered provenance/ablation cache."
        )

    config = EvalConfig(
        mode="rag_simple",
        dataset=args.dataset,
        questions=args.questions,
        seed=args.seed,
        source_filter=args.source_filter,
        sample_start=args.sample_start,
        sample_end=args.sample_end,
        retrieval_k=args.max_k,
        housing_state_filter=args.housing_state_filter,
        exclude_gold_ids=args.exclude_gold_ids,
        exclude_gold_ids_path=args.exclude_gold_ids_path,
    )
    questions = load_questions(config)
    if args.sample_start or args.sample_end is not None:
        start = max(0, int(args.sample_start or 0))
        end = None if args.sample_end is None else max(start, int(args.sample_end))
        questions = questions.iloc[start:end].reset_index(drop=True)

    collection = args.collection or _collection_for_config(config)
    embedding_model = os.getenv("EVAL_EMBEDDING_MODEL", "").strip() or ""
    label_prefix = args.label_prefix or QUERY_TYPE_TO_LABEL_PREFIX[args.query_type]
    hyre_cache = _load_hyre_cache(args.hyre_cache_path) if args.hyre_cache_path else {}
    use_stored_gold_embeddings = (
        args.query_type == "golden_neighbors"
        and _env_truthy("GOLDEN_NEIGHBORS_STORED_EMBEDDING")
    )
    use_hotpotqa_in_row = collection == "hotpotqa_passages"
    vectorstore = (
        None
        if use_stored_gold_embeddings or use_hotpotqa_in_row
        else get_vectorstore(collection, embedding_model=embedding_model or None)
    )
    direct_collection = _direct_chroma_collection(collection) if use_stored_gold_embeddings else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing_labels(args.out) if args.resume else set()
    open_mode = "a" if args.resume else "w"
    wrote = 0
    skipped = 0
    already_done = 0
    start_time = time.time()
    with args.out.open(open_mode) as f:
        for fallback_i, row in questions.iterrows():
            label = _row_label(row, config, fallback_i=fallback_i)
            if label in done:
                already_done += 1
                continue
            row_idx = str(row.get("idx", fallback_i))
            cache_entry: dict[str, Any] | None = None
            if args.query_type == "raw_question":
                queries = [_retrieval_question(row)]
            elif args.query_type == "golden_neighbors":
                gold = _gold_reference_text(row, config)
                if not gold:
                    skipped += 1
                    continue
                queries = [gold]
            else:
                cache_entry = hyre_cache.get(label)
                if not cache_entry or not cache_entry.get("hyde_passage"):
                    skipped += 1
                    continue
                _validate_generation_cache_entry(
                    entry=cache_entry,
                    path=args.hyre_cache_path,
                    label=label,
                    dataset=args.dataset,
                    query_type=args.query_type,
                    expected_provider=args.expected_provider,
                    expected_passage_style_variant=(
                        args.passage_style_variant
                        or os.getenv("EVAL_PASSAGE_STYLE_VARIANT", "")
                        or "single"
                    ),
                )
                queries = [str(cache_entry["hyde_passage"])]

            retrieve_k = args.max_k
            retrieval_backend = "langchain_chroma"
            stored_gold_embedding_ids: list[str] = []
            where = _retrieval_where_for_row(row, config)
            if args.query_type == "golden_neighbors":
                retrieve_k = args.max_k + max(args.max_k, len(_gold_ids(row)))

            if use_hotpotqa_in_row:
                docs = retrieve_hotpotqa_documents(
                    row,
                    queries,
                    k=retrieve_k,
                    embedding_model=embedding_model or None,
                )
                retrieval_backend = "hotpotqa_in_row_gte_ce"
            elif use_stored_gold_embeddings:
                docs, stored_gold_embedding_ids = _golden_neighbors_from_stored_embeddings(
                    collection=direct_collection,
                    gold_text=queries[0],
                    gold_ids=_gold_ids(row),
                    retrieve_k=retrieve_k,
                    where=where,
                )
                retrieval_backend = "stored_gold_embedding"
            else:
                docs = retrieve_documents_multi_query(
                    queries=queries,
                    k=retrieve_k,
                    vectorstore=vectorstore,
                    where=where,
                    rerank_query=None,
                )
            doc_ids = [str(doc.metadata.get("idx", "")) for doc in docs]
            row_source = str(row.get("source", "") or "").strip()
            source_doc = str(row.get("source_doc", "") or "").strip()
            target_doc = str(row.get("target_doc", "") or "").strip()
            same_source_retrieved_ids = [
                str(doc.metadata.get("idx", ""))
                for doc in docs
                if row_source and str(doc.metadata.get("source", "") or "").strip() == row_source
            ][:args.max_k]
            source_doc_retrieved_ids = [
                str(doc.metadata.get("idx", ""))
                for doc in docs
                if source_doc and (
                    str(doc.metadata.get("source", "") or "").strip() == source_doc
                    or str(doc.metadata.get("citation", "") or "").strip() == source_doc
                )
            ][:args.max_k]
            target_doc_retrieved_ids = [
                str(doc.metadata.get("idx", ""))
                for doc in docs
                if target_doc and (
                    str(doc.metadata.get("source", "") or "").strip() == target_doc
                    or str(doc.metadata.get("citation", "") or "").strip() == target_doc
                )
            ][:args.max_k]
            scores = [_score_from_doc(doc) for doc in docs]
            cross_encoder_max_chars = os.getenv("CROSS_ENCODER_MAX_CHARS", "4096").strip()
            cross_encoder_query_truncated = any(
                bool(doc.metadata.get("cross_encoder_query_truncated")) for doc in docs
            )
            cross_encoder_doc_truncated_count = sum(
                1 for doc in docs if doc.metadata.get("cross_encoder_doc_truncated")
            )
            if args.query_type == "golden_neighbors":
                gold_ids = _gold_ids(row)
                neighbor_ids = [idx for idx in doc_ids if idx not in set(gold_ids)]
                retrieved_ids = _dedupe(neighbor_ids)[:args.max_k]
                effective_retrieved_ids = _dedupe(gold_ids + retrieved_ids)[:args.max_k]
                score_by_id = {idx: score for idx, score in zip(doc_ids, scores)}
                scores = [score_by_id.get(idx, 0.0) for idx in retrieved_ids]
            else:
                retrieved_ids = _dedupe(doc_ids)[:args.max_k]
                effective_retrieved_ids = retrieved_ids
                scores = scores[:len(retrieved_ids)]

            record = {
                "label": label,
                "idx": row_idx,
                "dataset": args.dataset,
                "query_type": args.query_type,
                "label_prefix": label_prefix,
                "generation_source_mode": str(cache_entry.get("source_mode") or cache_entry.get("mode") or "") if cache_entry else "",
                "generation_provider": str(cache_entry.get("provider") or "") if cache_entry else "",
                "generation_passage_style_signal_variant": str(cache_entry.get("passage_style_signal_variant") or "") if cache_entry else "",
                "generation_passage_style_signal_ids": cache_entry.get("passage_style_signal_ids", []) if cache_entry else [],
                "generation_cache_path": str(args.hyre_cache_path or ""),
                "collection": collection,
                "embedding_model": embedding_model,
                "where": where or {},
                "housing_state_filter": bool(args.dataset == "housing" and args.housing_state_filter),
                "max_k": args.max_k,
                "retrieved_ids": retrieved_ids,
                "scores": scores,
                "gold_ids": _gold_ids(row),
                "gold_retrieved": _is_gold_retrieved(row, effective_retrieved_ids),
                "effective_retrieved_ids": effective_retrieved_ids,
                "row_source": row_source,
                "same_source_retrieved": bool(same_source_retrieved_ids),
                "same_source_retrieved_ids": same_source_retrieved_ids,
                "source_doc": source_doc,
                "source_doc_retrieved": bool(source_doc_retrieved_ids),
                "source_doc_retrieved_ids": source_doc_retrieved_ids,
                "target_doc": target_doc,
                "target_doc_retrieved": bool(target_doc_retrieved_ids),
                "target_doc_retrieved_ids": target_doc_retrieved_ids,
                "query_hash": _hash_texts(queries),
                "question_hash": _hash_texts([_fmt_intermediate(row, config)]),
                "retrieval_backend": retrieval_backend,
                "cross_encoder_max_chars": cross_encoder_max_chars,
                "cross_encoder_query_truncated": cross_encoder_query_truncated,
                "cross_encoder_doc_truncated_count": cross_encoder_doc_truncated_count,
            }
            if args.query_type == "golden_neighbors":
                record["injected_gold_ids"] = _gold_ids(row)
                record["gold_injected"] = bool(_gold_ids(row))
                if stored_gold_embedding_ids:
                    record["stored_gold_embedding_ids"] = stored_gold_embedding_ids
            f.write(json.dumps(record, sort_keys=True) + "\n")
            f.flush()
            wrote += 1
            if args.progress_interval > 0 and wrote % args.progress_interval == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                rate = wrote / elapsed
                remaining = max(len(questions) - already_done - wrote, 0)
                eta_min = remaining / rate / 60 if rate > 0 else 0
                print(
                    f"[{wrote}] {label:<35} OK "
                    f"({rate:.2f} rows/s, eta={eta_min:.1f}m)",
                    flush=True,
                )

    print(f"wrote {wrote} retrieval-cache rows to {args.out}")
    if already_done:
        print(f"skipped {already_done} existing rows from {args.out}")
    if skipped:
        print(f"skipped {skipped} rows without required query material")
        if _no_silent_fallback_enabled():
            raise SystemExit(
                f"NO_SILENT_FALLBACK blocked retrieval cache with skipped_rows={skipped}"
            )


if __name__ == "__main__":
    main()
