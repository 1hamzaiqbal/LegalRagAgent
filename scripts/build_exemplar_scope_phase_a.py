#!/usr/bin/env python3
"""Build Phase-A artifacts for exemplar-grounded SCOPE on BEIR.

Actions:
- select-exemplars: choose three non-gold Chroma medoids per BEIR corpus.
- build-csqe: build deterministic corpus-steered query-expansion caches from
  raw top-k retrieval snippets.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.cluster import MiniBatchKMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval_config import BEIR_DATASETS, EvalConfig, load_questions  # noqa: E402
from eval_harness import _fmt_intermediate, _row_label  # noqa: E402
from rag_utils import get_documents_by_idx, get_embeddings  # noqa: E402


DATASETS = ["beir_scifact", "beir_nfcorpus", "beir_fiqa", "beir_trec_covid", "beir_scidocs"]
MODEL = "or-gemma4-26b"
TOKEN_RE = re.compile(r"[a-z0-9]+")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    subset: str
    collection: str


def specs() -> list[DatasetSpec]:
    return [DatasetSpec(key=key, subset=BEIR_DATASETS[key], collection=key) for key in DATASETS]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _sanitize_text(text: str, limit: int = 850) -> str:
    text = re.sub(r"(?im)^\s*(?:answer|label|score|query_id|corpus_id)\s*:.*$", " ", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].strip()
    return cut + "."


def _doc_text(title: str, text: str) -> str:
    title = str(title or "").strip()
    text = str(text or "").strip()
    if title and title not in text[: max(80, len(title) + 10)]:
        return f"{title}. {text}"
    return text or title


def load_qrels_gold_ids(subset: str) -> set[str]:
    path = REPO_ROOT / "datasets" / "beir" / subset / "qrels_test.csv"
    gold: set[str] = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("corpus_id", "")).strip()
            if cid:
                gold.add(cid)
    return gold


def _json_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in re.split(r"[\s,]+", text) if part.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()] if str(parsed).strip() else []


def eval_exclusion_counts(spec: DatasetSpec, excluded_gold_ids: set[str]) -> tuple[int, int]:
    path = REPO_ROOT / "datasets" / "beir" / spec.subset / "questions.csv"
    total = 0
    removed = 0
    if not excluded_gold_ids:
        with path.open(newline="") as f:
            return sum(1 for _ in csv.DictReader(f)), 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            gold_ids = set(_json_id_list(row.get("gold_idx", "")))
            if gold_ids & excluded_gold_ids:
                removed += 1
    return total, removed


def iter_chroma_batches(collection_name: str, batch_size: int) -> Iterable[dict[str, Any]]:
    import chromadb

    client = chromadb.PersistentClient(path=str(REPO_ROOT / "chroma_db"))
    collection = client.get_collection(collection_name)
    total = collection.count()
    for offset in range(0, total, batch_size):
        yield collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "metadatas", "documents"],
        )


def _non_gold_arrays(batch: dict[str, Any], gold: set[str]) -> tuple[list[str], np.ndarray, list[dict[str, Any]], list[str]]:
    ids: list[str] = []
    metas: list[dict[str, Any]] = []
    docs: list[str] = []
    embs: list[Any] = []
    embeddings = batch.get("embeddings")
    if embeddings is None:
        embeddings = []
    for chroma_id, emb, meta, doc in zip(
        batch.get("ids") or [],
        embeddings,
        batch.get("metadatas") or [],
        batch.get("documents") or [],
    ):
        meta = dict(meta or {})
        idx = str(meta.get("idx") or str(chroma_id).removeprefix("doc_"))
        if not idx or idx in gold:
            continue
        ids.append(idx)
        metas.append(meta)
        docs.append(str(doc or ""))
        embs.append(emb)
    if not embs:
        return [], np.zeros((0, 1), dtype=np.float32), [], []
    return ids, np.asarray(embs, dtype=np.float32), metas, docs


def iter_corpus_text_batches(
    spec: DatasetSpec,
    gold: set[str],
    *,
    batch_size: int,
) -> Iterable[tuple[list[str], list[dict[str, Any]], list[str]]]:
    """Yield non-gold corpus CSV rows for fallback exemplar selection."""
    path = REPO_ROOT / "datasets" / "beir" / spec.subset / "corpus.csv"
    ids: list[str] = []
    metas: list[dict[str, Any]] = []
    docs: list[str] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            idx = str(row.get("idx") or "").strip()
            if not idx or idx in gold:
                continue
            text = _doc_text(str(row.get("title") or ""), str(row.get("text") or ""))
            if len(tokenize(text)) < 12:
                continue
            ids.append(idx)
            metas.append({
                "idx": idx,
                "title": str(row.get("title") or "").strip(),
                "source": str(row.get("source") or f"BEIR/{spec.subset}").strip(),
            })
            docs.append(text)
            if len(ids) >= batch_size:
                yield ids, metas, docs
                ids, metas, docs = [], [], []
    if ids:
        yield ids, metas, docs


def _csv_embedding_arrays(
    spec: DatasetSpec,
    gold: set[str],
    *,
    batch_size: int = 256,
) -> tuple[list[str], np.ndarray, list[dict[str, Any]], list[str]]:
    """Embed non-gold CSV rows for datasets whose indexed Chroma slice has no safe medoids."""
    embedder = get_embeddings()
    all_ids: list[str] = []
    all_metas: list[dict[str, Any]] = []
    all_docs: list[str] = []
    arrays: list[np.ndarray] = []
    seen = 0
    for ids, metas, docs in iter_corpus_text_batches(spec, gold, batch_size=batch_size):
        seen += len(ids)
        print(f"[exemplars:fallback] {spec.key}: embedded {seen}", flush=True)
        arrays.append(np.asarray(embedder.embed_documents(docs), dtype=np.float32))
        all_ids.extend(ids)
        all_metas.extend(metas)
        all_docs.extend(docs)
    if not arrays:
        return [], np.zeros((0, 1), dtype=np.float32), [], []
    return all_ids, np.vstack(arrays), all_metas, all_docs


def _select_medoids(
    *,
    kmeans: MiniBatchKMeans,
    batches: Iterable[tuple[list[str], np.ndarray, list[dict[str, Any]], list[str]]],
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[int]]:
    best: dict[int, tuple[float, str, np.ndarray, dict[str, Any], str]] = {}
    cluster_counts = [0, 0, 0]
    for ids, emb, metas, docs in batches:
        if not ids:
            continue
        labels = kmeans.predict(emb)
        for row_i, cluster_i in enumerate(labels):
            cluster_counts[int(cluster_i)] += 1
            vec = emb[row_i]
            dist = float(np.sum((vec - kmeans.cluster_centers_[int(cluster_i)]) ** 2))
            if int(cluster_i) not in best or dist < best[int(cluster_i)][0]:
                best[int(cluster_i)] = (dist, ids[row_i], vec.copy(), metas[row_i], docs[row_i])

    entries: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for cluster_i in sorted(best):
        dist, idx, vec, meta, doc = best[cluster_i]
        vectors.append(vec)
        title = str(meta.get("title") or "").strip()
        excerpt = _sanitize_text(_doc_text(title, doc))
        entries.append({
            "cluster": cluster_i,
            "idx": idx,
            "title": title,
            "source": str(meta.get("source") or ""),
            "distance_to_centroid": dist,
            "excerpt": excerpt,
            "token_count": len(tokenize(excerpt)),
        })
    return entries, vectors, cluster_counts


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def select_exemplars(out_json: Path, out_md: Path, *, batch_size: int = 4096) -> None:
    payload: dict[str, Any] = {"schema": "exemplar_scope_phase_a_v1", "datasets": {}}
    lines = ["# BEIR Orthogonal Exemplar Selection", ""]
    lines.append("| Dataset | Exemplar source | Chroma docs | Gold ids excluded | Candidates | Eval rows excluded | Exemplar ids | Mutual cosine range |")
    lines.append("|---|---|---:|---:|---:|---:|---|---:|")

    for spec in specs():
        print(f"[exemplars] {spec.key}", flush=True)
        gold = load_qrels_gold_ids(spec.subset)
        kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=2048, n_init=3)
        total_docs = 0
        candidates = 0
        cluster_fit = False
        for batch in iter_chroma_batches(spec.collection, batch_size):
            total_docs += len(batch.get("ids") or [])
            _, emb, _, _ = _non_gold_arrays(batch, gold)
            if emb.shape[0] < 3 and not cluster_fit:
                continue
            if emb.shape[0]:
                kmeans.partial_fit(emb)
                cluster_fit = True
                candidates += emb.shape[0]
        if not cluster_fit:
            exemplar_source = "corpus_csv_embedded_fallback"
            ids_all, emb_all, metas_all, docs_all = _csv_embedding_arrays(spec, gold)
            candidates = int(emb_all.shape[0])
            if candidates < 3:
                exemplar_source = "chroma_with_eval_row_exclusion"
                kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=2048, n_init=3)
                candidates = 0
                cluster_fit = False
                for batch in iter_chroma_batches(spec.collection, batch_size):
                    ids, emb, _, _ = _non_gold_arrays(batch, set())
                    if emb.shape[0] < 3 and not cluster_fit:
                        continue
                    if emb.shape[0]:
                        kmeans.partial_fit(emb)
                        cluster_fit = True
                        candidates += emb.shape[0]
                if not cluster_fit:
                    raise RuntimeError(f"{spec.key}: not enough Chroma embeddings for fallback clustering")
                entries, vectors, cluster_counts = _select_medoids(
                    kmeans=kmeans,
                    batches=(
                        _non_gold_arrays(batch, set())
                        for batch in iter_chroma_batches(spec.collection, batch_size)
                    ),
                )
            else:
                kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, batch_size=2048, n_init=3)
                kmeans.fit(emb_all)
                entries, vectors, cluster_counts = _select_medoids(
                    kmeans=kmeans,
                    batches=[(ids_all, emb_all, metas_all, docs_all)],
                )
        else:
            exemplar_source = "chroma"
            entries, vectors, cluster_counts = _select_medoids(
                kmeans=kmeans,
                batches=(
                    _non_gold_arrays(batch, gold)
                    for batch in iter_chroma_batches(spec.collection, batch_size)
                ),
            )
        if len(entries) != 3:
            raise RuntimeError(f"{spec.key}: expected 3 medoids, got {len(entries)}")

        mutual = [
            _cosine(vectors[i], vectors[j])
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        ]
        ids = [entry["idx"] for entry in entries]
        eval_exclude_gold_ids = ids if exemplar_source == "chroma_with_eval_row_exclusion" else []
        eval_total, eval_removed = eval_exclusion_counts(spec, set(eval_exclude_gold_ids))
        signal_lines = [
            f"A useful retrieval passage for BEIR/{spec.subset} should match the corpus style below: concrete entities, domain terminology, and source-like prose. Use these excerpts only as style and topical-coverage signals; do not copy them and do not treat them as evidence for the current query.",
            "",
        ]
        for i, entry in enumerate(entries, 1):
            signal_lines.append(f"Corpus passage excerpt {i} (doc id {entry['idx']}): {entry['excerpt']}")
        payload["datasets"][spec.key] = {
            "subset": spec.subset,
            "collection": spec.collection,
            "variant": "multi3",
            "ids": ids,
            "exemplars": entries,
            "signal": "\n\n".join(signal_lines),
            "embedding_source": exemplar_source,
            "chroma_docs": total_docs,
            "qrels_gold_ids_excluded": len(gold),
            "candidate_embeddings": candidates,
            "cluster_counts": cluster_counts,
            "mutual_cosine": mutual,
            "eval_exclude_gold_ids": eval_exclude_gold_ids,
            "eval_rows_total": eval_total,
            "eval_rows_excluded": eval_removed,
            "eval_rows_kept": eval_total - eval_removed,
        }
        lines.append(
            f"| {spec.key} | {exemplar_source} | {total_docs} | {len(gold)} | {candidates} | {eval_removed} | "
            f"`{', '.join(ids)}` | {min(mutual):.3f}..{max(mutual):.3f} |"
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    out_md.write_text("\n".join(lines).rstrip() + "\n")
    print(out_json)
    print(out_md)


def _sentence_snippets(text: str, max_sentences: int = 1) -> list[str]:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = [part.strip() for part in SENTENCE_RE.split(text) if len(tokenize(part)) >= 8]
    if not parts and len(tokenize(text)) >= 8:
        parts = [text]
    return [_sanitize_text(part, limit=320) for part in parts[:max_sentences]]


def _build_doc_lookup(collection: str, ids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    unique = list(dict.fromkeys(str(idx) for idx in ids if str(idx)))
    for start in range(0, len(unique), 5000):
        chunk = unique[start : start + 5000]
        docs = get_documents_by_idx(collection, chunk)
        for doc in docs:
            idx = str(doc.metadata.get("idx") or "")
            if idx:
                out[idx] = _doc_text(str(doc.metadata.get("title") or ""), doc.page_content)
    return out


def build_csqe(out_dir: Path, *, raw_top_k: int = 5, snippets_per_query: int = 3) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs():
        print(f"[csqe] {spec.key}", flush=True)
        raw_path = REPO_ROOT / f"caches/retrieval/full/{spec.key}_qfull_seed42_raw_question_k10.jsonl"
        raw_rows = read_jsonl(raw_path)
        all_doc_ids = [
            str(idx)
            for row in raw_rows
            for idx in (row.get("retrieved_ids") or [])[:raw_top_k]
        ]
        docs = _build_doc_lookup(spec.collection, all_doc_ids)
        config = EvalConfig(dataset=spec.key, questions="full", seed=42)
        q_by_label = {
            _row_label(row, config): (row, _fmt_intermediate(row, config))
            for _, row in load_questions(config).iterrows()
        }
        out_path = out_dir / f"{spec.key}_qfull_seed42_csqe.jsonl"
        with out_path.open("w") as f:
            for raw in raw_rows:
                label = str(raw["label"])
                row, question = q_by_label[label]
                snippets: list[str] = []
                source_ids: list[str] = []
                for idx in (raw.get("retrieved_ids") or [])[:raw_top_k]:
                    idx = str(idx)
                    for sent in _sentence_snippets(docs.get(idx, ""), max_sentences=1):
                        if sent and sent not in snippets:
                            snippets.append(sent)
                            source_ids.append(idx)
                    if len(snippets) >= snippets_per_query:
                        break
                steering = " ".join(snippets[:snippets_per_query]).strip()
                passage = (
                    f"{question}\n\nCorpus steering snippets: {steering}"
                    if steering
                    else question
                )
                record = {
                    "label": label,
                    "idx": str(row.get("idx", raw.get("idx", ""))),
                    "dataset": spec.key,
                    "mode": "csqe",
                    "source_mode": "csqe",
                    "provider": "deterministic",
                    "hyde_passage": passage,
                    "hyde_passage_raw": passage,
                    "hyde_contains_answer_artifact": False,
                    "hyde_used_fallback": not bool(steering),
                    "csqe_raw_top_k": raw_top_k,
                    "csqe_source_ids": source_ids,
                    "csqe_snippet_count": len(snippets[:snippets_per_query]),
                }
                f.write(json.dumps(record, sort_keys=True) + "\n")
        print(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select-exemplars", "build-csqe"])
    parser.add_argument("--exemplar-json", type=Path, default=REPO_ROOT / "caches/exemplars/beir_orthogonal3_exemplars_2026-05-26.json")
    parser.add_argument("--exemplar-report", type=Path, default=REPO_ROOT / "docs/generated/beir_orthogonal3_exemplars_2026-05-26.md")
    parser.add_argument("--csqe-dir", type=Path, default=REPO_ROOT / "caches/generation/full")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--raw-top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "select-exemplars":
        select_exemplars(args.exemplar_json, args.exemplar_report, batch_size=args.batch_size)
    elif args.action == "build-csqe":
        build_csqe(args.csqe_dir, raw_top_k=args.raw_top_k)


if __name__ == "__main__":
    main()
