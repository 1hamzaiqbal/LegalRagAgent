#!/usr/bin/env python3
"""Build MuSiQue dense/CE retrieval caches, raw-SCOPE pool, and report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _fmt_intermediate, _gold_ids, _retrieval_question, _row_label  # noqa: E402
from musique_retrieval import paragraph_text  # noqa: E402
from rag_utils import get_cross_encoder, get_embeddings  # noqa: E402


MODEL = "or-gemma4-26b"
DATASET = "musique"
PREFIX = "musique_qfull_seed42"
REPORT = ROOT / "docs/generated/musique_cross_domain_regime_2026-05-28.md"
POINTS = ROOT / "docs/generated/musique_cross_domain_regime_2026-05-28_points.jsonl"
PASSAGES_PATH = ROOT / "datasets/musique/passages.csv"
EMBED_CACHE_DIR = ROOT / "caches/retrieval/tmp"


@dataclass(frozen=True)
class Arm:
    key: str
    display: str
    query_type: str
    retrieval: Path
    generation: Path | None = None


ARMS = {
    "raw": Arm(
        "raw",
        "Raw question",
        "raw_question",
        ROOT / f"caches/retrieval/full/{PREFIX}_raw_question_k10.jsonl",
    ),
    "hyde": Arm(
        "hyde",
        "HyDE",
        "hyde_cache",
        ROOT / f"caches/retrieval/full/{PREFIX}_{MODEL}_rag_hyde_k10.jsonl",
        ROOT / f"caches/generation/full/{PREFIX}_{MODEL}_rag_hyde.jsonl",
    ),
    "scope": Arm(
        "scope",
        "SCOPE / snap_hyre",
        "hyre_cache",
        ROOT / f"caches/retrieval/full/{PREFIX}_{MODEL}_snap_hyre_k10.jsonl",
        ROOT / f"caches/generation/full/{PREFIX}_{MODEL}_snap_hyre.jsonl",
    ),
    "csqe": Arm(
        "csqe",
        "CSQE",
        "csqe_cache",
        ROOT / f"caches/retrieval/full/{PREFIX}_{MODEL}_csqe_k10.jsonl",
        ROOT / f"caches/generation/full/{PREFIX}_{MODEL}_csqe.jsonl",
    ),
    "pool": Arm(
        "pool",
        "raw∪SCOPE pool",
        "raw_scope_pool",
        ROOT / f"caches/retrieval/full/{PREFIX}_{MODEL}_raw_scope_pool_k5.jsonl",
    ),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_by_label(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in read_jsonl(path)}


def load_existing_labels(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("label")) for row in read_jsonl(path) if row.get("label")}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def pct(value: Any) -> str:
    if not finite(value):
        return "--"
    return f"{100.0 * float(value):.1f}%"


def signed_pp(value: Any) -> str:
    if not finite(value):
        return "--"
    return f"{100.0 * float(value):+.1f}pp"


def fmt(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def hash_texts(values: list[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def load_question_rows(limit: int = 0) -> list[tuple[int, str, Any]]:
    config = EvalConfig(dataset=DATASET, questions="full", seed=42)
    rows = list(load_questions(config).reset_index(drop=True).iterrows())
    if limit:
        rows = rows[: min(limit, len(rows))]
    return [(fallback_i, _row_label(row, config, fallback_i=fallback_i), row) for fallback_i, row in rows]


def load_passages() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    df = pd.read_csv(PASSAGES_PATH, keep_default_na=False)
    by_q: dict[str, list[dict[str, Any]]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        item = {
            "q_id": str(row.get("q_id", "")),
            "idx": str(row.get("idx", "")),
            "para_idx": int(row.get("para_idx", 0) or 0),
            "title": str(row.get("title", "")),
            "text": str(row.get("text", "")),
            "is_supporting": str(row.get("is_supporting", "")).strip() in {"1", "true", "True"},
        }
        by_q.setdefault(item["q_id"], []).append(item)
        by_id[item["idx"]] = item
    for rows in by_q.values():
        rows.sort(key=lambda item: item["para_idx"])
    return by_q, by_id


def query_texts_for_arm(arm_key: str, questions: list[tuple[int, str, Any]]) -> dict[str, str]:
    if arm_key == "raw":
        return {label: _retrieval_question(row) for _, label, row in questions}
    arm = ARMS[arm_key]
    if not arm.generation or not arm.generation.exists():
        raise SystemExit(f"Missing generation cache for {arm.display}: {arm.generation}")
    generations = load_by_label(arm.generation)
    out: dict[str, str] = {}
    missing: list[str] = []
    for _, label, _ in questions:
        entry = generations.get(label)
        passage = str((entry or {}).get("hyde_passage") or "").strip()
        if not passage:
            missing.append(label)
        else:
            out[label] = passage
    if missing:
        raise SystemExit(f"{arm.display}: missing generated passage for {len(missing)} labels, first={missing[:5]}")
    return out


def _model_cache_slug(model_name: str | None) -> str:
    raw = model_name or os.getenv("EMBEDDING_MODEL", "") or "Alibaba-NLP/gte-large-en-v1.5"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_") or "default"


def _ids_hash(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def _embed_texts_chunked(
    texts: list[str],
    *,
    embedding_model: str | None,
    chunk_size: int,
    encode_batch_size: int,
) -> np.ndarray:
    embeddings = get_embeddings(embedding_model)
    if encode_batch_size > 0 and hasattr(embeddings, "encode_kwargs"):
        embeddings.encode_kwargs["batch_size"] = encode_batch_size
    chunks: list[np.ndarray] = []
    for start in range(0, len(texts), chunk_size):
        chunk = texts[start:start + chunk_size]
        matrix = np.asarray(embeddings.embed_documents(chunk), dtype=np.float32)
        chunks.append(matrix)
        print(f"[embed] {min(start + len(chunk), len(texts))}/{len(texts)}", flush=True)
    return np.vstack(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def embed_passages(
    by_id: dict[str, dict[str, Any]],
    embedding_model: str | None,
    ids: list[str] | None = None,
    chunk_size: int = 512,
    encode_batch_size: int = 32,
    use_cache: bool = True,
) -> dict[str, np.ndarray]:
    ids = ids or list(by_id)
    texts = [paragraph_text(by_id[idx]) for idx in ids]
    cache_path = (
        EMBED_CACHE_DIR
        / f"musique_passage_embeddings_{_model_cache_slug(embedding_model)}_{_ids_hash(ids)}.npz"
    )
    if use_cache and cache_path.exists():
        print(f"[embed] loading cached passage embeddings {cache_path}", flush=True)
        loaded = np.load(cache_path, allow_pickle=False)
        cached_ids = [str(x) for x in loaded["ids"].tolist()]
        if cached_ids == ids:
            matrix = np.asarray(loaded["embeddings"], dtype=np.float32)
            return {idx: matrix[i] for i, idx in enumerate(ids)}
        print("[embed] cache id order mismatch; rebuilding", flush=True)

    print(f"[embed] passages={len(texts)}", flush=True)
    matrix = _embed_texts_chunked(
        texts,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        encode_batch_size=encode_batch_size,
    )
    if use_cache and len(ids) > 1000:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, ids=np.asarray(ids), embeddings=matrix)
        print(f"[embed] cached passage embeddings {cache_path}", flush=True)
    return {idx: matrix[i] for i, idx in enumerate(ids)}


def ce_scores(pairs: list[tuple[str, str]], *, batch_size: int, chunk_size: int) -> list[float]:
    ce = get_cross_encoder()
    scores: list[float] = []
    for start in range(0, len(pairs), chunk_size):
        chunk = pairs[start:start + chunk_size]
        pred = ce.predict(chunk, batch_size=batch_size)
        scores.extend(float(x) for x in pred)
        print(f"[ce] {min(start + len(chunk), len(pairs))}/{len(pairs)}", flush=True)
    return scores


def generation_meta(arm_key: str, label: str, generation_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if arm_key == "raw":
        return {}
    entry = generation_rows.get(label, {})
    return {
        "generation_source_mode": str(entry.get("source_mode") or entry.get("mode") or ""),
        "generation_provider": str(entry.get("provider") or ""),
        "generation_cache_path": str(ARMS[arm_key].generation or ""),
    }


def build_retrieval_arm(
    *,
    arm_key: str,
    questions: list[tuple[int, str, Any]],
    by_q: dict[str, list[dict[str, Any]]],
    passage_embeddings: dict[str, np.ndarray],
    embedding_model: str | None,
    max_k: int,
    resume: bool,
    ce_batch_size: int,
    ce_chunk_size: int,
) -> None:
    arm = ARMS[arm_key]
    out = arm.retrieval
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_existing_labels(out) if resume else set()
    pending = [(i, label, row) for i, label, row in questions if label not in done]
    if not pending:
        print(f"[retrieval] {arm.display}: already complete at {out}", flush=True)
        return
    queries_by_label = query_texts_for_arm(arm_key, questions)
    generation_rows = load_by_label(arm.generation) if arm.generation and arm.generation.exists() else {}
    query_texts = [queries_by_label[label] for _, label, _ in pending]
    embeddings = get_embeddings(embedding_model)
    print(f"[embed] {arm.display} queries={len(query_texts)}", flush=True)
    query_matrix = np.asarray(embeddings.embed_documents(query_texts), dtype=np.float32)

    config = EvalConfig(dataset=DATASET, questions="full", seed=42)
    max_chars = int(os.getenv("CROSS_ENCODER_MAX_CHARS", "4096") or 4096)
    pair_refs: list[tuple[int, dict[str, Any], float, bool]] = []
    pairs: list[tuple[str, str]] = []
    records_base: dict[int, dict[str, Any]] = {}
    question_hashes: dict[int, str] = {}
    query_hashes: dict[int, str] = {}
    for local_i, (fallback_i, label, row) in enumerate(pending):
        q_id = str(row.get("idx", ""))
        candidates = by_q.get(q_id, [])
        if not candidates:
            raise RuntimeError(f"{label}: no MuSiQue candidate paragraphs for q_id={q_id}")
        query = queries_by_label[label]
        q_emb = query_matrix[local_i]
        dense_scores = [
            float(q_emb @ passage_embeddings[str(candidate["idx"])])
            for candidate in candidates
        ]
        dense_order = np.argsort(-np.asarray(dense_scores))
        query_for_ce = query[:max_chars] if max_chars and len(query) > max_chars else query
        for candidate_i in dense_order:
            candidate = candidates[int(candidate_i)]
            doc_text = paragraph_text(candidate)
            doc_truncated = bool(max_chars and len(doc_text) > max_chars)
            pairs.append((query_for_ce, doc_text[:max_chars] if doc_truncated else doc_text))
            pair_refs.append((fallback_i, candidate, dense_scores[int(candidate_i)], doc_truncated))
        records_base[fallback_i] = {
            "label": label,
            "idx": q_id,
            "dataset": DATASET,
            "query_type": arm.query_type,
            "label_prefix": arm.key,
            "collection": "musique_passages",
            "embedding_model": embedding_model or "",
            "where": {},
            "housing_state_filter": False,
            "max_k": max_k,
            "gold_ids": _gold_ids(row),
            "row_source": "",
            "retrieval_backend": "musique_in_row_gte_ce_batch",
            "cross_encoder_max_chars": str(max_chars),
            "cross_encoder_query_truncated": bool(max_chars and len(query) > max_chars),
        }
        records_base[fallback_i].update(generation_meta(arm_key, label, generation_rows))
        question_hashes[fallback_i] = hash_texts([_fmt_intermediate(row, config)])
        query_hashes[fallback_i] = hash_texts([query])

    all_ce = ce_scores(pairs, batch_size=ce_batch_size, chunk_size=ce_chunk_size)
    per_row: dict[int, list[tuple[dict[str, Any], float, float, bool]]] = {}
    for ref, ce_score in zip(pair_refs, all_ce):
        fallback_i, candidate, dense_score, doc_truncated = ref
        per_row.setdefault(fallback_i, []).append((candidate, float(ce_score), float(dense_score), doc_truncated))

    records: dict[int, dict[str, Any]] = {}
    for fallback_i, values in per_row.items():
        ranked = sorted(values, key=lambda item: item[1], reverse=True)
        top = ranked[:max_k]
        retrieved_ids = [str(candidate["idx"]) for candidate, _, _, _ in top]
        record = dict(records_base[fallback_i])
        record.update({
            "retrieved_ids": retrieved_ids,
            "scores": [score for _, score, _, _ in top],
            "effective_retrieved_ids": retrieved_ids,
            "gold_retrieved": bool(set(record["gold_ids"]) & set(retrieved_ids)),
            "candidate_ids": [str(candidate["idx"]) for candidate, _, _, _ in ranked],
            "candidate_scores": [score for _, score, _, _ in ranked],
            "candidate_dense_scores": [dense for _, _, dense, _ in ranked],
            "cross_encoder_doc_truncated_count": sum(1 for _, _, _, truncated in ranked if truncated),
            "query_hash": query_hashes[fallback_i],
            "question_hash": question_hashes[fallback_i],
        })
        records[fallback_i] = record

    mode = "a" if resume and out.exists() else "w"
    with out.open(mode) as f:
        for fallback_i in sorted(records):
            f.write(json.dumps(records[fallback_i], sort_keys=True) + "\n")
    print(f"[retrieval] {arm.display}: wrote={len(records)} out={out}", flush=True)


def build_pool(
    *,
    questions: list[tuple[int, str, Any]],
    by_id: dict[str, dict[str, Any]],
    resume: bool,
    ce_batch_size: int,
    ce_chunk_size: int,
) -> None:
    out = ARMS["pool"].retrieval
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_existing_labels(out) if resume else set()
    raw = load_by_label(ARMS["raw"].retrieval)
    scope = load_by_label(ARMS["scope"].retrieval)
    pending = [(i, label, row) for i, label, row in questions if label not in done]
    if not pending:
        print(f"[pool] already complete at {out}", flush=True)
        return

    config = EvalConfig(dataset=DATASET, questions="full", seed=42)
    max_chars = int(os.getenv("CROSS_ENCODER_MAX_CHARS", "4096") or 4096)
    pairs: list[tuple[str, str]] = []
    refs: list[tuple[int, str, bool]] = []
    bases: dict[int, dict[str, Any]] = {}
    for fallback_i, label, row in pending:
        raw_row = raw[label]
        scope_row = scope[label]
        component = [
            [str(idx) for idx in raw_row.get("retrieved_ids", [])[:10]],
            [str(idx) for idx in scope_row.get("retrieved_ids", [])[:10]],
        ]
        pool_ids = list(dict.fromkeys(component[0] + component[1]))
        question = _retrieval_question(row)
        ce_query = question[:max_chars] if max_chars and len(question) > max_chars else question
        for idx in pool_ids:
            passage = by_id.get(idx)
            if not passage:
                continue
            text = paragraph_text(passage)
            doc_truncated = bool(max_chars and len(text) > max_chars)
            pairs.append((ce_query, text[:max_chars] if doc_truncated else text))
            refs.append((fallback_i, idx, doc_truncated))
        bases[fallback_i] = {
            "label": label,
            "idx": str(row.get("idx", "")),
            "dataset": DATASET,
            "query_type": "raw_scope_pool",
            "label_prefix": "raw_scope_pool",
            "provider": MODEL,
            "collection": "musique_passages",
            "where": {},
            "housing_state_filter": False,
            "max_k": 5,
            "component_top_k": 10,
            "component_count": 2,
            "component_retrieved_ids": component,
            "pool_size": len(pool_ids),
            "gold_ids": _gold_ids(row),
            "question_hash": hash_texts([_fmt_intermediate(row, config)]),
            "retrieval_backend": "musique_raw_scope_pool_ce_batch",
            "cross_encoder_max_chars": str(max_chars),
            "cross_encoder_query_truncated": bool(max_chars and len(question) > max_chars),
        }

    scores = ce_scores(pairs, batch_size=ce_batch_size, chunk_size=ce_chunk_size)
    per_row: dict[int, list[tuple[str, float, bool]]] = {}
    for (fallback_i, idx, doc_truncated), score in zip(refs, scores):
        per_row.setdefault(fallback_i, []).append((idx, float(score), doc_truncated))

    records: dict[int, dict[str, Any]] = {}
    for fallback_i, vals in per_row.items():
        ranked = sorted(vals, key=lambda item: item[1], reverse=True)
        top = ranked[:5]
        retrieved_ids = [idx for idx, _, _ in top]
        record = dict(bases[fallback_i])
        record.update({
            "retrieved_ids": retrieved_ids,
            "effective_retrieved_ids": retrieved_ids,
            "scores": [score for _, score, _ in top],
            "candidate_ids": [idx for idx, _, _ in ranked],
            "candidate_scores": [score for _, score, _ in ranked],
            "gold_retrieved": bool(set(record["gold_ids"]) & set(retrieved_ids)),
            "cross_encoder_doc_truncated_count": sum(1 for _, _, truncated in ranked if truncated),
        })
        records[fallback_i] = record

    mode = "a" if resume and out.exists() else "w"
    with out.open(mode) as f:
        for fallback_i in sorted(records):
            f.write(json.dumps(records[fallback_i], sort_keys=True) + "\n")
    print(f"[pool] wrote={len(records)} out={out}", flush=True)


def score_map(row: dict[str, Any], all_candidates: bool = False) -> dict[str, float]:
    ids_key = "candidate_ids" if all_candidates and row.get("candidate_ids") else "retrieved_ids"
    scores_key = "candidate_scores" if all_candidates and row.get("candidate_scores") else "scores"
    return {
        str(idx): float(score)
        for idx, score in zip(row.get(ids_key, []) or [], row.get(scores_key, []) or [])
        if finite(score)
    }


def max_gold_score(row: dict[str, Any]) -> float:
    scores = score_map(row, all_candidates=True)
    vals = [scores.get(str(idx), float("nan")) for idx in row.get("gold_ids", []) or []]
    vals = [v for v in vals if finite(v)]
    return max(vals) if vals else float("nan")


def bridge_id(raw_row: dict[str, Any]) -> str:
    scores = score_map(raw_row, all_candidates=True)
    gold = [str(idx) for idx in raw_row.get("gold_ids", []) or []]
    if not gold:
        return ""
    return min(gold, key=lambda idx: scores.get(idx, -1e9))


def metrics(row: dict[str, Any], bridge: str) -> dict[str, int]:
    ids = [str(idx) for idx in row.get("retrieved_ids", []) or []]
    gold = {str(idx) for idx in row.get("gold_ids", []) or []}
    return {
        "hit@5": int(bool(gold & set(ids[:5]))),
        "full@2": int(bool(gold) and gold.issubset(set(ids[:2]))),
        "full@5": int(bool(gold) and gold.issubset(set(ids[:5]))),
        "bridge@2": int(bool(bridge) and bridge in set(ids[:2])),
        "bridge@5": int(bool(bridge) and bridge in set(ids[:5])),
    }


def summarize_cache(path: Path, mode: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    labels = [str(row.get("label", "")) for row in rows]
    return {
        "rows": len(rows),
        "duplicates": len(labels) - len(set(labels)),
        "errors": sum(1 for row in rows if row.get("error")),
        "missing_passage": sum(1 for row in rows if mode != "retrieval" and not row.get("hyde_passage")),
        "parse_bad": sum(1 for row in rows if row.get("snap_hyre_parse_ok") is False),
        "answer_artifact": sum(1 for row in rows if row.get("hyde_contains_answer_artifact") is True),
        "short_retrieval": sum(1 for row in rows if mode == "retrieval" and len(row.get("retrieved_ids", []) or []) < int(row.get("max_k", 10))),
    }


def ri(raw_vals: list[int], arm_vals: list[int]) -> tuple[int, int, float]:
    helps = sum(1 for raw, arm in zip(raw_vals, arm_vals) if arm > raw)
    hurts = sum(1 for raw, arm in zip(raw_vals, arm_vals) if arm < raw)
    n = len(raw_vals)
    return helps, hurts, (helps - hurts) / n if n else float("nan")


def report() -> None:
    caches = {key: load_by_label(arm.retrieval) for key, arm in ARMS.items()}
    common = sorted(set.intersection(*(set(rows) for rows in caches.values())))
    metric_order = ["hit@5", "full@2", "full@5", "bridge@2", "bridge@5"]
    all_metrics: dict[str, dict[str, list[int]]] = {key: {m: [] for m in metric_order} for key in ARMS}
    points: list[dict[str, Any]] = []
    for label in common:
        raw_row = caches["raw"][label]
        bridge = bridge_id(raw_row)
        raw_m = metrics(raw_row, bridge)
        raw_gold = max_gold_score(raw_row)
        for key, rows in caches.items():
            row_m = metrics(rows[label], bridge)
            for metric_name, value in row_m.items():
                all_metrics[key][metric_name].append(value)
            if key != "raw":
                points.append({
                    "label": label,
                    "arm": key,
                    "bridge_id": bridge,
                    "raw_hit5": raw_m["hit@5"],
                    "arm_hit5": row_m["hit@5"],
                    "raw_full5": raw_m["full@5"],
                    "arm_full5": row_m["full@5"],
                    "raw_bridge5": raw_m["bridge@5"],
                    "arm_bridge5": row_m["bridge@5"],
                    "gold_affinity_delta": max_gold_score(rows[label]) - raw_gold,
                })

    POINTS.parent.mkdir(parents=True, exist_ok=True)
    with POINTS.open("w") as f:
        for point in points:
            f.write(json.dumps(point, sort_keys=True) + "\n")

    raw_hit = mean(all_metrics["raw"]["hit@5"])
    scope_hit = mean(all_metrics["scope"]["hit@5"])
    pool_hit = mean(all_metrics["pool"]["hit@5"])
    hyde_full_delta = mean(all_metrics["hyde"]["full@5"]) - mean(all_metrics["raw"]["full@5"])
    scope_full_delta = mean(all_metrics["scope"]["full@5"]) - mean(all_metrics["raw"]["full@5"])
    csqe_delta = mean(all_metrics["csqe"]["hit@5"]) - raw_hit
    help_supported = (hyde_full_delta > 0) or (scope_full_delta > 0)
    csqe_supported = csqe_delta < 0.05
    if raw_hit <= 0.25:
        regime_supported = pool_hit <= scope_hit
        regime_text = "weak raw regime; pool was expected not to beat SCOPE"
    elif raw_hit >= 0.30:
        regime_supported = pool_hit >= scope_hit
        regime_text = "moderate raw regime; pool was expected to help preserve raw candidates"
    else:
        regime_supported = False
        regime_text = "boundary raw regime; pre-registered threshold does not give a clean prediction"

    lines: list[str] = [
        "# MuSiQue Cross-Domain Regime Test - 2026-05-28",
        "",
        "This is a retrieval-only MuSiQue validation run over the per-question candidate paragraphs. Retrieval uses `Alibaba-NLP/gte-large-en-v1.5` dense scoring inside each question's candidate set followed by `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking. No answer calls and no `paper/` edits were made.",
        "",
        "## Verdict",
        "",
        f"- **H-cross-domain-help-side: {'supported' if help_supported else 'killed'}.** HyDE full-support@5 delta is {signed_pp(hyde_full_delta)} and SCOPE full-support@5 delta is {signed_pp(scope_full_delta)} versus raw.",
        f"- **H-csqe-collapse-cross-domain: {'supported' if csqe_supported else 'killed'}.** CSQE Hit@5 is {pct(mean(all_metrics['csqe']['hit@5']))} versus raw {pct(raw_hit)}.",
        f"- **H-regime-placement: {'supported' if regime_supported else 'mixed/killed'}.** MuSiQue raw Hit@5 is {pct(raw_hit)}, placing it in the {regime_text}; pool Hit@5 is {pct(pool_hit)} and SCOPE Hit@5 is {pct(scope_hit)}.",
        "- Optional q500 answer EM was not run in this phase; the task gate was retrieval-regime evidence first.",
        "",
        "## Source Files",
        "",
        "| Role | Path |",
        "|---|---|",
        "| Dataset questions | `datasets/musique/questions.csv` |",
        "| Per-question paragraphs | `datasets/musique/passages.csv` |",
    ]
    for key in ["raw", "hyde", "scope", "csqe", "pool"]:
        arm = ARMS[key]
        if arm.generation:
            lines.append(f"| {arm.display} generation | `{arm.generation.relative_to(ROOT)}` |")
        lines.append(f"| {arm.display} retrieval | `{arm.retrieval.relative_to(ROOT)}` |")
    lines.extend([
        f"| Row-level points | `{POINTS.relative_to(ROOT)}` |",
        "",
        "## Cache Health",
        "",
        "| Cache | Rows | Duplicates | Errors | Missing passage | Parse bad | Answer artifact | Short retrieval |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for key in ["raw", "hyde", "scope", "csqe", "pool"]:
        arm = ARMS[key]
        if arm.generation:
            h = summarize_cache(arm.generation, "generation")
            lines.append(f"| {arm.display} generation | {h['rows']} | {h['duplicates']} | {h['errors']} | {h['missing_passage']} | {h['parse_bad']} | {h['answer_artifact']} | -- |")
        h = summarize_cache(arm.retrieval, "retrieval")
        lines.append(f"| {arm.display} retrieval | {h['rows']} | {h['duplicates']} | {h['errors']} | -- | -- | -- | {h['short_retrieval']} |")

    lines.extend([
        "",
        "## Retrieval Metrics",
        "",
        "Bridge paragraph = the gold paragraph with the lowest raw-query CE score within the question's gold support set. Full-support requires every gold paragraph for the question to be present in the top-k.",
        "",
        "| Method | Hit@5 | Full-support@2 | Full-support@5 | Bridge@2 | Bridge@5 | Mean gold-affinity delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key in ["raw", "hyde", "scope", "csqe", "pool"]:
        delta = 0.0 if key == "raw" else mean([p["gold_affinity_delta"] for p in points if p["arm"] == key])
        lines.append(
            f"| {ARMS[key].display} | "
            + " | ".join(pct(mean(all_metrics[key][m])) for m in metric_order)
            + f" | {fmt(delta)} |"
        )

    lines.extend([
        "",
        "## Expansion vs Raw",
        "",
        "| Method | Metric | Delta | Help rows | Hurt rows | RI |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for key in ["hyde", "scope", "csqe", "pool"]:
        for metric_name in metric_order:
            helps, hurts, ri_value = ri(all_metrics["raw"][metric_name], all_metrics[key][metric_name])
            delta = mean(all_metrics[key][metric_name]) - mean(all_metrics["raw"][metric_name])
            lines.append(f"| {ARMS[key].display} | {metric_name} | {signed_pp(delta)} | {helps} | {hurts} | {fmt(ri_value)} |")

    lines.extend([
        "",
        "## Regime Gradient Context",
        "",
        "| Dataset/regime | Raw Hit@5 | SCOPE Hit@5 | raw-SCOPE pool Hit@5 | Note |",
        "|---|---:|---:|---:|---|",
        "| BarExamQA | 1.4% | 12.0% | 3.9% | extreme weak legal query; pool fails versus SCOPE |",
        "| CaseHOLD | 17.9% | 45.0% | 19.2% | intermediate weak legal query; pool fails versus SCOPE |",
        f"| MuSiQue | {pct(raw_hit)} | {pct(scope_hit)} | {pct(pool_hit)} | current cross-domain weak-query test |",
        "| HousingQA state-filtered | 36.8% | 38.0% | 41.1% | stronger raw state anchors; pool helps |",
        "| BEIR pooled | 62.2% | 49.8% | 65.9% | strong raw queries; pool preserves raw candidates |",
        "",
        "## Recommendation",
        "",
    ])
    if help_supported:
        lines.append("- Keep MuSiQue as the open-domain multi-hop retrieval anchor; it provides a cross-domain check on whether generative query expansion helps full-support/bridge retrieval.")
    else:
        lines.append("- Do not spend q500 answer budget on MuSiQue under this retrieval setup unless a later prompt or selector improves full-support/bridge retrieval.")
    lines.append("- Use the row-level points file to inspect where expansion helps bridge recall versus where it drops a required support paragraph.")
    lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines).rstrip() + "\n")
    print(REPORT.relative_to(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ret = sub.add_parser("retrieval")
    ret.add_argument("--arm", choices=["raw", "hyde", "scope", "csqe", "all"], default="all")
    ret.add_argument("--limit", type=int, default=0)
    ret.add_argument("--max-k", type=int, default=10)
    ret.add_argument("--embedding-model", default=os.getenv("EVAL_EMBEDDING_MODEL", "").strip())
    ret.add_argument("--resume", action="store_true")
    ret.add_argument("--embed-chunk-size", type=int, default=512)
    ret.add_argument("--embed-batch-size", type=int, default=32)
    ret.add_argument("--ce-batch-size", type=int, default=int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "32") or 32))
    ret.add_argument("--ce-chunk-size", type=int, default=4096)

    pool = sub.add_parser("pool")
    pool.add_argument("--limit", type=int, default=0)
    pool.add_argument("--resume", action="store_true")
    pool.add_argument("--ce-batch-size", type=int, default=int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "32") or 32))
    pool.add_argument("--ce-chunk-size", type=int, default=4096)

    sub.add_parser("report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "report":
        report()
        return

    questions = load_question_rows(limit=args.limit)
    by_q, by_id = load_passages()
    if args.command == "pool":
        build_pool(
            questions=questions,
            by_id=by_id,
            resume=args.resume,
            ce_batch_size=args.ce_batch_size,
            ce_chunk_size=args.ce_chunk_size,
        )
        return

    start = time.time()
    needed_ids = sorted({
        str(candidate["idx"])
        for _, _, row in questions
        for candidate in by_q.get(str(row.get("idx", "")), [])
    })
    passage_embeddings = embed_passages(
        by_id,
        args.embedding_model or None,
        ids=needed_ids,
        chunk_size=args.embed_chunk_size,
        encode_batch_size=args.embed_batch_size,
    )
    selected = ["raw", "hyde", "scope", "csqe"] if args.arm == "all" else [args.arm]
    for arm_key in selected:
        build_retrieval_arm(
            arm_key=arm_key,
            questions=questions,
            by_q=by_q,
            passage_embeddings=passage_embeddings,
            embedding_model=args.embedding_model or None,
            max_k=args.max_k,
            resume=args.resume,
            ce_batch_size=args.ce_batch_size,
            ce_chunk_size=args.ce_chunk_size,
        )
    print(f"[done] elapsed={time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
