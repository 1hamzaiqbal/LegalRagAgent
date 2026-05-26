#!/usr/bin/env python3
"""On-cache affinity-margin tests for SCOPE/Snap-HyRE.

This is a read-only results-lane analysis. It joins existing raw-question and
canonical snap_hyre retrieval caches with gold ids and the canonical Snap-HyRE
generation caches. For each query representation x, the gold margin is:

    M(x) = affinity(x, best gold) - max affinity(x, retrieved non-gold)

The CE distractor term comes from the retrieval cache's stored cross-encoder
scores. The cosine margin uses the stored Chroma document embeddings and fresh
query embeddings from the configured gte embedding model.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from analyze_perplexity_axis import (  # noqa: E402
    DATASETS,
    MODELS,
    MODEL_LABELS,
    build_or_load_lm,
    fmt_float,
    hit_at_5,
    load_by_label,
    mean,
    pct,
    question_scores,
    source_paths_for,
    spearman,
)
from analyze_scope_gap_mechanism import (  # noqa: E402
    HYRE_CACHE,
    load_questions_raw_text,
    score_ce,
)
from rag_utils import get_embeddings  # noqa: E402


DEFAULT_GOLD_SCORE_CACHE = Path("/tmp/scope_gap_mechanism_2026-05-25_points.jsonl")
DEFAULT_POINTS_OUT = Path("/tmp/affinity_margin_oncache_2026-05-26_points.jsonl")
DEFAULT_LM_CACHE_DIR = Path("/tmp/perplexity_axis_lm_cache_2026-05-25")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if finite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def pearson(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return float("nan")
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    if float(np.std(x_arr)) == 0.0 or float(np.std(y_arr)) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def kendall(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 2:
        return float("nan")
    val = kendalltau([x for x, _ in pairs], [y for _, y in pairs], nan_policy="omit").correlation
    return float(val) if val is not None and finite(val) else float("nan")


def corr_pack(points: list[dict[str, Any]], axis: str, y_axis: str = "retrieval_delta") -> dict[str, float]:
    xs = [float(p[axis]) for p in points if finite(p.get(axis)) and finite(p.get(y_axis))]
    ys = [float(p[y_axis]) for p in points if finite(p.get(axis)) and finite(p.get(y_axis))]
    return {
        "n": len(xs),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys) if len(xs) >= 2 else float("nan"),
        "kendall": kendall(xs, ys),
    }


def standardize_matrix(rows: list[dict[str, Any]], features: list[str]) -> tuple[np.ndarray, list[str]]:
    kept: list[dict[str, Any]] = []
    for row in rows:
        if all(finite(row.get(feature)) for feature in features):
            kept.append(row)
    if not kept:
        return np.zeros((0, len(features)), dtype=np.float64), []
    x = np.asarray([[float(row[feature]) for feature in features] for row in kept], dtype=np.float64)
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x == 0.0] = 1.0
    return (x - mean_x) / std_x, [str(row["row_id"]) for row in kept]


def standardize_vector(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    sd = arr.std()
    if sd == 0.0:
        sd = 1.0
    return (arr - arr.mean()) / sd


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def regression_summary(points: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    rows = [row for row in points if all(finite(row.get(f)) for f in features) and finite(row.get("retrieval_delta"))]
    if len(rows) < 3:
        return {"n": len(rows), "r2": float("nan"), "coefficients": {}, "partial_r2": {}}
    x_raw = np.asarray([[float(row[f]) for f in features] for row in rows], dtype=np.float64)
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    x = (x_raw - x_mean) / x_std
    y = standardize_vector([float(row["retrieval_delta"]) for row in rows])
    model = LinearRegression().fit(x, y)
    pred = model.predict(x)
    full_r2 = r2_score(y, pred)
    partial: dict[str, float] = {}
    for i, feature in enumerate(features):
        reduced = np.delete(x, i, axis=1)
        reduced_model = LinearRegression().fit(reduced, y)
        reduced_r2 = r2_score(y, reduced_model.predict(reduced))
        partial[feature] = max(0.0, full_r2 - reduced_r2)
    return {
        "n": len(rows),
        "r2": full_r2,
        "coefficients": {feature: float(coef) for feature, coef in zip(features, model.coef_)},
        "partial_r2": partial,
    }


def logistic_summary(points: list[dict[str, Any]], features: list[str]) -> dict[str, Any]:
    rows = [
        row for row in points
        if all(finite(row.get(feature)) for feature in features)
        and finite(row.get("ce_delta_margin"))
    ]
    if len(rows) < 3:
        return {"n": len(rows), "failures": 0, "auc": float("nan"), "log_loss": float("nan"), "pseudo_r2": float("nan"), "coefficients": {}}
    y = np.asarray([int(float(row["ce_delta_margin"]) < 0.0) for row in rows], dtype=np.int64)
    failures = int(y.sum())
    if failures == 0 or failures == len(y):
        return {"n": len(rows), "failures": failures, "auc": float("nan"), "log_loss": float("nan"), "pseudo_r2": float("nan"), "coefficients": {}}
    x_raw = np.asarray([[float(row[f]) for f in features] for row in rows], dtype=np.float64)
    x_mean = x_raw.mean(axis=0)
    x_std = x_raw.std(axis=0)
    x_std[x_std == 0.0] = 1.0
    x = (x_raw - x_mean) / x_std
    model = LogisticRegression(max_iter=2000).fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, proba))
    loss = float(log_loss(y, proba, labels=[0, 1]))
    null_p = float(y.mean())
    null_p = min(max(null_p, 1e-9), 1.0 - 1e-9)
    null_loss = float(log_loss(y, np.full_like(y, null_p, dtype=np.float64), labels=[0, 1]))
    pseudo_r2 = 1.0 - loss / null_loss if null_loss else float("nan")
    return {
        "n": len(rows),
        "failures": failures,
        "auc": auc,
        "log_loss": loss,
        "pseudo_r2": pseudo_r2,
        "coefficients": {feature: float(coef) for feature, coef in zip(features, model.coef_[0])},
    }


def best_non_gold_ce(row: dict[str, Any], gold_ids: list[str]) -> tuple[float, str]:
    gold = {str(gid) for gid in gold_ids}
    ids = [str(idx) for idx in (row.get("retrieved_ids") or [])]
    scores = [float(score) for score in (row.get("scores") or [])]
    best_score = float("nan")
    best_id = ""
    for idx, score in zip(ids, scores):
        if idx in gold:
            continue
        if not finite(best_score) or score > best_score:
            best_score = score
            best_id = idx
    return best_score, best_id


def load_gold_score_cache(path: Path) -> dict[tuple[str, str, str], dict[str, float]]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in read_jsonl(path):
        key = (str(row.get("dataset")), str(row.get("model")), str(row.get("label")))
        needed = ("ce_raw_gold", "ce_scope_gold", "ce_delta")
        if all(finite(row.get(k)) for k in needed):
            out[key] = {
                "ce_raw_gold": float(row["ce_raw_gold"]),
                "ce_scope_gold": float(row["ce_scope_gold"]),
                "ce_delta_gold_only": float(row["ce_delta"]),
            }
    return out


def fetch_gold_docs_for_ce(dataset: str, q_scores: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from analyze_scope_gap_mechanism import fetch_docs_by_idx  # noqa: WPS433

    spec = DATASETS[dataset]
    gold_ids = sorted({str(gid) for row in q_scores.values() for gid in row["gold_ids"]})
    return fetch_docs_by_idx(spec.collection, gold_ids)


def attach_ce_gold_scores(points: list[dict[str, Any]], gold_cache: dict[tuple[str, str, str], dict[str, float]]) -> None:
    missing = []
    for point in points:
        key = (point["dataset"], point["model"], point["label"])
        cached = gold_cache.get(key)
        if cached is None:
            missing.append(point)
            continue
        point.update(cached)
    if not missing:
        return

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in missing:
        by_dataset[point["dataset"]].append(point)

    for dataset, dpoints in by_dataset.items():
        q_scores = {p["label"]: {"gold_ids": p["gold_ids"]} for p in dpoints}
        gold_docs = fetch_gold_docs_for_ce(dataset, q_scores)
        ce_points: list[dict[str, Any]] = []
        for point in dpoints:
            clone = dict(point)
            clone["gold_texts"] = [gold_docs[gid]["text"] for gid in clone["gold_ids"]]
            ce_points.append(clone)
        score_ce(ce_points, batch_size=int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "32") or "32"))
        for src, scored in zip(dpoints, ce_points):
            src["ce_raw_gold"] = float(scored["ce_raw_gold"])
            src["ce_scope_gold"] = float(scored["ce_scope_gold"])
            src["ce_delta_gold_only"] = float(scored["ce_delta"])


def fetch_doc_embeddings(collection_name: str, idxs: Iterable[str], batch_size: int) -> dict[str, np.ndarray]:
    import chromadb

    requested = list(dict.fromkeys(str(idx) for idx in idxs if str(idx)))
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR", str(REPO_ROOT / "chroma_db")))
    collection = client.get_collection(collection_name)
    found: dict[str, np.ndarray] = {}

    def store(batch: dict[str, Any]) -> None:
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or []
        embeddings = batch.get("embeddings")
        if embeddings is None:
            return
        for chroma_id, meta, emb in zip(ids, metas, embeddings):
            meta = dict(meta or {})
            idx = str(meta.get("idx") or str(chroma_id).removeprefix("doc_"))
            arr = np.asarray(emb, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm:
                arr = arr / norm
            found[idx] = arr

    for start in range(0, len(requested), batch_size):
        chunk = requested[start:start + batch_size]
        store(collection.get(ids=[f"doc_{idx}" for idx in chunk], include=["metadatas", "embeddings"]))
        print(f"[docs] {collection_name}: {min(start + batch_size, len(requested))}/{len(requested)} direct", flush=True)

    missing = [idx for idx in requested if idx not in found]
    for start in range(0, len(missing), min(batch_size, 500)):
        chunk = missing[start:start + min(batch_size, 500)]
        try:
            store(collection.get(where={"idx": {"$in": chunk}}, include=["metadatas", "embeddings"]))
        except Exception:
            for idx in chunk:
                store(collection.get(where={"idx": idx}, include=["metadatas", "embeddings"]))

    still_missing = [idx for idx in requested if idx not in found]
    if still_missing:
        raise RuntimeError(f"{collection_name}: missing embeddings for {still_missing[:10]} n={len(still_missing)}")
    return found


def embed_queries(text_by_key: dict[Any, str], batch_size: int) -> dict[Any, np.ndarray]:
    emb = get_embeddings()
    keys = list(text_by_key)
    out: dict[Any, np.ndarray] = {}
    for start in range(0, len(keys), batch_size):
        chunk = keys[start:start + batch_size]
        vecs = emb.embed_documents([text_by_key[key] for key in chunk])
        for key, vec in zip(chunk, vecs):
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm:
                arr = arr / norm
            out[key] = arr
        print(f"[embed-query] {min(start + batch_size, len(keys))}/{len(keys)}", flush=True)
    return out


def best_cosine(query_vec: np.ndarray, ids: Iterable[str], doc_emb: dict[str, np.ndarray], exclude: set[str]) -> tuple[float, str]:
    best_score = float("nan")
    best_id = ""
    for idx in ids:
        idx = str(idx)
        if idx in exclude:
            continue
        vec = doc_emb.get(idx)
        if vec is None:
            continue
        score = float(np.dot(query_vec, vec))
        if not finite(best_score) or score > best_score:
            best_score = score
            best_id = idx
    return best_score, best_id


def build_points(
    *,
    dataset: str,
    q_scores: dict[str, dict[str, Any]],
    gold_score_cache: dict[tuple[str, str, str], dict[str, float]],
) -> tuple[list[dict[str, Any]], set[str]]:
    spec = DATASETS[dataset]
    raw_text = load_questions_raw_text(dataset)
    raw_cache = load_by_label(spec.raw_cache)
    doc_ids: set[str] = set()
    points: list[dict[str, Any]] = []
    for model in MODELS:
        scope_cache = load_by_label(spec.scope_cache_by_model[model])
        raw_log = load_by_label(spec.raw_log_by_model[model])
        scope_log = load_by_label(spec.scope_log_by_model[model])
        hyre_cache = load_by_label(HYRE_CACHE[(dataset, model)])
        missing = [
            label for label in q_scores
            if label not in raw_text
            or label not in raw_cache
            or label not in scope_cache
            or label not in raw_log
            or label not in scope_log
            or label not in hyre_cache
        ]
        if missing:
            raise RuntimeError(f"{dataset}/{model}: missing labels {missing[:5]} n={len(missing)}")
        for label, score in q_scores.items():
            gold_ids = [str(gid) for gid in score["gold_ids"] if str(gid)]
            raw_row = raw_cache[label]
            scope_row = scope_cache[label]
            raw_ids = [str(idx) for idx in (raw_row.get("retrieved_ids") or [])]
            scope_ids = [str(idx) for idx in (scope_row.get("retrieved_ids") or [])]
            raw_hit = hit_at_5(raw_ids, gold_ids)
            scope_hit = hit_at_5(scope_ids, gold_ids)
            raw_ce_d, raw_ce_d_id = best_non_gold_ce(raw_row, gold_ids)
            scope_ce_d, scope_ce_d_id = best_non_gold_ce(scope_row, gold_ids)
            doc_ids.update(gold_ids)
            doc_ids.update(raw_ids)
            doc_ids.update(scope_ids)
            point = {
                "row_id": f"{dataset}/{model}/{label}",
                "dataset": dataset,
                "dataset_display": spec.display,
                "model": model,
                "label": label,
                "gold_ids": gold_ids,
                "raw_retrieved_ids": raw_ids,
                "scope_retrieved_ids": scope_ids,
                "raw_question": raw_text[label],
                "scope_passage": str(hyre_cache[label].get("hyde_passage") or ""),
                "log_perplexity": float(score["log_perplexity"]),
                "question_tokens": float(score["token_count"]),
                "oov_rate": float(score["oov_rate"]),
                "raw_hit": raw_hit,
                "scope_hit": scope_hit,
                "retrieval_delta": int(scope_hit) - int(raw_hit),
                "scope_retrieval_win": int(scope_hit == 1 and raw_hit == 0),
                "raw_retrieval_win": int(scope_hit == 0 and raw_hit == 1),
                "raw_correct": int(bool(raw_log[label].get("is_correct"))),
                "scope_correct": int(bool(scope_log[label].get("is_correct"))),
                "answer_delta": int(bool(scope_log[label].get("is_correct"))) - int(bool(raw_log[label].get("is_correct"))),
                "ce_raw_distractor": raw_ce_d,
                "ce_raw_distractor_id": raw_ce_d_id,
                "ce_scope_distractor": scope_ce_d,
                "ce_scope_distractor_id": scope_ce_d_id,
                "multi_gold_count": len(gold_ids),
            }
            cached = gold_score_cache.get((dataset, model, label))
            if cached:
                point.update(cached)
            points.append(point)
    return points, doc_ids


def attach_ce_margins(points: list[dict[str, Any]]) -> None:
    for point in points:
        point["ce_margin_raw"] = float(point["ce_raw_gold"]) - float(point["ce_raw_distractor"])
        point["ce_margin_scope"] = float(point["ce_scope_gold"]) - float(point["ce_scope_distractor"])
        point["ce_delta_margin"] = float(point["ce_margin_scope"]) - float(point["ce_margin_raw"])


def attach_cosine_margins(points: list[dict[str, Any]], doc_embeddings_by_dataset: dict[str, dict[str, np.ndarray]], batch_size: int) -> None:
    raw_texts = {(p["dataset"], p["label"]): p["raw_question"] for p in points}
    scope_texts = {(p["dataset"], p["model"], p["label"]): p["scope_passage"] for p in points}
    print(f"[embed-query] raw={len(raw_texts)} scope={len(scope_texts)}", flush=True)
    raw_emb = embed_queries(raw_texts, batch_size)
    scope_emb = embed_queries(scope_texts, batch_size)

    raw_gold_cache: dict[tuple[str, str], tuple[float, str]] = {}
    raw_distractor_cache: dict[tuple[str, str], tuple[float, str]] = {}
    for point in points:
        raw_key = (point["dataset"], point["label"])
        if raw_key not in raw_gold_cache:
            gold_set = set(point["gold_ids"])
            docs = doc_embeddings_by_dataset[point["dataset"]]
            rv = raw_emb[raw_key]
            raw_gold_cache[raw_key] = best_cosine(rv, point["gold_ids"], docs, exclude=set())
            raw_distractor_cache[raw_key] = best_cosine(rv, point["raw_retrieved_ids"], docs, exclude=gold_set)
        docs = doc_embeddings_by_dataset[point["dataset"]]
        sv = scope_emb[(point["dataset"], point["model"], point["label"])]
        scope_gold, scope_gold_id = best_cosine(sv, point["gold_ids"], docs, exclude=set())
        scope_dist, scope_dist_id = best_cosine(sv, point["scope_retrieved_ids"], docs, exclude=set(point["gold_ids"]))
        raw_gold, raw_gold_id = raw_gold_cache[raw_key]
        raw_dist, raw_dist_id = raw_distractor_cache[raw_key]
        point["cos_raw_gold"] = raw_gold
        point["cos_raw_gold_id"] = raw_gold_id
        point["cos_scope_gold"] = scope_gold
        point["cos_scope_gold_id"] = scope_gold_id
        point["cos_delta_gold_only"] = scope_gold - raw_gold
        point["cos_raw_distractor"] = raw_dist
        point["cos_raw_distractor_id"] = raw_dist_id
        point["cos_scope_distractor"] = scope_dist
        point["cos_scope_distractor_id"] = scope_dist_id
        point["cos_margin_raw"] = raw_gold - raw_dist
        point["cos_margin_scope"] = scope_gold - scope_dist
        point["cos_delta_margin"] = point["cos_margin_scope"] - point["cos_margin_raw"]


def clean_point_for_cache(point: dict[str, Any]) -> dict[str, Any]:
    drop = {"raw_question", "scope_passage", "raw_retrieved_ids", "scope_retrieved_ids"}
    return {k: v for k, v in point.items() if k not in drop}


def summarize_outcomes(points: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(points)
    help_n = sum(int(p["scope_retrieval_win"]) for p in points)
    hurt_n = sum(int(p["raw_retrieval_win"]) for p in points)
    return {
        "n": n,
        "raw_hit": mean([p["raw_hit"] for p in points]),
        "scope_hit": mean([p["scope_hit"] for p in points]),
        "retrieval_delta": mean([p["retrieval_delta"] for p in points]),
        "help": help_n,
        "hurt": hurt_n,
        "ri": (help_n - hurt_n) / n if n else float("nan"),
        "answer_delta": mean([p["answer_delta"] for p in points]),
        "ce_delta_margin": mean([p["ce_delta_margin"] for p in points]),
        "cos_delta_margin": mean([p["cos_delta_margin"] for p in points]),
    }


def quintile_bins(points: list[dict[str, Any]], axis: str, bins: int = 5) -> list[dict[str, Any]]:
    valid = [p for p in points if finite(p.get(axis))]
    ordered = sorted(valid, key=lambda p: (float(p[axis]), p["dataset"], p["model"], p["label"]))
    n = len(ordered)
    out: list[dict[str, Any]] = []
    for b in range(bins):
        lo = round(b * n / bins)
        hi = round((b + 1) * n / bins)
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        vals = [float(p[axis]) for p in chunk]
        s = summarize_outcomes(chunk)
        out.append({
            "bin": b + 1,
            "axis_min": min(vals),
            "axis_median": median(vals),
            "axis_max": max(vals),
            **s,
        })
    return out


def crossover_read(bins: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [float(row["retrieval_delta"]) for row in bins]
    has_pos = any(v > 0.0 for v in deltas)
    has_neg = any(v < 0.0 for v in deltas)
    low = deltas[0] if deltas else float("nan")
    high = deltas[-1] if deltas else float("nan")
    sign_cross = has_pos and has_neg
    monotone_positive = bool(deltas) and all(v > 0.0 for v in deltas)
    declining = finite(low) and finite(high) and high < low
    return {
        "low": low,
        "high": high,
        "sign_cross": sign_cross,
        "monotone_positive": monotone_positive,
        "declining": declining,
    }


def p2_stats(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for dataset in ("barexam", "housing"):
        dpoints = [p for p in points if p["dataset"] == dataset]
        for model in MODELS:
            groups.append((dataset, model, [p for p in dpoints if p["model"] == model]))
        groups.append((dataset, "pooled", dpoints))
    groups.append(("pooled", "pooled", points))
    for dataset, model, gpoints in groups:
        for family, margin_axis, gold_axis in (
            ("CE", "ce_delta_margin", "ce_delta_gold_only"),
            ("Cosine", "cos_delta_margin", "cos_delta_gold_only"),
        ):
            margin = corr_pack(gpoints, margin_axis, "retrieval_delta")
            gold = corr_pack(gpoints, gold_axis, "retrieval_delta")
            rows.append({
                "dataset": dataset,
                "model": model,
                "family": family,
                "n": margin["n"],
                "full_spearman": margin["spearman"],
                "full_kendall": margin["kendall"],
                "gold_spearman": gold["spearman"],
                "gold_kendall": gold["kendall"],
                "spearman_gain": abs(margin["spearman"]) - abs(gold["spearman"]) if finite(margin["spearman"]) and finite(gold["spearman"]) else float("nan"),
                "kendall_gain": abs(margin["kendall"]) - abs(gold["kendall"]) if finite(margin["kendall"]) and finite(gold["kendall"]) else float("nan"),
            })
    return rows


def verdicts(points: list[dict[str, Any]], p2_rows: list[dict[str, Any]], p1_bins: dict[str, list[dict[str, Any]]], regressions: dict[str, dict[str, Any]], p4_rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    ce_pooled = next(row for row in p2_rows if row["dataset"] == "pooled" and row["model"] == "pooled" and row["family"] == "CE")
    p2_supported = (
        finite(ce_pooled["full_spearman"])
        and abs(float(ce_pooled["full_spearman"])) >= 0.2
        and finite(ce_pooled["spearman_gain"])
        and float(ce_pooled["spearman_gain"]) > 0.0
    )
    p1_reads = {key: crossover_read(rows) for key, rows in p1_bins.items()}
    p1_supported = all(read["sign_cross"] and read["declining"] and not read["monotone_positive"] for key, read in p1_reads.items() if key in {"barexam", "housing"})

    joint = regressions["joint"]
    partial = joint.get("partial_r2", {})
    margin_part = max(float(partial.get("ce_margin_raw", 0.0)), float(partial.get("ce_delta_margin", 0.0)))
    confound_part = max(float(partial.get("dataset_id", 0.0)), float(partial.get("question_tokens", 0.0)), float(partial.get("log_perplexity", 0.0)), float(partial.get("oov_rate", 0.0)))
    within_ok = all(crossover_read(p1_bins[key])["sign_cross"] for key in ("barexam", "housing"))
    p3_supported = within_ok and margin_part >= confound_part

    p4_q = p4_rows["quality"]
    p4_m = p4_rows["margin"]
    p4_supported = finite(p4_m.get("auc")) and finite(p4_q.get("auc")) and float(p4_m["auc"]) > float(p4_q["auc"]) + 0.05

    return {
        "P1": {
            "verdict": "supported" if p1_supported else "killed",
            "key": (
                f"BarExam low/high M_raw net {pct(p1_reads['barexam']['low'])}/{pct(p1_reads['barexam']['high'])}; "
                f"Housing low/high {pct(p1_reads['housing']['low'])}/{pct(p1_reads['housing']['high'])}"
            ),
        },
        "P2": {
            "verdict": "supported" if p2_supported else "killed",
            "key": (
                f"CE pooled full-margin rho={fmt_float(ce_pooled['full_spearman'])}, "
                f"tau={fmt_float(ce_pooled['full_kendall'])}; gold-only rho={fmt_float(ce_pooled['gold_spearman'])}; "
                f"gain={fmt_float(ce_pooled['spearman_gain'])}"
            ),
        },
        "P3": {
            "verdict": "supported" if p3_supported else "killed",
            "key": (
                f"within-dataset crossover={'yes' if within_ok else 'no'}; "
                f"joint max margin partial-R2={fmt_float(margin_part)}, max confound partial-R2={fmt_float(confound_part)}"
            ),
        },
        "P4": {
            "verdict": "supported" if p4_supported else "killed",
            "key": (
                f"failure AUC quality(logPPL+OOV)={fmt_float(p4_q.get('auc'))}; "
                f"margin(M_raw+CE(scope,gold))={fmt_float(p4_m.get('auc'))}"
            ),
        },
    }


def model_display(model: str) -> str:
    if model == "pooled":
        return "Pooled"
    return MODEL_LABELS.get(model, model)


def dataset_display(dataset: str) -> str:
    if dataset == "pooled":
        return "Pooled"
    return DATASETS[dataset].display


def coefficient_line(row: dict[str, Any]) -> str:
    coeffs = row.get("coefficients", {})
    partial = row.get("partial_r2", {})
    bits = []
    for key in coeffs:
        bits.append(f"`{key}` beta={fmt_float(coeffs[key])}, partial-R2={fmt_float(partial.get(key, float('nan')))}")
    return "; ".join(bits)


def fmt_range(lo: float, hi: float) -> str:
    return f"[{fmt_float(lo)}, {fmt_float(hi)}]"


def make_report(
    *,
    output: Path,
    points: list[dict[str, Any]],
    p2_rows: list[dict[str, Any]],
    p1_bins: dict[str, list[dict[str, Any]]],
    regressions: dict[str, dict[str, Any]],
    p4_rows: dict[str, dict[str, Any]],
    used_gold_cache: Path,
    lm_cache_dir: Path,
) -> None:
    verdict_table = verdicts(points, p2_rows, p1_bins, regressions, p4_rows)
    lines: list[str] = []
    lines.append("# Affinity-Margin Mechanism Test - 2026-05-26")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Read-only results-lane analysis over existing BarExamQA and HousingQA state-filtered caches. No answer/model calls were made and no files under `paper/` were edited.")
    lines.append("")
    lines.append("Gold margin definition: `M(x) = aff(x, best gold) - max_d aff(x, d)`, where `d` ranges over that condition's own retrieved top-10 non-gold cache entries. HousingQA multi-gold rows use the max over the gold set and exclude all gold ids from the distractor max.")
    lines.append("")
    lines.append("- CE gold affinities come from the prior query-gap CE scoring cache when present; CE distractor affinities use the stored retrieval-cache cross-encoder scores.")
    lines.append("- Cosine margins use the configured gte query embedder and stored Chroma document embeddings.")
    lines.append("- Retrieval gain is `SCOPE Hit@5 - raw Hit@5`; Collins-Thompson RI is `(n_help - n_hurt) / N`.")
    lines.append("")

    lines.append("## Verdicts")
    lines.append("")
    lines.append("| Prediction | Verdict | Key numbers |")
    lines.append("|---|---|---|")
    for pred in ("P1", "P2", "P3", "P4"):
        row = verdict_table[pred]
        lines.append(f"| {pred} | **{row['verdict']}** | {row['key']} |")
    lines.append("")

    lines.append("## Collins-Thompson Robustness Index")
    lines.append("")
    lines.append("| Dataset | Model | N | Raw Hit@5 | SCOPE Hit@5 | Net delta | Help | Hurt | RI | Answer delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in ("barexam", "housing"):
        dpoints = [p for p in points if p["dataset"] == dataset]
        for model in (*MODELS, "pooled"):
            gpoints = dpoints if model == "pooled" else [p for p in dpoints if p["model"] == model]
            s = summarize_outcomes(gpoints)
            lines.append(
                f"| {dataset_display(dataset)} | {model_display(model)} | {s['n']} | {pct(s['raw_hit'])} | {pct(s['scope_hit'])} | "
                f"{pct(s['retrieval_delta'])} | {s['help']} | {s['hurt']} | {fmt_float(s['ri'])} | {pct(s['answer_delta'])} |"
            )
    lines.append("")

    lines.append("## P2: Delta-Margin Correlation")
    lines.append("")
    lines.append("Full margin is `deltaM = M_scope - M_raw`. Gold-only delta is `aff(scope,gold) - aff(raw,gold)`. P2 requires the full margin to correlate with retrieval gain and improve over gold-only affinity.")
    lines.append("")
    lines.append("| Dataset | Model | Affinity | N | Full rho | Full tau | Gold-only rho | Gold-only tau | Rho gain | Tau gain |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in p2_rows:
        if row["dataset"] not in {"barexam", "housing", "pooled"}:
            continue
        lines.append(
            f"| {dataset_display(row['dataset'])} | {model_display(row['model'])} | {row['family']} | {row['n']} | "
            f"{fmt_float(row['full_spearman'])} | {fmt_float(row['full_kendall'])} | "
            f"{fmt_float(row['gold_spearman'])} | {fmt_float(row['gold_kendall'])} | "
            f"{fmt_float(row['spearman_gain'])} | {fmt_float(row['kendall_gain'])} |"
        )
    lines.append("")

    lines.append("## P1: Raw-Margin Quintiles")
    lines.append("")
    lines.append("Bins sort by CE `M_raw` within each dataset, pooled across model rows. A crossover means SCOPE helps more in low raw-margin bins and stops helping or hurts as raw margin rises.")
    lines.append("")
    for dataset in ("barexam", "housing", "pooled"):
        lines.append(f"### {dataset_display(dataset)}")
        lines.append("")
        lines.append("| Bin | N | CE M_raw median | CE M_raw range | Raw Hit@5 | SCOPE Hit@5 | Net delta | Help | Hurt | RI |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in p1_bins[dataset]:
            lines.append(
                f"| {row['bin']} | {row['n']} | {fmt_float(row['axis_median'])} | "
                f"{fmt_range(row['axis_min'], row['axis_max'])} | "
                f"{pct(row['raw_hit'])} | {pct(row['scope_hit'])} | {pct(row['retrieval_delta'])} | "
                f"{row['help']} | {row['hurt']} | {fmt_float(row['ri'])} |"
            )
        lines.append("")

    lines.append("## P3: Confound Checks")
    lines.append("")
    lines.append("The primary check is whether the help-to-hurt crossover appears within BarExamQA and within HousingQA, not only between datasets. The secondary check is a standardized OLS regression on retrieval gain with partial-R2 deltas from dropping each feature.")
    lines.append("")
    lines.append("| Regression | N | R2 | Standardized coefficients and partial-R2 |")
    lines.append("|---|---:|---:|---|")
    for name in ("joint", "barexam", "housing"):
        row = regressions[name]
        label = {"joint": "Joint + dataset id", "barexam": "BarExamQA only", "housing": "HousingQA only"}[name]
        lines.append(f"| {label} | {row['n']} | {fmt_float(row['r2'])} | {coefficient_line(row)} |")
    lines.append("")

    lines.append("## P4: Failure Model")
    lines.append("")
    lines.append("Failure is `1[CE deltaM < 0]`. The hallucination/surprise explanation would be plausible if OOV/log-perplexity explained these failures about as well as the margin features.")
    lines.append("")
    lines.append("| Model | N | Failures | AUC | Log loss | Pseudo-R2 | Coefficients |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for key, row in p4_rows.items():
        label = {
            "quality": "OOV + log-perplexity",
            "margin": "CE M_raw + CE(scope,gold)",
            "combined": "Combined",
        }.get(key, key)
        coeff = "; ".join(f"`{k}`={fmt_float(v)}" for k, v in row.get("coefficients", {}).items())
        lines.append(
            f"| {label} | {row['n']} | {row['failures']} | {fmt_float(row['auc'])} | "
            f"{fmt_float(row['log_loss'])} | {fmt_float(row['pseudo_r2'])} | {coeff} |"
        )
    lines.append("")

    lines.append("## Risk-Reward Reading")
    lines.append("")
    pooled = summarize_outcomes(points)
    be = summarize_outcomes([p for p in points if p["dataset"] == "barexam"])
    hq = summarize_outcomes([p for p in points if p["dataset"] == "housing"])
    ce_pooled = next(row for row in p2_rows if row["dataset"] == "pooled" and row["model"] == "pooled" and row["family"] == "CE")
    lines.append(
        f"- Overall RI is {fmt_float(pooled['ri'])}: {pooled['help']} SCOPE-only retrieval hits versus {pooled['hurt']} raw-only hits over {pooled['n']} question-model rows."
    )
    lines.append(
        f"- BarExamQA is favorable ({pct(be['retrieval_delta'])} net Hit@5, RI {fmt_float(be['ri'])}); HousingQA state-filtered is unfavorable ({pct(hq['retrieval_delta'])}, RI {fmt_float(hq['ri'])})."
    )
    lines.append(
        f"- The margin mechanism is useful only if the distractor term adds signal beyond gold affinity. In this run CE full-margin rho is {fmt_float(ce_pooled['full_spearman'])} versus gold-only rho {fmt_float(ce_pooled['gold_spearman'])}."
    )
    lines.append(
        "- Practical implication: a no-gold router should estimate raw-retrieval confidence and expansion risk before applying SCOPE broadly; raw state/jurisdiction anchors remain valuable on HousingQA."
    )
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    seen: list[str] = []
    for dataset in ("barexam", "housing"):
        for path in source_paths_for(DATASETS[dataset]):
            if path not in seen:
                seen.append(path)
        for model in MODELS:
            path = HYRE_CACHE[(dataset, model)]
            if path not in seen:
                seen.append(path)
    for path in seen:
        lines.append(f"- `{path}`")
    lines.append(f"- CE gold score cache: `{used_gold_cache}`")
    lines.append(f"- Perplexity LM cache: `{lm_cache_dir}`")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_affinity_margin_oncache.py \\")
    lines.append("  --output docs/generated/affinity_margin_oncache_2026-05-26.md")
    lines.append("```")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def load_points_cache(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/generated/affinity_margin_oncache_2026-05-26.md")
    parser.add_argument("--lm-cache-dir", type=Path, default=DEFAULT_LM_CACHE_DIR)
    parser.add_argument("--gold-score-cache", type=Path, default=DEFAULT_GOLD_SCORE_CACHE)
    parser.add_argument("--points-out", type=Path, default=DEFAULT_POINTS_OUT)
    parser.add_argument("--reuse-points", action="store_true", help="Reuse --points-out if present.")
    parser.add_argument("--doc-batch-size", type=int, default=5000)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", default=["barexam", "housing"], choices=sorted(DATASETS))
    args = parser.parse_args()

    points: list[dict[str, Any]] = load_points_cache(args.points_out) if args.reuse_points else []
    if points:
        print(f"[cache] loaded {len(points)} points from {args.points_out}", flush=True)
    else:
        gold_score_cache = load_gold_score_cache(args.gold_score_cache)
        all_points: list[dict[str, Any]] = []
        doc_ids_by_dataset: dict[str, set[str]] = defaultdict(set)
        for dataset in args.datasets:
            spec = DATASETS[dataset]
            print(f"[dataset] {dataset}: question scores", flush=True)
            q_scores = question_scores(spec, build_or_load_lm(spec, args.lm_cache_dir, 20000))
            dpoints, doc_ids = build_points(dataset=dataset, q_scores=q_scores, gold_score_cache=gold_score_cache)
            attach_ce_gold_scores(dpoints, gold_score_cache)
            attach_ce_margins(dpoints)
            all_points.extend(dpoints)
            doc_ids_by_dataset[dataset].update(doc_ids)

        doc_embeddings_by_dataset: dict[str, dict[str, np.ndarray]] = {}
        for dataset, doc_ids in doc_ids_by_dataset.items():
            spec = DATASETS[dataset]
            print(f"[dataset] {dataset}: fetch doc embeddings={len(doc_ids)}", flush=True)
            doc_embeddings_by_dataset[dataset] = fetch_doc_embeddings(spec.collection, doc_ids, args.doc_batch_size)
        attach_cosine_margins(all_points, doc_embeddings_by_dataset, args.embed_batch_size)
        points = all_points

        args.points_out.parent.mkdir(parents=True, exist_ok=True)
        with args.points_out.open("w") as f:
            for point in points:
                f.write(json.dumps(clean_point_for_cache(point), sort_keys=True) + "\n")
        print(f"[cache] wrote {args.points_out}", flush=True)

    p2_rows = p2_stats(points)
    p1_bins = {
        "barexam": quintile_bins([p for p in points if p["dataset"] == "barexam"], "ce_margin_raw"),
        "housing": quintile_bins([p for p in points if p["dataset"] == "housing"], "ce_margin_raw"),
        "pooled": quintile_bins(points, "ce_margin_raw"),
    }
    regression_features_joint = ["ce_margin_raw", "ce_delta_margin", "log_perplexity", "question_tokens", "oov_rate", "dataset_id"]
    for point in points:
        point["dataset_id"] = 1.0 if point["dataset"] == "housing" else 0.0
    regressions = {
        "joint": regression_summary(points, regression_features_joint),
        "barexam": regression_summary(
            [p for p in points if p["dataset"] == "barexam"],
            ["ce_margin_raw", "ce_delta_margin", "log_perplexity", "question_tokens", "oov_rate"],
        ),
        "housing": regression_summary(
            [p for p in points if p["dataset"] == "housing"],
            ["ce_margin_raw", "ce_delta_margin", "log_perplexity", "question_tokens", "oov_rate"],
        ),
    }
    p4_rows = {
        "quality": logistic_summary(points, ["oov_rate", "log_perplexity"]),
        "margin": logistic_summary(points, ["ce_margin_raw", "ce_scope_gold"]),
        "combined": logistic_summary(points, ["oov_rate", "log_perplexity", "ce_margin_raw", "ce_scope_gold"]),
    }

    make_report(
        output=args.output,
        points=points,
        p2_rows=p2_rows,
        p1_bins=p1_bins,
        regressions=regressions,
        p4_rows=p4_rows,
        used_gold_cache=args.gold_score_cache,
        lm_cache_dir=args.lm_cache_dir,
    )
    print(args.output)


if __name__ == "__main__":
    main()
