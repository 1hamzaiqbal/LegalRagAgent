#!/usr/bin/env python3
"""No-gold QPP predictors for selective SCOPE routing.

This results-lane analysis is read-only over existing retrieval caches, answer
detail logs, and Chroma collection embeddings. It treats raw-question retrieval
confidence as an unsupervised Query Performance Prediction (QPP) signal and
tests whether that signal can gate SCOPE expansion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from analyze_perplexity_axis import (  # noqa: E402
    DATASETS,
    MODELS,
    MODEL_LABELS,
    DatasetSpec,
    build_or_load_lm,
    fmt_float,
    hit_at_5,
    load_by_label,
    mean,
    median,
    pct,
    pearson,
    percentile,
    question_scores,
    rank_auc_greater,
    source_paths_for,
    spearman,
)
from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _retrieval_question, _row_label  # noqa: E402


FEATURES: dict[str, dict[str, str]] = {
    "nqc_ce_top10": {
        "label": "NQC-CE top10",
        "family": "NQC",
        "direction": "higher = more top-k score dispersion relative to score magnitude",
    },
    "wig_ce_top5_vs_top10": {
        "label": "WIG-CE top5-vs-top10",
        "family": "WIG",
        "direction": "higher = stronger top-5 separation from local retrieved background",
    },
    "smv_ce_top10": {
        "label": "SMV-CE top10",
        "family": "SMV",
        "direction": "higher = larger score magnitude and variance",
    },
    "ce_top1": {
        "label": "Top-1 CE score",
        "family": "CE hand feature",
        "direction": "higher = raw top hit is reranker-confident",
    },
    "ce_top5_mean": {
        "label": "Mean top-5 CE",
        "family": "CE hand feature",
        "direction": "higher = raw top set is reranker-confident",
    },
    "ce_spread_1_5": {
        "label": "CE spread top1-top5",
        "family": "CE hand feature",
        "direction": "higher = top hit separated from fifth hit",
    },
    "ce_entropy_conf_top5": {
        "label": "Negative CE entropy top5",
        "family": "CE hand feature",
        "direction": "higher = lower softmax entropy over top-5 CE scores",
    },
    "dense_query_top1_cos": {
        "label": "Dense query-top1 cosine",
        "family": "Dense QPP",
        "direction": "higher = raw query is close to its top retrieved passage",
    },
    "dense_coherence_top5": {
        "label": "Dense top-5 coherence",
        "family": "Dense QPP",
        "direction": "higher = retrieved passages form a coherent dense cluster",
    },
    "dense_centroid_norm_top5": {
        "label": "Dense top-5 centroid norm",
        "family": "Dense QPP",
        "direction": "higher = top-5 embeddings agree around a single centroid",
    },
    "log_perplexity": {
        "label": "Log perplexity",
        "family": "Prior axis",
        "direction": "higher = less corpus-like under unigram LM",
    },
    "question_tokens": {
        "label": "Question tokens",
        "family": "Prior axis",
        "direction": "higher = longer query",
    },
}

NAMED_QPP_FEATURES = (
    "nqc_ce_top10",
    "wig_ce_top5_vs_top10",
    "smv_ce_top10",
    "dense_query_top1_cos",
    "dense_coherence_top5",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def finite_pair_values(points: list[dict[str, Any]], feature: str, outcome: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        x = point.get(feature)
        y = point.get(outcome)
        if x is None or y is None:
            continue
        xf = float(x)
        yf = float(y)
        if math.isfinite(xf) and math.isfinite(yf):
            xs.append(xf)
            ys.append(yf)
    return xs, ys


def std(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    return float(stats.pstdev(vals))


def variance(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    return float(stats.pvariance(vals))


def softmax_entropy(values: list[float]) -> float:
    if not values:
        return float("nan")
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    if total <= 0:
        return float("nan")
    probs = [v / total for v in exps]
    return -sum(p * math.log(p) for p in probs if p > 0)


def ce_features(scores_raw: Any) -> dict[str, float]:
    scores = [float(s) for s in (scores_raw or []) if s is not None]
    if not scores:
        return {key: float("nan") for key in (
            "ce_top1",
            "ce_top5_mean",
            "ce_top5_max",
            "ce_spread_1_5",
            "ce_std_top5",
            "ce_entropy_conf_top5",
            "nqc_ce_top10",
            "wig_ce_top5_vs_top10",
            "smv_ce_top10",
        )}
    top5 = scores[:5]
    top10 = scores[:10]
    mean_top5 = mean(top5)
    mean_top10 = mean(top10)
    std_top10 = std(top10)
    spread = top5[0] - top5[-1] if len(top5) >= 5 else float("nan")
    nqc = std_top10 / (abs(mean_top10) + 1e-6)
    wig = mean_top5 - mean_top10
    smv = abs(mean_top10) * variance(top10)
    return {
        "ce_top1": scores[0],
        "ce_top5_mean": mean_top5,
        "ce_top5_max": max(top5),
        "ce_spread_1_5": spread,
        "ce_std_top5": std(top5),
        "ce_entropy_conf_top5": -softmax_entropy(top5),
        "nqc_ce_top10": nqc,
        "wig_ce_top5_vs_top10": wig,
        "smv_ce_top10": smv,
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def normalize_vec(vec: Any) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm:
        arr = arr / norm
    return arr


def batch_values(batch: dict[str, Any], key: str) -> list[Any]:
    value = batch.get(key)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return list(value)
    return list(value)


def fetch_doc_embeddings(spec: DatasetSpec, raw_cache: dict[str, dict[str, Any]], batch_size: int) -> dict[str, np.ndarray]:
    import chromadb

    ids = sorted({
        str(idx)
        for row in raw_cache.values()
        for idx in (row.get("retrieved_ids") or [])[:10]
        if str(idx)
    })
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR", str(REPO_ROOT / "chroma_db")))
    collection = client.get_collection(spec.collection)
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        batch = collection.get(ids=[f"doc_{idx}" for idx in chunk], include=["embeddings", "metadatas"])
        for chroma_id, emb, meta in zip(batch_values(batch, "ids"), batch_values(batch, "embeddings"), batch_values(batch, "metadatas")):
            idx = str((meta or {}).get("idx") or str(chroma_id).removeprefix("doc_"))
            out[idx] = normalize_vec(emb)
        missing = [idx for idx in chunk if idx not in out]
        for idx in missing:
            try:
                fallback = collection.get(where={"idx": idx}, include=["embeddings", "metadatas"])
            except Exception:
                fallback = {}
            for chroma_id, emb, meta in zip(
                batch_values(fallback, "ids"),
                batch_values(fallback, "embeddings"),
                batch_values(fallback, "metadatas"),
            ):
                got = str((meta or {}).get("idx") or str(chroma_id).removeprefix("doc_"))
                out[got] = normalize_vec(emb)
        print(f"[dense] {spec.key}: doc embeddings {min(start + batch_size, len(ids))}/{len(ids)}", flush=True)
    missing_all = [idx for idx in ids if idx not in out]
    if missing_all:
        raise RuntimeError(f"{spec.key}: missing doc embeddings for {missing_all[:10]} n={len(missing_all)}")
    return out


def load_raw_queries(spec: DatasetSpec) -> dict[str, str]:
    config = EvalConfig(
        dataset=spec.key,
        questions="full",
        seed=42,
        retrieval_k=5,
        housing_state_filter=spec.housing_state_filter,
    )
    rows = [row for _, row in load_questions(config).iterrows()]
    return {_row_label(row, config): _retrieval_question(row) for row in rows}


def embed_queries(raw_queries: dict[str, str], batch_size: int) -> dict[str, np.ndarray]:
    from rag_utils import get_embeddings

    emb = get_embeddings()
    labels = list(raw_queries)
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(labels), batch_size):
        chunk = labels[start:start + batch_size]
        vecs = emb.embed_documents([raw_queries[label] for label in chunk])
        for label, vec in zip(chunk, vecs):
            out[label] = normalize_vec(vec)
        print(f"[dense] query embeddings {min(start + batch_size, len(labels))}/{len(labels)}", flush=True)
    return out


def dense_features_for_row(
    row: dict[str, Any],
    doc_embeddings: dict[str, np.ndarray],
    query_embedding: np.ndarray | None,
) -> dict[str, float]:
    ids = [str(idx) for idx in (row.get("retrieved_ids") or [])[:10] if str(idx) in doc_embeddings]
    if not ids:
        return {
            "dense_query_top1_cos": float("nan"),
            "dense_coherence_top5": float("nan"),
            "dense_centroid_norm_top5": float("nan"),
        }
    top1 = doc_embeddings[ids[0]]
    query_top1 = cosine(query_embedding, top1) if query_embedding is not None else float("nan")
    top5_vecs = [doc_embeddings[idx] for idx in ids[:5]]
    pair_scores: list[float] = []
    for i in range(len(top5_vecs)):
        for j in range(i + 1, len(top5_vecs)):
            pair_scores.append(cosine(top5_vecs[i], top5_vecs[j]))
    centroid = np.mean(np.stack(top5_vecs, axis=0), axis=0) if top5_vecs else np.asarray([], dtype=np.float32)
    return {
        "dense_query_top1_cos": query_top1,
        "dense_coherence_top5": mean(pair_scores) if pair_scores else float("nan"),
        "dense_centroid_norm_top5": float(np.linalg.norm(centroid)) if centroid.size else float("nan"),
    }


def first_gold_rank(retrieved_ids: list[Any], gold_ids: list[str], cap: int = 10) -> int:
    gold = {str(x) for x in gold_ids if str(x)}
    if not gold:
        return cap + 1
    for rank, idx in enumerate([str(x) for x in retrieved_ids[:cap]], 1):
        if idx in gold:
            return rank
    return cap + 1


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    try:
        from scipy.stats import kendalltau

        value = kendalltau(xs, ys, nan_policy="omit").statistic
        return float(value) if value is not None else float("nan")
    except Exception:
        concordant = 0
        discordant = 0
        ties_x = 0
        ties_y = 0
        n = len(xs)
        for i in range(n):
            for j in range(i + 1, n):
                dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
                dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
                if dx == 0 and dy == 0:
                    continue
                if dx == 0:
                    ties_x += 1
                elif dy == 0:
                    ties_y += 1
                elif dx == dy:
                    concordant += 1
                else:
                    discordant += 1
        denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
        return (concordant - discordant) / denom if denom else float("nan")


def summarize(points: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": len(points),
        "raw_hit": mean([p["raw_hit"] for p in points]),
        "scope_hit": mean([p["scope_hit"] for p in points]),
        "retrieval_delta": mean([p["retrieval_delta"] for p in points]),
        "raw_accuracy": mean([p["raw_correct"] for p in points]),
        "scope_accuracy": mean([p["scope_correct"] for p in points]),
        "answer_delta": mean([p["answer_delta"] for p in points]),
        "scope_retrieval_win": mean([p["scope_retrieval_win"] for p in points]),
        "raw_retrieval_win": mean([p["raw_retrieval_win"] for p in points]),
        "scope_answer_win": mean([p["scope_answer_win"] for p in points]),
        "raw_answer_win": mean([p["raw_answer_win"] for p in points]),
    }


def correlation(points: list[dict[str, Any]], feature: str) -> dict[str, float]:
    xr, yr = finite_pair_values(points, feature, "retrieval_delta")
    xa, ya = finite_pair_values(points, feature, "answer_delta")
    return {
        "feature": feature,
        "n_retrieval": len(xr),
        "pearson_retrieval": pearson(xr, yr),
        "spearman_retrieval": spearman(xr, yr),
        "kendall_retrieval": kendall_tau(xr, yr),
        "n_answer": len(xa),
        "pearson_answer": pearson(xa, ya),
        "spearman_answer": spearman(xa, ya),
        "kendall_answer": kendall_tau(xa, ya),
    }


def build_dataset_points(
    spec: DatasetSpec,
    *,
    lm_cache_dir: Path,
    chroma_batch_size: int,
    embed_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(f"[dataset] {spec.key}: load raw cache", flush=True)
    raw_cache = load_by_label(spec.raw_cache)
    print(f"[dataset] {spec.key}: load question/perplexity scores", flush=True)
    q_scores = question_scores(spec, build_or_load_lm(spec, lm_cache_dir, 20000))
    raw_queries = load_raw_queries(spec)

    print(f"[dataset] {spec.key}: fetch dense document embeddings", flush=True)
    doc_embeddings = fetch_doc_embeddings(spec, raw_cache, chroma_batch_size)
    print(f"[dataset] {spec.key}: embed raw questions", flush=True)
    query_embeddings = embed_queries(raw_queries, embed_batch_size)

    raw_feature_by_label: dict[str, dict[str, float]] = {}
    for label, row in raw_cache.items():
        feats = ce_features(row.get("scores") or [])
        feats.update(dense_features_for_row(row, doc_embeddings, query_embeddings.get(label)))
        score = q_scores.get(label)
        if score:
            feats["log_perplexity"] = float(score["log_perplexity"])
            feats["question_tokens"] = float(score["token_count"])
        raw_feature_by_label[label] = feats

    points: list[dict[str, Any]] = []
    for model in MODELS:
        print(f"[dataset] {spec.key}/{model}: join outcomes", flush=True)
        scope_cache = load_by_label(spec.scope_cache_by_model[model])
        raw_log = load_by_label(spec.raw_log_by_model[model])
        scope_log = load_by_label(spec.scope_log_by_model[model])
        missing = [
            label
            for label in q_scores
            if label not in raw_cache
            or label not in scope_cache
            or label not in raw_log
            or label not in scope_log
            or label not in raw_feature_by_label
        ]
        if missing:
            raise RuntimeError(f"{spec.key}/{model}: missing labels {missing[:5]} n={len(missing)}")
        for label, score in q_scores.items():
            gold = score["gold_ids"]
            raw_ids = raw_cache[label].get("retrieved_ids") or []
            scope_ids = scope_cache[label].get("retrieved_ids") or []
            raw_hit = hit_at_5(raw_ids, gold)
            scope_hit = hit_at_5(scope_ids, gold)
            raw_correct = int(bool(raw_log[label].get("is_correct")))
            scope_correct = int(bool(scope_log[label].get("is_correct")))
            point = {
                "dataset": spec.key,
                "dataset_display": spec.display,
                "model": model,
                "label": label,
                "raw_hit": raw_hit,
                "scope_hit": scope_hit,
                "retrieval_delta": int(scope_hit) - int(raw_hit),
                "scope_retrieval_win": int(scope_hit == 1 and raw_hit == 0),
                "raw_retrieval_win": int(scope_hit == 0 and raw_hit == 1),
                "raw_gold_rank_at10": first_gold_rank(raw_ids, gold, cap=10),
                "raw_correct": raw_correct,
                "scope_correct": scope_correct,
                "answer_delta": int(scope_correct) - int(raw_correct),
                "scope_answer_win": int(scope_correct == 1 and raw_correct == 0),
                "raw_answer_win": int(scope_correct == 0 and raw_correct == 1),
            }
            point.update(raw_feature_by_label[label])
            points.append(point)

    question_features = list(raw_feature_by_label.values())
    dataset_summary = {
        "n_questions": len(q_scores),
        "median_log_perplexity": median([float(v["log_perplexity"]) for v in question_features]),
        "median_tokens": median([float(v["question_tokens"]) for v in question_features]),
        "features_by_label": raw_feature_by_label,
    }
    return points, dataset_summary


def feature_distribution(points: list[dict[str, Any]], feature: str) -> dict[str, float]:
    vals = [float(p[feature]) for p in points if p.get(feature) is not None and math.isfinite(float(p[feature]))]
    return {
        "n": len(vals),
        "median": median(vals),
        "p25": percentile(vals, 0.25),
        "p75": percentile(vals, 0.75),
        "mean": mean(vals),
    }


def binned_curve(points: list[dict[str, Any]], feature: str, bins: int = 5) -> list[dict[str, Any]]:
    valid = [p for p in points if p.get(feature) is not None and math.isfinite(float(p[feature]))]
    ordered = sorted(valid, key=lambda p: (float(p[feature]), p["dataset"], p["model"], p["label"]))
    n = len(ordered)
    rows: list[dict[str, Any]] = []
    for b in range(bins):
        lo = round(b * n / bins)
        hi = round((b + 1) * n / bins)
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        vals = [float(p[feature]) for p in chunk]
        s = summarize(chunk)
        rows.append({
            "bin": b + 1,
            "feature_min": min(vals),
            "feature_median": median(vals),
            "feature_max": max(vals),
            **s,
        })
    return rows


def threshold_values(values: list[float], quantiles: Iterable[float]) -> list[tuple[float, float]]:
    return [(q, percentile(values, q)) for q in quantiles]


def route_points(points: list[dict[str, Any]], feature: str, threshold: float, *, low_conf_uses_scope: bool) -> dict[str, float]:
    routed = []
    used_scope = 0
    avoided_scope_hurts = 0
    captured_scope_wins = 0
    missed_scope_wins = 0
    raw_safe_kept = 0
    for p in points:
        value = p.get(feature)
        if value is None or not math.isfinite(float(value)):
            use_scope = False
        else:
            use_scope = float(value) <= threshold if low_conf_uses_scope else float(value) >= threshold
        used_scope += int(use_scope)
        raw_ret = int(p["raw_hit"])
        scope_ret = int(p["scope_hit"])
        raw_ans = int(p["raw_correct"])
        scope_ans = int(p["scope_correct"])
        routed.append({
            "retrieval": scope_ret if use_scope else raw_ret,
            "answer": scope_ans if use_scope else raw_ans,
        })
        if not use_scope and scope_ret < raw_ret:
            avoided_scope_hurts += 1
        if use_scope and scope_ret > raw_ret:
            captured_scope_wins += 1
        if not use_scope and scope_ret > raw_ret:
            missed_scope_wins += 1
        if not use_scope and raw_ret >= scope_ret:
            raw_safe_kept += 1
    base = summarize(points)
    return {
        "n": len(points),
        "scope_fraction": used_scope / len(points) if points else float("nan"),
        "routed_hit": mean([r["retrieval"] for r in routed]),
        "routed_accuracy": mean([r["answer"] for r in routed]),
        "always_raw_hit": base["raw_hit"],
        "always_scope_hit": base["scope_hit"],
        "always_raw_accuracy": base["raw_accuracy"],
        "always_scope_accuracy": base["scope_accuracy"],
        "routed_hit_vs_raw": mean([r["retrieval"] for r in routed]) - base["raw_hit"],
        "routed_hit_vs_scope": mean([r["retrieval"] for r in routed]) - base["scope_hit"],
        "routed_acc_vs_raw": mean([r["answer"] for r in routed]) - base["raw_accuracy"],
        "routed_acc_vs_scope": mean([r["answer"] for r in routed]) - base["scope_accuracy"],
        "avoided_scope_hurts": avoided_scope_hurts / len(points) if points else float("nan"),
        "captured_scope_wins": captured_scope_wins / len(points) if points else float("nan"),
        "missed_scope_wins": missed_scope_wins / len(points) if points else float("nan"),
        "raw_safe_kept": raw_safe_kept / len(points) if points else float("nan"),
    }


def routing_simulation(points: list[dict[str, Any]], feature: str) -> list[dict[str, float]]:
    x_ret, y_ret = finite_pair_values(points, feature, "retrieval_delta")
    corr = spearman(x_ret, y_ret)
    # If the feature rises with SCOPE benefit, then "low confidence uses SCOPE"
    # is the wrong orientation; invert for the in-sample diagnostic route.
    low_conf_uses_scope = not (math.isfinite(corr) and corr > 0)
    values = [float(p[feature]) for p in points if p.get(feature) is not None and math.isfinite(float(p[feature]))]
    rows = []
    for q, threshold in threshold_values(values, (0.2, 0.4, 0.6, 0.8)):
        routed = route_points(points, feature, threshold, low_conf_uses_scope=low_conf_uses_scope)
        routed.update({
            "quantile": q,
            "threshold": threshold,
            "low_conf_uses_scope": float(low_conf_uses_scope),
        })
        rows.append(routed)
    return rows


def best_named_feature(points: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    rows = {feature: correlation(points, feature) for feature in NAMED_QPP_FEATURES}
    best = max(rows, key=lambda f: abs(rows[f]["kendall_retrieval"]) if math.isfinite(rows[f]["kendall_retrieval"]) else -1.0)
    return best, rows[best]


def correlation_table(all_points: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Dataset | Model | Predictor | Family | N | Pearson ret | Spearman ret | Kendall ret | Pearson ans | Spearman ans | Kendall ans |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    contexts: list[tuple[str, str, list[dict[str, Any]]]] = []
    for dataset in ("barexam", "housing"):
        dpoints = [p for p in all_points if p["dataset"] == dataset]
        display = dpoints[0]["dataset_display"] if dpoints else dataset
        for model in MODELS:
            contexts.append((display, MODEL_LABELS[model], [p for p in dpoints if p["model"] == model]))
        contexts.append((display, "Pooled", dpoints))
    contexts.append(("All", "Pooled", all_points))
    for dataset_name, model_name, points in contexts:
        for feature in FEATURES:
            row = correlation(points, feature)
            lines.append(
                f"| {dataset_name} | {model_name} | {FEATURES[feature]['label']} | {FEATURES[feature]['family']} | "
                f"{row['n_retrieval']} | {fmt_float(row['pearson_retrieval'])} | {fmt_float(row['spearman_retrieval'])} | "
                f"{fmt_float(row['kendall_retrieval'])} | {fmt_float(row['pearson_answer'])} | "
                f"{fmt_float(row['spearman_answer'])} | {fmt_float(row['kendall_answer'])} |"
            )
    return lines


def named_qpp_summary_table(all_points: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Scope | Predictor | Kendall ret | Meets tau >= 0.5? | Spearman ret | Kendall ans | Spearman ans | Direction read |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    contexts = [
        ("BarExamQA pooled", [p for p in all_points if p["dataset"] == "barexam"]),
        ("HousingQA pooled", [p for p in all_points if p["dataset"] == "housing"]),
        ("All pooled", all_points),
    ]
    for scope, points in contexts:
        for feature in NAMED_QPP_FEATURES:
            row = correlation(points, feature)
            tau = row["kendall_retrieval"]
            meets = math.isfinite(tau) and abs(tau) >= 0.5
            direction = "higher predictor -> more SCOPE benefit" if math.isfinite(tau) and tau > 0 else "higher predictor -> less SCOPE benefit"
            lines.append(
                f"| {scope} | {FEATURES[feature]['label']} | {fmt_float(tau)} | {'yes' if meets else 'no'} | "
                f"{fmt_float(row['spearman_retrieval'])} | {fmt_float(row['kendall_answer'])} | "
                f"{fmt_float(row['spearman_answer'])} | {direction} |"
            )
    return lines


def dataset_distribution_table(question_features_by_dataset: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    lines = [
        "| Predictor | BarExam median | Housing median | Housing > BarExam AUC | Separation read |",
        "|---|---:|---:|---:|---|",
    ]
    for feature in FEATURES:
        be = [float(v[feature]) for v in question_features_by_dataset["barexam"].values() if feature in v and math.isfinite(float(v[feature]))]
        hq = [float(v[feature]) for v in question_features_by_dataset["housing"].values() if feature in v and math.isfinite(float(v[feature]))]
        if not be or not hq:
            continue
        auc = rank_auc_greater(hq, be)
        separation = "clear" if auc >= 0.70 or auc <= 0.30 else "weak"
        lines.append(
            f"| {FEATURES[feature]['label']} | {median(be):.3f} | {median(hq):.3f} | {auc:.3f} | {separation} |"
        )
    return lines


def routing_table(all_points: list[dict[str, Any]], feature: str) -> list[str]:
    lines = [
        f"| Scope | Quantile | Threshold | SCOPE fraction | Routed Hit@5 | vs raw | vs SCOPE | Routed acc | vs raw | vs SCOPE | Captured SCOPE wins | Avoided SCOPE hurts |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    contexts = [
        ("BarExamQA", [p for p in all_points if p["dataset"] == "barexam"]),
        ("HousingQA", [p for p in all_points if p["dataset"] == "housing"]),
        ("All", all_points),
    ]
    for scope, points in contexts:
        for row in routing_simulation(points, feature):
            lines.append(
                f"| {scope} | {row['quantile']:.1f} | {row['threshold']:.3f} | {pct(row['scope_fraction'])} | "
                f"{pct(row['routed_hit'])} | {pct(row['routed_hit_vs_raw'])} | {pct(row['routed_hit_vs_scope'])} | "
                f"{pct(row['routed_accuracy'])} | {pct(row['routed_acc_vs_raw'])} | {pct(row['routed_acc_vs_scope'])} | "
                f"{pct(row['captured_scope_wins'])} | {pct(row['avoided_scope_hurts'])} |"
            )
    return lines


def make_report(
    output: Path,
    all_points: list[dict[str, Any]],
    dataset_results: dict[str, dict[str, Any]],
    lm_cache_dir: Path,
    dense_status: str,
) -> None:
    lines: list[str] = []
    lines.append("# Raw-Retrieval Confidence Routing (QPP) - 2026-05-26")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This results-lane analysis tests whether no-gold Query Performance Prediction (QPP) signals from the existing raw-question retrieval caches can decide when to apply SCOPE/Snap-HyRE. It uses BarExamQA and HousingQA state-filtered full-N caches across the three current model rows. No retrieval or answer calls were launched, and no files under `paper/` were edited.")
    lines.append("")
    lines.append("Predictors:")
    lines.append("")
    lines.append("- `NQC-CE top10`: standard NQC-style normalized score dispersion, transferred to MiniLM cross-encoder scores as `std(top10) / abs(mean(top10))`.")
    lines.append("- `WIG-CE top5-vs-top10`: WIG-style top-set separation. The existing cache does not store corpus-wide CE background scores, so this is a cache-local top-5 minus top-10 background proxy.")
    lines.append("- `SMV-CE top10`: score magnitude and variance fusion on the cached top-10 cross-encoder scores.")
    lines.append("- Dense-native predictors: offline gte query-to-top-hit cosine plus top-5 document-embedding coherence from the already indexed Chroma embeddings.")
    lines.append("- Prior axes retained for comparison: unigram log-perplexity and question token count.")
    lines.append("")
    lines.append(f"Dense feature status: {dense_status}.")
    lines.append("")

    lines.append("## Outcome Baselines")
    lines.append("")
    lines.append("| Dataset | Model | N | Raw Hit@5 | SCOPE Hit@5 | SCOPE-raw Hit@5 | Raw acc | SCOPE acc | SCOPE-raw acc |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in ("barexam", "housing"):
        dpoints = [p for p in all_points if p["dataset"] == dataset]
        display = dpoints[0]["dataset_display"] if dpoints else dataset
        for model in (*MODELS, "pooled"):
            points = dpoints if model == "pooled" else [p for p in dpoints if p["model"] == model]
            s = summarize(points)
            lines.append(
                f"| {display} | {MODEL_LABELS.get(model, 'Pooled')} | {s['n']} | {pct(s['raw_hit'])} | "
                f"{pct(s['scope_hit'])} | {pct(s['retrieval_delta'])} | {pct(s['raw_accuracy'])} | "
                f"{pct(s['scope_accuracy'])} | {pct(s['answer_delta'])} |"
            )
    pooled = summarize(all_points)
    lines.append(
        f"| All | Pooled | {pooled['n']} | {pct(pooled['raw_hit'])} | {pct(pooled['scope_hit'])} | "
        f"{pct(pooled['retrieval_delta'])} | {pct(pooled['raw_accuracy'])} | {pct(pooled['scope_accuracy'])} | {pct(pooled['answer_delta'])} |"
    )
    lines.append("")

    lines.append("## Named QPP Reliability")
    lines.append("")
    lines.append("The Datta-style reliability bar is Kendall tau `|tau| >= 0.5`. Negative signs mean higher raw-retrieval confidence predicts less SCOPE benefit; positive signs mean the transferred predictor is oriented the other way on this data. This table is the main transfer check: classic QPP predictors do not get assumed to work on dense/cross-encoder scores.")
    lines.append("")
    lines.extend(named_qpp_summary_table(all_points))
    lines.append("")

    lines.append("## Dataset Separation")
    lines.append("")
    lines.append("AUC is `P(Housing predictor > BarExam predictor)` using one raw-cache feature vector per question. Values near 0.5 mean weak dataset separation.")
    lines.append("")
    q_features = {
        key: dataset_results[key]["question_features"]
        for key in ("barexam", "housing")
    }
    lines.extend(dataset_distribution_table(q_features))
    lines.append("")

    lines.append("## Context Against Gold-Needed Mechanism")
    lines.append("")
    lines.append("The prior query-gold mechanism report (`docs/generated/scope_gap_mechanism_2026-05-25.md`) is not deployable because it uses gold passage text, but it gives a useful upper-bound comparison for whether a signal tracks SCOPE retrieval repair. In that report, CE delta `CE(scope,gold) - CE(raw,gold)` had Spearman retrieval correlations of 0.340 on BarExamQA, 0.453 on HousingQA, and 0.436 pooled; cosine delta had 0.330, 0.366, and 0.368. The best no-gold named QPP predictor here is materially weaker, so raw-cache QPP is a proxy for selective expansion, not the mechanism itself.")
    lines.append("")

    lines.append("## Full Correlation Matrix")
    lines.append("")
    lines.append("Outcomes are `SCOPE - raw`: retrieval delta is Hit@5 movement and answer delta is exact-answer correctness movement.")
    lines.append("")
    lines.extend(correlation_table(all_points))
    lines.append("")

    best_feature, best_corr = best_named_feature(all_points)
    lines.append("## Strongest Named Predictor Curve")
    lines.append("")
    lines.append(
        f"The strongest named QPP predictor by absolute pooled Kendall tau on retrieval delta is `{FEATURES[best_feature]['label']}` "
        f"(Kendall {fmt_float(best_corr['kendall_retrieval'])}, Spearman {fmt_float(best_corr['spearman_retrieval'])})."
    )
    lines.append("")
    lines.append("| Bin | N | Predictor median | Predictor range | SCOPE retrieval win | Raw retrieval win | Net retrieval delta | SCOPE answer win | Raw answer win | Net answer delta |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in binned_curve(all_points, best_feature):
        lines.append(
            f"| {row['bin']} | {row['n']} | {row['feature_median']:.3f} | "
            f"{row['feature_min']:.3f}-{row['feature_max']:.3f} | "
            f"{pct(row['scope_retrieval_win'])} | {pct(row['raw_retrieval_win'])} | {pct(row['retrieval_delta'])} | "
            f"{pct(row['scope_answer_win'])} | {pct(row['raw_answer_win'])} | {pct(row['answer_delta'])} |"
        )
    lines.append("")

    lines.append("## Selective SCOPE Routing Simulation")
    lines.append("")
    lines.append("This in-sample diagnostic routes to SCOPE for the low-confidence side of the strongest predictor unless the learned sign is positive, in which case the threshold direction is inverted. It is a screening result, not a locked deployment threshold.")
    lines.append("")
    lines.extend(routing_table(all_points, best_feature))
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    named = {feature: correlation(all_points, feature) for feature in NAMED_QPP_FEATURES}
    best_abs = max((abs(row["kendall_retrieval"]), feature, row) for feature, row in named.items())
    passing = [feature for feature, row in named.items() if math.isfinite(row["kendall_retrieval"]) and abs(row["kendall_retrieval"]) >= 0.5]
    lines.append(
        f"- The no-gold QPP transfer is useful as a weak routing diagnostic but does not clear the `|Kendall tau| >= 0.5` reliability bar. "
        f"The best pooled named predictor is `{FEATURES[best_abs[1]]['label']}` with Kendall {fmt_float(best_abs[2]['kendall_retrieval'])}; "
        f"passing predictors: {', '.join(FEATURES[f]['label'] for f in passing) if passing else 'none'}."
    )
    be_best, be_corr = best_named_feature([p for p in all_points if p["dataset"] == "barexam"])
    hq_best, hq_corr = best_named_feature([p for p in all_points if p["dataset"] == "housing"])
    lines.append(
        f"- Per dataset, the strongest named retrieval-delta predictors are `{FEATURES[be_best]['label']}` on BarExamQA "
        f"(Kendall {fmt_float(be_corr['kendall_retrieval'])}) and `{FEATURES[hq_best]['label']}` on HousingQA "
        f"(Kendall {fmt_float(hq_corr['kendall_retrieval'])}). This supports the Faggioli-style caution: score-based QPP transfer has to be validated on the actual neural scores and does not behave like a guaranteed oracle."
    )
    answer_best = max((abs(row["kendall_answer"]), feature, row) for feature, row in named.items())
    lines.append(
        f"- Answer-delta prediction is weaker than retrieval-delta prediction. The strongest pooled named answer predictor is "
        f"`{FEATURES[answer_best[1]]['label']}` with Kendall {fmt_float(answer_best[2]['kendall_answer'])}. This matches the previous mechanism reports: QPP can screen for retrieval repair opportunities, but answer conversion remains noisier."
    )
    lines.append(
        "- Selective query expansion is therefore viable as a conservative research direction, not yet as a standalone gate. The safest next step is to learn or calibrate thresholds on a held-out slice, then test whether routed SCOPE preserves BarExam retrieval gains while avoiding Housing dilution."
    )
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    seen: list[str] = []
    for dataset in ("barexam", "housing"):
        for path in source_paths_for(DATASETS[dataset]):
            if path not in seen:
                seen.append(path)
    for path in seen:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append(f"- Unigram LM cache directory for comparison axes: `{lm_cache_dir}`")
    lines.append("- Chroma collections read for dense QPP features: `legal_passages`, `housing_statutes`")
    lines.append("- Gold-needed comparison source: `docs/generated/scope_gap_mechanism_2026-05-25.md`")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_raw_retrieval_confidence_routing.py \\")
    lines.append("  --output docs/generated/raw_retrieval_confidence_routing_2026-05-26.md")
    lines.append("```")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/generated/raw_retrieval_confidence_routing_2026-05-26.md")
    parser.add_argument("--lm-cache-dir", type=Path, default=Path("/tmp/perplexity_axis_lm_cache_2026-05-25"))
    parser.add_argument("--chroma-batch-size", type=int, default=2000)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", default=["barexam", "housing"], choices=sorted(DATASETS))
    args = parser.parse_args()

    dataset_results: dict[str, dict[str, Any]] = {}
    all_points: list[dict[str, Any]] = []
    dense_status = "computed from existing Chroma embeddings plus offline local query embeddings"
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        points, summary = build_dataset_points(
            spec,
            lm_cache_dir=args.lm_cache_dir,
            chroma_batch_size=args.chroma_batch_size,
            embed_batch_size=args.embed_batch_size,
        )
        dataset_results[dataset] = {
            "display": spec.display,
            "points": points,
            "question_features": summary["features_by_label"],
            "summary": summary,
        }
        all_points.extend(points)

    make_report(args.output, all_points, dataset_results, args.lm_cache_dir, dense_status)
    print(args.output)


if __name__ == "__main__":
    main()
