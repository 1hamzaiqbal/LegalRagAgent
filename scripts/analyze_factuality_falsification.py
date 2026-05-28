#!/usr/bin/env python3
"""Analyze factuality-judge scores against expansion failure mechanisms."""
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_affinity_margin_oncache import best_non_gold_ce  # noqa: E402
from analyze_perplexity_axis import (  # noqa: E402
    DATASETS as LEGAL_DATASETS,
    build_or_load_lm,
    hit_at_5,
    load_by_label,
    question_scores,
)
from analyze_scope_gap_mechanism import fetch_docs_by_idx, load_questions_raw_text  # noqa: E402
from build_factuality_judge_cache import MODEL, dataset_specs, generation_passage  # noqa: E402
from rag_utils import get_cross_encoder  # noqa: E402


DEFAULT_JUDGE = ROOT / "docs/generated/factuality_judge_q200_2026-05-28.jsonl"
DEFAULT_FEATURES = ROOT / "docs/generated/factuality_feature_points_q200_2026-05-28.jsonl"
DEFAULT_REPORT = ROOT / "docs/generated/factuality_falsification_2026-05-28.md"
BEIR_POINTS = Path("/tmp/beir_phase1_verification_2026-05-26_points.jsonl")
LEGAL_SCOPE_POINTS = Path("/tmp/affinity_margin_oncache_2026-05-26_points.jsonl")
LM_CACHE_DIR = Path("/tmp/perplexity_axis_lm_cache_2026-05-25")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def fmt(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def pct(value: Any) -> str:
    if not finite(value):
        return "--"
    return f"{100.0 * float(value):.1f}%"


def mean(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return float("nan")
    x = [x for x, _ in pairs]
    y = [y for _, y in pairs]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    out = spearmanr(x, y, nan_policy="omit").statistic
    return float(out) if finite(out) else float("nan")


def safe_auc(y: np.ndarray, proba: np.ndarray) -> float:
    if len(set(int(v) for v in y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def standardize(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    x_raw = np.asarray([[float(row[f]) for f in features] for row in rows], dtype=np.float64)
    mu = x_raw.mean(axis=0)
    sd = x_raw.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (x_raw - mu) / sd


def fit_logistic(points: list[dict[str, Any]], features: list[str], target: str) -> dict[str, Any]:
    rows = [
        p for p in points
        if finite(p.get(target)) and all(finite(p.get(feature)) for feature in features)
    ]
    if len(rows) < 10:
        return {"n": len(rows), "failures": 0, "auc": float("nan"), "pseudo_r2": float("nan"), "coefficients": {}}
    y = np.asarray([int(float(row[target]) > 0.0) for row in rows], dtype=np.int64)
    failures = int(y.sum())
    if failures == 0 or failures == len(y):
        return {"n": len(rows), "failures": failures, "auc": float("nan"), "pseudo_r2": float("nan"), "coefficients": {}}
    x = standardize(rows, features)
    model = LogisticRegression(max_iter=2000).fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    loss = float(log_loss(y, proba, labels=[0, 1]))
    null_p = min(max(float(y.mean()), 1e-9), 1.0 - 1e-9)
    null_loss = float(log_loss(y, np.full_like(y, null_p, dtype=np.float64), labels=[0, 1]))
    return {
        "n": len(rows),
        "failures": failures,
        "auc": safe_auc(y, proba),
        "log_loss": loss,
        "pseudo_r2": 1.0 - loss / null_loss if null_loss else float("nan"),
        "coefficients": {feature: float(coef) for feature, coef in zip(features, model.coef_[0])},
    }


def fit_with_partials(points: list[dict[str, Any]], features: list[str], target: str) -> dict[str, Any]:
    full = fit_logistic(points, features, target)
    partial: dict[str, float] = {}
    if not finite(full.get("pseudo_r2")):
        full["partial_r2"] = partial
        return full
    for feature in features:
        reduced = [f for f in features if f != feature]
        if not reduced:
            partial[feature] = float(full["pseudo_r2"])
            continue
        row = fit_logistic(points, reduced, target)
        partial[feature] = max(0.0, float(full["pseudo_r2"]) - float(row.get("pseudo_r2", 0.0))) if finite(row.get("pseudo_r2")) else float("nan")
    full["partial_r2"] = partial
    return full


def normalize_beir_point(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": f"{row['dataset']}/{row['expansion']}/{row['label']}",
        "dataset": row["dataset"],
        "dataset_display": row["dataset_display"],
        "model": row.get("model", MODEL),
        "label": row["label"],
        "idx": row.get("idx", ""),
        "expansion": row["expansion"],
        "expansion_display": row.get("expansion_display", row["expansion"]),
        "gold_count": int(row.get("gold_count") or 0),
        "raw_hit5": int(row.get("raw_hit5") or 0),
        "exp_hit5": int(row.get("exp_hit5") or 0),
        "retrieval_delta": int(row.get("retrieval_delta") or 0),
        "log_perplexity": float(row.get("log_perplexity")),
        "oov_rate": float(row.get("oov_rate")),
        "question_tokens": float(row.get("token_count") or 0.0),
        "ce_margin_raw": float(row.get("ce_margin_raw")),
        "ce_exp_gold": float(row.get("ce_exp_gold")),
        "ce_delta_margin": float(row.get("ce_delta_margin")),
        "ce_exp_gold_id": str(row.get("ce_exp_gold_id") or ""),
    }


def normalize_legal_scope_point(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": f"{row['dataset']}/scope/{row['label']}",
        "dataset": row["dataset"],
        "dataset_display": row["dataset_display"],
        "model": row.get("model", MODEL),
        "label": row["label"],
        "idx": row.get("label", ""),
        "expansion": "scope",
        "expansion_display": "SCOPE",
        "gold_count": int(row.get("multi_gold_count") or len(row.get("gold_ids") or [])),
        "raw_hit5": int(row.get("raw_hit") or 0),
        "exp_hit5": int(row.get("scope_hit") or 0),
        "retrieval_delta": int(row.get("retrieval_delta") or 0),
        "log_perplexity": float(row.get("log_perplexity")),
        "oov_rate": float(row.get("oov_rate")),
        "question_tokens": float(row.get("question_tokens") or 0.0),
        "ce_margin_raw": float(row.get("ce_margin_raw")),
        "ce_exp_gold": float(row.get("ce_scope_gold")),
        "ce_delta_margin": float(row.get("ce_delta_margin")),
        "ce_exp_gold_id": str(row.get("ce_scope_gold_id") or ""),
    }


def load_existing_features(needed: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in read_jsonl(BEIR_POINTS):
        if row.get("model") != MODEL:
            continue
        key = (str(row.get("dataset")), str(row.get("label")), str(row.get("expansion")))
        if key in needed:
            out[key] = normalize_beir_point(row)
    for row in read_jsonl(LEGAL_SCOPE_POINTS):
        if row.get("model") != MODEL:
            continue
        key = (str(row.get("dataset")), str(row.get("label")), "scope")
        if key in needed:
            out[key] = normalize_legal_scope_point(row)
    return out


def truncate_for_ce(text: str) -> str:
    max_chars = int(os.getenv("CROSS_ENCODER_MAX_CHARS", "4096") or "4096")
    text = str(text or "")
    return text[:max_chars] if max_chars and len(text) > max_chars else text


def score_best_gold_ce(
    *,
    items: list[tuple[str, str, list[str]]],
    gold_docs: dict[str, dict[str, Any]],
    batch_size: int,
) -> dict[str, tuple[float, str]]:
    ce = get_cross_encoder()
    pairs: list[tuple[str, str]] = []
    meta: list[tuple[str, str]] = []
    for label, query, gold_ids in items:
        for gid in gold_ids:
            pairs.append((truncate_for_ce(query), truncate_for_ce(gold_docs[gid]["text"])))
            meta.append((label, gid))
    out: dict[str, tuple[float, str]] = {}
    for start in range(0, len(pairs), 2048):
        end = min(start + 2048, len(pairs))
        print(f"[ce legal hyde] {end}/{len(pairs)}", flush=True)
        scores = ce.predict(pairs[start:end], batch_size=batch_size, show_progress_bar=False)
        for (label, gid), score in zip(meta[start:end], scores):
            score_f = float(score)
            current = out.get(label)
            if current is None or score_f > current[0]:
                out[label] = (score_f, gid)
    return out


def build_legal_hyde_features(
    *,
    needed: set[tuple[str, str, str]],
    existing: dict[tuple[str, str, str], dict[str, Any]],
    ce_batch_size: int,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    specs = dataset_specs()
    for dataset in ("barexam", "housing"):
        labels = sorted(label for d, label, expansion in needed if d == dataset and expansion == "hyde")
        if not labels:
            continue
        print(f"[legal hyde] dataset={dataset} labels={len(labels)}", flush=True)
        legal_spec = LEGAL_DATASETS[dataset]
        q_scores = question_scores(legal_spec, build_or_load_lm(legal_spec, LM_CACHE_DIR, 20000))
        raw_text = load_questions_raw_text(dataset)
        raw_cache = load_by_label(legal_spec.raw_cache)
        judge_spec = specs[dataset]
        exp = judge_spec.expansions["hyde"]
        exp_cache = load_by_label(exp.retrieval)
        gen_cache = load_by_label(exp.generation)
        scope_rows = {
            str(row.get("label")): row
            for row in read_jsonl(LEGAL_SCOPE_POINTS)
            if row.get("dataset") == dataset and row.get("model") == MODEL
        }
        missing_scope_raw = [label for label in labels if label not in scope_rows]
        if missing_scope_raw:
            raise RuntimeError(f"{dataset}: missing scope raw CE proxy for {missing_scope_raw[:5]} n={len(missing_scope_raw)}")

        all_gold_ids = sorted({str(gid) for label in labels for gid in q_scores[label]["gold_ids"]})
        gold_docs = fetch_docs_by_idx(legal_spec.collection, all_gold_ids)
        items = [
            (label, generation_passage(gen_cache[label]), [str(gid) for gid in q_scores[label]["gold_ids"]])
            for label in labels
        ]
        exp_gold = score_best_gold_ce(items=items, gold_docs=gold_docs, batch_size=ce_batch_size)
        for label in labels:
            score = q_scores[label]
            gold_ids = [str(gid) for gid in score["gold_ids"] if str(gid)]
            raw_row = raw_cache[label]
            exp_row = exp_cache[label]
            raw_hit = hit_at_5(raw_row.get("retrieved_ids") or [], gold_ids)
            exp_hit = hit_at_5(exp_row.get("retrieved_ids") or [], gold_ids)
            exp_dist, _ = best_non_gold_ce(exp_row, gold_ids)
            scope_raw = scope_rows[label]
            ce_raw_gold = float(scope_raw["ce_raw_gold"])
            ce_margin_raw = float(scope_raw["ce_margin_raw"])
            ce_exp_gold, ce_exp_gold_id = exp_gold[label]
            ce_margin_exp = ce_exp_gold - exp_dist if finite(exp_dist) else float("nan")
            key = (dataset, label, "hyde")
            existing[key] = {
                "row_id": f"{dataset}/hyde/{label}",
                "dataset": dataset,
                "dataset_display": legal_spec.display,
                "model": MODEL,
                "label": label,
                "idx": label,
                "expansion": "hyde",
                "expansion_display": "HyDE",
                "gold_count": len(gold_ids),
                "raw_hit5": int(raw_hit),
                "exp_hit5": int(exp_hit),
                "retrieval_delta": int(exp_hit) - int(raw_hit),
                "log_perplexity": float(score["log_perplexity"]),
                "oov_rate": float(score["oov_rate"]),
                "question_tokens": float(score["token_count"]),
                "ce_raw_gold": ce_raw_gold,
                "ce_margin_raw": ce_margin_raw,
                "ce_exp_gold": float(ce_exp_gold),
                "ce_exp_gold_id": ce_exp_gold_id,
                "ce_delta_margin": ce_margin_exp - ce_margin_raw if finite(ce_margin_exp) and finite(ce_margin_raw) else float("nan"),
            }
    return existing


def load_judge_scores(judge_path: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], list[dict[str, Any]]]:
    rows = read_jsonl(judge_path)
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("label")), str(row.get("expansion")))
        premise_kind = str(row.get("premise_kind"))
        grouped[key][premise_kind] = row
    return grouped, rows


def build_feature_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    grouped, judge_rows = load_judge_scores(args.judge_cache)
    needed = set(grouped)
    features = load_existing_features(needed)
    missing = needed - set(features)
    if missing:
        legal_hyde_missing = {key for key in missing if key[0] in {"barexam", "housing"} and key[2] == "hyde"}
        other_missing = missing - legal_hyde_missing
        if other_missing:
            raise RuntimeError(f"missing feature points for {sorted(other_missing)[:8]} n={len(other_missing)}")
        build_legal_hyde_features(needed=legal_hyde_missing, existing=features, ce_batch_size=args.ce_batch_size)

    out: list[dict[str, Any]] = []
    for key in sorted(needed):
        if key not in features:
            raise RuntimeError(f"feature build failed for {key}")
        judges = grouped[key]
        if "gold" not in judges or "raw_top3" not in judges:
            raise RuntimeError(f"judge cache missing premise arm for {key}: {sorted(judges)}")
        point = dict(features[key])
        gold = judges["gold"]
        raw_top3 = judges["raw_top3"]
        point.update({
            "factuality_gold_score": float(gold["score"]),
            "factuality_gold_verdict": gold["verdict"],
            "factuality_gold_strategy": gold.get("gold_strategy", ""),
            "factuality_gold_premise_count": int(gold.get("premise_count") or 0),
            "factuality_raw_top3_score": float(raw_top3["score"]),
            "factuality_raw_top3_verdict": raw_top3["verdict"],
            "factuality_raw_top3_premise_count": int(raw_top3.get("premise_count") or 0),
            "target_margin_failure": float(float(point["ce_delta_margin"]) < 0.0) if finite(point.get("ce_delta_margin")) else float("nan"),
            "target_retrieval_hurt": float(int(point["retrieval_delta"]) < 0),
        })
        out.append(point)

    args.features_out.parent.mkdir(parents=True, exist_ok=True)
    with args.features_out.open("w") as f:
        for row in out:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[features] wrote {len(out)} rows to {args.features_out}", flush=True)
    _ = judge_rows
    return out


FEATURE_SETS = [
    ("OOV + logPPL", ["oov_rate", "log_perplexity"]),
    ("Factuality gold", ["factuality_gold_score"]),
    ("Factuality raw-top3", ["factuality_raw_top3_score"]),
    ("Geometry", ["ce_margin_raw", "ce_exp_gold"]),
    ("Gold factuality + geometry", ["factuality_gold_score", "ce_margin_raw", "ce_exp_gold"]),
    ("Raw-top3 factuality + geometry", ["factuality_raw_top3_score", "ce_margin_raw", "ce_exp_gold"]),
]


TARGETS = [
    ("target_retrieval_hurt", "retrieval hurt"),
    ("target_margin_failure", "deltaM < 0"),
]


def dataset_groups(points: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for dataset in sorted({p["dataset"] for p in points}):
        display = next(p["dataset_display"] for p in points if p["dataset"] == dataset)
        groups.append((dataset, display, [p for p in points if p["dataset"] == dataset]))
    groups.append(("pooled", "Pooled", points))
    return groups


def auc_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, display, group in dataset_groups(points):
        for target, target_display in TARGETS:
            for feature_name, features in FEATURE_SETS:
                fit = fit_logistic(group, features, target)
                rows.append({
                    "dataset": dataset,
                    "dataset_display": display,
                    "target": target_display,
                    "feature_set": feature_name,
                    **fit,
                })
    return rows


def stratified_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, display, group in dataset_groups(points):
        for label, pred in (
            ("high factuality", lambda p: float(p["factuality_gold_score"]) >= 0.75),
            ("low/mid factuality", lambda p: float(p["factuality_gold_score"]) < 0.75),
        ):
            chunk = [p for p in group if pred(p)]
            if not chunk:
                continue
            rows.append({
                "dataset": display,
                "stratum": label,
                "n": len(chunk),
                "mean_factuality": mean(p["factuality_gold_score"] for p in chunk),
                "retrieval_hurt_rate": mean(p["target_retrieval_hurt"] for p in chunk),
                "margin_failure_rate": mean(p["target_margin_failure"] for p in chunk),
                "mean_retrieval_delta": mean(p["retrieval_delta"] for p in chunk),
                "mean_delta_margin": mean(p["ce_delta_margin"] for p in chunk),
                "rho_deltaM_vs_retrieval_delta": spearman(
                    [float(p["ce_delta_margin"]) for p in chunk],
                    [float(p["retrieval_delta"]) for p in chunk],
                ),
            })
    return rows


def verdict(points: list[dict[str, Any]], aucs: list[dict[str, Any]]) -> dict[str, Any]:
    def pooled_auc(feature_set: str, target: str) -> float:
        for row in aucs:
            if row["dataset"] == "pooled" and row["feature_set"] == feature_set and row["target"] == target:
                return float(row["auc"]) if finite(row.get("auc")) else float("nan")
        return float("nan")

    factuality = pooled_auc("Factuality gold", "retrieval hurt")
    factuality_proxy = pooled_auc("Factuality raw-top3", "retrieval hurt")
    old = pooled_auc("OOV + logPPL", "retrieval hurt")
    geometry = pooled_auc("Geometry", "retrieval hurt")
    joint = pooled_auc("Gold factuality + geometry", "retrieval hurt")
    if finite(factuality) and finite(geometry) and factuality < 0.70 and geometry > 0.85:
        headline = "supported"
    elif finite(factuality) and finite(geometry) and abs(factuality - geometry) <= 0.05:
        headline = "killed"
    else:
        headline = "mixed"
    high = [p for p in points if float(p["factuality_gold_score"]) >= 0.75]
    high_hurt = mean(p["target_retrieval_hurt"] for p in high)
    high_rho = spearman([float(p["ce_delta_margin"]) for p in high], [float(p["retrieval_delta"]) for p in high])
    return {
        "headline": headline,
        "old_auc": old,
        "factuality_auc": factuality,
        "factuality_proxy_auc": factuality_proxy,
        "geometry_auc": geometry,
        "joint_auc": joint,
        "high_n": len(high),
        "high_hurt": high_hurt,
        "high_rho": high_rho,
    }


def score_distribution(judge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in judge_rows:
        key = (str(row.get("premise_kind")), str(row.get("verdict")))
        counts[key][str(row.get("dataset"))] += 1
        counts[key]["all"] += 1
    out = []
    for (premise, verdict_), by_dataset in sorted(counts.items()):
        row = {"premise_kind": premise, "verdict": verdict_, **dict(by_dataset)}
        out.append(row)
    return out


def coefficient_text(row: dict[str, Any]) -> str:
    coeffs = row.get("coefficients", {})
    partial = row.get("partial_r2", {})
    bits = []
    for name, coef in coeffs.items():
        p = partial.get(name, float("nan"))
        bits.append(f"`{name}` beta={fmt(coef)}, partial-R2={fmt(p)}")
    return "; ".join(bits)


def write_report(args: argparse.Namespace, points: list[dict[str, Any]]) -> None:
    judge_rows = read_jsonl(args.judge_cache)
    aucs = auc_rows(points)
    strata = stratified_rows(points)
    read = verdict(points, aucs)
    partial_rows = []
    for dataset, display, group in dataset_groups(points):
        for target, target_display in TARGETS:
            partial_rows.append({
                "dataset": display,
                "target": target_display,
                **fit_with_partials(group, ["factuality_gold_score", "ce_margin_raw", "ce_exp_gold"], target),
            })

    lines: list[str] = []
    lines.append("# Factuality Falsification - 2026-05-28")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"Phase: `{args.phase}`. This is a results-lane analysis over existing generation and retrieval caches plus LLM-as-judge factuality records. No files under `paper/` were edited.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("- `retrieval hurt` is `1[generated-passage Hit@5 < raw-question Hit@5]`; this is the headline target because it is less circular than the margin target.")
    lines.append("- `deltaM < 0` is the continuity target from the affinity-margin analysis.")
    lines.append("- Primary factuality is gold-grounded corpus-supportedness; the no-gold proxy uses the top-3 raw-retrieved passages.")
    lines.append("- Multi-gold rows with at most the configured cap are judged against the full gold passage set in one prompt. High-cardinality BEIR qrel rows use the CE-best gold passage proxy and are flagged by `gold_strategy` in the judge JSONL.")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        f"Headline verdict: **{read['headline']}**. On pooled `retrieval hurt`, AUC is "
        f"{fmt(read['old_auc'])} for OOV/logPPL, {fmt(read['factuality_auc'])} for gold factuality, "
        f"{fmt(read['factuality_proxy_auc'])} for raw-top3 factuality, {fmt(read['geometry_auc'])} for geometry, "
        f"and {fmt(read['joint_auc'])} for gold factuality plus geometry."
    )
    lines.append("")
    lines.append(
        f"High-gold-factuality rows still have retrieval hurt rate {pct(read['high_hurt'])} "
        f"over N={read['high_n']}; within that stratum, Spearman rho(deltaM, retrieval delta) is {fmt(read['high_rho'])}."
    )
    lines.append("")
    lines.append("## AUC Table")
    lines.append("")
    lines.append("| Dataset | Target | Feature set | N | Failures | AUC | Pseudo-R2 |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in aucs:
        lines.append(
            f"| {row['dataset_display']} | {row['target']} | {row['feature_set']} | {row['n']} | "
            f"{row['failures']} | {fmt(row['auc'])} | {fmt(row['pseudo_r2'])} |"
        )
    lines.append("")
    lines.append("## Stratified Analysis")
    lines.append("")
    lines.append("| Dataset | Stratum | N | Mean factuality | Retrieval hurt | deltaM<0 | Mean retrieval delta | Mean deltaM | rho(deltaM, retrieval delta) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in strata:
        lines.append(
            f"| {row['dataset']} | {row['stratum']} | {row['n']} | {fmt(row['mean_factuality'])} | "
            f"{pct(row['retrieval_hurt_rate'])} | {pct(row['margin_failure_rate'])} | "
            f"{pct(row['mean_retrieval_delta'])} | {fmt(row['mean_delta_margin'])} | {fmt(row['rho_deltaM_vs_retrieval_delta'])} |"
        )
    lines.append("")
    lines.append("## Joint Coefficients")
    lines.append("")
    lines.append("Standardized logistic coefficients for `{factuality_gold_score, ce_margin_raw, ce_exp_gold}`. Partial-R2 is the drop in pseudo-R2 when that feature is removed.")
    lines.append("")
    lines.append("| Dataset | Target | N | Failures | AUC | Pseudo-R2 | Coefficients |")
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for row in partial_rows:
        lines.append(
            f"| {row['dataset']} | {row['target']} | {row['n']} | {row['failures']} | "
            f"{fmt(row['auc'])} | {fmt(row['pseudo_r2'])} | {coefficient_text(row)} |"
        )
    lines.append("")
    lines.append("## Judge Score Distribution")
    lines.append("")
    dataset_names = sorted({p["dataset"] for p in points})
    lines.append("| Premise | Verdict | All | " + " | ".join(dataset_names) + " |")
    lines.append("|---|---|---:|" + "---:|" * len(dataset_names))
    for row in score_distribution(judge_rows):
        vals = [str(row.get(ds, 0)) for ds in dataset_names]
        lines.append(f"| {row['premise_kind']} | {row['verdict']} | {row.get('all', 0)} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    if finite(read["factuality_auc"]) and finite(read["old_auc"]):
        rel = "does" if read["factuality_auc"] > read["old_auc"] else "does not"
        lines.append(f"- The real factuality signal {rel} beat the old OOV/log-perplexity proxy on the headline retrieval-hurt target.")
    if finite(read["geometry_auc"]) and finite(read["factuality_auc"]):
        if read["geometry_auc"] > read["factuality_auc"] + 0.05:
            lines.append("- Geometry remains the stronger failure predictor on the pooled headline target.")
        elif read["factuality_auc"] >= read["geometry_auc"] - 0.05:
            lines.append("- Factuality is close to geometry on the pooled headline target, so the clean `geometry not hallucination` falsification is weakened.")
    lines.append("- The LLM judge is itself model-biased, but the same prompt and judge model are used across HyDE/SCOPE and premise arms.")
    lines.append("")
    lines.append("## Sources")
    lines.append("")
    lines.append(f"- Judge cache: `{args.judge_cache}`")
    lines.append(f"- Feature points: `{args.features_out}`")
    lines.append(f"- BEIR geometry points: `{BEIR_POINTS}`")
    lines.append(f"- Legal SCOPE geometry points: `{LEGAL_SCOPE_POINTS}`")
    specs = dataset_specs()
    seen: list[str] = []
    for spec in specs.values():
        paths = [spec.raw_cache]
        for exp in spec.expansions.values():
            paths.extend([exp.generation, exp.retrieval])
        for path in paths:
            rel = str(path.relative_to(ROOT))
            if rel not in seen:
                seen.append(rel)
    for path in seen:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    limit_arg = "--limit 0" if args.phase == "full" else "--limit 200"
    lines.append(f"NO_SILENT_FALLBACK=1 OPENROUTER_PROVIDER_ONLY=Cloudflare EVAL_CONCURRENCY=8 uv run python scripts/build_factuality_judge_cache.py {limit_arg} --resume --output {args.judge_cache}")
    lines.append(f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_factuality_falsification.py --phase {args.phase} --judge-cache {args.judge_cache} --features-out {args.features_out} --output {args.output}")
    lines.append("```")
    lines.append("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"[report] wrote {args.output}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["q200", "full"], default="q200")
    parser.add_argument("--judge-cache", type=Path, default=DEFAULT_JUDGE)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--ce-batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reuse_features and args.features_out.exists():
        points = read_jsonl(args.features_out)
        print(f"[features] loaded {len(points)} rows from {args.features_out}", flush=True)
    else:
        points = build_feature_rows(args)
    write_report(args, points)


if __name__ == "__main__":
    main()
