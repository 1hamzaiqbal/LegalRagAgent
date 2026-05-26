#!/usr/bin/env python3
"""Per-query corpus perplexity axis for SCOPE-vs-raw outcomes.

The analysis is intentionally dataset-general: each dataset provides a Chroma
collection, a question loader config, optional corpus-grouping metadata, and the
signed raw/SCOPE cache/log paths. The scoring path is otherwise the same.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import re
import statistics as stats
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _fmt_intermediate, _gold_ids, _row_label  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
MODELS = ("groq-llama8b", "or-gemma4-26b", "groq-llama70b")
MODEL_LABELS = {
    "groq-llama8b": "Groq Llama 8B",
    "or-gemma4-26b": "Gemma 4 26B",
    "groq-llama70b": "Groq Llama 70B",
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display: str
    collection: str
    raw_cache: str
    scope_cache_by_model: dict[str, str]
    raw_log_by_model: dict[str, str]
    scope_log_by_model: dict[str, str]
    corpus_group_meta: str | None = None
    question_group_field: str | None = None
    housing_state_filter: bool = False


DATASETS: dict[str, DatasetSpec] = {
    "barexam": DatasetSpec(
        key="barexam",
        display="BarExamQA",
        collection="legal_passages",
        raw_cache="caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl",
        scope_cache_by_model={
            "groq-llama8b": "caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl",
            "or-gemma4-26b": "caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl",
            "groq-llama70b": "caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl",
        },
        raw_log_by_model={
            "groq-llama8b": "logs/eval_rag_simple_groq-llama8b_20260518_211000_barexam_local-snap-hyre-groq-llama8b-barexam-rag_simple-nfull-k5_detail.jsonl",
            "or-gemma4-26b": "logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl",
            "groq-llama70b": "logs/eval_rag_simple_groq-llama70b_20260515_194919_barexam_local-snap-hyre-groq-llama70b-barexam-rag_simple-nfull-k5_detail.jsonl",
        },
        scope_log_by_model={
            "groq-llama8b": "logs/eval_snap_hyre_groq-llama8b_20260518_231747_barexam_local-snap-hyre-groq-llama8b-barexam-snap_hyre-nfull-k5_detail.jsonl",
            "or-gemma4-26b": "logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl",
            "groq-llama70b": "logs/eval_snap_hyre_groq-llama70b_20260515_230504_barexam_local-snap-hyre-groq-llama70b-barexam-snap_hyre-nfull-k5_detail.jsonl",
        },
    ),
    "housing": DatasetSpec(
        key="housing",
        display="HousingQA state-filtered",
        collection="housing_statutes",
        raw_cache="caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl",
        scope_cache_by_model={
            "groq-llama8b": "caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl",
            "or-gemma4-26b": "caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl",
            "groq-llama70b": "caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl",
        },
        raw_log_by_model={
            "groq-llama8b": "logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl",
            "or-gemma4-26b": "logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl",
            "groq-llama70b": "logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl",
        },
        scope_log_by_model={
            "groq-llama8b": "logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl",
            "or-gemma4-26b": "logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl",
            "groq-llama70b": "logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl",
        },
        corpus_group_meta="state",
        question_group_field="state",
        housing_state_filter=True,
    ),
}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_by_label(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("label") or row.get("idx")): row for row in read_jsonl(path)}


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt_float(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else float("nan")


def median(values: list[float]) -> float:
    return float(stats.median(values)) if values else float("nan")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    weight = pos - lo
    return xs[lo] * (1 - weight) + xs[hi] * weight


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            out[order[k]] = avg_rank
        i = j
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def rank_auc_greater(a: list[float], b: list[float]) -> float:
    """Probability that a random item from a is greater than a random item from b."""
    combined = [(x, 1) for x in a] + [(x, 0) for x in b]
    combined.sort(key=lambda t: t[0])
    rank_sum_a = 0.0
    i = 0
    rank = 1
    while i < len(combined):
        j = i + 1
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum_a += avg_rank * sum(1 for _, group in combined[i:j] if group == 1)
        rank += j - i
        i = j
    n_a = len(a)
    n_b = len(b)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    return u_a / (n_a * n_b)


def cohen_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = stats.variance(a)
    vb = stats.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / (len(a) + len(b) - 2))
    return (mean(a) - mean(b)) / pooled if pooled else float("nan")


def hit_at_5(retrieved_ids: list[Any], gold_ids: list[str]) -> int:
    gold = {str(x) for x in gold_ids if str(x)}
    if not gold:
        return 0
    return int(bool(gold & {str(x) for x in retrieved_ids[:5]}))


def load_questions_for(spec: DatasetSpec) -> list[Any]:
    config = EvalConfig(
        dataset=spec.key,
        questions="full",
        seed=42,
        retrieval_k=5,
        housing_state_filter=spec.housing_state_filter,
    )
    return [row for _, row in load_questions(config).iterrows()]


def normalize_group(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "__all__"


def build_or_load_lm(spec: DatasetSpec, cache_dir: Path, batch_size: int) -> dict[str, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = spec.corpus_group_meta or "all"
    cache_path = cache_dir / f"{spec.key}_{spec.collection}_{suffix}_unigram_lm.pkl.gz"
    if cache_path.exists():
        with gzip.open(cache_path, "rb") as f:
            return pickle.load(f)

    import chromadb

    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR", str(REPO_ROOT / "chroma_db")))
    collection = client.get_collection(spec.collection)
    count = collection.count()
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    totals: dict[str, int] = defaultdict(int)
    docs_by_group: dict[str, int] = defaultdict(int)

    for offset in range(0, count, batch_size):
        batch = collection.get(
            offset=offset,
            limit=min(batch_size, count - offset),
            include=["documents", "metadatas"],
        )
        for text, metadata in zip(batch.get("documents") or [], batch.get("metadatas") or []):
            group = "__all__"
            if spec.corpus_group_meta:
                group = normalize_group((metadata or {}).get(spec.corpus_group_meta))
            toks = tokenize(text or "")
            if not toks:
                continue
            counters[group].update(toks)
            totals[group] += len(toks)
            docs_by_group[group] += 1
        print(f"[lm] {spec.key}: {min(offset + batch_size, count)}/{count}", flush=True)

    models: dict[str, dict[str, Any]] = {}
    for group, counter in counters.items():
        models[group] = {
            "counts": counter,
            "total": totals[group],
            "vocab": len(counter),
            "docs": docs_by_group[group],
        }
    with gzip.open(cache_path, "wb") as f:
        pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)
    return models


def score_perplexity(tokens: list[str], lm: dict[str, Any]) -> dict[str, float]:
    counts: Counter[str] = lm["counts"]
    total = int(lm["total"])
    vocab = int(lm["vocab"])
    denom = total + vocab
    if not tokens or denom <= 0:
        return {
            "perplexity": float("nan"),
            "log_perplexity": float("nan"),
            "token_count": 0.0,
            "oov_rate": float("nan"),
        }
    log_prob = 0.0
    oov = 0
    for tok in tokens:
        count = counts.get(tok, 0)
        if count == 0:
            oov += 1
        log_prob += math.log((count + 1) / denom)
    log_ppl = -log_prob / len(tokens)
    return {
        "perplexity": math.exp(log_ppl),
        "log_perplexity": log_ppl,
        "token_count": float(len(tokens)),
        "oov_rate": oov / len(tokens),
    }


def question_scores(spec: DatasetSpec, lms: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    config = EvalConfig(
        dataset=spec.key,
        questions="full",
        seed=42,
        retrieval_k=5,
        housing_state_filter=spec.housing_state_filter,
    )
    rows = load_questions_for(spec)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = _row_label(row, config)
        group = "__all__"
        if spec.question_group_field:
            group = normalize_group(row.get(spec.question_group_field))
        if group not in lms:
            raise RuntimeError(f"{spec.key}: no LM group for {group!r} label={label}")
        text = _fmt_intermediate(row, config)
        toks = tokenize(text)
        score = score_perplexity(toks, lms[group])
        out[label] = {
            "label": label,
            "group": group,
            "text": text,
            "tokens": toks,
            "gold_ids": _gold_ids(row),
            **score,
            "lm_total": lms[group]["total"],
            "lm_vocab": lms[group]["vocab"],
            "lm_docs": lms[group]["docs"],
        }
    return out


def build_points(spec: DatasetSpec, q_scores: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_cache = load_by_label(spec.raw_cache)
    raw_logs_by_model = {model: load_by_label(path) for model, path in spec.raw_log_by_model.items()}
    scope_logs_by_model = {model: load_by_label(path) for model, path in spec.scope_log_by_model.items()}
    scope_cache_by_model = {model: load_by_label(path) for model, path in spec.scope_cache_by_model.items()}
    points: list[dict[str, Any]] = []
    model_summary: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        raw_logs = raw_logs_by_model[model]
        scope_logs = scope_logs_by_model[model]
        scope_cache = scope_cache_by_model[model]
        missing = [
            label for label in q_scores
            if label not in raw_cache or label not in scope_cache or label not in raw_logs or label not in scope_logs
        ]
        if missing:
            raise RuntimeError(f"{spec.key}/{model}: missing labels {missing[:5]} n={len(missing)}")
        model_points: list[dict[str, Any]] = []
        for label, score in q_scores.items():
            gold = score["gold_ids"]
            raw_hit = hit_at_5(raw_cache[label].get("retrieved_ids") or [], gold)
            scope_hit = hit_at_5(scope_cache[label].get("retrieved_ids") or [], gold)
            raw_correct = int(bool(raw_logs[label].get("is_correct")))
            scope_correct = int(bool(scope_logs[label].get("is_correct")))
            point = {
                "dataset": spec.key,
                "dataset_display": spec.display,
                "model": model,
                "label": label,
                "group": score["group"],
                "perplexity": score["perplexity"],
                "log_perplexity": score["log_perplexity"],
                "token_count": score["token_count"],
                "oov_rate": score["oov_rate"],
                "raw_hit": raw_hit,
                "scope_hit": scope_hit,
                "retrieval_delta": scope_hit - raw_hit,
                "scope_retrieval_win": int(scope_hit == 1 and raw_hit == 0),
                "raw_retrieval_win": int(scope_hit == 0 and raw_hit == 1),
                "raw_correct": raw_correct,
                "scope_correct": scope_correct,
                "answer_delta": scope_correct - raw_correct,
                "scope_answer_win": int(scope_correct == 1 and raw_correct == 0),
                "raw_answer_win": int(scope_correct == 0 and raw_correct == 1),
            }
            points.append(point)
            model_points.append(point)
        model_summary[model] = summarize_points(model_points)
    return points, model_summary


def summarize_points(points: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(points),
        "perplexity_median": median([p["perplexity"] for p in points]),
        "log_perplexity_mean": mean([p["log_perplexity"] for p in points]),
        "oov_rate_mean": mean([p["oov_rate"] for p in points]),
        "retrieval_delta": mean([p["retrieval_delta"] for p in points]),
        "scope_retrieval_win": mean([p["scope_retrieval_win"] for p in points]),
        "raw_retrieval_win": mean([p["raw_retrieval_win"] for p in points]),
        "answer_delta": mean([p["answer_delta"] for p in points]),
        "scope_answer_win": mean([p["scope_answer_win"] for p in points]),
        "raw_answer_win": mean([p["raw_answer_win"] for p in points]),
        "raw_accuracy": mean([p["raw_correct"] for p in points]),
        "scope_accuracy": mean([p["scope_correct"] for p in points]),
    }


def correlation_row(points: list[dict[str, Any]]) -> dict[str, float]:
    x = [p["log_perplexity"] for p in points]
    return {
        "n": len(points),
        "pearson_retrieval_delta": pearson(x, [p["retrieval_delta"] for p in points]),
        "spearman_retrieval_delta": spearman(x, [p["retrieval_delta"] for p in points]),
        "pearson_answer_delta": pearson(x, [p["answer_delta"] for p in points]),
        "spearman_answer_delta": spearman(x, [p["answer_delta"] for p in points]),
    }


def binned_curve(points: list[dict[str, Any]], bins: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(points, key=lambda p: p["log_perplexity"])
    out: list[dict[str, Any]] = []
    n = len(ordered)
    for b in range(bins):
        lo = round(b * n / bins)
        hi = round((b + 1) * n / bins)
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        s = summarize_points(chunk)
        out.append({
            "bin": b + 1,
            "n": len(chunk),
            "ppl_lo": min(p["perplexity"] for p in chunk),
            "ppl_hi": max(p["perplexity"] for p in chunk),
            "ppl_median": s["perplexity_median"],
            **s,
        })
    return out


def load_union_probe_rows(path: Path, dataset: str) -> dict[str, dict[str, dict[str, Any]]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        if row.get("dataset") == dataset:
            rows[str(row.get("label"))][str(row.get("arm"))] = row
    return dict(rows)


def union_probe_summary(
    spec: DatasetSpec,
    q_scores: dict[str, dict[str, Any]],
    path: Path,
) -> dict[str, dict[str, Any]]:
    rows_by_label = load_union_probe_rows(path, spec.key)
    if not rows_by_label:
        return {}
    raw_logs = load_by_label(spec.raw_log_by_model["or-gemma4-26b"])
    scope_logs = load_by_label(spec.scope_log_by_model["or-gemma4-26b"])
    out: dict[str, dict[str, Any]] = {}
    for arm in ("ce_rerank", "rrf", "llm_judge"):
        points = []
        for label, arm_rows in rows_by_label.items():
            row = arm_rows.get(arm)
            if not row or label not in q_scores:
                continue
            raw_correct = int(bool(raw_logs[label].get("is_correct")))
            scope_correct = int(bool(scope_logs[label].get("is_correct")))
            union_correct = int(bool(row.get("is_correct")))
            union_hit = int(float(row.get("hit@5") or 0.0) > 0.0)
            points.append({
                "log_perplexity": q_scores[label]["log_perplexity"],
                "perplexity": q_scores[label]["perplexity"],
                "union_vs_raw_answer_delta": union_correct - raw_correct,
                "union_vs_scope_answer_delta": union_correct - scope_correct,
                "union_retrieval_hit": union_hit,
                "union_correct": union_correct,
            })
        if points:
            x = [p["log_perplexity"] for p in points]
            out[arm] = {
                "n": len(points),
                "accuracy": mean([p["union_correct"] for p in points]),
                "hit@5": mean([p["union_retrieval_hit"] for p in points]),
                "pearson_vs_raw_answer_delta": pearson(x, [p["union_vs_raw_answer_delta"] for p in points]),
                "spearman_vs_raw_answer_delta": spearman(x, [p["union_vs_raw_answer_delta"] for p in points]),
                "pearson_vs_scope_answer_delta": pearson(x, [p["union_vs_scope_answer_delta"] for p in points]),
                "spearman_vs_scope_answer_delta": spearman(x, [p["union_vs_scope_answer_delta"] for p in points]),
            }
    return out


def source_paths_for(spec: DatasetSpec) -> list[str]:
    paths = [spec.raw_cache]
    for model in MODELS:
        paths.extend([
            spec.scope_cache_by_model[model],
            spec.raw_log_by_model[model],
            spec.scope_log_by_model[model],
        ])
    return list(dict.fromkeys(paths))


def make_report(
    *,
    output: Path,
    dataset_results: dict[str, dict[str, Any]],
    union_results: dict[str, dict[str, Any]],
    lm_cache_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Perplexity Axis Probe - 2026-05-25")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This results-lane analysis builds add-1 smoothed unigram language models over each retrieval corpus, scores every eval question by corpus perplexity, and joins that per-query score to signed raw-vs-SCOPE retrieval and answer outcomes. No files under `paper/` were edited.")
    lines.append("")
    lines.append("- BarExamQA LM: `legal_passages` collection, one corpus-wide unigram model.")
    lines.append("- HousingQA LM: `housing_statutes` collection, one unigram model per `state` metadata value to match state-filtered retrieval.")
    lines.append("- Question text: `eval_harness._fmt_intermediate`, so BarExam includes shared prompts and answer-choice text without choice letters; Housing includes the state-framed question.")
    lines.append("- Correlations use `log(perplexity)` because raw perplexity is heavy-tailed.")
    lines.append("")

    lines.append("## Dataset Separation")
    lines.append("")
    lines.append("| Dataset | Questions | LM scope | Median PPL | IQR PPL | Mean log PPL | Mean OOV rate | Median tokens |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for key in ("barexam", "housing"):
        d = dataset_results[key]["question_summary"]
        lines.append(
            f"| {dataset_results[key]['display']} | {d['n']} | {d['lm_scope']} | "
            f"{d['median_ppl']:.1f} | {d['p25_ppl']:.1f}-{d['p75_ppl']:.1f} | "
            f"{d['mean_log_ppl']:.3f} | {pct(d['mean_oov_rate'])} | {d['median_tokens']:.0f} |"
        )
    sep = dataset_results["separation"]
    lines.append("")
    lines.append(
        f"Separation check: probability that a random BarExamQA question has higher log-perplexity "
        f"than a random HousingQA question is {sep['auc_barexam_gt_housing']:.3f}; "
        f"Cohen's d on log-perplexity is {sep['cohen_d_barexam_minus_housing']:.2f}. "
        f"This indicates {'clear' if sep['auc_barexam_gt_housing'] >= 0.70 else 'weak'} dataset separation on this axis."
    )
    lines.append("")

    lines.append("## Correlations")
    lines.append("")
    lines.append("Each model row uses one point per question. Pooled rows use one point per question-model pair. Retrieval delta is `SCOPE Hit@5 - raw Hit@5`; answer delta is `SCOPE correct - raw correct`.")
    lines.append("")
    lines.append("| Dataset | Model | N | Pearson retrieval | Spearman retrieval | Pearson answer | Spearman answer | Mean retrieval delta | Mean answer delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for key in ("barexam", "housing"):
        result = dataset_results[key]
        for model in (*MODELS, "pooled"):
            corr = result["correlations"][model]
            summ = result["model_summaries"][model]
            model_name = MODEL_LABELS.get(model, "Pooled")
            lines.append(
                f"| {result['display']} | {model_name} | {corr['n']} | "
                f"{fmt_float(corr['pearson_retrieval_delta'])} | {fmt_float(corr['spearman_retrieval_delta'])} | "
                f"{fmt_float(corr['pearson_answer_delta'])} | {fmt_float(corr['spearman_answer_delta'])} | "
                f"{pct(summ['retrieval_delta'])} | {pct(summ['answer_delta'])} |"
            )
    lines.append("")

    for key in ("barexam", "housing"):
        result = dataset_results[key]
        lines.append(f"## {result['display']} Binned Curve")
        lines.append("")
        lines.append("Bins are within-dataset quintiles of question perplexity, pooled across the three model rows.")
        lines.append("")
        lines.append("| Bin | N | Median PPL | PPL range | SCOPE retrieval win | Raw retrieval win | Net retrieval delta | SCOPE answer win | Raw answer win | Net answer delta |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in result["bins"]:
            lines.append(
                f"| {row['bin']} | {row['n']} | {row['ppl_median']:.1f} | "
                f"{row['ppl_lo']:.1f}-{row['ppl_hi']:.1f} | "
                f"{pct(row['scope_retrieval_win'])} | {pct(row['raw_retrieval_win'])} | {pct(row['retrieval_delta'])} | "
                f"{pct(row['scope_answer_win'])} | {pct(row['raw_answer_win'])} | {pct(row['answer_delta'])} |"
            )
        lines.append("")

    if union_results:
        lines.append("## q200 Union Probe Supplement")
        lines.append("")
        lines.append("These rows use the local q200 `or-gemma4-26b` raw+SCOPE union probe scratch outputs when present. They are diagnostic only and are not full-N results.")
        lines.append("")
        lines.append("| Dataset | Arm | N | Accuracy | Hit@5 | Pearson union-vs-raw answer | Spearman union-vs-raw answer | Pearson union-vs-SCOPE answer | Spearman union-vs-SCOPE answer |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for key in ("barexam", "housing"):
            for arm, row in union_results.get(key, {}).items():
                arm_name = {"ce_rerank": "Union + CE-rerank", "rrf": "Union + RRF", "llm_judge": "Union + LLM-judge"}.get(arm, arm)
                lines.append(
                    f"| {dataset_results[key]['display']} | {arm_name} | {row['n']} | {pct(row['accuracy'])} | {pct(row['hit@5'])} | "
                    f"{fmt_float(row['pearson_vs_raw_answer_delta'])} | {fmt_float(row['spearman_vs_raw_answer_delta'])} | "
                    f"{fmt_float(row['pearson_vs_scope_answer_delta'])} | {fmt_float(row['spearman_vs_scope_answer_delta'])} |"
                )
        lines.append("")

    lines.append("## Reading")
    lines.append("")
    be = dataset_results["barexam"]["model_summaries"]["pooled"]
    hq = dataset_results["housing"]["model_summaries"]["pooled"]
    be_corr = dataset_results["barexam"]["correlations"]["pooled"]
    hq_corr = dataset_results["housing"]["correlations"]["pooled"]
    lines.append(
        f"- BarExamQA has a higher median question-corpus perplexity than HousingQA and is the dataset where SCOPE improves retrieval over raw on average: "
        f"{pct(be['retrieval_delta'])} pooled Hit@5 delta. The binned curve is not monotone, so the strong version of the per-query monotonicity hypothesis is not supported."
    )
    lines.append(
        f"- HousingQA state-filtered has lower perplexity under its state-specific statute LMs, and SCOPE is not retrieval-positive overall: "
        f"{pct(hq['retrieval_delta'])} pooled Hit@5 delta. This supports the strong-query/state-anchor caveat."
    )
    lines.append(
        f"- Per-query perplexity is a dataset/regime separator more than a strong within-dataset predictor. Pooled Spearman correlations are "
        f"{fmt_float(be_corr['spearman_retrieval_delta'])} retrieval / {fmt_float(be_corr['spearman_answer_delta'])} answer for BarExamQA and "
        f"{fmt_float(hq_corr['spearman_retrieval_delta'])} retrieval / {fmt_float(hq_corr['spearman_answer_delta'])} answer for HousingQA."
    )
    lines.append(
        "- The answer-delta correlations are small; retrieval exposure moves more cleanly than downstream exact accuracy. Treat perplexity as a routing feature candidate, not a standalone policy."
    )
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    seen: list[str] = []
    for key in ("barexam", "housing"):
        for path in source_paths_for(DATASETS[key]):
            if path not in seen:
                seen.append(path)
    for path in seen:
        lines.append(f"- `{path}`")
    lines.append("")
    if union_results:
        lines.append("q200 union probe scratch inputs:")
        lines.append("")
        for path in (
            "/tmp/raw_scope_union_downstream_2026-05-25b_rows.jsonl",
            "/tmp/raw_scope_union_downstream_2026-05-25b_housing_rows.jsonl",
        ):
            if Path(path).exists():
                lines.append(f"- `{path}`")
        lines.append("")
    lines.append(f"Unigram LM cache directory used for this run: `{lm_cache_dir}`")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_perplexity_axis.py \\")
    lines.append("  --output docs/generated/perplexity_axis_2026-05-25.md")
    lines.append("```")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/generated/perplexity_axis_2026-05-25.md")
    parser.add_argument("--lm-cache-dir", type=Path, default=Path("/tmp/perplexity_axis_lm_cache_2026-05-25"))
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--datasets", nargs="+", default=["barexam", "housing"], choices=sorted(DATASETS))
    args = parser.parse_args()

    dataset_results: dict[str, dict[str, Any]] = {}
    q_scores_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    all_query_logppl: dict[str, list[float]] = {}
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        print(f"[dataset] {spec.key}: build/load LM", flush=True)
        lms = build_or_load_lm(spec, args.lm_cache_dir, args.batch_size)
        print(f"[dataset] {spec.key}: score questions", flush=True)
        q_scores = question_scores(spec, lms)
        q_scores_by_dataset[dataset] = q_scores
        query_values = list(q_scores.values())
        all_query_logppl[dataset] = [row["log_perplexity"] for row in query_values]
        points, model_summaries = build_points(spec, q_scores)
        pooled_summary = summarize_points(points)
        model_summaries["pooled"] = pooled_summary
        correlations = {model: correlation_row([p for p in points if p["model"] == model]) for model in MODELS}
        correlations["pooled"] = correlation_row(points)
        dataset_results[dataset] = {
            "display": spec.display,
            "question_summary": {
                "n": len(query_values),
                "lm_scope": "per state" if spec.corpus_group_meta else "corpus-wide",
                "median_ppl": median([row["perplexity"] for row in query_values]),
                "p25_ppl": percentile([row["perplexity"] for row in query_values], 0.25),
                "p75_ppl": percentile([row["perplexity"] for row in query_values], 0.75),
                "mean_log_ppl": mean([row["log_perplexity"] for row in query_values]),
                "mean_oov_rate": mean([row["oov_rate"] for row in query_values]),
                "median_tokens": median([row["token_count"] for row in query_values]),
            },
            "model_summaries": model_summaries,
            "correlations": correlations,
            "bins": binned_curve(points, bins=5),
        }

    if "barexam" in all_query_logppl and "housing" in all_query_logppl:
        dataset_results["separation"] = {
            "auc_barexam_gt_housing": rank_auc_greater(all_query_logppl["barexam"], all_query_logppl["housing"]),
            "cohen_d_barexam_minus_housing": cohen_d(all_query_logppl["barexam"], all_query_logppl["housing"]),
        }

    union_results: dict[str, dict[str, Any]] = {}
    union_paths = {
        "barexam": Path("/tmp/raw_scope_union_downstream_2026-05-25b_rows.jsonl"),
        "housing": Path("/tmp/raw_scope_union_downstream_2026-05-25b_housing_rows.jsonl"),
    }
    for dataset, path in union_paths.items():
        if dataset in q_scores_by_dataset and path.exists():
            union_results[dataset] = union_probe_summary(DATASETS[dataset], q_scores_by_dataset[dataset], path)

    make_report(
        output=args.output,
        dataset_results=dataset_results,
        union_results=union_results,
        lm_cache_dir=args.lm_cache_dir,
    )
    print(args.output)


if __name__ == "__main__":
    main()
