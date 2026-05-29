#!/usr/bin/env python3
"""Credibility-shoring analyses for SCOPE generalization.

The script is deliberately results-lane only: it reads existing generation,
retrieval, geometry, and factuality caches and writes docs/generated reports.
It does not touch paper artifacts.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_beir_phase1 import BEIR_SPECS, BeirSpec, corpus_csv, load_questions_for_spec  # noqa: E402
from analyze_factuality_falsification import fit_logistic as factuality_fit_logistic  # noqa: E402
from analyze_factuality_falsification import read_jsonl as factuality_read_jsonl  # noqa: E402
from analyze_perplexity_axis import DATASETS as LEGAL_DATASETS  # noqa: E402
from analyze_scope_gap_mechanism import fetch_docs_by_idx, load_questions_raw_text  # noqa: E402
from build_factuality_judge_cache import dataset_specs as factuality_dataset_specs  # noqa: E402
from build_factuality_judge_cache import generation_passage  # noqa: E402
from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _gold_ids, _retrieval_question, _row_label  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "he", "her",
    "his", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}

AFFINITY_POINTS = Path("/tmp/affinity_margin_oncache_2026-05-26_points.jsonl")
BEIR_POINTS = Path("/tmp/beir_phase1_verification_2026-05-26_points.jsonl")
BM25_CACHE_DIR = Path("/tmp/credibility_bm25_2026-05-29")

PHASE_B_REPORT = ROOT / "docs/generated/credibility_B_partial_correlation_2026-05-29.md"
PHASE_B_POINTS = ROOT / "docs/generated/credibility_B_partial_correlation_2026-05-29_points.jsonl"
PHASE_C_REPORT = ROOT / "docs/generated/credibility_C_bm25_replication_2026-05-29.md"
PHASE_C_POINTS = ROOT / "docs/generated/credibility_C_bm25_replication_2026-05-29_points.jsonl"
PHASE_A_REPORT = ROOT / "docs/generated/credibility_A_independent_judge_2026-05-29.md"
PHASE_A_POINTS = ROOT / "docs/generated/credibility_A_independent_judge_2026-05-29_points.jsonl"
FINAL_REPORT = ROOT / "docs/generated/credibility_battery_summary_2026-05-29.md"

OR_GEMMA = "or-gemma4-26b"


@dataclass(frozen=True)
class DatasetSource:
    key: str
    display: str
    kind: str
    collection: str = ""
    beir_spec: BeirSpec | None = None
    housing_state_filter: bool = False


DATASET_SOURCES: dict[str, DatasetSource] = {
    "barexam": DatasetSource("barexam", "BarExamQA", "legal", "legal_passages"),
    "housing": DatasetSource("housing", "HousingQA state-filtered", "legal", "housing_statutes", housing_state_filter=True),
    **{
        key: DatasetSource(key, spec.display, "beir", spec.collection, spec)
        for key, spec in BEIR_SPECS.items()
    },
}


def tokenize(text: Any) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(str(text or "").lower()) if len(tok) > 1 and tok not in STOPWORDS]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_by_label(path: str | Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("label") or row.get("idx")): row for row in read_jsonl(path)}


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def mean(values: Iterable[Any]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt(value: Any, digits: int = 3) -> str:
    if not finite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def pct(value: Any) -> str:
    if not finite(value):
        return "--"
    return f"{100.0 * float(value):.1f}%"


def corr(xs: list[float], ys: list[float]) -> dict[str, float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": float("nan"), "kendall": float("nan"), "pearson": float("nan")}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(pairs), "spearman": float("nan"), "kendall": float("nan"), "pearson": float("nan")}
    s = spearmanr(x, y, nan_policy="omit").statistic
    k = kendalltau(x, y, nan_policy="omit").statistic
    p = np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]
    return {
        "n": len(pairs),
        "spearman": float(s) if finite(s) else float("nan"),
        "kendall": float(k) if finite(k) else float("nan"),
        "pearson": float(p) if finite(p) else float("nan"),
    }


def standardize(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (x - mu) / sd


def standardize_y(values: list[float]) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    sd = y.std()
    if sd == 0.0:
        sd = 1.0
    return (y - y.mean()) / sd


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def ols_with_partials(rows: list[dict[str, Any]], features: list[str], target: str) -> dict[str, Any]:
    kept = [r for r in rows if finite(r.get(target)) and all(finite(r.get(f)) for f in features)]
    if len(kept) < 5:
        return {"n": len(kept), "r2": float("nan"), "coefficients": {}, "partial_r2": {}}
    x = standardize(np.asarray([[float(r[f]) for f in features] for r in kept], dtype=np.float64))
    y = standardize_y([float(r[target]) for r in kept])
    model = LinearRegression().fit(x, y)
    full_r2 = r2_score(y, model.predict(x))
    partial = {}
    for i, feature in enumerate(features):
        reduced = np.delete(x, i, axis=1)
        reduced_model = LinearRegression().fit(reduced, y)
        reduced_r2 = r2_score(y, reduced_model.predict(reduced))
        partial[feature] = max(0.0, full_r2 - reduced_r2) if finite(full_r2) and finite(reduced_r2) else float("nan")
    return {
        "n": len(kept),
        "r2": full_r2,
        "coefficients": {f: float(c) for f, c in zip(features, model.coef_)},
        "partial_r2": partial,
    }


def safe_auc(y: np.ndarray, proba: np.ndarray) -> float:
    if len(set(int(v) for v in y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, proba))


def logistic_with_partials(rows: list[dict[str, Any]], features: list[str], target: str) -> dict[str, Any]:
    kept = [r for r in rows if finite(r.get(target)) and all(finite(r.get(f)) for f in features)]
    if len(kept) < 10:
        return {"n": len(kept), "failures": 0, "auc": float("nan"), "pseudo_r2": float("nan"), "coefficients": {}, "partial_r2": {}}
    y = np.asarray([int(float(r[target]) > 0) for r in kept], dtype=np.int64)
    failures = int(y.sum())
    if failures == 0 or failures == len(y):
        return {"n": len(kept), "failures": failures, "auc": float("nan"), "pseudo_r2": float("nan"), "coefficients": {}, "partial_r2": {}}
    x = standardize(np.asarray([[float(r[f]) for f in features] for r in kept], dtype=np.float64))
    model = LogisticRegression(max_iter=2000).fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    loss = float(log_loss(y, proba, labels=[0, 1]))
    null_p = min(max(float(y.mean()), 1e-9), 1.0 - 1e-9)
    null_loss = float(log_loss(y, np.full_like(y, null_p, dtype=np.float64), labels=[0, 1]))
    full_pr2 = 1.0 - loss / null_loss if null_loss else float("nan")
    partial: dict[str, float] = {}
    for feature in features:
        reduced = [f for f in features if f != feature]
        if not reduced:
            partial[feature] = full_pr2
            continue
        row = logistic_with_partials(kept, reduced, target)
        partial[feature] = max(0.0, full_pr2 - float(row["pseudo_r2"])) if finite(row.get("pseudo_r2")) else float("nan")
    return {
        "n": len(kept),
        "failures": failures,
        "auc": safe_auc(y, proba),
        "log_loss": loss,
        "pseudo_r2": full_pr2,
        "coefficients": {f: float(c) for f, c in zip(features, model.coef_[0])},
        "partial_r2": partial,
    }


def beir_doc_iter(spec: BeirSpec) -> Iterable[tuple[str, str]]:
    with corpus_csv(spec).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = str(row.get("idx") or row.get("_id") or row.get("id") or "")
            title = str(row.get("title") or "")
            text = str(row.get("text") or "")
            yield idx, f"{title}. {text}" if title and title not in text[:100] else text


def chroma_doc_iter(collection_name: str, batch_size: int = 10000) -> Iterable[tuple[str, str]]:
    import chromadb

    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR", str(ROOT / "chroma_db")))
    collection = client.get_collection(collection_name)
    count = collection.count()
    for offset in range(0, count, batch_size):
        batch = collection.get(
            offset=offset,
            limit=min(batch_size, count - offset),
            include=["documents", "metadatas"],
        )
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for chroma_id, doc, meta in zip(ids, docs, metas):
            meta = dict(meta or {})
            idx = str(meta.get("idx") or str(chroma_id).removeprefix("doc_"))
            title = str(meta.get("title") or meta.get("source") or "")
            text = str(doc or "")
            yield idx, f"{title}. {text}" if title and title.lower() not in text[:120].lower() else text
        print(f"[chroma] {collection_name}: {min(offset + batch.get('ids', []).__len__(), count)}/{count}", flush=True)


def iter_docs(source: DatasetSource) -> Iterable[tuple[str, str]]:
    if source.kind == "beir":
        assert source.beir_spec is not None
        yield from beir_doc_iter(source.beir_spec)
    else:
        yield from chroma_doc_iter(source.collection)


def bm25_idf(n_docs: int, df: int) -> float:
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def bm25_score(query: str, doc_text: str, stats: dict[str, Any], k1: float = 1.5, b: float = 0.75) -> float:
    q_terms = tokenize(query)
    if not q_terms:
        return float("nan")
    doc_terms = tokenize(doc_text)
    if not doc_terms:
        return 0.0
    tf = Counter(doc_terms)
    dl = len(doc_terms)
    n_docs = int(stats["n_docs"])
    avgdl = float(stats["avgdl"])
    df = stats["df"]
    score = 0.0
    for term in q_terms:
        freq = tf.get(term, 0)
        if not freq:
            continue
        idf = bm25_idf(n_docs, int(df.get(term, 0)))
        denom = freq + k1 * (1.0 - b + b * dl / avgdl)
        score += idf * (freq * (k1 + 1.0) / denom)
    return float(score)


def bm25_stats_for_terms(source: DatasetSource, needed_terms: set[str], cache_name: str) -> dict[str, Any]:
    BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = BM25_CACHE_DIR / f"{cache_name}.pkl.gz"
    if path.exists():
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    started = time.time()
    df: Counter[str] = Counter()
    total_len = 0
    n_docs = 0
    for _, text in iter_docs(source):
        toks = tokenize(text)
        n_docs += 1
        total_len += len(toks)
        if toks:
            df.update(set(toks).intersection(needed_terms))
    stats = {
        "n_docs": n_docs,
        "avgdl": total_len / n_docs if n_docs else 0.0,
        "df": dict(df),
        "terms": sorted(needed_terms),
        "elapsed_sec": round(time.time() - started, 3),
    }
    with gzip.open(path, "wb") as f:
        pickle.dump(stats, f)
    return stats


def legal_questions(dataset: str) -> dict[str, dict[str, Any]]:
    spec = LEGAL_DATASETS[dataset]
    config = EvalConfig(dataset=spec.key, questions="full", seed=42, housing_state_filter=spec.housing_state_filter)
    out = {}
    for fallback_i, row in load_questions(config).iterrows():
        label = _row_label(row, config, fallback_i=fallback_i)
        out[label] = {
            "label": label,
            "question": _retrieval_question(row),
            "gold_ids": [str(x) for x in _gold_ids(row) if str(x)],
        }
    return out


def beir_questions(dataset: str) -> dict[str, dict[str, Any]]:
    spec = BEIR_SPECS[dataset]
    raw = load_questions_for_spec(spec)
    return {
        label: {"label": label, "question": row["question"], "gold_ids": [str(x) for x in row["gold_ids"] if str(x)]}
        for label, row in raw.items()
    }


def questions_for(dataset: str) -> dict[str, dict[str, Any]]:
    return beir_questions(dataset) if dataset.startswith("beir_") else legal_questions(dataset)


def generation_for(dataset: str, expansion: str, model: str = OR_GEMMA) -> dict[str, dict[str, Any]]:
    if dataset.startswith("beir_"):
        suffix = "rag_hyde" if expansion == "hyde" else "snap_hyre"
        return load_by_label(ROOT / f"caches/generation/full/{dataset}_qfull_seed42_{model}_{suffix}.jsonl")
    specs = factuality_dataset_specs()
    return load_by_label(specs[dataset].expansions[expansion].generation)


def gold_docs_for(dataset: str, gold_ids: list[str]) -> dict[str, str]:
    source = DATASET_SOURCES[dataset]
    ids = list(dict.fromkeys(str(x) for x in gold_ids if str(x)))
    if source.kind == "beir":
        docs = {}
        wanted = set(ids)
        assert source.beir_spec is not None
        for idx, text in beir_doc_iter(source.beir_spec):
            if idx in wanted:
                docs[idx] = text
                if len(docs) == len(wanted):
                    break
        missing = sorted(wanted - set(docs))
        if missing:
            raise RuntimeError(f"{dataset}: missing BEIR gold docs {missing[:5]} n={len(missing)}")
        return docs
    fetched = fetch_docs_by_idx(source.collection, ids)
    return {idx: str(row.get("text") or "") for idx, row in fetched.items()}


def max_bm25_to_gold(query: str, gold_ids: list[str], docs: dict[str, str], stats: dict[str, Any]) -> float:
    vals = [bm25_score(query, docs[gid], stats) for gid in gold_ids if gid in docs]
    return max((v for v in vals if finite(v)), default=float("nan"))


def load_scope_rows_for_phase_b() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(AFFINITY_POINTS):
        dataset = str(row.get("dataset"))
        if dataset not in {"barexam", "housing"}:
            continue
        rows.append({
            "dataset": dataset,
            "dataset_display": row.get("dataset_display") or DATASET_SOURCES[dataset].display,
            "model": str(row.get("model")),
            "label": str(row.get("label")),
            "retrieval_delta": float(row.get("retrieval_delta")),
            "answer_delta": float(row.get("answer_delta")) if finite(row.get("answer_delta")) else float("nan"),
            "ce_raw_gold": float(row.get("ce_raw_gold")),
            "ce_scope_gold": float(row.get("ce_scope_gold")),
            "ce_delta_gold_only": float(row.get("ce_delta_gold_only")),
            "ce_margin_raw": float(row.get("ce_margin_raw")),
            "ce_delta_margin": float(row.get("ce_delta_margin")),
            "log_perplexity": float(row.get("log_perplexity")),
            "oov_rate": float(row.get("oov_rate")),
            "question_tokens": float(row.get("question_tokens")),
        })
    for row in read_jsonl(BEIR_POINTS):
        if row.get("expansion") != "scope" or row.get("model") != OR_GEMMA:
            continue
        dataset = str(row.get("dataset"))
        rows.append({
            "dataset": dataset,
            "dataset_display": row.get("dataset_display") or DATASET_SOURCES[dataset].display,
            "model": OR_GEMMA,
            "label": str(row.get("label")),
            "retrieval_delta": float(row.get("retrieval_delta")),
            "answer_delta": float("nan"),
            "ce_raw_gold": float(row.get("ce_raw_gold")),
            "ce_scope_gold": float(row.get("ce_exp_gold")),
            "ce_delta_gold_only": float(row.get("ce_gold_delta")),
            "ce_margin_raw": float(row.get("ce_margin_raw")),
            "ce_delta_margin": float(row.get("ce_delta_margin")),
            "log_perplexity": float(row.get("log_perplexity")),
            "oov_rate": float(row.get("oov_rate")),
            "question_tokens": float(row.get("token_count")),
        })
    return rows


def attach_bm25_gold_controls(rows: list[dict[str, Any]], *, skip_housing_stats: bool = False) -> list[dict[str, Any]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, drows in by_dataset.items():
        if dataset == "housing" and skip_housing_stats:
            for row in drows:
                row["bm25_raw_gold"] = float("nan")
                row["bm25_scope_gold"] = float("nan")
                row["bm25_control_note"] = "deferred_housing_bm25_stats"
            continue
        qmap = questions_for(dataset)
        gen = generation_for(dataset, "scope", OR_GEMMA)
        texts: dict[str, tuple[str, str, list[str]]] = {}
        needed_terms: set[str] = set()
        gold_ids_all: list[str] = []
        for row in drows:
            label = row["label"]
            q = qmap[label]
            passage = generation_passage(gen[label])
            texts[label] = (q["question"], passage, q["gold_ids"])
            needed_terms.update(tokenize(q["question"]))
            needed_terms.update(tokenize(passage))
            gold_ids_all.extend(q["gold_ids"])
        source = DATASET_SOURCES[dataset]
        print(f"[phase-b] BM25 stats dataset={dataset} terms={len(needed_terms)}", flush=True)
        stats = bm25_stats_for_terms(source, needed_terms, f"phaseB_{dataset}_scope_gold")
        docs = gold_docs_for(dataset, gold_ids_all)
        for row in drows:
            raw_q, scope_p, gold_ids = texts[row["label"]]
            row["bm25_raw_gold"] = max_bm25_to_gold(raw_q, gold_ids, docs, stats)
            row["bm25_scope_gold"] = max_bm25_to_gold(scope_p, gold_ids, docs, stats)
            row["bm25_control_note"] = "corpus_df_gold_doc_score"
    return rows


def phase_b(args: argparse.Namespace) -> None:
    rows = load_scope_rows_for_phase_b()
    if not rows:
        raise SystemExit("No Phase B rows loaded")
    rows = attach_bm25_gold_controls(rows, skip_housing_stats=args.skip_housing_bm25_stats)
    write_jsonl(PHASE_B_POINTS, rows)

    features = ["ce_delta_gold_only", "ce_raw_gold", "bm25_scope_gold", "bm25_raw_gold", "log_perplexity", "question_tokens", "oov_rate"]
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for dataset in sorted({r["dataset"] for r in rows}):
        display = next(r["dataset_display"] for r in rows if r["dataset"] == dataset)
        groups.append((dataset, display, [r for r in rows if r["dataset"] == dataset]))
    groups.append(("pooled", "Pooled", rows))

    ols_rows = []
    for dataset, display, group in groups:
        fit = ols_with_partials(group, features, "retrieval_delta")
        ols_rows.append({"dataset": dataset, "dataset_display": display, **fit})

    p4_features = ["ce_margin_raw", "ce_scope_gold", "ce_raw_gold", "bm25_scope_gold", "bm25_raw_gold"]
    for row in rows:
        row["target_margin_failure"] = float(float(row["ce_delta_margin"]) < 0.0) if finite(row.get("ce_delta_margin")) else float("nan")
        row["target_retrieval_hurt"] = float(float(row["retrieval_delta"]) < 0.0) if finite(row.get("retrieval_delta")) else float("nan")
    p4_rows = []
    for dataset, display, group in groups:
        p4_rows.append({"dataset": dataset, "dataset_display": display, "feature_set": "OOV + logPPL", **logistic_with_partials(group, ["oov_rate", "log_perplexity"], "target_margin_failure")})
        p4_rows.append({"dataset": dataset, "dataset_display": display, "feature_set": "Geometry + CE/BM25 controls", **logistic_with_partials(group, p4_features, "target_margin_failure")})

    pooled = next(row for row in ols_rows if row["dataset"] == "pooled")
    gold_partial = float(pooled["partial_r2"].get("ce_delta_gold_only", float("nan")))
    verdict = "survives" if finite(gold_partial) and gold_partial >= 0.05 else "mechanical/soften"

    lines = [
        "# Credibility Battery Phase B - Partial Correlation",
        "",
        "Read-only analysis over existing affinity-margin points plus regenerated BM25 gold-affinity controls. No `paper/` files were edited.",
        "",
        "## Verdict",
        "",
        f"- Mechanism circularity check: **{verdict}**. Pooled gold-affinity-delta partial-R2 after CE(raw,gold) and BM25 controls is `{fmt(gold_partial)}`.",
        "- Kill criterion: below 0.05 means the gold-affinity-delta mechanism is mostly mechanical after controlling for raw closeness and BM25-space affinity.",
        "",
        "## OLS: Retrieval Gain on Geometry and Controls",
        "",
        "| Dataset | N | R2 | Gold-delta beta | Gold-delta partial-R2 | CE(raw,gold) beta | BM25 scope beta | BM25 raw beta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ols_rows:
        coeff = row.get("coefficients", {})
        part = row.get("partial_r2", {})
        lines.append(
            f"| {row['dataset_display']} | {row['n']} | {fmt(row['r2'])} | "
            f"{fmt(coeff.get('ce_delta_gold_only'))} | {fmt(part.get('ce_delta_gold_only'))} | "
            f"{fmt(coeff.get('ce_raw_gold'))} | {fmt(coeff.get('bm25_scope_gold'))} | {fmt(coeff.get('bm25_raw_gold'))} |"
        )
    lines.extend([
        "",
        "## P4 Failure Model With Controls",
        "",
        "Target is `1[CE deltaM < 0]`. The controlled geometry model adds CE(raw,gold), BM25(scope,gold), and BM25(raw,gold) to the prior geometry features.",
        "",
        "| Dataset | Feature set | N | Failures | AUC | Pseudo-R2 | Key partial-R2 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for row in p4_rows:
        part = row.get("partial_r2", {})
        bits = "; ".join(f"`{k}`={fmt(v)}" for k, v in part.items())
        lines.append(
            f"| {row['dataset_display']} | {row['feature_set']} | {row['n']} | {row['failures']} | "
            f"{fmt(row['auc'])} | {fmt(row['pseudo_r2'])} | {bits} |"
        )
    lines.extend([
        "",
        "## BM25 Control Notes",
        "",
        "- BM25 controls score the raw question and SCOPE passage against the gold passage set using corpus-wide document-frequency statistics for the query terms.",
        "- BEIR corpora are read from `datasets/beir/*/corpus.csv`; legal corpora are streamed from the Chroma collections.",
        f"- Row-level points: `{PHASE_B_POINTS.relative_to(ROOT)}`",
        "",
    ])
    PHASE_B_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PHASE_B_REPORT.write_text("\n".join(lines) + "\n")
    print(PHASE_B_REPORT)


def hit_at(ids: list[str], gold_ids: list[str], k: int) -> int:
    return int(bool(set(str(x) for x in ids[:k]) & set(str(g) for g in gold_ids if str(g))))


def build_bm25_retrieval_rows(dataset: str, arms: list[str], *, limit: int = 0, max_docs: int = 0) -> tuple[list[dict[str, Any]], str]:
    source = DATASET_SOURCES[dataset]
    qmap = questions_for(dataset)
    labels = list(qmap)
    if limit:
        labels = labels[:limit]
    query_by_arm: dict[str, dict[str, str]] = {"raw": {label: qmap[label]["question"] for label in labels}}
    for arm in arms:
        if arm == "raw":
            continue
        gen = generation_for(dataset, arm, OR_GEMMA)
        query_by_arm[arm] = {label: generation_passage(gen[label]) for label in labels}
    needed_terms = {tok for amap in query_by_arm.values() for text in amap.values() for tok in tokenize(text)}
    started = time.time()
    # First pass: corpus stats and term filtering.
    n_docs = 0
    total_len = 0
    df: Counter[str] = Counter()
    doc_text_cache: dict[str, str] = {}
    for idx, text in iter_docs(source):
        if max_docs and n_docs >= max_docs:
            break
        toks = tokenize(text)
        n_docs += 1
        total_len += len(toks)
        df.update(set(toks).intersection(needed_terms))
        if dataset.startswith("beir_"):
            doc_text_cache[idx] = text
    if not n_docs:
        raise RuntimeError(f"{dataset}: empty corpus")
    avgdl = total_len / n_docs
    filtered_terms = {t for t in needed_terms if df.get(t, 0) and bm25_idf(n_docs, df[t]) > 0.05}
    print(f"[phase-c] {dataset}: docs={n_docs} terms={len(needed_terms)} filtered_terms={len(filtered_terms)}", flush=True)
    # Second pass: query scoring from only matching query terms.
    query_terms: dict[tuple[str, str], Counter[str]] = {}
    term_to_queries: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for arm, amap in query_by_arm.items():
        for label, text in amap.items():
            counter = Counter(tok for tok in tokenize(text) if tok in filtered_terms)
            query_terms[(arm, label)] = counter
            for tok in counter:
                term_to_queries[tok].append((arm, label))
    top: dict[tuple[str, str], list[tuple[float, str]]] = {key: [] for key in query_terms}
    seen_docs = 0
    for idx, text in iter_docs(source):
        if max_docs and seen_docs >= max_docs:
            break
        seen_docs += 1
        toks = tokenize(text)
        if not toks:
            continue
        tf = Counter(toks)
        active: set[tuple[str, str]] = set()
        for tok in tf:
            if tok in term_to_queries:
                active.update(term_to_queries[tok])
        if not active:
            continue
        dl = len(toks)
        for key in active:
            score = 0.0
            for term, qtf in query_terms[key].items():
                freq = tf.get(term, 0)
                if not freq:
                    continue
                idf = bm25_idf(n_docs, df[term])
                denom = freq + 1.5 * (1.0 - 0.75 + 0.75 * dl / avgdl)
                score += qtf * idf * (freq * 2.5 / denom)
            if score <= 0:
                continue
            heap = top[key]
            heap.append((score, idx))
            if len(heap) > 25:
                heap.sort(reverse=True)
                del heap[10:]
    gold_ids_all = [gid for label in labels for gid in qmap[label]["gold_ids"]]
    gold_docs = gold_docs_for(dataset, gold_ids_all)
    stats = {"n_docs": n_docs, "avgdl": avgdl, "df": dict(df)}
    rows: list[dict[str, Any]] = []
    for arm in arms:
        for label in labels:
            ranked = sorted(top[(arm, label)], reverse=True)[:10]
            ids = [idx for _, idx in ranked]
            scores = [score for score, _ in ranked]
            qtext = query_by_arm[arm][label]
            gold_ids = qmap[label]["gold_ids"]
            rows.append({
                "dataset": dataset,
                "dataset_display": source.display,
                "arm": arm,
                "label": label,
                "gold_ids": gold_ids,
                "retrieved_ids": ids,
                "scores": scores,
                "hit5": hit_at(ids, gold_ids, 5),
                "hit10": hit_at(ids, gold_ids, 10),
                "bm25_gold_affinity": max_bm25_to_gold(qtext, gold_ids, gold_docs, stats),
                "query_chars": len(qtext),
            })
    note = f"docs={n_docs}; max_docs={max_docs or 'full'}; elapsed_sec={time.time() - started:.1f}"
    return rows, note


def phase_c(args: argparse.Namespace) -> None:
    datasets = args.datasets or ["beir_scifact", "beir_nfcorpus", "beir_fiqa", "barexam"]
    if args.include_housing:
        datasets.append("housing")
    all_rows: list[dict[str, Any]] = []
    notes: dict[str, str] = {}
    failures: dict[str, str] = {}
    for dataset in datasets:
        try:
            rows, note = build_bm25_retrieval_rows(dataset, ["raw", "hyde", "scope"], limit=args.limit, max_docs=args.max_docs)
            all_rows.extend(rows)
            notes[dataset] = note
            write_jsonl(PHASE_C_POINTS, all_rows)
        except Exception as exc:
            failures[dataset] = str(exc)
            print(f"[phase-c] {dataset}: deferred/failure {exc}", flush=True)
    by_key = {(r["dataset"], r["arm"], r["label"]): r for r in all_rows}
    summary_rows = []
    corr_rows = []
    for dataset in sorted({r["dataset"] for r in all_rows}):
        raw_rows = [r for r in all_rows if r["dataset"] == dataset and r["arm"] == "raw"]
        for arm in ["raw", "hyde", "scope"]:
            rows = [r for r in all_rows if r["dataset"] == dataset and r["arm"] == arm]
            if not rows:
                continue
            help_n = hurt_n = 0
            deltas = []
            gold_deltas = []
            if arm != "raw":
                for row in rows:
                    raw = by_key[(dataset, "raw", row["label"])]
                    delta = int(row["hit5"]) - int(raw["hit5"])
                    deltas.append(delta)
                    gold_deltas.append(float(row["bm25_gold_affinity"]) - float(raw["bm25_gold_affinity"]))
                    if delta > 0:
                        help_n += 1
                    elif delta < 0:
                        hurt_n += 1
                corr_rows.append({
                    "dataset": dataset,
                    "arm": arm,
                    **corr(gold_deltas, deltas),
                    "mean_gold_delta": mean(gold_deltas),
                })
            summary_rows.append({
                "dataset": dataset,
                "display": rows[0]["dataset_display"],
                "arm": arm,
                "n": len(rows),
                "hit5": mean(r["hit5"] for r in rows),
                "hit10": mean(r["hit10"] for r in rows),
                "delta": mean(deltas) if deltas else 0.0,
                "help": help_n,
                "hurt": hurt_n,
                "ri": (help_n - hurt_n) / len(rows) if rows and arm != "raw" else 0.0,
            })
    pooled_scope = [r for r in corr_rows if r["arm"] == "scope"]
    pooled_rho = corr([r["mean_gold_delta"] for r in pooled_scope], [r["spearman"] for r in pooled_scope])["spearman"] if len(pooled_scope) >= 3 else float("nan")
    scope_corr_values = [r["spearman"] for r in corr_rows if r["arm"] == "scope" and finite(r.get("spearman"))]
    mean_scope_corr = mean(scope_corr_values)
    verdict = "travels" if finite(mean_scope_corr) and mean_scope_corr >= 0.3 else ("gte-CE-specific/soften" if finite(mean_scope_corr) and mean_scope_corr <= 0.2 else "mixed")
    lines = [
        "# Credibility Battery Phase C - BM25 Replication",
        "",
        "BM25 replication under a non-dense retriever. Generator remains `or-gemma4-26b`; only retrieval is changed. No `paper/` files were edited.",
        "",
        "## Verdict",
        "",
        f"- BM25 mechanism verdict: **{verdict}**. Mean per-dataset SCOPE Spearman between BM25 gold-affinity delta and BM25 retrieval gain is `{fmt(mean_scope_corr)}`.",
        "- Kill criterion: <=0.2 means the mechanism is likely gte/CE-specific; >=0.3 means it travels to BM25.",
        "",
        "## Retrieval Summary",
        "",
        "| Dataset | Arm | N | Hit@5 | Hit@10 | Delta vs raw | Help | Hurt | RI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['display']} | {row['arm']} | {row['n']} | {pct(row['hit5'])} | {pct(row['hit10'])} | "
            f"{pct(row['delta'])} | {row['help']} | {row['hurt']} | {fmt(row['ri'])} |"
        )
    lines.extend([
        "",
        "## BM25 Gold-Affinity Delta Correlations",
        "",
        "| Dataset | Arm | N | Spearman | Kendall | Pearson | Mean BM25 gold delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in corr_rows:
        display = DATASET_SOURCES[row["dataset"]].display
        lines.append(
            f"| {display} | {row['arm']} | {row['n']} | {fmt(row['spearman'])} | {fmt(row['kendall'])} | "
            f"{fmt(row['pearson'])} | {fmt(row['mean_gold_delta'])} |"
        )
    lines.extend([
        "",
        "## Run Notes",
        "",
    ])
    for dataset, note in notes.items():
        lines.append(f"- {DATASET_SOURCES[dataset].display}: {note}")
    for dataset, err in failures.items():
        lines.append(f"- {DATASET_SOURCES[dataset].display}: deferred/failed - `{err[:240]}`")
    lines.append(f"- Row-level BM25 points: `{PHASE_C_POINTS.relative_to(ROOT)}`")
    lines.append("")
    _ = pooled_rho
    PHASE_C_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PHASE_C_REPORT.write_text("\n".join(lines) + "\n")
    print(PHASE_C_REPORT)


def phase_a(args: argparse.Namespace) -> None:
    gemma_features = read_jsonl(ROOT / "docs/generated/factuality_feature_points_q200_2026-05-28.jsonl")
    independent_cache = Path(args.independent_judge_cache)
    if not independent_cache.exists():
        raise SystemExit(f"Independent judge cache missing: {independent_cache}. Build it with scripts/build_factuality_judge_cache.py first.")
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(independent_cache):
        grouped[(str(row.get("dataset")), str(row.get("label")), str(row.get("expansion")))] [str(row.get("premise_kind"))] = row
    points = []
    for row in gemma_features:
        key = (str(row.get("dataset")), str(row.get("label")), str(row.get("expansion")))
        pair = grouped.get(key, {})
        if "gold" not in pair or "raw_top3" not in pair:
            continue
        out = dict(row)
        out["independent_factuality_gold_score"] = float(pair["gold"]["score"])
        out["independent_factuality_raw_top3_score"] = float(pair["raw_top3"]["score"])
        out["independent_gold_verdict"] = pair["gold"].get("verdict")
        out["independent_raw_top3_verdict"] = pair["raw_top3"].get("verdict")
        out["independent_judge_model"] = pair["gold"].get("judge_model", args.judge_label)
        points.append(out)
    write_jsonl(PHASE_A_POINTS, points)
    for p in points:
        p["target_retrieval_hurt"] = float(int(p["retrieval_delta"]) < 0)
        p["target_margin_failure"] = float(float(p["ce_delta_margin"]) < 0.0) if finite(p.get("ce_delta_margin")) else float("nan")
    feature_sets = [
        ("OOV + logPPL", ["oov_rate", "log_perplexity"]),
        ("Original gemma factuality", ["factuality_gold_score"]),
        ("Independent factuality", ["independent_factuality_gold_score"]),
        ("Geometry", ["ce_margin_raw", "ce_exp_gold"]),
        ("Independent factuality + geometry", ["independent_factuality_gold_score", "ce_margin_raw", "ce_exp_gold"]),
    ]
    auc_rows = []
    for dataset in sorted({p["dataset"] for p in points}) + ["pooled"]:
        group = points if dataset == "pooled" else [p for p in points if p["dataset"] == dataset]
        display = "Pooled" if dataset == "pooled" else group[0]["dataset_display"]
        for name, features in feature_sets:
            auc_rows.append({"dataset": dataset, "display": display, "feature_set": name, **logistic_with_partials(group, features, "target_retrieval_hurt")})
    rho = corr([p["factuality_gold_score"] for p in points], [p["independent_factuality_gold_score"] for p in points])
    # Cohen's kappa on factuality >= 0.75.
    pairs = [(float(p["factuality_gold_score"]) >= 0.75, float(p["independent_factuality_gold_score"]) >= 0.75) for p in points]
    agree = mean([a == b for a, b in pairs])
    p_yes_a = mean([a for a, _ in pairs])
    p_yes_b = mean([b for _, b in pairs])
    pe = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    kappa = (agree - pe) / (1 - pe) if finite(pe) and pe != 1 else float("nan")
    def pooled_auc(name: str) -> float:
        return next(float(r["auc"]) for r in auc_rows if r["dataset"] == "pooled" and r["feature_set"] == name)
    old_auc = pooled_auc("Original gemma factuality")
    indep_auc = pooled_auc("Independent factuality")
    geom_auc = pooled_auc("Geometry")
    joint_auc = pooled_auc("Independent factuality + geometry")
    marginal = joint_auc - geom_auc if finite(joint_auc) and finite(geom_auc) else float("nan")
    verdict = "survives" if finite(indep_auc) and abs(indep_auc - old_auc) <= 0.1 and finite(marginal) and marginal <= 0.02 else "soften"
    lines = [
        "# Credibility Battery Phase A - Independent Factuality Judge",
        "",
        "Independent non-Gemma factuality judge analysis over the same q200 feature sample. No `paper/` files were edited.",
        "",
        "## Verdict",
        "",
        f"- Factuality falsification headline: **{verdict}**. Original Gemma factuality AUC `{fmt(old_auc)}`, independent factuality AUC `{fmt(indep_auc)}`, geometry AUC `{fmt(geom_auc)}`, independent+geometry AUC `{fmt(joint_auc)}`, marginal lift `{fmt(marginal)}`.",
        f"- Inter-rater reliability: Spearman `{fmt(rho['spearman'])}`, Pearson `{fmt(rho['pearson'])}`, Cohen kappa `{fmt(kappa)}` on score >= 0.75.",
        "",
        "## AUC Table",
        "",
        "| Dataset | Feature set | N | Failures | AUC | Pseudo-R2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in auc_rows:
        lines.append(f"| {row['display']} | {row['feature_set']} | {row['n']} | {row['failures']} | {fmt(row['auc'])} | {fmt(row['pseudo_r2'])} |")
    lines.extend([
        "",
        "## Sources",
        "",
        f"- Original features: `docs/generated/factuality_feature_points_q200_2026-05-28.jsonl`",
        f"- Independent judge cache: `{independent_cache}`",
        f"- Joined points: `{PHASE_A_POINTS.relative_to(ROOT)}`",
        "",
    ])
    PHASE_A_REPORT.write_text("\n".join(lines) + "\n")
    print(PHASE_A_REPORT)


def final_report(_: argparse.Namespace) -> None:
    reports = [PHASE_B_REPORT, PHASE_C_REPORT, PHASE_A_REPORT]
    missing = [p for p in reports if not p.exists()]
    if missing:
        raise SystemExit(f"Missing phase reports: {missing}")
    snippets = {p.name: p.read_text().split("## Verdict", 1)[-1].split("##", 1)[0].strip() for p in reports}
    lines = [
        "# Credibility Battery Summary - 2026-05-29",
        "",
        "Aggregate of Phases A, B, and C. No `paper/` files were edited.",
        "",
        "## Phase Reads",
        "",
    ]
    for name, text in snippets.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(text)
        lines.append("")
    lines.extend([
        "## Honest Claim",
        "",
        "The credibility battery should be read as a stress test, not advocacy. The paper claim should emphasize where geometry and retrieval-regime evidence survive independent controls, and explicitly soften any component that fails the phase kill criteria.",
        "",
    ])
    FINAL_REPORT.write_text("\n".join(lines))
    print(FINAL_REPORT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("phase-b")
    b.add_argument("--skip-housing-bm25-stats", action="store_true")
    b.set_defaults(func=phase_b)
    c = sub.add_parser("phase-c")
    c.add_argument("--datasets", nargs="*")
    c.add_argument("--include-housing", action="store_true")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--max-docs", type=int, default=0)
    c.set_defaults(func=phase_c)
    a = sub.add_parser("phase-a")
    a.add_argument("--independent-judge-cache", default=str(ROOT / "docs/generated/factuality_judge_independent_q200_2026-05-29.jsonl"))
    a.add_argument("--judge-label", default="independent")
    a.set_defaults(func=phase_a)
    f = sub.add_parser("final")
    f.set_defaults(func=final_report)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
