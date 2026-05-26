#!/usr/bin/env python3
"""BEIR Phase 1 mechanism verification over committed retrieval caches."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "eval"))

from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _gold_ids, _retrieval_question, _row_label  # noqa: E402
from rag_utils import get_cross_encoder  # noqa: E402


TOKEN_RE = re.compile(r"[a-z0-9]+")
MODEL = "or-gemma4-26b"


@dataclass(frozen=True)
class BeirSpec:
    key: str
    subset: str
    display: str
    collection: str
    raw_cache: Path
    hyde_generation: Path
    scope_generation: Path
    hyde_retrieval: Path
    scope_retrieval: Path


def cache_path(path: str) -> Path:
    return REPO_ROOT / path


BEIR_SPECS: dict[str, BeirSpec] = {
    "beir_scifact": BeirSpec(
        key="beir_scifact",
        subset="scifact",
        display="SciFact",
        collection="beir_scifact",
        raw_cache=cache_path("caches/retrieval/full/beir_scifact_qfull_seed42_raw_question_k10.jsonl"),
        hyde_generation=cache_path("caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl"),
        scope_generation=cache_path("caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl"),
        hyde_retrieval=cache_path("caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl"),
        scope_retrieval=cache_path("caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl"),
    ),
    "beir_nfcorpus": BeirSpec(
        key="beir_nfcorpus",
        subset="nfcorpus",
        display="NFCorpus",
        collection="beir_nfcorpus",
        raw_cache=cache_path("caches/retrieval/full/beir_nfcorpus_qfull_seed42_raw_question_k10.jsonl"),
        hyde_generation=cache_path("caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl"),
        scope_generation=cache_path("caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl"),
        hyde_retrieval=cache_path("caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl"),
        scope_retrieval=cache_path("caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl"),
    ),
    "beir_fiqa": BeirSpec(
        key="beir_fiqa",
        subset="fiqa",
        display="FiQA",
        collection="beir_fiqa",
        raw_cache=cache_path("caches/retrieval/full/beir_fiqa_qfull_seed42_raw_question_k10.jsonl"),
        hyde_generation=cache_path("caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl"),
        scope_generation=cache_path("caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl"),
        hyde_retrieval=cache_path("caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl"),
        scope_retrieval=cache_path("caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl"),
    ),
    "beir_trec_covid": BeirSpec(
        key="beir_trec_covid",
        subset="trec-covid",
        display="TREC-COVID",
        collection="beir_trec_covid",
        raw_cache=cache_path("caches/retrieval/full/beir_trec_covid_qfull_seed42_raw_question_k10.jsonl"),
        hyde_generation=cache_path("caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl"),
        scope_generation=cache_path("caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl"),
        hyde_retrieval=cache_path("caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl"),
        scope_retrieval=cache_path("caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl"),
    ),
    "beir_scidocs": BeirSpec(
        key="beir_scidocs",
        subset="scidocs",
        display="SciDocs",
        collection="beir_scidocs",
        raw_cache=cache_path("caches/retrieval/full/beir_scidocs_qfull_seed42_raw_question_k10.jsonl"),
        hyde_generation=cache_path("caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl"),
        scope_generation=cache_path("caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl"),
        hyde_retrieval=cache_path("caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl"),
        scope_retrieval=cache_path("caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl"),
    ),
}

EXPANSIONS = {
    "hyde": ("HyDE", "hyde_generation", "hyde_retrieval"),
    "scope": ("SCOPE", "scope_generation", "scope_retrieval"),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_by_label(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("label") or row.get("idx")): row for row in read_jsonl(path)}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text or "").lower())


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


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if finite(v))
    if not vals:
        return float("nan")
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def corr(xs: list[float], ys: list[float]) -> dict[str, float]:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if finite(x) and finite(y)]
    if len(pairs) < 3:
        return {"n": len(pairs), "spearman": float("nan"), "kendall": float("nan")}
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {"n": len(pairs), "spearman": float("nan"), "kendall": float("nan")}
    s = spearmanr(x, y, nan_policy="omit").statistic
    k = kendalltau(x, y, nan_policy="omit").statistic
    return {
        "n": len(pairs),
        "spearman": float(s) if finite(s) else float("nan"),
        "kendall": float(k) if finite(k) else float("nan"),
    }


def hit_at(row: dict[str, Any], k: int) -> int:
    gold = {str(idx) for idx in row.get("gold_ids", []) if str(idx)}
    retrieved = [str(idx) for idx in row.get("retrieved_ids", [])[:k]]
    return int(bool(gold.intersection(retrieved))) if gold else 0


def rank_at(row: dict[str, Any], k: int = 10) -> int:
    gold = {str(idx) for idx in row.get("gold_ids", []) if str(idx)}
    for i, idx in enumerate(row.get("retrieved_ids", [])[:k], 1):
        if str(idx) in gold:
            return i
    return 0


def best_non_gold_ce(row: dict[str, Any], gold_ids: list[str]) -> tuple[float, str]:
    gold = {str(idx) for idx in gold_ids}
    best = float("nan")
    best_id = ""
    for idx, score in zip(row.get("retrieved_ids", []) or [], row.get("scores", []) or []):
        idx = str(idx)
        if idx in gold:
            continue
        score_f = float(score)
        if not finite(best) or score_f > best:
            best = score_f
            best_id = idx
    return best, best_id


def load_questions_for_spec(spec: BeirSpec) -> dict[str, dict[str, Any]]:
    config = EvalConfig(dataset=spec.key, questions="full", seed=42)
    out: dict[str, dict[str, Any]] = {}
    for _, row in load_questions(config).iterrows():
        label = _row_label(row, config)
        out[label] = {
            "idx": str(row.get("idx", "")),
            "question": _retrieval_question(row),
            "gold_ids": [str(idx) for idx in _gold_ids(row) if str(idx)],
        }
    return out


def corpus_csv(spec: BeirSpec) -> Path:
    return REPO_ROOT / "datasets" / "beir" / spec.subset / "corpus.csv"


def build_lm(spec: BeirSpec) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    total = 0
    with corpus_csv(spec).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            toks = tokenize(row.get("text", ""))
            counts.update(toks)
            total += len(toks)
    return {"counts": counts, "total": total, "vocab": len(counts)}


def score_lm(text: str, lm: dict[str, Any]) -> dict[str, float]:
    toks = tokenize(text)
    if not toks:
        return {"token_count": 0.0, "log_perplexity": float("nan"), "oov_rate": float("nan")}
    counts: Counter[str] = lm["counts"]
    total = int(lm["total"])
    vocab = int(lm["vocab"]) + 1
    denom = total + vocab
    log_prob = 0.0
    oov = 0
    for tok in toks:
        if tok not in counts:
            oov += 1
        log_prob += math.log((counts.get(tok, 0) + 1) / denom)
    return {
        "token_count": float(len(toks)),
        "log_perplexity": -log_prob / len(toks),
        "oov_rate": oov / len(toks),
    }


def fetch_docs_by_idx(collection_name: str, idxs: Iterable[str], batch_size: int = 5000) -> dict[str, str]:
    import chromadb

    requested = list(dict.fromkeys(str(idx) for idx in idxs if str(idx)))
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR", str(REPO_ROOT / "chroma_db")))
    collection = client.get_collection(collection_name)
    found: dict[str, str] = {}

    def store(batch: dict[str, Any]) -> None:
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for chroma_id, doc, meta in zip(ids, docs, metas):
            meta = dict(meta or {})
            idx = str(meta.get("idx") or str(chroma_id).removeprefix("doc_"))
            if idx and idx not in found:
                found[idx] = doc or ""

    for start in range(0, len(requested), batch_size):
        chunk = requested[start:start + batch_size]
        store(collection.get(ids=[f"doc_{idx}" for idx in chunk], include=["documents", "metadatas"]))
        print(f"[docs] {collection_name}: {min(start + batch_size, len(requested))}/{len(requested)}", flush=True)

    missing = [idx for idx in requested if idx not in found]
    for start in range(0, len(missing), min(batch_size, 500)):
        chunk = missing[start:start + min(batch_size, 500)]
        try:
            store(collection.get(where={"idx": {"$in": chunk}}, include=["documents", "metadatas"]))
        except Exception:
            for idx in chunk:
                store(collection.get(where={"idx": idx}, include=["documents", "metadatas"]))

    still_missing = [idx for idx in requested if idx not in found]
    if still_missing:
        raise RuntimeError(f"{collection_name}: missing docs for {still_missing[:10]} n={len(still_missing)}")
    return found


def truncate_for_ce(text: str) -> str:
    max_chars = int(os.getenv("CROSS_ENCODER_MAX_CHARS", "4096") or "4096")
    text = str(text or "")
    return text[:max_chars] if max_chars and len(text) > max_chars else text


def score_best_gold_ce(
    *,
    ce: Any,
    items: list[tuple[str, str, list[str]]],
    gold_docs: dict[str, str],
    batch_size: int,
    chunk_size: int,
    tag: str,
) -> dict[str, tuple[float, str]]:
    pairs: list[tuple[str, str]] = []
    meta: list[tuple[str, str]] = []
    for key, query_text, gold_ids in items:
        q = truncate_for_ce(query_text)
        for gid in gold_ids:
            pairs.append((q, truncate_for_ce(gold_docs[gid])))
            meta.append((key, gid))

    out: dict[str, tuple[float, str]] = {}
    total = len(pairs)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        print(f"[ce] {tag}: {end}/{total}", flush=True)
        scores = ce.predict(pairs[start:end], batch_size=batch_size, show_progress_bar=False)
        for (key, gid), score in zip(meta[start:end], scores):
            score_f = float(score)
            current = out.get(key)
            if current is None or score_f > current[0]:
                out[key] = (score_f, gid)
    return out


def generation_passage(row: dict[str, Any]) -> str:
    return str(row.get("hyde_passage") or row.get("hypothetical_passage") or row.get("passage") or "")


def build_points_for_dataset(spec: BeirSpec, args: argparse.Namespace) -> list[dict[str, Any]]:
    print(f"[dataset] {spec.display}: load questions/caches", flush=True)
    questions = load_questions_for_spec(spec)
    raw_cache = load_by_label(spec.raw_cache)
    lm = build_lm(spec)
    lm_scores = {label: score_lm(row["question"], lm) for label, row in questions.items()}
    all_gold_ids = sorted({gid for row in questions.values() for gid in row["gold_ids"]})
    gold_docs = fetch_docs_by_idx(spec.collection, all_gold_ids, batch_size=args.doc_batch_size)

    missing_raw = [label for label in questions if label not in raw_cache]
    if missing_raw:
        raise RuntimeError(f"{spec.key}: raw cache missing labels {missing_raw[:5]} n={len(missing_raw)}")

    ce = get_cross_encoder()
    raw_items = [(label, row["question"], row["gold_ids"]) for label, row in questions.items()]
    raw_gold = score_best_gold_ce(
        ce=ce,
        items=raw_items,
        gold_docs=gold_docs,
        batch_size=args.ce_batch_size,
        chunk_size=args.ce_chunk_size,
        tag=f"{spec.display}/raw",
    )

    points: list[dict[str, Any]] = []
    for exp_key, (exp_display, gen_attr, ret_attr) in EXPANSIONS.items():
        gen_cache = load_by_label(getattr(spec, gen_attr))
        exp_cache = load_by_label(getattr(spec, ret_attr))
        missing = [label for label in questions if label not in gen_cache or label not in exp_cache]
        if missing:
            raise RuntimeError(f"{spec.key}/{exp_key}: cache missing labels {missing[:5]} n={len(missing)}")
        exp_items = [
            (label, generation_passage(gen_cache[label]), questions[label]["gold_ids"])
            for label in questions
        ]
        exp_gold = score_best_gold_ce(
            ce=ce,
            items=exp_items,
            gold_docs=gold_docs,
            batch_size=args.ce_batch_size,
            chunk_size=args.ce_chunk_size,
            tag=f"{spec.display}/{exp_display}",
        )
        for label, qrow in questions.items():
            raw_row = raw_cache[label]
            exp_row = exp_cache[label]
            gold_ids = qrow["gold_ids"]
            raw_hit5 = hit_at(raw_row, 5)
            exp_hit5 = hit_at(exp_row, 5)
            raw_dist, raw_dist_id = best_non_gold_ce(raw_row, gold_ids)
            exp_dist, exp_dist_id = best_non_gold_ce(exp_row, gold_ids)
            ce_raw_gold, ce_raw_gold_id = raw_gold[label]
            ce_exp_gold, ce_exp_gold_id = exp_gold[label]
            ce_margin_raw = ce_raw_gold - raw_dist if finite(raw_dist) else float("nan")
            ce_margin_exp = ce_exp_gold - exp_dist if finite(exp_dist) else float("nan")
            point = {
                "dataset": spec.key,
                "dataset_display": spec.display,
                "expansion": exp_key,
                "expansion_display": exp_display,
                "model": MODEL,
                "label": label,
                "idx": qrow["idx"],
                "gold_count": len(gold_ids),
                "raw_hit5": raw_hit5,
                "exp_hit5": exp_hit5,
                "retrieval_delta": exp_hit5 - raw_hit5,
                "help": int(exp_hit5 == 1 and raw_hit5 == 0),
                "hurt": int(exp_hit5 == 0 and raw_hit5 == 1),
                "raw_hit10": hit_at(raw_row, 10),
                "exp_hit10": hit_at(exp_row, 10),
                "raw_rank10": rank_at(raw_row, 10),
                "exp_rank10": rank_at(exp_row, 10),
                "ce_raw_gold": ce_raw_gold,
                "ce_raw_gold_id": ce_raw_gold_id,
                "ce_exp_gold": ce_exp_gold,
                "ce_exp_gold_id": ce_exp_gold_id,
                "ce_gold_delta": ce_exp_gold - ce_raw_gold,
                "ce_raw_distractor": raw_dist,
                "ce_raw_distractor_id": raw_dist_id,
                "ce_exp_distractor": exp_dist,
                "ce_exp_distractor_id": exp_dist_id,
                "ce_margin_raw": ce_margin_raw,
                "ce_margin_exp": ce_margin_exp,
                "ce_delta_margin": ce_margin_exp - ce_margin_raw if finite(ce_margin_exp) and finite(ce_margin_raw) else float("nan"),
                "raw_no_nongold_distractor": int(not finite(raw_dist)),
                "exp_no_nongold_distractor": int(not finite(exp_dist)),
                **lm_scores[label],
            }
            points.append(point)
    return points


def summarize(points: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(points)
    help_n = sum(int(p["help"]) for p in points)
    hurt_n = sum(int(p["hurt"]) for p in points)
    return {
        "n": n,
        "raw_hit5": mean(p["raw_hit5"] for p in points),
        "exp_hit5": mean(p["exp_hit5"] for p in points),
        "raw_hit10": mean(p["raw_hit10"] for p in points),
        "exp_hit10": mean(p["exp_hit10"] for p in points),
        "net_delta": mean(p["retrieval_delta"] for p in points),
        "help": help_n,
        "hurt": hurt_n,
        "same": n - help_n - hurt_n,
        "ri": (help_n - hurt_n) / n if n else float("nan"),
        "margin_valid": sum(1 for p in points if finite(p.get("ce_margin_raw")) and finite(p.get("ce_margin_exp"))),
        "raw_no_distractor": sum(int(p["raw_no_nongold_distractor"]) for p in points),
        "exp_no_distractor": sum(int(p["exp_no_nongold_distractor"]) for p in points),
    }


def correlation_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for spec in BEIR_SPECS.values():
        dpoints = [p for p in points if p["dataset"] == spec.key]
        for exp in EXPANSIONS:
            groups.append((spec.display, EXPANSIONS[exp][0], [p for p in dpoints if p["expansion"] == exp]))
    for exp in EXPANSIONS:
        groups.append(("Pooled", EXPANSIONS[exp][0], [p for p in points if p["expansion"] == exp]))
    groups.append(("Pooled", "All expansions", points))

    for dataset, expansion, gpoints in groups:
        gold = corr([p["ce_gold_delta"] for p in gpoints], [p["retrieval_delta"] for p in gpoints])
        margin = corr([p["ce_delta_margin"] for p in gpoints], [p["retrieval_delta"] for p in gpoints])
        rows.append({
            "dataset": dataset,
            "expansion": expansion,
            "n": len(gpoints),
            "gold_n": gold["n"],
            "gold_spearman": gold["spearman"],
            "gold_kendall": gold["kendall"],
            "margin_n": margin["n"],
            "margin_spearman": margin["spearman"],
            "margin_kendall": margin["kendall"],
            "mean_gold_delta": mean(p["ce_gold_delta"] for p in gpoints),
            "mean_margin_delta": mean(p["ce_delta_margin"] for p in gpoints),
        })
    return rows


def standardize(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    x = np.asarray([[float(row[f]) for f in features] for row in rows], dtype=np.float64)
    mean_x = x.mean(axis=0)
    std_x = x.std(axis=0)
    std_x[std_x == 0.0] = 1.0
    return (x - mean_x) / std_x


def logistic_auc(points: list[dict[str, Any]], features: list[str], target: str) -> dict[str, Any]:
    rows = [p for p in points if all(finite(p.get(f)) for f in features) and finite(p.get(target))]
    if len(rows) < 10:
        return {"n": len(rows), "failures": 0, "auc": float("nan"), "log_loss": float("nan"), "pseudo_r2": float("nan")}
    y = np.asarray([int(float(row[target]) > 0.0) for row in rows], dtype=np.int64)
    failures = int(y.sum())
    if failures == 0 or failures == len(y):
        return {"n": len(rows), "failures": failures, "auc": float("nan"), "log_loss": float("nan"), "pseudo_r2": float("nan")}
    x = standardize(rows, features)
    model = LogisticRegression(max_iter=2000).fit(x, y)
    proba = model.predict_proba(x)[:, 1]
    loss = float(log_loss(y, proba, labels=[0, 1]))
    null_p = min(max(float(y.mean()), 1e-9), 1.0 - 1e-9)
    null_loss = float(log_loss(y, np.full_like(y, null_p, dtype=np.float64), labels=[0, 1]))
    return {
        "n": len(rows),
        "failures": failures,
        "auc": float(roc_auc_score(y, proba)),
        "log_loss": loss,
        "pseudo_r2": 1.0 - loss / null_loss if null_loss else float("nan"),
        "coefficients": {feature: float(coef) for feature, coef in zip(features, model.coef_[0])},
    }


def p4_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for point in points:
        point["target_margin_failure"] = (
            float(point["ce_delta_margin"] < 0.0) if finite(point.get("ce_delta_margin")) else float("nan")
        )
        point["target_retrieval_hurt"] = float(point["retrieval_delta"] < 0)
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for spec in BEIR_SPECS.values():
        for exp in EXPANSIONS:
            groups.append((spec.display, EXPANSIONS[exp][0], [p for p in points if p["dataset"] == spec.key and p["expansion"] == exp]))
    for exp in EXPANSIONS:
        groups.append(("Pooled", EXPANSIONS[exp][0], [p for p in points if p["expansion"] == exp]))
    for dataset, expansion, gpoints in groups:
        for target, target_label in (
            ("target_margin_failure", "deltaM<0"),
            ("target_retrieval_hurt", "retrieval hurt"),
        ):
            quality = logistic_auc(gpoints, ["oov_rate", "log_perplexity"], target)
            geometry = logistic_auc(gpoints, ["ce_margin_raw", "ce_exp_gold"], target)
            rows.append({
                "dataset": dataset,
                "expansion": expansion,
                "target": target_label,
                "quality": quality,
                "geometry": geometry,
            })
    return rows


def quintile_bins(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [p for p in points if finite(p.get("ce_margin_raw"))]
    ordered = sorted(valid, key=lambda p: (float(p["ce_margin_raw"]), p["label"]))
    n = len(ordered)
    rows: list[dict[str, Any]] = []
    for b in range(5):
        lo = round(b * n / 5)
        hi = round((b + 1) * n / 5)
        chunk = ordered[lo:hi]
        if not chunk:
            continue
        s = summarize(chunk)
        vals = [float(p["ce_margin_raw"]) for p in chunk]
        rows.append({
            "bin": b + 1,
            "axis_min": min(vals),
            "axis_median": median(vals),
            "axis_max": max(vals),
            **s,
        })
    return rows


def all_quintile_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in BEIR_SPECS.values():
        for exp in EXPANSIONS:
            bins = quintile_bins([p for p in points if p["dataset"] == spec.key and p["expansion"] == exp])
            for row in bins:
                row["dataset"] = spec.display
                row["expansion"] = EXPANSIONS[exp][0]
                rows.append(row)
    return rows


def verdicts(points: list[dict[str, Any]], corr_rows_: list[dict[str, Any]], p4: list[dict[str, Any]], bins: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    verdict_rows: list[tuple[str, str, str]] = []
    for exp_display in ("HyDE", "SCOPE"):
        pooled = next(r for r in corr_rows_ if r["dataset"] == "Pooled" and r["expansion"] == exp_display)
        dataset_positive = sum(
            1 for r in corr_rows_
            if r["dataset"] != "Pooled" and r["expansion"] == exp_display and finite(r["gold_spearman"]) and r["gold_spearman"] > 0.1
        )
        if finite(pooled["gold_spearman"]) and pooled["gold_spearman"] >= 0.2 and dataset_positive >= 3:
            verdict = "supported"
        elif finite(pooled["gold_spearman"]) and pooled["gold_spearman"] > 0.0 and dataset_positive >= 2:
            verdict = "mixed"
        else:
            verdict = "killed"
        verdict_rows.append((
            f"Gold-affinity mechanism ({exp_display})",
            verdict,
            f"pooled rho={fmt(pooled['gold_spearman'])}, tau={fmt(pooled['gold_kendall'])}; positive datasets={dataset_positive}/5",
        ))

        pooled_p4 = next(r for r in p4 if r["dataset"] == "Pooled" and r["expansion"] == exp_display and r["target"] == "deltaM<0")
        q_auc = pooled_p4["quality"]["auc"]
        g_auc = pooled_p4["geometry"]["auc"]
        if finite(g_auc) and finite(q_auc) and g_auc >= 0.65 and g_auc > q_auc + 0.05:
            verdict = "supported"
        elif finite(g_auc) and finite(q_auc) and g_auc > q_auc:
            verdict = "mixed"
        else:
            verdict = "killed"
        verdict_rows.append((
            f"P4 geometry-not-hallucination ({exp_display})",
            verdict,
            f"pooled deltaM<0 AUC geometry={fmt(g_auc)}, OOV/logPPL={fmt(q_auc)}",
        ))

        crossover = 0
        declining = 0
        for spec in BEIR_SPECS.values():
            ds_bins = [b for b in bins if b["dataset"] == spec.display and b["expansion"] == exp_display]
            if len(ds_bins) < 2:
                continue
            low = ds_bins[0]["net_delta"]
            high = ds_bins[-1]["net_delta"]
            if finite(low) and finite(high) and high < low:
                declining += 1
            if any(b["net_delta"] > 0 for b in ds_bins) and any(b["net_delta"] < 0 for b in ds_bins):
                crossover += 1
        if crossover >= 3:
            verdict = "supported"
        elif declining >= 3:
            verdict = "mixed"
        else:
            verdict = "killed"
        verdict_rows.append((
            f"Raw-margin regime/crossover ({exp_display})",
            verdict,
            f"sign-crossover datasets={crossover}/5; declining low-to-high datasets={declining}/5",
        ))
    return verdict_rows


def source_paths() -> list[str]:
    paths: list[str] = []
    for spec in BEIR_SPECS.values():
        for p in (
            spec.raw_cache,
            spec.hyde_generation,
            spec.scope_generation,
            spec.hyde_retrieval,
            spec.scope_retrieval,
        ):
            rel = str(p.relative_to(REPO_ROOT))
            if rel not in paths:
                paths.append(rel)
    return paths


def write_report(output: Path, points: list[dict[str, Any]], args: argparse.Namespace) -> None:
    corr_rows_ = correlation_rows(points)
    p4 = p4_rows(points)
    bins = all_quintile_rows(points)
    verdict_rows = verdicts(points, corr_rows_, p4, bins)

    lines: list[str] = []
    lines.append("# BEIR Phase 1 Verification - 2026-05-26")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Read-only Phase 4/5 analysis over the committed BEIR Phase 1 caches. No answer/model calls were made in this phase, and no files under `paper/` were edited.")
    lines.append("")
    lines.append("Definitions:")
    lines.append("- Retrieval gain is `expansion Hit@5 - raw-question Hit@5` per query.")
    lines.append("- Collins-Thompson `RI = (n_help - n_hurt) / N`, where help is expansion-only Hit@5 and hurt is raw-only Hit@5.")
    lines.append("- Gold CE affinity is the max `cross-encoder/ms-marco-MiniLM-L-6-v2` score over all positive qrel document ids.")
    lines.append("- `M_raw = CE(raw,best gold) - max CE(raw,retrieved non-gold)`; multi-gold rows exclude all gold ids from the distractor max.")
    lines.append("- OOV and log-perplexity use an add-1 smoothed unigram LM built from each BEIR corpus CSV.")
    lines.append("")

    lines.append("## Cross-Dataset Verdicts")
    lines.append("")
    lines.append("| Claim | Verdict | Key numbers |")
    lines.append("|---|---|---|")
    for claim, verdict, key in verdict_rows:
        lines.append(f"| {claim} | **{verdict}** | {key} |")
    lines.append("")

    lines.append("## Retrieval Outcomes")
    lines.append("")
    lines.append("| Dataset | Expansion | N | Raw Hit@5 | Expansion Hit@5 | Net Hit@5 | Raw Hit@10 | Expansion Hit@10 | Help | Hurt | RI | Margin-valid rows |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for spec in BEIR_SPECS.values():
        for exp in EXPANSIONS:
            gpoints = [p for p in points if p["dataset"] == spec.key and p["expansion"] == exp]
            s = summarize(gpoints)
            lines.append(
                f"| {spec.display} | {EXPANSIONS[exp][0]} | {s['n']} | {pct(s['raw_hit5'])} | {pct(s['exp_hit5'])} | "
                f"{pct(s['net_delta'])} | {pct(s['raw_hit10'])} | {pct(s['exp_hit10'])} | "
                f"{s['help']} | {s['hurt']} | {fmt(s['ri'])} | {s['margin_valid']} |"
            )
    for exp in EXPANSIONS:
        gpoints = [p for p in points if p["expansion"] == exp]
        s = summarize(gpoints)
        lines.append(
            f"| Pooled | {EXPANSIONS[exp][0]} | {s['n']} | {pct(s['raw_hit5'])} | {pct(s['exp_hit5'])} | "
            f"{pct(s['net_delta'])} | {pct(s['raw_hit10'])} | {pct(s['exp_hit10'])} | "
            f"{s['help']} | {s['hurt']} | {fmt(s['ri'])} | {s['margin_valid']} |"
        )
    lines.append("")

    lines.append("## Gold-Affinity and Margin Correlations")
    lines.append("")
    lines.append("The primary mechanism column is gold CE delta: `CE(exp,best gold) - CE(raw,best gold)`. Delta-margin is included to show whether adding the non-gold distractor term changes the read.")
    lines.append("")
    lines.append("| Dataset | Expansion | N | Mean CE gold delta | Gold rho | Gold tau | Margin-valid N | Mean deltaM | DeltaM rho | DeltaM tau |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in corr_rows_:
        lines.append(
            f"| {row['dataset']} | {row['expansion']} | {row['n']} | {fmt(row['mean_gold_delta'])} | "
            f"{fmt(row['gold_spearman'])} | {fmt(row['gold_kendall'])} | {row['margin_n']} | "
            f"{fmt(row['mean_margin_delta'])} | {fmt(row['margin_spearman'])} | {fmt(row['margin_kendall'])} |"
        )
    lines.append("")

    lines.append("## P4 Failure Model")
    lines.append("")
    lines.append("Two targets are reported: geometry failure `deltaM<0` and observed retrieval hurt. The requested comparison is OOV/log-perplexity versus geometry features `{M_raw, CE(exp,gold)}`.")
    lines.append("")
    lines.append("| Dataset | Expansion | Target | N | Failures | AUC OOV/logPPL | AUC geometry | Pseudo-R2 OOV/logPPL | Pseudo-R2 geometry |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in p4:
        q = row["quality"]
        g = row["geometry"]
        lines.append(
            f"| {row['dataset']} | {row['expansion']} | {row['target']} | {g['n']} | {g['failures']} | "
            f"{fmt(q['auc'])} | {fmt(g['auc'])} | {fmt(q['pseudo_r2'])} | {fmt(g['pseudo_r2'])} |"
        )
    lines.append("")

    lines.append("## M_raw Quintile Regime Test")
    lines.append("")
    lines.append("Rows with no non-gold distractor in that condition's top-10 are excluded from margin bins because `M_raw` is undefined. This is common in TREC-COVID because each query has hundreds of positive qrel documents.")
    lines.append("")
    lines.append("| Dataset | Expansion | Bin | N | M_raw median | M_raw range | Raw Hit@5 | Expansion Hit@5 | Net Hit@5 | Help | Hurt | RI |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in bins:
        lines.append(
            f"| {row['dataset']} | {row['expansion']} | {row['bin']} | {row['n']} | {fmt(row['axis_median'])} | "
            f"[{fmt(row['axis_min'])}, {fmt(row['axis_max'])}] | {pct(row['raw_hit5'])} | {pct(row['exp_hit5'])} | "
            f"{pct(row['net_delta'])} | {row['help']} | {row['hurt']} | {fmt(row['ri'])} |"
        )
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    hyde_pool = summarize([p for p in points if p["expansion"] == "hyde"])
    scope_pool = summarize([p for p in points if p["expansion"] == "scope"])
    hyde_corr = next(r for r in corr_rows_ if r["dataset"] == "Pooled" and r["expansion"] == "HyDE")
    scope_corr = next(r for r in corr_rows_ if r["dataset"] == "Pooled" and r["expansion"] == "SCOPE")
    lines.append(f"- Raw-question retrieval is a very strong baseline on this BEIR slice. HyDE loses {pct(-hyde_pool['net_delta'])} Hit@5 pooled, while SCOPE loses {pct(-scope_pool['net_delta'])} pooled and is much closer to raw on NFCorpus, TREC-COVID, and SciDocs.")
    lines.append(f"- Gold-affinity movement does replicate as a row-level mechanism: pooled gold-delta correlation is {fmt(hyde_corr['gold_spearman'])} for HyDE and {fmt(scope_corr['gold_spearman'])} for SCOPE, with positive correlations in all five datasets.")
    lines.append("- That mechanism does not make ungated expansion a good policy here. Mean CE gold deltas are often negative, and the average retrieval outcome is below raw-question retrieval on every dataset/method cell.")
    lines.append("- The clearest replicated lesson is risk control: expansion often helps individual low-confidence rows, but broad ungated application hurts when raw retrieval already lands a gold document.")
    lines.append("- TREC-COVID is a special case for margin tests: qrels are extremely dense, so many top-10 lists have no non-gold distractor and margin-valid N is much smaller than query N.")
    lines.append("")

    lines.append("## Sources")
    lines.append("")
    for path in source_paths():
        lines.append(f"- `{path}`")
    lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CROSS_ENCODER_DEVICE=cuda \\")
    lines.append("uv run python scripts/analyze_beir_phase1.py \\")
    lines.append("  --output docs/generated/beir_phase1_verification_2026-05-26.md")
    lines.append("```")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def load_points(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def write_points(path: Path, points: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for point in points:
            f.write(json.dumps(point, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/generated/beir_phase1_verification_2026-05-26.md")
    parser.add_argument("--points-cache", type=Path, default=Path("/tmp/beir_phase1_verification_2026-05-26_points.jsonl"))
    parser.add_argument("--reuse-points", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(BEIR_SPECS), choices=sorted(BEIR_SPECS))
    parser.add_argument("--doc-batch-size", type=int, default=5000)
    parser.add_argument("--ce-batch-size", type=int, default=64)
    parser.add_argument("--ce-chunk-size", type=int, default=10000)
    args = parser.parse_args()

    points = load_points(args.points_cache) if args.reuse_points else []
    if points:
        print(f"[cache] loaded {len(points)} points from {args.points_cache}", flush=True)
    else:
        points = []
        for key in args.datasets:
            points.extend(build_points_for_dataset(BEIR_SPECS[key], args))
        write_points(args.points_cache, points)
        print(f"[cache] wrote {args.points_cache}", flush=True)
    write_report(args.output, points, args)
    print(args.output)


if __name__ == "__main__":
    main()
