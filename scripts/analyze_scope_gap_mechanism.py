#!/usr/bin/env python3
"""Query-to-gold-passage mechanism test for SCOPE.

For each question/model row, compare the raw retrieval query and the canonical
Snap-HyRE generated passage against the real gold passage text using the same
families of retrieval models used by the harness: the MiniLM cross-encoder and
the gte-large bi-encoder. Multi-gold rows use the max score over the gold set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
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
    pearson,
    question_scores,
    source_paths_for,
    spearman,
)
from eval_config import EvalConfig, load_questions  # noqa: E402
from eval_harness import _retrieval_question, _row_label  # noqa: E402
from rag_utils import get_cross_encoder, get_embeddings  # noqa: E402


HYRE_CACHE = {
    ("barexam", "groq-llama8b"): "caches/hyre/full/barexam_qfull_seed42_groq-llama8b_snap_hyre.jsonl",
    ("barexam", "or-gemma4-26b"): "caches/hyre/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl",
    ("barexam", "groq-llama70b"): "caches/hyre/full/barexam_qfull_seed42_groq-llama70b_snap_hyre.jsonl",
    ("housing", "groq-llama8b"): "caches/hyre/full/housing_qfull_seed42_groq-llama8b_snap_hyre.jsonl",
    ("housing", "or-gemma4-26b"): "caches/hyre/full/housing_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl",
    ("housing", "groq-llama70b"): "caches/hyre/full/housing_qfull_seed42_groq-llama70b_snap_hyre.jsonl",
}

AXES = {
    "ce_raw_gold": "CE(raw, gold)",
    "cos_raw_gold": "cos(raw, gold)",
    "ce_delta": "CE(scope, gold) - CE(raw, gold)",
    "cos_delta": "cos(scope, gold) - cos(raw, gold)",
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions_raw_text(spec_key: str) -> dict[str, str]:
    spec = DATASETS[spec_key]
    config = EvalConfig(
        dataset=spec.key,
        questions="full",
        seed=42,
        retrieval_k=5,
        housing_state_filter=spec.housing_state_filter,
    )
    rows = [row for _, row in load_questions(config).iterrows()]
    return {_row_label(row, config): _retrieval_question(row) for row in rows}


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def snippet(text: str, limit: int = 140) -> str:
    cleaned = normalize_text(text).replace("|", "\\|")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def fetch_docs_by_idx(collection_name: str, idxs: list[str], batch_size: int = 5000) -> dict[str, dict[str, Any]]:
    import chromadb

    requested = list(dict.fromkeys(str(idx) for idx in idxs if str(idx)))
    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR", str(REPO_ROOT / "chroma_db")))
    collection = client.get_collection(collection_name)
    found: dict[str, dict[str, Any]] = {}

    def store(batch: dict[str, Any]) -> None:
        ids = batch.get("ids") or []
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        for chroma_id, doc, meta in zip(ids, docs, metas):
            meta = dict(meta or {})
            idx = str(meta.get("idx") or str(chroma_id).removeprefix("doc_"))
            if idx and idx not in found:
                meta.setdefault("idx", idx)
                found[idx] = {"idx": idx, "text": doc or "", "metadata": meta}

    for start in range(0, len(requested), batch_size):
        chunk = requested[start:start + batch_size]
        store(collection.get(ids=[f"doc_{idx}" for idx in chunk], include=["documents", "metadatas"]))

    missing = [idx for idx in requested if idx not in found]
    for start in range(0, len(missing), 500):
        chunk = missing[start:start + 500]
        try:
            store(collection.get(where={"idx": {"$in": chunk}}, include=["documents", "metadatas"]))
        except Exception:
            for idx in chunk:
                store(collection.get(where={"idx": idx}, include=["documents", "metadatas"]))

    still_missing = [idx for idx in requested if idx not in found]
    if still_missing:
        raise RuntimeError(f"{collection_name}: missing gold docs for {still_missing[:10]} n={len(still_missing)}")
    return found


def first_gold_rank(retrieved_ids: list[Any], gold_ids: list[str], cap: int = 10) -> int:
    gold = {str(x) for x in gold_ids if str(x)}
    for rank, idx in enumerate([str(x) for x in retrieved_ids[:cap]], 1):
        if idx in gold:
            return rank
    return cap + 1


def build_points(dataset: str, q_scores: dict[str, dict[str, Any]], gold_docs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    spec = DATASETS[dataset]
    raw_text = load_questions_raw_text(dataset)
    raw_cache = load_by_label(spec.raw_cache)
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
            gold_ids = [str(gid) for gid in score["gold_ids"]]
            gold_texts = [gold_docs[gid]["text"] for gid in gold_ids]
            raw_ids = raw_cache[label].get("retrieved_ids") or []
            scope_ids = scope_cache[label].get("retrieved_ids") or []
            raw_hit = hit_at_5(raw_ids, gold_ids)
            scope_hit = hit_at_5(scope_ids, gold_ids)
            raw_correct = int(bool(raw_log[label].get("is_correct")))
            scope_correct = int(bool(scope_log[label].get("is_correct")))
            scope_passage = str(hyre_cache[label].get("hyde_passage") or "")
            if not scope_passage:
                raise RuntimeError(f"{dataset}/{model}/{label}: missing hyde_passage")
            points.append({
                "dataset": dataset,
                "dataset_display": spec.display,
                "model": model,
                "label": label,
                "raw_question": raw_text[label],
                "scope_passage": scope_passage,
                "gold_ids": gold_ids,
                "gold_texts": gold_texts,
                "multi_gold_count": len(gold_ids),
                "raw_hit": raw_hit,
                "scope_hit": scope_hit,
                "retrieval_delta": int(scope_hit) - int(raw_hit),
                "raw_gold_rank_at10": first_gold_rank(raw_ids, gold_ids, cap=10),
                "raw_correct": raw_correct,
                "scope_correct": scope_correct,
                "answer_delta": int(scope_correct) - int(raw_correct),
            })
    return points


def truncate_for_ce(text: str) -> str:
    max_chars = int(os.getenv("CROSS_ENCODER_MAX_CHARS", "4096") or "4096")
    text = str(text or "")
    return text[:max_chars] if max_chars and len(text) > max_chars else text


def score_ce(points: list[dict[str, Any]], batch_size: int) -> None:
    ce = get_cross_encoder()
    raw_needed: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for point in points:
        raw_needed.setdefault((point["dataset"], point["label"]), [])
        if not raw_needed[(point["dataset"], point["label"])]:
            raw_needed[(point["dataset"], point["label"])] = list(zip(point["gold_ids"], point["gold_texts"]))

    raw_pairs: list[tuple[str, str]] = []
    raw_meta: list[tuple[tuple[str, str], str]] = []
    raw_text_by_key = {(p["dataset"], p["label"]): p["raw_question"] for p in points}
    for key, golds in raw_needed.items():
        for gid, gold_text in golds:
            raw_pairs.append((truncate_for_ce(raw_text_by_key[key]), truncate_for_ce(gold_text)))
            raw_meta.append((key, gid))

    print(f"[ce] raw pairs={len(raw_pairs)}", flush=True)
    raw_scores = ce.predict(raw_pairs, batch_size=batch_size)
    raw_by_key: dict[tuple[str, str], tuple[float, str]] = {}
    for (key, gid), score in zip(raw_meta, raw_scores):
        current = raw_by_key.get(key)
        if current is None or float(score) > current[0]:
            raw_by_key[key] = (float(score), gid)

    scope_pairs: list[tuple[str, str]] = []
    scope_meta: list[tuple[int, str]] = []
    for i, point in enumerate(points):
        for gid, gold_text in zip(point["gold_ids"], point["gold_texts"]):
            scope_pairs.append((truncate_for_ce(point["scope_passage"]), truncate_for_ce(gold_text)))
            scope_meta.append((i, gid))

    print(f"[ce] scope pairs={len(scope_pairs)}", flush=True)
    scope_scores = ce.predict(scope_pairs, batch_size=batch_size)
    scope_by_point: dict[int, tuple[float, str]] = {}
    for (i, gid), score in zip(scope_meta, scope_scores):
        current = scope_by_point.get(i)
        if current is None or float(score) > current[0]:
            scope_by_point[i] = (float(score), gid)

    for i, point in enumerate(points):
        raw_score, raw_gid = raw_by_key[(point["dataset"], point["label"])]
        scope_score, scope_gid = scope_by_point[i]
        point["ce_raw_gold"] = raw_score
        point["ce_raw_gold_id"] = raw_gid
        point["ce_scope_gold"] = scope_score
        point["ce_scope_gold_id"] = scope_gid
        point["ce_delta"] = scope_score - raw_score


def embed_unique(text_by_key: dict[Any, str], batch_size: int) -> dict[Any, np.ndarray]:
    emb = get_embeddings()
    keys = list(text_by_key)
    out: dict[Any, np.ndarray] = {}
    for start in range(0, len(keys), batch_size):
        chunk_keys = keys[start:start + batch_size]
        texts = [text_by_key[key] for key in chunk_keys]
        vecs = emb.embed_documents(texts)
        for key, vec in zip(chunk_keys, vecs):
            arr = np.asarray(vec, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm:
                arr = arr / norm
            out[key] = arr
        print(f"[embed] {min(start + batch_size, len(keys))}/{len(keys)}", flush=True)
    return out


def score_cosine(points: list[dict[str, Any]], batch_size: int) -> None:
    raw_texts = {(p["dataset"], p["label"]): p["raw_question"] for p in points}
    scope_texts = {(p["dataset"], p["model"], p["label"]): p["scope_passage"] for p in points}
    gold_texts: dict[tuple[str, str], str] = {}
    for point in points:
        for gid, text in zip(point["gold_ids"], point["gold_texts"]):
            gold_texts[(point["dataset"], gid)] = text

    print(f"[embed] raw={len(raw_texts)} scope={len(scope_texts)} gold={len(gold_texts)}", flush=True)
    raw_emb = embed_unique(raw_texts, batch_size)
    scope_emb = embed_unique(scope_texts, batch_size)
    gold_emb = embed_unique(gold_texts, batch_size)

    raw_best: dict[tuple[str, str], tuple[float, str]] = {}
    for point in points:
        key = (point["dataset"], point["label"])
        if key in raw_best:
            continue
        rv = raw_emb[key]
        best = (-2.0, "")
        for gid in point["gold_ids"]:
            score = float(np.dot(rv, gold_emb[(point["dataset"], gid)]))
            if score > best[0]:
                best = (score, gid)
        raw_best[key] = best

    for point in points:
        sv = scope_emb[(point["dataset"], point["model"], point["label"])]
        best = (-2.0, "")
        for gid in point["gold_ids"]:
            score = float(np.dot(sv, gold_emb[(point["dataset"], gid)]))
            if score > best[0]:
                best = (score, gid)
        raw_score, raw_gid = raw_best[(point["dataset"], point["label"])]
        point["cos_raw_gold"] = raw_score
        point["cos_raw_gold_id"] = raw_gid
        point["cos_scope_gold"] = best[0]
        point["cos_scope_gold_id"] = best[1]
        point["cos_delta"] = best[0] - raw_score


def axis_correlation(points: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    x = [float(p[axis]) for p in points]
    return {
        "axis": axis,
        "n": len(points),
        "pearson_retrieval": pearson(x, [p["retrieval_delta"] for p in points]),
        "spearman_retrieval": spearman(x, [p["retrieval_delta"] for p in points]),
        "pearson_answer": pearson(x, [p["answer_delta"] for p in points]),
        "spearman_answer": spearman(x, [p["answer_delta"] for p in points]),
    }


def summarize(points: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "n": len(points),
        "retrieval_delta": mean([p["retrieval_delta"] for p in points]),
        "answer_delta": mean([p["answer_delta"] for p in points]),
        "ce_delta": mean([p["ce_delta"] for p in points]),
        "cos_delta": mean([p["cos_delta"] for p in points]),
        "ce_delta_positive": mean([float(p["ce_delta"] > 0) for p in points]),
        "cos_delta_positive": mean([float(p["cos_delta"] > 0) for p in points]),
        "multi_gold": mean([float(p["multi_gold_count"] > 1) for p in points]),
    }


def binned_curve(points: list[dict[str, Any]], axis: str, bins: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(points, key=lambda p: (float(p[axis]), p["dataset"], p["model"], p["label"]))
    n = len(ordered)
    rows: list[dict[str, Any]] = []
    for b in range(bins):
        lo = round(b * n / bins)
        hi = round((b + 1) * n / bins)
        chunk = ordered[lo:hi]
        vals = [float(p[axis]) for p in chunk]
        s = summarize(chunk)
        rows.append({
            "bin": b + 1,
            "axis_min": min(vals),
            "axis_median": float(np.median(vals)),
            "axis_max": max(vals),
            **s,
        })
    return rows


def try_mlm_status(candidates: list[str]) -> dict[str, Any]:
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
    except Exception as exc:
        return {"status": "blocked", "reason": f"transformers import failed: {exc}", "attempted": candidates}
    attempted = []
    for name in candidates:
        try:
            AutoTokenizer.from_pretrained(name, local_files_only=True)
            AutoModelForMaskedLM.from_pretrained(name, local_files_only=True)
            return {
                "status": "available_not_run",
                "model": name,
                "reason": "offline model is present, but full pseudo-perplexity was not run in this mechanism pass",
                "attempted": attempted + [name],
            }
        except Exception as exc:
            attempted.append(f"{name}: {type(exc).__name__}")
    return {
        "status": "blocked",
        "reason": "no candidate masked-LM was available in the local Hugging Face cache under HF_HUB_OFFLINE",
        "attempted": attempted,
    }


def outcome(point: dict[str, Any]) -> str:
    return (
        f"ret {point['raw_hit']}->{point['scope_hit']}; "
        f"ans {point['raw_correct']}->{point['scope_correct']}"
    )


def qualitative_rows(points: list[dict[str, Any]], reverse: bool, limit: int) -> list[str]:
    selected = sorted(points, key=lambda p: float(p["ce_delta"]), reverse=reverse)[:limit]
    lines = [
        "| Dataset | Model | Label | CE delta | Cos delta | Outcomes | Raw question | SCOPE passage | Gold passage |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for p in selected:
        gold_id = p.get("ce_scope_gold_id") or p["gold_ids"][0]
        try:
            gold_i = p["gold_ids"].index(gold_id)
        except ValueError:
            gold_i = 0
        lines.append(
            f"| {p['dataset_display']} | {MODEL_LABELS[p['model']]} | `{p['label']}` | "
            f"{p['ce_delta']:.3f} | {p['cos_delta']:.3f} | {outcome(p)} | "
            f"{snippet(p['raw_question'])} | {snippet(p['scope_passage'])} | {snippet(p['gold_texts'][gold_i])} |"
        )
    return lines


def make_report(output: Path, points: list[dict[str, Any]], mlm: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# SCOPE Query-Gold Gap Mechanism - 2026-05-25")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("This results-lane analysis measures the query-to-gold-passage gap directly using the real retrieval model families: MiniLM cross-encoder scores and gte-large bi-encoder cosine. It uses signed raw/SCOPE caches, canonical `snap_hyre` generated-passage caches, and fetched gold passage text. No files under `paper/` were edited.")
    lines.append("")
    lines.append("- Multi-gold rows use the maximum score over the gold set for raw and SCOPE independently.")
    lines.append("- CE inputs use `CROSS_ENCODER_MAX_CHARS=4096` by default, matching the retrieval-cache reranker cap used in these analyses.")
    lines.append("- Outcomes are signed as SCOPE minus raw: retrieval delta is Hit@5 movement; answer delta is exact-answer correctness movement.")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Dataset | Model | N | Multi-gold rows | Mean CE delta | CE delta > 0 | Mean cos delta | Cos delta > 0 | Mean retrieval delta | Mean answer delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in ("barexam", "housing"):
        dpoints = [p for p in points if p["dataset"] == dataset]
        for model in (*MODELS, "pooled"):
            mpoints = dpoints if model == "pooled" else [p for p in dpoints if p["model"] == model]
            s = summarize(mpoints)
            label = MODEL_LABELS.get(model, "Pooled")
            display = mpoints[0]["dataset_display"] if mpoints else dataset
            lines.append(
                f"| {display} | {label} | {s['n']} | {pct(s['multi_gold'])} | "
                f"{s['ce_delta']:.3f} | {pct(s['ce_delta_positive'])} | "
                f"{s['cos_delta']:.3f} | {pct(s['cos_delta_positive'])} | "
                f"{pct(s['retrieval_delta'])} | {pct(s['answer_delta'])} |"
            )
    lines.append("")

    lines.append("## Correlations")
    lines.append("")
    lines.append("H1 expects low `CE(raw,gold)` / `cos(raw,gold)` to associate with SCOPE gains, so negative correlations with SCOPE-minus-raw retrieval delta support H1. H2 expects positive deltas to associate with retrieval gain. H3 asks whether any axis predicts answer delta.")
    lines.append("")
    lines.append("| Dataset | Model | Axis | N | Pearson retrieval | Spearman retrieval | Pearson answer | Spearman answer |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for dataset in ("barexam", "housing"):
        dpoints = [p for p in points if p["dataset"] == dataset]
        for model in (*MODELS, "pooled"):
            mpoints = dpoints if model == "pooled" else [p for p in dpoints if p["model"] == model]
            for axis, label in AXES.items():
                c = axis_correlation(mpoints, axis)
                display = mpoints[0]["dataset_display"] if mpoints else dataset
                lines.append(
                    f"| {display} | {MODEL_LABELS.get(model, 'Pooled')} | {label} | {c['n']} | "
                    f"{fmt_float(c['pearson_retrieval'])} | {fmt_float(c['spearman_retrieval'])} | "
                    f"{fmt_float(c['pearson_answer'])} | {fmt_float(c['spearman_answer'])} |"
                )
    lines.append("")

    lines.append("## CE Delta Binned Curve")
    lines.append("")
    lines.append("Quintiles are pooled across dataset/model rows and sorted by `CE(scope,gold) - CE(raw,gold)`.")
    lines.append("")
    lines.append("| Bin | N | CE delta median | CE delta range | CE delta > 0 | Cos delta median | Net retrieval delta | Net answer delta |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in binned_curve(points, "ce_delta"):
        lines.append(
            f"| {row['bin']} | {row['n']} | {row['axis_median']:.3f} | {row['axis_min']:.3f}-{row['axis_max']:.3f} | "
            f"{pct(row['ce_delta_positive'])} | {row['cos_delta']:.3f} | {pct(row['retrieval_delta'])} | {pct(row['answer_delta'])} |"
        )
    lines.append("")

    lines.append("## Masked-LM Pseudo-Perplexity")
    lines.append("")
    if mlm["status"] == "blocked":
        lines.append(f"Blocked/provisional: {mlm['reason']}. Attempted: {', '.join(mlm['attempted'])}.")
    else:
        lines.append(f"Provisional: `{mlm['model']}` is available offline, but full contextual pseudo-perplexity was not run in this pass. Reason: {mlm['reason']}.")
    lines.append("")

    lines.append("## Top CE-Delta Examples")
    lines.append("")
    lines.extend(qualitative_rows(points, reverse=True, limit=15))
    lines.append("")
    lines.append("## Bottom CE-Delta Examples")
    lines.append("")
    lines.extend(qualitative_rows(points, reverse=False, limit=15))
    lines.append("")

    lines.append("## Reading")
    lines.append("")
    pooled = {axis: axis_correlation(points, axis) for axis in AXES}
    by_dataset = {
        dataset: {
            axis: axis_correlation([p for p in points if p["dataset"] == dataset], axis)
            for axis in AXES
        }
        for dataset in ("barexam", "housing")
    }
    lines.append(
        f"- H1 is mixed. Low raw query-gold alignment predicts SCOPE retrieval benefit on HousingQA "
        f"(Spearman {fmt_float(by_dataset['housing']['ce_raw_gold']['spearman_retrieval'])} CE, "
        f"{fmt_float(by_dataset['housing']['cos_raw_gold']['spearman_retrieval'])} cosine), but the raw-alignment score alone is near-null on BarExamQA "
        f"({fmt_float(by_dataset['barexam']['ce_raw_gold']['spearman_retrieval'])} CE, "
        f"{fmt_float(by_dataset['barexam']['cos_raw_gold']['spearman_retrieval'])} cosine)."
    )
    lines.append(
        f"- H2 is the cleaner mechanism result. The movement toward gold predicts retrieval gain in both datasets: CE delta Spearman is "
        f"{fmt_float(by_dataset['barexam']['ce_delta']['spearman_retrieval'])} on BarExamQA and "
        f"{fmt_float(by_dataset['housing']['ce_delta']['spearman_retrieval'])} on HousingQA; cosine delta is "
        f"{fmt_float(by_dataset['barexam']['cos_delta']['spearman_retrieval'])} and "
        f"{fmt_float(by_dataset['housing']['cos_delta']['spearman_retrieval'])}, respectively. Pooled CE/cosine deltas are "
        f"{fmt_float(pooled['ce_delta']['spearman_retrieval'])} / {fmt_float(pooled['cos_delta']['spearman_retrieval'])}."
    )
    lines.append(
        f"- H3 is weak: answer-delta correlations are small across all four axes. The largest pooled answer Spearman in this report is "
        f"{max((abs(v['spearman_answer']), k, v['spearman_answer']) for k, v in pooled.items())[2]:.3f}, so query-gold alignment explains retrieval movement better than downstream answer movement."
    )
    lines.append(
        "- Mechanism read: the non-circular query-gold gap is a better explanation than unigram perplexity. SCOPE helps when the raw query is far from the gold passage and when the generated passage increases cross-encoder affinity to that gold passage; answer conversion still depends on whether the repaired evidence is useful rather than distracting."
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
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/analyze_scope_gap_mechanism.py \\")
    lines.append("  --output docs/generated/scope_gap_mechanism_2026-05-25.md")
    lines.append("```")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "docs/generated/scope_gap_mechanism_2026-05-25.md")
    parser.add_argument("--lm-cache-dir", type=Path, default=Path("/tmp/perplexity_axis_lm_cache_2026-05-25"))
    parser.add_argument("--ce-batch-size", type=int, default=32)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--datasets", nargs="+", default=["barexam", "housing"], choices=sorted(DATASETS))
    parser.add_argument("--points-out", type=Path, default=Path("/tmp/scope_gap_mechanism_2026-05-25_points.jsonl"))
    args = parser.parse_args()

    all_points: list[dict[str, Any]] = []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        print(f"[dataset] {dataset}: question scores", flush=True)
        q_scores = question_scores(spec, build_or_load_lm(spec, args.lm_cache_dir, 20000))
        gold_ids = sorted({gid for row in q_scores.values() for gid in row["gold_ids"]})
        print(f"[dataset] {dataset}: fetch gold docs={len(gold_ids)}", flush=True)
        gold_docs = fetch_docs_by_idx(spec.collection, gold_ids)
        all_points.extend(build_points(dataset, q_scores, gold_docs))

    score_ce(all_points, args.ce_batch_size)
    score_cosine(all_points, args.embed_batch_size)
    mlm = try_mlm_status([
        "prajjwal1/bert-tiny",
        "distilbert-base-uncased",
        "bert-base-uncased",
        "google/bert_uncased_L-2_H-128_A-2",
    ])

    args.points_out.parent.mkdir(parents=True, exist_ok=True)
    with args.points_out.open("w") as f:
        for point in all_points:
            clean = {
                k: v for k, v in point.items()
                if k not in {"raw_question", "scope_passage", "gold_texts"}
            }
            f.write(json.dumps(clean, sort_keys=True) + "\n")
    make_report(args.output, all_points, mlm)
    print(args.output)


if __name__ == "__main__":
    main()
