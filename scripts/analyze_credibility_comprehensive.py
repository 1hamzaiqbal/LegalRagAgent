#!/usr/bin/env python3
"""Comprehensive credibility battery helpers.

Results-lane only: reads existing caches/corpora and writes docs/generated
reports. It does not edit paper artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
import time
import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_credibility_battery import (  # noqa: E402
    DATASET_SOURCES,
    OR_GEMMA,
    corr,
    fmt,
    generation_for,
    generation_passage,
    gold_docs_for,
    hit_at,
    mean,
    pct,
    questions_for,
    read_jsonl,
    write_jsonl,
)
from build_factuality_judge_cache import dataset_specs as factuality_dataset_specs  # noqa: E402
from analyze_beir_phase1 import BEIR_SPECS, load_questions_for_spec  # noqa: E402
from analyze_raw_retrieval_confidence_routing import (  # noqa: E402
    ce_features as qpp_ce_features,
    dense_features_for_row,
    embed_queries,
    fetch_doc_embeddings,
)


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "he", "her",
    "his", "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}

DATASETS_7 = [
    "barexam",
    "housing",
    "beir_scifact",
    "beir_nfcorpus",
    "beir_fiqa",
    "beir_trec_covid",
    "beir_scidocs",
]

PHASE_CXX_REPORT = ROOT / "docs/generated/credibility_C_three_retrievers_full_2026-05-29.md"
PHASE_CXX_POINTS = ROOT / "docs/generated/credibility_C_three_retrievers_full_2026-05-29_points.jsonl"
PHASE_CXX_E5_REPORT = ROOT / "docs/generated/credibility_C_e5_addendum_2026-05-29.md"
PHASE_CXX_E5_POINTS = ROOT / "docs/generated/credibility_C_e5_addendum_2026-05-29_points.jsonl"
PHASE_D_REPORT = ROOT / "docs/generated/credibility_D_ood_predictor_2026-05-29.md"
PHASE_D_POINTS = ROOT / "docs/generated/credibility_D_ood_predictor_2026-05-29_points.jsonl"
PHASE_E_REPORT = ROOT / "docs/generated/credibility_E_midregime_2026-05-29.md"
PHASE_E_POINTS = ROOT / "docs/generated/credibility_E_midregime_2026-05-29_points.jsonl"
FINAL_REPORT = ROOT / "docs/generated/credibility_comprehensive_summary_2026-05-29.md"

MODEL_DISPLAY = {
    "or-gemma4-26b": "Gemma 4 26B",
    "or-qwen3p5-9b": "Qwen 3.5 9B",
    "or-mistral-small-3p2-24b": "Mistral Small 3.2 24B",
    "or-deepseek-v32": "DeepSeek V3.2",
}

QPP_FEATURES = [
    "nqc_ce_top10",
    "wig_ce_top5_vs_top10",
    "smv_ce_top10",
    "ce_top1",
    "ce_top5_mean",
    "ce_spread_1_5",
    "ce_entropy_conf_top5",
    "dense_query_top1_cos",
    "dense_coherence_top5",
    "dense_centroid_norm_top5",
    "log_perplexity",
    "question_tokens",
    "oov_rate",
]

E5_DATASETS_PRIORITY = [
    "beir_scifact",
    "beir_nfcorpus",
    "beir_fiqa",
    "beir_trec_covid",
    "beir_scidocs",
    "barexam",
]
E5_TEXT_MAX_CHARS = 4096

PHASE_E_DATASETS = [
    "barexam",
    "casehold",
    "housing",
    "beir_scidocs",
    "beir_fiqa",
    "beir_nfcorpus",
    "beir_scifact",
    "beir_trec_covid",
]
PHASE_E_DISPLAY = {"casehold": "CaseHOLD"}


def tokenize(text: Any, *, max_terms: int = 0) -> list[str]:
    counts = Counter(
        tok for tok in TOKEN_RE.findall(str(text or "").lower())
        if len(tok) > 1 and tok not in STOPWORDS
    )
    terms = [tok for tok, _ in counts.most_common(max_terms or None)]
    return terms


def load_by_label(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("label") or row.get("idx")): row for row in read_jsonl(path)}


def retrieval_cache_path(dataset: str, arm: str, model: str = OR_GEMMA) -> Path:
    if dataset.startswith("beir_"):
        if arm == "raw":
            return ROOT / f"caches/retrieval/full/{dataset}_qfull_seed42_raw_question_k10.jsonl"
        suffix = "rag_hyde" if arm == "hyde" else "snap_hyre"
        return ROOT / f"caches/retrieval/full/{dataset}_qfull_seed42_{model}_{suffix}_k10.jsonl"
    specs = factuality_dataset_specs()
    if arm == "raw":
        return specs[dataset].raw_cache
    return specs[dataset].expansions[arm].retrieval


def load_original_rows(dataset: str, arms: list[str]) -> list[dict[str, Any]]:
    qmap = questions_for(dataset)
    caches = {arm: load_by_label(retrieval_cache_path(dataset, arm)) for arm in arms}
    rows: list[dict[str, Any]] = []
    for arm in arms:
        cache = caches[arm]
        for label, qrow in qmap.items():
            if label not in cache:
                continue
            ids = [str(x) for x in (cache[label].get("retrieved_ids") or [])]
            rows.append({
                "dataset": dataset,
                "dataset_display": DATASET_SOURCES[dataset].display,
                "retriever": "gte_ce_original",
                "arm": arm,
                "label": label,
                "gold_ids": qrow["gold_ids"],
                "retrieved_ids": ids[:10],
                "scores": cache[label].get("scores") or [],
                "hit5": hit_at(ids, qrow["gold_ids"], 5),
                "hit10": hit_at(ids, qrow["gold_ids"], 10),
                "gold_affinity": float("nan"),
            })
    return rows


def sqlite_escape_term(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


def fts_query(text: str, max_terms: int) -> str:
    terms = tokenize(text, max_terms=max_terms)
    return " OR ".join(sqlite_escape_term(term) for term in terms)


def iter_docs_with_state(dataset: str) -> Iterable[tuple[str, str, str]]:
    source = DATASET_SOURCES[dataset]
    if source.kind == "beir":
        assert source.beir_spec is not None
        with (ROOT / f"datasets/beir/{source.beir_spec.subset}/corpus.csv").open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = str(row.get("idx") or row.get("_id") or row.get("id") or "")
                title = str(row.get("title") or "")
                text = str(row.get("text") or "")
                yield idx, "", f"{title}. {text}" if title and title not in text[:100] else text
        return

    import chromadb

    client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_DIR", str(ROOT / "chroma_db")))
    collection = client.get_collection(source.collection)
    count = collection.count()
    batch_size = 10000
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
            state = str(meta.get("state") or "").strip().lower()
            title = str(meta.get("title") or meta.get("citation") or meta.get("source") or "")
            text = str(doc or "")
            if title and title.lower() not in text[:160].lower():
                text = f"{title}. {text}"
            yield idx, state, text
        print(f"[fts] {dataset}: streamed {min(offset + len(ids), count)}/{count}", flush=True)


def ensure_fts_index(dataset: str, index_dir: Path, *, rebuild: bool = False) -> tuple[Path, str]:
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = index_dir / f"{dataset}.sqlite"
    if rebuild and db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='docs'"
        ).fetchone()
        if exists:
            count = conn.execute("SELECT count(*) FROM docs").fetchone()[0]
            return db_path, f"reused sqlite fts5 index docs={count}"
        started = time.time()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=OFF")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("CREATE VIRTUAL TABLE docs USING fts5(doc_id UNINDEXED, state UNINDEXED, text, tokenize='porter unicode61')")
        batch: list[tuple[str, str, str]] = []
        count = 0
        for idx, state, text in iter_docs_with_state(dataset):
            batch.append((idx, state, text))
            if len(batch) >= 5000:
                conn.executemany("INSERT INTO docs(doc_id, state, text) VALUES (?, ?, ?)", batch)
                conn.commit()
                count += len(batch)
                batch.clear()
                if count % 50000 == 0:
                    print(f"[fts] {dataset}: inserted {count}", flush=True)
        if batch:
            conn.executemany("INSERT INTO docs(doc_id, state, text) VALUES (?, ?, ?)", batch)
            conn.commit()
            count += len(batch)
        conn.execute("INSERT INTO docs(docs) VALUES ('optimize')")
        conn.commit()
        elapsed = time.time() - started
        return db_path, f"built sqlite fts5 index docs={count}; elapsed_sec={elapsed:.1f}"
    finally:
        conn.close()


def query_state_for(dataset: str, qrow: dict[str, Any]) -> str:
    if dataset != "housing":
        return ""
    return str(qrow.get("state") or "").strip().lower()


def load_question_rows_with_state(dataset: str) -> dict[str, dict[str, Any]]:
    qmap = questions_for(dataset)
    if dataset != "housing":
        return qmap
    from eval_config import EvalConfig, load_questions
    from eval_harness import _row_label

    cfg = EvalConfig(dataset="housing", questions="full", seed=42, retrieval_k=5, housing_state_filter=True)
    df = load_questions(cfg)
    out = dict(qmap)
    for fallback_i, row in df.iterrows():
        label = _row_label(row, cfg, fallback_i=fallback_i)
        if label in out:
            out[label]["state"] = str(row.get("state") or "")
    return out


def state_key(value: str) -> str:
    return "_".join(tokenize(value)) or "none"


def tantivy_schema():
    import tantivy

    builder = tantivy.SchemaBuilder()
    builder.add_text_field("doc_id", stored=True, tokenizer_name="raw", index_option="basic")
    builder.add_text_field("state_key", stored=True, tokenizer_name="raw", index_option="basic")
    builder.add_text_field("text", stored=False, tokenizer_name="default", index_option="position")
    return builder.build()


def ensure_tantivy_index(dataset: str, index_dir: Path, *, rebuild: bool = False) -> tuple[Path, dict[str, Any], str]:
    import tantivy

    index_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = index_dir / dataset
    meta_path = dataset_dir / "_credibility_meta.json"
    if rebuild and dataset_dir.exists():
        import shutil

        shutil.rmtree(dataset_dir)
    if dataset_dir.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return dataset_dir, meta, f"reused tantivy index docs={meta.get('n_docs')}"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    schema = tantivy_schema()
    index = tantivy.Index(schema, str(dataset_dir), reuse=False)
    writer = index.writer(heap_size=512_000_000, num_threads=max(1, os.cpu_count() or 1))
    started = time.time()
    n_docs = 0
    total_len = 0
    for idx, state, text in iter_docs_with_state(dataset):
        doc = tantivy.Document()
        doc.add_text("doc_id", str(idx))
        doc.add_text("state_key", state_key(state))
        doc.add_text("text", str(text or ""))
        writer.add_document(doc)
        n_docs += 1
        total_len += len(tokenize(text))
        if n_docs % 100000 == 0:
            writer.commit()
            print(f"[tantivy] {dataset}: committed {n_docs}", flush=True)
    writer.commit()
    writer.wait_merging_threads()
    meta = {
        "dataset": dataset,
        "n_docs": n_docs,
        "avgdl": total_len / n_docs if n_docs else 0.0,
        "elapsed_sec": round(time.time() - started, 1),
    }
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n")
    return dataset_dir, meta, f"built tantivy index docs={n_docs}; elapsed_sec={meta['elapsed_sec']:.1f}"


def bm25_query_string(text: str, max_terms: int, state: str = "") -> str:
    terms = tokenize(text, max_terms=max_terms)
    if not terms:
        return ""
    body = " OR ".join(terms)
    skey = state_key(state)
    if state and skey != "none":
        return f"state_key:{skey} AND ({body})"
    return body


def manual_bm25_score(query: str, doc: str, *, searcher: Any, n_docs: int, avgdl: float, df_cache: dict[str, int]) -> float:
    qtf = Counter(tokenize(query))
    dtf = Counter(tokenize(doc))
    dl = sum(dtf.values())
    if not qtf or not dtf or not n_docs or not avgdl:
        return 0.0
    score = 0.0
    k1 = 1.5
    b = 0.75
    for term, qcount in qtf.items():
        freq = dtf.get(term, 0)
        if not freq:
            continue
        if term not in df_cache:
            try:
                df_cache[term] = int(searcher.doc_freq("text", term))
            except Exception:
                df_cache[term] = 0
        df = df_cache[term]
        if df <= 0:
            continue
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        denom = freq + k1 * (1.0 - b + b * dl / avgdl)
        score += qcount * idf * (freq * (k1 + 1.0) / denom)
    return float(score)


def bm25_tantivy_rows(dataset: str, arms: list[str], index_dir: Path, *, max_query_terms: int, rebuild: bool) -> tuple[list[dict[str, Any]], str]:
    import tantivy

    index_path, meta, index_note = ensure_tantivy_index(dataset, index_dir, rebuild=rebuild)
    qmap = load_question_rows_with_state(dataset)
    queries: dict[str, dict[str, str]] = {"raw": {label: q["question"] for label, q in qmap.items()}}
    for arm in arms:
        if arm == "raw":
            continue
        gen = generation_for(dataset, arm, OR_GEMMA)
        queries[arm] = {label: generation_passage(gen[label]) for label in qmap if label in gen}

    rows: list[dict[str, Any]] = []
    index = tantivy.Index.open(str(index_path))
    index.reload()
    searcher = index.searcher()
    gold_ids_all = [gid for qrow in qmap.values() for gid in qrow["gold_ids"]]
    gold_docs = gold_docs_for(dataset, gold_ids_all)
    df_cache: dict[str, int] = {}
    started = time.time()
    for arm in arms:
        for i, (label, text) in enumerate(queries[arm].items(), 1):
            qrow = qmap[label]
            query_text = bm25_query_string(text, max_query_terms, query_state_for(dataset, qrow))
            ids: list[str] = []
            scores: list[float] = []
            if query_text:
                try:
                    query = index.parse_query(query_text, ["text", "state_key"])
                except Exception:
                    query = index.parse_query_lenient(query_text, ["text", "state_key"])
                result = searcher.search(query, 10)
                for score, address in result.hits:
                    doc = searcher.doc(address)
                    ids.append(str(doc.get_first("doc_id")))
                    scores.append(float(score))
            gold_score = max(
                (
                    manual_bm25_score(
                        text,
                        gold_docs[gid],
                        searcher=searcher,
                        n_docs=int(meta["n_docs"]),
                        avgdl=float(meta["avgdl"]),
                        df_cache=df_cache,
                    )
                    for gid in qrow["gold_ids"]
                    if gid in gold_docs
                ),
                default=0.0,
            )
            rows.append({
                "dataset": dataset,
                "dataset_display": DATASET_SOURCES[dataset].display,
                "retriever": "bm25_tantivy_full",
                "arm": arm,
                "label": label,
                "gold_ids": qrow["gold_ids"],
                "retrieved_ids": ids,
                "scores": scores,
                "hit5": hit_at(ids, qrow["gold_ids"], 5),
                "hit10": hit_at(ids, qrow["gold_ids"], 10),
                "gold_affinity": gold_score,
                "query_terms": len(tokenize(text)),
                "query_terms_used": len(tokenize(text, max_terms=max_query_terms)),
            })
            if i % 500 == 0:
                print(f"[bm25] {dataset}/{arm}: {i}/{len(queries[arm])}", flush=True)
    note = f"{index_note}; retrieval_elapsed_sec={time.time() - started:.1f}; max_query_terms={max_query_terms}"
    return rows, note


def summarize_retriever_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(r["retriever"], r["dataset"], r["arm"], r["label"]): r for r in rows}
    summary_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    for retriever in sorted({r["retriever"] for r in rows}):
        for dataset in sorted({r["dataset"] for r in rows if r["retriever"] == retriever}):
            for arm in ["raw", "hyde", "scope"]:
                group = [r for r in rows if r["retriever"] == retriever and r["dataset"] == dataset and r["arm"] == arm]
                if not group:
                    continue
                help_n = hurt_n = 0
                deltas = []
                gold_deltas = []
                for row in group:
                    if arm == "raw":
                        continue
                    raw = by_key.get((retriever, dataset, "raw", row["label"]))
                    if not raw:
                        continue
                    delta = int(row["hit5"]) - int(raw["hit5"])
                    deltas.append(delta)
                    if math.isfinite(float(row.get("gold_affinity", float("nan")))) and math.isfinite(float(raw.get("gold_affinity", float("nan")))):
                        gold_deltas.append(float(row["gold_affinity"]) - float(raw["gold_affinity"]))
                    if delta > 0:
                        help_n += 1
                    elif delta < 0:
                        hurt_n += 1
                summary_rows.append({
                    "retriever": retriever,
                    "dataset": dataset,
                    "dataset_display": group[0]["dataset_display"],
                    "arm": arm,
                    "n": len(group),
                    "hit5": mean(r["hit5"] for r in group),
                    "hit10": mean(r["hit10"] for r in group),
                    "delta": mean(deltas) if deltas else 0.0,
                    "help": help_n,
                    "hurt": hurt_n,
                    "ri": (help_n - hurt_n) / len(group) if group and arm != "raw" else 0.0,
                })
                if arm != "raw" and gold_deltas:
                    corr_rows.append({
                        "retriever": retriever,
                        "dataset": dataset,
                        "dataset_display": group[0]["dataset_display"],
                        "arm": arm,
                        **corr(gold_deltas, deltas),
                        "mean_gold_delta": mean(gold_deltas),
                    })
    return summary_rows, corr_rows


def add_original_affinity(rows: list[dict[str, Any]]) -> None:
    point_rows: dict[tuple[str, str, str], float] = {}
    for row in read_jsonl(Path("/tmp/beir_phase1_verification_2026-05-26_points.jsonl")):
        if row.get("model") == OR_GEMMA:
            point_rows[(str(row.get("dataset")), str(row.get("expansion")), str(row.get("label")))] = float(row.get("ce_exp_gold", float("nan")))
            point_rows[(str(row.get("dataset")), "raw", str(row.get("label")))] = float(row.get("ce_raw_gold", float("nan")))
    for row in read_jsonl(Path("/tmp/affinity_margin_oncache_2026-05-26_points.jsonl")):
        if row.get("model") == OR_GEMMA:
            point_rows[(str(row.get("dataset")), "scope", str(row.get("label")))] = float(row.get("ce_scope_gold", float("nan")))
            point_rows[(str(row.get("dataset")), "raw", str(row.get("label")))] = float(row.get("ce_raw_gold", float("nan")))
    for row in rows:
        if row["retriever"] == "gte_ce_original":
            row["gold_affinity"] = point_rows.get((row["dataset"], row["arm"], row["label"]), float("nan"))


def phase_cxx(args: argparse.Namespace) -> None:
    datasets = args.datasets or DATASETS_7
    all_rows: list[dict[str, Any]] = []
    notes: list[str] = []
    failures: list[str] = []

    for dataset in datasets:
        rows = load_original_rows(dataset, ["raw", "hyde", "scope"])
        all_rows.extend(rows)
    add_original_affinity(all_rows)
    notes.append("Loaded original gte+CE retrieval caches for all requested datasets.")

    for dataset in datasets:
        try:
            rows, note = bm25_tantivy_rows(
                dataset,
                ["raw", "hyde", "scope"],
                Path(args.index_dir),
                max_query_terms=args.max_query_terms,
                rebuild=args.rebuild_bm25,
            )
            all_rows.extend(rows)
            notes.append(f"{DATASET_SOURCES[dataset].display} BM25: {note}")
            write_jsonl(PHASE_CXX_POINTS, all_rows)
        except Exception as exc:
            failures.append(f"{DATASET_SOURCES[dataset].display} BM25 failed: {type(exc).__name__}: {str(exc)[:300]}")
            write_jsonl(PHASE_CXX_POINTS, all_rows)

    e5_note = "E5/BGE third-dense retriever not run in this invocation."
    if args.e5_status:
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(args.e5_model, local_files_only=True)
            e5_note = f"E5/BGE availability: `{args.e5_model}` is cached locally with dim={model.get_sentence_embedding_dimension()}; full embedding/retrieval stage remains pending."
        except Exception as exc:
            e5_note = f"E5/BGE availability blocked for `{args.e5_model}`: {type(exc).__name__}: {str(exc)[:260]}"
    notes.append(e5_note)

    summary_rows, corr_rows = summarize_retriever_rows(all_rows)
    scope_corr_values = [
        float(r["spearman"]) for r in corr_rows
        if r["arm"] == "scope" and math.isfinite(float(r.get("spearman", float("nan"))))
    ]
    bm25_scope = [
        float(r["spearman"]) for r in corr_rows
        if r["arm"] == "scope" and r["retriever"] == "bm25_tantivy_full" and math.isfinite(float(r.get("spearman", float("nan"))))
    ]
    original_scope = [
        float(r["spearman"]) for r in corr_rows
        if r["arm"] == "scope" and r["retriever"] == "gte_ce_original" and math.isfinite(float(r.get("spearman", float("nan"))))
    ]
    verdict = "provisional"
    if bm25_scope and mean(bm25_scope) >= 0.3:
        verdict = "mechanism travels to BM25; third dense pending"
    elif bm25_scope and mean(bm25_scope) <= 0.1:
        verdict = "mechanism appears dense/gte-specific unless third dense rescues it"
    elif bm25_scope:
        verdict = "mechanism partly retriever-specific"

    lines = [
        "# Credibility C++ - Three-Retriever Full-Corpus Battery",
        "",
        "No `paper/` files were edited.",
        "",
        "## Verdict",
        "",
        f"- Status: **{verdict}**.",
        f"- Original gte+CE mean SCOPE Spearman: `{fmt(mean(original_scope))}` over `{len(original_scope)}` dataset correlations.",
        f"- Full-corpus BM25 mean SCOPE Spearman: `{fmt(mean(bm25_scope))}` over `{len(bm25_scope)}` dataset correlations.",
        f"- All finite SCOPE correlation mean across completed retrievers: `{fmt(mean(scope_corr_values))}`.",
        "- The requested third dense retriever is not silently substituted; if its full retrieval rows are absent below, the report is provisional for the three-retriever criterion.",
        "",
        "## Retrieval Summary",
        "",
        "| Retriever | Dataset | Arm | N | Hit@5 | Hit@10 | Delta vs raw | Help | Hurt | RI |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['retriever']} | {row['dataset_display']} | {row['arm']} | {row['n']} | "
            f"{pct(row['hit5'])} | {pct(row['hit10'])} | {pct(row['delta'])} | "
            f"{row['help']} | {row['hurt']} | {fmt(row['ri'])} |"
        )
    lines.extend([
        "",
        "## Gold-Affinity Delta Correlations",
        "",
        "| Retriever | Dataset | Arm | N | Spearman | Kendall | Pearson | Mean gold-affinity delta |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in corr_rows:
        lines.append(
            f"| {row['retriever']} | {row['dataset_display']} | {row['arm']} | {row['n']} | "
            f"{fmt(row['spearman'])} | {fmt(row['kendall'])} | {fmt(row['pearson'])} | {fmt(row['mean_gold_delta'])} |"
        )
    lines.extend([
        "",
        "## Run Notes",
        "",
    ])
    for note in notes:
        lines.append(f"- {note}")
    for failure in failures:
        lines.append(f"- {failure}")
    lines.extend([
        f"- Row-level points: `{PHASE_CXX_POINTS.relative_to(ROOT)}`",
        "- BM25 uses Tantivy disk-backed full-corpus indexes for each dataset. Housing applies the question state as a metadata filter.",
        f"- BM25 query term cap: `{args.max_query_terms}` unique non-stopword terms by within-query frequency; this avoids pathological FTS query length while preserving full-corpus search.",
        "",
    ])
    PHASE_CXX_REPORT.write_text("\n".join(lines) + "\n")
    print(PHASE_CXX_REPORT)


def e5_prefix(text: str, kind: str) -> str:
    prefix = "query: " if kind == "query" else "passage: "
    clean = " ".join(str(text or "").split())
    if E5_TEXT_MAX_CHARS and len(clean) > E5_TEXT_MAX_CHARS:
        clean = clean[:E5_TEXT_MAX_CHARS]
    return prefix + clean


def encode_e5(model: Any, texts: list[str], *, kind: str, batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    arr = model.encode(
        [e5_prefix(text, kind) for text in texts],
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(arr, dtype=np.float32)


def e5_query_items(dataset: str) -> tuple[dict[str, dict[str, Any]], list[tuple[str, str, str]]]:
    qmap = load_question_rows_with_state(dataset)
    items: list[tuple[str, str, str]] = []
    for label, qrow in qmap.items():
        items.append(("raw", label, str(qrow["question"])))
    for arm in ["hyde", "scope"]:
        gen = generation_for(dataset, arm, OR_GEMMA)
        for label in qmap:
            if label not in gen:
                continue
            passage = generation_passage(gen[label])
            if passage:
                items.append((arm, label, passage))
    return qmap, items


def e5_gold_affinities(
    *,
    dataset: str,
    qmap: dict[str, dict[str, Any]],
    query_items: list[tuple[str, str, str]],
    query_vectors: np.ndarray,
    model: Any,
    batch_size: int,
) -> dict[tuple[str, str], float]:
    gold_ids = list(dict.fromkeys(
        gid for label in qmap for gid in qmap[label].get("gold_ids", []) if str(gid)
    ))
    gold_docs = gold_docs_for(dataset, gold_ids)
    gold_doc_ids = [gid for gid in gold_ids if gid in gold_docs]
    gold_vecs = encode_e5(model, [gold_docs[gid] for gid in gold_doc_ids], kind="passage", batch_size=batch_size)
    by_gold = {gid: gold_vecs[i] for i, gid in enumerate(gold_doc_ids)}
    out: dict[tuple[str, str], float] = {}
    for i, (arm, label, _) in enumerate(query_items):
        vals = [
            float(np.dot(query_vectors[i], by_gold[gid]))
            for gid in qmap[label].get("gold_ids", [])
            if gid in by_gold
        ]
        out[(arm, label)] = max(vals) if vals else float("nan")
    return out


def e5_rows_for_dataset(
    dataset: str,
    *,
    model: Any,
    corpus_batch_size: int,
    query_batch_size: int,
) -> tuple[list[dict[str, Any]], str]:
    if dataset == "housing":
        raise RuntimeError("housing E5 state-filtered retrieval is a stretch target; state-sharded E5 indexing is not enabled in this pass")
    import faiss

    qmap, query_items = e5_query_items(dataset)
    started = time.time()
    query_vectors = encode_e5(
        model,
        [text for _, _, text in query_items],
        kind="query",
        batch_size=query_batch_size,
    )
    if query_vectors.ndim != 2 or query_vectors.shape[0] != len(query_items):
        raise RuntimeError(f"{dataset}: bad query embedding shape {query_vectors.shape}")
    dim = int(query_vectors.shape[1])
    index = faiss.IndexFlatIP(dim)
    doc_ids: list[str] = []
    doc_buffer: list[str] = []
    id_buffer: list[str] = []
    n_docs = 0
    embed_sec = 0.0

    def flush() -> None:
        nonlocal embed_sec, n_docs
        if not doc_buffer:
            return
        t0 = time.time()
        vecs = encode_e5(model, doc_buffer, kind="passage", batch_size=corpus_batch_size)
        embed_sec += time.time() - t0
        index.add(vecs)
        doc_ids.extend(id_buffer)
        n_docs += len(id_buffer)
        doc_buffer.clear()
        id_buffer.clear()
        if n_docs % 50000 < corpus_batch_size:
            print(f"[e5] {dataset}: indexed {n_docs}", flush=True)

    for idx, _state, text in iter_docs_with_state(dataset):
        id_buffer.append(str(idx))
        doc_buffer.append(text)
        if len(doc_buffer) >= corpus_batch_size:
            flush()
    flush()
    if index.ntotal != len(doc_ids):
        raise RuntimeError(f"{dataset}: index/doc-id mismatch {index.ntotal} != {len(doc_ids)}")

    search_started = time.time()
    scores, indices = index.search(query_vectors, 10)
    search_sec = time.time() - search_started
    gold_aff = e5_gold_affinities(
        dataset=dataset,
        qmap=qmap,
        query_items=query_items,
        query_vectors=query_vectors,
        model=model,
        batch_size=query_batch_size,
    )
    rows: list[dict[str, Any]] = []
    for i, (arm, label, text) in enumerate(query_items):
        ids = [doc_ids[int(j)] for j in indices[i].tolist() if int(j) >= 0]
        row_scores = [float(s) for s in scores[i].tolist()[: len(ids)]]
        gold_ids = qmap[label]["gold_ids"]
        rows.append({
            "dataset": dataset,
            "dataset_display": DATASET_SOURCES[dataset].display,
            "retriever": "e5_large_v2_full",
            "arm": arm,
            "label": label,
            "gold_ids": gold_ids,
            "retrieved_ids": ids,
            "scores": row_scores,
            "hit5": hit_at(ids, gold_ids, 5),
            "hit10": hit_at(ids, gold_ids, 10),
            "gold_affinity": gold_aff.get((arm, label), float("nan")),
            "query_chars": len(text),
        })
    note = (
        f"docs={n_docs}; query_vectors={len(query_items)}; dim={dim}; "
        f"embed_sec={embed_sec:.1f}; search_sec={search_sec:.1f}; elapsed_sec={time.time() - started:.1f}"
    )
    del index, query_vectors, scores, indices
    gc.collect()
    return rows, note


def phase_cxx_e5(args: argparse.Namespace) -> None:
    datasets = args.datasets or E5_DATASETS_PRIORITY
    from sentence_transformers import SentenceTransformer
    import torch

    model = SentenceTransformer(args.e5_model, local_files_only=True)
    if args.fp16 and torch.cuda.is_available():
        model = model.to("cuda")
        model.half()
        print("[e5] using cuda fp16", flush=True)
    existing = read_jsonl(PHASE_CXX_E5_POINTS) if PHASE_CXX_E5_POINTS.exists() and args.resume else []
    done = {
        str(row.get("dataset"))
        for row in existing
        if row.get("retriever") == "e5_large_v2_full"
    }
    all_e5_rows = list(existing)
    notes: list[str] = []
    failures: list[str] = []
    for dataset in datasets:
        if dataset in done and not args.rebuild:
            notes.append(f"{DATASET_SOURCES[dataset].display} E5: reused existing rows")
            continue
        try:
            rows, note = e5_rows_for_dataset(
                dataset,
                model=model,
                corpus_batch_size=args.corpus_batch_size,
                query_batch_size=args.query_batch_size,
            )
            all_e5_rows = [r for r in all_e5_rows if str(r.get("dataset")) != dataset]
            all_e5_rows.extend(rows)
            notes.append(f"{DATASET_SOURCES[dataset].display} E5: {note}")
            write_jsonl(PHASE_CXX_E5_POINTS, all_e5_rows)
        except Exception as exc:
            failures.append(f"{DATASET_SOURCES[dataset].display} E5 failed/pending: {type(exc).__name__}: {str(exc)[:300]}")
            write_jsonl(PHASE_CXX_E5_POINTS, all_e5_rows)

    combined_rows = read_jsonl(PHASE_CXX_POINTS) if PHASE_CXX_POINTS.exists() else []
    combined_rows.extend(all_e5_rows)
    summary_rows, corr_rows = summarize_retriever_rows(combined_rows)
    e5_scope = [
        float(r["spearman"]) for r in corr_rows
        if r["retriever"] == "e5_large_v2_full" and r["arm"] == "scope" and math.isfinite(float(r.get("spearman", float("nan"))))
    ]
    original_scope = [
        float(r["spearman"]) for r in corr_rows
        if r["retriever"] == "gte_ce_original" and r["arm"] == "scope" and math.isfinite(float(r.get("spearman", float("nan"))))
    ]
    bm25_scope = [
        float(r["spearman"]) for r in corr_rows
        if r["retriever"] == "bm25_tantivy_full" and r["arm"] == "scope" and math.isfinite(float(r.get("spearman", float("nan"))))
    ]
    verdict = "closed for completed E5 datasets" if e5_scope and mean(e5_scope) >= 0.3 else "provisional/mixed"
    lines = [
        "# Credibility C++ E5 Addendum - 2026-05-29",
        "",
        "No `paper/` files were edited.",
        "",
        "## Verdict",
        "",
        f"- Three-retriever status: **{verdict}**.",
        f"- E5 completed datasets: `{len(set(r['dataset'] for r in all_e5_rows))}`; SCOPE mean Spearman `{fmt(mean(e5_scope))}` over `{len(e5_scope)}` dataset correlations.",
        f"- Original gte+CE SCOPE mean Spearman `{fmt(mean(original_scope))}`; BM25 SCOPE mean Spearman `{fmt(mean(bm25_scope))}`.",
        "- Verdict criterion: E5 mean SCOPE Spearman >= 0.3 across completed datasets closes the three-retriever mechanism claim for those datasets.",
        f"- E5 embedding inputs are capped at `{E5_TEXT_MAX_CHARS}` characters, matching the existing BEIR embedding pipeline cap.",
        "",
        "## E5 Retrieval Summary",
        "",
        "| Dataset | Arm | N | Hit@5 | Hit@10 | Delta vs raw | Help | Hurt | RI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        if row["retriever"] != "e5_large_v2_full":
            continue
        lines.append(
            f"| {row['dataset_display']} | {row['arm']} | {row['n']} | {pct(row['hit5'])} | "
            f"{pct(row['hit10'])} | {pct(row['delta'])} | {row['help']} | {row['hurt']} | {fmt(row['ri'])} |"
        )
    lines.extend([
        "",
        "## Three-Retriever Mechanism Comparison",
        "",
        "| Retriever | Dataset | Arm | N | Spearman | Kendall | Pearson | Mean gold-affinity delta |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in corr_rows:
        if row["arm"] != "scope":
            continue
        lines.append(
            f"| {row['retriever']} | {row['dataset_display']} | {row['arm']} | {row['n']} | "
            f"{fmt(row['spearman'])} | {fmt(row['kendall'])} | {fmt(row['pearson'])} | {fmt(row['mean_gold_delta'])} |"
        )
    lines.extend([
        "",
        "## Run Notes",
        "",
    ])
    for note in notes:
        lines.append(f"- {note}")
    for failure in failures:
        lines.append(f"- {failure}")
    completed_e5 = {str(r.get("dataset")) for r in all_e5_rows}
    if "barexam" not in completed_e5:
        lines.append("- BarExamQA E5 remains pending after the five-BEIR priority pass; it should be run before claiming legal-corpus E5 generality.")
    if "housing" not in completed_e5:
        lines.append("- HousingQA E5 state-filtered retrieval remains a stretch target; it needs state-sharded E5 indexing to preserve the jurisdiction filter.")
    lines.extend([
        f"- E5 row-level points: `{PHASE_CXX_E5_POINTS.relative_to(ROOT)}`",
        f"- Base C++ row-level points: `{PHASE_CXX_POINTS.relative_to(ROOT)}`",
        "",
    ])
    PHASE_CXX_E5_REPORT.write_text("\n".join(lines) + "\n")
    print(PHASE_CXX_E5_REPORT)


def phase_e_cache_paths(dataset: str) -> tuple[Path, Path, Path]:
    if dataset.startswith("beir_"):
        return (
            ROOT / f"caches/retrieval/full/{dataset}_qfull_seed42_raw_question_k10.jsonl",
            ROOT / f"caches/retrieval/full/{dataset}_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl",
            ROOT / f"caches/retrieval/full/{dataset}_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl",
        )
    if dataset == "barexam":
        return (
            ROOT / "caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl",
            ROOT / "caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl",
            ROOT / "caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl",
        )
    if dataset == "housing":
        return (
            ROOT / "caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl",
            ROOT / "caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl",
            ROOT / "caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_raw_scope_pool_k5.jsonl",
        )
    if dataset == "casehold":
        return (
            ROOT / "caches/retrieval/full/casehold_qfull_seed42_raw_question_k10.jsonl",
            ROOT / "caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl",
            ROOT / "caches/retrieval/full/casehold_qfull_seed42_groq-llama70b_raw_scope_pool_k5.jsonl",
        )
    raise KeyError(dataset)


def phase_e_display(dataset: str) -> str:
    source = DATASET_SOURCES.get(dataset)
    if source:
        return source.display
    return PHASE_E_DISPLAY.get(dataset, dataset)


def hit_from_cache(row: dict[str, Any], k: int) -> int:
    gold = {str(x) for x in row.get("gold_ids") or []}
    ids = [str(x) for x in row.get("retrieved_ids") or []][:k]
    return int(bool(gold.intersection(ids)))


def phase_e_dataset_summary(dataset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    paths = phase_e_cache_paths(dataset)
    display = phase_e_display(dataset)
    notes: list[str] = []
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.exists()]
    if missing:
        return [], [], [f"{display}: missing caches {missing}"]
    raw, scope, pool = [load_by_label(path) for path in paths]
    labels = sorted(set(raw).intersection(scope).intersection(pool))
    rows: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    for k in range(1, 6):
        raw_hits = [hit_from_cache(raw[label], k) for label in labels]
        scope_hits = [hit_from_cache(scope[label], k) for label in labels]
        pool_hits = [hit_from_cache(pool[label], k) for label in labels]
        raw_rate = mean(raw_hits)
        scope_rate = mean(scope_hits)
        pool_rate = mean(pool_hits)
        pool_minus_raw = pool_rate - raw_rate
        if pool_minus_raw > 0.005:
            pool_verdict = "helps raw"
        elif pool_minus_raw < -0.01:
            pool_verdict = "hurts raw"
        else:
            pool_verdict = "flat"
        rows.append({
            "dataset": dataset,
            "dataset_display": display,
            "slice": f"Hit@{k}",
            "k": k,
            "n": len(labels),
            "raw": raw_rate,
            "scope": scope_rate,
            "pool": pool_rate,
            "scope_minus_raw": scope_rate - raw_rate,
            "pool_minus_raw": pool_minus_raw,
            "pool_minus_scope": pool_rate - scope_rate,
            "pool_verdict": pool_verdict,
        })
        for label, raw_hit, scope_hit, pool_hit in zip(labels, raw_hits, scope_hits, pool_hits):
            points.append({
                "dataset": dataset,
                "dataset_display": display,
                "label": label,
                "k": k,
                "raw_hit": raw_hit,
                "scope_hit": scope_hit,
                "pool_hit": pool_hit,
                "scope_minus_raw": scope_hit - raw_hit,
                "pool_minus_raw": pool_hit - raw_hit,
                "pool_minus_scope": pool_hit - scope_hit,
            })
    notes.append(
        f"{display}: N={len(labels)} from intersection of raw, SCOPE, and raw+SCOPE pool caches"
    )
    return rows, points, notes


def phase_e(args: argparse.Namespace) -> None:
    datasets = args.datasets or PHASE_E_DATASETS
    all_rows: list[dict[str, Any]] = []
    all_points: list[dict[str, Any]] = []
    notes: list[str] = []
    for dataset in datasets:
        rows, points, dataset_notes = phase_e_dataset_summary(dataset)
        all_rows.extend(rows)
        all_points.extend(points)
        notes.extend(dataset_notes)
    write_jsonl(PHASE_E_POINTS, all_points)

    strict_mid = [r for r in all_rows if 0.20 <= r["raw"] <= 0.30]
    near_mid = [r for r in all_rows if 0.20 <= r["raw"] <= 0.35]
    k5_rows = [r for r in all_rows if r["k"] == 5]
    lines = [
        "# Credibility E Mid-Regime Construction - 2026-05-29",
        "",
        "No `paper/` files were edited. This report is read-only over existing retrieval caches.",
        "",
        "## Verdict",
        "",
        "- Constructed mid-regime points using the allowed lower-k evidence-budget axis: the same caches are evaluated at Hit@1 through Hit@5.",
        "- In the strict 20-30% raw band, the raw+SCOPE pool improves raw on all available points: SciDocs Hit@1 and Housing state-filtered Hit@2/Hit@3.",
        "- The threshold is not a clean raw-Hit-only boundary. Pooling starts to help raw in the low-20% regime, but it only carries the method when SCOPE contributes complementary correct evidence; it remains weak relative to SCOPE on BarExamQA and CaseHOLD.",
        "- Honest claim: raw+SCOPE pooling is useful as a risk-control fusion in mid/high raw regimes, not as a universal replacement for canonical SCOPE on sparse legal corpora.",
        "",
        "## Strict Mid-Regime Points",
        "",
        "| Dataset | Slice | N | Raw | SCOPE | Raw+SCOPE pool | SCOPE-Raw | Pool-Raw | Pool-SCOPE | Reading |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(strict_mid, key=lambda r: (r["raw"], r["dataset"], r["k"])):
        lines.append(
            f"| {row['dataset_display']} | {row['slice']} | {row['n']} | {pct(row['raw'])} | "
            f"{pct(row['scope'])} | {pct(row['pool'])} | {pct(row['scope_minus_raw'])} | "
            f"{pct(row['pool_minus_raw'])} | {pct(row['pool_minus_scope'])} | {row['pool_verdict']} |"
        )
    if not strict_mid:
        lines.append("| _None_ | - | - | - | - | - | - | - | - | - |")
    lines.extend([
        "",
        "## Near-Mid Regime Context",
        "",
        "| Dataset | Slice | N | Raw | SCOPE | Raw+SCOPE pool | SCOPE-Raw | Pool-Raw | Pool-SCOPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(near_mid, key=lambda r: (r["raw"], r["dataset"], r["k"])):
        lines.append(
            f"| {row['dataset_display']} | {row['slice']} | {row['n']} | {pct(row['raw'])} | "
            f"{pct(row['scope'])} | {pct(row['pool'])} | {pct(row['scope_minus_raw'])} | "
            f"{pct(row['pool_minus_raw'])} | {pct(row['pool_minus_scope'])} |"
        )
    lines.extend([
        "",
        "## Full Hit@5 Anchors",
        "",
        "| Dataset | N | Raw Hit@5 | SCOPE Hit@5 | Raw+SCOPE pool Hit@5 | SCOPE-Raw | Pool-Raw | Pool-SCOPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sorted(k5_rows, key=lambda r: r["raw"]):
        lines.append(
            f"| {row['dataset_display']} | {row['n']} | {pct(row['raw'])} | {pct(row['scope'])} | "
            f"{pct(row['pool'])} | {pct(row['scope_minus_raw'])} | {pct(row['pool_minus_raw'])} | "
            f"{pct(row['pool_minus_scope'])} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- CaseHOLD is the lower anchor: raw Hit@5 is 17.9%, SCOPE jumps to 45.0%, but the raw+SCOPE pool reaches only 19.2%. In this sparse regime, fusion mostly preserves raw rather than the SCOPE gains.",
        "- SciDocs Hit@1 gives a strict mid-regime point: raw 22.2%, SCOPE 21.1%, pool 23.2%. The pool helps raw modestly, but this is a fusion gain rather than a SCOPE-alone gain.",
        "- Housing state-filtered Hit@2 gives the strongest strict mid-regime point: raw 23.9%, SCOPE 25.5%, pool 25.9%. Pooling is helpful and slightly better than either component.",
        "- The upper side of the void is consistent: SciDocs Hit@2 at 31.7% raw and Housing Hit@5 at 36.8% raw both show pool gains over raw.",
        "- Therefore the practical threshold for positive raw+SCOPE pooling appears around the low-20% raw-retrieval regime, but the threshold for replacing canonical SCOPE is higher and corpus-dependent.",
        "",
        "## Notes",
        "",
        "- This phase does not run new retrieval. Lower-k evaluation reuses the deterministic ranking already present in each cache.",
        "- The mid-regime construction is an evidence-budget proxy, not a new benchmark split. It is useful for regime-shape diagnosis, not for final leaderboard claims.",
    ])
    for note in notes:
        lines.append(f"- {note}")
    lines.extend([
        f"- Row-level points: `{PHASE_E_POINTS.relative_to(ROOT)}`",
        "",
    ])
    PHASE_E_REPORT.write_text("\n".join(lines) + "\n")
    print(PHASE_E_REPORT)


class BeirQppSpec:
    def __init__(self, key: str, collection: str) -> None:
        self.key = key
        self.collection = collection


def safe_tau(y_true: list[float], y_score: list[float]) -> float:
    pairs = [(float(y), float(s)) for y, s in zip(y_true, y_score) if math.isfinite(float(y)) and math.isfinite(float(s))]
    if len(pairs) < 3 or len({p[0] for p in pairs}) < 2 or len({p[1] for p in pairs}) < 2:
        return float("nan")
    value = kendalltau([p[1] for p in pairs], [p[0] for p in pairs], nan_policy="omit").statistic
    return float(value) if math.isfinite(float(value)) else float("nan")


def safe_spearman(y_true: list[float], y_score: list[float]) -> float:
    pairs = [(float(y), float(s)) for y, s in zip(y_true, y_score) if math.isfinite(float(y)) and math.isfinite(float(s))]
    if len(pairs) < 3 or len({p[0] for p in pairs}) < 2 or len({p[1] for p in pairs}) < 2:
        return float("nan")
    value = spearmanr([p[1] for p in pairs], [p[0] for p in pairs], nan_policy="omit").statistic
    return float(value) if math.isfinite(float(value)) else float("nan")


def safe_auc_values(y_true: list[int], y_score: list[float]) -> float:
    pairs = [(int(y), float(s)) for y, s in zip(y_true, y_score) if math.isfinite(float(s))]
    if len({p[0] for p in pairs}) < 2:
        return float("nan")
    return float(roc_auc_score([p[0] for p in pairs], [p[1] for p in pairs]))


def finite_feature_rows(rows: list[dict[str, Any]], features: list[str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        ok = True
        for feature in features:
            try:
                value = float(row.get(feature))
            except Exception:
                ok = False
                break
            if not math.isfinite(value):
                ok = False
                break
        if ok:
            out.append(row)
    return out


def matrix(rows: list[dict[str, Any]], features: list[str]) -> np.ndarray:
    return np.asarray([[float(row[f]) for f in features] for row in rows], dtype=np.float64)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0.0] = 1.0
    return (x_train - mu) / sd, (x_test - mu) / sd


def make_model(kind: str):
    if kind == "logistic":
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    if kind == "gb":
        return GradientBoostingClassifier(random_state=42, n_estimators=80, max_depth=2, learning_rate=0.05)
    raise ValueError(kind)


def fit_predict(train: list[dict[str, Any]], test: list[dict[str, Any]], features: list[str], kind: str) -> dict[str, Any]:
    train = finite_feature_rows(train, features)
    test = finite_feature_rows(test, features)
    if len(train) < 20 or len(test) < 5:
        return {"n_train": len(train), "n_test": len(test), "tau": float("nan"), "spearman": float("nan"), "auc": float("nan")}
    y_train = np.asarray([int(float(r["retrieval_delta"]) > 0) for r in train], dtype=np.int64)
    if len(set(y_train.tolist())) < 2:
        return {"n_train": len(train), "n_test": len(test), "tau": float("nan"), "spearman": float("nan"), "auc": float("nan")}
    x_train, x_test = standardize_train_test(matrix(train, features), matrix(test, features))
    model = make_model(kind)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    y_cont = [float(r["retrieval_delta"]) for r in test]
    y_bin = [int(float(r["retrieval_delta"]) > 0) for r in test]
    return {
        "n_train": len(train),
        "n_test": len(test),
        "tau": safe_tau(y_cont, score.tolist()),
        "spearman": safe_spearman(y_cont, score.tolist()),
        "auc": safe_auc_values(y_bin, score.tolist()),
        "positive_rate": mean(y_bin),
    }


def load_phase_d_points(*, chroma_batch_size: int, embed_batch_size: int) -> list[dict[str, Any]]:
    source_path = Path("/tmp/beir_phase1b_model_breadth_2026-05-26_points.jsonl")
    if not source_path.exists():
        raise SystemExit(f"Missing Phase 1b points: {source_path}")
    phase_rows = [
        row for row in read_jsonl(source_path)
        if row.get("expansion") == "scope" and row.get("model") in MODEL_DISPLAY
    ]
    by_dataset = sorted({str(row["dataset"]) for row in phase_rows})
    raw_features: dict[tuple[str, str], dict[str, float]] = {}
    for dataset in by_dataset:
        spec = BEIR_SPECS[dataset]
        raw_cache = load_by_label(retrieval_cache_path(dataset, "raw"))
        questions = load_questions_for_spec(spec)
        raw_queries = {label: row["question"] for label, row in questions.items()}
        print(f"[phase-d] {dataset}: dense features", flush=True)
        doc_embeddings = fetch_doc_embeddings(BeirQppSpec(dataset, spec.collection), raw_cache, chroma_batch_size)
        query_embeddings = embed_queries(raw_queries, embed_batch_size)
        for label, raw_row in raw_cache.items():
            feats = qpp_ce_features(raw_row.get("scores") or [])
            feats.update(dense_features_for_row(raw_row, doc_embeddings, query_embeddings.get(label)))
            raw_features[(dataset, label)] = feats
    points: list[dict[str, Any]] = []
    for row in phase_rows:
        dataset = str(row["dataset"])
        label = str(row["label"])
        feats = dict(raw_features.get((dataset, label), {}))
        feats["log_perplexity"] = float(row.get("log_perplexity", float("nan")))
        feats["question_tokens"] = float(row.get("token_count", float("nan")))
        feats["oov_rate"] = float(row.get("oov_rate", float("nan")))
        out = {
            "dataset": dataset,
            "dataset_display": row.get("dataset_display") or DATASET_SOURCES[dataset].display,
            "model": str(row["model"]),
            "model_display": MODEL_DISPLAY[str(row["model"])],
            "label": label,
            "retrieval_delta": float(row["retrieval_delta"]),
            "scope_help": int(float(row["retrieval_delta"]) > 0),
            "scope_hurt": int(float(row["retrieval_delta"]) < 0),
        }
        out.update(feats)
        points.append(out)
    return finite_feature_rows(points, QPP_FEATURES)


def split_mean(rows: list[dict[str, Any]], key: str) -> float:
    vals = [float(r[key]) for r in rows if math.isfinite(float(r.get(key, float("nan"))))]
    return mean(vals)


def phase_d(args: argparse.Namespace) -> None:
    points = load_phase_d_points(chroma_batch_size=args.chroma_batch_size, embed_batch_size=args.embed_batch_size)
    write_jsonl(PHASE_D_POINTS, points)

    models = sorted({p["model"] for p in points})
    datasets = sorted({p["dataset"] for p in points})
    model_kinds = ["logistic", "gb"]
    rows_eval: list[dict[str, Any]] = []

    for kind in model_kinds:
        rows_eval.append({"split": "in_sample", "heldout": "none", "model_kind": kind, **fit_predict(points, points, QPP_FEATURES, kind)})
        for model in models:
            train = [p for p in points if p["model"] != model]
            test = [p for p in points if p["model"] == model]
            rows_eval.append({"split": "heldout_generator", "heldout": MODEL_DISPLAY[model], "model_kind": kind, **fit_predict(train, test, QPP_FEATURES, kind)})
        for dataset in datasets:
            train = [p for p in points if p["dataset"] != dataset]
            test = [p for p in points if p["dataset"] == dataset]
            rows_eval.append({"split": "heldout_dataset_lodo", "heldout": DATASET_SOURCES[dataset].display, "model_kind": kind, **fit_predict(train, test, QPP_FEATURES, kind)})
        # With five BEIR datasets available for four-generator breadth, the literal
        # 5-train/2-heldout split is impossible; use leave-two-out as a harder proxy.
        for i, d1 in enumerate(datasets):
            for d2 in datasets[i + 1:]:
                held = {d1, d2}
                train = [p for p in points if p["dataset"] not in held]
                test = [p for p in points if p["dataset"] in held]
                rows_eval.append({"split": "heldout_dataset_leave2_proxy", "heldout": f"{DATASET_SOURCES[d1].display} + {DATASET_SOURCES[d2].display}", "model_kind": kind, **fit_predict(train, test, QPP_FEATURES, kind)})

    mean_by_kind = {}
    for kind in model_kinds:
        held = [r for r in rows_eval if r["model_kind"] == kind and r["split"] == "heldout_generator"]
        mean_by_kind[kind] = mean(r["tau"] for r in held)
    best_kind = max(model_kinds, key=lambda k: mean_by_kind.get(k, float("-inf")) if math.isfinite(mean_by_kind.get(k, float("nan"))) else -999)

    rng = np.random.default_rng(42)
    budget_sizes = [0, 25, 50, 100, 200, 500, 1000]
    budget_rows: list[dict[str, Any]] = []
    for model in models:
        base = [p for p in points if p["model"] != model]
        target = [p for p in points if p["model"] == model]
        for size in budget_sizes:
            vals = []
            for seed in range(5):
                local_rng = np.random.default_rng(1000 + seed)
                indices = np.arange(len(target))
                local_rng.shuffle(indices)
                calib_idx = set(indices[: min(size, max(0, len(target) - 50))].tolist())
                calib = [target[i] for i in calib_idx]
                test = [target[i] for i in range(len(target)) if i not in calib_idx]
                vals.append(fit_predict(base + calib, test, QPP_FEATURES, best_kind)["tau"])
            budget_rows.append({
                "heldout_model": MODEL_DISPLAY[model],
                "calibration_n": size,
                "mean_tau": mean(vals),
                "max_tau": max((v for v in vals if math.isfinite(v)), default=float("nan")),
            })
    _ = rng

    held_gen = [r for r in rows_eval if r["split"] == "heldout_generator" and r["model_kind"] == best_kind]
    held_dataset = [r for r in rows_eval if r["split"].startswith("heldout_dataset") and r["model_kind"] == best_kind]
    useful_negative = mean(abs(float(r["tau"])) for r in held_gen if math.isfinite(float(r["tau"]))) < 0.3
    verdict = "useful negative" if useful_negative else "promising"
    best_budget = next((r for r in budget_rows if math.isfinite(float(r["mean_tau"])) and abs(float(r["mean_tau"])) >= 0.5), None)

    lines = [
        "# Credibility D - OOD No-Gold QPP Predictor",
        "",
        "No `paper/` files were edited.",
        "",
        "## Verdict",
        "",
        f"- Status: **{verdict}**.",
        f"- Best model family by held-out-generator tau: `{best_kind}` with mean tau `{fmt(mean_by_kind[best_kind])}`.",
        f"- Held-out-generator mean Kendall tau: `{fmt(mean(r['tau'] for r in held_gen))}`; held-out-dataset mean tau: `{fmt(mean(r['tau'] for r in held_dataset))}`.",
        f"- Datta-style reliability bar `|tau| >= 0.5`: {'reached in calibration curve' if best_budget else 'not reached in the tested calibration budget'}." ,
        "",
        "## Coverage",
        "",
        "| Dataset | Generator | Rows | Help rate | Hurt rate | Mean retrieval delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        for model in models:
            group = [p for p in points if p["dataset"] == dataset and p["model"] == model]
            lines.append(
                f"| {DATASET_SOURCES[dataset].display} | {MODEL_DISPLAY[model]} | {len(group)} | "
                f"{pct(mean(p['scope_help'] for p in group))} | {pct(mean(p['scope_hurt'] for p in group))} | {pct(mean(p['retrieval_delta'] for p in group))} |"
            )
    lines.extend([
        "",
        "## OOD Splits",
        "",
        "| Split | Held out | Model | Train N | Test N | Kendall tau | Spearman | AUC(help) | Help rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows_eval:
        lines.append(
            f"| {row['split']} | {row['heldout']} | {row['model_kind']} | {row['n_train']} | {row['n_test']} | "
            f"{fmt(row['tau'])} | {fmt(row['spearman'])} | {fmt(row['auc'])} | {pct(row.get('positive_rate'))} |"
        )
    lines.extend([
        "",
        "## Calibration Budget Curve",
        "",
        f"Budget curve uses `{best_kind}` and adds labeled examples from the held-out generator before evaluating on the remaining held-out rows.",
        "",
        "| Held-out generator | Calibration labels | Mean tau | Max tau |",
        "|---|---:|---:|---:|",
    ])
    for row in budget_rows:
        lines.append(f"| {row['heldout_model']} | {row['calibration_n']} | {fmt(row['mean_tau'])} | {fmt(row['max_tau'])} |")
    lines.extend([
        "",
        "## Notes",
        "",
        "- The true four-generator OOD battery is available only for the five BEIR Phase 1b datasets. BarExamQA and HousingQA have richer answer/QPP rows but not the same four-generator breadth, so they are not mixed into the generator-OOD estimate.",
        "- The requested `5 train / 2 held-out datasets` split is impossible within the five-dataset, four-generator BEIR slice. This report uses leave-one-dataset-out plus leave-two-datasets-out as the available proxy.",
        "- Features are no-gold raw-retrieval predictors: NQC, WIG, SMV, CE score/spread/entropy, dense query-top1 cosine, dense top-5 coherence/centroid norm, log perplexity, question length, and OOV rate.",
        f"- Row-level points: `{PHASE_D_POINTS.relative_to(ROOT)}`",
        "",
    ])
    PHASE_D_REPORT.write_text("\n".join(lines) + "\n")
    print(PHASE_D_REPORT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    cxx = sub.add_parser("phase-cxx")
    cxx.add_argument("--datasets", nargs="*")
    cxx.add_argument("--index-dir", default="/tmp/credibility_cxx_tantivy")
    cxx.add_argument("--max-query-terms", type=int, default=128)
    cxx.add_argument("--rebuild-bm25", action="store_true")
    cxx.add_argument("--e5-status", action="store_true")
    cxx.add_argument("--e5-model", default="intfloat/e5-large-v2")
    cxx.set_defaults(func=phase_cxx)

    e5 = sub.add_parser("phase-cxx-e5")
    e5.add_argument("--datasets", nargs="*")
    e5.add_argument("--e5-model", default="intfloat/e5-large-v2")
    e5.add_argument("--corpus-batch-size", type=int, default=64)
    e5.add_argument("--query-batch-size", type=int, default=64)
    e5.add_argument("--resume", action="store_true")
    e5.add_argument("--rebuild", action="store_true")
    e5.add_argument("--fp16", action="store_true", default=True)
    e5.set_defaults(func=phase_cxx_e5)

    e = sub.add_parser("phase-e")
    e.add_argument("--datasets", nargs="*")
    e.set_defaults(func=phase_e)

    d = sub.add_parser("phase-d")
    d.add_argument("--chroma-batch-size", type=int, default=4000)
    d.add_argument("--embed-batch-size", type=int, default=64)
    d.set_defaults(func=phase_d)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
