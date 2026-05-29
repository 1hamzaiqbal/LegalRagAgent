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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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
PHASE_D_REPORT = ROOT / "docs/generated/credibility_D_ood_predictor_2026-05-29.md"
PHASE_D_POINTS = ROOT / "docs/generated/credibility_D_ood_predictor_2026-05-29_points.jsonl"
FINAL_REPORT = ROOT / "docs/generated/credibility_comprehensive_summary_2026-05-29.md"


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

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
