#!/usr/bin/env python3
"""Download and normalize the BEIR Phase 1 datasets from Hugging Face.

This phase intentionally runs online. Later phases can set HF_HUB_OFFLINE=1
because the normalized CSVs and HF cache will already exist locally.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "datasets" / "beir"
REPORT = REPO_ROOT / "docs" / "generated" / "beir_phase1_phase0_download_2026-05-26.md"

DATASETS = ("scifact", "nfcorpus", "fiqa", "trec-covid", "scidocs")


def beir_dataset_key(subset: str) -> str:
    return "beir_" + subset.replace("-", "_")


def combined_text(title: Any, text: Any) -> str:
    title_s = str(title or "").strip()
    text_s = str(text or "").strip()
    if title_s and text_s:
        return f"{title_s}\n\n{text_s}"
    return text_s or title_s


def normalize_subset(subset: str, out_root: Path) -> dict[str, Any]:
    out_dir = out_root / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_dataset(f"BeIR/{subset}", "corpus", split="corpus")
    queries = load_dataset(f"BeIR/{subset}", "queries", split="queries")
    qrels = load_dataset(f"BeIR/{subset}-qrels", split="test")

    corpus_rows = []
    corpus_ids: set[str] = set()
    for row in corpus:
        idx = str(row["_id"])
        corpus_ids.add(idx)
        corpus_rows.append({
            "idx": idx,
            "title": str(row.get("title") or ""),
            "text": combined_text(row.get("title"), row.get("text")),
            "source": f"BeIR/{subset}",
        })

    queries_by_id = {
        str(row["_id"]): {
            "idx": str(row["_id"]),
            "question": str(row.get("text") or ""),
            "query_title": str(row.get("title") or ""),
            "subject": subset,
            "source": f"BeIR/{subset}",
        }
        for row in queries
    }

    gold_by_query: dict[str, list[str]] = defaultdict(list)
    qrel_scores: dict[str, dict[str, int]] = defaultdict(dict)
    qrel_rows = []
    missing_query_ids: set[str] = set()
    missing_corpus_ids: set[str] = set()
    for row in qrels:
        qid = str(row["query-id"])
        cid = str(row["corpus-id"])
        score = int(row["score"])
        qrel_rows.append({"query_id": qid, "corpus_id": cid, "score": score})
        if qid not in queries_by_id:
            missing_query_ids.add(qid)
        if cid not in corpus_ids:
            missing_corpus_ids.add(cid)
        if score > 0:
            gold_by_query[qid].append(cid)
            qrel_scores[qid][cid] = score

    question_rows = []
    for qid in sorted(gold_by_query):
        if qid not in queries_by_id:
            continue
        gold_ids = sorted(dict.fromkeys(gold_by_query[qid]))
        row = dict(queries_by_id[qid])
        row["gold_idx"] = json.dumps(gold_ids)
        row["gold_scores"] = json.dumps({gid: qrel_scores[qid][gid] for gid in gold_ids}, sort_keys=True)
        row["gold_count"] = len(gold_ids)
        question_rows.append(row)

    pd.DataFrame(corpus_rows).to_csv(out_dir / "corpus.csv", index=False)
    pd.DataFrame(question_rows).to_csv(out_dir / "questions.csv", index=False)
    pd.DataFrame(qrel_rows).to_csv(out_dir / "qrels_test.csv", index=False)

    manifest = {
        "subset": subset,
        "dataset_key": beir_dataset_key(subset),
        "corpus_source": f"BeIR/{subset}:corpus",
        "queries_source": f"BeIR/{subset}:queries",
        "qrels_source": f"BeIR/{subset}-qrels:test",
        "corpus_count": len(corpus_rows),
        "queries_total": len(queries_by_id),
        "qrels_test_count": len(qrel_rows),
        "test_query_count": len(question_rows),
        "positive_qrels_test_count": sum(1 for row in qrel_rows if row["score"] > 0),
        "gold_doc_count": len({gid for ids in gold_by_query.values() for gid in ids}),
        "multi_gold_query_count": sum(1 for ids in gold_by_query.values() if len(set(ids)) > 1),
        "missing_query_id_count": len(missing_query_ids),
        "missing_corpus_id_count": len(missing_corpus_ids),
        "paths": {
            "corpus": str(out_dir / "corpus.csv"),
            "questions": str(out_dir / "questions.csv"),
            "qrels_test": str(out_dir / "qrels_test.csv"),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def write_report(manifests: list[dict[str, Any]], output: Path) -> None:
    lines = [
        "# BEIR Phase 1 Download - 2026-05-26",
        "",
        "Phase 0 downloaded BEIR corpus, query, and test-qrels splits from Hugging Face and normalized local CSVs under `datasets/beir/`. No files under `paper/` were edited.",
        "",
        "| Dataset | Eval key | Corpus docs | HF queries | Test queries with qrels | Test qrels | Positive qrels | Gold docs | Multi-gold queries | Missing query ids | Missing corpus ids |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifests:
        lines.append(
            f"| {row['subset']} | `{row['dataset_key']}` | {row['corpus_count']} | {row['queries_total']} | "
            f"{row['test_query_count']} | {row['qrels_test_count']} | {row['positive_qrels_test_count']} | "
            f"{row['gold_doc_count']} | {row['multi_gold_query_count']} | "
            f"{row['missing_query_id_count']} | {row['missing_corpus_id_count']} |"
        )
    lines.extend([
        "",
        "Local normalized files are intentionally under ignored `datasets/`; the committed artifact is this count report and the download script.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run python scripts/download_beir_phase1.py",
        "```",
        "",
    ])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = parser.parse_args()

    manifests = []
    for subset in args.datasets:
        print(f"[download] {subset}", flush=True)
        manifests.append(normalize_subset(subset, args.out_root))
    write_report(manifests, args.report)
    print(args.report)


if __name__ == "__main__":
    main()
