#!/usr/bin/env python3
"""Download and normalize MedQA-USMLE plus MedRAG textbook chunks.

This is the only MedQA widening phase that intentionally keeps Hugging Face
online. Later embedding/eval phases can run with HF_HUB_OFFLINE=1 against the
local CSVs and cached models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "datasets" / "medqa_usmle"
REPORT = ROOT / "docs" / "generated" / "medqa_usmle_widening_2026-05-26.md"


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_questions() -> pd.DataFrame:
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(ds):
        options = dict(row["options"])
        answer = str(row["answer_idx"]).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"row {i} has invalid answer_idx={answer!r}")
        rows.append(
            {
                "idx": f"medqa_test_{i:04d}",
                "question": _clean(row["question"]),
                "choice_a": _clean(options.get("A", "")),
                "choice_b": _clean(options.get("B", "")),
                "choice_c": _clean(options.get("C", "")),
                "choice_d": _clean(options.get("D", "")),
                "answer": answer,
                "answer_text": _clean(row.get("answer", "")),
                "meta_info": _clean(row.get("meta_info", "")),
                "metamap_phrases": json.dumps(row.get("metamap_phrases") or [], ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def normalize_textbooks() -> pd.DataFrame:
    ds = load_dataset("MedRAG/textbooks", split="train")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, row in enumerate(ds):
        idx = _clean(row.get("id") or f"medrag_textbook_{i:06d}")
        if not idx:
            idx = f"medrag_textbook_{i:06d}"
        if idx in seen:
            raise ValueError(f"duplicate textbook id={idx!r}")
        seen.add(idx)
        title = _clean(row.get("title", ""))
        content = _clean(row.get("content", ""))
        text = _clean(row.get("contents", "")) or ". ".join(part for part in (title, content) if part)
        if not text:
            continue
        rows.append(
            {
                "idx": idx,
                "text": text,
                "source": "medrag_textbooks",
                "title": title,
                "content": content,
            }
        )
    return pd.DataFrame(rows)


def write_report(questions: pd.DataFrame, passages: pd.DataFrame) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    title_counts = passages["title"].value_counts().sort_index()
    meta_counts = questions["meta_info"].value_counts().sort_index()

    q0 = questions.iloc[0].to_dict()
    q1 = questions.iloc[1].to_dict()
    p0 = passages.iloc[0].to_dict()
    p1 = passages.iloc[1].to_dict()

    lines = [
        "# MedQA-USMLE Widening - 2026-05-26",
        "",
        "## Phase 0 - Download And Normalization",
        "",
        "Downloaded online Hugging Face sources and normalized them to local CSVs. Later phases should use offline HF mode unless a missing model artifact must be explicitly bootstrapped.",
        "",
        "| Artifact | Source | Local path | Rows | Notes |",
        "|---|---|---|---:|---|",
        f"| MedQA questions | `GBaker/MedQA-USMLE-4-options`, test split | `datasets/medqa_usmle/questions.csv` | {len(questions)} | Four options, gold answer normalized to A-D. |",
        f"| Textbook corpus | `MedRAG/textbooks`, train split | `datasets/medqa_usmle/textbooks.csv` | {len(passages)} | Pre-chunked textbook snippets, `idx` preserves MedRAG `id`; retrieval text uses `contents`. |",
        "",
        "Question `meta_info` counts:",
        "",
        "| meta_info | rows |",
        "|---|---:|",
    ]
    for meta, count in meta_counts.items():
        lines.append(f"| {meta} | {count} |")
    lines.extend([
        "",
        "Textbook title counts:",
        "",
        "| title | chunks |",
        "|---|---:|",
    ])
    for title, count in title_counts.items():
        lines.append(f"| {title} | {count} |")
    lines.extend([
        "",
        "Question examples:",
        "",
        f"- `{q0['idx']}` answer `{q0['answer']}` / {q0['answer_text']}: {q0['question'][:220]}",
        f"- `{q1['idx']}` answer `{q1['answer']}` / {q1['answer_text']}: {q1['question'][:220]}",
        "",
        "Passage examples:",
        "",
        f"- `{p0['idx']}` ({p0['title']}): {p0['text'][:240]}",
        f"- `{p1['idx']}` ({p1['title']}): {p1['text'][:240]}",
        "",
        "## Current Status",
        "",
        "- Phase 0 complete.",
        "- Phase 1 pending: embed `datasets/medqa_usmle/textbooks.csv` into Chroma collection `medqa_textbooks` with `Alibaba-NLP/gte-large-en-v1.5`.",
        "- MedQA has no gold passage labels; downstream answer EM is the primary outcome and no Hit@k/MRR/Recall will be reported.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run python scripts/download_medqa_usmle.py",
        "```",
    ])
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    questions = normalize_questions()
    passages = normalize_textbooks()

    questions.to_csv(args.output_dir / "questions.csv", index=False)
    passages.to_csv(args.output_dir / "textbooks.csv", index=False)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "questions_source": "GBaker/MedQA-USMLE-4-options:test",
                "questions_rows": len(questions),
                "passages_source": "MedRAG/textbooks:train",
                "passages_rows": len(passages),
                "question_columns": list(questions.columns),
                "passage_columns": list(passages.columns),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_report(questions, passages)
    print(f"Wrote {len(questions):,} questions and {len(passages):,} passages to {args.output_dir}")
    print(f"Updated {REPORT}")


if __name__ == "__main__":
    main()
