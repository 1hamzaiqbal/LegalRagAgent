#!/usr/bin/env python3
"""Normalize MuSiQue validation split for per-question dense retrieval."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets" / "musique"
REPORT = REPO_ROOT / "docs" / "generated" / "musique_setup_2026-05-28.md"
HF_DATASET = "dgslibisey/MuSiQue"
HF_SPLIT = "validation"


def paragraph_id(q_id: str, para_i: int) -> str:
    return f"musique_{q_id}_{para_i}"


def _paragraphs(row: dict[str, Any]) -> list[dict[str, Any]]:
    paragraphs = row.get("paragraphs") or []
    out: list[dict[str, Any]] = []
    for fallback_i, paragraph in enumerate(paragraphs):
        para_i = int(paragraph.get("idx", fallback_i))
        out.append({
            "para_idx": para_i,
            "title": str(paragraph.get("title", "")),
            "text": str(paragraph.get("paragraph_text", "")),
            "is_supporting": bool(paragraph.get("is_supporting", False)),
        })
    out.sort(key=lambda item: item["para_idx"])
    return out


def _support_indices(row: dict[str, Any], paragraphs: list[dict[str, Any]]) -> list[int]:
    support = [int(p["para_idx"]) for p in paragraphs if p.get("is_supporting")]
    if support:
        return list(dict.fromkeys(support))
    out: list[int] = []
    for step in row.get("question_decomposition") or []:
        raw = step.get("paragraph_support_idx")
        try:
            out.append(int(raw))
        except Exception:
            continue
    return list(dict.fromkeys(out))


def normalize() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(HF_DATASET, split=HF_SPLIT)

    question_rows: list[dict[str, Any]] = []
    passage_rows: list[dict[str, Any]] = []
    gold_counts: Counter[int] = Counter()
    paragraph_counts: Counter[int] = Counter()
    hop_counts: Counter[int] = Counter()
    filtered_no_gold = 0
    filtered_unanswerable = 0

    for row in ds:
        if row.get("answerable") is False:
            filtered_unanswerable += 1
            continue
        q_id = str(row["id"])
        paragraphs = _paragraphs(row)
        support_idxs = _support_indices(row, paragraphs)
        if not support_idxs:
            filtered_no_gold += 1
            continue

        support_set = set(support_idxs)
        gold_ids = [paragraph_id(q_id, para_i) for para_i in support_idxs]
        support_titles: list[str] = []
        for paragraph in paragraphs:
            is_supporting = paragraph["para_idx"] in support_set
            if is_supporting:
                support_titles.append(str(paragraph["title"]))
            passage_rows.append({
                "q_id": q_id,
                "idx": paragraph_id(q_id, paragraph["para_idx"]),
                "para_idx": paragraph["para_idx"],
                "title": paragraph["title"],
                "text": paragraph["text"],
                "is_supporting": int(is_supporting),
            })

        aliases = row.get("answer_aliases") or []
        decomposition = row.get("question_decomposition") or []
        question_rows.append({
            "idx": q_id,
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "answer_aliases": json.dumps([str(alias) for alias in aliases]),
            "answerable": int(bool(row.get("answerable", True))),
            "gold_idx": ",".join(gold_ids),
            "gold_titles": json.dumps(support_titles),
            "support_para_idxs": json.dumps(support_idxs),
            "hop_count": len(decomposition) if decomposition else len(support_idxs),
        })
        gold_counts[len(gold_ids)] += 1
        paragraph_counts[len(paragraphs)] += 1
        hop_counts[question_rows[-1]["hop_count"]] += 1

    if not question_rows:
        raise RuntimeError("No MuSiQue questions were normalized")

    questions_path = OUT_DIR / "questions.csv"
    passages_path = OUT_DIR / "passages.csv"
    with questions_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(question_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(question_rows)
    with passages_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(passage_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(passage_rows)

    return {
        "dataset": HF_DATASET,
        "split": HF_SPLIT,
        "questions": len(question_rows),
        "passages": len(passage_rows),
        "questions_path": questions_path,
        "passages_path": passages_path,
        "filtered_unanswerable": filtered_unanswerable,
        "filtered_no_gold": filtered_no_gold,
        "gold_counts": dict(sorted(gold_counts.items())),
        "paragraph_counts": dict(sorted(paragraph_counts.items())),
        "hop_counts": dict(sorted(hop_counts.items())),
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# MuSiQue Setup - 2026-05-28",
        "",
        (
            f"Normalized Hugging Face `{summary['dataset']}`, split `{summary['split']}`, "
            "for per-question dense retrieval over the provided candidate paragraphs. No files under `paper/` were edited."
        ),
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Questions | {summary['questions']} |",
        f"| Candidate paragraphs | {summary['passages']} |",
        f"| Filtered unanswerable rows | {summary['filtered_unanswerable']} |",
        f"| Filtered rows without gold support | {summary['filtered_no_gold']} |",
        "",
        f"Gold paragraph count distribution: `{json.dumps(summary['gold_counts'], sort_keys=True)}`.",
        "",
        f"Candidate paragraph count distribution: `{json.dumps(summary['paragraph_counts'], sort_keys=True)}`.",
        "",
        f"Hop-count distribution: `{json.dumps(summary['hop_counts'], sort_keys=True)}`.",
        "",
        "Outputs:",
        f"- `{summary['questions_path'].relative_to(REPO_ROOT)}`",
        f"- `{summary['passages_path'].relative_to(REPO_ROOT)}`",
    ]
    REPORT.write_text("\n".join(lines) + "\n")


def main() -> None:
    summary = normalize()
    write_report(summary)
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in summary.items()}, indent=2, sort_keys=True))
    print(REPORT)


if __name__ == "__main__":
    main()
