#!/usr/bin/env python3
"""Normalize HotpotQA distractor validation split for per-question retrieval."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "datasets" / "hotpotqa_distractor"
REPORT = REPO_ROOT / "docs" / "generated" / "hotpotqa_distractor_setup_2026-05-27.md"


def paragraph_id(q_id: str, para_i: int) -> str:
    return f"hotpotqa_{q_id}_{para_i}"


def supporting_title_map(row: dict[str, Any]) -> dict[str, list[int]]:
    facts = row.get("supporting_facts") or {}
    titles = facts.get("title") or []
    sent_ids = facts.get("sent_id") or []
    out: dict[str, list[int]] = {}
    for title, sent_id in zip(titles, sent_ids):
        try:
            sent_i = int(sent_id)
        except Exception:
            continue
        out.setdefault(str(title), []).append(sent_i)
    return out


def normalize() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    question_rows: list[dict[str, Any]] = []
    passage_rows: list[dict[str, Any]] = []
    missing_gold_titles = 0
    gold_counts: Counter[int] = Counter()
    paragraph_counts: Counter[int] = Counter()

    for row in ds:
        q_id = str(row["id"])
        context = row.get("context") or {}
        titles = list(context.get("title") or [])
        sentence_groups = list(context.get("sentences") or [])
        support_by_title = supporting_title_map(row)
        title_to_ids: dict[str, list[str]] = {}

        for para_i, (title, sentences) in enumerate(zip(titles, sentence_groups)):
            text = " ".join(str(sent).strip() for sent in (sentences or []) if str(sent).strip())
            idx = paragraph_id(q_id, para_i)
            is_supporting = str(title) in support_by_title
            title_to_ids.setdefault(str(title), []).append(idx)
            passage_rows.append({
                "q_id": q_id,
                "idx": idx,
                "para_idx": para_i,
                "title": str(title),
                "text": text,
                "is_supporting": int(is_supporting),
                "supporting_sent_ids": json.dumps(support_by_title.get(str(title), [])),
            })

        gold_ids: list[str] = []
        missing_for_row = False
        for title in support_by_title:
            ids = title_to_ids.get(title) or []
            if not ids:
                missing_for_row = True
            gold_ids.extend(ids)
        gold_ids = list(dict.fromkeys(gold_ids))
        if missing_for_row:
            missing_gold_titles += 1
        gold_counts[len(gold_ids)] += 1
        paragraph_counts[len(titles)] += 1

        question_rows.append({
            "idx": q_id,
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "type": str(row.get("type", "")),
            "level": str(row.get("level", "")),
            "gold_idx": ",".join(gold_ids),
            "gold_titles": json.dumps(list(support_by_title)),
            "answer_aliases": "[]",
        })

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
        "questions": len(question_rows),
        "passages": len(passage_rows),
        "questions_path": questions_path,
        "passages_path": passages_path,
        "missing_gold_titles": missing_gold_titles,
        "gold_counts": dict(sorted(gold_counts.items())),
        "paragraph_counts": dict(sorted(paragraph_counts.items())),
    }


def write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# HotpotQA Distractor Setup - 2026-05-27",
        "",
        "Normalized Hugging Face `hotpot_qa`, config `distractor`, validation split for per-question retrieval. No files under `paper/` were edited.",
        "",
        "| Field | Value |",
        "|---|---:|",
        f"| Questions | {summary['questions']} |",
        f"| Candidate paragraphs | {summary['passages']} |",
        f"| Rows with missing supporting title | {summary['missing_gold_titles']} |",
        "",
        f"Gold paragraph count distribution: `{json.dumps(summary['gold_counts'], sort_keys=True)}`.",
        "",
        f"Candidate paragraph count distribution: `{json.dumps(summary['paragraph_counts'], sort_keys=True)}`.",
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
