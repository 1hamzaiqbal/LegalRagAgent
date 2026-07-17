#!/usr/bin/env python3
"""Audit a frozen Reasoning Gym pilot against its native scorers."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import reasoning_gym as rg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    configs = {
        (cell["task_family"], cell["difficulty_tier"]): cell["config"] for cell in summary["cells"]
    }
    datasets = {
        key: rg.create_dataset(key[0], seed=0, size=1, **config) for key, config in configs.items()
    }

    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    oracle_failures = []
    empty_scores: dict[str, list[float]] = defaultdict(list)
    prime_partial_examples = []
    native_scores: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        family = row["task_family"]
        dataset = datasets[(family, row["difficulty_tier"])]
        entry = {"question": row["question"], "answer": row["answer"], "metadata": row["metadata"]}
        oracle_score = float(dataset.score_answer(row["answer"], entry))
        native_scores[family][str(oracle_score)] += 1
        if oracle_score != 1.0:
            oracle_failures.append({"id": row["id"], "family": family, "score": oracle_score})
        empty_scores[family].append(float(dataset.score_answer("", entry)))

        if family == "prime_factorization" and len(row["metadata"]["factors"]) > 1:
            nonprime_answer = str(row["metadata"]["number"])
            nonprime_score = float(dataset.score_answer(nonprime_answer, entry))
            if nonprime_score > 0:
                prime_partial_examples.append(
                    {"id": row["id"], "answer": nonprime_answer, "native_score": nonprime_score}
                )

    ids = [row["id"] for row in rows]
    questions = [" ".join(row["question"].split()) for row in rows]
    result = {
        "rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "unique_normalized_questions": len(set(questions)),
        "duplicate_normalized_questions": len(questions) - len(set(questions)),
        "oracle_score_failures": oracle_failures,
        "native_oracle_score_counts": {family: dict(counts) for family, counts in native_scores.items()},
        "empty_answer_nonzero_counts": {
            family: sum(score != 0.0 for score in scores) for family, scores in empty_scores.items()
        },
        "empty_answer_scores": {family: sorted(set(scores)) for family, scores in empty_scores.items()},
        "prime_nonprime_product_partial_count": len(prime_partial_examples),
        "prime_nonprime_product_partial_examples": prime_partial_examples[:5],
        "conclusion": (
            "Use task_success_exact = int(native_score == 1.0) and retain native score separately. "
            "Do not subtract resource cost from either verifier output."
        ),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if oracle_failures or len(set(ids)) != len(ids) or len(set(questions)) != len(questions):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
