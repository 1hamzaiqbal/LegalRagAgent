#!/usr/bin/env python3
"""Freeze a small, deterministic Reasoning Gym pilot for elasticity EDA.

This script does not run a model. It materializes paired tasks whose verifier
reward can later be combined with independently logged resource costs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import reasoning_gym as rg


FAMILIES = {
    "prime_factorization": {
        "easy": {"min_value": 2, "max_value": 100},
        "medium": {"min_value": 101, "max_value": 5_000},
        "hard": {"min_value": 5_001, "max_value": 10_000},
    },
    "bitwise_arithmetic": {
        "easy": {"difficulty": 1},
        "medium": {"difficulty": 4},
        "hard": {"difficulty": 8},
    },
    "countdown": {
        "easy": {
            "min_numbers": 3,
            "max_numbers": 3,
            "min_value": 1,
            "max_value": 25,
            "min_target": 10,
            "max_target": 100,
        },
        "medium": {
            "min_numbers": 4,
            "max_numbers": 6,
            "min_value": 1,
            "max_value": 100,
            "min_target": 100,
            "max_target": 999,
        },
        "hard": {
            "min_numbers": 7,
            "max_numbers": 9,
            "min_value": 1,
            "max_value": 200,
            "min_target": 500,
            "max_target": 5_000,
        },
    },
    "shortest_path": {
        "easy": {"min_rows": 5, "max_rows": 8, "min_cols": 5, "max_cols": 8, "p_blocked": 0.30},
        "medium": {
            "min_rows": 10,
            "max_rows": 15,
            "min_cols": 10,
            "max_cols": 15,
            "p_blocked": 0.35,
        },
        "hard": {
            "min_rows": 20,
            "max_rows": 25,
            "min_cols": 20,
            "max_cols": 25,
            "p_blocked": 0.35,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-cell", type=int, default=25)
    parser.add_argument("--base-seed", type=int, default=2026071700)
    return parser.parse_args()


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cell_summaries = []
    seen_normalized_questions: set[str] = set()

    for family_idx, (family, tiers) in enumerate(FAMILIES.items()):
        for tier_idx, (tier, config) in enumerate(tiers.items()):
            seed = args.base_seed + family_idx * 100 + tier_idx * 10
            candidate_budget = args.samples_per_cell * 20
            dataset = rg.create_dataset(family, seed=seed, size=candidate_budget, **config)
            question_lengths = []
            answer_lengths = []
            generation_failures = 0
            duplicate_questions_skipped = 0
            accepted = 0
            for source_index in range(candidate_budget):
                if accepted >= args.samples_per_cell:
                    break
                try:
                    entry = dataset[source_index]
                except ValueError:
                    generation_failures += 1
                    continue
                normalized_question = " ".join(entry["question"].split())
                if normalized_question in seen_normalized_questions:
                    duplicate_questions_skipped += 1
                    continue
                stable_id = hashlib.sha256(
                    f"{family}|{tier}|{seed}|{source_index}|{entry['question']}".encode("utf-8")
                ).hexdigest()[:20]
                row = {
                    "id": stable_id,
                    "task_family": family,
                    "difficulty_tier": tier,
                    "generator_seed": seed,
                    "source_index": source_index,
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "metadata": entry["metadata"],
                }
                rows.append(row)
                seen_normalized_questions.add(normalized_question)
                question_lengths.append(len(entry["question"]))
                answer_lengths.append(len(str(entry["answer"])))
                accepted += 1

            if accepted != args.samples_per_cell:
                raise RuntimeError(
                    f"{family}/{tier}: accepted {accepted}/{args.samples_per_cell} tasks "
                    f"after {candidate_budget} deterministic candidate indices"
                )

            cell_summaries.append(
                {
                    "task_family": family,
                    "difficulty_tier": tier,
                    "generator_seed": seed,
                    "n": args.samples_per_cell,
                    "candidate_indices_examined": accepted + generation_failures + duplicate_questions_skipped,
                    "generation_failures": generation_failures,
                    "duplicate_questions_skipped": duplicate_questions_skipped,
                    "config": config,
                    "question_chars_min": min(question_lengths),
                    "question_chars_median": statistics.median(question_lengths),
                    "question_chars_max": max(question_lengths),
                    "answer_chars_median": statistics.median(answer_lengths),
                }
            )

    data_path = args.output_dir / "rg_pilot.jsonl"
    data_path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "generator": "reasoning-gym",
        "base_seed": args.base_seed,
        "samples_per_cell": args.samples_per_cell,
        "num_rows": len(rows),
        "cells": cell_summaries,
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "warning": (
            "Reasoning Gym native rewards are not uniformly binary. The experiment adapter must log the native "
            "score and a predeclared exact-success indicator separately; resource penalties must never overwrite either."
        ),
    }
    (args.output_dir / "rg_pilot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
