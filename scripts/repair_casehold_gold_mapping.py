#!/usr/bin/env python3
"""Repair local CaseHOLD gold_idx and holdings corpus without re-downloading."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def stable_holding_id(text: str, prefix: str = "casehold") -> str:
    normalized = " ".join(str(text or "").split())
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def repair(data_dir: Path) -> dict[str, int]:
    holding_text: dict[str, str] = {}
    holding_sources: dict[str, set[str]] = {}
    split_counts: dict[str, int] = {}

    for split in ("train", "test"):
        path = data_dir / f"{split}.csv"
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        df = pd.read_csv(path)
        gold_ids = []
        for _, row in df.iterrows():
            answer = str(row["answer"]).strip().upper()
            if answer not in {"A", "B", "C", "D", "E"}:
                raise SystemExit(f"{path}: invalid answer {answer!r}")
            for letter in ("a", "b", "c", "d", "e"):
                text = str(row[f"choice_{letter}"])
                idx = stable_holding_id(text)
                holding_text.setdefault(idx, text)
                holding_sources.setdefault(idx, set()).add(split)
            gold_text = str(row[f"choice_{answer.lower()}"])
            gold_ids.append(stable_holding_id(gold_text))
        df["gold_idx"] = gold_ids
        df.to_csv(path, index=False)
        split_counts[split] = len(df)

    corpus_rows = [
        {
            "idx": idx,
            "text": text,
            "source": "casehold_" + "+".join(sorted(holding_sources[idx])),
        }
        for idx, text in sorted(holding_text.items())
    ]
    pd.DataFrame(corpus_rows).to_csv(data_dir / "holdings_corpus.csv", index=False)
    split_counts["holdings_corpus"] = len(corpus_rows)
    return split_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "datasets/casehold")
    args = parser.parse_args()
    counts = repair(args.data_dir)
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
