#!/usr/bin/env python3
"""Report covered and missing HousingQA Gemma rag_simple rows.

This is a read-only operator helper for chunked recovery runs. It uses the
state-filter raw-question retrieval cache as the canonical HousingQA row order,
then scans collision-safe answer detail logs, including sample-suffixed chunks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO_ROOT / "caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl"
DEFAULT_LOGS = REPO_ROOT / "logs"


def read_jsonl(path: Path):
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def canonical_labels(cache: Path) -> list[str]:
    labels: list[str] = []
    for row in read_jsonl(cache):
        label = str(row.get("label") or row.get("question_id") or row.get("idx") or "")
        if label:
            labels.append(label)
    return labels


def observed_labels(logs_dir: Path, provider: str, mode: str) -> dict[str, Path]:
    pattern = f"eval_{mode}_{provider}_*_housing_*{mode}-nfull-k5*_detail.jsonl"
    observed: dict[str, Path] = {}
    for path in sorted(logs_dir.glob(pattern)):
        try:
            rows = read_jsonl(path)
            for row in rows:
                if (
                    row.get("provider") == provider
                    and row.get("mode") == mode
                    and row.get("dataset") == "housing"
                    and row.get("housing_state_filter") is True
                ):
                    label = str(row.get("label") or row.get("question_id") or "")
                    if label:
                        observed[label] = path
        except Exception:
            continue
    return observed


def compact_ranges(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ranges: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            ranges.append((start, prev))
            start = prev = idx
    ranges.append((start, prev))
    return ranges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default="or-gemma4-26b")
    parser.add_argument("--mode", default="rag_simple")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--max-ranges", type=int, default=40)
    args = parser.parse_args()

    cache = args.cache if args.cache.is_absolute() else REPO_ROOT / args.cache
    logs_dir = args.logs_dir if args.logs_dir.is_absolute() else REPO_ROOT / args.logs_dir
    labels = canonical_labels(cache)
    observed = observed_labels(logs_dir, args.provider, args.mode)
    canonical = set(labels)
    covered = canonical.intersection(observed)
    unexpected = sorted(set(observed).difference(canonical))
    missing_indices = [idx for idx, label in enumerate(labels) if label not in covered]

    print(
        f"provider={args.provider} mode={args.mode} "
        f"canonical={len(labels)} covered={len(covered)} missing={len(missing_indices)} "
        f"unexpected={len(unexpected)}"
    )
    for start, end in compact_ranges(missing_indices)[: args.max_ranges]:
        print(
            f"missing_range {start}:{end + 1} count={end - start + 1} "
            f"first={labels[start]} last={labels[end]}"
        )
    if unexpected:
        for label in unexpected[: args.max_ranges]:
            print(f"unexpected_label {label} source={observed[label].relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
