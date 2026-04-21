#!/usr/bin/env python3
"""Compare mini-eval detail logs to produce an apples-to-apples table.

Designed to run after scripts/hpc/slurm_gemma4_mini_eval.sh completes.
Generates three outputs:
  1) Accuracy + leakage table per mode
  2) Pairwise switch matrix (fixed/broke between every mode pair)
  3) Retrieval-quality-conditional accuracy (when available)

Usage:
    uv run python scripts/compare_mini_eval.py \
      logs/eval_rag_simple_cluster-vllm_*_detail.jsonl \
      logs/eval_rag_hyde_cluster-vllm_*_detail.jsonl \
      logs/eval_rag_snap_hyde_cluster-vllm_*_detail.jsonl \
      logs/eval_snap_only_in_final_cluster-vllm_*_detail.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

ANSWER_RE = re.compile(r"Answer:\s*\(?[A-D]\)?", re.I)


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize(mode: str, rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"mode": mode, "n": 0}
    correct = sum(1 for r in rows if r.get("is_correct"))
    gold_retrieved = sum(1 for r in rows if r.get("gold_retrieved"))
    # Use evidence_store presence (non-empty) to detect modes that actually ran
    # retrieval. After the schema-normalization fix, every record has
    # gold_retrieved/evidence_store/retrieved_ids keys, so key-presence no longer
    # distinguishes no-retrieval modes from missing instrumentation.
    retrieval_events = sum(1 for r in rows if r.get("evidence_store"))

    # HyDE leak check on raw output if available
    hyde_leak = 0
    hyde_count = 0
    for r in rows:
        raw = r.get("hyde_passage_raw") or r.get("hyde_passage") or ""
        if raw:
            hyde_count += 1
            if ANSWER_RE.search(raw):
                hyde_leak += 1

    # Conditional accuracy on gold retrieved vs not — only meaningful on modes
    # that actually ran retrieval. Skip for no-retrieval modes.
    if retrieval_events > 0:
        retr_correct = sum(1 for r in rows if r.get("gold_retrieved") and r.get("is_correct"))
        notr_correct = sum(1 for r in rows if not r.get("gold_retrieved") and r.get("is_correct"))
        notr_count = n - gold_retrieved
        acc_retr = retr_correct / gold_retrieved if gold_retrieved else None
        acc_notr = notr_correct / notr_count if notr_count else None
    else:
        acc_retr = acc_notr = None

    # Latency / calls
    lat = [r.get("elapsed_sec", 0) for r in rows]
    calls = [r.get("llm_calls", 0) for r in rows]

    return {
        "mode": mode,
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "gold_retrieved_n": gold_retrieved,
        "gold_retrieved_rate": gold_retrieved / n if n else 0,
        "hyde_leak_rate": hyde_leak / hyde_count if hyde_count else None,
        "hyde_output_n": hyde_count,
        "acc_given_gold": acc_retr,
        "acc_given_no_gold": acc_notr,
        "avg_latency_sec": sum(lat) / n,
        "avg_llm_calls": sum(calls) / n,
    }


def switch_matrix(rows_by_mode: dict[str, list[dict]]) -> list[tuple[str, str, int, int, int]]:
    modes = list(rows_by_mode.keys())
    by_mode_idx = {m: {r["idx"]: r for r in rows_by_mode[m]} for m in modes}

    pairs = []
    for i, left in enumerate(modes):
        for right in modes[i + 1:]:
            common = set(by_mode_idx[left]) & set(by_mode_idx[right])
            fixed = broke = both = neither = 0
            for idx in common:
                lc = by_mode_idx[left][idx].get("is_correct")
                rc = by_mode_idx[right][idx].get("is_correct")
                if not lc and rc: fixed += 1
                elif lc and not rc: broke += 1
                elif lc and rc: both += 1
                else: neither += 1
            pairs.append((left, right, fixed, broke, len(common)))
    return pairs


def fmt_pct(x):
    if x is None:
        return "   - "
    return f"{x*100:5.1f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="Detail log paths or globs")
    args = p.parse_args()

    # Resolve globs
    resolved = []
    for pat in args.paths:
        matches = sorted(glob.glob(pat))
        if not matches:
            print(f"[warn] no match: {pat}")
        resolved.extend(matches)

    # Load, one record set per mode. If duplicate mode, keep newest by filename.
    rows_by_mode: dict[str, list[dict]] = {}
    path_by_mode: dict[str, str] = {}
    for path in resolved:
        rows = load_rows(path)
        if not rows:
            continue
        mode = rows[0].get("mode", Path(path).stem)
        # Newest wins (paths are sorted)
        rows_by_mode[mode] = rows
        path_by_mode[mode] = path

    if not rows_by_mode:
        print("No data loaded. Exit.")
        return

    print(f"\n{'=' * 100}")
    print(f"MINI-EVAL COMPARISON — {len(rows_by_mode)} modes")
    print(f"{'=' * 100}\n")

    # Table 1: accuracy + leakage
    print(f"{'mode':25} {'N':>5} {'acc':>7} {'hyde leak':>10} {'gold%':>7} "
          f"{'acc|gold':>10} {'acc|no-gold':>12} {'calls':>6} {'sec/q':>7}")
    print("-" * 100)
    summaries = []
    for mode, rows in rows_by_mode.items():
        s = summarize(mode, rows)
        summaries.append(s)
        print(
            f"{s['mode']:25} {s['n']:>5} {fmt_pct(s['accuracy']):>7} "
            f"{fmt_pct(s['hyde_leak_rate']):>10} {fmt_pct(s['gold_retrieved_rate']):>7} "
            f"{fmt_pct(s['acc_given_gold']):>10} {fmt_pct(s['acc_given_no_gold']):>12} "
            f"{s['avg_llm_calls']:>6.2f} {s['avg_latency_sec']:>7.1f}"
        )

    print("\nFiles used:")
    for m, p in path_by_mode.items():
        print(f"  {m:25} {p}")

    # Table 2: switch matrix
    pairs = switch_matrix(rows_by_mode)
    if pairs:
        print(f"\n{'=' * 100}")
        print("SWITCH MATRIX (left → right)")
        print(f"{'=' * 100}")
        print(f"{'left':25} {'right':25} {'fixed':>7} {'broke':>7} {'net':>7} {'N':>5}")
        print("-" * 85)
        for left, right, fixed, broke, n in pairs:
            print(f"{left:25} {right:25} {fixed:>7} {broke:>7} {fixed - broke:>+7} {n:>5}")


if __name__ == "__main__":
    main()
