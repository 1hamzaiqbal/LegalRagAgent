"""Token / cost analysis across evaluation runs.

Reads all eval_*_detail.jsonl files from logs/ and produces a comparison
table grouped by (mode, provider).  For combinations with N >= 100 it
computes token totals, averages, accuracy, and a cost-efficiency metric
(accuracy gain per 1K tokens relative to llm_only baseline for the same
provider).

Usage:
    uv run python eval/token_analysis.py
    uv run python eval/token_analysis.py --min-n 50      # lower threshold
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
OUTPUT_CSV = LOGS_DIR / "token_analysis.csv"


# ── helpers ──────────────────────────────────────────────────────────
def load_all_detail_records(logs_dir: Path) -> list[dict]:
    """Load every record from every eval_*_detail.jsonl in *logs_dir*."""
    records: list[dict] = []
    patterns = [
        str(logs_dir / "eval_*_detail.jsonl"),
        str(logs_dir / "*" / "eval_*_detail.jsonl"),   # subdirs like old_gemma_bak/
    ]
    seen_paths: set[str] = set()
    for pat in patterns:
        for path in glob.glob(pat):
            if path in seen_paths:
                continue
            seen_paths.add(path)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
    return records


def aggregate(records: list[dict], min_n: int) -> list[dict]:
    """Return one row per (mode, provider) with N >= *min_n*."""
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        mode = r.get("mode", "unknown")
        provider = r.get("provider", "unknown")
        buckets[(mode, provider)].append(r)

    rows: list[dict] = []
    for (mode, provider), recs in buckets.items():
        n = len(recs)
        if n < min_n:
            continue
        total_in = sum(r.get("input_tokens", 0) or 0 for r in recs)
        total_out = sum(r.get("output_tokens", 0) or 0 for r in recs)
        total_tok = total_in + total_out
        avg_tok = total_tok / n if n else 0
        avg_calls = sum(r.get("llm_calls", 0) or 0 for r in recs) / n if n else 0
        correct = sum(1 for r in recs if r.get("is_correct"))
        accuracy = correct / n if n else 0

        # Count records that actually have token data (non-zero)
        tok_records = sum(1 for r in recs if (r.get("input_tokens", 0) or 0) + (r.get("output_tokens", 0) or 0) > 0)

        rows.append({
            "mode": mode,
            "provider": provider,
            "n": n,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_tok,
            "avg_tokens_per_q": round(avg_tok, 1),
            "avg_llm_calls": round(avg_calls, 2),
            "accuracy": round(accuracy * 100, 2),
            "correct": correct,
            "tok_records": tok_records,
            "cost_efficiency": None,  # filled below
        })
    return rows


def compute_cost_efficiency(rows: list[dict]) -> None:
    """Compute accuracy_gain_per_1K_tokens relative to llm_only baseline."""
    # Build baseline lookup: provider -> accuracy (from llm_only rows)
    baselines: dict[str, dict] = {}
    for r in rows:
        if r["mode"] == "llm_only":
            prov = r["provider"]
            # If multiple llm_only runs for same provider, keep the one with most data
            if prov not in baselines or r["n"] > baselines[prov]["n"]:
                baselines[prov] = r

    for r in rows:
        prov = r["provider"]
        baseline = baselines.get(prov)
        if baseline is None:
            # No llm_only baseline for this provider — can't compute
            r["cost_efficiency"] = None
            continue
        if r["mode"] == "llm_only":
            r["cost_efficiency"] = 0.0
            continue
        acc_gain = r["accuracy"] - baseline["accuracy"]  # percentage points
        avg_tok = r["avg_tokens_per_q"]
        if avg_tok > 0:
            # accuracy gain per 1K tokens used
            r["cost_efficiency"] = round(acc_gain / (avg_tok / 1000), 4)
        else:
            r["cost_efficiency"] = None


# ── display ──────────────────────────────────────────────────────────
COLUMNS = [
    ("mode", 26),
    ("provider", 18),
    ("n", 6),
    ("accuracy", 9),
    ("avg_tok/q", 10),
    ("avg_calls", 10),
    ("tot_in_tok", 12),
    ("tot_out_tok", 12),
    ("tok_records", 11),
    ("cost_eff", 10),
]


def fmt_row(r: dict) -> list[str]:
    ce = r["cost_efficiency"]
    ce_str = f"{ce:+.4f}" if ce is not None else "n/a"
    return [
        r["mode"],
        r["provider"],
        str(r["n"]),
        f"{r['accuracy']:.2f}%",
        f"{r['avg_tokens_per_q']:.0f}",
        f"{r['avg_llm_calls']:.2f}",
        f"{r['total_input_tokens']:,}",
        f"{r['total_output_tokens']:,}",
        str(r["tok_records"]),
        ce_str,
    ]


def print_table(rows: list[dict]) -> None:
    # header
    header = "  ".join(name.ljust(w) for name, w in COLUMNS)
    sep = "  ".join("-" * w for _, w in COLUMNS)
    print(header)
    print(sep)
    prev_mode = None
    for r in rows:
        if prev_mode is not None and r["mode"] != prev_mode:
            print()  # blank line between mode groups
        prev_mode = r["mode"]
        vals = fmt_row(r)
        line = "  ".join(v.ljust(w) for v, (_, w) in zip(vals, COLUMNS))
        print(line)


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "mode", "provider", "n", "accuracy", "avg_tokens_per_q",
        "avg_llm_calls", "total_input_tokens", "total_output_tokens",
        "total_tokens", "correct", "tok_records", "cost_efficiency",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\n✅ CSV written to {path}")


# ── main ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Token / cost analysis")
    parser.add_argument("--min-n", type=int, default=100,
                        help="Minimum N per (mode, provider) group (default: 100)")
    args = parser.parse_args()

    print(f"Loading detail logs from {LOGS_DIR} …")
    records = load_all_detail_records(LOGS_DIR)
    print(f"Loaded {len(records):,} total records.\n")

    if not records:
        print("No records found. Exiting.")
        sys.exit(1)

    rows = aggregate(records, min_n=args.min_n)
    if not rows:
        print(f"No (mode, provider) groups with N ≥ {args.min_n}. "
              "Try --min-n to lower the threshold.")
        sys.exit(1)

    compute_cost_efficiency(rows)

    # Sort: group by mode, then within each mode sort by cost_efficiency desc
    # (None values go last)
    def sort_key(r: dict):
        ce = r["cost_efficiency"] if r["cost_efficiency"] is not None else float("-inf")
        return (r["mode"], -ce)

    rows.sort(key=sort_key)

    print(f"{'─' * 80}")
    print(f"  Token & Cost-Efficiency Analysis  (min N = {args.min_n})")
    print(f"{'─' * 80}\n")
    print_table(rows)
    write_csv(rows, OUTPUT_CSV)

    # Summary
    providers_with_baseline = {r["provider"] for r in rows if r["mode"] == "llm_only"}
    providers_without = {r["provider"] for r in rows} - providers_with_baseline
    if providers_without:
        print(f"\n⚠  No llm_only baseline for: {', '.join(sorted(providers_without))}")
        print("   cost_efficiency is n/a for those providers.")


if __name__ == "__main__":
    main()
