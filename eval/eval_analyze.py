"""Analyze and compare evaluation runs from experiments.jsonl.

Usage:
    uv run python eval/eval_analyze.py                                    # Print all runs
    uv run python eval/eval_analyze.py --filter mode=llm_only             # Filter runs
    uv run python eval/eval_analyze.py --filter provider=deepseek         # Filter by provider
    uv run python eval/eval_analyze.py --diff RUN_ID_1 RUN_ID_2          # Per-question diff
    uv run python eval/eval_analyze.py --last 5                           # Show last 5 runs
"""
import argparse
import json
import os
import sys


EXPERIMENTS_PATH = os.path.join("logs", "experiments.jsonl")


def load_experiments(path: str = EXPERIMENTS_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    runs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def load_detail_log(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"Detail log not found: {path}")
        sys.exit(1)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def filter_runs(runs: list[dict], filters: list[str]) -> list[dict]:
    for f in filters:
        if "=" not in f:
            print(f"Invalid filter: {f} (expected key=value)")
            sys.exit(1)
        key, val = f.split("=", 1)
        runs = [r for r in runs if str(r.get(key, "")) == val]
    return runs


def print_table(runs: list[dict]):
    if not runs:
        print("No runs found.")
        return

    # Header
    print(f"\n{'Run ID':<45} {'Mode':<16} {'Provider':<14} {'Model':<25} "
          f"{'Acc':>6} {'N':>4} {'Lat':>6} {'Calls':>6} {'Tag'}")
    print("-" * 140)

    for r in runs:
        acc = f"{r['accuracy']*100:.1f}%"
        lat = f"{r.get('avg_latency_sec', 0):.1f}s"
        calls = f"{r.get('avg_llm_calls', 0):.1f}"
        tag = r.get("tag", "")
        print(
            f"{r['run_id']:<45} {r['mode']:<16} {r['provider']:<14} "
            f"{r.get('model', '?'):<25} {acc:>6} {r['total']:>4} "
            f"{lat:>6} {calls:>6} {tag}"
        )

    # Subject breakdown for each run
    print(f"\n{'Subject Breakdown':}")
    print("-" * 100)
    for r in runs:
        by_subj = r.get("by_subject", {})
        if by_subj:
            subj_parts = []
            for s, (c, t) in sorted(by_subj.items()):
                subj_parts.append(f"{s}={c}/{t}")
            print(f"  {r['run_id'][:40]:<42} {', '.join(subj_parts)}")


def print_diff(run1_id: str, run2_id: str, runs: list[dict]):
    """Show per-question comparison between two runs."""
    r1 = next((r for r in runs if r["run_id"] == run1_id), None)
    r2 = next((r for r in runs if r["run_id"] == run2_id), None)

    if not r1:
        print(f"Run not found: {run1_id}")
        # Show available run IDs
        print(f"Available: {[r['run_id'] for r in runs[-10:]]}")
        sys.exit(1)
    if not r2:
        print(f"Run not found: {run2_id}")
        print(f"Available: {[r['run_id'] for r in runs[-10:]]}")
        sys.exit(1)

    log1 = load_detail_log(r1["detail_log"])
    log2 = load_detail_log(r2["detail_log"])

    # Index by label
    by_label1 = {r["label"]: r for r in log1}
    by_label2 = {r["label"]: r for r in log2}

    common = set(by_label1) & set(by_label2)
    if not common:
        print("No overlapping questions between runs.")
        return

    print(f"\n{'=' * 80}")
    print(f"DIFF: {run1_id}")
    print(f"  vs: {run2_id}")
    print(f"Overlapping questions: {len(common)}")
    print(f"{'=' * 80}\n")

    flipped_to_correct = []
    flipped_to_wrong = []
    both_correct = 0
    both_wrong = 0

    for label in sorted(common):
        q1 = by_label1[label]
        q2 = by_label2[label]
        c1 = q1.get("is_correct", False)
        c2 = q2.get("is_correct", False)

        if c1 and c2:
            both_correct += 1
        elif not c1 and not c2:
            both_wrong += 1
        elif not c1 and c2:
            flipped_to_correct.append(label)
        else:
            flipped_to_wrong.append(label)

    print(f"Both correct:     {both_correct}")
    print(f"Both wrong:       {both_wrong}")
    print(f"Flipped FAIL→PASS: {len(flipped_to_correct)}")
    print(f"Flipped PASS→FAIL: {len(flipped_to_wrong)}")

    acc1 = sum(1 for l in common if by_label1[l].get("is_correct")) / len(common) * 100
    acc2 = sum(1 for l in common if by_label2[l].get("is_correct")) / len(common) * 100
    print(f"\nAccuracy (common set): {acc1:.1f}% → {acc2:.1f}% (delta: {acc2-acc1:+.1f}%)")

    if flipped_to_correct:
        print(f"\nGained (FAIL→PASS):")
        for label in flipped_to_correct:
            q1 = by_label1[label]
            q2 = by_label2[label]
            print(f"  {label:<35} gold={q1['correct_answer']} "
                  f"pred: {q1.get('predicted_answer','?')}→{q2.get('predicted_answer','?')}")

    if flipped_to_wrong:
        print(f"\nLost (PASS→FAIL):")
        for label in flipped_to_wrong:
            q1 = by_label1[label]
            q2 = by_label2[label]
            print(f"  {label:<35} gold={q1['correct_answer']} "
                  f"pred: {q1.get('predicted_answer','?')}→{q2.get('predicted_answer','?')}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation runs")
    parser.add_argument("--filter", action="append", default=[],
                        help="Filter runs by key=value (repeatable)")
    parser.add_argument("--diff", nargs=2, metavar=("RUN1", "RUN2"),
                        help="Compare two runs by run_id")
    parser.add_argument("--last", type=int, default=0,
                        help="Show only the last N runs")
    args = parser.parse_args()

    runs = load_experiments()
    if not runs:
        print(f"No experiments found. Run eval_harness.py first to generate {EXPERIMENTS_PATH}")
        return

    if args.diff:
        print_diff(args.diff[0], args.diff[1], runs)
        return

    runs = filter_runs(runs, args.filter)
    if args.last > 0:
        runs = runs[-args.last:]

    print_table(runs)


if __name__ == "__main__":
    main()
