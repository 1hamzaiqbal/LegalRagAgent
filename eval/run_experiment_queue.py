#!/usr/bin/env python3
"""Batch experiment queue runner.

Runs a predefined list of (mode, provider, N) experiments sequentially,
with rate-limit-aware delays between runs. Skips experiments that already
have logs with matching (mode, provider, N) in the logs/ directory.

Usage:
    uv run python eval/run_experiment_queue.py                # run all queued experiments
    uv run python eval/run_experiment_queue.py --dry-run      # preview what would run
    uv run python eval/run_experiment_queue.py --skip N       # skip first N experiments
    uv run python eval/run_experiment_queue.py --only-phase 1 # run only phase 1
"""

import argparse
import functools
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

from eval_config import EvalConfig, load_questions


# ──────────────────────────────────────────────────────────────
# EXPERIMENT QUEUE — edit this list to add/remove experiments
# Format: (mode, provider, questions, tag_comment)
# ──────────────────────────────────────────────────────────────
EXPERIMENTS = [
    # ═══════════════════════════════════════════════════════════
    # Phase 1: Small model baselines (llm_only) — full test set
    # Goal: understand how model size affects raw QA performance
    # ═══════════════════════════════════════════════════════════
    ("llm_only",        "or-qwen3-8b",    "full",  "qwen3-8b baseline full"),
    ("llm_only",        "or-qwen3-14b",   "full",  "qwen3-14b baseline full"),
    ("llm_only",        "or-qwen3-32b",   "full",  "qwen3-32b baseline full"),
    ("llm_only",        "groq-qwen",      "full",  "qwen3-32b groq baseline full"),
    ("llm_only",        "groq-llama8b",   "full",  "llama3.1-8b baseline full"),
    ("llm_only",        "or-nemotron",    "full",  "nemotron-9b baseline full"),
    ("llm_only",        "or-qwen35-9b",   "full",  "qwen3.5-9b baseline full"),

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Golden passage (isolate retrieval vs reasoning)
    # Prof feedback: "replace retrieved with golden to see if improves"
    # ═══════════════════════════════════════════════════════════
    ("golden_passage",  "or-qwen3-8b",    "full",  "golden qwen3-8b"),
    ("golden_passage",  "or-qwen3-14b",   "full",  "golden qwen3-14b"),
    ("golden_passage",  "groq-llama8b",   "full",  "golden llama-8b"),
    ("golden_passage",  "or-qwen35-9b",   "full",  "golden qwen3.5-9b"),

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Core RAG modes on small models
    # Key question: does RAG help or hurt smaller models?
    # ═══════════════════════════════════════════════════════════
    # --- rag_simple (1 LLM call, cheapest RAG) ---
    ("rag_simple",      "or-qwen3-8b",    "full",  "simple-rag qwen3-8b"),
    ("rag_simple",      "or-qwen3-14b",   "full",  "simple-rag qwen3-14b"),
    ("rag_simple",      "groq-llama8b",   "full",  "simple-rag llama-8b"),

    # --- rag_hyde (2 calls: HyDE passage + answer) ---
    ("rag_hyde",        "or-qwen3-8b",    "full",  "hyde qwen3-8b"),
    ("rag_hyde",        "or-qwen3-14b",   "full",  "hyde qwen3-14b"),
    ("rag_hyde",        "groq-llama8b",   "full",  "hyde llama-8b"),

    # --- rag_snap_hyde (3 calls: snap + HyDE + answer) ---
    ("rag_snap_hyde",   "or-qwen3-8b",    "full",  "snap-hyde qwen3-8b"),
    ("rag_snap_hyde",   "or-qwen3-14b",   "full",  "snap-hyde qwen3-14b"),
    ("rag_snap_hyde",   "groq-llama8b",   "full",  "snap-hyde llama-8b"),

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Devil RAG — weak model researches INCORRECT answer
    # Meeting idea: "what if we asked it to research incorrect"
    # ═══════════════════════════════════════════════════════════
    ("rag_devil_hyde",  "or-qwen3-8b",    "full",  "devil-hyde qwen3-8b"),
    ("rag_devil_hyde",  "or-qwen3-14b",   "full",  "devil-hyde qwen3-14b"),

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Confidence / self-consistency on small models
    # Meeting idea: "use multiple queries, get confidence"
    # ═══════════════════════════════════════════════════════════
    ("confidence_gated", "or-qwen3-8b",   "full",  "conf-gated qwen3-8b"),
    ("confidence_gated", "or-qwen3-14b",  "full",  "conf-gated qwen3-14b"),

    # ═══════════════════════════════════════════════════════════
    # Phase 6: CE-threshold on small models
    # ═══════════════════════════════════════════════════════════
    ("ce_threshold",    "or-qwen3-8b",    "full",  "ce-threshold qwen3-8b"),
    ("ce_threshold",    "or-qwen3-14b",   "full",  "ce-threshold qwen3-14b"),
]


@functools.lru_cache(maxsize=None)
def expected_question_count(questions: str, dataset: str = "barexam") -> int:
    """Resolve the actual question count for a queued eval request.

    Avoid hardcoding a stale "full" threshold — the authoritative size lives in
    eval_config/load_questions and can drift as datasets are filtered/updated.
    """
    if questions in ("full", "curated"):
        config = EvalConfig(dataset=dataset, questions=questions)
        return len(load_questions(config))
    return int(questions)


def find_existing_logs(mode: str, provider: str, n_questions: str, dataset: str = "barexam") -> list:
    """Check if a matching experiment log already exists."""
    pattern = f"logs/eval_{mode}_{provider}_*_detail.jsonl"
    matches = glob.glob(pattern)
    results = []
    expected = expected_question_count(n_questions, dataset)
    for m in matches:
        line_count = sum(1 for _ in open(m))
        if line_count >= expected:
            results.append((m, line_count))
    return results


def run_experiment(mode: str, provider: str, questions: str, tag: str, dry_run: bool = False):
    """Run a single experiment via eval_harness.py."""
    cmd = [
        "uv", "run", "python", "eval/eval_harness.py",
        "--mode", mode,
        "--provider", provider,
        "--questions", questions,
        "--tag", tag,
    ]
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running: {mode} / {provider} / N={questions}")
    print(f"  Tag: {tag}")
    print(f"  Cmd: {' '.join(cmd)}")
    sys.stdout.flush()

    if dry_run:
        print("  [DRY RUN] Skipping actual execution")
        return None

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)  # 2h max
        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"  ✓ Completed in {elapsed:.0f}s ({elapsed/60:.1f}m)")
            return True
        else:
            print(f"  ✗ Failed (exit code {result.returncode}) after {elapsed:.0f}s")
            return False
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        print(f"  ✗ Timed out after {elapsed:.0f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ✗ Error: {e} after {elapsed:.0f}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N experiments")
    parser.add_argument("--only-phase", type=int, help="Run only experiments from phase N (1-6)")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Phase boundaries for --only-phase
    phase_ranges = {
        1: (0, 7),    # small model baselines
        2: (7, 11),   # golden passage
        3: (11, 23),  # RAG configs (simple, hyde, snap_hyde)
        4: (23, 25),  # devil-rag
        5: (25, 27),  # confidence-gated
        6: (27, 29),  # ce-threshold
    }

    experiments = EXPERIMENTS[args.skip:]
    if args.only_phase and args.only_phase in phase_ranges:
        start, end = phase_ranges[args.only_phase]
        experiments = EXPERIMENTS[start:end]

    total = len(experiments)
    skipped = completed = failed = previewed = 0

    print(f"\n{'='*70}")
    print(f"EXPERIMENT QUEUE: {total} experiments")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    sys.stdout.flush()

    for i, (mode, provider, questions, tag) in enumerate(experiments, 1):
        # Check for existing results
        existing = find_existing_logs(mode, provider, questions)
        if existing:
            print(f"\n[{i}/{total}] SKIP {mode}/{provider} — already have {existing[0][1]} results")
            skipped += 1
            continue

        print(f"\n[{i}/{total}] QUEUED: {mode}/{provider}/N={questions}")
        sys.stdout.flush()
        success = run_experiment(mode, provider, questions, tag, dry_run=args.dry_run)

        if success is True:
            completed += 1
        elif success is False:
            failed += 1
        else:
            previewed += 1

        # Brief pause between experiments to respect rate limits
        if i < total and not args.dry_run:
            print("  Pausing 10s between experiments...")
            sys.stdout.flush()
            time.sleep(10)

    print(f"\n{'='*70}")
    if args.dry_run:
        print(f"DONE: {previewed} previewed, {skipped} skipped, {failed} failed")
    else:
        print(f"DONE: {completed} completed, {skipped} skipped, {failed} failed")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
