#!/usr/bin/env python3
"""Experiment dashboard — quick overview of all experiment status.

Usage:
    uv run python eval/monitor.py
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime

# ── Project root ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

LOGS_DIR = os.path.join(ROOT, "logs")
CHROMA_DIR = os.path.join(ROOT, "chroma_db")

# ── Import experiment/embedding lists (with fallbacks) ────────
try:
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    from run_experiment_queue import EXPERIMENTS as QUEUE_EXPERIMENTS, expected_question_count
except Exception:
    QUEUE_EXPERIMENTS = []
    expected_question_count = None

try:
    from run_embedding_comparison import EMBEDDING_CANDIDATES
except Exception:
    EMBEDDING_CANDIDATES = []


# ── Helpers ───────────────────────────────────────────────────
W = shutil.get_terminal_size((80, 24)).columns

def banner(title: str):
    print(f"\n{'─' * W}")
    print(f"  {title}")
    print(f"{'─' * W}")

def short_ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def du(path: str) -> str:
    """Human-readable disk usage via du -sh."""
    if not os.path.exists(path):
        return "n/a"
    try:
        out = subprocess.check_output(["du", "-sh", path], stderr=subprocess.DEVNULL, text=True)
        return out.split()[0]
    except Exception:
        return "?"


# ── 1. Currently Running Experiments ──────────────────────────
def show_running():
    banner("🔄  RUNNING EXPERIMENTS")
    logs = sorted(glob.glob(os.path.join(LOGS_DIR, "queue_phase*.log")))
    if not logs:
        print("  No queue_phase*.log files found — nothing running via queue.")
    for lf in logs:
        name = os.path.basename(lf)
        mtime = datetime.fromtimestamp(os.path.getmtime(lf))
        age_min = (datetime.now() - mtime).total_seconds() / 60
        stale = " ⚠ stale" if age_min > 30 else ""
        print(f"\n  📋 {name}  (last modified {mtime.strftime('%H:%M:%S')}, {age_min:.0f}m ago{stale})")
        try:
            with open(lf) as f:
                lines = f.readlines()
            tail = lines[-6:] if len(lines) >= 6 else lines
            for line in tail:
                print(f"     {line.rstrip()}")
        except Exception as e:
            print(f"     (read error: {e})")

    # Also check for any detail.jsonl being actively written (modified < 5 min ago)
    active = []
    for f in glob.glob(os.path.join(LOGS_DIR, "eval_*_detail.jsonl")):
        mtime = os.path.getmtime(f)
        age = (datetime.now().timestamp() - mtime) / 60
        if age < 5:
            lines = sum(1 for _ in open(f))
            active.append((os.path.basename(f), lines, age))
    if active:
        print(f"\n  📝 Recently-active detail logs (<5 min):")
        for name, n, age in sorted(active, key=lambda x: x[2]):
            print(f"     {name}: {n} lines ({age:.1f}m ago)")


# ── 2. Completed Experiments Summary ─────────────────────────
def parse_detail_logs():
    """Parse all eval_*_detail.jsonl files. Returns list of run dicts."""
    runs = []
    for f in sorted(glob.glob(os.path.join(LOGS_DIR, "eval_*_detail.jsonl"))):
        # Skip old backups
        if "/old_" in f or "/bak" in f:
            continue
        basename = os.path.basename(f)
        try:
            total = correct = 0
            mode = provider = None
            with open(f) as fh:
                for line in fh:
                    d = json.loads(line)
                    total += 1
                    if d.get("is_correct"):
                        correct += 1
                    if mode is None:
                        mode = d.get("mode", "?")
                    if provider is None:
                        provider = d.get("provider", "?")
            if total == 0:
                continue
            acc = correct / total * 100
            runs.append({
                "file": basename,
                "mode": mode or "?",
                "provider": provider or "?",
                "total": total,
                "correct": correct,
                "accuracy": acc,
            })
        except Exception:
            continue
    return runs


def show_summary():
    banner("📊  COMPLETED EXPERIMENTS — Best Per Mode")
    runs = parse_detail_logs()
    if not runs:
        print("  No completed experiments found in logs/")
        return runs

    # Group by mode
    by_mode = defaultdict(list)
    for r in runs:
        by_mode[r["mode"]].append(r)

    # Table header
    hdr = f"  {'Mode':<25} {'Best%':>6} {'Provider':<20} {'N':>5} {'Runs':>5}"
    print(hdr)
    print(f"  {'─'*25} {'─'*6} {'─'*20} {'─'*5} {'─'*5}")

    for mode in sorted(by_mode.keys()):
        mode_runs = by_mode[mode]
        best = max(mode_runs, key=lambda r: r["accuracy"])
        n_runs = len(mode_runs)
        print(f"  {mode:<25} {best['accuracy']:5.1f}% {best['provider']:<20} {best['total']:>5} {n_runs:>5}")

    print(f"\n  Total: {len(runs)} experiment runs across {len(by_mode)} modes")
    return runs


# ── 3. Queue Status ──────────────────────────────────────────
def show_queue_status(all_runs):
    banner("📋  QUEUE STATUS (run_experiment_queue.py)")
    if not QUEUE_EXPERIMENTS:
        print("  Could not import EXPERIMENTS list — skipping.")
        return

    # Build a set of (mode, provider) combos we have full results for
    completed_keys = set()
    full_count = expected_question_count("full", "barexam") if expected_question_count else 1900
    for r in all_runs:
        if r["total"] >= full_count:
            completed_keys.add((r["mode"], r["provider"]))

    # Assign phases by mode (robust regardless of list changes)
    mode_to_phase = {
        "llm_only":         "Phase 1 — LLM baselines",
        "golden_passage":   "Phase 2 — Golden passage",
        "rag_simple":       "Phase 3 — RAG modes",
        "rag_hyde":         "Phase 3 — RAG modes",
        "rag_snap_hyde":    "Phase 3 — RAG modes",
        "rag_devil_hyde":   "Phase 4 — Devil RAG",
        "confidence_gated": "Phase 5 — Confidence-gated",
        "ce_threshold":     "Phase 6 — CE-threshold",
    }

    done = pending = partial_count = 0
    grouped = OrderedDict()
    for mode, provider, questions, tag in QUEUE_EXPERIMENTS:
        phase = mode_to_phase.get(mode, f"Other — {mode}")
        grouped.setdefault(phase, []).append((mode, provider, questions, tag))

    for phase_name, exps in grouped.items():
        print(f"\n  {phase_name}:")
        for mode, provider, questions, tag in exps:
            key = (mode, provider)
            if key in completed_keys:
                status = "✅"
                done += 1
            else:
                # Check for partial results (< full but > 0)
                partials = [r for r in all_runs if r["mode"] == mode and r["provider"] == provider]
                if partials:
                    best = max(partials, key=lambda r: r["total"])
                    status = f"🔶 partial ({best['total']}Q)"
                    partial_count += 1
                else:
                    status = "⬜"
                    pending += 1
            print(f"    {status} {mode:<20} {provider:<18} [{tag}]")

    print(f"\n  Summary: {done} done, {partial_count} partial, {pending} pending  (of {len(QUEUE_EXPERIMENTS)} total)")


# ── 4. Embedding Comparison Status ───────────────────────────
def _get_chroma_collections() -> dict:
    """Get collection names from ChromaDB via lightweight sqlite3 read.
    Returns {name: True/False} (True if data exists).
    """
    db = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(db):
        return {}
    collections = {}
    try:
        import sqlite3
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        if "collections" not in tables:
            conn.close()
            return {}

        # Build segment map for data-existence check
        seg_map = {}
        if "segments" in tables:
            try:
                cur.execute("SELECT id, collection FROM segments")
                for sid, coll in cur.fetchall():
                    seg_map[sid] = coll
            except Exception:
                pass

        cur.execute("SELECT id, name FROM collections")
        for cid, name in cur.fetchall():
            has_data = False
            if "embeddings_queue" in tables:
                try:
                    cur.execute("SELECT count(*) FROM embeddings_queue WHERE collection_id=?", (cid,))
                    if cur.fetchone()[0] > 0:
                        has_data = True
                except Exception:
                    pass
            if not has_data and "max_seq_id" in tables:
                seg_ids = [sid for sid, scol in seg_map.items() if scol == cid]
                for sid in seg_ids:
                    try:
                        cur.execute("SELECT seq_id FROM max_seq_id WHERE segment_id=?", (sid,))
                        row = cur.fetchone()
                        if row and row[0] and int(row[0]) > 0:
                            has_data = True
                            break
                    except Exception:
                        pass
            collections[name] = has_data
        conn.close()
    except Exception:
        pass
    return collections


def show_embedding_status():
    banner("🧬  EMBEDDING COMPARISON STATUS")
    if not EMBEDDING_CANDIDATES:
        print("  Could not import EMBEDDING_CANDIDATES — skipping.")
        return

    # Try to import helper functions
    try:
        sys.path.insert(0, ROOT)
        from utils.fast_embed import EMBEDDING_MODELS, resolve_collection_name
        has_helpers = True
    except Exception:
        has_helpers = False

    # Get existing ChromaDB collections (lightweight — no chromadb import)
    collections = _get_chroma_collections()

    # Check each candidate
    print(f"  {'Model':<18} {'Collection':<42} {'Embedded':>10} {'Eval':>6}")
    print(f"  {'─'*18} {'─'*42} {'─'*10} {'─'*6}")

    for short_name, desc in EMBEDDING_CANDIDATES:
        # Resolve collection name
        if has_helpers and short_name in EMBEDDING_MODELS:
            model_id = EMBEDDING_MODELS[short_name]
            coll_name = resolve_collection_name("legal_passages", model_id)
        else:
            # Fallback: guess the collection name pattern
            coll_name = f"legal_passages__{short_name.replace('-', '_')}"

        has_data = collections.get(coll_name, False)
        embed_status = "✓ exists" if has_data else "✗"

        # Check for eval logs matching this embedding
        eval_pattern = os.path.join(LOGS_DIR, f"eval_*embed*{short_name}*_detail.jsonl")
        eval_logs = glob.glob(eval_pattern)
        if not eval_logs:
            eval_pattern2 = os.path.join(LOGS_DIR, f"eval_rag_simple_*embed-{short_name}*_detail.jsonl")
            eval_logs = glob.glob(eval_pattern2)
        eval_status = f"✓ {len(eval_logs)}" if eval_logs else "✗"

        print(f"  {short_name:<18} {coll_name:<42} {embed_status:>10} {eval_status:>6}")

    # Show all collections in ChromaDB
    if collections:
        print(f"\n  All ChromaDB collections:")
        for name, has_data in sorted(collections.items()):
            flag = "✓" if has_data else "○ empty"
            print(f"    {name:<45} {flag}")


# ── 5. Disk Usage ────────────────────────────────────────────
def show_disk():
    banner("💾  DISK USAGE")
    logs_size = du(LOGS_DIR)
    chroma_size = du(CHROMA_DIR)
    n_logs = len(glob.glob(os.path.join(LOGS_DIR, "eval_*_detail.jsonl")))
    n_queue_logs = len(glob.glob(os.path.join(LOGS_DIR, "queue_phase*.log")))

    print(f"  logs/      {logs_size:>8}   ({n_logs} detail logs, {n_queue_logs} queue logs)")
    print(f"  chroma_db/ {chroma_size:>8}")

    # Break down chroma_db if it exists
    if os.path.isdir(CHROMA_DIR):
        try:
            subdirs = sorted(glob.glob(os.path.join(CHROMA_DIR, "*")))
            if subdirs:
                for sd in subdirs[:10]:
                    name = os.path.basename(sd)
                    size = du(sd)
                    print(f"    └─ {name:<35} {size}")
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────
def main():
    print(f"{'═' * W}")
    print(f"  📡  EXPERIMENT MONITOR — {short_ts()}")
    print(f"  📂  {ROOT}")
    print(f"{'═' * W}")

    show_running()
    all_runs = show_summary()
    show_queue_status(all_runs or [])
    show_embedding_status()
    show_disk()

    print(f"\n{'═' * W}")
    print(f"  Done. Rerun: uv run python eval/monitor.py")
    print(f"{'═' * W}\n")


if __name__ == "__main__":
    main()
