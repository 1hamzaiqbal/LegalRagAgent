#!/usr/bin/env python3
"""Embedding model A/B comparison pipeline.

Phase 1: Re-embed the barexam corpus with each candidate model.
Phase 2: Run rag_simple eval with each embedding, using a fixed LLM (groq-llama70b).
Phase 3: Summary comparison table.

Each embedding creates a separate ChromaDB collection (e.g., legal_passages__stella_en_400m_v5).
The original legal_passages collection is never touched.

Usage:
    uv run python eval/run_embedding_comparison.py --embed     # Phase 1: embed all models
    uv run python eval/run_embedding_comparison.py --eval      # Phase 2: eval all embeddings
    uv run python eval/run_embedding_comparison.py --both      # Both phases
    uv run python eval/run_embedding_comparison.py --dry-run   # Preview
    uv run python eval/run_embedding_comparison.py --status    # Check what's done
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ──────────────────────────────────────────────────────────────
# Embedding models to compare (short_name from fast_embed.py)
# ──────────────────────────────────────────────────────────────
EMBEDDING_CANDIDATES = [
    # (short_name, description)
    # --- 1.5B heavyweights (different architectures, both fit 8GB) ---
    ("stella-1.5b",    "Stella 1.5B v5 — top MTEB retrieval under 2B, 1024d, ~59.8, ~3GB"),
    ("gte-qwen2-1.5b", "GTE-Qwen2 1.5B — LLM-based instruct embedder, 1536d, ~57.8, ~3GB"),
    # --- ~500M mid-range (each tests a different approach) ---
    ("bge-m3",         "BGE-M3 — hybrid dense+sparse+ColBERT, 1024d, ~56.5, ~1.1GB"),
    ("jina-v3",        "Jina v3 — task-specific LoRA for retrieval, 1024d, ~58.5, ~1.1GB"),
    ("arctic-l-v2",    "Arctic Embed L v2 — retrieval-optimized from scratch, 1024d, ~57.5, ~1.1GB"),
    # --- Small + control ---
    ("stella-400m",    "Stella 400M v5 — does size matter? 1024d, ~58.5, ~0.8GB"),
    ("legal-bert",     "Legal-BERT — domain-specific control, 768d, 512 tok, ~0.4GB"),
]

# Fixed LLM for embedding comparison (isolate embedding effect)
EVAL_LLM_PROVIDER = "groq-llama70b"
# Number of eval questions (full test set for proper comparison)
EVAL_QUESTIONS = "full"


def run_embed(short_name: str, dry_run: bool = False) -> bool:
    """Embed the barexam corpus with a specific model."""
    cmd = ["uv", "run", "python", "utils/fast_embed.py", "barexam", "--model", short_name]
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] EMBED: {short_name}")
    print(f"  Cmd: {' '.join(cmd)}")

    if dry_run:
        print("  [DRY RUN]")
        return True

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=14400)  # 4h max
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"  ✓ Embedded in {elapsed/60:.1f}min")
            return True
        else:
            print(f"  ✗ Failed (exit {result.returncode}) after {elapsed/60:.1f}min")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out after 4h")
        return False


def run_eval_with_embedding(short_name: str, collection_suffix: str, model_id: str,
                             dry_run: bool = False) -> bool:
    """Run rag_simple eval using a specific embedding collection."""
    # The eval harness needs to know which collection + embedding model to use
    # We pass this via environment variables
    env = os.environ.copy()
    env["EVAL_COLLECTION_OVERRIDE"] = collection_suffix
    env["EVAL_EMBEDDING_MODEL"] = model_id
    env["LLM_PROVIDER"] = EVAL_LLM_PROVIDER

    cmd = [
        "uv", "run", "python", "eval/eval_harness.py",
        "--mode", "rag_simple",
        "--provider", EVAL_LLM_PROVIDER,
        "--questions", EVAL_QUESTIONS,
        "--tag", f"embed-{short_name}",
    ]
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] EVAL: rag_simple + {short_name}")
    print(f"  Collection: {collection_suffix}")
    print(f"  Embedding: {model_id}")

    if dry_run:
        print("  [DRY RUN]")
        return True

    start = time.time()
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200, env=env)
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"  ✓ Eval done in {elapsed/60:.1f}min")
            return True
        else:
            print(f"  ✗ Failed (exit {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timed out")
        return False


def check_status():
    """Show which embeddings and evals exist."""
    from utils.fast_embed import EMBEDDING_MODELS, resolve_collection_name

    print("\n=== Embedding Collections ===")
    try:
        import chromadb
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = {c.name: c.count() for c in client.list_collections()}
    except Exception:
        collections = {}

    for short_name, desc in EMBEDDING_CANDIDATES:
        model_id = EMBEDDING_MODELS.get(short_name, short_name)
        coll_name = resolve_collection_name("legal_passages", model_id)
        count = collections.get(coll_name, 0)
        status = f"✓ {count:,} docs" if count > 0 else "✗ not embedded"
        print(f"  {short_name:<15} → {coll_name:<40} {status}")

    print("\n=== Eval Results (embed-* tags) ===")
    for f in sorted(glob.glob("logs/eval_rag_simple_*embed*_detail.jsonl")):
        basename = os.path.basename(f)
        lines = sum(1 for _ in open(f))
        correct = 0
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                if d.get('is_correct'):
                    correct += 1
        acc = correct / lines * 100 if lines > 0 else 0
        print(f"  {basename}: {lines} Q, {acc:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Embedding model A/B comparison")
    parser.add_argument("--embed", action="store_true", help="Phase 1: embed corpora")
    parser.add_argument("--eval", action="store_true", help="Phase 2: run evals")
    parser.add_argument("--both", action="store_true", help="Run both phases")
    parser.add_argument("--status", action="store_true", help="Check what's done")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if args.status:
        check_status()
        return

    if not (args.embed or args.eval or args.both):
        parser.print_help()
        return

    from utils.fast_embed import EMBEDDING_MODELS, resolve_collection_name

    do_embed = args.embed or args.both
    do_eval = args.eval or args.both

    if do_embed:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Embedding {len(EMBEDDING_CANDIDATES)} models")
        print(f"{'='*60}")

        for short_name, desc in EMBEDDING_CANDIDATES:
            print(f"\n  → {short_name}: {desc}")
            run_embed(short_name, dry_run=args.dry_run)

    if do_eval:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Evaluating {len(EMBEDDING_CANDIDATES)} embeddings")
        print(f"LLM: {EVAL_LLM_PROVIDER} | Mode: rag_simple | N: {EVAL_QUESTIONS}")
        print(f"{'='*60}")

        for short_name, desc in EMBEDDING_CANDIDATES:
            model_id = EMBEDDING_MODELS.get(short_name, short_name)
            coll_name = resolve_collection_name("legal_passages", model_id)
            run_eval_with_embedding(short_name, coll_name, model_id, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("DONE. Run --status to see results.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
