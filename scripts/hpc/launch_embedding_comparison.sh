#!/bin/bash
# Launch the full embedding comparison experiment
#
# Phase 1: Build embedding collections (parallel, ~2-6h each)
# Phase 2: Run evals once collections are built (sequential per embedder)
#
# Usage:
#   ./scripts/hpc/launch_embedding_comparison.sh build     # Phase 1 only
#   ./scripts/hpc/launch_embedding_comparison.sh eval 200   # Phase 2 with N=200
#   ./scripts/hpc/launch_embedding_comparison.sh eval full  # Phase 2 with full set
#   ./scripts/hpc/launch_embedding_comparison.sh status     # Check collection status
#
# Embedding models tested:
#   legal-bert  — Domain-specific (THE key question)
#   stella-1.5b — Top MTEB retrieval (does general quality help?)
#   bge-m3      — Hybrid dense+sparse (different retrieval strategy)
#   stella-400m — Efficiency control (if close to 1.5b, save compute)
#
# Baseline: gte-large-en-v1.5 (already built as 'legal_passages')

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
cd "$REPO"

MODELS="legal-bert stella-1.5b bge-m3 stella-400m"

case "${1:-help}" in
  build)
    echo "=== Phase 1: Building embedding collections ==="
    for model in $MODELS; do
      echo "Submitting embed build: $model"
      sbatch --job-name="embed-${model}" scripts/hpc/slurm_build_embed.sh "$model"
    done
    echo ""
    echo "Monitor with: squeue -u hiqbal"
    echo "Check status: source .venv/bin/activate && python utils/fast_embed.py status"
    ;;

  eval)
    N="${2:-200}"
    echo "=== Phase 2: Running evals with N=$N ==="
    echo "NOTE: Collections must be built first (Phase 1)"
    echo ""

    # Also run baseline (gte-large) for fair comparison at same N
    echo "Submitting eval: gte-large (baseline)"
    sbatch --job-name="emeval-baseline" scripts/hpc/slurm_gemma4_embed_eval.sh gte-large "$N"

    for model in $MODELS; do
      echo "Submitting eval: $model"
      sbatch --job-name="emeval-${model}" scripts/hpc/slurm_gemma4_embed_eval.sh "$model" "$N"
    done
    echo ""
    echo "Each job runs rag_simple + snap_hyde in sequence."
    echo "N=200: ~2.5h per embedder. N=full: ~13h per embedder."
    echo "Monitor with: squeue -u hiqbal"
    ;;

  status)
    echo "=== Collection status ==="
    source "$REPO/.venv/bin/activate"
    python utils/fast_embed.py status
    ;;

  *)
    echo "Usage: $0 {build|eval [N]|status}"
    echo ""
    echo "  build       Submit parallel SLURM jobs to build all embedding collections"
    echo "  eval [N]    Submit eval jobs (default N=200, use 'full' for all 1195)"
    echo "  status      Check which collections exist and their sizes"
    ;;
esac
