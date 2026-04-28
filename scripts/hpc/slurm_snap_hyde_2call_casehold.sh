#!/bin/bash
# Bottleneck taxonomy: snap_hyde_2call × Groq Llama 70b × CaseHOLD N=200.
#
# CaseHOLD = 5-way MC over case holdings (closer to BarExam in format than
# MuSiQue, but the corpus is not legal-doctrine-heavy — it's case-holding
# snippets). Goal: third dataset point on the snap-first × bottleneck-type
# axis. Predict CaseHOLD behaves more like BarExam (reasoning-bottlenecked,
# snap_hyde_2call ≈ rag_simple, modest lift) than like MuSiQue (retrieval-
# bottlenecked, snap_hyde_2call ≫ rag_simple).
#
# Same-job paired: rag_simple baseline + rag_snap_hyde_2call within the same
# Groq slot, seed=42. Then paired McNemar.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J snap-hyde-2call-casehold
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
DATA_REPO=${DATA_REPO:-$REPO}
EVAL_VENV=${EVAL_VENV:-$REPO/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}

PROVIDER=${PROVIDER:-groq-llama70b}
N_QUESTIONS=${N_QUESTIONS:-200}
SEED=${SEED:-42}
TAG_SUFFIX=${TAG_SUFFIX:-snap-hyde-2call-pair-${PROVIDER}-casehold-n${N_QUESTIONS}}
EVAL_TRACE_CALLS=${EVAL_TRACE_CALLS:-1}
EVAL_TRACE_EVENTS=${EVAL_TRACE_EVENTS:-$EVAL_TRACE_CALLS}
EVAL_TRACE_MAX_CHARS=${EVAL_TRACE_MAX_CHARS:-0}

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(rag_simple rag_snap_hyde_2call)
fi

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$REPO/logs"
ln -sfn "$DATA_REPO/datasets" "$REPO/datasets"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TORCH_HOME="$TORCH_HOME"
export TRITON_CACHE_DIR="/tmp/hiqbal-triton/${SLURM_JOB_ID:-local}"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
mkdir -p "$TRITON_CACHE_DIR" "$UV_CACHE_DIR"
export CHROMA_DB_DIR="$CHROMA_DB_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
else
  echo "[$(date -Is)] ERROR: $REPO/.env missing — API keys not loaded"
  exit 1
fi

echo "[$(date -Is)] LLM=$PROVIDER, Chroma=$CHROMA_DB_DIR, dataset=casehold"
echo "[$(date -Is)] Modes: ${MODES_ARR[*]} (N=$N_QUESTIONS, seed=$SEED)"

source "$EVAL_VENV/bin/activate"

FAILURES=0
FAILED_MODES=()
for mode in "${MODES_ARR[@]}"; do
  echo
  echo "[$(date -Is)] === MODE $mode (n=$N_QUESTIONS seed=$SEED) ==="
  set +e
  LLM_PROVIDER="$PROVIDER" \
  EVAL_TRACE_CALLS="$EVAL_TRACE_CALLS" \
  EVAL_TRACE_EVENTS="$EVAL_TRACE_EVENTS" \
  EVAL_TRACE_MAX_CHARS="$EVAL_TRACE_MAX_CHARS" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider "$PROVIDER" \
    --questions "$N_QUESTIONS" \
    --seed "$SEED" \
    --dataset casehold \
    --tag "$TAG_SUFFIX"
  status=$?
  set -e

  latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_"${PROVIDER}"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest_log" ]]; then
    python scripts/analyze_detail_flags.py "$latest_log" || true
  fi

  if [[ "$status" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    FAILED_MODES+=("$mode")
    echo "[$(date -Is)] MODE FAILED: $mode (exit $status)"
  else
    echo "[$(date -Is)] MODE OK: $mode"
  fi
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[$(date -Is)] Completed with failures: ${FAILED_MODES[*]}"
  exit 1
fi

echo "[$(date -Is)] Both CaseHOLD paired runs completed"
echo "[$(date -Is)] Pull logs back to laptop, then run paired McNemar:"
echo "  uv run python scripts/compute_mcnemar.py \\"
echo "    logs/eval_rag_simple_${PROVIDER}_*${TAG_SUFFIX}*detail.jsonl \\"
echo "    logs/eval_rag_snap_hyde_2call_${PROVIDER}_*${TAG_SUFFIX}*detail.jsonl"
