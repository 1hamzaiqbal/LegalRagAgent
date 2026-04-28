#!/bin/bash
# Top-1 vs top-5 retrieval-depth ablation on BarExam, Gemma 4 26B-A4B via OR.
#
# Meeting 2026-04-27 ask #4: "Check lift between passing in top-1 retrieved
# vs top-5 retrieved, see if more passages boost or hurt performance."
#
# Llama 70b MuSiQue N=200 result (paired McNemar): rag_simple top-1 = 13.0%
# vs top-5 = 27.5%, delta -14.5pp, p=4.18e-07. Top-5 helps multi-hop massively.
# This SLURM job tests the same question on legal MC where retrieval is less
# central: does top-1 hurt rag_simple/rag_snap_hyde on BarExam Gemma 4 26B?
#
# Pair via paired McNemar against the existing Tier 3 logs:
#   logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl    (top-5, N=1195)
#   logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl (top-5, N=1195)
#
# Note these baselines are cluster-vllm; this run is OR-Gemma. To control for
# provider drift, the comparison should be: this run's top-5 vs this run's top-1
# (also added below) — pair *within* this OR-Gemma slot, then check direction
# matches the cluster-vllm baseline.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J top1-ablation-barexam
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

PROVIDER=${PROVIDER:-or-gemma4-26b}
N_QUESTIONS=${N_QUESTIONS:-200}
SEED=${SEED:-42}
TAG_SUFFIX=${TAG_SUFFIX:-top1-ablation-${PROVIDER}-n${N_QUESTIONS}}
EVAL_TRACE_CALLS=${EVAL_TRACE_CALLS:-1}
EVAL_TRACE_EVENTS=${EVAL_TRACE_EVENTS:-$EVAL_TRACE_CALLS}
EVAL_TRACE_MAX_CHARS=${EVAL_TRACE_MAX_CHARS:-0}

# Modes to run: each mode runs once at top-1 and once at top-5 (paired).
if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(rag_simple rag_snap_hyde)
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

echo "[$(date -Is)] LLM=$PROVIDER (OpenRouter), Chroma=$CHROMA_DB_DIR"
echo "[$(date -Is)] Modes: ${MODES_ARR[*]} × {top-1, top-5} (N=$N_QUESTIONS, seed=$SEED)"

source "$EVAL_VENV/bin/activate"

FAILURES=0
FAILED_PAIRS=()
for mode in "${MODES_ARR[@]}"; do
  for k in 1 5; do
    echo
    echo "[$(date -Is)] === MODE $mode --retrieval-k $k (n=$N_QUESTIONS seed=$SEED) ==="
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
      --dataset barexam \
      --retrieval-k "$k" \
      --tag "${TAG_SUFFIX}-k${k}"
    status=$?
    set -e

    latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_"${PROVIDER}"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
    if [[ -n "$latest_log" ]]; then
      python scripts/analyze_detail_flags.py "$latest_log" || true
    fi

    if [[ "$status" -ne 0 ]]; then
      FAILURES=$((FAILURES + 1))
      FAILED_PAIRS+=("${mode}-k${k}")
      echo "[$(date -Is)] PAIR FAILED: ${mode}-k${k} (exit $status)"
    else
      echo "[$(date -Is)] PAIR OK: ${mode}-k${k}"
    fi
  done
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[$(date -Is)] Completed with failures: ${FAILED_PAIRS[*]}"
  exit 1
fi

echo "[$(date -Is)] All top-1/top-5 paired runs completed"
echo "[$(date -Is)] Pull logs and run paired McNemar:"
for mode in "${MODES_ARR[@]}"; do
  echo "  uv run python scripts/compute_mcnemar.py \\"
  echo "    logs/eval_${mode}_${PROVIDER}_*${TAG_SUFFIX}-k1*detail.jsonl \\"
  echo "    logs/eval_${mode}_${PROVIDER}_*${TAG_SUFFIX}-k5*detail.jsonl"
done
