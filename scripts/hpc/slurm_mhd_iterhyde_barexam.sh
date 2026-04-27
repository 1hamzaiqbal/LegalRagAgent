#!/bin/bash
# Cross-domain validation: does multi_hyde_diverse / iter_hyde lift on BarExam (legal MC)?
#
# Phase 13.5 finding (MuSiQue multi-hop): mhd lifts cross-FAMILY at N=100
#   (Llama 70b +12pp sig, Gemma 3 27B +8pp trending).
# Phase 14 finding (MuSiQue Gemma 3 27B): iter_hyde HURTS smaller dense model
#   (-20pp at N=30, synthesis overload).
#
# This script tests the Gemma 4 26B-A4B (cluster main model) BarExam corpus:
# - Does mhd's diversity lift on single-hop legal MC, or is it a multi-hop-only
#   intervention?
# - Does iter_hyde's multi-round structure HELP a stronger model on BarExam,
#   distinguishing "multi-round needs capacity floor" vs "multi-round adds
#   no value cross-domain"?
#
# Cluster baselines on Gemma 4 26B-A4B BarExam N=1195 (committed audit_log):
#   - rag_simple: 70.8% (cluster-vllm 20260421_0857)
#   - rag_snap_hyde: 81.17% (cluster-vllm 20260425_2226) ← cluster headline winner
#   - llm_only: 79.7% (cluster-vllm 20260426_0027)
# Cross-domain success = mhd or iter_hyde matches/beats rag_snap_hyde at N=200.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -J mhd-iterhyde-barexam
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean}
DATA_REPO=${DATA_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
GEMMA_VENV=${GEMMA_VENV:-/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4}
EVAL_VENV=${EVAL_VENV:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}
# Default to Gemma 4 26B-A4B (cluster headline model). Override with MODEL env var.
MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
PORT=${PORT:-8021}
N_QUESTIONS=${N_QUESTIONS:-200}
SEED=${SEED:-42}
TAG_SUFFIX=${TAG_SUFFIX:-cross-domain-mhd-iterhyde-n${N_QUESTIONS}}
EVAL_TRACE_CALLS=${EVAL_TRACE_CALLS:-1}
EVAL_TRACE_EVENTS=${EVAL_TRACE_EVENTS:-$EVAL_TRACE_CALLS}
EVAL_TRACE_MAX_CHARS=${EVAL_TRACE_MAX_CHARS:-0}

# Modes default: cross-domain validation set + baselines for paired comparison.
# rag_simple is the canonical baseline; rag_snap_hyde is the BarExam headline
# winner we need to beat for the lift claim.
if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(rag_simple rag_snap_hyde multi_hyde_diverse iter_hyde)
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
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date -Is)] Starting vLLM for $MODEL (port $PORT)"
"$GEMMA_VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 > "$LOG_DIR/vllm_mhd_iterhyde_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date -Is)] Waiting for vLLM (PID=$VLLM_PID) — up to 20 min"
READY=0
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: vLLM died during startup"
    tail -80 "$LOG_DIR/vllm_mhd_iterhyde_${SLURM_JOB_ID}.log" || true
    exit 1
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  echo "[$(date -Is)] ERROR: vLLM did not become ready after 20 minutes"
  tail -80 "$LOG_DIR/vllm_mhd_iterhyde_${SLURM_JOB_ID}.log" || true
  exit 1
fi

echo "[$(date -Is)] vLLM ready; running modes: ${MODES_ARR[*]} (N=$N_QUESTIONS, seed=$SEED)"
source "$EVAL_VENV/bin/activate"

FAILURES=0
FAILED_MODES=()
for mode in "${MODES_ARR[@]}"; do
  echo
  echo "[$(date -Is)] === MODE $mode (n=$N_QUESTIONS seed=$SEED) ==="
  set +e
  LLM_PROVIDER=cluster-vllm \
  LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  LLM_API_KEY=DUMMY_KEY \
  LLM_MODEL="$MODEL" \
  EVAL_TRACE_CALLS="$EVAL_TRACE_CALLS" \
  EVAL_TRACE_EVENTS="$EVAL_TRACE_EVENTS" \
  EVAL_TRACE_MAX_CHARS="$EVAL_TRACE_MAX_CHARS" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider cluster-vllm \
    --questions "$N_QUESTIONS" \
    --seed "$SEED" \
    --dataset barexam \
    --tag "$TAG_SUFFIX"
  status=$?
  set -e

  latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_cluster-vllm_*_detail.jsonl 2>/dev/null | head -n 1 || true)
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

echo "[$(date -Is)] All cross-domain modes completed successfully"
echo "[$(date -Is)] Pull logs back to laptop with:"
echo "  rsync -av engr-jacobsn:$REPO/logs/eval_*_cluster-vllm_*${TAG_SUFFIX}*detail.jsonl logs/"
