#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gres=gpu:a100-sxm4:1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 03:00:00
#SBATCH -J gemma4-31b-smoke
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Smoke test for google/gemma-4-31B-it (59GB on disk, ~64-70GB VRAM at bf16).
# Requires 80GB node — A100-sxm4 or H100. A100-sxm4 is the default target.
# To run on H100 instead: `sbatch --gres=gpu:h100:1 --nodelist=h100-2405 ...`
#
# max_model_len dropped to 4096 to shrink KV cache (~3-4GB instead of ~8GB)
# and leave headroom inside the 80GB budget. gpu_memory_utilization=0.92.
#
# Runs rag_hyde at N=5 to validate load+inference. No accuracy claim.

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
MODEL=${MODEL:-google/gemma-4-31B-it}
PORT=${PORT:-8015}
N_QUESTIONS=${N_QUESTIONS:-5}
SEED=${SEED:-42}
TAG_SUFFIX=${TAG_SUFFIX:-smoke-31b}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}

MODES_ARR=(rag_hyde)

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$REPO/logs"
ln -sfn "$DATA_REPO/datasets" "$REPO/datasets"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TORCH_HOME="$TORCH_HOME"
# Triton + uv caches default to $HOME (30GB quota) — redirect to engrfs.
# Per-job triton cache: NFS can't handle atomic tmp-rename across jobs.
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

VLLM_LOG="$LOG_DIR/vllm_gemma4_31b_${SLURM_JOB_ID}.log"

echo "[$(date -Is)] Starting vLLM for $MODEL (mem_util=$GPU_MEM_UTIL, max_len=$MAX_MODEL_LEN)"
nvidia-smi 2>&1 | head -15 || true
"$GEMMA_VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size 1 > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "[$(date -Is)] Waiting for vLLM (PID=$VLLM_PID) — 31B cold load can take 10-20 min"
READY=0
for _ in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: vLLM died during startup"
    tail -100 "$VLLM_LOG" || true
    exit 1
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  echo "[$(date -Is)] ERROR: vLLM did not become ready after 30 minutes"
  tail -100 "$VLLM_LOG" || true
  exit 1
fi

echo "[$(date -Is)] vLLM ready; running modes: ${MODES_ARR[*]}"
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
    python scripts/analyze_detail_flags.py "$latest_log"
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

echo "[$(date -Is)] All modes completed successfully"
