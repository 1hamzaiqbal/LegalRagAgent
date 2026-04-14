#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J gemma4-rag
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Gemma 4 E4B rag_simple eval on barexam (full 1195 questions)
# 1 LLM call + retrieval/q, ~15s/q at 61 tok/s → ~5h for full set
# Reduced gpu-memory-utilization to 0.8 for embedding model headroom
# Split-venv: gemma venv for vLLM, primary venv for eval harness

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
GEMMA_VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
EVAL_VENV="$REPO/.venv"
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=google/gemma-4-E4B-it
PORT=8005

mkdir -p "$LOG_DIR"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/engrfs/tmp/jacobsn/hiqbal_legalrag/cache"
export TORCH_HOME="/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch"
mkdir -p "$XDG_CACHE_HOME"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date '+%F %T')] Starting vLLM for $MODEL"
$GEMMA_VENV/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 > "$LOG_DIR/vllm_gemma4_e4b_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM to become ready (PID=$VLLM_PID)..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM process died. Check log:"
    tail -30 "$LOG_DIR/vllm_gemma4_e4b_${SLURM_JOB_ID}.log"
    exit 1
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "[$(date '+%F %T')] ERROR: vLLM did not become ready after 20 minutes"
  tail -30 "$LOG_DIR/vllm_gemma4_e4b_${SLURM_JOB_ID}.log"
  exit 1
fi

echo "[$(date '+%F %T')] vLLM ready; starting rag_simple eval"
source "$EVAL_VENV/bin/activate"

LLM_PROVIDER=cluster-vllm \
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
LLM_API_KEY=DUMMY_KEY \
LLM_MODEL="$MODEL" \
python eval/eval_harness.py \
  --mode rag_simple \
  --provider cluster-vllm \
  --questions full \
  --dataset barexam \
  --tag hpc-gemma4-e4b-rag-simple

echo "[$(date '+%F %T')] rag_simple eval complete"
