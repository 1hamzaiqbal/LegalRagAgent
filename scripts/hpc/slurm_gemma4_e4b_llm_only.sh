#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J gemma4-llm
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Gemma 4 E4B llm_only baseline on barexam (full 1195 questions)
# Model: google/gemma-4-E4B-it (8B total / 4.5B effective, ~16GB bf16)
# Uses Gemma 4 venv with vLLM nightly (0.19.1rc1)
# Estimated: ~20-25h eval + 15min startup (dense 8B, likely similar to Qwen3-8B)

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=google/gemma-4-E4B-it
PORT=8002

mkdir -p "$LOG_DIR"
source "$VENV/bin/activate"
cd "$REPO"

# Gemma models were downloaded via snapshot_download(cache_dir=HF_CACHE)
# so they live at HF_CACHE/models--google--* (not HF_CACHE/hub/)
# HUGGINGFACE_HUB_CACHE points directly to the dir containing models--* dirs
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
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

echo "[$(date '+%F %T')] Starting vLLM for $MODEL (nightly $(python -c 'import vllm; print(vllm.__version__)'))"
echo "[$(date '+%F %T')] HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE"

vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.9 \
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

echo "[$(date '+%F %T')] vLLM ready; starting llm_only eval"
echo "[$(date '+%F %T')] Model served: $(curl -s http://127.0.0.1:${PORT}/v1/models | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || echo 'unknown')"

LLM_PROVIDER=cluster-vllm \
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
LLM_API_KEY=DUMMY_KEY \
LLM_MODEL="$MODEL" \
python eval/eval_harness.py \
  --mode llm_only \
  --provider cluster-vllm \
  --questions full \
  --dataset barexam \
  --tag hpc-gemma4-e4b-llm-only

echo "[$(date '+%F %T')] llm_only eval complete"
