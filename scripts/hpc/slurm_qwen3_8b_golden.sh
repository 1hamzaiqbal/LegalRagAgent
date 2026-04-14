#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J qwen8b-gold
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Qwen3-8B golden_passage eval on barexam (full 1195 questions)
# Estimated: ~23-25h eval + 15min vLLM startup
# golden_passage adds ~200-500 tokens of gold passage context per question,
# so slightly more input tokens but similar generation time.
# Conservative estimate: ~28h total with overhead.

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV="$REPO/.venv"
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=Qwen/Qwen3-8B
PORT=8001

mkdir -p "$LOG_DIR" "$HF_CACHE"
source "$VENV/bin/activate"
cd "$REPO"

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

echo "[$(date '+%F %T')] Starting vLLM for $MODEL"
vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 > "$LOG_DIR/vllm_qwen3_8b_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM to become ready..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM process died. Check log:"
    tail -20 "$LOG_DIR/vllm_qwen3_8b_${SLURM_JOB_ID}.log"
    exit 1
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "[$(date '+%F %T')] ERROR: vLLM did not become ready after 20 minutes"
  tail -20 "$LOG_DIR/vllm_qwen3_8b_${SLURM_JOB_ID}.log"
  exit 1
fi

echo "[$(date '+%F %T')] vLLM ready; starting golden_passage eval"
LLM_PROVIDER=cluster-vllm \
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
LLM_API_KEY=DUMMY_KEY \
LLM_MODEL="$MODEL" \
python eval/eval_harness.py \
  --mode golden_passage \
  --provider cluster-vllm \
  --questions full \
  --dataset barexam \
  --tag hpc-qwen3-8b-golden

echo "[$(date '+%F %T')] golden_passage eval complete"
