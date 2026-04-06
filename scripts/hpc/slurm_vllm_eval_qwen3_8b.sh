#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=Qwen/Qwen3-8B
PORT=8000

mkdir -p "$LOG_DIR" "$HF_CACHE"
source "$VENV/bin/activate"
cd "$REPO"

export HF_HOME="$HF_CACHE"
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

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

for _ in $(seq 1 60); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done

curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null

echo "[$(date '+%F %T')] vLLM ready; starting eval"
LLM_PROVIDER=cluster-vllm \
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
LLM_API_KEY=DUMMY_KEY \
LLM_MODEL="$MODEL" \
uv run python eval/eval_harness.py \
  --mode llm_only \
  --provider cluster-vllm \
  --questions full \
  --dataset barexam \
  --tag hpc-qwen3-8b-full

echo "[$(date '+%F %T')] Eval complete"
