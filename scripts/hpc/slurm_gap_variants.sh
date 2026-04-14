#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J gap-variants
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Gap architecture input variants: same retrieval, different final-answer context.
# Tests what information the model actually needs to make good use of retrieved evidence.
#
# Variants (all use gap_hyde retrieval):
#   gap_hyde        — full: snap + gap labels + evidence (baseline, already done: 61.5%)
#   gap_hyde_ev     — evidence only: just passages + question
#   gap_hyde_nosnap — no snap: gaps + evidence + question
#   gap_hyde_flat   — snap + flat evidence: no gap structure
#
# Estimated: ~3-5h each × 3 variants = ~12-15h

set -uo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
GEMMA_VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
EVAL_VENV="$REPO/.venv"
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=google/gemma-4-E4B-it
PORT=8014

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

echo "[$(date '+%F %T')] Starting vLLM for $MODEL on port $PORT"
$GEMMA_VENV/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 > "$LOG_DIR/vllm_gap_var_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM died"; tail -30 "$LOG_DIR/vllm_gap_var_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "ERROR: vLLM timeout"; exit 1; }
echo "[$(date '+%F %T')] vLLM ready"

source "$EVAL_VENV/bin/activate"

run_eval() {
  local mode="$1" tag="$2"
  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] $tag: mode=$mode, N=200"
  echo "======================================================================"

  CHROMA_DB_DIR="./chroma_db" \
  LLM_PROVIDER=cluster-vllm \
  LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  LLM_API_KEY=DUMMY_KEY \
  LLM_MODEL="$MODEL" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider cluster-vllm \
    --questions 200 \
    --dataset barexam \
    --tag "$tag" || echo "[$(date '+%F %T')] WARN: $tag failed, continuing"
}

# Test each input variant
run_eval gap_hyde_ev     "gap-hyde-ev-gemma4"
run_eval gap_hyde_nosnap "gap-hyde-nosnap-gemma4"
run_eval gap_hyde_flat   "gap-hyde-flat-gemma4"

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] GAP VARIANTS COMPLETE"
echo "######################################################################"
