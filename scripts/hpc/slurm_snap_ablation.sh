#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 20:00:00
#SBATCH -J snap-ablation
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# SNAP ABLATION — the paper's core experiment.
# Tests the contribution of "snap" (reasoning first) across retrieval methods.
#
# Pairs:
#   rag_hyde (no snap) vs rag_snap_hyde (with snap) → HyDE snap contribution
#   vectorless_nosnap (no snap) vs vectorless_direct (with snap) → vectorless snap contribution
#   rag_simple (no snap) vs snap_rag (with snap) → plain RAG snap contribution [already done]
#
# Estimated: ~40min per mode × 4 = ~3h

set -uo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
GEMMA_VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
EVAL_VENV="$REPO/.venv"
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=google/gemma-4-E4B-it
PORT=8010

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
  --max-model-len 8192 > "$LOG_DIR/vllm_snap_ablation_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM died"; tail -30 "$LOG_DIR/vllm_snap_ablation_${SLURM_JOB_ID}.log"; exit 1
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

# HyDE pair: no-snap vs snap
run_eval rag_hyde "ablation-hyde-nosnap"
# snap_hyde already done (65.5%) — this gives us the no-snap HyDE baseline

# Vectorless pair: no-snap vs snap
run_eval vectorless_nosnap "ablation-vectorless-nosnap"
# vectorless_direct already done (64.5%) — this gives us the no-snap vectorless baseline

# Also run subagent_rag full N=1195 since we have time
echo "[$(date '+%F %T')] Starting subagent_rag full N=1195..."
CHROMA_DB_DIR="./chroma_db" \
LLM_PROVIDER=cluster-vllm \
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
LLM_API_KEY=DUMMY_KEY \
LLM_MODEL="$MODEL" \
python eval/eval_harness.py \
  --mode subagent_rag \
  --provider cluster-vllm \
  --questions full \
  --dataset barexam \
  --tag "subagent-rag-full-n1195" || echo "[$(date '+%F %T')] WARN: subagent_rag full failed"

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] SNAP ABLATION + SUBAGENT FULL COMPLETE"
echo "######################################################################"
