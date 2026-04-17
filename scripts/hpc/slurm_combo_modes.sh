#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J combo-modes
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# New combo modes: varying information passed to the final decision-maker.
# Also includes fixed rag_hyde for the snap ablation paper table.
#
# N=200 first to validate, then full runs for winners.
#
# Modes (cheapest first):
#   1. rag_hyde (fixed)          — 2 LLM calls, snap ablation critical path
#   2. snap_hyde_report          — 4 LLM calls (snap + hyde + summarize + answer)
#   3. snap_hyde_report_snap     — 4 LLM calls (same + snap visible in final)
#   4. subagent_rag_snap         — ~5 LLM calls (reports + snap in final)
#   5. subagent_rag_full         — ~5 LLM calls (reports + snap + raw in final)

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
  --max-model-len 8192 > "$LOG_DIR/vllm_combo_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM died"; tail -30 "$LOG_DIR/vllm_combo_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "ERROR: vLLM timeout"; exit 1; }
echo "[$(date '+%F %T')] vLLM ready"

source "$EVAL_VENV/bin/activate"

run_eval() {
  local mode="$1" tag="$2" n="${3:-200}"
  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] $tag: mode=$mode, N=$n"
  echo "======================================================================"

  CHROMA_DB_DIR="./chroma_db" \
  LLM_PROVIDER=cluster-vllm \
  LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  LLM_API_KEY=DUMMY_KEY \
  LLM_MODEL="$MODEL" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider cluster-vllm \
    --questions "$n" \
    --dataset barexam \
    --tag "$tag" || echo "[$(date '+%F %T')] WARN: $tag failed, continuing"
}

# --- N=200 validation runs (cheapest first) ---

# 1. Fixed rag_hyde — CRITICAL for snap ablation table
run_eval rag_hyde "rag-hyde-fixed-n200" 200

# 2. Snap-HyDE + summarization (report only, no snap)
run_eval snap_hyde_report "snap-hyde-report-n200" 200

# 3. Snap-HyDE + summarization + snap visible
run_eval snap_hyde_report_snap "snap-hyde-report-snap-n200" 200

# 4. Subagent RAG + snap visible in final
run_eval subagent_rag_snap "subagent-rag-snap-n200" 200

# 5. Subagent RAG maximum info (reports + snap + raw passages)
run_eval subagent_rag_full "subagent-rag-full-n200" 200

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] COMBO MODES N=200 COMPLETE"
echo "######################################################################"
