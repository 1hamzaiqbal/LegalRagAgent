#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2307
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J build-index
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Build structured search indices:
# 1. NLP entity graph (no LLM, ~1-2h CPU)
# 2. Case summaries via Gemma 4 E4B (~6h for 22K+ cases)
#
# The entity graph is pure NLP (spaCy/regex), no GPU needed for that part.
# Case summaries need vLLM for the LLM calls.

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

source "$EVAL_VENV/bin/activate"

# === Step 1: Build NLP entity graph (no LLM, ~2-3h with spaCy) ===
echo "[$(date '+%F %T')] Building NLP entity graph with spaCy..."
python utils/build_entity_graph.py --spacy
echo "[$(date '+%F %T')] Entity graph complete"

# === Step 2: Build case summaries via vLLM ===
cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date '+%F %T')] Starting vLLM for case summaries..."
$GEMMA_VENV/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 > "$LOG_DIR/vllm_index_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM died"; tail -30 "$LOG_DIR/vllm_index_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "ERROR: vLLM timeout"; exit 1; }
echo "[$(date '+%F %T')] vLLM ready"

# Start with MBE + wex (small, ~7K cases, ~30 min)
echo "[$(date '+%F %T')] Summarizing MBE passages..."
python utils/build_case_summaries.py --resume --batch mbe \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL"

echo "[$(date '+%F %T')] Summarizing wex entries..."
python utils/build_case_summaries.py --resume --batch wex \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL"

# Then caselaw (22K cases, ~5-6h)
echo "[$(date '+%F %T')] Summarizing caselaw (22K cases)..."
python utils/build_case_summaries.py --resume --batch caselaw \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "$MODEL"

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] INDEX BUILD COMPLETE"
echo "######################################################################"
