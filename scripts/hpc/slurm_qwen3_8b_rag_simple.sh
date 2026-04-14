#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J qwen8b-rag
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Qwen3-8B rag_simple eval on barexam (600 questions)
# Requires ChromaDB embeddings (686K passages, already built by job 40387)
# rag_simple = ChromaDB retrieval + answer with context (1 LLM call + retrieval/q)
# Estimated: ~93s/q → 600 × 93s ≈ 15.5h + 15min startup ≈ 16h
# Full 1195 would take ~31h (over 28h limit) — using 600 for safety
# Reduced gpu-memory-utilization to 0.8 for embedding model headroom

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV="$REPO/.venv"
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=Qwen/Qwen3-8B
PORT=8003

mkdir -p "$LOG_DIR" "$HF_CACHE"
source "$VENV/bin/activate"
cd "$REPO"

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
vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
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

echo "[$(date '+%F %T')] vLLM ready; starting rag_simple eval"
echo "[$(date '+%F %T')] ChromaDB at: $REPO/chroma_db/"
python -c "import chromadb; c=chromadb.PersistentClient('chroma_db'); print(f'ChromaDB collections: {[col.name for col in c.list_collections()]}')"

LLM_PROVIDER=cluster-vllm \
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
LLM_API_KEY=DUMMY_KEY \
LLM_MODEL="$MODEL" \
python eval/eval_harness.py \
  --mode rag_simple \
  --provider cluster-vllm \
  --questions 600 \
  --dataset barexam \
  --tag hpc-qwen3-8b-rag-simple

echo "[$(date '+%F %T')] rag_simple eval complete"
