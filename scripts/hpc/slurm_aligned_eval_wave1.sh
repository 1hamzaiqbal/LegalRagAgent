#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 14:00:00
#SBATCH -J aligned-w1
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Run snap_hyde_aligned for wave 1 embedders (collections already exist).
# This can run immediately — no dependency on wave 2 builds.
#
# 4 embedders × 1 mode × ~1.7h = ~7h

set -uo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
GEMMA_VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
EVAL_VENV="$REPO/.venv"
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=google/gemma-4-E4B-it
PORT=8011

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

# Start vLLM
echo "[$(date '+%F %T')] Starting vLLM for $MODEL on port $PORT"
$GEMMA_VENV/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 > "$LOG_DIR/vllm_aligned_w1_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM died"; tail -30 "$LOG_DIR/vllm_aligned_w1_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "ERROR: vLLM timeout"; exit 1; }
echo "[$(date '+%F %T')] vLLM ready"

source "$EVAL_VENV/bin/activate"

run_eval() {
  local mode="$1" chroma_dir="$2" embed_model="$3" tag="$4"
  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] $tag: mode=$mode, N=200"
  echo "  chroma=$chroma_dir embed=$embed_model"
  echo "======================================================================"

  CHROMA_DB_DIR="$chroma_dir" \
  EVAL_EMBEDDING_MODEL="$embed_model" \
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

# Wave 1 embedders — snap_hyde_aligned only (rag_simple + snap_hyde already done)
run_eval snap_hyde_aligned "./chroma_db" "Alibaba-NLP/gte-large-en-v1.5" "aligned-gte-large-snap-hyde-aligned"
run_eval snap_hyde_aligned "./chroma_db_legal_bert" "nlpaueb/legal-bert-base-uncased" "aligned-legal-bert-snap-hyde-aligned"
run_eval snap_hyde_aligned "./chroma_db_stella_400m" "dunzhang/stella_en_400M_v5" "aligned-stella-400m-snap-hyde-aligned"
run_eval snap_hyde_aligned "./chroma_db_bge_m3" "BAAI/bge-m3" "aligned-bge-m3-snap-hyde-aligned"

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] WAVE 1 ALIGNED EVALS COMPLETE"
echo "######################################################################"
