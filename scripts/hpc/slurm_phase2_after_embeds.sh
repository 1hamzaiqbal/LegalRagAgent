#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J phase2-all
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Phase 2: Run all embedding comparison evals + clean snap_hyde rerun
# Submit with: sbatch --dependency=afterok:<embed_build_job_id> slurm_phase2_after_embeds.sh
#
# Runs sequentially (single vLLM instance) to avoid ChromaDB conflicts:
# 1. Gemma snap_hyde with baseline (gte-large) — clean rerun
# 2. For each embedder: rag_simple + snap_hyde with N=200
#
# Estimated: snap_hyde full ~10h + 4 embedders × 2.5h = ~20h total

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

# Verify baseline chroma_db exists
source "$EVAL_VENV/bin/activate"
python -c "
import chromadb, os
db = os.environ.get('CHROMA_DB_DIR', './chroma_db')
c = chromadb.PersistentClient(db)
cols = {col.name: col.count() for col in c.list_collections()}
print(f'Baseline chroma_db: {cols}')
assert cols.get('legal_passages', 0) > 600000, f'Baseline not ready: {cols}'
"
if [ $? -ne 0 ]; then
  echo "[$(date '+%F %T')] ERROR: Baseline chroma_db not ready. Aborting."
  exit 1
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Start vLLM once, reuse for all evals
echo "[$(date '+%F %T')] Starting vLLM for $MODEL"
$GEMMA_VENV/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 > "$LOG_DIR/vllm_gemma4_e4b_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM to become ready..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM died"; tail -30 "$LOG_DIR/vllm_gemma4_e4b_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "ERROR: vLLM timeout"; exit 1; }
echo "[$(date '+%F %T')] vLLM ready"

run_eval() {
  local mode="$1" chroma_dir="$2" embed_model="$3" questions="$4" tag="$5"
  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] $tag: mode=$mode, N=$questions"
  echo "  chroma=$chroma_dir embed=$embed_model"
  echo "======================================================================"

  CHROMA_DB_DIR="$chroma_dir" \
  EVAL_COLLECTION_OVERRIDE="" \
  EVAL_EMBEDDING_MODEL="$embed_model" \
  LLM_PROVIDER=cluster-vllm \
  LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  LLM_API_KEY=DUMMY_KEY \
  LLM_MODEL="$MODEL" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider cluster-vllm \
    --questions "$questions" \
    --dataset barexam \
    --tag "$tag" || echo "[$(date '+%F %T')] WARN: $tag failed, continuing"
}

# --- 1. Clean snap_hyde rerun with baseline ---
echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] PART 1: Clean Gemma snap_hyde (baseline, full)"
echo "######################################################################"
run_eval rag_snap_hyde "./chroma_db" "" full "hpc-gemma4-e4b-snap-hyde-clean"

# --- 2. Embedding comparison (N=200 each) ---
echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] PART 2: Embedding comparison (N=200)"
echo "######################################################################"

# Resolve model info for each embedder
EMBEDDERS="gte-large legal-bert stella-400m bge-m3"

for short in $EMBEDDERS; do
  # Resolve chroma dir and model ID
  INFO=$(python -c "
from utils.fast_embed import EMBEDDING_MODELS, resolve_collection_name
model_id = EMBEDDING_MODELS.get('$short', '$short')
# Baseline uses ./chroma_db, others use ./chroma_db_<name>
if '$short' == 'gte-large':
    chroma = './chroma_db'
else:
    suffix = '$short'.replace('-','_')
    chroma = f'./chroma_db_{suffix}'
print(f'{chroma}|{model_id}')
")
  CHROMA=$(echo "$INFO" | cut -d'|' -f1)
  EMBED_ID=$(echo "$INFO" | cut -d'|' -f2)

  # Verify collection exists
  if ! python -c "
import chromadb
c = chromadb.PersistentClient('$CHROMA')
assert any(col.count() > 100000 for col in c.list_collections()), 'no collection'
" 2>/dev/null; then
    echo "[$(date '+%F %T')] SKIP $short: chroma_db not found at $CHROMA"
    continue
  fi

  run_eval rag_simple "$CHROMA" "$EMBED_ID" 200 "embed-${short}-rag-simple"
  run_eval rag_snap_hyde "$CHROMA" "$EMBED_ID" 200 "embed-${short}-snap-hyde"
done

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] ALL EVALS COMPLETE"
echo "######################################################################"
