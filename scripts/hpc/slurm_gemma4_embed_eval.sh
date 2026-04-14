#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Gemma 4 E4B eval with a specific embedding model
# Runs rag_simple + snap_hyde in sequence (single vLLM instance)
#
# Usage: sbatch --job-name=emeval-<name> slurm_gemma4_embed_eval.sh <embed_short> <n_questions>
# Example: sbatch --job-name=emeval-legal-bert slurm_gemma4_embed_eval.sh legal-bert 200
#
# Timing estimates (Gemma 4 E4B, N=200):
#   rag_simple:  ~15s/q × 200 = ~50min
#   snap_hyde:   ~24s/q × 200 = ~80min
#   Total: ~2.5h + 10min startup per embedding model
#
# For full N=1195: rag_simple ~5h + snap_hyde ~8h = ~13h total

set -euo pipefail

EMBED_SHORT="${1:?Usage: sbatch slurm_gemma4_embed_eval.sh <embed_short_name> [n_questions]}"
N_QUESTIONS="${2:-200}"

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
GEMMA_VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
EVAL_VENV="$REPO/.venv"
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=google/gemma-4-E4B-it
PORT=8010

mkdir -p "$LOG_DIR"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="/engrfs/tmp/jacobsn/hiqbal_legalrag/cache"
export TORCH_HOME="/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch"
mkdir -p "$XDG_CACHE_HOME"

# Resolve collection name and model ID from the short name
# Activate eval venv first to get access to fast_embed
source "$EVAL_VENV/bin/activate"
EMBED_INFO=$(python -c "
from utils.fast_embed import EMBEDDING_MODELS, resolve_collection_name
model_id = EMBEDDING_MODELS.get('$EMBED_SHORT', '$EMBED_SHORT')
coll = resolve_collection_name('legal_passages', model_id)
print(f'{coll}|{model_id}')
")
COLLECTION=$(echo "$EMBED_INFO" | cut -d'|' -f1)
EMBED_MODEL_ID=$(echo "$EMBED_INFO" | cut -d'|' -f2)

echo "[$(date '+%F %T')] Embedding eval: $EMBED_SHORT"
echo "[$(date '+%F %T')] Collection: $COLLECTION"
echo "[$(date '+%F %T')] Embed model: $EMBED_MODEL_ID"
echo "[$(date '+%F %T')] LLM: $MODEL"
echo "[$(date '+%F %T')] Questions: $N_QUESTIONS"

# Verify the collection exists
python -c "
import chromadb
c = chromadb.PersistentClient('chroma_db')
cols = {col.name: col.count() for col in c.list_collections()}
target = '$COLLECTION'
if target in cols:
    print(f'Collection {target}: {cols[target]} documents')
else:
    print(f'ERROR: Collection {target} not found!')
    print(f'Available: {list(cols.keys())}')
    exit(1)
"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Start vLLM with gemma venv
echo "[$(date '+%F %T')] Starting vLLM for $MODEL"
$GEMMA_VENV/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
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
    echo "[$(date '+%F %T')] ERROR: vLLM process died"
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

echo "[$(date '+%F %T')] vLLM ready"

# --- Run rag_simple ---
echo ""
echo "======================================================================"
echo "[$(date '+%F %T')] Running rag_simple with $EMBED_SHORT embeddings"
echo "======================================================================"

EVAL_COLLECTION_OVERRIDE="$COLLECTION" \
EVAL_EMBEDDING_MODEL="$EMBED_MODEL_ID" \
LLM_PROVIDER=cluster-vllm \
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
LLM_API_KEY=DUMMY_KEY \
LLM_MODEL="$MODEL" \
python eval/eval_harness.py \
  --mode rag_simple \
  --provider cluster-vllm \
  --questions "$N_QUESTIONS" \
  --dataset barexam \
  --tag "embed-${EMBED_SHORT}-rag-simple"

echo "[$(date '+%F %T')] rag_simple complete"

# --- Run snap_hyde ---
echo ""
echo "======================================================================"
echo "[$(date '+%F %T')] Running rag_snap_hyde with $EMBED_SHORT embeddings"
echo "======================================================================"

EVAL_COLLECTION_OVERRIDE="$COLLECTION" \
EVAL_EMBEDDING_MODEL="$EMBED_MODEL_ID" \
LLM_PROVIDER=cluster-vllm \
LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
LLM_API_KEY=DUMMY_KEY \
LLM_MODEL="$MODEL" \
python eval/eval_harness.py \
  --mode rag_snap_hyde \
  --provider cluster-vllm \
  --questions "$N_QUESTIONS" \
  --dataset barexam \
  --tag "embed-${EMBED_SHORT}-snap-hyde"

echo "[$(date '+%F %T')] All evals complete for $EMBED_SHORT"
