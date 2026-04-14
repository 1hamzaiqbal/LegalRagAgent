#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 28:00:00
#SBATCH -J aligned-eval
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Aligned embedding comparison: run 3 retrieval modes × N embedders.
#
# Modes tested per embedder:
#   1. rag_simple         — question → embed → retrieve → rerank(question) → answer
#   2. rag_snap_hyde      — snap → HyDE → embed → retrieve → rerank(HyDE) → answer
#   3. snap_hyde_aligned  — snap → HyDE → embed → retrieve → rerank(question) → answer
#
# snap_hyde_aligned isolates the embedding model by keeping cross-encoder input
# identical to rag_simple (both rerank against the raw question).
#
# Submit with: sbatch [--dependency=afterok:<build_job>] slurm_aligned_eval.sh
#
# Estimated: N=200 × 3 modes × 9 embedders = 27 evals
#   rag_simple: ~33min each, snap_hyde/aligned: ~1.7h each
#   Total: ~9 × (0.6 + 1.7 + 1.7) = ~36h → needs 28h limit, may not finish all
#
# Optimization: run only snap_hyde_aligned first (the new aligned mode),
# since we already have rag_simple and snap_hyde from wave 1 for 4 embedders.

set -uo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
GEMMA_VENV=/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4
EVAL_VENV="$REPO/.venv"
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
MODEL=google/gemma-4-E4B-it
PORT=8010
N_QUESTIONS=200

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

# Start vLLM once, reuse for all evals
echo "[$(date '+%F %T')] Starting vLLM for $MODEL"
$GEMMA_VENV/bin/vllm serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 > "$LOG_DIR/vllm_aligned_eval_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM to become ready..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM died"; tail -30 "$LOG_DIR/vllm_aligned_eval_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "ERROR: vLLM timeout"; exit 1; }
echo "[$(date '+%F %T')] vLLM ready"

# Switch to eval venv for harness
source "$EVAL_VENV/bin/activate"

run_eval() {
  local mode="$1" chroma_dir="$2" embed_model="$3" tag="$4"
  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] $tag: mode=$mode, N=$N_QUESTIONS"
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
    --questions "$N_QUESTIONS" \
    --dataset barexam \
    --tag "$tag" || echo "[$(date '+%F %T')] WARN: $tag failed, continuing"
}

# Define all embedders with their chroma dirs and model IDs
# Format: short_name|chroma_dir|model_id
EMBEDDERS=(
  "gte-large|./chroma_db|Alibaba-NLP/gte-large-en-v1.5"
  "legal-bert|./chroma_db_legal_bert|nlpaueb/legal-bert-base-uncased"
  "stella-400m|./chroma_db_stella_400m|dunzhang/stella_en_400M_v5"
  "bge-m3|./chroma_db_bge_m3|BAAI/bge-m3"
  "stella-1.5b|./chroma_db_stella_1.5b|dunzhang/stella_en_1.5B_v5"
  "gte-qwen2-1.5b|./chroma_db_gte_qwen2_1.5b|Alibaba-NLP/gte-Qwen2-1.5B-instruct"
  "jina-v3|./chroma_db_jina_v3|jinaai/jina-embeddings-v3"
  "arctic-l-v2|./chroma_db_arctic_l_v2|Snowflake/snowflake-arctic-embed-l-v2.0"
  "nomic-v2-moe|./chroma_db_nomic_v2_moe|nomic-ai/nomic-embed-text-v2-moe"
)

# --- Phase 1: Run snap_hyde_aligned for all available embedders ---
# This is the NEW mode — priority since we already have rag_simple + snap_hyde for wave 1

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] PHASE 1: snap_hyde_aligned for all available embedders"
echo "######################################################################"

for entry in "${EMBEDDERS[@]}"; do
  IFS='|' read -r short chroma embed_id <<< "$entry"

  # Check if chroma dir exists
  if [ ! -d "$chroma" ]; then
    echo "[$(date '+%F %T')] SKIP $short (aligned): chroma not found at $chroma"
    continue
  fi

  run_eval snap_hyde_aligned "$chroma" "$embed_id" "aligned-${short}-snap-hyde-aligned"
done

# --- Phase 2: Run rag_simple + snap_hyde for NEW embedders (wave 2) only ---
# Wave 1 embedders (gte-large, legal-bert, stella-400m, bge-m3) already have these results

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] PHASE 2: rag_simple + snap_hyde for wave 2 embedders"
echo "######################################################################"

WAVE2_EMBEDDERS=(
  "stella-1.5b|./chroma_db_stella_1.5b|dunzhang/stella_en_1.5B_v5"
  "gte-qwen2-1.5b|./chroma_db_gte_qwen2_1.5b|Alibaba-NLP/gte-Qwen2-1.5B-instruct"
  "jina-v3|./chroma_db_jina_v3|jinaai/jina-embeddings-v3"
  "arctic-l-v2|./chroma_db_arctic_l_v2|Snowflake/snowflake-arctic-embed-l-v2.0"
  "nomic-v2-moe|./chroma_db_nomic_v2_moe|nomic-ai/nomic-embed-text-v2-moe"
)

for entry in "${WAVE2_EMBEDDERS[@]}"; do
  IFS='|' read -r short chroma embed_id <<< "$entry"

  if [ ! -d "$chroma" ]; then
    echo "[$(date '+%F %T')] SKIP $short (wave2): chroma not found at $chroma"
    continue
  fi

  run_eval rag_simple "$chroma" "$embed_id" "embed-${short}-rag-simple"
  run_eval rag_snap_hyde "$chroma" "$embed_id" "embed-${short}-snap-hyde"
done

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] ALL ALIGNED EVALS COMPLETE"
echo "######################################################################"
