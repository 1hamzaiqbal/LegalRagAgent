#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 20:00:00
#SBATCH -J cross-dataset
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Cross-dataset validation: test key modes on HousingQA and CaseHOLD.
# Tests if snap lift generalizes to domains where model lacks knowledge.
#
# Step 1: Download datasets if missing
# Step 2: Run llm_only, snap_hyde, vectorless_direct, subagent_rag on each
#
# Note: HousingQA uses housing_statutes collection (1.8M docs).
# If not built on cluster, snap_hyde will fail but vectorless will work.

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

# Download datasets if missing
source "$EVAL_VENV/bin/activate"
if [ ! -f "datasets/housing_qa/questions.csv" ]; then
  echo "[$(date '+%F %T')] Downloading HousingQA..."
  python utils/download_housingqa.py || echo "WARN: HousingQA download failed"
fi
if [ ! -f "datasets/casehold/test.csv" ]; then
  echo "[$(date '+%F %T')] Downloading CaseHOLD..."
  python utils/download_new_datasets.py || echo "WARN: CaseHOLD download failed"
fi

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
  --max-model-len 8192 > "$LOG_DIR/vllm_cross_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "[$(date '+%F %T')] Waiting for vLLM..."
READY=0
for i in $(seq 1 240); do
  if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: vLLM died"; tail -30 "$LOG_DIR/vllm_cross_${SLURM_JOB_ID}.log"; exit 1
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "ERROR: vLLM timeout"; exit 1; }
echo "[$(date '+%F %T')] vLLM ready"

run_eval() {
  local mode="$1" dataset="$2" tag="$3"
  echo ""
  echo "======================================================================"
  echo "[$(date '+%F %T')] $tag: mode=$mode, dataset=$dataset, N=200"
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
    --dataset "$dataset" \
    --tag "$tag" || echo "[$(date '+%F %T')] WARN: $tag failed, continuing"
}

# === HOUSING QA ===
if [ -f "datasets/housing_qa/questions.csv" ]; then
  echo "[$(date '+%F %T')] === HOUSING QA ==="
  run_eval llm_only housing "cross-housing-llm-only"
  run_eval vectorless_direct housing "cross-housing-vectorless"
  run_eval vectorless_nosnap housing "cross-housing-vectorless-nosnap"
  # snap_hyde needs housing_statutes chroma — may fail
  run_eval rag_snap_hyde housing "cross-housing-snap-hyde"
else
  echo "[$(date '+%F %T')] SKIP HousingQA — dataset not available"
fi

# === CASEHOLD ===
if [ -f "datasets/casehold/test.csv" ]; then
  echo "[$(date '+%F %T')] === CASEHOLD ==="
  run_eval llm_only casehold "cross-casehold-llm-only"
  run_eval vectorless_direct casehold "cross-casehold-vectorless"
  run_eval vectorless_nosnap casehold "cross-casehold-vectorless-nosnap"
else
  echo "[$(date '+%F %T')] SKIP CaseHOLD — dataset not available"
fi

echo ""
echo "######################################################################"
echo "[$(date '+%F %T')] CROSS-DATASET EVAL COMPLETE"
echo "######################################################################"
