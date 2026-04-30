#!/bin/bash
# HousingQA state-filtered retrieval diagnostic.
#
# Runs rag_state_filter at k=5 and k=10 over the already-embedded
# housing_statutes collection. This is the next gate before SpecRAG-lite:
# if state filtering closes the gap cheaply, the intervention is metadata-aware
# retrieval rather than multi-draft verification.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J housing-state-filter
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
DATA_REPO=${DATA_REPO:-$REPO}
EVAL_VENV=${EVAL_VENV:-$REPO/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}

PROVIDER=${PROVIDER:-or-gemma4-26b}
N_QUESTIONS=${N_QUESTIONS:-200}
SEED=${SEED:-42}
TAG_SUFFIX=${TAG_SUFFIX:-housing-state-filter-${PROVIDER}-n${N_QUESTIONS}}

if [[ -n "${RUN_SPECS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_SPECS_ARR=(${RUN_SPECS})
else
  RUN_SPECS_ARR=(
    rag_state_filter:5:state5
    rag_state_filter:10:state10
  )
fi

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$REPO/logs"
ln -sfn "$DATA_REPO/datasets" "$REPO/datasets"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TORCH_HOME="$TORCH_HOME"
export TRITON_CACHE_DIR="/tmp/hiqbal-triton/${SLURM_JOB_ID:-local}"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
mkdir -p "$TRITON_CACHE_DIR" "$UV_CACHE_DIR"
export CHROMA_DB_DIR="$CHROMA_DB_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
else
  echo "[$(date -Is)] ERROR: $REPO/.env missing - API keys not loaded"
  exit 1
fi

source "$EVAL_VENV/bin/activate"

python - <<PY
import chromadb
client = chromadb.PersistentClient(path="$CHROMA_DB_DIR")
collection = client.get_collection("housing_statutes")
count = collection.count()
print(f"[preflight] housing_statutes has {count:,} docs")
if count <= 0:
    raise SystemExit("housing_statutes collection is empty")
PY

echo "[$(date -Is)] LLM=$PROVIDER, dataset=housing, seed=$SEED"
echo "[$(date -Is)] Run specs: ${RUN_SPECS_ARR[*]}"

for spec in "${RUN_SPECS_ARR[@]}"; do
  IFS=: read -r mode retrieval_k spec_tag <<< "$spec"
  run_tag="${TAG_SUFFIX}-${spec_tag}-k${retrieval_k}"
  echo
  echo "[$(date -Is)] === MODE $mode retrieval_k=$retrieval_k tag=$run_tag ==="
  LLM_PROVIDER="$PROVIDER" python eval/eval_harness.py \
    --mode "$mode" \
    --provider "$PROVIDER" \
    --questions "$N_QUESTIONS" \
    --seed "$SEED" \
    --dataset housing \
    --retrieval-k "$retrieval_k" \
    --tag "$run_tag"

  latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_"${PROVIDER}"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest_log" ]]; then
    python scripts/analyze_detail_flags.py "$latest_log" || true
  fi
done

echo "[$(date -Is)] Housing state-filter runs complete."
