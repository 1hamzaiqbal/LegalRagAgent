#!/bin/bash
# Chunked HousingQA state-filtered retrieval diagnostic.
#
# Evaluates the same deterministic N-question sample as the unchunked harness,
# but splits it into independent sample slices. This avoids losing all detail
# rows when one long serial job hits the wall clock.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J housing-state-chunk
#SBATCH --array=0-7%2
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%A_%a.out

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
MODE=${MODE:-rag_state_filter}
N_QUESTIONS=${N_QUESTIONS:-200}
SEED=${SEED:-42}
CHUNK_SIZE=${CHUNK_SIZE:-50}
KS_CSV=${KS_CSV:-5,10}
TAG_SUFFIX=${TAG_SUFFIX:-housing-state-filter-chunked-${PROVIDER}-n${N_QUESTIONS}}

IFS=, read -r -a K_VALUES <<< "$KS_CSV"
NUM_CHUNKS=$(( (N_QUESTIONS + CHUNK_SIZE - 1) / CHUNK_SIZE ))
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
K_INDEX=$(( TASK_ID / NUM_CHUNKS ))
CHUNK_INDEX=$(( TASK_ID % NUM_CHUNKS ))

if (( K_INDEX >= ${#K_VALUES[@]} )); then
  echo "[$(date -Is)] task $TASK_ID has no k assignment; exiting."
  exit 0
fi

RETRIEVAL_K=${K_VALUES[$K_INDEX]}
SAMPLE_START=$(( CHUNK_INDEX * CHUNK_SIZE ))
SAMPLE_END=$(( SAMPLE_START + CHUNK_SIZE ))
if (( SAMPLE_END > N_QUESTIONS )); then
  SAMPLE_END=$N_QUESTIONS
fi

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$REPO/logs"
ln -sfn "$DATA_REPO/datasets" "$REPO/datasets"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TORCH_HOME="$TORCH_HOME"
export TRITON_CACHE_DIR="/tmp/hiqbal-triton/${SLURM_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
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

RUN_TAG="${TAG_SUFFIX}-${MODE}-k${RETRIEVAL_K}-s${SAMPLE_START}-${SAMPLE_END}"
echo "[$(date -Is)] LLM=$PROVIDER mode=$MODE dataset=housing seed=$SEED k=$RETRIEVAL_K sample=${SAMPLE_START}:${SAMPLE_END}"
echo "[$(date -Is)] Tag: $RUN_TAG"

LLM_PROVIDER="$PROVIDER" python eval/eval_harness.py \
  --mode "$MODE" \
  --provider "$PROVIDER" \
  --questions "$N_QUESTIONS" \
  --seed "$SEED" \
  --dataset housing \
  --retrieval-k "$RETRIEVAL_K" \
  --sample-start "$SAMPLE_START" \
  --sample-end "$SAMPLE_END" \
  --tag "$RUN_TAG"

latest_log=$(ls -t "$REPO"/logs/eval_"$MODE"_"${PROVIDER}"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
if [[ -n "$latest_log" ]]; then
  python scripts/analyze_detail_flags.py "$latest_log" || true
fi

echo "[$(date -Is)] Housing state-filter chunk complete."
