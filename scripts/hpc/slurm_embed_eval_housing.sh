#!/bin/bash
# End-to-end: embed HousingQA statutes into Chroma, then run paired
# rag_simple + rag_snap_hyde_2call × OR-Gemma 4 26B-A4B × HousingQA N=200.
#
# Why bundled: prior split job (55489) failed at preflight because
# housing_statutes collection was registered but unembedded. Embedding 1.8M
# statutes is ~3-6h on a single cluster GPU; eval is ~30 min via OR. Total
# wallclock fits comfortably in 8h SBATCH window.
#
# Bottleneck-taxonomy 4th dataset: tests whether snap_hyde_2call lifts on
# Yes/No statutory QA over a huge sparse corpus. Predict: retrieval-
# bottlenecked (like MuSiQue) — model already knows Yes/No, hard part is
# finding the controlling statute.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 08:00:00
#SBATCH -J embed-eval-housing
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
TAG_SUFFIX=${TAG_SUFFIX:-snap-hyde-2call-pair-${PROVIDER}-housing-n${N_QUESTIONS}}

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(rag_simple rag_snap_hyde_2call)
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
export PYTHONUNBUFFERED=1

if [[ -f "$REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
else
  echo "[$(date -Is)] ERROR: $REPO/.env missing — API keys not loaded"
  exit 1
fi

source "$EVAL_VENV/bin/activate"

echo "[$(date -Is)] === STAGE 1: embed HousingQA statutes into Chroma ==="
echo "[$(date -Is)] Corpus: datasets/housing_qa/statutes.csv (~1.8M statutes)"
echo "[$(date -Is)] Target collection: housing_statutes"
echo "[$(date -Is)] Chroma dir: $CHROMA_DB_DIR"
echo "[$(date -Is)] Estimated build time: 3-6h on cluster GPU"

# Use --resume so a crash mid-build can be picked up by a follow-up run.
python utils/fast_embed.py housing --resume

echo "[$(date -Is)] === Embedding complete; collection status ==="
python -c "
import chromadb
client = chromadb.PersistentClient(path='$CHROMA_DB_DIR')
for col in client.list_collections():
    try:
        n = col.count()
    except Exception as e:
        n = f'ERROR: {e}'
    print(f'  {col.name}: {n}')
"

# Now go offline for eval to mirror other cluster jobs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo
echo "[$(date -Is)] === STAGE 2: paired eval rag_simple + rag_snap_hyde_2call × HousingQA N=$N_QUESTIONS ==="
echo "[$(date -Is)] LLM=$PROVIDER (OpenRouter), dataset=housing, seed=$SEED"

FAILURES=0
FAILED_MODES=()
for mode in "${MODES_ARR[@]}"; do
  echo
  echo "[$(date -Is)] --- MODE $mode ---"
  set +e
  LLM_PROVIDER="$PROVIDER" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider "$PROVIDER" \
    --questions "$N_QUESTIONS" \
    --seed "$SEED" \
    --dataset housing \
    --tag "$TAG_SUFFIX"
  status=$?
  set -e

  latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_"${PROVIDER}"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest_log" ]]; then
    python scripts/analyze_detail_flags.py "$latest_log" || true
  fi

  if [[ "$status" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    FAILED_MODES+=("$mode")
    echo "[$(date -Is)] MODE FAILED: $mode (exit $status)"
  else
    echo "[$(date -Is)] MODE OK: $mode"
  fi
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[$(date -Is)] Completed with failures: ${FAILED_MODES[*]}"
  exit 1
fi

echo
echo "[$(date -Is)] All stages complete. Pull logs and run paired McNemar:"
echo "  rsync -av wustl:$REPO/logs/eval_*_${PROVIDER}_*${TAG_SUFFIX}*detail.jsonl logs/"
echo "  uv run python scripts/compute_mcnemar.py \\"
echo "    logs/eval_rag_simple_${PROVIDER}_*${TAG_SUFFIX}*detail.jsonl \\"
echo "    logs/eval_rag_snap_hyde_2call_${PROVIDER}_*${TAG_SUFFIX}*detail.jsonl"
