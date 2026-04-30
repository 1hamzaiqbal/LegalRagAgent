#!/bin/bash
# End-to-end: embed HousingQA statutes into Chroma, then run the clean
# retrieval-depth diagnostic slice:
#   rag_simple k=1, rag_simple k=5, rag_simple k=10, rag_snap_hyde_2call k=5
# × OR-Gemma 4 26B-A4B × HousingQA N=200.
#
# Why bundled: prior split job (55489) failed at preflight because
# housing_statutes collection was registered but unembedded. Embedding 1.8M
# statutes is ~3-6h on a single cluster GPU; eval is ~30 min via OR. Total
# wallclock fits in one longer SBATCH window and avoids a partial "embedded but
# not evaluated" state.
#
# Bottleneck-taxonomy 4th dataset: tests whether snap_hyde_2call lifts on
# Yes/No statutory QA over a huge sparse corpus. Predict: retrieval-
# bottlenecked (like MuSiQue) — model already knows Yes/No, hard part is
# finding the controlling statute. The k=1/5/10 slice tests retrieval-depth
# sensitivity; the 2-call arm tests answer-conditioned pseudo-doc retrieval.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 10:00:00
#SBATCH -J embed-eval-housing-k
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
DOWNLOAD_HOUSING=${DOWNLOAD_HOUSING:-1}
RETRIEVAL_K=${RETRIEVAL_K:-5}

if [[ -n "${RUN_SPECS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_SPECS_ARR=(${RUN_SPECS})
elif [[ -n "${MODES:-}" ]]; then
  # Backward-compatible mode list: all modes use RETRIEVAL_K.
  RUN_SPECS_ARR=()
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
  for mode in "${MODES_ARR[@]}"; do
    RUN_SPECS_ARR+=("${mode}:${RETRIEVAL_K}:k${RETRIEVAL_K}")
  done
else
  RUN_SPECS_ARR=(
    rag_simple:1:top1
    rag_simple:5:top5
    rag_simple:10:top10
    rag_snap_hyde_2call:5:2call
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

if [[ "$DOWNLOAD_HOUSING" != "0" && ! -f "datasets/housing_qa/questions.csv" ]]; then
  echo "[$(date -Is)] HousingQA questions missing; downloading dataset"
  python utils/download_housingqa.py
fi

if [[ ! -f "datasets/housing_qa/statutes.csv" || ! -f "datasets/housing_qa/questions.csv" ]]; then
  echo "[$(date -Is)] ERROR: HousingQA files missing under $REPO/datasets/housing_qa"
  echo "[$(date -Is)] Expected statutes.csv and questions.csv"
  exit 4
fi

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
housing_count = None
for col in client.list_collections():
    name = getattr(col, 'name', str(col))
    try:
        collection = client.get_collection(name)
        n = collection.count()
    except Exception as e:
        n = f'ERROR: {e}'
    print(f'  {name}: {n}')
    if name == 'housing_statutes':
        housing_count = n
if not isinstance(housing_count, int) or housing_count <= 0:
    raise SystemExit('housing_statutes collection missing or empty after embedding')
"

# Now go offline for eval to mirror other cluster jobs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo
echo "[$(date -Is)] === STAGE 2: HousingQA k-sweep + 2-call eval N=$N_QUESTIONS ==="
echo "[$(date -Is)] LLM=$PROVIDER (OpenRouter), dataset=housing, seed=$SEED"
echo "[$(date -Is)] Run specs: ${RUN_SPECS_ARR[*]}"

FAILURES=0
FAILED_SPECS=()
for spec in "${RUN_SPECS_ARR[@]}"; do
  IFS=: read -r mode retrieval_k spec_tag <<< "$spec"
  if [[ -z "$mode" || -z "$retrieval_k" ]]; then
    echo "[$(date -Is)] Invalid RUN_SPEC '$spec'; expected mode:k[:tag]"
    exit 5
  fi
  spec_tag=${spec_tag:-k${retrieval_k}}
  run_tag="${TAG_SUFFIX}-${spec_tag}-k${retrieval_k}"
  echo
  echo "[$(date -Is)] --- MODE $mode retrieval_k=$retrieval_k tag=$run_tag ---"
  set +e
  LLM_PROVIDER="$PROVIDER" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider "$PROVIDER" \
    --questions "$N_QUESTIONS" \
    --seed "$SEED" \
    --dataset housing \
    --retrieval-k "$retrieval_k" \
    --tag "$run_tag"
  status=$?
  set -e

  latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_"${PROVIDER}"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest_log" ]]; then
    python scripts/analyze_detail_flags.py "$latest_log" || true
  fi

  if [[ "$status" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    FAILED_SPECS+=("$spec")
    echo "[$(date -Is)] SPEC FAILED: $spec (exit $status)"
  else
    echo "[$(date -Is)] SPEC OK: $spec"
  fi
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[$(date -Is)] Completed with failures: ${FAILED_SPECS[*]}"
  exit 1
fi

echo
echo "[$(date -Is)] All stages complete. Pull logs and run paired McNemar:"
echo "  rsync -av wustl:$REPO/logs/eval_*_${PROVIDER}_*_detail.jsonl logs/"
echo "  python scripts/build_speculative_metrics_report.py --log housing_top1=<detail> --log housing_top5=<detail> --log housing_top10=<detail> --log housing_2call=<detail> --out docs/housing_speculative_metrics_<date>.md"
