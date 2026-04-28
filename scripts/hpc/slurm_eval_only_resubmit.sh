#!/bin/bash
# Eval-only resubmit: complete the modes that hit time limit in
# 55451 (BarExam snap_hyde_2call × Gemma 4 26B), 55452 (BarExam top-1/top-5
# rag_snap_hyde × Gemma 4 26B), and 55524 (HousingQA paired × Gemma 4 26B).
#
# All target collections (legal_passages, housing_statutes) are already
# embedded on the cluster — no embed step needed. Eval-only is fast.
#
# 5 modes total in 1 job:
#   1. rag_snap_hyde_2call × BarExam (paired with 55451's rag_snap_hyde 86.0%)
#   2. rag_snap_hyde --retrieval-k 1 × BarExam
#   3. rag_snap_hyde --retrieval-k 5 × BarExam
#   4. rag_simple × HousingQA
#   5. rag_snap_hyde_2call × HousingQA

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J eval-only-resubmit
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
TAG=${TAG:-resubmit-pair-${PROVIDER}-n${N_QUESTIONS}}

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
  echo "[$(date -Is)] ERROR: $REPO/.env missing"
  exit 1
fi

source "$EVAL_VENV/bin/activate"

# (mode, dataset, retrieval_k, tag_suffix)
RUNS=(
  "rag_snap_hyde_2call|barexam|5|barexam-2call"
  "rag_snap_hyde|barexam|1|barexam-snaphyde-k1"
  "rag_snap_hyde|barexam|5|barexam-snaphyde-k5"
  "rag_simple|housing|5|housing-rsimple"
  "rag_snap_hyde_2call|housing|5|housing-2call"
)

FAILURES=0
FAILED=()
for spec in "${RUNS[@]}"; do
  IFS='|' read -r mode dataset rk tagsuf <<< "$spec"
  echo
  echo "[$(date -Is)] === $mode × $dataset --retrieval-k $rk ==="
  set +e
  LLM_PROVIDER="$PROVIDER" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider "$PROVIDER" \
    --questions "$N_QUESTIONS" \
    --seed "$SEED" \
    --dataset "$dataset" \
    --retrieval-k "$rk" \
    --tag "${TAG}-${tagsuf}"
  status=$?
  set -e

  latest=$(ls -t "$REPO"/logs/eval_"$mode"_"${PROVIDER}"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest" ]]; then
    python scripts/analyze_detail_flags.py "$latest" || true
  fi

  if [[ "$status" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    FAILED+=("${mode}-${dataset}-k${rk}")
    echo "[$(date -Is)] FAILED: ${mode}-${dataset}-k${rk} (exit $status)"
  else
    echo "[$(date -Is)] OK: ${mode}-${dataset}-k${rk}"
  fi
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[$(date -Is)] Failures: ${FAILED[*]}"
  exit 1
fi

echo "[$(date -Is)] All eval-only resubmits complete."
