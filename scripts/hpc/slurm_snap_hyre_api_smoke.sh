#!/bin/bash
# Tiny API-provider smoke for the comprehensive Snap-HyRE branch.
#
# Runs one BarExam question through llm_only and snap_hyre for API-backed
# providers. This validates key loading, answer formatting, retrieval, and
# detail-log health before larger API jobs.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J snap-hyre-api-smoke
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-snap-hyre-comprehensive}
DATA_REPO=${DATA_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
EVAL_VENV=${EVAL_VENV:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}
N_QUESTIONS=${N_QUESTIONS:-1}
SEED=${SEED:-42}
RETRIEVAL_K=${RETRIEVAL_K:-3}
LLM_MAX_COMPLETION_TOKENS=${LLM_MAX_COMPLETION_TOKENS:-768}

if [[ -n "${PROVIDERS:-}" ]]; then
  # shellcheck disable=SC2206
  PROVIDERS_ARR=(${PROVIDERS})
else
  PROVIDERS_ARR=(groq-llama70b or-gemma4-26b)
fi

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(llm_only snap_hyre)
fi

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$REPO/logs"
ln -sfn "$DATA_REPO/datasets" "$REPO/datasets"
ln -sfn "$DATA_REPO/chroma_db" "$REPO/chroma_db"
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
export DISABLE_CROSS_ENCODER=1
export LLM_MAX_COMPLETION_TOKENS

if [[ -f "$DATA_REPO/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$DATA_REPO/.env"
  set +a
else
  echo "[$(date -Is)] ERROR: $DATA_REPO/.env missing - API keys not loaded"
  exit 1
fi

source "$EVAL_VENV/bin/activate"

echo "[$(date -Is)] API smoke repo=$REPO commit=$(git rev-parse --short HEAD)"
echo "[$(date -Is)] providers=${PROVIDERS_ARR[*]} modes=${MODES_ARR[*]}"

FAILURES=0
for provider in "${PROVIDERS_ARR[@]}"; do
  for mode in "${MODES_ARR[@]}"; do
    tag="snap-hyre-api-smoke-${provider}-${mode}-n${N_QUESTIONS}-k${RETRIEVAL_K}"
    echo
    echo "[$(date -Is)] === provider=$provider mode=$mode tag=$tag ==="
    set +e
    LLM_PROVIDER="$provider" \
    EVAL_TRACE_CALLS=1 \
    EVAL_TRACE_EVENTS=1 \
    EVAL_TRACE_MAX_CHARS=1200 \
    python eval/eval_harness.py \
      --mode "$mode" \
      --provider "$provider" \
      --dataset barexam \
      --questions "$N_QUESTIONS" \
      --seed "$SEED" \
      --retrieval-k "$RETRIEVAL_K" \
      --tag "$tag"
    status=$?
    set -e

    latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_"$provider"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
    if [[ -n "$latest_log" ]]; then
      python scripts/analyze_detail_flags.py "$latest_log" || status=1
      python - "$latest_log" <<'PY' || status=1
import json
import sys

path = sys.argv[1]
bad = []
with open(path) as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        pred = row.get("predicted_answer")
        if pred is None or str(pred).strip() == "":
            bad.append(str(row.get("label") or row.get("idx") or line_no))
if bad:
    raise SystemExit("missing predicted_answer rows: " + ",".join(bad[:10]))
PY
    else
      echo "[$(date -Is)] ERROR: no detail log found for $provider/$mode"
      status=1
    fi

    if [[ "$status" -ne 0 ]]; then
      FAILURES=$((FAILURES + 1))
      echo "[$(date -Is)] FAILED provider=$provider mode=$mode exit=$status"
    else
      echo "[$(date -Is)] OK provider=$provider mode=$mode"
    fi
  done
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[$(date -Is)] API smoke completed with $FAILURES failure(s)"
  exit 1
fi

echo "[$(date -Is)] API smoke complete."
