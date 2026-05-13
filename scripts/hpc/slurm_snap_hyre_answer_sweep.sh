#!/bin/bash
# Run the canonical Snap-HyRE answer ladder for one dataset/model cell.
#
# Keep this deliberately narrow: submit one dataset/model at a time after the
# top-k cache diagnostics choose RETRIEVAL_K.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -J snap-hyre-answer
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-snap-hyre-comprehensive}
DATA_REPO=${DATA_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
GEMMA_VENV=${GEMMA_VENV:-/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4}
EVAL_VENV=${EVAL_VENV:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}
HYRE_CACHE_ROOT=${HYRE_CACHE_ROOT:-$REPO/caches/hyre/full}
RETRIEVAL_CACHE_ROOT=${RETRIEVAL_CACHE_ROOT:-$REPO/caches/retrieval/full}
BACKEND=${BACKEND:-api}
PROVIDER=${PROVIDER:-groq-llama70b}
MODEL=${MODEL:-}
MODEL_LABEL=${MODEL_LABEL:-$PROVIDER}
PORT=${PORT:-8014}
DATASET=${DATASET:-barexam}
QUESTIONS=${QUESTIONS:-full}
SEED=${SEED:-42}
RETRIEVAL_K=${RETRIEVAL_K:-5}
USE_CACHES=${USE_CACHES:-1}
STOP_ON_FAILURE=${STOP_ON_FAILURE:-1}
LLM_MAX_COMPLETION_TOKENS=${LLM_MAX_COMPLETION_TOKENS:-768}

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(llm_only rag_simple rag_hyde snap_hyre golden_passage golden_plus_neighbors rag_rewrite)
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
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
export LLM_MAX_COMPLETION_TOKENS

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date -Is)] repo=$REPO commit=$(git rev-parse --short HEAD)"
echo "[$(date -Is)] backend=$BACKEND provider=$PROVIDER model=${MODEL:-none} model_label=$MODEL_LABEL"
echo "[$(date -Is)] dataset=$DATASET questions=$QUESTIONS retrieval_k=$RETRIEVAL_K modes=${MODES_ARR[*]} use_caches=$USE_CACHES"
git status --short --branch

if [[ "$BACKEND" == "vllm" ]]; then
  if [[ -z "$MODEL" ]]; then
    echo "[$(date -Is)] ERROR: MODEL is required for BACKEND=vllm"
    exit 2
  fi
  "$GEMMA_VENV/bin/vllm" serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 > "$LOG_DIR/vllm_snap_hyre_answer_${SLURM_JOB_ID}.log" 2>&1 &
  VLLM_PID=$!

  READY=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[$(date -Is)] ERROR: vLLM died during startup"
      tail -100 "$LOG_DIR/vllm_snap_hyre_answer_${SLURM_JOB_ID}.log" || true
      exit 1
    fi
    sleep 5
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[$(date -Is)] ERROR: vLLM did not become ready"
    tail -100 "$LOG_DIR/vllm_snap_hyre_answer_${SLURM_JOB_ID}.log" || true
    exit 1
  fi
  export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
  export LLM_API_KEY=DUMMY_KEY
  export LLM_MODEL="$MODEL"
fi

source "$EVAL_VENV/bin/activate"

python -m py_compile \
  eval/eval_config.py \
  eval/eval_harness.py \
  scripts/analyze_detail_flags.py

add_cache_args_for_mode() {
  local mode="$1"
  local hyre_cache=""
  local retrieval_cache=""
  case "$mode" in
    rag_simple)
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_raw_question_k10.jsonl"
      ;;
    rag_hyde)
      hyre_cache="$HYRE_CACHE_ROOT/${DATASET}_${MODEL_LABEL}_rag_hyde.jsonl"
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_${MODEL_LABEL}_rag_hyde_k10.jsonl"
      ;;
    snap_hyre)
      hyre_cache="$HYRE_CACHE_ROOT/${DATASET}_${MODEL_LABEL}_snap_hyre.jsonl"
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_${MODEL_LABEL}_snap_hyre_k10.jsonl"
      ;;
    golden_plus_neighbors)
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_golden_neighbors_k10.jsonl"
      ;;
  esac

  if [[ "$USE_CACHES" != "1" ]]; then
    return 0
  fi
  if [[ -n "$hyre_cache" ]]; then
    [[ -f "$hyre_cache" ]] || { echo "missing hyre cache $hyre_cache" >&2; return 2; }
    extra_args+=(--hyre-cache-path "$hyre_cache")
  fi
  if [[ -n "$retrieval_cache" ]]; then
    [[ -f "$retrieval_cache" ]] || { echo "missing retrieval cache $retrieval_cache" >&2; return 2; }
    extra_args+=(--retrieval-cache-path "$retrieval_cache")
  fi
}

for mode in "${MODES_ARR[@]}"; do
  tag="snap-hyre-answer-${MODEL_LABEL}-${DATASET}-k${RETRIEVAL_K}-${mode}-job${SLURM_JOB_ID}"
  echo
  echo "[$(date -Is)] run dataset=$DATASET mode=$mode tag=$tag"
  extra_args=()
  if ! add_cache_args_for_mode "$mode"; then
    echo "[$(date -Is)] FAILED dataset=$DATASET mode=$mode while resolving caches"
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit 2
    fi
    continue
  fi

  set +e
  LLM_PROVIDER="$PROVIDER" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --questions "$QUESTIONS" \
    --seed "$SEED" \
    --retrieval-k "$RETRIEVAL_K" \
    --tag "$tag" \
    "${extra_args[@]}"
  status=$?
  set -e

  latest_log=$(
    find "$REPO/logs" -maxdepth 1 \
      -name "eval_${mode}_${PROVIDER}_*_${DATASET}_*${tag}*_detail.jsonl" \
      -print 2>/dev/null | sort | tail -n 1
  )
  if [[ -z "$latest_log" ]]; then
    echo "[$(date -Is)] ERROR: no detail log found for mode=$mode tag=$tag"
    status=1
  else
    python scripts/analyze_detail_flags.py "$latest_log" || status=1
    python - "$latest_log" <<'PY' || status=1
import json
import sys

path = sys.argv[1]
bad = []
errors = []
with open(path) as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        pred = row.get("predicted_answer")
        if pred is None or str(pred).strip() == "":
            bad.append(str(row.get("label") or row.get("idx") or line_no))
        if row.get("error"):
            errors.append(str(row.get("label") or row.get("idx") or line_no))
if bad:
    raise SystemExit("missing predicted_answer rows: " + ",".join(bad[:10]))
if errors:
    raise SystemExit("error rows: " + ",".join(errors[:10]))
PY
  fi

  if [[ "$status" -ne 0 ]]; then
    echo "[$(date -Is)] FAILED dataset=$DATASET mode=$mode exit=$status"
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit "$status"
    fi
  else
    echo "[$(date -Is)] OK dataset=$DATASET mode=$mode"
  fi
done

echo "[$(date -Is)] answer sweep complete."
