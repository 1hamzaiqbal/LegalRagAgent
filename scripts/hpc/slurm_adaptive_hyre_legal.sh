#!/bin/bash
# Adaptive HyRE legal sweep.
#
# Runs the new one-policy adaptive Snap-HyDE method plus focused controls on a
# single legal dataset. The intended benchmark set is BarExam, HousingQA,
# CaseHOLD, and LegalBench-SCALR. Defaults to serving Gemma 4 26B on cluster
# vLLM to keep iteration off paid APIs. Override DATASET, N_QUESTIONS,
# RETRIEVAL_K, and RUN_SPECS at submit time. Set USE_VLLM=0
# PROVIDER=or-gemma4-26b for API fallback.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -J adaptive-hyre-legal
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
DATA_REPO=${DATA_REPO:-$REPO}
GEMMA_VENV=${GEMMA_VENV:-/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4}
EVAL_VENV=${EVAL_VENV:-$REPO/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}

USE_VLLM=${USE_VLLM:-1}
MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
PORT=${PORT:-8015}
PROVIDER=${PROVIDER:-cluster-vllm}
DATASET=${DATASET:-legalbench_scalr}
N_QUESTIONS=${N_QUESTIONS:-200}
SEED=${SEED:-42}
RETRIEVAL_K=${RETRIEVAL_K:-5}
TAG_PROVIDER=${TAG_PROVIDER:-$PROVIDER}
TAG_SUFFIX=${TAG_SUFFIX:-adaptive-hyre-${TAG_PROVIDER}-${DATASET}-n${N_QUESTIONS}-k${RETRIEVAL_K}}
EVAL_TRACE_CALLS=${EVAL_TRACE_CALLS:-1}
EVAL_TRACE_EVENTS=${EVAL_TRACE_EVENTS:-$EVAL_TRACE_CALLS}
EVAL_TRACE_MAX_CHARS=${EVAL_TRACE_MAX_CHARS:-0}

if [[ -n "${RUN_SPECS:-}" ]]; then
  # shellcheck disable=SC2206
  RUN_SPECS_ARR=(${RUN_SPECS})
else
  case "$DATASET" in
    housing)
      RUN_SPECS_ARR=(rag_state_filter snap_hyre_state adaptive_snap_hyre)
      ;;
    casehold|legalbench_scalr|barexam)
      RUN_SPECS_ARR=(rag_simple rag_snap_hyde_2call snap_hyre_option adaptive_snap_hyre)
      ;;
    *)
      echo "[$(date -Is)] ERROR: unsupported adaptive HyRE legal dataset: $DATASET"
      echo "[$(date -Is)] Use one of: barexam, housing, casehold, legalbench_scalr"
      exit 2
      ;;
  esac
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
export VLLM_NO_USAGE_STATS=1
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

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ "$USE_VLLM" != "0" ]]; then
  VLLM_LOG="$LOG_DIR/vllm_adaptive_hyre_${DATASET}_${SLURM_JOB_ID:-local}.log"
  echo "[$(date -Is)] Starting vLLM for $MODEL on port $PORT"
  "$GEMMA_VENV/bin/vllm" serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  echo "[$(date -Is)] Waiting for vLLM (PID=$VLLM_PID)"
  READY=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[$(date -Is)] ERROR: vLLM died during startup"
      tail -80 "$VLLM_LOG" || true
      exit 1
    fi
    sleep 5
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[$(date -Is)] ERROR: vLLM did not become ready after 20 minutes"
    tail -80 "$VLLM_LOG" || true
    exit 1
  fi
  PROVIDER=cluster-vllm
  echo "[$(date -Is)] vLLM ready"
fi

python - <<PY
from eval.eval_harness import DATASET_COLLECTIONS
import chromadb
dataset = "$DATASET"
collection = DATASET_COLLECTIONS.get(dataset)
if collection and collection != "musique_passages":
    client = chromadb.PersistentClient(path="$CHROMA_DB_DIR")
    count = client.get_collection(collection).count()
    print(f"[preflight] {collection} has {count:,} docs")
    if count <= 0:
        raise SystemExit(f"{collection} collection is empty")
PY

echo "[$(date -Is)] provider=$PROVIDER dataset=$DATASET N=$N_QUESTIONS seed=$SEED k=$RETRIEVAL_K"
echo "[$(date -Is)] modes: ${RUN_SPECS_ARR[*]}"

FAILURES=0
FAILED_MODES=()
for mode in "${RUN_SPECS_ARR[@]}"; do
  run_tag="${TAG_SUFFIX}-${mode}"
  echo
  echo "[$(date -Is)] === MODE $mode tag=$run_tag ==="
  set +e
  if [[ "$USE_VLLM" != "0" ]]; then
    LLM_PROVIDER="$PROVIDER" \
    LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
    LLM_API_KEY=DUMMY_KEY \
    LLM_MODEL="$MODEL" \
    EVAL_TRACE_CALLS="$EVAL_TRACE_CALLS" \
    EVAL_TRACE_EVENTS="$EVAL_TRACE_EVENTS" \
    EVAL_TRACE_MAX_CHARS="$EVAL_TRACE_MAX_CHARS" \
    python eval/eval_harness.py \
      --mode "$mode" \
      --provider "$PROVIDER" \
      --questions "$N_QUESTIONS" \
      --seed "$SEED" \
      --dataset "$DATASET" \
      --retrieval-k "$RETRIEVAL_K" \
      --tag "$run_tag"
  else
    LLM_PROVIDER="$PROVIDER" \
    EVAL_TRACE_CALLS="$EVAL_TRACE_CALLS" \
    EVAL_TRACE_EVENTS="$EVAL_TRACE_EVENTS" \
    EVAL_TRACE_MAX_CHARS="$EVAL_TRACE_MAX_CHARS" \
    python eval/eval_harness.py \
      --mode "$mode" \
      --provider "$PROVIDER" \
      --questions "$N_QUESTIONS" \
      --seed "$SEED" \
      --dataset "$DATASET" \
      --retrieval-k "$RETRIEVAL_K" \
      --tag "$run_tag"
  fi
  status=$?
  set -e

  latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_"${PROVIDER}"_*_detail.jsonl 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest_log" ]]; then
    python scripts/analyze_detail_flags.py "$latest_log" || true
    case "$mode" in
      adaptive_snap_hyre|snap_hyre_option|snap_hyre_state)
        python scripts/audit_adaptive_hyre_logs.py "$latest_log" || true
        ;;
    esac
  fi

  if [[ "$status" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    FAILED_MODES+=("$mode")
    echo "[$(date -Is)] MODE FAILED: $mode (exit $status)"
  fi
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[$(date -Is)] completed with failures: ${FAILED_MODES[*]}"
  exit 1
fi

echo "[$(date -Is)] adaptive HyRE sweep complete"
