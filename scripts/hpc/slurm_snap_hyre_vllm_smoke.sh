#!/bin/bash
# Tiny vLLM-backed Gemma smoke for the comprehensive Snap-HyRE branch.
#
# Starts one Gemma model with vLLM, then runs one BarExam question through
# llm_only and snap_hyre. Submit once per model before larger Gemma runs.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -J snap-hyre-vllm-smoke
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
MODEL=${MODEL:-google/gemma-4-E4B-it}
PORT=${PORT:-8011}
N_QUESTIONS=${N_QUESTIONS:-1}
SEED=${SEED:-42}
RETRIEVAL_K=${RETRIEVAL_K:-3}
LLM_MAX_COMPLETION_TOKENS=${LLM_MAX_COMPLETION_TOKENS:-768}

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
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
export DISABLE_CROSS_ENCODER=1
export LLM_MAX_COMPLETION_TOKENS

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date -Is)] vLLM smoke repo=$REPO commit=$(git rev-parse --short HEAD)"
echo "[$(date -Is)] model=$MODEL port=$PORT modes=${MODES_ARR[*]}"

"$GEMMA_VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization 0.8 \
  --max-model-len 8192 > "$LOG_DIR/vllm_snap_hyre_smoke_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

READY=0
for _ in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date -Is)] ERROR: vLLM died during startup"
    tail -100 "$LOG_DIR/vllm_snap_hyre_smoke_${SLURM_JOB_ID}.log" || true
    exit 1
  fi
  sleep 5
done

if [[ "$READY" -ne 1 ]]; then
  echo "[$(date -Is)] ERROR: vLLM did not become ready"
  tail -100 "$LOG_DIR/vllm_snap_hyre_smoke_${SLURM_JOB_ID}.log" || true
  exit 1
fi

source "$EVAL_VENV/bin/activate"

FAILURES=0
for mode in "${MODES_ARR[@]}"; do
  tag="snap-hyre-vllm-smoke-${MODEL//\//_}-${mode}-n${N_QUESTIONS}-k${RETRIEVAL_K}"
  echo
  echo "[$(date -Is)] === model=$MODEL mode=$mode tag=$tag ==="
  set +e
  LLM_PROVIDER=cluster-vllm \
  LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  LLM_API_KEY=DUMMY_KEY \
  LLM_MODEL="$MODEL" \
  EVAL_TRACE_CALLS=1 \
  EVAL_TRACE_EVENTS=1 \
  EVAL_TRACE_MAX_CHARS=1200 \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider cluster-vllm \
    --dataset barexam \
    --questions "$N_QUESTIONS" \
    --seed "$SEED" \
    --retrieval-k "$RETRIEVAL_K" \
    --tag "$tag"
  status=$?
  set -e

  latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_cluster-vllm_*_detail.jsonl 2>/dev/null | head -n 1 || true)
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
    echo "[$(date -Is)] ERROR: no detail log found for $mode"
    status=1
  fi

  if [[ "$status" -ne 0 ]]; then
    FAILURES=$((FAILURES + 1))
    echo "[$(date -Is)] FAILED model=$MODEL mode=$mode exit=$status"
  else
    echo "[$(date -Is)] OK model=$MODEL mode=$mode"
  fi
done

if [[ "$FAILURES" -gt 0 ]]; then
  echo "[$(date -Is)] vLLM smoke completed with $FAILURES failure(s)"
  exit 1
fi

echo "[$(date -Is)] vLLM smoke complete."
