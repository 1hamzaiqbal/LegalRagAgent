#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J gemma4-rerun
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

# Parameterized Gemma 4 E4B full-rerun matrix.
# Pass MODES env var to control what runs:
#
#   # Priority 1: core retrieval ablation
#   sbatch --export=ALL,MODES="rag_simple rag_hyde snap_only_in_final",TAG_SUFFIX=p1-reruns \
#     scripts/hpc/slurm_gemma4_rerun.sh
#
#   # Priority 1b: snap+hyde alone (longer, split from 1a)
#   sbatch --export=ALL,MODES="rag_snap_hyde",TAG_SUFFIX=p1b-reruns \
#     scripts/hpc/slurm_gemma4_rerun.sh
#
#   # Priority 2: subagent variants
#   sbatch --export=ALL,MODES="subagent_rag subagent_hyde subagent_hybrid snap_hyde_report",TAG_SUFFIX=p2-reruns \
#     scripts/hpc/slurm_gemma4_rerun.sh
#
#   # Smaller / bigger Gemma: override MODEL
#   sbatch --export=ALL,MODEL=google/gemma-4-26B-A4B-it,MODES="rag_hyde",TAG_SUFFIX=26b-hyde \
#     --gres=gpu:a100-sxm4:1 scripts/hpc/slurm_gemma4_rerun.sh
#
# Defaults to N_QUESTIONS=full (1195 BarExam). Set N_QUESTIONS=200 for faster smoke.

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-clean}
DATA_REPO=${DATA_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
GEMMA_VENV=${GEMMA_VENV:-/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4}
EVAL_VENV=${EVAL_VENV:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}
MODEL=${MODEL:-google/gemma-4-E4B-it}
PORT=${PORT:-8014}
N_QUESTIONS=${N_QUESTIONS:-full}
SEED=${SEED:-42}
TAG_SUFFIX=${TAG_SUFFIX:-rerun-leak-fix}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
EVAL_TRACE_CALLS=${EVAL_TRACE_CALLS:-1}
EVAL_TRACE_EVENTS=${EVAL_TRACE_EVENTS:-$EVAL_TRACE_CALLS}
EVAL_TRACE_MAX_CHARS=${EVAL_TRACE_MAX_CHARS:-0}

if [[ -z "${MODES:-}" ]]; then
  echo "ERROR: MODES env var is required. Example:" >&2
  echo "  sbatch --export=ALL,MODES=\"rag_hyde rag_snap_hyde\" scripts/hpc/slurm_gemma4_rerun.sh" >&2
  exit 1
fi
# shellcheck disable=SC2206
MODES_ARR=(${MODES})

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$REPO/logs"
ln -sfn "$DATA_REPO/datasets" "$REPO/datasets"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TORCH_HOME="$TORCH_HOME"
export CHROMA_DB_DIR="$CHROMA_DB_DIR"
# Force HF offline — models are cached and cluster network is flaky.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

VLLM_LOG="$LOG_DIR/vllm_gemma4_rerun_${SLURM_JOB_ID}.log"

echo "[$(date -Is)] Starting vLLM for $MODEL (mem_util=$GPU_MEM_UTIL, max_len=$MAX_MODEL_LEN)"
"$GEMMA_VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "[$(date -Is)] Waiting for vLLM (PID=$VLLM_PID)"
READY=0
for _ in $(seq 1 360); do
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
  echo "[$(date -Is)] ERROR: vLLM did not become ready after 30 minutes"
  tail -80 "$VLLM_LOG" || true
  exit 1
fi

echo "[$(date -Is)] vLLM ready; running modes: ${MODES_ARR[*]} at N=$N_QUESTIONS"
source "$EVAL_VENV/bin/activate"

FAILURES=0
FAILED_MODES=()
for mode in "${MODES_ARR[@]}"; do
  echo
  echo "[$(date -Is)] === MODE $mode (n=$N_QUESTIONS seed=$SEED tag=$TAG_SUFFIX) ==="
  set +e
  LLM_PROVIDER=cluster-vllm \
  LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  LLM_API_KEY=DUMMY_KEY \
  LLM_MODEL="$MODEL" \
  EVAL_TRACE_CALLS="$EVAL_TRACE_CALLS" \
  EVAL_TRACE_EVENTS="$EVAL_TRACE_EVENTS" \
  EVAL_TRACE_MAX_CHARS="$EVAL_TRACE_MAX_CHARS" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider cluster-vllm \
    --questions "$N_QUESTIONS" \
    --seed "$SEED" \
    --dataset barexam \
    --tag "$TAG_SUFFIX"
  status=$?
  set -e

  latest_log=$(ls -t "$REPO"/logs/eval_"$mode"_cluster-vllm_*_detail.jsonl 2>/dev/null | head -n 1 || true)
  if [[ -n "$latest_log" ]]; then
    python scripts/analyze_detail_flags.py "$latest_log"
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

echo "[$(date -Is)] All modes completed successfully"
