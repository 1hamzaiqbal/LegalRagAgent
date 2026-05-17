#!/bin/bash
# Build full HyDE / Snap-HyRE generation caches for one model.
#
# API example:
#   PROVIDER=groq-llama70b MODEL_LABEL=llama70b sbatch scripts/hpc/slurm_snap_hyre_generation_cache.sh
#
# vLLM example:
#   BACKEND=vllm PROVIDER=cluster-vllm MODEL=google/gemma-4-E4B-it MODEL_LABEL=gemma4-e4b PORT=8013 \
#     sbatch scripts/hpc/slurm_snap_hyre_generation_cache.sh

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -J snap-hyre-gen-cache
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
CACHE_DIR=${CACHE_DIR:-$REPO/caches/hyre/full}
BACKEND=${BACKEND:-api}
PROVIDER=${PROVIDER:-groq-llama70b}
MODEL=${MODEL:-}
MODEL_LABEL=${MODEL_LABEL:-$PROVIDER}
PORT=${PORT:-8013}
QUESTIONS=${QUESTIONS:-full}
SEED=${SEED:-42}
CACHE_SCOPE=${CACHE_SCOPE:-}
PARSE_FAIL_MAX=${PARSE_FAIL_MAX:-0}
LLM_MAX_COMPLETION_TOKENS=${LLM_MAX_COMPLETION_TOKENS:-2048}
EVAL_MIN_COMPLETION_TOKENS=${EVAL_MIN_COMPLETION_TOKENS:-2048}
EVAL_GENERATION_FORMAT_RETRY=${EVAL_GENERATION_FORMAT_RETRY:-1}
NO_SILENT_FALLBACK=${NO_SILENT_FALLBACK:-1}

if [[ -z "$CACHE_SCOPE" ]]; then
  CACHE_SCOPE="q${QUESTIONS}_seed${SEED}"
fi
if ! [[ "$LLM_MAX_COMPLETION_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "LLM_MAX_COMPLETION_TOKENS must be a positive integer, got $LLM_MAX_COMPLETION_TOKENS" >&2
  exit 2
fi
if ! [[ "$EVAL_MIN_COMPLETION_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "EVAL_MIN_COMPLETION_TOKENS must be a positive integer, got $EVAL_MIN_COMPLETION_TOKENS" >&2
  exit 2
fi
if (( LLM_MAX_COMPLETION_TOKENS < EVAL_MIN_COMPLETION_TOKENS )); then
  echo "LLM_MAX_COMPLETION_TOKENS=$LLM_MAX_COMPLETION_TOKENS is below EVAL_MIN_COMPLETION_TOKENS=$EVAL_MIN_COMPLETION_TOKENS; refusing truncation-prone generation cache run" >&2
  exit 2
fi
case "${NO_SILENT_FALLBACK,,}" in
  1|true|yes|on) ;;
  *) echo "NO_SILENT_FALLBACK must be enabled for generation cache runs, got $NO_SILENT_FALLBACK" >&2; exit 2 ;;
esac

if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASETS_ARR=(${DATASETS})
else
  DATASETS_ARR=(barexam housing casehold legalbench_scalr)
fi

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(rag_hyde snap_hyre)
fi

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$CACHE_DIR" "$REPO/logs"
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
export EVAL_GENERATION_FORMAT_RETRY
export NO_SILENT_FALLBACK

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date -Is)] repo=$REPO commit=$(git rev-parse --short HEAD)"
echo "[$(date -Is)] backend=$BACKEND provider=$PROVIDER model=${MODEL:-none} model_label=$MODEL_LABEL"
echo "[$(date -Is)] datasets=${DATASETS_ARR[*]} modes=${MODES_ARR[*]} questions=$QUESTIONS cache_scope=$CACHE_SCOPE cache_dir=$CACHE_DIR"
echo "[$(date -Is)] no_silent_fallback=$NO_SILENT_FALLBACK"
echo "[$(date -Is)] llm_max_completion_tokens=$LLM_MAX_COMPLETION_TOKENS eval_min_completion_tokens=$EVAL_MIN_COMPLETION_TOKENS"
echo "[$(date -Is)] eval_generation_format_retry=$EVAL_GENERATION_FORMAT_RETRY"
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
    --max-model-len 8192 > "$LOG_DIR/vllm_snap_hyre_generation_cache_${SLURM_JOB_ID}.log" 2>&1 &
  VLLM_PID=$!

  READY=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[$(date -Is)] ERROR: vLLM died during startup"
      tail -100 "$LOG_DIR/vllm_snap_hyre_generation_cache_${SLURM_JOB_ID}.log" || true
      exit 1
    fi
    sleep 5
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[$(date -Is)] ERROR: vLLM did not become ready"
    tail -100 "$LOG_DIR/vllm_snap_hyre_generation_cache_${SLURM_JOB_ID}.log" || true
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
  scripts/build_generation_cache.py \
  scripts/build_retrieval_cache.py

for dataset in "${DATASETS_ARR[@]}"; do
  for mode in "${MODES_ARR[@]}"; do
    out="$CACHE_DIR/${dataset}_${CACHE_SCOPE}_${MODEL_LABEL}_${mode}.jsonl"
    tag="snap-hyre-gen-cache-${MODEL_LABEL}-${dataset}-${mode}"
    echo
    echo "[$(date -Is)] build generation cache dataset=$dataset mode=$mode out=$out"
    set +e
    NO_SILENT_FALLBACK="$NO_SILENT_FALLBACK" \
    python scripts/build_generation_cache.py \
      --mode "$mode" \
      --provider "$PROVIDER" \
      --dataset "$dataset" \
      --questions "$QUESTIONS" \
      --seed "$SEED" \
      --tag "$tag" \
      --out "$out" \
      --resume
    status=$?
    set -e

    python - "$out" "$mode" "$PARSE_FAIL_MAX" <<'PY' || status=1
import json
import sys

path, mode, parse_fail_max = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
with open(path) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
errors = [r for r in rows if r.get("error")]
missing = [r for r in rows if not r.get("hyde_passage")]
fallbacks = [r for r in rows if r.get("hyde_used_fallback") is True or any(k.endswith("_used_fallback") and v for k, v in r.items())]
parse_fail = [r for r in rows if mode == "snap_hyre" and r.get("snap_hyre_parse_ok") is False]
missing_snap = [r for r in rows if mode == "snap_hyre" and not r.get("snap_letter")]
print(f"[postcheck] path={path} rows={len(rows)} errors={len(errors)} missing_hyde={len(missing)} fallbacks={len(fallbacks)} parse_fail={len(parse_fail)} missing_snap_letter={len(missing_snap)}")
if errors:
    raise SystemExit("generation errors: " + ",".join(str(r.get("label")) for r in errors[:10]))
if missing:
    raise SystemExit("missing hyde_passage: " + ",".join(str(r.get("label")) for r in missing[:10]))
if fallbacks:
    raise SystemExit("generation fallback rows: " + ",".join(str(r.get("label")) for r in fallbacks[:10]))
if missing_snap:
    raise SystemExit("missing snap_letter: " + ",".join(str(r.get("label")) for r in missing_snap[:10]))
if len(parse_fail) > parse_fail_max:
    raise SystemExit(f"parse_failures={len(parse_fail)} > {parse_fail_max}")
PY

    if [[ "$status" -ne 0 ]]; then
      echo "[$(date -Is)] FAILED generation cache dataset=$dataset mode=$mode exit=$status"
      exit "$status"
    fi
  done
done

echo "[$(date -Is)] generation cache job complete."
