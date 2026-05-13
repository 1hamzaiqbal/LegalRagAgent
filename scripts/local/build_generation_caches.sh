#!/usr/bin/env bash
# Build local HyDE/Snap-HyRE generation caches and retrieval caches from them.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

UV="${UV:-uv}"
PROVIDER="${PROVIDER:-or-gemma4-26b}"
MODEL_LABEL="${MODEL_LABEL:-$PROVIDER}"
QUESTIONS="${QUESTIONS:-50}"
SEED="${SEED:-42}"
MAX_K="${MAX_K:-10}"
KS="${KS:-1,3,5,10}"
RESUME="${RESUME:-1}"
TRACE_CALLS="${TRACE_CALLS:-1}"
TRACE_EVENTS="${TRACE_EVENTS:-1}"
HYRE_CACHE_ROOT="${HYRE_CACHE_ROOT:-$ROOT/caches/hyre/full}"
RETRIEVAL_CACHE_ROOT="${RETRIEVAL_CACHE_ROOT:-$ROOT/caches/retrieval/full}"

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

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

case "$PROVIDER" in
  or-*) [[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "missing OPENROUTER_API_KEY for $PROVIDER" >&2; exit 2; } ;;
  groq-*) [[ -n "${GROQ_API_KEY:-}" ]] || { echo "missing GROQ_API_KEY for $PROVIDER" >&2; exit 2; } ;;
esac

mkdir -p "$HYRE_CACHE_ROOT" "$RETRIEVAL_CACHE_ROOT" docs/generated

export CHROMA_DB_DIR="${CHROMA_DB_DIR:-$ROOT/chroma_db}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DISABLE_CROSS_ENCODER="${DISABLE_CROSS_ENCODER:-1}"
export LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-768}"
export PYTHONUNBUFFERED=1

echo "[$(ts)] local generation cache root=$ROOT commit=$(git rev-parse --short HEAD)"
echo "[$(ts)] provider=$PROVIDER model_label=$MODEL_LABEL questions=$QUESTIONS max_k=$MAX_K"
echo "[$(ts)] datasets=${DATASETS_ARR[*]} modes=${MODES_ARR[*]}"

"$UV" run python -m py_compile \
  eval/eval_config.py \
  eval/eval_harness.py \
  scripts/build_generation_cache.py \
  scripts/build_retrieval_cache.py \
  scripts/audit_retrieval_cache.py \
  scripts/compile_retrieval_cache_matrix.py

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  resume_args+=(--resume)
fi

trace_args=()
if [[ "$TRACE_CALLS" == "1" ]]; then
  trace_args+=(--trace-calls)
fi
if [[ "$TRACE_EVENTS" == "1" ]]; then
  trace_args+=(--trace-events)
fi

outputs=()

for dataset in "${DATASETS_ARR[@]}"; do
  for mode in "${MODES_ARR[@]}"; do
    case "$mode" in
      rag_hyde) query_type="hyde_cache" ;;
      snap_hyre) query_type="hyre_cache" ;;
      *) echo "unknown generation mode=$mode; expected rag_hyde or snap_hyre" >&2; exit 2 ;;
    esac

    gen_out="$HYRE_CACHE_ROOT/${dataset}_${MODEL_LABEL}_${mode}.jsonl"
    ret_out="$RETRIEVAL_CACHE_ROOT/${dataset}_${MODEL_LABEL}_${mode}_k${MAX_K}.jsonl"
    tag="local-gen-${MODEL_LABEL}-${dataset}-${mode}-n${QUESTIONS}"

    echo
    echo "[$(ts)] build generation dataset=$dataset mode=$mode out=$gen_out"
    LLM_PROVIDER="$PROVIDER" \
    "$UV" run python scripts/build_generation_cache.py \
      --mode "$mode" \
      --provider "$PROVIDER" \
      --dataset "$dataset" \
      --questions "$QUESTIONS" \
      --seed "$SEED" \
      --tag "$tag" \
      --out "$gen_out" \
      "${resume_args[@]}" \
      "${trace_args[@]}"

    echo "[$(ts)] build retrieval-from-generation dataset=$dataset mode=$mode out=$ret_out"
    "$UV" run python scripts/build_retrieval_cache.py \
      --dataset "$dataset" \
      --questions "$QUESTIONS" \
      --query-type "$query_type" \
      --hyre-cache-path "$gen_out" \
      --max-k "$MAX_K" \
      --out "$ret_out"

    "$UV" run python scripts/audit_retrieval_cache.py \
      --cache "$ret_out" \
      --dataset "$dataset" \
      --query-type "$query_type" \
      --min-k "$MAX_K" \
      --ks "$KS"
    outputs+=("$ret_out")
  done
done

if [[ "${#outputs[@]}" -gt 0 ]]; then
  cache_args=()
  for out in "${outputs[@]}"; do
    cache_args+=(--cache "$out")
  done

  "$UV" run python scripts/compile_retrieval_cache_matrix.py \
    "${cache_args[@]}" \
    --ks "$KS" \
    --min-k "$MAX_K" \
    --out-md "docs/generated/retrieval_cache_matrix_${MODEL_LABEL}_generated.md" \
    --out-csv "docs/generated/retrieval_cache_matrix_${MODEL_LABEL}_generated.csv"

  echo "[$(ts)] wrote docs/generated/retrieval_cache_matrix_${MODEL_LABEL}_generated.md"
  echo "[$(ts)] wrote docs/generated/retrieval_cache_matrix_${MODEL_LABEL}_generated.csv"
fi

echo "[$(ts)] local generation cache pass complete."
