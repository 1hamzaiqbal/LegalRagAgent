#!/usr/bin/env bash
# Build a strict HousingQA state-filter doc cache, then run a cached answer mode.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

UV="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
PROVIDER="${PROVIDER:-groq-llama8b}"
MODEL_LABEL="${MODEL_LABEL:-$PROVIDER}"
MODE="${MODE:-rag_simple}"
QUESTIONS="${QUESTIONS:-full}"
SEED="${SEED:-42}"
RETRIEVAL_K="${RETRIEVAL_K:-5}"
SAMPLE_START="${SAMPLE_START:-0}"
SAMPLE_END="${SAMPLE_END:-}"
CACHE_SCOPE="${CACHE_SCOPE:-q${QUESTIONS}_seed${SEED}_statefilter}"
HYRE_CACHE_ROOT="${HYRE_CACHE_ROOT:-$ROOT/caches/hyre/full}"
RETRIEVAL_CACHE_ROOT="${RETRIEVAL_CACHE_ROOT:-$ROOT/caches/retrieval/full}"
LOCK_PATH="${RETRIEVAL_CACHE_LOCK_PATH:-$ROOT/.locks/retrieval_cache.lock}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

case "$MODE" in
  rag_simple)
    LABEL_PREFIX="${LABEL_PREFIX:-simple}"
    DEFAULT_RETRIEVAL_CACHE="$ROOT/caches/retrieval/full/housing_${CACHE_SCOPE}_raw_question_k10.jsonl"
    DEFAULT_DOC_CACHE="$ROOT/caches/retrieval_doc/full/housing_${CACHE_SCOPE}_raw_question_k10_doc_cache.jsonl"
    ;;
  rag_hyde)
    LABEL_PREFIX="${LABEL_PREFIX:-hyde}"
    DEFAULT_RETRIEVAL_CACHE="$ROOT/caches/retrieval/full/housing_${CACHE_SCOPE}_${MODEL_LABEL}_${MODE}_k10.jsonl"
    DEFAULT_DOC_CACHE="$ROOT/caches/retrieval_doc/full/housing_${CACHE_SCOPE}_${MODEL_LABEL}_${MODE}_k10_doc_cache.jsonl"
    ;;
  snap_hyre)
    LABEL_PREFIX="${LABEL_PREFIX:-$MODE}"
    DEFAULT_RETRIEVAL_CACHE="$ROOT/caches/retrieval/full/housing_${CACHE_SCOPE}_${MODEL_LABEL}_${MODE}_k10.jsonl"
    DEFAULT_DOC_CACHE="$ROOT/caches/retrieval_doc/full/housing_${CACHE_SCOPE}_${MODEL_LABEL}_${MODE}_k10_doc_cache.jsonl"
    ;;
  golden_plus_neighbors)
    LABEL_PREFIX="${LABEL_PREFIX:-golden_neighbors}"
    DEFAULT_RETRIEVAL_CACHE="$ROOT/caches/retrieval/full/housing_${CACHE_SCOPE}_golden_neighbors_k10.jsonl"
    DEFAULT_DOC_CACHE="$ROOT/caches/retrieval_doc/full/housing_${CACHE_SCOPE}_golden_neighbors_k10_doc_cache.jsonl"
    ;;
  *)
    echo "unsupported cached Housing mode: $MODE" >&2
    exit 2
    ;;
esac

RETRIEVAL_CACHE="${RETRIEVAL_CACHE:-$DEFAULT_RETRIEVAL_CACHE}"
DOC_CACHE="${DOC_CACHE:-$DEFAULT_DOC_CACHE}"

mkdir -p "$LOG_DIR" "$(dirname "$DOC_CACHE")" "$(dirname "$LOCK_PATH")"

echo "[$(ts)] housing state-filter cached launch provider=$PROVIDER model_label=$MODEL_LABEL mode=$MODE"
echo "[$(ts)] sample=${SAMPLE_START}:${SAMPLE_END:-end}"
echo "[$(ts)] retrieval_cache=$RETRIEVAL_CACHE"
echo "[$(ts)] doc_cache=$DOC_CACHE"
echo "[$(ts)] hyre_cache_root=$HYRE_CACHE_ROOT"
echo "[$(ts)] retrieval_cache_root=$RETRIEVAL_CACHE_ROOT"
echo "[$(ts)] waiting for retrieval lock before doc-cache hydration: $LOCK_PATH"

flock "$LOCK_PATH" "$UV" run python scripts/build_retrieval_doc_cache.py \
  --retrieval-cache "$RETRIEVAL_CACHE" \
  --include-effective \
  --out "$DOC_CACHE" \
  --resume \
  --strict \
  --batch-size 500

RETRIEVAL_DOC_CACHE_PATH="$DOC_CACHE" \
RETRIEVAL_DOC_CACHE_STRICT=1 \
"$UV" run python scripts/smoke_retrieval_cache_hydration.py \
  --cache "$RETRIEVAL_CACHE" \
  --dataset housing \
  --label-prefix "$LABEL_PREFIX" \
  --questions "$QUESTIONS" \
  --seed "$SEED" \
  --retrieval-k "$RETRIEVAL_K" \
  --limit 20 \
  --housing-state-filter \
  --require-doc-cache

HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
NO_SILENT_FALLBACK=1 \
EVAL_FINAL_FORMAT_RETRY=1 \
EVAL_GENERATION_FORMAT_RETRY=1 \
LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}" \
EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}" \
PROVIDER="$PROVIDER" \
MODEL_LABEL="$MODEL_LABEL" \
DATASET=housing \
QUESTIONS="$QUESTIONS" \
SEED="$SEED" \
CACHE_SCOPE="$CACHE_SCOPE" \
HYRE_CACHE_ROOT="$HYRE_CACHE_ROOT" \
RETRIEVAL_CACHE_ROOT="$RETRIEVAL_CACHE_ROOT" \
SAMPLE_START="$SAMPLE_START" \
SAMPLE_END="$SAMPLE_END" \
RETRIEVAL_K="$RETRIEVAL_K" \
MODES="$MODE" \
USE_CACHES=1 \
REQUIRE_RETRIEVAL_CACHES=1 \
EVAL_HOUSING_STATE_FILTER=1 \
RETRIEVAL_DOC_CACHE_PATH="$DOC_CACHE" \
RETRIEVAL_DOC_CACHE_STRICT=1 \
scripts/local/run_answer_cell.sh
