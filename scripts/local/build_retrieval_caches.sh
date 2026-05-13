#!/usr/bin/env bash
# Build and audit local retrieval-id caches for Snap-HyRE top-k selection.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

UV="${UV:-uv}"
CACHE_DIR="${CACHE_DIR:-$ROOT/caches/retrieval/full}"
QUESTIONS="${QUESTIONS:-full}"
MAX_K="${MAX_K:-10}"
KS="${KS:-1,3,5,10}"
ALIGN_MIN_EXISTS="${ALIGN_MIN_EXISTS:-0.95}"
ALIGN_METADATA_FALLBACK="${ALIGN_METADATA_FALLBACK:-0}"

if [[ -n "${DATASETS:-}" ]]; then
  # shellcheck disable=SC2206
  DATASETS_ARR=(${DATASETS})
else
  DATASETS_ARR=(barexam housing casehold legalbench_scalr)
fi

if [[ -n "${QUERY_TYPES:-}" ]]; then
  # shellcheck disable=SC2206
  QUERY_TYPES_ARR=(${QUERY_TYPES})
else
  QUERY_TYPES_ARR=(raw_question golden_neighbors)
fi

mkdir -p "$CACHE_DIR" docs/generated

export CHROMA_DB_DIR="${CHROMA_DB_DIR:-$ROOT/chroma_db}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DISABLE_CROSS_ENCODER="${DISABLE_CROSS_ENCODER:-1}"
export PYTHONUNBUFFERED=1

echo "[$(ts)] local retrieval cache root=$ROOT commit=$(git rev-parse --short HEAD)"
echo "[$(ts)] chroma=$CHROMA_DB_DIR cache_dir=$CACHE_DIR questions=$QUESTIONS max_k=$MAX_K"
echo "[$(ts)] datasets=${DATASETS_ARR[*]} query_types=${QUERY_TYPES_ARR[*]}"

"$UV" run python -m py_compile \
  eval/eval_config.py \
  eval/eval_harness.py \
  rag_utils.py \
  scripts/audit_retrieval_id_alignment.py \
  scripts/build_retrieval_cache.py \
  scripts/audit_retrieval_cache.py \
  scripts/compile_retrieval_cache_matrix.py

outputs=()

for dataset in "${DATASETS_ARR[@]}"; do
  alignment_report="$CACHE_DIR/retrieval_id_alignment_${dataset}.txt"
  alignment_cmd=(
    "$UV" run python scripts/audit_retrieval_id_alignment.py
    --dataset "$dataset"
    --questions "$QUESTIONS"
    --min-exists "$ALIGN_MIN_EXISTS"
  )
  if [[ "$ALIGN_METADATA_FALLBACK" == "1" ]]; then
    alignment_cmd+=(--metadata-fallback)
  fi

  echo
  echo "[$(ts)] audit retrieval-id alignment dataset=$dataset"
  if "${alignment_cmd[@]}" > "$alignment_report" 2>&1; then
    echo "[$(ts)] alignment OK dataset=$dataset report=$alignment_report"
  else
    echo "[$(ts)] WARNING: alignment failed dataset=$dataset; Hit/MRR is not promotable without repair"
    cat "$alignment_report"
  fi

  for query_type in "${QUERY_TYPES_ARR[@]}"; do
    out="$CACHE_DIR/${dataset}_${query_type}_k${MAX_K}.jsonl"
    echo
    echo "[$(ts)] build dataset=$dataset query_type=$query_type out=$out"
    "$UV" run python scripts/build_retrieval_cache.py \
      --dataset "$dataset" \
      --questions "$QUESTIONS" \
      --query-type "$query_type" \
      --max-k "$MAX_K" \
      --out "$out"

    "$UV" run python scripts/audit_retrieval_cache.py \
      --cache "$out" \
      --dataset "$dataset" \
      --query-type "$query_type" \
      --min-k "$MAX_K" \
      --ks "$KS"
    outputs+=("$out")
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
    --out-md docs/generated/retrieval_cache_matrix.md \
    --out-csv docs/generated/retrieval_cache_matrix.csv

  echo "[$(ts)] wrote docs/generated/retrieval_cache_matrix.md"
  echo "[$(ts)] wrote docs/generated/retrieval_cache_matrix.csv"
fi

echo "[$(ts)] local retrieval cache pass complete."
