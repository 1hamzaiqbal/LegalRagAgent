#!/usr/bin/env bash
# Build full HousingQA retrieval caches with jurisdiction state filtering.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

UV="${UV:-uv}"
LOG="${LOG:-logs/build_housing_statefilter_caches_$(date -u +%Y%m%d_%H%M%S).out}"
RESUME="${RESUME:-1}"

mkdir -p "$(dirname "$LOG")" caches/retrieval/full

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NO_SILENT_FALLBACK="${NO_SILENT_FALLBACK:-1}"
export CROSS_ENCODER_MAX_CHARS="${CROSS_ENCODER_MAX_CHARS:-4096}"
export PYTHONUNBUFFERED=1

resume_args=()
case "${RESUME,,}" in
  1|true|yes|on) resume_args=(--resume) ;;
  0|false|no|off) resume_args=() ;;
  *) echo "RESUME must be truthy/falsy, got $RESUME" >&2; exit 2 ;;
esac

run_step() {
  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
  "$@"
}

{
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] start housing state-filter cache build"
  echo "root=$ROOT"
  echo "cross_encoder_max_chars=$CROSS_ENCODER_MAX_CHARS"
  echo "resume=$RESUME"

  run_step "$UV" run python scripts/build_retrieval_cache.py \
    --dataset housing \
    --questions full \
    --seed 42 \
    --query-type raw_question \
    --max-k 10 \
    --housing-state-filter \
    --out caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl \
    "${resume_args[@]}" \
    --progress-interval 100

  run_step "$UV" run python scripts/build_retrieval_cache.py \
    --dataset housing \
    --questions full \
    --seed 42 \
    --query-type hyde_cache \
    --hyre-cache-path caches/hyre/full/housing_qfull_seed42_groq-llama8b_rag_hyde.jsonl \
    --expected-provider groq-llama8b \
    --max-k 10 \
    --housing-state-filter \
    --out caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_rag_hyde_k10.jsonl \
    "${resume_args[@]}" \
    --progress-interval 100

  run_step "$UV" run python scripts/build_retrieval_cache.py \
    --dataset housing \
    --questions full \
    --seed 42 \
    --query-type hyre_cache \
    --hyre-cache-path caches/hyre/full/housing_qfull_seed42_groq-llama8b_snap_hyre.jsonl \
    --expected-provider groq-llama8b \
    --max-k 10 \
    --housing-state-filter \
    --out caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl \
    "${resume_args[@]}" \
    --progress-interval 100

  run_step "$UV" run python scripts/build_retrieval_cache.py \
    --dataset housing \
    --questions full \
    --seed 42 \
    --query-type hyde_cache \
    --hyre-cache-path caches/hyre/full/housing_qfull_seed42_groq-llama70b_rag_hyde.jsonl \
    --expected-provider groq-llama70b \
    --max-k 10 \
    --housing-state-filter \
    --out caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_rag_hyde_k10.jsonl \
    "${resume_args[@]}" \
    --progress-interval 100

  run_step "$UV" run python scripts/build_retrieval_cache.py \
    --dataset housing \
    --questions full \
    --seed 42 \
    --query-type hyre_cache \
    --hyre-cache-path caches/hyre/full/housing_qfull_seed42_groq-llama70b_snap_hyre.jsonl \
    --expected-provider groq-llama70b \
    --max-k 10 \
    --housing-state-filter \
    --out caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl \
    "${resume_args[@]}" \
    --progress-interval 100

  run_step env GOLDEN_NEIGHBORS_STORED_EMBEDDING=1 "$UV" run python scripts/build_retrieval_cache.py \
    --dataset housing \
    --questions full \
    --seed 42 \
    --query-type golden_neighbors \
    --max-k 10 \
    --housing-state-filter \
    --out caches/retrieval/full/housing_qfull_seed42_statefilter_golden_neighbors_k10.jsonl \
    "${resume_args[@]}" \
    --progress-interval 100

  echo
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done housing state-filter cache build"
} 2>&1 | tee -a "$LOG"
