#!/usr/bin/env bash
# Replay cached Gemma 26B Snap-HyRE exemplar probes through the answer stage.
#
# Default scope intentionally excludes HousingQA: the available q500 Housing
# exemplar cache is the unfiltered provenance probe, while the main Housing
# matrix now requires state-filtered retrieval.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NO_SILENT_FALLBACK="${NO_SILENT_FALLBACK:-1}"
export EVAL_FINAL_FORMAT_RETRY="${EVAL_FINAL_FORMAT_RETRY:-1}"
export EVAL_GENERATION_FORMAT_RETRY="${EVAL_GENERATION_FORMAT_RETRY:-1}"
export LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}"
export EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}"
export LLM_CALL_MIN_INTERVAL_SEC="${LLM_CALL_MIN_INTERVAL_SEC:-2.0}"
export LLM_CALL_RATE_LIMIT_COOLDOWN_SEC="${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-8.0}"
export OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"
export GENERATION_CACHE_ROOT="${GENERATION_CACHE_ROOT:-$ROOT/caches/generation/probes}"
export RETRIEVAL_CACHE_ROOT="${RETRIEVAL_CACHE_ROOT:-$ROOT/caches/retrieval/probes}"
export USE_CACHES=1
export REQUIRE_RETRIEVAL_CACHES=1
export STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"

PROVIDER="${PROVIDER:-or-gemma4-26b}"
MODEL_LABEL="${MODEL_LABEL:-$PROVIDER}"
DATASETS="${DATASETS:-barexam legal_link_eu mas_legal_bench}"

echo "[$(ts)] snap_hyre_exemplar answer probe start provider=$PROVIDER model_label=$MODEL_LABEL datasets=$DATASETS"
echo "[$(ts)] generation_cache_root=$GENERATION_CACHE_ROOT"
echo "[$(ts)] retrieval_cache_root=$RETRIEVAL_CACHE_ROOT"

for dataset in $DATASETS; do
  questions="500"
  cache_scope="q500_seed42"
  sample_start="0"
  if [[ "$dataset" == "mas_legal_bench" ]]; then
    questions="full"
    cache_scope="qfull_seed42"
  fi
  if [[ "$dataset" == "barexam" ]]; then
    sample_start="${BAREXAM_SAMPLE_START:-0}"
  fi
  echo
  echo "[$(ts)] exemplar answer dataset=$dataset questions=$questions cache_scope=$cache_scope sample_start=$sample_start"
  PROVIDER="$PROVIDER" \
  MODEL_LABEL="$MODEL_LABEL" \
  DATASET="$dataset" \
  QUESTIONS="$questions" \
  SAMPLE_START="$sample_start" \
  CACHE_SCOPE="$cache_scope" \
  MODES="snap_hyre_exemplar" \
  RETRIEVAL_K=5 \
  scripts/local/run_answer_cell.sh
done

echo "[$(ts)] snap_hyre_exemplar answer probe complete"
