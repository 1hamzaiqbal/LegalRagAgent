#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NO_SILENT_FALLBACK=1
export EVAL_GENERATION_FORMAT_RETRY=1
export PYTHONUNBUFFERED=1
export OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"

provider="or-gemma4-26b"
seed="42"
max_k="10"

echo "[start] $(date -Is) provider=${provider} route=${OPENROUTER_PROVIDER_ONLY}"

build_generation() {
  local dataset="$1"
  local questions="$2"
  local qtag="$3"
  local mode="$4"
  local out="caches/generation/probes/${dataset}_${qtag}_seed${seed}_${provider}_${mode}"
  if [[ "${mode}" == "snap_hyre_exemplar" ]]; then
    out="${out}_realpassage"
  fi
  out="${out}.jsonl"
  echo "[generation] dataset=${dataset} questions=${questions} mode=${mode} out=${out}"
  uv run python scripts/build_generation_cache.py \
    --mode "${mode}" \
    --provider "${provider}" \
    --dataset "${dataset}" \
    --questions "${questions}" \
    --seed "${seed}" \
    --out "${out}" \
    --trace-calls \
    --trace-events \
    --resume
}

build_retrieval() {
  local dataset="$1"
  local questions="$2"
  local qtag="$3"
  local query_type="$4"
  local label_prefix="$5"
  local source_cache="${6:-}"
  local out_suffix="$7"
  local out="caches/retrieval/probes/${dataset}_${qtag}_seed${seed}_${out_suffix}_k${max_k}.jsonl"
  echo "[retrieval] dataset=${dataset} questions=${questions} query_type=${query_type} out=${out}"
  if [[ -n "${source_cache}" ]]; then
    uv run python scripts/build_retrieval_cache.py \
      --dataset "${dataset}" \
      --questions "${questions}" \
      --seed "${seed}" \
      --query-type "${query_type}" \
      --label-prefix "${label_prefix}" \
      --hyre-cache-path "${source_cache}" \
      --max-k "${max_k}" \
      --out "${out}" \
      --resume
  else
    uv run python scripts/build_retrieval_cache.py \
      --dataset "${dataset}" \
      --questions "${questions}" \
      --seed "${seed}" \
      --query-type "${query_type}" \
      --label-prefix "${label_prefix}" \
      --max-k "${max_k}" \
      --out "${out}" \
      --resume
  fi
}

if [[ "${RUN_GENERATION:-1}" == "1" ]]; then
  # Generation: exemplar for all exact-scored benchmarks, plus missing Housing canonical Snap.
  build_generation "barexam" "500" "q500" "snap_hyre_exemplar"
  build_generation "housing" "500" "q500" "snap_hyre"
  build_generation "housing" "500" "q500" "snap_hyre_exemplar"
  build_generation "legal_link_eu" "500" "q500" "snap_hyre_exemplar"
  build_generation "mas_legal_bench" "full" "qfull" "snap_hyre_exemplar"
else
  echo "[generation] skipped RUN_GENERATION=${RUN_GENERATION}"
fi

# BarExam retrieval.
build_retrieval "barexam" "500" "q500" "raw_question" "raw_question" "" "raw_question"
build_retrieval "barexam" "500" "q500" "hyre_cache" "snap_hyre" \
  "caches/hyre/full/barexam_qfull_seed${seed}_${provider}_snap_hyre.jsonl" \
  "${provider}_snap_hyre"
build_retrieval "barexam" "500" "q500" "hyre_cache" "snap_hyre_exemplar" \
  "caches/generation/probes/barexam_q500_seed${seed}_${provider}_snap_hyre_exemplar_realpassage.jsonl" \
  "${provider}_snap_hyre_exemplar_realpassage"

# Housing retrieval.
build_retrieval "housing" "500" "q500" "raw_question" "raw_question" "" "raw_question"
build_retrieval "housing" "500" "q500" "hyre_cache" "snap_hyre" \
  "caches/generation/probes/housing_q500_seed${seed}_${provider}_snap_hyre.jsonl" \
  "${provider}_snap_hyre"
build_retrieval "housing" "500" "q500" "hyre_cache" "snap_hyre_exemplar" \
  "caches/generation/probes/housing_q500_seed${seed}_${provider}_snap_hyre_exemplar_realpassage.jsonl" \
  "${provider}_snap_hyre_exemplar_realpassage"

# Legal-Link-EU retrieval uses the current full-run CE window.
export CROSS_ENCODER_MAX_CHARS=22000
build_retrieval "legal_link_eu" "500" "q500" "raw_question" "raw_question" "" "raw_question_ce22000"
build_retrieval "legal_link_eu" "500" "q500" "hyre_cache" "snap_hyre" \
  "caches/hyre/full/legal_link_eu_qfull_seed${seed}_${provider}_snap_hyre.jsonl" \
  "${provider}_snap_hyre_ce22000"
build_retrieval "legal_link_eu" "500" "q500" "hyre_cache" "snap_hyre_exemplar" \
  "caches/generation/probes/legal_link_eu_q500_seed${seed}_${provider}_snap_hyre_exemplar_realpassage.jsonl" \
  "${provider}_snap_hyre_exemplar_realpassage_ce22000"
unset CROSS_ENCODER_MAX_CHARS

# MASLegalBench has only 303 exact-scored rows; retrieval is evaluated with source-document proxy.
build_retrieval "mas_legal_bench" "full" "qfull" "raw_question" "raw_question" "" "raw_question"
build_retrieval "mas_legal_bench" "full" "qfull" "hyre_cache" "snap_hyre" \
  "caches/generation/full/mas_legal_bench_qfull_seed${seed}_${provider}_snap_hyre.jsonl" \
  "${provider}_snap_hyre"
build_retrieval "mas_legal_bench" "full" "qfull" "hyre_cache" "snap_hyre_exemplar" \
  "caches/generation/probes/mas_legal_bench_qfull_seed${seed}_${provider}_snap_hyre_exemplar_realpassage.jsonl" \
  "${provider}_snap_hyre_exemplar_realpassage"

echo "[done] $(date -Is)"
