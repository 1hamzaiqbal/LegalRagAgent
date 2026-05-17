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
SAMPLE_START="${SAMPLE_START:-0}"
SAMPLE_END="${SAMPLE_END:-}"
MAX_K="${MAX_K:-10}"
KS="${KS:-1,3,5,10}"
RESUME="${RESUME:-1}"
TRACE_CALLS="${TRACE_CALLS:-1}"
TRACE_EVENTS="${TRACE_EVENTS:-1}"
ENV_LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-}"
LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}"
EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}"
EVAL_GENERATION_FORMAT_RETRY="${EVAL_GENERATION_FORMAT_RETRY:-1}"
HYRE_CACHE_ROOT="${HYRE_CACHE_ROOT:-$ROOT/caches/hyre/full}"
RETRIEVAL_CACHE_ROOT="${RETRIEVAL_CACHE_ROOT:-$ROOT/caches/retrieval/full}"
BAREXAM_COLLECTION="${BAREXAM_COLLECTION:-}"
CACHE_SCOPE="${CACHE_SCOPE:-}"

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
if [[ -n "$ENV_LLM_MAX_COMPLETION_TOKENS" ]]; then
  LLM_MAX_COMPLETION_TOKENS="$ENV_LLM_MAX_COMPLETION_TOKENS"
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
NO_SILENT_FALLBACK="${NO_SILENT_FALLBACK:-1}"
case "${NO_SILENT_FALLBACK,,}" in
  1|true|yes|on) ;;
  *) echo "NO_SILENT_FALLBACK must be enabled for generation cache runs, got $NO_SILENT_FALLBACK" >&2; exit 2 ;;
esac

case "$PROVIDER" in
  or-*) [[ -n "${OPENROUTER_API_KEY:-}" ]] || { echo "missing OPENROUTER_API_KEY for $PROVIDER" >&2; exit 2; } ;;
  groq-*) [[ -n "${GROQ_API_KEY:-}" ]] || { echo "missing GROQ_API_KEY for $PROVIDER" >&2; exit 2; } ;;
esac

mkdir -p "$HYRE_CACHE_ROOT" "$RETRIEVAL_CACHE_ROOT" docs/generated

if [[ -z "$CACHE_SCOPE" ]]; then
  CACHE_SCOPE="q${QUESTIONS}_seed${SEED}"
  if [[ "$SAMPLE_START" != "0" || -n "$SAMPLE_END" ]]; then
    CACHE_SCOPE="${CACHE_SCOPE}_s${SAMPLE_START}_e${SAMPLE_END:-end}"
  fi
fi

sample_args=(--sample-start "$SAMPLE_START")
if [[ -n "$SAMPLE_END" ]]; then
  sample_args+=(--sample-end "$SAMPLE_END")
fi

export CHROMA_DB_DIR="${CHROMA_DB_DIR:-$ROOT/chroma_db}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DISABLE_CROSS_ENCODER="${DISABLE_CROSS_ENCODER:-0}"
export LLM_MAX_COMPLETION_TOKENS
export EVAL_GENERATION_FORMAT_RETRY
export NO_SILENT_FALLBACK
if [[ "$PROVIDER" == or-* ]]; then
  export LLM_CALL_MIN_INTERVAL_SEC="${LLM_CALL_MIN_INTERVAL_SEC:-2.0}"
  export LLM_CALL_RATE_LIMIT_COOLDOWN_SEC="${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-8.0}"
fi
export PYTHONUNBUFFERED=1

echo "[$(ts)] local generation cache root=$ROOT commit=$(git rev-parse --short HEAD)"
echo "[$(ts)] provider=$PROVIDER model_label=$MODEL_LABEL questions=$QUESTIONS seed=$SEED sample=${SAMPLE_START}:${SAMPLE_END:-end} max_k=$MAX_K"
echo "[$(ts)] cache_scope=$CACHE_SCOPE"
echo "[$(ts)] datasets=${DATASETS_ARR[*]} modes=${MODES_ARR[*]}"
echo "[$(ts)] no_silent_fallback=$NO_SILENT_FALLBACK"
echo "[$(ts)] llm_max_completion_tokens=$LLM_MAX_COMPLETION_TOKENS eval_min_completion_tokens=$EVAL_MIN_COMPLETION_TOKENS"
echo "[$(ts)] eval_generation_format_retry=$EVAL_GENERATION_FORMAT_RETRY"
if [[ -n "${LLM_CALL_MIN_INTERVAL_SEC:-}" || -n "${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-}" ]]; then
  echo "[$(ts)] llm_call_min_interval=${LLM_CALL_MIN_INTERVAL_SEC:-0} rate_limit_cooldown=${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-0}"
fi
if [[ -n "$BAREXAM_COLLECTION" ]]; then
  echo "[$(ts)] barexam_collection=$BAREXAM_COLLECTION"
fi

"$UV" run python -m py_compile \
  eval/eval_config.py \
  eval/eval_harness.py \
  scripts/build_generation_cache.py \
  scripts/build_retrieval_cache.py \
  scripts/audit_retrieval_cache.py \
  scripts/compile_retrieval_cache_matrix.py

outputs=()

for dataset in "${DATASETS_ARR[@]}"; do
  collection_args=()
  if [[ "$dataset" == "barexam" && -n "$BAREXAM_COLLECTION" ]]; then
    collection_args=(--collection "$BAREXAM_COLLECTION")
  fi

  for mode in "${MODES_ARR[@]}"; do
    case "$mode" in
      rag_hyde) query_type="hyde_cache" ;;
      snap_hyre) query_type="hyre_cache" ;;
      *) echo "unknown generation mode=$mode; expected rag_hyde or snap_hyre" >&2; exit 2 ;;
    esac

    gen_out="$HYRE_CACHE_ROOT/${dataset}_${CACHE_SCOPE}_${MODEL_LABEL}_${mode}.jsonl"
    ret_out="$RETRIEVAL_CACHE_ROOT/${dataset}_${CACHE_SCOPE}_${MODEL_LABEL}_${mode}_k${MAX_K}.jsonl"
    tag="local-gen-${MODEL_LABEL}-${dataset}-${mode}-n${QUESTIONS}"

    echo
    echo "[$(ts)] build generation dataset=$dataset mode=$mode out=$gen_out"
    gen_cmd=(
      "$UV" run python scripts/build_generation_cache.py
      --mode "$mode"
      --provider "$PROVIDER"
      --dataset "$dataset"
      --questions "$QUESTIONS"
      --seed "$SEED"
      "${sample_args[@]}"
      --tag "$tag"
      --out "$gen_out"
    )
    if [[ "$RESUME" == "1" ]]; then
      gen_cmd+=(--resume)
    fi
    if [[ "$TRACE_CALLS" == "1" ]]; then
      gen_cmd+=(--trace-calls)
    fi
    if [[ "$TRACE_EVENTS" == "1" ]]; then
      gen_cmd+=(--trace-events)
    fi
    LLM_PROVIDER="$PROVIDER" \
    NO_SILENT_FALLBACK="$NO_SILENT_FALLBACK" \
    "${gen_cmd[@]}"

    "$UV" run python - "$gen_out" "$mode" <<'PY'
import json
import sys

path, mode = sys.argv[1], sys.argv[2]
rows = []
with open(path) as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
errors = [r for r in rows if r.get("error")]
missing = [r for r in rows if not r.get("hyde_passage")]
fallbacks = [
    r for r in rows
    if r.get("hyde_used_fallback") is True
    or any(k.endswith("_used_fallback") and v for k, v in r.items())
]
parse_fail = [
    r for r in rows
    if (mode == "snap_hyre" and r.get("snap_hyre_parse_ok") is False)
    or r.get("hyde_parse_ok") is False
]
missing_snap = [r for r in rows if mode == "snap_hyre" and not r.get("snap_letter")]
artifacts = [r for r in rows if r.get("hyde_contains_answer_artifact") is True]
print(f"[postcheck] generation path={path} rows={len(rows)} errors={len(errors)} missing_hyde={len(missing)} fallbacks={len(fallbacks)} parse_fail={len(parse_fail)} missing_snap_letter={len(missing_snap)} answer_artifacts={len(artifacts)}")
if errors:
    raise SystemExit("generation errors: " + ",".join(str(r.get("label")) for r in errors[:10]))
if missing:
    raise SystemExit("missing hyde_passage: " + ",".join(str(r.get("label")) for r in missing[:10]))
if fallbacks:
    raise SystemExit("generation fallback rows: " + ",".join(str(r.get("label")) for r in fallbacks[:10]))
if parse_fail:
    raise SystemExit("generation parse failures: " + ",".join(str(r.get("label")) for r in parse_fail[:10]))
if missing_snap:
    raise SystemExit("missing snap_letter: " + ",".join(str(r.get("label")) for r in missing_snap[:10]))
if artifacts:
    raise SystemExit("generation answer-artifact rows: " + ",".join(str(r.get("label")) for r in artifacts[:10]))
PY

    echo "[$(ts)] build retrieval-from-generation dataset=$dataset mode=$mode out=$ret_out"
    "$UV" run python scripts/build_retrieval_cache.py \
      --dataset "$dataset" \
      --questions "$QUESTIONS" \
      --seed "$SEED" \
      "${sample_args[@]}" \
      --query-type "$query_type" \
      --hyre-cache-path "$gen_out" \
      "${collection_args[@]}" \
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
