#!/usr/bin/env bash
# Run one local dataset/provider Snap-HyRE answer ladder cell.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

UV="${UV:-uv}"
PROVIDER="${PROVIDER:-or-gemma4-26b}"
MODEL_LABEL="${MODEL_LABEL:-$PROVIDER}"
DATASET="${DATASET:-legalbench_scalr}"
QUESTIONS="${QUESTIONS:-50}"
SEED="${SEED:-42}"
SAMPLE_START="${SAMPLE_START:-0}"
SAMPLE_END="${SAMPLE_END:-}"
RETRIEVAL_K="${RETRIEVAL_K:-5}"
USE_CACHES="${USE_CACHES:-1}"
REQUIRE_RETRIEVAL_CACHES="${REQUIRE_RETRIEVAL_CACHES:-1}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-1}"
LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-768}"
HYRE_CACHE_ROOT="${HYRE_CACHE_ROOT:-$ROOT/caches/hyre/full}"
RETRIEVAL_CACHE_ROOT="${RETRIEVAL_CACHE_ROOT:-$ROOT/caches/retrieval/full}"
BAREXAM_COLLECTION="${BAREXAM_COLLECTION:-}"
CACHE_SCOPE="${CACHE_SCOPE:-}"

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(llm_only rag_simple rag_hyde snap_hyre golden_passage golden_plus_neighbors rag_rewrite)
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

export CHROMA_DB_DIR="${CHROMA_DB_DIR:-$ROOT/chroma_db}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DISABLE_CROSS_ENCODER="${DISABLE_CROSS_ENCODER:-0}"
export LLM_MAX_COMPLETION_TOKENS
export PYTHONUNBUFFERED=1

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

if [[ "$DATASET" == "barexam" && -n "$BAREXAM_COLLECTION" ]]; then
  if [[ -n "${EVAL_COLLECTION_OVERRIDE:-}" && "$EVAL_COLLECTION_OVERRIDE" != "$BAREXAM_COLLECTION" ]]; then
    echo "conflicting EVAL_COLLECTION_OVERRIDE=$EVAL_COLLECTION_OVERRIDE and BAREXAM_COLLECTION=$BAREXAM_COLLECTION" >&2
    exit 2
  fi
  export EVAL_COLLECTION_OVERRIDE="$BAREXAM_COLLECTION"
fi

mkdir -p logs

echo "[$(ts)] local answer cell root=$ROOT commit=$(git rev-parse --short HEAD)"
echo "[$(ts)] provider=$PROVIDER model_label=$MODEL_LABEL dataset=$DATASET questions=$QUESTIONS seed=$SEED sample=${SAMPLE_START}:${SAMPLE_END:-end} retrieval_k=$RETRIEVAL_K"
echo "[$(ts)] cache_scope=$CACHE_SCOPE"
echo "[$(ts)] modes=${MODES_ARR[*]} use_caches=$USE_CACHES require_retrieval_caches=$REQUIRE_RETRIEVAL_CACHES"
if [[ "$DATASET" == "barexam" && -n "${EVAL_COLLECTION_OVERRIDE:-}" ]]; then
  echo "[$(ts)] barexam_collection=$EVAL_COLLECTION_OVERRIDE"
fi

"$UV" run python -m py_compile eval/eval_harness.py scripts/analyze_detail_flags.py

add_cache_args_for_mode() {
  local mode="$1"
  local hyre_cache=""
  local retrieval_cache=""
  extra_args=()

  case "$mode" in
    rag_simple)
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_${CACHE_SCOPE}_raw_question_k10.jsonl"
      ;;
    rag_hyde)
      hyre_cache="$HYRE_CACHE_ROOT/${DATASET}_${CACHE_SCOPE}_${MODEL_LABEL}_rag_hyde.jsonl"
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_${CACHE_SCOPE}_${MODEL_LABEL}_rag_hyde_k10.jsonl"
      ;;
    snap_hyre)
      hyre_cache="$HYRE_CACHE_ROOT/${DATASET}_${CACHE_SCOPE}_${MODEL_LABEL}_snap_hyre.jsonl"
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_${CACHE_SCOPE}_${MODEL_LABEL}_snap_hyre_k10.jsonl"
      ;;
    golden_plus_neighbors)
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_${CACHE_SCOPE}_golden_neighbors_k10.jsonl"
      ;;
  esac

  if [[ "$USE_CACHES" != "1" ]]; then
    return 0
  fi
  if [[ -n "$hyre_cache" ]]; then
    [[ -s "$hyre_cache" ]] || { echo "missing or empty hyre cache $hyre_cache" >&2; return 2; }
    extra_args+=(--hyre-cache-path "$hyre_cache")
  fi
  if [[ -n "$retrieval_cache" ]]; then
    if [[ -s "$retrieval_cache" ]]; then
      extra_args+=(--retrieval-cache-path "$retrieval_cache")
    elif [[ "$REQUIRE_RETRIEVAL_CACHES" == "1" ]]; then
      echo "missing or empty retrieval cache $retrieval_cache" >&2
      return 2
    else
      echo "[$(ts)] WARNING: missing or empty retrieval cache $retrieval_cache; running mode=$mode without cache"
    fi
  fi
}

for mode in "${MODES_ARR[@]}"; do
  tag="local-snap-hyre-${MODEL_LABEL}-${DATASET}-${mode}-n${QUESTIONS}-k${RETRIEVAL_K}"
  echo
  echo "[$(ts)] run dataset=$DATASET provider=$PROVIDER mode=$mode tag=$tag"

  if ! add_cache_args_for_mode "$mode"; then
    echo "[$(ts)] FAILED dataset=$DATASET mode=$mode while resolving caches"
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit 2
    fi
    continue
  fi

  set +e
  eval_cmd=(
    "$UV" run python eval/eval_harness.py
    --mode "$mode"
    --provider "$PROVIDER"
    --dataset "$DATASET"
    --questions "$QUESTIONS"
    --seed "$SEED"
    "${sample_args[@]}"
    --retrieval-k "$RETRIEVAL_K"
    --tag "$tag"
  )
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    eval_cmd+=("${extra_args[@]}")
  fi
  LLM_PROVIDER="$PROVIDER" \
  EVAL_TRACE_CALLS=1 \
  EVAL_TRACE_EVENTS=1 \
  EVAL_TRACE_MAX_CHARS=1200 \
  "${eval_cmd[@]}"
  status=$?
  set -e

  latest_log="$(find logs -maxdepth 1 -name "eval_${mode}_${PROVIDER}_*_${DATASET}_*${tag}*_detail.jsonl" -print | sort | tail -n 1)"
  if [[ -z "$latest_log" ]]; then
    echo "[$(ts)] ERROR: no detail log found for dataset=$DATASET provider=$PROVIDER mode=$mode"
    status=1
  else
    "$UV" run python scripts/analyze_detail_flags.py "$latest_log" || status=1
    "$UV" run python - "$latest_log" <<'PY' || status=1
import json
import sys

path = sys.argv[1]
bad = []
errors = []
with open(path) as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        pred = row.get("predicted_answer")
        if pred is None or str(pred).strip() == "":
            bad.append(str(row.get("label") or row.get("idx") or line_no))
        if row.get("error"):
            errors.append(str(row.get("label") or row.get("idx") or line_no))
if bad:
    raise SystemExit("missing predicted_answer rows: " + ",".join(bad[:10]))
if errors:
    raise SystemExit("error rows: " + ",".join(errors[:10]))
PY
  fi

  if [[ "$status" -ne 0 ]]; then
    echo "[$(ts)] FAILED dataset=$DATASET provider=$PROVIDER mode=$mode exit=$status"
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit "$status"
    fi
  else
    echo "[$(ts)] OK dataset=$DATASET provider=$PROVIDER mode=$mode"
  fi
done

echo "[$(ts)] local answer cell complete."
