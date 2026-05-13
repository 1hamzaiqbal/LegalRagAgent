#!/usr/bin/env bash
# Smoke API providers and core fixed-method modes before full local sweeps.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

UV="${UV:-uv}"
DATASET="${DATASET:-legalbench_scalr}"
QUESTIONS="${QUESTIONS:-2}"
SEED="${SEED:-42}"
SAMPLE_START="${SAMPLE_START:-0}"
SAMPLE_END="${SAMPLE_END:-}"
RETRIEVAL_K="${RETRIEVAL_K:-3}"
LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-768}"
BAREXAM_COLLECTION="${BAREXAM_COLLECTION:-}"

if [[ -n "${PROVIDERS:-}" ]]; then
  # shellcheck disable=SC2206
  PROVIDERS_ARR=(${PROVIDERS})
else
  PROVIDERS_ARR=(or-gemma3n-e4b or-gemma4-26b groq-llama70b)
fi

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(rag_simple snap_hyre)
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

need_key_for_provider() {
  local provider="$1"
  case "$provider" in
    or-*) [[ -n "${OPENROUTER_API_KEY:-}" ]] || return 1 ;;
    groq-*) [[ -n "${GROQ_API_KEY:-}" ]] || return 1 ;;
  esac
}

for provider in "${PROVIDERS_ARR[@]}"; do
  if ! need_key_for_provider "$provider"; then
    echo "[$(ts)] ERROR: missing API key for provider=$provider" >&2
    exit 2
  fi
done

export CHROMA_DB_DIR="${CHROMA_DB_DIR:-$ROOT/chroma_db}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export DISABLE_CROSS_ENCODER="${DISABLE_CROSS_ENCODER:-0}"
export LLM_MAX_COMPLETION_TOKENS
export PYTHONUNBUFFERED=1

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

echo "[$(ts)] local API smoke root=$ROOT commit=$(git rev-parse --short HEAD)"
echo "[$(ts)] dataset=$DATASET questions=$QUESTIONS seed=$SEED sample=${SAMPLE_START}:${SAMPLE_END:-end} retrieval_k=$RETRIEVAL_K"
echo "[$(ts)] providers=${PROVIDERS_ARR[*]} modes=${MODES_ARR[*]}"
if [[ "$DATASET" == "barexam" && -n "${EVAL_COLLECTION_OVERRIDE:-}" ]]; then
  echo "[$(ts)] barexam_collection=$EVAL_COLLECTION_OVERRIDE"
fi

"$UV" run python -m py_compile eval/eval_harness.py scripts/analyze_detail_flags.py

failures=0
for provider in "${PROVIDERS_ARR[@]}"; do
  for mode in "${MODES_ARR[@]}"; do
    tag="local-api-smoke-${provider}-${mode}-n${QUESTIONS}-k${RETRIEVAL_K}"
    echo
    echo "[$(ts)] run provider=$provider mode=$mode tag=$tag"
    set +e
    LLM_PROVIDER="$provider" \
    EVAL_TRACE_CALLS=1 \
    EVAL_TRACE_EVENTS=1 \
    EVAL_TRACE_MAX_CHARS=1200 \
    "$UV" run python eval/eval_harness.py \
      --mode "$mode" \
      --provider "$provider" \
      --dataset "$DATASET" \
      --questions "$QUESTIONS" \
      --seed "$SEED" \
      "${sample_args[@]}" \
      --retrieval-k "$RETRIEVAL_K" \
      --tag "$tag"
    status=$?
    set -e

    latest_log="$(find logs -maxdepth 1 -name "eval_${mode}_${provider}_*_${DATASET}_*${tag}*_detail.jsonl" -print | sort | tail -n 1)"
    if [[ -z "$latest_log" ]]; then
      echo "[$(ts)] ERROR: no detail log found for provider=$provider mode=$mode"
      status=1
    else
      "$UV" run python scripts/analyze_detail_flags.py "$latest_log" || status=1
      "$UV" run python - "$latest_log" <<'PY' || status=1
import json
import sys

path = sys.argv[1]
bad = []
with open(path) as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        pred = row.get("predicted_answer")
        if pred is None or str(pred).strip() == "":
            bad.append(str(row.get("label") or row.get("idx") or line_no))
if bad:
    raise SystemExit("missing predicted_answer rows: " + ",".join(bad[:10]))
PY
    fi

    if [[ "$status" -ne 0 ]]; then
      failures=$((failures + 1))
      echo "[$(ts)] FAILED provider=$provider mode=$mode exit=$status"
    else
      echo "[$(ts)] OK provider=$provider mode=$mode"
    fi
  done
done

if [[ "$failures" -gt 0 ]]; then
  echo "[$(ts)] API smoke completed with $failures failure(s)"
  exit 1
fi

echo "[$(ts)] API smoke complete."
