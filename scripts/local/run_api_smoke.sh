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
ENV_LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-}"
LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}"
EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}"
EVAL_FINAL_FORMAT_RETRY="${EVAL_FINAL_FORMAT_RETRY:-1}"
BAREXAM_COLLECTION="${BAREXAM_COLLECTION:-}"

if [[ -n "${PROVIDERS:-}" ]]; then
  # shellcheck disable=SC2206
  PROVIDERS_ARR=(${PROVIDERS})
else
  PROVIDERS_ARR=(groq-llama8b or-gemma4-26b groq-llama70b)
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
  echo "LLM_MAX_COMPLETION_TOKENS=$LLM_MAX_COMPLETION_TOKENS is below EVAL_MIN_COMPLETION_TOKENS=$EVAL_MIN_COMPLETION_TOKENS; refusing truncation-prone smoke run" >&2
  exit 2
fi
NO_SILENT_FALLBACK="${NO_SILENT_FALLBACK:-1}"
case "${NO_SILENT_FALLBACK,,}" in
  1|true|yes|on) ;;
  *) echo "NO_SILENT_FALLBACK must be enabled for API smoke, got $NO_SILENT_FALLBACK" >&2; exit 2 ;;
esac

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
export EVAL_FINAL_FORMAT_RETRY
export NO_SILENT_FALLBACK
if printf '%s\n' "${PROVIDERS_ARR[@]}" | grep -q '^or-'; then
  export LLM_CALL_MIN_INTERVAL_SEC="${LLM_CALL_MIN_INTERVAL_SEC:-2.0}"
  export LLM_CALL_RATE_LIMIT_COOLDOWN_SEC="${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-8.0}"
fi
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
echo "[$(ts)] no_silent_fallback=$NO_SILENT_FALLBACK"
echo "[$(ts)] llm_max_completion_tokens=$LLM_MAX_COMPLETION_TOKENS eval_min_completion_tokens=$EVAL_MIN_COMPLETION_TOKENS eval_final_format_retry=$EVAL_FINAL_FORMAT_RETRY"
if [[ -n "${LLM_CALL_MIN_INTERVAL_SEC:-}" || -n "${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-}" ]]; then
  echo "[$(ts)] llm_call_min_interval=${LLM_CALL_MIN_INTERVAL_SEC:-0} rate_limit_cooldown=${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-0}"
fi
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
    NO_SILENT_FALLBACK="$NO_SILENT_FALLBACK" \
    EVAL_TRACE_CALLS=1 \
    EVAL_TRACE_EVENTS=1 \
    EVAL_TRACE_MAX_CHARS="${EVAL_TRACE_MAX_CHARS:-1200}" \
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
errors = []
fallbacks = []
parse_fallbacks = []
think_tags = []
with open(path) as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        pred = row.get("predicted_answer")
        if pred is None or str(pred).strip() == "":
            bad.append(str(row.get("label") or row.get("idx") or line_no))
        row_id = str(row.get("label") or row.get("idx") or line_no)
        if row.get("error"):
            errors.append(row_id)
        routed_to = str(row.get("routed_to") or "")
        if "fallback" in routed_to.lower():
            fallbacks.append(f"{row_id}:routed_to={routed_to}")
        for key, value in row.items():
            if (key.endswith("_fallback") or key.endswith("_used_fallback")) and value:
                fallbacks.append(f"{row_id}:{key}={value}")
        for key in ("hyde_parse_ok", "snap_hyre_parse_ok", "snap_hyde_2call_parse_ok", "choice_hyre_parse_ok", "route_parse_ok", "passage_parse_ok", "adaptive_parse_ok"):
            if row.get(key) is False:
                parse_fallbacks.append(f"{row_id}:{key}")
        if "<think>" in str(row.get("final_answer") or "").lower():
            think_tags.append(row_id)
if bad:
    raise SystemExit("missing predicted_answer rows: " + ",".join(bad[:10]))
if errors:
    raise SystemExit("error rows: " + ",".join(errors[:10]))
if fallbacks:
    raise SystemExit("fallback marker rows: " + " | ".join(fallbacks[:10]))
if parse_fallbacks:
    raise SystemExit("parse fallback rows: " + ",".join(parse_fallbacks[:10]))
if think_tags:
    raise SystemExit("unclosed think tag rows: " + ",".join(think_tags[:10]))
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
