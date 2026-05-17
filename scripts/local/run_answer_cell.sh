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
ENV_LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-}"
LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}"
EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}"
EVAL_FINAL_FORMAT_RETRY="${EVAL_FINAL_FORMAT_RETRY:-1}"
EVAL_GENERATION_FORMAT_RETRY="${EVAL_GENERATION_FORMAT_RETRY:-1}"
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
  echo "LLM_MAX_COMPLETION_TOKENS=$LLM_MAX_COMPLETION_TOKENS is below EVAL_MIN_COMPLETION_TOKENS=$EVAL_MIN_COMPLETION_TOKENS; refusing truncation-prone answer run" >&2
  exit 2
fi
NO_SILENT_FALLBACK="${NO_SILENT_FALLBACK:-1}"
case "${NO_SILENT_FALLBACK,,}" in
  1|true|yes|on) ;;
  *) echo "NO_SILENT_FALLBACK must be enabled for answer runs, got $NO_SILENT_FALLBACK" >&2; exit 2 ;;
esac

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
export EVAL_FINAL_FORMAT_RETRY
export EVAL_GENERATION_FORMAT_RETRY
export NO_SILENT_FALLBACK
if [[ "$PROVIDER" == or-* ]]; then
  export LLM_CALL_MIN_INTERVAL_SEC="${LLM_CALL_MIN_INTERVAL_SEC:-2.0}"
  export LLM_CALL_RATE_LIMIT_COOLDOWN_SEC="${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-8.0}"
fi
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
echo "[$(ts)] no_silent_fallback=$NO_SILENT_FALLBACK"
echo "[$(ts)] llm_max_completion_tokens=$LLM_MAX_COMPLETION_TOKENS"
echo "[$(ts)] eval_min_completion_tokens=$EVAL_MIN_COMPLETION_TOKENS"
echo "[$(ts)] eval_final_format_retry=$EVAL_FINAL_FORMAT_RETRY"
echo "[$(ts)] eval_generation_format_retry=$EVAL_GENERATION_FORMAT_RETRY"
if [[ -n "${LLM_CALL_MIN_INTERVAL_SEC:-}" || -n "${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-}" ]]; then
  echo "[$(ts)] llm_call_min_interval=${LLM_CALL_MIN_INTERVAL_SEC:-0} rate_limit_cooldown=${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-0}"
fi
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
  NO_SILENT_FALLBACK="$NO_SILENT_FALLBACK" \
  EVAL_TRACE_CALLS=1 \
  EVAL_TRACE_EVENTS=1 \
  EVAL_TRACE_MAX_CHARS="${EVAL_TRACE_MAX_CHARS:-1200}" \
  "${eval_cmd[@]}"
  status=$?
  set -e

  latest_log="$(find logs -maxdepth 1 -name "eval_${mode}_${PROVIDER}_*_${DATASET}_*${tag}*_detail.jsonl" -print | sort | tail -n 1)"
  if [[ -z "$latest_log" ]]; then
    echo "[$(ts)] ERROR: no detail log found for dataset=$DATASET provider=$PROVIDER mode=$mode"
    status=1
  else
    "$UV" run python scripts/analyze_detail_flags.py "$latest_log" || status=1
    "$UV" run python - "$latest_log" "$mode" <<'PY' || status=1
import json
import os
import re
import sys

path = sys.argv[1]
mode = sys.argv[2]
bad = []
errors = []
fallbacks = []
parse_fallbacks = []
think_tags = []
missing_answer_marker = []
long_answers = []
near_cap_outputs = []
missing_retrieval_cache = []
missing_hyre_cache = []
oracle_missing = []
cache_required_modes = {"rag_simple", "rag_hyde", "snap_hyre", "golden_plus_neighbors"}
hyre_required_modes = {"rag_hyde", "snap_hyre"}
oracle_modes = {"golden_passage", "golden_plus_neighbors"}
use_caches = os.getenv("USE_CACHES", "1") == "1"
max_final_answer_chars = int(os.getenv("EVAL_MAX_FINAL_ANSWER_CHARS", "20000"))
max_completion_tokens = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "0") or "0")
output_token_margin = int(os.getenv("EVAL_OUTPUT_TOKEN_MARGIN", "16"))
answer_marker = re.compile(r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:")

def has_required_final_line(row, text):
    dataset = str(row.get("dataset") or "")
    pred = str(row.get("predicted_answer") or "").strip()
    if dataset == "housing":
        if pred.lower() == "yes":
            target = "Answer: Yes"
        elif pred.lower() == "no":
            target = "Answer: No"
        else:
            return False
    elif dataset in {"barexam", "housing", "casehold", "legalbench_scalr"}:
        pred = pred.upper()
        if pred not in {"A", "B", "C", "D", "E"}:
            return False
        target = f"Answer: ({pred})"
    else:
        return True
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return bool(lines) and lines[-1] == target

with open(path) as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row.get("label") or row.get("idx") or line_no)
        pred = row.get("predicted_answer")
        if pred is None or str(pred).strip() == "":
            bad.append(row_id)
        if row.get("error"):
            errors.append(row_id)
        routed_to = str(row.get("routed_to") or "")
        if "fallback" in routed_to.lower():
            fallbacks.append(f"{row_id}:routed_to={routed_to}")
        for key, value in row.items():
            if (key.endswith("_fallback") or key.endswith("_used_fallback")) and value:
                fallbacks.append(f"{row_id}:{key}={value}")
        for key in ("hyde_parse_ok", "snap_hyre_parse_ok", "snap_hyde_2call_parse_ok", "choice_hyre_parse_ok", "rewrite_parse_ok", "route_parse_ok", "passage_parse_ok", "adaptive_parse_ok"):
            if row.get(key) is False:
                parse_fallbacks.append(f"{row_id}:{key}")
        final_answer = str(row.get("final_answer") or "")
        if "<think>" in final_answer.lower():
            think_tags.append(row_id)
        dataset = str(row.get("dataset") or "")
        if dataset in {"barexam", "housing", "casehold", "legalbench_scalr"} and not answer_marker.search(final_answer):
            missing_answer_marker.append(row_id)
        elif dataset in {"barexam", "housing", "casehold", "legalbench_scalr"} and not has_required_final_line(row, final_answer):
            missing_answer_marker.append(f"{row_id}:missing_required_final_answer_line")
        if max_final_answer_chars > 0 and len(final_answer) > max_final_answer_chars:
            long_answers.append(f"{row_id}:{len(final_answer)}")
        if (
            max_completion_tokens > 0
            and int(row.get("llm_calls") or 0) <= 1
            and int(row.get("output_tokens") or 0) >= max(1, max_completion_tokens - output_token_margin)
        ):
            near_cap_outputs.append(f"{row_id}:{row.get('output_tokens')}")
        retry_output_tokens = int(row.get("answer_format_retry_output_tokens") or 0)
        if (
            max_completion_tokens > 0
            and retry_output_tokens >= max(1, max_completion_tokens - output_token_margin)
        ):
            near_cap_outputs.append(f"{row_id}:retry={retry_output_tokens}")
        if use_caches and mode in cache_required_modes and row.get("retrieval_cache_hit") is not True:
            missing_retrieval_cache.append(row_id)
        if use_caches and mode in hyre_required_modes and row.get("hyre_cache_hit") is not True:
            missing_hyre_cache.append(row_id)
        if mode in oracle_modes and (row.get("gold_retrieved") is not True or not row.get("evidence_store")):
            oracle_missing.append(row_id)
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
if missing_answer_marker:
    raise SystemExit("missing explicit Answer marker rows: " + ",".join(missing_answer_marker[:10]))
if long_answers:
    raise SystemExit("long final answer rows: " + ",".join(long_answers[:10]))
if near_cap_outputs:
    raise SystemExit("near-cap output token rows: " + ",".join(near_cap_outputs[:10]))
if missing_retrieval_cache:
    raise SystemExit("retrieval cache not hit rows: " + ",".join(missing_retrieval_cache[:10]))
if missing_hyre_cache:
    raise SystemExit("HyRE generation cache not hit rows: " + ",".join(missing_hyre_cache[:10]))
if oracle_missing:
    raise SystemExit("oracle evidence missing rows: " + ",".join(oracle_missing[:10]))
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
