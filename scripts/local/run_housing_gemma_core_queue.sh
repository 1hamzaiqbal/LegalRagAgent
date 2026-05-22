#!/usr/bin/env bash
# Build and run Gemma 26B HousingQA state-filter core HyDE/Snap-HyRE rows.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

PROVIDER="${PROVIDER:-or-gemma4-26b}"
MODEL_LABEL="${MODEL_LABEL:-$PROVIDER}"
export PROVIDER MODEL_LABEL
UV="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
QUESTIONS="${QUESTIONS:-full}"
SEED="${SEED:-42}"
MAX_K="${MAX_K:-10}"
RETRIEVAL_K="${RETRIEVAL_K:-5}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
GEN_ROOT="${GEN_ROOT:-$ROOT/caches/hyre/full}"
RET_ROOT="${RET_ROOT:-$ROOT/caches/retrieval/full}"
DOC_ROOT="${DOC_ROOT:-$ROOT/caches/retrieval_doc/full}"
# Use ${VAR-default} instead of ${VAR:-default} so callers can intentionally
# pass WAIT_PATTERNS="" to skip stale tmux-session waits.
WAIT_PATTERNS="${WAIT_PATTERNS-housing_gemma_rag_simple housing_gemma_followup_queue housing_gemma_exemplar_slot_queue housing_gemma_exemplar_gate_retry}"
LAUNCH_LOCK_DIR="${GEMMA_CORE_QUEUE_LOCK_DIR:-/tmp/housing_gemma_core_queue.lock}"

mkdir -p "$LOG_DIR" "$GEN_ROOT" "$RET_ROOT" "$DOC_ROOT"

echo "[$(ts)] housing Gemma core queue start provider=$PROVIDER model_label=$MODEL_LABEL"
echo "[$(ts)] waiting for active Gemma queues: $WAIT_PATTERNS"

if ! mkdir "$LAUNCH_LOCK_DIR" 2>/dev/null; then
  echo "[$(ts)] ERROR: another Housing Gemma core queue appears to hold $LAUNCH_LOCK_DIR" >&2
  if [[ -f "$LAUNCH_LOCK_DIR/metadata" ]]; then
    echo "[$(ts)] Existing lock metadata:" >&2
    sed 's/^/[lock] /' "$LAUNCH_LOCK_DIR/metadata" >&2
  fi
  echo "[$(ts)] Remove the lock only after verifying no matching Gemma cache/answer queue is still running." >&2
  exit 11
fi
cleanup_lock() {
  rm -rf "$LAUNCH_LOCK_DIR"
}
trap cleanup_lock EXIT
trap 'cleanup_lock; exit 130' INT
trap 'cleanup_lock; exit 143' TERM
{
  echo "pid=$$"
  echo "created_utc=$(ts)"
  echo "cwd=$ROOT"
  echo "provider=$PROVIDER"
  echo "model_label=$MODEL_LABEL"
  echo "questions=$QUESTIONS"
  echo "seed=$SEED"
  echo "max_k=$MAX_K"
  echo "wait_patterns=$WAIT_PATTERNS"
  echo "command=$0 $*"
} > "$LAUNCH_LOCK_DIR/metadata"
echo "[$(ts)] acquired Gemma core queue lock: $LAUNCH_LOCK_DIR"

for pattern in $WAIT_PATTERNS; do
  while tmux ls 2>/dev/null | grep -q "$pattern"; do
    sleep 60
  done
done

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NO_SILENT_FALLBACK=1
export LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}"
export EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}"
export EVAL_GENERATION_FORMAT_RETRY="${EVAL_GENERATION_FORMAT_RETRY:-1}"
export EVAL_FINAL_FORMAT_RETRY="${EVAL_FINAL_FORMAT_RETRY:-1}"
export OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"
export PYTHONUNBUFFERED=1
export EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
RUN_OPENROUTER_CHAT_PREFLIGHT="${RUN_OPENROUTER_CHAT_PREFLIGHT:-1}"
OPENROUTER_FREE_SUFFIX_ARG=()
if [[ "${OPENROUTER_ALLOW_FREE_SUFFIX:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  OPENROUTER_FREE_SUFFIX_ARG=(--allow-openrouter-free-suffix)
fi

python3 scripts/check_expected_provider_model.py \
  --provider "$PROVIDER" \
  --expected-model "$EXPECTED_GEMMA_MODEL" \
  --expected-label "or-gemma4-26b" \
  "${OPENROUTER_FREE_SUFFIX_ARG[@]}"
python3 scripts/check_openrouter_key_status.py --min-limit-remaining "${OPENROUTER_MIN_LIMIT_REMAINING:-0.01}"
if [[ "$RUN_OPENROUTER_CHAT_PREFLIGHT" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  python3 scripts/check_openrouter_chat_route.py \
    --provider "$PROVIDER" \
    --expected-model "$EXPECTED_GEMMA_MODEL" \
    --provider-only "$OPENROUTER_PROVIDER_ONLY" \
    "${OPENROUTER_FREE_SUFFIX_ARG[@]}"
fi

build_and_run_mode() {
  local mode="$1"
  local query_type="$2"
  local label_prefix="$3"
  local gen="$GEN_ROOT/housing_q${QUESTIONS}_seed${SEED}_${MODEL_LABEL}_${mode}.jsonl"
  local ret="$RET_ROOT/housing_q${QUESTIONS}_seed${SEED}_statefilter_${MODEL_LABEL}_${mode}_k${MAX_K}.jsonl"
  local doc="$DOC_ROOT/housing_q${QUESTIONS}_seed${SEED}_statefilter_${MODEL_LABEL}_${mode}_k${MAX_K}_doc_cache.jsonl"

  echo
  echo "[$(ts)] generation cache mode=$mode out=$gen route=OPENROUTER_PROVIDER_ONLY=$OPENROUTER_PROVIDER_ONLY"
  LLM_PROVIDER="$PROVIDER" \
  "$UV" run python scripts/build_generation_cache.py \
    --mode "$mode" \
    --provider "$PROVIDER" \
    --dataset housing \
    --questions "$QUESTIONS" \
    --seed "$SEED" \
    --tag "housing-gemma-core-${mode}" \
    --out "$gen" \
    --resume \
    --trace-calls \
    --trace-events

  "$UV" run python - "$gen" "$mode" <<'PY'
import json
import sys

path, mode = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(path, errors="ignore") if line.strip()]

def truthy_fallback(row):
    falsey_strings = {"", "0", "false", "no", "none", "null", "[]", "{}"}
    for key, value in row.items():
        if "fallback" not in str(key).lower():
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip().lower() not in falsey_strings:
                return True
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
            continue
        if value:
            return True
    return False

errors = [r for r in rows if r.get("error")]
missing = [r for r in rows if not r.get("hyde_passage")]
fallbacks = [r for r in rows if truthy_fallback(r)]
parse_fail = [r for r in rows if r.get("hyde_parse_ok") is False or (mode == "snap_hyre" and r.get("snap_hyre_parse_ok") is False)]
missing_snap = [r for r in rows if mode == "snap_hyre" and not r.get("snap_letter")]
artifacts = [r for r in rows if r.get("hyde_contains_answer_artifact") is True]
print(f"[postcheck] generation rows={len(rows)} errors={len(errors)} missing_hyde={len(missing)} fallbacks={len(fallbacks)} parse_fail={len(parse_fail)} missing_snap={len(missing_snap)} answer_artifacts={len(artifacts)}")
if len(rows) != 6853:
    raise SystemExit(f"generation cache row-count mismatch rows={len(rows)} expected=6853")
for name, bad in [("errors", errors), ("missing_hyde", missing), ("fallbacks", fallbacks), ("parse_fail", parse_fail), ("missing_snap", missing_snap), ("answer_artifacts", artifacts)]:
    if bad:
        raise SystemExit(f"{name}: " + ",".join(str(r.get("label")) for r in bad[:10]))
PY

  echo "[$(ts)] state-filter retrieval cache mode=$mode out=$ret"
  "$UV" run python scripts/build_retrieval_cache.py \
    --dataset housing \
    --questions "$QUESTIONS" \
    --seed "$SEED" \
    --query-type "$query_type" \
    --label-prefix "$label_prefix" \
    --hyre-cache-path "$gen" \
    --expected-provider "$PROVIDER" \
    --max-k "$MAX_K" \
    --housing-state-filter \
    --out "$ret" \
    --resume \
    --progress-interval 100

  echo "[$(ts)] document cache mode=$mode out=$doc"
  "$UV" run python scripts/build_retrieval_doc_cache.py \
    --retrieval-cache "$ret" \
    --include-effective \
    --out "$doc" \
    --resume \
    --strict \
    --batch-size 500

  RETRIEVAL_DOC_CACHE_PATH="$doc" RETRIEVAL_DOC_CACHE_STRICT=1 \
  "$UV" run python scripts/smoke_retrieval_cache_hydration.py \
    --cache "$ret" \
    --dataset housing \
    --label-prefix "$label_prefix" \
    --questions "$QUESTIONS" \
    --seed "$SEED" \
    --retrieval-k "$RETRIEVAL_K" \
    --limit 20 \
    --housing-state-filter \
    --require-doc-cache

  echo "[$(ts)] answer row mode=$mode"
  MODE="$mode" \
  PROVIDER="$PROVIDER" \
  MODEL_LABEL="$MODEL_LABEL" \
  QUESTIONS="$QUESTIONS" \
  SEED="$SEED" \
  RETRIEVAL_K="$RETRIEVAL_K" \
  CACHE_SCOPE="q${QUESTIONS}_seed${SEED}_statefilter" \
  HYRE_CACHE_ROOT="$GEN_ROOT" \
  RETRIEVAL_CACHE_ROOT="$RET_ROOT" \
  RETRIEVAL_CACHE="$ret" \
  DOC_CACHE="$doc" \
  scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh
}

build_and_run_mode rag_hyde hyde_cache hyde
build_and_run_mode snap_hyre hyre_cache snap_hyre

echo
echo "[$(ts)] housing Gemma core queue complete"
