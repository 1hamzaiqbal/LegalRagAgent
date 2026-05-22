#!/usr/bin/env bash
# Resume the HousingQA Gemma 26B state-filtered rag_simple row after the
# 2026-05-21 OpenRouter key-limit failure.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

export UV="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export PROVIDER="${PROVIDER:-or-gemma4-26b}"
export MODEL_LABEL="${MODEL_LABEL:-or-gemma4-26b}"
export MODE=rag_simple
export QUESTIONS=full
export SEED=42
export RETRIEVAL_K=5
export CACHE_SCOPE=qfull_seed42_statefilter
export OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"
export NO_SILENT_FALLBACK=1
export EVAL_FINAL_FORMAT_RETRY=1
export EVAL_GENERATION_FORMAT_RETRY=1
export LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}"
export EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}"
export LLM_CALL_MIN_INTERVAL_SEC="${LLM_CALL_MIN_INTERVAL_SEC:-2.0}"
export LLM_CALL_RATE_LIMIT_COOLDOWN_SEC="${LLM_CALL_RATE_LIMIT_COOLDOWN_SEC:-8.0}"
export PYTHONUNBUFFERED=1
export EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
RUN_OPENROUTER_CHAT_PREFLIGHT="${RUN_OPENROUTER_CHAT_PREFLIGHT:-1}"
OPENROUTER_FREE_SUFFIX_ARG=()
if [[ "${OPENROUTER_ALLOW_FREE_SUFFIX:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  OPENROUTER_FREE_SUFFIX_ARG=(--allow-openrouter-free-suffix)
fi
RUN_PARALLEL="${RUN_PARALLEL:-1}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
RAG_SIMPLE_RANGES="${RAG_SIMPLE_RANGES:-}"

acquire_launch_lock() {
  [[ "$VERIFY_ONLY" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]] && {
    echo "[$(ts)] verify-only mode skips rag_simple resume launch lock"
    return 0
  }

  LAUNCH_LOCK_DIR="${RAG_SIMPLE_RESUME_LOCK_DIR:-/tmp/housing_gemma_rag_simple_resume.lock}"
  if ! mkdir "$LAUNCH_LOCK_DIR" 2>/dev/null; then
    echo "[$(ts)] ERROR: another Housing Gemma rag_simple resume appears to hold $LAUNCH_LOCK_DIR" >&2
    if [[ -f "$LAUNCH_LOCK_DIR/metadata" ]]; then
      echo "[$(ts)] Existing lock metadata:" >&2
      sed 's/^/[lock] /' "$LAUNCH_LOCK_DIR/metadata" >&2
    fi
    echo "[$(ts)] Remove the lock only after verifying no matching resume chunks are still running." >&2
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
    echo "openrouter_provider_only=$OPENROUTER_PROVIDER_ONLY"
    echo "run_parallel=$RUN_PARALLEL"
    echo "command=$0 $*"
  } > "$LAUNCH_LOCK_DIR/metadata"
  echo "[$(ts)] acquired rag_simple resume lock: $LAUNCH_LOCK_DIR"
}

run_chunk() {
  local start="$1"
  local end="${2:-}"
  local suffix="${start}_${end:-end}"
  local out="$LOG_DIR/run_housing_statefilter_gemma_rag_simple_resume_${suffix}_$(date -u +%Y%m%d_%H%M%S).out"

  echo "[$(ts)] resume Gemma rag_simple sample=${start}:${end:-end} out=$out"
  SAMPLE_START="$start" SAMPLE_END="$end" \
    scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh > "$out" 2>&1
  echo "[$(ts)] chunk complete sample=${start}:${end:-end} out=$out"
}

build_ranges() {
  RANGES_START=()
  RANGES_END=()
  if [[ -n "${RAG_SIMPLE_RANGES//[[:space:]]/}" ]]; then
    for range in $RAG_SIMPLE_RANGES; do
      if [[ ! "$range" =~ ^[0-9]+:[0-9]*$ ]]; then
        echo "invalid RAG_SIMPLE_RANGES entry '$range'; expected start:end or start:" >&2
        exit 2
      fi
      RANGES_START+=("${range%%:*}")
      RANGES_END+=("${range#*:}")
    done
    return
  fi

  # Historical safe ranges that supersede the three original failed-closed
  # records. Set RAG_SIMPLE_RANGES after checking the gap helper to target only
  # the remaining holes once partial recovery shards have finished.
  RANGES_START=(3478 4634 5796)
  RANGES_END=(4530 5690 "")
}

verify_resume_offsets() {
  python3 - <<'PY'
import csv
from pathlib import Path

path = Path("datasets/housing_qa/questions.csv")
rows = list(csv.DictReader(path.open(newline="")))
expected = [
    (3478, "hqa_Nebraska_2941"),
    (4634, "hqa_Ohio_6341"),
    (5796, "hqa_Texas_4530"),
]
failures = []
for pos, expected_label in expected:
    try:
        row = rows[pos]
    except IndexError:
        failures.append(f"{pos}:out-of-range")
        continue
    actual = f"hqa_{row.get('state')}_{row.get('idx')}"
    if actual != expected_label:
        failures.append(f"{pos}:expected {expected_label}, found {actual}")
if failures:
    raise SystemExit("HousingQA resume offsets no longer match failed labels: " + "; ".join(failures))
print("resume offsets verified: " + ", ".join(f"{pos}->{label}" for pos, label in expected))
PY
}

echo "[$(ts)] resume HousingQA Gemma rag_simple after key reset"
echo "[$(ts)] provider=$PROVIDER model_label=$MODEL_LABEL route=OPENROUTER_PROVIDER_ONLY=$OPENROUTER_PROVIDER_ONLY"
echo "[$(ts)] strict guards: NO_SILENT_FALLBACK=$NO_SILENT_FALLBACK CACHE_SCOPE=$CACHE_SCOPE"
echo "[$(ts)] run_parallel=$RUN_PARALLEL"
echo "[$(ts)] verify_only=$VERIFY_ONLY"
if [[ -n "${RAG_SIMPLE_RANGES//[[:space:]]/}" ]]; then
  echo "[$(ts)] requested RAG_SIMPLE_RANGES=$RAG_SIMPLE_RANGES"
fi
verify_resume_offsets
python3 scripts/check_expected_provider_model.py \
  --provider "$PROVIDER" \
  --expected-model "$EXPECTED_GEMMA_MODEL" \
  --expected-label "or-gemma4-26b" \
  "${OPENROUTER_FREE_SUFFIX_ARG[@]}"
build_ranges
echo "[$(ts)] resume ranges:"
for i in "${!RANGES_START[@]}"; do
  echo "  ${RANGES_START[$i]}:${RANGES_END[$i]:-end}"
done
acquire_launch_lock

if [[ "$VERIFY_ONLY" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "[$(ts)] verify-only passed; exiting before OpenRouter key preflight or row launch"
  exit 0
fi

python3 scripts/check_openrouter_key_status.py --min-limit-remaining "${OPENROUTER_MIN_LIMIT_REMAINING:-0.01}"
if [[ "$RUN_OPENROUTER_CHAT_PREFLIGHT" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  python3 scripts/check_openrouter_chat_route.py \
    --provider "$PROVIDER" \
    --expected-model "$EXPECTED_GEMMA_MODEL" \
    --provider-only "$OPENROUTER_PROVIDER_ONLY" \
    "${OPENROUTER_FREE_SUFFIX_ARG[@]}"
fi

# The default starts intentionally include the failed rows so a later
# merge_detail_logs.py --on-duplicate last supersedes the failed-closed records:
# - hqa_Nebraska_2941 from the [3368:4530] chunk
# - hqa_Ohio_6341 from the [4530:5690] chunk
# - hqa_Texas_4530 from the [5690:] chunk.
# For post-recovery cleanup, prefer RAG_SIMPLE_RANGES with the output of
# scripts/report_housing_gemma_rag_simple_gaps.py.
if [[ "$RUN_PARALLEL" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  pids=()
  for i in "${!RANGES_START[@]}"; do
    run_chunk "${RANGES_START[$i]}" "${RANGES_END[$i]}" & pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "[$(ts)] at least one resume chunk failed; inspect logs/run_housing_statefilter_gemma_rag_simple_resume_*.out" >&2
    exit 1
  fi
else
  for i in "${!RANGES_START[@]}"; do
    run_chunk "${RANGES_START[$i]}" "${RANGES_END[$i]}"
  done
fi

echo "[$(ts)] resume chunks complete; merge/audit/signoff is still required"
