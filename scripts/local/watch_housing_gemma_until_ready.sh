#!/usr/bin/env bash
# Poll the exact HousingQA Gemma OpenRouter route and optionally launch once ready.
#
# Default mode is non-launching. Set LAUNCH_ON_READY=1 to run the canonical
# after-key-reset continuation when the exact-model/budget/chat-route preflight
# succeeds. This helper is meant for a tmux/screen shell, not for a Codex
# short-lived exec session.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

export PROVIDER="${PROVIDER:-or-gemma4-26b}"
export MODEL_LABEL="${MODEL_LABEL:-or-gemma4-26b}"
export EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
export OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"
export OPENROUTER_MIN_LIMIT_REMAINING="${OPENROUTER_MIN_LIMIT_REMAINING:-0.01}"
export RUN_OPENROUTER_CHAT_PREFLIGHT="${RUN_OPENROUTER_CHAT_PREFLIGHT:-1}"
export NO_SILENT_FALLBACK=1

INTERVAL_SECONDS="${INTERVAL_SECONDS:-900}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"
LAUNCH_ON_READY="${LAUNCH_ON_READY:-0}"
RUN_FINAL_GATE_AFTER_LAUNCH="${RUN_FINAL_GATE_AFTER_LAUNCH:-1}"
LOG_DIR="$ROOT/logs/monitors"
LOCK_ROOT="${LOCK_ROOT:-$LOG_DIR/locks}"
WATCH_LOCK_DIR="${WATCH_LOCK_DIR:-$LOCK_ROOT/housing_gemma_budget_watch.lock}"
export LAUNCH_LOCK_DIR="${LAUNCH_LOCK_DIR:-$LOCK_ROOT/housing_gemma_after_key_reset.lock}"
export RAG_SIMPLE_RESUME_LOCK_DIR="${RAG_SIMPLE_RESUME_LOCK_DIR:-$LOCK_ROOT/housing_gemma_rag_simple_resume.lock}"
export GEMMA_CORE_QUEUE_LOCK_DIR="${GEMMA_CORE_QUEUE_LOCK_DIR:-$LOCK_ROOT/housing_gemma_core_queue.lock}"
export EXEMPLAR_FULL_LOCK_DIR="${EXEMPLAR_FULL_LOCK_DIR:-$LOCK_ROOT/housing_gemma_exemplar_full.lock}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/housing_gemma_budget_watch_$(date -u +%Y%m%d_%H%M%S).log}"

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || (( INTERVAL_SECONDS < 30 )); then
  echo "INTERVAL_SECONDS must be an integer >= 30" >&2
  exit 2
fi
if ! [[ "$MAX_ATTEMPTS" =~ ^[0-9]+$ ]]; then
  echo "MAX_ATTEMPTS must be an integer >= 0" >&2
  exit 2
fi

mkdir -p "$LOG_DIR" "$LOCK_ROOT"
if [[ -d "$WATCH_LOCK_DIR" && -f "$WATCH_LOCK_DIR/metadata" ]]; then
  existing_pid="$(awk -F= '$1 == "pid" {print $2; exit}' "$WATCH_LOCK_DIR/metadata" 2>/dev/null || true)"
  if [[ "$existing_pid" =~ ^[0-9]+$ ]] && ! kill -0 "$existing_pid" 2>/dev/null; then
    echo "[$(ts)] removing stale Housing Gemma budget watcher lock for pid=$existing_pid: $WATCH_LOCK_DIR" >&2
    rm -rf "$WATCH_LOCK_DIR"
  fi
fi
if ! mkdir "$WATCH_LOCK_DIR" 2>/dev/null; then
  echo "[$(ts)] ERROR: another Housing Gemma budget watcher holds $WATCH_LOCK_DIR" >&2
  if [[ -f "$WATCH_LOCK_DIR/metadata" ]]; then
    sed 's/^/[lock] /' "$WATCH_LOCK_DIR/metadata" >&2
  fi
  exit 11
fi
cleanup_lock() {
  rm -rf "$WATCH_LOCK_DIR"
}
finish() {
  local code="${1:-0}"
  cleanup_lock
  exit "$code"
}
trap cleanup_lock EXIT
trap 'finish 130' INT
trap 'finish 143' TERM
{
  echo "pid=$$"
  echo "created_utc=$(ts)"
  echo "cwd=$ROOT"
  echo "provider=$PROVIDER"
  echo "model_label=$MODEL_LABEL"
  echo "openrouter_provider_only=$OPENROUTER_PROVIDER_ONLY"
  echo "openrouter_min_limit_remaining=$OPENROUTER_MIN_LIMIT_REMAINING"
  echo "interval_seconds=$INTERVAL_SECONDS"
  echo "max_attempts=$MAX_ATTEMPTS"
  echo "launch_on_ready=$LAUNCH_ON_READY"
  echo "launch_lock_dir=$LAUNCH_LOCK_DIR"
  echo "rag_simple_resume_lock_dir=$RAG_SIMPLE_RESUME_LOCK_DIR"
  echo "gemma_core_queue_lock_dir=$GEMMA_CORE_QUEUE_LOCK_DIR"
  echo "exemplar_full_lock_dir=$EXEMPLAR_FULL_LOCK_DIR"
  echo "log_file=$LOG_FILE"
} > "$WATCH_LOCK_DIR/metadata"

echo "[$(ts)] Housing Gemma budget watcher started" | tee -a "$LOG_FILE"
echo "[$(ts)] provider=$PROVIDER model_label=$MODEL_LABEL route=OPENROUTER_PROVIDER_ONLY=$OPENROUTER_PROVIDER_ONLY launch_on_ready=$LAUNCH_ON_READY" | tee -a "$LOG_FILE"
echo "[$(ts)] interval_seconds=$INTERVAL_SECONDS max_attempts=$MAX_ATTEMPTS log=$LOG_FILE" | tee -a "$LOG_FILE"

attempt=0
while true; do
  if [[ ! -d "$WATCH_LOCK_DIR" ]]; then
    echo "[$(ts)] watcher lock disappeared; exiting to avoid untracked auto-launch" | tee -a "$LOG_FILE"
    finish 13
  fi
  if [[ -f "$WATCH_LOCK_DIR/metadata" ]]; then
    recorded_pid="$(awk -F= '$1 == "pid" {print $2; exit}' "$WATCH_LOCK_DIR/metadata")"
    if [[ "$recorded_pid" != "$$" ]]; then
      echo "[$(ts)] watcher lock pid mismatch (recorded=${recorded_pid:-missing}, self=$$); exiting to avoid duplicate auto-launch" | tee -a "$LOG_FILE"
      finish 14
    fi
  fi

  attempt=$((attempt + 1))
  echo | tee -a "$LOG_FILE"
  echo "[$(ts)] preflight attempt $attempt" | tee -a "$LOG_FILE"

  launch_lock_found=0
  for lock_dir in \
    "/tmp/housing_gemma_after_key_reset.lock" \
    "${LAUNCH_LOCK_DIR:-$LOCK_ROOT/housing_gemma_after_key_reset.lock}" \
    "/tmp/housing_gemma_rag_simple_resume.lock" \
    "${RAG_SIMPLE_RESUME_LOCK_DIR:-$LOCK_ROOT/housing_gemma_rag_simple_resume.lock}" \
    "/tmp/housing_gemma_core_queue.lock" \
    "${GEMMA_CORE_QUEUE_LOCK_DIR:-$LOCK_ROOT/housing_gemma_core_queue.lock}" \
    "/tmp/housing_gemma_exemplar_full.lock" \
    "${EXEMPLAR_FULL_LOCK_DIR:-$LOCK_ROOT/housing_gemma_exemplar_full.lock}"; do
    if [[ -d "$lock_dir" ]]; then
      launch_lock_found=1
      echo "[$(ts)] launch lock present; not starting another job: $lock_dir" | tee -a "$LOG_FILE"
      if [[ -f "$lock_dir/metadata" ]]; then
        sed 's/^/[lock] /' "$lock_dir/metadata" | tee -a "$LOG_FILE"
      fi
    fi
  done
  if (( launch_lock_found > 0 )); then
    echo "[$(ts)] exiting because a Housing Gemma launch may already be active" | tee -a "$LOG_FILE"
    finish 12
  fi

  if PREFLIGHT_ONLY=1 scripts/local/run_housing_gemma_after_key_reset.sh >> "$LOG_FILE" 2>&1; then
    echo "[$(ts)] OpenRouter route/budget preflight passed" | tee -a "$LOG_FILE"
    if truthy "$LAUNCH_ON_READY"; then
      echo "[$(ts)] running read-only network readiness gate before launch" | tee -a "$LOG_FILE"
      CHECK_NETWORK=1 scripts/local/check_housing_gemma_readiness.sh >> "$LOG_FILE" 2>&1
      echo "[$(ts)] launching canonical Housing Gemma continuation" | tee -a "$LOG_FILE"
      scripts/local/run_housing_gemma_after_key_reset.sh >> "$LOG_FILE" 2>&1
      if truthy "$RUN_FINAL_GATE_AFTER_LAUNCH"; then
        echo "[$(ts)] running strict Housing completion gate after launch" | tee -a "$LOG_FILE"
        scripts/local/verify_housing_statefilter_goal_complete.sh >> "$LOG_FILE" 2>&1
      fi
      echo "[$(ts)] watcher launch path finished" | tee -a "$LOG_FILE"
    else
      echo "[$(ts)] LAUNCH_ON_READY=0, exiting without launching rows" | tee -a "$LOG_FILE"
    fi
    finish 0
  fi

  echo "[$(ts)] preflight still blocked; last 12 log lines:" | tee -a "$LOG_FILE"
  tail -n 12 "$LOG_FILE"

  if (( MAX_ATTEMPTS > 0 && attempt >= MAX_ATTEMPTS )); then
    echo "[$(ts)] max attempts reached without ready route/budget" | tee -a "$LOG_FILE"
    finish 20
  fi
  echo "[$(ts)] sleeping ${INTERVAL_SECONDS}s" | tee -a "$LOG_FILE"
  sleep "$INTERVAL_SECONDS"
done
