#!/usr/bin/env bash
# Final gate for the HousingQA state-filter goal.
#
# This is intentionally stricter than the readiness check. It should pass only
# after all required Gemma rows have finished, clean signoff rows exist or can be
# appended by the finalizer, current_status.md is refreshed, and the focused
# 9-cell audit succeeds without --allow-incomplete.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

export UV="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
ALLOW_EXISTING_LOCKS="${ALLOW_EXISTING_LOCKS:-0}"
RUN_FINALIZER="${RUN_FINALIZER:-1}"
REQUIRE_FULL_EXEMPLAR="${REQUIRE_FULL_EXEMPLAR:-1}"
LOCK_ROOT="${LOCK_ROOT:-$ROOT/logs/monitors/locks}"
HOUSING_LOCK_DIRS=(
  "${LAUNCH_LOCK_DIR:-/tmp/housing_gemma_after_key_reset.lock}"
  "$LOCK_ROOT/housing_gemma_after_key_reset.lock"
  "${RAG_SIMPLE_RESUME_LOCK_DIR:-/tmp/housing_gemma_rag_simple_resume.lock}"
  "$LOCK_ROOT/housing_gemma_rag_simple_resume.lock"
  "${GEMMA_CORE_QUEUE_LOCK_DIR:-/tmp/housing_gemma_core_queue.lock}"
  "$LOCK_ROOT/housing_gemma_core_queue.lock"
  "${EXEMPLAR_FULL_LOCK_DIR:-/tmp/housing_gemma_exemplar_full.lock}"
  "$LOCK_ROOT/housing_gemma_exemplar_full.lock"
  "${WATCH_LOCK_DIR:-/tmp/housing_gemma_budget_watch.lock}"
  "$LOCK_ROOT/housing_gemma_budget_watch.lock"
)

echo "[$(ts)] HousingQA state-filter completion gate"
echo "[$(ts)] run_finalizer=$RUN_FINALIZER require_full_exemplar=$REQUIRE_FULL_EXEMPLAR allow_existing_locks=$ALLOW_EXISTING_LOCKS"

echo
echo "[$(ts)] launch-lock gate"
existing_locks=0
for lock_dir in "${HOUSING_LOCK_DIRS[@]}"; do
  if [[ -d "$lock_dir" ]]; then
    existing_locks=$((existing_locks + 1))
    echo "lock present: $lock_dir"
    if [[ -f "$lock_dir/metadata" ]]; then
      sed 's/^/[lock] /' "$lock_dir/metadata"
    fi
  else
    echo "lock absent: $lock_dir"
  fi
done
if (( existing_locks > 0 )) && ! truthy "$ALLOW_EXISTING_LOCKS"; then
  echo "[$(ts)] existing launch lock(s) found; completion is not provable while a run may be active" >&2
  exit 12
fi

echo
echo "[$(ts)] syntax and Python helper checks"
bash -n \
  scripts/local/check_housing_gemma_readiness.sh \
  scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh \
  scripts/local/run_housing_gemma_core_queue.sh \
  scripts/local/run_housing_gemma_after_key_reset.sh \
  scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh \
  scripts/local/run_housing_gemma_exemplar_full_after_core.sh \
  scripts/local/merge_audit_housing_gemma_rag_simple.sh \
  scripts/local/audit_housing_gemma_core_rows.sh \
  scripts/local/audit_housing_gemma_exemplar_full.sh \
  scripts/local/finalize_housing_gemma_signoff.sh \
  scripts/local/verify_housing_statefilter_goal_complete.sh \
  scripts/local/watch_housing_gemma_until_ready.sh \
  scripts/local/housing_gemma_budget_watcher.sh
python3 -m py_compile \
  scripts/audit_housing_statefilter_goal.py \
  scripts/audit_housing_statefilter_detail.py \
  scripts/summarize_housing_statefilter_signoff.py \
  scripts/update_current_status.py \
  scripts/check_expected_provider_model.py \
  scripts/check_openrouter_key_status.py \
  scripts/check_openrouter_chat_route.py

if truthy "$RUN_FINALIZER"; then
  echo
  echo "[$(ts)] finalizing any missing Gemma signoff rows"
  scripts/local/finalize_housing_gemma_signoff.sh
else
  echo
  echo "[$(ts)] skipping finalizer by request; refreshing status directly"
  python3 scripts/update_current_status.py
fi

if truthy "$REQUIRE_FULL_EXEMPLAR"; then
  echo
  echo "[$(ts)] full-N snap_hyre_exemplar diagnostic audit"
  scripts/local/audit_housing_gemma_exemplar_full.sh
  echo
  echo "[$(ts)] finalizing snap_hyre_exemplar signoff row"
  MODES=snap_hyre_exemplar scripts/local/finalize_housing_gemma_signoff.sh
else
  echo
  echo "[$(ts)] skipping full-N snap_hyre_exemplar diagnostic gate by request"
fi

echo
echo "[$(ts)] focused 9-cell completion audit"
python3 scripts/audit_housing_statefilter_goal.py

echo
echo "[$(ts)] HousingQA state-filter goal is complete"
