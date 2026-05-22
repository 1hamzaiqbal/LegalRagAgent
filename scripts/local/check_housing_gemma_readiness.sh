#!/usr/bin/env bash
# Read-only readiness check for the remaining HousingQA Gemma 26B state-filter rows.
#
# Default mode is offline-only and exits 0 if the local launch prerequisites are
# sane, even though the Housing goal is intentionally still incomplete. Set
# CHECK_NETWORK=1 to include the OpenRouter budget/API preflight.

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
export PROVIDER="${PROVIDER:-or-gemma4-26b}"
export MODEL_LABEL="${MODEL_LABEL:-or-gemma4-26b}"
export OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"
export OPENROUTER_MIN_LIMIT_REMAINING="${OPENROUTER_MIN_LIMIT_REMAINING:-0.01}"
export EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
export NO_SILENT_FALLBACK=1

CHECK_NETWORK="${CHECK_NETWORK:-0}"
ALLOW_EXISTING_LOCKS="${ALLOW_EXISTING_LOCKS:-0}"
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

echo "[$(ts)] Housing Gemma readiness check"
echo "[$(ts)] provider=$PROVIDER model_label=$MODEL_LABEL route=OPENROUTER_PROVIDER_ONLY=$OPENROUTER_PROVIDER_ONLY"
echo "[$(ts)] check_network=$CHECK_NETWORK"

echo
echo "[$(ts)] launch-lock snapshot"
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
  echo "[$(ts)] existing launch lock(s) found; verify active jobs or remove stale locks before launch" >&2
  echo "[$(ts)] set ALLOW_EXISTING_LOCKS=1 only for a read-only inspection" >&2
  exit 12
fi

echo
echo "[$(ts)] focused 9-cell Housing gate (allowed incomplete while Gemma rows remain)"
python3 scripts/audit_housing_statefilter_goal.py --allow-incomplete

echo
echo "[$(ts)] verifying Gemma rag_simple resume offsets and exact model label without API calls"
VERIFY_ONLY=1 scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh

echo
echo "[$(ts)] remaining Gemma cache/detail snapshot"
for mode in rag_hyde snap_hyre; do
  gen="caches/hyre/full/housing_qfull_seed42_${MODEL_LABEL}_${mode}.jsonl"
  ret="caches/retrieval/full/housing_qfull_seed42_statefilter_${MODEL_LABEL}_${mode}_k10.jsonl"
  doc="caches/retrieval_doc/full/housing_qfull_seed42_statefilter_${MODEL_LABEL}_${mode}_k10_doc_cache.jsonl"
  for path in "$gen" "$ret" "$doc"; do
    if [[ -s "$path" ]]; then
      rows="$(wc -l < "$path" | tr -d ' ')"
      echo "present rows=$rows $path"
    else
      echo "missing $path"
    fi
  done
done

echo
echo "[$(ts)] Gemma rag_simple merge gate status"
python3 scripts/report_housing_gemma_rag_simple_gaps.py
probe_stamp="$(date -u +%Y%m%d_%H%M%S)"
merge_probe_json="/tmp/housing_${MODEL_LABEL}_rag_simple_readiness_probe_${probe_stamp}.jsonl"
merge_probe_log="/tmp/housing_${MODEL_LABEL}_rag_simple_readiness_probe_${probe_stamp}.out"
if OUT="$merge_probe_json" scripts/local/merge_audit_housing_gemma_rag_simple.sh >"$merge_probe_log" 2>&1; then
  echo "rag_simple merge gate passed; full clean detail is available for finalization/signoff"
  echo "merge_probe_json=$merge_probe_json"
else
  echo "rag_simple merge gate failed; row is still partial or structurally blocked"
  echo "merge_probe_log=$merge_probe_log"
  tail -n 12 "$merge_probe_log" || true
fi

if truthy "$CHECK_NETWORK"; then
  echo
  echo "[$(ts)] checking OpenRouter API/budget preflight without launching rows"
  PREFLIGHT_ONLY=1 scripts/local/run_housing_gemma_after_key_reset.sh
else
  echo
  echo "[$(ts)] skipped network preflight; set CHECK_NETWORK=1 to test OpenRouter readiness"
fi

echo
echo "[$(ts)] readiness check complete"
echo "Next launch after CHECK_NETWORK=1 passes: scripts/local/run_housing_gemma_after_key_reset.sh"
