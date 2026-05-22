#!/usr/bin/env bash
# One-command HousingQA Gemma continuation after OpenRouter capacity is restored.
#
# This fails closed before launch if the exact-model route or OpenRouter budget
# is unavailable. After clean full-N audits, it finalizes missing Gemma signoff
# rows in docs/signoff_log.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

export UV="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export PROVIDER="${PROVIDER:-or-gemma4-26b}"
export MODEL_LABEL="${MODEL_LABEL:-or-gemma4-26b}"
export OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"
export OPENROUTER_MIN_LIMIT_REMAINING="${OPENROUTER_MIN_LIMIT_REMAINING:-0.01}"
export NO_SILENT_FALLBACK=1
export EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
export SIGNOFF_SNIPPETS_OUT="${SIGNOFF_SNIPPETS_OUT:-$ROOT/docs/generated/housing_gemma_signoff_candidates_$(date -u +%Y%m%d_%H%M%S).md}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
RUN_FULL_EXEMPLAR_AFTER_CORE="${RUN_FULL_EXEMPLAR_AFTER_CORE:-1}"
RUN_OPENROUTER_CHAT_PREFLIGHT="${RUN_OPENROUTER_CHAT_PREFLIGHT:-1}"
OPENROUTER_FREE_SUFFIX_ARG=()
if [[ "${OPENROUTER_ALLOW_FREE_SUFFIX:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  OPENROUTER_FREE_SUFFIX_ARG=(--allow-openrouter-free-suffix)
fi

echo "[$(ts)] Housing Gemma after-key-reset continuation"
echo "[$(ts)] provider=$PROVIDER model_label=$MODEL_LABEL route=OPENROUTER_PROVIDER_ONLY=$OPENROUTER_PROVIDER_ONLY"
echo "[$(ts)] signoff_snippets_out=$SIGNOFF_SNIPPETS_OUT"
echo "[$(ts)] preflight_only=$PREFLIGHT_ONLY"
echo "[$(ts)] run_full_exemplar_after_core=$RUN_FULL_EXEMPLAR_AFTER_CORE"
echo "[$(ts)] run_openrouter_chat_preflight=$RUN_OPENROUTER_CHAT_PREFLIGHT"

if [[ ! "$PREFLIGHT_ONLY" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  LAUNCH_LOCK_DIR="${LAUNCH_LOCK_DIR:-/tmp/housing_gemma_after_key_reset.lock}"
  if ! mkdir "$LAUNCH_LOCK_DIR" 2>/dev/null; then
    echo "[$(ts)] ERROR: another Housing Gemma full launch appears to hold $LAUNCH_LOCK_DIR" >&2
    if [[ -f "$LAUNCH_LOCK_DIR/metadata" ]]; then
      echo "[$(ts)] Existing lock metadata:" >&2
      sed 's/^/[lock] /' "$LAUNCH_LOCK_DIR/metadata" >&2
    fi
    echo "[$(ts)] Remove the lock only after verifying no matching launch is still running." >&2
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
    echo "signoff_snippets_out=$SIGNOFF_SNIPPETS_OUT"
    echo "command=$0 $*"
  } > "$LAUNCH_LOCK_DIR/metadata"
  echo "[$(ts)] acquired full-launch lock: $LAUNCH_LOCK_DIR"
else
  echo "[$(ts)] preflight-only mode skips full-launch lock"
fi

echo "[$(ts)] local integrity checks before model/API preflight"
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
  scripts/local/housing_gemma_budget_watcher.sh \
  scripts/local/run_answer_cell.sh
python3 -m py_compile \
  scripts/audit_housing_statefilter_goal.py \
  scripts/audit_housing_statefilter_detail.py \
  scripts/summarize_housing_statefilter_signoff.py \
  scripts/update_current_status.py \
  scripts/check_expected_provider_model.py \
  scripts/check_openrouter_key_status.py \
  scripts/check_openrouter_chat_route.py \
  scripts/analyze_detail_flags.py \
  scripts/audit_retrieval_cache.py \
  scripts/build_generation_cache.py \
  scripts/build_retrieval_cache.py \
  scripts/build_retrieval_doc_cache.py

python3 scripts/check_expected_provider_model.py \
  --provider "$PROVIDER" \
  --expected-model "$EXPECTED_GEMMA_MODEL" \
  --expected-label "or-gemma4-26b" \
  "${OPENROUTER_FREE_SUFFIX_ARG[@]}"
python3 scripts/check_openrouter_key_status.py --min-limit-remaining "$OPENROUTER_MIN_LIMIT_REMAINING"
if [[ "$RUN_OPENROUTER_CHAT_PREFLIGHT" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  python3 scripts/check_openrouter_chat_route.py \
    --provider "$PROVIDER" \
    --expected-model "$EXPECTED_GEMMA_MODEL" \
    --provider-only "$OPENROUTER_PROVIDER_ONLY" \
    "${OPENROUTER_FREE_SUFFIX_ARG[@]}"
fi

if [[ "$PREFLIGHT_ONLY" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "[$(ts)] preflight passed; exiting before resume/cache/answer/signoff work"
  exit 0
fi

mkdir -p "$(dirname "$SIGNOFF_SNIPPETS_OUT")"
if [[ ! -s "$SIGNOFF_SNIPPETS_OUT" ]]; then
  {
    echo "# HousingQA Gemma Signoff Candidates"
    echo
    echo "Generated: $(ts)"
    echo
    echo "Provider: \`$PROVIDER\`"
    echo "Model label: \`$MODEL_LABEL\`"
    echo "Route: \`OPENROUTER_PROVIDER_ONLY=$OPENROUTER_PROVIDER_ONLY\`"
    echo
    echo "These rows mirror clean audit output. This wrapper later appends"
    echo "missing rows through \`scripts/local/finalize_housing_gemma_signoff.sh\`"
    echo "after an independent full-log check."
    echo
  } > "$SIGNOFF_SNIPPETS_OUT"
fi

scripts/local/resume_housing_gemma_rag_simple_after_key_reset.sh
scripts/local/merge_audit_housing_gemma_rag_simple.sh

echo "[$(ts)] rag_simple merge/audit gate passed; launching Gemma HyDE/Snap-HyRE core queue"
WAIT_PATTERNS="" scripts/local/run_housing_gemma_core_queue.sh

echo "[$(ts)] Gemma HyDE/Snap-HyRE queue finished; auditing generated rows"
MODES="rag_hyde snap_hyre" scripts/local/audit_housing_gemma_core_rows.sh

echo "[$(ts)] Generated-row audits passed; finalizing Gemma signoff rows"
scripts/local/finalize_housing_gemma_signoff.sh

if [[ "$RUN_FULL_EXEMPLAR_AFTER_CORE" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  echo "[$(ts)] Core Gemma rows finalized; launching full-N snap_hyre_exemplar diagnostic"
  scripts/local/run_housing_gemma_exemplar_full_after_core.sh
else
  echo "[$(ts)] skipping full-N snap_hyre_exemplar diagnostic by RUN_FULL_EXEMPLAR_AFTER_CORE=$RUN_FULL_EXEMPLAR_AFTER_CORE"
fi

echo "[$(ts)] Gemma continuation finished"
echo "[$(ts)] signoff candidate snippets: $SIGNOFF_SNIPPETS_OUT"
