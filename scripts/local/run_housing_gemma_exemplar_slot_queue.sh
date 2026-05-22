#!/usr/bin/env bash
# Run the Gemma Housing q500 exemplar gate once one Groq answer slot frees.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

LOG_DIR="${LOG_DIR:-$ROOT/logs}"
# Preserve an intentionally empty WAIT_PATTERNS from callers that have already
# audited active jobs and want this queue to start immediately.
WAIT_PATTERNS="${WAIT_PATTERNS-housing_70b_rag_simple_ housing_8b_rag_hyde_}"
mkdir -p "$LOG_DIR"

echo "[$(ts)] housing Gemma exemplar slot queue start"
echo "[$(ts)] waiting until at least one Groq answer slot is free: $WAIT_PATTERNS"

while true; do
  active=0
  for pattern in $WAIT_PATTERNS; do
    if tmux ls 2>/dev/null | grep -q "$pattern"; then
      active=$((active + 1))
    fi
  done
  if (( active < 2 )); then
    break
  fi
  sleep 60
done

if tmux ls 2>/dev/null | grep -q "housing_gemma_followup_queue"; then
  echo "[$(ts)] note: follow-up queue is still waiting on Gemma rag_simple and will skip q500 if this queue completes it first"
fi

echo "[$(ts)] launching q500 exemplar answer gate on explicit same-model Cloudflare route"
OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}" \
scripts/local/run_housing_gemma_exemplar_q500_answer_gate.sh \
  > "$LOG_DIR/run_housing_gemma_exemplar_slot_q500_gate_$(date -u +%Y%m%d_%H%M%S).out" 2>&1

echo "[$(ts)] housing Gemma exemplar slot queue complete"
