#!/usr/bin/env bash
# Wait for the Gemma Housing rag_simple row, then run the q500 exemplar gate.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

# Preserve an intentionally empty WAIT_PATTERN from callers that have already
# audited active jobs and want this queue to skip its initial wait.
WAIT_PATTERN="${WAIT_PATTERN-housing_gemma_rag_simple}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

echo "[$(ts)] housing Gemma follow-up queue start"
echo "[$(ts)] waiting for tmux sessions matching '$WAIT_PATTERN' to finish"

if [[ -n "$WAIT_PATTERN" ]]; then
  while tmux ls 2>/dev/null | grep -q "$WAIT_PATTERN"; do
    sleep 60
  done
fi

if tmux ls 2>/dev/null | grep -Eq "housing_gemma_exemplar_slot_queue|housing_gemma_exemplar_gate_retry"; then
  echo "[$(ts)] waiting for active exemplar queue before deciding whether q500 is needed"
  while tmux ls 2>/dev/null | grep -Eq "housing_gemma_exemplar_slot_queue|housing_gemma_exemplar_gate_retry"; do
    sleep 60
  done
fi

echo "[$(ts)] launching q500 exemplar answer gate on explicit same-model Cloudflare route"
OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}" \
scripts/local/run_housing_gemma_exemplar_q500_answer_gate.sh \
  > "$LOG_DIR/run_housing_gemma_exemplar_q500_gate_$(date -u +%Y%m%d_%H%M%S).out" 2>&1

echo "[$(ts)] housing Gemma follow-up queue complete"
