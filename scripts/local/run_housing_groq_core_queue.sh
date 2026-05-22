#!/usr/bin/env bash
# Sequentially run the remaining Groq HousingQA state-filter core rows.

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

echo "[$(ts)] housing Groq core queue start"
echo "[$(ts)] waiting for active Groq answer sessions to finish before launching queued Groq rows: $WAIT_PATTERNS"

for pattern in $WAIT_PATTERNS; do
  if tmux ls 2>/dev/null | grep -q "$pattern"; then
    echo "[$(ts)] waiting for tmux sessions matching '$pattern'"
    while tmux ls 2>/dev/null | grep -q "$pattern"; do
      sleep 60
    done
  fi
done

run_row() {
  local provider="$1"
  local mode="$2"
  local label="$3"
  local log="$LOG_DIR/run_housing_statefilter_${label}_$(date -u +%Y%m%d_%H%M%S).out"
  local active_pattern="housing_${label}_"

  if tmux ls 2>/dev/null | grep -q "$active_pattern"; then
    echo "[$(ts)] waiting for active standalone session matching '$active_pattern' before provider=$provider mode=$mode"
    while tmux ls 2>/dev/null | grep -q "$active_pattern"; do
      sleep 60
    done
  fi

  if python3 - "$provider" "$mode" <<'PY'
import glob
import json
import sys

provider, mode = sys.argv[1], sys.argv[2]
paths = glob.glob(f"logs/eval_{mode}_{provider}_*_housing_*detail.jsonl")
for path in paths:
    rows = 0
    ok = True
    with open(path, errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            ok = ok and row.get("dataset") == "housing"
            ok = ok and row.get("mode") == mode
            ok = ok and row.get("provider") == provider
            ok = ok and row.get("housing_state_filter") is True
            ok = ok and bool(row.get("retrieval_where"))
    if rows >= 6853 and ok:
        print(path)
        raise SystemExit(0)
raise SystemExit(1)
PY
  then
    echo "[$(ts)] skip provider=$provider mode=$mode because a complete state-filtered detail log already exists"
    return 0
  fi

  echo
  echo "[$(ts)] launch provider=$provider mode=$mode log=$log"
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  NO_SILENT_FALLBACK=1 \
  MODE="$mode" \
  PROVIDER="$provider" \
  MODEL_LABEL="$provider" \
  scripts/local/run_housing_statefilter_rag_simple_with_doc_cache.sh 2>&1 | tee "$log"
  echo "[$(ts)] complete provider=$provider mode=$mode"
}

run_row groq-llama70b rag_hyde 70b_rag_hyde
run_row groq-llama8b rag_hyde 8b_rag_hyde
run_row groq-llama8b snap_hyre 8b_snap_hyre

echo
echo "[$(ts)] housing Groq core queue complete"
