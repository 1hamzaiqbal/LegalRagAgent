#!/usr/bin/env bash
# Append missing HousingQA Gemma 26B state-filter signoff rows after clean audits.
#
# This is intentionally post-run only. It refuses to append a row unless
# summarize_housing_statefilter_signoff.py accepts a full clean detail log.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SIGNOFF="${SIGNOFF:-docs/signoff_log.md}"
PROVIDER="${PROVIDER:-or-gemma4-26b}"
MODES="${MODES:-rag_simple rag_hyde snap_hyre}"
REQUIRE_OPENROUTER_PROVIDER_ONLY="${REQUIRE_OPENROUTER_PROVIDER_ONLY:-}"
REQUIRE_CORE_COMPLETE_FOR_EXEMPLAR="${REQUIRE_CORE_COMPLETE_FOR_EXEMPLAR:-1}"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

truthy() {
  [[ "${1:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]
}

signed_already() {
  local mode="$1"
  grep -F "| HousingQA state-filtered | \`$PROVIDER\` | \`$mode\` |" "$SIGNOFF" >/dev/null 2>&1
}

find_clean_detail() {
  local mode="$1"
  python3 - "$PROVIDER" "$mode" <<'PY'
import json
import sys
from pathlib import Path

provider, mode = sys.argv[1], sys.argv[2]
patterns = [
    f"logs/merged/*{provider}*{mode}*housing*detail.jsonl",
    f"logs/merged/*housing*{provider}*{mode}*detail.jsonl",
    f"logs/eval_{mode}_{provider}_*_housing_*nfull-k5*_detail.jsonl",
]
paths = []
for pattern in patterns:
    paths.extend(Path(".").glob(pattern))
paths = sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)
for path in paths:
    rows = []
    try:
        with path.open(errors="ignore") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    except Exception:
        continue
    if len(rows) != 6853:
        continue
    if not rows:
        continue
    if not all(
        row.get("provider") == provider
        and row.get("mode") == mode
        and row.get("dataset") == "housing"
        and row.get("housing_state_filter") is True
        for row in rows
    ):
        continue
    print(path)
    raise SystemExit(0)
raise SystemExit(1)
PY
}

echo "[$(ts)] finalize Housing Gemma signoff provider=$PROVIDER modes=$MODES"

if truthy "$REQUIRE_CORE_COMPLETE_FOR_EXEMPLAR"; then
  for mode in $MODES; do
    if [[ "$mode" == "snap_hyre_exemplar" ]]; then
      echo "[$(ts)] verifying core Housing Gemma rows before exemplar signoff"
      python3 scripts/audit_housing_statefilter_goal.py
      break
    fi
  done
fi

rows_to_append=()
for mode in $MODES; do
  if signed_already "$mode"; then
    echo "[$(ts)] signoff already present for mode=$mode; skipping"
    continue
  fi
  detail="$(find_clean_detail "$mode")" || {
    echo "[$(ts)] no full clean state-filter detail log found for mode=$mode" >&2
    exit 1
  }
  echo "[$(ts)] signoff candidate mode=$mode detail=$detail"
  row="$(python3 scripts/summarize_housing_statefilter_signoff.py "$detail" --provider "$PROVIDER" --mode "$mode" --require-openrouter-provider-only "$REQUIRE_OPENROUTER_PROVIDER_ONLY")"
  rows_to_append+=("$row")
done

if (( ${#rows_to_append[@]} > 0 )); then
  {
    echo
    echo "### HousingQA Gemma state-filter finalization - $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    for row in "${rows_to_append[@]}"; do
      echo "$row"
    done
  } >> "$SIGNOFF"
  echo "[$(ts)] appended ${#rows_to_append[@]} signoff row(s) to $SIGNOFF"
else
  echo "[$(ts)] no signoff rows needed"
fi

python3 scripts/update_current_status.py
python3 scripts/audit_housing_statefilter_goal.py
