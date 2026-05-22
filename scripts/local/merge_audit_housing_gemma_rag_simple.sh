#!/usr/bin/env bash
# Merge and audit the full HousingQA Gemma 26B state-filtered rag_simple row.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

UV="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
OUT="${OUT:-logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_$(date -u +%Y%m%d_%H%M%S)_detail.jsonl}"

mkdir -p "$(dirname "$OUT")"

mapfile -t DETAIL_LOGS < <(
  python3 - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("logs").glob("eval_rag_simple_or-gemma4-26b_*_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5*_detail.jsonl")):
    keep = False
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("housing_state_filter") is True:
                keep = True
                break
    if keep:
        print(path)
PY
)

if (( ${#DETAIL_LOGS[@]} == 0 )); then
  echo "no Gemma Housing rag_simple detail logs found" >&2
  exit 2
fi

echo "merging ${#DETAIL_LOGS[@]} detail logs into $OUT"
printf '  %s\n' "${DETAIL_LOGS[@]}"

"$UV" run python scripts/merge_detail_logs.py \
  --output "$OUT" \
  --key label \
  --on-duplicate last \
  "${DETAIL_LOGS[@]}"

"$UV" run python scripts/analyze_detail_flags.py "$OUT"

"$UV" run python scripts/audit_housing_statefilter_detail.py \
  --provider or-gemma4-26b \
  --mode rag_simple \
  --expected-rows 6853 \
  "$OUT"

"$UV" run python scripts/audit_retrieval_cache.py \
  --cache caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl \
  --dataset housing \
  --min-k 10 \
  --ks 1,5,10

if [[ -n "${SIGNOFF_SNIPPETS_OUT:-}" ]]; then
  "$UV" run python scripts/summarize_housing_statefilter_signoff.py \
    "$OUT" \
    --provider or-gemma4-26b \
    --mode rag_simple \
    | tee -a "$SIGNOFF_SNIPPETS_OUT"
else
  "$UV" run python scripts/summarize_housing_statefilter_signoff.py \
    "$OUT" \
    --provider or-gemma4-26b \
    --mode rag_simple
fi

echo "merged/audited detail log: $OUT"
echo "If clean, add the signoff entry to docs/signoff_log.md and refresh current_status.md."
