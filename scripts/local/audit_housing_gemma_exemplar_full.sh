#!/usr/bin/env bash
# Merge and audit the full-N HousingQA Gemma snap_hyre_exemplar diagnostic.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

UV="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
PROVIDER="${PROVIDER:-or-gemma4-26b}"
MODEL_LABEL="${MODEL_LABEL:-$PROVIDER}"
export PROVIDER MODEL_LABEL
MODE="${MODE:-snap_hyre_exemplar}"
QUESTIONS="${QUESTIONS:-full}"
SEED="${SEED:-42}"
EXPECTED_ROWS="${EXPECTED_ROWS:-6853}"
MAX_K="${MAX_K:-10}"
RETRIEVAL_K="${RETRIEVAL_K:-5}"
GEN_ROOT="${GEN_ROOT:-$ROOT/caches/generation/full}"
RET_ROOT="${RET_ROOT:-$ROOT/caches/retrieval/full}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/merged}"
EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
SIGNOFF_SNIPPETS_OUT="${SIGNOFF_SNIPPETS_OUT:-}"

if [[ "$MODE" != "snap_hyre_exemplar" ]]; then
  echo "unsupported exemplar audit mode: $MODE" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
if [[ -n "$SIGNOFF_SNIPPETS_OUT" ]]; then
  mkdir -p "$(dirname "$SIGNOFF_SNIPPETS_OUT")"
fi

python3 scripts/check_expected_provider_model.py \
  --provider "$PROVIDER" \
  --expected-model "$EXPECTED_GEMMA_MODEL" \
  --expected-label "or-gemma4-26b"

gen="$GEN_ROOT/housing_q${QUESTIONS}_seed${SEED}_${MODEL_LABEL}_${MODE}_realpassage.jsonl"
ret="$RET_ROOT/housing_q${QUESTIONS}_seed${SEED}_statefilter_${MODEL_LABEL}_${MODE}_realpassage_k${MAX_K}.jsonl"
out="$OUT_ROOT/housing_${MODEL_LABEL}_${MODE}_statefilter_full_$(date -u +%Y%m%d_%H%M%S)_detail.jsonl"

echo "[$(ts)] Housing Gemma exemplar audit provider=$PROVIDER model_label=$MODEL_LABEL"
echo "[$(ts)] generation_cache=$gen"
echo "[$(ts)] retrieval_cache=$ret"

"$UV" run python - "$gen" "$EXPECTED_ROWS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(f"missing generation cache: {path}")
rows = [json.loads(line) for line in path.open(errors="ignore") if line.strip()]
labels = [str(row.get("label") or row.get("idx") or "") for row in rows]
duplicate_labels = len(labels) - len(set(labels))
errors = [row for row in rows if row.get("error")]
missing_passages = [row for row in rows if not row.get("hyde_passage")]

def truthy_fallback(row):
    falsey_strings = {"", "0", "false", "no", "none", "null", "[]", "{}"}
    for key, value in row.items():
        if "fallback" not in str(key).lower():
            continue
        if isinstance(value, bool):
            if value:
                return True
            continue
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip().lower() not in falsey_strings:
                return True
            continue
        if isinstance(value, (list, tuple, set, dict)):
            if value:
                return True
            continue
        if value:
            return True
    return False

fallbacks = [row for row in rows if truthy_fallback(row)]
parse_failures = [
    row for row in rows
    if row.get("hyde_parse_ok") is False or row.get("snap_hyre_parse_ok") is False
]
missing_snap = [row for row in rows if not row.get("snap_letter")]
answer_artifacts = [row for row in rows if row.get("hyde_contains_answer_artifact") is True]
style_signal_missing = [row for row in rows if row.get("passage_style_signal_used") is not True]
print(
    f"rows={len(rows)} expected={expected} duplicate_labels={duplicate_labels} "
    f"errors={len(errors)} missing_passages={len(missing_passages)} "
    f"fallbacks={len(fallbacks)} parse_failures={len(parse_failures)} "
    f"missing_snap={len(missing_snap)} answer_artifacts={len(answer_artifacts)} "
    f"style_signal_missing={len(style_signal_missing)}"
)
bad = {
    "duplicate_labels": [None] * duplicate_labels,
    "errors": errors,
    "missing_passages": missing_passages,
    "fallbacks": fallbacks,
    "parse_failures": parse_failures,
    "missing_snap": missing_snap,
    "answer_artifacts": answer_artifacts,
    "style_signal_missing": style_signal_missing,
}
if len(rows) != expected:
    raise SystemExit(f"expected {expected} generation rows, found {len(rows)}")
for name, rows_for_name in bad.items():
    if rows_for_name:
        labels_for_name = [
            str(row.get("label") or row.get("idx") or "?")
            for row in rows_for_name[:10]
            if row is not None
        ]
        suffix = ": " + ",".join(labels_for_name) if labels_for_name else ""
        raise SystemExit(f"{name}{suffix}")
PY

mapfile -t detail_logs < <(
  python3 - "$PROVIDER" "$MODEL_LABEL" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

provider, model_label, mode = sys.argv[1], sys.argv[2], sys.argv[3]
pattern = (
    f"eval_{mode}_{provider}_*_housing_"
    f"local-snap-hyre-{model_label}-housing-{mode}-nfull-k5_detail.jsonl"
)
for path in sorted(Path("logs").glob(pattern)):
    keep = False
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("provider") == provider
                and row.get("mode") == mode
                and row.get("dataset") == "housing"
                and row.get("housing_state_filter") is True
            ):
                keep = True
                break
    if keep:
        print(path)
PY
)

if (( ${#detail_logs[@]} == 0 )); then
  echo "no state-filtered full-N candidate detail logs found for provider=$PROVIDER mode=$MODE" >&2
  exit 1
fi

echo "[$(ts)] merging exemplar detail logs=${#detail_logs[@]} out=$out"
printf '  %s\n' "${detail_logs[@]}"

"$UV" run python scripts/merge_detail_logs.py \
  --output "$out" \
  --key label \
  --on-duplicate last \
  "${detail_logs[@]}"

"$UV" run python scripts/analyze_detail_flags.py "$out"

"$UV" run python scripts/audit_housing_statefilter_detail.py \
  --provider "$PROVIDER" \
  --mode "$MODE" \
  --expected-rows "$EXPECTED_ROWS" \
  --retrieval-k "$RETRIEVAL_K" \
  --require-hyre-cache \
  "$out"

"$UV" run python scripts/audit_retrieval_cache.py \
  --cache "$ret" \
  --dataset housing \
  --min-k "$MAX_K" \
  --ks 1,5,10

if [[ -n "$SIGNOFF_SNIPPETS_OUT" ]]; then
  "$UV" run python scripts/summarize_housing_statefilter_signoff.py \
    "$out" \
    --provider "$PROVIDER" \
    --mode "$MODE" \
    | tee -a "$SIGNOFF_SNIPPETS_OUT"
else
  "$UV" run python scripts/summarize_housing_statefilter_signoff.py \
    "$out" \
    --provider "$PROVIDER" \
    --mode "$MODE"
fi

echo "[$(ts)] CLEAN exemplar detail_log=$out"
