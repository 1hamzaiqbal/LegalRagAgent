#!/usr/bin/env bash
# Merge and audit HousingQA Gemma 26B state-filter core rows after answer runs.
#
# This script is intentionally a gate, not a signoff writer. It fails closed
# until every requested row is full-N and clean, then prints the merged detail
# logs that can be cited in docs/signoff_log.md.

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
QUESTIONS="${QUESTIONS:-full}"
SEED="${SEED:-42}"
EXPECTED_ROWS="${EXPECTED_ROWS:-6853}"
MAX_K="${MAX_K:-10}"
RETRIEVAL_K="${RETRIEVAL_K:-5}"
GEN_ROOT="${GEN_ROOT:-$ROOT/caches/hyre/full}"
RET_ROOT="${RET_ROOT:-$ROOT/caches/retrieval/full}"
OUT_ROOT="${OUT_ROOT:-$ROOT/logs/merged}"
EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
SIGNOFF_SNIPPETS_OUT="${SIGNOFF_SNIPPETS_OUT:-}"

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(rag_simple rag_hyde snap_hyre)
fi

mkdir -p "$OUT_ROOT"
if [[ -n "$SIGNOFF_SNIPPETS_OUT" ]]; then
  mkdir -p "$(dirname "$SIGNOFF_SNIPPETS_OUT")"
  if [[ ! -s "$SIGNOFF_SNIPPETS_OUT" ]]; then
    {
      echo "# HousingQA Gemma Signoff Candidates"
      echo
      echo "Generated: $(ts)"
      echo
      echo "Provider: \`$PROVIDER\`"
      echo "Model label: \`$MODEL_LABEL\`"
      echo
      echo "These rows are emitted only after the local audit helper passes. Review"
      echo "them, then paste accepted rows into \`docs/signoff_log.md\`."
      echo
    } > "$SIGNOFF_SNIPPETS_OUT"
  fi
fi

emit_signoff_row() {
  local detail_log="$1"
  local mode="$2"
  if [[ -n "$SIGNOFF_SNIPPETS_OUT" ]]; then
    "$UV" run python scripts/summarize_housing_statefilter_signoff.py \
      "$detail_log" \
      --provider "$PROVIDER" \
      --mode "$mode" \
      | tee -a "$SIGNOFF_SNIPPETS_OUT"
  else
    "$UV" run python scripts/summarize_housing_statefilter_signoff.py \
      "$detail_log" \
      --provider "$PROVIDER" \
      --mode "$mode"
  fi
}

python3 scripts/check_expected_provider_model.py \
  --provider "$PROVIDER" \
  --expected-model "$EXPECTED_GEMMA_MODEL" \
  --expected-label "or-gemma4-26b"

check_generation_cache() {
  local mode="$1"
  local cache="$GEN_ROOT/housing_q${QUESTIONS}_seed${SEED}_${MODEL_LABEL}_${mode}.jsonl"

  echo "[$(ts)] generation-cache audit mode=$mode cache=$cache"
  [[ -s "$cache" ]] || { echo "missing generation cache: $cache" >&2; exit 1; }

  "$UV" run python - "$cache" "$mode" "$EXPECTED_ROWS" <<'PY'
import json
import sys

path, mode, expected_raw = sys.argv[1], sys.argv[2], sys.argv[3]
expected = int(expected_raw)
rows = []
with open(path, errors="ignore") as handle:
    for line in handle:
        if line.strip():
            rows.append(json.loads(line))

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
    if row.get("hyde_parse_ok") is False
    or (mode == "snap_hyre" and row.get("snap_hyre_parse_ok") is False)
]
missing_snap = [
    row for row in rows
    if mode == "snap_hyre" and not row.get("snap_letter")
]
answer_artifacts = [
    row for row in rows
    if row.get("hyde_contains_answer_artifact") is True
]

print(
    f"rows={len(rows)} expected={expected} duplicate_labels={duplicate_labels} "
    f"errors={len(errors)} missing_passages={len(missing_passages)} "
    f"fallbacks={len(fallbacks)} parse_failures={len(parse_failures)} "
    f"missing_snap={len(missing_snap)} answer_artifacts={len(answer_artifacts)}"
)

bad = {
    "duplicate_labels": [None] * duplicate_labels,
    "errors": errors,
    "missing_passages": missing_passages,
    "fallbacks": fallbacks,
    "parse_failures": parse_failures,
    "missing_snap": missing_snap,
    "answer_artifacts": answer_artifacts,
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
}

select_detail_logs() {
  local mode="$1"
  python3 - "$PROVIDER" "$MODEL_LABEL" "$mode" <<'PY'
import json
import sys
from pathlib import Path

provider, model_label, mode = sys.argv[1], sys.argv[2], sys.argv[3]
pattern = (
    f"eval_{mode}_{provider}_*_housing_"
    f"local-snap-hyre-{model_label}-housing-{mode}-nfull-k5*_detail.jsonl"
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
}

audit_generated_mode() {
  local mode="$1"
  local ts_suffix
  ts_suffix="$(date -u +%Y%m%d_%H%M%S)"
  local out="$OUT_ROOT/housing_${MODEL_LABEL}_${mode}_statefilter_full_${ts_suffix}_detail.jsonl"
  local retrieval_cache="$RET_ROOT/housing_q${QUESTIONS}_seed${SEED}_statefilter_${MODEL_LABEL}_${mode}_k${MAX_K}.jsonl"

  check_generation_cache "$mode"

  mapfile -t detail_logs < <(select_detail_logs "$mode")
  if (( ${#detail_logs[@]} == 0 )); then
    echo "no state-filtered full-N candidate detail logs found for provider=$PROVIDER mode=$mode" >&2
    exit 1
  fi

  echo "[$(ts)] merging mode=$mode detail_logs=${#detail_logs[@]} out=$out"
  printf '  %s\n' "${detail_logs[@]}"

  "$UV" run python scripts/merge_detail_logs.py \
    --output "$out" \
    --key label \
    --on-duplicate last \
    "${detail_logs[@]}"

  "$UV" run python scripts/analyze_detail_flags.py "$out"

  "$UV" run python scripts/audit_housing_statefilter_detail.py \
    --provider "$PROVIDER" \
    --mode "$mode" \
    --expected-rows "$EXPECTED_ROWS" \
    --retrieval-k "$RETRIEVAL_K" \
    --require-hyre-cache \
    "$out"

  "$UV" run python scripts/audit_retrieval_cache.py \
    --cache "$retrieval_cache" \
    --dataset housing \
    --min-k "$MAX_K" \
    --ks 1,5,10

  emit_signoff_row "$out" "$mode"

  echo "[$(ts)] CLEAN mode=$mode detail_log=$out"
}

echo "[$(ts)] Housing Gemma core audit provider=$PROVIDER model_label=$MODEL_LABEL modes=${MODES_ARR[*]}"

for mode in "${MODES_ARR[@]}"; do
  echo
  echo "[$(ts)] audit mode=$mode"
  case "$mode" in
    rag_simple)
      scripts/local/merge_audit_housing_gemma_rag_simple.sh
      ;;
    rag_hyde|snap_hyre)
      audit_generated_mode "$mode"
      ;;
    *)
      echo "unsupported Housing Gemma core audit mode: $mode" >&2
      exit 2
      ;;
  esac
done

echo
echo "[$(ts)] Housing Gemma core audit complete. Add clean rows to docs/signoff_log.md, then run python3 scripts/update_current_status.py."
