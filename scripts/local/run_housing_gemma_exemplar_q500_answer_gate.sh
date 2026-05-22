#!/usr/bin/env bash
# Run the HousingQA Gemma q500 canonical-vs-exemplar Snap-HyRE answer gate.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ts() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

PROVIDER="${PROVIDER:-or-gemma4-26b}"
MODEL_LABEL="${MODEL_LABEL:-$PROVIDER}"
EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"
RUN_OPENROUTER_CHAT_PREFLIGHT="${RUN_OPENROUTER_CHAT_PREFLIGHT:-1}"
OPENROUTER_FREE_SUFFIX_ARG=()
if [[ "${OPENROUTER_ALLOW_FREE_SUFFIX:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  OPENROUTER_FREE_SUFFIX_ARG=(--allow-openrouter-free-suffix)
fi
QUESTIONS="${QUESTIONS:-500}"
SEED="${SEED:-42}"
RETRIEVAL_K="${RETRIEVAL_K:-5}"
CACHE_SCOPE="${CACHE_SCOPE:-q${QUESTIONS}_seed${SEED}_statefilter}"
GENERATION_CACHE_ROOT="${GENERATION_CACHE_ROOT:-$ROOT/caches/generation/probes}"
RETRIEVAL_CACHE_ROOT="${RETRIEVAL_CACHE_ROOT:-$ROOT/caches/retrieval/probes}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"

mkdir -p "$LOG_DIR"

mode_complete() {
  local mode="$1"
  python3 - "$mode" "$PROVIDER" <<'PY'
import glob
import json
import sys

mode, provider = sys.argv[1], sys.argv[2]
max_completion_tokens = 2048
near_cap_cutoff = max_completion_tokens - 16

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

paths = glob.glob(f"logs/eval_{mode}_{provider}_*_housing_*n500-k5_detail.jsonl")
for path in paths:
    rows = []
    ok = True
    with open(path, errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(row)
            ok = ok and row.get("dataset") == "housing"
            ok = ok and row.get("mode") == mode
            ok = ok and row.get("provider") == provider
            ok = ok and row.get("housing_state_filter") is True
            ok = ok and bool(row.get("retrieval_where"))
            ok = ok and row.get("retrieval_cache_hit") is True
            ok = ok and row.get("retrieval_doc_cache_hit") is True
            ok = ok and row.get("hyre_cache_hit") is True
            ok = ok and not row.get("error")
            ok = ok and str(row.get("predicted_answer") or "").strip() in {"Yes", "No"}
            final = str(row.get("final_answer") or "").strip().splitlines()
            ok = ok and bool(final) and final[-1].strip() in {"Answer: Yes", "Answer: No"}
            ok = ok and not truthy_fallback(row)
            text = "\n".join(str(row.get(k, "")) for k in ("final_answer", "hyde_passage", "snap_answer")).lower()
            ok = ok and "<think" not in text and "</think" not in text
            ok = ok and len(row.get("evidence_store") or []) == 5
            ok = ok and int(row.get("output_tokens") or 0) < near_cap_cutoff
            ok = ok and int(row.get("answer_format_retry_output_tokens") or 0) < near_cap_cutoff
    if len(rows) == 500 and ok:
        print(path)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_mode() {
  local mode="$1"
  local doc_cache="$2"
  local log="$LOG_DIR/run_housing_gemma_q500_${mode}_$(date -u +%Y%m%d_%H%M%S).out"

  if mode_complete "$mode"; then
    echo "[$(ts)] q500 mode=$mode already complete and clean; skipping"
    return 0
  fi
  MODEL_LABEL="$MODEL_LABEL" python3 scripts/check_expected_provider_model.py \
    --provider "$PROVIDER" \
    --expected-model "$EXPECTED_GEMMA_MODEL" \
    --expected-label "or-gemma4-26b" \
    "${OPENROUTER_FREE_SUFFIX_ARG[@]}"
  python3 scripts/check_openrouter_key_status.py --min-limit-remaining "${OPENROUTER_MIN_LIMIT_REMAINING:-0.01}"
  if [[ "$RUN_OPENROUTER_CHAT_PREFLIGHT" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
    python3 scripts/check_openrouter_chat_route.py \
      --provider "$PROVIDER" \
      --expected-model "$EXPECTED_GEMMA_MODEL" \
      --provider-only "$OPENROUTER_PROVIDER_ONLY" \
      "${OPENROUTER_FREE_SUFFIX_ARG[@]}"
  fi

  echo
  echo "[$(ts)] q500 answer gate mode=$mode doc_cache=$doc_cache log=$log"
  [[ -s "$doc_cache" ]] || { echo "missing doc cache $doc_cache" >&2; exit 2; }

  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  HF_DATASETS_OFFLINE=1 \
  NO_SILENT_FALLBACK=1 \
  EVAL_FINAL_FORMAT_RETRY=1 \
  EVAL_GENERATION_FORMAT_RETRY=1 \
  LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}" \
  EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}" \
  PROVIDER="$PROVIDER" \
  MODEL_LABEL="$MODEL_LABEL" \
  DATASET=housing \
  QUESTIONS="$QUESTIONS" \
  SEED="$SEED" \
  CACHE_SCOPE="$CACHE_SCOPE" \
  RETRIEVAL_K="$RETRIEVAL_K" \
  MODES="$mode" \
  USE_CACHES=1 \
  REQUIRE_RETRIEVAL_CACHES=1 \
  EVAL_HOUSING_STATE_FILTER=1 \
  GENERATION_CACHE_ROOT="$GENERATION_CACHE_ROOT" \
  HYRE_CACHE_ROOT="$GENERATION_CACHE_ROOT" \
  RETRIEVAL_CACHE_ROOT="$RETRIEVAL_CACHE_ROOT" \
  RETRIEVAL_DOC_CACHE_PATH="$doc_cache" \
  RETRIEVAL_DOC_CACHE_STRICT=1 \
  scripts/local/run_answer_cell.sh 2>&1 | tee "$log"
}

run_mode snap_hyre \
  "$ROOT/caches/retrieval_doc/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_k10_doc_cache.jsonl"
run_mode snap_hyre_exemplar \
  "$ROOT/caches/retrieval_doc/probes/housing_q500_seed42_statefilter_or-gemma4-26b_snap_hyre_exemplar_realpassage_k10_doc_cache.jsonl"

echo
echo "[$(ts)] q500 answer gate complete"
