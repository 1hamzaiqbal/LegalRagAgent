#!/usr/bin/env bash
# Run the full-N HousingQA Gemma snap_hyre_exemplar diagnostic after core rows.

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
MAX_K="${MAX_K:-10}"
RETRIEVAL_K="${RETRIEVAL_K:-5}"
GEN_ROOT="${GEN_ROOT:-$ROOT/caches/generation/full}"
RET_ROOT="${RET_ROOT:-$ROOT/caches/retrieval/full}"
DOC_ROOT="${DOC_ROOT:-$ROOT/caches/retrieval_doc/full}"
EXPECTED_GEMMA_MODEL="${EXPECTED_GEMMA_MODEL:-google/gemma-4-26b-a4b-it}"
RUN_OPENROUTER_CHAT_PREFLIGHT="${RUN_OPENROUTER_CHAT_PREFLIGHT:-1}"
OPENROUTER_FREE_SUFFIX_ARG=()
if [[ "${OPENROUTER_ALLOW_FREE_SUFFIX:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  OPENROUTER_FREE_SUFFIX_ARG=(--allow-openrouter-free-suffix)
fi
LAUNCH_LOCK_DIR="${EXEMPLAR_FULL_LOCK_DIR:-/tmp/housing_gemma_exemplar_full.lock}"
SIGNOFF_SNIPPETS_OUT="${SIGNOFF_SNIPPETS_OUT:-$ROOT/docs/generated/housing_gemma_exemplar_signoff_candidates_$(date -u +%Y%m%d_%H%M%S).md}"

mkdir -p "$GEN_ROOT" "$RET_ROOT" "$DOC_ROOT" "$(dirname "$SIGNOFF_SNIPPETS_OUT")"

echo "[$(ts)] Housing Gemma full exemplar start provider=$PROVIDER model_label=$MODEL_LABEL"
echo "[$(ts)] signoff_snippets_out=$SIGNOFF_SNIPPETS_OUT"

if ! mkdir "$LAUNCH_LOCK_DIR" 2>/dev/null; then
  echo "[$(ts)] ERROR: another Housing Gemma exemplar full launch appears to hold $LAUNCH_LOCK_DIR" >&2
  if [[ -f "$LAUNCH_LOCK_DIR/metadata" ]]; then
    echo "[$(ts)] Existing lock metadata:" >&2
    sed 's/^/[lock] /' "$LAUNCH_LOCK_DIR/metadata" >&2
  fi
  echo "[$(ts)] Remove the lock only after verifying no matching exemplar full run is still active." >&2
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
  echo "questions=$QUESTIONS"
  echo "seed=$SEED"
  echo "max_k=$MAX_K"
  echo "command=$0 $*"
} > "$LAUNCH_LOCK_DIR/metadata"
echo "[$(ts)] acquired exemplar full lock: $LAUNCH_LOCK_DIR"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export NO_SILENT_FALLBACK=1
export EVAL_GENERATION_FORMAT_RETRY="${EVAL_GENERATION_FORMAT_RETRY:-1}"
export EVAL_FINAL_FORMAT_RETRY="${EVAL_FINAL_FORMAT_RETRY:-1}"
export LLM_MAX_COMPLETION_TOKENS="${LLM_MAX_COMPLETION_TOKENS:-2048}"
export EVAL_MIN_COMPLETION_TOKENS="${EVAL_MIN_COMPLETION_TOKENS:-2048}"
export OPENROUTER_PROVIDER_ONLY="${OPENROUTER_PROVIDER_ONLY:-Cloudflare}"
export PYTHONUNBUFFERED=1

python3 scripts/check_expected_provider_model.py \
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
echo "[$(ts)] checking core 9-cell gate before exemplar scale-up"
python3 scripts/audit_housing_statefilter_goal.py

echo
echo "[$(ts)] confirming q500 exemplar answer gate before full-N scale-up"
scripts/local/run_housing_gemma_exemplar_q500_answer_gate.sh

mode="snap_hyre_exemplar"
gen="$GEN_ROOT/housing_q${QUESTIONS}_seed${SEED}_${MODEL_LABEL}_${mode}_realpassage.jsonl"
ret="$RET_ROOT/housing_q${QUESTIONS}_seed${SEED}_statefilter_${MODEL_LABEL}_${mode}_realpassage_k${MAX_K}.jsonl"
doc="$DOC_ROOT/housing_q${QUESTIONS}_seed${SEED}_statefilter_${MODEL_LABEL}_${mode}_realpassage_k${MAX_K}_doc_cache.jsonl"

echo
echo "[$(ts)] generation cache mode=$mode out=$gen route=OPENROUTER_PROVIDER_ONLY=$OPENROUTER_PROVIDER_ONLY"
LLM_PROVIDER="$PROVIDER" \
"$UV" run python scripts/build_generation_cache.py \
  --mode "$mode" \
  --provider "$PROVIDER" \
  --dataset housing \
  --questions "$QUESTIONS" \
  --seed "$SEED" \
  --tag "housing-gemma-full-${mode}" \
  --out "$gen" \
  --resume \
  --trace-calls \
  --trace-events

"$UV" run python - "$gen" <<'PY'
import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, errors="ignore") if line.strip()]

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

errors = [r for r in rows if r.get("error")]
missing = [r for r in rows if not r.get("hyde_passage")]
fallbacks = [r for r in rows if truthy_fallback(r)]
parse_fail = [r for r in rows if r.get("hyde_parse_ok") is False or r.get("snap_hyre_parse_ok") is False]
missing_snap = [r for r in rows if not r.get("snap_letter")]
artifacts = [r for r in rows if r.get("hyde_contains_answer_artifact") is True]
style_missing = [r for r in rows if r.get("passage_style_signal_used") is not True]
print(f"[postcheck] generation rows={len(rows)} errors={len(errors)} missing_hyde={len(missing)} fallbacks={len(fallbacks)} parse_fail={len(parse_fail)} missing_snap={len(missing_snap)} answer_artifacts={len(artifacts)} style_missing={len(style_missing)}")
if len(rows) != 6853:
    raise SystemExit(f"generation cache row-count mismatch rows={len(rows)} expected=6853")
for name, bad in [
    ("errors", errors),
    ("missing_hyde", missing),
    ("fallbacks", fallbacks),
    ("parse_fail", parse_fail),
    ("missing_snap", missing_snap),
    ("answer_artifacts", artifacts),
    ("style_missing", style_missing),
]:
    if bad:
        raise SystemExit(f"{name}: " + ",".join(str(r.get("label")) for r in bad[:10]))
PY

echo "[$(ts)] state-filter retrieval cache mode=$mode out=$ret"
"$UV" run python scripts/build_retrieval_cache.py \
  --dataset housing \
  --questions "$QUESTIONS" \
  --seed "$SEED" \
  --query-type hyre_cache \
  --label-prefix "$mode" \
  --hyre-cache-path "$gen" \
  --expected-provider "$PROVIDER" \
  --max-k "$MAX_K" \
  --housing-state-filter \
  --out "$ret" \
  --resume \
  --progress-interval 100

echo "[$(ts)] document cache mode=$mode out=$doc"
"$UV" run python scripts/build_retrieval_doc_cache.py \
  --retrieval-cache "$ret" \
  --include-effective \
  --out "$doc" \
  --resume \
  --strict \
  --batch-size 500

RETRIEVAL_DOC_CACHE_PATH="$doc" RETRIEVAL_DOC_CACHE_STRICT=1 \
"$UV" run python scripts/smoke_retrieval_cache_hydration.py \
  --cache "$ret" \
  --dataset housing \
  --label-prefix "$mode" \
  --questions "$QUESTIONS" \
  --seed "$SEED" \
  --retrieval-k "$RETRIEVAL_K" \
  --limit 20 \
  --housing-state-filter \
  --require-doc-cache

echo "[$(ts)] answer row mode=$mode"
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
NO_SILENT_FALLBACK=1 \
EVAL_FINAL_FORMAT_RETRY=1 \
EVAL_GENERATION_FORMAT_RETRY=1 \
LLM_MAX_COMPLETION_TOKENS="$LLM_MAX_COMPLETION_TOKENS" \
EVAL_MIN_COMPLETION_TOKENS="$EVAL_MIN_COMPLETION_TOKENS" \
PROVIDER="$PROVIDER" \
MODEL_LABEL="$MODEL_LABEL" \
DATASET=housing \
QUESTIONS="$QUESTIONS" \
SEED="$SEED" \
CACHE_SCOPE="q${QUESTIONS}_seed${SEED}_statefilter" \
RETRIEVAL_K="$RETRIEVAL_K" \
MODES="$mode" \
USE_CACHES=1 \
REQUIRE_RETRIEVAL_CACHES=1 \
EVAL_HOUSING_STATE_FILTER=1 \
GENERATION_CACHE_ROOT="$GEN_ROOT" \
HYRE_CACHE_ROOT="$GEN_ROOT" \
RETRIEVAL_CACHE_ROOT="$RET_ROOT" \
RETRIEVAL_DOC_CACHE_PATH="$doc" \
RETRIEVAL_DOC_CACHE_STRICT=1 \
scripts/local/run_answer_cell.sh

echo "[$(ts)] auditing full exemplar row"
SIGNOFF_SNIPPETS_OUT="$SIGNOFF_SNIPPETS_OUT" scripts/local/audit_housing_gemma_exemplar_full.sh

echo "[$(ts)] finalizing full exemplar signoff row"
MODES=snap_hyre_exemplar scripts/local/finalize_housing_gemma_signoff.sh

echo "[$(ts)] Housing Gemma full exemplar complete"
