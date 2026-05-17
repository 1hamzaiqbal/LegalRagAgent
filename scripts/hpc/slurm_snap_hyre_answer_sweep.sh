#!/bin/bash
# Run the canonical Snap-HyRE answer ladder for one dataset/model cell.
#
# Keep this deliberately narrow: submit one dataset/model at a time after the
# top-k cache diagnostics choose RETRIEVAL_K.

#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus 1
#SBATCH --exclude=r28-1801,a100-2207,a100s-2305,a100s-2306,a100s-2307,a100s-2308
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -J snap-hyre-answer
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

set -euo pipefail

REPO=${REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-snap-hyre-comprehensive}
DATA_REPO=${DATA_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
GEMMA_VENV=${GEMMA_VENV:-/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4}
EVAL_VENV=${EVAL_VENV:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-$DATA_REPO/chroma_db}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
HF_CACHE=${HF_CACHE:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}
XDG_CACHE_HOME=${XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache}
TORCH_HOME=${TORCH_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/cache/torch}
HYRE_CACHE_ROOT=${HYRE_CACHE_ROOT:-$REPO/caches/hyre/full}
RETRIEVAL_CACHE_ROOT=${RETRIEVAL_CACHE_ROOT:-$REPO/caches/retrieval/full}
BACKEND=${BACKEND:-api}
PROVIDER=${PROVIDER:-groq-llama70b}
MODEL=${MODEL:-}
MODEL_LABEL=${MODEL_LABEL:-$PROVIDER}
PORT=${PORT:-8014}
DATASET=${DATASET:-barexam}
QUESTIONS=${QUESTIONS:-full}
SEED=${SEED:-42}
RETRIEVAL_K=${RETRIEVAL_K:-5}
USE_CACHES=${USE_CACHES:-1}
STOP_ON_FAILURE=${STOP_ON_FAILURE:-1}
LLM_MAX_COMPLETION_TOKENS=${LLM_MAX_COMPLETION_TOKENS:-2048}
EVAL_MIN_COMPLETION_TOKENS=${EVAL_MIN_COMPLETION_TOKENS:-2048}
EVAL_FINAL_FORMAT_RETRY=${EVAL_FINAL_FORMAT_RETRY:-1}
EVAL_GENERATION_FORMAT_RETRY=${EVAL_GENERATION_FORMAT_RETRY:-1}
NO_SILENT_FALLBACK=${NO_SILENT_FALLBACK:-1}

if ! [[ "$LLM_MAX_COMPLETION_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "LLM_MAX_COMPLETION_TOKENS must be a positive integer, got $LLM_MAX_COMPLETION_TOKENS" >&2
  exit 2
fi
if ! [[ "$EVAL_MIN_COMPLETION_TOKENS" =~ ^[0-9]+$ ]]; then
  echo "EVAL_MIN_COMPLETION_TOKENS must be a positive integer, got $EVAL_MIN_COMPLETION_TOKENS" >&2
  exit 2
fi
if (( LLM_MAX_COMPLETION_TOKENS < EVAL_MIN_COMPLETION_TOKENS )); then
  echo "LLM_MAX_COMPLETION_TOKENS=$LLM_MAX_COMPLETION_TOKENS is below EVAL_MIN_COMPLETION_TOKENS=$EVAL_MIN_COMPLETION_TOKENS; refusing truncation-prone answer run" >&2
  exit 2
fi
case "${NO_SILENT_FALLBACK,,}" in
  1|true|yes|on) ;;
  *) echo "NO_SILENT_FALLBACK must be enabled for answer runs, got $NO_SILENT_FALLBACK" >&2; exit 2 ;;
esac

if [[ -n "${MODES:-}" ]]; then
  # shellcheck disable=SC2206
  MODES_ARR=(${MODES})
else
  MODES_ARR=(llm_only rag_simple rag_hyde snap_hyre golden_passage golden_plus_neighbors rag_rewrite)
fi

mkdir -p "$LOG_DIR" "$HF_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$REPO/logs"
ln -sfn "$DATA_REPO/datasets" "$REPO/datasets"
ln -sfn "$DATA_REPO/chroma_db" "$REPO/chroma_db"
cd "$REPO"

export HUGGINGFACE_HUB_CACHE="$HF_CACHE"
export HF_HOME="$HF_CACHE"
export SENTENCE_TRANSFORMERS_HOME="$HF_CACHE"
export XDG_CACHE_HOME="$XDG_CACHE_HOME"
export TORCH_HOME="$TORCH_HOME"
export TRITON_CACHE_DIR="/tmp/hiqbal-triton/${SLURM_JOB_ID:-local}"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
mkdir -p "$TRITON_CACHE_DIR" "$UV_CACHE_DIR"
export CHROMA_DB_DIR="$CHROMA_DB_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
export LLM_MAX_COMPLETION_TOKENS
export EVAL_FINAL_FORMAT_RETRY
export EVAL_GENERATION_FORMAT_RETRY
export NO_SILENT_FALLBACK

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[$(date -Is)] repo=$REPO commit=$(git rev-parse --short HEAD)"
echo "[$(date -Is)] backend=$BACKEND provider=$PROVIDER model=${MODEL:-none} model_label=$MODEL_LABEL"
echo "[$(date -Is)] dataset=$DATASET questions=$QUESTIONS retrieval_k=$RETRIEVAL_K modes=${MODES_ARR[*]} use_caches=$USE_CACHES"
echo "[$(date -Is)] no_silent_fallback=$NO_SILENT_FALLBACK"
echo "[$(date -Is)] llm_max_completion_tokens=$LLM_MAX_COMPLETION_TOKENS eval_min_completion_tokens=$EVAL_MIN_COMPLETION_TOKENS eval_final_format_retry=$EVAL_FINAL_FORMAT_RETRY eval_generation_format_retry=$EVAL_GENERATION_FORMAT_RETRY"
git status --short --branch

if [[ "$BACKEND" == "vllm" ]]; then
  if [[ -z "$MODEL" ]]; then
    echo "[$(date -Is)] ERROR: MODEL is required for BACKEND=vllm"
    exit 2
  fi
  "$GEMMA_VENV/bin/vllm" serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization 0.8 \
    --max-model-len 8192 > "$LOG_DIR/vllm_snap_hyre_answer_${SLURM_JOB_ID}.log" 2>&1 &
  VLLM_PID=$!

  READY=0
  for _ in $(seq 1 240); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      READY=1
      break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[$(date -Is)] ERROR: vLLM died during startup"
      tail -100 "$LOG_DIR/vllm_snap_hyre_answer_${SLURM_JOB_ID}.log" || true
      exit 1
    fi
    sleep 5
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[$(date -Is)] ERROR: vLLM did not become ready"
    tail -100 "$LOG_DIR/vllm_snap_hyre_answer_${SLURM_JOB_ID}.log" || true
    exit 1
  fi
  export LLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
  export LLM_API_KEY=DUMMY_KEY
  export LLM_MODEL="$MODEL"
fi

source "$EVAL_VENV/bin/activate"

python -m py_compile \
  eval/eval_config.py \
  eval/eval_harness.py \
  scripts/analyze_detail_flags.py

add_cache_args_for_mode() {
  local mode="$1"
  local hyre_cache=""
  local retrieval_cache=""
  case "$mode" in
    rag_simple)
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_raw_question_k10.jsonl"
      ;;
    rag_hyde)
      hyre_cache="$HYRE_CACHE_ROOT/${DATASET}_${MODEL_LABEL}_rag_hyde.jsonl"
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_${MODEL_LABEL}_rag_hyde_k10.jsonl"
      ;;
    snap_hyre)
      hyre_cache="$HYRE_CACHE_ROOT/${DATASET}_${MODEL_LABEL}_snap_hyre.jsonl"
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_${MODEL_LABEL}_snap_hyre_k10.jsonl"
      ;;
    golden_plus_neighbors)
      retrieval_cache="$RETRIEVAL_CACHE_ROOT/${DATASET}_golden_neighbors_k10.jsonl"
      ;;
  esac

  if [[ "$USE_CACHES" != "1" ]]; then
    return 0
  fi
  if [[ -n "$hyre_cache" ]]; then
    [[ -f "$hyre_cache" ]] || { echo "missing hyre cache $hyre_cache" >&2; return 2; }
    extra_args+=(--hyre-cache-path "$hyre_cache")
  fi
  if [[ -n "$retrieval_cache" ]]; then
    [[ -f "$retrieval_cache" ]] || { echo "missing retrieval cache $retrieval_cache" >&2; return 2; }
    extra_args+=(--retrieval-cache-path "$retrieval_cache")
  fi
}

for mode in "${MODES_ARR[@]}"; do
  tag="snap-hyre-answer-${MODEL_LABEL}-${DATASET}-k${RETRIEVAL_K}-${mode}-job${SLURM_JOB_ID}"
  echo
  echo "[$(date -Is)] run dataset=$DATASET mode=$mode tag=$tag"
  extra_args=()
  if ! add_cache_args_for_mode "$mode"; then
    echo "[$(date -Is)] FAILED dataset=$DATASET mode=$mode while resolving caches"
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit 2
    fi
    continue
  fi

  set +e
  LLM_PROVIDER="$PROVIDER" \
  NO_SILENT_FALLBACK="$NO_SILENT_FALLBACK" \
  python eval/eval_harness.py \
    --mode "$mode" \
    --provider "$PROVIDER" \
    --dataset "$DATASET" \
    --questions "$QUESTIONS" \
    --seed "$SEED" \
    --retrieval-k "$RETRIEVAL_K" \
    --tag "$tag" \
    "${extra_args[@]}"
  status=$?
  set -e

  latest_log=$(
    find "$REPO/logs" -maxdepth 1 \
      -name "eval_${mode}_${PROVIDER}_*_${DATASET}_*${tag}*_detail.jsonl" \
      -print 2>/dev/null | sort | tail -n 1
  )
  if [[ -z "$latest_log" ]]; then
    echo "[$(date -Is)] ERROR: no detail log found for mode=$mode tag=$tag"
    status=1
  else
    python scripts/analyze_detail_flags.py "$latest_log" || status=1
    python - "$latest_log" "$mode" <<'PY' || status=1
import json
import os
import re
import sys

path = sys.argv[1]
mode = sys.argv[2]
bad = []
errors = []
fallbacks = []
parse_fallbacks = []
think_tags = []
missing_answer_marker = []
long_answers = []
near_cap_outputs = []
missing_retrieval_cache = []
missing_hyre_cache = []
oracle_missing = []
cache_required_modes = {"rag_simple", "rag_hyde", "snap_hyre", "golden_plus_neighbors"}
hyre_required_modes = {"rag_hyde", "snap_hyre"}
oracle_modes = {"golden_passage", "golden_plus_neighbors"}
use_caches = os.getenv("USE_CACHES", "1") == "1"
max_final_answer_chars = int(os.getenv("EVAL_MAX_FINAL_ANSWER_CHARS", "20000"))
max_completion_tokens = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "0") or "0")
output_token_margin = int(os.getenv("EVAL_OUTPUT_TOKEN_MARGIN", "16"))
answer_marker = re.compile(r"(?im)^\s*(?:\*\*)?(?:final\s+)?answer(?:\*\*)?\s*:")

def has_required_final_line(row, text):
    dataset = str(row.get("dataset") or "")
    pred = str(row.get("predicted_answer") or "").strip()
    if dataset == "housing":
        if pred.lower() == "yes":
            target = "Answer: Yes"
        elif pred.lower() == "no":
            target = "Answer: No"
        else:
            return False
    elif dataset in {"barexam", "housing", "casehold", "legalbench_scalr"}:
        pred = pred.upper()
        if pred not in {"A", "B", "C", "D", "E"}:
            return False
        target = f"Answer: ({pred})"
    else:
        return True
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return bool(lines) and lines[-1] == target

with open(path) as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = str(row.get("label") or row.get("idx") or line_no)
        pred = row.get("predicted_answer")
        if pred is None or str(pred).strip() == "":
            bad.append(row_id)
        if row.get("error"):
            errors.append(row_id)
        routed_to = str(row.get("routed_to") or "")
        if "fallback" in routed_to.lower():
            fallbacks.append(f"{row_id}:routed_to={routed_to}")
        for key, value in row.items():
            if (key.endswith("_fallback") or key.endswith("_used_fallback")) and value:
                fallbacks.append(f"{row_id}:{key}={value}")
        for key in ("hyde_parse_ok", "snap_hyre_parse_ok", "snap_hyde_2call_parse_ok", "choice_hyre_parse_ok", "rewrite_parse_ok", "route_parse_ok", "passage_parse_ok", "adaptive_parse_ok"):
            if row.get(key) is False:
                parse_fallbacks.append(f"{row_id}:{key}")
        final_answer = str(row.get("final_answer") or "")
        if "<think>" in final_answer.lower():
            think_tags.append(row_id)
        dataset = str(row.get("dataset") or "")
        if dataset in {"barexam", "housing", "casehold", "legalbench_scalr"} and not answer_marker.search(final_answer):
            missing_answer_marker.append(row_id)
        elif dataset in {"barexam", "housing", "casehold", "legalbench_scalr"} and not has_required_final_line(row, final_answer):
            missing_answer_marker.append(f"{row_id}:missing_required_final_answer_line")
        if max_final_answer_chars > 0 and len(final_answer) > max_final_answer_chars:
            long_answers.append(f"{row_id}:{len(final_answer)}")
        if (
            max_completion_tokens > 0
            and int(row.get("llm_calls") or 0) <= 1
            and int(row.get("output_tokens") or 0) >= max(1, max_completion_tokens - output_token_margin)
        ):
            near_cap_outputs.append(f"{row_id}:{row.get('output_tokens')}")
        retry_output_tokens = int(row.get("answer_format_retry_output_tokens") or 0)
        if (
            max_completion_tokens > 0
            and retry_output_tokens >= max(1, max_completion_tokens - output_token_margin)
        ):
            near_cap_outputs.append(f"{row_id}:retry={retry_output_tokens}")
        if use_caches and mode in cache_required_modes and row.get("retrieval_cache_hit") is not True:
            missing_retrieval_cache.append(row_id)
        if use_caches and mode in hyre_required_modes and row.get("hyre_cache_hit") is not True:
            missing_hyre_cache.append(row_id)
        if mode in oracle_modes and (row.get("gold_retrieved") is not True or not row.get("evidence_store")):
            oracle_missing.append(row_id)
if bad:
    raise SystemExit("missing predicted_answer rows: " + ",".join(bad[:10]))
if errors:
    raise SystemExit("error rows: " + ",".join(errors[:10]))
if fallbacks:
    raise SystemExit("fallback marker rows: " + " | ".join(fallbacks[:10]))
if parse_fallbacks:
    raise SystemExit("parse fallback rows: " + ",".join(parse_fallbacks[:10]))
if think_tags:
    raise SystemExit("unclosed think tag rows: " + ",".join(think_tags[:10]))
if missing_answer_marker:
    raise SystemExit("missing explicit Answer marker rows: " + ",".join(missing_answer_marker[:10]))
if long_answers:
    raise SystemExit("long final answer rows: " + ",".join(long_answers[:10]))
if near_cap_outputs:
    raise SystemExit("near-cap output token rows: " + ",".join(near_cap_outputs[:10]))
if missing_retrieval_cache:
    raise SystemExit("retrieval cache not hit rows: " + ",".join(missing_retrieval_cache[:10]))
if missing_hyre_cache:
    raise SystemExit("HyRE generation cache not hit rows: " + ",".join(missing_hyre_cache[:10]))
if oracle_missing:
    raise SystemExit("oracle evidence missing rows: " + ",".join(oracle_missing[:10]))
PY
  fi

  if [[ "$status" -ne 0 ]]; then
    echo "[$(date -Is)] FAILED dataset=$DATASET mode=$mode exit=$status"
    if [[ "$STOP_ON_FAILURE" == "1" ]]; then
      exit "$status"
    fi
  else
    echo "[$(date -Is)] OK dataset=$DATASET mode=$mode"
  fi
done

echo "[$(date -Is)] answer sweep complete."
