#!/usr/bin/env bash
# Monitor a chunked HousingQA state-filter SLURM array and validate after finish.

set -euo pipefail

JOB_ID=${JOB_ID:-58937}
REMOTE=${REMOTE:-wustl}
REMOTE_REPO=${REMOTE_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent}
REMOTE_STDOUT_DIR=${REMOTE_STDOUT_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
TASK_IDS_CSV=${TASK_IDS_CSV:-4,5,6,7}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-120}
MAX_POLLS=${MAX_POLLS:-90}

LOCAL_REPO=${LOCAL_REPO:-$(git rev-parse --show-toplevel)}
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-$LOCAL_REPO/logs}
MERGED_DETAIL=${MERGED_DETAIL:-$LOCAL_LOG_DIR/eval_rag_state_filter_or-gemma4-26b_20260501_k10_merged_detail.jsonl}
MONITOR_LOG=${MONITOR_LOG:-$LOCAL_LOG_DIR/monitor_housing_state_chunks_${JOB_ID}.log}
VALIDATION_LOG=${VALIDATION_LOG:-$LOCAL_LOG_DIR/housing_state_k10_${JOB_ID}_validation.txt}

mkdir -p "$LOCAL_LOG_DIR"
cd "$LOCAL_REPO"

IFS=, read -r -a TASK_IDS <<< "$TASK_IDS_CSV"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$MONITOR_LOG"
}

remote_stdout_paths() {
  local joined=""
  for task_id in "${TASK_IDS[@]}"; do
    joined+="$REMOTE_STDOUT_DIR/${JOB_ID}_${task_id}.out "
  done
  printf '%s' "$joined"
}

poll_once() {
  ssh "$REMOTE" "squeue -j $JOB_ID -o '%.18i %.9T %.10M %.6D %R' || true" | tee -a "$MONITOR_LOG"
  ssh "$REMOTE" "for f in $(remote_stdout_paths); do
    echo ===\$f
    grep -E '\\[[0-9]+/50\\]|RESULTS|Detail log|Run summary|summary-guard|CANCELLED|TIME|Traceback|Error|429|rate|timeout|JSON|parse|empty response|format' \$f 2>/dev/null | tail -n 12
  done" | tee -a "$MONITOR_LOG"
}

running_state() {
  ssh "$REMOTE" "squeue -h -j $JOB_ID -o '%T' 2>/dev/null | tr '\n' ' '"
}

collect_detail_paths() {
  ssh "$REMOTE" "for f in $(remote_stdout_paths); do
    grep -o 'logs/eval_[^ ]*_detail.jsonl' \$f 2>/dev/null | tail -n 1
  done"
}

validate_finished() {
  mapfile -t detail_paths < <(collect_detail_paths | sed '/^$/d')
  if (( ${#detail_paths[@]} != ${#TASK_IDS[@]} )); then
    log "ERROR: expected ${#TASK_IDS[@]} detail logs, found ${#detail_paths[@]}"
    printf '%s\n' "${detail_paths[@]}" | tee -a "$MONITOR_LOG"
    exit 2
  fi

  local local_inputs=()
  for relpath in "${detail_paths[@]}"; do
    log "pulling $relpath"
    scp "$REMOTE:$REMOTE_REPO/$relpath" "$LOCAL_LOG_DIR/"
    local_inputs+=("$LOCAL_LOG_DIR/$(basename "$relpath")")
  done

  {
    echo "### merge"
    UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      uv run python scripts/merge_detail_logs.py --output "$MERGED_DETAIL" "${local_inputs[@]}"

    echo
    echo "### artifact audit"
    python scripts/analyze_detail_flags.py "$MERGED_DETAIL"

    echo
    echo "### paired: generic top-5 -> state-filter k10"
    UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      uv run python scripts/compute_mcnemar.py \
      logs/eval_rag_simple_or-gemma4-26b_20260430_0502_detail.jsonl \
      "$MERGED_DETAIL" --key idx --bootstrap-samples 2000 --seed 42

    echo
    echo "### paired: generic top-10 -> state-filter k10"
    UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      uv run python scripts/compute_mcnemar.py \
      logs/eval_rag_simple_or-gemma4-26b_20260430_0542_detail.jsonl \
      "$MERGED_DETAIL" --key idx --bootstrap-samples 2000 --seed 42

    echo
    echo "### paired: state-filter k5 -> state-filter k10"
    UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
      uv run python scripts/compute_mcnemar.py \
      logs/eval_rag_state_filter_or-gemma4-26b_20260501_1406_detail.jsonl \
      "$MERGED_DETAIL" --key idx --bootstrap-samples 2000 --seed 42
  } | tee "$VALIDATION_LOG"

  log "validation complete: $VALIDATION_LOG"
}

log "monitoring job $JOB_ID tasks ${TASK_IDS_CSV}; interval=${INTERVAL_SECONDS}s"
for ((poll = 1; poll <= MAX_POLLS; poll++)); do
  log "poll $poll/$MAX_POLLS"
  poll_once
  state=$(running_state)
  if [[ -z "${state// }" ]]; then
    log "job $JOB_ID no longer in queue; collecting detail logs"
    validate_finished
    exit 0
  fi
  sleep "$INTERVAL_SECONDS"
done

log "ERROR: max polls reached while job state remained: $(running_state)"
exit 3
