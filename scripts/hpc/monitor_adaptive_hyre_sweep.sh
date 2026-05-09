#!/bin/bash
# Cluster-side status/audit helper for adaptive HyRE jobs.

set -euo pipefail

LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-logs}
USER_NAME=${USER_NAME:-$(whoami)}
PROVIDER=${PROVIDER:-cluster-vllm}
EVAL_VENV=${EVAL_VENV:-.venv}
if [[ -x "$EVAL_VENV/bin/python" ]]; then
  PYTHON_CMD=("$EVAL_VENV/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi

echo "== SLURM queue =="
if command -v squeue >/dev/null 2>&1; then
  squeue -u "$USER_NAME" -o "%.18i %.9P %.32j %.8T %.10M %.6D %R" | grep -E "JOBID|hyre|adaptive" || true
else
  echo "squeue not found"
fi

echo
echo "== Recent adaptive HyRE SLURM stdout =="
if [[ -d "$LOG_DIR" ]]; then
  { ls -t "$LOG_DIR"/*.out 2>/dev/null || true; } \
    | head -20 \
    | while read -r path; do
        if grep -qiE "adaptive|hyre|Traceback|ERROR|rate|timeout|empty" "$path"; then
          echo "-- $path"
          tail -80 "$path"
        fi
      done
else
  echo "missing LOG_DIR=$LOG_DIR"
fi

echo
echo "== Recent local adaptive detail logs =="
if [[ -d "$LOCAL_LOG_DIR" ]]; then
  { ls -t "$LOCAL_LOG_DIR"/eval_adaptive_snap_hyre_*_detail.jsonl "$LOCAL_LOG_DIR"/eval_snap_hyre_*_detail.jsonl 2>/dev/null || true; } \
    | head -12 \
    | while read -r detail; do
        if head -n 1 "$detail" | grep -q '"dataset": "musique"'; then
          continue
        fi
        echo "-- $detail"
        "${PYTHON_CMD[@]}" scripts/analyze_detail_flags.py "$detail" || true
        "${PYTHON_CMD[@]}" scripts/audit_adaptive_hyre_logs.py "$detail" || true
      done
else
  echo "missing LOCAL_LOG_DIR=$LOCAL_LOG_DIR"
fi

echo
echo "== Sweep summary, non-smoke logs only =="
"${PYTHON_CMD[@]}" scripts/postprocess_adaptive_hyre_sweep.py --min-n 20 --provider "$PROVIDER" || true

echo
echo "== Adaptive readiness by dataset =="
PROVIDER="$PROVIDER" MIN_N=20 EVAL_VENV="$EVAL_VENV" scripts/check_adaptive_hyre_readiness.sh || true

echo
echo "== Recent persisted adaptive summaries =="
if [[ -d "$LOG_DIR" ]]; then
  echo
  echo "== Recent submit manifests =="
  { ls -t "$LOG_DIR"/adaptive_hyre_submit_*.tsv 2>/dev/null || true; } \
    | head -5 \
    | while read -r manifest; do
        echo "-- $manifest"
        column -t -s $'\t' "$manifest" 2>/dev/null || cat "$manifest"
      done
  echo
  echo "== Submitted adaptive job states =="
  if command -v squeue >/dev/null 2>&1; then
    latest_manifest=$({ ls -t "$LOG_DIR"/adaptive_hyre_submit_*.tsv 2>/dev/null || true; } | head -1)
    if [[ -n "$latest_manifest" ]]; then
      awk -F '\t' 'NR > 1 && $4 != "" {print $4}' "$latest_manifest" \
        | paste -sd, - \
        | while read -r job_ids; do
            if [[ -n "$job_ids" ]]; then
              squeue -j "$job_ids" -o "%.18i %.32j %.8T %.10M %.6D %R" || true
            fi
          done
    else
      echo "no adaptive submit manifest found"
    fi
  else
    echo "squeue not found"
  fi
  { ls -t "$LOG_DIR"/adaptive_hyre_*.md 2>/dev/null || true; } \
    | head -5 \
    | while read -r summary; do
        echo "-- $summary"
        sed -n '1,120p' "$summary"
      done
  echo
  echo "== Recent adaptive JSON summary statuses =="
  { ls -t "$LOG_DIR"/adaptive_hyre_*.json 2>/dev/null || true; } \
    | head -5 \
    | while read -r summary_json; do
        echo "-- $summary_json"
        "${PYTHON_CMD[@]}" - "$summary_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    payload = json.load(f)
for record in payload.get("adaptive_parity_frontier", []):
    print(
        f"{record['dataset']} {record['provider'] or '-'} "
        f"status={record['status']} delta_pp={record['delta_pp']}"
    )
PY
      done
else
  echo "missing LOG_DIR=$LOG_DIR"
fi
