#!/bin/bash
# Cluster-side status/audit helper for adaptive HyRE jobs.

set -euo pipefail

LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
LOCAL_LOG_DIR=${LOCAL_LOG_DIR:-logs}
USER_NAME=${USER_NAME:-$(whoami)}

echo "== SLURM queue =="
if command -v squeue >/dev/null 2>&1; then
  squeue -u "$USER_NAME" -o "%.18i %.9P %.32j %.8T %.10M %.6D %R" | grep -E "JOBID|hyre|adaptive" || true
else
  echo "squeue not found"
fi

echo
echo "== Recent adaptive HyRE SLURM stdout =="
if [[ -d "$LOG_DIR" ]]; then
  ls -t "$LOG_DIR"/*.out 2>/dev/null \
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
  ls -t "$LOCAL_LOG_DIR"/eval_adaptive_snap_hyre_*_detail.jsonl "$LOCAL_LOG_DIR"/eval_snap_hyre_*_detail.jsonl 2>/dev/null \
    | head -12 \
    | while read -r detail; do
        if head -n 1 "$detail" | grep -q '"dataset": "musique"'; then
          continue
        fi
        echo "-- $detail"
        python scripts/analyze_detail_flags.py "$detail" || true
        python scripts/audit_adaptive_hyre_logs.py "$detail" || true
      done
else
  echo "missing LOCAL_LOG_DIR=$LOCAL_LOG_DIR"
fi

echo
echo "== Sweep summary, non-smoke logs only =="
python scripts/postprocess_adaptive_hyre_sweep.py --min-n 20 || true
