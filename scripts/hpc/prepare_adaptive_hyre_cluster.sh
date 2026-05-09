#!/bin/bash
# Verify the cluster checkout is ready before submitting adaptive HyRE jobs.

set -euo pipefail

EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/final-report-snap-hyde}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}

echo "== Git checkout =="
branch=$(git rev-parse --abbrev-ref HEAD)
commit=$(git rev-parse --short HEAD)
echo "branch=$branch"
echo "commit=$commit"

if [[ "$branch" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected branch $EXPECTED_BRANCH, found $branch" >&2
  echo "Run: git fetch origin && git switch $EXPECTED_BRANCH && git pull --ff-only" >&2
  exit 2
fi

if [[ -n "$EXPECTED_COMMIT" ]]; then
  if [[ "$commit" != "$EXPECTED_COMMIT" ]]; then
    echo "ERROR: expected commit $EXPECTED_COMMIT, found $commit" >&2
    echo "Run: git pull --ff-only" >&2
    exit 2
  fi
fi

if [[ "$ALLOW_DIRTY" != "1" ]]; then
  dirty=$(git status --short)
  if [[ -n "$dirty" ]]; then
    echo "ERROR: checkout is dirty; set ALLOW_DIRTY=1 only if intentional" >&2
    git status --short >&2
    exit 2
  fi
fi

echo
echo "== Required files =="
required=(
  eval/eval_harness.py
  eval/eval_config.py
  scripts/hpc/slurm_adaptive_hyre_legal.sh
  scripts/hpc/submit_adaptive_hyre_legal_sweep.sh
  scripts/hpc/monitor_adaptive_hyre_sweep.sh
  scripts/audit_adaptive_hyre_logs.py
  scripts/postprocess_adaptive_hyre_sweep.py
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing $path" >&2
    exit 2
  fi
  echo "OK $path"
done

echo
echo "== Mode preflight =="
python - <<'PY'
from eval.eval_config import EVAL_MODES

required = [
    "snap_hyre_option",
    "snap_hyre_state",
    "adaptive_snap_hyre",
    "adaptive_snap_hyre_anchor",
]
missing = [mode for mode in required if mode not in EVAL_MODES]
if missing:
    raise SystemExit("ERROR missing eval modes: " + ", ".join(missing))
print("OK modes: " + ", ".join(required))
PY

echo
echo "== Script syntax =="
bash -n scripts/hpc/slurm_adaptive_hyre_legal.sh
bash -n scripts/hpc/submit_adaptive_hyre_legal_sweep.sh
bash -n scripts/hpc/monitor_adaptive_hyre_sweep.sh
python -m py_compile scripts/audit_adaptive_hyre_logs.py scripts/postprocess_adaptive_hyre_sweep.py
echo "OK syntax"

echo
echo "== Submit dry run =="
DRY_RUN=1 scripts/hpc/submit_adaptive_hyre_legal_sweep.sh "$@"

echo
echo "READY: adaptive HyRE cluster checkout is prepared."
