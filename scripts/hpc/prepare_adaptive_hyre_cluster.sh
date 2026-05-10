#!/bin/bash
# Verify the cluster checkout is ready before submitting adaptive HyRE jobs.

set -euo pipefail

EXPECTED_BRANCH=${EXPECTED_BRANCH:-codex/final-report-snap-hyde}
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
ALLOW_DIRTY=${ALLOW_DIRTY:-0}
CHECK_CHROMA=${CHECK_CHROMA:-1}
CHROMA_DB_DIR=${CHROMA_DB_DIR:-chroma_db}
EVAL_VENV=${EVAL_VENV:-.venv}

if [[ -x "$EVAL_VENV/bin/python" ]]; then
  PYTHON_CMD=("$EVAL_VENV/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi

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
  scripts/hpc/launch_adaptive_hyre_sweep.sh
  scripts/hpc/submit_adaptive_hyre_legal_sweep.sh
  scripts/hpc/monitor_adaptive_hyre_sweep.sh
  scripts/audit_adaptive_hyre_logs.py
  scripts/postprocess_adaptive_hyre_sweep.py
  scripts/smoke_adaptive_hyre_modes.py
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
"${PYTHON_CMD[@]}" - <<'PY'
from eval.eval_config import EVAL_MODES

required = [
    "snap_hyre_option",
    "snap_hyre_state",
    "adaptive_snap_hyre",
    "adaptive_snap_hyre_anchor",
    "adaptive_snap_hyre_diverse",
    "adaptive_snap_hyre_v2",
    "adaptive_snap_hyre_frontier",
    "adaptive_snap_hyre_stability",
    "adaptive_snap_hyre_housing_verifier",
    "adaptive_snap_hyre_candidate_verifier",
    "adaptive_snap_hyre_option_reranker",
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
"${PYTHON_CMD[@]}" -m py_compile scripts/audit_adaptive_hyre_logs.py scripts/postprocess_adaptive_hyre_sweep.py
echo "OK syntax"

echo
echo "== Adaptive HyRE smoke =="
"${PYTHON_CMD[@]}" scripts/smoke_adaptive_hyre_modes.py

if [[ "$CHECK_CHROMA" == "1" ]]; then
  echo
  echo "== Chroma collections =="
  "${PYTHON_CMD[@]}" - <<'PY'
import os

import chromadb

from eval.eval_harness import DATASET_COLLECTIONS

datasets = ("barexam", "housing", "casehold", "legalbench_scalr")
db_dir = os.environ.get("CHROMA_DB_DIR", "chroma_db")
client = chromadb.PersistentClient(path=db_dir)
for dataset in datasets:
    collection_name = DATASET_COLLECTIONS[dataset]
    collection = client.get_collection(collection_name)
    count = collection.count()
    print(f"OK {dataset}: {collection_name} has {count:,} docs")
    if count <= 0:
        raise SystemExit(f"ERROR {collection_name} collection is empty")
PY
else
  echo
  echo "== Chroma collections =="
  echo "SKIP CHECK_CHROMA=$CHECK_CHROMA"
fi

echo
echo "== Submit dry run =="
DRY_RUN=1 scripts/hpc/submit_adaptive_hyre_legal_sweep.sh "$@"

echo
echo "READY: adaptive HyRE cluster checkout is prepared."
