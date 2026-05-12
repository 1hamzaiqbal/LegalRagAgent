#!/bin/bash
# Strict readiness gate for the legal adaptive HyRE sweep.

set -euo pipefail

PROVIDER=${PROVIDER:-cluster-vllm}
MIN_N=${MIN_N:-20}
TAG_CONTAINS=${TAG_CONTAINS:-}
EVAL_VENV=${EVAL_VENV:-.venv}
if [[ -x "$EVAL_VENV/bin/python" ]]; then
  PYTHON_CMD=("$EVAL_VENV/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  PYTHON_CMD=(python)
fi
DATASETS=("$@")
if [[ "$#" -eq 0 ]]; then
  DATASETS=(barexam housing casehold legalbench_scalr)
fi

failures=0
for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    barexam|housing|casehold|legalbench_scalr)
      ;;
    *)
      echo "ERROR: unsupported dataset '$dataset'" >&2
      exit 2
      ;;
  esac

  echo "== $dataset | provider=$PROVIDER | min_n=$MIN_N =="
  extra_args=()
  if [[ -n "$TAG_CONTAINS" ]]; then
    extra_args+=(--tag-contains "$TAG_CONTAINS")
  fi
  if "${PYTHON_CMD[@]}" scripts/postprocess_adaptive_hyre_sweep.py \
      --min-n "$MIN_N" \
      --dataset "$dataset" \
      --provider "$PROVIDER" \
      "${extra_args[@]}" \
      --require-ready >/tmp/adaptive_hyre_ready_"$dataset".out 2>/tmp/adaptive_hyre_ready_"$dataset".err; then
    echo "READY"
  else
    failures=$((failures + 1))
    cat /tmp/adaptive_hyre_ready_"$dataset".err
  fi
done

if [[ "$failures" -gt 0 ]]; then
  echo
  echo "NOT READY: $failures dataset(s) failed adaptive HyRE readiness."
  exit 1
fi

echo
echo "READY: all requested adaptive HyRE dataset gates passed."
