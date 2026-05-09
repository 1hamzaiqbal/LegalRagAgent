#!/bin/bash
# Submit the legal-only adaptive HyRE sweep on the cluster.

set -euo pipefail

SCRIPT=${SCRIPT:-scripts/hpc/slurm_adaptive_hyre_legal.sh}
N_QUESTIONS=${N_QUESTIONS:-200}
RETRIEVAL_K=${RETRIEVAL_K:-5}
SEED=${SEED:-42}
USE_VLLM=${USE_VLLM:-1}
MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
if [[ "${PROVIDER:-}" == "" && "$USE_VLLM" == "0" ]]; then
  PROVIDER=or-gemma4-26b
fi
DRY_RUN=${DRY_RUN:-0}

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(barexam housing casehold legalbench_scalr)
fi

echo "Submitting adaptive HyRE legal sweep"
echo "  script=$SCRIPT"
echo "  datasets=${DATASETS[*]}"
echo "  n=$N_QUESTIONS seed=$SEED k=$RETRIEVAL_K use_vllm=$USE_VLLM model=$MODEL provider=${PROVIDER:-cluster-vllm} dry_run=$DRY_RUN"
echo "  run_specs=${RUN_SPECS:-default-by-dataset}"

for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    barexam|housing|casehold|legalbench_scalr)
      ;;
    *)
      echo "ERROR: unsupported dataset '$dataset' (expected barexam, housing, casehold, legalbench_scalr)" >&2
      exit 2
      ;;
  esac

  job_name="hyre-${dataset}"
  exports="ALL,DATASET=${dataset},N_QUESTIONS=${N_QUESTIONS},RETRIEVAL_K=${RETRIEVAL_K},SEED=${SEED},USE_VLLM=${USE_VLLM},MODEL=${MODEL}"
  if [[ "${PROVIDER:-}" != "" ]]; then
    exports="${exports},PROVIDER=${PROVIDER}"
  fi
  if [[ "${RUN_SPECS:-}" != "" ]]; then
    exports="${exports},RUN_SPECS=${RUN_SPECS}"
  fi
  cmd=(
    sbatch
    --job-name="$job_name"
    --export="$exports"
    "$SCRIPT"
  )
  echo
  echo "Submitting $job_name"
  printf '  '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    continue
  fi
  "${cmd[@]}"
done
