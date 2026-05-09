#!/bin/bash
# Submit the legal-only adaptive HyRE sweep on the cluster.

set -euo pipefail

SCRIPT=${SCRIPT:-scripts/hpc/slurm_adaptive_hyre_legal.sh}
N_QUESTIONS=${N_QUESTIONS:-200}
RETRIEVAL_K=${RETRIEVAL_K:-5}
SEED=${SEED:-42}
USE_VLLM=${USE_VLLM:-1}
MODEL=${MODEL:-google/gemma-4-26B-A4B-it}

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(barexam housing casehold legalbench_scalr)
fi

echo "Submitting adaptive HyRE legal sweep"
echo "  script=$SCRIPT"
echo "  datasets=${DATASETS[*]}"
echo "  n=$N_QUESTIONS seed=$SEED k=$RETRIEVAL_K use_vllm=$USE_VLLM model=$MODEL"

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
  echo
  echo "Submitting $job_name"
  sbatch --job-name="$job_name" \
    --export=ALL,DATASET="$dataset",N_QUESTIONS="$N_QUESTIONS",RETRIEVAL_K="$RETRIEVAL_K",SEED="$SEED",USE_VLLM="$USE_VLLM",MODEL="$MODEL" \
    "$SCRIPT"
done
