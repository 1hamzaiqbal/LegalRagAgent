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
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
SUBMIT_STAMP=${SUBMIT_STAMP:-$(date +%Y%m%d_%H%M%S)}
SUBMIT_MANIFEST=${SUBMIT_MANIFEST:-$LOG_DIR/adaptive_hyre_submit_${SUBMIT_STAMP}.tsv}

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
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  printf 'timestamp\tdataset\tjob_name\tjob_id\tn_questions\tretrieval_k\tseed\tprovider\tuse_vllm\tmodel\trun_specs\n' > "$SUBMIT_MANIFEST"
  echo "  submit_manifest=$SUBMIT_MANIFEST"
fi

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
  submit_output=$("${cmd[@]}")
  echo "$submit_output"
  job_id=$(awk '/Submitted batch job/ {print $4}' <<<"$submit_output")
  if [[ -z "$job_id" ]]; then
    echo "ERROR: could not parse sbatch job id for $job_name" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -Is)" \
    "$dataset" \
    "$job_name" \
    "$job_id" \
    "$N_QUESTIONS" \
    "$RETRIEVAL_K" \
    "$SEED" \
    "${PROVIDER:-cluster-vllm}" \
    "$USE_VLLM" \
    "$MODEL" \
    "${RUN_SPECS:-default-by-dataset}" >> "$SUBMIT_MANIFEST"
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo
  echo "Wrote submit manifest: $SUBMIT_MANIFEST"
fi
