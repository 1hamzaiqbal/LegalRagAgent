#!/bin/bash
# Submit adaptive HyRE as one SLURM job per dataset/mode.
#
# This is the safer shape for API-backed runs: each mode writes an independent
# detail log and can finish within wallclock instead of serializing all modes
# inside one long dataset job.

set -euo pipefail

SCRIPT=${SCRIPT:-scripts/hpc/slurm_adaptive_hyre_legal.sh}
REPO=${REPO:-$(pwd)}
N_QUESTIONS=${N_QUESTIONS:-50}
RETRIEVAL_K=${RETRIEVAL_K:-5}
SEED=${SEED:-42}
USE_VLLM=${USE_VLLM:-0}
PROVIDER=${PROVIDER:-or-gemma4-26b}
MODEL=${MODEL:-google/gemma-4-26B-A4B-it}
DRY_RUN=${DRY_RUN:-0}
LOG_DIR=${LOG_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/logs}
SBATCH_EXTRA_ARGS=${SBATCH_EXTRA_ARGS:-}
SUBMIT_STAMP=${SUBMIT_STAMP:-$(date +%Y%m%d_%H%M%S)}
SUBMIT_MANIFEST=${SUBMIT_MANIFEST:-$LOG_DIR/adaptive_hyre_mode_matrix_${SUBMIT_STAMP}.tsv}

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
else
  DATASETS=(barexam housing casehold legalbench_scalr)
fi

echo "Submitting adaptive HyRE mode matrix"
echo "  script=$SCRIPT"
echo "  datasets=${DATASETS[*]}"
echo "  n=$N_QUESTIONS seed=$SEED k=$RETRIEVAL_K use_vllm=$USE_VLLM model=$MODEL provider=$PROVIDER dry_run=$DRY_RUN"
echo "  sbatch_extra_args=${SBATCH_EXTRA_ARGS:-none}"
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_DIR"
  printf 'timestamp\tdataset\tmode\tjob_name\tjob_id\tn_questions\tretrieval_k\tseed\tprovider\tuse_vllm\tmodel\n' > "$SUBMIT_MANIFEST"
  echo "  submit_manifest=$SUBMIT_MANIFEST"
fi

for dataset in "${DATASETS[@]}"; do
  if [[ "${RUN_SPECS:-}" != "" ]]; then
    # shellcheck disable=SC2206
    MODES=(${RUN_SPECS})
  else
    case "$dataset" in
      housing)
        MODES=(rag_state_filter snap_hyre_state adaptive_snap_hyre adaptive_snap_hyre_anchor adaptive_snap_hyre_diverse)
        ;;
      barexam|casehold|legalbench_scalr)
        MODES=(rag_simple rag_snap_hyde_2call snap_hyre_option adaptive_snap_hyre adaptive_snap_hyre_anchor adaptive_snap_hyre_diverse)
        ;;
      *)
        echo "ERROR: unsupported dataset '$dataset'" >&2
        exit 2
        ;;
    esac
  fi

  for mode in "${MODES[@]}"; do
    job_name="hyre-${dataset}-${mode}"
    exports="ALL,DATASET=${dataset},RUN_SPECS=${mode},N_QUESTIONS=${N_QUESTIONS},RETRIEVAL_K=${RETRIEVAL_K},SEED=${SEED},USE_VLLM=${USE_VLLM},MODEL=${MODEL},PROVIDER=${PROVIDER}"
    for name in REPO DATA_REPO EVAL_VENV CHROMA_DB_DIR LOG_DIR HF_CACHE XDG_CACHE_HOME TORCH_HOME HYRE_CACHE_PATH EMBEDDING_DEVICE CROSS_ENCODER_DEVICE; do
      if [[ "${!name:-}" != "" ]]; then
        exports="${exports},${name}=${!name}"
      fi
    done
    extra_args=()
    if [[ "$SBATCH_EXTRA_ARGS" != "" ]]; then
      # shellcheck disable=SC2206
      extra_args=($SBATCH_EXTRA_ARGS)
    fi
    cmd=(
      sbatch
      "${extra_args[@]}"
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
      "$mode" \
      "$job_name" \
      "$job_id" \
      "$N_QUESTIONS" \
      "$RETRIEVAL_K" \
      "$SEED" \
      "$PROVIDER" \
      "$USE_VLLM" \
      "$MODEL" >> "$SUBMIT_MANIFEST"
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo
  echo "Wrote submit manifest: $SUBMIT_MANIFEST"
fi
