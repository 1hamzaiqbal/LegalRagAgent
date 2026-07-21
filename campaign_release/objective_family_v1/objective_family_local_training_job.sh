#!/bin/bash
set -euo pipefail
umask 077

: "${OPD_RELEASE_PROGRAM:?Set the sealed external release controller}"
: "${OPD_RELEASE_PLAN:?Set the sealed external release plan}"
: "${OPD_RELEASE_REPO:?Set the exact clean d89 experiment checkout}"
: "${OPD_RELEASE_SELECTOR_KIND:?Set arm_key}"
: "${OPD_RELEASE_SELECTOR_VALUE:?Set the preregistered arm key}"
: "${OPD_RELEASE_TRAINING_CONSUMPTION:?Set the frozen training-consumption receipt path}"
: "${SLURM_JOB_ID:?This wrapper must run inside the registered Slurm job}"

: "${OPD_MATH_TRAIN_ENV:?Set the pinned controller/training environment}"
: "${OPD_MATH_SERVE_ENV:?Set the pinned serving environment}"
: "${OPD_MATH_RUN_ROOT:?Set the frozen OPD artifact root}"
: "${OPD_MATH_HF_HOME:?Set the frozen Hugging Face cache root}"
: "${OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root}"
: "${OPD_MATH_OBJECTIVE_ID:?Set one registered local objective ID}"
: "${OPD_MATH_STUDENT_SOURCE:?Set M or O}"
: "${OPD_MATH_SEED:?Set registered seed 0, 1, or 2}"
: "${OPD_MATH_CAMPAIGN_KIND:?Set scientific}"
: "${OPD_MATH_OBJECTIVE_OUT:?Set the fresh preregistered output directory}"
: "${OPD_MATH_STUDENT_SUPPORT_MANIFEST:?Set the same-commit support gate}"
: "${OPD_MATH_OBJECTIVE_PROMPT_PLAN:?Set the exact source/seed prompt plan}"
: "${OPD_MATH_OBJECTIVE_INITIALIZATION_MANIFEST:?Set the exact seed adapter manifest}"
: "${OPD_MATH_OBJECTIVE_PREREGISTRATION:?Set the sealed scientific preregistration}"
: "${OPD_MATH_OBJECTIVE_LAUNCH_PLAN:?Set the sealed scientific launch plan}"

[[ "$SLURM_JOB_ID" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid SLURM_JOB_ID" >&2
  exit 2
}
[[ "$OPD_RELEASE_PROGRAM" = /* && "$OPD_RELEASE_PLAN" = /* && "$OPD_RELEASE_REPO" = /* ]] || {
  echo "release controller, plan, and repo paths must be absolute" >&2
  exit 2
}
[[ "$OPD_RELEASE_TRAINING_CONSUMPTION" = /* ]] || {
  echo "training-consumption receipt path must be absolute" >&2
  exit 2
}
[[ "$OPD_RELEASE_SELECTOR_KIND" == "arm_key" ]] || {
  echo "training custody accepts only the arm_key selector" >&2
  exit 2
}
[[ "$OPD_MATH_CAMPAIGN_KIND" == "scientific" ]] || {
  echo "external training custody is restricted to the preregistered scientific campaign" >&2
  exit 2
}
case "$OPD_MATH_OBJECTIVE_ID" in
  task_rl|task_rl_k1_ungated_clip5|task_rl_k1_ungated_unclipped|task_rl_k1_gated_clip5_beta5|k1_bare_verl_compatible_clip10) ;;
  *) echo "invalid local objective-family ID" >&2; exit 2 ;;
esac
case "$OPD_MATH_STUDENT_SOURCE" in M|O) ;; *) echo "invalid student source" >&2; exit 2 ;; esac
case "$OPD_MATH_SEED" in 0|1|2) ;; *) echo "invalid objective-family seed" >&2; exit 2 ;; esac

expected_arm_key="${OPD_MATH_OBJECTIVE_ID}__${OPD_MATH_STUDENT_SOURCE}__seed${OPD_MATH_SEED}"
[[ "$OPD_RELEASE_SELECTOR_VALUE" == "$expected_arm_key" ]] || {
  echo "release selector does not match the local objective/source/seed" >&2
  exit 2
}

python_bin="$OPD_MATH_TRAIN_ENV/bin/python"
launcher="$OPD_RELEASE_REPO/scripts/hpc/slurm_opd_math_objective_family_train.sh"
test -x "$python_bin"
test -f "$OPD_RELEASE_PROGRAM" && test ! -L "$OPD_RELEASE_PROGRAM"
test -f "$OPD_RELEASE_PLAN" && test ! -L "$OPD_RELEASE_PLAN"
test -d "$OPD_RELEASE_REPO" && test ! -L "$OPD_RELEASE_REPO"
test -f "$launcher" && test ! -L "$launcher"
[[ ! -e "$OPD_RELEASE_TRAINING_CONSUMPTION" && ! -L "$OPD_RELEASE_TRAINING_CONSUMPTION" ]] || {
  echo "refusing to reuse training-consumption receipt" >&2
  exit 2
}

if [[ -n "${OPD_MATH_REPO:-}" && "$OPD_MATH_REPO" != "$OPD_RELEASE_REPO" ]]; then
  echo "OPD_MATH_REPO differs from the registered release checkout" >&2
  exit 2
fi
export OPD_MATH_REPO="$OPD_RELEASE_REPO"

"$python_bin" "$OPD_RELEASE_PROGRAM" consume-training-authorization \
  --repo "$OPD_RELEASE_REPO" \
  --release-plan "$OPD_RELEASE_PLAN" \
  --arm-key "$OPD_RELEASE_SELECTOR_VALUE" \
  --scheduler-job-id "$SLURM_JOB_ID" \
  --output "$OPD_RELEASE_TRAINING_CONSUMPTION"

test -f "$OPD_RELEASE_TRAINING_CONSUMPTION"
test ! -L "$OPD_RELEASE_TRAINING_CONSUMPTION"
exec /bin/bash "$launcher"
