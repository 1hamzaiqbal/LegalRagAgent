#!/bin/bash
set -euo pipefail
umask 077

: "${OPD_RELEASE_PHASE:?}"
: "${OPD_RELEASE_PROGRAM:?}"
: "${OPD_RELEASE_PLAN:?}"
: "${OPD_RELEASE_REPO:?}"
: "${OPD_RELEASE_SELECTOR_KIND:?}"
: "${OPD_RELEASE_SELECTOR_VALUE:?}"
: "${OPD_MATH_TRAIN_ENV:?}"

selector=()
if [[ "$OPD_RELEASE_SELECTOR_KIND" == "arm_key" ]]; then
  selector=(--arm-key "$OPD_RELEASE_SELECTOR_VALUE")
elif [[ "$OPD_RELEASE_SELECTOR_KIND" == "raw_source" ]]; then
  selector=(--raw-source "$OPD_RELEASE_SELECTOR_VALUE")
else
  echo "invalid OPD release selector" >&2
  exit 2
fi

python_bin="$OPD_MATH_TRAIN_ENV/bin/python"
public_sentinel='evaluation phase terminal'

case "$OPD_RELEASE_PHASE" in
  shard)
    : "${SLURM_ARRAY_TASK_ID:?}"
    : "${OPD_RELEASE_SHARD_CONSUMPTION_ROOT:?}"
    : "${OPD_RELEASE_PRIVATE_LOG_ROOT:?}"
    receipt=$(printf '%s/shard_%05d.json' \
      "$OPD_RELEASE_SHARD_CONSUMPTION_ROOT" "$SLURM_ARRAY_TASK_ID")
    mkdir -p "$OPD_RELEASE_PRIVATE_LOG_ROOT"
    authorization_partial=$(printf '%s/shard_%05d.authorization.private.log.partial.%s' \
      "$OPD_RELEASE_PRIVATE_LOG_ROOT" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID")
    authorization_log=$(printf '%s/shard_%05d.authorization.private.log' \
      "$OPD_RELEASE_PRIVATE_LOG_ROOT" "$SLURM_ARRAY_TASK_ID")
    [[ ! -e "$authorization_partial" && ! -L "$authorization_partial" && ! -e "$authorization_log" && ! -L "$authorization_log" ]]
    set +e
    "$python_bin" "$OPD_RELEASE_PROGRAM" consume-evaluation-authorization \
      --repo "$OPD_RELEASE_REPO" \
      --release-plan "$OPD_RELEASE_PLAN" \
      "${selector[@]}" \
      --phase shard \
      --shard-index "$SLURM_ARRAY_TASK_ID" \
      --output "$receipt" >"$authorization_partial" 2>&1
    authorization_rc=$?
    set -e
    mv "$authorization_partial" "$authorization_log"
    chmod 0400 "$authorization_log"
    if (( authorization_rc != 0 )); then
      printf '%s\n' "$public_sentinel"
      exit "$authorization_rc"
    fi
    private_partial=$(printf '%s/shard_%05d.private.log.partial.%s' \
      "$OPD_RELEASE_PRIVATE_LOG_ROOT" "$SLURM_ARRAY_TASK_ID" "$SLURM_JOB_ID")
    private_log=$(printf '%s/shard_%05d.private.log' \
      "$OPD_RELEASE_PRIVATE_LOG_ROOT" "$SLURM_ARRAY_TASK_ID")
    [[ ! -e "$private_partial" && ! -L "$private_partial" && ! -e "$private_log" && ! -L "$private_log" ]]
    set +e
    /bin/bash "$OPD_RELEASE_REPO/scripts/hpc/slurm_opd_math_evaluate.sh" >"$private_partial" 2>&1
    child_rc=$?
    set -e
    mv "$private_partial" "$private_log"
    chmod 0400 "$private_log"
    printf '%s\n' "$public_sentinel"
    exit "$child_rc"
    ;;
  merge_supervisor)
    : "${OPD_RELEASE_PRIVATE_LOG_ROOT:?}"
    mkdir -p "$OPD_RELEASE_PRIVATE_LOG_ROOT"
    private_partial="$OPD_RELEASE_PRIVATE_LOG_ROOT/merge_supervisor.controller.private.log.partial.$SLURM_JOB_ID"
    private_log="$OPD_RELEASE_PRIVATE_LOG_ROOT/merge_supervisor.controller.private.log"
    [[ ! -e "$private_partial" && ! -L "$private_partial" && ! -e "$private_log" && ! -L "$private_log" ]]
    set +e
    "$python_bin" "$OPD_RELEASE_PROGRAM" supervise-evaluation-merge \
      --repo "$OPD_RELEASE_REPO" \
      --release-plan "$OPD_RELEASE_PLAN" \
      "${selector[@]}" \
      --output "$OPD_RELEASE_MERGE_SUPERVISOR" >"$private_partial" 2>&1
    controller_rc=$?
    set -e
    mv "$private_partial" "$private_log"
    chmod 0400 "$private_log"
    printf '%s\n' "$public_sentinel"
    exit "$controller_rc"
    ;;
  seal_supervisor)
    : "${OPD_RELEASE_PRIVATE_LOG_ROOT:?}"
    mkdir -p "$OPD_RELEASE_PRIVATE_LOG_ROOT"
    private_partial="$OPD_RELEASE_PRIVATE_LOG_ROOT/seal_supervisor.controller.private.log.partial.$SLURM_JOB_ID"
    private_log="$OPD_RELEASE_PRIVATE_LOG_ROOT/seal_supervisor.controller.private.log"
    [[ ! -e "$private_partial" && ! -L "$private_partial" && ! -e "$private_log" && ! -L "$private_log" ]]
    set +e
    "$python_bin" "$OPD_RELEASE_PROGRAM" supervise-evaluation-seal \
      --repo "$OPD_RELEASE_REPO" \
      --release-plan "$OPD_RELEASE_PLAN" \
      "${selector[@]}" \
      --output "$OPD_RELEASE_SEAL_SUPERVISOR" >"$private_partial" 2>&1
    controller_rc=$?
    set -e
    mv "$private_partial" "$private_log"
    chmod 0400 "$private_log"
    printf '%s\n' "$public_sentinel"
    exit "$controller_rc"
    ;;
  wave_finalizer)
    : "${OPD_RELEASE_EVALUATION_WAVE_SEAL:?}"
    : "${OPD_RELEASE_FINALIZER_RECEIPT:?}"
    finalizer_private_root=$(dirname "$OPD_RELEASE_EVALUATION_WAVE_SEAL")
    mkdir -p "$finalizer_private_root"
    private_partial="$finalizer_private_root/evaluation_wave_finalizer.controller.private.log.partial.$SLURM_JOB_ID"
    private_log="$finalizer_private_root/evaluation_wave_finalizer.controller.private.log"
    [[ ! -e "$private_partial" && ! -L "$private_partial" && ! -e "$private_log" && ! -L "$private_log" ]]
    set +e
    "$python_bin" "$OPD_RELEASE_PROGRAM" seal-evaluation-wave \
      --repo "$OPD_RELEASE_REPO" \
      --release-plan "$OPD_RELEASE_PLAN" \
      --output "$OPD_RELEASE_EVALUATION_WAVE_SEAL" >"$private_partial" 2>&1
    controller_rc=$?
    set -e
    mv "$private_partial" "$private_log"
    chmod 0400 "$private_log"
    set +e
    "$python_bin" "$OPD_RELEASE_PROGRAM" record-wave-finalizer \
      --repo "$OPD_RELEASE_REPO" \
      --release-plan "$OPD_RELEASE_PLAN" \
      --controller-rc "$controller_rc" \
      --private-log "$private_log" \
      --output "$OPD_RELEASE_FINALIZER_RECEIPT" >/dev/null 2>&1
    receipt_rc=$?
    set -e
    printf '%s\n' "$public_sentinel"
    if (( receipt_rc != 0 )); then
      exit "$receipt_rc"
    fi
    exit "$controller_rc"
    ;;
  *)
    echo "invalid OPD release phase" >&2
    exit 2
    ;;
esac
