#!/bin/bash
#SBATCH --job-name=opd_math_eval_merge
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_eval_merge_%j.out

set -euo pipefail
: "${OPD_MATH_EVAL_SOURCE:?Set M or O exactly as in the shard jobs}"
: "${OPD_MATH_EVAL_ROLE:?Set the evaluation role used by the shard jobs}"
: "${OPD_MATH_EVAL_LABEL:?Set the evaluation label used by the shard jobs}"
: "${OPD_MATH_EVAL_RUN_ID:?Set the stable sharded-evaluation run ID}"
: "${OPD_MATH_EVAL_SHARDS:?Set the exact positive shard count}"
: "${OPD_MATH_EVAL_MAX_RECORDS:?Set the exact shard-job record budget; use 0 for full role}"
: "${OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root}"

case "$OPD_MATH_EVAL_SOURCE" in M|O) ;; *) echo "invalid OPD_MATH_EVAL_SOURCE" >&2; exit 2 ;; esac
case "$OPD_MATH_EVAL_ROLE" in
  teacher_skill_dev|target_gap_dev) TASK_REL="roles/$OPD_MATH_EVAL_SOURCE/teacher_gap_dev.jsonl" ;;
  student_support) TASK_REL="roles/$OPD_MATH_EVAL_SOURCE/student_opd.jsonl" ;;
  source_holdout) TASK_REL="roles/$OPD_MATH_EVAL_SOURCE/source_holdout.jsonl" ;;
  external_M_test)
    test "$OPD_MATH_EVAL_SOURCE" = M
    TASK_REL="eval/M_test.jsonl"
    ;;
  *) echo "invalid OPD_MATH_EVAL_ROLE" >&2; exit 2 ;;
esac
[[ "$OPD_MATH_EVAL_LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe OPD_MATH_EVAL_LABEL" >&2; exit 2; }
[[ "$OPD_MATH_EVAL_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe OPD_MATH_EVAL_RUN_ID" >&2; exit 2; }
[[ "$OPD_MATH_EVAL_SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "OPD_MATH_EVAL_SHARDS must be positive" >&2; exit 2; }

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
DATA_ROOT="$OPD_MATH_DATA_ROOT"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "repository HEAD is not immutable" >&2; exit 2; }
test -z "$(git -C "$REPO" status --porcelain=v1)" || { echo "evaluation merge requires a clean worktree" >&2; exit 2; }
FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"
TRAIN_FREEZE="$FREEZE_ROOT/train.freeze.txt"
VERIFY_ENVIRONMENT="$REPO/scripts/opd_math/verify_environment.py"
PLAN_VALIDATOR="$REPO/scripts/opd_math/plan_evaluation_shards.py"
RUN_DIR="$RUN_ROOT/evaluations/$OPD_MATH_EVAL_ROLE/$OPD_MATH_EVAL_LABEL/$OPD_MATH_EVAL_RUN_ID"
SHARD_ROOT="$RUN_DIR/shards"
OUT="$RUN_DIR/merged"
TASK="$DATA_ROOT/$TASK_REL"

test -x "$ENV_DIR/bin/python"
test -f "$VERIFY_ENVIRONMENT"
test ! -L "$VERIFY_ENVIRONMENT"
test -f "$PLAN_VALIDATOR"
test ! -L "$PLAN_VALIDATOR"
VERIFY_SHA="$(sha256sum "$VERIFY_ENVIRONMENT" | awk '{print $1}')"
[[ "$VERIFY_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "could not hash environment verifier" >&2; exit 2; }
echo "Environment verifier SHA-256: $VERIFY_SHA"
test -f "$TRAIN_FREEZE"
test -d "$SHARD_ROOT"
test -f "$TASK"
if [[ "$OPD_MATH_EVAL_SOURCE" == O && "$OPD_MATH_EVAL_ROLE" == teacher_skill_dev && "$OPD_MATH_EVAL_MAX_RECORDS" == 0 ]]; then
  : "${OPD_MATH_EVAL_SHARD_PLAN:?Full O teacher merge requires the immutable shard plan}"
  : "${OPD_MATH_EVAL_PLAN_ARM:?Set base or trained for the full O plan}"
  : "${OPD_MATH_EVAL_MODEL:?Full O teacher merge requires the planned model}"
  : "${OPD_MATH_EVAL_MODEL_REVISION:?Full O teacher merge requires the planned revision}"
  PLAN_ADAPTER_ARGS=()
  if [[ -n "${OPD_MATH_EVAL_ADAPTER:-}" ]]; then
    PLAN_ADAPTER_ARGS=(--adapter "$OPD_MATH_EVAL_ADAPTER")
  fi
  "$ENV_DIR/bin/python" "$PLAN_VALIDATOR" validate-launch \
    --plan "$OPD_MATH_EVAL_SHARD_PLAN" \
    --arm "$OPD_MATH_EVAL_PLAN_ARM" \
    --phase merge \
    --source O \
    --role teacher_gap_dev \
    --model "$OPD_MATH_EVAL_MODEL" \
    --model-revision "$OPD_MATH_EVAL_MODEL_REVISION" \
    --task-file "$TASK" \
    --max-records "$OPD_MATH_EVAL_MAX_RECORDS" \
    --shard-count "$OPD_MATH_EVAL_SHARDS" \
    --git-commit "$COMMIT" \
    --train-freeze "$TRAIN_FREEZE" \
    "${PLAN_ADAPTER_ARGS[@]}"
elif [[ -n "${OPD_MATH_EVAL_SHARD_PLAN:-}" || -n "${OPD_MATH_EVAL_PLAN_ARM:-}" ]]; then
  echo "O shard plans may only be supplied to the full O teacher_skill_dev merge" >&2
  exit 2
fi
source "$ENV_DIR/bin/activate"
echo "Verifying live train environment before evaluation merge"
"$ENV_DIR/bin/python" "$VERIFY_ENVIRONMENT" \
  --environment-root "$ENV_DIR" \
  --commit-freeze "$TRAIN_FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind train
"$ENV_DIR/bin/python" "$REPO/scripts/opd_math/merge_evaluations.py" \
  --shard-root "$SHARD_ROOT" \
  --shard-count "$OPD_MATH_EVAL_SHARDS" \
  --task-file "$TASK" \
  --output-dir "$OUT" \
  --train-environment-root "$ENV_DIR" \
  --train-environment-freeze "$TRAIN_FREEZE"
test -f "$OUT/summary.json"
test -f "$OUT/samples.jsonl"
test -f "$OUT.custody.json"
echo "PASS merged evaluation artifact only; no scientific gate inferred: $OUT"
