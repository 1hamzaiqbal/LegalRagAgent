#!/bin/bash
#SBATCH --job-name=opd_math_eval
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_eval_%j.out

set -euo pipefail
: "${OPD_MATH_EVAL_SOURCE:?Set M or O}"
: "${OPD_MATH_EVAL_ROLE:?Set teacher_skill_dev, target_gap_dev, student_support, source_holdout, or external_M_test}"
: "${OPD_MATH_EVAL_MODEL:?Set the pinned base model name}"
: "${OPD_MATH_EVAL_MODEL_REVISION:?Set the pinned base model revision}"
: "${OPD_MATH_EVAL_MAX_RECORDS:?Set an explicit record budget; use 0 only for the full role file}"
: "${OPD_MATH_EVAL_LABEL:?Set a filesystem-safe output label such as M_base or M_trained}"
: "${OPD_MATH_EVAL_RUN_ID:?Set a stable filesystem-safe evaluation run ID}"
: "${OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root}"

case "$OPD_MATH_EVAL_SOURCE" in M|O) ;; *) echo "invalid OPD_MATH_EVAL_SOURCE" >&2; exit 2 ;; esac
[[ "$OPD_MATH_EVAL_MAX_RECORDS" =~ ^(0|[1-9][0-9]*)$ ]] || {
  echo "OPD_MATH_EVAL_MAX_RECORDS must be a canonical nonnegative integer" >&2
  exit 2
}
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
if [[ "$OPD_MATH_EVAL_ROLE" == student_support || "$OPD_MATH_EVAL_ROLE" == source_holdout ]]; then
  EVAL_TEMPERATURE="${OPD_MATH_EVAL_TEMPERATURE:-1.0}"
  EVAL_TOP_P="${OPD_MATH_EVAL_TOP_P:-1.0}"
  EVAL_TOP_K="${OPD_MATH_EVAL_TOP_K:-0}"
  EVAL_MAX_NEW_TOKENS="${OPD_MATH_EVAL_MAX_NEW_TOKENS:-512}"
else
  EVAL_TEMPERATURE="${OPD_MATH_EVAL_TEMPERATURE:-0.7}"
  EVAL_TOP_P="${OPD_MATH_EVAL_TOP_P:-0.8}"
  EVAL_TOP_K="${OPD_MATH_EVAL_TOP_K:-20}"
  EVAL_MAX_NEW_TOKENS="${OPD_MATH_EVAL_MAX_NEW_TOKENS:-1024}"
fi
EVAL_SAMPLES_PER_PROBLEM="${OPD_MATH_EVAL_SAMPLES_PER_PROBLEM:-4}"
EVAL_SEED="${OPD_MATH_SEED:-0}"
[[ "$EVAL_SAMPLES_PER_PROBLEM" =~ ^[1-9][0-9]*$ ]] || { echo "evaluation samples per problem must be a canonical positive integer" >&2; exit 2; }
[[ "$EVAL_SEED" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "evaluation seed must be a canonical nonnegative integer" >&2; exit 2; }
[[ "$OPD_MATH_EVAL_LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe OPD_MATH_EVAL_LABEL" >&2; exit 2; }
SHARD_COUNT="${OPD_MATH_EVAL_SHARDS:-1}"
[[ "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "OPD_MATH_EVAL_SHARDS must be a positive integer" >&2; exit 2; }
if [[ -z "${OPD_MATH_EVAL_SHARD_INDEX:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] && (( SHARD_COUNT > 1 )); then
  echo "sharded evaluation requires SLURM_ARRAY_TASK_ID or OPD_MATH_EVAL_SHARD_INDEX" >&2
  exit 2
fi
if [[ -n "${OPD_MATH_EVAL_SHARD_INDEX:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" && "$OPD_MATH_EVAL_SHARD_INDEX" != "$SLURM_ARRAY_TASK_ID" ]]; then
  echo "explicit shard index conflicts with SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${OPD_MATH_EVAL_SHARD_INDEX:-0}}"
[[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || { echo "evaluation shard index must be nonnegative" >&2; exit 2; }
if (( SHARD_INDEX >= SHARD_COUNT )); then
  echo "evaluation shard index must be smaller than OPD_MATH_EVAL_SHARDS" >&2
  exit 2
fi

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
DATA_ROOT="$OPD_MATH_DATA_ROOT"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "repository HEAD is not immutable" >&2; exit 2; }
test -z "$(git -C "$REPO" status --porcelain=v1)" || { echo "evaluation requires a clean worktree" >&2; exit 2; }
FREEZE_ROOT="$RUN_ROOT/environment_freezes/$COMMIT"
TRAIN_FREEZE="$FREEZE_ROOT/train.freeze.txt"
VERIFY_ENVIRONMENT="$REPO/scripts/opd_math/verify_environment.py"
PLAN_VALIDATOR="$REPO/scripts/opd_math/plan_evaluation_shards.py"
TASK="$DATA_ROOT/$TASK_REL"
[[ "$OPD_MATH_EVAL_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "unsafe OPD_MATH_EVAL_RUN_ID" >&2; exit 2; }
printf -v SHARD_NAME 'shard_%05d' "$SHARD_INDEX"
OUT="$RUN_ROOT/evaluations/$OPD_MATH_EVAL_ROLE/$OPD_MATH_EVAL_LABEL/$OPD_MATH_EVAL_RUN_ID/shards/$SHARD_NAME"

test -x "$ENV_DIR/bin/python"
test -f "$VERIFY_ENVIRONMENT"
test ! -L "$VERIFY_ENVIRONMENT"
test -f "$PLAN_VALIDATOR"
test ! -L "$PLAN_VALIDATOR"
VERIFY_SHA="$(sha256sum "$VERIFY_ENVIRONMENT" | awk '{print $1}')"
[[ "$VERIFY_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo "could not hash environment verifier" >&2; exit 2; }
echo "Environment verifier SHA-256: $VERIFY_SHA"
test -f "$TRAIN_FREEZE"
test -f "$TASK"
test -f "$DATA_ROOT/prepared_manifest.json"
TASK_RECORDS="$(wc -l < "$TASK" | tr -d '[:space:]')"
[[ "$TASK_RECORDS" =~ ^[1-9][0-9]*$ ]] || { echo "evaluation task must be nonempty newline-terminated JSONL" >&2; exit 2; }
FULL_O_GAP=0
if [[ "$OPD_MATH_EVAL_SOURCE" == O && "$TASK_REL" == roles/O/teacher_gap_dev.jsonl ]] && (( OPD_MATH_EVAL_MAX_RECORDS == 0 || OPD_MATH_EVAL_MAX_RECORDS >= TASK_RECORDS )); then
  FULL_O_GAP=1
fi
if (( FULL_O_GAP == 1 )); then
  : "${OPD_MATH_EVAL_SHARD_PLAN:?Full O teacher evaluation requires the immutable shard plan}"
  : "${OPD_MATH_EVAL_PLAN_ARM:?Set base or trained for the full O plan}"
  : "${OPD_MATH_EVAL_ARRAY_SPEC:?Set the literal planned Slurm array specification}"
  : "${SLURM_ARRAY_TASK_COUNT:?Full O teacher evaluation must launch as the planned array}"
  : "${SLURM_ARRAY_TASK_MIN:?Full O teacher evaluation lacks array minimum custody}"
  : "${SLURM_ARRAY_TASK_MAX:?Full O teacher evaluation lacks array maximum custody}"
  PLAN_ADAPTER_ARGS=()
  if [[ -n "${OPD_MATH_EVAL_ADAPTER:-}" ]]; then
    PLAN_ADAPTER_ARGS=(--adapter "$OPD_MATH_EVAL_ADAPTER")
  fi
  "$ENV_DIR/bin/python" "$PLAN_VALIDATOR" validate-launch \
    --plan "$OPD_MATH_EVAL_SHARD_PLAN" \
    --arm "$OPD_MATH_EVAL_PLAN_ARM" \
    --phase shard \
    --source O \
    --role teacher_gap_dev \
    --model "$OPD_MATH_EVAL_MODEL" \
    --model-revision "$OPD_MATH_EVAL_MODEL_REVISION" \
    --task-file "$TASK" \
    --max-records "$OPD_MATH_EVAL_MAX_RECORDS" \
    --shard-count "$SHARD_COUNT" \
    --git-commit "$COMMIT" \
    --train-freeze "$TRAIN_FREEZE" \
    --array-spec "$OPD_MATH_EVAL_ARRAY_SPEC" \
    --samples-per-problem "$EVAL_SAMPLES_PER_PROBLEM" \
    --temperature "$EVAL_TEMPERATURE" \
    --top-p "$EVAL_TOP_P" \
    --top-k "$EVAL_TOP_K" \
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    --seed "$EVAL_SEED" \
    --array-task-count "$SLURM_ARRAY_TASK_COUNT" \
    --array-task-min "$SLURM_ARRAY_TASK_MIN" \
    --array-task-max "$SLURM_ARRAY_TASK_MAX" \
    "${PLAN_ADAPTER_ARGS[@]}"
elif [[ -n "${OPD_MATH_EVAL_SHARD_PLAN:-}" || -n "${OPD_MATH_EVAL_PLAN_ARM:-}" || -n "${OPD_MATH_EVAL_ARRAY_SPEC:-}" ]]; then
  echo "O shard plans may only be supplied to the full O teacher-gap evaluation" >&2
  exit 2
fi
mkdir -p "$(dirname "$OUT")"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source "$ENV_DIR/bin/activate"
echo "Verifying live train environment before evaluation"
"$ENV_DIR/bin/python" "$VERIFY_ENVIRONMENT" \
  --environment-root "$ENV_DIR" \
  --commit-freeze "$TRAIN_FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind train

ARGS=(
  --model "$OPD_MATH_EVAL_MODEL"
  --model-revision "$OPD_MATH_EVAL_MODEL_REVISION"
  --task-file "$TASK"
  --output-dir "$OUT"
  --train-environment-root "$ENV_DIR"
  --train-environment-freeze "$TRAIN_FREEZE"
  --max-records "$OPD_MATH_EVAL_MAX_RECORDS"
  --samples-per-problem "$EVAL_SAMPLES_PER_PROBLEM"
  --max-new-tokens "$EVAL_MAX_NEW_TOKENS"
  --temperature "$EVAL_TEMPERATURE"
  --top-p "$EVAL_TOP_P"
  --top-k "$EVAL_TOP_K"
  --seed "$EVAL_SEED"
  --shard-count "$SHARD_COUNT"
  --shard-index "$SHARD_INDEX"
  --write-completions
  --local-files-only
)
if (( FULL_O_GAP == 1 )); then
  ARGS+=(
    --shard-plan "$OPD_MATH_EVAL_SHARD_PLAN"
    --plan-arm "$OPD_MATH_EVAL_PLAN_ARM"
    --array-spec "$OPD_MATH_EVAL_ARRAY_SPEC"
    --array-task-count "$SLURM_ARRAY_TASK_COUNT"
    --array-task-min "$SLURM_ARRAY_TASK_MIN"
    --array-task-max "$SLURM_ARRAY_TASK_MAX"
  )
fi
if [[ -n "${OPD_MATH_EVAL_ADAPTER:-}" ]]; then
  test -f "$OPD_MATH_EVAL_ADAPTER/adapter_config.json"
  ARGS+=(--adapter "$OPD_MATH_EVAL_ADAPTER")
fi
"$ENV_DIR/bin/python" "$REPO/scripts/opd_math/evaluate_math.py" "${ARGS[@]}"
test -f "$OUT/summary.json"
test -f "$OUT/samples.jsonl"
test -f "$OUT.custody.json"
echo "PASS evaluation artifact only; no gate inferred: $OUT"
