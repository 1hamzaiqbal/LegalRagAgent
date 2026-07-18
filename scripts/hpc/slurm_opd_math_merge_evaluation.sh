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
RUN_DIR="$RUN_ROOT/evaluations/$OPD_MATH_EVAL_ROLE/$OPD_MATH_EVAL_LABEL/$OPD_MATH_EVAL_RUN_ID"
SHARD_ROOT="$RUN_DIR/shards"
OUT="$RUN_DIR/merged"
TASK="$DATA_ROOT/$TASK_REL"

test -x "$ENV_DIR/bin/python"
test -d "$SHARD_ROOT"
test -f "$TASK"
source "$ENV_DIR/bin/activate"
"$ENV_DIR/bin/python" "$REPO/scripts/opd_math/merge_evaluations.py" \
  --shard-root "$SHARD_ROOT" \
  --shard-count "$OPD_MATH_EVAL_SHARDS" \
  --task-file "$TASK" \
  --output-dir "$OUT"
test -f "$OUT/summary.json"
test -f "$OUT/samples.jsonl"
echo "PASS merged evaluation artifact only; no scientific gate inferred: $OUT"
