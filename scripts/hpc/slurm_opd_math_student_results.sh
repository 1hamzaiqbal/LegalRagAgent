#!/bin/bash
#SBATCH --job-name=opd_math_result
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_result_%j.out

set -euo pipefail
: "${OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root}"
: "${OPD_MATH_RESULT_KIND:?Set heldout, o_teacher, or matrix}"

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
DATA_ROOT="$OPD_MATH_DATA_ROOT"
STUDENT="${OPD_MATH_STUDENT_MODEL:-Qwen/Qwen3-1.7B}"
STUDENT_REVISION="${OPD_MATH_STUDENT_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"

test -x "$ENV_DIR/bin/python"
test -f "$DATA_ROOT/prepared_manifest.json"
source "$ENV_DIR/bin/activate"

if [[ "$OPD_MATH_RESULT_KIND" == heldout ]]; then
  : "${OPD_MATH_MATRIX_KEY:?Set baseline_M, baseline_O, M_M, M_O, O_M, or O_O}"
  : "${OPD_MATH_RESULT_SOURCE:?Set the student/holdout source M or O}"
  : "${OPD_MATH_STUDENT_RUN_MANIFEST:?Set the eligible student's run_manifest.json}"
  : "${OPD_MATH_STUDENT_COMPLETION_MANIFEST:?Set the eligible student's completion_manifest.json}"
  : "${OPD_MATH_STUDENT_EVAL_SUMMARY:?Set the source_holdout summary.json}"
  : "${OPD_MATH_STUDENT_EVAL_SAMPLES:?Set the source_holdout samples.jsonl}"
  : "${OPD_MATH_STUDENT_ADAPTER:?Set the exact evaluated final adapter}"
  : "${OPD_MATH_RESULT_OUTPUT:?Set a new held-out result JSON path}"
  case "$OPD_MATH_MATRIX_KEY" in
    baseline_M|baseline_O|M_M|M_O|O_M|O_O) ;;
    *) echo "invalid OPD_MATH_MATRIX_KEY" >&2; exit 2 ;;
  esac
  case "$OPD_MATH_RESULT_SOURCE" in M|O) ;; *) echo "invalid OPD_MATH_RESULT_SOURCE" >&2; exit 2 ;; esac
  "$ENV_DIR/bin/python" "$REPO/scripts/opd_math/student_results.py" heldout \
    --matrix-key "$OPD_MATH_MATRIX_KEY" \
    --student-run-manifest "$OPD_MATH_STUDENT_RUN_MANIFEST" \
    --student-completion-manifest "$OPD_MATH_STUDENT_COMPLETION_MANIFEST" \
    --student-summary "$OPD_MATH_STUDENT_EVAL_SUMMARY" \
    --student-samples "$OPD_MATH_STUDENT_EVAL_SAMPLES" \
    --trained-adapter "$OPD_MATH_STUDENT_ADAPTER" \
    --prepared-manifest "$DATA_ROOT/prepared_manifest.json" \
    --student-model "$STUDENT" \
    --student-revision "$STUDENT_REVISION" \
    --task-source "$OPD_MATH_RESULT_SOURCE" \
    --output "$OPD_MATH_RESULT_OUTPUT"
elif [[ "$OPD_MATH_RESULT_KIND" == o_teacher ]]; then
  : "${OPD_MATH_RESULT_BASELINE_M:?Set baseline_M held-out gate}"
  : "${OPD_MATH_RESULT_O_M:?Set O_M held-out gate}"
  : "${OPD_MATH_RESULT_BASELINE_O:?Set baseline_O held-out gate}"
  : "${OPD_MATH_RESULT_O_O:?Set O_O held-out gate}"
  : "${OPD_MATH_RESULT_PREREGISTRATION:?Set the sealed prelaunch O-teacher preregistration}"
  : "${OPD_MATH_RESULT_LAUNCH_LEDGER:?Set the immutable prelaunch O-teacher ledger}"
  : "${OPD_MATH_RESULT_OUTPUT_JSON:?Set a new O-teacher readout JSON path}"
  : "${OPD_MATH_RESULT_OUTPUT_MARKDOWN:?Set a new O-teacher readout Markdown path}"
  : "${OPD_MATH_RESULT_OUTPUT_MANIFEST:?Set a new O-teacher readout bundle manifest path}"
  "$ENV_DIR/bin/python" "$REPO/scripts/opd_math/student_results.py" o-teacher-readout \
    --baseline-m "$OPD_MATH_RESULT_BASELINE_M" \
    --o-m "$OPD_MATH_RESULT_O_M" \
    --baseline-o "$OPD_MATH_RESULT_BASELINE_O" \
    --o-o "$OPD_MATH_RESULT_O_O" \
    --preregistration "$OPD_MATH_RESULT_PREREGISTRATION" \
    --launch-ledger "$OPD_MATH_RESULT_LAUNCH_LEDGER" \
    --output-json "$OPD_MATH_RESULT_OUTPUT_JSON" \
    --output-markdown "$OPD_MATH_RESULT_OUTPUT_MARKDOWN" \
    --output-manifest "$OPD_MATH_RESULT_OUTPUT_MANIFEST"
elif [[ "$OPD_MATH_RESULT_KIND" == matrix ]]; then
  : "${OPD_MATH_RESULT_BASELINE_M:?Set baseline_M held-out gate}"
  : "${OPD_MATH_RESULT_BASELINE_O:?Set baseline_O held-out gate}"
  : "${OPD_MATH_RESULT_M_M:?Set M_M held-out gate}"
  : "${OPD_MATH_RESULT_M_O:?Set M_O held-out gate}"
  : "${OPD_MATH_RESULT_O_M:?Set O_M held-out gate}"
  : "${OPD_MATH_RESULT_O_O:?Set O_O held-out gate}"
  : "${OPD_MATH_RESULT_OUTPUT_JSON:?Set a new matrix JSON path}"
  : "${OPD_MATH_RESULT_OUTPUT_MARKDOWN:?Set a new matrix Markdown path}"
  "$ENV_DIR/bin/python" "$REPO/scripts/opd_math/student_results.py" matrix \
    --baseline-m "$OPD_MATH_RESULT_BASELINE_M" \
    --baseline-o "$OPD_MATH_RESULT_BASELINE_O" \
    --m-m "$OPD_MATH_RESULT_M_M" \
    --m-o "$OPD_MATH_RESULT_M_O" \
    --o-m "$OPD_MATH_RESULT_O_M" \
    --o-o "$OPD_MATH_RESULT_O_O" \
    --output-json "$OPD_MATH_RESULT_OUTPUT_JSON" \
    --output-markdown "$OPD_MATH_RESULT_OUTPUT_MARKDOWN"
else
  echo "invalid OPD_MATH_RESULT_KIND" >&2
  exit 2
fi

echo "PASS student result custody and deterministic recomputation"
