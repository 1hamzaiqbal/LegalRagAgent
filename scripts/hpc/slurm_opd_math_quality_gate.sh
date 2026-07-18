#!/bin/bash
#SBATCH --job-name=opd_math_gate
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_gate_%j.out

set -euo pipefail
: "${OPD_MATH_DATA_ROOT:?Set the exact reviewed canonical data root}"
: "${OPD_MATH_GATE_KIND:?Set teacher_gap or student_support}"
: "${OPD_MATH_GATE_SOURCE:?Set M or O}"
: "${OPD_MATH_GATE_OUTPUT:?Set a new persistent output JSON path}"
case "$OPD_MATH_GATE_SOURCE" in M|O) ;; *) echo "invalid OPD_MATH_GATE_SOURCE" >&2; exit 2 ;; esac
STRENGTH="${OPD_MATH_GATE_STRENGTH:-scientific}"
case "$STRENGTH" in scientific|smoke) ;; *) echo "invalid OPD_MATH_GATE_STRENGTH" >&2; exit 2 ;; esac

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
DATA_ROOT="$OPD_MATH_DATA_ROOT"
PREPARED_MANIFEST="${OPD_MATH_PREPARED_MANIFEST:-$DATA_ROOT/prepared_manifest.json}"
test -x "$ENV_DIR/bin/python"
test -f "$PREPARED_MANIFEST"
if [[ -e "$OPD_MATH_GATE_OUTPUT" || -L "$OPD_MATH_GATE_OUTPUT" ]]; then
  echo "Refusing to reuse gate output path: $OPD_MATH_GATE_OUTPUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$OPD_MATH_GATE_OUTPUT")"
source "$ENV_DIR/bin/activate"

SMOKE_ARGS=()
if [[ "$STRENGTH" == smoke ]]; then SMOKE_ARGS+=(--smoke-gate); fi

if [[ "$OPD_MATH_GATE_KIND" == teacher_gap ]]; then
  : "${OPD_MATH_GATE_BASE_SUMMARY:?Set base summary.json}"
  : "${OPD_MATH_GATE_BASE_SAMPLES:?Set base samples.jsonl}"
  : "${OPD_MATH_GATE_TRAINED_SUMMARY:?Set trained summary.json}"
  : "${OPD_MATH_GATE_TRAINED_SAMPLES:?Set trained samples.jsonl}"
  : "${OPD_MATH_GATE_BASE_MODEL:?Set the pinned teacher base model}"
  : "${OPD_MATH_GATE_BASE_REVISION:?Set the pinned teacher base revision}"
  : "${OPD_MATH_GATE_TRAINED_ADAPTER:?Set the evaluated teacher adapter}"
  : "${OPD_MATH_GATE_TEACHER_RUN_MANIFEST:?Set the completed teacher run_manifest.json}"
  "$ENV_DIR/bin/python" "$REPO/scripts/opd_math/quality_gates.py" teacher-gap \
    --base-summary "$OPD_MATH_GATE_BASE_SUMMARY" \
    --base-samples "$OPD_MATH_GATE_BASE_SAMPLES" \
    --trained-summary "$OPD_MATH_GATE_TRAINED_SUMMARY" \
    --trained-samples "$OPD_MATH_GATE_TRAINED_SAMPLES" \
    --base-model "$OPD_MATH_GATE_BASE_MODEL" \
    --base-revision "$OPD_MATH_GATE_BASE_REVISION" \
    --trained-adapter "$OPD_MATH_GATE_TRAINED_ADAPTER" \
    --prepared-manifest "$PREPARED_MANIFEST" \
    --teacher-run-manifest "$OPD_MATH_GATE_TEACHER_RUN_MANIFEST" \
    --task-source "$OPD_MATH_GATE_SOURCE" \
    --task-role teacher_gap_dev \
    --min-delta "${OPD_MATH_GATE_MIN_DELTA:-0.0}" \
    --bootstrap-draws "${OPD_MATH_GATE_BOOTSTRAP_DRAWS:-10000}" \
    --seed "${OPD_MATH_SEED:-0}" \
    --output "$OPD_MATH_GATE_OUTPUT" \
    "${SMOKE_ARGS[@]}"
elif [[ "$OPD_MATH_GATE_KIND" == student_support ]]; then
  : "${OPD_MATH_GATE_STUDENT_SUMMARY:?Set raw-student summary.json}"
  : "${OPD_MATH_GATE_STUDENT_SAMPLES:?Set raw-student samples.jsonl}"
  : "${OPD_MATH_GATE_STUDENT_MODEL:?Set the pinned student model}"
  : "${OPD_MATH_GATE_STUDENT_REVISION:?Set the pinned student revision}"
  "$ENV_DIR/bin/python" "$REPO/scripts/opd_math/quality_gates.py" student-support \
    --student-summary "$OPD_MATH_GATE_STUDENT_SUMMARY" \
    --student-samples "$OPD_MATH_GATE_STUDENT_SAMPLES" \
    --student-model "$OPD_MATH_GATE_STUDENT_MODEL" \
    --student-revision "$OPD_MATH_GATE_STUDENT_REVISION" \
    --prepared-manifest "$PREPARED_MANIFEST" \
    --task-source "$OPD_MATH_GATE_SOURCE" \
    --task-role student_opd \
    --min-pass-at-k "${OPD_MATH_GATE_MIN_PASS_AT_K:-0.01}" \
    --min-mixed-group-fraction "${OPD_MATH_GATE_MIN_MIXED_FRACTION:-0.01}" \
    --output "$OPD_MATH_GATE_OUTPUT" \
    "${SMOKE_ARGS[@]}"
else
  echo "invalid OPD_MATH_GATE_KIND" >&2
  exit 2
fi
echo "PASS gate computation completed; inspect passed/strength before use: $OPD_MATH_GATE_OUTPUT"
