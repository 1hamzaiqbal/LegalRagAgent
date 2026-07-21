#!/bin/bash
#SBATCH --job-name=opd_C_finalize
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_C_finalize_%j.out

set -euo pipefail
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
INVENTORY_ROOT="${OPD_DEEPMATH_INVENTORY_ROOT:?Bind the immutable inventory root}"
SCAN_ROOT="${OPD_DEEPMATH_AUDIT_ROOT:?Bind the immutable completed scan root}"
DECISIONS="${OPD_DEEPMATH_REVIEW_DECISIONS:?Bind the complete semantic decision JSONL}"
OUTPUT_DIR="${OPD_DEEPMATH_FINAL_ROOT:?Set a new immutable finalization root}"
FINALIZER="$REPO/scripts/opd_math/finalize_deepmath_audit.py"

test -x "$ENV_DIR/bin/python"
test -f "$FINALIZER"
test -f "$INVENTORY_ROOT/inventory_manifest.json"
test -f "$SCAN_ROOT/audit_manifest.json"
test -f "$DECISIONS"
test ! -L "$DECISIONS"
test ! -e "$OUTPUT_DIR"
test -z "$(git -C "$REPO" status --porcelain=v1)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPD_DEEPMATH_FINALIZE_LAUNCHER_PATH="$0"

cd "$REPO"
"$ENV_DIR/bin/python" "$FINALIZER" \
  --inventory-manifest "$INVENTORY_ROOT/inventory_manifest.json" \
  --scan-manifest "$SCAN_ROOT/audit_manifest.json" \
  --review-decisions "$DECISIONS" \
  --output-dir "$OUTPUT_DIR"

echo "PASS DeepMath semantic review finalized; teacher training remains unauthorized"
