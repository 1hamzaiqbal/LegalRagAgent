#!/bin/bash
#SBATCH --job-name=opsd_pc_normaudit
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opsd_pc_normaudit_%j.out

set -euo pipefail
REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
NORMALIZED_BASE="${OPD_IDENT_NORMALIZED_BASE:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1_normalized}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
HF_HOME="${OPD_IDENT_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
EXPECTED_COMMIT="${OPD_IDENT_EXPECTED_COMMIT:?set OPD_IDENT_EXPECTED_COMMIT at submission}"
NORMALIZE_JOB_ID="${OPD_IDENT_NORMALIZE_JOB_ID:?set OPD_IDENT_NORMALIZE_JOB_ID at submission}"

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
TARGET="$NORMALIZED_BASE/$EXPECTED_COMMIT"
OUTPUT="$RUN_ROOT/normalized_data/producer_job_${NORMALIZE_JOB_ID}/audit_job_${SLURM_JOB_ID}.json"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$ENV_DIR/bin/python" "$REPO/scripts/opd/audit_positive_control_normalized_data.py" \
  --normalized-root "$TARGET" \
  --producer-commit "$EXPECTED_COMMIT" \
  --auditor-commit "$EXPECTED_COMMIT" \
  --output "$OUTPUT"
echo "PASS normalized OPSD data audit: $OUTPUT"
