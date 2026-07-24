#!/bin/bash
#SBATCH --job-name=opsd_pc_trdata
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opsd_pc_trdata_%j.out

set -euo pipefail
REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
SOURCE="${OPD_IDENT_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1}/opsd_train"
TARGET_BASE="${OPD_IDENT_TRAINER_DATA_BASE:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1_trainer_data}"
EXPECTED_COMMIT="${OPD_IDENT_EXPECTED_COMMIT:?set OPD_IDENT_EXPECTED_COMMIT at submission}"

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
TARGET="$TARGET_BASE/$EXPECTED_COMMIT"
"$ENV_DIR/bin/python" "$REPO/scripts/opd/normalize_positive_control_trainer_data.py" \
  --source-root "$SOURCE" \
  --output-root "$TARGET" \
  --repository-commit "$EXPECTED_COMMIT"
echo "PASS projected OPSD trainer data: $TARGET/manifest.json"
