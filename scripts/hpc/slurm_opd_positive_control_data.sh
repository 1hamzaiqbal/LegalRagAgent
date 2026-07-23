#!/bin/bash
#SBATCH --job-name=opsd_pc_data
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opsd_pc_data_%j.out

set -euo pipefail
REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
PYTHON="${OPD_IDENT_DATA_PYTHON:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train/bin/python}"
OUTPUT="${OPD_IDENT_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1}"
HF_HOME="${OPD_IDENT_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"

test -z "$(git -C "$REPO" status --porcelain=v1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
test -x "$PYTHON"
export HF_HOME
"$PYTHON" "$REPO/scripts/opd/materialize_positive_control.py" \
  --output-root "$OUTPUT" \
  --repository-commit "$COMMIT"
