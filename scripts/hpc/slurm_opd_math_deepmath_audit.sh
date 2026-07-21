#!/bin/bash
#SBATCH --job-name=opd_C_audit
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_C_audit_%j.out

set -euo pipefail
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
INVENTORY_ROOT="${OPD_DEEPMATH_INVENTORY_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/deepmath_inventory_v1}"
OUTPUT_DIR="${OPD_DEEPMATH_AUDIT_ROOT:?Set a new immutable DeepMath audit output root}"
AUDITOR="$REPO/scripts/opd_math/audit_deepmath_inventory.py"

test -x "$ENV_DIR/bin/python"
test -f "$AUDITOR"
test -f "$INVENTORY_ROOT/inventory_manifest.json"
test ! -e "$OUTPUT_DIR"
test -z "$(git -C "$REPO" status --porcelain=v1)"
export HF_HOME="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export XDG_CACHE_HOME="${OPD_XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/xdg_cache}"
export OPD_DEEPMATH_AUDIT_LAUNCHER_PATH="$0"
mkdir -p "$XDG_CACHE_HOME"

cd "$REPO"
"$ENV_DIR/bin/python" "$AUDITOR" \
  --inventory-manifest "$INVENTORY_ROOT/inventory_manifest.json" \
  --output-dir "$OUTPUT_DIR" \
  --local-files-only

echo "PASS DeepMath global data scan completed; teacher training remains unauthorized"
