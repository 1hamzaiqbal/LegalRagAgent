#!/bin/bash
#SBATCH --job-name=opd_C_inventory
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_C_inventory_%j.out

set -euo pipefail
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
RAW_ROOT="${OPD_DEEPMATH_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/deepmath_C/5cf055d1fe3d7a2eb19719ac020211469736ae44}"
OUTPUT_DIR="${OPD_DEEPMATH_INVENTORY_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/deepmath_inventory_v1}"
CACHE_DIR="${OPD_DEEPMATH_INVENTORY_CACHE:-/engrfs/project/jacobsn/hiqbal/cache/legalrag/deepmath_inventory_datasets}"
PLAN="$REPO/configs/opd_math/deepmath_inventory_plan.json"
QUALIFICATION_PLAN="$REPO/configs/opd_math/deepmath_qualification_plan.json"
MATERIALIZER="$REPO/scripts/opd_math/materialize_deepmath_inventory.py"

test -x "$ENV_DIR/bin/python"
test -f "$PLAN"
test -f "$QUALIFICATION_PLAN"
test -f "$MATERIALIZER"
test -z "$(git -C "$REPO" status --porcelain=v1)"
export HF_HOME="${OPD_HF_HOME:-/engrfs/project/jacobsn/hiqbal/cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$CACHE_DIR"
export XDG_CACHE_HOME="${OPD_XDG_CACHE_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/xdg_cache}"
export UV_CACHE_DIR="${OPD_UV_CACHE_DIR:-/engrfs/tmp/jacobsn/hiqbal_legalrag/uv_cache}"
export TOKENIZERS_PARALLELISM=false
export OPD_INVENTORY_LAUNCHER_PATH="$0"
mkdir -p "$CACHE_DIR" "$HF_HUB_CACHE" "$XDG_CACHE_HOME" "$UV_CACHE_DIR"

"$ENV_DIR/bin/python" "$MATERIALIZER" \
  --plan "$PLAN" \
  --qualification-plan "$QUALIFICATION_PLAN" \
  --deepmath-raw-root "$RAW_ROOT" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$OUTPUT_DIR"

echo "PASS DeepMath global inventory materialized; collision and training authorization remain closed"
