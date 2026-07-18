#!/bin/bash
#SBATCH --job-name=opd_math_merge
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_merge_%j.out

set -euo pipefail
: "${OPD_MATH_MERGE_BASE_MODEL:?Set the teacher base model recorded by the gate}"
: "${OPD_MATH_MERGE_BASE_REVISION:?Set the teacher base revision recorded by the gate}"
: "${OPD_MATH_MERGE_ADAPTER:?Set the exact evaluated adapter directory}"
: "${OPD_MATH_MERGE_GATE:?Set the passing scientific teacher-gap manifest}"
: "${OPD_MATH_MERGE_OUTPUT:?Set a new merged-checkpoint directory}"

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
test -x "$ENV_DIR/bin/python"
test -f "$OPD_MATH_MERGE_ADAPTER/adapter_config.json"
test -f "$OPD_MATH_MERGE_GATE"
if [[ -e "$OPD_MATH_MERGE_OUTPUT" || -L "$OPD_MATH_MERGE_OUTPUT" ]]; then
  echo "Refusing to reuse merged checkpoint path: $OPD_MATH_MERGE_OUTPUT" >&2
  exit 2
fi
mkdir -p "$(dirname "$OPD_MATH_MERGE_OUTPUT")"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
source "$ENV_DIR/bin/activate"

python "$REPO/scripts/opd_math/merge_adapter.py" \
  --base-model "$OPD_MATH_MERGE_BASE_MODEL" \
  --base-revision "$OPD_MATH_MERGE_BASE_REVISION" \
  --adapter "$OPD_MATH_MERGE_ADAPTER" \
  --teacher-gap-manifest "$OPD_MATH_MERGE_GATE" \
  --output-dir "$OPD_MATH_MERGE_OUTPUT" \
  --local-files-only
test -f "$OPD_MATH_MERGE_OUTPUT/merge_provenance.json"
echo "PASS scientifically gated teacher merge: $OPD_MATH_MERGE_OUTPUT"
