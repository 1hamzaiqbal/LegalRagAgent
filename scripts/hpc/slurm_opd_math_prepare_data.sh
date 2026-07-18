#!/bin/bash
#SBATCH --job-name=opd_math_data
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_data_%j.out

set -euo pipefail
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
DATA_ROOT="${OPD_MATH_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/v1}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"

test -x "$ENV_DIR/bin/python"
test -f "$REPO/configs/opd_math/source_manifest.json"
if [[ -e "$DATA_ROOT/prepared_manifest.json" ]]; then
  echo "Refusing to overwrite prepared dataset: $DATA_ROOT" >&2
  exit 2
fi
mkdir -p "$DATA_ROOT" "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
source "$ENV_DIR/bin/activate"

ARGS=()
if [[ "${OPD_MATH_AUDIT_LIMIT:-0}" != "0" ]]; then
  ARGS+=(--audit-limit-per-split "$OPD_MATH_AUDIT_LIMIT")
fi
if [[ -n "${OPD_MATH_SEMANTIC_REVIEW_JSONL:-}" ]]; then
  test -f "$OPD_MATH_SEMANTIC_REVIEW_JSONL"
  ARGS+=(--semantic-review-jsonl "$OPD_MATH_SEMANTIC_REVIEW_JSONL")
fi
if [[ -n "${OPD_MATH_SEMANTIC_MAX_BUCKET_SIZE:-}" ]]; then
  [[ "$OPD_MATH_SEMANTIC_MAX_BUCKET_SIZE" =~ ^[1-9][0-9]*$ ]]
  ARGS+=(--semantic-max-bucket-size "$OPD_MATH_SEMANTIC_MAX_BUCKET_SIZE")
fi
if [[ -n "${OPD_MATH_SEMANTIC_FINGERPRINT_SIZE:-}" ]]; then
  [[ "$OPD_MATH_SEMANTIC_FINGERPRINT_SIZE" =~ ^[1-9][0-9]*$ ]]
  ARGS+=(--semantic-fingerprint-size "$OPD_MATH_SEMANTIC_FINGERPRINT_SIZE")
fi
python "$REPO/scripts/opd_math/prepare_data.py" \
  --manifest "$REPO/configs/opd_math/source_manifest.json" \
  --output-dir "$DATA_ROOT" \
  "${ARGS[@]}"
python - "$DATA_ROOT/prepared_manifest.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1]))
print({
    "prepared_manifest": sys.argv[1],
    "scientific_use_allowed": manifest.get("scientific_use_allowed"),
    "audit_limit_per_split": manifest.get("audit_limit_per_split"),
    "semantic_near_duplicate_audit": manifest.get("semantic_near_duplicate_audit"),
})
PY
echo "PASS preparation artifact; scientific use is controlled by prepared_manifest.json"
