#!/bin/bash
#SBATCH --job-name=opsd_pc_trdata_audit
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opsd_pc_trdata_audit_%j.out

set -euo pipefail
REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
UPSTREAM="${OPD_UPSTREAM_REPO:-/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/OPSD}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
TARGET_BASE="${OPD_IDENT_TRAINER_DATA_BASE:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1_trainer_data}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
HF_HOME="${OPD_IDENT_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
EXPECTED_COMMIT="${OPD_IDENT_EXPECTED_COMMIT:?set OPD_IDENT_EXPECTED_COMMIT at submission}"
PRODUCER_JOB_ID="${OPD_IDENT_TRAINER_DATA_JOB_ID:?set producer job id at submission}"

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$UPSTREAM" status --porcelain=v1)"
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = "7448751f307a9cdbcc1246dd1565a1a605b443df"
TARGET="$TARGET_BASE/$EXPECTED_COMMIT"
OUTPUT="$RUN_ROOT/trainer_data/producer_job_${PRODUCER_JOB_ID}/audit_job_${SLURM_JOB_ID}.json"
export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
MODEL="$($ENV_DIR/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    local_files_only=True,
))
PY
)"
"$ENV_DIR/bin/python" "$REPO/scripts/opd/audit_positive_control_trainer_data.py" \
  --trainer-root "$TARGET" \
  --producer-commit "$EXPECTED_COMMIT" \
  --auditor-commit "$EXPECTED_COMMIT" \
  --model-dir "$MODEL" \
  --upstream "$UPSTREAM" \
  --output "$OUTPUT"
echo "PASS OPSD trainer-data audit: $OUTPUT"
