#!/bin/bash
# Submit only the preregistered one-step OPSD diagnostic. No dependent job.
set -euo pipefail

REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
BRANCH="codex/opd_identifiability_v1"

test "$(git -C "$REPO" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$REPO" status --porcelain=v1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
CONFIG="${OPD_IDENT_ONE_STEP_CONFIG:-$REPO/configs/opd_math/identifiability_v1_one_step_retry3.json}"
test -f "$CONFIG"

JOB_ID="$(sbatch --parsable \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT,OPD_IDENT_ONE_STEP_CONFIG=$CONFIG" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_one_step.sh")"
case "$JOB_ID" in
  *';'*) JOB_ID="${JOB_ID%%;*}" ;;
esac

LEDGER="$RUN_ROOT/submissions/one_step_${COMMIT}_${JOB_ID}.json"
test ! -e "$LEDGER"
mkdir -p "$(dirname "$LEDGER")"
python3 - "$LEDGER" "$COMMIT" "$JOB_ID" "$CONFIG" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

output, commit, job_id, config_name = sys.argv[1:]
config = Path(config_name).resolve()
payload = {
    "schema_version": 1,
    "artifact_type": "opd_positive_control_one_step_submission",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "repository_commit": commit,
    "slurm_job_id": job_id,
    "preregistration": str(config),
    "preregistration_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
    "dependent_jobs": [],
    "full_training_queued": False,
}
descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "$JOB_ID"
