#!/bin/bash
# Submit only trainer-data projection and its full pinned-runtime CPU audit.
set -euo pipefail

REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
BRANCH="codex/opd_identifiability_v1"

test "$(git -C "$REPO" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$REPO" status --porcelain=v1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
PRODUCER_JOB="$(sbatch --parsable \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_trainer_data.sh")"
PRODUCER_JOB="${PRODUCER_JOB%%;*}"
AUDIT_JOB="$(sbatch --parsable \
  --dependency="afterok:${PRODUCER_JOB}" \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT,OPD_IDENT_TRAINER_DATA_JOB_ID=$PRODUCER_JOB" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_trainer_data_audit.sh")"
AUDIT_JOB="${AUDIT_JOB%%;*}"

LEDGER="$RUN_ROOT/submissions/trainer_data_${COMMIT}_${PRODUCER_JOB}.json"
test ! -e "$LEDGER"
mkdir -p "$(dirname "$LEDGER")"
python3 - "$LEDGER" "$COMMIT" "$PRODUCER_JOB" "$AUDIT_JOB" <<'PY'
import json, os, sys
from datetime import datetime, timezone

path, commit, producer_job, audit_job = sys.argv[1:]
payload = {
    "schema_version": 2,
    "artifact_type": "opd_positive_control_trainer_data_submission",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "repository_commit": commit,
    "jobs": {"producer": producer_job, "independent_audit": audit_job},
    "dependency": f"independent_audit afterok {producer_job}",
    "training_queued": False,
}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
echo "$PRODUCER_JOB $AUDIT_JOB"
