#!/bin/bash
# Submit only data normalization and its independent read-only audit.
set -euo pipefail

REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
BRANCH="codex/opd_identifiability_v1"

test "$(git -C "$REPO" branch --show-current)" = "$BRANCH"
test -z "$(git -C "$REPO" status --porcelain=v1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
NORMALIZE_JOB="$(sbatch --parsable \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_normalize.sh")"
NORMALIZE_JOB="${NORMALIZE_JOB%%;*}"
AUDIT_JOB="$(sbatch --parsable \
  --dependency="afterok:${NORMALIZE_JOB}" \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT,OPD_IDENT_NORMALIZE_JOB_ID=$NORMALIZE_JOB" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_normalized_audit.sh")"
AUDIT_JOB="${AUDIT_JOB%%;*}"

LEDGER="$RUN_ROOT/submissions/normalization_${COMMIT}_${NORMALIZE_JOB}.json"
test ! -e "$LEDGER"
mkdir -p "$(dirname "$LEDGER")"
python3 - "$LEDGER" "$COMMIT" "$NORMALIZE_JOB" "$AUDIT_JOB" <<'PY'
import json, os, sys
from datetime import datetime, timezone

path, commit, normalize_job, audit_job = sys.argv[1:]
payload = {
    "schema_version": 1,
    "artifact_type": "opd_positive_control_normalization_submission",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "repository_commit": commit,
    "jobs": {"normalize": normalize_job, "independent_audit": audit_job},
    "dependency": f"independent_audit afterok {normalize_job}",
    "training_queued": False,
}
descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
echo "$NORMALIZE_JOB $AUDIT_JOB"
