#!/usr/bin/env bash
# Queue only the setup-to-base-evaluation ladder for the OPSD positive control.
set -euo pipefail

REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
DATA_ROOT="${OPD_IDENT_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1}"

test "$(git -C "$REPO" branch --show-current)" = "codex/opd_identifiability_v1"
test -z "$(git -C "$REPO" status --porcelain=v1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
test ! -e "$ENV_DIR"
test ! -e "$DATA_ROOT"

ENV_JOB="$(sbatch --parsable "$REPO/scripts/hpc/slurm_opd_positive_control_setup.sh")"
DATA_JOB="$(sbatch --parsable "$REPO/scripts/hpc/slurm_opd_positive_control_data.sh")"
PREFLIGHT_JOB="$(sbatch --parsable \
  --dependency="afterok:${ENV_JOB}:${DATA_JOB}" \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_preflight.sh")"
BASE_JOB="$(sbatch --parsable \
  --dependency="afterok:${PREFLIGHT_JOB}" \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT,OPD_IDENT_PREFLIGHT_JOB_ID=$PREFLIGHT_JOB" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_base_eval.sh")"

mkdir -p "$RUN_ROOT/submissions"
LEDGER="$RUN_ROOT/submissions/setup_to_base_${COMMIT}_${ENV_JOB}.json"
python3 - "$LEDGER" "$COMMIT" "$ENV_JOB" "$DATA_JOB" "$PREFLIGHT_JOB" "$BASE_JOB" <<'PY'
import datetime, json, os, sys
path, commit, env_job, data_job, preflight_job, base_job = sys.argv[1:]
payload = {
    "schema_version": 1,
    "artifact_type": "opd_positive_control_submission",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repository_commit": commit,
    "jobs": {
        "environment": env_job,
        "data": data_job,
        "four_gpu_preflight": preflight_job,
        "base_evaluation": base_job,
    },
    "dependency_chain": [
        "environment || data",
        "four_gpu_preflight afterok environment and data",
        "base_evaluation afterok semantic preflight receipt",
    ],
    "later_training_prequeued": False,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "Queued environment=$ENV_JOB data=$DATA_JOB preflight=$PREFLIGHT_JOB base_eval=$BASE_JOB"
echo "Submission ledger: $LEDGER"
