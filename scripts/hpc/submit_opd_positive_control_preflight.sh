#!/usr/bin/env bash
# Resume at the four-GPU preflight using immutable setup artifacts from a named commit.
set -euo pipefail

REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
DATA_ROOT="${OPD_IDENT_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1}"
PRODUCER_COMMIT="${OPD_IDENT_PRODUCER_COMMIT:?set the commit that produced the immutable environment and data}"
GPU_TYPE="${OPD_IDENT_EVAL_GPU_TYPE:-a100-sxm4}"

case "$GPU_TYPE" in
  a100-sxm4|a6000) ;;
  *) echo "Unsupported positive-control evaluation GPU type: $GPU_TYPE" >&2; exit 2 ;;
esac

test "$(git -C "$REPO" branch --show-current)" = "codex/opd_identifiability_v1"
test -z "$(git -C "$REPO" status --porcelain=v1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
test -x "$ENV_DIR/bin/python"
test -f "$DATA_ROOT/manifest.json"
test -f "$RUN_ROOT/environment_freezes/$PRODUCER_COMMIT/upstream_opsd.freeze.txt"
python3 - "$DATA_ROOT/manifest.json" "$PRODUCER_COMMIT" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
assert payload["repository_commit"] == sys.argv[2]
assert payload["sources"]["opsd_train"]["rows"] == 29434
assert payload["sources"]["aime24"]["rows"] == 30
PY

PREFLIGHT_JOB="$(sbatch --parsable \
  --gpus="${GPU_TYPE}:4" \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT,OPD_IDENT_ENVIRONMENT_COMMIT=$PRODUCER_COMMIT,OPD_IDENT_DATA_COMMIT=$PRODUCER_COMMIT" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_preflight.sh")"
BASE_JOB="$(sbatch --parsable \
  --gpus="${GPU_TYPE}:4" \
  --dependency="afterok:${PREFLIGHT_JOB}" \
  --export="ALL,OPD_IDENT_EXPECTED_COMMIT=$COMMIT,OPD_IDENT_PREFLIGHT_JOB_ID=$PREFLIGHT_JOB" \
  "$REPO/scripts/hpc/slurm_opd_positive_control_base_eval.sh")"

mkdir -p "$RUN_ROOT/submissions"
LEDGER="$RUN_ROOT/submissions/preflight_to_base_${COMMIT}_${PREFLIGHT_JOB}.json"
python3 - "$LEDGER" "$COMMIT" "$PRODUCER_COMMIT" "$GPU_TYPE" "$PREFLIGHT_JOB" "$BASE_JOB" <<'PY'
import datetime, json, os, sys
path, commit, producer_commit, gpu_type, preflight_job, base_job = sys.argv[1:]
payload = {
    "schema_version": 1,
    "artifact_type": "opd_positive_control_resumed_submission",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "repository_commit": commit,
    "immutable_setup_producer_commit": producer_commit,
    "evaluation_gpu_type": gpu_type,
    "evaluation_gpu_count": 4,
    "jobs": {"four_gpu_preflight": preflight_job, "base_evaluation": base_job},
    "single_node_required": True,
    "later_training_prequeued": False,
}
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "Queued single-node $GPU_TYPE preflight=$PREFLIGHT_JOB base_eval=$BASE_JOB"
echo "Submission ledger: $LEDGER"
