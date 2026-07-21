#!/bin/bash
#SBATCH --job-name=opd_deepmath_raw
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_deepmath_raw_%j.out

set -euo pipefail
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
PLAN="$REPO/configs/opd_math/deepmath_qualification_plan.json"
QUALIFIER="$REPO/scripts/opd_math/deepmath_qualification.py"
DATA_ROOT="${OPD_DEEPMATH_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_math/deepmath_C/5cf055d1fe3d7a2eb19719ac020211469736ae44}"
RECEIPT="$DATA_ROOT/qualification/raw_identity_manifest.json"
CUSTODY="$DATA_ROOT/qualification/download_custody.json"

test -x "$ENV_DIR/bin/python"
test -f "$PLAN"
test -f "$QUALIFIER"
test -z "$(git -C "$REPO" status --porcelain=v1)"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]]
PLAN_SHA="$(sha256sum "$PLAN" | awk '{print $1}')"
SCRIPT_SHA="$(sha256sum "$0" | awk '{print $1}')"
mkdir -p "$DATA_ROOT/data" "$DATA_ROOT/qualification"

if [[ -e "$RECEIPT" || -e "$CUSTODY" ]]; then
  test -f "$RECEIPT"
  test -f "$CUSTODY"
  "$ENV_DIR/bin/python" "$QUALIFIER" --plan "$PLAN" verify-raw \
    --data-dir "$DATA_ROOT" >/dev/null
  "$ENV_DIR/bin/python" - "$CUSTODY" "$COMMIT" "$PLAN_SHA" "$SCRIPT_SHA" "$RECEIPT" <<'PY'
import hashlib
import json
import pathlib
import sys

custody_path, commit, plan_sha, script_sha, receipt_path = sys.argv[1:]
custody = json.loads(pathlib.Path(custody_path).read_text())
receipt_sha = hashlib.sha256(pathlib.Path(receipt_path).read_bytes()).hexdigest()
assert custody["git_commit"] == commit
assert custody["plan_sha256"] == plan_sha
assert custody["download_wrapper_sha256"] == script_sha
assert custody["raw_identity_manifest_sha256"] == receipt_sha
assert custody["teacher_training_authorized"] is False
print("PASS existing DeepMath raw custody revalidated")
PY
  exit 0
fi

mapfile -t SHARDS < <("$ENV_DIR/bin/python" - "$PLAN" <<'PY'
import json
import sys
from urllib.parse import quote

plan = json.load(open(sys.argv[1]))
candidate = plan["candidate"]
base = "https://huggingface.co/datasets/{}/resolve/{}/".format(
    candidate["dataset_id"], candidate["revision"]
)
for shard in candidate["raw_shards"]:
    print("\t".join((shard["path"], str(shard["bytes"]), shard["sha256"], base + quote(shard["path"]))))
PY
)
[[ "${#SHARDS[@]}" -eq 10 ]]

for row in "${SHARDS[@]}"; do
  IFS=$'\t' read -r relative expected_bytes expected_sha url <<<"$row"
  target="$DATA_ROOT/$relative"
  partial="$target.partial"
  mkdir -p "$(dirname "$target")"
  if [[ -e "$target" ]]; then
    test -f "$target"
    [[ "$(stat -c %s "$target")" == "$expected_bytes" ]]
    [[ "$(sha256sum "$target" | awk '{print $1}')" == "$expected_sha" ]]
    continue
  fi
  test ! -L "$partial"
  curl --fail --location --retry 5 --retry-all-errors --continue-at - \
    --output "$partial" "$url"
  [[ "$(stat -c %s "$partial")" == "$expected_bytes" ]]
  [[ "$(sha256sum "$partial" | awk '{print $1}')" == "$expected_sha" ]]
  mv "$partial" "$target"
done

"$ENV_DIR/bin/python" "$QUALIFIER" --plan "$PLAN" verify-raw \
  --data-dir "$DATA_ROOT" --output "$RECEIPT" >/dev/null
"$ENV_DIR/bin/python" - "$CUSTODY" "$COMMIT" "$PLAN_SHA" "$SCRIPT_SHA" "$RECEIPT" "${SLURM_JOB_ID:-none}" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

custody_path, commit, plan_sha, script_sha, receipt_path, job_id = sys.argv[1:]
receipt_sha = hashlib.sha256(pathlib.Path(receipt_path).read_bytes()).hexdigest()
payload = {
    "schema_version": 1,
    "stage": "deepmath_raw_download_and_identity",
    "status": "passed",
    "git_commit": commit,
    "git_worktree_clean": True,
    "plan_sha256": plan_sha,
    "download_wrapper_sha256": script_sha,
    "raw_identity_manifest": str(pathlib.Path(receipt_path).resolve()),
    "raw_identity_manifest_sha256": receipt_sha,
    "slurm_job_id": job_id,
    "teacher_training_authorized": False,
    "scientific_use_allowed": False,
}
path = pathlib.Path(custody_path)
if path.exists():
    raise FileExistsError(f"refusing to overwrite custody receipt: {path}")
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
os.chmod(path, 0o444)
os.chmod(receipt_path, 0o444)
print(json.dumps(payload, sort_keys=True))
PY

echo "PASS DeepMath raw bytes/schema only; teacher training remains unauthorized"
