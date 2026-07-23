#!/bin/bash
#SBATCH --job-name=opsd_pc_preflight
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opsd_pc_preflight_%j.out

set -euo pipefail
REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
UPSTREAM="${OPD_UPSTREAM_REPO:-/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/OPSD}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
DATA_ROOT="${OPD_IDENT_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
HF_HOME="${OPD_IDENT_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
EXPECTED_COMMIT="${OPD_IDENT_EXPECTED_COMMIT:?set OPD_IDENT_EXPECTED_COMMIT at submission}"

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git -C "$UPSTREAM" rev-parse HEAD)" = "7448751f307a9cdbcc1246dd1565a1a605b443df"
test -z "$(git -C "$UPSTREAM" status --porcelain=v1 --untracked-files=no)"
test -f "$DATA_ROOT/manifest.json"
FREEZE="$RUN_ROOT/environment_freezes/$EXPECTED_COMMIT/upstream_opsd.freeze.txt"
test -f "$FREEZE"

RECEIPT_ROOT="$RUN_ROOT/preflight/job_${SLURM_JOB_ID}"
test ! -e "$RECEIPT_ROOT"
mkdir -p "$RECEIPT_ROOT"

"$ENV_DIR/bin/python" "$REPO/scripts/opd/verify_positive_control_environment.py" \
  --environment-root "$ENV_DIR" \
  --freeze "$FREEZE" \
  --expected-cuda-devices 4 \
  >"$RECEIPT_ROOT/environment.json"

HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$ENV_DIR/bin/python" - "$DATA_ROOT" "$RECEIPT_ROOT" "$EXPECTED_COMMIT" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

data_root, receipt_root, commit = map(Path, sys.argv[1:])
manifest = json.loads((data_root / "manifest.json").read_text())
for row in manifest["files"]:
    path = data_root / row["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != row["bytes"] or digest != row["sha256"]:
        raise RuntimeError(f"data custody mismatch: {path}")
if manifest["repository_commit"] != str(commit):
    raise RuntimeError("dataset manifest belongs to another repository commit")
model = snapshot_download(
    "Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    local_files_only=True,
)
payload = {
    "schema_version": 1,
    "artifact_type": "opd_positive_control_preflight",
    "status": "passed",
    "repository_commit": str(commit),
    "data_manifest": str(data_root / "manifest.json"),
    "data_manifest_sha256": hashlib.sha256((data_root / "manifest.json").read_bytes()).hexdigest(),
    "model_snapshot": model,
    "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
out = receipt_root / "receipt.json"
fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
with os.fdopen(fd, "w") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, sort_keys=True))
PY

echo "PASS four-GPU positive-control preflight: $RECEIPT_ROOT/receipt.json"
