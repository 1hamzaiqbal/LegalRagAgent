#!/bin/bash
#SBATCH --job-name=opd_math_cache
#SBATCH --partition=general-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_cache_%j.out

# Populate the shared HF cache with the exact primary teacher/student revisions.
# This stage is intentionally online and idempotent; all training/evaluation jobs
# remain offline and fail if either pinned snapshot is absent.
set -euo pipefail

REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
MANIFEST="$REPO/configs/opd_math/source_manifest.json"

test -x "$ENV_DIR/bin/python"
test -f "$MANIFEST"
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

"$ENV_DIR/bin/python" - "$MANIFEST" <<'PY'
import json
import re
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

manifest_path = Path(sys.argv[1]).resolve()
manifest = json.loads(manifest_path.read_text())
resolved = {}
for key in ("teacher", "student"):
    spec = manifest["models"][key]
    repo_id = spec["id"]
    revision = spec["revision"]
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"models.{key}.revision is not an immutable commit: {revision!r}")
    resolved[key] = {
        "id": repo_id,
        "revision": revision,
        "snapshot": snapshot_download(repo_id=repo_id, revision=revision),
    }
print(json.dumps({"manifest": str(manifest_path), "cached_models": resolved}, sort_keys=True))
PY

echo "PASS exact-revision OPD-math model cache fill"
