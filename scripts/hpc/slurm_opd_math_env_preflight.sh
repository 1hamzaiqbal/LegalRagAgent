#!/bin/bash
#SBATCH --job-name=opd_math_env
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_env_%j.out

set -euo pipefail
ENV_DIR="${OPD_MATH_TRAIN_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_train}"
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
test -x "$ENV_DIR/bin/python"
test -f "$REPO/configs/opd_math/source_manifest.json"
export HF_HOME="$HF_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
source "$ENV_DIR/bin/activate"
python - "$REPO/configs/opd_math/source_manifest.json" <<'PY'
import importlib.metadata as m
import json
import sys
import torch
from huggingface_hub import snapshot_download
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "bf16 unavailable"
expected = {
    "torch": "2.11.0",
    "transformers": "4.57.6",
    "trl": "1.8.0",
    "datasets": "4.8.5",
    "peft": "0.19.1",
    "math-verify": "0.9.0",
}
actual = {name: m.version(name) for name in expected}
assert actual == expected, (actual, expected)
manifest = json.load(open(sys.argv[1]))
cached_models = {}
for key in ("teacher", "student"):
    spec = manifest["models"][key]
    cached_models[key] = snapshot_download(
        spec["id"], revision=spec["revision"], local_files_only=True
    )
print({
    "packages": actual,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "bf16": torch.cuda.is_bf16_supported(),
    "cached_models": cached_models,
})
PY
echo "PASS opd-math environment GPU preflight"
