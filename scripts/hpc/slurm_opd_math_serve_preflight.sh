#!/bin/bash
#SBATCH --job-name=opd_math_serve_env
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_math_serve_env_%j.out

set -euo pipefail
ENV_DIR="${OPD_MATH_SERVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_serve}"
test -x "$ENV_DIR/bin/python"
test -x "$ENV_DIR/bin/vllm"
source "$ENV_DIR/bin/activate"
python - <<'PY'
import importlib.metadata as m
import torch

expected = {
    "torch": "2.11.0",
    "transformers": "5.12.1",
    "vllm": "0.24.0",
    "peft": "0.19.1",
}
actual = {name: m.version(name) for name in expected}
assert actual == expected, (actual, expected)
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.is_bf16_supported(), "bf16 unavailable"
print({
    "packages": actual,
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "bf16": torch.cuda.is_bf16_supported(),
})
PY
vllm --version
echo "PASS opd-math serving environment GPU preflight"
