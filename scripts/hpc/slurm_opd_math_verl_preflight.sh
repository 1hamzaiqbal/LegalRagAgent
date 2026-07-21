#!/bin/bash
#SBATCH --job-name=opd_verl_preflight
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_verl_preflight_%j.out

set -euo pipefail
REPO="${OPD_MATH_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
VERL="${OPD_MATH_VERL_CHECKOUT:-/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/verl}"
ENV_DIR="${OPD_MATH_VERL_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_math_verl}"
RUN_ROOT="${OPD_MATH_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_math}"
HF_CACHE="${OPD_MATH_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FREEZE="$RUN_ROOT/environment_freezes/$COMMIT/upstream_verl.freeze.txt"

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$VERL" rev-parse HEAD)" = "6a6242f3d8ec7d9f8b4936f4905144707d91fe3b"
test -z "$(git -C "$VERL" status --porcelain=v1 --untracked-files=no)"
test -x "$ENV_DIR/bin/python"
test -f "$FREEZE"

"$ENV_DIR/bin/python" "$REPO/scripts/opd_math/verify_environment.py" \
  --environment-root "$ENV_DIR" \
  --commit-freeze "$FREEZE" \
  --expected-commit "$COMMIT" \
  --freeze-kind upstream_verl

HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$ENV_DIR/bin/python" - <<'PY'
import json
import torch
import ray
import transformers
import vllm
import verl
from huggingface_hub import snapshot_download

student = snapshot_download(
    "Qwen/Qwen3-1.7B",
    revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    local_files_only=True,
)
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(f"expected exactly two visible CUDA devices, got {torch.cuda.device_count()}")
payload = {
    "status": "passed",
    "cuda_devices": torch.cuda.device_count(),
    "cuda_names": [torch.cuda.get_device_name(i) for i in range(2)],
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "verl": getattr(verl, "__version__", "unknown"),
    "ray": ray.__version__,
    "student_snapshot": student,
}
print(json.dumps(payload, sort_keys=True))
PY

echo "PASS pinned-veRL two-GPU environment preflight"
