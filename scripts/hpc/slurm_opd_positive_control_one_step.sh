#!/bin/bash
#SBATCH --job-name=opsd_pc_1step
#SBATCH --partition=general-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opsd_pc_1step_%j.out

set -euo pipefail
REPO="${OPD_IDENT_REPO:-/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-math}"
UPSTREAM="${OPD_UPSTREAM_REPO:-/engrfs/project/jacobsn/hiqbal/literature/legalrag/repos/OPSD}"
ENV_DIR="${OPD_POSITIVE_ENV:-/engrfs/project/jacobsn/hiqbal/envs/opd_positive_control_7448751}"
DATA_ROOT="${OPD_IDENT_DATA_ROOT:-/engrfs/project/jacobsn/hiqbal/data/legalrag/opd_identifiability_v1}"
RUN_ROOT="${OPD_IDENT_RUN_ROOT:-/engrfs/project/jacobsn/hiqbal/artifacts/legalrag/opd_identifiability_v1}"
HF_HOME="${OPD_IDENT_HF_HOME:-/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache}"
CACHE_ROOT="${OPD_IDENT_CACHE_ROOT:-/engrfs/tmp/jacobsn/hiqbal_legalrag/runtime_caches/opd_identifiability_v1}"
EXPECTED_COMMIT="${OPD_IDENT_EXPECTED_COMMIT:?set OPD_IDENT_EXPECTED_COMMIT at submission}"
CONFIG="${OPD_IDENT_ONE_STEP_CONFIG:-$REPO/configs/opd_math/identifiability_v1_one_step_retry3.json}"

JOB_CACHE="$CACHE_ROOT/job_${SLURM_JOB_ID:?missing Slurm job id}"
export XDG_CACHE_HOME="$JOB_CACHE/xdg"
export VLLM_CACHE_ROOT="$JOB_CACHE/vllm"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torchinductor"
export TRITON_CACHE_DIR="$JOB_CACHE/triton"
export CUDA_CACHE_PATH="$JOB_CACHE/cuda"
export TORCH_HOME="$JOB_CACHE/torch"
export TORCH_EXTENSIONS_DIR="$JOB_CACHE/torch_extensions"
export TMPDIR="$JOB_CACHE/tmp"
mkdir -p \
  "$XDG_CACHE_HOME" \
  "$VLLM_CACHE_ROOT" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$CUDA_CACHE_PATH" \
  "$TORCH_HOME" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR"
case "$JOB_CACHE" in
  /engrfs/tmp/jacobsn/hiqbal_legalrag/runtime_caches/opd_identifiability_v1/job_*) ;;
  *) echo "invalid per-job runtime-cache root: $JOB_CACHE" >&2; exit 1 ;;
esac

test -z "$(git -C "$REPO" status --porcelain=v1)"
test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -f "$CONFIG"
test "${SLURM_JOB_NUM_NODES:?missing Slurm node count}" = "1"
test "${SLURM_GPUS_ON_NODE:?missing Slurm GPU count}" = "4"
"$ENV_DIR/bin/python" - <<'PY'
import torch
if torch.cuda.device_count() != 4:
    raise RuntimeError(f"expected exactly four visible CUDA devices, got {torch.cuda.device_count()}")
for index in range(4):
    name = torch.cuda.get_device_name(index)
    if "A6000" not in name:
        raise RuntimeError(f"GPU {index} is not an A6000: {name}")
PY

OUT="$RUN_ROOT/one_step/job_${SLURM_JOB_ID}"
test ! -e "$OUT"
mkdir -p "$OUT"
"$ENV_DIR/bin/python" "$REPO/scripts/opd/prepare_opsd_execution_tree.py" \
  --upstream "$UPSTREAM" \
  --target "$OUT/execution_tree" \
  --repository-commit "$EXPECTED_COMMIT"

MODEL="$($ENV_DIR/bin/python - "$HF_HOME" <<'PY'
import os, sys
os.environ["HF_HOME"] = sys.argv[1]
os.environ["HF_HUB_OFFLINE"] = "1"
from huggingface_hub import snapshot_download
print(snapshot_download("Qwen/Qwen3-1.7B", revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", local_files_only=True))
PY
)"

export HF_HOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
LEGALRAG_OPSD_TRAIN_PARQUET="$($ENV_DIR/bin/python - "$CONFIG" "$DATA_ROOT" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
print(config.get("training_data", {}).get("parquet_glob", f"{sys.argv[2]}/opsd_train/**/*.parquet"))
PY
)"
export LEGALRAG_OPSD_TRAIN_PARQUET
export LEGALRAG_OPSD_AIME24_PARQUET="$DATA_ROOT/aime24/**/*.parquet"
export NCCL_P2P_DISABLE=1
PORT="$((12000 + SLURM_JOB_ID % 20000))"

"$ENV_DIR/bin/python" "$REPO/scripts/opd/positive_control_one_step.py" \
  --config "$CONFIG" \
  --repository-commit "$EXPECTED_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID" \
  --env-dir "$ENV_DIR" \
  --execution-tree "$OUT/execution_tree" \
  --model-dir "$MODEL" \
  --output-root "$OUT" \
  --port "$PORT"

echo "PASS in-job one-step update gate: $OUT/in_job_gate.json"
