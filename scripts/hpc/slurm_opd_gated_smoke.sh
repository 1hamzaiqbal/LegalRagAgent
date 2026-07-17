#!/bin/bash
#SBATCH --job-name=opd_gated_smoke
#SBATCH --partition=general-gpu
#SBATCH --gpus=a100-sxm4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --exclude=a100s-2307,a100-2207,r28-1801
#SBATCH -A engr-lab-jacobsn
#SBATCH --output=/engrfs/tmp/jacobsn/hiqbal_legalrag/opd_gated_smoke_%j.out

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-opd-distillation
ENV_DIR=/engrfs/project/jacobsn/hiqbal/envs/opd_lane

test -d "$REPO/.git" || test -f "$REPO/.git"
test -x "$ENV_DIR/bin/python"

export HF_HOME=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPD_SMOKE_MODE=opd_gated
export OPD_SMOKE_PORT=$((8800 + SLURM_JOB_ID % 100))
export OPD_TEACHER_GPU_FRAC=0.55
export OPD_READY_TRIES=600

nvidia-smi --query-gpu=name,memory.total --format=csv
source "$ENV_DIR/bin/activate"
python -c "import peft, requests, vllm; print('opd env ok', vllm.__version__)"

cd "$REPO"
bash scripts/opd/smoke_test.sh
echo "ALL DONE opd-gated-smoke"
