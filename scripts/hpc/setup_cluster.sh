#!/bin/bash
# One-time cluster setup for LegalRagAgent.
# Run this from the login node after `uv sync` completes.
# For vLLM, run on a compute node instead:
#   srun -p general-gpu -A engr-lab-jacobsn --gpus a40:1 -c 8 --mem=64G -t 2:00:00 --pty /bin/bash
#   bash scripts/hpc/setup_cluster.sh

set -euo pipefail

REPO=/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
VENV="$REPO/.venv"
HF_CACHE=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
LOG_DIR=/engrfs/tmp/jacobsn/hiqbal_legalrag/logs

echo "=== Checking directories ==="
mkdir -p "$LOG_DIR" "$HF_CACHE"

echo "=== Checking venv ==="
source "$VENV/bin/activate"
python --version
echo "pip: $(pip --version)"

echo "=== Installing vLLM ==="
pip install vllm
echo "vLLM installed: $(vllm --version 2>&1 || python -c 'import vllm; print(vllm.__version__)')"

echo "=== Checking data ==="
if [ -f "$REPO/datasets/barexam_qa/qa/qa.csv" ]; then
    echo "qa.csv exists: $(wc -l < "$REPO/datasets/barexam_qa/qa/qa.csv") lines"
else
    echo "WARNING: qa.csv missing! Run download_data.py first."
fi

echo "=== Checking model cache ==="
if [ -d "$HF_CACHE/models--Qwen--Qwen3-8B" ]; then
    echo "Qwen3-8B cache: $(du -sh "$HF_CACHE/models--Qwen--Qwen3-8B" | cut -f1)"
else
    echo "WARNING: Qwen3-8B not cached. It will download on first vllm serve."
fi

echo "=== Quick import test ==="
cd "$REPO"
python -c "
from eval_config import EvalConfig, load_questions
from llm_config import get_provider_info
print('eval_config + llm_config imports OK')
config = EvalConfig(mode='llm_only', provider='cluster-vllm', questions='3', dataset='barexam')
qs = load_questions(config)
print(f'Loaded {len(qs)} questions successfully')
"

echo "=== Setup complete ==="
echo "Next: sbatch scripts/hpc/slurm_vllm_eval_qwen3_8b_baseline_golden.sh"
