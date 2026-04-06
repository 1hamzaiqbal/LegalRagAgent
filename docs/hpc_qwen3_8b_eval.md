# HPC Qwen3-8B Eval (WashU Cluster)

Minimal path for running a local 8B-model eval on the WashU engineering cluster.

## Why this candidate
Use **`Qwen/Qwen3-8B`** first because it is:
- in the same family as the current 32B reference runs
- easy to fit on an A40 48GB or A100
- a good first vLLM bring-up target before attempting larger 24B-32B models

## Required repo resources
- SLURM script: `scripts/hpc/slurm_vllm_eval_qwen3_8b.sh`
- Existing cluster note: `docs/cluster_workflow.md`

## Assumed cluster layout
- Repo: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent`
- Venv: `/engrfs/project/jacobsn/hiqbal/venvs/legalrag`
- Log dir: `/engrfs/tmp/jacobsn/hiqbal_legalrag/logs`

## One-time prep on cluster
```bash
mkdir -p /engrfs/project/jacobsn/hiqbal/src
cd /engrfs/project/jacobsn/hiqbal/src
git clone https://github.com/shrango/adaptive-plan-and-solve-agent.git LegalRagAgent
cd LegalRagAgent

python3 -m venv /engrfs/project/jacobsn/hiqbal/venvs/legalrag
source /engrfs/project/jacobsn/hiqbal/venvs/legalrag/bin/activate
pip install -U pip uv
uv sync
```

## Run the eval
```bash
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
sbatch scripts/hpc/slurm_vllm_eval_qwen3_8b.sh
```

## Monitor
```bash
squeue -u hiqbal
tail -f /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/<jobid>.out
tail -f /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/vllm_qwen3_8b_<jobid>.log
```

## If Qwen3-8B works
Best next local-model targets:
- `Qwen/Qwen3-14B`
- `google/gemma-3-27b-it` (cleaner on A100 80GB than A40)
- `Qwen/Qwen3-32B` (A100 80GB / quantized / tensor-parallel territory)
