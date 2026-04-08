# HPC Setup Log

Tracking the cluster bootstrap for LegalRagAgent on the WashU Engineering HPC.

## SSH
- Alias: `ssh wustl` (configured in `~/.ssh/config`)
- Login node: `shell.engr.wustl.edu`

## Cluster Layout

```
/engrfs/project/jacobsn/hiqbal/
├── src/
│   ├── LegalRagAgent/         # This repo (cloned from shrango remote)
│   ├── hullclip/              # Existing - do not touch
│   ├── satclip/               # Existing - archived
│   └── TTE/                   # Existing - active project
├── venvs/
│   └── legalrag/              # Python 3.11.15 venv (uv-managed)
└── ...

/engrfs/tmp/jacobsn/hiqbal_legalrag/
├── logs/                      # SLURM job output
└── hf_cache/                  # HuggingFace model cache (Qwen3-8B etc)
```

## Environment

- **Python**: 3.11.15 (installed via `uv python install 3.11`)
- **Venv**: `/engrfs/project/jacobsn/hiqbal/venvs/legalrag`
- **uv**: Available at `~/.local/bin/uv` (installed via `python3 -m pip install --user uv`)
- **Deps**: Installed via `uv sync` from the repo's `pyproject.toml`
- **vLLM**: Needs separate install (`pip install vllm` in the venv on a compute node)

## Data Downloaded

- `datasets/barexam_qa/qa/qa.csv` — 1195 combined QA rows (train+val+test)
- `datasets/barexam_qa/qa/{train,validation,test}.csv` — Individual splits
- `datasets/barexam_qa/train.tsv` — 686K raw passages (TSV)
- `datasets/barexam_qa/barexam_qa_train.csv` — Same passages as CSV

## Model Cache

- `Qwen/Qwen3-8B` — downloading to `/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache/`
  - ~16GB total, downloading unauthenticated (set HF_TOKEN for faster downloads)

## Embedding Model

- `Alibaba-NLP/gte-large-en-v1.5` — NOT yet cached on cluster
  - Needed for any RAG-based eval modes
  - Not needed for `llm_only` or `golden_passage` modes

## SLURM Scripts

- `scripts/hpc/slurm_vllm_eval_qwen3_8b_baseline_golden.sh` — Baseline + golden passage (recommended first run)
- `scripts/hpc/slurm_vllm_eval_qwen3_8b.sh` — Baseline only

## Running Evals

```bash
# SSH in
ssh wustl

# Submit the baseline + golden job
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
sbatch scripts/hpc/slurm_vllm_eval_qwen3_8b_baseline_golden.sh

# Monitor
squeue -u hiqbal
tail -f /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/<jobid>.out
```

## Install vLLM (on compute node)

```bash
srun -p general-gpu -A engr-lab-jacobsn --gpus a40:1 -c 8 --mem=64G -t 2:00:00 --pty /bin/bash
source /engrfs/project/jacobsn/hiqbal/venvs/legalrag/bin/activate
pip install vllm
```

## Notes

- Login node has no GPU — use compute nodes for GPU ops and vLLM install
- System Python is 3.9; the venv uses uv-managed Python 3.11
- The repo was cloned from `git@github.com:shrango/adaptive-plan-and-solve-agent.git`
- `.env` on cluster is minimal: `LLM_PROVIDER=cluster-vllm` (SLURM scripts override via env vars)
- Existing projects (TTE, HullCLIP) use conda envs, not this venv — no conflicts
