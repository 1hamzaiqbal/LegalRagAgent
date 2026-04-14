# LegalRAG Cluster Workflow

Historical/bootstrap note: the cluster is already set up. For the current realized paths, venvs, bad-node list, and completed run state, use `docs/hpc_setup_log.md` and `docs/hpc_throughput.md`.

Practical plan for running LegalRagAgent on the WashU engineering cluster, using the same overall pattern as HullCLIP: **edit/orchestrate from the MacBook, push code to cluster, run heavy jobs under SLURM**.

## Why use the cluster?
- Full evals stop competing with RL-on-RL and local workstation usage
- We can run open-weight models without API rate limits once a good model is chosen
- Embedding-model sweeps become easier when a dedicated GPU window is available

## Quick reference
- SSH target: `ssh hiqbal@shell.engr.wustl.edu`
- Account: `engr-lab-jacobsn`
- Dedicated partition: `condo-jacobsn` (A40 48GB)
- Shared partition: `general-gpu` (A40 / A100 availability varies)
- Preferred orchestration pattern: MacBook local edits -> `git push` -> SSH -> `git pull` -> `sbatch`

## Important first-time SSH note
From this machine, a non-interactive SSH check currently fails at **host-key verification**, which usually means the host has not been trusted yet on this client.

First interactive connection should be done manually:
```bash
ssh hiqbal@shell.engr.wustl.edu
```
Accept the host key when prompted.

If you want to pre-seed `known_hosts` manually:
```bash
ssh-keyscan -H shell.engr.wustl.edu >> ~/.ssh/known_hosts
```

A useful MacBook `~/.ssh/config` snippet:
```sshconfig
Host wustl-shell
  HostName shell.engr.wustl.edu
  User hiqbal
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 10m
```

## Recommended repo layout on cluster
```text
/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
/engrfs/project/jacobsn/hiqbal/venvs/legalrag
/engrfs/tmp/jacobsn/hiqbal_legalrag/logs
/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
```

## Environment setup
The repo already uses `pyproject.toml` + `uv`, so the simplest cluster bootstrap is:

```bash
mkdir -p /engrfs/project/jacobsn/hiqbal/src
cd /engrfs/project/jacobsn/hiqbal/src
git clone <YOUR-REMOTE-URL> LegalRagAgent
cd LegalRagAgent

python3 -m venv /engrfs/project/jacobsn/hiqbal/venvs/legalrag
source /engrfs/project/jacobsn/hiqbal/venvs/legalrag/bin/activate
pip install -U pip uv
uv sync
```

Recommended cache/env vars in SLURM jobs:
```bash
export HF_HOME=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## Two useful execution modes

### 1. Batch evals under SLURM
Best for full benchmark runs.

```bash
#!/bin/bash
#SBATCH -p general-gpu
#SBATCH -A engr-lab-jacobsn
#SBATCH --gpus a40:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/%j.out

source /engrfs/project/jacobsn/hiqbal/venvs/legalrag/bin/activate
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent

export HF_HOME=/engrfs/tmp/jacobsn/hiqbal_legalrag/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

uv run python eval/eval_harness.py \
  --mode llm_only \
  --provider or-qwen3-14b \
  --questions full
```

This is the easiest path for reproducing the existing API-driven eval workflow on the cluster.

### 2. Unlimited local-query mode via vLLM
Best once we pick an open-weight model we want to hammer repeatedly.

The important codebase fact: `llm_config.py` already supports **raw OpenAI-compatible env vars** via:
- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

So we do **not** need a major refactor to talk to a cluster-hosted model server.

Example vLLM server launch on a compute node:
```bash
vllm serve Qwen/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192
```

Then point LegalRAG at it:
```bash
export LLM_PROVIDER=cluster-vllm
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export LLM_API_KEY=DUMMY_KEY
export LLM_MODEL=Qwen/Qwen3-14B

uv run python eval/eval_harness.py \
  --mode llm_only \
  --provider cluster-vllm \
  --questions full
```

If `cluster-vllm` is not a registered provider, the code falls back to the raw env vars above.

## Model-sizing guidance
Good first candidates for single-GPU cluster runs:
- `Qwen3-8B`
- `Qwen3.5-9B`
- `Nemotron Nano 9B`
- `Llama-3.1-8B`
- likely `Qwen3-14B` on a 48GB A40 / 80GB A100

Heavier option:
- `Qwen3-32B` is more realistic on an A100 80GB or with quantization / tensor parallelism

Practical recommendation: **start with Qwen3.5-9B or Qwen3-14B** for cluster local-inference bring-up.

## Interactive debugging
```bash
srun -p condo-jacobsn -A engr-lab-jacobsn --gpus a40:1 -c 8 --mem=64G -t 2:00:00 --pty /bin/bash
```

Useful checks:
```bash
squeue -u hiqbal
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
nvidia-smi
```

## Suggested next cluster milestone
Before launching any new full evals, the next useful step is:
1. bootstrap the repo on cluster
2. bring up one local model (`Qwen3.5-9B` or `Qwen3-14B`)
3. run a **3-question sanity check** through `eval/eval_harness.py`
4. only then queue a full eval

That gets us to the "unlimited querying" setup without prematurely burning compute on the whole matrix.
