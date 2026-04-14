# HPC Setup Log

Tracking the cluster bootstrap for LegalRagAgent on the WashU Engineering HPC.

## SSH
- Alias: `ssh wustl` (configured in `~/.ssh/config`)
- Login node: `shell.engr.wustl.edu`
- Account: `engr-lab-jacobsn`, partitions: `general-gpu`, `condo-jacobsn`

## Cluster Layout

```
/engrfs/project/jacobsn/hiqbal/
├── src/
│   └── LegalRagAgent/              # This repo (cloned from shrango remote)
│       ├── .venv/                   # Primary venv (Python 3.11, uv-managed, vLLM 0.19.0)
│       ├── chroma_db/               # ChromaDB vector store (686K docs, 3.1 GB)
│       └── ...
├── venvs/
│   └── legalrag-gemma4/             # Gemma 4 venv (vLLM nightly, transformers>=5.5.0)
└── ...

/engrfs/tmp/jacobsn/hiqbal_legalrag/
├── logs/                            # SLURM job output + download logs
├── hf_cache/                        # HuggingFace model cache
│   ├── models--Qwen--Qwen3-8B/         # 16 GB
│   ├── models--Alibaba-NLP--gte-large-en-v1.5/  # Embedding model
│   ├── models--google--gemma-4-E4B-it/  # 15 GB
│   └── models--google--gemma-4-26B-A4B-it/  # 49 GB
└── cache/                           # XDG_CACHE_HOME (vLLM torch compile cache)
```

## Environments

### Primary venv (bootstrap / Qwen / general evals)
- **Path**: `/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/.venv`
- **Python**: 3.11.15 (uv-managed)
- **vLLM**: 0.19.0
- **Use**: Initial Qwen3-8B full-set runs plus general eval / embedding helpers

### Gemma 4 venv (used for completed Gemma 4 runs)
- **Path**: `/engrfs/project/jacobsn/hiqbal/venvs/legalrag-gemma4/`
- **vLLM**: nightly build with Gemma 4 support
- **Use**: Completed Gemma 4 E4B llm_only / golden / rag_simple / rag_snap_hyde runs plus embedding, gap, and vectorless sweeps

## Data

- `datasets/barexam_qa/qa/qa.csv` — 1195 combined QA rows (train+val+test)
- `datasets/barexam_qa/barexam_qa_train.csv` — 686K raw passages (CSV)
- ChromaDB `legal_passages` collection: **686,324 documents** (built 2026-04-07, job 40387, 2.2h)

## Cached Models

| Model | Size | Path | Status |
|---|---|---|---|
| Qwen/Qwen3-8B | 16 GB | `hf_cache/hub/models--Qwen--Qwen3-8B/` | ✅ Cached |
| gte-large-en-v1.5 | ~1 GB | `hf_cache/models--Alibaba-NLP--gte-large-en-v1.5/` | ✅ Cached |
| gemma-4-E4B-it | 15 GB | `hf_cache/models--google--gemma-4-E4B-it/` | ✅ Cached |
| gemma-4-26B-A4B-it | 49 GB | `hf_cache/models--google--gemma-4-26B-A4B-it/` | ✅ Cached |

## Representative SLURM Scripts

| Script | Purpose | Time limit |
|---|---|---|
| `scripts/hpc/slurm_qwen3_8b_llm_only.sh` | Qwen3-8B llm_only baseline (port 8000) | 28h |
| `scripts/hpc/slurm_qwen3_8b_golden.sh` | Qwen3-8B golden_passage eval (port 8001) | 28h |
| `scripts/hpc/slurm_build_embeddings.sh` | Build ChromaDB embeddings (686K docs) | 16h |
| `scripts/hpc/slurm_gemma4_e4b_llm_only.sh` | Gemma 4 E4B full llm_only baseline | 28h |
| `scripts/hpc/slurm_gemma4_e4b_golden.sh` | Gemma 4 E4B full golden_passage eval | 28h |
| `scripts/hpc/slurm_gemma4_e4b_rag_simple.sh` | Gemma 4 E4B full rag_simple eval | 28h |
| `scripts/hpc/slurm_gemma4_e4b_snap_hyde.sh` | Gemma 4 E4B full `rag_snap_hyde` eval | 28h |
| `scripts/hpc/slurm_vectorless.sh` | Gemma 4 E4B vectorless sweep (`direct` / `role` / `elements` / `choice_map` / `hybrid`) | 20h |
| `scripts/hpc/slurm_gap_variants.sh` | Gap input-variant sweep (`gap_hyde*`) | 24h |

## Running Evals

```bash
ssh wustl
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent

# Submit jobs
sbatch scripts/hpc/slurm_gemma4_e4b_llm_only.sh
sbatch scripts/hpc/slurm_gemma4_e4b_snap_hyde.sh
sbatch scripts/hpc/slurm_vectorless.sh

# Monitor
squeue -u hiqbal
grep -c 'PASS\|FAIL' /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/<jobid>.out
grep -c PASS /engrfs/tmp/jacobsn/hiqbal_legalrag/logs/<jobid>.out
```

## Important Lessons Learned

- **`uv run` breaks vLLM**: `uv run` re-resolves deps, breaking vLLM's pinned versions → use `python` directly
- **Home cache fills up**: vLLM torch compile cache fills `$HOME/.cache/` → always set `XDG_CACHE_HOME` to `/engrfs/tmp/...`
- **vLLM startup is slow on NFS**: ~15 min for model load + CUDA kernel compile → health check timeout ≥ 20 min
- **Exclude r28-1801**: Has RTX 2080 (8GB) — too small for any model > 4B
- **Exclude a100-2207 and a100s-2307**: both are now known bad vLLM nodes (`a100-2207` fails vLLM init; `a100s-2307` was the earlier bad A100)
- **Results only on completion**: `eval_harness.py` writes detail log + experiments.jsonl at END of run — killing mid-run loses all results
- **Gemma models in non-standard cache path**: Downloaded via `snapshot_download(cache_dir=...)` → stored at `hf_cache/models--google--*`, not `hf_cache/hub/`

## GPU Nodes Reference

| Node | GPU | VRAM | Notes |
|---|---|---|---|
| a40-2206 | A40 | 48 GB | Primary for 8B models |
| a60-2208/2209 | A6000 | 48 GB | Alternative to A40 |
| a100-2207 | A100 | 80 GB | EXCLUDED — bad vLLM init node |
| a100s-2305/2306/2308 | A100 | 80 GB | For 14B-32B models |
| a100s-2307 | A100 | 80 GB | EXCLUDED — bad vLLM node |
| h100-2405 | H100 | 80 GB | Fastest, for large models |
| r28-1801 | RTX 2080 | 8 GB | EXCLUDED — too small |
