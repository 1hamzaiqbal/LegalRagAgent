# Local Snap-HyRE Runner Scripts

These scripts mirror the comprehensive Snap-HyRE SLURM helpers for a local
machine with populated `datasets/`, `chroma_db/`, and API keys in `.env`.

On Windows, run these from WSL Ubuntu if possible. Native PowerShell can run the
underlying Python commands, but these helper scripts are Bash-first.

Use them in this order:

1. `run_api_smoke.sh` - provider and harness smoke, no result promotion.
2. `build_retrieval_caches.sh` - raw/golden retrieval caches plus qrel
   alignment and top-k matrix.
3. `build_generation_caches.sh` - HyDE/Snap-HyRE generation caches plus
   retrieval caches from those generated passages.
4. `run_answer_cell.sh` - one dataset/model answer ladder at a time.
5. `build_result_package.sh` - source-gated package status tables and optional
   plots after caches/logs exist.

The scripts intentionally default to small or bounded runs where possible.
Set `QUESTIONS=full` only after smokes and cache audits are clean.

Example generation-cache pass:

```bash
PROVIDER=or-gemma4-26b MODEL_LABEL=gemma4-26b QUESTIONS=50 \
  scripts/local/build_generation_caches.sh
```

Common environment:

```bash
export CHROMA_DB_DIR="$PWD/chroma_db"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DISABLE_CROSS_ENCODER=1
export LLM_MAX_COMPLETION_TOKENS=768
```

Required `.env` keys for the planned providers:

```env
OPENROUTER_API_KEY=...
GROQ_API_KEY=...
```

After any clean batch, rebuild the package status:

```bash
scripts/local/build_result_package.sh
```
