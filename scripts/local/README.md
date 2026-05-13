# Local Snap-HyRE Runner Scripts

These scripts mirror the comprehensive Snap-HyRE SLURM helpers for a local
machine with populated `datasets/`, `chroma_db/`, and API keys in `.env`.

Use them in this order:

1. `run_api_smoke.sh` - provider and harness smoke, no result promotion.
2. `build_retrieval_caches.sh` - raw/golden retrieval caches plus qrel
   alignment and top-k matrix.
3. `run_answer_cell.sh` - one dataset/model answer ladder at a time.

The scripts intentionally default to small or bounded runs where possible.
Set `QUESTIONS=full` only after smokes and cache audits are clean.

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

