# Local API Mirror Setup - 2026-05-12

## Recommendation

Set up a local mirror on the Windows machine if it has at least 80 GB free. The
current preferred path is local execution with API providers and local Chroma,
not WUSTL/SLURM. WSL Ubuntu is preferred over native PowerShell for the repo
scripts and Chroma path handling.

Last checked populated artifact sizes:

- `chroma_db/`: about 33 GB.
- `datasets/`: about 4.8 GB.
- Retrieval/generation caches: small now, but budget another 10-20 GB as full
  sweeps accumulate.

The current Mac checkout has only about 17 GB free, so it is not a good target
unless space is cleared first.

## What Local Helps With

- API-backed `llm_only`, `rag_simple`, `rag_hyde`, `snap_hyre`,
  `golden_passage`, `golden_plus_neighbors`, and `rag_rewrite` runs.
- Retrieval-cache replay and top-k slicing once the passage ids are cached.
- Source-gated log validation with `scripts/analyze_detail_flags.py`,
  `scripts/audit_retrieval_cache.py`, and `scripts/compile_retrieval_cache_matrix.py`.
- Faster iteration without scheduler queues or SSH control-socket failures.

## What Local Does Not Remove

- First-time embedding can be slower on CPU-only local hardware.
- Exact historical `google/gemma-4-E4B-it` rows still need vLLM or another
  exact provider. For the current API-only comprehensive package, use the
  small-model replacement row instead of trying to force this historical axis.
  Do not use OpenRouter `or-gemma3n-e4b` for this row; it is Gemma 3n E4B, not
  Gemma 4 E4B.

## Data Population

Do not assume the Windows local machine can access WUSTL. Prefer downloading
datasets and re-embedding locally on that machine. If another
already-populated machine is directly reachable, copying `datasets/` and
`chroma_db/` from that machine is fine, but this branch no longer relies on
WUSTL SSH.

## Local Smoke

Use API providers and local Chroma:

```bash
export CHROMA_DB_DIR="$PWD/chroma_db"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DISABLE_CROSS_ENCODER=0
export LLM_MAX_COMPLETION_TOKENS=2048

uv run python eval/eval_harness.py \
  --mode snap_hyre \
  --provider or-gemma4-26b \
  --dataset barexam \
  --questions 1 \
  --retrieval-k 3 \
  --tag local-api-mirror-smoke
```

Leave `DISABLE_CROSS_ENCODER=0` for canonical retrieval caches and promoted
answer rows so retrieval uses `cross-encoder/ms-marco-MiniLM-L-6-v2`. Use
`DISABLE_CROSS_ENCODER=1` only for explicitly labeled dense-only speed smokes.

Then validate the generated detail log:

```bash
uv run python scripts/analyze_detail_flags.py logs/<detail-log>.jsonl
```

## Promotion Rule

Local rows should use the same source gates as HPC rows: detail JSONL,
`analyze_detail_flags.py`, retrieval-cache/qrel audits where applicable, and a
`docs/signoff_log.md` entry before the row is cited in tables.
