# Local API Mirror Setup - 2026-05-12

## Recommendation

Set up a local mirror if the machine has at least 80 GB free. This does not
replace the current HPC run, but it removes SSH/SLURM from API-backed answer
sweeps, cache replay, plotting, and log validation.

Current artifact sizes on WUSTL:

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

- Initial transfer still needs one reliable copy path from WUSTL or another
  already-populated machine.
- First-time embedding retrieval can be slower on CPU-only local hardware.
- Exact historical `google/gemma-4-E4B-it` rows still need vLLM or another
  exact provider; the API E4B row is `or-gemma3n-e4b`.

## Sync Commands

From the local checkout root:

```bash
mkdir -p datasets chroma_db caches

rsync -aH --info=progress2 \
  -e 'ssh -o ControlMaster=no -o ControlPath=none' \
  wustl:/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/datasets/ \
  datasets/

rsync -aH --info=progress2 \
  -e 'ssh -o ControlMaster=no -o ControlPath=none' \
  wustl:/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent/chroma_db/ \
  chroma_db/
```

Optional cache sync after HPC jobs finish:

```bash
rsync -aH --info=progress2 \
  -e 'ssh -o ControlMaster=no -o ControlPath=none' \
  wustl:/engrfs/project/jacobsn/hiqbal/src/LegalRagAgent-snap-hyre-comprehensive/caches/ \
  caches/
```

## Local Smoke

Use API providers and local Chroma:

```bash
export CHROMA_DB_DIR="$PWD/chroma_db"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export DISABLE_CROSS_ENCODER=1
export LLM_MAX_COMPLETION_TOKENS=768

uv run python eval/eval_harness.py \
  --mode snap_hyre \
  --provider or-gemma3n-e4b \
  --dataset barexam \
  --questions 1 \
  --retrieval-k 3 \
  --tag local-api-mirror-smoke
```

Then validate the generated detail log:

```bash
uv run python scripts/analyze_detail_flags.py logs/<detail-log>.jsonl
```

## Promotion Rule

Local rows should use the same source gates as HPC rows: detail JSONL,
`analyze_detail_flags.py`, retrieval-cache/qrel audits where applicable, and a
`docs/signoff_log.md` entry before the row is cited in tables.
