# Adaptive HyRE Legal Runbook (2026-05-09)

## Scope

Adaptive HyRE is currently a legal-only evaluation target. Use these four
benchmarks for the next sweep:

| Dataset | Why it is in scope | Expected adaptive route |
|---|---|---|
| `barexam` | Legal multiple choice with BarExam corpus coverage. | `option_grounding` |
| `housing` | Legal yes/no statutory QA with state metadata. | `state_filter` |
| `casehold` | Holding-selection task where retrieval gains may not convert to answers. | `option_grounding` |
| `legalbench_scalr` | Holding-selection / SCALR legal benchmark with local and cluster collection support. | `option_grounding` |

Do not use MuSiQue in the headline adaptive HyRE sweep. It is non-legal and
requires short-span multi-hop machinery that would contort the current legal
RAG harness.

## Modes

Run these modes for option-style datasets:

```bash
RUN_SPECS="rag_simple rag_snap_hyde_2call snap_hyre_option adaptive_snap_hyre"
```

Run these modes for HousingQA:

```bash
RUN_SPECS="rag_state_filter snap_hyre_state adaptive_snap_hyre"
```

## Cluster Launches

Default path: cluster vLLM Gemma 4 26B, no paid API calls.

```bash
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent

sbatch --job-name=hyre-barexam \
  --export=ALL,DATASET=barexam,N_QUESTIONS=200,RETRIEVAL_K=5 \
  scripts/hpc/slurm_adaptive_hyre_legal.sh

sbatch --job-name=hyre-housing \
  --export=ALL,DATASET=housing,N_QUESTIONS=200,RETRIEVAL_K=5 \
  scripts/hpc/slurm_adaptive_hyre_legal.sh

sbatch --job-name=hyre-casehold \
  --export=ALL,DATASET=casehold,N_QUESTIONS=200,RETRIEVAL_K=5 \
  scripts/hpc/slurm_adaptive_hyre_legal.sh

sbatch --job-name=hyre-scalr \
  --export=ALL,DATASET=legalbench_scalr,N_QUESTIONS=200,RETRIEVAL_K=5 \
  scripts/hpc/slurm_adaptive_hyre_legal.sh
```

OpenRouter fallback for quick debugging only:

```bash
sbatch --job-name=hyre-scalr-api \
  --export=ALL,USE_VLLM=0,PROVIDER=or-gemma4-26b,DATASET=legalbench_scalr,N_QUESTIONS=50,RETRIEVAL_K=5 \
  scripts/hpc/slurm_adaptive_hyre_legal.sh
```

## Validation Gates

Every landed adaptive/snap-HyRE detail log should pass:

```bash
python scripts/analyze_detail_flags.py logs/<detail_log>.jsonl
python scripts/audit_adaptive_hyre_logs.py logs/<detail_log>.jsonl
```

The adaptive audit should show:

- dataset is one of `barexam`, `housing`, `casehold`, `legalbench_scalr`
- no `error` rows
- no empty retrieval
- no missing `gold_retrieved` field
- no `snap_hyre_parse_ok=False` rows
- route distribution matches the expected route for the dataset
- call count is near the intended method cost

Small local smoke logs may warn on `small_n`; do not promote those as result
claims. Treat them only as harness health checks.

## Pairwise Tests

After logs land locally, compare adaptive controls against the nearest baseline:

```bash
uv run python scripts/compute_mcnemar.py \
  logs/eval_rag_simple_<provider>_<timestamp>_detail.jsonl \
  logs/eval_adaptive_snap_hyre_<provider>_<timestamp>_detail.jsonl
```

For HousingQA, compare against `rag_state_filter` and `snap_hyre_state` as
well as generic `rag_simple` if a matched run exists.

## Interpretation

The immediate question is not whether one prompt wins every task. The useful
signal is whether a single legal adaptive policy can choose the right
intervention family:

- answer-option conversion for BarExam, CaseHOLD, and LegalBench-SCALR
- state-constrained retrieval for HousingQA
- aligned HyDE reranking for legal datasets without explicit option/state
  structure

If a dataset fails while its retrieval metrics improve, treat that as an answer
conversion problem rather than a retrieval failure. If state-filtered Housing
fails with empty retrieval, reject the run before comparing accuracy.
