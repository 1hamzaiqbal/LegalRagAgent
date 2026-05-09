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
RUN_SPECS="rag_simple rag_snap_hyde_2call snap_hyre_option adaptive_snap_hyre adaptive_snap_hyre_anchor adaptive_snap_hyre_diverse"
```

Run these modes for HousingQA:

```bash
RUN_SPECS="rag_state_filter snap_hyre_state adaptive_snap_hyre adaptive_snap_hyre_anchor adaptive_snap_hyre_diverse"
```

`adaptive_snap_hyre_anchor` is the next low-cost generalization probe: it keeps
the same two LLM calls as `adaptive_snap_hyre`, but retrieves with both the
generated HyRE passage and the raw question/intermediate prompt. This tests
whether reasoning-shaped retrieval needs a lexical/task anchor on datasets
where generated passages can drift away from option text, state metadata, or
the original fact pattern.

`adaptive_snap_hyre_diverse` keeps the same two LLM calls again, but retrieves
with three query views: generated HyRE passage, raw task anchor, and sanitized
snap reasoning. This directly tests query-diversity lift without increasing the
model-call budget.

## Cluster Launches

Default path: cluster vLLM Gemma 4 26B, no paid API calls.
The SLURM script chooses a per-job default vLLM port from `SLURM_JOB_ID`; set
`PORT=...` only when debugging a single job manually.

Before submitting, verify the cluster checkout is current and runnable:

```bash
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
scripts/hpc/prepare_adaptive_hyre_cluster.sh
```

This prep check also verifies the four legal Chroma collections:
`legal_passages`, `housing_statutes`, `casehold_holdings`, and
`legalbench_scalr_holdings`. If you only need a code-path check without local
cluster data mounted, set `CHECK_CHROMA=0`.

For a fast API-free mode wiring check only:

```bash
python scripts/smoke_adaptive_hyre_modes.py
```

Submit the full four-dataset sweep:

```bash
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
scripts/hpc/submit_adaptive_hyre_legal_sweep.sh
```

One-command cluster launch with branch freshness and preflight:

```bash
cd /engrfs/project/jacobsn/hiqbal/src/LegalRagAgent
AUTO_PULL=1 scripts/hpc/launch_adaptive_hyre_sweep.sh
```

Preview the exact `sbatch` commands without launching:

```bash
DRY_RUN=1 scripts/hpc/submit_adaptive_hyre_legal_sweep.sh
```

Or submit selected datasets:

```bash
scripts/hpc/submit_adaptive_hyre_legal_sweep.sh housing casehold
```

To rerun only selected modes on a dataset:

```bash
RUN_SPECS="snap_hyre_state adaptive_snap_hyre adaptive_snap_hyre_anchor adaptive_snap_hyre_diverse" \
  scripts/hpc/submit_adaptive_hyre_legal_sweep.sh housing
```

The equivalent explicit commands are:

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
USE_VLLM=0 PROVIDER=or-gemma4-26b N_QUESTIONS=50 \
  scripts/hpc/submit_adaptive_hyre_legal_sweep.sh legalbench_scalr
```

Monitor queue state, recent SLURM stdout, adaptive detail logs, and the
postprocessed non-smoke summary:

```bash
scripts/hpc/monitor_adaptive_hyre_sweep.sh
```

Each completed dataset job also writes postprocess summaries to
`$LOG_DIR/adaptive_hyre_<provider>_<dataset>_n<N>_k<K>_<job>.md` and a
machine-readable JSON companion with the same stem. Keep generated cluster
summaries out of the repo checkout until we intentionally promote a specific
one into `docs/`.

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

For a sweep-level summary over landed logs:

```bash
python scripts/postprocess_adaptive_hyre_sweep.py \
  --provider cluster-vllm \
  --output docs/adaptive_hyre_sweep_latest.md
```

For automation, also emit JSON:

```bash
python scripts/postprocess_adaptive_hyre_sweep.py \
  --provider cluster-vllm \
  --output docs/adaptive_hyre_sweep_latest.md \
  --json-output docs/adaptive_hyre_sweep_latest.json
```

To make monitors fail until every expected adaptive mode has landed:

```bash
python scripts/postprocess_adaptive_hyre_sweep.py --min-n 20 --provider cluster-vllm --require-ready
```

By default this ignores smoke logs with fewer than 20 rows. For harness-health
inspection only, include smoke logs with:

```bash
python scripts/postprocess_adaptive_hyre_sweep.py --min-n 1
```

The summary includes coverage and paired tests for `adaptive_snap_hyre` versus
`adaptive_snap_hyre_anchor` and `adaptive_snap_hyre_anchor` versus
`adaptive_snap_hyre_diverse`, so anchor probes should show up as first-class
adaptive comparisons rather than orphaned modes. It also emits an adaptive
parity frontier: for each dataset/provider, the best adaptive policy is
compared against the best available control with accuracy and average LLM-call
cost.

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
