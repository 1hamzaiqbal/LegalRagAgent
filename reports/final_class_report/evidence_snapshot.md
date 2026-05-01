# Evidence Snapshot - 2026-05-01

Purpose: compact audit trail for the report headline rows. Source-gating
follows `docs/README.md`: `docs/signoff_log.md` first, then the dataset and
mechanism audit docs, then detail logs.

## Validated With `scripts/compute_mcnemar.py`

All commands were run from the repository root with:

```bash
UV_CACHE_DIR=/tmp/uv-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

| Pair | N | Baseline | Treatment | Delta | b/c | McNemar p | Source logs |
|---|---:|---:|---:|---:|---:|---:|---|
| MuSiQue `rag_simple` -> `rag_snap_hyde_2call` | 200 | 27.5% | 37.0% | +9.5pp | 33/14 | 0.0079427 | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0040_detail.jsonl` |
| MuSiQue `rag_simple` top-5 -> top-1 | 200 | 27.5% | 13.0% | -14.5pp | 3/32 | 4.17698e-07 | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; `logs/eval_rag_simple_groq-llama70b_20260428_0011_detail.jsonl` |
| BarExam `rag_simple` top-5 -> top-1 | 200 | 82.5% | 83.0% | +0.5pp | 18/17 | 1.0 | `logs/eval_rag_simple_or-gemma4-26b_20260428_0231_detail.jsonl`; `logs/eval_rag_simple_or-gemma4-26b_20260428_0138_detail.jsonl` |
| SCALR `rag_simple` top-5 -> top-1 | 200 | 77.0% | 59.5% | -17.5pp | 3/38 | 1.04792e-08 | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl`; `logs/eval_rag_simple_groq-llama70b_20260429_2159_detail.jsonl` |
| SCALR `rag_simple` top-5 -> top-10 | 200 | 77.0% | 77.0% | 0.0pp | 8/8 | 1.0 | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl`; `logs/eval_rag_simple_groq-llama70b_20260430_0054_detail.jsonl` |
| HousingQA `rag_simple` top-1 -> top-10 | 200 | 50.5% | 58.0% | +7.5pp | 38/23 | 0.0721774 | `logs/eval_rag_simple_or-gemma4-26b_20260430_0415_detail.jsonl`; `logs/eval_rag_simple_or-gemma4-26b_20260430_0542_detail.jsonl` |
| HousingQA `rag_simple` top-10 -> `rag_snap_hyde_2call` | 200 | 58.0% | 57.0% | -1.0pp | 26/28 | 0.8919232 | `logs/eval_rag_simple_or-gemma4-26b_20260430_0542_detail.jsonl`; `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260430_0644_detail.jsonl` |
| SCALR `rag_simple` top-5 -> `rag_snap_hyde_2call` | 200 | 77.0% | 75.0% | -2.0pp | 8/12 | 0.5034447 | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl`; `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_1520_detail.jsonl` |
| SCALR `rag_simple` top-5 -> `rag_hyde` | 200 | 77.0% | 77.5% | +0.5pp | 9/8 | 1.0 | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl`; `logs/eval_rag_hyde_groq-llama70b_20260501_1515_detail.jsonl` |
| SCALR `rag_simple` top-5 -> `rag_snap_hyde_1call` | 200 | 77.0% | 70.5% | -6.5pp | 2/15 | 0.0023499 | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl`; `logs/eval_rag_snap_hyde_1call_groq-llama70b_20260501_1519_detail.jsonl` |
| SCALR `rag_simple` top-5 -> `multi_hyde_diverse` | 200 | 77.0% | 75.5% | -1.5pp | 6/9 | 0.6072388 | `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl`; `logs/eval_multi_hyde_diverse_groq-llama70b_20260501_1531_detail.jsonl` |
| BarExam `rag_simple` top-5 -> `rag_snap_hyde_2call` | 200 | 82.5% | 85.5% | +3.0pp | 19/13 | 0.3770856 | `logs/eval_rag_simple_or-gemma4-26b_20260428_0231_detail.jsonl`; `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260428_1435_detail.jsonl` |
| CaseHOLD repaired `rag_simple` top-1 -> top-5 | 200 | 64.5% | 69.5% | +5.0pp | 16/6 | 0.0524788 | `logs/eval_rag_simple_groq-llama70b_20260501_1432_detail.jsonl`; `logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl` |
| CaseHOLD repaired `rag_simple` top-5 -> top-10 | 200 | 69.5% | 68.0% | -1.5pp | 6/9 | 0.6072388 | `logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl`; `logs/eval_rag_simple_groq-llama70b_20260501_1440_detail.jsonl` |
| CaseHOLD repaired `rag_simple` top-5 -> `rag_hyde` | 200 | 69.5% | 72.0% | +2.5pp | 18/13 | 0.4731297 | `logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl`; `logs/eval_rag_hyde_groq-llama70b_20260501_1449_detail.jsonl` |
| CaseHOLD repaired `rag_snap_hyde_2call` -> `rag_hyde` | 200 | 72.0% | 72.0% | 0.0pp | 12/12 | 1.0 | `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260430_1751_detail.jsonl`; `logs/eval_rag_hyde_groq-llama70b_20260501_1449_detail.jsonl` |
| CaseHOLD repaired `rag_simple` -> `rag_snap_hyde_2call` | 200 | 69.5% | 72.0% | +2.5pp | 16/11 | 0.4420683 | `logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl`; `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260430_1751_detail.jsonl` |
| MuSiQue `rag_simple` -> `golden_passage` | 200 | 27.5% | 56.5% | +29.0pp | 64/6 | 2.44273e-13 | `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`; `logs/eval_golden_passage_groq-llama70b_20260430_1515_detail.jsonl` |
| BarExam 26B `rag_simple` -> `rag_snap_hyde` | 1195 | 78.08% | 81.17% | +3.096pp | 124/87 | 0.0130126 | `logs/eval_rag_simple_cluster-vllm_20260425_2020_detail.jsonl`; `logs/eval_rag_snap_hyde_cluster-vllm_20260425_2226_detail.jsonl` |
| BarExam E4B `rag_simple` -> `rag_snap_hyde` | 1195 | 58.49% | 62.18% | +3.682pp | 172/128 | 0.0129118 | `logs/eval_rag_simple_cluster-vllm_20260426_0020_detail.jsonl`; `logs/eval_rag_snap_hyde_cluster-vllm_20260426_0614_detail.jsonl` |

## BarExam Full-Corpus Method Counts

Direct raw count check over Gemma 4 26B-A4B full-corpus logs:

| Method | Correct / N | Accuracy |
|---|---:|---:|
| `rag_simple` | 933/1195 | 78.08% |
| `rag_hyde` | 943/1195 | 78.91% |
| `golden_passage` | 940/1195 | 78.66% |
| `llm_only` | 953/1195 | 79.75% |
| `snap_only_in_final` | 963/1195 | 80.59% |
| `rag_snap_hyde` | 970/1195 | 81.17% |

## Caveats Kept In The Report

- MuSiQue is excluded from the main class-report benchmark table because it is
  not legal-domain data; it remains an internal multi-hop mechanism check.
- SCALR, HousingQA, and CaseHOLD are N=200 diagnostic legal slices, not
  full-corpus claims.
- Housing state-filter job `58282` is invalid due empty retrieval; the fixed
  run `58799` timed out at 135/200 before reaching a citeable paired row, so no
  state-filter result is cited.
- CaseHOLD repaired two-call improves gold retrieval from 16.0% to 47.0% but
  answer lift is not significant.
- `subagent_rag` is cited only as a negative over-abstention result, not as a
  broad claim against agentic RAG.
- Speculative RAG is related work and future evaluation guidance; this repo
  does not yet log true draft/verifier scores.

## Live Cluster Follow-Ups

- SLURM `58799`: HousingQA state-filtered retrieval at k=5 and k=10 after the
  state-metadata casing fix timed out at 135/200 (`TIMEOUT`, 03:00:15) before
  starting the planned k=10 leg. Do not promote the partial k=5 tail.
- SLURM `58871`: CaseHOLD repaired top-k / HyDE follow-up over the cluster
  `casehold_holdings` collection completed. All pulled rows have 200/200 rows
  and no empty retrieval: k=1 64.5% / 9.0% gold, k=10 68.0% / 19.0% gold,
  `rag_hyde` 72.0% / 52.0% gold.
- SLURM `58912`: LegalBench-SCALR snap ablation failed preflight because
  `legalbench_scalr_holdings` was empty on the cluster checkout. This failure is
  not a method result.
- SLURM `58913`: LegalBench-SCALR embedding repair populated
  `legalbench_scalr_holdings` with 1,733 documents.
- SLURM `58914`: LegalBench-SCALR `rag_hyde`, `rag_snap_hyde_1call`, and
  `multi_hyde_diverse` follow-up completed from
  `scripts/hpc/slurm_scalr_snap_ablation.sh`. The `rag_hyde` leg has been
  pulled and validated: 155/200 (77.5%), no empty retrieval, gold retrieval
  132/200, paired vs top-5 `rag_simple` p=1.0. The `rag_snap_hyde_1call` leg
  has also been pulled and validated: 141/200 (70.5%), no empty retrieval, gold
  retrieval 108/200, paired vs top-5 `rag_simple` p=0.00235. The
  `multi_hyde_diverse` leg is validated at 151/200 (75.5%), no empty retrieval,
  gold retrieval 121/200, paired vs top-5 `rag_simple` p=0.607.

## Housing Prediction Bias Check

HousingQA top-k gains should not be over-described as jurisdiction repair.
The top-10 lift also changes answer bias on a yes/no task:

| Run | Gold Yes/No | Pred Yes | Pred No | Null |
|---|---:|---:|---:|---:|
| top-1 `rag_simple` | 71/129 | 126 | 74 | 0 |
| top-10 `rag_simple` | 71/129 | 100 | 99 | 1 |
| `rag_snap_hyde_2call` | 71/129 | 119 | 81 | 0 |

This supports cautious wording: top-10 retrieval is directional, but the
current evidence does not prove that the lift is caused by jurisdiction repair.

## Figure Metrics

Report figures under `reports/final_class_report/figures/` are generated from
detail logs, not hand-entered plotting tables:

```bash
python reports/final_class_report/build_figures.py
```

The generated `figures/figure_metrics.csv` records per-run `n`, accuracy,
calls/question, tokens/question, empty-retrieval count, and gold-retrieval rate
for the plotted N=200 legal slices. Headline examples used in the report:

| Slice | Accuracy | Calls/q | Tokens/q | Gold retrieved |
|---|---:|---:|---:|---:|
| BarExam top-5 `rag_simple` | 82.5% | 1.0 | 1.6k | 2.5% |
| BarExam `rag_snap_hyde` | 86.0% | 3.0 | 3.9k | 9.5% |
| BarExam `rag_snap_hyde_2call` | 85.5% | 2.0 | 2.8k | 9.0% |
| Housing top-10 `rag_simple` | 58.0% | 1.0 | 4.6k | 5.5% |
| Housing `rag_snap_hyde_2call` | 57.0% | 2.0 | 3.2k | 9.5% |
| SCALR top-5 `rag_simple` | 77.0% | 1.0 | 1.1k | 54.0% |
| SCALR `rag_snap_hyde_2call` | 75.0% | 2.0 | 2.2k | 55.0% |
| CaseHOLD top-5 `rag_simple` | 69.5% | 1.0 | 1.1k | 16.0% |
| CaseHOLD top-10 `rag_simple` | 68.0% | 1.0 | 1.4k | 19.0% |
| CaseHOLD `rag_hyde` | 72.0% | 2.0 | 2.8k | 52.0% |
| CaseHOLD `rag_snap_hyde_2call` | 72.0% | 2.0 | 2.2k | 47.0% |
