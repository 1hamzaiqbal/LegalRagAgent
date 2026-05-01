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
| CaseHOLD repaired `rag_simple` top-1 -> top-5 | 200 | 64.5% | 69.5% | +5.0pp | 16/6 | 0.0524788 | `logs/eval_rag_simple_groq-llama70b_20260501_1432_detail.jsonl`; `logs/eval_rag_simple_groq-llama70b_20260430_1738_detail.jsonl` |
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
  run is SLURM `58799`, so no state-filter result is cited.
- CaseHOLD repaired two-call improves gold retrieval from 16.0% to 47.0% but
  answer lift is not significant.
- `subagent_rag` is cited only as a negative over-abstention result, not as a
  broad claim against agentic RAG.
- Speculative RAG is related work and future evaluation guidance; this repo
  does not yet log true draft/verifier scores.

## Live Cluster Follow-Ups

- SLURM `58799`: HousingQA state-filtered retrieval at k=5 and k=10 after the
  state-metadata casing fix. The k=5 leg has completed in the cluster tail at
  123/200 (61.5%), but it is not promoted until the detail log is pulled and
  paired locally.
- SLURM `58871`: CaseHOLD repaired top-k follow-up over the cluster
  `casehold_holdings` collection. The top-1 `rag_simple` leg landed at
  129/200 (64.5%) with 18/200 gold retrieved and no empty retrieval; k=10 and
  `rag_hyde` remain in progress in the same job.
- SLURM `58885`: LegalBench-SCALR snap ablation, currently queued for
  resources at last check.
