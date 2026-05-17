# Top-K Prelaunch Probe - 2026-05-14

Purpose: choose a shared evidence depth for the comprehensive API answer
sweeps without spending another full day on top-k tuning. This is a prelaunch
gate, not a final per-model top-k benchmark.

## Decision

Use `RETRIEVAL_K=5` as the default comprehensive answer setting.

Rationale:

- q100 retrieval exposure continues to rise through k=10, but MRR changes very
  little after k=5. Extra passages are often deeper hits, not earlier-rank
  evidence.
- On the default model BarExam downstream q100 slice, k=10 did not beat k=5:
  `rag_simple` fell from 83/100 to 81/100, and `rag_hyde` fell from 87/100 to
  84/100.
- k=10 costs more context, more latency, and more output-format risk. It is
  still useful as an analysis ablation, but it should not hold up the main
  comprehensive grid.

## Fine-Grained Retrieval Curve

Source:
`docs/generated/retrieval_cache_matrix_or-gemma4-26b_q100_k1_to_k10.md`
and `.csv`, compiled from cached top-10 retrieval ids for q100 seed-42 rows
across BarExamQA, HousingQA, CaseHOLD, and LegalBench-SCALR.

Macro over `rag_simple`, `rag_hyde`, and `snap_hyre` caches:

| k | Macro Hit@k | Macro MRR@k |
|---:|---:|---:|
| 1 | 0.1950 | 0.1950 |
| 2 | 0.2433 | 0.2192 |
| 3 | 0.2717 | 0.2286 |
| 4 | 0.2950 | 0.2344 |
| 5 | 0.3175 | 0.2389 |
| 6 | 0.3300 | 0.2410 |
| 7 | 0.3367 | 0.2420 |
| 8 | 0.3508 | 0.2438 |
| 9 | 0.3542 | 0.2441 |
| 10 | 0.3617 | 0.2449 |

Method macro at k=5 versus k=10:

| Method | Hit@5 | Hit@10 | MRR@5 | MRR@10 |
|---|---:|---:|---:|---:|
| `rag_simple` | 0.1775 | 0.2150 | 0.1135 | 0.1184 |
| `rag_hyde` | 0.4050 | 0.4575 | 0.3078 | 0.3152 |
| `snap_hyre` | 0.3700 | 0.4125 | 0.2956 | 0.3010 |

Interpretation: k=10 is better for retrieval exposure, but the marginal MRR
lift from k=5 to k=10 is small: +0.0049 for `rag_simple`, +0.0074 for
`rag_hyde`, and +0.0054 for `snap_hyre`. That makes k=10 useful for retrieval
analysis and recall plots, not clearly worth making the main answer setting.

## Default-Model Downstream Check

Provider: `or-gemma4-26b` (`google/gemma-4-26b-a4b-it`).
Dataset: BarExamQA q100 seed-42 sample.
Run controls: strict retrieval replay caches, `NO_SILENT_FALLBACK=1`,
`LLM_MAX_COMPLETION_TOKENS=2048`, `EVAL_FINAL_FORMAT_RETRY=1`, OpenRouter
provider fallback disabled.

| Mode | k | Accuracy | Detail log | Health |
|---|---:|---:|---|---|
| `rag_simple` | 5 | 83/100 = 83.0% | `logs/merged/barexam_or-gemma4-26b_rag_simple_q100_k5_20260514_detail.jsonl` | clean |
| `rag_simple` | 10 | 81/100 = 81.0% | `logs/merged/barexam_or-gemma4-26b_rag_simple_q100_k10_20260514_detail.jsonl` | clean |
| `rag_hyde` | 5 | 87/100 = 87.0% | `logs/merged/barexam_or-gemma4-26b_rag_hyde_q100_k5_20260514_detail.jsonl` | clean; one same-model format retry |
| `rag_hyde` | 10 | 84/100 = 84.0% | `logs/merged/barexam_or-gemma4-26b_rag_hyde_q100_k10_20260514_detail.jsonl` | clean |

The merged logs passed `scripts/analyze_detail_flags.py`: zero errors, zero
missing predictions, zero parse failures, zero empty retrieval rows, zero
long-answer rows, and no answer-artifact flags.

## Guardrail Change From This Probe

The answer runner now refuses truncation-prone answer runs:

- `scripts/local/run_answer_cell.sh` defaults `LLM_MAX_COMPLETION_TOKENS` to
  2048 for answer cells.
- Explicit environment overrides still win.
- The runner fails closed when `LLM_MAX_COMPLETION_TOKENS` is below
  `EVAL_MIN_COMPLETION_TOKENS` (default 2048). This prevents `.env` from
  silently pulling answer sweeps back to the old 768-token cap.
- Local and HPC answer runners default `EVAL_FINAL_FORMAT_RETRY=1`. The retry
  uses the same provider/model and same evidence, logs `answer_format_retry`,
  and is only for malformed/missing final answer formatting.
- Local and HPC answer runners now require `NO_SILENT_FALLBACK` to be truthy
  and fail before launch if it is disabled.

## Launch Implication

Do not run a broader per-model/per-k downstream search before the main grid.
Use k=5 for the main comprehensive answer table, report retrieval curves at
k=1 through k=10, and reserve k=10 answer runs for a targeted appendix/analysis
only if the comprehensive results make that worthwhile.
