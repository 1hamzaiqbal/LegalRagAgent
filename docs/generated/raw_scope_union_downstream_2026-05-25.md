# Raw + SCOPE Union Downstream Probe - 2026-05-25

## Task

Post-hoc candidate pooling over existing retrieval caches only, followed by a
new downstream answer call for each selected evidence set:

1. Build a candidate pool from `raw_question` top-10 union canonical
   `snap_hyre` / SCOPE top-10.
2. Select a final top-5 evidence set with one of three arms:
   `ce_rerank`, `rrf`, or `llm_judge`.
3. Answer with `or-gemma4-26b` on the selected top-5 and score exact answer
   accuracy.

This run used the seed-42 q200 slice for BarExamQA and HousingQA with the
Housing state filter enabled. Raw RAG and canonical SCOPE baselines are reused
from signed detail logs rather than rerun.

## Run Setup

- Provider: `or-gemma4-26b`
- OpenRouter route guard: `OPENROUTER_PROVIDER_ONLY=Cloudflare`
- Guards: `NO_SILENT_FALLBACK=1`, `EVAL_FINAL_FORMAT_RETRY=1`
- Completion cap: `LLM_MAX_COMPLETION_TOKENS=2048`
- Retrieval depth: raw top-10 union SCOPE top-10, final selected top-5
- BarExamQA passage hydration: Chroma document lookup by cached passage id
- HousingQA passage hydration: existing retrieval doc caches

No files under `paper/` were edited.

## Results

Accuracy is exact answer accuracy on the q200 slice. Hit@5 and Recall@5 are
computed on the selected final top-5 evidence set.

### BarExamQA q200

| Row | Correct | Accuracy | Hit@5 | Recall@5 | Avg actual calls | Avg logical calls | Avg input toks | Avg output toks | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw question RAG baseline | 162/200 | 81.0% | 2.5% | 2.5% | 1.00 | 1.00 | 1060 | 635 | 0 |
| Canonical SCOPE baseline | 176/200 | 88.0% | 14.0% | 14.0% | 1.00 | 2.00 | 1214 | 646 | 0 |
| Union + CE-rerank | 167/200 | 83.5% | 4.0% | 4.0% | 1.00 | 1.00 | 1125 | 632 | 0 |
| Union + RRF | 174/200 | 87.0% | 5.5% | 5.5% | 1.00 | 1.00 | 1142 | 644 | 0 |
| Union + LLM-judge | 177/200 | 88.5% | 11.5% | 11.5% | 2.00 | 2.00 | 4616 | 669 | 0 |

BarExamQA takeaways:

- LLM-judge is the only union arm that beats both baselines, but only by one
  row over canonical SCOPE: 177/200 vs 176/200.
- The answer gain is not backed by better gold exposure than SCOPE: LLM-judge
  Hit@5 is 11.5%, below canonical SCOPE's 14.0%.
- RRF improves over raw RAG by 6.0pp but remains below SCOPE by 1.0pp.
- CE-rerank is worse than SCOPE and only modestly above raw RAG.

### HousingQA State-Filtered q200

| Row | Correct | Accuracy | Hit@5 | Recall@5 | Avg actual calls | Avg logical calls | Avg input toks | Avg output toks | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw question RAG baseline | 124/200 | 62.0% | 40.0% | 25.7% | 1.00 | 1.00 | 2452 | 476 | 0 |
| Canonical SCOPE baseline | 118/200 | 59.0% | 37.5% | 22.8% | 1.00 | 2.00 | 2346 | 479 | 0 |
| Union + CE-rerank | 130/200 | 65.0% | 38.0% | 23.7% | 1.00 | 1.00 | 2787 | 479 | 0 |
| Union + RRF | 121/200 | 60.5% | 45.5% | 29.2% | 1.00 | 1.00 | 2534 | 466 | 0 |
| Union + LLM-judge | 126/200 | 63.0% | 58.0% | 39.5% | 2.01 | 2.01 | 6238 | 467 | 0 |

HousingQA takeaways:

- CE-rerank is the best answer arm on this slice: 130/200, +6 rows over raw
  RAG and +12 rows over canonical SCOPE.
- LLM-judge is the best retrieval-exposure arm: Hit@5 58.0% and Recall@5
  39.5%, far above both baselines, but it converts that exposure into only
  +2 answer rows over raw RAG.
- RRF improves retrieval exposure over both baselines but does not beat raw RAG
  on answer accuracy.
- The strongest Housing signal is that selection/order can matter even when
  Hit@5 does not fully explain answer movement: CE-rerank beats raw RAG on
  accuracy while slightly trailing raw RAG on Hit@5 and Recall@5.

## Health Notes

- Final q200 result rows are complete: 200 rows per dataset x arm.
- Final result rows have zero errors, zero missing predictions, and zero final
  answer format retries.
- Two HousingQA LLM-judge rows initially copied passage ids instead of bracketed
  candidate numbers. They were rerun with the same selection task but with
  passage ids omitted from the judge display, preserving the bracket-number
  selection contract. Both reruns completed cleanly.
- One BarExamQA LLM-judge selection call hit an upstream idle timeout; the
  provider wrapper retried successfully and the row completed cleanly.

## Scale Gate

Strictly applying the requested q200 gate, three arms beat both baselines:

- BarExamQA: Union + LLM-judge, 88.5% vs SCOPE 88.0% and raw 81.0%.
- HousingQA: Union + CE-rerank, 65.0% vs raw 62.0% and SCOPE 59.0%.
- HousingQA: Union + LLM-judge, 63.0% vs raw 62.0% and SCOPE 59.0%.

Recommended next scaling order:

1. Scale HousingQA CE-rerank first. It has the clearest answer gain and costs
   one answer call per row with no judge call.
2. Treat HousingQA LLM-judge as a retrieval-exposure follow-up, not the first
   answer-accuracy scale target. Its Hit@5 gain is large, but answer conversion
   is modest and the arm costs about two model calls per row.
3. Do not prioritize full BarExamQA LLM-judge scaling from this result alone.
   The pass over SCOPE is one row on q200 and its Hit@5 is below SCOPE.

This document stops at the requested q200 gate and does not promote any q200
result as a full-corpus claim.

## Source Files

Retrieval caches:

- `caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`

Document text caches / hydration:

- `caches/retrieval_doc/full/housing_qfull_seed42_statefilter_raw_question_k10_doc_cache.jsonl`
- `caches/retrieval_doc/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10_doc_cache.jsonl`
- BarExamQA text was hydrated by cached Chroma document id lookup.

Baseline detail logs:

- `logs/eval_rag_simple_or-gemma4-26b_20260516_164128_barexam_local-snap-hyre-or-gemma4-26b-barexam-rag_simple-nfull-k5_detail.jsonl`
- `logs/eval_snap_hyre_or-gemma4-26b_20260517_091147_barexam_local-snap-hyre-or-gemma4-26b-barexam-snap_hyre-nfull-k5_detail.jsonl`
- `logs/merged/housing_or-gemma4-26b_rag_simple_statefilter_full_20260521_185315_detail.jsonl`
- `logs/merged/housing_or-gemma4-26b_snap_hyre_statefilter_full_20260523_113019_detail.jsonl`

Scratch row outputs used to compute this document:

- `/tmp/raw_scope_union_downstream_2026-05-25b_rows.jsonl`
- `/tmp/raw_scope_union_downstream_2026-05-25b_housing_rows.jsonl`
