# Evidence Matrix - 2026-04-30

Generated from landed detail logs only. This matrix is a triage artifact: use it to test whether the bottleneck-taxonomy story survives direct log recomputation before promoting any row to paper-grade prose.

## Run Matrix

| Label | Dataset | Provider | Mode | N | Acc | Gold hit | Retrieval rows | Empty retrieval | Calls/q | Sec/q | Evidence docs/q | In tok/q | Out tok/q |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| musique_rag | musique | groq-llama70b | rag_simple | 200 | 55/200 (27.5%) | 84.0% | 100.0% | 0.0% | 1.00 | 0.6 | 5.0 | 833 | 84 |
| musique_2call | musique | groq-llama70b | rag_snap_hyde_2call | 200 | 74/200 (37.0%) | 86.5% | 100.0% | 0.0% | 2.00 | 1.2 | 5.0 | 1135 | 249 |
| musique_mhd | musique | groq-llama70b | multi_hyde_diverse | 200 | 71/200 (35.5%) | 84.0% | 100.0% | 0.0% | 2.00 | 5.1 | 5.0 | 1033 | 417 |
| musique_iter | musique | groq-llama70b | iterative_planning_table | 200 | 72/200 (36.0%) | 92.0% | 100.0% | 0.0% | 6.76 | 3.2 | 6.0 | 2507 | 422 |
| musique_rag_top1 | musique | groq-llama70b | rag_simple | 200 | 26/200 (13.0%) | 47.0% | 100.0% | 0.0% | 1.00 | 0.4 | 1.0 | 256 | 74 |
| barexam_rag_top5 | barexam | or-gemma4-26b | rag_simple | 200 | 165/200 (82.5%) | 2.5% | 100.0% | 0.0% | 1.00 | 14.8 | 5.0 | 989 | 620 |
| barexam_rag_top1 | barexam | or-gemma4-26b | rag_simple | 200 | 166/200 (83.0%) | 0.5% | 100.0% | 0.0% | 1.00 | 22.3 | 1.0 | 469 | 633 |
| barexam_2call | barexam | or-gemma4-26b | rag_snap_hyde_2call | 200 | 171/200 (85.5%) | 9.0% | 100.0% | 0.0% | 2.00 | 58.3 | 5.0 | 1625 | 1221 |
| casehold_rag | casehold | groq-llama70b | rag_simple | 200 | 144/200 (72.0%) | 0.0% | 100.0% | 0.0% | 1.00 | 1.5 | 5.0 | 672 | 438 |
| casehold_rag_top1 | casehold | groq-llama70b | rag_simple | 200 | 141/200 (70.5%) | 0.0% | 100.0% | 0.0% | 1.00 | 1.6 | 1.0 | 521 | 436 |
| casehold_2call | casehold | groq-llama70b | rag_snap_hyde_2call | 200 | 139/200 (69.5%) | 0.0% | 100.0% | 0.0% | 2.00 | 2.6 | 5.0 | 1232 | 700 |
| scalr_rag | legalbench_scalr | groq-llama70b | rag_simple | 200 | 154/200 (77.0%) | 54.0% | 100.0% | 0.0% | 1.00 | 1.9 | 5.0 | 723 | 423 |
| scalr_rag_top1 | legalbench_scalr | groq-llama70b | rag_simple | 200 | 119/200 (59.5%) | 32.5% | 100.0% | 0.0% | 1.00 | 1.9 | 1.0 | 518 | 430 |
| scalr_2call | legalbench_scalr | groq-llama70b | rag_snap_hyde_2call | 200 | 150/200 (75.0%) | 55.0% | 100.0% | 0.0% | 2.00 | 3.5 | 5.0 | 1337 | 848 |

## Paired Deltas

| Pair | Baseline | Treatment | Key | N | Baseline acc | Treatment acc | Delta | b/c | McNemar p | 95% bootstrap CI |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MuSiQue_2call_vs_rag | musique_rag | musique_2call | idx | 200 | 27.5% | 37.0% | +9.5pp | 33/14 | 0.007943 | [+3.0, +16.0] pp |
| MuSiQue_mhd_vs_rag | musique_rag | musique_mhd | idx | 200 | 27.5% | 35.5% | +8.0pp | 29/13 | 0.01952 | [+1.5, +14.5] pp |
| MuSiQue_iter_vs_rag | musique_rag | musique_iter | idx | 200 | 27.5% | 36.0% | +8.5pp | 43/26 | 0.05329 | [+0.5, +16.5] pp |
| MuSiQue_top1_ablation | musique_rag | musique_rag_top1 | idx | 200 | 27.5% | 13.0% | -14.5pp | 3/32 | 4.177e-07 | [-20.0, -9.5] pp |
| BarExam_2call_vs_rag | barexam_rag_top5 | barexam_2call | idx | 200 | 82.5% | 85.5% | +3.0pp | 19/13 | 0.3771 | [-2.5, +8.5] pp |
| BarExam_top1_ablation | barexam_rag_top5 | barexam_rag_top1 | idx | 200 | 82.5% | 83.0% | +0.5pp | 18/17 | 1 | [-5.5, +6.5] pp |
| CaseHOLD_top1_ablation | casehold_rag | casehold_rag_top1 | idx | 200 | 72.0% | 70.5% | -1.5pp | 10/13 | 0.6776 | [-6.0, +3.0] pp |
| CaseHOLD_2call_vs_rag | casehold_rag | casehold_2call | idx | 200 | 72.0% | 69.5% | -2.5pp | 14/19 | 0.4869 | [-8.0, +3.0] pp |
| SCALR_top1_ablation | scalr_rag | scalr_rag_top1 | idx | 200 | 77.0% | 59.5% | -17.5pp | 3/38 | 1.048e-08 | [-23.5, -12.0] pp |
| SCALR_2call_vs_rag | scalr_rag | scalr_2call | idx | 200 | 77.0% | 75.0% | -2.0pp | 8/12 | 0.5034 | [-6.5, +2.5] pp |

## Parse And Route Health

- `musique_rag`: parse `-`; route `-`
- `musique_2call`: parse `snap_hyde_2call_parse_ok=197/200`; route `routed_to: snap_hyde_2call_parse_failed_fallback_to_question=3`
- `musique_mhd`: parse `-`; route `routed_to: single_hyde_fallback_only_2_passages=1`
- `musique_iter`: parse `-`; route `-`
- `musique_rag_top1`: parse `-`; route `-`
- `barexam_rag_top5`: parse `-`; route `-`
- `barexam_rag_top1`: parse `-`; route `-`
- `barexam_2call`: parse `snap_hyde_2call_parse_ok=200/200`; route `-`
- `casehold_rag`: parse `-`; route `-`
- `casehold_rag_top1`: parse `-`; route `-`
- `casehold_2call`: parse `snap_hyde_2call_parse_ok=198/200`; route `routed_to: snap_hyde_2call_parse_failed_fallback_to_question=2`
- `scalr_rag`: parse `-`; route `-`
- `scalr_rag_top1`: parse `-`; route `-`
- `scalr_2call`: parse `snap_hyde_2call_parse_ok=200/200`; route `-`

## Source Logs

- `musique_rag`: `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`
- `musique_2call`: `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0040_detail.jsonl`
- `musique_mhd`: `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl`
- `musique_iter`: `logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl`
- `musique_rag_top1`: `logs/eval_rag_simple_groq-llama70b_20260428_0011_detail.jsonl`
- `barexam_rag_top5`: `logs/eval_rag_simple_or-gemma4-26b_20260428_0231_detail.jsonl`
- `barexam_rag_top1`: `logs/eval_rag_simple_or-gemma4-26b_20260428_0138_detail.jsonl`
- `barexam_2call`: `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260428_1435_detail.jsonl`
- `casehold_rag`: `logs/eval_rag_simple_groq-llama70b_20260428_0259_detail.jsonl`
- `casehold_rag_top1`: `logs/eval_rag_simple_groq-llama70b_20260429_2318_detail.jsonl`
- `casehold_2call`: `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0309_detail.jsonl`
- `scalr_rag`: `logs/eval_rag_simple_groq-llama70b_20260428_1508_detail.jsonl`
- `scalr_rag_top1`: `logs/eval_rag_simple_groq-llama70b_20260429_2159_detail.jsonl`
- `scalr_2call`: `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_1520_detail.jsonl`
