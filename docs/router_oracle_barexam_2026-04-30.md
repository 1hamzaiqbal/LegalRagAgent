# Routing Oracle - BarExam Gemma 4 26B N=200

This is an offline upper-bound analysis. It does not prove an online router can identify the best arm; it estimates whether enough per-question variation exists to justify building one.

## Arm Summary

| Arm | N | Accuracy | Calls/q | Sec/q |
|---|---:|---:|---:|---:|
| rag_top5 | 200 | 82.5% | 1.00 | 14.8 |
| rag_top1 | 200 | 83.0% | 1.00 | 22.3 |
| two_call | 200 | 85.5% | 2.00 | 58.3 |

## Oracle Routing Upper Bounds

Common rows: `200` on key `idx`.

| Policy | Accuracy | Calls/q | Sec/q | Notes |
|---|---:|---:|---:|---|
| Accuracy-first oracle | 93.5% | 1.02 | 12.7 | Choose the cheapest correct arm when any arm is correct; otherwise choose the cheapest arm. |
| Reward oracle | 93.5% | 1.02 | 12.7 | Maximize `correct - 0.02*calls - 0*sec`. |

## Chosen Arm Distribution

| Policy | Arm | Count |
|---|---|---:|
| accuracy_first | rag_top1 | 78 |
| accuracy_first | rag_top5 | 118 |
| accuracy_first | two_call | 4 |
| reward | rag_top1 | 77 |
| reward | rag_top5 | 119 |
| reward | two_call | 4 |

## Source Logs

- `rag_top5`: `logs/eval_rag_simple_or-gemma4-26b_20260428_0231_detail.jsonl`
- `rag_top1`: `logs/eval_rag_simple_or-gemma4-26b_20260428_0138_detail.jsonl`
- `two_call`: `logs/eval_rag_snap_hyde_2call_or-gemma4-26b_20260428_1435_detail.jsonl`
