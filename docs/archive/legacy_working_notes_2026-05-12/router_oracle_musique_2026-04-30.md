# Routing Oracle - MuSiQue Llama 70b N=200

This is an offline upper-bound analysis. It does not prove an online router can identify the best arm; it estimates whether enough per-question variation exists to justify building one.

## Arm Summary

| Arm | N | Accuracy | Calls/q | Sec/q |
|---|---:|---:|---:|---:|
| rag | 200 | 27.5% | 1.00 | 0.6 |
| two_call | 200 | 37.0% | 2.00 | 1.2 |
| mhd | 200 | 35.5% | 2.00 | 5.1 |
| iter | 200 | 36.0% | 6.76 | 3.2 |

## Oracle Routing Upper Bounds

Common rows: `200` on key `idx`.

| Policy | Accuracy | Calls/q | Sec/q | Notes |
|---|---:|---:|---:|---|
| Accuracy-first oracle | 57.0% | 1.75 | 1.0 | Choose the cheapest correct arm when any arm is correct; otherwise choose the cheapest arm. |
| Reward oracle | 57.0% | 1.75 | 1.0 | Maximize `correct - 0.02*calls - 0*sec`. |

## Chosen Arm Distribution

| Policy | Arm | Count |
|---|---|---:|
| accuracy_first | iter | 19 |
| accuracy_first | mhd | 10 |
| accuracy_first | rag | 141 |
| accuracy_first | two_call | 30 |
| reward | iter | 19 |
| reward | mhd | 10 |
| reward | rag | 141 |
| reward | two_call | 30 |

## Source Logs

- `rag`: `logs/eval_rag_simple_groq-llama70b_20260427_0952_detail.jsonl`
- `two_call`: `logs/eval_rag_snap_hyde_2call_groq-llama70b_20260428_0040_detail.jsonl`
- `mhd`: `logs/eval_multi_hyde_diverse_groq-llama70b_20260427_1010_detail.jsonl`
- `iter`: `logs/eval_iterative_planning_table_groq-llama70b_20260427_1208_detail.jsonl`
