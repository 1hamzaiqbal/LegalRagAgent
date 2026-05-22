# HousingQA State-Filtered Efficiency Snapshot - 2026-05-21 06:15 CDT

Generated offline from eval detail JSONL logs. Active/partial rows are operational diagnostics only; paper-facing result claims still need `docs/signoff_log.md` signoff.

Interpretation notes:

- `Actual calls/q` is the live answer-pass calls recorded in the detail log.
- `Logical calls/q` is the conceptual method footprint recorded by the harness; cached HyDE/Snap-HyRE answer replays usually show one actual answer call but two logical calls.
- Latency mixes provider latency and local harness overhead, so compare only like-for-like runs.
- `Correct / 1M tok.` uses answer-pass input plus output tokens. Generation-cache token cost is not included unless the generation run itself is passed to this script.
- Health `near-cap` counts rows whose logged answer-pass output tokens meet the configured near-cap threshold, not merely verbose rows.

| Label | Status | Rows | Acc | Hit@5 | Tok/q | In/q | Out/q | Actual calls/q | Logical calls/q | Lat avg/p95 | Correct / 1M tok. | Tokens / correct | Health |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8B rag_simple | signed | 6853 | 62.3% | 36.9% | 2953 | 2630 | 324 | 1.02 | 1.02 | 1.07/2.00 | 210.9 | 4741 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0% |
| 8B rag_hyde | signed | 6853 | 59.1% | 27.0% | 2781 | 2449 | 332 | 1.02 | 2.00 | 0.99/1.80 | 212.4 | 4709 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0%, hyre=100.0% |
| 8B snap_hyre | signed | 6853 | 59.0% | 29.6% | 2698 | 2377 | 321 | 1.02 | 2.00 | 0.99/1.80 | 218.7 | 4573 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0%, hyre=100.0% |
| 70B rag_simple | signed | 6853 | 62.1% | 36.9% | 2871 | 2593 | 278 | 1.00 | 1.00 | 1.32/2.10 | 216.4 | 4621 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0% |
| 70B rag_hyde | signed | 6853 | 62.2% | 34.9% | 2783 | 2513 | 270 | 1.00 | 2.00 | 1.35/2.20 | 223.5 | 4474 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0%, hyre=100.0% |
| 70B snap_hyre | signed | 6853 | 59.6% | 23.1% | 2437 | 2156 | 281 | 1.00 | 2.00 | 1.41/2.20 | 244.7 | 4086 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0%, hyre=100.0% |
| Gemma rag_simple tail | active_partial | 3204 | 66.7% | 34.7% | 3322 | 2839 | 483 | 1.00 | 1.00 | 7.62/13.48 | 200.8 | 4981 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0% |
| Gemma q500 snap_hyre | signed_probe | 500 | 63.0% | 38.2% | 2899 | 2424 | 476 | 1.00 | 2.00 | 7.63/13.11 | 217.3 | 4602 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0%, hyre=100.0% |
| Gemma q500 snap_hyre_exemplar | signed_probe | 500 | 62.8% | 41.2% | 2989 | 2523 | 466 | 1.00 | 2.00 | 7.92/13.00 | 210.1 | 4760 | final=100.0%, state=100.0%, ret=100.0%, doc=100.0%, hyre=100.0% |

## Provenance

| Label | Detail log |
|---|---|
| 8B rag_simple | `logs/eval_rag_simple_groq-llama8b_20260520_132953_housing_local-snap-hyre-groq-llama8b-housing-rag_simple-nfull-k5_detail.jsonl` |
| 8B rag_hyde | `logs/eval_rag_hyde_groq-llama8b_20260520_233346_housing_local-snap-hyre-groq-llama8b-housing-rag_hyde-nfull-k5_detail.jsonl` |
| 8B snap_hyre | `logs/eval_snap_hyre_groq-llama8b_20260521_041736_housing_local-snap-hyre-groq-llama8b-housing-snap_hyre-nfull-k5_detail.jsonl` |
| 70B rag_simple | `logs/eval_rag_simple_groq-llama70b_20260520_230339_housing_local-snap-hyre-groq-llama70b-housing-rag_simple-nfull-k5_detail.jsonl` |
| 70B rag_hyde | `logs/eval_rag_hyde_groq-llama70b_20260521_013539_housing_local-snap-hyre-groq-llama70b-housing-rag_hyde-nfull-k5_detail.jsonl` |
| 70B snap_hyre | `logs/merged/housing_groq-llama70b_snap_hyre_statefilter_full_20260520_detail.jsonl` |
| Gemma rag_simple tail | `logs/eval_rag_simple_or-gemma4-26b_20260520_233010_housing_local-snap-hyre-or-gemma4-26b-housing-rag_simple-nfull-k5_detail.jsonl` |
| Gemma q500 snap_hyre | `logs/eval_snap_hyre_or-gemma4-26b_20260521_012744_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre-n500-k5_detail.jsonl` |
| Gemma q500 snap_hyre_exemplar | `logs/eval_snap_hyre_exemplar_or-gemma4-26b_20260521_023301_housing_local-snap-hyre-or-gemma4-26b-housing-snap_hyre_exemplar-n500-k5_detail.jsonl` |
