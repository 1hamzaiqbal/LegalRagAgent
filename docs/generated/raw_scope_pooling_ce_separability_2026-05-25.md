# Raw + SCOPE Pooling and CE Separability - 2026-05-25

Read-only cache analysis over `caches/retrieval/full/`. No evals or generation
runs were launched. `Snap` below means the model-specific `snap_hyre` retrieval
cache. `Union` means `raw_question` top-k OR `snap_hyre` top-k for the same
question label. Table values are percentages; gains are percentage points over
the better single method in the same row.

## Part A - Pooling Union Recall

Rows are questions present in both the raw-question and Snap-HyRE caches with at
least one gold id.

### k = 5

| Dataset | Provider | Rows | Raw Hit@5 | Snap Hit@5 | Union Hit@5 | Hit gain | Raw Recall@5 | Snap Recall@5 | Union Recall@5 | Recall gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | `groq-llama8b` | 1195 | 1.42 | 9.54 | 10.63 | 1.09 | 1.42 | 9.54 | 10.63 | 1.09 |
| BarExamQA | `or-gemma4-26b` | 1195 | 1.42 | 12.05 | 12.97 | 0.92 | 1.42 | 12.05 | 12.97 | 0.92 |
| BarExamQA | `groq-llama70b` | 1195 | 1.42 | 11.05 | 12.13 | 1.09 | 1.42 | 11.05 | 12.13 | 1.09 |
| HousingQA statefilter | `groq-llama8b` | 6853 | 36.95 | 29.56 | 47.32 | 10.38 | 24.13 | 18.86 | 32.84 | 8.71 |
| HousingQA statefilter | `or-gemma4-26b` | 6853 | 36.95 | 38.07 | 51.88 | 13.80 | 24.13 | 25.05 | 36.55 | 11.50 |
| HousingQA statefilter | `groq-llama70b` | 6853 | 36.95 | 23.11 | 44.35 | 7.40 | 24.13 | 14.73 | 30.02 | 5.89 |

### k = 10

| Dataset | Provider | Rows | Raw Hit@10 | Snap Hit@10 | Union Hit@10 | Hit gain | Raw Recall@10 | Snap Recall@10 | Union Recall@10 | Recall gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | `groq-llama8b` | 1195 | 2.18 | 14.81 | 16.23 | 1.42 | 2.18 | 14.81 | 16.23 | 1.42 |
| BarExamQA | `or-gemma4-26b` | 1195 | 2.18 | 18.66 | 19.58 | 0.92 | 2.18 | 18.66 | 19.58 | 0.92 |
| BarExamQA | `groq-llama70b` | 1195 | 2.18 | 18.49 | 19.58 | 1.09 | 2.18 | 18.49 | 19.58 | 1.09 |
| HousingQA statefilter | `groq-llama8b` | 6853 | 48.11 | 36.96 | 57.65 | 9.54 | 33.70 | 24.96 | 42.75 | 9.05 |
| HousingQA statefilter | `or-gemma4-26b` | 6853 | 48.11 | 46.96 | 61.32 | 13.21 | 33.70 | 32.86 | 46.32 | 12.62 |
| HousingQA statefilter | `groq-llama70b` | 6853 | 48.11 | 31.30 | 55.38 | 7.27 | 33.70 | 20.79 | 40.22 | 6.51 |

## Part B - Cross-Encoder Gold-vs-Non-Gold Separability

Computed from the cache `scores` field over top-10 retrieved passages. Gold CE
statistics use retrieved passages whose ids are in `gold_ids`; non-gold CE
statistics use the other retrieved passages. Rank statistics are over rows
where at least one gold passage appears in top 10, using the best-ranked gold
passage after sorting by CE score descending.

| Dataset | Query type | Provider | Rows | Gold rows@10 | Gold passages | Gold CE mean | Gold CE median | Non-gold CE mean | Non-gold CE median | Best-gold rank mean | Best-gold rank median | Gold rank 1 | Gold rank >5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | `raw_question` | `model-independent` | 1195 | 26 | 26 | -1.200 | -1.173 | -2.141 | -2.206 | 4.731 | 5.000 | 15.4 | 34.6 |
| BarExamQA | `rag_hyde` | `groq-llama8b` | 1195 | 161 | 161 | 4.204 | 4.288 | 3.780 | 4.002 | 4.807 | 5.000 | 20.5 | 38.5 |
| BarExamQA | `snap_hyre` | `groq-llama8b` | 1195 | 177 | 177 | 5.297 | 5.602 | 5.040 | 5.311 | 4.638 | 4.000 | 15.3 | 35.6 |
| BarExamQA | `rag_hyde` | `or-gemma4-26b` | 1195 | 228 | 228 | 4.347 | 4.549 | 4.090 | 4.411 | 4.886 | 4.000 | 13.2 | 40.4 |
| BarExamQA | `snap_hyre` | `or-gemma4-26b` | 1195 | 223 | 223 | 4.811 | 5.094 | 4.580 | 4.957 | 4.543 | 4.000 | 17.0 | 35.4 |
| BarExamQA | `rag_hyde` | `groq-llama70b` | 1195 | 210 | 210 | 3.620 | 3.810 | 3.395 | 3.562 | 4.895 | 5.000 | 14.8 | 40.5 |
| BarExamQA | `snap_hyre` | `groq-llama70b` | 1195 | 221 | 221 | 4.510 | 4.728 | 4.336 | 4.641 | 4.810 | 5.000 | 16.3 | 40.3 |
| HousingQA statefilter | `raw_question` | `model-independent` | 6853 | 3297 | 4597 | -1.220 | -1.048 | -3.149 | -3.015 | 3.487 | 3.000 | 32.2 | 23.2 |
| HousingQA statefilter | `rag_hyde` | `groq-llama8b` | 6853 | 2364 | 2980 | 2.358 | 2.504 | 1.485 | 1.584 | 3.443 | 3.000 | 32.4 | 21.9 |
| HousingQA statefilter | `snap_hyre` | `groq-llama8b` | 6853 | 2533 | 3207 | 2.357 | 2.452 | 0.860 | 0.978 | 3.334 | 2.000 | 35.0 | 20.0 |
| HousingQA statefilter | `rag_hyde` | `or-gemma4-26b` | 6853 | 2750 | 3323 | 1.711 | 1.650 | 0.637 | 0.616 | 3.523 | 3.000 | 33.7 | 23.7 |
| HousingQA statefilter | `snap_hyre` | `or-gemma4-26b` | 6853 | 3218 | 4225 | 1.832 | 1.825 | 0.285 | 0.193 | 3.196 | 2.000 | 35.4 | 18.9 |
| HousingQA statefilter | `rag_hyde` | `groq-llama70b` | 6853 | 2917 | 3657 | 2.909 | 2.946 | 1.830 | 1.917 | 3.173 | 2.000 | 36.9 | 17.9 |
| HousingQA statefilter | `snap_hyre` | `groq-llama70b` | 6853 | 2145 | 2516 | 1.950 | 2.054 | 1.129 | 1.239 | 3.726 | 3.000 | 29.7 | 26.2 |

## Reading

The Mac-side pattern holds at full N: raw+SCOPE pooling gives only a small
BarExamQA gain, but gives a large HousingQA state-filtered recall gain across
all three models. The HousingQA union gain is strongest for Gemma 26B, where
Hit@5 rises from the best single-method value of 38.07% to 51.88%, and
Recall@5 rises from 25.05% to 36.55%. CE scores do separate gold from non-gold
when gold appears in the candidate list, especially on HousingQA, but CE does
not reliably place gold first: BarExamQA's best gold passage has median rank
4-5 and is below rank 5 in roughly 35-40% of gold-hit rows. The main opportunity
is therefore not just reranking a fixed Snap-HyRE list; HousingQA benefits from
pooling complementary raw and SCOPE candidates before selection, while BarExamQA
still appears candidate-generation limited.

## Source Cache Files

- `caches/retrieval/full/barexam_qfull_seed42_raw_question_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama8b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/barexam_qfull_seed42_groq-llama70b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_raw_question_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama8b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/housing_qfull_seed42_statefilter_groq-llama70b_snap_hyre_k10.jsonl`
