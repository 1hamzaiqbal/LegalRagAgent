# 3SCOPE + Raw Pool - 2026-05-28

This report evaluates the 3SCOPE+raw arm: raw query plus three independently generated exemplar-anchored SCOPE passages, dense top-10 retrieval for each representation, unique-document pooling, and MiniLM CE reranking to top-5. No files under `paper/` were edited.

## Hypothesis Verdicts

| Hypothesis | Verdict | Key read |
|---|---|---|
| H-strong-noregress | **supported** | BEIR macro Hit@5 3SCOPE+raw 75.3% vs raw 73.0% (delta 2.3%). |
| H-vs-CSQE strong side | **supported** | BEIR macro 3SCOPE+raw 75.3% vs CSQE 70.0%. |
| H-vs-raw∪SCOPE | **killed** | BEIR macro 3SCOPE+raw 75.3% vs raw∪SCOPE 75.8%. |
| H-net-positive | **supported** | Positive RI on SciFact, FiQA, TREC-COVID, SciDocs. |
| H-weak-help | **killed** | BarExam 3SCOPE+raw 3.4% vs SCOPE 12.0%. |
| H-vs-CSQE weak side | **mixed** | BarExam 3SCOPE+raw 3.4% vs CSQE 2.0%. |

## Regime Table

| Dataset | Arm | N | Hit@5 | Hits | RI vs raw | Help | Hurt | Mean CE delta vs raw | Avg pool size | CE coverage |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | Raw | 300 | 82.0% | 246 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| SciFact | HyDE | 300 | 35.0% | 105 | -0.470 | 12 | 153 | -7.360 | -- | -- |
| SciFact | SCOPE | 300 | 65.7% | 197 | -0.163 | 12 | 61 | -0.909 | -- | -- |
| SciFact | CSQE | 300 | 78.3% | 235 | -0.037 | 8 | 19 | -0.837 | -- | -- |
| SciFact | raw∪SCOPE-pool | 300 | 83.3% | 250 | 0.013 | 7 | 3 | -0.909 | 13.99 | 100.0% |
| SciFact | 3SCOPE+raw | 300 | 84.0% | 252 | 0.020 | 8 | 2 | -1.234 | 16.98 | 100.0% |
| NFCorpus | Raw | 323 | 69.3% | 224 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| NFCorpus | HyDE | 323 | 33.4% | 108 | -0.359 | 6 | 122 | -5.005 | -- | -- |
| NFCorpus | SCOPE | 323 | 65.0% | 210 | -0.043 | 20 | 34 | -0.919 | -- | -- |
| NFCorpus | CSQE | 323 | 61.9% | 200 | -0.074 | 6 | 30 | 0.324 | -- | -- |
| NFCorpus | raw∪SCOPE-pool | 323 | 70.0% | 226 | 0.006 | 12 | 10 | -0.919 | 15.23 | 100.0% |
| NFCorpus | 3SCOPE+raw | 323 | 69.3% | 224 | 0.000 | 8 | 8 | -0.435 | 20.46 | 100.0% |
| FiQA | Raw | 648 | 66.2% | 429 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| FiQA | HyDE | 648 | 32.3% | 209 | -0.340 | 38 | 258 | -4.055 | -- | -- |
| FiQA | SCOPE | 648 | 35.2% | 228 | -0.310 | 25 | 226 | -2.947 | -- | -- |
| FiQA | CSQE | 648 | 63.9% | 414 | -0.023 | 23 | 38 | -1.351 | -- | -- |
| FiQA | raw∪SCOPE-pool | 648 | 71.9% | 466 | 0.057 | 44 | 7 | -2.947 | 17.17 | 100.0% |
| FiQA | 3SCOPE+raw | 648 | 70.7% | 458 | 0.045 | 32 | 3 | -3.485 | 23.58 | 100.0% |
| TREC-COVID | Raw | 50 | 98.0% | 49 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| TREC-COVID | HyDE | 50 | 70.0% | 35 | -0.280 | 1 | 15 | -7.662 | -- | -- |
| TREC-COVID | SCOPE | 50 | 96.0% | 48 | -0.020 | 1 | 2 | -1.824 | -- | -- |
| TREC-COVID | CSQE | 50 | 98.0% | 49 | 0.000 | 0 | 0 | -3.305 | -- | -- |
| TREC-COVID | raw∪SCOPE-pool | 50 | 100.0% | 50 | 0.020 | 1 | 0 | -1.824 | 17.62 | 100.0% |
| TREC-COVID | 3SCOPE+raw | 50 | 100.0% | 50 | 0.020 | 1 | 0 | -2.122 | 26.42 | 100.0% |
| SciDocs | Raw | 989 | 49.3% | 488 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| SciDocs | HyDE | 989 | 25.8% | 255 | -0.236 | 58 | 291 | -3.286 | -- | -- |
| SciDocs | SCOPE | 989 | 47.2% | 467 | -0.021 | 84 | 105 | 1.298 | -- | -- |
| SciDocs | CSQE | 989 | 47.9% | 474 | -0.014 | 56 | 70 | 0.240 | -- | -- |
| SciDocs | raw∪SCOPE-pool | 989 | 53.6% | 530 | 0.042 | 67 | 25 | 1.298 | 14.18 | 100.0% |
| SciDocs | 3SCOPE+raw | 989 | 52.4% | 518 | 0.030 | 56 | 26 | 0.518 | 18.03 | 100.0% |
| BarExamQA | Raw | 1192 | 1.4% | 17 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| BarExamQA | HyDE | 1192 | 11.4% | 136 | 0.100 | 130 | 11 | 4.120 | -- | -- |
| BarExamQA | SCOPE | 1192 | 12.0% | 143 | 0.106 | 137 | 11 | 3.881 | -- | -- |
| BarExamQA | CSQE | 1192 | 2.0% | 24 | 0.006 | 15 | 8 | -0.530 | -- | -- |
| BarExamQA | raw∪SCOPE-pool | 1192 | 3.9% | 47 | 0.025 | 35 | 5 | 3.881 | 19.55 | 100.0% |
| BarExamQA | 3SCOPE+raw | 1192 | 3.4% | 40 | 0.019 | 30 | 7 | 3.518 | 27.62 | 100.0% |
| HousingQA state-filtered | Raw | 6832 | 36.8% | 2512 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| HousingQA state-filtered | HyDE | 6832 | 30.7% | 2096 | -0.061 | 864 | 1280 | 3.372 | -- | -- |
| HousingQA state-filtered | SCOPE | 6832 | 38.0% | 2596 | 0.012 | 1022 | 938 | 2.998 | -- | -- |
| HousingQA state-filtered | CSQE | 6832 | 37.4% | 2552 | 0.006 | 506 | 466 | 4.562 | -- | -- |
| HousingQA state-filtered | raw∪SCOPE-pool | 6832 | 41.1% | 2809 | 0.043 | 556 | 259 | 2.998 | 17.42 | 100.0% |
| HousingQA state-filtered | 3SCOPE+raw | 6832 | 40.1% | 2737 | 0.033 | 456 | 231 | 2.005 | 21.15 | 100.0% |
| BEIR pooled | Raw | 2310 | 62.2% | 1436 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| BEIR pooled | HyDE | 2310 | 30.8% | 712 | -0.313 | 115 | 839 | -4.366 | -- | -- |
| BEIR pooled | SCOPE | 2310 | 49.8% | 1150 | -0.124 | 142 | 428 | -0.557 | -- | -- |
| BEIR pooled | CSQE | 2310 | 59.4% | 1372 | -0.028 | 93 | 157 | -0.411 | -- | -- |
| BEIR pooled | raw∪SCOPE-pool | 2310 | 65.9% | 1522 | 0.037 | 131 | 45 | -0.557 | 15.22 | 100.0% |
| BEIR pooled | 3SCOPE+raw | 2310 | 65.0% | 1502 | 0.029 | 105 | 39 | -1.023 | 19.97 | 100.0% |
| Legal pooled | Raw | 8024 | 31.5% | 2529 | 0.000 | 0 | 0 | 0.000 | -- | -- |
| Legal pooled | HyDE | 8024 | 27.8% | 2232 | -0.037 | 994 | 1291 | 3.483 | -- | -- |
| Legal pooled | SCOPE | 8024 | 34.1% | 2739 | 0.026 | 1159 | 949 | 3.129 | -- | -- |
| Legal pooled | CSQE | 8024 | 32.1% | 2576 | 0.006 | 521 | 474 | 3.806 | -- | -- |
| Legal pooled | raw∪SCOPE-pool | 8024 | 35.6% | 2856 | 0.041 | 591 | 264 | 3.129 | 17.73 | 100.0% |
| Legal pooled | 3SCOPE+raw | 8024 | 34.6% | 2777 | 0.031 | 486 | 238 | 2.230 | 22.11 | 100.0% |

## 3SCOPE Candidate Drift

| Dataset | N | Mean delta s1 | Mean delta s2 | Mean delta s3 | Mean delta avg |
|---|---:|---:|---:|---:|---:|
| SciFact | 300 | -0.915 | -1.376 | -1.410 | -1.234 |
| NFCorpus | 323 | -0.557 | -0.435 | -0.314 | -0.435 |
| FiQA | 648 | -3.481 | -3.530 | -3.444 | -3.485 |
| TREC-COVID | 50 | -2.489 | -2.322 | -1.554 | -2.122 |
| SciDocs | 989 | 0.332 | 0.442 | 0.780 | 0.518 |
| BarExamQA | 1192 | 3.528 | 3.713 | 3.314 | 3.518 |
| HousingQA state-filtered | 6832 | 1.856 | 2.020 | 2.138 | 2.005 |

## Sources

- Row-level points: `docs/generated/3scope_raw_pool_2026-05-28_points.jsonl`
- Exemplar source: `caches/exemplars/beir_orthogonal3_exemplars_2026-05-26.json` plus the built-in BarExam/Housing orthogonal signal bank in `eval/eval_harness.py`.
- SciFact 3SCOPE generation: `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_3scope_raw.jsonl`
- SciFact 3SCOPE+raw pool: `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_3scope_raw_pool_k5.jsonl`
- SciFact raw∪SCOPE pool: `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl`
- NFCorpus 3SCOPE generation: `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_3scope_raw.jsonl`
- NFCorpus 3SCOPE+raw pool: `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_3scope_raw_pool_k5.jsonl`
- NFCorpus raw∪SCOPE pool: `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl`
- FiQA 3SCOPE generation: `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_3scope_raw.jsonl`
- FiQA 3SCOPE+raw pool: `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_3scope_raw_pool_k5.jsonl`
- FiQA raw∪SCOPE pool: `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl`
- TREC-COVID 3SCOPE generation: `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_3scope_raw.jsonl`
- TREC-COVID 3SCOPE+raw pool: `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_3scope_raw_pool_k5.jsonl`
- TREC-COVID raw∪SCOPE pool: `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl`
- SciDocs 3SCOPE generation: `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_3scope_raw.jsonl`
- SciDocs 3SCOPE+raw pool: `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_3scope_raw_pool_k5.jsonl`
- SciDocs raw∪SCOPE pool: `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl`
- BarExamQA 3SCOPE generation: `caches/generation/full/barexam_qfull_seed42_or-gemma4-26b_3scope_raw.jsonl`
- BarExamQA 3SCOPE+raw pool: `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_3scope_raw_pool_k5.jsonl`
- BarExamQA raw∪SCOPE pool: `caches/retrieval/full/barexam_qfull_seed42_or-gemma4-26b_raw_scope_pool_k5.jsonl`
- HousingQA state-filtered 3SCOPE generation: `caches/generation/full/housing_qfull_seed42_statefilter_or-gemma4-26b_3scope_raw.jsonl`
- HousingQA state-filtered 3SCOPE+raw pool: `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_3scope_raw_pool_k5.jsonl`
- HousingQA state-filtered raw∪SCOPE pool: `caches/retrieval/full/housing_qfull_seed42_statefilter_or-gemma4-26b_raw_scope_pool_k5.jsonl`
