# Credibility C++ E5 Addendum - 2026-05-29

No `paper/` files were edited.

## Verdict

- Three-retriever status: **closed**.
- E5 completed datasets: `7`; SCOPE mean Spearman `0.387` over `7` dataset correlations.
- Original gte+CE SCOPE mean Spearman `0.342`; BM25 SCOPE mean Spearman `0.354`.
- Verdict criterion: E5 mean SCOPE Spearman >= 0.3 across completed datasets closes the three-retriever mechanism claim for those datasets.
- E5 embedding inputs are capped at `4096` characters, matching the existing BEIR embedding pipeline cap.

## E5 Retrieval Summary

| Dataset | Arm | N | Hit@5 | Hit@10 | Delta vs raw | Help | Hurt | RI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BarExamQA | raw | 1195 | 0.5% | 1.0% | 0.0% | 0 | 0 | 0.000 |
| BarExamQA | hyde | 1195 | 11.2% | 15.8% | 10.7% | 133 | 5 | 0.107 |
| BarExamQA | scope | 1195 | 11.7% | 17.2% | 11.2% | 136 | 2 | 0.112 |
| FiQA | raw | 648 | 61.9% | 68.4% | 0.0% | 0 | 0 | 0.000 |
| FiQA | hyde | 648 | 30.4% | 36.4% | -31.5% | 24 | 228 | -0.315 |
| FiQA | scope | 648 | 31.9% | 42.4% | -29.9% | 23 | 217 | -0.299 |
| NFCorpus | raw | 323 | 69.0% | 74.6% | 0.0% | 0 | 0 | 0.000 |
| NFCorpus | hyde | 323 | 44.0% | 52.9% | -25.1% | 12 | 93 | -0.251 |
| NFCorpus | scope | 323 | 66.3% | 72.8% | -2.8% | 25 | 34 | -0.028 |
| SciDocs | raw | 1000 | 50.5% | 61.0% | 0.0% | 0 | 0 | 0.000 |
| SciDocs | hyde | 1000 | 24.1% | 33.0% | -26.4% | 48 | 312 | -0.264 |
| SciDocs | scope | 1000 | 45.8% | 57.4% | -4.7% | 71 | 118 | -0.047 |
| SciFact | raw | 300 | 79.3% | 85.7% | 0.0% | 0 | 0 | 0.000 |
| SciFact | hyde | 300 | 54.7% | 59.3% | -24.7% | 12 | 86 | -0.247 |
| SciFact | scope | 300 | 78.0% | 84.3% | -1.3% | 24 | 28 | -0.013 |
| TREC-COVID | raw | 50 | 98.0% | 100.0% | 0.0% | 0 | 0 | 0.000 |
| TREC-COVID | hyde | 50 | 64.0% | 68.0% | -34.0% | 0 | 17 | -0.340 |
| TREC-COVID | scope | 50 | 86.0% | 94.0% | -12.0% | 1 | 7 | -0.120 |
| HousingQA state-filtered | raw | 6853 | 39.5% | 50.7% | 0.0% | 0 | 0 | 0.000 |
| HousingQA state-filtered | hyde | 6853 | 50.5% | 60.3% | 11.0% | 1270 | 518 | 0.110 |
| HousingQA state-filtered | scope | 6853 | 47.0% | 56.7% | 7.5% | 1120 | 609 | 0.075 |

## Three-Retriever Mechanism Comparison

| Retriever | Dataset | Arm | N | Spearman | Kendall | Pearson | Mean gold-affinity delta |
|---|---|---|---:|---:|---:|---:|---:|
| bm25_tantivy_full | BarExamQA | scope | 1195 | 0.305 | 0.249 | 0.354 | 8.840 |
| bm25_tantivy_full | FiQA | scope | 648 | 0.394 | 0.319 | 0.439 | 10.071 |
| bm25_tantivy_full | NFCorpus | scope | 323 | 0.380 | 0.301 | 0.345 | 17.995 |
| bm25_tantivy_full | SciDocs | scope | 1000 | 0.349 | 0.281 | 0.354 | 14.828 |
| bm25_tantivy_full | SciFact | scope | 300 | 0.416 | 0.338 | 0.382 | 16.096 |
| bm25_tantivy_full | TREC-COVID | scope | 50 | 0.195 | 0.159 | 0.165 | 27.132 |
| bm25_tantivy_full | HousingQA state-filtered | scope | 6853 | 0.440 | 0.354 | 0.432 | 10.996 |
| e5_large_v2_full | BarExamQA | scope | 1195 | 0.344 | 0.281 | 0.374 | 0.044 |
| e5_large_v2_full | FiQA | scope | 648 | 0.513 | 0.419 | 0.529 | -0.049 |
| e5_large_v2_full | NFCorpus | scope | 323 | 0.361 | 0.294 | 0.395 | -0.002 |
| e5_large_v2_full | SciDocs | scope | 1000 | 0.425 | 0.345 | 0.438 | -0.013 |
| e5_large_v2_full | SciFact | scope | 300 | 0.379 | 0.308 | 0.428 | -0.004 |
| e5_large_v2_full | TREC-COVID | scope | 50 | 0.234 | 0.193 | 0.343 | -0.006 |
| e5_large_v2_full | HousingQA state-filtered | scope | 6853 | 0.454 | 0.368 | 0.467 | 0.001 |
| gte_ce_original | BarExamQA | scope | 1195 | 0.354 | 0.287 | 0.342 | 3.885 |
| gte_ce_original | FiQA | scope | 648 | 0.505 | 0.411 | 0.512 | -2.947 |
| gte_ce_original | NFCorpus | scope | 323 | 0.296 | 0.241 | 0.305 | -0.919 |
| gte_ce_original | SciDocs | scope | 1000 | 0.299 | 0.240 | 0.311 | 1.302 |
| gte_ce_original | SciFact | scope | 300 | 0.329 | 0.270 | 0.409 | -0.909 |
| gte_ce_original | TREC-COVID | scope | 50 | 0.108 | 0.088 | 0.137 | -1.824 |
| gte_ce_original | HousingQA state-filtered | scope | 6853 | 0.504 | 0.410 | 0.530 | 2.990 |

## Run Notes

- BarExamQA E5: persisted E5 shards docs=856835; states=1; shards=86; encoded_this_run=856835; skipped_existing=0; embed_sec=26689.7; index_dir=caches/e5/legal/barexam/intfloat_e5-large-v2_cap4096; query_vectors=3585; search_sec=67.5; elapsed_sec=27006.8; state_searches=all:856835docs/3585queries/67.5s
- HousingQA state-filtered E5: persisted E5 shards docs=1837403; states=51; shards=210; encoded_this_run=1837403; skipped_existing=0; embed_sec=59956.4; index_dir=caches/e5/legal/housing/intfloat_e5-large-v2_cap4096; query_vectors=20559; search_sec=28.2; elapsed_sec=60747.5; state_searches=alabama:43768docs/441queries/1.1s; alaska:17706docs/276queries/0.2s; arizona:15259docs/207queries/0.1s; arkansas:33857docs/405queries/0.5s; california:155974docs/543queries/3.0s; colorado:30056docs/576queries/0.6s; connecticut:25832docs/546queries/0.5s; delaware:20294docs/522queries/0.4s; plus 40 more states
- E5 row-level points: `docs/generated/credibility_C_e5_addendum_2026-05-29_points.jsonl`
- Base C++ row-level points: `docs/generated/credibility_C_three_retrievers_full_2026-05-29_points.jsonl`
