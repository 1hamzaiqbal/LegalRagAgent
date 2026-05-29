# Credibility C++ - Three-Retriever Full-Corpus Battery

No `paper/` files were edited.

## Verdict

- Status: **mechanism travels to BM25; third dense pending**.
- Original gte+CE mean SCOPE Spearman: `0.342` over `7` dataset correlations.
- Full-corpus BM25 mean SCOPE Spearman: `0.354` over `7` dataset correlations.
- All finite SCOPE correlation mean across completed retrievers: `0.348`.
- The requested third dense retriever is not silently substituted; if its full retrieval rows are absent below, the report is provisional for the three-retriever criterion.

## Retrieval Summary

| Retriever | Dataset | Arm | N | Hit@5 | Hit@10 | Delta vs raw | Help | Hurt | RI |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| bm25_tantivy_full | BarExamQA | raw | 1195 | 0.6% | 0.8% | 0.0% | 0 | 0 | 0.000 |
| bm25_tantivy_full | BarExamQA | hyde | 1195 | 7.7% | 11.8% | 7.1% | 91 | 6 | 0.071 |
| bm25_tantivy_full | BarExamQA | scope | 1195 | 7.9% | 12.4% | 7.4% | 93 | 5 | 0.074 |
| bm25_tantivy_full | FiQA | raw | 648 | 38.7% | 48.9% | 0.0% | 0 | 0 | 0.000 |
| bm25_tantivy_full | FiQA | hyde | 648 | 19.6% | 25.8% | -19.1% | 48 | 172 | -0.191 |
| bm25_tantivy_full | FiQA | scope | 648 | 19.4% | 26.9% | -19.3% | 49 | 174 | -0.193 |
| bm25_tantivy_full | NFCorpus | raw | 323 | 62.5% | 68.1% | 0.0% | 0 | 0 | 0.000 |
| bm25_tantivy_full | NFCorpus | hyde | 323 | 32.5% | 40.2% | -30.0% | 11 | 108 | -0.300 |
| bm25_tantivy_full | NFCorpus | scope | 323 | 58.2% | 67.8% | -4.3% | 34 | 48 | -0.043 |
| bm25_tantivy_full | SciDocs | raw | 1000 | 38.7% | 48.1% | 0.0% | 0 | 0 | 0.000 |
| bm25_tantivy_full | SciDocs | hyde | 1000 | 22.7% | 29.9% | -16.0% | 40 | 200 | -0.160 |
| bm25_tantivy_full | SciDocs | scope | 1000 | 34.9% | 46.2% | -3.8% | 95 | 133 | -0.038 |
| bm25_tantivy_full | SciFact | raw | 300 | 74.0% | 80.0% | 0.0% | 0 | 0 | 0.000 |
| bm25_tantivy_full | SciFact | hyde | 300 | 43.3% | 51.0% | -30.7% | 7 | 99 | -0.307 |
| bm25_tantivy_full | SciFact | scope | 300 | 71.7% | 79.7% | -2.3% | 22 | 29 | -0.023 |
| bm25_tantivy_full | TREC-COVID | raw | 50 | 96.0% | 98.0% | 0.0% | 0 | 0 | 0.000 |
| bm25_tantivy_full | TREC-COVID | hyde | 50 | 66.0% | 78.0% | -30.0% | 2 | 17 | -0.300 |
| bm25_tantivy_full | TREC-COVID | scope | 50 | 88.0% | 92.0% | -8.0% | 1 | 5 | -0.080 |
| bm25_tantivy_full | HousingQA state-filtered | raw | 6853 | 29.0% | 37.3% | 0.0% | 0 | 0 | 0.000 |
| bm25_tantivy_full | HousingQA state-filtered | hyde | 6853 | 48.1% | 57.6% | 19.1% | 1692 | 380 | 0.191 |
| bm25_tantivy_full | HousingQA state-filtered | scope | 6853 | 41.0% | 51.1% | 12.0% | 1396 | 572 | 0.120 |
| gte_ce_original | BarExamQA | raw | 1195 | 1.4% | 2.2% | 0.0% | 0 | 0 | 0.000 |
| gte_ce_original | BarExamQA | hyde | 1195 | 11.4% | 19.1% | 10.0% | 130 | 11 | 0.100 |
| gte_ce_original | BarExamQA | scope | 1195 | 12.1% | 18.7% | 10.6% | 138 | 11 | 0.106 |
| gte_ce_original | FiQA | raw | 648 | 66.2% | 77.0% | 0.0% | 0 | 0 | 0.000 |
| gte_ce_original | FiQA | hyde | 648 | 32.3% | 42.4% | -34.0% | 38 | 258 | -0.340 |
| gte_ce_original | FiQA | scope | 648 | 35.2% | 47.7% | -31.0% | 25 | 226 | -0.310 |
| gte_ce_original | NFCorpus | raw | 323 | 69.3% | 74.3% | 0.0% | 0 | 0 | 0.000 |
| gte_ce_original | NFCorpus | hyde | 323 | 33.4% | 44.6% | -35.9% | 6 | 122 | -0.359 |
| gte_ce_original | NFCorpus | scope | 323 | 65.0% | 74.6% | -4.3% | 20 | 34 | -0.043 |
| gte_ce_original | SciDocs | raw | 1000 | 49.0% | 64.2% | 0.0% | 0 | 0 | 0.000 |
| gte_ce_original | SciDocs | hyde | 1000 | 25.5% | 39.3% | -23.5% | 58 | 293 | -0.235 |
| gte_ce_original | SciDocs | scope | 1000 | 47.1% | 60.4% | -1.9% | 87 | 106 | -0.019 |
| gte_ce_original | SciFact | raw | 300 | 82.0% | 89.0% | 0.0% | 0 | 0 | 0.000 |
| gte_ce_original | SciFact | hyde | 300 | 35.0% | 48.3% | -47.0% | 12 | 153 | -0.470 |
| gte_ce_original | SciFact | scope | 300 | 65.7% | 77.3% | -16.3% | 12 | 61 | -0.163 |
| gte_ce_original | TREC-COVID | raw | 50 | 98.0% | 100.0% | 0.0% | 0 | 0 | 0.000 |
| gte_ce_original | TREC-COVID | hyde | 50 | 70.0% | 74.0% | -28.0% | 1 | 15 | -0.280 |
| gte_ce_original | TREC-COVID | scope | 50 | 96.0% | 98.0% | -2.0% | 1 | 2 | -0.020 |
| gte_ce_original | HousingQA state-filtered | raw | 6853 | 36.9% | 48.1% | 0.0% | 0 | 0 | 0.000 |
| gte_ce_original | HousingQA state-filtered | hyde | 6853 | 30.6% | 40.1% | -6.3% | 864 | 1297 | -0.063 |
| gte_ce_original | HousingQA state-filtered | scope | 6853 | 38.1% | 47.0% | 1.1% | 1023 | 946 | 0.011 |

## Gold-Affinity Delta Correlations

| Retriever | Dataset | Arm | N | Spearman | Kendall | Pearson | Mean gold-affinity delta |
|---|---|---|---:|---:|---:|---:|---:|
| bm25_tantivy_full | BarExamQA | hyde | 1195 | 0.322 | 0.263 | 0.366 | 13.000 |
| bm25_tantivy_full | BarExamQA | scope | 1195 | 0.305 | 0.249 | 0.354 | 8.840 |
| bm25_tantivy_full | FiQA | hyde | 648 | 0.363 | 0.294 | 0.423 | 11.301 |
| bm25_tantivy_full | FiQA | scope | 648 | 0.394 | 0.319 | 0.439 | 10.071 |
| bm25_tantivy_full | NFCorpus | hyde | 323 | 0.403 | 0.328 | 0.414 | 10.891 |
| bm25_tantivy_full | NFCorpus | scope | 323 | 0.380 | 0.301 | 0.345 | 17.995 |
| bm25_tantivy_full | SciDocs | hyde | 1000 | 0.372 | 0.303 | 0.408 | 9.018 |
| bm25_tantivy_full | SciDocs | scope | 1000 | 0.349 | 0.281 | 0.354 | 14.828 |
| bm25_tantivy_full | SciFact | hyde | 300 | 0.598 | 0.491 | 0.591 | 4.367 |
| bm25_tantivy_full | SciFact | scope | 300 | 0.416 | 0.338 | 0.382 | 16.096 |
| bm25_tantivy_full | TREC-COVID | hyde | 50 | 0.429 | 0.347 | 0.401 | 18.809 |
| bm25_tantivy_full | TREC-COVID | scope | 50 | 0.195 | 0.159 | 0.165 | 27.132 |
| bm25_tantivy_full | HousingQA state-filtered | hyde | 6853 | 0.409 | 0.327 | 0.388 | 20.629 |
| bm25_tantivy_full | HousingQA state-filtered | scope | 6853 | 0.440 | 0.354 | 0.432 | 10.996 |
| gte_ce_original | BarExamQA | scope | 1195 | 0.354 | 0.287 | 0.342 | 3.885 |
| gte_ce_original | FiQA | hyde | 648 | 0.565 | 0.457 | 0.564 | -4.055 |
| gte_ce_original | FiQA | scope | 648 | 0.505 | 0.411 | 0.512 | -2.947 |
| gte_ce_original | NFCorpus | hyde | 323 | 0.406 | 0.331 | 0.413 | -5.005 |
| gte_ce_original | NFCorpus | scope | 323 | 0.296 | 0.241 | 0.305 | -0.919 |
| gte_ce_original | SciDocs | hyde | 1000 | 0.476 | 0.382 | 0.468 | -3.269 |
| gte_ce_original | SciDocs | scope | 1000 | 0.299 | 0.240 | 0.311 | 1.302 |
| gte_ce_original | SciFact | hyde | 300 | 0.475 | 0.388 | 0.494 | -7.360 |
| gte_ce_original | SciFact | scope | 300 | 0.329 | 0.270 | 0.409 | -0.909 |
| gte_ce_original | TREC-COVID | hyde | 50 | 0.313 | 0.255 | 0.312 | -7.662 |
| gte_ce_original | TREC-COVID | scope | 50 | 0.108 | 0.088 | 0.137 | -1.824 |
| gte_ce_original | HousingQA state-filtered | scope | 6853 | 0.504 | 0.410 | 0.530 | 2.990 |

## Run Notes

- Loaded original gte+CE retrieval caches for all requested datasets.
- BarExamQA BM25: built tantivy index docs=856835; elapsed_sec=169.6; retrieval_elapsed_sec=22.3; max_query_terms=64
- HousingQA state-filtered BM25: built tantivy index docs=1837403; elapsed_sec=473.3; retrieval_elapsed_sec=53.1; max_query_terms=64
- SciFact BM25: built tantivy index docs=5183; elapsed_sec=2.0; retrieval_elapsed_sec=0.4; max_query_terms=64
- NFCorpus BM25: built tantivy index docs=3633; elapsed_sec=2.0; retrieval_elapsed_sec=3.1; max_query_terms=64
- FiQA BM25: built tantivy index docs=57638; elapsed_sec=3.9; retrieval_elapsed_sec=1.8; max_query_terms=64
- TREC-COVID BM25: built tantivy index docs=171332; elapsed_sec=14.1; retrieval_elapsed_sec=5.6; max_query_terms=64
- SciDocs BM25: built tantivy index docs=25657; elapsed_sec=2.4; retrieval_elapsed_sec=2.9; max_query_terms=64
- E5/BGE availability: `intfloat/e5-large-v2` is cached locally with dim=1024; full embedding/retrieval stage remains pending.
- Row-level points: `docs/generated/credibility_C_three_retrievers_full_2026-05-29_points.jsonl`
- BM25 uses Tantivy disk-backed full-corpus indexes for each dataset. Housing applies the question state as a metadata filter.
- BM25 query term cap: `64` unique non-stopword terms by within-query frequency; this avoids pathological FTS query length while preserving full-corpus search.

