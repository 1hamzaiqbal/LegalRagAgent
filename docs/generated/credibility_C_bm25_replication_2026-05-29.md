# Credibility Battery Phase C - BM25 Replication

BM25 replication under a non-dense retriever. Generator remains `or-gemma4-26b`; only retrieval is changed. No `paper/` files were edited.

## Verdict

- BM25 mechanism verdict: **travels**. Mean per-dataset SCOPE Spearman between BM25 gold-affinity delta and BM25 retrieval gain is `0.342`.
- Kill criterion: <=0.2 means the mechanism is likely gte/CE-specific; >=0.3 means it travels to BM25.

## Retrieval Summary

| Dataset | Arm | N | Hit@5 | Hit@10 | Delta vs raw | Help | Hurt | RI |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FiQA | raw | 648 | 38.1% | 47.7% | 0.0% | 0 | 0 | 0.000 |
| FiQA | hyde | 648 | 20.2% | 26.2% | -17.9% | 50 | 166 | -0.179 |
| FiQA | scope | 648 | 21.6% | 28.7% | -16.5% | 48 | 155 | -0.165 |
| NFCorpus | raw | 323 | 62.2% | 67.2% | 0.0% | 0 | 0 | 0.000 |
| NFCorpus | hyde | 323 | 32.8% | 40.6% | -29.4% | 9 | 104 | -0.294 |
| NFCorpus | scope | 323 | 61.9% | 69.3% | -0.3% | 31 | 32 | -0.003 |
| SciFact | raw | 300 | 73.7% | 80.0% | 0.0% | 0 | 0 | 0.000 |
| SciFact | hyde | 300 | 43.3% | 51.3% | -30.3% | 8 | 99 | -0.303 |
| SciFact | scope | 300 | 75.7% | 83.3% | 2.0% | 30 | 24 | 0.020 |

## BM25 Gold-Affinity Delta Correlations

| Dataset | Arm | N | Spearman | Kendall | Pearson | Mean BM25 gold delta |
|---|---|---:|---:|---:|---:|---:|
| FiQA | hyde | 648 | 0.331 | 0.267 | 0.403 | 24.053 |
| FiQA | scope | 648 | 0.368 | 0.298 | 0.415 | 20.761 |
| NFCorpus | hyde | 323 | 0.413 | 0.338 | 0.440 | 15.086 |
| NFCorpus | scope | 323 | 0.354 | 0.283 | 0.328 | 29.324 |
| SciFact | hyde | 300 | 0.603 | 0.495 | 0.576 | 6.960 |
| SciFact | scope | 300 | 0.304 | 0.243 | 0.268 | 32.982 |

## Run Notes

- SciFact: docs=5183; max_docs=full; elapsed_sec=9.7
- NFCorpus: docs=3633; max_docs=full; elapsed_sec=9.3
- FiQA: docs=57638; max_docs=full; elapsed_sec=179.8
- BarExamQA feasibility: the corpus-wide document-frequency pass completed over 856,835 `legal_passages` documents, but exact full-corpus BM25 retrieval was deferred after the second pass processed only 10,000 documents in about 90 seconds. This is recorded as a feasibility blocker for the current pure-Python scorer, not as a negative BM25 result.
- HousingQA state-filtered was not attempted for Phase C retrieval after the BarExamQA full-corpus BM25 pass proved too slow; the task allowed Housing to be deferred if indexing was a blocker.
- Row-level BM25 points: `docs/generated/credibility_C_bm25_replication_2026-05-29_points.jsonl`
