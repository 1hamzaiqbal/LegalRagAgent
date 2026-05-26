# BEIR Phase 1 Verification - 2026-05-26

## Scope

Read-only Phase 4/5 analysis over the committed BEIR Phase 1 caches. No answer/model calls were made in this phase, and no files under `paper/` were edited.

Definitions:
- Retrieval gain is `expansion Hit@5 - raw-question Hit@5` per query.
- Collins-Thompson `RI = (n_help - n_hurt) / N`, where help is expansion-only Hit@5 and hurt is raw-only Hit@5.
- Gold CE affinity is the max `cross-encoder/ms-marco-MiniLM-L-6-v2` score over all positive qrel document ids.
- `M_raw = CE(raw,best gold) - max CE(raw,retrieved non-gold)`; multi-gold rows exclude all gold ids from the distractor max.
- OOV and log-perplexity use an add-1 smoothed unigram LM built from each BEIR corpus CSV.

## Cross-Dataset Verdicts

| Claim | Verdict | Key numbers |
|---|---|---|
| Gold-affinity mechanism (HyDE) | **supported** | pooled rho=0.501, tau=0.404; positive datasets=5/5 |
| P4 geometry-not-hallucination (HyDE) | **supported** | pooled deltaM<0 AUC geometry=0.944, OOV/logPPL=0.520 |
| Raw-margin regime/crossover (HyDE) | **supported** | sign-crossover datasets=4/5; declining low-to-high datasets=5/5 |
| Gold-affinity mechanism (SCOPE) | **supported** | pooled rho=0.426, tau=0.346; positive datasets=5/5 |
| P4 geometry-not-hallucination (SCOPE) | **supported** | pooled deltaM<0 AUC geometry=0.909, OOV/logPPL=0.509 |
| Raw-margin regime/crossover (SCOPE) | **supported** | sign-crossover datasets=4/5; declining low-to-high datasets=4/5 |

## Retrieval Outcomes

| Dataset | Expansion | N | Raw Hit@5 | Expansion Hit@5 | Net Hit@5 | Raw Hit@10 | Expansion Hit@10 | Help | Hurt | RI | Margin-valid rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | HyDE | 300 | 82.0% | 35.0% | -47.0% | 89.0% | 48.3% | 12 | 153 | -0.470 | 300 |
| SciFact | SCOPE | 300 | 82.0% | 65.7% | -16.3% | 89.0% | 77.3% | 12 | 61 | -0.163 | 300 |
| NFCorpus | HyDE | 323 | 69.3% | 33.4% | -35.9% | 74.3% | 44.6% | 6 | 122 | -0.359 | 310 |
| NFCorpus | SCOPE | 323 | 69.3% | 65.0% | -4.3% | 74.3% | 74.6% | 20 | 34 | -0.043 | 307 |
| FiQA | HyDE | 648 | 66.2% | 32.3% | -34.0% | 77.0% | 42.4% | 38 | 258 | -0.340 | 648 |
| FiQA | SCOPE | 648 | 66.2% | 35.2% | -31.0% | 77.0% | 47.7% | 25 | 226 | -0.310 | 648 |
| TREC-COVID | HyDE | 50 | 98.0% | 70.0% | -28.0% | 100.0% | 74.0% | 1 | 15 | -0.280 | 27 |
| TREC-COVID | SCOPE | 50 | 98.0% | 96.0% | -2.0% | 100.0% | 98.0% | 1 | 2 | -0.020 | 27 |
| SciDocs | HyDE | 1000 | 49.0% | 25.5% | -23.5% | 64.2% | 39.3% | 58 | 293 | -0.235 | 1000 |
| SciDocs | SCOPE | 1000 | 49.0% | 47.1% | -1.9% | 64.2% | 60.4% | 87 | 106 | -0.019 | 1000 |
| Pooled | HyDE | 2321 | 62.0% | 30.7% | -31.3% | 73.2% | 42.8% | 115 | 841 | -0.313 | 2285 |
| Pooled | SCOPE | 2321 | 62.0% | 49.7% | -12.2% | 73.2% | 61.8% | 145 | 429 | -0.122 | 2282 |

## Gold-Affinity and Margin Correlations

The primary mechanism column is gold CE delta: `CE(exp,best gold) - CE(raw,best gold)`. Delta-margin is included to show whether adding the non-gold distractor term changes the read.

| Dataset | Expansion | N | Mean CE gold delta | Gold rho | Gold tau | Margin-valid N | Mean deltaM | DeltaM rho | DeltaM tau |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | HyDE | 300 | -7.360 | 0.475 | 0.388 | 300 | -2.879 | 0.455 | 0.373 |
| SciFact | SCOPE | 300 | -0.909 | 0.329 | 0.270 | 300 | -2.408 | 0.411 | 0.337 |
| NFCorpus | HyDE | 323 | -5.005 | 0.406 | 0.331 | 310 | -2.313 | 0.419 | 0.341 |
| NFCorpus | SCOPE | 323 | -0.919 | 0.296 | 0.241 | 307 | -0.983 | 0.198 | 0.159 |
| FiQA | HyDE | 648 | -4.055 | 0.565 | 0.457 | 648 | -1.352 | 0.543 | 0.441 |
| FiQA | SCOPE | 648 | -2.947 | 0.505 | 0.411 | 648 | -1.798 | 0.537 | 0.440 |
| TREC-COVID | HyDE | 50 | -7.662 | 0.313 | 0.255 | 27 | -0.663 | 0.336 | 0.284 |
| TREC-COVID | SCOPE | 50 | -1.824 | 0.108 | 0.088 | 27 | -0.066 | 0.516 | 0.434 |
| SciDocs | HyDE | 1000 | -3.269 | 0.476 | 0.382 | 1000 | 1.208 | 0.501 | 0.402 |
| SciDocs | SCOPE | 1000 | 1.302 | 0.299 | 0.240 | 1000 | 1.179 | 0.349 | 0.281 |
| Pooled | HyDE | 2321 | -4.354 | 0.501 | 0.404 | 2285 | -0.555 | 0.517 | 0.418 |
| Pooled | SCOPE | 2321 | -0.546 | 0.426 | 0.346 | 2282 | -0.444 | 0.442 | 0.358 |
| Pooled | All expansions | 4642 | -2.450 | 0.497 | 0.402 | 4567 | -0.499 | 0.482 | 0.390 |

## P4 Failure Model

Two targets are reported: geometry failure `deltaM<0` and observed retrieval hurt. The requested comparison is OOV/log-perplexity versus geometry features `{M_raw, CE(exp,gold)}`.

| Dataset | Expansion | Target | N | Failures | AUC OOV/logPPL | AUC geometry | Pseudo-R2 OOV/logPPL | Pseudo-R2 geometry |
|---|---|---|---:|---:|---:|---:|---:|---:|
| SciFact | HyDE | deltaM<0 | 300 | 225 | 0.570 | 0.964 | 0.016 | 0.625 |
| SciFact | HyDE | retrieval hurt | 300 | 153 | 0.532 | 0.839 | 0.003 | 0.271 |
| SciFact | SCOPE | deltaM<0 | 300 | 213 | 0.572 | 0.889 | 0.024 | 0.391 |
| SciFact | SCOPE | retrieval hurt | 300 | 61 | 0.538 | 0.819 | 0.004 | 0.209 |
| NFCorpus | HyDE | deltaM<0 | 310 | 202 | 0.574 | 0.956 | 0.009 | 0.602 |
| NFCorpus | HyDE | retrieval hurt | 311 | 121 | 0.566 | 0.762 | 0.019 | 0.156 |
| NFCorpus | SCOPE | deltaM<0 | 307 | 188 | 0.548 | 0.881 | 0.008 | 0.378 |
| NFCorpus | SCOPE | retrieval hurt | 311 | 34 | 0.536 | 0.674 | 0.003 | 0.047 |
| FiQA | HyDE | deltaM<0 | 648 | 412 | 0.544 | 0.936 | 0.006 | 0.534 |
| FiQA | HyDE | retrieval hurt | 648 | 258 | 0.548 | 0.815 | 0.004 | 0.241 |
| FiQA | SCOPE | deltaM<0 | 648 | 423 | 0.527 | 0.923 | 0.003 | 0.491 |
| FiQA | SCOPE | retrieval hurt | 648 | 226 | 0.563 | 0.751 | 0.005 | 0.135 |
| TREC-COVID | HyDE | deltaM<0 | 27 | 17 | 0.488 | 0.753 | 0.001 | 0.148 |
| TREC-COVID | HyDE | retrieval hurt | 28 | 7 | 0.638 | 0.741 | 0.044 | 0.099 |
| TREC-COVID | SCOPE | deltaM<0 | 27 | 13 | 0.665 | 0.835 | 0.069 | 0.261 |
| TREC-COVID | SCOPE | retrieval hurt | 28 | 2 | 0.635 | 0.865 | 0.007 | 0.150 |
| SciDocs | HyDE | deltaM<0 | 1000 | 400 | 0.508 | 0.949 | 0.000 | 0.588 |
| SciDocs | HyDE | retrieval hurt | 1000 | 293 | 0.536 | 0.816 | 0.002 | 0.230 |
| SciDocs | SCOPE | deltaM<0 | 1000 | 369 | 0.507 | 0.909 | 0.001 | 0.443 |
| SciDocs | SCOPE | retrieval hurt | 1000 | 106 | 0.503 | 0.701 | 0.000 | 0.077 |
| Pooled | HyDE | deltaM<0 | 2285 | 1256 | 0.520 | 0.944 | 0.000 | 0.571 |
| Pooled | HyDE | retrieval hurt | 2287 | 832 | 0.490 | 0.798 | 0.001 | 0.206 |
| Pooled | SCOPE | deltaM<0 | 2282 | 1206 | 0.509 | 0.909 | 0.000 | 0.450 |
| Pooled | SCOPE | retrieval hurt | 2287 | 429 | 0.598 | 0.743 | 0.016 | 0.119 |

## M_raw Quintile Regime Test

Rows with no non-gold distractor in that condition's top-10 are excluded from margin bins because `M_raw` is undefined. This is common in TREC-COVID because each query has hundreds of positive qrel documents.

| Dataset | Expansion | Bin | N | M_raw median | M_raw range | Raw Hit@5 | Expansion Hit@5 | Net Hit@5 | Help | Hurt | RI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | HyDE | 1 | 60 | -5.988 | [-14.436, -2.832] | 20.0% | 25.0% | 5.0% | 11 | 8 | 0.050 |
| SciFact | HyDE | 2 | 60 | -1.596 | [-2.831, -0.061] | 90.0% | 25.0% | -65.0% | 1 | 40 | -0.650 |
| SciFact | HyDE | 3 | 60 | 1.170 | [0.091, 2.263] | 100.0% | 35.0% | -65.0% | 0 | 39 | -0.650 |
| SciFact | HyDE | 4 | 60 | 3.610 | [2.398, 5.375] | 100.0% | 50.0% | -50.0% | 0 | 30 | -0.500 |
| SciFact | HyDE | 5 | 60 | 7.265 | [5.398, 13.701] | 100.0% | 40.0% | -60.0% | 0 | 36 | -0.600 |
| SciFact | SCOPE | 1 | 60 | -5.988 | [-14.436, -2.832] | 20.0% | 23.3% | 3.3% | 9 | 7 | 0.033 |
| SciFact | SCOPE | 2 | 60 | -1.596 | [-2.831, -0.061] | 90.0% | 56.7% | -33.3% | 3 | 23 | -0.333 |
| SciFact | SCOPE | 3 | 60 | 1.170 | [0.091, 2.263] | 100.0% | 78.3% | -21.7% | 0 | 13 | -0.217 |
| SciFact | SCOPE | 4 | 60 | 3.610 | [2.398, 5.375] | 100.0% | 86.7% | -13.3% | 0 | 8 | -0.133 |
| SciFact | SCOPE | 5 | 60 | 7.265 | [5.398, 13.701] | 100.0% | 83.3% | -16.7% | 0 | 10 | -0.167 |
| NFCorpus | HyDE | 1 | 62 | -6.294 | [-13.968, -3.713] | 16.1% | 9.7% | -6.5% | 2 | 6 | -0.065 |
| NFCorpus | HyDE | 2 | 62 | -1.968 | [-3.631, -0.957] | 51.6% | 11.3% | -40.3% | 2 | 27 | -0.403 |
| NFCorpus | HyDE | 3 | 63 | 0.009 | [-0.918, 1.166] | 81.0% | 39.7% | -41.3% | 1 | 27 | -0.413 |
| NFCorpus | HyDE | 4 | 62 | 2.715 | [1.169, 4.863] | 93.5% | 51.6% | -41.9% | 0 | 26 | -0.419 |
| NFCorpus | HyDE | 5 | 62 | 7.847 | [4.900, 16.872] | 98.4% | 43.5% | -54.8% | 1 | 35 | -0.548 |
| NFCorpus | SCOPE | 1 | 62 | -6.294 | [-13.968, -3.713] | 16.1% | 12.9% | -3.2% | 4 | 6 | -0.032 |
| NFCorpus | SCOPE | 2 | 62 | -1.968 | [-3.631, -0.957] | 51.6% | 48.4% | -3.2% | 8 | 10 | -0.032 |
| NFCorpus | SCOPE | 3 | 63 | 0.009 | [-0.918, 1.166] | 81.0% | 76.2% | -4.8% | 5 | 8 | -0.048 |
| NFCorpus | SCOPE | 4 | 62 | 2.715 | [1.169, 4.863] | 93.5% | 83.9% | -9.7% | 2 | 8 | -0.097 |
| NFCorpus | SCOPE | 5 | 62 | 7.847 | [4.900, 16.872] | 98.4% | 96.8% | -1.6% | 1 | 2 | -0.016 |
| FiQA | HyDE | 1 | 130 | -8.151 | [-16.212, -5.105] | 6.9% | 13.1% | 6.2% | 15 | 7 | 0.062 |
| FiQA | HyDE | 2 | 129 | -3.430 | [-5.018, -2.055] | 34.9% | 23.3% | -11.6% | 18 | 33 | -0.116 |
| FiQA | HyDE | 3 | 130 | -0.884 | [-2.043, -0.080] | 90.0% | 33.8% | -56.2% | 4 | 77 | -0.562 |
| FiQA | HyDE | 4 | 129 | 1.160 | [-0.061, 2.616] | 100.0% | 42.6% | -57.4% | 0 | 74 | -0.574 |
| FiQA | HyDE | 5 | 130 | 4.287 | [2.635, 14.575] | 99.2% | 48.5% | -50.8% | 1 | 67 | -0.508 |
| FiQA | SCOPE | 1 | 130 | -8.151 | [-16.212, -5.105] | 6.9% | 10.8% | 3.8% | 12 | 7 | 0.038 |
| FiQA | SCOPE | 2 | 129 | -3.430 | [-5.018, -2.055] | 34.9% | 19.4% | -15.5% | 10 | 30 | -0.155 |
| FiQA | SCOPE | 3 | 130 | -0.884 | [-2.043, -0.080] | 90.0% | 37.7% | -52.3% | 2 | 70 | -0.523 |
| FiQA | SCOPE | 4 | 129 | 1.160 | [-0.061, 2.616] | 100.0% | 52.7% | -47.3% | 0 | 61 | -0.473 |
| FiQA | SCOPE | 5 | 130 | 4.287 | [2.635, 14.575] | 99.2% | 55.4% | -43.8% | 1 | 58 | -0.438 |
| TREC-COVID | HyDE | 1 | 6 | -0.145 | [-0.881, -0.012] | 83.3% | 100.0% | 16.7% | 1 | 0 | 0.167 |
| TREC-COVID | HyDE | 2 | 5 | 0.294 | [-0.003, 0.565] | 100.0% | 80.0% | -20.0% | 0 | 1 | -0.200 |
| TREC-COVID | HyDE | 3 | 6 | 0.843 | [0.731, 1.251] | 100.0% | 66.7% | -33.3% | 0 | 2 | -0.333 |
| TREC-COVID | HyDE | 4 | 5 | 1.347 | [1.253, 1.479] | 100.0% | 80.0% | -20.0% | 0 | 1 | -0.200 |
| TREC-COVID | HyDE | 5 | 6 | 2.133 | [1.648, 3.942] | 100.0% | 50.0% | -50.0% | 0 | 3 | -0.500 |
| TREC-COVID | SCOPE | 1 | 6 | -0.145 | [-0.881, -0.012] | 83.3% | 100.0% | 16.7% | 1 | 0 | 0.167 |
| TREC-COVID | SCOPE | 2 | 5 | 0.294 | [-0.003, 0.565] | 100.0% | 80.0% | -20.0% | 0 | 1 | -0.200 |
| TREC-COVID | SCOPE | 3 | 6 | 0.843 | [0.731, 1.251] | 100.0% | 100.0% | 0.0% | 0 | 0 | 0.000 |
| TREC-COVID | SCOPE | 4 | 5 | 1.347 | [1.253, 1.479] | 100.0% | 80.0% | -20.0% | 0 | 1 | -0.200 |
| TREC-COVID | SCOPE | 5 | 6 | 2.133 | [1.648, 3.942] | 100.0% | 100.0% | 0.0% | 0 | 0 | 0.000 |
| SciDocs | HyDE | 1 | 200 | -9.897 | [-19.018, -7.498] | 2.5% | 5.0% | 2.5% | 10 | 5 | 0.025 |
| SciDocs | HyDE | 2 | 200 | -5.688 | [-7.491, -4.233] | 15.0% | 14.5% | -0.5% | 22 | 23 | -0.005 |
| SciDocs | HyDE | 3 | 200 | -2.923 | [-4.217, -1.867] | 41.5% | 23.5% | -18.0% | 21 | 57 | -0.180 |
| SciDocs | HyDE | 4 | 200 | -0.918 | [-1.846, 0.383] | 87.0% | 33.5% | -53.5% | 5 | 112 | -0.535 |
| SciDocs | HyDE | 5 | 200 | 2.160 | [0.405, 13.012] | 99.0% | 51.0% | -48.0% | 0 | 96 | -0.480 |
| SciDocs | SCOPE | 1 | 200 | -9.897 | [-19.018, -7.498] | 2.5% | 7.0% | 4.5% | 12 | 3 | 0.045 |
| SciDocs | SCOPE | 2 | 200 | -5.688 | [-7.491, -4.233] | 15.0% | 22.0% | 7.0% | 25 | 11 | 0.070 |
| SciDocs | SCOPE | 3 | 200 | -2.923 | [-4.217, -1.867] | 41.5% | 46.5% | 5.0% | 38 | 28 | 0.050 |
| SciDocs | SCOPE | 4 | 200 | -0.918 | [-1.846, 0.383] | 87.0% | 69.5% | -17.5% | 12 | 47 | -0.175 |
| SciDocs | SCOPE | 5 | 200 | 2.160 | [0.405, 13.012] | 99.0% | 90.5% | -8.5% | 0 | 17 | -0.085 |

## Reading

- Raw-question retrieval is a very strong baseline on this BEIR slice. HyDE loses 31.3% Hit@5 pooled, while SCOPE loses 12.2% pooled and is much closer to raw on NFCorpus, TREC-COVID, and SciDocs.
- Gold-affinity movement does replicate as a row-level mechanism: pooled gold-delta correlation is 0.501 for HyDE and 0.426 for SCOPE, with positive correlations in all five datasets.
- That mechanism does not make ungated expansion a good policy here. Mean CE gold deltas are often negative, and the average retrieval outcome is below raw-question retrieval on every dataset/method cell.
- The clearest replicated lesson is risk control: expansion often helps individual low-confidence rows, but broad ungated application hurts when raw retrieval already lands a gold document.
- TREC-COVID is a special case for margin tests: qrels are extremely dense, so many top-10 lists have no non-gold distractor and margin-valid N is much smaller than query N.

## Sources

- `caches/retrieval/full/beir_scifact_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_scifact_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_nfcorpus_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_fiqa_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_trec_covid_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_raw_question_k10.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde.jsonl`
- `caches/generation/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_rag_hyde_k10.jsonl`
- `caches/retrieval/full/beir_scidocs_qfull_seed42_or-gemma4-26b_snap_hyre_k10.jsonl`

## Reproduction

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CROSS_ENCODER_DEVICE=cuda \
uv run python scripts/analyze_beir_phase1.py \
  --output docs/generated/beir_phase1_verification_2026-05-26.md
```
